"""Claude as the LLM: the reasoning engine and the memory bank run in this sandbox; Claude reads what they
recall and proposes the missing pieces, and every proposal is checked against the held-out measurement.

WHY. Notebook section 16 (colab/cell_reasoning_llm_a100.ipynb) has a 30B open model read the engine's recalled
memories and propose which pathways rise and fall after a knockdown (16d). The user asked to run it here, with
Claude (this session) as the LLM. Claude cannot be fine-tuned or write an 8,248-gene vector, so this runs the
PROPOSAL LOOP only; gates M1-M5 (fine-tuned predictions) stay with the Colab run.

WHAT RUNS. Data, split, features and pathway sets are executed from the notebook's own cells (sections 0-5, 7,
9's model code, 11's pathway sets), so everything matches it. Then, as notebook 16a-16d:
  memory bank  every TRAIN knockdown as a measured fact: top 3 pathways up / down, top 4 genes up / down;
               pathways about viruses or disease are left out. TEST genes are never stored.
  engine       the section 9 engine retrained once (seed 0, 150 epochs, all TRAIN in memory, own item masked)
               on the non-LLM features (DepMap, interactions, HPA location / class / K-562 RNA): there is no 30B
               model here to make context vectors. It predicts each TEST response, and its snapping step
               recalls the 3 most relevant TRAIN experiments; the memory adds up to 2 measured partners.
  Claude       for the notebook's 100 random TEST genes (seed 21) reads the gene's Atlas document, the recalled
               facts, the engine's 3 strongest rises / falls and a 20-item menu (engine top 8, bottom 8, then
               the recalled facts' lead pathways), and picks EXACTLY 4 UP and 4 DOWN with one sentence why.
               (The notebook allows up to 4; always picking 8 removes any gain from abstaining.)

BLINDING. `prepare` writes prompts and menus built from TRAIN data, the Atlas and the engine only, and prints or
saves nothing from a TEST measurement. Claude writes its picks to outputs/claudeloop/proposals.jsonl, and they
are COMMITTED before `score` runs. `score` reads the TEST responses for the first time, checks every pick, and
writes each one to the memory bank as a hypothesis (confirmed / refuted) with its sources.

GATES, PREDECLARED (as notebook 16d)
  CONFIRMED  a pick is confirmed if, in that gene's measured response, the pathway ranks in the top 10% (UP)
             or bottom 10% (DOWN) of the kept pathways. Precision = confirmed share of a gene's 8 picks.
             A missing or malformed answer is reported and its gene left out of every comparison.
  L1  Claude vs null: the same picks scored on another TEST gene's measurement (the previous gene in the draw).
      BETTER if one-sided sign-flip permutation p < 0.05 (5,000 flips, per-gene differences).
  L2  PRIMARY. Claude vs the engine's own picks from the same menu (its top 4 and bottom 4): per-gene precision,
      paired sign test: BETTER if p < 0.05 and more wins, WORSE if p < 0.05 and more losses, else NO DIFFERENCE.
  E1  descriptive: the engine's picks vs null, same test as L1 (is there anything to find?).
  D   descriptive: how many picks Claude changed from the engine's, and the confirmed share of the changed picks
      vs the engine picks they replaced.
NOT: Claude may have read the Replogle 2022 papers in training, so part of any gain could be recall of published
     results (as for the 30B model, whose nameless test was UNRESOLVED). One reader, one pass, no retries.
     The engine here lacks the LLM context vectors the notebook engine uses.

Run:  python3 -P rem/atlas/claudeloop.py prepare  ->  Claude writes proposals.jsonl, commit  ->  ... score
"""

from __future__ import annotations
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
NB = R("colab", "cell_reasoning_llm_a100.ipynb")
OUTD = R("outputs", "claudeloop")
CACHE = os.path.join(HERE, "_cache", "claudeloop")
OUT = os.path.join(HERE, "RESULTS_claudeloop.txt")
CACHED = {   # notebook file name -> where this repo caches it (anything missing is downloaded by the notebook code)
    "gwps.h5ad": "perturbseq/K562_gwps_normalized_bulk_01.h5ad",
    "ess.h5ad": "perturbseq_ess/K562_essential_normalized_bulk_01.h5ad",
    "proteinatlas.tsv.zip": "hpa/proteinatlas.tsv.zip", "rna_celline.tsv.zip": "hpa/rna_celline.tsv.zip",
    "subcellular_location.tsv.zip": "hpa/subcellular_location.tsv.zip",
    "interaction_consensus.tsv.zip": "hpa/interaction_consensus.tsv.zip",
    "CRISPRGeneEffect.csv": "depmap/CRISPRGeneEffect.csv", "ReactomePathways.gmt.zip": "reactome/ReactomePathways.gmt.zip",
}
N_PROPOSE, N_PICK, RULE = 100, 4, "=" * 97
BAD = re.compile(r"virus|viral|SARS|HIV|influenza|infect|disease|cancer|amyloid|parasit|bacteri|hepatitis|"
                 r"defective|mutant|syndrome|tumou?r|leishmania|tubercul|covid|_\d{5,}", re.I)
INSTRUCTION = ("Using the gene's biology and the recalled experiments, choose exactly 4 menu items that go UP and "
               "exactly 4 that go DOWN when the gene is knocked down. Reply with JSON only, like "
               "{\"up\": [1, 5, 7, 9], \"down\": [3, 11, 12, 20], \"why\": \"one sentence\"}.")


def notebook():
    """Execute the notebook's own data, feature, model and pathway code (no LLM; full training settings)."""
    data = os.path.join(CACHE, "data"); os.makedirs(data, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    for name, rel in CACHED.items():
        src, dst = os.path.join(HERE, "_cache", rel), os.path.join(data, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    os.environ.update(CELL_SMOKE="1", CELL_DATA=data, CELL_OUT=os.path.join(CACHE, "nb_out"))   # SMOKE: no installs, no LLM
    cells = {}
    for c in json.load(open(NB))["cells"]:
        t = "".join(c["source"]); m = re.match(r"# ---- (\d+)\.", t)
        if c["cell_type"] == "code" and m:
            cells[int(m.group(1))] = t
    G = {"__name__": "notebook"}
    run = lambda k, code: exec(compile(code, f"notebook section {k}", "exec"), G)
    run(0, cells[0]); G["EPOCHS"] = 150
    for k in (1, 2, 3, 4, 5, 7):
        run(k, cells[k])
    head = cells[9].split("\nfor name, kind, X in (")[0]; assert head != cells[9]; run(9, head)
    path = cells[11].split("\nrng = np.random.default_rng(3)")[0]; assert path != cells[11]; run(11, path)
    return G


def prepare():
    t0 = time.time()
    G = notebook()
    torch, np, nn, F = G["torch"], G["np"], G["nn"], G["F"]
    names, pw, gidx, A, DEV = G["names"], G["pw"], G["gidx"], G["A"], G["DEV"]
    KN, KE, rtr, rte, Y_tr, Ztr, Vy, X = G["KN"], G["KE"], G["rtr"], G["rte"], G["Y_tr"], G["Ztr"], G["Vy"], G["X_nollm"]
    PPI, prompt_for, pearson_rows = G["PPI"], G["prompt_for"], G["pearson_rows"]
    okp = torch.tensor([not BAD.search(nm) for nm in names], device=DEV) & (G["Mp"].sum(0) >= 15)
    pw_ok = lambda v: torch.where(okp, pw(v), torch.tensor(float("nan"), device=DEV))

    # 16a. the memory bank: every TRAIN knockdown as a measured fact
    def fact_of(v, own_gene):
        sc_ = torch.nan_to_num(pw_ok(v), nan=0.0); vv = torch.nan_to_num(v.clone())
        if own_gene in gidx:
            vv[gidx[own_gene]] = 0.0
        return dict(up=[names[i] for i in sc_.argsort(descending=True)[:3].tolist()],
                    down=[names[i] for i in sc_.argsort()[:3].tolist()],
                    genes_up=[A["genes"][i] for i in vv.argsort(descending=True)[:4].tolist()],
                    genes_down=[A["genes"][i] for i in vv.argsort()[:4].tolist()])

    MEM = [dict(id=f"M-{KN[rtr[j]]}", gene=KN[rtr[j]], type="measured",
                source="Replogle 2022 K562 genome-wide CRISPRi, TRAIN knockdown", **fact_of(Y_tr[j], KN[rtr[j]]))
           for j in range(len(rtr))]
    with open(os.path.join(OUTD, "memory_bank_measured.jsonl"), "w") as fh:
        for m_ in MEM:
            fh.write(json.dumps(m_) + "\n")
    fact_txt = lambda m_, why: (f"- {m_['gene']} knockdown ({why}): raised {'; '.join(m_['up'])}. Lowered {'; '.join(m_['down'])}. "
                                f"Most raised genes {', '.join(m_['genes_up'])}; most lowered {', '.join(m_['genes_down'])}.")
    print(f"memory bank: {len(MEM):,} measured TRAIN knockdowns; pathways kept {int(okp.sum()):,} of {len(names):,}")

    # 16b. the reasoning engine, exposing which memories its snapping step recalls
    class EngineR(G["Engine"]):
        def forward(self, x, mx, my, self_idx=None):
            h = self.enc(x); K = self.key(mx); V = self.val(my); self.tops = []
            for _ in range(self.steps):
                S = self.q(h) @ K.T / math.sqrt(K.shape[1])
                if self_idx is not None:
                    S = S.masked_fill(F.one_hot(self_idx, S.shape[1]).bool(), float("-inf"))
                top = S.topk(self.k, dim=1); self.tops.append(top.indices)
                w = torch.softmax(top.values, 1).unsqueeze(-1)
                r_y = (w * my[top.indices]).sum(1); r_h = (w * V[top.indices]).sum(1)
                h = torch.tanh(self.tree(torch.cat([x, r_h, h], 1)))
            g = torch.sigmoid(self.mix(h))
            return g * r_y + (1 - g) * self.out(h)

    torch.manual_seed(0)
    Xtr_, Xte_ = X[rtr.to(DEV)], X[rte.to(DEV)]
    eng = EngineR(X.shape[1], Ztr.shape[1]).to(DEV)
    opt_ = torch.optim.AdamW(eng.parameters(), lr=2e-3, weight_decay=1e-4)
    n_ = len(rtr); slot_all = torch.arange(n_, device=DEV)
    for ep in range(G["EPOCHS"]):
        eng.train()
        for b in torch.randperm(n_, device=DEV).split(256):
            loss = 1 - pearson_rows(eng(Xtr_[b], Xtr_, Ztr, self_idx=slot_all[b]) @ Vy.T, Y_tr[b]).mean()
            opt_.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(eng.parameters(), 1.0); opt_.step()
        if ep % 25 == 24:
            print(f"  engine epoch {ep + 1}: TRAIN loss {float(loss):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    eng.eval()
    with torch.no_grad():
        eng(Xtr_, Xtr_, Ztr, self_idx=slot_all); rec_tr = eng.tops[-1][:, :3].cpu().numpy()
        pred_te = eng(Xte_, Xtr_, Ztr) @ Vy.T; rec_te = eng.tops[-1][:, :3].cpu().numpy()
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, "engine_pred_test.npy"), pred_te.cpu().numpy().astype(np.float32))
    tr_of_ens = {KE[rtr[j]]: j for j in range(n_) if KE[rtr[j]]}
    norm_tr = Y_tr.norm(dim=1).cpu().numpy()

    def memory_block(e, recalled):
        lines = ["Measured experiments recalled from the memory bank (K562 CRISPRi knockdowns of OTHER genes):"]
        used = []
        for j in recalled:
            if j not in used:
                used.append(int(j)); lines.append(fact_txt(MEM[j], "recalled by the reasoning engine"))
        for j in sorted([tr_of_ens[q] for q in PPI.get(e, ()) if q in tr_of_ens], key=lambda j: -norm_tr[j])[:2]:
            if j not in used:
                used.append(int(j)); lines.append(fact_txt(MEM[j], "interaction partner"))
        return "\n".join(lines), used

    # 16d. the menus, drawn exactly as the notebook draws them
    pickp = np.random.default_rng(21).choice(len(rte), size=min(N_PROPOSE, len(rte)), replace=False)
    jobs, text = [], [f"INSTRUCTION (the same for every gene): {INSTRUCTION}\n"]
    for k, t in enumerate(pickp.tolist()):
        sym, e = KN[rte[t]], KE[rte[t]]
        blk, used = memory_block(e, rec_te[t])
        sc_ = torch.nan_to_num(pw_ok(pred_te[t]), nan=0.0)
        cand = sc_.argsort(descending=True)[:8].tolist() + sc_.argsort()[:8].tolist()
        for j in used:
            for nm in MEM[j]["up"][:1] + MEM[j]["down"][:1]:
                ix = names.index(nm)
                if ix not in cand:
                    cand.append(ix)
        cand = cand[:20]
        eng_up = sorted(cand, key=lambda i: -float(sc_[i]))[:N_PICK]; eng_dn = sorted(cand, key=lambda i: float(sc_[i]))[:N_PICK]
        doc = prompt_for(sym, e).rsplit("Question:", 1)[0]
        text.append(f"### {k + 1}. {sym}\n{doc}{blk}\n"
                    f"The reasoning engine predicts the strongest rises in: {'; '.join(names[i] for i in sc_.argsort(descending=True)[:3].tolist())}; "
                    f"and the strongest falls in: {'; '.join(names[i] for i in sc_.argsort()[:3].tolist())}.\nMenu of pathways:\n"
                    + "\n".join(f"{q + 1}. {names[ix]}" for q, ix in enumerate(cand)) + "\n")
        jobs.append(dict(n=k + 1, t=int(t), gene=sym, ens=e, cand=cand, menu=[names[i] for i in cand],
                         engine_up=eng_up, engine_down=eng_dn, memory=[MEM[j]["id"] for j in used]))
    open(os.path.join(OUTD, "prompts.txt"), "w").write("\n".join(text))
    json.dump(dict(names=names, kept=okp.cpu().tolist(), jobs=jobs), open(os.path.join(OUTD, "menus.json"), "w"))
    print(f"prompts for {len(jobs)} TEST genes -> {OUTD}/prompts.txt  ({time.time() - t0:.0f}s)")


def score():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    G = notebook()
    torch, np = G["torch"], G["np"]
    pw, A_te, DEV, pearson_rows = G["pw"], G["A_te"], G["DEV"], G["pearson_rows"]
    from scipy.stats import binomtest
    M = json.load(open(os.path.join(OUTD, "menus.json")))
    assert M["names"] == G["names"], "pathway sets changed since prepare"
    okp = torch.tensor(M["kept"], device=DEV)
    props = {}
    for line in open(os.path.join(OUTD, "proposals.jsonl")):
        if line.strip():
            d = json.loads(line); props[d["n"]] = d

    def ranks(v):
        sc_ = torch.where(okp, pw(v), torch.tensor(float("nan"), device=DEV)); ok_ = torch.isfinite(sc_)
        order = torch.argsort(torch.where(ok_, sc_, torch.tensor(-1e9, device=DEV)), descending=True)
        rk = torch.empty_like(order); rk[order] = torch.arange(len(order), device=DEV)
        return rk.cpu().numpy(), int(ok_.sum())

    jobs = M["jobs"]; t_null = np.roll([j["t"] for j in jobs], 1)
    rows, HYP, bad = [], [], []
    for job, tn in zip(jobs, t_null):
        d = props.get(job["n"], {}); nc = len(job["cand"])
        okans = (d.get("gene") == job["gene"] and all(len(set(d.get(x, []))) == N_PICK and
                 all(isinstance(i, int) and 1 <= i <= nc for i in d[x]) for x in ("up", "down")))
        if not okans:
            bad.append(job["n"]); continue
        pick = {x: [job["cand"][i - 1] for i in d[x]] for x in ("up", "down")}
        rk, n_ok = ranks(A_te[job["t"]]); rk_n, _ = ranks(A_te[int(tn)]); cut = max(1, int(0.10 * n_ok))
        hit = lambda ix, x, r_: bool(r_[ix] < cut) if x == "up" else bool(r_[ix] >= n_ok - cut)
        mine = [(ix, x) for x in ("up", "down") for ix in pick[x]]
        eng = [(ix, "up") for ix in job["engine_up"]] + [(ix, "down") for ix in job["engine_down"]]
        conf = [hit(ix, x, rk) for ix, x in mine]
        changed = [p for p in mine if p not in eng]; replaced = [p for p in eng if p not in mine]
        rows.append(dict(n=job["n"], gene=job["gene"], claude=float(np.mean(conf)),
                         null=float(np.mean([hit(ix, x, rk_n) for ix, x in mine])),
                         engine=float(np.mean([hit(ix, x, rk) for ix, x in eng])),
                         engine_null=float(np.mean([hit(ix, x, rk_n) for ix, x in eng])),
                         changed=len(changed), changed_hits=sum(hit(ix, x, rk) for ix, x in changed),
                         replaced_hits=sum(hit(ix, x, rk) for ix, x in replaced)))
        for (ix, x), c in zip(mine, conf):
            HYP.append(dict(id=f"H-{job['gene']}-{x}-{ix}", gene=job["gene"], type="hypothesis",
                            claim=f"knocking down {job['gene']} {'raises' if x == 'up' else 'lowers'} {M['names'][ix]}",
                            source=f"Claude (this session) reading memory facts {job['memory']}, the gene's Human Protein Atlas "
                                   f"document and the reasoning engine's prediction", why=d.get("why", ""),
                            status="confirmed" if c else "refuted", engine_also_picked=(ix, x) in eng,
                            checked_against="the held-out measured TEST response (top / bottom 10% of kept pathways)"))
    with open(os.path.join(OUTD, "memory_bank_hypotheses.jsonl"), "w") as fh:
        for h_ in HYP:
            fh.write(json.dumps(h_) + "\n")

    def flip_p(diffs, rng):
        obs = diffs.mean()
        return float(np.mean([np.mean(diffs * rng.choice([-1, 1], len(diffs))) >= obs for _ in range(5000)]))

    P_(RULE); P_("CLAUDE AS THE LLM, WITH THE REASONING ENGINE AND THE MEMORY BANK: THE PROPOSAL LOOP"); P_(RULE)
    pred = torch.tensor(np.load(os.path.join(CACHE, "engine_pred_test.npy")), device=DEV)
    r_eng = torch.nan_to_num(pearson_rows(pred, A_te)).cpu().numpy()
    P_(f"  engine on all {len(r_eng):,} TEST knockdowns (no LLM features): median r {np.median(r_eng):+.3f}")
    P_(f"  genes answered {len(rows)} of {len(jobs)}" + (f"; malformed / missing: {bad}" if bad else ""))
    c, nl, e, en = (np.array([r[k] for r in rows]) for k in ("claude", "null", "engine", "engine_null"))
    P_(f"  confirmed share of picks: Claude {c.mean():.1%} | Claude's picks on another gene's measurement {nl.mean():.1%} | "
       f"engine's picks {e.mean():.1%} | engine's picks on another gene {en.mean():.1%}")
    p1 = flip_p(c - nl, np.random.default_rng(3)); pe = flip_p(e - en, np.random.default_rng(4))
    w, l = int((c > e).sum()), int((c < e).sum()); p2 = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    v1 = "BETTER" if p1 < 0.05 else "NO DIFFERENCE"
    v2 = "BETTER" if p2 < 0.05 and w > l else "WORSE" if p2 < 0.05 and l > w else "NO DIFFERENCE"
    P_(f"  L1 Claude vs null: mean +{(c - nl).mean():.3f}, sign-flip p {p1:.2g} -> {v1}")
    P_(f"  L2 Claude vs the engine's own picks (PRIMARY): better on {w} genes, worse on {l}, tied {len(rows) - w - l}, "
       f"p {p2:.2g} -> {v2}")
    P_(f"  E1 engine vs null (descriptive): mean +{(e - en).mean():.3f}, sign-flip p {pe:.2g}")
    nch = sum(r["changed"] for r in rows)
    P_(f"  D  Claude changed {nch} of {8 * len(rows)} picks from the engine's; confirmed: changed picks "
       f"{sum(r['changed_hits'] for r in rows)}/{nch}, the engine picks they replaced {sum(r['replaced_hits'] for r in rows)}/{nch}")
    P_(f"  memory bank: {len(HYP)} hypotheses written ({sum(h['status'] == 'confirmed' for h in HYP)} confirmed) -> "
       f"outputs/claudeloop/memory_bank_hypotheses.jsonl")
    open(OUT, "w").write("\n".join(out) + "\n")
    json.dump(dict(engine_test_median_r=float(np.median(r_eng)), answered=len(rows), malformed=bad, per_gene=rows,
                   L1=dict(p=p1, verdict=v1), L2=dict(better=w, worse=l, p=float(p2), verdict=v2), E1=dict(p=pe)),
              open(os.path.join(OUTD, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    {"prepare": prepare, "score": score}[sys.argv[1]]()
