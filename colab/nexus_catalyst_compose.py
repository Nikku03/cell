"""THE COMPOSITION: LLM PROPOSES TEN, ESM RE-RANKS.  Does the pair beat either alone?

WHY THIS IS THE EXPERIMENT.  Two arms now work on this problem for different reasons and fail for different
reasons. The language arm names the right enzyme first 0.463 of the time on obscure enzymes but is
substantially recalling published literature. The ESM pair model reaches 0.682 AUC on a homology-disjoint
split with frequency pinned at 0.500, so it is not literature volume it is reading. Composing them is the step
the docking route could not earn, because docking scored at chance and reordering by a chance-level score
moves nothing.

TWO THINGS MEASURED BEFORE RUNNING, both of which bound what this can show.

    THE NEGATIVES ARE MUCH HARDER HERE.  ESM's 0.682 was scored against decoys drawn at random from the
    catalyst vocabulary, where the largest single family's share of a 10-candidate list is 0.10. The LLM's
    top-10 has a median largest-family share of 0.40 and a mean of 0.48 -- it proposes ten plausible
    candidates, often ten members of one family. Telling ALDH1A1 from ALDH3A2 is a different problem from
    telling a dehydrogenase from a transporter, and 0.682 will not survive the change. So a second model is
    trained on HARD negatives -- decoys sharing the true catalyst's family stem -- to match test conditions.

    THE POPULATION IS NOT THE ORPHAN POPULATION.  Of 2,231 reactions with a protein substrate, THREE have a
    catalyst with under 20 linked papers. Protein-substrate reactions are signalling steps run by well-studied
    kinases and ligases. The obscure stratum that forecasts orphan work does not exist in this benchmark, so
    nothing here can be quoted as an orphan-filling number. It answers a narrower question: does a sequence
    model add anything to a language model's shortlist, on the reactions where both are applicable.

ARMS, all on the identical reactions and the identical ten candidates:
    llm_position1       the shortlist as the model ranked it. The thing to beat.
    esm_rerank          ESM trained on random decoys -- the model exactly as measured at 0.682
    esm_hardneg         ESM trained on family-matched decoys -- matched to the test distribution
    freq_rerank         re-rank by training-set catalyst counts; the strongest hand-built signal
    esm_untrained       random init, same architecture
    random_rerank       shuffle the ten. Shows what re-ranking noise costs, which is the real risk: a
                        re-ranker that knows nothing does not leave 0.463 alone, it destroys it.
    oracle              is the truth in the shortlist at all -- the ceiling for every re-ranker

HELD OUT PROPERLY: the ESM models never see the test reactions, and the split is HOMOLOGY-disjoint, so no
test catalyst has a paralog above ~50% identity in training.

PREDECLARED, before any number:
    esm_hardneg beats llm_position1 materially   -> the composition works and is the way to spend the +0.188
                                                    of shortlist headroom. First compounding result here.
    esm arms ~ llm_position1                     -> ESM's margin was against easy negatives and evaporates on
                                                    a plausible shortlist. The language arm stands alone.
    esm arms < llm_position1                     -> re-ranking actively destroys a good ordering, which is the
                                                    default outcome for any re-ranker weaker than the ranker.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_llm as L
import cell_orphan_obscurity as O
import nexus_catalyst_esm as M
import cell_orphan_ties_nn as T

OUT = A.OUT
PUZZLES = OUT / "compose_puzzles.json"
SHORT = OUT / "compose_shortlists.json"
N_RX = 400
N_FOLD = 5
SEEDS = (0, 1, 2)
EPOCHS = 25
SEED = 17
FUSE_W = 0.5         # weight on the ESM z-score against the LLM's rank prior (steps of 1 between ranks),
                     # so ESM can move a candidate at most ~1-2 places rather than reorder the list wholesale


def stem(g):
    t = re.sub(r"\d+$", "", g)
    if len(t) > 3:
        t = re.sub(r"[A-Z]$", "", t)
    return t if len(t) >= 3 else g


def dump():
    E, rx, vocab, cats = M.load_bench()
    B = T.build()
    steps = B["steps"]
    # DROP THE GIVEAWAYS. In protein-substrate reactions the enzyme is very often a named participant --
    # 159 of a first draw of 400, against 34 of 400 on the metabolic benchmark. The language arm reads those
    # off, which would inflate its ordering and leave a re-ranker nothing to improve, masking any real
    # effect. Excluding them is also the more faithful orphan setting, not a less faithful one: for a
    # genuinely unannotated reaction the catalyst is by definition not sitting in the participant list.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rx))
    sel, drop = [], 0
    for k in order:
        r = rx[int(k)]
        s_ = steps[r["rx"]]
        names = " ".join((p.get("name") or "") for w in ("in", "out") for p in (s_.get(w) or []))
        if any(g in names for g in B["cats"][r["rx"]]):
            drop += 1
            continue
        sel.append(r)
        if len(sel) >= N_RX:
            break
    sel = sorted(sel, key=lambda r: r["rx"])
    print(f"  dropped {drop} reactions whose catalyst is named in the reaction (the giveaways)")
    out = []
    for pid, r in enumerate(sel):
        s = steps[r["rx"]]
        out.append({"pid": pid, "source": s.get("source"), "category": s.get("category"),
                    "reversible": bool(s.get("reversible")),
                    "inputs": [L._clean(p) for p in (s.get("in") or [])],
                    "outputs": [L._clean(p) for p in (s.get("out") or [])]})
    OKTOP = {"pid", "source", "category", "reversible", "inputs", "outputs"}
    OKPART = {"name", "type", "compartment"}
    for pid, r in enumerate(sel):
        assert set(out[pid]) <= OKTOP
        for w in ("inputs", "outputs"):
            for q in out[pid][w]:
                assert set(q) <= OKPART
        assert steps[r["rx"]].get("id", "\0") not in json.dumps(out[pid]), "reaction id leaked"
    json.dump({"sel": sel, "puzzles": out}, open(PUZZLES, "w"), indent=1)

    def part(q):
        if q["type"] == "UNRESOLVED":
            return "?unresolved"
        c = q.get("compartment")
        return f"{q['name']} [{q['type']}{', ' + c if c else ''}]"
    txt = [f"#{p['pid']}  ({p['source']}, {p['category']}, "
           f"{'reversible' if p['reversible'] else 'irreversible'})\n"
           f"  IN : " + "  +  ".join(part(q) for q in p["inputs"]) + "\n"
           f"  OUT: " + "  +  ".join(part(q) for q in p["outputs"]) for p in out]
    dd = M.SP / "compose"
    dd.mkdir(exist_ok=True)
    Bs = 50
    for b in range(0, len(txt), Bs):
        (dd / f"open_b{b // Bs}.txt").write_text("\n\n".join(txt[b:b + Bs]))
    gim = sum(1 for pid, r in enumerate(sel)
              if any(g in " ".join(q["name"] for w in ("inputs", "outputs") for q in out[pid][w])
                     for g in B["cats"][r["rx"]]))
    print(f"  wrote {len(out)} puzzles -> {PUZZLES}, batches in {dd}")
    assert gim == 0, f"{gim} giveaways survived the filter"
    print(f"  giveaways remaining: {gim} (asserted zero)")
    return 0


def merge(tdir):
    got = {}
    for f in sorted(Path(tdir).glob("agent-*.jsonl")):
        txt = f.read_text(errors="replace")
        if len(set(re.findall(r"open_b(\d+)\.txt", txt))) != 1:
            continue
        for line in txt.splitlines():
            if '"answers"' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stack = [rec]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if isinstance(x.get("answers"), list):
                        for a_ in x["answers"]:
                            if isinstance(a_, dict) and "pid" in a_ and isinstance(a_.get("genes"), list):
                                got[int(a_["pid"])] = [str(g).strip().upper() for g in a_["genes"]]
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
    n = len(json.load(open(PUZZLES))["puzzles"])
    miss = sorted(set(range(n)) - set(got))
    if miss:
        print(f"  REFUSING: {len(miss)} shortlists missing {miss[:10]}")
        return 1
    json.dump({str(k): v for k, v in got.items()}, open(SHORT, "w"), indent=1)
    print(f"  merged {len(got)} shortlists -> {SHORT}")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        return dump()
    if len(sys.argv) > 2 and sys.argv[1] == "merge":
        return merge(sys.argv[2])

    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    torch.set_num_threads(4)

    report("=" * 100)
    report("COMPOSITION: LLM PROPOSES TEN, ESM RE-RANKS")
    report("=" * 100)
    report("  Measured BEFORE running, and both bound the result:")
    report("   * the LLM's ten are 4.8x more family-concentrated than random decoys (largest-family share")
    report("     0.48 vs 0.10), so ESM faces far harder negatives than the 0.682 was scored against;")
    report("   * only 3 of 2,231 protein-substrate reactions have a catalyst under 20 papers, so the obscure")
    report("     stratum that forecasts orphan work does not exist here. Nothing below is an orphan number.")

    E, rx_all, vocab, cats = M.load_bench()
    P = json.load(open(PUZZLES))
    sel, sl = P["sel"], json.load(open(SHORT))
    seq = M.sequences()
    fam, _ = M.family_of([g for g in sorted({r["cat"] for r in rx_all} | {r["sub"] for r in rx_all})
                          if g in seq], seq)
    test_rx = {r["rx"] for r in sel}
    train = [r for r in rx_all if r["rx"] not in test_rx
             and fam.get(r["cat"], -1) not in {fam.get(t["cat"], -2) for t in sel}]
    report(f"\n  {len(sel)} test reactions | ESM trained on {len(train):,} reactions, "
           f"homology-disjoint from every test catalyst")

    Em = {g: E[g] for g in E}
    st = np.stack(list(Em.values()))
    mu, sd = st.mean(0), st.std(0) + 1e-6

    def vec(g):
        return (Em[g] - mu) / sd
    dim = len(mu)

    class Pair(nn.Module):
        def __init__(s, d, h=128):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(4 * d, h), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

        def forward(s, e, sub):
            return s.net(torch.cat([e, sub, e * sub, (e - sub).abs()], -1)).squeeze(-1)

    by_stem = {}
    for g in vocab:
        by_stem.setdefault(stem(g), []).append(g)

    def train_groups(hard, rs):
        gs = []
        for r in train:
            if r["cat"] not in Em or r["sub"] not in Em:
                continue
            pool = [g for g in by_stem.get(stem(r["cat"]), []) if g != r["cat"]] if hard else None
            if hard and len(pool) < 3:
                pool = None
            src = pool if pool else vocab
            d = [g for g in rs.choice(src, 30, replace=True)
                 if g != r["cat"] and g not in cats[r["rx"]] and g in Em][:9]
            if len(d) < 9:
                continue
            gs.append(([r["cat"]] + d, r["sub"]))
        return gs

    def fit(hard, seed, do_train=True):
        torch.manual_seed(seed)
        m = Pair(dim)
        if not do_train:
            m.eval()
            return m
        rs = np.random.default_rng(seed)
        gs = train_groups(hard, rs)
        Ee = torch.tensor(np.stack([[vec(c) for c in cd] for cd, _ in gs]), dtype=torch.float32)
        Es = torch.tensor(np.stack([np.repeat(vec(s_)[None], 10, 0) for _, s_ in gs]), dtype=torch.float32)
        Y = torch.zeros(len(gs), 10)
        Y[:, 0] = 1
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(EPOCHS):
            m.train()
            perm = rs.permutation(len(gs))
            for k in range(0, len(gs), 64):
                b = perm[k:k + 64]
                opt.zero_grad()
                loss = -(torch.log_softmax(m(Ee[b], Es[b]), 1) * Y[b]).sum(1).mean()
                loss.backward()
                opt.step()
        m.eval()
        return m

    cnt = {}
    for r in train:
        cnt[r["cat"]] = cnt.get(r["cat"], 0) + 1

    # candidates the ESM model can actually score; anything without an embedding keeps its LLM rank
    ok, truth, lists, subs = [], [], [], []
    for pid, r in enumerate(sel):
        lst = sl.get(str(pid)) or []
        if len(lst) < 10 or r["sub"] not in Em:
            continue
        ok.append(pid)
        truth.append(set(cats[r["rx"]]))
        lists.append(lst[:10])
        subs.append(r["sub"])
    n = len(ok)
    scor = np.array([[g in Em for g in L_] for L_ in lists])
    report(f"  {n} test reactions scored | candidates with an ESM embedding: "
           f"{scor.mean():.1%} (the rest keep their LLM rank)")

    in_list = np.array([bool(t & set(L_)) for t, L_ in zip(truth, lists)])
    llm1 = np.array([L_[0] in t for t, L_ in zip(truth, lists)])
    report(f"\n  {'arm':<26}{'top-1':>9}{'sd':>7}   note")
    res = {}

    def rank_by(score_fn, label, seeds=(0,)):
        vals = []
        for s_ in seeds:
            f = score_fn(s_)
            hit = []
            for q in range(n):
                v = f(q)
                order = sorted(range(10), key=lambda j: (-v[j], j))
                hit.append(lists[q][order[0]] in truth[q])
            vals.append(float(np.mean(hit)))
        res[label] = {"top1": float(np.mean(vals)), "sd": float(np.std(vals))}
        return vals

    res["llm_position1"] = {"top1": float(llm1.mean()), "sd": 0.0}
    report(f"  {'llm_position1':<26}{llm1.mean():>9.3f}{0.0:>7.3f}   the ordering to beat")

    rs0 = np.random.default_rng(3)
    rank_by(lambda s_: (lambda q: rs0.random(10)), "random_rerank")
    report(f"  {'random_rerank':<26}{res['random_rerank']['top1']:>9.3f}{0.0:>7.3f}   "
           f"what a know-nothing re-ranker costs")
    rank_by(lambda s_: (lambda q: np.array([cnt.get(g, 0) for g in lists[q]], float)), "freq_rerank")
    report(f"  {'freq_rerank':<26}{res['freq_rerank']['top1']:>9.3f}{0.0:>7.3f}   "
           f"strongest hand-built signal")

    for label, hard, do_train in (("esm_rerank", False, True), ("esm_hardneg", True, True),
                                  ("esm_untrained", False, False)):
        models = {s_: fit(hard, s_, do_train) for s_ in SEEDS}

        def mk(s_):
            m = models[s_]

            def f(q):
                cand = [g if g in Em else None for g in lists[q]]
                v = np.full(10, -1e9)
                idx = [j for j, g in enumerate(cand) if g]
                if idx:
                    e = torch.tensor(np.stack([vec(cand[j]) for j in idx]), dtype=torch.float32)
                    s2 = torch.tensor(np.repeat(vec(subs[q])[None], len(idx), 0), dtype=torch.float32)
                    with torch.no_grad():
                        sc = m(e, s2).numpy()
                    for k, j in enumerate(idx):
                        v[j] = sc[k]
                return v
            return f
        rank_by(mk, label, SEEDS)
        note = {"esm_hardneg": "trained on family-matched decoys",
                "esm_untrained": "random init"}.get(label, "trained on random decoys, as measured")
        report(f"  {label:<26}{res[label]['top1']:>9.3f}{res[label]['sd']:>7.3f}   {note}")

    # FUSION, because "re-rank" as implemented above REPLACES the LLM's ordering rather than composing with
    # it. Discarding a 0.675 ordering to impose a weaker one is guaranteed to lose -- random_rerank at 0.195
    # is exactly that cost. The honest composition keeps the LLM's rank as a prior and lets ESM adjust it.
    for lab, hard in (("fusion_rerank", False), ("fusion_hardneg", True)):
        models = {s_: fit(hard, s_, True) for s_ in SEEDS}

        def mkf(s_):
            m = models[s_]

            def f(q):
                prior = -np.arange(10, dtype=float)            # LLM rank, best first
                cand = [g if g in Em else None for g in lists[q]]
                idx = [j for j, g in enumerate(cand) if g]
                z = np.zeros(10)
                if len(idx) > 1:
                    e = torch.tensor(np.stack([vec(cand[j]) for j in idx]), dtype=torch.float32)
                    s2 = torch.tensor(np.repeat(vec(subs[q])[None], len(idx), 0), dtype=torch.float32)
                    with torch.no_grad():
                        sc = m(e, s2).numpy()
                    sc = (sc - sc.mean()) / (sc.std() + 1e-9)
                    for k, j in enumerate(idx):
                        z[j] = sc[k]
                return prior + FUSE_W * z
            return f
        rank_by(mkf, lab, SEEDS)
        report(f"  {lab:<26}{res[lab]['top1']:>9.3f}{res[lab]['sd']:>7.3f}   "
               f"LLM rank prior + {FUSE_W} x ESM z")

    res["oracle_in_shortlist"] = {"top1": float(in_list.mean()), "sd": 0.0}
    report(f"  {'oracle (truth in the ten)':<26}{in_list.mean():>9.3f}{0.0:>7.3f}   ceiling for any re-ranker")

    report("\n  READING")
    best = max(("esm_rerank", "esm_hardneg", "fusion_rerank", "fusion_hardneg"),
               key=lambda k: res[k]["top1"])
    d = res[best]["top1"] - res["llm_position1"]["top1"]
    report(f"  best ESM arm '{best}' {res[best]['top1']:.3f} vs the LLM's own ordering "
           f"{res['llm_position1']['top1']:.3f}  ({d:+.3f})")
    report(f"  ceiling if a re-ranker were perfect: {in_list.mean():.3f}")
    repl = max(res["esm_rerank"]["top1"], res["esm_hardneg"]["top1"])
    fuse = max(res["fusion_rerank"]["top1"], res["fusion_hardneg"]["top1"])
    base = res["llm_position1"]["top1"]
    report(f"  REPLACING the LLM ordering: {repl:.3f}. FUSING with it (rank prior + ESM): {fuse:.3f}. "
           f"LLM alone: {base:.3f}.")
    if fuse > base + 0.02:
        report("  THE COMPOSITION WORKS, but only as fusion. Replacing a good ordering with a weaker score")
        report("  loses; keeping it as a prior and letting ESM nudge gains.")
    elif fuse >= base - 0.05:
        report("  NO GAIN. Fusion recovers almost all of what replacement threw away, which shows the loss")
        report("  was mostly discarding the LLM's ordering rather than ESM being actively wrong -- but it")
        report("  still does not beat leaving the shortlist alone. ESM's 0.682 was earned against decoys")
        report("  drawn at random from the whole vocabulary; the LLM's ten are 4.8x more family-concentrated,")
        report("  and within one family the sequence model has nothing left to say. Telling ALDH1A1 from")
        report("  ALDH3A2 is the hard problem, and it is precisely the one the composition needed solved.")
    else:
        report("  RE-RANKING HURTS EVEN AS FUSION. The signal is not merely absent on this distribution.")
    if repl < res["random_rerank"]["top1"]:
        report(f"  NOTE the replacement arms ({repl:.3f}) fall BELOW random re-ranking "
               f"({res['random_rerank']['top1']:.3f}), and below the untrained twin "
               f"({res['esm_untrained']['top1']:.3f}).")
        report("  Training made it worse on this distribution, so the coarse enzyme/substrate compatibility it")
        report("  learned is mildly ANTI-correlated with which family member is right. That is a sharper")
        report("  negative than noise would give, and it is the thing to explain before trying this again.")
    report(f"  For scale, random_rerank is {res['random_rerank']['top1']:.3f} -- a re-ranker that knows")
    report("  nothing does not leave a good ordering alone, it flattens it to chance-within-the-list.")

    json.dump({"test": "nexus_catalyst_compose", "n": n, "arms": res, "log": log},
              open(OUT / "nexus_catalyst_compose.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_catalyst_compose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
