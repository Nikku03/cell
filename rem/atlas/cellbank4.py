"""A compact, fast memory bank of all the processed cell data; the reasoning engine reasons over it; Claude arbitrates.

WHY. The memory bank's job is to take in the processed data and supply it back: a lot of information in little
storage, with fast access. The reasoning engine's job is to reason over what the bank supplies. claudeloop.py
found that Claude's wins came from close partners' measured experiments read out of the memory. Here the bank
holds ALL the measured cell data, the engine itself is given what the bank supplies, and Claude works on top.

THE MEMORY BANK (memory_bank/cell_v4/, written by `build`)
  records  every knockdown with a phenotype (energy-test p < 0.05) in three Replogle 2022 screens (CC BY 4.0):
           A K562 genome-wide (day 8), B K562 essential (day 6), C RPE1 essential. Each record is the specific
           response (the screen's own mean TRAIN-gene response removed), mapped onto screen A's 8,248 genes,
           projected on the 128 components fitted on A's TRAIN responses, stored as int8 codes + one scale.
           Only knockdowns of genes with screen-A features are kept (the count dropped is reported).
  genes    interaction partners (HPA consensus) and the 20 most co-essential genes (DepMap, K562 removed).
  index    SQLite: record id, gene, Ensembl id, screen, split, energy p; journal.jsonl for later writes.
  access   get(gene), nearest(query codes, k) by one matrix multiply on unit codes, partners, coessential.
  ALL records are stored, TEST included, with their split. Every use below hides all screen-A TEST records
  and, for each query gene, that gene's own records in every screen.

THE ENGINE ARMS (the notebook's section 9 engine and recipe, non-LLM features, seeds 0-2 averaged, all TEST)
  E0   memory = screen-A TRAIN records only (as before).
  E1   FULL BANK: snap over A TRAIN + B + C records (screen-tagged keys, per-screen value maps starting at
       identity), plus two inputs from the bank: the mean K562 record of the gene's interaction partners, and
       the correlation-weighted mean K562 record of its 5 most co-essential genes.
  E1s  control: E1 with gene labels shuffled within B and C and in both inputs (records still hidden by their
       true gene, so nothing leaks).
  E2   E1 + the query gene's own RPE1 record (the same knockdown in another cell type) as an input.
  R1   ridge (the notebook's lambda-by-CV ridge) on the same inputs as E1.

GATES, PREDECLARED (per-gene Pearson r on all TEST knockdowns; paired sign test;
                    BETTER = p < 0.05 and median gain >= +0.01, WORSE = p < 0.05 and median gain <= -0.01)
  H1  blocking, compression keeps what matters. On TRAIN only, predicting each gene as the mean of its
      partners' records: median r from the int8 codes >= median r from the full float records - 0.005.
      Storage ratio and query speed reported.
  G1  PRIMARY: E1 vs E0. Counts only if G3 is BETTER.
  G2  E2 vs E1: does another cell type's measurement of the same knockdown help?
  G3  E1 vs E1s: the right records, not just more records.
  G4  descriptive: E1 vs R1 (does the engine use the bank better than a plain learner?).

CLAUDE ARBITRATES (50 random TEST genes, seed 31, none of claudeloop's 100, whose results Claude has seen)
  For each gene Claude reads its Atlas document and up to 8 sources from the bank, each summarised by its top
  pathways and genes: S0 the E2 engine's prediction (seed 0); S1-S3 the records its snapping step recalled;
  S4-S5 the largest partner records; S6 the most co-essential gene's record; S7 the gene's own RPE1 record.
  Claude gives each source a weight 0-3; Claude's prediction is the weighted mean of the unit-scaled sources
  in gene space. The weights are committed before `score` reads any TEST answer.
  C1  PRIMARY for Claude: Claude's blend vs S0 alone.
  C2  Claude's blend vs the equal-weight blend of the same sources (does choosing the weights matter?).
NOT: one compression (128 components, int8) and one memory design; Claude may have read these screens' papers
     in training; screen B of the query gene is never used (it is the noise ceiling, 0.315).

Run: python3 -P rem/atlas/cellbank4.py build | engine | score   (Claude writes outputs/cellbank4/weights.jsonl
     between engine and score, and it is committed first)
"""

from __future__ import annotations
import json
import math
import os
import re
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
NB = R("colab", "cell_reasoning_llm_a100.ipynb")
BANK = R("memory_bank", "cell_v4")
OUTD = R("outputs", "cellbank4")
CACHE = os.path.join(HERE, "_cache", "cellbank4")
OUT = os.path.join(HERE, "RESULTS_cellbank4.txt")
CACHED = {
    "gwps.h5ad": "perturbseq/K562_gwps_normalized_bulk_01.h5ad",
    "ess.h5ad": "perturbseq_ess/K562_essential_normalized_bulk_01.h5ad",
    "proteinatlas.tsv.zip": "hpa/proteinatlas.tsv.zip", "rna_celline.tsv.zip": "hpa/rna_celline.tsv.zip",
    "subcellular_location.tsv.zip": "hpa/subcellular_location.tsv.zip",
    "interaction_consensus.tsv.zip": "hpa/interaction_consensus.tsv.zip",
    "CRISPRGeneEffect.csv": "depmap/CRISPRGeneEffect.csv", "ReactomePathways.gmt.zip": "reactome/ReactomePathways.gmt.zip",
}
RPE1 = os.path.join(HERE, "_cache", "perturbseq_rpe1", "rpe1_normalized_bulk_01.h5ad")
SCREENS = ("A", "B", "C")
SCREEN_NAME = {"A": "K562 genome-wide screen", "B": "K562 essential screen", "C": "RPE1 screen (another cell type)"}
ARMS = ("E0", "E1", "E1s", "E2")
SEEDS = (0, 1, 2)
N_CLAUDE, N_COESS, RULE = 50, 5, "=" * 97
QUICK = os.environ.get("CB4_QUICK") == "1"          # plumbing check only: seed 999, 2 epochs, scratch output
BAD = re.compile(r"virus|viral|SARS|HIV|influenza|infect|disease|cancer|amyloid|parasit|bacteri|hepatitis|"
                 r"defective|mutant|syndrome|tumou?r|leishmania|tubercul|covid|_\d{5,}", re.I)
if QUICK:
    _Q = os.path.join(HERE, "_cache", "cellbank4", "quick")
    OUTD, CACHE, BANK, OUT = (os.path.join(_Q, x) for x in ("out", "cache", "bank", "RESULTS.txt"))
    SEEDS, N_CLAUDE = (999,), 5


def notebook():
    """Execute the notebook's data, feature, ridge, model and pathway code (no LLM)."""
    data = os.path.join(HERE, "_cache", "cellbank4", "data"); os.makedirs(data, exist_ok=True)
    for name, rel in CACHED.items():
        src, dst = os.path.join(HERE, "_cache", rel), os.path.join(data, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    os.environ.update(CELL_SMOKE="1", CELL_DATA=data, CELL_OUT=os.path.join(HERE, "_cache", "cellbank4", "nb_out"))
    cells = {}
    for c in json.load(open(NB))["cells"]:
        t = "".join(c["source"]); m = re.match(r"# ---- (\d+)\.", t)
        if c["cell_type"] == "code" and m:
            cells[int(m.group(1))] = t
    G = {"__name__": "notebook"}
    run = lambda k, code: exec(compile(code, f"notebook section {k}", "exec"), G)
    run(0, cells[0]); G["EPOCHS"] = 2 if QUICK else 150
    for k in (1, 2, 3, 4, 5, 7):
        run(k, cells[k])
    for k, cut in ((8, "\nPRED, INFO = {}, {}"), (9, "\nfor name, kind, X in ("), (11, "\nrng = np.random.default_rng(3)")):
        head = cells[k].split(cut)[0]; assert head != cells[k]; run(k, head)
    G["torch"].set_num_threads(4)
    return G


# ---------------------------------------------------------------------------------------------------------
# the memory bank
# ---------------------------------------------------------------------------------------------------------
class Bank:
    """Compact store of measured knockdown responses: int8 codes on a shared 128-component basis."""

    def __init__(self, path):
        import numpy as np
        z = np.load(os.path.join(path, "records.npz"), allow_pickle=False)
        self.codes, self.scale, self.basis = z["codes"], z["scale"], z["basis"]
        meta = json.load(open(os.path.join(path, "records_meta.json")))
        self.gene, self.screen, self.split, self.ens = meta["gene"], meta["screen"], meta["split"], meta["ens"]
        self.genes = json.load(open(os.path.join(path, "genes.json")))
        self.by_gene = {}
        for i, (g, s) in enumerate(zip(self.gene, self.screen)):
            self.by_gene.setdefault(g, {})[s] = i
        u = self.codes.astype(np.float32); self.unit = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        self.path = path

    def z(self, ids):
        import numpy as np
        ids = np.asarray(ids)
        return self.codes[ids].astype(np.float32) * (self.scale[ids, None] / 127.0)

    def get(self, gene, screen=None):
        d = self.by_gene.get(gene, {})
        return d if screen is None else d.get(screen)

    def nearest(self, zq, k=16, allowed=None):
        import numpy as np
        q = zq / np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
        S = q @ self.unit.T
        if allowed is not None:
            S[:, ~allowed] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        return idx, np.take_along_axis(S, idx, 1)

    def add(self, entry):
        with open(os.path.join(self.path, "journal.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(entry, written=time.strftime("%Y-%m-%d"))) + "\n")


def build():
    t0 = time.time()
    G = notebook()
    torch, np = G["torch"], G["np"]
    A, KN, KE, phen, ACT, tide, Vy, split = G["A"], G["KN"], G["KE"], G["phen"], G["ACT"], G["tide"], G["Vy"], G["split"]
    gidx, PPI, D = G["gidx"], G["PPI"], G["D"]
    row_of = {g: j for j, g in enumerate(KN)}
    recs = dict(z=[], gene=[], ens=[], screen=[], split=[], ep=[])

    def add(Z, genes, eps, ens, scr):
        recs["z"].append(Z); recs["gene"] += genes; recs["ep"] += [float(e) for e in eps]
        recs["ens"] += [e or "" for e in ens]; recs["screen"] += [scr] * len(genes); recs["split"] += [split(g) for g in genes]

    add(torch.nan_to_num(ACT[phen] - tide) @ Vy, list(KN), [A["ep"][i] for i in phen], list(KE), "A")
    dropped = {}
    for scr, path in (("B", G["P"]["ess.h5ad"]), ("C", RPE1)):
        S = G["load_screen"](path); sg = {}
        for j, g in enumerate(S["genes"]):
            sg.setdefault(g, j)
        cols = [j for j, g in enumerate(A["genes"]) if g in sg]; scol = [sg[A["genes"][j]] for j in cols]
        ok = [i for i, e in enumerate(S["ep"]) if np.isfinite(e) and e < 0.05]
        keep = [i for i in ok if S["kos"][i] in row_of]; dropped[scr] = (len(ok) - len(keep), len(ok), len(cols))
        M = torch.full((len(keep), A["X"].shape[1]), float("nan"))
        M[:, cols] = S["X"][keep][:, scol]
        for r_, i in enumerate(keep):
            if S["kos"][i] in gidx:
                M[r_, gidx[S["kos"][i]]] = float("nan")          # never a knockdown's own gene
        trk = [r_ for r_, i in enumerate(keep) if split(S["kos"][i]) == "TRAIN"]
        td = torch.nanmean(M[trk], 0)
        add(torch.nan_to_num(M - td) @ Vy, [S["kos"][i] for i in keep], [S["ep"][i] for i in keep],
            [KE[row_of[S["kos"][i]]] for i in keep], scr)
    Z = torch.cat(recs["z"]).cpu()
    scale = Z.abs().max(1).values.clamp(min=1e-12)
    codes = torch.round(Z / scale[:, None] * 127).to(torch.int8)

    # gene table: partners and co-essential genes
    sym_of = {e: g for g, e in zip(KN, KE) if e}
    dep_ok = D[:, -1] > 0
    Dz = D[:, :-1] - D[:, :-1].mean(1, keepdim=True); Dz = Dz / Dz.norm(dim=1, keepdim=True).clamp(min=1e-12)
    C = Dz @ Dz.T; C[~dep_ok] = float("nan"); C[:, ~dep_ok] = float("nan"); C.fill_diagonal_(float("nan"))
    genes = {}
    for j, (g, e) in enumerate(zip(KN, KE)):
        part = sorted({sym_of[q] for q in PPI.get(e, ()) if q in sym_of and sym_of[q] != g})
        co = []
        if bool(dep_ok[j]):
            v = torch.nan_to_num(C[j], nan=-2.0); top = v.argsort(descending=True)[:20]
            co = [[KN[int(i)], round(float(v[i]), 3)] for i in top if float(v[i]) > -2]
        genes[g] = dict(row=j, ens=e or "", partners=part, coessential=co)

    os.makedirs(BANK, exist_ok=True)
    np.savez_compressed(os.path.join(BANK, "records.npz"), codes=codes.numpy(), scale=scale.numpy().astype(np.float32),
                        basis=Vy.cpu().numpy().astype(np.float16))
    json.dump(dict(gene=recs["gene"], screen=recs["screen"], split=recs["split"], ens=recs["ens"], ep=recs["ep"]),
              open(os.path.join(BANK, "records_meta.json"), "w"))
    json.dump(genes, open(os.path.join(BANK, "genes.json"), "w"))
    db = os.path.join(BANK, "index.sqlite")
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, gene TEXT, ensembl TEXT, screen TEXT, split TEXT, energy_p REAL)")
    con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)",
                    [(i, recs["gene"][i], recs["ens"][i], recs["screen"][i], recs["split"][i], recs["ep"][i]) for i in range(len(recs["gene"]))])
    con.execute("CREATE INDEX by_gene ON records(gene)"); con.commit(); con.close()
    open(os.path.join(BANK, "journal.jsonl"), "a").close()

    # H1: does the compression keep what matters? (TRAIN only)
    bank = Bank(BANK)
    Y_tr, rtr = G["Y_tr"], G["rtr"]
    trow = {KN[int(rtr[j])]: j for j in range(len(rtr))}
    rf, rc = [], []
    pred_f, pred_c, truth = [], [], []
    for g, j in trow.items():
        ps = [trow[p] for p in genes[g]["partners"] if p in trow]
        if ps:
            pred_f.append(Y_tr[ps].mean(0)); truth.append(Y_tr[j])
            pred_c.append(torch.tensor(bank.z([bank.get(KN[int(rtr[p])], "A") for p in ps]).mean(0)) @ Vy.T.cpu())
    T = torch.stack(truth)
    rf = torch.nan_to_num(G["pearson_rows"](torch.stack(pred_f), T)).numpy(); rc = torch.nan_to_num(G["pearson_rows"](torch.stack(pred_c), T)).numpy()
    raw_bytes = len(bank.gene) * A["X"].shape[1] * 4
    small = bank.codes.nbytes + bank.scale.nbytes + bank.basis.nbytes
    q = bank.z(np.arange(1000)); t1 = time.perf_counter(); bank.nearest(q, 16); ms = (time.perf_counter() - t1) * 1000 / 1000
    t1 = time.perf_counter()
    for g in KN[:1000]:
        bank.get(g)
    ms_get = (time.perf_counter() - t1) * 1000 / 1000
    h1 = float(np.median(rc)) >= float(np.median(rf)) - 0.005
    rep = dict(records={s: bank.screen.count(s) for s in SCREENS}, dropped_no_features=dropped,
               H1=dict(n=len(rf), median_r_full=float(np.median(rf)), median_r_int8=float(np.median(rc)), passed=h1),
               bytes_full_float32=raw_bytes, bytes_bank=int(small), ratio=raw_bytes / small,
               ms_per_query_nearest16=ms, ms_per_get=ms_get, seconds=time.time() - t0)
    json.dump(rep, open(os.path.join(BANK, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


# ---------------------------------------------------------------------------------------------------------
# the engine over the bank
# ---------------------------------------------------------------------------------------------------------
def inputs(G, bank, shuffle):
    """Per-gene inputs from the bank: partner mean, co-essential mean (K562 records; A only if TRAIN)."""
    torch, np = G["torch"], G["np"]
    KN, NPH = G["KN"], G["NPH"]
    k562 = {}
    for g in KN:
        d = bank.get(g)
        if "A" in d and bank.split[d["A"]] == "TRAIN":
            k562[g] = d["A"]
        elif "B" in d:
            k562[g] = d["B"]
    label = {g: g for g in k562}
    if shuffle:                                       # which gene's record is filed under which name
        ks = sorted(k562); perm = np.random.default_rng(5).permutation(len(ks))
        label = {ks[i]: ks[perm[i]] for i in range(len(ks))}
    recof = {label[g]: k562[g] for g in k562}         # name -> record id (shuffled names in the control)
    P_, C_ = torch.zeros(NPH, 129), torch.zeros(NPH, 129)
    for j, g in enumerate(KN):
        info = bank.genes[g]
        ids = [recof[p] for p in info["partners"] if p in recof and bank.gene[recof[p]] != g]
        if ids:
            P_[j, :128] = torch.tensor(bank.z(ids).mean(0)); P_[j, 128] = math.log1p(len(ids))
        co = [(recof[p], r) for p, r in info["coessential"] if p in recof and bank.gene[recof[p]] != g][:N_COESS]
        w = np.array([max(r, 0.0) for _, r in co])
        if co and w.sum() > 0:
            C_[j, :128] = torch.tensor((bank.z([i for i, _ in co]) * w[:, None]).sum(0) / w.sum()); C_[j, 128] = float(w.mean())
    return P_, C_


def scaled(G, blocks):
    """X_nollm followed by each bank input scaled like a notebook feature block (unit mean squared TRAIN norm)."""
    torch, rtr = G["torch"], G["rtr"]
    out = [G["X_nollm"].cpu()]
    for M in blocks:
        M = M.float(); c = float((M[rtr] ** 2).sum(1).mean()) + 1e-12
        out.append(M / math.sqrt(c))
    return torch.cat(out, 1).to(G["DEV"])


def make_engine(G, d_x, d_y, n_screen):
    torch, nn, F = G["torch"], G["nn"], G["F"]
    Engine = G["Engine"]

    class EngineBank(Engine):
        """The section 9 engine; memory records carry a screen tag (key and value) and a per-screen value map."""

        def __init__(self):
            super().__init__(d_x, d_y)
            self.sk, self.sv = nn.Embedding(n_screen, self.q.out_features), nn.Embedding(n_screen, self.enc[0].out_features)
            nn.init.zeros_(self.sk.weight); nn.init.zeros_(self.sv.weight)
            self.adapt = nn.Parameter(torch.eye(d_y).repeat(n_screen, 1, 1))

        def forward(self, x, mx, my, ms, hide):
            mya = torch.zeros_like(my)
            for s in range(self.adapt.shape[0]):
                sel = ms == s
                if bool(sel.any()):
                    mya[sel] = my[sel] @ self.adapt[s]
            h = self.enc(x); K = self.key(mx) + self.sk(ms); V = self.val(mya) + self.sv(ms); self.tops = []
            for _ in range(self.steps):
                S = (self.q(h) @ K.T / math.sqrt(K.shape[1])).masked_fill(hide, float("-inf"))
                top = S.topk(self.k, dim=1); self.tops.append(top.indices)
                w = torch.softmax(top.values, 1).unsqueeze(-1)
                r_y = (w * mya[top.indices]).sum(1); r_h = (w * V[top.indices]).sum(1)
                h = torch.tanh(self.tree(torch.cat([x, r_h, h], 1)))
            g = torch.sigmoid(self.mix(h))
            return g * r_y + (1 - g) * self.out(h)

    return EngineBank()


def run_arm(G, bank, arm, seed, X, keep_tops=None):
    """Train one arm (section 9 recipe) and return TEST predictions in the 128-component space."""
    torch, np, nn = G["torch"], G["np"], G["nn"]
    rtr, rte, Y_tr, Ztr, Vy, DEV, KN = G["rtr"], G["rte"], G["Y_tr"], G["Ztr"], G["Vy"], G["DEV"], G["KN"]
    pearson_rows = G["pearson_rows"]
    row_of = {g: j for j, g in enumerate(KN)}
    screens = ("A",) if arm == "E0" else SCREENS
    ids = [i for i in range(len(bank.gene)) if bank.screen[i] in screens and not (bank.screen[i] == "A" and bank.split[i] != "TRAIN")]
    true_row = torch.tensor([row_of[bank.gene[i]] for i in ids], device=DEV)
    key_row = true_row.clone()
    if arm == "E1s":                                  # file B and C records under other genes of the same screen
        for s in ("B", "C"):
            pos = [p for p, i in enumerate(ids) if bank.screen[i] == s]
            perm = np.random.default_rng(6 + SCREENS.index(s)).permutation(len(pos))
            key_row[pos] = true_row[[pos[q] for q in perm]]
    my = torch.tensor(bank.z(ids), device=DEV); ms = torch.tensor([SCREENS.index(bank.screen[i]) for i in ids], device=DEV)
    isA = ms == 0
    torch.manual_seed(seed)
    n = len(rtr); perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + seed)).to(DEV)
    va, fi = perm[: n // 10], perm[n // 10:]
    rows_tr = rtr.to(DEV)
    va_rows = rows_tr[va]
    in_train_mem = ~(isA & torch.isin(true_row, va_rows))           # as section 9: validation genes' A records out
    m = make_engine(G, X.shape[1], Ztr.shape[1], len(screens)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    mx_all = X[key_row]

    def fwd(q_rows, mem):
        hide = (true_row[mem][None, :] == q_rows[:, None])
        return m(X[q_rows], mx_all[mem], my[mem], ms[mem], hide)

    mem_tr = torch.nonzero(in_train_mem).squeeze(1); mem_all = torch.arange(len(ids), device=DEV)
    best, state = -2.0, None
    for ep in range(G["EPOCHS"]):
        m.train()
        for b in fi[torch.randperm(len(fi), device=DEV)].split(256):
            loss = 1 - pearson_rows(fwd(rows_tr[b], mem_tr) @ Vy.T, Y_tr[b]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 5 == 4 or ep == G["EPOCHS"] - 1:
            m.eval()
            with torch.no_grad():
                r = float(pearson_rows(fwd(rows_tr[va], mem_tr) @ Vy.T, Y_tr[va]).median())
            if r > best:
                best, state = r, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        pz = fwd(rte.to(DEV), mem_all)
        tops = [[ids[int(i)] for i in row] for row in m.tops[-1][:, :3].tolist()] if keep_tops else None
    return pz.cpu(), best, tops


def engine():
    t0 = time.time()
    G = notebook(); bank = Bank(BANK)
    torch, np = G["torch"], G["np"]
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    KN, KE, rte, Vy = G["KN"], G["KE"], G["rte"], G["Vy"]
    Pt, Ct = inputs(G, bank, False); Ps, Cs = inputs(G, bank, True)
    rp = torch.zeros(G["NPH"], 129)
    for j, g in enumerate(KN):
        c = bank.get(g, "C")
        if c is not None:
            rp[j, :128] = torch.tensor(bank.z([c])[0]); rp[j, 128] = 1.0
    X = dict(E0=G["X_nollm"], E1=scaled(G, [Pt, Ct]), E1s=scaled(G, [Ps, Cs]), E2=scaled(G, [Pt, Ct, rp]))
    preds, info, tops = {}, {}, None
    for arm in ARMS:
        outs = []
        for s in SEEDS:
            pz, best, tp = run_arm(G, bank, arm, s, X[arm], keep_tops=(arm == "E2" and s == SEEDS[0]))
            outs.append(pz); tops = tp if tp is not None else tops
            print(f"  {arm} seed {s}: held-out TRAIN r {best:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
            info.setdefault(arm, []).append(round(best, 4))
        preds[arm] = torch.stack(outs).mean(0); preds[arm + "_seed0"] = outs[0]
    pr, lam = G["ridge"](X["E1"], G["Y_tr"])
    np.save(os.path.join(CACHE, "R1.npy"), pr.cpu().numpy().astype(np.float16))
    torch.save(preds, os.path.join(CACHE, "engine_preds.pt"))
    json.dump(dict(heldout_train_r=info, ridge_lambda=lam), open(os.path.join(OUTD, "engine_info.json"), "w"), indent=1)

    # Claude's sources for 50 TEST genes (no TEST answer is read)
    names, pw = G["names"], G["pw"]
    okp = torch.tensor([not BAD.search(nm) for nm in names])
    seen = {j["t"] for j in json.load(open(R("outputs", "claudeloop", "menus.json")))["jobs"]}
    pool = [t for t in range(len(rte)) if t not in seen]
    pick = np.random.default_rng(31).choice(pool, size=N_CLAUDE, replace=False).tolist()
    Vt = Vy.T.cpu()

    def summary(z, own):
        v = torch.tensor(z) @ Vt
        sc = torch.where(okp, pw(v), torch.tensor(float("nan")))
        sc = torch.nan_to_num(sc, nan=0.0); vv = v.clone()
        if own in G["gidx"]:
            vv[G["gidx"][own]] = 0.0
        return (f"raises {'; '.join(names[i] for i in sc.argsort(descending=True)[:3].tolist())}. "
                f"Lowers {'; '.join(names[i] for i in sc.argsort()[:3].tolist())}. "
                f"Most raised genes {', '.join(G['A']['genes'][i] for i in vv.argsort(descending=True)[:4].tolist())}; "
                f"most lowered {', '.join(G['A']['genes'][i] for i in vv.argsort()[:4].tolist())}.")

    k562 = {}
    for g in KN:
        d = bank.get(g)
        if "A" in d and bank.split[d["A"]] == "TRAIN":
            k562[g] = d["A"]
        elif "B" in d:
            k562[g] = d["B"]
    jobs, text, vecs = [], [], {}
    for k, t in enumerate(pick):
        g, e = KN[int(rte[t])], KE[int(rte[t])]
        src = [("S0", "the reasoning engine's prediction (all bank inputs)", preds["E2_seed0"][t].numpy(), None)]
        for rid in tops[t]:
            if bank.gene[rid] != g and len(src) < 4 and all(s[3] != rid for s in src):
                src.append((f"S{len(src)}", f"{bank.gene[rid]} knockdown, {SCREEN_NAME[bank.screen[rid]]} (recalled by the engine)", bank.z([rid])[0], rid))
        part = sorted([k562[p] for p in bank.genes[g]["partners"] if p in k562 and p != g], key=lambda i: -float(bank.scale[i]))[:2]
        for rid in part:
            if all(s[3] != rid for s in src):
                src.append((f"S{len(src)}", f"{bank.gene[rid]} knockdown, {SCREEN_NAME[bank.screen[rid]]} (interaction partner)", bank.z([rid])[0], rid))
        co = [(p, r) for p, r in bank.genes[g]["coessential"] if p in k562 and p != g][:1]
        for p, r in co:
            rid = k562[p]
            if all(s[3] != rid for s in src):
                src.append((f"S{len(src)}", f"{p} knockdown, {SCREEN_NAME[bank.screen[rid]]} (most co-essential gene in DepMap, r = {r:.2f})", bank.z([rid])[0], rid))
        c = bank.get(g, "C")
        if c is not None:
            src.append((f"S{len(src)}", f"{g} itself knocked down in RPE1 cells (another cell type)", bank.z([c])[0], c))
        doc = G["prompt_for"](g, e).rsplit("Question:", 1)[0]
        text.append(f"### {k + 1}. {g}\n{doc}" + "\n".join(f"{s}: {d}: {summary(z, g)}" for s, d, z, _ in src) + "\n")
        jobs.append(dict(n=k + 1, t=int(t), gene=g, sources=[[s, d, (int(r) if r is not None else None)] for s, d, _, r in src]))
        vecs[str(k + 1)] = np.stack([z for _, _, z, _ in src])
    np.savez(os.path.join(CACHE, "claude_sources.npz"), **vecs)
    json.dump(dict(jobs=jobs), open(os.path.join(OUTD, "claude_jobs.json"), "w"), indent=1)
    head = ("INSTRUCTION (the same for every gene): give every listed source a weight from 0 to 3 for how much it should "
            "count in predicting this gene's knockdown response in K562 cells (0 = ignore). Reply as JSON lines "
            "{\"n\": 1, \"gene\": \"X\", \"w\": {\"S0\": 3, \"S1\": 0, ...}, \"why\": \"one sentence\"}.\n")
    open(os.path.join(OUTD, "claude_prompts.txt"), "w").write(head + "\n" + "\n".join(text))
    print(f"engine arms done; {len(jobs)} prompts -> {OUTD}/claude_prompts.txt  ({time.time() - t0:.0f}s)")


def score():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    G = notebook()
    torch, np = G["torch"], G["np"]
    from scipy.stats import binomtest
    A_te, Vy, pr = G["A_te"], G["Vy"].cpu(), G["pearson_rows"]
    preds = torch.load(os.path.join(CACHE, "engine_preds.pt"))
    Rr = {a: torch.nan_to_num(pr(preds[a] @ Vy.T, A_te)).numpy() for a in ARMS}
    Rr["R1"] = torch.nan_to_num(pr(torch.tensor(np.load(os.path.join(CACHE, "R1.npy"))).float(), A_te)).numpy()
    rep = json.load(open(os.path.join(BANK, "build_report.json")))

    def verdict(a, b, ra=None, rb=None):
        ra = Rr[a] if ra is None else ra; rb = Rr[b] if rb is None else rb
        d = ra - rb; w, l = int((d > 0).sum()), int((d < 0).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0; md = float(np.median(d))
        v = "BETTER" if p < 0.05 and md >= 0.01 else "WORSE" if p < 0.05 and md <= -0.01 else "NO DIFFERENCE"
        return dict(better=w, worse=l, p=float(p), median_diff=md, verdict=v)

    P_(RULE); P_("THE MEMORY BANK OF ALL THE CELL DATA, THE ENGINE REASONING OVER IT, AND CLAUDE ARBITRATING"); P_(RULE)
    P_(f"  bank: records {rep['records']} (dropped for lack of features: {rep['dropped_no_features']})")
    P_(f"  storage {rep['bytes_bank'] / 1e6:.1f} MB vs {rep['bytes_full_float32'] / 1e6:.0f} MB as float32 responses ({rep['ratio']:.0f}x smaller); "
       f"nearest-16 query {rep['ms_per_query_nearest16']:.3f} ms, get {rep['ms_per_get'] * 1000:.1f} us")
    h = rep["H1"]
    P_(f"  H1 compression keeps what matters: partner-mean prediction on {h['n']} TRAIN genes, median r full "
       f"{h['median_r_full']:+.3f} vs int8 {h['median_r_int8']:+.3f} -> {'PASS' if h['passed'] else 'FAIL'}")
    P_("  TEST knockdowns (2,525), median / mean r:")
    for a, lab in (("E0", "E0 engine, memory = screen-A TRAIN"), ("E1", "E1 engine, FULL BANK"), ("E1s", "E1s full bank, labels shuffled"),
                   ("E2", "E2 full bank + own RPE1 record"), ("R1", "R1 ridge on E1's inputs")):
        P_(f"     {lab:40s} {np.median(Rr[a]):+.3f} / {np.mean(Rr[a]):+.3f}")
    V = {}
    for tag, a, b in (("G1 full bank vs screen-A memory (PRIMARY)", "E1", "E0"), ("G2 + own RPE1 record", "E2", "E1"),
                      ("G3 right records vs shuffled", "E1", "E1s"), ("G4 engine vs ridge, same inputs (descr.)", "E1", "R1")):
        V[tag] = verdict(a, b); v = V[tag]
        P_(f"  {tag:44s} better {v['better']:,} / worse {v['worse']:,}  median {v['median_diff']:+.4f}  p {v['p']:.2g} -> {v['verdict']}")
    if V["G3 right records vs shuffled"]["verdict"] != "BETTER":
        P_("  NOTE: G3 is not BETTER, so G1 does not count.")

    # Claude's blends
    J = json.load(open(os.path.join(OUTD, "claude_jobs.json")))["jobs"]
    W = {}
    for line in open(os.path.join(OUTD, "weights.jsonl")):
        if line.strip():
            d = json.loads(line); W[d["n"]] = d
    vecs = np.load(os.path.join(CACHE, "claude_sources.npz"))
    unit = lambda M: (M - M.mean(1, keepdim=True)) / (M - M.mean(1, keepdim=True)).norm(dim=1, keepdim=True).clamp(min=1e-12)
    rc, r0, req, bad = [], [], [], []
    for j in J:
        d = W.get(j["n"], {}); names_ = [s[0] for s in j["sources"]]
        w = d.get("w", {})
        if d.get("gene") != j["gene"] or set(w) != set(names_) or not all(isinstance(w[s], int) and 0 <= w[s] <= 3 for s in names_) or not any(w.values()):
            bad.append(j["n"]); continue
        S = unit(torch.tensor(vecs[str(j["n"])]).float() @ Vy.T)
        truth = A_te[j["t"]][None]
        wt = torch.tensor([float(w[s]) for s in names_])
        blend = (wt[:, None] * S).sum(0, keepdim=True) / wt.sum()
        rc.append(float(torch.nan_to_num(pr(blend, truth))[0])); r0.append(float(torch.nan_to_num(pr(S[:1], truth))[0]))
        req.append(float(torch.nan_to_num(pr(S.mean(0, keepdim=True), truth))[0]))
    rc, r0, req = np.array(rc), np.array(r0), np.array(req)
    P_(f"  Claude arbitrating on {len(rc)} TEST genes" + (f" (malformed: {bad})" if bad else "") +
       f": median r Claude's blend {np.median(rc):+.3f} | engine (S0) {np.median(r0):+.3f} | equal-weight blend {np.median(req):+.3f}")
    for tag, a, b in (("C1 Claude's blend vs the engine (PRIMARY)", rc, r0), ("C2 Claude's weights vs equal weights", rc, req)):
        V[tag] = verdict(None, None, a, b); v = V[tag]
        P_(f"  {tag:44s} better {v['better']} / worse {v['worse']}  median {v['median_diff']:+.4f}  p {v['p']:.2g} -> {v['verdict']}")
    open(OUT, "w").write("\n".join(out) + "\n")
    json.dump(dict(median_r={a: float(np.median(Rr[a])) for a in Rr}, verdicts=V, claude=dict(blend=rc.tolist(), engine=r0.tolist(), equal=req.tolist())),
              open(os.path.join(OUTD, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    {"build": build, "engine": engine, "score": score}[sys.argv[1]]()
