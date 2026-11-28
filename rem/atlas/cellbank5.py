"""More cell types in the memory bank: HepG2 (liver) and Jurkat (T cell) knockdowns join K562 and RPE1.

WHY. cellbank4 found that adding more of OTHER genes' experiments did not help the engine predict an unmeasured
K562 knockdown, but a measurement of the SAME knockdown in another cell type (RPE1) did: on the 980 TEST genes
with an RPE1 record the engine reached median r 0.350 vs 0.285 (post-hoc). The user asked for more cell types.

THE DATA. Nadig et al. 2025 (Nature Genetics), essential-gene CRISPRi Perturb-seq in HepG2 and Jurkat from the
same lab and reagents as Replogle 2022: figshare 29498366 (CC BY 4.0), tables of log2 fold change and p value
(rows = measured genes by Ensembl id, columns = knocked-down genes). A knockdown is stored if at least 20 measured
genes, not counting its own gene, have p < 0.001 (about 8 expected by chance): the stand-in for the energy-test
filter used for the Replogle screens. Each record is processed exactly as in cellbank4 (the screen's own mean
TRAIN-gene response removed, mapped onto screen A's genes by Ensembl id, projected on the same 128 components,
int8 codes + one scale). Other public screens (HCT116, HEK293T, iPSC, hESC) exist only as 2-40 GB files under a
non-commercial licence and are not added.

THE BANK. memory_bank/cell_v5 = cell_v4 (screens A, B, C, unchanged and in the same order) + D HepG2 + E Jurkat.

ARMS (cellbank4's engine and recipe exactly, seeds 0-2 averaged)
  E2  as cellbank4: snap over A TRAIN + B + C; partner and co-essential inputs; the gene's own RPE1 record.
  E3  snap over A TRAIN + B + C + D + E; the same partner and co-essential inputs; the gene's own records in
      RPE1, HepG2 and Jurkat as three inputs (each with a present / absent flag).
  As always, all screen-A TEST records and the query gene's own records are hidden from the snap memory.

GATES, PREDECLARED (per-gene Pearson r on TEST; paired sign test; BETTER = p < 0.05 and median gain >= +0.01,
                    WORSE = p < 0.05 and median gain <= -0.01)
  H2  blocking, data check: for TRAIN genes with records in both the new screen and K562-essential (B), the
      median r between the two decoded records must exceed 0.05 for HepG2 and for Jurkat (a mapping or sign
      error would give about 0 or below); the same pairs shuffled are reported for reference.
  G5  PRIMARY: E3 vs E2 on the TEST genes that have their own record in HepG2 or Jurkat.
  G6  E3 vs E2 on all 2,525 TEST genes.
  G7  descriptive: E3 vs E2 on TEST genes newly covered (a HepG2 or Jurkat record but no RPE1 record), and
      E3 by the number of other cell types holding the gene's record (0, 1, 2, 3).
  R   descriptive: E2 here vs cellbank4's E2 (same code and seeds; should match closely).
NOT: one filter for the new screens; log2 fold changes (Nadig) and normalised expression (Replogle) differ in
     scale, which the engine's per-screen value maps and input scaling must absorb; one memory design.

Run: python3 -P rem/atlas/cellbank5.py build | engine | score
"""

from __future__ import annotations
import importlib.util
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
_s = importlib.util.spec_from_file_location("cellbank4", os.path.join(HERE, "cellbank4.py"))
cb4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb4)
BANK = R("memory_bank", "cell_v5")
OUTD = R("outputs", "cellbank5")
CACHE = os.path.join(HERE, "_cache", "cellbank5")
OUT = os.path.join(HERE, "RESULTS_cellbank5.txt")
NADIG = os.path.join(HERE, "_cache", "nadig")
NEW = {"D": "HepG2Essential", "E": "JurkatEssential"}
NAME = {"D": "HepG2 (liver)", "E": "Jurkat (T cell)"}
ALL = ("A", "B", "C", "D", "E")
SEEDS, MIN_SIG, P_SIG, RULE = (0, 1, 2), 20, 1e-3, "=" * 97


def build():
    t0 = time.time()
    import h5py
    import pandas as pd
    G = cb4.notebook(); torch, np = G["torch"], G["np"]
    A, KN, KE, gidx, Vy, split = G["A"], G["KN"], G["KE"], G["gidx"], G["Vy"].cpu(), G["split"]
    old = cb4.Bank(cb4.BANK)
    meta = json.load(open(os.path.join(cb4.BANK, "records_meta.json")))
    ens_A = [x.decode() for x in h5py.File(G["P"]["gwps.h5ad"], "r")["var/gene_id"][()]]
    col_of_ens = {}
    for j, e in enumerate(ens_A):
        col_of_ens.setdefault(e, j)
    row_of = {g: j for j, g in enumerate(KN)}
    codes, scale = [old.codes], [old.scale]
    rep = dict(records={}, kept={}, h2={})
    for scr, stem in NEW.items():
        L = pd.read_csv(os.path.join(NADIG, f"{stem}_Log2FoldChange.csv.gz"), index_col=0)
        Pv = pd.read_csv(os.path.join(NADIG, f"{stem}_p.csv.gz"), index_col=0).reindex(index=L.index, columns=L.columns)
        rows = [i for i, e in enumerate(L.index) if e in col_of_ens]; cols_A = [col_of_ens[L.index[i]] for i in rows]
        own_row = {e: i for i, e in enumerate(L.index)}
        keep, nsig = [], []
        for c, g in enumerate(L.columns):
            if g not in row_of:
                continue
            p = Pv.iloc[:, c].to_numpy(dtype=float, copy=True)
            e = KE[row_of[g]]
            if e in own_row:
                p[own_row[e]] = np.nan                       # the knocked-down gene itself does not count
            k = int(np.nansum(p < P_SIG))
            if k >= MIN_SIG:
                keep.append(c); nsig.append(k)
        X = torch.tensor(L.iloc[rows, keep].to_numpy(dtype=np.float32).T)
        M = torch.full((len(keep), A["X"].shape[1]), float("nan")); M[:, cols_A] = X
        genes = [L.columns[c] for c in keep]
        for r_, g in enumerate(genes):
            if g in gidx:
                M[r_, gidx[g]] = float("nan")
        trk = [r_ for r_, g in enumerate(genes) if split(g) == "TRAIN"]
        td = torch.nanmean(M[trk], 0)
        Z = torch.nan_to_num(M - td) @ Vy
        s = Z.abs().max(1).values.clamp(min=1e-12)
        codes.append(torch.round(Z / s[:, None] * 127).to(torch.int8).numpy()); scale.append(s.numpy().astype(np.float32))
        meta["gene"] += genes; meta["ens"] += [KE[row_of[g]] or "" for g in genes]; meta["screen"] += [scr] * len(genes)
        meta["split"] += [split(g) for g in genes]; meta["ep"] += [float("nan")] * len(genes)
        meta.setdefault("n_sig", {})[scr] = nsig
        rep["records"][scr] = len(genes); rep["kept"][scr] = dict(perturbations=int(L.shape[1]), with_features=sum(g in row_of for g in L.columns),
                                                                 kept=len(keep), measured_genes_mapped=len(rows))
    os.makedirs(BANK, exist_ok=True)
    codes, scale = np.concatenate(codes), np.concatenate(scale)
    np.savez_compressed(os.path.join(BANK, "records.npz"), codes=codes, scale=scale, basis=old.basis)
    json.dump(meta, open(os.path.join(BANK, "records_meta.json"), "w"))
    json.dump(old.genes, open(os.path.join(BANK, "genes.json"), "w"))
    db = os.path.join(BANK, "index.sqlite")
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, gene TEXT, ensembl TEXT, screen TEXT, split TEXT, energy_p REAL)")
    con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", [(i, meta["gene"][i], meta["ens"][i], meta["screen"][i], meta["split"][i],
                                                                 None if meta["ep"][i] != meta["ep"][i] else meta["ep"][i]) for i in range(len(meta["gene"]))])
    con.execute("CREATE INDEX by_gene ON records(gene)"); con.commit(); con.close()
    open(os.path.join(BANK, "journal.jsonl"), "a").close()

    # H2: does each new screen agree with K562-essential on the same TRAIN knockdowns?
    bank = cb4.Bank(BANK); Vt = torch.tensor(bank.basis.astype(np.float32)).T
    for scr in NEW:
        pairs = [(d[scr], d["B"]) for g, d in bank.by_gene.items() if scr in d and "B" in d and split(g) == "TRAIN"]
        a = torch.tensor(bank.z([p for p, _ in pairs])) @ Vt; b = torch.tensor(bank.z([q for _, q in pairs])) @ Vt
        r = torch.nan_to_num(G["pearson_rows"](a, b)).numpy()
        rs = torch.nan_to_num(G["pearson_rows"](a, b[torch.randperm(len(b), generator=torch.Generator().manual_seed(1))])).numpy()
        rep["h2"][scr] = dict(n=len(pairs), median_r=float(np.median(r)), shuffled=float(np.median(rs)), passed=bool(np.median(r) > 0.05))
    rep["total_records"] = len(bank.gene); rep["bytes_bank"] = int(bank.codes.nbytes + bank.scale.nbytes + bank.basis.nbytes)
    rep["seconds"] = time.time() - t0
    json.dump(rep, open(os.path.join(BANK, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def own_inputs(G, bank, screens):
    torch = G["torch"]
    out = []
    for s in screens:
        M = torch.zeros(G["NPH"], 129)
        for j, g in enumerate(G["KN"]):
            i = bank.get(g, s)
            if i is not None:
                M[j, :128] = torch.tensor(bank.z([i])[0]); M[j, 128] = 1.0
        out.append(M)
    return out


def engine():
    t0 = time.time()
    rep = json.load(open(os.path.join(BANK, "build_report.json")))
    if not all(v["passed"] for v in rep["h2"].values()):
        print("H2 FAILED: not running the engine"); return
    G = cb4.notebook(); bank = cb4.Bank(BANK); torch = G["torch"]
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    Pt, Ct = cb4.inputs(G, bank, False)
    oC, oD, oE = own_inputs(G, bank, ("C", "D", "E"))
    X = dict(E2=cb4.scaled(G, [Pt, Ct, oC]), E3=cb4.scaled(G, [Pt, Ct, oC, oD, oE]))
    preds, info = {}, {}
    for arm, screens in (("E2", ("A", "B", "C")), ("E3", ALL)):
        cb4.SCREENS = screens                         # run_arm reads the memory's screens from this global
        outs = []
        for s in SEEDS:
            pz, best, _ = cb4.run_arm(G, bank, arm, s, X[arm])
            outs.append(pz); info.setdefault(arm, []).append(round(best, 4))
            print(f"  {arm} seed {s}: held-out TRAIN r {best:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
        preds[arm] = torch.stack(outs).mean(0)
    torch.save(preds, os.path.join(CACHE, "engine_preds.pt"))
    json.dump(dict(heldout_train_r=info), open(os.path.join(OUTD, "engine_info.json"), "w"), indent=1)
    print(f"engine arms done ({time.time() - t0:.0f}s)")


def score():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    G = cb4.notebook(); bank = cb4.Bank(BANK); torch, np = G["torch"], G["np"]
    from scipy.stats import binomtest
    A_te, Vy, pr, KN, rte = G["A_te"], G["Vy"].cpu(), G["pearson_rows"], G["KN"], G["rte"]
    preds = torch.load(os.path.join(CACHE, "engine_preds.pt"))
    Rr = {a: torch.nan_to_num(pr(preds[a] @ Vy.T, A_te)).numpy() for a in ("E2", "E3")}
    old = torch.load(os.path.join(cb4.CACHE, "engine_preds.pt"))
    r_old = torch.nan_to_num(pr(old["E2"] @ Vy.T, A_te)).numpy()
    rep = json.load(open(os.path.join(BANK, "build_report.json")))
    genes = [KN[int(rte[t])] for t in range(len(rte))]
    has = {s: np.array([bank.get(g, s) is not None for g in genes]) for s in ("C", "D", "E")}
    new_any = has["D"] | has["E"]; n_other = has["C"].astype(int) + has["D"] + has["E"]

    def verdict(m):
        d = Rr["E3"][m] - Rr["E2"][m]; w, l = int((d > 0).sum()), int((d < 0).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0; md = float(np.median(d)) if len(d) else 0.0
        v = "BETTER" if p < 0.05 and md >= 0.01 else "WORSE" if p < 0.05 and md <= -0.01 else "NO DIFFERENCE"
        return dict(n=int(m.sum()), better=w, worse=l, p=float(p), median_diff=md, E2=float(np.median(Rr["E2"][m])),
                    E3=float(np.median(Rr["E3"][m])), verdict=v)

    P_(RULE); P_("MORE CELL TYPES IN THE MEMORY BANK: HEPG2 AND JURKAT JOIN K562 AND RPE1"); P_(RULE)
    for s in NEW:
        k = rep["kept"][s]; h = rep["h2"][s]
        P_(f"  {NAME[s]}: {k['perturbations']:,} knockdowns, {k['with_features']:,} with features, {k['kept']:,} stored "
           f"(>= {MIN_SIG} genes at p < {P_SIG:g}); {k['measured_genes_mapped']:,} measured genes mapped")
        P_(f"     H2 agreement with K562-essential on {h['n']} TRAIN knockdowns: median r {h['median_r']:+.3f} "
           f"(shuffled {h['shuffled']:+.3f}) -> {'PASS' if h['passed'] else 'FAIL'}")
    P_(f"  bank: {rep['total_records']:,} records, {rep['bytes_bank'] / 1e6:.1f} MB")
    P_(f"  TEST genes with their own record in RPE1 {has['C'].sum()}, HepG2 {has['D'].sum()}, Jurkat {has['E'].sum()}; "
       f"in HepG2 or Jurkat {new_any.sum()}, of which without RPE1 {(new_any & ~has['C']).sum()}")
    P_(f"  R  E2 here {np.median(Rr['E2']):+.3f} vs cellbank4's E2 {np.median(r_old):+.3f} (all TEST, median r)")
    V = {"G5": verdict(new_any), "G6": verdict(np.ones(len(genes), bool)), "G7": verdict(new_any & ~has["C"])}
    for tag, lab in (("G5", "PRIMARY, genes with a HepG2 or Jurkat record"), ("G6", "all TEST genes"), ("G7", "newly covered (no RPE1 record), descr.")):
        v = V[tag]
        P_(f"  {tag} E3 vs E2, {lab:44s} n {v['n']:5d}: E3 {v['E3']:+.3f} vs E2 {v['E2']:+.3f}; better {v['better']} / worse {v['worse']}, "
           f"per-gene median {v['median_diff']:+.4f}, p {v['p']:.2g} -> {v['verdict'] if tag != 'G7' else '(descriptive)'}")
    for k in range(4):
        m = n_other == k
        if m.any():
            P_(f"     E3 on genes with their own record in {k} other cell type(s): n {m.sum():5d}, median r {np.median(Rr['E3'][m]):+.3f} (E2 {np.median(Rr['E2'][m]):+.3f})")
    open(OUT, "w").write("\n".join(out) + "\n")
    json.dump(dict(verdicts=V, median_r={a: float(np.median(Rr[a])) for a in Rr}), open(os.path.join(OUTD, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    {"build": build, "engine": engine, "score": score}[sys.argv[1]]()
