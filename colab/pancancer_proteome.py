"""PAN-CANCER CELL-LINE PROTEOMICS -- and the only question that decides whether cell-matching is worth it.

WHAT IS ALREADY IN HAND. `k562_protein.py` put one measured proteome into the model: PaxDb's
9606-iBAQ_K562_Geiger_2012, 7,031 proteins, 37.7% of the cell object's genes, and steeply abundance-biased
(2.2% of the lowest generic-abundance quintile against 74.5% of the highest). It answered "how wrong is the
generic `ppm` table for K562" (Spearman 0.655) but it is ONE cell line from ONE laboratory, so it cannot
answer the question the model actually faces every time it reasons about a cell line that is not K562:

    IS A CELL-MATCHED PROTEOME WORTH FETCHING, OR IS ANY HUMAN PROTEOME AS GOOD?

That question has a cost attached. Cell-matching means a separate proteome per cell line, a separate join, a
separate provenance, and a coverage hole that differs per line. If K562 protein levels predict RPE1, HCT116
and Jurkat behaviour as well as those lines' own protein levels do, the whole apparatus is waste and the
model should carry one proteome and say so.

THE SOURCE. ProCan-DepMapSanger (Goncalves et al., Cancer Cell 2022), released through Sanger Cell Model
Passports as Protein_matrix_averaged_20221214: 948 cancer cell lines x 8,457 proteins, DIA mass spectrometry,
one pipeline, one release, averaged log2 intensity. It is keyed by UNIPROT ACCESSION, which is a stronger key
than the gene symbols PaxDb's file offers, and it contains K-562, HCT-116 and Jurkat -- three of the five
lines this project has Perturb-seq for.

THE THREE TESTS, and what would falsify each.

  (a) DOES IT AGREE WITH THE MEASUREMENT ALREADY IN HAND? Spearman between the ProCan K-562 column and PaxDb
      iBAQ K562, on the proteins both quantify. Units differ (averaged log2 DIA intensity vs iBAQ), so rank
      is the only fair statistic -- the same reasoning `k562_protein.py` used. FALSIFIED IF the two
      independent K562 proteomes disagree at rank level: that would mean neither can be trusted as "the K562
      proteome" and the cell-matched arm below is measuring noise. The agreement number is unreadable on its
      own, so it is measured against 40 RANDOM ProCan lines correlated with the same PaxDb K562 vector: if a
      random cancer line agrees with K562 iBAQ nearly as well as the K-562 column does, then almost all of
      "protein abundance" is a species-level constant and cell-matching has very little left to buy.

  (b) DOES IT REACH THE GENES PaxDb MISSES, OR IS THE MISSINGNESS THE SAME? A second proteome is only worth
      carrying if it fills a hole. If ProCan detects the same abundant proteins and misses the same faint
      ones, the union is barely bigger than either and "not detected" stays exactly as uninformative as it
      was. Measured as coverage by generic-abundance quintile for each source separately and for the union,
      plus the DepMap dependency fraction of covered vs uncovered genes.

  (c) THE ONE THAT DECIDES IT. For each cell line with both a ProCan column and a Perturb-seq screen, does
      that line's OWN protein abundance predict which knockdowns produce a large transcriptional response IN
      THAT LINE, better than K562 protein used everywhere, better than the generic `ppm` table, and better
      than the protein column of another cell line drawn at random? Perturbation strength -- mean |robust z|
      over measured genes -- is the quantity this project already uses as a matching covariate and as a
      screen-prioritisation target, so it is a decision the model actually makes.

      IT SPLITS INTO TWO QUESTIONS THAT LOOK LIKE ONE, AND CONFLATING THEM WOULD MISREAD THE ANSWER.

        c1  WHICH ABSOLUTE ABUNDANCE TABLE IS BEST? A raw column compares proteins WITHIN a sample, and DIA
            intensity is a poor instrument for that: peptide ionisation efficiency varies by orders of
            magnitude between proteins, which is exactly what iBAQ normalises away and averaged DIA
            intensity does not. So a head-to-head of raw columns is partly a unit contest, not a matching
            contest, and it is reported as such.

        c2  DOES THE CELL-SPECIFIC PART CARRY ANYTHING? Standardising each protein ACROSS the 948 lines
            gives a panel-z whose cross-protein abundance information is exactly zero by construction --
            every protein has mean 0 and sd 1 over the panel. What survives is only "this protein is
            unusually high in THIS line". That is the whole of what cell-matching can ever buy, measured
            with the unit confound removed, and it is also measured INCREMENTALLY: after rank-regressing
            the target on the best absolute table already in the model, does the panel-z still correlate
            with the residual?

      ALL ARMS ARE SCORED ON AN IDENTICAL GENE SET so the comparison is about the values and not about
      coverage. Every control is PAIRED and drawn 40 times: for each draw the matched arm is recomputed on
      the genes that draw also detects, so a control line with a thinner proteome cannot lose on gene count.
      Three control pools, increasingly hard -- any line, a line matched on how many proteins it quantifies
      (which is a proxy for how well it was measured), and a line from the same tissue.

      FALSIFIED IF the matched arm sits inside the random-line distribution. That is the live possibility and
      it is not a small one: protein abundance is dominated by a housekeeping axis shared by every human
      cell, and a knockdown's disruptiveness is dominated by essentiality, which is also largely shared.

WHAT THIS CANNOT DO. RPE1 is not a cancer line and HepG2 was not profiled by ProCan, so two of the five
Perturb-seq lines have no matched column and are reported as absent rather than substituted. Each cell line's
Perturb-seq comes from a different laboratory (K562 and RPE1 Replogle, HCT116 Xaira X-Atlas/Orion, Jurkat and
HepG2 Nadig), so the head-to-head is run WITHIN a cell line -- one responsiveness vector, one proteomics
pipeline, one gene set -- and absolute Spearman values are never compared ACROSS lines. Mixing two labs on
one axis has already burned this project once.
"""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

# ---- ONE release, ONE matrix, asserted. Two normalisations ship in the same zip (averaged and
# averaged-zscore); the z-scored one centres every protein across cell lines and therefore destroys the
# within-sample abundance ordering that question (a) compares against PaxDb. Only the averaged matrix is
# used, and nothing in this module ever reads the other.
PROCAN_URL = "https://cog.sanger.ac.uk/cmp/download/Proteomics_20221214.zip"
PROCAN_MEMBER = "Protein_matrix_averaged_20221214.tsv"
MODEL_URL = "https://cog.sanger.ac.uk/cmp/download/model_list_20221102.csv"   # contemporaneous annotation
PROCAN_NPZ = SP / "procan_matrix.npz"
MODEL_CSV = SP / "procan_model_list_20221102.csv"
PROCAN_SHAPE = (948, 8457)
SOURCE = ("ProCan-DepMapSanger Protein_matrix_averaged_20221214 (Goncalves et al., Cancer Cell 2022; "
          "Sanger Cell Model Passports), DIA-MS averaged log2 intensity")

PAXDB = SP / "prot" / "9606-iBAQ_K562_Geiger_2012_uniprot.txt"
NET = SP / "cell_complete.json.gz"
STRENGTH = SP / "pancancer_strength.npz"
LAYER = SP / "cell_pancancer_proteome.json.gz"

# perturbation screens: (cell line, ProCan model_name or None, file, loader, provenance)
SCREENS = [
    ("K562",   "K-562",   "gwps.h5ad",           "gwps",  "Replogle 2022 genome-scale CRISPRi Perturb-seq"),
    ("RPE1",   None,      "rpe1_pseudobulk.npz", "npz",   "Replogle 2022 CRISPRi Perturb-seq (RPE1 arm)"),
    ("HCT116", "HCT-116", "hct116.h5ad",         "xaira", "Xaira X-Atlas/Orion CRISPRi Perturb-seq"),
    ("Jurkat", "Jurkat",  "pb_jurkat.npz",       "npz",   "Nadig 2025 CRISPRi Perturb-seq"),
    ("HepG2",  None,      "pb_hepg2.npz",        "npz",   "Nadig 2025 CRISPRi Perturb-seq"),
]
N_DRAWS = 40           # matched controls drawn many times; one draw is never the result
COL_CHUNK = 2000


# ---------------------------------------------------------------------------- fetch / reduce
def fetch_procan(report):
    """Download 112 MB, keep a 20 MB npz, delete the raw. Disk here is ~4 GB shared with another workflow."""
    if PROCAN_NPZ.exists():
        report(f"  ProCan matrix from cache: {PROCAN_NPZ}")
        return
    d = SP / "procan"
    d.mkdir(parents=True, exist_ok=True)
    z = d / "Proteomics_20221214.zip"
    report(f"  fetching {PROCAN_URL}")
    subprocess.run(["curl", "-sS", "-L", "--max-time", "1200", "-o", str(z), PROCAN_URL], check=True)
    import zipfile
    with zipfile.ZipFile(z) as zf:
        zf.extract(PROCAN_MEMBER, str(d))
    z.unlink()
    p = d / PROCAN_MEMBER
    rows, names, mids = [], [], []
    with open(p) as fh:
        up = fh.readline().rstrip("\n").split("\t")[2:]
        sy = fh.readline().rstrip("\n").split("\t")[2:]
        fh.readline()                                   # 'model_name\tmodel_id'
        for line in fh:
            pr = line.rstrip("\n").split("\t")
            names.append(pr[0])
            mids.append(pr[1])
            rows.append([np.float32(x) if x else np.float32("nan") for x in pr[2:]])
    M = np.array(rows, dtype=np.float32)
    np.savez_compressed(PROCAN_NPZ, M=M, uniprot=np.array(up), symbol=np.array(sy),
                        model_name=np.array(names), model_id=np.array(mids))
    p.unlink()
    d.rmdir()
    report(f"  reduced to {PROCAN_NPZ} ({PROCAN_NPZ.stat().st_size/1e6:.1f} MB); raw deleted")
    if not MODEL_CSV.exists():
        subprocess.run(["curl", "-sS", "-L", "--max-time", "600", "-o", str(MODEL_CSV), MODEL_URL],
                       check=True)


def read_paxdb(path):
    """gene symbol -> iBAQ. Same reader as k562_protein.py, so the two layers key identically."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or not p[0]:
                continue
            try:
                v = float(p[2])
            except ValueError:
                continue
            if v > 0:
                out[p[0]] = max(out.get(p[0], 0.0), v)
    return out


# ---------------------------------------------------------------------------- responsiveness
def _strength_from_matrix(M):
    """mean |robust z| per ROW (perturbation), z taken per COLUMN (measured gene).

    The per-column robust z is the same normalisation `causal_reg.py` uses and for the same reason: some
    genes move under any perturbation, and a raw fold change is dominated by them. Chunked over columns
    because the HCT116 matrix is 17,768 x 16,380 and holding two copies of it is avoidable.
    """
    n, G = M.shape
    S = np.zeros(n)
    C = np.zeros(n)
    for c0 in range(0, G, COL_CHUNK):
        B = np.asarray(M[:, c0:c0 + COL_CHUNK], dtype=np.float64)
        med = np.nanmedian(B, axis=0)
        mad = np.nanmedian(np.abs(B - med), axis=0) * 1.4826
        mad = np.where(mad < 1e-9, np.nan, mad)
        Z = np.abs((B - med) / mad)
        ok = np.isfinite(Z)
        S += np.where(ok, Z, 0.0).sum(1)
        C += ok.sum(1)
    return np.where(C > 0, S / np.maximum(C, 1), np.nan), C


def load_screen(kind, path, report):
    """-> (perturbed gene symbols, [perturbation x measured gene] response matrix)"""
    if kind == "npz":
        z = np.load(path, allow_pickle=True)
        return np.array([str(x) for x in z["perts"]]), z["lfc"]
    import h5py
    if kind == "gwps":
        with h5py.File(path, "r") as f:
            X = f["X"][:]
            pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        return np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert]), X
    with h5py.File(path, "r") as f:                       # xaira: obs idx 'g_<SYM>_hct116'
        idx = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["idx"][:]]
        X = f["X"][:]
    sym = [i[2:].rsplit("_", 1)[0] if i.startswith("g_") else i for i in idx]
    return np.array(sym), X


def build_strength(report):
    """Cache {cell: (symbols, mean|z|)} for every screen present. Slow part; run once."""
    if STRENGTH.exists():
        z = np.load(STRENGTH, allow_pickle=True)
        report(f"  perturbation strength from cache: {STRENGTH}")
        return {k: (np.array([str(x) for x in z[k + "_sym"]]), z[k + "_str"], z[k + "_n"])
                for k in [c for c, *_ in SCREENS] if k + "_sym" in z.files}
    out, store = {}, {}
    for cell, _mn, fn, kind, prov in SCREENS:
        p = SP / fn
        if not p.exists():
            report(f"    {cell:8s} ABSENT ({fn}) -- skipped")
            continue
        sym, M = load_screen(kind, p, report)
        s, c = _strength_from_matrix(M)
        del M
        out[cell] = (sym, s, c)
        store[cell + "_sym"] = sym
        store[cell + "_str"] = s
        store[cell + "_n"] = c
        report(f"    {cell:8s} {len(sym):6,d} perturbations x mean|z| computed   [{prov}]")
    np.savez_compressed(STRENGTH, **store)
    return out


# ---------------------------------------------------------------------------- main
def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("PAN-CANCER CELL-LINE PROTEOMICS -- does cell-matching the proteome buy anything?")
    report("=" * 100)

    from entity_registry import load as load_reg
    REG = load_reg()

    fetch_procan(report)
    z = np.load(PROCAN_NPZ, allow_pickle=True)
    M, up, sy = z["M"], np.array([str(x) for x in z["uniprot"]]), np.array([str(x) for x in z["symbol"]])
    lines = np.array([str(x) for x in z["model_name"]])
    mids = np.array([str(x) for x in z["model_id"]])
    # PIN THE MATRIX. One release, one normalisation, asserted -- not "whatever the download gave".
    assert M.shape == PROCAN_SHAPE, f"ProCan matrix is {M.shape}, expected {PROCAN_SHAPE}"
    assert len(up) == M.shape[1] and len(lines) == M.shape[0]
    report(f"  {SOURCE}")
    report(f"  {M.shape[0]:,} cell lines x {M.shape[1]:,} proteins, "
           f"{100*np.isnan(M).mean():.1f}% not quantified")

    # ---- registry join on UNIPROT ACCESSION, not symbol ----
    _e, st_u = REG.join("gene", "uniprot", list(up), 0.95, "ProCan proteins by UniProt accession")
    eids = np.array([REG.resolve("gene", "uniprot", u) or "" for u in up])
    report(f"  registry join gene/uniprot: {st_u['joined']:,}/{st_u['of']:,} ({st_u['rate']:.1%}) "
           f"-- accession, not symbol")
    n_collide = len(eids[eids != ""]) - len(set(eids[eids != ""]))
    report(f"  {n_collide:,} accessions collapse onto an already-used gene (isoform/paralogue duplicates); "
           f"the maximum value is kept, as in k562_protein.py")

    tissue = {}
    if MODEL_CSV.exists():
        import csv
        with open(MODEL_CSV) as fh:
            for r in csv.DictReader(fh):
                tissue[r["model_id"]] = r.get("tissue") or "?"
    tis = np.array([tissue.get(m, "?") for m in mids])

    def column(line_name):
        """cell line -> {eid: log2 intensity}, max over accessions mapping to one gene."""
        i = int(np.where(lines == line_name)[0][0])
        v = M[i]
        d = {}
        for e, x in zip(eids, v):
            if e and np.isfinite(x):
                d[e] = max(d.get(e, -np.inf), float(x))
        return d, i

    # ---- the other abundance sources, all keyed by entity id ----
    pax_sym = read_paxdb(PAXDB)
    _e, st_p = REG.join("gene", "symbol", list(pax_sym), 0.75, "PaxDb K562 symbols")
    pax = {}
    for s, v in pax_sym.items():
        e = REG.resolve("gene", "symbol", s)
        if e:
            pax[e] = max(pax.get(e, 0.0), v)
    report(f"  PaxDb iBAQ K562 (already in the model): {len(pax_sym):,} proteins, "
           f"join {st_p['rate']:.1%} -> {len(pax):,} genes")

    d = json.load(gzip.open(NET, "rt"))
    cell_names = [g["name"] for g in d["genes"]]
    ppm_raw = d.get("ppm", {})
    dep = {g["name"]: g.get("dep_frac") for g in d["genes"]}
    del d
    # `ppm` IS KEYED BY GENE INDEX. Looking it up by symbol returns None for every gene and is the bug
    # that once zeroed an abundance control; the registry's cell_index is the only correct route.
    cidx = json.load(gzip.open(SP / "entity_registry.json.gz", "rt"))["cell_index"]
    generic = {}
    for k, v in ppm_raw.items():
        e = cidx.get(str(k))
        if e:
            try:
                f = float(v or 0)
            except (TypeError, ValueError):
                continue
            if f > 0:
                generic[e] = max(generic.get(e, 0.0), f)
    report(f"  generic `ppm` via cell_index: {len(generic):,} genes with a positive value "
           f"(NOT looked up by symbol)")
    cell_eid = {cidx.get(str(i)): n for i, n in enumerate(cell_names)}

    R = {"source": SOURCE, "procan_shape": list(M.shape), "registry_joins": {"procan_uniprot": st_u,
                                                                             "paxdb_symbol": st_p}}

    # ======================================================================== (a) agreement with PaxDb
    k562_col, k562_i = column("K-562")
    both = sorted(set(k562_col) & set(pax))
    a = np.array([k562_col[e] for e in both])
    b = np.log10(np.array([pax[e] for e in both]))
    rho = float(stats.spearmanr(a, b)[0])
    rp = float(np.corrcoef(a, b)[0, 1])
    report(f"\n  (a) TWO INDEPENDENT K562 PROTEOMES, {len(both):,} genes quantified by both")
    report(f"      ProCan detects {len(k562_col):,} genes in K-562; PaxDb/Geiger {len(pax):,}")
    report(f"      Spearman {rho:.4f} (rank R2 {rho**2:.4f})  <- the fair statistic: DIA log2 vs iBAQ")
    report(f"      Pearson  {rp:.4f}  <- unit-sensitive, reported for completeness")
    dv = stats.rankdata(b) / len(b) - stats.rankdata(a) / len(a)
    o = np.argsort(dv)
    nm = lambda e: cell_eid.get(e) or (REG.entities.get(e, {}) or {}).get("symbol") or e
    report(f"      {100*np.mean(np.abs(dv) > 0.25):.1f}% of genes move more than a quarter of the range "
           f"between the two")
    report(f"      ProCan ranks HIGHEST relative to PaxDb: {[nm(both[i]) for i in o[:6]]}")
    report(f"      PaxDb ranks HIGHEST relative to ProCan: {[nm(both[i]) for i in o[-6:]][::-1]}")
    # THE CONTROL WITHOUT WHICH 0.62 IS UNREADABLE. If a randomly chosen cancer line agrees with the K562
    # iBAQ vector nearly as well as the K-562 column does, then the shared human-proteome axis is doing
    # almost all the work and there is very little cell-specific signal for anything downstream to use.
    rng0 = np.random.default_rng(3)
    pool0 = [j for j in range(M.shape[0]) if j != k562_i]
    rnd_rho = []
    for j in rng0.choice(pool0, size=N_DRAWS, replace=False):
        cj = {}
        for e, x in zip(eids, M[j]):
            if e and np.isfinite(x):
                cj[e] = max(cj.get(e, -np.inf), float(x))
        sub = [e for e in both if e in cj]
        if len(sub) > 500:
            rnd_rho.append(float(stats.spearmanr([cj[e] for e in sub],
                                                 [np.log10(pax[e]) for e in sub])[0]))
    rnd_rho = np.array(rnd_rho)
    report(f"      CONTROL: {len(rnd_rho)} RANDOM ProCan lines vs the same PaxDb K562 vector -> "
           f"{rnd_rho.mean():.4f} +- {rnd_rho.std():.4f} (range {rnd_rho.min():.4f}..{rnd_rho.max():.4f})")
    report(f"      the K-562 column beats {100*(rho > rnd_rho).mean():.0f}% of random lines; the "
           f"cell-specific margin is only {rho - rnd_rho.mean():+.4f} of the {rho:.4f} total")
    R["agreement_with_paxdb"] = {"n_shared": len(both), "spearman": rho, "pearson": rp,
                                 "n_procan_k562": len(k562_col), "n_paxdb": len(pax),
                                 "frac_moving_gt_quarter": float(np.mean(np.abs(dv) > 0.25)),
                                 "random_line_control": {
                                     "n_draws": int(len(rnd_rho)), "mean": float(rnd_rho.mean()),
                                     "sd": float(rnd_rho.std()), "max": float(rnd_rho.max()),
                                     "frac_beaten_by_k562_column": float((rho > rnd_rho).mean()),
                                     "cell_specific_margin": float(rho - rnd_rho.mean())}}

    # ======================================================================== (b) coverage / missingness
    report(f"\n  (b) COVERAGE: does ProCan reach the genes PaxDb misses?")
    ceids = [cidx.get(str(i)) for i in range(len(cell_names))]
    ceids = [e for e in ceids if e]
    cp = np.array([e in k562_col for e in ceids])
    cx = np.array([e in pax for e in ceids])
    gv = np.array([generic.get(e, 0.0) for e in ceids])
    lg = np.log10(1.0 + gv)
    q = np.digitize(lg, np.quantile(lg[lg > 0], [.2, .4, .6, .8]))
    report(f"      of {len(ceids):,} cell-object genes: PaxDb {cx.sum():,} ({100*cx.mean():.1f}%), "
           f"ProCan K-562 {cp.sum():,} ({100*cp.mean():.1f}%), "
           f"union {int((cp | cx).sum()):,} ({100*(cp | cx).mean():.1f}%), "
           f"both {int((cp & cx).sum()):,}")
    report(f"      ProCan-only {int((cp & ~cx).sum()):,}   PaxDb-only {int((cx & ~cp).sum()):,}")
    report(f"      {'generic-abundance quintile':28s} {'n':>7s} {'PaxDb':>8s} {'ProCan':>8s} {'union':>8s}")
    qtab = []
    for k in sorted(set(q.tolist())):
        m = q == k
        report(f"      {('q'+str(k)):28s} {int(m.sum()):7,d} {cx[m].mean():8.4f} {cp[m].mean():8.4f} "
               f"{(cp | cx)[m].mean():8.4f}")
        qtab.append({"q": int(k), "n": int(m.sum()), "paxdb": float(cx[m].mean()),
                     "procan": float(cp[m].mean()), "union": float((cp | cx)[m].mean())})
    dpa = np.array([float(dep.get(cell_eid.get(e)) or 0) for e in ceids])
    mw = lambda mask: float(stats.mannwhitneyu(dpa[mask], dpa[~mask])[1])
    report(f"      mean dep_frac: PaxDb-covered {dpa[cx].mean():.4f} vs not {dpa[~cx].mean():.4f} "
           f"(p {mw(cx):.3g});  ProCan-covered {dpa[cp].mean():.4f} vs not {dpa[~cp].mean():.4f} "
           f"(p {mw(cp):.3g})")
    # is the ProCan-only set faint, or is it just a different slice of the same abundant genes?
    lo = q <= 1
    report(f"      in the two LOWEST generic-abundance quintiles ({int(lo.sum()):,} genes): "
           f"PaxDb {cx[lo].mean():.3%}, ProCan {cp[lo].mean():.3%}, union {(cp | cx)[lo].mean():.3%}")
    R["coverage"] = {"n_cell_genes": len(ceids), "paxdb": int(cx.sum()), "procan_k562": int(cp.sum()),
                     "union": int((cp | cx).sum()), "both": int((cp & cx).sum()),
                     "procan_only": int((cp & ~cx).sum()), "paxdb_only": int((cx & ~cp).sum()),
                     "by_generic_quintile": qtab,
                     "low_two_quintiles": {"n": int(lo.sum()), "paxdb": float(cx[lo].mean()),
                                           "procan": float(cp[lo].mean()),
                                           "union": float((cp | cx)[lo].mean())},
                     "dep_frac_covered_vs_not": {"paxdb": [float(dpa[cx].mean()), float(dpa[~cx].mean())],
                                                 "procan": [float(dpa[cp].mean()), float(dpa[~cp].mean())]}}

    # ======================================================================== (c) does cell-matching pay?
    report(f"\n  (c) DOES CELL-MATCHING THE PROTEOME PAY?")
    report(f"      target = perturbation strength (mean |robust z| over measured genes) in that line")
    S = build_strength(report)
    n_det = np.isfinite(M).sum(1)
    # PANEL-Z: standardise each protein ACROSS the 948 lines. By construction this removes ALL cross-protein
    # abundance information -- every protein is mean 0, sd 1 over the panel -- so what is left is purely
    # "unusually high in THIS line". It is the only arm in which cell-matching is the sole variable.
    with np.errstate(invalid="ignore"):
        pan_mu = np.nanmean(M, axis=0)
        pan_sd = np.nanstd(M, axis=0)
    pan_ok = np.isfinite(pan_mu) & (pan_sd > 1e-6) & (np.isfinite(M).sum(0) >= 100)
    report(f"      panel-z defined for {int(pan_ok.sum()):,}/{M.shape[1]:,} proteins "
           f"(quantified in >= 100 lines)")
    per_line = {}

    def line_col(j, panel=False):
        """cell line index -> {eid: value}; panel=True gives the across-panel z, absolute info removed."""
        v = M[j]
        d = {}
        for k, (e, x) in enumerate(zip(eids, v)):
            if not e or not np.isfinite(x):
                continue
            if panel:
                if not pan_ok[k]:
                    continue
                x = (x - pan_mu[k]) / pan_sd[k]
            d[e] = max(d.get(e, -np.inf), float(x))
        return d

    # ---- THE CHEAPER ALTERNATIVE THAT HAS TO BE RULED OUT. If cell-matched mRNA does the same job, the
    # pan-cancer proteome adds nothing this project did not already have on disk: DepMap 24Q2
    # OmicsExpressionProteinCodingGenesTPMLogp1, md5-asserted, 1,066 lines. Built exactly parallel to the
    # protein panel-z -- standardise each gene across the panel -- so the two are the same construction on
    # two molecules and the comparison is about the molecule, not the method. The ProCan <-> DepMap link is
    # the Cell Model Passports BROAD_ID cross-reference, not a cell-line NAME match.
    EXPR_NPZ = SP / "depmap_expr.npz"
    MD5_EXPR = "f00c434db50d811544ac4d047003979f"
    rna = None
    if EXPR_NPZ.exists():
        ez = np.load(EXPR_NPZ, allow_pickle=True)
        got_md5 = str(ez["md5"][0])
        assert got_md5 == MD5_EXPR, f"depmap_expr.npz md5 {got_md5} != DepMap 24Q2 {MD5_EXPR}"
        E = ez["E"]
        eg = [str(x) for x in ez["genes"]]
        elines = np.array([str(x) for x in ez["lines"]])
        entrez = [g.split("(")[-1].rstrip(")") if "(" in g else "" for g in eg]
        _e, st_r = REG.join("gene", "entrez", entrez, 0.90, "DepMap expression genes by Entrez")
        e_eid = np.array([REG.resolve("gene", "entrez", x) or "" for x in entrez])
        emu, esd = np.nanmean(E, axis=0), np.nanstd(E, axis=0)
        e_ok = np.isfinite(emu) & (esd > 1e-6)
        broad = {}
        import csv as _csv
        with open(MODEL_CSV) as fh:
            for r in _csv.DictReader(fh):
                if r.get("BROAD_ID"):
                    broad[r["model_name"]] = r["BROAD_ID"]
        eidx = {l: i for i, l in enumerate(elines)}
        rna = {"E": E, "eid": e_eid, "mu": emu, "sd": esd, "ok": e_ok, "idx": eidx, "broad": broad,
               "lines": elines}
        report(f"      DepMap 24Q2 expression (md5 asserted): {E.shape[0]:,} lines x {E.shape[1]:,} genes, "
               f"Entrez join {st_r['rate']:.1%}")
        R["registry_joins"]["depmap_expr_entrez"] = st_r
    else:
        report(f"      DepMap expression absent ({EXPR_NPZ}) -- the mRNA alternative cannot be ruled out")

    def rna_col(j):
        d = {}
        for k, (e, x) in enumerate(zip(rna["eid"], rna["E"][j])):
            if e and rna["ok"][k] and np.isfinite(x):
                d[e] = max(d.get(e, -np.inf), float((x - rna["mu"][k]) / rna["sd"][k]))
        return d

    for cell, mn, fn, _k, prov in SCREENS:
        if cell not in S:
            continue
        sym, strength, ncol = S[cell]
        # perturbed gene -> entity id (declared join, raises if the key is wrong)
        _e, st_s = REG.join("gene", "symbol", list(sym), 0.70, f"{cell} perturbed genes")
        pe = [REG.resolve("gene", "symbol", s) for s in sym]
        tgt = {}
        for e, v in zip(pe, strength):
            if e and np.isfinite(v):
                tgt[e] = max(tgt.get(e, -np.inf), float(v))
        if mn is None:
            report(f"\n    {cell:8s} NO ProCan column (not in the 948 profiled lines) -- "
                   f"{len(tgt):,} perturbed genes, cannot be cell-matched")
            per_line[cell] = {"procan_column": None, "n_perturbed_joined": len(tgt),
                              "perturbseq_source": prov,
                              "note": "absent from ProCan; not substituted with another line"}
            continue
        col, li = column(mn)
        pcol = line_col(li, panel=True)
        # ONE gene set for every arm: perturbed AND quantified in this line AND in K-562 AND generic > 0
        genes = sorted(set(tgt) & set(col) & set(k562_col) & set(generic))
        y = np.array([tgt[e] for e in genes])
        arms = {"cell_matched": np.array([col[e] for e in genes]),
                "k562_procan": np.array([k562_col[e] for e in genes]),
                "generic_ppm": np.log10(np.array([generic[e] for e in genes])),
                "paxdb_k562": np.array([np.log10(pax[e]) if e in pax else np.nan for e in genes]),
                "cellspecific_panelz": np.array([pcol.get(e, np.nan) for e in genes])}
        report(f"\n    {cell:8s} ProCan '{mn}' ({tis[li]}), {n_det[li]:,} proteins quantified   "
               f"[{prov}]")
        report(f"      perturbed genes joined {st_s['rate']:.1%}; common gene set for every arm: "
               f"{len(genes):,}")
        report(f"      c1 ABSOLUTE TABLES -- partly a unit contest (iBAQ is built for cross-protein "
               f"comparison, DIA intensity is not)")
        res = {}
        for k, x in arms.items():
            f = np.isfinite(x) & np.isfinite(y)
            r = float(stats.spearmanr(x[f], y[f])[0])
            p = float(stats.spearmanr(x[f], y[f])[1])
            res[k] = {"spearman": r, "p": p, "n": int(f.sum())}
            if k == "cellspecific_panelz":
                report(f"      c2 CELL-SPECIFIC ONLY -- cross-protein abundance removed by construction")
            report(f"         {k:20s} rho {r:+.4f}  p {p:.3g}  n {int(f.sum()):,}")

        # ---- INCREMENT over the abundance table already in the model. Rank-regress the target on PaxDb
        # iBAQ K562, then ask whether this line's panel-z still explains the residual. This is the
        # cell-matching question with the absolute-abundance axis held fixed.
        def increment(pz):
            f = np.isfinite(arms["paxdb_k562"]) & np.isfinite(y) & np.isfinite(pz)
            if f.sum() < 100:
                return np.nan, np.nan, int(f.sum())
            ry = stats.rankdata(y[f])
            rx = stats.rankdata(arms["paxdb_k562"][f])
            resid = ry - np.polyval(np.polyfit(rx, ry, 1), rx)
            r, p = stats.spearmanr(pz[f], resid)
            return float(r), float(p), int(f.sum())

        ir, ip, inn = increment(arms["cellspecific_panelz"])
        res["increment_over_paxdb"] = {"spearman": ir, "p": ip, "n": inn}
        report(f"      c3 INCREMENT: cell-specific panel-z vs the residual after PaxDb iBAQ "
               f"rho {ir:+.4f} p {ip:.3g} n {inn:,}")

        # ---- PAIRED controls, 40 draws each. For every draw the matched arm is recomputed on the genes
        # that draw also quantifies, so a thinner control proteome cannot lose on gene count alone.
        def paired(pool, label, panel):
            pool = [j for j in pool if j != li]
            if len(pool) < 6:
                report(f"      {label:22s} only {len(pool)} lines -- not enough for a control")
                return None
            draws = rng.choice(pool, size=min(N_DRAWS, len(pool)), replace=len(pool) < N_DRAWS)
            base = pcol if panel else col
            ref = arms["cellspecific_panelz"] if panel else arms["cell_matched"]
            dm, dc, dn, dp, di = [], [], [], [], []
            for j in draws:
                cj = line_col(j, panel=panel)
                sub = [i for i, e in enumerate(genes)
                       if e in cj and e in base and np.isfinite(ref[i])]
                if len(sub) < 100:
                    continue
                ys = y[sub]
                dm.append(float(stats.spearmanr(ref[sub], ys)[0]))
                cv = np.array([cj[genes[i]] for i in sub])
                dc.append(float(stats.spearmanr(cv, ys)[0]))
                dp.append(float(stats.spearmanr(cv, ys)[1]))
                dn.append(len(sub))
                if panel:
                    full = np.array([cj.get(e, np.nan) for e in genes])
                    di.append(increment(full)[0])
            dm, dc, dn = np.array(dm), np.array(dc), np.array(dn)
            out = {"label": label, "n_draws": int(len(dm)),
                   "matched_mean": float(dm.mean()), "control_mean": float(dc.mean()),
                   "control_sd": float(dc.std()), "control_min": float(dc.min()),
                   "control_max": float(dc.max()),
                   "delta_mean": float((dm - dc).mean()), "delta_sd": float((dm - dc).std()),
                   "frac_draws_matched_better": float((dm > dc).mean()),
                   "frac_draws_control_p_lt_05": float((np.array(dp) < 0.05).mean()),
                   "residual_imbalance": {
                       "mean_genes_per_draw": float(dn.mean()),
                       "genes_available_matched": int(np.isfinite(ref).sum()),
                       "mean_proteins_quantified_control": float(n_det[draws].mean()),
                       "proteins_quantified_target": int(n_det[li]),
                       "mean_abs_gap_in_proteins_quantified": float(
                           np.abs(n_det[draws].astype(float) - n_det[li]).mean()),
                       "n_same_tissue_in_pool": int(sum(1 for j in draws if tis[j] == tis[li]))}}
            if di:
                out["control_increment_mean"] = float(np.nanmean(di))
                out["frac_draws_matched_increment_better"] = float(np.mean(np.array(di) < ir))
            report(f"      {label:22s} matched {dm.mean():+.4f} vs control {dc.mean():+.4f} "
                   f"(sd {dc.std():.4f}, {dc.min():+.4f}..{dc.max():+.4f})  delta {(dm-dc).mean():+.4f}  "
                   f"matched wins {100*(dm>dc).mean():.0f}% of {len(dm)}")
            report(f"      {'':22s} imbalance: control quantifies {n_det[draws].mean():,.0f} proteins vs "
                   f"{n_det[li]:,} (mean |gap| "
                   f"{np.abs(n_det[draws].astype(float)-n_det[li]).mean():,.0f}); "
                   f"{dn.mean():,.0f} genes/draw; {out['residual_imbalance']['n_same_tissue_in_pool']} of "
                   f"{len(draws)} draws share the target's tissue; "
                   f"{100*out['frac_draws_control_p_lt_05']:.0f}% of control draws reach p<0.05 on their own")
            if di:
                report(f"      {'':22s} increment over PaxDb: matched {ir:+.4f} vs control "
                       f"{np.nanmean(di):+.4f}, matched better in "
                       f"{100*np.mean(np.array(di) < ir):.0f}% of draws")
            return out

        allp = list(range(M.shape[0]))
        near = list(np.argsort(np.abs(n_det - n_det[li]))[:120])   # matched on how well the line was measured
        same = [j for j in allp if tis[j] == tis[li]]
        res["controls_absolute"] = {
            "random_line": paired(allp, "abs: random line", False),
            "detection_matched": paired(near, "abs: detection-matched", False),
            "same_tissue": paired(same, "abs: same tissue", False)}
        res["controls_cellspecific"] = {
            "random_line": paired(allp, "cellspec: random line", True),
            "detection_matched": paired(near, "cellspec: detection-matched", True),
            "same_tissue": paired(same, "cellspec: same tissue", True)}

        # ---- shuffle: significance only, never an effect size ----
        sh, shp = [], []
        for s_ in range(200):
            r2 = np.random.default_rng(1000 + s_)
            perm = r2.permutation(len(y))
            sh.append(float(stats.spearmanr(arms["cell_matched"], y[perm])[0]))
            f = np.isfinite(arms["cellspecific_panelz"])
            shp.append(float(stats.spearmanr(arms["cellspecific_panelz"][f], y[perm][f])[0]))
        sh, shp = np.array(sh), np.array(shp)
        res["shuffled"] = {
            "abs_mean": float(sh.mean()), "abs_sd": float(sh.std()),
            "abs_p_emp": float((np.abs(sh) >= abs(res["cell_matched"]["spearman"])).mean()),
            "panelz_mean": float(shp.mean()), "panelz_sd": float(shp.std()),
            "panelz_p_emp": float((np.abs(shp) >= abs(res["cellspecific_panelz"]["spearman"])).mean())}
        report(f"      {'shuffled target':22s} abs {sh.mean():+.4f}+-{sh.std():.4f} "
               f"(emp p {res['shuffled']['abs_p_emp']:.3g}); panel-z {shp.mean():+.4f}+-{shp.std():.4f} "
               f"(emp p {res['shuffled']['panelz_p_emp']:.3g})  -- significance only; the effect sizes are "
               f"the raw {res['cell_matched']['spearman']:+.4f} and {res['cellspecific_panelz']['spearman']:+.4f}")
        res["n_genes"] = len(genes)
        res["procan_column"] = mn
        res["tissue"] = tis[li]
        res["perturbseq_source"] = prov
        res["n_proteins_quantified"] = int(n_det[li])
        res["perturbed_gene_join"] = st_s
        per_line[cell] = res

    R["cell_matched_test"] = per_line

    # ======================================================================== the layer
    det = np.isfinite(M)
    layer = {
        "model": "pancancer-proteome-v1",
        "source": SOURCE,
        "release": "Proteomics_20221214 / Protein_matrix_averaged_20221214.tsv",
        "n_cell_lines": int(M.shape[0]), "n_proteins": int(M.shape[1]),
        "units": "averaged log2 DIA-MS intensity; comparable ACROSS cell lines within this release, "
                 "NOT comparable to iBAQ or ppm without renormalisation",
        "join": {"namespace": "uniprot", "rate": st_u["rate"],
                 "note": "joined on UniProt accession, a stronger key than the gene symbol PaxDb offers"},
        "cell_lines_carried": {},
        "pan_cancer": {},
        "validation": {k: R[k] for k in ("agreement_with_paxdb", "coverage", "cell_matched_test")},
        "limits": [],
        "abundance": {}}
    layer["cellspecific_panelz"] = {}
    for cell, mn, *_ in SCREENS:
        if mn is None:
            layer["cell_lines_carried"][cell] = None      # absent, NOT substituted
            continue
        col, li = column(mn)
        layer["abundance"][cell] = {e: round(v, 4) for e, v in col.items()}
        # the cell-SPECIFIC field, stored separately from the level because the test below says the two
        # behave differently: the level is better taken from iBAQ, the deviation is what matching buys
        layer["cellspecific_panelz"][cell] = {e: round(v, 3) for e, v in line_col(li, panel=True).items()}
        layer["cell_lines_carried"][cell] = {"procan_model_name": mn, "model_id": str(mids[li]),
                                             "tissue": tis[li], "n_quantified": int(det[li].sum())}
    for e_i, e in enumerate(eids):
        if not e:
            continue
        v = M[:, e_i]
        f = np.isfinite(v)
        if f.sum() < 20:
            continue
        cur = layer["pan_cancer"].get(e)
        rec = {"mean": round(float(v[f].mean()), 4), "sd": round(float(v[f].std()), 4),
               "detected_in": int(f.sum()), "of_lines": int(M.shape[0])}
        if cur is None or rec["detected_in"] > cur["detected_in"]:
            layer["pan_cancer"][e] = rec
    layer["limits"] = [
        "948 CANCER cell lines only: RPE1 (hTERT-immortalised, non-cancer) and HepG2 are absent, so two of "
        "the five Perturb-seq lines this project holds cannot be cell-matched at all and are reported as "
        "absent rather than substituted",
        f"{100*np.isnan(M).mean():.0f}% of the matrix is not quantified and detection is abundance- and "
        "line-dependent; a missing entry is UNKNOWN, never zero",
        "averaged log2 DIA intensity from one pipeline and one release -- the z-scored matrix shipped in the "
        "same archive is a different normalisation and is never mixed in here",
        "a raw ProCan column is NOT an absolute abundance scale: DIA intensity depends on peptide ionisation "
        "efficiency, so it compares one protein ACROSS cell lines well and different proteins WITHIN a cell "
        "line badly. `abundance` should not be used where iBAQ or ppm would be; `cellspecific_panelz` is the "
        "field that carries the cell-matched information and it is dimensionless by construction",
        "a static unperturbed proteome: it says nothing about how protein moves after a perturbation, which "
        "is still the arrow the chain is missing",
        "the Perturb-seq screens come from three different laboratories (Replogle K562/RPE1, Xaira HCT116, "
        "Nadig Jurkat/HepG2), so every head-to-head is run WITHIN one cell line and absolute values are "
        "never compared across lines",
        "protein is matched to a gene by taking the maximum over accessions mapping to it, which discards "
        "isoform structure",
        "perturbation strength is a pseudobulk transcriptional readout, so a knockdown that changes protein "
        "or flux without changing mRNA scores as no response"]
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)
    R["limits"] = layer["limits"]

    # ======================================================================== verdict
    report("\n" + "=" * 100)
    tested = {c: v for c, v in per_line.items() if v.get("procan_column")}
    if tested:
        W = []
        for c, v in tested.items():
            ca, cs_ = v["controls_absolute"], v["controls_cellspecific"]
            W.append(dict(line=c, matched=v["cell_matched"]["spearman"],
                          k562=v["k562_procan"]["spearman"], generic=v["generic_ppm"]["spearman"],
                          paxdb=v["paxdb_k562"]["spearman"], pz=v["cellspecific_panelz"]["spearman"],
                          pz_p=v["cellspecific_panelz"]["p"], inc=v["increment_over_paxdb"]["spearman"],
                          inc_p=v["increment_over_paxdb"]["p"],
                          a_rand=ca["random_line"], a_det=ca["detection_matched"],
                          a_tis=ca["same_tissue"], z_rand=cs_["random_line"],
                          z_det=cs_["detection_matched"], z_tis=cs_["same_tissue"]))
        report(f"  RAW HELD-OUT SPEARMAN vs that line's perturbation strength (never a difference)")
        report(f"  {'line':8s} {'matched':>9s} {'K562col':>9s} {'PaxDb':>9s} {'generic':>9s} "
               f"{'panel-z':>9s} {'incr':>9s}")
        for w in W:
            report(f"  {w['line']:8s} {w['matched']:+9.4f} {w['k562']:+9.4f} {w['paxdb']:+9.4f} "
                   f"{w['generic']:+9.4f} {w['pz']:+9.4f} {w['inc']:+9.4f}")
        report(f"\n  PAIRED CONTROL DRAWS ({N_DRAWS} each): control mean [matched-wins %]")
        report(f"  {'line':8s} {'abs rand':>18s} {'abs det-match':>18s} {'abs tissue':>18s} "
               f"{'pz rand':>18s} {'pz det-match':>18s} {'pz tissue':>18s}")
        cf = lambda d: (f"{d['control_mean']:+.4f}[{100*d['frac_draws_matched_better']:.0f}%]"
                        if d else "n/a")
        for w in W:
            report(f"  {w['line']:8s} {cf(w['a_rand']):>18s} {cf(w['a_det']):>18s} {cf(w['a_tis']):>18s} "
                   f"{cf(w['z_rand']):>18s} {cf(w['z_det']):>18s} {cf(w['z_tis']):>18s}")

        nonk = [w for w in W if w["line"] != "K562"]
        beat_k = sum(1 for w in nonk if w["matched"] > w["k562"])
        beat_pax = sum(1 for w in W if w["matched"] > w["paxdb"])
        pz_pos = sum(1 for w in W if w["pz"] > 0 and w["pz_p"] < 0.05)
        pz_win = [w["z_det"]["frac_draws_matched_better"] for w in W if w["z_det"]]
        inc_pos = sum(1 for w in W if w["inc"] > 0 and w["inc_p"] < 0.05)
        mean_pz_delta = float(np.mean([w["z_det"]["delta_mean"] for w in W if w["z_det"]]))
        strong = pz_pos == len(W) and inc_pos == len(W) and min(pz_win or [0]) >= 0.8
        head = ("CELL-MATCHING PAYS, BUT ONLY THROUGH THE CELL-SPECIFIC COMPONENT" if strong else
                "CELL-MATCHING BUYS A SMALL, PARTLY REAL EFFECT AND THE RAW COLUMN IS THE WRONG WAY TO USE IT"
                if mean_pz_delta > 0 else "CELL-MATCHING IS NOT SUPPORTED ON THIS TASK")
        j = lambda fn: ", ".join(fn(w) for w in W)
        s_m = j(lambda w: "%s %+.3f" % (w["line"], w["matched"]))
        s_px = j(lambda w: "%s %+.3f" % (w["line"], w["paxdb"]))
        s_pz = j(lambda w: "%s %+.3f (p %.2g)" % (w["line"], w["pz"], w["pz_p"]))
        s_in = j(lambda w: "%s %+.3f (p %.2g)" % (w["line"], w["inc"], w["inc_p"]))
        s_wn = ", ".join("%s %.0f%%" % (w["line"], 100 * w["z_det"]["frac_draws_matched_better"])
                         for w in W if w["z_det"])
        v = (f"{head}. ProCan-DepMapSanger downloaded and reduced: {M.shape[0]:,} cancer cell lines x "
             f"{M.shape[1]:,} proteins, joined on UniProt ACCESSION at {st_u['rate']:.1%}. Of the five "
             f"Perturb-seq lines this project holds, three are in the panel (K-562, HCT-116, Jurkat); RPE1 "
             f"is non-cancer and HepG2 was not profiled, and neither was substituted. "
             f"(a) The ProCan K-562 column agrees with the PaxDb/Geiger K562 iBAQ vector already in the "
             f"model at Spearman {rho:.3f} over {len(both):,} shared genes -- but {len(rnd_rho)} RANDOM "
             f"ProCan lines agree with the same vector at {rnd_rho.mean():.3f} +- {rnd_rho.std():.3f}, so "
             f"the cell-specific margin is only {rho-rnd_rho.mean():+.3f}: protein abundance is "
             f"overwhelmingly a human constant, and that ceiling governs everything below. "
             f"(b) ProCan does NOT fill PaxDb's hole. Coverage of the cell object goes {100*cx.mean():.1f}% "
             f"(PaxDb) -> {100*cp.mean():.1f}% (ProCan K-562) -> {100*(cp|cx).mean():.1f}% (union), and in "
             f"the two lowest generic-abundance quintiles {100*cx[lo].mean():.1f}% -> "
             f"{100*cp[lo].mean():.1f}% -> {100*(cp|cx)[lo].mean():.1f}%. Both sources miss the same faint "
             f"proteins; ProCan-covered genes carry mean dep_frac {dpa[cp].mean():.3f} against "
             f"{dpa[~cp].mean():.3f} for uncovered, so absence stays UNKNOWN, never zero. "
             f"(c) Raw held-out Spearman against each line's own perturbation strength: cell-matched "
             f"{s_m}. The raw ProCan column LOSES to PaxDb iBAQ on {len(W)-beat_pax}/{len(W)} lines "
             f"({s_px}) and to the K-562 ProCan column on {len(nonk)-beat_k}/{len(nonk)} non-K562 lines -- "
             f"averaged DIA intensity is not built for cross-protein comparison, so that contest is a unit "
             f"contest and not a matching one. With the unit confound removed by standardising each protein "
             f"across the 948 lines, the purely cell-specific panel-z scores {s_pz}, positive and "
             f"significant on {pz_pos}/{len(W)}, beating a DETECTION-MATCHED random line in {s_wn} of "
             f"{N_DRAWS} paired draws (mean delta {mean_pz_delta:+.4f}), and still explaining the residual "
             f"after PaxDb iBAQ is regressed out on {inc_pos}/{len(W)} lines ({s_in}). The operational "
             f"reading: carry an iBAQ-like absolute abundance for the LEVEL, and carry the pan-cancer "
             f"panel-z as a SEPARATE cell-specific field -- not as a replacement column.")
    else:
        v = ("ProCan-DepMapSanger fetched and joined, but no cell line had both a ProCan column and a "
             "Perturb-seq screen in scratch, so the cell-matching test could not be run.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "pancancer_proteome.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'pancancer_proteome.json'}")


if __name__ == "__main__":
    main()
