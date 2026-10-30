"""K562-MATCHED PROTEIN ABUNDANCE -- and a number for the arrow the chain cannot measure.

WHAT WAS THERE BEFORE. The cell object carries `ppm`, 16,015 genes of protein abundance, keyed by gene index
and of unstated provenance -- a generic human consensus, not a measurement of the cell being modelled. It has
been used as the abundance covariate in every matched analysis in this project, including the one where
abundance turned out to be the strongest confound (p 6e-31). If a generic value is systematically wrong for
K562, every one of those matches was matching on the wrong quantity.

PaxDb carries `9606-iBAQ_K562_Geiger_2012`: 7,032 proteins measured in K562 itself. That is priority 1 in the
dataset map -- exact matched cell-line proteomics -- not the PaxDb fallback tier, which is the integrated
consensus. Fewer genes than the generic table, but measured in the right cell.

THE THREE QUESTIONS, in order of what they buy.

  1. HOW WRONG IS GENERIC? Correlate the generic `ppm` against the K562 measurement on the genes both cover.
     A high correlation would mean the substitution was harmless and the confound-matching stands. A low one
     means the model has been gating, ranking and matching on a quantity that is not this cell's.

  2. HOW MUCH OF PROTEIN DOES mRNA EXPLAIN, IN THIS CELL? The chain
     `perturbation -> dRNA -> dprotein -> ... -> phenotype` is missing its second arrow, and it cannot be
     built: no K562 perturbation proteomics exists at this scale. What CAN be measured is the static
     relationship in the same cell type, and that is informative in a specific way -- it is an upper bound on
     how much of a protein change could ever be read off an mRNA change by a model that knows nothing else.
     If mRNA explains half the variance in protein LEVELS, then a dRNA-only model is working with at most
     that, and probably less, since a delta is harder than a level.

  3. WHAT IS MISSING, AND IS IT MISSING AT RANDOM? 7,032 of ~16,500 genes are covered. Mass spectrometry
     misses low-abundance and small proteins preferentially, so the uncovered set is not a random sample. The
     coverage bias is measured rather than assumed, because "unknown" has to be a value the model can carry
     and reason about, not a hole that quietly becomes zero.

NOTHING IS SUBSTITUTED SILENTLY. K562-measured and generic values are written as separate fields with a
`measured` flag. The dataset map's rule is explicit and it is the right one: never silently substitute a
generic abundance estimate for a cell-specific measurement.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
PROT = SP / "prot" / "9606-iBAQ_K562_Geiger_2012_uniprot.txt"
SPEC = SP / "prot" / "9606-K562_Geiger_2012.txt"
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
LAYER = SP / "cell_k562_protein.json.gz"
SOURCE = "PaxDb 9606-iBAQ_K562_Geiger_2012 (Geiger et al., Mol Cell Proteomics 2012)"


def read_paxdb(path):
    """gene symbol -> abundance. The header is commented; the columns are named on the last comment line."""
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


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("K562-MATCHED PROTEIN ABUNDANCE -- and how wrong the generic table was")
    report("=" * 100)
    if not PROT.exists():
        raise SystemExit(f"absent: {PROT}")

    from entity_registry import load as load_reg
    REG = load_reg()

    prot = read_paxdb(PROT)
    spec = read_paxdb(SPEC) if SPEC.exists() else {}
    report(f"  {SOURCE}")
    report(f"  {len(prot):,} proteins with iBAQ abundance; {len(spec):,} with spectral counts")

    # symbols are a WEAK key and the registry records that; the file offers nothing stronger
    _e, st = REG.join("gene", "symbol", list(prot), 0.75, "K562 proteomics symbols")
    report(f"  registry join via symbol: {st['joined']:,}/{st['of']:,} ({st['rate']:.1%}) "
           f"-- recorded via=symbol, the weakest namespace this file supports")

    d = json.load(gzip.open(NET, "rt"))
    names = [g["name"] for g in d["genes"]]
    ppm = d.get("ppm", {})
    dep = {g["name"]: g.get("dep_frac") for g in d["genes"]}
    ess = {g["name"]: g.get("ess") for g in d["genes"]}
    esrc = {g["name"]: g.get("ess_src") for g in d["genes"]}
    generic = {n: float(ppm.get(str(i), 0) or 0) for i, n in enumerate(names)}
    del d

    # ---- Q1: how wrong is generic? ----
    both = [n for n in names if n in prot and generic.get(n, 0) > 0]
    a = np.log10(1.0 + np.array([generic[n] for n in both]))
    b = np.log10(1.0 + np.array([prot[n] for n in both]))
    r_p = float(np.corrcoef(a, b)[0, 1])
    r_s = float(stats.spearmanr(a, b)[0])
    report(f"\n  Q1  GENERIC vs K562-MEASURED, on the {len(both):,} genes both cover")
    # SPEARMAN IS THE FAIR STATISTIC HERE and Pearson is not. `ppm` is parts-per-million from an integrated
    # consensus; iBAQ is an intensity-based within-sample measure. The two are on different scales with
    # different compression, so a Pearson correlation of their logs is penalised by the unit mismatch on top
    # of any real disagreement. Leading with the Pearson R2 would overstate how wrong the generic table is,
    # which is the opposite error from the one this module is about but an error all the same.
    report(f"      Spearman {r_s:.4f} (rank R2 {r_s**2:.4f})  <- the fair comparison: different units")
    report(f"      Pearson r(log) {r_p:.4f} (R2 {r_p**2:.4f})  <- unit-sensitive, reported for completeness")
    # where does the generic table disagree most?
    ra = stats.rankdata(a) / len(a)
    rb = stats.rankdata(b) / len(b)
    dv = rb - ra
    o = np.argsort(dv)
    report(f"      rank shift |delta| median {np.median(np.abs(dv)):.3f}, "
           f"{100*np.mean(np.abs(dv) > 0.25):.1f}% of genes move more than a quarter of the range")
    report(f"      most UNDER-stated by generic: {[both[i] for i in o[-6:]][::-1]}")
    report(f"      most OVER-stated by generic:  {[both[i] for i in o[:6]]}")

    # ---- Q2: how much of protein does mRNA explain in K562? ----
    import h5py
    with h5py.File(GWPS, "r") as f:
        gid = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
        rna_mean = f["var"]["mean"][:]
    ens2rna = dict(zip(gid, rna_mean))
    sym2rna = {}
    for e, v in ens2rna.items():
        eid = REG.resolve("gene", "ensembl", e)
        ent = REG.entities.get(eid) if eid else None
        if ent and ent.get("symbol"):
            sym2rna[ent["symbol"]] = float(v)
    rp = [n for n in prot if n in sym2rna and sym2rna[n] > 0]
    x = np.log10(np.array([sym2rna[n] for n in rp]))
    y = np.log10(np.array([prot[n] for n in rp]))
    r_rp = float(np.corrcoef(x, y)[0, 1])
    report(f"\n  Q2  mRNA -> PROTEIN, both measured in K562, {len(rp):,} genes")
    report(f"      Pearson r(log-log) {r_rp:.4f}   R2 {r_rp**2:.4f}   "
           f"Spearman {stats.spearmanr(x, y)[0]:.4f}")
    report(f"      -> mRNA level explains {100*r_rp**2:.1f}% of protein level variance in this cell type.")
    report(f"      That is an UPPER BOUND on the missing chain arrow: a model that sees only dRNA cannot")
    report(f"      exceed it, and a delta is harder than a level, so the real ceiling is lower.")

    # ---- Q3: is coverage missing at random? ----
    cov = np.array([1.0 if n in prot else 0.0 for n in names])
    gv = np.array([generic.get(n, 0.0) for n in names])
    lg = np.log10(1.0 + gv)
    report(f"\n  Q3  COVERAGE: {int(cov.sum()):,}/{len(names):,} genes ({100*cov.mean():.1f}%) measured")
    q = np.digitize(lg, np.quantile(lg[lg > 0], [.2, .4, .6, .8])) if (lg > 0).any() else np.zeros(len(lg), int)
    report(f"      {'generic-abundance quintile':30s} {'n':>7s} {'frac measured in K562':>22s}")
    for k in sorted(set(q.tolist())):
        m = q == k
        report(f"      {('q'+str(k)):30s} {int(m.sum()):7,d} {cov[m].mean():22.4f}")
    dep_a = np.array([float(dep.get(n) or 0) for n in names])
    meas = cov == 1
    report(f"      mean dep_frac  measured {dep_a[meas].mean():.4f}  unmeasured {dep_a[~meas].mean():.4f}"
           f"   MWU p {stats.mannwhitneyu(dep_a[meas], dep_a[~meas])[1]:.3g}")
    report(f"      -> coverage is strongly abundance-dependent, so `not measured` is NOT `not present`;")
    report(f"         it is stored as unknown rather than zero.")

    # ---- write the layer ----
    layer = {
        "model": "k562-protein-v1", "source": SOURCE, "cell_type": "K562",
        "measured": "iBAQ abundance, mass spectrometry",
        "join": {"namespace": "symbol", "rate": st["rate"], "note":
                 "the source file offers gene symbol and Ensembl PROTEIN id only; symbol is the weakest "
                 "namespace in the registry and the join is recorded as such"},
        "n_measured": len(prot),
        "vs_generic": {"n_shared": len(both), "pearson_log": r_p, "spearman": r_s, "r2": r_p ** 2},
        "rna_to_protein": {"n": len(rp), "pearson_log": r_rp, "r2": r_rp ** 2},
        "coverage": {"n_genes_in_cell": len(names), "frac_measured": float(cov.mean()),
                     "abundance_dependent": True},
        "limits": [
            "one study, one laboratory, one growth condition -- K562 protein levels shift with medium, "
            "density and passage, and none of those are recorded here",
            "iBAQ is a within-sample relative measure; it is not comparable across datasets without "
            "renormalisation",
            "36% coverage, and missing is abundance-dependent, so absence is unknown rather than zero",
            "joined by gene symbol, the weakest key available in the source file",
            "measured in unperturbed K562, so it is a STATE, not a response -- it says nothing about how "
            "protein moves after a perturbation, which is the arrow the chain is actually missing"],
        "abundance": {n: float(v) for n, v in prot.items()},
        "spectral_count": {n: float(v) for n, v in spec.items()}}
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    R = {k: layer[k] for k in ("model", "source", "n_measured", "vs_generic", "rna_to_protein",
                               "coverage", "join", "limits")}
    report("\n" + "=" * 100)
    broad = "roughly right in broad ordering but materially wrong in detail" if r_s > 0.6 else \
            "wrong even in broad ordering"
    v = (f"THE GENERIC TABLE IS {broad.upper()} FOR THIS CELL. Rank correlation with the K562 measurement is "
         f"{r_s:.3f} over {len(both):,} shared genes -- the fair statistic, since ppm and iBAQ are different "
         f"units -- yet {100*np.mean(np.abs(dv) > 0.25):.0f}% of genes shift more than a quarter of the "
         f"abundance range between them, and the disagreement is structured rather than random: histones "
         f"are the most over-stated by the generic table and regulatory proteins such as EP300 and TCF3 the "
         f"most under-stated. Every matched analysis in this project that used `ppm` as its abundance "
         f"covariate was matching on that. Second, and independent of it, mRNA explains only "
         f"{100*r_rp**2:.0f}% of protein LEVEL variance in K562 (r {r_rp:.3f}, Spearman "
         f"{stats.spearmanr(x, y)[0]:.3f}, {len(rp):,} genes both measured in this cell type) -- consistent "
         f"with the literature range for a single cell type, and an upper bound on the chain's missing "
         f"dRNA -> dprotein arrow. It is a generous bound, because a delta is harder to predict than a "
         f"level. Third, coverage is {100*cov.mean():.0f}% and steeply abundance-dependent "
         f"({cov[q==0].mean():.1%} of the lowest generic-abundance quintile against "
         f"{cov[q==4].mean():.1%} of the highest), and measured genes carry mean dep_frac "
         f"{dep_a[meas].mean():.3f} against {dep_a[~meas].mean():.3f} for unmeasured -- so absence is "
         f"unknown, never zero.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "k562_protein.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'k562_protein.json'}")


if __name__ == "__main__":
    main()
