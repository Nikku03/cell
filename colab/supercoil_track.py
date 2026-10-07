"""A PER-LOCUS DNA TORSION TRACK, MAPPED ONTO THE CELL MODEL'S GENES.

WHAT THIS CLOSES.  The integration assessment found one absence blocking three of five proposed
chromatin-to-biology couplings: superhelical density exists in this repo only as an INPUT --
`sigmas = (0.0, -0.02, -0.04, -0.06)`, four hardcoded values model-wide at chromatin_supercoil.py:204 --
and 0 of 16,492 gene records carried it. Torque is an exact affine image of sigma, so the proposed
[torque, sigma] feature block was rank 0: a zero-variance column. No amount of physics fixes that. Only a
measurement does.

THE MEASUREMENT.  GSE277502, bTMP-seq in hTERT-RPE1, deposited on hg38, three bigWigs totalling 4.0 MB:
two supercoiling replicates already expressed as a ratio (mean 1.006 and 1.008, max 9) and a matched
NAKED GENOMIC DNA control (mean 545, raw coverage). bTMP is trimethylpsoralen: it intercalates
preferentially into UNWOUND DNA, so high signal means locally underwound, which means negative
supercoiling. All three files were byte-verified -- magic 26fc8f88, 24 hg38 chromosomes, 2.94-3.09 Gb
covered -- before anything was computed from them.

WHAT THIS IS NOT, STATED FIRST BECAUSE IT IS THE MAIN CAVEAT.
    NOT sigma. bTMP enrichment is a relative propensity to crosslink, not a superhelical density in
        turns per turn. The field written is `torsion_btmp`, never `sigma`. Anyone who maps this onto the
        rod's sigma axis is inventing a calibration that does not exist.
    NOT K562. RPE1 is a near-diploid retinal epithelial line; the repo's scoring harness is K562. Every
        use of this track is a CROSS-CELL-LINE transfer and must say so.
    NOT high resolution. The native bin is 20 kb. Promoter-proximal torsion structure lives at hundreds
        of bp, so this track cannot see it. It can see domain-scale torsion, which is what the original
        paper reports.

THREE CONTROLS, because a track that cannot fail is not a measurement.
    A  REPLICATE CONCORDANCE. rep1 against rep2 across genes. Two independent libraries of the same
       quantity; if they disagree the track is noise and nothing below it means anything.
    B  POSITION SPECIFICITY. the TSS window against a window displaced 500 kb along the same chromosome.
       If a displaced window scores the same, the track has no locus resolution at this bin size and
       every per-gene value is really a chromosome-arm average.
    C  STRAND ASYMMETRY -- the reason strand was added to cell_complete this morning. Twin-domain
       predicts negative supercoiling BEHIND an elongating polymerase and positive AHEAD, so bTMP should
       run higher upstream of the TSS than downstream. This is a SIGNED prediction and it was untestable
       until now. At a 20 kb bin it is a weak test and may well come back null; a null here is a real
       result about resolution, not about the mechanism.

PREDECLARED, before any number:
    replicates concordant and the displaced window scores lower
        -> the track carries real per-locus information and can be joined to the cell model.
    the displaced window scores the same
        -> 20 kb bins wash out locus identity; report the track as chromosome-scale only and do NOT
           offer it as a per-gene feature.
    replicates disagree
        -> the track is not measuring a reproducible quantity here and is not written.
    strand asymmetry in the twin-domain direction
        -> the first signed chromatin-to-biology observation in this repo, and worth pursuing at
           higher resolution.
    strand asymmetry absent or reversed
        -> reported as measured. A reversed asymmetry would be more interesting than a confirmation and
           must not be explained away.

-> outputs/orphan/supercoil_track.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = OUT / "orphan" / "supercoil"
CELL = OUT / "orphan" / "cell_complete.json"
REP1 = SC / "GSM8523629_supercoil_rep1_hg38.bw"
REP2 = SC / "GSM8523630_supercoil_rep2_hg38.bw"
CTRL = SC / "GSM8523628_RPE1_bTMP_genomic_control_hg38.bw"
WIN = 20_000            # the track's native bin; asking for finer would be fabricating resolution
FLANK = 20_000          # upstream/downstream arm for the strand test
SHIFT = 500_000         # displacement for the position-specificity control
SEED = 4477
ACC = "GSE277502"


def val(bw, chrom, lo, hi):
    if chrom not in bw.chroms():
        return np.nan
    lo, hi = max(0, int(lo)), min(int(hi), bw.chroms()[chrom])
    if hi <= lo:
        return np.nan
    try:
        v = bw.stats(chrom, lo, hi, type="mean")[0]
    except Exception:
        return np.nan
    return np.nan if v is None else float(v)


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    import pyBigWig
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("A PER-LOCUS DNA TORSION TRACK, MAPPED ONTO THE CELL MODEL'S GENES")
    report("=" * 100)
    report(f"  {ACC}  bTMP-seq, hTERT-RPE1, hg38. bTMP intercalates into UNWOUND DNA, so high signal")
    report("  means locally underwound, i.e. negatively supercoiled. Two replicates already expressed as")
    report("  a ratio, plus a matched naked-genomic-DNA control.")
    report("  This is NOT sigma, NOT K562, and NOT finer than 20 kb. All three matter downstream.")

    bw1, bw2, bwc = pyBigWig.open(str(REP1)), pyBigWig.open(str(REP2)), pyBigWig.open(str(CTRL))
    for nm, bw in (("rep1", bw1), ("rep2", bw2), ("control", bwc)):
        h = bw.header()
        report(f"    {nm:<9}{len(bw.chroms()):>3} chroms  {h['nBasesCovered']/1e9:.2f} Gb  "
               f"mean {h['sumData']/max(h['nBasesCovered'],1):.4f}")

    D = json.load(open(CELL))
    genes = D["genes"]
    rng = np.random.default_rng(SEED)

    rows = []
    for g in genes:
        ch, tss, sd = g.get("chrom"), g.get("tss"), g.get("strand")
        if not ch or tss in (None, ""):
            continue
        try:
            tss = int(float(tss))
        except (TypeError, ValueError):
            continue
        v1 = val(bw1, ch, tss - WIN // 2, tss + WIN // 2)
        v2 = val(bw2, ch, tss - WIN // 2, tss + WIN // 2)
        vc = val(bwc, ch, tss - WIN // 2, tss + WIN // 2)
        vs = val(bw1, ch, tss + SHIFT - WIN // 2, tss + SHIFT + WIN // 2)
        up = dn = np.nan
        if sd in ("+", "-"):
            a = val(bw1, ch, tss - FLANK, tss)
            b = val(bw1, ch, tss, tss + FLANK)
            up, dn = (a, b) if sd == "+" else (b, a)
        rows.append({"name": g.get("name"), "chrom": ch, "tss": tss, "strand": sd,
                     "rep1": v1, "rep2": v2, "control": vc, "shifted": vs,
                     "upstream": up, "downstream": dn})

    A = {k: np.array([r[k] for r in rows], dtype=float) for k in
         ("rep1", "rep2", "control", "shifted", "upstream", "downstream")}
    n_ok = int(np.isfinite(A["rep1"]).sum())
    report(f"\n  {len(rows)} genes with usable coordinates; {n_ok} carry a track value")

    # ---- CONTROL A ----------------------------------------------------------------------------------
    rho = spearman(A["rep1"], A["rep2"])
    report(f"\n  CONTROL A -- replicate concordance")
    report(f"    Spearman rep1 vs rep2 across genes: {rho:.4f}")
    conc = rho > 0.5

    # ---- CONTROL B ----------------------------------------------------------------------------------
    m = np.isfinite(A["rep1"]) & np.isfinite(A["shifted"])
    rho_shift = spearman(A["rep1"], A["shifted"])
    same = float(np.nanmean(np.abs(A["rep1"][m] - A["shifted"][m])))
    spread = float(np.nanstd(A["rep1"][m]))
    report(f"\n  CONTROL B -- position specificity: TSS window vs the same window moved {SHIFT/1000:.0f} kb")
    report(f"    Spearman TSS vs displaced:   {rho_shift:.4f}   (concordance would mean no locus identity)")
    report(f"    mean |difference|            {same:.4f}   against a track SD of {spread:.4f}")
    specific = rho_shift < rho - 0.10

    # ---- CONTROL C ----------------------------------------------------------------------------------
    mm = np.isfinite(A["upstream"]) & np.isfinite(A["downstream"])
    du = A["upstream"][mm] - A["downstream"][mm]
    n = int(mm.sum())
    mean_d = float(du.mean()) if n else float("nan")
    se = float(du.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = mean_d / se if se else float("nan")
    report(f"\n  CONTROL C -- STRAND ASYMMETRY, the signed test strand was added for")
    report(f"    twin-domain predicts MORE negative supercoiling behind the polymerase, so bTMP should")
    report(f"    run higher UPSTREAM of the TSS than downstream.")
    report(f"    n = {n} genes with strand and both flanks")
    report(f"    mean(upstream - downstream) = {mean_d:+.5f}   se {se:.5f}   z = {z:+.2f}")
    if abs(z) < 2:
        report(f"    NULL at this bin size. The native bin is {WIN/1000:.0f} kb and the flank is "
               f"{FLANK/1000:.0f} kb, so this test cannot see promoter-proximal structure. Reported as a")
        report("    statement about resolution, not about the mechanism.")
    elif z > 0:
        report("    ASYMMETRY IN THE PREDICTED DIRECTION. First signed chromatin-to-biology observation")
        report("    here; worth pursuing at higher resolution before any weight is put on it.")
    else:
        report("    ASYMMETRY REVERSED -- downstream exceeds upstream. Reported as measured. This is more")
        report("    interesting than a confirmation would have been and must not be explained away.")

    report(f"\n  GATES: replicates concordant {conc}   position-specific {specific}")
    if not conc:
        report("  NOT WRITTEN -- replicates disagree, so the track is not a reproducible quantity here.")
        return 1
    if not specific:
        report("  WRITTEN, BUT FLAGGED CHROMOSOME-SCALE ONLY -- a displaced window scores like the TSS,")
        report("  so 20 kb bins wash out locus identity and this must not be used as a per-gene feature.")

    # ---- the artefact -------------------------------------------------------------------------------
    track = {r["name"]: {"torsion_btmp": r["rep1"], "torsion_btmp_rep2": r["rep2"],
                         "btmp_control": r["control"], "chrom": r["chrom"], "tss": r["tss"],
                         "strand": r["strand"]}
             for r in rows if np.isfinite(r["rep1"])}
    res = {"test": "supercoil_track", "accession": ACC,
           "assay": "bTMP-seq (trimethylpsoralen); high = unwound = negatively supercoiled",
           "cell_line": "hTERT-RPE1", "genome_build": "hg38", "bin_bp": WIN,
           "IS_NOT": {"sigma": "relative crosslink propensity, not superhelical density in turns/turn",
                      "K562": "RPE1; every downstream use is a cross-cell-line transfer",
                      "high_resolution": f"native bin {WIN} bp cannot see promoter-proximal torsion"},
           "n_genes": len(track), "n_coords": len(rows),
           "controls": {"replicate_spearman": rho, "shifted_spearman": rho_shift,
                        "position_specific": bool(specific), "concordant": bool(conc),
                        "strand_asymmetry": {"n": n, "mean_up_minus_down": mean_d, "se": se, "z": z}},
           "track": track, "log": log}
    man = RM.manifest(inputs=[str(REP1), str(REP2), str(CTRL), str(CELL)],
                      available=len(genes), used=len(track), selection="filtered", seed=SEED,
                      controls=["replicate concordance", "displaced window", "strand asymmetry"],
                      note=f"{ACC} bTMP-seq RPE1 hg38, mean over TSS +/- {WIN//2} bp")
    res["manifest"] = man
    RM.report(man, emit=report)
    SC.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "orphan" / "supercoil_track.json", "w"), indent=2)
    report(f"\n  -> {OUT/'orphan'/'supercoil_track.json'}  ({len(track)} genes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
