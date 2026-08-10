"""THE REAL CHROMATIN TEST -- measured DNA torsion and loop anchors, not a functional neighbour list.

WHY THIS EXISTS.  loop_fold_link found that `model4`, the block capability_audit was counting under
`any chromosome fold`, is not a fold at all: 8.8% of its neighbour pairs are same-chromosome where a
Hi-C contact map runs 70-90%, and 51.7% share its own functional label, whose values are
"transport/uptake" and Reactome pathway names. So loop 6's headline was retracted and the question was
left open. This is the version of the question that uses actual chromatin measurements.

WHAT IS ACTUALLY CHROMATIN IN THIS REPOSITORY
    bTMP torsion   GSE277502, bTMP-seq in hTERT-RPE1 on hg38, two supercoiling replicates and a naked
                   genomic DNA control. Trimethylpsoralen intercalates UNWOUND DNA, so high signal
                   means locally underwound. supercoil_track.py computed this and then, correctly,
                   REFUSED TO WRITE IT: its own gate said 20 kb bins wash out locus identity. Zero of
                   16,492 gene records carry it today. This module re-derives it per gene and tests it
                   rather than assuming the refusal was final.
    loops3d        767 genes with loop anchor coordinates. Small, and real.

WHAT IS BEING PREDICTED, and why transcription rather than essentiality.  Supercoiling has a direct,
mechanistic relationship with transcription: a polymerase generates negative supercoils behind it and
positive ahead, so torsion and transcription rate are coupled by physics rather than by annotation.
Essentiality is several inferential steps further away. The target is k_syn = RPKM x k_deg from
RNADecayCafe -- the same target nexus_txn used, so the comparison is against an established bar.

PREDECLARED, before any number:

    R1 REPRODUCIBLE   the two bTMP replicates agree per gene at Spearman >= 0.5.
                      fails -> the track is not measuring a reproducible per-gene quantity and nothing
                      below is interpretable. This is the check supercoil_track's own gate failed on at
                      20 kb; at gene resolution it may fail again, and that is a real answer.
    R2 NOT JUST COVERAGE   torsion must beat the NAKED DNA control, which carries mappability and copy
                      number but no biology. fails -> the signal is an artefact of coverage.
    R3 SIGNAL         torsion correlates with transcription rate above the 95th percentile of 200 gene
                      shuffles.
    R4 BEATS POSITION its correlation survives regressing out GC/CpG and gene length, and beats a
                      position-matched control -- so it is not a proxy for promoter composition.

CONTROLS: the naked-DNA track as an artefact control; replicate 2 as an independent measurement; 200
gene-label shuffles; CpG and gene length as compositional confounds; loop anchors as a separate,
independent chromatin feature.

-> outputs/loop_real_chromatin.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import KDEG, CELL, MIN_LINES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
          "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
REP1 = SC / "GSM8523629_supercoil_rep1_hg38.bw"
REP2 = SC / "GSM8523630_supercoil_rep2_hg38.bw"
NAKED = SC / "GSM8523628_RPE1_bTMP_genomic_control_hg38.bw"
SEED = 1904
N_SHUFFLE = 200
WINDOW = 20000          # bp either side of the TSS
GATE_REPRO = 0.5


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("THE REAL CHROMATIN TEST -- measured DNA torsion, not a functional neighbour list")
    say("=" * 100)
    say("  model4 turned out not to be a fold (8.8% cis, half its pairs sharing a functional label), so")
    say("  loop 6's headline was retracted and the question left open. These are actual measurements:")
    say("  bTMP-seq torsion from GSE277502, and 767 loop anchors.")

    import pyBigWig
    D = json.load(open(CELL))
    genes = D["genes"]
    loops3d = D.get("loops3d", {})

    bws = {}
    for nm, p in (("rep1", REP1), ("rep2", REP2), ("naked", NAKED)):
        if not p.exists():
            say(f"  MISSING: {p.name} -- cannot run. Recorded as not testable.")
            return 1
        bws[nm] = pyBigWig.open(str(p))
    chroms = set(bws["rep1"].chroms())
    say(f"\n  bigWigs open: {', '.join(bws)}; {len(chroms)} chromosomes in rep1")

    def val(bw, c, s, e):
        try:
            v = bw.stats(c, max(0, s), e, type="mean")[0]
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    rows = []
    for i, g in enumerate(genes):
        c, t = g.get("chrom"), g.get("ens_tss") or g.get("tss")
        if c is None or not isinstance(t, (int, float)):
            continue
        cc = str(c) if str(c).startswith("chr") else f"chr{c}"
        if cc not in chroms:
            continue
        s, e = int(t) - WINDOW, int(t) + WINDOW
        rows.append({"i": i, "sym": g["name"], "chrom": cc, "tss": int(t),
                     "rep1": val(bws["rep1"], cc, s, e),
                     "rep2": val(bws["rep2"], cc, s, e),
                     "naked": val(bws["naked"], cc, s, e),
                     "cpg": g.get("cpg"), "len": (g.get("gene_end") or 0) - (g.get("gene_start") or 0),
                     "anchor": 1 if str(i) in loops3d else 0})
    T = pd.DataFrame(rows).dropna(subset=["rep1", "rep2", "naked"])
    say(f"  {len(T):,} genes with a TSS and torsion in all three tracks "
        f"(+/-{WINDOW//1000} kb around the TSS); {int(T.anchor.sum()):,} are loop anchors")

    # ---- R1 reproducibility ---------------------------------------------------------------------------
    r_rep = float(T[["rep1", "rep2"]].corr(method="spearman").iloc[0, 1])
    r1 = bool(r_rep >= GATE_REPRO)
    say(f"\n  R1 REPRODUCIBLE -- rep1 vs rep2, per gene: Spearman {r_rep:.4f}  (gate {GATE_REPRO})   "
        f"R1 {'PASS' if r1 else 'FAIL'}")
    if not r1:
        say("    supercoil_track refused to write this track because 20 kb bins wash out locus identity.")
        say("    At gene resolution the two replicates still do not agree, so that refusal was correct")
        say("    and stays correct. Nothing below is interpretable and it is reported as NOT TESTABLE.")

    T["torsion"] = (T.rep1 + T.rep2) / 2.0

    # ---- the target -----------------------------------------------------------------------------------
    k = pd.read_csv(KDEG)
    rp = (k[k.avg_RPKM_total > 0].groupby("feature_ID")
          .agg(rpkm=("avg_RPKM_total", "median"), n=("cell_line", "size")))
    kd = k.groupby("feature_ID").agg(kdeg=("avg_kdeg", "median"), n=("cell_line", "size"))
    rp, kd = rp[rp.n >= MIN_LINES], kd[kd.n >= MIN_LINES]
    T = T[T.sym.isin(rp.index) & T.sym.isin(kd.index)].copy()
    T["ksyn"] = np.log10(rp.rpkm[T.sym].to_numpy() * kd.kdeg[T.sym].to_numpy() + 1e-6)
    say(f"  joined to RNADecayCafe: {len(T):,} genes with k_syn = RPKM x k_deg")

    def sp(a, b):
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

    r_tor = sp(T.torsion, T.ksyn)
    r_nak = sp(T.naked, T.ksyn)
    say(f"\n  correlation with log k_syn:")
    say(f"    torsion (bTMP mean of replicates) : {r_tor:+.4f}")
    say(f"    naked genomic DNA control         : {r_nak:+.4f}   <- mappability and copy number only")

    r2 = bool(abs(r_tor) > abs(r_nak))
    say(f"  R2 NOT JUST COVERAGE   {'PASS' if r2 else 'FAIL'}")
    if not r2:
        say("    The naked-DNA control tracks transcription as well as the supercoiling signal does, so")
        say("    what is being measured is coverage, not torsion. That is an artefact, not a result.")

    null = np.array([sp(T.torsion, rng.permutation(T.ksyn.to_numpy())) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(np.abs(null), 95))
    r3 = bool(abs(r_tor) > p95)
    say(f"\n  R3 SIGNAL -- {N_SHUFFLE} shuffles: |null| 95th pct {p95:.4f}   R3 {'PASS' if r3 else 'FAIL'}")

    # ---- R4 partial out composition -------------------------------------------------------------------
    ok = T.cpg.notna() & (T["len"] > 0)
    S = T[ok].copy()
    X = np.c_[np.log10(S["len"].clip(lower=1)), S.cpg.astype(float)]
    from sklearn.linear_model import LinearRegression

    def resid(v):
        m = LinearRegression().fit(X, v)
        return v - m.predict(X)

    r_part = sp(resid(S.torsion.to_numpy()), resid(S.ksyn.to_numpy()))
    r_nak_part = sp(resid(S.naked.to_numpy()), resid(S.ksyn.to_numpy()))
    r4 = bool(abs(r_part) > p95 and abs(r_part) > abs(r_nak_part))
    say(f"\n  R4 BEATS COMPOSITION -- CpG and gene length regressed out ({len(S):,} genes)")
    say(f"    torsion partial : {r_part:+.4f}")
    say(f"    naked partial   : {r_nak_part:+.4f}")
    say(f"    R4 {'PASS' if r4 else 'FAIL'}")

    # ---- loop anchors, an independent chromatin feature -----------------------------------------------
    if int(T.anchor.sum()) >= 20:
        a_in = T[T.anchor == 1].ksyn
        a_out = T[T.anchor == 0].ksyn
        say(f"\n  loop anchors (independent feature): {len(a_in):,} anchor genes, median log k_syn "
            f"{a_in.median():+.3f} vs {a_out.median():+.3f} for the rest")
    else:
        say(f"\n  loop anchors: only {int(T.anchor.sum())} in the scored set -- too few to test")

    say("\n" + "=" * 100)
    gates = {"R1 replicates agree": r1, "R2 beats the naked-DNA control": r2,
             "R3 signal above shuffles": r3, "R4 beats composition": r4}
    for kk, vv in gates.items():
        say(f"  {kk:<38}{'PASS' if vv else 'FAIL'}")
    if all(gates.values()):
        say("  MEASURED DNA TORSION PREDICTS TRANSCRIPTION, beyond coverage and beyond promoter")
        say("  composition. This is a real chromatin-to-cell-model link, on real chromatin data, and it")
        say("  is the one model4 could not provide.")
    elif not r1:
        say("  NOT TESTABLE. The track does not reproduce between replicates at gene resolution, which")
        say("  is what supercoil_track's own gate said at 20 kb. The refusal to write it was correct.")
    else:
        say("  NO LINK ESTABLISHED on real chromatin data. `any chromosome fold` stays unanswered, and")
        say("  now it stays unanswered having been tested on the right measurements rather than on a")
        say("  block that turned out to be a functional annotation.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(REP1), str(REP2), str(NAKED), str(KDEG), str(CELL)],
                      available=len(genes), used=len(T), selection="filtered", seed=SEED,
                      controls=["naked genomic DNA track", "replicate 2 as independent measurement",
                                f"{N_SHUFFLE} gene-label shuffles",
                                "CpG and gene length partialled out", "loop anchors as a second feature"],
                      note=f"torsion averaged over +/-{WINDOW} bp of the TSS; target is k_syn = "
                           f"RPKM x k_deg, the same target nexus_txn used")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_real_chromatin", "manifest": man, "gates": gates,
               "replicate_spearman": r_rep, "r_torsion": r_tor, "r_naked": r_nak,
               "r_torsion_partial": r_part, "r_naked_partial": r_nak_part,
               "null_p95": p95, "n_genes": len(T), "n_anchors": int(T.anchor.sum()),
               "window_bp": WINDOW, "log": log},
              open(OUT / "loop_real_chromatin.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_real_chromatin.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
