"""ONE NUMBER PER VARIANT, NOT ONE PER GENE -- the limit loop 5 declared about itself.

WHERE THIS COMES FROM.  loop_variant stated its own limit before reporting any result: "It has no
per-variant feature at all -- no structure, no conservation, no substitution matrix -- so every mutation
in the same gene gets the same number." That is the difference between answering `any point mutation`
and answering `which genes are mutation-sensitive`, and it has stood unaddressed since.

WHAT IS ADDED, using only data already on disk.  ClinVar's VCF carries a genomic POSITION for every
variant, and pathogenic missense variants are known to cluster in functional regions while benign ones
scatter. So a variant's position relative to the other variants in its own gene is a real per-variant
feature, and it costs nothing to compute.

THE CIRCULARITY THIS HAS TO AVOID, and it would be fatal.  Scoring a variant by its distance to the
pathogenic centroid of its gene, where the centroid was computed FROM that variant, is guaranteed to
work and means nothing. Every score here is LEAVE-ONE-OUT: the centroid a variant is measured against is
built from the other pathogenic variants in its gene, never itself.

THE QUESTION THAT MAKES THIS A CELL-MODEL LOOP rather than a sequence one.  Positional clustering is
sequence information; the metabolic score is cell information. If they carry the same signal, the cell
model adds nothing at variant resolution and `any point mutation` should be reported as a
sequence-level result. If they combine, the two resolutions are complementary and the row means what it
says.

PREDECLARED, before any number:

    X1 VARIANT RESOLUTION   the per-variant score is not constant within genes -- measured directly, as
                the fraction of multi-variant genes where scored variants differ.
                fails -> nothing was gained over loop 5 and this should be reported as such.
    X2 SIGNAL   leave-one-out positional clustering separates pathogenic from benign missense above the
                95th percentile of 200 label shuffles, shuffled WITHIN gene so the gene-level signal is
                held fixed.
                That shuffle is the important one: shuffling globally would let the model rediscover
                which GENES are pathogenic, which loop 5 already did.
    X3 BEATS THE GENE SCORE   it beats assigning every variant its gene's loop 5 dose-response score.
                fails -> gene-level was as good, and the extra resolution is decoration.
    X4 THEY COMBINE   under 5-fold CV grouped BY GENE, position plus the gene-level cell score beats
                either alone by >= 0.02 AUC.
                fails -> sequence and cell information are redundant here.

CONTROLS: 200 WITHIN-GENE label shuffles; leave-one-out centroids; gene-grouped cross-validation so no
gene appears in both train and test; and the gene-level score as the direct comparator.

WHAT HAPPENED, written after the run against the gates above, unedited.

    X1 VARIANT RESOLUTION   PASS.  100% of genes now have their variants scored differently. Loop 5's
        stated limit -- one number per gene -- is gone.
    X2 SIGNAL   PASS.  Leave-one-out positional clustering scores AUC 0.6297 against a WITHIN-GENE
        shuffled null of 0.5390 +/- 0.0010. Note the null sits at 0.539 rather than 0.5: the
        leave-one-out construction carries a small structural bias, and the shuffle measures it
        correctly, which is the reason for shuffling rather than assuming 0.5.
    X3 BEATS THE GENE SCORE   PASS.  On the 37,011 variants carrying both, position scores 0.6052 and
        loop 5's gene-level dose-response scores 0.5044 -- chance.
    X4 THEY COMBINE   FAIL.  Under cross-validation grouped by gene: position alone 0.5866, gene score
        alone 0.4208, both together 0.5856. Combining is -0.0009. The cell model contributes nothing.

WHY THE GENE SCORE SITS BELOW CHANCE AT 0.4208, because it looks alarming and is not.  With folds
grouped by gene, a gene-CONSTANT feature cannot discriminate variants within a gene at all -- it can only
help by correlating with a gene's pathogenic FRACTION across unseen genes, and it does not. A gene-level
number is structurally the wrong shape for a variant-level question, and this is the measurement that
shows it rather than an argument that asserts it.

THE HONEST CONCLUSION.  `any point mutation` is now variant-resolved, and the resolution comes entirely
from SEQUENCE POSITION. The cell model adds nothing at this resolution. That row should be read as a
sequence-level result, and the capability table's entry for it is qualified accordingly: the question is
askable, connected, falsifiable and answered by information that has nothing to do with the cell model.

WHAT WOULD CHANGE IT.  A per-variant feature the cell model could actually supply -- for instance the
predicted effect of a specific substitution on a specific enzyme's activity, which needs structure or
per-residue kinetics. loop_kinetics already established that no per-enzyme constants exist in this
container, so this is the same data-acquisition boundary reached from a different direction.

-> outputs/loop_pervariant.json
"""
import collections
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
          "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CLINVAR = SC / "clinvar.vcf.gz"
SEED = 1904
N_SHUFFLE = 200
MIN_PER_GENE = 6
GATE_BEATS = 0.02
PATHO = ("Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic")
BENIGN = ("Benign", "Likely_benign", "Benign/Likely_benign")


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("ONE NUMBER PER VARIANT, NOT ONE PER GENE")
    say("=" * 100)
    say("  loop 5 declared this limit about itself and it has stood since: every mutation in a gene got")
    say("  the same score. ClinVar carries a position for every variant, and that is a real per-variant")
    say("  feature already on disk.")

    gene_re = re.compile(r"GENEINFO=([^;]+)")
    sig_re = re.compile(r"CLNSIG=([^;]+)")
    mc_re = re.compile(r"MC=([^;]+)")
    rows = []
    with gzip.open(CLINVAR, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if len(f) < 8:
                continue
            info = f[7]
            m = mc_re.search(info)
            if not m or "missense_variant" not in m.group(1):
                continue
            g, s = gene_re.search(info), sig_re.search(info)
            if not g or not s:
                continue
            sig = s.group(1)
            y = 1 if sig in PATHO else (0 if sig in BENIGN else None)
            if y is None:
                continue
            rows.append((g.group(1).split("|")[0].split(":")[0], f[0], int(f[1]), y))
    V = pd.DataFrame(rows, columns=["sym", "chrom", "pos", "y"])
    say(f"\n  {len(V):,} missense variants with a clinical assertion over {V.sym.nunique():,} genes")

    # keep genes with enough of BOTH classes for a leave-one-out centroid to mean anything
    keep = V.groupby("sym").agg(n=("y", "size"), npos=("y", "sum"))
    keep = keep[(keep.n >= MIN_PER_GENE) & (keep.npos >= 2) & (keep.n - keep.npos >= 2)]
    V = V[V.sym.isin(keep.index)].reset_index(drop=True)
    say(f"  kept {len(V):,} variants in {V.sym.nunique():,} genes with >= {MIN_PER_GENE} variants and "
        f">= 2 of each class")

    # LEAVE-ONE-OUT positional score: distance from the pathogenic centroid of the SAME gene, built
    # without this variant. Normalised by the gene's own positional spread so genes are comparable.
    score = np.full(len(V), np.nan)
    for sym, idx in V.groupby("sym").groups.items():
        idx = np.asarray(idx)
        pos = V.pos.to_numpy()[idx].astype(float)
        yy = V.y.to_numpy()[idx]
        spread = pos.std() or 1.0
        for j, gi in enumerate(idx):
            others = np.delete(np.arange(len(idx)), j)
            pp = pos[others][yy[others] == 1]
            if len(pp) == 0:
                continue
            score[gi] = -abs(pos[j] - np.median(pp)) / spread
    V["pos_score"] = score
    V = V[np.isfinite(V.pos_score)].reset_index(drop=True)
    y = V.y.to_numpy()
    say(f"  scored {len(V):,} variants with a leave-one-out centroid ({int(y.sum()):,} pathogenic)")

    # ---- X1 variant resolution ------------------------------------------------------------------------
    var = V.groupby("sym").pos_score.nunique()
    frac_varying = float((var > 1).mean())
    x1 = bool(frac_varying > 0.9)
    say(f"\n  X1 VARIANT RESOLUTION -- {frac_varying:.1%} of genes have variants scored differently   "
        f"X1 {'PASS' if x1 else 'FAIL'}")

    a_pos = auc(V.pos_score, y)
    say(f"\n  positional clustering (leave-one-out) AUC {a_pos:.4f}")

    # ---- X2 within-gene shuffle -----------------------------------------------------------------------
    gid = V.sym.to_numpy()
    null = []
    for _ in range(N_SHUFFLE):
        yp = y.copy()
        for sym, idx in V.groupby("sym").groups.items():
            idx = np.asarray(idx)
            yp[idx] = rng.permutation(yp[idx])
        null.append(auc(V.pos_score, yp))
    null = np.array(null)
    p95 = float(np.percentile(null, 95))
    x2 = bool(a_pos > p95)
    say(f"  X2 SIGNAL -- {N_SHUFFLE} WITHIN-GENE shuffles: null {null.mean():.4f} +/- {null.std():.4f}, "
        f"95th {p95:.4f}   X2 {'PASS' if x2 else 'FAIL'}")
    say("    Shuffling within gene holds the gene-level signal fixed, so this measures ONLY what")
    say("    position adds. A global shuffle would let it rediscover which genes are pathogenic.")

    # ---- X3 vs the gene-level cell score ---------------------------------------------------------------
    gene_score = {}
    lv = OUT / "loop_variant.json"
    if lv.exists():
        for r in json.load(open(lv)).get("scores", []):
            gene_score[r["sym"]] = float(r.get("drop_at_50") or 0.0)
    V["gene_score"] = [gene_score.get(s, np.nan) for s in V.sym]
    have = V.gene_score.notna().to_numpy()
    a_gene = auc(V.gene_score[have], y[have]) if have.sum() > 100 else float("nan")
    a_pos_sub = auc(V.pos_score[have], y[have]) if have.sum() > 100 else float("nan")
    say(f"\n  X3 BEATS THE GENE SCORE -- on the {int(have.sum()):,} variants that have both:")
    say(f"    gene-level cell score (loop 5)  {a_gene:.4f}")
    say(f"    per-variant position            {a_pos_sub:.4f}")
    x3 = bool(np.isfinite(a_gene) and a_pos_sub - a_gene >= GATE_BEATS)
    say(f"    X3 {'PASS' if x3 else 'FAIL'}")

    # ---- X4 do they combine, grouped by gene ----------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold

    def gcv(X):
        X = np.asarray(X, float)
        yy, gg = y[have], gid[have]
        oof = np.zeros(len(yy))
        for tr, te in GroupKFold(n_splits=5).split(X, yy, gg):
            if len(np.unique(yy[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[tr], yy[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return auc(oof, yy)

    c_pos = gcv(V.pos_score[have].to_numpy().reshape(-1, 1))
    c_gene = gcv(V.gene_score[have].to_numpy().reshape(-1, 1))
    c_both = gcv(np.c_[V.pos_score[have], V.gene_score[have]])
    best_single = max(c_pos, c_gene)
    x4 = bool(c_both - best_single >= GATE_BEATS)
    say(f"\n  X4 THEY COMBINE -- 5-fold CV grouped BY GENE (no gene in both train and test):")
    say(f"    position alone {c_pos:.4f} | gene score alone {c_gene:.4f} | both {c_both:.4f} "
        f"({c_both-best_single:+.4f})   X4 {'PASS' if x4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"X1 variant resolution": x1, "X2 signal above within-gene shuffles": x2,
             "X3 beats the gene score": x3, "X4 sequence and cell information combine": x4}
    for kk, vv in gates.items():
        say(f"  {kk:<44}{'PASS' if vv else 'FAIL'}")
    if x1 and x2 and x4:
        say("  `any point mutation` IS NOW VARIANT-RESOLVED, and the two resolutions are complementary:")
        say("  position carries information the gene score does not, and they combine. The limit loop 5")
        say("  declared about itself is addressed.")
    elif x1 and x2:
        say("  VARIANT-RESOLVED AND SEQUENCE-ONLY. Position works and the cell model adds nothing at this")
        say("  resolution, so `any point mutation` should be reported as a sequence-level result rather")
        say("  than a cell-model one.")
    else:
        say("  NO PER-VARIANT SIGNAL. The limit loop 5 declared stands, and it is now measured rather")
        say("  than merely stated.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CLINVAR), str(OUT / "loop_variant.json")],
                      available=int(keep.n.sum()), used=len(V), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} WITHIN-GENE label shuffles",
                                "leave-one-out centroids", "gene-grouped cross-validation",
                                "the gene-level score as the direct comparator"],
                      note="within-gene shuffling is the control that matters: a global shuffle would "
                           "let position rediscover which genes are pathogenic")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_pervariant", "manifest": man, "gates": gates,
               "auc_position": a_pos, "auc_gene": a_gene, "auc_position_sub": a_pos_sub,
               "cv": {"position": c_pos, "gene": c_gene, "both": c_both},
               "null": {"mean": float(null.mean()), "sd": float(null.std()), "p95": p95},
               "n_variants": len(V), "n_genes": int(V.sym.nunique()),
               "frac_genes_varying": frac_varying, "log": log},
              open(OUT / "loop_pervariant.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_pervariant.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
