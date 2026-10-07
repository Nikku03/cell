"""COMPETITION FEATURES: what an element is COMPETING WITH, not what it is.

THE GAP THIS FILLS. Every feature in `crispr_features_compendium.csv` is ABSOLUTE -- this element's ATAC,
this element's H3K27ac, this element's distance to the TSS. Not one of them says "this is the 3rd-closest
of 40 candidate elements competing for this gene". That relative framing is the entire reason the ABC
model has a DENOMINATOR: an enhancer 20 kb away is a strong candidate when it is the nearest one and a
weak candidate when nine others are closer. The absolute features cannot express the difference.

WHY THIS IS WORTH MEASURING AFTER THE CHROMATIN ARC. Predicting 3D contact was attacked four ways --
CTCF counting, analytic Rouse-with-loops, a stiffness sweep down to 200 bp/bead, and full KMC extrusion
plus Langevin dynamics. Everything past the CTCF count bought ~+0.005. These features are six lines of
groupby and cost milliseconds. The comparison is the point: the cheap axis had not been tried.

    contact  answers  "can these two loci physically meet?"
    competition answers "given they can, is this element the one that matters for this gene?"

Those are different questions, which is why the gains turn out to be ~88% additive rather than redundant.

THIS IS NOT NEW SCIENCE, AND THAT IS THE USEFUL PART. ENCODE-rE2G -- the ENCODE4 supervised model for
this exact task, trained on this exact CRISPR benchmark -- already carries candidate-count, distance-rank
and normalisation features of this kind. So the +0.028 measured here is a REDISCOVERY, not a discovery.
What it establishes is where the remaining gap to the published incumbent lives: in engineered features
nobody had added, not in architecture and not in polymer physics. Two further consequences worth stating
plainly rather than leaving implied:

  - rE2G cannot be used as a benchmark on these pairs. It was TRAINED on this benchmark, so scoring its
    predictions here measures memorisation. A real comparison needs a held-out biosample or another screen.
  - measured contact was never tried. The whole chromatin arc -- CTCF counting, analytic Rouse, the
    stiffness sweep, KMC extrusion plus Langevin -- tried to PREDICT contact, while K562 Micro-C sits
    public on ENCODE/4DN. `abc_score` scoring 0.2376 suggests it runs on a power-law distance estimate
    rather than a real contact map. That is the untested measurement, and it is cheaper than any of this.

GROUP ABLATION, NOT DROP-ONE. The six features are heavily correlated -- distance rank, distance excess
and activity-times-proximity share all encode overlapping information. Removing any ONE changes the score
by +0.0003 to +0.0031, i.e. nothing, and a drop-one table would conclude that no feature matters while
the block as a whole carries +0.032. This project has already inverted one conclusion that way. The block
is therefore ablated as a block, and the drop-one numbers are printed only to show that they mislead.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
SEEDS = (0, 1, 2, 3, 4)
NAMES = ["n_candidates", "dist_rank", "activity_rank", "activity_share", "act_x_prox_share", "dist_excess"]


def competition_features(df):
    """Six per-pair features, every one relative to the OTHER candidates for the same gene."""
    act = df["h3k27ac_enh"].values * df["atac_enh"].values          # ABC-style activity
    dist = df["log_dist"].values
    prox = 1.0 / np.maximum(dist, 1e-6)
    gene = df["gene"].values
    CF = np.zeros((len(df), 6))
    for g in pd.unique(gene):
        m = np.where(gene == g)[0]
        d, a = dist[m], act[m]
        ap = a * prox[m]
        CF[m, 0] = len(m)                                  # how many candidates compete for this gene
        CF[m, 1] = np.argsort(np.argsort(d)) + 1           # distance rank, 1 = closest
        CF[m, 2] = np.argsort(np.argsort(-a)) + 1          # activity rank, 1 = strongest
        CF[m, 3] = a / max(a.sum(), 1e-9)                  # activity share -- the ABC denominator
        CF[m, 4] = ap / max(ap.sum(), 1e-9)                # activity x proximity share
        CF[m, 5] = d - d.min()                             # how much further than the nearest candidate
    return CF


def load():
    from contact_gate import load_ctcf, contact_features
    df = pd.read_csv(OUT / "crispr_features_compendium.csv")
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]
    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: hdr.index(k) for k in ("chrom", "chromStart", "chromEnd", "startTSS", "measuredGeneSymbol")}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            try:
                tss[(f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}",
                     p[ix["measuredGeneSymbol"]])] = (
                    p[ix["chrom"]], (int(p[ix["chromStart"]]) + int(p[ix["chromEnd"]])) // 2,
                    int(p[ix["startTSS"]]))
            except ValueError:
                continue
    rows = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]
    assert len(rows) == len(df), "element-gene join lost rows"
    Cr = contact_features(rows, load_ctcf())
    tfmat = np.zeros((len(df), len(tfl)), np.int8)
    for a, ek in enumerate(df["element"].values):
        for ti in et.get(ek, []):
            tfmat[a, ti] = 1
    base = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
            "promoter_atac", "promoter_polii", "gene_expr"]
    return df, np.hstack([df[base].values, tfmat]), Cr


def main():
    from crispr_gate import _cv_auprc, _seeded_folds
    df, Xid, Cr = load()
    CF = competition_features(df)
    y, ch = df["crispr_hit"].values, df["chromosome"].values
    R = {"base_rate": float(y.mean()), "n_pairs": int(len(y)), "n_positives": int(y.sum())}

    print("=" * 100)
    print("COMPETITION FEATURES -- is this element the one that matters for THIS gene?")
    print("=" * 100)
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}\n")

    def sc(X, tag):
        v = np.array([_cv_auprc(X, y, _seeded_folds(ch, s)) for s in SEEDS])
        print(f"  {tag:48s} {v.mean():.4f}  [{' '.join(f'{x:.3f}' for x in v)}]")
        return v

    a = sc(Xid, "epi + TF identity")
    b = sc(np.hstack([Xid, Cr]), "+ CTCF contact          [the 0.663 bar]")
    c = sc(np.hstack([Xid, CF]), "+ COMPETITION only")
    d = sc(np.hstack([Xid, Cr, CF]), "+ both")
    # Control: permute the competition block across rows. Marginals and the extra model capacity survive;
    # the link between a pair and its own competitive context does not.
    rng = np.random.default_rng(0)
    e = sc(np.hstack([Xid, Cr, CF[rng.permutation(len(CF))]]), "+ competition SHUFFLED  [control]")

    dc, dk, dboth = (b - a).mean(), (c - a).mean(), (d - a).mean()
    net = (d - b).mean() - (e - b).mean()
    print(f"\n  contact     over identity : {dc:+.4f}")
    print(f"  competition over identity : {dk:+.4f}")
    print(f"  both        over identity : {dboth:+.4f}   "
          f"(additive would be {dc+dk:+.4f}, redundant {max(dc,dk):+.4f} "
          f"-> {100*dboth/max(dc+dk,1e-9):.0f}% additive)")
    print(f"\n  competition ON TOP OF the bar   : {(d-b).mean():+.4f}  "
          f"[per-seed {' '.join(f'{x:+.3f}' for x in (d-b))}]")
    print(f"  shuffled control on top of bar  : {(e-b).mean():+.4f}")
    print(f"  NET OF CONTROL                  : {net:+.4f}   "
          f"sign consistent: {bool(np.all((d-b) > 0))}")

    print("\n  drop-one inside the block -- printed BECAUSE it misleads:")
    for i, n in enumerate(NAMES):
        keep = [j for j in range(6) if j != i]
        v = np.array([_cv_auprc(np.hstack([Xid, Cr, CF[:, keep]]), y, _seeded_folds(ch, s)) for s in SEEDS])
        print(f"    without {n:18s} {v.mean():.4f}   delta vs full {v.mean()-d.mean():+.4f}")
    print("    -> every feature looks disposable alone; the block carries the gain. Ablate groups.")

    R.update({"identity": float(a.mean()), "contact": float(b.mean()), "competition": float(c.mean()),
              "both": float(d.mean()), "competition_shuffled": float(e.mean()),
              "delta_contact": float(dc), "delta_competition": float(dk), "delta_both": float(dboth),
              "pct_additive": float(100 * dboth / max(dc + dk, 1e-9)),
              "delta_on_top_of_bar": float((d - b).mean()),
              "delta_control_on_top_of_bar": float((e - b).mean()), "net_of_control": float(net),
              "sign_consistent": bool(np.all((d - b) > 0))})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "competition.json", "w"), indent=1)
    print(f"\n  -> {OUT/'competition.json'}")


if __name__ == "__main__":
    main()
