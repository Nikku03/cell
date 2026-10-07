"""PHASE 0 GATE: does a loop-aware CONTACT feature beat the 0.608 TF-identity model?

THE QUESTION, AND WHY IT IS WORTH ONE DAY BEFORE ANY POLYMER PHYSICS. The 4D chromatin plan exists to produce
ONE scalar per element-gene pair: how close they are in 3D. The production model already has two contact
columns -- `log_dist` and `abc_score` -- and reaches AUPRC 0.608 with 311 measured TF-occupancy features.
If a better contact estimate cannot move that number, Stages 2 and 3 of the plan (KMC extrusion + Langevin
polymer dynamics, ~10^4-10^5 GPU-hours) have nothing to buy, and the honest move is to not build them.

WHAT IS ACTUALLY NEW HERE. CTCF is already one of the 311 TFs, so whether CTCF binds THE ELEMENT is a feature
the 0.608 model has. What it has never had is CTCF BETWEEN the element and the promoter -- which is the actual
loop-domain physics. An enhancer and a promoter separated by a strong CTCF boundary sit in different insulated
neighbourhoods and should not contact, however close they are in base pairs. That is the whole content of
Stages 1-2 of the plan, collapsed into features that cost seconds instead of GPU-months:

    ctcf_between        number of CTCF peaks strictly between element and TSS   (boundary crossings)
    ctcf_between_max    strongest boundary in the interval                      (one strong > many weak)
    ctcf_between_sum    total insulation in the interval
    ctcf_near_elem      distance to the nearest CTCF peak from the element      (anchor proximity)
    ctcf_near_tss       same from the TSS
    contact_insul       log-contact under an insulated-neighbourhood model:
                        log P ~ -gamma*log10(d) - beta*n_between
                        Trees approximate smooth multiplicative forms badly, so the composite is supplied
                        explicitly rather than left to be discovered from its parts.

THE CONTROL THAT DECIDES IT. CTCF peaks are not uniformly distributed -- they cluster at active, gene-dense,
accessible regions, which is exactly where CRISPR-positive elements live. A count of anything genomic between
two points therefore correlates with the answer for reasons that have nothing to do with insulation. So every
contact feature is recomputed with CTCF peak POSITIONS SHUFFLED WITHIN EACH CHROMOSOME, preserving peak count
and chromosome composition while destroying where the boundaries actually are. If the real features do not
beat their shuffled twin, this is measuring genomic density, not loop domains.

Protocol is identical to the run that produced 0.608 -- same GroupKFold-by-chromosome, same XGBoost settings,
same 3 seeds, same label-shuffle control -- so the numbers are directly comparable and not re-tuned.
"""
import bisect
import collections
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path("outputs/orphan")
INV = OUT / "invivo"
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CTCF = SP / "ctcf_gw.bed.gz"
GAMMA, BETA = 1.0, 1.5      # ABC's power-law exponent; boundary penalty in log10 units per crossing


def load_ctcf(shuffle_seed=None):
    """chrom -> (sorted centre positions, matching signal values).

    shuffle_seed relocates every peak uniformly within its own chromosome's span, keeping the number of peaks
    per chromosome and the signal distribution intact. That is the control for CTCF simply marking active,
    gene-dense territory rather than marking insulation.
    """
    pos, sig = collections.defaultdict(list), collections.defaultdict(list)
    with gzip.open(CTCF, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                c = (int(p[1]) + int(p[2])) // 2
                s = float(p[6]) if len(p) > 6 and p[6] not in ("", ".", "-1") else 1.0
            except ValueError:
                continue
            pos[p[0]].append(c)
            sig[p[0]].append(s)
    out = {}
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    for c in pos:
        a = np.array(pos[c], np.int64)
        v = np.array(sig[c], np.float64)
        if rng is not None:
            a = rng.integers(a.min(), a.max() + 1, size=len(a))
        o = np.argsort(a)
        out[c] = (a[o], v[o])
    return out


def contact_features(rows, ctcf):
    """rows = [(chrom, elem_centre, tss)] -> feature matrix in the column order of NAMES."""
    F = np.zeros((len(rows), 6), np.float64)
    for i, (c, e, t) in enumerate(rows):
        a, v = ctcf.get(c, (np.array([], np.int64), np.array([], np.float64)))
        lo, hi = (e, t) if e <= t else (t, e)
        l = bisect.bisect_right(a, lo)
        r = bisect.bisect_left(a, hi)
        n = max(r - l, 0)
        seg = v[l:r] if n else np.array([0.0])
        d = max(abs(e - t), 1)
        # nearest peak to each end, capped so an empty chromosome cannot produce an outlier
        def near(x):
            if len(a) == 0:
                return 6.0
            j = bisect.bisect_left(a, x)
            cand = [a[k] for k in (j - 1, j) if 0 <= k < len(a)]
            return np.log10(max(min(abs(x - y) for y in cand), 1))
        F[i] = (n, float(seg.max()), float(seg.sum()), near(e), near(t),
                -GAMMA * np.log10(d) - BETA * n)
    return F


NAMES = ["ctcf_between", "ctcf_between_max", "ctcf_between_sum",
         "ctcf_near_elem", "ctcf_near_tss", "contact_insul"]


def main():
    import pandas as pd
    from crispr_gate import _cv_auprc, _seeded_folds

    print("=" * 100)
    print("PHASE 0 GATE -- loop-aware contact vs the 0.608 TF-identity model")
    print("=" * 100)

    df = pd.read_csv(OUT / "crispr_features_compendium.csv")
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]
    print(f"  {len(df):,} element-gene pairs, {len(tfl)} TFs, base rate {df['crispr_hit'].mean():.3f}")

    # element+gene -> TSS, from the CRISPR source. Joined on BOTH keys: one element pairs with many genes,
    # so joining on the element alone would attach the wrong TSS to most rows.
    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: hdr.index(k) for k in ("chrom", "chromStart", "chromEnd", "startTSS", "measuredGeneSymbol")}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            try:
                key = (f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}", p[ix["measuredGeneSymbol"]])
                tss[key] = (p[ix["chrom"]], (int(p[ix["chromStart"]]) + int(p[ix["chromEnd"]])) // 2,
                            int(p[ix["startTSS"]]))
            except ValueError:
                continue
    rows, miss = [], 0
    for el, gn in zip(df["element"].values, df["gene"].values):
        r = tss.get((el, gn))
        if r is None:
            miss += 1
            c = el.split(":")[0]
            s, e = el.split(":")[1].split("-")
            r = (c, (int(s) + int(e)) // 2, (int(s) + int(e)) // 2)
        rows.append(r)
    print(f"  TSS join: {len(df)-miss:,}/{len(df):,} matched ({100*(len(df)-miss)/len(df):.1f}%), {miss:,} fell back")
    assert miss < 0.1 * len(df), f"TSS join failed for {miss} rows -- the coordinates would be meaningless"

    real = load_ctcf()
    print(f"  CTCF: {sum(len(v[0]) for v in real.values()):,} peaks over {len(real)} chromosomes")
    Fr = contact_features(rows, real)
    Fs = contact_features(rows, load_ctcf(shuffle_seed=7))
    nz = (Fr[:, 0] > 0).mean()
    print(f"  pairs with >=1 CTCF peak between element and TSS: {100*nz:.1f}%  "
          f"(median count {np.median(Fr[:,0]):.0f}, shuffled {np.median(Fs[:,0]):.0f})")

    tfmat = np.zeros((len(df), len(tfl)), np.int8)
    for i, ek in enumerate(df["element"].values):
        for ti in et.get(ek, []):
            tfmat[i, ti] = 1
    base = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
            "promoter_atac", "promoter_polii", "gene_expr"]
    y = df["crispr_hit"].values
    chroms = df["chromosome"].values
    Xd = df[["log_dist"]].values
    Xb = df[base].values
    Xa = np.hstack([Xb, tfmat])

    arms = [("distance", Xd),
            ("epigenetics", Xb),
            ("epi + TF identity (311)", Xa),
            ("epi + TF + CONTACT", np.hstack([Xa, Fr])),
            ("epi + TF + contact SHUFFLED", np.hstack([Xa, Fs])),
            ("CONTACT only", Fr)]
    res = {}
    for name, X in arms:
        sc = [_cv_auprc(X, y, _seeded_folds(chroms, s)) for s in (0, 1, 2)]
        res[name] = {"auprc": float(np.mean(sc)), "seeds": [round(float(x), 4) for x in sc]}
        print(f"    {name:30s} AUPRC {np.mean(sc):.4f}   seeds {[round(float(x),3) for x in sc]}")

    rng = np.random.default_rng(10)
    ys = y.copy()
    for c in sorted(set(chroms)):
        m = chroms == c
        v = ys[m].copy()
        rng.shuffle(v)
        ys[m] = v
    sh = _cv_auprc(np.hstack([Xa, Fr]), ys, _seeded_folds(chroms, 0))
    print(f"    {'label-shuffle control':30s} AUPRC {sh:.4f}   (base rate {y.mean():.3f})")
    res["label_shuffle"] = float(sh)

    # GROUP ablation, not drop-one. Drop-one on this feature set is actively misleading: the three BETWEEN
    # features are mutually redundant, so removing any ONE costs nothing (the other two cover it) and the
    # readout said "only ctcf_near_tss matters, everything else is noise". Ablating the correlated GROUP says
    # the opposite -- insulation carries +0.0514 of the +0.0556, anchor proximity only +0.0131. Correlated
    # features have to be ablated together or the conclusion inverts.
    BETW = ["ctcf_between", "ctcf_between_max", "ctcf_between_sum"]
    NEAR = ["ctcf_near_elem", "ctcf_near_tss"]
    idx = lambda ns: [NAMES.index(x) for x in ns]
    base_id = res["epi + TF identity (311)"]["auprc"]
    print("\n  group ablation -- which physics carries the gain (with its shuffled-CTCF twin):")
    res["groups"] = {}
    for tag, cols in (("BETWEEN (insulation)", BETW), ("NEAR (anchor proximity)", NEAR)):
        a = float(np.mean([_cv_auprc(np.hstack([Xa, Fr[:, idx(cols)]]), y, _seeded_folds(chroms, s))
                           for s in (0, 1, 2)]))
        b = float(np.mean([_cv_auprc(np.hstack([Xa, Fs[:, idx(cols)]]), y, _seeded_folds(chroms, s))
                           for s in (0, 1, 2)]))
        print(f"    {tag:26s} {a:.4f} ({a-base_id:+.4f})   shuffled twin {b:.4f} ({a-b:+.4f} real vs shuffled)")
        res["groups"][tag] = {"real": a, "shuffled": b, "vs_identity": a - base_id, "vs_shuffled": a - b}

    ident = res["epi + TF identity (311)"]["auprc"]
    cont = res["epi + TF + CONTACT"]["auprc"]
    csh = res["epi + TF + contact SHUFFLED"]["auprc"]
    print("\n  " + "-" * 96)
    print(f"  contact MINUS TF-identity model : {cont - ident:+.4f}")
    print(f"  contact MINUS shuffled contact  : {cont - csh:+.4f}   <- is it insulation, or genomic density?")
    clean = sh <= y.mean() + 0.03
    if cont - ident >= 0.01 and cont - csh >= 0.005 and clean:
        v = ("GO -- loop-aware contact adds real signal beyond 311 measured TFs AND beats its own shuffled "
             "control. A better contact model is worth building.")
    elif cont - ident >= 0.01 and clean:
        v = ("AMBIGUOUS -- contact helps, but not more than CTCF placed at random. The gain is genomic "
             "density, not insulation; a polymer model would not add what this does not measure.")
    else:
        v = ("NO-GO -- a loop-aware contact feature does not move the 0.608 model. Stages 2-3 of the 4D plan "
             "(KMC + Langevin, ~10^4-10^5 GPU-hours) have nothing to buy here.")
    print(f"\n  VERDICT: {v}")
    res["verdict"] = v
    res["deltas"] = {"vs_identity": cont - ident, "vs_shuffled_contact": cont - csh}
    json.dump(res, open(OUT / "contact_gate.json", "w"), indent=1)
    print(f"\n  -> {OUT/'contact_gate.json'}")


if __name__ == "__main__":
    main()
