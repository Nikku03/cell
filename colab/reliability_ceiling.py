"""reliability_ceiling -- the replicate experiment, run computationally. Item #5 on the needs list was "run replicates", which is wet-lab. But
each knockout was measured in MANY CELLS, so we can split those cells in half and treat the halves as independent replicates. That measures the
TEST-RETEST RELIABILITY of a knockout profile without generating any new data -- and reliability sets a hard ceiling on predictability.

WHY IT SETTLES THE ARGUMENT. If a measurement has reliability r (correlation between two independent measurements of the same thing), then no
predictor -- however perfect its biology -- can correlate with that measurement better than sqrt(r). This is the classic attenuation bound. The
project has spent a long time asking whether the ~0.62 oracle is a BIOLOGY ceiling or a NOISE ceiling; this distinguishes them:
   high reliability  -> the ceiling is real biology, and only new modalities/perturbations can move it
   low reliability   -> a large part of the ceiling is measurement noise, and simply MEASURING DEEPER moves it, which is far cheaper

WHAT IT REPORTS, per cell line:
   split-half reliability of the knockout z-profile, and the Spearman-Brown correction to full depth
   the implied ceiling sqrt(r) on any predictor's correlation with the measured profile
   MOVER reproducibility: of the genes called movers in half A, how many are movers in half B (the quantity that actually matters, since every
     metric in this project is a mover-recall)
   reliability as a function of CELL COUNT, which says directly how many cells per knockout are needed to make a profile trustworthy
Deterministic (fixed seed). Usage: python reliability_ceiling.py [line ...]"""
import sys, json, collections
from pathlib import Path
import numpy as np
import h5py

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = Path("outputs/orphan")
FILES = {"RPE1": "rpe1.h5ad", "HepG2": "nadig_hepg2.h5ad", "Jurkat": "nadig_jurkat.h5ad"}
MINCELL = 100          # need enough cells that each half is still meaningful
CHUNK = 20000
THR = 4.0
RNG = np.random.RandomState(0)


def decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def run(line):
    f = SP / FILES[line]
    with h5py.File(f, "r") as h:
        pert = h["obs"]["perturbation"]
        cats = decode(pert["categories"][:]); codes = pert["codes"][:]
        genes = decode(h["var"]["gene_name"][:]) if "gene_name" in h["var"] else decode(h["var"]["_index"][:])
        X = h["X"]; ncell, ngene = X.shape
        ctrl_code = next((i for i, c in enumerate(cats) if c.lower() == "control"), None)
        counts = collections.Counter(codes.tolist())
        keep = [i for i in range(len(cats)) if i != ctrl_code and counts[i] >= MINCELL]
        print(f"[{line}] {len(keep)} knockouts with >={MINCELL} cells (needed so each half is still usable)")
        # assign every cell to half A or B, deterministically
        half = RNG.randint(0, 2, size=ncell)
        SA = np.zeros((len(cats), ngene), np.float64); SB = np.zeros((len(cats), ngene), np.float64)
        CA = np.zeros(ngene, np.float64); CB = np.zeros(ngene, np.float64)
        for start in range(0, ncell, CHUNK):
            end = min(start + CHUNK, ncell)
            blk = X[start:end].astype(np.float32); cb = codes[start:end]; hb = half[start:end]
            for c in np.unique(cb):
                m = cb == c
                a = m & (hb == 0); b = m & (hb == 1)
                if a.any():
                    SA[c] += blk[a].sum(0)
                if b.any():
                    SB[c] += blk[b].sum(0)
            mc = cb == ctrl_code
            if mc.any():
                CA += blk[mc & (hb == 0)].sum(0); CB += blk[mc & (hb == 1)].sum(0)

    def norm(v):
        t = v.sum()
        return np.log1p(v / (t if t > 0 else 1.0) * 1e4)
    ca, cb_ = norm(CA), norm(CB)
    # a single pooled sigma is fine here: we compare A vs B, and any shared scaling cancels in a correlation
    sd = np.abs(ca - cb_).std() + 1e-6

    rows = []
    for i in keep:
        za = (norm(SA[i]) - ca) / sd; zb = (norm(SB[i]) - cb_) / sd
        a_, b_ = za - za.mean(), zb - zb.mean()
        d = np.linalg.norm(a_) * np.linalg.norm(b_)
        if d < 1e-9:
            continue
        r = float(a_ @ b_ / d)
        ma, mb = np.abs(za) >= THR, np.abs(zb) >= THR
        rep = float((ma & mb).sum() / max(ma.sum(), 1)) if ma.sum() else np.nan
        rows.append((cats[i], counts[i], r, rep, int(ma.sum())))

    R = np.array([x[2] for x in rows]); NC = np.array([x[1] for x in rows])
    REP = np.array([x[3] for x in rows]); MV = np.array([x[4] for x in rows])
    strong = MV >= 20
    r_half = float(np.nanmedian(R[strong])) if strong.any() else float(np.nanmedian(R))
    r_full = 2 * r_half / (1 + r_half)            # Spearman-Brown: half-depth -> full depth
    print(f"  split-half reliability (knockouts with >=20 movers, n={int(strong.sum())}): median r = {r_half:.3f}")
    print(f"  Spearman-Brown corrected to full depth:                                    r = {r_full:.3f}")
    print(f"  => attenuation ceiling on ANY predictor's correlation with the profile:  sqrt(r) = {np.sqrt(max(r_full,0)):.3f}")
    rep_med = float(np.nanmedian(REP[strong])) if strong.any() else float(np.nanmedian(REP))
    print(f"  MOVER reproducibility: of genes called movers in half A, {100*rep_med:.0f}% are movers in half B")
    print(f"\n  reliability vs cell count (how many cells a trustworthy profile needs):")
    print(f"    {'cells/KO':>12s} {'n':>5s} {'median r (half)':>16s} {'mover repro':>12s}")
    edges = [100, 150, 250, 400, 700, 1200, 10 ** 9]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (NC >= lo) & (NC < hi) & strong
        if m.sum() >= 5:
            bands.append((lo, hi, int(m.sum()), float(np.nanmedian(R[m])), float(np.nanmedian(REP[m]))))
            print(f"    {str(lo)+'-'+(str(hi) if hi<10**9 else '+'):>12s} {int(m.sum()):>5d} {np.nanmedian(R[m]):>16.3f} "
                  f"{100*np.nanmedian(REP[m]):>11.0f}%")
    return {"line": line, "n_ko": len(rows), "r_half": r_half, "r_full": r_full,
            "ceiling_sqrt_r": float(np.sqrt(max(r_full, 0))), "mover_reproducibility": rep_med,
            "bands": bands}


def main():
    lines = [x for x in (sys.argv[1:] or ["RPE1"]) if x in FILES]
    res = [run(ln) for ln in lines]
    r = res[0]
    noisy = r["r_full"] < 0.5
    verdict = (
        "RELIABILITY CEILING (reliability_ceiling.py): the replicate experiment run computationally -- each knockout's CELLS are split into two "
        "halves and treated as independent replicates, which measures test-retest reliability without new data. This matters because reliability "
        "sets a hard attenuation bound: no predictor can correlate with a measurement better than sqrt(its reliability). "
        + "; ".join(f"{x['line']}: split-half r={x['r_half']:.3f}, Spearman-Brown full-depth r={x['r_full']:.3f}, "
                    f"ceiling sqrt(r)={x['ceiling_sqrt_r']:.3f}, mover reproducibility {100*x['mover_reproducibility']:.0f}%" for x in res)
        + ". "
        + ("THE CEILING IS SUBSTANTIALLY NOISE. Reliability is low, so a large part of what looked like unpredictable biology is simply an "
           "unstable measurement -- and the cheapest way to raise every number in this project is to MEASURE MORE CELLS PER KNOCKOUT, not to "
           "build better models or new assays. " if noisy else
           "THE MEASUREMENT IS RELIABLE. Knockout profiles reproduce well between independent halves of the cells, so the prediction ceiling is "
           "NOT mainly measurement noise -- it is genuine biological specificity, and only new perturbations or modalities can move it. ")
        + "The reliability-vs-cell-count table says directly how deep a screen must be for a trustworthy profile, which is the single most "
        "actionable design number produced in this project. Deterministic; halves assigned by a fixed seed, mover threshold |z|>=4.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"results": res, "ceiling_is_mostly_noise": bool(noisy), "verdict": verdict, "note": verdict},
              open(OUT / "reliability_ceiling.json", "w"), indent=1)
    print("\n  -> outputs/orphan/reliability_ceiling.json")


if __name__ == "__main__":
    main()
