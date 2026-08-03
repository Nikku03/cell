"""STEP 4b: decompose G2. How much of the state gain is CELL STATE, and how much is just knowing the CELL LINE?

THE ERROR THIS FIXES, which is mine.  sc_adrn.py bundled the state block as [s, g2m, log-depth, bin one-hot,
LINE one-hot]. `gene_state` therefore knew which of RPE1 / Jurkat / HepG2 a group came from and `gene_only` did
not -- while the target differs substantially by line. So the +0.050 p@20 attributed to "state" conflates two
completely different things:

    knowing the CELL LINE   a context label available for free, nothing to do with single-cell data
    knowing the CELL STATE  the cycle position of the specific cells, which ONLY single-cell data provides

The whole justification for the single-cell route rests on the second. Reporting the combined number as evidence
for it would have been an overstatement, so the block is split into nested arms and each increment priced:

    gene              gene channels only
    gene_line         + line one-hot            <- context that needs no single-cell data at all
    gene_line_bin     + state-bin one-hot       <- which tertile, coarse
    gene_line_state   + s, g2m, log-depth       <- the continuous state the cells actually carry
    gene_line_nodepth + s, g2m only             <- depth dropped, because the gate showed depth alone gives
                                                   ratios of 1.14-1.28 and is partly technical

PREDECLARED, and this is the number that decides whether the single-cell route earns its keep:

    G2a  gene_line       > gene              line identity helps (expected, and NOT a single-cell result)
    G2b  gene_line_state > gene_line         cell state helps BEYOND line. THE CLAIM.
    G2c  gene_line_nodepth > gene_line       the same without the depth covariate, so a pass cannot be
                                             attributed to sequencing depth leaking in as a proxy.

Swept over state bins and holdout seed -- NOT over the SVD rank, which sc_adrn.py showed is inert for these arms
and produced eight duplicate cells that inflated the verdict from 4 real observations to 12.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
import os
# bins swept WIDE deliberately. The 2-vs-3 run showed state beyond line is positive and tight at 3
# bins (+0.021..+0.024 on all 6 cells) and RESOLVED NEGATIVE at 2. That is either a real resolution
# effect -- too few bins cannot represent state, so the features add variance without information --
# or a corner of the grid. Monotone improvement with bin count supports the first; a peak at 3 alone
# supports the second. This is the falsifier for the only lead the decomposition left alive.
BINS_SWEEP = tuple(int(x) for x in os.environ.get('SC_BINS', '2,3').split(','))
SEED_SWEEP = (0, 1, 2)
N_HOLDOUT, MIN_GROUP, PENALTIES, NPRED = 40, 25, (1.0, 10.0, 100.0, 1000.0), 20


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("STEP 4b -- DECOMPOSE: is the gain cell STATE, or just cell LINE?")
    report("=" * 100)
    report("  PREDECLARED  G2a line>gene | G2b state>line THE CLAIM | G2c state-without-depth>line")

    z = np.load(SP / "sc_cohort_cells.npz", allow_pickle=True)
    X = z["X"]
    line = np.array([str(x) for x in z["line"]])
    pert = np.array([str(x) for x in z["pert"]])
    batch, ntc = z["batch"], z["is_ntc"]
    s_sc, g2, umi = (z["s_score"].astype(np.float64), z["g2m_score"].astype(np.float64),
                     z["umi"].astype(np.float64))
    perts = sorted(set(pert[~ntc]))
    lines = sorted(set(line))

    zk = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in zk["kos"]]
    universe = sorted(set(kos) | set(perts))
    base, _ = A.build_channels(universe, set(kos), report)
    extra, _, tb = C.extra_channels(universe, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    Cg = chan[[urow[g] for g in perts]]
    pidx = {p: i for i, p in enumerate(perts)}

    sws = {n: Sweeper(n, axes={"bins": list(BINS_SWEEP), "seed": list(SEED_SWEEP)})
           for n in ("G2a gene_line - gene", "G2b gene_line_state - gene_line",
                     "G2c gene_line_nodepth - gene_line")}
    means = {}

    for nbins in BINS_SWEEP:
        binid = np.zeros(len(X), np.int64)
        for ln in lines:
            m = line == ln
            q = np.quantile(g2[m], np.linspace(0, 1, nbins + 1)[1:-1])
            binid[m] = np.digitize(g2[m], q)

        groups, targets, st_rows = [], [], []
        for ln in lines:
            for p in perts:
                for b in range(nbins):
                    m = (line == ln) & (pert == p) & ~ntc & (binid == b)
                    if m.sum() < MIN_GROUP:
                        continue
                    bs = set(batch[m].tolist())
                    cm = (line == ln) & ntc & np.isin(batch, list(bs)) & (binid == b)
                    if cm.sum() < 15:
                        continue
                    groups.append((ln, p, b))
                    targets.append(X[m].mean(0) - X[cm].mean(0))
                    st_rows.append([float(s_sc[m].mean()), float(g2[m].mean()),
                                    float(np.log10(umi[m].mean()))])
        Y = np.array(targets, np.float32)
        S3 = np.array(st_rows, np.float32)
        gp = np.array([g[1] for g in groups])
        gl = np.array([g[0] for g in groups])
        gb = np.array([g[2] for g in groups])
        L1 = np.eye(len(lines), dtype=np.float32)[[lines.index(x) for x in gl]]
        B1 = np.eye(nbins, dtype=np.float32)[gb]
        G = Cg[[pidx[p] for p in gp]]
        ones = np.ones((len(groups), 1), np.float32)
        report(f"\n  bins={nbins}: {len(groups):,} groups")

        for seed in SEED_SWEEP:
            rr = np.random.default_rng(A.SEED + 17 * seed)
            hold = set(np.array(perts)[rr.choice(len(perts), N_HOLDOUT, replace=False)].tolist())
            hm = np.isin(gp, list(hold))
            Sz = (S3 - S3[~hm].mean(0)) / (S3[~hm].std(0) + 1e-6)

            def fit_predict(F):
                r2 = np.random.default_rng(A.SEED + 1).permutation(int((~hm).sum()))
                nv = max(20, int(0.2 * len(r2)))
                Ftr, Ytr = F[~hm], Y[~hm]
                best, bp = -1e18, PENALTIES[0]
                for pen in PENALTIES:
                    w = A.ridge_fit(Ftr[r2[nv:]], Ytr[r2[nv:]], pen)
                    e = -float(np.mean((A.ridge_apply(Ftr[r2[:nv]], w) - Ytr[r2[:nv]]) ** 2))
                    if e > best:
                        best, bp = e, pen
                return A.ridge_apply(F[hm], A.ridge_fit(Ftr, Ytr, bp))

            Yte = Y[hm]

            def score(P):
                r = np.array([float(np.corrcoef(P[i], Yte[i])[0, 1]) if np.std(P[i]) > 0 else 0.0
                              for i in range(len(Yte))])
                pr = np.array([len(set(np.argsort(-np.abs(P[i]))[:NPRED].tolist())
                                   & set(np.argsort(-np.abs(Yte[i]))[:NPRED].tolist())) / float(NPRED)
                               for i in range(len(Yte))])
                return np.nan_to_num(r), pr

            F = {"gene": np.concatenate([ones, G], 1),
                 "gene_line": np.concatenate([ones, G, L1], 1),
                 "gene_line_bin": np.concatenate([ones, G, L1, B1], 1),
                 "gene_line_state": np.concatenate([ones, G, L1, B1, Sz], 1),
                 "gene_line_nodepth": np.concatenate([ones, G, L1, B1, Sz[:, :2]], 1)}
            sc = {k: score(fit_predict(v)) for k, v in F.items()}
            for a, b in (("G2a gene_line - gene", ("gene_line", "gene")),
                         ("G2b gene_line_state - gene_line", ("gene_line_state", "gene_line")),
                         ("G2c gene_line_nodepth - gene_line", ("gene_line_nodepth", "gene_line"))):
                sws[a].add({"bins": nbins, "seed": seed}, f"r|{nbins}|{seed}", sc[b[0]][0] - sc[b[1]][0])
                sws[a].add({"bins": nbins, "seed": seed}, f"p|{nbins}|{seed}", sc[b[0]][1] - sc[b[1]][1])
            means[f"bins={nbins}|seed={seed}"] = {k: {"r": float(v[0].mean()), "p20": float(v[1].mean())}
                                                  for k, v in sc.items()}
            report("    seed %d  " % seed + "  ".join(
                f"{k}: r {v[0].mean():.4f}/p {v[1].mean():.4f}" for k, v in sc.items()))

    for n, s in sws.items():
        report(s.report())
    V = {n: s.verdict() for n, s in sws.items()}

    report("\n  READING")
    a = V["G2a gene_line - gene"]["verdict"]
    b = V["G2b gene_line_state - gene_line"]["verdict"]
    c = V["G2c gene_line_nodepth - gene_line"]["verdict"]
    report(f"  line identity: {a} | cell state beyond line: {b} | state without depth: {c}")
    if b in ("ROBUST", "DIRECTIONAL") and c in ("ROBUST", "DIRECTIONAL"):
        report("  CELL STATE EARNS ITS KEEP beyond line identity, and not via the depth covariate.")
        report("  The single-cell route is justified on prediction, not just on the raw interaction.")
    elif b in ("ROBUST", "DIRECTIONAL"):
        report("  State helps but the gain does NOT survive dropping depth -- it is substantially a")
        report("  sequencing-depth proxy, and the gate's umi ratios of 1.14-1.28 predicted exactly this.")
    else:
        report("  The G2 gain reported by sc_adrn.py was LINE IDENTITY, not cell state. Single-cell")
        report("  conditioning does not improve prediction of a held-out knockout, and pseudobulk stands.")

    json.dump({"test": "sc_adrn_decompose", "bins": list(BINS_SWEEP), "seeds": list(SEED_SWEEP),
               "means": means, "verdicts": V, "log": log},
              open(OUT / ("sc_adrn_decompose.json" if len(BINS_SWEEP) == 2 else "sc_adrn_decompose_bins.json"), "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'sc_adrn_decompose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
