"""Loop 196. A response-time statistic that survives four timepoints -- chosen before the answer.

WHAT LOOP 192 ESTABLISHED, AND WHY IT BLOCKS EVERYTHING. The accessibility clock -- promoter
accessibility reaching half its plateau before the mRNA does -- cannot be tested on the
dendritic-cell LPS series with the statistic loop 191d used. W3 measured that directly: on A549's
full 11-point grid the lead is +154 min at p 1.1e-105 over 1,023 genes; downsampled to the
dendritic-cell 4-point shape it becomes -61 min at p 0.947 over 211. The effect does not weaken, it
REVERSES. So W4 was VOID and the replication is unanswered rather than negative.

The dendritic-cell series itself is fine: one lab for all 59 experiments, a graded clock with no
interval carrying more than 71% of the median response, and a donor split-half of +0.366. The
problem is the statistic. Half-of-plateau needs a crossing to interpolate, and it estimates the
plateau from the last three points -- on a four-point grid that is three of the four, so the
statistic spends almost its whole trajectory defining its own endpoint and has one interval left in
which to find a crossing. It was designed for nine points and borrowed for four.

THE DISCIPLINE THIS LOOP IS BUILT AROUND, and it is the point of the loop. Choosing a statistic by
trying several and keeping the one that gives a nice answer on the dendritic-cell data is how a
replication becomes a fishing expedition. So the selection happens ENTIRELY on A549, where the
answer is already known and was measured before any of these candidates existed, and the winner is
fixed before the dendritic-cell lead is computed once. X2 through X5 never touch the LPS series
except for reliability, which is a property of the statistic and not of the answer. X6 runs once.

THE CANDIDATES, declared here before any is scored, with what each is for.

  half_plateau   the incumbent. Time at which the signed trajectory crosses half its steady state,
                 linearly interpolated (Rosenfeld, Elowitz & Alon's definition of response time).
                 Included so the comparison has a baseline and so its failure is on the record
                 rather than assumed.

  centroid       the increment-weighted mean time: sum over intervals of (midpoint x increment)
                 divided by total increment. Uses every point, needs no crossing, and degrades
                 gracefully as points are removed because removing a point merges two intervals
                 rather than destroying a threshold.

  mrt            mean residence time, the classical control-theory response time:
                 integral of (1 - v(t)/plateau) dt over the observed window, by trapezoid. This is
                 what "response time" means for a first-order system and it is defined by an area
                 rather than by a level crossing, which is exactly the property four points need.

  tau            a one-parameter exponential fit, v(t) = plateau x (1 - exp(-(t-t0)/tau)), tau
                 found by search with the plateau fixed. Principled if the kinetics are first
                 order, and fragile if they are not; included so that assumption is tested rather
                 than relied upon.

PREDECLARED, BEFORE ANY NUMBER.

  X1 DO THE CANDIDATES AGREE WITH EACH OTHER WHERE THE DATA IS RICH? All four computed on A549's
     full grid and correlated pairwise.
     Gate: descriptive. A candidate that disagrees with the incumbent on eleven points is measuring
     something else, and that has to be visible before its behaviour on four points is interpreted.

  X2 THE SANITY FLOOR. Each candidate must recover the A549 lead on the FULL grid, where it is
     known to exist.
     Gate: PASS iff at least one candidate gives a positive lead at one-sided p < 0.05. A candidate
     that fails here is disqualified regardless of what it does on four points -- a statistic that
     cannot see the effect where it is strongest is not a coarse-grid solution.

  X3 THE SELECTION. Every candidate surviving X2, run on A549 downsampled to the dendritic-cell
     4-point shape.
     Gate: PASS iff at least one survivor keeps the lead positive at one-sided p < 0.05. This is
     the gate loop 192's W3 failed with the incumbent alone.

  X4 IS THE WINNER ROBUST TO WHICH FOUR POINTS? A statistic that works on [60,120,240,420] and not
     on other four-point subsets is fitting the subset, not solving the resolution problem.
     Gate: PASS iff the winner keeps a positive significant lead on at least 3 of the 4 subsets
     tested.

  X5 IS THE WINNER RELIABLE ON THE TARGET SERIES? Donor split-half of the winner's response time on
     the dendritic-cell data. This is a property of the statistic, not of the lead, so computing it
     before X6 does not leak the answer.
     Gate: PASS iff Spearman >= 0.30.

  X6 THE REPLICATION, RUN ONCE. The winner, fixed by X2-X5, applied to dendritic-cell ATAC against
     RNA on the shared grid.
     Gate: PASS iff the lead is positive at one-sided p < 0.05 AND holds in at least 2 of 3
     magnitude terciles.

  X7 THE STRANGER SWAP on the X6 result.
     Gate: PASS iff the real lead exceeds the swapped lead at p < 0.05.

  X8 WHAT THIS CANNOT SHOW.

-> outputs/loop_timing_statistic.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
import loop_response_timing_d as L191        # noqa: E402
import loop_timing_replication as L192       # noqa: E402

from scipy.stats import spearmanr, wilcoxon                       # noqa: E402

SP = L191.SP
DC = SP / "dclps"
A549 = SP / "grtc"
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_timing_statistic.json"
MIN_PLATEAU = L192.MIN_PLATEAU
MIN_TPM = L192.MIN_TPM
MIN_GROUP = L192.MIN_GROUP
ALPHA = L192.ALPHA
N_STRATA = 3
MIN_SPLIT_RHO = 0.30
PROM_PAD = L191.PROM_PAD
MIN_SUBSETS = 3
SEED = 196196

# four-point subsets of the A549 grid, all with dendritic-cell-like spacing. The first is loop
# 192's; the others exist so a winner cannot be a winner on one lucky choice of points.
SUBSETS = ([60.0, 120.0, 240.0, 420.0],
           [60.0, 120.0, 180.0, 360.0],
           [30.0, 120.0, 240.0, 480.0],
           [60.0, 180.0, 300.0, 600.0])

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


# ---------------------------------------------------------------------------------------------
# the candidates. Each takes a (n_time, n_gene) trajectory ALREADY baseline-subtracted, and the
# grid, and returns one time per gene plus the plateau it used.
# ---------------------------------------------------------------------------------------------
def _plateau(V):
    k = min(3, max(1, V.shape[0] - 1))
    return V[-k:].mean(0)


def st_half_plateau(V, grid):
    pl = _plateau(V)
    out = np.full(V.shape[1], np.nan)
    for j in range(V.shape[1]):
        p = pl[j]
        if abs(p) < 1e-9:
            continue
        tg, v = p / 2.0, V[:, j]
        for i in range(1, len(v)):
            if (p > 0 and v[i] >= tg) or (p < 0 and v[i] <= tg):
                lo, hi = v[i - 1], v[i]
                out[j] = grid[i] if hi == lo else \
                    grid[i - 1] + (tg - lo) / (hi - lo) * (grid[i] - grid[i - 1])
                break
    return out, pl


def st_centroid(V, grid):
    """Increment-weighted mean time. Removing a point merges two intervals instead of destroying a
    threshold, which is why this should degrade gracefully."""
    pl = _plateau(V)
    d = np.diff(V, axis=0)
    mid = ((grid[1:] + grid[:-1]) / 2.0)[:, None]
    tot = d.sum(0)
    out = np.where(np.abs(tot) > 1e-9, (mid * d).sum(0) / np.where(np.abs(tot) > 1e-9, tot, 1.0),
                   np.nan)
    return out, pl


def st_mrt(V, grid):
    """Mean residence time: the area above the normalised approach curve.

    For a first-order system this IS the response time, and it is defined by an integral rather
    than by a level crossing -- the property a four-point grid needs."""
    pl = _plateau(V)
    safe = np.where(np.abs(pl) > 1e-9, pl, np.nan)
    y = 1.0 - V / safe                              # 1 at baseline, 0 at plateau
    out = np.trapezoid(y, x=grid, axis=0)
    return np.where(np.isfinite(out), out, np.nan), pl


def st_tau(V, grid, n_grid=60):
    """One-parameter exponential fit with the plateau fixed; tau by search.

    Fragile if the kinetics are not first order, which is the point of including it."""
    pl = _plateau(V)
    span = float(grid[-1] - grid[0])
    taus = np.geomspace(max(span / 200.0, 1e-3), span * 3.0, n_grid)
    t = (grid - grid[0])[:, None]
    best = np.full(V.shape[1], np.nan)
    err = np.full(V.shape[1], np.inf)
    safe = np.where(np.abs(pl) > 1e-9, pl, np.nan)
    Y = V / safe
    for tau in taus:
        pred = 1.0 - np.exp(-t / tau)
        e = np.nansum((Y - pred) ** 2, axis=0)
        m = e < err
        err[m], best[m] = e[m], tau
    best[~np.isfinite(pl) | (np.abs(pl) < 1e-9)] = np.nan
    return best, pl


CANDIDATES = {"half_plateau": st_half_plateau, "centroid": st_centroid,
              "mrt": st_mrt, "tau": st_tau}


def load(dirp, assay):
    z = np.load(dirp / "rna.npz", allow_pickle=True)
    man = json.load(open(dirp / "manifest.json"))
    return dict(tpm=z["tpm"], ensg=np.array([str(g).split(".")[0] for g in z["genes"]]),
                mins=z["mins"].astype(int), reps=z["reps"].astype(int),
                exps=np.array([str(x) for x in z["exps"]]), man=man, assay=assay)


def tss_table():
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    out = {}
    for line in open(SP / "_tss_hg38.bed"):
        q = line.split()
        if len(q) >= 4 and q[3].startswith("G"):
            i = int(q[3][1:])
            if i < len(tab):
                out[str(tab[i]["name"]).upper()] = (q[0], int(q[2]))
    return out


_PT_CACHE = {}


def promoter_cached(assay, tsslist):
    """promoter_track re-parses every peak file on each call and does not depend on the grid.

    X4 alone evaluates several candidates across four subsets, so without this the same nine to
    eleven bed files are parsed twenty times. The key is the assay plus the TSS list identity,
    which is constant within a series."""
    key = (assay, id(tsslist))
    if key not in _PT_CACHE:
        _PT_CACHE[key] = L191.promoter_track(assay, tsslist, PROM_PAD, lambda *_: None)
    return _PT_CACHE[key]


def arm(S, grid, e2s, tss, stat, group=None):
    """One (statistic, grid) evaluation: returns accessibility time, expression time, mask, plateau."""
    reps = S["reps"] if group is None else group
    keep = [r for r in sorted(set(reps.tolist()))
            if all(((S["mins"] == int(t)) & (reps == r)).any() for t in grid)]
    if not keep:
        return None
    M, _ = L191.rep_trajectories(S["tpm"], S["mins"], reps, keep, grid)
    sym = np.array([e2s.get(g, "") for g in S["ensg"]])
    base = S["tpm"][(S["mins"] == int(grid[0])) & np.isin(reps, keep)].mean(0)
    eh, pl = stat(M, grid)
    tl = [tss.get(s) for s in sym]
    pt, PM = promoter_cached(S["assay"], S.setdefault("_tl", tl))
    have = set(pt.tolist())
    if not set(grid.tolist()) <= have:
        return None
    idx = [int(np.where(pt == t)[0][0]) for t in grid]
    P0 = PM[idx] - PM[idx][0]                     # accessibility on the same baseline convention
    ah, _ = stat(P0, grid)
    resp = (base >= MIN_TPM) & (np.abs(pl) >= MIN_PLATEAU) & np.isfinite(eh)
    m = resp & (PM[idx] > 0).any(0) & np.isfinite(ah)
    return dict(ah=ah, eh=eh, mask=m, pl=pl, sym=sym)


def lead(a, label, report=print):
    if a is None or a["mask"].sum() < MIN_GROUP:
        report(f"     {label}: unavailable or under the power floor")
        return None
    m = a["mask"]
    _, p = wilcoxon(a["ah"][m], a["eh"][m], alternative="less")
    ld = float(np.median(a["eh"][m] - a["ah"][m]))
    report(f"     {label:34s} n {int(m.sum()):5,}  lead {ld:+8.1f}  p {p:.3g}")
    return dict(n=int(m.sum()), lead=ld, p=float(p))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 196  A RESPONSE-TIME STATISTIC THAT SURVIVES FOUR TIMEPOINTS")
    say("=" * 104)
    say("  PREDECLARED: the statistic is selected ENTIRELY on A549, where the answer was measured")
    say("  before any of these candidates existed; the dendritic-cell lead is computed ONCE, after")
    say("  the winner is fixed. A candidate must recover the lead on the full grid (X2) before its")
    say("  four-point behaviour means anything, must survive downsampling (X3), must hold on at")
    say(f"  least {MIN_SUBSETS} of {len(SUBSETS)} different four-point subsets (X4) so it is not")
    say("  fitting one lucky choice, and must be reliable across donors (X5). Only then X6.")
    say()

    e2s = L191.ensg_to_symbol(say)
    tss = tss_table()
    A = load(A549, "DNase")
    D = load(DC, "ATAC")
    a_grid = np.array(sorted(set(A["mins"].tolist()) &
                             {int(k) for k in A["man"]["peaks"]["DNase"]}), dtype=float)
    a_grid = a_grid[a_grid >= 30.0]
    d_grid = np.array(sorted(set(D["mins"].tolist()) &
                             {int(k) for k in D["man"]["peaks"]["ATAC"]}), dtype=float)
    say(f"    A549 grid  {[int(x) for x in a_grid]}")
    say(f"    dendritic  {[int(x) for x in d_grid]}")

    # ---- X1 ------------------------------------------------------------------------------------
    say()
    say("X1 DO THE CANDIDATES AGREE WHERE THE DATA IS RICH?")
    full = {k: arm(A, a_grid, e2s, tss, f) for k, f in CANDIDATES.items()}
    ref = full["half_plateau"]
    agree = {}
    for k, a in full.items():
        if a is None or ref is None:
            continue
        m = ref["mask"] & a["mask"] & np.isfinite(ref["eh"]) & np.isfinite(a["eh"])
        r, _ = spearmanr(ref["eh"][m], a["eh"][m]) if m.sum() > 10 else (np.nan, np.nan)
        agree[k] = float(r)
        say(f"     {k:14s} vs half_plateau on expression times: Spearman {r:+.3f} "
            f"(n {int(m.sum()):,})")
    say("     X1 (descriptive)")

    # ---- X2 ------------------------------------------------------------------------------------
    say()
    say("X2 THE SANITY FLOOR: does each candidate see the lead where it is strongest?")
    d2 = {}
    for k in CANDIDATES:
        d2[k] = lead(full[k], f"A549 full grid, {k}", say)
    survivors = [k for k, v in d2.items() if v and v["p"] < ALPHA and v["lead"] > 0]
    x2 = bool(survivors)
    GG.verdict(x2, emit=say,
               if_true=f"X2 PASS -- {len(survivors)} of {len(CANDIDATES)} recover the lead on "
                       f"eleven points: {survivors}",
               if_false="X2 FAIL -- no candidate sees the lead on the full grid, which would mean "
                        "the effect itself is not robust to how it is measured")

    # ---- X3 ------------------------------------------------------------------------------------
    say()
    say("X3 THE SELECTION: the same candidates on A549 downsampled to the dendritic-cell shape")
    sub0 = np.array([t for t in SUBSETS[0] if t in set(a_grid.tolist())], dtype=float)
    say(f"     downsampled grid {[int(x) for x in sub0]}")
    d3 = {}
    for k in survivors:
        d3[k] = lead(arm(A, sub0, e2s, tss, CANDIDATES[k]), f"A549 downsampled, {k}", say)
    passed = [k for k, v in d3.items() if v and v["p"] < ALPHA and v["lead"] > 0]
    x3 = bool(passed)
    GG.verdict(x3, emit=say,
               if_true=f"X3 PASS -- {passed} keep the lead on four points, where loop 192's "
                       f"incumbent reversed to -61 min at p 0.947",
               if_false="X3 FAIL -- no candidate survives four points. The dendritic-cell series "
                        "cannot answer the question with any statistic tried here, and the "
                        "replication needs a denser series rather than a better estimator")

    void = set()

    # ---- X4 ------------------------------------------------------------------------------------
    say()
    say("X4 IS THE WINNER ROBUST TO WHICH FOUR POINTS?")
    d4, winner = {}, None
    x4_void = not x3
    if x4_void:
        void.add("X4")
        say("     X4 VOID -- nothing survived X3, so there is no candidate to test for robustness")
    else:
        for k in passed:
            held, rows = 0, []
            for ss in SUBSETS:
                g = np.array([t for t in ss if t in set(a_grid.tolist())], dtype=float)
                if len(g) < 4:
                    rows.append(None)
                    continue
                r = lead(arm(A, g, e2s, tss, CANDIDATES[k]),
                         f"  {k} on {[int(x) for x in g]}", say)
                rows.append(r)
                held += int(bool(r and r["p"] < ALPHA and r["lead"] > 0))
            d4[k] = dict(held=held, rows=rows)
            say(f"     {k}: holds on {held}/{len(SUBSETS)} subsets")
        ok = {k: v["held"] for k, v in d4.items() if v["held"] >= MIN_SUBSETS}
        winner = max(ok, key=ok.get) if ok else None
    x4 = winner is not None
    # both branches of GG.verdict are f-strings and BOTH are evaluated before the call, so a PASS
    # message that indexes a success-only value crashes on failure. It did: d4[None]. Any gate
    # whose success text references something that exists only on success needs this guard.
    held_txt = f"{d4[winner]['held']}/{len(SUBSETS)}" if winner is not None else "n/a"
    if not x4_void:
        GG.verdict(x4, emit=say,
                   if_true=f"X4 PASS -- '{winner}' holds on {held_txt} four-point subsets, "
                           f"so it is solving the resolution problem rather than fitting one "
                           f"choice of points",
                   if_false=f"X4 FAIL -- no candidate holds on {MIN_SUBSETS} of {len(SUBSETS)} "
                            f"subsets; a statistic that works on one is fitting that one")

    if not (x2 and x3 and x4):
        void |= {"X5", "X6", "X7"}
        say()
        say("     selection failed, so X5-X7 are VOID: there is no fixed statistic to apply")

    # ---- X5 ------------------------------------------------------------------------------------
    say()
    say("X5 IS THE WINNER RELIABLE ON THE TARGET SERIES?")
    x5, d5 = False, {}
    if "X5" in void:
        say("     X5 VOID -- see above")
    else:
        grp = L192.donor_split(D["mins"], D["exps"])
        a1 = arm(D, d_grid, e2s, tss, CANDIDATES[winner], group=np.where(grp == 1, 1, 0))
        a2 = arm(D, d_grid, e2s, tss, CANDIDATES[winner], group=np.where(grp == 2, 1, 0))
        if a1 is None or a2 is None:
            void.add("X5")
            say("     X5 VOID -- a donor half does not cover the grid")
        else:
            m = a1["mask"] & a2["mask"]
            r, p = spearmanr(a1["eh"][m], a2["eh"][m]) if m.sum() > 10 else (np.nan, np.nan)
            say(f"     donor split-half of '{winner}' expression times: Spearman {r:+.3f} "
                f"(n {int(m.sum()):,}, p {p:.3g})")
            d5 = dict(rho=float(r), p=float(p), n=int(m.sum()))
            x5 = bool(np.isfinite(r) and r >= MIN_SPLIT_RHO)
            GG.verdict(x5, emit=say,
                       if_true=f"X5 PASS -- {r:+.3f}",
                       if_false=f"X5 FAIL -- {r:+.3f} against {MIN_SPLIT_RHO}; the winner is not "
                                f"reliable on this series whatever it does on A549")
    if not x5:
        void |= {"X6", "X7"}

    # ---- X6 ------------------------------------------------------------------------------------
    say()
    say("X6 THE REPLICATION, RUN ONCE")
    x6, d6, strata = False, None, {}
    if "X6" in void:
        say("     X6 VOID -- see above")
    else:
        say(f"     statistic fixed by X2-X5: '{winner}'. this is the first time the "
            f"dendritic-cell LEAD is computed.")
        a = arm(D, d_grid, e2s, tss, CANDIDATES[winner])
        d6 = lead(a, "dendritic cell + LPS", say)
        if d6 is None:
            void.add("X6")
        else:
            edges = np.quantile(np.abs(a["pl"][a["mask"]]), np.linspace(0, 1, N_STRATA + 1))
            edges[-1] += 1e-9
            held = 0
            for qi in range(N_STRATA):
                st = a["mask"] & (np.abs(a["pl"]) >= edges[qi]) & (np.abs(a["pl"]) < edges[qi + 1])
                r = lead(dict(ah=a["ah"], eh=a["eh"], mask=st, pl=a["pl"], sym=a["sym"]),
                         f"  tercile {qi+1}", say)
                held += int(bool(r and r["p"] < ALPHA and r["lead"] > 0))
            strata["X6"] = held
            x6 = bool(d6["p"] < ALPHA and d6["lead"] > 0 and held >= 2)
            GG.verdict(x6, emit=say,
                       if_true=f"X6 PASS -- accessibility leads by {d6['lead']:+.1f} in a "
                               f"different cell type, stimulus and assay, holding in {held}/3 "
                               f"terciles. The clock is not an A549 or dexamethasone artefact",
                       if_false=f"X6 FAIL -- lead {d6['lead']:+.1f} at p {d6['p']:.3g}, {held}/3 "
                                f"terciles. X3 and X4 established the statistic CAN see this "
                                f"effect on four points, so this is an absence and not a limit")

    # ---- X7 ------------------------------------------------------------------------------------
    say()
    say("X7 THE STRANGER SWAP")
    x7, d7 = False, None
    if "X7" in void or d6 is None:
        say("     X7 VOID -- see above")
        void.add("X7")
    else:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(a["ah"]))
        m = a["mask"]
        _, p7 = wilcoxon(a["eh"][m] - a["ah"][perm][m], a["eh"][m] - a["ah"][m],
                         alternative="less")
        sw = float(np.median(a["eh"][m] - a["ah"][perm][m]))
        say(f"     real lead {d6['lead']:+.1f} vs swapped {sw:+.1f}; real exceeds swapped "
            f"at p {p7:.3g}")
        d7 = dict(swapped=sw, p=float(p7))
        x7 = bool(p7 < ALPHA)
        GG.verdict(x7, emit=say,
                   if_true="X7 PASS -- the lead belongs to the gene's own promoter",
                   if_false="X7 FAIL -- a stranger's accessibility leads just as well, so the "
                            "lead is a property of the two assay grids")

    # ---- X8 ------------------------------------------------------------------------------------
    say()
    say("X8 WHAT THIS CANNOT SHOW")
    say("     The winner was selected on A549 and A549 alone. That protects against choosing a")
    say("     statistic by its dendritic-cell answer; it does NOT protect against choosing one")
    say("     that suits A549's particular kinetics. A statistic tuned on one system and applied")
    say("     to another is a weaker instrument than one derived from first principles, and the")
    say("     four candidates here are conveniences rather than a theory of transcription timing.")
    say("     X4 varies which four points are used but every subset comes from the same A549")
    say("     series, so it tests robustness to sampling and not to biology.")
    say("     mrt and centroid are defined over the OBSERVED window, so both are bounded by it. A")
    say("     gene still moving at the last timepoint has its response time underestimated, and")
    say("     that bias is larger on the shorter dendritic-cell window than on A549's.")
    say("     Leading is still not causing. Nothing here perturbs accessibility.")
    say("     A pass replicates the CLOCK only. Loop 191d's other findings -- feedback sign not")
    say("     ordering response times, occupancy not carrying timing once size is controlled --")
    say("     remain single-system results and are untouched by this loop.")
    say("     X8 PASS")

    gates = {"X1": True, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "X6": x6, "X7": x7, "X8": True}
    man_out = RM.manifest(inputs=[A549 / "rna.npz", DC / "rna.npz"],
                          available=int(len(A["ensg"])), used=int(len(D["ensg"])),
                          selection="filtered", seed=SEED,
                          controls=["the statistic selected on A549 only, before the DC lead exists",
                                    f"{len(SUBSETS)} different four-point subsets",
                                    "donor split-half reliability of the winner",
                                    "magnitude terciles", "a stranger swap"],
                          note="a response-time statistic that survives four timepoints")
    out_d = dict(test="timing statistic selection", gates=gates, void=sorted(void),
                 winner=winner, agreement=agree, x2=d2, x3=d3,
                 x4={k: dict(held=v["held"]) for k, v in d4.items()},
                 x5=d5, x6=d6, x7=d7, strata=strata,
                 a549_grid=[int(x) for x in a_grid], dc_grid=[int(x) for x in d_grid],
                 subsets=[[int(x) for x in s] for s in SUBSETS],
                 manifest=man_out, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'VOID' if k in void else ('PASS' if v else 'FAIL')}")
    scored = [k for k in gates if k not in void]
    say(f"  {sum(gates[k] for k in scored)}/{len(scored)}   [{time.time()-t0:.0f}s]"
        + (f"   ({len(void)} VOID: {', '.join(sorted(void))})" if void else ""))
    say("=" * 104)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
