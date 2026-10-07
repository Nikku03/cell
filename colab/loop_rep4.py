"""Loop 218. The fourth replicate: is it a rescue, or a fourth opinion?

WHAT WAS SITTING THERE. Every loop from 191d to 217 has used REPS = (1, 2, 3), a constant declared
in loop_response_timing_d.py and inherited unchanged ever since. The grid was defined as
"timepoints where replicates 1-3 all exist and t >= 30", so replicate 4 was never in the selection
criterion and never got picked up. It covers all nine grid points, and it also covers 300 and 360
minutes, which replicate 1 lacks and which the grid therefore drops.

WHY IT MATTERS RIGHT NOW. Loop 217's L7 concluded that no method could separate signal from noise
here, for a structural reason: with three replicates and replicate 1 an outlier (it agrees with 2
and 3 at pearson +0.12 while they agree with each other at +0.60), selecting on 1+2 inherits the
outlier and selecting on 2+3 is not independent of the scorer. A fourth replicate breaks that
deadlock -- IF it behaves. If replicate 4 is itself scattered then there are two odd replicates out
of four and the deadlock is worse, not better.

So this loop asks what replicate 4 is before asking what it buys, and it does the second only if
the first survives.

PREDECLARED, BEFORE ANY NUMBER.

  M1 IS REPLICATE 4 THE SAME EXPERIMENT?
     Gate: PASS iff replicate 4 covers all nine grid points, its baseline expression at t=30
     correlates with the other three at Spearman >= 0.90, and its library size is within 3x of
     theirs. FAIL means it is a different assay and must not be pooled.

  M2 WHERE DOES REPLICATE 4 SIT?
     All six pairwise agreements on the per-interval change, so replicate 4 is placed against the
     known structure rather than assumed into it.
     Gate: PASS iff replicate 4's mean pairwise correlation with 2 and 3 exceeds replicate 1's
     mean with 2 and 3. A PASS makes replicate 1 the lone outlier and licenses the clean split;
     a FAIL means there are two scattered replicates and M4 must not be read as a rescue.

  M3 WHAT DOES A FOUR-REPLICATE NOISE CEILING LOOK LIKE?
     The per-interval ceiling recomputed on all four, and on the three-replicate subsets, so the
     effect of dropping replicate 1 is separated from the effect of adding replicate 4.
     Gate: PASS iff the best three-replicate subset's ceiling exceeds the (1,2,3) ceiling of
     -0.54028 by more than 0.20.

  M4 THE CLEAN SPLIT, WHICH THREE REPLICATES COULD NOT PROVIDE.
     Select genes by signal-to-noise on replicates 2 and 3 ONLY; score the model and persistence
     against replicate 4, which took no part in the selection and shares no replicate with it.
     Gate: PASS iff the model beats persistence by more than 0.01 on the selected genes. This is
     loop 217's L4 with the confound removed, and it is the reason the loop exists.
     Requires M2 -- if replicate 4 is itself an outlier this split is not clean.

  M5 DOES THE DENSER GRID HELP?
     Replicates 2, 3 and 4 all cover 300 and 360 minutes, giving 11 grid points and 10 intervals
     instead of 9 and 8.
     Gate: PASS iff the eleven-point ceiling exceeds the nine-point ceiling on the same three
     replicates. Shorter intervals mean smaller true changes against the same noise, so this can
     easily go the other way and it is not predicted.

  M6 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, pickle, sys, time, warnings
from itertools import combinations
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
MODEL = ROOT / "colab" / "models" / "setpoint_stack_v1.pkl"
OUT = "outputs/loop_rep4.json"
GRID9 = [30, 60, 120, 180, 240, 420, 480, 600, 720]
GRID11 = [30, 60, 120, 180, 240, 300, 360, 420, 480, 600, 720]
REF123 = -0.54028
N_TRAIN, SEED = 6, 218218

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "the fourth replicate"}
    say("=" * 104)
    say("LOOP 218 -- THE FOURTH REPLICATE: A RESCUE, OR A FOURTH OPINION?")
    say("=" * 104)

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    art = pickle.load(open(MODEL, "rb"))
    names = art["genes"]
    pos = {s: k for k, s in enumerate([sym[i] for i in gi])}
    idx = gi[np.array([pos[s] for s in names])]
    S = np.array(art["stack_prediction"])
    ALL = (1, 2, 3, 4)

    # ---------------------------------------------------------------- M1
    say("M1 IS REPLICATE 4 THE SAME EXPERIMENT?")
    cov = {r: [t for t in GRID9 if ((mins == t) & (reps == r)).any()] for r in ALL}
    for r in ALL:
        say(f"       replicate {r}: covers {len(cov[r])}/9 grid points, "
            f"{len(set(mins[reps==r].tolist()))} timepoints overall")
    base = {}
    for r in ALL:
        ix = np.where((mins == 30) & (reps == r))[0]
        base[r] = tpm[ix].mean(0)
    lib = {r: float(np.sum(base[r])) for r in ALL}
    sp4 = [spear(base[4][idx], base[r][idx]) for r in (1, 2, 3)]
    say(f"       library size at t=30: " + "  ".join(f"r{r} {lib[r]:.3e}" for r in ALL))
    say(f"       baseline Spearman, replicate 4 against 1/2/3: "
        + "  ".join(f"{v:.4f}" for v in sp4))
    ratio = max(lib.values()) / min(lib.values())
    ok1 = (len(cov[4]) == 9 and min(sp4) >= 0.90 and ratio < 3.0)
    G.add("M1", ok1, stat=min(sp4),
          if_true=lambda: f"M1 PASS -- 9/9 grid points, baseline Spearman >= {min(sp4):.3f}, "
                          f"library sizes within {ratio:.2f}x",
          if_false=lambda: f"M1 FAIL -- cover {len(cov[4])}/9, min Spearman {min(sp4):.3f}, "
                           f"library ratio {ratio:.2f}x")

    # per-replicate interval changes on the 9-point grid
    def deltas(rs, gridpts):
        g = np.array(gridpts, float)
        out = {}
        for r in rs:
            Mi, n = L191.rep_trajectories(tpm, mins, reps, (r,), g)
            if n == 0:
                continue
            A = Mi[:, idx]
            out[r] = np.array([A[j] - A[j - 1] for j in range(1, len(g))])
        return out
    D9 = deltas(ALL, GRID9)

    # ---------------------------------------------------------------- M2
    say("M2 WHERE DOES REPLICATE 4 SIT?")
    prs = {}
    for a, b in combinations(ALL, 2):
        p = float(np.corrcoef(D9[a].ravel(), D9[b].ravel())[0, 1])
        prs[f"{a}v{b}"] = p
        say(f"       replicate {a} vs {b}   pearson {p:+.4f}   R2 {r2s(D9[b].ravel(), D9[a].ravel()):+.5f}")
    m1 = float(np.mean([prs["1v2"], prs["1v3"]]))
    m4 = float(np.mean([prs["2v4"], prs["3v4"]]))
    say(f"     replicate 1's mean agreement with 2 and 3: {m1:+.4f}")
    say(f"     replicate 4's mean agreement with 2 and 3: {m4:+.4f}")
    G.add("M2", bool(m4 > m1), stat=m4, requires=("M1",),
          if_true=lambda: f"M2 PASS -- replicate 4 agrees with 2 and 3 at {m4:+.4f} against "
                          f"replicate 1's {m1:+.4f}, so replicate 1 is the lone outlier and the "
                          f"clean split is available",
          if_false=lambda: f"M2 FAIL -- replicate 4 agrees at {m4:+.4f} against replicate 1's "
                           f"{m1:+.4f}. There are two scattered replicates and M4 is not a rescue")
    res["pairwise"] = prs

    # ---------------------------------------------------------------- M3
    say("M3 WHAT DOES A FOUR-REPLICATE NOISE CEILING LOOK LIKE?")
    subs = {}
    for trio in combinations(ALL, 3):
        vals = [r2s(D9[b].ravel(), D9[a].ravel()) for a, b in combinations(trio, 2)]
        subs[trio] = float(np.mean(vals))
        say(f"       replicates {trio}   mean pairwise R2 {subs[trio]:+.5f}")
    allv = float(np.mean([r2s(D9[b].ravel(), D9[a].ravel())
                          for a, b in combinations(ALL, 2)]))
    say(f"       all four                mean pairwise R2 {allv:+.5f}")
    best = max(subs, key=lambda k: subs[k])
    say(f"     best trio {best} at {subs[best]:+.5f}; loop 216's (1,2,3) recorded {REF123:+.5f}")
    G.add("M3", bool(subs[best] - REF123 > 0.20), stat=subs[best], requires=("M1",),
          if_true=lambda: f"M3 PASS -- dropping replicate 1 for replicate 4 moves the ceiling "
                          f"from {REF123:+.4f} to {subs[best]:+.4f}",
          if_false=lambda: f"M3 FAIL -- the best trio {best} reaches {subs[best]:+.4f} against "
                           f"{REF123:+.4f}")
    res["ceilings"] = {str(k): v for k, v in subs.items()}
    res["ceiling_all4"] = allv

    # ---------------------------------------------------------------- M4
    say("M4 THE CLEAN SPLIT: SELECT ON 2+3, SCORE ON 4")
    w23 = np.stack([D9[2], D9[3]]).var(axis=0, ddof=1).mean(axis=0)
    t23 = np.mean([D9[2], D9[3]], axis=0).var(axis=0, ddof=1)
    snr23 = np.maximum(t23 - w23 / 2, 0.0) / (w23 / 2 + 1e-12)
    Mm = M[:, idx]
    g9 = np.array(GRID9, float)
    lvl = np.array([Mm[j - 1] for j in range(1, len(g9))])
    dts = np.array([g9[j] - g9[j - 1] for j in range(1, len(g9))])
    Dm = np.mean([D9[r] for r in (2, 3, 4)], axis=0)
    trj = np.arange(len(dts)) < (N_TRAIN - 1)
    hon = {}
    for frac in (0.10, 0.25, 0.50, 1.00):
        k = max(20, int(frac * len(names)))
        sel = np.argsort(-snr23)[:k]
        d_tr = (dts[trj, None] * (S[None, sel] - lvl[trj][:, sel])).ravel()
        y_tr = Dm[trj][:, sel].ravel()
        lam = float(d_tr @ y_tr / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
        d_te = (dts[~trj, None] * (S[None, sel] - lvl[~trj][:, sel])).ravel()
        y4 = D9[4][~trj][:, sel].ravel()
        rm, rp_ = r2s(y4, lam * d_te), r2s(y4, np.zeros_like(y4))
        hon[frac] = {"model": rm, "persistence": rp_, "margin": rm - rp_, "n": k}
        say(f"       top {int(frac*100):>3}%   model {rm:+.5f}   persistence {rp_:+.5f}   "
            f"margin {rm-rp_:+.5f}   n {k:,}")
    bestf = max(hon, key=lambda f: hon[f]["margin"])
    G.add("M4", bool(hon[bestf]["margin"] > 0.01), stat=hon[bestf]["margin"], requires=("M2",),
          if_true=lambda: f"M4 PASS -- on the top {int(bestf*100)}% selected from replicates 2+3, "
                          f"scored on replicate 4, the model beats persistence by "
                          f"{hon[bestf]['margin']:+.5f}",
          if_false=lambda: f"M4 FAIL -- the best selection buys {hon[bestf]['margin']:+.5f} even "
                           f"with a fully independent scorer")
    res["clean_split"] = {str(k): v for k, v in hon.items()}

    # ---------------------------------------------------------------- M5
    say("M5 DOES THE DENSER GRID HELP?")
    D11 = deltas((2, 3, 4), GRID11)
    if len(D11) == 3:
        c11 = float(np.mean([r2s(D11[b].ravel(), D11[a].ravel())
                             for a, b in combinations((2, 3, 4), 2)]))
        c9 = subs[(2, 3, 4)]
        say(f"       9-point grid  (8 intervals)   ceiling {c9:+.5f}")
        say(f"      11-point grid (10 intervals)   ceiling {c11:+.5f}   delta {c11-c9:+.5f}")
        say(f"       the extra points are 300 and 360 min, which replicate 1 lacks")
        G.add("M5", bool(c11 > c9), stat=c11, requires=("M1",),
              if_true=lambda: f"M5 PASS -- the denser grid raises the ceiling by {c11-c9:+.5f}",
              if_false=lambda: f"M5 FAIL -- the denser grid moves the ceiling by {c11-c9:+.5f}; "
                               f"shorter intervals mean smaller true changes against the same "
                               f"noise")
        res["dense_grid"] = {"nine": c9, "eleven": c11}
    else:
        G.add("M5", None, void_if=True,
              void_reason="not all of replicates 2, 3 and 4 cover the eleven-point grid")

    say("M6 WHAT THIS CANNOT SHOW")
    say("     Replicate 4 was on disk the whole time and no loop used it. That is a fetch")
    say("     failure in this project, not a discovery about biology, and nothing below should")
    say("     be read as new data.")
    say("     Four ENCODE isogenic replicates still share cell line, protocol and lab. They bound")
    say("     technical reproducibility, not biological variability, so every ceiling here")
    say("     remains an overestimate of what is really measurable.")
    say("     If M2 passes, replicate 1 has been in every number this project has reported since")
    say("     loop 191d, including the plateau that the set-point model was trained against.")
    say("     Re-deriving that target on replicates 2, 3 and 4 is a separate loop and is not")
    say("     done here.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
