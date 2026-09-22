"""Loop 250. The non-Markov claim, tested in single cells where bulk hit a ceiling.

WHY BULK COULD NOT ANSWER IT. Loop 247's W3 asked whether S(t-2) and S(t-3) predict S(t+1) once
S(t-1) is known, on the dexamethasone withdrawal series, and got +0.0000 (+0.2 se). But the
one-step model already scored 0.9936, so there was almost no room left for a lag to occupy. That
result bounds the memory term rather than excluding it, and loop 247 said so.

WHY SINGLE CELLS ARE DIFFERENT, AND WHAT THEY STILL CANNOT DO. Single-cell sequencing is
destructive: no cell is ever measured twice, so the textbook test -- follow one trajectory and
regress its future on its past -- is unavailable in principle, not merely inconvenient. What
sci-fate (GSE131351, A549, dexamethasone) adds is a second channel per cell. 4sU labelling
separates RNA transcribed during the labelling window from RNA already present:

    new    = transcription happening NOW          7,404 cells, 43,167 genes, 21% of counts
    total  = the current state S                  timepoints 0, 2, 4, 6, 8, 10 h after dex
    old    = total - new = accumulated history

THE TRAP, AND IT IS THE WHOLE REASON THIS LOOP IS DESIGNED THE WAY IT IS. The obvious test --
"does old RNA predict new RNA beyond total?" -- is ALGEBRAICALLY VOID. total = new + old exactly,
so total together with old determines new with no biology involved whatsoever. Any model given
both would score perfectly and the number would mean nothing. This is loop 231's L2 confound, where
features and target shared terms by construction, in its purest available form. A weaker version --
predict gene g from OTHER genes' old and total -- does not escape it either, because old_h and
total_h differ exactly by new_h, and new_h predicts new_g through the cell's shared global
transcription rate. Feeding a model `old` alongside `total` is feeding it `new` on the other genes.

SO THE TEST USES TIME, WHICH IS NOT ALGEBRAICALLY RELATED TO ANY COUNT. If the process is Markov
in S, then a cell's transcription rate depends on its current state and nothing else, so:

    two cells in the SAME state should transcribe the same way
    regardless of how long they have been in dexamethasone

Time-since-treatment is exactly the accumulated history that M is supposed to carry. If knowing it
adds nothing once the state is known, the process is Markov in S at this resolution. If it adds,
the cell contains history its current transcriptome does not express.

PREDECLARED, BEFORE ANY NUMBER.

  Z1 IS THE STATE MEASURED WELL ENOUGH FOR A CONDITIONAL TEST TO MEAN ANYTHING?
     "Time adds nothing GIVEN the state" is vacuous if the state predicts nothing. The total-RNA
     profile must predict the new-RNA profile of a held-out cell.
     Gate: PASS iff held-out correlation across genes exceeds 0.30. Everything requires this.

  Z2 IS THIS A REAL DEXAMETHASONE TIMECOURSE?
     Textbook glucocorticoid targets, named before looking: TSC22D3 (GILZ), FKBP5, PER1, KLF15,
     ZBTB16. These rise on dexamethasone in A549.
     Gate: PASS iff at least 3 of the 5 have higher mean new-RNA at 10 h than at 0 h.

  Z3 DOES TIME ADD BEYOND STATE?      -- requires Z1. The non-Markov test.
     Held out by cell: state alone against state plus time-since-treatment.
     Gate: PASS iff adding time improves the held-out across-gene correlation by at least 0.02,
     paired over held-out cells.

  Z4 THE STATE-MATCHED FORM.      -- requires Z1
     Z3 can pass because the state is truncated to a finite number of components rather than
     because history exists. Z4 removes that reading: each cell is matched to its k nearest
     neighbours in state space, and time is tested WITHIN those neighbourhoods, where the state is
     as close to held constant as the data allows.
     Gate: PASS iff, among state-matched neighbours, the correlation between time difference and
     new-RNA profile difference exceeds what the same statistic gives with time permuted inside
     each neighbourhood, by at least 3 standard errors.

  Z5 CONTROL: TIME PERMUTED ACROSS ALL CELLS.      -- requires Z3, VOID if Z3's margin is under 0.005
     Gate: PASS iff Z3's improvement collapses to under 25%.

  Z6 THE REVERSE DIRECTION, SO A Z3 PASS IS NOT JUST "TIME IS INFORMATIVE".      -- requires Z1
     State added to time, rather than time added to state.
     Gate: PASS iff state adds more to time than time adds to state. If time alone were doing the
     work, the state would be the redundant variable, and that is worth knowing either way.

  Z7 WHAT THIS CANNOT SHOW -- written before the run.
     Destructive sampling stands. This tests whether history is visible in a population of cells
     at different treatment durations, not whether a single cell's trajectory is non-Markov.
     Time-since-treatment is a coarse summary of history: a cell at 10 h that responded late is
     scored identically to one that responded early.
     Cells are not synchronised. Cell-cycle phase varies within every timepoint and is itself a
     hidden state with its own history, so a Z3 pass has a cell-cycle reading as well as a
     memory reading, and Z4 does not separate them.
     One stimulus, one cell line, ten hours. The labelling window sets the meaning of "now" and
     anything faster than it is invisible.
     A Z3 FAIL bounds the memory term at this resolution rather than excluding it, which is the
     same limit loop 247 recorded and is not removed by having single cells.
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_markov_singlecell.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SF = SCR / "scifate"

SEED, NFOLD, NPC, KNN = 250250, 10, 40, 30
MIN_CELLS_GENE, MAX_DOUBLET, MIN_UMI = 100, 0.15, 1000
DEX_TARGETS = ["TSC22D3", "FKBP5", "PER1", "KLF15", "ZBTB16"]
Z1_BAR, Z2_MIN, Z3_BAR, Z4_SE, Z5_MAX = 0.30, 3, 0.02, 3.0, 0.25

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "non-Markov test in single cells via sci-fate new/total RNA"}
    say("=" * 104)
    say("LOOP 250 -- THE NON-MARKOV CLAIM, IN SINGLE CELLS")
    say("=" * 104)
    say("     The obvious test -- does OLD RNA predict NEW RNA beyond TOTAL -- is algebraically")
    say("     void: total = new + old exactly, so the two determine the third with no biology.")
    say("     That is loop 231's confound in its purest form. This loop uses TIME instead, which")
    say("     is not algebraically related to any count: if the process is Markov in S, two cells")
    say("     in the same state transcribe the same way however long they have been in dex.")

    M = sparse.load_npz(SF / "total.npz").tocsr()
    N = sparse.load_npz(SF / "new.npz").tocsr()
    meta = np.load(SF / "meta.npz", allow_pickle=True)
    tstr = np.array([str(x) for x in meta["time"]])
    dbl = np.asarray(meta["doublet"], float)
    genes = np.array([str(x) for x in meta["genes"]])
    hours = np.array([float(s.replace("h", "")) for s in tstr])

    umi = np.asarray(M.sum(1)).ravel()
    ok = (dbl <= MAX_DOUBLET) & (umi >= MIN_UMI)
    say(f"     {M.shape[0]:,} cells; keeping doublet score <= {MAX_DOUBLET} and >= {MIN_UMI:,} "
        f"UMI -> {int(ok.sum()):,}")
    M, N, hours, umi = M[ok], N[ok], hours[ok], umi[ok]
    nc = np.asarray((M > 0).sum(0)).ravel()
    gk = np.where(nc >= MIN_CELLS_GENE)[0]
    M, N, genes = M[:, gk], N[:, gk], genes[gk]
    say(f"     {len(gk):,} genes detected in {MIN_CELLS_GENE}+ cells")
    for h in sorted(set(hours)):
        say(f"       t={h:4.0f} h  {int((hours == h).sum()):5d} cells")

    def lognorm(A):
        s = np.asarray(A.sum(1)).ravel()
        B = sparse.diags(1e4 / np.maximum(s, 1)) @ A
        B = B.tocsr(); B.data = np.log1p(B.data)
        return B
    S = lognorm(M)              # the state
    Rn = lognorm(N)             # what the cell is transcribing now
    Sd = np.asarray(S.todense())
    Nd = np.asarray(Rn.todense())

    # ---------------------------------------------------------------- Z2
    say("Z2 IS THIS A REAL DEXAMETHASONE TIMECOURSE?")
    gi = {g: i for i, g in enumerate(genes)}
    up = 0; det = []
    for g in DEX_TARGETS:
        if g not in gi:
            det.append(f"{g} absent"); continue
        a = Nd[hours == 0, gi[g]].mean(); b = Nd[hours == 10, gi[g]].mean()
        up += b > a
        det.append(f"{g} {a:.3f}->{b:.3f}")
    say(f"     new-RNA at 0 h vs 10 h: {'; '.join(det)}")
    G.add("Z2", bool(up >= Z2_MIN), stat=float(up),
          if_true=lambda: f"Z2 PASS -- {up} of {len(DEX_TARGETS)} textbook glucocorticoid targets "
                          f"rise",
          if_false=lambda: f"Z2 FAIL -- only {up} of {len(DEX_TARGETS)} rise; this does not behave "
                           f"like a dexamethasone timecourse")
    res["Z2"] = {"n_up": int(up), "detail": det}

    # ---------------------------------------------------------------- state components
    mu = Sd.mean(0)
    U, sv, Vt = np.linalg.svd(Sd - mu, full_matrices=False)
    P = (U[:, :NPC] * sv[:NPC])
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)
    say(f"     state reduced to {NPC} components "
        f"({np.sum(sv[:NPC] ** 2) / np.sum(sv ** 2):.1%} of variance), unsupervised")
    th = (hours - hours.mean()) / hours.std()
    depth = (np.log(umi) - np.log(umi).mean()) / np.log(umi).std()

    order = rng.permutation(len(hours))
    folds = [order[i::NFOLD] for i in range(NFOLD)]

    def cv(F):
        """Predict each held-out cell's new-RNA profile; score across genes within the cell."""
        sc = np.full(len(hours), np.nan)
        for te in folds:
            tr = np.setdiff1d(np.arange(len(hours)), te)
            Z = np.concatenate([F[tr], np.ones((len(tr), 1))], 1)
            A = Z.T @ Z + 1e-2 * len(tr) * np.eye(Z.shape[1])
            B = np.linalg.solve(A, Z.T @ Nd[tr])
            Pr = np.concatenate([F[te], np.ones((len(te), 1))], 1) @ B
            for j, c in enumerate(te): sc[c] = pear(Pr[j], Nd[c])
        return sc

    # ---------------------------------------------------------------- Z1
    say("Z1 IS THE STATE MEASURED WELL ENOUGH FOR A CONDITIONAL TEST TO MEAN ANYTHING?")
    Fs = np.concatenate([P, depth[:, None]], 1)
    s_state = cv(Fs)
    r1 = float(np.nanmean(s_state))
    say(f"     total-RNA state predicts a held-out cell's new-RNA profile at r = {r1:.4f}")
    G.add("Z1", bool(r1 >= Z1_BAR), stat=float(r1),
          if_true=lambda: f"Z1 PASS -- the state predicts at {r1:.4f}, so 'given the state' is not "
                          f"vacuous",
          if_false=lambda: f"Z1 FAIL -- {r1:.4f} against a {Z1_BAR} bar; conditioning on a state "
                           f"this poorly measured would make Z3 meaningless")
    res["Z1"] = {"state_r": r1}

    # ---------------------------------------------------------------- Z3
    say("Z3 DOES TIME ADD BEYOND STATE?")
    Ft = np.concatenate([P, depth[:, None], th[:, None]], 1)
    s_both = cv(Ft)
    d3, se3, z3 = paired(s_both, s_state)
    say(f"     state alone {r1:.4f}   state + time {np.nanmean(s_both):.4f}")
    say(f"     paired over {int(np.isfinite(s_both).sum()):,} held-out cells: {d3:+.5f} "
        f"+/- {se3:.5f}  ({z3:+.1f} se)")
    G.add("Z3", bool(d3 >= Z3_BAR), stat=float(d3), requires=("Z1",),
          if_true=lambda: f"Z3 PASS -- time adds {d3:+.4f} beyond the state; the cell carries "
                          f"history its transcriptome does not express",
          if_false=lambda: f"Z3 FAIL -- time adds {d3:+.4f} beyond the state, against a {Z3_BAR} "
                           f"bar; at this resolution the process is Markov in S")
    res["Z3"] = {"state": r1, "state_time": float(np.nanmean(s_both)), "delta": d3, "se": se3,
                 "z": z3}

    # ---------------------------------------------------------------- Z6
    say("Z6 THE REVERSE DIRECTION")
    Ftime = np.concatenate([th[:, None], depth[:, None]], 1)
    s_time = cv(Ftime)
    d6a, _, _ = paired(s_both, s_time)
    say(f"     time alone {np.nanmean(s_time):.4f}; state added to time {d6a:+.4f}, "
        f"time added to state {d3:+.4f}")
    G.add("Z6", bool(d6a > d3), stat=float(d6a), requires=("Z1",),
          if_true=lambda: f"Z6 PASS -- the state adds {d6a:+.4f} to time while time adds {d3:+.4f} "
                          f"to the state; the state is the informative variable",
          if_false=lambda: f"Z6 FAIL -- time adds more to the state ({d3:+.4f}) than the state "
                           f"adds to time ({d6a:+.4f}); the state is the redundant one")
    res["Z6"] = {"time_only": float(np.nanmean(s_time)), "state_added": d6a, "time_added": d3}

    # ---------------------------------------------------------------- Z4
    say("Z4 THE STATE-MATCHED FORM")
    say("     each cell matched to its nearest neighbours in state space, time tested WITHIN those")
    say("     neighbourhoods, so a Z3 result cannot be explained by the state being truncated")
    idx = rng.choice(len(hours), size=min(1500, len(hours)), replace=False)
    real, null = [], []
    for c in idx:
        d = np.linalg.norm(P - P[c], axis=1)
        nb = np.argsort(d)[1:KNN + 1]
        dt = np.abs(hours[nb] - hours[c])
        dn = np.linalg.norm(Nd[nb] - Nd[c], axis=1)
        if np.std(dt) < 1e-9: continue
        real.append(pear(dt, dn))
        null.append(pear(rng.permutation(dt), dn))
    real, null = np.asarray(real), np.asarray(null)
    d4, se4, z4 = paired(real, null)
    say(f"     within {len(real):,} state-matched neighbourhoods of {KNN}: corr(|time gap|, "
        f"|new-RNA difference|) = {np.nanmean(real):+.4f}")
    say(f"     the same with time permuted inside each neighbourhood: {np.nanmean(null):+.4f}")
    say(f"     paired {d4:+.5f} +/- {se4:.5f}  ({z4:+.1f} se)")
    G.add("Z4", bool(d4 > 0 and z4 >= Z4_SE), stat=float(d4), requires=("Z1",),
          if_true=lambda: f"Z4 PASS -- at matched state, cells further apart in time differ more "
                          f"in what they transcribe ({z4:+.1f} se)",
          if_false=lambda: f"Z4 FAIL -- {d4:+.5f} ({z4:+.1f} se); with the state held as fixed as "
                           f"the data allows, time carries no additional signal")
    res["Z4"] = {"real": float(np.nanmean(real)), "null": float(np.nanmean(null)), "delta": d4,
                 "se": se4, "z": z4, "n": int(len(real))}

    # ---------------------------------------------------------------- Z5
    say("Z5 CONTROL: TIME PERMUTED ACROSS ALL CELLS")
    if d3 < 0.005:
        G.add("Z5", False, stat=float(d3), requires=("Z3",), void_if=True,
              void_reason=f"Z3's margin is {d3:+.5f}; there is nothing to collapse")
    else:
        Fp = np.concatenate([P, depth[:, None], rng.permutation(th)[:, None]], 1)
        s_perm = cv(Fp)
        dp, _, _ = paired(s_perm, s_state)
        f5 = dp / d3
        say(f"     time permuted: adds {dp:+.5f} against a real {d3:+.5f}  ({f5:.0%})")
        G.add("Z5", bool(f5 <= Z5_MAX), stat=float(f5), requires=("Z3",),
              if_true=lambda: f"Z5 PASS -- collapses to {f5:.0%} with time permuted",
              if_false=lambda: f"Z5 FAIL -- {f5:.0%} survives permuting time")
        res["Z5"] = {"real": d3, "shuffled": dp, "fraction": f5}

    say("Z7 WHAT THIS CANNOT SHOW")
    say("     Destructive sampling stands: this asks whether history is visible ACROSS cells at")
    say("     different treatment durations, not whether one cell's trajectory is non-Markov.")
    say("     Time-since-treatment is a coarse summary of history -- a cell at 10 h that responded")
    say("     late scores identically to one that responded early.")
    say("     Cells are unsynchronised. Cell-cycle phase varies within every timepoint and is")
    say("     itself a hidden state with its own history, so a pass has a cell-cycle reading as")
    say("     well as a memory reading and Z4 does not separate them.")
    say("     The labelling window defines 'now'; anything faster is invisible.")
    say("     A FAIL bounds the memory term at this resolution rather than excluding it -- the")
    say("     same limit loop 247 recorded, and single cells do not remove it.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
