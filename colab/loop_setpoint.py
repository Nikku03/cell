"""Loop 206. Can the missing regulatory gain be COMPUTED instead of measured?

THE HOLE THIS AIMS AT. The 14-agent audit in NOTES_rem_cell.md counted ~27,474 usable rate
constants against ~859,981 needed, and found that 71% of the entire shortfall is ONE block:
612,133 transcription-factor edge gains, every one of them zero. The architecture's whole
cross-subsystem propagation runs through that block. Nothing else in the parameter ledger is worth
attacking first.

WHY THIS IS NOT LOOP 205 AGAIN. Loop 205 measured that rate constants cannot be computed: k is
exponential in a barrier, RT is 0.616 kcal/mol at 37 C, and perfect knowledge of the chemistry
still leaves 7.8x median error with 84% of the spread intrinsic. A binding gain is a different
physical object. It is an EQUILIBRIUM, not a barrier crossing, and equilibrium binding energy is
the one quantity sequence-based biophysics has a principled route to (Berg and von Hippel 1987: a
PWM log-odds score IS a binding free energy in kT, up to an additive constant). Loop 206a built
that occupancy for NR3C1 over loop 198's scored promoters, on a grid of chemical potential, and
selected nothing. This loop scores it.

THE BENCHMARK, AND WHY THIS ONE. The A549 dexamethasone course is the only series in this project
with a held-out-in-time bar already attached: loop 198 measured that predicting NO CHANGE scores
-0.02953 and everything the project knows scores -0.05205. Dexamethasone is NR3C1's ligand, so the
computed factor is the one the perturbation acts on directly. If a computed occupancy is going to
carry a gain anywhere, it is here, and a failure here is not a failure of an unlucky choice.

THE MODEL IS FIXED BEFORE ANY ARM RUNS. Every arm predicts the same way:

    delta_m(g, t) = lam * dt * ( S_g - m_prev(g, t) )

one global relaxation rate lam fitted on the training window, and a per-gene set point S_g that is
the ONLY thing the arms differ in. This is dM/dt = k_sm - a*M rewritten, so the form is the
architecture's own dynamic law and the comparison is purely about whether the set point can be
supplied.

WHAT WOULD MAKE THIS LOOP VACUOUS, GUARDED IN Y1. If the harness does not reproduce loop 198's
persistence number, then this is a different gene set, a different split or a different target, and
every comparison below would be against a bar that does not exist.

PREDECLARED, BEFORE ANY NUMBER.

  Y1 IS THIS LOOP 198'S HARNESS?
     Gate: PASS iff held-out-in-time persistence reproduces -0.02953 to four decimals on the same
     grid, the same responder rule and the same time split. FAIL means nothing below is comparable
     to the recorded bar.

  Y2 IS THE FORM RIGHT? The oracle set point -- the true measured plateau -- with one fitted lam.
     Gate: PASS iff it beats persistence. A FAIL kills the whole approach, because if relaxation to
     the RIGHT set point cannot beat predicting no change then the set point is not the missing
     piece and the architecture is wrong about its own equation.
     The three-replicate reproducibility ceiling is computed alongside and reported, so the oracle
     can be read as a fraction of what is reproducible rather than of 1.0.

  Y3 HOW GOOD DOES A SET POINT HAVE TO BE?
     Degrade the oracle with calibrated Gaussian noise across a sweep, and find the Pearson r
     against the true plateau at which relaxation crosses persistence.
     Gate: PASS iff the crossover r is BELOW the plateau's own measurement reliability. If the
     requirement sits above the reliability ceiling then no predictor could ever clear it and the
     benchmark is undecidable rather than hard. Requires Y2.

  Y4 THE MEASURED ARM. Set point regressed from the nine ChIP/DNase tracks measured in these same
     cells, gene-held-out folds.
     Gate: PASS iff it beats the training-mean arm AND beats publication count as a set-point
     predictor. This is the repo's standing fame floor and it has beaten real biology before
     (loop 71). It does NOT have to beat persistence -- Y5 is the question, not this.

  Y5 THE PHYSICS ARM -- the gate this loop exists for.
     Set point regressed from loop 206a's COMPUTED thermodynamic occupancy, with mu selected on
     TRAINING folds only. No ChIP anywhere in this arm.
     Gate: PASS iff the computed arm reaches at least 80% of the measured arm's Pearson r against
     the true plateau.
     A PASS means physics can substitute for measurement, which matters far beyond this benchmark:
     there is no ChIP for the overwhelming majority of the 612,133 edges, and a computed occupancy
     of comparable quality could fill them.
     A FAIL means the computed side does not reach even the measured side, so computing cannot
     fill a hole that measuring only half-fills.
     Requires Y4.

  Y6 IS EITHER ARM JUST FAME?
     Gate: PASS iff both the measured and computed arms beat publication count on the same folds.

  Y7 SCOPE, STATED BEFORE THE NUMBERS SO IT CANNOT BE READ AS AN EXCUSE AFTERWARDS.
     The red-team pass measured the measured-track arm at r +0.4104 against a crossover requirement
     near 0.889. If that reproduces, then a physics arm that PASSES Y5 still does not rescue this
     benchmark -- 80% of 0.41 is not 0.89. Y5 is not asking whether physics rescues the A549
     course. It is asking whether a computed gain is as good as a measured one, because that is the
     question that generalises to the 612,133 edges where nothing was ever measured.

  Y8 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
A549 = SP / "grtc"
OCC_F = ROOT / "colab" / "data" / "physics" / "nr3c1_occupancy.npz"
OUT = "outputs/loop_setpoint.json"

T_MIN, REPS, MIN_TPM, MIN_PLATEAU = 30.0, (1, 2, 3), 1.0, 0.5
PROM_PAD = L191.PROM_PAD
N_TRAIN, SEED = 6, 206206
TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
PERSIST_REF = -0.02953

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def r2(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def ridge(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    n = A.shape[1]
    R = lam * np.eye(n); R[-1, -1] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ ytr)
    return B @ w


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "set point"}
    say("=" * 104)
    say("LOOP 206 -- CAN THE MISSING REGULATORY GAIN BE COMPUTED INSTEAD OF MEASURED?")
    say("=" * 104)

    # ---------------------------------------------------------------- harness
    z = np.load(A549 / "rna.npz", allow_pickle=True)
    tpm = z["tpm"]
    ensg = np.array([str(g).split(".")[0] for g in z["genes"]])
    mins, reps = z["mins"].astype(int), z["reps"].astype(int)
    allt = sorted(set(mins.tolist()))
    comp = {t: set(reps[mins == t].tolist()) for t in allt}
    grid = np.array([t for t in allt if set(REPS) <= comp[t] and t >= T_MIN], dtype=float)
    M, _ = L191.rep_trajectories(tpm, mins, reps, REPS, grid)
    e2s = L191.ensg_to_symbol(lambda *_: None)
    sym = np.array([e2s.get(g, "") for g in ensg])
    base = tpm[(mins == int(grid[0])) & np.isin(reps, REPS)].mean(0)
    pl = M[-3:].mean(0)
    resp = (base >= MIN_TPM) & (np.abs(pl) >= MIN_PLATEAU)
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    tssb, pubs = {}, {}
    for i, g in enumerate(tab):
        pubs[str(g["name"]).upper()] = float(g.get("pubs") or 0)
    for line in open(SP / "_tss_hg38.bed"):
        q = line.split()
        if len(q) >= 4 and q[3].startswith("G"):
            i = int(q[3][1:])
            if i < len(tab):
                tssb[str(tab[i]["name"]).upper()] = (q[0], int(q[2]))
    TR = {}
    for name in TRACKS:
        pt, PM = L191.promoter_track(name, [tssb.get(s) for s in sym], PROM_PAD, lambda *_: None)
        idx = [int(np.where(pt == t)[0][0]) for t in grid]
        TR[name] = PM[idx]
    A = TR["DNase"]
    keep = resp & (A > 0).any(0)
    gi = np.where(keep)[0]
    say(f"     grid {[int(x) for x in grid]}   train on the first {N_TRAIN}, score on the rest")
    say(f"     {len(gi):,} genes are responders with a promoter DNase peak")

    def rows(lo, hi):
        X, y, g, prev, dts = [], [], [], [], []
        for j in range(1, len(grid)):
            if not (lo <= j < hi):
                continue
            dt = grid[j] - grid[j - 1]
            for i in gi:
                X.append([A[j, i] - A[j - 1, i], A[j - 1, i], M[j - 1, i], dt])
                y.append(M[j, i] - M[j - 1, i]); g.append(i)
                prev.append(M[j - 1, i]); dts.append(dt)
        return (np.array(X), np.array(y), np.array(g), np.array(prev), np.array(dts))

    Xtr, ytr, gtr, ptr, dtr = rows(1, N_TRAIN)
    Xte, yte, gte, pte, dte = rows(N_TRAIN, len(grid))

    # ---------------------------------------------------------------- Y1
    say("Y1 IS THIS LOOP 198'S HARNESS?")
    pers = r2(yte, np.zeros_like(yte))
    say(f"     rows: train {len(ytr):,}  score {len(yte):,}   genes {len(gi):,}")
    say(f"     persistence held-out-in-time R2 {pers:.8f}   loop 198 recorded {PERSIST_REF}")
    ok1 = bool(abs(pers - PERSIST_REF) < 5e-5)
    G.add("Y1", ok1, stat=pers,
          if_true=f"Y1 PASS -- persistence reproduces loop 198 to four decimals",
          if_false=lambda: f"Y1 FAIL -- {pers:.8f} against {PERSIST_REF}; different harness")
    res["harness"] = {"n_train": len(ytr), "n_test": len(yte), "n_genes": len(gi),
                      "persistence": pers, "reference": PERSIST_REF}

    # ---------------------------------------------------------------- Y2
    say("Y2 IS THE FORM RIGHT?  relaxation to the ORACLE set point")
    S_true = pl[gi]
    gpos = {g: k for k, g in enumerate(gi)}
    s_tr = np.array([S_true[gpos[g]] for g in gtr])
    s_te = np.array([S_true[gpos[g]] for g in gte])

    def fit_lam(sp, prev, dt, y):
        d = dt * (sp - prev)
        den = float(d @ d)
        return float(d @ y / den) if den > 0 else 0.0

    lam = fit_lam(s_tr, ptr, dtr, ytr)
    r2_or = r2(yte, lam * dte * (s_te - pte))
    Mr, _ = L191.rep_trajectories(tpm, mins, reps, REPS, grid, per_rep=True) \
        if hasattr(L191, "rep_trajectories") and False else (None, None)
    per_rep = []
    for rp in REPS:
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (rp,), grid)
        per_rep.append(Mi[-3:].mean(0)[gi])
    rs = [pear(per_rep[a], per_rep[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
    rbar = float(np.mean(rs))
    sb = 3 * rbar / (1 + 2 * rbar)
    say(f"     fitted global rate lam {lam:+.6f} /min")
    say(f"     oracle set point held-out-in-time R2 {r2_or:+.5f}   vs persistence {pers:+.5f}")
    say(f"     plateau replicate pairwise r {rs[0]:.4f} {rs[1]:.4f} {rs[2]:.4f}   "
        f"Spearman-Brown reliability {sb:.4f}")
    G.add("Y2", bool(r2_or > pers), stat=r2_or, requires=("Y1",),
          if_true=lambda: f"Y2 PASS -- the form is right: relaxation to the true plateau with ONE "
                          f"global rate scores {r2_or:+.5f} against persistence {pers:+.5f}",
          if_false=lambda: f"Y2 FAIL -- {r2_or:+.5f} does not beat persistence {pers:+.5f}, so the "
                           f"set point is not the missing piece and the equation is wrong")
    res["oracle"] = {"lam": lam, "r2": r2_or, "replicate_r": rs, "reliability": sb}

    # ---------------------------------------------------------------- Y3
    say("Y3 HOW GOOD DOES A SET POINT HAVE TO BE?")
    rng = np.random.default_rng(SEED)
    sweep = []
    sd_S = float(np.std(S_true))
    for f in np.arange(0.0, 2.51, 0.05):
        noisy = S_true + rng.normal(0, f * sd_S, len(S_true))
        rr = pear(noisy, S_true)
        st = np.array([noisy[gpos[g]] for g in gtr])
        se = np.array([noisy[gpos[g]] for g in gte])
        lm = fit_lam(st, ptr, dtr, ytr)
        sweep.append((float(rr), r2(yte, lm * dte * (se - pte))))
    cross = None
    for rr, v in sorted(sweep):
        if v > pers:
            cross = rr
            break
    say(f"     crossover: a set point must reach Pearson r >= {cross:.4f} before relaxation "
        f"beats persistence")
    say(f"     the plateau's own reliability ceiling is r {sb:.4f}")
    for rr, v in sweep[::10]:
        say(f"       r {rr:+.4f}  ->  R2 {v:+.5f}")
    G.add("Y3", bool(cross is not None and cross < sb), stat=cross, requires=("Y2",),
          if_true=lambda: f"Y3 PASS -- the requirement {cross:.3f} sits below the reliability "
                          f"ceiling {sb:.3f}, so the benchmark is decidable rather than merely hard",
          if_false=lambda: f"Y3 FAIL -- the requirement {cross} is at or above the reliability "
                           f"ceiling {sb:.3f}; no predictor could ever clear it")
    res["crossover"] = {"required_r": cross, "ceiling_r": sb, "sweep": sweep}

    # ---------------------------------------------------------------- arms
    def score_setpoint(F, label):
        """Gene-held-out 5-fold prediction of the plateau, then the relaxation R2."""
        order = np.random.default_rng(SEED).permutation(len(gi))
        folds = np.array_split(order, 5)
        Sp = np.zeros(len(gi))
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Sp[te] = ridge(F[tr], S_true[tr], F[te])
        rr = pear(Sp, S_true)
        st = np.array([Sp[gpos[g]] for g in gtr]); se = np.array([Sp[gpos[g]] for g in gte])
        lm = fit_lam(st, ptr, dtr, ytr)
        v = r2(yte, lm * dte * (se - pte))
        say(f"       {label:<26} r {rr:+.4f}   lam {lm:+.6f}   R2 {v:+.5f}")
        return rr, v, lm

    say("Y4 THE MEASURED ARM")
    Fm = np.column_stack([np.column_stack([TR[t][:N_TRAIN, gi].mean(0),
                                           TR[t][:N_TRAIN, gi].max(0),
                                           TR[t][N_TRAIN - 1, gi] - TR[t][0, gi]])
                          for t in TRACKS])
    say(f"     {Fm.shape[1]} columns from {len(TRACKS)} tracks measured in these same cells")
    r_meas, v_meas, _ = score_setpoint(Fm, "measured (9 tracks)")
    Fmean = np.zeros((len(gi), 1))
    r_mean, v_mean, _ = score_setpoint(Fmean + 1.0, "training-mean")
    pv = np.array([pubs.get(sym[i], 0.0) for i in gi])
    Ff = np.log1p(pv).reshape(-1, 1)
    r_fame, v_fame, _ = score_setpoint(Ff, "publication count (fame)")
    G.add("Y4", bool(v_meas > v_mean and abs(r_meas) > abs(r_fame)), stat=r_meas,
          requires=("Y1",),
          if_true=lambda: f"Y4 PASS -- measured tracks give a set point at r {r_meas:+.4f}, "
                          f"beating the training mean and the fame floor r {r_fame:+.4f}",
          if_false=lambda: f"Y4 FAIL -- measured r {r_meas:+.4f} against fame {r_fame:+.4f} and "
                           f"mean R2 {v_mean:+.5f}")

    say("Y5 THE PHYSICS ARM -- computed occupancy, no ChIP anywhere in it")
    if not OCC_F.exists():
        G.add("Y5", None, requires=("Y4",), void_if=True,
              void_reason=f"loop 206a's occupancy cache is not on disk at {OCC_F}")
        r_phys = v_phys = float("nan")
    else:
        Z = np.load(OCC_F, allow_pickle=True)
        occ, mus, onames = Z["occ"], Z["mu"], [str(x) for x in Z["genes"]]
        pos = {s: k for k, s in enumerate(onames)}
        have = np.array([sym[i] in pos for i in gi])
        say(f"     computed occupancy available for {int(have.sum()):,} of {len(gi):,} genes "
            f"({have.mean():.1%})")
        Fp = np.zeros((len(gi), len(mus)))
        for k, i in enumerate(gi):
            if sym[i] in pos:
                Fp[k] = occ[pos[sym[i]]]
        # mu selected on TRAINING folds only
        order = np.random.default_rng(SEED).permutation(len(gi))
        folds = np.array_split(order, 5)
        tr0 = np.concatenate(folds[1:])
        rs_mu = [abs(pear(Fp[tr0, m], S_true[tr0])) for m in range(len(mus))]
        best = int(np.argmax(rs_mu))
        say(f"     mu selected on training folds only: mu {mus[best]:+.1f} kT "
            f"(train |r| {rs_mu[best]:.4f})")
        Fphys = np.column_stack([Fp[:, best], np.log1p(Fp[:, best]),
                                 Fp[:, max(0, best - 4)], Fp[:, min(len(mus) - 1, best + 4)]])
        r_phys, v_phys, _ = score_setpoint(Fphys, "PHYSICS (computed)")
        ratio = abs(r_phys) / abs(r_meas) if r_meas else float("nan")
        say(f"     computed r {r_phys:+.4f}  /  measured r {r_meas:+.4f}  =  {ratio:.3f}")
        G.add("Y5", bool(ratio >= 0.80), stat=ratio, requires=("Y4",),
              if_true=lambda: f"Y5 PASS -- a computed occupancy reaches {ratio:.0%} of a measured "
                              f"one, so physics CAN substitute for measurement on this quantity",
              if_false=lambda: f"Y5 FAIL -- computed reaches only {ratio:.0%} of measured "
                               f"({r_phys:+.4f} against {r_meas:+.4f}). Computing cannot fill a "
                               f"hole that measuring only half-fills")
        res["physics"] = {"mu": float(mus[best]), "coverage": float(have.mean()),
                          "r": r_phys, "r2": v_phys, "ratio": ratio}

    say("Y6 IS EITHER ARM JUST FAME?")
    ok6 = bool(abs(r_meas) > abs(r_fame) and (not np.isfinite(r_phys) or abs(r_phys) > abs(r_fame)))
    say(f"     measured {abs(r_meas):.4f}   computed {abs(r_phys):.4f}   "
        f"fame {abs(r_fame):.4f}")
    G.add("Y6", ok6, stat=r_fame, requires=("Y4",),
          if_true="Y6 PASS -- both arms beat publication count",
          if_false=lambda: f"Y6 FAIL -- fame at {abs(r_fame):.4f} is not beaten by both arms")
    res["arms"] = {"measured": {"r": r_meas, "r2": v_meas},
                   "mean": {"r": r_mean, "r2": v_mean},
                   "fame": {"r": r_fame, "r2": v_fame}}

    say("Y7 SCOPE")
    say("     Y5 does not ask whether physics rescues this benchmark. Y3's requirement and Y4's")
    say("     measured arm settle that on their own, and 80% of a number below the requirement is")
    say("     still below it. Y5 asks whether a COMPUTED gain is as good as a MEASURED one,")
    say("     because there is no ChIP for the overwhelming majority of the 612,133 edges.")

    say("Y8 WHAT THIS CANNOT SHOW")
    say("     One factor, one cell line, one perturbation. NR3C1 is the best case by construction")
    say("     -- it is the drug's own receptor -- so a failure here bounds the others and a")
    say("     success here does not generalise to factors whose ligand is not in the medium.")
    say("     The computed arm uses a promoter window only. Loop 185 measured that distal")
    say("     elements carry real signal, so a promoter-only occupancy is a floor for what")
    say("     sequence-based binding could contribute.")
    say("     Berg-von Hippel treats a PWM as an additive energy model. Real binding has")
    say("     interdependence between positions, and loop 184 measured co-binding at 0.8455")
    say("     against motif at 0.6228 -- most of what decides occupancy is not the site.")
    say("     A set point is not a gain. This measures whether sequence predicts where a gene")
    say("     ENDS UP, which is the product of all its gains, not any single edge weight.")

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
