"""Loop 222. Patching the measurement, not naming it.

WHAT LOOP 221 GOT WRONG IN KIND. Loop 221 spent its whole budget deciding what the loop-220
shared component IS, and its answer was refuted by its own predeclared gates. That was the wrong
job. Whether the component is library chemistry or biology does not change what has to be built:
an estimator of the per-interval change that is less contaminated than the raw difference. This
loop does no naming. It applies six standard corrections and measures which of them, if any,
raises a number that was falling.

THE SCORED QUANTITY, FIXED BEFORE ANY PATCH IS WRITTEN.

    S = Pearson r between the mean per-interval change of replicates 1, 2 and 3
        and the per-interval change of replicate 4, pooled over genes x intervals.

Three properties make this the right target and each is a lesson paid for earlier:

  IT CROSSES THE ARTEFACT BOUNDARY. Loop 220 measured that replicates 2 and 3 share a component
  explaining 27.63% and 31.01% of their change variance against 3.91% and 3.06% for 1 and 4.
  Scoring 2-against-3 rewards the contamination. Scoring against replicate 4 does not, because
  replicate 4 contributes to nothing that is fitted anywhere in this loop.

  IT IS PEARSON, NOT R-SQUARED, AND THAT IS DELIBERATE. Every ceiling in loops 216-220 was an
  R-squared, and R-squared here is trivially gamed: r2s(y, 0) is approximately 0, so a patch that
  shrinks its prediction to nothing scores 0.00 and appears to beat loop 220's -0.33410 by a third
  of a unit while predicting exactly nothing. That is the loop 212 C4 failure -- shrinkage "fixed
  the pit" by making the model inert -- and it would recur here silently. Pearson is scale
  invariant and cannot be moved by shrinking. R-squared after optimal rescaling is reported
  alongside, since it is r^2 by construction and adds no independent information.

  NOTHING FITTED SEES IT. Every tuning constant below -- the smoothing strength, the factor count,
  the interval weights -- is selected on replicates 1, 2 and 3 alone. Replicate 4 enters once, at
  scoring time. Loop 217's projection was defeated partly because the subspace it removed was
  estimated on the same pair it was scored against.

THE ANTI-INERTNESS CONTROL IS NOT OPTIONAL AND IS BUILT INTO EACH GATE. Retention is defined as
sum(patched^2) / sum(raw^2) on the replicate-1-2-3 mean. A patch qualifies only if it retains at
least 25% of the raw change variance. This is stated here so that a patch which wins by deleting
the data cannot be reported as a win, which is what loop 221's Q4 numbers would have looked like
had they run the other way.

THE SIX PATCHES, ALL STANDARD, NONE INVENTED HERE.

  RAW        the per-interval difference as loops 216-220 computed it. The thing to beat.

  SMOOTH     Whittaker smoothing of each gene's nine-point trajectory before differencing,
             s = (I + lam D2' D2)^-1 y with a second-difference penalty. Differencing amplifies
             independent noise by a factor of two in variance; smoothing first is the textbook
             answer to that and it is the patch most directly aimed at loop 216's finding that
             the plateau is measurable (+0.83380) and the difference is not (-0.54028).
             lam is chosen from a fixed grid by mean pairwise agreement among replicates 1, 2, 3.

  RUV        remove unwanted variation in the Gagnon-Bartsch and Speed sense, which is NOT what
             loop 221 did. Loop 221 DELETED candidate control genes and lost signal with them.
             RUV USES control genes as an estimator: take the genes with the smallest measured
             plateau, take the top k directions in TIME of their change matrix, and regress those
             directions out of every gene. The controls are never removed, and the correction
             applies to genes that were never controls. k is chosen on replicates 1, 2, 3.

  WEIGHT     errors-in-variables interval weighting. Each of the eight intervals has its own
             reliability; weight interval j by its measured within-{1,2,3} agreement, floored at
             zero. Intervals that carry no reproducible signal stop diluting the pooled statistic.

  ROBUST     median of replicates 1, 2, 3 instead of mean. If two of three carry a shared
             component, the mean inherits two thirds of it and the median does not.

  COMBO      the patches that individually qualified, applied in sequence.

PREDECLARED, BEFORE ANY NUMBER.

  V1 IS THERE ANYTHING TO IMPROVE?
     Gate: PASS iff the raw S is outside +/-2 standard deviations of a gene-label permutation null
     on replicate 4, over 200 draws. A patch that raises a statistic which was already
     indistinguishable from zero is the ratio-with-no-denominator family, and V2 onward must not
     be read if this fails.

  V2 DOES SMOOTHING HELP?
     Gate: PASS iff S rises by at least 0.03 absolute over RAW and retention is at least 25%.
     Requires V1.

  V3 DOES RUV HELP?
     Gate: PASS iff S rises by at least 0.03 over RAW, retention is at least 25%, AND it beats a
     random-direction control -- the same k directions drawn at random and orthonormalised, 20
     draws -- by at least 0.02. Without that control, removing ANY k directions from an 8-vector
     changes the statistic and the change would be read as the method working. Requires V1.

  V4 DOES INTERVAL WEIGHTING HELP?
     Gate: PASS iff S rises by at least 0.03 over RAW and retention is at least 25%. Requires V1.

  V5 DOES THE MEDIAN BEAT THE MEAN?
     Gate: PASS iff S rises by at least 0.03 over RAW and retention is at least 25%. Requires V1.

  V6 IS THE COMBINATION MORE THAN ITS PARTS?
     Gate: PASS iff the sequential combination of every patch that qualified exceeds the best
     single patch by at least 0.02. VOID if no single patch qualified -- there is then nothing to
     combine, and that is not a failure of combination.

  V7 DOES THE WINNER TRANSFER TO A FORWARD TASK?
     Train ridge on the first six intervals to predict the seventh and eighth from the preceding
     interval plus the running level, on replicates 1-2-3 mean, score on replicate 4, patched and
     unpatched by the same winning patch. Compare each against ITS OWN persistence baseline,
     computed on the same patched data, so the comparison is like for like.
     Gate: PASS iff the patched model beats its own persistence baseline by more than the
     unpatched model beats its own. VOID if no patch qualified.

  V8 WHAT THIS CANNOT SHOW -- stated before running, not after.
     A rise in S is a rise in agreement between one three-replicate average and one held-out
     replicate of one dexamethasone series in one cell line. It is not a demonstration that the
     corrected change is closer to the true transcription rate; nothing here measures the truth.
     If SMOOTH wins, part of the win is arithmetic -- smoothing correlates neighbouring intervals
     and pooled Pearson over intervals will rise even if no gene is better estimated -- and V7 is
     the only gate that can distinguish that, because a forward task cannot be helped by having
     smeared the answer into its own input.
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
OUT = "outputs/loop_patch_stack.json"
GRID = [30, 60, 120, 180, 240, 420, 480, 600, 720]
MIN_TPM, SEED = 1.0, 222222
LAMS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
KS = [0, 1, 2, 3, 4]
GAIN, RETAIN, CTRL_MARGIN, COMBO_MARGIN = 0.03, 0.25, 0.02, 0.02
TRAIN_REP = (1, 2, 3)
TEST_REP = 4

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def whittaker(V, lam):
    """Smooth each column of V (n_time, n_gene) with a second-difference penalty."""
    if lam <= 0:
        return V
    n = V.shape[0]
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    A = np.eye(n) + lam * (D.T @ D)
    return np.linalg.solve(A, V)


def diff_of(V):
    return np.array([V[j] - V[j - 1] for j in range(1, V.shape[0])])


def ruv_remove(D_, dirs):
    """Regress out `dirs` (k, n_interval) from every gene column of D_ (n_interval, n_gene)."""
    if dirs is None or len(dirs) == 0:
        return D_
    B = np.asarray(dirs, float).T                       # (n_interval, k)
    P = B @ np.linalg.pinv(B.T @ B) @ B.T               # projector onto the nuisance span
    return D_ - P @ D_


def orthonormal_random(k, n, rng):
    M = rng.normal(size=(n, k))
    Q, _ = np.linalg.qr(M)
    return Q.T[:k]


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "patching the change estimator"}
    rng = np.random.default_rng(SEED)
    say("=" * 104)
    say("LOOP 222 -- PATCHING THE MEASUREMENT, NOT NAMING IT")
    say("=" * 104)
    say("     Scored quantity S = Pearson r between the mean per-interval change of replicates")
    say("     1, 2, 3 and the per-interval change of replicate 4, pooled over genes x intervals.")
    say("     Every tuning constant is chosen on replicates 1, 2, 3. Replicate 4 enters at scoring.")

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    g = np.array(GRID, float)
    base = {r: tpm[(mins == 30) & (reps == r)].mean(0) for r in (1, 2, 3, 4)}
    sel = np.where(np.all([base[r] >= MIN_TPM for r in (1, 2, 3, 4)], axis=0))[0]
    ngen = len(sel)
    V = {}
    for r in (1, 2, 3, 4):
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (r,), g)
        V[r] = Mi[:, sel]                                # (n_time, n_gene) level trajectories
    n_iv = len(g) - 1
    say(f"     {ngen:,} genes retained at TPM >= {MIN_TPM} in all four replicates; "
        f"{n_iv} intervals")

    D_raw = {r: diff_of(V[r]) for r in (1, 2, 3, 4)}
    P_raw = np.mean([D_raw[r] for r in TRAIN_REP], axis=0)
    T_raw = D_raw[TEST_REP]
    s_raw = pear(P_raw, T_raw)
    raw_energy = float(np.sum(P_raw ** 2))

    # ---------------------------------------------------------------- V1
    say("V1 IS THERE ANYTHING TO IMPROVE?")
    null = []
    for _ in range(200):
        perm = rng.permutation(ngen)
        null.append(pear(P_raw, T_raw[:, perm]))
    null = np.array(null, float)
    z1 = (s_raw - null.mean()) / (null.std() + 1e-12)
    say(f"     RAW S = {s_raw:+.5f}   (R2 after optimal rescaling {s_raw**2:+.5f})")
    say(f"     gene-label permutation null on replicate 4, 200 draws: "
        f"{null.mean():+.5f} +/- {null.std():.5f}   z = {z1:+.2f}")
    G.add("V1", bool(abs(z1) > 2.0), stat=float(z1),
          if_true=lambda: f"V1 PASS -- RAW S {s_raw:+.4f} sits {abs(z1):.1f} sd outside the "
                          f"permutation null, so there is a real quantity here to raise",
          if_false=lambda: f"V1 FAIL -- RAW S {s_raw:+.4f} is within {abs(z1):.1f} sd of the "
                           f"permutation null; raising it would be raising noise")
    res["raw"] = {"S": s_raw, "null_mean": float(null.mean()), "null_sd": float(null.std()),
                  "z": float(z1), "n_genes": int(ngen)}

    def score(Pm, Tm):
        return pear(Pm, Tm)

    def retention(Pm):
        return float(np.sum(Pm ** 2) / raw_energy) if raw_energy > 0 else float("nan")

    qualified = {}

    # ---------------------------------------------------------------- V2
    say("V2 DOES SMOOTHING HELP?")
    best_lam, best_in = None, -np.inf
    for lam in LAMS:
        Ds = {r: diff_of(whittaker(V[r], lam)) for r in TRAIN_REP}
        inner = np.mean([pear(Ds[a], Ds[b]) for a, b in ((1, 2), (1, 3), (2, 3))])
        if inner > best_in:
            best_in, best_lam = inner, lam
    Ds_all = {r: diff_of(whittaker(V[r], best_lam)) for r in (1, 2, 3, 4)}
    P_sm = np.mean([Ds_all[r] for r in TRAIN_REP], axis=0)
    s_sm, ret_sm = score(P_sm, Ds_all[TEST_REP]), retention(P_sm)
    say(f"     lambda chosen on replicates 1,2,3 from {LAMS}: lam = {best_lam:g} "
        f"(within-123 agreement {best_in:+.4f})")
    say(f"     S {s_raw:+.5f} -> {s_sm:+.5f}   delta {s_sm-s_raw:+.5f}   retention {ret_sm:.1%}")
    ok2 = bool((s_sm - s_raw) >= GAIN and ret_sm >= RETAIN)
    G.add("V2", ok2, stat=float(s_sm - s_raw), requires=("V1",),
          if_true=lambda: f"V2 PASS -- smoothing raises S by {s_sm-s_raw:+.4f} to {s_sm:+.4f} "
                          f"while retaining {ret_sm:.0%} of the change variance",
          if_false=lambda: f"V2 FAIL -- delta {s_sm-s_raw:+.4f} against a {GAIN:+.2f} bar, "
                           f"retention {ret_sm:.0%} against {RETAIN:.0%}")
    if ok2:
        qualified["SMOOTH"] = s_sm
    res["smooth"] = {"lam": best_lam, "S": s_sm, "delta": s_sm - s_raw, "retention": ret_sm,
                     "inner": float(best_in)}

    # ---------------------------------------------------------------- V3
    say("V3 DOES RUV HELP?")
    P123 = np.mean([V[r] for r in TRAIN_REP], axis=0)
    plateau = P123[-3:].mean(0)
    ctrl = np.argsort(np.abs(plateau))[: max(200, ngen // 5)]
    say(f"     negative controls: {len(ctrl):,} genes with the smallest |plateau| on 1,2,3 "
        f"({len(ctrl)/ngen:.0%} of the roster) -- used as an ESTIMATOR, not deleted")
    Dc = np.mean([D_raw[r] for r in TRAIN_REP], axis=0)[:, ctrl]
    Uc, Sc, _ = np.linalg.svd(Dc - Dc.mean(1, keepdims=True), full_matrices=False)
    best_k, best_ink = 0, -np.inf
    for k in KS:
        dirs = Uc[:, :k].T if k > 0 else None
        Dk = {r: ruv_remove(D_raw[r], dirs) for r in TRAIN_REP}
        inner = np.mean([pear(Dk[a], Dk[b]) for a, b in ((1, 2), (1, 3), (2, 3))])
        if inner > best_ink:
            best_ink, best_k = inner, k
    dirs = Uc[:, :best_k].T if best_k > 0 else None
    Dr_all = {r: ruv_remove(D_raw[r], dirs) for r in (1, 2, 3, 4)}
    P_ruv = np.mean([Dr_all[r] for r in TRAIN_REP], axis=0)
    s_ruv, ret_ruv = score(P_ruv, Dr_all[TEST_REP]), retention(P_ruv)
    say(f"     k chosen on replicates 1,2,3 from {KS}: k = {best_k} "
        f"(within-123 agreement {best_ink:+.4f})")
    say(f"     S {s_raw:+.5f} -> {s_ruv:+.5f}   delta {s_ruv-s_raw:+.5f}   retention {ret_ruv:.1%}")
    if best_k > 0:
        rr = []
        for _ in range(20):
            rd = orthonormal_random(best_k, n_iv, rng)
            Dq = {r: ruv_remove(D_raw[r], rd) for r in (1, 2, 3, 4)}
            rr.append(score(np.mean([Dq[r] for r in TRAIN_REP], axis=0), Dq[TEST_REP]))
        rr = np.array(rr, float)
        say(f"     random-direction control, {best_k} random orthonormal directions, 20 draws: "
            f"S {rr.mean():+.5f} +/- {rr.std():.5f}")
        margin = s_ruv - rr.mean()
    else:
        rr = np.array([s_raw]); margin = 0.0
        say("     k = 0 was selected, so RUV removes nothing and the random control is not "
            "applicable")
    ok3 = bool((s_ruv - s_raw) >= GAIN and ret_ruv >= RETAIN and margin >= CTRL_MARGIN)
    G.add("V3", ok3, stat=float(s_ruv - s_raw), requires=("V1",),
          if_true=lambda: f"V3 PASS -- RUV raises S by {s_ruv-s_raw:+.4f} to {s_ruv:+.4f}, "
                          f"{margin:+.4f} above random directions, retaining {ret_ruv:.0%}",
          if_false=lambda: f"V3 FAIL -- delta {s_ruv-s_raw:+.4f} (bar {GAIN:+.2f}), margin over "
                           f"random directions {margin:+.4f} (bar {CTRL_MARGIN:+.2f}), retention "
                           f"{ret_ruv:.0%}")
    if ok3:
        qualified["RUV"] = s_ruv
    res["ruv"] = {"k": int(best_k), "S": s_ruv, "delta": s_ruv - s_raw, "retention": ret_ruv,
                  "random_mean": float(rr.mean()), "margin": float(margin),
                  "n_controls": int(len(ctrl))}

    # ---------------------------------------------------------------- V4
    say("V4 DOES INTERVAL WEIGHTING HELP?")
    w = np.array([np.mean([pear(D_raw[a][j], D_raw[b][j]) for a, b in ((1, 2), (1, 3), (2, 3))])
                  for j in range(n_iv)], float)
    w = np.clip(w, 0.0, None)
    wn = w / (w.mean() + 1e-12)
    say("     within-123 reliability per interval: " +
        "  ".join(f"{GRID[j]}->{GRID[j+1]} {w[j]:+.3f}" for j in range(n_iv)))
    P_w = P_raw * wn[:, None]
    T_w = T_raw * wn[:, None]
    s_w, ret_w = score(P_w, T_w), retention(P_w)
    say(f"     S {s_raw:+.5f} -> {s_w:+.5f}   delta {s_w-s_raw:+.5f}   retention {ret_w:.1%}")
    ok4 = bool((s_w - s_raw) >= GAIN and ret_w >= RETAIN)
    G.add("V4", ok4, stat=float(s_w - s_raw), requires=("V1",),
          if_true=lambda: f"V4 PASS -- weighting raises S by {s_w-s_raw:+.4f} to {s_w:+.4f}",
          if_false=lambda: f"V4 FAIL -- delta {s_w-s_raw:+.4f} against a {GAIN:+.2f} bar, "
                           f"retention {ret_w:.0%}")
    if ok4:
        qualified["WEIGHT"] = s_w
    res["weight"] = {"S": s_w, "delta": s_w - s_raw, "retention": ret_w,
                     "interval_r": [float(x) for x in w]}

    # ---------------------------------------------------------------- V5
    say("V5 DOES THE MEDIAN BEAT THE MEAN?")
    P_med = np.median(np.array([D_raw[r] for r in TRAIN_REP]), axis=0)
    s_med, ret_med = score(P_med, T_raw), retention(P_med)
    say(f"     S {s_raw:+.5f} -> {s_med:+.5f}   delta {s_med-s_raw:+.5f}   "
        f"retention {ret_med:.1%}")
    ok5 = bool((s_med - s_raw) >= GAIN and ret_med >= RETAIN)
    G.add("V5", ok5, stat=float(s_med - s_raw), requires=("V1",),
          if_true=lambda: f"V5 PASS -- the median raises S by {s_med-s_raw:+.4f} to {s_med:+.4f}; "
                          f"the mean was inheriting the shared component from two of three",
          if_false=lambda: f"V5 FAIL -- delta {s_med-s_raw:+.4f} against a {GAIN:+.2f} bar, "
                           f"retention {ret_med:.0%}")
    if ok5:
        qualified["ROBUST"] = s_med
    res["robust"] = {"S": s_med, "delta": s_med - s_raw, "retention": ret_med}

    # ---------------------------------------------------------------- V6
    say("V6 IS THE COMBINATION MORE THAN ITS PARTS?")
    if qualified:
        Vc = {r: whittaker(V[r], best_lam) if "SMOOTH" in qualified else V[r] for r in (1, 2, 3, 4)}
        Dc2 = {r: diff_of(Vc[r]) for r in (1, 2, 3, 4)}
        if "RUV" in qualified:
            Dm = np.mean([Dc2[r] for r in TRAIN_REP], axis=0)[:, ctrl]
            Uc2, _, _ = np.linalg.svd(Dm - Dm.mean(1, keepdims=True), full_matrices=False)
            d2 = Uc2[:, :best_k].T if best_k > 0 else None
            Dc2 = {r: ruv_remove(Dc2[r], d2) for r in (1, 2, 3, 4)}
        if "ROBUST" in qualified:
            Pc = np.median(np.array([Dc2[r] for r in TRAIN_REP]), axis=0)
        else:
            Pc = np.mean([Dc2[r] for r in TRAIN_REP], axis=0)
        Tc = Dc2[TEST_REP]
        if "WEIGHT" in qualified:
            Pc, Tc = Pc * wn[:, None], Tc * wn[:, None]
        s_c, ret_c = score(Pc, Tc), retention(Pc)
        best_single = max(qualified.values())
        best_name = max(qualified, key=qualified.get)
        say(f"     qualified patches applied in sequence: {', '.join(sorted(qualified))}")
        say(f"     combination S {s_c:+.5f}   best single ({best_name}) {best_single:+.5f}   "
            f"delta {s_c-best_single:+.5f}   retention {ret_c:.1%}")
        ok6 = bool((s_c - best_single) >= COMBO_MARGIN and ret_c >= RETAIN)
    else:
        s_c, ret_c, best_single, best_name = float("nan"), float("nan"), float("nan"), None
        say("     no single patch qualified, so there is nothing to combine")
        ok6 = False
    G.add("V6", ok6, stat=float(s_c) if qualified else None, requires=("V1",),
          void_if=(not qualified),
          void_reason="no single patch qualified, so combination has nothing to combine",
          if_true=lambda: f"V6 PASS -- the combination reaches {s_c:+.4f}, {s_c-best_single:+.4f} "
                          f"above the best single patch",
          if_false=lambda: f"V6 FAIL -- {s_c:+.4f} against the best single {best_single:+.4f}; "
                           f"the patches are correcting the same thing")
    res["combo"] = {"S": s_c, "best_single": best_single, "best_name": best_name,
                    "retention": ret_c, "members": sorted(qualified)}

    # ---------------------------------------------------------------- V7
    say("V7 DOES THE WINNER TRANSFER TO A FORWARD TASK?")
    if qualified:
        winner = best_name if not ok6 else "COMBO"
        say(f"     winner by S: {winner}")

        def forward(Pm, Tm):
            """Ridge on intervals 0..5 predicting 6..7 from previous change + running level."""
            lvl = np.cumsum(Pm, axis=0)
            Xtr, ytr, Xte, yte, pers_te = [], [], [], [], []
            for j in range(1, n_iv):
                X = np.column_stack([Pm[j - 1], lvl[j - 1], np.ones(ngen)])
                if j < n_iv - 2:
                    Xtr.append(X); ytr.append(Tm[j])
                else:
                    Xte.append(X); yte.append(Tm[j]); pers_te.append(Pm[j - 1])
            Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
            Xte = np.vstack(Xte); yte = np.concatenate(yte)
            pers_te = np.concatenate(pers_te)
            A = Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])
            b = np.linalg.solve(A, Xtr.T @ ytr)
            pred = Xte @ b
            def r2(y, p):
                ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
                return 1 - ss / tt if tt > 0 else float("nan")
            return r2(yte, pred), r2(yte, pers_te)

        m_raw, p_raw_b = forward(P_raw, T_raw)
        m_pat, p_pat_b = forward(Pc if ok6 else
                                 {"SMOOTH": P_sm, "RUV": P_ruv, "WEIGHT": P_w,
                                  "ROBUST": P_med}[winner],
                                 Tc if ok6 else
                                 {"SMOOTH": Ds_all[TEST_REP], "RUV": Dr_all[TEST_REP],
                                  "WEIGHT": T_w, "ROBUST": T_raw}[winner])
        lift_raw, lift_pat = m_raw - p_raw_b, m_pat - p_pat_b
        say(f"     unpatched: ridge R2 {m_raw:+.5f}   its own persistence {p_raw_b:+.5f}   "
            f"lift {lift_raw:+.5f}")
        say(f"     patched  : ridge R2 {m_pat:+.5f}   its own persistence {p_pat_b:+.5f}   "
            f"lift {lift_pat:+.5f}")
        ok7 = bool(np.isfinite(lift_pat) and np.isfinite(lift_raw) and lift_pat > lift_raw)
        stat7 = float(lift_pat - lift_raw)
    else:
        m_raw = p_raw_b = m_pat = p_pat_b = lift_raw = lift_pat = float("nan")
        winner, stat7, ok7 = None, None, False
    G.add("V7", ok7, stat=stat7, requires=("V1",), void_if=(not qualified),
          void_reason="no patch qualified, so there is no winner to transfer",
          if_true=lambda: f"V7 PASS -- the patched model beats its own persistence by "
                          f"{lift_pat:+.4f} against {lift_raw:+.4f} unpatched; the correction "
                          f"survives a forward task",
          if_false=lambda: f"V7 FAIL -- patched lift {lift_pat:+.4f} against unpatched "
                           f"{lift_raw:+.4f}; the gain in S does not reach a forward task")
    res["forward"] = {"winner": winner, "raw_model": m_raw, "raw_persistence": p_raw_b,
                      "patched_model": m_pat, "patched_persistence": p_pat_b,
                      "lift_raw": lift_raw, "lift_patched": lift_pat}

    # ---------------------------------------------------------------- V8
    say("V8 WHAT THIS CANNOT SHOW")
    say("     S is agreement between a three-replicate average and one held-out replicate of one")
    say("     dexamethasone series in one cell line. Nothing here measures the true transcription")
    say("     rate, so a higher S is a less contaminated estimator, not a verified one.")
    say("     If SMOOTH wins, part of that win is arithmetic: smoothing correlates neighbouring")
    say("     intervals, so pooled Pearson over intervals rises even when no single gene is")
    say("     estimated better. V7 is the only gate that separates the two, because a forward")
    say("     task cannot be helped by having smeared the answer into its own input.")
    say("     The negative controls for RUV are chosen by small plateau, which selects genes that")
    say("     are quiet OR poorly measured; if the component lives mostly in quiet genes then RUV")
    say("     estimates it well and if it lives in responders then RUV cannot see it at all.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
