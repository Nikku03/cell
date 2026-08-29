"""
LOOP 270 -- THE TWO NUMBERS LOOP 268 FLAGGED AS UNSAFE TO QUOTE

Loop 268 recorded two problems in its own output rather than reporting past them. This
resolves both. Neither changes a gate verdict in loop 268; both change what its numbers mean.

  1. THE TRANSFER CURVE IS NOT SMOOTH. Step ratios ran 1.45, 1.94, 1.81, 1.46, 1.41, 1.61
     and then 4.46 at the final point, n=48. That point is also special: 48 IS the number of
     available training lines, so the "random subset" is the entire set, both repeats are
     identical, and there is no sampling variation at all. A 4.46x step after six steps
     averaging 1.6x is either a real threshold or an artefact of that specialness, and
     loop 268's headline +0.0824 rests on it.

  2. THE DOSE REPLICATION MOVED THREE THINGS. Splitting by concentration halves the data,
     and K is derived from the conditions available, so dose 0.05 ran 253 conditions per
     line at K=126 while dose 0.5 ran 204 at K=102. Dose, sample size and operator size all
     moved together. That is DEFECT I, a control that moves more than one thing, which I
     added to the ledger after the wrong-line control in loops 256-259 and then wrote again.

GATES, DECLARED BEFORE THE NUMBERS:

  W1 IS THE JUMP AT n=48 REAL?
     The curve at n = 36, 40, 42, 44, 46, 47, 48, with n=47 and n=48 both included so the
     "all available lines" point is bracketed by one that is not.
     Gate: PASS iff no step ratio exceeds 2.5x, i.e. the curve is smooth and the jump was
     an artefact of the coarse grid rather than a threshold.
     A FAIL means the jump is real and concentrated in the last one or two lines, which
     would itself need explaining before loop 268's magnitude is quoted.

  W2 DO THE DOSES AGREE ONCE MATCHED?                                -- requires W1
     Both doses subsampled to the SAME number of conditions per line and fitted with the
     SAME K, so dose is the only thing that differs.
     Gate: PASS iff the off-diagonal statistics agree within 0.01, loop 268's U6 tolerance.
     A PASS means U6's FAIL was my confound, not concentration dependence. A FAIL means the
     effect really does depend on dose, and U6 was right for the wrong reason.

  W3 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, collections
from pathlib import Path
import numpy as np

from gate_guard import Gates
from loop_tahoe import build, decompose, fit_dense, diag_fit, pear

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_tahoe_diagnostics.json"
SEED = 270270
FINE_N = [36, 40, 42, 44, 46, 47, 48]
NHOLD, NREP = 20, 3
LAM = [1e1, 1e2, 1e3, 1e4]
W1_MAX_RATIO, W2_TOL = 2.5, 0.01
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def prep(M, combo, cell, K):
    sel = np.argsort(-M.var(0))[:K]
    Ms = M[:, sel]
    A, R, _, _, _ = decompose(Ms, combo, cell)
    cmean = {k: Ms[combo == k].mean(0) for k in np.unique(combo)}
    Xg = np.stack([cmean[k] for k in combo]) - Ms.mean(0)
    return Ms, A, R, Xg


def pick_lam(Xg, R, ix, rng):
    q = rng.permutation(len(ix)); q1, q2 = ix[q[:len(q)//2]], ix[q[len(q)//2:]]
    best, be = LAM[0], np.inf
    for lam in LAM:
        W_ = fit_dense(Xg[q1], R[q1], lam)
        e = float(((Xg[q2] @ W_ - R[q2]) ** 2).mean())
        if e < be: be, best = e, lam
    return best


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "resolve the two numbers loop 268 flagged as unsafe to quote"}
    say("=" * 104)
    say("LOOP 270 -- THE TWO NUMBERS LOOP 268 FLAGGED AS UNSAFE TO QUOTE")
    say("=" * 104)
    say("     Loop 268 recorded both problems in its own output rather than reporting past")
    say("     them. Neither changes a gate verdict there; both change what its numbers mean.")

    rng = np.random.default_rng(SEED)
    M, combo, cell, genes = build()
    lines = sorted(set(cell.tolist()))
    npc = int(np.median([int((cell == l).sum()) for l in lines]))
    K = max(50, min(len(genes), int(npc / 2.0)))
    Ms, A, R, Xg = prep(M, combo, cell, K)
    say(f"     {len(lines)} lines, ~{npc} conditions each, K={K} as loop 268 used")

    say("W1 IS THE JUMP AT n=48 REAL?")
    curve, null = collections.defaultdict(list), collections.defaultdict(list)
    hold = list(rng.choice(len(lines), size=min(NHOLD, len(lines)), replace=False))
    for hi, li in enumerate(hold):
        hold_l = lines[li]
        hm = cell == hold_l
        ix = np.where(hm)[0]
        p = rng.permutation(len(ix)); h1, h2 = ix[p[:len(p)//2]], ix[p[len(p)//2:]]
        b0 = float(np.nanmean([pear(A[i], Ms[i]) for i in h2]))
        sc = lambda P: float(np.nanmean([pear(A[h2[j]] + P[j], Ms[h2[j]])
                                         for j in range(len(h2))]))
        tr_lines = [l for l in lines if l != hold_l]
        Wtr = {}
        for l in tr_lines:
            m = np.where(cell == l)[0]
            Wtr[l] = fit_dense(Xg[m], R[m], 1e3)
        for n in FINE_N:
            nn = min(n, len(tr_lines))
            for _ in range(NREP):
                s = rng.choice(len(tr_lines), size=nn, replace=False)
                for tag, mats in (("real", [Wtr[tr_lines[i]] for i in s]),
                                  ("null", [Wtr[tr_lines[i]][:, rng.permutation(K)]
                                            for i in s])):
                    P1 = np.stack([Xg[h1] @ W_ for W_ in mats])
                    P2 = np.stack([Xg[h2] @ W_ for W_ in mats])
                    F = P1.reshape(nn, -1)
                    a = np.linalg.solve(F @ F.T + 1e-6 * np.eye(nn), F @ R[h1].ravel())
                    (curve if tag == "real" else null)[nn].append(
                        sc(np.tensordot(a, P2, axes=(0, 0))) - b0)
        del Wtr
        if hi % 5 == 0:
            say(f"       held-out line {hi+1}/{len(hold)}   [{time.time()-t0:.0f}s]")
    ns = sorted(curve)
    exc = {n: float(np.mean(curve[n]) - np.mean(null[n])) for n in ns}
    say(f"     {'n':>4s} {'excess':>10s} {'null':>10s} {'step ratio':>12s}")
    ratios = []
    for i, n in enumerate(ns):
        r = exc[n] / exc[ns[i-1]] if i and exc[ns[i-1]] > 1e-9 else float("nan")
        if i: ratios.append(r)
        say(f"     {n:4d} {exc[n]:+10.4f} {np.mean(null[n]):+10.4f} "
            f"{('' if i == 0 else f'{r:.2f}x'):>12s}")
    worst = max(ratios) if ratios else float("nan")
    say(f"     largest step ratio {worst:.2f}x; loop 268's coarse grid showed 4.46x at n=48")
    G_.add("W1", bool(worst <= W1_MAX_RATIO), stat=float(worst),
           if_true=lambda: f"W1 PASS -- no step exceeds {worst:.2f}x, so the curve is smooth "
                           f"and loop 268's 4.46x was an artefact of its coarse grid",
           if_false=lambda: f"W1 FAIL -- a step of {worst:.2f}x survives the finer grid, so "
                            f"the jump is concentrated in the last lines and needs explaining "
                            f"before loop 268's magnitude is quoted")
    res["W1"] = {"n": ns, "excess": [exc[n] for n in ns],
                 "null": [float(np.mean(null[n])) for n in ns], "worst_ratio": worst}

    say("W2 DO THE DOSES AGREE ONCE MATCHED?")
    per = {}
    nmin, kmin = None, None
    slices = {}
    for dose in (0.05, 0.5):
        Md, cb, cl, gn = build(dose=dose)
        ln = sorted(set(cl.tolist()))
        # The floor must be the MINIMUM over all (line, dose) cells, not the median. Setting
        # it from the median and then drawing that many from every line asks the lines below
        # the median for more conditions than they have, which is what crashed run 1.
        npc_d = min(int((cl == l).sum()) for l in ln)
        slices[dose] = (Md, cb, cl, gn, npc_d)
        nmin = npc_d if nmin is None else min(nmin, npc_d)
    kmin = max(50, int(nmin / 2.0))
    say(f"     matched: {nmin} conditions per line and K={kmin} for BOTH doses")
    say(f"     loop 268 used 253/K=126 and 204/K=102 -- three things moving at once")
    for dose in (0.05, 0.5):
        Md, cb, cl, gn, _ = slices[dose]
        ln = sorted(set(cl.tolist()))
        r2 = np.random.default_rng(SEED)
        keep = np.concatenate([r2.choice(np.where(cl == l)[0], size=nmin, replace=False)
                               for l in ln])
        Md, cb, cl = Md[keep], cb[keep], cl[keep]
        Ms2, A2, R2, Xg2 = prep(Md, cb, cl, kmin)
        full, dg = [], []
        hl = list(r2.choice(len(ln), size=min(NHOLD, len(ln)), replace=False))
        for li in hl:
            m = np.where(cl == ln[li])[0]
            p = r2.permutation(len(m)); a_, b_ = m[p[:len(p)//2]], m[p[len(p)//2:]]
            lam = pick_lam(Xg2, R2, a_, r2)
            W = fit_dense(Xg2[a_], R2[a_], lam)
            sd = diag_fit(Xg2[a_], R2[a_])
            b0 = float(np.nanmean([pear(A2[i], Ms2[i]) for i in b_]))
            full.append(float(np.nanmean([pear(A2[i] + Xg2[i] @ W, Ms2[i]) for i in b_])) - b0)
            dg.append(float(np.nanmean([pear(A2[i] + sd * Xg2[i], Ms2[i]) for i in b_])) - b0)
        per[dose] = {"full": float(np.mean(full)), "diag": float(np.mean(dg)),
                     "off": float(np.mean(full) - np.mean(dg))}
        say(f"     dose {dose}: full {per[dose]['full']:+.4f}, diagonal "
            f"{per[dose]['diag']:+.4f}, off-diagonal {per[dose]['off']:+.4f}")
    d2 = abs(per[0.05]["off"] - per[0.5]["off"])
    say(f"     matched difference {d2:.4f}; loop 268's unmatched U6 reported 0.0430")
    G_.add("W2", bool(d2 <= W2_TOL), stat=float(d2), requires=("W1",),
           if_true=lambda: f"W2 PASS -- once conditions and K are matched the doses agree to "
                           f"{d2:.4f}. Loop 268's U6 FAIL was my confound, not concentration "
                           f"dependence, and the off-diagonal result replicates across dose",
           if_false=lambda: f"W2 FAIL -- matched on conditions and K the doses still differ by "
                            f"{d2:.4f}; U6's verdict stands and the effect really does depend "
                            f"on concentration")
    res["W2"] = {"matched_n": int(nmin), "matched_K": int(kmin),
                 "per_dose": {str(k): v for k, v in per.items()}, "delta": d2}

    say("W3 WHAT THIS CANNOT SHOW")
    say("     W1 tests smoothness on a finer grid of the SAME sampling scheme. If the jump")
    say("     came from something other than grid coarseness -- a threshold in how many lines")
    say("     span the held-out one -- a smooth curve here would not detect it.")
    say("     W2 matches conditions and K. It does not match which DRUGS appear at each dose,")
    say("     and if the two doses were run on different compound sets that difference")
    say("     remains inside the comparison.")
    say("     Neither gate revises loop 268's verdicts. U4 and U6 stand as recorded; what")
    say("     changes is which of their numbers may be quoted and what they mean.")

    res["gates"] = {k: (v == "PASS") for k, v in G_.status.items()}
    res["void"] = [k for k, v in G_.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G_.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
