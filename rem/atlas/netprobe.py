"""Watch the channel tree while it learns: where is it weak, and which repairs earn a decision?

WHAT IS KNOWN. The channel tree solves 12-bit parity over 16 inputs from 4,000 examples (~0.95),
fails at 1,500 (~0.58), and beats a matched deep MLP on sample efficiency (edgedecide.py, 9/10
fresh seeds). Nothing has yet looked INSIDE it while it trains.

ONE THING NOTICED BEFORE ANY RUN (algebra, not measurement). At initialisation g_dep = g_hyp = 0.5
and theta = 0, so every node computes
    out = v + 0.5 m (1 - v) + 0.5 m (-1 - v) = v (1 - m) = v * sigmoid(-v) = v - SiLU(v),
which is ALREADY non-monotone (rises to 0.278 at v = 1.278, then falls back to 0). The network
starts with a fixed bend in every node; whether learning the conductances adds anything is R3 below.

=================================================================================================
PART A -- INSTRUMENTED TRAINING. k = 12, FRESH seeds 40-44, n in {1500, 4000} (fails vs succeeds).
Diagnostics are read every 100 steps without touching the random stream; stopping is unchanged.
Every diagnostic has a flag fixed now. A flag that fires is a named weak point.
=================================================================================================

A0  BLOCKING. Instrumentation must not perturb training: identical steps, train and test accuracy
    to the uninstrumented trainer (nonbio.job) on every (seed, n).

A1  MEMORISATION. Flag if, at some n, >= 3/5 seeds stop with train = 1.0 and test < 0.70.
A2  STOPPING RULE. Flag if the best test accuracy seen during training beats the final one by
    >= 0.03 on average at some n (stopping at train = 1.0 throws accuracy away).
A3  INPUT SELECTION. (a) share of leaf squared-weight mass on the 12 relevant inputs (chance 0.75);
    flag if < 0.85 at stop at n = 1500. (b) sensitivity: flip one input bit across the test set, share
    of predictions that change. Perfect parity: 1.0 on relevant bits, 0.0 on irrelevant ones.
    Flag if irrelevant-bit sensitivity > 0.10 at n = 1500.
A4  GRADIENT FLOW. Mean relative gradient |g| / |p| per parameter group over training. Flag
    VANISHING if any group's is < 1/100 of the largest group's.
A5  CHANNEL USE. Mean |change from init| of g_dep, g_hyp, theta per level: flag UNUSED if < 0.05 at
    a level. Gate saturation, share of m outside [0.02, 0.98]: flag SATURATED if > 0.5 at a level.
A6  BEND IN USE. Node slope d out/dv = 1 - G m + (D - G v) m (1 - m), G = g_dep + g_hyp,
    D = g_dep - g_hyp. A node USES THE BEND if >= 10% of training inputs land where the slope < 0.
    Flag if < 10% of nodes use it at some level.
A7  SUBUNIT USE. Contribution of subunit u = |wo_u| * std(s_u); effective number exp(entropy) of
    the shares. Flag REDUNDANT if < 4 of 8.
A8  COST. Milliseconds per training step and where they go, against the deep MLP. Flag COST if the
    channel tree is > 3x slower per step. The leaf contraction is also timed in matmul form; the
    two forms must agree to 1e-10 before a speed-up is reported.

=================================================================================================
PART B -- REPAIR SCREEN. Each candidate is fixed now and RUN WHETHER OR NOT ITS FLAG FIRES.
Seeds 40-44, n in {1500, 2000, 2500} (where the tree is weakest), budget 10,000 steps.
Unit = per-seed mean test accuracy over the three n. Compared against the unmodified tree.
=================================================================================================

    R1  leaf sparsity     proximal L1 on leaf weights, shrink 1e-4 per step    targets A1, A3
    R2  validation stop   hold out the last 10% of the training set; return the weights with the
                          best validation accuracy (ties -> later)                targets A2
    R3  frozen channels   g_dep, g_hyp, theta fixed at init (every node = v - SiLU(v))
                          -> does LEARNING the conductances matter?               attribution
    R4  shallower tree    depth 3 (64 leaves per subunit, 4x fewer leaf weights)  targets A8

    PROMISING   mean gain >= +0.02 AND higher on >= 4/5 seeds
    HARMFUL     mean loss >= 0.02 AND lower on >= 4/5 seeds
    otherwise   NO CLEAR EFFECT
    R3 and R4 are simplifications: additionally SAFE if mean change >= -0.01.
    THIS IS A SCREEN. 4/5 has sign p = 0.19; a PROMISING repair earns a predeclared 10-seed decision,
    not a claim.

A9  WHAT THIS IS AND IS NOT. One task, one tree shape, one learning rate.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import json
import math
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_netprobe.txt")
ART = R("outputs", "netprobe.json")
RULE = "=" * 97

_s = importlib.util.spec_from_file_location("nonbio", os.path.join(HERE, "nonbio.py"))
nb = importlib.util.module_from_spec(_s); _s.loader.exec_module(nb)
lim, ch = nb.lim, nb.ch

K = 12
SEEDS = (40, 41, 42, 43, 44)
N_PROBE = (1500, 4000)
N_SCREEN = (1500, 2000, 2500)
EVERY = 100
SCREEN_CAP = 10000
L1_SHRINK = 1e-4
ARMS = ("R0", "R1", "R2", "R3", "R4")
ARM = {"R0": "unmodified tree", "R1": "leaf sparsity (L1)", "R2": "validation stop",
       "R3": "frozen channels", "R4": "shallower tree (depth 3)"}


def relevant(seed):
    rng = np.random.default_rng(100000 * K + 1000 * seed + 7)
    return np.sort(rng.choice(lim.D, size=K, replace=False))


def groups(m):
    L = m.L
    return (["leaf W"] + [f"W L{l + 1}" for l in range(L)] + [f"theta L{l + 1}" for l in range(L)]
            + [f"g_dep L{l + 1}" for l in range(L)] + [f"g_hyp L{l + 1}" for l in range(L)]
            + ["readout w", "readout b"])


def inside(m, X):
    """Read the tree's state on X: saturation, bend use, subunit shares, per-level spread."""
    m.forward(X)
    lv = []
    for l in range(m.L):
        v, g = m.cV[l], m.cM[l]
        G, Dd = m.gd[l][None] + m.gh[l][None], m.gd[l][None] - m.gh[l][None]
        slope = 1 - G * g + (Dd - G * v) * g * (1 - g)
        neg = (slope < 0).mean(0)
        lv.append(dict(saturated=float(((g < 0.02) | (g > 0.98)).mean()),
                       bend_nodes=float((neg >= 0.10).mean()),
                       falling_share=float((slope < 0).mean()),
                       spread=float(m.cH[l + 1].std())))
    c = np.abs(m.wo) * m.s.std(0)
    sh = c / c.sum()
    eff = float(np.exp(-(sh * np.log(sh + 1e-300)).sum()))
    return lv, sh.tolist(), eff


def sensitivity(m, X):
    base = m.forward(X) > 0
    out = []
    for d in range(X.shape[1]):
        Xf = X.copy(); Xf[:, d] *= -1
        out.append(float(((m.forward(Xf) > 0) != base).mean()))
    return out


def probe_job(args):
    seed, ntr = args
    Xtr, ytr, Xte, yte = lim.data(K, seed, ntr)
    sub = relevant(seed)
    assert np.array_equal(ytr, (((Xtr[:, sub] + 1) / 2).sum(1) % 2)), "relevant-bit reconstruction"
    m = lim.build("channel", np.random.default_rng(200 + seed))
    init = dict(th=[x.copy() for x in m.th], gd=[x.copy() for x in m.gd], gh=[x.copy() for x in m.gh])
    lv0, _, eff0 = inside(m, Xtr)
    mass0 = float((m.Wl[:, :, sub] ** 2).sum() / (m.Wl ** 2).sum())
    rng = np.random.default_rng(300 + seed)
    st, steps, traj = {}, 0, []
    rel = np.zeros(len(m.ps)); nrel = 0
    t0 = time.time()
    while steps < nb.MAXSTEPS:
        for j in range(lim.CHECK):
            i = rng.integers(0, len(Xtr), lim.BS)
            z = m.forward(Xtr[i])
            gs = m.backward((1 / (1 + np.exp(-z)) - ytr[i]) / lim.BS)
            rel += [np.linalg.norm(g) / (np.linalg.norm(p) + 1e-12) for p, g in zip(m.ps, gs)]
            nrel += 1
            ch.adam(m.ps, gs, st, lim.LR)
            if (steps + j + 1) % EVERY == 0:
                traj.append(dict(step=steps + j + 1, train=lim.acc(m, Xtr, ytr),
                                 test=lim.acc(m, Xte, yte), loss=ch.loss_of(m, Xtr, ytr)))
        steps += lim.CHECK
        if not np.all(np.isfinite(m.forward(Xtr[:64]))):
            break
        if lim.acc(m, Xtr, ytr) == 1.0:
            break
    lv, shares, eff = inside(m, Xtr)
    moved = [dict(gd=float(np.abs(m.gd[l] - init["gd"][l]).mean()),
                  gh=float(np.abs(m.gh[l] - init["gh"][l]).mean()),
                  th=float(np.abs(m.th[l] - init["th"][l]).mean())) for l in range(m.L)]
    sens = sensitivity(m, Xte)
    isrel = np.isin(np.arange(lim.D), sub)
    return dict(seed=seed, ntr=ntr, steps=steps, train=lim.acc(m, Xtr, ytr), test=lim.acc(m, Xte, yte),
                seconds=round(time.time() - t0, 1), traj=traj,
                best_test=max(t["test"] for t in traj) if traj else None,
                rel_grad=dict(zip(groups(m), (rel / max(nrel, 1)).tolist())),
                mass_init=mass0, mass_stop=float((m.Wl[:, :, sub] ** 2).sum() / (m.Wl ** 2).sum()),
                sens_rel=float(np.mean([sens[d] for d in range(lim.D) if isrel[d]])),
                sens_irr=float(np.mean([sens[d] for d in range(lim.D) if not isrel[d]])),
                levels_init=lv0, levels=lv, moved=moved, shares=shares, eff_init=eff0, eff=eff,
                gd_stop=[float(x.mean()) for x in m.gd], gh_stop=[float(x.mean()) for x in m.gh],
                th_stop=[float(x.mean()) for x in m.th])


def ref_job(args):
    seed, ntr = args
    return nb.job(("channel", K, seed, ntr))


def screen_job(args):
    arm, seed, ntr = args
    Xtr, ytr, Xte, yte = lim.data(K, seed, ntr)
    if arm == "R2":
        cut = int(round(0.9 * len(Xtr)))
        Xtr, ytr, Xva, yva = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]
    brng = np.random.default_rng(200 + seed)
    m = ch.ChannelTree(lim.D, 8, lim.B, 3, brng) if arm == "R4" else lim.build("channel", brng)
    frozen = [x.copy() for x in m.th + m.gd + m.gh]
    rng = np.random.default_rng(300 + seed)
    st, steps, best, snap = {}, 0, -1.0, None
    while steps < SCREEN_CAP:
        for j in range(lim.CHECK):
            i = rng.integers(0, len(Xtr), lim.BS)
            z = m.forward(Xtr[i])
            ch.adam(m.ps, m.backward((1 / (1 + np.exp(-z)) - ytr[i]) / lim.BS), st, lim.LR)
            if arm == "R1":
                m.Wl[...] = np.sign(m.Wl) * np.maximum(np.abs(m.Wl) - L1_SHRINK, 0.0)
            if arm == "R3":
                for p, f in zip(m.th + m.gd + m.gh, frozen):
                    p[...] = f
            if arm == "R2" and (steps + j + 1) % EVERY == 0:
                va = lim.acc(m, Xva, yva)
                if va >= best:
                    best, snap = va, [p.copy() for p in m.ps]
        steps += lim.CHECK
        if not np.all(np.isfinite(m.forward(Xtr[:64]))):
            break
        if lim.acc(m, Xtr, ytr) == 1.0:
            break
    fit = lim.acc(m, Xtr, ytr)
    if arm == "R2" and snap is not None:
        for p, q in zip(m.ps, snap):
            p[...] = q
    return dict(arm=arm, seed=seed, ntr=ntr, steps=steps, train=fit, test=lim.acc(m, Xte, yte),
                params=ch.nparams(m), best_val=best if arm == "R2" else None)


def cost():
    """Per-step time and its split, single process, after the pool has closed."""
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 1.0], size=(lim.BS, lim.D)); y = rng.integers(0, 2, lim.BS).astype(float)
    reps = 200
    res = {}
    for name in ("channel", "mlpdeep"):
        m = lim.build(name, np.random.default_rng(1)); st = {}
        tf = tb = ta = 0.0
        for _ in range(reps):
            a = time.perf_counter(); z = m.forward(X)
            b = time.perf_counter(); gs = m.backward((1 / (1 + np.exp(-z)) - y) / lim.BS)
            c = time.perf_counter(); ch.adam(m.ps, gs, st, lim.LR)
            d = time.perf_counter()
            tf += b - a; tb += c - b; ta += d - c
        res[name] = dict(forward=1e3 * tf / reps, backward=1e3 * tb / reps, adam=1e3 * ta / reps,
                         total=1e3 * (tf + tb + ta) / reps)
    m = lim.build("channel", np.random.default_rng(1))
    U, Kl, D = m.Wl.shape
    dh = rng.normal(size=(lim.BS, U, Kl))
    t = {}
    a = time.perf_counter()
    for _ in range(reps): e1 = np.einsum('nd,ukd->nuk', X, m.Wl)
    t["leaf fwd einsum"] = 1e3 * (time.perf_counter() - a) / reps
    a = time.perf_counter()
    for _ in range(reps): m1 = (X @ m.Wl.reshape(U * Kl, D).T).reshape(lim.BS, U, Kl)
    t["leaf fwd matmul"] = 1e3 * (time.perf_counter() - a) / reps
    a = time.perf_counter()
    for _ in range(reps): e2 = np.einsum('nuk,nd->ukd', dh, X)
    t["leaf grad einsum"] = 1e3 * (time.perf_counter() - a) / reps
    a = time.perf_counter()
    for _ in range(reps): m2 = (dh.reshape(lim.BS, U * Kl).T @ X).reshape(U, Kl, D)
    t["leaf grad matmul"] = 1e3 * (time.perf_counter() - a) / reps
    agree = float(max(np.abs(e1 - m1).max(), np.abs(e2 - m2).max()))
    return res, t, agree


def verdict(diffs, simplification):
    w = sum(d > 0 for d in diffs); l = sum(d < 0 for d in diffs); md = float(np.mean(diffs))
    v = ("PROMISING" if md >= 0.02 and w >= 4 else "HARMFUL" if md <= -0.02 and l >= 4
         else "NO CLEAR EFFECT")
    if simplification:
        v += ", SAFE simplification" if md >= -0.01 else ", NOT a safe simplification"
    return w, l, md, v


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)
    P_(RULE); P_("PART A -- THE CHANNEL TREE, INSTRUMENTED WHILE IT LEARNS"); P_(RULE)
    cells = [(s, n) for n in N_PROBE for s in SEEDS]
    probes = pool.map(probe_job, cells)
    refs = pool.map(ref_job, cells)
    same = all(p["steps"] == r["steps"] and p["train"] == r["train"] and p["test"] == r["test"]
               for p, r in zip(probes, refs))
    P_(f"  A0 instrumentation leaves training untouched (steps, train, test identical on 10/10): "
       f"{'PASS' if same else 'FAIL'}")
    if not same:
        for p, r in zip(probes, refs):
            P_(f"    seed {p['seed']} n {p['ntr']}: probe {p['steps']} {p['test']:.4f}   ref {r['steps']} {r['test']:.4f}")
        open(OUT, "w").write("\n".join(out) + "\n"); pool.close(); return

    flags = {}
    by = {n: [p for p in probes if p["ntr"] == n] for n in N_PROBE}

    P_("\n  A1/A2  FIT VERSUS GENERALISE")
    for n in N_PROBE:
        P_(f"    n = {n}")
        for p in by[n]:
            curve = "  ".join(f"{t['step']}:{t['train']:.2f}/{t['test']:.2f}" for t in p["traj"][::2])
            P_(f"      seed {p['seed']}  stop {p['steps']:>5}  train {p['train']:.3f}  test {p['test']:.3f}"
               f"  best-seen {p['best_test']:.3f}   [step:train/test] {curve}")
    a1 = {n: sum(p["train"] == 1.0 and p["test"] < 0.70 for p in by[n]) for n in N_PROBE}
    a2 = {n: float(np.mean([p["best_test"] - p["test"] for p in by[n]])) for n in N_PROBE}
    flags["A1 memorisation"] = any(v >= 3 for v in a1.values())
    flags["A2 stopping rule"] = any(v >= 0.03 for v in a2.values())
    P_(f"    A1 seeds memorising (train 1.0, test < 0.70): " + ", ".join(f"n={n}: {a1[n]}/5" for n in N_PROBE)
       + f"  -> {'FLAG' if flags['A1 memorisation'] else 'ok'}")
    P_(f"    A2 best-seen minus final test: " + ", ".join(f"n={n}: {a2[n]:+.3f}" for n in N_PROBE)
       + f"  -> {'FLAG' if flags['A2 stopping rule'] else 'ok'}")

    P_("\n  A3  WHICH INPUTS THE LEAVES LISTEN TO (12 relevant, 4 irrelevant)")
    for n in N_PROBE:
        ps = by[n]
        P_(f"    n = {n}: leaf mass on relevant inputs {np.mean([p['mass_init'] for p in ps]):.3f} at init -> "
           f"{np.mean([p['mass_stop'] for p in ps]):.3f} at stop   |   flip sensitivity: relevant bits "
           f"{np.mean([p['sens_rel'] for p in ps]):.3f}, irrelevant bits {np.mean([p['sens_irr'] for p in ps]):.3f}")
    ma = float(np.mean([p["mass_stop"] for p in by[1500]]))
    si = float(np.mean([p["sens_irr"] for p in by[1500]]))
    flags["A3a leaf mass off-target"] = ma < 0.85
    flags["A3b irrelevant-bit sensitivity"] = si > 0.10
    P_(f"    A3a -> {'FLAG' if flags['A3a leaf mass off-target'] else 'ok'}   A3b -> "
       f"{'FLAG' if flags['A3b irrelevant-bit sensitivity'] else 'ok'}")

    P_("\n  A4  GRADIENT FLOW, mean |g|/|p| per group (all 10 runs)")
    names = list(probes[0]["rel_grad"])
    rg = {g: float(np.mean([p["rel_grad"][g] for p in probes])) for g in names}
    for g in names:
        P_(f"    {g:<11} {rg[g]:.2e}")
    top = max(rg.values())
    weak = [g for g in names if rg[g] < top / 100]
    flags["A4 vanishing gradient"] = bool(weak)
    P_(f"    A4 groups below 1/100 of the largest: {weak if weak else 'none'}  -> "
       f"{'FLAG' if weak else 'ok'}")

    L = len(probes[0]["levels"])
    P_("\n  A5/A6  CHANNELS AND THE BEND, per level (1 = just above the leaves, 4 = the soma)")
    P_("    level   moved g_dep  g_hyp  theta   saturated   nodes using bend (init -> stop)   falling share")
    unused = sat = nobend = False
    for l in range(L):
        mv = {k: float(np.mean([p["moved"][l][k] for p in probes])) for k in ("gd", "gh", "th")}
        s_ = float(np.mean([p["levels"][l]["saturated"] for p in probes]))
        b0 = float(np.mean([p["levels_init"][l]["bend_nodes"] for p in probes]))
        b1 = float(np.mean([p["levels"][l]["bend_nodes"] for p in probes]))
        fs = float(np.mean([p["levels"][l]["falling_share"] for p in probes]))
        P_(f"    L{l + 1}      {mv['gd']:>11.3f}  {mv['gh']:.3f}  {mv['th']:.3f}   {s_:>9.3f}   "
           f"{b0:>14.3f} -> {b1:.3f}           {fs:.3f}")
        unused |= max(mv.values()) < 0.05
        sat |= s_ > 0.5
        nobend |= b1 < 0.10
    P_("    learned values at stop, mean over runs: " + "   ".join(
        f"L{l + 1} g_dep {np.mean([p['gd_stop'][l] for p in probes]):.2f} g_hyp "
        f"{np.mean([p['gh_stop'][l] for p in probes]):.2f} theta {np.mean([p['th_stop'][l] for p in probes]):+.2f}"
        for l in range(L)))
    flags["A5 unused channels"] = unused
    flags["A5 saturated gates"] = sat
    flags["A6 bend not in use"] = nobend
    P_(f"    A5 unused -> {'FLAG' if unused else 'ok'}   saturated -> {'FLAG' if sat else 'ok'}   "
       f"A6 -> {'FLAG' if nobend else 'ok'}")

    P_("\n  A7  SUBUNIT USE (8 subunits)")
    for n in N_PROBE:
        P_(f"    n = {n}: effective subunits {np.mean([p['eff_init'] for p in by[n]]):.2f} at init -> "
           f"{np.mean([p['eff'] for p in by[n]]):.2f} at stop   (largest share "
           f"{np.mean([max(p['shares']) for p in by[n]]):.2f})")
    flags["A7 redundant subunits"] = float(np.mean([p["eff"] for p in probes])) < 4
    P_(f"    A7 -> {'FLAG' if flags['A7 redundant subunits'] else 'ok'}")

    # ---- PART B -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("PART B -- REPAIR SCREEN (seeds 40-44, n = 1500/2000/2500)"); P_(RULE)
    sc = pool.map(screen_job, [(a, s, n) for a in ARMS for s in SEEDS for n in N_SCREEN])
    pool.close()
    unit = {(a, s): float(np.mean([r["test"] for r in sc if r["arm"] == a and r["seed"] == s]))
            for a in ARMS for s in SEEDS}
    P_("    seed " + "".join(f"{ARM[a]:>26}" for a in ARMS))
    for s in SEEDS:
        P_(f"    {s:>4} " + "".join(f"{unit[(a, s)]:>26.3f}" for a in ARMS))
    P_("    mean " + "".join(f"{np.mean([unit[(a, s)] for s in SEEDS]):>26.3f}" for a in ARMS))
    P_("    params " + "".join(f"{[r['params'] for r in sc if r['arm'] == a][0]:>24,}  " for a in ARMS))
    P_("    capped " + "".join(f"{sum(r['steps'] >= SCREEN_CAP and r['train'] < 1.0 for r in sc if r['arm'] == a):>24}/15" for a in ARMS))
    screen = {}
    for a in ARMS[1:]:
        w, l, md, v = verdict([unit[(a, s)] - unit[("R0", s)] for s in SEEDS], a in ("R3", "R4"))
        screen[a] = dict(wins=w, losses=l, mean_diff=md, verdict=v)
        P_(f"  {a} {ARM[a]:<26} higher on {w}/5, lower on {l}/5, mean {md:+.3f}  -> {v}")

    # ---- A8 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("A8  COST PER TRAINING STEP (batch 128, single process)"); P_(RULE)
    res, t, agree = cost()
    for nm in res:
        r = res[nm]
        P_(f"    {nm:<8} {r['total']:.2f} ms/step   forward {r['forward']:.2f}  backward {r['backward']:.2f}  adam {r['adam']:.2f}")
    ratio = res["channel"]["total"] / res["mlpdeep"]["total"]
    flags["A8 cost"] = ratio > 3
    P_(f"    channel / deep MLP: {ratio:.1f}x  -> {'FLAG' if flags['A8 cost'] else 'ok'}")
    P_(f"    leaf contraction, same arithmetic two ways (max disagreement {agree:.1e}):")
    for k_ in t:
        P_(f"      {k_:<18} {t[k_]:.3f} ms")
    ok = agree < 1e-10
    P_(f"    matmul speed-up: forward {t['leaf fwd einsum'] / t['leaf fwd matmul']:.1f}x, gradient "
       f"{t['leaf grad einsum'] / t['leaf grad matmul']:.1f}x" + ("" if ok else "  (NOT REPORTED: forms disagree)"))

    P_("\n" + RULE); P_("FLAGS THAT FIRED"); P_(RULE)
    for k_, v in flags.items():
        P_(f"    {'FLAG' if v else ' ok '}  {k_}")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"probes": probes, "refs": refs, "screen": sc, "screen_verdicts": screen,
               "units": {f"{a}|{s}": v for (a, s), v in unit.items()}, "flags": flags,
               "cost": res, "leaf_timing": t, "leaf_agree": agree},
              open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/netprobe.json   runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


def a4_job(args):
    """POST-RUN RECHECK OF A4 (added after the run; the original instrument divided by |p| of groups
    initialised at exactly zero -- theta and the readout bias -- so their ratio exploded and every
    other group fell below 1/100 of it). Here: RMS gradient per element, absolute, and |g|/|p|
    averaged only from step 100 on, when no group is still at zero. Same training, same stopping."""
    seed, ntr = args
    Xtr, ytr, Xte, yte = lim.data(K, seed, ntr)
    m = lim.build("channel", np.random.default_rng(200 + seed))
    rng = np.random.default_rng(300 + seed)
    st, steps, t = {}, 0, 0
    rms = np.zeros(len(m.ps)); rel = np.zeros(len(m.ps)); nr = 0
    while steps < nb.MAXSTEPS:
        for _ in range(lim.CHECK):
            i = rng.integers(0, len(Xtr), lim.BS)
            z = m.forward(Xtr[i])
            gs = m.backward((1 / (1 + np.exp(-z)) - ytr[i]) / lim.BS)
            t += 1
            rms += [float(np.sqrt((g ** 2).mean())) for g in gs]
            if t > 100:
                rel += [np.linalg.norm(g) / np.linalg.norm(p) for p, g in zip(m.ps, gs)]
                nr += 1
            ch.adam(m.ps, gs, st, lim.LR)
        steps += lim.CHECK
        if lim.acc(m, Xtr, ytr) == 1.0:
            break
    return dict(seed=seed, ntr=ntr, steps=steps, groups=groups(m),
                rms=(rms / t).tolist(), rel=(rel / max(nr, 1)).tolist())


def a4_recheck():
    pool = mp.get_context("fork").Pool(4)
    rs = pool.map(a4_job, [(s, n) for n in N_PROBE for s in SEEDS]); pool.close()
    names = rs[0]["groups"]
    rms = np.mean([r["rms"] for r in rs], 0); rel = np.mean([r["rel"] for r in rs], 0)
    out = ["", RULE, "POST-RUN RECHECK OF A4 (added after the run; see a4_job's docstring)", RULE,
           "  The original A4 flag is an INSTRUMENT DEFECT: theta and the readout bias start at exactly",
           "  zero, so |g|/|p| for them was ~1e6-1e7 and every other group fell under 1/100 of it.",
           "    group        RMS grad/element   |g|/|p| from step 100"]
    for g, a, b in zip(names, rms, rel):
        out.append(f"    {g:<11}  {a:>16.2e}   {b:>20.2e}")
    top = rel.max()
    weak = [g for g, b in zip(names, rel) if b < top / 100]
    out.append(f"  corrected A4, same 1/100 criterion on |g|/|p| from step 100: below it -> {weak if weak else 'none'}")
    out.append(f"  soma-to-leaf attenuation of |g|/|p|, W L4 / leaf W: {rel[names.index('W L4')] / rel[names.index('leaf W')]:.1f}x")
    print("\n".join(out))
    open(OUT, "a").write("\n".join(out) + "\n")


if __name__ == "__main__":
    import sys
    if "--a4-recheck" in sys.argv:
        a4_recheck()
    else:
        main()
