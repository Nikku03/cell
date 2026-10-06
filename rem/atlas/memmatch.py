"""Skill-matched rerun of memcommit.py: does the channel-node memory still commit like the mouse?

THE DEFECT IT FIXES (recorded in RESULTS_memcommit.txt). memcommit's networks were 0.86-1.00 accurate
on plain trials when first checked; the mice are 0.824. A memory that rarely errs is not comparable
to one that errs 18% of the time.

THE FIX, fixed now: the SAME networks (memcommit's training, same seeds and random streams) are
probed with their internal noise raised until plain-trial accuracy (sample vs none, equal halves)
equals the mice's 0.824. Noise sd found by bisection on log sd in [0.001, 20], 14 steps, 2,000 trials
with common random numbers. All probing then happens at that noise.

GATES, PREDECLARED
M0 SAME NETWORKS, BLOCKING. Retrained networks must reproduce memcommit.json's steps and plain-trial
   accuracy exactly, run for run.
M1 HARNESS, BLOCKING. memcommit's latch harness must still pass (committing COMMITS, pure does not).
M2 CALIBRATION. A run counts only if its calibrated accuracy is within 0.02 of 0.824.
C1/C2 unchanged from memcommit: COMMITS if the commitment index > 0 on >= 9/10 calibrated seeds;
   CLOSER TO THE MOUSE if lower 14-condition MAE on >= 9/10 paired seeds.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import json
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_memmatch.txt")
ART = R("outputs", "memmatch.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("memcommit", os.path.join(HERE, "memcommit.py"))
mc = importlib.util.module_from_spec(_s); _s.loader.exec_module(mc)
TARGET = 0.824


def trained(node, seed, regime):
    """memcommit.train's loop verbatim, returning the network."""
    rng = np.random.default_rng(1000 * (1 + mc.MODELS.index(node)) + seed + (0 if regime == "A" else 50000))
    m = mc.MemNet(node, np.random.default_rng(seed))
    bar = 0.95 if regime == "A" else 0.82
    st, steps, acc = {}, 0, 0.0
    while steps < mc.CAP:
        X, y = mc.batch(rng, regime)
        z = m.forward(X, rng.normal(0, mc.NOISE, (mc.BS, mc.T, mc.N)))
        if not np.all(np.isfinite(z)):
            break
        mc.adam_clip(m.ps, m.backward((1 / (1 + np.exp(-z)) - y) / mc.BS), st, mc.LR)
        steps += 1
        if steps % mc.EVERY == 0:
            acc = mc.basic_acc(m, np.random.default_rng(seed + 777))
            if acc >= bar:
                break
    return m, steps, acc


def acc_at(m, sd, eps, X, ys):
    z = m.forward(X, sd * eps)
    return float(((z > 0) == (ys > 0)).mean())


def calibrate(m, seed):
    ys = np.repeat([0, 1], 1000)
    X = np.stack([mc.trial_inputs(bool(y)) for y in ys])
    eps = np.random.default_rng(seed + 4242).normal(0, 1, (2000, mc.T, mc.N))
    lo, hi = np.log(0.001), np.log(20.0)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if acc_at(m, np.exp(mid), eps, X, ys) > TARGET:
            lo = mid
        else:
            hi = mid
    sd = float(np.exp(0.5 * (lo + hi)))
    return sd, acc_at(m, sd, eps, X, ys)


def job(args):
    node, seed, regime = args
    m, steps, acc = trained(node, seed, regime)
    sd, cacc = calibrate(m, seed)
    mc.NOISE = sd                       # probe at the calibrated noise (fork-local global)
    pr = mc.probe(m, np.random.default_rng(seed + 999), fine=True)
    return dict(node=node, seed=seed, regime=regime, steps=steps, basic_acc=acc, sd=sd, cal_acc=cacc,
                ok=abs(cacc - TARGET) <= 0.02, probe=pr)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    tgt = mc.mouse_target()
    keys14 = sorted(tgt, key=str)
    prev = {(r["node"], r["seed"], r["regime"]): r for r in json.load(open(R("outputs", "memcommit.json")))["runs"]}
    P_(RULE); P_("SKILL-MATCHED: EVERY MEMORY PROBED AT THE MICE'S PLAIN-TRIAL ACCURACY (0.824)"); P_(RULE)
    h = {nm: mc.verdict([mc.index_of(mc.probe(mc.Latch(c), np.random.default_rng(s))) for s in range(10)], 1.0)[2]
         for nm, c in (("committing", True), ("pure", False))}
    m1 = h["committing"] == "COMMITS" and h["pure"] != "COMMITS"
    P_(f"  M1 harness: committing latch {h['committing']}, pure latch {h['pure']} -> {'PASS' if m1 else 'FAIL'}")
    if not m1:
        open(OUT, "w").write("\n".join(out) + "\n"); return
    mc.MIX = mc.mouse_mix()
    pool = mp.get_context("fork").Pool(4)
    runs = pool.map(job, [(nd, s, rg) for rg in ("A", "B") for nd in mc.MODELS for s in mc.SEEDS])
    pool.close()
    same = all(r["steps"] == prev[(r["node"], r["seed"], r["regime"])]["steps"]
               and r["basic_acc"] == prev[(r["node"], r["seed"], r["regime"])]["basic_acc"] for r in runs)
    P_(f"  M0 same networks as memcommit (steps and accuracy identical, 60/60): {'PASS' if same else 'FAIL'}")
    if not same:
        open(OUT, "w").write("\n".join(out) + "\n"); return
    res = {}
    for rg in ("A", "B"):
        P_("\n" + RULE)
        P_(f"REGIME {rg}: " + ("ZERO-SHOT" if rg == "A" else "MOUSE TRIAL MIX") + " -- probed at calibrated noise")
        P_(RULE)
        P_(f"    {'':<24}" + "".join(f"{('none' if dt is None else '-%.1f %s' % (dt, it)):>12}" for dt, it in mc.CONDS))
        P_(f"    {'MOUSE':<24}" + "".join(f"{tgt[(False, dt, it)]:>12.3f}" for dt, it in mc.CONDS))
        for nd in mc.MODELS:
            rs = [r for r in runs if r["regime"] == rg and r["node"] == nd and r["ok"]]
            P_(f"    {nd + ' (' + str(len(rs)) + '/10 calibrated)':<24}" + "".join(
                f"{np.mean([r['probe'][(False, dt, it)] for r in rs]):>12.3f}" for dt, it in mc.CONDS))
        P_("\n  C1 COMMITMENT INDEX (mouse +0.256); noise sd needed to reach 0.824")
        for nd in mc.MODELS:
            rs = [r for r in runs if r["regime"] == rg and r["node"] == nd and r["ok"]]
            if len(rs) < 9:
                res[(rg, nd)] = dict(verdict="NOT ENOUGH CALIBRATED SEEDS", n=len(rs))
                P_(f"    {nd:<8} calibrated {len(rs)}/10 -> NOT ENOUGH CALIBRATED SEEDS"); continue
            vals = [mc.index_of(r["probe"]) for r in rs]
            fooled = float(np.mean([r["probe"][(False, 1.6, "Full")] - r["probe"][(False, None, None)] for r in rs]))
            w, l, v = mc.verdict(vals, fooled)
            mae = float(np.mean([np.mean([abs(r["probe"][k] - tgt[k]) for k in keys14]) for r in rs]))
            res[(rg, nd)] = dict(verdict=v, wins=w, n=len(rs), index=float(np.mean(vals)), mae=mae,
                                 sd=float(np.median([r["sd"] for r in rs])))
            P_(f"    {nd:<8} sd {res[(rg, nd)]['sd']:.2f}  index {np.mean(vals):+.3f} (positive {w}/{len(rs)})  "
               f"early fooling {fooled:+.3f}  MAE to mouse {mae:.3f}  -> {v}")
        P_("\n  C2 CLOSER TO THE MOUSE? (paired per seed, >= 9/10)")
        for a_, b_ in (("channel", "bump"), ("channel", "tanh"), ("bump", "tanh")):
            A_ = {r["seed"]: r for r in runs if r["regime"] == rg and r["node"] == a_ and r["ok"]}
            B_ = {r["seed"]: r for r in runs if r["regime"] == rg and r["node"] == b_ and r["ok"]}
            sd_ = sorted(set(A_) & set(B_))
            if len(sd_) < 9:
                P_(f"    {a_} vs {b_}: fewer than 9 shared seeds"); continue
            ma = lambda r: float(np.mean([abs(r["probe"][k] - tgt[k]) for k in keys14]))
            d = [ma(A_[s]) - ma(B_[s]) for s in sd_]
            w = sum(x < 0 for x in d); l = sum(x > 0 for x in d)
            P_(f"    {a_} vs {b_}: {a_} closer on {w}/{len(sd_)}, mean MAE diff {np.mean(d):+.3f} -> "
               + (f"{a_.upper()} CLOSER" if w >= 9 else f"{b_.upper()} CLOSER" if l >= 9 else "NO CLEAR DIFFERENCE"))
        P_("\n  fine fooling curve, Full pulse (seconds before go):")
        grid = sorted(runs[0]["probe"]["grid"], reverse=True)
        P_("    " + f"{'':<10}" + "".join(f"{t:>6.1f}" for t in grid))
        for nd in mc.MODELS:
            rs = [r for r in runs if r["regime"] == rg and r["node"] == nd and r["ok"]]
            P_(f"    {nd:<10}" + "".join(f"{np.mean([r['probe']['grid'][t] for r in rs]):>+6.2f}" for t in grid))
    for r in runs:
        r["probe"] = {str(k): v for k, v in r["probe"].items()}
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"target": TARGET, "results": {f"{a}|{b}": v for (a, b), v in res.items()}, "runs": runs},
              open(ART, "w"), indent=1, default=str)
    P_(f"\n  artifact: outputs/memmatch.json   runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
