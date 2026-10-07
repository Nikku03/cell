"""Which biological number hurt the memory? The three constants of biomemory.py, one at a time.

FROM biomemory.py. A synaptic (Hebbian, decaying) memory solved "facts shown once" -- EXACT on
10/10 seeds when its constants were learnt, 0.77 at 8 hops with all three biological constants:
    lambda 0.863 per step  (Blue Brain facilitating synapses, tau_F 680 ms)
    U      0.092           (Blue Brain release probability of those synapses)
    alpha  0.135           (ALM intrinsic timescale 0.692 s, DANDI 000060)
Three constants changed at once; this run fixes ONE at its biological value and learns the rest.

ARMS (the values are read from outputs/biomemory.json, not retyped):
    LAM    lambda biological; U learnt; no leak (alpha = 1)
    GAIN   U biological; lambda learnt; no leak
    LEAK   alpha biological; lambda and U learnt
References, from biomemory.json (same code, seeds and questions): GENERIC (all learnt, no leak)
and BIO (all three biological).
Versions P1 (6 items, facts every step) and H2 (facts shown once) -- where the biological arm lost.
H1 is not run: the generic arm failed it too (2/10), so it cannot attribute a loss to a constant.

GATES, PREDECLARED
Q0 BPTT gradient check for each arm (only the learnt constants receive gradients), both formats;
   the broken copy (trace carry-over gradient dropped) MUST fail.
C1 Per arm and version, versus GENERIC on the long score (depths > 3, P1 without 64), per seed:
   lower on >= 9/10 -> THIS CONSTANT HURTS; higher on >= 9/10 -> HELPS; else NO CLEAR EFFECT.
C2 Learnt (1-3 hops >= 0.90) and exact (P1: >= 0.99 at 16 and 64; H2: at 8), >= 9/10 seeds.
C3 Descriptive: does the worst single constant account for BIO's loss?
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
OUT = os.path.join(HERE, "RESULTS_bioablate.txt")
ART = R("outputs", "bioablate.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("biomemory", os.path.join(HERE, "biomemory.py"))
bm = importlib.util.module_from_spec(_s); _s.loader.exec_module(bm)
ce = bm.ce
ARMS = {"LAM": {"lam"}, "GAIN": {"U"}, "LEAK": {"alpha"}}
VERSIONS = ("P1", "H2")


class AblNet(bm.MemNet):
    def __init__(self, kind, rng, dx, m, fixed, u=ce.U):
        super().__init__(kind, rng, dx, m, "generic", u=u)
        self.fixed = fixed
        self.ps = self.F + [self.Wo, self.bo, self.Ek, self.Ev] + \
            ([] if "lam" in fixed else [self.lam_l]) + ([] if "U" in fixed else [self.U_l])

    def lamU(self):
        lam = bm.BIO["lambda"] if "lam" in self.fixed else float(bm.sig(self.lam_l[0]))
        U = bm.BIO["U"] if "U" in self.fixed else float(np.log1p(np.exp(self.U_l[0])))
        return lam, U

    def alpha(self):
        return bm.BIO["alpha"] if "alpha" in self.fixed else 1.0

    def backward(self, dlogs):
        g = super().backward(dlogs)          # generic layout: ... + [g_lam, g_U]
        core, glam, gU = g[:-2], g[-2], g[-1]
        return core + ([] if "lam" in self.fixed else [glam]) + ([] if "U" in self.fixed else [gU])


class BrokenAbl(AblNet):
    broken = True


def gradcheck(fixed, fmt, broken=False, seed=5):
    rng = np.random.default_rng(seed)
    m = 4
    net = (BrokenAbl if broken else AblNet)("combined", rng, ce.dx_of(m, fmt), m, fixed, u=3)
    for p in net.a + net.mu + net.s:
        p[...] = rng.normal(0.4, 0.3, p.shape)
    X, Y = ce.batch(rng, 5, 3, m, fmt)

    def L():
        net.run(X, Y, fmt); return ce.step_losses(net, Y, True)[1]

    net.run(X, Y, fmt); dl, _ = ce.step_losses(net, Y, True); gs = net.backward(dl)
    assert len(gs) == len(net.ps)
    ratio = 0.0
    for p, g in zip(net.ps, gs):
        for _ in range(4):
            idx = tuple(rng.integers(0, s_) for s_ in p.shape)
            o = p[idx]
            p[idx] = o + 1e-6; lp = L()
            p[idx] = o - 1e-6; lm = L()
            p[idx] = o
            num = (lp - lm) / 2e-6
            ratio = max(ratio, abs(num - g[idx]) / (1e-5 * max(abs(num), abs(g[idx])) + 1e-9))
    return ratio


def job(args):
    ver, arm, seed = args
    V = bm.VERS[ver]; m, fmt = V["m"], V["fmt"]
    net = AblNet("combined", np.random.default_rng(seed), ce.dx_of(m, fmt), m, ARMS[arm])
    rng = np.random.default_rng(700 + seed)
    st = {}
    for _ in range(ce.STEPS):
        k = int(rng.choice(ce.TRAIN_K))
        X, Y = ce.batch(rng, ce.BS, k, m, fmt)
        lg = net.run(X, Y, fmt, ce.NOISE, rng)
        if not np.all(np.isfinite(lg)):
            break
        dl, _ = ce.step_losses(net, Y, True)
        ce.adam_clip(net.ps, net.backward(dl), st, ce.LR)

    def pred(X):
        lg = net.run(X, None, fmt)
        return np.where(np.all(np.isfinite(lg), 1), lg.argmax(1), -1)

    acc = {k_: ce.evaluate(pred, k_, seed, m, fmt) for k_ in V["test"]}
    lam, U = net.lamU()
    return dict(version=ver, arm=arm, seed=seed, acc=acc, lam=lam, U=U, alpha=net.alpha(),
                ind=float(np.mean([acc[k_] for k_ in ce.TRAIN_K])))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    prev = json.load(open(R("outputs", "biomemory.json")))
    bm.BIO.update(prev["bio"])
    P_(RULE); P_("WHICH BIOLOGICAL NUMBER HURT? ONE AT A TIME"); P_(RULE)
    P_(f"  biological values (from outputs/biomemory.json): lambda {bm.BIO['lambda']:.4f}, U {bm.BIO['U']:.3f}, alpha {bm.BIO['alpha']:.4f}")
    ok = True
    for arm, fx in ARMS.items():
        for fmt in ("every", "once"):
            g, bg = gradcheck(fx, fmt), gradcheck(fx, fmt, broken=True)
            ok &= g <= 1.0 and bg > 1.0
            P_(f"  Q0 {arm:<5} {fmt:<5} gradient/tol {g:.3f}   broken {bg:.2e} ({'caught' if bg > 1 else 'NOT CAUGHT'})")
    if not ok:
        P_("  BLOCKED."); open(OUT, "w").write("\n".join(out) + "\n"); return
    jobs = [(v, a, s) for v in VERSIONS for a in ARMS for s in bm.VERS[v]["seeds"]]
    jobs.sort(key=lambda j: j[0] != "H2")
    pool = mp.get_context("fork").Pool(4)
    runs = list(pool.imap_unordered(job, jobs, chunksize=1))
    pool.close()
    res = {}
    for ver in VERSIONS:
        V = bm.VERS[ver]
        shared = [k_ for k_ in V["test"] if k_ > 3 and k_ != 64]
        P_("\n" + RULE); P_({"P1": "P1 -- 6 items, facts every step", "H2": "H2 -- facts shown ONCE"}[ver] + "; trained 1-3 hops"); P_(RULE)
        P_("    " + f"{'':<34}" + "".join(f"{'k=' + str(k_):>7}" for k_ in V["test"]) + "  learnt exact")
        units = {}
        rows = [("GENERIC (all learnt)", [r for r in prev["runs"] if r["version"] == ver and r["mode"] == "generic"], True)]
        rows += [(f"{a}: {'decay' if a == 'LAM' else 'write gain' if a == 'GAIN' else 'activity leak'} biological",
                  [r for r in runs if r["version"] == ver and r["arm"] == a], False) for a in ARMS]
        rows += [("BIO (all three biological)", [r for r in prev["runs"] if r["version"] == ver and r["mode"] == "bio"], True)]
        keys = ["GENERIC"] + list(ARMS) + ["BIO"]
        for key, (nm, rs, from_prev) in zip(keys, rows):
            rs = sorted(rs, key=lambda r: r["seed"])
            get = (lambda r, k_: r["acc"][str(k_)]) if from_prev else (lambda r, k_: r["acc"][k_])
            lt = sum(np.mean([get(r, k_) for k_ in ce.TRAIN_K]) >= 0.9 for r in rs)
            ex = sum(all(get(r, e) >= 0.99 for e in V["exact"]) for r in rs)
            P_(f"    {nm:<34}" + "".join(f"{np.mean([get(r, k_) for r in rs]):>7.3f}" for k_ in V["test"])
               + f"  {lt:>2}/10 {ex:>2}/10" + ("  EXACT" if ex >= 9 else ""))
            if key in ARMS:
                P_(f"      learnt: lambda {np.median([r['lam'] for r in rs]):.3f}  U {np.median([r['U'] for r in rs]):.3f}  (alpha {rs[0]['alpha']:.3f})")
            units[key] = {r["seed"]: float(np.mean([get(r, k_) for k_ in shared])) for r in rs}
            res[(ver, key, "learnt")] = int(lt); res[(ver, key, "exact")] = int(ex)
        P_("\n  C1 each single biological constant vs GENERIC (long score, per seed):")
        for a in ARMS:
            d = [units[a][s] - units["GENERIC"][s] for s in V["seeds"]]
            w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
            v = "THIS CONSTANT HURTS" if l >= 9 else "HELPS" if w >= 9 else "NO CLEAR EFFECT"
            res[(ver, a, "c1")] = dict(higher=w, lower=l, mean=float(np.mean(d)), verdict=v)
            P_(f"    {a:<5} higher on {w}/10, lower on {l}/10, mean {np.mean(d):+.3f} -> {v}")
        bio_loss = float(np.mean([units["BIO"][s] - units["GENERIC"][s] for s in V["seeds"]]))
        worst = min(ARMS, key=lambda a: np.mean([units[a][s] - units["GENERIC"][s] for s in V["seeds"]]))
        P_(f"  C3 BIO's loss vs GENERIC {bio_loss:+.3f}; worst single constant {worst} "
           f"{np.mean([units[worst][s] - units['GENERIC'][s] for s in V['seeds']]):+.3f}")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"bio": bm.BIO, "results": {"|".join(k): v for k, v in res.items()},
               "runs": [{**r, "acc": {str(k_): float(v) for k_, v in r["acc"].items()}} for r in runs]},
              open(ART, "w"), indent=1, default=float)
    P_(f"\n  artifact: outputs/bioablate.json   runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
