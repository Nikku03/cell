"""ADRN POSITIVE CONTROL -- spec section 21 tests 1-3 against the section 22 baseline ladder.

WHY THIS FILE MUST RUN BEFORE ANY BIOLOGY TEST. Every mechanism in ADRN is about TIME: eligibility traces
exist to bridge delayed credit, fast/slow weights exist to separate timescales, adaptive halting exists to
spend more internal steps on harder cases, structural plasticity exists to specialise across sequential
tasks. The CellOS blocks -- knockout -> response profile, gene+line -> dependency -- are static, i.i.d. and
tabular. On those, most of that machinery is inert BY CONSTRUCTION.

So if ADRN were wired straight into the blocks and scored at the mean-pool baseline, that result would be
UNREADABLE: it could not distinguish "ADRN's advantages do not apply to static tabular biology" from "the
implementation is broken". This project has already shipped that exact class of unreadable null -- "evidence
strength not detected", reported four separate times, was stratifying on a column that was 0.000% populated.

This file is the positive control. It runs the tasks the spec itself nominates, where ADRN's mechanisms are
supposed to matter, against the baselines the spec itself nominates.

    Test 1  TEMPORAL XOR          two pulses at different times; the answer is their XOR.
                                  Purpose: dendritic coincidence detection and temporal memory.
    Test 2  DELAYED ASSOCIATION   a cue, then a long empty delay, then a choice.
                                  Purpose: working memory and eligibility traces across a gap.
    Test 3  CONTEXT-DEPENDENT MAP the same input requires a different answer under a context signal.
                                  Purpose: branch gating and apical context.

BASELINES, spec section 22. Every one is built to a MATCHED PARAMETER BUDGET and the count is printed, so
a baseline cannot lose merely by being smaller:
    1 MLP                       point neurons, no state
    2 GRU                       gated recurrence, the strong conventional baseline
    3 point-spiking (LIF)       spiking but NO dendrites -- isolates the dendritic claim
    4 small Transformer         attention over the time axis
    5 ADRN no dendrites         B = 1, the ablation that matters most
    6 ADRN no structural        pruning disabled
    7 ADRN no fast/slow         one weight timescale
    8 ADRN full

THE GATE, DECLARED BEFORE ANY NUMBER. ADRN is credited on a task only if it beats the BEST baseline by more
than the MDE. The unit of replication is the SEED (independent initialisation and independent data draw),
n = N_SEEDS, MDE = 3*sd/sqrt(n) on the paired per-seed gap. A win over the MLP alone is not a result: the
GRU is the baseline that matters, because it is the conventional way to solve exactly these tasks.

WHAT A NULL HERE WOULD MEAN, and it is the useful outcome either way. If ADRN cannot beat a GRU on temporal
XOR and delayed association -- tasks designed to favour it -- then the primitive does not carry its claimed
advantage at this scale, and no result it produces on static biology data would be interpretable. That is a
clean, early, cheap refutation, which is exactly what a positive control is for.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

N_SEEDS = int(os.environ.get("ADRN_SEEDS", "5"))
EPOCHS = int(os.environ.get("ADRN_EPOCHS", "60"))
N_TRAIN = int(os.environ.get("ADRN_NTRAIN", "1024"))
N_TEST = int(os.environ.get("ADRN_NTEST", "512"))
BUDGET = int(os.environ.get("ADRN_BUDGET", "12000"))
T_STEPS = int(os.environ.get("ADRN_T", "12"))
SMOKE = os.environ.get("ADRN_SMOKE", "0") == "1"
if SMOKE:
    N_SEEDS, EPOCHS, N_TRAIN, N_TEST, T_STEPS = 2, 15, 256, 256, 8


# ---------------------------------------------------------------- tasks (spec section 21)
def task_temporal_xor(n, T, rng):
    """Two pulses on separate channels at random times. Label = XOR of their presence."""
    X = np.zeros((n, T, 4), np.float32)
    y = np.zeros(n, np.int64)
    for i in range(n):
        a, b = rng.integers(0, 2), rng.integers(0, 2)
        ta, tb = rng.integers(0, T // 2), rng.integers(T // 2, T)
        if a:
            X[i, ta, 0] = 1.0
        if b:
            X[i, tb, 1] = 1.0
        X[i, :, 2] = 0.1 * rng.standard_normal(T)      # distractor noise
        X[i, :, 3] = 1.0                                # tonic drive
        y[i] = int(a ^ b)
    return X, y


def task_delayed_association(n, T, rng, delay=None):
    """Cue at t=0, long delay filled with DISTRACTORS, then a probe. Answer = which cue.

    The first version put a clean cue in an empty channel and every arm reached 1.0000 -- a task at
    ceiling discriminates nothing. The delay is now filled with distractor pulses on the SAME channels
    the cue uses, so the model must hold the FIRST event specifically rather than the last thing it saw.
    """
    delay = delay or max(2, T - 3)
    X = np.zeros((n, T, 4), np.float32)
    y = np.zeros(n, np.int64)
    for i in range(n):
        c = int(rng.integers(0, 2))
        X[i, 0, c] = 1.0
        for t in range(2, max(3, delay)):               # distractors on the cue channels themselves
            if rng.random() < 0.35:
                X[i, t, int(rng.integers(0, 2))] = 0.7
        X[i, min(delay + 1, T - 1), 2] = 1.0            # probe
        X[i, :, 3] = 0.1 * rng.standard_normal(T)
        y[i] = c
    return X, y


def task_context_mapping(n, T, rng):
    """Same stimulus, different required answer under a context bit. Tests gating, not memorisation."""
    X = np.zeros((n, T, 4), np.float32)
    y = np.zeros(n, np.int64)
    for i in range(n):
        s, ctx = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        # The first version held stimulus and context on constant channels for every step and every arm
        # reached 1.0000. Both are now BRIEF and SEPARATED in time, so the mapping has to be gated across
        # a gap rather than read off two constant wires.
        X[i, rng.integers(0, 2), 0] = float(s)
        X[i, T - 1 - rng.integers(0, 2), 1] = float(ctx)
        X[i, :, 2] = 0.1 * rng.standard_normal(T)
        X[i, :, 3] = 1.0
        y[i] = s if ctx == 0 else 1 - s                 # context INVERTS the mapping
    return X, y


TASKS = {"1 temporal XOR": task_temporal_xor,
         "2 delayed association": task_delayed_association,
         "3 context-dependent map": task_context_mapping}


def main():
    import torch
    import torch.nn as nn
    torch.set_num_threads(int(os.environ.get("ADRN_THREADS", "2")))
    from adrn import ADRN, count_params

    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 104)
    report("ADRN POSITIVE CONTROL -- spec sec 21 tests vs the sec 22 baseline ladder")
    report("=" * 104)
    if SMOKE:
        report("  *** SMOKE RUN -- plumbing check, NOT a result. Rerun with ADRN_SMOKE unset. ***")
    report(f"  GATE, declared before any number: a model is credited on a task only if it beats the BEST")
    report(f"  baseline by more than MDE = 3*sd/sqrt(n), n = {N_SEEDS} SEEDS (independent init AND")
    report(f"  independent data draw), sd of the PAIRED per-seed gap. Beating the MLP alone is not a result;")
    report(f"  the GRU is the baseline that matters. Parameter budget matched to ~{BUDGET:,} and printed.")

    # ------------------------------------------------------------ baselines
    class MLP(nn.Module):
        def __init__(self, i, o, h):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                     nn.Linear(h, o))

        def forward(self, x):
            return self.net(x.flatten(1)), {}

    class GRU(nn.Module):
        def __init__(self, i, o, h):
            super().__init__()
            self.rnn = nn.GRU(i, h, batch_first=True)
            self.out = nn.Linear(h, o)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.out(h[:, -1]), {}

    class LIF(nn.Module):
        """Point-neuron spiking baseline: identical spiking machinery, NO dendritic branches."""
        def __init__(self, i, o, h, T):
            super().__init__()
            self.fc = nn.Linear(i, h)
            self.rec = nn.Linear(h, h, bias=False)
            self.out = nn.Linear(h, o)
            self.h, self.T = h, T
            # FAIRNESS: the LIF baseline gets the SAME initial excitability as ADRN (theta 0.25, gain 2.0)
            # and the same homeostatic threshold regulation. ADRN's threshold was lowered from 1.0 because
            # at 1.0 it was silent; leaving the baseline at 1.0 would hand ADRN a win by handicap rather
            # than by mechanism. Both spiking arms start equally excitable and both self-regulate.
            self.register_buffer("theta", torch.full((h,), 0.25))
            self.gain = 2.0
            self.target_rate, self.eta_theta = 0.08, 0.01

        def forward(self, x):
            from adrn import spike
            B = x.shape[0]
            v = torch.zeros(B, self.h, device=x.device)
            s = torch.zeros(B, self.h, device=x.device)
            acc = torch.zeros(B, self.h, device=x.device)
            for t in range(x.shape[1]):
                v = v + (-(v) + self.gain * self.fc(x[:, t]) + self.rec(s)) / 5.0
                s = spike(v - self.theta)
                v = v * (1 - s.detach())
                acc = acc + s
            with torch.no_grad():
                r = acc.mean(0) / max(x.shape[1], 1)
                g = torch.where(r < 1e-4, torch.full_like(r, 10.0), torch.ones_like(r))
                self.theta.add_(self.eta_theta * g * (r - self.target_rate)).clamp_(0.02, 5.0)
            # instrumented so the silence canary can see it: an uninstrumented arm looks identical to a
            # silent one, and the canary must not fire on missing telemetry.
            return self.out(acc / x.shape[1]), {"spikes": float(acc.detach().sum()) / x.shape[0]}

    class Tformer(nn.Module):
        def __init__(self, i, o, h, heads=2):
            super().__init__()
            self.inp = nn.Linear(i, h)
            enc = nn.TransformerEncoderLayer(h, heads, dim_feedforward=2 * h, batch_first=True,
                                             dropout=0.0)
            self.tr = nn.TransformerEncoder(enc, 1)
            self.out = nn.Linear(h, o)

        def forward(self, x):
            return self.out(self.tr(self.inp(x)).mean(1)), {}

    def width_for(ctor, lo=4, hi=512):
        """Bisect hidden width so every arm lands on the same parameter budget."""
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                n = count_params(ctor(mid))
            except Exception:
                hi = mid - 1
                continue
            if n <= BUDGET:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return best

    def build(name, n_in, n_out, T):
        if name == "1 MLP":
            w = width_for(lambda h: MLP(n_in * T, n_out, h))
            return MLP(n_in * T, n_out, w)
        if name == "2 GRU":
            w = width_for(lambda h: GRU(n_in, n_out, h))
            return GRU(n_in, n_out, w)
        if name == "3 point-spiking LIF":
            w = width_for(lambda h: LIF(n_in, n_out, h, T))
            return LIF(n_in, n_out, w, T)
        if name == "4 small Transformer":
            w = width_for(lambda h: Tformer(n_in, n_out, max(2, (h // 2) * 2)))
            return Tformer(n_in, n_out, max(2, (w // 2) * 2))
        kw = dict(dendrites=True, fast_slow=True, structural=True)
        if name == "5 ADRN no dendrites":
            kw["dendrites"] = False
        if name == "6 ADRN no structural":
            kw["structural"] = False
        if name == "7 ADRN no fast/slow":
            kw["fast_slow"] = False
        w = width_for(lambda h: ADRN(n_in, n_out, n_neurons=h, n_branch=4, T=T, **kw), 4, 128)
        return ADRN(n_in, n_out, n_neurons=w, n_branch=4, T=T, **kw)

    ARMS = ["1 MLP", "2 GRU", "3 point-spiking LIF", "4 small Transformer",
            "5 ADRN no dendrites", "6 ADRN no structural", "7 ADRN no fast/slow", "8 ADRN full"]
    BASE = {"1 MLP", "2 GRU", "3 point-spiking LIF", "4 small Transformer"}

    results, params, energy, firing = {}, {}, {}, {}
    for tname, tfn in TASKS.items():
        report(f"\n  ── TEST {tname} ──")
        for arm in ARMS:
            accs, ens, spk = [], [], []
            for seed in range(N_SEEDS):
                torch.manual_seed(seed)
                rng = np.random.default_rng(1000 + seed)
                Xtr, ytr = tfn(N_TRAIN, T_STEPS, rng)
                Xte, yte = tfn(N_TEST, T_STEPS, rng)
                net = build(arm, Xtr.shape[2], 2, T_STEPS)
                params[arm] = count_params(net)
                opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-3)
                Xt = torch.from_numpy(Xtr)
                yt = torch.from_numpy(ytr)
                for _ in range(EPOCHS):
                    perm = torch.randperm(len(Xt))
                    for i0 in range(0, len(Xt), 128):
                        b = perm[i0:i0 + 128]
                        o = net(Xt[b])
                        logits = o[0] if isinstance(o, tuple) else o
                        loss = nn.functional.cross_entropy(logits, yt[b])
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in net.parameters() if p.requires_grad], 1.0)
                        opt.step()
                with torch.no_grad():
                    o = net(torch.from_numpy(Xte))
                    logits, info = (o if isinstance(o, tuple) else (o, {}))
                    accs.append(float((logits.argmax(-1).numpy() == yte).mean()))
                    ens.append(float(info.get("energy", float("nan"))))
                    if "spikes" in info:
                        spk.append(float(info["spikes"]))
            if arm.startswith(("3 ", "5 ", "6 ", "7 ", "8 ")):
                # SILENCE CANARY. Every spiking arm must actually spike. A silent network still returns a
                # number -- chance accuracy -- and every mechanism under test (eligibility, plasticity,
                # dendritic events, halting) is a no-op inside it. That is a vacuous null and it must be
                # raised, not reported. This project has already shipped one unreadable null of exactly
                # this shape ("evidence strength not detected", stratifying on an empty column).
                if not spk:
                    raise SystemExit(
                        f"TELEMETRY MISSING: arm '{arm}' reports no spike count, so the silence canary "
                        f"cannot check it. Instrument the arm or exclude it from the canary list.")
                sp = float(np.nanmean(spk))
                if sp < 1e-6:
                    raise SystemExit(
                        f"SILENCE CANARY FAILED: arm '{arm}' on '{tname}' emitted {sp:.3g} spikes per "
                        f"example. A spiking model that never spikes cannot test any claim about spiking; "
                        f"fix the threshold/gain scaling before trusting any number from this suite.")
            results.setdefault(tname, {})[arm] = accs
            energy.setdefault(tname, {})[arm] = float(np.nanmean(ens))
            firing.setdefault(tname, {})[arm] = float(np.nanmean(spk)) if spk else float("nan")
            e = energy[tname][arm]
            report(f"    {arm:26s} acc {np.mean(accs):.4f} ± {np.std(accs, ddof=1) if len(accs)>1 else 0:.4f}"
                   f"   params {params[arm]:>7,}"
                   f"{'' if not np.isfinite(e) else f'   energy {e:8.1f}'}"
                   f"{'' if not spk else f'   spk/ex {np.nanmean(spk):7.2f}'}")

    # ------------------------------------------------------------ gates
    gates, gaps = {}, {}
    for tname in TASKS:
        r = results[tname]
        best_base = max(BASE, key=lambda a: np.mean(r[a]))
        full = np.array(r["8 ADRN full"])
        bb = np.array(r[best_base])
        gap = float((full - bb).mean())
        sd = float(np.std(full - bb, ddof=1)) if len(full) > 1 else 0.0
        m = 3 * sd / np.sqrt(len(full))
        gates[f"{tname}: ADRN beats best baseline ({best_base})"] = bool(gap > m)
        gaps[tname] = {"best_baseline": best_base, "gap": gap, "mde": m,
                       "adrn": float(full.mean()), "baseline": float(bb.mean())}
        dend = float((full - np.array(r["5 ADRN no dendrites"])).mean())
        gates[f"{tname}: dendrites contribute"] = bool(dend > m)
        gaps[tname]["dendrite_gap"] = dend

    report("\n  PREDECLARED GATES")
    for k, v in gates.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k}")

    npass = sum(gates.values())
    verdict = (
        ("SMOKE RUN -- plumbing check, NOT a result. " if SMOKE else "") +
        f"ADRN POSITIVE CONTROL: {npass}/{len(gates)} gates. " +
        " ".join(f"On {t}, ADRN reaches {g['adrn']:.4f} against the best baseline "
                 f"({g['best_baseline']}) at {g['baseline']:.4f}, a gap of {g['gap']:+.4f} against an MDE "
                 f"of {g['mde']:.4f}; the dendrite ablation costs {g['dendrite_gap']:+.4f}."
                 for t, g in gaps.items()) +
        f" Every arm is built to a matched budget of ~{BUDGET:,} parameters and the realised counts are "
        f"reported per arm, so no baseline loses for being smaller. The unit of replication is the SEED "
        f"({N_SEEDS} independent initialisations and data draws), not an epoch and not a batch. "
        f"THIS IS A PREREQUISITE, NOT A BIOLOGY RESULT: if ADRN cannot beat a GRU on the temporal tasks "
        f"its own specification nominates, then any null it produces on static tabular biology data is "
        f"uninterpretable, because a broken implementation and an inapplicable mechanism would look "
        f"identical there.")
    report("\n" + "=" * 104)
    report(f"  VERDICT: {verdict}")

    R = {"model": "adrn-control-suite-v1", "smoke": bool(SMOKE), "seeds": N_SEEDS, "epochs": EPOCHS,
         "budget": BUDGET, "T": T_STEPS,
         "results": {t: {a: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                             "per_seed": v} for a, v in r.items()} for t, r in results.items()},
         "params": params, "energy": energy, "firing_rate": firing, "gates": gates, "gaps": gaps,
         "unit_of_replication": "seed (independent init AND independent data draw)",
         "limits": [
             "these are SYNTHETIC tasks chosen because the ADRN spec nominates them; success here is a "
             "prerequisite for interpreting ADRN elsewhere and is NOT evidence about biology",
             "the parameter budget is matched but the compute budget is not -- ADRN runs T internal steps "
             "per example and is therefore slower per parameter than the MLP or GRU. Energy is reported "
             "for ADRN arms only because the baselines have no spike/event count to report",
             "the outer loop trains developmental parameters by backprop through the unrolled dynamics "
             "with a surrogate spike gradient; the inner local rules run without gradient. A pure "
             "local-only arm is available via local_only=True and is NOT run here",
             "the MLP baseline receives the WHOLE sequence flattened, so it does not have to solve these "
             "tasks temporally at all -- it is the hardest baseline here for that reason and beating it "
             "is a stronger claim than beating the GRU",
             "T is fixed for the baselines and adaptive for ADRN via the halting rule, so ADRN may use "
             "fewer effective steps; the realised step count is in the per-episode info dict"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "adrn_control_suite.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_control_suite.json'}")


if __name__ == "__main__":
    main()
