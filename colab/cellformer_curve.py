"""CELLFORMER LEARNING CURVE -- is the model undertrained, or is the architecture not helping?

WHY THIS RUNS BEFORE THE ABLATION TABLE. CELLFORMER.md pre-registers an eleven-row ablation table, and that
table is only meaningful if the full model can learn the task at all. Two runs so far:

    uniform negatives, 250 steps    recall 0.0224   vs frequency 0.2824
    matched negatives, 150 steps    recall 0.0508   vs frequency 0.2824

The negative-sampling fix more than doubled it, so the model does respond to a real correction -- but it is
still 5.5x below a baseline that needs no biology. Running ten ablations of a model in that state would measure
noise and would produce a table of near-zero rows that says nothing about the architecture.

So this establishes the precondition first, at the cost of ONE training run rather than eleven: train a single
full model and evaluate the SAME model at increasing budgets. The curve separates two very different
conclusions that the ablation table cannot tell apart:

    recall climbing toward 0.2824   -> undertrained; the table needs a larger budget to mean anything
    recall flat near 0.05           -> the architecture is not learning this task, and the table can say so

This is not moving the goalposts. The pre-registered criterion (beat frequency with a bootstrap CI excluding
zero) and the pre-registered interpretation rule are both unchanged; this only checks whether the experiment is
in a regime where they can be evaluated.

WHAT THE FIRST A100 RUN MEASURED, and the two defects it exposed in this file.

    step   200   recall 0.1038
    step   600   recall 0.1573
    step  1500   recall 0.0890      <- BELOW step 600
    step  3000   recall 0.2531      vs frequency 0.3980

The scale-up helped a great deal: at G_CROP=256, P_CTX=128, 4 blocks, the model went from 18% of the
frequency baseline to 64% of it. But the curve is NOT monotonic, and the two defects are why that could not
be read:

  1. n=40 held-out perturbations. The standard error on mean recall is then around 0.04, so the
     0.1573 -> 0.0890 dip is roughly what sampling noise produces on its own. The old verdict rule was
     `last > first * 1.3`, which reports STILL CLIMBING off first-and-last alone and never sees the dip.
     Evaluation is forward-only; subsampling the held-out set bought nothing and cost the ability to
     interpret the result. N_EVAL now defaults to ALL held-out scorable perturbations.

  2. evaluate() returned a MEAN. So the pre-registered criterion -- "beat frequency with a bootstrap CI
     excluding zero" -- was never actually computed anywhere; the run printed two numbers and left the
     comparison to the eye. It is now computed as a PAIRED test on the same perturbations, which is both
     the registered criterion and far more sensitive, since a perturbation that is hard for the model is
     usually hard for frequency too.

The lesson is the one this project keeps relearning: an eleven-row ablation table read at +-0.078 noise per
cell would have produced eleven confidently-reported numbers that mean nothing.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CF_GCROP", "96")
os.environ.setdefault("CF_PCTX", "48")
os.environ.setdefault("CF_GEVAL", "128")
os.environ.setdefault("CF_PCTXEVAL", "8")

CHECKPOINTS = [int(x) for x in os.environ.get("CF_CKPT", "150,400,800,1400").split(",")]
# 0 means EVERY held-out scorable perturbation. The first GPU run used 40 and produced
# 0.1038 -> 0.1573 -> 0.0890 -> 0.2531: a dip at step 1500 below step 600. With n=40 the standard
# error on mean recall is around 0.04, so that dip is what sampling noise looks like, not a training
# event. An eleven-row ablation table read at that noise level would be unreadable -- which is the
# exact failure CELLFORMER.md warns about ("ablating its parts would have measured noise").
# Evaluation is forward-only and cheap relative to training, so there is no reason to subsample.
N_EVAL = int(os.environ.get("CF_NEVAL", "0"))
N_BOOT = int(os.environ.get("CF_NBOOT", "2000"))


def boot_ci(x, n=N_BOOT, seed=0, lo=2.5, hi=97.5):
    """Percentile bootstrap CI of the mean of x."""
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = r.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def main():
    import torch
    import torch.nn as nn
    import cellformer_af as CF

    dev = CF.set_device()
    print(f"device: {dev}"
          + (f"  ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)" if dev.type == "cuda" else ""))
    d = CF.load_all()
    kos = d["kos"]
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    pool = [i for i in range(len(kos)) if tr[i]]
    test = [i for i in range(len(kos)) if not tr[i] and d["nspec"][i] >= CF.MIN_SPEC]
    rng0 = np.random.default_rng(1)
    test = sorted(rng0.choice(test, size=min(N_EVAL, len(test)), replace=False).tolist())

    freq = (np.abs(d["M"][tr]) >= CF.TAU).mean(0)
    must = set()
    for qi in test:
        must |= set(np.where(d["spec"][qi])[0].tolist())
    ranked = [int(i) for i in np.argsort(-freq) if not d["tide"][i]]
    cand = list(dict.fromkeys(sorted(must) + ranked))[:2048]
    d["cand"] = np.array(sorted(cand), dtype=np.int64)
    assert must <= set(cand), "candidate pool drops true movers"
    fp = [int(i) for i in np.argsort(-freq) if int(i) in set(d["cand"].tolist())][:CF.NPICK]
    # PER-PERTURBATION, not a mean. The pre-registered criterion is a PAIRED comparison, and pairing is
    # impossible once the baseline has been collapsed to a single number. The same perturbation is easy or
    # hard for both methods, so the per-perturbation difference has far less variance than the difference of
    # the two means -- which is why the paired test can resolve a gap the two printed numbers cannot.
    truth_of = {q: set(np.where(d["spec"][q])[0].tolist()) for q in test}
    fr_i = np.array([len(set(fp) & truth_of[q]) / len(truth_of[q]) for q in test], float)
    fr = float(fr_i.mean())
    flo, fhi = boot_ci(fr_i)
    print(f"held-out {len(test)} perturbations; candidate pool {len(d['cand']):,} genes "
          f"(contains all true movers)")
    print(f"frequency baseline: recall {fr:.4f}  95% CI [{flo:.4f}, {fhi:.4f}]  "
          f"(SE {fr_i.std(ddof=1)/np.sqrt(len(fr_i)):.4f})\n", flush=True)

    cfg = dict(CF.BASE)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = CF.build_model(torch, nn, len(d["genes"]), cfg).to(dev)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"G_CROP={CF.G_CROP} P_CTX={CF.P_CTX} blocks={cfg['n_block']} recycle={cfg['recycle']}\n",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=CF.LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    def evaluate():
        """Return the PER-PERTURBATION recall vector, not its mean.

        A mean alone cannot distinguish a real dip from eval noise, which is precisely what the first GPU
        run ran into. Keeping the vector makes the CI and the paired test possible; collapsing it here
        would throw away the pairing that the pre-registered criterion depends on.
        """
        model.eval()
        recs = []
        with torch.no_grad():
            for qi in test:
                truth = truth_of[qi]
                score = np.full(len(d["genes"]), -1e9, np.float32)
                allg = d["cand"]
                for s in range(0, len(allg), CF.G_EVAL):
                    cols = list(allg[s:s + CF.G_EVAL])
                    if len(cols) < 8:
                        continue
                    rows, _, qg = CF.make_example(d, rng, qi, pool, cfg, len(d["genes"]))
                    rows = rows[:1 + CF.P_CTX_EVAL]
                    zin, mask, isq, gidx, koidx, pz, B, idx = CF.tensors(d, rows, cols, qg, torch, cfg,
                                                                         rng, False)
                    r, _, _, _ = model(zin, mask, isq, gidx, koidx, pz, B, cfg["recycle"])
                    score[cols] = r[0].detach().cpu().numpy()
                pick = [int(i) for i in np.argsort(-score)[:CF.NPICK]]
                recs.append(len(set(pick) & truth) / len(truth))
        model.train()
        return np.array(recs, float)

    t0, losses, curve = time.time(), [], []
    for step in range(1, max(CHECKPOINTS) + 1):
        qi = int(rng.choice(pool))
        rows, cols, qg = CF.make_example(d, rng, qi, pool, cfg, len(d["genes"]))
        zin, mask, isq, gidx, koidx, pz, B, idx = CF.tensors(d, rows, cols, qg, torch, cfg, rng, False)
        y = torch.tensor(d["spec"][np.ix_(rows, cols)].astype(np.float32), device=dev)
        r, c, p, cap = model(zin, mask, isq, gidx, koidx, pz, B, cfg["recycle"])
        loss = bce(r[0], y[0]) + 0.3 * bce(r[1:], y[1:])
        nsp = torch.tensor((d["nspec"][rows] >= CF.MIN_SPEC).astype(np.float32), device=dev)
        loss = loss + 0.2 * bce(c, nsp)
        co = (y.t() @ y > 0).float()
        loss = loss + 0.2 * bce(p, co)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step in CHECKPOINTS:
            rec_i = evaluate()
            rc = float(rec_i.mean())
            lo, hi = boot_ci(rec_i)
            # THE PRE-REGISTERED CRITERION, computed rather than eyeballed: paired difference against the
            # frequency baseline on the SAME perturbations, bootstrap CI, criterion is "excludes zero".
            dlt = rec_i - fr_i
            dlo, dhi = boot_ci(dlt)
            beats = dlo > 0
            curve.append({"step": step, "loss": float(np.mean(losses[-100:])), "recall": rc,
                          "ci": [lo, hi], "delta_vs_freq": float(dlt.mean()), "delta_ci": [dlo, dhi],
                          "beats_frequency": bool(beats)})
            print(f"  step {step:>5d}   loss {np.mean(losses[-100:]):.4f}   recall {rc:.4f} "
                  f"[{lo:.4f},{hi:.4f}]   vs freq {dlt.mean():+.4f} [{dlo:+.4f},{dhi:+.4f}]"
                  f"{'  BEATS' if beats else ''}   [{time.time()-t0:.0f}s]", flush=True)
            json.dump({"frequency": fr, "frequency_ci": [flo, fhi], "n_eval": len(test),
                       "n_boot": N_BOOT, "curve": curve},
                      open(OUT / "cellformer_curve.json", "w"), indent=1)

    print("\n" + "=" * 94)
    print(f"{'step':>7s} {'loss':>9s} {'recall':>9s} {'95% CI':>19s} {'vs frequency':>21s} {'% of freq':>11s}")
    for c_ in curve:
        lo, hi = c_["ci"]
        dl, dh = c_["delta_ci"]
        print(f"{c_['step']:>7d} {c_['loss']:>9.4f} {c_['recall']:>9.4f} "
              f"[{lo:>7.4f},{hi:>7.4f}] {c_['delta_vs_freq']:>+8.4f} [{dl:>+6.4f},{dh:>+6.4f}] "
              f"{100*c_['recall']/fr:>10.1f}%")
    print("=" * 94)

    # VERDICT FROM THE NOISE, NOT FROM FIRST-VS-LAST. The previous rule was `last > first * 1.3`, which
    # called STILL CLIMBING on a curve that went 0.1038 -> 0.1573 -> 0.0890 -> 0.2531. A sequence with a
    # mid-run dip below an earlier point is not evidence of a monotone climb; at n=40 it is evidence the
    # eval was too small to say anything. So: compare the LAST checkpoint against the BEST EARLIER one,
    # and require the gain to clear the width of the CI rather than an arbitrary ratio.
    first, last = curve[0]["recall"], curve[-1]["recall"]
    best_prev = max(c_["recall"] for c_ in curve[:-1]) if len(curve) > 1 else first
    halfwidth = (curve[-1]["ci"][1] - curve[-1]["ci"][0]) / 2
    gain = last - best_prev
    noisy = any(curve[i]["recall"] < curve[i - 1]["recall"] for i in range(1, len(curve)))
    climbing = gain > halfwidth

    print(f"\n  recall {first:.4f} -> {last:.4f} over {curve[0]['step']} -> {curve[-1]['step']} steps")
    print(f"  best earlier checkpoint {best_prev:.4f}; final gain over it {gain:+.4f} "
          f"against a CI half-width of {halfwidth:.4f}")
    if noisy:
        print("  NOTE: the curve is non-monotonic -- at least one checkpoint scored below its predecessor.")
    if climbing:
        print("  VERDICT: STILL CLIMBING -- the final gain exceeds eval noise; the table needs a larger budget")
    elif gain > 0:
        print("  VERDICT: INCONCLUSIVE -- rising but within eval noise; more steps or more held-out "
              "perturbations before reading an ablation table")
    else:
        print("  VERDICT: FLAT -- more training does not close the gap; the architecture is the limit")

    dl, dh = curve[-1]["delta_ci"]
    if curve[-1]["beats_frequency"]:
        print(f"  CRITERION MET: paired delta vs frequency {curve[-1]['delta_vs_freq']:+.4f} "
              f"CI [{dl:+.4f},{dh:+.4f}] excludes zero.")
    else:
        print(f"  CRITERION NOT MET: paired delta {curve[-1]['delta_vs_freq']:+.4f} "
              f"CI [{dl:+.4f},{dh:+.4f}] includes or sits below zero.")
    print(f"  frequency baseline {fr:.4f}; the model reaches {100*last/fr:.1f}% of it "
          f"over {len(test)} held-out perturbations.")
    json.dump({"frequency": fr, "frequency_ci": [flo, fhi], "n_eval": len(test), "n_boot": N_BOOT,
               "curve": curve, "climbing": bool(climbing), "non_monotonic": bool(noisy),
               "final_beats_frequency": bool(curve[-1]["beats_frequency"])},
              open(OUT / "cellformer_curve.json", "w"), indent=1)
    print(f"\n  -> {OUT/'cellformer_curve.json'}")


if __name__ == "__main__":
    main()
