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
N_EVAL = int(os.environ.get("CF_NEVAL", "20"))


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
    fr = float(np.mean([len(set(fp) & set(np.where(d["spec"][q])[0].tolist()))
                        / len(set(np.where(d["spec"][q])[0].tolist())) for q in test]))
    print(f"held-out {len(test)} perturbations; candidate pool {len(d['cand']):,} genes "
          f"(contains all true movers)")
    print(f"frequency baseline: recall {fr:.4f}\n", flush=True)

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
        model.eval()
        recs = []
        with torch.no_grad():
            for qi in test:
                truth = set(np.where(d["spec"][qi])[0].tolist())
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
        return float(np.mean(recs))

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
            rc = evaluate()
            curve.append({"step": step, "loss": float(np.mean(losses[-100:])), "recall": rc})
            print(f"  step {step:>5d}   loss {np.mean(losses[-100:]):.4f}   held-out recall {rc:.4f}   "
                  f"(frequency {fr:.4f})   [{time.time()-t0:.0f}s]", flush=True)
            json.dump({"frequency": fr, "n_eval": len(test), "curve": curve},
                      open(OUT / "cellformer_curve.json", "w"), indent=1)

    print("\n" + "=" * 78)
    print(f"{'step':>7s} {'loss':>9s} {'recall':>9s} {'% of frequency':>16s}")
    for c_ in curve:
        print(f"{c_['step']:>7d} {c_['loss']:>9.4f} {c_['recall']:>9.4f} {100*c_['recall']/fr:>15.1f}%")
    print("=" * 78)
    first, last = curve[0]["recall"], curve[-1]["recall"]
    climbing = last > first * 1.3
    print(f"\n  recall {first:.4f} -> {last:.4f} over {curve[0]['step']} -> {curve[-1]['step']} steps")
    print(f"  VERDICT: {'STILL CLIMBING -- undertrained, the ablation table needs a larger budget' if climbing else 'FLAT -- more training does not close the gap; the architecture is the limit'}")
    print(f"  frequency baseline remains {fr:.4f}; the model reaches {100*last/fr:.1f}% of it.")
    json.dump({"frequency": fr, "n_eval": len(test), "curve": curve, "climbing": bool(climbing)},
              open(OUT / "cellformer_curve.json", "w"), indent=1)
    print(f"\n  -> {OUT/'cellformer_curve.json'}")


if __name__ == "__main__":
    main()
