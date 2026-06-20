"""Active-learning (self-learning) loop: how many NEW labels would it take to
move the conditional residual, and which genes should we query?

Simulation (leave-one-clade-out, honest): hold out a clade entirely, then
iteratively 'reveal' a budget of its labels back to the trainer (added as
supervised targets, their MSA rows un-masked), retrain, and measure AUC /
coverage on the STILL-HIDDEN remainder of the clade. We compare acquisition
strategies:
  - random      : reveal a random subset
  - uncertainty : reveal genes the model is least sure about (|p-0.5| small)
  - lowbright   : reveal genes with the lowest brightness (the abstention bucket)

A steep curve => labels are cheap and active learning works; a flat curve =>
the residual is irreducibly conditional and we need condition-aware data, not
more labels. Output: outputs/orphan/active_learning_results.json + .png
"""
import json, argparse
import numpy as np

from af_common import Cache, OUT, auc, cov_at_p
from af_torch import AFConfig, Trainer


def run(clades=("ralstonia", "shewanella"), budgets=(0, 200, 500, 1000, 2000),
        strategies=("random", "uncertainty", "lowbright"), cfg=None, seed=0):
    cfg = cfg or AFConfig(epochs=6)
    c = Cache(); tr = Trainer(c, cfg)
    rng = np.random.default_rng(seed)
    results = {}
    for cl in clades:
        held = np.where(c.clade == cl)[0]
        if len(held) < 200:
            continue
        base_train = np.where(c.clade != cl)[0]
        # initial model (no revealed labels) to get acquisition rankings
        msa0 = tr.masked_msa(cl)
        m0 = tr.fit(base_train, msa0)
        p0, br0 = tr.predict(m0, held, msa0)
        uncert = -np.abs(p0 - 0.5)          # higher = more uncertain
        results[cl] = {}
        for strat in strategies:
            curve = []
            for B in budgets:
                if B == 0:
                    reveal = np.array([], dtype=int)
                else:
                    if strat == "random":
                        reveal = rng.choice(held, min(B, len(held)), replace=False)
                    elif strat == "uncertainty":
                        reveal = held[np.argsort(-uncert)[:B]]
                    else:  # lowbright
                        reveal = held[np.argsort(br0)[:B]]
                msa_t = tr.masked_msa(cl, reveal_idx=reveal)
                train_idx = np.concatenate([base_train, reveal]).astype(int)
                model = tr.fit(train_idx, msa_t)
                remain = np.setdiff1d(held, reveal)
                p, br = tr.predict(model, remain, msa_t)
                yt = c.y[remain]
                curve.append(dict(budget=int(B), n_eval=int(len(remain)),
                                  auc=round(auc(p, yt), 4),
                                  ess_cov_p70=round(cov_at_p(p, yt, 0.7), 4),
                                  bright_cov=round(float((br >= 0.85).mean()), 4)))
                print(f"  {cl:<11} {strat:<11} budget={B:<5} AUC={curve[-1]['auc']:.3f} "
                      f"essCov@P70={curve[-1]['ess_cov_p70']:.3f}")
            results[cl][strat] = curve
    json.dump(dict(config=cfg.__dict__, budgets=list(budgets), results=results),
              open(OUT / "active_learning_results.json", "w"), indent=2)
    _plot(results, budgets)
    print("wrote active_learning_results.json + active_learning.png")
    return results


def _plot(results, budgets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clades = list(results)
    fig, axes = plt.subplots(1, max(1, len(clades)), figsize=(6 * len(clades), 5), squeeze=False)
    for ax, cl in zip(axes[0], clades):
        for strat, cv in results[cl].items():
            ax.plot([d["budget"] for d in cv], [d["auc"] for d in cv], "o-", label=strat)
        ax.set_title(f"{cl}: AUC vs labels revealed")
        ax.set_xlabel("# held-clade labels revealed"); ax.set_ylabel("AUC on remainder")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Active-learning curves (steep => labels help; flat => residual is conditional)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "active_learning.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clades", default="ralstonia,shewanella")
    ap.add_argument("--budgets", default="0,200,500,1000,2000")
    a = ap.parse_args()
    run(clades=tuple(a.clades.split(",")),
        budgets=tuple(int(x) for x in a.budgets.split(",")))
