"""Active-learning budget sweep with full paper-trail logging.

Pushes the budget out to 10,000 to find where the curve flattens (the question
left open by the first AL run). For each fit, prints:

  - which genes were revealed and the score that ranked them (uncertainty),
  - delta AUC / AUPRC / coverage vs the no-reveal baseline,
  - a sample of 'flipped' genes (the model's prediction changed after revealing
    nearby labels).

Default focus: Ralstonia + uncertainty (the steepest curve in the first run).
Use --clades / --strategies / --budgets to override.

Output: outputs/orphan/al_sweep_deep.json + .png
"""
import json, argparse, time
import numpy as np

from af_common import Cache, OUT, MAJOR_CLADES, auc, auprc, cov_at_p
from af_torch import AFConfig, Trainer


def section(t):
    bar = "=" * 78
    print(f"\n{bar}\n  {t}\n{bar}")


def run(clades=("ralstonia",), strategies=("uncertainty", "random"),
        budgets=(0, 500, 1000, 2000, 5000, 10000), cfg=None, seed=0,
        sample_genes=8):
    cfg = cfg or AFConfig(epochs=6)
    c = Cache()
    tr = Trainer(c, cfg)
    rng = np.random.default_rng(seed)
    out = {}; t_top = time.time()

    section("AL DEEP TRACE  -  configuration")
    print(f"  device       : {cfg.device}")
    print(f"  clades       : {list(clades)}")
    print(f"  strategies   : {list(strategies)}")
    print(f"  budgets      : {list(budgets)}")
    print(f"  per-fit cfg  : epochs={cfg.epochs} bs={cfg.bs} lr={cfg.lr} "
          f"heads={cfg.heads}x{cfg.head_dim} blocks={cfg.blocks}")

    for cl in clades:
        held = np.where(c.clade == cl)[0]
        if len(held) < 200:
            print(f"\n  skip {cl}: only {len(held)} genes")
            continue
        base_train = np.where(c.clade != cl)[0]
        section(f"CLADE = {cl}   (held-out genes: {len(held):,}; training pool: {len(base_train):,})")

        # initial (no reveals) -- gives us the acquisition rankings
        msa0 = tr.masked_msa(cl)
        t0 = time.time()
        m0 = tr.fit(base_train, msa0)
        p0, br0 = tr.predict(m0, held, msa0)
        y0 = c.y[held]
        a0 = auc(p0, y0); ap0 = auprc(p0, y0)
        ec70_0 = cov_at_p(p0, y0, 0.7); ne90_0 = cov_at_p(1 - p0, 1 - y0, 0.9)
        bright0 = float((br0 >= 0.85).mean())
        print(f"\n  baseline (budget=0) fit in {time.time()-t0:.1f}s")
        print(f"    AUC={a0:.4f}  AUPRC={ap0:.4f}  essCov@P70={ec70_0:.4f}  "
              f"neCov@P90={ne90_0:.4f}  bright>=.85={bright0:.4f}")
        uncert = -np.abs(p0 - 0.5)        # higher = more uncertain
        # show the top genes uncertainty would query first
        top_unc = np.argsort(-uncert)[:sample_genes]
        print(f"\n  top {sample_genes} held-clade genes ranked by UNCERTAINTY (|p-0.5| smallest):")
        print(f"    {'org':<28s} {'locus':<18s} {'p0':>6s} {'br0':>6s} {'true':>5s}")
        for i in top_unc:
            full = int(held[i])
            nm = str(c.orgs[int(c.meta_org[full])]); lt = str(c.lt[full])
            print(f"    {nm[:28]:<28s} {lt[:18]:<18s} {p0[i]:>6.3f} {br0[i]:>6.3f} {int(y0[i]):>5d}")

        out[cl] = {}
        for strat in strategies:
            print(f"\n  --- strategy = {strat} ---")
            curve = []
            for B in budgets:
                # pick reveal set
                if B == 0:
                    reveal = np.array([], dtype=int)
                elif strat == "random":
                    reveal = rng.choice(held, min(B, len(held)), replace=False)
                elif strat == "uncertainty":
                    reveal = held[np.argsort(-uncert)[:B]]
                elif strat == "lowbright":
                    reveal = held[np.argsort(br0)[:B]]
                else:
                    raise ValueError(strat)

                t0 = time.time()
                msa_t = tr.masked_msa(cl, reveal_idx=reveal)
                train_idx = np.concatenate([base_train, reveal]).astype(int)
                model = tr.fit(train_idx, msa_t)
                remain = np.setdiff1d(held, reveal)
                p, br = tr.predict(model, remain, msa_t)
                yt = c.y[remain]
                a = auc(p, yt); ap = auprc(p, yt)
                ec70 = cov_at_p(p, yt, 0.7); ne90 = cov_at_p(1 - p, 1 - yt, 0.9)
                bright = float((br >= 0.85).mean())
                dt = time.time() - t0
                rec = dict(budget=int(B), n_eval=int(len(remain)),
                           auc=round(a, 4), auprc=round(ap, 4),
                           ess_cov_p70=round(ec70, 4), ne_cov_p90=round(ne90, 4),
                           bright_cov=round(bright, 4), seconds=round(dt, 1))
                curve.append(rec)
                # delta vs B=0
                if B == 0:
                    a_base = a; ap_base = ap; ec_base = ec70; ne_base = ne90; br_base = bright
                    delta = "  (baseline)"
                else:
                    delta = (f"  d AUC{a-a_base:+.4f}  d AUPRC{ap-ap_base:+.4f}  "
                             f"d ess70{ec70-ec_base:+.4f}  d ne90{ne90-ne_base:+.4f}")
                print(f"    B={B:<6d} n_eval={len(remain):<5d} AUC={a:.4f} AUPRC={ap:.4f} "
                      f"ess70={ec70:.4f} ne90={ne90:.4f} bright={bright:.4f} ({dt:.1f}s){delta}")
                # if the fit produced a flip from baseline on the top-uncertain genes, show a few
                if B > 0 and strat == "uncertainty":
                    # find genes that were uncertain in baseline AND are now confident
                    remain_set = set(remain.tolist())
                    # map remain positions to original held positions for comparison
                    held_pos = {int(held[i]): i for i in range(len(held))}
                    rem_pos = {int(remain[j]): j for j in range(len(remain))}
                    flipped = []
                    for i in top_unc[:sample_genes]:
                        full = int(held[i])
                        if full in rem_pos:
                            j = rem_pos[full]
                            if abs(p[j] - 0.5) > 0.3:    # now confident
                                flipped.append((full, p0[i], p[j], int(y0[i])))
                    if flipped:
                        print(f"      flipped-to-confident sample (was unsure -> now sure):")
                        for full, p_old, p_new, tr_ in flipped[:5]:
                            nm = str(c.orgs[int(c.meta_org[full])])
                            lt = str(c.lt[full])
                            print(f"        {nm[:24]:<24s} {lt[:18]:<18s} "
                                  f"p {p_old:.3f} -> {p_new:.3f}  truth={tr_}")
            out[cl][strat] = curve

        # detect plateau on each strategy
        print(f"\n  saturation check for {cl}:")
        for strat, cv in out[cl].items():
            if len(cv) < 3: continue
            auc_seq = [r["auc"] for r in cv]
            # compute marginal AUC per 1000 labels for the last two steps
            ms = []
            for k in range(1, len(cv)):
                dB = cv[k]["budget"] - cv[k-1]["budget"]
                if dB > 0:
                    ms.append(((cv[k]["auc"] - cv[k-1]["auc"]) / dB) * 1000)
            print(f"    {strat:<11s} AUC trajectory: " +
                  " -> ".join(f"{a:.4f}" for a in auc_seq))
            if ms:
                print(f"    {strat:<11s} marginal AUC per 1000 labels (segment-by-segment): " +
                      "  ".join(f"{m:+.4f}" for m in ms))
                plateau = ms[-1] < 0.002
                print(f"    {strat:<11s} -> {'PLATEAUED' if plateau else 'still climbing'} at top budget")

    section("DONE")
    print(f"  total wall time: {time.time()-t_top:.0f}s")
    out_json = dict(config=cfg.__dict__, clades=list(clades),
                    strategies=list(strategies), budgets=list(budgets),
                    results=out)
    json.dump(out_json, open(OUT / "al_sweep_deep.json", "w"), indent=2)
    print(f"  wrote al_sweep_deep.json")

    _plot(out, list(budgets))
    return out


def _plot(results, budgets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clades = list(results)
    if not clades: return
    fig, axes = plt.subplots(1, max(1, len(clades)), figsize=(7 * len(clades), 5), squeeze=False)
    for ax, cl in zip(axes[0], clades):
        for strat, cv in results[cl].items():
            ax.plot([r["budget"] for r in cv], [r["auc"] for r in cv], "o-", label=strat, lw=2)
        ax.set_title(f"{cl}: AL budget sweep")
        ax.set_xlabel("# held-clade labels revealed"); ax.set_ylabel("AUC on remainder")
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle("Active learning - where does the curve flatten?", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "al_sweep_deep.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clades", default="ralstonia")
    ap.add_argument("--strategies", default="uncertainty,random")
    ap.add_argument("--budgets", default="0,500,1000,2000,5000,10000")
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    cfg = AFConfig(epochs=a.epochs)
    run(clades=tuple(a.clades.split(",")),
        strategies=tuple(a.strategies.split(",")),
        budgets=tuple(int(x) for x in a.budgets.split(",")),
        cfg=cfg)
