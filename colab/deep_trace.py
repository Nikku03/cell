"""Deep trace: channel ablation of the smart cooccur stacker, with every step
printed like the model is solving it on paper.

What this exposes:
  1. The raw inputs (transformer predictions distribution, channel score stats).
  2. Channel ablation table: drop each channel in turn and report marginal
     coverage at the 90% precision floor.
  3. Stacker coefficients on the full-feature model -- which channels the
     calibrated logistic regression actually leans on.
  4. Hand-picked example genes: 5 highest-confidence rescues + 5 just-below
     threshold, with every channel score visible alongside the decision.
  5. Channel score distributions on rescued vs not-rescued populations.

Reuses smart_cooccur.py's data plumbing (no retraining; reads
outputs/orphan/af_torch_preds.npz).

Output: outputs/orphan/deep_trace_ablation.json (machine-readable summary)
"""
import json
from pathlib import Path
import numpy as np

from af_common import Cache, OUT, MAJOR_CLADES
from smart_cooccur import (load_orthology, channel_features, evaluate,
                            fit_logreg, pred_logreg, BRIGHT, PRECISION_FLOOR,
                            FEATURE_COLS as COLS)


def section(t):
    bar = "=" * 78
    print(f"\n{bar}\n  {t}\n{bar}")


def stat_line(name, x):
    x = np.asarray(x, np.float64)
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        print(f"  {name:<25s} (empty)")
        return
    print(f"  {name:<25s} n={len(finite):<7d} min={finite.min():+.4f} "
          f"max={finite.max():+.4f} mean={finite.mean():+.4f} std={finite.std():.4f} "
          f"nonzero={(finite != 0).sum()}")


def histogram(name, x, bins=10, width=40):
    """ASCII histogram so you can see the shape on the terminal."""
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    if len(x) == 0:
        return
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        hi = lo + 1e-9
    h, edges = np.histogram(x, bins=bins, range=(lo, hi))
    mx = max(h.max(), 1)
    print(f"\n  {name} (n={len(x)})")
    for i, c in enumerate(h):
        bar = "#" * int(round(width * c / mx))
        print(f"    [{edges[i]:+7.3f}, {edges[i+1]:+7.3f})  {c:>6d}  {bar}")


def main():
    cache = Cache()

    # ---------------------------------------------------------------- STAGE 1
    section("STAGE 1 - Load transformer predictions and assemble channels")
    preds = dict(np.load(OUT / "af_torch_preds.npz", allow_pickle=True))
    preds["foc"] = cache.foc
    preds["orgs"] = np.array(cache.orgs, dtype=object)
    p_all = preds["p"]; br_all = preds["brightness"]; y_all = preds["y"]; clade = preds["clade"]
    ev = ~np.isnan(p_all)
    print(f"  total cache genes:           {len(p_all):,}")
    print(f"  evaluated (in major clades): {int(ev.sum()):,}")
    print(f"  base essentiality rate:      {y_all[ev].mean():.4f}")
    stat_line("transformer p", p_all[ev])
    stat_line("brightness", br_all[ev])
    print(f"\n  brightness>=0.85 (transformer-confident bucket): "
          f"{int((br_all[ev] >= BRIGHT).sum()):,} genes "
          f"({(br_all[ev] >= BRIGHT).mean():.1%})")
    print(f"  brightness< 0.85 (abstention bucket -> input to stacker): "
          f"{int((br_all[ev] < BRIGHT).sum()):,} genes "
          f"({(br_all[ev] < BRIGHT).mean():.1%})")
    histogram("brightness distribution (evaluated genes)", br_all[ev], bins=12, width=40)

    orth = load_orthology()
    print(f"\n  orthology matrix: {orth['O']} orgs x {orth['G']} OGs")
    print(f"  candidate presence partners: {int(((orth['P'].sum(0) >= 5) & (orth['P'].sum(0) <= orth['O']-5)).sum()):,}")
    print(f"  candidate essentiality partners: "
          f"{int(((~np.isnan(orth['Y'])).sum(0) >= 10).sum()):,}")
    print(f"  prebuilt phyletic feature: {len(orth['cofeat']):,} genes")

    X, y, meta = channel_features(preds, orth, use_dnds=True)
    print(f"\n  channel feature matrix: {X.shape}  (rows = abstained genes, cols = {COLS})")
    print(f"  abstained essentiality rate: {y.mean():.4f}")
    for i, name in enumerate(COLS):
        stat_line(name, X[:, i])

    # ---------------------------------------------------------------- STAGE 2
    section("STAGE 2 - Channel ablation table  (cross-clade, P>=0.90 floor)")
    print("  Each row drops the listed channel(s) by zeroing those columns,")
    print("  then refits the cross-clade stacker. Coverage delta vs baseline =")
    print("  that channel's marginal contribution.\n")

    drop_specs = [
        ("baseline (all channels)",                    []),
        ("- backup",                                    ["backup"]),
        ("- presence",                                  ["presence"]),
        ("- coess",                                     ["coess"]),
        ("- phyl",                                      ["phyl"]),
        ("- dnds (+ has_dnds)",                         ["dnds", "has_dnds"]),
        ("- backup AND presence (paired)",              ["backup", "presence"]),
        ("- all cooccur, keep brightness+p only",       ["backup", "presence", "coess", "phyl", "dnds", "has_dnds"]),
    ]
    table = []
    baseline_cov = None; baseline_resc = None
    print(f"  {'config':<48s}  {'cov':>7s}  {'prec':>7s}  {'rescued':>8s}  {'tau':>6s}  {'Δcov_pp':>8s}")
    print(f"  {'-'*48}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*8}")
    for label, drop in drop_specs:
        Xa = X.copy()
        for col in drop:
            Xa[:, COLS.index(col)] = 0.0
        res = evaluate(preds, Xa, y, meta, label=label)
        s = res["smart_combined"]
        if baseline_cov is None:
            baseline_cov = s["coverage"]; baseline_resc = s["rescued"]
            dlt = " (ref)"
        else:
            dlt = f"{(s['coverage'] - baseline_cov)*100:+.2f}"
        print(f"  {label:<48s}  {s['coverage']:>7.4f}  {s['precision']:>7.4f}  "
              f"{s['rescued']:>8d}  {s['tau']:>6.3f}  {dlt:>8s}")
        table.append(dict(label=label, dropped=drop, **s))
    print(f"\n  baseline rescued {baseline_resc} genes.")

    # ---------------------------------------------------------------- STAGE 3
    section("STAGE 3 - Stacker coefficients (full-feature, fit-on-all)")
    print("  These are the standardised logistic-regression weights the calibrated")
    print("  stacker learns when given ALL channels (intercept + one weight each).")
    print("  Magnitude > 0.3 = strongly used; sign tells the direction the channel")
    print("  pushes the essentiality call.\n")
    model = fit_logreg(X, y)
    w, mu, sd = model
    print(f"  {'feature':<14s}  {'mean':>9s}  {'std':>9s}  {'weight':>9s}  pull")
    print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*9}  ----")
    print(f"  {'(intercept)':<14s}  {'-':>9s}  {'-':>9s}  {w[0]:>+9.4f}")
    for i, c in enumerate(COLS):
        wi = w[i+1]
        pull = "++" if wi > 0.5 else ("+" if wi > 0.1 else ("--" if wi < -0.5 else ("-" if wi < -0.1 else ".")))
        print(f"  {c:<14s}  {mu[i]:>+9.4f}  {sd[i]:>9.4f}  {wi:>+9.4f}   {pull}")

    # ---------------------------------------------------------------- STAGE 4
    section("STAGE 4 - Example genes: top rescues and just-below-threshold")
    # Cross-clade predict to get an honest score per abstained gene
    clades_arr = np.array([m[0] for m in meta])
    stack_p = np.zeros(len(y))
    for cl in MAJOR_CLADES:
        te = clades_arr == cl; trn = ~te
        if te.sum() == 0 or trn.sum() < 50: continue
        mdl = fit_logreg(X[trn], y[trn])
        stack_p[te] = pred_logreg(mdl, X[te])
    conf = np.maximum(stack_p, 1 - stack_p)
    call = (stack_p >= 0.5).astype(int)

    # find the tau the baseline picked
    baseline = next(t for t in table if t["label"].startswith("baseline"))
    tau = baseline["tau"]
    accepted = conf >= tau
    # rank
    order = np.argsort(-conf)
    accepted_idx = [i for i in order if accepted[i]]
    nearmiss_idx = [i for i in order if not accepted[i]][:5]

    def gene_row(i):
        cl_i, full_i = meta[i]
        nm = str(preds["orgs"][int(preds["meta_org"][full_i])])
        lt = str(preds["lt"][full_i])
        return cl_i, nm, lt, full_i

    print(f"  acceptance threshold tau = {tau:.4f}; rescued = {accepted.sum()}\n")
    print("  --- Top 5 highest-confidence rescues ---")
    print(f"  {'clade':<11s} {'org':<28s} {'locus':<18s} {'p':>6s} {'br':>5s} "
          f"{'bkp':>6s} {'pres':>6s} {'coess':>6s} {'phyl':>6s} {'dnds':>6s} {'stack':>7s} {'true':>4s} ok")
    for i in accepted_idx[:5]:
        cl_i, nm, lt, full_i = gene_row(i)
        row = X[i]
        truth = int(y[i]); pred = int(call[i])
        ok = "Y" if pred == truth else "N"
        print(f"  {cl_i:<11s} {nm[:28]:<28s} {lt[:18]:<18s} {row[0]:>6.3f} {row[1]:>5.3f} "
              f"{row[2]:>+6.2f} {row[3]:>+6.2f} {row[4]:>+6.2f} {row[5]:>+6.2f} {row[6]:>+6.2f} "
              f"{stack_p[i]:>7.4f} {truth:>4d}  {ok}")
    print("\n  --- 5 just-below-threshold (abstained by stacker too) ---")
    nearest = np.argsort(np.abs(conf - tau))[:5]
    for i in nearest:
        cl_i, nm, lt, full_i = gene_row(i)
        row = X[i]
        truth = int(y[i]); pred = int(call[i])
        was = "rescued" if accepted[i] else "abstain"
        print(f"  {cl_i:<11s} {nm[:28]:<28s} {lt[:18]:<18s} {row[0]:>6.3f} {row[1]:>5.3f} "
              f"{row[2]:>+6.2f} {row[3]:>+6.2f} {row[4]:>+6.2f} {row[5]:>+6.2f} {row[6]:>+6.2f} "
              f"{stack_p[i]:>7.4f} {truth:>4d}  {was}")

    # ---------------------------------------------------------------- STAGE 5
    section("STAGE 5 - Channel score distributions: rescued vs not-rescued")
    for name in ["backup", "presence", "coess", "phyl"]:
        col = COLS.index(name)
        print(f"\n  channel: {name}")
        histogram(f"    rescued    ", X[accepted, col], bins=10, width=30)
        histogram(f"    not-rescued", X[~accepted, col], bins=10, width=30)

    # ---------------------------------------------------------------- save
    out = dict(
        n_abstained=int(len(y)),
        baseline_coverage=float(baseline_cov),
        baseline_rescued=int(baseline_resc),
        ablation=table,
        stacker_weights={c: float(w[i+1]) for i, c in enumerate(COLS)},
        stacker_intercept=float(w[0]),
        tau=float(tau))
    json.dump(out, open(OUT / "deep_trace_ablation.json", "w"), indent=2)
    print(f"\n  wrote deep_trace_ablation.json")


if __name__ == "__main__":
    main()
