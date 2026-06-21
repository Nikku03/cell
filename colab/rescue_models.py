"""Does a more powerful rescue model beat the logistic stacker?

Same abstention-bucket features as smart_cooccur (transformer p, brightness, +
the cooccur/dN-dS channels), same cross-clade protocol, same 90%-precision
threshold rule -- only the model class changes:

    logistic  (the current stacker)
    mlp       (1 hidden layer, numpy)
    gbm       (sklearn HistGradientBoosting, if available)

If the MLP/GBM don't beat logistic, the rescue ceiling is the FEATURES, not the
model -- which means the only way to "make it a transformer and get more" is to
feed it a new modality (presence/partner matrix), not more capacity on 8 scalars.

Output: outputs/orphan/rescue_models_results.json
"""
import json, argparse
import numpy as np

from af_common import Cache, CACHE, OUT, MAJOR_CLADES
from smart_cooccur import (load_orthology, channel_features,
                            fit_logreg, pred_logreg, PRECISION_FLOOR, BRIGHT,
                            FEATURE_COLS as COLS)


# ---- a small numpy MLP (1 hidden ReLU layer), class-balanced BCE -----------
def fit_mlp(X, y, hidden=32, iters=400, lr=0.05, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd
    n, d = Xs.shape
    W1 = rng.normal(0, 0.3, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.3, hidden); b2 = 0.0
    pos = y.mean() or 1e-6
    w_pos, w_neg = 0.5 / pos, 0.5 / (1 - pos)
    sw = np.where(y == 1, w_pos, w_neg); sw = sw / sw.mean()
    for _ in range(iters):
        h = np.maximum(Xs @ W1 + b1, 0)
        logit = h @ W2 + b2
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        g = (p - y) * sw / n
        dW2 = h.T @ g + l2 * W2; db2 = g.sum()
        dh = np.outer(g, W2) * (h > 0)
        dW1 = Xs.T @ dh + l2 * W1; db1 = dh.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return (W1, b1, W2, b2, mu, sd)


def pred_mlp(model, X):
    W1, b1, W2, b2, mu, sd = model
    Xs = (X - mu) / sd
    h = np.maximum(Xs @ W1 + b1, 0)
    return 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -30, 30)))


def fit_gbm(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                       learning_rate=0.05, l2_regularization=1.0,
                                       class_weight="balanced")
    m.fit(X, y)
    return m


def pred_gbm(model, X):
    return model.predict_proba(X)[:, 1]


def fit_skmlp(X, y):
    """sklearn MLP (preferred when available -- properly tuned/calibrated)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                    alpha=1e-3, early_stopping=True, random_state=0))
    m.fit(X, y)
    return m


def pred_skmlp(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate(preds, X, y, meta, fit_fn, pred_fn, label):
    p_all = preds["p"]; br_all = preds["brightness"]; y_all = preds["y"]
    clades = np.array([m[0] for m in meta])
    ev = ~np.isnan(p_all); hi = ev & (br_all >= BRIGHT)
    T_called = int(hi.sum()); T_correct = int(((p_all[hi] > 0.5).astype(int) == y_all[hi]).sum())
    N = int(ev.sum())

    stack_p = np.zeros(len(y))
    for cl in np.unique(clades):           # all clades present, not just the 5 majors
        te = clades == cl; trn = ~te
        if te.sum() == 0 or trn.sum() < 50:
            continue
        m = fit_fn(X[trn], y[trn])
        stack_p[te] = pred_fn(m, X[te])
    conf = np.maximum(stack_p, 1 - stack_p)
    correct = ((stack_p >= 0.5).astype(int) == y).astype(int)
    order = np.argsort(-conf)
    cc = np.cumsum(correct[order]); cn = np.arange(1, len(order) + 1)
    best = dict(cov=T_called / N, prec=(T_correct / T_called) if T_called else 0, resc=0, tau=1.0, racc=0.0)
    for k in range(len(order)):
        prec = (T_correct + cc[k]) / (T_called + cn[k])
        if prec >= PRECISION_FLOOR and cn[k] > best["resc"]:
            best = dict(cov=(T_called + cn[k]) / N, prec=prec, resc=int(cn[k]),
                        tau=float(conf[order[k]]), racc=cc[k] / cn[k])
    print(f"  {label:<10s} combined {best['cov']:.4f} @ {best['prec']:.4f}  "
          f"(rescued {best['resc']} @ {best['racc']:.3f}, tau={best['tau']:.3f})")
    return dict(label=label, coverage=round(best["cov"], 4), precision=round(best["prec"], 4),
                rescued=best["resc"], rescued_acc=round(best["racc"], 4), tau=round(best["tau"], 4))


def main(cache_path=None, preds_path=None, labels_dir=None, out_tag=""):
    preds = dict(np.load(preds_path or (OUT / "af_torch_preds.npz"), allow_pickle=True))
    c = Cache(cache_path or CACHE)
    preds["foc"] = c.foc; preds["orgs"] = np.array(c.orgs, dtype=object)
    orth = load_orthology(labels_dir=labels_dir)
    X, y, meta = channel_features(preds, orth, use_dnds=True)
    print(f"abstention-bucket genes: {len(y)}  features: {COLS}\n")

    models = [("logistic", fit_logreg, pred_logreg)]
    try:
        import sklearn  # noqa
        models.append(("mlp", fit_skmlp, pred_skmlp))     # sklearn MLP (preferred)
        models.append(("gbm", fit_gbm, pred_gbm))
    except Exception:
        print("  (sklearn not available -> numpy-MLP fallback, no gbm)\n")
        models.append(("mlp", fit_mlp, pred_mlp))
    out = {}
    for name, ff, pf in models:
        out[name] = evaluate(preds, X, y, meta, ff, pf, name)
    base = out["logistic"]["coverage"]
    print("\n  vs logistic:")
    for name in out:
        if name != "logistic":
            print(f"    {name:<8s} {(out[name]['coverage']-base)*100:+.2f}pp coverage")
    json.dump(out, open(OUT / f"rescue_models_results{out_tag}.json", "w"), indent=2)
    print(f"\n  wrote rescue_models_results{out_tag}.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--labels_dir", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(cache_path=a.cache, preds_path=a.preds, labels_dir=a.labels_dir, out_tag=a.tag)
