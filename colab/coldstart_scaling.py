"""coldstart_scaling -- is the cold-start model limited by DATA or by MODEL CAPACITY?

This decides whether it is worth training a bigger model (transformer, deep net) on this task, and the answer is not a matter of taste --
a learning curve settles it. If held-out recall is still climbing at the largest training set we can build, the binding constraint is the
number of measured knockouts and extra capacity will only overfit harder. If the curve has flattened, capacity is worth buying.

Three pieces of evidence already point the same way and this makes it explicit:
  - the ridge does BETTER at alpha=100 than at alpha=10, i.e. it wants MORE regularisation, which is the signature of too little data
  - the response matrix has participation ratio ~21.6, so there are only about 21 distinguishable things a knockout does here
  - 1-NN retrieval, which has no parameters at all, reaches 0.288 against the ridge's 0.341

TWO CURVES, because they answer different questions:
  FULL      everything is rebuilt from the training subset -- programs, mu, tide mask, vocabularies, scaling, ridge. This is what you would
            actually have if you had run fewer experiments, so it is the curve that says whether MEASURING MORE KNOCKOUTS pays.
  REGRESSOR the programs/mu/tide are held fixed at the full training set and only the ridge's training rows are subsampled. Comparing it to
            FULL localises the limitation: if REGRESSOR is much flatter, the shortage is in the program basis, not in the annotation->loading map.

Test set is held fixed across every training size within a split, so the curves are directly comparable. Deterministic."""
import os
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TIDE_FRAC = 0.05
MINSRC = 20
MAXCOMPLEX = 200
K_RECALL = 50
KPROG = 50
ALPHAS = (10.0, 100.0)
SEEDS = (0, 1, 2, 3, 4)
NTRAIN = (40, 80, 120, 160, 200, 235)     # 235 = the whole 70% train half


def main():
    from sklearn.linear_model import Ridge
    H = Harness("K562")
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); Aall = np.abs(Z)

    tc = [int((np.abs(H.M[H.ki[k]]) >= 1.0).sum()) for k in H.kos
          if k in kidx and (np.abs(H.M[H.ki[k]]) > 0).any() and np.abs(H.M[H.ki[k]])[np.abs(H.M[H.ki[k]]) > 0].min() < 1.0]
    med = float(np.median(tc)); T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        e = abs(float(np.median((Aall >= t).sum(1))) - med)
        if best is None or e < best:
            best, T = e, float(t)
    per = (Aall >= T).sum(1)
    sources = [kos[i] for i in range(len(kos)) if per[i] >= MINSRC]
    si_of = {s: i for i, s in enumerate(sources)}
    print(f"K562: |z|>={T:.1f} | {len(sources)} sources | learning curve over {list(NTRAIN)} training knockouts")

    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    comps = collections.defaultdict(set)
    for cid, mem in D["complexes"].items():
        mm = sorted({names[x] for x in mem if isinstance(x, int) and x < N})
        if 2 <= len(mm) <= MAXCOMPLEX:
            for g in mm:
                comps[g].add(cid)
    doms = {}
    try:
        for k, v in json.load(open(OUT / "domains.json")).get("domains", {}).items():
            try:
                doms[names[int(k)]] = set(str(x) for x in v)
            except (ValueError, IndexError):
                pass
    except Exception:
        pass
    BE = json.load(open(SP / "k562_baseline_expression.json"))
    base_panel = BE["measured_baseline"]; ctrl_expr = BE["source_control_expr"]
    NUM = ["ess", "loeuf", "tf", "ppi", "npath", "ndis", "cpg", "pubs", "dark", "conf", "enh", "dep_frac"]

    def build_X(train):
        def vocab(getter, minn):
            c = collections.Counter()
            for s in train:
                c.update(getter(s))
            return sorted(k for k, v in c.items() if v >= minn)
        v_comp = vocab(lambda s: comps.get(s, ()), 2)
        v_dom = vocab(lambda s: doms.get(s, ()), 3)
        v_proc = vocab(lambda s: [(info.get(s, {}) or {}).get("proc") or "?"], 3)
        v_loc = vocab(lambda s: [(info.get(s, {}) or {}).get("comp") or "?"], 3)

        def feats(s):
            gi = info.get(s, {}) or {}
            x = []
            for k in NUM:
                try:
                    x.append(float(gi.get(k) or 0.0))
                except (TypeError, ValueError):
                    x.append(0.0)
            bp = base_panel.get(s)
            x.append(np.log1p(float(ctrl_expr.get(s, 0.0))))
            x.append(1.0 if s in ctrl_expr else 0.0)
            x += [np.log1p(bp[0]), bp[1], bp[2], bp[3], 1.0] if bp else [0.0] * 5
            x += [1.0 if (gi.get("proc") or "?") == p else 0.0 for p in v_proc]
            x += [1.0 if (gi.get("comp") or "?") == p else 0.0 for p in v_loc]
            dl = doms.get(s, set())
            x += [1.0 if d in dl else 0.0 for d in v_dom]
            cl = comps.get(s, set())
            x += [1.0 if c in cl else 0.0 for c in v_comp]
            return x
        return np.array([feats(s) for s in sources], float)

    def fit_predict(train, test, Vt, mu, keep, Xall, alpha, kk):
        Xtr = Xall[[si_of[s] for s in train]]; Xte = Xall[[si_of[s] for s in test]]
        m = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
        Ptr = (Z[[kidx[s] for s in train]][:, keep].astype(np.float64) - mu) @ Vt[:kk].T
        Pte = Ridge(alpha=alpha).fit((Xtr - m) / sd, Ptr).predict((Xte - m) / sd)
        return mu + Pte @ Vt[:kk]

    res = collections.defaultdict(list)
    per_seed = collections.defaultdict(dict)      # label -> seed -> mean recall, for honest error bars
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        seed_acc = collections.defaultdict(list)
        ss = sorted(sources); rng.shuffle(ss)
        cut = int(0.7 * len(ss)); pool, test = ss[:cut], ss[cut:]

        # reference basis + tide from the FULL training half, used by the REGRESSOR curve
        rows_full = [kidx[s] for s in pool]
        tide_f = (Aall[rows_full] >= T).mean(0) >= TIDE_FRAC
        keep_f = np.where(~tide_f)[0]
        Mf = Z[rows_full][:, keep_f].astype(np.float64); mu_f = Mf.mean(0)
        _, _, Vt_f = np.linalg.svd(Mf - mu_f, full_matrices=False)
        X_f = build_X(pool)
        truth_f = {}
        for s in test:
            t = {genes[q] for q in np.where((Aall[kidx[s]] >= T) & ~tide_f)[0] if genes[q] != s}
            if t:
                truth_f[s] = t

        for n in NTRAIN:
            sub = pool[:n]
            # ---- FULL curve: everything rebuilt from the subset ----
            rows = [kidx[s] for s in sub]
            tide_s = (Aall[rows] >= T).mean(0) >= TIDE_FRAC
            keep_s = np.where(~tide_s)[0]
            Ms = Z[rows][:, keep_s].astype(np.float64); mu_s = Ms.mean(0)
            _, _, Vt_s = np.linalg.svd(Ms - mu_s, full_matrices=False)
            ks = min(KPROG, len(sub) - 1, Vt_s.shape[0])
            X_s = build_X(sub)
            truth_s = {}
            for s in test:
                t = {genes[q] for q in np.where((Aall[kidx[s]] >= T) & ~tide_s)[0] if genes[q] != s}
                if t:
                    truth_s[s] = t
            for a in ALPHAS:
                P = fit_predict(sub, test, Vt_s, mu_s, keep_s, X_s, a, ks)
                for j, s in enumerate(test):
                    if s not in truth_s:
                        continue
                    pr = np.zeros(len(genes)); pr[keep_s] = P[j]
                    top = keep_s[np.argsort(-np.abs(pr[keep_s]))[:K_RECALL]]
                    res[f"FULL a={a} n={n}"].append(
                        len({genes[q] for q in top} & truth_s[s]) / min(len(truth_s[s]), K_RECALL))
                    seed_acc[f"FULL a={a} n={n}"].append(res[f"FULL a={a} n={n}"][-1])
            # ---- REGRESSOR curve: basis/tide/vocab fixed at full train, only ridge rows subsampled ----
            for a in ALPHAS:
                P = fit_predict(sub, test, Vt_f, mu_f, keep_f, X_f, a, KPROG)
                for j, s in enumerate(test):
                    if s not in truth_f:
                        continue
                    pr = np.zeros(len(genes)); pr[keep_f] = P[j]
                    top = keep_f[np.argsort(-np.abs(pr[keep_f]))[:K_RECALL]]
                    res[f"REGR a={a} n={n}"].append(
                        len({genes[q] for q in top} & truth_f[s]) / min(len(truth_f[s]), K_RECALL))
                    seed_acc[f"REGR a={a} n={n}"].append(res[f"REGR a={a} n={n}"][-1])

        for lbl, vals in seed_acc.items():
            per_seed[lbl][seed] = float(np.mean(vals))

    print(f"\n  LEARNING CURVE -- recall@50 on a fixed held-out test set, mean over {len(SEEDS)} seeds")
    print(f"  {'n_train':>8s} " + " ".join(f"{('FULL a=' + format(a, 'g')):>13s}" for a in ALPHAS)
          + " " + " ".join(f"{('REGR a=' + format(a, 'g')):>13s}" for a in ALPHAS))
    curve = {}
    for n in NTRAIN:
        row = f"  {n:>8d} "
        for tag in ("FULL", "REGR"):
            for a in ALPHAS:
                v = float(np.mean(res[f"{tag} a={a} n={n}"])); curve[f"{tag} a={a} n={n}"] = v
                row += f" {v:>13.4f}"
        print(row)

    # READ THE CURVE ON LOG-X. The raw last step is NOT the right statistic: 200->235 is only 0.23 doublings while 40->80 is a
    # full one, so a small raw gain there says nothing about saturation. A first draft of this script called the curve "flattened"
    # on exactly that mistake. What matters is the gain PER DOUBLING, and whether the final segments depart from the log-linear
    # trend by more than the across-seed noise.
    def se(lbl):
        v = np.array([per_seed[lbl][sd] for sd in SEEDS])
        return float(v.std(ddof=1) / np.sqrt(len(SEEDS)))

    L = np.log2(np.array(NTRAIN, float))
    fits = {}
    print("\n  read on log-x: gain per DOUBLING of training knockouts (+- across-seed SE of each point)")
    for tag in ("FULL", "REGR"):
        for a in ALPHAS:
            v = np.array([curve[f"{tag} a={a} n={n}"] for n in NTRAIN])
            slope, icept = np.polyfit(L, v, 1)
            r2 = float(np.corrcoef(L, v)[0, 1] ** 2)
            segs = [(v[i + 1] - v[i]) / (L[i + 1] - L[i]) for i in range(len(NTRAIN) - 1)]
            fits[f"{tag} a={a}"] = {"slope_per_doubling": round(float(slope), 4), "r2": round(r2, 4),
                                    "segments_per_doubling": [round(float(x), 4) for x in segs],
                                    "se_at_largest": round(se(f"{tag} a={a} n={NTRAIN[-1]}"), 4),
                                    "extrapolate_one_doubling": round(float(v[-1] + slope), 4)}
            print(f"  {tag} a={a:>5g}: {slope:+.4f}/doubling (R^2={r2:.3f}), SE at n={NTRAIN[-1]} is +-{se(f'{tag} a={a} n={NTRAIN[-1]}'):.4f}"
                  f" | segments {[f'{x:+.3f}' for x in segs]}")

    best = max(ALPHAS, key=lambda a: curve[f"FULL a={a} n={NTRAIN[-1]}"])
    F = fits[f"FULL a={best}"]
    slope = F["slope_per_doubling"]; sem = F["se_at_largest"]
    # "saturated" would mean the fitted slope is small relative to the noise on a single point; it is not enough for one short
    # segment to look flat.
    still_climbing = bool(slope > 2 * sem)
    outs = {f"{tag} a={a}": {"final": round(curve[f"{tag} a={a} n={NTRAIN[-1]}"], 4), **fits[f"{tag} a={a}"]}
            for tag in ("FULL", "REGR") for a in ALPHAS}
    # where is the shortage -- program basis or the annotation->loading map?
    gaps = {n: round(curve[f"REGR a={best} n={n}"] - curve[f"FULL a={best} n={n}"], 4) for n in NTRAIN}
    basis_saturates = next((n for n in NTRAIN if gaps[n] <= 0.01), NTRAIN[-1])
    print(f"\n  REGR minus FULL (value of already having the full program basis): "
          + "  ".join(f"n={n}:{gaps[n]:+.3f}" for n in NTRAIN))
    print(f"  -> the program basis stops being the bottleneck at about n={basis_saturates} knockouts; beyond that the limit is the "
          f"annotation->loading map")

    verdict = (
        "COLD-START SCALING (coldstart_scaling.py): a learning curve to decide whether the cold-start model is limited by DATA or by MODEL "
        "CAPACITY -- i.e. whether it is worth training a transformer or other high-capacity model on this task. Held-out test set fixed within "
        f"each split; training set grown from {NTRAIN[0]} to {NTRAIN[-1]} knockouts over {len(SEEDS)} seeds. "
        f"RESULT at alpha={best:g}: {curve[f'FULL a={best} n={NTRAIN[0]}']:.4f} at {NTRAIN[0]} knockouts rising to "
        f"{curve[f'FULL a={best} n={NTRAIN[-1]}']:.4f} at {NTRAIN[-1]}. "
        "THE CURVE MUST BE READ ON LOG-X, and getting this wrong reverses the conclusion: the raw final step from "
        f"{NTRAIN[-2]} to {NTRAIN[-1]} is only {curve[f'FULL a={best} n={NTRAIN[-1]}'] - curve[f'FULL a={best} n={NTRAIN[-2]}']:+.4f}, which "
        "looks like saturation, but it is a 0.23-doubling step while the first step was a full doubling. Normalised, recall grows "
        f"{slope:+.4f} per doubling of training knockouts with R^2={F['r2']:.3f}, and the across-seed SE on a single point is +-{sem:.4f}. "
        + (f"THE CURVE HAS NOT FLATTENED (slope {slope:+.4f} is more than 2x the {sem:.4f} noise), so the binding constraint is the NUMBER OF "
           "MEASURED KNOCKOUTS, not model capacity. Adding parameters would add variance in a regime that already prefers more regularisation "
           "(alpha=100 beats alpha=10) and where a zero-parameter 1-NN retrieval reaches 0.288 against the ridge's 0.341. The lever is more "
           f"knockouts -- a lower mover threshold takes 337 sources to 664, and the fit predicts that doubling is worth about {slope:+.3f}, "
           f"landing near {F['extrapolate_one_doubling']:.3f}. Extrapolation from six points, so treat it as a direction, not a promise. "
           if still_climbing else
           f"The fitted slope {slope:+.4f} is within twice the {sem:.4f} across-seed noise, so extra knockouts have stopped paying and richer "
           "features or more capacity become the sensible next move. ")
        + f"WHERE THE SHORTAGE IS: holding the program basis, mu, tide mask and vocabularies fixed at the full training half (REGR) and "
        f"subsampling only the ridge rows is worth {gaps[NTRAIN[0]]:+.3f} at {NTRAIN[0]} knockouts but only {gaps[basis_saturates]:+.3f} by "
        f"n={basis_saturates}. So below roughly {basis_saturates} knockouts the missing ingredient is the program basis itself -- there are not "
        "enough measured responses to define the response space -- and above it the limit shifts to the annotation-to-loading map, which is what "
        "better gene features (protein sequence embeddings) would attack. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"threshold": T, "n_train": list(NTRAIN), "alphas": list(ALPHAS), "seeds": list(SEEDS),
               "curve": {k: round(v, 4) for k, v in curve.items()}, "summary": outs, "fits": fits,
               "regr_minus_full": gaps, "basis_saturates_at": basis_saturates, "slope_per_doubling": slope, "se_at_largest": sem,
               "still_climbing": still_climbing, "verdict": verdict, "note": verdict},
              open(OUT / "coldstart_scaling.json", "w"), indent=1)
    print("\n  -> outputs/orphan/coldstart_scaling.json")


if __name__ == "__main__":
    main()
