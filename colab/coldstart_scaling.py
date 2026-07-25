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
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
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
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
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

    # is the curve still climbing at the largest training set?
    outs = {}
    for tag in ("FULL", "REGR"):
        for a in ALPHAS:
            last = curve[f"{tag} a={a} n={NTRAIN[-1]}"]; prev = curve[f"{tag} a={a} n={NTRAIN[-2]}"]
            first = curve[f"{tag} a={a} n={NTRAIN[0]}"]
            outs[f"{tag} a={a}"] = {"final": round(last, 4), "last_step": round(last - prev, 4),
                                    "total_gain": round(last - first, 4),
                                    "frac_of_gain_in_last_step": round((last - prev) / max(last - first, 1e-9), 3)}
            print(f"  {tag} a={a:g}: {first:.4f} -> {last:.4f}; last step ({NTRAIN[-2]}->{NTRAIN[-1]}) "
                  f"{last-prev:+.4f} = {(last-prev)/max(last-first,1e-9)*100:.0f}% of the total gain")
    best = max(ALPHAS, key=lambda a: curve[f"FULL a={a} n={NTRAIN[-1]}"])
    step = curve[f"FULL a={best} n={NTRAIN[-1]}"] - curve[f"FULL a={best} n={NTRAIN[-2]}"]
    still_climbing = bool(step > 0.005)

    verdict = (
        "COLD-START SCALING (coldstart_scaling.py): a learning curve to decide whether the cold-start model is limited by DATA or by MODEL "
        "CAPACITY -- i.e. whether it is worth training a transformer or other high-capacity model on this task. Held-out test set fixed within "
        f"each split; training set grown from {NTRAIN[0]} to {NTRAIN[-1]} knockouts over {len(SEEDS)} seeds. "
        f"FULL curve (everything rebuilt from the subset: programs, mu, tide, vocabularies, ridge) at alpha={best:g} runs "
        f"{curve[f'FULL a={best} n={NTRAIN[0]}']:.4f} -> {curve[f'FULL a={best} n={NTRAIN[-1]}']:.4f}, and the final step from "
        f"{NTRAIN[-2]} to {NTRAIN[-1]} knockouts is still worth {step:+.4f}. "
        + ("THE CURVE HAS NOT FLATTENED: recall is still rising at the largest training set available, so the binding constraint is the number of "
           "measured knockouts, not model capacity. Adding parameters would increase variance in a regime that already prefers MORE "
           "regularisation (alpha=100 beats alpha=10) and where a zero-parameter 1-NN retrieval reaches 0.288 against the ridge's 0.341. "
           "More knockouts -- lower mover threshold, other cell lines -- is the lever; a bigger model is not. "
           if still_climbing else
           "THE CURVE HAS FLATTENED: extra training knockouts have stopped paying, so the limitation is no longer sample size and additional "
           "capacity or richer features become the sensible next move. ")
        + "The REGRESSOR curve holds the program basis, mu, tide mask and vocabularies fixed at the full training half and subsamples only the "
        "ridge's rows, which localises the shortage: a REGRESSOR curve that is much flatter than FULL means the missing ingredient is the program "
        "basis (more knockouts to define the response space) rather than the annotation-to-loading map. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"threshold": T, "n_train": list(NTRAIN), "alphas": list(ALPHAS), "seeds": list(SEEDS),
               "curve": {k: round(v, 4) for k, v in curve.items()}, "summary": outs,
               "still_climbing": still_climbing, "verdict": verdict, "note": verdict},
              open(OUT / "coldstart_scaling.json", "w"), indent=1)
    print("\n  -> outputs/orphan/coldstart_scaling.json")


if __name__ == "__main__":
    main()
