"""coldstart_diagnose -- what is the 0.35 actually made of, and how much of the missing 0.65 is even reachable?

recall@50 on a truth set binarised at |z|>=4.2 is the ONLY thing this project has ever measured about the cold-start model. That single
number cannot distinguish between very different failure modes, and until they are separated there is no way to know which fix is worth
building. This script separates them.

  A  CONTINUOUS AGREEMENT. The model is scored on a hard top-50 cut but it predicts a real-valued profile. If predicted and true profiles
     correlate well, the shape is right and the loss is at the cut -- a ranking/calibration problem. If they barely correlate, the shape
     is wrong and the model is winning its 0.35 on a handful of easy genes. Also scored with the SIGN, which the recall metric throws
     away entirely: nothing so far has ever checked whether the model knows a gene goes UP rather than DOWN.
  B  WHERE THE MISSES LAND. For every true mover the model failed to put in its top 50, what rank did it get? Near-misses at rank 51-200
     mean a better cut or a calibration fix pays; misses at rank 5,000 mean the model has no idea about those genes. These are the same
     recall number and completely different problems.
  C  IS BREADTH PREDICTABLE? Knockouts move between a handful and 621 genes. The reconstruction is mu + P_hat @ G and its overall scale is
     free to vary, so the model implicitly predicts a breadth. Does that correlate with the truth? If not, every knockout is being given
     roughly the average-sized response, which would systematically hurt both the broadest and the narrowest.
  D  WHERE IT WORKS AND WHERE IT FAILS. The headline is an average over knockouts that are not alike. Stratify by truth-set size, complex
     membership, essentiality and annotated process. A model that is excellent on complex members and useless elsewhere is a different
     object from one that is uniformly mediocre, and only the first suggests an obvious next move.
  E  HOW MUCH OF THE TARGET IS NOISE? Genes near the |z|=4.2 boundary are close to coin flips. Re-deriving the truth set at 3.8 and 4.6
     and measuring the churn puts a floor under how much of the missing recall is unreachable rather than unlearned. This is not a
     replicate estimate -- there are no duplicate guides in the data -- but threshold churn is a lower bound on target instability.

Model, splits, tide mask and controls are identical to program_coldstart.py so the numbers are comparable. Deterministic."""
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
RIDGE = 10.0
SEEDS = (0, 1, 2)
T_ALT = (3.8, 4.6)


def main():
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr, spearmanr
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
    print(f"K562: |z|>={T:.1f} | {len(sources)} sources | diagnosing what the recall is made of")

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
        v_comp = vocab(lambda s: comps.get(s, ()), 2); v_dom = vocab(lambda s: doms.get(s, ()), 3)
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
            x.append(np.log1p(float(ctrl_expr.get(s, 0.0)))); x.append(1.0 if s in ctrl_expr else 0.0)
            x += [np.log1p(bp[0]), bp[1], bp[2], bp[3], 1.0] if bp else [0.0] * 5
            x += [1.0 if (gi.get("proc") or "?") == p else 0.0 for p in v_proc]
            x += [1.0 if (gi.get("comp") or "?") == p else 0.0 for p in v_loc]
            dl = doms.get(s, set()); x += [1.0 if d in dl else 0.0 for d in v_dom]
            cl = comps.get(s, set()); x += [1.0 if c in cl else 0.0 for c in v_comp]
            return x
        return np.array([feats(s) for s in sources], float)

    rec = collections.defaultdict(list)     # per-observation diagnostics
    strat = collections.defaultdict(list)
    churn = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        ss = sorted(sources); rng.shuffle(ss)
        cut = int(0.7 * len(ss)); train, test = ss[:cut], ss[cut:]
        tr_rows = [kidx[s] for s in train]
        tide = (Aall[tr_rows] >= T).mean(0) >= TIDE_FRAC
        keep = np.where(~tide)[0]
        Mtr = Z[tr_rows][:, keep].astype(np.float64)
        mu = Mtr.mean(0); _, _, Vt = np.linalg.svd(Mtr - mu, full_matrices=False)
        Gk = Vt[:KPROG]; Ptr = (Mtr - mu) @ Gk.T
        X = build_X(train)
        Xtr = X[[si_of[s] for s in train]]; Xte = X[[si_of[s] for s in test]]
        m = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
        Pte = Ridge(alpha=RIDGE).fit((Xtr - m) / sd, Ptr).predict((Xte - m) / sd)

        for j, s in enumerate(test):
            r = kidx[s]
            zt = Z[r][keep].astype(np.float64)                      # true signed profile on the candidate set
            pv = mu + Pte[j] @ Gk                                    # predicted signed profile
            # the knocked-out gene's own transcript is excluded from its truth set (it drops by construction, not by regulation)
            tidxs = np.array([q for q in np.where(np.abs(zt) >= T)[0] if genes[keep[q]] != s])
            if len(tidxs) == 0:
                continue
            order = np.argsort(-np.abs(pv))
            rank_of = np.empty(len(pv), dtype=np.int64); rank_of[order] = np.arange(len(pv))
            hits = int((rank_of[tidxs] < K_RECALL).sum())
            rc = hits / min(len(tidxs), K_RECALL)
            rec["recall"].append(rc)

            # A -- continuous agreement, on all candidates and on the true movers only, magnitude and SIGN
            rec["r_all"].append(float(pearsonr(np.abs(pv), np.abs(zt))[0]))
            rec["rho_all"].append(float(spearmanr(np.abs(pv), np.abs(zt))[0]))
            rec["r_signed"].append(float(pearsonr(pv, zt)[0]))
            rec["sign_acc_movers"].append(float((np.sign(pv[tidxs]) == np.sign(zt[tidxs])).mean()))
            top = order[:K_RECALL]
            got = np.intersect1d(top, tidxs)
            if len(got):
                rec["sign_acc_hits"].append(float((np.sign(pv[got]) == np.sign(zt[got])).mean()))

            # B -- where the misses land
            rk = rank_of[tidxs]
            rec["median_rank_of_true_movers"].append(float(np.median(rk)))
            missed = rk[rk >= K_RECALL]
            if len(missed):
                rec["median_rank_of_missed"].append(float(np.median(missed)))
                rec["frac_missed_in_top200"].append(float((missed < 200).mean()))
                rec["frac_missed_in_top500"].append(float((missed < 500).mean()))
                rec["frac_missed_beyond_2000"].append(float((missed >= 2000).mean()))

            # C -- breadth: does predicted response scale track the true number of movers?
            rec["_pred_scale"].append(float(np.linalg.norm(Pte[j])))
            rec["_true_breadth"].append(float(len(tidxs)))

            # D -- stratifiers
            gi = info.get(s, {}) or {}
            strat["by_breadth"].append((len(tidxs), rc))
            strat["by_complex"].append((1 if comps.get(s) else 0, rc))
            try:
                strat["by_ess"].append((1 if float(gi.get("ess") or 0) > 0.5 else 0, rc))
            except (TypeError, ValueError):
                pass
            strat["by_proc"].append(((gi.get("proc") or "?"), rc))

            # E -- how unstable is the target itself?
            for ta in T_ALT:
                alt = {genes[keep[q]] for q in np.where(np.abs(zt) >= ta)[0] if genes[keep[q]] != s}
                base = {genes[keep[q]] for q in tidxs}
                if base or alt:
                    churn.append((ta, len(base & alt) / max(len(base | alt), 1)))

    n = len(rec["recall"])
    print(f"\n  n = {n} held-out knockout evaluations ({len(SEEDS)} seeds)")
    print(f"  headline recall@50                       {np.mean(rec['recall']):.4f}")

    print("\n  A. CONTINUOUS AGREEMENT  (recall throws away everything except the top-50 membership)")
    print(f"     Pearson r, |predicted| vs |true|      {np.mean(rec['r_all']):+.4f}")
    print(f"     Spearman rho, |predicted| vs |true|   {np.mean(rec['rho_all']):+.4f}")
    print(f"     Pearson r, SIGNED profiles            {np.mean(rec['r_signed']):+.4f}")
    print(f"     sign accuracy on true movers          {np.mean(rec['sign_acc_movers']):.4f}   (0.5 = coin flip)")
    print(f"     sign accuracy on the ones it caught   {np.mean(rec['sign_acc_hits']):.4f}")

    print("\n  B. WHERE THE MISSES LAND  (out of ~8,500 candidates)")
    print(f"     median rank of a true mover           {np.mean(rec['median_rank_of_true_movers']):.0f}")
    print(f"     median rank of a MISSED true mover    {np.mean(rec['median_rank_of_missed']):.0f}")
    print(f"     missed movers that sit in top 200     {np.mean(rec['frac_missed_in_top200']):.3f}")
    print(f"     missed movers that sit in top 500     {np.mean(rec['frac_missed_in_top500']):.3f}")
    print(f"     missed movers beyond rank 2000        {np.mean(rec['frac_missed_beyond_2000']):.3f}")

    ps = np.array(rec["_pred_scale"]); tb = np.array(rec["_true_breadth"])
    r_b = float(spearmanr(ps, tb)[0])
    print("\n  C. IS BREADTH PREDICTABLE?")
    print(f"     Spearman(predicted response scale, true number of movers) = {r_b:+.4f}")

    print("\n  D. WHERE IT WORKS")
    bb = sorted(strat["by_breadth"])
    q = len(bb) // 4
    for i, lbl in enumerate(["narrowest 25%", "2nd quartile", "3rd quartile", "broadest 25%"]):
        chunk = bb[i * q:(i + 1) * q] if i < 3 else bb[3 * q:]
        print(f"     truth size {lbl:>14s} (n={len(chunk):3d}, median {int(np.median([c[0] for c in chunk])):3d} movers): "
              f"recall {np.mean([c[1] for c in chunk]):.4f}")
    for key, lbls in (("by_complex", {1: "in a curated complex", 0: "not in any complex"}),
                      ("by_ess", {1: "essential", 0: "non-essential"})):
        for v, lbl in lbls.items():
            vals = [c[1] for c in strat[key] if c[0] == v]
            if vals:
                print(f"     {lbl:>28s} (n={len(vals):3d}): recall {np.mean(vals):.4f}")
    pr = collections.defaultdict(list)
    for p, v in strat["by_proc"]:
        pr[p].append(v)
    for p, vals in sorted(pr.items(), key=lambda kv: -np.mean(kv[1])):
        if len(vals) >= 15:
            print(f"     process {p[:24]:>24s} (n={len(vals):3d}): recall {np.mean(vals):.4f}")

    print("\n  E. HOW STABLE IS THE TARGET ITSELF?")
    ch = collections.defaultdict(list)
    for ta, v in churn:
        ch[ta].append(v)
    for ta in sorted(ch):
        print(f"     Jaccard(truth at |z|>={T:.1f}, truth at |z|>={ta:.1f}) = {np.mean(ch[ta]):.3f}")
    stab = float(np.mean([np.mean(ch[ta]) for ta in ch]))

    verdict = (
        f"COLD-START DIAGNOSIS (coldstart_diagnose.py): decomposing the {np.mean(rec['recall']):.3f} recall@50 into the failure modes it "
        "conflates, because a single binarised top-50 number cannot tell a ranking problem from a comprehension problem and the two need "
        "different fixes. "
        f"A) CONTINUOUS AGREEMENT: |predicted| vs |true| profile correlates r={np.mean(rec['r_all']):+.3f} (Spearman "
        f"{np.mean(rec['rho_all']):+.3f}), the SIGNED correlation is r={np.mean(rec['r_signed']):+.3f}, and on the genes that really move "
        f"the model gets the DIRECTION right {np.mean(rec['sign_acc_movers']):.1%} of the time against a 50% coin flip -- a property the "
        "recall metric never tested. "
        f"B) WHERE THE MISSES LAND: the median true mover sits at rank {np.mean(rec['median_rank_of_true_movers']):.0f} of ~8,500, and a "
        f"MISSED mover at rank {np.mean(rec['median_rank_of_missed']):.0f}; {np.mean(rec['frac_missed_in_top200']):.0%} of misses are still "
        f"inside the top 200 and {np.mean(rec['frac_missed_beyond_2000']):.0%} are beyond rank 2,000. "
        f"C) BREADTH: Spearman between predicted response scale and true mover count is {r_b:+.3f}. "
        f"D) The headline hides real heterogeneity across knockouts -- see the stratification table. "
        f"E) TARGET STABILITY: moving the threshold by only +-0.4 changes the truth set enough to give a mean Jaccard of {stab:.2f} against "
        "the truth set actually used, which is a LOWER bound on how much of the missing recall is unreachable target noise rather than "
        "unlearned signal. It is not a replicate estimate; there are no duplicate guides in this data. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"threshold": T, "n_eval": n, "recall": round(float(np.mean(rec["recall"])), 4),
               "continuous": {k: round(float(np.mean(rec[k])), 4) for k in
                              ("r_all", "rho_all", "r_signed", "sign_acc_movers", "sign_acc_hits")},
               "misses": {k: round(float(np.mean(rec[k])), 4) for k in
                          ("median_rank_of_true_movers", "median_rank_of_missed", "frac_missed_in_top200",
                           "frac_missed_in_top500", "frac_missed_beyond_2000")},
               "breadth_spearman": round(r_b, 4), "target_jaccard_under_threshold_shift": round(stab, 4),
               "verdict": verdict, "note": verdict}, open(OUT / "coldstart_diagnose.json", "w"), indent=1)
    print("\n  -> outputs/orphan/coldstart_diagnose.json")


if __name__ == "__main__":
    main()
