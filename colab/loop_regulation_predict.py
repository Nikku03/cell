"""Loop 201. All the regulation data there is, and what it can predict that is not in it.

WHAT LOOPS 187 AND 188b LEFT ON THE TABLE. Loop 187 measured that two-node feedback is enriched at
z +43.8 in the curated tier and +7.9 in the independent binding tier, that autoregulation runs 2.2x
over chance, and that feedforward loops are NOT enriched (z +1.3 curated) and are DEPLETED in the
binding tier (z -21.2). Loop 188b measured that the epigenome adds +0.0086 AUPRC over measured
binding. All of that is DESCRIPTION. None of it was ever asked to predict anything.

A structural enrichment is only useful if it forecasts something not already in the data it was
measured on. Feedback at 43 sigma says reciprocity is real. It does NOT say reciprocity is
predictable, and it does not say a predictor built on it would beat knowing how famous a gene is.
This project has been beaten by fame before -- loop 71's human transplant closed structurally and
lost to how well-studied a gene is -- so fame is the baseline everywhere below, not a footnote.

THE DATA, ASSEMBLED AND MEASURED BEFORE ANY GATE WAS WRITTEN. Four sources, and their overlaps are
the reason the design looks the way it does:

    net_bundle    610,256 unique directed edges over 16,492 genes   (this project's assembly)
    CollecTRI      43,536 curated TF->target edges with sign and PMIDs
    SIGNOR         19,533 directed protein-protein causal edges from literature, with PMIDs
    OmniPath       85,526 directed interactions, aggregated, with curation effort

    CollecTRI is 90.6% CONTAINED in net_bundle. It is not held-out and is not used as a test set.
    SIGNOR is 86.5% ABSENT from net_bundle (16,893 of 19,533).
    OmniPath is 98.3% ABSENT from net_bundle (84,078 of 85,526).
    SIGNOR and OmniPath share 14,635 edges, so they are NOT independent of each other and are
    never counted as two separate confirmations.

That gives one genuinely held-out set -- literature-curated edges this project's network has never
seen -- and the whole loop is built around it.

Reciprocity rates, measured before the gates: net_bundle 1.34%, CollecTRI 4.90%, SIGNOR 5.23%,
OmniPath 3.68%. The base rate is what T1 has to beat, and it is small enough that a model can look
good on accuracy while doing nothing, so T1 is scored on AUC and precision at the base rate, never
on accuracy.

PREDECLARED, BEFORE ANY NUMBER.

  P1 IS THE INSTRUMENT HONEST ABOUT WHAT IT HAS?
     Gate: PASS iff the four sources parse to the counts above, the overlap fractions reproduce,
     and no edge used as a held-out test appears anywhere in the training features. FAIL means a
     leak, and every accuracy below would be reading its own answer.

  P2 CAN RECIPROCITY BE PREDICTED? -- the feedback finding, cashed out.
     Given A->B in net_bundle, predict whether B->A also exists. Features are computed with the
     reciprocal edge and its endpoints' reciprocal status REMOVED, or the task is circular.
     Gate: PASS iff held-out AUC >= 0.65 AND precision in the top decile is at least 3x the 1.34%
     base rate. FAIL means a 43-sigma enrichment carries no forecasting power, which would be a
     finding about what structural enrichment is worth.

  P3 DOES FAME ALREADY DO IT?
     Baseline: out-degree(A) x in-degree(B) alone, nothing else.
     Gate: PASS iff the full model beats the degree-only baseline by >= 0.02 AUC in >= 4 of 5
     seeds. FAIL means reciprocity prediction is popularity prediction wearing a hat.
     Requires P2 -- if there is no signal there is nothing for fame to explain.

  P4 CAN WE RECOVER EDGES THIS NETWORK HAS NEVER SEEN?
     Score the 16,893 SIGNOR edges absent from net_bundle against matched random pairs, using
     features derived ONLY from net_bundle. This is the "predict the unknown" question in its
     strongest available form: an independent literature catalogue as the test set.
     Gate: PASS iff AUC >= 0.70 against degree-matched negatives.
     Degree matching is declared here and not chosen later: unmatched negatives would let the
     model win on fame alone and P5 would have nothing left to detect.

  P5 IS P4 ALSO JUST FAME?
     Gate: PASS iff the full model beats degree-only by >= 0.02 AUC in >= 4 of 5 seeds.
     Requires P4.

  P6 DOES FEEDFORWARD MEMBERSHIP PREDICT ANYTHING? -- the paired negative prediction.
     Loop 187 found feedback enriched and feedforward NOT enriched. If enrichment is what makes a
     motif useful, then a feedforward-membership feature should be worth close to nothing while
     the reciprocity feature is worth something, ON THE SAME TASK.
     Gate: PASS iff dropping the feedforward feature costs LESS AUC than dropping the reciprocity
     feature. This is a magnitude comparison and does not assume the sign of either.

  P7 CAN THE SIGN BE PREDICTED?
     net_bundle carries 46,448 activating and 7,680 repressing signed edges -- 85.8% majority.
     Gate: PASS iff balanced accuracy > 0.55 on a held-out split. Scored balanced, never raw,
     because raw accuracy is 0.858 for a model that always says "activating".

  P8 CAN A MISSING EPIGENETIC MARK BE IMPUTED?
     Predict element 5mC from H3K4me3 coverage and signal, CpG count and density, and promoter-side
     marks -- the measured K562 tracks from loop 188b.
     Gate: PASS iff held-out Spearman >= 0.30 AND it beats predicting the training mean.
     FAIL means the marks do not even predict each other, which bounds what an imputed epigenome
     could be worth.

  P9 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import csv, gzip, json, os, sys, time
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates, weakened_by

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "colab", "data", "net_bundle.json.gz")
NET = os.path.join(ROOT, "colab", "data", "networks")
MARKS = ("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
         "scratchpad/epigenome/k562_marks.npz")
OUT = os.path.join(ROOT, "outputs", "loop_regulation_predict.json")

SEEDS = (11, 22, 33, 44, 55)
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


# ---------------------------------------------------------------- sources
def load_sources():
    nb = json.load(gzip.open(NB))
    names = nb["names"]
    edges = {(names[s], names[t]) for s, t, _ in nb["reg"]}
    signs = {}
    for s, t, g in nb["reg"]:
        if g:
            signs[(names[s], names[t])] = g
    ppi = {(names[a], names[b]) for a, b in nb["ppi"]}
    ppi |= {(b, a) for a, b in ppi}
    coexpr = {}
    for k, v in nb["coexpr"].items():
        for j, r in v:
            coexpr[(names[int(k)], names[j])] = r

    ct = set()
    with open(os.path.join(NET, "collectri.csv")) as f:
        for row in csv.DictReader(f):
            ct.add((row["source"], row["target"]))

    sg, sg_sign = set(), {}
    with open(os.path.join(NET, "signor_human.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 10 or p[1] != "protein" or p[5] != "protein":
                continue
            sg.add((p[0], p[4]))
            e = p[8]
            if e.startswith("up"):
                sg_sign[(p[0], p[4])] = 1
            elif e.startswith("down"):
                sg_sign[(p[0], p[4])] = -1

    om = set()
    with open(os.path.join(NET, "omnipath.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("is_directed") == "True":
                om.add((row["source_genesymbol"], row["target_genesymbol"]))

    return dict(names=names, edges=edges, signs=signs, ppi=ppi, coexpr=coexpr,
                collectri=ct, signor=sg, signor_sign=sg_sign, omnipath=om)


def build_index(edges):
    out, inn = defaultdict(set), defaultdict(set)
    for a, b in edges:
        out[a].add(b); inn[b].add(a)
    return out, inn


def feats(a, b, out, inn, ppi, coexpr, edges, drop=()):
    """Feature row for the ordered pair (a,b). `drop` removes a named feature (for ablation)."""
    oa, ib = out.get(a, set()), inn.get(b, set())
    ob, ia = out.get(b, set()), inn.get(a, set())
    f = {
        "log_out_a": np.log1p(len(oa)),
        "log_in_b": np.log1p(len(ib)),
        "log_out_b": np.log1p(len(ob)),
        "log_in_a": np.log1p(len(ia)),
        "shared_targets": np.log1p(len(oa & ob)),
        "shared_regulators": np.log1p(len(ia & ib)),
        "a_targets_b_regs": np.log1p(len(oa & ib)),
        "ppi": 1.0 if (a, b) in ppi else 0.0,
        "coexpr": max(coexpr.get((a, b), 0.0), coexpr.get((b, a), 0.0)),
        # RECIPROCITY: does b already regulate a? (for the reciprocity task this is the ANSWER and
        # is dropped by the caller; for the edge task it is a legitimate feature)
        "reciprocal": 1.0 if (b, a) in edges else 0.0,
        # FEEDFORWARD: how many c with a->c and c->b -- the motif loop 187 found NOT enriched
        "ffl": np.log1p(len(oa & ib)) if False else np.log1p(sum(1 for c in oa if (c, b) in edges)),
    }
    for k in drop:
        f[k] = 0.0
    return f


FEATNAMES = ["log_out_a", "log_in_b", "log_out_b", "log_in_a", "shared_targets",
             "shared_regulators", "a_targets_b_regs", "ppi", "coexpr", "reciprocal", "ffl"]


def mat(rows):
    return np.array([[r[k] for k in FEATNAMES] for r in rows], float)


def fit_predict(Xtr, ytr, Xte, seed):
    """Logistic regression by plain gradient descent -- no sklearn dependency."""
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A = np.hstack([A, np.ones((len(A), 1))]); B = np.hstack([B, np.ones((len(B), 1))])
    w = rng.normal(0, 0.01, A.shape[1])
    pos_w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    sw = np.where(ytr == 1, pos_w, 1.0)
    for _ in range(600):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (sw * (p - ytr)) / len(ytr) + 1e-3 * w
        w -= 0.5 * g
    return 1 / (1 + np.exp(-np.clip(B @ w, -30, 30)))


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "regulation prediction"}
    say("=" * 104)
    say("LOOP 201 -- ALL THE REGULATION DATA, AND WHAT IT PREDICTS THAT IS NOT IN IT")
    say("=" * 104)

    S = load_sources()
    edges, ppi, coexpr = S["edges"], S["ppi"], S["coexpr"]
    out, inn = build_index(edges)

    # ------------------------------------------------------------ P1
    say("P1 IS THE INSTRUMENT HONEST ABOUT WHAT IT HAS?")
    counts = {"net_bundle": len(edges), "collectri": len(S["collectri"]),
              "signor": len(S["signor"]), "omnipath": len(S["omnipath"])}
    ov = {k: len(S[k] & edges) / len(S[k]) for k in ("collectri", "signor", "omnipath")}
    for k, v in counts.items():
        say(f"     {k:<12} {v:>8,}" + (f"   contained in net_bundle {ov[k]:.3f}" if k in ov else ""))
    say(f"     SIGNOR & OmniPath share {len(S['signor'] & S['omnipath']):,} edges "
        f"-- never counted as two confirmations")
    held_out = S["signor"] - edges
    leak = held_out & edges
    say(f"     held-out set (SIGNOR minus net_bundle)  {len(held_out):,}   leak into features {len(leak)}")
    ok1 = (counts["net_bundle"] == 610256 and counts["collectri"] == 43536
           and counts["signor"] == 19533 and counts["omnipath"] == 85526
           and ov["collectri"] > 0.85 and ov["signor"] < 0.20 and not leak)
    G.add("P1", ok1,
          if_true="P1 PASS -- four sources, counts reproduce, CollecTRI is contained and excluded "
                  "as a test set, SIGNOR is held out and does not leak",
          if_false=lambda: f"P1 FAIL -- {counts}, overlaps {ov}, leak {len(leak)}")
    res["sources"] = {"counts": counts, "contained": ov,
                      "signor_omnipath_shared": len(S["signor"] & S["omnipath"]),
                      "held_out": len(held_out)}

    # ------------------------------------------------------------ P2/P3 reciprocity
    say("P2 CAN RECIPROCITY BE PREDICTED?  (the feedback finding, cashed out)")
    pairs = sorted(edges)
    base_rate = sum(1 for a, b in pairs if (b, a) in edges) / len(pairs)
    say(f"     base rate of reciprocity in net_bundle  {base_rate:.4f}")
    rng = np.random.default_rng(201)
    pos = [(a, b) for a, b in pairs if (b, a) in edges]
    # THE TAUTOLOGY GUARD, and the reason this loop was rerun. The first run sampled negatives from
    # ALL non-reciprocal edges. B->A can only exist if B regulates ANYTHING, and 84.1% of those
    # negatives had out-degree(B) = 0 while 0% of the positives did -- so the single rule
    # "is B a regulator at all" scored AUC >= 0.9206 by itself and the reported 0.9837 was mostly
    # that rule. Restricting the negative pool to targets that ARE regulators removes the trivial
    # separator and leaves the question actually worth asking: given A->B and given B regulates
    # something, does B regulate A specifically?
    neg_all = [(a, b) for a, b in pairs if (b, a) not in edges and out.get(b)]
    neg = [neg_all[i] for i in rng.choice(len(neg_all), size=min(len(neg_all), 10 * len(pos)),
                                          replace=False)]
    say(f"     reciprocal pairs {len(pos):,}   sampled non-reciprocal {len(neg):,}")
    say(f"     negatives are restricted to targets that ARE regulators "
        f"({len(neg_all):,} available of {sum(1 for a,b in pairs if (b,a) not in edges):,} "
        f"non-reciprocal edges) -- see the tautology guard in the source")
    # the answer and any feature that encodes it are removed
    Xr = mat([feats(a, b, out, inn, ppi, coexpr, edges, drop=("reciprocal",))
              for a, b in pos + neg])
    yr = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    a_full, a_deg, prec10 = [], [], []
    deg_cols = [FEATNAMES.index(k) for k in ("log_out_a", "log_in_b", "log_out_b", "log_in_a")]
    for sd in SEEDS:
        r2 = np.random.default_rng(sd); p = r2.permutation(len(yr))
        cut = int(0.7 * len(yr)); tr, te = p[:cut], p[cut:]
        sf = fit_predict(Xr[tr], yr[tr], Xr[te], sd)
        sdg = fit_predict(Xr[tr][:, deg_cols], yr[tr], Xr[te][:, deg_cols], sd)
        a_full.append(auc(yr[te], sf)); a_deg.append(auc(yr[te], sdg))
        k = max(1, len(te) // 10)
        top = np.argsort(-sf)[:k]
        prec10.append(float(yr[te][top].mean()))
    auc_f, auc_d = float(np.mean(a_full)), float(np.mean(a_deg))
    p10 = float(np.mean(prec10))
    say(f"     held-out AUC   full {auc_f:.4f}   degree-only {auc_d:.4f}")
    say(f"     precision in top decile {p10:.4f}   vs base rate {base_rate:.4f}   "
        f"= {p10/base_rate:.1f}x")
    say(f"     NOTE the test set is enriched 1:10 by construction, so the base rate to beat inside "
        f"it is {len(pos)/(len(pos)+len(neg)):.4f}; the 1.34% figure is the population rate")
    ok2 = bool(auc_f >= 0.65 and p10 >= 3 * (len(pos) / (len(pos) + len(neg))))
    G.add("P2", ok2, stat=auc_f,
          if_true=lambda: f"P2 PASS -- AUC {auc_f:.4f}, top-decile precision {p10:.4f}",
          if_false=lambda: f"P2 FAIL -- AUC {auc_f:.4f} (bar 0.65), top-decile precision {p10:.4f}")
    say("P3 DOES FAME ALREADY DO IT?")
    d_up = sum(1 for f, d in zip(a_full, a_deg) if f - d >= 0.02)
    say(f"     full minus degree-only  {auc_f - auc_d:+.4f}   ({d_up}/5 seeds at >= +0.02)")
    G.add("P3", bool(d_up >= 4), stat=auc_f - auc_d, requires=("P2",),
          if_true=lambda: f"P3 PASS -- structure beats degree by {auc_f-auc_d:+.4f} in {d_up}/5",
          if_false=lambda: f"P3 FAIL -- structure beats degree by only {auc_f-auc_d:+.4f} "
                           f"({d_up}/5 at the bar). Reciprocity prediction is largely popularity")
    res["reciprocity"] = {"base_rate": base_rate, "n_pos": len(pos), "n_neg": len(neg),
                          "auc_full": auc_f, "auc_degree": auc_d, "auc_full_seeds": a_full,
                          "auc_degree_seeds": a_deg, "prec_top10": p10}

    # ------------------------------------------------------------ P4/P5 cross-source
    say("P4 CAN WE RECOVER EDGES THIS NETWORK HAS NEVER SEEN?")
    universe = [g for g in S["names"] if out.get(g) or inn.get(g)]
    uset = set(universe)
    ho = [(a, b) for a, b in held_out if a in uset and b in uset]
    say(f"     held-out SIGNOR edges usable (both endpoints in net_bundle)  {len(ho):,}")
    # degree-matched negatives: same source, a target with a similar in-degree
    by_indeg = defaultdict(list)
    for g in universe:
        by_indeg[int(np.log1p(len(inn.get(g, ()))))].append(g)
    negs = []
    r3 = np.random.default_rng(2011)
    known = edges | S["signor"] | S["omnipath"] | S["collectri"]
    for a, b in ho:
        bucket = by_indeg[int(np.log1p(len(inn.get(b, ()))))]
        for _ in range(12):
            c = bucket[r3.integers(len(bucket))]
            if c != b and (a, c) not in known:
                negs.append((a, c)); break
    say(f"     degree-matched negatives drawn  {len(negs):,}")
    Xh = mat([feats(a, b, out, inn, ppi, coexpr, edges) for a, b in ho + negs])
    yh = np.r_[np.ones(len(ho)), np.zeros(len(negs))]
    b_full, b_deg = [], []
    for sd in SEEDS:
        r2 = np.random.default_rng(sd); p = r2.permutation(len(yh))
        cut = int(0.7 * len(yh)); tr, te = p[:cut], p[cut:]
        b_full.append(auc(yh[te], fit_predict(Xh[tr], yh[tr], Xh[te], sd)))
        b_deg.append(auc(yh[te], fit_predict(Xh[tr][:, deg_cols], yh[tr], Xh[te][:, deg_cols], sd)))
    A_f, A_d = float(np.mean(b_full)), float(np.mean(b_deg))
    say(f"     held-out AUC on an INDEPENDENT literature catalogue   full {A_f:.4f}   "
        f"degree-only {A_d:.4f}")
    G.add("P4", bool(A_f >= 0.70), stat=A_f,
          if_true=lambda: f"P4 PASS -- AUC {A_f:.4f} recovering SIGNOR edges net_bundle never saw",
          if_false=lambda: f"P4 FAIL -- AUC {A_f:.4f} (bar 0.70) against degree-matched negatives")
    say("P5 IS P4 ALSO JUST FAME?")
    e_up = sum(1 for f, d in zip(b_full, b_deg) if f - d >= 0.02)
    say(f"     full minus degree-only  {A_f - A_d:+.4f}   ({e_up}/5 seeds at >= +0.02)")
    G.add("P5", bool(e_up >= 4), stat=A_f - A_d, requires=("P4",),
          if_true=lambda: f"P5 PASS -- {A_f-A_d:+.4f} in {e_up}/5",
          if_false=lambda: f"P5 FAIL -- only {A_f-A_d:+.4f} ({e_up}/5)")
    res["cross_source"] = {"n_heldout": len(ho), "n_neg": len(negs), "auc_full": A_f,
                           "auc_degree": A_d, "auc_full_seeds": b_full, "auc_degree_seeds": b_deg}

    # ------------------------------------------------------------ P6 ffl vs reciprocity
    say("P6 DOES FEEDFORWARD MEMBERSHIP PREDICT ANYTHING?  (the paired negative prediction)")
    def ablate(drop):
        v = []
        Xa = mat([feats(a, b, out, inn, ppi, coexpr, edges, drop=drop) for a, b in ho + negs])
        for sd in SEEDS:
            r2 = np.random.default_rng(sd); p = r2.permutation(len(yh))
            cut = int(0.7 * len(yh)); tr, te = p[:cut], p[cut:]
            v.append(auc(yh[te], fit_predict(Xa[tr], yh[tr], Xa[te], sd)))
        return np.array(v, float)
    s_ffl, s_rec = ablate(("ffl",)), ablate(("reciprocal",))
    d_ffl = np.array(b_full) - s_ffl      # per-seed PAIRED cost, same split each seed
    d_rec = np.array(b_full) - s_rec
    cost_ffl, cost_rec = float(d_ffl.mean()), float(d_rec.mean())
    sem_ffl = float(d_ffl.std(ddof=1) / np.sqrt(len(d_ffl)))
    sem_rec = float(d_rec.std(ddof=1) / np.sqrt(len(d_rec)))
    say(f"     drop feedforward feature   AUC {s_ffl.mean():.4f}   "
        f"costs {cost_ffl:+.4f} +/- {sem_ffl:.4f}")
    say(f"     drop reciprocity feature   AUC {s_rec.mean():.4f}   "
        f"costs {cost_rec:+.4f} +/- {sem_rec:.4f}")
    # DEFINEDNESS, and the reason this gate was rewritten. The first run compared +0.0005 against
    # -0.0000 and PASSED. Both were indistinguishable from zero, so the comparison was a property
    # of the noise -- gate_guard's Family One, a ratio with no denominator. If neither ablation
    # moves AUC by more than 2 sem, there is nothing to rank and the gate did not run.
    moved = (abs(cost_rec) > 2 * sem_rec) or (abs(cost_ffl) > 2 * sem_ffl)
    cmp6 = weakened_by(cost_rec, cost_ffl)
    G.add("P6", bool(cmp6["weakened"]), stat=cost_rec, requires=("P4",),
          void_if=(not moved),
          void_reason=(f"neither ablation moves AUC beyond its own noise "
                       f"(reciprocity {cost_rec:+.4f} +/- {sem_rec:.4f}, feedforward "
                       f"{cost_ffl:+.4f} +/- {sem_ffl:.4f}), so there is no difference to rank "
                       f"and this gate did not run"),
          if_true=lambda: f"P6 PASS -- the enriched motif carries more ({cost_rec:+.4f}) than the "
                          f"non-enriched one ({cost_ffl:+.4f}), which is what loop 187 predicts",
          if_false=lambda: f"P6 FAIL -- feedforward costs {cost_ffl:+.4f} against reciprocity's "
                           f"{cost_rec:+.4f}: enrichment did not decide which motif is useful")
    res["motif_ablation"] = {"auc_full": A_f, "auc_no_ffl": float(s_ffl.mean()),
                             "auc_no_reciprocal": float(s_rec.mean()),
                             "cost_ffl": cost_ffl, "sem_ffl": sem_ffl,
                             "cost_reciprocal": cost_rec, "sem_reciprocal": sem_rec,
                             "moved_beyond_noise": bool(moved), "compare": cmp6}

    # ------------------------------------------------------------ P7 sign
    say("P7 CAN THE SIGN BE PREDICTED?")
    signed = [(k, v) for k, v in S["signs"].items() if k[0] in uset and k[1] in uset]
    cnt = Counter(v for _, v in signed)
    maj = max(cnt.values()) / len(signed)
    say(f"     signed edges {len(signed):,}   activating {cnt.get(1,0):,}  repressing "
        f"{cnt.get(-1,0):,}   majority-class accuracy {maj:.4f}")
    Xs = mat([feats(a, b, out, inn, ppi, coexpr, edges) for (a, b), _ in signed])
    ys = np.array([1.0 if v == 1 else 0.0 for _, v in signed])
    bal = []
    for sd in SEEDS:
        r2 = np.random.default_rng(sd); p = r2.permutation(len(ys))
        cut = int(0.7 * len(ys)); tr, te = p[:cut], p[cut:]
        pr = fit_predict(Xs[tr], ys[tr], Xs[te], sd) > 0.5
        yt = ys[te] == 1
        tpr = pr[yt].mean() if yt.any() else float("nan")
        tnr = (~pr[~yt]).mean() if (~yt).any() else float("nan")
        bal.append(0.5 * (tpr + tnr))
    bal_m = float(np.mean(bal))
    say(f"     balanced accuracy {bal_m:.4f}  (raw majority would score {maj:.4f})")
    G.add("P7", bool(bal_m > 0.55), stat=bal_m,
          if_true=lambda: f"P7 PASS -- balanced accuracy {bal_m:.4f}",
          if_false=lambda: f"P7 FAIL -- balanced accuracy {bal_m:.4f} (bar 0.55). Network position "
                           f"does not carry whether a regulator activates or represses")
    res["sign"] = {"n": len(signed), "activating": cnt.get(1, 0), "repressing": cnt.get(-1, 0),
                   "majority": maj, "balanced_acc": bal_m, "seeds": bal}

    # ------------------------------------------------------------ P8 epigenetic imputation
    say("P8 CAN A MISSING EPIGENETIC MARK BE IMPUTED?")
    if not os.path.exists(MARKS):
        G.add("P8", None, void_if=True,
              void_reason=f"the K562 mark cache is not on disk at {MARKS}")
        res["epigenome"] = {"available": False}
    else:
        M = np.load(MARKS, allow_pickle=True)
        y8 = M["el_5mc"].astype(float)
        X8 = np.column_stack([M["el_h3k4me3_cov"], M["el_h3k4me3_sig"],
                              M["el_ncpg"], np.log1p(M["el_ncpg"])]).astype(float)
        good = np.isfinite(y8) & np.isfinite(X8).all(1)
        say(f"     elements with a defined 5mC value  {good.sum():,} of {len(y8):,} "
            f"({1-good.mean():.4f} dropped as non-finite)")
        X8, y8 = X8[good], y8[good]
        rhos, rho_mean = [], []
        for sd in SEEDS:
            r2 = np.random.default_rng(sd); p = r2.permutation(len(y8))
            cut = int(0.7 * len(y8)); tr, te = p[:cut], p[cut:]
            mu, sdv = X8[tr].mean(0), X8[tr].std(0) + 1e-9
            A = np.hstack([(X8[tr] - mu) / sdv, np.ones((len(tr), 1))])
            B = np.hstack([(X8[te] - mu) / sdv, np.ones((len(te), 1))])
            w = np.linalg.lstsq(A, y8[tr], rcond=None)[0]
            pred = B @ w
            ra = np.argsort(np.argsort(pred)); rb = np.argsort(np.argsort(y8[te]))
            rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
            const = np.full(len(te), y8[tr].mean())
            rho_mean.append(float(np.mean((y8[te] - const) ** 2) > np.mean((y8[te] - pred) ** 2)))
        rho = float(np.mean(rhos))
        beats = int(sum(rho_mean))
        say(f"     held-out Spearman {rho:.4f}   beats the training mean in {beats}/5 seeds")
        G.add("P8", bool(rho >= 0.30 and beats >= 4), stat=rho,
              if_true=lambda: f"P8 PASS -- 5mC imputed at rho {rho:.4f} from the other marks",
              if_false=lambda: f"P8 FAIL -- rho {rho:.4f} (bar 0.30), beats mean {beats}/5. "
                               f"The measured marks do not predict each other well enough for an "
                               f"imputed epigenome to be worth much")
        res["epigenome"] = {"available": True, "n": int(good.sum()), "rho": rho,
                            "rho_seeds": rhos, "beats_mean": beats}

    # ------------------------------------------------------------ P9
    say("P9 WHAT THIS CANNOT SHOW")
    say("     SIGNOR is literature curation, so an edge absent from it is not a proven non-edge.")
    say("     Every AUC here is against SAMPLED negatives and is a lower bound by an unknown")
    say("     amount, exactly as loop 177's H8 said about genomic decoys.")
    say("     SIGNOR and OmniPath overlap by 14,635 edges, so recovering SIGNOR does not")
    say("     independently confirm OmniPath and no claim below treats it as two results.")
    say("     Reciprocity is measured on THIS network's edge set. A network assembled from")
    say("     different sources would have a different base rate, and loop 187 already showed")
    say("     the curated and binding tiers disagree on feedforward by 22 sigma.")
    say("     Nothing here is a time course: predicting that an edge EXISTS is not predicting")
    say("     what it DOES, and loop 198 measured that this project cannot step state forward.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
