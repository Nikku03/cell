"""TWO PIVOT TESTS: is there a formulation where the measured signal is actually large enough to matter?

WHY. The audit established two things that close the original framing. The benchmark is 92.3% saturated by a
fixed gene list -- the best possible perturbation-INDEPENDENT predictor scores 0.3792 and frequency reaches
0.3502 of it -- so almost all the available score requires no biology. And the network's signal, while real
(2-4x frequency-matched lift), sits on a base rate near 1%, so it cannot fill 50 slots out of 8,246 and a
network-only predictor scores 0.0246 against frequency's 0.3502.

Neither of those says the network is uninformative. They say the OPERATING POINT is wrong. A 2-4x enrichment
is worthless when you must name 50 genes from 8,246 and a fixed list already gets most of the credit. It is
not obviously worthless when the question is smaller. So: two tests, at two smaller questions.

TEST 1 -- SHORTLIST. Given a handful of candidate genes, can the network rank a knockout's true movers above
decoys? This is what an experimentalist actually faces: not "name 50 from 8,246" but "I have 20 hypotheses,
order them".

    The decoys decide whether this test means anything. Drawn at random, the frequency prior wins on its own
    and the test measures nothing new -- frequently-moved genes are frequently movers. So the primary regime
    matches each decoy to the true mover on the EXACT number of training perturbations it moves. That forces
    frequency to AUC 0.5000 by construction, which is both the point and the built-in check: any AUC above 0.5
    is then knowledge of WHICH gene was perturbed, not of which genes move often. This is CELLFORMER2.md's
    Metric 2 idea applied without training anything. Two coarser matchings were tried first and BOTH failed
    that check (frequency 0.7316 and 0.6104); reporting either would have credited the prior to the network.

TEST 2 -- RESPONSIVE OR SILENT. Forget which genes move: does this knockout do ANYTHING measurable? Only 838
of 5,120 perturbations clear >=5 specific movers. Predicting that is directly useful for screen design, and
it is a different question from the one that failed.

    CIRCULARITY IS THE WHOLE RISK HERE. Any feature derived from the query perturbation's own response row
    trivially predicts its own response. Every feature used is a property of the KNOCKED-OUT GENE measured
    independently of this screen -- DepMap dependency, LOEUF constraint, network degree, pathway membership,
    literature count -- and the split is the same md5 split as everywhere else. Shuffled labels are run as a
    floor, and the literature-count feature is reported separately because "well-studied" is popularity, not
    biology, and would otherwise be quietly credited as a discovery.
"""
import collections
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TAU, TIDE_FRAC, MIN_SPEC = 1.0, 0.05, 5
RNG = np.random.default_rng(0)
R = {}


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def bci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = r.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auc_of(pos, neg):
    """P(a random positive scores above a random negative), ties counted as half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    rank = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(allv)
    s = allv[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            rank[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    rp = rank[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ============================================================= TEST 1: SHORTLIST
def test1(d, spec, nspec, tide, tr, test, freq):
    hdr("TEST 1 -- SHORTLIST: rank a knockout's movers above decoys")
    gi = d["gi"]
    kos = d["kos"]
    genes = d["genes"]
    adj = [collections.defaultdict(set) for _ in range(5)]
    for (i, j), v in d["pair"].items():
        for c in range(5):
            if v[c]:
                adj[c][i].add(j)
                adj[c][j].add(i)

    # EXACT frequency matching, reached after TWO weaker controls failed their own sanity check.
    # 20 quantile bins    -> frequency scored AUC 0.7316 against its "matched" decoys
    # +-60 rank window    -> frequency still scored 0.6104
    # A correctly matched control forces frequency to 0.5000, because every shortlist entry then has the same
    # movement frequency. Bins were too coarse for a steep distribution; the rank window is asymmetric for the
    # most-moved genes, since nothing sits above them, so their decoys were drawn systematically lower.
    # Matching on the exact INTEGER count of training perturbations a gene moves removes the last degree of
    # freedom, and the printed frequency AUC is the check that it worked. Without that check the first version
    # would have reported the frequency prior as network specificity.
    moved = spec.any(0)
    movedidx = np.array([g for g in range(len(genes)) if moved[g] and not tide[g]])
    cnt = spec[tr].sum(0)                                 # integer, not a rate: exact match is possible
    by_cnt = collections.defaultdict(list)
    for g in movedidx:
        by_cnt[int(cnt[g])].append(int(g))
    by_cnt = {k: np.array(v) for k, v in by_cnt.items()}

    N_DECOY = int(os.environ.get("PIVOT_DECOYS", "19"))
    REPS = int(os.environ.get("PIVOT_REPS", "20"))

    def netscore(ki, cols):
        s = np.zeros(len(cols), float)
        if ki is None:
            return s
        for c in range(5):
            nb_ = adj[c].get(ki, ())
            if nb_:
                s += np.fromiter((1.0 if x in nb_ else 0.0 for x in cols), float, len(cols))
        return s

    for regime in ("frequency-MATCHED decoys", "random decoys"):
        aucs_net, aucs_freq, top1, ties = [], [], [], []
        skipped = 0
        for q in test:
            ki = gi.get(kos[q])
            if ki is None:
                continue
            truth = np.where(spec[q])[0]
            truth = truth[~tide[truth]]
            if len(truth) < 2:
                continue
            for _ in range(REPS):
                t = int(RNG.choice(truth))
                if regime.startswith("frequency"):
                    cand = by_cnt.get(int(cnt[t]))
                    if cand is None:
                        skipped += 1
                        continue
                    dec = [int(x) for x in cand if x != t and not spec[q, x]]
                    if len(dec) > N_DECOY:
                        dec = [int(x) for x in RNG.choice(dec, size=N_DECOY, replace=False)]
                    if len(dec) < 5:
                        skipped += 1          # too few genes share this exact frequency to match against
                        continue
                else:
                    dec = [int(x) for x in RNG.choice(movedidx, size=N_DECOY, replace=False)
                           if not spec[q, x]]
                    if len(dec) < 5:
                        continue
                cols = [t] + dec
                sn = netscore(ki, cols)
                sf = freq[cols]
                aucs_net.append(auc_of(sn[:1], sn[1:]))
                aucs_freq.append(auc_of(sf[:1], sf[1:]))
                # TIES COUNT AS PARTIAL CREDIT. Most shortlist entries have network score 0 -- no edge to the
                # knocked-out gene at all -- so a strict `>` scored an all-zero shortlist as a miss and drove
                # top-1 BELOW chance, which looks like anti-signal and is really just an unbroken tie.
                mx = sn[1:].max()
                if sn[0] > mx:
                    top1.append(1.0)
                elif sn[0] == mx:
                    top1.append(1.0 / (1 + int((sn[1:] == mx).sum())))
                else:
                    top1.append(0.0)
                ties.append(1.0 if (sn == sn[0]).all() else 0.0)
        an, af, t1, tz = (np.array(aucs_net), np.array(aucs_freq), np.array(top1), np.array(ties))
        lo, hi = bci(an)
        flo, fhi = bci(af)
        print(f"\n  {regime}  ({len(an):,} shortlists of 1 mover + up to {N_DECOY} decoys)")
        print(f"    network AUC    {an.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]   (0.5 = no information)")
        print(f"    frequency AUC  {af.mean():.4f}  95% CI [{flo:.4f},{fhi:.4f}]")
        print(f"    network ranks the true mover FIRST in {100*t1.mean():.1f}% of shortlists "
              f"(chance {100/(N_DECOY+1):.1f}%; ties split)")
        print(f"    shortlists where EVERY entry scores 0 (no edge to the KO gene): {100*tz.mean():.1f}%")
        if skipped:
            print(f"    skipped {skipped:,} draws with too few exact-frequency matches "
                  f"({100*skipped/(skipped+len(an)):.1f}%) -- these are the most-moved genes")
        if regime.startswith("frequency") and not (0.45 < af.mean() < 0.55):
            print(f"    *** CONTROL FAILED: matched decoys must force frequency to ~0.50, got {af.mean():.4f}."
                  f" The network AUC above does NOT isolate specificity. ***")
        R.setdefault("test1", {})[regime] = {
            "n": int(len(an)), "network_auc": float(an.mean()), "network_ci": [lo, hi],
            "frequency_auc": float(af.mean()), "frequency_ci": [flo, fhi], "top1": float(t1.mean()),
            "all_zero_frac": float(tz.mean())}


# ============================================================= TEST 2: RESPONSIVE OR SILENT
def test2(d, spec, nspec, tr):
    hdr("TEST 2 -- RESPONSIVE OR SILENT: does this knockout do anything measurable?")
    gi, kos, genes = d["gi"], d["kos"], d["genes"]
    C = json.load(gzip.open(Path(__file__).resolve().parent / "data" / "cell_complete.json.gz", "rt"))
    meta = {g["name"]: g for g in C["genes"]}
    print(f"cell_complete: {len(meta):,} genes with annotation")

    deg = collections.Counter()
    degc = [collections.Counter() for _ in range(5)]
    for (i, j), v in d["pair"].items():
        for c in range(5):
            if v[c]:
                degc[c][i] += 1
                degc[c][j] += 1
                deg[i] += 1
                deg[j] += 1

    NAMES = ["dep_frac", "ess", "loeuf", "ppi_ann", "npath", "ndis", "tf", "cpg", "enh", "dark",
             "deg_total", "deg_reaction", "deg_complex", "deg_ppi", "deg_coexpr", "deg_reg", "pubs"]

    X, y, keep = [], [], []
    for p, k in enumerate(kos):
        m = meta.get(k)
        if m is None:
            continue
        ki = gi.get(k)
        row = [
            float(m.get("dep_frac") or 0.0), float(m.get("ess") or 0), float(m.get("loeuf") or 2.0),
            float(m.get("ppi") or 0), float(m.get("npath") or 0), float(m.get("ndis") or 0),
            float(m.get("tf") or 0), float(m.get("cpg") or 0), float(m.get("enh") or 0),
            float(m.get("dark") or 0),
            float(deg.get(ki, 0)) if ki is not None else 0.0,
        ] + [float(degc[c].get(ki, 0)) if ki is not None else 0.0 for c in range(5)] + [
            float(m.get("pubs") or 0)]
        X.append(row)
        y.append(1 if nspec[p] >= MIN_SPEC else 0)
        keep.append(p)
    X, y, keep = np.array(X, float), np.array(y, int), np.array(keep)
    trm = tr[keep]
    print(f"usable perturbations {len(y):,}  (KO gene has annotation); positives {y.sum():,} "
          f"({100*y.mean():.1f}%)")
    print(f"train {int(trm.sum()):,} / test {int((~trm).sum()):,}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def run(cols, tag, shuffle=False):
        Xi = X[:, cols]
        ytr = y[trm].copy()
        if shuffle:
            ytr = np.random.default_rng(1).permutation(ytr)
        mdl = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        mdl.fit(Xi[trm], ytr)
        s = mdl.predict_proba(Xi[~trm])[:, 1]
        yt = y[~trm]
        a = auc_of(s[yt == 1], s[yt == 0])
        # bootstrap the AUC itself
        rr = np.random.default_rng(0)
        bs = []
        pi, ni = np.where(yt == 1)[0], np.where(yt == 0)[0]
        for _ in range(400):
            bs.append(auc_of(s[rr.choice(pi, len(pi))], s[rr.choice(ni, len(ni))]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"    {tag:38s} AUC {a:.4f}  95% CI [{lo:.4f},{hi:.4f}]")
        return {"tag": tag, "auc": float(a), "ci": [float(lo), float(hi)]}

    # PAIRED comparison of feature sets on the SAME held-out rows. Comparing two AUCs by whether their
    # separate CIs overlap is the wrong test and is too conservative -- the two models score the same
    # perturbations, so the difference has far less variance than either estimate.
    def paired(colsA, colsB, tagA, tagB):
        def sc(cols):
            m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[trm][:, cols], y[trm])
            return m.predict_proba(X[~trm][:, cols])[:, 1]
        sA, sB, yt = sc(colsA), sc(colsB), y[~trm]
        rr = np.random.default_rng(0)
        pi, ni = np.where(yt == 1)[0], np.where(yt == 0)[0]
        ds = []
        for _ in range(600):
            bp, bn = rr.choice(pi, len(pi)), rr.choice(ni, len(ni))
            ds.append(auc_of(sA[bp], sA[bn]) - auc_of(sB[bp], sB[bn]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        d = auc_of(sA[yt == 1], sA[yt == 0]) - auc_of(sB[yt == 1], sB[yt == 0])
        print(f"    {tagA} MINUS {tagB}: {d:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{'-> real gain' if lo > 0 else '-> not distinguishable from zero'}")
        return {"a": tagA, "b": tagB, "delta": float(d), "ci": [float(lo), float(hi)]}

    allc = list(range(len(NAMES)))
    nopubs = [i for i in allc if NAMES[i] != "pubs"]
    netc = [i for i, n in enumerate(NAMES) if n.startswith("deg_")]
    depc = [i for i, n in enumerate(NAMES) if n in ("dep_frac", "ess")]
    print("\n  held-out AUC (positive = >=5 specific movers):")
    res = [run(allc, "all features"),
           run(nopubs, "all EXCEPT literature count"),
           run(depc, "DepMap essentiality only"),
           run(netc, "network degree only"),
           run([i for i, n in enumerate(NAMES) if n == "loeuf"], "LOEUF constraint only"),
           run(nopubs, "SHUFFLED labels (floor)", shuffle=True)]
    print("\n  does the network add anything on top of DepMap essentiality?")
    R["test2"] = {"n": int(len(y)), "positives": int(y.sum()), "results": res,
                  "paired": [paired(nopubs, depc, "all-except-pubs", "DepMap-only"),
                             paired(depc + netc, depc, "DepMap+network", "DepMap-only")]}

    mdl = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[trm][:, nopubs], y[trm])
    s = mdl.predict_proba(X[~trm][:, nopubs])[:, 1]
    yt = y[~trm]
    print("\n  screen-design view -- if you could only assay the top K% of candidates:")
    for frac in (0.05, 0.10, 0.20, 0.50):
        k = max(int(len(s) * frac), 1)
        idx = np.argsort(-s)[:k]
        print(f"    top {100*frac:>4.0f}%  ({k:>4} perturbations)  hit rate {100*yt[idx].mean():>5.1f}%  "
              f"vs base rate {100*yt.mean():.1f}%   enrichment {yt[idx].mean()/yt.mean():.2f}x")
        R["test2"].setdefault("screen", []).append(
            {"frac": frac, "k": int(k), "hit_rate": float(yt[idx].mean()),
             "enrichment": float(yt[idx].mean() / yt.mean())})


def main():
    import cellformer_af as CF
    d = CF.load_all()
    spec, nspec, tide = d["spec"], d["nspec"], d["tide"]
    kos = d["kos"]
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    test = [i for i in range(len(kos)) if not tr[i] and nspec[i] >= MIN_SPEC]
    freq = spec[tr].mean(0)
    print(f"\nsplit {int(tr.sum()):,}/{int((~tr).sum()):,}; held-out scorable {len(test):,}")
    test1(d, spec, nspec, tide, tr, test, freq)
    test2(d, spec, nspec, tr)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "pivot_tests.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'pivot_tests.json'}")


if __name__ == "__main__":
    main()
