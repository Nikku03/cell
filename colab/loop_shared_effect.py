"""LOOP 50 -- DO DRUGS THAT HIT THE SAME PART OF THE CELL SHARE SIDE EFFECTS?

The side-effect row is one gate from answered-well and has been for eighteen loops. loop 12 passes
S1, S2 and S4 and fails S3: the model predicts HOW MANY side effects a drug has, and "how many
proteins does it hit" predicts that better (r = 0.2898) than any biology in the model (partial r
0.0515 against a null 95th of 0.0758). Four more loops chipped at the same formulation.

The formulation is the problem. A count is not a side effect. Asking a cell model how many adverse
events a drug causes is asking it to be a promiscuity detector, and promiscuity is exactly what
target count already measures -- so the confound is not a nuisance in that design, it IS the design.

So ask the question a cell model is actually for: WHICH side effects, via the sharpest claim the
data supports -- two drugs that perturb the same part of the cell should share adverse effects. That
is a statement about cell organisation, it is falsifiable, and target count cannot explain it
because the comparison is between drugs rather than about one.

THE MEASURE OF "SAME PART OF THE CELL" is network separation (Menche et al. 2015):

    S_AB = <d_AB> - ( <d_AA> + <d_BB> ) / 2

the mean shortest-path distance between two target sets in the interactome, minus how spread out
each set is on its own. Negative means the two sets overlap as modules; positive means they sit in
different neighbourhoods. The interactome is BioGRID -- fetched and hash-pinned in loop 49 -- and
NOT cell_complete.json's own ppi table, whose provenance loop 49 showed is unknown. A result about
cell organisation should not rest on a network nobody can source.

PREDECLARED, before any number:

    E1 THE JOIN EXISTS AND CAN BE SPLIT   at least 500 drugs carrying both a target set and
                PT-level side effects, at least 100,000 pairs, and target sets that land in the
                interactome. Below that the held-out split has nothing to hold out.
    E2 SHARED TARGETS PREDICT SHARED SIDE EFFECTS   Spearman between target-set Jaccard and
                side-effect Jaccard, above a permutation null. If this fails the join is broken and
                nothing after it means anything. IT IS NOT A CELL-MODEL RESULT -- two drugs hitting
                the same protein sharing an effect is a lookup, and it is here as a floor.
    E3 THE NETWORK ADDS WHERE THE LOOKUP CANNOT   THE REAL GATE. Restricted to pairs sharing ZERO
                targets, network separation still predicts side-effect overlap above the null. This
                is the whole claim: drugs hitting DIFFERENT proteins that sit in the same
                neighbourhood share effects. If it fails, the honest answer is that shared side
                effects mean shared targets and the cell model adds nothing here.
    E4 IT SURVIVES THE FREQUENCY BASELINE   some side effects are near-universal, and two drugs
                both causing nausea is not evidence of anything. E3's correlation is re-run against
                a RESIDUAL: side-effect overlap with the part predictable from each drug's
                side-effect count and the population frequency of the terms removed first. The
                count confound that killed S3 is regressed out by construction rather than hoped
                away.
    E5 IT HOLDS ON HELD-OUT DRUGS   5-fold split GROUPED BY DRUG. The frequency baseline is fitted
                on training drugs only and applied to test pairs, and only pairs with BOTH drugs in
                the test fold are scored, so no drug informs its own prediction.

-> outputs/loop_shared_effect.json
"""
import collections
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
DCACHE = SC / "_shared_effect_dist.npz"

MIN_DRUGS = 500          # E1
MIN_PAIRS = 100_000      # E1
N_NULL = 200
N_FOLD = 5
SEED = 1904

log = []


def say(s=""):
    print(s)
    log.append(s)


def load_drug_targets(D):
    """cell_complete's `drugs` is keyed on GENE index -> the drugs hitting it. Invert it."""
    genes = D["genes"]
    tgt = collections.defaultdict(set)
    for gi, lst in D["drugs"].items():
        i = int(gi)
        if i >= len(genes):
            continue
        sym = genes[i]["name"]
        for e in lst:
            nm = (e.get("d") or "").strip().lower()
            if nm and nm != "null":
                tgt[nm].add(sym)
    return tgt


def load_side_effects():
    """SIDER PT-level terms per drug name. LLT rows are synonyms of the PT and would double-count."""
    cid = {}
    for ln in open(SC / "sider_names.tsv"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 1:
            cid[p[0]] = p[1].strip().lower()
    se = collections.defaultdict(set)
    with gzip.open(SC / "sider_se.tsv.gz", "rt", errors="ignore") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 6 and p[3] == "PT":
                n = cid.get(p[0])
                if n:
                    se[n].add(p[5])
    return se


def load_network():
    """BioGRID physical interactions as an undirected unweighted graph."""
    edges = set()
    with gzip.open(SC / "biogrid_hs_edges.tsv.gz", "rt", errors="ignore") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3 or p[2] != "physical":
                continue
            a, b = p[0], p[1]
            if a != b and a != "-" and b != "-":
                edges.add((a, b) if a < b else (b, a))
    nodes = sorted({x for e in edges for x in e})
    idx = {s: i for i, s in enumerate(nodes)}
    r = [idx[a] for a, b in edges] + [idx[b] for a, b in edges]
    c = [idx[b] for a, b in edges] + [idx[a] for a, b in edges]
    g = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(len(nodes), len(nodes)))
    return idx, g, len(edges)


def distances(idx, g, srcs):
    """Shortest-path distances between the target genes only, cached.

    BFS from 3,965 sources over 20,265 nodes is 643 MB if held whole, so it runs in chunks and only
    the source-by-source block is kept -- that is all the separation measure ever reads.
    """
    if DCACHE.exists():
        z = np.load(DCACHE, allow_pickle=True)
        if list(z["srcs"]) == list(srcs):
            return z["D"]
    rows = [idx[s] for s in srcs]
    out = np.empty((len(rows), len(rows)), np.float32)
    for a in range(0, len(rows), 250):
        b = min(a + 250, len(rows))
        d = shortest_path(g, method="D", unweighted=True, directed=False, indices=rows[a:b])
        out[a:b] = d[:, rows].astype(np.float32)
    np.savez_compressed(DCACHE, D=out, srcs=np.array(srcs))
    return out


def separation(A, B, D):
    """Menche S_AB. Negative = the two target sets are the same neighbourhood."""
    if not A or not B:
        return np.nan
    dab = D[np.ix_(A, B)]
    m = np.concatenate([dab.min(axis=1), dab.min(axis=0)])
    m = m[np.isfinite(m)]
    if not m.size:
        return np.nan
    daa = _within(A, D)
    dbb = _within(B, D)
    if not np.isfinite(daa) or not np.isfinite(dbb):
        return np.nan
    return float(m.mean() - (daa + dbb) / 2)


def _within(A, D):
    if len(A) == 1:
        return 0.0
    d = D[np.ix_(A, A)].copy()
    np.fill_diagonal(d, np.inf)
    m = d.min(axis=1)
    m = m[np.isfinite(m)]
    return float(m.mean()) if m.size else np.nan


def jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 50 -- do drugs that hit the same part of the cell share side effects?")
    say("=" * 100)

    D_all = json.load(open(CELL))
    tgt = load_drug_targets(D_all)
    se = load_side_effects()
    idx, g, n_edge = load_network()
    say(f"\n  BioGRID physical: {len(idx):,} proteins, {n_edge:,} edges")

    drugs = sorted(d for d in (set(tgt) & set(se))
                   if len([s for s in tgt[d] if s in idx]) > 0 and len(se[d]) >= 5)
    T = {d: sorted(s for s in tgt[d] if s in idx) for d in drugs}
    srcs = sorted({s for d in drugs for s in T[d]})
    say(f"  drugs with targets {len(tgt):,} · with side effects {len(se):,} · JOIN {len(drugs):,}")
    say(f"  distinct target proteins in the interactome {len(srcs):,}")

    # ---- E1 -------------------------------------------------------------------------------------
    n_pairs = len(drugs) * (len(drugs) - 1) // 2
    say(f"\n  E1 THE JOIN EXISTS AND CAN BE SPLIT")
    say(f"     {len(drugs):,} drugs, {n_pairs:,} pairs, median {np.median([len(T[d]) for d in drugs]):.0f} "
        f"targets and {np.median([len(se[d]) for d in drugs]):.0f} side effects per drug")
    e1 = bool(len(drugs) >= MIN_DRUGS and n_pairs >= MIN_PAIRS)
    say(f"     E1 {'PASS' if e1 else 'FAIL'}  (>= {MIN_DRUGS} drugs and >= {MIN_PAIRS:,} pairs)")

    say(f"\n  building the interactome distance block, {len(srcs):,} x {len(srcs):,}")
    Dm = distances(idx, g, srcs)
    pos = {s: i for i, s in enumerate(srcs)}
    TI = {d: [pos[s] for s in T[d]] for d in drugs}
    say(f"     done in {time.time() - t0:.1f}s  (reachable fraction "
        f"{np.isfinite(Dm).mean():.3f})")

    # ---- pair table -------------------------------------------------------------------------------
    # The side-effect Jaccard is computed as a DRUG x DRUG matrix rather than pair by pair, because
    # the permutation null needs it 200 times over. As a matrix, permuting the drug labels is
    # JS[perm][:, perm] -- pure indexing -- where the pairwise loop would be 38 million set
    # operations per gate. Same numbers, three orders of magnitude cheaper.
    terms = sorted({t for d in drugs for t in se[d]})
    tpos = {t: i for i, t in enumerate(terms)}
    M = np.zeros((len(drugs), len(terms)), np.float32)
    for a, d in enumerate(drugs):
        for t in se[d]:
            M[a, tpos[t]] = 1
    inter = M @ M.T
    cnt = M.sum(1)
    JS = inter / np.maximum(cnt[:, None] + cnt[None, :] - inter, 1e-9)
    ii, jj = np.triu_indices(len(drugs), k=1)
    js = JS[ii, jj]
    jt = np.array([jac(set(T[drugs[a]]), set(T[drugs[b]])) for a, b in zip(ii, jj)])
    sep = np.array([separation(TI[drugs[a]], TI[drugs[b]], Dm) for a, b in zip(ii, jj)], float)
    ok = np.isfinite(sep)
    say(f"     pair table built: {len(jt):,} pairs, {ok.sum():,} with a defined separation")

    # ---- E2 -------------------------------------------------------------------------------------
    def null95(x, groups=None):
        """Permute the OUTCOME across drugs, not across pairs.

        Pairs are not independent -- every drug appears in hundreds of them -- so shuffling pairs
        would break that structure and produce a null far too tight. Permuting the drug labels on
        the side-effect side keeps each drug's whole row of pairs together.
        """
        out = []
        for _ in range(N_NULL):
            perm = rng.permutation(len(drugs))
            sh = JS[perm][:, perm][ii, jj]
            m = np.isfinite(x) & np.isfinite(sh)
            if groups is not None:
                m = m & groups
            r = spearmanr(x[m], sh[m]).statistic
            out.append(abs(r) if np.isfinite(r) else 0.0)
        return float(np.percentile(out, 95))

    r_t = float(spearmanr(jt, js).statistic)
    n_t = null95(jt)
    say(f"\n  E2 SHARED TARGETS PREDICT SHARED SIDE EFFECTS")
    say(f"     Spearman(target Jaccard, side-effect Jaccard)  {r_t:+.4f}   null 95th {n_t:.4f}")
    e2 = bool(r_t > n_t)
    say(f"     E2 {'PASS' if e2 else 'FAIL'}  -- and this is the LOOKUP floor, not the cell model")

    # ---- E3 the real gate --------------------------------------------------------------------------
    zero = (jt == 0) & ok
    r_s = float(spearmanr(-sep[zero], js[zero]).statistic)
    n_s = null95(-sep, groups=zero)
    say(f"\n  E3 THE NETWORK ADDS WHERE THE LOOKUP CANNOT")
    say(f"     restricted to the {int(zero.sum()):,} pairs sharing ZERO targets "
        f"({zero.sum()/len(jt):.1%} of all pairs)")
    say(f"     Spearman(-separation, side-effect Jaccard)     {r_s:+.4f}   null 95th {n_s:.4f}")
    e3 = bool(r_s > n_s)
    say(f"     E3 {'PASS' if e3 else 'FAIL'}")

    # ---- E4 the frequency residual ------------------------------------------------------------------
    def freq_expect(train_drugs, pairs_a, pairs_b):
        """Expected side-effect overlap from term frequency and per-drug counts alone.

        Each drug is modelled as drawing its side effects independently with probability
        proportional to how often the term appears across the TRAINING drugs. The expected
        intersection is then sum_t p_t(A) p_t(B), and the expected Jaccard follows. Nothing about
        targets or the network enters, which is what makes it the right thing to subtract.
        """
        cnt = collections.Counter(t for d in train_drugs for t in se[d])
        n = len(train_drugs)
        p = {t: c / n for t, c in cnt.items()}
        tot = sum(p.values())
        out = np.empty(len(pairs_a))
        for k, (a, b) in enumerate(zip(pairs_a, pairs_b)):
            na, nb = len(se[drugs[a]]), len(se[drugs[b]])
            # each drug draws na terms with probability p_t/tot -> E|A cap B| = sum_t q_t(A) q_t(B)
            inter = sum((min(1.0, na * v / tot)) * (min(1.0, nb * v / tot)) for v in p.values())
            out[k] = inter / max(na + nb - inter, 1e-9)
        return out

    fe = freq_expect(drugs, ii, jj)
    # residual: side-effect overlap with the frequency-and-count expectation regressed out
    A = np.column_stack([fe, np.array([len(se[drugs[a]]) for a in ii], float),
                         np.array([len(se[drugs[b]]) for b in jj], float), np.ones(len(fe))])
    coef, *_ = np.linalg.lstsq(A, js, rcond=None)
    resid = js - A @ coef
    r_r = float(spearmanr(-sep[zero], resid[zero]).statistic)

    def null95_resid():
        out = []
        for _ in range(N_NULL):
            perm = rng.permutation(len(drugs))
            sh = JS[perm][:, perm][ii, jj]
            c, *_ = np.linalg.lstsq(A, sh, rcond=None)
            rr = sh - A @ c
            r = spearmanr(-sep[zero], rr[zero]).statistic
            out.append(abs(r) if np.isfinite(r) else 0.0)
        return float(np.percentile(out, 95))

    n_r = null95_resid()
    say(f"\n  E4 IT SURVIVES THE FREQUENCY BASELINE")
    say(f"     frequency+count baseline explains R2 "
        f"{1 - np.var(resid)/np.var(js):.4f} of side-effect overlap on its own")
    say(f"     Spearman(-separation, RESIDUAL overlap)        {r_r:+.4f}   null 95th {n_r:.4f}")
    e4 = bool(r_r > n_r)
    say(f"     E4 {'PASS' if e4 else 'FAIL'}")

    # ---- E5 held out by drug -------------------------------------------------------------------------
    fold = rng.permutation(len(drugs)) % N_FOLD
    fr, fn = [], []
    for k in range(N_FOLD):
        te = fold == k
        both = te[ii] & te[jj] & zero
        if both.sum() < 200:
            continue
        tr = [drugs[i] for i in range(len(drugs)) if not te[i]]
        fe_k = freq_expect(tr, ii[both], jj[both])
        Ak = np.column_stack([fe_k, np.array([len(se[drugs[a]]) for a in ii[both]], float),
                              np.array([len(se[drugs[b]]) for b in jj[both]], float),
                              np.ones(int(both.sum()))])
        ck, *_ = np.linalg.lstsq(Ak, js[both], rcond=None)
        rk = js[both] - Ak @ ck
        fr.append(float(spearmanr(-sep[both], rk).statistic))
        nulls = []
        for _ in range(50):
            nulls.append(abs(spearmanr(-sep[both], rng.permutation(rk)).statistic))
        fn.append(float(np.percentile(nulls, 95)))
    say(f"\n  E5 IT HOLDS ON HELD-OUT DRUGS")
    say(f"     {N_FOLD}-fold grouped by drug, only pairs with BOTH drugs held out, baseline fitted "
        f"on the training drugs")
    for k, (a, b) in enumerate(zip(fr, fn)):
        say(f"     fold {k}   r {a:+.4f}   null 95th {b:.4f}   "
            f"{'above' if a > b else 'BELOW'}")
    e5 = bool(len(fr) == N_FOLD and float(np.mean(fr)) > float(np.mean(fn)) and all(a > 0 for a in fr))
    say(f"     mean r {np.mean(fr):+.4f} vs mean null {np.mean(fn):.4f}")
    say(f"     E5 {'PASS' if e5 else 'FAIL'}  (every fold positive and the mean above the null)")

    say("\n" + "=" * 100)
    gates = {"E1 join exists and can be split": e1, "E2 shared targets predict shared effects": e2,
             "E3 network adds where the lookup cannot": e3,
             "E4 survives the frequency baseline": e4, "E5 holds on held-out drugs": e5}
    for k, v in gates.items():
        say(f"  {k:<44}{'PASS' if v else 'FAIL'}")
    if e3 and e4 and e5:
        say(f"  DRUGS THAT HIT DIFFERENT PROTEINS IN THE SAME NEIGHBOURHOOD SHARE SIDE EFFECTS.")
        say(f"  On {int(zero.sum()):,} pairs with NO target in common, network separation predicts")
        say(f"  side-effect overlap at {r_s:+.4f}, and {r_r:+.4f} once the part explained by how many")
        say(f"  side effects each drug has and how common the terms are is removed. That residual is")
        say(f"  the whole point: the count confound that has failed this row since loop 12 is")
        say(f"  regressed out by construction rather than argued with.")
    else:
        say(f"  THE ANSWER IS NO, OR NOT YET. Shared targets do predict shared side effects "
            f"({r_t:+.4f}),")
        say(f"  which is a lookup. On the {int(zero.sum()):,} pairs with no target in common the")
        say(f"  network gives {r_s:+.4f} against a null of {n_s:.4f}, and {r_r:+.4f} against "
            f"{n_r:.4f} on the residual.")
        say(f"  Reformulating the question from HOW MANY to WHICH was the right move and it is not")
        say(f"  enough on its own; recorded as a negative rather than retried until it passes.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "sider_se.tsv.gz"), str(SC / "sider_names.tsv"),
                              str(SC / "biogrid_hs_edges.tsv.gz"), str(CELL)],
                      available=n_pairs, used=int(zero.sum()), selection="filtered", seed=SEED,
                      controls=[f"{N_NULL} permutations of drug labels on the side-effect side, "
                                f"which keeps each drug's pairs together",
                                "restricted to pairs with ZERO shared targets so the lookup cannot "
                                "explain the result",
                                "side-effect count and term frequency regressed out before the "
                                "network is scored",
                                "5-fold grouped by drug with the baseline fitted on training drugs"],
                      note="BioGRID rather than cell_complete's own ppi table, whose provenance "
                           "loop 49 showed is unknown")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_shared_effect", "manifest": man, "gates": gates,
               "n_drugs": len(drugs), "n_pairs": int(len(jt)), "n_zero_target": int(zero.sum()),
               "r_target": r_t, "null_target": n_t, "r_sep": r_s, "null_sep": n_s,
               "r_resid": r_r, "null_resid": n_r,
               "baseline_r2": float(1 - np.var(resid) / np.var(js)),
               "folds": fr, "fold_nulls": fn, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_shared_effect.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_shared_effect.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
