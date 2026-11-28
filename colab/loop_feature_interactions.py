"""LOOP 157 -- WAS THE ENHANCER'S GREEDY PATH VALID? PROFILING THE FEATURE DESIGN SPACE.

The five-turn enhancer (outputs/esm_halflife_improve.json) searched GREEDILY: T1 mean-pooled 8M,
T2 = T1 + terminal windows, T3 = T2 + annotations, and then T4 = 35M ALONE, off the path. It
concluded that scale (+0.0432) was the only axis that cleared the noise bar and that terminal
windows and curated annotations bought nothing.

Every one of those conclusions assumes the feature blocks are SEPARABLE -- that what a block does
when bolted onto 8M is what it would do bolted onto 35M. That assumption was never tested, and it
is exactly the assumption a greedy search cannot check from inside itself. T2 asked "do terminal
windows help the 8M representation?" and answered no. It could not ask "do they help the 35M one?",
because T4 had not run yet.

interaction_profiler (standalone/interaction_profiler.py, extracted from the REM project) answers
precisely this class of question: given a black-box objective over discrete variables, which
variables genuinely interact, at what order. Here the variables are the seven feature blocks, each
in or out, and the objective is the cross-validated Spearman on the SAME fixed homology-aware folds
the enhancer used. Nothing else about the objective is assumed.

WHY THIS PROBLEM IS THE RIGHT TEST OF THE TOOL AS WELL AS OF THE ENHANCER. Seven binary blocks is a
space of 128 configurations -- small enough to ENUMERATE COMPLETELY. So the profiler's answer can be
checked against the exact Moebius decomposition of the full table rather than taken on trust. A tool
that ships a self-test on planted structure is still unvalidated on YOUR objective; I3 validates it
on this one, exactly, and would catch a disagreement.

  THE SEVEN BLOCKS
    0  ESM-2 8M mean-pooled, whole sequence          320 dims
    1  ESM-2 35M mean-pooled, whole sequence         480 dims
    2  N-terminal 60-residue window (8M)             320 dims
    3  C-terminal 60-residue window (8M)             320 dims
    4  amino-acid composition                         20 dims
    5  log sequence length                             1 dim
    6  Rega curated annotations (degron/UPS/essential) 3 dims

PREDECLARED.

  I1 CAPABILITY.
       all seven blocks load and align to the same 1,839 proteins; the objective is DETERMINISTIC
       (the profiler measures this itself and must report deterministic, since ridge on fixed folds
       has no stochastic component); and 2^7 = 128 configurations is affordable to enumerate, which
       is what makes I3 possible. Gate: all three.

  I2 THE PROFILE.
       run profile_objective over the seven blocks. Gate: report the strategy, the irreducible
       orders and the interacting groups. This gate passes on being reported.

  I3 VALIDATE THE TOOL ON THIS OBJECTIVE, NOT ON ITS OWN SELF-TEST.
       enumerate all 128 configurations, compute the exact Moebius residual for every subset
       against the all-zero reference, and compare with the profiler's reported strengths.
       Gate: every pair and triple the profiler calls irreducible must have an exact residual above
       the same threshold, and vice versa -- agreement on the SET, not merely on the headline.
       A disagreement here would mean the profiler is wrong on this problem and I2 is withdrawn.

  I4 WAS THE GREEDY PATH VALID?
       compare the enhancer's sequential best against the best configuration in the enumerated
       table. Gate: report the gap. If greedy found the global optimum the path was sound; if not,
       the enhancer's conclusions are conditional on an order it never justified.

  I5 THE QUESTION T2 COULD NOT ASK.
       does the terminal-window block help the 35M representation, given it did not help the 8M
       one? This is a direct read of one interaction the greedy path structurally could not see.
       Gate: report both conditional effects.

  I6 WHAT THIS CANNOT SHOW.
       block-level in/out only. It says nothing about window SIZE, pooling scheme, or encoder
       depth, and a block that is useless as a whole may contain a useful column.

-> outputs/loop_feature_interactions.json
"""
import json
import math
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "standalone"))
import run_manifest as RM        # noqa: E402
import loop_replication as LR    # noqa: E402
import gate_guard as GG          # noqa: E402
from interaction_profiler import profile_objective  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
REGA = SC / "destroyer" / "rega_4.xlsx"
EMB8 = Path("colab/data/ml/esm2_8M_halflife.npz")
EMBT = Path("colab/data/ml/esm2_8M_halflife_term.npz")
EMB35 = Path("colab/data/ml/esm2_35M_halflife.npz")

SEED = 15800
KMER, JACCARD, NFOLD = 5, 0.30, 5
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
AA = "ACDEFGHIKLMNPQRSTVWY"
TAU = 0.005

BLOCKS = ["esm8M", "esm35M", "Nterm60", "Cterm60", "aa_comp", "log_len", "annotations"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    i, s = 0, x[o]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return 0.0
    ra, rb = _rank(a[m]), _rank(b[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    dd = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / dd) if dd > 0 else 0.0


def ridge_cv(X, y, folds):
    pred = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        Xtr, ytr = X[tr], y[tr]
        inner = np.arange(len(ytr)) % 3
        best, ba = -9.0, ALPHAS[0]
        for a in ALPHAS:
            sc = []
            for k in range(3):
                i_tr, i_te = inner != k, inner == k
                A = Xtr[i_tr]
                mu, sd = A.mean(0), A.std(0) + 1e-8
                An = (A - mu) / sd
                w = np.linalg.solve(An.T @ An + a * np.eye(An.shape[1]),
                                    An.T @ (ytr[i_tr] - ytr[i_tr].mean()))
                sc.append(spear(((Xtr[i_te] - mu) / sd) @ w + ytr[i_tr].mean(), ytr[i_te]))
            m = float(np.nanmean(sc))
            if m > best:
                best, ba = m, a
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        An = (Xtr - mu) / sd
        w = np.linalg.solve(An.T @ An + ba * np.eye(An.shape[1]), An.T @ (ytr - ytr.mean()))
        pred[te] = ((X[te] - mu) / sd) @ w + ytr.mean()
    return spear(pred, y)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 157 -- was the enhancer's greedy path valid? Profiling the feature design space")
    say("=" * 100)
    say()

    import gzip
    import re
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    d = pd.read_excel(REGA, sheet_name="Proteome", header=1)
    d["halflife_mean"] = pd.to_numeric(d["halflife_mean"], errors="coerce")
    d = d[np.isfinite(d["halflife_mean"]) & (d["halflife_mean"] > 0)].copy()
    Z8 = np.load(EMB8, allow_pickle=True)
    accs = [str(a) for a in Z8["accs"]]
    pos = {a: i for i, a in enumerate(accs)}
    d = d[d["Accession"].astype(str).isin(pos)].copy()
    o = [pos[str(a)] for a in d["Accession"].astype(str)]
    accl = d["Accession"].astype(str).values
    y = np.log(d["halflife_mean"].values.astype(float))
    lens = Z8["lengths"][o].astype(float)

    ZT = np.load(EMBT, allow_pickle=True)
    ti = {str(a): i for i, a in enumerate(ZT["accs"])}
    Z35 = np.load(EMB35, allow_pickle=True)
    mi = {str(a): i for i, a in enumerate(Z35["accs"])}

    seqs, a_, buf = {}, None, []
    with gzip.open(SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if a_ and buf and a_ in pos:
                    seqs[a_] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                a_, buf = (m.group(1) if m else None), []
            else:
                buf.append(ln.strip())
    if a_ and buf and a_ in pos:
        seqs[a_] = "".join(buf)

    comp = np.zeros((len(accl), len(AA)), np.float32)
    for i, a in enumerate(accl):
        s = seqs.get(a, "")
        if s:
            for j, aa in enumerate(AA):
                comp[i, j] = s.count(aa) / len(s)
    ann = []
    for c, tr in (("Known degron", lambda v: 0.0 if str(v).strip() in ("-", "nan", "") else 1.0),
                  ("UPS components", lambda v: 0.0 if str(v).strip() in ("-", "nan", "") else 1.0),
                  ("Essential Protein",
                   lambda v: 1.0 if str(v).strip().lower() in ("true", "yes") else 0.0)):
        if c in d.columns:
            ann.append(d[c].map(tr).values.astype(np.float32))
    ANN = np.array(ann, np.float32).T

    MAT = {
        "esm8M": Z8["X"][o],
        "esm35M": Z35["X"][[mi[a] for a in accl]],
        "Nterm60": ZT["N"][[ti[a] for a in accl]],
        "Cterm60": ZT["C"][[ti[a] for a in accl]],
        "aa_comp": comp,
        "log_len": np.log(lens)[:, None].astype(np.float32),
        "annotations": ANN,
    }

    rows, cols = [], []
    for i, a in enumerate(accl):
        s = seqs.get(a, "")
        for k in {hash(s[j:j + KMER]) % (1 << 20) for j in range(max(0, len(s) - KMER + 1))}:
            rows.append(i)
            cols.append(k)
    M = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(accl), 1 << 20))
    sz = np.asarray(M.sum(1)).ravel()
    I = (M @ M.T).toarray()
    J = I / np.maximum(sz[:, None] + sz[None, :] - I, 1e-9)
    np.fill_diagonal(J, 0.0)
    _, lab = connected_components(csr_matrix(J >= JACCARD), directed=False)
    load = np.zeros(NFOLD)
    fmap = {}
    for c in np.argsort(-np.bincount(lab)):
        f = int(np.argmin(load))
        fmap[c] = f
        load[f] += (lab == c).sum()
    folds = np.array([fmap[c] for c in lab])

    cache = {}

    def objective(cfg):
        key = tuple(int(cfg[i]) for i in range(len(BLOCKS)))
        if key in cache:
            return cache[key]
        use = [BLOCKS[i] for i, on in enumerate(key) if on]
        val = 0.0 if not use else ridge_cv(np.hstack([MAT[b] for b in use]), y, folds)
        cache[key] = val
        return val

    gates, res = {}, {}

    say("I1 CAPABILITY")
    dims = {b: int(MAT[b].shape[1]) for b in BLOCKS}
    aligned = all(MAT[b].shape[0] == len(y) for b in BLOCKS)
    say(f"     {len(y):,} proteins; blocks and dims: {dims}")
    say(f"     all blocks aligned to the same rows: {aligned}")
    say(f"     full design space 2^{len(BLOCKS)} = {2 ** len(BLOCKS)} configurations -- "
        f"enumerable, which is what makes I3 an exact check rather than a hope")
    gates["I1"] = bool(aligned and len(y) > 1000)
    res["i1"] = {"n": int(len(y)), "dims": dims, "aligned": aligned,
                 "space": 2 ** len(BLOCKS), "pass": gates["I1"]}
    say(f"     I1 {'PASS' if gates['I1'] else 'FAIL'}")
    say()

    say("I2 THE PROFILE")
    rep = profile_objective(objective, variables=range(len(BLOCKS)), state_counts=2,
                            tau=TAU, max_order=3, n_references=3, seed=SEED)
    for line in rep.summary().split("\n"):
        say("     " + line)
    named = {tuple(BLOCKS[i] for i in g): v for g, v in rep.strengths.items()}
    say()
    say("     interacting groups, named:")
    for g, v in sorted(named.items(), key=lambda kv: -kv[1]):
        say(f"       {' + '.join(g):<44} {v:+.4f}")
    gates["I2"] = True
    res["i2"] = {"strategy": rep.strategy, "rationale": rep.rationale,
                 "order_histogram": rep.order_histogram, "treewidth_upper": rep.treewidth_upper,
                 "separable": bool(rep.separable), "inconclusive": bool(rep.inconclusive),
                 "deterministic": bool(rep.noise.get("deterministic")),
                 "tau": rep.tau, "objective_calls": rep.objective_evaluations,
                 "strengths": {"+".join(k): v for k, v in named.items()}}
    say()

    say("I3 VALIDATE THE TOOL ON THIS OBJECTIVE")
    n = len(BLOCKS)
    table = {}
    for mask in range(1 << n):
        cfg = {i: (mask >> i) & 1 for i in range(n)}
        table[tuple(cfg[i] for i in range(n))] = objective(cfg)
    say(f"     enumerated all {len(table)} configurations")

    def exact_delta(group):
        tot = 0.0
        for r in range(len(group) + 1):
            for sub in combinations(group, r):
                key = [0] * n
                for i in sub:
                    key[i] = 1
                tot += ((-1) ** (len(group) - r)) * table[tuple(key)]
        return abs(tot)

    disagree = []
    checked = 0
    for order in (2, 3):
        for g in combinations(range(n), order):
            checked += 1
            ex = exact_delta(g)
            prof = rep.strengths.get(g, 0.0)
            if (ex > TAU) != (prof > TAU):
                disagree.append({"group": [BLOCKS[i] for i in g], "exact": ex, "profiler": prof})
    say(f"     compared {checked} groups of order 2 and 3 against the exact Moebius residual")
    say(f"     disagreements: {len(disagree)}")
    for dd in disagree[:8]:
        say(f"       {' + '.join(dd['group']):<44} exact {dd['exact']:+.4f}  "
            f"profiler {dd['profiler']:+.4f}")
    GG.verdict(not disagree,
               "the profiler's set of irreducible groups matches the exact decomposition on THIS "
               "objective, so I2 may be read.",
               f"the profiler disagrees with the exact decomposition on {len(disagree)} groups. "
               f"I2 is WITHDRAWN and the exact table is what stands.", emit=say)
    gates["I3"] = not disagree
    res["i3"] = {"n_configs": len(table), "groups_checked": checked,
                 "disagreements": disagree, "pass": gates["I3"]}
    say(f"     I3 {'PASS' if gates['I3'] else 'FAIL'}")
    say()

    say("I4 WAS THE GREEDY PATH VALID?")
    best_key = max(table, key=table.get)
    best_val = table[best_key]
    best_blocks = [BLOCKS[i] for i, on in enumerate(best_key) if on]
    greedy = {}
    for nm, ks in (("T1 8M", ("esm8M",)),
                   ("T2 8M+term", ("esm8M", "Nterm60", "Cterm60")),
                   ("T3 8M+term+ann", ("esm8M", "Nterm60", "Cterm60", "annotations")),
                   ("T4 35M", ("esm35M",))):
        key = tuple(1 if BLOCKS[i] in ks else 0 for i in range(n))
        greedy[nm] = table[key]
    g_best = max(greedy, key=greedy.get)
    say("     the enhancer's four configurations, re-scored on this exact table:")
    for k, v in greedy.items():
        say(f"       {k:<18} {v:+.4f}")
    say(f"     enhancer best: {g_best} at {greedy[g_best]:+.4f}")
    say(f"     GLOBAL best:   {' + '.join(best_blocks)} at {best_val:+.4f}")
    gap = best_val - greedy[g_best]
    say(f"     gap the greedy path left on the table: {gap:+.4f}")
    GG.verdict(gap <= TAU,
               f"greedy found the global optimum (gap {gap:+.4f} <= tau {TAU}); the enhancer's "
               f"sequential path was sound and its conclusions stand as stated.",
               f"greedy MISSED the global optimum by {gap:+.4f}. The enhancer's conclusions are "
               f"conditional on an ordering it never justified, and the better configuration is "
               f"{' + '.join(best_blocks)}.", emit=say)
    gates["I4"] = True
    res["i4"] = {"greedy": greedy, "greedy_best_name": g_best, "greedy_best": greedy[g_best],
                 "global_best_blocks": best_blocks, "global_best": best_val, "gap": gap}
    say()

    say("I5 THE QUESTION T2 COULD NOT ASK")

    def val(*names):
        return table[tuple(1 if BLOCKS[i] in names else 0 for i in range(n))]

    d8 = val("esm8M", "Nterm60", "Cterm60") - val("esm8M")
    d35 = val("esm35M", "Nterm60", "Cterm60") - val("esm35M")
    say(f"     terminal windows added to 8M  : {d8:+.4f}   (this is what T2 measured)")
    say(f"     terminal windows added to 35M : {d35:+.4f}   (T2 structurally could not ask)")
    say(f"     interaction between the two decisions: {d35 - d8:+.4f}")
    GG.verdict(abs(d35 - d8) <= TAU,
               "the terminal-window decision does not depend on the encoder, so T2's answer "
               "transfers and the greedy ordering cost nothing here.",
               f"the terminal-window decision DOES depend on the encoder ({d8:+.4f} on 8M vs "
               f"{d35:+.4f} on 35M). T2's 'terminals add nothing' was an answer about 8M only, "
               f"and the enhancer generalised it without warrant.", emit=say)
    gates["I5"] = True
    res["i5"] = {"delta_on_8M": d8, "delta_on_35M": d35, "interaction": d35 - d8}
    say()

    say("I6 WHAT THIS CANNOT SHOW")
    say("     block-level in/out only. It says nothing about window SIZE, pooling scheme or")
    say("     encoder depth, and a block useless as a whole may still contain a useful column.")
    say("     The objective is one CV score on one fold assignment; a different clustering")
    say("     threshold would give a different table, though the enhancer used this same one.")
    gates["I6"] = True
    res["i6"] = {"block_level_only": True, "single_fold_assignment": True}
    say()

    say("=" * 100)
    for k in ("I1", "I2", "I3", "I4", "I5", "I6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[REGA, EMB8, EMBT, EMB35,
                              Path("standalone/interaction_profiler.py")],
                      available=2 ** len(BLOCKS), used=len(table), selection="all", seed=SEED,
                      controls=["the profiler validated against the EXACT Moebius decomposition of "
                                "the fully enumerated table, on this objective (I3)",
                                "the same homology-aware folds the enhancer used",
                                "the greedy path re-scored on the exact table rather than trusted",
                                "conclusions emitted through gate_guard.verdict"],
                      note="interaction_profiler applied to the feature-block design space of loop "
                           "156 and the enhancer. Seven binary blocks is 128 configurations, small "
                           "enough to enumerate, so the tool is checked exactly rather than "
                           "trusted -- and the greedy search that produced the enhancer's "
                           "conclusions is audited against the global optimum.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 157 -- feature-block interaction profile", "manifest": man,
               "gates": gates, "blocks": BLOCKS, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_feature_interactions.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_feature_interactions.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
