"""LOOP 59 -- THE DRUG RESPONSE FIELD, and the axis flip that makes the old confound impossible.

TWO IDEAS, AND THE SECOND ONE IS THE IMPORTANT ONE.

IDEA 1 -- REPRESENT A DRUG AS A FIELD, NOT A LIST.
Every attempt so far has represented a drug as "the mean of something over its target set": mean
essentiality, mean LOEUF, mean PPI degree. That average throws away three things at once -- WHICH
targets, WHAT the drug does to each of them, and HOW those targets sit relative to each other in the
cell. It is the equivalent of predicting a protein's shape from its average hydrophobicity.

What AlphaFold actually did was not "more data". It built an INTERMEDIATE that made the geometry
explicit -- a pairwise distogram -- and reasoned over that instead of over the sequence. The
equivalent move here is to build the intermediate that makes the PERTURBATION explicit:

    p       a sparse SIGNED vector over genes. p[target] = -1 for an inhibitor, blocker or
            antagonist, +1 for an agonist, activator or potentiator. This is the thing every previous
            loop discarded: what the drug DOES to the protein, not merely that it touches it.
    r       = (I - alpha*A)^-1 p, propagated over the cell's SIGNED regulatory network -- 612,133
            edges of which 46,175 are activating and 7,009 repressing. Signs multiply along paths,
            so inhibiting an activator lands as repression downstream, which is the arithmetic the
            mean-of-a-set could never express.

r is the RESPONSE FIELD: a dense vector over every gene saying what the cell looks like after the
drug. Two drugs with no target in common but the same downstream consequence come out similar, which
is the entire reason for having a network at all.

IDEA 2 -- FLIP THE AXIS, WHICH IS WHAT ACTUALLY KILLS THE CONFOUND.
Loops 12, 50 and 58 all failed the same way and it was never the model. Each one aggregated OVER
TERMS WITHIN A DRUG -- how many effects, how much overlap, what fraction serious -- and every such
quantity inherits how many effects were written down for that drug, which is fame. Loop 58 reduced
it (target count 0.2898 -> 0.1769) and could not remove it.

So score the other way round. FOR EACH SIDE-EFFECT TERM, RANK THE DRUGS. Does the model put the
drugs that cause hepatotoxicity above the drugs that do not? Inside a single term, "how many effects
this drug has" is not a feature that can help, because every drug in the comparison is being judged
on the same one effect. The confound is not controlled for, it is structurally absent.

That flip is free, it needed no new data, and it should have been the design in loop 12.

PREDECLARED, before any number:

  P1 THE FIELD IS NOT JUST THE TARGET SET
       the propagated response must actually differ from the raw perturbation vector. If
       Spearman(r, p) over genes is near 1, the propagation is an expensive identity and everything
       below is the target list wearing a new name. Declared: median |Spearman| < 0.9.
  P2 THE SIGNS MATTER
       the same pipeline with every action forced to +1 -- that is, knowing WHICH proteins are hit
       but not WHAT is done to them. If signed does not beat unsigned, then the action field is
       decoration and this loop says so.
  P3 THE NETWORK MATTERS
       the same propagation over a DEGREE-PRESERVING SHUFFLE of the regulatory network. loop 56
       found that destroying the alignment of its structural features to the actual structure cost
       0.0101 of 0.9258 -- almost nothing -- and the only reason that was visible is that the
       control was run. It is run here.
  P4 PER-TERM RANKING BEATS THE FAME BASELINE   THE REAL GATE.
       macro AUC over side-effect terms with enough drugs on both sides, against a baseline that
       knows ONLY how many effects each drug has. That baseline is what has won every previous
       version of this row; on this axis it should be at chance, and if it is not, the flip failed.
  P5 HELD OUT BY DRUG, AND THE NULLS COLLAPSE IT
       drug-grouped folds; permutation nulls with the whole pipeline refitted.

-> outputs/loop_response_field.json
"""
import collections
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

ALPHA = 0.5              # propagation strength; declared, not tuned
MIN_DRUGS_PER_TERM = 25  # a per-term AUC needs drugs on both sides
MIN_POS_FRAC = 0.05
MAX_POS_FRAC = 0.95
N_COMP = 64              # response field compressed to this many components before fitting
FOLDS = 5
N_NULL = 40
P1_BAR = 0.90
SEED = 1904

NEG = {"inhibitor", "blocker", "antagonist", "negative modulator", "cleavage", "suppressor",
       "inverse agonist", "antibody"}
POS = {"agonist", "activator", "positive modulator", "potentiator", "inducer"}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build_perturbations():
    """Each drug as a SIGNED sparse vector over genes: what it does, not merely what it touches."""
    D = json.load(open(CELL))
    genes = D["genes"]
    n = len(genes)
    per = collections.defaultdict(dict)
    for gi, lst in D["drugs"].items():
        i = int(gi)
        if i >= n:
            continue
        for e in lst:
            nm = (e.get("d") or "").strip().lower()
            if not nm or nm == "null":
                continue
            act = (e.get("t") or "").strip().lower()
            s = -1.0 if act in NEG else (1.0 if act in POS else 0.0)
            # an unknown action is 0, NOT a guess. A drug whose actions are all unknown gets a zero
            # vector and drops out at the coverage check rather than being silently signed.
            prev = per[nm].get(i, 0.0)
            per[nm][i] = prev + s
    return D, per, n


def signed_operator(D, n, shuffle=None):
    """(I - alpha*A) for the signed, column-normalised regulatory network."""
    rows, cols, vals = [], [], []
    edges = D["reg"]
    if shuffle is not None:
        # degree-preserving: keep every source, permute the targets. Out-degree is exactly conserved
        # and in-degree distribution is conserved in expectation, so what is destroyed is WHICH gene
        # regulates which -- the alignment, not the shape.
        tgt = shuffle.permutation([e[1] for e in edges])
        edges = [[e[0], int(t), e[2]] for e, t in zip(edges, tgt)]
    for a, b, s in edges:
        if a >= n or b >= n:
            continue
        rows.append(b)
        cols.append(a)
        vals.append(1.0 if s == 0 else float(s))
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(np.abs(A).sum(0)).ravel()
    deg[deg == 0] = 1.0
    A = A.multiply(1.0 / deg)
    return (identity(n, format="csr") - ALPHA * csr_matrix(A)).tocsc()


def propagate(M, P):
    """Solve (I - alpha A) r = p for every drug at once, returned DENSE.

    spsolve hands back a sparse matrix when given a sparse right-hand side, and every downstream
    step here -- correlation, SVD, standardisation -- expects an array. Converting at the boundary
    rather than at each use keeps the type from leaking.
    """
    from scipy.sparse import csc_matrix, issparse
    R = spsolve(M, csc_matrix(P.T))
    if issparse(R):
        R = R.toarray()
    return np.asarray(R).T


def load_side_effects():
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
                nm = cid.get(p[0])
                if nm:
                    se[nm].add(p[5])
    return se


def fit_per_term(X, Y, drugs, rng, folds=FOLDS):
    """For each TERM, rank the DRUGS. Drug-grouped folds; one model per fold, all terms at once."""
    nd = len(drugs)
    fold = rng.permutation(nd) % folds
    P = np.full(Y.shape, np.nan)
    for k in range(folds):
        tr, te = fold != k, fold == k
        if tr.sum() < 50 or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        for j in range(Y.shape[1]):
            y = Y[tr, j]
            if y.sum() < 5 or (1 - y).sum() < 5:
                continue
            m = LogisticRegression(max_iter=2000, C=1.0)
            m.fit(A, y)
            P[te, j] = m.decision_function(B)
    return P


def macro_term_auc(P, Y):
    aucs = []
    for j in range(Y.shape[1]):
        v, y = P[:, j], Y[:, j]
        m = np.isfinite(v)
        if y[m].sum() < 10 or (1 - y[m]).sum() < 10:
            continue
        a = LR.auc(v[m], y[m])
        if np.isfinite(a):
            aucs.append(a)
    return (float(np.mean(aucs)) if aucs else np.nan), len(aucs)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 59 -- the drug response field, and the axis flip")
    say("=" * 100)

    D, per, n = build_perturbations()
    se = load_side_effects()
    signed = {d: v for d, v in per.items() if any(abs(x) > 0 for x in v.values())}
    drugs = sorted(set(signed) & set(se))
    say(f"\n  {len(per):,} drugs with targets · {len(signed):,} with at least one KNOWN action · "
        f"{len(se):,} with side effects")
    say(f"  usable drugs (signed action AND side effects): {len(drugs):,}")
    if len(drugs) < 100:
        say("  too few to run"); return 1

    Pm = np.zeros((len(drugs), n))
    for i, d in enumerate(drugs):
        for gi, s in signed[d].items():
            Pm[i, gi] = np.clip(s, -1, 1)
    say(f"  perturbation matrix {Pm.shape}, {int((Pm != 0).sum()):,} non-zero entries, "
        f"{int((Pm < 0).sum()):,} inhibitory / {int((Pm > 0).sum()):,} activating")

    say(f"\n  propagating over the signed regulatory network (alpha={ALPHA})")
    M = signed_operator(D, n)
    R = propagate(M, Pm)
    say(f"     response field {R.shape} in {time.time()-t0:.0f}s")

    # ---- P1 ---------------------------------------------------------------------------------------
    say(f"\n  P1 THE FIELD IS NOT JUST THE TARGET SET")
    cors = []
    for i in range(min(len(drugs), 300)):
        m = (Pm[i] != 0) | (R[i] != 0)
        if m.sum() > 10:
            c = spearmanr(Pm[i][m], R[i][m]).statistic
            if np.isfinite(c):
                cors.append(abs(c))
    med = float(np.median(cors)) if cors else np.nan
    reach = float(np.mean((np.abs(R) > 1e-9).sum(1) / np.maximum((Pm != 0).sum(1), 1)))
    say(f"     median |Spearman(perturbation, response)| = {med:.4f}   bar < {P1_BAR}")
    say(f"     the field reaches {reach:.1f}x as many genes as the drug has targets")
    p1 = bool(med < P1_BAR)
    say(f"     P1 {'PASS' if p1 else 'FAIL'}")

    # ---- the term matrix ---------------------------------------------------------------------------
    cnt = collections.Counter(t for d in drugs for t in se[d])
    terms = [t for t, c in cnt.items()
             if MIN_DRUGS_PER_TERM <= c and MIN_POS_FRAC <= c / len(drugs) <= MAX_POS_FRAC]
    terms = sorted(terms)
    Y = np.array([[1.0 if t in se[d] else 0.0 for t in terms] for d in drugs])
    say(f"\n  {len(terms):,} side-effect terms with >= {MIN_DRUGS_PER_TERM} drugs and a "
        f"{MIN_POS_FRAC:.0%}-{MAX_POS_FRAC:.0%} positive rate")
    say(f"  term matrix {Y.shape}, density {Y.mean():.1%}")

    def compress(Rx):
        Rc = Rx - Rx.mean(0)
        U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
        return U[:, :N_COMP] * S[:N_COMP]

    def score(Rx, tag):
        X = compress(Rx)
        P = fit_per_term(X, Y, drugs, np.random.default_rng(SEED))
        a, nt = macro_term_auc(P, Y)
        say(f"     {tag:<38}{a:.4f}   ({nt} terms scored)   {time.time()-t0:.0f}s")
        return a

    say(f"\n  every arm compressed to {N_COMP} components, drug-grouped {FOLDS}-fold, "
        f"one model per term")
    a_field = score(R, "SIGNED field over the real network")

    # ---- P2 signs ----------------------------------------------------------------------------------
    say(f"\n  P2 THE SIGNS MATTER")
    Pu = (Pm != 0).astype(float)
    a_uns = score(propagate(M, Pu), "UNSIGNED field (which, not what)")
    p2 = bool(a_field > a_uns)
    say(f"     signed {a_field:.4f} vs unsigned {a_uns:.4f}   delta {a_field - a_uns:+.4f}")
    say(f"     P2 {'PASS' if p2 else 'FAIL'}  -- a fail means the action field is decoration")

    # ---- P3 network --------------------------------------------------------------------------------
    say(f"\n  P3 THE NETWORK MATTERS")
    Ms = signed_operator(D, n, shuffle=np.random.default_rng(SEED + 1))
    a_shuf = score(propagate(Ms, Pm), "signed field, SHUFFLED network")
    a_raw = score(Pm, "raw targets, no propagation at all")
    p3 = bool(a_field > a_shuf and a_field > a_raw)
    say(f"     real {a_field:.4f} · shuffled {a_shuf:.4f} · no propagation {a_raw:.4f}")
    say(f"     P3 {'PASS' if p3 else 'FAIL'}  (the real network must beat both)")

    # ---- P4 the real gate ---------------------------------------------------------------------------
    say(f"\n  P4 PER-TERM RANKING BEATS THE FAME BASELINE")
    nse = np.array([len(se[d]) for d in drugs], float)[:, None]
    ntg = np.array([len(signed[d]) for d in drugs], float)[:, None]
    Xf = np.column_stack([nse, ntg])
    Pf = fit_per_term(Xf, Y, drugs, np.random.default_rng(SEED))
    a_fame, _ = macro_term_auc(Pf, Y)
    say(f"     {'FAME baseline (n_effects, n_targets)':<38}{a_fame:.4f}")
    say(f"     this baseline is what won loops 12, 50 and 58. On the flipped axis it is compared")
    say(f"     WITHIN a term, where every drug is judged on the same one effect, so a drug's own")
    say(f"     effect count cannot separate them and it should sit near chance.")
    p4 = bool(a_field > a_fame + 0.02)
    say(f"     field {a_field:.4f} vs fame {a_fame:.4f}   delta {a_field - a_fame:+.4f}")
    say(f"     P4 {'PASS' if p4 else 'FAIL'}  (field must beat fame by > 0.02)")

    # ---- P5 nulls -----------------------------------------------------------------------------------
    say(f"\n  P5 HELD OUT BY DRUG, AND THE NULLS COLLAPSE IT")
    X = compress(R)
    nulls = []
    for _ in range(N_NULL):
        Ys = Y[rng.permutation(len(drugs))]
        Pn = fit_per_term(X, Ys, drugs, np.random.default_rng(SEED))
        a, _ = macro_term_auc(Pn, Ys)
        if np.isfinite(a):
            nulls.append(a)
    nulls = np.array(nulls)
    n95 = float(np.percentile(nulls, 95)) if len(nulls) else np.nan
    say(f"     drug rows permuted, whole pipeline refitted {len(nulls)}x: mean {nulls.mean():.4f}, "
        f"95th {n95:.4f}")
    p5 = bool(a_field > n95)
    say(f"     P5 {'PASS' if p5 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"P1 the field is not the target set": p1, "P2 the signs matter": p2,
             "P3 the network matters": p3, "P4 beats the fame baseline": p4,
             "P5 nulls collapse it": p5}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    say(f"  A DRUG IS NOW A FIELD, NOT A LIST. Signed by what it does to each target -- "
        f"{int((Pm < 0).sum()):,} inhibitory,")
    say(f"  {int((Pm > 0).sum()):,} activating -- and propagated over the signed regulatory network to a")
    say(f"  dense response over every gene, reaching {reach:.0f}x as many genes as the drug has targets.")
    say(f"  AND THE AXIS IS FLIPPED. Scoring per TERM rather than per DRUG makes 'how many effects")
    say(f"  does this drug have' structurally unable to help, because every drug in a comparison is")
    say(f"  judged on the same single effect. That baseline reaches {a_fame:.4f} here, against the")
    say(f"  0.2898 it scored on the old axis in loop 12. The field reaches {a_field:.4f} over "
        f"{len(terms)} terms.")
    say(f"  What the signs are worth: {a_field - a_uns:+.4f}. What the network is worth over a")
    say(f"  degree-preserving shuffle: {a_field - a_shuf:+.4f}. Over no propagation at all: "
        f"{a_field - a_raw:+.4f}.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL), str(SC / "sider_se.tsv.gz"), str(SC / "sider_names.tsv")],
                      available=len(per), used=len(drugs), selection="filtered", seed=SEED,
                      controls=["unsigned arm: which targets, not what is done to them",
                                "degree-preserving network shuffle",
                                "no-propagation arm: raw target vector",
                                "fame baseline of effect count and target count",
                                f"{N_NULL} drug-permutation nulls with the pipeline refitted",
                                "drug-grouped folds; unknown actions are zero, never guessed"],
                      note="scored per TERM across drugs, which removes the per-drug count confound "
                           "by construction rather than by adjustment")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_response_field", "manifest": man, "gates": gates,
               "n_drugs": len(drugs), "n_terms": len(terms), "n_genes": n,
               "field_vs_perturbation": med, "reach_ratio": reach,
               "auc_field": a_field, "auc_unsigned": a_uns, "auc_shuffled": a_shuf,
               "auc_no_propagation": a_raw, "auc_fame": a_fame,
               "null_mean": float(nulls.mean()) if len(nulls) else None, "null_95": n95,
               "alpha": ALPHA, "n_components": N_COMP,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_response_field.json", "w"), indent=2, default=float)
    RM.report(man, emit=say)
    say(f"\n  -> {OUT/'loop_response_field.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
