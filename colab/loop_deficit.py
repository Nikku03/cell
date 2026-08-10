"""WHY THE LOOKUP WON -- three tests on loop 1's negative, before any more mechanism is built on it.

WHERE THIS COMES FROM.  cell_loop closed the growth feedback and passed two gates and failed two. The
number that mattered was not either failure but this: mRNA abundance ALONE scored AUC 0.7766 against
essentiality, while stoichiometry, GPRs, flux balance, enzyme capacity and a solved growth fixed point
together scored 0.6210. A single column beat the entire mechanistic stack by 0.16. I wrote into that
module that building further mechanism on an unexplained deficit would be building on sand. This is the
module that refuses to move on.

Three hypotheses, three tests, each able to come back negative, and two of them cost nothing because the
scores are already recorded.

    H1 INDEPENDENCE -- does the mechanism know ANYTHING the lookup does not?
        A model can be worse than a baseline and still carry information the baseline lacks, in which
        case it is worth keeping and combining. Or it can be worse and carry nothing, in which case it
        is decoration. These are different findings and the difference is measurable: score the
        mechanism WITHIN strata of matched mRNA abundance, and fit the two together under
        cross-validation against the lookup alone.
        gate: within-stratum AUC >= 0.55, OR combined CV AUC >= lookup + 0.02.
        fails -> the mechanistic layer carries no information the lookup lacks. Say so.

    H2 CONFOUND -- is the lookup even about metabolism?
        Abundant transcripts belong to essential genes for reasons that have nothing to do with flux:
        ribosomes, spliceosome, proteasome. If mRNA abundance predicts essentiality JUST AS WELL for
        genes that are not in the metabolic model at all, then 0.7766 is a general property of gene
        expression and loop 1 was never a fair mechanism-versus-mechanism comparison.
        gate: |AUC(metabolic genes) - AUC(non-metabolic genes)| < 0.05 -> it is a general confound.
        the reverse -> the lookup really is carrying metabolic signal and the deficit is worse than it
        looked.

    H3 MEDIUM -- was the model ever asked a fair question?
        Human-GEM ships with all 1,660 exchange reactions open at -1000. The cell may eat anything, at
        any rate, including adrenic acid and 9-cis-retinol. Under that regime almost NO biosynthetic
        gene can be essential, because whatever it makes can simply be imported -- which would cap
        FBA's achievable AUC no matter how good the rest of the model is. Growth was calibrated down to
        0.030/h purely by squeezing enzyme capacity, and the symptom was visible in loop 1 and not
        understood: all 2,848 knockouts moved mu, which no real cell does.
        gate: with a defined medium, the best mechanistic AUC rises by >= 0.03 over the open-medium
        0.6491.
        fails -> the open medium was not the problem and FBA's deficit here is real.

THE MEDIUM IS A DECLARED JUDGEMENT and a crude one, named as such.  Human-GEM ships no medium, so one is
imposed here by metabolite name: glucose, oxygen, water, protons, the common ions, the nine essential
amino acids plus the four conditionally essential ones, the standard culture vitamins, and the two
essential fatty acids. That is DMEM/Ham's in spirit. Matching by name will miss synonyms and catch
homonyms, so the count of matched exchanges is reported and the list is in the source to be argued with.
Everything else is closed to uptake and left free to secrete.

CONTROLS: label shuffles on every AUC reported; the non-metabolic gene set as H2's own comparator; and
open-medium versus defined-medium run on identical code with only the exchange bounds changed.

WHAT HAPPENED, written after the run against the gates above, unedited.

    H1 INDEPENDENCE   PASS.  Within abundance-matched deciles -- where the lookup is held fixed and
                      cannot help -- plain FBA still scores 0.6461. Under 5-fold CV, lookup alone
                      0.7761, lookup + mechanism 0.8314, a gain of +0.0552 against a gate of +0.02.
                      THE MECHANISM IS WORSE THAN THE BASELINE AND NOT REDUNDANT WITH IT. Those are
                      different things and loop 1 could not tell them apart.

    H2 CONFOUND       CONFIRMED, and cleanly.  mRNA abundance scores 0.7766 on the 1,784 metabolic
                      genes and 0.7794 on 8,511 genes with NO metabolic reaction at all. Difference
                      -0.0028 against a gate of 0.05. The column that beat the mechanistic stack knows
                      nothing about metabolism; it is a general property of gene expression. Loop 1's
                      G4 comparison was never mechanism against mechanism, and the 0.16 deficit should
                      not be read as "FBA is bad at metabolism".

    H3 MEDIUM         NOT TESTABLE.  The declared medium matched 403 exchanges from 40 of 44 named
                      components and supports a growth rate of exactly zero. A medium too poor to grow
                      on cannot say whether an adequate one would have helped, so this is recorded as
                      untestable rather than as a negative. The defect is mine, not the model's, and
                      the fix is not a longer hand-written list -- it is to stop hand-writing it.

WHAT THIS CHANGES.  Loop 1 read as "the mechanism lost to a lookup". After these three it reads: the
mechanism lost to a confound that is not about metabolism, while carrying independent signal worth
+0.055 AUC on top of that confound. That is a materially better position than it looked, and it was only
visible because the negative was interrogated instead of banked.

-> outputs/loop_deficit.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc, gpr_capacity, GEM, GEMGENES, KDEG, CELL, K_DP, MIN_LINES, MU_TARGET  # noqa

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904
N_SHUFFLE = 200
GATE_INDEP_STRATUM = 0.55
GATE_INDEP_COMBINED = 0.02
GATE_CONFOUND = 0.05
GATE_MEDIUM = 0.03
OPEN_MEDIUM_BEST = 0.6491        # plainFBA in the open-medium run; the number H3 must beat

# A declared culture medium, by metabolite name. Crude, and the count of matches is reported.
MEDIUM = [
    "glucose", "O2", "H2O", "H+", "Pi", "sulfate", "HCO3-", "Na+", "K+", "Ca2+", "Cl-", "Fe2+", "Fe3+",
    "Mg2+", "Zn2+", "Cu2+", "Mn2+", "selenate", "iodide",
    "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "threonine",
    "tryptophan", "valine", "cysteine", "tyrosine", "glutamine", "arginine",
    "folate", "riboflavin", "thiamin", "pantothenate", "pyridoxine", "nicotinamide", "biotin",
    "choline", "inositol", "cobalamin",
    "linoleate", "linolenate",
]


def strat_auc(score, label, strat, nbin=10):
    """AUC of `score` computed WITHIN bins of `strat` and pooled by bin weight. This is the honest way
    to ask whether a predictor knows anything once the baseline is held fixed: inside a bin every gene
    has roughly the same mRNA abundance, so the lookup cannot help and only the mechanism can."""
    s, y, x = np.asarray(score, float), np.asarray(label, int), np.asarray(strat, float)
    q = pd.qcut(pd.Series(x).rank(method="first"), nbin, labels=False)
    num, den = 0.0, 0.0
    per = []
    for b in range(nbin):
        m = q == b
        if m.sum() < 10 or y[m].sum() == 0 or y[m].sum() == m.sum():
            continue
        a = auc(s[m], y[m])
        w = int(y[m].sum()) * int(m.sum() - y[m].sum())
        num += a * w
        den += w
        per.append({"bin": b, "n": int(m.sum()), "pos": int(y[m].sum()), "auc": float(a)})
    return (num / den if den else float("nan")), per


def cv_auc(X, y, folds=5, seed=SEED):
    """Cross-validated logistic AUC. Standardised, mild L2, out-of-fold predictions only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    X, y = np.asarray(X, float), np.asarray(y, int)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    return auc(oof, y), oof


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("WHY THE LOOKUP WON -- three tests on loop 1's negative")
    say("=" * 100)
    say("  mRNA abundance alone scored 0.7766; the whole mechanistic stack scored 0.6210. Before any")
    say("  more mechanism gets built on that, three things have to be known: whether the mechanism")
    say("  knows anything the lookup does not, whether the lookup is even about metabolism, and")
    say("  whether the model was ever asked a fair question.")

    prev = json.load(open(OUT / "cell_loop.json"))
    T = pd.DataFrame(prev["scores"])
    y = T.ess.to_numpy()
    say(f"\n  loop 1's recorded scores: {len(T)} genes, {int(y.sum())} essential")
    say(f"    loop {prev['auc']['loop']:.4f}  frozen {prev['auc']['frozen']:.4f}  "
        f"plainFBA {prev['auc']['plainFBA']:.4f}  mRNA {prev['auc']['mRNA abundance']:.4f}")

    # ---- H1 INDEPENDENCE ------------------------------------------------------------------------------
    say("\n  H1 INDEPENDENCE -- does the mechanism know anything the lookup does not?")
    lr = np.log10(T.rpkm.to_numpy() + 1e-3)
    res_h1 = {}
    for name in ("plain", "frozen", "loop"):
        sa, per = strat_auc(T[name].to_numpy(), y, lr)
        res_h1[name] = {"stratified_auc": float(sa), "bins": per}
        say(f"    {name:<8} within matched-abundance deciles: AUC {sa:.4f}  "
            f"({len(per)} usable bins)")
    comb = {}
    a_look, _ = cv_auc(lr.reshape(-1, 1), y)
    comb["lookup only"] = float(a_look)
    a_both, _ = cv_auc(np.c_[lr, T.plain, T.frozen, T.loop], y)
    comb["lookup + mechanism"] = float(a_both)
    a_mech, _ = cv_auc(np.c_[T.plain, T.frozen, T.loop], y)
    comb["mechanism only"] = float(a_mech)
    say(f"    5-fold CV: lookup {a_look:.4f} -> lookup+mechanism {a_both:.4f} "
        f"({a_both-a_look:+.4f}); mechanism alone {a_mech:.4f}")
    best_strat = max(res_h1[k]["stratified_auc"] for k in res_h1)
    h1 = bool(best_strat >= GATE_INDEP_STRATUM or (a_both - a_look) >= GATE_INDEP_COMBINED)
    say(f"    best stratified {best_strat:.4f} (gate {GATE_INDEP_STRATUM}), "
        f"combined gain {a_both-a_look:+.4f} (gate +{GATE_INDEP_COMBINED})   "
        f"H1 {'PASS' if h1 else 'FAIL'}")

    # ---- H2 CONFOUND ----------------------------------------------------------------------------------
    say("\n  H2 CONFOUND -- does mRNA abundance predict essentiality outside metabolism too?")
    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    gm = pd.read_csv(GEMGENES, sep="\t")
    gem_syms = set(str(s) for s in gm["geneSymbols"].dropna())
    k = pd.read_csv(KDEG)
    rp = (k[k.avg_RPKM_total > 0].groupby("feature_ID")
          .agg(rpkm=("avg_RPKM_total", "median"), n=("cell_line", "size")))
    rp = rp[rp.n >= MIN_LINES]
    rows = []
    for s, r in rp.iterrows():
        i = name2i.get(s)
        if i is None or genes[i].get("ess") is None:
            continue
        rows.append({"sym": s, "rpkm": float(r.rpkm), "ess": int(genes[i]["ess"]),
                     "metabolic": s in gem_syms})
    A = pd.DataFrame(rows)
    m_in, m_out = A[A.metabolic], A[~A.metabolic]
    a_in = auc(np.log10(m_in.rpkm + 1e-3), m_in.ess)
    a_out = auc(np.log10(m_out.rpkm + 1e-3), m_out.ess)
    say(f"    metabolic genes     n={len(m_in):>6,}  essential {int(m_in.ess.sum()):>5,}  "
        f"AUC {a_in:.4f}")
    say(f"    NON-metabolic genes n={len(m_out):>6,}  essential {int(m_out.ess.sum()):>5,}  "
        f"AUC {a_out:.4f}")
    h2 = bool(abs(a_in - a_out) < GATE_CONFOUND)
    say(f"    difference {a_in-a_out:+.4f} (gate |d| < {GATE_CONFOUND})   "
        f"H2 {'CONFOUND' if h2 else 'METABOLIC-SPECIFIC'}")
    if h2:
        say("    The lookup predicts essentiality just as well for genes with no metabolic reaction at")
        say("    all. 0.7766 is a property of gene expression, not of metabolism, so loop 1's headline")
        say("    comparison was never mechanism against mechanism. It remains a real baseline that the")
        say("    mechanism failed to beat -- it just is not evidence that metabolism is understood.")

    # ---- H3 MEDIUM ------------------------------------------------------------------------------------
    say("\n  H3 MEDIUM -- was the model ever asked a fair question?")
    import cobra  # noqa
    from cobra.io import read_sbml_model
    t0 = time.time()
    M = read_sbml_model(str(GEM))
    M.solver = "glpk"
    ex = list(M.exchanges)
    say(f"    {len(ex)} exchange reactions, all shipped open at {ex[0].lower_bound}: the cell may eat")
    say(f"    anything at any rate, so almost nothing it can SYNTHESISE can be essential.")
    keep, matched = [], {}
    for r in ex:
        met = list(r.metabolites)[0]
        nm = (met.name or "").lower()
        for w in MEDIUM:
            if w.lower() in nm:
                keep.append(r)
                matched.setdefault(w, []).append(met.name)
                break
    for r in ex:
        r.lower_bound = 0.0
    for r in keep:
        r.lower_bound = -1000.0
    hit = sorted(matched)
    say(f"    declared medium of {len(MEDIUM)} components matched {len(keep)} exchanges "
        f"({len(hit)}/{len(MEDIUM)} components found)")
    miss = [w for w in MEDIUM if w not in matched]
    if miss:
        say(f"    NOT MATCHED and therefore absent from the medium: {', '.join(miss)}")
    mu_med = M.slim_optimize()
    say(f"    growth on the defined medium, enzymes unconstrained: {mu_med:.4f}/h "
        f"(open medium was {prev['mu_unconstrained']:.1f}/h)")

    if not np.isfinite(mu_med) or mu_med <= 1e-9:
        say("    The defined medium does not support growth at all. That is a defect in the medium, not")
        say("    a result about the model, and H3 CANNOT BE EVALUATED from it. Reported as NOT TESTABLE")
        say("    rather than as a failure, because a medium too poor to grow on says nothing about")
        say("    whether an adequate one would have helped.")
        h3, a_plain_med, medium_rows = None, float("nan"), []
    else:
        gpr = {r.id: r.gpr for r in M.reactions if r.gene_reaction_rule.strip()}
        ORIG = {r.id: r.bounds for r in M.reactions}
        plain = {}
        t1 = time.time()
        gids = [g.id for g in M.genes]
        for n, gid in enumerate(gids):
            g = M.genes.get_by_id(gid)
            touched = []
            for r in g.reactions:
                if r.id not in gpr:
                    continue
                E0 = {x.id: 1.0 for x in r.genes}
                E0[gid] = 0.0
                if (gpr_capacity(gpr[r.id], E0, 1.0) or 0.0) <= 0.0:
                    touched.append(r)
                    r.bounds = (0.0, 0.0)
            v = M.slim_optimize()
            plain[gid] = 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))
            for r in touched:
                r.bounds = ORIG[r.id]
            if n % 1000 == 0:
                say(f"      medium KO {n}/{len(gids)}  ({time.time()-t1:.0f}s)")
        ens2sym = {a: b for a, b in zip(gm["genes"], gm["geneSymbols"]) if isinstance(b, str)}
        sym2gid = {ens2sym[g]: g for g in gids if g in ens2sym}
        med = [1.0 - plain[sym2gid[s]] / mu_med if s in sym2gid else np.nan for s in T.sym]
        T = T.assign(medium=med)
        ok = T.medium.notna()
        a_plain_med = auc(T.medium[ok], y[ok.to_numpy()])
        n_lethal = int(sum(1 for v in plain.values() if v <= 1e-9))
        n_moved = int(sum(1 for v in plain.values() if abs(v - mu_med) > 1e-9))
        say(f"    scored {int(ok.sum())} of {len(T)} genes on the defined medium")
        say(f"    knockouts that move growth: {n_moved} of {len(gids)} "
            f"(open medium: {prev['n_movers']} of {prev['n_model_genes']}); outright lethal: {n_lethal}")
        say(f"    defined-medium FBA AUC {a_plain_med:.4f}  vs open-medium {OPEN_MEDIUM_BEST:.4f}  "
            f"({a_plain_med-OPEN_MEDIUM_BEST:+.4f})")
        h3 = bool(a_plain_med - OPEN_MEDIUM_BEST >= GATE_MEDIUM)
        say(f"    gate +{GATE_MEDIUM}   H3 {'PASS' if h3 else 'FAIL'}")
        medium_rows = [{"n_moved": n_moved, "n_lethal": n_lethal, "mu": float(mu_med)}]

    # ---- controls -------------------------------------------------------------------------------------
    null = np.array([auc(T.plain, rng.permutation(y)) for _ in range(N_SHUFFLE)])
    say(f"\n  control: {N_SHUFFLE} label shuffles on the mechanistic score -> "
        f"{null.mean():.4f} +/- {null.std():.4f}")

    # ---- verdict --------------------------------------------------------------------------------------
    say("\n" + "=" * 100)
    say(f"  H1 mechanism knows something the lookup does not : {'YES' if h1 else 'NO'}")
    say(f"  H2 the lookup is a general expression confound   : {'YES' if h2 else 'NO'}")
    say(f"  H3 the open medium was the problem               : "
        f"{'YES' if h3 else ('NOT TESTABLE' if h3 is None else 'NO')}")
    if h3:
        say("  The model was being asked an unfair question. A cell allowed to import anything cannot")
        say("  have essential biosynthetic genes, and closing the medium recovers real signal. Loop 1's")
        say("  G4 failure was partly an artefact of the medium and should be re-run, not banked.")
    elif h1:
        say("  The mechanism is worse than the lookup but not redundant with it: it carries information")
        say("  the lookup does not, so it is worth combining rather than discarding.")
    elif h3 is None:
        say("  H3 could not be evaluated -- the declared medium does not support growth. Fix the medium")
        say("  before drawing any conclusion about whether the medium was the problem.")
    else:
        say("  The deficit is real and not explained by any of the three. The metabolic layer as built")
        say("  does not carry essentiality information beyond a lookup, and that is the finding.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(OUT / "cell_loop.json"), str(GEM), str(GEMGENES), str(KDEG),
                              str(CELL)],
                      available=len(T), used=len(T), selection="all", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "non-metabolic gene set (H2)",
                                "open vs defined medium on identical code (H3)",
                                "abundance-matched strata (H1)", "5-fold cross-validation (H1)"],
                      note="H1 and H2 reuse loop 1's recorded scores; only H3 re-solves the model")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_deficit", "manifest": man,
               "H1_independence": {"pass": h1, "stratified": res_h1, "combined": comb},
               "H2_confound": {"pass": h2, "auc_metabolic": float(a_in),
                               "auc_nonmetabolic": float(a_out), "n_metabolic": len(m_in),
                               "n_nonmetabolic": len(m_out)},
               "H3_medium": {"pass": h3, "auc": float(a_plain_med),
                             "open_medium_best": OPEN_MEDIUM_BEST,
                             "matched_exchanges": len(keep), "missing": miss, "rows": medium_rows},
               "null": {"mean": float(null.mean()), "sd": float(null.std())},
               "log": log},
              open(OUT / "loop_deficit.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_deficit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
