"""LOOP 53 -- FIT THE PHYSICS. Does energy carry anything geometry does not?

WHAT LOOP 52 LEFT. A force field with every weight pinned at 1.0 reached 0.6268 on nexus's own
interface-breaker task, and the ablation ladder said why it should not have: Lennard-Jones ALONE was
0.7446 and each further term made it worse. Four quantities carrying different units were being added
as though they were the same quantity. That is a modelling error, not a fact about the physics.

And the number to beat was not nexus's. It was 0.7765 -- COUNTING ATOM PAIRS within 4.5 A of the
partner chain. No energy, no force field, no fitting. Pure geometry beat the force field, beat a
fitted baseline stack, and beat nexus's own recorded 0.755 and rebuilt 0.770.

SO THE QUESTION THIS LOOP ASKS IS NARROWER AND HARDER THAN LOOP 52'S. Not "can physics do this" --
loop 52 answered that badly and honestly. The question is:

    ONCE THE TERMS ARE WEIGHTED PROPERLY, DOES THE ENERGY KNOW ANYTHING THAT COUNTING DOES NOT?

WHAT CHANGES, AND WHAT IT COSTS. The weights are now FITTED, so the parameter-free claim of loop 52
is gone and is not being smuggled back. What replaces it is a harder standard: fitting must be shown
to have learned physics rather than complex identity, and F2 exists to try to prove it did not.

THE TERMS, split so that "hindrance" and "attraction" are separable rather than summed in the dark.
The Lennard-Jones sum EPS*(x^12 - 2x^6) is two physically different things sharing one number:

    d_lj_rep    EPS * x^12          STERIC HINDRANCE -- the new side chain does not fit
    d_lj_att    EPS * (-2 x^6)      PACKING -- the new side chain touches less, or more
    d_elec      Coulomb with eps(r)=r
    d_induc     charge-induced-dipole polarisation
    d_desolv    Eisenberg-McLachlan atomic solvation
    contacts    partner atoms within 4.5 A of the WILD-TYPE side chain   <- the 0.7765 baseline
    frac_lost   the FRACTION of those contacts the mutation destroys

Splitting the LJ is the one modelling change made on the strength of loop 52's ladder, and it is
made for a stated physical reason rather than because it helps: a mutation that clashes and a
mutation that loses contact are different failures, and a single signed sum cannot express both.

PREDECLARED, before any number:

  F1 THE FIT IS HELD OUT BY COMPLEX, AND THE GAP IS REPORTED
       weights fitted with complex-grouped 5-fold; no complex appears in both train and test. TRAIN
       and TEST are both printed. Train ~ test means nothing was fitted; train >> test means it was
       fitted to the wrong thing. Neither is failed on -- the number is reported so the reader can
       see which regime this is in.
  F2 THE NULL COLLAPSES IT   THE GATE THAT MAKES THE REST MEAN ANYTHING.
       the ENTIRE pipeline -- refit and all -- is re-run on labels permuted WITHIN complex, 200
       times. Held-out AUC must fall to the 0.5 band. 277 complexes, a median of 4 mutations each,
       and a fitted model is exactly the setting where "which complex is this" gets learned and
       reported as biology. If the null does not collapse, nothing else in this file is evidence.
  F3 FITTED PHYSICS BEATS THE UNWEIGHTED SUM
       0.6268 is the bar. If weighting does not beat not-weighting, the fit is broken.
  F4 FITTED PHYSICS BEATS GEOMETRY ALONE   THE REAL QUESTION.
       energy terms ONLY -- no geometry feature at all -- against the 0.7765 contact count. This is
       the gate that asks whether a force field knows anything a tape measure does not.
  F5 PHYSICS ADDS ON TOP OF GEOMETRY
       contacts + energy against contacts alone, complex-grouped, paired per complex. F4 can fail
       while F5 passes: the energy might be weaker than counting and still carry something counting
       misses. Both are asked because they are different claims.
  F6 EVERY TERM EARNS ITS PLACE UNDER FITTING
       leave-one-term-out: drop each feature, refit, measure the held-out loss. A term whose removal
       costs nothing is reported as contributing nothing, no matter how physical it is.
  F7 IT BEATS nexus's OWN NUMBER ON nexus's OWN TASK
       0.770, loop 32's rebuild, which is the highest number this repository has recorded for the
       thing nexus was built to do.

-> outputs/loop_fitted_bind.json
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flex_physics as F
import loop_physics_bind as PB
import run_manifest as RM

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CACHE = SC / "_fitted_bind_feats.pkl"

BREAKER = 2.0
NEXUS_REBUILT = 0.770
UNWEIGHTED = 0.6268     # loop 52's declared sum, the F3 bar
GEOMETRY = 0.7765       # loop 52's contact count, the F4 bar
N_NULL = 200
FOLDS = 5
SEED = 1904

ENERGY = ["d_lj_rep", "d_lj_att", "d_elec", "d_induc", "d_desolv"]
# GEOMETRY. The first version paired `contacts` with `d_contacts` (mutant minus wild-type count) and
# the two turned out to correlate at -0.998 -- they are one feature and its negation. Two collinear
# columns give a logistic fit an unidentifiable direction, the coefficients swing between folds, and
# the held-out score came out at 0.5388 for a pair whose better half alone scores 0.7722. Replaced
# with the BOUNDED ratio: what FRACTION of the wild-type's partner contacts the mutation destroys.
# Same information, one direction, and it cannot be collinear with the count it is divided by.
GEOM = ["contacts", "frac_lost"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def lj_split(ca, ra, cb, rb, cutoff=6.0, cap=10.0):
    """Lennard-Jones separated into its repulsive and attractive halves.

    THE FIRST VERSION OF THIS FUNCTION BLEW UP AND THE FEATURE TABLE SHOWED IT. flex_physics._lj_sum
    caps the COMBINED pair energy at `cap`; splitting the sum and capping only the repulsive half let
    the attractive half run free, and since a rebuilt rotamer can sit almost on top of a partner atom,
    x = rmin/d goes large and -2*EPS*x^6 goes with it. d_lj_att came out spanning -1.1e6 to +3.9e7
    with a standard deviation of 8.2e5. That is not a packing energy, it is a divide-by-almost-zero,
    and a fitted model handed a feature like that will chase it.

    The fix is the soft core done properly: clip x at the value where the REPULSION alone reaches the
    cap, x_max = (cap/EPS)^(1/12). Below that separation the pair is simply "clashing" and both halves
    saturate together, which is what a soft-core potential means and what the unsplit function was
    already doing implicitly.
    """
    if len(ca) == 0 or len(cb) == 0:
        return 0.0, 0.0
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
    rmin = ra[:, None] + rb[None, :]
    m = (d < cutoff) & (d > 0.1)
    if not m.any():
        return 0.0, 0.0
    x = np.minimum(rmin[m] / d[m], (cap / F.EPS) ** (1.0 / 12.0))
    return float((F.EPS * x ** 12).sum()), float((F.EPS * (-2 * x ** 6)).sum())


def terms(t, ch, pos, coords=None, names=None, wt=None):
    """All across-interface terms for one side chain, with LJ split and contacts counted."""
    scm = (t["ch"] == ch) & (t["rn"] == pos) & ~np.isin(t["nm"], list(F.BB) + ["CB"])
    if coords is None:
        if not scm.any():
            return None
        scc, scn = t["co"][scm], t["nm"][scm]
    else:
        scc, scn = coords, names
    zero = {"lj_rep": 0.0, "lj_att": 0.0, "elec": 0.0, "induc": 0.0, "desolv": 0.0, "nc": 0.0}
    if len(scc) == 0:
        return zero
    other = t["ch"] != ch
    if not other.any():
        return zero
    d0 = np.linalg.norm(t["co"][:, None, :] - scc[None, :, :], axis=2).min(1)
    near = other & (d0 < 12.0)
    if not near.any():
        return zero
    el = np.array([str(n)[0] for n in scn])
    rad = F._RAD(el)
    qa = F._charge_res(wt if wt else "", scn) if wt else np.zeros(len(scn))
    qb = np.array([F._charge_name(n) for n in t["nm"][near]])
    rep, att = lj_split(scc, rad, t["co"][near], t["rad"][near])
    dd = np.linalg.norm(scc[:, None, :] - t["co"][near][None, :, :], axis=2)
    return {"lj_rep": rep, "lj_att": att,
            "elec": F._elec(scc, qa, t["co"][near], qb),
            "induc": F._induction(scc, el, t["co"][near], qb),
            "desolv": F._desolv(scc, el, scn, t["co"][near]),
            "nc": float((dd < 4.5).sum())}


def build_features():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    muts = F.load_pdb_muts()
    rn3 = {}
    recs = []
    for r in muts:
        t = F.table(r["pdb"])
        if t is None:
            continue
        if r["pdb"] not in rn3:
            rn3[r["pdb"]] = PB.build_table_with_resnames(r["pdb"]) or {}
        got = rn3[r["pdb"]].get((r["chain"], r["pos"]))
        if got is None or PB.AA3TO1.get(got) != r["wt"]:
            continue
        w = terms(t, r["chain"], r["pos"], wt=got)
        if w is None:
            continue
        mb = PB.mutant_sidechain(t, r["chain"], r["pos"], r["mut"])
        if mb is None:
            continue
        m = terms(t, r["chain"], r["pos"], coords=mb[0], names=mb[1],
                  wt=F.AA3.get(r["mut"], ""))
        if m is None:
            continue
        recs.append({
            "pdb": r["pdb"], "ddg": r["ddg"], "wt": r["wt"], "mut": r["mut"], "loc": r["loc"],
            "d_lj_rep": m["lj_rep"] - w["lj_rep"], "d_lj_att": m["lj_att"] - w["lj_att"],
            "d_elec": m["elec"] - w["elec"], "d_induc": m["induc"] - w["induc"],
            "d_desolv": m["desolv"] - w["desolv"],
            "contacts": w["nc"],
            "frac_lost": (w["nc"] - m["nc"]) / max(w["nc"], 1.0),
        })
    pickle.dump(recs, open(CACHE, "wb"))
    return recs


def fit_predict(X, y, grp, rng, folds=FOLDS, ret_train=False):
    """Complex-grouped logistic regression, returning out-of-fold predictions."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ug = np.unique(grp)
    fold = {g: i for g, i in zip(ug, rng.permutation(len(ug)) % folds)}
    fid = np.array([fold[g] for g in grp])
    oof = np.full(len(y), np.nan)
    trn = np.full(len(y), np.nan)
    for k in range(folds):
        tr, te = fid != k, fid == k
        if tr.sum() < 40 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        # RANK-TRANSFORM, fitted on train only, and the reason is the third bug this loop found in
        # itself. `contacts` runs 0 to 10,278 with a standard deviation of 378: after standardising,
        # almost every value sits in a sliver near zero and a handful are enormous. L2 logistic
        # regression shrinks that coefficient toward nothing to avoid the outliers -- measured at
        # +0.046, -0.058, -0.035, -0.063, -0.052 across the five folds, on a feature whose own AUC is
        # 0.74 to 0.80 in those same folds. The fitted arm scored 0.4196 against 0.7722 unfitted.
        # AUC IS RANK-BASED AND LOGISTIC REGRESSION IS VALUE-BASED, so a monotone but wildly
        # nonlinear feature can rank almost perfectly and still be useless to a linear fit. Comparing
        # a raw feature's AUC against a fitted linear model's AUC is not the same comparison unless
        # the fit is given the ranks. Empirical CDF from the TRAINING rows only, applied to test by
        # interpolation -- monotone, so it changes no feature's own AUC, and leak-free.
        A = np.empty_like(X[tr])
        B = np.empty_like(X[te])
        for j in range(X.shape[1]):
            srt = np.sort(X[tr, j])
            A[:, j] = np.searchsorted(srt, X[tr, j], side="right") / max(len(srt), 1)
            B[:, j] = np.searchsorted(srt, X[te, j], side="right") / max(len(srt), 1)
        m = LogisticRegression(max_iter=5000, C=1.0).fit(A, y[tr])
        oof[te] = m.decision_function(B)
        if ret_train:
            v = m.decision_function(A)
            trn[tr] = np.where(np.isnan(trn[tr]), v, trn[tr])
    return (oof, trn) if ret_train else oof


def per_complex_auc(v, y, grp, min_each=3):
    out = {}
    v = np.asarray(v, float)
    for g in np.unique(grp):
        m = (grp == g) & np.isfinite(v)
        if (y[m] == 1).sum() >= min_each and (y[m] == 0).sum() >= min_each:
            out[g] = PB.auc(v[m], y[m])
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 53 -- fit the physics: does energy know anything geometry does not?")
    say("=" * 100)

    recs = build_features()
    y = (np.array([r["ddg"] for r in recs]) > BREAKER).astype(int)
    ddg = np.array([r["ddg"] for r in recs])
    grp = np.array([r["pdb"] for r in recs])
    col = lambda k: np.array([r[k] for r in recs], float)
    say(f"\n  {len(recs):,} mutations, {len(set(grp)):,} complexes, {y.sum():,} breakers "
        f"({y.mean():.1%}) at ddG_bind > {BREAKER}")
    say(f"  features: {' '.join(ENERGY)} | {' '.join(GEOM)}")

    def X(cols):
        return np.column_stack([col(c) for c in cols])

    # ---- F1 ---------------------------------------------------------------------------------------
    say(f"\n  F1 THE FIT IS HELD OUT BY COMPLEX, AND THE GAP IS REPORTED")
    oof_all, trn_all = fit_predict(X(ENERGY + GEOM), y, grp, rng, ret_train=True)
    a_train, a_test = PB.auc(trn_all, y), PB.auc(oof_all, y)
    say(f"     full model (energy + geometry)   TRAIN {a_train:.4f}   HELD-OUT {a_test:.4f}   "
        f"gap {a_train - a_test:+.4f}")
    f1 = bool(np.isfinite(a_test))
    say(f"     F1 {'PASS' if f1 else 'FAIL'}  (reported, not gated on a threshold)")

    # ---- F2 the null ------------------------------------------------------------------------------
    say(f"\n  F2 THE NULL COLLAPSES IT")
    say(f"     the ENTIRE pipeline is refitted on labels permuted WITHIN complex, {N_NULL} times.")
    idx = {g: np.where(grp == g)[0] for g in set(grp)}
    Xfull = X(ENERGY + GEOM)
    nulls = []
    for i in range(N_NULL):
        ys = y.copy()
        for g, ii in idx.items():
            ys[ii] = rng.permutation(y[ii])
        nulls.append(PB.auc(fit_predict(Xfull, ys, grp, rng), ys))
    nulls = np.array([x for x in nulls if np.isfinite(x)])
    n95 = float(np.percentile(nulls, 95))
    say(f"     null mean {nulls.mean():.4f}   95th {n95:.4f}   observed {a_test:.4f}")
    f2 = bool(a_test > n95 and abs(nulls.mean() - 0.5) < 0.05)
    say(f"     F2 {'PASS' if f2 else 'FAIL'}  (observed above the 95th AND the null centred on 0.5)")

    # ---- F3/F4/F5 ---------------------------------------------------------------------------------
    say(f"\n  THE THREE COMPARISONS")
    oof_e = fit_predict(X(ENERGY), y, grp, rng)
    oof_g = fit_predict(X(GEOM), y, grp, rng)
    a_e, a_g = PB.auc(oof_e, y), PB.auc(oof_g, y)
    contacts_raw = PB.auc(col("contacts"), y)
    say(f"     {'arm':<30}{'held-out AUC':>14}")
    for lab, v in (("contacts alone, unfitted", contacts_raw), ("geometry fitted", a_g),
                   ("ENERGY fitted (no geometry)", a_e), ("energy + geometry fitted", a_test),
                   ("loop 52 unweighted sum", UNWEIGHTED), ("nexus rebuilt", NEXUS_REBUILT)):
        say(f"     {lab:<30}{v:>14.4f}")

    say(f"\n  F3 FITTED PHYSICS BEATS THE UNWEIGHTED SUM")
    f3 = bool(a_e > UNWEIGHTED)
    say(f"     {a_e:.4f} vs {UNWEIGHTED:.4f}   F3 {'PASS' if f3 else 'FAIL'}")

    say(f"\n  F4 FITTED PHYSICS BEATS GEOMETRY ALONE")
    f4 = bool(a_e > GEOMETRY)
    say(f"     energy-only {a_e:.4f} vs the contact count {GEOMETRY:.4f}   "
        f"F4 {'PASS' if f4 else 'FAIL'}")

    say(f"\n  F5 PHYSICS ADDS ON TOP OF GEOMETRY")
    pc_full, pc_geom = per_complex_auc(oof_all, y, grp), per_complex_auc(oof_g, y, grp)
    keys = sorted(set(pc_full) & set(pc_geom))
    d = np.array([pc_full[k] - pc_geom[k] for k in keys]) if keys else np.array([])
    if len(d) >= 10:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        pw = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) >= 10 else np.nan
    else:
        pw = np.nan
    say(f"     pooled  energy+geometry {a_test:.4f}  vs  geometry {a_g:.4f}   "
        f"delta {a_test - a_g:+.4f}")
    say(f"     paired over {len(keys)} complexes: mean delta "
        f"{np.mean(d) if len(d) else float('nan'):+.4f}, wins "
        f"{np.mean(d > 0) if len(d) else float('nan'):.1%}, Wilcoxon p {pw:.3e}")
    f5 = bool(a_test > a_g and np.isfinite(pw) and pw < 0.05)
    say(f"     F5 {'PASS' if f5 else 'FAIL'}")

    # ---- F6 ---------------------------------------------------------------------------------------
    say(f"\n  F6 EVERY TERM EARNS ITS PLACE UNDER FITTING")
    say(f"     leave-one-out: drop the term, refit, measure what the held-out score loses")
    loo = {}
    for c in ENERGY + GEOM:
        rest = [x for x in ENERGY + GEOM if x != c]
        a = PB.auc(fit_predict(X(rest), y, grp, rng), y)
        loo[c] = {"without": a, "loss": a_test - a}
        say(f"     without {c:<12}{a:.4f}   loss {a_test - a:+.4f}"
            f"{'   <- earns it' if a_test - a > 0.002 else ''}")
    earned = [c for c, v in loo.items() if v["loss"] > 0.002]
    f6 = bool(len(earned) >= 4)
    say(f"     F6 {'PASS' if f6 else 'FAIL'}  ({len(earned)} of {len(ENERGY+GEOM)} terms cost more "
        f"than 0.002 to remove)")

    # ---- F7 ---------------------------------------------------------------------------------------
    say(f"\n  F7 IT BEATS nexus's OWN NUMBER ON nexus's OWN TASK")
    f7 = bool(a_test > NEXUS_REBUILT)
    say(f"     {a_test:.4f} vs {NEXUS_REBUILT:.3f}   F7 {'PASS' if f7 else 'FAIL'}")

    # regression, reported not gated
    rho_full = float(spearmanr(oof_all[np.isfinite(oof_all)], ddg[np.isfinite(oof_all)]).statistic)
    rho_geom = float(spearmanr(col("contacts"), ddg).statistic)
    say(f"\n     on the CONTINUOUS binding free energy (reported, not gated):")
    say(f"       fitted model Spearman {rho_full:+.4f}   ·   contact count {rho_geom:+.4f}   ·   "
        f"loop 52's unweighted sum +0.2209")

    say("\n" + "=" * 100)
    gates = {"F1 fit held out by complex": f1, "F2 the null collapses it": f2,
             "F3 beats the unweighted sum": f3, "F4 beats geometry alone": f4,
             "F5 physics adds on top of geometry": f5, "F6 every term earns its place": f6,
             "F7 beats nexus's own number": f7}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    say(f"  WEIGHTING THE TERMS TAKES THE FORCE FIELD FROM {UNWEIGHTED:.4f} TO {a_e:.4f} ON ENERGY")
    say(f"  ALONE, and the whole model to {a_test:.4f} held out by complex against a within-complex")
    say(f"  null of {n95:.4f}. Counting atoms, unfitted, is {GEOMETRY:.4f}.")
    if f4:
        say(f"  SO THE ENERGY DOES KNOW SOMETHING THE TAPE MEASURE DOES NOT -- fitted physics beats")
        say(f"  the count outright, and loop 52's failure was the unweighted sum, not the force field.")
    else:
        say(f"  THE ENERGY STILL DOES NOT BEAT THE TAPE MEASURE ON ITS OWN ({a_e:.4f} vs "
            f"{GEOMETRY:.4f}), which")
        say(f"  is the third time counting has won here. What F5 asks separately is whether it adds")
        say(f"  anything on top, and that answer is {'yes' if f5 else 'no'}.")
    say(f"  The parameter-free claim of loop 52 is GONE and is not being smuggled back: these weights")
    say(f"  were fitted, F1 prints the train-test gap, and F2 refits the entire pipeline under "
        f"{N_NULL} within-complex")
    say(f"  label permutations to show what the fitting could have manufactured.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "skempi_v2.csv"), str(SC / "pdb_cache")],
                      available=len(recs), used=len(recs), selection="all", seed=SEED,
                      controls=[f"{N_NULL} within-complex label permutations with the model REFITTED "
                                f"under each",
                                "complex-grouped 5-fold; no complex in both train and test",
                                "train and held-out AUC both reported",
                                "leave-one-term-out refit for every feature",
                                "energy-only arm carries no geometry feature"],
                      note="the parameter-free claim of loop 52 does not apply here; these weights "
                           "are fitted and the controls exist to bound what that bought")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_fitted_bind", "manifest": man, "gates": gates,
               "n_mutations": len(recs), "n_complexes": int(len(set(grp))),
               "auc_train": a_train, "auc_heldout": a_test, "auc_energy_only": a_e,
               "auc_geometry_fitted": a_g, "auc_contacts_raw": contacts_raw,
               "null_mean": float(nulls.mean()), "null_95": n95,
               "leave_one_out": loo, "spearman_fitted": rho_full, "spearman_contacts": rho_geom,
               "bars": {"unweighted": UNWEIGHTED, "geometry": GEOMETRY, "nexus": NEXUS_REBUILT},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_fitted_bind.json", "w"), indent=2, default=float)
    say(f"\n  -> {OUT/'loop_fitted_bind.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
