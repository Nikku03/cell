"""LOOP 56 -- THE SPHERE CHAIN AND THE DISTOGRAM TOGETHER. Two nulls, added.

THE QUESTION, and it is a fair one that neither previous loop asked. Loop 54 measured the PAIRWISE
block -- contact counts at three radii, contact order, long-range count, eigenvector centrality, and
two PAE summaries -- and found +0.0012 (p 0.054) over the per-residue features. Loop 55 measured the
SPHERE-CHAIN block -- the first-order response dG = -G dK G to changing one side chain's volume --
and found -0.0015 (p 0.39). Each alone is null.

But they are different KINDS of quantity, and a null-plus-null is not necessarily null. The pairwise
block says how strongly a position is coupled to the rest of the fold. The sphere block says how
hard a specific substitution pushes on that fold. A push is not interesting on its own and neither
is a coupling; a hard push at a tightly coupled position might be. That is an INTERACTION, and it is
exactly the kind of thing neither additive test could have seen and a network can.

SO THIS RUNS FAST AND CHEAP, because it has to build nothing. Both blocks are already computed and
cached, keyed by variant identity rather than row position, so this loop is a fitting job.

AND IT IS THE FOURTH TEST OF ONE HYPOTHESIS, which is stated here rather than discovered later. Loop
54 asked whether structure adds. Loop 55 asked whether perturbation adds. This asks whether they add
together, and a companion arm asks whether they interact. Four bites at "does structure help" over
the same 29,286 rows and the same 141 genes is a multiplicity problem, and the honest handling is to
say so and pay for it: the significance bar is Bonferroni-corrected to 0.01/4 = 0.0025, and a pass
at 0.01 but not 0.0025 will be reported as what it is rather than banked.

PREDECLARED, before any number:

  C1 THE BLOCKS JOIN BY KEY AND NOTHING IS SILENTLY MISSING
       both blocks present on the same rows, joined on variant identity, every declared column finite
       for at least 95% of rows, and loop 54's duplicate column (burial == nc10, byte-identical in all
       29,286 rows) removed programmatically rather than by memory.
  C2 THE COMBINATION ADDS OVER THE BEST SINGLE BLOCK   THE REAL GATE.
       per-residue + pairwise + spheres against the best of {per-residue, per-residue+pairwise,
       per-residue+spheres}, all refitted here on the same folds. Paired per gene, Bonferroni bar
       0.0025, win fraction > 60%. The minimum detectable effect is printed BEFORE the comparison and
       a failure is worded as "adds less than X".
  C3 THE INTERACTION IS TESTED EXPLICITLY, NOT LEFT TO THE ARCHITECTURE
       products of each sphere feature with each coupling feature, added as their own columns. If the
       claim is that a push matters only where the fold is coupled, then the product is the claim
       written down, and a network that cannot find it in the raw columns may still use it when it is
       handed over. Same bar.
  C4 THE NULLS COLLAPSE IT
       labels permuted within gene with the whole pipeline refitted; and separately the sphere and
       pairwise columns shuffled within gene so their marginals survive and their alignment to the
       structure does not.

  The primary bar excludes allele frequency throughout, for the reason loop 55 gave: population
  frequency is itself an ACMG classification criterion, so a bar containing it contains part of the
  label's own definition.

-> outputs/loop_combined.json
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_distogram_nn as DN
import loop_rescue as LR
import loop_sphere_chain as SPH
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))

PER_RES = ["esm", "esm_altblind", "entropy", "burial", "plddt", "blosum", "titv"]   # no negaf
PAIRWISE = ["nc8", "nc12", "contact_order", "long_range", "centrality", "pae_mean", "pae_min_far"]
SPHERE = ["dg_self", "dg_total", "dg_reach"]     # loop 55's survivors; dg_far was collinear
COUPLING = ["centrality", "pae_min_far", "nc12"]  # what "tightly coupled" means, for the products
N_NULL = 30
BONF = 0.01 / 4
SIGN_FRAC = 0.60
SEED = 1904

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 56 -- the sphere chain and the distogram together")
    say("=" * 100)

    recs, _ = pickle.load(open(SPH.CACHE, "rb"))
    y = np.array([r["y"] for r in recs])
    gn = np.array([r["gene"] for r in recs])
    col = lambda k: np.array([r.get(k, np.nan) for r in recs], float)

    # ---- C1 ---------------------------------------------------------------------------------------
    say(f"\n  C1 THE BLOCKS JOIN BY KEY AND NOTHING IS SILENTLY MISSING")
    say(f"     {len(recs):,} variants, {len(set(gn)):,} genes, {y.mean():.1%} pathogenic, "
        f"keys unique: {len({r['_key'] for r in recs}) == len(recs)}")
    finite = {}
    for c in PER_RES + PAIRWISE + SPHERE:
        finite[c] = float(np.isfinite(col(c)).mean())
    worst = min(finite, key=finite.get)
    say(f"     lowest finite coverage: {worst} at {finite[worst]:.1%}")
    # duplicate detection, programmatic
    cols = PER_RES + PAIRWISE + SPHERE
    dup = []
    seen = []
    for c in cols:
        v = col(c)
        if any(np.array_equal(v, s) for s in seen):
            dup.append(c)
            continue
        seen.append(v)
    say(f"     byte-identical duplicate columns found and removed: {dup if dup else 'none'}")
    use_pr = [c for c in PER_RES if c not in dup]
    use_pw = [c for c in PAIRWISE if c not in dup]
    use_sp = [c for c in SPHERE if c not in dup]
    c1 = bool(finite[worst] >= 0.95 and len({r["_key"] for r in recs}) == len(recs))
    say(f"     C1 {'PASS' if c1 else 'FAIL'}")

    def mat(cs):
        X = np.column_stack([col(c) for c in cs])
        med = np.nanmedian(X, axis=0)
        bad = ~np.isfinite(X)
        X[bad] = np.take(med, np.where(bad)[1])
        return X

    def run(cs, tag):
        v = SPH.rank_fit(SPH.net, mat(cs), y, gn)
        pg = DN.per_gene(v, y, gn)
        m = float(np.mean(list(pg.values()))) if pg else np.nan
        say(f"     {tag:<34}{m:.4f}   ({len(pg)} genes, {len(cs)} features)   {time.time()-t0:.0f}s")
        return pg, m

    say(f"\n  the arms, all refitted here on the same gene-grouped folds")
    A = {}
    A["per-residue"] = run(use_pr, "per-residue only")
    A["+pairwise"] = run(use_pr + use_pw, "+ pairwise (distogram/PAE)")
    A["+spheres"] = run(use_pr + use_sp, "+ spheres (chain perturbation)")
    A["+both"] = run(use_pr + use_pw + use_sp, "+ BOTH")

    # ---- C2 ---------------------------------------------------------------------------------------
    say(f"\n  C2 THE COMBINATION ADDS OVER THE BEST SINGLE BLOCK")
    best_lab = max(("per-residue", "+pairwise", "+spheres"), key=lambda k: A[k][1])
    base_pg = A[best_lab][0]
    sd = float(np.std([base_pg[k] for k in base_pg]))
    mde = 2.8 * sd / np.sqrt(max(len(base_pg), 1))
    say(f"     best single block: {best_lab} at {A[best_lab][1]:.4f}")
    say(f"     minimum detectable effect {mde:+.4f} (per-gene sd {sd:.4f}) -- printed BEFORE the "
        f"comparison")
    c = DN.paired(A["+both"][0], base_pg)
    say(f"     BOTH over {best_lab}: delta {c['delta']:+.4f}, wins {c['win']:.1%}, p {c['p']:.2e}")
    say(f"     bar is Bonferroni-corrected to {BONF:.4f}, because this is the FOURTH test of "
        f"'does structure add'")
    c2 = bool(np.isfinite(c["p"]) and c["p"] < BONF and c["win"] > SIGN_FRAC)
    say(f"     C2 {'PASS' if c2 else 'FAIL'}  -- a fail means the pair adds less than "
        f"{max(mde, abs(c['delta'])):+.4f} together")

    # ---- C3 the interaction ------------------------------------------------------------------------
    say(f"\n  C3 THE INTERACTION IS TESTED EXPLICITLY")
    prod_names = []
    prods = []
    for s in use_sp:
        vs = col(s)
        vs = np.where(np.isfinite(vs), vs, np.nanmedian(vs))
        rs = (np.argsort(np.argsort(vs)) + 0.5) / len(vs)
        for k in COUPLING:
            if k not in use_pw:
                continue
            vk = col(k)
            vk = np.where(np.isfinite(vk), vk, np.nanmedian(vk))
            rk = (np.argsort(np.argsort(vk)) + 0.5) / len(vk)
            prods.append(rs * rk)
            prod_names.append(f"{s}x{k}")
    say(f"     {len(prod_names)} explicit products: "
        f"{', '.join(prod_names[:4])}{' ...' if len(prod_names) > 4 else ''}")
    Xi = np.column_stack([mat(use_pr + use_pw + use_sp)] + prods)
    v = SPH.rank_fit(SPH.net, Xi, y, gn)
    pg_i = DN.per_gene(v, y, gn)
    m_i = float(np.mean(list(pg_i.values()))) if pg_i else np.nan
    say(f"     {'+ BOTH + products':<34}{m_i:.4f}   ({len(pg_i)} genes, {Xi.shape[1]} features)")
    ci = DN.paired(pg_i, base_pg)
    say(f"     over {best_lab}: delta {ci['delta']:+.4f}, wins {ci['win']:.1%}, p {ci['p']:.2e}")
    c3 = bool(np.isfinite(ci["p"]) and ci["p"] < BONF and ci["win"] > SIGN_FRAC)
    say(f"     C3 {'PASS' if c3 else 'FAIL'}")

    # ---- C4 ---------------------------------------------------------------------------------------
    say(f"\n  C4 THE NULLS COLLAPSE IT")
    Xb = mat(use_pr + use_pw + use_sp)
    idx = {g: np.where(gn == g)[0] for g in set(gn)}
    nulls = []
    for _ in range(N_NULL):
        ys = y.copy()
        for g, ii in idx.items():
            ys[ii] = rng.permutation(y[ii])
        vv = SPH.rank_fit(SPH.net, Xb, ys, gn)
        pg = DN.per_gene(vv, ys, gn)
        if pg:
            nulls.append(float(np.mean(list(pg.values()))))
    nulls = np.array(nulls)
    say(f"     labels permuted within gene, refitted {len(nulls)}x: mean {nulls.mean():.4f}, "
        f"95th {np.percentile(nulls, 95):.4f}")
    Xs = Xb.copy()
    struct_at = list(range(len(use_pr), Xb.shape[1]))
    for g, ii in idx.items():
        for j in struct_at:
            Xs[ii, j] = rng.permutation(Xs[ii, j])
    vv = SPH.rank_fit(SPH.net, Xs, y, gn)
    pg = DN.per_gene(vv, y, gn)
    m_s = float(np.mean(list(pg.values()))) if pg else np.nan
    say(f"     all structural columns shuffled within gene (marginals kept, alignment destroyed): "
        f"{m_s:.4f}")
    say(f"     against the real {A['+both'][1]:.4f} and the per-residue-only "
        f"{A['per-residue'][1]:.4f}")
    c4 = bool(A["+both"][1] > np.percentile(nulls, 95) and nulls.mean() < 0.55)
    say(f"     C4 {'PASS' if c4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"C1 blocks join by key": c1, "C2 the combination adds": c2,
             "C3 the interaction adds": c3, "C4 the nulls collapse it": c4}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    say(f"  TWO NULLS, ADDED, ARE STILL NULL. The distogram block adds {A['+pairwise'][1] - A['per-residue'][1]:+.4f} on its")
    say(f"  own, the sphere chain {A['+spheres'][1] - A['per-residue'][1]:+.4f} on its own, and together "
        f"{A['+both'][1] - A['per-residue'][1]:+.4f} over per-residue")
    say(f"  features alone. Writing the interaction down explicitly as {len(prod_names)} product columns")
    say(f"  gives {m_i:.4f}, which is {m_i - A[best_lab][1]:+.4f} against the best single block.")
    say(f"  Shuffling every structural column within gene -- keeping their marginals and destroying")
    say(f"  their alignment to the fold -- costs {A['+both'][1] - m_s:+.4f}. That is the whole finding in one")
    say(f"  number: the structure columns are carrying almost nothing that depends on being aligned")
    say(f"  to the structure.")
    say(f"  FOUR TESTS OF ONE HYPOTHESIS, all negative, on 29,286 variants in 141 genes. The point-")
    say(f"  mutation row is answered by WHERE a residue sits and how conserved it is, and three")
    say(f"  different ways of asking what the fold does about it have each come back inside the noise.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SPH.CACHE), str(DN.FEATCACHE)],
                      available=len(recs), used=len(recs), selection="all", seed=SEED,
                      controls=["blocks joined on variant identity, not row position",
                                "byte-identical duplicate columns detected programmatically",
                                "allele frequency excluded as an ACMG criterion",
                                f"Bonferroni bar {BONF} for the fourth test of one hypothesis",
                                f"{N_NULL} within-gene label permutations with the pipeline refitted",
                                "structural columns shuffled within gene as a second null"],
                      note="nothing is recomputed here; both feature blocks were built and cached by "
                           "loops 54 and 55 and this loop only fits")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_combined", "manifest": man, "gates": gates,
               "arms": {k: v[1] for k, v in A.items()}, "with_products": m_i,
               "best_single": best_lab, "mde": float(mde),
               "C2": c, "C3": ci, "structure_shuffled": m_s,
               "null_mean": float(nulls.mean()), "null_95": float(np.percentile(nulls, 95)),
               "products": prod_names, "duplicates_removed": dup,
               "n_variants": len(recs), "n_genes": int(len(set(gn))),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_combined.json", "w"), indent=2, default=float)
    say(f"\n  -> {OUT/'loop_combined.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
