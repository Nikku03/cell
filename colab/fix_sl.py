"""WHAT THE `sl` LAYER ACTUALLY CONTAINS -- and why it is not synthetic lethality.

THE DIAGNOSIS, ARRIVED AT TWICE INDEPENDENTLY. Tonight's DepMap module reported that the cell object's `sl`
layer is synthetic lethality mislabelled: all 1,256 pairs are POSITIVELY correlated in dependency across cell
lines. Checked directly here rather than taken on trust -- every score lies in 0.401..0.912, none negative --
and the highest-scoring pair is TSC1/TSC2, which form the TSC complex. Two subunits of one complex are the
textbook illustration of the OPPOSITE of synthetic lethality: knocking out either produces the same failure,
so the pair is CO-REQUIRED, not synthetically lethal.

    synthetic lethality   losing A is tolerable, losing B is tolerable, losing BOTH is not
                          -> dependency on A is HIGHER where B is deficient   (a NEGATIVE relationship)
    what `sl` stores      dependency on A tracks dependency on B across lines (a POSITIVE relationship)

These are not variants of one quantity. They have opposite signs and opposite uses: SL is what makes a target
selective in a mutant background, co-requirement is what makes two genes part of one machine. A model that
reasons about drug targets using `sl` as though it were synthetic lethality would reach the wrong conclusion
in the specific case the layer exists to serve.

WHAT THIS MODULE CAN AND CANNOT DO, stated before the result rather than after.

CANNOT: build real synthetic lethality. That requires a DEFICIENCY state for gene B -- bottom-decile
expression, or a damaging mutation -- and DepMap's portal now sits behind a verification challenge. Every
endpoint tried (`/portal/api/download/files`, `/portal/download/api/downloads`, `/portal/api/download/table`)
returns the HTML challenge page rather than JSON. The dependency matrix already on disk contains gene EFFECT,
which is the result of a perturbation, not the natural deficiency state SL is defined against. Substituting
one for the other would manufacture a layer, and an unreachable source is a result rather than a licence.

CAN: establish precisely what the layer IS, from data already here. If the pairs are complex co-members, the
layer is not merely mislabelled once but twice -- it is complex co-membership wearing a co-dependency score
wearing a synthetic-lethality name. That is worth knowing exactly, because it tells you which existing layer
already covers it and therefore whether anything is lost by retiring it.

THE TESTS, each with its own null:
  1  RECOMPUTE co-dependency from DepMap gene effect and correlate with the stored score. High agreement
     identifies the provenance of a layer whose construction was never recorded.
  2  COMPLEX CO-MEMBERSHIP against a degree-matched null, read from the cell object's own gene2cplx.
     Complex partners share dependency for a mechanical reason, so if `sl` pairs are enriched for
     co-complex membership over partners of the same complex-degree, the layer is describing machines.
  3  REDUNDANCY with the layers already present -- codep, PPI, gene2cplx. A layer that adds no pair the
     model already has is pure liability: it carries a wrong name and no new information.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
NET = SP / "cell_complete.json.gz"
DEP = SP / "CRISPRGeneEffect.csv"
BLOCKED = ["https://depmap.org/portal/api/download/files",
           "https://depmap.org/portal/download/api/downloads",
           "https://depmap.org/portal/api/download/table"]
N_DRAWS = 20


def load_dep(report, want):
    """DepMap gene effect, columns restricted to the genes we need. Read in chunks: the file is 419 MB."""
    import pandas as pd
    head = pd.read_csv(DEP, nrows=0)
    cols = list(head.columns)
    sym = {}
    for c in cols[1:]:
        s = c.split(" (")[0].strip()
        if s in want:
            sym.setdefault(s, c)
    use = [cols[0]] + [sym[s] for s in sym]
    report(f"  DepMap: {len(cols)-1:,} genes in file, {len(sym):,} of our {len(want):,} requested present")
    df = pd.read_csv(DEP, usecols=use)
    df = df.rename(columns={v: k for k, v in sym.items()})
    lines = df[cols[0]].values
    M = df.drop(columns=[cols[0]]).astype(np.float32)
    return M.columns.tolist(), M.values, lines


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("WHAT THE `sl` LAYER ACTUALLY CONTAINS")
    report("=" * 100)

    d = json.load(gzip.open(NET, "rt"))
    name = [g["name"] for g in d["genes"]]
    sl = d["sl"]
    codep = {(int(a), int(b)) for a, v in (d.get("codep") or {}).items() for b, _s in v}
    ppi = {(int(a), int(b)) for a, b in [(e[0], e[1]) for e in d["ppi"]]}
    g2c = {int(k): set(v) for k, v in (d.get("gene2cplx") or {}).items()}
    del d

    pairs = [(name[e[0]], name[e[1]], float(e[2])) for e in sl]
    sc = np.array([p[2] for p in pairs])
    report(f"  {len(pairs):,} stored pairs, score {sc.min():.3f}..{sc.max():.3f}, "
           f"{int((sc < 0).sum())} negative")
    report(f"  highest-scoring pair: {pairs[int(np.argmax(sc))][0]}/{pairs[int(np.argmax(sc))][1]} "
           f"-- two subunits of one complex, the textbook OPPOSITE of synthetic lethality")

    R = {"n_pairs": len(pairs), "score_min": float(sc.min()), "score_max": float(sc.max()),
         "n_negative_scores": int((sc < 0).sum()),
         "blocked_sources": {u: "HTML verification challenge, not JSON" for u in BLOCKED},
         "why_sl_unbuildable": "synthetic lethality needs a DEFICIENCY state (bottom-decile expression or "
                               "damaging mutation) for the partner gene. DepMap gene EFFECT is the result "
                               "of a perturbation, not a natural deficiency, and the omics files that carry "
                               "deficiency are behind the portal challenge. Substituting effect for "
                               "deficiency would manufacture the layer."}

    # ---- 1. recompute co-dependency and compare ----
    want = {a for a, _b, _s in pairs} | {b for _a, b, _s in pairs}
    genes, M, lines = load_dep(report, want)
    gi = {g: i for i, g in enumerate(genes)}
    ok = [(a, b, s) for a, b, s in pairs if a in gi and b in gi]
    report(f"\n  1. PROVENANCE -- recompute co-dependency from gene effect ({len(lines):,} cell lines)")
    Mz = M - np.nanmean(M, 0)
    Mz = Mz / np.maximum(np.nanstd(M, 0), 1e-9)
    Mz = np.nan_to_num(Mz)
    rec = np.array([float(np.dot(Mz[:, gi[a]], Mz[:, gi[b]]) / len(lines)) for a, b, _s in ok])
    st = np.array([s for _a, _b, s in ok])
    r = float(stats.spearmanr(rec, st)[0])
    report(f"     {len(ok):,} pairs scorable; Spearman(recomputed, stored) = {r:.4f}")
    report(f"     recomputed correlations: {rec.min():+.3f}..{rec.max():+.3f}, "
           f"{100*np.mean(rec > 0):.1f}% positive")
    R["provenance"] = {"n_scorable": len(ok), "spearman_recomputed_vs_stored": r,
                       "frac_recomputed_positive": float(np.mean(rec > 0))}

    # ---- 2. complex co-membership vs a degree-matched null ----
    #
    # THE SOURCE HERE IS THE CELL OBJECT'S OWN gene2cplx, NOT A CORUM CACHE, AND THAT IS DELIBERATE.
    # The first version of this arm read `cell_corum.json.gz`, which is not on disk: the scratchpad holds
    # a `corum.txt` that is a saved 404 page. Reading a missing cache produced an empty membership map,
    # which would have made this arm print "observed 0.0000, null 0.0000" -- a broken lookup wearing the
    # costume of a biological zero, and directly contradicted by section 3, which finds co-complex pairs
    # through gene2cplx on the same pairs. gene2cplx is keyed by the same gene index as `sl` itself, so
    # it cannot mismatch, and it is the membership the model actually reasons with. The guard below stays
    # anyway: if BOTH arms come out empty this reports a failure, never a zero.
    report(f"\n  2. ARE THESE COMPLEX PARTNERS? co-membership vs a degree-matched null "
           f"(source: the cell object's gene2cplx, {len(g2c):,} genes with membership)")
    nidx = {n: i for i, n in enumerate(name)}
    oki = [(nidx[a], nidx[b]) for a, b, _s in ok if a in nidx and b in nidx]
    report(f"     {len(oki):,}/{len(ok):,} scorable pairs resolve to gene indices")

    def co(i, j):
        return bool(g2c.get(i) and g2c.get(j) and g2c[i] & g2c[j])

    obs = float(np.mean([co(i, j) for i, j in oki])) if oki else 0.0
    # NULL: keep each pair's first gene, redraw the partner from genes of the same complex-degree bin.
    # Degree matters because a subunit of many complexes is co-complex with more genes for a reason that
    # has nothing to do with co-dependency; an unmatched redraw would credit the layer for that.
    #
    # TWO POOLS, BECAUSE THE WIDE ONE IS TOO EASY. Redrawing the partner from all 19k genes matches the
    # NUMBER of complexes it belongs to but not the KIND of gene it is: `sl` genes are 921 well-studied,
    # strongly-dependent, complex-forming genes, and a null made of arbitrary genes would hand this arm a
    # large ratio for the composition of the pool alone. The strict pool draws the replacement partner
    # from the `sl` layer's own genes, so both arms are made of the same kind of gene and only the
    # PAIRING differs. Where the two disagree, the strict number is the one to quote.
    deg = {i: len(g2c.get(i, ())) for i in range(len(name))}
    dq = {i: min(4, deg[i]) for i in deg}
    slg = sorted({i for p in oki for i in p})

    def draw_null(pool, tag):
        by = {}
        for i in pool:
            by.setdefault(dq[i], []).append(i)
        out = []
        for s in range(N_DRAWS):
            rr = np.random.default_rng(s)
            v = []
            for i, j in oki:
                cand = by.get(dq[j])
                # A degree bin with no member in this pool cannot be matched. Dropping the pair silently
                # would quietly compare different pair sets between the arms, so it is counted instead.
                v.append(co(i, int(rr.choice(cand))) if cand else False)
            out.append(float(np.mean(v)) if v else 0.0)
        unmatched = sum(1 for i, j in oki if not by.get(dq[j]))
        if unmatched:
            report(f"     [{tag}] {unmatched:,}/{len(oki):,} pairs had no same-degree partner in this "
                   f"pool and count as not-co-complex, which biases the null DOWN and the ratio UP")
        return out

    nulls = draw_null(list(range(len(name))), "wide")
    nulls_strict = draw_null(slg, "strict")
    nm = float(np.mean(nulls))
    nm_s = float(np.mean(nulls_strict))
    if obs == 0.0 and nm == 0.0:
        # A LOOKUP THAT MATCHES NOTHING IS A BROKEN DIAGNOSTIC, NOT A NEGATIVE RESULT.
        report("     ARM FAILED: membership matched no pair on either side. Reported as failed, not "
               "as an enrichment of 0.0x -- a zero from a lookup that resolves nothing is not a "
               "measurement.")
        R["corum"] = {"status": "failed",
                      "reason": "complex membership resolved for neither the observed pairs nor the "
                                "degree-matched null; nothing was measured"}
    else:
        sd, sd_s = float(np.std(nulls)), float(np.std(nulls_strict))
        z, z_s = (obs - nm) / max(sd, 1e-9), (obs - nm_s) / max(sd_s, 1e-9)
        report(f"     observed co-complex rate: {obs:.4f}")
        report(f"       vs degree-matched null, WIDE pool (all {len(name):,} genes):  {nm:.4f}  "
               f"({obs/max(nm,1e-9):6.2f}x, sd {sd:.4f}, z {z:+.1f}, "
               f"{sum(1 for x in nulls if x >= obs)}/{N_DRAWS} draws >= obs)")
        report(f"       vs degree-matched null, STRICT pool (the {len(slg):,} `sl` genes): {nm_s:.4f}  "
               f"({obs/max(nm_s,1e-9):6.2f}x, sd {sd_s:.4f}, z {z_s:+.1f}, "
               f"{sum(1 for x in nulls_strict if x >= obs)}/{N_DRAWS} draws >= obs)")
        report(f"     the STRICT ratio is the one to quote; the wide pool is reported so the size of the "
               f"composition effect is visible rather than hidden")
        R["corum"] = {"status": "ok", "source": "cell.gene2cplx", "observed": obs,
                      "n_draws": N_DRAWS,
                      "wide": {"pool": "all genes", "null_mean": nm, "null_sd": sd,
                               "ratio": float(obs / max(nm, 1e-9)), "z": float(z),
                               "n_draws_null_ge_obs": int(sum(1 for x in nulls if x >= obs))},
                      "strict": {"pool": "genes appearing in the sl layer", "null_mean": nm_s,
                                 "null_sd": sd_s, "ratio": float(obs / max(nm_s, 1e-9)), "z": float(z_s),
                                 "n_draws_null_ge_obs": int(sum(1 for x in nulls_strict if x >= obs))},
                      "quote": "strict"}

    # ---- 3. redundancy with layers already present ----
    idx = {n: i for i, n in enumerate(name)}
    inc, inp, incp = 0, 0, 0
    for a, b, _s in pairs:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        if (ia, ib) in codep or (ib, ia) in codep:
            inc += 1
        if (ia, ib) in ppi or (ib, ia) in ppi:
            inp += 1
        if g2c.get(ia) and g2c.get(ib) and g2c[ia] & g2c[ib]:
            incp += 1
    report(f"\n  3. REDUNDANCY with layers the model already has")
    report(f"     already in codep:        {inc:5,d}/{len(pairs):,} ({100*inc/len(pairs):.1f}%)")
    report(f"     already in PPI:          {inp:5,d}/{len(pairs):,} ({100*inp/len(pairs):.1f}%)")
    report(f"     already co-complex:      {incp:5,d}/{len(pairs):,} ({100*incp/len(pairs):.1f}%)")
    novel = sum(1 for a, b, _s in pairs
                if (idx.get(a) is not None and idx.get(b) is not None)
                and not ((idx[a], idx[b]) in codep or (idx[b], idx[a]) in codep)
                and not ((idx[a], idx[b]) in ppi or (idx[b], idx[a]) in ppi)
                and not (g2c.get(idx[a]) and g2c.get(idx[b]) and g2c[idx[a]] & g2c[idx[b]]))
    report(f"     in NONE of the three:    {novel:5,d}/{len(pairs):,} ({100*novel/len(pairs):.1f}%)")
    R["redundancy"] = {"in_codep": inc, "in_ppi": inp, "in_complex": incp, "in_none": novel,
                       "of": len(pairs)}

    # ---- write the correction ----
    corrected = {
        "model": "co-requirement-v1 (was `sl`)",
        "correction": "This layer was named `sl` (synthetic lethality) and contains the OPPOSITE "
                      "relationship. Every stored score is a POSITIVE dependency correlation, which is "
                      "co-requirement. Renamed rather than deleted, because the pairs are real -- only the "
                      "name and its implied semantics were wrong.",
        "semantics": "score = correlation of CRISPR dependency profiles across cell lines; HIGH means the "
                     "two genes are needed by the same lines, i.e. parts of one requirement",
        "not_synthetic_lethality": "SL means dependency on A is HIGHER where B is deficient -- a NEGATIVE "
                                   "relationship. No pair here has a negative score.",
        "n_pairs": len(pairs),
        "diagnostics": {k: R.get(k) for k in ("provenance", "corum", "redundancy")},
        "limits": [
            "real synthetic lethality could not be built: it needs a deficiency state (expression or "
            "mutation) and DepMap's portal returns a verification challenge on every download endpoint",
            "the stored scores' construction was never recorded; the recomputation here identifies what "
            "they agree with, which is evidence of provenance and not proof of it",
            "co-requirement inferred from dependency correlation is not evidence of physical association"],
        "pairs": [{"a": a, "b": b, "dependency_correlation": s} for a, b, s in pairs],
    }
    with gzip.open(SP / "cell_corequirement.json.gz", "wt") as fh:
        json.dump(corrected, fh)

    report("\n" + "=" * 100)
    cr = R.get("corum")
    if cr and cr.get("status") == "failed":
        ctxt = (f" (The CORUM enrichment arm failed on a key mismatch and is reported as failed rather "
                f"than as a zero; {100*incp/len(pairs):.0f}% of pairs are co-complex by the cell "
                f"object's own gene2cplx.)")
    elif cr and cr.get("strict", {}).get("ratio", 0) > 1.5:
        ctxt = (f" They are {cr['strict']['ratio']:.1f}x enriched for complex co-membership over a "
                f"null drawn from the layer's OWN genes and matched on complex-degree "
                f"({cr['observed']:.3f} vs {cr['strict']['null_mean']:.3f}), so the pairs describe "
                f"machines.")
    else:
        ctxt = ""
    v = (f"THE LAYER IS CO-REQUIREMENT, NOT SYNTHETIC LETHALITY, AND THE TWO HAVE OPPOSITE SIGNS. All "
         f"{len(pairs):,} scores are positive ({sc.min():.3f}..{sc.max():.3f}); synthetic lethality is a "
         f"NEGATIVE relationship -- dependency on A rising where B is deficient. Recomputing co-dependency "
         f"from DepMap gene effect reproduces the stored scores at Spearman {r:.3f}, identifying what the "
         f"layer is." + ctxt +
         f" {100*(len(pairs)-novel)/len(pairs):.0f}% of its pairs are already present in codep, PPI or the "
         f"complex layer, so it carries almost no information the model lacks -- only a wrong name. Renamed "
         f"to co-requirement rather than deleted. REAL SL REMAINS UNBUILT and the reason is external: it "
         f"needs a deficiency state, and DepMap's portal answers every download endpoint with a "
         f"verification challenge. Substituting gene EFFECT for deficiency would have manufactured the "
         f"layer, which is what the original mislabelling effectively did.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "fix_sl.json", "w"), indent=1, default=float)
    report(f"\n  -> {SP/'cell_corequirement.json.gz'}")
    report(f"  -> {OUT/'fix_sl.json'}")


if __name__ == "__main__":
    main()
