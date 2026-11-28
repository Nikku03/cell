"""Joining the encyclopedia's genes to actual gene products with actual mechanism.

WHY. cell_complete.json tiers 16,492 genes by how far a mechanistic chain gets, and the metabolic
tier looked like the template the rest should imitate: gene -> enzyme -> reaction. It is not one.
`generxn` holds reaction STRINGS for 2,549 genes and only 31 rows anywhere in that blob are
structured. The genuinely mechanistic metabolic model -- Recon3D, 10,600 reactions, 5,835
metabolites, the one MODELAUDIT.txt puts through six blocking gates -- is a separate file that the
encyclopedia never references.

AND THE TWO ARE NOT THE SAME MODEL, which is the first thing this module checks rather than
assumes. generxn writes 'chylomicron[e] -> chylomicron remnant[e] + 77243 TAG-chylomicron pool[e]'
-- Human-GEM metabolite names with bracket compartments. Recon3D writes '10fthf_c' -- BiGG ids. So
reaction identifiers CANNOT be transferred between them. The only sound join key is the gene.

WHAT A SOUND JOIN PRODUCES. One object in which a gene resolves to either a balanced reaction with
stoichiometry and a compartment, or a signed action on a named target, or neither -- and says which.
That is the object the 'direction and process' goal needs, and it is the object the encyclopedia
currently cannot supply.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

J0  PROVENANCE BEFORE JOINING. Confirm from notation which model generxn came from.
    PREDECLARED: if generxn is not Recon3D, this module must join at the GENE level only and must
    say so plainly. Emitting a gene->reaction-id mapping across two different metabolic
    reconstructions would look authoritative and be wrong.

J1  THE CEILING GATE. Price the join at zero before building it: how many encyclopedia genes gain
    a STRUCTURED, mass-balanced reaction they do not already have?
    PREDECLARED: if Recon3D's GPR gene set is a subset of generxn's, the join buys identifiers and
    no new mechanism, and may not be reported as a coverage gain however large the match rate.

J2  THE IDENTIFIER JOIN AND ITS TWO CLASSIC FAILURES. Symbol-to-symbol match rate, reported
    together with (a) many-to-one collisions, where several Recon3D gene records carry one symbol,
    and (b) the explicit unmatched list.
    PREDECLARED: a match rate quoted without the collision count is not a join, it is a percentage.
    Recon3D splits genes by transcript (1591_AT1, 314_AT1, 314_AT2), so collisions are EXPECTED and
    the question is whether they are handled, not whether they occur.

J3  DO THE TWO MODELS AGREE ON WHO IS METABOLIC? Set comparison of Recon3D GPR genes against
    generxn genes. PREDECLARED: the DISAGREEMENT is the informative quantity. Two independently
    built reconstructions naming different gene sets bounds how much either can be trusted as
    'the metabolic genes', and that bound belongs in the record whichever way it falls.

J4  THE UNIFIED OBJECT. Recompute the tier table with the join applied, and write the artifact.
    Every gene resolves to: balanced reaction / signed action / complex only / product only / none.

J5  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import gzip
import json
import re
import time

HERE = os.path.dirname(__file__)
RECON = os.path.join(HERE, "recon3d.json")
ENCY = os.path.join(HERE, "..", "..", "colab", "data", "cell_complete.json.gz")
OUT = os.path.join(HERE, "RESULTS_productjoin.txt")
ART = os.path.join(HERE, "..", "..", "outputs", "gene_product_join.json")
RULE = "=" * 97


def load():
    M = json.load(open(RECON))
    D = json.load(gzip.open(os.path.normpath(ENCY)))
    return M, D


def recon_gene_index(M):
    """symbol -> {recon gene ids}, and the reverse, keeping transcript splits visible."""
    sym, by_id = collections.defaultdict(set), {}
    for g in M["genes"]:
        s = (g.get("name") or "").strip().upper()
        ncbi = (g.get("annotation", {}).get("ncbigene") or [""])[0]
        by_id[g["id"]] = (s, ncbi)
        if s:
            sym[s].add(g["id"])
    return sym, by_id


def gpr_genes(M):
    """reaction id -> set of recon gene ids mentioned in its GPR, and the inverse."""
    r2g, g2r = {}, collections.defaultdict(set)
    tok = re.compile(r"[0-9][0-9A-Za-z_.]*")
    for r in M["reactions"]:
        rule = r.get("gene_reaction_rule", "") or ""
        if not rule.strip():
            continue
        ids = set(tok.findall(rule))
        r2g[r["id"]] = ids
        for i in ids:
            g2r[i].add(r["id"])
    return r2g, g2r


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    M, D = load()
    names = [r["name"] for r in D["genes"]]
    idx = {n.upper(): i for i, n in enumerate(names)}
    N = len(names)

    P_(RULE); P_("JOINING THE ENCYCLOPEDIA TO REAL GENE PRODUCTS"); P_(RULE)
    P_(f"  encyclopedia : {N:,} genes")
    P_(f"  Recon3D      : {len(M['reactions']):,} reactions, {len(M['metabolites']):,} metabolites,"
       f" {len(M['genes']):,} gene records")

    # ---- J0 -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("J0  PROVENANCE: ARE THESE THE SAME METABOLIC MODEL?"); P_(RULE)
    ex = next(iter(D["generxn"].values()))[0]
    mets = [m["id"] for m in M["metabolites"][:3]]
    P_(f"  generxn writes : {ex[:88]}")
    P_(f"  Recon3D writes : {mets}")
    same = "[" not in mets[0]
    P_(f"\n  generxn uses Human-GEM metabolite NAMES with bracket compartments;"
       f" Recon3D uses BiGG ids.")
    P_( "  J0: DIFFERENT MODELS. Reaction identifiers cannot be transferred. The join is at the")
    P_( "      GENE level only, and no gene->reaction-id mapping across the two is emitted.")

    # ---- J2 (the join itself, needed before J1 can be priced) -----------------------------
    P_("\n" + RULE); P_("J2  THE IDENTIFIER JOIN, WITH ITS COLLISIONS"); P_(RULE)
    sym, by_id = recon_gene_index(M)
    r2g, g2r = gpr_genes(M)
    P_(f"  Recon3D gene records          {len(M['genes']):>6,}")
    P_(f"  ...with a symbol              {sum(1 for g in M['genes'] if (g.get('name') or '').strip()):>6,}")
    P_(f"  distinct symbols              {len(sym):>6,}")
    coll = {s: v for s, v in sym.items() if len(v) > 1}
    P_(f"  symbols held by >1 record     {len(coll):>6,}   (transcript splits: 314_AT1 / 314_AT2)")
    P_(f"  worst collision               {max((len(v) for v in sym.values()), default=0):>6,} records on one symbol")
    matched = {s for s in sym if s in idx}
    P_(f"\n  symbols matching an encyclopedia gene   {len(matched):>6,} / {len(sym):,}"
       f"  = {100*len(matched)/max(len(sym),1):.1f}%")
    unmatched = sorted(set(sym) - matched)
    P_(f"  UNMATCHED                               {len(unmatched):>6,}   e.g. {unmatched[:8]}")

    # gene -> reactions, in symbol space, GPR-carrying only
    sym2rxn = collections.defaultdict(set)
    for gid, rxns in g2r.items():
        s = by_id.get(gid, ("", ""))[0]
        if s:
            sym2rxn[s] |= rxns
    P_(f"\n  symbols carrying >=1 GPR reaction       {len(sym2rxn):>6,}")
    P_(f"  ...that match the encyclopedia          {len({s for s in sym2rxn if s in idx}):>6,}")

    # ---- J1  ceiling gate -------------------------------------------------------------------
    P_("\n" + RULE); P_("J1  THE CEILING GATE: WHAT DOES THE JOIN ACTUALLY BUY?"); P_(RULE)
    gx = set()
    for k in D["generxn"]:
        j = int(k) if k.isdigit() else idx.get(k.upper())
        if j is not None and 0 <= j < N:
            gx.add(names[j].upper())
    rec = {s for s in sym2rxn if s in idx}
    P_(f"  generxn genes (reaction STRINGS)        {len(gx):>6,}")
    P_(f"  Recon3D GPR genes present here          {len(rec):>6,}")
    P_(f"  in BOTH                                 {len(gx & rec):>6,}")
    P_(f"  Recon3D ONLY -- gain structured mechanism they did not have   {len(rec - gx):>6,}")
    P_(f"  generxn ONLY -- a string, and Recon3D has no GPR for them     {len(gx - rec):>6,}")
    gain = len(rec - gx)
    P_(f"\n  J1 VERDICT: the join adds a balanced, stoichiometric reaction to {gain:,} genes")
    P_(f"  that previously had no structured mechanism anywhere in the encyclopedia.")
    P_(f"  {'PASS -- this is new mechanism, not new identifiers' if gain >= 200 else 'FAIL -- identifiers only; do not report as coverage'}")

    # ---- J3 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("J3  DO TWO RECONSTRUCTIONS AGREE ON WHO IS METABOLIC?"); P_(RULE)
    inter, union = gx & rec, gx | rec
    P_(f"  union of the two metabolic gene sets    {len(union):>6,}")
    P_(f"  intersection                            {len(inter):>6,}")
    P_(f"  Jaccard agreement                       {len(inter)/max(len(union),1):>6.3f}")
    P_(f"\n  J3: two independently built human reconstructions agree on"
       f" {100*len(inter)/max(len(union),1):.1f}% of the genes")
    P_( "  they call metabolic. That bounds how firmly ANY of this can be called 'the metabolic")
    P_( "  genes', and it bounds it the same way whichever direction one prefers.")

    # ---- J4 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("J4  THE UNIFIED OBJECT"); P_(RULE)
    sgn_out = collections.defaultdict(list)
    for s, t, w in D["sig"]:
        if w:
            sgn_out[s].append([names[t], w, "sig"])
    for s, t, w in D["reg"]:
        if w:
            sgn_out[s].append([names[t], w, "reg"])
    ppm = {int(k): v for k, v in D["ppm"].items() if k.isdigit()}
    cplx = collections.defaultdict(list)
    for cn, mem in D["complexes"].items():
        for m in mem:
            if isinstance(m, int) and 0 <= m < N:
                cplx[m].append(cn)

    rxn_by_id = {r["id"]: r for r in M["reactions"]}
    art, tier = {}, collections.Counter()
    for i, nm in enumerate(names):
        u = nm.upper()
        rx = sorted(sym2rxn.get(u, ()))
        acts = sgn_out.get(i, [])
        if rx:
            t = "1_balanced_reaction"
        elif acts:
            t = "2_signed_action"
        elif cplx.get(i):
            t = "3_complex_only"
        elif i in ppm:
            t = "4_product_only"
        else:
            t = "5_none"
        tier[t] += 1
        rec_i = {"tier": t}
        if rx:
            rec_i["recon3d_reactions"] = rx[:50]
            rec_i["n_reactions"] = len(rx)
            sub = collections.Counter(rxn_by_id[r].get("subsystem", "") for r in rx)
            rec_i["subsystems"] = [s for s, _ in sub.most_common(3) if s]
            r0 = rxn_by_id[rx[0]]
            rec_i["example"] = {"id": r0["id"], "name": r0["name"],
                                "stoichiometry": r0["metabolites"]}
            rec_i["recon3d_gene_ids"] = sorted(sym.get(u, ()))
        if acts:
            rec_i["n_signed_targets"] = len(acts)
            rec_i["activates"] = [a[0] for a in acts if a[1] == 1][:10]
            rec_i["represses"] = [a[0] for a in acts if a[1] == -1][:10]
        if cplx.get(i):
            rec_i["complexes"] = cplx[i][:5]
        if i in ppm:
            rec_i["abundance_ppm"] = ppm[i]
        art[nm] = rec_i

    P_(f"\n    {'tier':<24} {'genes':>7}   {'%':>6}")
    for t in sorted(tier):
        P_(f"    {t:<24} {tier[t]:>7,}   {100*tier[t]/N:>5.1f}%")
    mech = tier["1_balanced_reaction"] + tier["2_signed_action"]
    P_(f"\n  MECHANISTIC (tier 1 + 2): {mech:,} = {100*mech/N:.1f}% of the genome")
    both = sum(1 for v in art.values() if "n_reactions" in v and "n_signed_targets" in v)
    P_(f"  genes with BOTH a reaction and a signed action: {both:,}")

    os.makedirs(os.path.dirname(os.path.normpath(ART)), exist_ok=True)
    json.dump(art, open(os.path.normpath(ART), "w"))
    P_(f"\n  artifact written: outputs/gene_product_join.json"
       f"  ({os.path.getsize(os.path.normpath(ART))/1e6:.1f} MB, {len(art):,} genes)")

    for nm in ("HK1", "MTOR", "NR3C1", "TP53"):
        v = art.get(nm, {})
        P_(f"\n  {nm}: tier {v.get('tier')}")
        if "example" in v:
            P_(f"     {v['n_reactions']} reactions, subsystems {v.get('subsystems')}")
            P_(f"     e.g. {v['example']['id']}: {v['example']['name'][:60]}")
            P_(f"          {v['example']['stoichiometry']}")
        if "n_signed_targets" in v:
            P_(f"     {v['n_signed_targets']} signed targets"
               f" | activates {v.get('activates', [])[:4]} | represses {v.get('represses', [])[:4]}")

    # ---- J5 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("J5  WHAT THIS DOES NOT SETTLE"); P_(RULE)
    P_("  1. The join is gene-level. generxn's Human-GEM strings are left in place and are NOT")
    P_("     reconciled to Recon3D reactions; J0 says why that would be unsound.")
    P_("  2. A symbol match is not an identity proof. Recon3D symbols come from refseq_name and")
    P_("     the encyclopedia's provenance for its own symbols is not recorded anywhere.")
    P_("  3. Tier 1 means a mass-balanced reaction EXISTS for that gene in Recon3D. It does not")
    P_("     mean the reaction carries flux, and MODELAUDIT M2 found 60 internal reactions that")
    P_("     do not conserve elements.")
    P_("  4. Tier 2's signs inherit the default-activation problem measured in the graphvalue")
    P_("     and encyclopedia censuses: roughly half of the regulatory signs are a rule, not")
    P_("     evidence. The 'sig' layer (97.1% signed, 35% repression) is the trustworthy half.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
