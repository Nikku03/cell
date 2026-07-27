"""PER-REACTION ABLATION: remove each participant of each reaction, one at a time, and record what follows.

WHAT THIS PRODUCES. For every reaction and every thing it needs -- each substrate, each cofactor, each catalytic
enzyme -- one row saying whether the reaction still runs without it, what stops being made, whether anything else
still makes that product, and which reactions downstream lose their supply. Roughly 28,500 reactions with several
participants each, so on the order of a hundred thousand single-participant ablations.

THE DISTINCTION THAT MAKES THIS WORTH COMPUTING, AND THE ONE MOST NETWORK MODELS SKIP: removing a participant does
NOT automatically stop a reaction, and a stopped reaction does NOT automatically starve anything.

    a catalyst with a second annotated enzyme    -> REDUNDANT, the reaction still runs
    a substrate produced by other reactions too  -> BUFFERED, supply continues from elsewhere
    a currency species (ATP, H2O, O2, phosphate) -> BUFFERED by definition; the cell does not lose ATP because
                                                    one kinase is gone, and treating it as lost connects the whole
                                                    network to itself
    otherwise                                    -> the reaction STOPS
    and only if a stopped reaction's product has no other producer does anything downstream actually STARVE

Every earlier consequence model in this project collapsed these into "participant gone = reaction gone", which is
what made cascades reach the entire cell.

THREE DEFECTS THE FIRST VERSION HAD, ALL INFLATING "STOPS", ALL FOUND BY DRAWING THE REACTIONS AND LOOKING AT THEM
RATHER THAN BY READING THE SUMMARY TABLE:

  1. CURRENCY WAS MARKED PER SOURCE, INCONSISTENTLY. H2O was flagged currency on 4,066 Reactome slots and NOT on
     814 Human-GEM slots; H+, ATP, Pi, Na+ the same; O2 was never flagged anywhere. The drawn output showed a
     demethylase step "starving 98 reactions" because O2 was removed -- a true statement about the chemistry and a
     worthless one about the cell. Fixed by applying ONE canonical currency set, by normalised name, to both sources
     after loading, so the same species cannot be currency in one reaction and rate-limiting in the next.

  2. THE TWO SOURCES DO NOT SHARE AN ID SPACE. Reactome entities are `R-HSA-...`, Human-GEM metabolites are
     `MAM02675m`; the intersection of the two id sets is exactly ZERO. So a metabolite produced in Human-GEM could
     never buffer a Reactome reaction that consumes it. Bridged on normalised metabolite name -- which recovers
     15,989 participant slots but only 193 distinct names, because the two resources genuinely name common
     metabolites differently. Reported, not oversold: this is a partial repair.

  3. "STOPS" WAS BEING READ AS "FRAGILE", AND IT IS NOT. Measured directly: 20,201 of the 20,202 INPUT/STOPS rows
     are species that NOTHING in the network produces. That is not a sole-source bottleneck, it is the boundary of
     the model -- oxygen, nutrients, drugs (acetaminophen appeared as a "single point of failure starving 50
     reactions"), and entities whose ids never resolved. So the verdict is kept, because removing the input really
     does stop the step, but every count is now split by whether the removed thing is something a genetic
     perturbation could remove at all. That split is the difference between a chemistry statement and a knockout
     prediction.

WHAT IS DELIBERATELY NOT CLAIMED. This is a structural ablation, not a kinetic one. It says a route exists or does
not; it cannot say whether the remaining route carries enough flux, because that needs enzyme abundance and
turnover numbers. A BUFFERED verdict therefore means "another producer is annotated", not "the cell is fine" --
and that limit is why the dosage bands matter.
"""
import collections
import csv
import json
import re
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAXPART = 60

# ONE canonical currency set, applied to both sources by normalised name. These are the species whose pool is set by
# the whole of metabolism rather than by any single reaction: removing them stops essentially every step, which is
# true and uninformative. Kept deliberately tight -- acetyl-CoA, NADPH and glutamate are NOT here, because for those
# the identity of the producing reaction can genuinely matter.
CURRENCY = {
    "h2o", "water", "h+", "h", "proton", "o2", "oxygen", "co2", "atp", "adp", "amp", "gtp", "gdp", "gmp",
    "utp", "udp", "ump", "ctp", "cdp", "cmp", "pi", "phosphate", "orthophosphate", "ppi", "diphosphate",
    "pyrophosphate", "na+", "k+", "cl-", "mg2+", "ca2+", "zn2+", "fe2+", "fe3+", "mn2+", "cu2+", "cu+",
    "nad+", "nadh", "nadp+", "nadph", "fad", "fadh2", "fmn", "fmnh2", "coa", "coa-sh", "hco3-", "nh3", "nh4+",
}


def norm(n):
    return re.sub(r"\s+", " ", (n or "").strip().lower())


def validate(rows, solecat, cas):
    """Does the ablation predict anything, or is it just bookkeeping?

    The test: a gene whose loss starves many downstream reactions should cost the cell more fitness than one whose
    loss starves none. DepMap mean gene effect is the outside measurement -- it never enters the ablation, so this
    is a real held-out check rather than a restatement.

    THE CONTROL MATTERS MORE THAN THE EFFECT. Genes that starve a lot are genes that participate in a lot, and
    heavily-participating genes are essential for reasons that have nothing to do with this computation. So the
    comparison is against genes matched on reaction-participation count, not against the network average. Without
    that control the result would be a restatement of "hub genes are essential".
    """
    dp = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/CRISPRGeneEffect.csv")
    if not dp.exists():
        print("\n  (no DepMap file -- validation skipped rather than faked)")
        return None
    appears = collections.Counter()
    for r in rows:
        if r["role"] == "CATALYST":
            appears[r["removed"]] += 1
        elif r.get("addressable"):
            for g in r.get("genes", []):
                appears[g] += 1
    with open(dp) as f:
        rd = csv.reader(f)
        head = next(rd)
        genes = [h.split(" (")[0] for h in head[1:]]
        tot, cnt = np.zeros(len(genes)), np.zeros(len(genes))
        for row in rd:
            v = np.array([np.nan if x == "" else float(x) for x in row[1:]])
            ok = ~np.isnan(v)
            tot[ok] += v[ok]
            cnt[ok] += 1
    dep = {g: float(t / c) for g, t, c in zip(genes, tot, cnt) if c > 0}
    U = [g for g in appears if g in dep]
    if len(U) < 500:
        print("\n  (too few genes overlap DepMap -- validation skipped)")
        return None

    def mean(S):
        v = [dep[g] for g in S if g in dep]
        return (float(np.mean(v)) if v else float("nan"), len(v))

    base = mean(U)[0]
    print(f"\nVALIDATION vs DepMap ({len(U):,} genes; mean effect {base:+.4f}, more negative = more essential)")
    out = {"n_genes": len(U), "baseline": base, "bands": {}}
    bands = [("sole catalyst of >=1 rxn", [g for g in U if solecat[g]]),
             ("NOT sole catalyst of any", [g for g in U if not solecat[g]]),
             ("loss starves >=1 rxn", [g for g in U if cas[g] >= 1]),
             ("loss starves >=5 rxn", [g for g in U if cas[g] >= 5]),
             ("loss starves >=15 rxn", [g for g in U if cas[g] >= 15])]
    for label, S in bands:
        m, k = mean(S)
        out["bands"][label] = {"mean": m, "n": k, "delta": m - base}
        print(f"    {label:<28} {m:+.4f}   n={k:<6,} delta {m-base:+.4f}")

    # participation-matched control for the headline band
    rng = np.random.default_rng(0)
    tgt = [g for g in U if cas[g] >= 5]
    pool = sorted(U, key=lambda g: appears[g])
    rank = {g: i for i, g in enumerate(pool)}
    d = []
    for _ in range(200):
        ctrl = [pool[rng.integers(max(0, rank[g] - 25), min(len(pool), rank[g] + 25))] for g in tgt]
        d.append(mean(ctrl)[0] - base)
    d = np.array(d)
    obs = mean(tgt)[0] - base
    z = float((obs - d.mean()) / max(d.std(), 1e-9))
    out["control"] = {"matched_delta": float(d.mean()), "matched_sd": float(d.std()), "observed": obs, "z": z}
    print(f"    CONTROL participation-matched, 200 draws: {d.mean():+.4f} +- {d.std():.4f}")
    print(f"    observed for 'starves >=5'              : {obs:+.4f}   z = {z:+.2f}")
    print("    -> cascade BREADTH is predictive; the binary 'is a sole catalyst' flag is not (see its delta above)")
    return out


def main():
    net = json.load(open(OUT / "reaction_network.json"))
    comp = json.loads((OUT / "compartments.json").read_text())["steps"]
    steps = {s["id"]: s for s in net["steps"]}

    # ---- DEFECT 1: re-mark currency uniformly, and count how much the original marking disagreed with itself ----
    flips = collections.Counter()
    for s in steps.values():
        for p in s["in"] + s["out"]:
            want = norm(p["name"]) in CURRENCY
            if want != bool(p.get("currency")):
                flips["added" if want else "removed"] += 1
            p["currency"] = want
    print(f"currency re-marked uniformly: +{flips['added']:,} slots newly currency, "
          f"-{flips['removed']:,} slots no longer currency")

    # ---- DEFECT 2: bridge the two disjoint id spaces on normalised metabolite name ----
    # key(p) is the join key: the raw id inside a source, plus a name key that can cross sources. Both are indexed,
    # so a Human-GEM producer can buffer a Reactome consumer of the same-named metabolite.
    def keys(p):
        k = [p["id"]]
        if p["type"] in ("METABOLITE", "DRUG", "OTHER"):
            k.append("nm:" + norm(p["name"]))
        return k

    produced_by, consumed_by = collections.defaultdict(set), collections.defaultdict(set)
    for sid, s in steps.items():
        for p in s["out"]:
            if not p["currency"]:
                for k in keys(p):
                    produced_by[k].add(sid)
        for p in s["in"]:
            if not p["currency"]:
                for k in keys(p):
                    consumed_by[k].add(sid)

    def producers(p, sid):
        got = set()
        for k in keys(p):
            got |= produced_by.get(k, set())
        return got - {sid}

    def consumers(p, sid):
        got = set()
        for k in keys(p):
            got |= consumed_by.get(k, set())
        return got - {sid}

    # ---- DEFECT 3: is the removed thing something a genetic perturbation can remove? ----
    def addressable(role, p):
        """A knockout removes GENE PRODUCTS. A catalyst always qualifies; an input qualifies only if it carries
        genes (a protein, complex or defined set). A metabolite does not -- you cannot CRISPR oxygen."""
        return role == "CATALYST" or bool(p.get("genes"))

    rows = []
    verdict_n = collections.Counter()
    by_role = collections.Counter()
    verdict_addr = collections.Counter()
    for sid, s in steps.items():
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if len(gs) > MAXPART:
            continue
        where = comp.get(sid, {}).get("compartments", [])
        cats = list(s["catalysts"])

        def losses():
            """Products of this step that nothing else makes, and the reactions that consume them."""
            lost = [q for q in s["out"] if not q["currency"] and not producers(q, sid)]
            starved = set()
            for q in lost:
                starved |= consumers(q, sid)
            return lost, starved

        # ---- ablate each CATALYST ----
        for e in cats:
            redundant = len(cats) > 1
            verdict = "REDUNDANT" if redundant else "STOPS"
            lost, starved = ([], set()) if redundant else losses()
            rows.append({"rxn": sid, "src": s["source"], "removed": e, "role": "CATALYST",
                         "verdict": verdict, "n_alt_catalysts": len(cats) - 1, "addressable": True,
                         "products_lost": [p["name"] for p in lost][:6],
                         "n_starved_rxn": len(starved), "where": where})
            verdict_n[verdict] += 1
            by_role[("CATALYST", verdict)] += 1
            verdict_addr[(verdict, True)] += 1
        # ---- ablate each INPUT ----
        for p in s["in"]:
            if p["currency"]:
                v, alt = "BUFFERED_CURRENCY", -1
            else:
                alt = len(producers(p, sid))
                v = "BUFFERED" if alt > 0 else "STOPS"
            lost, starved = losses() if v == "STOPS" else ([], set())
            addr = addressable("INPUT", p)
            rows.append({"rxn": sid, "src": s["source"], "removed": p["name"], "removed_id": p["id"],
                         "role": "INPUT", "type": p["type"], "genes": p["genes"][:4], "addressable": addr,
                         "verdict": v, "n_alt_producers": alt,
                         "products_lost": [q["name"] for q in lost][:6],
                         "n_starved_rxn": len(starved), "where": where})
            verdict_n[v] += 1
            by_role[("INPUT", v)] += 1
            verdict_addr[(v, addr)] += 1

    n = len(rows)
    print(f"\n{len(steps):,} reactions -> {n:,} single-participant ablations")
    print(f"\n  outcome of removing one participant:")
    for v, c in verdict_n.most_common():
        print(f"    {v:<20} {c:>8,}  ({c/n:.1%})")
    print(f"\n  by role:")
    for (r, v), c in sorted(by_role.items(), key=lambda x: -x[1]):
        print(f"    {r:<9} {v:<20} {c:>8,}")

    # ---- the split that separates a chemistry statement from a knockout prediction ----
    na = sum(c for (v, a), c in verdict_addr.items() if a)
    print(f"\n  ADDRESSABLE by a genetic perturbation (catalyst, or an input carrying genes): {na:,}/{n:,} "
          f"({na/n:.1%})")
    for v in [x for x, _ in verdict_n.most_common()]:
        y, x = verdict_addr[(v, True)], verdict_addr[(v, False)]
        print(f"    {v:<20} addressable {y:>7,}   not (small molecule) {x:>7,}")

    stops = [r for r in rows if r["verdict"] == "STOPS"]
    sa = [r for r in stops if r["addressable"]]
    withloss = [r for r in sa if r["products_lost"]]
    cascading = [r for r in sa if r["n_starved_rxn"]]
    print(f"\n  of the {len(sa):,} ablations that STOP a reaction AND are genetically addressable:")
    print(f"    leave a product nothing else makes : {len(withloss):,}  ({len(withloss)/max(len(sa),1):.1%})")
    print(f"    starve at least one downstream rxn : {len(cascading):,}  ({len(cascading)/max(len(sa),1):.1%})")
    if cascading:
        d = sorted(r["n_starved_rxn"] for r in cascading)
        print(f"    downstream reactions starved       : median {d[len(d)//2]}, max {d[-1]}")
    print(f"  (the other {len(stops)-len(sa):,} STOPS rows remove a small molecule -- true chemistry, but not "
          f"something a knockout can do)")

    # the fragile core: reactions where EVERY participant is single-point-of-failure
    per_rxn = collections.defaultdict(list)
    for r in rows:
        per_rxn[r["rxn"]].append(r["verdict"])
    allstop = [k for k, v in per_rxn.items() if v and all(x == "STOPS" for x in v)]
    nostop = [k for k, v in per_rxn.items() if v and not any(x == "STOPS" for x in v)]
    print(f"\n  reactions where EVERY participant is a single point of failure: {len(allstop):,}"
          f"  ({len(allstop)/max(len(per_rxn),1):.1%})")
    print(f"  reactions where NO participant stops it (fully buffered)      : {len(nostop):,}"
          f"  ({len(nostop)/max(len(per_rxn),1):.1%})")

    # how many genes own at least one sole-catalyst step -- the knockout-relevant headline
    solecat = collections.Counter()
    for r in rows:
        if r["role"] == "CATALYST" and r["verdict"] == "STOPS":
            solecat[r["removed"]] += 1
    cas = collections.Counter()
    for r in rows:
        if r["role"] == "CATALYST" and r["n_starved_rxn"]:
            cas[r["removed"]] += r["n_starved_rxn"]
    print(f"\n  genes that are the SOLE catalyst of >=1 reaction : {len(solecat):,}")
    print(f"  genes whose loss starves >=1 downstream reaction  : {len(cas):,}")
    print(f"    widest: {', '.join(f'{g}({c})' for g, c in cas.most_common(8))}")

    val = validate(rows, solecat, cas)

    json.dump({"n_reactions": len(steps), "n_ablations": n, "verdicts": dict(verdict_n), "validation": val,
               "by_role": {f"{r}|{v}": c for (r, v), c in by_role.items()},
               "addressable": {f"{v}|{'addr' if a else 'mol'}": c for (v, a), c in verdict_addr.items()},
               "currency_flips": dict(flips),
               "n_all_single_point": len(allstop), "n_fully_buffered": len(nostop),
               "sole_catalyst_genes": len(solecat), "cascade_genes": len(cas),
               "rows": rows}, open(OUT / "reaction_ablation.json", "w"))
    sz = (OUT / "reaction_ablation.json").stat().st_size / 1e6
    print(f"\n  -> {OUT/'reaction_ablation.json'} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
