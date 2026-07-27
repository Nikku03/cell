"""BUILD 1 of 5 -- ENZYME CAPACITY: turn "an alternative exists" into "the alternative carries enough flux".

THE MEASURED PROBLEM THIS FIXES. The FBA cell finds 73% of its lethal calls correct but only 20% of what actually
kills K562 (recall 0.203). It systematically UNDER-calls lethality, and the reason is structural: flux balance
declares a gene tolerated the moment ANY alternative route exists. The reaction ablation hit the identical wall and
said so -- its BUFFERED verdict means "another producer is annotated", never "enough flux gets through". Without a
quantity, a route carrying 1% of the required flux is indistinguishable from one carrying all of it.

THE FIX. Every reaction gets a CAPACITY CEILING:

    v_max(reaction)  =  sum over its enzymes of  E0(enzyme) * kcat(enzyme)

E0 is the protein's measured abundance -- already present for 95.4% of genes from PaxDb. kcat is the turnover
number, which this project had at 0% coverage. Both are needed: abundance alone says how much enzyme is present,
kcat alone says how fast one copy works, and only the product is a rate.

    without capacity   knocking out one of two isozymes changes nothing; the survivor absorbs all the flux
    with capacity      the survivor absorbs only what ITS abundance x turnover allows, and if that is not enough
                       the reaction is throttled and the cell suffers -- which is the lethality FBA was missing

WHERE THE NUMBERS COME FROM, AND THE HONEST LIMITS.

  kcat   DLKcat's combined BRENDA + SABIO-RK table (Kcat_combination_0918.json), 17,010 records of which 2,439 are
         Homo sapiens, keyed by EC number. Human-GEM carries 6,652 ec-code annotations, so the join is EC-to-EC and
         needs no sequence matching. LIMIT: kcat is measured in vitro on a purified enzyme with a chosen substrate,
         at a temperature and pH that are not the cell's. It is the right order of magnitude, not the right number.

  E0     PaxDb ppm. LIMIT: ppm is a relative abundance, not molecules per cell, so the absolute scale of v_max is
         arbitrary. That is why the capacity is applied as a RELATIVE constraint, calibrated so the wild-type cell
         still grows, rather than as an absolute mmol/gDW/h ceiling. The comparison it supports is between
         perturbed and wild-type, which is what the DepMap test asks anyway.

  EC->kcat is MANY-TO-MANY. One EC number has many measured substrates with kcat spanning orders of magnitude. The
  MEDIAN over human records is used, and the spread is reported, because picking the maximum would silently restore
  the "everything is buffered" behaviour this file exists to remove.

THE ACCEPTANCE TEST, SET IN ADVANCE (DESIGN.md sec 3): FBA recall must rise from 0.203 without precision falling
below 0.6. If enzyme-capacity constraints do not recover missed essentials, the hypothesis that QUANTITY is the
blocker is wrong and is reported as such.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SBML = SP / "HumanGEM.xml"
KCAT = SP / "dlkcat.tsv"          # DLKcat's combined BRENDA+SABIO table, despite the name it is JSON


def load_kcat_by_ec(organism="Homo sapiens"):
    """EC -> median kcat (1/s) over human measurements, with the spread kept for reporting."""
    recs = json.load(open(KCAT))
    by = collections.defaultdict(list)
    for r in recs:
        if r.get("Organism") != organism:
            continue
        ec = (r.get("ECNumber") or "").strip()
        try:
            v = float(r.get("Value"))
        except (TypeError, ValueError):
            continue
        if ec and v > 0 and np.isfinite(v):
            by[ec].append(v)
    out = {ec: float(np.median(v)) for ec, v in by.items()}
    spread = [max(v) / min(v) for v in by.values() if len(v) > 1 and min(v) > 0]
    print(f"kcat: {len(recs):,} records, {sum(len(v) for v in by.values()):,} for {organism}; "
          f"{len(out):,} distinct EC numbers")
    if spread:
        print(f"  within-EC spread (max/min, EC with >1 measurement): median {np.median(spread):.0f}x, "
              f"p90 {np.percentile(spread, 90):.0f}x  <- why the MEDIAN is used, not the max")
    return out


def reaction_ec():
    """reaction id -> set of EC numbers, parsed from the SBML annotation block."""
    rid, out = None, collections.defaultdict(set)
    pat = re.compile(r'ec-code/([0-9]+\.[0-9]+\.[0-9]+\.[0-9-]+)')
    with open(SBML) as f:
        for line in f:
            m = re.search(r'<reaction [^>]*\bid="([^"]+)"', line)
            if m:
                rid = m.group(1)
            if rid:
                for ec in pat.findall(line):
                    out[rid].add(ec)
            if "</reaction>" in line:
                rid = None
    return out


def main():
    import cobra
    cobra.Configuration().solver = "glpk"
    kcat = load_kcat_by_ec()
    r2ec = reaction_ec()
    print(f"Human-GEM: {len(r2ec):,} reactions carry an ec-code annotation")

    m = cobra.io.read_sbml_model(str(SBML))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cell_sim import set_medium, ensembl_to_symbol, depmap_k562, border   # reuse the validated setup
    set_medium(m)
    e2s = ensembl_to_symbol()

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items()
           if k.isdigit() and int(k) < len(names) and v}
    print(f"abundance: {len(ppm):,} genes with PaxDb ppm")

    # ---- capacity per reaction: sum over its enzymes of abundance x turnover ----
    cap, why = {}, collections.Counter()
    for r in m.reactions:
        if r.boundary:
            continue
        ecs = r2ec.get(r.id, set())
        ks = [kcat[e] for e in ecs if e in kcat]
        if not ks:
            why["no kcat for any EC"] += 1
            continue
        gs = [e2s.get(g.id, g.id) for g in r.genes]
        ab = [ppm[g] for g in gs if g in ppm]
        if not ab:
            why["no abundance for any enzyme"] += 1
            continue
        cap[r.id] = float(np.median(ks) * sum(ab))     # 1/s x ppm -- an arbitrary but CONSISTENT scale
        why["capacity assigned"] += 1
    print(f"\ncapacity: {len(cap):,} reactions constrained; {dict(why.most_common())}")
    if not cap:
        print("  NO reactions got a capacity -- the EC or abundance join failed. Refusing to report a result.")
        return

    v = np.array(list(cap.values()))
    print(f"  capacity spans {v.min():.3g} to {v.max():.3g} (median {np.median(v):.3g})")

    # ---- calibrate: scale so the wild-type cell still grows, then measure what the constraint does ----
    wt_free = m.slim_optimize()
    print(f"\nwild-type growth WITHOUT capacity constraints: {wt_free:.4f}")
    results = {}
    for scale in (1e-2, 1e-3, 1e-4, 1e-5):
        with m:
            nb = 0
            for rid, c in cap.items():
                r = m.reactions.get_by_id(rid)
                lim = c * scale
                if lim < abs(r.upper_bound):
                    r.upper_bound = min(r.upper_bound, lim)
                    if r.lower_bound < 0:
                        r.lower_bound = max(r.lower_bound, -lim)
                    nb += 1
            g = m.slim_optimize()
        results[scale] = (g, nb)
        print(f"  scale {scale:>7.0e}: {nb:>5,} reactions actually bound, growth {g:.4f} "
              f"({100*g/max(wt_free,1e-9):.1f}% of unconstrained)")

    # pick the tightest scale that still leaves the cell viable -- the point is to constrain, not to kill
    usable = [(s, g, n) for s, (g, n) in results.items() if g > 0.05 * wt_free]
    if not usable:
        print("\n  every scale kills the cell; capacity units are wrong. Reported, not tuned away.")
        return
    scale, gwt, nb = min(usable, key=lambda t: t[0])
    print(f"\nusing scale {scale:.0e}: {nb:,} reactions bound, wild-type growth {gwt:.4f}")

    # ---- THE ACCEPTANCE TEST: does recall rise from 0.203 with precision >= 0.6? ----
    from cobra.flux_analysis import single_gene_deletion
    dep = depmap_k562()
    with m:
        for rid, c in cap.items():
            r = m.reactions.get_by_id(rid)
            lim = c * scale
            if lim < abs(r.upper_bound):
                r.upper_bound = min(r.upper_bound, lim)
                if r.lower_bound < 0:
                    r.lower_bound = max(r.lower_bound, -lim)
        ko = single_gene_deletion(m)
    pred = {}
    for ids, val in zip(ko["ids"], ko["growth"]):
        g = list(ids)[0] if isinstance(ids, (set, frozenset)) else str(ids)
        pred[e2s.get(g, g)] = 0.0 if val is None or np.isnan(val) else float(val)
    ratio = {g: (x / gwt if gwt > 1e-9 else 0.0) for g, x in pred.items()}
    common = [g for g in ratio if g in dep]
    print(f"\nACCEPTANCE TEST -- {len(common):,} genes in both model and DepMap K562")
    if len(common) < 500:
        print("  JOIN TOO SMALL -- refusing to report (this is the failure mode that bit four times today)")
        return
    y = np.array([1 if dep[g] < -0.5 else 0 for g in common])
    for thr in (0.5, 0.95):
        leth = [g for g in common if ratio[g] < thr]
        tp = sum(1 for g in leth if dep[g] < -0.5)
        prec = tp / max(len(leth), 1)
        rec = tp / max(int(y.sum()), 1)
        print(f"  lethal if growth < {thr:.2f} of WT: {len(leth):>5,} called, "
              f"precision {prec:.3f}, recall {rec:.3f}  (baseline: precision 0.733, recall 0.203)")
        results[f"thr{thr}"] = {"n_called": len(leth), "precision": prec, "recall": rec}
    x = np.array([-ratio[g] for g in common])
    order = np.argsort(x)
    rr = np.empty(len(x), float)
    rr[order] = np.arange(1, len(x) + 1)
    auc = float((rr[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum()))
    print(f"  AUC {auc:.4f}   (baseline 0.6558)")

    json.dump({"n_ec_kcat": len(kcat), "n_reactions_with_ec": len(r2ec), "n_capacity": len(cap),
               "scale": scale, "wt_growth": gwt, "auc": auc,
               "results": {str(k): v for k, v in results.items()}},
              open(OUT / "ec_capacity.json", "w"), indent=1)
    print(f"\n  -> {OUT/'ec_capacity.json'}")


if __name__ == "__main__":
    main()
