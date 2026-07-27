"""BUILD 1 of 5 -- ENZYME CAPACITY: turn "an alternative exists" into "the alternative carries enough flux".

THE MEASURED PROBLEM THIS FIXES. The FBA cell gets 73% of its lethal calls right but finds only 20% of what
actually kills K562 (recall 0.203). It systematically UNDER-calls lethality, and the reason is structural: flux
balance declares a gene tolerated the moment ANY alternative route exists. The reaction ablation hit the identical
wall -- its BUFFERED verdict means "another producer is annotated", never "enough flux gets through". Without a
quantity, a route carrying 1% of the required flux is indistinguishable from one carrying all of it.

THE FIX. Every reaction gets a CAPACITY CEILING built from the GPR's own logic:

    capacity(reaction) = kcat(reaction) x  CAP( gene rule )
        CAP( a OR b )  = CAP(a) + CAP(b)        isozymes ADD -- two routes carry more than one
        CAP( a AND b ) = min( CAP(a), CAP(b) )  a complex runs at the rate of its scarcest subunit
        CAP( gene )    = abundance of that gene (PaxDb ppm)

and -- THIS IS THE WHOLE POINT -- the ceiling is RECOMPUTED for every deletion:

    capacity(reaction | gene g deleted) = kcat x CAP( gene rule, with abundance[g] := 0 )

    without capacity   knocking out one of two isozymes changes nothing; the survivor absorbs all the flux
    with capacity      the survivor absorbs only what ITS abundance x turnover allows, and if that is not enough
                       the reaction is throttled and the cell suffers -- the lethality FBA was missing

WHY THE FIRST VERSION OF THIS FILE FAILED, RECORDED RATHER THAN QUIETLY REPLACED. It imposed a STATIC capacity
ceiling and then called cobra's single_gene_deletion. That routine asks the GPR only WHETHER a reaction survives a
deletion, never how much capacity the survivors have left, so the ceilings never moved. Measured consequence: the
lethal set was bit-identical to the unconstrained model -- 86 genes called, precision 0.733, recall 0.203, exactly
the baseline -- while AUC fell 0.681 -> 0.591 because the ceilings flattened the growth-ratio ranking without
informing it. A static ceiling is not an enzyme constraint; it is a global speed limit. Both are reported below.

WHERE THE NUMBERS COME FROM, AND THE HONEST LIMITS.

  kcat   DLKcat's combined BRENDA + SABIO-RK table, 17,010 records of which 2,437 are Homo sapiens, covering 362
         distinct EC numbers, joined EC-to-EC against Human-GEM's 5,387 ec-code-annotated reactions. LIMIT: kcat is
         measured in vitro on purified enzyme with a chosen substrate, at a temperature and pH that are not the
         cell's. Right order of magnitude, not the right number. One EC also has many measured substrates spanning
         orders of magnitude (median 27x, p90 17,227x within an EC), so the MEDIAN is used -- the maximum would
         silently restore the "everything is buffered" behaviour this file exists to remove.

  E0     PaxDb ppm, 15,741 genes. LIMIT: ppm is relative, not molecules per cell, so ppm x 1/s is not
         mmol/gDW/h. ONE global unit conversion is unavoidable and is fitted to ONE number: the measured K562
         doubling time. It is fitted to nothing in the acceptance test -- DepMap essentiality never enters the fit.

  MISSING DATA IS NOT EVIDENCE OF ZERO CAPACITY. A reaction whose enzymes have no PaxDb entry gets NO ceiling
  rather than a ceiling of zero, because otherwise the model would kill the cell wherever the database is thin and
  score that as a correct lethal call.

THE ACCEPTANCE TEST, SET IN ADVANCE (DESIGN.md sec 3): FBA recall must rise from 0.203 without precision falling
below 0.6. If enzyme-capacity constraints do not recover missed essentials, the hypothesis that QUANTITY is the
blocker is wrong and is reported as such.
"""
import ast
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

K562_DOUBLING_H = 24.0            # K562 doubles in ~20-24 h; the calibration target, and the ONLY fitted number
LETHAL_THR = 0.5                  # a gene is called lethal if deleting it leaves < 50% of wild-type growth
N_SHUFFLE = 3                     # quantity-permutation controls


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


def load_kcat_hierarchical(verbose=True):
    """EC -> (kcat, provenance tier), using the fallback hierarchy standard in ecModel pipelines.

    WHY THIS EXISTS. Build 1 matched Homo sapiens records only and got 362 EC numbers, covering 1,811 of
    Human-GEM's 5,387 EC-annotated reactions. The DLKcat table holds 1,706 distinct EC numbers across all
    organisms; restricting to human threw away 1,344 of them. kcat is well conserved within an EC across
    orthologs, so the accepted practice is to fall back rather than to drop:

        tier 1  same EC, Homo sapiens                     most specific, fewest hits
        tier 2  same EC, any organism                     median over organisms
        tier 3  same first three EC fields, any organism  the enzyme CLASS median

    This is a COVERAGE upgrade, not an accuracy upgrade, and the distinction matters: build 7 showed that
    improving kcat ACCURACY changes nothing, because 1,578 of 1,651 ceilinged reactions carry no flux. Coverage
    is the axis that was actually binding, and it costs no new data.
    """
    recs = json.load(open(KCAT))
    hum, any_org = collections.defaultdict(list), collections.defaultdict(list)
    for r in recs:
        ec = (r.get("ECNumber") or "").strip()
        try:
            v = float(r.get("Value"))
        except (TypeError, ValueError):
            continue
        if not ec or v <= 0 or not np.isfinite(v):
            continue
        any_org[ec].append(v)
        if r.get("Organism") == "Homo sapiens":
            hum[ec].append(v)
    cls = collections.defaultdict(list)
    for ec, vs in any_org.items():
        parts = ec.split(".")
        if len(parts) >= 3:
            cls[".".join(parts[:3])].extend(vs)
    out, tier = {}, {}
    for ec in any_org:
        if hum.get(ec):
            out[ec], tier[ec] = float(np.median(hum[ec])), 1
        else:
            out[ec], tier[ec] = float(np.median(any_org[ec])), 2
    if verbose:
        n1 = sum(1 for t in tier.values() if t == 1)
        print(f"kcat (hierarchical): {len(out):,} EC numbers -- tier 1 human {n1:,}, "
              f"tier 2 cross-organism {len(out)-n1:,}; tier 3 class medians available for {len(cls):,} EC classes")
    return out, tier, {k: float(np.median(v)) for k, v in cls.items()}


def resolve_kcat(ecs, table, classes):
    """kcat for a reaction's EC set, falling through the hierarchy. Returns (value, tier) or (None, None)."""
    hits = [table[e] for e in ecs if e in table]
    if hits:
        return float(np.median(hits)), 2
    for e in ecs:
        pre = ".".join(e.split(".")[:3])
        if pre in classes:
            return float(classes[pre]), 3
    return None, None


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


# ---------------------------------------------------------------------------------------------------------------
# GPR arithmetic. Human-GEM gene ids are valid Python identifiers (ENSG...), and its rules use `and`/`or` with
# parentheses, so Python's own parser gives the tree for free -- no hand-rolled tokeniser to get subtly wrong.
# ---------------------------------------------------------------------------------------------------------------
def gpr_tree(rule):
    rule = (rule or "").strip()
    if not rule:
        return None
    try:
        return ast.parse(rule, mode="eval").body
    except SyntaxError:
        return None


def cap_of(node, abund):
    """CAP(a OR b) = CAP(a)+CAP(b); CAP(a AND b) = min; CAP(gene) = its abundance. Returns None when the whole
    branch has no abundance data at all, so "unmeasured" stays distinguishable from "measured as zero"."""
    if isinstance(node, ast.Name):
        return abund.get(node.id)
    if isinstance(node, ast.BoolOp):
        vals = [cap_of(v, abund) for v in node.values]
        known = [v for v in vals if v is not None]
        if not known:
            return None
        if isinstance(node.op, ast.Or):
            return float(sum(known))                    # unmeasured isozymes contribute nothing, not infinity
        return float(min(known))                        # a subunit with no data cannot be shown to be limiting
    return None


def gpr_alive(node, dead):
    """Standard boolean GPR: is the reaction still catalysed with `dead` genes removed?"""
    if isinstance(node, ast.Name):
        return node.id not in dead
    if isinstance(node, ast.BoolOp):
        vals = [gpr_alive(v, dead) for v in node.values]
        return any(vals) if isinstance(node.op, ast.Or) else all(vals)
    return True


def build_capacity(m, kcat, r2ec, abund):
    """reaction id -> (kcat, gpr tree, wild-type capacity). Only reactions with BOTH a kcat and abundance."""
    cap, tree, kc, why = {}, {}, {}, collections.Counter()
    for r in m.reactions:
        if r.boundary:
            continue
        t = gpr_tree(r.gene_reaction_rule)
        if t is None:
            why["no parsable GPR"] += 1
            continue
        ks = [kcat[e] for e in r2ec.get(r.id, set()) if e in kcat]
        if not ks:
            why["no kcat for any EC"] += 1
            continue
        c = cap_of(t, abund)
        if c is None or c <= 0:
            why["no abundance for any enzyme"] += 1
            continue
        kc[r.id], tree[r.id] = float(np.median(ks)), t
        cap[r.id] = float(np.median(ks) * c)
        why["capacity assigned"] += 1
    return cap, tree, kc, why


def apply_cap(m, cap, scale):
    n = 0
    for rid, c in cap.items():
        r = m.reactions.get_by_id(rid)
        lim = c * scale
        if lim < abs(r.upper_bound):
            r.upper_bound = min(r.upper_bound, lim)
            if r.lower_bound < 0:
                r.lower_bound = max(r.lower_bound, -lim)
            n += 1
    return n


def growth_at(m, cap, scale):
    with m:
        n = apply_cap(m, cap, scale)
        g = m.slim_optimize()
    return (0.0 if g is None or np.isnan(g) else float(g)), n


def calibrate(m, cap, target, lo=1e-12, hi=1e3):
    """Growth is monotone non-decreasing in the scale, so bisect in log space for the scale that reproduces the
    measured growth rate. One free parameter, one number fitted."""
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        g, _ = growth_at(m, cap, mid)
        if g < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0005:
            break
    return float(np.sqrt(lo * hi))


def deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, scale, dose):
    """Delete each gene and re-optimise.

    dose=False   the standard FBA deletion: a reaction dies only when its GPR goes false. This reproduces the
                 published baseline and is recomputed here rather than quoted, so both arms share a gene set.
    dose=True    additionally RECOMPUTES each surviving reaction's ceiling with the deleted gene's abundance set
                 to zero -- losing one of two isozymes now costs that isozyme's share of the capacity.
    """
    with m:
        if dose:
            apply_cap(m, cap, scale)
        wt = m.slim_optimize()
        out = {}
        for g in model_genes:
            dead = {g}
            touched = []
            for rid in r_of_gene.get(g, ()):
                r = m.reactions.get_by_id(rid)
                lo, hi = r.lower_bound, r.upper_bound
                t = tree.get(rid)
                if t is None:
                    t = gpr_tree(r.gene_reaction_rule)
                if t is not None and not gpr_alive(t, dead):
                    touched.append((r, lo, hi))
                    r.lower_bound, r.upper_bound = 0.0, 0.0
                elif dose and rid in cap:
                    a2 = dict(abund)
                    a2[g] = 0.0
                    c = cap_of(tree[rid], a2)
                    if c is None:
                        continue
                    lim = kc[rid] * c * scale
                    if lim < hi or (lo < 0 and -lim > lo):
                        touched.append((r, lo, hi))
                        r.upper_bound = min(hi, lim)
                        if lo < 0:
                            r.lower_bound = max(lo, -lim)
            v = m.slim_optimize()
            out[g] = 0.0 if v is None or np.isnan(v) else float(v)
            for r, lo, hi in touched:
                r.lower_bound, r.upper_bound = lo, hi
    return out, (0.0 if wt is None or np.isnan(wt) else float(wt))


def metrics(pred, wt, e2s, dep, tag):
    ratio = {e2s.get(g, g): (v / wt if wt > 1e-9 else 0.0) for g, v in pred.items()}
    common = sorted(g for g in ratio if g in dep)
    if len(common) < 500:
        print(f"  [{tag}] JOIN TOO SMALL ({len(common)}) -- refusing to report")
        return None
    y = np.array([1 if dep[g] < -0.5 else 0 for g in common])
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  [{tag}] degenerate label set -- refusing to report")
        return None
    leth = [g for g in common if ratio[g] < LETHAL_THR]
    tp = sum(1 for g in leth if dep[g] < -0.5)
    prec, rec = tp / max(len(leth), 1), tp / int(y.sum())
    x = np.array([-ratio[g] for g in common])
    o = np.argsort(x, kind="mergesort")
    rr = np.empty(len(x), float)
    rr[o] = np.arange(1, len(x) + 1)
    a = float((rr[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum()))
    print(f"  [{tag}] n={len(common):,} essential={int(y.sum()):,} | called {len(leth):,} "
          f"precision {prec:.3f} recall {rec:.3f} AUC {a:.4f}")
    return {"n": len(common), "n_essential": int(y.sum()), "n_called": len(leth),
            "precision": float(prec), "recall": float(rec), "auc": a}


def main():
    import cobra
    cobra.Configuration().solver = "glpk"
    kcat = load_kcat_by_ec()
    r2ec = reaction_ec()
    print(f"Human-GEM: {len(r2ec):,} reactions carry an ec-code annotation")

    m = cobra.io.read_sbml_model(str(SBML))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cell_sim import set_medium, ensembl_to_symbol, depmap_k562   # reuse the validated medium + joins
    set_medium(m)
    e2s = ensembl_to_symbol()

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items()
           if k.isdigit() and int(k) < len(names) and v}
    abund = {g.id: ppm[e2s[g.id]] for g in m.genes if g.id in e2s and e2s[g.id] in ppm}
    print(f"abundance: {len(ppm):,} genes with PaxDb ppm; {len(abund):,} of {len(m.genes):,} model genes joined")
    assert len(abund) > 1000, "ABUNDANCE JOIN TOO SMALL -- refusing to continue"

    cap, tree, kc, why = build_capacity(m, kcat, r2ec, abund)
    print(f"\ncapacity: {len(cap):,} reactions constrained; {dict(why.most_common())}")
    if not cap:
        print("  NO reactions got a capacity -- the EC or abundance join failed. Refusing to report a result.")
        return
    v = np.array(list(cap.values()))
    print(f"  capacity spans {v.min():.3g} to {v.max():.3g} (median {np.median(v):.3g})")

    r_of_gene = collections.defaultdict(list)
    for r in m.reactions:
        for g in r.genes:
            r_of_gene[g.id].append(r.id)
    dep = depmap_k562()
    model_genes = sorted(g.id for g in m.genes if e2s.get(g.id, g.id) in dep)
    reach = sum(1 for g in model_genes if any(r in cap for r in r_of_gene.get(g, ())))
    print(f"scored genes: {len(model_genes):,} model genes are in DepMap K562; {reach:,} of them "
          f"({100*reach/max(len(model_genes),1):.1f}%) touch at least one capacity-constrained reaction")
    print(f"  ^ this bounds how much the capacity arm CAN differ from the baseline, whatever it scores")

    # ---- calibrate the one free parameter against the MEASURED growth rate ----
    #
    # The first version picked the scale by "the cell must still grow at >5% of its unconstrained rate", and on
    # that criterion every scale failed. The criterion was the error: unconstrained growth is 20.56/h, a doubling
    # time of TWO MINUTES, already recorded as one of cell_sim.py's defects. Anchoring "alive" to a number known
    # to be wrong rejects the correct answer. The anchor is the measured doubling time.
    wt_free = m.slim_optimize()
    mu = float(np.log(2) / K562_DOUBLING_H)
    print(f"\nwild-type growth WITHOUT capacity: {wt_free:.4f}/h = {np.log(2)/max(wt_free,1e-9)*60:.1f} min doubling")
    print(f"measured K562: {K562_DOUBLING_H:.0f} h doubling = {mu:.4f}/h  <- the calibration target")
    scale = calibrate(m, cap, mu)
    gwt, nb = growth_at(m, cap, scale)
    print(f"calibrated scale {scale:.4g}: {nb:,} of {len(cap):,} reactions bound, "
          f"growth {gwt:.4f}/h ({np.log(2)/max(gwt,1e-9):.1f} h doubling)")
    if abs(gwt - mu) > 0.15 * mu:
        print("  calibration did not converge to the measured rate -- refusing to score.")
        return

    # ---- the three arms ----
    print("\nrunning deletion sweeps...")
    pb, wb = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, scale, dose=False)
    base = metrics(pb, wb, e2s, dep, "boolean GPR only (baseline)")
    pd_, wd = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, scale, dose=True)
    dose = metrics(pd_, wd, e2s, dep, "dose-aware enzyme capacity")
    if base is None or dose is None:
        return
    moved = sum(1 for g in model_genes if abs(pb[g] / max(wb, 1e-12) - pd_[g] / max(wd, 1e-12)) > 1e-6)
    print(f"  genes whose growth ratio actually changed: {moved:,} of {len(model_genes):,}")

    # ---- CONTROL: is it the measured QUANTITIES, or just the shape of the constraint? ----
    # Abundances are permuted across genes and kcats across ECs. Every reaction still gets a ceiling, the GPR
    # arithmetic is identical, the calibration is redone -- only the pairing of number to entity is destroyed.
    shuf = []
    rng = np.random.default_rng(0)
    for i in range(N_SHUFFLE):
        ks, vs = list(abund), list(abund.values())
        rng.shuffle(vs)
        a2 = dict(zip(ks, vs))
        ke, ve = list(kcat), list(kcat.values())
        rng.shuffle(ve)
        k2 = dict(zip(ke, ve))
        c2, t2, kc2, _ = build_capacity(m, k2, r2ec, a2)
        if not c2:
            continue
        s2 = calibrate(m, c2, mu)
        g2, _ = growth_at(m, c2, s2)
        if abs(g2 - mu) > 0.15 * mu:
            print(f"  shuffle {i}: calibration failed ({g2:.4f}/h), skipped")
            continue
        p2, w2 = deletion_sweep(m, model_genes, r_of_gene, t2, kc2, a2, c2, s2, dose=True)
        r2 = metrics(p2, w2, e2s, dep, f"SHUFFLED quantities #{i}")
        if r2:
            shuf.append(r2)

    print("\n" + "=" * 84)
    print(f"{'':36s} {'called':>7s} {'prec':>7s} {'recall':>7s} {'AUC':>8s}")
    for tag, r in [("boolean GPR only (baseline)", base), ("dose-aware enzyme capacity", dose)]:
        print(f"{tag:36s} {r['n_called']:>7,} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['auc']:>8.4f}")
    for i, r in enumerate(shuf):
        print(f"{'  SHUFFLED quantities #%d' % i:36s} {r['n_called']:>7,} {r['precision']:>7.3f} "
              f"{r['recall']:>7.3f} {r['auc']:>8.4f}")
    print("=" * 84)
    d_rec = dose["recall"] - base["recall"]
    passed = bool(d_rec > 0 and dose["precision"] >= 0.6)
    beat = bool(shuf) and dose["recall"] > max(s["recall"] for s in shuf)
    print(f"\nPRE-REGISTERED TEST (DESIGN.md sec 3): recall must RISE from the baseline, precision stay >= 0.6")
    print(f"  recall {base['recall']:.3f} -> {dose['recall']:.3f} ({d_rec:+.3f}), "
          f"precision {base['precision']:.3f} -> {dose['precision']:.3f} "
          f"({dose['precision']-base['precision']:+.3f})")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}"
          + (f"; beats every shuffled control: {beat}" if shuf else "; NO CONTROL RAN"))
    if passed and not beat:
        print("  ...but permuted quantities do as well, so the gain is the constraint's shape, not biochemistry.")

    json.dump({"n_ec_kcat": len(kcat), "n_reactions_with_ec": len(r2ec), "n_capacity": len(cap),
               "scale": scale, "wt_growth": gwt, "mu_target": mu, "wt_unconstrained": float(wt_free),
               "n_scored": len(model_genes), "n_touching_capacity": reach, "n_ratio_changed": moved,
               "baseline": base, "dose_capacity": dose, "shuffled": shuf,
               "static_ceiling_note": "a static ceiling + cobra single_gene_deletion left the lethal set "
                                      "bit-identical to baseline (86 called, prec 0.733, rec 0.203) and dropped "
                                      "AUC 0.681 -> 0.591; that is why deletion is dose-aware here",
               "passed": passed, "beats_control": beat},
              open(OUT / "ec_capacity.json", "w"), indent=1)
    print(f"\n  -> {OUT/'ec_capacity.json'}")


if __name__ == "__main__":
    main()
