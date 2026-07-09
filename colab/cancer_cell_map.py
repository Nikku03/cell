"""cancer_cell_map — stop chasing 'the target'; PLOT THE WHOLE PERTURBED CELL.

The goal was never a drug target — it was a complete cell you can perturb: apply a genome's mutations (or knock
out a protein, or delete a pathway), propagate the consequences through every signed layer, and read the ENTIRE
altered-cell state off the map. The target is then just one thing you read off it (the node driving the broken
pathways), and — crucially — an activating mutation that isn't near any active site (JAK2 V617F) is no longer
invisible: JAK2 is a known oncogene, so it is set ON, and JAK-STAT lights up DOWNSTREAM in the map. The signal is
the pathway state, not the mutation's location.

Pipeline (genome in -> whole perturbed cell out):
  1. CALL each mutation's effect by BIOLOGICAL ROLE, not proximity:
        oncogene   -> gain of function  -> clamp node ON   (+1)     [Cancer-Gene-Census role, a known layer]
        suppressor -> loss of function  -> clamp node OFF  (-1)
        (destabilizing ΔΔG / truncation refine unknowns toward LOF; otherwise neutral = passenger, no clamp)
  2. PROPAGATE the clamps through the signed causal+signalling network (decayed signed walks): every gene gets a
     net perturbation score (+ = driven up, - = driven down).
  3. AGGREGATE to pathways -> which pathways are HYPERACTIVE and which are LOST = the cancerous-cell map.
  4. READ OFF phenotype (hallmark pathways up/down), the driver(s), and the reversal point.

Honest scope: this is a TOPOLOGICAL/sign propagation (direction of change), not a kinetic simulation (how much,
how fast) — kinetics stays the model's wall. GOF/LOF is called from curated gene role + ΔΔG, not invented. It is
the whole-cell consequence map of a genome, generalising to any point mutation, protein knockout, or pathway
deletion — not only cancer.
-> outputs/orphan/cancer_cell_map.json
"""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
ALPHA = 0.7      # per-hop decay (weights are row-normalised, so signal stays bounded)
KHOP = 4         # propagation depth
THRESH = 0.01    # |perturbation| below this = untouched (noise floor after row-normalisation)

# curated Cancer-Gene-Census roles — a KNOWN-BIOLOGY layer (which genes activate cancer when hit vs which are
# tumour suppressors). This is the annotation a researcher uses; it is NOT the per-tumour answer key.
ONCOGENE = {
    "BRAF", "KRAS", "NRAS", "HRAS", "EGFR", "ERBB2", "ERBB3", "KIT", "PDGFRA", "PDGFRB", "MET", "ALK", "ROS1",
    "RET", "FLT3", "JAK1", "JAK2", "JAK3", "ABL1", "SRC", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "PIK3CA", "PIK3CB",
    "AKT1", "AKT2", "MTOR", "MYC", "MYCN", "MYCL", "MDM2", "MDM4", "CCND1", "CCNE1", "CDK4", "CDK6", "CTNNB1",
    "IDH1", "IDH2", "GNAQ", "GNA11", "GNAS", "EZH2", "MPL", "SMO", "MYD88", "ESR1", "AR", "BCL2", "NOTCH1",
    "CALR", "CBL", "CRLF2", "FOXL2", "HRAS", "KDR", "MAP2K1", "MAPK1", "RAC1", "RHOA", "STAT3", "STAT5B",
}
SUPPRESSOR = {
    "TP53", "RB1", "PTEN", "APC", "VHL", "BRCA1", "BRCA2", "CDKN2A", "CDKN2B", "CDKN1B", "SMAD4", "SMAD2",
    "STK11", "NF1", "NF2", "TSC1", "TSC2", "ARID1A", "ARID1B", "ARID2", "CDH1", "KEAP1", "BAP1", "ATM", "ATR",
    "MLH1", "MSH2", "MSH6", "PMS2", "WT1", "SMARCB1", "SMARCA4", "MEN1", "FBXW7", "PBRM1", "SETD2", "NF1",
    "PTCH1", "DAXX", "ATRX", "CIC", "FUBP1", "GATA3", "RUNX1", "CEBPA", "TET2", "DNMT3A", "ASXL1", "EP300",
    "CREBBP", "KMT2D", "KMT2C", "NCOR1", "SPEN", "CHEK2", "PALB2", "RAD51", "STAG2",
    "TGFBR2", "ACVR2A", "RNF43", "BAX", "TGFBR1", "AXIN2",       # recurrent MSI-H frameshift-target suppressors
}


def _signed_out(C):
    """merge the curated signed directed layers (SIGNOR/CollecTRI causal + signalling) into {u:[(w,weight)]},
    ROW-NORMALISED: each source's outgoing signs are divided by its out-degree, so a hub distributes bounded
    signal (total |outflow| <= 1) instead of dumping full strength into hundreds of nodes. Without this the
    propagation floods the whole network (the artefact seen in v1). Signs are preserved; only magnitude is
    normalised."""
    raw = defaultdict(list)
    for u, lst in (C.causal_out or {}).items():
        for item in lst:
            w = item[0]; s = item[1] if len(item) > 1 else 0
            if s and w < len(C.name):
                raw[u].append((w, 1 if s > 0 else -1))
    for u, lst in (getattr(C, "sig_out", {}) or {}).items():
        for item in lst:
            w = item[0]; s = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else 1
            if s:
                raw[u].append((w, 1 if s > 0 else -1))
    # U1: degradation edges (E3 -> substrate, sign -1). An E3 loss stabilises its substrate -- the edge type the
    # signalling network lacked (why VHL loss never lit up HIF2A). Added BEFORE row-normalisation so it shares the
    # bounded-outflow treatment.
    try:
        import degradation
        for a, b, s in degradation.signed_edges(C):
            if b < len(C.name):
                raw[a].append((b, s))
    except Exception:
        pass
    out = {}
    for u, edges in raw.items():
        d = len(edges)
        out[u] = [(w, s / d) for w, s in edges]        # row-normalise: bounded outflow per source
    return out


def classify(panel, C):
    """genome -> {gene_idx: +1 (GOF, clamp ON) | -1 (LOF, clamp OFF)} + per-gene call. Role first; ΔΔG/absence
    refine unknowns; passengers get no clamp."""
    clamps, calls = {}, []
    for sym, pos in panel:
        if sym not in C.idx:
            calls.append({"gene": sym, "call": "not in model", "clamp": 0}); continue
        i = C.idx[sym]
        if sym in ONCOGENE:
            clamps[i] = 1;  calls.append({"gene": sym, "role": "oncogene", "effect": "GOF (activated)", "clamp": 1})
        elif sym in SUPPRESSOR:
            clamps[i] = -1; calls.append({"gene": sym, "role": "suppressor", "effect": "LOF (lost)", "clamp": -1})
        else:
            # unknown role: only clamp if there is positive evidence of loss (disease-linked). Default passenger.
            ndis = (C.genes[i].get("ndis") or 0)
            if ndis >= 6:
                clamps[i] = -1; calls.append({"gene": sym, "role": "unknown", "effect": "LOF? (disease-linked)", "clamp": -1})
            else:
                calls.append({"gene": sym, "role": "unknown", "effect": "neutral / passenger", "clamp": 0})
    return clamps, calls


def propagate_signed(out, source_vals, alpha=ALPHA, khop=KHOP):
    """decayed SIGNED walks: inject each source's sign (+1 GOF / -1 LOF), spread along signed edges. Returns
    {node_idx: net perturbation}  (+ driven up, - driven down)."""
    v = defaultdict(float)
    cur = dict(source_vals)
    for _ in range(khop):
        nxt = defaultdict(float)
        for u, val in cur.items():
            for w, s in out.get(u, ()):
                nxt[w] += alpha * s * val
        for w, val in nxt.items():
            v[w] += val
        cur = nxt
    for i, val in source_vals.items():
        v[i] += val                               # the mutated node itself is perturbed
    return v


def _pathways(C):
    p = f"{OUT}/reactome_pathways.json"
    if not os.path.exists(p):
        return {}
    return json.load(open(p)).get("pathways", {})


# Reactome carries infection/viral pathways (host genes reused) that are irrelevant here and read as noise —
# drop them from the cell map so they can't masquerade as cancer biology.
_JUNK = ("sars-cov", "hiv", "influenza", "syncytial", "virus", "viral", "listeria", "measles", "epstein",
         "leishmania", "tuberculosis", "hepatitis", "rna polymerase iii", "7sl", "rrna", "trna")


def pathway_delta(pert, paths, min_hit=3):
    """per pathway: mean perturbation over its perturbed members. up (mean>0) = hyperactive; down = lost."""
    rows = []
    for name, members in paths.items():
        if any(k in name.lower() for k in _JUNK):
            continue
        hits = [pert[g] for g in members if g in pert and abs(pert[g]) > 1e-6]
        if len(hits) < min_hit:
            continue
        mean = sum(hits) / len(hits)
        rows.append({"pathway": name, "delta": round(mean, 3), "n_perturbed": len(hits),
                     "score": round(mean * len(hits) ** 0.5, 3)})
    up = sorted([r for r in rows if r["delta"] > 0], key=lambda r: -r["score"])
    down = sorted([r for r in rows if r["delta"] < 0], key=lambda r: r["score"])
    return up, down


def plot_cell(C, panel, out=None, paths=None):
    out = out if out is not None else _signed_out(C)
    paths = paths if paths is not None else _pathways(C)
    clamps, calls = classify(panel, C)
    pert = propagate_signed(out, clamps)
    up, down = pathway_delta(pert, paths)

    # whole-cell footprint: how many genes each mutation drives, and the top up/down genes
    reach = {}
    for i, sgn in clamps.items():
        single = propagate_signed(out, {i: sgn})
        reach[C.name[i]] = sum(1 for x, val in single.items() if abs(val) > THRESH)
    genes_up = sorted(((v, C.name[g]) for g, v in pert.items() if v > THRESH and g < len(C.name)), reverse=True)[:15]
    genes_dn = sorted(((v, C.name[g]) for g, v in pert.items() if v < -THRESH and g < len(C.name)))[:15]

    # driver = the GOF oncogene with the largest activating footprint (revert it -> collapse the hyperactive
    # programme). If there is no oncogene, the cell is suppressor-driven -> no direct target (SL territory).
    onco_hits = [(reach[c["gene"]], c["gene"]) for c in calls if c.get("clamp") == 1 and c["gene"] in reach]
    if onco_hits:
        driver = max(onco_hits)[1]
        target_note = (f"revert {driver} (GOF driver of the hyperactive programme) — its activation propagates to "
                       f"{max(onco_hits)[0]} genes; collapsing it should restore the map toward healthy")
    else:
        driver = None
        target_note = ("no activated oncogene — cell is tumour-SUPPRESSOR-driven (lost checkpoints/apoptosis); "
                       "no node to 'revert', this is synthetic-lethal territory, not a direct target")

    return {
        "panel": [(g, p) for g, p in panel],
        "mutation_calls": calls,
        "n_genes_perturbed": sum(1 for v in pert.values() if abs(v) > THRESH),
        "hyperactive_pathways": up[:12],
        "lost_pathways": down[:12],
        "top_genes_up": [(round(v, 3), n) for v, n in genes_up],
        "top_genes_down": [(round(v, 3), n) for v, n in genes_dn],
        "driver": driver, "target_note": target_note,
        "footprint_per_mutation": reach,
    }


def run(panels=None):
    C = CompleteCell()
    out = _signed_out(C); paths = _pathways(C)
    panels = panels or {
        "A375 melanoma":  [("BRAF", 600), ("CDKN2A", 58), ("TTN", 20000)],
        "HEL (JAK2)":     [("JAK2", 617), ("TP53", 248), ("TTN", 7000)],
        "H69 SCLC":       [("TP53", 175), ("RB1", 600), ("MYC", 100), ("TTN", 13000)],
        "K562 CML":       [("ABL1", 315), ("TP53", 234), ("TTN", 10000)],
    }
    report = {}
    for name, panel in panels.items():
        r = plot_cell(C, panel, out, paths)
        report[name] = r
        print("=" * 84)
        print(f"CANCEROUS CELL MAP — {name}   input mutations: {[p[0] for p in panel]}")
        print("=" * 84)
        for c in r["mutation_calls"]:
            print(f"    {c['gene']:8} {c.get('role','-'):11} -> {c.get('effect','-')}")
        print(f"  whole-cell footprint: {r['n_genes_perturbed']} genes changed state")
        print(f"  HYPERACTIVE pathways (driven up):")
        for p in r["hyperactive_pathways"][:6]:
            print(f"      +{p['delta']:<6} {p['pathway'][:66]}")
        print(f"  LOST pathways (driven down):")
        for p in r["lost_pathways"][:6]:
            print(f"      {p['delta']:<7} {p['pathway'][:66]}")
        print(f"  -> DRIVER / target: {r['driver']}")
        print(f"     {r['target_note']}")
        print()
    json.dump(report, open(f"{OUT}/cancer_cell_map.json", "w"), indent=2)
    return report


if __name__ == "__main__":
    run()
