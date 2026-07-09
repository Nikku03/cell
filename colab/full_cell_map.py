"""full_cell_map — the COMPLETE perturbed cell: apply a genome across EVERY layer we built, ML values included,
then conclude. This is the honest 'whole cancer cell = cell model + ML', not a single-layer signalling map.

Layers integrated per genome:
  A. MOLECULAR   (role + domain position + DepMap selective-dependency of the gene)      [call each mutation]
  B. PATHWAY     (signed causal+signalling propagation -> hyperactive / lost programme)  [cancer_cell_map]
  C. REGULATORY  (which TFs are hit and how much of the transcriptome they command)       [reg layer]
  D. COMPLEXES   (which protein complexes lose a subunit to a LOF mutation)               [gene2cplx]
  E. METABOLIC   (which metabolic genes/reactions the mutations touch)                    [generxn]
  F. DEPENDENCY  (ML): co-essential partners of the drivers  +  synthetic-lethal partners of the lost
                 suppressors  +  the genome-wide SELECTIVE dependencies the tumour may fall into
                 (this is the only layer that can reach a WRN-type context dependency)    [DepMap, sl]
  G. CONCLUDE    integrate -> phenotype, driver, and the full target set (driver + dependencies), with the
                 honest gaps stated.

Honest boundaries proven before building: DepMap co-essentiality does NOT link WRN to the MMR genes (corr~0) —
a WRN-type dependency is a phenotype-context effect, so it can only appear in the genome-wide selective-dependency
scan, not as a co-essential partner; our SL set holds no WRN pair; and the ΔΔG model needs the exact residue
substitution the panels don't carry. Those are reported, not hidden.
-> outputs/orphan/full_cell_map.json
"""
import os, sys, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell
import cancer_cell_map as ccm

OUT = "outputs/orphan"


def _depmap():
    p = f"{OUT}/depmap_vecs.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    syms = list(z["syms"])
    return {"syms": syms, "Z": z["Z"], "col": {s: i for i, s in enumerate(syms)}}


def selective_dependencies(dm, min_lines=100):
    """genome-wide: genes that are STRONG SELECTIVE dependencies (lethal in a subset of lines, neutral overall) —
    the pool a context (MSI, HRD, amplification) can fall into. WRN-type targets live here, not in co-essentiality.
    Returns the FULL ranked list (most-selective first) so a panel gene's rank can be read off."""
    Z = dm["Z"]; sel = []
    for i, s in enumerate(dm["syms"]):
        row = Z[i]; nz = row[row != 0]
        if len(nz) < min_lines:
            continue
        frac = float((row < -3).mean())            # fraction of lines strongly dependent
        if frac > 0 and row.min() < -4 and abs(nz.mean()) < 0.6:   # selective, not pan-essential
            sel.append((frac, float(row.min()), s))
    sel.sort(reverse=True)
    return sel


def coessential(dm, gene, topn=8):
    if dm is None or gene not in dm["col"]:
        return []
    Z = dm["Z"]; x = Z[dm["col"][gene]]
    out = []
    for s, j in dm["col"].items():
        if s == gene:
            continue
        y = Z[j]; m = (x != 0) & (y != 0)
        if m.sum() < 80:
            continue
        c = float(np.corrcoef(x[m], y[m])[0, 1])
        if c > 0.2:
            out.append((round(c, 3), s))
    out.sort(reverse=True)
    return out[:topn]


def full_map(C, panel, dm=None, seldep=None):
    dm = dm if dm is not None else _depmap()
    seldep = seldep if seldep is not None else (selective_dependencies(dm) if dm else [])
    out = ccm._signed_out(C); paths = ccm._pathways(C)
    clamps, calls = ccm.classify(panel, C)

    # ---- A. molecular: enrich each call with domain + DepMap selectivity ----
    seldep_rank = {s: k for k, (_, _, s) in enumerate(seldep)}    # rank in the genome-wide selective-dep table
    for c in calls:
        g = c["gene"]
        if g in C.idx:
            dp = C.gene(g).get("domain_positions") or []
            pos = dict(panel).get(g)
            c["in_domain"] = [d["name"] for d in dp if d.get("start", 0) <= (pos or -1) <= d.get("end", 0)] or None
            c["selective_dep_rank"] = seldep_rank.get(g)          # None if not a selective dependency

    # ---- B. pathway layer ----
    pmap = ccm.plot_cell(C, panel, out, paths)

    # ---- C. regulatory: which mutated genes are TFs, and how much transcriptome they command ----
    reg_hits = []
    for c in calls:
        g = c["gene"]
        if g in C.idx and C.reg_out.get(C.idx[g]):
            reg_hits.append({"tf": g, "effect": c.get("effect"), "n_targets": len(C.reg_out[C.idx[g]])})
    reg_hits.sort(key=lambda r: -r["n_targets"])

    # ---- D. complexes disrupted by a LOF subunit ----
    g2c = C.D.get("gene2cplx", {})
    cplx_hit = defaultdict(list)
    for i, sgn in clamps.items():
        if sgn < 0:                                   # a lost subunit disrupts its complexes
            for cid in g2c.get(str(i), g2c.get(i, [])):
                cplx_hit[cid].append(C.name[i])
    complexes = sorted(cplx_hit.items(), key=lambda kv: -len(kv[1]))[:8]

    # ---- E. metabolic genes touched ----
    gen = set(int(k) for k in C.D.get("generxn", {}))
    met_hit = [C.name[i] for i in clamps if i in gen]

    # ---- F. dependency layer (ML) ----
    driver_gof = [c["gene"] for c in calls if c.get("clamp") == 1]
    coess = {g: coessential(dm, g) for g in driver_gof} if dm else {}
    # SL partners of the LOST suppressors (our sl set)
    sl = C.D.get("sl", [])
    lost = {i for i, s in clamps.items() if s < 0}
    sl_part = []
    for a, b, sc in sl:
        if a in lost or b in lost:
            partner = b if a in lost else a
            if partner < len(C.name):
                sl_part.append((C.name[a] if a in lost else C.name[b], C.name[partner], round(sc, 2)))

    # ---- G. conclude ----
    concl = {
        "driver": pmap.get("driver"),
        "phenotype_pathways_up": [p["pathway"] for p in pmap["hyperactive_pathways"][:4]],
        "phenotype_pathways_lost": [p["pathway"] for p in pmap["lost_pathways"][:4]],
        "dependency_targets": {
            "coessential_with_driver": {g: v[:5] for g, v in coess.items() if v},
            "synthetic_lethal_of_lost_suppressors": sl_part[:8],
            "context_selective_dependencies_genomewide": [(s, round(mn, 1)) for _, mn, s in seldep[:12]],
        },
    }
    return {
        "panel": [(g, p) for g, p in panel],
        "A_molecular": calls,
        "B_pathway": {"up": pmap["hyperactive_pathways"][:6], "lost": pmap["lost_pathways"][:6],
                      "n_genes_perturbed": pmap["n_genes_perturbed"]},
        "C_regulatory": reg_hits[:6],
        "D_complexes_disrupted": [{"complex": cid, "lost_subunits": subs} for cid, subs in complexes],
        "E_metabolic_genes_hit": met_hit,
        "F_dependency": concl["dependency_targets"],
        "G_conclusion": concl,
    }


def run(panels=None):
    C = CompleteCell()
    dm = _depmap()
    seldep = selective_dependencies(dm) if dm else []
    print(f"  DepMap selective-dependency pool: {len(seldep)} genes  (top: {[s for _,_,s in seldep[:8]]})")
    panels = panels or {
        "A375 melanoma": [("BRAF", 600), ("CDKN2A", 58), ("TTN", 20000)],
        "MSI-H colorectal": [("MLH1", 300), ("MSH6", 1088), ("PMS2", 600), ("TGFBR2", 128), ("ACVR2A", 400),
                             ("RNF43", 117), ("BAX", 41), ("BRAF", 600), ("B2M", 50), ("JAK1", 860), ("WRN", 577)],
    }
    report = {}
    for name, panel in panels.items():
        r = full_map(C, panel, dm, seldep)
        report[name] = r
        _print(name, r)
    json.dump(report, open(f"{OUT}/full_cell_map.json", "w"), indent=2)
    return report


def _print(name, r):
    print("=" * 88)
    print(f"COMPLETE CELL MAP — {name}")
    print("=" * 88)
    print("A. MOLECULAR (per-mutation call):")
    for c in r["A_molecular"]:
        rk = c.get("selective_dep_rank")
        sd = f"  [selective-dep #{rk}]" if rk is not None else ""
        dom = f"  dom={c['in_domain']}" if c.get("in_domain") else ""
        print(f"     {c['gene']:8} {c.get('role','-'):11} {c.get('effect','-'):22}{dom}{sd}")
    b = r["B_pathway"]
    print(f"B. PATHWAY  ({b['n_genes_perturbed']} genes changed):")
    print(f"     UP:   " + " | ".join(p["pathway"][:34] for p in b["up"][:3]))
    print(f"     LOST: " + " | ".join(p["pathway"][:34] for p in b["lost"][:3]))
    reg_str = ", ".join("{}({})".format(h["tf"], h["n_targets"]) for h in r["C_regulatory"][:5])
    print("C. REGULATORY (TFs hit): " + (reg_str or "none"))
    print(f"D. COMPLEXES disrupted: {len(r['D_complexes_disrupted'])}  " +
          ("e.g. " + ", ".join(str(d['complex']) for d in r['D_complexes_disrupted'][:3]) if r['D_complexes_disrupted'] else ""))
    print(f"E. METABOLIC genes hit: {r['E_metabolic_genes_hit'] or 'none'}")
    dep = r["F_dependency"]
    print("F. DEPENDENCY (ML):")
    for g, v in dep["coessential_with_driver"].items():
        print(f"     co-essential w/ {g}: " + ", ".join(f"{s}({c})" for c, s in v))
    if dep["synthetic_lethal_of_lost_suppressors"]:
        print(f"     SL of lost suppressors: {dep['synthetic_lethal_of_lost_suppressors'][:4]}")
    print(f"     genome-wide selective deps (context targets): " +
          ", ".join(f"{s}" for s, _ in dep["context_selective_dependencies_genomewide"][:10]))
    print(f"G. CONCLUSION -> driver {r['G_conclusion']['driver']}")
    print()


if __name__ == "__main__":
    run()
