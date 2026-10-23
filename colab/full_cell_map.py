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


# ---- ΔΔG structural layer: refine the LOF call from protein stability (ML) ----
def ddg_layer(calls, panel_meta):
    """for mutations with a known residue substitution, predict ΔΔG from the AlphaFold structure and refine the
    call: a strongly DESTABILIZING mutation is confident LOF; a stability-neutral one is consistent with GOF or a
    non-destabilizing LOF (so it does NOT overturn the role call, only corroborates a suppressor as LOF)."""
    try:
        import ddg_predictor as ddg
        P = ddg.DDGPredictor(f"{OUT}/ddg_model.pkl")
    except Exception as e:
        for c in calls:
            c["ddg"] = None
        return f"unavailable ({str(e)[:40]})"
    for c in calls:
        meta = panel_meta.get(c["gene"])
        if not meta:
            c["ddg"] = None; continue
        try:
            pdb = P.alphafold_pdb(meta["uniprot"])
            d, _ = P.predict_from_structure(pdb, "A", meta["pos"], meta["wt"], meta["mut"])
            c["ddg"] = round(d, 2)
            c["ddg_note"] = ("destabilizing -> LOF" if d > 1.5 else
                             "mildly destabilizing" if d > 0.6 else "stability-neutral")
        except Exception:
            c["ddg"] = None
    return "on"


# ---- cell-type baseline (emask): which tissue this cancer arises in, and its restricted genes ----
def emask_layer(C, tissue_kw):
    """the 200-cell-type healthy census -> pick the baseline tissue for this cancer. U8: try keywords in PRIORITY
    order (the specific tissue term first, generic ones like 'epitheli' only as a last resort) so an ovarian
    tumour no longer matches kidney epithelium via a generic substring."""
    ct = C.D.get("ctnames") or []
    if not ct or not tissue_kw:
        return None
    for k in tissue_kw:                                          # specific -> generic; take the first that hits
        hits = [i for i, nm in enumerate(ct) if k in nm.lower()]
        if hits:
            return {"tissue_types_matched": [ct[i] for i in hits][:6], "n_types": len(hits), "matched_on": k}
    return None


# ---- independent perturbation lens (fixed CellGraph, numpy) ----
def cellgraph_layer(C, clamps):
    """propagate the knockouts through the fixed CellGraph directed-signed network — an INDEPENDENT perturbation
    lens (different construction from the causal map) so agreement is corroboration, disagreement is a flag."""
    try:
        import cellgraph as cg
        import scipy.sparse as sp
        n = len(C.name)
        W = cg.directed_signed(C.D, n)                      # sparse signed out-adjacency
        seeds = [i for i, s in clamps.items() if s < 0]     # knockouts (lost genes)
        if not seeds or W is None:
            return None
        v = cg.perturb_downstream(np.array(seeds), W, alpha=0.5, hops=3)
        # U9: perturb_downstream is unnormalised and floods; keep only genes above a real magnitude threshold and
        # report the STRONGLY-affected count, not the raw fan-out.
        thr = max(0.05, float(np.percentile(np.abs(v[v != 0]), 99)) if (v != 0).any() else 0.05)
        idx = np.argsort(v)
        down = [(C.name[i], round(float(v[i]), 3)) for i in idx[:8] if v[i] < -thr]
        up = [(C.name[i], round(float(v[i]), 3)) for i in idx[::-1][:8] if v[i] > thr]
        return {"n_strongly_affected": int((np.abs(v) > thr).sum()), "top_down": down, "top_up": up}
    except Exception as e:
        return {"error": str(e)[:60]}


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


def full_map(C, panel, dm=None, seldep=None, panel_meta=None, tissue_kw=None, variant_meta=None):
    dm = dm if dm is not None else _depmap()
    seldep = seldep if seldep is not None else (selective_dependencies(dm) if dm else [])
    panel_meta = panel_meta or {}
    out = ccm._signed_out(C); paths = ccm._pathways(C)
    clamps, calls = ccm.classify(panel, C, variant_meta=variant_meta)

    # ---- A. molecular: enrich each call with domain + DepMap selectivity ----
    seldep_rank = {s: k for k, (_, _, s) in enumerate(seldep)}    # rank in the genome-wide selective-dep table
    for c in calls:
        g = c["gene"]
        if g in C.idx:
            dp = C.gene(g).get("domain_positions") or []
            pos = dict(panel).get(g)
            c["in_domain"] = [d["name"] for d in dp if d.get("start", 0) <= (pos or -1) <= d.get("end", 0)] or None
            c["selective_dep_rank"] = seldep_rank.get(g)          # None if not a selective dependency
            cdp = getattr(C, "context_dep", {}).get(C.idx[g])     # U13: surface the attribution inline
            if cdp:
                c["attribution"] = (f"addiction:{cdp['addiction_context'][0][0]}" if cdp.get("addiction_context")
                                    else f"lesion:{cdp['lesion_context']}" if cdp.get("lesion_context")
                                    else "ORPHAN (selective-dep, lesion unattributed — needs Colab table)")
    ddg_status = ddg_layer(calls, panel_meta)                     # structural LOF refinement (ML)

    # ---- B. pathway layer ----
    pmap = ccm.plot_cell(C, panel, out, paths, variant_meta=variant_meta)

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
    # ---- H. tissue baseline (emask) + I. independent perturbation lens (CellGraph) ----
    tissue = emask_layer(C, tissue_kw or ())
    cg = cellgraph_layer(C, clamps)

    return {
        "panel": [(g, p) for g, p in panel],
        "layers_active": {"ddg": ddg_status, "emask": bool(tissue), "cellgraph": bool(cg and "error" not in cg),
                          "depmap": dm is not None},
        "A_molecular": calls,
        "B_pathway": {"up": pmap["hyperactive_pathways"][:6], "lost": pmap["lost_pathways"][:6],
                      "n_genes_perturbed": pmap["n_genes_perturbed"]},
        "C_regulatory": reg_hits[:6],
        "D_complexes_disrupted": [{"complex": cid, "lost_subunits": subs} for cid, subs in complexes],
        "E_metabolic_genes_hit": met_hit,
        "F_dependency": concl["dependency_targets"],
        "G_cellgraph_perturbation": cg,
        "H_tissue_baseline": tissue,
        "conclusion": concl,
    }


# residue substitution + UniProt for the ΔΔG structural layer (hotspot alleles). Missing genes just skip ΔΔG.
META = {
    "BRAF":  {"pos": 600, "wt": "V", "mut": "E", "uniprot": "P15056"},
    "MLH1":  {"pos": 300, "wt": "R", "mut": "H", "uniprot": "P40692"},
    "MSH6":  {"pos": 1088, "wt": "R", "mut": "C", "uniprot": "P52701"},
    "TGFBR2":{"pos": 128, "wt": "R", "mut": "H", "uniprot": "P37173"},
    "JAK1":  {"pos": 860, "wt": "P", "mut": "S", "uniprot": "P23458"},
    "WRN":   {"pos": 577, "wt": "R", "mut": "C", "uniprot": "Q14191"},
    "CDKN2A":{"pos": 58,  "wt": "R", "mut": "H", "uniprot": "P42771"},
    "B2M":   {"pos": 50,  "wt": "S", "mut": "F", "uniprot": "P61769"},
}
TISSUE = {"A375 melanoma": ("melanocyte", "skin"), "MSI-H colorectal": ("colon", "intestin", "enterocyte")}


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
        r = full_map(C, panel, dm, seldep, panel_meta=META, tissue_kw=TISSUE.get(name))
        report[name] = r
        _print(name, r)
    json.dump(report, open(f"{OUT}/full_cell_map.json", "w"), indent=2)
    return report


def _print(name, r):
    print("=" * 88)
    print(f"COMPLETE CELL MAP — {name}")
    print("=" * 88)
    la = r.get("layers_active", {})
    print(f"[layers active: {', '.join(k for k,v in la.items() if v)}]")
    print("A. MOLECULAR (role + ΔΔG structural + domain + DepMap selectivity):")
    for c in r["A_molecular"]:
        rk = c.get("selective_dep_rank")
        sd = f"  sel-dep#{rk}" if rk is not None else ""
        dd = f"  ΔΔG={c['ddg']}({c.get('ddg_note','')})" if c.get("ddg") is not None else ""
        print(f"     {c['gene']:8} {c.get('effect','-'):22}{dd}{sd}")
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
    cg = r.get("G_cellgraph_perturbation")
    if cg and "error" not in cg:
        print(f"G. CELLGRAPH perturbation (independent lens): {cg['n_strongly_affected']} strongly affected; "
              f"top-down {[x[0] for x in cg['top_down'][:5]]}")
    t = r.get("H_tissue_baseline")
    if t:
        print(f"H. TISSUE baseline (emask): {t['n_types']} matched types e.g. {t['tissue_types_matched'][:3]}")
    print(f"CONCLUSION -> driver {r['conclusion']['driver']}")
    print()


if __name__ == "__main__":
    run()
