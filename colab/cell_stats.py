"""cell_stats — the WHOLE-CELL summary: every layer, all relations, kinetics/flux, coverage, and the ML
outcomes — not just PPI. Reads the CompleteCell plus whatever result files are present (per-relation combiners,
the self-healing loop report, corrected kinetics) and prints one honest dashboard of the entire model.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"


def _load(name):
    p = os.path.join(OUT, name)
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except Exception:
        return None


def _n(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def report():
    C = CompleteCell()
    D = C.D
    cov = C.layers()["coverage"]
    n = len(C.genes)
    kin = C.kin
    meas = sum(1 for v in kin.values() if v.get("tier") in ("measured", "EC-measured"))
    corrected = sum(1 for v in kin.values() if v.get("kcat_original") is not None)
    dark = D.get("dark_count", 0)
    pw_cov = set(m for mem in D.get("pathways", {}).values() for m in mem if isinstance(m, int))
    emask = D.get("emask", {})
    drugs = D.get("drugs", {})
    dis = sum(1 for g in C.genes if g.get("ndis"))

    def relstat(rel):
        v = _load(f"signal_combiner_{rel}_validation.json") if rel != "ppi" else _load("signal_combiner_validation.json")
        edges = len(D.get(rel, []))
        if v:
            return (f"AUC {v.get('combined_auc')}  kept {v.get('kept_features')}", edges)
        return ("(combiner not trained)", edges)

    L = []
    L.append("=" * 82)
    L.append("WHOLE-CELL MODEL — every layer")
    L.append("=" * 82)
    L.append(f"  genome / proteins        {_n(n)} genes  ({round(100*n/19900)}% of protein-coding genome)")
    L.append(f"  GO-annotated             {_n(len(D.get('go', {})))}")
    L.append(f"  dark proteome            {_n(dark)}  ({_n(len(C.ghost))} literature-patched)")
    L.append("")
    L.append("  NETWORKS (relations) — edge count | whole-cell combiner:")
    for rel, lab in (("ppi", "protein-protein"), ("reg", "regulatory TF->target"), ("sig", "signaling")):
        s, e = relstat(rel)
        L.append(f"    {rel:4} {lab:22} {_n(e):>9} edges | {s}")
    cx_extra = getattr(C, "n_complexes_extra", 0)
    cx_note = f" (+{_n(cx_extra)} hu.MAP/CORUM)" if cx_extra else ""
    L.append(f"    complexes {_n(len(D.get('complexes', {})))}{cx_note} | synthetic-lethal {_n(len(D.get('sl', [])))}"
             f" | ligand-receptor {_n(len(D.get('lr', [])))}")
    cm = D.get("causal_meta")
    if cm:
        L.append(f"    causal direction (SIGNOR+CollecTRI) {_n(cm.get('total', 0))} directed edges, "
                 f"{_n(cm.get('signed', 0))} signed (activate/inhibit)")
    chm = D.get("chip_meta")
    if chm:
        L.append(f"    ChIP-derived TF->target {_n(chm.get('n_edges', 0))} edges ({_n(chm.get('n_tf', 0))} TFs) "
                 f"- experimental-binding channel")
    pem = D.get("ppi_extra_meta")
    if pem:
        L.append(f"    extra PPI channels     {pem}  (orthogonal evidence + negatives)")
    L.append("")
    L.append("  KINETICS / METABOLISM:")
    L.append(f"    enzymes with kinetics  {_n(len(kin))}  (measured {_n(meas)}, predicted {_n(len(kin)-meas)},"
             f" corrected {_n(corrected)})")
    L.append(f"    metabolic reactions    12,931 (Human-GEM) | reactions with genes {_n(len(D.get('generxn', {})))}")
    mm = D.get("metabolite_meta", {})
    if mm:
        L.append(f"    metabolites (HMDB)     {_n(mm.get('n', 0))} ({_n(mm.get('n_with_conc', 0))} with a measured "
                 f"concentration)")
    L.append("")
    L.append("  FUNCTION / PATHWAYS:")
    rx = _load("reactome_pathways.json")
    if rx and rx.get("pathways"):
        rxcov = set(g for v in rx["pathways"].values() for g in v)
        allcov = pw_cov | rxcov
        L.append(f"    curated pathways       {_n(len(D.get('pathways', {})))} base + {_n(len(rx['pathways']))} "
                 f"Reactome -> covering {_n(len(allcov))}/{_n(n)} genes ({round(100*len(allcov)/n)}%, was "
                 f"{round(100*len(pw_cov)/n)}%)")
    else:
        L.append(f"    curated pathways       {_n(len(D.get('pathways', {})))} covering {_n(len(pw_cov))}/{_n(n)}"
                 f" genes ({round(100*len(pw_cov)/n)}%)")
    # honest coverage: NAMED-pathway membership (Reactome/curated) is distinct from any FUNCTIONAL annotation.
    # We already hold GO for ~all genes, so functional coverage is far higher than the named-pathway number.
    go = D.get("go", {})
    func = set(C.idx[g] for g in go if g in C.idx) | pw_cov
    if rx and rx.get("pathways"):
        func |= set(g for v in rx["pathways"].values() for g in v)
    L.append(f"    functional annotation  {_n(len(func))}/{_n(n)} genes ({round(100*len(func)/n)}%) have a GO "
             f"process/function term or a named pathway  ({n-len(func)} have neither)")
    tfm = D.get("motif_meta", {})
    if tfm:
        L.append(f"    TF binding motifs      {_n(tfm.get('n_tf', 0))} TFs with a JASPAR PWM (mechanism under reg edges)")
    dom = getattr(C, "domains", {}) or {}
    disr = getattr(C, "disorder", {}) or {}
    if dom or disr:
        L.append(f"    domains / disorder     {_n(len(dom))} genes w/ InterPro domains | {_n(len(disr))} w/ MobiDB disorder")
    plddt = getattr(C, "plddt", {}) or {}
    enh = getattr(C, "enhancers", {}) or {}
    if plddt or enh:
        L.append(f"    structure / enhancers  {_n(len(plddt))} genes w/ AlphaFold pLDDT | {_n(len(enh))} w/ enhancer links")
    tdl = getattr(C, "tdl", {}) or {}
    if tdl:
        import collections as _c
        L.append(f"    target dev. (Pharos)   {_n(len(tdl))} scored  {dict(_c.Counter(tdl.values()))}")
    L.append(f"    PTMs {_n(len(D.get('ptm', {})))} | dark-function predictions {_n(len(D.get('darkfn', {})))}")
    L.append("")
    L.append("  EXPRESSION / CONTEXT:")
    L.append(f"    single-cell expression {_n(len(emask))}/{_n(n)} genes ({round(100*len(emask)/n)}%)"
             f" across {_n(len(D.get('ctnames', [])))} cell types")
    L.append(f"    abundance (ppm)        {_n(len(D.get('ppm', {})))} | co-expression {_n(len(D.get('coexpr', {})))}"
             f" | co-dependency {_n(len(D.get('codep', {})))}")
    cop = getattr(C, "copies", {}) or {}
    tr = getattr(C, "translation", {}) or {}
    if cop or tr:
        L.append(f"    absolute copies/cell   {_n(len(cop))} genes | translation/half-life {_n(len(tr))} genes"
                 f"  (enzyme-constrained flux inputs)")
    vmax = getattr(C, "vmax_ref", {}) or {}
    icr = getattr(C, "incell_rate", {}) or {}
    if vmax or icr:
        sm = D.get("saturation_kapp_meta", {})
        L.append(f"    in-cell rates (from kcat)  {_n(len(vmax))} cell-type capacities (Vmax=kcat*ppm) | "
                 f"{_n(len(icr))} saturation operating rates (median sigma {sm.get('median_sigma')}, "
                 f"Davidi 0.5)")
    L.append("")
    L.append("  DISEASE / DRUGS:")
    L.append(f"    disease-linked genes   {_n(dis)} | drug targets {_n(len(drugs))} | structures {_n(len(D.get('struct', {})))}")

    # ---- ML self-healing outcomes (whole-cell audit loop) ----
    loop = _load("phase2_loop_report.json")
    if loop:
        L.append("")
        L.append("  ML SELF-HEALING LOOP (audit every field -> verified fixes):")
        L.append(f"    FIXED (locked)   {_n(loop.get('total_locked_fixes', 0))}  {loop.get('locked_by_kind', {})}")
        L.append(f"    NEEDS-DATA       {_n(loop.get('total_needs_data', 0))}  {loop.get('needs_data_by_kind', {})}")
        L.append(f"    ACCEPTED (ok)    {_n(loop.get('total_accepted', 0))}  {loop.get('accepted_by_kind', {})}")
        L.append(f"    REGRESSIONS      {loop.get('total_regressions', 0)}   | converged: {loop.get('converged')}")
        L.append(f"    PPI edges {_n(loop.get('base_ppi_pairs', 0))} -> {_n(loop.get('final_ppi_pairs', 0))}")

    # ---- scorecard ----
    sc = _load("recovery_scorecard.json")
    if sc:
        L.append("")
        L.append(f"  CAPABILITY SCORECARD: {sc.get('n_pass','?')}/{sc.get('n_total','?')} axes pass")
    L.append("=" * 82)
    out = "\n".join(L)
    print(out)
    return out


if __name__ == "__main__":
    report()
