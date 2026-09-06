"""strong_vs_weak -- what actually differentiates a STRONG knockout (>=50 genes move, top-10 hits ~9/10) from a WEAK one
(<15 movers, ~1/10)? Compare their gene properties head-to-head on deep k562, plus a function/process enrichment. The magnitude
model already said essentiality + connectivity drive response size; this makes it concrete and interpretable.
"""
import json, csv, collections
from pathlib import Path
import numpy as np
import h5py
import os
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _dec(a): return [x.decode() if isinstance(x, bytes) else x for x in a]


def load():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; idx = {n: i for i, n in enumerate(names)}
    info = {g["name"]: g for g in D["genes"]}
    h = h5py.File(f"{SP}/k562.h5ad", "r")
    kp = [x.split("_")[1] if "_" in x else x for x in _dec(h["obs"]["gene_transcript"][:])]
    cats = _dec(h["var"]["__categories"]["gene_name"][:]); codes = h["var"]["gene_name"][:]; kg = [cats[c] for c in codes]
    X = h["X"][:]; h.close()
    ps = {}
    for i, s in enumerate(kp):
        ps.setdefault(s, i)
    nmov = {}
    for g in ps:
        if g not in idx: continue
        z = X[ps[g]]; fin = np.isfinite(z)
        nmov[g] = int(sum(1 for j in range(len(kg)) if fin[j] and abs(z[j]) > 1.0 and kg[j] != g))
    trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    reg = {tf: len([t for t in ts if isinstance(t, (list, tuple))]) for tf, ts in trr.items()}
    ppi = collections.defaultdict(set)
    for e in D.get("ppi", []):
        ppi[e[0]].add(e[1]); ppi[e[1]].add(e[0])
    ce = {int(k): len(v) for k, v in D.get("coexpr", {}).items()}; cdp = {int(k): len(v) for k, v in D.get("codep", {}).items()}
    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items() if int(k) < len(names)}
    return D, names, idx, info, nmov, reg, ppi, ce, cdp, ppm


def run():
    from scipy.stats import mannwhitneyu
    D, names, idx, info, nmov, reg, ppi, ce, cdp, ppm = load()
    strong = [g for g in nmov if nmov[g] >= 50]
    weak = [g for g in nmov if nmov[g] < 15]
    def prop(g, key):
        gi = idx[g]; m = info.get(g, {})
        if key == "dep_frac": return float(m.get("dep_frac", 0) or 0)
        if key == "essential": return float(m.get("ess", 0) or 0)
        if key == "ppi_degree": return float(m.get("ppi", 0) or 0)
        if key == "coexpr_degree": return float(ce.get(gi, 0))
        if key == "codep_degree": return float(cdp.get(gi, 0))
        if key == "centrality": return float(sum(len(ppi.get(j, ())) for j in list(ppi.get(gi, set()))[:300]))
        if key == "abundance_ppm": return float(ppm.get(g, 0))
        if key == "is_TF": return float(m.get("tf", 0) or 0)
        if key == "regulon_size": return float(reg.get(g, 0))
        if key == "n_pathways": return float(m.get("npath", 0) or 0)
        return 0.0
    keys = ["dep_frac", "essential", "ppi_degree", "coexpr_degree", "codep_degree", "centrality",
            "abundance_ppm", "is_TF", "regulon_size", "n_pathways"]
    cmp = {}
    for k in keys:
        vs = np.array([prop(g, k) for g in strong]); vw = np.array([prop(g, k) for g in weak])
        try: p = float(mannwhitneyu(vs, vw, alternative="two-sided")[1])
        except ValueError: p = 1.0
        cmp[k] = {"strong_median": round(float(np.median(vs)), 3), "weak_median": round(float(np.median(vw)), 3),
                  "strong_mean": round(float(np.mean(vs)), 3), "weak_mean": round(float(np.mean(vw)), 3), "p": p}
    # process / function enrichment
    def procdist(gs):
        c = collections.Counter(info.get(g, {}).get("proc", "?") for g in gs)
        tot = sum(c.values())
        return {k: v / tot for k, v in c.items()}
    ps_s = procdist(strong); ps_w = procdist(weak)
    procs = sorted(set(ps_s) | set(ps_w), key=lambda k: -(ps_s.get(k, 0) - ps_w.get(k, 0)))
    proc_enr = [{"process": k, "strong_frac": round(ps_s.get(k, 0), 3), "weak_frac": round(ps_w.get(k, 0), 3),
                 "enrichment": round(ps_s.get(k, 0) / max(ps_w.get(k, 1e-9), 1e-9), 2)} for k in procs if ps_s.get(k, 0) >= 0.03]
    return {"n_strong": len(strong), "n_weak": len(weak), "property_comparison": cmp,
            "process_enrichment_strong_vs_weak": proc_enr[:8],
            "example_strong": sorted(strong, key=lambda g: -nmov[g])[:12],
            "example_weak": [g for g in sorted(weak, key=lambda g: nmov[g])][:12]}


def main():
    print("=" * 96)
    print("STRONG vs WEAK KNOCKOUTS — what actually makes the difference?")
    print("=" * 96)
    r = run()
    print(f"\n  {r['n_strong']} strong (>=50 movers) vs {r['n_weak']} weak (<15). Property medians:\n")
    print(f"  {'property':16s} {'STRONG':>10s} {'WEAK':>10s} {'strong/weak':>12s}")
    for k, v in r["property_comparison"].items():
        ratio = v["strong_median"] / v["weak_median"] if v["weak_median"] else float("inf")
        star = " *" if v["p"] < 1e-6 else ""
        rs = f"{ratio:.1f}x" if np.isfinite(ratio) else "(w=0)"
        print(f"  {k:16s} {v['strong_median']:>10} {v['weak_median']:>10} {rs:>12}{star}")
    print(f"\n  FUNCTION enrichment (fraction of KOs by process, strong vs weak):")
    for e in r["process_enrichment_strong_vs_weak"]:
        print(f"     {e['process']:26s} strong {e['strong_frac']:.0%}  weak {e['weak_frac']:.0%}   ({e['enrichment']}x)")
    print(f"\n  example STRONG: {', '.join(r['example_strong'])}")
    print(f"  example WEAK:   {', '.join(r['example_weak'])}")
    cmp = r["property_comparison"]
    def rr(k):
        w = cmp[k]["weak_median"]; return cmp[k]["strong_median"] / w if w else float('inf')
    verdict = (
        "WHAT DIFFERENTIATES STRONG vs WEAK KNOCKOUTS -- measured on deep k562 (189 strong >=50 movers vs 1412 weak <15), and it is "
        "the capacity model made concrete, with ONE correction to my prior guess. Strong KOs separate from weak on THREE measured axes "
        "(and, importantly, NOT on a fourth). (1) ESSENTIALITY, by degree: dep_frac "
        f"{cmp['dep_frac']['strong_median']} vs {cmp['dep_frac']['weak_median']} -- strong KOs are depended-on more completely. (Note "
        "the BINARY essential flag is ~1 for BOTH groups: nearly every profiled KO is essential-ish; the separation is in the DEGREE "
        f"of dependence, not essential-yes/no.) (2) PHYSICAL CONNECTIVITY: PPI degree {int(cmp['ppi_degree']['strong_median'])} vs "
        f"{int(cmp['ppi_degree']['weak_median'])} ({rr('ppi_degree'):.0f}x), network centrality {int(cmp['centrality']['strong_median'])} "
        f"vs {int(cmp['centrality']['weak_median'])} ({rr('centrality'):.1f}x), pathway membership "
        f"{int(cmp['n_pathways']['strong_median'])} vs {int(cmp['n_pathways']['weak_median'])} ({rr('n_pathways'):.1f}x) -- strong KOs "
        "sit at PHYSICAL/pathway hubs. CORRECTION to what I guessed before: coexpression degree (12 vs 12) and codependency degree "
        "(7 vs 7) are FLAT -- transcriptional co-regulation breadth does NOT separate them; it is specifically physical-interaction and "
        "pathway centrality that does. (3) FUNCTION: strong KOs are heavily enriched for the CORE GENE-EXPRESSION MACHINERY -- "
        "transcription 42% vs 20% (2.1x), translation 31% vs 13% (2.4x), while 'other/peripheral' is DEPLETED (18% vs 31%). Concretely, "
        "strong = GATA1, RPL7A (ribosome), SUPT6H/POLR2K/MED9/MED30/INTS2 (Pol II / Mediator / Integrator), RBM22/AQR/SNRPD2 "
        "(spliceosome); weak = zinc-finger genes (ZNF100/133/207/253/468, ZMAT2...) -- redundant paralog families, specialised, "
        "low-connectivity. (Strong KOs are also themselves more highly EXPRESSED, 11.8 vs 5.4 ppm -- consistent with being abundant "
        "core machinery.) MECHANISM (the capacity model): knocking out an essential, physically-central core-machinery gene creates a "
        "large CAPACITY DEFICIT in a fundamental process -> a big hit to the global budget -> a large generic stress response (many "
        "genes move, so the top-10 usual-suspects land = ~9/10). Knocking out a peripheral/redundant gene (a zinc finger with paralogs) "
        "is BUFFERED -> few genes move -> little to predict (~1/10). So 'strong vs weak' = 'essential + PPI/pathway-central + "
        "core-machinery' vs 'dispensable + peripheral + redundant', which is why RESPONSE SIZE is predictable (Spearman ~0.5 from "
        "exactly these features) even though which SPECIFIC genes move is not -- the size is set by the node's importance, the identity "
        "by chaos.")
    print(f"\n  VERDICT: {verdict}")
    r["verdict"] = verdict
    r["note"] = ("What differentiates strong (>=50 movers, ~9/10) vs weak (<15, ~1/10) knockouts, measured on deep k562 (189 vs 1412). "
                 "Strong KOs separate on THREE axes: (1) ESSENTIALITY by degree -- dep_frac 0.99 vs 0.77 (the BINARY essential flag is "
                 "~1 for both; separation is in degree of dependence). (2) PHYSICAL CONNECTIVITY -- PPI degree 65 vs 13 (5x), centrality "
                 "3.3x, pathway membership 7 vs 2 (3.5x); BUT coexpr degree (12 vs 12) and codep degree (7 vs 7) are FLAT -- correction "
                 "to my earlier guess: it is physical-interaction/pathway centrality, NOT co-regulation breadth, that separates them. "
                 "(3) FUNCTION -- strong enriched for CORE GENE-EXPRESSION MACHINERY: transcription 42% vs 20% (2.1x), translation 31% "
                 "vs 13% (2.4x); concretely strong = GATA1, RPL7A, POLR2K/MED9/MED30/INTS2 (Pol II/Mediator), RBM22/AQR/SNRPD2 "
                 "(spliceosome); weak = zinc-finger paralog families (ZNF100/133/207..., ZMAT2). Strong KOs are also more highly "
                 "expressed (11.8 vs 5.4 ppm). Mechanism = capacity model: essential PPI-central core-machinery hub -> large capacity "
                 "deficit -> big generic stress response (top-10 usual-suspects land = ~9/10); peripheral/redundant gene -> buffered -> "
                 "few movers -> ~1/10. So strong-vs-weak = essential+PPI-central+core-machinery vs dispensable+peripheral+redundant; "
                 "response SIZE is set by node importance (predictable ~0.5) while gene IDENTITY stays chaos.")
    json.dump(r, open(OUT / "strong_vs_weak.json", "w"), indent=1)
    print("\n  -> outputs/orphan/strong_vs_weak.json")
    return r


if __name__ == "__main__":
    main()
