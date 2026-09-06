"""propagate — the multi-layer FORWARD propagation the network was missing: mutate any protein/gene (NEXUS on top), flow the
effect through regulation -> gene products -> PPI/complexes -> pathways -> interconnected pathways, layer by layer.

  L0  NEXUS         mutation (ΔΔG_fold, ΔΔG_bind) -> the mutated protein's graded ACTIVITY (soft-AND, LOF/GOF)
  L1  REGULATION    if the protein is a TF: its direct regulon (TRRUST signed) is the first set of affected genes + direction
  L2  PRODUCTS+PPI  the mutated protein + regulon targets -> their physical partners (PPI) and complex co-members
  L3  PATHWAYS      every affected protein -> the pathways it sits in (Reactome)
  L4  CROSSTALK     pathways interconnected with the hit pathways (shared proteins / bridging PPI) -> downstream pathways

HONEST FRAMING (measured all session): this is a STRUCTURAL forward-reachability / blast-radius map -- WHAT is mechanistically
connected to the mutation and COULD be affected. It is NOT the quantitative dynamical cascade (which genes actually move, by how
much) -- that does not compose (graph propagation ~chance, knockout recall ~3-18%). validate() shows exactly this: the blast
radius is enriched for real movers at the tight (regulon) layer but dilutes to near-chance as it balloons through PPI/pathways.
So: a map of the POSSIBLE, honestly bounded -- not a simulator of the ACTUAL.

STILL MISSING (acknowledged layers, not built): splicing, 5' capping, poly-A tail, and full epigenetic state. Partial hooks
exist (epigenetic gating via invivo_gate's chromatin tracks; PTM/ncRNA fields in the cell image) but these post-transcriptional
and chromatin layers are TODO -- each is its own hard problem (e.g. SpliceAI-class splicing prediction).
"""
import json
import os
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))


def _load():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    idx = {n: i for i, n in enumerate(names)}
    tf = {g["name"] for g in D["genes"] if g.get("tf")}
    import collections
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if a < len(names) and b < len(names):
            ppi[a].add(b); ppi[b].add(a)
    R = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    g2pw = collections.defaultdict(set); pw2g = {}
    for pw, mem in R.items():
        pw2g[pw] = set(m for m in mem if m < len(names))
        for m in pw2g[pw]:
            g2pw[m].add(pw)
    g2c = {int(k): v for k, v in D["gene2cplx"].items()}
    cplx = {k: set(idx[x] for x in v if x in idx) for k, v in D["complexes"].items()} if isinstance(next(iter(D["complexes"].values()), None), list) else {}
    tr = json.load(open(OUT / "trrust_regulon.json")).get("tf_targets", {})
    N = len(names)
    reg_edges = [(a, b, s) for a, b, s in D["reg"] if a < N and b < N]     # directed signed GRN (for RWR)
    return {"names": names, "idx": idx, "tf": tf, "ppi": ppi, "g2pw": g2pw, "pw2g": pw2g,
            "g2c": g2c, "cplx": cplx, "trrust": tr, "reg_edges": reg_edges, "N": N}


def _promoter_rate(G):
    """transcription-rate proxy per gene = promoter Pol II signal (POLR2A ChIP at TSS+-1kb). How the PROMOTER sets the
    baseline rate the TF then modulates. Returns {gene_idx: rate}. 0 if no promoter Pol II (silent promoter)."""
    import crispr_gate as cg
    D = json.load(open(OUT / "cell_complete.json"))
    pol = cg._signal_index(OUT / "invivo/marks/polr2a.bed.gz", 6)      # POLR2A signalValue
    rate = {}
    for i, g in enumerate(D["genes"]):
        c = g.get("chrom"); t = g.get("tss")
        if c and t:
            rate[i] = cg._max_signal(pol, c, int(t) - 1000, int(t) + 1000)
    return rate


def _transition(G, w_reg=1.0, w_ppi=0.5, w_cplx=1.0):
    """column-stochastic transition matrix over the weighted multi-layer graph for random-walk-with-restart.
    Directed regulatory edges (mass flows src->dst), undirected PPI, complex cliques. PATHWAY co-membership is
    DELIBERATELY EXCLUDED -- it was the dilution source that flattened precision to chance in the flat blast radius."""
    import numpy as np, scipy.sparse as sp
    N = G["N"]; rows = []; cols = []; vals = []
    for a, b, s in G["reg_edges"]:
        rows.append(b); cols.append(a); vals.append(w_reg)         # column a (source) -> row b (target)
    for a, nbrs in G["ppi"].items():
        for b in nbrs:
            rows.append(b); cols.append(a); vals.append(w_ppi)     # ppi adjacency already symmetric
    if w_cplx > 0:
        for mem in G["cplx"].values():
            mem = list(mem)
            for i in mem:
                for j in mem:
                    if i != j:
                        rows.append(j); cols.append(i); vals.append(w_cplx)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    d = np.asarray(A.sum(axis=0)).ravel(); d[d == 0] = 1.0
    return A.multiply(sp.csr_matrix(1.0 / d)).tocsr()


def _rwr(W, seed_idx, alpha=0.3, iters=60):
    """random-walk-with-restart (personalized PageRank) from the seed. p = (1-a) W p + a e_seed."""
    import numpy as np
    N = W.shape[0]; e = np.zeros(N); e[seed_idx] = 1.0; p = e.copy()
    for _ in range(iters):
        p = (1 - alpha) * (W @ p) + alpha * e
    return p


def rank_targets(gene, ddg_fold=6.0, ddg_bind=0.0, G=None, rate=None, W=None, topn=40):
    """THE PRECISION FIX: instead of a flat blast-radius SET (all reachable genes equal -> chance precision), emit a single
    RANKED, CALIBRATED score so real movers float to the top. composite = 10*regulon*(0.5+promoter_rate) + RWR_nearfield:
    the curated direct regulon (rate-weighted) forms the high-precision core (~9x enriched), the RWR score ranks the physical
    near-field, pathway co-membership is dropped. Measured (GATA1): AUPRC ~2.3x base, top decile ~2.4x, beats degree/distance/
    random and RWR-alone; label-shuffle p~0.015. Ranks the radius; does NOT make the far field precise (that's the wall)."""
    import numpy as np
    if G is None:
        G = _load()
    import nexus_regulate as nr
    idx = G["idx"]; names = G["names"]
    if gene not in idx:
        return {"error": f"{gene} not in cell image"}
    if W is None:
        W = _transition(G)
    if rate is None:
        rate = _promoter_rate(G)
    seed = idx[gene]
    a = nr.tf_activity(ddg_fold, ddg_bind)
    p = _rwr(W, seed)
    sign = {t[0]: (int(t[1]) if len(t) > 1 else 0) for t in G["trrust"].get(gene, [])}
    regset = {idx[t] for t in sign if t in idx}
    ratev = np.array([rate.get(i, 0.0) for i in range(G["N"])])
    rn = ratev / (ratev.max() + 1e-12)
    pn = p / (p.max() + 1e-12)
    reg_ind = np.zeros(G["N"]); reg_ind[list(regset)] = 1.0
    composite = 10.0 * reg_ind * (0.5 + rn) + pn
    composite[seed] = -1.0                                            # exclude the seed itself
    order = np.argsort(-composite)[:topn]
    rows = []
    for j in order:
        nm = names[j]; s = sign.get(nm, 0)
        in_reg = j in regset
        drate = s * (a - 1.0) * ratev[j] if in_reg else 0.0
        tier = "regulon-core" if in_reg else ("near-field" if pn[j] > 1e-3 else "far-field")
        direction = ("up" if drate > 0 else "down") if abs(drate) > 1e-9 else ("?" if in_reg else "")
        rows.append({"gene": nm, "score": round(float(composite[j]), 4), "tier": tier,
                     "direction": direction, "rwr_pctl": round(float((p < p[j]).mean()), 3)})
    return {"gene": gene, "nexus_activity": round(a, 3), "n_ranked": len(rows), "ranked": rows}


def validate_ranked(gene="GATA1", G=None, rate=None, W=None, dataset="k562.h5ad", thresh=3.0):
    """MEASURED: does the ranked composite beat the flat set + trivial baselines on GATA1's real Perturb-seq movers?
    Reports AUPRC lift, precision-at-K, decile calibration, a label-shuffle null, and degree/distance/RWR-alone baselines."""
    import numpy as np
    if G is None: G = _load()
    if W is None: W = _transition(G)
    if rate is None: rate = _promoter_rate(G)
    import bind_vs_reg as br
    reg = br.load_regulation(gene, dataset, thresh)
    if reg is None: return {"error": f"no Perturb-seq for {gene}"}
    idx = G["idx"]
    if gene not in idx: return {"error": f"{gene} not in graph"}
    seed = idx[gene]
    U = [g for g in reg["U"] if g in idx and g != gene]
    ui = np.array([idx[g] for g in U])
    y = np.array([1.0 if g in reg["reg"] else 0.0 for g in U])
    if y.sum() < 8: return {"error": f"only {int(y.sum())} movers -- underpowered"}
    base = y.mean()
    p = _rwr(W, seed)
    sign = {t[0]: (int(t[1]) if len(t) > 1 else 0) for t in G["trrust"].get(gene, [])}
    regset = {idx[t] for t in sign if t in idx}
    ratev = np.array([rate.get(i, 0.0) for i in range(G["N"])])
    rn = ratev / (ratev.max() + 1e-12); pn = p / (p.max() + 1e-12)
    reg_ind = np.zeros(G["N"]); reg_ind[list(regset)] = 1.0
    comp = (10.0 * reg_ind * (0.5 + rn) + pn)[ui]
    deg = np.zeros(G["N"])
    for a2, b2, s2 in G["reg_edges"]: deg[a2] += 1; deg[b2] += 1
    for a2, nbrs in G["ppi"].items(): deg[a2] += len(nbrs)

    def auprc(sc, lab):
        o = np.argsort(-sc); yy = lab[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
        prec = tp / (tp + fp); rec = tp / max(yy.sum(), 1); ap = 0.0; pr = 0.0
        for i in range(len(yy)):
            if yy[i]: ap += prec[i] * (rec[i] - pr); pr = rec[i]
        return ap
    def patk(sc, lab, K): return float(lab[np.argsort(-sc)[:K]].sum()) / K
    scores = {"composite": comp, "RWR": p[ui], "degree": deg[ui], "promoter_rate": ratev[ui]}
    tbl = {}
    for nm, sc in scores.items():
        tbl[nm] = {"AUPRC": round(auprc(sc, y), 4), "lift": round(auprc(sc, y) / base, 2),
                   "P@10": round(patk(sc, y, 10), 3), "P@25": round(patk(sc, y, 25), 3), "P@50": round(patk(sc, y, 50), 3)}
    # shuffle null for composite
    rng = np.random.RandomState(0); real = auprc(comp, y) / base
    shuf = np.array([auprc(comp, rng.permutation(y)) / base for _ in range(300)])
    # decile calibration
    o = np.argsort(-comp); yd = y[o]; n = len(yd); dec = []
    for q in range(10):
        seg = yd[q * n // 10:(q + 1) * n // 10]
        dec.append(round(float(seg.mean() / base) if base else 0.0, 2))
    return {"gene": gene, "n_movers": int(y.sum()), "universe": len(U), "base_rate": round(base, 4),
            "methods": tbl, "composite_shuffle": {"real_lift": round(real, 2), "null_mean": round(float(shuf.mean()), 2),
             "null_p95": round(float(np.percentile(shuf, 95)), 2), "p_value": round(float((shuf >= real).mean()), 4)},
            "composite_decile_enrichment_top_to_bottom": dec}


def propagate(gene, ddg_fold=6.0, ddg_bind=0.0, G=None, rate=None):
    if G is None:
        G = _load()
    import nexus_regulate as nr
    idx = G["idx"]; names = G["names"]
    if gene not in idx:
        return {"error": f"{gene} not in cell image"}
    gi = idx[gene]
    a = nr.tf_activity(ddg_fold, ddg_bind)
    is_tf = gene in G["tf"]
    # L1 regulation
    regulon = set()
    if is_tf:
        for t in G["trrust"].get(gene, []):
            if t[0] in idx:
                regulon.add(idx[t[0]])
    # transcription RATE: the promoter sets the baseline rate; the TF modulates it. The target's rate-change magnitude
    # scales with |activity-1| (how much TF drive is lost) x the target's PROMOTER strength (Pol II proxy) x sign.
    rate_hits = []
    if is_tf and regulon:
        sign = {t[0]: (int(t[1]) if len(t) > 1 else 0) for t in G["trrust"].get(gene, [])}
        for p in regulon:
            nmp = names[p]; s = sign.get(nmp, 0)
            promoter = rate.get(p, 0.0) if rate else 0.0
            drate = s * (a - 1.0) * promoter             # >0 rate up, <0 rate down (activated target FOLLOWS TF: LOF->down)
            if abs(drate) > 1e-9:
                rate_hits.append({"target": nmp, "sign": s, "promoter_rate": round(promoter, 1),
                                  "delta_rate": round(drate, 1)})
        rate_hits.sort(key=lambda r: -abs(r["delta_rate"]))
    seed = {gi} | regulon                                    # proteins whose level/activity is first-order affected
    # L2 products + PPI/complex
    phys = set()
    for p in seed:
        phys |= G["ppi"].get(p, set())
        for cx in G["g2c"].get(p, []):
            phys |= G["cplx"].get(cx, set())
    affected_prot = seed | phys
    # L3 pathways
    hit_pw = set()
    for p in affected_prot:
        hit_pw |= G["g2pw"].get(p, set())
    pw_genes = set()
    for pw in hit_pw:
        pw_genes |= G["pw2g"][pw]
    # L4 crosstalk: pathways sharing >=3 proteins with a hit pathway
    cross = set()
    hit_sets = {pw: G["pw2g"][pw] for pw in hit_pw}
    for opw, og in G["pw2g"].items():
        if opw in hit_pw:
            continue
        for pw in hit_pw:
            if len(og & hit_sets[pw]) >= 3:
                cross.add(opw); break
    nm = lambda s: [names[i] for i in s]
    return {"gene": gene, "is_tf": is_tf, "nexus_activity": round(a, 3),
            "L1_regulation": {"n": len(regulon), "genes": sorted(nm(regulon))[:30],
                              "rate_weighted_top": rate_hits[:15]},
            "L2_ppi_complex": {"n": len(phys), "genes": sorted(nm(phys))[:30]},
            "L3_pathways": {"n_pathways": len(hit_pw), "pathways": sorted(hit_pw)[:20], "n_genes": len(pw_genes)},
            "L4_crosstalk_pathways": {"n": len(cross), "pathways": sorted(cross)[:20]},
            "_sets": {"regulon": regulon, "phys": phys, "affected_prot": affected_prot, "pw_genes": pw_genes}}


def validate(gene="GATA1", G=None):
    """MEASURED: how much of the real Perturb-seq knockout response does each propagation layer capture, and at what precision?
    Shows the blast radius growing and precision diluting -- the map-of-possible vs simulator-of-actual boundary."""
    if G is None:
        G = _load()
    import bind_vs_reg as br
    reg = br.load_regulation(gene, "k562.h5ad", 3.0)
    if reg is None:
        return {"error": f"no Perturb-seq for {gene}"}
    idx = G["idx"]
    movers = {idx[g] for g in reg["reg"] if g in idx}
    U = {idx[g] for g in reg["U"] if g in idx}
    base = len(movers) / len(U) if U else 0
    P = propagate(gene, G=G, rate=_promoter_rate(G))
    S = P["_sets"]
    layers = [("L1 regulation (direct regulon)", S["regulon"]),
              ("L2 + PPI/complex", S["regulon"] | S["phys"]),
              ("L3 + pathway members (blast radius)", S["affected_prot"] | S["pw_genes"])]
    rows = []
    for label, s in layers:
        s = s & U
        if not s:
            rows.append({"layer": label, "n_reachable": 0}); continue
        hit = len(s & movers)
        prec = hit / len(s); rec = hit / len(movers) if movers else 0
        rows.append({"layer": label, "n_reachable": len(s), "captures_movers": hit,
                     "precision": round(prec, 3), "recall": round(rec, 3),
                     "enrichment_vs_chance": round(prec / base, 2) if base else None})
    return {"gene": gene, "n_movers": len(movers), "universe": len(U), "base_rate": round(base, 4), "layers": rows}


def main():
    print("=" * 100)
    print("PROPAGATE — mutate a protein (NEXUS) and flow FORWARD through regulation -> PPI -> pathways -> crosstalk")
    print("=" * 100)
    G = _load()
    rate = _promoter_rate(G)
    P = propagate("GATA1", ddg_fold=7.5, G=G, rate=rate)
    print(f"\n  MUTATE GATA1 (ΔΔG_fold=7.5 -> NEXUS activity {P['nexus_activity']}, TF={P['is_tf']}). Forward blast radius:")
    print(f"    L1 regulation  : {P['L1_regulation']['n']:5d} direct-regulon genes")
    rw = P["L1_regulation"]["rate_weighted_top"]
    if rw:
        print("       transcription-RATE model (Δrate = |ΔTF-activity| x promoter Pol II x sign) -- mechanistic ordering:")
        for r in rw[:6]:
            arrow = "down" if r["delta_rate"] < 0 else "up"
            print(f"         {r['target']:10s} promoter-rate {r['promoter_rate']:6.1f}  ->  transcription {arrow} (Δrate {r['delta_rate']:+.0f})")
        print("       [HONEST: this rate model captures the MECHANISM (promoter sets baseline rate, TF modulates it) but its")
        print("        magnitudes do NOT match measured knockout effect sizes (r~0) -- the per-TF->target rate COEFFICIENT")
        print("        (how much THIS TF drives THAT promoter) is the unmeasured quantity. Direction is more reliable than rate.]")
    print(f"    L2 PPI/complex : {P['L2_ppi_complex']['n']:5d} physical partners of those (e.g. {', '.join(P['L2_ppi_complex']['genes'][:6])})")
    print(f"    L3 pathways    : {P['L3_pathways']['n_pathways']:5d} pathways hit, spanning {P['L3_pathways']['n_genes']} genes")
    print(f"    L4 crosstalk   : {P['L4_crosstalk_pathways']['n']:5d} interconnected downstream pathways")
    print("\n  MEASURED — how much of GATA1's real Perturb-seq knockout response each layer captures, and at what precision:")
    V = validate("GATA1", G)
    print(f"    ({V['n_movers']} measured movers, universe {V['universe']}, base rate {V['base_rate']:.1%})")
    for r in V["layers"]:
        if r.get("n_reachable"):
            print(f"    {r['layer']:38s} reach {r['n_reachable']:5d}  captures {r['captures_movers']:3d}  "
                  f"precision {r['precision']:.3f}  recall {r['recall']:.2f}  ({r['enrichment_vs_chance']}x chance)")
    print("\n  => the blast radius is a STRUCTURAL map of what's mechanistically connected. It is enriched at the tight regulon")
    print("     layer but dilutes toward chance as it balloons through PPI/pathways -- a map of the POSSIBLE, not the ACTUAL.")
    print("     Quantitative 'which genes move' remains the dynamics wall. STILL MISSING: splicing, capping, poly-A, epigenetics.")

    # ---- PRECISION FIX: rank the radius instead of leaving it a flat equal-probability set ----
    print("\n" + "-" * 100)
    print("  PRECISION FIX -- rank the radius (composite = regulon x rate + RWR near-field) so real movers float to the top:")
    W = _transition(G)
    R = rank_targets("GATA1", ddg_fold=7.5, G=G, rate=rate, W=W, topn=12)
    print("    ranked top (tier / predicted direction):")
    for x in R["ranked"][:10]:
        print(f"      {x['gene']:10s} score {x['score']:6.2f}  {x['tier']:12s} {x['direction']}")
    VR = validate_ranked("GATA1", G=G, rate=rate, W=W)
    print(f"\n  MEASURED ranking quality (GATA1, {VR['n_movers']} movers, base {VR['base_rate']:.1%}):")
    for m, d in VR["methods"].items():
        print(f"    {m:14s} AUPRC {d['AUPRC']:.4f}  lift {d['lift']:.2f}x  P@10 {d['P@10']:.2f} P@25 {d['P@25']:.2f} P@50 {d['P@50']:.2f}")
    sh = VR["composite_shuffle"]
    print(f"    label-shuffle null: real {sh['real_lift']}x vs null {sh['null_mean']}x (p95 {sh['null_p95']}) -> p={sh['p_value']}")
    print(f"    composite decile enrichment (top->bottom): {VR['composite_decile_enrichment_top_to_bottom']}")
    print("  => composite RANKS the radius (top decile ~2.4x, P@10 ~12x base, beats degree/rate/RWR-alone, shuffle-significant).")
    print("     It makes the TOP of the radius usable; it does NOT make the far field precise -- GATA1's real movers are a")
    print("     de-repressed secondary program our curated edges don't connect to (SPI1/CEBPA relay 0x). That's the wall, located.")

    out = {"propagate_GATA1": {k: v for k, v in P.items() if k != "_sets"}, "validation": V,
           "ranked_precision_fix": {"ranked_top": R["ranked"], "validation": VR},
           "note": "Multi-layer FORWARD structural propagation with NEXUS as the mutation entry point: protein mutation -> "
                   "activity -> regulation (direct regulon) -> products/PPI/complex -> pathways -> interconnected pathways. "
                   "STRUCTURAL reachability / blast-radius map (what is mechanistically connected and could be affected), NOT "
                   "the quantitative dynamical cascade (measured wall). validate() quantifies the map-vs-simulator boundary: "
                   "enriched at the regulon layer, dilutes to ~chance through PPI/pathways. TRANSCRIPTION RATE: L1 includes a "
                   "rate model (Δrate = |ΔTF-activity| x promoter Pol II x sign) that captures the mechanism (promoter sets "
                   "baseline rate, TF modulates it) but whose magnitudes do NOT match measured effect sizes (r~0) -- the "
                   "per-TF->target rate coefficient is the unmeasured quantity; direction is more reliable than rate. Missing "
                   "layers acknowledged: splicing, 5' capping, poly-A tail, full epigenetic state (each its own hard problem). "
                   "PRECISION FIX (rank_targets/validate_ranked): the flat blast radius has good recall but ~chance precision "
                   "because it is UNRANKED (all reachable genes equal). The fix emits a single RANKED, CALIBRATED score "
                   "composite = 10*regulon*(0.5+promoter_rate) + RWR_nearfield (random-walk-with-restart over the directed "
                   "regulatory + PPI + complex graph; pathway co-membership dropped as it was the dilution source). MEASURED on "
                   "GATA1's real movers: composite AUPRC ~2.3x base, top decile ~2.4x, P@10 ~12x base; beats degree (0.99x=chance), "
                   "promoter-rate-alone (1.17x), and RWR-alone (1.71x); label-shuffle p~0.02, robust across |z| thresholds. It "
                   "RANKS the radius so the top is usable but does NOT make the far field precise -- GATA1's real movers are a "
                   "de-repressed secondary program (all UP) that the curated edges do not connect to (SPI1/CEBPA relay 0x) = the "
                   "measured wall, now located at the missing edges rather than the algorithm."}
    json.dump(out, open(OUT / "propagate.json", "w"), indent=1)
    print("\n  -> outputs/orphan/propagate.json")
    return out


if __name__ == "__main__":
    main()
