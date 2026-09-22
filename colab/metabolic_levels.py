"""metabolic_levels — scale tier-1 (metabolic step level) from the 2 validated pathways to ALL KEGG human metabolic
maps. Same KEGG-topology source that scored 0.99 on glycolysis, now generalized: auto-detect each map's entry
compounds (no hardcoded roots) and SCC-condense to survive metabolic cycles.

For each KEGG metabolism map (hsa000xx–hsa001xx, excluding the giant global-overview maps):
  1. parse reactions → substrate→product compound edges (currency metabolites dropped)
  2. SCC-condense the compound graph (a metabolic cycle like the TCA loop collapses to one node) → a DAG
  3. longest-path rank of each compound on the DAG = its depth in the pathway
  4. enzyme step level = rank of the compound its reaction produces, normalized 0..1 within the map
Genes in a collapsed cycle are flagged (no clean step). A gene in several maps is aggregated; if its level differs
across maps it is flagged context-dependent — same honesty rule as the signaling tier.
-> outputs/orphan/metabolic_levels.json
"""
import os, sys, json, ssl, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")
_CA = "/root/.ccr/ca-bundle.crt"
_CTX = ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None

# KEGG currency compound IDs (ubiquitous cofactors/inorganics) — excluded so they don't shortcut the topology
CURRENCY_CPD = {"C00001", "C00002", "C00008", "C00020", "C00009", "C00013", "C00003", "C00004", "C00006",
                "C00005", "C00011", "C00007", "C00010", "C00080", "C00014", "C00027", "C00028", "C00030",
                "C00138", "C00139", "C00016", "C00061", "C00255", "C00342", "C00399", "C00390"}
# global / overview maps that are NOT single ordered pathways — skip them
GLOBAL_MAPS = {"hsa01100", "hsa01110", "hsa01120", "hsa01200", "hsa01210", "hsa01212", "hsa01230",
               "hsa01232", "hsa01250", "hsa01240", "hsa01220"}


def _get(url):
    return urllib.request.urlopen(url, timeout=30, context=_CTX).read().decode()


def metabolic_map_ids():
    """all human KEGG metabolism pathway maps (id 00010–01999), minus global-overview maps."""
    txt = _get("https://rest.kegg.jp/list/pathway/hsa")
    ids = []
    for ln in txt.splitlines():
        pid = ln.split("\t")[0].strip()
        num = pid.replace("hsa", "")
        if num.isdigit() and 10 <= int(num) < 2000 and pid not in GLOBAL_MAPS:
            ids.append(pid)
    return ids


def _kgml(kegg_id):
    p = f"/tmp/{kegg_id}.kgml"
    if not os.path.exists(p):
        try:
            open(p, "w").write(_get(f"https://rest.kegg.jp/get/{kegg_id}/kgml"))
        except Exception:
            return None
    return open(p).read()


def map_levels(kegg_id):
    """{gene_symbol: (normalized_level 0..1, in_cycle bool)} for one KEGG map, via SCC-condensation topology."""
    import xml.etree.ElementTree as ET
    import networkx as nx
    x = _kgml(kegg_id)
    if not x:
        return {}
    root = ET.fromstring(x)
    G = nx.DiGraph()
    rxn_prod = {}
    for r in root.findall("reaction"):
        rid = r.get("name")
        subs = [s.get("name") for s in r.findall("substrate") if s.get("name")]
        prods = [p.get("name") for p in r.findall("product") if p.get("name")]
        rxn_prod[rid] = [p for p in prods if p.split(":")[-1] not in CURRENCY_CPD]
        for a in subs:
            if a.split(":")[-1] in CURRENCY_CPD:
                continue
            for b in prods:
                if b.split(":")[-1] in CURRENCY_CPD:
                    continue
                G.add_edge(a, b)
    if G.number_of_edges() == 0:
        return {}
    # SCC condensation -> DAG, longest-path rank of each SCC
    cond = nx.condensation(G)
    order = list(nx.topological_sort(cond))
    depth = {n: 0 for n in cond.nodes}
    for n in order:
        for m in cond.successors(n):
            depth[m] = max(depth[m], depth[n] + 1)
    scc_of = {}
    incyc = {}
    for n in cond.nodes:
        mem = cond.nodes[n]["members"]
        for c in mem:
            scc_of[c] = n; incyc[c] = len(mem) > 1
    mx = max(depth.values()) or 1
    # gene entries -> reaction -> product compound depth
    out = {}
    for e in root.findall("entry"):
        if e.get("type") != "gene" or not e.get("reaction"):
            continue
        g = e.find("graphics")
        names = [t.strip() for t in (g.get("name", "") if g is not None else "").replace(",", " ").split()]
        cds = [scc_of[p] for rid in e.get("reaction").split() for p in rxn_prod.get(rid, []) if p in scc_of]
        if not cds:
            continue
        d = float(np.mean([depth[c] for c in cds])) / mx
        cyc = any(incyc[p] for rid in e.get("reaction").split() for p in rxn_prod.get(rid, []) if p in incyc)
        for nm in names:                                 # assign to ALL isozyme symbols in the entry (they share the step)
            nm = nm.rstrip(".")                          # KEGG truncates long lists with "..."
            if nm and nm[0].isalpha() and len(nm) <= 12 and all(ch.isalnum() or ch in "-" for ch in nm):
                out.setdefault(nm, (d, cyc))
    return out


def _bucket(x):
    return "upstream" if x < 0.34 else "downstream" if x > 0.66 else "midstream"


def validate(maps_levels):
    """report Spearman vs textbook order for glycolysis + TCA (the two we know the true order of)."""
    from scipy.stats import spearmanr
    import pathway_position as pp
    res = {}
    for pname, kegg in (("glycolysis", "hsa00010"), ("tca_cycle", "hsa00020")):
        lv = maps_levels.get(kegg, {})
        order = pp.PATHWAYS[pname]["order"]
        t, e = [], []
        for i, (sym, _) in enumerate(order):
            if sym in lv:
                t.append(i); e.append(lv[sym][0])
        res[pname] = round(float(spearmanr(t, e).statistic), 2) if len(set(e)) > 1 else None
    return res


def build(limit=None):
    ids = metabolic_map_ids()
    if limit:
        ids = ids[:limit]
    print(f"  {len(ids)} KEGG metabolic maps")
    maps_levels = {}
    per_gene = {}                    # symbol -> list of (level, in_cycle, map)
    for k, kid in enumerate(ids):
        lv = map_levels(kid)
        maps_levels[kid] = lv
        for sym, (d, cyc) in lv.items():
            per_gene.setdefault(sym, []).append((d, cyc, kid))
        if (k + 1) % 20 == 0:
            print(f"    {k+1}/{len(ids)} maps, {len(per_gene)} enzymes so far")
    return ids, maps_levels, per_gene


def main(limit=None):
    ids, maps_levels, per_gene = build(limit=limit)
    val = validate(maps_levels)
    labels = {}
    n_step = n_ctx = n_cyc = 0
    for sym, obs in per_gene.items():
        levels = np.array([o[0] for o in obs])
        anycyc = any(o[1] for o in obs)
        mean, spread = float(levels.mean()), float(levels.std())
        best_map = obs[0][2]
        if len(levels) >= 3 and spread > 0.3:
            labels[sym] = {"status": "context-dependent", "level": None,
                           "detail": f"metabolic level varies across {len(levels)} maps (mean {mean:.2f} ± {spread:.2f})"}
            n_ctx += 1
        elif anycyc and len(levels) == 1:
            labels[sym] = {"status": "metabolic-cycle", "level": round(mean, 2), "bucket": _bucket(mean),
                           "detail": f"in a metabolic cycle (e.g. {best_map}); order is conventional", "map": best_map}
            n_cyc += 1
        else:
            labels[sym] = {"status": "metabolic-step", "level": round(mean, 2), "bucket": _bucket(mean),
                           "detail": f"{_bucket(mean)} across {len(levels)} metabolic map(s)", "map": best_map}
            n_step += 1
    out = {"n_maps": len(ids), "validation_spearman": val, "n_enzymes": len(labels),
           "metabolic_step": n_step, "metabolic_cycle": n_cyc, "context_dependent": n_ctx, "labels": labels}
    json.dump(out, open(f"{OUT}/metabolic_levels.json", "w"), indent=1)
    print("=" * 80)
    print(f"METABOLIC LEVELS scaled to {len(ids)} KEGG maps → {len(labels)} enzymes")
    print(f"  validation vs textbook: glycolysis {val.get('glycolysis')}, TCA {val.get('tca_cycle')}")
    print(f"  metabolic-step {n_step}   metabolic-cycle {n_cyc}   context-dependent {n_ctx}")
    print("=" * 80)
    return out


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
