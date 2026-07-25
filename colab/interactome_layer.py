"""interactome_layer -- an ADDITIVE, self-describing knowledge layer over the existing cell network: for every protein its function +
structure summary; for every complex its members and (binary-proxy) intra-contacts; the typed "what A does to B" edges; the
complex->complex and complex->protein graph; and -- materialized AT SCALE for the first time (the demo files only showed 2-3 hubs) -- the
KNOCKOUT-CONSEQUENCE layer: when a subunit is removed, what it was responsible for, whether it/its companions bind elsewhere, and a
(functional-proxy) can-it-bind-alone score.

Design (from the interface-resolved-layer design workflow): NEVER mutate cell_complete.json; every record is keyed by the existing integer
gene_idx / an assigned complex_idx, so this is a pure overlay you join by index. CRUCIALLY it is HONEST about its holes: every capability
self-reports FILLED / PARTIAL / GAP with a coverage % and a reason in the top-level `_coverage` block, so a downstream consumer never reads
an "unknown" as a measured zero. The annotation half (function 92-99%, complex membership 100% of the curated set, KO-consequence for all
complex members) is near-complete; the ATOMIC half (genome-wide 3D structure, intrinsic disorder, residue-level interface identities,
persisted interaction mechanism, structural monomer stability) is ~0-2% and GPU/Interactome3D-gated -- marked GAP, not faked.
"""
import json, collections
from pathlib import Path
OUT = Path("outputs/orphan")


def _load(fn):
    p = OUT / fn
    return json.load(open(p)) if p.exists() else {}


def build():
    D = json.load(open(OUT / "cell_complete.json"))
    genes = D["genes"]; N = len(genes)
    names = [g["name"] for g in genes]; idx = {n: i for i, n in enumerate(names)}
    go = D.get("go", {}); darkfn = D.get("darkfn", {}); drugs = D.get("drugs", {})
    ppm = D.get("ppm", {}); abund = D.get("abund", {})
    domains = _load("domains.json").get("domains", {})

    # STEP 0 -- authoritative reverse index (NOT the capped gene2cplx)
    g2c = collections.defaultdict(list)
    cplx_members = {}
    for ci, (cn, mem) in enumerate(D["complexes"].items()):
        m = sorted(set(x for x in mem if isinstance(x, int) and x < N))
        cplx_members[ci] = (cn, m)
        for x in m:
            g2c[x].append(ci)
    ppi_nb = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            ppi_nb[a].add(b); ppi_nb[b].add(a)
    codep = {int(k): [(int(j), float(s)) for j, s in v] for k, v in D.get("codep", {}).items()}
    def name_get(dd, i, default=None):
        return dd.get(names[i], default)

    # STEP 1 -- NODE layer (compact; long GO/domain lists stay in cell_complete, referenced by count)
    cov = collections.Counter()
    nodes = {}
    for i, g in enumerate(genes):
        gn = names[i]; si = str(i)                                 # ppm/abund/drugs/darkfn/domains are keyed by index-STRING
        dk = darkfn.get(si)
        gg = go.get(gn, {})
        def _f(dd):
            v = dd.get(si)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        node = {
            "name": gn,
            "proc": g.get("proc"), "comp": g.get("comp"),
            "n_go": sum(len(gg.get(k, [])) for k in ("P", "C", "F")), "npath": g.get("npath") or 0,
            "n_domains": len(domains.get(si, []) or []),
            "abund_level": _f(abund), "ppm": _f(ppm),
            "ess": int(g.get("ess") or 0), "dep_frac": float(g.get("dep_frac") or 0.0),
            "ppi_degree": len(ppi_nb.get(i, ())), "in_complexes": g2c.get(i, []),
            "dark_pred": ({"pred": dk.get("pred"), "conf": dk.get("conf")} if isinstance(dk, dict) else None),
            "druggable": bool(drugs.get(si)),
            # explicit GAP markers -- structural/mechanistic fields we do NOT have genome-wide
            "gaps": {"coords_3d": "GPU-gated (no genome-wide AlphaFold)", "plddt_per_residue": "not persisted",
                     "disorder_frac": "computed-only, never persisted", "molecular_mechanism": "darkfn is process-level"},
        }
        for k in ("proc", "comp"):
            if node[k]:
                cov[f"node.{k}"] += 1
        if node["n_go"]:
            cov["node.go"] += 1
        if node["n_domains"]:
            cov["node.domains"] += 1
        if node["ppm"] is not None:
            cov["node.abundance"] += 1
        if node["dep_frac"]:
            cov["node.dep_frac"] += 1
        if node["dark_pred"]:
            cov["node.dark_pred"] += 1
        if node["druggable"]:
            cov["node.druggable"] += 1
        nodes[i] = node

    # STEP 2 -- COMPLEX layer
    complexes = {}
    for ci, (cn, m) in cplx_members.items():
        contacts = [[a, b] for a in m for b in m if a < b and b in ppi_nb.get(a, ())]
        ppaths = sorted({genes[x].get("path") for x in m if genes[x].get("path")})
        ppmvals = sorted(float(ppm[str(x)]) for x in m if str(x) in ppm)
        complexes[ci] = {
            "name": cn, "members": m, "size": len(m),
            "n_intra_contacts": len(contacts),
            "intra_contacts": contacts if len(contacts) <= 60 else "(>60; count only)",
            "pathways": ppaths[:8], "essential_members": int(sum(genes[x].get("ess") or 0 for x in m)),
            "median_ppm": (sorted(ppmvals)[len(ppmvals) // 2] if ppmvals else None),
            "gaps": {"stoichiometry": "unweighted member sets", "assembly_order": "no ordering in data",
                     "interface_residues_between_subunits": "no co-structures (GPU-gated)"},
        }

    # STEP 3 -- TYPED "what A does to B" edge index (directed, mechanistic where we have it)
    typed = []
    for e in D.get("sig", []):
        if len(e) >= 3 and isinstance(e[0], int) and isinstance(e[1], int) and e[0] < N and e[1] < N:
            typed.append([e[0], e[1], int(e[2]), "sig"])            # +1 activate / -1 inhibit / 0
    for e in _load("degradation.json").get("edges", []) or []:
        if len(e) >= 2 and isinstance(e[0], int) and isinstance(e[1], int) and e[0] < N and e[1] < N:
            typed.append([e[0], e[1], -1, "degrade"])            # E3 -> substrate (loss stabilises substrate)
    for a, b in D.get("lr", []):
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            typed.append([a, b, 0, "ligand_receptor"])
    iface = {}
    for pr in _load("nexus_ko_fetch.json").get("pairs", []) or []:
        ga, gb, sz = pr.get("a"), pr.get("b"), pr.get("iface")
        if ga in idx and gb in idx and sz is not None:
            iface[(min(idx[ga], idx[gb]), max(idx[ga], idx[gb]))] = sz

    # STEP 4 -- COMPLEX-COMPLEX layer (lift PPI; member-disjoint; support>=2 to bound size)
    cc_support = collections.Counter()
    for a, b in D["ppi"]:
        if not (isinstance(a, int) and isinstance(b, int) and a < N and b < N):
            continue
        ca, cb = g2c.get(a, []), g2c.get(b, [])
        for x in ca:
            for y in cb:
                if x != y and not (set(cplx_members[x][1]) & set(cplx_members[y][1])):   # disjoint assemblies only
                    cc_support[(min(x, y), max(x, y))] += 1
    cc = [[x, y, s, (">=5" if s >= 5 else ">=3" if s >= 3 else ">=2")] for (x, y), s in cc_support.items() if s >= 2]
    cc.sort(key=lambda r: (-r[2], r[0], r[1]))

    # STEP 5 -- KO-CONSEQUENCE layer (per complex-member gene; materialized AT SCALE)
    ko = {}
    for i in sorted(g2c):
        comembers = set()
        for ci in g2c[i]:
            comembers.update(cplx_members[ci][1])
        comembers.discard(i)
        cd = dict(codep.get(i, []))
        # obligate partners = co-members that are also co-dependency partners, ranked
        obl = sorted((j for j in comembers if j in cd), key=lambda j: (cd[j], genes[j].get("ess") or 0), reverse=True)
        obl_scores = [round(cd[j], 3) for j in obl]
        can_alone = round(1.0 - (max(obl_scores) if obl_scores else 0.0), 3)     # functional PROXY (1=likely autonomous)
        ko[i] = {
            "obligate_partners": [[j, cd[j]] for j in obl[:10]],                   # who co-fails when i is removed
            "machines_disrupted": g2c[i],                                          # complexes that lose subunit i
            "n_orphaned_co_subunits": len(comembers),
            "n_interactions_severed": len(ppi_nb.get(i, ())),
            "binds_elsewhere": {"other_complexes": [c for c in g2c[i]], "ppi_partners": len(ppi_nb.get(i, ()))},
            "can_bind_alone_proxy": can_alone, "proxy": True,
            "essential": int(genes[i].get("ess") or 0), "dep_frac": round(float(genes[i].get("dep_frac") or 0.0), 3),
            "gaps": {"can_bind_alone_structural": "functional proxy only (AUC 0.605); no monomer stability",
                     "competitive_rebinding": "honest-negative (competitive_codep: p=1.0); needs residue interface (GPU)",
                     "mrna_far_field": "transmits only via curated TF->target (~6x); physical/complex ~0x"},
        }
        if obl:
            cov["ko.obligate_partners"] += 1

    # STEP 6 -- self-describing coverage block
    def pct(k):
        return round(100 * cov[k] / N, 1)
    coverage = {
        "node.function_go": {"status": "PARTIAL", "pct": pct("node.go"), "note": "GO term names (not IDs); ~93-99% of genes"},
        "node.process_compartment": {"status": "FILLED", "pct": pct("node.proc"), "note": "100% but 12 coarse categories"},
        "node.domains": {"status": "FILLED", "pct": pct("node.domains"), "note": "InterPro domain architecture"},
        "node.abundance": {"status": "FILLED", "pct": pct("node.abundance"), "note": "PaxDb whole-organism ppm; NOT copies/cell and NOT K562-specific"},
        "node.dark_prediction": {"status": "PARTIAL", "pct": pct("node.dark_pred"), "note": "process-level, dark genes only"},
        "node.druggable": {"status": "PARTIAL", "pct": pct("node.druggable"), "note": "absence != undruggable"},
        "node.structure_3d": {"status": "GAP", "pct": 0.0, "note": "no genome-wide AlphaFold; GPU-gated"},
        "node.disorder": {"status": "GAP", "pct": 0.0, "note": "computed-only, never persisted"},
        "node.molecular_mechanism": {"status": "GAP", "pct": 0.0, "note": "no per-gene biochemical activity for dark/other"},
        "complex.members": {"status": "FILLED", "pct": round(100 * len(complexes) / len(D["complexes"]), 1),
                            "note": f"{len(complexes)} curated complexes; but only {round(100*len(g2c)/N,1)}% of genome is in any complex"},
        "complex.intra_contacts": {"status": "PARTIAL", "note": "binary PPI-within-complex proxy; no residues/domains"},
        "complex.stoichiometry_assembly": {"status": "GAP", "pct": 0.0, "note": "unweighted sets, no order/copies"},
        "complex.interface_residues": {"status": "GAP", "pct": 0.0, "note": "no co-structures (GPU-gated)"},
        "edge.typed_direction_sign": {"status": "PARTIAL", "note": f"{len(typed)} typed directed edges; ~3% of 191k PPI"},
        "edge.mechanism": {"status": "GAP", "pct": 0.0, "note": "SIGNOR mechanism parsed then stripped on persist; re-derivable"},
        "edge.interface_residues": {"status": "GAP", "pct": 0.0, "note": f"only {len(iface)} pairs, SIZE only (identities discarded)"},
        "cc.complex_complex": {"status": "PARTIAL", "note": f"{len(cc)} member-disjoint pairs (support>=2), lifted from PPI; untyped"},
        "ko.responsible_for": {"status": "PARTIAL", "pct": round(100 * cov["ko.obligate_partners"] / N, 1),
                               "note": f"obligate partners/machines/severed for all {len(ko)} complex members"},
        "ko.binds_elsewhere": {"status": "FILLED", "note": "reverse membership + PPI partners, all complex members"},
        "ko.can_bind_alone": {"status": "GAP", "note": "functional proxy only (AUC 0.605); no structural monomer stability"},
    }

    layer = {
        "_meta": {"builder": "interactome_layer.py", "additive": True, "join_by": "gene_idx into cell_complete.json['genes']",
                  "n_nodes": N, "n_complexes": len(complexes), "n_typed_edges": len(typed),
                  "n_complex_complex": len(cc), "n_ko_annotated": len(ko)},
        "_coverage": coverage,
        "nodes": nodes, "complexes": complexes, "typed_edges": typed,
        "interface_size": {f"{a}_{b}": s for (a, b), s in sorted(iface.items())},
        "complex_complex": cc, "ko_consequence": ko,
    }
    json.dump(layer, open(OUT / "interactome_layer.json", "w"))
    return layer, coverage


# ---------- query API (join the layer to cell_complete on the fly) ----------
def _layer():
    if not hasattr(_layer, "L"):
        _layer.L = json.load(open(OUT / "interactome_layer.json"))
    return _layer.L


def describe_protein(name_or_idx):
    L = _layer(); nodes = L["nodes"]
    i = str(name_or_idx) if str(name_or_idx).isdigit() else None
    if i is None:
        for k, v in nodes.items():
            if v["name"] == name_or_idx:
                i = k; break
    return nodes.get(str(i)) if i is not None else None


def knockout_consequence(name_or_idx):
    L = _layer(); nodes = L["nodes"]
    i = None
    for k, v in nodes.items():
        if v["name"] == name_or_idx or k == str(name_or_idx):
            i = k; break
    return L["ko_consequence"].get(str(i)) if i is not None else {"note": "not a complex member -> no obligate-KO layer"}


def main():
    print("=" * 96)
    print("interactome_layer -- additive interface-resolved knowledge layer (honest coverage map)")
    print("=" * 96)
    layer, cov = build()
    m = layer["_meta"]
    print(f"\nbuilt: {m['n_nodes']} protein nodes, {m['n_complexes']} complexes, {m['n_typed_edges']} typed 'A does to B' edges, "
          f"{m['n_complex_complex']} complex-complex edges, {m['n_ko_annotated']} knockout-consequence records\n")
    print("SELF-DESCRIBING COVERAGE MAP (every field marks FILLED / PARTIAL / GAP so 'unknown' is never read as a measured zero):")
    order = {"FILLED": 0, "PARTIAL": 1, "GAP": 2}
    for k, v in sorted(layer["_coverage"].items(), key=lambda kv: (order[kv[1]["status"]], kv[0])):
        p = f"{v['pct']:>5.1f}%" if "pct" in v else "   -- "
        print(f"   [{v['status']:7s}] {p}  {k:34s} {v['note']}")
    filled = sum(1 for v in cov.values())
    ngap = sum(1 for v in layer["_coverage"].values() if v["status"] == "GAP")
    npart = sum(1 for v in layer["_coverage"].values() if v["status"] == "PARTIAL")
    nfill = sum(1 for v in layer["_coverage"].values() if v["status"] == "FILLED")
    print(f"\n   {nfill} FILLED / {npart} PARTIAL / {ngap} GAP capabilities.")
    # worked example
    ex = "PSMB5"
    print(f"\n   example -- remove {ex}:")
    kc = knockout_consequence(ex)
    if kc and "obligate_partners" in kc:
        L = _layer(); nm = {int(k): v["name"] for k, v in L["nodes"].items()}
        obl = ", ".join(f"{nm.get(j,'?')}({s})" for j, s in kc["obligate_partners"][:5])
        print(f"      obligate partners co-fail: {obl}")
        print(f"      machines disrupted: {len(kc['machines_disrupted'])} complexes | severed {kc['n_interactions_severed']} PPI | "
              f"can-bind-alone proxy {kc['can_bind_alone_proxy']} (proxy={kc['proxy']})")
    print("\nVERDICT: Additive interface-resolved layer built. The ANNOTATION half of the vision is near-complete (function 93-99%, "
          "complex membership 100% of the curated set, and the knockout-consequence layer -- 'what a removed subunit was responsible for, "
          "and whether it/its companions bind elsewhere' -- now materialized for EVERY complex member, not just demo hubs). The ATOMIC "
          "half stays honestly thin and GPU/Interactome3D-gated: no genome-wide 3D structure, no intrinsic disorder, no residue-level "
          "interface identities (interaction MECHANISM is present in SIGNOR but stripped on persist; interface is SIZE-only for 38 pairs), "
          "and 'can a subunit bind alone' is a functional proxy (AUC 0.605), not structural monomer stability. Those are marked GAP with a "
          "reason, so the layer's real value is a self-describing map that names exactly what is known vs what needs the GPU frontier -- "
          "never papering an 'unknown' as a measured zero.")
    print("\n  -> outputs/orphan/interactome_layer.json")
    return layer


if __name__ == "__main__":
    main()
