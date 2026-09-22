"""cell_map — turn the complete-cell database into a spatial NETWORK MAP: nodes = genes placed in their subcellular
compartment (laid out like a cell), edges = the physical-interaction network (STRING) AND the co-dependency network
(DepMap). A compartment-anchored force layout is computed here so the browser only renders precomputed positions.
-> outputs/orphan/cell_map.json  (compact arrays, inlined by the HTML artifact)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")

# 12 compartments laid out like a cell (x,y in [0,1]; plasma membrane = perimeter, extracellular = outside).
COMP = {
    "nucleus":         {"anchor": (0.50, 0.46), "r": 0.15, "color": "#4d8df0"},
    "cytoplasm":       {"anchor": (0.50, 0.54), "r": 0.30, "color": "#8891a0"},
    "ER":              {"anchor": (0.33, 0.60), "r": 0.11, "color": "#9b7ce8"},
    "Golgi":           {"anchor": (0.62, 0.67), "r": 0.08, "color": "#db6fb0"},
    "mitochondrion":   {"anchor": (0.73, 0.36), "r": 0.10, "color": "#e35f3f"},
    "cytoskeleton":    {"anchor": (0.27, 0.38), "r": 0.10, "color": "#e39338"},
    "endosome":        {"anchor": (0.70, 0.60), "r": 0.06, "color": "#46a9d6"},
    "lysosome":        {"anchor": (0.78, 0.52), "r": 0.05, "color": "#e06a86"},
    "peroxisome":      {"anchor": (0.66, 0.74), "r": 0.04, "color": "#93c73f"},
    "plasma membrane": {"anchor": (0.50, 0.50), "r": 0.46, "color": "#2fb27f", "ring": True},
    "membrane":        {"anchor": (0.50, 0.50), "r": 0.40, "color": "#34b3ac", "ring": True},
    "extracellular":   {"anchor": (0.50, 0.50), "r": 0.52, "color": "#c9a541", "ring": True},
}
CIDX = {c: i for i, c in enumerate(COMP)}


def build(n_target=3200, max_ppi=22000, max_codep=14000, iters=140, seed=0):
    import cellos
    C = cellos.CellKernel(quiet=True).C
    N = len(C.genes)
    name = [C.genes[i].get("name") for i in range(N)]
    idx = {name[i]: i for i in range(N)}

    def depf(i):
        v = C.genes[i].get("dep_frac")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # PPI degree over the whole cell (for hub selection + node size)
    deg = {i: len(C.ppi_adj.get(i, ())) for i in range(N)}

    # ---- select the connected functional CORE: essential + TFs + complex members + top hubs ----
    ess = {i for i in range(N) if (depf(i) or 0) > 0.5}
    tfs = {i for i in range(N) if C.genes[i].get("tf")}
    cplx = set()
    for members in (C.D.get("complexes") or {}).values() if isinstance(C.D.get("complexes"), dict) else []:
        for g in (members if isinstance(members, (list, tuple)) else []):
            gi = idx.get(g) if isinstance(g, str) else (g if isinstance(g, int) else None)
            if gi is not None:
                cplx.add(gi)
    core = set(ess) | set(tfs) | cplx
    # fill to n_target with the highest-degree remaining genes that have a compartment
    rest = sorted((i for i in range(N) if i not in core and C.genes[i].get("comp") in COMP),
                  key=lambda i: -deg[i])
    for i in rest:
        if len(core) >= n_target:
            break
        core.add(i)
    core = [i for i in core if C.genes[i].get("comp") in COMP][:n_target]
    cset = set(core)
    ci = {g: a for a, g in enumerate(core)}
    n = len(core)

    # ---- edges: physical PPI among core (unweighted adj), capped by combined degree ----
    ppi = set()
    for gi in core:
        for gj in C.ppi_adj.get(gi, ()):
            if gj in cset and gi < gj:
                ppi.add((gi, gj))
    ppi = sorted(ppi, key=lambda e: -(deg[e[0]] + deg[e[1]]))[:max_ppi]
    ppi_e = [[ci[a], ci[b]] for a, b in ppi]

    # ---- edges: co-dependency among core (DepMap), top by co-essentiality ----
    codep_e = []
    try:
        z = np.load(f"{OUT}/depmap_vecs.npz", allow_pickle=True)
        dsym = {str(s): k for k, s in enumerate(z["syms"])}
        Z = np.nan_to_num(z["Z"].astype("float32"), nan=0.0)
        nz = np.linalg.norm(Z, axis=1, keepdims=True); nz[nz == 0] = 1.0
        Zn = Z / nz
        have = [(a, dsym[name[g]]) for a, g in enumerate(core) if name[g] in dsym]
        rows = np.array([h[1] for h in have]); locs = np.array([h[0] for h in have])
        sub = Zn[rows]
        pairs = []
        for t in range(len(have)):
            sims = sub @ sub[t]; sims[t] = -1
            for u in np.argsort(-sims)[:6]:
                if sims[u] > 0.35 and locs[t] < locs[u]:
                    pairs.append((float(sims[u]), int(locs[t]), int(locs[u])))
        pairs.sort(reverse=True)
        seen = set()
        for s, a, b in pairs[:max_codep]:
            if (a, b) not in seen:
                seen.add((a, b)); codep_e.append([a, b])
    except Exception as e:
        print("  (co-dependency edges skipped:", str(e)[:60], ")")

    # ---- compartment-anchored force layout (concentric cell: membrane ring on the perimeter) ----
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    comp_of = np.array([CIDX[C.genes[g].get("comp")] for g in core])
    comp_core = defaultdict(list)
    for a in range(n):
        comp_core[C.genes[core[a]].get("comp")].append(a)
    anchors = np.zeros((n, 2)); grav = np.zeros(n)
    for c, members in comp_core.items():
        spec = COMP[c]; R = spec["r"]
        if spec.get("ring"):                                # perimeter anchor by angular rank -> a clean circle
            m = len(members)
            for j, a in enumerate(members):
                th = 2 * np.pi * j / max(m, 1)
                anchors[a] = [0.5 + R * np.cos(th), 0.5 + R * np.sin(th)]; grav[a] = 0.17
        elif c == "cytoplasm":                              # loose mid annulus so it rings the nucleus, not overlaps
            m = len(members)
            for j, a in enumerate(members):
                th = 2 * np.pi * j / max(m, 1); rr = 0.15 + 0.07 * rng.random()
                anchors[a] = [0.5 + rr * np.cos(th), 0.5 + rr * np.sin(th)]; grav[a] = 0.11
        else:
            cx, cy = spec["anchor"]
            for a in members:
                anchors[a] = [cx + rng.normal(0, 0.015), cy + rng.normal(0, 0.015)]; grav[a] = 0.13
    P = anchors + rng.normal(0, 0.015, (n, 2))
    E = np.array(ppi_e) if ppi_e else np.zeros((0, 2), int)
    for it in range(iters):
        cooling = 1.0 - it / (iters + 6)
        disp = np.zeros_like(P)
        d = P[:, None, :] - P[None, :, :]
        dist2 = (d ** 2).sum(2) + 0.010
        disp += (d / dist2[:, :, None]).sum(1) * 0.00006        # very WEAK repulsion (only local spacing)
        if len(E):
            vec = P[E[:, 1]] - P[E[:, 0]]
            np.add.at(disp, E[:, 0], vec * 0.005)
            np.add.at(disp, E[:, 1], -vec * 0.005)
        disp += (anchors - P) * grav[:, None]                   # STRONG anchor gravity -> compartments are tight blobs
        dn = np.linalg.norm(disp, axis=1, keepdims=True)
        disp = disp / np.maximum(dn, 1e-9) * np.minimum(dn, 0.018)
        P += disp * cooling
        # radial soft clamp (safety only — strong gravity already keeps things in)
        v = P - 0.5; r = np.hypot(v[:, 0], v[:, 1]) + 1e-9
        over = r > 0.66
        newr = np.where(over, 0.66 + 0.2 * (r - 0.66), r)
        P = 0.5 + v / r[:, None] * newr[:, None]
    # UNIFORM centred scale (preserve aspect so the cell stays circular) -> logical, center (800,520)
    P -= P.mean(0)
    Rmax = np.percentile(np.linalg.norm(P, axis=1), 99.5)
    P *= (500.0 / max(Rmax, 1e-6))
    xs = (800 + P[:, 0]).round(1).tolist()
    ys = (520 + P[:, 1]).round(1).tolist()
    # per-compartment LABEL position: ring compartments at 12 o'clock of their ring, others at member centroid
    Pn = np.column_stack([xs, ys])
    label_xy = {}
    ring_ang = {"extracellular": -np.pi / 2, "plasma membrane": np.pi / 2, "membrane": np.pi}  # top / bottom / left
    for c, members in comp_core.items():
        pts = Pn[members]
        if COMP[c].get("ring"):                                 # pin at a fixed peripheral angle (no overlap)
            a = ring_ang.get(c, -np.pi / 2); rr = 468.0
            label_xy[c] = [800 + rr * np.cos(a), 520 + rr * np.sin(a)]
        else:
            label_xy[c] = [float(pts[:, 0].mean()), float(pts[:, 1].mean())]

    flags = []
    for g in core:
        f = 0
        if (depf(g) or 0) > 0.5: f |= 1
        if C.genes[g].get("dark"): f |= 2
        if C.genes[g].get("tf"): f |= 4
        flags.append(f)
    comp_counts = {c: int(sum(1 for g in range(N) if C.genes[g].get("comp") == c)) for c in COMP}

    out = {
        "compartments": [{"name": c, "color": COMP[c]["color"], "cell_total": comp_counts[c],
                          "label": label_xy.get(c)} for c in COMP],
        "nodes": {"name": [name[g] for g in core], "x": xs, "y": ys,
                  "comp": comp_of.tolist(), "deg": [int(min(deg[g], 400)) for g in core], "flag": flags},
        "edges_ppi": ppi_e, "edges_codep": codep_e,
        "stats": {"cell_genes": N, "mapped": n, "ppi_edges_total": sum(len(v) for v in C.ppi_adj.values()) // 2,
                  "ppi_shown": len(ppi_e), "codep_shown": len(codep_e),
                  "essential": int(sum(1 for f in flags if f & 1)), "tf": int(sum(1 for f in flags if f & 4)),
                  "dark": int(sum(1 for f in flags if f & 2))},
    }
    json.dump(out, open(f"{OUT}/cell_map.json", "w"))
    sz = os.path.getsize(f"{OUT}/cell_map.json") / 1024
    print(f"  cell_map: {n} nodes, {len(ppi_e)} PPI + {len(codep_e)} co-dep edges  ({sz:.0f} KB)")
    print("  compartment counts:", {c: comp_counts[c] for c in list(COMP)[:6]})
    return out


if __name__ == "__main__":
    build()
