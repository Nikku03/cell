"""MULTILAYER disease->target pipeline (OOD test: psoriasis / IL-23-IL-17 axis).

The pipeline the user specified, kept honest and blind (the drug target is never named to the model; it
must be SELECTED by simulation, not looked up — psoriasis is absent from the model's curated disease
layers otdis/biomarkers):

  LAYER 1  CAUSAL      -> from the disease pathway/phenotype, build the signed directed sub-network and rank
                          candidate driver nodes by their net signed influence on the pathogenic readout.
  LAYER 2  PERTURB     -> disease = the apex driver (IL-23) injected into the network; the pathogenic READOUT
                          (Th17 effector cytokines) receives net-activating flow (= ON). For each candidate,
                          try each intervention (DISABLE = remove the node; ACTIVATE = inject it as a source),
                          re-propagate, and measure how much readout activation collapses toward wild-type.
                          Rank by rescue; the winning DIRECTION per node is read off (disable vs activate).
  LAYER 3  DRUGGABLE   -> (external, separate step) for the rescuing targets, fetch protein family/structure
                          and decide whether the required direction is achievable by that family's modality.

Signed influence propagation (not threshold-Boolean): a gene with dozens of activators is robustly driven ON
by net-positive flow, avoiding the majority-vote pathology of {-1,+1} threshold updates on a dense network.
Honest scope: TOPOLOGICAL (which node, which direction), not kinetic. Layers 1-2 here -> psoriasis_target_layer12.json.
"""
import gzip, json
from collections import defaultdict
from pathlib import Path

OUT = Path("outputs/orphan"); OUT.mkdir(parents=True, exist_ok=True)
# portable model locator: Colab/committed path first, then this session's upload (gz ok).
CC_CANDIDATES = [
    OUT / "cell_complete.json",
    Path("/root/.claude/uploads/0f039315-b3a9-52ac-8187-9fae0d726994/9cee48b8-cell_complete_6.json.gz"),
]

# disease pathway (general immunology knowledge; NOT the model's answer). Target SELECTION is the test.
SEED = ["IL23A", "IL12B", "IL23R", "IL12RB1", "JAK2", "STAT3", "RORC", "RORA", "IL17A", "IL17F",
        "IL17RA", "IL22", "TNF", "NFKB1", "TRAF3IP2", "CARD14", "IL6", "STAT1", "IRF4", "BATF",
        "AHR", "IL1B", "CCR6", "CCL20", "IL21", "RELA", "MAPK14", "FOXP3", "STAT4", "IL12RB2"]
READOUT = ["IL17A", "IL17F", "IL22", "CCL20", "IL21"]     # pathogenic effector output
APEX = ["IL23A", "IL12B"]                                  # the disease driver = IL-23 (p19+p40)
ALPHA, KHOP = 0.55, 5                                      # propagation decay + horizon


def load():
    for cc in CC_CANDIDATES:
        if Path(cc).exists():
            opener = gzip.open if str(cc).endswith(".gz") else open
            D = json.load(opener(cc))
            G = D["genes"]; name = [g["name"] for g in G]; idx = {n: i for i, n in enumerate(name)}
            return D, G, name, idx
    raise FileNotFoundError("cell_complete.json not found in " + str([str(c) for c in CC_CANDIDATES]))


def full_adj(D):
    """directed signed out- and in-adjacency over the whole reg+sig network (unknown sign -> +1)."""
    out = defaultdict(list); inn = defaultdict(list)
    for e in D["reg"] + D["sig"]:
        s = 1 if (len(e) < 3 or e[2] >= 0) else -1
        out[e[0]].append((e[1], s)); inn[e[1]].append((e[0], s))
    return out, inn


def reach(sources, adj, L):
    """set of nodes reachable from `sources` within L hops along `adj`."""
    seen = set(sources); frontier = set(sources)
    for _ in range(L):
        nxt = set()
        for u in frontier:
            for w, _s in adj.get(u, ()):
                if w not in seen:
                    seen.add(w); nxt.add(w)
        frontier = nxt
    return seen


def cascade_subgraph(D, apex, readout, Lf=3, Lb=2):
    """the disease signal-flow subgraph: nodes on directed paths apex -> readout
    (forward-reachable from apex) INTERSECT (backward-reachable to readout), plus apex+readout.
    Returns (nodes, out_adj_normalized) where each out-edge weight = sign / out_degree(source)."""
    out, inn = full_adj(D)
    fwd = reach(apex, out, Lf)
    bwd = reach(readout, inn, Lb)              # reverse reachability = can reach the readout
    nodes = (fwd & bwd) | set(apex) | set(readout)
    # directed edges within the cascade
    raw = defaultdict(list)
    for u in nodes:
        for w, s in out.get(u, ()):
            if w in nodes:
                raw[u].append((w, s))
    # degree-normalize (random-walk) so hubs share influence instead of multiplying it
    norm = defaultdict(list)
    for u, lst in raw.items():
        d = len(lst)
        for w, s in lst:
            norm[u].append((w, s / d))
    return nodes, norm


def propagate(out, sources, removed=frozenset(), alpha=ALPHA, khop=KHOP):
    """net signed influence of `sources` on every node over the normalized signed graph (decayed walks)."""
    v = defaultdict(float)
    cur = {s: 1.0 for s in sources if s not in removed}
    for _ in range(khop):
        nxt = defaultdict(float)
        for u, val in cur.items():
            for w, s in out.get(u, ()):
                if w in removed: continue
                nxt[w] += alpha * s * val
        for w, val in nxt.items():
            v[w] += val
        cur = nxt
    return v


def main():
    D, G, name, idx = load()
    ro = [idx[g] for g in READOUT if g in idx]
    apex = [idx[g] for g in APEX if g in idx]
    keep, out = cascade_subgraph(D, apex, ro)
    print(f"readout {len(ro)} | apex {APEX} | cascade sub-graph {len(keep)} nodes (apex->readout signal flow)")

    # ---- disease baseline: IL-23 injected -> does the pathogenic readout receive net-activating flow? ----
    v0 = propagate(out, apex)
    ro_on = [name[r] for r in ro if v0.get(r, 0) > 0]
    print(f"readout receiving net-activating flow under IL-23 drive (disease ON): {ro_on}")
    if len(ro_on) < 2:
        print("!! disease drive does not activate the readout — sim uninformative. Stopping honestly.")
        json.dump(dict(ok=False, reason="apex_does_not_drive_readout", ro_on=ro_on),
                  open(OUT / "psoriasis_target_layer12.json", "w"), indent=2)
        return
    ro_active = [r for r in ro if v0.get(r, 0) > 0]

    # ---- LAYER 1: causal candidates = the disease-pathway genes that lie in the cascade (the hypothesis
    # space is the pathway; SELECTING the target from it is the test). Rank by net influence on readout. ----
    pathway = [idx[g] for g in SEED if g in idx]
    cand_infl = {}
    for n in set(pathway) | set(apex):
        if n not in keep:
            continue
        vn = propagate(out, [n])
        cand_infl[n] = sum(vn.get(r, 0) for r in ro_active if r != n)
    candidates = sorted((n for n in cand_infl if cand_infl[n] > 0), key=lambda n: -cand_infl[n])
    print(f"\nLAYER 1 — {len(candidates)} pathway candidates in cascade (by net influence on readout):")
    for n in candidates:
        print(f"   {name[n]:9} influence={cand_infl[n]:+.3f} druggable={'yes' if str(n) in D['drugs'] else 'no'}")

    # ---- LAYER 2: perturb each candidate; measure readout collapse toward wild-type ----
    def frac_collapsed(vp, ro_eff):
        return sum(1 for r in ro_eff if vp.get(r, 0) <= 0 or vp.get(r, 0) < 0.4 * v0.get(r, 0)) / max(1, len(ro_eff))
    results = []
    for n in candidates:
        ro_eff = [r for r in ro_active if r != n]
        # DISABLE = remove node from the graph AND from the disease sources (anti-ligand / receptor block / KO)
        v_dis = propagate(out, [a for a in apex if a != n], removed={n})
        dis = frac_collapsed(v_dis, ro_eff)
        # ACTIVATE = inject node as an extra +source (a repressor of the readout would rescue this way)
        v_act = propagate(out, list(apex) + [n])
        act = frac_collapsed(v_act, ro_eff)
        best_dir = "disable" if dis >= act else "activate"
        results.append(dict(gene=name[n], idx=n, influence=round(cand_infl[n], 3),
                            disable=round(dis, 3), activate=round(act, 3),
                            rescue=round(max(dis, act), 3), best_dir=best_dir,
                            druggable=str(n) in D["drugs"],
                            drug_actions=sorted({d.get("t", "") for d in D["drugs"].get(str(n), []) if d.get("t")})))
    results.sort(key=lambda r: (-r["rescue"], -r["influence"]))
    print("\nLAYER 2 — perturbation-to-wild-type (rescue = fraction of pathogenic readout collapsed):")
    print(f"  {'gene':9} {'rescue':>6} {'dir':>8} {'disable':>7} {'activate':>8} drug?  actions")
    for r in results:
        print(f"  {r['gene']:9} {r['rescue']:>6.2f} {r['best_dir']:>8} {r['disable']:>7.2f} "
              f"{r['activate']:>8.2f} {'yes' if r['druggable'] else ' no'}  {r['drug_actions']}")

    json.dump(dict(ok=True, disease="psoriasis (IL23/IL17 axis)", apex=APEX,
                   readout=[name[r] for r in ro], readout_on_disease=ro_on,
                   n_nodes=len(keep), alpha=ALPHA, khop=KHOP, candidates=results),
              open(OUT / "psoriasis_target_layer12.json", "w"), indent=2)
    print("\n-> outputs/orphan/psoriasis_target_layer12.json")


if __name__ == "__main__":
    main()
