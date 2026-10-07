"""DISEASE -> REVERSAL — integrate phenotype-localization with attractor reprogramming.

This is the module the conversation converged on: instead of asking a weak signature-retriever to *name*
the target, we (1) localize the disrupted regulatory sub-state from the phenotype, (2) treat the signed
TF->TF network as a threshold-Boolean dynamical system whose stable states are the disease and healthy
*attractors*, and (3) SEARCH for the minimal set of interventions (TF activations/repressions) that
reprogram the disease basin back to the healthy basin -- "reach the target or a generic corrector".

It is deliberately SELF-CONTAINED and network-agnostic: the reprogramming algorithm operates on any signed
regulatory network passed in, so it can be *validated in isolation* (see validate_disease_to_reversal.py)
without the 16k-gene model. `load_cell_model()` is the adapter that plugs it into the NRxBW branch's
`cell_complete.json` on Colab; when that file is absent the module still runs on any provided network.

Two driver-finding methods are provided so the improvement is explicit:
  - `transition_setdiff` : the CURRENT attractor-layer behaviour (compute_attractors.py) — report every TF
                           that differs between the two basins, ranked by out-degree. High recall, but the
                           causal drivers are drowned in downstream passenger TFs (low precision).
  - `control_drivers`    : proper Boolean target-control — greedily clamp the intervention that, once the
                           dynamics re-settle, moves the disease basin closest to the healthy basin; stop
                           when the healthy state is reached. Recovers the MINIMAL causal driver set.

Honest scope: this is TOPOLOGICAL (which TFs to flip and in what order), not kinetic (by how much / how
fast) — kinetics is the model's acknowledged wall. Output is a ranked, wet-lab-testable reprogramming
hypothesis, not a validated cure.
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
MAXIT = 100


# ---------------------------------------------------------------------------
# Boolean regulatory dynamics (kinetics-free; matches compute_attractors.py semantics)
# ---------------------------------------------------------------------------
def build_incoming(edges):
    """edges: iterable of (src, tgt, sign in {+1,-1}). -> (W_in, out_deg, nodes).
    W_in[j] = list of (regulator i, sign) driving node j."""
    W_in = defaultdict(list)
    out_deg = defaultdict(int)
    nodes = set()
    for e in edges:
        i, j = e[0], e[1]
        s = 1 if (len(e) < 3 or e[2] >= 0) else -1
        W_in[j].append((i, s))
        out_deg[i] += 1
        nodes.add(i); nodes.add(j)
    return W_in, out_deg, nodes


def iterate(state, W_in, order, clamp=None):
    """Synchronous threshold-Boolean update to a fixed point.
        s_j(t+1) = sign( SUM_i W_ij * s_i(t) ),  net==0 -> hold previous value.
    `clamp` nodes are held fixed (the applied interventions) so the cell settles into the basin those
    interventions define. States are in {-1, 0, +1}."""
    s = dict(state)
    clamp = clamp or {}
    for c, v in clamp.items():
        s[c] = v
    for _ in range(MAXIT):
        nxt = dict(s)
        for j in order:
            if j in clamp:
                nxt[j] = clamp[j]; continue
            net = 0
            for i, sgn in W_in.get(j, ()):
                net += sgn * s.get(i, 0)
            if net > 0:
                nxt[j] = 1
            elif net < 0:
                nxt[j] = -1
            # net == 0 -> keep previous (nxt[j] already = s[j])
        if nxt == s:
            break
        s = nxt
    return s


def active(state):
    """The 'ON' set — a basin is identified by which nodes are active (>0)."""
    return frozenset(n for n, v in state.items() if v > 0)


def settle(seed, W_in, order, clamp=None):
    """seed a state, run to its attractor, return (fixed_point, active_set)."""
    fp = iterate(seed, W_in, order, clamp)
    return fp, active(fp)


def hamming(a, b):
    """symmetric-difference size between two active sets = basin distance."""
    return len(a ^ b)


# ---------------------------------------------------------------------------
# Driver finding
# ---------------------------------------------------------------------------
def transition_setdiff(a_active, b_active, out_deg, top=None):
    """BASELINE (what compute_attractors.py reports): every TF differing between basins A and B, split into
    activate (ON in B not A) / repress (ON in A not B), ranked by out-degree. barrier = # differing TFs.
    Returns dict(activate, repress, barrier, drivers) where `drivers` is the flat ranked node list."""
    turn_on = b_active - a_active           # must switch ON to go A->B
    turn_off = a_active - b_active          # must switch OFF to go A->B
    key = lambda n: -out_deg.get(n, 0)
    on = sorted(turn_on, key=key)
    off = sorted(turn_off, key=key)
    drivers = sorted(turn_on | turn_off, key=key)
    if top:
        on, off, drivers = on[:top], off[:top], drivers[:top]
    return dict(activate=on, repress=off, barrier=len(turn_on) + len(turn_off), drivers=drivers)


def control_drivers(start_state, target_active, W_in, order, budget=8, candidates=None,
                    out_deg=None):
    """PROPER Boolean target-control. Greedily choose the intervention (clamp a TF ON/OFF) that, after the
    network re-settles from the diseased state with all chosen clamps held, minimises distance to the
    healthy basin. Stop when the healthy state is reached or `budget` is spent.

    This is the faithful "step-by-step reverse the pathway, changing everything accordingly, until we reach
    the target": each step is one intervention, applied to the *current* diseased cell, and we let the whole
    network respond before choosing the next.

    Returns dict(path, reached, final_distance, steps) where path is the ordered list of
    dict(gene, direction, distance_after, is_hub)."""
    start_active = active(start_state)
    # default candidate interventions: the TFs that differ between the two basins (fix what's wrong).
    if candidates is None:
        candidates = set(start_active ^ target_active)
    else:
        candidates = set(candidates)
    out_deg = out_deg or {}
    hub_cut = 0
    if out_deg:
        degs = sorted(out_deg.values())
        hub_cut = degs[int(0.9 * (len(degs) - 1))] if degs else 0    # top-decile out-degree = "hub"

    clamps = {}
    path = []
    cur_active = start_active
    reached = (hamming(cur_active, target_active) == 0)
    for _ in range(budget):
        if reached:
            break
        best = None                                   # (distance, node, value, resulting_active)
        for c in candidates:
            if c in clamps:
                continue
            v = 1 if c in target_active else -1        # push the TF toward its healthy value
            trial = dict(clamps); trial[c] = v
            _, res_active = settle(start_state, W_in, order, clamp=trial)
            d = hamming(res_active, target_active)
            if best is None or d < best[0] or (d == best[0] and out_deg.get(c, 0) > out_deg.get(best[1], 0)):
                best = (d, c, v, res_active)
        if best is None:
            break
        d, c, v, res_active = best
        # only accept the intervention if it helps (strictly reduces distance); else stop (local optimum).
        if d >= hamming(cur_active, target_active) and path:
            break
        clamps[c] = v
        cur_active = res_active
        path.append(dict(gene=c, direction=("activate" if v > 0 else "repress"),
                         distance_after=d, is_hub=bool(out_deg.get(c, 0) >= hub_cut and hub_cut > 0)))
        if d == 0:
            reached = True
    return dict(path=path, reached=reached, final_distance=hamming(cur_active, target_active),
                steps=len(path))


def classify_intervention(step, out_deg, target_active):
    """root-cause driver (high-influence / upstream master) vs generic downstream corrector."""
    if step.get("is_hub"):
        return "root_cause_driver"          # high out-degree master whose flip cascades
    return "local_corrector"


# ---------------------------------------------------------------------------
# Robustness to the incomplete/noisy interactome (Menche 2015): the human
# interactome is incomplete and the reg network is 83% activating (under-annotated
# for repression). A prediction that only holds on the exact edge set is fragile.
# Resample the network (drop/add a fraction of edges), re-run the reversal, and
# score each driver by how often it survives -> a stability/confidence flag.
# ---------------------------------------------------------------------------
def resample_edges(edges, drop_p, rng, add_p=0.0, node_pool=None):
    """Drop a `drop_p` fraction of edges; optionally add `add_p`*|E| random edges (the interactome is both
    incomplete AND noisy). Returns a resampled edge list."""
    keep = [e for e in edges if rng.random() > drop_p]
    if add_p > 0 and node_pool:
        for _ in range(int(add_p * len(edges))):
            a = rng.choice(node_pool); b = rng.choice(node_pool)
            if a != b:
                keep.append((a, b, rng.choice([1, -1])))
    return keep


def bootstrap_stability(run_fn, edges, K=8, drop_p=0.15, add_p=0.0, rng_seed=0):
    """Engine-agnostic robustness wrapper. `run_fn(edges) -> iterable of predicted items` does the full
    prediction on a given edge list; we resample the network K times and return {item: selection_frequency}.
    A frequency near 1.0 = robust to interactome incompleteness; low = fragile, should be flagged."""
    rng = random.Random(rng_seed)
    node_pool = list({e[0] for e in edges} | {e[1] for e in edges})
    freq = defaultdict(int)
    for _ in range(K):
        keep = resample_edges(edges, drop_p, rng, add_p, node_pool)
        for item in set(run_fn(keep)):
            freq[item] += 1
    return {k: v / K for k, v in freq.items()}


def flag_by_stability(items, stability, robust_thresh=0.6):
    """Split predicted items into robust (stability >= threshold) and fragile (flagged)."""
    robust = [i for i in items if stability.get(i, 0.0) >= robust_thresh]
    fragile = [i for i in items if stability.get(i, 0.0) < robust_thresh]
    return dict(robust=robust, fragile=fragile,
                stability={i: round(stability.get(i, 0.0), 3) for i in items})


# ---------------------------------------------------------------------------
# Top-level: reprogram a disease basin back to healthy
# ---------------------------------------------------------------------------
def disease_to_reversal(disease_active, healthy_active, W_in, order, out_deg, budget=8):
    """Given the disease and healthy ATTRACTORS (as active TF sets) of a signed regulatory network, return
    both the naive set-difference drivers (baseline) and the minimal Boolean-control reprogramming path.

    In real use, `disease_active` comes from localizing the phenotype (seed the network from the disease
    signature and settle); `healthy_active` is the wild-type / baseline attractor."""
    disease_state = {n: (1 if n in disease_active else -1) for n in order}
    setdiff = transition_setdiff(disease_active, healthy_active, out_deg, top=12)
    control = control_drivers(disease_state, healthy_active, W_in, order, budget=budget, out_deg=out_deg)
    for step in control["path"]:
        step["role"] = classify_intervention(step, out_deg, healthy_active)
    return dict(
        barrier=setdiff["barrier"],
        setdiff_drivers=setdiff["drivers"],                 # what the current attractor layer reports
        reprogramming_path=control["path"],                 # the minimal ordered intervention set
        reached_healthy=control["reached"],
        residual_distance=control["final_distance"],
        n_interventions=control["steps"],
        confidence="hypothesis: topological reprogramming route; wet-lab-testable, not a validated cure",
    )


# ---------------------------------------------------------------------------
# Adapter to the real NRxBW model (Colab). Gates gracefully when data is absent.
# ---------------------------------------------------------------------------
def load_cell_model(path=None):
    """Read cell_complete.json (NRxBW branch, Colab) -> the signed TF->TF regulatory core, matching
    compute_attractors.py's construction. Returns (W_in, out_deg, order, name) or None if absent."""
    cc = Path(path) if path else (OUT / "cell_complete.json")
    if not cc.exists():
        print(f"{cc} absent -> real-model reversal skipped (runs on Colab / NRxBW branch).")
        return None
    D = json.load(open(cc))
    G = D["genes"]
    tfset = set(i for i, g in enumerate(G) if g.get("tf"))
    edges = []
    for e in D.get("reg", []):
        a, b = e[0], e[1]
        s = e[2] if len(e) > 2 else 1
        if a in tfset and b in tfset:
            edges.append((a, b, 1 if s >= 0 else -1))
    W_in, out_deg, nodes = build_incoming(edges)
    order = sorted(nodes)
    name = {i: G[i]["name"] for i in nodes}
    return dict(W_in=W_in, out_deg=out_deg, order=order, name=name)


def main():
    """Tiny demo on a hand-built toggle so the module is runnable stand-alone; the real validation with
    aggregate numbers is validate_disease_to_reversal.py, and the real biology runs via load_cell_model()."""
    # mutual-repression toggle with self-activation: two attractors (A-high, B-high).
    edges = [("A", "A", 1), ("B", "B", 1), ("A", "B", -1), ("B", "A", -1),
             ("A", "x", 1), ("B", "y", 1)]               # x follows A, y follows B (downstream readouts)
    W_in, out_deg, nodes = build_incoming(edges)
    order = sorted(nodes)
    _, healthy = settle({"A": 1, "B": -1, "x": 1, "y": -1}, W_in, order)   # A-high basin
    _, disease = settle({"A": -1, "B": 1, "x": -1, "y": 1}, W_in, order)   # B-high basin
    res = disease_to_reversal(disease, healthy, W_in, order, out_deg)
    print("demo toggle — disease (B-high) -> healthy (A-high)")
    print("  healthy active:", sorted(healthy), "| disease active:", sorted(disease))
    print("  set-diff drivers (current attractor layer):", res["setdiff_drivers"], "barrier", res["barrier"])
    print("  control reprogramming path (minimal):",
          [(s["gene"], s["direction"]) for s in res["reprogramming_path"]],
          "| reached healthy:", res["reached_healthy"])


if __name__ == "__main__":
    main()
