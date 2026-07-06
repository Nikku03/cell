"""VALIDATE disease_to_reversal — honest, self-contained, no external data or deps.

The real biological accuracy of reprogramming on the 16k-gene model needs cell_complete.json (Colab). What
we CAN test here, rigorously and with real numbers, is the thing everything rests on: given a signed
regulatory network with a KNOWN minimal driver set separating a disease basin from a healthy basin, does
the Boolean target-control search recover THOSE drivers (and only those), and reach the healthy state?

We build modular GRNs where the ground truth is known by construction: M master TFs each strictly control a
downstream module (self-activating master -> feed-forward fan). A "disease" flips a subset D of masters;
flipping exactly D reprograms disease->healthy, and D is minimal (each master independently gates its
module; downstream TFs cannot fix an upstream master). Crosstalk + noise edges are added so the naive
"report every differing TF" baseline is polluted by passenger TFs while the true drivers stay = D.

Tests:
  1. Driver recovery over many random GRNs — control (this module) vs set-diff (current attractor layer).
  2. Textbook Boolean motifs — attractor recovery is correct (sanity that the dynamics are right).
  3. Single-gene "mutation" disease — the canonical case: recover the 1 driver, 1-step cure.
  4. Noise robustness — how recovery degrades as crosstalk/noise edges are added.

-> prints a summary and writes outputs/orphan/disease_to_reversal_validation.json
"""
import json, random
from pathlib import Path
from statistics import mean, pstdev

from disease_to_reversal import (build_incoming, settle, active, hamming,
                                  transition_setdiff, control_drivers, disease_to_reversal)

OUT = Path("outputs/orphan")


# ---------------------------------------------------------------------------
# Ground-truth GRN generator: master TFs each gate a downstream module.
# ---------------------------------------------------------------------------
def make_modular_grn(M, msize, cross, noise, rng):
    """M masters, each self-activating and driving a feed-forward module of `msize` TFs. `cross` = random
    inter-module edges (crosstalk), `noise` = random edges (both should be irrelevant as drivers).
    Returns (edges, masters, modules{k:[genes]})."""
    edges = []
    masters = [f"M{k}" for k in range(M)]
    modules = {}
    for k, mk in enumerate(masters):
        edges.append((mk, mk, 1))                      # self-activation -> bistable master
        mod = [f"g{k}_{j}" for j in range(msize)]
        modules[k] = mod
        hub = mod[0]
        for g in mod:
            edges.append((mk, g, 1))                    # master drives every module TF
        for g in mod[1:]:
            edges.append((hub, g, 1))                   # + intra-module reinforcement (robust following)
    allmod = [g for k in modules for g in modules[k]]
    for _ in range(cross):
        a, b = rng.choice(allmod), rng.choice(allmod)
        if a != b:
            edges.append((a, b, rng.choice([1, -1])))
    for _ in range(noise):
        a, b = rng.choice(allmod), rng.choice(allmod)
        if a != b:
            edges.append((a, b, rng.choice([1, -1])))
    return edges, masters, modules


def seed_from_config(sigma, masters, modules):
    """sigma: {k: +1/-1} master config. Seed masters to sigma, downstream to -1; the attractor follows."""
    s = {}
    for k, mk in enumerate(masters):
        s[mk] = sigma[k]
        for g in modules[k]:
            s[g] = -1
    return s


def instance(M, msize, cross, noise, n_flip, rng):
    """One disease/healthy pair with known drivers. Returns a dict of the settled basins + ground truth."""
    edges, masters, modules = make_modular_grn(M, msize, cross, noise, rng)
    W_in, out_deg, nodes = build_incoming(edges)
    order = sorted(nodes)
    sigma_h = {k: rng.choice([1, -1]) for k in range(M)}          # healthy master config
    flip = rng.sample(range(M), n_flip)                          # the "mutations"
    sigma_d = dict(sigma_h)
    for k in flip:
        sigma_d[k] = -sigma_d[k]
    _, healthy = settle(seed_from_config(sigma_h, masters, modules), W_in, order)
    _, disease = settle(seed_from_config(sigma_d, masters, modules), W_in, order)
    drivers_true = {masters[k] for k in flip}
    return dict(W_in=W_in, out_deg=out_deg, order=order, healthy=healthy, disease=disease,
                drivers_true=drivers_true, masters=set(masters))


def score(inst, budget=None):
    """Run both methods on one instance -> per-instance metrics."""
    D = inst["drivers_true"]
    budget = budget or (len(D) + 3)
    # stable basins?  (are the two seeded attractors genuine fixed points that differ)
    basins_ok = inst["disease"] != inst["healthy"]
    # BASELINE: set-difference drivers (uncapped) — precision vs the true masters
    sd = transition_setdiff(inst["disease"], inst["healthy"], inst["out_deg"], top=None)
    sd_drivers = set(sd["drivers"])
    sd_prec = len(sd_drivers & D) / len(sd_drivers) if sd_drivers else 0.0
    # CONTROL: minimal reprogramming path
    dstate = {n: (1 if n in inst["disease"] else -1) for n in inst["order"]}
    ctrl = control_drivers(dstate, inst["healthy"], inst["W_in"], inst["order"],
                           budget=budget, out_deg=inst["out_deg"])
    rec = {s["gene"] for s in ctrl["path"]}
    recall = len(rec & D) / len(D) if D else 1.0
    prec = len(rec & D) / len(rec) if rec else 0.0
    return dict(basins_ok=basins_ok, barrier=sd["barrier"], setdiff_precision=sd_prec,
                control_recall=recall, control_precision=prec, reached=ctrl["reached"],
                n_true=len(D), n_recovered=len(rec), path_len=ctrl["steps"])


# ---------------------------------------------------------------------------
# Test 1: driver recovery over many random GRNs
# ---------------------------------------------------------------------------
def test_driver_recovery(n=60, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        M = rng.choice([6, 8, 10, 12])
        msize = rng.choice([3, 4, 5])
        cross = rng.randint(2, 8)
        noise = rng.randint(2, 8)
        n_flip = rng.randint(1, max(2, M // 3))
        rows.append(score(instance(M, msize, cross, noise, n_flip, rng)))
    agg = lambda key: round(mean(r[key] for r in rows), 3)
    aggsd = lambda key: round(pstdev(r[key] for r in rows), 3)
    reached_rate = round(mean(1.0 if r["reached"] else 0.0 for r in rows), 3)
    exact = round(mean(1.0 if (r["control_recall"] == 1.0 and r["control_precision"] == 1.0) else 0.0
                       for r in rows), 3)
    return dict(
        n_instances=n, basins_valid=all(r["basins_ok"] for r in rows),
        control_recall=agg("control_recall"), control_recall_sd=aggsd("control_recall"),
        control_precision=agg("control_precision"), control_precision_sd=aggsd("control_precision"),
        setdiff_precision=agg("setdiff_precision"),
        reached_healthy_rate=reached_rate,
        exact_driver_set_rate=exact,
        mean_barrier=agg("barrier"), mean_true_drivers=agg("n_true"), mean_path_len=agg("path_len"),
    )


# ---------------------------------------------------------------------------
# Test 2: textbook motifs -> attractor recovery
# ---------------------------------------------------------------------------
def test_motifs():
    results = {}
    # (a) mutual-repression toggle + self-activation -> exactly two attractors {A} and {B}
    edges = [("A", "A", 1), ("B", "B", 1), ("A", "B", -1), ("B", "A", -1)]
    W, deg, nodes = build_incoming(edges); order = sorted(nodes)
    found = set()
    for a in (1, -1):
        for b in (1, -1):
            _, act = settle({"A": a, "B": b}, W, order)
            found.add(act)
    results["toggle"] = dict(attractors=sorted(sorted(a) for a in found),
                             ok=({frozenset(["A"]), frozenset(["B"])} <= found))
    # (b) 3-gene mutual repression (lateral inhibition) -> three single-winner attractors
    edges = [(x, x, 1) for x in "ABC"] + [(a, b, -1) for a in "ABC" for b in "ABC" if a != b]
    W, deg, nodes = build_incoming(edges); order = sorted(nodes)
    found = set()
    for seed in ({"A": 1, "B": -1, "C": -1}, {"A": -1, "B": 1, "C": -1}, {"A": -1, "B": -1, "C": 1}):
        _, act = settle(seed, W, order)
        found.add(act)
    results["lateral_inhibition_3"] = dict(
        attractors=sorted(sorted(a) for a in found),
        ok=({frozenset(["A"]), frozenset(["B"]), frozenset(["C"])} <= found))
    # (c) linear cascade M -> a -> b -> c : the master's state propagates to a single attractor per master state
    edges = [("M", "M", 1), ("M", "a", 1), ("a", "b", 1), ("b", "c", 1)]
    W, deg, nodes = build_incoming(edges); order = sorted(nodes)
    _, on = settle({"M": 1, "a": -1, "b": -1, "c": -1}, W, order)
    _, off = settle({"M": -1, "a": 1, "b": 1, "c": 1}, W, order)
    results["cascade"] = dict(master_on=sorted(on), master_off=sorted(off),
                              ok=(on == frozenset(["M", "a", "b", "c"]) and off == frozenset()))
    results["all_ok"] = all(v["ok"] for v in results.values() if isinstance(v, dict))
    return results


# ---------------------------------------------------------------------------
# Test 3: single-gene "mutation" disease -> recover the one driver, one-step cure
# ---------------------------------------------------------------------------
def test_single_mutation(n=50, seed=1):
    rng = random.Random(seed)
    ok = 0; onestep = 0; rows = []
    for _ in range(n):
        M = rng.choice([6, 8, 10]); msize = rng.choice([3, 4, 5])
        inst = instance(M, msize, cross=rng.randint(2, 6), noise=rng.randint(2, 6), n_flip=1, rng=rng)
        m = score(inst, budget=4)
        rows.append(m)
        if m["control_recall"] == 1.0 and m["control_precision"] == 1.0 and m["reached"]:
            ok += 1
            if m["path_len"] == 1:
                onestep += 1
    return dict(n=n, exact_and_reached_rate=round(ok / n, 3), one_step_cure_rate=round(onestep / n, 3),
                mean_control_precision=round(mean(r["control_precision"] for r in rows), 3))


# ---------------------------------------------------------------------------
# Test 4: noise robustness sweep
# ---------------------------------------------------------------------------
def test_noise_sweep(seed=2):
    sweep = {}
    for extra in (0, 4, 8, 16, 32):
        rng = random.Random(seed + extra)
        rows = []
        for _ in range(40):
            M = 8; msize = 4; n_flip = rng.randint(1, 3)
            rows.append(score(instance(M, msize, cross=extra, noise=extra, n_flip=n_flip, rng=rng)))
        sweep[f"cross+noise={2*extra}"] = dict(
            control_recall=round(mean(r["control_recall"] for r in rows), 3),
            control_precision=round(mean(r["control_precision"] for r in rows), 3),
            reached_rate=round(mean(1.0 if r["reached"] else 0.0 for r in rows), 3))
    return sweep


def main():
    print("=" * 78)
    print("VALIDATE disease_to_reversal — Boolean target-control on GRNs with KNOWN drivers")
    print("=" * 78)

    t1 = test_driver_recovery()
    print("\n[1] Driver recovery over", t1["n_instances"], "random modular GRNs (ground-truth known):")
    print(f"    basins are valid distinct attractors : {t1['basins_valid']}")
    print(f"    CONTROL   recall    = {t1['control_recall']} +/- {t1['control_recall_sd']}")
    print(f"    CONTROL   precision = {t1['control_precision']} +/- {t1['control_precision_sd']}")
    print(f"    set-diff  precision = {t1['setdiff_precision']}   <- current attractor-layer baseline")
    print(f"    reached healthy state        : {t1['reached_healthy_rate']}")
    print(f"    EXACT driver set (rec=prec=1): {t1['exact_driver_set_rate']}")
    print(f"    mean barrier {t1['mean_barrier']} | mean true drivers {t1['mean_true_drivers']} "
          f"| mean path length {t1['mean_path_len']}")

    t2 = test_motifs()
    print("\n[2] Textbook Boolean motifs — attractor recovery correct:", t2["all_ok"])
    print(f"    toggle -> {t2['toggle']['attractors']} (ok {t2['toggle']['ok']})")
    print(f"    3-gene lateral inhibition -> {t2['lateral_inhibition_3']['attractors']} "
          f"(ok {t2['lateral_inhibition_3']['ok']})")
    print(f"    cascade M-on -> {t2['cascade']['master_on']} ; M-off -> {t2['cascade']['master_off']} "
          f"(ok {t2['cascade']['ok']})")

    t3 = test_single_mutation()
    print("\n[3] Single-gene 'mutation' disease (canonical case),", t3["n"], "instances:")
    print(f"    exact driver recovered + healthy reached : {t3['exact_and_reached_rate']}")
    print(f"    achieved in ONE intervention             : {t3['one_step_cure_rate']}")

    t4 = test_noise_sweep()
    print("\n[4] Noise robustness (control recall / precision / reached vs crosstalk+noise edges):")
    for k, v in t4.items():
        print(f"    {k:18}: recall {v['control_recall']} | precision {v['control_precision']} "
              f"| reached {v['reached_rate']}")

    payload = dict(driver_recovery=t1, motifs=t2, single_mutation=t3, noise_sweep=t4)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT / "disease_to_reversal_validation.json", "w"), indent=2)
    print("\n-> outputs/orphan/disease_to_reversal_validation.json")
    return payload


if __name__ == "__main__":
    main()
