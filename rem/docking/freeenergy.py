"""Algorithm 4 -- REM-Z: binding free energy from the SUM semiring on the repack graph.

THE POINT, in one sentence: Algorithm 2 and Algorithm 4 run on the SAME factor graph and
differ only in the semiring, so anything true of the elimination cost of one is true of
the other -- cost = d ** treewidth either way.

    eliminate("min")   ->   min_x  sum_f phi_f(x)          the best single conformation
    eliminate("sum")   ->   log sum_x exp( sum_f phi_f(x) )    the whole ensemble

Feed the graph  phi = -E / RT  and the second one IS  log Z, exactly, with no sampling and
no convergence criterion. That is the actual difference from the usual practice: molecular
docking estimates conformational free energy by Monte Carlo or by a rigid-rotor entropy
approximation, and both are estimates. Here the sum over all d^n rotamer configurations is
exact and the error bar is floating point.

WHAT IT IS NOT. This is the CONFORMATIONAL free energy of a fixed rotamer library under a
fixed force field. It is not a solvated binding free energy: there is no explicit water, no
Poisson-Boltzmann term, no configurational integral over backbone or over continuous chi.
A number from this module can be exactly right about the model and still wrong about the
molecule. Every claim below is a claim about the model.

THE THERMODYNAMICS THAT MUST HOLD, and each one is a gate.
    F = -RT ln Z  <=  E_min                (Z >= exp(-E_min/RT), always)
    T*S_conf = E_min - F  >=  0            the entropy bonus is never negative
    T -> 0    =>  F -> E_min               the ensemble collapses onto the ground state
    T -> inf  =>  ln Z -> ln(n_configs)    every configuration equally weighted

Those are not stylistic checks. Each one is a place where a sign error, a missing RT, or a
log-sum-exp overflow would show up as a violated inequality rather than as a plausible
wrong number.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  Z1  ln Z by elimination vs ln Z by explicit enumeration of every configuration.
      GATE: |difference| < 1e-8.
  Z2  F <= E_min, i.e. T*S_conf >= 0, on every instance tested.
      GATE: min over instances of (E_min - F) >= -1e-9.
  Z3  low-temperature limit at T = 1 K.  GATE: |F - E_min| < 0.01 kcal/mol.
  Z4  high-temperature limit at T = 1e6 K.  GATE: |ln Z - ln(n_configs)| < 1e-3.
      THIS GATE FAILED, at 3.87e-2, and the verdict stands -- see ledger defect L in
      colab/gate_guard.py. The code was right and the BAR was wrong: the residual of that
      limit is exactly -<E>/RT, and <E> here is 77.5 kcal/mol, so 0.039 at 1e6 K was the
      smallest value physically available. The bar was written before <E> was known.
  Z4b THE REPAIR, declared separately rather than by moving Z4's bar. A limit is a
      statement about a RATE, so gate the rate: sweep T over four decades and fit the
      log-log slope of |ln Z - ln N| against T. GATE: slope within 0.05 of -1, AND the
      measured residual within 1% of the predicted -<E>/RT at every temperature. This is
      strictly stronger than the point bar -- code converging to the WRONG constant passes
      a loose point bar and fails the slope.
  Z5  per-residue rotamer marginals by elimination vs by enumeration.
      GATE: max |difference| < 1e-8, and every marginal sums to 1.
  Z6  POSITIVE CONTROL. A deliberate steric clash is introduced by translating the mobile
      chain 1.5 A into its partner. GATE: BOTH E_min and F must RISE. A pipeline that
      cannot see a clash it was handed cannot be trusted to see one it was not.
  Z7  Z-rescoring vs single-pose scoring, REPORTED not gated: how far apart are the two
      rankings? If they agree everywhere, the ensemble bought nothing and this module
      says so rather than implying it helped.
      THE FIRST VERSION OF Z7 WAS BROKEN and is recorded as ledger defect K. It ranked
      five problems of 3, 4, 5, 6 and 7 residues; adding a residue adds a large negative
      unary term, so both orderings were the residue count sorted descending and the test
      could not have come out any other way. It reported "identical ordering" and had
      measured nothing. The corrected Z7 ranks EXCHANGEABLE items: N decoy poses of the
      SAME residue set, same variables, same rotamers, differing only in the rigid-body
      pose being scored. Then a reordering is information.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.factorgraph import FactorGraph
from rem.docking.data import Structure
from rem.docking.repack import RepackProblem, build_from_case

R_GAS = 0.0019872041          # kcal / (mol K)
T_ROOM = 298.15               # K


def rt(temperature: float = T_ROOM) -> float:
    return R_GAS * float(temperature)


def boltzmann_graph(problem: RepackProblem, temperature: float = T_ROOM,
                    energy_graph: Optional[FactorGraph] = None) -> FactorGraph:
    """Turn the ENERGY graph into a LOG-WEIGHT graph: phi = -E / RT.

    Same variables, same factors, same treewidth -- only the tables are rescaled. So
    eliminate("sum") on the result is exactly log Z, and it costs what eliminate("min")
    on the energy graph costs.
    """
    if energy_graph is None:
        energy_graph, _ = problem.to_factorgraph()
    beta = 1.0 / rt(temperature)
    g = FactorGraph()
    for v, c in energy_graph.cards.items():
        g.add_var(v, c)
    for f in energy_graph.factors:
        g.add_factor(list(f.vars), -beta * np.asarray(f.table))
    return g


def log_partition(g_boltz: FactorGraph, order: Optional[Sequence[str]] = None
                  ) -> Tuple[float, dict]:
    val, _arg, info = g_boltz.eliminate("sum", order=order)
    return float(val), info


def free_energy(problem: RepackProblem, temperature: float = T_ROOM,
                energy_graph: Optional[FactorGraph] = None) -> dict:
    """F = -RT ln Z, plus E_min and the conformational entropy that separates them."""
    if energy_graph is None:
        energy_graph, _ = problem.to_factorgraph()
    gb = boltzmann_graph(problem, temperature, energy_graph)
    t0 = time.perf_counter()
    lnZ, info = log_partition(gb)
    z_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    emin, arg, _minfo = energy_graph.eliminate("min")
    min_sec = time.perf_counter() - t0
    F = -rt(temperature) * lnZ
    n_conf = float(np.prod([float(c) for c in energy_graph.cards.values()]))
    return {"logZ": lnZ, "F": F, "E_min": float(emin), "TS_conf": float(emin) - F,
            "assignment": arg, "temperature": float(temperature),
            "treewidth": info["treewidth"], "n_configs": n_conf,
            "seconds_Z": z_sec, "seconds_min": min_sec,
            "ln_n_configs": float(np.log(n_conf))}


def rotamer_probabilities(problem: RepackProblem, temperature: float = T_ROOM,
                          energy_graph: Optional[FactorGraph] = None
                          ) -> Dict[str, np.ndarray]:
    """P(residue takes rotamer k) marginalized over the whole ensemble, exactly."""
    gb = boltzmann_graph(problem, temperature, energy_graph)
    return gb.marginals()


def binding_free_energy(case: Dict[str, Structure], side: str = "r", bound: bool = True,
                        max_residues: int = 10, n_chi1: int = 3, n_chi2: int = 2,
                        temperature: float = T_ROOM) -> dict:
    """F(complex) - F(isolated chain): the side-chain conformational part of binding.

    The two states share EVERY variable and EVERY rotamer; they differ only in whether
    the partner contributes to the unary term. So the difference is attributable to the
    partner and to nothing else -- the control moves exactly one thing.
    """
    prob_c = build_from_case(case, side=side, bound=bound, max_residues=max_residues,
                             n_chi1=n_chi1, n_chi2=n_chi2)
    empty = Structure(np.zeros((0, 3)), np.array([], dtype=object),
                      np.array([], dtype=object), np.array([], dtype=object),
                      np.array([], dtype=object))
    prob_u = RepackProblem(prob_c.mobile, empty, prob_c.res_keys,
                           n_chi1=n_chi1, n_chi2=n_chi2, cutoff=prob_c.cutoff)
    fc = free_energy(prob_c, temperature)
    fu = free_energy(prob_u, temperature)
    return {"F_complex": fc["F"], "F_isolated": fu["F"], "dF": fc["F"] - fu["F"],
            "Emin_complex": fc["E_min"], "Emin_isolated": fu["E_min"],
            "dEmin": fc["E_min"] - fu["E_min"],
            "dTS_conf": fc["TS_conf"] - fu["TS_conf"],
            "treewidth": fc["treewidth"], "n_residues": len(prob_c.res_keys),
            "n_configs": fc["n_configs"]}


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def _mean_energy(g: FactorGraph) -> float:
    """<E> over ALL configurations, by explicit enumeration. Small graphs only; this is a
    reference quantity for Z4b, so it must not share a code path with the elimination."""
    import itertools
    names = list(g.cards)
    tot, n = 0.0, 0
    for combo in itertools.product(*[range(g.cards[v]) for v in names]):
        a = dict(zip(names, combo))
        tot += sum(f.table[tuple(a[v] for v in f.vars)] for f in g.factors)
        n += 1
    return tot / n


def verify(case_id: str = "1A2K", verbose: bool = True) -> dict:
    """Run Z1-Z7. Bars are fixed in the module docstring, above, before any number."""
    from rem.docking.data import load_case
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    out: Dict[str, object] = {"case": case_id}
    case = load_case(case_id)

    # A small instance, so brute force over every configuration is an affordable reference.
    small = build_from_case(case, side="r", bound=True, max_residues=5,
                            n_chi1=3, n_chi2=2)
    g_small, edges = small.to_factorgraph()
    n_conf = float(np.prod([float(c) for c in g_small.cards.values()]))
    say(f"  case {case_id}: {len(small.res_keys)} residues, "
        f"cards {[g_small.cards[r] for r in small.res_keys]}, "
        f"{len(edges)} contact edges, {n_conf:,.0f} configurations")

    # ---- Z1: ln Z, elimination vs enumeration --------------------------------------------
    gb = boltzmann_graph(small, T_ROOM, g_small)
    lnZ_elim, info = log_partition(gb)
    lnZ_bf, _ = gb.brute_force("sum")
    d1 = abs(lnZ_elim - lnZ_bf)
    out["Z1_err"], out["Z1"] = float(d1), d1 < 1e-8
    say(f"\n  Z1 ln Z   elimination {lnZ_elim:.10f}   enumeration {lnZ_bf:.10f}")
    say(f"      |diff| {d1:.3e}   treewidth {info['treewidth']}   "
        f"{'PASS' if out['Z1'] else 'FAIL'}")

    # ---- Z2: F <= E_min on several instances ----------------------------------------------
    say("\n  Z2 F <= E_min, i.e. the entropy bonus T*S_conf is never negative")
    rows, worst = [], np.inf
    for nres in (3, 4, 5, 6, 7):
        p = build_from_case(case, side="r", bound=True, max_residues=nres,
                            n_chi1=3, n_chi2=2)
        fe = free_energy(p, T_ROOM)
        worst = min(worst, fe["TS_conf"])
        rows.append((nres, fe["E_min"], fe["F"], fe["TS_conf"], fe["treewidth"]))
        say(f"      n={nres}  E_min {fe['E_min']:10.4f}  F {fe['F']:10.4f}  "
            f"T*S_conf {fe['TS_conf']:8.4f}  tw {fe['treewidth']}")
    out["Z2_min_TS"], out["Z2"] = float(worst), worst >= -1e-9
    say(f"      min T*S_conf over instances {worst:.6f}   "
        f"{'PASS' if out['Z2'] else 'FAIL'}")

    # ---- Z3 / Z4: the two temperature limits ------------------------------------------------
    cold = free_energy(small, 1.0, g_small)
    hot = free_energy(small, 1.0e6, g_small)
    d3 = abs(cold["F"] - cold["E_min"])
    d4 = abs(hot["logZ"] - np.log(n_conf))
    out["Z3_err"], out["Z3"] = float(d3), d3 < 0.01
    out["Z4_err"], out["Z4"] = float(d4), d4 < 1e-3
    say(f"\n  Z3 T=1 K      F {cold['F']:.6f}  E_min {cold['E_min']:.6f}  "
        f"|diff| {d3:.3e}   {'PASS' if out['Z3'] else 'FAIL'}")
    say(f"  Z4 T=1e6 K    ln Z {hot['logZ']:.8f}  ln(n_configs) {np.log(n_conf):.8f}  "
        f"|diff| {d4:.3e}   {'PASS' if out['Z4'] else 'FAIL'}")

    # ---- Z4b: THE REPAIR. A limit is a rate, so gate the rate. ---------------------------
    say("\n  Z4b high-T limit as a RATE (Z4's point bar failed; see ledger defect L)")
    Ts = np.array([1e5, 1e6, 1e7, 1e8])
    Ebar = _mean_energy(g_small)
    resid, pred_rows = [], []
    for T in Ts:
        fe = free_energy(small, float(T), g_small)
        r = fe["logZ"] - np.log(n_conf)
        pred = -Ebar / rt(float(T))
        resid.append(abs(r))
        pred_rows.append((T, r, pred, abs(r - pred) / max(abs(pred), 1e-30)))
        say(f"      T={T:8.0e}  lnZ-lnN {r:11.4e}   -<E>/RT {pred:11.4e}   "
            f"rel dev {pred_rows[-1][3]:.3%}")
    slope = float(np.polyfit(np.log10(Ts), np.log10(np.array(resid)), 1)[0])
    worst_dev = max(r[3] for r in pred_rows)
    out["Z4b_slope"], out["Z4b_worst_dev"] = slope, float(worst_dev)
    out["Z4b"] = abs(slope + 1.0) < 0.05 and worst_dev < 0.01
    say(f"      <E> over all {n_conf:,.0f} configurations: {Ebar:.4f} kcal/mol")
    say(f"      log-log slope {slope:.4f} (theory -1, bar +/-0.05); worst deviation from "
        f"-<E>/RT {worst_dev:.3%} (bar 1%)   {'PASS' if out['Z4b'] else 'FAIL'}")

    # ---- Z5: marginals ---------------------------------------------------------------------
    m_elim = gb.marginals()
    m_bf = gb.brute_force_marginals()
    d5 = max(float(np.abs(m_elim[v] - m_bf[v]).max()) for v in m_elim)
    sums = [float(m_elim[v].sum()) for v in m_elim]
    out["Z5_err"], out["Z5"] = float(d5), d5 < 1e-8 and max(abs(s - 1) for s in sums) < 1e-9
    say(f"\n  Z5 rotamer marginals, elimination vs enumeration: max |diff| {d5:.3e}   "
        f"{'PASS' if out['Z5'] else 'FAIL'}")
    for v in list(m_elim)[:3]:
        say(f"      {v:>12}  " + " ".join(f"{x:.4f}" for x in m_elim[v]))

    # ---- Z6: POSITIVE CONTROL -- a clash the pipeline must see ------------------------------
    base = free_energy(small, T_ROOM, g_small)
    push = small.fixed.coords.mean(axis=0) - small.mobile.coords.mean(axis=0)
    push = 1.5 * push / max(np.linalg.norm(push), 1e-9)
    clashed_mobile = Structure(small.mobile.coords + push, small.mobile.atom_names,
                               small.mobile.res_ids, small.mobile.res_names,
                               small.mobile.elements)
    p_clash = RepackProblem(clashed_mobile, small.fixed, small.res_keys,
                            n_chi1=3, n_chi2=2, cutoff=small.cutoff)
    fe_clash = free_energy(p_clash, T_ROOM)
    rose_E = fe_clash["E_min"] > base["E_min"]
    rose_F = fe_clash["F"] > base["F"]
    out["Z6"], out["Z6_dE"], out["Z6_dF"] = bool(rose_E and rose_F), \
        float(fe_clash["E_min"] - base["E_min"]), float(fe_clash["F"] - base["F"])
    say(f"\n  Z6 POSITIVE CONTROL, mobile chain pushed 1.5 A into its partner")
    say(f"      E_min {base['E_min']:9.4f} -> {fe_clash['E_min']:9.4f}  "
        f"(rose: {rose_E})")
    say(f"      F     {base['F']:9.4f} -> {fe_clash['F']:9.4f}  (rose: {rose_F})")
    say(f"      Z6 {'PASS' if out['Z6'] else 'FAIL'}")

    # ---- Z7: ranking over EXCHANGEABLE items (corrected; see ledger defect K) ------------
    say("\n  Z7 does Z-rescoring reorder DECOY POSES of the same residue set? "
        "(reported, not gated)")
    say("      the first version of this gate ranked problems of different SIZES and could "
        "not fail; see ledger defect K")
    rng = np.random.default_rng(0)
    rows = []
    for k in range(8):
        if k == 0:
            mob = small.mobile
            tag = "native"
        else:
            d = rng.normal(size=3)
            d = 1.2 * d / np.linalg.norm(d)
            mob = Structure(small.mobile.coords + d, small.mobile.atom_names,
                            small.mobile.res_ids, small.mobile.res_names,
                            small.mobile.elements)
            tag = f"decoy{k}"
        pk = RepackProblem(mob, small.fixed, small.res_keys, n_chi1=3, n_chi2=2,
                           cutoff=small.cutoff)
        fe = free_energy(pk, T_ROOM)
        rows.append((tag, fe["E_min"], fe["F"], fe["TS_conf"]))
    by_e = [r[0] for r in sorted(rows, key=lambda r: r[1])]
    by_f = [r[0] for r in sorted(rows, key=lambda r: r[2])]
    same = by_e == by_f
    n_disagree = sum(1 for a, b in zip(by_e, by_f) if a != b)
    ts = [r[3] for r in rows]
    out["Z7_same_order"] = bool(same)
    out["Z7_n_positions_differing"] = int(n_disagree)
    out["Z7_TS_spread"] = float(max(ts) - min(ts))
    out["Z7_native_rank_E"] = by_e.index("native") + 1
    out["Z7_native_rank_F"] = by_f.index("native") + 1
    for tag, e, f, t in rows:
        say(f"      {tag:8s}  E_min {e:9.4f}   F {f:9.4f}   T*S_conf {t:7.4f}")
    say(f"      by E_min: {by_e}")
    say(f"      by F    : {by_f}")
    say(f"      identical ordering: {same}  ({n_disagree} of {len(rows)} positions differ)")
    say(f"      native ranks {out['Z7_native_rank_E']} by E_min, "
        f"{out['Z7_native_rank_F']} by F, of {len(rows)}")
    say(f"      T*S_conf spread across poses {max(ts) - min(ts):.4f} kcal/mol")
    if same:
        say("      -> the ensemble did not reorder these poses. The entropy term is real "
            "but varies less between poses than the energy does, so it cancels in a rank.")

    gates = ["Z1", "Z2", "Z3", "Z4", "Z4b", "Z5", "Z6"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    verify()
