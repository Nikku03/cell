"""FROM VERDICTS TO A SCHEME -- design, improve, upgrade.

WHAT WAS MISSING.  physics_reduce measures which reductions hold and what each is worth. That is a
diagnosis, and a diagnosis is not a plan. Three things have to happen after it before anyone can act:

    DESIGN    turn the verdicts into an ordered scheme with ONE composed speedup and ONE error budget,
              and say plainly what is not available and why.
    IMPROVE   turn every NO into a boundary. "Locality fails" is not actionable; "locality fails at R=10
              and holds at R=50, which costs 8x more per event" is.
    UPGRADE   given a scheme somebody is already running, say what it is leaving on the table.

THE ONE RULE THAT MAKES COMPOSITION HONEST.  Reductions do not multiply just because they both hold. They
multiply when they cut INDEPENDENT costs. Total work is

    (number of solves) x (cost of each solve) x (number of steps)

and a reduction attacks exactly one of those. Two reductions on the same axis are alternatives, not
partners -- taking the better one, not their product. This was learned by publishing a composed 1.09e15
that double-counted linearity against factorisation, both of which eliminate re-solving; the honest figure
on the same measurements was 5.48e12, a factor of 200,000 smaller.

THE ONE RULE THAT MAKES DESIGN SAFE.  Superposition is GATED on finite-amplitude linearity. The operator
being linear is not the question -- every Hessian is linear -- and a scheme that superposes precomputed
responses on a system whose PHYSICS is not linear is wrong at exactly the amplitudes anyone would care
about. So: if the finite-amplitude probe did not run, or declined, or failed, superposition is not
counted. It is listed as UNVERIFIED with the reason, which is a different thing from unavailable.

    That gate changes the nucleosome's headline. Superposition was measured there at the operator level
    and never at finite amplitude, because that system has no energy function wired up. The designer
    therefore refuses to count its 6.1e6x, and says why, rather than quoting a number whose supporting
    test was never run.

PREDECLARED, before any number:
    the designer's composed figure is BELOW the naive product of everything that holds
        -> the axis rule is doing work, which is the only reason to have it.
    the improver returns a boundary for every failed reduction
        -> the negatives become actionable, which was the point.
    the improver returns "no boundary exists" for some reduction
        -> more interesting than a boundary. It means the reduction is not merely unavailable here but
           unavailable at any setting, and the scheme has to be built without it.
    the designer counts superposition on a system where finite-amplitude linearity never ran
        -> the gate is broken and the whole file is unsafe.

-> outputs/physics_design.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics_reduce as PR  # noqa: E402
import physics_reduce_unknown as UNK  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 20260809
N_PT = int(os.environ.get("DESIGN_N", 1200))

# WHICH COST EACH REDUCTION ATTACKS. This table is the whole composition rule.
AXIS = {
    "linearity -> superposition":             "solves",
    "reusable factorisation":                 "solves",
    "locality (norm) -> truncate":            "dimensions",
    "modal truncation":                       "dimensions",
    "conserved directions (null space)":      "dimensions",
    "timescale separation -> event stepping": "steps",
    "finite-amplitude linearity":             "gate",      # buys nothing; licenses something
}
GATED_BY = {"linearity -> superposition": "finite-amplitude linearity"}


# ---- DESIGN ------------------------------------------------------------------------------------------
def design(results):
    """Turn verdicts into a scheme: one composed speedup, one error budget, and an explicit list of what
    is unavailable with the reason."""
    by = {r["name"]: r for r in results}
    used, blocked, unverified = [], [], []

    for r in results:
        ax = AXIS.get(r["name"])
        if ax in (None, "gate"):
            continue
        gate = GATED_BY.get(r["name"])
        if gate:
            g = by.get(gate)
            if g is None:
                unverified.append((r, f"{gate} never ran on this system, so the operator-level result "
                                      f"does not license superposition at finite amplitude"))
                continue
            if g["verdict"] != "HOLDS":
                blocked.append((r, f"{gate} is {g['verdict']} ({g['note'].split('|')[0].strip()})"))
                continue
        if r["verdict"] == "HOLDS":
            used.append(r)
        elif r["verdict"] == "MARGINAL":
            unverified.append((r, f"MARGINAL at {r['quantity']:.4g} against a {r['threshold']:.4g} "
                                  f"threshold -- available if you accept the margin"))
        else:
            blocked.append((r, f"{r['verdict']}: {r['note']}"))

    axes = {}
    for r in used:
        ax = AXIS[r["name"]]
        if ax not in axes or r["speedup"] > axes[ax]["speedup"]:
            axes[ax] = r
    composed = 1.0
    for a in axes.values():
        composed *= max(a["speedup"], 1.0)
    naive = 1.0
    for r in used:
        naive *= max(r["speedup"], 1.0)

    err = 0.0
    for r in used:
        q = r["quantity"]
        err += q if r["direction"] == "below" else (1.0 - q if 0 <= q <= 1 else 0.0)

    return {"used": [r["name"] for r in used],
            "per_axis": {k: {"name": v["name"], "speedup": v["speedup"]} for k, v in axes.items()},
            "composed": composed, "naive_product": naive,
            "error_budget": err,
            "blocked": [{"name": r["name"], "why": w} for r, w in blocked],
            "unverified": [{"name": r["name"], "why": w} for r, w in unverified]}


# ---- IMPROVE -----------------------------------------------------------------------------------------
def improve(S, results, rng):
    """For every reduction that did not hold, find the setting at which it WOULD -- or establish that
    none exists. A boundary is actionable; a NO is not."""
    by = {r["name"]: r for r in results}
    out = []

    r = by.get("locality (norm) -> truncate")
    if r and r["verdict"] != "HOLDS":
        rows = r["detail"].get("rows", [])
        hit = [x for x in rows if x["norm_in"] >= 0.9]
        if hit:
            best = min(hit, key=lambda x: x["frac"])
            out.append({"name": r["name"], "boundary": True,
                        "note": f"holds at R={best['R']:.0f} where {best['frac']:.0%} of atoms carry 90% "
                                f"of the norm, against the {r['threshold']:.0%} target -- "
                                f"{best['frac']/r['threshold']:.1f}x more work per event than a scheme "
                                f"that assumed locality"})
        else:
            top = rows[-1] if rows else {}
            out.append({"name": r["name"], "boundary": False,
                        "note": f"NO radius up to {top.get('R', 0):.0f} reaches 90% of the norm "
                                f"(largest tested holds {top.get('norm_in', 0):.2f}). Truncation by "
                                f"distance cannot be made to work here at any radius; the response has a "
                                f"floor spread over the whole system"})

    r = by.get("modal truncation")
    if r and r["verdict"] != "HOLDS":
        u = S.solve(S.couple(int(rng.integers(S.n)), rng))
        found = None
        for k in (60, 120, 240, 480):
            if k >= S.n3 - 2:
                break
            vals, vecs = eigsh(S.K.tocsc(), k=k, sigma=-1e-4, which="LM")
            o = np.argsort(np.abs(vals))
            vals, vecs = np.abs(vals[o]), vecs[:, o]
            B = vecs[:, vals > S.cut]
            cap = float(np.linalg.norm(B @ (B.T @ u)) / max(np.linalg.norm(u), 1e-300))
            if cap >= 0.9:
                found = (B.shape[1], cap, S.n3 / max(B.shape[1], 1))
                break
        if found:
            out.append({"name": r["name"], "boundary": True,
                        "note": f"holds at {found[0]} modes, capturing {found[1]:.1%}, still "
                                f"{found[2]:.0f}x fewer dimensions than the full problem"})
        else:
            out.append({"name": r["name"], "boundary": False,
                        "note": f"even 480 modes do not reach 90% capture. The response is not carried "
                                f"by a soft subspace here, so a modal scheme cannot be repaired by "
                                f"taking more modes"})

    r = by.get("finite-amplitude linearity")
    if r and r["verdict"] == "FAILS":
        devs, amp = r["detail"].get("devs", []), r["detail"].get("rel_amp", float("nan"))
        if devs and devs[0] < 1e-3:
            out.append({"name": r["name"], "boundary": True,
                        "note": f"deviation is {devs[0]:.1e} at 2x and {devs[-1]:.1e} at 8x, so "
                                f"superposition is usable up to roughly {2*amp:.3f} bond lengths and "
                                f"not beyond"})
        else:
            out.append({"name": r["name"], "boundary": False,
                        "note": f"already {devs[0]:.2f} off at the smallest amplitude tested "
                                f"({amp:.3f} bond lengths). There is no amplitude window where "
                                f"superposition is safe on this system"})
    elif r and r["verdict"] == "NOT TESTABLE":
        out.append({"name": r["name"], "boundary": False,
                    "note": f"the probe declined, so there is nothing to improve toward yet: {r['note']}"})

    r = by.get("reusable factorisation")
    if r and r["verdict"] != "HOLDS":
        out.append({"name": r["name"], "boundary": True,
                    "note": f"setup pays for itself after {max(r['quantity'], 1):.0f} solves; below that "
                            f"a direct solve per event is cheaper, which is a scheduling decision rather "
                            f"than a property of the system"})

    r = by.get("timescale separation -> event stepping")
    if r and r["verdict"] != "HOLDS":
        out.append({"name": r["name"], "boundary": False,
                    "note": f"separation is {r['quantity']:.3g} against the {r['threshold']:.3g} needed. "
                            f"This is a property of the spectrum and cannot be tuned -- to gain it you "
                            f"would have to change the model, e.g. by constraining the fast modes out"})

    r = by.get("conserved directions (null space)")
    if r and r["verdict"] in ("UNRESOLVABLE", "NOT TESTABLE"):
        out.append({"name": r["name"], "boundary": False,
                    "note": f"unresolvable at a noise floor of {S.noise:.1e} relative. An analytic "
                            f"Hessian reaches ~1e-16, so this is repairable by differentiating the "
                            f"energy by hand rather than by finite differences"})
    return out


# ---- UPGRADE -----------------------------------------------------------------------------------------
SCHEMES = {
    "direct solve per event": set(),
    "factorise once, solve per event": {"reusable factorisation"},
    "factorise + superpose": {"reusable factorisation", "linearity -> superposition"},
    "modal, factorised": {"reusable factorisation", "modal truncation"},
}


def upgrade(results, running):
    """Given the set of reductions a scheme already exploits, report what is measured as available and
    unused -- and what it is currently relying on that the measurements do not support."""
    plan = design(results)
    avail = set(plan["used"])
    unused = sorted(avail - running)
    unsupported = sorted(running - avail)
    by = {r["name"]: r for r in results}
    gain = 1.0
    axes_now = {AXIS[n] for n in running if n in AXIS}
    for n in unused:
        if AXIS.get(n) not in axes_now:
            gain *= max(by[n]["speedup"], 1.0)
    return {"running": sorted(running), "unused": unused, "unsupported": unsupported,
            "gain_available": gain}


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("FROM VERDICTS TO A SCHEME -- design, improve, upgrade")
    report("=" * 100)
    report("  Reductions multiply only when they cut INDEPENDENT costs: how many solves, how big each")
    report("  solve is, how many steps. Two on the same axis are alternatives, not partners.")
    report("  Superposition is GATED on finite-amplitude linearity, because every Hessian is linear and")
    report("  the operator-level result licenses nothing on its own.")

    co = UNK.cloud(rng, n=N_PT, box=7.0)
    K, npair = UNK.network(co, rng=rng)
    S = PR.System(co, K, PR.rigid_basis(co))
    S.setup()
    report(f"\n  system: {N_PT} points, {npair} couplings, null space {S.n_null}, "
           f"gap {S.gap:.2e}")

    results = [fn(S, rng, []) for fn in (PR.probe_linearity, PR.probe_precompute, PR.probe_locality)]
    fin = PR.probe_finite_linearity(S, rng, [])
    if fin is not None:
        results.append(fin)
    rm, vals, nz = PR.probe_modal(S, rng, [], k=40)
    results += [rm, PR.probe_nullspace(vals, nz, S), PR.probe_timescale(vals, S)]

    report(f"\n  MEASURED\n    {'reduction':<42}{'verdict':<14}{'quantity':>12}{'speedup':>12}")
    for r in results:
        report(f"    {r['name']:<42}{r['verdict']:<14}{r['quantity']:>12.4g}{r['speedup']:>12.4g}")

    plan = design(results)
    report("\n  DESIGNED SCHEME")
    for ax in ("solves", "dimensions", "steps"):
        a = plan["per_axis"].get(ax)
        report(f"    {ax:<14}{a['name'] + '  x' + format(a['speedup'], '.4g') if a else 'nothing available'}")
    report(f"    composed speedup     {plan['composed']:.4g}")
    report(f"    naive product        {plan['naive_product']:.4g}  "
           f"({plan['naive_product']/max(plan['composed'],1e-300):.4g}x optimistic)")
    report(f"    error budget         {plan['error_budget']:.3e}")
    if plan["blocked"]:
        report("    NOT AVAILABLE")
        for b in plan["blocked"]:
            report(f"      {b['name']}: {b['why'][:110]}")
    if plan["unverified"]:
        report("    NOT COUNTED, and why")
        for b in plan["unverified"]:
            report(f"      {b['name']}: {b['why'][:110]}")

    report("\n  IMPROVER -- every NO turned into a boundary, or shown not to have one")
    imp = improve(S, results, rng)
    for i in imp:
        report(f"    {i['name']:<42}{'BOUNDARY' if i['boundary'] else 'NO SETTING WORKS'}")
        for line in _wrap(i["note"], 92):
            report(f"      {line}")
    if not imp:
        report("    nothing failed, so there is nothing to improve")

    report("\n  UPGRADER -- what existing schemes are leaving on the table")
    report(f"    {'scheme in use':<34}{'unused':<46}{'gain':>10}")
    ups = {}
    for nm, using in SCHEMES.items():
        u = upgrade(results, using)
        ups[nm] = u
        report(f"    {nm:<34}{', '.join(u['unused'])[:44] or '--':<46}{u['gain_available']:>10.4g}")
        if u["unsupported"]:
            report(f"      RELIES ON UNSUPPORTED: {', '.join(u['unsupported'])}")

    report("\n  READING")
    report(f"    The axis rule removed a factor of "
           f"{plan['naive_product']/max(plan['composed'],1e-300):.4g} of double counting.")
    nb = [i for i in imp if not i["boundary"]]
    report(f"    {len(imp)-len(nb)} of {len(imp)} failed reductions have a setting where they hold; "
           f"{len(nb)} have none at any setting.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "physics_design", "n_pt": N_PT, "reductions": results, "plan": plan,
               "improvements": imp, "upgrades": ups, "log": log},
              open(OUT / "physics_design.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'physics_design.json'}")
    return 0


def _wrap(s, w):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > w:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
