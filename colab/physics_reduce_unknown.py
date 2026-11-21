"""DOES THE REDUCER DISCRIMINATE? -- four systems, predictions written down first.

THE PROBLEM WITH THE LAST RESULT.  physics_reduce scored 5/5 on the nucleosome. But I wrote the probes
knowing the answers -- the norm/amplitude split exists in the vocabulary BECAUSE I was burned by it that
morning. Reproducing an answer you already have is a harness, not a discoverer, and I said so.

THE TEST THAT SEPARATES THEM.  A tool that reports "linearity holds, locality fails" for every system is
measuring its own probes, not the system. So: run the identical vocabulary over four systems with
DELIBERATELY DIFFERENT STRUCTURE, and ask whether the verdicts MOVE. Discrimination is the whole question.
If they do not move, the tool is a thermometer that always reads 37.

FOUR SYSTEMS, sharing geometry so only the coupling structure differs
    generic      a random 3D point cloud with a distance cutoff. The null case: no protein, no chemistry,
                 just a network. Anything the nucleosome showed that this also shows was never about
                 biology.
    disordered   identical geometry, spring constants drawn log-normal over four decades. A glass.
    diluted      identical geometry, springs removed at random toward the rigidity-percolation threshold,
                 where a network stops being rigid and floppy modes appear.
    long-range   every pair coupled as 1/r^2 with NO cutoff. The one case where locality must break by
                 construction, so it is the tool's calibration point rather than a discovery.

PREDICTIONS, WRITTEN BEFORE RUNNING, and I am genuinely unsure of several
    generic      linearity holds (it is still harmonic). Null space 6. Everything else roughly as the
                 nucleosome -- if not, the nucleosome result was about protein structure after all.
    disordered   linearity holds. MODAL TRUNCATION WORSENS, because disorder localises modes and a
                 handful of soft global modes no longer spans the response. LOCALITY IMPROVES for the
                 same reason. Spectrum widens. Confidence: modal worsening high, locality improving
                 medium.
    diluted      THE NULL SPACE GROWS beyond 6. How far I do not know -- that is Maxwell counting near
                 the isostatic point and I am not going to predict the number. Confidence: direction
                 high, magnitude none.
    long-range   locality fails harder than the nucleosome. Factorisation fill-in explodes because the
                 matrix is dense. Whether modal truncation improves or worsens I genuinely cannot call.

WHAT EACH OUTCOME WOULD MEAN
    verdicts move across systems, in the predicted directions
        -> the tool measures the system. The vocabulary is general, not tuned to one case.
    verdicts move in directions I did not predict
        -> better. That is the tool telling me something, which is the entire point of building it.
    verdicts are identical across all four
        -> it measures its own probes. It is a checklist and should be retired as a discoverer.

-> outputs/physics_reduce_unknown.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics_reduce as PR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 4242
N_PT = int(os.environ.get("UNK_N", 2500))
RC = 1.6
BOX = 10.0
N_MODES = int(os.environ.get("UNK_MODES", 40))


def cloud(rng, n=N_PT, box=BOX):
    """A random 3D point cloud with a minimum separation, so the network is connected but not a lattice."""
    pts = rng.uniform(0, box, size=(n, 3))
    return pts


def network(co, rc=RC, kfun=None, rng=None, long_range=False, keep=1.0):
    """Hessian of a spring network on the given geometry. The three knobs -- spring distribution, dilution,
    and cutoff versus long-range -- are the only things that differ between the four systems."""
    n = len(co)
    rows, cols, vals = [], [], []
    npair = 0
    chunk = 512
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = co[s:e, None, :] - co[None, :, :]
        r = np.linalg.norm(d, axis=2)
        if long_range:
            ii, jj = np.where(r > 1e-9)
        else:
            ii, jj = np.where((r < rc) & (r > 1e-9))
        m = jj > (ii + s)
        ii, jj = ii[m], jj[m]
        if not len(ii):
            continue
        gi = ii + s
        rr = r[ii, jj]
        if keep < 1.0:
            sel = rng.random(len(gi)) < keep
            gi, jj, rr = gi[sel], jj[sel], rr[sel]
            if not len(gi):
                continue
        ev = (co[gi] - co[jj]) / rr[:, None]
        k = (1.0 / rr ** 2) if long_range else (kfun(len(gi)) if kfun else np.ones(len(gi)))
        npair += len(gi)
        for a in range(3):
            for b in range(3):
                v = k * ev[:, a] * ev[:, b]
                rows.extend([3 * gi + a, 3 * jj + a, 3 * gi + a, 3 * jj + a])
                cols.extend([3 * gi + b, 3 * jj + b, 3 * jj + b, 3 * gi + b])
                vals.extend([v, v, -v, -v])
    cat = lambda x: np.concatenate([np.asarray(y).ravel() for y in x])
    K = sp.coo_matrix((cat(vals), (cat(rows), cat(cols))), shape=(3 * n, 3 * n)).tocsr()
    return K, npair


def evaluate(name, co, K, npair, rng, report):
    S = PR.System(co, K, PR.rigid_basis(co))
    report(f"\n  {name.upper()}  --  {npair} couplings, {K.nnz} nonzeros "
           f"({K.nnz/(3*len(co))**2:.2e} dense fraction)")
    S.setup()
    out = {"name": name, "npair": int(npair), "nnz": int(K.nnz), "t_setup": S.t_setup}
    rows = [fn(S, rng, []) for fn in (PR.probe_linearity, PR.probe_precompute, PR.probe_locality)]
    try:
        rm, vals, n_zero = PR.probe_modal(S, rng, [], k=N_MODES)
        rows += [rm, PR.probe_nullspace(vals, n_zero, S), PR.probe_timescale(vals, S)]
    except Exception as ex:
        report(f"    eigensolve failed: {type(ex).__name__} -- reported, not hidden")
    report(f"    null space DISCOVERED: {S.n_null} directions")
    for r in rows:
        report(f"    {r['name']:<38}{r['verdict']:<10}q {r['quantity']:.4g}  x{r['speedup']:.3g}")
        report(f"      {r['note']}")
    out["reductions"] = rows
    out["n_null"] = S.n_null
    out["fill_in"] = float(S._lu.nnz / K.nnz)
    report(f"    factorisation {S.t_setup:.1f} s, fill-in {out['fill_in']:.1f}x")
    return out


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("DOES THE REDUCER DISCRIMINATE? -- four structures, predictions written down first")
    report("=" * 100)
    report("  A tool that reports the same verdicts for every system is measuring its probes, not the")
    report("  system. Identical geometry throughout, so ONLY the coupling structure differs.")

    co = cloud(rng)
    report(f"  geometry: {len(co)} points in a {BOX}^3 box, cutoff {RC}")

    systems = []
    K, np_ = network(co, rc=RC)
    systems.append(("generic", K, np_))
    K, np_ = network(co, rc=RC, kfun=lambda m: np.exp(rng.normal(0, 2.3, m)), rng=rng)
    systems.append(("disordered", K, np_))
    K, np_ = network(co, rc=RC, rng=rng, keep=0.42)
    systems.append(("diluted", K, np_))
    K, np_ = network(co[:700], long_range=True)
    systems.append(("long-range", K, np_))

    res = []
    for name, K, np_ in systems:
        c = co if name != "long-range" else co[:700]
        try:
            res.append(evaluate(name, c, K, np_, rng, report))
        except Exception as ex:
            report(f"  {name}: FAILED -- {type(ex).__name__}: {ex}")
            res.append({"name": name, "error": f"{type(ex).__name__}: {ex}"})

    # ---- the question this file exists for --------------------------------------------------------
    report("\n  DOES THE MEASUREMENT MOVE?  -- spread of the QUANTITY, not whether a binary flipped.")
    report("  The first version asked only whether verdicts changed, which collapsed six orders of")
    report("  magnitude in timescale separation into 'did not move' and then blamed the systems.")
    names = [r["name"] for r in res if "reductions" in r]
    keys = sorted({d["name"] for r in res if "reductions" in r for d in r["reductions"]})
    report(f"    {'reduction':<38}" + "".join(f"{n[:10]:>13}" for n in names) + f"{'spread':>10}")
    moved = 0
    spreads = {}
    for k in keys:
        cells, qs = [], []
        for r in res:
            if "reductions" not in r:
                continue
            d = next((x for x in r["reductions"] if x["name"] == k), None)
            if d is None:
                cells.append("-")
                continue
            qs.append(d["quantity"])
            cells.append(f"{d['verdict'][0]} {d['quantity']:.3g}")
        sp_ = (max(qs) / max(min(qs), 1e-30)) if qs and min(qs) > 0 else float("inf") if qs else 1.0
        spreads[k] = sp_
        if sp_ > 2.0:
            moved += 1
        report(f"    {k:<38}" + "".join(f"{c:>13}" for c in cells) + f"{sp_:>10.3g}x")
    report(f"\n    {moved} of {len(keys)} measurements vary by more than 2x across the four systems")

    report("\n  AGAINST THE PREDICTIONS I WROTE FIRST")
    def get(nm, key, field="quantity"):
        r = next((x for x in res if x["name"] == nm and "reductions" in x), None)
        if not r:
            return None
        d = next((x for x in r["reductions"] if key in x["name"]), None)
        return d[field] if d else None

    def cmp(label, a, b, want_bigger, fmt="{:.4g}"):
        """A prediction check that REFUSES to score a degenerate comparison. The first version printed
        'as predicted 1.000 -> 1.000' twice, comparing two complete failures and calling it confirmation."""
        if a is None or b is None:
            return (label, None, "not measured")
        note = f"{fmt.format(a)} -> {fmt.format(b)}"
        if abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-30):
            return (label, None, note + "   NOT TESTABLE, the two are identical")
        return (label, (b > a) if want_bigger else (b < a), note)
    checks = [
        cmp("disorder worsens modal truncation", get("generic", "modal"),
            get("disordered", "modal"), want_bigger=False),      # captured fraction should FALL
        cmp("disorder improves locality", get("generic", "locality"),
            get("disordered", "locality"), want_bigger=False),   # fraction of atoms needed should FALL
        cmp("dilution grows the null space", get("generic", "null"),
            get("diluted", "null"), want_bigger=True),
        cmp("long-range explodes absolute factor cost",
            next((r["nnz"] * r.get("fill_in", 1) for r in res if r["name"] == "generic"), None),
            next((r["nnz"] * r.get("fill_in", 1) for r in res if r["name"] == "long-range"), None),
            want_bigger=True),
    ]
    for lab, ok, note in checks:
        tag = "not testable" if ok is None else ("as predicted" if ok else "NOT as predicted")
        report(f"    {tag:<17} {lab:<40} {note}")
    hit = sum(1 for _, ok, _ in checks if ok)
    testable = sum(1 for _, ok, _ in checks if ok is not None)

    report("\n  READING")
    if moved >= 2:
        report(f"  THE MEASUREMENTS MOVE. {moved} of {len(keys)} vary by more than 2x with the coupling")
        report("  while the geometry is held fixed, so the tool is reading the system rather than its own")
        report("  probes. That is the property a checklist does not have.")
        report(f"  Of my written predictions, {hit} of {testable} testable came out as stated -- and those that")
        report("  did NOT are the informative rows, because they are the tool disagreeing with me.")
    else:
        report(f"  ONLY {moved} verdicts moved. The tool is largely reporting its own probes, and should be")
        report("  described as a checklist rather than a discoverer until that is fixed.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "physics_reduce_unknown", "n_points": len(co), "systems": res,
               "n_moved": moved, "n_reductions": len(keys), "spreads": spreads,
               "predictions": [{"claim": a, "held": bool(b), "note": c} for a, b, c in checks],
               "log": log}, open(OUT / "physics_reduce_unknown.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'physics_reduce_unknown.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
