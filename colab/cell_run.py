"""RUN THE CELL: every wiring active at once, and an honest count of what happens.

WHAT THIS IS. Not an experiment -- there is no hypothesis here and nothing to predeclare. It is the
integration test the repository did not have: the layers were built one loop at a time, each gated
against its own null, and nobody had ever switched them all on together. Wirings that pass
separately can still fail to compose, and the only way to find that out is to run them.

WHAT IS WIRED, as of loop 127:

    transcription      dM/dt = k_sm*(1 + TF drive) - a_deg*M          loops 112, 120
    translation        dP/dt = k_sp*(1 + beta_tl) * M - ...           loop 122
    degradation                 ... - b_deg*(1 + beta_deg)*P          loop 121
    division           halve M and P at every t = T                   loop 125
    kinetics           1,130 measured reaction kcats + a flagged constant   loop 127
    geometry           7 compartment volumes, 163 mg/mL cytosol       loops 116, 118
    diffusion          Da ~ 5e-4, so well-stirred is justified        loop 126

THE FOUR CHECKS THIS RUN MAKES, each of which can fail:

    C1  EVERY DRIVE OFF reproduces loop 125's closed form. Wirings that change the answer when
        switched off are not additions to a model, they are a different model.
    C2  THE COMBINED RUN IS STABLE and returns to its periodic steady state.
    C3  THE BUDGETS STILL CLOSE with division explicit rather than approximated -- the ribosome,
        the proteasome and the doubling time were all computed under the continuous-dilution
        assumption loop 125 bounded at 3.97%.
    C4  THE COVERAGE CASCADE. How many genes survive every requirement at once, which is the
        number that decides what "the cell model runs" is actually a statement about.

-> outputs/cell_run.json
"""
import collections
import csv
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_replication as LR  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
FILT = Path("colab/data/kinetics_filtered.json.gz")
BUNDLE = Path("colab/data/kinetics_bundle.json.gz")
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H
C1_TOL = 1e-6

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  RUN THE CELL -- every wiring at once")
    say("=" * 100)
    say()

    D = CA.load()
    st = CA.state_vector(D)
    n = len(st["genes"])
    say(f"  state vector: {n:,} genes with mRNA copies, protein copies and both half-lives")
    say(f"  model: {len(D['names']):,} genes, 12,931 reactions")
    say()

    checks = {}

    # ---------------------------------------------------------------- C1
    say("C1 EVERY DRIVE OFF REPRODUCES THE CLOSED FORM")
    trM, trP, Mend, Pend = CA.integrate_cell(st, T_DOUBLE_H, divide=True, ncyc=60, nstep=800)
    a, b = st["k_loss_mrna_deg"], st["k_loss_prot_deg"]
    xa = np.exp(-a * T_DOUBLE_H)
    M0_an = (st["k_sm"] / a) * (1 - xa) / (2 - xa)
    eM = float(np.max(np.abs(trM[0] - M0_an) / M0_an))
    say(f"     mRNA  P(0+) against (k/a)(1-x)/(2-x): max relative error {eM:.2e}")
    say(f"     mRNA  max/min over the cycle: {np.median(Mend / trM[0]):.4f} (analytic 2.0000)")
    say(f"     protein max/min over the cycle: {np.median(Pend / trP[0]):.4f}")
    say(f"     gate < {C1_TOL:.0e}")
    checks["C1"] = bool(eM < C1_TOL and abs(np.median(Mend / trM[0]) - 2.0) < 1e-3)
    say(f"     C1 {'PASS' if checks['C1'] else 'FAIL'} -- the wirings "
        f"{'compose and reduce' if checks['C1'] else 'DO NOT reduce to the unwired model'}")
    say()

    # ---------------------------------------------------------------- C2
    say("C2 THE COMBINED RUN IS STABLE")
    wiring = CA.tf_wiring(D)
    regs = sorted({r for v in wiring.values() for r, _ in v})
    ix = CA.tf_index(wiring, st["genes"], regs)
    nreg = ix[4]
    say(f"     TF wiring reaches {int((nreg > 0).sum()):,} of {n:,} state genes "
        f"({(nreg > 0).mean():.1%}) through {len(regs):,} signed regulators")
    rng = np.random.default_rng(12800)
    drive = np.zeros(len(regs))
    drive[rng.random(len(regs)) < 0.10] = 0.5      # 10% of regulators oscillating at 50%
    w = 2.0 * np.pi / T_DOUBLE_H

    def dev_at(t):
        return drive * np.sin(w * t)
    trM2, trP2, M2, P2 = CA.integrate_cell(st, T_DOUBLE_H, ix=ix, dev_at=dev_at,
                                           beta_deg=0.3, beta_tl=0.3, divide=True,
                                           ncyc=60, nstep=400)
    finite = bool(np.isfinite(trM2).all() and np.isfinite(trP2).all())
    pos = bool((trM2 > 0).all() and (trP2 > 0).all())
    # periodicity: the state after one more cycle must match
    trM3, trP3, _, _ = CA.integrate_cell(st, T_DOUBLE_H, ix=ix, dev_at=dev_at,
                                         beta_deg=0.3, beta_tl=0.3, divide=True,
                                         ncyc=61, nstep=400)
    per = float(np.max(np.abs(trP3[0] - trP2[0]) / np.maximum(trP2[0], 1e-300)))
    say(f"     all drives on: 10% of regulators oscillating, degradation +/-30%, "
        f"translation +/-30%, halving at {T_DOUBLE_H} h")
    say(f"     finite {finite}, strictly positive {pos}")
    say(f"     periodic to {per:.2e} after an extra cycle")
    say(f"     mRNA relative swing median {np.median((trM2.max(0) - trM2.min(0)) / (2 * trM2.mean(0))):.4f}")
    say(f"     protein relative swing median "
        f"{np.median((trP2.max(0) - trP2.min(0)) / (2 * trP2.mean(0))):.4f}")
    checks["C2"] = bool(finite and pos and per < 1e-6)
    say(f"     C2 {'PASS' if checks['C2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- C3
    say("C3 THE BUDGETS STILL CLOSE WITH DIVISION EXPLICIT")
    S = D["schwan"]
    import re as _re
    rp = _re.compile(r"^(RPL|RPS)\d+[A-Z]?$|^RPLP\d$|^RPSA$")
    L = {}
    nm, c = None, 0
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if nm and c:
                    L[nm] = max(L.get(nm, 0), c)
                c, nm = 0, None
                for p in ln.split():
                    if p.startswith("GN="):
                        nm = p[3:]
                        break
            else:
                c += len(ln.strip())
    if nm and c:
        L[nm] = max(L.get(nm, 0), c)
    gs = st["genes"]
    res = np.array([L.get(g, 0) for g in gs], float)
    # CODON DEMAND, AND THE FIRST VERSION OF THIS WAS WRONG. I counted only the NET increase over a
    # cycle -- the P0 a cell must add to reach 2*P0 before halving -- and got a 3.18 h doubling
    # time against a measured 27.5. But a cell does not only add protein, it also replaces
    # everything that degraded while it was adding, and for a proteome whose median half-life is
    # 47.7 h against a 27.5 h cycle that replacement term is most of the work. The honest quantity
    # is total SYNTHESIS over the cycle, integral of k_sp*M dt, which the trajectory already has.
    # Under the continuous approximation this reduces to sum(P*b*res), which is loop 101's formula,
    # so the two are comparable rather than a new number.
    synth_rate = st["k_sp"] * trM.mean(0)            # molecules per hour, cycle-averaged
    demand = float((synth_rate * res).sum())
    xb = np.exp(-b * T_DOUBLE_H)
    P0 = (st["k_sp"] * st["M"] / b) * (1 - xb) / (2 - xb)
    net = float((P0 * res).sum() / T_DOUBLE_H)
    say(f"     net accumulation alone would be {net / 1e9:.2f} Gcodons/h -- the replacement of "
        f"what degrades is {demand / max(net, 1e-9):.1f}x larger and is the real cost")
    ribo = float(np.median([S[g]["prot_copies"] for g in S
                            if rp.match(g) and S[g].get("prot_copies")]))
    cap = ribo * 6.0 * 3600.0
    say(f"     ribosomes (median RPL/RPS copy number, strict regex) {ribo:,.0f}")
    say(f"     codon demand {demand / 1e9:.2f} Gcodons/h over {len(gs):,} genes")
    say(f"     capacity at 6 aa/s {cap / 1e9:.2f} Gcodons/h -> utilisation "
        f"{100 * demand / cap:.1f}%")
    T_pred = float((demand * T_DOUBLE_H) / cap)
    say(f"     doubling time for THIS SUBSET at 100% ribosome occupancy: {T_pred:.2f} h")
    say(f"     NOT COMPARABLE TO LOOP 101's 13.28 h, and the difference is stated rather than")
    say(f"     reconciled away. That figure is the whole cell's codon content (1.64e12, coverage-")
    say(f"     corrected past the 61.9% of codon mass these genes cover) at 75% ribosome occupancy.")
    say(f"     This one is {len(gs):,} measured genes at 100%. Two different quantities; the number")
    say(f"     that carries over is the UTILISATION, {100 * demand / cap:.1f}% here against loop 92's 22.3%.")
    say(f"     loop 125 bounded the continuous-dilution error at 3.97%, so making division explicit "
        f"moves these by at most that much")
    checks["C3"] = bool(demand < cap)
    say(f"     C3 {'PASS' if checks['C3'] else 'FAIL'} -- the ribosome budget "
        f"{'closes' if checks['C3'] else 'DOES NOT close'}")
    say()

    # ---------------------------------------------------------------- C4
    say("C4 THE COVERAGE CASCADE -- what 'the cell model runs' is a statement about")
    C = D["model"]
    names = set(D["names"])
    ens = {}
    with open(LR.SC / "HumanGEM_genes.tsv") as f:
        rr = csv.reader(f, delimiter="\t")
        hd = [x.strip('"') for x in next(rr)]
        i1, i2 = hd.index("genes"), hd.index("geneSymbols")
        for x in rr:
            e_, s_ = x[i1].strip('"'), x[i2].strip('"')
            if e_ and s_:
                ens[e_] = s_.split(";")[0]
    B = json.load(gzip.open(BUNDLE, "rt"))
    F = json.load(gzip.open(FILT, "rt")) if FILT.exists() else None
    rxg = B["reaction_genes"]
    metab = {ens[z] for gg in rxg.values() for z in gg if z in ens}
    kept_rx = {r for r, t in (F["reaction_tier"].items() if F else [])
               if t == "1_human_EC_narrow"}
    kin_g = {ens[z] for r in kept_rx for z in rxg.get(r, []) if z in ens}
    wired = {gs[i] for i in range(n) if nreg[i] > 0}
    steps = [("in the model", names),
             ("+ full dynamical state", names & set(gs)),
             ("+ a metabolic reaction", names & set(gs) & metab),
             ("+ a signed TF regulator", names & set(gs) & metab & wired),
             ("+ a MEASURED kcat surviving loop 127", names & set(gs) & metab & wired & kin_g)]
    prev = None
    for lbl, s_ in steps:
        d_ = "" if prev is None else f"   (-{len(prev) - len(s_):,})"
        say(f"     {lbl:<40} {len(s_):>7,}   {len(s_) / len(names):>6.1%} of 16,492{d_}")
        prev = s_
    core = steps[-1][1]
    mf = CA.mass_fraction(D, core) if core else float("nan")
    say(f"     the fully-wired core carries {mf:.2%} of measured proteome mass")
    say(f"     by REACTION: {len(kept_rx):,} of 12,931 = {len(kept_rx) / 12931:.1%} carry a "
        f"measured kcat")
    checks["C4"] = bool(len(core) > 0)
    say(f"     C4 {'PASS' if checks['C4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- the table
    say("=" * 100)
    say("  THE AUDIT TABLE")
    say("=" * 100)
    cnt = collections.Counter(s for _, s, _, _, _ in CA.LAYERS)
    for status in ("RUNS", "CLOSES", "STATIC", "FAILED", "ABSENT"):
        rows = [r for r in CA.LAYERS if r[1] == status]
        if not rows:
            continue
        say(f"\n  {status}  ({len(rows)})")
        for nm_, _, src, _, _ in rows:
            say(f"     {nm_:<46} {src}")
    say()
    say(f"  {len(CA.LAYERS)} layers: " + ", ".join(f"{k} {v}" for k, v in cnt.most_common()))
    say()
    say("=" * 100)
    for k in ("C1", "C2", "C3", "C4"):
        say(f"  {k}  {'PASS' if checks[k] else 'FAIL'}")
    say(f"  {sum(checks.values())}/4")
    say("=" * 100)

    json.dump({"test": "cell_run", "checks": checks,
               "state_genes": n, "model_genes": len(D["names"]),
               "c1": {"max_rel_error": eM,
                      "mrna_maxmin": float(np.median(Mend / trM[0])),
                      "protein_maxmin": float(np.median(Pend / trP[0]))},
               "c2": {"tf_reach": int((nreg > 0).sum()), "regulators": len(regs),
                      "periodic_to": per, "finite": finite, "positive": pos,
                      "mrna_swing": float(np.median((trM2.max(0) - trM2.min(0))
                                                    / (2 * trM2.mean(0)))),
                      "protein_swing": float(np.median((trP2.max(0) - trP2.min(0))
                                                       / (2 * trP2.mean(0))))},
               "c3": {"ribosomes": ribo, "demand_gcodons_h": demand / 1e9,
                      "net_only_gcodons_h": net / 1e9,
                      "capacity_gcodons_h": cap / 1e9, "utilisation": demand / cap,
                      "doubling_h": T_pred},
               "c4": {lbl: len(s_) for lbl, s_ in steps} | {"core_mass_fraction": mf,
                                                            "reactions_measured": len(kept_rx)},
               "layers": dict(cnt), "n_layers": len(CA.LAYERS),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "cell_run.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'cell_run.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
