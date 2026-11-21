"""LOOP 96 -- GIVE SIGNALLING A STOICHIOMETRY, AND MAKE IT PAY FOR ITSELF.

THE HOLE. cell_sim.py states it plainly: 12,931 of the project's reactions have stoichiometry and
the 15,597 Reactome signalling steps do NOT, because there is no "2 A + 1 B -> 1 C" for a
phosphorylation cascade in that annotation, so they cannot enter a mass-balance model at all. The
metabolic economy runs; the regulatory government is absent. That is the single largest structural
gap between this repository and a whole-cell model, and it has been described rather than closed.

IT DOES NOT NEED THE ANNOTATION. A phosphorylation is one of the most stoichiometrically rigid
events in biology:

        kinase + substrate + ATP  ->  kinase + phospho-substrate + ADP

One phosphate, one ATP, every time. The identity of the kinase changes which substrate, not the
arithmetic. So the stoichiometry that Reactome does not record can be RECONSTRUCTED from what the
model already holds: 7,210 genes carry phosphosite counts, and 13,195 of the 17,432 signalling edges
point at a phospho-bearing target. Each site that turns over costs exactly one ATP.

AND THAT TURNS AN UNQUANTIFIED LAYER INTO A BUDGET. Phosphosites are not permanent -- kinases and
phosphatases cycle them continuously, which is what makes signalling fast and what makes it
expensive. The cost per hour is

        ATP/h = sum over proteins of ( sites x copies x ln2 / phosphosite half-life )

and every term is either measured or swept. Loop 92 established the comparison: translation demands
30.76 Gcodons/h, and a peptide bond costs about 4 ATP, so protein synthesis runs at roughly 123
G ATP/h. If signalling comes out comparable to that, either the phosphosite lifetime assumption is
wrong or signalling is a far larger energy sink than anyone reports. If it comes out small, the
layer can be priced and carried.

THE ABUNDANCE RULE FROM LOOP 92 IS OBEYED. Copy numbers come from Schwanhausser, the same cells the
half-lives come from. Loop 92's second run mixed HeLa protein copies with NIH3T3 mRNA copies and got
a median translation rate of 1,655/h against a published 140, because the ratio of two proteomes is
a cell-type difference rather than a rate. Nothing here divides one dataset by another.

PREDECLARED, before any number:

  Y1 THE STOICHIOMETRY CAN ACTUALLY BE WRITTEN                      THE COVERAGE.
       how many signalling edges receive a balanced form, and what fraction of the layer that is.
       Reported. An edge whose target carries no phosphosite gets no reaction and is counted as
       uncovered rather than assumed to be something else.
  Y2 SIGNALLING IS A MINORITY OF THE ATP BUDGET                     THE GATE.
       phosphorylation ATP against translation ATP from loop 92's measured codon demand. Gate:
       signalling must cost LESS than translation, swept over phosphosite half-lives from 10 s to
       10 min. If it exceeds translation at every lifetime in that range, the reconstruction is
       wrong and this loop says so rather than adjusting the lifetime until it fits.
  Y3 THE COST IS CARRIED BY ABUNDANCE, NOT BY HEAVY PHOSPHORYLATION THE DIAGNOSTIC.
       which proteins dominate the bill, and whether it is the most-phosphorylated or the most
       abundant. Reported, because it decides whether a signalling model needs site-level detail
       or only concentrations.
  Y4 THE FAME CONTROL                                               THE RECURRING KILLER.
       phosphosite counts come from mass spectrometry, which finds abundant and heavily studied
       proteins first. `pubs` against site count, and against the per-protein ATP bill. Reported
       next to the biology.
  Y5 THE NULL FIRES, AND IS CHECKED FOR THE ABILITY TO              THE GUARD, GUARDED.
       shuffle site counts across proteins, keeping the totals. gate_guard.null_can_move() confirms
       the shuffle actually changes the input before its output is used, and gate_guard.survival()
       decides whether a fraction is even defined. Twelve gates this session fired while measuring
       nothing; loop 105 made the check mechanical and this is the first loop built on it.
  Y6 WHAT THIS DOES NOT GIVE                                        THE HONEST LIMIT.
       a stoichiometry is not a rate law. This prices the layer; it does not simulate it, and the
       difference is stated so nobody reads a budget as a running signalling model.

-> outputs/loop_signalling_cost.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LN2 = float(np.log(2.0))

TRANSLATION_CODONS_H = 30_762_337_520.0     # loop 92, self-consistent Schwanhausser set
ATP_PER_PEPTIDE_BOND = 4.0
PHOSPHO_HL_SWEEP_S = (10.0, 60.0, 300.0, 600.0)
NPERM = 20
SEED = 9601

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 96 -- give signalling a stoichiometry, and make it pay for itself")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    idx = {n: i for i, n in enumerate(names)}
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    ptm = C["ptm"]
    sig = C["sig"]
    S = json.load(open(SC / "_schwan2011.json"))
    copies = {g: v["prot_copies"] for g, v in S.items() if v.get("prot_copies")}

    sites = {}
    for k, v in (ptm.items() if isinstance(ptm, dict) else ptm):
        nm = names[int(k)] if str(k).isdigit() and int(k) < len(names) else str(k)
        if "Phospho" in v.get("c", []):
            sites[nm] = int(v.get("n", 1))
    say(f"  {len(sites):,} genes carry phosphosite counts; {len(copies):,} have protein copies")
    say(f"  {len(sig):,} signalling edges in the model")
    say()

    say("Y1 THE STOICHIOMETRY CAN ACTUALLY BE WRITTEN")
    phos_idx = {idx[g] for g in sites if g in idx}
    covered = [e for e in sig if e[1] in phos_idx]
    rxn = []
    for e in covered:
        src = names[e[0]] if e[0] < len(names) else None
        tgt = names[e[1]] if e[1] < len(names) else None
        if tgt is None:
            continue
        rxn.append({"kinase": src, "substrate": tgt, "sign": e[2] if len(e) > 2 else 0,
                    "sites": sites.get(tgt, 1)})
    say(f"     edges whose target carries a phosphosite: {len(covered):,} of {len(sig):,} "
        f"({len(covered)/len(sig):.1%})")
    say(f"     each becomes:  kinase + substrate + ATP -> kinase + phospho-substrate + ADP")
    say(f"     balanced reactions written: {len(rxn):,}")
    say(f"     uncovered edges are counted as uncovered, not assumed to be something else")
    say()

    say("Y2 SIGNALLING IS A MINORITY OF THE ATP BUDGET")
    both = [g for g in sites if g in copies]
    total_sites = sum(sites[g] * copies[g] for g in both)
    say(f"     {len(both):,} proteins with both a site count and a copy number")
    say(f"     total phosphosites in the cell: {total_sites:,.0f}")
    tr_atp = TRANSLATION_CODONS_H * ATP_PER_PEPTIDE_BOND
    say(f"     translation: {TRANSLATION_CODONS_H:,.0f} codons/h x {ATP_PER_PEPTIDE_BOND:.0f} ATP "
        f"= {tr_atp:,.0f} ATP/h  (loop 92)")
    sweep = {}
    for hl_s in PHOSPHO_HL_SWEEP_S:
        k = LN2 / (hl_s / 3600.0)
        atp = total_sites * k
        sweep[hl_s] = {"k_per_h": k, "atp_per_h": atp, "vs_translation": atp / tr_atp}
        say(f"     phosphosite half-life {hl_s:6.0f} s -> {atp:>20,.0f} ATP/h   "
            f"{atp/tr_atp:>8.2f}x translation")
    y2 = all(v["vs_translation"] < 1.0 for v in sweep.values())
    y2_any = any(v["vs_translation"] < 1.0 for v in sweep.values())
    say(f"     gate: signalling below translation at EVERY lifetime in the swept range")
    say(f"     Y2 {'PASS' if y2 else 'FAIL'} -- "
        f"{'signalling is affordable across the whole range' if y2 else ('affordable only at the LONGER lifetimes' if y2_any else 'signalling exceeds translation everywhere, so the reconstruction is wrong')}")
    say()

    say("Y3 THE COST IS CARRIED BY ABUNDANCE, NOT BY HEAVY PHOSPHORYLATION")
    bill = {g: sites[g] * copies[g] for g in both}
    order = sorted(both, key=lambda g: -bill[g])
    top = sum(bill[g] for g in order[:20]) / sum(bill.values())
    r_sc, _ = spear([sites[g] for g in both], [bill[g] for g in both])
    r_ab, _ = spear([copies[g] for g in both], [bill[g] for g in both])
    say(f"     top 20 proteins carry {top:.1%} of the phosphorylation bill: "
        f"{', '.join(order[:6])}")
    say(f"     bill vs site count {r_sc:+.4f}   bill vs copy number {r_ab:+.4f}")
    say(f"     median sites/protein {np.median([sites[g] for g in both]):.0f}, "
        f"max {max(sites.values())}")
    say(f"     -> a signalling model needs {'CONCENTRATIONS' if r_ab > r_sc else 'SITE-LEVEL DETAIL'} "
        f"first")
    say()

    say("Y4 THE FAME CONTROL")
    pb = [pubs.get(g, 0.0) for g in both]
    r_p_sites, n_p = spear([sites[g] for g in both], pb)
    r_p_bill, _ = spear([bill[g] for g in both], pb)
    r_p_cop, _ = spear([copies[g] for g in both], pb)
    say(f"     pubs vs phosphosite count {r_p_sites:+.4f}   vs the ATP bill {r_p_bill:+.4f}   "
        f"vs copy number {r_p_cop:+.4f}   (n {n_p:,})")
    say(f"     phosphosite counts come from mass spectrometry, which finds abundant and studied")
    say(f"     proteins first, so a positive correlation here is expected and is NOT evidence of")
    say(f"     anything biological. It bounds what a site-count-based claim could ever mean.")
    say()

    say("Y5 THE NULL FIRES, AND IS CHECKED FOR THE ABILITY TO")
    rng = np.random.default_rng(SEED)
    real_r = r_ab
    real_vec = [sites[g] for g in both]
    nulls, moved = [], []
    for _ in range(NPERM):
        perm = rng.permutation(len(both))
        shuf = [real_vec[i] for i in perm]
        moved.append(GG.null_can_move(real_vec, shuf)["changed"])
        b2 = {g: shuf[i] * copies[g] for i, g in enumerate(both)}
        nulls.append(spear([copies[g] for g in both], [b2[g] for g in both])[0])
    cap = GG.null_can_move(real_vec, [real_vec[i] for i in rng.permutation(len(both))])
    say(f"     CAPABILITY CHECK: shuffling site counts changes {np.mean(moved):.1%} of entries "
        f"-- capable: {cap['capable']}")
    s = GG.survival(real_r, nulls)
    GG.report("bill-vs-abundance under a site-count shuffle", s, emit=say)
    say(f"     interpretation: if the correlation survives shuffling the SITE COUNTS, the bill is")
    say(f"     abundance and the site annotation adds nothing to it")
    y5 = bool(cap["capable"])
    say(f"     Y5 {'PASS' if y5 else 'FAIL'} -- the null "
        f"{'is capable and its verdict is usable' if y5 else 'is INERT'}")
    say()

    say("Y6 WHAT THIS DOES NOT GIVE")
    say(f"     a stoichiometry is not a rate law. {len(rxn):,} balanced reactions now exist and can")
    say(f"     enter a mass balance, but nothing here supplies kinase kinetics, phosphatase")
    say(f"     activities, or the ordering of a cascade. This prices the layer; it does not run it.")
    say(f"     The phosphosite half-life is swept rather than known, and it is the single largest")
    say(f"     uncertainty in the number: a factor of 60 across the swept range.")
    say()

    gates = {"Y1 the stoichiometry can be written": True,
             "Y2 signalling is a minority of the ATP budget": bool(y2),
             "Y3 the cost driver is identified": True,
             "Y4 fame reported": True,
             "Y5 the null is capable and fired": bool(y5),
             "Y6 the limit is stated": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(SC / "_schwan2011.json")],
                      available=len(sig), used=len(rxn), selection="filtered", seed=SEED,
                      controls=["the phosphosite lifetime swept over 60x rather than assumed",
                                "translation ATP taken from loop 92's measured codon demand",
                                "copy numbers and site counts from one cell type, per loop 92's rule",
                                "publication count against site count and against the bill",
                                "a site-count shuffle checked for capability before use",
                                "uncovered edges counted as uncovered"],
                      note="cell_sim.py records that signalling cannot enter mass balance for want "
                           "of stoichiometry; phosphorylation's stoichiometry does not need the "
                           "annotation, only the site counts")
    RM.report(man, emit=say)
    json.dump({"test": "loop_signalling_cost", "manifest": man, "gates": gates,
               "y1": {"n_edges": len(sig), "n_covered": len(covered), "n_reactions": len(rxn),
                      "fraction": len(covered) / len(sig)},
               "y2": {"total_phosphosites": total_sites, "translation_atp_h": tr_atp,
                      "sweep": {str(k): v for k, v in sweep.items()}},
               "y3": {"top20_share": top, "rho_bill_sites": r_sc, "rho_bill_copies": r_ab,
                      "top": order[:20]},
               "y4": {"pubs_vs_sites": r_p_sites, "pubs_vs_bill": r_p_bill,
                      "pubs_vs_copies": r_p_cop},
               "y5": {"capability": cap, "survival": s},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_signalling_cost.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_signalling_cost.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
