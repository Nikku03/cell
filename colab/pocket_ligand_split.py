"""POST-HOC DIAGNOSTIC ON LOOP 60 -- what kind of site does the geometry model actually find?

THIS IS NOT A GATE AND MUST NOT BE QUOTED AS ONE. Loop 60's gates were declared before it ran; this
was written AFTER seeing its per-protein spread and is therefore a hypothesis about that spread, not
a test of it. It is here because the spread is large -- KIF5A 0.9778 down to SERPING1 0.2810 -- and
an unexplained 0.7 range across proteins is a result nobody should build on.

THE HYPOTHESIS, formed by reading the eight best and eight worst protein names. The winners are
motors, ATPases and chromatin enzymes (KIF5A, SPAST, CHD8, CHD2, KDM5C, NSD1); the losers are
secreted coagulation and complement proteins (SERPING1, F9, F8) plus large multi-domain surface
proteins (NOTCH1, CDH1, OTOF). Those two groups differ in what their annotated residues BIND:

    a nucleotide pocket is a BURIED CLEFT      -- ATP, GDP, FAD, NADP, acetyl-CoA, SAM
    a metal site is often SURFACE COORDINATION -- the Ca2+ of a Gla or EGF-like domain sits on the
                                                  outside of the protein, holding it rigid

If that is what is happening, the geometry block is a buried-pocket detector and its failures are
not noise -- they are a named class of site it should be expected to miss. If it is not, the spread
is unexplained and loop 60's macro number is an average over something the model does not understand.

HOW SEPARATE THIS TEST IS FROM THE HYPOTHESIS, stated honestly: the hypothesis came from 16 protein
names, and the test runs over all 81 proteins in loop 60's primary set. That is partial independence,
not full independence. It is reported as a diagnostic and the falsifiable prediction it makes -- that
nucleotide-site proteins score higher than metal-site proteins, and that nucleotide residues are more
buried -- is the thing a later predeclared loop would have to confirm on held-out proteins.

Ligand identity comes from UniProt featureCrossReferences (ChEBI), which 94.5% of binding and active
sites carry. This project's earlier cache kept only the free-text `description` field, which is empty
for 508 of 541 sites, and that is why the identity was previously reported as unavailable.

RESULT, AND THE QUALIFICATION THE VERDICT STRING DOES NOT MAKE. nucleotide 0.8481 mean over 32
proteins at burial percentile 0.843; metal 0.7428 over 21 at 0.755. The automated verdict reads
SUPPORTED on a +0.1053 mean gap and +0.089 more burial.

But the MEDIANS are 0.8539 and 0.8096 -- a gap of only +0.044. So the effect is carried by a TAIL of
bad metal proteins (F9 0.3538, NOTCH1 0.4287, OTOF 0.4399) and not by a uniform shift: plenty of
metal-site proteins score at the top (SCN8A 0.9584, COL5A1 0.9461, SMPD1 0.9227). The honest reading
is that "metal site" is not itself the failure mode -- surface-exposed metal coordination in large
secreted multi-domain proteins is, and that is a narrower claim than the one the mean supports.

-> outputs/pocket_ligand_split.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC

NUCLEOTIDE = {"CHEBI:30616": "ATP", "CHEBI:58189": "GDP", "CHEBI:37565": "GTP",
              "CHEBI:57692": "FAD", "CHEBI:58349": "NADP", "CHEBI:57288": "acetyl-CoA",
              "CHEBI:59789": "SAM", "CHEBI:61560": "dNTP", "CHEBI:57783": "NADPH",
              "CHEBI:58248": "GDP-analog", "CHEBI:57945": "NADH", "CHEBI:15422": "ATP",
              "CHEBI:456216": "ADP", "CHEBI:57540": "NAD", "CHEBI:33019": "triphosphate"}
METAL = {"CHEBI:29108": "Ca2+", "CHEBI:29105": "Zn2+", "CHEBI:18420": "Mg2+",
         "CHEBI:49552": "Cu+", "CHEBI:29101": "Na+", "CHEBI:60344": "heme b",
         "CHEBI:29033": "Fe2+", "CHEBI:29034": "Fe3+", "CHEBI:48828": "Mn2+",
         "CHEBI:29103": "K+", "CHEBI:48775": "Cd2+", "CHEBI:16991": "DNA"}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def klass(ids):
    n = sum(i in NUCLEOTIDE for i in ids)
    m = sum(i in METAL and METAL[i] != "DNA" for i in ids)
    if n and not m:
        return "nucleotide"
    if m and not n:
        return "metal"
    if n and m:
        return "mixed"
    return "other"


def main():
    res = json.load(open(OUT / "loop_pocket.json"))
    per = res["per_protein"]
    B = pickle.load(open(SC / "_pocket_v1.pkl", "rb"))
    LIG = json.load(open(SC / "_ligand_by_residue.json"))
    rows = {r["gene"]: r for r in B["rows"]}

    say("=" * 100)
    say("  POST-HOC on loop 60 -- what kind of site does the geometry actually find?")
    say("  NOT A GATE. Written after seeing the per-protein spread.")
    say("=" * 100)
    say()

    groups = {}
    burial = {}
    for g, auc in per.items():
        lg = LIG.get(g)
        r = rows.get(g)
        if not lg or r is None:
            continue
        pos = [i for i, y in enumerate(r["y"]) if y]
        if not pos:
            continue
        idx = r["idx"]
        ks = [klass(lg.get(str(idx[i]), lg.get(idx[i], []))) for i in pos]
        cnt = {k: ks.count(k) for k in set(ks)}
        dom = max(cnt, key=cnt.get)
        if cnt[dom] < 0.6 * len(ks):
            dom = "mixed"
        groups.setdefault(dom, []).append((g, auc))
        # deg16 percentile of the functional residues -- column 2 of the geometry block, which
        # survives the collinearity drop; burial is read straight off the within-protein rank
        col = [c for c in range(r["R"].shape[1])]
        b = float(np.mean(r["R"][pos, 2])) if r["R"].shape[1] > 2 else float("nan")
        burial.setdefault(dom, []).append(b)

    say(f"  {'site class':14s} {'proteins':>9s} {'mean AUC':>10s} {'median':>9s} {'burial pctile':>15s}")
    summary = {}
    for k in ("nucleotide", "metal", "mixed", "other"):
        if k not in groups:
            continue
        v = [a for _, a in groups[k]]
        bs = burial[k]
        say(f"  {k:14s} {len(v):9d} {np.mean(v):10.4f} {np.median(v):9.4f} {np.mean(bs):15.3f}")
        summary[k] = {"n": len(v), "mean_auc": float(np.mean(v)), "median_auc": float(np.median(v)),
                      "mean_burial_pctile": float(np.mean(bs))}
    say()

    for k in ("nucleotide", "metal"):
        if k in groups:
            say(f"  {k}:")
            for g, a in sorted(groups[k], key=lambda kv: -kv[1])[:6]:
                say(f"     {g:10s} {a:.4f}")
            if len(groups[k]) > 6:
                say("     ...")
                for g, a in sorted(groups[k], key=lambda kv: kv[1])[:3]:
                    say(f"     {g:10s} {a:.4f}")
            say()

    verdict = None
    if "nucleotide" in summary and "metal" in summary:
        d = summary["nucleotide"]["mean_auc"] - summary["metal"]["mean_auc"]
        db = summary["nucleotide"]["mean_burial_pctile"] - summary["metal"]["mean_burial_pctile"]
        say(f"  nucleotide - metal:  AUC {d:+.4f}   burial percentile {db:+.3f}")
        if d > 0.05 and db > 0.02:
            verdict = ("SUPPORTED -- the geometry block is a buried-pocket detector, and its failures "
                       "are a named class of site rather than noise")
        elif d > 0.05:
            verdict = ("PARTIAL -- nucleotide sites score higher but are not measurably more buried, "
                       "so burial is not the mechanism and the reason is still unidentified")
        else:
            verdict = ("REFUTED -- the ligand class does not explain the per-protein spread, so "
                       "loop 60's macro is an average over variation the model does not account for")
        say(f"  {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "pocket_ligand_split", "post_hoc": True,
               "not_a_gate": "hypothesis formed after seeing loop 60's per-protein spread",
               "by_class": summary, "verdict": verdict,
               "members": {k: sorted(v, key=lambda kv: -kv[1]) for k, v in groups.items()},
               "log": log}, open(OUT / "pocket_ligand_split.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'pocket_ligand_split.json'}")


if __name__ == "__main__":
    main()
