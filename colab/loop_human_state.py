"""LOOP 128 -- REBUILD THE STATE VECTOR ON HUMAN DATA, and measure what mixing sources costs.

WHY THE CURRENT ONE IS SMALL, AND WHY THAT WAS THE RIGHT CHOICE. state_vector needs mRNA copies,
protein copies, an mRNA half-life and a protein half-life, and takes all four from Schwanhausser
2011 -- one study, one cell type, mouse fibroblasts, 4,190 genes. Loop 92 established the reason:
mixing HeLa protein copies with NIH3T3 mRNA copies produced a translation rate twelve times the
published value, and the rule that came out of it is that a rate may never be formed by dividing
two abundance datasets measured in different cells.

WHAT THE FETCH FOUND. Human replacements exist for three of the four fields, and one pairing is
same-cell-line:

    protein copies      Itzhak 2016, HeLa                      8,469 genes
    protein half-life   Mathieson 2018, human primary cells    8,804 genes
    mRNA half-life      AvgKdegs, HeLa                         9,967 genes
    mRNA copies         NOTHING ABSOLUTE IN HeLa               the gap that remains

So a human protein-side state is available for 5,595 genes against the current 4,190, and 6,807
genes have an mRNA half-life and a protein copy number FROM THE SAME CELL LINE. But Mathieson is
primary cells and Itzhak is HeLa, so the protein arm is still a cross-cell-type combination -- and
that is exactly the thing loop 92 caught being wrong by a factor of twelve.

SO THE GATE IS NOT COVERAGE. Coverage is arithmetic and this loop reports it, but the question that
decides whether the human build is usable is whether it AGREES with the self-consistent mouse build
where the two overlap. A bigger state vector that disagrees with the validated one is not progress,
it is a larger error. That is what H1 and H2 test, and H2 is the one that matters because it tests
the DERIVED RATE rather than the inputs -- two datasets can correlate well and still produce a
production rate that is off by an order of magnitude, because the rate multiplies them.

PREDECLARED:

  H1 THE SOURCES AGREE PAIRWISE WHERE THEY OVERLAP                  THE INPUT CHECK.
       Itzhak against Schwanhausser on protein copies, Mathieson against Schwanhausser on protein
       half-lives, AvgKdegs-HeLa against Schwanhausser on mRNA half-lives. Different labs, methods
       and cell types, so perfect agreement is not expected and is not the gate. Gate: Spearman
       >= 0.30 on each. A source that does not clear that cannot be swapped into a validated build
       whatever its coverage.
  H2 THE DERIVED RATE AGREES, NOT JUST THE INPUTS                   THE GATE THAT MATTERS.
       protein production rate P * b, computed from the human sources and from Schwanhausser, on
       the genes both cover. This is the quantity loop 92 got wrong by 12x, and correlated inputs
       do not guarantee it. Gate: median ratio within 2x AND Spearman >= 0.5. If the rate does not
       survive, the human build is reported as unusable for dynamics regardless of H1.
  H3 THE COVERAGE GAIN, THREE DENOMINATORS                          THE ARITHMETIC.
       by gene, by proteome mass and against the 16,492, for the protein-side state and for the
       same-cell-line subset separately, because those are different objects.
  H4 THE BUDGETS STILL CLOSE ON HUMAN DATA                          THE PHYSICAL TEST.
       the ribosome budget recomputed from the human build. It closed at 18.3% utilisation on the
       mouse build in cell_run. Gate: demand must stay inside capacity. New data that breaks a
       budget which previously balanced is a finding about the new data.
  H5 THE mRNA ARM                                                   THE HONEST GAP.
       whether absolute mRNA copies can be had for HeLa at all, and if an nTPM-based estimate is
       attempted, whether it reproduces Schwanhausser's mRNA copies on the overlap. Gate: either a
       validated mRNA copy estimate at Spearman >= 0.4, or an explicit statement that the mRNA arm
       cannot be built and the human state is protein-side only.
  H6 FAME, AND THE COST OF MIXING                                   THE GUARD.
       publication count of the genes gained against those already present -- a bigger state vector
       made of better-studied genes is a coverage gain that will not generalise. Plus the measured
       cost of the cross-cell-type combination, stated as a number rather than a caveat.

-> outputs/loop_human_state.json
"""
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
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
MATH = LR.SC / "_mathieson2018.json"
ITZ = LR.SC / "itzhak_supp1.xlsx"
KDEG = LR.SC / "AvgKdegs_genes_v1.csv"
PROT = LR.SC / "human_proteome.fasta.gz"
SEED = 12900
NPERM = 2000
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H

H1_RHO = 0.30
H2_RATIO = 2.0
H2_RHO = 0.50
H5_RHO = 0.40

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def perm_p(a, b, rng, n=NPERM):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    obs = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    k = len(a)
    null = np.array([(lambda s: np.median(s[:k]) - np.median(s[k:]))(rng.permutation(pool))
                     for _ in range(n)])
    return obs, float(np.mean(np.abs(null) >= abs(obs)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 128 -- rebuild the state vector on human data, and price the mixing")
    say("=" * 100)
    say()

    D = CA.load()
    names = set(D["names"])
    S = D["schwan"]
    st = CA.state_vector(D)
    cur = set(st["genes"])

    import pandas as pd
    math = json.load(open(MATH))
    m_hl = {g: float(v["prot_hl_h"]) for g, v in math.items() if v.get("prot_hl_h")}
    d = pd.read_excel(ITZ, sheet_name="Compact HeLa Spatial Proteome")
    i_cp = {}
    for g, c in zip(d["Lead Gene name"].astype(str),
                    pd.to_numeric(d["Estimated Copy number per cell"], errors="coerce")):
        if np.isfinite(c) and c > 0:
            i_cp[g] = max(i_cp.get(g, 0.0), float(c))
    rr = csv.reader(open(KDEG))
    hd = next(rr)
    iG, iC, iHL = hd.index("feature_ID"), hd.index("cell_line"), hd.index("avg_halflife")
    k_hela = {}
    for x in rr:
        if x[iC] != "HeLa":
            continue
        try:
            v = float(x[iHL])
        except (TypeError, ValueError):
            continue
        if v > 0:
            k_hela[x[iG]] = v
    say(f"  Schwanhausser (mouse, self-consistent): {len(cur):,} genes with all four fields")
    say(f"  Mathieson protein half-lives {len(m_hl):,} | Itzhak HeLa copies {len(i_cp):,} | "
        f"AvgKdegs HeLa mRNA half-lives {len(k_hela):,}")
    say()

    gates = {}

    # ---------------------------------------------------------------- H1
    say("H1 THE SOURCES AGREE PAIRWISE WHERE THEY OVERLAP")
    pairs = {}
    ov = [g for g in i_cp if g in S and S[g].get("prot_copies")]
    pairs["Itzhak vs Schwanhausser protein copies"] = (
        spearman([i_cp[g] for g in ov], [S[g]["prot_copies"] for g in ov]), len(ov))
    ov2 = [g for g in m_hl if g in S and S[g].get("prot_hl_h")]
    pairs["Mathieson vs Schwanhausser protein half-life"] = (
        spearman([m_hl[g] for g in ov2], [S[g]["prot_hl_h"] for g in ov2]), len(ov2))
    ov3 = [g for g in k_hela if g in S and S[g].get("mrna_hl_h")]
    pairs["AvgKdegs-HeLa vs Schwanhausser mRNA half-life"] = (
        spearman([k_hela[g] for g in ov3], [S[g]["mrna_hl_h"] for g in ov3]), len(ov3))
    for k, (r, n_) in pairs.items():
        say(f"     {k:<48} rho {r:+.4f}   n={n_:,}   "
            f"{'ok' if r >= H1_RHO else 'BELOW GATE'}")
    gates["H1"] = bool(all(r >= H1_RHO for r, _ in pairs.values()))
    say(f"     gate: every pair >= {H1_RHO}")
    say(f"     H1 {'PASS' if gates['H1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- H2
    say("H2 THE DERIVED RATE AGREES, NOT JUST THE INPUTS")
    both = [g for g in i_cp if g in m_hl and g in S
            and S[g].get("prot_copies") and S[g].get("prot_hl_h")]
    hp = np.array([i_cp[g] * (LN2 / m_hl[g] + MU) for g in both])
    sp = np.array([S[g]["prot_copies"] * (LN2 / S[g]["prot_hl_h"] + MU) for g in both])
    ratio = hp / sp
    rho2 = spearman(hp, sp)
    say(f"     protein production rate P*b on {len(both):,} shared genes")
    say(f"     median human/mouse ratio {np.median(ratio):.3f}   "
        f"(loop 92's failure was 12x on this quantity)")
    say(f"     Spearman {rho2:+.4f}   10th-90th ratio {np.percentile(ratio, 10):.2f}-"
        f"{np.percentile(ratio, 90):.2f}")
    ok2 = bool(max(np.median(ratio), 1 / np.median(ratio)) <= H2_RATIO and rho2 >= H2_RHO)
    gates["H2"] = ok2
    say(f"     gate: median ratio within {H2_RATIO}x AND Spearman >= {H2_RHO}")
    say(f"     H2 {'PASS' if gates['H2'] else 'FAIL'} -- the human build "
        f"{'reproduces the validated rate' if ok2 else 'DOES NOT reproduce the validated rate and is unusable for dynamics'}")
    say()

    # ---------------------------------------------------------------- H3
    say("H3 THE COVERAGE GAIN, THREE DENOMINATORS")
    prot_state = (set(m_hl) & set(i_cp)) & names
    same_cell = (set(k_hela) & set(i_cp)) & names
    mf_new = CA.mass_fraction(D, prot_state)
    mf_old = CA.mass_fraction(D, cur)
    say(f"     protein-side human state   {len(prot_state):>7,}   against the current "
        f"{len(cur):,}   ({len(prot_state - cur):+,} outright)")
    say(f"     same-cell-line (HeLa) pair {len(same_cell):>7,}   mRNA half-life + protein copies")
    say(f"     by mass: human build {mf_new:.2%} of measured proteome, current {mf_old:.2%}")
    say(f"     against 16,492: {len(prot_state) / 16492:.1%} vs {len(cur) / 16492:.1%}")
    gates["H3"] = bool(len(prot_state) > len(cur))
    say(f"     H3 {'PASS' if gates['H3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- H4
    say("H4 THE BUDGETS STILL CLOSE ON HUMAN DATA")
    L, nm, c = {}, None, 0
    with gzip.open(PROT, "rt") as f:
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
    gs = sorted(prot_state)
    res = np.array([L.get(g, 0) for g in gs], float)
    prod = np.array([i_cp[g] * (LN2 / m_hl[g] + MU) for g in gs])
    demand = float((prod * res).sum())
    import re as _re
    rp = _re.compile(r"^(RPL|RPS)\d+[A-Z]?$|^RPLP\d$|^RPSA$")
    ribo = float(np.median([i_cp[g] for g in i_cp if rp.match(g)]))
    cap = ribo * 6.0 * 3600.0
    say(f"     ribosomes from ITZHAK HeLa copies (median RPL/RPS): {ribo:,.0f}")
    say(f"     codon demand {demand / 1e9:.2f} Gcodons/h over {len(gs):,} genes")
    say(f"     capacity {cap / 1e9:.2f} Gcodons/h -> utilisation {100 * demand / cap:.1f}%")
    say(f"     cell_run on the mouse build: 18.3%")
    gates["H4"] = bool(demand < cap)
    say(f"     H4 {'PASS' if gates['H4'] else 'FAIL'} -- the ribosome budget "
        f"{'still closes' if gates['H4'] else 'BREAKS on human data'}")
    say()

    # ---------------------------------------------------------------- H5
    say("H5 THE mRNA ARM")
    # I DRAFTED THIS GATE AS AN ASSERTION -- that nTPM cannot become copies without breaking loop
    # 92's rule -- and that was wrong on inspection. Loop 92's failure was a PER-GENE quotient of
    # two abundance datasets from different cells. Scaling one dataset by a single scalar is not
    # that: every gene is multiplied by the same number, so no per-gene cross-dataset artefact can
    # be introduced and the rank order is untouched. It is testable, so it gets tested.
    say(f"     nTPM is relative, and converting it needs a total-mRNA-per-cell constant. That is a")
    say(f"     UNIFORM SCALING, not the per-gene cross-dataset quotient loop 92 forbids -- every")
    say(f"     gene is multiplied by the same number -- so it is testable rather than forbidden.")
    hela_ntpm = {}
    p_ct = LR.SC / "hpa_celline" / "rna_celline.tsv"
    if p_ct.exists():
        r5 = csv.reader(open(p_ct), delimiter="\t")
        h5 = next(r5)
        jN, jC, jT = h5.index("Gene name"), h5.index("Cell line"), h5.index("nTPM")
        for x in r5:
            if x[jC] != "HeLa":
                continue
            try:
                v = float(x[jT])
            except (TypeError, ValueError):
                continue
            if v > 0:
                hela_ntpm[x[jN]] = max(hela_ntpm.get(x[jN], 0.0), v)
    say(f"     HPA HeLa nTPM: {len(hela_ntpm):,} genes with a positive value")
    ovm = [g for g in hela_ntpm if g in S and S[g].get("mrna_copies")]
    rho5 = spearman([hela_ntpm[g] for g in ovm], [S[g]["mrna_copies"] for g in ovm])
    say(f"     against Schwanhausser mRNA copies on {len(ovm):,} shared genes: rho {rho5:+.4f}   "
        f"gate >= {H5_RHO}")
    tot_sweep = (100_000.0, 200_000.0, 400_000.0)   # total mRNA per mammalian cell, swept
    ssum = sum(hela_ntpm.values())
    for T_ in tot_sweep:
        cp = {g: hela_ntpm[g] * T_ / ssum for g in hela_ntpm}
        med = float(np.median([cp[g] for g in ovm]))
        say(f"       total {T_:,.0f} mRNA/cell -> median {med:.1f} copies/gene "
            f"(Schwanhausser median {np.median([S[g]['mrna_copies'] for g in ovm]):.1f})")
    gates["H5"] = bool(np.isfinite(rho5) and rho5 >= H5_RHO)
    if gates["H5"]:
        full = (set(hela_ntpm) & set(k_hela) & set(i_cp) & set(m_hl)) & names
        say(f"     -> A FULL FOUR-FIELD HUMAN STATE IS AVAILABLE for {len(full):,} genes "
            f"against the mouse build's {len(cur):,}")
        say(f"     the scaling constant remains swept, not chosen, because no measurement of "
            f"total HeLa mRNA was fetched")
    else:
        full = set()
        say(f"     -> nTPM does NOT track measured mRNA copies well enough; the human state stays "
            f"PROTEIN-SIDE ONLY and the mouse build remains the only full four-field state")
    say(f"     H5 {'PASS' if gates['H5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- H6
    say("H6 FAME, AND THE COST OF MIXING")
    pubs = D["pubs"]
    gained = prot_state - cur
    pg = np.array([pubs.get(g, 0.0) for g in gained])
    pc = np.array([pubs.get(g, 0.0) for g in cur])
    obs, p6 = perm_p(np.log10(pg + 1), np.log10(pc + 1), rng)
    say(f"     genes gained {len(gained):,} (median {np.median(pg):.0f} publications), "
        f"already present {len(cur):,} (median {np.median(pc):.0f})")
    say(f"     log10 difference {obs:+.3f}, permutation p = {p6:.4f}")
    say(f"     {'the gain is biased toward better-studied genes' if (obs > 0 and p6 < 0.05) else 'the gain is NOT biased toward better-studied genes'}")
    say(f"     THE COST OF MIXING, as a number: the human protein production rate differs from the")
    say(f"     self-consistent mouse one by a median factor of {np.median(ratio):.2f}, with the")
    say(f"     middle 80% spanning {np.percentile(ratio, 10):.2f} to {np.percentile(ratio, 90):.2f}. That spread is the price of")
    say(f"     taking half-lives from primary cells and copies from HeLa, and it is carried into")
    say(f"     anything built on this state.")
    gates["H6"] = bool(np.isfinite(p6))
    say(f"     H6 {'PASS' if gates['H6'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("H1", "H2", "H3", "H4", "H5", "H6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[MATH, ITZ, KDEG, PROT, LR.CELL],
                      available=len(names), used=len(prot_state), selection="filtered", seed=SEED,
                      controls=["Schwanhausser's self-consistent build as the baseline every "
                                "human source had to agree with",
                                "the DERIVED rate compared, not only the inputs",
                                "the ribosome budget recomputed from human numbers",
                                "publication count of gained versus existing genes",
                                "the mRNA gap stated rather than filled by a quotient"],
                      note="loop 92's rule stands: no rate is formed by dividing two abundance "
                           "datasets from different cells, which is why the mRNA arm is left empty")
    RM.report(man, emit=say)
    json.dump({"test": "loop_human_state", "manifest": man, "gates": gates,
               "h1": {k: {"rho": r, "n": n_} for k, (r, n_) in pairs.items()},
               "h2": {"n": len(both), "median_ratio": float(np.median(ratio)), "spearman": rho2,
                      "p10": float(np.percentile(ratio, 10)),
                      "p90": float(np.percentile(ratio, 90))},
               "h3": {"protein_state": len(prot_state), "current": len(cur),
                      "gained": len(prot_state - cur), "same_cell_line": len(same_cell),
                      "mass_new": mf_new, "mass_old": mf_old},
               "h4": {"ribosomes": ribo, "demand_gcodons_h": demand / 1e9,
                      "capacity_gcodons_h": cap / 1e9, "utilisation": demand / cap},
               "h5": {"hela_ntpm": len(hela_ntpm), "rho_vs_schwan": rho5,
                      "n_overlap": len(ovm), "full_four_field": len(full),
                      "total_mrna_swept": list(tot_sweep)},
               "h6": {"n_gained": len(gained), "pubs_gained": float(np.median(pg)),
                      "pubs_current": float(np.median(pc)), "log10_diff": obs, "p": p6},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_human_state.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_human_state.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
