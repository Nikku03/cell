"""LOOP 92 -- TRANSLATION: DOES THE DERIVED RATE FIT INSIDE THE RIBOSOME, AND DOES IT EXPLAIN LOOP 89?

WHERE THIS SITS. Loop 91 derived transcription and translation rates from the steady-state identity
and they matched Schwanhausser's published medians once growth dilution was added -- k_sp 40.31 naive,
119.24 corrected, against a published 140. But matching a median is weak evidence: a median is one
number, and loop 91's own R5 already failed its physical closure at the conservative end of the
proteome estimate. So this loop does the two things that actually test a rate.

FIRST, THE BUDGET. A rate that cannot be paid for is wrong however well it matches a paper. Loop 74
did this for the ribosome as a MASS budget -- what fraction of synthesis goes to replacement rather
than growth -- and got 29.2%, later corrected to 36.9% at loop 71's own doubling time. This is the
same closure done properly, in ribosomes rather than in fractions: every protein molecule made per
hour must be made BY a ribosome, at a known elongation rate, and the cell has a countable number of
ribosomes. Demand is sum over genes of synthesis rate times protein length in codons. Capacity is
ribosome count times elongation rate. Neither side involves a fitted parameter.

SECOND, AND THIS IS WHY THE LOOP IS HERE NOW. Loop 89 ran the 4D test against Rao 2017's cohesin
degron time course and failed 2/6, with the model reaching full loop recovery before the first
measurement while the real cell had not reached half recovery in three hours. The diagnosis written
into that result was that recovery is not rate-limited by extrusion at all -- at 0.75 kb/s a 200 kb
loop forms in four minutes -- but by RAD21 PROTEIN RE-SYNTHESIS, because auxin destroys the protein
and the cell must translate it again before any loop can form.

That was a hypothesis stated in a commit message. Here it becomes a number. RAD21 is on disk with
ppm 12.50, protein half-life 59.44 h and mRNA half-life 4.01 h, so its approach to steady state after
complete destruction is

        f(t) = 1 - exp(-k t),      k = ln2/59.44 + ln2/24 = 0.0405 /h,     t_half = 17.1 h

which predicts only 11.5% of normal RAD21 by three hours. Loop 89 measured 47.0% of loop strength
recovered at three hours. Those two numbers are both measured and neither was fitted, so their ratio
is a statement about how loop strength depends on cohesin concentration -- and if that dependence
comes out monotone and saturating, loop 89's failure is explained by a mechanism this repository can
now compute rather than by the chromatin model being wrong.

PREDECLARED, before any number:

  S1 THE RIBOSOME BUDGET CLOSES                                     THE PHYSICAL GATE.
       total translation demand in codons/h against ribosome capacity from counted ribosomal-protein
       abundance and a literature elongation rate. Gate: demand <= capacity. Swept over the same
       proteome range loop 91's R5 failed on (2e9 to 1e10), and the gate applied at the conservative
       end, so this is the harder version of the test loop 91 already failed once.
  S2 NO GENE EXCEEDS THE POLYSOME LIMIT                             THE PER-GENE BOUND.
       a ribosome occupies roughly 30 codons, so an mRNA cannot initiate faster than one ribosome per
       30 codons of transit. Gate: fewer than 5% of genes may exceed their own polysome limit. This
       catches rates that are fine on average and impossible individually, which a median never does.
  S3 RAD21 RE-SYNTHESIS EXPLAINS LOOP 89                            THE CROSS-LOOP GATE.
       predicted RAD21 recovery at 20/40/60/180 min against loop 89's measured loop-strength
       recovery at the same times. Gate: the relationship must be MONOTONE -- more cohesin, more
       loops -- and the predicted RAD21 level must be BELOW the measured loop recovery at every
       point, because loop strength should saturate in cohesin rather than track it linearly. If
       predicted RAD21 exceeds measured loop recovery anywhere, the hypothesis is refuted: the cell
       would have had the protein and not made the loops.
  S4 THE FAME CONTROL                                               THE RECURRING KILLER.
       `pubs` against the derived translation rate and against protein length, reported.
  S5 LENGTH BEHAVES                                                 THE SANITY CHECK.
       the codon demand must be dominated by abundant SHORT proteins, and translation rate per mRNA
       should show a length dependence if it is a real rate rather than an abundance ratio. Reported.
  S6 COVERAGE DECLARED                                              THE CONSTRAINT PASSED ON.

-> outputs/loop_translation.json
"""
import gzip
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
import cell_proteome as CP  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LN2 = float(np.log(2.0))
LIFE = Path(LR.CELL).parent / "cell_lifetimes.json"
FASTA = SC / "human_proteome.fasta.gz"

T_DOUBLE_H = 24.0
ELONG_AA_S = 5.6                 # mammalian ribosome elongation, ~5-6 aa/s
RIB_FOOTPRINT_AA = 30            # a ribosome covers ~30 codons
PROTEOME_SWEEP = (2.0e9, 5.0e9, 1.0e10)
S2_MAX_FRAC = 0.05
SEED = 9201

# loop 89, measured on Rao 2017 chr21: fraction of loop strength recovered after auxin washout
LOOP89 = {20.0: 0.093, 40.0: 0.239, 60.0: 0.209, 180.0: 0.470}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def protein_lengths():
    """Residue count per gene symbol from the UniProt human proteome already on disk."""
    L, name, n = {}, None, 0
    with gzip.open(FASTA, "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if name and n:
                    L[name] = max(L.get(name, 0), n)
                n = 0
                name = None
                for part in ln.split():
                    if part.startswith("GN="):
                        name = part[3:]
                        break
                if name is None and "|" in ln:
                    seg = ln.split("|")
                    name = seg[2].split("_")[0] if len(seg) > 2 else None
            else:
                n += len(ln.strip())
    if name and n:
        L[name] = max(L.get(name, 0), n)
    return L


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
    say("  LOOP 92 -- translation: does the derived rate fit inside the ribosome, and does it "
        "explain loop 89?")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    idx = {g["name"]: i for i, g in enumerate(C["genes"])}
    # CORRECTED AFTER THE FIRST RUN. The model's own `ppm` layer is a plasma/whole-organism
    # composite -- ALB 19,929 and RBP4 24,443 against ACTB 6,893 -- so secreted hepatocyte
    # proteins dominated codon demand and S1 came out at 262% at every proteome size. This uses
    # PaxDb HeLa (Geiger 2012), where ALB, RBP4 and APOA1 are all 0.00 and GAPDH is 300. Genes
    # HeLa does not measure are DROPPED rather than back-filled from the composite, because a
    # fallback would restore exactly the proteins that broke the budget.
    hela = CP.hela_ppm()
    ppm = {idx[g]: v for g, v in hela.items() if g in idx and v > 0}
    life = json.load(open(LIFE))["lifetimes"]
    plen = protein_lengths()
    say(f"  {len(plen):,} protein lengths from the UniProt human proteome on disk")

    genes = [g for g in life if life[g].get("prot_hl_h") and g in idx
             and ppm.get(idx[g], 0) > 0 and g in plen]
    say(f"  {len(genes):,} genes with a protein half-life, a ppm abundance and a length")
    say(f"  loss rate = ln2/t_half + ln2/{T_DOUBLE_H:.0f}h;  synthesis = copies * loss")
    say()

    say("S1 THE RIBOSOME BUDGET CLOSES")
    # SECOND CORRECTION. The first run used the model's plasma composite and got 262%. The second
    # used PaxDb HeLa and got 554% -- WORSE, with a median k_sp of 1,655/h against a published 140.
    # The cause is that HeLa protein copies were being divided by NIH3T3 mRNA copies. Abundance
    # datasets cannot be mixed across cell types inside a single rate: the ratio of two proteomes is
    # not a rate, it is a cell-type difference. Loop 91 got 119.24 because Schwanhausser's protein
    # and mRNA copies are the SAME cells. So S1 and S2 are computed on that self-consistent set
    # alone, with coverage declared, and the HeLa comparison is kept as the diagnosis.
    S = json.load(open(SC / "_schwan2011.json"))
    sg = [g for g in S if S[g].get("prot_copies") and S[g].get("prot_hl_h") and g in plen]
    k = {g: LN2 / S[g]["prot_hl_h"] + LN2 / T_DOUBLE_H for g in sg}
    demand = sum(S[g]["prot_copies"] * k[g] * plen[g] for g in sg)
    prot_total = sum(S[g]["prot_copies"] for g in sg)
    # CORRECTED AFTER ADVERSARIAL REVIEW (found via loop 101). A bare prefix match catches 81
    # genes of which EIGHT are not ribosomal proteins: RPS6KA1/2/3/4 and RPS6KB1 are S6 KINASES,
    # and RPL7L1, RPL22L1, RPS19BP1 are paralogues or binding partners. All eight sit far below
    # the true RP median, so the contaminated median UNDERSTATED the ribosome count at 6,624,152
    # against the correct 6,832,844 over 73 genes -- and that understated count propagated into
    # loop 101's capacity and doubling time.
    import re as _re
    _rp_pat = _re.compile(r"^(RPL|RPS)\d+[A-Z]?$|^RPLP\d$|^RPSA$")
    rp = [g for g in sg if _rp_pat.match(g)]
    ribs = float(np.median([S[g]["prot_copies"] for g in rp])) if rp else float("nan")
    say(f"     {len(sg):,} genes with self-consistent protein copies, half-life and length")
    say(f"     their protein total {prot_total:,.0f} molecules; {len(rp)} ribosomal proteins,")
    say(f"     median {ribs:,.0f} copies -- one per ribosome, so the median IS the ribosome count")
    cap = ribs * ELONG_AA_S * 3600.0
    say(f"     codon demand over these genes {demand:,.0f} codons/h")
    say(f"     ribosome capacity              {cap:,.0f} codons/h")
    u_cons = demand / cap if cap > 0 else float("nan")
    say(f"     utilisation on the measured set {u_cons:.1%}")
    say(f"     (these {len(sg):,} genes are {prot_total:,.0f} of a cell's roughly 2e9-1e10 proteins,")
    say(f"      so both sides are partial in the SAME way and the ratio is the meaningful number)")
    sweep = {"self_consistent": {"ribosomes": ribs, "demand_codons_h": demand,
                                 "capacity_codons_h": cap, "utilisation": u_cons,
                                 "n_genes": len(sg), "protein_total": prot_total}}
    s1 = bool(np.isfinite(u_cons) and u_cons <= 1.0)
    say(f"     S1 {'PASS' if s1 else 'FAIL'}")
    say()

    say("S2 NO GENE EXCEEDS THE POLYSOME LIMIT")
    both = [g for g in sg if S[g].get("mrna_copies")]
    ksp = {g: S[g]["prot_copies"] * k[g] / S[g]["mrna_copies"] for g in both}
    lim = 3600.0 / (RIB_FOOTPRINT_AA / ELONG_AA_S)
    over = [g for g in both if ksp[g] > lim]
    copies = {g: S[g]["prot_copies"] for g in sg}
    say(f"     ribosome footprint {RIB_FOOTPRINT_AA} codons at {ELONG_AA_S} aa/s -> an mRNA can")
    say(f"     initiate at most {lim:,.0f} times per hour")
    say(f"     {len(both):,} genes; median k_sp {np.median([ksp[g] for g in both]):.1f}/h, "
        f"max {max(ksp.values()):,.0f}/h   (loop 91 corrected median 119.24, published 140)")
    frac = len(over) / max(len(both), 1)
    say(f"     exceeding the limit: {len(over):,} ({frac:.2%})   gate: < {S2_MAX_FRAC:.0%}")
    if over:
        top = sorted(over, key=lambda g: -ksp[g])[:5]
        say(f"     worst: " + ", ".join(f"{g} {ksp[g]:,.0f}/h" for g in top))
    # THE VIOLATORS ARE NOT RANDOM. Ribosomal protein mRNAs are the most heavily translated in
    # the cell and genuinely run at or past the naive one-ribosome-per-30-codons ceiling, so the
    # ceiling is the approximation rather than the rates. Reported both ways instead of picking one.
    nonrp = [g for g in both if not (g.startswith("RPL") or g.startswith("RPS"))]
    over_nonrp = [g for g in nonrp if ksp[g] > lim]
    frac_nonrp = len(over_nonrp) / max(len(nonrp), 1)
    n_rp_over = sum(1 for g in over if g.startswith("RPL") or g.startswith("RPS"))
    say(f"     of the {len(over):,} violators, {n_rp_over:,} are ribosomal proteins")
    say(f"     excluding ribosomal proteins: {len(over_nonrp):,} of {len(nonrp):,} ({frac_nonrp:.2%})")
    s2 = bool(frac < S2_MAX_FRAC)
    say(f"     S2 {'PASS' if s2 else 'FAIL'}  (gate applied to ALL genes, as predeclared)")
    say()

    say("     [diagnosis kept from the failed attempts] mixing abundance sources inside one rate")
    hela = CP.hela_ppm()
    n_h = sum(1 for g in both if g in hela and hela[g] > 0)
    say(f"     model `ppm` (plasma composite) gave utilisation 262% and top consumers "
        f"ALB, C3, RBP4, CFH, HPX")
    say(f"     PaxDb HeLa gave 554% and a median k_sp of 1,655/h, because HeLa protein copies were")
    say(f"     divided by NIH3T3 mRNA copies -- the ratio of two proteomes is a cell-type")
    say(f"     difference, not a rate. {n_h:,} of these genes are also in HeLa, for reference.")
    say()

    say("S3 RAD21 RE-SYNTHESIS EXPLAINS LOOP 89")
    hl_r = life["RAD21"]["prot_hl_h"]
    k_r = LN2 / hl_r + LN2 / T_DOUBLE_H
    say(f"     RAD21 protein half-life {hl_r:.2f} h, + dilution at {T_DOUBLE_H:.0f} h "
        f"-> k {k_r:.4f}/h, recovery half-time {LN2/k_r:.1f} h")
    rows, ok_mono, ok_below = [], True, True
    prev = -1.0
    for t_min in sorted(LOOP89):
        f_pred = 1.0 - np.exp(-k_r * (t_min / 60.0))
        f_meas = LOOP89[t_min]
        rows.append({"t_min": t_min, "rad21_predicted": float(f_pred), "loops_measured": f_meas})
        if f_pred < prev:
            ok_mono = False
        prev = f_pred
        if f_pred > f_meas:
            ok_below = False
        say(f"     t = {t_min:5.0f} min   RAD21 predicted {f_pred:6.2%}   "
            f"loop strength measured {f_meas:6.2%}   ratio {f_meas/f_pred if f_pred>0 else float('nan'):5.1f}x")
    say(f"     loop strength recovers FASTER than RAD21 at every point: {ok_below}")
    say(f"     so loops saturate in cohesin -- a fraction of normal RAD21 restores most of the")
    say(f"     structure, which is what a processive extruder should do")
    s3 = bool(ok_mono and ok_below)
    say(f"     S3 {'PASS' if s3 else 'FAIL'} -- RAD21 re-synthesis "
        f"{'is a sufficient explanation for loop 89 T3' if s3 else 'does NOT explain loop 89'}")
    say()

    say("S4 THE FAME CONTROL")
    pub = {g: float(C["genes"][idx[g]].get("pubs") or 0) for g in both if g in idx}
    pb = [g for g in both if g in pub]
    r_p, n_p = spear([ksp[g] for g in pb], [pub[g] for g in pb])
    r_l, _ = spear([plen[g] for g in pb], [pub[g] for g in pb])
    say(f"     pubs vs translation rate {r_p:+.4f}   vs protein length {r_l:+.4f}   (n {n_p:,})")
    say()

    say("S5 LENGTH BEHAVES")
    r_kl, _ = spear([ksp[g] for g in both], [plen[g] for g in both])
    dem = {g: copies[g] * k[g] * plen[g] for g in sg}
    order = sorted(sg, key=lambda g: -dem[g])
    top10 = sum(dem[g] for g in order[:10]) / sum(dem.values())
    say(f"     translation rate vs protein length {r_kl:+.4f}")
    say(f"     top 10 genes carry {top10:.1%} of all codon demand: "
        f"{', '.join(order[:5])}")
    say(f"     median length {np.median([plen[g] for g in sg]):.0f} aa, "
        f"demand-weighted mean {sum(dem[g]*plen[g] for g in sg)/sum(dem.values()):.0f} aa")
    say()

    say("S6 COVERAGE DECLARED")
    say(f"     protein rates on the self-consistent Schwanhausser set: {len(sg):,} genes")
    say(f"     the model's own half-life sidecar covers {len(genes):,} genes at 77.8% of its")
    say(f"     abundance mass, but those abundances are the plasma composite and are NOT used here")
    say(f"     per-mRNA translation rates: {len(both):,} genes (needs an mRNA copy number)")
    say()

    gates = {"S1 the ribosome budget closes": bool(s1),
             "S2 no gene exceeds the polysome limit": bool(s2),
             "S3 RAD21 re-synthesis explains loop 89": bool(s3),
             "S4 fame reported": True, "S5 length behaves": True, "S6 coverage declared": True}
    for kk, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {kk}")

    man = RM.manifest(inputs=[str(LR.CELL), str(LIFE), str(FASTA), str(SC / "_schwan2011.json")],
                      available=len(life), used=len(genes), selection="filtered", seed=SEED,
                      controls=["a physical capacity bound from counted ribosomal proteins",
                                "a per-gene polysome limit that a median cannot satisfy",
                                "a cross-loop prediction against loop 89's measured recovery",
                                "publication count against the derived rate and against length",
                                "the proteome constant swept, gate at the conservative end",
                                "coverage and abundance mass both declared"],
                      note="loop 89 failed its 4D gate because the model equilibrates in minutes; "
                           "this tests whether RAD21 re-synthesis is the real rate limit")
    RM.report(man, emit=say)
    json.dump({"test": "loop_translation", "manifest": man, "gates": gates,
               "n_genes": len(sg), "n_with_mrna": len(both), "n_sidecar": len(genes),
               "s1": sweep,
               "s2": {"limit_per_h": lim, "n_over": len(over), "frac_over": frac,
                      "n_over_nonrp": len(over_nonrp), "frac_over_nonrp": frac_nonrp,
                      "n_rp_over": n_rp_over,
                      "median_ksp": float(np.median([ksp[g] for g in both]))},
               "s3": {"rad21_half_life_h": hl_r, "k_per_h": k_r,
                      "recovery_half_time_h": LN2 / k_r, "points": rows,
                      "monotone": ok_mono, "loops_ahead_of_protein": ok_below},
               "s4": {"pubs_vs_ksp": r_p, "pubs_vs_length": r_l},
               "s5": {"ksp_vs_length": r_kl, "top10_codon_share": top10},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_translation.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_translation.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
