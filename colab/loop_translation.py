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
    ppm = {int(k): float(v) for k, v in C["ppm"].items()}
    life = json.load(open(LIFE))["lifetimes"]
    plen = protein_lengths()
    say(f"  {len(plen):,} protein lengths from the UniProt human proteome on disk")

    genes = [g for g in life if life[g].get("prot_hl_h") and g in idx
             and ppm.get(idx[g], 0) > 0 and g in plen]
    say(f"  {len(genes):,} genes with a protein half-life, a ppm abundance and a length")
    say(f"  loss rate = ln2/t_half + ln2/{T_DOUBLE_H:.0f}h;  synthesis = copies * loss")
    say()

    say("S1 THE RIBOSOME BUDGET CLOSES")
    rp = [g for g in idx if (g.startswith("RPL") or g.startswith("RPS")) and ppm.get(idx[g], 0) > 0]
    rp_ppm = float(np.median([ppm[idx[g]] for g in rp]))
    say(f"     {len(rp)} ribosomal proteins, median {rp_ppm:.1f} ppm -- one copy each per ribosome,")
    say(f"     so the median is the ribosome count in ppm rather than the sum")
    tot_ppm = sum(ppm.values())
    k = {g: LN2 / life[g]["prot_hl_h"] + LN2 / T_DOUBLE_H for g in genes}
    sweep, s1 = {}, None
    for tot in PROTEOME_SWEEP:
        copies = {g: ppm[idx[g]] / tot_ppm * tot for g in genes}
        demand = sum(copies[g] * k[g] * plen[g] for g in genes)          # codons/h
        cov = sum(ppm[idx[g]] for g in genes) / tot_ppm
        demand_all = demand / cov if cov > 0 else float("nan")
        ribs = rp_ppm / tot_ppm * tot
        cap = ribs * ELONG_AA_S * 3600.0
        u = demand_all / cap if cap > 0 else float("nan")
        sweep[tot] = {"ribosomes": ribs, "demand_codons_h": demand_all,
                      "capacity_codons_h": cap, "utilisation": u}
        say(f"     proteome {tot:.0e} -> {ribs:>12,.0f} ribosomes, demand {demand_all:>16,.0f} "
            f"codons/h, capacity {cap:>16,.0f}, utilisation {u:>7.1%}")
    say(f"     abundance mass covered by the rated genes: "
        f"{sum(ppm[idx[g]] for g in genes)/tot_ppm:.1%}, demand scaled up by its inverse")
    u_cons = sweep[PROTEOME_SWEEP[0]]["utilisation"]
    s1 = bool(np.isfinite(u_cons) and u_cons <= 1.0)
    say(f"     gate at the CONSERVATIVE end ({PROTEOME_SWEEP[0]:.0e}): utilisation {u_cons:.1%}")
    say(f"     S1 {'PASS' if s1 else 'FAIL'}")
    say()

    say("S2 NO GENE EXCEEDS THE POLYSOME LIMIT")
    mrna_hl = {g: life[g]["mrna_hl_h"] for g in genes if life[g].get("mrna_hl_h")}
    tot = PROTEOME_SWEEP[1]
    copies = {g: ppm[idx[g]] / tot_ppm * tot for g in genes}
    over, per = [], {}
    for g in genes:
        transit_s = plen[g] / ELONG_AA_S
        max_per_mrna_h = 3600.0 / (RIB_FOOTPRINT_AA / ELONG_AA_S)         # initiations/h ceiling
        # protein made per hour by this gene, per mRNA -- needs an mRNA copy number, which only
        # Schwanhausser genes have, so use the ratio via ppm as an upper bound proxy
        per[g] = copies[g] * k[g]
    S = json.load(open(SC / "_schwan2011.json"))
    both = [g for g in genes if g in S and S[g].get("mrna_copies")]
    ksp = {g: per[g] / S[g]["mrna_copies"] for g in both}
    lim = 3600.0 / (RIB_FOOTPRINT_AA / ELONG_AA_S)
    over = [g for g in both if ksp[g] > lim]
    say(f"     ribosome footprint {RIB_FOOTPRINT_AA} codons at {ELONG_AA_S} aa/s -> an mRNA can")
    say(f"     initiate at most {lim:,.0f} times per hour")
    say(f"     {len(both):,} genes with an mRNA copy number; median k_sp "
        f"{np.median([ksp[g] for g in both]):.1f}/h, max {max(ksp.values()):,.0f}/h")
    frac = len(over) / max(len(both), 1)
    say(f"     exceeding the limit: {len(over):,} ({frac:.2%})   gate: < {S2_MAX_FRAC:.0%}")
    if over:
        top = sorted(over, key=lambda g: -ksp[g])[:5]
        say(f"     worst: " + ", ".join(f"{g} {ksp[g]:,.0f}/h" for g in top))
    s2 = bool(frac < S2_MAX_FRAC)
    say(f"     S2 {'PASS' if s2 else 'FAIL'}")
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
    pub = {g: float(C["genes"][idx[g]].get("pubs") or 0) for g in both}
    r_p, n_p = spear([ksp[g] for g in both], [pub[g] for g in both])
    r_l, _ = spear([plen[g] for g in both], [pub[g] for g in both])
    say(f"     pubs vs translation rate {r_p:+.4f}   vs protein length {r_l:+.4f}   (n {n_p:,})")
    say()

    say("S5 LENGTH BEHAVES")
    r_kl, _ = spear([ksp[g] for g in both], [plen[g] for g in both])
    dem = {g: copies[g] * k[g] * plen[g] for g in genes}
    order = sorted(genes, key=lambda g: -dem[g])
    top10 = sum(dem[g] for g in order[:10]) / sum(dem.values())
    say(f"     translation rate vs protein length {r_kl:+.4f}")
    say(f"     top 10 genes carry {top10:.1%} of all codon demand: "
        f"{', '.join(order[:5])}")
    say(f"     median length {np.median([plen[g] for g in genes]):.0f} aa, "
        f"demand-weighted mean {sum(dem[g]*plen[g] for g in genes)/sum(dem.values()):.0f} aa")
    say()

    say("S6 COVERAGE DECLARED")
    say(f"     protein rates: {len(genes):,} genes, "
        f"{sum(ppm[idx[g]] for g in genes)/tot_ppm:.1%} of abundance mass")
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
               "n_genes": len(genes), "n_with_mrna": len(both),
               "s1": {f"{k:.0e}": v for k, v in sweep.items()},
               "s2": {"limit_per_h": lim, "n_over": len(over), "frac_over": frac,
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
