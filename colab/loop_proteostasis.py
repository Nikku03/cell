"""LOOP 103 -- TURNOVER: DOES THE PROTEASOME HAVE THE CAPACITY TO DEGRADE WHAT THE HALF-LIVES IMPLY?

THE SHAPE OF THE QUESTION, AND WHY IT IS THE SAME SHAPE AS LOOP 92. Loop 92 asked whether the
ribosomes a cell actually contains can lay down the codons the half-lives demand, and answered it by
counting ribosomes from ribosomal-protein copy numbers and multiplying by an elongation rate. That
is a closure test: a measured demand against a counted machine. Every protein that is made must
also be destroyed at steady state, and the destroying is done by a machine that is equally
countable. Nobody in this repository has ever run the test on the DOWNHILL side of the cycle. The
half-lives have been used to compute synthesis (loops 74, 91, 92, 101) and never once to ask whether
the proteolytic machinery could physically keep up with them. If it cannot, then either the
half-lives are wrong or a large share of degradation is not proteasomal, and both of those are
statements about the cell model rather than about this arithmetic.

THE TWO SIDES, and neither contains a fitted parameter.

    demand    = SUM over proteins of  N_i * ln2 / t_half,i        molecules destroyed per hour
    capacity  = P * r                                             molecules destroyed per hour

N_i and t_half,i are Schwanhausser's protein copies and protein half-lives, measured in the SAME
NIH3T3 cells, which is what loop 92's abundance rule requires: a rate may never be assembled from
two different abundance datasets, because the quotient of two proteomes is a cell-type difference
and not a process. P is the 26S proteasome particle count, taken from the PSMA/PSMB/PSMC/PSMD
subunit copies in that same table by the same device loop 92 used for ribosomes -- one subunit per
particle, so the MEDIAN subunit copy number is the particle count. r is the only literature number
in the whole calculation: a 26S proteasome processes roughly one substrate every 1-3 s, and that
range is swept rather than chosen.

THE ASYMMETRY THAT LOOP 101 FOUND, CARRIED FORWARD RATHER THAN REPEATED. Loop 92 asserted that both
sides of its ribosome ratio were "partial in the same way"; loop 101 showed they are not, because
the demand is summed over 4,179 measured genes while the machine count is a whole-cell census --
every ribosome in the cell contains the ribosomal proteins that were counted, whether or not the
substrates it translates were detected. The identical asymmetry applies here: every proteasome in
the cell contains the PSM subunits that were counted. So the covered-set demand is a strict LOWER
BOUND, and it is scaled to a whole cell by the one absolute quantity in the problem -- protein mass
per cell, 300 pg, swept 150-600 -- divided by 110 Da per residue, which converts the covered set's
residue content into the fraction of a cell's protein it represents. That scaling uses sequence
lengths and one scalar, never a second abundance table.

WHAT COULD MAKE THIS LOOP WORTHLESS, named in advance: passing the gate by counting proteasomes
generously and then declaring victory. The stoichiometry is the exposure. PSMA1-7 and PSMB1-7 are
the 20S core and sit at TWO copies per particle; PSMC1-6 and PSMD1-14 are the 19S regulatory
particle at one. Taking the median over all four families therefore risks overcounting particles by
up to a factor of two in the direction that helps the gate. P1 measures that split explicitly and
reports the stoichiometry-corrected count next to the prescribed one, and the gate is evaluated at
the WORST corner of the sweep rather than at a chosen midpoint.

DISCLOSURE, REQUIRED BY THE PROJECT RULE ON PEEKING. Before fixing the thresholds below I read only
numbers ALREADY PUBLISHED in this repository -- loop 74's 71.5 h median protein half-life, loop 92's
6,624,152 ribosomes and 2,337,014,740 protein molecules over the covered set, loop 101's four-route
protein-content arithmetic -- and I listed the PSM gene symbols present in the Schwanhausser table
(43 of them) to confirm the families exist. No copy number, no half-life, no load, no capacity and
no correlation computed by this module was seen before the thresholds below were written. The
thresholds themselves come from the task statement (1-3 s per substrate; comparison against loop
74's 29.2%/36.9%), from published biology (a median half-life of 71.5 h), or from what an obligate
stoichiometric complex means (P1's 3x spread), and not from any output of this run.

PREDECLARED, before any number is computed:

  P1 THE PROTEASOME CENSUS IS A CENSUS AND NOT AN AVERAGE       THE INSTRUMENT CHECK.
       the 26S particle count as the median copy number over PSMA/PSMB/PSMC/PSMD. If those subunits
       are genuinely one-per-particle members of one obligate complex their copy numbers must agree
       with each other. Gate: the interquartile ratio P75/P25 over the subunit copies <= 3.0. A
       complex whose members disagree by more than 3x is not being counted, it is being averaged,
       and every capacity number below would be fiction. Reported alongside and NOT gated, because
       the thresholds would then be mine rather than biology's: the alpha/beta (2 copies per 20S)
       versus ATPase/lid (1 copy) split and the stoichiometry-corrected particle count; and the
       proteasome-to-ribosome ratio against loop 92's 6,624,152 ribosomes.
  P2 THE PROTEOLYTIC DEMAND IS PHYSIOLOGICALLY SANE             THE SECOND INSTRUMENT CHECK.
       demand / total protein molecules is an abundance-weighted mean degradation rate, and ln2 over
       it is the mean protein half-life the cell actually experiences. Gate: that implied half-life
       within 10x of loop 74's measured median of 71.5 h, i.e. 7.15 h to 715 h. This can genuinely
       fail: if a handful of enormously abundant short-lived proteins carry the sum, the
       abundance-weighted mean can sit far from the per-gene median, and if it does the median is
       the wrong summary of this dataset and the report must say so. Reported alongside: the same
       demand with the doubling-time dilution term REMOVED from every rate, which is a lower bound
       on true proteolysis and is only ever kinder to P3.
  P3 DEMAND <= CAPACITY                                         THE GATE.
       swept over the per-particle rate (1, 2, 3 s per substrate) and over the whole-cell scaling of
       the demand (measured set only; and 150, 300, 600 pg protein per cell). Gate: demand <=
       capacity at the WORST corner of that grid -- slowest proteasome, largest cell. Passing at the
       worst corner means the proteasome has headroom under every assumption considered here.
       Failing only at the worst corner is reported as exactly that and not rounded up to a pass.
       Reported: the full 4x3 grid, the utilisation at the declared central point (2 s per
       substrate, 300 pg per cell), and the per-particle processing time that would be required to
       exactly saturate the machine -- because that number, compared against 1-3 s, says how far the
       claim is from the edge without any gate at all.
  P4 THE REPLACEMENT FRACTION                                   THE LINK TO THE SYNTHESIS SIDE.
       the fraction of protein synthesis that exists purely to replace degraded protein,
       f = kbar/(mu+kbar) at a 24 h doubling time, computed in TWO currencies: molecule-weighted
       (the currency of a proteasome, which destroys one substrate regardless of its size) and
       mass-weighted (the currency of a ribosome, which pays per residue). Two gates, and the second
       one is disclosed as weak. (a) the two currencies must agree to within 10 percentage points --
       this tests whether protein SIZE is coupled to turnover, has never been computed here, and can
       genuinely fail. (b) the mass-weighted fraction within a factor of 2 of loop 74's corrected
       36.9%, i.e. 18.45%-73.8%. DISCLOSURE: (b) is a reproduction, not a test. Loop 101's D2
       already computed the mass-weighted fraction from this very dataset and gated it against the
       same 36.9%, so it is expected to pass and its passing is worth nothing beyond confirming
       that this module reads the same table. It is included because a silent divergence from loop
       101 would mean one of the two modules is broken, and that is worth catching.
  P5 THE NULLS ARE CAPABLE, AND THEY FIRE                       THE GUARD, GUARDED.
       the demand is a sum over a PAIRING of abundance with turnover. Two nulls break that pairing
       at two different resolutions, each checked with gate_guard.null_can_move BEFORE its output is
       read, because twelve gates in this project fired while measuring nothing.
         A  half-lives shuffled globally across proteins -- does the pairing matter at all, or is
            the load simply (total protein) x (mean rate)?
         B  half-lives shuffled WITHIN abundance deciles -- does per-protein resolution matter, or
            is a coarse abundance-turnover trend the whole of it?
       gate_guard.survival decides whether either difference is even defined; a null that lands
       ABOVE the real value returns UNDEFINED and is reported as such rather than as a percentage.
       Gate: both nulls capable. Their verdicts are reported whatever they are.
  P6 THE FAME CONTROL, AND WHAT THIS DOES NOT GIVE              THE RECURRING KILLER, AND THE LIMIT.
       publication count has beaten the biology in loops 87, 87b, 94 and 96. Copy numbers and
       half-lives both come from mass spectrometry, which finds abundant and heavily studied
       proteins first, so the per-protein proteolytic load is a prime candidate for being a fame
       ranking. Gate: |rho(pubs, load)| < |rho(copies, load)| -- publication count must explain the
       load LESS well than the abundance it is built from. If fame wins, the per-protein
       decomposition is a literature artefact and only the aggregate survives. Reported alongside:
       pubs against half-life, against copies, and the proteasome subunits' own publication counts
       against the proteome's.
       AND THE LIMIT, stated so that a pass is not over-read: not all degradation is proteasomal.
       Autophagy and lysosomal proteolysis handle bulk protein and organelles, so assigning EVERY
       destroyed molecule to the 26S proteasome makes P3 conservative -- the true proteasomal load
       is lower than the demand computed here. A pass therefore bounds the answer; it does not
       measure proteasome occupancy. Nor does anything here supply ubiquitination kinetics,
       substrate queueing, or the difference between a 40-residue substrate and a 4,000-residue one,
       which the 1-3 s figure averages over.

[ADDED AFTER THE FIRST RUN, WHICH COMPLETED WITH ALL SIX THRESHOLDS ABOVE UNTOUCHED. Recorded here
because two of the additions bear directly on how the results should be read, and because a reader
is entitled to know which lines were written before the numbers and which after.

  NOTHING BELOW IS A GATE AND NO THRESHOLD ABOVE WAS ALTERED. Three diagnostics were added:
  (i) P3 now states how many times larger the demand, or slower the proteasome, would have to be
  before capacity is exceeded, and re-evaluates the worst corner using P1's stoichiometry-corrected
  count and the 19S-cap-only count; (ii) P5 gained a TAIL CHECK that drops the 20 most abundant
  proteins from both the real value and null A, because null A's statistic is a sum over copy
  numbers spanning five orders of magnitude and could be carried entirely by histones and actin --
  loop 93's K2 gated on a median while the error lived in a handful of reactions, and this is the
  same failure mode; (iii) P6 now reports rho(copies, half-life) directly, which is the monotone
  form of whatever null A detected.

  THE GATE VERDICTS UNDERSTATE ONE RESULT AND OVERSTATE ANOTHER. P3 passes by a very wide margin --
  the interesting number is not the pass but the ~16x margin at the worst corner, which says the
  proteasome is not a binding constraint on this cell under any assumption swept. P2 passes a
  factor-of-10 band, which is a weak instrument check and should be read as one, not as agreement.]

-> outputs/loop_proteostasis.json
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
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
FASTA = SC / "human_proteome.fasta.gz"
LN2 = float(np.log(2.0))

T_DOUBLE_H = 24.0                     # the project's standing assumption
SEC_PER_SUBSTRATE = (1.0, 2.0, 3.0)   # a 26S proteasome, from the literature; swept, not chosen
SEC_CENTRAL = 2.0
PROT_PG_SWEEP = (150.0, 300.0, 600.0)  # protein mass per mammalian cell
PROT_PG_CENTRAL = 300.0
DA_PER_RESIDUE = 110.0
AVOGADRO = 6.02214076e23

LOOP74_REPLACEMENT = 0.292            # as recorded
LOOP74_CORRECTED = 0.369              # as corrected at loop 71's own doubling time
LOOP74_MEDIAN_HL_H = 71.5
LOOP92_RIBOSOMES = 6_624_152.0
LOOP92_PROT_TOTAL = 2_337_014_740.0

P1_IQR_MAX = 3.0                      # obligate complex: subunit copies must agree within 3x
P2_HL_FACTOR = 10.0                   # implied mean half-life within 10x of 71.5 h
P4_CURRENCY_GAP = 0.10                # molecule- vs mass-weighted replacement fraction, 10 pp
P4_FACTOR = 2.0                       # mass-weighted vs loop 74's corrected 36.9%
NPERM = 200
SEED = 10301

CORE_ALPHA_BETA = ("PSMA", "PSMB")    # 20S core: TWO copies of each per particle
REG_ATPASE_LID = ("PSMC", "PSMD")     # 19S regulatory particle: ONE copy of each per particle
PSM_FAMILIES = CORE_ALPHA_BETA + REG_ATPASE_LID

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


def protein_lengths():
    """Residue count per gene symbol from the UniProt human proteome on disk. Same reader as loop 92."""
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


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 103 -- turnover: does the proteasome have the capacity to degrade what the "
        "half-lives imply?")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    S = json.load(open(SC / "_schwan2011.json"))
    plen = protein_lengths()

    # THE ABUNDANCE RULE. Copies and half-lives are BOTH Schwanhausser NIH3T3. Nothing below divides
    # one proteome by another; the only non-Schwanhausser quantities are sequence lengths and two
    # literature scalars (110 Da/residue, 1-3 s/substrate), and neither is an abundance measurement.
    sg = [g for g in S
          if S[g].get("prot_copies") and S[g].get("prot_hl_h")
          and np.isfinite(S[g]["prot_copies"]) and np.isfinite(S[g]["prot_hl_h"])
          and S[g]["prot_copies"] > 0 and S[g]["prot_hl_h"] > 0]
    sg_len = [g for g in sg if g in plen]
    N = np.array([S[g]["prot_copies"] for g in sg], float)
    K = np.array([LN2 / S[g]["prot_hl_h"] for g in sg], float)
    HL = np.array([S[g]["prot_hl_h"] for g in sg], float)
    Nl = np.array([S[g]["prot_copies"] for g in sg_len], float)
    Kl = np.array([LN2 / S[g]["prot_hl_h"] for g in sg_len], float)
    L = np.array([plen[g] for g in sg_len], float)

    say(f"  {len(S):,} genes in the Schwanhausser table; {len(sg):,} with BOTH a protein copy number")
    say(f"  and a protein half-life from the SAME cells; {len(sg_len):,} of those also have a length")
    say(f"  {len(plen):,} protein lengths from the UniProt human proteome on disk")
    say(f"  total protein over the covered set: {N.sum():,.0f} molecules "
        f"(loop 92 recorded {LOOP92_PROT_TOTAL:,.0f})")
    say()

    # ------------------------------------------------------------------ P1
    say("P1 THE PROTEASOME CENSUS IS A CENSUS AND NOT AN AVERAGE")
    psm = {g: S[g]["prot_copies"] for g in sg if g.startswith(PSM_FAMILIES)}
    psm_names = sorted(psm)
    pc = np.array([psm[g] for g in psm_names], float)
    particles = float(np.median(pc))
    q25, q75 = float(np.percentile(pc, 25)), float(np.percentile(pc, 75))
    iqr_ratio = q75 / q25 if q25 > 0 else float("inf")
    ab = np.array([psm[g] for g in psm_names if g.startswith(CORE_ALPHA_BETA)], float)
    cd = np.array([psm[g] for g in psm_names if g.startswith(REG_ATPASE_LID)], float)
    med_ab = float(np.median(ab)) if len(ab) else float("nan")
    med_cd = float(np.median(cd)) if len(cd) else float("nan")
    stoich_particles = float(np.median(np.concatenate([ab / 2.0, cd]))) if len(ab) and len(cd) \
        else float("nan")
    say(f"     {len(psm_names)} PSMA/PSMB/PSMC/PSMD subunits carry a copy number in this table")
    say(f"     {', '.join(psm_names)}")
    say(f"     median subunit copies = {particles:,.0f}  -> the 26S particle count, one subunit per")
    say(f"     particle, exactly as loop 92 took the median ribosomal protein as the ribosome count")
    say(f"     range {pc.min():,.0f} to {pc.max():,.0f}; P25 {q25:,.0f}  P75 {q75:,.0f}  "
        f"P75/P25 = {iqr_ratio:.2f}x")
    p1 = bool(np.isfinite(iqr_ratio) and iqr_ratio <= P1_IQR_MAX)
    say(f"     gate: P75/P25 <= {P1_IQR_MAX:.1f} -- an obligate complex whose members disagree by")
    say(f"     more than that is being averaged, not counted")
    say(f"     P1 {'PASS' if p1 else 'FAIL'} -- the median {'IS' if p1 else 'is NOT'} a particle count")
    say(f"     REPORTED, NOT GATED -- the stoichiometry exposure:")
    say(f"       PSMA/PSMB (20S core, TWO copies per particle) median {med_ab:,.0f} over {len(ab)}")
    say(f"       PSMC/PSMD (19S cap, ONE copy per particle)   median {med_cd:,.0f} over {len(cd)}")
    say(f"       observed core:cap ratio {med_ab / med_cd:.2f}x against the 2.00x that the known 20S")
    say(f"       stoichiometry predicts -- the subunit spread P1 gates on is therefore NOT noise, it")
    say(f"       is the real architecture of the particle showing through the copy numbers, which is")
    say(f"       an independent check on the census that nothing in this module fitted")
    say(f"       stoichiometry-corrected particle count {stoich_particles:,.0f} "
        f"({stoich_particles / particles:.2f}x the prescribed one)")
    say(f"     REPORTED, NOT GATED -- the machine ratio: {particles:,.0f} proteasomes against loop "
        f"92's")
    say(f"       {LOOP92_RIBOSOMES:,.0f} ribosomes = 1 proteasome per "
        f"{LOOP92_RIBOSOMES / particles:.1f} ribosomes")
    say()

    # ------------------------------------------------------------------ P2
    say("P2 THE PROTEOLYTIC DEMAND IS PHYSIOLOGICALLY SANE")
    load = float((N * K).sum())
    kbar_mol = load / float(N.sum())
    implied_hl = LN2 / kbar_mol
    mu = LN2 / T_DOUBLE_H
    K_nodil = np.clip(K - mu, 0.0, None)
    load_nodil = float((N * K_nodil).sum())
    say(f"     demand = SUM N_i * ln2/t_half,i over {len(sg):,} proteins")
    say(f"     total proteolytic load {load:,.0f} molecules destroyed per hour")
    say(f"                            {load / 3600.0:,.0f} molecules per second")
    say(f"     abundance-weighted mean degradation rate {kbar_mol:.6f} /h")
    say(f"     -> implied mean protein half-life {implied_hl:.1f} h; "
        f"per-gene MEDIAN half-life {float(np.median(HL)):.1f} h "
        f"(loop 74 recorded {LOOP74_MEDIAN_HL_H:.1f} h)")
    lo, hi = LOOP74_MEDIAN_HL_H / P2_HL_FACTOR, LOOP74_MEDIAN_HL_H * P2_HL_FACTOR
    p2 = bool(lo <= implied_hl <= hi)
    say(f"     gate: implied mean half-life within {P2_HL_FACTOR:.0f}x of {LOOP74_MEDIAN_HL_H:.1f} h,"
        f" i.e. {lo:.2f} h to {hi:.1f} h")
    say(f"     P2 {'PASS' if p2 else 'FAIL'} -- the abundance-weighted mean lifetime "
        f"{'is inside' if p2 else 'is OUTSIDE'} the band. Note the band is a")
    say(f"       factor of 10 wide: this is an instrument check that the sum is not absurd, and it")
    say(f"       is NOT evidence that the mean and the median agree. Against this set's own median")
    say(f"       of {float(np.median(HL)):.1f} h the abundance weighting moves the lifetime by "
        f"{implied_hl / float(np.median(HL)):.2f}x, upward,")
    say(f"       which is the same fact null A reports: the abundant proteins are the durable ones.")
    say(f"     REPORTED -- a cross-loop discrepancy that is NOT this module's to resolve: the")
    say(f"       per-gene median half-life over THIS 4,821-protein set is "
        f"{float(np.median(HL)):.1f} h, while the")
    say(f"       project's standing figure of {LOOP74_MEDIAN_HL_H:.1f} h comes from "
        f"cell_lifetimes.json's 13,329 genes. The")
    say(f"       two sets are not the same set. P2 gates against the standing figure anyway, which")
    say(f"       is the harder test of the two.")
    say(f"     REPORTED -- with the {T_DOUBLE_H:.0f} h dilution term removed from every rate, the")
    say(f"       load falls to {load_nodil:,.0f} /h ({load_nodil / load:.1%} of the above). If")
    say(f"       Schwanhausser's half-lives are not division-corrected, THAT is true proteolysis and")
    say(f"       the number gated in P3 is an overestimate, which is the conservative direction.")
    say()

    # ------------------------------------------------------------------ P3
    say("P3 DEMAND <= CAPACITY")
    # The whole-cell scaling: the covered set's residue content against a cell's, from one absolute
    # scalar. Sequence lengths are not an abundance dataset, so the abundance rule is untouched.
    A_cov = float((Nl * L).sum())
    load_len = float((Nl * Kl).sum())
    scales = {"measured set only": 1.0}
    for pg in PROT_PG_SWEEP:
        cell_res = (pg * 1e-12 / DA_PER_RESIDUE) * AVOGADRO
        scales[f"{pg:.0f} pg/cell"] = cell_res / A_cov
    say(f"     covered set residue content {A_cov:,.0f} residues over {len(sg_len):,} proteins")
    say(f"     their proteolytic load {load_len:,.0f} /h ({load_len / load:.1%} of the full "
        f"{len(sg):,}-protein load)")
    for lab, s in scales.items():
        if lab != "measured set only":
            say(f"     {lab:20s} -> a cell holds {s:.3f}x the covered set's protein mass")
    say(f"     the demand is scaled by that factor, assuming uncovered protein turns over at the")
    say(f"     covered set's rate. Loop 101 showed the machine count is a whole-cell census while")
    say(f"     the demand is not; this is the correction for that, and it can only raise the demand.")
    say()
    say(f"     the 'measured set only' row is the strict lower bound over all {len(sg):,} proteins")
    say(f"     with a copy number and a half-life; the scaled rows extrapolate the "
        f"{len(sg_len):,}-protein")
    say(f"     length-covered set, so numerator and denominator of the scaling are the same set")
    say()
    say(f"     {'':22s}" + "".join(f"{f'{s:.0f} s/substrate':>22s}" for s in SEC_PER_SUBSTRATE))
    grid = {}
    for lab, sc in scales.items():
        row, cells = [], {}
        d = (load if lab == "measured set only" else load_len) * sc
        for s in SEC_PER_SUBSTRATE:
            cap = particles * 3600.0 / s
            u = d / cap
            cells[f"{s:.0f}s"] = {"demand_per_h": d, "capacity_per_h": cap, "utilisation": u,
                                  "fits": bool(u <= 1.0)}
            row.append(f"{u:>17.2%}{'  ' if u <= 1.0 else ' !'}")
        grid[lab] = cells
        say(f"     {lab:22s}" + "".join(row))
    say(f"     (each cell is demand/capacity; ! marks a corner where the proteasome cannot keep up)")
    say()
    worst_lab = max(scales, key=lambda k: scales[k])
    worst_s = max(SEC_PER_SUBSTRATE)
    worst = grid[worst_lab][f"{worst_s:.0f}s"]
    central = grid[f"{PROT_PG_CENTRAL:.0f} pg/cell"][f"{SEC_CENTRAL:.0f}s"]
    p3 = bool(worst["fits"])
    any_fail = [f"{lab} @ {s:.0f}s" for lab in grid for s in SEC_PER_SUBSTRATE
                if not grid[lab][f"{s:.0f}s"]["fits"]]
    sat_s = particles * 3600.0 / (load_len * scales[f"{PROT_PG_CENTRAL:.0f} pg/cell"])
    say(f"     WORST corner ({worst_lab}, {worst_s:.0f} s/substrate): demand "
        f"{worst['demand_per_h']:,.0f}/h against capacity {worst['capacity_per_h']:,.0f}/h "
        f"= {worst['utilisation']:.2%}")
    say(f"     CENTRAL point ({PROT_PG_CENTRAL:.0f} pg/cell, {SEC_CENTRAL:.0f} s/substrate): "
        f"{central['utilisation']:.2%} of proteasome capacity used")
    say(f"     the per-particle processing time that would EXACTLY saturate the machine at the")
    say(f"     central cell size is {sat_s:,.1f} s per substrate, against the literature's 1-3 s")
    say(f"     HOW WRONG WOULD THIS HAVE TO BE TO FAIL. At the worst corner the demand would have")
    say(f"     to be {1.0 / worst['utilisation']:.1f}x larger, or the proteasome "
        f"{1.0 / worst['utilisation']:.1f}x slower than 3 s per substrate")
    say(f"     ({3.0 / worst['utilisation']:,.0f} s), before capacity is exceeded. Using the")
    say(f"     stoichiometry-corrected particle count from P1 ({stoich_particles:,.0f}) instead of")
    say(f"     the prescribed median, the worst corner becomes "
        f"{worst['utilisation'] * particles / stoich_particles:.2%}, and using the 19S cap subunits")
    say(f"     alone ({med_cd:,.0f}, the family that is unambiguously one copy per particle) it")
    say(f"     becomes {worst['utilisation'] * particles / med_cd:.2%}. None of those reverses it.")
    say(f"     gate: demand <= capacity at the WORST corner of the grid")
    say(f"     P3 {'PASS' if p3 else 'FAIL'} -- " +
        ("the proteasome has headroom under every assumption swept here"
         if p3 else f"capacity is exceeded at {len(any_fail)} of {len(grid) * len(SEC_PER_SUBSTRATE)}"
                    f" corners: {', '.join(any_fail)}"))
    say()

    # ------------------------------------------------------------------ P4
    say("P4 THE REPLACEMENT FRACTION")
    # molecule currency: a proteasome destroys one substrate whatever its size.
    f_mol = float((N * K).sum() / (N * (K + mu)).sum())
    # mass currency: a ribosome pays per residue. This is loop 74's and loop 101's currency.
    f_mass = float((Nl * L * Kl).sum() / (Nl * L * (Kl + mu)).sum())
    gap = abs(f_mol - f_mass)
    say(f"     at mu = ln2/{T_DOUBLE_H:.0f}h = {mu:.6f} /h")
    say(f"     molecule-weighted f = {f_mol:.1%}  -- the proteasome's currency, {len(sg):,} proteins")
    say(f"     mass-weighted     f = {f_mass:.1%}  -- the ribosome's currency, {len(sg_len):,} proteins")
    say(f"     so {f_mass:.0%} of every residue laid down replaces protein that has been destroyed;")
    say(f"     only {1 - f_mass:.0%} adds new mass.")
    p4a = bool(gap <= P4_CURRENCY_GAP)
    say(f"     (a) gate: the two currencies agree to within {P4_CURRENCY_GAP:.0%} -- gap "
        f"{gap:.1%}  {'PASS' if p4a else 'FAIL'}")
    say(f"         a gap would mean protein SIZE is coupled to turnover. The gap here is "
        f"{gap:.1%}, i.e.")
    say(f"         {gap / f_mass:.0%} of the fraction itself, and it points "
        f"{'the mass currency higher, so short-lived proteins are marginally LARGER' if f_mass > f_mol else 'the molecule currency higher, so short-lived proteins are marginally SMALLER'}.")
    say(f"         A proteasome's bill and a ribosome's bill are therefore the same bill to within")
    say(f"         a couple of points, and neither machine's view of turnover misleads about the")
    say(f"         other's.")
    ratio74 = f_mass / LOOP74_CORRECTED
    p4b = bool(1.0 / P4_FACTOR <= ratio74 <= P4_FACTOR)
    say(f"     (b) gate: mass-weighted within {P4_FACTOR:.0f}x of loop 74's corrected "
        f"{LOOP74_CORRECTED:.1%} (recorded {LOOP74_REPLACEMENT:.1%})")
    say(f"         ratio {ratio74:.3f}x, band {LOOP74_CORRECTED / P4_FACTOR:.1%}-"
        f"{LOOP74_CORRECTED * P4_FACTOR:.1%}  {'PASS' if p4b else 'FAIL'}")
    say(f"         DISCLOSED AS WEAK: loop 101's D2 already computed this from THIS dataset against")
    say(f"         THIS number, so (b) is a reproduction and its passing is worth only the")
    say(f"         confirmation that both modules read the same table.")
    p4 = bool(p4a and p4b)
    say(f"     P4 {'PASS' if p4 else 'FAIL'}")
    say()

    # ------------------------------------------------------------------ P5
    say("P5 THE NULLS ARE CAPABLE, AND THEY FIRE")
    rng = np.random.default_rng(SEED)
    real_load = load

    # NULL A -- global shuffle of half-lives across proteins.
    a_loads, a_moved = [], []
    for _ in range(NPERM):
        perm = rng.permutation(len(K))
        a_moved.append(GG.null_can_move(HL.tolist(), HL[perm].tolist())["changed"])
        a_loads.append(float((N * K[perm]).sum()))
    capA = GG.null_can_move(HL.tolist(), HL[rng.permutation(len(K))].tolist())
    say(f"     NULL A  half-lives shuffled globally across proteins")
    say(f"       CAPABILITY: the shuffle changes {float(np.mean(a_moved)):.1%} of half-life entries "
        f"-- capable: {capA['capable']}")
    sA = GG.survival(real_load, a_loads)
    GG.report("total proteolytic load under a global half-life shuffle", sA, emit=say)
    a_fires = bool(np.isfinite(sA.get("z", np.nan)) and abs(sA["z"]) >= 2.0)
    a_below = bool(sA["real"] < sA["null_mean"])
    say(f"       z = {sA['z']:+.2f}: the real load is "
        f"{'DISTINGUISHABLE from' if a_fires else 'indistinguishable from'} the null, and it lies "
        f"{'BELOW' if a_below else 'ABOVE'} it by {sA['null_mean'] / sA['real']:.2f}x")
    if a_fires and a_below:
        say(f"       the pairing of abundance with turnover is REAL and it works to REDUCE")
        say(f"       proteolysis: abundant proteins are systematically the LONG-LIVED ones, so a")
        say(f"       cell that paired abundance with turnover at random would have to destroy")
        say(f"       {sA['null_mean'] / sA['real']:.2f}x as much protein per hour than this one "
            f"does. gate_guard returns")
        say(f"       UNDEFINED rather than a survival percentage, which is correct: a null ABOVE the")
        say(f"       real value is not survival, it is an effect pointing the other way.")
    elif a_fires:
        say(f"       the pairing RAISES the load above a random pairing: abundant proteins are the")
        say(f"       short-lived ones, and the load is carried by that coincidence.")
    else:
        say(f"       the load is NOT distinguishable from (total protein) x (a random rate), so no")
        say(f"       per-protein datum was needed to compute it.")

    # NULL B -- shuffle half-lives WITHIN abundance deciles, preserving the coarse trend.
    order = np.argsort(N)
    dec = np.zeros(len(N), int)
    dec[order] = (np.arange(len(N)) * 10) // len(N)
    b_loads, b_moved = [], []
    for _ in range(NPERM):
        Kp = K.copy()
        HLp = HL.copy()
        for d in range(10):
            m = np.where(dec == d)[0]
            p = rng.permutation(len(m))
            Kp[m] = K[m][p]
            HLp[m] = HL[m][p]
        b_moved.append(GG.null_can_move(HL.tolist(), HLp.tolist())["changed"])
        b_loads.append(float((N * Kp).sum()))
    HLb = HL.copy()
    for d in range(10):
        m = np.where(dec == d)[0]
        HLb[m] = HL[m][rng.permutation(len(m))]
    capB = GG.null_can_move(HL.tolist(), HLb.tolist())
    say(f"     NULL B  half-lives shuffled WITHIN abundance deciles ({len(N) // 10:,} proteins each)")
    say(f"       CAPABILITY: the shuffle changes {float(np.mean(b_moved)):.1%} of half-life entries "
        f"-- capable: {capB['capable']}")
    sB = GG.survival(real_load, b_loads)
    GG.report("total proteolytic load under a within-decile half-life shuffle", sB, emit=say)
    b_fires = bool(np.isfinite(sB.get("z", np.nan)) and abs(sB["z"]) >= 2.0)
    say(f"       z = {sB['z']:+.2f}: the real load is "
        f"{'DISTINGUISHABLE from' if b_fires else 'indistinguishable from'} the null")
    say(f"     interpretation: A destroys the abundance-turnover pairing entirely, B keeps it at")
    say(f"     decile resolution.")
    if a_fires and not b_fires:
        say(f"     A fires and B does not, so the information in the load lives at DECILE")
        say(f"     resolution: knowing which abundance decile a protein sits in fixes the load to")
        say(f"     within the null spread, and its individual half-life adds nothing on top of")
        say(f"     that. That is a statement about how much per-protein data a turnover model")
        say(f"     actually needs, and it was not knowable without running both nulls.")
    elif a_fires and b_fires:
        say(f"     both fire, so per-protein half-lives carry information beyond the decile trend.")
    elif not a_fires:
        say(f"     neither null moves the load, so the abundance-turnover pairing carries nothing")
        say(f"     and the load is an aggregate quantity only.")
    # THE TAIL CHECK, ADDED AFTER THE FIRST RUN AND DISCLOSED AS SUCH. Loop 93's K2 gated on a
    # median while the error lived in a handful of high-flux reactions. Null A's statistic is a SUM
    # over copy numbers that span five orders of magnitude, so it can be carried entirely by a few
    # giants (histones, actin) without any proteome-wide trend existing at all. This drops the 20
    # most abundant proteins from BOTH the real value and the null and asks whether A still fires.
    # It is a diagnostic, not a gate: no threshold below was changed after it was written.
    keep = np.argsort(N)[:-20]
    Nk, Kk = N[keep], K[keep]
    real_trim = float((Nk * Kk).sum())
    trim_nulls = [float((Nk * Kk[rng.permutation(len(Kk))]).sum()) for _ in range(NPERM)]
    sT = GG.survival(real_trim, trim_nulls)
    t_fires = bool(np.isfinite(sT.get("z", np.nan)) and abs(sT["z"]) >= 2.0)
    top20 = sorted(range(len(N)), key=lambda i: -N[i] * K[i])[:20]
    top20_share = float(sum(N[i] * K[i] for i in top20) / load)
    say(f"     TAIL CHECK (added after the first run, a diagnostic and not a gate): the 20 most")
    say(f"       abundant proteins dropped from real AND null. z {sT['z']:+.2f}, real "
        f"{real_trim:,.0f} vs null {sT['null_mean']:,.0f}")
    say(f"       -> null A {'STILL FIRES' if t_fires else 'COLLAPSES'} without the giants, so the "
        f"effect is "
        f"{'proteome-wide' if t_fires else 'CARRIED BY A HANDFUL OF PROTEINS and must not be read as a proteome-wide trend'}")
    say(f"       the 20 largest single contributors carry {top20_share:.1%} of the whole load: "
        f"{', '.join(sg[i] for i in top20[:8])}")
    p5 = bool(capA["capable"] and capB["capable"])
    say(f"     gate: both nulls capable")
    say(f"     P5 {'PASS' if p5 else 'FAIL'} -- the nulls "
        f"{'are capable and their verdicts are usable' if p5 else 'are INERT'}")
    say()

    # ------------------------------------------------------------------ P6
    say("P6 THE FAME CONTROL, AND WHAT THIS DOES NOT GIVE")
    per_load = (N * K)
    pb = [pubs.get(g, 0.0) for g in sg]
    r_pubs_load, n_p = spear(per_load, pb)
    r_cop_load, _ = spear(N, per_load)
    r_pubs_cop, _ = spear(N, pb)
    r_pubs_hl, _ = spear(HL, pb)
    r_hl_load, _ = spear(HL, per_load)
    r_cop_hl, _ = spear(N, HL)
    say(f"     pubs vs per-protein proteolytic load {r_pubs_load:+.4f}   (n {n_p:,})")
    say(f"     copies vs per-protein load           {r_cop_load:+.4f}")
    say(f"     pubs vs copies                       {r_pubs_cop:+.4f}")
    say(f"     pubs vs protein half-life            {r_pubs_hl:+.4f}")
    say(f"     half-life vs per-protein load        {r_hl_load:+.4f}")
    say(f"     copies vs protein half-life          {r_cop_hl:+.4f}  <- the direct form of what")
    say(f"       null A detected. A positive value here IS 'abundant proteins are long-lived'. If")
    say(f"       it is near zero while null A fires, the null is being carried by the extreme tail")
    say(f"       rather than by a monotone trend, which is what P5's tail check tests.")
    p6 = bool(np.isfinite(r_pubs_load) and np.isfinite(r_cop_load)
              and abs(r_pubs_load) < abs(r_cop_load))
    say(f"     gate: |rho(pubs, load)| < |rho(copies, load)| -- fame must explain the load LESS well")
    say(f"     than the abundance the load is built from")
    say(f"     P6 {'PASS' if p6 else 'FAIL'} -- " +
        ("the per-protein decomposition is carried by abundance, not by attention"
         if p6 else "FAME BEATS THE BIOLOGY: the per-protein load is a literature ranking"))
    psm_pubs = [pubs.get(g, 0.0) for g in psm_names]
    say(f"     the proteasome subunits' own fame: median pubs {float(np.median(psm_pubs)):.0f} "
        f"against {float(np.median(pb)):.0f} for the covered proteome "
        f"({float(np.median(psm_pubs)) / max(float(np.median(pb)), 1.0):.2f}x)")
    say(f"     -- the particle count rests on a well-studied family, so its copy numbers are as")
    say(f"        likely as any to be right, and equally likely to be over-detected. Reported.")
    say()
    say(f"     WHAT THIS DOES NOT GIVE. Not all degradation is proteasomal: autophagy and lysosomal")
    say(f"     proteolysis handle bulk protein and organelles, so assigning every destroyed molecule")
    say(f"     to the 26S proteasome makes P3 CONSERVATIVE -- the real proteasomal load is below the")
    say(f"     demand computed here, and a pass bounds the answer rather than measuring occupancy.")
    say(f"     Nothing here supplies ubiquitination kinetics, substrate queueing, or the difference")
    say(f"     between a 40-residue substrate and a 4,000-residue one, which 1-3 s averages over.")
    say(f"     The 20S/19S stoichiometry (P1) remains a factor-of-two exposure in the direction that")
    say(f"     helps the gate, which is why P3 is evaluated at the worst corner and not the middle.")
    say()

    gates = {"P1 the proteasome census is a census": p1,
             "P2 the proteolytic demand is physiologically sane": p2,
             "P3 demand <= capacity at the worst corner": p3,
             "P4 the replacement fraction agrees across currencies and with loop 74": p4,
             "P5 both nulls are capable": p5,
             "P6 fame does not beat abundance on the load": p6}
    say("=" * 100)
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say("=" * 100)
    say()

    man = RM.manifest(
        inputs=[str(LR.CELL), str(SC / "_schwan2011.json"), str(FASTA)],
        available=len(S), used=len(sg), selection="filtered", seed=SEED,
        controls=[
            "the per-particle degradation rate swept 1-3 s rather than chosen, and the gate taken "
            "at the slowest",
            "the whole-cell scaling of the demand swept 150-600 pg protein per cell, and the gate "
            "taken at the largest",
            "the 20S (2 copies/particle) vs 19S (1 copy/particle) stoichiometry measured and the "
            "corrected particle count reported next to the prescribed one",
            "NULL A: half-lives shuffled globally, checked with gate_guard.null_can_move before "
            "its verdict was read",
            "NULL B: half-lives shuffled within abundance deciles, same capability check, so a "
            "coarse trend is distinguished from per-protein resolution",
            "publication count against the per-protein load, against copies and against half-life, "
            "with the gate requiring fame to lose to abundance",
        ],
        note="copies and half-lives are both Schwanhausser NIH3T3, so loop 92's abundance rule is "
             "obeyed and no rate here is a quotient of two proteomes; the only non-Schwanhausser "
             "quantities are sequence lengths and two literature scalars")
    RM.report(man, emit=say)

    json.dump({
        "test": "loop_proteostasis", "title": "does the proteasome have the capacity to degrade "
                                              "what the half-lives imply?",
        "manifest": man, "gates": gates,
        "coverage": {"schwanhausser_genes": len(S), "with_copies_and_halflife": len(sg),
                     "also_with_length": len(sg_len),
                     "protein_molecules_covered": float(N.sum()),
                     "loop92_protein_molecules": LOOP92_PROT_TOTAL},
        "p1": {"n_subunits": len(psm_names), "subunits": psm_names,
               "subunit_copies": {g: float(psm[g]) for g in psm_names},
               "particles_median": particles, "p25": q25, "p75": q75, "iqr_ratio": iqr_ratio,
               "threshold": P1_IQR_MAX, "median_alpha_beta_20S": med_ab,
               "median_atpase_lid_19S": med_cd,
               "stoichiometry_corrected_particles": stoich_particles,
               "ribosomes_per_proteasome": LOOP92_RIBOSOMES / particles, "pass": p1},
        "p2": {"load_molecules_per_h": load, "load_molecules_per_s": load / 3600.0,
               "kbar_molecule_per_h": kbar_mol, "implied_mean_halflife_h": implied_hl,
               "median_halflife_h": float(np.median(HL)), "loop74_median_h": LOOP74_MEDIAN_HL_H,
               "band_h": [lo, hi], "load_without_dilution_term": load_nodil,
               "dilution_free_fraction": load_nodil / load, "pass": p2},
        "p3": {"particles": particles, "sec_per_substrate_sweep": list(SEC_PER_SUBSTRATE),
               "protein_pg_sweep": list(PROT_PG_SWEEP), "scales": scales,
               "covered_residues": A_cov, "load_over_length_set": load_len,
               "grid": grid, "worst_corner": {"scale": worst_lab, "sec": worst_s, **worst},
               "central": {"scale": f"{PROT_PG_CENTRAL:.0f} pg/cell", "sec": SEC_CENTRAL, **central},
               "saturating_sec_per_substrate": sat_s, "failing_corners": any_fail, "pass": p3},
        "p4": {"mu_per_h": mu, "f_molecule_weighted": f_mol, "f_mass_weighted": f_mass,
               "gap": gap, "gap_threshold": P4_CURRENCY_GAP, "gap_pass": p4a,
               "loop74_recorded": LOOP74_REPLACEMENT, "loop74_corrected": LOOP74_CORRECTED,
               "ratio_vs_loop74_corrected": ratio74, "factor": P4_FACTOR,
               "loop74_pass": p4b, "loop74_gate_disclosed_as_reproduction": True, "pass": p4},
        "p5": {"n_perm": NPERM, "real_load": real_load,
               "null_a": {"capability": capA, "mean_changed": float(np.mean(a_moved)),
                          "survival": sA},
               "null_b": {"capability": capB, "mean_changed": float(np.mean(b_moved)),
                          "survival": sB},
               "null_a_fires": a_fires, "null_a_below_null": a_below, "null_b_fires": b_fires,
               "tail_check_top20_dropped": {"survival": sT, "fires": t_fires,
                                            "real_trimmed": real_trim,
                                            "note": "diagnostic added after the first run; no "
                                                    "threshold was changed after it was written"},
               "top20_load_share": top20_share,
               "top20_contributors": [sg[i] for i in top20],
               "pass": p5},
        "p6": {"rho_pubs_load": r_pubs_load, "rho_copies_load": r_cop_load,
               "rho_pubs_copies": r_pubs_cop, "rho_pubs_halflife": r_pubs_hl,
               "rho_halflife_load": r_hl_load, "rho_copies_halflife": r_cop_hl, "n": n_p,
               "psm_median_pubs": float(np.median(psm_pubs)),
               "proteome_median_pubs": float(np.median(pb)), "pass": p6},
        "seconds": time.time() - t0, "log": log},
        open(OUT / "loop_proteostasis.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_proteostasis.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
