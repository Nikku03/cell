"""LOOP 101 -- DIVISION: PREDICT THE DOUBLING TIME FROM THE PROTEIN BUDGET ALONE.

THE ONE NUMBER THIS PROJECT HAS ALWAYS ASSUMED. Every loop that needed a growth rate typed 24 h in
by hand. Loop 65 assumed it, loop 74 gated on it, loop 91 corrected its translation rates with it,
loop 92 built the whole ribosome budget on it. It has never once been DERIVED. That is the wrong way
round, because a doubling time is not a free parameter of a cell -- it is what falls out when you
ask how long it takes to build a cell's worth of protein with the ribosomes a cell has.

THE ARITHMETIC, and it has no fitted term. At steady exponential growth every protein sits at a
concentration where synthesis pays for both dilution and degradation:

        s_i = N_i (mu + k_i),        mu = ln2 / T,     k_i = ln2 / t_half,i

Summing the codons, with L_i the length in residues,

        codons/h  =  mu * SUM(N_i L_i)  +  SUM(N_i L_i k_i)  =  A (mu + kbar)

where A is A CELL'S ENTIRE PROTEIN CONTENT IN CODONS and kbar = SUM(N_i L_i k_i)/A is the
mass-weighted mean degradation rate. Every codon must be laid down by a ribosome, so

        A (mu + kbar)  <=  U * C,        C = R * e,   U = the fraction of ribosomes elongating

and the doubling time the cell can actually achieve is

        T  =  ln2 / ( U*C/A  -  kbar )

Nothing on the right-hand side is the doubling time or was fitted to it. R and e come from loop 92's
ribosome census, kbar from measured half-lives, A from a cell's protein content, U from polysome
measurements. The kbar term is exactly the correction the task asks for: it is the synthesis that
REPLACES degraded protein instead of adding new mass, and at the predicted rate the replacement
fraction is kbar/(mu+kbar), which falls out rather than being assumed.

THE TRAP, NAMED IN ADVANCE, BECAUSE IT WOULD MAKE THIS LOOP WORTHLESS. Loop 92's 23% utilisation is
NOT an input here and must not be. It was computed as demand/capacity where the demand already
carried mu = ln2/24 inside it. Feed 23% back into the equation above with loop 92's own covered-set
codon content and it returns 24.0 h exactly, by construction, and the "prediction" is a tautology
dressed as physics. D5 runs that tautology deliberately and gates on it returning 24 h, so the
circularity is demonstrated on the record rather than avoided in prose. The prediction proper uses a
utilisation from POLYSOME measurements (0.60-0.90 in growing mammalian cells), which knows nothing
about any doubling time.

THE HARD PART IS A, NOT THE ALGEBRA. Schwanhausser measures 4,179 usable genes -- copies, half-life
and length in one table, one cell type, so loop 92's abundance rule is obeyed and no rate here is
ever a quotient of two proteomes. But those 4,179 genes are a FRACTION of the proteome, whereas R,
the ribosome count, is a whole-cell census: every ribosome contains the ribosomal proteins that were
counted. Loop 92 wrote that "both sides are partial in the SAME way"; they are not, and that asymmetry
is the single largest error term in this loop. It is handled by four routes to A, two of them
absolute and independent of each other:

    M  measured only        A = the covered set's codons. A strict LOWER BOUND on a cell's content,
                            and therefore a lower bound on T. Reported as a bound, not a prediction.
    P  protein per cell     A = 300 pg protein/cell / 110 Da per residue. The most directly measured
       (CENTRAL)            absolute quantity in the problem, and independent of any proteomics
                            detection limit. Swept 150-600 pg.
    R  ribosome share       A = (ribosomal-protein codons in the cell) / 0.055, ribosomal protein
                            being ~5.5% of mammalian cell protein. A second absolute route that
                            shares no constant with P. Swept 0.04-0.08.
    H  PaxDb HeLa coverage  A = covered codons / (HeLa codon-mass fraction the covered symbols carry).
                            Reported and NOT central, because PaxDb's denominator is its own 7,222
                            detected proteins rather than the proteome, so this route can only ever
                            OVERSTATE coverage and understate A. Kept because a route that is wrong
                            in a known direction bounds the answer from the other side.

DISCLOSURE, REQUIRED BY THE PROJECT RULE ON PEEKING. Before fixing the gates below I did
order-of-magnitude arithmetic on ALREADY-PUBLISHED loop numbers -- loop 92's 30.76 Gcodons/h demand
and 133.5 Gcodons/h capacity, loop 74's 71.5 h median half-life -- purely to check the equation has a
positive solution at all. That told me A over the covered set is around 8e11 codons and that route M
would land near single-digit hours. It did not involve running any code in this module, and no
quantity this module computes was seen before the thresholds below were written. The thresholds
themselves come from the task statement (a factor of 2 on the prediction) and from what a
reproduction claim means (1% on loop 92's numbers), not from any output of this run.

PREDECLARED, before any number is computed:

  D1 THE ARITHMETIC REPRODUCES LOOP 92                              THE INSTRUMENT CHECK.
       recomputing the covered set from scratch must return loop 92's published demand
       30,762,337,520 codons/h, capacity 133,542,895,248 codons/h, 2,337,014,740 protein molecules
       and 4,179 genes. Gate: all four within 1%. If this fails, every number below is being
       computed on a different set than the one the task describes and nothing else matters.
  D2 THE REPLACEMENT FRACTION, RECOMPUTED                           THE TERM THE TASK ASKS FOR.
       f = kbar/(mu+kbar) at the 24 h reference, from the measured half-lives, abundance- and
       length-weighted. Loop 74 measured 29.2% at the same doubling time from a different abundance
       source. Gate: within a factor of 2 of 29.2%, i.e. 14.6%-58.4%. This is a replication across
       abundance sources and it can genuinely fail.
  D3 THE TWO ABSOLUTE ROUTES TO A AGREE                             THE PRECONDITION FOR D4.
       route P (protein mass per cell) against route R (ribosomal-protein share of the proteome).
       They share no constant. Gate: their ratio within a factor of 2. If they disagree, the
       prediction has no defensible input, D4's number is reported but must not be believed, and
       this loop's answer is "the cell's protein content is not known well enough", which is a
       result.
  D4 THE PREDICTED DOUBLING TIME                                    THE GATE.
       T from the equation above at the declared central parameters: route P at 300 pg/cell,
       U = 0.75 (midpoint of the 0.60-0.90 polysome range), e = 5.6 aa/s, R from loop 92.
       Gate: 12 h <= T <= 48 h -- within a factor of 2 of the measured 24 h. Reported alongside:
       the full T grid over four routes x eleven utilisations, the utilisation each route WOULD
       need to reproduce 24 h exactly, and whether that required utilisation is inside the
       measured polysome range. A prediction that needs a physically impossible U > 1 is refuted
       whatever the central point does.
  D5 THE CIRCULARITY CONTROL                                        THE TAUTOLOGY, RUN ON PURPOSE.
       feeding loop 92's own 23.04% utilisation into route M must return 24.00 h. Gate: within 1%
       of 24 h. Passing proves the route is circular and justifies excluding it; FAILING would mean
       my algebra is not loop 92's algebra and D4 is built on sand. Either way it is information,
       which is why it is a gate and not a footnote.
  D6 THE NULLS ARE CAPABLE, AND THE FAME CONTROL                    THE GUARDS.
       two nulls, each checked with gate_guard.null_can_move BEFORE its verdict is read: shuffle
       half-lives across genes (does the pairing of turnover with mass carry information, or is
       kbar just the median half-life?), and shuffle copy numbers across lengths (does the
       length-abundance pairing matter, or is A just n * <L> * <N>?). gate_guard.survival decides
       whether either difference is even defined. Gate: both nulls capable. And the recurring
       killer: `pubs` against copies, against each gene's codon demand, against half-life, and --
       the one that bears on the answer -- against MEMBERSHIP in the measured set, because if the
       4,179 measured genes are the famous ones then the missing proteome is the unstudied one and
       route M's shortfall is a fame artefact.

  WHAT WOULD MAKE THIS LOOP A FAILURE: a PASS on D4 obtained by choosing U or the protein content
  after seeing T. Both are fixed above and swept afterwards, and the sweep is printed in full so a
  reader can see what any other choice would have given.

[ADDED AFTER THE FIRST RUN, WHICH COMPLETED WITH EVERY THRESHOLD ABOVE UNTOUCHED. Recorded here
because the gate verdict is a weaker statement than what the run actually showed, and the record
should carry the stronger one.

  D4 PASSES AT ITS EDGE, AND IN ONE DIRECTION. 13.79 h against 24 h is 0.575x -- inside the
  0.5x-2.0x band by 15% of its width. Every route and every constant corner errs the SAME WAY,
  fast, never slow. A gate that passes at its boundary with a one-sided residual is not a
  confirmation; it is a mild refutation that happens to clear the bar that was set.

  THE NUMBER THAT SAYS MORE THAN THE GATE. Inverting the prediction, the whole-cell ribosome
  utilisation needed to make 24 h come out is 0.487 by route P and 0.454 by route R. The measured
  polysome fraction in growing mammalian cells is 0.60-0.90. All four routes land BELOW that range.
  So the honest reading of this loop is not "the budget predicts 24 h" -- it is: THE PROTEIN BUDGET
  PERMITS A ~14 h DOUBLING AND THE CELL TAKES 24, so translation capacity is not the binding
  constraint on division, and roughly half the cell's ribosome-hours are not being spent.

  AND ONE THING THAT IS AN IDENTITY, NOT A CONFIRMATION, so nobody reads it as one. That required
  0.487 is exactly loop 92's 0.2304 divided by route P's 47.3% coverage, because both are
  demand-at-24-h over capacity with the same numerator scaled. It shows loop 92's headline 23% was
  a COVERAGE ARTEFACT of pricing 47% of the proteome against 100% of the ribosomes, and the
  whole-cell figure is about twice that -- which is worth recording, but it is arithmetic, not
  evidence.]

-> outputs/loop_doubling.json
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
import cell_proteome as CP  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
FASTA = SC / "human_proteome.fasta.gz"
LN2 = float(np.log(2.0))
AVOGADRO = 6.02214076e23

# ---- loop 92, published. D1 gates on reproducing every one of these.
LOOP92_DEMAND = 30_762_337_520.101093
LOOP92_CAPACITY = 133_542_895_248.0
LOOP92_UTIL = 0.23035547838747195
LOOP92_NGENES = 4179
LOOP92_PROT_TOTAL = 2_337_014_740.050006
LOOP92_RIBOSOMES = 6_624_151.55
LOOP74_REPLACEMENT = 0.292          # loop 74, 24 h, different abundance source

# ---- literature constants, all declared before anything is computed
T_MEASURED_H = 24.0                 # the number this loop is trying to predict, never an input
ELONG_AA_S = 5.6                    # loop 92's mammalian elongation rate
U_CENTRAL = 0.75                    # midpoint of the polysome range below
POLYSOME_LIT = (0.60, 0.90)         # fraction of ribosomes in polysomes, growing mammalian cells
U_SWEEP = (0.10, 0.20, 0.23, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00)
PROTEIN_PG_CELL = 300.0             # route P central: pg of protein per mammalian cell
PROTEIN_PG_SWEEP = (150.0, 300.0, 600.0)
RES_MASS_DA = 110.0                 # mean residue mass
PHI_R_LIT = 0.055                   # route R central: ribosomal protein share of cell protein
PHI_R_SWEEP = (0.04, 0.055, 0.08)
RIB_PROTEIN_AA = 12_700.0           # residues of protein in one 80S ribosome (~1.4 MDa / 110 Da)

# ---- gate thresholds, fixed here and never touched again
D1_TOL = 0.01
D2_FACTOR = 2.0
D3_FACTOR = 2.0
D4_LO, D4_HI = T_MEASURED_H / 2.0, T_MEASURED_H * 2.0
D5_TOL = 0.01

NPERM = 200
SEED = 10101

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


def auc(pos, neg):
    """Rank AUC of `pos` against `neg`. Ties handled by average rank."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


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


def predict_T(A, kbar, U, cap):
    """T = ln2 / (U*C/A - kbar). Returns inf when capacity cannot even pay maintenance."""
    net = U * cap / A - kbar
    if not np.isfinite(net) or net <= 0:
        return float("inf")
    return float(LN2 / net)


def required_U(A, kbar, cap, T=T_MEASURED_H):
    """The utilisation that would reproduce T exactly. Physically impossible above 1."""
    return float(A * (LN2 / T + kbar) / cap)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 101 -- division: predict the doubling time from the protein budget alone")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    idx = {n: i for i, n in enumerate(names)}
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    S = json.load(open(SC / "_schwan2011.json"))
    plen = protein_lengths()
    say(f"  {len(names):,} model genes; {len(S):,} Schwanhausser entries; "
        f"{len(plen):,} protein lengths")

    # THE ABUNDANCE RULE. copies and half-lives are the same table, the same cells. Length is a
    # per-gene constant, not an abundance, so it cannot make a rate into a cell-type difference.
    sg = [g for g in S if S[g].get("prot_copies") and S[g].get("prot_hl_h") and g in plen]
    N = np.array([S[g]["prot_copies"] for g in sg], float)
    L = np.array([plen[g] for g in sg], float)
    HL = np.array([S[g]["prot_hl_h"] for g in sg], float)
    K = LN2 / HL
    say(f"  {len(sg):,} genes with protein copies, protein half-life and a length, ONE cell type")
    say(f"  median protein half-life {np.median(HL):.1f} h; median length {np.median(L):.0f} aa; "
        f"median copies {np.median(N):,.0f}")
    say()

    A_cov = float((N * L).sum())
    B_cov = float((N * L * K).sum())
    kbar = B_cov / A_cov
    prot_total = float(N.sum())
    is_rp = np.array([g.startswith("RPL") or g.startswith("RPS") for g in sg])
    R = float(np.median(N[is_rp]))
    cap = R * ELONG_AA_S * 3600.0
    demand24 = A_cov * LN2 / T_MEASURED_H + B_cov

    # ------------------------------------------------------------------ D1
    say("D1 THE ARITHMETIC REPRODUCES LOOP 92")
    checks = {
        "codon demand at 24 h": (demand24, LOOP92_DEMAND),
        "ribosome capacity": (cap, LOOP92_CAPACITY),
        "protein molecules": (prot_total, LOOP92_PROT_TOTAL),
        "genes used": (float(len(sg)), float(LOOP92_NGENES)),
    }
    d1_each = {}
    for k, (mine, theirs) in checks.items():
        rel = abs(mine / theirs - 1.0) if theirs else float("inf")
        d1_each[k] = {"recomputed": mine, "loop92": theirs, "rel_error": rel}
        say(f"     {k:24s} recomputed {mine:>22,.2f}   loop 92 {theirs:>22,.2f}   "
            f"rel err {rel:.2e}")
    d1 = all(v["rel_error"] <= D1_TOL for v in d1_each.values())
    say(f"     ribosomes {R:,.0f} = median copies over {int(is_rp.sum())} ribosomal proteins")
    say(f"     A, the covered set's codon content: {A_cov:,.0f} codons")
    say(f"     B, its maintenance demand:          {B_cov:,.0f} codons/h  (no doubling time in it)")
    say(f"     kbar = B/A = {kbar:.6f} /h   <->  a mass-weighted mean protein half-life of "
        f"{LN2 / kbar:.1f} h")
    say(f"     D1 {'PASS' if d1 else 'FAIL'} -- the set below "
        f"{'IS' if d1 else 'is NOT'} loop 92's set (tolerance {D1_TOL:.0%})")
    say()

    # ------------------------------------------------------------------ D2
    say("D2 THE REPLACEMENT FRACTION, RECOMPUTED")
    mu24 = LN2 / T_MEASURED_H
    f24 = kbar / (mu24 + kbar)
    say(f"     at the 24 h reference: mu {mu24:.6f} /h, kbar {kbar:.6f} /h")
    say(f"     replacement fraction f = kbar/(mu+kbar) = {f24:.1%}")
    say(f"     loop 74, same doubling time, a DIFFERENT abundance source: {LOOP74_REPLACEMENT:.1%}")
    ratio74 = f24 / LOOP74_REPLACEMENT
    d2 = bool(1.0 / D2_FACTOR <= ratio74 <= D2_FACTOR)
    say(f"     ratio {ratio74:.3f}x   gate: within {D2_FACTOR:.0f}x, i.e. "
        f"{LOOP74_REPLACEMENT / D2_FACTOR:.1%}-{LOOP74_REPLACEMENT * D2_FACTOR:.1%}")
    say(f"     so {f24:.0%} of every codon laid down at 24 h replaces protein that has been "
        f"destroyed;")
    say(f"     only {1 - f24:.0%} adds new mass. That factor is what makes T longer than a naive")
    say(f"     'time to build one proteome' calculation, and it is measured, not assumed.")
    say(f"     D2 {'PASS' if d2 else 'FAIL'} -- the replacement term "
        f"{'replicates across abundance sources' if d2 else 'does NOT replicate loop 74'}")
    say()

    # ------------------------------------------------------------------ D3
    say("D3 THE TWO ABSOLUTE ROUTES TO A CELL'S CODON CONTENT AGREE")
    A_P = PROTEIN_PG_CELL * 1e-12 / RES_MASS_DA * AVOGADRO
    rp_codons = R * RIB_PROTEIN_AA
    rp_codons_seen = float((N[is_rp] * L[is_rp]).sum())
    A_R = rp_codons / PHI_R_LIT
    hela = CP.hela_ppm()
    sgset = set(sg)
    hela_num = sum(v * plen[g] for g, v in hela.items() if g in plen and g in sgset and v > 0)
    hela_den = sum(v * plen[g] for g, v in hela.items() if g in plen and v > 0)
    cov_hela = hela_num / hela_den if hela_den else float("nan")
    A_H = A_cov / cov_hela if cov_hela > 0 else float("inf")
    A_M = A_cov
    say(f"     P  protein per cell   {PROTEIN_PG_CELL:.0f} pg / {RES_MASS_DA:.0f} Da  -> "
        f"A = {A_P:,.0f} codons   (covered set is {A_cov / A_P:.1%} of it)")
    say(f"     R  ribosome share     {R:,.0f} ribosomes x {RIB_PROTEIN_AA:,.0f} aa / "
        f"{PHI_R_LIT:.3f}  -> A = {A_R:,.0f} codons   (covered set is {A_cov / A_R:.1%})")
    A_R_seen = rp_codons_seen / PHI_R_LIT
    say(f"        the {int(is_rp.sum())} ribosomal proteins actually detected sum to "
        f"{rp_codons_seen:,.0f} codons against the {rp_codons:,.0f} a full 80S complement implies")
    say(f"        -- using the detected sum instead of the literature complement gives "
        f"A = {A_R_seen:,.0f}")
    say(f"        and T = {predict_T(A_R_seen, kbar, U_CENTRAL, cap):.2f} h, so route R does not "
        f"turn on which of the two is taken")
    say(f"     H  PaxDb HeLa         covered symbols carry {cov_hela:.1%} of HeLa's codon mass "
        f"-> A = {A_H:,.0f} codons")
    say(f"        (HeLa's own denominator is {len(hela):,} DETECTED proteins, not the proteome, so "
        f"this route")
    say(f"         can only overstate coverage and understate A. It is a bound, not the central "
        f"value.)")
    say(f"     M  measured only      A = {A_M:,.0f} codons -- a strict LOWER bound on a cell's "
        f"content")
    ratioPR = A_P / A_R
    d3 = bool(1.0 / D3_FACTOR <= ratioPR <= D3_FACTOR)
    say(f"     P/R = {ratioPR:.3f}x   gate: within {D3_FACTOR:.0f}x. These two share no constant:")
    say(f"     one is grams of protein per cell, the other is the ribosome's share of the proteome.")
    say(f"     D3 {'PASS' if d3 else 'FAIL'} -- a cell's codon content "
        f"{'is pinned to within a factor of 2' if d3 else 'is NOT known well enough for D4 to mean anything'}")
    say()

    routes = {"P protein-per-cell (CENTRAL)": A_P, "R ribosome-share": A_R,
              "H HeLa-coverage (overstates coverage)": A_H, "M measured-only (lower bound on T)": A_M}

    # ------------------------------------------------------------------ D4
    say("D4 THE PREDICTED DOUBLING TIME")
    T_pred = predict_T(A_P, kbar, U_CENTRAL, cap)
    say(f"     central parameters, all fixed in the docstring before anything was computed:")
    say(f"       A  = {A_P:,.0f} codons        (route P, {PROTEIN_PG_CELL:.0f} pg protein/cell)")
    say(f"       C  = {cap:,.0f} codons/h  ({R:,.0f} ribosomes x {ELONG_AA_S} aa/s)")
    say(f"       U  = {U_CENTRAL:.2f}                        (midpoint of the "
        f"{POLYSOME_LIT[0]:.2f}-{POLYSOME_LIT[1]:.2f} polysome range)")
    say(f"       kbar = {kbar:.6f} /h                (measured half-lives, mass-weighted)")
    say(f"     gross capacity U*C/A = {U_CENTRAL * cap / A_P:.6f} /h;  maintenance takes "
        f"{kbar:.6f} /h;  net mu = {U_CENTRAL * cap / A_P - kbar:.6f} /h")
    say(f"     PREDICTED DOUBLING TIME  {T_pred:.2f} h        measured {T_MEASURED_H:.0f} h        "
        f"ratio {T_pred / T_MEASURED_H:.3f}x")
    d4 = bool(D4_LO <= T_pred <= D4_HI)
    say(f"     gate: {D4_LO:.0f} h <= T <= {D4_HI:.0f} h (within a factor of 2 of the measured 24 h)")
    say(f"     D4 {'PASS' if d4 else 'FAIL'}")
    say()
    say(f"     THE SENSITIVITY, printed in full so no reader has to take the central point on "
        f"trust.")
    say(f"     Rows are routes to A; columns are ribosome utilisation. Hours.")
    hdr = "       " + f"{'route':<40}" + "".join(f"{u:>7.2f}" for u in U_SWEEP)
    say(hdr)
    grid = {}
    for rn, Av in routes.items():
        row = []
        for u in U_SWEEP:
            row.append(predict_T(Av, kbar, u, cap))
        grid[rn] = {str(u): row[i] for i, u in enumerate(U_SWEEP)}
        say("       " + f"{rn:<40}" +
            "".join((f"{v:>7.1f}" if np.isfinite(v) else f"{'--':>7}") for v in row))
    say(f"       ('--' = capacity cannot even pay maintenance at that utilisation, so the cell "
        f"cannot grow at all)")
    say()
    say(f"     THE INVERSE QUESTION, which is the honest way to read a sweep: what utilisation "
        f"would each")
    say(f"     route need to reproduce 24 h exactly, and is that utilisation physically possible?")
    ureq = {}
    for rn, Av in routes.items():
        u = required_U(Av, kbar, cap)
        inside = POLYSOME_LIT[0] <= u <= POLYSOME_LIT[1]
        possible = u <= 1.0
        ureq[rn] = {"U_required": u, "inside_polysome_range": bool(inside), "possible": bool(possible)}
        say(f"       {rn:<40} U = {u:>6.3f}   "
            f"{'INSIDE' if inside else ('possible but outside' if possible else 'IMPOSSIBLE, U > 1')}"
            f" the measured {POLYSOME_LIT[0]:.2f}-{POLYSOME_LIT[1]:.2f} range")
    n_inside = sum(1 for v in ureq.values() if v["inside_polysome_range"])
    u_P = ureq["P protein-per-cell (CENTRAL)"]["U_required"]
    say(f"     {n_inside} of {len(ureq)} routes need a utilisation inside the measured polysome "
        f"range.")
    say(f"     REPORTED, NOT GATED. Every route errs the same way -- the budget permits a FASTER")
    say(f"     doubling than the cell achieves, never a slower one. The gate asked only for a "
        f"factor")
    say(f"     of 2 and got {T_pred / T_MEASURED_H:.3f}x, but a one-sided residual across four "
        f"independent routes is")
    say(f"     a statement: at U={u_P:.3f} the cell is leaving roughly "
        f"{1 - u_P / U_CENTRAL:.0%} of its ribosome-hours unspent")
    say(f"     relative to the polysome midpoint, so TRANSLATION CAPACITY IS NOT WHAT LIMITS "
        f"DIVISION here.")
    say(f"     An identity, flagged so it is not mistaken for a second confirmation: this "
        f"{u_P:.3f} is")
    say(f"     exactly loop 92's {LOOP92_UTIL:.4f} divided by route P's {A_cov / A_P:.1%} coverage. "
        f"Loop 92's 23% priced")
    say(f"     {A_cov / A_P:.0%} of the proteome against 100% of the ribosomes; the whole-cell "
        f"figure is {u_P:.0%}.")
    say()
    say(f"     the same prediction over the swept literature constants, at U = {U_CENTRAL:.2f}:")
    sw = {}
    for pg in PROTEIN_PG_SWEEP:
        A2 = pg * 1e-12 / RES_MASS_DA * AVOGADRO
        sw[f"P {pg:.0f} pg"] = predict_T(A2, kbar, U_CENTRAL, cap)
    for ph in PHI_R_SWEEP:
        A2 = rp_codons / ph
        sw[f"R phi_R {ph:.3f}"] = predict_T(A2, kbar, U_CENTRAL, cap)
    for kn, kv in sw.items():
        say(f"       {kn:<20} T = {kv:>7.2f} h   "
            f"{'within 2x' if D4_LO <= kv <= D4_HI else 'OUTSIDE 2x'}")
    n_in = sum(1 for v in sw.values() if D4_LO <= v <= D4_HI)
    say(f"     {n_in} of {len(sw)} literature-constant corners land within a factor of 2 of 24 h.")
    say()
    # replacement fraction AT THE PREDICTED rate, which is the self-consistent version of D2
    mu_pred = LN2 / T_pred if np.isfinite(T_pred) and T_pred > 0 else float("nan")
    f_pred = kbar / (mu_pred + kbar) if np.isfinite(mu_pred) else float("nan")
    say(f"     self-consistently, at the PREDICTED {T_pred:.1f} h rather than the assumed 24 h, the")
    say(f"     replacement fraction is {f_pred:.1%} (it was {f24:.1%} at 24 h): a faster cell spends")
    say(f"     less of its ribosome time on upkeep, which is the same direction loop 74 found.")
    say()
    # bootstrap over genes: under route P the data enters only through kbar
    rng = np.random.default_rng(SEED)
    boot = []
    n = len(sg)
    for _ in range(NPERM):
        b = rng.integers(0, n, n)
        kb = float((N[b] * L[b] * K[b]).sum() / (N[b] * L[b]).sum())
        boot.append(predict_T(A_P, kb, U_CENTRAL, cap))
    boot = np.array(boot, float)
    say(f"     bootstrap over the {n:,} genes ({NPERM} resamples), route P at U={U_CENTRAL:.2f}: "
        f"T = {np.median(boot):.2f} h  [{np.percentile(boot, 2.5):.2f}, "
        f"{np.percentile(boot, 97.5):.2f}]")
    say(f"     -- the gene-level sampling error is small because under route P the data enters only")
    say(f"        through kbar. The uncertainty in this prediction is the CONSTANTS, not the genes.")
    say()

    # ------------------------------------------------------------------ D5
    say("D5 THE CIRCULARITY CONTROL -- the tautology, run on purpose")
    T_circ = predict_T(A_M, kbar, LOOP92_UTIL, cap)
    d5 = bool(abs(T_circ / T_MEASURED_H - 1.0) <= D5_TOL)
    say(f"     loop 92's utilisation {LOOP92_UTIL:.5f} was DEFINED as demand/capacity, and that "
        f"demand")
    say(f"     already had mu = ln2/24 inside it. Feeding it back with loop 92's own covered-set A:")
    say(f"       T = {T_circ:.4f} h   against the 24 h that was fed in   "
        f"(rel err {abs(T_circ / T_MEASURED_H - 1.0):.2e})")
    say(f"     D5 {'PASS' if d5 else 'FAIL'} -- the route "
        f"{'IS circular and is therefore excluded from D4' if d5 else 'does NOT reproduce loop 92, so my algebra is not loop 92 algebra'}")
    say(f"     this is why D4 uses a POLYSOME utilisation, which was measured on cells nobody timed.")
    say()

    # ------------------------------------------------------------------ D6
    say("D6 THE NULLS ARE CAPABLE, AND THE FAME CONTROL")
    # null 1: shuffle half-lives across genes. Does turnover-mass pairing carry information?
    n1_nulls, n1_moved = [], []
    for _ in range(NPERM):
        p = rng.permutation(n)
        Kp = K[p]
        n1_moved.append(GG.null_can_move(list(K), list(Kp))["changed"])
        n1_nulls.append(float((N * L * Kp).sum() / A_cov))
    cap1 = GG.null_can_move(list(K), list(K[rng.permutation(n)]))
    say(f"     NULL 1  half-lives shuffled across genes -- tests whether kbar is more than the "
        f"mean rate")
    say(f"     CAPABILITY: {np.mean(n1_moved):.1%} of entries change -- capable: {cap1['capable']}")
    s1 = GG.survival(kbar, n1_nulls)
    GG.report("kbar under a half-life shuffle", s1, emit=say)
    say(f"     z = {s1.get('z', float('nan')):+.2f}. UNDEFINED here is the guard working, not the "
        f"null failing: the shuffle")
    say(f"     moves kbar far outside its own spread, so the pairing DOES carry information -- but "
        f"it moves")
    say(f"     it UPWARD, and a survival fraction of a quantity the null exceeds has no meaning. "
        f"The size")
    say(f"     of the move is the result; the ratio would have been the artefact.")
    kbar_unw = float(K.mean())
    say(f"     for scale: the UNWEIGHTED mean degradation rate is {kbar_unw:.6f} /h and the "
        f"mass-weighted one is {kbar:.6f} /h")
    say(f"     -> abundant, long proteins are {'LONGER' if kbar < kbar_unw else 'SHORTER'}-lived "
        f"than the typical gene, by a factor of {kbar_unw / kbar:.2f} in rate")
    T_unw = predict_T(A_P, kbar_unw, U_CENTRAL, cap)
    say(f"     using the unweighted rate instead would give T = {T_unw:.2f} h against "
        f"{T_pred:.2f} h -- the weighting is worth {abs(T_unw - T_pred):.2f} h")

    # null 2: shuffle copy numbers across lengths. Does the length-abundance pairing set A?
    n2_nulls, n2_moved = [], []
    for _ in range(NPERM):
        p = rng.permutation(n)
        Np = N[p]
        n2_moved.append(GG.null_can_move(list(N), list(Np))["changed"])
        n2_nulls.append(float((Np * L).sum()))
    cap2 = GG.null_can_move(list(N), list(N[rng.permutation(n)]))
    say(f"     NULL 2  copy numbers shuffled across lengths -- tests whether A is more than "
        f"n x <L> x <N>")
    say(f"     CAPABILITY: {np.mean(n2_moved):.1%} of entries change -- capable: {cap2['capable']}")
    s2 = GG.survival(A_cov, n2_nulls)
    GG.report("A under a copy-number shuffle", s2, emit=say)
    say(f"     z = {s2.get('z', float('nan')):+.2f}")
    say(f"     a null ABOVE the real value here is the informative outcome, not a failure: it says")
    say(f"     the abundant proteins are the SHORT ones, so a cell's codon content is lower than "
        f"its")
    say(f"     molecule count and mean length would suggest.")
    d6_nulls = bool(cap1["capable"] and cap2["capable"])

    say()
    say(f"     THE FAME CONTROL")
    pb = np.array([pubs.get(g, 0.0) for g in sg], float)
    dem_i = N * L * (mu24 + K)
    r_cop, n_f = spear(N, pb)
    r_dem, _ = spear(dem_i, pb)
    r_hl, _ = spear(HL, pb)
    r_len, _ = spear(L, pb)
    say(f"     pubs vs protein copies {r_cop:+.4f}   vs this gene's codon demand {r_dem:+.4f}   "
        f"vs half-life {r_hl:+.4f}   vs length {r_len:+.4f}   (n {n_f:,})")
    inset = np.array([pubs.get(g, 0.0) for g in sg if g in idx], float)
    outset = np.array([pubs[g] for g in names if g not in sgset], float)
    a_mem = auc(inset, outset)
    say(f"     pubs predicts MEMBERSHIP of the measured set at AUC {a_mem:.4f}  "
        f"({len(inset):,} measured vs {len(outset):,} not)")
    say(f"     median pubs inside the measured set {np.median(inset):.0f}, outside "
        f"{np.median(outset):.0f}")
    say(f"     THIS IS THE ONE THAT BEARS ON THE ANSWER. The gap between route M and route P is")
    say(f"     {1 - A_cov / A_P:.0%} of a cell's protein, and at AUC {a_mem:.2f} that missing mass "
        f"is disproportionately")
    say(f"     the UNSTUDIED proteome. Route P is immune to it -- grams of protein per cell do not")
    say(f"     care what has been published -- which is the reason it is central and route H, which")
    say(f"     is built out of a second detection-limited proteomics dataset, is not.")
    d6 = d6_nulls
    say(f"     D6 {'PASS' if d6 else 'FAIL'} -- both nulls "
        f"{'are capable, so their verdicts are usable' if d6 else 'are INERT'}")
    say()

    say("  THE HONEST LIMIT")
    say(f"     This predicts the doubling time a protein budget PERMITS, not the one a cell chooses.")
    say(f"     A cell can always grow more slowly than its ribosomes allow, so the calculation is a")
    say(f"     LOWER bound on T that happens to be tight -- and it would be equally consistent with")
    say(f"     the data if the real constraint were nucleotides, membrane, or a checkpoint. Nothing")
    say(f"     here shows translation is the binding constraint; it shows that if it is, the number")
    say(f"     comes out right. The four routes to A disagree by a factor of "
        f"{max(v for v in routes.values() if np.isfinite(v)) / min(routes.values()):.1f}, and that")
    say(f"     spread, not the gene-level noise, is the whole uncertainty of this loop.")
    say()

    gates = {"D1 arithmetic reproduces loop 92": bool(d1),
             "D2 replacement fraction replicates loop 74": bool(d2),
             "D3 the two absolute routes to A agree": bool(d3),
             "D4 predicted doubling time within 2x of 24 h": bool(d4),
             "D5 the circular route is demonstrated circular": bool(d5),
             "D6 both nulls capable, fame reported": bool(d6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(SC / "_schwan2011.json"), str(FASTA),
                              str(CP.HELA), str(CP.GTF)],
                      available=len(S), used=len(sg), selection="filtered", seed=SEED,
                      controls=[
                          "loop 92's published demand, capacity, protein total and gene count "
                          "recomputed from scratch as an arithmetic control that could have failed",
                          "the replacement fraction re-derived and compared against loop 74's 29.2% "
                          "from a different abundance source",
                          "two absolute routes to a cell's codon content that share no constant, "
                          "cross-checked against each other before either is used",
                          "the circular route run deliberately: loop 92's 23% utilisation returns "
                          "24 h by construction and is therefore excluded from the prediction",
                          "ribosome utilisation swept 0.10-1.00 and the utilisation each route "
                          "would need to reproduce 24 h reported against the polysome range",
                          "a half-life shuffle and a copy-number shuffle, both checked with "
                          "gate_guard.null_can_move before their verdicts were read, plus "
                          "publication count against copies, demand, half-life and set membership"],
                      note="the doubling time has been an assumed constant in every loop that "
                           "needed it; here it is predicted from ribosome capacity, measured "
                           "protein turnover and a cell's protein content, with no term fitted "
                           "to 24 h")
    RM.report(man, emit=say)

    json.dump({"test": "loop_doubling", "manifest": man, "gates": gates,
               "n_genes": len(sg), "n_schwanhausser": len(S),
               "d1": d1_each,
               "d2": {"kbar_per_h": kbar, "mass_weighted_half_life_h": LN2 / kbar,
                      "mu24": mu24, "replacement_fraction_24h": f24,
                      "loop74": LOOP74_REPLACEMENT, "ratio": ratio74},
               "d3": {"A_P_protein_per_cell": A_P, "A_R_ribosome_share": A_R,
                      "A_H_hela_coverage": A_H, "A_M_measured_only": A_M,
                      "hela_codon_mass_coverage": cov_hela, "ratio_P_over_R": ratioPR,
                      "ribosomes": R, "rp_codons_full_complement": rp_codons,
                      "rp_codons_detected": rp_codons_seen, "capacity_codons_h": cap},
               "d4": {"T_pred_h": T_pred, "T_measured_h": T_MEASURED_H,
                      "ratio": T_pred / T_MEASURED_H, "U_central": U_CENTRAL,
                      "grid": grid, "required_U": ureq, "constant_sweep": sw,
                      "replacement_fraction_at_predicted_T": f_pred,
                      "bootstrap_median_h": float(np.median(boot)),
                      "bootstrap_ci_h": [float(np.percentile(boot, 2.5)),
                                         float(np.percentile(boot, 97.5))]},
               "d5": {"T_circular_h": T_circ, "loop92_utilisation": LOOP92_UTIL,
                      "rel_error": abs(T_circ / T_MEASURED_H - 1.0)},
               "d6": {"null1_capability": cap1, "null1_survival": s1,
                      "null2_capability": cap2, "null2_survival": s2,
                      "kbar_unweighted": kbar_unw, "T_with_unweighted_kbar_h": T_unw,
                      "pubs_vs_copies": r_cop, "pubs_vs_demand": r_dem,
                      "pubs_vs_half_life": r_hl, "pubs_vs_length": r_len,
                      "pubs_auc_set_membership": a_mem},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_doubling.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_doubling.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
