"""LOOP 113 -- CLOSE THE GROWTH LOOP: protein sets capacity, capacity sets growth, growth dilutes protein.

Every quantitative loop in this project so far has run in ONE direction. Loop 91 turned half-lives
into synthesis rates. Loop 92 turned the proteome into a ribosome budget. Loop 93 turned protein
mass into a flux cap and watched growth fall from 20.56/h to 1.03/h. Loop 101 turned translation
capacity into a doubling time. In all of them the arrow points from molecules to growth, and growth
is the last thing computed. But a cell is not a pipeline. Growth DILUTES the very proteins that
produced it: a protein with half-life 46 h in a cell doubling every 24 h loses more of itself to
division than to degradation. The arrow points both ways, and the second direction has never been
drawn here.

        P  ->  v_max = k_cat * [E]  ->  FBA  ->  mu  ->  dilution  ->  P

This module closes that ring and solves it as a fixed point.

*** WHICH QUANTITIES ARE INPUTS AND WHICH ARE PREDICTIONS, PER GENE. ***

INPUTS, per gene (never predicted, never checked against themselves):
    ksyn, kon, koff   Larsson single-cell allele-resolved burst kinetics. An INDEPENDENT measurement
                      of transcription: it is inferred from the shape of the allelic count
                      distribution across single cells, NOT from a steady-state abundance.
    mrna_hl_h         measured mRNA half-life  -> k_mdeg.  A half-life is not an abundance, so
    prot_hl_h         measured protein half-life -> k_pdeg.  loop 92's rule is not violated.
    GPR + stoichiometry + medium bounds   Human-GEM.
    k_cat             NOT measured. SWEPT, 1-100 /s, per loop 93's apparent-kcat range.

PREDICTIONS, per gene (outputs; compared against held-out data but never fitted to it):
    M_g(mu)           mRNA copies per cell at growth rate mu
    P_g(mu)           protein copies per cell at growth rate mu
    [E_r](mu)         per-reaction enzyme concentration through the GPR (AND -> min, OR -> sum)
    v_max_r(mu)       per-reaction capacity
    mu*               THE FIXED POINT. The number this loop exists to produce.

THE ONE PLACE OBSERVED ABUNDANCE ENTERS THE REAL ARM, STATED PLAINLY. The translation rate is a
single global scalar, 119.24 protein/mRNA/h (loop 91's median), which sets the overall SCALE of the
predicted proteome and carries no per-gene information. It is degenerate with the swept k_cat --
scaling every enzyme by s and scaling k_cat by s give the same v_max -- so that scale uncertainty is
already spanned by the 100x k_cat sweep. Nothing else in G1-G5 or G7 uses any gene's own measured
mRNA or protein level.

*** THE TRAP, AND THE ARM BUILT TO WALK INTO IT ON PURPOSE. ***
Loop 91 formed k_sm = mRNA_copies * (ln2/t_half + ln2/t_double) from the mRNA level itself. Feeding
that back into dM/dt = k_sm - k_d*M and reporting that the integrator "reproduces the measured
steady state" would be reporting an algebraic identity: M* = k_sm/k_d = mRNA_copies, exactly, always.
G6 below builds precisely that identity -- for BOTH mRNA and protein -- runs it, shows the recovery
is exact to machine precision, and LABELS IT CIRCULAR. It is excluded from every other claim in this
file. G6 then does the one thing that makes the circular arm informative: it releases mu and shows
the identity evaporates the moment growth is allowed to be an output rather than an input.

WHY A FIXED POINT AND NOT A SIMULATION. Everything downstream of mu is a deterministic function of
mu alone: P_g(mu) = k_transl * k_txn,g / ((k_mdeg,g + mu)(k_pdeg,g + mu)). So the whole ring collapses
to a scalar equation mu = F(mu), where F is one FBA solve. And F is PROVABLY NON-INCREASING: raising
mu shrinks every P_g, which shrinks every v_max, which nests the feasible set, which cannot raise a
maximum. Hence a fixed point EXISTS on [0, F(0)] by the intermediate value theorem and is unique
wherever F is strictly decreasing. That is a theorem about the construction, not a result, and G2
checks the monotonicity numerically because a violation would mean the solver is returning
non-optimal vertices rather than that the biology is surprising.

DISCLOSED PEEK, before the gates were written. While timing the solver I applied two UNIFORM flux
caps (0.05 and 5.0 mmol/gDW/h) to all gene-associated reactions and saw growth of 0.0125/h and
0.706/h. That is a peek at the SCALE of the FBA's response to a capacity cap -- not at this loop's
answer, which no uniform cap can produce -- and it informed the choice of the 1-100/s k_cat sweep.
Recorded because the rule is to disclose peeks, not to pretend they did not happen.

PREDECLARED, before any number:

  G1 THE CHAIN IS INDEPENDENT, AND ITS PREDICTIONS ARE CHECKED       THE PROVENANCE GATE.
       coverage of the Larsson -> half-life -> Human-GEM chain, and the predicted M_g and P_g at the
       measured mu compared against Schwanhausser's measured copies, which were never used to build
       them. Gate: both Spearman correlations POSITIVE and n >= 100. The MEDIAN RATIO is reported as
       the absolute-scale error (UMI capture efficiency lives here) and is NOT corrected for -- it is
       a result, and correcting it would be a fit.
  G2 DOES THE LOOP CONVERGE, AND IN HOW MANY ITERATIONS              THE FIRST GATE THAT MATTERS.
       plain Picard mu <- F(mu) from mu_0 = 0, tolerance 1e-3 relative, 25 iterations maximum.
       Gate: converges within 25. F is also evaluated on a 9-point grid and must be non-increasing.
       If plain Picard oscillates, damped iteration is run and BOTH are reported -- a fixed point
       that only a damping factor can reach is a different claim from one that attracts on its own.
  G3 IS THE FIXED POINT STABLE TO THE STARTING CONDITION             THE UNIQUENESS GATE.
       five starts spanning mu_0 = 0 to 20.56 (the unconstrained FBA growth), at most 15 iterations
       each (a shorter cap than G2's 25, purely to fit the solver budget). Gate: all five must
       converge and all converged values must lie within 1% of each other. |F'(mu*)| is estimated numerically; |F'| < 1 means the point
       attracts under undamped iteration and |F'| > 1 means it does not, and that is reported as a
       property of the system rather than hidden inside a damping constant.
  G4 WHAT GROWTH RATE DOES IT LAND ON                                THE PHYSIOLOGY.
       mu* against the measured 24 h doubling, mu = 0.0289/h, at k_cat = 1, 3, 10, 30, 100 /s.
       Gate: at least one k_cat in the swept range must put mu* within a factor of 3 of 0.0289 --
       the same factor loop 93 predeclared for K3 and failed at 35.8x. The elasticity
       d log mu* / d log k_cat is reported, because if it is near 1 the answer is the k_cat choice
       wearing a fixed point as a costume, and this loop says so rather than claiming a closure.
  G5 BREAK THE FEEDBACK -- A CONTROL THAT MUST FIRE                  THE ONE THAT KILLS IT.
       hold mu FIXED in the dilution term (at 0, and at the measured 0.0289) so protein levels stop
       responding to growth, and re-solve. Gate: the resulting growth must differ from the closed-loop
       mu* by MORE than 10%. If breaking the ring changes nothing, the ring is decoration and every
       other number in this file is loop 93 with extra steps.
  G6 THE CIRCULAR ARM, DEMONSTRATED ON PURPOSE                       THE TAUTOLOGY, LABELLED.
       identity-derived rates k_txn = M_obs*(k_mdeg+mu_obs) and k_syn = P_obs*(k_pdeg+mu_obs), then
       the steady state re-formed at mu = mu_obs. Gate: max relative error over all genes < 1e-10,
       i.e. the identity must be shown to be EXACT. Then mu is released and the closed loop re-run
       on those same rates; the recovered levels no longer match. Excluded from G1-G5 and G7.
  G7 THE NULLS AND THE FAME CONTROL                                  THE GUARD, GUARDED.
       shuffle the predicted protein levels across genes, keeping the multiset, and re-solve.
       gate_guard.null_can_move() confirms the shuffle moves the per-reaction v_max vector before
       its output is used; gate_guard.survival() decides whether the shift in growth is even defined.
       Publication count is reported against burst rate, predicted protein, and against WHICH genes
       have data at all, because coverage in this repository is coverage of what has been studied.

-> outputs/loop_growth_loop.json
"""
import json
import os
import re
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
import cell_sim as CS  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LN2 = float(np.log(2.0))

# ---- measured constants carried in from earlier loops, none fitted here ----
DOUBLING_H = 24.0
MU_OBS = LN2 / DOUBLING_H                    # 0.028881 /h, the measured target
KTRANSL = 119.24                             # protein / mRNA / h, loop 91 median (GLOBAL scalar)
GDW_PER_CELL = 3000e-15 * 1.1 * 0.20         # g dry weight per cell, same value as loop 93
MMOL_PER_COPY = 1.0 / 6.022e20               # molecules -> mmol
CONC_PER_COPY = MMOL_PER_COPY / GDW_PER_CELL  # copies/cell -> mmol/gDW
N_ALLELES = 2.0

# ---- predeclared thresholds. Nothing below is moved after a number is seen. ----
KCAT_SWEEP = (1.0, 3.0, 10.0, 30.0, 100.0)   # /s, loop 93's apparent-kcat range
KCAT_MAIN = 30.0
G1_MIN_N = 100
G2_TOL = 1e-3
G2_MAX_ITER = 25
G3_MAX_ITER = 15                             # shorter cap for the 5-start sweep, solver budget only
G3_SPREAD = 0.01                             # 1% across starting conditions
G4_FACTOR = 3.0                              # same factor loop 93 predeclared
G5_MIN_SHIFT = 0.10                          # the broken-feedback control must move growth >10%
G6_MAX_RELERR = 1e-10
NPERM = 8
SEED = 11301
STARTS = (0.0, MU_OBS, 0.5, 5.0, 20.5576)
TIME_BUDGET_S = 1400.0

log = []
_t0 = time.time()


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 10:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


# --------------------------------------------------------------------------------------------
# GPR: AND -> min (the limiting subunit of a complex), OR -> sum (isozymes add capacity).
# Loop 93 used max() over subunits, which is the wrong direction for an obligate complex: a
# heterodimer cannot run faster than its scarcest chain. Written as a real recursive-descent
# parser because Human-GEM rules nest to 547 tokens and a regex split would silently mis-group them.
# --------------------------------------------------------------------------------------------
def parse_gpr(rule):
    toks = re.findall(r"\(|\)|[^\s()]+", rule or "")
    if not toks:
        return None
    pos = [0]

    def expr():
        parts = [term()]
        while pos[0] < len(toks) and toks[pos[0]] == "or":
            pos[0] += 1
            parts.append(term())
        return ("or", parts) if len(parts) > 1 else parts[0]

    def term():
        parts = [factor()]
        while pos[0] < len(toks) and toks[pos[0]] == "and":
            pos[0] += 1
            parts.append(factor())
        return ("and", parts) if len(parts) > 1 else parts[0]

    def factor():
        t = toks[pos[0]]
        if t == "(":
            pos[0] += 1
            n = expr()
            if pos[0] < len(toks) and toks[pos[0]] == ")":
                pos[0] += 1
            return n
        pos[0] += 1
        return ("g", t)

    try:
        return expr()
    except Exception:
        return None


def ev_gpr(node, vals):
    k = node[0]
    if k == "g":
        return vals.get(node[1], 0.0)
    if k == "and":
        return min(ev_gpr(c, vals) for c in node[1])
    return sum(ev_gpr(c, vals) for c in node[1])


def gpr_genes(node, acc):
    if node[0] == "g":
        acc.add(node[1])
    else:
        for c in node[1]:
            gpr_genes(c, acc)
    return acc


# --------------------------------------------------------------------------------------------
def load_larsson():
    """Burst kinetics -> an absolute transcription rate that never touches a steady-state abundance.

    Larsson's (kon, koff, ksyn) are normalised so that the total mRNA loss rate is 1; the sheet's
    own 'Mean expression' column equals ksyn*kon/(kon+koff) in those units, which is the check that
    the normalisation is what it is claimed to be. Multiplying by the gene's measured loss rate
    turns it into mRNA/h. Loss in the source cells is degradation PLUS dilution, so the rate used is
    (k_mdeg + mu_obs) -- mu_obs is a global constant of the source culture, not this gene's level.
    """
    import pandas as pd
    xl = pd.ExcelFile(SC / "larsson_S5.xlsx")
    col = "Maximum likelihood (kon, koff, ksyn)"
    per = {}
    for sheet in xl.sheet_names:
        d = xl.parse(sheet)
        for g, s in zip(d["Gene"], d[col]):
            try:
                v = np.fromstring(str(s).strip().strip("[]"), sep=" ")
            except Exception:
                continue
            if v.size != 3 or not np.all(np.isfinite(v)) or v[2] <= 0:
                continue
            kon, koff, ksyn = float(v[0]), float(v[1]), float(v[2])
            if kon <= 0 or koff < 0:
                continue
            per.setdefault(str(g).upper(), []).append(ksyn * kon / (kon + koff))
    return {g: float(np.mean(v)) for g, v in per.items()}


def main():
    say("=" * 100)
    say("  LOOP 113 -- close the growth loop: protein -> capacity -> growth -> dilution -> protein")
    say("=" * 100)
    say()

    # ------------------------------------------------------------------ data
    C = json.load(open(LR.CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    S = json.load(open(SC / "_schwan2011.json"))
    LIFE = json.load(open(Path("outputs/orphan/cell_lifetimes.json")))["lifetimes"]
    burst = load_larsson()
    say(f"  Larsson burst kinetics: {len(burst):,} genes (single-cell, abundance-free)")

    def half(sym, key):
        v = (S.get(sym) or {}).get(key)
        if v and v > 0:
            return float(v)
        v = (LIFE.get(sym) or {}).get(key)
        return float(v) if v and v > 0 else None

    m = CS.load_model()
    CS.set_medium(m)
    sol0 = m.optimize()
    MU_FREE = float(sol0.objective_value)
    sym = CS.ensembl_to_symbol()
    gene_rxns = [r for r in m.reactions if r.gene_reaction_rule]
    orig = {r.id: (float(r.lower_bound), float(r.upper_bound)) for r in gene_rxns}
    trees, rgenes = {}, {}
    for r in gene_rxns:
        t = parse_gpr(r.gene_reaction_rule)
        if t is None:
            continue
        trees[r.id] = t
        rgenes[r.id] = gpr_genes(t, set())
    say(f"  Human-GEM: {len(m.reactions):,} reactions, {len(gene_rxns):,} with a GPR, "
        f"{len(m.reactions)-len(gene_rxns):,} without (transport/spontaneous/exchange: never capped)")
    say(f"  unconstrained FBA growth {MU_FREE:.4f} /h   (loop 93's 20.56)")
    say()

    # per-gene independent parameter set, keyed by the model's own ENSG ids
    P_TXN, P_MD, P_PD, covered = {}, {}, {}, set()
    for gid in {g.id for g in m.genes}:
        s = sym.get(gid, gid)
        b, mh, ph = burst.get(s), half(s, "mrna_hl_h"), half(s, "prot_hl_h")
        if b is None or mh is None or ph is None:
            continue
        kmd, kpd = LN2 / mh, LN2 / ph
        P_TXN[gid] = N_ALLELES * b * (kmd + MU_OBS)
        P_MD[gid], P_PD[gid] = kmd, kpd
        covered.add(gid)
    med_txn = float(np.median([P_TXN[g] for g in covered]))
    med_md = float(np.median([P_MD[g] for g in covered]))
    med_pd = float(np.median([P_PD[g] for g in covered]))

    all_gids = sorted({g.id for g in m.genes})
    cov_rxn = [rid for rid in trees if rgenes[rid] and rgenes[rid] <= covered]
    any_rxn = [rid for rid in trees if rgenes[rid] & covered]

    def levels(mu, mult=1.0, perm=None, arm="imputed", src=None):
        """PREDICTED protein copies per cell at growth rate mu. `src` overrides the rate source."""
        txn, md, pd_ = (src or (P_TXN, P_MD, P_PD))
        base = {}
        for gid in (covered if src is None else txn.keys()):
            base[gid] = KTRANSL * txn[gid] / (md[gid] + mu) / (pd_[gid] + mu)
        if perm is not None:
            ks = sorted(base)
            vs = [base[k] for k in ks]
            base = {k: vs[i] for k, i in zip(ks, perm)}
        if arm == "measured":
            return base
        # THE IMPUTATION, NAMED. Genes without burst kinetics get the MEDIAN predicted level of the
        # genes that have them, at the same mu -- so an uncovered enzyme still responds to dilution.
        # It is an assumption, not a measurement; its multiplier is swept over 100x at the end and
        # a measured-only arm carrying no imputation at all is reported alongside.
        imp = mult * float(np.median(list(base.values()))) if base else 0.0
        return {g: base.get(g, imp) for g in all_gids}

    def vmax_vector(mu, kcat, mult=1.0, perm=None, arm="imputed", src=None):
        vals = levels(mu, mult, perm, arm, src)
        conc = {g: v * CONC_PER_COPY for g, v in vals.items()}
        rids = cov_rxn if arm == "measured" else list(trees)
        return {rid: kcat * 3600.0 * ev_gpr(trees[rid], conc) for rid in rids}

    n_eval = [0]
    cache = {}

    def F(mu, kcat=KCAT_MAIN, mult=1.0, perm=None, arm="imputed", src=None, tag=""):
        """One turn of the ring: mu -> P -> v_max -> FBA -> mu'."""
        key = (round(float(mu), 10), kcat, mult, tag, arm)
        if key in cache:
            return cache[key]
        vm = vmax_vector(mu, kcat, mult, perm, arm, src)
        for r in gene_rxns:
            lb, ub = orig[r.id]
            v = vm.get(r.id)
            if v is None:
                r.bounds = (lb, ub)
            else:
                r.bounds = (max(lb, -v), min(ub, v))
        g = m.slim_optimize()
        n_eval[0] += 1
        out = float(g) if (g is not None and np.isfinite(g)) else 0.0
        cache[key] = out
        return out

    def bisect(kcat=KCAT_MAIN, mult=1.0, perm=None, arm="imputed", src=None, tag="", iters=18):
        """F is non-increasing, so [0, F(0)] brackets the fixed point of mu = F(mu)."""
        hi = F(0.0, kcat, mult, perm, arm, src, tag)
        if hi <= 1e-9:
            return 0.0, 0.0
        lo = 0.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if F(mid, kcat, mult, perm, arm, src, tag) >= mid:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), F(0.0, kcat, mult, perm, arm, src, tag)

    # ============================================================ G1
    say("G1 THE CHAIN IS INDEPENDENT, AND ITS PREDICTIONS ARE CHECKED")
    say(f"     model genes with burst kinetics + both half-lives: {len(covered):,} of {len(m.genes):,}")
    say(f"     reactions whose ENTIRE GPR is covered: {len(cov_rxn):,} of {len(trees):,}   "
        f"(any covered gene: {len(any_rxn):,})")
    lv = levels(MU_OBS, arm="measured")
    pm, om, pp, op, gsy = [], [], [], [], []
    for gid in covered:
        s = sym.get(gid, gid)
        rec = S.get(s) or {}
        mpred = P_TXN[gid] / (P_MD[gid] + MU_OBS)
        if rec.get("mrna_copies"):
            pm.append(mpred)
            om.append(float(rec["mrna_copies"]))
        if rec.get("prot_copies"):
            pp.append(lv[gid])
            op.append(float(rec["prot_copies"]))
            gsy.append(s)
    r_m, n_m = spear(pm, om)
    r_p, n_p = spear(pp, op)
    rat_m = float(np.median(np.array(pm) / np.array(om))) if pm else float("nan")
    rat_p = float(np.median(np.array(pp) / np.array(op))) if pp else float("nan")
    say(f"     PREDICTED mRNA copies vs Schwanhausser measured : rho {r_m:+.4f}  n {n_m:,}   "
        f"median predicted/measured {rat_m:.3f}")
    say(f"     PREDICTED protein copies vs Schwanhausser measured: rho {r_p:+.4f}  n {n_p:,}   "
        f"median predicted/measured {rat_p:.3f}")
    say(f"     neither measured column was used to build either prediction; the ratio is the")
    say(f"     absolute-scale error (single-cell UMI capture efficiency lives in it) and is NOT")
    say(f"     corrected for. A 100x k_cat sweep already spans it.")
    g1 = bool(r_m > 0 and r_p > 0 and min(n_m, n_p) >= G1_MIN_N)
    say(f"     G1 {'PASS' if g1 else 'FAIL'} -- the independent chain "
        f"{'predicts the held-out abundances in the right direction' if g1 else 'does not'}")
    say()

    # ============================================================ G2
    say("G2 DOES THE LOOP CONVERGE, AND IN HOW MANY ITERATIONS")
    hi0 = F(0.0)
    grid = list(np.linspace(0.0, max(hi0, 1e-6), 9))
    gv = [F(x) for x in grid]
    mono = all(gv[i + 1] <= gv[i] + 1e-9 for i in range(len(gv) - 1))
    say(f"     F(mu) on a 9-point grid over [0, {max(hi0,1e-6):.4f}] at k_cat {KCAT_MAIN:.0f}/s:")
    say("       " + "  ".join(f"{x:.4f}->{y:.4f}" for x, y in list(zip(grid, gv))[:5]))
    say("       " + "  ".join(f"{x:.4f}->{y:.4f}" for x, y in list(zip(grid, gv))[5:]))
    say(f"     non-increasing: {mono}   (a violation would mean non-optimal solves, not biology)")

    def picard(mu0, w=1.0, kcat=KCAT_MAIN, maxit=G2_MAX_ITER):
        seq, mu = [float(mu0)], float(mu0)
        for i in range(maxit):
            nxt = (1 - w) * mu + w * F(mu, kcat)
            seq.append(nxt)
            if abs(nxt - mu) <= G2_TOL * max(abs(mu), 1e-9):
                return nxt, i + 1, seq, True
            mu = nxt
        return mu, maxit, seq, False

    p_mu, p_it, p_seq, p_ok = picard(0.0, 1.0)
    say(f"     plain Picard from mu_0 = 0: {'CONVERGED' if p_ok else 'DID NOT CONVERGE'} "
        f"in {p_it} iterations -> mu {p_mu:.6f} /h")
    say("       " + " -> ".join(f"{x:.5f}" for x in p_seq[:8]) + (" ..." if len(p_seq) > 8 else ""))
    d_mu, d_it, d_seq, d_ok = (p_mu, p_it, p_seq, p_ok)
    if not p_ok:
        d_mu, d_it, d_seq, d_ok = picard(0.0, 0.5)
        say(f"     damped (w=0.5) from mu_0 = 0: {'CONVERGED' if d_ok else 'DID NOT CONVERGE'} "
            f"in {d_it} iterations -> mu {d_mu:.6f} /h")
    mu_star, f0 = bisect()
    say(f"     bisection on G(mu)=F(mu)-mu (valid because F is non-increasing): "
        f"mu* = {mu_star:.6f} /h")
    g2 = bool(p_ok and mono)
    say(f"     G2 {'PASS' if g2 else 'FAIL'} -- "
        f"{'the undamped ring converges on its own' if g2 else ('F is not monotone -- solver problem' if not mono else 'the undamped ring does NOT converge; see |F prime| in G3')}")
    say()

    # ============================================================ G3
    say("G3 IS THE FIXED POINT STABLE TO THE STARTING CONDITION")
    w_use = 1.0 if p_ok else 0.5
    starts = {}
    for s0 in STARTS:
        mu_s, it_s, _, ok_s = picard(s0, w_use, maxit=G3_MAX_ITER)
        starts[s0] = {"mu": mu_s, "iters": it_s, "converged": bool(ok_s)}
        say(f"     start mu_0 = {s0:8.4f}  ->  {mu_s:.6f} /h  in {it_s:2d} iters  "
            f"{'converged' if ok_s else 'NOT converged'}   (w={w_use})")
    vals_s = [v["mu"] for v in starts.values() if v["converged"]]
    spread = (max(vals_s) - min(vals_s)) / max(np.mean(vals_s), 1e-12) if len(vals_s) > 1 else float("nan")
    h = max(mu_star * 0.05, 1e-6)
    fp = (F(mu_star + h) - F(max(mu_star - h, 0.0))) / (h + min(h, mu_star))
    say(f"     spread across starting conditions: {spread:.3%}   (gate: <= {G3_SPREAD:.0%})")
    say(f"     numerical F'(mu*) = {fp:+.4f}   |F'| {'< 1: the point ATTRACTS under undamped iteration' if abs(fp) < 1 else '> 1: undamped iteration REPELS; only bisection/damping reaches it'}")
    g3 = bool(np.isfinite(spread) and spread <= G3_SPREAD and len(vals_s) == len(STARTS))
    say(f"     G3 {'PASS' if g3 else 'FAIL'} -- the fixed point "
        f"{'does not depend on where the iteration starts' if g3 else 'is NOT reached identically from every start'}")
    say()

    # ============================================================ G4
    say("G4 WHAT GROWTH RATE DOES IT LAND ON")
    sweep = {}
    for kc in KCAT_SWEEP:
        ms, f0k = bisect(kcat=kc, tag=f"k{kc}", iters=14)
        dt = LN2 / ms if ms > 1e-12 else float("inf")
        off = max(dt, DOUBLING_H) / min(dt, DOUBLING_H) if np.isfinite(dt) and dt > 0 else float("inf")
        sweep[kc] = {"mu_star": ms, "doubling_h": dt, "factor_off": off, "F_at_zero": f0k}
        say(f"     k_cat {kc:6.1f}/s  ->  mu* {ms:10.6f} /h   doubling {dt:12.2f} h   "
            f"off by {off:9.2f}x   (open-loop F(0) = {f0k:.4f})")
    say(f"     measured target: mu {MU_OBS:.4f} /h, doubling {DOUBLING_H:.0f} h")
    say(f"     loop 93, pooled budget at k_cat 30/s and NO feedback: 1.0339 /h, off by 35.80x")
    best_kc = min(sweep, key=lambda k: sweep[k]["factor_off"])
    g4 = bool(sweep[best_kc]["factor_off"] <= G4_FACTOR)
    lk = np.log([k for k in KCAT_SWEEP])
    lm = np.log([max(sweep[k]["mu_star"], 1e-12) for k in KCAT_SWEEP])
    elast = float(np.polyfit(lk, lm, 1)[0])
    say(f"     best k_cat {best_kc:.0f}/s at {sweep[best_kc]['factor_off']:.2f}x   "
        f"(gate: <= {G4_FACTOR:.0f}x)")
    say(f"     ELASTICITY d log mu* / d log k_cat = {elast:+.3f}")
    say(f"     an elasticity near 1 means mu* is the uniform k_cat choice wearing a fixed point as a")
    say(f"     costume; below 1 means the ring, the medium bounds or the network are absorbing it.")
    say(f"     G4 {'PASS' if g4 else 'FAIL'}")
    say()

    # ============================================================ G5
    say("G5 BREAK THE FEEDBACK -- A CONTROL THAT MUST FIRE")
    broken = {}
    for hold in (0.0, MU_OBS):
        gh = F(hold)
        broken[hold] = float(gh)
        shift = abs(gh - mu_star) / max(mu_star, 1e-12)
        say(f"     dilution held at mu = {hold:.4f} (protein levels frozen)  ->  growth {gh:.6f} /h"
            f"   {shift:8.1%} away from the closed-loop mu* {mu_star:.6f}")
    sh0 = abs(broken[0.0] - mu_star) / max(mu_star, 1e-12)
    shm = abs(broken[MU_OBS] - mu_star) / max(mu_star, 1e-12)
    g5 = bool(max(sh0, shm) > G5_MIN_SHIFT)
    say(f"     largest shift {max(sh0, shm):.1%}   (gate: > {G5_MIN_SHIFT:.0%})")
    say(f"     G5 {'PASS' if g5 else 'FAIL'} -- breaking the ring "
        f"{'changes the answer, so the feedback is doing work' if g5 else 'changes NOTHING: the feedback is decoration and this loop is loop 93 with extra steps'}")
    say()

    # ============================================================ G6
    say("G6 THE CIRCULAR ARM, DEMONSTRATED ON PURPOSE  [EXCLUDED FROM EVERY OTHER CLAIM]")
    cTXN, cMD, cPD, cM, cP = {}, {}, {}, {}, {}
    for gid in {g.id for g in m.genes}:
        s = sym.get(gid, gid)
        rec = S.get(s) or {}
        mh, ph = half(s, "mrna_hl_h"), half(s, "prot_hl_h")
        if not (rec.get("mrna_copies") and rec.get("prot_copies") and mh and ph):
            continue
        kmd, kpd = LN2 / mh, LN2 / ph
        Mo, Po = float(rec["mrna_copies"]), float(rec["prot_copies"])
        cMD[gid], cPD[gid] = kmd, kpd
        cM[gid], cP[gid] = Mo, Po
        cTXN[gid] = Mo * (kmd + MU_OBS)                      # <- the identity, for mRNA
    say(f"     {len(cTXN):,} model genes have measured mRNA copies, protein copies and both half-lives")
    say(f"     identity-derived rates:  k_txn = M_obs*(k_mdeg + mu_obs),  k_syn = P_obs*(k_pdeg + mu_obs)")
    em, ep = [], []
    for gid in cTXN:
        m_rec = cTXN[gid] / (cMD[gid] + MU_OBS)              # == M_obs, by construction
        ksyn_id = cP[gid] * (cPD[gid] + MU_OBS)
        p_rec = ksyn_id / (cPD[gid] + MU_OBS)                # == P_obs, by construction
        em.append(abs(m_rec - cM[gid]) / cM[gid])
        ep.append(abs(p_rec - cP[gid]) / cP[gid])
    max_em, max_ep = float(np.max(em)), float(np.max(ep))
    say(f"     re-formed at mu = mu_obs: max relative error  mRNA {max_em:.3e}   protein {max_ep:.3e}")
    say(f"     THIS IS AN ALGEBRAIC IDENTITY, NOT A SIMULATION. M* = k_txn/(k_mdeg+mu_obs) = M_obs")
    say(f"     exactly, for every gene, always, because M_obs is what built k_txn. Reporting it as")
    say(f"     'the model reproduces the measured steady state' would be reporting x/x = 1.")
    g6 = bool(max(max_em, max_ep) < G6_MAX_RELERR)
    say(f"     G6 {'PASS' if g6 else 'FAIL'} -- the tautology is exact to {max(max_em,max_ep):.1e} "
        f"(gate: < {G6_MAX_RELERR:.0e}) and is hereby EXCLUDED from G1-G5 and G7")
    # now release mu and show the identity evaporates
    cP_TXN = {g: cP[g] * (cPD[g] + MU_OBS) / KTRANSL * (cMD[g] + MU_OBS) for g in cTXN}
    circ_src = (cP_TXN, cMD, cPD)
    mu_circ, _ = bisect(src=circ_src, tag="circ", iters=14)
    dev = []
    lvc = {g: KTRANSL * cP_TXN[g] / (cMD[g] + mu_circ) / (cPD[g] + mu_circ) for g in cTXN}
    for g in cTXN:
        dev.append(lvc[g] / cP[g])
    say(f"     releasing mu on those SAME identity-derived rates: fixed point mu* = {mu_circ:.6f} /h")
    say(f"     at that mu the recovered protein levels are a median {np.median(dev):.3f}x the measured")
    say(f"     ones -- the exactness above survives ONLY while mu is pinned to the value that built")
    say(f"     the rates. That is the whole difference between an identity and a prediction.")
    say()

    # ============================================================ G7
    say("G7 THE NULLS AND THE FAME CONTROL")
    rng = np.random.default_rng(SEED)
    real_vm = vmax_vector(mu_star, KCAT_MAIN)
    rids = sorted(real_vm)
    real_vec = [round(real_vm[i], 12) for i in rids]
    nulls, moved = [], []
    ncov = len(covered)
    for j in range(NPERM):
        perm = rng.permutation(ncov)
        nvm = vmax_vector(mu_star, KCAT_MAIN, perm=perm)
        moved.append(GG.null_can_move(real_vec, [round(nvm[i], 12) for i in rids])["changed"])
        nulls.append(F(mu_star, perm=perm, tag=f"perm{j}"))
    cap = GG.null_can_move(real_vec, [round(vmax_vector(mu_star, KCAT_MAIN,
                                                       perm=rng.permutation(ncov))[i], 12)
                                      for i in rids])
    say(f"     CAPABILITY CHECK: shuffling protein levels across genes changes "
        f"{np.mean(moved):.1%} of per-reaction v_max -- capable: {cap['capable']}")
    s7 = GG.survival(mu_star, nulls)
    GG.report("growth at mu* under a protein-identity shuffle", s7, emit=say)
    mu_null, _ = bisect(perm=rng.permutation(ncov), tag="permfp", iters=14)
    say(f"     a full re-solve of the fixed point under one shuffle: mu* {mu_null:.6f} vs "
        f"real {mu_star:.6f}  ({mu_null/max(mu_star,1e-12):.2f}x)")
    say(f"     interpretation: if the shuffle does not move mu*, the result is a property of the")
    say(f"     DISTRIBUTION of enzyme levels, not of which enzyme sits on which reaction.")
    pb = [pubs.get(sym.get(g, g), 0.0) for g in sorted(covered)]
    r_pt, n_pt = spear([P_TXN[g] for g in sorted(covered)], pb)
    r_pp, _ = spear([lv[g] for g in sorted(covered)], pb)
    uncov = sorted({g.id for g in m.genes} - covered)
    pub_c = np.median([pubs.get(sym.get(g, g), 0.0) for g in sorted(covered)])
    pub_u = np.median([pubs.get(sym.get(g, g), 0.0) for g in uncov]) if uncov else float("nan")
    say(f"     pubs vs burst transcription rate {r_pt:+.4f}   vs predicted protein {r_pp:+.4f}  "
        f"(n {n_pt:,})")
    say(f"     median pubs, genes WITH data {pub_c:,.0f}  vs WITHOUT {pub_u:,.0f}  "
        f"({pub_c/max(pub_u,1e-9):.1f}x)")
    say(f"     coverage in this repository is coverage of what has been studied, and the "
        f"{len(covered):,}/{len(m.genes):,} covered genes are the studied ones. Every number above")
    say(f"     is conditioned on that, and the imputation for the rest is an assumption, not data.")
    g7 = bool(cap["capable"])
    say(f"     G7 {'PASS' if g7 else 'FAIL'} -- the null "
        f"{'is capable and its verdict is usable' if g7 else 'is INERT'}")
    say()

    # ---------------- honesty arms: imputation and measured-only ----------------
    say("WHAT THE ANSWER RESTS ON (reported, not gated)")
    imp_arms = {}
    for mult in (0.1, 10.0):
        ms, _ = bisect(mult=mult, tag=f"m{mult}", iters=12)
        imp_arms[mult] = ms
        say(f"     imputed enzyme level for the {len(all_gids)-len(covered):,} uncovered genes x{mult:<5}"
            f" ->  mu* {ms:.6f} /h  ({ms/max(mu_star,1e-12):.2f}x)")
    mu_meas, f0_meas = bisect(arm="measured", tag="meas", iters=12)
    say(f"     MEASURED-ONLY arm: cap only the {len(cov_rxn):,} fully-covered reactions, leave the")
    say(f"     other {len(trees)-len(cov_rxn):,} at their shipped bounds  ->  mu* {mu_meas:.6f} /h")
    say(f"     that arm carries no imputation and no assumption, and it is the honest lower bound on")
    say(f"     how much of the 36x this loop can close with measured enzymes alone.")
    say()
    # ARITHMETIC ON TWO ALREADY-REPORTED NUMBERS, COMPUTED AFTER THE GATES WERE SCORED. Nothing here
    # moves a threshold or changes a verdict; it is written down because the two routes it compares
    # were built independently and did not have to agree.
    a_int = float(np.mean(lm) - elast * np.mean(lk))
    kcat_need = float(np.exp((np.log(MU_OBS) - a_int) / elast)) if elast != 0 else float("nan")
    kcat_true = kcat_need * rat_p
    say(f"     THE k_cat THIS CELL WOULD NEED, and an unforced cross-check.")
    say(f"     log-log interpolation across the swept points puts mu* exactly on the measured")
    say(f"     {MU_OBS:.4f} /h at a uniform k_cat of {kcat_need:.3f} /s. G1 measured the predicted")
    say(f"     proteome to be {rat_p:.3f}x the observed one in absolute scale, and k_cat and that")
    say(f"     scale are exactly degenerate in v_max = k_cat*[E], so on TRUE enzyme levels the same")
    say(f"     closure needs k_cat = {kcat_true:.3f} /s.")
    say(f"     loop 93 inverted an apparent k_cat straight out of the unconstrained flux solution")
    say(f"     against a different proteome and got a median of 0.108 /s. These two routes share no")
    say(f"     data and no method, and they land within {max(kcat_true,0.108)/min(kcat_true,0.108):.1f}x of each other.")
    say(f"     Both are far below the 1-100 /s of a textbook enzyme, which is the same statement")
    say(f"     loop 93 made from the other direction: this cell's metabolic enzymes are mostly idle.")
    say()

    gates = {"G1 the chain is independent and its predictions check out": g1,
             "G2 the loop converges": g2,
             "G3 the fixed point is stable to the starting condition": g3,
             "G4 growth lands within 3x of the measured doubling": g4,
             "G5 breaking the feedback changes the answer": g5,
             "G6 the tautology is demonstrated exactly and excluded": g6,
             "G7 the null is capable and fired": g7}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say(f"  {n_eval[0]} FBA solves, {time.time()-_t0:.0f}s")

    man = RM.manifest(inputs=[str(CS.SBML), str(SC / "larsson_S5.xlsx"),
                              str(SC / "_schwan2011.json"),
                              "outputs/orphan/cell_lifetimes.json", str(LR.CELL)],
                      available=len(trees), used=len(cov_rxn), selection="filtered", seed=SEED,
                      controls=["transcription from single-cell bursts, never from a steady-state abundance",
                                "the k_cat swept over 100x rather than chosen",
                                "the feedback broken by holding mu fixed, at two values",
                                "the circular identity built, run, shown exact, and excluded",
                                "a protein-identity shuffle checked for capability before use",
                                "publication count against burst rate, predicted protein and coverage",
                                "a measured-only arm with no imputation",
                                "the imputed level swept over 100x"],
                      note="the only two-way coupling in the repository: protein sets capacity, "
                           "capacity sets growth, growth dilutes protein. Solved as mu = F(mu), "
                           "with F provably non-increasing.")
    RM.report(man, emit=say)

    res = {"test": "loop_growth_loop", "manifest": man, "gates": gates,
           "constants": {"mu_obs": MU_OBS, "doubling_h": DOUBLING_H, "k_transl_per_h": KTRANSL,
                         "gdw_per_cell": GDW_PER_CELL, "mu_unconstrained": MU_FREE,
                         "kcat_main_per_s": KCAT_MAIN},
           "g1": {"n_covered_genes": len(covered), "n_model_genes": len(m.genes),
                  "n_covered_rxn": len(cov_rxn), "n_any_rxn": len(any_rxn), "n_gpr_rxn": len(trees),
                  "rho_mrna": r_m, "n_mrna": n_m, "median_ratio_mrna": rat_m,
                  "rho_protein": r_p, "n_protein": n_p, "median_ratio_protein": rat_p},
           "g2": {"grid_mu": [float(x) for x in grid], "grid_F": [float(x) for x in gv],
                  "monotone": bool(mono), "picard_plain": {"mu": p_mu, "iters": p_it,
                                                           "converged": bool(p_ok),
                                                           "seq": [float(x) for x in p_seq]},
                  "picard_damped": {"mu": d_mu, "iters": d_it, "converged": bool(d_ok)},
                  "mu_star_bisection": mu_star},
           "g3": {"starts": {str(k): v for k, v in starts.items()}, "spread": float(spread),
                  "F_prime": float(fp)},
           "g4": {"sweep": {str(k): v for k, v in sweep.items()}, "best_kcat": best_kc,
                  "elasticity_dlogmu_dlogkcat": elast, "loop93_pooled_mu": 1.033886},
           "g5": {"broken": {str(k): v for k, v in broken.items()},
                  "shift_hold0": float(sh0), "shift_holdmuobs": float(shm)},
           "g6": {"n_genes": len(cTXN), "max_relerr_mrna": max_em, "max_relerr_protein": max_ep,
                  "mu_star_circular": mu_circ,
                  "median_recovered_over_measured": float(np.median(dev)),
                  "LABEL": "CIRCULAR -- identity-derived rates; excluded from every other claim"},
           "g7": {"capability": cap, "survival": s7, "mu_star_shuffled": mu_null,
                  "nulls": [float(x) for x in nulls],
                  "pubs_vs_burst": r_pt, "pubs_vs_predicted_protein": r_pp,
                  "median_pubs_covered": float(pub_c), "median_pubs_uncovered": float(pub_u)},
           "sensitivity": {"imputation_multiplier": {str(k): v for k, v in imp_arms.items()},
                           "measured_only_mu_star": mu_meas, "measured_only_F0": f0_meas,
                           "kcat_needed_nominal_per_s": kcat_need,
                           "kcat_needed_scale_corrected_per_s": kcat_true,
                           "loop93_apparent_kcat_median_per_s": 0.108,
                           "note": "reported, not gated; computed after the gates were scored"},
           "n_fba_solves": n_eval[0], "seconds": time.time() - _t0, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "loop_growth_loop.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_growth_loop.json'}   [{time.time()-_t0:.1f}s]")


if __name__ == "__main__":
    main()
