"""THE LOOP -- the first thing in this repository that is a model rather than a catalogue.

WHY THIS EXISTS.  The audit in colab/cell_model_audit.py measured seven criteria on cell_complete.json
and the decisive one came back negative: the coupling graph between its 42 blocks is a pure DAG. Zero
cycles. Expression predicts essentiality, essentiality is scored against dependency, dependency is looked
up -- and nothing ever flows back to change the expression it started from. Every block is downstream of
some other block and the arrows never close. A thing with no cycles is a catalogue with predictors
attached, however large. A cell is not that; a cell is a fixed point.

So this builds one cycle, end to end, and measures whether it earns its place.

THE CYCLE
                      transcription     translation      capacity        FBA
        k_syn  ------------------>  m  ----------->  P  --------->  v  ------>  mu
          ^                         ^               ^                           |
          |                         |               |                           |
          +------- dilution --------+---------------+---------------------------+

Growth dilutes what it is made of. Both balances carry mu on the loss side:

        dm/dt = k_syn      - (k_deg  + mu) m           mRNA
        dP/dt = k_tl * m   - (k_dp   + mu) P           protein

so at steady state

        m(mu) = k_syn / (k_deg + mu)          P(mu) = k_tl k_syn / ((k_dp + mu)(k_deg + mu))

and mu itself comes from an FBA whose reaction capacities are set by those very P. mu appears on both
sides. That is the whole point: it is not a pipeline that can be evaluated left to right, it is an
equation mu = F(mu) that has to be solved. The repo has never contained one.

THE MECHANISM ACTUALLY UNDER TEST, stated before any number, because it is the only reason the loop could
possibly beat the same model without it.  A UNIFORM change in capacity cannot change the RANKING of
knockouts at all -- rescale every bound by the same factor and every knockout's relative growth drop is
essentially unchanged. If dilution were a global knob the loop would be enzyme-constrained FBA with extra
steps and it would deserve to be called that. It is not global, and the reason is measured, not assumed:
the pooled mRNA half-lives used here span 0.22 h to 58 h, so at mu ~ 0.03/h the dilution term
mu/(k_deg + mu) removes 71% of the most stable transcript and 0.9% of the least stable one -- an 80-fold
spread in how much of its own product growth takes from a gene. Dilution therefore
REDISTRIBUTES capacity toward short-lived transcripts as growth rises. Genes differ in how much of their
own product growth takes away from them. That is a real, gene-specific, testable consequence of closing
the loop, and gate 3 exists to check whether it survives contact with data.

WHAT IS FITTED AND WHAT IS NOT.  One number is fitted: the global turnover constant kappa, bisected so
that wild-type mu lands on 0.030/h (a ~23 h doubling). It is fitted to ONE growth rate, never to the
labels. Everything else is measured input (RPKM and k_deg per gene, 11 cell lines, RNADecayCafe) or a
declared constant (protein k_dp = ln2/24h, uniform, because no per-protein half-life data is in this
container -- an assumption, and named as one).

PREDECLARED GATES.  Written before the run.

    G1 CLOSURE      the iteration converges to the same fixed point from mu0 = 0 and mu0 = 2 mu*,
                    to 1e-6 relative, in <= 200 damped steps.
                    fails -> the loop does not close and nothing below is meaningful.

    G2 SIGNAL       loop AUC (predicted growth drop vs DepMap-derived essentiality) exceeds the 95th
                    percentile of 200 label shuffles.
                    fails -> the loop predicts nothing; report that.

    G3 THE FEEDBACK EARNS ITS PLACE     loop AUC - frozen-mu AUC >= 0.02, where frozen-mu is the SAME
                    enzyme-constrained model with P evaluated once at mu_WT and never re-solved. This
                    isolates the feedback and nothing else.
                    fails -> HONEST NEGATIVE. The cycle is real arithmetic but changes no conclusion,
                    and the correct description is "enzyme-constrained FBA with extra steps".

    G4 BEATS THE CATALOGUE      loop AUC exceeds (a) plain FBA gene deletion with default bounds -- no
                    protein layer at all -- and (b) the best single-column lookup among mRNA abundance,
                    protein abundance and mRNA stability.
                    fails -> a table would have done as well, which is the thing the audit accused the
                    repo of being.

CONTROLS: 200 label shuffles; plain FBA deletion; frozen-mu; three abundance lookups. Four comparators,
declared here, all run.

WHAT HAPPENED, written after the run against the gates above, unedited.

    G1 closure                    PASS   mu = F(mu) reached 0.03000022/h from mu0 = 0 and from
                                         2*mu_WT, agreeing to 1.7e-08 relative. The cycle is real.
    G2 signal                     PASS   loop AUC 0.6210 against a shuffled null of 0.5001 +/- 0.0118
                                         (95th pct 0.5200) over 1,784 scored genes, 241 essential.
    G3 feedback earns its place   FAIL   loop 0.6210 - frozen-mu 0.6284 = -0.0074. The feedback made
                                         it very slightly WORSE, and the gate was +0.02.
    G4 beats the catalogue        FAIL   plain FBA deletion 0.6491. mRNA abundance ALONE 0.7766.

AND THE FAILURE IS SHARPER THAN "NO EFFECT", which is worth stating precisely because it is the
opposite of the convenient reading.  1,705 of 1,784 knockout scores CHANGED when the feedback was
switched on, and the rank correlation between the two score vectors is only 0.3732. So the feedback is
not a negligible term that washes out -- it is large, it reorders most of the predictions, and it
reorders them in a direction that is not better. Dilution genuinely redistributes capacity, exactly as
the mechanism section predicted; that redistribution simply does not track essentiality.

THE HARDEST NUMBER HERE IS 0.7766.  A single column of cell_complete -- how much mRNA a gene has --
predicts essentiality better than the entire mechanistic stack: stoichiometry, GPRs, flux balance,
enzyme capacity and a solved growth fixed point, together, score 0.6210. That is a 0.16 AUC deficit
against a lookup, and it is the most important result on this page. Building more mechanism on top of
this loop, without first understanding why the lookup wins, would be building on a known deficit.

Two candidate explanations, neither tested here, both cheap to test and neither to be asserted until
they are: (a) expression-essentiality is partly a confound rather than a rival model -- highly expressed
genes are more likely essential for reasons outside metabolism -- so the comparison may be unfair to
FBA in a way that does not help the loop either; (b) the medium is wide open (unconstrained mu = 125/h,
calibrated down to 0.030/h purely by enzyme capacity), which leaves the model with no metabolic slack --
all 2,848 knockouts move mu, which no real cell does. (b) is the more likely defect and is a specific,
falsifiable follow-up: constrain uptake to a physiological medium and re-run these same four gates.

-> outputs/cell_loop.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SCRATCH = Path(os.environ.get("CELL_SCRATCH",
               "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GEM = SCRATCH / "HumanGEM.xml"
GEMGENES = SCRATCH / "HumanGEM_genes.tsv"
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
KDEG = ROOT / "outputs" / "orphan" / "rnadecay" / "AvgKdegs.csv"

SEED = 1904
MU_TARGET = 0.030            # /h, ~23 h doubling
K_DP = np.log(2) / 24.0      # /h, uniform protein degradation -- DECLARED ASSUMPTION
MIN_LINES = 3                # a gene needs >= 3 cell lines before its RPKM/k_deg is used
DAMP = 0.5                   # damping on the fixed-point iteration; F is decreasing in mu
TOL = 1e-6
MAXIT = 200
N_SHUFFLE = 200
GATE_FEEDBACK = 0.02


# ---------------------------------------------------------------------------------------------------
# AUC with correct tie handling. Most knockouts score exactly zero, so ties are the common case and a
# naive sort-based AUC would silently depend on gene order -- the same order-dependence the manifest's
# `prefix` warning exists for.
# ---------------------------------------------------------------------------------------------------
def auc(score, label):
    s = np.asarray(score, float)
    y = np.asarray(label, int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gpr_capacity(node, E, default):
    """Enzyme capacity of a reaction from its gene-protein rule.

    OR is isozymes -- either enzyme can carry the flux, so capacities ADD.
    AND is a complex -- every subunit is needed, so the capacity is the MIN.
    That is the standard reading and it is the only place gene-level protein numbers enter the LP."""
    import ast
    if isinstance(node, ast.Module):
        b = node.body
        if isinstance(b, list):                       # a real ast.Module
            return gpr_capacity(b[0], E, default) if b else None
        return gpr_capacity(b, E, default)            # cobra's GPR subclass: body IS the expression
    if isinstance(node, ast.Expr):
        return gpr_capacity(node.value, E, default)
    if isinstance(node, ast.Name):
        return E.get(node.id, default)
    if isinstance(node, ast.BoolOp):
        vals = [gpr_capacity(v, E, default) for v in node.values]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return float(sum(vals)) if isinstance(node.op, ast.Or) else float(min(vals))
    return None


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("THE LOOP -- transcription -> protein -> flux -> growth -> dilution -> transcription")
    say("=" * 100)
    say("  The audit found the cell model's coupling graph is a pure DAG: zero cycles, so nothing it")
    say("  computes ever feeds back. This closes one cycle and measures whether closing it changes any")
    say("  answer. Gates G1-G4 were written into the docstring before the run.")

    # ---- inputs --------------------------------------------------------------------------------------
    t0 = time.time()
    import cobra
    from cobra.io import read_sbml_model
    say(f"\n  loading Human-GEM ({GEM.stat().st_size/1e6:.0f} MB SBML) ...")
    M = read_sbml_model(str(GEM))
    M.solver = "glpk"
    say(f"    {len(M.reactions)} reactions, {len(M.metabolites)} metabolites, {len(M.genes)} genes "
        f"({time.time()-t0:.0f}s)")
    bio = str(M.objective.expression).split("*")[1].split(" ")[0]
    say(f"    objective: {bio}")

    gm = pd.read_csv(GEMGENES, sep="\t")
    ens2sym = {a: b for a, b in zip(gm["genes"], gm["geneSymbols"]) if isinstance(b, str)}

    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    ppm = D["ppm"]

    k = pd.read_csv(KDEG)
    rp = (k[k.avg_RPKM_total > 0].groupby("feature_ID")
          .agg(rpkm=("avg_RPKM_total", "median"), n=("cell_line", "size")))
    kd = k.groupby("feature_ID").agg(kdeg=("avg_kdeg", "median"), n=("cell_line", "size"))
    rp = rp[rp.n >= MIN_LINES]
    kd = kd[kd.n >= MIN_LINES]
    say(f"\n  RNADecayCafe: {k.feature_ID.nunique()} genes x {k.cell_line.nunique()} cell lines; "
        f"{len(rp)} with RPKM and {len(kd)} with k_deg in >= {MIN_LINES} lines")
    hl = np.log(2) / kd.kdeg
    say(f"    mRNA half-life spans {hl.min():.2f} h to {hl.max():.0f} h -- at mu = {MU_TARGET}/h "
        f"dilution removes {MU_TARGET/(MU_TARGET+kd.kdeg.max()):.1%} of the least stable transcript "
        f"and {MU_TARGET/(MU_TARGET+kd.kdeg.min()):.1%} of the most stable. THIS is what makes the "
        f"feedback non-uniform.")

    # ---- per-model-gene inputs -----------------------------------------------------------------------
    med_r, med_d = float(rp.rpkm.median()), float(kd.kdeg.median())
    KSYN, KDEG_G, measured = {}, {}, set()
    for g in M.genes:
        s = ens2sym.get(g.id)
        r = float(rp.rpkm[s]) if (s in rp.index) else None
        d = float(kd.kdeg[s]) if (s in kd.index) else None
        if r is not None and d is not None:
            measured.add(g.id)
        r = med_r if r is None else r
        d = med_d if d is None else d
        # k_syn is BACK-CALCULATED from the measured steady state at the reference growth rate:
        # RPKM was observed in growing cells, so RPKM = k_syn/(k_deg + mu_ref), not k_syn/k_deg.
        KSYN[g.id] = r * (d + MU_TARGET)
        KDEG_G[g.id] = d
    say(f"    {len(measured)} of {len(M.genes)} model genes carry BOTH measurements; the remaining "
        f"{len(M.genes)-len(measured)} get the median so the model is complete, and are EXCLUDED from "
        f"scoring")

    # ---- the capacity map ----------------------------------------------------------------------------
    gpr = {}
    for r in M.reactions:
        if r.gene_reaction_rule.strip():
            try:
                gpr[r.id] = r.gpr
            except Exception:
                pass
    RX = [M.reactions.get_by_id(rid) for rid in gpr]
    ORIG = {r.id: r.bounds for r in M.reactions}
    say(f"    {len(RX)} of {len(M.reactions)} reactions carry a GPR and are capacity-constrained; the "
        f"rest (transport, exchange, spontaneous) keep their SBML bounds")

    def protein(mu, ko=None):
        """P(mu) for every model gene. Both balances carry mu; the knockout sets k_syn to zero."""
        out = {}
        for gid, ks in KSYN.items():
            if gid == ko:
                out[gid] = 0.0
            else:
                out[gid] = ks / ((KDEG_G[gid] + mu) * (K_DP + mu))
        return out

    def apply_capacity(kappa, E):
        default = float(np.median(list(E.values()))) if E else 0.0
        for r in RX:
            c = gpr_capacity(gpr[r.id], E, default)
            if c is None:
                continue
            b = kappa * c
            lo, hi = ORIG[r.id]
            r.bounds = (max(lo, -b), min(hi, b))

    def restore():
        for r in RX:
            r.bounds = ORIG[r.id]

    def F(mu, kappa, ko=None):
        """One turn of the loop: mu in, mu out."""
        apply_capacity(kappa, protein(mu, ko))
        v = M.slim_optimize()
        return 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))

    def fixed_point(kappa, ko=None, mu0=MU_TARGET, tol=TOL, maxit=MAXIT):
        mu = mu0
        for it in range(maxit):
            nxt = F(mu, kappa, ko)
            new = (1 - DAMP) * mu + DAMP * nxt
            if abs(new - mu) <= tol * max(mu, 1e-9):
                return new, it + 1
            mu = new
        return mu, maxit

    # ---- calibrate kappa on ONE number: wild-type growth ----------------------------------------------
    say(f"\n  CALIBRATION -- one fitted constant, bisected so wild-type mu = {MU_TARGET}/h. Fitted to a "
        f"growth rate, never to the labels.")
    lo, hi = 1e-12, 1.0
    for _ in range(60):
        if fixed_point(hi)[0] < MU_TARGET:
            hi *= 10
        else:
            break
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        if fixed_point(mid)[0] < MU_TARGET:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0001:
            break
    KAPPA = float(np.sqrt(lo * hi))
    MU_WT, its = fixed_point(KAPPA)
    say(f"    kappa = {KAPPA:.4g}   ->   mu_WT = {MU_WT:.6f}/h in {its} damped steps")

    # ---- G1 CLOSURE ----------------------------------------------------------------------------------
    say("\n  G1 CLOSURE -- does mu = F(mu) actually have a fixed point, reached from either side?")
    a, ia = fixed_point(KAPPA, mu0=0.0)
    b, ib = fixed_point(KAPPA, mu0=2 * MU_WT)
    spread = abs(a - b) / max(MU_WT, 1e-12)
    g1 = bool(spread < 1e-4 and ia < MAXIT and ib < MAXIT)
    say(f"    from mu0=0        -> {a:.8f}/h in {ia} steps")
    say(f"    from mu0=2*mu_WT  -> {b:.8f}/h in {ib} steps")
    say(f"    relative spread   -> {spread:.2e}      G1 {'PASS' if g1 else 'FAIL'}")
    if not g1:
        say("    The loop does not close. Nothing downstream is meaningful; stopping here would be the")
        say("    honest result, and the remaining gates are reported only as diagnostics.")

    # ---- the knockouts -------------------------------------------------------------------------------
    say(f"\n  KNOCKOUTS -- {len(M.genes)} single-gene deletions, three ways")
    say("    loop     : k_syn -> 0, then the fixed point in mu is re-solved")
    say("    frozen   : same capacity model, P held at its mu_WT value, one solve  <- isolates feedback")
    say("    plainFBA : no protein layer at all, reactions whose GPR goes false are shut off")

    E_WT = protein(MU_WT)
    gids = [g.id for g in M.genes]

    # frozen-mu: capacities from mu_WT, KO only edits that gene's own reactions
    apply_capacity(KAPPA, E_WT)
    FROZEN_B = {r.id: r.bounds for r in RX}
    default_wt = float(np.median(list(E_WT.values())))
    frozen = {}
    t1 = time.time()
    for n, gid in enumerate(gids):
        g = M.genes.get_by_id(gid)
        touched = [r for r in g.reactions if r.id in gpr]
        E2 = dict(E_WT)
        E2[gid] = 0.0
        for r in touched:
            c = gpr_capacity(gpr[r.id], E2, default_wt)
            if c is None:
                continue
            b = KAPPA * c
            lo0, hi0 = ORIG[r.id]
            r.bounds = (max(lo0, -b), min(hi0, b))
        v = M.slim_optimize()
        frozen[gid] = 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))
        for r in touched:
            r.bounds = FROZEN_B[r.id]
        if n % 500 == 0:
            say(f"      frozen {n}/{len(gids)}  ({time.time()-t1:.0f}s)")
    say(f"    frozen pass done in {time.time()-t1:.0f}s")

    # loop: only genes that moved mu need the fixed point re-solved. If the frozen solve returns mu_WT
    # exactly then the capacities are unchanged, so mu_WT is already a fixed point of the knocked-out
    # map -- this is an identity, not an approximation, and it is what makes the full pass affordable.
    movers = [g for g in gids if abs(frozen[g] - MU_WT) > 1e-9]
    say(f"    {len(movers)} of {len(gids)} knockouts move mu at all; the other "
        f"{len(gids)-len(movers)} are exact fixed points already and need no iteration")
    loop = dict(frozen)
    t2 = time.time()
    for n, gid in enumerate(movers):
        loop[gid] = fixed_point(KAPPA, ko=gid, mu0=MU_WT)[0]
        if n % 200 == 0:
            say(f"      loop {n}/{len(movers)}  ({time.time()-t2:.0f}s)")
    restore()
    say(f"    loop pass done in {time.time()-t2:.0f}s")

    # plain FBA deletion -- no protein layer whatsoever
    plain = {}
    mu_open = M.slim_optimize()
    t3 = time.time()
    for n, gid in enumerate(gids):
        g = M.genes.get_by_id(gid)
        touched = []
        for r in g.reactions:
            if r.id not in gpr:
                continue
            E0 = {x.id: 1.0 for x in r.genes}
            E0[gid] = 0.0
            if (gpr_capacity(gpr[r.id], E0, 1.0) or 0.0) <= 0.0:
                touched.append(r)
                r.bounds = (0.0, 0.0)
        v = M.slim_optimize()
        plain[gid] = 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))
        for r in touched:
            r.bounds = ORIG[r.id]
        if n % 1000 == 0:
            say(f"      plainFBA {n}/{len(gids)}  ({time.time()-t3:.0f}s)")
    say(f"    plainFBA pass done in {time.time()-t3:.0f}s (unconstrained mu = {mu_open:.3g}/h)")

    # ---- score ---------------------------------------------------------------------------------------
    rows = []
    for gid in gids:
        s = ens2sym.get(gid)
        i = name2i.get(s)
        if gid not in measured or i is None or genes[i].get("ess") is None:
            continue
        rows.append({"gene": gid, "sym": s, "ess": int(genes[i]["ess"]),
                     "dep_frac": genes[i].get("dep_frac"),
                     "loop": 1.0 - loop[gid] / MU_WT,
                     "frozen": 1.0 - frozen[gid] / MU_WT,
                     "plain": 1.0 - plain[gid] / max(mu_open, 1e-12),
                     "rpkm": float(rp.rpkm[s]), "kdeg": float(kd.kdeg[s]),
                     "ppm": float(ppm.get(str(i), np.nan))})
    T = pd.DataFrame(rows)
    y = T.ess.to_numpy()
    say(f"\n  SCORING SET: {len(T)} genes with a measurement, a model gene and a label -- "
        f"{int(y.sum())} essential, {len(y)-int(y.sum())} not")

    A = {"loop": auc(T.loop, y), "frozen": auc(T.frozen, y), "plainFBA": auc(T.plain, y),
         "mRNA abundance": auc(T.rpkm.fillna(0), y),
         "protein abundance": auc(T.ppm.fillna(0), y),
         "mRNA stability": auc(-T.kdeg, y)}
    say(f"\n  {'predictor':<22}{'AUC':>8}   what it is")
    what = {"loop": "the closed loop -- mu re-solved per knockout",
            "frozen": "same model, feedback switched off",
            "plainFBA": "no protein layer at all",
            "mRNA abundance": "a column of cell_complete",
            "protein abundance": "a column of cell_complete",
            "mRNA stability": "a column of the decay table"}
    for kk, vv in sorted(A.items(), key=lambda x: -x[1]):
        say(f"  {kk:<22}{vv:>8.4f}   {what[kk]}")

    # ---- G2 permutation ------------------------------------------------------------------------------
    null = np.array([auc(T.loop, rng.permutation(y)) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(null, 95))
    g2 = bool(A["loop"] > p95)
    say(f"\n  G2 SIGNAL -- {N_SHUFFLE} label shuffles: null AUC {null.mean():.4f} +/- {null.std():.4f}, "
        f"95th pct {p95:.4f}")
    say(f"    loop {A['loop']:.4f}      G2 {'PASS' if g2 else 'FAIL'}")

    # ---- G3 the feedback -----------------------------------------------------------------------------
    d = A["loop"] - A["frozen"]
    g3 = bool(d >= GATE_FEEDBACK)
    diff = int((np.abs(T.loop - T.frozen) > 1e-9).sum())
    rho = float(pd.Series(T.loop).corr(pd.Series(T.frozen), method="spearman"))
    say(f"\n  G3 DOES THE FEEDBACK EARN ITS PLACE -- loop minus frozen-mu")
    say(f"    {diff} of {len(T)} scored knockouts differ at all between the two; Spearman {rho:.4f}")
    say(f"    AUC {A['loop']:.4f} - {A['frozen']:.4f} = {d:+.4f}   "
        f"(gate {GATE_FEEDBACK})   G3 {'PASS' if g3 else 'FAIL'}")
    if not g3:
        say("    NEGATIVE, and it is the predeclared one. The cycle closes and is real arithmetic, but")
        say("    re-solving mu per knockout changes no conclusion at this resolution. The honest name")
        say("    for what was built is enzyme-constrained FBA; the feedback is a term, not a finding.")

    # ---- G4 beats the catalogue ----------------------------------------------------------------------
    lookups = {kk: A[kk] for kk in ("mRNA abundance", "protein abundance", "mRNA stability")}
    best_lookup = max(lookups, key=lookups.get)
    g4 = bool(A["loop"] > A["plainFBA"] and A["loop"] > lookups[best_lookup])
    say(f"\n  G4 BEATS THE CATALOGUE")
    say(f"    vs plain FBA deletion : {A['loop']:.4f} vs {A['plainFBA']:.4f}  "
        f"({A['loop']-A['plainFBA']:+.4f})")
    say(f"    vs best lookup ({best_lookup}) : {A['loop']:.4f} vs {lookups[best_lookup]:.4f}  "
        f"({A['loop']-lookups[best_lookup]:+.4f})")
    say(f"    G4 {'PASS' if g4 else 'FAIL'}")

    # ---- verdict -------------------------------------------------------------------------------------
    say("\n" + "=" * 100)
    gates = {"G1 closure": g1, "G2 signal": g2, "G3 feedback earns its place": g3,
             "G4 beats the catalogue": g4}
    for kk, vv in gates.items():
        say(f"  {kk:<34}{'PASS' if vv else 'FAIL'}")
    if g1 and g2 and g3 and g4:
        say("  A CYCLE THAT PAYS. The cell model now contains one closed loop that predicts better than")
        say("  the same machinery without it, better than FBA without the protein layer, and better")
        say("  than any single column of the catalogue.")
    elif g1 and g2:
        say("  THE LOOP CLOSES AND PREDICTS, BUT THE FEEDBACK IS NOT WHAT IS DOING THE PREDICTING.")
        say("  Report it as enzyme-constrained FBA. The DAG in cell_complete now has one cycle in it,")
        say("  which was the structural point, but the cycle has not yet bought an answer.")
    elif g1:
        say("  THE LOOP CLOSES AND PREDICTS NOTHING. Recorded as a negative.")
    else:
        say("  THE LOOP DOES NOT CLOSE. Recorded as a negative.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(GEM), str(GEMGENES), str(KDEG), str(CELL)],
                      available=len(M.genes), used=len(T), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "frozen-mu (feedback off)",
                                "plain FBA deletion (no protein layer)",
                                "mRNA / protein / stability lookups",
                                "two-sided fixed-point convergence"],
                      note="scored genes = model genes with RPKM and k_deg in >=3 cell lines AND an "
                           "essentiality label; kappa fitted to wild-type growth rate only")
    RM.report(man, emit=say)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "cell_loop", "manifest": man, "gates": gates, "auc": A,
               "kappa": KAPPA, "mu_wt": MU_WT, "mu_unconstrained": float(mu_open),
               "mu_target": MU_TARGET, "k_dp": K_DP,
               "closure": {"from_zero": a, "from_double": b, "spread": spread,
                           "iters": [ia, ib]},
               "null": {"mean": float(null.mean()), "sd": float(null.std()), "p95": p95},
               "feedback": {"delta_auc": d, "n_differing": diff, "spearman": rho},
               "n_model_genes": len(M.genes), "n_measured": len(measured), "n_scored": len(T),
               "n_essential": int(y.sum()), "n_movers": len(movers),
               "scores": T.to_dict("records"), "log": log},
              open(OUT / "cell_loop.json", "w"), indent=2)
    say(f"\n  -> {OUT/'cell_loop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
