"""V4 BLOCK 7A -- is HumanGEM a CONSTRAINT LAYER, or just an enzyme list? Sv = 0 vs abundance alone.

THE NARROW CLAIM UNDER TEST, AND NOTHING WIDER. The project's stated reason for carrying a 42 MB
genome-scale reconstruction is that it supplies a CONSTRAINT: the steady-state mass balance Sv = 0 turns a
bag of enzymes into a network in which knocking out one reaction propagates. If that is true, an actual
Sv = 0 solve must predict which metabolic genes are essential BETTER than the enzyme list alone does. This
module tests exactly that sentence and refuses to test anything larger. It is not a test of whether FBA is
good biology, not a test of HumanGEM's curation, and not a claim about non-metabolic genes.

THE ENDPOINT. Gene-level essentiality for METABOLIC genes, taken from DepMap `CRISPRGeneEffect.csv`
(Chronos gene effect, cell lines x genes). Truth for a gene is its MEAN gene effect across every cell line
in which it was measured, and the primary label is `mean effect <= -0.5`, DepMap's conventional
essential/not boundary (0 = no effect, -1 = the median common-essential). The universe is every HumanGEM
gene that has a DepMap column and enough measured cell lines; EVERY arm is scored on that identical
universe and on identical held-out gene blocks. An arm that ranked by a quantity the truth excludes would
score ~0 by construction -- that defect has shipped three times in this project, so the universe is
asserted equal across arms at run time rather than assumed.

MISSING IS NOT ZERO. DepMap gene effect is ~1.5% NaN. NaNs are dropped from the per-gene mean, never read
as 0.0, and a gene needs MIN_LINES measured lines to enter the universe at all. (The `nlz_*` benches are
not used anywhere here; they are 3.5% dense and their zeros mean NOT RECORDED.)

THE ARMS. Every arm is a score over the SAME gene universe. None of them is trained; the only quantity fit
from data is a single SIGN per arm per fold, fit on the TRAINING genes of that fold and applied to the
held-out block, so that no arm is penalised for a hard-coded direction convention.

    0   reaction count                  |R(g)|, the number of HumanGEM reactions the gene's GPR can carry.
                                        THE CHEAPEST SENSIBLE NON-LEARNED BASELINE: no kcat, no
                                        stoichiometry, no solver, one integer per gene. Reported next to
                                        every other arm, not in a footnote. If it wins, that IS the result.
    0N  uniform random score            the null floor. Establishes that the metric itself is at 0.5.
    1   ABUNDANCE-ONLY                  predicted enzyme demand per unit flux, summed over the gene's
                                        reactions and discounted by isozyme redundancy:
                                        d(g) = sum_r 1 / (kcat_r * n_isozymes_r). kcat comes from
                                        `kcat_dl.json` by EC number (human medians first, all-organism
                                        medians next, 3-level EC prefix next, global median last; the
                                        share resolved at each tier is reported). NO STOICHIOMETRY IS
                                        USED. This is the thing Sv = 0 has to beat.
    2   REACTION-NEIGHBOURHOOD          d(g) + mean d over co-reaction enzymes + mean d over
                                        metabolite-adjacent enzymes. Uses the SPARSITY PATTERN of S and
                                        nothing else -- which enzymes touch which metabolites, never by
                                        how much. This is connectivity without stoichiometry and it is
                                        deliberately the strongest stoichiometry-free opponent available.
    3   FBA-CONSTRAINED                 a real Sv = 0 solve. Maximise the HumanGEM biomass reaction subject
                                        to Sv = 0 and the model's own bounds; then for each gene delete
                                        the reactions its GPR can no longer support and re-solve.
                                        score = 1 - growth_KO / growth_WT. Genes whose GPR leaves every
                                        reaction supported by an isozyme score exactly 0 with no LP, which
                                        is a prediction, not a missing value.
    4   SHUFFLED-STOICHIOMETRY CONTROL  THE ARM THIS EXPERIMENT IS BUILT AROUND. Identical sparsity
                                        pattern, identical per-reaction coefficient multiset, entries
                                        PERMUTED WITHIN each reaction column; same GPR, same bounds, same
                                        solver, same knockout sets. If arm 4 reproduces arm 3, then what
                                        arm 3 knows is which enzymes touch which metabolites --
                                        connectivity -- and the coefficients are decoration. BE WARNED
                                        THAT THIS CONTROL MAY NOT BE CONSTRUCTIBLE: permuting entries
                                        moves SIGNS, so substrates become products, and on a genome-scale
                                        model that can destroy the directed network so completely that
                                        nothing grows. See the DEGENERACY rule below.
    4b  MAGNITUDE-SHUFFLE CONTROL       the sharper and better-posed variant: the SIGN at every nonzero
                                        position is preserved -- substrates stay substrates, products stay
                                        products, the directed network is preserved EXACTLY -- and only
                                        the magnitudes are permuted within the column. This isolates
                                        mass-balance RATIOS from connectivity, which is the actual
                                        question, and it cannot break the network the way arm 4 can.
    5   DEGREE-MATCHED REWIRING         configuration model on the bipartite metabolite x reaction
                                        incidence: metabolite stubs permuted, so every reaction keeps its
                                        exact degree and coefficient multiset and every metabolite keeps
                                        its exact degree. Only the wiring moves. Uniform rewiring would
                                        flatten the hub structure too and could not attribute a drop to
                                        wiring; this can.
    2R  RANDOM-PARTNER CONTROL          arm 2 with both neighbour tiers replaced by uniformly drawn
                                        metabolic genes, matched exactly on COUNT and on TIER MIX.
    2S  NEIGHBOURHOOD SWAP              COUNTERFACTUAL. Arm 2's own-demand term is kept correct and the two
                                        neighbour aggregates are taken from ANOTHER gene. Accuracy alone
                                        cannot distinguish "the neighbourhood channel is useless" from
                                        "the neighbourhood channel is IGNORED" -- both look like
                                        arm 2 == arm 1. Only a swap can.
    I   WRONG-GENE IDENTITY CONTROL     arm 3's scores permuted across genes. Bounds how much of any
                                        AUROC is the metric's own degeneracy rather than gene identity.

THE BIOMASS-DEFINITION CLOSURE IS EXCLUDED FROM ARMS 4, 4b AND 5, BY A STRUCTURAL RULE DECLARED HERE. The
closure is the objective column plus, transitively, any reaction that is the SOLE producer of a metabolite
the closure consumes -- on HumanGEM that is the explicit "protein pool", "lipid pool", "small metabolite
pool" and "cofactor pool" pseudo-reactions and a short tail behind them. Those reactions ARE the definition
of growth. Permuting their coefficients changes what the endpoint MEANS rather than changing the network,
and a control that did that would not be a control. The realised closure is listed in the JSON.

THE DIRECTION FIT, AND THE BIAS IT BUYS. Arms 0-2 have no self-evident sign -- it is not obvious a priori
whether high predicted enzyme demand should mark a gene as more or less essential -- and hard-coding a
convention would cripple whichever baseline guessed wrong, which is exactly the defect this project keeps
shipping. So exactly ONE parameter is fit from the endpoint anywhere in this module: a per-arm, per-fold
SIGN, fit on the TRAINING genes of that fold against the continuous mean gene effect and applied unchanged
to the held-out block. This is not free. On an uninformative arm the sign is chosen on noise, which biases
its held-out AUROC UPWARD. That is why arm 0N exists: the uniform-random arm measures the bias directly and
IS THE FLOOR EVERY OTHER NUMBER MUST BE READ AGAINST, not 0.5. The unfitted a-priori-direction AUROC is
reported next to every arm so both readings are visible.

THE UNIT OF REPLICATION IS A DISJOINT METABOLIC-GENE PARTITION. The claim is "Sv = 0 ranks metabolic genes
by essentiality better than abundance does", so the error bar must be computed across UNSEEN GENE SETS, not
across solver restarts, cell lines, or random seeds of a single arm. The universe is split into FOLDS
disjoint blocks, stratified on the essentiality label so no block is degenerate; each block is held out
once; an arm's replicates are its FOLDS held-out AUROCs.

    MDE = 3 * sd / sqrt(n),  n = number of DISJOINT HELD-OUT GENE BLOCKS (default 5),
    sd = the across-block sd of an arm's held-out AUROC, pooled over arms (ddof=1).

A per-CELL-LINE spread is also computed and written to the JSON. It is REPORTED AND NOT GRADED, for the
same reason model-init seeds are not graded elsewhere in this project: cell lines are strongly correlated
and none of them is an unseen gene, so an n of ~1,150 there would manufacture significance for a claim
about genes. Nothing in the gates reads it.

THE GATE, DECLARED BEFORE ANY NUMBER IS PRODUCED.

    PRIMARY   The constraint layer is credited only if arm 3 beats BOTH arm 2 (connectivity, no
              stoichiometry) AND arm 4 (shuffled stoichiometry) by more than the cross-block MDE on AUROC.
              Beating only arm 2 means the coefficients are decoration. Beating only arm 4 means the
              network is doing the work and the solve is unnecessary.
    DEGENERACY If a null variant cannot grow at all (WT objective ~ 0) then every one of its knockout
              scores is identically 0, its AUROC is 0.5 BY CONSTRUCTION, and it is a BROKEN MODEL rather
              than a null network. Two rules, both declared in advance, both of which can only make the
              gate harder:
              (a) the PRIMARY shuffle control is the FIRST of [arm 4, arm 4b] that grows at full strength.
                  A degenerate variant is marked DEGENERATE and cannot carry a gate. If NEITHER grows,
                  PRIMARY is UNRESOLVED -- not a pass, and not evidence of absence.
              (b) arm 5, if it cannot grow at full strength, is retried at the strongest perturbation that
                  still grows, found by bisection on GROWTH ALONE. The endpoint is never consulted, so this
                  cannot tune toward a result. A partial null leaves most of the real network intact, so it
                  is a WEAKER null; the realised strength is reported and the WIRING gate is marked
                  UNRESOLVED whenever it is below 1.
    SHARP     Stoichiometric COEFFICIENTS are credited only if arm 3 beats arm 4b by more than the MDE. If
              arm 3 beats arm 4 but not arm 4b, the signal is the substrate/product sign pattern -- the
              directed network -- and not mass-balance ratios.
    WIRING    The specific wiring is credited only if arm 3 beats arm 5 by more than the MDE.
    ABUNDANCE The headline sentence "Sv = 0 beats predicted enzyme abundance" holds only if arm 3 beats
              arm 1 by more than the MDE.
    FREE      Any of this is worth the 42 MB reconstruction and the LP solver only if arm 3 also beats
              arm 0 -- a bare count of reactions per gene -- by more than the MDE. On the block already
              run in this project a zero-parameter untrained baseline beat the trained transformer, so a
              trivial baseline matching the model is the headline here too, not a caveat.
    AGREE     The gate is reported as sound only if the threshold-free Spearman correlation against the
              continuous mean gene effect reaches the SAME verdict as AUROC on the PRIMARY gaps.
              Disagreement means metric artifact and the result is not reportable as biology.

EFFECT SIZES ARE RAW HELD-OUT VALUES. Every number reported is an arm's own held-out AUROC or Spearman on
the metabolic-gene universe. The shuffled, rewired, random-partner, swap and wrong-gene arms establish
significance and attribution only; no arm is ever reported net of a control. Where a control reproduces
most of an arm's advantage, that is stated as the finding.

SOLVER. cobrapy is used to parse the SBML, to build S, and to evaluate the gene-protein-reaction rules; the
LPs themselves are solved with `scipy.optimize.linprog(method="highs")` so that the real, shuffled,
magnitude-shuffled and rewired S matrices all pass through one identical solver path with no cobra-side
model rebuilding between them. The wild-type solve is cross-checked against cobrapy's own optimiser and the
agreement is asserted and reported. Which solver produced which number is recorded in the JSON. Every
reported FBA knockout is a FULL solve of the complete model -- no reduced model, no flux variability
approximation, and no subsampling of genes in a full run.

NO CIRCULARITY, CHECKED. HumanGEM is a literature-curated reconstruction, its EC annotations come from the
model, and kcat values come from a BRENDA/SABIO-derived deposit. The scored endpoint is DepMap CRISPR gene
effect. No feature is derived from DepMap and no DepMap quantity enters any arm's score. The only quantity
fit from the endpoint is one sign per arm per fold, fit on training genes and applied to held-out genes.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
SBML = Path(os.environ.get("MC_SBML", str(SP / "HumanGEM.xml")))
KCAT = Path(os.environ.get("MC_KCAT", str(SP / "kcat_dl.json")))
DEPMAP = Path(os.environ.get("MC_DEPMAP", str(SP / "CRISPRGeneEffect.csv")))
GENEMAP = Path(os.environ.get("MC_GENEMAP", str(ROOT / "HumanGEM_genes.tsv")))

# --- endpoint conventions -------------------------------------------------------------------------
ESS_THRESH = float(os.environ.get("MC_ESS_THRESH", "-0.5"))   # DepMap convention: -1 = median common-essential
MIN_LINES = int(os.environ.get("MC_MIN_LINES", "100"))        # measured cell lines a gene needs to be scorable
# --- structure conventions ------------------------------------------------------------------------
CURRENCY_Q = float(os.environ.get("MC_CURRENCY_Q", "0.95"))   # top 5% of metabolites by reaction degree are
                                                              # currency and are excluded from arm 2's tier 2.
                                                              # Same 5% convention dense_stage1 uses for the tide.
# --- this module's own knobs ----------------------------------------------------------------------
FOLDS = int(os.environ.get("MC_FOLDS", "5"))      # DISJOINT held-out GENE blocks == the unit of replication
SEED = int(os.environ.get("MC_SEED", "0"))
SMOKE = os.environ.get("MC_SMOKE", "0") == "1"
SMOKE_N = int(os.environ.get("MC_SMOKE_N", "96"))
if SMOKE:
    # A SMOKE RUN IS A PLUMBING CHECK AND NOT A RESULT. SMOKE_N genes give a handful of essential genes per
    # held-out block, every AUROC is dominated by which two or three genes landed where, and no arm can be
    # separated from any other. The subsample is announced in the verdict string and written to the JSON.
    FOLDS = 3
LP_METHOD = os.environ.get("MC_LP_METHOD", "highs")


def auroc(score, y):
    """Rank AUROC with midranks, so the enormous tie block that FBA produces is handled correctly.

    FBA hands most genes exactly 0.0 (their GPR leaves every reaction covered by an isozyme). Those are
    genuine ties -- the arm asserts no order among them -- and midranks score them at chance, which is
    the honest treatment. A tie-breaking rule would invent information the arm does not have.
    """
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(score, float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    import time
    import collections
    import pandas as pd
    import scipy.sparse as sp
    from scipy.optimize import linprog
    from scipy.stats import spearmanr
    try:
        import torch
        torch.set_num_threads(1)      # another job owns most of this box's 4 cores
    except Exception:
        pass
    t0 = time.time()
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 108)
    report("V4 BLOCK 7A -- HumanGEM as a CONSTRAINT LAYER vs abundance alone (Sv = 0 vs an enzyme list)")
    report("=" * 108)
    report("  GATE, declared before the run: the constraint layer is credited only if the FBA arm beats BOTH")
    report("  the stoichiometry-free reaction-neighbourhood arm AND the shuffled-stoichiometry control by")
    report(f"  more than MDE = 3*sd/sqrt(n), n = {FOLDS} DISJOINT HELD-OUT GENE BLOCKS. A shuffled model that")
    report("  cannot grow is DEGENERATE, cannot satisfy the gate, and forces the gate onto arm 4b instead.")
    if SMOKE:
        report(f"  *** SMOKE MODE: universe subsampled to {SMOKE_N} genes. PLUMBING CHECK, NOT A RESULT. ***")

    # ============================================================ 1. the reconstruction (cobrapy parse)
    solver_note = {}
    try:
        import cobra
        solver_note["parser"] = f"cobrapy {cobra.__version__}"
        HAVE_COBRA = True
    except Exception as e:                                     # pragma: no cover - environment dependent
        HAVE_COBRA = False
        solver_note["parser"] = f"cobrapy UNAVAILABLE ({e}); SBML parsed with libsbml fallback"
        raise SystemExit("cobrapy is required to parse the GPR rules; no fallback is implemented for GPRs")
    solver_note["lp"] = f"scipy.optimize.linprog(method='{LP_METHOD}') -- HiGHS"

    t = time.time()
    model = cobra.io.read_sbml_model(str(SBML))
    report(f"\n  HumanGEM parsed with {solver_note['parser']} in {time.time()-t:.1f}s: "
           f"{len(model.reactions):,} reactions x {len(model.metabolites):,} metabolites, "
           f"{len(model.genes):,} genes")

    rxns = list(model.reactions)
    mets = list(model.metabolites)
    ridx = {r.id: j for j, r in enumerate(rxns)}
    midx = {m_.id: i for i, m_ in enumerate(mets)}
    rows, cols, vals = [], [], []
    for j, r in enumerate(rxns):
        for m_, c_ in r.metabolites.items():
            rows.append(midx[m_.id])
            cols.append(j)
            vals.append(float(c_))
    S = sp.csc_matrix((np.asarray(vals, float), (np.asarray(rows), np.asarray(cols))),
                      shape=(len(mets), len(rxns)))
    LB = np.array([r.lower_bound for r in rxns], float)
    UB = np.array([r.upper_bound for r in rxns], float)
    OBJ = np.zeros(len(rxns))
    for r in rxns:
        if r.objective_coefficient:
            OBJ[ridx[r.id]] = -float(r.objective_coefficient)     # linprog minimises
    obj_cols = np.where(OBJ != 0)[0]
    assert obj_cols.size == 1, "expected a single biomass objective reaction"
    OBJ_COL = int(obj_cols[0])
    assert np.isfinite(LB).all() and np.isfinite(UB).all(), \
        "infinite bounds would let a shuffled S produce an UNBOUNDED LP and a meaningless control"
    report(f"  S: {S.shape[0]:,} x {S.shape[1]:,}, {S.nnz:,} nonzeros ({100*S.nnz/(S.shape[0]*S.shape[1]):.3f}% "
           f"dense); objective = {rxns[OBJ_COL].id}; all bounds finite so no shuffle can make the LP unbounded")

    BEQ = np.zeros(S.shape[0])
    lp_status = collections.Counter()

    def solve(Smat, lb, ub):
        """v = 0 is always feasible (0 lies inside every bound) and all bounds are finite, so a
        non-optimal status is a SOLVER FAILURE, not a zero-growth model. The two are counted separately
        rather than both collapsing to 0.0, which would silently turn numerical trouble into biology."""
        r = linprog(OBJ, A_eq=Smat, b_eq=BEQ, bounds=np.column_stack([lb, ub]), method=LP_METHOD)
        lp_status[int(r.status)] += 1
        return float(-r.fun) if (r.status == 0 and r.fun is not None) else float("nan")

    # THE BIOMASS-DEFINITION CLOSURE. Declared structurally, before any number: the objective column,
    # plus transitively any reaction that is the SOLE producer of a metabolite the closure consumes.
    # These reactions ARE the definition of growth (HumanGEM builds biomass from explicit "protein pool",
    # "lipid pool", "small metabolite pool" and "cofactor pool" pseudo-reactions). Permuting their
    # coefficients changes what growth MEANS rather than changing the network, so a control that did it
    # would not be a control. They are excluded from arms 4, 4b and 5 exactly as the objective column is.
    Scsr0 = S.tocsr()
    producers = [Scsr0.indices[Scsr0.indptr[i]:Scsr0.indptr[i + 1]]
                 [Scsr0.data[Scsr0.indptr[i]:Scsr0.indptr[i + 1]] > 0] for i in range(S.shape[0])]
    CLOSURE, frontier = {OBJ_COL}, [OBJ_COL]
    while frontier:
        nxt = []
        for j in frontier:
            sl = slice(S.indptr[j], S.indptr[j + 1])
            for i, v in zip(S.indices[sl], S.data[sl]):
                if v < 0 and len(producers[i]) == 1 and int(producers[i][0]) not in CLOSURE:
                    CLOSURE.add(int(producers[i][0]))
                    nxt.append(int(producers[i][0]))
        frontier = nxt
    report(f"  biomass-definition closure held FIXED in every null variant: {len(CLOSURE)} reactions "
           f"({100*len(CLOSURE)/S.shape[1]:.2f}% of the model), e.g. "
           + ", ".join(rxns[j].id for j in sorted(CLOSURE)[:4]) +
           " -- permuting these would change what growth MEANS, not the network")

    wt_lp = solve(S, LB, UB)
    t = time.time()
    wt_cobra = float(model.slim_optimize())
    report(f"  WT growth: linprog/HiGHS {wt_lp:.6f}   cobrapy {wt_cobra:.6f} "
           f"(agree to {abs(wt_lp-wt_cobra):.2e}; cobra took {time.time()-t:.1f}s)")
    assert abs(wt_lp - wt_cobra) <= 1e-6 * max(1.0, abs(wt_cobra)), \
        "the LP path used for every arm does not reproduce cobrapy's own solution"
    solver_note["wt_growth_linprog"] = wt_lp
    solver_note["wt_growth_cobrapy"] = wt_cobra

    # GPR knockout sets. Computed ONCE from cobrapy and shared by every S variant, so the ONLY thing that
    # differs between arms 3, 4, 4b and 5 is the stoichiometric matrix.
    ko_rxn = {}
    solo_kill = np.zeros(len(rxns), int)      # genes whose SOLO deletion inactivates this reaction
    n_gpr_gene = np.array([len(r.genes) for r in rxns], int)
    for g in model.genes:
        dead = [ridx[r.id] for r in g.reactions if not r.gpr.eval({g.id})]
        if dead:
            ko_rxn[g.id] = np.asarray(sorted(dead))
            solo_kill[ko_rxn[g.id]] += 1
    # n_iso = the number of genes that can EACH ALONE sustain the reaction, read straight off the GPR:
    # len(genes) - (genes whose solo deletion kills it). A 3-way OR of isozymes gives 3; a single-gene
    # reaction gives 0 -> clamped to 1; an AND complex gives 0 -> clamped to 1, which is correct, because
    # a complex is NOT redundant. Using len(r.genes) directly would have handed every multi-subunit
    # complex a redundancy discount and quietly crippled the abundance baseline.
    n_iso = np.maximum(1, n_gpr_gene - solo_kill)
    report(f"  GPR: {len(ko_rxn):,}/{len(model.genes):,} genes inactivate >= 1 reaction when deleted alone; "
           f"the rest are fully isozyme-covered and score exactly 0 in every FBA arm WITHOUT an LP "
           f"(a prediction of non-essentiality, not a missing value)")

    # ============================================================ 2. the endpoint (DepMap, NaN-aware)
    gm = pd.read_csv(GENEMAP, sep="\t", dtype=str)
    ensg2sym = dict(zip(gm["genes"], gm["geneSymbols"].fillna("")))
    ensg2ent = dict(zip(gm["genes"], gm["geneEntrezID"].fillna("")))
    hdr = pd.read_csv(DEPMAP, nrows=0)
    dcols = list(hdr.columns)
    by_ent, by_sym = {}, {}
    for c in dcols[1:]:
        if " (" in c and c.endswith(")"):
            s_, e_ = c.rsplit(" (", 1)
            by_ent[e_[:-1]] = c
            by_sym[s_] = c
    gene_col, matched_by, used = {}, collections.Counter(), set()
    for g in model.genes:
        c = by_ent.get(ensg2ent.get(g.id, ""))
        how = "entrez"
        if c is None:
            c = by_sym.get(ensg2sym.get(g.id, ""))
            how = "symbol"
        if c is not None and c not in used:
            gene_col[g.id] = c
            used.add(c)
            matched_by[how] += 1
    t = time.time()
    dfe = pd.read_csv(DEPMAP, usecols=[dcols[0]] + sorted(set(gene_col.values())))
    lines = dfe.iloc[:, 0].astype(str).to_numpy()
    Ecol = {c: k for k, c in enumerate(dfe.columns[1:])}
    E = dfe.iloc[:, 1:].to_numpy(float)
    report(f"\n  DepMap CRISPRGeneEffect: {E.shape[0]:,} cell lines x {E.shape[1]:,} matched metabolic gene "
           f"columns, loaded in {time.time()-t:.1f}s ({matched_by['entrez']:,} matched on Entrez id, "
           f"{matched_by['symbol']:,} on symbol)")
    nan_frac = float(np.isnan(E).mean())
    report(f"  {100*nan_frac:.2f}% of gene-effect entries are NaN. THEY ARE DROPPED FROM THE PER-GENE MEAN, "
           f"NEVER READ AS 0.0, and a gene needs >= {MIN_LINES} measured lines to enter the universe")

    genes = sorted(gene_col)
    kcols = np.array([Ecol[gene_col[g]] for g in genes])
    Eg = E[:, kcols]
    n_meas = np.isfinite(Eg).sum(0)
    mu_eff = np.where(n_meas > 0, np.nanmean(np.where(np.isfinite(Eg), Eg, np.nan), axis=0), np.nan)
    ok = (n_meas >= MIN_LINES) & np.isfinite(mu_eff)
    genes = [g for g, o in zip(genes, ok) if o]
    Eg = Eg[:, ok]
    mu_eff = mu_eff[ok]
    y = (mu_eff <= ESS_THRESH).astype(int)
    report(f"  metabolic-gene universe: {len(genes):,} genes with a HumanGEM GPR and >= {MIN_LINES} measured "
           f"lines; {int(y.sum()):,} essential at mean effect <= {ESS_THRESH} ({100*y.mean():.1f}% prevalence)")

    rng = np.random.default_rng(SEED)
    if SMOKE:
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        npos = max(FOLDS * 2, int(round(SMOKE_N * y.mean())))
        take = np.concatenate([rng.choice(pos, min(npos, len(pos)), replace=False),
                               rng.choice(neg, min(SMOKE_N - npos, len(neg)), replace=False)])
        take = np.sort(take)
        genes = [genes[i] for i in take]
        Eg = Eg[:, take]
        mu_eff = mu_eff[take]
        y = y[take]
        report(f"  *** SMOKE SUBSAMPLE: {len(genes)} genes, {int(y.sum())} essential. LOUD, NOT SILENT. "
               f"A full run never subsamples. ***")
    NG = len(genes)
    gpos = {g: i for i, g in enumerate(genes)}

    # ============================================================ 3. kcat -> predicted enzyme demand
    t = time.time()
    kd = json.load(open(KCAT))
    by_ec, by_ec_h = collections.defaultdict(list), collections.defaultdict(list)
    for x in kd:
        try:
            v = float(x["Value"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        by_ec[x["ECNumber"]].append(v)
        if x.get("Organism") == "Homo sapiens":
            by_ec_h[x["ECNumber"]].append(v)
    med_ec = {k: float(np.median(v)) for k, v in by_ec.items()}
    med_ec_h = {k: float(np.median(v)) for k, v in by_ec_h.items()}
    pre3 = collections.defaultdict(list)
    for k, v in med_ec.items():
        pre3[k.rsplit(".", 1)[0]].append(v)
    med_pre3 = {k: float(np.median(v)) for k, v in pre3.items()}
    GLOBAL_KCAT = float(np.median(list(med_ec.values())))
    tier = collections.Counter()

    def kcat_of(r):
        ec = r.annotation.get("ec-code")
        ec = [] if ec is None else ([ec] if isinstance(ec, str) else list(ec))
        for name, tbl, key in (("human_EC", med_ec_h, lambda e: e), ("any_EC", med_ec, lambda e: e),
                               ("EC3", med_pre3, lambda e: e.rsplit(".", 1)[0])):
            v = [tbl[key(e)] for e in ec if key(e) in tbl]
            if v:
                tier[name] += 1
                return float(np.median(v))
        tier["global_median"] += 1
        return GLOBAL_KCAT

    kcat = np.array([kcat_of(r) for r in rxns])
    report(f"\n  kcat from {KCAT.name} ({len(kd):,} records, {len(med_ec):,} EC numbers) in {time.time()-t:.1f}s. "
           f"Reaction coverage by tier: " + ", ".join(f"{k} {v:,}" for k, v in tier.most_common()) +
           f"  (global median {GLOBAL_KCAT:.2f} 1/s); isozyme redundancy from the GPR: "
           f"{int((n_iso > 1).sum()):,} reactions have >= 2 independently-sufficient genes")

    # Demand is defined for EVERY HumanGEM gene, not only the scored ones, so that arm 2's neighbourhood
    # is the real reaction neighbourhood rather than one truncated to the scored universe. Only the
    # SCORED universe is restricted; the neighbours a gene aggregates over are not.
    all_genes = [g.id for g in model.genes]
    apos = {g: i for i, g in enumerate(all_genes)}
    g2r_all = {g: np.asarray(sorted(ridx[r.id] for r in model.genes.get_by_id(g).reactions), int)
               for g in all_genes}
    demand_all = np.array([float(np.sum(1.0 / (kcat[g2r_all[g]] * n_iso[g2r_all[g]])))
                           if g2r_all[g].size else 0.0 for g in all_genes])
    uidx = np.array([apos[g] for g in genes])
    demand = demand_all[uidx]
    rxn_count = np.array([float(g2r_all[g].size) for g in genes])

    # ============================================================ 4. arm 2: connectivity WITHOUT values
    Scsr = S.tocsr()
    mdeg = np.diff(Scsr.indptr)
    cur_cut = float(np.quantile(mdeg, CURRENCY_Q))
    noncur = mdeg < cur_cut
    r2g = [[g.id for g in r.genes] for r in rxns]
    tier1, tier2 = [], []
    for g in genes:
        rs = g2r_all[g]
        n1 = set()
        for j in rs:
            n1.update(r2g[j])
        n1.discard(g)
        ms = set()
        for j in rs:
            ms.update(S.indices[S.indptr[j]:S.indptr[j + 1]].tolist())
        n2 = set()
        for i in ms:
            if not noncur[i]:
                continue
            for j in Scsr.indices[Scsr.indptr[i]:Scsr.indptr[i + 1]]:
                n2.update(r2g[j])
        n2 -= n1
        n2.discard(g)
        tier1.append(np.array([apos[x] for x in n1], int))
        tier2.append(np.array([apos[x] for x in n2], int))
    report(f"  arm 2 neighbourhood (SPARSITY of S only, never its values): currency metabolites = the top "
           f"{100*(1-CURRENCY_Q):.0f}% by reaction degree (degree >= {cur_cut:.0f}, "
           f"{int((~noncur).sum()):,} of {len(mdeg):,}) are excluded from tier 2. "
           f"co-reaction partners median {np.median([len(a) for a in tier1]):.0f}; "
           f"metabolite-adjacent median {np.median([len(a) for a in tier2]):.0f}")

    def agg(idx_lists):
        return np.array([float(demand_all[a].mean()) if a.size else 0.0 for a in idx_lists])

    a_own, a_t1, a_t2 = demand, agg(tier1), agg(tier2)
    NA = len(all_genes)
    rng_r = np.random.default_rng(SEED + 11)
    rnd1 = [rng_r.choice(NA, len(a), replace=True) if len(a) else np.array([], int) for a in tier1]
    rnd2 = [rng_r.choice(NA, len(a), replace=True) if len(a) else np.array([], int) for a in tier2]
    swap = np.random.default_rng(SEED + 12).permutation(NG)

    # ============================================================ 5. the S variants
    ELIGIBLE = np.array([j for j in range(S.shape[1])
                         if (S.indptr[j + 1] - S.indptr[j]) >= 2 and j not in CLOSURE])

    def shuffle_within_column(seed, keep_sign, strength=1.0):
        """Same sparsity, per-column coefficient multiset preserved, entries permuted inside the column.

        keep_sign=False is the LITERAL control: the value at each nonzero position is replaced by another
        value from the same column, so a substrate can become a product and the DIRECTED network moves as
        well as the coefficients. keep_sign=True permutes only the MAGNITUDES and leaves every sign where
        it is, so the directed network is preserved EXACTLY and only mass-balance ratios are destroyed --
        which is the sharper reading of "stoichiometry or mere connectivity?".
        `strength` is the fraction of eligible columns perturbed; see grow_max_strength.
        """
        r_ = np.random.default_rng(seed)
        cols = ELIGIBLE if strength >= 1.0 else np.sort(r_.choice(
            ELIGIBLE, int(round(strength * len(ELIGIBLE))), replace=False))
        M = S.tocsc(copy=True)
        d = M.data.copy()
        for j in cols:
            a, b = M.indptr[j], M.indptr[j + 1]
            if keep_sign:
                d[a:b] = np.sign(d[a:b]) * r_.permutation(np.abs(d[a:b]))
            else:
                d[a:b] = r_.permutation(d[a:b])
        M.data = d
        return M, int((M.data != S.data).sum())

    def rewire_degree_matched(seed, strength=1.0):
        """Configuration model on the bipartite metabolite x reaction incidence.

        The metabolite index of every stub is permuted, so each metabolite keeps its exact reaction degree
        and each reaction keeps its exact number of metabolites AND its exact coefficient multiset. Only
        which metabolite sits at which stub changes. Uniform rewiring would also flatten the metabolite
        degree distribution and could not attribute a drop to wiring rather than to hubs; this can.
        Stubs of the biomass-definition closure are held fixed. Collisions (two stubs of one reaction
        landing on the same metabolite) are summed by the COO constructor and counted, because they are
        the one way this control fails to be exactly degree-matched.
        """
        r_ = np.random.default_rng(seed)
        C = S.tocoo()
        in_clo = np.isin(C.col, np.fromiter(CLOSURE, int))
        free = np.where(~in_clo)[0]
        if strength < 1.0:
            free = r_.choice(free, int(round(strength * len(free))), replace=False)
        mv = C.row.copy()
        mv[free] = C.row[free][r_.permutation(len(free))]
        M = sp.coo_matrix((C.data, (mv, C.col)), shape=S.shape).tocsc()
        M.eliminate_zeros()
        return M, int(C.nnz - M.nnz)

    def grow_max_strength(make, label):
        """DECLARED BEFORE ANY NUMBER: a null variant that cannot grow at full strength is retried at the
        strongest perturbation that still grows, found by bisection on GROWTH ALONE. The endpoint is never
        consulted, so this cannot tune toward a result. A partial null is a WEAKER null -- it leaves most
        of the real network intact -- so it makes the gate HARDER for the FBA arm, never easier, and any
        gate decided against a partial null is reported UNRESOLVED rather than as evidence.
        """
        M, ch = make(1.0)
        g = solve(M, LB, UB)
        if np.isfinite(g) and g > 1e-9:
            return M, ch, 1.0, g
        lo, hi, best = 0.0, 1.0, None
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            Mm, cm = make(mid)
            gm = solve(Mm, LB, UB)
            if np.isfinite(gm) and gm > 1e-9:
                lo, best = mid, (Mm, cm, mid, gm)
            else:
                hi = mid
        if best is None:
            report(f"    {label}: cannot grow at ANY perturbation strength down to {hi:.3f} -> DEGENERATE")
            return M, ch, 1.0, (g if np.isfinite(g) else 0.0)
        report(f"    {label}: full strength does not grow; bisected to the maximal growth-competent "
               f"strength {best[2]:.3f} ({best[1]:,} entries moved). PARTIAL NULL -- any gate decided "
               f"against it is UNRESOLVED, not evidence.")
        return best[0], best[1], best[2], best[3]

    report(f"\n  S variants ({len(CLOSURE)} biomass-definition reactions and every single-metabolite column "
           f"held fixed; {len(ELIGIBLE):,} eligible columns):")
    S4, ch4 = shuffle_within_column(SEED + 101, keep_sign=False)
    S4b, ch4b = shuffle_within_column(SEED + 102, keep_sign=True)
    S5, coll, str5, _g5 = grow_max_strength(lambda s: rewire_degree_matched(SEED + 103, s),
                                            "5 degree-matched rewiring")
    report(f"    arm 4  literal entry permutation within each reaction column: {ch4:,} entries moved "
           f"({100*ch4/S.nnz:.1f}% of nonzeros); substrates and products may exchange places")
    report(f"    arm 4b sign-preserving magnitude permutation: {ch4b:,} entries moved "
           f"({100*ch4b/S.nnz:.1f}%); the directed network is preserved EXACTLY")
    report(f"    arm 5  degree-matched configuration model at strength {str5:.3f}: {coll:,} stub "
           f"collisions merged ({100*coll/S.nnz:.2f}% of nonzeros)")
    for M in (S4, S4b, S5):
        assert M.shape == S.shape and M.nnz <= S.nnz

    # ============================================================ 6. the FBA arms
    def fba_scores(Smat, label):
        wt = solve(Smat, LB, UB)
        wt = 0.0 if not np.isfinite(wt) else wt
        sc = np.zeros(NG)
        cache, n_lp = {}, 0
        if wt <= 1e-9:
            report(f"    {label:34s} WT growth {wt:.6g} -> DEGENERATE: this model cannot grow, so every "
                   f"knockout score is identically 0 and its AUROC is 0.5 BY CONSTRUCTION. Declared in "
                   f"advance: it cannot satisfy the PRIMARY gate.")
            return sc, wt, 0, True
        t_ = time.time()
        for i, g in enumerate(genes):
            dead = ko_rxn.get(g)
            if dead is None:
                continue                      # fully isozyme-covered: predicted non-essential, no LP needed
            key = dead.tobytes()
            if key not in cache:
                lb, ub = LB.copy(), UB.copy()
                lb[dead] = 0.0
                ub[dead] = 0.0
                v = solve(Smat, lb, ub)
                cache[key] = 0.0 if not np.isfinite(v) else v
                n_lp += 1
            sc[i] = 1.0 - cache[key] / wt
        report(f"    {label:34s} WT growth {wt:10.4f}   {n_lp:,} full LP solves in {time.time()-t_:.1f}s   "
               f"lethal (ratio < 0.01) {int((sc > 0.99).sum()):,}   any growth loss {int((sc > 1e-9).sum()):,}")
        return sc, wt, n_lp, False

    report(f"\n  FBA arms -- every knockout is a FULL solve of the complete {S.shape[1]:,}-reaction model "
           f"(no reduced model, no FVA approximation, no gene subsampling):")
    s3, wt3, nlp3, deg3 = fba_scores(S, "3 FBA-constrained (real S)")
    s4, wt4, nlp4, deg4 = fba_scores(S4, "4 shuffled stoichiometry")
    s4b, wt4b, nlp4b, deg4b = fba_scores(S4b, "4b magnitude-shuffled")
    s5, wt5, nlp5, deg5 = fba_scores(S5, "5 degree-matched rewiring")

    # ============================================================ 7. the arm table
    ARMS = collections.OrderedDict()
    ARMS["0N uniform random (null floor)"] = np.random.default_rng(SEED + 7).random(NG)
    ARMS["0 reaction count (cheapest baseline)"] = rxn_count
    ARMS["1 ABUNDANCE-ONLY (kcat demand)"] = a_own
    ARMS["2 reaction-neighbourhood (no stoich)"] = a_own + a_t1 + a_t2
    ARMS["2R random partners (CONTROL)"] = a_own + agg(rnd1) + agg(rnd2)
    ARMS["2S neighbourhood SWAP (COUNTERFACT)"] = a_own + a_t1[swap] + a_t2[swap]
    ARMS["3 FBA-CONSTRAINED (Sv=0)"] = s3
    ARMS["4 SHUFFLED STOICHIOMETRY (CONTROL)"] = s4
    ARMS["4b magnitude-shuffled (CONTROL)"] = s4b
    ARMS["5 degree-matched rewiring (CONTROL)"] = s5
    ARMS["I wrong-gene identity (CONTROL)"] = s3[np.random.default_rng(SEED + 13).permutation(NG)]
    DEGEN = {"4 SHUFFLED STOICHIOMETRY (CONTROL)": deg4, "4b magnitude-shuffled (CONTROL)": deg4b,
             "5 degree-matched rewiring (CONTROL)": deg5, "3 FBA-CONSTRAINED (Sv=0)": deg3}
    for k, v in ARMS.items():
        assert v.shape == (NG,) and np.isfinite(v).all(), f"arm {k} does not score the whole universe"

    # DISJOINT held-out GENE blocks, stratified so no block is label-degenerate.
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rp = np.random.default_rng(SEED + 3)
    rp.shuffle(pos)
    rp.shuffle(neg)
    blocks = [np.sort(np.concatenate([a, b])) for a, b in
              zip(np.array_split(pos, FOLDS), np.array_split(neg, FOLDS))]
    report(f"\n  {FOLDS} DISJOINT held-out GENE blocks (stratified on the label), sizes "
           + "/".join(str(len(b)) for b in blocks) + "; essential per block "
           + "/".join(str(int(y[b].sum())) for b in blocks))

    res = {k: {"auroc": [], "rho": [], "sign": [], "auroc_apriori": []} for k in ARMS}
    import warnings
    for f in range(FOLDS):
        te = blocks[f]
        tr = np.sort(np.concatenate([b for i, b in enumerate(blocks) if i != f]))
        for k, v in ARMS.items():
            # ONE parameter is fit from the endpoint anywhere in this module: the SIGN, on TRAINING genes
            # only, against the CONTINUOUS mean effect (more power than the binary label, so the sign
            # flips less often on noise). It is fit because arms 0-2 have no a-priori-obvious direction
            # and a hard-coded convention would cripple whichever baseline guessed wrong -- and a crippled
            # baseline is exactly the defect this project keeps shipping. It is not free: on an
            # uninformative arm the sign is chosen on noise, which biases AUROC UPWARD. The
            # uniform-random arm measures that bias directly and is reported for exactly that purpose,
            # and the unfitted a-priori-direction AUROC is reported alongside every arm.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho_tr = spearmanr(v[tr], -mu_eff[tr]).statistic
                s_ = 1.0 if (not np.isfinite(rho_tr) or rho_tr >= 0) else -1.0
                a_ = auroc(s_ * v[te], y[te])
                ap_ = auroc(v[te], y[te])
                r_ = spearmanr(s_ * v[te], -mu_eff[te]).statistic
            res[k]["auroc"].append(float(a_))
            res[k]["auroc_apriori"].append(float(ap_))
            res[k]["rho"].append(0.0 if not np.isfinite(r_) else float(r_))
            res[k]["sign"].append(s_)

    def m(k, met="auroc"):
        return float(np.mean(res[k][met]))

    report(f"\n  {'arm':40s} {'AUROC':>8s} {'sd':>7s} {'apriori':>8s} {'spearman':>9s} {'sd':>7s}  note")
    for k, v in res.items():
        note = "DEGENERATE: model cannot grow, AUROC 0.5 by construction" if DEGEN.get(k) else ""
        report(f"  {k:40s} {np.mean(v['auroc']):8.4f} {np.std(v['auroc'], ddof=1):7.4f} "
               f"{np.mean(v['auroc_apriori']):8.4f} "
               f"{np.mean(v['rho']):+9.4f} {np.std(v['rho'], ddof=1):7.4f}  {note}")
    report(f"  'apriori' is the same AUROC with the sign NOT fitted. The gap between the two columns on the "
           f"uniform-random arm\n  ({m('0N uniform random (null floor)')-float(np.mean(res['0N uniform random (null floor)']['auroc_apriori'])):+.4f}) "
           f"is the upward bias the per-fold sign fit buys every arm. Read every AUROC against the "
           f"random arm's\n  {m('0N uniform random (null floor)'):.4f}, NOT against 0.5.")

    sd_a = float(np.mean([np.std(v["auroc"], ddof=1) for v in res.values()]))
    sd_r = float(np.mean([np.std(v["rho"], ddof=1) for v in res.values()]))
    MDE = 3 * sd_a / np.sqrt(FOLDS)
    MDE_R = 3 * sd_r / np.sqrt(FOLDS)
    report(f"\n  MDE = 3*sd/sqrt(n), n = {FOLDS} DISJOINT HELD-OUT GENE BLOCKS -- the unit of replication, "
           f"because the\n  claim is about generalising to metabolic genes, not to solver restarts or to "
           f"cell lines. Pooled\n  across-block sd {sd_a:.4f} -> MDE {MDE:+.4f} AUROC; Spearman MDE {MDE_R:+.4f}")

    # per-CELL-LINE spread: REPORTED, NEVER GRADED. Cell lines are strongly correlated and none of them is
    # an unseen gene; an n of ~1,150 here would manufacture significance for a claim about genes.
    perline = {}
    for k in ("3 FBA-CONSTRAINED (Sv=0)", "1 ABUNDANCE-ONLY (kcat demand)", "0 reaction count (cheapest baseline)"):
        v = ARMS[k] * (1.0 if np.mean(res[k]["sign"]) >= 0 else -1.0)
        a_ = []
        for i in range(Eg.shape[0]):
            yi = Eg[i] <= ESS_THRESH
            ok_i = np.isfinite(Eg[i])
            if ok_i.sum() < 10:
                continue
            u = auroc(v[ok_i], yi[ok_i])
            if np.isfinite(u):
                a_.append(u)
        perline[k] = {"mean": float(np.mean(a_)), "sd": float(np.std(a_, ddof=1)), "n_lines": len(a_)}
    report(f"  REPORTED AND NOT GRADED -- per-cell-line AUROC (n = {perline['3 FBA-CONSTRAINED (Sv=0)']['n_lines']:,} "
           f"lines, each scored on the whole gene universe): "
           + "; ".join(f"{k.split()[0]} {d['mean']:.4f} +/- {d['sd']:.4f}" for k, d in perline.items())
           + ". Cell lines are not the unit of replication and no gate reads this.")

    # ============================================================ 8. the predeclared gates
    g32 = m("3 FBA-CONSTRAINED (Sv=0)") - m("2 reaction-neighbourhood (no stoich)")
    g34 = m("3 FBA-CONSTRAINED (Sv=0)") - m("4 SHUFFLED STOICHIOMETRY (CONTROL)")
    g34b = m("3 FBA-CONSTRAINED (Sv=0)") - m("4b magnitude-shuffled (CONTROL)")
    g35 = m("3 FBA-CONSTRAINED (Sv=0)") - m("5 degree-matched rewiring (CONTROL)")
    g31 = m("3 FBA-CONSTRAINED (Sv=0)") - m("1 ABUNDANCE-ONLY (kcat demand)")
    g30 = m("3 FBA-CONSTRAINED (Sv=0)") - m("0 reaction count (cheapest baseline)")
    g3I = m("3 FBA-CONSTRAINED (Sv=0)") - m("I wrong-gene identity (CONTROL)")
    g21 = m("2 reaction-neighbourhood (no stoich)") - m("1 ABUNDANCE-ONLY (kcat demand)")
    g22r = m("2 reaction-neighbourhood (no stoich)") - m("2R random partners (CONTROL)")
    g22s = m("2 reaction-neighbourhood (no stoich)") - m("2S neighbourhood SWAP (COUNTERFACT)")
    g32_r = m("3 FBA-CONSTRAINED (Sv=0)", "rho") - m("2 reaction-neighbourhood (no stoich)", "rho")
    g34_r = m("3 FBA-CONSTRAINED (Sv=0)", "rho") - m("4 SHUFFLED STOICHIOMETRY (CONTROL)", "rho")

    # DECLARED ORDER: the PRIMARY shuffle control is the first of [4, 4b] that is growth-competent at full
    # strength. A model that cannot grow scores 0.5 by construction and is a broken model, not a null
    # network, so it cannot be allowed to carry a gate.
    shuffle_arm, shuffle_gap = "4 SHUFFLED STOICHIOMETRY (CONTROL)", g34
    if deg4:
        shuffle_arm, shuffle_gap = "4b magnitude-shuffled (CONTROL)", g34b
    primary = bool((g32 > MDE) and (shuffle_gap > MDE) and not (deg4 and deg4b))
    unresolved = {
        "PRIMARY": bool(deg4 and deg4b),
        "SHARP": bool(deg4b),
        "WIRING": bool(deg5 or str5 < 1.0),
    }
    gates = {
        "PRIMARY  FBA beats BOTH the stoichiometry-free neighbourhood AND the shuffled-S control by > MDE":
            primary,
        "SHARP    FBA beats the sign-preserving magnitude shuffle by > MDE (coefficients, not sign pattern)":
            bool(g34b > MDE),
        "WIRING   FBA beats the degree-matched configuration-model rewiring by > MDE": bool(g35 > MDE),
        "ABUNDANCE FBA beats ABUNDANCE-ONLY by > MDE (the headline sentence)": bool(g31 > MDE),
        "FREE     FBA beats a bare reaction count per gene by > MDE": bool(g30 > MDE),
        "AGREE    threshold-free Spearman reaches the same verdict as AUROC on both PRIMARY gaps":
            bool(((g32 > MDE) == (g32_r > MDE_R)) and ((g34 > MDE) == (g34_r > MDE_R))),
    }
    report("\n  PREDECLARED GATES")
    for k, v in gates.items():
        tag = "UNRESOLVED" if unresolved.get(k.split()[0]) else ("PASS" if v else "FAIL")
        report(f"    {tag:10s}  {k}")
    if any(unresolved.values()):
        report(f"    UNRESOLVED means the control that gate is measured against could not be built as a "
               f"growth-competent null (see the S-variant block above). It is NOT a pass and NOT evidence "
               f"of absence; it means this design cannot decide that gate on this model.")
    report(f"    FBA {m('3 FBA-CONSTRAINED (Sv=0)'):.4f} | neighbourhood {m('2 reaction-neighbourhood (no stoich)'):.4f} "
           f"({g32:+.4f}) | shuffled-S {m('4 SHUFFLED STOICHIOMETRY (CONTROL)'):.4f} ({g34:+.4f}) | "
           f"magnitude-shuffled {m('4b magnitude-shuffled (CONTROL)'):.4f} ({g34b:+.4f})")
    report(f"    abundance-only {m('1 ABUNDANCE-ONLY (kcat demand)'):.4f} ({g31:+.4f}) | reaction count "
           f"{m('0 reaction count (cheapest baseline)'):.4f} ({g30:+.4f}) | null floor "
           f"{m('0N uniform random (null floor)'):.4f} | wrong-gene identity "
           f"{m('I wrong-gene identity (CONTROL)'):.4f} ({g3I:+.4f}) -- all against MDE {MDE:.4f}")
    report(f"    arm 2 channel is LIVE not ignored: neighbourhood vs random partners {g22r:+.4f}, vs the "
           f"COUNTERFACTUAL swap {g22s:+.4f} (a swap gap at or below the MDE would mean the neighbourhood "
           f"is IGNORED rather than useless -- accuracy alone cannot tell those apart)")

    # A control that reproduces most of the effect IS the finding.
    base = m("0N uniform random (null floor)")
    denom = m("3 FBA-CONSTRAINED (Sv=0)") - base

    def frac(k):
        return float((m(k) - base) / denom) if abs(denom) > 1e-9 else float("nan")

    # Every entry is "fraction of arm 3's raw lift over the uniform-random floor that this arm reproduces".
    repro = {"shuffled_stoichiometry": frac("4 SHUFFLED STOICHIOMETRY (CONTROL)"),
             "magnitude_shuffled": frac("4b magnitude-shuffled (CONTROL)"),
             "degree_matched_rewiring": frac("5 degree-matched rewiring (CONTROL)"),
             "stoichiometry_free_neighbourhood": frac("2 reaction-neighbourhood (no stoich)"),
             "abundance_only": frac("1 ABUNDANCE-ONLY (kcat demand)"),
             "bare_reaction_count": frac("0 reaction count (cheapest baseline)")}
    report("\n  CONTROLS AND BASELINES THAT REPRODUCE THE EFFECT (the finding, not a footnote)")
    report(f"    denominator = FBA's raw lift over the uniform-random floor, {denom:+.4f} AUROC")
    for k, v in repro.items():
        report(f"    {100*v:7.1f}%  {k}")
    repro_ok = bool(denom > MDE)
    if not repro_ok:
        report(f"    ^ FBA's own lift over the null floor is {denom:+.4f}, itself at or below the {MDE:.4f} "
               f"MDE, so these are ratios of two undetected quantities and mean nothing. They are printed "
               f"rather than suppressed so the denominator stays visible.")

    best_free = max(["0 reaction count (cheapest baseline)", "0N uniform random (null floor)"], key=m)
    winner = max(res, key=m)
    smoke_pre = (
        f"SMOKE RUN -- THIS IS A PLUMBING CHECK AND NOT A RESULT. The gene universe was deliberately "
        f"subsampled to {NG} genes ({int(y.sum())} essential) across {FOLDS} blocks, so every AUROC here is "
        f"decided by which two or three essential genes landed in which block, no arm can be separated from "
        f"any other, and the MDE is an artifact of that subsample. Nothing in this paragraph or this JSON is "
        f"quotable as a finding. Rerun with MC_SMOKE unset for the full {len(ko_rxn):,}-LP-per-arm result. "
        if SMOKE else "")
    verdict = (
        smoke_pre +
        f"{'BLOCK 7A PASSES' if primary else 'BLOCK 7A FAILS'}: on {NG:,} HumanGEM metabolic genes with a "
        f"DepMap column and >= {MIN_LINES} measured cell lines ({int(y.sum()):,} essential at mean Chronos "
        f"gene effect <= {ESS_THRESH}), a real Sv = 0 solve of the full {S.shape[1]:,}-reaction "
        f"reconstruction -- {nlp3:,} full single-gene-knockout LPs through "
        f"scipy.optimize.linprog/HiGHS, cross-checked against cobrapy's own wild-type solution to "
        f"{abs(wt_lp-wt_cobra):.1e} -- reaches held-out AUROC {m('3 FBA-CONSTRAINED (Sv=0)'):.4f}, while "
        f"predicted enzyme abundance alone reaches {m('1 ABUNDANCE-ONLY (kcat demand)'):.4f} "
        f"({g31:+.4f}), the stoichiometry-free reaction neighbourhood reaches "
        f"{m('2 reaction-neighbourhood (no stoich)'):.4f} ({g32:+.4f}) and the SHUFFLED-STOICHIOMETRY "
        f"control -- same sparsity, same per-reaction coefficient multiset, entries permuted within the "
        f"column -- reaches {m('4 SHUFFLED STOICHIOMETRY (CONTROL)'):.4f} ({g34:+.4f}), all against a "
        f"cross-block minimum detectable increment of {MDE:.4f} (3*sd/sqrt(n), n = {FOLDS} DISJOINT "
        f"HELD-OUT GENE BLOCKS; the per-cell-line spread is reported in the JSON and is deliberately NOT "
        f"what anything is graded on, because cell lines are correlated and none of them is an unseen "
        f"gene). "
        + (f"THE LITERAL SHUFFLE IS NOT CONSTRUCTIBLE ON THIS MODEL: permuting entries within a reaction "
           f"column lets substrates and products exchange places, which destroys the directed network so "
           f"completely that the wild-type objective falls to {wt4:.4g} and every knockout score is "
           f"identically zero, giving AUROC 0.5 by construction rather than by biology. That is a property "
           f"of the control, not a result about stoichiometry. By the rule declared before the run the "
           f"PRIMARY gate was therefore taken against the SIGN-PRESERVING magnitude shuffle "
           f"({m('4b magnitude-shuffled (CONTROL)'):.4f}, gap {g34b:+.4f}), which keeps every "
           f"substrate a substrate and every product a product, destroys only the mass-balance ratios, "
           f"and grows at {wt4b:.4g} against a wild-type {wt3:.4g} -- the sharper reading of "
           f"'stoichiometry or mere connectivity?' in any case. "
           if deg4 else
           f"The shuffled model still grows ({wt4:.4g} against a wild-type {wt3:.4g}), so it is a genuine "
           f"null network rather than a broken one and the comparison is meaningful. ")
        + (f"The degree-matched rewiring could not grow at full strength either and was bisected, on "
           f"growth alone and never on the endpoint, to strength {str5:.3f}; a partial null leaves most of "
           f"the real network intact, so the WIRING gate is reported UNRESOLVED rather than as evidence. "
           if unresolved["WIRING"] else
           f"The degree-matched configuration-model rewiring is growth-competent at full strength "
           f"({wt5:.4g}). ")
        + (f"Note that the per-fold sign fit lifts even a uniform-random score to "
           f"{m('0N uniform random (null floor)'):.4f}, so that -- not 0.5 -- is the floor every AUROC "
           f"here must be read against. ")
        + f"ATTRIBUTION, as fractions of FBA's raw lift over the uniform-random floor: the "
        f"shuffled-stoichiometry control reproduces {100*repro['shuffled_stoichiometry']:.1f}%, the "
        f"sign-preserving magnitude shuffle {100*repro['magnitude_shuffled']:.1f}%, the degree-matched "
        f"configuration-model rewiring {100*repro['degree_matched_rewiring']:.1f}%, the "
        f"stoichiometry-free neighbourhood {100*repro['stoichiometry_free_neighbourhood']:.1f}%, and a "
        f"bare count of reactions per gene -- no kcat, no stoichiometry, no solver -- "
        f"{100*repro['bare_reaction_count']:.1f}%"
        + ("" if repro_ok else " (though FBA's own lift over the floor is itself below the MDE, so these "
                               "percentages are ratios of two undetected quantities)")
        + f". THE CHEAPEST NON-LEARNED ARM, a bare reaction count, scores "
        f"{m('0 reaction count (cheapest baseline)'):.4f} -- "
        + ("the solve is ahead of it. " if g30 > MDE else
           "A TRIVIAL ZERO-COST BASELINE MATCHES OR BEATS THE FULL LP PIPELINE, WHICH IS THE HEADLINE AND "
           "NOT A FOOTNOTE. ")
        + f"The best arm overall is '{winner}' at {m(winner):.4f}. The wrong-gene identity control sits at "
        f"{m('I wrong-gene identity (CONTROL)'):.4f}, bounding the metric's own degeneracy, and the "
        f"uniform-random floor at {m('0N uniform random (null floor)'):.4f}. The stoichiometry-free "
        f"neighbourhood channel is live rather than ignored: swapping a gene's neighbours for another "
        f"gene's while keeping its own demand term moves AUROC by {g22s:+.4f} and random count-matched and "
        f"tier-matched partners by {g22r:+.4f}. The threshold-free Spearman correlation against the "
        f"continuous mean gene effect "
        f"{'AGREES' if gates['AGREE    threshold-free Spearman reaches the same verdict as AUROC on both PRIMARY gaps'] else 'DISAGREES'} "
        f"with AUROC on both PRIMARY gaps ({g32_r:+.4f} and {g34_r:+.4f} against a {MDE_R:.4f} Spearman "
        f"MDE)"
        + ("." if gates["AGREE    threshold-free Spearman reaches the same verdict as AUROC on both PRIMARY gaps"]
           else ", and when the two metrics disagree the result is a metric artifact and is not reportable "
                "as biology.")
        + " All effect sizes quoted are RAW held-out values; the shuffled, rewired, random-partner, swap "
        "and wrong-gene arms establish significance and attribution only, and nothing is reported net of a "
        "control.")

    report("\n" + "=" * 108)
    report(f"  VERDICT: {verdict}")

    R = {
        "model": "v4-metabolic-constraint-v1",
        "title": "V4 Block 7A -- HumanGEM as a constraint layer vs abundance-only",
        "question": "Does imposing Sv = 0 via HumanGEM predict metabolic-gene essentiality better than "
                    "predicted enzyme abundance alone, better than stoichiometry-free connectivity, and "
                    "better than a shuffled-stoichiometry model with the same sparsity?",
        "smoke": bool(SMOKE),
        "solver": solver_note,
        "inputs": {"sbml": str(SBML), "kcat": str(KCAT), "depmap": str(DEPMAP), "genemap": str(GENEMAP)},
        "endpoint": {
            "source": "DepMap CRISPRGeneEffect.csv (Chronos gene effect)",
            "truth": f"per-gene MEAN gene effect across measured cell lines; essential = mean <= {ESS_THRESH}",
            "nan_policy": "NaNs dropped from the mean, NEVER read as 0.0; a gene needs "
                          f">= {MIN_LINES} measured lines",
            "nan_fraction": nan_frac,
            "n_genes": int(NG), "n_essential": int(y.sum()), "prevalence": float(y.mean()),
            "n_cell_lines": int(Eg.shape[0]),
            "primary_metric": "AUROC with midranks (FBA's large tie block scores at chance, which is honest)",
            "secondary_metric": "Spearman rho against the continuous mean gene effect (threshold-free)"},
        "model_structure": {"reactions": int(S.shape[1]), "metabolites": int(S.shape[0]), "nnz": int(S.nnz),
                            "objective_reaction": rxns[OBJ_COL].id,
                            "genes_with_nonempty_gpr_knockout_set": int(len(ko_rxn)),
                            "kcat_tier_counts": {k: int(v) for k, v in tier.items()},
                            "kcat_global_median_per_s": GLOBAL_KCAT,
                            "currency_metabolite_degree_cut": cur_cut,
                            "rewiring_stub_collisions": int(coll)},
        "wt_growth": {"real_S": wt3, "shuffled_S": wt4, "magnitude_shuffled_S": wt4b, "rewired_S": wt5,
                      "degenerate": {"shuffled_S": bool(deg4), "magnitude_shuffled_S": bool(deg4b),
                                     "rewired_S": bool(deg5)},
                      "perturbation_strength": {"shuffled_S": 1.0, "magnitude_shuffled_S": 1.0,
                                                "rewired_S": float(str5)},
                      "entries_moved": {"shuffled_S": int(ch4), "magnitude_shuffled_S": int(ch4b)},
                      "biomass_definition_closure_held_fixed": sorted(rxns[j].id for j in CLOSURE),
                      "note": "a null variant that cannot grow scores 0.5 by construction and is a BROKEN "
                              "model, not a null network; it cannot carry a gate. Perturbation strength "
                              "below 1.0 means the variant was bisected down on GROWTH ALONE (never on "
                              "the endpoint) and is a PARTIAL null, which makes the gate harder, not "
                              "easier, and renders that gate UNRESOLVED."},
        "lp_status_counts": {str(k): int(v) for k, v in lp_status.items()},
        "lp_solves": {"real_S": int(nlp3), "shuffled_S": int(nlp4), "magnitude_shuffled_S": int(nlp4b),
                      "rewired_S": int(nlp5),
                      "note": "every solve is the FULL model; no reduced model, no FVA approximation, no "
                              "gene subsampling in a full run"},
        "arms": {k: {"auroc_mean": float(np.mean(v["auroc"])),
                     "auroc_sd": float(np.std(v["auroc"], ddof=1)),
                     "auroc_per_fold": v["auroc"],
                     "auroc_apriori_mean": float(np.mean(v["auroc_apriori"])),
                     "auroc_apriori_per_fold": v["auroc_apriori"],
                     "spearman_mean": float(np.mean(v["rho"])),
                     "spearman_sd": float(np.std(v["rho"], ddof=1)),
                     "spearman_per_fold": v["rho"],
                     "sign_fit_per_fold": v["sign"],
                     "degenerate": bool(DEGEN.get(k, False))}
                 for k, v in res.items()},
        "sign_fit_bias_floor": {
            "uniform_random_arm_auroc": m("0N uniform random (null floor)"),
            "uniform_random_arm_auroc_apriori":
                float(np.mean(res["0N uniform random (null floor)"]["auroc_apriori"])),
            "note": "the per-fold sign fit is chosen on training genes and therefore lifts even a "
                    "meaningless score above 0.5. Read every AUROC in this file against the uniform-random "
                    "arm, not against 0.5."},
        "unit_of_replication": f"a DISJOINT HELD-OUT GENE BLOCK. The per_fold lists hold one value per "
                               f"held-out block (n = {FOLDS}), NOT cell lines and NOT solver restarts.",
        "mde": {"value": float(MDE), "convention": "3*sd/sqrt(n)", "n": int(FOLDS),
                "n_counts": "disjoint held-out metabolic-gene blocks, stratified on the essentiality label",
                "pooled_across_block_sd": sd_a,
                "spearman_mde": float(MDE_R), "pooled_across_block_sd_spearman": sd_r},
        "per_cell_line_REPORTED_NOT_GRADED": perline,
        "per_cell_line_note": "cell lines are strongly correlated and none of them is an unseen gene; an n "
                              "of ~1,150 there would manufacture significance for a claim about genes. No "
                              "gate reads this.",
        "gates": {k: bool(v) for k, v in gates.items()},
        "gates_unresolved": {k: bool(v) for k, v in unresolved.items()},
        "gates_unresolved_note": "an UNRESOLVED gate is not a pass and not evidence of absence: the null "
                                 "control it is measured against could not be built as a growth-competent "
                                 "model, so this design cannot decide it on this reconstruction.",
        "gate_shuffle_arm_used_for_PRIMARY": shuffle_arm,
        "gaps": {"fba_minus_neighbourhood": g32, "fba_minus_shuffled_stoichiometry": g34,
                 "fba_minus_magnitude_shuffled": g34b, "fba_minus_degree_matched_rewiring": g35,
                 "fba_minus_abundance_only": g31, "fba_minus_reaction_count": g30,
                 "fba_minus_wrong_gene_identity": g3I,
                 "neighbourhood_minus_abundance_only": g21,
                 "neighbourhood_minus_random_partners": g22r,
                 "neighbourhood_minus_counterfactual_swap": g22s,
                 "fba_minus_neighbourhood_spearman": g32_r,
                 "fba_minus_shuffled_spearman": g34_r},
        "controls_reproducing_the_effect": {k: (float(v) if np.isfinite(v) else None)
                                            for k, v in repro.items()},
        "controls_reproducing_the_effect_denominator": float(denom),
        "controls_reproducing_the_effect_note":
            "each value is the fraction of arm 3's RAW lift over the uniform-random floor that the named "
            "arm reproduces. If the denominator is itself below the MDE these are ratios of two undetected "
            "quantities and mean nothing, and the flag below is then false.",
        "controls_reproducing_the_effect_are_meaningful": repro_ok,
        "best_arm_overall": winner, "best_zero_cost_arm": best_free,
        "circularity_check": "HumanGEM is a literature-curated reconstruction, its EC annotations come from "
                             "the model file, and kcat values come from a BRENDA/SABIO-derived deposit. The "
                             "scored endpoint is DepMap CRISPR gene effect. No arm's score is derived from "
                             "DepMap. The only quantity fit from the endpoint anywhere in this module is one "
                             "SIGN per arm per fold, fit on training genes and applied to held-out genes.",
        "limits": [
            f"n = {FOLDS} held-out gene blocks gives the sd {FOLDS-1} degrees of freedom, so the MDE is "
            f"itself poorly estimated and a gap just under it is not evidence of equality. Raising MC_FOLDS "
            f"is the cheapest way to falsify or confirm a near-threshold call",
            "WHAT WOULD FALSIFY A POSITIVE: if arm 4b or arm 5 moves up to arm 3 under a different shuffle "
            "seed, the separation was one draw of the permutation and not stoichiometry. Only one seed per "
            "shuffled variant is run here (MC_SEED); a seed sweep is the direct falsification test and it "
            "was NOT run",
            "THE LITERAL SHUFFLED-STOICHIOMETRY CONTROL MAY NOT BE CONSTRUCTIBLE AT ALL. Permuting entries "
            "within a reaction column moves signs, so substrates become products and the directed network "
            "is destroyed along with the coefficients; on HumanGEM that drives the wild-type objective to "
            "zero, and a model that cannot grow scores AUROC 0.5 by construction rather than by biology. "
            "Check wt_growth.degenerate before quoting any arm-4 number. The sign-preserving magnitude "
            "shuffle (arm 4b) is the control that actually isolates stoichiometry from connectivity, and "
            "the gate falls back to it by a rule declared before the run",
            "the biomass-definition closure (the objective reaction plus, transitively, the sole producer "
            "of anything it consumes -- HumanGEM's explicit protein/lipid/metabolite/cofactor pool "
            "pseudo-reactions) is held FIXED in every null variant, because permuting it would change what "
            "growth MEANS rather than changing the network. Without that exclusion no shuffled variant "
            "grows at all. The closure is listed in wt_growth; it is a structural rule, but a different "
            "rule would pick a different set and was not swept",
            "THE PER-FOLD SIGN FIT BIASES EVERY AUROC UPWARD. Each arm's direction is chosen on the "
            "training genes, so an uninformative arm still picks the luckier sign and scores above 0.5. "
            "That is why the uniform-random arm is run at all: its AUROC is the empirical floor, the "
            "unfitted a-priori AUROC is reported next to every arm, and reading any number here against "
            "0.5 instead of against the random arm will overstate it. The bias shrinks with universe size "
            "and is severe in smoke mode",
            "WHAT WOULD FALSIFY A NEGATIVE: this uses HumanGEM's own default bounds as the medium and makes "
            "no cell-line-specific model. Context-specific FBA (expression-constrained models, a defined "
            "medium) is the standard way this prediction is made and it is not what was run. A null here is "
            "a null for the GENERIC reconstruction, not for constraint-based modelling",
            "'predicted enzyme abundance' is a PROXY. No proteomics was available in this environment, so "
            "arm 1 is enzyme demand per unit flux, sum_r 1/(kcat_r * n_isozymes_r). A real abundance "
            "measurement could beat it, which would make arm 1 a stronger baseline and the headline claim "
            "harder, not easier",
            "kcat coverage is partial: reactions with no usable EC match fall back to the global median, "
            "which collapses arm 1 toward arm 0 (a reaction count) for those genes. The per-tier counts are "
            "in model_structure.kcat_tier_counts and this WEAKENS the abundance baseline, which biases the "
            "experiment toward a positive verdict and must be read as such",
            "CONFOUNDED: DepMap essentiality is correlated with gene expression, with pathway membership "
            "and with how well-studied a gene is, and HumanGEM's curation is itself denser around "
            "well-studied central metabolism. A gene that is essential and well-curated is over-represented "
            "in both, so part of any positive AUROC is annotation depth rather than mass balance. Nothing "
            "here separates those",
            "the essentiality label is a MEAN across ~1,150 heterogeneous cell lines thresholded at "
            f"{ESS_THRESH}. That discards context-specific essentiality entirely, which is exactly the "
            "regime where a constraint-based model should have its advantage. The per-cell-line numbers are "
            "reported but the design is not powered for them",
            "arm 3's score is a huge tie block: every gene whose GPR leaves its reactions isozyme-covered "
            "gets exactly 0.0 and is scored at chance by midranks. That is the honest treatment of an arm "
            "that asserts no order, but it caps arm 3's achievable AUROC in a way arms 0-2 are not capped, "
            "and the cap depends on GPR curation rather than on stoichiometry",
            "the objective (biomass) column is excluded from the shuffled and rewired matrices, because "
            "shuffling it would change what growth MEANS rather than changing the network. That makes the "
            "controls slightly closer to the real model than a full shuffle would be, i.e. it is the "
            "conservative direction for a positive verdict but the anti-conservative direction for a null",
            "the degree-matched rewiring merges stub collisions, so a small fraction of nonzeros "
            "(reported as rewiring_stub_collisions) leaves it not exactly degree-matched",
            "single-gene knockouts only. Synthetic lethality, isozyme pairs and any epistasis are outside "
            "this design entirely, and they are where a network model would be expected to earn its keep",
            "arm 2's currency-metabolite cut (top 5% by reaction degree) is a declared preprocessing "
            "choice, not a fitted one, but a different cut changes arm 2's neighbourhood size and therefore "
            "the difficulty of the PRIMARY gate. It was not swept"],
        "verdict": verdict,
        "runtime_seconds": float(time.time() - t0),
        "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "v4_metabolic_constraint.json"
    json.dump(R, open(path, "w"), indent=1)
    report(f"\n  -> {path}   ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
