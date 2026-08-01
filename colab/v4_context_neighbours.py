"""V4 SECTION 19 -- N_c(p), THE CELL-CONDITIONED PARTNER SET. The one term of F_context never tested.

THE QUESTION, AND WHY IT IS DISTINCT FROM EVERY CONTEXT TEST THIS PROJECT HAS ALREADY FAILED.
`CELLFORMER4.md` section 19 writes the context branch as

    F_context( z_c, h[p,c], h[g,c], N_c(p), E, t )

and the subscript on `N_c(p)` is not decoration: it says the PARTNER SET ITSELF depends on the cell. Every
context experiment run here so far -- `context_tests.py`, `contrast_context.py`, `dense_context.py` -- fed
CELL-CONDITIONED NODE FEATURES (baseline expression of the perturbed gene, of the response gene, of the
neighbours, read in cell c) over a CELL-INVARIANT NEIGHBOUR SET. The set was the same object in every cell;
only the numbers hanging off it moved. Those tests returned 0/7 predeclared criteria, a swap gap of -0.0375,
and a trivial nearest-cell copy beating the model. What has NEVER been tested is the other half of the
subscript: does choosing DIFFERENT PARTNERS in different cells help? That is this module, and it is the only
part of `F_context` that the existing negative results do not already cover.

WHAT THE EXISTING EVIDENCE PREDICTS, STATED BEFORE THE RUN SO IT CANNOT BE RETROFITTED. Three numbers already
in the registry point the same way, and one measured property of the gating variable points harder:

    correct vs swapped cell context (dense_context)          -0.0375   FAILED, negative in 3 of 4 cells
    context residual vs nearest-cell copy (dense_context)     model +0.0622 vs copy +0.0674   COPY WINS
    internal wiring vs shuffled wiring at matched density     +0.005 +/- 0.008   NOT DETECTED
    cross-cell Jaccard of N_c(p) under percentile gating      ~0.85 at the 25th percentile (measured below)

The last line is the one that matters most and it is a property of the DATA, not of any model. The four
baseline transcriptomes correlate at r 0.80-0.88, so a partner set gated on "is this gene expressed in cell
c" is ~85% THE SAME SET in every cell. The predeclared expectation is therefore that the primary gated arm
is indistinguishable from the cell-invariant arm, and that if anything separates it will be the DIFFERENTIAL
gate (cross-cell Jaccard ~0.22), which is the only construction here that makes the set genuinely
cell-specific. Both are run, and which one was primary was fixed before the run.

THE TARGET, TAKEN FROM `dense_context.py` UNCHANGED so the numbers sit on the same axis as the -0.0375 swap
gap and the +0.0674 nearest-cell copy.

    Y[c,p,g] = mu[p,g] + delta[c,p,g],   mu = mean over the THREE TRAINING cells only

Because mu is the mean of the training cells, `sum_c delta[c,p,g] = 0 EXACTLY` over those cells, so on the
training target every cell-blind feature -- gene identity, perturbation identity, the shared response, the
tide -- has an optimal prediction of ZERO. There is no cell-blind shortcut left. Data is
`dense_response.npz`: 4 cells x 1,763 perturbations x 6,550 genes, 100% DENSE. It is never the `nlz_*`
benches, which are 3.5% dense and whose remaining 96.5% of zeros mean NOT RECORDED and are not a target.

WHAT IS AND IS NOT ALLOWED AS INPUT. The held-out cell contributes its BASELINE TRANSCRIPTOME
(`baseline_state.json.gz`, pooled non-targeting-control cells, response-free by construction) and nothing
else. Its perturbation responses are never an input, never used to fit mu, and never used to standardise a
feature. The gate that builds N_c(p) is allowed to read the held-out cell's baseline -- that is precisely
the quantity a cell encoder would have to read -- and is never allowed to read its responses.

THE ARMS. Every arm gets the IDENTICAL seven-column shared block (perturbed-gene baseline, response-gene
baseline, both of their deviations from the training cells, mu, and two mu-interactions) and its own
five-column neighbour block (mean partner baseline, its deviation from the training cells, a mu-interaction,
the within-set spread, and the retention rate). ONLY WHICH PARTNERS ENTER THE NEIGHBOUR BLOCK VARIES.

    0-param  Z  zero contrast                predict delta = 0. The cell-blind optimum BY CONSTRUCTION
    0-param  NC nearest-cell copy            copy delta from the training cell with the closest baseline
                                             transcriptome. NO PARAMETERS, NO GRADIENT. This is the arm that
                                             BEAT the model in dense_context, and it is reported next to the
                                             model, not in a footnote
    0-param  LS ordinary least squares       closed-form linear fit on the primary gated arm's own columns.
                                             The cheapest sensible USE of N_c(p): if it matches the MLP the
                                             network is not what produced anything
    1  cell-invariant N(p)                   THE INCUMBENT. Cell-conditioned features, cell-invariant set.
                                             This is exactly what every previous context test had
    2  cell-gated N_c(p), ABSOLUTE           PRIMARY. Drop partners below the V4C_PCTL-th percentile of
                                             baseline expression IN CELL c. This is the construction named
                                             in the brief and it is what the gate is declared on
    3  cell-gated N_c(p), DIFFERENTIAL       SECONDARY, declared as secondary before the run. Keep a partner
                                             only if it is expressed ABOVE its own mean over the other three
                                             lines -- relatively high HERE. The only construction whose sets
                                             genuinely differ between cells
    4  cell-context EVIDENCE preference      prefer partners whose `v4_relations.parquet` edge carries a
                                             `cell_context` naming this line, falling back to N(p) when none
                                             exist. SEVERELY ASYMMETRIC: K562 1,205 perturbations, HepG2 868,
                                             Jurkat 235, RPE1 ZERO -- for RPE1 this arm IS arm 1 by
                                             construction. Reported per cell for that reason
    5  SWAPPED-cell gating       CONTROL     COUNTERFACTUAL. The set is chosen by ANOTHER cell's expression
                                             while the features are still read in cell c. Accuracy alone
                                             cannot distinguish "the cell-conditioned set is useless" from
                                             "gating helps for a reason that has nothing to do with WHICH
                                             cell"; only this can. Same gate, same retention regime, wrong
                                             cell's evidence
    6  random gating, matched retention CONTROL  drop a RANDOM subset of N(p) of exactly the size arm 2's
                                             gate left behind, per (perturbation, cell). Separates "a smaller
                                             set is better" from "an EXPRESSION-SELECTED smaller set is
                                             better"
    7  degree-matched rewiring   CONTROL     configuration model: partner stubs permuted WITHIN each relation
                                             type, so every perturbation keeps its exact per-type degree and
                                             every partner gene keeps its exact multiplicity. Then gated in
                                             cell c exactly like arm 2. Uniform random destroys hub structure
                                             AS WELL AS wiring; this destroys only the wiring
    8  random partners           CONTROL     count-matched and type-mix-matched uniform draws, then gated in
                                             cell c. The blunt control that arm 7 refines
    9  wrong-perturbation set    CONTROL     IDENTITY control: arm 2's neighbour block taken from a DIFFERENT
                                             perturbation. Bounds how much of any gap is the neighbour block
                                             carrying perturbation identity rather than partner biology
    10 no neighbourhood                      the seven shared columns only. THE PRECONDITION ARM
    11 test-time neighbour swap  CONTROL     COUNTERFACTUAL on the trained arm-2 network: at test it is
                                             handed ANOTHER cell's whole neighbour block. Separates a
                                             neighbour channel that is IGNORED from one that is useless

THE UNIT OF REPLICATION IS A HELD-OUT CELL LINE, AND THERE ARE FOUR OF THEM. The claim under test is "a
cell-conditioned partner set generalises to a cell the model has never seen", so the error bar is computed
ACROSS CELLS. Each of the four lines is held out once; an arm's replicates are its four held-out-cell values;
every gap is formed PER CELL FIRST and then averaged, because pooling the arms and differencing the pooled
values is a different quantity and in `dense_context.py` that exact substitution hid a sign flip. Model-init
seeds inside one fold measure optimiser noise; rotating them in place of cells already flipped a verdict in
this project from 4/7 to 0/7. Seed spread IS computed here and is reported purely so the two floors can be
compared. NOTHING IS GRADED ON IT.

    MDE = 3 * sd / sqrt(n),  n = 4 HELD-OUT CELL LINES.
    For an ARM, sd is the across-cell sd of that arm's four held-out-cell values, pooled over arms (ddof=1).
    For a GAP, sd is the across-cell sd of the four PAIRED per-cell differences (ddof=1), which is the
    quantity every gate below is compared against.

    AT n = 4 THE MDE IS LARGE AND THIS MODULE IS UNDERPOWERED BY DESIGN, NOT BY ACCIDENT. Section 18 of
    CELLFORMER4.md already computed what that costs: a cross-cell MDE of 0.170 against a measured effect of
    +0.062 with a 95% CI of -0.118 to +0.242, and an estimate of ~30 cell contexts needed at 3 sigma. Four
    contexts also means the sd itself has THREE degrees of freedom, so the MDE is poorly estimated. A NULL
    HERE IS AN UNDERPOWERED NULL AND IS NOT A REFUTATION OF N_c(p). It is a statement that if the
    cell-conditioned partner set buys anything, it buys less than this design can see, and the binding
    constraint is the number of cell lines, which is not architectural and cannot be fixed from disk.

THE GATES, DECLARED BEFORE ANY NUMBER IS PRODUCED.

    G0  PRECONDITION  the neighbour channel must carry signal AT ALL: arm 1 beats arm 10 by > the paired
                      MDE. IF G0 FAILS, G1-G4 ARE COMPARISONS BETWEEN VARIANTS OF A CHANNEL THAT CONTRIBUTES
                      NOTHING, and no verdict about cell-conditioning can be drawn from them. This is stated
                      first because it is the most likely outcome given dense_context's numbers.
    G1  PRIMARY       arm 2 (ABSOLUTE percentile gating, the construction named in the brief) beats arm 1 by
                      > the paired MDE. Arm 3's differential gate is SECONDARY and is reported in full; if
                      arm 3 separates while arm 2 does not, that is a hypothesis for a future run and NOT a
                      passed gate, because arm 2 was fixed as primary before the run.
    G2  COUNTERFACTUAL arm 2 beats arm 5 (swapped-cell gating) by > the paired MDE. Without this a gain from
                      gating is not attributable to the cell.
    G3  RETENTION     arm 2 beats arm 6 (random gating at matched retention) by > the paired MDE. Without
                      this a gain from gating is a set-size effect.
    G4  WIRING        arm 2 beats arm 7 (degree-matched configuration-model rewiring) by > the paired MDE.
    G5  FREE          the best learned arm beats the best 0-parameter arm by > the pooled MDE. If a trivial
                      baseline wins, THAT IS THE HEADLINE and not a footnote.
    G6  EVERY-CELL    G1's gap is positive in ALL FOUR held-out cells. At n = 4 a pooled mean is nearly
                      uninformative; consistency of sign is the only thing four points can say cheaply.
    G7  AGREE         the sign-accuracy metric reaches the SAME verdict as Pearson r on G1. Disagreement
                      means metric artifact and the result is not reportable as biology.

    N_c(p) IS CREDITED ONLY IF G0, G1, G2 AND G3 ALL PASS. Effect sizes quoted anywhere in the output are
    RAW held-out values; the controls establish significance and attribution and no arm is ever reported net
    of a control. Where a control reproduces most of an arm's advantage, that is stated as the finding.

NO CIRCULARITY, CHECKED, AND ONE CONFOUND DECLARED. The partner sets come from `v4_relations.parquet`
(IntAct PPI, huMAP2/ComplexPortal co-membership, DepMap co-essentiality, UniBind ChIP-seq regulatory edges);
the gate comes from pooled non-targeting-control baseline expression; the scored target is Perturb-seq
response z per cell line. No feature is scored on the dataset it derives from, and every target-side
statistic (mu, the feature centring, the standardisation) is refit on the three TRAINING cells inside every
fold. THE CONFOUND, stated rather than hidden: the baseline profiles and the responses come from the SAME
four deposits, so a gene's expression level in cell c and the measurement noise on its z in cell c are not
independent. The gate acts on PARTNER genes of the perturbation while the target is at a response gene, so
this is not a direct leak, but a gating variable correlated with per-cell noise structure is a live
alternative explanation for any positive result and appears in `limits`.
"""
import gzip
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
DENSE = SP / "dense_response.npz"
BASE = OUT / "baseline_state.json.gz"
REL = OUT / "v4_relations.parquet"

# --- fixed by dense_context.py so the numbers stay on one axis, changed here for no reason whatsoever ---
ZCLIP = 10.0          # |z| clip: robust z blows up where a gene's MAD is near zero within a line
HID = 64              # MLP width
LR = 3e-3
WD = 1e-5
BATCH = 4096
# --- this module's own knobs -----------------------------------------------------------------------
N_SAMPLE = int(os.environ.get("V4C_N", "200000"))       # (p, g) pairs sampled per fold
EPOCHS = int(os.environ.get("V4C_EPOCHS", "8"))
N_SEEDS = int(os.environ.get("V4C_SEEDS", "2"))         # model-init seeds. REPORTED, NEVER GRADED ON
PCTL = float(os.environ.get("V4C_PCTL", "25"))          # the DECLARED baseline-expression percentile gate
MAX_PER_TYPE = int(os.environ.get("V4C_MAXTYPE", "8"))  # partners kept per relation type, by confidence
MAX_NB = int(os.environ.get("V4C_MAXNB", "40"))         # >= 5 types x MAX_PER_TYPE so no type is truncated
SMOKE = os.environ.get("V4C_SMOKE", "0") == "1"
if SMOKE:
    # A SMOKE RUN IS A PLUMBING CHECK AND NOT A RESULT. 20,000 sampled pairs and 2 epochs put every arm
    # inside its own noise; the arms cannot be separated at this size and the verdict string says so.
    # The full-scale settings are this module's defaults with V4C_SMOKE unset.
    N_SAMPLE, EPOCHS, N_SEEDS = 20000, 2, 1

TYPE_ID = {"codependency": 1, "complex": 2, "ppi": 3, "regulatory": 4, "signalling": 5}


def main():
    import pandas as pd
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)          # two full-scale training runs own most of this box's 4 cores
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 108)
    report("V4 SECTION 19 -- N_c(p), the CELL-CONDITIONED PARTNER SET (never tested)")
    report("=" * 108)
    report("  GATE, declared before the run: N_c(p) is credited only if G0 (the neighbour channel carries")
    report("  signal at all), G1 (gated beats cell-invariant), G2 (gated beats SWAPPED-cell gating) and G3")
    report("  (gated beats random gating at matched retention) all pass, each against a PAIRED MDE =")
    report(f"  3*sd/sqrt(n) with n = 4 HELD-OUT CELL LINES. At n = 4 the MDE is large: a null here is")
    report("  UNDERPOWERED, not a refutation. Model-init seeds are reported and are NOT graded on.")
    if SMOKE:
        report("\n  *** V4C_SMOKE=1 -- PLUMBING CHECK ONLY. Nothing below is a result. ***")

    for f_ in (DENSE, BASE, REL):
        if not f_.exists():
            raise SystemExit(f"absent: {f_}")

    # ------------------------------------------------------------------ the dense 4-cell target
    z = np.load(DENSE, allow_pickle=True)
    Z = z["Z"]                                    # (cells, perts, genes) float16, per-gene robust z per line
    lines = [str(x) for x in z["lines"]]
    perts = [str(x) for x in z["perts"]]
    genes = [str(x) for x in z["genes"]]
    C, P0, G0 = Z.shape
    report(f"\n  dense rectangle: {C} cells x {len(perts):,} perturbations x {len(genes):,} genes, "
           f"100% DENSE ({lines})")
    report(f"  NOT the nlz_* benches: those are 3.5% dense and their zeros mean NOT RECORDED, never a target")

    Zf = Z.astype(np.float32)
    bad_g = ~np.isfinite(Zf).all(0).all(0)
    n_clip = int((np.abs(np.nan_to_num(Zf, nan=0.0)) > ZCLIP).sum())
    maxz = float(np.nanmax(np.abs(Zf)))
    Zf = Zf[:, :, ~bad_g]
    genes = [g for g, b in zip(genes, bad_g) if not b]
    Z = np.clip(Zf, -ZCLIP, ZCLIP)
    del Zf
    report(f"  dropped {int(bad_g.sum()):,} genes with an undefined robust z; clipped values at |z| = "
           f"{ZCLIP:.0f} (max |z| was {maxz:.0f}, an artefact of a near-zero MAD, not a 357-sigma response)")

    prof = json.load(gzip.open(BASE, "rt"))["profiles"]
    keep_g = [i for i, g in enumerate(genes) if all(g in prof[ln] for ln in lines)]
    Z = Z[:, :, keep_g]
    genes = [genes[i] for i in keep_g]
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    P = len(perts)
    B = np.array([[prof[ln][g] for g in genes] for ln in lines], np.float32)      # (cells, genes) baseline
    report(f"  genes with a control-only baseline in all {C} lines: {G:,}")
    report(f"  baseline source: pooled NON-TARGETING CONTROL cells (response-free by construction); the "
           f"held-out cell contributes THIS and nothing else")

    pmiss = 0
    Bp = np.zeros((C, P), np.float32)
    for j, p in enumerate(perts):
        if p in gi:
            Bp[:, j] = B[:, gi[p]]
        else:
            pmiss += 1
            Bp[:, j] = np.median(B, axis=1)
    report(f"  perturbations whose target gene has no baseline here: {pmiss:,}/{P:,} (given the line median)")

    # ------------------------------------------------------------------ the partner sets, N(p)
    # Built ONCE. Every variant below is derived from this same list, so the arms are count-comparable and
    # type-mix-comparable by construction rather than by hope.
    cols = ["source", "target", "relation_type", "cell_context", "confidence"]
    df = pd.read_parquet(REL, columns=cols)
    pset, gset = set(perts), set(genes)
    a_ = df[df["source"].isin(pset) & df["target"].isin(gset)][cols]
    b_ = df[df["target"].isin(pset) & df["source"].isin(gset)].rename(
        columns={"source": "target", "target": "source"})[cols]
    E = pd.concat([a_, b_], ignore_index=True)
    E = E[E["source"] != E["target"]]
    del df, a_, b_
    ctx_pairs = {}
    for ln in lines:
        m = E["cell_context"].fillna("").str.contains(ln, regex=False)
        ctx_pairs[ln] = set(zip(E.loc[m, "source"], E.loc[m, "target"]))
    E["conf"] = E["confidence"].fillna(0.0)
    E = E.sort_values(["source", "relation_type", "conf"], ascending=[True, True, False])
    E = E.drop_duplicates(["source", "target", "relation_type"])
    E = E.groupby(["source", "relation_type"], sort=False).head(MAX_PER_TYPE)
    E = E.groupby("source", sort=False).head(MAX_NB)

    base_idx = [np.zeros(0, np.int64) for _ in range(P)]
    base_typ = [np.zeros(0, np.int64) for _ in range(P)]
    base_ctx = [np.zeros((C, 0), bool) for _ in range(P)]
    prow = {p: i for i, p in enumerate(perts)}
    acc = {}
    for s, t, rt, _cc, _cf, _c2 in E.itertuples(index=False):
        acc.setdefault(prow[s], []).append((gi[t], TYPE_ID.get(rt, 0),
                                            tuple((s, t) in ctx_pairs[ln] for ln in lines)))
    for i, v in acc.items():
        base_idx[i] = np.array([x[0] for x in v], np.int64)
        base_typ[i] = np.array([x[1] for x in v], np.int64)
        base_ctx[i] = np.array([x[2] for x in v], bool).T if v else np.zeros((C, 0), bool)
    deg = np.array([len(x) for x in base_idx])
    report(f"\n  N(p) from v4_relations.parquet, <= {MAX_PER_TYPE} partners per relation type, "
           f"<= {MAX_NB} total, ranked by edge confidence")
    report(f"    perturbations with >= 1 partner carrying a baseline: {int((deg > 0).sum()):,}/{P:,}; "
           f"degree median {np.median(deg):.0f}  q25 {np.percentile(deg, 25):.0f}  "
           f"q75 {np.percentile(deg, 75):.0f}  max {deg.max()}")
    tmix = {k: int(sum(int((t == v).sum()) for t in base_typ)) for k, v in TYPE_ID.items()}
    report(f"    type mix of the kept edges: " + "  ".join(f"{k} {v:,}" for k, v in tmix.items()))
    ctx_cov = {ln: int(sum(int(base_ctx[i][c].any()) for i in range(P))) for c, ln in enumerate(lines)}
    report(f"    perturbations with >= 1 partner whose edge names this line in `cell_context`: "
           + "  ".join(f"{k} {v:,}" for k, v in ctx_cov.items()))
    if min(ctx_cov.values()) == 0:
        report(f"    ^ ASYMMETRIC BY A LOT. For any line at 0 the evidence-preference arm IS the "
               f"cell-invariant arm by construction, so that arm is not comparable across cells and is "
               f"reported per cell for exactly that reason.")

    # degree-matched configuration model: permute the partner stubs WITHIN each relation type. Every
    # perturbation keeps its exact per-type degree and every partner gene keeps its exact multiplicity, so
    # only the WIRING moves. Uniform random (arm 8) flattens the partner-degree distribution too and
    # therefore cannot attribute a drop to wiring.
    rng_c = np.random.default_rng(2)
    cfg_idx = [np.zeros(len(base_idx[i]), np.int64) for i in range(P)]
    for t in sorted(set(TYPE_ID.values())):
        stubs = np.concatenate([base_idx[i][base_typ[i] == t] for i in range(P)]) if P else np.zeros(0, np.int64)
        if not len(stubs):
            continue
        perm = rng_c.permutation(len(stubs))
        pos = 0
        for i in range(P):
            sel = base_typ[i] == t
            k = int(sel.sum())
            if k:
                cfg_idx[i][sel] = stubs[perm[pos:pos + k]]
                pos += k
    rng_r = np.random.default_rng(3)
    rand_idx = [rng_r.integers(0, G, len(base_idx[i])).astype(np.int64) for i in range(P)]

    # ------------------------------------------------------------------ the gates that build N_c(p)
    thr_abs = np.percentile(B, PCTL, axis=1)                    # per-line absolute expression threshold
    oth_mean = np.stack([B[[k for k in range(C) if k != c]].mean(0) for c in range(C)])   # differential ref
    uni_mean = B.mean(1)                                        # fallback when a gate empties a set
    swap_of = [(c + 1) % C for c in range(C)]                    # fixed derangement for the counterfactual
    report(f"\n  THE GATE. N_c(p) = partners of p whose baseline expression in cell c clears the "
           f"{PCTL:.0f}th percentile")
    report(f"    per-line thresholds: " + "  ".join(f"{ln} {t:.2f}" for ln, t in zip(lines, thr_abs)))
    report(f"    swapped-cell gating maps " + ", ".join(f"{lines[c]}<-{lines[swap_of[c]]}" for c in range(C)))

    rng_g = np.random.default_rng(11)
    rand_keep = [[None] * P for _ in range(C)]                  # matched-retention random gate, drawn once

    def sel_inv(c, i):
        return base_idx[i]

    def sel_abs(c, i):
        v = base_idx[i]
        return v[B[c, v] >= thr_abs[c]] if len(v) else v

    def sel_diff(c, i):
        v = base_idx[i]
        return v[B[c, v] > oth_mean[c, v]] if len(v) else v

    def sel_swap(c, i):
        """The SET is chosen by another cell's expression; the FEATURES are still read in cell c."""
        v = base_idx[i]
        s = swap_of[c]
        return v[B[s, v] >= thr_abs[s]] if len(v) else v

    def sel_randgate(c, i):
        """A RANDOM subset of N(p) of exactly the size arm 2's gate left behind, per (perturbation, cell)."""
        if rand_keep[c][i] is None:
            v = base_idx[i]
            k = len(sel_abs(c, i))
            rand_keep[c][i] = (rng_g.permutation(len(v))[:k] if len(v) else np.zeros(0, np.int64))
        v = base_idx[i]
        return v[rand_keep[c][i]] if len(v) else v

    def sel_cfg(c, i):
        v = cfg_idx[i]
        return v[B[c, v] >= thr_abs[c]] if len(v) else v

    def sel_randp(c, i):
        v = rand_idx[i]
        return v[B[c, v] >= thr_abs[c]] if len(v) else v

    def sel_ctx(c, i):
        """Prefer partners whose edge `cell_context` names this line; fall back to N(p) when none exist."""
        v = base_idx[i]
        if not len(v):
            return v
        m = base_ctx[i][c]
        return v[m] if m.any() else v

    def summarise(sel):
        """(mean, sd, retention) of the arm's partner baselines in each cell. Depends only on (arm, c, p)."""
        V = np.zeros((C, P), np.float32)
        S = np.zeros((C, P), np.float32)
        R = np.zeros((C, P), np.float32)
        empty = 0
        for c in range(C):
            for i in range(P):
                idx = sel(c, i)
                n0 = max(len(base_idx[i]), 1)
                if not len(idx):
                    V[c, i], S[c, i], R[c, i] = uni_mean[c], 0.0, 0.0
                    empty += 1
                else:
                    v = B[c, idx]
                    V[c, i], S[c, i], R[c, i] = v.mean(), v.std(), len(idx) / n0
        return V, S, R, empty

    SEL = {"1 cell-invariant N(p)": sel_inv,
           "2 cell-gated N_c(p) ABSOLUTE": sel_abs,
           "3 cell-gated N_c(p) DIFFERENTIAL": sel_diff,
           "4 cell-context EVIDENCE preference": sel_ctx,
           "5 SWAPPED-cell gating (COUNTERFACTUAL)": sel_swap,
           "6 random gating, matched retention (CONTROL)": sel_randgate,
           "7 degree-matched rewiring (CONTROL)": sel_cfg,
           "8 random partners (CONTROL)": sel_randp}
    NBT = {}
    empties = {}
    for k, f_ in SEL.items():
        NBT[k], empties[k] = summarise(f_)[:3], summarise(f_)[3]
    # identity control: arm 2's neighbour block attached to the WRONG perturbation
    wperm = np.random.default_rng(17).permutation(P)
    NBT["9 wrong-perturbation set (identity CONTROL)"] = tuple(x[:, wperm] for x in NBT[
        "2 cell-gated N_c(p) ABSOLUTE"])
    empties["9 wrong-perturbation set (identity CONTROL)"] = empties["2 cell-gated N_c(p) ABSOLUTE"]

    def jaccard(sel):
        """How cell-specific is this partner set REALLY? A gate whose sets are identical across cells
        cannot test cell-conditioning no matter what the model does, and this is a property of the DATA."""
        vals, ret = [], []
        step = max(1, P // 400)
        for i in range(0, P, step):
            ss = [set(sel(c, i).tolist()) for c in range(C)]
            n0 = max(len(base_idx[i]), 1)
            ret.append(np.mean([len(s) / n0 for s in ss]))
            for x in range(C):
                for y in range(x + 1, C):
                    u = len(ss[x] | ss[y])
                    vals.append(len(ss[x] & ss[y]) / u if u else 1.0)
        return float(np.mean(vals)), float(np.mean(ret))

    report(f"\n  HOW CELL-SPECIFIC IS N_c(p) REALLY? (a property of the DATA, measured before any model)")
    jac = {}
    for k in ("1 cell-invariant N(p)", "2 cell-gated N_c(p) ABSOLUTE", "3 cell-gated N_c(p) DIFFERENTIAL",
              "4 cell-context EVIDENCE preference", "6 random gating, matched retention (CONTROL)"):
        j, r_ = jaccard(SEL[k])
        jac[k] = {"cross_cell_jaccard": j, "mean_retention": r_}
        report(f"    {k:46s} retention {r_:.3f}   cross-cell Jaccard {j:.3f}")
    report(f"    ^ the four baseline transcriptomes correlate at r 0.80-0.88, so an ABSOLUTE percentile gate "
           f"keeps nearly the same set in every cell. Only the DIFFERENTIAL gate makes N_c(p) genuinely "
           f"cell-specific, and that asymmetry is a limit of the data, not of the model.")
    report(f"    empty gated sets falling back to the line's gene-universe mean, per arm: "
           + "  ".join(f"{k.split()[0]}:{v}" for k, v in empties.items()))

    # ------------------------------------------------------------------------------------ the folds
    SHARED = 7          # xg, xp, dxg, dxp, mu, mu*dxg, mu*dxp   -- IDENTICAL in every arm
    NBC = 5             # nb, dnb, mu*dnb, nbsd, retention       -- THE ONLY THING THAT VARIES
    ARMS = list(NBT) + ["10 no neighbourhood"]

    def fold(h, seed):
        tr_c = [c for c in range(C) if c != h]
        rng = np.random.default_rng(1000 * seed + h)
        MU = Z[tr_c].mean(0)                                    # (perts, genes) shared response
        Bm, Bpm = B[tr_c].mean(0), Bp[tr_c].mean(0)             # training-cell references

        pi = rng.integers(0, P, N_SAMPLE)
        gj = rng.integers(0, G, N_SAMPLE)
        mu = MU[pi, gj]
        yte = Z[h, pi, gj] - mu

        def shared(c):
            xg, xp = B[c, gj], Bp[c, pi]
            dxg, dxp = xg - Bm[gj], xp - Bpm[pi]
            return np.stack([xg, xp, dxg, dxp, mu, mu * dxg, mu * dxp], 1)

        def nbblock(arm, c, src=None):
            """`src` is the cell whose neighbour block is READ. src != c is the test-time counterfactual."""
            V, S, R = NBT[arm]
            s = c if src is None else src
            nbm = V[tr_c].mean(0)
            nb = V[s, pi]
            dnb = nb - nbm[pi]
            return np.stack([nb, dnb, mu * dnb, S[s, pi], R[s, pi]], 1)

        SH = {c: shared(c) for c in range(C)}

        def X(arm, c, src=None):
            if arm == "10 no neighbourhood":
                return SH[c]
            return np.concatenate([SH[c], nbblock(arm, c, src)], 1)

        ytr = np.concatenate([Z[c, pi, gj] - mu for c in tr_c])
        assert abs(float(ytr.mean())) < 1e-3, "training delta must be ~zero-mean by construction"

        def fit_pred(arm, extra_test=None):
            Xtr = np.concatenate([X(arm, c) for c in tr_c])
            m_, s_ = Xtr.mean(0), Xtr.std(0)
            keep = s_ > 1e-6                       # a constant column (retention == 1 for arm 1) is dropped

            def std(Xa):
                return ((Xa - m_) / np.where(keep, s_, 1.0))[:, keep].astype(np.float32)
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(int(keep.sum()), HID), nn.ReLU(),
                                nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))
            opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
            Xn = torch.from_numpy(std(Xtr))
            Y = torch.from_numpy(ytr.astype(np.float32)).unsqueeze(1)
            n = len(Xn)
            for _e in range(EPOCHS):
                pm = torch.randperm(n)
                for i0 in range(0, n, BATCH):
                    b = pm[i0:i0 + BATCH]
                    loss = (net(Xn[b]) - Y[b]).pow(2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            def pr(Xa):
                with torch.no_grad():
                    return net(torch.from_numpy(std(Xa))).squeeze(1).numpy()
            out = {None: pr(X(arm, h))}
            if extra_test is not None:
                out["swap"] = pr(X(arm, h, src=extra_test))
            lin = None
            if arm == "2 cell-gated N_c(p) ABSOLUTE":
                Xa = np.c_[std(Xtr), np.ones(len(Xtr))]
                w = np.linalg.lstsq(Xa, ytr, rcond=None)[0]
                lin = np.c_[std(X(arm, h)), np.ones(N_SAMPLE)] @ w
            return out, lin

        big = np.abs(yte) >= np.median(np.abs(yte))

        def score(p_):
            """EVERY arm is scored on the IDENTICAL (pi, gj) pairs against the IDENTICAL yte. A baseline
            evaluated in a different universe than the truth scores ~0 by construction; that defect has
            shipped three times in this project and this assertion is what stops a fourth."""
            assert p_.shape == yte.shape
            r = float(np.corrcoef(p_, yte)[0, 1]) if np.std(p_) > 1e-9 else 0.0
            sg = float((np.sign(p_[big]) == np.sign(yte[big])).mean())
            return {"r": r, "sign": sg}

        out = {}
        # --- 0-parameter arms. Reported NEXT TO the model. In dense_context the nearest-cell copy WON.
        out["Z zero contrast (0 param)"] = {"r": 0.0, "sign": 0.5}
        near = tr_c[int(np.argmin([np.linalg.norm(B[h] - B[c]) for c in tr_c]))]
        out["NC nearest-cell copy (0 param)"] = score(Z[near, pi, gj] - mu)
        # --- learned arms
        lin_pred = None
        for arm in ARMS:
            extra = swap_of[h] if arm == "2 cell-gated N_c(p) ABSOLUTE" else None
            pd_, lin = fit_pred(arm, extra_test=extra)
            out[arm] = score(pd_[None])
            if lin is not None:
                lin_pred = lin
            if extra is not None:
                out["11 test-time neighbour-block SWAP (COUNTERFACTUAL)"] = score(pd_["swap"])
        out["LS least squares on arm 2's columns (0 param)"] = score(lin_pred)
        return out, lines[near]

    res, nearest = {}, {}
    for h, ln in enumerate(lines):
        report(f"\n  HELD-OUT CELL: {ln}  (train on {', '.join(l for i, l in enumerate(lines) if i != h)})")
        for seed in range(N_SEEDS):
            o, nr = fold(h, seed)
            nearest[ln] = nr
            for k, v in o.items():
                res.setdefault(ln, {}).setdefault(k, []).append(v)
        for k in res[ln]:
            v = res[ln][k]
            report(f"    {k:52s} r {np.mean([x['r'] for x in v]):+.4f}   "
                   f"sign {np.mean([x['sign'] for x in v]):.4f}")

    ALL = list(res[lines[0]])

    def per_cell(k, m="r"):
        return np.array([np.mean([x[m] for x in res[ln][k]]) for ln in lines])

    def agg(k, m="r"):
        return float(per_cell(k, m).mean())

    report(f"\n  {'arm':52s} " + " ".join(f"{ln:>8s}" for ln in lines) + f" {'mean':>8s} {'sd':>7s} {'sign':>7s}")
    for k in ALL:
        v = per_cell(k)
        report(f"  {k:52s} " + " ".join(f"{x:+8.4f}" for x in v) +
               f" {v.mean():+8.4f} {np.std(v, ddof=1):7.4f} {agg(k, 'sign'):7.4f}")

    # ------------------------------------------------------------------------------------ the MDE
    # n COUNTS HELD-OUT CELL LINES. Gaps are formed PER CELL FIRST and then averaged: pooling the arms and
    # differencing the pooled values is a different quantity, and in dense_context.py that substitution hid
    # a sign flip (+0.0198 pooled where the mean of the per-cell gaps was negative).
    def mde_of(vec):
        return 3 * float(np.std(np.asarray(vec), ddof=1)) / np.sqrt(C)
    sd_pool = float(np.mean([np.std(per_cell(k), ddof=1) for k in ALL]))
    MDE = 3 * sd_pool / np.sqrt(C)
    sd_pool_s = float(np.mean([np.std(per_cell(k, "sign"), ddof=1) for k in ALL]))
    MDE_S = 3 * sd_pool_s / np.sqrt(C)
    sd_seed = float(np.mean([np.std([x["r"] for x in res[ln][k]]) for ln in lines for k in ALL]))
    mde_seed = 3 * sd_seed / np.sqrt(max(N_SEEDS, 2))

    def gap(a, b, m="r"):
        v = per_cell(a, m) - per_cell(b, m)
        return float(v.mean()), mde_of(v), v

    A1, A2, A3 = "1 cell-invariant N(p)", "2 cell-gated N_c(p) ABSOLUTE", "3 cell-gated N_c(p) DIFFERENTIAL"
    A4, A5 = "4 cell-context EVIDENCE preference", "5 SWAPPED-cell gating (COUNTERFACTUAL)"
    A6, A7 = "6 random gating, matched retention (CONTROL)", "7 degree-matched rewiring (CONTROL)"
    A8, A9 = "8 random partners (CONTROL)", "9 wrong-perturbation set (identity CONTROL)"
    A10, A11 = "10 no neighbourhood", "11 test-time neighbour-block SWAP (COUNTERFACTUAL)"
    FREE = ["Z zero contrast (0 param)", "NC nearest-cell copy (0 param)",
            "LS least squares on arm 2's columns (0 param)"]
    LEARNED = [A1, A2, A3, A4, A10]

    g_pre, m_pre, v_pre = gap(A1, A10)
    g_pri, m_pri, v_pri = gap(A2, A1)
    g_pri_s, m_pri_s, _ = gap(A2, A1, "sign")
    g_sec, m_sec, v_sec = gap(A3, A1)
    g_ctx, m_ctx, v_ctx = gap(A4, A1)
    g_cf, m_cf, v_cf = gap(A2, A5)
    g_ret, m_ret, v_ret = gap(A2, A6)
    g_wir, m_wir, v_wir = gap(A2, A7)
    g_rnd, m_rnd, _ = gap(A2, A8)
    g_idt, m_idt, _ = gap(A2, A9)
    g_nbs, m_nbs, _ = gap(A2, A11)
    best_learned = max(LEARNED, key=lambda k: agg(k))
    best_free = max(FREE, key=lambda k: agg(k))
    g_free, m_free, _ = gap(best_learned, best_free)

    report(f"\n  MDE = 3*sd/sqrt(n), n = {C} HELD-OUT CELL LINES (the unit of replication -- the claim is")
    report(f"  generalisation to a cell never seen). Pooled across-cell sd {sd_pool:.4f} -> arm-level MDE "
           f"{MDE:+.4f}; sign MDE {MDE_S:+.4f}.")
    report(f"  Every GAP below carries its OWN paired MDE from the four per-cell differences.")
    report(f"  For comparison only: the model-init-seed sd inside a fold is {sd_seed:.4f} (a {mde_seed:+.4f} "
           f"floor), {sd_pool/max(sd_seed,1e-9):.1f}x smaller than the across-cell sd. Seeds measure "
           f"optimiser noise inside a FIXED cell set and NOTHING HERE IS GRADED ON THEM.")
    report(f"  AT n = 4 THIS MODULE IS UNDERPOWERED BY DESIGN: the sd has 3 degrees of freedom, so the MDE "
           f"is itself poorly estimated and a null is a null of POWER, not a refutation of N_c(p).")

    gates = {
        "G0 PRECONDITION  the neighbour channel carries signal at all (arm 1 > arm 10)": bool(g_pre > m_pre),
        "G1 PRIMARY       cell-gated N_c(p) beats cell-invariant N(p)": bool(g_pri > m_pri),
        "G2 COUNTERFACTUAL gated beats SWAPPED-cell gating": bool(g_cf > m_cf),
        "G3 RETENTION     gated beats random gating at matched retention": bool(g_ret > m_ret),
        "G4 WIRING        gated beats degree-matched configuration-model rewiring": bool(g_wir > m_wir),
        "G5 FREE          the best learned arm beats the best 0-parameter arm": bool(g_free > m_free),
        "G6 EVERY-CELL    the primary gap is positive in ALL four held-out cells": bool((v_pri > 0).all()),
        "G7 AGREE         sign accuracy reaches the same verdict as r on the primary gap":
            bool((g_pri > m_pri) == (g_pri_s > m_pri_s)),
    }
    report("\n  PREDECLARED GATES")
    for k, v in gates.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k}")
    report(f"\n  {'gap (formed PER CELL, then averaged)':56s} {'mean':>9s} {'paired MDE':>11s}  per-cell")
    for nm, (gv, mv, vv) in (
            ("G0  arm 1 cell-invariant  -  arm 10 no neighbourhood", (g_pre, m_pre, v_pre)),
            ("G1  arm 2 ABSOLUTE gate   -  arm 1 cell-invariant", (g_pri, m_pri, v_pri)),
            ("    arm 3 DIFFERENTIAL gate - arm 1 (SECONDARY, declared)", (g_sec, m_sec, v_sec)),
            ("    arm 4 cell-context evidence - arm 1 (asymmetric)", (g_ctx, m_ctx, v_ctx)),
            ("G2  arm 2  -  arm 5 SWAPPED-cell gating", (g_cf, m_cf, v_cf)),
            ("G3  arm 2  -  arm 6 random gating, matched retention", (g_ret, m_ret, v_ret)),
            ("G4  arm 2  -  arm 7 degree-matched rewiring", (g_wir, m_wir, v_wir))):
        report(f"  {nm:56s} {gv:+9.4f} {mv:11.4f}  {np.round(vv, 4)}")
    report(f"  {'    arm 2  -  arm 8 random partners':56s} {g_rnd:+9.4f} {m_rnd:11.4f}")
    report(f"  {'    arm 2  -  arm 9 wrong-perturbation (identity)':56s} {g_idt:+9.4f} {m_idt:11.4f}")
    report(f"  {'    arm 2  -  arm 11 test-time neighbour-block swap':56s} {g_nbs:+9.4f} {m_nbs:11.4f}")
    report(f"  {'G5  best learned  -  best 0-parameter arm':56s} {g_free:+9.4f} {m_free:11.4f}")
    report(f"    best learned = '{best_learned}' at r {agg(best_learned):+.4f}; "
           f"best 0-parameter = '{best_free}' at r {agg(best_free):+.4f}")
    report(f"    nearest training cell by baseline transcriptome, per held-out cell: "
           + ", ".join(f"{k}<-{v}" for k, v in nearest.items()))

    # Rule: A CONTROL THAT REPRODUCES MOST OF THE EFFECT IS THE FINDING, not a footnote.
    def frac(x, whole):
        return (x / whole) if abs(whole) > 1e-9 else float("nan")
    repro = {
        "arm 5 SWAPPED-cell gating reproduces this fraction of arm 2's gain over cell-invariant N(p)":
            frac(agg(A5) - agg(A1), g_pri),
        "arm 6 random gating at matched retention reproduces this fraction of the same gain":
            frac(agg(A6) - agg(A1), g_pri),
        "arm 7 degree-matched rewiring reproduces this fraction of the same gain":
            frac(agg(A7) - agg(A1), g_pri),
        "arm 10 NO NEIGHBOURHOOD AT ALL reproduces this fraction of arm 2's r":
            frac(agg(A10), agg(A2)),
        "the 0-parameter nearest-cell copy reproduces this fraction of arm 2's r":
            frac(agg("NC nearest-cell copy (0 param)"), agg(A2))}
    report("\n  CONTROLS THAT REPRODUCE THE EFFECT (reported as the finding, not as a footnote)")
    for k, v in repro.items():
        report(f"    {100*v:8.1f}%  {k}")
    repro_meaningful = bool(g_pri > m_pri)
    if not repro_meaningful:
        report(f"    ^ the primary gap is {g_pri:+.4f} against a {m_pri:.4f} paired MDE, i.e. undetected, so "
               f"the first three percentages are ratios of two undetected quantities and mean nothing. They "
               f"are printed rather than suppressed so the denominator stays visible.")

    credited = all(gates[k] for k in list(gates)[:4])
    npass = int(sum(bool(v) for v in gates.values()))
    verdict = (
        (f"SMOKE RUN ({N_SAMPLE:,} sampled (perturbation, gene) pairs per fold, {EPOCHS} epochs, "
         f"{N_SEEDS} seed) -- THIS IS A PLUMBING CHECK AND NOT A RESULT. At this sample size every arm sits "
         f"inside its own noise, the arms cannot be separated, the gate outcomes below are arbitrary, and no "
         f"sentence in this paragraph should be quoted as a finding. Rerun with V4C_SMOKE unset. " if SMOKE
         else "") +
        f"{'N_c(p) IS CREDITED' if credited else 'N_c(p) IS NOT CREDITED'} ({npass}/8 predeclared gates): on "
        f"a 100%-dense {C}-cell x {P:,}-perturbation x {G:,}-gene rectangle, with the shared response frozen "
        f"out by construction so that no cell-blind feature can beat zero, and with the seven-column shared "
        f"block held IDENTICAL across arms so that ONLY THE MEMBERSHIP OF THE PARTNER SET VARIES, the "
        f"cell-gated partner set N_c(p) reaches r {agg(A2):+.4f} against the cell-invariant N(p)'s "
        f"{agg(A1):+.4f} -- a paired per-cell gap of {g_pri:+.4f} against a cross-cell minimum detectable "
        f"increment of {m_pri:.4f} (3*sd/sqrt(n), n = {C} HELD-OUT CELL LINES; the model-init-seed floor of "
        f"{mde_seed:.4f} measures optimiser noise inside a fixed cell set and is deliberately NOT what this "
        f"is graded on). THE PRECONDITION COMES FIRST: the cell-invariant neighbour block beats having no "
        f"neighbourhood at all by {g_pre:+.4f} against a {m_pre:.4f} floor, so the neighbour channel "
        f"{'does carry signal and the gating comparison is interpretable' if gates[list(gates)[0]] else 'IS NOT DETECTED AS CARRYING ANY SIGNAL, which means every gating comparison below is a comparison between variants of a channel that contributes nothing and NO VERDICT ABOUT CELL-CONDITIONING CAN BE DRAWN FROM THEM'}. "
        f"THE COUNTERFACTUAL, which accuracy alone cannot supply: gating the set with ANOTHER cell's "
        f"expression while still reading features in the correct cell changes r by {g_cf:+.4f} against a "
        f"{m_cf:.4f} floor, and handing the trained network another cell's entire neighbour block at test "
        f"time changes it by {g_nbs:+.4f} -- a gap at or below the floor means the cell-conditioned partner "
        f"set is IGNORED rather than useless, and those two are not distinguishable by accuracy. Matched "
        f"controls: random gating at the same retention rate {g_ret:+.4f}, a degree-matched "
        f"configuration-model rewiring that preserves every per-type degree and every partner multiplicity "
        f"{g_wir:+.4f}, count- and type-matched random partners {g_rnd:+.4f}, and the wrong-perturbation "
        f"identity control {g_idt:+.4f}. THE CHEAPEST ARMS ARE REPORTED NEXT TO THE MODEL: the best "
        f"0-parameter arm is '{best_free}' at r {agg(best_free):+.4f} against the best learned arm "
        f"'{best_learned}' at {agg(best_learned):+.4f}, a gap of {g_free:+.4f} against a {m_free:.4f} floor "
        f"-- {'the learned arm is ahead of it' if gates[list(gates)[5]] else 'A ZERO-PARAMETER BASELINE MATCHES OR BEATS EVERY LEARNED ARM, WHICH IS THE HEADLINE AND NOT A FOOTNOTE'}. "
        f"A STRUCTURAL FACT ABOUT THE DATA THAT BOUNDS WHAT THIS COULD EVER HAVE SHOWN, measured before any "
        f"model was fit: the four baseline transcriptomes correlate at r 0.80-0.88, so the ABSOLUTE "
        f"percentile gate named in the specification keeps sets whose mean cross-cell Jaccard is "
        f"{jac[A2]['cross_cell_jaccard']:.3f} at retention {jac[A2]['mean_retention']:.3f} -- N_c(p) under "
        f"that gate is barely a function of c. The DIFFERENTIAL gate, declared before the run as SECONDARY, "
        f"reaches a Jaccard of {jac[A3]['cross_cell_jaccard']:.3f} and is the only construction here whose "
        f"sets genuinely differ between cells; its gap over cell-invariant N(p) is {g_sec:+.4f} against a "
        f"{m_sec:.4f} floor, reported as a hypothesis for a future run and NOT as a passed gate, because the "
        f"primary was fixed in advance. The cell_context evidence route is unusable as a cross-cell "
        f"comparison and is reported per cell only: K562 {ctx_cov.get('K562', 0):,} perturbations carry a "
        f"line-matched edge, HepG2 {ctx_cov.get('HepG2', 0):,}, Jurkat {ctx_cov.get('Jurkat', 0):,} and RPE1 "
        f"{ctx_cov.get('RPE1', 0):,}, so for RPE1 that arm IS the cell-invariant arm by construction. "
        f"FINALLY AND DECISIVELY FOR HOW THIS SHOULD BE READ: n = 4 cell lines gives the cross-cell sd three "
        f"degrees of freedom and an arm-level MDE of {MDE:.4f}. CELLFORMER4.md section 18 already estimated "
        f"that roughly 30 cell contexts are needed to resolve an effect of the size this branch plausibly "
        f"has. A NULL HERE IS THEREFORE UNDERPOWERED AND IS NOT A REFUTATION OF N_c(p); it says that if the "
        f"cell-conditioned partner set buys anything it buys less than four cells can see, and the binding "
        f"constraint is the number of independent cell contexts, which cannot be recovered from disk. All "
        f"effect sizes quoted are RAW held-out values; the controls establish significance and attribution "
        f"and no arm is reported net of a control.")
    report("\n" + "=" * 108)
    report(f"  VERDICT: {verdict}")

    R = {
        "model": "v4-context-neighbours-v1",
        "question": "CELLFORMER4.md sec 19 -- F_context takes N_c(p), a partner set that DEPENDS ON THE "
                    "CELL. Every prior context test used cell-conditioned NODE FEATURES over a "
                    "cell-INVARIANT neighbour set and returned 0/7. Does choosing DIFFERENT PARTNERS per "
                    "cell help? Never tested.",
        "task": {"source": "colab/dense_context.py target and decomposition, unchanged",
                 "data": "dense_response.npz -- 4 cells x 1,763 perturbations x 6,550 genes, 100% DENSE "
                         "(NOT the nlz_* benches, which are 3.5% dense and whose zeros mean NOT RECORDED)",
                 "target": "delta = Y - mean over the THREE TRAINING cells; zero-mean by construction, so "
                           "no cell-blind feature can beat 0",
                 "primary_metric": "Pearson r between predicted and true held-out-cell delta",
                 "secondary_metric": "sign accuracy on the large-|delta| half",
                 "held_out_cell_inputs": "baseline transcriptome only (pooled non-targeting controls); its "
                                         "responses are never an input, never fit mu, never standardise",
                 "n_cells": C, "n_perturbations": P, "n_genes": G,
                 "sampled_pairs_per_fold": N_SAMPLE,
                 "only_thing_that_varies": "which partners enter the 5-column neighbour block; the 7-column "
                                           "shared block is byte-identical across arms"},
        "partner_sets": {"source": "outputs/orphan/v4_relations.parquet",
                         "max_per_relation_type": MAX_PER_TYPE, "max_total": MAX_NB,
                         "perturbations_with_partners": int((deg > 0).sum()),
                         "degree_median": float(np.median(deg)), "type_mix": tmix,
                         "gate": f"drop partners below the {PCTL:.0f}th percentile of baseline expression "
                                 f"in cell c",
                         "gate_thresholds_per_line": {ln: float(t) for ln, t in zip(lines, thr_abs)},
                         "swap_map": {lines[c]: lines[swap_of[c]] for c in range(C)},
                         "cell_context_coverage_perturbations": ctx_cov,
                         "empty_gated_sets_per_arm": {k: int(v) for k, v in empties.items()}},
        "how_cell_specific_is_the_set": jac,
        "how_cell_specific_is_the_set_note":
            "cross-cell Jaccard measured on the DATA before any model. The ABSOLUTE percentile gate named "
            "in the specification produces sets that are ~85% identical across the four lines, because the "
            "four baseline transcriptomes correlate at r 0.80-0.88. That bounds what any model could have "
            "extracted from this construction and is the single most important number in this file.",
        "arms": {k: {"r_mean": float(per_cell(k).mean()),
                     "r_sd_across_cells": float(np.std(per_cell(k), ddof=1)),
                     "r_per_cell": {ln: float(x) for ln, x in zip(lines, per_cell(k))},
                     "sign_mean": float(agg(k, "sign")),
                     "sign_sd_across_cells": float(np.std(per_cell(k, "sign"), ddof=1)),
                     "sign_per_cell": {ln: float(x) for ln, x in zip(lines, per_cell(k, "sign"))}}
                 for k in ALL},
        "unit_of_replication": f"a HELD-OUT CELL LINE. The per-cell dicts above hold one value per held-out "
                               f"cell (n = {C}), NOT model-init seeds. Gaps are formed per cell FIRST and "
                               f"then averaged, because pooling arms and differencing the pooled values is "
                               f"a different quantity and hid a sign flip in dense_context.py.",
        "mde": {"value": float(MDE), "convention": "3*sd/sqrt(n)", "n": C,
                "n_counts": "held-out cell lines",
                "pooled_across_cell_sd": float(sd_pool),
                "sign_mde": float(MDE_S), "pooled_across_cell_sd_sign": float(sd_pool_s),
                "paired_mde_per_gap": {"G0_invariant_minus_none": float(m_pre),
                                       "G1_gated_minus_invariant": float(m_pri),
                                       "differential_minus_invariant": float(m_sec),
                                       "cellcontext_minus_invariant": float(m_ctx),
                                       "G2_gated_minus_swapped_cell": float(m_cf),
                                       "G3_gated_minus_random_gating": float(m_ret),
                                       "G4_gated_minus_degree_matched": float(m_wir),
                                       "gated_minus_random_partners": float(m_rnd),
                                       "gated_minus_wrong_perturbation": float(m_idt),
                                       "gated_minus_testtime_nb_swap": float(m_nbs),
                                       "G5_best_learned_minus_best_free": float(m_free)},
                "init_seed_sd_within_fold_DO_NOT_GRADE_ON_THIS": float(sd_seed),
                "init_seed_mde_DO_NOT_GRADE_ON_THIS": float(mde_seed),
                "power_note": f"n = {C} gives the cross-cell sd 3 degrees of freedom, so the MDE is itself "
                              f"poorly estimated. CELLFORMER4.md sec 18 estimates ~30 contexts are needed "
                              f"at 3 sigma for an effect of this size. A NULL HERE IS UNDERPOWERED AND IS "
                              f"NOT A REFUTATION."},
        "gates": {k: bool(v) for k, v in gates.items()},
        "gates_passed": npass,
        "n_c_p_credited": bool(credited),
        "credit_rule": "N_c(p) is credited only if G0, G1, G2 and G3 all pass",
        "gaps": {"G0_invariant_minus_no_neighbourhood": float(g_pre),
                 "G1_gated_minus_invariant_PRIMARY": float(g_pri),
                 "G1_per_cell": {ln: float(x) for ln, x in zip(lines, v_pri)},
                 "differential_minus_invariant_SECONDARY": float(g_sec),
                 "differential_per_cell": {ln: float(x) for ln, x in zip(lines, v_sec)},
                 "cellcontext_minus_invariant_ASYMMETRIC": float(g_ctx),
                 "cellcontext_per_cell": {ln: float(x) for ln, x in zip(lines, v_ctx)},
                 "G2_gated_minus_swapped_cell_gating": float(g_cf),
                 "G3_gated_minus_random_gating_matched_retention": float(g_ret),
                 "G4_gated_minus_degree_matched_rewiring": float(g_wir),
                 "gated_minus_random_partners": float(g_rnd),
                 "gated_minus_wrong_perturbation_identity": float(g_idt),
                 "gated_minus_testtime_neighbour_block_swap": float(g_nbs),
                 "G5_best_learned_minus_best_zero_parameter": float(g_free),
                 "best_learned_arm": best_learned, "best_zero_parameter_arm": best_free},
        "controls_reproducing_the_effect": {k: (float(v) if np.isfinite(v) else None)
                                            for k, v in repro.items()},
        "controls_reproducing_the_effect_are_meaningful": repro_meaningful,
        "controls_reproducing_the_effect_note":
            "fractions of arm 2's raw gain over cell-invariant N(p). If that denominator is itself below "
            "the paired MDE the ratios are meaningless and the flag above is false.",
        "nearest_training_cell_by_baseline": nearest,
        "circularity_check":
            "partner sets come from v4_relations.parquet (IntAct PPI, huMAP2/ComplexPortal co-membership, "
            "DepMap co-essentiality, UniBind ChIP-seq); the gate comes from pooled non-targeting-control "
            "baseline expression; the scored target is Perturb-seq response z per line. No feature is "
            "scored on the dataset it derives from. mu, the feature centring and the standardisation are "
            "refit on the three TRAINING cells inside every fold. DECLARED CONFOUND: baselines and "
            "responses come from the SAME four deposits, so expression level and per-cell measurement "
            "noise are not independent -- see limits.",
        "config": {"pctl": PCTL, "epochs": EPOCHS, "seeds": N_SEEDS, "n_sample": N_SAMPLE,
                   "max_per_type": MAX_PER_TYPE, "max_nb": MAX_NB, "zclip": ZCLIP},
        "smoke": bool(SMOKE),
        "limits": [
            f"n = {C} CELL LINES. This is the binding constraint and it is not architectural. The cross-cell "
            f"sd has 3 degrees of freedom, the arm-level MDE is {MDE:.4f}, and CELLFORMER4.md sec 18 puts "
            f"the contexts needed at ~30. A NULL HERE IS UNDERPOWERED AND DOES NOT REFUTE N_c(p). What "
            f"would falsify the null is more independent cell contexts, not more perturbations or genes",
            "WHAT WOULD FALSIFY A POSITIVE RESULT: if arm 5 (swapped-cell gating) or arm 6 (random gating "
            "at matched retention) reproduces the gap, the gain is a set-size or set-sparsification effect "
            "and has nothing to do with the cell. Both are run and both are reported next to the primary",
            f"THE GATE NAMED IN THE SPECIFICATION BARELY CONDITIONS ON THE CELL. Cross-cell Jaccard of the "
            f"absolute-percentile sets is {jac[A2]['cross_cell_jaccard']:.3f}: the four baseline "
            f"transcriptomes correlate at r 0.80-0.88 so 'expressed here' is very nearly 'expressed "
            f"everywhere'. A null on arm 2 is therefore partly a statement about the CONSTRUCTION, not "
            f"about the hypothesis. Arm 3's differential gate (Jaccard "
            f"{jac[A3]['cross_cell_jaccard']:.3f}) is the version that actually varies and was declared "
            f"secondary in advance; a positive there needs its own confirmatory run",
            "CONFOUNDED: baseline profiles and response z-scores come from the SAME four Perturb-seq "
            "deposits, so a gene's expression level in cell c and the noise on its z in cell c are not "
            "independent. The gate acts on PARTNER genes while the target is at a response gene, so this "
            "is not a direct leak, but a gating variable correlated with per-cell noise structure remains "
            "a live alternative explanation for any positive result",
            "the cell_context evidence route (arm 4) is not comparable across cells: K562 and HepG2 have "
            "hundreds to a thousand line-matched perturbations, Jurkat a few hundred, RPE1 exactly zero. "
            "For RPE1 arm 4 IS arm 1 by construction, so its cross-cell mean mixes two different "
            "estimands and only its per-cell values should be read",
            "context is BASELINE RNA only. Chromatin accessibility and protein state are absent for these "
            "four lines, so a null is a null for RNA-defined partner selection, not for cell state",
            "the held-out cell's delta is measured against three cells that exclude it, so it carries a "
            "systematic offset no model trained on the other three can learn. That caps the achievable r "
            "for every arm equally and is why absolute r values here are small",
            "the perturbation set is the ~1,763 knockouts screened in all four lines, a biased subset: "
            "perturbations chosen to be screened everywhere are not a random sample of perturbations",
            "K562 is not re-derived while the other three are pseudobulked in dense_response.py, so the "
            "processing asymmetry lies along the cell axis, which is the axis being measured",
            "the neighbour block is five summary scalars (mean, deviation, mu-interaction, spread, "
            "retention), not a learned set encoder. A richer aggregator could read structure these five "
            "numbers discard; what is tested here is whether CHANGING THE MEMBERSHIP of the set moves a "
            "fixed aggregator, which is the cheapest honest version of the question and not the only one",
            f"the gate percentile is a free parameter fixed at {PCTL:.0f} before the run and not tuned. "
            f"Sweeping it would be a multiple-comparison problem unless the sweep is itself predeclared "
            f"and the MDE adjusted; it was not swept here"],
        "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "v4_context_neighbours.json", "w"), indent=1)
    report(f"\n  -> {OUT/'v4_context_neighbours.json'}")


if __name__ == "__main__":
    main()
