"""V4 BLOCK 8 -- the VIABILITY head against its own marginal, and here n is LARGE.

THE QUESTION. `CELLFORMER4.md` section 11 gives Block 8 a viability head supervised by DepMap
`CRISPRGeneEffect` and states that "codependency belongs in the viability head, not in generic
transcription prediction". Section 17 criterion 3 is "every head beats its marginal by more than its MDE",
and section 18 records that criterion as IN PROGRESS with the only numbers so far sitting AT or BELOW the
marginal on the transcription heads. This module runs criterion 3 for the viability head, on the one
endpoint where this project has an enormous number of independent replicates and has never used them.

    Predict the CRISPR gene effect of a gene in a CELL LINE THE MODEL HAS NEVER SEEN, from that line's
    features, the gene's identity/features, and the gene's co-dependency partners.

WHY THE MARGINAL IS THE WHOLE STORY HERE, STATED BEFORE THE RUN SO IT CANNOT BE RETROFITTED. DepMap gene
effect is dominated by pan-essentiality: a ribosomal subunit kills nearly every line and an olfactory
receptor kills none, and that ordering is almost identical in every cell line on earth. A model that has
learned nothing except "which genes are essential in general" will therefore look extremely accurate. The
prediction made here in advance is that MARGINAL-gene is very strong, that MARGINAL-both is close to
whatever the model achieves, and that the honest quantity is the LINE-SPECIFIC RESIDUAL that survives both
marginals. If the pan-essential marginal wins, that is the result and it is the first sentence of the
verdict, not a footnote.

THE TASK, PRECISELY. Cell lines are split into FOLDS disjoint blocks; each block is held out once, so every
line in the file is a held-out line exactly once. Genes are split ONCE, by a data-independent seeded
permutation, into two DISJOINT panels:

    EVAL panel   the genes whose effect is predicted in the held-out line. Never revealed.
    PROBE panel  the genes whose effect IS revealed for a held-out line. This is what makes the task
                 well-posed: a cell line with no measurements at all carries no line-specific information
                 whatsoever, and every arm would collapse onto the gene marginal by construction. The probe
                 panel is the cold-start "we screened a few hundred genes in this new line" setting.

Because EVAL and PROBE are disjoint, no cell of the matrix that is scored is ever an input to any arm.
That disjointness is asserted at runtime.

THE PARAMETERISATION. Everything is fit on TRAINING LINES ONLY and refit inside every fold.
    mu    global mean gene effect over training lines
    a_g   gene marginal, mean over TRAINING lines minus mu.  THE PAN-ESSENTIAL PRIOR
    b_c   line marginal, mean over that line's PROBE residuals. Available for a held-out line because
          probe genes are revealed; uses no eval gene
    R     residual, R[c,g] = E[c,g] - mu - a_g - b_c. This is the only thing any model can add

THE ARMS. Every arm emits a prediction for the SAME (held-out line, eval gene) cells and is scored by the
SAME metric on the SAME matrix. There is no arm ranked by a quantity the truth excludes.

    0-param  GLOBAL-mean               mu. The absolute floor
    0-param  MARGINAL-gene             mu + a_g. THE PAN-ESSENTIAL PRIOR, expected to be very strong
    0-param  MARGINAL-line             mu + b_c. Constant across genes within a line
    0-param  MARGINAL-both             mu + a_g + b_c. Additive. THE GATE'S REFERENCE ARM
    0-param  PARTNER-MEAN              MARGINAL-both + a correlation-signed mean of the held-out line's
                                       PROBE residuals at the gene's co-dependency partners. NO training,
                                       NO gradient, NO fitted parameter. THE CHEAPEST SENSIBLE NON-LEARNED
                                       BASELINE, reported next to the model because on the block already
                                       run in this project a zero-parameter untrained baseline beat the
                                       trained transformer
    1-param  PARTNER-MEAN (shrunk)     the same channel with a single scalar fit by least squares on the
                                       TRAINING lines. One parameter, closed form, no gradient
    MODEL                              MLP on [gene embedding, line embedding, their product, partner
                                       channel, gene sd, gene marginal] -> residual
    ABLATION no-partner                the identical network with the partner channel zeroed. Same input
                                       width, same parameter count, so the comparison is not a capacity
                                       comparison
    CONTROL  random partners           MATCHED ON COUNT: k partners per gene drawn uniformly from the probe
                                       panel, weights recomputed by the identical training-line procedure
    CONTROL  degree-matched rewiring   CONFIGURATION MODEL: the partner stubs are permuted, so every eval
                                       gene keeps its exact degree AND every probe gene keeps its exact
                                       multiplicity as a partner. Uniform random ALSO flattens the
                                       partner-usage distribution, so it destroys hub structure and wiring
                                       at once and cannot attribute a drop to the wiring
    CONTROL  0-param partner, random   the zero-parameter arm's own random-partner control
    COUNTERFACTUAL partner swap        the TRAINED network is handed ANOTHER HELD-OUT LINE'S partner-channel
                                       values while keeping the correct gene features and the correct line
                                       embedding. Accuracy alone cannot distinguish "the partner channel is
                                       useless" from "the partner channel is IGNORED"; only this can
    CONTROL  wrong-line (identity)     the model's residual predictions permuted across held-out lines
    CONTROL  wrong-gene (identity)     the model's residual predictions permuted across eval genes

THE UNIT OF REPLICATION IS A HELD-OUT CELL LINE, AND THERE ARE ABOUT ELEVEN HUNDRED OF THEM. The claim under
test is "this head generalises to a cell line it has never seen", so the error bar is computed ACROSS CELL
LINES. Rotating model-init seeds inside one partition measures optimiser noise instead, and in this project
that exact substitution already flipped a verdict from 4/7 to 0/7 (`dense_context.py`). Init seeds ARE run
here, on fold 0 only, and are REPORTED AND NOT GRADED so the two floors can be compared.

    MDE = 3 * sd / sqrt(n),  n = THE NUMBER OF HELD-OUT CELL LINES (every line in the file, each held out
    exactly once). Two sds are computed and both are reported:

    PAIRED      sd of the PER-LINE DIFFERENCE between the two arms being compared. Arms are evaluated on
                the identical held-out lines and the identical genes, so the comparison is paired and this
                is the correct error bar for a gap. This is what the gates use.
    UNPAIRED    sd of a single arm's per-line score, pooled over arms. Always larger, because it contains
                the between-line difficulty variance that cancels in a paired comparison. Reported next to
                every gate so a gate that only passes under pairing is visible as such.
    FOLD        sd of the per-fold mean gap over the FOLDS folds, with n = FOLDS. A third, much coarser
                bar, reported so that a gap driven by one fold cannot hide.

THE GATES, DECLARED BEFORE ANY NUMBER IS PRODUCED.

    PRIMARY   The viability head is credited only if MODEL beats MARGINAL-both on held-out-line RMSE by
              more than the paired across-line MDE. Criterion 3 of section 17, for this head.
    HONEST    The same gap must also exceed the UNPAIRED across-line MDE. If PRIMARY passes and HONEST
              fails, the gain is real but smaller than the line-to-line spread and must be described that
              way.
    STABLE    The same gap must exceed the across-FOLD MDE, so a gain carried by one fold does not pass.
    FREE      MODEL must beat the best 0-parameter arm by more than the paired MDE. If a zero-parameter
              baseline matches or beats the trained network, THAT IS THE HEADLINE.
    PARTNER   The partner channel is credited only if MODEL beats the random-partner control AND the
              degree-matched rewiring control by more than the paired MDE.
    IGNORED   If MODEL does not beat the COUNTERFACTUAL PARTNER SWAP by more than the paired MDE, the
              partner channel is being IGNORED by the trained network rather than being uninformative,
              and the two must not be reported as the same finding.
    AGREE     The secondary per-line Pearson correlation must reach the same verdict as RMSE on the
              PRIMARY gap. Disagreement means metric artifact and the result is not reportable as biology.

EFFECT SIZES ARE RAW HELD-OUT VALUES. Every number quoted is a raw held-out RMSE or correlation. The
controls establish significance and attribution; no arm is ever reported "net of" a control. Where a
control reproduces most of an arm's advantage, that is stated as the finding.

A MISSING VALUE IS NOT ZERO. `CRISPRGeneEffect` is ~96.4% observed and its NaNs mean THE GENE WAS NOT
SCREENED IN THAT LINE. Genes with any missing value across the lines in use are DROPPED, never imputed as
zero. A real 0.0 in this file means "knocking this gene out did nothing", which is a measurement; that is
the opposite of the `nlz_*` benches, which are 3.5% dense and whose zeros mean NOT RECORDED and which are
therefore never used here.

NO CIRCULARITY, AND THIS ONE NEEDED CARE. Features and target come from the SAME dataset, so the check is
not "different datasets" but "no cell of the scored matrix, and nothing derived from a held-out line's eval
genes, enters any input". Concretely: (i) mu, a_g, the gene embedding, the line-embedding basis, the
co-dependency correlations and the 1-parameter shrinkage are computed from TRAINING LINES ONLY and refit in
every fold; (ii) the only held-out-line information any arm sees is that line's PROBE genes, which are
disjoint from the scored EVAL genes; (iii) the prebuilt co-dependency graph in `cell_complete.json.gz` is
DELIBERATELY NOT USED, because it was derived from all of DepMap including the held-out lines and would leak
them into the partner set. The partner graph is rebuilt from training lines inside every fold instead.
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
GE = SP / "CRISPRGeneEffect.csv"

# --- task geometry ---------------------------------------------------------------------------------
FOLDS = int(os.environ.get("VH_FOLDS", "5"))        # disjoint CELL-LINE blocks; every line held out once
G_EVAL = int(os.environ.get("VH_EVAL_GENES", "1500"))   # scored genes, seeded data-independent subsample
G_PROBE = int(os.environ.get("VH_PROBE_GENES", "1500"))  # revealed genes, DISJOINT from the eval panel
KPART = int(os.environ.get("VH_K", "8"))            # co-dependency partners per eval gene
SEL_N = int(os.environ.get("VH_SELECTIVE", "300"))  # secondary universe: top-variance genes, TRAIN-only
# --- model ------------------------------------------------------------------------------------------
DIM = int(os.environ.get("VH_DIM", "16"))           # gene / line embedding width
HID = int(os.environ.get("VH_HID", "64"))
EPOCHS = int(os.environ.get("VH_EPOCHS", "12"))
PAIRS = int(os.environ.get("VH_PAIRS", "250000"))   # (line, gene) training pairs drawn per epoch
BATCH = int(os.environ.get("VH_BATCH", "1024"))
INIT_SEEDS = int(os.environ.get("VH_INIT_SEEDS", "3"))  # fold 0 only. REPORTED, NOT GRADED
N_LINES = int(os.environ.get("VH_LINES", "0"))      # 0 = every cell line in the file
SMOKE = os.environ.get("VH_SMOKE", "0") == "1"
if SMOKE:
    # A SMOKE RUN IS A PLUMBING CHECK AND NOT A RESULT. At 200 lines / 250 genes / 2 epochs the network
    # is undertrained, the co-dependency correlations are estimated from ~100 lines and are mostly noise,
    # and n is a fifth of the real n, so the MDE is not the real MDE. Nothing below is a finding.
    N_LINES, G_EVAL, G_PROBE, FOLDS = 200, 250, 250, 2
    EPOCHS, PAIRS, INIT_SEEDS, SEL_N = 2, 20000, 2, 60


def main():
    import time
    import pandas as pd
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)          # two full-scale training runs own most of this 4-core box
    t0 = time.time()
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 108)
    report("V4 BLOCK 8 -- the VIABILITY head vs its own marginal (CELLFORMER4.md sec 11 / criterion 3)")
    report("=" * 108)
    report("  GATE, declared before the run: MODEL must beat MARGINAL-both on held-out-line RMSE by more")
    report(f"  than MDE = 3*sd/sqrt(n) with n = THE NUMBER OF HELD-OUT CELL LINES and sd the PER-LINE PAIRED")
    report( "  difference sd. The unpaired and across-fold bars are reported next to every gate. Model-init")
    report( "  seeds are run on fold 0 and are REPORTED, NOT GRADED -- they measure optimiser noise.")

    # ------------------------------------------------------------------ target: DepMap CRISPRGeneEffect
    df = pd.read_csv(GE, index_col=0, engine="c")
    X = df.values.astype(np.float32)
    genes = np.array([str(c).split(" ")[0] for c in df.columns])
    lines = np.array([str(i) for i in df.index])
    del df
    n_line_all, n_gene_all = X.shape
    nanfrac = float(np.isnan(X).mean())
    if N_LINES:
        sel = np.random.default_rng(11).permutation(n_line_all)[:N_LINES]
        sel.sort()
        X, lines = X[sel], lines[sel]
    # A MISSING VALUE IS NOT ZERO. NaN here means the gene was not screened in that line. Drop, never fill.
    ok = np.isfinite(X).all(0)
    X, genes = X[:, ok], genes[ok]
    NL, NG = X.shape
    report(f"\n  DepMap CRISPRGeneEffect: {n_line_all:,} cell lines x {n_gene_all:,} genes, "
           f"{100*(1-nanfrac):.1f}% observed")
    report(f"  genes complete across the {NL:,} lines in use: {NG:,}  (NaN = NOT SCREENED and is DROPPED, "
           f"never imputed as 0; a real 0.0 here means 'no effect' and is a measurement)")
    report(f"  gene effect: mean {X.mean():+.4f}  sd {X.std():.4f}  "
           f"(Chronos units; -1 ~ the median common-essential)")

    # gene panels: ONE seeded, data-independent permutation. EVAL and PROBE are disjoint by construction.
    gperm = np.random.default_rng(0).permutation(NG)
    EV = np.sort(gperm[:G_EVAL])
    PR = np.sort(gperm[G_EVAL:G_EVAL + G_PROBE])
    assert len(np.intersect1d(EV, PR)) == 0, "EVAL and PROBE panels must be disjoint"
    Xe = np.ascontiguousarray(X[:, EV]).astype(np.float64)
    Xp = np.ascontiguousarray(X[:, PR]).astype(np.float64)
    del X
    report(f"  EVAL panel {len(EV):,} genes (scored, never revealed) / PROBE panel {len(PR):,} genes "
           f"(revealed for a held-out line) -- DISJOINT, asserted")
    report(f"  no cell that is scored is ever an input to any arm; the only held-out-line information any "
           f"arm sees is that line's PROBE values")

    # ------------------------------------------------------------------------------- folds over LINES
    lperm = np.random.default_rng(1).permutation(NL)
    blocks = np.array_split(lperm, FOLDS)
    report(f"\n  {FOLDS} DISJOINT CELL-LINE folds; every one of the {NL:,} lines is held out exactly once, "
           f"so n = {NL:,} held-out lines")

    class MLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(),
                                   nn.Linear(HID, 1))

        def forward(self, x):
            return self.f(x).squeeze(-1)

    D_IN = 3 * DIM + 3
    NPAR = sum(p.numel() for p in MLP(D_IN).parameters())

    # per-arm, per-LINE scores. Every list ends up with exactly NL entries, one per held-out line.
    per_line = {}
    per_fold = {}
    order_lines = []

    def record(name, fold, rmse_v, r_v, resid_v, rmse_sel_v):
        d = per_line.setdefault(name, {"rmse": [], "r": [], "resid_r": [], "rmse_sel": []})
        d["rmse"].extend(rmse_v)
        d["r"].extend(r_v)
        d["resid_r"].extend(resid_v)
        d["rmse_sel"].extend(rmse_sel_v)
        per_fold.setdefault(name, {"rmse": [], "r": [], "resid_r": [], "rmse_sel": []})
        per_fold[name]["rmse"].append(float(np.mean(rmse_v)))
        per_fold[name]["r"].append(float(np.mean(r_v)))
        per_fold[name]["resid_r"].append(float(np.mean(resid_v)))
        per_fold[name]["rmse_sel"].append(float(np.mean(rmse_sel_v)))

    init_seed_scores = []
    fold_meta = []

    for f in range(FOLDS):
        te = np.sort(blocks[f])
        tr = np.sort(np.concatenate([b for g, b in enumerate(blocks) if g != f]))
        order_lines.extend(lines[te].tolist())

        # ---- marginals, TRAINING LINES ONLY, refit in this fold -------------------------------------
        mu = float(np.concatenate([Xe[tr].ravel(), Xp[tr].ravel()]).mean())
        a_ev = Xe[tr].mean(0) - mu                       # THE PAN-ESSENTIAL PRIOR
        a_pr = Xp[tr].mean(0) - mu
        b = (Xp - mu - a_pr[None, :]).mean(1)            # line offset from PROBE genes only, any line
        Pres = (Xp - mu - a_pr[None, :]) - b[:, None]    # probe residual, any line
        Eres = (Xe - mu - a_ev[None, :]) - b[:, None]    # eval residual == what a model must supply

        # ---- co-dependency partners, rebuilt from TRAINING LINES ONLY -------------------------------
        # NOT cell_complete.json.gz's codep: that graph was built on all of DepMap and would leak the
        # held-out lines into the partner set. Rebuilding it per fold is the whole point.
        def codep(rows):
            ae = Eres[rows] - Eres[rows].mean(0)
            ap = Pres[rows] - Pres[rows].mean(0)
            s_e = np.maximum(ae.std(0), 1e-6)
            s_p = np.maximum(ap.std(0), 1e-6)
            return np.clip((ae / s_e).T @ (ap / s_p) / len(rows), -1.0, 1.0), ae, ap, s_e

        C, Ae, Ap, se = codep(tr)                        # (G_EVAL, G_PROBE) train-line correlation
        idx_real = np.argpartition(-np.abs(C), KPART, axis=1)[:, :KPART]
        W_real = np.take_along_axis(C, idx_real, 1)

        rngc = np.random.default_rng(200 + f)
        idx_rand = rngc.integers(0, len(PR), idx_real.shape)          # matched on COUNT only
        W_rand = np.take_along_axis(C, idx_rand, 1)
        # CONFIGURATION MODEL: permute the stubs. Every eval gene keeps its degree (KPART) and every probe
        # gene keeps its exact multiplicity as a partner, so ONLY the wiring moves. Uniform random above
        # also flattens the partner-usage distribution and destroys two things at once.
        stubs = idx_real.ravel().copy()
        rngc.shuffle(stubs)
        idx_cfg = stubs.reshape(idx_real.shape)
        W_cfg = np.take_along_axis(C, idx_cfg, 1)

        def channel(idx, W):
            num = np.zeros((NL, len(EV)))
            for j in range(KPART):
                num += Pres[:, idx[:, j]] * W[:, j][None, :]
            return num / (np.abs(W).sum(1)[None, :] + 1e-6)

        pc_real = channel(idx_real, W_real)
        pc_rand = channel(idx_rand, W_rand)
        pc_cfg = channel(idx_cfg, W_cfg)

        # ---- embeddings, TRAINING LINES ONLY --------------------------------------------------------
        _, _, Vte = np.linalg.svd(Ae, full_matrices=False)
        Ug = Vte[:DIM].T.copy()                                   # (G_EVAL, DIM) gene loadings
        Ug = (Ug - Ug.mean(0)) / np.maximum(Ug.std(0), 1e-6)
        _, _, Vtp = np.linalg.svd(Ap, full_matrices=False)
        Vl = Pres @ Vtp[:DIM].T                                   # (NL, DIM) line scores, any line
        Vl = (Vl - Vl[tr].mean(0)) / np.maximum(Vl[tr].std(0), 1e-6)
        sg = (se - se.mean()) / max(se.std(), 1e-6)
        ag = (a_ev - a_ev.mean()) / max(a_ev.std(), 1e-6)

        # SELECTIVE universe: the most variable eval genes BY TRAINING LINES ONLY. No held-out line takes
        # part in the selection, so this secondary universe is not a leak.
        sel_g = np.argpartition(-se, min(SEL_N, len(EV) - 1))[:SEL_N]

        # 1-PARAMETER shrinkage, closed form, on an INNER SPLIT OF THE TRAINING LINES. Fitting alpha on
        # the same training lines that estimated C would fit it to C's own overfitting and return alpha
        # near or above 1 -- the channel looks better in-sample than it is. The correlations are therefore
        # re-estimated on 80% of the training lines and alpha is fit on the held-back 20%. Held-out lines
        # take no part in either step.
        inner = np.random.default_rng(600 + f).permutation(len(tr))
        ia, ib = tr[inner[:int(.8 * len(tr))]], tr[inner[int(.8 * len(tr)):]]
        Ci = codep(ia)[0]
        ii = np.argpartition(-np.abs(Ci), KPART, axis=1)[:, :KPART]
        pc_i = channel(ii, np.take_along_axis(Ci, ii, 1))
        pv, ev_ = pc_i[ib].ravel(), Eres[ib].ravel()
        alpha = float(pv @ ev_ / max(pv @ pv, 1e-12))

        report(f"\n  FOLD {f}: {len(tr):,} training lines / {len(te):,} HELD-OUT lines. mu {mu:+.4f}; "
               f"gene marginal a_g sd {a_ev.std():.4f}; line marginal b_c sd {b[te].std():.4f}")
        report(f"    train-line co-dependency: median |corr| of the {KPART} kept partners "
               f"{np.median(np.abs(W_real)):.3f} (real) vs {np.median(np.abs(W_rand)):.3f} (random) vs "
               f"{np.median(np.abs(W_cfg)):.3f} (degree-matched); 1-param shrinkage alpha {alpha:+.3f}")

        Ytrue = Xe[te]                                   # THE SAME matrix every arm is scored against
        base_both = (mu + a_ev[None, :] + b[te][:, None])
        Rtrue = Eres[te]

        def score(pred, name):
            """Per-HELD-OUT-LINE RMSE, Pearson r, and residual-space r. Identical cells for every arm."""
            assert pred.shape == Ytrue.shape
            d = pred - Ytrue
            rmse = np.sqrt((d ** 2).mean(1))
            rmse_sel = np.sqrt((d[:, sel_g] ** 2).mean(1))
            r, rr = [], []
            for i in range(len(te)):
                p, y = pred[i], Ytrue[i]
                # A CONSTANT PREDICTOR carries no across-gene ordering, so its Pearson r is 0 by
                # definition rather than undefined. Stated, not silently patched.
                r.append(0.0 if p.std() < 1e-12 else
                         float(np.corrcoef(p, y)[0, 1]))
                q = p - base_both[i]
                rr.append(0.0 if q.std() < 1e-12 or Rtrue[i].std() < 1e-12 else
                          float(np.corrcoef(q, Rtrue[i])[0, 1]))
            record(name, f, rmse.tolist(), r, rr, rmse_sel.tolist())
            report(f"    {name:38s} RMSE {rmse.mean():.4f}  r {np.mean(r):+.4f}  "
                   f"resid-r {np.mean(rr):+.4f}  RMSE(selective) {rmse_sel.mean():.4f}")

        ones = np.ones_like(Ytrue)
        score(mu * ones, "GLOBAL-mean (0 param)")
        score(mu + a_ev[None, :] + 0 * ones, "MARGINAL-gene (0 param)")
        score(mu + b[te][:, None] + 0 * ones, "MARGINAL-line (0 param)")
        score(base_both, "MARGINAL-both (0 param)")
        score(base_both + pc_real[te], "PARTNER-MEAN (0 param)")
        score(base_both + pc_rand[te], "PARTNER-MEAN random (0p CONTROL)")
        score(base_both + alpha * pc_real[te], "PARTNER-MEAN shrunk (1 param)")

        # ---- the learned head -----------------------------------------------------------------------
        rngb = np.random.default_rng(300 + f)

        def feats(ci, gi, pc, use_pc):
            u, v = Ug[gi], Vl[ci]
            p = (pc[ci, gi] if use_pc else np.zeros(len(ci)))
            return np.concatenate([u, v, u * v, p[:, None], sg[gi][:, None], ag[gi][:, None]],
                                  1).astype(np.float32)

        def fit(pc, use_pc, seed):
            torch.manual_seed(seed)
            net = MLP(D_IN)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            for _e in range(EPOCHS):
                for _s in range(max(PAIRS // BATCH, 1)):
                    ci = tr[rngb.integers(0, len(tr), BATCH)]
                    gi = rngb.integers(0, len(EV), BATCH)
                    x = torch.from_numpy(feats(ci, gi, pc, use_pc))
                    y = torch.from_numpy(Eres[ci, gi].astype(np.float32))
                    loss = (net(x) - y).pow(2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return net

        def emit(net, pc, use_pc, rows=None):
            rows = te if rows is None else rows
            gi = np.arange(len(EV))
            o = np.empty((len(rows), len(EV)))
            with torch.no_grad():
                for i, c in enumerate(rows):
                    o[i] = net(torch.from_numpy(
                        feats(np.full(len(EV), c), gi, pc, use_pc))).numpy()
            return o

        net = fit(pc_real, True, 1000 + f)
        Rhat = emit(net, pc_real, True)
        score(base_both + Rhat, "MODEL (gene+line+partners)")

        net_np = fit(pc_real, False, 1000 + f)
        score(base_both + emit(net_np, pc_real, False), "ABLATION no-partner (same width)")

        net_rd = fit(pc_rand, True, 1000 + f)
        score(base_both + emit(net_rd, pc_rand, True), "CONTROL random partners")

        net_cf = fit(pc_cfg, True, 1000 + f)
        score(base_both + emit(net_cf, pc_cfg, True), "CONTROL degree-matched rewiring")

        # COUNTERFACTUAL: the trained network, correct gene features and correct line embedding, but the
        # partner channel taken from ANOTHER held-out line. Separates "useless" from "IGNORED".
        sw = np.random.default_rng(400 + f).permutation(len(te))
        pc_swap = pc_real.copy()
        pc_swap[te] = pc_real[te[sw]]
        score(base_both + emit(net, pc_swap, True), "COUNTERFACTUAL partner swap")

        score(base_both + Rhat[sw], "CONTROL wrong-line (identity)")
        gsw = np.random.default_rng(500 + f).permutation(len(EV))
        score(base_both + Rhat[:, gsw], "CONTROL wrong-gene (identity)")

        # model-init seeds, FOLD 0 ONLY, REPORTED AND NOT GRADED
        if f == 0:
            for s_ in range(INIT_SEEDS):
                ns = fit(pc_real, True, 7000 + s_)
                p = base_both + emit(ns, pc_real, True)
                init_seed_scores.append(float(np.sqrt(((p - Ytrue) ** 2).mean(1)).mean()))
            report(f"    model-init seed spread on THIS FOLD ONLY (not graded): RMSE sd "
                   f"{np.std(init_seed_scores, ddof=1):.5f} over {INIT_SEEDS} seeds")
        fold_meta.append({"fold": f, "n_train_lines": int(len(tr)), "n_held_out_lines": int(len(te)),
                          "mu": mu, "alpha_1param": alpha,
                          "median_abs_corr_real": float(np.median(np.abs(W_real))),
                          "median_abs_corr_random": float(np.median(np.abs(W_rand))),
                          "median_abs_corr_degree_matched": float(np.median(np.abs(W_cfg)))})

    # ------------------------------------------------------------------------------------- aggregate
    A = {k: {m: np.array(v[m]) for m in v} for k, v in per_line.items()}
    n = len(A["MARGINAL-both (0 param)"]["rmse"])
    assert n == NL, f"every line must be held out exactly once ({n} vs {NL})"

    def mean(nm, m="rmse"):
        return float(A[nm][m].mean())

    report(f"\n  {'arm':38s} {'RMSE':>8s} {'sd/line':>8s} {'sd/fold':>8s} {'r':>8s} {'resid-r':>8s} "
           f"{'RMSEsel':>8s}")
    for k in A:
        report(f"  {k:38s} {A[k]['rmse'].mean():8.4f} {A[k]['rmse'].std(ddof=1):8.4f} "
               f"{np.std(per_fold[k]['rmse'], ddof=1):8.4f} {A[k]['r'].mean():+8.4f} "
               f"{A[k]['resid_r'].mean():+8.4f} {A[k]['rmse_sel'].mean():8.4f}")

    REF = "MARGINAL-both (0 param)"
    ZERO = ["GLOBAL-mean (0 param)", "MARGINAL-gene (0 param)", "MARGINAL-line (0 param)", REF,
            "PARTNER-MEAN (0 param)"]

    def gap(arm, ref=REF, m="rmse"):
        """POSITIVE = the arm is better. RMSE is lower-is-better, so the sign is flipped for it."""
        d = A[ref][m] - A[arm][m]
        return d if m == "rmse" else A[arm][m] - A[ref][m]

    def bars(arm, ref=REF, m="rmse"):
        d = gap(arm, ref, m)
        paired = 3 * float(d.std(ddof=1)) / np.sqrt(n)
        # UNPAIRED: the across-line sd of a single arm's own score, averaged over the TWO arms being
        # compared. Pooling over every arm would drag in the degenerate GLOBAL-mean / MARGINAL-line arms,
        # which live at a completely different RMSE, and would inflate the bar for reasons unrelated to
        # the comparison being made.
        pooled = float(np.mean([A[arm][m].std(ddof=1), A[ref][m].std(ddof=1)]))
        unpaired = 3 * pooled / np.sqrt(n)
        fd = np.array(per_fold[ref][m]) - np.array(per_fold[arm][m])
        if m != "rmse":
            fd = -fd
        fold = 3 * float(np.std(fd, ddof=1)) / np.sqrt(FOLDS)
        return float(d.mean()), paired, unpaired, fold, float((d > 0).mean())

    g_model, mde_p, mde_u, mde_f, win = bars("MODEL (gene+line+partners)")
    g_model_r, mde_pr, _, _, _ = bars("MODEL (gene+line+partners)", m="r")
    best_zero = min(ZERO, key=lambda k: mean(k))
    g_free, mde_free, _, _, win_free = bars("MODEL (gene+line+partners)", best_zero)
    g_rand, mde_rand, _, _, _ = bars("MODEL (gene+line+partners)", "CONTROL random partners")
    g_cfg, mde_cfg, _, _, _ = bars("MODEL (gene+line+partners)", "CONTROL degree-matched rewiring")
    g_swap, mde_swap, _, _, _ = bars("MODEL (gene+line+partners)", "COUNTERFACTUAL partner swap")
    g_abl, mde_abl, _, _, _ = bars("MODEL (gene+line+partners)", "ABLATION no-partner (same width)")
    g_wl, _, _, _, _ = bars("MODEL (gene+line+partners)", "CONTROL wrong-line (identity)")
    g_pm, mde_pm, _, _, win_pm = bars("PARTNER-MEAN (0 param)")
    g_gene, mde_gene, _, _, _ = bars("MARGINAL-gene (0 param)")
    sd_seed = float(np.std(init_seed_scores, ddof=1)) if len(init_seed_scores) > 1 else float("nan")
    mde_seed = 3 * sd_seed / np.sqrt(max(len(init_seed_scores), 2))

    report(f"\n  MDE = 3*sd/sqrt(n), n = {n:,} HELD-OUT CELL LINES (every line held out exactly once). "
           f"The unit of\n  replication is a CELL LINE because the claim is generalisation to an unseen "
           f"cell line.")
    report(f"    PAIRED   (sd of the per-line MODEL-minus-MARGINAL-both difference)   {mde_p:.5f}   <- the gate")
    report(f"    UNPAIRED (pooled sd of a single arm's per-line RMSE)                 {mde_u:.5f}")
    report(f"    FOLD     (sd of the per-fold mean gap, n = {FOLDS})                            {mde_f:.5f}")
    report(f"    model-init-seed sd inside fold 0, REPORTED AND NOT GRADED: {sd_seed:.5f} "
           f"(a {mde_seed:.5f} floor). It measures optimiser noise inside a FIXED line set.")
    report(f"  PER-LINE SPREAD, which is the point of a large n: MODEL-minus-MARGINAL-both RMSE gain has "
           f"per-line sd {gap('MODEL (gene+line+partners)').std(ddof=1):.4f}, "
           f"q10 {np.quantile(gap('MODEL (gene+line+partners)'), .10):+.4f}, "
           f"median {np.median(gap('MODEL (gene+line+partners)')):+.4f}, "
           f"q90 {np.quantile(gap('MODEL (gene+line+partners)'), .90):+.4f}; the model is better on "
           f"{100*win:.1f}% of held-out lines.")

    gates = {
        "PRIMARY  MODEL beats MARGINAL-both by > paired across-line MDE": bool(g_model > mde_p),
        "HONEST   the same gap also exceeds the UNPAIRED across-line MDE": bool(g_model > mde_u),
        "STABLE   the same gap also exceeds the across-FOLD MDE": bool(g_model > mde_f),
        "FREE     MODEL beats the best 0-parameter arm by > MDE": bool(g_free > mde_free),
        "PARTNER  MODEL beats random-partner AND degree-matched rewiring by > MDE":
            bool(g_rand > mde_rand and g_cfg > mde_cfg),
        "IGNORED  MODEL beats the COUNTERFACTUAL partner swap by > MDE": bool(g_swap > mde_swap),
        "AGREE    per-line Pearson r reaches the same verdict as RMSE on the primary gap":
            bool((g_model > mde_p) == (g_model_r > mde_pr)),
    }
    report("\n  PREDECLARED GATES")
    for k, v in gates.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k}")
    report(f"    MODEL {mean('MODEL (gene+line+partners)'):.4f} vs MARGINAL-both {mean(REF):.4f}  "
           f"gap {g_model:+.5f} (paired MDE {mde_p:.5f}, unpaired {mde_u:.5f}, fold {mde_f:.5f})")
    report(f"    best 0-parameter arm = '{best_zero}' at {mean(best_zero):.4f}; MODEL is {g_free:+.5f} "
           f"better, and better on {100*win_free:.1f}% of held-out lines")
    report(f"    MARGINAL-gene alone (the PAN-ESSENTIAL PRIOR) {mean('MARGINAL-gene (0 param)'):.4f}, "
           f"i.e. {g_gene:+.5f} vs MARGINAL-both; GLOBAL-mean {mean('GLOBAL-mean (0 param)'):.4f}")
    report(f"    partner attribution: vs random {g_rand:+.5f}, vs degree-matched rewiring {g_cfg:+.5f}, "
           f"vs no-partner ablation {g_abl:+.5f}, vs counterfactual swap {g_swap:+.5f}")

    # Rule: a control that reproduces most of the effect IS the finding.
    def frac(x, whole):
        return float(x / whole) if abs(whole) > 1e-12 else float("nan")
    denom = g_model
    repro = {
        "0-parameter PARTNER-MEAN reproduces this fraction of MODEL's gain over MARGINAL-both":
            frac(g_pm, denom),
        "the 1-parameter shrunk partner mean reproduces this fraction":
            frac(bars("PARTNER-MEAN shrunk (1 param)")[0], denom),
        "the no-partner ABLATION reproduces this fraction":
            frac(bars("ABLATION no-partner (same width)")[0], denom),
        "the RANDOM-partner control reproduces this fraction":
            frac(bars("CONTROL random partners")[0], denom),
        "the DEGREE-MATCHED rewiring control reproduces this fraction":
            frac(bars("CONTROL degree-matched rewiring")[0], denom),
        "the COUNTERFACTUAL partner swap reproduces this fraction":
            frac(bars("COUNTERFACTUAL partner swap")[0], denom),
        "the WRONG-LINE identity control reproduces this fraction":
            frac(bars("CONTROL wrong-line (identity)")[0], denom)}
    report("\n  CONTROLS THAT REPRODUCE THE EFFECT (reported as the finding, not as a footnote)")
    for k, v in repro.items():
        report(f"    {100*v:8.1f}%  {k}")
    repro_meaningful = bool(g_model > mde_p)
    if not repro_meaningful:
        report(f"    ^ the denominator, MODEL's gain over MARGINAL-both, is {g_model:+.5f}, itself at or "
               f"below the {mde_p:.5f} MDE. These percentages are ratios of two undetected quantities and "
               f"mean nothing; they are printed rather than hidden so the denominator is visible.")

    pan_wins = mean("MARGINAL-gene (0 param)") <= mean("MODEL (gene+line+partners)")
    zero_wins = mean(best_zero) <= mean("MODEL (gene+line+partners)")
    verdict = (
        (f"SMOKE RUN ({NL:,} lines, {len(EV):,} eval genes, {EPOCHS} epochs, {FOLDS} folds) -- THIS IS A "
         f"PLUMBING CHECK AND NOT A RESULT. The network is undertrained, the co-dependency correlations "
         f"are estimated from ~{int(NL*(FOLDS-1)/FOLDS)} lines and are mostly noise, and n is a fraction "
         f"of the real n so the MDE printed here is not the real MDE. No sentence below may be quoted as "
         f"a finding. Rerun with VH_SMOKE unset. " if SMOKE else "") +
        (f"THE PAN-ESSENTIAL MARGINAL WINS: predicting every held-out cell line with nothing but the "
         f"gene's mean effect across the training lines reaches RMSE {mean('MARGINAL-gene (0 param)'):.4f}, "
         f"which the trained viability head ({mean('MODEL (gene+line+partners)'):.4f}) does not beat. "
         if pan_wins else
         f"The trained viability head reaches held-out-cell-line RMSE "
         f"{mean('MODEL (gene+line+partners)'):.4f} against the pan-essential gene marginal's "
         f"{mean('MARGINAL-gene (0 param)'):.4f}. ") +
        f"CRITERION 3 of CELLFORMER4.md section 17 asks whether this head beats its marginal: against the "
        f"additive MARGINAL-both ({mean(REF):.4f}) the model is {g_model:+.5f} RMSE, versus a minimum "
        f"detectable increment of {mde_p:.5f} (3*sd/sqrt(n), n = {n:,} HELD-OUT CELL LINES, sd = the "
        f"per-line PAIRED difference sd; the unpaired bar is {mde_u:.5f} and the across-fold bar "
        f"{mde_f:.5f}), so criterion 3 for the viability head "
        f"{'PASSES' if gates['PRIMARY  MODEL beats MARGINAL-both by > paired across-line MDE'] else 'FAILS'}"
        f"{'' if gates['HONEST   the same gap also exceeds the UNPAIRED across-line MDE'] else ' on the paired bar only -- the gain is smaller than the line-to-line spread and must be described that way'}"
        f". The model is better than the additive marginal on {100*win:.1f}% of held-out lines, with a "
        f"per-line gain spread of {gap('MODEL (gene+line+partners)').std(ddof=1):.4f} "
        f"(q10 {np.quantile(gap('MODEL (gene+line+partners)'), .10):+.4f}, "
        f"q90 {np.quantile(gap('MODEL (gene+line+partners)'), .90):+.4f}), which is the number a large n "
        f"buys and which a fold-level mean would have hidden. "
        + (f"A ZERO-PARAMETER, UNTRAINED BASELINE ('{best_zero}', {mean(best_zero):.4f}) MATCHES OR BEATS "
           f"THE TRAINED NETWORK by {-g_free:+.5f}, AND THAT IS THE HEADLINE, not a footnote. "
           if zero_wins else
           f"The best zero-parameter arm ('{best_zero}') is {mean(best_zero):.4f}, i.e. {g_free:+.5f} "
           f"behind the trained network. ") +
        f"Attribution of the co-dependency partner channel, which section 11 places specifically in this "
        f"head: the untrained correlation-signed partner mean alone moves RMSE by {g_pm:+.5f} against "
        f"MARGINAL-both and reproduces {100*frac(g_pm, denom):.0f}% of the trained model's gain; the model "
        f"beats uniformly random partners by {g_rand:+.5f} and a DEGREE-MATCHED configuration-model "
        f"rewiring by {g_cfg:+.5f}; removing the channel entirely from the identical network costs "
        f"{g_abl:+.5f}. THE COUNTERFACTUAL SWAP, in which the trained network keeps the correct gene and "
        f"line features but is handed another held-out line's partner values, changes RMSE by "
        f"{g_swap:+.5f}, so the partner channel is "
        f"{'genuinely being read' if g_swap > mde_swap else 'IGNORED by the trained network rather than merely uninformative -- accuracy alone cannot separate those two and this is what separates them'}. "
        f"The identity control (the model's residual predictions permuted across held-out lines) is "
        f"{g_wl:+.5f}, which bounds how much of the gap the metric would concede to any line-shaped "
        f"prediction. The secondary per-line Pearson correlation "
        f"{'AGREES' if gates['AGREE    per-line Pearson r reaches the same verdict as RMSE on the primary gap'] else 'DISAGREES'} "
        f"with RMSE on the primary gap ({g_model_r:+.5f} against a {mde_pr:.5f} bar)"
        f"{'.' if gates['AGREE    per-line Pearson r reaches the same verdict as RMSE on the primary gap'] else ', and when the two metrics disagree the result is a metric artifact and is not reportable as biology.'} "
        f"Note by construction that a per-line constant offset cannot change a per-line correlation, so "
        f"MARGINAL-gene and MARGINAL-both are identical on r and differ only on RMSE; that is a property of "
        f"the metric, not a bug. All effect sizes above are RAW held-out values; the controls establish "
        f"significance and attribution and nothing is reported net of a control.")
    report("\n" + "=" * 108)
    report(f"  VERDICT: {verdict}")
    report(f"\n  wall clock {time.time()-t0:.1f}s")

    R = {"model": "v4-viability-head-v1",
         "title": "V4 Block 8 -- viability head vs its own marginal (n = held-out CELL LINES, and it is large)",
         "question": "CELLFORMER4.md sec 11 (Block 8 viability head, codependency belongs here) and sec 17 "
                     "criterion 3 (every head beats its marginal by more than its MDE): does a learned "
                     "viability head predict a gene's CRISPR effect in an UNSEEN CELL LINE better than the "
                     "pan-essential gene marginal plus a line offset?",
         "task": {"data": f"DepMap CRISPRGeneEffect.csv, {n_line_all:,} cell lines x {n_gene_all:,} genes, "
                          f"{100*(1-nanfrac):.1f}% observed. NaN means NOT SCREENED and rows/columns with "
                          f"any NaN are DROPPED, never imputed as zero. A real 0.0 means 'no effect' and "
                          f"is a measurement -- unlike the nlz_* benches (3.5% dense, zeros = NOT "
                          f"RECORDED), which are never used here",
                  "lines_used": int(NL), "genes_complete": int(NG),
                  "eval_panel": int(len(EV)), "probe_panel": int(len(PR)),
                  "panels_disjoint": True,
                  "setup": "cold-start cell line: a held-out line reveals only its PROBE genes; the "
                           "scored EVAL genes are disjoint from the probe panel and are never revealed",
                  "primary_metric": "per-held-out-line RMSE in Chronos gene-effect units, over the eval "
                                    "panel (lower is better)",
                  "secondary_metric": "per-held-out-line Pearson r across the eval panel; a constant "
                                      "predictor is scored r = 0 by definition",
                  "diagnostic_metric": "residual-space per-line Pearson r after removing MARGINAL-both; "
                                       "MARGINAL-both scores exactly 0 there by construction",
                  "selective_universe": f"top-{SEL_N} eval genes by TRAINING-LINE sd, reselected per fold; "
                                        f"no held-out line takes part in the selection",
                  "partners_per_gene": KPART,
                  "model_params": int(NPAR)},
         "unit_of_replication": f"a HELD-OUT CELL LINE. n = {n:,}: the lines are split into {FOLDS} disjoint "
                                f"blocks and every line is held out exactly once. Model-init seeds are run "
                                f"on fold 0 only and are REPORTED, NOT GRADED",
         "arms": {k: {"rmse_mean": float(v["rmse"].mean()),
                      "rmse_sd_across_held_out_lines": float(v["rmse"].std(ddof=1)),
                      "rmse_per_fold": per_fold[k]["rmse"],
                      "rmse_sd_across_folds": float(np.std(per_fold[k]["rmse"], ddof=1)),
                      "rmse_line_quantiles": [float(q) for q in
                                              np.quantile(v["rmse"], [.1, .25, .5, .75, .9])],
                      "pearson_r_mean": float(v["r"].mean()),
                      "pearson_r_sd_across_held_out_lines": float(v["r"].std(ddof=1)),
                      "pearson_r_per_fold": per_fold[k]["r"],
                      "residual_r_mean": float(v["resid_r"].mean()),
                      "residual_r_per_fold": per_fold[k]["resid_r"],
                      "rmse_selective_mean": float(v["rmse_sel"].mean()),
                      "rmse_selective_per_fold": per_fold[k]["rmse_sel"],
                      "n_held_out_lines": int(len(v["rmse"]))}
                  for k, v in A.items()},
         "mde": {"convention": "3*sd/sqrt(n)",
                 "n": int(n), "n_counts": "held-out CELL LINES, each line held out exactly once",
                 "paired_value": float(mde_p),
                 "paired_definition": "sd of the PER-LINE difference between the two arms compared; arms "
                                      "see identical lines and identical genes so the comparison is paired",
                 "unpaired_value": float(mde_u),
                 "unpaired_definition": "3 * (pooled across-line sd of a single arm's per-line RMSE) / "
                                        "sqrt(n); larger, because it contains between-line difficulty "
                                        "variance that cancels in a paired comparison",
                 "fold_value": float(mde_f), "fold_n": int(FOLDS),
                 "init_seed_sd_fold0_DO_NOT_GRADE_ON_THIS": sd_seed,
                 "init_seed_mde_DO_NOT_GRADE_ON_THIS": float(mde_seed),
                 "init_seed_rmse_fold0": init_seed_scores},
         "gates": gates,
         "gaps_positive_means_better": {
             "model_minus_marginal_both": float(g_model),
             "model_minus_best_zero_parameter_arm": float(g_free),
             "best_zero_parameter_arm": best_zero,
             "marginal_gene_minus_marginal_both": float(g_gene),
             "zero_param_partner_mean_minus_marginal_both": float(g_pm),
             "model_minus_random_partners": float(g_rand),
             "model_minus_degree_matched_rewiring": float(g_cfg),
             "model_minus_no_partner_ablation": float(g_abl),
             "model_minus_counterfactual_partner_swap": float(g_swap),
             "model_minus_wrong_line_identity_control": float(g_wl),
             "model_minus_marginal_both_pearson_r": float(g_model_r),
             "fraction_of_held_out_lines_where_model_beats_marginal_both": float(win),
             "fraction_of_held_out_lines_where_zero_param_partner_mean_beats_marginal_both": float(win_pm),
             "per_line_gain_sd": float(gap("MODEL (gene+line+partners)").std(ddof=1)),
             "per_line_gain_quantiles_10_50_90": [
                 float(np.quantile(gap("MODEL (gene+line+partners)"), q)) for q in (.1, .5, .9)]},
         "controls_reproducing_the_effect": {k: (v if np.isfinite(v) else None) for k, v in repro.items()},
         "controls_reproducing_the_effect_are_meaningful": repro_meaningful,
         "controls_reproducing_the_effect_note":
             "fractions of MODEL's raw RMSE gain over MARGINAL-both. If that denominator is itself at or "
             "below the MDE these ratios are meaningless and the flag above is false.",
         "pan_essential_marginal_wins": bool(pan_wins),
         "zero_parameter_arm_wins": bool(zero_wins),
         "folds": fold_meta,
         "circularity_check":
             "features and target come from the SAME dataset, so the check is not 'different datasets' but "
             "'no scored cell, and nothing derived from a held-out line, enters any input'. (i) mu, the "
             "gene marginal, the gene embedding, the line-embedding basis, the co-dependency correlations, "
             "the selective-gene selection and the 1-parameter shrinkage are all computed from TRAINING "
             "LINES ONLY and refit inside every fold; (ii) the only held-out-line information any arm sees "
             "is that line's PROBE genes, asserted disjoint from the scored EVAL genes; (iii) the prebuilt "
             "co-dependency graph in cell_complete.json.gz is deliberately NOT used because it was derived "
             "from all of DepMap including the held-out lines and would leak them into the partner set.",
         "epochs": EPOCHS, "smoke": bool(SMOKE),
         "limits": [
             "WHAT WOULD FALSIFY THIS: a larger probe panel. Every line-specific quantity here flows "
             "through the probe genes, so the model's whole advantage over the gene marginal is bounded by "
             f"how much {len(PR):,} revealed genes say about a line. Rerunning at VH_PROBE_GENES=4000 and "
             "seeing the gap grow would show this is a probe-budget result, not a ceiling on the head",
             "CONFOUNDED: 'the model' is one MLP at one width on one feature set. A null is a null for "
             "THIS head at THIS capacity, not for any learnable viability head. What is falsified is the "
             "claim as currently made, that a learned head beats the pan-essential marginal",
             "the co-dependency partner graph is rebuilt from training lines inside every fold, which is "
             "what keeps it honest but also makes it noisier than the published DepMap co-dependency "
             f"graph. With {FOLDS} folds each graph rests on ~{int(NL*(FOLDS-1)/FOLDS)} lines, and partner "
             "correlations estimated on fewer lines are attenuated; using the prebuilt graph would score "
             "higher and would be leakage",
             "eval genes are a seeded uniform subsample of the complete-data genes, so the panel is "
             "dominated by non-essential genes with almost no across-line variance. RMSE over that panel "
             "is easy for every arm and compresses all differences. The selective universe (top-variance "
             "genes by TRAINING lines) is reported per arm for exactly this reason and should be read "
             "alongside the primary number",
             "dropping genes with any NaN removes genes not screened in every line, which is a biased "
             "subset: it discards library-specific and late-added genes. That filter is applied "
             "identically to every arm, so it cannot create a between-arm gap, but it does mean the "
             "absolute RMSE is for the always-screened gene universe",
             "the per-line paired MDE is the correct bar for a paired comparison but it is the SMALLEST of "
             "the three reported. A gap that clears the paired bar and not the unpaired bar is real yet "
             "smaller than the spread between cell lines, and both bars are in the JSON so the reader "
             "picks the one matching the claim being made",
             "cell lines are not independent replicates in the biological sense: DepMap contains related "
             "lineages and near-duplicate lines, so the effective n is below the nominal n and every MDE "
             "here is therefore optimistic. A lineage-blocked split (hold out an entire lineage) is the "
             "stricter test and was NOT run",
             "Chronos gene effect is already normalised against common essentials, so part of the "
             "'pan-essential prior' has been baked into the target by DepMap's own pipeline; the marginal "
             "arm's strength is partly a property of that preprocessing and is not purely biology",
             "one screen type (CRISPR KO), one readout (proliferation). Nothing here speaks to drug "
             "response (PRISM), which section 11 also assigns to this head"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "v4_viability_head.json"
    json.dump(R, open(p, "w"), indent=1)
    report(f"\n  -> {p}")


if __name__ == "__main__":
    main()
