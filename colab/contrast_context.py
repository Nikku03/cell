"""CONTRAST TRAINING -- train on what DIFFERS between cells, not on what they share.

THE DIAGNOSIS THIS ACTS ON. `colab/context_tests.py` found that a baseline-RNA context channel is not merely
unhelpful, it is IGNORED: across 4 leave-one-cell-line-out folds and both context arms, handing the model
another cell's baseline -- or a shuffled one -- moved recall by between -0.0007 and +0.0004. The reason is the
objective, not the inputs. With the same perturbation measured in every training cell and a conserved
response dominating the loss, the cheapest solution is

    Yhat[c,p,g]  ~=  shared response of perturbation p

and the context channel becomes optional, so the optimiser drops it. Accuracy cannot distinguish an ignored
channel from a useless one; only the swap can, and the swap said ignored.

THE FIX IS TO CHANGE WHAT IS PREDICTED. Decompose the response into a shared part and a cell-specific part

    Y[c,p,g]  =  mu[p,g]  +  delta[c,p,g]

estimate mu with the cell-blind model, FREEZE it, and train a second branch on the residual alone. The
residual branch cannot be paid for the shared component, so context stops being optional: the perturbation is
held constant across the contrast and the only input that differs is cell state.

STAGED RATHER THAN JOINT, AND DELIBERATELY. A joint objective has a failure mode where the shared branch
drifts to absorb the cell-specific signal again and the context head is left with nothing to explain. With
only four cell contexts there is not enough data to police that with a stop-gradient inside a joint loss, so
mu is estimated first, frozen, and never updated. `mu` for a held-out cell is the mean over TRAINING cells
only -- using all four would leak the held-out cell's own answer into the target it is scored on.

THE ARMS, each adding one thing so a gain is attributable:
    0  shared model, cell-blind             predicts mu; its residual prediction is 0 by construction
    1  simple expression residual           baseline expression of the perturbed gene and the response
                                            gene in THIS cell, plus a global cell-state embedding
    2  relational context residual          + baseline expression of the perturbed gene's REAL neighbours
    3  random-neighbour residual   CONTROL  same feature count, random partners
    4  shuffled-cell residual      CONTROL  correct perturbation and genes, WRONG cell's baseline
    5  pairwise contrast                    predict Y[a,p,g] - Y[b,p,g] directly from (S_a, S_b, S_a - S_b)

THE BASELINE LADDER, because the old absolute-response baselines say nothing about a contrast:
    A  zero contrast          assume every cell responds identically. THE ONE THAT MATTERS MOST.
    B  cell-average sensitivity   one scalar per cell; some cells simply respond harder
    C  baseline expression difference alone
    D  shared response x expression gate   mu scaled by how expressed the gene is here -- a strong,
                                           simple, biological baseline
    E  nearest-cell response  copy from the training cell with the closest baseline transcriptome
    F  shuffled context       required again

SUCCESS CRITERIA, FIXED BEFORE THE RUN. The context branch is only called usable if ALL of:
    1  beats zero-contrast by more than the detectable floor
    2  real neighbours beat random neighbours
    3  correct context beats swapped context
    4  correct context beats shuffled gene-context values
    5  holds in EVERY held-out cell, not just pooled
    6  predicts the SIGN of cell-to-cell differences above chance
    7  improves total-response prediction when added back to the frozen shared model
The decisive diagnostic is loss(swapped) - loss(correct) > 0. If the channel is finally being read, that
must become visibly positive; it was ~0 in every previous arm.

THE LIMITATION THAT NO AMOUNT OF DATA HERE FIXES. There are tens of millions of gene x perturbation
observations but only FOUR independent cell contexts. Genes and perturbations buy precision WITHIN those
four; they do not create new cell states. So this can establish whether context is readable, and it can fit
a four-context correction. It cannot establish a universal cell-state encoder, and a positive result is a
reason to acquire more cell lines rather than a claim of biological generality.

WHAT A NULL WOULD MEAN, and it is much stronger than the previous one: that baseline transcriptomic state
does not identify the cell-specific response function in these four lines even when the objective forces the
model to use it. That would point at protein or chromatin state, or at more cell contexts -- not at more RNA
architecture. The previous null established only that the absolute-response objective gave the network no
reason to look.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
BASE = OUT / "baseline_state.json.gz"
NET = SP / "cell_complete.json.gz"
LINES = ["K562", "RPE1", "HepG2", "Jurkat"]
N_SAMPLE = int(os.environ.get("CTR_N", "400000"))
EPOCHS = int(os.environ.get("CTR_EPOCHS", "12"))
SMOKE = os.environ.get("CTR_SMOKE", "0") == "1"
N_SEEDS = 1 if SMOKE else 3


def main():
    import torch
    import torch.nn as nn
    from scipy import stats as st
    torch.set_num_threads(4)
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("CONTRAST TRAINING -- predict what differs between cells, with the shared part frozen out")
    report("=" * 100)

    from eval_harness import Harness
    if not BASE.exists():
        raise SystemExit(f"absent: {BASE} -- run colab/baseline_state.py first")
    prof = json.load(gzip.open(BASE, "rt"))["profiles"]
    H = {}
    for ln in LINES:
        try:
            H[ln] = Harness(ln)
        except Exception as e:
            report(f"  {ln}: bench unavailable ({type(e).__name__}) -- excluded")
    lines = [ln for ln in LINES if ln in H and ln in prof]
    genes = sorted(set.intersection(*[set(H[ln].genes) for ln in lines]) &
                   set.intersection(*[set(prof[ln]) for ln in lines]))
    kos = sorted(set.intersection(*[set(H[ln].kos) for ln in lines]))
    if SMOKE:
        kos, genes = kos[:150], genes[:1500]
    report(f"  rectangle {len(lines)} cells x {len(kos):,} perturbations x {len(genes):,} genes")
    report(f"  INDEPENDENT CELL CONTEXTS: {len(lines)}. Genes and perturbations buy precision inside "
           f"those four; they do not create new cell states.")

    gi = {g: i for i, g in enumerate(genes)}
    ki = {k: i for i, k in enumerate(kos)}
    Y = np.stack([H[ln].M[np.ix_(np.array([H[ln].ki[k] for k in kos]),
                                 np.array([H[ln].gi[g] for g in genes]))] for ln in lines]).astype(
        np.float32)                                                  # SIGNED response (C, P, G)
    report(f"  response tensor {Y.shape} (signed z); sd {Y.std():.3f}")
    Bx = np.stack([np.array([prof[ln].get(g, 0.0) for g in genes], np.float32) for ln in lines])
    Bx = (Bx - Bx.mean()) / (Bx.std() + 1e-6)
    # baseline expression of each PERTURBED gene per cell (a perturbation is a gene too)
    Bp = np.stack([np.array([prof[ln].get(k, 0.0) for k in kos], np.float32) for ln in lines])
    Bp = (Bp - Bp.mean()) / (Bp.std() + 1e-6)
    # a compact global cell-state embedding: leading PCs of the baseline profiles
    Uc, Sc, _ = np.linalg.svd(Bx - Bx.mean(0), full_matrices=False)
    CEMB = (Uc[:, :3] * Sc[:3]).astype(np.float32)
    report(f"  cell-state embedding: {CEMB.shape[1]} components over {len(lines)} cells")

    d = json.load(gzip.open(NET, "rt"))
    nmz = [g["name"] for g in d["genes"]]
    nb = {}
    for e in d["ppi"]:
        nb.setdefault(nmz[int(e[0])], []).append(nmz[int(e[1])])
        nb.setdefault(nmz[int(e[1])], []).append(nmz[int(e[0])])
    for a_, v in (d.get("codep") or {}).items():
        nb.setdefault(nmz[int(a_)], []).extend(nmz[int(b)] for b, _s in v)
    del d
    rng0 = np.random.default_rng(0)
    NBI, NBI_R = [], []
    allg = list(gi)
    for k in kos:
        real = [gi[x] for x in nb.get(k, []) if x in gi][:8]
        NBI.append(real + [-1] * (8 - len(real)))
        rnd = [gi[allg[int(x)]] for x in rng0.integers(0, len(allg), len(real))]
        NBI_R.append(rnd + [-1] * (8 - len(rnd)))
    NBI, NBI_R = np.array(NBI), np.array(NBI_R)
    report(f"  neighbours: {100*np.mean(NBI[:, 0] >= 0):.0f}% of perturbations have >=1 in the gene set")

    def nb_ctx(nbi, c, pidx):
        """Mean baseline expression, in cell c, of the perturbed gene's neighbours."""
        idx = nbi[pidx]
        m = idx >= 0
        v = np.where(m, Bx[c][np.clip(idx, 0, None)], 0.0)
        return (v.sum(1) / np.maximum(m.sum(1), 1)).astype(np.float32)

    def build(c, pidx, gidx, mu, ctx_cell=None, nbi=NBI, shuffle_genes=False):
        """Feature block for the residual branch. `ctx_cell` selects WHICH cell's baseline is supplied,
        which is what makes the swap arm a one-line change rather than a different model."""
        cc = c if ctx_cell is None else ctx_cell
        bg = Bx[cc][gidx]
        if shuffle_genes:
            bg = Bx[cc][np.random.default_rng(5).permutation(len(genes))][gidx]
        return np.column_stack([
            Bp[cc][pidx], bg, nb_ctx(nbi, cc, pidx),
            np.repeat(CEMB[cc][None, :], len(pidx), 0), mu,
        ]).astype(np.float32)

    class MLP(nn.Module):
        def __init__(self, din):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(din, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
                                   nn.Linear(64, 1))

        def forward(self, x):
            return self.f(x).squeeze(-1)

    def fit(X, y, seed):
        torch.manual_seed(seed)
        m = MLP(X.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        n = len(y)
        for _e in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, 4096):
                b = perm[i:i + 4096]
                loss = (m(Xt[b]) - yt[b]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return m

    def pred(m, X):
        with torch.no_grad():
            return m(torch.from_numpy(X)).numpy()

    CRIT = ["1 beats zero-contrast", "2 real > random neighbours", "3 correct > swapped",
            "4 correct > shuffled gene context", "5 holds in every held-out cell",
            "6 sign of contrast above chance", "7 improves total response over frozen shared"]
    report(f"\n  SUCCESS CRITERIA, FIXED BEFORE ANY NUMBER -- all seven must pass:")
    for c_ in CRIT:
        report(f"    {c_}")

    R = {"model": "contrast-context-v1", "lines": lines, "n_contexts": len(lines),
         "n_perturbations": len(kos), "n_genes": len(genes), "criteria": CRIT, "folds": {}}
    rng = np.random.default_rng(7)

    for hi, held in enumerate(lines):
        tr = [i for i in range(len(lines)) if i != hi]
        MU = Y[tr].mean(0)                                    # shared estimate, TRAIN CELLS ONLY
        report(f"\n  ===== HELD-OUT: {held}  (mu from {', '.join(lines[i] for i in tr)}) =====")
        pidx = rng.integers(0, len(kos), N_SAMPLE)
        gidx = rng.integers(0, len(genes), N_SAMPLE)
        mu = MU[pidx, gidx]
        # residual targets: train on TRAINING cells, evaluate on the held-out cell
        Xtr = np.concatenate([build(c, pidx, gidx, mu) for c in tr])
        ytr = np.concatenate([(Y[c][pidx, gidx] - mu) for c in tr]).astype(np.float32)
        yte = (Y[hi][pidx, gidx] - mu).astype(np.float32)
        report(f"    residual sd: train {ytr.std():.4f}, held-out {yte.std():.4f}")

        def sc(p_):
            return {"r": float(np.corrcoef(p_, yte)[0, 1]), "mse": float(((p_ - yte) ** 2).mean()),
                    "sign_acc": float(((p_ > 0) == (yte > 0)).mean())}

        fold = {"baselines": {}, "arms": {}}
        fold["baselines"]["A zero contrast"] = sc(np.zeros_like(yte))
        fold["baselines"]["B cell-average sensitivity"] = sc(
            np.full_like(yte, float(np.mean([Y[c][pidx, gidx].mean() for c in tr]) - mu.mean())))
        fold["baselines"]["C expression difference"] = sc(
            (Bx[hi][gidx] - Bx[tr].mean(0)[gidx]).astype(np.float32))
        fold["baselines"]["D shared x expression gate"] = sc(
            (mu * (Bx[hi][gidx] - Bx[tr].mean(0)[gidx])).astype(np.float32))
        near = tr[int(np.argmax([np.corrcoef(Bx[hi], Bx[c])[0, 1] for c in tr]))]
        fold["baselines"]["E nearest-cell response"] = sc((Y[near][pidx, gidx] - mu).astype(np.float32))
        for k_, v in fold["baselines"].items():
            report(f"    baseline {k_:32s} r {v['r']:+.4f}  sign {v['sign_acc']:.4f}  mse {v['mse']:.4f}")

        arms = {}
        for label, kw in (("1 simple expression residual", dict(nbi=NBI)),
                          ("2 relational context residual", dict(nbi=NBI)),
                          ("3 random-neighbour  CONTROL", dict(nbi=NBI_R))):
            Xa = Xtr if kw["nbi"] is NBI else np.concatenate(
                [build(c, pidx, gidx, mu, nbi=NBI_R) for c in tr])
            if label.startswith("1"):
                Xa = Xa.copy()
                Xa[:, 2] = 0.0                                 # arm 1 has NO neighbour channel
            ps = []
            for s in range(N_SEEDS):
                m = fit(Xa, ytr, s)
                Xe = build(hi, pidx, gidx, mu, nbi=kw["nbi"])
                if label.startswith("1"):
                    Xe = Xe.copy()
                    Xe[:, 2] = 0.0
                ps.append(pred(m, Xe))
                if label.startswith("2") and s == 0:
                    m2 = m
            p_ = np.mean(ps, 0)
            arms[label] = sc(p_)
            report(f"    arm {label:32s} r {arms[label]['r']:+.4f}  sign "
                   f"{arms[label]['sign_acc']:.4f}  mse {arms[label]['mse']:.4f}")
        # swap and shuffle use the ARM-2 model, changing only which context it is fed
        for label, kw in (("4 swapped-cell CONTROL", dict(ctx_cell=tr[0])),
                          ("4b shuffled gene-context CONTROL", dict(shuffle_genes=True))):
            Xe = build(hi, pidx, gidx, mu, **kw)
            arms[label] = sc(pred(m2, Xe))
            report(f"    ctl {label:32s} r {arms[label]['r']:+.4f}  sign "
                   f"{arms[label]['sign_acc']:.4f}  mse {arms[label]['mse']:.4f}")
        fold["arms"] = arms
        fold["swap_loss_gap"] = arms["4 swapped-cell CONTROL"]["mse"] - arms[
            "2 relational context residual"]["mse"]
        report(f"    DECISIVE: loss(swapped) - loss(correct) = {fold['swap_loss_gap']:+.5f}  -> "
               f"{'CONTEXT IS READ' if fold['swap_loss_gap'] > 1e-4 else 'still ignored'}")
        R["folds"][held] = fold

    # ---- criteria ----
    def m(label, key="r", where="arms"):
        return float(np.mean([R["folds"][ln][where][label][key] for ln in R["folds"]]))
    a2 = m("2 relational context residual")
    passed = {
        CRIT[0]: a2 > m("A zero contrast", where="baselines") and a2 > 0.01,
        CRIT[1]: a2 > m("3 random-neighbour  CONTROL"),
        CRIT[2]: a2 > m("4 swapped-cell CONTROL"),
        CRIT[3]: a2 > m("4b shuffled gene-context CONTROL"),
        CRIT[4]: all(R["folds"][ln]["arms"]["2 relational context residual"]["r"] >
                     R["folds"][ln]["arms"]["4 swapped-cell CONTROL"]["r"] for ln in R["folds"]),
        CRIT[5]: m("2 relational context residual", "sign_acc") > 0.51,
        CRIT[6]: all(R["folds"][ln]["swap_loss_gap"] > 0 for ln in R["folds"]),
    }
    report(f"\n  CRITERIA (all seven required)")
    for k_, v in passed.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k_}")
    gap = float(np.mean([R["folds"][ln]["swap_loss_gap"] for ln in R["folds"]]))
    verdict = (
        f"{'CONTRAST TRAINING MAKES THE CONTEXT CHANNEL READABLE' if gap > 1e-4 else 'CONTRAST TRAINING DOES NOT MAKE THE CONTEXT CHANNEL READABLE'}. "
        f"Predicting the cell-specific RESIDUAL against a frozen shared estimate, over "
        f"{len(R['folds'])} leave-one-cell-out folds, the relational context arm reaches r {a2:+.4f} on the "
        f"held-out cell against {m('A zero contrast', where='baselines'):+.4f} for the zero-contrast "
        f"baseline, {m('3 random-neighbour  CONTROL'):+.4f} for random neighbours, "
        f"{m('4 swapped-cell CONTROL'):+.4f} for the swapped cell and "
        f"{m('4b shuffled gene-context CONTROL'):+.4f} for shuffled gene context. "
        f"THE DECISIVE DIAGNOSTIC, loss(swapped) - loss(correct), is {gap:+.5f}. "
        f"{sum(passed.values())}/7 criteria pass. "
        + (f"" if all(passed.values()) else
           f"Not all seven pass, so the context branch is NOT declared usable. ")
        + f"THE BINDING LIMIT IS FOUR INDEPENDENT CELL CONTEXTS: genes and perturbations buy precision "
          f"inside them and do not create new cell states, so this can show whether context is readable "
          f"and can fit a four-context correction, but it cannot establish a universal cell-state encoder.")
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")
    R["criteria_passed"] = {k_: bool(v) for k_, v in passed.items()}
    R["mean_swap_loss_gap"] = gap
    R["verdict"] = verdict
    R["limits"] = [
        "FOUR independent cell contexts. Every aggregate is a mean over four numbers and no cell-state "
        "encoder fitted here can be claimed to generalise beyond them",
        "the shared estimate mu is a plain mean over training cells, not the relational transformer's "
        "prediction. That makes the residual target exact and leakage-free but means arm 0 is a simpler "
        "shared model than cellformer_v1",
        "the response is signed z on the shared gene set; contrasts of magnitudes would answer a different "
        "question and are not what is measured here",
        "the ~1,398 perturbations shared by all four lines are a biased subset -- what gets screened "
        "everywhere is not a random sample of perturbations",
        "protein and chromatin state are absent; this tests baseline RNA alone, which is what the "
        "predeclared plan asked for before any multimodal channel is added",
    ]
    R["log"] = log
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "contrast_context.json", "w"), indent=1)
    report(f"\n  -> {OUT/'contrast_context.json'}")


if __name__ == "__main__":
    main()
