"""STAGE 3 + 4 -- is the cell channel still ignored at 1,186x the data, and does contrast training fix it?

WHAT THIS IS THE SEQUEL TO. `colab/context_tests.py` ran five narrow context arms under a strict
leave-one-cell-line-out rule and found something sharper than "context does not help": across 4 folds and
both identifiable arms, handing a trained model ANOTHER CELL LINE'S baseline moved recall by between -0.0007
and +0.0004. Eight of eight, both directions. The channel is not weak, it is NOT READ. Accuracy alone could
never have separated those two, and only the counterfactual swap did.

`colab/contrast_context.py` implemented the fix -- decompose Y into a shared part and a cell-specific part,
freeze the shared part, train on the residual -- and was then VOID, because the cross-cell target it needs is
the one that turned out to be 96.5% censored. Genes jointly observed in all four lines for the same
perturbation: MEDIAN 3. `colab/dense_response.py` rebuilt that rectangle from the cell-level deposits at
median 6,550, and this module is what that rebuild was for.

    4 cells x 1,763 perturbations x 6,550 genes, 100% dense.

THE DECOMPOSITION, AND THE PROPERTY THAT MAKES THE TEST HONEST.

    Y[c,p,g] = mu[p,g] + delta[c,p,g],    mu = mean over the THREE TRAINING cells only

Because mu is the mean of the training cells, `sum_c delta[c,p,g] = 0 EXACTLY` over those cells. So on the
training target, any feature that depends on (p, g) alone -- gene identity, perturbation identity, the shared
response, the tide, anything cell-blind whatsoever -- has an optimal prediction of ZERO. There is no cell-blind
shortcut left to take. That is not a regularisation trick; it is arithmetic, and it is the whole reason this
objective can distinguish an ignored channel from a useless one where the absolute-response objective could not.

The held-out cell's delta is NOT zero-mean, because it is measured against three cells that exclude it. That
asymmetry is real and it caps what is achievable here; it is reported, not hidden.

WHAT IS AND IS NOT ALLOWED AS INPUT. The held-out cell contributes its BASELINE TRANSCRIPTOME and nothing
else. Its perturbation responses are never an input, never used to fit mu, and never used to standardise a
feature. Every cell-relative feature is centred on the training cells' mean, so a held-out feature is a
deviation from cells the model has seen -- which is exactly the quantity a cell encoder would have to read.

THE ARMS, each adding one thing so a gain is attributable:
    0  zero contrast                     predict delta = 0. The cell-blind optimum by construction.
    1  expression residual               baseline of the perturbed gene and the response gene IN THIS CELL,
                                         their deviations from the training cells, and mu as a scale
    2  relational residual               + baseline of the perturbed gene's REAL neighbours in this cell
    3  random-neighbour residual  CONTROL same feature count, same type mix, wrong partners
    4  shuffled-cell residual     CONTROL correct perturbation and genes, WRONG cell's baseline, in TRAINING
                                         as well as at test -- so the model is fit on a broken channel
    5  pairwise contrast                 predict Y[a,p,g] - Y[b,p,g] directly from (S_a, S_b, S_a - S_b)

THE BASELINE LADDER, because absolute-response baselines say nothing about a contrast:
    A  zero contrast              assume every cell responds identically. THE ONE THAT MATTERS MOST.
    C  expression difference only linear in the two baseline deviations, no network, no mu
    D  shared response x gate     mu scaled by how expressed the gene is here -- simple and biological
    E  nearest-cell copy          copy delta from the training cell with the closest baseline transcriptome
    F  shuffled context   CONTROL the cell-varying features permuted across rows at test time

    (B, cell-average sensitivity, is deliberately omitted: estimating a held-out cell's response scalar
     requires that cell's responses, which is the one thing the task forbids.)

SUCCESS CRITERIA, FIXED BEFORE THE RUN. The context branch is called usable only if ALL of:
    1  arm 2 beats zero contrast by more than the minimum detectable increment
    2  real neighbours beat random neighbours
    3  correct context beats SWAPPED context            <- the decisive one. It was ~0 in every prior arm.
    4  correct context beats SHUFFLED context
    5  criterion 3 holds in EVERY held-out cell, not just pooled
    6  the SIGN of the cell-specific deviation is predicted above chance
    7  adding the residual back to the frozen shared model improves total-response prediction

WHAT A NULL WOULD MEAN NOW, and it is much stronger than the previous null. The previous null established
only that the absolute-response objective gave the network no reason to look at the cell. A null here would
say that baseline transcriptomic state does not identify the cell-specific response function in these four
lines EVEN WHEN THE OBJECTIVE MAKES IT THE ONLY THING THAT CAN PAY. That points at protein or chromatin
state, or at more cell contexts -- not at more RNA architecture.

THE LIMIT NO AMOUNT OF DATA HERE FIXES. Tens of millions of gene x perturbation observations, but FOUR
independent cell contexts. Genes and perturbations buy precision WITHIN those four; they do not create new
cell states. This can establish whether context is readable and can fit a four-context correction. It cannot
establish a universal cell-state encoder, and a positive result is a reason to acquire more cell lines.
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
DENSE = SP / "dense_response.npz"
BASE = OUT / "baseline_state.json.gz"
NET = SP / "cell_complete.json.gz"

N_SAMPLE = int(os.environ.get("DC_N", "200000"))     # (p, g) pairs sampled per fold
EPOCHS = int(os.environ.get("DC_EPOCHS", "8"))
N_SEEDS = int(os.environ.get("DC_SEEDS", "3"))
SMOKE = os.environ.get("DC_SMOKE", "0") == "1"
if SMOKE:
    N_SAMPLE, EPOCHS, N_SEEDS = 20000, 2, 1


def main():
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("STAGE 3+4 -- context on a DENSE target, and the objective that makes it unavoidable")
    report("=" * 100)

    if not DENSE.exists():
        raise SystemExit(f"absent: {DENSE} -- run colab/dense_response.py first")
    z = np.load(DENSE, allow_pickle=True)
    Z = z["Z"]                                   # (cells, perts, genes) float16, per-gene robust z per line
    lines = [str(x) for x in z["lines"]]
    perts = [str(x) for x in z["perts"]]
    genes = [str(x) for x in z["genes"]]
    report(f"  dense rectangle: {len(lines)} cells x {len(perts):,} perturbations x {len(genes):,} genes")
    report(f"  the bench this replaces gave a MEDIAN OF 3 genes jointly observed per perturbation")

    # TWO DATA DEFECTS THE ZERO-MEAN ASSERTION BELOW CAUGHT, both of which would have silently poisoned a
    # correlation metric rather than crashing it.
    #   (a) 442 non-finite entries. A robust z is undefined where a gene's MAD is zero within a line.
    #   (b) values reaching |z| = 357, from genes whose MAD is near zero rather than exactly zero. Pearson r
    #       on 200,000 sampled points is decided by a handful of those. They are an artefact of the scale
    #       estimator, not a 357-sigma biological response.
    # Genes carrying any non-finite value are dropped outright; the rest are clipped, and both counts are
    # printed so the reader can judge the surgery instead of trusting it.
    bad_g = ~np.isfinite(Z.astype(np.float32)).all(0).all(0)
    if bad_g.any():
        Z = Z[:, :, ~bad_g]
        genes = [g for g, b in zip(genes, bad_g) if not b]
    ZCLIP = 10.0
    Zf = Z.astype(np.float32)
    n_clip = int((np.abs(Zf) > ZCLIP).sum())
    Z = np.clip(Zf, -ZCLIP, ZCLIP)
    report(f"  dropped {int(bad_g.sum()):,} genes with an undefined robust z; clipped {n_clip:,} values "
           f"({100*n_clip/Z.size:.4f}%) at |z| = {ZCLIP:.0f}; max |z| was {np.abs(Zf).max():.0f}")
    del Zf

    prof = json.load(gzip.open(BASE, "rt"))["profiles"]
    gi = {g: i for i, g in enumerate(genes)}
    keep_g = [i for i, g in enumerate(genes) if all(g in prof[ln] for ln in lines)]
    Z = Z[:, :, keep_g]
    genes = [genes[i] for i in keep_g]
    gi = {g: i for i, g in enumerate(genes)}
    B = np.array([[prof[ln][g] for g in genes] for ln in lines], np.float32)     # (cells, genes)
    report(f"  genes with a baseline in all {len(lines)} lines: {len(genes):,}")

    # perturbation-level baseline: the perturbed gene's own expression. A perturbation whose target is not
    # in the shared gene set gets the line's median, and is counted so the substitution is visible.
    pmiss = 0
    Bp = np.zeros((len(lines), len(perts)), np.float32)
    for j, p in enumerate(perts):
        if p in gi:
            Bp[:, j] = B[:, gi[p]]
        else:
            pmiss += 1
            Bp[:, j] = np.median(B, axis=1)
    report(f"  perturbations whose target gene has no baseline here: {pmiss:,}/{len(perts):,} "
           f"(given the line median)")

    d = json.load(gzip.open(NET, "rt"))
    nmz = [g["name"] for g in d["genes"]]
    nbrs = {}
    for a_, v in (d.get("codep") or {}).items():
        nbrs.setdefault(nmz[int(a_)], []).extend(nmz[int(b)] for b, _s in v[:7])
    g2c = {int(k_): set(v) for k_, v in (d.get("gene2cplx") or {}).items()}
    bym = {}
    for x, cs in g2c.items():
        for c in cs:
            bym.setdefault(c, []).append(x)
    for c, mem in bym.items():
        for x in mem[:8]:
            nbrs.setdefault(nmz[x], []).extend(nmz[y] for y in mem[:8] if y != x)
    for e in d["ppi"]:
        nbrs.setdefault(nmz[int(e[0])], []).append(nmz[int(e[1])])
        nbrs.setdefault(nmz[int(e[1])], []).append(nmz[int(e[0])])
    del d

    rng0 = np.random.default_rng(0)
    NB = np.zeros((len(lines), len(perts)), np.float32)       # real neighbours
    NBr = np.zeros((len(lines), len(perts)), np.float32)      # random partners, matched count
    nb_hit = 0
    for j, p in enumerate(perts):
        idx = [gi[x] for x in dict.fromkeys(nbrs.get(p, [])) if x in gi][:21]
        if idx:
            nb_hit += 1
            NB[:, j] = B[:, idx].mean(1)
            ridx = rng0.integers(0, len(genes), len(idx))
            NBr[:, j] = B[:, ridx].mean(1)
        else:
            NB[:, j] = NBr[:, j] = B.mean(1)
    report(f"  perturbations with at least one typed neighbour carrying a baseline: {nb_hit:,}/{len(perts):,}")

    def fold(h, seed):
        """One held-out cell, one seed. Returns a dict of arm -> metrics."""
        tr_c = [c for c in range(len(lines)) if c != h]
        rng = np.random.default_rng(1000 * seed + h)

        MU = Z[tr_c].mean(0)                     # (perts, genes) shared response
        # training-cell means for every cell-relative feature. A held-out feature is a deviation from cells
        # the model has actually seen -- the held-out cell never contributes to its own reference.
        Bm, Bpm, NBm, NBrm = B[tr_c].mean(0), Bp[tr_c].mean(0), NB[tr_c].mean(0), NBr[tr_c].mean(0)

        pi = rng.integers(0, len(perts), N_SAMPLE)
        gj = rng.integers(0, len(genes), N_SAMPLE)
        mu = MU[pi, gj]

        def feats(c, rand_nb=False, force_pair=None):
            src = c if force_pair is None else force_pair          # which cell's BASELINE is read
            xg, xp = B[src, gj], Bp[src, pi]
            dxg, dxp = xg - Bm[gj], xp - Bpm[pi]
            nb_ = (NBr if rand_nb else NB)[src, pi]
            dnb = nb_ - (NBrm if rand_nb else NBm)[pi]
            return np.stack([xg, xp, dxg, dxp, mu, mu * dxg, mu * dxp, nb_, dnb, mu * dnb], 1)

        SIMPLE = [0, 1, 2, 3, 4, 5, 6]        # arm 1: no neighbourhood columns
        ALL = list(range(10))

        def target(c):
            return Z[c, pi, gj] - mu

        Xtr = {k: np.concatenate([feats(c, rand_nb=(k == "rand")) for c in tr_c]) for k in ("real", "rand")}
        # arm 4 is trained on a broken channel, not merely tested on one: each training cell is handed a
        # different training cell's baseline throughout.
        Xtr["swapcell"] = np.concatenate([feats(c, force_pair=tr_c[(i + 1) % len(tr_c)])
                                          for i, c in enumerate(tr_c)])
        ytr = np.concatenate([target(c) for c in tr_c])
        assert abs(float(ytr.mean())) < 1e-3, "training delta must be ~zero-mean by construction"

        yte = target(h)
        Xte = {"real": feats(h), "rand": feats(h, rand_nb=True)}
        # THE COUNTERFACTUAL SWAP: the model trained on the correct channel is handed a TRAINING cell's
        # baseline at test time. mu is cell-independent and is left alone, so only the cell channel moves.
        Xte["swap"] = feats(h, force_pair=tr_c[0])
        Xte["swap2"] = feats(h, force_pair=tr_c[1])
        sh = rng.permutation(N_SAMPLE)
        Xsh = Xte["real"].copy()
        cellcols = [0, 1, 2, 3, 7, 8]                       # every column that varies with the cell
        Xsh[:, cellcols] = Xsh[sh][:, cellcols]
        Xsh[:, 5] = mu * Xsh[:, 2]
        Xsh[:, 6] = mu * Xsh[:, 3]
        Xsh[:, 9] = mu * Xsh[:, 8]
        Xte["shuffled"] = Xsh

        m, s = Xtr["real"].mean(0), Xtr["real"].std(0) + 1e-6

        def fit(Xt, yt, cols, sd):
            torch.manual_seed(sd)
            net = nn.Sequential(nn.Linear(len(cols), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                nn.Linear(64, 1))
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
            Xn = torch.from_numpy(((Xt - m) / s)[:, cols].astype(np.float32))
            Y = torch.from_numpy(yt.astype(np.float32)).unsqueeze(1)
            n = len(Xn)
            for _e in range(EPOCHS):
                pm = torch.randperm(n)
                for i0 in range(0, n, 4096):
                    b = pm[i0:i0 + 4096]
                    loss = (net(Xn[b]) - Y[b]).pow(2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return net

        def pred(net, X, cols):
            with torch.no_grad():
                return net(torch.from_numpy(((X - m) / s)[:, cols].astype(np.float32))).squeeze(1).numpy()

        def score(p_):
            r = float(np.corrcoef(p_, yte)[0, 1]) if np.std(p_) > 1e-9 else 0.0
            big = np.abs(yte) >= np.median(np.abs(yte))
            sgn = float((np.sign(p_[big]) == np.sign(yte[big])).mean())
            tot = float(np.corrcoef(mu + p_, mu + yte)[0, 1])
            return {"r": r, "sign": sgn, "total_r": tot}

        out = {}
        out["0 zero contrast"] = {"r": 0.0, "sign": 0.5,
                                  "total_r": float(np.corrcoef(mu, mu + yte)[0, 1])}
        n1 = fit(Xtr["real"], ytr, SIMPLE, seed)
        out["1 expression residual"] = score(pred(n1, Xte["real"], SIMPLE))
        n2 = fit(Xtr["real"], ytr, ALL, seed)
        p2 = pred(n2, Xte["real"], ALL)
        out["2 relational residual"] = score(p2)
        n3 = fit(Xtr["rand"], ytr, ALL, seed)
        out["3 random neighbours (CONTROL)"] = score(pred(n3, Xte["rand"], ALL))
        n4 = fit(Xtr["swapcell"], ytr, ALL, seed)
        out["4 shuffled-cell training (CONTROL)"] = score(pred(n4, Xte["real"], ALL))
        out["2 SWAPPED context (CONTROL)"] = score(pred(n2, Xte["swap"], ALL))
        out["2 SWAPPED context alt (CONTROL)"] = score(pred(n2, Xte["swap2"], ALL))
        out["2 SHUFFLED context (CONTROL)"] = score(pred(n2, Xte["shuffled"], ALL))

        # arm 5, pairwise contrast: predict the difference between two named cells directly.
        b_ = tr_c[0]
        yp = Z[h, pi, gj] - Z[b_, pi, gj]
        Xp_tr = np.concatenate([np.concatenate([feats(a), feats(c), feats(a) - feats(c)], 1)
                                for a, c in ((tr_c[0], tr_c[1]), (tr_c[1], tr_c[2]), (tr_c[2], tr_c[0]))])
        yp_tr = np.concatenate([Z[a, pi, gj] - Z[c, pi, gj]
                                for a, c in ((tr_c[0], tr_c[1]), (tr_c[1], tr_c[2]), (tr_c[2], tr_c[0]))])
        mp, sp = Xp_tr.mean(0), Xp_tr.std(0) + 1e-6
        torch.manual_seed(seed)
        netp = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        optp = torch.optim.Adam(netp.parameters(), lr=3e-3, weight_decay=1e-5)
        Xn = torch.from_numpy(((Xp_tr - mp) / sp).astype(np.float32))
        Y = torch.from_numpy(yp_tr).unsqueeze(1)
        for _e in range(EPOCHS):
            pm = torch.randperm(len(Xn))
            for i0 in range(0, len(Xn), 4096):
                b = pm[i0:i0 + 4096]
                loss = (netp(Xn[b]) - Y[b]).pow(2).mean()
                optp.zero_grad()
                loss.backward()
                optp.step()
        Xp_te = np.concatenate([feats(h), feats(b_), feats(h) - feats(b_)], 1)
        with torch.no_grad():
            pp = netp(torch.from_numpy(((Xp_te - mp) / sp).astype(np.float32))).squeeze(1).numpy()
        big = np.abs(yp) >= np.median(np.abs(yp))
        out["5 pairwise contrast"] = {
            "r": float(np.corrcoef(pp, yp)[0, 1]), "total_r": float("nan"),
            "sign": float((np.sign(pp[big]) == np.sign(yp[big])).mean())}

        # THE BASELINE LADDER
        def lin(cols):
            Xa = np.c_[Xtr["real"][:, cols], np.ones(len(ytr))]
            w = np.linalg.lstsq(Xa, ytr, rcond=None)[0]
            return score(np.c_[Xte["real"][:, cols], np.ones(N_SAMPLE)] @ w)
        out["C expression difference only"] = lin([2, 3])
        out["D shared x expression gate"] = lin([4, 5])
        near = tr_c[int(np.argmin([np.linalg.norm(B[h] - B[c]) for c in tr_c]))]
        out["E nearest-cell copy"] = score(Z[near, pi, gj] - mu)
        out["F shuffled context (CONTROL)"] = out["2 SHUFFLED context (CONTROL)"]
        return out

    res = {}
    for h, ln in enumerate(lines):
        report(f"\n  HELD-OUT CELL: {ln}  (train on {', '.join(l for i, l in enumerate(lines) if i != h)})")
        for seed in range(N_SEEDS):
            o = fold(h, seed)
            for k_, v in o.items():
                res.setdefault(ln, {}).setdefault(k_, []).append(v)
        for k_ in res[ln]:
            v = res[ln][k_]
            report(f"    {k_:36s} r {np.mean([x['r'] for x in v]):+.4f}  "
                   f"sign {np.mean([x['sign'] for x in v]):.4f}  "
                   f"total_r {np.mean([x['total_r'] for x in v]):+.4f}")

    def agg(k_, m="r"):
        return float(np.mean([np.mean([x[m] for x in res[ln][k_]]) for ln in lines]))

    arms = list(res[lines[0]])
    report(f"\n  {'arm (mean over 4 held-out cells)':36s} {'r':>8s} {'sign':>8s} {'total_r':>9s}")
    for k_ in arms:
        report(f"  {k_:36s} {agg(k_):+8.4f} {agg(k_,'sign'):8.4f} {agg(k_,'total_r'):+9.4f}")

    sd = float(np.mean([np.std([x["r"] for x in res[ln][k_]]) for ln in lines for k_ in arms]))
    mde = 3 * sd / np.sqrt(N_SEEDS)
    # the sign metric lives on a different scale from r and needs its own detectable floor
    sd_s = float(np.mean([np.std([x["sign"] for x in res[ln][k_]]) for ln in lines for k_ in arms]))
    mde_s = 3 * sd_s / np.sqrt(N_SEEDS)
    swap_gap = agg("2 relational residual") - max(agg("2 SWAPPED context (CONTROL)"),
                                                  agg("2 SWAPPED context alt (CONTROL)"))
    per_cell_swap = {ln: float(np.mean([x["r"] for x in res[ln]["2 relational residual"]])
                               - max(np.mean([x["r"] for x in res[ln]["2 SWAPPED context (CONTROL)"]]),
                                     np.mean([x["r"] for x in res[ln]["2 SWAPPED context alt (CONTROL)"]])))
                     for ln in lines}
    crit = {
        "1 beats zero contrast": agg("2 relational residual") > mde,
        "2 real > random neighbours": agg("2 relational residual") - agg("3 random neighbours (CONTROL)") > mde,
        "3 correct > SWAPPED context": swap_gap > mde,
        "4 correct > SHUFFLED context":
            agg("2 relational residual") - agg("2 SHUFFLED context (CONTROL)") > mde,
        "5 swap gap holds in EVERY cell": all(v > 0 for v in per_cell_swap.values()),
        "6 sign above chance": agg("2 relational residual", "sign") - 0.5 > mde_s,
        "7 improves total response": agg("2 relational residual", "total_r") - agg("0 zero contrast", "total_r") > mde,
    }
    report(f"\n  pooled seed sd {sd:.4f} -> minimum detectable increment {mde:+.4f} at {N_SEEDS} seeds "
           f"(sign: sd {sd_s:.4f} -> MDE {mde_s:+.4f})")
    report("\n  PREDECLARED CRITERIA")
    for k_, v in crit.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k_}")
    report(f"  per-cell swap gap: " + ", ".join(f"{k_} {v:+.4f}" for k_, v in per_cell_swap.items()))

    npass = int(sum(bool(v) for v in crit.values()))
    # THE LADDER EXISTS BECAUSE BEATING ZERO IS NOT THE SAME AS BEING WORTH BUILDING. Reported next to the
    # model rather than buried: if the best rung beats arm 2, the architecture is not what produced the gain.
    ladder = {k_: agg(k_) for k_ in arms if k_[0] in "CDE"}
    best_rung = max(ladder, key=ladder.get)
    verdict = (
        f"{'CONTRAST TRAINING WORKS' if npass == 7 else f'{npass}/7 CRITERIA MET'}: on a 100%-dense 4-cell "
        f"rectangle with the shared response frozen out by construction, the relational residual reaches "
        f"r {agg('2 relational residual'):+.4f} against a zero-contrast baseline of 0.0000 and a minimum "
        f"detectable increment of {mde:.4f}. THE DECISIVE NUMBER is the counterfactual swap: handing the "
        f"trained model another cell's baseline changes r by {swap_gap:+.4f}. In every previous experiment "
        f"in this project that quantity was between -0.0007 and +0.0004 -- indistinguishable from the "
        f"channel not existing. {'It is now separated from zero, which means the cell channel is being READ' if crit['3 correct > SWAPPED context'] else 'It is STILL not separated from zero, which means the objective change did not make the channel readable and the limit is the information in baseline RNA, not the loss'}. "
        f"Real neighbours vs random partners: {agg('2 relational residual') - agg('3 random neighbours (CONTROL)'):+.4f}. "
        f"THE LADDER: the best non-learned rung is '{best_rung}' at r {ladder[best_rung]:+.4f} against the "
        f"model's {agg('2 relational residual'):+.4f} -- "
        f"{'the model is ahead of it' if agg('2 relational residual') > ladder[best_rung] else 'A TRIVIAL BASELINE MATCHES OR BEATS THE MODEL, so the architecture is not what produced the gain'}. "
        f"Four independent cell contexts is the binding limit and no result here generalises past them.")
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {"model": "dense-context-v1", "lines": lines, "n_perturbations": len(perts), "n_genes": len(genes),
         "density": 1.0, "n_sampled_pairs_per_fold": N_SAMPLE, "seeds": N_SEEDS,
         "target": "delta = Y - mean over the THREE TRAINING cells; zero-mean by construction, so no "
                   "cell-blind feature can beat 0",
         "per_cell": {ln: {k_: {m: (lambda z_: float(z_) if np.isfinite(z_) else None)(
                                       np.mean([x[m] for x in v])) for m in ("r", "sign", "total_r")}
                           for k_, v in res[ln].items()} for ln in lines},
         "pooled": {k_: {m: (float(agg(k_, m)) if np.isfinite(agg(k_, m)) else None)
                         for m in ("r", "sign", "total_r")} for k_ in arms},
         "baseline_ladder": ladder, "best_rung": best_rung,
         "model_beats_best_rung": bool(agg("2 relational residual") > ladder[best_rung]),
         "mde": mde, "swap_gap": swap_gap, "per_cell_swap_gap": per_cell_swap,
         "criteria": {k_: bool(v) for k_, v in crit.items()}, "criteria_met": npass,
         "limits": [
             "FOUR independent cell contexts. Genes and perturbations buy precision inside them; they do "
             "not create new cell states. This cannot establish a universal cell-state encoder",
             "the held-out cell's delta is measured against three cells that exclude it, so it carries a "
             "systematic offset no model trained on the other three can learn. That caps the achievable r",
             "context is baseline RNA only -- accessibility and protein are absent for these four lines. A "
             "null is a null for RNA state, not for cell state",
             "K562 is not re-derived while the other three are pseudobulked here; the processing asymmetry "
             "lies along the cell axis, which is the axis measured. See dense_response.json for the "
             "cross-line variance correlations that make it inspectable",
             "the perturbation set is the ~1,763 knockouts screened in all four lines, which is a biased "
             "subset: perturbations chosen to be screened everywhere are not a random sample"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "dense_context.json", "w"), indent=1)
    report(f"\n  -> {OUT/'dense_context.json'}")


if __name__ == "__main__":
    main()
