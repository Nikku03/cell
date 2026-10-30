"""THE SIGN DEFECT: the pipeline discards direction, so it cannot do the one thing we proved works.

TWO OF TODAY'S RESULTS COLLIDE.

    Norman (measured)     response(A+B) ~ response(A) + response(B), at 82-87% of the noise floor,
                          +0.4326 over scrambled, ROBUST 6/6. Effects ADD.
    the ADRN (built)      `Aabs = np.abs(z["M"])`. The NMF, the ridge, and the target are all MAGNITUDES.

**You cannot add magnitudes.** |a| + |b| != |a+b| the moment the signs differ, and cancellation is exactly where
composition carries information: a gene that goes up under one perturbation and down under another is telling you
something that |z| erases. We measured that additive composition works and then trained a model that is
arithmetically incapable of it.

This is not a hypothesis about biology. It is a defect, and the line of code is above.

THE FIX, and why it keeps NMF usable. NMF needs a non-negative matrix, which is presumably why abs() was reached
for. Splitting each gene into two non-negative halves keeps that property and loses nothing:

    signed profile x  ->  [max(x, 0), max(-x, 0)]        2 x 8,246 columns, all >= 0

An up-move and a down-move now occupy different columns, so the factorisation can represent "this programme
pushes gene G down" as distinct from "pushes it up", and two profiles that cancel no longer look identical.
Reconstruction folds back: signed = up_half - down_half.

ARMS, all on the same sealed cohorts, same channels, same penalty grid tuned on a train-only split:

    abs        the incumbent. NMF on |M|.
    signed     NMF on the split representation, folded back before ranking.
    abs_ORACLE     retrieval by |response| cosine  -- the ceiling the lookup test measured (+0.2128 over annot)
    signed_ORACLE  retrieval by SIGNED response cosine

The two oracles are the diagnostic that makes this interpretable whichever way the model arms go. They ask
whether direction carries RETRIEVABLE information at all, independently of whether our model can use it:

    signed_ORACLE > abs_ORACLE   direction distinguishes genes that magnitude confuses. The information exists.
    signed_ORACLE ~ abs_ORACLE   for this task, sign carries nothing, abs() cost us nothing, and the defect is
                                 real but harmless -- which would be worth knowing before building on it.

PREDECLARED, before any number:
    signed > abs, resolved in a majority   -> the abs() was costing real accuracy and should be reverted.
    signed ~ abs                           -> the representation is not the constraint here; report it as a
                                              defect that turned out not to bind, and do not claim a fix.
    signed < abs                           -> splitting doubles the output width and halves the data per column;
                                              that cost exceeds what direction buys at this sample size.

Swept over basis rank x prediction depth on robustness.Sweeper.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
K_VALUES = (30, 60, 120)
NPREDS = (10, 20, 50)
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
KNN = (1, 5, 10)


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("THE SIGN DEFECT: does discarding direction cost anything?")
    report("=" * 100)
    report("  PREDECLARED: signed > abs => abs() was costing accuracy, revert it.")
    report("               signed ~ abs => a real defect that does not bind. Do not claim a fix.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gj = {g: j for j, g in enumerate(genes)}
    M = z["M"].astype(np.float32)
    Aabs = np.abs(M)
    tide = (Aabs >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag or "cohort1"] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    train_rows = np.array([ki[k] for k in train])
    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}
    tr_u = np.array([urow[k] for k in train])
    report(f"\n  {len(kos):,} perturbations x {len(genes):,} genes; sealed {len(sealed)}; train {len(train):,}")

    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1).astype(np.float32)

    # how much sign is there to lose? a diagnostic, printed before any arm runs
    strong = Aabs >= A.TAU
    ndn = int((M[strong] < 0).sum())
    report(f"  of {int(strong.sum()):,} strong moves, {ndn:,} are DOWN ({100*ndn/max(int(strong.sum()),1):.1f}%) "
           f"-- this is the information abs() erases")

    Xabs = Aabs[train_rows].copy()
    Xabs[:, tide] = 0.0
    Msig = M.copy()
    Msig[:, tide] = 0.0
    Xsig = np.concatenate([np.maximum(Msig[train_rows], 0), np.maximum(-Msig[train_rows], 0)], 1)
    report(f"  abs matrix {Xabs.shape} | signed split matrix {Xsig.shape}")

    freq = (Aabs[train_rows] >= A.TAU).mean(0)
    freq[tide] = -np.inf

    def hit(v, movers, n, k):
        v = np.abs(v).astype(np.float64)
        v[tide] = -np.inf
        if k in gj:
            v[gj[k]] = -np.inf
        return len({genes[t] for t in np.argsort(-v)[:n]} & set(movers)) / float(n)

    from sklearn.decomposition import NMF

    def fit_pipeline(X, K):
        nmf = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=0)
        W = nmf.fit_transform(X).astype(np.float32)
        H = nmf.components_.astype(np.float32)
        rv = np.random.default_rng(A.SEED + 4).permutation(len(tr_u))
        nv = max(50, len(rv) // 5)
        bv, bp = -1e18, PENALTIES[0]
        for pen in PENALTIES:
            w = A.ridge_fit(chan[tr_u[rv[nv:]]], W[rv[nv:]], pen)
            e = -float(np.mean((A.ridge_apply(chan[tr_u[rv[:nv]]], w) - W[rv[:nv]]) ** 2))
            if e > bv:
                bv, bp = e, pen
        return A.ridge_fit(chan[tr_u], W, bp), H, bp

    # oracles: retrieval by response similarity, magnitude vs signed
    Rabs = Aabs.copy()
    Rabs[:, tide] = 0.0
    Rabs_tr = Rabs[train_rows]
    Rsig_tr = Msig[train_rows]
    nrm = lambda Q: Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    Rabs_n, Rsig_n = nrm(Rabs_tr), nrm(Rsig_tr)

    sw = Sweeper("signed - abs (model)", axes={"K": list(K_VALUES), "npred": list(NPREDS)})
    swo = Sweeper("signed ORACLE - abs ORACLE", axes={"K": list(K_VALUES), "npred": list(NPREDS)})
    cells = {}
    for K in K_VALUES:
        wa, Ha, pa = fit_pipeline(Xabs, K)
        ws, Hs, ps = fit_pipeline(Xsig, K)
        report(f"\n  K={K}: abs penalty {pa} | signed penalty {ps}")
        for npred in NPREDS:
            arms = {}
            for tag, ans in answers.items():
                keys = [k for k in sorted(ans) if k in urow]
                Ca = chan[[urow[k] for k in keys]]
                Pa = A.ridge_apply(Ca, wa) @ Ha
                Ps_ = A.ridge_apply(Ca, ws) @ Hs
                Ps = Ps_[:, :len(genes)] - Ps_[:, len(genes):]      # fold the split halves back to signed
                Qa = nrm(np.array([Rabs[ki[k]] for k in keys], np.float32))
                Qs = nrm(np.array([Msig[ki[k]] for k in keys], np.float32))
                Sa, Ss = Qa @ Rabs_n.T, Qs @ Rsig_n.T
                for i, k in enumerate(keys):
                    mv = ans[k]["movers"]
                    arms.setdefault("frequency", []).append(hit(freq.copy(), mv, npred, k))
                    arms.setdefault("abs", []).append(hit(Pa[i], mv, npred, k))
                    arms.setdefault("signed", []).append(hit(Ps[i], mv, npred, k))
                    for kk in KNN:
                        ia = np.argsort(-Sa[i])[:kk]
                        arms.setdefault(f"abs_ORACLE{kk}", []).append(hit(Rabs_tr[ia].mean(0), mv, npred, k))
                        isg = np.argsort(-Ss[i])[:kk]
                        arms.setdefault(f"signed_ORACLE{kk}", []).append(hit(Rsig_tr[isg].mean(0), mv, npred, k))
            arms = {n: np.array(v) for n, v in arms.items()}
            ba = max(KNN, key=lambda kk: arms[f"abs_ORACLE{kk}"].mean())
            bs = max(KNN, key=lambda kk: arms[f"signed_ORACLE{kk}"].mean())
            cfg, tg = {"K": K, "npred": npred}, f"K{K}|n{npred}"
            sw.add(cfg, tg, arms["signed"] - arms["abs"])
            swo.add(cfg, tg, arms[f"signed_ORACLE{bs}"] - arms[f"abs_ORACLE{ba}"])
            cells[tg] = {n: float(v.mean()) for n, v in arms.items()}
            report(f"    N={npred:>2}: freq {arms['frequency'].mean():.4f} | abs {arms['abs'].mean():.4f} | "
                   f"signed {arms['signed'].mean():.4f} | absORACLE {arms[f'abs_ORACLE{ba}'].mean():.4f} | "
                   f"sigORACLE {arms[f'signed_ORACLE{bs}'].mean():.4f}")

    for nm, s in (("model", sw), ("oracle", swo)):
        report(s.report())
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    gm = float(np.mean([c["signed"] - c["abs"] for c in cells.values()]))
    npos = sum(1 for c in sw._cells.values() if c["lo"] > 0)
    opos = sum(1 for c in swo._cells.values() if c["lo"] > 0)
    report(f"\n  signed - abs (model): {gm:+.4f}, resolved positive in {npos}/{len(sw._cells)}")
    report("\n  READING")
    if npos > len(sw._cells) / 2:
        report(f"  Keeping the sign is worth {gm:+.4f}. The abs() was costing real accuracy and the whole")
        report("  pipeline should be rebuilt signed -- which also makes additive composition expressible.")
    else:
        report(f"  Keeping the sign is worth {gm:+.4f}, not resolved. The defect is real -- the model genuinely")
        report("  cannot represent the additive composition Norman measured -- but on THIS task, ranking the")
        report("  top-N movers, it does not bind. Report it as a defect that does not cost anything here, and")
        report("  do not claim a fix. Where it would bind is any use that needs to ADD two predicted responses.")
    if opos > len(swo._cells) / 2:
        report("  The SIGNED oracle beats the magnitude oracle: direction distinguishes genes that magnitude")
        report("  confuses, so the information is retrievable in principle even if our model does not use it.")
    else:
        report("  The signed oracle does not beat the magnitude oracle -- for retrieval on this task, direction")
        report("  carries nothing extra, and abs() was not throwing away anything findable.")

    json.dump({"test": "adrn_signed", "K_values": list(K_VALUES), "npreds": list(NPREDS),
               "pct_down": 100 * ndn / max(int(strong.sum()), 1), "cells": cells,
               "gap_signed_minus_abs": gm,
               "sweeps": {"model": sw.verdict(), "oracle": swo.verdict()}, "log": log},
              open(OUT / "adrn_signed.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_signed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
