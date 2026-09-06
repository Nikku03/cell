"""FEW-SHOT CROSS-LINE: given N knockouts from a new cell line, does K562 pretraining beat training from scratch?

TWO FIXES TO THE ZERO-SHOT TEST, and the first is a bug in my own protocol rather than a new idea.

1. PER-LINE TIDE.  The K562 sealed protocol excludes genes that move in >=5% of knockouts -- the "tide", the
   shared response every perturbation produces.  adrn_crossline removed K562's tide from RPE1's answer key, which
   is the wrong tide: RPE1's own shared programme stayed in the key, and `freq_line` scored 0.6305 by predicting
   it.  Each line's tide is now estimated from that line's own training knockouts and removed from the key and
   from every arm's predictions alike, exactly as K562 does.  This makes the task consistent with the protocol the
   rest of the project uses, and it is the single largest reason the previous numbers looked the way they did.

2. FEW-SHOT INSTEAD OF ZERO-SHOT.  Conditioning on a cell's state requires data from that cell, so it is few-shot
   by definition; pretending otherwise would be dishonest framing.  The practical question is therefore not "can
   K562 alone predict RPE1" (answered: no) but **"given N knockouts from RPE1, is a K562-pretrained model better
   than one trained on those N alone?"**  That is what transferable biology would buy, and it is the number
   anybody deciding whether to reuse this model would want.

ARMS at each (line, N)

    scratch_N     NMF + ridge fitted on the N target-line knockouts only.  No K562 at all.
    k562_plus_N   K562 programmes, with the N target-line profiles projected onto that basis and added to the
                  ridge's training rows.  The K562 model used as a PRIOR.
    freq_N        the most frequent movers among the N samples.  A lookup table needing no model.
    within_full   fitted on ALL the line's training knockouts.  The ceiling, not a competitor.

PREDECLARED GATE: `k562_plus_N` must beat `scratch_N` past a bootstrap CI. That is the whole claim -- if K562
pretraining does not help when target-line data is scarce, the model carries nothing transferable and every
cross-line use is better served by measuring the line.

Reported alongside: k562_plus_N vs freq_N, because beating scratch while losing to a frequency table would be a
hollow pass; and the N at which scratch overtakes the prior, which is the practically useful number.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
LINES = ("RPE1", "Jurkat", "HepG2")
SHOTS = (25, 100, 400)
K_FIT = 60
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
N_TEST, TIDE_FRAC = 200, 0.05


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("FEW-SHOT CROSS-LINE: does K562 pretraining beat scratch, given N knockouts from a new line?")
    report("=" * 100)
    report("  PREDECLARED: k562_plus_N must beat scratch_N. Per-line tide removed from key AND predictions.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    gi_all = {g: i for i, g in enumerate(genes)}
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)

    sealed = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    k562_train = [k for k in kos if k not in sealed]

    prof = {ln: pickle.load(open(SP / f"nlz_{ln}.pkl", "rb")) for ln in LINES}
    universe = sorted(set(kos) | sealed | {g for d in prof.values() for g in d})
    urow = {g: i for i, g in enumerate(universe)}
    base, _ = A.build_channels(universe, set(k562_train), report)
    extra, _, tb = C.extra_channels(universe, set(k562_train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)

    from sklearn.decomposition import NMF

    def fit_ridge(X, W):
        rng = np.random.default_rng(A.SEED + 5)
        p = rng.permutation(len(X))
        nv = max(10, int(0.2 * len(X)))
        best, bp = -1e18, PENALTIES[0]
        for pen in PENALTIES:
            w = A.ridge_fit(X[p[nv:]], W[p[nv:]], pen)
            e = -float(np.mean((A.ridge_apply(X[p[:nv]], w) - W[p[:nv]]) ** 2))
            if e > best:
                best, bp = e, pen
        return A.ridge_fit(X, W, bp)

    sw = Sweeper("k562_plus_N - scratch_N", axes={"line": list(LINES), "N": list(SHOTS)})
    sw_freq = Sweeper("k562_plus_N - freq_N", axes={"line": list(LINES), "N": list(SHOTS)})
    means = {}

    for line in LINES:
        d = prof[line]
        vocab = sorted({t for p in d.values() for t in p} & set(genes))
        vix = {g: i for i, g in enumerate(vocab)}
        kcol = np.array([gi_all[g] for g in vocab])
        lk = sorted(d)
        M = np.zeros((len(lk), len(vocab)), np.float32)
        for r, g in enumerate(lk):
            for t, v in d[g].items():
                j = vix.get(t)
                if j is not None:
                    M[r, j] = float(v)
        rng = np.random.default_rng(A.SEED)
        tix = rng.choice(len(lk), N_TEST, replace=False)
        pool = np.setdiff1d(np.arange(len(lk)), tix)
        test_keys = [lk[i] for i in tix]

        # PER-LINE TIDE from the line's own training pool -- the fix. Defines the task; no arm is given it.
        tide_L = (np.abs(M[pool]) > 0).mean(0) >= TIDE_FRAC
        truth = {lk[i]: set(np.where((np.abs(M[i]) > 0) & ~tide_L)[0]) for i in tix}
        nmov = np.mean([len(v) for v in truth.values()])
        report(f"\n  === {line}: {len(vocab):,} genes, tide {int(tide_L.sum()):,} removed, "
               f"{nmov:.0f} movers left per held-out knockout ===")
        if nmov < 5:
            report("  SKIP: fewer than 5 non-tide movers survive -- the key would be noise")
            continue

        Xk = Amat[[ki[k] for k in k562_train]][:, kcol].copy()
        Xk[:, tide_L] = 0.0
        nk = NMF(n_components=K_FIT, init="nndsvda", max_iter=300, random_state=0)
        Wk = nk.fit_transform(Xk).astype(np.float32)
        Hk = nk.components_.astype(np.float32)
        Hpinv = np.linalg.pinv(Hk)

        Mfull = np.abs(M[pool]).copy()
        Mfull[:, tide_L] = 0.0
        nf = NMF(n_components=K_FIT, init="nndsvda", max_iter=300, random_state=0)
        Wf = nf.fit_transform(Mfull).astype(np.float32)
        Hf = nf.components_.astype(np.float32)
        w_full = fit_ridge(chan[[urow[lk[i]] for i in pool]], Wf)

        def prec(scores):
            out = []
            for j, k in enumerate(test_keys):
                s = scores[j].copy()
                s[tide_L] = -np.inf
                out.append(len(set(np.argsort(-s)[:A.NPRED]) & truth[k]) / float(A.NPRED))
            return np.array(out)

        Ct = chan[[urow[k] for k in test_keys]]
        p_full = prec(A.ridge_apply(Ct, w_full) @ Hf)

        for N in SHOTS:
            if N > len(pool):
                continue
            shot = rng.choice(pool, N, replace=False)
            Ms = np.abs(M[shot]).copy()
            Ms[:, tide_L] = 0.0

            kk = max(5, min(K_FIT, N // 3))
            ns = NMF(n_components=kk, init="nndsvda", max_iter=300, random_state=0)
            Ws = ns.fit_transform(Ms).astype(np.float32)
            Hs = ns.components_.astype(np.float32)
            w_s = fit_ridge(chan[[urow[lk[i]] for i in shot]], Ws)
            p_scratch = prec(A.ridge_apply(Ct, w_s) @ Hs)

            # K562 prior: project the N line profiles onto the K562 basis and append as extra ridge rows
            Wl_on_k = (Ms @ Hpinv).astype(np.float32)
            Xrows = np.concatenate([chan[[urow[k] for k in k562_train]],
                                    chan[[urow[lk[i]] for i in shot]]], 0)
            Wrows = np.concatenate([Wk, Wl_on_k], 0)
            w_kp = fit_ridge(Xrows, Wrows)
            p_kplus = prec(A.ridge_apply(Ct, w_kp) @ Hk)

            fq = Ms.mean(0)
            fq[tide_L] = -np.inf
            top = np.argsort(-fq)[:A.NPRED]
            p_freq = np.array([len(set(top) & truth[k]) / float(A.NPRED) for k in test_keys])

            sw.add({"line": line, "N": N}, line, p_kplus - p_scratch)
            sw_freq.add({"line": line, "N": N}, line, p_kplus - p_freq)
            means[f"{line}|N={N}"] = {"k562_plus_N": float(p_kplus.mean()),
                                      "scratch_N": float(p_scratch.mean()),
                                      "freq_N": float(p_freq.mean()),
                                      "within_full": float(p_full.mean())}
            report(f"    N={N:>4}  k562_plus {p_kplus.mean():.4f} | scratch {p_scratch.mean():.4f} "
                   f"| freq_N {p_freq.mean():.4f} | within_full {p_full.mean():.4f}")

    for s in (sw, sw_freq):
        report(s.report())
    v1, v2 = sw.verdict(), sw_freq.verdict()

    report("\n  READING")
    if v1["verdict"] in ("ROBUST", "DIRECTIONAL"):
        report("  K562 pretraining HELPS when target-line data is scarce -- transferable signal exists.")
        if v2["verdict"] not in ("ROBUST", "DIRECTIONAL"):
            report("  BUT it does not beat the frequency table from the same N samples: a hollow pass.")
    elif v1["verdict"] == "HARMFUL":
        report("  K562 pretraining HURTS relative to training on the N samples alone. The prior is worse than")
        report("  nothing, and cross-line reuse of this model is not justified at any budget tested.")
    else:
        report("  K562 pretraining neither helps nor hurts -- it carries nothing that the target line's own")
        report("  samples do not already provide.")

    json.dump({"test": "adrn_fewshot", "lines": list(LINES), "shots": list(SHOTS), "K": K_FIT,
               "means": means, "sweep_vs_scratch": v1, "sweep_vs_freq": v2, "log": log},
              open(OUT / "adrn_fewshot.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_fewshot.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
