"""IS THE GROWN CONJUNCTION SET STABLE? Re-grow it and see what survives.

WHY THIS PROBE EXISTS. Two runs of adrn_ko_conjunctions.py with identical inputs and identical seeds produced
DIFFERENT top-64 conjunction sets, and adrnconj's sealed-cohort precision moved by 0.0016 (cohort 1) and 0.0037
(cohort 2) between them. Those deltas are the same size as the entire gain the mechanism claims over its controls
(+0.0075 and +0.0025), so the stability of the selection is not a detail -- it decides whether there is anything
to report at all.

THE SUSPECT. 20,000 candidates are scored and the top 64 are kept. If the score distribution is nearly flat near
the cut, then float-level differences -- multithreaded BLAS reassociating a reduction differently between runs --
are enough to reshuffle which candidates land above the line. That is the signature of selecting on noise: a real
effect would sit clear of the cut and survive.

WHAT IS MEASURED
    determinism   grow twice inside ONE process from byte-identical inputs. Any difference here is pure
                  float non-determinism, with no seed or data change to hide behind.
    seed spread   grow from independent candidate pools. If the mechanism is finding real structure, different
                  pools should rediscover overlapping products; if it is fitting noise, overlap will be at the
                  rate two random 64-subsets of the pool would collide by chance, which is ~0.
    margin        how far the 64th score sits above the 65th, and above the threshold. A mechanism with a real
                  signal has a gap there; one fitting noise has none.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import adrn_ko_conjunctions as A

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = A.SP


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 100)
    report("STABILITY OF THE GROWN CONJUNCTION SET")
    report("=" * 100)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"])
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    sealed: set[str] = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    train = [k for k in kos if k not in sealed]

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    del X

    universe = sorted(set(kos) | sealed)
    channels, names = A.build_channels(universe, set(train), report)
    urow = {g: i for i, g in enumerate(universe)}
    train_rows = np.array([urow[k] for k in train])
    tc = channels[train_rows]
    thr = np.median(tc, axis=0)
    signed = np.where(tc > thr[None, :], 1.0, -1.0).astype(np.float32)
    residual = W - A.ridge_apply(tc, A.ridge_fit(tc, W, A.PENALTY))

    # ---- 1. determinism: identical inputs, same process, twice ----
    pool = A.candidate_pool(channels.shape[1], np.random.default_rng(A.SEED + 1))
    first = A.grow(signed, residual, pool, max_grown=A.MAX_GROWN, threshold=A.THRESHOLD)
    second = A.grow(signed, residual, pool, max_grown=A.MAX_GROWN, threshold=A.THRESHOLD)
    same = set(first) == set(second)
    report(f"\n  1. DETERMINISM (same process, byte-identical inputs, grown twice)")
    report(f"     identical set: {same}   overlap {len(set(first) & set(second))}/{len(first)}")

    # ---- 2. margin at the cut ----
    n = len(signed)
    scale = residual.std(axis=0, ddof=1)
    scale[scale <= 0] = 1.0
    standard = (residual / scale[None, :]).astype(np.float32)
    score = np.empty(len(pool))
    for start in range(0, len(pool), 2048):
        block = pool[start:start + 2048]
        products = np.stack([np.prod(signed[:, list(p)], axis=1) for p in block])
        score[start:start + len(block)] = np.abs((products @ standard) / n).max(axis=1) * np.sqrt(n)
    srt = np.sort(score)[::-1]
    report(f"\n  2. MARGIN AT THE CUT (threshold {A.THRESHOLD}, keeping top {A.MAX_GROWN} of {len(pool):,})")
    report(f"     score at rank 1 {srt[0]:.4f}, rank 64 {srt[63]:.4f}, rank 65 {srt[64]:.4f}, "
           f"rank 200 {srt[199]:.4f}")
    report(f"     gap between the last kept and the first dropped: {srt[63] - srt[64]:.6f}")
    report(f"     candidates within 1% of the rank-64 score: "
           f"{int((score >= srt[63] * 0.99).sum()):,}  <- all of these are interchangeable at the cut")
    report(f"     candidates clearing the threshold at all: {int((score >= A.THRESHOLD).sum()):,}")

    # ---- 3. independent candidate pools ----
    report(f"\n  3. SEED SPREAD (independent candidate pools; a real product should be rediscovered)")
    sets = []
    for s in range(4):
        p = A.candidate_pool(channels.shape[1], np.random.default_rng(1000 + s))
        sets.append(set(A.grow(signed, residual, p, max_grown=A.MAX_GROWN, threshold=A.THRESHOLD)))
        report(f"     pool seed {1000 + s}: {len(sets[-1])} grown")
    pairs = [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    overlaps = [len(sets[i] & sets[j]) for i, j in pairs]
    # chance overlap for two 64-subsets drawn from a 20,000 candidate pool
    chance = A.MAX_GROWN * A.MAX_GROWN / A.CANDIDATES
    report(f"     pairwise overlap of grown sets: {overlaps}  (mean {np.mean(overlaps):.2f})")
    report(f"     overlap expected by chance for two random 64-subsets of {A.CANDIDATES:,}: {chance:.2f}")

    # channel-level: do the same CHANNELS recur even if the exact products differ?
    chan_sets = [{c for conj in s for c in conj} for s in sets]
    chan_over = [len(chan_sets[i] & chan_sets[j]) / max(len(chan_sets[i] | chan_sets[j]), 1) for i, j in pairs]
    report(f"     Jaccard over the CHANNELS used: {[round(x, 3) for x in chan_over]}  "
           f"(mean {np.mean(chan_over):.3f})")

    json.dump({"test": "adrn_conjunction_stability",
               "deterministic": bool(same),
               "margin_rank64": float(srt[63]), "margin_rank65": float(srt[64]),
               "gap": float(srt[63] - srt[64]),
               "within_1pct_of_cut": int((score >= srt[63] * 0.99).sum()),
               "clearing_threshold": int((score >= A.THRESHOLD).sum()),
               "pairwise_overlap": overlaps, "chance_overlap": float(chance),
               "channel_jaccard": [float(x) for x in chan_over], "log": log},
              open(OUT / "adrn_ko_stability.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_ko_stability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
