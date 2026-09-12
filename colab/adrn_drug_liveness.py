"""LIVENESS CHECK for adrn_drug_response: is the target->channel arm actually varying with the drug?

WHY THIS EXISTS.  adrn_drug_response tied its own shuffled control (+0.0038, CI spans zero).  There are two very
different reasons that can happen and they demand opposite conclusions:

    (a) the arm produces a DIFFERENT prediction per drug and those predictions are simply not better than
        permuted ones -- a real null.  The chain runs and carries no drug-specific information.
    (b) the arm produces the SAME prediction for every drug, because the 87 target families collapse to nearly
        identical channel vectors.  Then shuffling the drug->target map cannot change anything, the tie is
        arithmetic rather than empirical, and the correct report is "mechanism switched off", not "no effect".

This project has already shipped one bug of exactly type (b) -- the screen-design `diversity` arm that sorted
1-conf descending and selected the same genes as `uncertainty`.  So the tie is not written up until this is run.

It reads only `obs` from the h5ad (drug -> target strings), never the expression matrix, so it costs seconds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from adrn_drug_response import targets_of, _dec

OUT, SP = A.OUT, A.SP


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("LIVENESS: does the drug -> target -> channel arm vary with the drug at all?")
    report("=" * 100)

    scored = json.load(open(OUT / "adrn_drug_response.json"))["drugs"]
    hf = h5py.File(SP / "sciplex.h5ad", "r")
    pc = hf["obs"]["perturbation"]
    pcats = _dec(pc["categories"][:])
    pcodes = pc["codes"][:]
    tg = hf["obs"]["target"]
    tgcats = _dec(tg["categories"][:])
    tgcodes = tg["codes"][:]
    want = set(scored)
    drug_tg, seen = {}, set()
    for i in range(len(pcodes)):
        d = pcats[pcodes[i]]
        if d in want and d not in seen:
            seen.add(d)
            drug_tg[d] = targets_of(tgcats[tgcodes[i]])
    hf.close()
    scored = [d for d in scored if drug_tg.get(d)]
    report(f"  {len(scored)} scored drugs recovered from obs")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    tset = set(kos)
    universe = sorted(tset | {g for d in scored for g in drug_tg[d]})
    base, _ = A.build_channels(universe, tset, report)
    extra, _, tb_ = C.extra_channels(universe, tset, report)
    chan = np.concatenate([base, extra[:, :tb_]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}

    from sklearn.decomposition import NMF
    X = Amat.copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    w = A.ridge_fit(chan[[urow[k] for k in kos]], W, A.PENALTY)

    V = np.array([chan[[urow[g] for g in drug_tg[d]]].mean(0) for d in scored])
    S = A.ridge_apply(V, w) @ H
    S[:, tide] = -np.inf
    tops = [frozenset(np.argsort(-S[i])[:A.NPRED].tolist()) for i in range(len(scored))]

    n_distinct = len(set(tops))
    ov = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            ov.append(len(tops[i] & tops[j]) / float(A.NPRED))
    ov = np.array(ov)
    # how many genes are in EVERY drug's top-10 -- the constant core the arm always emits
    core = set.intersection(*[set(t) for t in tops]) if tops else set()
    # channel-vector spread: if the target families collapse, cosine similarity sits at ~1
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    cs = Vn @ Vn.T
    iu = np.triu_indices(len(scored), 1)

    report(f"\n  distinct top-10 sets across {len(scored)} drugs : {n_distinct}")
    report(f"  mean pairwise top-10 overlap                : {ov.mean():.3f}  (1.000 = identical predictions)")
    report(f"  median pairwise top-10 overlap              : {np.median(ov):.3f}")
    report(f"  genes present in EVERY drug's top-10        : {len(core)} of {A.NPRED}")
    report(f"  mean pairwise cosine of drug channel vectors: {cs[iu].mean():.3f}")
    report(f"  distinct target-gene sets                   : {len({frozenset(drug_tg[d]) for d in scored})}")

    dead = n_distinct <= 2 or ov.mean() > 0.9
    verdict = ("SWITCHED OFF -- the tie is arithmetic, not empirical" if dead else
               "LIVE -- predictions genuinely vary by drug, so the shuffled tie is a real null")
    report(f"\n  VERDICT: {verdict}")

    json.dump({"test": "adrn_drug_liveness", "n_drugs": len(scored), "n_distinct_top10": n_distinct,
               "mean_pairwise_overlap": float(ov.mean()), "median_pairwise_overlap": float(np.median(ov)),
               "constant_core": len(core), "mean_channel_cosine": float(cs[iu].mean()),
               "n_distinct_target_sets": len({frozenset(drug_tg[d]) for d in scored}),
               "dead": bool(dead), "log": log},
              open(OUT / "adrn_drug_liveness.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_drug_liveness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
