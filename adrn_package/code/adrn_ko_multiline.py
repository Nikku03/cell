"""FIVE CELL LINES AS ONE: does pooling other lines help predict a K562 knockout it has never seen?

THE IDEA AND WHY IT IS NOT OBVIOUSLY RIGHT. The DepMap variance budget says 86.5% of gene-effect variance is the
GENE marginal -- shared across lines -- 0.0% is the line marginal, and 15.1% is the (gene, line) interaction, which
split-half CCA showed is real and reproducible (0.769 vs 0.025 permuted). If our model is mostly predicting the
shared gene marginal, then more lines is simply more data for the same target and should help.

Against that, ceiling_cartography measured naive cross-line transfer at 0.1835 against a marginal floor of 0.2498
-- transferring a model to another line scored WORSE than predicting the average. So pooling is not free, and the
question is which of those two facts dominates here.

CONSTRUCTION. The output vocabulary is frozen: the 60 NMF components come from K562 training knockouts only,
exactly as in every other arm, so a change in score cannot come from the basis moving. Each other line's profiles
are projected onto that same basis, giving every knockout in every line a loading vector in one shared space. Only
the TRAINING SET of the channels -> loading ridge changes between arms.

    other lines available   HCT116 1,400 · HepG2 2,151 · Jurkat 2,151 · RPE1 2,151 · Melanoma 218
                            = 8,071 extra profiles against K562's 4,720

ARMS
    mlk562    K562 training only. The reference, identical pipeline to the deployed model.
    mlpool    K562 + all five lines, one shared ridge. The realistic deployment: you have the other lines.
    mlstrict  the same pool, but every knockout whose GENE is in either sealed cohort is dropped from the OTHER
              lines too. This is the arm that tests the user's actual hypothesis -- more data for the shared
              gene marginal -- with no possibility of reading the held-out gene's behaviour off another line.
    mlother   the five other lines ONLY, no K562 at all. Pure cross-line transfer, which is the thing
              ceiling_cartography measured at below the marginal floor.
    mlbank    shared ridge fitted on the pool, then a K562-SPECIFIC correction fitted on the K562 residual.
              Pool for the 86% that is shared, bank for the 15% that is not. Per-context adapter banking is the
              one ADRN idea that has reproducibly worked, and this is it applied to cell lines.

PREDECLARED, three outcomes and what each means.
    mlpool/mlstrict > mlk562   the shared gene marginal was not saturated at 4,720 profiles; more lines is more
                               data and the answer to "can we use many lines as one" is yes.
    mlpool ~= mlk562           the shared part was already saturated; extra lines add nothing and the interaction
                               is where the remaining signal lives.
    mlpool < mlk562            lines actively interfere, banking is mandatory rather than optional, and mlbank
                               is the only viable pooled form.
    The gap mlpool - mlstrict prices how much comes from seeing the held-out GENE measured elsewhere, as opposed
    to from more data generally. If that gap is most of the effect, the honest headline is transfer, not scale.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT = A.OUT
SP = A.SP
LINES = ("HCT116", "HepG2", "Jurkat", "RPE1", "Melanoma")


def load_line(name: str, gene_index: dict[str, int], n_genes: int, report):
    """|z| profiles for one line, laid out on the gwps gene axis. Missing genes stay 0."""

    path = SP / f"nlz_{name}.pkl"
    if not path.exists():
        report(f"  [warn] {path.name} missing, skipping {name}")
        return [], np.zeros((0, n_genes), np.float32)
    KO = pickle.load(open(path, "rb"))
    kos = sorted(KO)
    M = np.zeros((len(kos), n_genes), np.float32)
    for i, k in enumerate(kos):
        for g, zval in KO[k].items():
            j = gene_index.get(g)
            if j is not None:
                M[i, j] = abs(float(zval))
    return kos, M


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 108)
    report("FIVE CELL LINES AS ONE -- does pooling help predict an unseen K562 knockout?")
    report("=" * 108)
    report("  PREDECLARED: pool > k562 means the shared gene marginal was not saturated. pool == k562 means it")
    report("  was. pool < k562 means lines interfere and banking is mandatory. mlpool - mlstrict prices how much")
    report("  comes from seeing the held-out GENE measured in another line rather than from more data generally.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gi = {g: i for i, g in enumerate(genes)}
    Amat = np.abs(z["M"])
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed: set[str] = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    train = [k for k in kos if k not in sealed]

    # ---- the output vocabulary, frozen on K562 training exactly as everywhere else ----
    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    report(f"\n  K562 training pool {len(train):,}; factorising into {A.K} components ...")
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W_k562 = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    del X
    project = np.linalg.pinv(H.T).astype(np.float32)          # loadings = project @ profile, one matmul for all

    # ---- channels: the richer TIER A set, identical to chan2a ----
    universe_seed = sorted(set(kos) | sealed)
    base, _ = A.build_channels(universe_seed, set(train), report)
    extra, _, tier_b_start = C.extra_channels(universe_seed, set(train), report)
    channels_seed = np.concatenate([base, extra[:, :tier_b_start]], axis=1)
    urow = {g: i for i, g in enumerate(universe_seed)}
    report(f"  channel set: {channels_seed.shape[1]} columns (same as chan2a)")

    # ---- every other line, projected onto the frozen basis ----
    pooled_names: list[str] = []
    pooled_loadings: list[np.ndarray] = []
    pooled_line: list[str] = []
    for line in LINES:
        lkos, M = load_line(line, gi, len(genes), report)
        if not lkos:
            continue
        M[:, tide] = 0.0
        keep = [i for i, k in enumerate(lkos) if k in urow]     # need channels for the knocked-out gene
        if not keep:
            continue
        loadings = (project @ M[keep].T).T
        pooled_names += [lkos[i] for i in keep]
        pooled_loadings.append(loadings.astype(np.float32))
        pooled_line += [line] * len(keep)
        report(f"    {line:<10} {len(keep):5d} knockouts usable of {len(lkos)}")
    other_L = np.concatenate(pooled_loadings, axis=0) if pooled_loadings else np.zeros((0, A.K), np.float32)
    other_rows = np.array([urow[k] for k in pooled_names], dtype=int)
    other_is_sealed = np.array([k in sealed for k in pooled_names])
    report(f"  other lines total {len(pooled_names):,} usable profiles; "
           f"{int(other_is_sealed.sum())} of them are knockouts of a SEALED gene")

    k_rows = np.array([urow[k] for k in train], dtype=int)
    sets = {
        "mlk562": (k_rows, W_k562),
        "mlpool": (np.concatenate([k_rows, other_rows]), np.concatenate([W_k562, other_L])),
        "mlstrict": (np.concatenate([k_rows, other_rows[~other_is_sealed]]),
                     np.concatenate([W_k562, other_L[~other_is_sealed]])),
        "mlother": (other_rows, other_L),
    }
    fitted = {}
    for arm, (rows, target) in sets.items():
        fitted[arm] = A.ridge_fit(channels_seed[rows], target, A.PENALTY)
        report(f"    {arm:<9} fitted on {len(rows):,} profiles")

    # ---- mlbank: shared map on the pool, then a K562-specific correction on the K562 residual ----
    prows, ptarget = sets["mlpool"]
    shared = fitted["mlpool"]
    residual = W_k562 - A.ridge_apply(channels_seed[k_rows], shared)
    bank = A.ridge_fit(channels_seed[k_rows], residual, A.PENALTY)
    report(f"    mlbank    shared map on {len(prows):,} + K562 correction on {len(k_rows):,}")

    globalmix = W_k562.mean(0)
    for tag in ("", "_holdout"):
        dp = OUT / f"ko_pred_dossiers{tag}.json"
        if not dp.exists():
            continue
        cohort = sorted(json.loads(dp.read_text()))
        rows = np.array([urow[k] for k in cohort])
        design = channels_seed[rows]
        outputs = {a: A.ridge_apply(design, w) for a, w in fitted.items()}
        outputs["mlbank"] = A.ridge_apply(design, shared) + A.ridge_apply(design, bank)
        for arm, mixes in outputs.items():
            preds = {}
            for j, k in enumerate(cohort):
                mix = mixes[j] if np.isfinite(mixes[j]).all() else globalmix
                score = mix @ H
                score[tide] = 0.0
                if k in gi:
                    score[gi[k]] = 0.0
                order = np.argsort(-score)[:A.NPRED]
                preds[k] = {"variant": arm,
                            "reasoning": f"{design.shape[1]} named channels -> ridge fitted on "
                                         f"{'K562 + 5 lines' if arm != 'mlk562' else 'K562'} -> mixture of "
                                         f"{A.K} K562 response components -> ranked over {len(genes):,} genes",
                            "predicted_genes": [genes[i] for i in order if score[i] > 0]}
            p = OUT / f"ko_predictions_{arm}{tag}.json"
            json.dump(preds, open(p, "w"), indent=1)
            report(f"  {tag or 'cohort1':<9} {arm:<9} -> {p.name}")

    json.dump({"test": "adrn_ko_multiline", "lines": list(LINES),
               "n_k562_train": len(train), "n_other": len(pooled_names),
               "n_other_sealed_gene": int(other_is_sealed.sum()),
               "n_channels": int(channels_seed.shape[1]), "log": log},
              open(OUT / "adrn_ko_multiline.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_ko_multiline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
