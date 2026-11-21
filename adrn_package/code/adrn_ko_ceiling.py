"""WHAT IS THE CEILING ON THE SEALED KNOCKOUT PROTOCOL? Price the 0.2620 before optimising past it.

WHY THIS RUNS FIRST. adrn_ko_conjunctions.py reached 0.2620 per-knockout precision at 20 predicted genes and
nobody knows whether that is 90% of what is reachable or 40%. The old ceiling work (output_ceiling.py) priced the
retired 89-gene sensor menu against a shared vocabulary; it says nothing about a per-knockout ranking over 8,246
genes. Every improvement proposed after this is unpriced until these numbers exist.

FOUR CEILINGS, EACH CLOSING A DIFFERENT ESCAPE ROUTE. All are computed against the identical sealed answers, at
the identical budget of 20 predicted genes, so they sit on the same axis as every published number.

    absolute        the best possible 20 genes, chosen with the answers in hand. min(20, |truth|)/20. Nothing
                    can beat this; it is bounded below 1.0 only because many knockouts have fewer than 20 movers.

    basis           the best any method routing through the 60 NMF components can do. The held-out knockout's
                    TRUE profile is projected onto the component basis and ranked. This is an ORACLE -- it reads
                    the answer -- and it is a denominator, never a competitor. If the deployed model is close to
                    it, the vocabulary is the binding constraint and better channels cannot help.

    channel_nn      lookup by nearest annotation: predict the measured movers of the nearest TRAINING knockout in
                    channel space.  THIS WAS FIRST WRITTEN AS A CEILING AND IT IS NOT ONE -- the deployed ridge
                    beats it (0.2127 vs 0.1320, 0.2620 vs 0.1938), because pooling hundreds of training knockouts
                    averages out noise that a single donor carries.  It is reported as a BASELINE, and the
                    original claim is corrected rather than deleted.

    twin_ceiling    the cap that actually binds any function of the channels.  Knockouts with an IDENTICAL channel
                    vector must receive an IDENTICAL prediction -- no model that sees only annotation can separate
                    them.  So group every knockout by exact channel vector and give each group the single best
                    20-gene list for its members, chosen with all their answers in hand.  That is the highest
                    score any channel-only function can reach, whatever its form.

    profile_nn      the retrieval ceiling: the nearest training knockout in TRUE PROFILE space. Also an oracle.
                    The gap between channel_nn and profile_nn is exactly the information annotation does not
                    carry -- two knockouts that look identical on paper but behave differently.

IDENTIFIABILITY, WHICH IS THE HARDEST CAP OF ALL. If two knockouts have the same channel vector, no function of
the channels can give them different predictions. This file measures how often that happens among the sealed
cohorts and how much the truths of such pairs actually agree. A low agreement is a hard, structural cap that no
amount of modelling removes -- only new channels do.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import adrn_ko_conjunctions as A

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = A.SP
NPRED = A.NPRED


def precision_at(pred: list[str], truth: set[str], budget: int = NPRED) -> float:
    return len(set(pred[:budget]) & truth) / budget


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 104)
    report("CEILING MAP FOR THE SEALED KNOCKOUT PROTOCOL (20 predicted genes)")
    report("=" * 104)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"])
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    nontide = ~tide

    sealed: set[str] = set()
    answers: dict[str, dict] = {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    tidx = np.array([ki[k] for k in train])

    from sklearn.decomposition import NMF
    X = Amat[tidx].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    report(f"\n  readout {len(kos):,} x {len(genes):,}; training pool {len(train):,}; "
           f"{A.K} components; tide {int(tide.sum())}")
    del X

    universe = sorted(set(kos) | sealed)
    channels, names = A.build_channels(universe, set(train), report)
    urow = {g: i for i, g in enumerate(universe)}
    train_channels = channels[[urow[k] for k in train]]
    # profiles restricted to the scored (non-tide) genes, unit-normalised for cosine retrieval
    train_profile = Amat[tidx][:, nontide]
    tp_norm = train_profile / np.maximum(np.linalg.norm(train_profile, axis=1, keepdims=True), 1e-9)
    tc_norm = train_channels / np.maximum(np.linalg.norm(train_channels, axis=1, keepdims=True), 1e-9)
    nontide_genes = [g for g, keep in zip(genes, nontide) if keep]

    summary: dict[str, dict] = {}
    for tag, label in (("", "cohort 1"), ("_holdout", "cohort 2 (holdout)")):
        ans = answers.get(tag)
        if not ans:
            continue
        cohort = sorted(ans)
        rows: dict[str, list[float]] = {k: [] for k in
                                        ("absolute", "basis", "channel_nn", "profile_nn")}
        truth_sizes, nn_channel_sim, nn_truth_overlap = [], [], []
        for k in cohort:
            truth = set(ans[k]["movers"])
            truth_sizes.append(len(truth))
            rows["absolute"].append(min(NPRED, len(truth)) / NPRED)

            # --- basis oracle: project the TRUE profile onto the component basis, then rank
            true_row = Amat[ki[k]].copy()
            true_row[tide] = 0.0
            loading, *_ = np.linalg.lstsq(H.T, true_row, rcond=None)
            score = np.maximum(loading, 0) @ H
            score[tide] = 0.0
            if k in genes:
                score[genes.index(k)] = 0.0
            rows["basis"].append(precision_at([genes[i] for i in np.argsort(-score)], truth))

            # --- channel nearest training knockout
            cvec = channels[urow[k]]
            cvec = cvec / max(np.linalg.norm(cvec), 1e-9)
            sim = tc_norm @ cvec
            j = int(np.argmax(sim))
            nn_channel_sim.append(float(sim[j]))
            donor = train_profile[j]
            order = np.argsort(-donor)
            pred = [nontide_genes[i] for i in order if donor[i] > 0][:NPRED]
            rows["channel_nn"].append(precision_at(pred, truth))
            donor_truth = {nontide_genes[i] for i in np.where(donor >= A.TAU)[0]}
            nn_truth_overlap.append(len(donor_truth & truth) / max(len(truth | donor_truth), 1))

            # --- profile nearest training knockout (oracle retrieval)
            pvec = Amat[ki[k]][nontide]
            pvec = pvec / max(np.linalg.norm(pvec), 1e-9)
            j2 = int(np.argmax(tp_norm @ pvec))
            donor2 = train_profile[j2]
            pred2 = [nontide_genes[i] for i in np.argsort(-donor2) if donor2[i] > 0][:NPRED]
            rows["profile_nn"].append(precision_at(pred2, truth))

        # deployed model + frequency, read from the scores already on disk
        deployed = {}
        for arm in ("adrnlin", "adrnconj", "basis", "nbr"):
            f = OUT / f"ko_score_{arm}{tag}.json"
            if f.exists():
                r = json.loads(f.read_text())["rows"]
                deployed[arm] = float(np.mean([x["precision"] for x in r]))
                if arm == "adrnlin":
                    deployed["frequency"] = float(np.mean(
                        [x["frequency_hits"] / max(x["n_pred"], 1) for x in r]))

        report(f"\n### {label}   ({len(cohort)} knockouts, median truth size "
               f"{int(np.median(truth_sizes))})")
        report(f"  {'level':<26} {'precision@20':>13}   what it bounds")
        report(f"  {'ABSOLUTE (answers in hand)':<26} {np.mean(rows['absolute']):>13.4f}   "
               f"nothing can beat this")
        report(f"  {'profile_nn ORACLE':<26} {np.mean(rows['profile_nn']):>13.4f}   "
               f"retrieval, if you already knew the profile")
        report(f"  {'basis ORACLE':<26} {np.mean(rows['basis']):>13.4f}   "
               f"anything routing through the {A.K} components")
        report(f"  {'channel_nn':<26} {np.mean(rows['channel_nn']):>13.4f}   "
               f"any function of the named channels")
        for arm in ("adrnconj", "adrnlin", "nbr", "basis"):
            if arm in deployed:
                tagname = f"  DEPLOYED {arm}" if arm.startswith("adrn") else f"  incumbent {arm}"
                report(f"  {tagname:<26} {deployed[arm]:>13.4f}")
        if "frequency" in deployed:
            report(f"  {'  frequency baseline':<26} {deployed['frequency']:>13.4f}")

        best = deployed.get("adrnlin", 0.0)
        floor = deployed.get("frequency", 0.0)
        for name in ("channel_nn", "basis", "absolute"):
            cap = float(np.mean(rows[name]))
            span = cap - floor
            got = (best - floor) / span if span > 0 else float("nan")
            report(f"    fraction of the {name} range covered by the deployed model: {100 * got:5.1f}%  "
                   f"(floor {floor:.4f} -> cap {cap:.4f})")

        # ---- twin ceiling: knockouts sharing a channel vector must share a prediction ----
        # every knockout in the readout with a truth set, grouped by exact channel vector
        truth_of: dict[str, set[str]] = {}
        for t2 in ("", "_holdout"):
            for k2, v2 in answers.get(t2, {}).items():
                truth_of[k2] = set(v2["movers"])
        for k2 in train:
            row = Amat[ki[k2]]
            hits = {genes[i] for i in np.where(row >= A.TAU)[0] if not tide[i]}
            if len(hits) >= 5:
                truth_of[k2] = hits
        groups: dict[bytes, list[str]] = {}
        for k2 in truth_of:
            if k2 in urow:
                groups.setdefault(channels[urow[k2]].tobytes(), []).append(k2)
        twin_scores, group_sizes = [], []
        for k in cohort:
            members = groups.get(channels[urow[k]].tobytes(), [k])
            group_sizes.append(len(members))
            tally: dict[str, int] = {}
            for m in members:
                for g in truth_of.get(m, ()):          # answers in hand for the whole group: an ORACLE
                    tally[g] = tally.get(g, 0) + 1
            best = sorted(tally, key=lambda g: (-tally[g], g))[:NPRED]
            twin_scores.append(precision_at(best, truth_of[k]))
        report(f"\n  TWIN CEILING -- the cap on any function of these channels")
        report(f"    knockouts sharing an exact channel vector must share a prediction; median group size "
               f"{int(np.median(group_sizes))}, max {max(group_sizes)}")
        report(f"    best single 20-gene list per group, chosen with every member's answer in hand: "
               f"{np.mean(twin_scores):.4f}")
        span = float(np.mean(twin_scores)) - deployed.get("frequency", 0.0)
        got = (deployed.get("adrnlin", 0.0) - deployed.get("frequency", 0.0)) / span if span > 0 else float("nan")
        report(f"    fraction of THAT range covered by the deployed model: {100 * got:5.1f}%")

        report(f"\n  IDENTIFIABILITY of the channel representation")
        report(f"    mean cosine to the nearest training knockout in channel space: "
               f"{np.mean(nn_channel_sim):.4f}")
        report(f"    knockouts whose nearest channel neighbour is an EXACT match (cos > 0.999): "
               f"{sum(1 for s in nn_channel_sim if s > 0.999)}/{len(cohort)}")
        report(f"    Jaccard between a knockout's truth and its channel-nearest neighbour's truth: "
               f"{np.mean(nn_truth_overlap):.4f}")
        report(f"    -> two knockouts that look identical on paper share {100*np.mean(nn_truth_overlap):.1f}% "
               f"of their movers; that is the hard cap new CHANNELS have to move")

        summary[tag or "cohort1"] = {
            "n": len(cohort),
            "absolute": float(np.mean(rows["absolute"])),
            "profile_nn_oracle": float(np.mean(rows["profile_nn"])),
            "basis_oracle": float(np.mean(rows["basis"])),
            "channel_nn": float(np.mean(rows["channel_nn"])),
            "deployed": deployed,
            "nn_channel_cosine": float(np.mean(nn_channel_sim)),
            "exact_channel_twins": sum(1 for s in nn_channel_sim if s > 0.999),
            "nn_truth_jaccard": float(np.mean(nn_truth_overlap)),
            "twin_ceiling": float(np.mean(twin_scores)),
            "median_twin_group": int(np.median(group_sizes)),
            "max_twin_group": int(max(group_sizes)),
        }

    json.dump({"test": "adrn_ko_ceiling", "npred": NPRED, "summary": summary, "log": log},
              open(OUT / "adrn_ko_ceiling.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_ko_ceiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
