"""THE MECHANISM DECODER: the first thing in this project that scores a "why", not just a "which genes".

WHY THIS EXISTS.  Every knockout-prediction result here scores a ranked GENE LIST.  The dossiers carry function,
compartment, pathways, signed regulon and catalysed reactions, and ko_reason.py writes a genuine mechanistic
chain -- but nothing has ever graded that chain against ground truth.  The only architecture that generated its
list purely FROM mechanism, the 89-gene sensor menu, scored 0.0336 against a frequency baseline of 0.1956 and was
retired.  So "and here is the mechanism" has, until now, been unvalidated text attached to a validated list.

THE TASK, made gradeable.  A mechanism claim that can be scored is: WHICH BIOLOGICAL PROCESSES DOES THIS
KNOCKOUT DISTURB?  For a held-out knockout, rank Reactome pathways by how strongly they are over-represented
among its movers.  The truth is the same ranking computed from its MEASURED movers.  Both sides are the same kind
of object, so precision@10 over pathway terms means exactly what it looks like.

THE CEILING COMES FIRST, because a mechanism truth built from ~16 measured movers may be mostly noise.  SPLIT-HALF
AGREEMENT: split a knockout's true movers in two, enrich each half separately, and measure the overlap of the two
top-10 pathway lists.  If halves of the same knockout's own answer disagree, then no model can be scored against
it and the metric is dead on arrival.  That number is reported before any arm.

ARMS, each answering a different objection.

    own_pathways     the naive biologist's answer: predict the pathways the KNOCKED-OUT GENE belongs to.  "Knock
                     out a ribosome subunit, disturb the ribosome."  Needs no model and no training, and it is
                     the bar a mechanism decoder has to clear to be worth anything.
    frequency        the globally most-often-disturbed pathways, ignoring which gene was knocked out.  The
                     mechanism analogue of the usual-suspects baseline that has beaten every mechanistic method
                     in this project.
    from_genes       enrich the DEPLOYED MODEL's predicted gene list (chan2a, 0.2885 on the holdout).  This is
                     the mechanism the current system could actually emit today, obtained by reading its output.
    direct           a ridge from the 696 named channels straight to the pathway-enrichment vector, fitted on
                     TRAINING knockouts only.  Tests whether the mechanism is predictable without first
                     predicting genes.
    shuffled         enrichment of a random gene list of matched size.  The floor.
    oracle_half      enrichment of HALF the knockout's true movers, scored against the truth built from all of
                     them.  An oracle and a denominator, never a competitor: it is the split-half ceiling
                     expressed on the same axis as every arm.

PREDECLARED.  A mechanism decoder is worth reporting only if `from_genes` or `direct` beats BOTH `own_pathways`
AND `frequency` with a bootstrap CI excluding zero, on BOTH sealed cohorts.  If the naive "the gene's own
pathways" answer wins, then the honest statement is that the system's mechanism is a lookup of the perturbed
gene's annotation and adds nothing beyond it -- which is worth knowing and would be said in those words.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT = A.OUT
SP = A.SP
TOP_TERMS = 10
MIN_SET, MAX_SET = 10, 500
MIN_OVERLAP = 2
BOOT = 10_000


def load_pathways(genes: list[str], report):
    gi = {g: i for i, g in enumerate(genes)}
    sets: dict[str, list[int]] = {}
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) <= 3:
            continue
        members = sorted({gi[g] for g in parts[2:] if g in gi})
        if MIN_SET <= len(members) <= MAX_SET:
            sets[parts[0]] = members
    names = sorted(sets)
    P = np.zeros((len(names), len(genes)), np.float32)
    for r, n in enumerate(names):
        P[r, sets[n]] = 1.0
    report(f"  Reactome: {len(names):,} pathways with {MIN_SET}-{MAX_SET} measured members "
           f"(of {len(genes):,} scored genes)")
    return names, P


def enrich(indicator: np.ndarray, P: np.ndarray, n_genes: int) -> np.ndarray:
    """-log10 hypergeometric survival for every (row, pathway), vectorised.

    indicator: rows x genes, 1 where the gene is in the row's list.
    """

    from scipy.stats import hypergeom
    overlap = indicator @ P.T                                  # rows x pathways
    draw = indicator.sum(1, keepdims=True)                     # rows x 1
    size = P.sum(1)[None, :]                                   # 1 x pathways
    logsf = hypergeom.logsf(overlap - 1, n_genes, size, draw)
    score = -logsf / np.log(10.0)
    score[overlap < MIN_OVERLAP] = 0.0
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def top_terms(scores: np.ndarray, k: int = TOP_TERMS) -> list[set[int]]:
    out = []
    for row in scores:
        live = np.where(row > 0)[0]
        if len(live) <= k:
            out.append(set(live.tolist()))
        else:
            out.append(set(live[np.argsort(-row[live])[:k]].tolist()))
    return out


def precision(pred: set[int], truth: set[int]) -> float:
    return len(pred & truth) / TOP_TERMS if truth else float("nan")


def boot_ci(diff: np.ndarray):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    b = diff[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 106)
    report("MECHANISM DECODER -- which processes does this knockout disturb, and can we predict them?")
    report("=" * 106)
    report("  PREDECLARED: worth reporting only if from_genes or direct beats BOTH own_pathways AND frequency,")
    report("  CI excluding zero, on BOTH cohorts. If the gene's own pathways win, the mechanism is a lookup.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    all_genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    nontide = np.where(~tide)[0]
    genes = [all_genes[i] for i in nontide]
    gidx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    names, P = load_pathways(genes, report)

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

    # ---------------- the pathways each gene itself belongs to ----------------
    member_of: dict[str, set[int]] = collections.defaultdict(set)
    for r, n in enumerate(names):
        for j in np.where(P[r] > 0)[0]:
            member_of[genes[j]].add(r)

    # ---------------- training-side enrichment, for the frequency baseline and the direct ridge ----------
    def indicator_for(ko_list: list[str]) -> np.ndarray:
        ind = np.zeros((len(ko_list), n_genes), np.float32)
        for r, k in enumerate(ko_list):
            row = Amat[ki[k]][nontide]
            hits = np.where(row >= A.TAU)[0]
            ind[r, hits] = 1.0
        return ind

    train_scored = [k for k in train if (Amat[ki[k]][nontide] >= A.TAU).sum() >= 5]
    report(f"  training knockouts with >= 5 movers: {len(train_scored):,}")
    train_enrich = enrich(indicator_for(train_scored), P, n_genes)
    freq_rank = np.argsort(-(train_enrich > 0).sum(0))[:TOP_TERMS]
    freq_set = set(freq_rank.tolist())
    report(f"  frequency baseline = the {TOP_TERMS} most-often-disturbed pathways across training:")
    for r in freq_rank[:5]:
        report(f"      {names[r]}")

    # ---------------- named channels + the direct ridge ----------------
    universe = sorted(set(kos) | sealed)
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tier_b_start = C.extra_channels(universe, set(train), report)
    channels = np.concatenate([base, extra[:, :tier_b_start]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    fit_rows = np.array([urow[k] for k in train_scored if k in urow])
    fit_target = train_enrich[[i for i, k in enumerate(train_scored) if k in urow]]
    weights = A.ridge_fit(channels[fit_rows], fit_target, 10.0)
    report(f"  direct ridge fitted: {channels.shape[1]} channels -> {len(names):,} pathway scores "
           f"on {len(fit_rows):,} training knockouts")

    rng = np.random.default_rng(A.SEED)
    summary: dict[str, dict] = {}
    for tag, label in (("", "cohort 1"), ("_holdout", "cohort 2 (holdout)")):
        ans = answers.get(tag)
        if not ans:
            continue
        cohort = sorted(ans)
        preds_file = OUT / f"ko_predictions_chan2a{tag}.json"
        model_pred = json.loads(preds_file.read_text()) if preds_file.exists() else {}

        truth_ind = np.zeros((len(cohort), n_genes), np.float32)
        half_ind = np.zeros((len(cohort), n_genes), np.float32)
        shuf_ind = np.zeros((len(cohort), n_genes), np.float32)
        model_ind = np.zeros((len(cohort), n_genes), np.float32)
        sizes = []
        for r, k in enumerate(cohort):
            movers = [gidx[g] for g in ans[k]["movers"] if g in gidx]
            sizes.append(len(movers))
            truth_ind[r, movers] = 1.0
            if movers:
                half = rng.choice(movers, size=max(1, len(movers) // 2), replace=False)
                half_ind[r, half] = 1.0
                shuf_ind[r, rng.choice(n_genes, size=len(movers), replace=False)] = 1.0
            pl = [gidx[g] for g in model_pred.get(k, {}).get("predicted_genes", []) if g in gidx]
            model_ind[r, pl] = 1.0

        truth_terms = top_terms(enrich(truth_ind, P, n_genes))
        half_terms = top_terms(enrich(half_ind, P, n_genes))
        shuf_terms = top_terms(enrich(shuf_ind, P, n_genes))
        model_terms = top_terms(enrich(model_ind, P, n_genes))
        direct_terms = top_terms(A.ridge_apply(channels[[urow[k] for k in cohort]], weights))
        own_terms = [set(sorted(member_of.get(k, set()))[:TOP_TERMS]) for k in cohort]

        scored = [r for r, t in enumerate(truth_terms) if t]
        report(f"\n### {label}   {len(cohort)} knockouts, {len(scored)} with a non-empty mechanism truth; "
               f"median movers {int(np.median(sizes))}")

        arms = {
            "oracle_half (CEILING)": half_terms,
            "from_genes (deployed)": model_terms,
            "direct (channels->pathways)": direct_terms,
            "own_pathways (naive)": own_terms,
            "frequency": [freq_set] * len(cohort),
            "shuffled (floor)": shuf_terms,
        }
        vals = {a: np.array([precision(v[r], truth_terms[r]) for r in scored]) for a, v in arms.items()}
        report(f"  {'arm':<28} {'precision@10':>13} {'sd':>8}")
        for a, v in vals.items():
            report(f"  {a:<28} {np.nanmean(v):>13.4f} {np.nanstd(v):>8.4f}")

        report(f"\n  {'contrast':<36} {'gap':>8} {'95% CI':>22}  verdict")
        rows = {}
        for name, x, y in (("from_genes - own_pathways", "from_genes (deployed)", "own_pathways (naive)"),
                           ("from_genes - frequency", "from_genes (deployed)", "frequency"),
                           ("direct - own_pathways", "direct (channels->pathways)", "own_pathways (naive)"),
                           ("direct - frequency", "direct (channels->pathways)", "frequency"),
                           ("from_genes - shuffled", "from_genes (deployed)", "shuffled (floor)"),
                           ("own_pathways - frequency", "own_pathways (naive)", "frequency")):
            d = vals[x] - vals[y]
            lo, hi = boot_ci(d)
            verdict = "RESOLVED" if (lo > 0 or hi < 0) else "within noise"
            report(f"  {name:<36} {d.mean():>+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22}  {verdict}")
            rows[name] = {"gap": float(d.mean()), "ci": [lo, hi], "resolved": bool(lo > 0 or hi < 0)}

        ceiling = float(np.nanmean(vals["oracle_half (CEILING)"]))
        floor = float(np.nanmean(vals["shuffled (floor)"]))
        best = max(float(np.nanmean(vals["from_genes (deployed)"])),
                   float(np.nanmean(vals["direct (channels->pathways)"])))
        span = ceiling - floor
        report(f"\n  SPLIT-HALF CEILING {ceiling:.4f}: half of a knockout's own movers recovers only this much")
        report(f"  of the mechanism built from all of them -- the truth's own reproducibility.")
        report(f"  best arm covers {100 * (best - floor) / span if span > 0 else float('nan'):5.1f}% "
               f"of the floor->ceiling range")
        summary[tag or "cohort1"] = {
            "n_scored": len(scored),
            "means": {a: float(np.nanmean(v)) for a, v in vals.items()},
            "contrasts": rows, "ceiling": ceiling, "floor": floor}

    json.dump({"test": "adrn_mechanism_decoder", "top_terms": TOP_TERMS,
               "n_pathways": len(names), "summary": summary, "log": log},
              open(OUT / "adrn_mechanism_decoder.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_mechanism_decoder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
