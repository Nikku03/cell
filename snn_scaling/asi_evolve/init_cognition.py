"""Seed the ASI-Evolve cognition store with snn_scaling reversal priors.

These lessons encode what we LEARNED from 5+ hand-tuned experiments
(supervised one-shot, Pavlovian, pure reversal, CLS on reversal, CLS on
mixed regime). Each is a transferable sentence that lets the
Researcher avoid known dead ends and aim at promising knobs.
"""
from __future__ import annotations

import json
import sys


COGNITION_SEED = [
    {
        "title": "Composite scoring punishes regressions hard",
        "content": (
            "composite = -best_integrated_cum + 0.1 * wall_time_min + 200 * "
            "(naive > best_integrated). A naive-wins regression costs 200 "
            "points - more than ANY plausible cumulative gain. Every "
            "mutation must keep MemRec or CLS strictly above the naive "
            "baseline."
        ),
    },
    {
        "title": "MemRec at TAU=50 is the prior reversal local optimum",
        "content": (
            "Hand-tuning at MEMREC_TAU=50 hit +91 cumulative over naive in "
            "4/5 seeds (full scale, N=2000, 600 trials). TAU in [30, 80] "
            "is the safe exploration range. TAU<20 forgets too fast "
            "(memory acts like 1-shot); TAU>100 carries stale phase-0 "
            "info into phase 1+."
        ),
    },
    {
        "title": "Reservoir features have weak class separability",
        "content": (
            "After centering, the 5-class centroid pairwise cosine is ~0.51 "
            "(mean) and ~0.69 (max). Same-class samples cluster within "
            "~14 degrees of cross-class. This caps best-possible "
            "accuracy regardless of architecture. Tuning reservoir params "
            "alone won't push it past this ceiling."
        ),
    },
    {
        "title": "CLS_ALPHA < 0.5 makes cortex dominate (bad for reversal)",
        "content": (
            "The cortex's slow EMA carries stale Q-values from old phases. "
            "On pure reversal, putting >50% weight on cortex pulls the "
            "agent toward old (now-wrong) actions. Safe range: [0.6, 0.85]. "
            "0.7 is the prior hand-tuned default."
        ),
    },
    {
        "title": "CLS_HIPPO_TAU should be < MEMREC_TAU",
        "content": (
            "CLS's hippocampus uses a faster recency (20 vs 50) because the "
            "cortex now provides the stability. If CLS_HIPPO_TAU = "
            "MEMREC_TAU, the two architectures end up similar and adding "
            "the cortex just adds noise. Sweet spot: CLS_HIPPO_TAU around "
            "MEMREC_TAU / 2."
        ),
    },
    {
        "title": "CLS_CORTEX_THRESHOLD has a narrow useful band",
        "content": (
            "Too low (<0.4) and every state collapses into one cluster - "
            "cortex Q-table degenerates to a single row. Too high (>0.8) "
            "and no consolidation happens - cortex never finds existing "
            "clusters, just keeps creating new ones. Useful range "
            "[0.5, 0.7]."
        ),
    },
    {
        "title": "RES_NOISE_STD dominates feature quality below 0.1, above 0.4",
        "content": (
            "Lower noise (<0.1) makes the reservoir near-deterministic, "
            "reducing the dynamic range that feature centering needs. "
            "Higher noise (>0.4) overwhelms the input signal in late-half "
            "V statistics. Useful range [0.15, 0.30]. Default 0.2 was "
            "hand-tuned."
        ),
    },
    {
        "title": "Sanity penalty trigger: naive beats integrated",
        "content": (
            "If naive_cum > best_integrated_cum, composite gets +200 "
            "(catastrophic). Common cause: reducing CLS_ALPHA too far so "
            "stale cortex dominates, or making MEMREC_TAU too short so "
            "memory becomes 1-shot (worse than gradient descent)."
        ),
    },
    {
        "title": "Wall-time savings are small relative to reward gains",
        "content": (
            "Composite penalty is 0.1 per minute. Cutting wall time from "
            "3 min to 1.5 min saves 0.15 points. A modest reward gain "
            "(+5 cum) is worth 5 points - 33x more. Therefore: do not "
            "trade reward quality for speed unless you're already at "
            "the reservoir-feature ceiling."
        ),
    },
    {
        "title": "T_TIMESTEPS=200 is overprovisioned",
        "content": (
            "Membrane tau is 20ms; the reservoir settles within ~5 tau = "
            "100ms. T_TIMESTEPS=200 wastes ~half the simulation time on "
            "settled state. Reducing to 120-150 is a near-free wall-time "
            "win that should not affect feature quality."
        ),
    },
    {
        "title": "TASK_NOISE_STD trade-off",
        "content": (
            "Lower task noise makes class separation easier (better "
            "features) but also lets the naive baseline learn faster, "
            "narrowing the integrated-vs-naive gap. Higher noise makes "
            "the task harder and rewards memory architectures more. "
            "0.3-0.5 is the band where the integrated advantage is "
            "biggest."
        ),
    },
    {
        "title": "N_RESERVOIR scaling has diminishing returns past 2000",
        "content": (
            "Per the centroid-cosine diagnostic, going from N=400 to N=2000 "
            "barely changed feature separability (0.89 -> 0.97 at N=2000 "
            "uncentered; 0.41 -> 0.51 centered). The class signal in the "
            "reservoir is mostly in the first ~1000 dimensions. Going "
            "above 2500 hurts wall time without help."
        ),
    },
    {
        "title": "Single-axis mutations preferred",
        "content": (
            "Change ONE hyperparameter per round. Lets the database learn "
            "which knob moves which metric. Multi-knob mutations make it "
            "impossible to attribute the result, and the noisier signal "
            "is harder to learn from."
        ),
    },
    {
        "title": "Numerical knobs map quickly; structural changes are the unexplored frontier",
        "content": (
            "Round-1 hand-tuning gave best_integrated_cum +28.52 via "
            "TASK_NOISE_STD=0.25. 15 follow-up numerical mutations couldn't "
            "beat it. The hyperparameter landscape is shallow. Bigger gains, "
            "if they exist, live in structural changes to the EvolvedAgent "
            "class - new memory shape, new retrieval rule, adaptive "
            "exploration, per-action memory banks - not in tweaking dials."
        ),
    },
    {
        "title": "EvolvedAgent is the architectural mutation target",
        "content": (
            "The MUTABLE SECTION includes a full EvolvedAgent class with "
            ".act() and .update() methods. The Researcher can rewrite "
            "these to try new algorithms. The 3 baseline agents (Naive, "
            "MemRec, CLS) are immutable references. Best_integrated_cum "
            "is the MAX across {MemRec, CLS, EvolvedAgent}, so a "
            "well-evolved EvolvedAgent can win even if MemRec also "
            "improves."
        ),
    },
    {
        "title": "Per-action memory banks - untested",
        "content": (
            "All current agents (MemRec, CLS, EvolvedAgent) use a SINGLE "
            "memory bank with action stored as a slot index in the value "
            "vector. An untested alternative: 3 separate memory banks, "
            "one per action. Retrieval reads the bank for each candidate "
            "action; decision is argmax of per-bank cosine-weighted means. "
            "May give cleaner per-action statistics on noisy features."
        ),
    },
    {
        "title": "Confidence-gated writes - untested",
        "content": (
            "Current agents write every trial. An untested alternative: "
            "only commit trials whose retrieval score (or some confidence "
            "signal) is high. Stale post-reversal entries get filtered out "
            "automatically. Risk: under-storage in early phase 0 when no "
            "history exists for confidence comparison; cold-start may need "
            "a warmup window."
        ),
    },
    {
        "title": "evolved_feature_transform is the highest-leverage mutation",
        "content": (
            "Every prior experiment was capped by the feature ceiling: the "
            "fixed (mean, std, range) representation gives centroid pairwise "
            "cosine ~0.51, so same-class and cross-class samples are barely "
            "separable. The evolved_feature_transform function lets the "
            "EvolvedAgent use a DIFFERENT representation. Temporal bins, "
            "derivatives, FFT magnitudes, or spike-rate proxies could pull "
            "out class signal the global mean discards. This is the only "
            "mutation surface with a path to a LARGE (not incremental) gain."
        ),
    },
    {
        "title": "Keep centering + L2 in any feature transform",
        "content": (
            "Centering (subtract per-feature batch mean) removes the "
            "reservoir's strong common-mode component near rest potential; "
            "without it centroid cosine is ~0.97 (useless). L2-normalisation "
            "makes cosine == dot for memory keys. Any rewrite of "
            "evolved_feature_transform should keep both as the final two "
            "steps, or class separability collapses."
        ),
    },
    {
        "title": "TASK_NOISE_STD is already optimized - do not re-tune it",
        "content": (
            "TASK_NOISE_STD=0.25 is the confirmed optimum (verified +28, "
            "+20, +13 across 3 separate runs vs the old 0.4 baseline) and "
            "is ALREADY baked into the initial_program default. Do not "
            "propose lowering it further - 0.10-0.20 was tried and stacks "
            "WORSE than 0.25 alone. Re-proposing TASK_NOISE_STD mutations "
            "wastes a round."
        ),
    },
]


def write_seed(path: str = "cognition_seed.json") -> None:
    with open(path, "w") as f:
        json.dump(COGNITION_SEED, f, indent=2)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cognition_seed.json"
    write_seed(out)
    print(f"wrote {len(COGNITION_SEED)} cognition seeds to {out}",
          file=sys.stderr)
