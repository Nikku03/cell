"""ADRN ON THE SEALED KNOCKOUT PROTOCOL: named annotation channels, grown conjunctions, held-out knockouts.

WHY THIS FILE EXISTS
--------------------
Two facts sat next to each other and had never been put in the same experiment.

FACT 1 -- the sealed protocol is the task that matters, and no neural model has ever been scored on it.
Every method ever run through ko_predict.py's sealed cohorts is non-neural: an NMF basis, neighbour transfer, a
Markov walk, a hand-written 89-gene sensor menu.  Both the transformer line and the ADRN line were developed on
PROXIES -- recall@50 after nearest-10 retrieval on gwps pseudobulk, gene-effect regression on DepMap blocks -- and
neither has ever predicted a held-out knockout's movers under a sealed protocol.

FACT 2 -- ADRN-3's conjunction growth needs an AXIS-ALIGNED NAMED BASIS, and has never been given one here.
v4_conj_basis.py measured this decisively.  On 60 named DepMap annotation channels the mechanism was worth -0.0096
RMSE past a label-shuffled control (RESOLVED, 8 blocks).  Rotate those same channels -- same genes, same
information, same targets, invertible transform, ridge unchanged at 0.3303 -- and retention falls to 0.0% (PCA) and
1.6% (random orthogonal).  A conjunction that is sparse in named coordinates has no sparse representative after a
rotation, so a rotated basis deletes the object the mechanism searches for.

That explains ADRN's failure on the Replogle criterion-2 task, where the inputs were 64 DepMap principal
components: it was an INGESTION failure, not a mechanism failure.  The mechanism has never been run on this
project's central task with the input representation it demonstrably requires.  This file does that.

WHAT IS HELD FIXED, SO THAT ONLY THE FEATURE SIDE VARIES
-------------------------------------------------------
The skeleton is response_basis.py's, unchanged and deliberately so:

    annotation -> [SOMETHING] -> mixture over 60 NMF components -> ranked list over all measured genes

response_basis puts hand-assigned LESION CLASSES in the [SOMETHING] slot and scores 0.2013 on cohort 2.  This file
puts NAMED ANNOTATION CHANNELS + GROWN CONJUNCTIONS there.  Same NMF basis, same K, same training pool, same
NPRED, same sealed cohorts, same scorer.  The lesion classes are themselves included as channels, so the incumbent's
feature set is a strict SUBSET of this one and any difference is attributable to the representation alone.

THE CHANNELS ARE NAMED AND BINARY, WHICH IS THE WHOLE POINT
----------------------------------------------------------
Every channel is an interpretable indicator a biologist could read off a dossier -- "is in the nucleus", "is in
Reactome Cell Cycle", "has a signed regulon", "catalyses >=5 reactions".  No PCs, no embeddings, no learned
projection.  That is not an aesthetic choice: it is the condition the basis test says the mechanism requires.

ARMS -- each is a prediction file scored by ko_predict.score, so the metric and controls are identical by
construction to every number this project has published on this task.

    adrnlin    named channels -> ridge -> component mixture.  ADRN's LINEAR part alone.
    adrnconj   named channels + grown conjunctions.  THE TEST.
    adrnshuf   channels + conjunctions grown against a PERMUTED residual, same count, threshold -inf so the
               count matches exactly.  Prices the top-k selection: growing 64 conjunctions out of 20,000
               candidates will find apparent structure in pure noise, and this is what that is worth.
    adrnrand   channels + conjunctions drawn at random from the identical candidate pool, same count and arity
               mix.  Separates "conjunctions help" from "THESE conjunctions help".
    adrnrot    a full-rank PCA rotation of the channels, then the identical growth.  Information preserved
               exactly, axis alignment destroyed.  If this matches adrnconj, the basis-test diagnosis does not
               transfer to this task and the named-channel premise of this whole file is wrong.

PREDECLARED, BEFORE RUNNING.  Three ways this fails and one way it succeeds.

    FAILS if adrnconj does not beat the FREQUENCY BASELINE.  "Predict the genes that always move" needs no
          biology and has beaten every mechanistic method here by 4-10x; only neighbour transfer has ever cleared
          it, and that borrows measured answers rather than reasoning.
    FAILS if adrnconj does not beat adrnshuf.  Then the gain is selection-on-noise, which is exactly how the
          89-gene sensor menu turned out to be decorative.
    FAILS if adrnconj ~= adrnlin.  Then the conjunctions add nothing and ADRN reduces to a ridge on annotations.
    SUCCEEDS only if adrnconj > frequency AND > adrnshuf AND > adrnlin, and adrnrot loses most of the gain.

LEAKAGE DISCIPLINE, stricter than the protocol requires
-------------------------------------------------------
    - the NMF basis is factorised on the training pool with BOTH sealed cohorts (400 knockouts) removed wholesale,
      not just the knockout being predicted -- a cohort member contributing a component would leak into every
      other cohort member's prediction
    - the ridge from channels to loadings is fitted on training knockouts only
    - conjunctions are grown on the TRAINING residual only
    - the binarisation threshold is the TRAINING median only
    - a held-out knockout's own profile row is never read at any point
    - tide genes are zeroed (they are excluded from the truth by construction) and so is the knockout's own gene
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

K = int(os.environ.get("RB_K", "60"))              # NMF components, matched to response_basis.py
NPRED = int(os.environ.get("RB_NPRED", "20"))      # genes predicted per knockout, matched to every other method
TAU = 1.0
MAX_GROWN = int(os.environ.get("ADRN_MAX_GROWN", "64"))
CANDIDATES = int(os.environ.get("ADRN_CANDIDATES", "20000"))
THRESHOLD = float(os.environ.get("ADRN_THRESHOLD", "4.0"))
ARITIES = (2, 3, 5)                                # matched to the DepMap run that produced the one ADRN win
PENALTY = float(os.environ.get("ADRN_PENALTY", "10.0"))
TOP_PATHWAYS = int(os.environ.get("ADRN_TOP_PATHWAYS", "120"))
SEED = int(os.environ.get("ADRN_SEED", "20260802"))


# ---------------------------------------------------------------------------------------------------------------
# NAMED CHANNELS.  Every column below is an indicator with a name, built from a-priori annotation only.
# ---------------------------------------------------------------------------------------------------------------
def build_channels(all_genes: list[str], train_set: set[str], report) -> tuple[np.ndarray, list[str]]:
    """Binary annotation channels for every gene, plus the channel names.

    Channel SELECTION (which pathways and compartments make the cut) is done by frequency among TRAINING
    knockouts only.  Selecting on the full set would let the sealed cohorts influence which columns exist -- a
    weak leak, but a free one to close.
    """

    idx = {g: i for i, g in enumerate(all_genes)}

    # --- Reactome pathway membership -----------------------------------------------------------------------
    paths: dict[str, list[str]] = collections.defaultdict(list)
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > 3:
            for g in parts[2:]:
                paths[g].append(parts[0])

    # --- reaction network: catalysis, participation, compartments, partner counts ---------------------------
    net = json.load(open(OUT / "reaction_network.json"))
    comp_of_step = json.loads((OUT / "compartments.json").read_text())["steps"]
    n_catalyses: collections.Counter = collections.Counter()
    n_participates: collections.Counter = collections.Counter()
    partner_count: collections.Counter = collections.Counter()
    gene_compartment: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for step in net["steps"]:
        members = set(step["catalysts"])
        for side in ("in", "out"):
            for p in step[side]:
                members |= set(p["genes"])
        if not (1 <= len(members) <= 60):
            continue
        where = comp_of_step.get(step["id"], {}).get("compartments", [])
        for g in step["catalysts"]:
            n_catalyses[g] += 1
        for g in members - set(step["catalysts"]):
            n_participates[g] += 1
        for g in members:
            partner_count[g] += len(members) - 1
            for c in where:
                gene_compartment[g][c] += 1

    # --- signed regulon + PPI degree from the assembled cell ------------------------------------------------
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    n_up: collections.Counter = collections.Counter()
    n_down: collections.Counter = collections.Counter()
    n_unsigned: collections.Counter = collections.Counter()
    n_upstream: collections.Counter = collections.Counter()
    for e in D.get("reg", []):
        try:
            s, t = int(e[0]), int(e[1])
            w = int(e[2]) if len(e) > 2 else 0
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= s < N and 0 <= t < N) or s == t:
            continue
        if w == 0:
            n_unsigned[names[s]] += 1
        elif w > 0:
            n_up[names[s]] += 1
        else:
            n_down[names[s]] += 1
        n_upstream[names[t]] += 1
    ppi_degree: collections.Counter = collections.Counter()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            ppi_degree[names[a]] += 1
            ppi_degree[names[b]] += 1
    del D

    # --- lesion classes: the incumbent's features, included so its set is a strict subset -------------------
    try:
        from multilayer_reason import classify
        cmap = json.loads((OUT / "consequence_map.json").read_text())
        dark = {}
        try:
            for g, v in json.loads((OUT / "dark_function_table.json").read_text())["table"].items():
                dark[g] = v.get("function")
        except Exception:
            pass
        lesion: dict[str, set] = {}
        for g in set(list(paths) + list(cmap)):
            m = cmap.get(g, {})
            pw = paths.get(g, []) + (m.get("pathways_member") or [])
            lesion[g] = set(classify(g, pw, m.get("compartments") or [], dark.get(g)))
        del cmap
    except Exception as exc:                                        # pragma: no cover - annotation is optional
        report(f"  [warn] lesion classes unavailable ({exc}); continuing without them")
        lesion = {}

    # --- choose which categorical channels exist, by TRAINING frequency only -------------------------------
    # REPRODUCIBILITY, and this was a real bug rather than a precaution.  The first version iterated `train_set`
    # (a Python set) and cut with Counter.most_common(), which breaks ties by FIRST-INSERTION order.  Set
    # iteration order depends on hash randomisation, so two runs of this file in separate processes selected
    # DIFFERENT tied pathways, grew different conjunctions, and scored adrnconj 0.2218 vs 0.2202 on cohort 1.
    # Iterating in sorted order and breaking ties by name makes the channel set a function of the data alone.
    path_freq: collections.Counter = collections.Counter()
    comp_freq: collections.Counter = collections.Counter()
    lesion_freq: collections.Counter = collections.Counter()
    for g in sorted(train_set):
        for p in sorted(set(paths.get(g, ()))):
            path_freq[p] += 1
        for c in sorted(gene_compartment.get(g, ())):
            comp_freq[c] += 1
        for c in sorted(lesion.get(g, ())):
            lesion_freq[c] += 1

    def ranked(counter: collections.Counter, limit: int | None = None, floor: int = 0) -> list[str]:
        items = sorted(((k, v) for k, v in counter.items() if v >= floor), key=lambda kv: (-kv[1], kv[0]))
        return [k for k, _ in (items[:limit] if limit else items)]

    keep_paths = ranked(path_freq, limit=TOP_PATHWAYS)
    keep_comps = ranked(comp_freq, floor=20)
    keep_lesion = ranked(lesion_freq, floor=5)

    columns: list[tuple[str, np.ndarray]] = []

    def add(name: str, fn) -> None:
        col = np.zeros(len(all_genes), np.float32)
        for g, i in idx.items():
            if fn(g):
                col[i] = 1.0
        columns.append((name, col))

    for p in keep_paths:
        pset = {g for g in idx if p in paths.get(g, ())}
        add(f"pathway:{p}", pset.__contains__)
    for c in keep_comps:
        add(f"compartment:{c}", lambda g, c=c: c in gene_compartment.get(g, ()))
    for c in keep_lesion:
        add(f"lesion:{c}", lambda g, c=c: c in lesion.get(g, ()))

    # graded structural properties, cut at fixed thresholds so each column stays a named indicator
    for name, counter, cuts in (
        ("catalyses", n_catalyses, (1, 5, 20)),
        ("participates", n_participates, (1, 5, 20)),
        ("reaction_partners", partner_count, (1, 10, 50)),
        ("regulon_up", n_up, (1, 5, 25)),
        ("regulon_down", n_down, (1, 5, 25)),
        ("chip_unsigned", n_unsigned, (1, 50, 250)),
        ("upstream_tfs", n_upstream, (1, 5, 25)),
        ("ppi_degree", ppi_degree, (1, 10, 50)),
        ("n_pathways", collections.Counter({g: len(v) for g, v in paths.items()}), (1, 10, 40)),
    ):
        for cut in cuts:
            add(f"{name}>={cut}", lambda g, c=counter, k=cut: c.get(g, 0) >= k)
    add("has_signed_regulon", lambda g: (n_up.get(g, 0) + n_down.get(g, 0)) > 0)
    add("no_annotation_context", lambda g: not (paths.get(g) or n_catalyses.get(g) or n_participates.get(g)
                                                or n_up.get(g) or n_down.get(g)))

    matrix = np.stack([c for _, c in columns], axis=1)
    names_out = [n for n, _ in columns]
    live = matrix.sum(0)
    keep = (live >= 10) & (live <= len(all_genes) - 10)          # drop constants and near-constants
    matrix, names_out = matrix[:, keep], [n for n, k in zip(names_out, keep) if k]
    report(f"  named channels: {matrix.shape[1]} kept of {len(columns)} built "
           f"({len(keep_paths)} pathways, {len(keep_comps)} compartments, {len(keep_lesion)} lesion classes, "
           f"rest structural); mean channels per gene {matrix.sum(1).mean():.1f}")
    return matrix, names_out


# ---------------------------------------------------------------------------------------------------------------
# ADRN-3 conjunction growth, extended to a multi-output target
# ---------------------------------------------------------------------------------------------------------------
def candidate_pool(n_channels: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    pool: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(pool) < CANDIDATES:
        arity = int(rng.choice(ARITIES))
        pick = tuple(sorted(rng.choice(n_channels, size=arity, replace=False).tolist()))
        if pick not in seen:
            seen.add(pick)
            pool.append(pick)
    return pool


def grow(signed: np.ndarray, residual: np.ndarray, pool: list[tuple[int, ...]], *,
         max_grown: int, threshold: float) -> list[tuple[int, ...]]:
    """The ADRN-3 rule -- score = |mean(product x error)| * sqrt(n) -- with one extension.

    ADRN-3 was defined for a SCALAR error.  The target here is a 60-dimensional component mixture, so each
    candidate gets one score per component and is ranked by its MAXIMUM.  For a single component this reduces
    exactly to the published rule.  The residual is standardised per component first, so the threshold keeps its
    meaning as a t-like statistic instead of tracking whichever component happens to have the largest scale.
    """

    n = len(signed)
    scale = residual.std(axis=0, ddof=1)
    scale[scale <= 0] = 1.0
    standard = (residual / scale[None, :]).astype(np.float32)
    # chunked: the full candidates x n product matrix is ~0.4 GB at 20k candidates and is only ever needed
    # one block at a time
    score = np.empty(len(pool), np.float64)
    for start in range(0, len(pool), 2048):
        block = pool[start:start + 2048]
        products = np.stack([np.prod(signed[:, list(p)], axis=1) for p in block])   # block x n
        means = (products @ standard) / n                                           # block x components
        score[start:start + len(block)] = np.abs(means).max(axis=1) * np.sqrt(n)
    order = np.argsort(-score)
    return [pool[int(i)] for i in order[:max_grown] if score[i] >= threshold]


def expand(signed: np.ndarray, conjunctions: list[tuple[int, ...]]) -> np.ndarray:
    if not conjunctions:
        return np.zeros((len(signed), 0), np.float32)
    return np.stack([np.prod(signed[:, list(c)], axis=1) for c in conjunctions], axis=1)


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
    gram = design.T @ design + penalty * np.eye(design.shape[1], dtype=design.dtype)
    return np.linalg.solve(gram, design.T @ y)


def ridge_apply(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1) @ weights


def rotate_pca(features: np.ndarray) -> np.ndarray:
    centred = features - features.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return (centred @ vt.T).astype(np.float32)                    # full rank: nothing is discarded


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 110)
    report("ADRN-3 CONJUNCTIONS ON THE SEALED KNOCKOUT PROTOCOL -- named channels, held-out knockouts")
    report("=" * 110)
    report("  PREDECLARED: succeeds only if adrnconj beats the frequency baseline AND its own shuffle AND its")
    report("  own linear part, with adrnrot losing most of the gain. Any other pattern is a failure.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gi = {g: i for i, g in enumerate(genes)}
    A = np.abs(z["M"])
    tide = (A >= TAU).mean(0) >= 0.05

    sealed: set[str] = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    train = [k for k in kos if k not in sealed]
    tidx = [ki[k] for k in train]
    report(f"\n  readout {len(kos):,} perturbations x {len(genes):,} genes; tide {int(tide.sum()):,}")
    report(f"  training pool {len(train):,} (both sealed cohorts, {len(sealed)} knockouts, removed BEFORE "
           f"anything is fitted)")

    # ---------------- the shared output vocabulary, identical to response_basis.py ----------------
    from sklearn.decomposition import NMF
    X = A[tidx].copy()
    X[:, tide] = 0.0
    report(f"  factorising {X.shape[0]:,} x {X.shape[1]:,} into {K} components ...")
    nmf = NMF(n_components=K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    report(f"    reconstruction error {nmf.reconstruction_err_:.1f}")
    del X

    # ---------------- named channels ----------------
    universe = sorted(set(kos) | sealed)
    channels, channel_names = build_channels(universe, set(train), report)
    urow = {g: i for i, g in enumerate(universe)}
    train_channels = channels[[urow[k] for k in train]]

    # ---------------- linear part, then growth on its residual ----------------
    rng = np.random.default_rng(SEED)
    threshold = np.median(train_channels, axis=0)
    signed_train = np.where(train_channels > threshold[None, :], 1.0, -1.0).astype(np.float32)
    pool = candidate_pool(channels.shape[1], np.random.default_rng(SEED + 1))

    base_weights = ridge_fit(train_channels, W, PENALTY)
    residual = W - ridge_apply(train_channels, base_weights)
    learned = grow(signed_train, residual, pool, max_grown=MAX_GROWN, threshold=THRESHOLD)
    shuffled = grow(signed_train, rng.permutation(residual), pool,
                    max_grown=max(len(learned), 1), threshold=float("-inf"))
    random_conj = [pool[int(i)] for i in
                   np.random.default_rng(SEED + 2).choice(len(pool), size=max(len(learned), 1), replace=False)]
    report(f"\n  conjunctions grown on the TRAINING residual: {len(learned)} cleared threshold {THRESHOLD} "
           f"out of {CANDIDATES:,} candidates, arities {ARITIES}")
    if learned:
        report("    top 5 by score, as named products:")
        for conj in learned[:5]:
            report("      " + "  AND  ".join(channel_names[c] for c in conj))
    else:
        report("    NONE cleared the threshold -- the mechanism is INERT on this task and adrnconj == adrnlin")

    rotated = rotate_pca(channels)
    rot_train = rotated[[urow[k] for k in train]]
    rot_threshold = np.median(rot_train, axis=0)
    rot_signed = np.where(rot_train > rot_threshold[None, :], 1.0, -1.0).astype(np.float32)
    rot_weights = ridge_fit(rot_train, W, PENALTY)
    rot_residual = W - ridge_apply(rot_train, rot_weights)
    rot_learned = grow(rot_signed, rot_residual, pool, max_grown=MAX_GROWN, threshold=THRESHOLD)
    report(f"  same growth on a full-rank PCA rotation of the identical channels: {len(rot_learned)} cleared")

    # ---------------- two confound controls the first draft was missing ----------------
    # adrnles: ONLY the lesion-class channels, through the identical ridge. response_basis reaches its mixture by
    #   AVERAGING the training loadings of a knockout's classes; this arm uses the same classes but fits them, so
    #   any gap between adrnles and basis is the estimator and any gap between adrnlin and adrnles is the extra
    #   channels. Without this the two explanations are entangled.
    # adrnperm: the full channel matrix with its ROWS PERMUTED ACROSS GENES -- every knockout keeps a real
    #   annotation profile, just somebody else's. If this scores like adrnlin then the channels are decorative and
    #   the win is the NMF basis plus the ridge's intercept, not the biology.
    lesion_cols = [i for i, n in enumerate(channel_names) if n.startswith("lesion:")]
    channels_lesion = channels[:, lesion_cols] if lesion_cols else channels[:, :1]
    channels_perm = channels[np.random.default_rng(SEED + 3).permutation(len(channels))]
    report(f"\n  confound controls: adrnles uses {channels_lesion.shape[1]} lesion channels only; "
           f"adrnperm uses all {channels.shape[1]} with rows permuted across genes")

    # ---------------- assemble every arm's design and fit ----------------
    train_rows = np.array([urow[k] for k in train])

    def make_design(matrix: np.ndarray, conjunctions):
        thr = np.median(matrix[train_rows], axis=0)

        def build(rows: np.ndarray) -> np.ndarray:
            base = matrix[rows]
            sgn = np.where(base > thr[None, :], 1.0, -1.0).astype(np.float32)
            return np.concatenate([base, expand(sgn, conjunctions)], axis=1)

        return build

    arms = {
        "adrnlin": make_design(channels, []),
        "adrnconj": make_design(channels, learned),
        "adrnshuf": make_design(channels, shuffled),
        "adrnrand": make_design(channels, random_conj),
        "adrnrot": make_design(rotated, rot_learned),
        "adrnles": make_design(channels_lesion, []),
        "adrnperm": make_design(channels_perm, []),
    }
    fitted = {}
    for arm, build in arms.items():
        dx = build(train_rows)
        fitted[arm] = ridge_fit(dx, W, PENALTY)
        report(f"    {arm:<9} design {dx.shape[1]:>4d} columns")

    globalmix = W.mean(0)

    # ---------------- predict both sealed cohorts ----------------
    written: list[tuple[str, str, Path]] = []
    for tag in ("", "_holdout"):
        dp = OUT / f"ko_pred_dossiers{tag}.json"
        if not dp.exists():
            continue
        cohort = sorted(json.loads(dp.read_text()))
        rows = np.array([urow[k] for k in cohort])
        for arm, build in arms.items():
            dx = build(rows)
            mixes = ridge_apply(dx, fitted[arm])
            preds = {}
            for j, k in enumerate(cohort):
                mix = mixes[j]
                if not np.isfinite(mix).all():
                    mix = globalmix
                score = mix @ H
                score[tide] = 0.0
                if k in gi:
                    score[gi[k]] = 0.0
                order = np.argsort(-score)[:NPRED]
                preds[k] = {
                    "variant": arm,
                    "reasoning": f"arm {arm}: {dx.shape[1]} design columns -> ridge -> mixture of {K} "
                                 f"response components -> ranked over {len(genes):,} genes",
                    "predicted_genes": [genes[i] for i in order if score[i] > 0],
                }
            p = OUT / f"ko_predictions_{arm}{tag}.json"
            json.dump(preds, open(p, "w"), indent=1)
            empty = sum(1 for v in preds.values() if not v["predicted_genes"])
            written.append((arm, tag, p))
            report(f"  {tag or 'cohort1':<9} {arm:<9} {len(cohort)} knockouts, empty {empty}/{len(cohort)} "
                   f"-> {p.name}")

    json.dump({"test": "adrn_conjunctions_sealed_ko",
               "k": K, "npred": NPRED, "n_train": len(train), "n_sealed": len(sealed),
               "n_channels": int(channels.shape[1]), "channel_names": channel_names,
               "candidates": CANDIDATES, "arities": list(ARITIES), "threshold": THRESHOLD,
               "n_learned": len(learned), "n_rot_learned": len(rot_learned),
               "learned_named": [[channel_names[c] for c in conj] for conj in learned],
               "log": log},
              open(OUT / "adrn_ko_conjunctions.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_ko_conjunctions.json'}")
    report("\n  now score with:  python3 ko_predict.py score <prediction file> [_holdout]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
