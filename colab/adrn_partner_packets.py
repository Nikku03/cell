"""NAMED PER-PARTNER PACKETS: success condition 3 -- does named partner ingestion beat the raw partner bag?

WHAT THIS IS.  Items 1-4 of the spec's "build now" list are one experiment, and it is the one the whole ADRN
argument rests on.  Criterion-2 is where ADRN lost to a ZERO-PARAMETER bag of partners: learned conjunctions
0.0319 against the bag's 0.0340, RESOLVED against it, and losing to its own random and shuffled controls too.

The spec's diagnosis is that the partners were ingested as 64 DepMap principal components -- a rotation -- and
v4_conj_basis.py measured that rotation destroys conjunction growth entirely (0.0% / 1.6% retention, ridge
unchanged).  So the claim is that criterion-2 failed at INGESTION, and that presenting each partner as NAMED
annotation channels instead of compressed coordinates should recover it.

That claim is falsifiable in one run, and this is the run.

THE TASK, held identical to criterion-2's structure.  Each knockout is embedded from its partner set, the
nearest training knockouts are retrieved by cosine, their centred profiles are averaged, and the prediction is
scored as recall@50 against the knockout's true top-50 non-tide movers.  Nothing about the task, the truth, the
retrieval or the metric changes between arms -- ONLY how a partner is written down.

SCOPE NOTE, stated rather than buried.  The published criterion-2 numbers (bag 0.0340, Set Transformer 0.0297,
ADRN 0.0319) were measured on dense_stage1's build: 10,337 knockouts x 8,175 genes.  This runs on the gwps
rebuild that every sealed-protocol result uses: 5,120 x 8,246.  Absolute recall is therefore NOT comparable to
those published figures.  Every arm here is measured on the same rows with the same blocks, so the ORDERING and
the paired gaps are valid, and that is what the question turns on.

ARMS -- the spec's section 12 control matrix for the "Named channels" and "Partners" rows, in one frame.

  representation of a partner
    bag_named        mean of the partners' 696 NAMED annotation channels.  0 parameters.  THE TEST.
    bag_pca          the identical information, full-rank PCA rotated before averaging.  Invertible, so nothing
                     is lost -- only the axis alignment.  If bag_named beats this, presentation is doing work.
    bag_perm         named channels permuted across genes: every partner keeps a real annotation profile, just
                     somebody else's.  Must lose.

  which partners
    self_only        the query gene's own channels, no partners at all.  Prices how much the partners add.
    partner_random   the same count and type mix, uniformly drawn partners.  Prices whether it matters WHO.

  the ADRN mechanism, on the named basis it demonstrably requires
    conj_named       bag_named plus grown sparse conjunctions of named channels.
    conj_shuffled    conjunctions grown against a PERMUTED target, same count.  Prices top-k selection on noise.
    conj_random      conjunctions drawn at random from the same candidate pool, same count and arity mix.

PREDECLARED, and this is the whole point of running it.
    SUCCESS CONDITION 3 passes only if bag_named beats bag_pca past the MDE.  That is the ingestion claim.
    The ADRN MECHANISM passes only if conj_named beats bag_named AND conj_shuffled AND conj_random.  It has
    failed that test five times; a sixth failure on the basis it was said to need would end the conjunction line.
    If bag_named also loses to bag_pca, then the named-basis story does not transfer to partner ingestion and
    the criterion-2 failure needs an explanation other than the input representation.
"""

from __future__ import annotations

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
TRUTH_N = 50
RECALL_K = 50
RETRIEVE = 10
MAX_PARTNERS = 7
BLOCKS = 3
MIN_SPEC = 5
MAX_GROWN = 64
CANDIDATES = 20_000
THRESHOLD = 4.0
ARITIES = (2, 3, 5)
SEED = A.SEED


def partner_table(names_wanted: set[str], report):
    """codep / complex / PPI partners per gene, exactly the three relation types criterion-2 used."""

    D = json.load(open(OUT / "cell_complete.json"))
    nm = [g["name"] for g in D["genes"]]
    codep: dict[str, list[str]] = {}
    for a_, v in (D.get("codep") or {}).items():
        try:
            codep.setdefault(nm[int(a_)], []).extend(nm[int(b)] for b, _ in v)
        except (TypeError, ValueError, IndexError):
            continue
    cplx: dict[str, list[str]] = {}
    members: dict[str, list[int]] = {}
    for k_, cs in (D.get("gene2cplx") or {}).items():
        try:
            gi_ = int(k_)
        except (TypeError, ValueError):
            continue
        for c in (cs if isinstance(cs, list) else [cs]):
            members.setdefault(str(c), []).append(gi_)
    for c, mem in members.items():
        for x in mem:
            cplx.setdefault(nm[x], []).extend(nm[y] for y in mem if y != x)
    ppi: dict[str, list[str]] = {}
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < len(nm) and 0 <= b_ < len(nm) and a_ != b_:
            ppi.setdefault(nm[a_], []).append(nm[b_])
            ppi.setdefault(nm[b_], []).append(nm[a_])
    del D
    report(f"  partner sources: codep {len(codep):,} genes, complex {len(cplx):,}, ppi {len(ppi):,}")
    return (("codep", codep), ("complex", cplx), ("ppi", ppi))


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 108)
    report("NAMED PER-PARTNER PACKETS -- does named ingestion beat the raw bag, and does ADRN beat its controls?")
    report("=" * 108)
    report("  PREDECLARED: success condition 3 passes only if bag_named beats bag_pca past the MDE.")
    report("  The ADRN mechanism passes only if conj_named beats bag_named AND conj_shuffled AND conj_random.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    nontide = ~tide

    sealed: set[str] = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    train_pool = sorted(k for k in kos if k not in sealed)

    universe = sorted(set(kos) | sealed)
    base, _ = A.build_channels(universe, set(train_pool), report)
    extra, _, tier_b_start = C.extra_channels(universe, set(train_pool), report)
    channels = np.concatenate([base, extra[:, :tier_b_start]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    report(f"  named channel basis: {channels.shape[1]} columns")

    types = partner_table(set(universe), report)

    # ---- keep knockouts that have a scorable profile and at least one annotated partner ----
    spec_count = (Amat >= A.TAU)[:, nontide].sum(1)
    usable = []
    partners: list[list[tuple[str, list[str]]]] = []
    for i, k in enumerate(kos):
        if spec_count[i] < MIN_SPEC or k not in urow:
            continue
        lists = [(t, [g for g in src.get(k, [])[:MAX_PARTNERS] if g in urow]) for t, src in types]
        if not any(gs for _, gs in lists):
            continue
        usable.append(i)
        partners.append(lists)
    usable = np.array(usable)
    ko_names = [kos[i] for i in usable]
    N = len(usable)
    degrees = np.array([sum(len(gs) for _, gs in p) for p in partners])
    report(f"  usable knockouts: {N:,} (>= {MIN_SPEC} specific movers AND >= 1 annotated partner); "
           f"mean partners {degrees.mean():.2f}")

    Amat = Amat[usable]
    is_sealed = np.array([k in sealed for k in ko_names])

    # ---- partner variants: real, random (count- and type-matched) ----
    rng = np.random.default_rng(SEED)
    pool_genes = [g for g in universe]
    rand_partners = [[(t, [pool_genes[int(x)] for x in rng.integers(0, len(pool_genes), len(gs))])
                      for t, gs in p] for p in partners]

    # ---- three representations of the SAME channel matrix ----
    centred = channels - channels.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    rotated = (centred @ vt.T).astype(np.float32)                       # full rank, invertible
    permuted = channels[np.random.default_rng(SEED + 3).permutation(len(channels))]

    def embed(matrix: np.ndarray, per_ko, include_self: bool) -> np.ndarray:
        E = np.zeros((N, matrix.shape[1]), np.float32)
        for i, lists in enumerate(per_ko):
            rows = [urow[ko_names[i]]] if include_self else []
            for _, gs in lists:
                rows.extend(urow[g] for g in gs)
            if rows:
                E[i] = matrix[rows].mean(0)
        return E

    embeddings = {
        "bag_named": embed(channels, partners, True),
        "bag_pca": embed(rotated, partners, True),
        "bag_perm": embed(permuted, partners, True),
        "self_only": embed(channels, [[(t, []) for t, _ in p] for p in partners], True),
        "partner_random": embed(channels, rand_partners, True),
    }

    # ---- blocks over knockouts; sealed knockouts are never used as retrieval donors ----
    order = np.random.default_rng(SEED + 7).permutation(N)
    blocks = np.array_split(order, BLOCKS)

    def score(E: np.ndarray, held: np.ndarray, donors: np.ndarray, truth: np.ndarray,
              centred_profiles: np.ndarray) -> float:
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sim = En[held] @ En[donors].T
        take = np.argsort(-sim, axis=1)[:, :RETRIEVE]
        hits = []
        for r, h in enumerate(held):
            pred = centred_profiles[donors[take[r]]].mean(0)
            pred = np.where(nontide, pred, -np.inf)
            top = np.argpartition(-pred, RECALL_K)[:RECALL_K]
            hits.append(len(set(top.tolist()) & set(truth[r].tolist())) / RECALL_K)
        return float(np.mean(hits))

    def grow(signed: np.ndarray, residual: np.ndarray, pool, max_grown: int, threshold: float):
        n = len(signed)
        sd = residual.std(axis=0, ddof=1)
        sd[sd <= 0] = 1.0
        std = (residual / sd[None, :]).astype(np.float32)
        sc = np.empty(len(pool))
        for st in range(0, len(pool), 2048):
            blk = pool[st:st + 2048]
            prod = np.stack([np.prod(signed[:, list(p)], axis=1) for p in blk])
            sc[st:st + len(blk)] = np.abs((prod @ std) / n).max(axis=1) * np.sqrt(n)
        idx = np.argsort(-sc)
        return [pool[int(i)] for i in idx[:max_grown] if sc[i] >= threshold]

    cand_rng = np.random.default_rng(SEED + 1)
    pool_c: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(pool_c) < CANDIDATES:
        ar = int(cand_rng.choice(ARITIES))
        pick = tuple(sorted(cand_rng.choice(channels.shape[1], size=ar, replace=False).tolist()))
        if pick not in seen:
            seen.add(pick)
            pool_c.append(pick)

    arms = list(embeddings) + ["conj_named", "conj_shuffled", "conj_random"]
    per_block: dict[str, list[float]] = {a: [] for a in arms}
    grown_counts: list[int] = []

    for b, held in enumerate(blocks):
        mask = np.ones(N, bool)
        mask[held] = False
        donors = np.where(mask & ~is_sealed)[0]           # sealed knockouts never donate
        mu = Amat[donors].mean(0)
        S = np.where(nontide[None, :], Amat - mu[None, :], 0.0).astype(np.float32)
        truth = np.stack([np.argpartition(-np.where(nontide, S[i], -np.inf), TRUTH_N)[:TRUTH_N]
                          for i in held])

        for arm, E in embeddings.items():
            per_block[arm].append(score(E, held, donors, truth, S))

        # --- ADRN conjunctions grown on the DONOR rows only, target = donor bag embedding residual ---
        bag = embeddings["bag_named"]
        thr = np.median(bag[donors], axis=0)
        signed = np.where(bag[donors] > thr[None, :], 1.0, -1.0).astype(np.float32)
        target = S[donors][:, nontide][:, :256]           # a compact view of the donors' own responses
        w = A.ridge_fit(bag[donors], target, 10.0)
        resid = target - A.ridge_apply(bag[donors], w)
        learned = grow(signed, resid, pool_c, MAX_GROWN, THRESHOLD)
        grown_counts.append(len(learned))
        shuffled = grow(signed, np.random.default_rng(SEED + 991 * (b + 1)).permutation(resid),
                        pool_c, max(len(learned), 1), float("-inf"))
        randconj = [pool_c[int(i)] for i in np.random.default_rng(SEED + 2 + b).choice(
            len(pool_c), size=max(len(learned), 1), replace=False)]

        all_signed = np.where(bag > thr[None, :], 1.0, -1.0).astype(np.float32)
        for arm, conj in (("conj_named", learned), ("conj_shuffled", shuffled),
                          ("conj_random", randconj)):
            E = np.concatenate([bag, A.expand(all_signed, conj).astype(np.float32)], axis=1)
            per_block[arm].append(score(E, held, donors, truth, S))
        report(f"  block {b + 1}/{BLOCKS}: {len(held):,} held, {len(donors):,} donors, "
               f"{len(learned)} conjunctions grown")

    report(f"\n  random floor recall@{RECALL_K} = {RECALL_K / int(nontide.sum()):.4f}; "
           f"{BLOCKS} blocks; mean conjunctions grown {np.mean(grown_counts):.1f}")
    report(f"\n  {'arm':<16} {'recall@50':>10} {'sd':>8}   per block")
    means = {}
    for arm in arms:
        v = np.array(per_block[arm])
        means[arm] = float(v.mean())
        report(f"  {arm:<16} {v.mean():>10.4f} {v.std(ddof=1):>8.4f}   "
               + " ".join(f"{x:.4f}" for x in v))

    report(f"\n  {'contrast':<34} {'gap':>9} {'MDE':>8}  verdict")
    contrasts = [
        ("bag_named - bag_pca (INGESTION)", "bag_named", "bag_pca"),
        ("bag_named - bag_perm (leakage)", "bag_named", "bag_perm"),
        ("bag_named - self_only (partners)", "bag_named", "self_only"),
        ("bag_named - partner_random (WHO)", "bag_named", "partner_random"),
        ("conj_named - bag_named (MECHANISM)", "conj_named", "bag_named"),
        ("conj_named - conj_shuffled (noise)", "conj_named", "conj_shuffled"),
        ("conj_named - conj_random (structure)", "conj_named", "conj_random"),
    ]
    summary = {}
    for label, x, y in contrasts:
        gap = np.array(per_block[x]) - np.array(per_block[y])
        mde = 3 * gap.std(ddof=1) / np.sqrt(len(gap))
        resolved = abs(gap.mean()) > mde
        report(f"  {label:<34} {gap.mean():>+9.4f} {mde:>8.4f}  "
               f"{'RESOLVED' if resolved else 'within noise'}")
        summary[label] = {"gap": float(gap.mean()), "mde": float(mde), "resolved": bool(resolved)}

    json.dump({"test": "adrn_named_partner_packets", "n_knockouts": int(N),
               "n_channels": int(channels.shape[1]), "blocks": BLOCKS,
               "random_floor": RECALL_K / int(nontide.sum()),
               "means": means, "per_block": per_block, "contrasts": summary,
               "mean_grown": float(np.mean(grown_counts)), "log": log},
              open(OUT / "adrn_partner_packets.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_partner_packets.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
