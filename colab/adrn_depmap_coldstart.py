"""DEPMAP COLD-START ENTITY LEARNER: predict a gene's dependency propensity from biology alone.

THE TASK.  For a gene never seen in training, predict mu_g -- its average CRISPR dependency across ~1,100 cell
lines, after removing the per-line marginal.  This is the spec's Task 1: universal gene-level phenotype, from
gene biology and partners only.  No dependency value for the held-out gene, its paralogs, or anything else is
ever read.

WHY FOUR SPLIT TYPES AND NOT ONE.  A random gene split is the easy case: a held-out ribosomal protein still has
fifty ribosomal proteins in training, so the model can succeed by recognising the family rather than by
understanding it.  The three structured splits remove exactly that crutch, each in a different way.

    random     the usual split.  Reported so the structured splits can be read against it.
    pfam       whole Pfam domain families held out together.  No paralog of a test gene is in training.
    complex    whole protein complexes held out together.  The strongest test: complex members share dependency
               almost by definition, so this removes the single most predictive relationship available.
    pathway    whole Reactome pathways held out together.

A model that scores well on `random` and collapses on `complex` has been doing lookup, not biology, and the gap
between those two numbers is the honest measure of what it learned.

LEAKAGE, closed explicitly.
    - `ess`, `ess_src`, `ess_prob` and `dep_frac` are DepMap-derived and are asserted absent from the feature
      matrix at build time; the run aborts rather than warns if one appears.
    - CO-DEPENDENCY IS EXCLUDED FROM THE PARTNER SETS.  codep edges are correlations computed across the same
      DepMap matrix this target comes from, so a co-dependency partner is a function of the answer.  Partners
      here are protein complexes and PPI only.  A separate arm prices what codep would have added, clearly
      labelled, so the number exists without contaminating the headline.
    - Channel SELECTION uses gene membership frequency only and never the target, so it is target-blind.

THE GATE, taken verbatim from the spec's Phase 2.  ADRN must beat ALL of:
    annotation ridge · sparse polynomial · gradient-boosted trees · partner mean · raw partner bag
"global mean" is added as the floor, because a target with a strong central tendency can make everything look
good.  Anything that does not clear the floor is not a result.

PREDECLARED: ADRN passes only if `adrn` beats every one of those five arms past the MDE on at least the `random`
split, and the claim is only interesting if it also survives `complex`.  Five previous ADRN mechanisms and the
named-partner rerun all failed their gates; there is no prior reason to expect this one to pass, and the value of
running it is that the cold-start entity task is the one setting the spec argues is ADRN's strongest substrate.
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
FOLDS = 5
PENALTY = 10.0
MAX_GROWN = 64
CANDIDATES = 20_000
THRESHOLD = 4.0
ARITIES = (2, 3, 5)
POLY_TOP = 40
BANNED = {"ess", "ess_src", "ess_prob", "dep_frac"}


def depmap_target(report):
    import pandas as pd
    frame = pd.read_csv(SP / "CRISPRGeneEffect.csv", index_col=0)
    frame.columns = [c.split(" (")[0] for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    values = frame.to_numpy(np.float32)
    genes = list(frame.columns)
    keep = np.isfinite(values).sum(0) >= 100
    values, genes = values[:, keep], [g for g, k in zip(genes, keep) if k]
    line_mean = np.nanmean(values, axis=1, keepdims=True)
    target = np.nanmean(values - line_mean, axis=0)
    ok = np.isfinite(target)
    report(f"  DepMap: {values.shape[0]:,} lines x {values.shape[1]:,} genes (>=100 measured lines); "
           f"target sd {target[ok].std():.4f}")
    return [g for g, o in zip(genes, ok) if o], target[ok].astype(np.float32)


def grouping(genes: list[str], report):
    """Pfam family, complex and pathway labels per gene, for the structured splits."""
    D = json.load(open(OUT / "cell_complete.json"))
    nm = [g["name"] for g in D["genes"]]
    cplx: dict[str, str] = {}
    for k_, cs in (D.get("gene2cplx") or {}).items():
        try:
            g = nm[int(k_)]
        except (TypeError, ValueError, IndexError):
            continue
        lst = sorted(str(c) for c in (cs if isinstance(cs, list) else [cs]))
        if lst:
            cplx[g] = lst[0]
    path: dict[str, str] = {}
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > 3:
            for g in parts[2:]:
                path.setdefault(g, parts[0])
    members: dict[str, list[str]] = collections.defaultdict(list)
    for c, mem in ((c, []) for c in []):
        pass
    cplx_members: dict[str, list[str]] = collections.defaultdict(list)
    for g, c in cplx.items():
        cplx_members[c].append(g)
    ppi: dict[str, list[str]] = collections.defaultdict(list)
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < len(nm) and 0 <= b_ < len(nm) and a_ != b_:
            ppi[nm[a_]].append(nm[b_])
            ppi[nm[b_]].append(nm[a_])
    codep: dict[str, list[str]] = collections.defaultdict(list)
    for a_, v in (D.get("codep") or {}).items():
        try:
            codep[nm[int(a_)]].extend(nm[int(b)] for b, _ in v)
        except (TypeError, ValueError, IndexError):
            continue
    del D
    pfam: dict[str, str] = {}
    try:
        raw = json.loads((OUT / "prot_pfam.json").read_text())
        for k_, v in raw.items():
            if k_.isdigit() and int(k_) < len(nm) and v:
                pfam[nm[int(k_)]] = sorted(v)[0]
    except Exception as exc:
        report(f"  [warn] pfam unavailable ({exc})")

    groups = {}
    for name, table in (("pfam", pfam), ("complex", cplx), ("pathway", path)):
        lab = [table.get(g, f"__own__{g}") for g in genes]
        groups[name] = np.array(lab)
        n_multi = sum(1 for _, c in collections.Counter(lab).items() if c > 1)
        report(f"  {name:<8} groups: {len(set(lab)):,} distinct, {n_multi:,} with >1 gene")
    groups["random"] = np.array([f"__own__{g}" for g in genes])
    partner_sets = {}
    for g in genes:
        partner_sets[g] = {
            "clean": sorted(set(cplx_members.get(cplx.get(g, ""), []) + ppi.get(g, [])) - {g}),
            "codep": sorted(set(codep.get(g, [])) - {g}),
        }
    return groups, partner_sets


def folds_for(labels: np.ndarray, seed: int) -> list[np.ndarray]:
    uniq = sorted(set(labels.tolist()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    assign = {uniq[int(u)]: i % FOLDS for i, u in enumerate(order)}
    where = np.array([assign[l] for l in labels])
    return [np.where(where == f)[0] for f in range(FOLDS)]


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 104)
    report("DEPMAP COLD-START ENTITY LEARNER -- predict a gene's dependency propensity from biology alone")
    report("=" * 104)
    report("  GATE (spec Phase 2): ADRN must beat ridge, sparse polynomial, trees, partner mean and raw bag.")
    report("  Co-dependency is EXCLUDED from partners: codep is computed from this same DepMap matrix.")

    genes, target = depmap_target(report)
    groups, partners = grouping(genes, report)

    base, base_names = A.build_channels(genes, set(genes), report)
    extra, extra_names, tier_b_start = C.extra_channels(genes, set(genes), report)
    channels = np.concatenate([base, extra[:, :tier_b_start]], axis=1)
    names = base_names + extra_names[:tier_b_start]
    leaked = {n for n in names if any(b in n.lower() for b in BANNED)}
    if leaked:
        raise SystemExit(f"LEAK: DepMap-derived channels reached the matrix: {sorted(leaked)[:5]}")
    report(f"  channels: {channels.shape[1]} named columns; asserted free of {sorted(BANNED)}")

    gi = {g: i for i, g in enumerate(genes)}
    cand_rng = np.random.default_rng(A.SEED + 1)
    pool: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(pool) < CANDIDATES:
        ar = int(cand_rng.choice(ARITIES))
        pick = tuple(sorted(cand_rng.choice(channels.shape[1], size=ar, replace=False).tolist()))
        if pick not in seen:
            seen.add(pick)
            pool.append(pick)

    from sklearn.ensemble import HistGradientBoostingRegressor

    ARMS = ["global_mean", "ridge", "poly", "trees", "partner_mean", "raw_bag", "adrn",
            "partner_mean_CODEP"]
    summary: dict[str, dict] = {}

    for split in ("random", "pfam", "complex", "pathway"):
        blocks = folds_for(groups[split], A.SEED + 5)
        per_fold: dict[str, list[float]] = {a: [] for a in ARMS}
        for f, held in enumerate(blocks):
            mask = np.ones(len(genes), bool)
            mask[held] = False
            tr = np.where(mask)[0]
            train_set = {genes[i] for i in tr}
            y_tr, y_te = target[tr], target[held]
            X_tr, X_te = channels[tr], channels[held]

            per_fold["global_mean"].append(rmse(np.full(len(held), y_tr.mean()), y_te))

            w = A.ridge_fit(X_tr, y_tr[:, None], PENALTY)
            per_fold["ridge"].append(rmse(A.ridge_apply(X_te, w).ravel(), y_te))

            # sparse polynomial: pairwise products of the channels most correlated with the target ON TRAIN
            corr = np.abs((X_tr - X_tr.mean(0)).T @ (y_tr - y_tr.mean())) / (len(tr) + 1e-9)
            top = np.argsort(-corr)[:POLY_TOP]
            def poly(M):
                sub = M[:, top]
                prods = np.einsum("ni,nj->nij", sub, sub)[:, *np.triu_indices(POLY_TOP, 1)]
                return np.concatenate([M, prods], axis=1)
            wp = A.ridge_fit(poly(X_tr), y_tr[:, None], PENALTY)
            per_fold["poly"].append(rmse(A.ridge_apply(poly(X_te), wp).ravel(), y_te))

            gb = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X_tr, y_tr)
            per_fold["trees"].append(rmse(gb.predict(X_te), y_te))

            for key, arm in (("clean", "partner_mean"), ("codep", "partner_mean_CODEP")):
                pred = np.full(len(held), y_tr.mean())
                for r, i in enumerate(held):
                    vals = [target[gi[p]] for p in partners[genes[i]][key]
                            if p in gi and genes[gi[p]] in train_set]
                    if vals:
                        pred[r] = float(np.mean(vals))
                per_fold[arm].append(rmse(pred, y_te))

            bag = np.zeros_like(channels)
            for i, g in enumerate(genes):
                rows = [gi[p] for p in partners[g]["clean"] if p in gi]
                bag[i] = channels[rows].mean(0) if rows else channels[i]
            wb = A.ridge_fit(np.concatenate([X_tr, bag[tr]], 1), y_tr[:, None], PENALTY)
            per_fold["raw_bag"].append(
                rmse(A.ridge_apply(np.concatenate([X_te, bag[held]], 1), wb).ravel(), y_te))

            thr = np.median(X_tr, axis=0)
            sgn_tr = np.where(X_tr > thr[None, :], 1.0, -1.0).astype(np.float32)
            resid = (y_tr - A.ridge_apply(X_tr, w).ravel())[:, None]
            learned = []
            n = len(sgn_tr)
            sd = resid.std(axis=0, ddof=1)
            sd[sd <= 0] = 1.0
            std = (resid / sd[None, :]).astype(np.float32)
            sc = np.empty(len(pool))
            for st in range(0, len(pool), 2048):
                blk = pool[st:st + 2048]
                prod = np.stack([np.prod(sgn_tr[:, list(p)], axis=1) for p in blk])
                sc[st:st + len(blk)] = np.abs((prod @ std) / n).max(axis=1) * np.sqrt(n)
            learned = [pool[int(i)] for i in np.argsort(-sc)[:MAX_GROWN] if sc[i] >= THRESHOLD]
            sgn_te = np.where(X_te > thr[None, :], 1.0, -1.0).astype(np.float32)
            wa = A.ridge_fit(np.concatenate([X_tr, A.expand(sgn_tr, learned)], 1), y_tr[:, None], PENALTY)
            per_fold["adrn"].append(rmse(
                A.ridge_apply(np.concatenate([X_te, A.expand(sgn_te, learned)], 1), wa).ravel(), y_te))
            if f == 0:
                report(f"\n  [{split}] fold 1: {len(tr):,} train / {len(held):,} held; "
                       f"{len(learned)} conjunctions grown")

        report(f"\n### {split} split   ({FOLDS} folds, RMSE, lower is better)")
        report(f"  {'arm':<20} {'RMSE':>8} {'sd':>8}")
        for a in ARMS:
            v = np.array(per_fold[a])
            report(f"  {a:<20} {v.mean():>8.4f} {v.std(ddof=1):>8.4f}")
        ref = np.array(per_fold["adrn"])
        report(f"\n  {'ADRN vs':<20} {'gap':>9} {'MDE':>8}  verdict   (negative = ADRN better)")
        rows = {}
        for a in ("ridge", "poly", "trees", "partner_mean", "raw_bag", "global_mean"):
            gap = ref - np.array(per_fold[a])
            mde = 3 * gap.std(ddof=1) / np.sqrt(FOLDS)
            res = abs(gap.mean()) > mde
            report(f"  {a:<20} {gap.mean():>+9.4f} {mde:>8.4f}  "
                   f"{'RESOLVED' if res else 'within noise'}"
                   f"{'  ADRN BETTER' if res and gap.mean() < 0 else ''}")
            rows[a] = {"gap": float(gap.mean()), "mde": float(mde), "resolved": bool(res)}
        summary[split] = {"means": {a: float(np.mean(per_fold[a])) for a in ARMS},
                          "adrn_vs": rows}

    report("\n### the honest cold-start summary: random vs structured")
    report(f"  {'arm':<20} " + " ".join(f"{s:>10}" for s in ("random", "pfam", "complex", "pathway")))
    for a in ARMS:
        report(f"  {a:<20} " + " ".join(f"{summary[s]['means'][a]:>10.4f}"
                                        for s in ("random", "pfam", "complex", "pathway")))

    json.dump({"test": "adrn_depmap_coldstart", "n_genes": len(genes),
               "n_channels": int(channels.shape[1]), "folds": FOLDS,
               "summary": summary, "log": log},
              open(OUT / "adrn_depmap_coldstart.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_depmap_coldstart.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
