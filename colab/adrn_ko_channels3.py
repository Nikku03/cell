"""PROTEOMICS, THE TF NETWORK, AND EVERY OTHER TABLE IN THE CELL: does more data help, and which part?

WHAT IS ADDED AND WHY IT IS SPLIT INTO THREE BLOCKS.

The 696-channel chan2a set uses the regulatory network only as DEGREE BUCKETS -- "has a signed regulon",
"upstream_tfs >= 5". It never uses the identity of a single transcription factor. That is the largest obvious
omission, and it is exactly the axis-aligned named-channel form that has been paying: "is regulated by GATA1" is a
named indicator a biologist can read, not a compressed number.

Proteomics, co-dependency, PTMs, drugs, 3D loops and cell-type breadth are all sitting unused in the same file.

The three blocks are priced separately because they have different evidential status, and pooling them would let a
win in one hide a failure in another.

    BLOCK T -- curated TF network identity.  `regulated_by:<TF>` from the SIGNED regulatory edges, which come from
        SIGNOR and TRRUST (literature-curated causal), plus NicheNet ligand->target membership.  UPDATES.md
        measured these against the thing being predicted: curated causal edges predict measured knockout movers at
        2.7x enrichment.  This is the block with a prior reason to work.

    BLOCK C -- ChIP binding identity.  `chip_bound_by:<TF>` from the 612k unsigned edges.  Measured, but
        bind_vs_reg.py showed ChIP binding overlaps Perturb-seq regulation at CHANCE (fold 0.96, p = 0.68,
        permutation-verified), and the regulon comparison put it at 1.1x against SIGNOR's 2.7x.  So this block is
        predicted to be inert, and it is included precisely because that prediction is falsifiable.

    BLOCK P -- proteomics and dependency measurements.  Protein abundance (ppm, 16,015 genes) and copy-number
        abundance as named deciles, PTM classes, co-dependency degree and strength, synthetic-lethal degree, drug
        targeting, cell-type expression breadth from the emask bitmask, 3D genome loops, biomarker status.  All
        measured, none of them the answer to this question.

PENALTY IS TUNED PER ARM, ON TRAINING DATA ONLY.  chan2a already carries 696 columns against 4,720 training rows;
these blocks push past 1,000.  A fixed ridge penalty would confound "more channels did not help" with "the penalty
was wrong for a wider design", so each arm picks its own penalty on a validation slice carved out of the TRAINING
knockouts.  The sealed cohorts are never touched by that choice.

PREDECLARED
    BLOCK T should help if curated regulation carries knockout-specific information.
    BLOCK C should NOT help -- it was measured at chance against this exact target.  If it does help, the
        chance-overlap result needs revisiting, and that would be the interesting outcome.
    BLOCK P is the open one.  Tier B of chan2 (essentiality, abundance buckets, FBA) already added nothing, so the
        prior is weak; richer proteomics is the fair retest.
    A permuted control must, as always, score below the frequency baseline.
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
TOP_TF = int(os.environ.get("CH3_TOP_TF", "220"))
TOP_CHIP = int(os.environ.get("CH3_TOP_CHIP", "220"))
TOP_LIGAND = int(os.environ.get("CH3_TOP_LIGAND", "80"))
PENALTIES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def blocks(all_genes: list[str], train_set: set[str], report):
    """Returns (matrix, names, index where block C starts, index where block P starts)."""

    idx = {g: i for i, g in enumerate(all_genes)}
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)

    def gname(key: str) -> str:
        return names[int(key)] if key.isdigit() and int(key) < N else key

    signed_by: dict[str, set[str]] = collections.defaultdict(set)   # target -> curated TFs
    chip_by: dict[str, set[str]] = collections.defaultdict(set)     # target -> ChIP-bound TFs
    for e in D.get("reg", []):
        try:
            s, t = int(e[0]), int(e[1])
            w = int(e[2]) if len(e) > 2 else 0
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= s < N and 0 <= t < N) or s == t:
            continue
        (signed_by if w != 0 else chip_by)[names[t]].add(names[s])

    ligand_targets: dict[str, set[str]] = {}
    for lig, targets in (D.get("nichenet") or {}).items():
        ligand_targets[gname(lig)] = {gname(str(t)) for t in targets}

    ptm_classes: dict[str, list[str]] = {}
    for g, v in (D.get("ptm") or {}).items():
        ptm_classes[gname(g)] = list(v.get("c") or []) if isinstance(v, dict) else []

    codep_deg: dict[str, int] = {}
    codep_max: dict[str, float] = {}
    for g, v in (D.get("codep") or {}).items():
        key = gname(g)
        try:
            codep_deg[key] = len(v)
            codep_max[key] = max((float(p[1]) for p in v), default=0.0)
        except (TypeError, ValueError, IndexError):
            pass

    sl_deg: collections.Counter = collections.Counter()
    for e in (D.get("sl") or []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N:
            sl_deg[names[a]] += 1
            sl_deg[names[b]] += 1

    n_drugs: dict[str, int] = {gname(g): len(v) for g, v in (D.get("drugs") or {}).items()}
    ppm: dict[str, float] = {}
    for g, v in (D.get("ppm") or {}).items():
        try:
            ppm[gname(g)] = float(v)
        except (TypeError, ValueError):
            pass
    abund: dict[str, float] = {}
    for g, v in (D.get("abund") or {}).items():
        try:
            abund[gname(g)] = float(v)
        except (TypeError, ValueError):
            pass
    breadth: dict[str, int] = {}
    for g, v in (D.get("emask") or {}).items():
        try:
            breadth[gname(g)] = bin(int(v)).count("1")
        except (TypeError, ValueError):
            pass
    has_loop = {gname(g) for g in (D.get("loops3d") or {})}
    is_biomarker = {gname(g) for g in (D.get("biomarkers") or {})}
    has_disease = {gname(g) for g in (D.get("otdis") or {})}
    del D

    def rank_by_train(mapping: dict[str, set[str]], limit: int) -> list[str]:
        freq: collections.Counter = collections.Counter()
        for g in sorted(train_set):
            for t in sorted(mapping.get(g, ())):
                freq[t] += 1
        return C.ranked(freq, limit=limit)

    keep_tf = rank_by_train(signed_by, TOP_TF)
    keep_chip = rank_by_train(chip_by, TOP_CHIP)
    lig_member: dict[str, set[str]] = collections.defaultdict(set)
    for lig, targets in ligand_targets.items():
        for t in targets:
            lig_member[t].add(lig)
    keep_lig = rank_by_train(lig_member, TOP_LIGAND)
    ptm_sets: dict[str, set[str]] = collections.defaultdict(set)
    for g, cs in ptm_classes.items():
        for c in cs:
            ptm_sets[g].add(c)
    keep_ptm = rank_by_train(ptm_sets, 30)

    columns: list[tuple[str, np.ndarray]] = []

    def add(name: str, fn) -> None:
        col = np.zeros(len(all_genes), np.float32)
        for g, i in idx.items():
            try:
                if fn(g):
                    col[i] = 1.0
            except (TypeError, ValueError):
                pass
        columns.append((name, col))

    def quantile_cuts(values: dict[str, float], n: int = 5) -> list[float]:
        vals = np.array([values[g] for g in sorted(train_set) if g in values], float)
        if len(vals) < 50:
            return []
        return [float(x) for x in np.quantile(vals, np.linspace(0, 1, n + 1)[1:-1])]

    # ---------------- BLOCK T: curated TF network ----------------
    for tf in keep_tf:
        add(f"regulated_by:{tf}", lambda g, t=tf: t in signed_by.get(g, ()))
    for lig in keep_lig:
        add(f"nichenet_target_of:{lig}", lambda g, l=lig: l in lig_member.get(g, ()))
    block_c_start = len(columns)

    # ---------------- BLOCK C: ChIP binding identity ----------------
    for tf in keep_chip:
        add(f"[chip] bound_by:{tf}", lambda g, t=tf: t in chip_by.get(g, ()))
    block_p_start = len(columns)

    # ---------------- BLOCK P: proteomics and dependency ----------------
    for label, values in (("ppm", ppm), ("abund", abund)):
        for q, cut in enumerate(quantile_cuts(values, 5)):
            add(f"[prot] {label}_q{q+1}>={cut:.4g}", lambda g, v=values, k=cut: v.get(g, 0.0) >= k)
        add(f"[prot] {label}_measured", lambda g, v=values: g in v)
    for c in keep_ptm:
        add(f"[prot] ptm:{c}", lambda g, c=c: c in ptm_sets.get(g, ()))
    for cut in (1, 10, 50):
        add(f"[prot] codep_degree>={cut}", lambda g, k=cut: codep_deg.get(g, 0) >= k)
    for cut in (0.2, 0.3, 0.4):
        add(f"[prot] codep_max>={cut}", lambda g, k=cut: codep_max.get(g, 0.0) >= k)
    for cut in (1, 3):
        add(f"[prot] synthetic_lethal_degree>={cut}", lambda g, k=cut: sl_deg.get(g, 0) >= k)
    for cut in (1, 5, 20):
        add(f"[prot] n_drugs>={cut}", lambda g, k=cut: n_drugs.get(g, 0) >= k)
    for q, cut in enumerate(quantile_cuts(breadth, 4)):
        add(f"[prot] celltype_breadth_q{q+1}>={cut:.4g}", lambda g, k=cut: breadth.get(g, 0) >= k)
    add("[prot] has_3d_loop", has_loop.__contains__)
    add("[prot] is_biomarker", is_biomarker.__contains__)
    add("[prot] has_disease_link", has_disease.__contains__)

    matrix = np.stack([c for _, c in columns], axis=1)
    out = [n for n, _ in columns]
    live = matrix.sum(0)
    keep = (live >= 10) & (live <= len(all_genes) - 10)
    drop_c = int((~keep[:block_c_start]).sum())
    drop_p = int((~keep[:block_p_start]).sum())
    matrix, out = matrix[:, keep], [n for n, k in zip(out, keep) if k]
    block_c_start -= drop_c
    block_p_start -= drop_p
    report(f"  new blocks: {matrix.shape[1]} kept of {len(columns)}  "
           f"(T {block_c_start}, C {block_p_start - block_c_start}, P {matrix.shape[1] - block_p_start})")
    report(f"    curated TFs {len(keep_tf)}, ChIP TFs {len(keep_chip)}, ligands {len(keep_lig)}, "
           f"PTM classes {len(keep_ptm)}")
    return matrix, out, block_c_start, block_p_start


def tuned_ridge(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, report, label: str):
    """Pick the ridge penalty on a validation slice of the TRAINING rows. Sealed data is never consulted."""

    n = len(x)
    order = rng.permutation(n)
    cut = int(0.8 * n)
    tr, va = order[:cut], order[cut:]
    best, best_err = None, float("inf")
    for penalty in PENALTIES:
        w = A.ridge_fit(x[tr], y[tr], penalty)
        err = float(np.sqrt(np.mean((A.ridge_apply(x[va], w) - y[va]) ** 2)))
        if err < best_err:
            best, best_err = penalty, err
    report(f"    {label:<10} {x.shape[1]:>5d} columns, penalty {best:>6.1f} (validation RMSE {best_err:.4f})")
    return A.ridge_fit(x, y, best), best


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 110)
    report("PROTEOMICS + TF NETWORK + EVERYTHING ELSE on the sealed knockout protocol")
    report("=" * 110)
    report("  PREDECLARED: block T (curated TF identity) should help; block C (ChIP identity) should NOT --")
    report("  it was measured at chance against this exact target; block P (proteomics/dependency) is open.")

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

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    report(f"\n  training pool {len(train):,}; factorising into {A.K} components ...")
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    del X

    universe = sorted(set(kos) | sealed)
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tier_b_start = C.extra_channels(universe, set(train), report)
    new, new_names, c_start, p_start = blocks(universe, set(train), report)
    urow = {g: i for i, g in enumerate(universe)}
    train_rows = np.array([urow[k] for k in train])

    chan2a = np.concatenate([base, extra[:, :tier_b_start]], axis=1)
    arms = {
        "chan3t": np.concatenate([chan2a, new[:, :c_start]], axis=1),
        "chan3c": np.concatenate([chan2a, new[:, :p_start]], axis=1),
        "chan3p": np.concatenate([chan2a, new], axis=1),
    }
    arms["chan3perm"] = arms["chan3p"][np.random.default_rng(A.SEED + 3).permutation(len(chan2a))]
    # chan2a itself refitted with a TUNED penalty, so the comparison is not penalty-confounded
    arms["chan2atuned"] = chan2a

    report(f"\n  fitting, penalty tuned per arm on a validation slice of the training knockouts")
    rng = np.random.default_rng(A.SEED + 11)
    fitted, chosen = {}, {}
    for arm, matrix in arms.items():
        fitted[arm], chosen[arm] = tuned_ridge(matrix[train_rows], W,
                                               np.random.default_rng(A.SEED + 11), report, arm)

    globalmix = W.mean(0)
    for tag in ("", "_holdout"):
        dp = OUT / f"ko_pred_dossiers{tag}.json"
        if not dp.exists():
            continue
        cohort = sorted(json.loads(dp.read_text()))
        rows = np.array([urow[k] for k in cohort])
        for arm, matrix in arms.items():
            mixes = A.ridge_apply(matrix[rows], fitted[arm])
            preds = {}
            for j, k in enumerate(cohort):
                mix = mixes[j] if np.isfinite(mixes[j]).all() else globalmix
                score = mix @ H
                score[tide] = 0.0
                if k in gi:
                    score[gi[k]] = 0.0
                order = np.argsort(-score)[:A.NPRED]
                preds[k] = {"variant": arm,
                            "reasoning": f"{matrix.shape[1]} named channels (ridge penalty {chosen[arm]}) -> "
                                         f"mixture of {A.K} response components -> ranked over "
                                         f"{len(genes):,} genes",
                            "predicted_genes": [genes[i] for i in order if score[i] > 0]}
            p = OUT / f"ko_predictions_{arm}{tag}.json"
            json.dump(preds, open(p, "w"), indent=1)
            report(f"  {tag or 'cohort1':<9} {arm:<12} -> {p.name}")

    json.dump({"test": "adrn_ko_channels3",
               "n_chan2a": int(chan2a.shape[1]),
               "n_block_T": int(c_start), "n_block_C": int(p_start - c_start),
               "n_block_P": int(new.shape[1] - p_start),
               "penalties": {a: float(p) for a, p in chosen.items()},
               "block_T_names": new_names[:c_start],
               "block_C_names": new_names[c_start:p_start],
               "block_P_names": new_names[p_start:], "log": log},
              open(OUT / "adrn_ko_channels3.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_ko_channels3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
