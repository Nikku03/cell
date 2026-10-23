"""RICHER NAMED CHANNELS: the lever that already worked, pushed as far as the annotation goes.

WHAT THE MEASUREMENT SAID TO DO. Going from 9 lesion channels to 170 named channels was worth +0.0557 on the
holdout cohort -- the single largest effect in this line of work. And adrn_ko_ceiling.py then showed the deployed
model covers only 14.7% of the twin-ceiling range and 19.2% of the basis-oracle range, so the vocabulary is NOT
the binding constraint and richer channels have room to pay. This file spends that room.

WHAT IS ADDED, all of it already sitting unused in cell_complete.json and the domain tables:

    GO terms                 16,412 genes annotated; the largest untouched source in the repo
    complex membership       gene2cplx, 3,257 genes across 2,039 complexes
    Pfam families            prot_pfam.json
    InterPro domains         domains.json
    compartment / process    the curated `comp` and `proc` fields on every gene record
    chromosome               `chrom`
    constraint               LOEUF buckets -- how intolerant the gene is to loss of function
    study depth              `pubs`, `ndis`, `conf`, `dark` -- how much is known, which is a real confounder
                             worth modelling explicitly rather than leaving implicit
    regulatory architecture  `cpg`, `enh`, `tf`, `npath`
    PTM / cell-cycle flags   ptm, cellcycle

TWO TIERS, KEPT SEPARATE ON PURPOSE.

    TIER A is pure annotation: what is known about the gene from curation and sequence, with no measurement of any
    perturbation anywhere.  This is the honest "predict from first principles" arm.

    TIER B adds MEASURED PRIORS FROM OTHER ASSAYS: DepMap viability (`ess`, `dep_frac`), protein abundance and
    expression level (`abund`, `ppm`), and FBA-predicted essentiality.  None of these is the answer to this
    question -- the target is the K562 transcriptional response, not viability -- but they are measurements, and
    conflating them with curation would let a reader think the model reasons from biology when it partly looks up
    an experiment.  They are reported as a separate arm so the pure-annotation claim stays clean either way.

    `coexpr` and `sig` are EXCLUDED from both tiers.  Co-expression is derived from expression covariance in the
    same kind of data the target comes from, so it is the one source here that could smuggle the answer in.

ARMS
    chan2a       170 base channels + TIER A
    chan2b       170 base channels + TIER A + TIER B
    chan2perm    chan2a with rows permuted across genes -- the leakage control that already caught nothing last
                 time and has to keep catching nothing

PREDECLARED: chan2a must beat the 170-channel adrnlin (0.2620 on cohort 2) on BOTH cohorts to count. If chan2b
beats chan2a by more than chan2a beats adrnlin, then the gain is coming from looked-up measurements rather than
from biology, and that has to be said in those words.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import numpy as np

import adrn_ko_conjunctions as A

OUT = A.OUT
SP = A.SP
TOP_GO = int(os.environ.get("CH2_TOP_GO", "250"))
TOP_CPLX = int(os.environ.get("CH2_TOP_CPLX", "120"))
TOP_DOMAIN = int(os.environ.get("CH2_TOP_DOMAIN", "120"))


def ranked(counter: collections.Counter, limit: int | None = None, floor: int = 0) -> list[str]:
    items = sorted(((k, v) for k, v in counter.items() if v >= floor), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in (items[:limit] if limit else items)]


def extra_channels(all_genes: list[str], train_set: set[str], report) -> tuple[np.ndarray, list[str], int]:
    """Tier A then Tier B columns, and the index at which Tier B starts."""

    idx = {g: i for i, g in enumerate(all_genes)}
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    rec = {g["name"]: g for g in D["genes"]}

    # `go` is keyed by gene NAME and each value is {'F': [...], 'P': [...], 'C': [...]} -- molecular function,
    # biological process, cellular component. Namespacing keeps a term that appears in two ontologies distinct.
    go_of: dict[str, list[str]] = {}
    for g, v in (D.get("go") or {}).items():
        key = names[int(g)] if g.isdigit() and int(g) < len(names) else g
        if isinstance(v, dict):
            go_of[key] = [f"{ns}:{t}" for ns in sorted(v) for t in (v[ns] or [])]
        elif isinstance(v, list):
            go_of[key] = [str(t) for t in v]
        else:
            go_of[key] = [str(v)]

    cplx_of: dict[str, list[str]] = {}
    for g, v in (D.get("gene2cplx") or {}).items():
        key = names[int(g)] if g.isdigit() and int(g) < len(names) else g
        cplx_of[key] = [str(x) for x in (v if isinstance(v, list) else [v])]

    ptm_genes = set()
    for g in (D.get("ptm") or {}):
        ptm_genes.add(names[int(g)] if g.isdigit() and int(g) < len(names) else g)
    cc_genes = set()
    for g in (D.get("cellcycle") or {}):
        cc_genes.add(names[int(g)] if g.isdigit() and int(g) < len(names) else g)

    abund: dict[str, float] = {}
    for src in ("abund", "ppm"):
        for g, v in (D.get(src) or {}).items():
            key = names[int(g)] if g.isdigit() and int(g) < len(names) else g
            try:
                abund.setdefault(key, float(v if not isinstance(v, (list, tuple)) else v[0]))
            except (TypeError, ValueError):
                pass
    del D

    pfam_of: dict[str, list[str]] = {}
    try:
        for g, v in json.loads((OUT / "prot_pfam.json").read_text()).items():
            if g.isdigit() and int(g) < len(names):
                pfam_of[names[int(g)]] = list(v)
    except Exception as exc:
        report(f"  [warn] pfam unavailable ({exc})")
    dom_of: dict[str, list[str]] = {}
    try:
        for g, v in json.loads((OUT / "domains.json").read_text())["domains"].items():
            if g.isdigit() and int(g) < len(names):
                dom_of[names[int(g)]] = list(v)
    except Exception as exc:
        report(f"  [warn] domains unavailable ({exc})")
    fba: dict[str, float] = {}
    try:
        fba = {k: float(v) for k, v in
               json.loads((OUT / "fba_essentiality.json").read_text())["predicted_growth_on_knockout"].items()}
    except Exception as exc:
        report(f"  [warn] fba essentiality unavailable ({exc})")

    # ---- select categorical levels by TRAINING frequency, deterministic tie-breaks ----
    def pick(mapping: dict[str, list[str]], limit: int) -> list[str]:
        freq: collections.Counter = collections.Counter()
        for g in sorted(train_set):
            for t in sorted(set(mapping.get(g, ()))):
                freq[t] += 1
        return ranked(freq, limit=limit)

    keep_go = pick(go_of, TOP_GO)
    keep_cplx = pick(cplx_of, TOP_CPLX)
    keep_pfam = pick(pfam_of, TOP_DOMAIN)
    keep_dom = pick(dom_of, TOP_DOMAIN)

    comp_freq: collections.Counter = collections.Counter()
    proc_freq: collections.Counter = collections.Counter()
    chrom_freq: collections.Counter = collections.Counter()
    for g in sorted(train_set):
        r = rec.get(g, {})
        for field, counter in (("comp", comp_freq), ("proc", proc_freq), ("chrom", chrom_freq)):
            v = (r.get(field) or "").strip()
            if v:
                counter[v] += 1
    keep_comp = ranked(comp_freq, floor=10)
    keep_proc = ranked(proc_freq, floor=10)
    keep_chrom = ranked(chrom_freq, floor=10)

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

    def numeric(g: str, field: str) -> float:
        try:
            return float((rec.get(g, {}).get(field) or 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    # ---------------- TIER A ----------------
    for t in keep_go:
        s = {g for g in idx if t in go_of.get(g, ())}
        add(f"go:{t}", s.__contains__)
    for c in keep_cplx:
        s = {g for g in idx if c in cplx_of.get(g, ())}
        add(f"complex:{c}", s.__contains__)
    for p in keep_pfam:
        s = {g for g in idx if p in pfam_of.get(g, ())}
        add(f"pfam:{p}", s.__contains__)
    for dm in keep_dom:
        s = {g for g in idx if dm in dom_of.get(g, ())}
        add(f"domain:{dm}", s.__contains__)
    for c in keep_comp:
        add(f"gene_comp:{c}", lambda g, c=c: (rec.get(g, {}).get("comp") or "").strip() == c)
    for p in keep_proc:
        add(f"gene_proc:{p}", lambda g, p=p: (rec.get(g, {}).get("proc") or "").strip() == p)
    for c in keep_chrom:
        add(f"chrom:{c}", lambda g, c=c: (rec.get(g, {}).get("chrom") or "").strip() == c)
    add("is_tf", lambda g: numeric(g, "tf") > 0)
    add("is_dark", lambda g: numeric(g, "dark") > 0)
    add("in_complex", lambda g: bool(cplx_of.get(g)))
    add("has_ptm", ptm_genes.__contains__)
    add("cellcycle_gene", cc_genes.__contains__)
    add("conf_high", lambda g: (rec.get(g, {}).get("conf") or "") == "high")
    for field, cuts in (("loeuf", (0.35, 0.6, 1.0)), ("pubs", (10, 100, 500)), ("ndis", (1, 3)),
                        ("npath", (5, 20, 60)), ("cpg", (1,)), ("enh", (20, 100, 400))):
        for cut in cuts:
            add(f"{field}>={cut}", lambda g, f=field, k=cut: numeric(g, f) >= k)
    for cut in (10, 50, 200):
        add(f"n_go>={cut}", lambda g, k=cut: len(go_of.get(g, ())) >= k)
    tier_b_start = len(columns)

    # ---------------- TIER B: measured priors from OTHER assays ----------------
    for cut in (0.25, 0.5, 0.75):
        add(f"[measured] dep_frac>={cut}", lambda g, k=cut: numeric(g, "dep_frac") >= k)
    add("[measured] ess", lambda g: numeric(g, "ess") > 0)
    for cut in (1.0, 10.0, 100.0):
        add(f"[measured] abundance>={cut}", lambda g, k=cut: abund.get(g, 0.0) >= k)
    for cut in (0.5, 0.9, 0.99):
        add(f"[measured] fba_growth<={cut}", lambda g, k=cut: fba.get(g, 1.0) <= k)

    matrix = np.stack([c for _, c in columns], axis=1)
    out_names = [n for n, _ in columns]
    live = matrix.sum(0)
    keep = (live >= 10) & (live <= len(all_genes) - 10)
    dropped_before_b = int((~keep[:tier_b_start]).sum())
    matrix, out_names = matrix[:, keep], [n for n, k in zip(out_names, keep) if k]
    tier_b_start -= dropped_before_b
    report(f"  extra channels: {matrix.shape[1]} kept of {len(columns)} built  "
           f"(TIER A {tier_b_start}, TIER B {matrix.shape[1] - tier_b_start}); "
           f"{len(keep_go)} GO, {len(keep_cplx)} complexes, {len(keep_pfam)} pfam, {len(keep_dom)} domains")
    return matrix, out_names, tier_b_start


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 104)
    report("RICHER NAMED CHANNELS on the sealed knockout protocol")
    report("=" * 104)
    report("  PREDECLARED: chan2a must beat adrnlin (0.2127 / 0.2620) on BOTH cohorts to count.")
    report("  If chan2b - chan2a exceeds chan2a - adrnlin, the gain is looked-up measurement, not biology.")

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
    base, base_names = A.build_channels(universe, set(train), report)
    extra, extra_names, tier_b_start = extra_channels(universe, set(train), report)
    urow = {g: i for i, g in enumerate(universe)}
    train_rows = np.array([urow[k] for k in train])

    tier_a = np.concatenate([base, extra[:, :tier_b_start]], axis=1)
    tier_ab = np.concatenate([base, extra], axis=1)
    perm = tier_a[np.random.default_rng(A.SEED + 3).permutation(len(tier_a))]
    report(f"\n  chan2a {tier_a.shape[1]} columns, chan2b {tier_ab.shape[1]} columns "
           f"(base {base.shape[1]})")

    arms = {"chan2a": tier_a, "chan2b": tier_ab, "chan2perm": perm}
    fitted = {a: A.ridge_fit(m[train_rows], W, A.PENALTY) for a, m in arms.items()}
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
                            "reasoning": f"{matrix.shape[1]} named channels -> ridge -> mixture of {A.K} "
                                         f"response components -> ranked over {len(genes):,} genes",
                            "predicted_genes": [genes[i] for i in order if score[i] > 0]}
            p = OUT / f"ko_predictions_{arm}{tag}.json"
            json.dump(preds, open(p, "w"), indent=1)
            report(f"  {tag or 'cohort1':<9} {arm:<10} {len(cohort)} knockouts -> {p.name}")

    json.dump({"test": "adrn_ko_channels2",
               "n_base": int(base.shape[1]), "n_tier_a": int(tier_a.shape[1]),
               "n_tier_b_total": int(tier_ab.shape[1]),
               "tier_a_names": extra_names[:tier_b_start], "tier_b_names": extra_names[tier_b_start:],
               "log": log}, open(OUT / "adrn_ko_channels2.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_ko_channels2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
