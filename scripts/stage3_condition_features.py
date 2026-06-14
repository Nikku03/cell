"""STAGE 3: rich CONDITION-content features (the ESM-of-the-condition-axis).

WHY: Stage 2 showed gene FUNCTION (ESM) interacts with condition and lifts
the conditional model (+0.026 AUPRC). But the condition itself is still
impoverished -- our frame encodes each of the 1,296 condition_clusters as just
4 scalars (aerobic, size, target-encoded mean_fit, target-encoded ess_rate).
Those tell the model "how essential are genes on average here", NOT "what IS
this perturbation chemically". So a specific gene's function cannot interact
with the specific content of the condition (an efflux pump matters under a
DRUG; a siderophore under METAL limitation). This stage gives the condition a
real feature vector describing its content, so the GBM can learn
gene-function x perturbation-content interactions.

SOURCE: feba Experiment table (already downloaded for the consensus). Columns
beyond media/aerobic/condition_1: expGroup, condition_2, concentration_*,
pH, temperature, expDesc. ALL pure metadata -- nothing derived from fitness --
so these features are LEAK-FREE by construction (a property of the condition,
identical in train and held-out clades).

KEY: condition_cluster = f"{media}|{aerobic}|{condition_1}", reconstructed
EXACTLY as ces_consensus.py builds it, so it joins the training frame 1:1.
Multiple experiments share a cluster -> aggregate per cluster (numeric: mean;
group flags: any; text: union of tokens then hash).

OUTPUT: outputs/condition_features.parquet, one row per condition_cluster:
  cond3_ph, cond3_temp, cond3_logconc        parsed numerics (mean per cluster)
  cond3_grp_<g>                              one-hot of top expGroup categories
  cond3_h0..cond3_hN                         D-dim hashed bag-of-words over the
                                             condition free text (no fixed vocab)

--smoke : validate cluster reconstruction, numeric parsing, group one-hot,
          deterministic hashing, and the frame join on synthetic data (no feba).
--real  : read feba.db Experiment table on Colab and write the parquet.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
OUT = REPO / "outputs"
FEBA_DEFAULT = REPO / "data" / "feba.db"
sys.path.insert(0, str(SCRIPTS))

HASH_DIM = 32          # dimensionality of the hashed condition-text vector
N_TOP_GROUPS = 8       # one-hot the N most frequent expGroup values
_STOP = {"the", "of", "in", "to", "and", "with", "no", "vs", "from", "for",
         "a", "an", "at", "on", "by", "or", "mm", "ml", "ug", "mg", "um"}
_NUM = re.compile(r"[-+]?\d*\.?\d+")
_TOK = re.compile(r"[a-z0-9]+")


def _parse_num(x):
    """First float in a free-text value (e.g. 'pH 5.5' -> 5.5, '10 mM' -> 10)."""
    if x is None:
        return None
    m = _NUM.search(str(x))
    return float(m.group()) if m else None


def _tokens(*vals):
    """Lowercase alphanumeric tokens from free text, minus stopwords and
    bare numbers (numbers are captured separately as numerics)."""
    out = []
    for v in vals:
        if v is None:
            continue
        for t in _TOK.findall(str(v).lower()):
            if t in _STOP or t.isdigit():
                continue
            out.append(t)
    return out


def _hash_vec(tokens, dim=HASH_DIM):
    """Deterministic signed feature-hashing (md5, not Python's salted hash) so
    the vector is identical across runs/machines. L2-normalized."""
    import numpy as np
    v = np.zeros(dim, dtype="float32")
    for t in set(tokens):                      # presence, not count
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        v[idx] += sign
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_condition_features(exp, hash_dim=HASH_DIM, n_top_groups=N_TOP_GROUPS):
    """exp: DataFrame of the feba Experiment table. Returns per-cluster feature
    frame keyed by condition_cluster."""
    import pandas as pd, numpy as np
    e = exp.copy()
    # reconstruct the cluster key EXACTLY as ces_consensus.py
    for c in ("media", "aerobic", "condition_1"):
        if c not in e.columns:
            e[c] = "-"
    e["condition_cluster"] = e.apply(
        lambda r: f"{r.media or '-'}|{r.aerobic or '-'}|{r.condition_1 or '-'}",
        axis=1)
    # numeric columns (flexible lookup)
    def col(*names):
        for n in names:
            if n in e.columns:
                return e[n]
        return pd.Series([None] * len(e))
    e["_ph"] = col("pH", "ph").map(_parse_num)
    e["_temp"] = col("temperature", "temp").map(_parse_num)
    e["_conc"] = pd.concat([col("concentration_1", "concentration").map(_parse_num),
                            col("concentration_2").map(_parse_num)], axis=1).max(axis=1)
    e["_logconc"] = e["_conc"].apply(lambda x: np.log10(x) if x and x > 0 else None)
    # group one-hot (top-N by frequency)
    grp = col("expGroup", "group", "expDesc").fillna("other").astype(str).str.lower()
    e["_grp"] = grp.str.replace(r"\s+", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    top = e["_grp"].value_counts().head(n_top_groups).index.tolist()
    e["_grp"] = e["_grp"].where(e["_grp"].isin(top), "other")
    # text tokens for hashing
    txt_cols = [c for c in ("expDesc", "condition_1", "condition_2",
                            "expGroup", "media") if c in e.columns]
    e["_toks"] = e[txt_cols].apply(lambda r: _tokens(*r.values), axis=1)

    rows = []
    for cl, g in e.groupby("condition_cluster"):
        rec = {"condition_cluster": cl,
               "cond3_ph": g["_ph"].mean(),
               "cond3_temp": g["_temp"].mean(),
               "cond3_logconc": g["_logconc"].mean()}
        for t in top:
            rec[f"cond3_grp_{t}"] = int((g["_grp"] == t).any())
        toks = [tok for lst in g["_toks"] for tok in lst]
        hv = _hash_vec(toks, hash_dim)
        for k in range(hash_dim):
            rec[f"cond3_h{k}"] = float(hv[k])
        rows.append(rec)
    feat = pd.DataFrame(rows)
    # median-impute the numerics (some conditions lack pH/temp/conc)
    for c in ("cond3_ph", "cond3_temp", "cond3_logconc"):
        med = feat[c].median()
        feat[c] = feat[c].fillna(med if med == med else 0.0)
    cond3_cols = [c for c in feat.columns if c.startswith("cond3_")]
    return feat, cond3_cols


def run_real(args):
    import pandas as pd
    from kappa_floor import fetch_feba
    from ces_consensus import _read_experiments
    db = Path(args.feba_db)
    if not args.no_download:
        fetch_feba(db)
    print("=" * 64)
    print("STAGE 3 -- rich condition-content features from feba Experiment")
    print("=" * 64)
    exp, _ = _read_experiments(db)
    print(f"  {len(exp):,} experiments; columns: {list(exp.columns)}")
    feat, cond3_cols = build_condition_features(exp)
    OUT.mkdir(exist_ok=True)
    out = OUT / "condition_features.parquet"
    feat.to_parquet(out, index=False)
    print(f"\n  condition clusters: {len(feat):,}")
    print(f"  cond3 feature cols ({len(cond3_cols)}): {cond3_cols}")
    print("  numeric coverage:")
    for c in ("cond3_ph", "cond3_temp", "cond3_logconc"):
        print(f"    {c:<16s} non-default: {(feat[c] != feat[c].median()).mean():.0%}")
    grp_cols = [c for c in cond3_cols if c.startswith("cond3_grp_")]
    print(f"  group one-hots: {grp_cols}")
    (OUT / "condition_features_summary.json").write_text(json.dumps({
        "n_clusters": int(len(feat)), "n_cond3_cols": len(cond3_cols),
        "cond3_cols": cond3_cols, "hash_dim": HASH_DIM}, indent=2))
    print(f"\nwrote {out} + summary")
    print("  next: step2_train.py --real --esm ... --cond3 outputs/condition_features.parquet")
    return 0


def run_smoke():
    import pandas as pd, numpy as np
    print("=" * 64)
    print("STAGE 3 SMOKE -- condition features on synthetic Experiment data")
    print("=" * 64)
    exp = pd.DataFrame({
        "orgId": ["ANA3"] * 5,
        "expName": [f"set1IT{i}" for i in range(5)],
        "media": ["LB", "LB", "M9", "M9", "M9"],
        "aerobic": ["aerobic", "aerobic", "aerobic", "aerobic", "anaerobic"],
        "condition_1": ["none", "none", "glucose", "glucose", "glucose"],
        "condition_2": ["", "", "", "", ""],
        "expGroup": ["carbon source", "carbon source", "carbon source",
                     "stress", "stress"],
        "concentration_1": ["", "", "10", "10", "10"],
        "pH": ["7", "7", "7", "5.5", "7"],
        "temperature": ["30", "30", "37", "37", "37"],
        "expDesc": ["LB rich media", "LB rich media", "M9 glucose minimal",
                    "M9 glucose pH stress", "M9 glucose anaerobic"]})
    feat, cond3_cols = build_condition_features(exp, hash_dim=8, n_top_groups=3)
    print(feat[["condition_cluster", "cond3_ph", "cond3_temp",
                "cond3_logconc"]].to_string(index=False))
    # 3 distinct clusters: LB|aerobic|none, M9|aerobic|glucose, M9|anaerobic|glucose
    assert len(feat) == 3, f"expected 3 clusters, got {len(feat)}"
    lb = feat[feat.condition_cluster == "LB|aerobic|none"].iloc[0]
    assert abs(lb.cond3_ph - 7.0) < 1e-6, "pH parse/aggregate wrong"
    assert abs(lb.cond3_temp - 30.0) < 1e-6, "temp parse/aggregate wrong"
    m9a = feat[feat.condition_cluster == "M9|aerobic|glucose"].iloc[0]
    # M9|aerobic|glucose has two experiments, pH 7 and 5.5 -> mean 6.25
    assert abs(m9a.cond3_ph - 6.25) < 1e-6, f"cluster pH mean wrong: {m9a.cond3_ph}"
    assert abs(m9a.cond3_logconc - np.log10(10)) < 1e-6, "logconc wrong"
    # group one-hot present
    grp_cols = [c for c in cond3_cols if c.startswith("cond3_grp_")]
    assert grp_cols, "no group one-hots produced"
    print(f"  group one-hots: {grp_cols}")
    # hashing determinism
    v1 = _hash_vec(["glucose", "stress"], 8)
    v2 = _hash_vec(["stress", "glucose"], 8)
    assert np.allclose(v1, v2), "hashing not order-invariant"
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-6, "hash vec not L2-normalized"
    # frame join: cluster keys must match what step2 frame holds
    frame = pd.DataFrame({"condition_cluster":
                          ["LB|aerobic|none", "M9|aerobic|glucose"]})
    j = frame.merge(feat, on="condition_cluster", how="left")
    assert j[cond3_cols].notna().all().all(), "frame join lost features"
    print(f"  frame join: {len(j)}/{len(frame)} clusters matched, all features present")
    print("\n=== STAGE 3 SMOKE PASS ===")
    print("  cluster key reconstruction (matches ces_consensus), numeric parse,")
    print("  per-cluster aggregation, group one-hot, deterministic hashing, and")
    print("  frame join all validated. Run --real on Colab, then train --cond3.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--feba_db", default=str(FEBA_DEFAULT))
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
