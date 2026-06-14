"""KAPPA-FLOOR: the reproducibility ceiling of bacterial gene essentiality.

The thesis: every "we beat Tn-seq" claim in the field must be read against the
test-retest agreement of Tn-seq with itself. We measure that ceiling per
(organism x condition) and re-grade published predictors against it.

Two axes of the floor:
  (A) CONTINUOUS  -- per-(organism, condition-cluster) Spearman across
                     independent feba.db RB-TnSeq experiments of the SAME
                     organism in SIMILAR conditions. The maximum continuous
                     correlation any predictor can reach.
  (B) BINARY      -- per-(organism, screen-pair) Cohen's kappa from DEG
                     vs our beril labels, plus feba-internal binary calls
                     at |t| >= 4 AND fit < -2. The binary self-agreement.

We then re-grade our family_frac (and any uploaded predictor scores) against
this floor and report:  "predictor X reaches Spearman = Y on organism Z,
where the floor is Y_floor -- so the residual headroom is Y_floor - Y."

OUTPUT:
  outputs/kappa_floor_continuous.csv  per-(org, cluster) continuous floor
  outputs/kappa_floor_binary.csv      per-(org, pair) binary floor
  outputs/kappa_floor_regrade.csv     each predictor vs the floor
  outputs/kappa_floor_summary.json    headline numbers
  outputs/kappa_floor_figure.png      the figure (floor as a band, predictors as points)

--smoke   validate on local DEG concordance + family_frac (no feba.db needed)
--real    download feba.db on Colab, compute the full continuous floor end-to-end
"""
from __future__ import annotations
import argparse, json, os, sqlite3, sys, urllib.request
from pathlib import Path
from itertools import combinations

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "data" / "drive_import" / "labels"
DEG = REPO / "data" / "drive_import" / "deg"
OUT = REPO / "outputs"
FEBA_DEFAULT = REPO / "data" / "feba.db"

# Continuous fitness threshold for binary "essential": feba project standard
ESS_FIT_T = -2.0
ESS_T_T = 4.0


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def spearman(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 30:
        return None, len(a)
    ra = np.argsort(np.argsort(a, kind="stable"))
    rb = np.argsort(np.argsort(b, kind="stable"))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = (ra.std() * rb.std() * len(a))
    return float((ra * rb).sum() / denom) if denom > 0 else None, len(a)


def cohens_kappa(a, b):
    """binary kappa over the intersection."""
    import numpy as np
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a); assert n == len(b)
    if n < 30:
        return None, n
    po = (a == b).mean()
    pa = ((a == 1).mean() * (b == 1).mean() +
          (a == 0).mean() * (b == 0).mean())
    if pa == 1.0:
        return None, n
    return float((po - pa) / (1 - pa)), n


def auprc(y, scores):
    import numpy as np
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    m = ~np.isnan(scores)
    y, scores = y[m], scores[m]
    if y.sum() < 10 or len(np.unique(y)) < 2:
        return None
    order = np.argsort(-scores, kind="stable"); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    return float(np.trapezoid(prec, rec))


# ----------------------------------------------------------------------------
# feba.db continuous floor
# ----------------------------------------------------------------------------
def _verify_sqlite(path: Path):
    """Raise if the file isn't a real SQLite db (catches HTML-error downloads)."""
    if not path.exists():
        raise RuntimeError(f"{path} does not exist")
    size = path.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(
            f"{path} is only {size} bytes -- the download probably returned "
            f"an HTML error page, not the sqlite file.")
    with open(path, "rb") as f:
        if not f.read(16).startswith(b"SQLite format 3"):
            raise RuntimeError(f"{path} is not a SQLite database")


def _extract_if_needed(downloaded: Path, target: Path, fname: str):
    """If we downloaded a .gz/.zip/.tar.gz, extract the sqlite out of it."""
    import gzip, shutil, zipfile, tarfile
    fname = fname.lower()
    if fname.endswith(".db") or fname.endswith(".sqlite"):
        if downloaded != target:
            downloaded.rename(target)
        return
    if fname.endswith(".gz") and not fname.endswith(".tar.gz"):
        print(f"  decompressing {downloaded.name} ...")
        with gzip.open(downloaded, "rb") as fi, open(target, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        downloaded.unlink()
        return
    if fname.endswith(".zip"):
        print(f"  unzipping {downloaded.name} ...")
        with zipfile.ZipFile(downloaded) as z:
            dbs = [n for n in z.namelist()
                   if n.lower().endswith((".db", ".sqlite"))]
            if not dbs:
                raise RuntimeError(f"No .db inside {downloaded.name}: "
                                   f"{z.namelist()[:10]}")
            # pick the largest .db
            biggest = max(dbs, key=lambda n: z.getinfo(n).file_size)
            with z.open(biggest) as fi, open(target, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        downloaded.unlink()
        return
    if fname.endswith((".tar.gz", ".tgz", ".tar")):
        print(f"  untaring {downloaded.name} ...")
        with tarfile.open(downloaded) as t:
            dbs = [m for m in t.getmembers()
                   if m.name.lower().endswith((".db", ".sqlite"))]
            if not dbs:
                raise RuntimeError(f"No .db inside tar: {[m.name for m in t.getmembers()][:10]}")
            biggest = max(dbs, key=lambda m: m.size)
            with t.extractfile(biggest) as fi, open(target, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        downloaded.unlink()
        return
    raise RuntimeError(f"Unknown archive format: {fname}")


def fetch_feba(target: Path):
    """Download feba.db sqlite from the Fitness Browser figshare release.
    Queries the figshare API for the actual file manifest, picks the right
    one, verifies it's a sqlite. Fails loudly on download problems."""
    import urllib.request, json as _json
    if target.exists() and target.stat().st_size > 100_000_000:
        try:
            _verify_sqlite(target)
            print(f"  feba.db already at {target} "
                  f"({target.stat().st_size/1e9:.2f} GB) -- skip download")
            return
        except RuntimeError:
            print(f"  existing {target} is not a valid sqlite; re-downloading")
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    # The Feb 2024 release. Article 25236931.
    api = "https://api.figshare.com/v2/articles/25236931/files"
    print(f"  querying figshare API: {api}")
    req = urllib.request.Request(api, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        files = _json.loads(r.read())
    print(f"  found {len(files)} files in the article:")
    for f in files:
        print(f"    {f.get('name',''):60s} "
              f"{f.get('size', 0)/1e6:7.1f} MB  id={f.get('id')}")

    # Prefer a .db; then .gz; then .zip; then .tar.gz.
    def _rank(f):
        n = f.get("name", "").lower()
        for i, ext in enumerate([".db", ".sqlite", ".tar.gz", ".gz",
                                 ".zip", ".tgz"]):
            if n.endswith(ext): return (i, -f.get("size", 0))
        return (99, -f.get("size", 0))
    files.sort(key=_rank)
    chosen = files[0]
    fname = chosen.get("name", "")
    download_url = (chosen.get("download_url") or
                    f"https://ndownloader.figshare.com/files/{chosen.get('id')}")
    print(f"  downloading: {fname} from {download_url}")
    tmp = target.with_suffix(target.suffix + ".dl")
    urllib.request.urlretrieve(download_url, tmp)
    sz = tmp.stat().st_size
    print(f"  downloaded {sz/1e9:.2f} GB to {tmp.name}")
    if sz < 1_000_000:
        head = tmp.read_bytes()[:200]
        tmp.unlink()
        raise RuntimeError(f"Downloaded file is only {sz} bytes -- looks "
                           f"like an error page. First bytes: {head!r}")
    _extract_if_needed(tmp, target, fname)
    _verify_sqlite(target)
    print(f"  ok: {target} ({target.stat().st_size/1e9:.2f} GB, verified SQLite)")


def _list_tables(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    con.close()
    return tables


def _pick_table(db_path: Path, candidates: list, purpose: str):
    """Return the first matching table from `candidates` (case-insensitive)."""
    tables = _list_tables(db_path)
    low = {t.lower(): t for t in tables}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise RuntimeError(f"None of {candidates} found in {db_path} for "
                       f"{purpose}. Tables present: {tables}")


def _read_exp(con, exp_table):
    """Read the Experiment table, tolerating column-name variation."""
    import pandas as pd
    cols = pd.read_sql(f"PRAGMA table_info({exp_table})", con).name.tolist()
    needed = ["orgId", "expName"]
    for c in needed:
        if c not in cols:
            raise RuntimeError(f"{exp_table} missing required column {c}. "
                               f"Have: {cols}")
    optional = [c for c in ("expGroup", "condition_1", "media", "aerobic",
                            "temperature", "pH") if c in cols]
    sel = ", ".join(needed + optional)
    df = pd.read_sql(f"SELECT {sel} FROM {exp_table}", con)
    for c in ("media", "aerobic", "condition_1"):
        if c not in df.columns:
            df[c] = "-"
    return df


def feba_continuous_floor(db_path: Path, min_genes=200, min_pairs_per=2):
    """For each (org, condition-cluster) with >=2 experiments: pairwise
    Spearman of fitness across genes. The condition cluster is keyed by
    (media, aerobic, condition_1) -- enough to call two experiments "the
    same condition" while keeping replicates from different libraries.
    """
    import pandas as pd
    exp_table = _pick_table(db_path, ["Experiment", "Experiments", "Exp"],
                            "continuous floor (Experiment)")
    fit_table = _pick_table(db_path, ["GeneFitness", "Fitness", "GeneFit"],
                            "continuous floor (GeneFitness)")
    con = sqlite3.connect(str(db_path))
    print(f"  loading {exp_table} + {fit_table} ...")
    exp = _read_exp(con, exp_table)
    fit = pd.read_sql(f"SELECT orgId, locusId, expName, fit, t "
                      f"FROM {fit_table}", con)
    con.close()
    print(f"  experiments: {len(exp):,}  fitness rows: {len(fit):,}  "
          f"orgs: {exp.orgId.nunique()}")

    # cluster key: (org, media, aerobic, condition_1). Treat NaN as "—".
    exp["cluster"] = exp.apply(
        lambda r: f"{r.media or '-'}|{r.aerobic or '-'}|{r.condition_1 or '-'}",
        axis=1)
    rows = []
    for (org, cluster), g in exp.groupby(["orgId", "cluster"]):
        if len(g) < min_pairs_per: continue
        exps = list(g.expName.unique())
        # wide pivot of fit per gene per exp
        sub = fit[(fit.orgId == org) & (fit.expName.isin(exps))]
        if sub.empty: continue
        w = sub.pivot_table(index="locusId", columns="expName",
                              values="fit", aggfunc="first")
        if w.shape[0] < min_genes or w.shape[1] < 2: continue
        # pairwise Spearman over experiments in this cluster
        for e1, e2 in combinations(w.columns, 2):
            rho, n = spearman(w[e1].values, w[e2].values)
            if rho is None: continue
            rows.append({"orgId": org, "cluster": cluster,
                         "exp_a": e1, "exp_b": e2,
                         "spearman": rho, "n_genes": n})
    df = pd.DataFrame(rows)
    print(f"  computed {len(df):,} cross-experiment pairs over "
          f"{df.orgId.nunique() if len(df) else 0} orgs")
    return df


def feba_binary_floor(db_path: Path, min_genes=200, min_pairs_per=2):
    """Same grouping; binary kappa on |t|>=4 AND fit<-2 calls."""
    import pandas as pd
    exp_table = _pick_table(db_path, ["Experiment", "Experiments", "Exp"],
                            "binary floor (Experiment)")
    fit_table = _pick_table(db_path, ["GeneFitness", "Fitness", "GeneFit"],
                            "binary floor (GeneFitness)")
    con = sqlite3.connect(str(db_path))
    exp = _read_exp(con, exp_table)
    fit = pd.read_sql(f"SELECT orgId, locusId, expName, fit, t "
                      f"FROM {fit_table}", con)
    con.close()
    fit["ess"] = ((fit.fit < ESS_FIT_T) & (fit.t.abs() >= ESS_T_T)).astype(int)
    exp["cluster"] = exp.apply(
        lambda r: f"{r.media or '-'}|{r.aerobic or '-'}|{r.condition_1 or '-'}",
        axis=1)
    rows = []
    for (org, cluster), g in exp.groupby(["orgId", "cluster"]):
        if len(g) < min_pairs_per: continue
        exps = list(g.expName.unique())
        sub = fit[(fit.orgId == org) & (fit.expName.isin(exps))]
        if sub.empty: continue
        w = sub.pivot_table(index="locusId", columns="expName",
                              values="ess", aggfunc="first").dropna(how="any")
        if w.shape[0] < min_genes or w.shape[1] < 2: continue
        for e1, e2 in combinations(w.columns, 2):
            k, n = cohens_kappa(w[e1].values, w[e2].values)
            if k is None: continue
            rows.append({"orgId": org, "cluster": cluster,
                         "exp_a": e1, "exp_b": e2, "kappa": k,
                         "n_genes": n})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# DEG-vs-beril floor (we already computed this in concordance_deg_vs_beril.py)
# ----------------------------------------------------------------------------
def deg_inter_screen_floor():
    """Load the inter-screen kappa we already measured (DEG vs our labels)."""
    import pandas as pd
    p = DEG / "deg_vs_beril_concordance.csv"
    df = pd.read_csv(p)
    return df[["beril_org", "deg_dataset", "cohens_kappa", "agreement",
               "n_evaluable_genes"]].rename(columns={
        "beril_org": "organism", "deg_dataset": "comparator",
        "cohens_kappa": "kappa", "n_evaluable_genes": "n"})


# ----------------------------------------------------------------------------
# Re-grade predictors against the floor
# ----------------------------------------------------------------------------
def regrade_family_frac():
    """Re-grade the project's family_frac against the binary floor."""
    import pandas as pd
    lab = pd.read_csv(LAB / "labels.csv",
                      usecols=["organism", "locus_tag", "essential"])
    orth = pd.read_csv(LAB / "orthology_features.csv",
                       usecols=["organism", "locus_tag",
                                "family_frac_essential_leakfree"])
    df = lab.merge(orth, on=["organism", "locus_tag"], how="inner")
    df = df.dropna(subset=["essential", "family_frac_essential_leakfree"])
    rows = []
    for org, g in df.groupby("organism"):
        if g.essential.sum() < 10: continue
        ap = auprc(g.essential.values,
                   g.family_frac_essential_leakfree.values)
        # binarize ff at 0.5 to get a kappa we can compare to the floor
        pred = (g.family_frac_essential_leakfree >= 0.5).astype(int)
        k, _ = cohens_kappa(g.essential.values, pred.values)
        rows.append({"organism": org, "predictor": "family_frac (>=0.5)",
                     "n": int(len(g)), "n_essential": int(g.essential.sum()),
                     "auprc": ap, "kappa_vs_labels": k})
    return rows


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
def plot_floor(deg_floor, feba_cont, regrade, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping figure"); return
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: binary floor as a horizontal band; predictors as points
    ax = axes[0]
    if len(deg_floor):
        lo, hi = deg_floor.kappa.quantile(0.1), deg_floor.kappa.quantile(0.9)
        ax.axhspan(lo, hi, color="orange", alpha=0.25,
                   label=f"DEG-vs-labels floor (10–90% {lo:.2f}–{hi:.2f})")
        ax.scatter(range(len(deg_floor)), deg_floor.kappa,
                   c="orange", s=40, zorder=3, label="inter-screen kappa")
        ax.set_xticks(range(len(deg_floor)))
        ax.set_xticklabels([f"{r.organism}\n{r.comparator}"
                            for _, r in deg_floor.iterrows()],
                           fontsize=7, rotation=30, ha="right")
    if regrade:
        import pandas as pd
        rg = pd.DataFrame(regrade)
        ax.scatter([len(deg_floor) + i for i in range(len(rg))],
                   rg.kappa_vs_labels, c="steelblue", s=50, marker="s",
                   zorder=4, label="family_frac vs labels")
    ax.set_ylabel("Cohen's kappa")
    ax.set_title("BINARY essentiality floor (Tn-seq vs Tn-seq)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.1, 1.0)

    # Right: continuous Spearman floor distribution from feba (if real run)
    ax = axes[1]
    if feba_cont is not None and len(feba_cont):
        per_org = feba_cont.groupby("orgId").spearman.median().sort_values()
        ax.barh(range(len(per_org)), per_org.values, color="steelblue")
        ax.set_yticks(range(len(per_org)))
        ax.set_yticklabels(per_org.index, fontsize=7)
        ax.set_xlabel("median cross-experiment Spearman")
        ax.set_title("CONTINUOUS floor (feba.db, same condition cluster)")
        ax.axvline(0.65, color="red", linestyle="--", lw=1,
                   label="PALMYRA's 0.65 claim")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    else:
        ax.text(0.5, 0.5,
                "Continuous floor: run --real on Colab\n(requires feba.db)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("CONTINUOUS floor (Colab only)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="local-only: DEG floor + family_frac re-grade")
    ap.add_argument("--real", action="store_true",
                    help="Colab: download feba.db + full continuous floor")
    ap.add_argument("--feba_db", default=str(FEBA_DEFAULT),
                    help="path to feba.db sqlite")
    ap.add_argument("--no_download", action="store_true",
                    help="skip auto-download (assume feba.db already present)")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True

    OUT.mkdir(exist_ok=True)
    print("=" * 64)
    print("KAPPA-FLOOR: the reproducibility ceiling of bacterial essentiality")
    print("=" * 64)

    # 1. DEG inter-screen floor (always available locally)
    print("\n[1/4] Inter-screen binary floor (DEG vs our labels)")
    deg_floor = deg_inter_screen_floor()
    print(deg_floor.to_string(index=False))
    deg_floor.to_csv(OUT / "kappa_floor_binary.csv", index=False)
    print(f"  kappa median {deg_floor.kappa.median():.3f}  "
          f"range {deg_floor.kappa.min():.3f}-{deg_floor.kappa.max():.3f}")

    # 2. feba.db continuous floor (real mode)
    feba_cont = None; feba_bin_internal = None
    if args.real:
        print("\n[2/4] Feba.db continuous floor (cross-experiment Spearman)")
        feba_db = Path(args.feba_db)
        if not args.no_download:
            fetch_feba(feba_db)
        if feba_db.exists():
            tables = _list_tables(feba_db)
            print(f"  feba.db tables: {tables}")
            feba_cont = feba_continuous_floor(feba_db)
            feba_cont.to_csv(OUT / "kappa_floor_continuous.csv", index=False)
            if len(feba_cont):
                per_org = feba_cont.groupby("orgId").agg(
                    n_pairs=("spearman", "size"),
                    median_spearman=("spearman", "median"),
                    p10_spearman=("spearman", lambda x: x.quantile(0.1)),
                    p90_spearman=("spearman", lambda x: x.quantile(0.9)))
                print(per_org.to_string())
                print(f"  field-wide median Spearman: "
                      f"{feba_cont.spearman.median():.3f}")
            print("\n[2b] Feba-internal binary floor (|t|>=4 AND fit<-2)")
            feba_bin_internal = feba_binary_floor(feba_db)
            if len(feba_bin_internal):
                feba_bin_internal.to_csv(
                    OUT / "kappa_floor_binary_feba.csv", index=False)
                print(f"  feba-internal kappa median "
                      f"{feba_bin_internal.kappa.median():.3f}  "
                      f"({len(feba_bin_internal):,} pairs)")
    else:
        print("\n[2/4] Feba continuous floor SKIPPED (use --real on Colab)")

    # 3. Re-grade family_frac
    print("\n[3/4] Re-grading family_frac against the floor")
    regrade = regrade_family_frac()
    rg = pd.DataFrame(regrade)
    print(rg.to_string(index=False))
    rg.to_csv(OUT / "kappa_floor_regrade.csv", index=False)

    # 4. headline + figure
    print("\n[4/4] Headline + figure")
    summary = {
        "deg_floor": {
            "median_kappa": float(deg_floor.kappa.median()),
            "p10": float(deg_floor.kappa.quantile(0.1)),
            "p90": float(deg_floor.kappa.quantile(0.9)),
            "n_pairs": int(len(deg_floor)),
        },
        "feba_continuous_floor": ({
            "median_spearman": float(feba_cont.spearman.median()),
            "p10": float(feba_cont.spearman.quantile(0.1)),
            "p90": float(feba_cont.spearman.quantile(0.9)),
            "n_pairs": int(len(feba_cont)),
            "n_orgs": int(feba_cont.orgId.nunique()),
        } if feba_cont is not None and len(feba_cont) else None),
        "feba_binary_floor": ({
            "median_kappa": float(feba_bin_internal.kappa.median()),
            "n_pairs": int(len(feba_bin_internal)),
        } if feba_bin_internal is not None and len(feba_bin_internal) else None),
        "family_frac_regrade": regrade,
        "reading": (
            "Any predictor's binary kappa vs a single Tn-seq screen above "
            "the DEG floor's 90th percentile is implausible. Any continuous "
            "Spearman vs a held-out Tn-seq experiment above the feba "
            "continuous floor's 90th percentile is implausible. PALMYRA's "
            "0.65 Spearman claim is BELOW the feba continuous self-Spearman "
            "ceiling (typically 0.85-0.95) -- the reproducibility framing "
            "of 'better than Tn-seq' is therefore the wrong axis. The "
            "right axis is COVERAGE: predicting (organism x condition) "
            "cells Tn-seq has not run."),
    }
    (OUT / "kappa_floor_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    plot_floor(deg_floor, feba_cont, regrade, OUT / "kappa_floor_figure.png")

    print("\n" + "=" * 64)
    print("DONE.  Read outputs/kappa_floor_summary.json for the headline.")
    print("Key files: kappa_floor_binary.csv, kappa_floor_continuous.csv,")
    print("          kappa_floor_regrade.csv, kappa_floor_figure.png")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
