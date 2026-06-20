"""Parameterised cache builder -- a faithful port of scripts/af_build_msa.py that
reads its three driver CSVs from an arbitrary directory, so we can rebuild
af_msa_cache.npz after augmenting the labels with DEG organisms.

Driver CSVs (in --labels_dir):
  labels.csv               organism, locus_tag, essential
  clade_splits.csv         organism, clade
  orthology_features.csv   organism, locus_tag, og_id, n_paralogs_in_genome

Optional (looked up in --out_dir, used only where present -- i.e. the original
beril organisms): dnds_*.parquet, cross_org_coverage_scores.csv. DEG organisms
have no geometry/dN-dS and fall back to the same neutral defaults af_build_msa
already uses for unlabelled-geometry genes.

Usage:
  python colab/build_cache.py --labels_dir data/drive_import/labels_aug \\
                              --out outputs/orphan/af_msa_cache_aug.npz
"""
import csv, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

K = 24            # max MSA depth (must match the model)
D_F, D_M = 7, 6


def build(labels_dir, out_npz, geom_dir):
    labels_dir = Path(labels_dir); geom_dir = Path(geom_dir)
    out_npz = Path(out_npz)

    labels = {}
    for r in csv.DictReader(open(labels_dir / "labels.csv")):
        labels[(r["organism"], r["locus_tag"])] = int(r["essential"])
    clade = {}
    for r in csv.DictReader(open(labels_dir / "clade_splits.csv")):
        clade[r["organism"]] = r.get("clade", "?")
    gene_og = {}; og_members = defaultdict(list); gfeat = {}
    for r in csv.DictReader(open(labels_dir / "orthology_features.csv")):
        o, lt, og = r["organism"], r["locus_tag"], r["og_id"]
        if not og:
            continue
        gene_og[(o, lt)] = og
        og_members[og].append((o, lt))
        npar = r.get("n_paralogs_in_genome", "") or "0"
        try:
            gfeat[(o, lt)] = dict(n_paralogs=float(npar))
        except ValueError:
            gfeat[(o, lt)] = dict(n_paralogs=0.0)

    # dN/dS (only beril orgs have it)
    dnds = {}
    try:
        import pandas as pd
        for p in geom_dir.glob("dnds_*.parquet"):
            d = pd.read_parquet(p); org = p.stem.replace("dnds_", "")
            for lt, dn, ds, om in zip(d.locus_tag, d.dn, d.ds, d.dnds):
                dnds[(org, lt)] = (dn, ds, om)
    except Exception as e:
        print(f"  (no dN/dS parquets used: {e})")

    # genome geometry (only the 6 well-characterised orgs)
    geo = {}
    cov_path = geom_dir / "cross_org_coverage_scores.csv"
    if cov_path.exists():
        import pandas as pd
        cov = pd.read_csv(cov_path)
        for r in cov.itertuples():
            geo[(r.organism, r.locus_tag)] = dict(
                gene_len=r.gene_len_aa, leading=r.leading, oriC_frac=r.oriC_frac,
                gc=r.gc, mobile=r.mobile_dist_kb)

    orgs = sorted({o for (o, _) in labels})
    org_idx = {o: i for i, o in enumerate(orgs)}
    print(f"  {len(orgs)} organisms, {len(labels)} labelled genes, {len(og_members)} OGs")

    def focal_vec(o, lt):
        g = geo.get((o, lt), {})
        gl = g.get("gene_len", np.nan)
        dn, ds, om = dnds.get((o, lt), (np.nan, np.nan, np.nan))
        npar = gfeat.get((o, lt), {}).get("n_paralogs", 0.0)
        f = lambda v, d: (v if v == v else d)
        return [
            np.log1p(gl) if gl == gl else 0.0,
            f(g.get("leading", np.nan), 0.5),
            f(g.get("oriC_frac", np.nan), 0.5),
            f(g.get("gc", np.nan), 0.6),
            np.log1p(npar),
            om if om == om else 0.0,
            1.0 if om == om else 0.0,
        ]

    foc = []; msa = []; msa_org = []; msa_mask = []; ys = []
    meta_org = []; meta_clade = []; meta_lt = []
    rng = np.random.default_rng(0)
    for (o, lt), y in labels.items():
        og = gene_og.get((o, lt))
        if og is None:
            continue
        foc.append(focal_vec(o, lt)); ys.append(y)
        meta_org.append(org_idx[o]); meta_clade.append(clade.get(o, "?")); meta_lt.append(lt)
        rows = []; rorg = []
        for (oo, ol) in og_members[og]:
            if oo == o:
                continue
            yl = labels.get((oo, ol))
            dn, ds, om = dnds.get((oo, ol), (np.nan, np.nan, np.nan))
            rows.append([
                float(yl) if yl is not None else 0.0,
                1.0 if yl is not None else 0.0,
                om if om == om else 0.0,
                dn if dn == dn else 0.0,
                ds if ds == ds else 0.0,
                1.0 if clade.get(oo, "?") == clade.get(o, "?") else 0.0,
            ])
            rorg.append(org_idx[oo])
        if len(rows) > K:
            sel = rng.choice(len(rows), K, replace=False)
            rows = [rows[i] for i in sel]; rorg = [rorg[i] for i in sel]
        m = len(rows)
        while len(rows) < K:
            rows.append([0.0] * D_M); rorg.append(-1)
        msa.append(rows); msa_org.append(rorg)
        msa_mask.append([1.0] * m + [0.0] * (K - m))

    foc = np.array(foc, np.float32); msa = np.array(msa, np.float32)
    msa_org = np.array(msa_org, np.int32); msa_mask = np.array(msa_mask, np.float32)
    ys = np.array(ys, np.float32)

    def znorm(a, cols):
        for c in cols:
            v = a[..., c]; nz = v != 0
            mu = v[nz].mean() if nz.any() else 0.0; sd = v[nz].std() + 1e-6
            a[..., c] = (v - mu) / sd
        return a
    foc = znorm(foc, [0, 4, 5]); msa = znorm(msa, [2, 3, 4])
    clades = np.array(meta_clade)
    print(f"  tensors: foc {foc.shape}, msa {msa.shape}, "
          f"median MSA depth {np.median(msa_mask.sum(1)):.0f}, ess rate {ys.mean():.3f}")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, foc=foc, msa=msa, msa_org=msa_org, msa_mask=msa_mask,
                        y=ys, meta_org=np.array(meta_org, np.int32), clade=clades,
                        lt=np.array(meta_lt), orgs=np.array(orgs))
    print(f"  wrote {out_npz}  ({out_npz.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="data/drive_import/labels")
    ap.add_argument("--out", default="outputs/orphan/af_msa_cache_aug.npz")
    ap.add_argument("--geom_dir", default="outputs/orphan",
                    help="where dnds_*.parquet + cross_org_coverage_scores.csv live")
    a = ap.parse_args()
    build(a.labels_dir, a.out, a.geom_dir)
