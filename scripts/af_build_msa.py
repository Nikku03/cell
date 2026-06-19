"""Build the OG-MSA feature tensors for the AlphaFold-style essentiality model.

For each labeled gene (focal) across all 48 organisms, assemble:
  focal[d_f]   intrinsic features only (NO cross-org label-derived term):
               gene_len(log), leading, oriC_frac, gc, n_paralogs(log),
               own dN/dS, has_dnds
  msa[K, d_m]  up to K ortholog rows (same OG, OTHER organisms), each:
               ortholog_label(0/1), label_known(0/1), dN/dS, dn, ds,
               same_clade(0/1)   -- label_known is masked per-fold at train time
  msa_org[K]   organism index of each ortholog row (for per-fold label masking)
  meta         organism, clade, locus_tag, y(label)

Leakage control (AlphaFold-style): conservation is NOT a feature. The model must
learn it from the MSA label rows, which are masked whenever the row's organism is
in the held-out clade. So the same tensors serve any leave-one-clade-out split.

Output: outputs/orphan/af_msa_cache.npz
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
K = 24   # max MSA depth

# labels
labels = {}
for r in csv.DictReader(open(DATA/"labels"/"labels.csv")):
    labels[(r["organism"], r["locus_tag"])] = int(r["essential"])

# clades
clade = {}
for r in csv.DictReader(open(DATA/"labels"/"clade_splits.csv")):
    clade[r["organism"]] = r.get("clade","?")

# orthology: og -> members; gene -> og + intrinsic features
gene_og = {}; og_members = defaultdict(list); gfeat = {}
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    o, lt, og = r["organism"], r["locus_tag"], r["og_id"]
    if not og: continue
    gene_og[(o,lt)] = og
    og_members[og].append((o,lt))
    npar = float(r["n_paralogs_in_genome"]) if r["n_paralogs_in_genome"] not in ("",) else 0.0
    gfeat[(o,lt)] = dict(n_paralogs=npar)

# dN/dS per gene (where computed)
dnds = {}
for p in OUT.glob("dnds_beril_*.parquet"):
    import pandas as pd
    d = pd.read_parquet(p)
    org = p.stem.replace("dnds_","")
    for lt, dn, ds, om in zip(d.locus_tag, d.dn, d.ds, d.dnds):
        dnds[(org,lt)] = (dn, ds, om)

# genome geometry features (gene_len, leading, oriC_frac, gc, mobile) -- reuse the
# cross_org_coverage_scores for the 6 well-characterized bacteria; others get NaN->0
geo = {}
import pandas as pd
cov = pd.read_csv(OUT/"cross_org_coverage_scores.csv")
for r in cov.itertuples():
    geo[(r.organism, r.locus_tag)] = dict(
        gene_len=r.gene_len_aa, leading=r.leading, oriC_frac=r.oriC_frac,
        gc=r.gc, mobile=r.mobile_dist_kb)

orgs = sorted({o for (o,_) in labels})
org_idx = {o:i for i,o in enumerate(orgs)}
print(f"{len(orgs)} organisms, {len(labels)} labeled genes")

def focal_vec(o, lt):
    g = geo.get((o,lt), {})
    gl = g.get("gene_len", np.nan)
    dn, ds, om = dnds.get((o,lt), (np.nan,np.nan,np.nan))
    npar = gfeat.get((o,lt),{}).get("n_paralogs", 0.0)
    return [
        np.log1p(gl) if gl==gl else 0.0,
        g.get("leading", 0.5) if g.get("leading",np.nan)==g.get("leading",np.nan) else 0.5,
        g.get("oriC_frac", 0.5) if g.get("oriC_frac",np.nan)==g.get("oriC_frac",np.nan) else 0.5,
        g.get("gc", 0.6) if g.get("gc",np.nan)==g.get("gc",np.nan) else 0.6,
        np.log1p(npar),
        om if om==om else 0.0,
        1.0 if om==om else 0.0,
    ]
D_F = 7
D_M = 6

foc=[]; msa=[]; msa_org=[]; msa_mask=[]; ys=[]; meta_org=[]; meta_clade=[]; meta_lt=[]
rng = np.random.default_rng(0)
for (o, lt), y in labels.items():
    og = gene_og.get((o,lt))
    if og is None: continue
    foc.append(focal_vec(o,lt)); ys.append(y)
    meta_org.append(org_idx[o]); meta_clade.append(clade.get(o,"?")); meta_lt.append(lt)
    rows=[]; rorg=[]
    for (oo, ol) in og_members[og]:
        if oo == o: continue
        yl = labels.get((oo,ol))
        dn, ds, om = dnds.get((oo,ol),(np.nan,np.nan,np.nan))
        rows.append([
            float(yl) if yl is not None else 0.0,     # ortholog label (masked per-fold)
            1.0 if yl is not None else 0.0,            # label_known
            om if om==om else 0.0,
            dn if dn==dn else 0.0,
            ds if ds==ds else 0.0,
            1.0 if clade.get(oo,"?")==clade.get(o,"?") else 0.0,  # same clade
        ])
        rorg.append(org_idx[oo])
    if len(rows) > K:
        sel = rng.choice(len(rows), K, replace=False)
        rows = [rows[i] for i in sel]; rorg = [rorg[i] for i in sel]
    m = len(rows)
    while len(rows) < K: rows.append([0.0]*D_M); rorg.append(-1)
    msa.append(rows); msa_org.append(rorg)
    msa_mask.append([1.0]*m + [0.0]*(K-m))

foc=np.array(foc,np.float32); msa=np.array(msa,np.float32)
msa_org=np.array(msa_org,np.int32); msa_mask=np.array(msa_mask,np.float32)
ys=np.array(ys,np.float32)
# normalize focal continuous cols (0,4,5) and msa continuous (2,3,4)
def znorm(a, cols):
    for c in cols:
        v=a[...,c]; mu=v[v!=0].mean() if (v!=0).any() else 0; sd=v[v!=0].std()+1e-6
        a[...,c]=(v-mu)/sd
    return a
foc=znorm(foc,[0,4,5]); msa=znorm(msa,[2,3,4])
clades=np.array(meta_clade)
print(f"tensors: foc {foc.shape}, msa {msa.shape}, mask depth median {np.median(msa_mask.sum(1)):.0f}")
print(f"essential rate {ys.mean():.3f}")
np.savez_compressed(OUT/"af_msa_cache.npz",
    foc=foc, msa=msa, msa_org=msa_org, msa_mask=msa_mask, y=ys,
    meta_org=np.array(meta_org,np.int32), clade=clades, lt=np.array(meta_lt),
    orgs=np.array(orgs))
print(f"wrote af_msa_cache.npz  ({(OUT/'af_msa_cache.npz').stat().st_size/1e6:.0f} MB)")
