"""Compute dN/dS for the 5 bacteria that lack a dnds parquet (GMI1000 already has one).

Reuses orphan_dnds.py's Nei-Gojobori + CDS extraction, but takes the gene list
from labels.csv + orthology_features.csv instead of requiring a bridge parquet.
Writes outputs/orphan/dnds_<org>.parquet with the same schema.
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from orphan_dnds import extract_cds, nei_gojobori, tr, _identity, _manifest_acc, DATA, OUT

TARGETS = ["beril_RalstoniaPSI07","beril_RalstoniaBSBF1503","beril_Dda3937",
           "beril_HerbieS","beril_Magneto"]

# gene -> og per org, and og -> members across orgs
gene_og = {}; og_members = {}
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    if r["og_id"]:
        gene_og[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_members.setdefault(r["og_id"], []).append((r["organism"], r["locus_tag"]))

acc = _manifest_acc()
fna_orgs = [o for o,a in acc.items() if (DATA/"genome_cache"/a/"genome.fna").exists()]

# extract CDS once for all fna orgs (cache across targets)
print("extracting CDS for all source organisms ...")
cds = {}
for o in fna_orgs:
    if o in acc:
        for lt, seq in extract_cds(acc[o]).items():
            cds[(o, lt)] = seq
print(f"  {len(cds):,} CDS across {len(fna_orgs)} orgs")

for org in TARGETS:
    out_path = OUT / f"dnds_{org}.parquet"
    if out_path.exists():
        print(f"{org}: cached, skipping"); continue
    src = set(fna_orgs) - {org}
    genes = [lt for (o,lt) in gene_og if o == org]
    rows = []; n_done = 0
    for lt in genes:
        og = gene_og.get((org, lt))
        c_self = cds.get((org, lt))
        res = {"locus_tag": lt, "dnds": np.nan, "dn": np.nan, "ds": np.nan,
               "ortholog_org":"", "ortholog_locus":"", "pct_identity": np.nan,
               "prot_len": np.nan, "n_same_len_orthologs": 0}
        if c_self and og in og_members and len(c_self) % 3 == 0:
            p_self = "".join(tr(c_self[i:i+3]) for i in range(0,len(c_self),3)).rstrip("*")
            best = None; n_same = 0
            for (oo, ol) in og_members[og]:
                if oo == org or oo not in src: continue
                c_o = cds.get((oo, ol))
                if not c_o or len(c_o) != len(c_self) or len(c_o) % 3: continue
                p_o = "".join(tr(c_o[i:i+3]) for i in range(0,len(c_o),3)).rstrip("*")
                if len(p_o) != len(p_self): continue
                n_same += 1
                idn = _identity(p_self, p_o)
                if best is None or idn > best[0]: best = (idn, oo, ol, c_o)
            res["n_same_len_orthologs"] = n_same
            if best is not None and best[0] < 1.0:
                ng = nei_gojobori(c_self, best[3])
                res.update(dnds=ng["omega"], dn=ng["dN"], ds=ng["dS"],
                           ortholog_org=best[1], ortholog_locus=best[2],
                           pct_identity=round(best[0],4), prot_len=len(p_self))
                n_done += 1
        rows.append(res)
    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"{org}: {n_done}/{len(genes)} genes with dN/dS -> {out_path.name}")
print("done")
