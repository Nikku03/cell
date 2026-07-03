"""NicheNet ligand -> target regulatory potential -> tissue-model downstream wiring.

NicheNet's prior model scores, for each ligand, how strongly it can influence each target gene
(integrating signaling + gene-regulatory networks). We take each ligand's top target genes and
emit ligand -> [targets] -> nichenet_ligand_targets.json (loaded by build_tissue_model to answer
'when this ligand hits a receiver cell, which genes does it ultimately move?').

Reads NicheNet's ligand_target_matrix.rds (genes x ligands). Needs pyreadr. Skips if absent.
-> nichenet_ligand_targets.json
"""
import json
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
f=H/"nichenet_ligand_target_matrix.rds"
if not f.exists():
    print("NicheNet .rds absent -> skipping (runs on Colab)"); raise SystemExit(0)
try:
    import pyreadr
except Exception as e:
    print("pyreadr not installed (pip install pyreadr) -> skipping NicheNet:",e); raise SystemExit(0)
TOPN=25
res=pyreadr.read_r(str(f))
df=next(iter(res.values()))            # rows = target genes (index), cols = ligands
# ensure gene names are the index
if df.index.name is None and df.shape[0] and not str(df.index[0]).isupper():
    df=df.set_index(df.columns[0])
lt={}
for lig in df.columns:
    s=df[lig].astype(float).sort_values(ascending=False)
    targets=[g for g in s.index[:TOPN] if float(s[g])>0]
    if targets: lt[str(lig)]=list(map(str,targets))
json.dump(lt,open(OUT/"nichenet_ligand_targets.json","w"))
print(f"NicheNet: top-{TOPN} targets for {len(lt)} ligands -> nichenet_ligand_targets.json")
