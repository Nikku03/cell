"""Literal before/after structure: ESMFold the wild-type and the MUTANT sequence, superimpose
(Kabsch), report RMSD (global + local around the mutation) and pLDDT change. Tests whether a
point mutation actually changes the predicted fold."""
import re, time, warnings, io, gc, json
import numpy as np, torch
warnings.filterwarnings("ignore")
from pathlib import Path
SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad"); OUT=Path("outputs/orphan")
# sequences from human.faa
seqs={}; sym=None; buf=[]
for l in open(SC/"human.faa"):
    if l.startswith(">"):
        if sym and buf: seqs[sym]=max(seqs.get(sym,""),"".join(buf),key=len)
        buf=[]; gn=[t[3:] for t in l.split() if t.startswith("GN=")]; sym=gn[0] if gn else None
    else: buf.append(l.strip())
if sym and buf: seqs[sym]=max(seqs.get(sym,""),"".join(buf),key=len)
# mutations by motif (verified), pos = index in sequence (0-based) of the changed residue
def mutate(seq, motif, newmotif):
    i=seq.find(motif)
    assert i>=0, f"motif {motif} not found"
    j=[k for k in range(len(motif)) if motif[k]!=newmotif[k]][0]
    return seq[:i]+newmotif+seq[i+len(motif):], i+j
CASES=[
 ("HBB","E6V (sickle, surface)","LTPEEK","LTPVEK","DISEASE-surface"),
 ("SOD1","A4V (ALS, buried)","ATKAVCV","ATKVVCV","DISEASE-buried"),
]
print("loading ESMFold ...",flush=True); t0=time.time()
from transformers import AutoTokenizer, EsmForProteinFolding
tok=AutoTokenizer.from_pretrained("facebook/esmfold_v1")
mdl=EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",low_cpu_mem_usage=True).eval(); torch.set_num_threads(8)
mdl.trunk.set_chunk_size(32)  # lower peak memory
print(f"loaded ({time.time()-t0:.0f}s)",flush=True)
@torch.no_grad()
def fold(seq):
    pdb=mdl.infer_pdb(seq)
    ca=[]; plddt=[]
    _lines=pdb.splitlines()
    for ln in _lines:
        if ln.startswith("ATOM") and ln[12:16].strip()=="CA":
            ca.append([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])]); plddt.append(float(ln[60:66]))
    del pdb,_lines; gc.collect()
    return np.array(ca), np.array(plddt)
def kabsch_rmsd(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0)
    V,S,Wt=np.linalg.svd(Pc.T@Qc); d=np.sign(np.linalg.det(Wt.T@V.T))
    D=np.diag([1,1,d]); R=Wt.T@D@V.T
    Pr=Pc@R.T
    return float(np.sqrt(((Pr-Qc)**2).sum(1).mean())), np.sqrt(((Pr-Qc)**2).sum(1))
res=[]
for gene,desc,motif,newmotif,kind in CASES:
    seq=seqs.get(gene)
    if not seq: print(f"{gene}: no seq"); continue
    mut,pos=mutate(seq,motif,newmotif)
    print(f"\n{gene} {desc}: len {len(seq)}, mutating index {pos} {seq[pos]}->{mut[pos]}",flush=True)
    t1=time.time(); wt_ca,wt_p=fold(seq); print(f"  WT folded ({time.time()-t1:.0f}s)",flush=True)
    t1=time.time(); mu_ca,mu_p=fold(mut); print(f"  mutant folded ({time.time()-t1:.0f}s)",flush=True)
    n=min(len(wt_ca),len(mu_ca)); 
    g_rmsd,perres=kabsch_rmsd(wt_ca[:n],mu_ca[:n])
    lo,hi=max(0,pos-6),min(n,pos+7); local=float(perres[lo:hi].mean())
    dpl=float(mu_p[pos]-wt_p[pos]) if pos<len(mu_p) else None
    res.append(dict(gene=gene,mutation=desc,kind=kind,length=n,
        global_rmsd=round(g_rmsd,3),local_rmsd_around_site=round(local,3),
        wt_plddt_site=round(float(wt_p[pos]),1),mut_plddt_site=round(float(mu_p[pos]),1),dplddt=round(dpl,1)))
    print(f"  => global CA-RMSD {g_rmsd:.3f} A | local (±6) {local:.3f} A | pLDDT@site {wt_p[pos]:.0f}->{mu_p[pos]:.0f}",flush=True)
    json.dump(res,open(OUT/"fold_before_after.json","w"),indent=2)  # incremental
    del wt_ca,mu_ca,wt_p,mu_p; gc.collect()
print("\n=== SUMMARY ===")
for r in res: print(f"  {r['gene']:5s} {r['mutation']:24s} global {r['global_rmsd']:.2f}A  local {r['local_rmsd_around_site']:.2f}A  dpLDDT {r['dplddt']:+.0f}")
print("wrote fold_before_after.json")
