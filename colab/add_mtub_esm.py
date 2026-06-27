"""Embed mtub (MycoTube) proteins with ESM-2, save aligned to its gene list.
Also save mtub fitness-derived essentiality features for completeness."""
import sqlite3, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
INP=Path("/home/user/cell/data/external_data/transformer_inputs")
torch.set_num_threads(4)
CODON={'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def translate(nt):
    s=''.join(CODON.get(nt[i:i+3],'X') for i in range(0,len(nt)-2,3))
    return s[:-1] if s.endswith('*') else s
def rc(s):
    t={'A':'T','T':'A','G':'C','C':'G','N':'N'}; return ''.join(t.get(c,'N') for c in reversed(s))
print("loading ESM-2 8M...",flush=True)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"); model=AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D"); model.eval()
DIM=model.config.hidden_size
@torch.no_grad()
def embed(seqs):
    x=tok(seqs,return_tensors="pt",padding=True,truncation=True,max_length=402)
    out=model(**x).last_hidden_state; m=x["attention_mask"].unsqueeze(-1).float()
    return ((out*m).sum(1)/m.sum(1).clamp(min=1)).numpy().astype(np.float32)
fb="MycoTube"; t0=time.time()
scaf={r[0]:r[1] for r in cur.execute("SELECT scaffoldId,sequence FROM ScaffoldSeq WHERE orgId=?",(fb,))}
genes=list(cur.execute("SELECT locusId,scaffoldId,begin,end,strand FROM Gene WHERE orgId=? AND type=1",(fb,)))
g_keys=[g[0] for g in genes]; seqs=[]
for lid,sc,b,e,st in genes:
    s=scaf.get(sc,""); nt=s[b-1:e]
    if st=='-': nt=rc(nt)
    seqs.append((translate(nt) or "M")[:400])
E=np.zeros((len(seqs),DIM),np.float32); B=64
for i in range(0,len(seqs),B):
    E[i:i+B]=embed(seqs[i:i+B])
    if (i//B)%15==0: print(f"  {i}/{len(seqs)} ({time.time()-t0:.0f}s)",flush=True)
# fitness-derived essential proxy: absent from GeneFitness OR min fit < -3
seen=set(); minfit={}
for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId=?",(fb,)):
    if fit is None: continue
    seen.add(lid);
    if lid not in minfit or fit<minfit[lid]: minfit[lid]=fit
np.savez_compressed(INP/"MycoTube_esm.npz", G_esm=E, g_keys=np.array(g_keys),
                    seen=np.array([1 if k in seen else 0 for k in g_keys]),
                    minfit=np.array([minfit.get(k,0.0) for k in g_keys],dtype=np.float32))
print(f"saved MycoTube_esm.npz {E.shape} ({time.time()-t0:.0f}s)")
