"""STAGE 3a: precompute ESM-2 embeddings for pilot genes, aligned to the
existing npz g_keys ordering (so gi indices stay valid).

ESM-2 t6 8M (320-dim), mean-pooled over residues, proteins truncated to 400 aa.
Saves <org>_esm.npz with G_esm (n_genes, 320) in g_keys order.
"""
import sqlite3, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
INP=Path("/home/user/cell/data/external_data/transformer_inputs")
torch.set_num_threads(4)
PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]

CODON={'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def translate(nt):
    p=[CODON.get(nt[i:i+3],'X') for i in range(0,len(nt)-2,3)]
    s=''.join(p)
    return s[:-1] if s.endswith('*') else s
def revcomp(s):
    t={'A':'T','T':'A','G':'C','C':'G','N':'N'}; return ''.join(t.get(c,'N') for c in reversed(s))

print("loading ESM-2 8M...", flush=True)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model=AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D"); model.eval()
DIM=model.config.hidden_size

@torch.no_grad()
def embed_batch(seqs):
    x=tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=402)
    out=model(**x).last_hidden_state  # B,L,320
    mask=x["attention_mask"].unsqueeze(-1).float()
    pooled=(out*mask).sum(1)/mask.sum(1).clamp(min=1)
    return pooled.numpy().astype(np.float32)

for fb in PILOT:
    t0=time.time()
    d=np.load(INP/f"{fb}.npz", allow_pickle=True)
    g_keys=[str(k) for k in d["g_keys"]]
    # build locus -> protein
    scaf={r[0]:r[1] for r in cur.execute("SELECT scaffoldId,sequence FROM ScaffoldSeq WHERE orgId=?",(fb,))}
    coords={r[0]:(r[1],r[2],r[3]) for r in cur.execute("SELECT locusId,scaffoldId,begin,end FROM Gene WHERE orgId=? AND type=1",(fb,))}
    strand={r[0]:r[1] for r in cur.execute("SELECT locusId,strand FROM Gene WHERE orgId=? AND type=1",(fb,))}
    seqs=[]
    for lid in g_keys:
        c=coords.get(lid)
        if not c: seqs.append("M"); continue
        sc,b,e=c; s=scaf.get(sc,"")
        nt=s[b-1:e]
        if strand.get(lid)=='-': nt=revcomp(nt)
        prot=translate(nt) or "M"
        seqs.append(prot[:400])
    # embed in batches
    E=np.zeros((len(seqs),DIM),np.float32); B=64
    for i in range(0,len(seqs),B):
        E[i:i+B]=embed_batch(seqs[i:i+B])
        if (i//B)%20==0: print(f"  {fb} {i}/{len(seqs)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(INP/f"{fb}_esm.npz", G_esm=E, g_keys=np.array(g_keys))
    print(f"  {fb}: ESM embeddings {E.shape} saved ({time.time()-t0:.0f}s)", flush=True)
print("done")
