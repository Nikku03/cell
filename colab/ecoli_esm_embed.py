"""Embed the whole E. coli proteome with ESM-2 -> b-number -> 320-dim vector.
For the learned essentiality model (ESM + conservation + FBA + fitness + network)."""
import csv, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
torch.set_num_threads(4)
CODON={'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def tr(nt):
    s=''.join(CODON.get(nt[i:i+3],'X') for i in range(0,len(nt)-2,3)); return s[:-1] if s.endswith('*') else s
def rc(s):
    t={'A':'T','T':'A','G':'C','C':'G','N':'N'}; return ''.join(t.get(c,'N') for c in reversed(s))
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]
G="".join(seq).upper()
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
genes=[]
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        b=name2b.get(p[1].lower())
        if b: genes.append((b,int(p[3]),int(p[4]),p[5]))
# dedup by b
seen=set(); G2=[]
for b,s,e,st in genes:
    if b in seen: continue
    seen.add(b); G2.append((b,s,e,st))
genes=G2
prot=[]
for b,s,e,st in genes:
    nt=G[s-1:e]; nt=rc(nt) if st=="reverse" else nt
    prot.append((tr(nt) or "M")[:400])
print(f"E. coli proteins to embed: {len(prot)}",flush=True)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"); model=AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D"); model.eval()
DIM=model.config.hidden_size
@torch.no_grad()
def embed(ss):
    x=tok(ss,return_tensors="pt",padding=True,truncation=True,max_length=402)
    out=model(**x).last_hidden_state; mmask=x["attention_mask"].unsqueeze(-1).float()
    return ((out*mmask).sum(1)/mmask.sum(1).clamp(min=1)).numpy().astype(np.float32)
E=np.zeros((len(prot),DIM),np.float32); B=64; t0=time.time()
for i in range(0,len(prot),B):
    E[i:i+B]=embed(prot[i:i+B])
    if (i//B)%10==0: print(f"  {i}/{len(prot)} ({time.time()-t0:.0f}s)",flush=True)
np.savez_compressed(OUT/"ecoli_esm.npz", b=np.array([g[0] for g in genes]), E=E)
print(f"saved ecoli_esm.npz {E.shape} ({time.time()-t0:.0f}s)")
