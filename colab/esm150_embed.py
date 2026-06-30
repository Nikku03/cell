"""Embed E. coli + M. tuberculosis (MycoTube) proteomes with the bigger ESM-2 150M,
for the cross-organism essentiality transfer test (8M vs 150M, where sequence matters)."""
import csv, time, sqlite3
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
OUT=Path("outputs/orphan"); REG=Path("data/external_data/regulon")
torch.set_num_threads(8)
CODON={'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def tr(nt): s=''.join(CODON.get(nt[i:i+3],'X') for i in range(0,len(nt)-2,3)); return s[:-1] if s.endswith('*') else s
def rc(s): t={'A':'T','T':'A','G':'C','C':'G','N':'N'}; return ''.join(t.get(c,'N') for c in reversed(s))
print("loading ESM-2 150M ...",flush=True)
MODEL="facebook/esm2_t30_150M_UR50D"
tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModel.from_pretrained(MODEL); model.eval(); DIM=model.config.hidden_size
@torch.no_grad()
def embed(ss):
    x=tok(ss,return_tensors="pt",padding=True,truncation=True,max_length=402)
    out=model(**x).last_hidden_state; mm=x["attention_mask"].unsqueeze(-1).float()
    return ((out*mm).sum(1)/mm.sum(1).clamp(min=1)).numpy().astype(np.float32)
def run(bs,prot,tag):
    E=np.zeros((len(prot),DIM),np.float32); B=16; t0=time.time()
    for i in range(0,len(prot),B):
        E[i:i+B]=embed(prot[i:i+B])
        if (i//B)%20==0: print(f"  {tag} {i}/{len(prot)} ({time.time()-t0:.0f}s)",flush=True)
    np.savez_compressed(OUT/f"{tag}_esm150.npz", b=np.array(bs), E=E)
    print(f"saved {tag}_esm150.npz {E.shape} ({time.time()-t0:.0f}s)",flush=True)
# E. coli from genome
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]; G="".join(seq).upper()
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
ec_b=[]; ec_p=[]; seen=set()
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        b=name2b.get(p[1].lower())
        if not b or b in seen: continue
        seen.add(b); s,e,st=int(p[3]),int(p[4]),p[5]; nt=G[s-1:e]; nt=rc(nt) if st=="reverse" else nt
        ec_b.append(b); ec_p.append((tr(nt) or "M")[:400])
print(f"E. coli proteins: {len(ec_p)}",flush=True); run(ec_b,ec_p,"ecoli")
# mtub from feba ScaffoldSeq
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
scaf={r[0]:r[1] for r in cur.execute("SELECT scaffoldId,sequence FROM ScaffoldSeq WHERE orgId='MycoTube'")}
mt_b=[]; mt_p=[]
for lid,sc,b,e,st in cur.execute("SELECT locusId,scaffoldId,begin,end,strand FROM Gene WHERE orgId='MycoTube' AND type=1"):
    s=scaf.get(sc,""); nt=s[b-1:e]; nt=rc(nt) if st=='-' else nt
    mt_b.append(lid); mt_p.append((tr(nt) or "M")[:400])
print(f"mtub proteins: {len(mt_p)}",flush=True); run(mt_b,mt_p,"mtub")
print("DONE")
