"""STAGE 0: tensorize fitness data + build gene & condition feature encoders.

What we build (per-org, saved to disk for Stage 1):
  - gene_features[org_id, locus] : protein-derived feature vector
       AAC (20) + dipeptide-bigram (400) + length + GC + #domains + #pfam_top20 (20)
       = 442 dims  (replaces ESM as a baseline encoder)
  - condition_features[org_id, expName] : compound bag vector
       multi-hot over the K most-common feba compounds + aerobic flag + condition_1 tokens
  - fitness triples (org, locus, expName, fit, t)

We DO NOT load ESM here -- the proxy truncates the checkpoint. We use a smart
classical encoder that captures sequence composition + domain content. If
Stage 1 gates pass with this, we know the framing is right; if not, ESM cannot
save it.

Pilot subset for speed: 8 organisms (mix of densely-measured Putida/Keio/Koxy/
Caulo/Btheta/MR1/PV4/Dda3937) -- ~3.5M fitness rows, fits in memory.
"""
import sqlite3, csv, json, time, re, os
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
OUT=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT.mkdir(exist_ok=True)

PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]
print(f"Pilot orgs: {PILOT}")

# 1. extract protein sequences from genome scaffolds
CODON={
 'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
 'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
 'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
 'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
 'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
 'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
 'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
 'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def translate(nt):
    p=[]
    for i in range(0,len(nt)-2,3):
        c=nt[i:i+3]
        if 'N' in c: p.append('X')
        else: p.append(CODON.get(c,'X'))
    s=''.join(p)
    if s.endswith('*'): s=s[:-1]
    return s
def revcomp(s):
    t={'A':'T','T':'A','G':'C','C':'G','N':'N'}
    return ''.join(t.get(c,'N') for c in reversed(s))

AA="ACDEFGHIKLMNPQRSTVWY"
AA2I={a:i for i,a in enumerate(AA)}
DI=[a+b for a in AA for b in AA]
DI2I={d:i for i,d in enumerate(DI)}

# Pfam: pick top 20 domains across pilot orgs
print("collecting top Pfam domains across pilot...", flush=True); t0=time.time()
pfam=Counter()
for fb in PILOT:
    for r in cur.execute("SELECT domainId FROM GeneDomain WHERE orgId=? AND domainDb='PFam'",(fb,)):
        pfam[r[0]]+=1
TOP_PFAM=[d for d,_ in pfam.most_common(20)]
pfam2i={d:i for i,d in enumerate(TOP_PFAM)}
print(f"  top Pfam: {TOP_PFAM[:5]}... ({time.time()-t0:.0f}s)")

# 2. per-org: extract proteins, compute features
def featurize(seq, n_domains, top_pfam_hits):
    L=len(seq) or 1
    aac=np.zeros(20); di=np.zeros(400)
    for i,a in enumerate(seq):
        if a in AA2I: aac[AA2I[a]]+=1
        if i<L-1:
            d=seq[i:i+2]
            if d in DI2I: di[DI2I[d]]+=1
    aac/=L; di/=max(L-1,1)
    gc=0  # filled later
    feat=np.concatenate([aac, di, [np.log1p(L), gc, np.log1p(n_domains)], top_pfam_hits])
    return feat.astype(np.float32)

print("\nbuilding gene features per org...", flush=True)
all_gene_feats={}; gene_meta={}  # (fb,locus) -> meta
for fb in PILOT:
    t0=time.time()
    # load scaffolds
    scaf={r[0]:r[1] for r in cur.execute("SELECT scaffoldId,sequence FROM ScaffoldSeq WHERE orgId=?",(fb,))}
    # genes
    genes=list(cur.execute("SELECT locusId,scaffoldId,begin,end,strand FROM Gene WHERE orgId=? AND type=1",(fb,)))
    # domains per locus
    dom_n=Counter(); pfam_hits=defaultdict(lambda: np.zeros(20))
    for r in cur.execute("SELECT locusId,domainId FROM GeneDomain WHERE orgId=?",(fb,)):
        dom_n[r[0]]+=1
        if r[1] in pfam2i: pfam_hits[r[0]][pfam2i[r[1]]]=1.0
    feats={}; n_ok=0
    for lid,sc,b,e,strand in genes:
        s=scaf.get(sc)
        if not s: continue
        nt=s[b-1:e]
        if strand=='-': nt=revcomp(nt)
        prot=translate(nt)
        if len(prot)<10: continue
        f=featurize(prot, dom_n.get(lid,0), pfam_hits[lid])
        # add GC from gene region
        g=nt.count('G')+nt.count('C')
        f[421]=g/max(len(nt),1)
        feats[lid]=f; n_ok+=1
    all_gene_feats[fb]=feats
    print(f"  {fb:<10} {n_ok}/{len(genes)} proteins encoded ({time.time()-t0:.0f}s)")

# 3. condition features (compound multi-hot)
print("\nbuilding condition features (media compound bags)...", flush=True)
# collect compounds across all media in the pilot
media_comps=defaultdict(set)   # media -> set of compounds
for r in cur.execute("SELECT media,compound FROM MediaComponents"):
    media_comps[r[0]].add(r[1])
all_comps=Counter()
for s in media_comps.values():
    for c in s: all_comps[c]+=1
TOP_COMP=[c for c,_ in all_comps.most_common(200)]
comp2i={c:i for i,c in enumerate(TOP_COMP)}
print(f"  total media: {len(media_comps)}  total compounds: {len(all_comps)}  top200 cover: {sum(all_comps[c] for c in TOP_COMP)}/{sum(all_comps.values())}")

cond_feats={}
for fb in PILOT:
    cf={}
    for expName,media,aerobic,cond1 in cur.execute("SELECT expName,media,aerobic,condition_1 FROM Experiment WHERE orgId=?",(fb,)):
        v=np.zeros(200+3,dtype=np.float32)
        for c in media_comps.get(media,set()):
            if c in comp2i: v[comp2i[c]]=1.0
        v[200]=1.0 if aerobic=='aerobic' else 0.0
        # condition_1 token: "none" or chemical-like; mark binary "has stress"
        v[201]=0.0 if (cond1 or '').lower() in ('none','') else 1.0
        # log-num-compounds as a length-style feature
        v[202]=np.log1p(int(v[:200].sum()))
        cf[expName]=v
    cond_feats[fb]=cf
    print(f"  {fb:<10} {len(cf)} experiments")

# 4. fitness triples -> arrays
print("\nbuilding fitness triples per org...", flush=True)
triples_per_org={}
for fb in PILOT:
    gset=all_gene_feats[fb]; cset=cond_feats[fb]
    rows=[]
    for lid,exp,fit,t in cur.execute(f'SELECT locusId,expName,fit,t FROM GeneFitness WHERE orgId=?',(fb,)):
        if fit is None or lid not in gset or exp not in cset: continue
        rows.append((lid,exp,float(fit),float(t) if t is not None else 0.0))
    triples_per_org[fb]=rows
    print(f"  {fb:<10} {len(rows)} fitness triples")

# 5. save
print("\nsaving compact npz...", flush=True)
for fb in PILOT:
    g_keys=sorted(all_gene_feats[fb])
    G=np.stack([all_gene_feats[fb][k] for k in g_keys])
    c_keys=sorted(cond_feats[fb])
    C=np.stack([cond_feats[fb][k] for k in c_keys])
    g_idx={k:i for i,k in enumerate(g_keys)}
    c_idx={k:i for i,k in enumerate(c_keys)}
    tri=triples_per_org[fb]
    gi=np.array([g_idx[t[0]] for t in tri],dtype=np.int32)
    ci=np.array([c_idx[t[1]] for t in tri],dtype=np.int32)
    fit=np.array([t[2] for t in tri],dtype=np.float32)
    tt=np.array([t[3] for t in tri],dtype=np.float32)
    np.savez_compressed(OUT/f"{fb}.npz", G=G, C=C, gi=gi, ci=ci, fit=fit, t=tt,
                        g_keys=np.array(g_keys),c_keys=np.array(c_keys))
print(f"\nwrote {OUT}/<org>.npz x {len(PILOT)}")
