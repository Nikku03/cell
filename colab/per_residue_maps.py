"""Per-residue VITAL-domain maps: ESM-2 zero-shot constraint (mask each position, WT
log-prob) + observed gnomAD missense density. Shows the exact vital regions within a
gene, and that adaptation variants sit in tolerant regions."""
import json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
raw=json.load(open(SC/"gnomad_genes.json"))
def parse_faa(path):
    seqs={}; sym=None; buf=[]
    def flush():
        if sym and buf:
            s="".join(buf)
            if sym not in seqs or len(s)>len(seqs[sym]): seqs[sym]=s
    for l in open(path):
        if l.startswith(">"):
            flush(); buf=[]; sym=None
            for t in l.split():
                if t.startswith("GN="): sym=t[3:]
        else: buf.append(l.strip())
    flush(); return seqs
hsym=parse_faa(SC/"human.faa")
import re
def obs_missense(sym):
    d={} # pos -> max AF
    for v in raw.get(sym,[]):
        if not v.get("exome") or not v["exome"].get("an"): continue
        m=re.search(r"p\.[A-Za-z]{3}(\d+)",v.get("hgvsp") or "")
        if not m: continue
        p=int(m.group(1)); ex=v["exome"]; af=ex["ac"]/ex["an"] if ex["an"] else 0
        d[p]=max(d.get(p,0),af)
    return d
tok=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
mdl=AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D").eval(); torch.set_num_threads(8)
MASK=tok.mask_token_id
@torch.no_grad()
def esm_constraint(seq):
    seq=seq[:600]; ids=tok(seq,return_tensors="pt")["input_ids"][0]
    L=len(seq); wtlp=np.zeros(L)
    for i in range(0,L,32):
        js=list(range(i,min(i+32,L)))
        batch=ids.repeat(len(js),1)
        for k,j in enumerate(js): batch[k,j+1]=MASK    # +1 for CLS
        out=mdl(input_ids=batch).logits
        lp=torch.log_softmax(out,dim=-1)
        for k,j in enumerate(js):
            wt=ids[j+1]; wtlp[j]=lp[k,j+1,wt].item()
    return wtlp   # higher = model confident WT is right = constrained
GENES=[("EEF2","essential"),("SLC24A5","adaptation"),("MC1R","adaptation")]
fig,axes=plt.subplots(len(GENES),1,figsize=(12,3*len(GENES)))
res={}
for ax,(sym,cat) in zip(axes,GENES):
    seq=hsym.get(sym)
    if not seq: ax.set_title(f"{sym}: no seq"); continue
    c=esm_constraint(seq); obs=obs_missense(sym)
    x=np.arange(1,len(c)+1)
    ax.plot(x,-c,lw=.6,color="#333")   # -logprob: higher = more tolerant
    ax.fill_between(x,-c,-c.min(),color="#ddd",alpha=.4)
    # observed common missense
    for p,af in obs.items():
        if p<=len(c):
            ax.scatter(p,-c[p-1],s=10+200*min(af,0.5),
                c=("#e74c3c" if af>0.01 else "#3498db"),zorder=3,alpha=.7)
    vital=int((c> np.quantile(c,0.75)).sum())
    ax.set_title(f"{sym} ({cat}) — line=ESM tolerance (up=tolerant), dots=observed missense (red=common>1%)")
    ax.set_ylabel("tolerance")
    res[sym]=dict(mean_esm_logprob=round(float(c.mean()),3),
        n_common_missense=int(sum(1 for a in obs.values() if a>0.01)),
        n_obs_missense=len(obs))
axes[-1].set_xlabel("residue position")
plt.tight_layout(); plt.savefig(OUT/"per_residue_maps.png",dpi=120)
json.dump(res,open(OUT/"per_residue_maps.json","w"),indent=2)
print("wrote per_residue_maps.png +", json.dumps(res))
