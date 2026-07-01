"""Take a normal (wild-type) protein and a POPULATION (non-disease) variant, compare the
structure, and report: same fold or changed? For a single point substitution the backbone
fold is essentially fixed; what we quantify is (a) where the residue sits (surface/core/
functional site) on the AlphaFold structure, (b) how big the amino-acid change is
(volume/hydrophobicity/charge), (c) ESM's variant-effect score. Contrast with disease
variants."""
import json, urllib.request, re, time, warnings, socket
socket.setdefaulttimeout(35); warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
OUT=Path("outputs/orphan"); SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
# amino-acid properties
VOL={'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
KD={'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
CH={'D':-1,'E':-1,'K':1,'R':1,'H':0.1}
AA3={'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
# gene -> UniProt ACC + sequence
acc={}; seqs={}; sym=None; buf=[]
for l in open(SC/"human.faa"):
    if l.startswith(">"):
        if sym and buf: seqs[sym]=max(seqs.get(sym,""),"".join(buf),key=len)
        buf=[]; m=re.match(r">\w+\|(\w+)\|",l); gn=[t[3:] for t in l.split() if t.startswith("GN=")]
        sym=gn[0] if gn else None
        if m and sym: acc.setdefault(sym,m.group(1))
    else: buf.append(l.strip())
if sym and buf: seqs[sym]=max(seqs.get(sym,""),"".join(buf),key=len)
# variants: (gene, pos, WT, MUT, label)
VARS=[
 ("SLC24A5",111,"T","A","population/adaptive (light skin)"),
 ("SLC45A2",374,"L","F","population/adaptive (pigmentation)"),
 ("MC1R",163,"R","Q","population/adaptive (E.Asian)"),
 ("ACKR1",44,"G","D","population/adaptive (Duffy, malaria)"),
 ("ALDH2",504,"E","K","population (Asian-flush, alcohol metabolism)"),
 ("MTHFR",222,"A","V","population (common polymorphism)"),
 ("HBB",6,"E","V","DISEASE (sickle-cell)"),
 ("PAH",408,"R","W","DISEASE (phenylketonuria)"),
]
def hj(url):
    for _ in range(2):
        try: return json.load(urllib.request.urlopen(url,timeout=30))
        except Exception: time.sleep(2)
    return None
def af(ac):
    api=hj(f"https://alphafold.ebi.ac.uk/api/prediction/{ac}")
    if not api: return None
    try: txt=urllib.request.urlopen(api[0]["pdbUrl"],timeout=30).read().decode()
    except: return None
    ca={}
    for ln in txt.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip()=="CA":
            ca[int(ln[22:26])]=(float(ln[30:38]),float(ln[38:46]),float(ln[46:54]),float(ln[60:66]))
    return ca
print("loading ESM-2 8M ...",flush=True)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
mdl=AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D").eval(); torch.set_num_threads(4)
MASK=tok.mask_token_id
@torch.no_grad()
def esm_llr(seq,pos,wt,mut):
    seq=seq[:1022]; ids=tok(seq,return_tensors="pt")["input_ids"]
    if pos>len(seq): return None
    ids[0,pos]=MASK
    lp=torch.log_softmax(mdl(input_ids=ids).logits[0,pos],-1)
    return float(lp[tok.convert_tokens_to_ids(mut)]-lp[tok.convert_tokens_to_ids(wt)])
rows=[]
for g,pos,wt,mut,lab in VARS:
    ac=acc.get(g); seq=seqs.get(g); ca=af(ac) if ac else None
    if not ca or not seq: print(f"{g}: skip"); continue
    rn=sorted(ca); xyz=np.array([ca[r][:3] for r in rn])
    i=rn.index(pos) if pos in ca else None
    contacts=int((np.sqrt(((xyz-xyz[i])**2).sum(1))<10).sum())-1 if i is not None else None
    plddt=ca[pos][3] if pos in ca else None
    dvol=round(VOL[mut]-VOL[wt],1); dkd=round(KD[mut]-KD[wt],1); dch=round(CH.get(mut,0)-CH.get(wt,0),1)
    llr=esm_llr(seq,pos,wt,mut)
    buried = contacts is not None and contacts>=16
    # verdict
    if llr is not None and llr<-6: verdict="STRUCTURE/FUNCTION LIKELY CHANGED (ESM strongly disfavors)"
    elif buried and (abs(dvol)>40 or abs(dch)>=1): verdict="local core perturbation (buried, big change)"
    elif abs(dch)>=1: verdict="surface change of CHARGE (fold same; new surface property)"
    else: verdict="FOLD ESSENTIALLY UNCHANGED (surface/conservative side-chain swap)"
    rows.append(dict(gene=g,variant=f"{wt}{pos}{mut}",kind=lab,contacts=contacts,plddt=round(plddt,0) if plddt else None,
        location=("buried core" if buried else "surface/exposed"),
        dVolume=dvol,dHydrophobicity=dkd,dCharge=dch,esm_llr=round(llr,2) if llr is not None else None,verdict=verdict))
    print(f"  {g:8s} {wt}{pos}{mut} [{lab.split('(')[0].strip()}] loc={'core' if buried else 'surface'} "
          f"dVol={dvol:+.0f} dCharge={dch:+.0f} ESM={llr:+.1f} -> {verdict}",flush=True)
    time.sleep(0.3)
json.dump(rows,open(OUT/"wt_vs_mutant.json","w"),indent=2)
print("\nwrote wt_vs_mutant.json")
