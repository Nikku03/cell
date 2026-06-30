"""Kitchen-sink sequence promoter model: UP element + -35 + -10 + extended-10 +
discriminator + spacer + operator/dyad + position + phase + DNA shape (roll/twist/
MGW) + H-bond energy + dwell. Predict the functional outputs that matter:
  (A) expression level (RNAP recruitment rate, PRECISE)   -> regression
  (B) essentiality (Keio)                                  -> classification
Held-out (70/30). Report combined model vs best single feature vs measured-expr.
All features are sequence-derivable (universal-from-genome).
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); COE=Path("data/external_data/coexpression")
rng=np.random.default_rng(0)
B={"A":0,"C":1,"G":2,"T":3}
def enc(s): return np.array([B.get(c,-1) for c in s],dtype=np.int8)
def rc(a): comp=np.array([3,2,1,0,-1]); return comp[a][::-1]
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]
G="".join(seq).upper(); GLEN=len(G); Genc=enc(G)
genes={}
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes[p[1].lower()]=(int(p[3]),int(p[4]),p[5])
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
expr={}
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); next(r)
    for line in r: expr[line[0]]=float(np.mean([float(x) for x in line[1:]]))
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"]);
        if b: ess_b[b]=int(r["essential"])
M35=enc("TTGACA"); M10=enc("TATAAT")
TW={"AA":35.6,"AT":31.5,"TA":36.0,"AC":34.4,"GT":34.4,"AG":27.7,"CT":27.7,"CA":34.5,"TG":34.5,"CC":33.7,"GG":33.7,"CG":29.8,"GA":36.9,"TC":36.9,"GC":40.0,"TT":35.6}
RO={"AA":0.7,"AT":1.1,"TA":3.3,"AC":0.6,"GT":0.6,"AG":3.1,"CT":3.1,"CA":3.3,"TG":3.3,"CC":3.6,"GG":3.6,"CG":6.0,"GA":1.6,"TC":1.6,"GC":1.2,"TT":0.7}
inv="ACGT"
def matchrun(a,m):
    n=len(a)-len(m)+1
    return np.array([int((a[i:i+len(m)]==m).sum()) for i in range(n)]) if n>0 else np.zeros(0)
def feats(name,pad=150):
    if name not in genes: return None
    s,e,st=genes[name]
    a=Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
    if len(a)<60 or not (a>=0).all(): return None
    L=len(a)  # a[L-1] is just upstream of gene start (position -1)
    def reg(lo,hi):  # positions relative to start: -hi..-lo  -> slice
        return a[max(0,L-hi):max(0,L-lo)]
    # UP element: AT-rich -40..-60
    up=reg(40,60); up_at=float(np.mean((up==0)|(up==3))) if len(up) else 0.5
    # -35 / -10 boxes + best sigma pair with spacer 15-19
    s35=matchrun(a,M35); s10=matchrun(a,M10); best=0; bpos=L; bsp=17
    for i in range(len(s35)):
        for sp in range(15,20):
            j=i+6+sp
            if j<len(s10):
                v=s35[i]+s10[j]
                if v>best: best=v; bpos=j; bsp=sp
    m35=int(s35.max()) if len(s35) else 0; m10=int(s10.max()) if len(s10) else 0
    spacer_opt=-abs(bsp-17)
    # extended -10: TG dinucleotide just upstream of a -10 hit (approx): count TG in window
    tg=sum(1 for k in range(L-1) if a[k]==3 and a[k+1]==2)
    # discriminator: GC fraction in -1..-8
    disc=reg(1,8); disc_gc=float(np.mean((disc==1)|(disc==2))) if len(disc) else 0.5
    # operator/dyad: max palindrome self-complementarity over 16bp windows
    dy=0.0
    for k in range(0,L-16,3):
        w=a[k:k+16]; rcw=rc(w); dy=max(dy,float((w==rcw).mean()))
    # position/phase of best sigma pair
    dist=(L-bpos)/pad; phase=2*np.pi*((L-bpos)%10.5)/10.5
    # shape: mean roll/twist, AT-run (MGW proxy)
    di=[inv[a[k]]+inv[a[k+1]] for k in range(L-1)]
    roll=float(np.mean([RO[d] for d in di])); tw=float(np.mean([TW[d] for d in di]))
    run=mx=0
    for b_ in a:
        if b_ in (0,3): run+=1; mx=max(mx,run)
        else: run=0
    atrun=mx/L
    # energy / H-bond, gc, dwell
    nAT=int(((a==0)|(a==3)).sum()); hb=(2*nAT+3*(L-nAT))/L; gc=1-nAT/L
    dwell=np.exp(best/3.0)
    return [up_at,m35,m10,best,spacer_opt,tg,disc_gc,dy,dist,np.sin(phase),np.cos(phase),roll,tw,atrun,hb,gc,dwell]
FN=["up_AT","-35","-10","sigma_pair","spacer_opt","ext10_TG","disc_GC","operator_dyad","dist","sin","cos","roll","twist","AT_run","Hbond","GC","dwell"]

X=[];nm=[];bs=[]
for g in genes:
    b=name2b.get(g); f=feats(g)
    if b and f: X.append(f); nm.append(g); bs.append(b)
X=np.array(X,float)
yE=np.array([expr.get(b,np.nan) for b in bs])
yS=np.array([ess_b.get(b,-1) for b in bs])
print(f"genes with features: {len(X)}; features: {len(FN)}")

def ridge(Xtr,ytr,Xte,l2=1.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-9]=1; Xn=(Xtr-mu)/sd; Xen=(Xte-mu)/sd
    Xn=np.hstack([Xn,np.ones((len(Xn),1))]); Xen=np.hstack([Xen,np.ones((len(Xen),1))])
    A=Xn.T@Xn+l2*np.eye(Xn.shape[1]); w=np.linalg.solve(A,Xn.T@ytr); return Xen@w
def logit(Xtr,ytr,Xte,l2=1.0,it=400,lr=0.2):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-9]=1; Xn=(Xtr-mu)/sd; Xen=(Xte-mu)/sd
    Xn=np.hstack([Xn,np.ones((len(Xn),1))]); Xen=np.hstack([Xen,np.ones((len(Xen),1))]); w=np.zeros(Xn.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xn@w)); g=Xn.T@(p-ytr)/len(ytr)+l2*np.r_[w[:-1],0]/len(ytr); w-=lr*g
    return 1/(1+np.exp(-Xen@w))
def spear(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return float(np.corrcoef(ra,rb)[0,1])
def auc(p,y):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

# ---- (A) expression regression ----
mE=~np.isnan(yE); Xe=X[mE]; ye=yE[mE]
idx=rng.permutation(len(Xe)); cut=int(0.7*len(idx)); tr,te=idx[:cut],idx[cut:]
pred=ridge(Xe[tr],ye[tr],Xe[te])
print(f"\n(A) EXPRESSION (RNAP recruitment) from sequence promoter model:")
print(f"    combined model: Spearman(held-out) = {spear(pred,ye[te]):.3f}")
best_single=max(FN,key=lambda f: abs(spear(Xe[:,FN.index(f)],ye)))
print(f"    best single feature: {best_single} (corr {spear(Xe[:,FN.index(best_single)],ye):+.3f})")
# top feature correlations
cors=sorted([(f,spear(Xe[:,FN.index(f)],ye)) for f in FN],key=lambda x:-abs(x[1]))
print("    top feature corrs:", ", ".join(f"{f}{c:+.2f}" for f,c in cors[:6]))

# ---- (B) essentiality classification ----
mS=yS>=0; Xs=X[mS]; ys=yS[mS]
idx=rng.permutation(len(Xs)); cut=int(0.7*len(idx)); tr,te=idx[:cut],idx[cut:]
ps=logit(Xs[tr],ys[tr].astype(float),Xs[te])
print(f"\n(B) ESSENTIALITY from sequence promoter model:")
print(f"    combined model: AUC(held-out) = {auc(ps,ys[te]):.3f}   (vs measured-expression baseline 0.682)")
print(f"    n={len(Xs)} ess={int(ys.sum())}")
