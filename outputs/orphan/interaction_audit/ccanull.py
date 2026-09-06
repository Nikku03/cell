import numpy as np, sys, json, time
sys.path.insert(0,"/home/user/cell/colab")
from v4_interaction import factors, subspace_alignment
O="/home/user/cell/colab/skep"
I=np.load(O+"/interaction.npy"); n,p=I.shape; RANK=16
real=np.load(O+"/cca_real.npy"); t0=time.time()
def splithalf(M,seed=20_260_802):
    rng=np.random.default_rng(seed); order=rng.permutation(M.shape[0]); h=len(order)//2
    _,L,_=factors(M,order[:h],RANK); _,R,_=factors(M,order[h:2*h],RANK)
    return L,R
# module row-shuffle floor, replicated over seeds on the REAL data
L,R=splithalf(I)
fl=[subspace_alignment(L,R[np.random.default_rng(s).permutation(p)])[0] for s in range(20)]
print("module row-shuffle floor top-CC over 20 seeds: mean %.4f min %.4f max %.4f"%(np.mean(fl),np.min(fl),np.max(fl)))
# nulls
G=(I@I.T).astype(np.float64)/p
w,V=np.linalg.eigh(G); idx=np.argsort(-w)[:100]; P=V[:,idx].astype(np.float32)
resid=I-P@(P.T@I); noise_sd=(resid.std(0)*np.sqrt(n/(n-100))).astype(np.float32)
def signflip(r): return I*((r.integers(0,2,size=I.shape).astype(np.float32)*2-1))
def gauss(r): return r.standard_normal(I.shape).astype(np.float32)*noise_sd[None,:]
def colperm(r):
    M=np.empty_like(I)
    for j in range(p): M[:,j]=I[r.permutation(n),j]
    return M
res={}
for name,fn in (("signflip",signflip),("gauss_hetero",gauss),("colperm",colperm)):
    sp=[];rowsh=[]
    for k in range(5):
        r=np.random.default_rng(2000+k); M=fn(r)
        Ln,Rn=splithalf(M); a=subspace_alignment(Ln,Rn); sp.append(a)
        rowsh.append(subspace_alignment(Ln,Rn[np.random.default_rng(k).permutation(p)])[0])
    sp=np.array(sp); res[name]=sp
    print("%-13s top-CC mean %.4f max %.4f | all-16 mean %.4f | module row-shuffle recipe ON this null: %.4f"%(
        name,sp[:,0].mean(),sp[:,0].max(),sp.mean(),np.mean(rowsh)))
    print("               null CC profile mean:",np.round(sp.mean(0),3).tolist())
    print("   real factors clearing this floor (%.4f): %d/16   t=%.0f"%(sp[:,0].max(),(real>sp[:,0].max()).sum(),time.time()-t0))
json.dump({k:v.tolist() for k,v in res.items()},open(O+"/cca_nulls.json","w"))
# line-side heteroskedasticity check (why colperm differs)
print("\nper-LINE interaction sd: p10 %.4f median %.4f p90 %.4f"%tuple(np.percentile(np.sqrt((I**2).mean(1)),[10,50,90])))
print("per-GENE interaction sd: p10 %.4f median %.4f p90 %.4f"%tuple(np.percentile(np.sqrt((I**2).mean(0)),[10,50,90])))
