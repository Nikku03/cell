import numpy as np, sys, json, time
sys.path.insert(0,"/home/user/cell/colab")
from v4_interaction import factors, subspace_alignment
O="/home/user/cell/colab/skep"
I=np.nan_to_num(np.load(O+"/interaction.npy"),nan=0.0)   # zero-filled, as factors() does
n,p=I.shape; RANK=16; real=np.load(O+"/cca_real.npy"); t0=time.time()
def splithalf(M,seed=20_260_802):
    rng=np.random.default_rng(seed); o=rng.permutation(M.shape[0]); h=len(o)//2
    _,L,_=factors(M,o[:h],RANK); _,R,_=factors(M,o[h:2*h],RANK); return L,R
G=(I@I.T).astype(np.float64)/p
w,V=np.linalg.eigh(G); idx=np.argsort(-w)[:100]; P=V[:,idx].astype(np.float32)
noise_sd=((I-P@(P.T@I)).std(0)*np.sqrt(n/(n-100))).astype(np.float32)
print("per-gene NOISE sd p10/med/p90: %.4f %.4f %.4f"%tuple(np.percentile(noise_sd,[10,50,90])))
sp=[]
for k in range(5):
    r=np.random.default_rng(3000+k)
    M=(r.standard_normal(I.shape).astype(np.float32))*noise_sd[None,:]
    Ln,Rn=splithalf(M); sp.append(subspace_alignment(Ln,Rn))
sp=np.array(sp)
print("gauss_hetero null top-CC mean %.4f max %.4f ; profile"%(sp[:,0].mean(),sp[:,0].max()),np.round(sp.mean(0),3).tolist())
print("real factors clearing gauss floor (%.4f): %d/16"%(sp[:,0].max(),(real>sp[:,0].max()).sum()))
print("per-LINE interaction sd p10/med/p90: %.4f %.4f %.4f"%tuple(np.percentile(np.sqrt((I**2).mean(1)),[10,50,90])))
print("per-GENE interaction sd p10/med/p90: %.4f %.4f %.4f"%tuple(np.percentile(np.sqrt((I**2).mean(0)),[10,50,90])))
print("t=%.0f"%(time.time()-t0))
