import numpy as np, sys, json, time
sys.path.insert(0,"/home/user/cell/colab")
from v4_interaction import factors, subspace_alignment
O="/home/user/cell/colab/skep"
A="/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/audit"
I=np.load(O+"/interaction.npy")
Ith=np.load(A+"/interaction.npy")
print("my interaction vs auditor cache: max abs diff %.3e"%np.nanmax(np.abs(I-Ith)))
n,p=I.shape; RANK=16
rng=np.random.default_rng(20_260_802)
order=rng.permutation(n); h=n//2
_,L,_=factors(I,order[:h],RANK); _,R,_=factors(I,order[h:2*h],RANK)
real=subspace_alignment(L,R)
shuf=R[rng.permutation(p)]
modfloor=subspace_alignment(L,shuf)
print("REAL :",np.round(real,4).tolist())
print("MODFLOOR:",np.round(modfloor,4).tolist())
print("mean real %.4f  floor max %.4f  above %d/16"%(real.mean(),modfloor.max(),(real>modfloor.max()).sum()))
np.save(O+"/cca_real.npy",real); np.save(O+"/cca_modfloor.npy",modfloor)
