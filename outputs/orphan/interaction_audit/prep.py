import numpy as np, pandas as pd, json, time
SP="/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
OUT="/home/user/cell/colab/skep"
t0=time.time()
frame=pd.read_csv(SP+"/CRISPRGeneEffect.csv", index_col=0)
frame.columns=[c.split(" (")[0] for c in frame.columns]
frame=frame.loc[:,~frame.columns.duplicated()]
raw=frame.to_numpy(np.float32)
genes_all=list(frame.columns)
keep=np.isfinite(raw).sum(0)>=100
raw,genes_all=raw[:,keep],[g for g,k in zip(genes_all,keep) if k]
observed=np.isfinite(raw)
print("shape",raw.shape,"obs frac %.6f"%observed.mean(),"t=%.0f"%(time.time()-t0))
np.save(OUT+"/raw.npy",raw); np.save(OUT+"/observed.npy",observed)
json.dump(genes_all,open(OUT+"/genes_all.json","w"))
json.dump(list(frame.index),open(OUT+"/lines.json","w"))

# replicate decompose() exactly
filled=np.where(observed,raw,np.nan)
mu=float(np.nanmean(filled))
gene=np.nanmean(filled,axis=0)-mu
line=np.nanmean(filled-gene[None,:],axis=1)-mu
interaction=filled-mu-gene[None,:]-line[:,None]
total=float(np.nanvar(filled))
parts={
 "gene_marginal": float(np.nanvar(np.broadcast_to(gene[None,:],filled.shape))),
 "line_marginal": float(np.nanvar(np.broadcast_to(line[:,None],filled.shape))),
 "interaction": float(np.nanvar(interaction)),
}
print("mu",mu,"total",total)
s=0.0
for k,v in parts.items():
    print("  %-16s %.6f  %7.4f%%"%(k,v,100*v/total)); s+=v/total
print("  SUM OF SHARES = %.6f"%s)
# gene marginal restricted to observed cells
gm_obs=float(np.nanvar(np.where(observed,np.broadcast_to(gene[None,:],filled.shape),np.nan)))
lm_obs=float(np.nanvar(np.where(observed,np.broadcast_to(line[:,None],filled.shape),np.nan)))
print("  gene_marginal on OBSERVED cells only: %.6f = %.4f%%"%(gm_obs,100*gm_obs/total))
print("  line_marginal on OBSERVED cells only: %.6f = %.4f%%"%(lm_obs,100*lm_obs/total))
print("  corrected sum = %.6f"%((gm_obs+lm_obs+parts["interaction"])/total))
np.save(OUT+"/interaction.npy",interaction.astype(np.float32))
np.save(OUT+"/gene_mean.npy",gene); np.save(OUT+"/line_mean.npy",line)
json.dump({"mu":mu,"total":total,"parts":parts,"gm_obs":gm_obs,"lm_obs":lm_obs},open(OUT+"/budget.json","w"))
cnt=observed.sum(0)
print("per-gene observed count: min %d p1 %d median %d ; genes<1100: %d"%(cnt.min(),np.percentile(cnt,1),np.median(cnt),(cnt<1100).sum()))
print("t=%.0f"%(time.time()-t0))
