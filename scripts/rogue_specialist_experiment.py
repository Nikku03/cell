"""Two-tier rogue-essential experiment -- leak-free, sandbox-runnable.

THESIS: a fused model with a dominant conservation prior (family_frac)
cannot recover rogue essentials (essential genes whose FAMILY is not
usually essential), because the prior is confidently WRONG on exactly
those genes. A dedicated specialist trained ONLY on the rogue zone,
with family_frac REMOVED, forced to use organism-INTRINSIC features,
should recover more of them.

WHAT THIS TESTS (sandbox, no GPU/ESM):
  Tier 1 (fused)      : XGBoost on family_frac + all intrinsic features
  Tier 2 (specialist) : XGBoost on rogue-zone genes only, family_frac
                        REMOVED, organism-intrinsic features only
  Compared on: rogue-essential recovery + rogue-zone MCC.

ORGANISM-INTRINSIC FEATURES (the ESM structural-isozyme is replaced by
a sandbox-computable sequence proxy; the real run swaps in ESM):
  - structural_uniqueness : within-genome best k-mer match. LOW = no
                            sequence backup in this genome = candidate
                            sole-provider. (label-free, intra-genome)
  - functional_redundancy : another gene with same product (isozyme)
  - biophysics            : length, GRAVY, aromaticity, charged-aa frac
  - operon/position       : intergenic dist, same-strand flags, operon
  (NO neighbour-ESSENTIALITY-LABEL feature -- that would leak test labels)

================ LEAK CONTROL (audited at runtime) ================
  1. family_frac recomputed per held-out organism: excludes that
     organism's own essentiality from the numerator/denominator.
  2. Tier-2 specialist trained ONLY on training organisms' rogue genes;
     held-out organism never in its training set.
  3. All sequence/structure/position features are computed from each
     gene's OWN genome with NO labels -> cannot leak.
  4. Rogue-zone membership uses the leak-free family_frac.
  5. Fixed hyperparameters -- no tuning that peeks at the test fold.
==================================================================
"""
from __future__ import annotations
import math, re, sys, time
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/drive_import"

ROGUE_FF_THRESHOLD = 0.40   # "rogue zone" = leak-free family_frac < this
KMER_K = 6
KMER_CAP = 60               # ignore k-mers shared by >this many proteins (low-complexity)

KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"E":-3.5,"Q":-3.5,"G":-0.4,
      "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
      "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
AROM = set("FWY"); CHARGED = set("DEKR")


def parse_faa(path):
    out={}; cur=None; buf=[]
    for line in open(path):
        if line.startswith(">"):
            if cur is not None: out[cur]="".join(buf)
            cur=line[1:].split()[0]; buf=[]
        else: buf.append(line.strip())
    if cur is not None: out[cur]="".join(buf)
    return out


def parse_gff(path):
    gene_attrs={}; cds=[]
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        p=line.rstrip("\n").split("\t")
        if len(p)<9: continue
        d={}
        for kv in p[8].split(";"):
            if "=" in kv: k,v=kv.split("=",1); d[k]=v
        if p[2]=="gene" and "ID" in d:
            gene_attrs[d["ID"]]={"locus_tag":d.get("locus_tag",""),"gene":d.get("gene","")}
        elif p[2]=="CDS":
            cds.append((p,d))
    feats=[]
    for p,d in cds:
        lt=d.get("locus_tag",""); old=d.get("old_locus_tag","")
        if not lt and d.get("Parent","") in gene_attrs:
            lt=gene_attrs[d["Parent"]]["locus_tag"]
        feats.append({"seqid":p[0],"start":int(p[3]),"end":int(p[4]),"strand":p[6],
                       "locus_tag":lt,"old_locus_tag":old,
                       "product":d.get("product",""),"protein_id":d.get("protein_id","")})
    return feats


def structural_uniqueness(seqs, k=KMER_K, cap=KMER_CAP):
    """For each protein, best normalized k-mer overlap with ANY OTHER protein
    in the SAME genome. Label-free, intra-genome. LOW => structurally unique
    => candidate sole-provider. This is the sandbox proxy for ESM-space
    within-genome nearest-neighbour."""
    ksets=[]
    for s in seqs:
        ksets.append(set(s[i:i+k] for i in range(len(s)-k+1)) if len(s)>=k else set())
    inv=defaultdict(list)
    for idx,ks in enumerate(ksets):
        for km in ks: inv[km].append(idx)
    for km in list(inv):
        if len(inv[km])>cap: del inv[km]
    best=[0.0]*len(seqs)
    for idx,ks in enumerate(ksets):
        if not ks: continue
        shared=Counter()
        for km in ks:
            for j in inv.get(km,()):
                if j!=idx: shared[j]+=1
        if shared:
            j,c=shared.most_common(1)[0]
            denom=min(len(ks),len(ksets[j])) or 1
            best[idx]=c/denom
    return best


def biophys(seq):
    aa=[a for a in seq if a in KD]
    n=len(aa) or 1
    return {
        "length_aa": len(seq)//1,
        "gravy": sum(KD[a] for a in aa)/n,
        "aromaticity": sum(1 for a in aa if a in AROM)/n,
        "charged_frac": sum(1 for a in aa if a in CHARGED)/n,
        "frac_K": sum(1 for a in aa if a=="K")/n,
        "frac_R": sum(1 for a in aa if a=="R")/n,
    }


def norm_product(p):
    p=p.lower().strip()
    p=re.sub(r"\b(putative|probable|predicted)\b","",p)
    return re.sub(r"\s+"," ",p).strip()


def main():
    import pandas as pd, numpy as np, xgboost as xgb
    from sklearn.metrics import matthews_corrcoef

    print("="*70)
    print("TWO-TIER ROGUE-ESSENTIAL EXPERIMENT (leak-free)")
    print("="*70)

    labels=pd.read_csv(D/"labels/labels.csv")
    orth=pd.read_csv(D/"labels/orthology_features.csv")
    mf=pd.read_csv(D/"labels/genome_cache_manifest.csv")
    elig=mf[(mf.kind=="ours")&mf.join_rate.notna()&(mf.join_rate>=0.3)]
    accs=dict(zip(elig.organism,elig.accession))

    # ---- OG aggregates for LEAK-FREE family_frac ----
    # built from ALL organisms' labels via orthology; per held-out org we
    # SUBTRACT that org's contribution.
    j=labels.merge(orth[["organism","locus_tag","og_id"]],on=["organism","locus_tag"],how="inner")
    j=j[j.og_id.notna()].copy(); j.essential=j.essential.astype(int)
    og_tot_e=j.groupby("og_id").essential.sum()
    og_tot_n=j.groupby("og_id").essential.size()
    og_org=j.groupby(["og_id","organism"]).essential.agg(["sum","size"])

    def family_frac_excl(org):
        try:
            s=og_org.xs(org,level="organism"); e,n=s["sum"],s["size"]
        except KeyError:
            e=pd.Series(0,index=og_tot_e.index); n=pd.Series(0,index=og_tot_n.index)
        E=og_tot_e.sub(e,fill_value=0); N=og_tot_n.sub(n,fill_value=0)
        return E/N.where(N>0)

    # ---- per-organism feature table (intrinsic features, label-free) ----
    print("\n[build] per-organism intrinsic features (label-free)...")
    og_map=dict(zip(zip(j.organism,j.locus_tag),j.og_id))
    lab_map=dict(zip(zip(labels.organism,labels.locus_tag),labels.essential.astype(int)))
    rows=[]
    for org in sorted(accs):
        gff=D/"genome_cache"/accs[org]/"genomic.gff"
        faa=D/"genome_cache"/accs[org]/"protein.faa"
        if not gff.exists() or not faa.exists(): continue
        feats=parse_gff(gff); feats.sort(key=lambda r:(r["seqid"],r["start"]))
        seqd=parse_faa(faa)
        # gather sequences in feature order (for structural uniqueness)
        idxs=[]; seqs=[]
        for i,f in enumerate(feats):
            s=seqd.get(f["protein_id"],"")
            if s: idxs.append(i); seqs.append(s)
        t0=time.time()
        uniq=structural_uniqueness(seqs)
        uniq_map={feats[idxs[m]]["locus_tag"]:uniq[m] for m in range(len(idxs))}
        # functional redundancy (product duplicates in genome)
        prod_counts=Counter(norm_product(f["product"]) for f in feats if f["product"])
        for i,f in enumerate(feats):
            lt=f["locus_tag"]
            e=lab_map.get((org,lt))
            if e is None and f["old_locus_tag"]: e=lab_map.get((org,f["old_locus_tag"]))
            if e is None: continue   # only labeled genes enter the experiment
            ogid=og_map.get((org,lt)) or og_map.get((org,f["old_locus_tag"]))
            s=seqd.get(f["protein_id"],"")
            if not s: continue
            bp=biophys(s)
            prev=feats[i-1] if i>0 and feats[i-1]["seqid"]==f["seqid"] else None
            nxt=feats[i+1] if i+1<len(feats) and feats[i+1]["seqid"]==f["seqid"] else None
            ig_prev=(f["start"]-prev["end"]-1) if prev else 0
            ig_next=(nxt["start"]-f["end"]-1) if nxt else 0
            ss_prev=int(prev["strand"]==f["strand"]) if prev else 0
            ss_next=int(nxt["strand"]==f["strand"]) if nxt else 0
            npd=norm_product(f["product"]) if f["product"] else ""
            rows.append({
                "organism":org,"locus_tag":lt,"og_id":ogid,"essential":e,
                # ---- organism-INTRINSIC (label-free) ----
                "structural_uniqueness":uniq_map.get(lt,0.0),
                "func_redundancy":prod_counts.get(npd,1),
                "length_aa":bp["length_aa"],"gravy":bp["gravy"],
                "aromaticity":bp["aromaticity"],"charged_frac":bp["charged_frac"],
                "frac_K":bp["frac_K"],"frac_R":bp["frac_R"],
                "intergenic_prev":ig_prev,"intergenic_next":ig_next,
                "same_strand_prev":ss_prev,"same_strand_next":ss_next,
                "operon":int(ig_prev<50 and ss_prev==1),
                "strand":1 if f["strand"]=="+" else -1,
            })
        print(f"  {org:<28s} {sum(1 for r in rows if r['organism']==org):>5d} genes  "
              f"(uniq {time.time()-t0:.0f}s)")
    df=pd.DataFrame(rows)
    print(f"\n[data] {len(df):,} labeled genes across {df.organism.nunique()} organisms, "
          f"{int(df.essential.sum()):,} essential ({df.essential.mean()*100:.0f}%)")

    INTRINSIC=["structural_uniqueness","func_redundancy","length_aa","gravy",
               "aromaticity","charged_frac","frac_K","frac_R",
               "intergenic_prev","intergenic_next","same_strand_prev",
               "same_strand_next","operon","strand"]

    # ---- LEAK AUDIT ----
    print("\n"+"="*70)
    print("LEAK AUDIT")
    print("="*70)
    print("  family_frac: recomputed per held-out org via family_frac_excl(org)")
    print("               -> excludes the test organism's own labels. LEAK-FREE.")
    print(f"  intrinsic features ({len(INTRINSIC)}): all computed from each gene's")
    print("               OWN genome sequence/position, NO labels. LEAK-FREE.")
    print("  NO neighbour-essentiality-label feature is used (would leak).")
    print("  Tier-2 trains only on TRAINING orgs' rogue genes; test org excluded.")
    print("  Fixed hyperparameters (no test-peeking tuning).")
    print("="*70)

    XGB=dict(n_estimators=500,max_depth=5,learning_rate=0.04,subsample=0.85,
             colsample_bytree=0.7,min_child_weight=2,tree_method="hist",
             eval_metric="logloss",verbosity=0,n_jobs=-1,random_state=0)

    eval_orgs=sorted(df.organism.unique())
    fused_pred={}; tier2_pred={}; cons_pred={}
    t2_importances=[]
    for O in eval_orgs:
        ff=family_frac_excl(O)
        df["family_frac"]=df.og_id.map(ff)        # leak-free for THIS fold
        tr=df[df.organism!=O].copy(); te=df[df.organism==O].copy()
        if te.essential.nunique()<2: continue

        # ----- Tier 1: FUSED (family_frac + intrinsic) -----
        cols_fused=["family_frac"]+INTRINSIC
        Xtr=tr[cols_fused].fillna(0).values; ytr=tr.essential.values
        Xte=te[cols_fused].fillna(0).values; yte=te.essential.values
        spw=(len(ytr)-ytr.sum())/max(ytr.sum(),1)
        m=xgb.XGBClassifier(scale_pos_weight=spw,**XGB); m.fit(Xtr,ytr)
        pf=m.predict_proba(Xte)[:,1]

        # ----- Conservation-only baseline (family_frac alone) -----
        mc=xgb.XGBClassifier(scale_pos_weight=spw,**XGB)
        mc.fit(tr[["family_frac"]].fillna(0).values,ytr)
        pc=mc.predict_proba(te[["family_frac"]].fillna(0).values)[:,1]

        # ----- Tier 2: ROGUE SPECIALIST -----
        # rogue zone = leak-free family_frac < threshold (train + test)
        tr_rogue=tr[tr.family_frac.fillna(0)<ROGUE_FF_THRESHOLD]
        te_rogue_mask=te.family_frac.fillna(0)<ROGUE_FF_THRESHOLD
        te_rogue=te[te_rogue_mask]
        if len(tr_rogue)>200 and tr_rogue.essential.nunique()==2 and len(te_rogue)>0:
            Xr=tr_rogue[INTRINSIC].fillna(0).values; yr=tr_rogue.essential.values
            spw_r=(len(yr)-yr.sum())/max(yr.sum(),1)
            mr=xgb.XGBClassifier(scale_pos_weight=spw_r,**XGB); mr.fit(Xr,yr)
            pr=mr.predict_proba(te_rogue[INTRINSIC].fillna(0).values)[:,1]
            t2_importances.append(dict(zip(INTRINSIC,mr.feature_importances_)))
        else:
            pr=np.zeros(len(te_rogue))

        for i,lt in enumerate(te.locus_tag.values):
            fused_pred[(O,lt)]=(yte[i],pf[i],float(te.family_frac.fillna(0).values[i]))
            cons_pred[(O,lt)]=pc[i]
        for i,lt in enumerate(te_rogue.locus_tag.values):
            tier2_pred[(O,lt)]=pr[i]

        print(f"  [{O:<26s}] test={len(te):>5d}  rogue-zone={int(te_rogue_mask.sum()):>4d}  "
              f"rogue-ess={int(te_rogue.essential.sum()):>3d}")

    # ---- METRICS ----
    print("\n"+"="*70); print("RESULTS"); print("="*70)

    keys=list(fused_pred); y=np.array([fused_pred[k][0] for k in keys])
    pf=np.array([fused_pred[k][1] for k in keys])
    ff=np.array([fused_pred[k][2] for k in keys])
    pc=np.array([cons_pred[k] for k in keys])

    def mcc_at(yv,pv,t=0.5):
        pred=(pv>t).astype(int)
        return matthews_corrcoef(yv,pred) if yv.sum() and yv.sum()<len(yv) else float("nan")

    print(f"\nOVERALL (all {len(keys):,} genes):")
    print(f"  conservation-only  MCC@0.5 = {mcc_at(y,pc):+.4f}")
    print(f"  fused (cons+intr)  MCC@0.5 = {mcc_at(y,pf):+.4f}")

    # ---- the part that matters: the ROGUE ZONE ----
    rogue_mask=ff<ROGUE_FF_THRESHOLD
    yr=y[rogue_mask]; pfr=pf[rogue_mask]; pcr=pc[rogue_mask]; keysr=[keys[i] for i in range(len(keys)) if rogue_mask[i]]
    pt=np.array([tier2_pred.get(k,np.nan) for k in keysr])
    has_t2=~np.isnan(pt)
    print(f"\nROGUE ZONE (family_frac<{ROGUE_FF_THRESHOLD}): {len(yr):,} genes, "
          f"{int(yr.sum())} rogue essentials ({100*yr.mean():.0f}%)")
    print(f"  conservation-only  MCC = {mcc_at(yr,pcr):+.4f}")
    print(f"  fused              MCC = {mcc_at(yr,pfr):+.4f}")
    print(f"  TIER-2 specialist  MCC = {mcc_at(yr[has_t2],pt[has_t2]):+.4f}")

    # rogue-essential RECOVERY at matched precision-ish (threshold 0.5)
    def recovery(yv,pv,t=0.5):
        pred=(pv>t).astype(int); tp=int(((pred==1)&(yv==1)).sum())
        return tp,int(yv.sum())
    rt,tot=recovery(yr,pfr); print(f"\n  rogue-essential RECOVERY @0.5:")
    print(f"    conservation-only : {recovery(yr,pcr)[0]:>4d}/{tot}  ({100*recovery(yr,pcr)[0]/max(tot,1):.0f}%)")
    print(f"    fused             : {rt:>4d}/{tot}  ({100*rt/max(tot,1):.0f}%)")
    rt2,_=recovery(yr[has_t2],pt[has_t2]); tot2=int(yr[has_t2].sum())
    print(f"    TIER-2 specialist : {rt2:>4d}/{tot2}  ({100*rt2/max(tot2,1):.0f}%)")

    # ===== DECISIVE: threshold-FREE ranking (AUC) in the rogue zone =====
    from sklearn.metrics import roc_auc_score, average_precision_score
    def safe_auc(yv,pv):
        return roc_auc_score(yv,pv) if yv.sum() and yv.sum()<len(yv) else float("nan")
    print(f"\n  *** ROGUE-ZONE RANKING QUALITY (threshold-free) ***")
    print(f"    ROC-AUC  conservation : {safe_auc(yr,pcr):.4f}")
    print(f"    ROC-AUC  fused        : {safe_auc(yr,pfr):.4f}")
    print(f"    ROC-AUC  TIER-2       : {safe_auc(yr[has_t2],pt[has_t2]):.4f}   <-- vs fused decides it")
    print(f"    PR-AUC   fused        : {average_precision_score(yr,pfr):.4f}")
    print(f"    PR-AUC   TIER-2       : {average_precision_score(yr[has_t2],pt[has_t2]):.4f}")
    print(f"    (base rate = {yr.mean():.3f})")

    # ===== FAIR comparison: recall at MATCHED precision =====
    def recall_at_precision(yv,pv,target_p=0.40):
        order=np.argsort(-pv); ys=yv[order]
        tp=fp=0; tot=int(yv.sum()); best_rec=0.0
        for yi in ys:
            if yi==1: tp+=1
            else: fp+=1
            p=tp/(tp+fp)
            if p>=target_p: best_rec=max(best_rec,tp/max(tot,1))
        return best_rec
    print(f"\n  *** rogue-essential RECALL at matched precision (fair) ***")
    print(f"    {'':14s} {'conservation':>13s} {'fused':>8s} {'TIER-2':>8s}")
    for tp_ in (0.30,0.40,0.50):
        rc=recall_at_precision(yr,pcr,tp_)
        rf=recall_at_precision(yr,pfr,tp_); rt=recall_at_precision(yr[has_t2],pt[has_t2],tp_)
        print(f"    @prec>={tp_:.2f}:   {rc:>13.3f} {rf:>8.3f} {rt:>8.3f}")
    print(f"    (constructive: fused vs conservation = does adding INTRINSIC")
    print(f"     features to the prior improve rogue recovery at usable precision?)")

    # ===== what did tier-2 actually use? =====
    if t2_importances:
        agg=defaultdict(float)
        for d in t2_importances:
            for k_,v in d.items(): agg[k_]+=v/len(t2_importances)
        print(f"\n  TIER-2 feature importance (mean across folds):")
        for k_,v in sorted(agg.items(),key=lambda x:-x[1])[:8]:
            star=" <-- ESM proxy" if k_=="structural_uniqueness" else ""
            print(f"    {k_:<22s} {v:.3f}{star}")

    # save
    out=REPO/"outputs/rogue_specialist_results.md"
    out.parent.mkdir(exist_ok=True)
    with open(out,"w") as f:
        f.write(f"# Two-tier rogue-essential experiment (leak-free)\n\n")
        f.write(f"{len(keys):,} genes, {df.organism.nunique()} organisms.\n\n")
        f.write(f"## Overall\n- conservation MCC {mcc_at(y,pc):+.4f}\n- fused MCC {mcc_at(y,pf):+.4f}\n\n")
        f.write(f"## Rogue zone (family_frac<{ROGUE_FF_THRESHOLD}, {len(yr)} genes, {int(yr.sum())} rogue essentials)\n")
        f.write(f"- conservation MCC {mcc_at(yr,pcr):+.4f}\n- fused MCC {mcc_at(yr,pfr):+.4f}\n")
        f.write(f"- **tier-2 specialist MCC {mcc_at(yr[has_t2],pt[has_t2]):+.4f}**\n\n")
        f.write(f"### Rogue recovery @0.5\n")
        f.write(f"- conservation {recovery(yr,pcr)[0]}/{tot}\n- fused {rt}/{tot}\n")
        f.write(f"- **tier-2 {rt2}/{tot2}**\n")
    print(f"\nwrote {out}")
    return 0

if __name__=="__main__":
    sys.exit(main())
