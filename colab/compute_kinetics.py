"""KINETICS (imputed, honest) — measured anchors + family prior + network propagation x abundance.

The honest 'toward kinetics' layer (see KINETICS_ASSESSMENT.md). Measured kcat exists for <10% of human
enzymes, so we IMPUTE the rest and label every value with a confidence tier — never pretend a number is
measured when it isn't:
  1. ANCHORS  — measured kcat from SABIO-RK / eHMN (fact, kept WITH pH/temperature) + BRENDA (flagged
     in-vitro, not trusted as in-vivo).
  2. FAMILY prior — median kcat of the enzyme's family (EC class / Pfam domain): the strongest signal.
  3. NETWORK propagation — graph label-propagation of log(kcat) from anchors over the PPI + same-pathway
     graph (a prior for enzymes near measured ones; weak, flagged).
  4. Vmax = kcat x abundance (PaxDb) — the in-vivo reaction CAPACITY, more meaningful than kcat alone;
     per pathway, the lowest-Vmax step is the predicted rate-limiting bottleneck.

Reads cell_complete.json (+ paxdb ppm, domains) and kinetics_measured.tsv (assembled on Colab from
SABIO/eHMN/BRENDA). Skips the measured part gracefully; still emits family+propagated estimates.
-> kinetics.json {gene:{kcat, log_kcat, vmax_rel, tier, conditions, family}}, bottlenecks per pathway.
"""
import json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")

def load_anchors(idx):
    """gene -> (log_kcat, source, ph, temp, in_vitro). tsv cols: gene kcat[1/s] km ph temp source in_vitro.
    Reads BOTH our assembled anchors (kinetics_measured.tsv) and the fetched extras (kinetics_measured_extra.tsv:
    CatPred-DB MIT + UniProt CC-BY). Source is preserved so refine_kinetics can exclude CatPred-DB from the
    CatPred held-out validation (CatPred was trained on CatPred-DB)."""
    import csv
    a={}
    for fn in ("kinetics_measured.tsv","kinetics_measured_extra.tsv"):
        f=OUT/fn
        if not f.exists(): continue
        rd=csv.reader(open(f),delimiter="\t"); hdr=[h.lower() for h in next(rd,[])]
        def c(row,*k,_hdr=hdr):
            for kk in k:
                for i,h in enumerate(_hdr):
                    if kk in h and i<len(row): return row[i]
            return ""
        for r in rd:
            g=c(r,"gene","symbol")
            if g not in idx: continue
            try: kc=float(c(r,"kcat"))
            except: continue
            if kc<=0: continue
            invitro = str(c(r,"in_vitro","invitro")).strip() in ("1","true","True")
            prev=a.get(g)
            rec=(math.log10(kc), c(r,"source") or "?", c(r,"ph") or "?", c(r,"temp") or "?", invitro)
            # prefer non-in-vitro (SABIO/eHMN) over BRENDA/CatPred-DB in-vitro; our own file wins ties (seen first)
            if prev is None or (prev[4] and not invitro): a[g]=rec
    return a

def load_ec():
    """gene -> EC number, from uniprot_ec.tsv (accession, gene_primary, ec)."""
    import csv
    m={}; f=H/"uniprot_ec.tsv"
    if f.exists():
        rd=csv.reader(open(f),delimiter="\t"); hdr=[h.lower() for h in next(rd,[])]
        gi=next((i for i,h in enumerate(hdr) if "gene" in h),1); ei=next((i for i,h in enumerate(hdr) if h=="ec number" or h=="ec"),-1)
        for r in rd:
            if ei>=0 and gi<len(r) and ei<len(r) and r[gi] and r[ei]:
                g=r[gi].split()[0]; ec=r[ei].split(";")[0].strip()
                if g and ec and ec[0].isdigit(): m[g]=ec
    return m

def load_pfam():
    import csv
    m={}; f=H/"uniprot_domains.tsv"
    if f.exists():
        rd=csv.reader(open(f),delimiter="\t"); hdr=[h.lower() for h in next(rd,[])]
        gi=next((i for i,h in enumerate(hdr) if "gene" in h),1); pi=next((i for i,h in enumerate(hdr) if "pfam" in h),-1)
        for r in rd:
            if pi>=0 and gi<len(r) and pi<len(r) and r[gi] and r[pi]:
                g=r[gi].split()[0]; d=[x for x in r[pi].replace(",",";").split(";") if x.strip()]
                if g and d: m[g]=d[0].strip()
    return m

def families(G, ec_map, pfam_map):
    """gene -> family key: EC (class.subclass.sub-subclass) > top Pfam domain > cellular process."""
    fam={}
    for i,g in enumerate(G):
        nm=g["name"]; ec=ec_map.get(nm)
        if ec: fam[i]="EC:"+".".join(ec.split(".")[:3])
        elif pfam_map.get(nm): fam[i]="PF:"+pfam_map[nm]
        else: fam[i]="proc:"+g["proc"]
    return fam

def validate(enz, log_anchor, fam, adj, global_prior, seed=0):
    """hold out 25% of anchors, predict via family-median(+propagation), report median fold-error + logR."""
    import random
    anc=list(log_anchor); random.Random(seed).shuffle(anc)
    te=set(anc[:max(1,len(anc)//4)]); tr={i:log_anchor[i] for i in anc if i not in te}
    fv=defaultdict(list)
    for i,lk in tr.items(): fv[fam[i]].append(lk)
    fprior={f:float(np.median(v)) for f,v in fv.items()}
    gp=float(np.median(list(tr.values()))) if tr else global_prior
    errs=[]; preds=[]; trues=[]
    for i in te:
        nb=[tr[j] for j in adj.get(i,[]) if j in tr]
        pred=0.5*fprior.get(fam[i],gp)+0.5*float(np.mean(nb)) if nb else fprior.get(fam[i],gp)
        errs.append(abs(pred-log_anchor[i])); preds.append(pred); trues.append(log_anchor[i])
    if len(errs)<3: return None
    fold=10**float(np.median(errs))
    r=float(np.corrcoef(preds,trues)[0,1]) if np.std(preds)>0 else 0.0
    return dict(n_test=len(te), median_fold_error=round(fold,2), log_pearson_r=round(r,3),
               within_2x=round(sum(1 for e in errs if e<=np.log10(2))/len(errs),2),
               within_10x=round(sum(1 for e in errs if e<=1)/len(errs),2))

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]; N=len(G)
    idx={g["name"]:i for i,g in enumerate(G)}
    generxn=D.get("generxn",{}); ppm={int(k):v for k,v in D.get("ppm",{}).items()}
    anchors=load_anchors(idx)
    ec_map=load_ec(); pfam_map=load_pfam()
    fam=families(G, ec_map, pfam_map)
    # DLKcat: real human kcat per EC number -> ec -> log_kcat
    ec_kcat={}
    ekf=OUT/"ec_kcat.tsv"
    if ekf.exists():
        for l in open(ekf).readlines()[1:]:
            p=l.rstrip("\n").split("\t")
            try: ec_kcat[p[0]]=math.log10(float(p[1]))
            except: pass
    # enzymes = genes with a metabolic reaction (Human-GEM); kinetics is defined for these
    enz=[i for i in range(N) if (str(i) in generxn or i in generxn)]
    log_anchor={idx[g]:v[0] for g,v in anchors.items() if g in idx}
    # (2) family prior from gene-level anchors
    fam_vals=defaultdict(list)
    for i,lk in log_anchor.items(): fam_vals[fam[i]].append(lk)
    fam_prior={f:float(np.median(v)) for f,v in fam_vals.items()}
    allvals=list(log_anchor.values())+list(ec_kcat.values())
    global_prior=float(np.median(allvals)) if allvals else 1.0
    # (3) propagation graph: PPI (within enzymes); label-propagate log_kcat
    adj=defaultdict(list); enzset=set(enz)
    for a,b in D.get("ppi",[]):
        if a in enzset and b in enzset: adj[a].append(b); adj[b].append(a)
    # hierarchical EC medians for backoff (exact EC missing -> use EC subclass / class median)
    ec_levels=defaultdict(list)
    for ec,lk in ec_kcat.items():
        parts=ec.split(".")
        for L in range(1,len(parts)): ec_levels[".".join(parts[:L])].append(lk)
    ec_level_med={k:float(np.median(v)) for k,v in ec_levels.items()}
    def ec_est(ec):
        if not ec: return None,None
        if ec in ec_kcat: return ec_kcat[ec],"EC-measured"
        parts=ec.split(".")
        for L in range(len(parts)-1,0,-1):
            pre=".".join(parts[:L])
            if pre in ec_level_med: return ec_level_med[pre],"EC-class"
        return None,None
    est={}; tier={}
    for i in enz:
        ee,et=ec_est(ec_map.get(G[i]["name"]))
        if i in log_anchor: est[i]=log_anchor[i]; tier[i]="measured"                    # gene-level, with conditions
        elif ee is not None: est[i]=ee; tier[i]=et                                      # real human kcat for the EC (DLKcat), exact or class-backoff
        elif fam[i] in fam_prior: est[i]=fam_prior[fam[i]]; tier[i]="family-prior"
        else: est[i]=global_prior; tier[i]="global-prior"
    for _ in range(15):                                  # harmonic label propagation (anchors fixed)
        new={}
        for i in enz:
            if tier[i] in ("measured","EC-measured","EC-class"): continue
            nb=[est[j] for j in adj.get(i,[])]
            if nb:
                blended=0.6*(fam_prior.get(fam[i],global_prior))+0.4*float(np.mean(nb))
                new[i]=blended
                if tier[i]!="family-prior": tier[i]="network-propagated"
        est.update(new)
    # (4) Vmax = kcat x abundance; pathway bottleneck
    out={}; path_v=defaultdict(list)
    for i in enz:
        g=G[i]; kc=10**est[i]; ab=ppm.get(i)
        vmax = (kc*ab) if ab is not None else None
        rec=dict(kcat_per_s=round(kc,3), log_kcat=round(est[i],3), tier=tier[i], family=fam[i],
                 vmax_rel=(round(math.log10(vmax),3) if vmax else None))
        if g["name"] in anchors:
            _,src,ph,tmp,iv=anchors[g["name"]]; rec.update(source=src, pH=ph, temp=tmp, in_vitro=iv)
        out[g["name"]]=rec
        p=(g.get("path") or "").strip()
        if p and vmax: path_v[p].append((g["name"], math.log10(vmax)))
    bottleneck={p:min(v,key=lambda x:x[1])[0] for p,v in path_v.items() if len(v)>=2}
    tiers=Counter(tier.values())
    val=validate(enz, log_anchor, fam, adj, global_prior) if len(log_anchor)>=12 else None
    payload=dict(kinetics=out, bottlenecks=bottleneck, validation=val,
                 summary=dict(enzymes=len(enz), measured=tiers.get("measured",0),
                              ec_measured=tiers.get("EC-measured",0), ec_class=tiers.get("EC-class",0),
                              family_prior=tiers.get("family-prior",0),
                              network_propagated=tiers.get("network-propagated",0),
                              n_families=len(set(fam[i] for i in enz)),
                              with_abundance=sum(1 for i in enz if i in ppm)))
    json.dump(payload, open(OUT/"kinetics.json","w"))
    print(f"kinetics: {len(enz)} enzymes | measured {tiers.get('measured',0)}, EC-measured(DLKcat) "
          f"{tiers.get('EC-measured',0)}, EC-class {tiers.get('EC-class',0)}, family {tiers.get('family-prior',0)}, propagated {tiers.get('network-propagated',0)} "
          f"| {payload['summary']['with_abundance']} with abundance -> Vmax | {len(bottleneck)} bottlenecks")
    if val: print(f"  imputation validation (held-out kcat): median fold-error {val['median_fold_error']}x, "
                  f"within-2x {val['within_2x']:.0%}, within-10x {val['within_10x']:.0%}, log-R {val['log_pearson_r']} (n={val['n_test']})")
    print(f"  families: {payload['summary']['n_families']}")
    if not anchors: print("  (no measured anchors present -> all estimates are family/global priors; add kinetics_measured.tsv on Colab)")

if __name__=="__main__":
    main()
