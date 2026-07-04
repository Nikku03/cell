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
    """gene -> (log_kcat, source, ph, temp, in_vitro). tsv cols: gene kcat[1/s] km ph temp source in_vitro"""
    a={}; f=OUT/"kinetics_measured.tsv"
    if not f.exists(): return a
    import csv
    rd=csv.reader(open(f),delimiter="\t"); hdr=[h.lower() for h in next(rd,[])]
    def c(row,*k):
        for kk in k:
            for i,h in enumerate(hdr):
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
        # prefer non-in-vitro (SABIO/eHMN) over BRENDA in-vitro
        if prev is None or (prev[4] and not invitro): a[g]=rec
    return a

def families(G, generxn, dom):
    """gene -> family key (EC class from Human-GEM reaction, else top Pfam domain, else process)."""
    fam={}
    for i,g in enumerate(G):
        rx=generxn.get(str(i)) or generxn.get(i)
        ec=None
        if rx:
            for r in rx:
                m=[t for t in str(r).replace(";"," ").split() if t.count(".")>=2 and t.replace(".","").isdigit()]
                if m: ec=".".join(m[0].split(".")[:2]); break   # EC class.subclass
        d=(dom.get(g["name"]) or [None])[0] if dom else None
        fam[i]= ("EC:"+ec) if ec else (("PF:"+d) if d else ("proc:"+g["proc"]))
    return fam

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]; N=len(G)
    idx={g["name"]:i for i,g in enumerate(G)}
    generxn=D.get("generxn",{}); ppm={int(k):v for k,v in D.get("ppm",{}).items()}
    dom={}
    df=OUT/"domain_function.json"        # reuse if present; else Pfam from build not needed here
    anchors=load_anchors(idx)
    fam=families(G, generxn, {})
    # enzymes = genes with a metabolic reaction (Human-GEM); kinetics is defined for these
    enz=[i for i in range(N) if (str(i) in generxn or i in generxn)]
    log_anchor={idx[g]:v[0] for g,v in anchors.items() if g in idx}
    # (2) family prior: median log_kcat per family from anchors
    fam_vals=defaultdict(list)
    for i,lk in log_anchor.items(): fam_vals[fam[i]].append(lk)
    fam_prior={f:float(np.median(v)) for f,v in fam_vals.items()}
    global_prior=float(np.median(list(log_anchor.values()))) if log_anchor else 1.0  # ~10 1/s default
    # (3) propagation graph: PPI (within enzymes) + same-pathway; label-propagate log_kcat
    adj=defaultdict(list); enzset=set(enz)
    for a,b in D.get("ppi",[]):
        if a in enzset and b in enzset: adj[a].append(b); adj[b].append(a)
    est={}; tier={}
    for i in enz:
        if i in log_anchor: est[i]=log_anchor[i]; tier[i]="measured"
        elif fam[i] in fam_prior: est[i]=fam_prior[fam[i]]; tier[i]="family-prior"
        else: est[i]=global_prior; tier[i]="global-prior"
    for _ in range(15):                                  # harmonic label propagation (anchors fixed)
        new={}
        for i in enz:
            if tier[i]=="measured": continue
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
    payload=dict(kinetics=out, bottlenecks=bottleneck,
                 summary=dict(enzymes=len(enz), measured=tiers.get("measured",0),
                              family_prior=tiers.get("family-prior",0),
                              network_propagated=tiers.get("network-propagated",0),
                              with_abundance=sum(1 for i in enz if i in ppm)))
    json.dump(payload, open(OUT/"kinetics.json","w"))
    print(f"kinetics: {len(enz)} enzymes | measured {tiers.get('measured',0)}, "
          f"family-prior {tiers.get('family-prior',0)}, network-propagated {tiers.get('network-propagated',0)} "
          f"| {payload['summary']['with_abundance']} with abundance -> Vmax | {len(bottleneck)} pathway bottlenecks")
    if not anchors: print("  (no measured anchors present -> all estimates are family/global priors; add kinetics_measured.tsv on Colab)")

if __name__=="__main__":
    main()
