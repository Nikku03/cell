"""CALIBRATION / VALIDATION harness — measure how ACCURATE each answer type actually is.

Turns "high/medium/low" into a real number: hold out gold-standard labels and measure accuracy, so
every query answer can carry "predictions of this type are X% accurate" (roadmap #2 + #4). Runs on the
assembled model against its own gold standards. -> calibration.json {answer_type:{accuracy,n,method}}
"""
import json, random
from collections import Counter, defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]; N=len(G)
    idx={g["name"]:i for i,g in enumerate(G)}
    path=[(g.get("path") or "").strip() for g in G]; proc=[g["proc"] for g in G]
    codep={int(k):[j for j,r in v] for k,v in D.get("codep",{}).items()}
    ppiadj=defaultdict(list)
    for a,b in D.get("ppi",[]): ppiadj[a].append(b); ppiadj[b].append(a)
    coexpr={int(k):[e[0] for e in v] for k,v in D.get("coexpr",{}).items()}
    cal={}

    # (1) FUNCTION-PREDICTION accuracy — hold out annotated genes, predict from neighbors (the darkfn method)
    annotated=[i for i in range(N) if path[i] and path[i]!="other"]
    random.Random(0).shuffle(annotated); sample=annotated[:2000]
    hit_path=hit_proc=tot=0; by_conf={"high":[0,0],"low":[0,0]}
    for i in sample:
        neigh=list(coexpr.get(i,[]))[:8]+codep.get(i,[])[:8]+ppiadj.get(i,[])[:6]
        neigh=[j for j in neigh if j!=i]
        votes=Counter()
        for j in neigh:
            p=path[j] or proc[j]
            if p and p!="other": votes[p]+=1
        if not votes: continue
        pred,cnt=votes.most_common(1)[0]; tot+=1
        conf="high" if cnt>=3 else "low"
        ok = pred==path[i]
        hit_path+=ok
        by_conf[conf][0]+= ok; by_conf[conf][1]+=1
    # proc-level: predicted pathway's genes share the held-out gene's broad process
    pth2proc={}
    for i in range(N):
        if path[i]: pth2proc.setdefault(path[i],Counter())[proc[i]]+=1
    hit_proc=0
    for i in sample:
        neigh=[j for j in (list(coexpr.get(i,[]))[:8]+codep.get(i,[])[:8]+ppiadj.get(i,[])[:6]) if j!=i]
        votes=Counter(p for p in (path[j] or proc[j] for j in neigh) if p and p!="other")
        if not votes: continue
        pred=votes.most_common(1)[0][0]
        predproc=pth2proc.get(pred,Counter()).most_common(1)[0][0] if pred in pth2proc else pred
        hit_proc += (pred==path[i]) or (predproc==proc[i])
    cal["dark_gene_function"]=dict(
        exact_pathway=round(hit_path/max(tot,1),3), same_process=round(hit_proc/max(tot,1),3),
        high_conf_accuracy=round(by_conf["high"][0]/max(by_conf["high"][1],1),3),
        low_conf_accuracy=round(by_conf["low"][0]/max(by_conf["low"][1],1),3),
        n=tot, method="held-out annotated genes, predicted from co-expr+co-essential+PPI neighbors")

    # (2) CONVERGENCE precision — fraction of known-complex pairs recovered as convergent (positive control)
    conv=OUT/"convergence.json"
    if conv.exists():
        c=json.load(open(conv))
        kc=c.get("n_known_control",0); nov=c.get("n_novel",0)
        cal["novel_links"]=dict(known_complex_recovered=kc, novel_proposed=nov,
                                method="known same-complex pairs that pass the >=2-lens convergence bar (positive control)",
                                note="novel links are UNVALIDATED hypotheses; recovery of known complexes is the confidence proxy")

    # (3) ESSENTIALITY — measured vs Model-1 predicted coverage (Model-1 AUC is reported at train time)
    meas=sum(1 for g in G if g.get("ess_src")=="measured")
    cal["essentiality"]=dict(measured_fraction=round(meas/N,3), source="DepMap measured; Model-1 fills the rest",
                             note="measured essentiality is fact (confidence=high); predicted carries Model-1 AUC ~0.97")

    json.dump(cal, open(OUT/"calibration.json","w"))
    print("calibration:")
    f=cal["dark_gene_function"]
    print(f"  dark-gene function: exact-pathway {f['exact_pathway']:.0%}, same-process {f['same_process']:.0%} "
          f"(high-conf {f['high_conf_accuracy']:.0%} vs low-conf {f['low_conf_accuracy']:.0%}), n={f['n']}")
    if "novel_links" in cal:
        print(f"  novel links: {cal['novel_links']['known_complex_recovered']} known complexes recovered "
              f"(positive control) | {cal['novel_links']['novel_proposed']} novel proposed (unvalidated)")
    print(f"  essentiality: {cal['essentiality']['measured_fraction']:.0%} measured")

if __name__=="__main__":
    main()
