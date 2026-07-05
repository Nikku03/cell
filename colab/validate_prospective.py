"""PROSPECTIVE VALIDATION — do the model's DARK-GENE hypotheses hold up against future annotation?

The flagship novelty is darkfn: for each unannotated ('dark') gene the model predicts a function by
guilt-by-association with its MEASURED neighbors (Perturb-seq / co-essentiality / PPI / structure), i.e.
"gene X works with genes {ev}". This is the honest test of "does it produce RIGHT hypotheses":

  For dark genes that have since GAINED a GO biological-process annotation (data the model did NOT have),
  does X share >=1 GO BP term with the neighbors {ev} the model bet on -- more often than random gene sets?

  hit_rate(model)  vs  hit_rate(random same-size gene sets)  ->  ENRICHMENT over chance.

Direct-GO-annotation predictions are EXCLUDED (those aren't predictions). Stratified by the model's own
high/low confidence. This is a genuine forward test: the truth (current GO for a formerly-dark gene) is
information the model did not use. Caveat: some GO annotations are themselves guilt-by-association, which
would inflate the rate -- so the RANDOM baseline (same annotation process) is the honest yardstick, not the
absolute rate. -> prospective_validation.json
"""
import json, os, csv, re, random, urllib.request
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

def fetch_go_bp():
    """gene symbol -> set of current GO biological-process IDs (UniProt reviewed human)."""
    f=H/"uniprot_go_bp.tsv"
    if not (f.exists() and f.stat().st_size>100000):
        url=("https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true"
             "&fields=gene_primary,go_p&format=tsv")
        print("  fetching current GO biological-process annotations (UniProt) ..."); urllib.request.urlretrieve(url,f)
    go={}
    for r in csv.DictReader(open(f), delimiter="\t"):
        g=(r.get("Gene Names (primary)") or "").strip()
        if not g: continue
        ids=set(re.findall(r"GO:\d+", r.get("Gene Ontology (biological process)") or ""))
        if ids: go[g]=ids
    return go

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> prospective validation skipped"); return
    D=json.load(open(cc)); G=D["genes"]
    darkfn={int(k):v for k,v in D.get("darkfn",{}).items()}
    if not darkfn: print("no darkfn predictions in cell_complete -> skipped"); return
    go=fetch_go_bp()
    if not go: print("GO annotations unavailable -> skipped"); return
    pool=[g for g in go]                                     # genes with a current GO BP annotation
    def hit(genes, xgo):
        u=set()
        for e in genes:
            u|=go.get(e,set())
        return len(u & xgo)>0
    tests=[]
    for i,v in darkfn.items():
        if "GO(annotation)" in v.get("src",""): continue     # direct annotation, not a prediction
        if i>=len(G): continue
        name=G[i]["name"]; xgo=go.get(name)
        ev=[e for e in v.get("ev",[]) if e in go and e!=name]
        if not xgo or len(ev)<2: continue                    # need: X now annotated + >=2 neighbors with GO
        tests.append((name, ev, xgo, v.get("conf","low")))
    if len(tests)<20: print(f"prospective: only {len(tests)} testable dark genes -> too few; skipped"); return
    rnd=random.Random(0); R=20
    model=[]; rand=[]; by={"high":[],"low":[]}
    for name,ev,xgo,conf in tests:
        h=hit(ev,xgo); model.append(h); by[conf].append(h)
        for _ in range(R): rand.append(hit(rnd.sample(pool,len(ev)), xgo))
    def rate(x): return sum(x)/len(x) if x else 0.0
    mrate=rate(model); rrate=rate(rand); enr=mrate/rrate if rrate>0 else None
    # binomial-ish z vs the random rate
    import math
    n=len(model); se=math.sqrt(rrate*(1-rrate)/n) if n and 0<rrate<1 else 0
    z=(mrate-rrate)/se if se>0 else None
    res=dict(n_testable=n, model_hit_rate=round(mrate,3), random_baseline_rate=round(rrate,3),
             enrichment_over_chance=round(enr,2) if enr else None,
             high_conf=dict(n=len(by["high"]), hit_rate=round(rate(by["high"]),3)),
             low_conf=dict(n=len(by["low"]), hit_rate=round(rate(by["low"]),3)),
             z_vs_random=round(z,2) if z else None,
             note="dark-gene function predictions vs current GO BP of formerly-dark genes; random same-size "
                  "gene-set null. Enrichment>1 with z>~2 => hypotheses beat chance.")
    json.dump(res, open(OUT/"prospective_validation.json","w"))
    print(f"PROSPECTIVE VALIDATION — dark-gene function vs current GO ({n} testable formerly-dark genes)")
    print(f"  model hit rate {res['model_hit_rate']} vs random baseline {res['random_baseline_rate']} "
          f"-> {res['enrichment_over_chance']}x enrichment (z={res['z_vs_random']})")
    print(f"  by confidence: high {res['high_conf']['hit_rate']} (n={res['high_conf']['n']}) | "
          f"low {res['low_conf']['hit_rate']} (n={res['low_conf']['n']})")
    verdict=("HYPOTHESES BEAT CHANCE — the model's dark-gene predictions are enriched for genes that share a "
             "process with the predicted partners in FUTURE annotation" if (enr and enr>1.2 and (z or 0)>2)
             else "no clear enrichment over chance (predictions ~ random on this test)")
    print(f"  >>> {verdict}")

if __name__=="__main__":
    main()
