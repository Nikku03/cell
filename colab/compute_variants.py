"""VARIANTS (Problem 6) — genetic-variation -> effect, on a commercial-safe data base.

Per gene we integrate:
  - ClinVar (public domain): counts of curated Pathogenic/Likely-pathogenic vs Benign vs VUS variants,
    with review-status confidence, and example pathogenic variants.
  - gnomAD constraint (LOEUF, already in the model): how intolerant the gene is to loss-of-function.
  - DepMap essentiality (already in the model).
-> a per-gene VARIANT VULNERABILITY tier, and two cross-layer products:
  - metabolic-variant nodes: enzymes with pathogenic variants that are also flux bottlenecks
    (candidate inborn-errors-of-metabolism mechanisms);
  - predicted-vulnerable-but-understudied: highly constrained + essential genes with FEW known variants and
    FEW publications -> disease genes whose variants likely matter but haven't been catalogued yet
    (constraint x essentiality x literature x ClinVar = a signal none of them give alone).

(AlphaMissense would add proteome-wide per-residue missense effect but is CC BY-NC-SA / non-commercial;
 left as an opt-in research enhancement via AM_GENE_URL, not part of the default build.)

Reads cell_complete.json (+ optional flux.json). Fetches ClinVar. Skips gracefully. -> variants.json
"""
import json, os, gzip, urllib.request
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
CLINVAR_URL=os.environ.get("CLINVAR_URL","https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz")

def load_clinvar(names):
    """gene -> {path, benign, vus, missense_path, examples[], stars}. GRCh38 rows only."""
    f=H/"variant_summary.txt.gz"
    if not (f.exists() and f.stat().st_size>1e6):
        try:
            print("  fetching ClinVar variant_summary (~250 MB) ..."); urllib.request.urlretrieve(CLINVAR_URL, f)
        except Exception as e:
            print("  ClinVar fetch failed:",repr(e)[:100],"-> variant layer uses constraint/essentiality only"); return {}
    STAR={"practice guideline":4,"reviewed by expert panel":3,"criteria provided, multiple submitters, no conflicts":2,
          "criteria provided, single submitter":1,"criteria provided, conflicting interpretations":1}
    out=defaultdict(lambda: dict(path=0,benign=0,vus=0,missense_path=0,examples=[],stars=0))
    hdr=None
    with gzip.open(f,"rt",errors="ignore") as fh:
        for l in fh:
            if l.startswith("#") or l.startswith("AlleleID"):
                hdr=l.lstrip("#").rstrip("\n").split("\t"); continue
            if hdr is None: continue
            p=l.rstrip("\n").split("\t")
            if len(p)<len(hdr): continue
            row=dict(zip(hdr,p))
            if row.get("Assembly")!="GRCh38": continue
            g=row.get("GeneSymbol","")
            if g not in names: continue
            sig=(row.get("ClinicalSignificance") or "").lower(); typ=(row.get("Type") or "").lower()
            r=out[g]
            if "conflicting" in sig: r["vus"]+=1
            elif "pathogenic" in sig:
                r["path"]+=1
                if "single nucleotide" in typ or "missense" in (row.get("Name","").lower()): r["missense_path"]+=1
                if len(r["examples"])<5 and row.get("Name"): r["examples"].append(row["Name"][:60])
            elif "benign" in sig: r["benign"]+=1
            elif "uncertain" in sig: r["vus"]+=1
            r["stars"]=max(r["stars"], STAR.get((row.get("ReviewStatus") or "").lower(),0))
    return dict(out)

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> variants skipped"); return
    D=json.load(open(cc)); G=D["genes"]; names=set(g["name"] for g in G)
    cv=load_clinvar(names)
    flux=(json.load(open(OUT/"flux.json")).get("flux",{}) if (OUT/"flux.json").exists() else {})
    flux_genes=set(g for r in flux.values() for g in r.get("genes",[]))
    out={}; metabolic=[]; understudied=[]
    for g in G:
        nm=g["name"]; c=cv.get(nm)
        loeuf=g.get("loeuf",-1); ess=g.get("ess",0); pubs=g.get("pubs",0)
        constrained = 0<=loeuf<0.35                     # top LoF-intolerant decile-ish
        rec=dict(loeuf=loeuf, essential=bool(ess), pubs=pubs)
        if c: rec.update(clinvar_pathogenic=c["path"], clinvar_benign=c["benign"], clinvar_vus=c["vus"],
                         review_stars=c["stars"], example_pathogenic=c["examples"])
        npath=(c or {}).get("path",0)
        # vulnerability tier: constraint + essentiality + known pathogenic burden
        score=(2 if constrained else 0)+(2 if ess else 0)+(1 if npath>=1 else 0)+(1 if npath>=10 else 0)
        rec["vulnerability_tier"]="high" if score>=4 else ("medium" if score>=2 else "low")
        out[nm]=rec
        if npath>=1 and nm in flux_genes:               # enzyme with disease variants + carries flux
            metabolic.append(dict(gene=nm, pathogenic=npath, flux_enzyme=True))
        if constrained and ess and npath==0 and pubs<20:  # should matter, not catalogued, understudied
            understudied.append(dict(gene=nm, loeuf=loeuf, pubs=pubs))
    understudied.sort(key=lambda x:(x["loeuf"], x["pubs"]))
    n_cv=sum(1 for r in out.values() if "clinvar_pathogenic" in r)
    n_path=sum(1 for r in out.values() if r.get("clinvar_pathogenic",0)>0)
    payload=dict(variants=out, metabolic_variant_nodes=metabolic[:200],
                 predicted_vulnerable_understudied=understudied[:150],
                 summary=dict(genes=len(out), with_clinvar=n_cv, genes_with_pathogenic=n_path,
                              metabolic_variant_nodes=len(metabolic),
                              understudied_candidates=len(understudied),
                              high_vulnerability=sum(1 for r in out.values() if r["vulnerability_tier"]=="high")))
    json.dump(payload, open(OUT/"variants.json","w"))
    s=payload["summary"]
    print(f"variants: {s['genes']} genes | {s['with_clinvar']} with ClinVar ({s['genes_with_pathogenic']} carry pathogenic "
          f"variants) | {s['high_vulnerability']} high-vulnerability | {s['metabolic_variant_nodes']} metabolic-variant nodes | "
          f"{s['understudied_candidates']} predicted-vulnerable-but-understudied")

if __name__=="__main__":
    main()
