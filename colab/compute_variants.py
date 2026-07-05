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
            sig=(row.get("ClinicalSignificance") or "").lower(); nm=row.get("Name","")
            is_missense = "missense" in nm.lower() or ("p." in nm and ">" in nm)
            # a ClinVar row can list several genes (e.g. "BRCA1;NBR2"); credit each modelled gene
            for g in (row.get("GeneSymbol","") or "").split(";"):
                g=g.strip()
                if g not in names: continue
                r=out[g]
                if "conflicting" in sig: r["vus"]+=1
                elif "pathogenic" in sig:
                    r["path"]+=1
                    if is_missense: r["missense_path"]+=1
                    if len(r["examples"])<5 and nm: r["examples"].append(nm[:60])
                elif "benign" in sig: r["benign"]+=1
                elif "uncertain" in sig: r["vus"]+=1
                r["stars"]=max(r["stars"], STAR.get((row.get("ReviewStatus") or "").lower(),0))
    return dict(out)

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> variants skipped"); return
    D=json.load(open(cc)); G=D["genes"]; names=set(g["name"] for g in G)
    idx={g["name"]:i for i,g in enumerate(G)}
    drugs=D.get("drugs",{}); otdis=D.get("otdis",{})     # DGIdb interactions (by index) + OpenTargets druggability (by name)
    def druggable_of(nm):
        """(is_druggable, drug_names) — a DGIdb drug interaction OR an OpenTargets 'druggable' flag."""
        dl=drugs.get(str(idx[nm])) if nm in idx else None
        dn=[d.get("d") for d in (dl or []) if d.get("d")]
        return (bool(dn) or bool(otdis.get(nm,{}).get("druggable"))), dn[:5]
    cv=load_clinvar(names)
    flux=(json.load(open(OUT/"flux.json")).get("flux",{}) if (OUT/"flux.json").exists() else {})
    gene2rxn={}                                          # gene -> (reaction id, tier) it carries most flux through
    for rid,r in flux.items():
        for gg in r.get("genes",[]):
            if gg not in gene2rxn or abs(r.get("v",0))>abs(flux.get(gene2rxn[gg][0],{}).get("v",0)):
                gene2rxn[gg]=(rid, r.get("tier"))
    out={}; metabolic=[]; understudied=[]; validated_targets=[]; unmet_need=[]
    for g in G:
        nm=g["name"]; c=cv.get(nm)
        loeuf=g.get("loeuf",-1); ess=g.get("ess",0); pubs=g.get("pubs",0); dark=g.get("dark",0)
        constrained = 0<=loeuf<0.35                     # top LoF-intolerant decile-ish
        drug_ok, drug_names = druggable_of(nm)
        rec=dict(loeuf=loeuf, essential=bool(ess), pubs=pubs, druggable=drug_ok)
        if drug_names: rec["drugs"]=drug_names
        if c: rec.update(clinvar_pathogenic=c["path"], clinvar_missense_pathogenic=c["missense_path"],
                         clinvar_benign=c["benign"], clinvar_vus=c["vus"],
                         review_stars=c["stars"], example_pathogenic=c["examples"])
        npath=(c or {}).get("path",0)
        # vulnerability tier: constraint + essentiality + known pathogenic burden
        score=(2 if constrained else 0)+(2 if ess else 0)+(1 if npath>=1 else 0)+(1 if npath>=10 else 0)
        rec["vulnerability_tier"]="high" if score>=4 else ("medium" if score>=2 else "low")
        if npath>=1 and c and c["stars"]>=2: rec["known_disease_gene"]=True
        out[nm]=rec
        if npath>=1 and nm in gene2rxn:                 # enzyme with disease variants that carries flux -> IEM candidate
            rid,tier=gene2rxn[nm]
            metabolic.append(dict(gene=nm, pathogenic=npath, reaction=rid, flux=flux.get(rid,{}).get("v"),
                                  flux_tier=tier))
        # should matter (constrained+essential) but under-explored: few/no pathogenic variants AND under-studied.
        # (ClinVar now covers ~16k genes, so 'zero pathogenic' alone is too strict -- allow a small burden.)
        if constrained and ess and npath<=1 and pubs<40:
            understudied.append(dict(gene=nm, loeuf=loeuf, pubs=pubs, clinvar_pathogenic=npath, dark=bool(dark)))
        # CROSS-LAYER (genetics x pharmacology): a confident disease gene that is ALSO druggable = a drug target
        # with human genetic validation. If it is NOT druggable but constrained/essential = an unmet-need target.
        if rec.get("known_disease_gene") and drug_ok:
            validated_targets.append(dict(gene=nm, pathogenic=npath, drugs=drug_names, loeuf=loeuf,
                                          essential=bool(ess), review_stars=(c or {}).get("stars",0)))
        elif (npath>=1 or ess) and constrained and not drug_ok:
            unmet_need.append(dict(gene=nm, pathogenic=npath, essential=bool(ess), loeuf=loeuf, pubs=pubs, dark=bool(dark)))
    # rank understudied: dark + most-constrained + least-studied first (Play E: highest-value unknowns)
    understudied.sort(key=lambda x:(not x["dark"], x["loeuf"], x["pubs"]))
    validated_targets.sort(key=lambda x:(-x["pathogenic"], x["loeuf"]))       # strongest genetic evidence first
    unmet_need.sort(key=lambda x:(not x["dark"], x["loeuf"], x["pubs"]))      # most-constrained, least-studied first
    n_cv=sum(1 for r in out.values() if "clinvar_pathogenic" in r)
    n_path=sum(1 for r in out.values() if r.get("clinvar_pathogenic",0)>0)
    payload=dict(variants=out, metabolic_variant_nodes=metabolic[:200],
                 predicted_vulnerable_understudied=understudied[:150],
                 genetically_validated_targets=validated_targets[:200],
                 unmet_need_targets=unmet_need[:200],
                 summary=dict(genes=len(out), with_clinvar=n_cv, genes_with_pathogenic=n_path,
                              known_disease_genes=sum(1 for r in out.values() if r.get("known_disease_gene")),
                              druggable_genes=sum(1 for r in out.values() if r.get("druggable")),
                              metabolic_variant_nodes=len(metabolic),
                              understudied_candidates=len(understudied),
                              understudied_dark=sum(1 for u in understudied if u["dark"]),
                              genetically_validated_targets=len(validated_targets),
                              unmet_need_targets=len(unmet_need),
                              high_vulnerability=sum(1 for r in out.values() if r["vulnerability_tier"]=="high")))
    json.dump(payload, open(OUT/"variants.json","w"))
    s=payload["summary"]
    print(f"variants: {s['genes']} genes | {s['with_clinvar']} with ClinVar ({s['genes_with_pathogenic']} carry pathogenic, "
          f"{s['known_disease_genes']} high-confidence disease genes) | {s['high_vulnerability']} high-vulnerability | "
          f"{s['metabolic_variant_nodes']} metabolic-variant (IEM) nodes | {s['understudied_candidates']} understudied "
          f"candidates ({s['understudied_dark']} also DARK -> highest-value unknowns)")
    print(f"  genetics x pharmacology: {s['genetically_validated_targets']} genetically-VALIDATED druggable targets "
          f"(disease variants + a drug), {s['unmet_need_targets']} UNMET-NEED targets (disease/essential + constrained + undrugged) "
          f"| {s['druggable_genes']} druggable genes total")

if __name__=="__main__":
    main()
