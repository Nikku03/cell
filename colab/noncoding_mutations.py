"""Account for NON-CODING disease mutations. Categorize each gene's ClinVar pathogenic
variants by ELEMENT: coding/ORF vs splice vs regulatory (promoter/UTR/upstream/intron).
Regulatory pathogenic variants implicate the regulatory layer -- a promoter mutation
disrupts a TF binding site, so the measured TF->gene edge is disease-relevant."""
import json, urllib.request, warnings, socket, time
socket.setdefaulttimeout(35); warnings.filterwarnings("ignore")
from collections import defaultdict, Counter
from pathlib import Path
OUT=Path("outputs/orphan")
GENES=["TERT","HBB","F9","LDLR","HBA2","PAH","RB1","BRCA1","MLH1","CFTR","TP53","GATA1",
 "FOXP3","HNF4A","RET","VHL","PTEN","SOX9","PAX6","TBX5","ATM","MSH2"]
CODING={"missense_variant","stop_gained","stop_lost","start_lost","frameshift_variant",
 "inframe_insertion","inframe_deletion","protein_altering_variant","coding_sequence_variant"}
SPLICE={"splice_donor_variant","splice_acceptor_variant","splice_region_variant",
 "splice_donor_5th_base_variant","splice_polypyrimidine_tract_variant","splice_donor_region_variant"}
REG={"5_prime_UTR_variant","3_prime_UTR_variant","upstream_gene_variant","downstream_gene_variant",
 "regulatory_region_variant","intron_variant","non_coding_transcript_exon_variant","non_coding_transcript_variant","intergenic_variant"}
def cat(c):
    if c in CODING: return "coding_ORF"
    if c in SPLICE: return "splice"
    if c in REG: return "regulatory_noncoding"
    return "other"
def gq(sym):
    q='{gene(gene_symbol:"%s",reference_genome:GRCh38){clinvar_variants{clinical_significance major_consequence}}}'%sym
    req=urllib.request.Request("https://gnomad.broadinstitute.org/api",data=json.dumps({"query":q}).encode(),headers={"Content-Type":"application/json"})
    for _ in range(2):
        try: return json.load(urllib.request.urlopen(req,timeout=30))
        except Exception: time.sleep(2)
    return None
tot=Counter(); per=[]
regdetail=Counter()
for sym in GENES:
    d=gq(sym); g=(d or {}).get("data",{}).get("gene")
    if not g: print(f"{sym}: skip"); continue
    c=Counter()
    for v in g["clinvar_variants"]:
        sig=(v.get("clinical_significance") or "").lower()
        if "pathogenic" not in sig or "conflicting" in sig: continue
        cons=v.get("major_consequence") or ""; k=cat(cons); c[k]+=1; tot[k]+=1
        if k=="regulatory_noncoding": regdetail[cons]+=1
    n=sum(c.values())
    per.append(dict(gene=sym,total=n,coding=c["coding_ORF"],splice=c["splice"],regulatory=c["regulatory_noncoding"],other=c["other"]))
    print(f"  {sym:7s} path {n:4d}: coding {c['coding_ORF']:4d}  splice {c['splice']:3d}  regulatory/noncoding {c['regulatory_noncoding']:3d}",flush=True)
    time.sleep(0.3)
T=sum(tot.values())
print(f"\n=== disease mutations by ELEMENT (pathogenic, {T} total) ===")
for k in ["coding_ORF","splice","regulatory_noncoding","other"]:
    print(f"  {k:22s}: {tot[k]:5d} ({100*tot[k]/max(T,1):.1f}%)")
print(f"\n  regulatory/non-coding breakdown: {dict(regdetail)}")
print(f"  genes with most regulatory/non-coding pathogenic variants:")
for r in sorted(per,key=lambda x:-x['regulatory'])[:6]:
    print(f"    {r['gene']:7s} regulatory {r['regulatory']} / total {r['total']}")
json.dump(dict(by_element={k:tot[k] for k in tot},total=T,
    frac_noncoding=round((tot['splice']+tot['regulatory_noncoding'])/max(T,1),3),
    regulatory_breakdown=dict(regdetail),per_gene=per),open(OUT/"noncoding_mutations.json","w"),indent=2)
print("\nwrote noncoding_mutations.json")
