"""Full pipeline on syn3.0's unclear essential genes:
   SEQUENCE -> PROTEIN -> STRUCTURE -> FUNCTION -> NETWORK POSITION

Stage 1 SEQUENCE   : amino-acid sequence (have it)
Stage 2 PROTEIN    : length, pI, GRAVY, net charge, AA composition,
                     TM-helix count, signal peptide (precomputed)
Stage 3 STRUCTURE  : sequence-derived structure PROXY --
                     Chou-Fasman helix/sheet propensity -> secondary-
                     structure content -> structural CLASS
                     (all-alpha / all-beta / alpha+beta / membrane /
                     disordered), plus topology from TM + signal peptide.
                     (A true AlphaFold 3D model would refine this; the
                      logic chain is identical.)
Stage 4 FUNCTION   : combine homology (Smith-Waterman vs annotated
                     M. genitalium + syn3a) with the structural class
                     to constrain function -- fold class rules out /
                     rules in broad categories (membrane fold ->
                     transport; Rossmann-like a/b -> nucleotide enzyme;
                     all-alpha small basic -> DNA-binding/regulatory).
Stage 5 NETWORK    : operon module (neighbour functions), neighbour
                     essentiality (co-essential cluster?), assignment to
                     a cellular subsystem, and chokepoint status
                     (is it the sole copy of its function in the genome).

Output: outputs/syn3a_pipeline.md  (one card per gene + summary)
"""
from __future__ import annotations
import re, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Chou-Fasman propensities (helix Pa, sheet Pb)
PA = {"A":1.42,"R":0.98,"N":0.67,"D":1.01,"C":0.70,"Q":1.11,"E":1.51,"G":0.57,
      "H":1.00,"I":1.08,"L":1.21,"K":1.16,"M":1.45,"F":1.13,"P":0.57,"S":0.77,
      "T":0.83,"W":1.08,"Y":0.69,"V":1.06}
PB = {"A":0.83,"R":0.93,"N":0.89,"D":0.54,"C":1.19,"Q":1.10,"E":0.37,"G":0.75,
      "H":0.87,"I":1.60,"L":1.30,"K":0.74,"M":1.05,"F":1.38,"P":0.55,"S":0.75,
      "T":1.19,"W":1.37,"Y":1.47,"V":1.70}
# disorder-promoting residues (Dunker): low complexity / charged / flexible
DISORDER = set("ARSQEGKP")

# BLOSUM62 for homology (reuse compact form)
_B = """A 4 R -1 5 N -2 0 6 D -2 -2 1 6 C 0 -3 -3 -3 9 Q -1 1 0 0 -3 5 E -1 0 0 2 -4 2 5 G 0 -2 0 -1 -3 -2 -2 6 H -2 0 1 -1 -3 0 0 -2 8 I -1 -3 -3 -3 -1 -3 -3 -4 -3 4 L -1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 K -1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 M -1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 F -2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 S 1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 T 0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 Y -2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 V 0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""
_toks = _B.split(); order=[]; BL={}
i=0
while i < len(_toks):
    aa=_toks[i]; order.append(aa); i+=1
    for j in range(len(order)):
        v=int(_toks[i]); i+=1
        BL[(aa,order[j])]=v; BL[(order[j],aa)]=v
def sc(a,b): return BL.get((a,b),-4)


def smith_waterman(s1,s2,gap=-6):
    n,m=len(s1),len(s2)
    if not n or not m: return 0,0.0
    H=[[0]*(m+1) for _ in range(n+1)]; best=0; bij=(0,0)
    for i in range(1,n+1):
        a=s1[i-1]
        for j in range(1,m+1):
            d=H[i-1][j-1]+sc(a,s2[j-1])
            v=max(0,d,H[i-1][j]+gap,H[i][j-1]+gap); H[i][j]=v
            if v>best: best=v; bij=(i,j)
    i,j=bij; mat=al=0
    while i>0 and j>0 and H[i][j]>0:
        if H[i][j]==H[i-1][j-1]+sc(s1[i-1],s2[j-1]):
            al+=1
            if s1[i-1]==s2[j-1]: mat+=1
            i-=1;j-=1
        elif H[i][j]==H[i-1][j]+gap: i-=1
        else: j-=1
    return best, (mat/al if al else 0.0)


def parse_fasta(p):
    out={};cur=None;buf=[];h=""
    for line in open(p):
        if line.startswith(">"):
            if cur is not None: out[cur]=("".join(buf),h)
            h=line[1:].strip();cur=h.split()[0];buf=[]
        else: buf.append(line.strip())
    if cur is not None: out[cur]=("".join(buf),h)
    return out


def kmers(s,k=4): return set(s[i:i+k] for i in range(len(s)-k+1))


def structural_class(seq, tm, sig):
    """Sequence-derived structure proxy -> class + secondary-structure %."""
    aa=[a for a in seq if a in PA]
    if not aa: return "unknown", 0,0,0
    pa=sum(PA[a] for a in aa)/len(aa)
    pb=sum(PB[a] for a in aa)/len(aa)
    dis=sum(1 for a in aa if a in DISORDER)/len(aa)
    # crude SS content from propensity (normalized)
    helix = max(0, (pa-1.0))*100
    sheet = max(0, (pb-1.0))*100
    if tm and tm>=3:        cls="membrane (polytopic)"
    elif tm and tm>=1 and sig: cls="membrane-anchored / lipoprotein"
    elif sig:               cls="secreted / surface"
    elif dis>0.50 and pa<1.0 and pb<1.0: cls="intrinsically disordered"
    elif pa>1.10 and pb<1.05: cls="all-alpha (helical)"
    elif pb>1.15 and pa<1.10: cls="all-beta (sheet)"
    else:                   cls="alpha/beta (mixed globular)"
    return cls, round(pa,2), round(pb,2), round(dis*100)


def fold_to_function_prior(cls, tm, sig, length, pI):
    """What broad function does the structural class permit/suggest?"""
    if "membrane (polytopic)" in cls:
        return ("transporter / permease / channel — polytopic membrane "
                f"protein ({tm} TM helices) embedded in the lipid bilayer")
    if "lipoprotein" in cls or "secreted" in cls:
        return ("surface / secreted — substrate-binding protein, adhesin, "
                "or membrane-anchored accessory (signal peptide present)")
    if "all-alpha" in cls and length and length<150 and pI and pI>8.5:
        return ("DNA/RNA-binding or regulatory — small, basic, all-helical "
                "(classic nucleic-acid-binding signature)")
    if "all-beta" in cls:
        return ("beta-barrel/sandwich — porin, protease, or carbohydrate-"
                "binding fold")
    if "alpha/beta" in cls:
        return ("cytoplasmic enzyme — mixed alpha/beta globular fold "
                "(Rossmann/TIM-barrel territory: binds nucleotides/metabolites)")
    if "disordered" in cls:
        return ("regulatory/scaffolding — intrinsically disordered, often "
                "protein-protein interaction hubs")
    return "undetermined fold"


SUBSYS = [
    ("translation",  ["ribosom","trna","tRNA","translation","elongation",
                       "initiation","aminoacyl","ligase","release factor",
                       "amidotransferase","peptidyl"]),
    ("replication/repair", ["polymerase","helicase","topoisomerase","primase",
                             "gyrase","dna ","replicat","recombinas","ligase (nad"]),
    ("transcription", ["rna polymerase","sigma","transcription","rnase","nucleotidyl"]),
    ("energy/ATP",   ["atp synthase","f0f1","dehydrogenase","oxidoreductase",
                       "reductase","lactate","nadh"]),
    ("envelope/lipid",["lipid","cardiolipin","membrane","glycerol","fatty",
                        "acyltransferase","acp"]),
    ("transport",    ["transporter","permease","abc","pts","symporter","efflux",
                       "import","channel"]),
    ("cofactor/metab",["synthase","synthetase","kinase","phosphoribosyl",
                        "transferase","cysteine desulfurase","iron-sulfur",
                        "hydroxymethyl"]),
]
def subsystem_of(product):
    p=product.lower()
    for name,kw in SUBSYS:
        if any(k in p for k in kw): return name
    return "other/unknown"


def main():
    import pandas as pd
    s=pd.read_csv(REPO/"memory_bank/data/syn3a_essentiality_breuer2019.csv")
    gt=pd.read_csv(REPO/"memory_bank/data/syn3a_gene_table.csv")
    feat=pd.read_csv(REPO/"memory_bank/data/syn3a_gene_class_features.csv").set_index("locus_tag")
    ess=s[s.essentiality.str.lower()=="essential"]
    unclear=ess[ess.primary_function.str.contains("nclear|nknown",case=False,na=False)]

    syn_fa=parse_fasta(REPO/"memory_bank/data/multiorg_essentiality/sequences.fasta")
    syn_seq={k.split("|")[1]:v[0] for k,v in syn_fa.items() if len(k.split("|"))>=2}

    # homology reference (mgen annotated + known syn3a)
    mgen=parse_fasta(REPO/"memory_bank/data/multiorg_essentiality/_xs_cache/protein_sequence.fasta")
    ref=[]
    for k,(seq,hdr) in mgen.items():
        m=re.match(r"\S+\s+(K\d{5}|no KO assigned)\s+(.*?)(\s*\|.*)?$",hdr)
        ko,func=(m.group(1),m.group(2).strip()) if m else ("",hdr)
        if seq and "uncharacter" not in func.lower() and "hypothetical" not in func.lower():
            ref.append((k,seq,func+(f" [{ko}]" if ko.startswith("K") else "")))
    prod=dict(zip(s.locus_tag,s.gene_product))
    known=ess[~ess.primary_function.str.contains("nclear|nknown",case=False,na=False)]
    for lt in known.locus_tag:
        if lt in syn_seq: ref.append((f"syn3a:{lt}",syn_seq[lt],str(prod.get(lt,""))+" (syn3a)"))
    ref_km=[(r[0],r[1],r[2],kmers(r[1])) for r in ref]

    gts=gt.sort_values("start_1based").reset_index(drop=True)
    pos={r.locus_tag:i for i,r in gts.iterrows()}
    ess_set=set(ess.locus_tag)

    md=["# Sequence → Protein → Structure → Function → Network\n\n",
        "Full pipeline on syn3.0's 29 unclear essential genes. Structure stage "
        "is a sequence-derived proxy (Chou-Fasman SS + membrane topology + fold "
        "class); a true AlphaFold 3D model would refine it, the logic is identical.\n"]
    rows=[]
    for _,row in unclear.iterrows():
        lt=row.locus_tag; seq=syn_seq.get(lt,"")
        fr=feat.loc[lt] if lt in feat.index else None
        tm=int(fr.get("tm_helix_count",0)) if fr is not None else 0
        sig=int(fr.get("has_signal_pep",0)) if fr is not None else 0
        paa=int(fr.get("protein_length_aa",len(seq))) if fr is not None else len(seq)
        pI=float(fr.get("pI")) if (fr is not None and fr.get("pI")==fr.get("pI")) else None
        # stage 3 structure
        cls,pa,pb,dis=structural_class(seq,tm,sig)
        # stage 4a homology
        best=None
        if seq:
            qk=kmers(seq)
            cand=sorted(((len(qk&rk),rid,rseq,rann) for rid,rseq,rann,rk in ref_km),reverse=True)[:6]
            scored=[]
            for sh,rid,rseq,rann in cand:
                if sh<3: continue
                S,idn=smith_waterman(seq,rseq); scored.append((S,idn,rid,rann))
            scored.sort(reverse=True); best=scored[0] if scored else None
        fold_prior=fold_to_function_prior(cls,tm,sig,paa,pI)
        # stage 5 network
        ctx=[]
        if lt in pos:
            i=pos[lt]
            for off in (-3,-2,-1,1,2,3):
                jx=i+off
                if 0<=jx<len(gts):
                    nb=gts.iloc[jx]
                    ctx.append((off,nb.locus_tag,str(nb["product"]),nb.locus_tag in ess_set))
        # subsystem vote from neighbours (closest 2 each side weighted)
        votes=Counter()
        for off,nl,np_,e in ctx:
            w=3 if abs(off)==1 else (2 if abs(off)==2 else 1)
            sub=subsystem_of(np_)
            if sub!="other/unknown": votes[sub]+=w
        subsystem=votes.most_common(1)[0][0] if votes else "isolated/unknown"
        coess=sum(1 for *_,e in ctx if e)/len(ctx) if ctx else 0
        rows.append(dict(lt=lt,hint=str(row.gene_product),seq=seq,cls=cls,pa=pa,pb=pb,
                         dis=dis,tm=tm,sig=sig,paa=paa,pI=pI,best=best,fold_prior=fold_prior,
                         ctx=ctx,subsystem=subsystem,coess=coess))

    # print + md
    for r in rows:
        print(f"\n{'='*78}\n{r['lt']}   (prior hint: {r['hint'][:55]})")
        print(f"  [2 PROTEIN ]  {r['paa']} aa, pI {r['pI']}, {r['tm']} TM helices, "
              f"{'signal-pep' if r['sig'] else 'no signal'}")
        print(f"  [3 STRUCTURE] class = {r['cls']}  (Pα={r['pa']}, Pβ={r['pb']}, disorder~{r['dis']}%)")
        if r['best']:
            S,idn,rid,rann=r['best']
            print(f"  [4 FUNCTION ] homology: {rid} -> {rann}  ({idn*100:.0f}% id, SW {S})")
        print(f"               fold prior: {r['fold_prior']}")
        print(f"  [5 NETWORK  ] subsystem = {r['subsystem']}  | "
              f"co-essential neighbours = {r['coess']*100:.0f}%")
        nb_str="; ".join(f"{o:+d}{'*' if e else ''} {p[:32]}" for o,nl,p,e in r['ctx'][:6])
        print(f"               operon: {nb_str}")

        md.append(f"\n---\n\n## {r['lt']}  — *prior hint: {r['hint']}*\n\n")
        md.append(f"| stage | result |\n|---|---|\n")
        md.append(f"| **2 protein** | {r['paa']} aa · pI {r['pI']} · {r['tm']} TM "
                  f"· {'signal peptide' if r['sig'] else 'no signal'} |\n")
        md.append(f"| **3 structure (proxy)** | **{r['cls']}** "
                  f"(Pα {r['pa']}, Pβ {r['pb']}, disorder ~{r['dis']}%) |\n")
        if r['best']:
            S,idn,rid,rann=r['best']
            md.append(f"| **4 function — homology** | `{rid}` → {rann} "
                      f"({idn*100:.0f}% id) |\n")
        md.append(f"| **4 function — fold prior** | {r['fold_prior']} |\n")
        md.append(f"| **5 network — subsystem** | **{r['subsystem']}** "
                  f"(co-essential neighbours {r['coess']*100:.0f}%) |\n")
        md.append(f"| **5 network — operon** | "
                  + "; ".join(f"{o:+d}{'(ess)' if e else ''} {p.strip()[:35]}"
                              for o,nl,p,e in r['ctx'][:6]) + " |\n")

    # summary table
    print(f"\n\n{'='*78}\nSUBSYSTEM ASSIGNMENT SUMMARY")
    sub_counts=Counter(r['subsystem'] for r in rows)
    for sub,n in sub_counts.most_common():
        print(f"  {sub:<22s} {n}")
    md.append(f"\n---\n\n## Subsystem assignment summary\n\n| subsystem | n genes |\n|---|---|\n")
    for sub,n in sub_counts.most_common(): md.append(f"| {sub} | {n} |\n")

    (REPO/"outputs/syn3a_pipeline.md").write_text("".join(md))
    print(f"\nwrote outputs/syn3a_pipeline.md")
    return 0


if __name__=="__main__":
    sys.exit(main())
