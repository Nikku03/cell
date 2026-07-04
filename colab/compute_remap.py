"""ReMap TF ChIP-seq -> measured TF -> target-gene regulatory edges.

ReMap is a catalog of TF binding peaks. We assign each peak to genes whose TSS lies within a
window (default +/-5 kb), giving TF -> target edges supported by actual binding. Emitted only when
a TF hits a gene's promoter in >= MIN_SUPPORT peaks (noise filter) -> remap_tf_targets.tsv
(loaded by build_cell_complete into the regulatory layer alongside CollecTRI).

Reads the ReMap non-redundant BED + refGene (for TSS). Skips if absent (runs on Colab).
First-pass: binding != causal regulation; treat these as candidate edges. -> remap_tf_targets.tsv
"""
import gzip, bisect
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
bed=H/"remap.bed";     bedgz=H/"remap.bed.gz"
rg =H/"refGene.txt.gz"
WIN=5000; MIN_SUPPORT=3; MAX_TARGETS=400
src=bed if bed.exists() else (bedgz if bedgz.exists() else None)
if not src or not rg.exists():
    print("ReMap bed / refGene absent -> skipping (runs on Colab)"); raise SystemExit(0)

def _open(p): return gzip.open(p,"rt") if str(p).endswith(".gz") else open(p)
# TSS per chrom: sorted [(pos, symbol)]
tss=defaultdict(list)
with gzip.open(rg,"rt") as f:
    for l in f:
        p=l.rstrip("\n").split("\t")
        if len(p)<13: continue
        chrom,strand,txStart,txEnd,sym=p[2],p[3],p[4],p[5],p[12]
        if "_" in chrom or not sym: continue
        try: pos=int(txStart) if strand=="+" else int(txEnd)
        except: continue
        tss[chrom].append((pos,sym))
for c in tss: tss[c].sort()
pos_by_chrom={c:[x[0] for x in v] for c,v in tss.items()}

def genes_near(chrom,mid):
    arr=pos_by_chrom.get(chrom)
    if not arr: return []
    lo=bisect.bisect_left(arr,mid-WIN); hi=bisect.bisect_right(arr,mid+WIN)
    return [tss[chrom][k][1] for k in range(lo,hi)]

# ReMap BED name field encodes the TF (e.g. "GSE.TF.cell" or just "TF")
print(f"ReMap: {sum(len(v) for v in tss.values())} TSS across {len(tss)} chroms indexed from refGene")
pair=defaultdict(int); npeaks=0; tfs_seen=set(); sample=[]
with _open(src) as f:
    for l in f:
        p=l.split("\t")
        if len(p)<4: continue
        chrom=p[0]
        try: mid=(int(p[1])+int(p[2]))//2
        except: continue
        name=p[3]
        # ReMap 'all' name = GSE.TF.biotype (3 parts); 'nr' name = TF directly
        tf=name.split(".")[1] if name.count(".")>=2 else name.split(".")[0]
        tf=tf.strip().upper()
        if not tf: continue
        npeaks+=1; tfs_seen.add(tf)
        if len(sample)<3: sample.append(name)
        for g in genes_near(chrom,mid):
            if g.upper()!=tf: pair[(tf,g)]+=1
print(f"ReMap: {npeaks} peaks, {len(tfs_seen)} unique TFs (name samples: {sample})")

bytf=defaultdict(list)
for (tf,g),n in pair.items():
    if n>=MIN_SUPPORT: bytf[tf].append((g,n))
with open(OUT/"remap_tf_targets.tsv","w") as o:
    ne=0
    for tf,gs in bytf.items():
        for g,n in sorted(gs,key=lambda x:-x[1])[:MAX_TARGETS]:
            o.write(f"{tf}\t{g}\t{n}\n"); ne+=1
print(f"ReMap: {len(bytf)} TFs -> {ne} TF-target edges (>= {MIN_SUPPORT} peaks within {WIN}bp of a TSS)")
