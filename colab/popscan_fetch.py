"""Fetch per-population allele frequencies for a few-hundred-gene panel (adaptation
controls + human TFs + random background) to run a differentiation scan. Appends to a
shared cache so nothing is re-fetched. NOT all 20k genes (API is per-gene) — an honest
targeted scan of ~400 genes."""
import json, urllib.request, time, gzip, csv, random
from pathlib import Path
H=Path("data/external_data/human"); SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
random.seed(0)
cache=SC/"gnomad_genes.json"; raw=json.load(open(cache)) if cache.exists() else {}
# gene list
tfs=[]
for l in open(H/"trrust_human.tsv"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3: tfs.append(p[0])
tfs=sorted(set(tfs))
random.shuffle(tfs); tf_sample=tfs[:150]
bg=[]
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    h=f.readline().split("\t"); gi=h.index("gene")
    for l in f: bg.append(l.split("\t")[gi])
bg=sorted(set(bg)); random.shuffle(bg); bg_sample=bg[:230]
want=list(dict.fromkeys(list(raw)+tf_sample+bg_sample))   # keep already-cached first
print(f"panel: {len(want)} genes ({len(raw)} cached, {len(tf_sample)} TFs, {len(bg_sample)} background)",flush=True)
def gq(sym):
    q='{gene(gene_symbol:"%s",reference_genome:GRCh37){variants(dataset:gnomad_r2_1){pos consequence hgvsp exome{ac an populations{id ac an}}}}}'%sym
    req=urllib.request.Request("https://gnomad.broadinstitute.org/api",
        data=json.dumps({"query":q}).encode(),headers={"Content-Type":"application/json"})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=120))
        except Exception: time.sleep(4)
    return None
done=0; t0=time.time()
for i,sym in enumerate(want):
    if sym in raw: continue
    d=gq(sym)
    if d and d.get("data",{}).get("gene"):
        # keep only missense to shrink cache
        vs=[v for v in d["data"]["gene"]["variants"] if v.get("consequence")=="missense_variant" and v.get("exome") and v["exome"].get("an")]
        raw[sym]=vs
    else: raw[sym]=[]
    done+=1
    if done%25==0:
        json.dump(raw,open(cache,"w")); print(f"  {done} fetched ({int(time.time()-t0)}s)",flush=True)
    time.sleep(0.8)
json.dump(raw,open(cache,"w"))
print(f"DONE: cache has {len(raw)} genes",flush=True)
