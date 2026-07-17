"""Fetch DNA sequence (600bp centered) for each unique CRISPR element via UCSC API."""
import urllib.request, json, time
import pandas as pd
OUT = "outputs/orphan/invivo"
df = pd.read_csv("outputs/orphan/crispr_features_compendium.csv")
elems = sorted(df["element"].unique())
W = 300
seqs = {}
for i, ek in enumerate(elems):
    chrom, rng = ek.split(":")
    s, e = rng.split("-"); c = (int(s) + int(e)) // 2
    a, b = c - W, c + W
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={a};end={b}"
    ok = False
    for _ in range(4):
        try:
            d = json.load(urllib.request.urlopen(url, timeout=30))
            dna = d.get("dna", "")
            if len(dna) >= 2 * W - 10:
                seqs[ek] = dna.upper(); ok = True; break
        except Exception:
            time.sleep(0.8)
    if (i + 1) % 250 == 0:
        print(f"{i+1}/{len(elems)} ({len(seqs)} ok)", flush=True)
json.dump(seqs, open(f"{OUT}/element_seqs.json", "w"))
print(f"DONE: {len(seqs)}/{len(elems)} element sequences -> {OUT}/element_seqs.json", flush=True)
