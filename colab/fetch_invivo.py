"""Fetch real ENCODE K562 tracks + UCSC chr fasta for the in-vivo gating test."""
import urllib.request, json, gzip, os, sys
OUT = "outputs/orphan/invivo"
os.makedirs(OUT, exist_ok=True)
CHR = "chr22"

def q(url):
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=90))

def dl(href, dest):
    url = "https://www.encodeproject.org" + href if href.startswith("/") else href
    urllib.request.urlretrieve(url, dest)
    return os.path.getsize(dest)

def find_file(assay, target=None, prefer=("optimal IDR thresholded peaks",
                                          "conservative IDR thresholded peaks",
                                          "IDR thresholded peaks", "replicated peaks",
                                          "pseudoreplicated peaks", "peaks")):
    url = ("https://www.encodeproject.org/search/?type=File&file_format=bed&file_format_type=narrowPeak"
           "&assembly=GRCh38&status=released&biosample_ontology.term_name=K562&limit=25&format=json"
           f"&assay_title={assay.replace(' ', '+')}")
    if target:
        url += f"&target.label={target}"
    d = q(url)
    byot = {}
    for f in d['@graph']:
        byot.setdefault(f.get('output_type'), f)
    for ot in prefer:
        if ot in byot:
            return byot[ot]['accession'], byot[ot]['href'], ot
    return (None, None, None)

def filt_bed(gzpath, chrom, dest, cols=3):
    """keep only rows on `chrom`; write chrom<TAB>start<TAB>end[..]."""
    n = 0
    with gzip.open(gzpath, 'rt') as fh, open(dest, 'w') as out:
        for line in fh:
            p = line.split('\t')
            if p[0] == chrom:
                out.write('\t'.join(p[:max(cols, 7)]).rstrip('\n') + '\n')
                n += 1
    return n

# --- 1. chromosome fasta ---
faz = f"{OUT}/{CHR}.fa.gz"
if not os.path.exists(f"{OUT}/{CHR}.fa"):
    print(f"fetching {CHR}.fa.gz ...", flush=True)
    urllib.request.urlretrieve(f"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{CHR}.fa.gz", faz)
    with gzip.open(faz, 'rt') as fh, open(f"{OUT}/{CHR}.fa", 'w') as out:
        for line in fh:
            out.write(line)
    os.remove(faz)
    print(f"  {CHR}.fa ready ({os.path.getsize(f'{OUT}/{CHR}.fa')//1_000_000} MB)", flush=True)

manifest = {"chrom": CHR, "cell": "K562", "tracks": {}}

# --- 2. accessibility + active tracks ---
for key, (assay, tgt) in {"dnase": ("DNase-seq", None),
                          "h3k27ac": ("Histone ChIP-seq", "H3K27ac")}.items():
    acc, href, ot = find_file(assay, tgt)
    print(f"{key}: {acc} ({ot})", flush=True)
    tmp = f"{OUT}/_{key}.bed.gz"; dl(href, tmp)
    n = filt_bed(tmp, CHR, f"{OUT}/{key}.{CHR}.bed"); os.remove(tmp)
    manifest["tracks"][key] = {"accession": acc, "output_type": ot, "n_peaks_chr": n}
    print(f"   {n} peaks on {CHR}", flush=True)

# --- 3. TF ChIP-seq ground truth ---
TFS = ["CTCF", "REST", "GATA1", "GATA2", "SPI1", "MYC", "MAX", "YY1", "NRF1", "JUND", "USF1", "EGR1"]
manifest["tf_chip"] = {}
for tf in TFS:
    acc, href, ot = find_file("TF ChIP-seq", tf)
    if not acc:
        print(f"{tf}: NO ChIP file", flush=True); continue
    tmp = f"{OUT}/_{tf}.bed.gz"; dl(href, tmp)
    n = filt_bed(tmp, CHR, f"{OUT}/chip.{tf}.{CHR}.bed"); os.remove(tmp)
    manifest["tf_chip"][tf] = {"accession": acc, "output_type": ot, "n_peaks_chr": n}
    print(f"{tf}: {acc} ({ot}) -> {n} peaks on {CHR}", flush=True)

# --- 4. Gate 3: ABC / ENCODE-rE2G element->gene 3D links ---
ABC_ACC = "ENCFF902VSE"    # K562 thresholded element gene links (ENCODE-rE2G/ABC), bedpe
tmp = f"{OUT}/_abc.bedpe.gz"
dl(f"/files/{ABC_ACC}/@@download/{ABC_ACC}.bedpe.gz", tmp)
rows = []; ensgs = set()
with gzip.open(tmp, 'rt') as fh:
    for line in fh:
        if line.startswith('#'):
            continue
        p = line.rstrip('\n').split('\t')
        if p[0] != CHR:
            continue
        ensg = p[6].split(':')[-1].split('.')[0]     # name = element:ENSG
        rows.append((int(p[1]), int(p[2]), ensg, p[7]))
        ensgs.add(ensg)
os.remove(tmp)
# map ENSG -> symbol via mygene.info (batch)
sym = {}
ids = list(ensgs)
for i in range(0, len(ids), 400):
    data = ("q=" + ",".join(ids[i:i + 400]) + "&scopes=ensembl.gene&fields=symbol&species=human").encode()
    req = urllib.request.Request("https://mygene.info/v3/query", data=data,
                                 headers={'Content-Type': 'application/x-www-form-urlencoded'})
    for r in json.load(urllib.request.urlopen(req, timeout=120)):
        if 'symbol' in r:
            sym[r['query']] = r['symbol']
kept = 0
with open(f"{OUT}/abc.{CHR}.bed", 'w') as out:
    for s, e, ensg, sc in rows:
        g = sym.get(ensg)
        if g:
            out.write(f"{CHR}\t{s}\t{e}\t{g}\t{sc}\n"); kept += 1
manifest["tracks"]["abc"] = {"accession": ABC_ACC, "output_type": "thresholded element gene links (ENCODE-rE2G/ABC)",
                             "n_links_chr": kept, "n_target_genes": len(set(sym.values()))}
print(f"abc: {ABC_ACC} -> {kept} element-gene links on {CHR}", flush=True)

json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print("DONE ->", f"{OUT}/manifest.json", flush=True)
