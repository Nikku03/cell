"""AG2BANK -- the data behind AlphaGenome 2 (memory_bank/datasets/alphagenome2.json), pulled into the memory bank as
compressed per-cell-type records, not raw files.

Raw, these datasets run to many terabytes (CELLxGENE alone is >100 million cells); this sandbox has ~25 GB. The memory
bank's job is to hold a lot in little space, so each source is reduced to what a cell-type model needs: one profile
per (tissue, cell type) or per biosample, quantised, with the source and cell counts kept. Everything lands in the
git-ignored cache rem/atlas/_cache/ag2bank (licences vary by dataset; see memory_bank/sources); only code and build
reports are committed.

  cellxgene   CZ CELLxGENE 'Gene Expression' aggregates (the WMG service), human and mouse, every gene x every
              (tissue, cell type) pair, rolled up over descendant cell types (the service's only working mode):
              me = mean log-normalised expression among cells expressing the gene, pc = share of cells expressing it,
              n = number of expressing cells. Stored sparse: uint8-quantised me (step 0.04) and pc (step 1/255),
              int32 n.
  (further sources are added as separate commands; see the build report)

Run: python3 -P rem/atlas/ag2bank.py cellxgene [human|mouse]
"""

from __future__ import annotations
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
BANK = os.path.join(HERE, "_cache", "ag2bank")
RAW = os.path.join(BANK, "raw")
WMG = "https://api.cellxgene.cziscience.com/wmg/v2/"
ORG = {"human": "NCBITaxon:9606", "mouse": "NCBITaxon:10090"}
BATCH, WORKERS = 40, 3
ME_STEP = 0.04


def _post(path, body, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(WMG + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": "cell-memory-bank"})
            return json.loads(urllib.request.urlopen(req, timeout=900).read())
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as ex:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def _shard(org, k, ids, outdir):
    path = os.path.join(outdir, f"shard_{k:05d}.npz")
    if os.path.exists(path):
        return k, None
    r = _post("query", {"filter": {"gene_ontology_term_ids": ids, "organism_ontology_term_id": org}, "is_rollup": True})
    es = r["expression_summary"]
    g_, t_, c_, me, pc, n = [], [], [], [], [], []
    for g, tissues in es.items():
        for t, cts in tissues.items():
            for c, v in cts.items():
                a = v.get("aggregated", v)
                g_.append(g); t_.append(t); c_.append(c)
                me.append(a.get("me", 0.0)); pc.append(a.get("pc", 0.0)); n.append(a.get("n", 0))
    np.savez_compressed(path, gene=np.array(g_), tissue=np.array(t_), cell=np.array(c_),
                        me=np.clip(np.round(np.array(me) / ME_STEP), 0, 255).astype(np.uint8),
                        pc=np.clip(np.round(np.array(pc) * 255), 0, 255).astype(np.uint8), n=np.array(n, dtype=np.int32))
    return k, r.get("term_id_labels")


def cellxgene(species="human"):
    t0 = time.time()
    org = ORG[species]; outdir = os.path.join(BANK, "cellxgene", species, "shards"); os.makedirs(outdir, exist_ok=True)
    dims_p = os.path.join(RAW, "wmg_dims.json")
    if not os.path.exists(dims_p):
        os.makedirs(RAW, exist_ok=True)
        dims = json.loads(urllib.request.urlopen(WMG + "primary_filter_dimensions", timeout=300).read())
        json.dump(dims, open(dims_p, "w"))
    dims = json.load(open(dims_p))
    genes = [list(g.items())[0] for g in dims["gene_terms"][org]]
    json.dump(dict(snapshot_id=dims["snapshot_id"], organism=org, genes=genes, tissues=dims["tissue_terms"][org]),
              open(os.path.join(BANK, "cellxgene", species, "genes.json"), "w"))
    batches = [[g for g, _ in genes[i:i + BATCH]] for i in range(0, len(genes), BATCH)]
    labels_p = os.path.join(BANK, "cellxgene", species, "labels.json")
    labels = json.load(open(labels_p)) if os.path.exists(labels_p) else {}
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_shard, org, k, ids, outdir) for k, ids in enumerate(batches)]
        for f in as_completed(futs):
            k, lab = f.result(); done += 1
            if lab:
                for key, val in lab.items():
                    if isinstance(val, dict):
                        for t, cts in val.items():
                            labels.setdefault(key, {}).setdefault(t, cts if not isinstance(cts, dict) else {})
                            if isinstance(cts, dict):
                                labels[key][t].update(cts)
            if done % 25 == 0 or done == len(batches):
                json.dump(labels, open(labels_p, "w"))
                print(f"cellxgene {species}: {done}/{len(batches)} batches ({time.time() - t0:.0f}s)", flush=True)
    json.dump(labels, open(labels_p, "w"))
    print(f"cellxgene {species}: done, {len(batches)} batches of {BATCH} genes ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------
# step 2: bulk expression profiles (GTEx tissues, ENCODE RNA-seq biosamples, FANTOM5 CAGE samples)
# stored as float16 log1p(TPM), samples x genes, with a metadata table per row
# ---------------------------------------------------------------------------------------------------------
UA = {"User-Agent": "cell-memory-bank"}


def _get(url, tries=6, timeout=900):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def _gunzip(b):
    import gzip
    return gzip.decompress(b).decode("utf-8", errors="replace")


def _save_profiles(name, M, rows, cols, meta, info):
    d = os.path.join(BANK, "bulk"); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{name}.npz"), X=np.log1p(np.clip(M, 0, None)).astype(np.float16),
                        rows=np.array(rows), cols=np.array(cols))
    json.dump(dict(meta=meta, info=info), open(os.path.join(d, f"{name}_meta.json"), "w"))
    print(f"{name}: {M.shape[0]:,} profiles x {M.shape[1]:,} genes  {info}", flush=True)


def gtex():
    url = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    lines = _gunzip(_get(url)).splitlines()
    head = lines[2].split("\t"); tissues = head[2:]
    genes, sym, M = [], [], []
    for l in lines[3:]:
        f = l.split("\t")
        if len(f) == len(head):
            genes.append(f[0].split(".")[0]); sym.append(f[1]); M.append([float(x) for x in f[2:]])
    M = np.array(M, dtype=np.float32).T
    _save_profiles("gtex_v8_tissue_median", M, tissues, genes, dict(symbols=sym),
                   dict(source="gtex_portal", url=url, unit="median TPM per tissue (log1p stored)", tissues=len(tissues)))


def fantom5(species="human"):
    asm = "hg38" if species == "human" else "mm10"
    url = (f"https://fantom.gsc.riken.jp/5/datafiles/reprocessed/{asm}_latest/extra/CAGE_peaks_expression/"
           f"{asm}_fair+new_genes_phase1and2_tpm.osc.txt.gz")
    import urllib.parse
    lines = [l for l in _gunzip(_get(url)).splitlines() if not l.startswith("##")]
    head = lines[0].split("\t"); samples = head[1:]
    genes, M = [], []
    for l in lines[1:]:
        f = l.split("\t")
        if len(f) == len(head):
            genes.append(f[0]); M.append([float(x) for x in f[1:]])
    M = np.array(M, dtype=np.float32).T
    desc = [urllib.parse.unquote(s.split(".CNhs")[0].replace("tpm.", "", 1)) for s in samples]
    lib = [s.split(".")[-4] if s.count(".") >= 4 else "" for s in samples]
    _save_profiles(f"fantom5_cage_{species}", M, samples, genes, dict(description=desc, library=lib),
                   dict(source="fantom5_cage", url=url, unit="CAGE TPM per gene (log1p stored)", samples=len(samples), assembly=asm))


ENC = "https://www.encodeproject.org"
ANN_PREF = {"GRCh38": ["V29", "V24", "V22", None], "mm10": ["M21", "M4", None]}


def _enc_json(q):
    return json.loads(_get(ENC + q + "&format=json&limit=all").decode())["@graph"]


def _enc_tpm(href):
    lines = _get(ENC + href).decode().splitlines()
    h = lines[0].split("\t"); gi, ti = h.index("gene_id"), h.index("TPM")
    out = {}
    for l in lines[1:]:
        f = l.split("\t")
        if len(f) > ti and f[gi].startswith("ENS"):
            out[f[gi].split(".")[0]] = float(f[ti])
    return out


def encode_rna():
    t0 = time.time()
    exps = _enc_json("/search/?type=Experiment&status=released&assay_title=total+RNA-seq&assay_title=polyA+plus+RNA-seq"
                     "&files.output_type=gene+quantifications&field=accession&field=assay_title&field=biosample_ontology.term_name"
                     "&field=biosample_ontology.term_id&field=biosample_ontology.classification&field=replicates.library.biosample.organism.scientific_name"
                     "&field=replicates.library.biosample.life_stage")
    meta = {}
    for e in exps:
        reps = [r["library"]["biosample"] for r in e.get("replicates", []) if r.get("library", {}).get("biosample")]
        meta[e["accession"]] = dict(assay=e["assay_title"], biosample=e["biosample_ontology"]["term_name"],
                                    term_id=e["biosample_ontology"]["term_id"], classification=e["biosample_ontology"]["classification"],
                                    organism=reps[0]["organism"]["scientific_name"] if reps else "?",
                                    life_stage=sorted({b.get("life_stage", "") for b in reps}))
    files = _enc_json("/search/?type=File&status=released&output_type=gene+quantifications&file_format=tsv"
                      "&assay_title=total+RNA-seq&assay_title=polyA+plus+RNA-seq&field=accession&field=href&field=dataset"
                      "&field=assembly&field=genome_annotation")
    by = {}
    for f in files:
        acc = f["dataset"].strip("/").split("/")[-1]
        if f.get("assembly") in ANN_PREF and acc in meta:
            by.setdefault((acc, f["assembly"]), []).append(f)
    chosen = {}
    for (acc, asm), fl in by.items():
        for ann in ANN_PREF[asm]:
            sel = [f for f in fl if f.get("genome_annotation") == ann]
            if sel:
                chosen[(acc, asm)] = (ann, sel); break
    for asm, species in (("GRCh38", "human"), ("mm10", "mouse")):
        keys = sorted(k for k in chosen if k[1] == asm)
        sh = os.path.join(BANK, "bulk", f"encode_rna_{species}_shards"); os.makedirs(sh, exist_ok=True)
        ref_p = os.path.join(sh, "genes.json")
        if not os.path.exists(ref_p):
            ann0 = ANN_PREF[asm][0]
            first = next(f for k in keys for f in chosen[k][1] if chosen[k][0] == ann0)
            json.dump(sorted(_enc_tpm(first["href"])), open(ref_p, "w"))
        ref = json.load(open(ref_p)); col = {g: i for i, g in enumerate(ref)}

        def one(k):
            p = os.path.join(sh, f"{k[0]}.npy")
            if os.path.exists(p):
                return
            vs = []
            for f in chosen[k][1]:
                v = np.full(len(ref), np.nan, np.float32)
                for g, x in _enc_tpm(f["href"]).items():
                    if g in col:
                        v[col[g]] = x
                vs.append(v)
            np.save(p, np.nanmean(np.stack(vs), 0).astype(np.float32))
        done = 0
        with ThreadPoolExecutor(4) as ex:
            for _ in as_completed([ex.submit(one, k) for k in keys]):
                done += 1
                if done % 100 == 0 or done == len(keys):
                    print(f"encode_rna {species}: {done}/{len(keys)} experiments ({time.time() - t0:.0f}s)", flush=True)
        M = np.stack([np.load(os.path.join(sh, f"{k[0]}.npy")) for k in keys])
        rows = [k[0] for k in keys]
        m = dict(experiment=rows, annotation=[chosen[k][0] for k in keys], replicates=[len(chosen[k][1]) for k in keys],
                 **{f: [meta[k[0]][f] for k in keys] for f in ("assay", "biosample", "term_id", "classification", "life_stage")})
        _save_profiles(f"encode_rna_{species}", np.nan_to_num(M), rows, ref, m,
                       dict(source="encode_portal", unit="RSEM TPM per gene, replicates averaged (log1p stored)", experiments=len(rows),
                            biosamples=len(set(m["biosample"])), assembly=asm))


# ---------------------------------------------------------------------------------------------------------
# step 3: chromatin per biosample -- ENCODE Registry v4 biosample-specific cCRE classes
# (from each biosample's DNase/ATAC, H3K4me3, H3K27ac, CTCF): only active elements kept (class != Low-DNase),
# as uint32 index into the registry + uint8 class code
# ---------------------------------------------------------------------------------------------------------
CCRE_CLASSES = ["PLS", "pELS", "dELS", "CA-H3K4me3", "CA-CTCF", "CA-TF", "CA-only", "TF-only", "Low-DNase"]


def _ccre_rows(acc):
    import gzip
    import io
    b = _get(f"{ENC}/files/{acc}/@@download/{acc}.bed.gz", timeout=1800)
    for line in io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(b)), encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 10:
            yield f


def ccre():
    t0 = time.time()
    ann = _enc_json("/search/?type=Annotation&annotation_type=candidate+Cis-Regulatory+Elements&status=released&encyclopedia_version=ENCODE+v4"
                    "&field=accession&field=assembly&field=biosample_ontology.term_name&field=biosample_ontology.term_id"
                    "&field=biosample_ontology.classification&field=description&field=files.accession&field=files.file_format&field=files.assembly")
    code = {c: i for i, c in enumerate(CCRE_CLASSES)}
    for asm, species in (("GRCh38", "human"), ("mm10", "mouse")):
        items = [a for a in ann if a.get("biosample_ontology") and asm in a.get("assembly", [])]
        d = os.path.join(BANK, "ccre", species); sh = os.path.join(d, "shards"); os.makedirs(sh, exist_ok=True)
        reg_p = os.path.join(d, "registry.npz")
        bed = lambda a: next((f["accession"] for f in a["files"] if f.get("file_format") == "bed" and f.get("assembly") == asm), None)
        items = [a for a in items if bed(a)]
        if not os.path.exists(reg_p):
            ids, ch, st, en = [], [], [], []
            for f in _ccre_rows(bed(items[0])):
                ids.append(f[3]); ch.append(f[0]); st.append(int(f[1])); en.append(int(f[2]))
            np.savez_compressed(reg_p, id=np.array(ids), chrom=np.array(ch), start=np.array(st, np.int32), end=np.array(en, np.int32))
        reg = np.load(reg_p); index = {x: i for i, x in enumerate(reg["id"].tolist())}

        def one(a):
            p = os.path.join(sh, f"{a['accession']}.npz")
            if os.path.exists(p):
                return 0
            idx, cls, extra = [], [], 0
            for f in _ccre_rows(bed(a)):
                c = f[9]
                if c == "Low-DNase":
                    continue
                i = index.get(f[3])
                if i is None:
                    extra += 1; continue
                idx.append(i); cls.append(code.get(c, 255))
            np.savez_compressed(p, idx=np.array(idx, np.uint32), cls=np.array(cls, np.uint8))
            return extra
        done = 0
        with ThreadPoolExecutor(4) as ex:
            for _ in as_completed([ex.submit(one, a) for a in items]):
                done += 1
                if done % 50 == 0 or done == len(items):
                    print(f"ccre {species}: {done}/{len(items)} biosamples ({time.time() - t0:.0f}s)", flush=True)
        meta = [dict(annotation=a["accession"], file=bed(a), biosample=a["biosample_ontology"]["term_name"],
                     term_id=a["biosample_ontology"].get("term_id"), classification=a["biosample_ontology"]["classification"],
                     description=a.get("description", "")) for a in items]
        json.dump(dict(classes=CCRE_CLASSES, registry_elements=int(len(reg["id"])), biosamples=meta,
                       source="encode_portal", note="ENCODE Registry v4 biosample-specific cCRE classifications; Low-DNase (inactive) omitted"),
                  open(os.path.join(d, "meta.json"), "w"))
        print(f"ccre {species}: {len(items)} biosamples, registry {len(reg['id']):,} elements ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ccre":
        ccre()
    if cmd == "cellxgene":
        cellxgene(sys.argv[2] if len(sys.argv) > 2 else "human")
    elif cmd == "gtex":
        gtex()
    elif cmd == "fantom5":
        fantom5(sys.argv[2] if len(sys.argv) > 2 else "human")
    elif cmd == "encode_rna":
        encode_rna()
