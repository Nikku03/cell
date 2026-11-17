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


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cellxgene":
        cellxgene(sys.argv[2] if len(sys.argv) > 2 else "human")
