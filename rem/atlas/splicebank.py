"""SPLICEBANK -- what happens to each protein's RNA versions when one RNA-binding protein is knocked down (ENCODE), added
to the memory bank (cellbrain), to test what governs the choice of RNA version (G11, splicetest.py).

Asked by the user: RNA maturation -- "we need when and in what condition and what governs them"; then "yes start #1"
(check the splicing proteins that follow the address-changing versions across tissues against real knockdowns).
Source: ENCODE shRNA knockdown RNA-seq of RNA-binding proteins (Van Nostrand et al. 2020 Nature; Fu lab / Graveley lab;
encodeproject.org; ENCODE data, free use): K562 and HepG2, two replicates per knockdown, matched control-shRNA
experiments ("possible_controls"). Files: ENCODE4 uniform pipeline (v1.2.x, GRCh38, GENCODE V29) kallisto transcript
quantifications, one per replicate, md5 from ENCODE. Each file is read as data, summed per UniProt isoform (Ensembl
transcript -> isoform from ADDR_UNIPROT) and deleted; only the per-isoform sums are kept (~33 GB of tables otherwise).
Layer: SPLICE_KD_ENCODE (npz): isoforms x replicate files, TPM and estimated counts, with per-file metadata (cell line,
experiment, knockdown target or control, replicate, control experiments).

Run: python3 -I rem/atlas/splicebank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import concurrent.futures
import csv
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("xcellbank", os.path.join(HERE, "xcellbank.py"))
xb = importlib.util.module_from_spec(_s); _s.loader.exec_module(xb)
cbb, Fact = xb.cbb, xb.Fact

RAW = os.path.join(HERE, "_cache", "splice")
LINES = ("K562", "HepG2")
UA = {"User-Agent": "cell-atlas-research/1.0", "Accept": "application/json"}
cbb.SRC["encode_rbp_kd"] = ("ENCODE shRNA knockdown RNA-seq of RNA-binding proteins (Van Nostrand et al. 2020 Nature), K562 and HepG2, "
                            "ENCODE4 kallisto transcript quantifications (GRCh38, GENCODE V29)", "https://www.encodeproject.org", "ENCODE data, free use")


def api(url):
    for k in range(6):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read())
        except Exception:
            if k == 5:
                raise
            time.sleep(5 * 2 ** k)


def kallisto_files(exp):
    """released GRCh38 V29 kallisto transcript quantifications of one experiment: [(replicate, accession, md5, url)]."""
    d = api(f"https://www.encodeproject.org/experiments/{exp}/?format=json"); out = []
    for f in d["files"]:
        if (f.get("output_type") == "transcript quantifications" and f.get("assembly") == "GRCh38" and f.get("genome_annotation") == "V29"
                and f.get("status") == "released" and "kallisto" in (f.get("submitted_file_name") or "")):
            out.append((f["biological_replicates"][0], f["accession"], f["md5sum"], "https://www.encodeproject.org" + f["href"]))
    tgt = d.get("target"); tgt = tgt.get("label") if isinstance(tgt, dict) else None
    ctl = [c["accession"] if isinstance(c, dict) else c.split("/")[-2] for c in d.get("possible_controls", [])]
    return sorted(out), tgt, ctl


def isoform_map():
    P = json.load(open(cbb.layer("ADDR_UNIPROT")[1]))
    tx2iso, iso_gene = {}, {}
    for g, v in P.items():
        for t, iso in v["transcripts"]:
            tx2iso[t] = iso; iso_gene[iso] = g
    isos = sorted(iso_gene)
    return tx2iso, isos, iso_gene


def _one(job, tx2iso, ii):
    """download one replicate table, sum TPM and counts per isoform, delete the table."""
    acc, md5, url = job; out = os.path.join(RAW, "rows", acc + ".npz")
    if os.path.exists(out):
        return acc, "kept"
    tmp = os.path.join(RAW, "tmp", acc + ".tsv"); xb.fetch(url, tmp, md5)
    tpm = np.zeros(len(ii), np.float32); cnt = np.zeros(len(ii), np.float32)
    with open(tmp) as fh:
        rd = csv.reader(fh, delimiter="\t"); head = next(rd); it, ic = head.index("tpm"), head.index("est_counts")
        for row in rd:
            t = row[0].split("|")[0].split(".")[0]; iso = tx2iso.get(t)
            if iso is not None:
                k = ii[iso]; tpm[k] += float(row[it]); cnt[k] += float(row[ic])
    np.savez_compressed(out + ".tmp.npz", tpm=tpm, cnt=cnt); os.replace(out + ".tmp.npz", out); os.remove(tmp)
    return acc, "fetched"


def pull():
    os.makedirs(os.path.join(RAW, "rows"), exist_ok=True); os.makedirs(os.path.join(RAW, "tmp"), exist_ok=True)
    tx2iso, isos, _ = isoform_map(); ii = {s: i for i, s in enumerate(isos)}
    meta_path = os.path.join(RAW, "experiments.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
        meta = {"experiments": {}, "searched": cbb.now()}
        for line in LINES:
            s = api("https://www.encodeproject.org/search/?type=Experiment&assay_title=shRNA+RNA-seq&biosample_ontology.term_name="
                    f"{line}&status=released&format=json&limit=all&field=accession")
            exps = sorted(e["accession"] for e in s["@graph"]); ctls = set()
            for e in exps:
                files, tgt, ctl = kallisto_files(e); meta["experiments"][e] = dict(line=line, role="knockdown", target=tgt, controls=ctl, files=files); ctls |= set(ctl)
            for c in sorted(ctls):
                if c not in meta["experiments"]:
                    files, _, _ = kallisto_files(c); meta["experiments"][c] = dict(line=line, role="control", target=None, controls=[], files=files)
            print(line, len(exps), "knockdowns,", len(ctls), "controls", flush=True)
        json.dump(meta, open(meta_path, "w"), indent=1)
    jobs = sorted({(f[1], f[2], f[3]) for v in meta["experiments"].values() for f in v["files"]})
    print(f"{len(jobs)} replicate tables to read", flush=True); t0 = time.time(); n = 0
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        for acc, how in ex.map(lambda j: _one(j, tx2iso, ii), jobs):
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)


def build():
    meta = json.load(open(os.path.join(RAW, "experiments.json"))); tx2iso, isos, iso_gene = isoform_map()
    for v in meta["experiments"].values():                                      # ENCODE lists the control-shRNA experiments in the same
        v["role"] = "control" if not v["target"] else "knockdown"              # search; a control has no target protein
    cols = []
    for e, v in sorted(meta["experiments"].items()):
        for rep, acc, md5, url in v["files"]:
            cols.append(dict(file=acc, experiment=e, line=v["line"], role=v["role"], target=v["target"] or "", replicate=rep, controls=v["controls"]))
    T = np.zeros((len(isos), len(cols)), np.float32); C = np.zeros_like(T)
    for j, c in enumerate(cols):
        z = np.load(os.path.join(RAW, "rows", c["file"] + ".npz")); T[:, j] = z["tpm"]; C[:, j] = z["cnt"]
    keep = (T > 0).any(1)
    np.savez_compressed(os.path.join(cbb.LAY, "splice_kd_encode.npz"), tpm=T[keep], cnt=C[keep], isoform=np.array(isos)[keep],
                        gene=np.array([iso_gene[i] for i in np.array(isos)[keep]]), cols=np.array([c["file"] for c in cols]),
                        meta=np.array(json.dumps(cols)))
    info = dict(isoforms=int(keep.sum()), files=len(cols), knockdown_experiments=sum(v["role"] == "knockdown" for v in meta["experiments"].values()),
                control_experiments=sum(v["role"] == "control" for v in meta["experiments"].values()),
                targets={ln: len({v["target"] for v in meta["experiments"].values() if v["line"] == ln and v["role"] == "knockdown"}) for ln in LINES},
                knockdowns_without_files=sorted(e for e, v in meta["experiments"].items() if not v["files"]))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps(info, indent=1))


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raw_f = cbb.layer_fact("SPLICE_RAW_ENCODE", 1, "encode_rbp_kd", os.path.join(RAW, "experiments.json"), "experiment list with file accessions and md5",
                           "-", "as published (tables summed per isoform, then deleted)")
    lay = cbb.layer_fact("SPLICE_KD_ENCODE", 1, "encode_rbp_kd", os.path.join(cbb.LAY, "splice_kd_encode.npz"), "UniProt isoforms (gene)",
                         "replicate files (K562/HepG2 knockdowns and controls)", "TPM and estimated counts summed over Ensembl transcripts",
                         inputs=(raw_f.ref,), tier="derived")
    n = cbb.bank().add_many([raw_f, lay])
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={"SPLICE_RAW_ENCODE": raw_f.ref, "SPLICE_KD_ENCODE": lay.ref}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "splice_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:1500])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
