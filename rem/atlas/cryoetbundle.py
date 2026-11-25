"""CRYOETBUNDLE -- the fixed run list and RNA-based prediction for the ribosome test (G14, colab/cryoet/ribotest.py, run
on the user's Colab GPU), its declarations, and recording the results that come back.

Asked by the user: "do the two partner checks, then make the colab notebook" (count ribosomes in Jurkat and RPE1
tomograms of the CryoET Data Portal as a first direct test of "is the machine built").
Runs (public bucket s3://cryoet-data-portal-public; only paths and metadata are written here, no image data):
  ground truth: dataset 10426 (HSC-3 unroofed cells), every run with a hand-picked ("manual", ground truth) cytosolic
    ribosome annotation; DEV = the 3 runs with the smallest sha256('cellbrain-cryoet:' + run), EXAM = the rest;
  cells: Jurkat 10169, 10170, 10171 (first 4 runs of each, sorted by name); RPE1 10002 (all 3); HeLa 10476
    (untreated control, first 6 runs; reported only).
  For each run: the visualisation-default tomogram, the OME-Zarr level whose voxel is closest to 25 A, the run's
  tilt range, and a membrane mask at the same level (MemBrain-seg prediction if present, else "membrane").
Prediction (frozen here): share of ribosomes making ER proteins = sum over nuclear-encoded UniProt genes with a signal
peptide or transmembrane segment of HPA nTPM x protein length / the same over all genes (EXPR_HPA_CELLLINES).

Run: python3 -I rem/atlas/cryoetbundle.py build | declare | record RESULTS.json
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


gx = _load("generegx")
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
COLAB = R("colab", "cryoet")
OUT = R("outputs", "cryoet")
B = "https://cryoet-data-portal-public.s3.us-west-2.amazonaws.com/"
CLAIMS = ["G14-R1", "G14-R2"]
cbb.SRC["cryoet_portal"] = ("CryoET Data Portal (Ermel et al. 2024 Nat Methods), datasets 10426, 10169-10171, 10002, 10476", "https://cryoetdataportal.czscience.com",
                            "no data licence stated (terms: public, as is) -- local only")
cbb.LOCAL_ONLY.add("cryoet_portal")
CELLS = [("Jurkat", "10169", 4), ("Jurkat", "10170", 4), ("Jurkat", "10171", 4), ("RPE1", "10002", 3), ("HeLa", "10476", 6)]


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cellbrain/1.0"}), timeout=120).read()


def ls(prefix):
    out, tok = [], None
    while True:
        q = {"list-type": "2", "delimiter": "/", "prefix": prefix, "max-keys": "1000"}
        if tok:
            q["continuation-token"] = tok
        x = get(B + "?" + urllib.parse.urlencode(q)).decode()
        out += [("dir", p) for p in re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", x)]
        out += [("file", k) for k in re.findall(r"<Key>([^<]*)</Key>", x)]
        m = re.search(r"<NextContinuationToken>([^<]*)<", x)
        if not m:
            return out
        tok = m.group(1)


def run_entry(run):
    """tomogram level prefix, voxel, tilt range, membrane mask prefix, annotations of one run."""
    vsd = [p for t, p in ls(run + "Reconstructions/") if t == "dir"][0]; vs = float(vsd.rstrip("/").split("VoxelSpacing")[1])
    tomos = [p for t, p in ls(vsd + "Tomograms/") if t == "dir"]; pick = None
    for tdir in tomos:
        m = json.loads(get(B + tdir + "tomogram_metadata.json"))
        if m.get("is_visualization_default"):
            pick = (tdir, m); break
    pick = pick or (tomos[0], json.loads(get(B + tomos[0] + "tomogram_metadata.json")))
    zdir = [p for t, p in ls(pick[0]) if t == "dir" and p.endswith(".zarr/")][0]
    L = int(np.argmin([abs(vs * 2 ** k - 25.0) for k in range(3)]))
    ts = [p for t, p in ls(run + "TiltSeries/") if t == "dir"]
    tilt = 60.0
    if ts:
        tm = json.loads(get(B + ts[0] + "tiltseries_metadata.json")); tr = tm.get("tilt_range") or {}
        if tr:
            tilt = float(max(abs(tr.get("min", -60)), abs(tr.get("max", 60))))
    anns = []
    for t, a in ls(vsd + "Annotations/"):
        if t != "dir":
            continue
        for t2, f in ls(a):
            if t2 == "file" and f.endswith(".json"):
                anns.append(json.loads(get(B + f)))
    mem = None
    for want in ("membrain_seg_prediction", "membrane-1.0_segmentationmask"):
        for m in anns:
            for fl in m.get("files", []):
                if fl.get("format") == "zarr" and want in fl.get("path", "") and m.get("annotation_object", {}).get("name") == "membrane":
                    mem = mem or fl["path"].rstrip("/") + f"/{L}/"
    return dict(run=run.rstrip("/").split("/")[-1], dataset=run.split("/")[0], tomo_prefix=f"{zdir}{L}/", vox=round(vs * 2 ** L, 4), vox_folder=vs,
                tilt_max=tilt, membrane_prefix=mem), anns


def prediction():
    P = json.load(open(gx.L("ADDR_UNIPROT"))); H = cbb.hgnc(); K = sorted(P)
    E = gx.hpa_expr(K); out = {}
    er = np.array([bool(P[g]["signal"]) or P[g]["tm"] > 0 for g in K]); ln = np.array([P[g]["length"] for g in K], float)
    nuc = np.array([not H["e2s"].get(g, "").startswith("MT-") for g in K])
    for line, col in (("K562", 0), ("RPE1", 1), ("HepG2", 2), ("Jurkat", 3)):
        e = np.nan_to_num(E[:, col]) * nuc; w = e * ln
        out[line] = round(float(w[er].sum() / w.sum()), 4)
    return out


def build():
    os.makedirs(COLAB, exist_ok=True)
    gt = []
    for t, run in ls("10426/"):
        if t != "dir" or run.endswith("Images/"):
            continue
        e, anns = run_entry(run)
        man = [m for m in anns if m.get("ground_truth_status") and "ribosome" in m.get("annotation_object", {}).get("name", "")]
        if not man:
            continue
        e["gt_path"] = [f["path"] for f in man[0]["files"] if f.get("format") == "ndjson"][0]; e["gt_vox"] = e["vox_folder"]
        auto = [m for m in anns if m.get("method_type") == "automated" and m.get("annotation_object", {}).get("name") == "cytosolic ribosome"]
        e["portal_auto_path"] = ([f["path"] for f in auto[0]["files"] if f.get("format") == "ndjson"] or [None])[0] if auto else None
        gt.append(e); print("ground truth", e["run"], e["vox"], e["tilt_max"], flush=True)
    order = sorted(gt, key=lambda e: hashlib.sha256(("cellbrain-cryoet:" + e["run"]).encode()).hexdigest())
    for k, e in enumerate(order):
        e["split"] = "dev" if k < 3 else "exam"
    cells = []
    for line, ds, n in CELLS:
        runs = sorted(p for t, p in ls(ds + "/") if t == "dir" and not p.endswith("Images/"))[:n]
        for run in runs:
            e, _ = run_entry(run); e["line"] = line; cells.append(e); print(line, e["run"], e["vox"], e["tilt_max"], "mask" if e["membrane_prefix"] else "no mask", flush=True)
    bun = dict(ground_truth=order, cells=cells, prediction=prediction(), template="EMD-54265", working_voxel_A=25.0,
               note="paths are in the public CryoET Data Portal bucket; no image data is stored here")
    json.dump(bun, open(os.path.join(COLAB, "bundle.json"), "w"), indent=1, sort_keys=True)
    print("prediction (share of ribosomes making ER proteins):", bun["prediction"])


def code_sha():
    h = hashlib.sha256(open(os.path.join(COLAB, "ribotest.py"), "rb").read()); h.update(open(os.path.join(COLAB, "bundle.json"), "rb").read())
    return h.hexdigest()[:16]


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    doc = open(os.path.join(COLAB, "ribotest.py")).read().split('"""')[1]; have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=doc.split(" " + cid)[1].split(" G14-")[0].split("REPORTED:")[0].strip()[:1500], trial=False), sort_keys=True) + "\n")
            print(cid, "declared")


def record(path):
    res = json.load(open(path)); meta = res["meta"]
    dec = {r["id"]: r for r in _read() if r["op"] == "declare" and r["id"] in CLAIMS}
    if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
        raise SystemExit("G14 already recorded")
    for cid in CLAIMS:
        if cid not in dec or dec[cid]["code_sha"] != meta["code_sha"]:
            raise SystemExit(f"{cid}: results from code {meta['code_sha']}, declared {dec.get(cid, {}).get('code_sha')}")
    if meta.get("trial"):
        raise SystemExit("trial run")
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "ribosome_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "ribotest:G14", "code_sha": meta["code_sha"], "ran_on": meta.get("gpu")}
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "cryoet_test", res[cid], t[:10], t, "derived", src, inputs=("cryoet_portal",)) for cid in CLAIMS])
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=meta["code_sha"], ran_on=meta.get("gpu"), ran_at=meta.get("finished"), **res[cid]),
                               sort_keys=True, default=float) + "\n")
            print(cid, res[cid]["gate"]["verdict"])


if __name__ == "__main__":
    a = sys.argv[1:]
    {"build": build, "declare": declare, "record": lambda: record(a[1])}[a[0]]()
