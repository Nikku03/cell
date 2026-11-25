"""ISOFORMWHEN -- descriptive (no claims): for each gene with RNA versions that change the protein's address,
(1) when and where the address-changing version is used (40 tissues, HPA transcripts) and which splicing proteins follow
it, recomputed with corrected sums; with the governors measured by knockdown (G11) beside them; and (2) the ESM C
prediction of where each address-changing version goes (G10 catalogue from the user's Colab run).

Asked by the user: "Do the first 3" -- (1) redo G9's governor list with corrected sums, (2) copy the 965 ESM C isoform
predictions into the memory bank, (3) update the cell-map page.
(1) G9's when_and_governors summed HPA nTPM over a gene's transcripts with a plain sum, so one transcript that HPA did
    not normalise (nTPM NA in every tissue; 56% of transcripts) blanked the gene, and only 19 of 212 splicing proteins
    were considered (disclosed in the ledger). Here NA versions are left out of the sums (as in G11-K2). G11-K2 showed
    that the tissue correlation does NOT pick the protein whose knockdown moves the version, so the correlated
    proteins are kept only as "follow across tissues"; the measured governors (G11; noisy) are the better answer.
(2) G10 catalogue: produced by colab/esmc/esmctest.py (declared code 6c574b9b3f1f2408) on the user's L4 and saved
    to their Drive; copied here, checked (965 rows, the columns the code writes). Predictions, not measurements; the
    G10-S1 test of these shifts was real but under its bar (AUROC 0.57 vs 0.60).

Run: python3 -I rem/atlas/isoformwhen.py governors | catalogue CSV
"""

from __future__ import annotations
import collections
import csv
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


gx = _load("generegx")
cbb, Fact = gx.cbb, gx.Fact
CLASSES = ["nuclear", "secretory", "mito", "cyto"]


def code_sha():
    return hashlib.sha256(open(os.path.join(HERE, "isoformwhen.py"), "rb").read()).hexdigest()[:16]


def governors():
    P = json.load(open(gx.L("ADDR_UNIPROT")))
    z = gx.npz("ISOFORM_HPA_TISSUE"); X = np.nan_to_num(z["X"].astype(np.float64)); tg = z["gene"].astype(str); tt = z["transcript"].astype(str)
    tissues = list(z["cols"].astype(str))
    row = collections.defaultdict(list)
    for i, (g, t) in enumerate(zip(tg, tt)):
        row[g].append((t, i))
    pw = json.load(open(gx.L("SETS_PATHWAY"))); spl = set()
    for v in pw.values():
        if v["name"] in ("mRNA Splicing", "mRNA Splicing - Major Pathway", "mRNA Splicing - Minor Pathway"):
            spl |= set(v["members"])
    gexp = {g: X[[i for _, i in row[g]]].sum(0) for g in spl if g in row}
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    measured = json.load(open(R("outputs", "splice", "governors_measured.json")))
    meas_by_gene = {v["gene"]: v for v in measured.values()}
    out = {}
    for g, v in P.items():
        ch = v.get("isoforms_changing_tag")
        if not ch or g not in row:
            continue
        acc = v["uniprot"]; iso_of = {t: i for t, i in v["transcripts"]}; changed = {f"{acc}-{k}" for k in ch if k.isdigit()}
        tx = [(t, i) for t, i in row[g] if t in iso_of]
        ch_rows = [i for t, i in tx if iso_of[t] in changed]; all_rows = [i for t, i in tx]
        if not tx or not changed or not ch_rows:
            continue
        tot = X[all_rows].sum(0); frac = np.where(tot >= 1, X[ch_rows].sum(0) / np.maximum(tot, 1e-9), np.nan)
        if np.isfinite(frac).sum() < 10 or np.nanmax(frac) - np.nanmin(frac) < 0.3:
            continue
        ok = np.isfinite(frac); cors = []
        for s_, e_ in gexp.items():
            if e_[ok].std() > 0:
                cors.append((float(np.corrcoef(frac[ok], np.log2(e_[ok] + 1))[0, 1]), sym(s_)))
        cors.sort(key=lambda x: -abs(x[0]))
        top = np.argsort(-np.nan_to_num(frac, nan=-1))[:3]; low = np.argsort(np.nan_to_num(frac, nan=2))[:3]
        m = meas_by_gene.get(g)
        out[sym(g)] = dict(gene=g, tag_changed=sorted({k for x in ch.values() for k in x}), range=round(float(np.nanmax(frac) - np.nanmin(frac)), 2),
                           tissues_most_tag_changed=[[tissues[j], round(float(frac[j]), 2)] for j in top],
                           tissues_least_tag_changed=[[tissues[j], round(float(frac[j]), 2)] for j in low if np.isfinite(frac[j])],
                           splicing_proteins_following_across_tissues=[[n, round(c, 2)] for c, n in cors[:5]],
                           governors_measured_by_knockdown=(m["governors"][:8] if m else []))
    os.makedirs(R("outputs", "address"), exist_ok=True)
    json.dump(out, open(R("outputs", "address", "isoform_when_and_governors_v2.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "isoformwhen:governors", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED | {"outputs/splice/governors_measured.json"}))
    facts = [Fact(v["gene"], "isoform/when_and_governors_v2", v, t[:10], t, "derived", src, inputs=inp,
                  context={"note": "replaces isoform/tag_changing_by_tissue (G9 sums skipped most splicing proteins); 'following across tissues' is a "
                                   "correlation that G11-K2 found does not pick the real governor; governors_measured_by_knockdown are ENCODE "
                                   "knockdowns in K562/HepG2 (noisy, direct or indirect)"}) for v in out.values()]
    n = cbb.bank().add_many(facts)
    nm = sum(1 for v in out.values() if v["governors_measured_by_knockdown"])
    print(f"{len(out)} genes whose address-changing version varies across tissues (G9 had 189); {nm} with knockdown-measured governors; {n} facts")


def catalogue(path):
    rows = list(csv.DictReader(open(path)))
    want = ["gene", "symbol", "isoform", "tag_changed", "split", "canonical_class", "isoform_class", "shift", "moves"] + \
        [f"p_canonical_{c}" for c in CLASSES] + [f"p_isoform_{c}" for c in CLASSES]
    if list(rows[0].keys()) != want or len(rows) != 965:
        raise SystemExit("not the G10 catalogue")
    res = json.load(open(R("outputs", "esmc", "esmc_tests.json")))
    if res["meta"]["code_sha"] != "6c574b9b3f1f2408" or res["meta"]["catalogue_rows"] != 965:
        raise SystemExit("results and catalogue do not match")
    os.makedirs(R("outputs", "esmc"), exist_ok=True)
    with open(R("outputs", "esmc", "G10_isoform_catalogue.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=want); w.writeheader(); w.writerows(rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["gene"]].append(dict(isoform=r["isoform"], tag_changed=r["tag_changed"].split("+"), canonical_room=r["canonical_class"],
                                  isoform_room=r["isoform_class"], shift=float(r["shift"]), predicted_to_move=r["moves"] == "1",
                                  p_isoform={c: float(r[f"p_isoform_{c}"]) for c in CLASSES}))
    t = cbb.now(); src = {"name": "esmctest:G10 catalogue", "code_sha": "6c574b9b3f1f2408", "model": "biohub/ESMC-600M", "ran_on": res["meta"]["gpu"]}
    sym = {r["gene"]: r["symbol"] for r in rows}
    facts = [Fact(g, "isoform/esmc_predicted_room", dict(symbol=sym[g], versions=v), t[:10], t, "derived", src,
                  inputs=("outputs/esmc/G10_isoform_catalogue.csv",),
                  context={"note": "ESM C 600M prediction of the room class of each address-changing RNA version vs the main version; G10-S1 found the "
                                   "shifts real but under the bar (AUROC 0.57 vs 0.60); a prediction, not a measurement"}) for g, v in by.items()]
    n = cbb.bank().add_many(facts)
    print(f"{len(rows)} versions of {len(by)} genes; {sum(r['moves'] == '1' for r in rows)} predicted to move; {n} facts")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"governors": governors, "catalogue": lambda: catalogue(a[1])}[a[0]]()
