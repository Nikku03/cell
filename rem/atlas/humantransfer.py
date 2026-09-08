"""Does the yeast promoter result transfer to human cells? The weakest link, tested.

WHAT IS AT STAKE. Every accuracy figure in this build order comes from ONE experiment: Sharon et
al. 2012, ~5,800 designed yeast promoters carrying synthetic transcription-factor sites. From it
come the refutation of the count summary (8.89 noise floors), the refutation of the signed count,
the class ladder that says C = 64 classes are needed to sit inside one floor, and therefore every
cap in the running total -- 3, 13, 140 -- since the cap is only meaningful at a stated accuracy.
Those numbers have been applied throughout to a HUMAN regulatory network, TRRUST. Four separate
modules flag that transfer as the weakest link in the chain and none of them tests it. This does.

THE TEST HAS TO BE MATCHED, NOT MERELY HUMAN. The yeast experiment asks whether a promoter's
output depends on WHICH factors are bound or only on HOW MANY. The human analogue is not another
reporter assay -- it is the same question asked of real regulatory relationships: for a gene with
several known regulators, does it matter WHICH one you perturb, or are they interchangeable?

THE DATA. ENCODE CRISPRi RNA-seq in K562: each experiment silences one transcription factor and
measures the whole transcriptome, with two biological replicates. Restricted to factors that
TRRUST lists as regulators, that is 52 factors, and 284 TRRUST target genes have two or more of
them perturbed, 106 have three or more, and the most-regulated has eight. A single assay type in a
single cell line, so effect sizes are comparable across factors -- mixing shRNA, CRISPR and CRISPRi
would have confounded the very comparison being made.

NO CONTROL EXPERIMENT IS NEEDED, AND THAT IS NOT A SHORTCUT. Each gene's response is centred
across the perturbation panel, which removes exactly the per-gene constant a control would supply,
and that constant cancels anyway in a comparison BETWEEN regulators of the same gene.

HOW THE FLOOR AND THE HELD-OUT SPLIT ARE THE SAME OBJECT HERE. In the yeast experiment the two
replicates gave the noise floor and the held-out test was over promoters. Here each (gene,
regulator) pair is measured once per replicate, so the identity model has nothing to generalise
over -- predicting replicate 2 from replicate 1 for the SAME pair IS the replicate noise. That is
not a defect of the design, it is what makes the comparison sharp: the question becomes how much
WORSE a regulator-blind model is than the noise floor, on the same held-out split. A gene whose
response does not depend on which regulator was hit will sit AT the floor.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

H0  THE CEILING GATE, ALREADY RUN BEFORE THIS FILE EXISTED. Can human data resolve the question at
    all? Measured on the ENCODE catalogue: 52 TRRUST regulators perturbed in K562 CRISPRi, 284
    targets with >= 2 of them, 106 with >= 3. PREDECLARED at the time: fewer than ~50 targets with
    two or more perturbed regulators and the data cannot settle it and nothing is downloaded.

H1  THE NOISE FLOOR, from the two biological replicates, measured BEFORE any model is fitted, so
    it cannot be chosen to suit the answer. And it must be the noise in the quantity actually
    scored -- the mistake this build order made in the yeast module and had to correct.

H2  THE BASELINE THAT MUST FAIL. A model that knows nothing about the target gene must be clearly
    worse than one that does. If it is not, these data are not measuring regulation and nothing
    below is readable.

H3  THE HEAD-TO-HEAD, held out across replicates: a REGULATOR-BLIND model, which predicts a gene's
    response from the gene alone, against the replicate-noise floor.
    PREDECLARED: the yeast result TRANSFERS if the regulator-blind model is worse than the floor
    by more than one floor, and FAILS TO TRANSFER if it sits within one floor -- because that
    would mean that in human cells the regulators of a gene ARE interchangeable, which is exactly
    what the count summary assumes and what yeast refuted.

H4  THE CLASS LADDER, the direct analogue of signed.py's S1. Partition the regulators into C
    classes by their fitted effect and predict from the class mean, C from 1 to all. Classes are
    fitted on one replicate and scored on the other, never both.
    PREDECLARED: yeast needed C = 64 of 404 factors -- 16% of the alphabet -- to sit inside one
    floor. If human needs a similar FRACTION, the transfer holds and the caps stand. If it needs
    far fewer, every cap in this build order is too pessimistic. If far more, they are too
    optimistic and the 140 is not real.

H5  TRRUST's OWN SIGN AS THE CLASS, which is the one class label that exists for this network
    without being fitted. The signed count was refuted on yeast at 3.70 floors; this asks whether
    it is refuted on human too, using real annotations rather than a learned split.

H6  WHAT DIFFERS BETWEEN THE TWO EXPERIMENTS, and which differences could produce the answer by
    themselves. A transfer test that does not enumerate its own confounds is a press release.

H7  WHAT THIS DOES AND DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import gzip
import hashlib
import json
import urllib.request
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.trrust_engine import signed_edges

CACHE = os.path.join(os.path.dirname(__file__), "encode_k562_crispri.npz")
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
ENC = "https://www.encodeproject.org"


def _get(url):
    r = urllib.request.Request(url, headers={"Accept": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=180))


def symbol_map(path=None):
    """HGNC symbol <-> Ensembl gene id. Not vendored: fetched and checksummed."""
    path = path or os.path.join(os.path.dirname(__file__), "hgnc_complete_set.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(HGNC_URL, path)
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()[:32]
    head = raw.decode("utf8", "replace").splitlines()
    cols = head[0].split("\t")
    si, ei = cols.index("symbol"), cols.index("ensembl_gene_id")
    s2e, e2s = {}, {}
    for ln in head[1:]:
        f = ln.split("\t")
        if len(f) > max(si, ei) and f[ei]:
            s2e[f[si]] = f[ei]
            e2s[f[ei]] = f[si]
    return s2e, e2s, sha


def build_cache(cell="K562", assay="CRISPRi RNA-seq", path=None):
    """Fetch the ENCODE panel and reduce it, on the fly, to the genes TRRUST names.

    The gene quantifications are ~10 MB each and there are two per experiment. Each is streamed,
    reduced to the rows that matter, and DELETED before the next is fetched, so peak disk stays at
    one file rather than a gigabyte."""
    path = path or CACHE
    if os.path.exists(path):
        return path
    s2e, e2s, sha_h = symbol_map()
    E, _ = signed_edges()
    want_sym = {u for u, v, _ in E} | {v for u, v, _ in E}
    want_ens = {s2e[s] for s in want_sym if s in s2e}
    url = (f"{ENC}/search/?type=Experiment&assay_title={assay.replace(' ', '+')}"
           f"&biosample_ontology.term_name={cell}&status=released&limit=all&format=json"
           f"&field=accession&field=target.label&field=files.@id")
    cat = _get(url)
    tfs_trrust = {u for u, v, _ in E}
    exps = []
    for g in cat.get("@graph", []):
        t = (g.get("target") or {}).get("label")
        if t and t in tfs_trrust:
            exps.append((t, g["accession"]))
    exps.sort()
    genes = sorted(want_ens)
    gidx = {g: i for i, g in enumerate(genes)}
    tf_names, R1, R2 = [], [], []
    tmp = path + ".part.tsv"
    for t, acc in exps:
        try:
            d = _get(f"{ENC}/experiments/{acc}/?format=json")
        except Exception:
            continue
        byrep = {}
        for f in d.get("files", []):
            if (f.get("output_type") == "gene quantifications" and f.get("file_format") == "tsv"
                    and f.get("assembly") == "GRCh38"):
                br = f.get("biological_replicates") or []
                if len(br) == 1:
                    byrep.setdefault(br[0], []).append(f["accession"])
        reps = sorted(byrep)
        if len(reps) < 2:
            continue
        cols = []
        ok = True
        for rp in reps[:2]:
            fa = sorted(byrep[rp])[0]
            try:
                urllib.request.urlretrieve(f"{ENC}/files/{fa}/@@download/{fa}.tsv", tmp)
            except Exception:
                ok = False
                break
            v = np.full(len(genes), np.nan)
            with open(tmp) as fh:
                hdr = fh.readline().rstrip("\n").split("\t")
                gi, ti = hdr.index("gene_id"), hdr.index("TPM")
                for ln in fh:
                    p = ln.split("\t")
                    if len(p) <= ti:
                        continue
                    g = p[gi].split(".")[0]
                    j = gidx.get(g)
                    if j is not None:
                        try:
                            v[j] = float(p[ti])
                        except ValueError:
                            pass
            os.remove(tmp)
            cols.append(v)
        if not ok or len(cols) < 2:
            continue
        tf_names.append(t)
        R1.append(cols[0])
        R2.append(cols[1])
        print(f"  {t:<10} {acc}  {len(tf_names)} factors cached", flush=True)
    np.savez_compressed(path, genes=np.array(genes), tfs=np.array(tf_names),
                        r1=np.array(R1), r2=np.array(R2), hgnc_sha=sha_h)
    return path


def load_panel(path=None):
    z = np.load(path or CACHE, allow_pickle=False)
    return (list(z["genes"]), list(z["tfs"]), z["r1"], z["r2"], str(z["hgnc_sha"]))
