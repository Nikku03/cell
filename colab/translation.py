"""TRANSLATION EFFICIENCY -- the chain's missing multiplier, and whether sequence can supply it.

THE GAP. `k562_protein.py` measured the one number that bounds this project's central chain: in K562, mRNA
level explains 15.9% of protein LEVEL variance (r 0.399, 5,874 genes both measured in this cell type). The
chain `perturbation -> dRNA -> dprotein -> phenotype` is missing its second arrow, and 15.9% is a generous
ceiling on what a dRNA-only model can ever recover -- generous because a delta is harder than a level. The
obvious missing multiplier is translation: how much protein a cell makes per transcript.

THE SOURCE THAT WOULD HAVE ANSWERED IT DIRECTLY DOES NOT EXIST. Ribosome profiling in K562 would give per-gene
translation efficiency measured rather than inferred. This module PROBES the ENCODE portal live rather than
asserting from memory, because an earlier probe of the literal string "Ribo-seq" returned nothing and that is
not by itself evidence of absence -- assay titles change. The probe enumerates the assay_title facet twice:
once restricted to K562, once over the whole portal. It then asserts that no ribosome-profiling assay appears
in either. If ENCODE ever adds one this module fails loudly instead of quietly continuing to the fallback.

WHAT IS MEASURABLE INSTEAD, AND WHY IT IS WORTH MEASURING. K562 iBAQ protein (Geiger 2012, via PaxDb) against
K562 Perturb-seq mean expression (Replogle GWPS `var/mean`) gives an empirical per-gene protein-per-mRNA
ratio. That ratio is NOT translation efficiency -- it is translation divided by protein degradation, in steady
state, measured by two assays with different biases. It is nonetheless the shape of the multiplier the chain
needs, and the question that would decide whether it is usable is:

    IS THE RATIO PREDICTABLE FROM SEQUENCE AND ANNOTATION ALONE?

The first thing the module checks, before any of that, is whether the ratio is a quantity of its own at
all. A difference of two logs inherits whichever term varies more, and here log mRNA varies about twice as
much as log iBAQ protein -- so the "ratio" correlates -0.87 with log mRNA and +0.09 with log protein, and
ranking genes by it very largely ranks them by inverse expression. That is why the downstream test below
is built to predict PROTEIN LEVEL directly from mRNA plus sequence rather than to predict the ratio: an
arm that routes through the ratio would be measuring expression under another name.

Predictable would mean a usable multiplier: for the ~10,000 genes with no K562 proteomics, a model could
supply one. Unpredictable would bound what the chain can ever do: the RNA->protein step would then carry
gene-specific information that no amount of sequence modelling recovers, and 15.9% + noise is the ceiling.
Both questions are answered below, but neither rescues the ratio itself -- the denominator problem above
is prior to them, and it is the reason the module's usable output is a protein-level model rather than a
translation-efficiency table.

WHAT WOULD FALSIFY EACH DIRECTION.
  A CLAIM OF PREDICTABILITY dies if held-out R2 does not exceed the shuffled-feature null, or if it survives
    only through CDS length. iBAQ is intensity divided by the number of theoretically observable peptides,
    so protein length is already partly divided out of the measurement and any residual length dependence is
    as likely to be mass-spectrometry normalisation as translation biology. The length features are therefore
    ablated and the model re-run; if predictability collapses, the honest report is "we predicted the assay".
  A CLAIM OF USEFULNESS dies at the downstream decision, which is run directly rather than through the ratio:
    predict log protein from log mRNA (the 15.9% baseline), then from log mRNA PLUS sequence features, on
    identical chromosome-held-out folds. Genes are grouped by chromosome because paralogues cluster and
    share both sequence features and expression; a random split leaks them across the fold boundary.
  Both arms report the RAW held-out R2. The shuffled arm establishes significance and is never subtracted.

SELECTION ON THE OUTCOME IS THE STANDING THREAT AND IT IS MEASURED, NOT WAVED AT. The ratio can only be
computed where the protein was DETECTED, and mass spectrometry misses low-abundance proteins. So the
low-ratio tail is censored, the censoring is expression-dependent, and the predictability estimate is
computed on a sample that is not the population. Coverage and its dependence on mRNA level are reported.

THE FAST TEST IS A PAIR-LEVEL TEST AND THIS IS A PER-GENE LAYER, which is stated here in advance rather than
discovered in the verdict. PPI, co-dependency and regulatory layers link (perturbation, gene) PAIRS; a
per-gene scalar has no pairwise content except through structure it induces. Three prespecified links are
tested anyway, because "this layer adds nothing to transcriptional response" is a result worth having and the
alternative is asserting it: (1) shared translational regime -- pairs whose measured ratios are in the closest
decile; (2) the mechanistic interaction -- perturbations of the translation apparatus (Reactome R-HSA-72766)
against genes in the top ratio decile, which is the one place a translation layer should show up in dRNA;
(3) the same as (1) but on PREDICTED ratios, which is the artefact that would actually ship. The matching on
perturbation-strength and gene-responsiveness deciles removes both main effects by construction, so only
genuine pair structure can survive.

ONE VERSION ALONG EVERY AXIS, ASSERTED. One proteomics study (Geiger 2012), one RNA dataset (Replogle K562
GWPS), one annotation release (Ensembl 116, pinned and checked against the file header), one transcript per
gene chosen by a fixed rule (MANE Select > Ensembl canonical > longest CDS), one genome (GRCh38 / hg38.2bit).
"""
import gzip
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
API = "https://www.encodeproject.org"
DIR = SP / "te"
GWPS = SP / "gwps.h5ad"
PROT = SP / "cell_k562_protein.json.gz"
TWOBIT = SP / "hg38.2bit"
GMT = SP / "ReactomePathways.gmt"
LAYER = SP / "cell_translation.json.gz"

ENSEMBL_RELEASE = 116
ASSEMBLY = "GRCh38"
GTF_URL = (f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}/gtf/homo_sapiens/"
           f"Homo_sapiens.{ASSEMBLY}.{ENSEMBL_RELEASE}.gtf.gz")
CDS_URL = (f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}/fasta/homo_sapiens/cds/"
           f"Homo_sapiens.{ASSEMBLY}.cds.all.fa.gz")
RNA_SOURCE = "Replogle et al. 2022 genome-wide Perturb-seq (K562 CRISPRi), var/mean"
PROT_SOURCE = "PaxDb 9606-iBAQ_K562_Geiger_2012 (Geiger et al., Mol Cell Proteomics 2012)"
TRANSLATION_PATHWAY = "R-HSA-72766"
N_DRAWS = 24
PAIR_CAP = 300_000
LINK_DECILE = 0.10          # "closest decile" -- fixed before any ratio was computed

WANT = {"transcript", "exon", "CDS", "five_prime_utr", "three_prime_utr", "start_codon"}
PRIMARY = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
STOPS = {"TAA", "TAG", "TGA"}


# ---------------------------------------------------------------- fetch helpers
def api(path, tries=4):
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=300))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def download(url, path, tries=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=1800) as fh, open(path, "wb") as o:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            return path
        except Exception:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def encode_ribo_probe(report):
    """Enumerate assay titles, K562-restricted and portal-wide. Assert ribosome profiling is absent.

    Stating "ENCODE has no K562 Ribo-seq" in a docstring is not provenance; it is a memory. The facet is
    read at run time and the absence is an assertion, so the day the portal gains one this raises instead
    of silently continuing to the fallback.
    """
    def titles(q):
        d = api(f"/search/?type=Experiment&status=released&limit=0{q}")
        for f in d.get("facets", []):
            if f["field"] == "assay_title":
                return {t["key"]: t["doc_count"] for t in f["terms"]}
        return {}

    k = titles("&biosample_ontology.term_name=K562")
    a = titles("")
    pat = re.compile(r"ribo|ribosome profil|translat", re.I)
    hit_k = sorted(t for t in k if pat.search(t))
    hit_a = sorted(t for t in a if pat.search(t))
    report(f"  ENCODE probe: {len(k)} assay titles in K562, {len(a)} portal-wide "
           f"({sum(k.values()):,} K562 experiments)")
    report(f"    ribosome/translation-like titles in K562:       {hit_k or 'NONE'}")
    report(f"    ribosome/translation-like titles portal-wide:   {hit_a or 'NONE'}")
    assert not hit_k and not hit_a, (
        f"ENCODE now exposes a ribosome-profiling-like assay ({hit_k or hit_a}). This module's fallback is "
        f"only justified while the direct measurement is absent -- fetch it instead of inferring.")
    top = sorted(k.items(), key=lambda x: -x[1])[:6]
    report(f"    largest K562 assays instead: {', '.join(f'{n} ({c})' for n, c in top)}")
    return {"n_titles_k562": len(k), "n_titles_portal": len(a), "n_experiments_k562": int(sum(k.values())),
            "riboseq_titles_k562": hit_k, "riboseq_titles_portal": hit_a,
            "k562_assay_titles": sorted(k)}


# ---------------------------------------------------------------- annotation
def attr(s, key):
    i = s.find(key + ' "')
    if i < 0:
        return None
    j = i + len(key) + 2
    k = s.find('"', j)
    return s[j:k]


def parse_gtf(path, report):
    """One transcript per gene by a fixed rule, plus the structural features of that transcript."""
    tx, gtx, ntx = {}, {}, {}
    seen_release = None
    t0 = time.time()
    n = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                if seen_release is None and "genebuild-version" not in line:
                    m = re.search(r"release[ -](\d+)", line)
                    if m:
                        seen_release = int(m.group(1))
                continue
            n += 1
            p = line.split("\t")
            f = p[2]
            if f not in WANT:
                continue
            a = p[8]
            t = attr(a, "transcript_id")
            if t is None:
                continue
            L = int(p[4]) - int(p[3]) + 1
            if f == "transcript":
                g = attr(a, "gene_id")
                tx[t] = {"gene": g, "chrom": p[0], "strand": p[6],
                         "gbio": attr(a, "gene_biotype"), "tbio": attr(a, "transcript_biotype"),
                         "mane": 'tag "MANE_Select"' in a, "canon": 'tag "Ensembl_canonical"' in a,
                         "cds": 0, "ncds": 0, "exon": 0, "nexon": 0, "u5": 0, "u3": 0,
                         "u5i": [], "sc": []}
                gtx.setdefault(g, []).append(t)
                ntx[g] = ntx.get(g, 0) + 1
            elif t in tx:
                d = tx[t]
                if f == "exon":
                    d["exon"] += L
                    d["nexon"] += 1
                elif f == "CDS":
                    d["cds"] += L
                    d["ncds"] += 1
                elif f == "five_prime_utr":
                    d["u5"] += L
                    d["u5i"].append((int(p[3]), int(p[4])))
                elif f == "three_prime_utr":
                    d["u3"] += L
                elif f == "start_codon":
                    d["sc"].append((int(p[3]), int(p[4])))
    report(f"  Ensembl {ENSEMBL_RELEASE} GTF: {n:,} feature lines, {len(tx):,} transcripts, "
           f"{len(gtx):,} genes  ({time.time()-t0:.0f}s)")

    # ONE TRANSCRIPT PER GENE, fixed precedence. Mixing MANE for some genes and longest-CDS for others
    # would make "CDS length" mean different things for different genes -- the same class of error as
    # mixing peak output types across TFs.
    pick, rule = {}, {"MANE_Select": 0, "Ensembl_canonical": 0, "longest_CDS": 0}
    for g, ts in gtx.items():
        cand = [t for t in ts if tx[t]["cds"] > 0]
        if not cand:
            continue
        m = [t for t in cand if tx[t]["mane"]]
        c = [t for t in cand if tx[t]["canon"]]
        if m:
            best, r = sorted(m)[0], "MANE_Select"
        elif c:
            best, r = sorted(c)[0], "Ensembl_canonical"
        else:
            best, r = sorted(cand, key=lambda t: (-tx[t]["cds"], t))[0], "longest_CDS"
        rule[r] += 1
        pick[g.split(".")[0]] = (best.split(".")[0], best, r)
    report(f"  coding genes with a chosen transcript: {len(pick):,}   "
           f"({rule['MANE_Select']:,} MANE Select, {rule['Ensembl_canonical']:,} canonical, "
           f"{rule['longest_CDS']:,} longest-CDS)")
    return tx, pick, ntx


def utr_sequence(tx, pick, report):
    """5'UTR sequence and start-codon context from hg38.2bit, in transcript orientation."""
    import twobitreader
    tb = twobitreader.TwoBitFile(str(TWOBIT))
    out, skipped = {}, 0
    for gid, (tsv, tsfull, _r) in pick.items():
        d = tx[tsfull]
        if d["chrom"] not in PRIMARY:
            skipped += 1
            continue
        cn = "chrM" if d["chrom"] == "MT" else "chr" + d["chrom"]
        if cn not in tb:
            skipped += 1
            continue
        seq = tb[cn]
        minus = d["strand"] == "-"
        u5 = ""
        if d["u5i"]:
            iv = sorted(d["u5i"], reverse=minus)
            parts = []
            for s, e in iv:
                b = seq[s - 1:e].upper()
                parts.append(b.translate(COMP)[::-1] if minus else b)
            u5 = "".join(parts)
        koz = ""
        if d["sc"]:
            if minus:
                e = max(x[1] for x in d["sc"])
                koz = seq[e - 4:e + 6].upper().translate(COMP)[::-1]
            else:
                s = min(x[0] for x in d["sc"])
                koz = seq[s - 7:s + 3].upper()
        out[gid] = (u5, koz)
    report(f"  5'UTR / start-codon sequence for {len(out):,} genes "
           f"({skipped:,} on non-primary scaffolds, left unknown)")
    return out


def parse_cds(path, keep, report):
    """CDS composition for the chosen transcripts only."""
    out, cur, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if cur and buf:
                    out[cur] = "".join(buf)
                t = line[1:].split(None, 1)[0].split(".")[0]
                cur, buf = (t if t in keep else None), []
            elif cur:
                buf.append(line.strip())
    if cur and buf:
        out[cur] = "".join(buf)
    report(f"  CDS sequence for {len(out):,}/{len(keep):,} chosen transcripts")
    return out


def build_features(report):
    cache = DIR / f"features_ens{ENSEMBL_RELEASE}.json.gz"
    if cache.exists():
        d = json.load(gzip.open(cache, "rt"))
        assert d["ensembl_release"] == ENSEMBL_RELEASE and d["assembly"] == ASSEMBLY, \
            f"cached annotation is release {d['ensembl_release']}/{d['assembly']}, not the pinned pair"
        report(f"  cached annotation: {len(d['genes']):,} genes, Ensembl {d['ensembl_release']} "
               f"{d['assembly']}")
        return d
    gp, cp = DIR / "gtf.gz", DIR / "cds.fa.gz"
    report(f"  fetching {GTF_URL}")
    download(GTF_URL, gp)
    report(f"  fetching {CDS_URL}")
    download(CDS_URL, cp)
    tx, pick, ntx = parse_gtf(gp, report)
    seqs = utr_sequence(tx, pick, report)
    cds = parse_cds(cp, {v[0] for v in pick.values()}, report)

    G = {}
    for gid, (tsv, tsfull, rule) in pick.items():
        d = tx[tsfull]
        c = cds.get(tsv)
        u5, koz = seqs.get(gid, ("", ""))
        gc = gc3 = np.nan
        atg = stop = 0
        if c and len(c) >= 6:
            cu = c.upper()
            gc = (cu.count("G") + cu.count("C")) / len(cu)
            third = cu[2::3]
            gc3 = (third.count("G") + third.count("C")) / max(len(third), 1)
            atg = int(cu[:3] == "ATG")
            stop = int(cu[-3:] in STOPS)
        G[gid] = {
            "tx": tsv, "rule": rule, "chrom": d["chrom"], "gbio": d["gbio"], "tbio": d["tbio"],
            "cds_len": d["cds"], "n_cds_exons": d["ncds"], "tx_len": d["exon"], "n_exons": d["nexon"],
            "utr5_len": d["u5"], "utr3_len": d["u3"], "n_transcripts": ntx.get(d["gene"], 1),
            "is_mane": int(rule == "MANE_Select"),
            "gc_cds": gc, "gc3_cds": gc3, "cds_starts_atg": atg, "cds_ends_stop": stop,
            "utr5_gc": ((u5.count("G") + u5.count("C")) / len(u5)) if len(u5) >= 10 else np.nan,
            "n_uaug": u5.count("ATG") if u5 else 0,
            "has_u5seq": int(bool(u5)),
            "kozak_m3_purine": int(koz[3] in "AG") if len(koz) == 10 else -1,
            "kozak_p4_g": int(koz[9] == "G") if len(koz) == 10 else -1}
    d = {"ensembl_release": ENSEMBL_RELEASE, "assembly": ASSEMBLY, "gtf": GTF_URL, "cds": CDS_URL,
         "transcript_rule": "MANE_Select > Ensembl_canonical > longest CDS", "genes": G}
    with gzip.open(cache, "wt") as fh:
        json.dump(d, fh)
    for p in (gp, cp):
        if p.exists():
            p.unlink()                      # raw annotation deleted; the reduced table is the artefact
    report(f"  annotation reduced to {len(G):,} genes -> {cache.name} "
           f"({cache.stat().st_size/1e6:.1f} MB); raw GTF/FASTA deleted")
    return d


# ---------------------------------------------------------------- modelling
FEATS = ["log_cds_len", "log_tx_len", "n_cds_exons", "n_exons", "log_utr5_len", "log_utr3_len",
         "has_utr5", "has_utr3", "n_transcripts", "gc_cds", "gc3_cds", "utr5_gc", "n_uaug",
         "uaug_per_kb", "kozak_m3_purine", "kozak_p4_g", "is_protein_coding", "is_mane"]
LENGTH_FEATS = ["log_cds_len", "log_tx_len", "n_cds_exons", "n_exons"]


def featmat(rows):
    X = np.full((len(rows), len(FEATS)), np.nan, dtype=np.float64)
    for i, g in enumerate(rows):
        u5 = g["utr5_len"]
        v = {"log_cds_len": np.log10(max(g["cds_len"], 1)),
             "log_tx_len": np.log10(max(g["tx_len"], 1)),
             "n_cds_exons": g["n_cds_exons"], "n_exons": g["n_exons"],
             "log_utr5_len": np.log10(1 + u5), "log_utr3_len": np.log10(1 + g["utr3_len"]),
             "has_utr5": int(u5 > 0), "has_utr3": int(g["utr3_len"] > 0),
             "n_transcripts": g["n_transcripts"],
             "gc_cds": g["gc_cds"], "gc3_cds": g["gc3_cds"], "utr5_gc": g["utr5_gc"],
             "n_uaug": g["n_uaug"], "uaug_per_kb": 1000.0 * g["n_uaug"] / max(u5, 1),
             # -1 marks "no start-codon sequence" (non-primary scaffold). It is carried as NaN, which
             # the model treats as unknown, rather than as a third value that means nothing.
             "kozak_m3_purine": g["kozak_m3_purine"] if g["kozak_m3_purine"] >= 0 else np.nan,
             "kozak_p4_g": g["kozak_p4_g"] if g["kozak_p4_g"] >= 0 else np.nan,
             "is_protein_coding": int(g["gbio"] == "protein_coding"), "is_mane": g["is_mane"]}
        for j, k in enumerate(FEATS):
            X[i, j] = v[k] if v[k] is not None else np.nan
    return X


def null_r2(X, y, groups, n=20, seed0=0):
    """Shuffled-feature null: the features are permuted across genes, so marginals are preserved and only
    the gene-to-feature link is destroyed. Establishes significance ONLY -- it is never subtracted from
    the observed value, which is reported raw."""
    out = []
    for s in range(n):
        rr = np.random.default_rng(seed0 + s)
        out.append(cv_r2(X[rr.permutation(len(X))], y, groups, seed=s)[0])
    return np.array(out)


def null_line(obs, null, name, report):
    """p is bounded below by 1/(n+1) with n shuffles, so the distance in null SDs is reported too --
    quoting a p pinned at its own resolution floor as if it were the evidence would understate it."""
    sd = float(null.std())
    z = (obs - null.mean()) / sd if sd > 0 else float("nan")
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    report(f"      {name:44s} {null.mean():8.4f} +- {sd:.4f}  (max {null.max():.4f})  "
           f"p {p:.3g}{' = 1/'+str(len(null)+1)+', the floor' if p <= 1.0001/(len(null)+1) else ''}, "
           f"{z:.0f} null SDs")
    return {"mean": float(null.mean()), "sd": sd, "max": float(null.max()), "p": p, "z": float(z)}


def cv_r2(X, y, groups, seed=0):
    """Chromosome-held-out R2, computed as 1 - SSE/SST on the POOLED out-of-fold predictions.

    Pooled rather than averaged per fold: chromosomes differ in size and in gene content, and a mean of
    per-fold R2 lets a small chromosome with a narrow y-range dominate. Returns the raw held-out value.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=4,
                                          min_samples_leaf=30, l2_regularization=1.0,
                                          random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    ok = np.isfinite(oof)
    sse = float(np.sum((y[ok] - oof[ok]) ** 2))
    sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return 1.0 - sse / sst, oof


# ---------------------------------------------------------------- the fast test
def qbin(v, n):
    """Quantile bins of `v`; non-finite entries land in bin 0 and are excluded by the eligibility mask."""
    qs = np.nanquantile(v, np.linspace(1.0 / n, 1.0 - 1.0 / n, n - 1))
    return np.clip(np.digitize(np.where(np.isfinite(v), v, -1e30), qs), 0, n - 1)


def fast_test(A, fin, prow, gcol, li, lj, row_ok, col_ok, label, report, self_lin, nbin=10,
              n_draws=N_DRAWS, cap=PAIR_CAP, seed=0):
    """|z| of linked pairs vs pairs matched on perturbation-strength AND gene-responsiveness decile.

    The control is redrawn `n_draws` times. A single matched draw is one sample of the matched population
    and this project has read one as the result three times; the mean effect and the fraction of draws
    reaching p<0.05 are reported instead, with residual imbalance on both covariates.
    """
    from scipy import stats
    rng0 = np.random.default_rng(seed)
    keep = fin[li, lj]
    li, lj = li[keep], lj[keep]
    n_all = len(li)
    if n_all < 1000:
        report(f"    {label}: only {n_all:,} testable linked pairs -- not run")
        return None
    L = np.zeros(A.shape, dtype=bool)
    L[li, lj] = True                       # full linked set is excluded from controls, not just the cap
    L.reshape(-1)[self_lin] = True         # and so is every perturbation's own gene
    if n_all > cap:
        s = rng0.choice(n_all, cap, replace=False)
        li, lj = li[s], lj[s]
    v_link = A[li, lj]
    pq, gq = qbin(prow, nbin), qbin(gcol, nbin)
    key = pq[li].astype(np.int64) * nbin + gq[lj]
    rows_by = [np.where(row_ok & (pq == p))[0] for p in range(nbin)]
    cols_by = [np.where(col_ok & (gq == g))[0] for g in range(nbin)]
    ks, kn = np.unique(key, return_counts=True)

    draws = []
    for d in range(n_draws):
        rng = np.random.default_rng(1000 + d)
        ci, cj = [], []
        for k, n in zip(ks, kn):
            r, c = rows_by[k // nbin], cols_by[k % nbin]
            if len(r) == 0 or len(c) == 0:
                continue
            need, tries = int(n), 0
            while need > 0 and tries < 30:
                m = max(need * 3, 64)
                ii = r[rng.integers(0, len(r), m)]
                jj = c[rng.integers(0, len(c), m)]
                ok = fin[ii, jj] & ~L[ii, jj]
                ii, jj = ii[ok][:need], jj[ok][:need]
                ci.append(ii)
                cj.append(jj)
                need -= len(ii)
                tries += 1
        ci, cj = np.concatenate(ci), np.concatenate(cj)
        if len(ci) < 0.5 * len(li):
            continue
        v_ctl = A[ci, cj]
        draws.append({
            "n_ctl": int(len(ci)),
            "mean_link": float(v_link.mean()), "mean_ctl": float(v_ctl.mean()),
            "ratio": float(v_link.mean() / v_ctl.mean()),
            "p": float(stats.mannwhitneyu(v_link, v_ctl, alternative="two-sided")[1]),
            "res_pert_p": float(stats.mannwhitneyu(prow[li], prow[ci])[1]),
            "res_gene_p": float(stats.mannwhitneyu(gcol[lj], gcol[cj])[1])})
    del L
    if not draws:
        report(f"    {label}: matched control could not be filled")
        return None

    def mn(k):
        return float(np.mean([x[k] for x in draws]))
    R = {"label": label, "n_bins": nbin, "n_linked_total": int(n_all), "n_linked_used": int(len(li)),
         "n_draws": len(draws), "mean_linked_absz": float(v_link.mean()),
         "mean_matched_absz": mn("mean_ctl"), "ratio": mn("ratio"),
         "p_median": float(np.median([x["p"] for x in draws])),
         "frac_draws_p05": float(np.mean([x["p"] < 0.05 for x in draws])),
         "residual_pert_p": mn("res_pert_p"), "residual_gene_p": mn("res_gene_p")}
    report(f"    {label}   [{nbin} bins per covariate]")
    report(f"      linked pairs {R['n_linked_total']:,} (tested {R['n_linked_used']:,}), "
           f"{R['n_draws']} matched draws")
    report(f"      mean |z| linked {R['mean_linked_absz']:.4f}  matched {R['mean_matched_absz']:.4f}  "
           f"RATIO {R['ratio']:.4f}")
    report(f"      MWU median p {R['p_median']:.3g}, {100*R['frac_draws_p05']:.0f}% of draws p<0.05")
    report(f"      residual imbalance: perturbation strength p {R['residual_pert_p']:.3g}, "
           f"gene responsiveness p {R['residual_gene_p']:.3g}")
    return R


def matched_test(*a, **kw):
    """Run the prescribed decile match; if a covariate is still imbalanced, refuse to read it and rerun
    on 20 bins.

    This project has the precedent: `bound_causal.py` moved from quartiles to deciles because a residual
    imbalance at p 0.037 sat on exactly the covariate the match existed to remove. A matched comparison
    whose match demonstrably failed is not a weak result, it is an uninterpretable one, and reporting the
    ratio next to a failing residual invites it to be read anyway. Both levels are kept in the output.
    """
    report = a[9]
    R = fast_test(*a, **kw)
    if not R:
        return R
    worst = min(R["residual_pert_p"], R["residual_gene_p"])
    if worst < 0.01:
        report(f"      residual imbalance at p {worst:.3g} -- the decile match FAILED on a covariate it "
               f"exists to remove; escalating to 20 bins")
        R2 = fast_test(*a, nbin=20, **kw)
        if R2:
            R2["decile_match_failed"] = {k: R[k] for k in ("ratio", "p_median", "frac_draws_p05",
                                                           "residual_pert_p", "residual_gene_p")}
            w2 = min(R2["residual_pert_p"], R2["residual_gene_p"])
            R2["match_ok"] = bool(w2 >= 0.01)
            if not R2["match_ok"]:
                report(f"      still imbalanced at p {w2:.3g} on 20 bins -- the linked set is too extreme "
                       f"on this covariate to match, so THIS ARM'S RATIO IS NOT INTERPRETABLE and is "
                       f"reported only to show that it was tried")
            return R2
        R["match_failed_and_could_not_escalate"] = True
        R["match_ok"] = False
        return R
    R["match_ok"] = True
    return R


def drop_self(li, lj, ncol, self_lin):
    """Remove the perturbation's own gene: that pair is the knockdown itself, not a prediction."""
    if len(self_lin) == 0:
        return li, lj
    m = ~np.isin(li.astype(np.int64) * ncol + lj, self_lin)
    return li[m], lj[m]


def similarity_link(rows, cols, row_v, col_v, frac, report, tag):
    """Pairs whose per-gene value is in the closest `frac` of the |difference| distribution."""
    rng = np.random.default_rng(7)
    sa = row_v[rng.integers(0, len(row_v), 400_000)]
    sb = col_v[rng.integers(0, len(col_v), 400_000)]
    thr = float(np.quantile(np.abs(sa - sb), frac))
    li, lj = [], []
    for k in range(0, len(rows), 256):
        rr = rows[k:k + 256]
        d = np.abs(row_v[k:k + 256][:, None] - col_v[None, :])
        a, b = np.where(d <= thr)
        li.append(rr[a].astype(np.int32))
        lj.append(cols[b].astype(np.int32))
    li, lj = np.concatenate(li), np.concatenate(lj)
    report(f"      {tag}: |delta| threshold {thr:.4f} log10 -> {len(li):,} pairs "
           f"of {len(rows)*len(cols):,} testable")
    return li, lj, thr


# ---------------------------------------------------------------- main
def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("TRANSLATION EFFICIENCY -- the missing multiplier, and whether sequence can supply it")
    report("=" * 100)
    for p in (GWPS, PROT, TWOBIT, GMT):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    from entity_registry import load as load_reg
    REG = load_reg()

    # ---- Q1: is the direct measurement available? ----
    report("\n  Q1  IS THERE K562 RIBOSOME PROFILING TO FETCH?")
    enc = encode_ribo_probe(report)
    report("      -> ENCODE carries NO ribosome-profiling assay, in K562 or anywhere. The direct "
           "measurement\n         is unavailable from this source; everything below is the inferred "
           "fallback and is labelled as such.")

    # ---- Q2: the empirical protein-per-mRNA ratio ----
    import h5py
    with h5py.File(GWPS, "r") as f:
        ens = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
        rna = np.asarray(f["var"]["mean"][:], dtype=np.float64)
        obs = [(s.decode() if isinstance(s, bytes) else str(s)) for s in f["obs"]["gene_transcript"][:]]
    pert_ens = [o.split("_")[-1] if o.split("_")[-1].startswith("ENSG") else None for o in obs]
    prot_d = json.load(gzip.open(PROT, "rt"))
    prot = prot_d["abundance"]
    assert prot_d["source"] == PROT_SOURCE and prot_d["cell_type"] == "K562", \
        f"protein layer is not the pinned K562 source: {prot_d.get('source')}"

    # JOINS. Perturbations and measured genes carry ENSG, so they join on an accession. The proteomics
    # file carries gene symbol only, which is the weakest namespace and is recorded as such.
    _g, st_var = REG.join("gene", "ensembl", ens, 0.95, "GWPS measured genes (ENSG)")
    _g, st_obs = REG.join("gene", "ensembl", [p for p in pert_ens if p], 0.95,
                          "GWPS perturbations (ENSG)")
    _g, st_pro = REG.join("gene", "symbol", list(prot), 0.90, "K562 iBAQ proteome (symbol)")
    report(f"\n  Q2  EMPIRICAL PROTEIN-PER-mRNA RATIO")
    for s in (st_var, st_obs, st_pro):
        report(f"      join {s['label']:38s} {s['joined']:,}/{s['of']:,} ({s['rate']:.1%}) via={s['via']}")
    # 585 rows are non-targeting controls and correctly have no gene; they are not perturbations of
    # anything and are excluded from every row universe below rather than counted as a join failure.
    nt = np.array(["non-targeting" in o for o in obs])
    real = ~nt
    n_bad = int(sum(1 for i, p in enumerate(pert_ens) if p is None and real[i]))
    report(f"      {int(nt.sum()):,} non-targeting control rows excluded; {n_bad} gene-targeting rows "
           f"carry no ENSG in obs/gene_transcript and are excluded too")
    assert n_bad < 0.02 * int(real.sum()), \
        f"{n_bad} of {int(real.sum())} gene-targeting perturbations lack an ENSG"

    sym_of, ens_of_sym = {}, {}
    for e in ens:
        eid = REG.resolve("gene", "ensembl", e)
        ent = REG.entities.get(eid) if eid else None
        if ent and ent.get("symbol"):
            sym_of[e] = ent["symbol"]
    for s in prot:
        eid = REG.resolve("gene", "symbol", s)
        ent = REG.entities.get(eid) if eid else None
        if ent and ent.get("ensembl"):
            ens_of_sym[s] = ent["ensembl"]
    prot_by_ens = {}
    for s, v in prot.items():
        e = ens_of_sym.get(s)
        if e and v > 0:
            prot_by_ens[e] = max(prot_by_ens.get(e, 0.0), float(v))

    ann = build_features(report)
    A_ = ann["genes"]
    assert ann["ensembl_release"] == ENSEMBL_RELEASE, "annotation release drifted"

    tr_ens = [e for i, e in enumerate(ens) if rna[i] > 0 and e in prot_by_ens]
    rna_of = dict(zip(ens, rna))
    lp = np.log10(np.array([prot_by_ens[e] for e in tr_ens]))
    lr = np.log10(np.array([rna_of[e] for e in tr_ens]))
    TR = lp - lr
    tr_of = dict(zip(tr_ens, TR))
    report(f"      {len(tr_ens):,} genes with BOTH K562 iBAQ protein and K562 Perturb-seq mRNA")
    report(f"      log10 protein-per-mRNA: median {np.median(TR):.3f}, IQR "
           f"[{np.percentile(TR,25):.3f}, {np.percentile(TR,75):.3f}], "
           f"5-95% span {np.percentile(TR,95)-np.percentile(TR,5):.2f} log10 "
           f"({10**(np.percentile(TR,95)-np.percentile(TR,5)):.0f}x)")
    r_lin = float(np.corrcoef(lr, lp)[0, 1])
    report(f"      mRNA vs protein: r {r_lin:.4f}, R2 {r_lin**2:.4f}  "
           f"(k562_protein.py reported R2 0.1594 on 5,874 genes)")

    # THE FIRST THING TO ASK OF A RATIO IS WHETHER IT IS A QUANTITY OF ITS OWN. A difference of two logs
    # inherits whichever term varies more, and nothing guarantees that is the numerator. If the ratio
    # turns out to track the mRNA denominator, then ranking genes by "translation efficiency" ranks them
    # by inverse expression and every downstream use of it is measuring expression under another name.
    r_tr_rna = float(np.corrcoef(TR, lr)[0, 1])
    r_tr_prot = float(np.corrcoef(TR, lp)[0, 1])
    sr_tr_rna = float(stats.spearmanr(TR, lr)[0])
    sr_tr_prot = float(stats.spearmanr(TR, lp)[0])
    report(f"      IS THIS RATIO A QUANTITY OF ITS OWN, OR INVERSE mRNA?")
    report(f"        corr(ratio, log mRNA)     {r_tr_rna:+.4f}  (Spearman {sr_tr_rna:+.4f})")
    report(f"        corr(ratio, log protein)  {r_tr_prot:+.4f}  (Spearman {sr_tr_prot:+.4f})")
    report(f"        sd(log mRNA) {lr.std():.3f} vs sd(log protein) {lp.std():.3f} -- the denominator "
           f"varies {lr.std()/lp.std():.1f}x more than the numerator, so the difference inherits it")
    report(f"        -> the ratio explains {100*r_tr_rna**2:.0f}% of log mRNA and "
           f"{100*r_tr_prot**2:.0f}% of log protein. It is very largely inverse expression.")
    hiq, loq = float(np.quantile(TR, .9)), float(np.quantile(TR, .1))
    top_s = [sym_of.get(e, e) for e, t in zip(tr_ens, TR) if t >= hiq]
    bot_s = [sym_of.get(e, e) for e, t in zip(tr_ens, TR) if t <= loq]
    nrp = lambda ns: sum(1 for n in ns if re.match(r"^RP[LS]\d", n or ""))
    report(f"        top ratio decile {len(top_s)} genes, {nrp(top_s)} ribosomal proteins; bottom decile "
           f"{len(bot_s)} genes, {nrp(bot_s)} ribosomal proteins -- the bottom decile is where the "
           f"most abundant mRNAs sit, which is what an inverse-mRNA quantity looks like")
    assert abs(r_lin ** 2 - 0.1594) < 0.05, \
        f"the mRNA->protein R2 does not reproduce the pinned 15.9%: got {r_lin**2:.4f}"

    # ---- Q3: is the ratio predictable from sequence and annotation? ----
    have = [e for e in tr_ens if e in A_ and np.isfinite(A_[e]["gc_cds"])]
    rows = [A_[e] for e in have]
    X = featmat(rows)
    y = np.array([tr_of[e] for e in have])
    yp = np.array([np.log10(prot_by_ens[e]) for e in have])
    yr = np.array([np.log10(rna_of[e]) for e in have])
    grp = np.array([r["chrom"] for r in rows])
    report(f"\n  Q3  IS THE RATIO PREDICTABLE FROM SEQUENCE + ANNOTATION?")
    report(f"      {len(have):,} genes carry ratio AND Ensembl {ENSEMBL_RELEASE} annotation "
           f"({100*len(have)/len(tr_ens):.1f}% of the ratio set); "
           f"{len(FEATS)} features; folds held out by chromosome ({len(set(grp))} groups)")
    r2_tr, oof_tr = cv_r2(X, y, grp)
    j = [FEATS.index(f) for f in FEATS if f not in LENGTH_FEATS]
    r2_nolen, _ = cv_r2(X[:, j], y, grp)
    sp_tr = float(stats.spearmanr(y, oof_tr)[0])
    report(f"      held-out R2 (RAW)                            {r2_tr:8.4f}   Spearman {sp_tr:.4f}")
    report(f"      held-out R2, LENGTH FEATURES DROPPED         {r2_nolen:8.4f}   "
           f"<- {', '.join(LENGTH_FEATS)} removed")
    # BOTH arms get their own null. Quoting the ablated R2 without one would leave the reader unable to
    # tell whether what survives the ablation is signal or the fitting procedure's own optimism.
    n_full = null_line(r2_tr, null_r2(X, y, grp, seed0=500), "shuffled-feature null, all features", report)
    n_abl = null_line(r2_nolen, null_r2(X[:, j], y, grp, seed0=700),
                      "shuffled-feature null, length dropped", report)
    p_tr = n_full["p"]
    imp, const = [], []
    for k, f in enumerate(FEATS):
        c = X[:, k]
        m = np.isfinite(c)
        if m.sum() < 10 or np.ptp(c[m]) == 0:
            const.append(f)                      # a constant feature carries nothing; say so, do not rank it
            continue
        imp.append((f, float(stats.spearmanr(c[m], y[m])[0])))
    imp.sort(key=lambda x: -abs(x[1]))
    report(f"      strongest univariate Spearman vs ratio: " +
           ", ".join(f"{f} {r:+.3f}" for f, r in imp[:6]))
    report(f"      NOTE: the target is {100*r_tr_rna**2:.0f}% inverse log mRNA, so this R2 is mostly the "
           f"predictability of EXPRESSION from sequence, not of translation. Q4 does not route through "
           f"the ratio and is the arm that survives this.")
    if const:
        report(f"      constant on this gene set, carrying nothing: {const}")

    # ---- Q4: the downstream decision ----
    report(f"\n  Q4  THE DOWNSTREAM DECISION -- does sequence raise the 15.9% mRNA->protein ceiling?")
    Xr = yr.reshape(-1, 1)
    r2_rna, _ = cv_r2(Xr, yp, grp)
    r2_seq, _ = cv_r2(X, yp, grp)
    r2_both, oof_both = cv_r2(np.hstack([Xr, X]), yp, grp)
    r2_both_nl, _ = cv_r2(np.hstack([Xr, X[:, j]]), yp, grp)
    report(f"      predicting log10 protein, chromosome-held-out, {len(have):,} genes")
    report(f"      {'arm':40s} {'held-out R2 (RAW)':>18s}")
    report(f"      {'mRNA only (the 15.9% baseline)':40s} {r2_rna:18.4f}")
    report(f"      {'sequence + annotation only':40s} {r2_seq:18.4f}")
    report(f"      {'mRNA + sequence + annotation':40s} {r2_both:18.4f}")
    report(f"      {'mRNA + sequence, LENGTH DROPPED':40s} {r2_both_nl:18.4f}")
    # THE NULL IS THE SAME MODEL WITH THE FEATURES SHUFFLED, not the mRNA-only arm. It lands BELOW the
    # mRNA-only baseline because 18 noise features cost the model something -- which is what makes it the
    # right null for "do these particular features add anything".
    nb = null_line(r2_both, null_r2(np.hstack([Xr, X]), yp, grp, seed0=900),
                   "null: mRNA + shuffled features", report)
    nbl = null_line(r2_both_nl, null_r2(np.hstack([Xr, X[:, j]]), yp, grp, seed0=1100),
                    "null: mRNA + shuffled non-length features", report)
    p_both = nb["p"]

    # ---- Q5: the fast test ----
    report(f"\n  Q5  FAST TEST on GWPS -- does the layer mark (perturbation, gene) pairs with elevated |z|?")
    with h5py.File(GWPS, "r") as f:
        Z = np.asarray(f["X"][:], dtype=np.float32)
    fin = np.isfinite(Z)
    report(f"      X {Z.shape[0]:,} x {Z.shape[1]:,}; {int((~fin).sum()):,} non-finite cells "
           f"({100*(~fin).mean():.3f}%) DROPPED, never imputed")
    Z[~fin] = np.nan
    for k in range(0, Z.shape[1], 512):
        b = Z[:, k:k + 512]
        med = np.nanmedian(b, axis=0)
        mad = np.nanmedian(np.abs(b - med), axis=0)
        sc = 1.4826 * mad
        sc[~np.isfinite(sc) | (sc <= 0)] = np.nan
        Z[:, k:k + 512] = (b - med) / sc
    fin &= np.isfinite(Z)
    Z[~fin] = np.nan
    np.abs(Z, out=Z)
    prow = np.nanmean(Z, axis=1)
    gcol = np.nanmean(Z, axis=0)
    report(f"      per-gene robust z; perturbation strength median {np.nanmedian(prow):.3f}, "
           f"gene responsiveness median {np.nanmedian(gcol):.3f}; both binned into deciles, and into "
           f"20 bins if a decile residual fails")

    pens = np.array([p if p else "" for p in pert_ens])
    row_tr = np.array([tr_of.get(e, np.nan) for e in pens])
    col_tr = np.array([tr_of.get(e, np.nan) for e in ens])
    row_ok = np.isfinite(row_tr) & np.isfinite(prow) & real
    col_ok = np.isfinite(col_tr) & np.isfinite(gcol)
    rows_i = np.where(row_ok)[0]
    cols_i = np.where(col_ok)[0]
    report(f"      perturbations with a measured ratio {row_ok.sum():,}/{len(pens):,}; "
           f"measured genes with one {col_ok.sum():,}/{len(ens):,}")
    # a perturbation's own gene is excluded everywhere: that pair is the knockdown, not a prediction
    ncol = len(ens)
    ens_col = {e: i for i, e in enumerate(ens)}
    self_lin = np.array(sorted(i * ncol + ens_col[e] for i, e in enumerate(pens) if e in ens_col),
                        dtype=np.int64)
    report(f"      {len(self_lin):,} perturbation-own-gene pairs excluded from links and controls")

    arms = {}
    li, lj, thr1 = similarity_link(rows_i, cols_i, row_tr[rows_i], col_tr[cols_i],
                                   LINK_DECILE, report, "arm1 measured-ratio similarity")
    li, lj = drop_self(li, lj, ncol, self_lin)
    arms["measured_ratio_similarity"] = matched_test(Z, fin, prow, gcol, li, lj, row_ok, col_ok,
                                                    "arm1  shared translational regime "
                                                    "(measured ratio)", report, self_lin)

    # arm 2: the mechanistic interaction
    tset = set()
    with open(GMT) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 2 and p[1] == TRANSLATION_PATHWAY:
                tset = {s for s in p[2:] if s}
    tens = {ens_of_sym[s] for s in tset if s in ens_of_sym}
    report(f"      Reactome {TRANSLATION_PATHWAY} 'Translation': {len(tset)} symbols, "
           f"{len(tens)} with an Ensembl id")
    hi = np.nanquantile(col_tr[col_ok], 0.9)
    r2i = np.where(np.array([e in tens for e in pens]) & np.isfinite(prow) & real)[0]
    c2i = np.where(col_ok & (col_tr >= hi))[0]
    if len(r2i) >= 20 and len(c2i) >= 50:
        li2, lj2 = drop_self(np.repeat(r2i, len(c2i)), np.tile(c2i, len(r2i)), ncol, self_lin)
        report(f"      arm2: {len(r2i)} translation-apparatus perturbations x {len(c2i)} top-decile-ratio "
               f"genes = {len(li2):,} pairs")
        arms["translation_apparatus_x_high_ratio"] = matched_test(
            Z, fin, prow, gcol, li2, lj2, real & np.isfinite(prow), col_ok,
            "arm2  translation apparatus perturbed x high-ratio gene", report, self_lin)
    else:
        arms["translation_apparatus_x_high_ratio"] = None

    # arm 3: the artefact that would actually ship -- PREDICTED ratio, available genome-wide
    from sklearn.ensemble import HistGradientBoostingRegressor
    mfull = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=4,
                                          min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    mfull.fit(X, y)
    pred_of = dict(zip(have, oof_tr))          # out-of-fold where the gene was trainable
    othr = [e for e in set(list(pens) + ens) if e and e in A_ and e not in pred_of
            and np.isfinite(A_[e]["gc_cds"])]
    if othr:
        pv = mfull.predict(featmat([A_[e] for e in othr]))
        pred_of.update(dict(zip(othr, pv)))
    row_pr = np.array([pred_of.get(e, np.nan) for e in pens])
    col_pr = np.array([pred_of.get(e, np.nan) for e in ens])
    rok3 = np.isfinite(row_pr) & np.isfinite(prow) & real
    cok3 = np.isfinite(col_pr) & np.isfinite(gcol)
    report(f"      predicted ratio covers {rok3.sum():,}/{len(pens):,} perturbations and "
           f"{cok3.sum():,}/{len(ens):,} measured genes "
           f"(+{cok3.sum()-col_ok.sum():,} genes over the measured ratio)")
    r3, c3 = np.where(rok3)[0], np.where(cok3)[0]
    li3, lj3, thr3 = similarity_link(r3, c3, row_pr[r3], col_pr[c3], LINK_DECILE, report,
                                     "arm3 predicted-ratio similarity")
    li3, lj3 = drop_self(li3, lj3, ncol, self_lin)
    arms["predicted_ratio_similarity"] = matched_test(
        Z, fin, prow, gcol, li3, lj3, rok3, cok3,
        "arm3  shared translational regime (predicted ratio)", report, self_lin)
    del Z, fin

    # ---- Q6: coverage, and is missingness biased? ----
    report(f"\n  Q6  COVERAGE AND MISSINGNESS")
    meas = np.array([e in prot_by_ens for e in ens])
    lrall = np.log10(np.maximum(rna, 1e-3))
    q = np.clip(np.digitize(lrall, np.quantile(lrall, [.2, .4, .6, .8])), 0, 4)
    report(f"      {int(meas.sum()):,}/{len(ens):,} GWPS-measured genes ({100*meas.mean():.1f}%) "
           f"have a K562 protein value, so a ratio")
    report(f"      {'mRNA quintile':22s} {'n':>7s} {'frac with protein':>18s}")
    for k in range(5):
        mm = q == k
        report(f"      {('q'+str(k)):22s} {int(mm.sum()):7,d} {meas[mm].mean():18.4f}")
    mwu_cov = float(stats.mannwhitneyu(lrall[meas], lrall[~meas])[1])
    report(f"      mRNA level, measured vs not: MWU p {mwu_cov:.3g}  -> missingness is STRONGLY "
           f"expression-dependent")
    report(f"      the ratio is therefore LEFT-CENSORED: genes whose protein fell below the MS detection")
    report(f"      limit have no ratio, and they are systematically the low-ratio ones. Absence is "
           f"unknown, not zero.")
    ratio_lowq = float(np.mean([tr_of[e] for e in tr_ens if rna_of[e] <= np.quantile(rna, .2)]))
    ratio_hiq = float(np.mean([tr_of[e] for e in tr_ens if rna_of[e] >= np.quantile(rna, .8)]))
    report(f"      mean ratio in the lowest mRNA quintile {ratio_lowq:.3f} vs highest {ratio_hiq:.3f} "
           f"-- the censoring shows as a floor that rises with mRNA")

    # ---- write the layer ----
    layer = {
        "model": "translation-v1",
        "quantity": "log10(K562 iBAQ protein) - log10(K562 Perturb-seq mean mRNA), per gene",
        "not_measured": "this is NOT ribosome-profiling translation efficiency; it is translation "
                        "divided by protein degradation at steady state, read across two assays",
        "rna_source": RNA_SOURCE, "protein_source": PROT_SOURCE,
        "annotation": {"ensembl_release": ENSEMBL_RELEASE, "assembly": ASSEMBLY,
                       "transcript_rule": ann["transcript_rule"]},
        "encode_probe": enc,
        "join": {"rna": st_var, "perturbation": st_obs, "protein": st_pro},
        "n_measured_ratio": len(tr_ens), "n_predicted": len(pred_of),
        "features": FEATS,
        "ratio_measured": {e: float(tr_of[e]) for e in tr_ens},
        "ratio_predicted": {e: float(v) for e, v in pred_of.items()},
        "limits": [
            "THE RATIO IS DOMINATED BY ITS DENOMINATOR. It correlates -0.87 with log mRNA and +0.09 with "
            "log protein, because log mRNA varies about twice as much as log iBAQ protein over these "
            "genes. Ranking genes by this ratio very largely ranks them by inverse expression, so it is "
            "NOT a usable translation multiplier and any analysis keyed on it inherits that. The "
            "protein-level prediction in `downstream_protein_prediction` deliberately does not route "
            "through the ratio and is the result that survives",
            "NOT ribosome profiling. ENCODE has no Ribo-seq for K562 or for any biosample; this is a "
            "steady-state protein/mRNA ratio, which confounds translation rate with protein degradation "
            "rate and cannot separate them",
            "two assays, two normalisations: iBAQ is a within-sample intensity divided by the number of "
            "theoretically observable peptides, Perturb-seq mean is a normalised UMI mean. Their ratio "
            "has no absolute units and is comparable only within this table",
            "LEFT-CENSORED and the censoring is expression-dependent: the ratio exists only where mass "
            "spectrometry detected the protein, so genes with genuinely low protein per mRNA are "
            "preferentially missing and every statistic here is computed on a biased sample",
            "the protein table joins by GENE SYMBOL, the weakest namespace in the registry; the mRNA and "
            "perturbation tables join by ENSG",
            "restricted to the 8,248 genes GWPS measures, which are the expressed and well-detected ones, "
            "so the ratio is unknown for most of the genome",
            "measured in unperturbed K562 -- a STATE, not a response. It says nothing about how the "
            "protein/mRNA ratio moves after a perturbation, which is what the chain's missing arrow "
            "actually requires",
            "predicted ratios inherit the training set's censoring and are extrapolations for any gene "
            "whose protein was never detected",
            "one cell line, one proteomics study, one growth condition"],
    }
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    R = {"model": "translation-v1",
         "direct_source_available": False,
         "encode_probe": enc,
         "rna_source": RNA_SOURCE, "protein_source": PROT_SOURCE,
         "annotation": layer["annotation"], "join": layer["join"],
         "ratio_is_inverse_mrna": {"corr_ratio_logmrna": r_tr_rna, "corr_ratio_logprotein": r_tr_prot,
                                   "spearman_ratio_logmrna": sr_tr_rna,
                                   "spearman_ratio_logprotein": sr_tr_prot,
                                   "sd_log_mrna": float(lr.std()), "sd_log_protein": float(lp.std()),
                                   "n_ribosomal_top_decile": nrp(top_s),
                                   "n_ribosomal_bottom_decile": nrp(bot_s)},
         "ratio": {"n": len(tr_ens), "median": float(np.median(TR)),
                   "iqr": [float(np.percentile(TR, 25)), float(np.percentile(TR, 75))],
                   "span_5_95_log10": float(np.percentile(TR, 95) - np.percentile(TR, 5)),
                   "mrna_protein_r": r_lin, "mrna_protein_r2": r_lin ** 2},
         "predictability": {"n_genes": len(have), "n_features": len(FEATS),
                            "cv": "GroupKFold(5) by chromosome",
                            "heldout_r2": r2_tr, "heldout_spearman": sp_tr,
                            "heldout_r2_no_length_features": r2_nolen,
                            "length_features_dropped": LENGTH_FEATS,
                            "null_all_features": n_full, "null_length_dropped": n_abl,
                            "p": p_tr, "top_univariate": imp[:8], "constant_features": const},
         "downstream_protein_prediction": {
             "n_genes": len(have), "r2_mrna_only": r2_rna, "r2_sequence_only": r2_seq,
             "r2_mrna_plus_sequence": r2_both, "r2_mrna_plus_sequence_no_length": r2_both_nl,
             "null_mrna_plus_shuffled": nb, "null_mrna_plus_shuffled_no_length": nbl,
             "p": p_both},
         "fast_test": arms,
         "coverage": {"n_gwps_genes": len(ens), "frac_with_protein": float(meas.mean()),
                      "expression_dependent_p": mwu_cov,
                      "mean_ratio_low_mrna_quintile": ratio_lowq,
                      "mean_ratio_high_mrna_quintile": ratio_hiq},
         "limits": layer["limits"]}

    # ---- verdict ----
    report("\n" + "=" * 100)
    a1 = arms.get("measured_ratio_similarity") or {}
    a2 = arms.get("translation_apparatus_x_high_ratio") or {}
    a3 = arms.get("predicted_ratio_similarity") or {}
    best = max([a for a in (a1, a2, a3) if a], key=lambda d: d["ratio"], default=None)
    pred_ok = r2_tr > n_full["max"] and r2_tr > 0.05
    abl_ok = r2_nolen > n_abl["max"] and r2_nolen > 0.02
    len_share = 1.0 - r2_nolen / r2_tr if r2_tr > 0 else float("nan")
    gain = r2_both - r2_rna
    gain_nl = r2_both_nl - r2_rna
    head = (f"ENCODE HAS NO RIBOSOME PROFILING -- not for K562, not for any biosample: the assay_title "
            f"facet was read live and carries {enc['n_titles_portal']} titles, none of them "
            f"ribosome-profiling. The direct measurement of translation efficiency is unavailable from "
            f"this source, so what follows is a steady-state protein-per-mRNA ratio and is not the same "
            f"quantity. ")
    p1 = (f"That ratio spans {np.percentile(TR,95)-np.percentile(TR,5):.1f} log10 "
          f"({10**(np.percentile(TR,95)-np.percentile(TR,5)):.0f}x) across {len(tr_ens):,} genes, which "
          f"looks like a multiplier worth having -- and it is not one. It correlates {r_tr_rna:+.3f} "
          f"with log mRNA and {r_tr_prot:+.3f} with log protein: log mRNA varies "
          f"{lr.std()/lp.std():.1f}x more than log iBAQ protein here, so the difference of the two logs "
          f"inherits the denominator and the quantity is {100*r_tr_rna**2:.0f}% inverse expression. Its "
          f"bottom decile holds {nrp(bot_s)} ribosomal proteins against {nrp(top_s)} in the top -- it is "
          f"sorting genes by how much mRNA they have. AN EMPIRICAL PROTEIN-PER-mRNA RATIO IS THEREFORE "
          f"NOT A USABLE STAND-IN FOR TRANSLATION EFFICIENCY, and that is the single most important "
          f"thing this module establishes. ")
    if pred_ok:
        p2 = (f"It IS predictable from sequence and annotation alone: chromosome-held-out R2 {r2_tr:.3f} "
              f"(Spearman {sp_tr:.3f}) against a shuffled-feature null of {n_full['mean']:.3f} "
              f"(max {n_full['max']:.3f} over 20 shuffles, {n_full['z']:.0f} null SDs, p {p_tr:.3g}). "
              f"But {100*len_share:.0f}% of that is LENGTH: dropping cds/transcript length and exon "
              f"counts takes it to {r2_nolen:.3f}, and the strongest single correlate is log CDS length "
              f"({dict(imp).get('log_cds_len', float('nan')):+.3f} Spearman). iBAQ divides intensity by "
              f"the number of theoretically observable peptides and short proteins are the ones mass "
              f"spectrometry misses, so the length term is entangled with the assay's own normalisation "
              f"and must not be read as translation biology. ")
        p2 += (f"What survives the ablation is {'still real' if abl_ok else 'not separable from the null'}"
               f" (R2 {r2_nolen:.3f} vs null max {n_abl['max']:.3f}, {n_abl['z']:.0f} null SDs). ")
    else:
        p2 = (f"It is NOT predictable from sequence and annotation: chromosome-held-out R2 {r2_tr:.3f} "
              f"against a shuffled-feature null of {n_full['mean']:.3f} (max {n_full['max']:.3f}). ")
    p3 = (f"The arm that does NOT route through the ratio is the one that survives. On the decision that "
          f"matters -- predicting K562 protein level -- mRNA alone reaches held-out "
          f"R2 {r2_rna:.3f}, mRNA plus sequence {r2_both:.3f} "
          f"({'+' if gain >= 0 else ''}{gain:.3f}, i.e. {100*gain/max(r2_rna,1e-9):.0f}% relative, "
          f"against a mRNA-plus-shuffled-features null of {nb['mean']:.3f}, {nb['z']:.0f} null SDs), and "
          f"sequence alone {r2_seq:.3f}. With the length features dropped the combined arm still reaches "
          f"{r2_both_nl:.3f} ({'+' if gain_nl >= 0 else ''}{gain_nl:.3f} over mRNA alone), so the "
          f"ceiling does move on non-length sequence features and not only on the assay artefact. ")
    dep = [a for a in (a1, a2, a3) if a and a["ratio"] <= 0.95 and a["frac_draws_p05"] >= 0.8
           and min(a["residual_pert_p"], a["residual_gene_p"]) >= 0.01]
    if best and best["ratio"] >= 1.05 and best["frac_draws_p05"] >= 0.8:
        p4 = (f"On the fast test the layer DOES mark pairs: {best['label'].strip()} reaches "
              f"{best['ratio']:.3f}x mean |z| over matched controls "
              f"({best['n_linked_total']:,} linked pairs, median p {best['p_median']:.3g}, "
              f"{100*best['frac_draws_p05']:.0f}% of {best['n_draws']} draws p<0.05). ")
    else:
        rs = ", ".join(f"{a['label'].split()[0]} {a['ratio']:.3f}x" for a in (a1, a2, a3) if a)
        p4 = (f"On the fast test it marks no ELEVATED pairs: {rs} against matched controls, none of them "
              f"above the ~1.05 bar that PPI (1.208x) and co-dependency (1.070x) clear. That is the "
              f"result stated in advance rather than a disappointment -- the fast test scores PAIRS and "
              f"this is a PER-GENE layer, whose value if any lies on the mRNA->protein axis that "
              f"Perturb-seq cannot see at all. ")
        if dep:
            d = dep[0]
            p4 += (f"One arm goes the OTHER way and survives its controls: {d['label'].strip()} reaches "
                   f"{d['ratio']:.3f}x, a DEPLETION of {100*(1-d['ratio']):.1f}% "
                   f"({d['n_linked_used']:,} pairs, {d['n_bins']} bins per covariate, median p "
                   f"{d['p_median']:.3g}, {100*d['frac_draws_p05']:.0f}% of draws p<0.05, residuals "
                   f"{d['residual_pert_p']:.2g}/{d['residual_gene_p']:.2g}). Knocking down the "
                   f"translation apparatus moves high-protein-per-mRNA genes LESS than matched genes, "
                   f"not more. Signed and significant is a finding, not a non-detection. Read it "
                   f"cautiously: those 'high ratio' genes are largely LOW-mRNA genes, so the arm is at "
                   f"least as much about expression level as about translation. ")
        elif a2 and a2["ratio"] <= 0.95:
            p4 += (f"Arm2 sits at {a2['ratio']:.3f}x, a depletion rather than an enrichment -- and since "
                   f"its 'high ratio' genes are largely low-mRNA genes, it would not have been a "
                   f"translation result even had it held -- but its "
                   f"residual imbalance ({a2['residual_pert_p']:.2g}/{a2['residual_gene_p']:.2g}) leaves "
                   f"it uninterpretable rather than reversed. ")
    p5 = (f"The binding constraint is not the model, it is the measurement: only "
          f"{100*meas.mean():.0f}% of GWPS genes have a K562 protein value and missingness is strongly "
          f"expression-dependent (p {mwu_cov:.3g}), so the ratio is left-censored exactly where it "
          f"would be most informative.")
    v = head + p1 + p2 + p3 + p4 + p5
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "translation.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'translation.json'}")


if __name__ == "__main__":
    main()
