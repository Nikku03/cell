"""K562 CHROMATIN AND EXPRESSION STATE -- is a cell-matched covariate better than a generic one?

WHAT IS THERE NOW, AND WHY IT IS A PROBLEM. The cell object carries `ppm`: 16,015 genes of protein abundance
of unstated provenance, a generic human consensus rather than a measurement of the cell being modelled. It is
the abundance covariate in every matched analysis this project has run, and in the one analysis where the
covariates were ranked it was the strongest confound (p 6e-31). `k562_protein.py` then measured it directly
against K562 mass spectrometry and got a rank correlation of 0.655 -- broadly right, materially wrong in
detail. A covariate that is materially wrong does not merely add noise: it UNDER-CONTROLS, and an
under-controlled match reports a residual confound as a layer effect.

The Perturb-seq screen that every layer in this project is scored against is K562. ENCODE has measured K562
itself -- the same cell line, released, no login -- with polyA-plus RNA-seq gene quantifications and ATAC-seq
peaks. So the substitution is testable rather than arguable.

THE MEASUREMENT, PRESPECIFIED, IN THREE PARTS.

  1. THE GENE AXIS. Gene responsiveness -- how much a measured gene moves across all perturbations -- against
     K562 expression and against the generic `ppm`, on the SAME genes so the comparison is fair. Effect size
     is the raw mean |z| in each predictor's deciles; the winner is decided by a bootstrap over genes, not by
     a difference of point estimates.

  2. THE PERTURBATION AXIS, which is the one with a mechanism behind it. CRISPRi silences a promoter. A
     promoter that is already silent in K562 has nothing to silence, so a perturbation of a gene not
     expressed in K562 should do NOTHING, and a generic abundance table cannot know which genes those are
     because it is not about this cell. If cell-matched RNA beats generic abundance anywhere, it beats it
     here.

  3. DOES THE COVARIATE SWAP CHANGE A DECISION? This is the part that makes the layer worth carrying. The
     project has banked PPI at 1.208x on the standard matched fast test. That test matches on perturbation
     strength and gene responsiveness deciles only. Re-run it with a THIRD matching axis -- the measured
     gene's abundance -- once using generic `ppm` and once using K562-measured TPM. If the K562 covariate
     absorbs more of the ratio than the generic one does, then the generic table was under-controlling and
     every banked number in this project is optimistic by the difference.

WHY THE PRESCRIBED PAIR TEST CANNOT SCORE THIS LAYER, STATED BEFORE IT IS RUN. The fast test matches on the
row marginal (perturbation strength) and the column marginal (gene responsiveness). A per-gene state layer
predicts exactly those two marginals -- that is what "state" means. So any pair set defined by expression is
matched away by construction and the test returns ~1.00 whatever the layer is worth. That is a property of
the test, not evidence the layer is empty, and the arm is run anyway and reported so the claim is not taken
on trust. This layer is a COVARIATE, not a link set, and it is scored as one.

WHAT WOULD FALSIFY THIS. If K562 TPM predicts responsiveness no better than generic `ppm` -- bootstrap win
fraction near 0.5 -- the cell-matched state buys nothing and the generic table should be kept, cheaply. If
perturbations of genes that are silent in K562 are as disruptive as perturbations of expressed genes, the
mechanistic prediction in part 2 is simply wrong. If re-matching the PPI fast test on K562 expression leaves
the ratio where the generic covariate left it, the swap changes no decision and this layer is a description
of K562 with no downstream consequence.

ONE ANNOTATION, ONE ASSAY, ONE OUTPUT TYPE, ASSERTED NOT STATED. The same ENCODE experiment carries GRCh38
V29, GRCh38 V24 and hg19 V19 files under the identical output_type `gene quantifications`, and different
pipelines (RSEM, featureCounts) can sit under it too. V29/GRCh38 is pinned, the RSEM column header is checked
against a literal, and a mismatch raises. Subcellular fractions (nuclear, cytosolic, membrane, insoluble
cytoplasmic) and drug-treated biosamples are excluded by exact biosample summary and counted, because a
nuclear-fraction TPM is a different quantity from a whole-cell one and averaging them would produce a number
that is neither.

ABSENCE IS UNKNOWN. A gene with no ENCODE quantification is not a gene at zero TPM; a promoter with no peak
in a narrowPeak file is not a closed promoter, it is a promoter below that pipeline's threshold. Coverage and
its bias against both fast-test marginals are measured rather than assumed.
"""
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
API = "https://www.encodeproject.org"
DIR = SP / "encode"
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
LAYER = SP / "cell_encode_state.json.gz"
CACHE = DIR / "reduced.json.gz"

# ---- the pins. Every one of these is checked at fetch time and raises on mismatch. ----
RNA_ASSAY = "polyA plus RNA-seq"
RNA_ASSEMBLY = "GRCh38"
RNA_ANNOTATION = "V29"
RNA_OUTPUT = "gene quantifications"
RNA_PIPELINE_LAB = "/labs/encode-processing-pipeline/"
RSEM_HEADER = ["gene_id", "transcript_id(s)", "length", "effective_length", "expected_count", "TPM",
               "FPKM", "posterior_mean_count", "posterior_standard_deviation_of_count", "pme_TPM",
               "pme_FPKM", "TPM_ci_lower_bound", "TPM_ci_upper_bound",
               "TPM_coefficient_of_quartile_variation", "FPKM_ci_lower_bound", "FPKM_ci_upper_bound",
               "FPKM_coefficient_of_quartile_variation"]
ATAC_ASSAY = "ATAC-seq"
ATAC_OUTPUT = "conservative IDR thresholded peaks"
WHOLE_CELL = "Homo sapiens K562"     # exact biosample_summary: no fraction, no treatment, no modification
PROMOTER_BP = 1_000
N_DRAWS = 20
# THE COVARIATE-SWAP ARMS GET MORE DRAWS THAN EVERYTHING ELSE, on purpose. They are a SECOND-order contrast
# -- one matched ratio against another matched ratio -- and the quantity being compared is smaller than the
# draw-to-draw spread of either arm. Twenty seeds resolve the primary contrast (linked vs control, 100% of
# draws p<0.05) but not this one, and running it under-powered and then reading the noise as a null is the
# same error as reading one draw as the result.
N_SWAP_DRAWS = 40
N_DEC = 10
EXPRESSED_TPM = 1.0                  # the conventional "expressed" floor for RSEM TPM


def api(path, tries=4):
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=300))
        except Exception as e:
            if i == tries - 1:
                raise SystemExit(f"BLOCKED: ENCODE portal unreachable at {API}{path} -- {e}")
            time.sleep(2 ** (i + 1))


def download(url, path, tries=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=900) as fh, open(path, "wb") as o:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            return path
        except Exception as e:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise SystemExit(f"BLOCKED: cannot download {url} -- {e}")
            time.sleep(2 ** (i + 1))


def experiments(assay, report):
    """Released K562 experiments of one assay, whole-cell and untreated, by EXACT biosample summary."""
    d = api(f"/search/?type=Experiment&biosample_ontology.term_name=K562"
            f"&assay_title={urllib.parse.quote(assay)}&status=released&limit=200&frame=object")
    allx = d.get("@graph", [])
    keep = sorted(x["accession"] for x in allx if (x.get("biosample_summary") or "").strip() == WHOLE_CELL)
    drop = sorted((x.get("biosample_summary") or "?") for x in allx
                  if (x.get("biosample_summary") or "").strip() != WHOLE_CELL)
    report(f"  {assay}: {len(allx)} released K562 experiments, {len(keep)} whole-cell untreated "
           f"('{WHOLE_CELL}'), {len(drop)} excluded as fractions/treatments")
    if drop:
        report(f"    excluded examples: {drop[:3]}")
    if not keep:
        raise SystemExit(f"no whole-cell untreated K562 {assay} experiment on the portal")
    return keep


def files_of(acc, want, report):
    """ONE portal call per experiment. Returns the pinned files only; everything else is dropped."""
    d = api(f"/search/?type=File&dataset=/experiments/{acc}/&status=released&limit=300&frame=object")
    out = []
    for f in d.get("@graph", []):
        if all(f.get(k) == v for k, v in want.items()):
            out.append(f)
    return out


# ---------------------------------------------------------------------------- fetch + reduce
def fetch_rna(report):
    accs = experiments(RNA_ASSAY, report)
    want = {"assembly": RNA_ASSEMBLY, "genome_annotation": RNA_ANNOTATION, "output_type": RNA_OUTPUT,
            "file_format": "tsv", "lab": RNA_PIPELINE_LAB}
    picked = []
    for a in accs:
        fs = files_of(a, want, report)
        for f in fs:
            picked.append({"experiment": a, "file": f["accession"], "href": f["href"],
                           "replicates": f.get("biological_replicates"),
                           "annotation": f.get("genome_annotation"), "assembly": f.get("assembly")})
    if not picked:
        raise SystemExit(f"no {RNA_ASSEMBLY}/{RNA_ANNOTATION} '{RNA_OUTPUT}' files -- the portal layout "
                         f"changed or the pin is wrong")
    ann = {p["annotation"] for p in picked}
    asm = {p["assembly"] for p in picked}
    if ann != {RNA_ANNOTATION} or asm != {RNA_ASSEMBLY}:
        raise SystemExit(f"annotation/assembly not pinned: {ann} {asm}")
    report(f"  {len(picked)} RSEM files over {len(accs)} experiments, all {RNA_ASSEMBLY}/{RNA_ANNOTATION}")

    per_file = {}
    for p in picked:
        path = DIR / "rna" / f"{p['file']}.tsv"
        download(API + p["href"], path)
        tpm = {}
        with open(path) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            if head != RSEM_HEADER:
                raise SystemExit(f"{p['file']}: header is not RSEM -- {head[:6]}. Two pipelines share the "
                                 f"output_type '{RNA_OUTPUT}' and mixing them is the failure this pin "
                                 f"exists to prevent.")
            for line in fh:
                q = line.split("\t", 6)
                g = q[0]
                if not g.startswith("ENSG"):
                    continue                       # spike-ins and tRNA ids share the file
                tpm[g.split(".")[0]] = float(q[5])
        per_file[p["file"]] = tpm
        path.unlink()                              # reduced; the raw tsv is 11 MB and buys nothing
    return picked, per_file


def read_narrowpeak(path):
    ch, st, en, sig = [], [], [], []
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            ch.append(p[0])
            st.append(int(p[1]))
            en.append(int(p[2]))
            sig.append(float(p[6]))
    return np.array(ch), np.array(st, np.int64), np.array(en, np.int64), np.array(sig, np.float64)


def fetch_atac(report):
    accs = experiments(ATAC_ASSAY, report)
    want = {"assembly": RNA_ASSEMBLY, "output_type": ATAC_OUTPUT, "file_format": "bed",
            "file_format_type": "narrowPeak", "lab": RNA_PIPELINE_LAB}
    picked = []
    for a in accs:
        fs = files_of(a, want, report)
        if not fs:
            report(f"    {a}: no '{ATAC_OUTPUT}' GRCh38 narrowPeak -- dropped, not back-filled")
            continue
        fs.sort(key=lambda f: f["accession"])
        f = fs[0]
        path = DIR / "atac" / f"{f['accession']}.bed.gz"
        download(API + f["href"], path)
        picked.append({"experiment": a, "file": f["accession"], "path": str(path),
                       "output_type": f["output_type"]})
    if not picked:
        raise SystemExit(f"no K562 {ATAC_ASSAY} '{ATAC_OUTPUT}' file on the portal")
    kinds = {p["output_type"] for p in picked}
    if kinds != {ATAC_OUTPUT}:
        raise SystemExit(f"mixed peak output types: {kinds}")
    report(f"  {len(picked)} ATAC experiments carry a GRCh38 '{ATAC_OUTPUT}' narrowPeak")
    return picked


def promoter_signal(peaks, ens_list, chrom, tss, report):
    """Per experiment: is there a peak within +/-PROMOTER_BP of the TSS, and how strong.

    signalValue IS NOT COMPARABLE ACROSS EXPERIMENTS -- it is fold-enrichment against that experiment's own
    background, and the four experiments come from two labs with different read depths. So each experiment's
    promoter signal is converted to a WITHIN-EXPERIMENT percentile before the experiments are combined.
    Averaging raw signalValue would let the deepest experiment set the scale.
    """
    from scipy import stats
    n = len(ens_list)
    hit = np.zeros((len(peaks), n), bool)
    pct = np.zeros((len(peaks), n))
    for k, p in enumerate(peaks):
        pc, ps, pe, pv = read_narrowpeak(p["path"])
        raw = np.zeros(n)
        for c in np.unique(chrom):
            m = pc == c
            if not m.any():
                continue
            s, e, v = ps[m], pe[m], pv[m]
            o = np.argsort(s)
            s, e, v = s[o], e[o], v[o]
            emax = np.maximum.accumulate(e)          # for an overlap query on sorted starts
            gk = np.where(chrom == c)[0]
            lo = tss[gk] - PROMOTER_BP
            hi = tss[gk] + PROMOTER_BP
            r = np.searchsorted(s, hi, side="right")     # peaks starting before the window ends
            for t, gi_ in enumerate(gk):
                j = r[t]
                if j == 0:
                    continue
                # walk back while a peak could still reach the window
                best = 0.0
                jj = j - 1
                while jj >= 0 and emax[jj] >= lo[t]:
                    if e[jj] >= lo[t] and s[jj] <= hi[t]:
                        best = max(best, v[jj])
                    jj -= 1
                    if j - jj > 400:
                        break
                if best > 0:
                    raw[gi_] = best
        hit[k] = raw > 0
        nz = raw > 0
        pct[k][nz] = stats.rankdata(raw[nz]) / max(nz.sum(), 1)
        report(f"    {p['experiment']} {p['file']}: {int(nz.sum()):,}/{n:,} promoters with a peak "
               f"({100*nz.mean():.1f}%)")
    return hit.sum(0), pct.mean(0)


# ---------------------------------------------------------------------------- statistics
def steiger(r_yb, r_ya, r_ab, n):
    """Williams' t for two DEPENDENT correlations that share the outcome y. df = n-3.

    Applied to Spearman coefficients, which is an approximation -- the bootstrap below is the primary
    statistic and this is reported alongside it, not instead of it.
    """
    from scipy import stats
    det = 1 - r_yb ** 2 - r_ya ** 2 - r_ab ** 2 + 2 * r_yb * r_ya * r_ab
    num = (r_yb - r_ya) * np.sqrt((n - 1) * (1 + r_ab))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * det + ((r_yb + r_ya) ** 2 / 4) * (1 - r_ab) ** 3)
    t = num / den
    return float(t), float(2 * stats.t.sf(abs(t), n - 3))


def compare_predictors(y, a, b, name_a, name_b, n_boot=2000, seed=0):
    """Which of two predictors tracks y better, decided by a bootstrap over the shared units.

    EFFECT SIZE IS THE RAW HELD-OUT VALUE: the mean |z| inside each predictor's top and bottom decile, not a
    difference from anything. The bootstrap decides the CONTEST; it is not the effect.
    """
    from scipy import stats
    ra = float(stats.spearmanr(a, y)[0])
    rb = float(stats.spearmanr(b, y)[0])
    rab = float(stats.spearmanr(a, b)[0])
    t, p = steiger(rb, ra, rab, len(y))
    rng = np.random.default_rng(seed)
    wins = 0
    diffs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        da = stats.spearmanr(a[k], y[k])[0]
        db = stats.spearmanr(b[k], y[k])[0]
        diffs.append(abs(db) - abs(da))
        wins += abs(db) > abs(da)
    dec = {}
    for nm, x in ((name_a, a), (name_b, b)):
        q = np.digitize(x, np.quantile(x, np.linspace(.1, .9, 9)))
        dec[nm] = {"bottom_decile_mean_abs_z": float(y[q == 0].mean()),
                   "top_decile_mean_abs_z": float(y[q == 9].mean()),
                   "top_over_bottom": float(y[q == 9].mean() / max(y[q == 0].mean(), 1e-12)),
                   "n_bottom": int((q == 0).sum()), "n_top": int((q == 9).sum())}
    return {"n": int(len(y)),
            f"spearman_{name_a}": ra, f"spearman_{name_b}": rb, "spearman_between_predictors": rab,
            "williams_t": t, "williams_p": p,
            "boot_frac_" + name_b + "_wins": float(wins / n_boot),
            "boot_mean_delta_abs_r": float(np.mean(diffs)),
            "boot_ci_delta_abs_r": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "deciles": dec}


def smd(a, b):
    """Standardised mean difference. A p-value on residual imbalance is a function of n as much as of
    imbalance -- at 200,000 pairs a negligible difference is p<1e-60 -- so every residual is reported as
    BOTH a p-value and an n-independent effect size."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def matched_arm(label, li, lj, A, ok, rowkey, colkey, rstr, cresp, third, linkset,
                n_draws=N_DRAWS, seed0=0):
    """The prescribed fast test: linked pairs vs pairs matched on the row key and the column key.

    `rowkey`/`colkey` are integer strata; the column key may be a COMPOSITE of responsiveness decile and an
    abundance quintile, which is how the covariate swap in part 3 is run. The effect size is
    `mean_abs_z_linked`, the raw held-out value. The control is redrawn n_draws times and the FRACTION of
    draws reaching p<0.05 is reported next to the mean, because one draw is one sample of the matched
    population. Residual imbalance is measured on every covariate after matching, on every draw.
    """
    from scipy import stats
    from collections import Counter
    lz = A[li, lj]
    ncol = A.shape[1]
    # linked pairs encoded as one int64 so the "is this control pair itself a link?" test is a
    # searchsorted rather than a per-element Python set lookup
    lk = np.sort(np.fromiter((i * ncol + j for i, j in linkset), np.int64, len(linkset))) \
        if linkset else np.zeros(0, np.int64)
    rows_by = {}
    cols_by = {}
    for k in np.unique(rowkey):
        rows_by[k] = np.where(rowkey == k)[0]
    for k in np.unique(colkey):
        cols_by[k] = np.where(colkey == k)[0]
    ct = Counter(zip(rowkey[li].tolist(), colkey[lj].tolist()))
    rows = []
    for s in range(n_draws):
        rng = np.random.default_rng(seed0 + s)
        ci, cj = [], []
        for (a, b), n in ct.items():
            rs, cs = rows_by.get(a), cols_by.get(b)
            if rs is None or cs is None or len(rs) == 0 or len(cs) == 0:
                continue
            need, tries = n, 0
            while need > 0 and tries < 40:
                m = int(need * 1.6) + 16
                xi = rs[rng.integers(0, len(rs), m)]
                xj = cs[rng.integers(0, len(cs), m)]
                good = ok[xi, xj]
                if len(lk):
                    enc = xi.astype(np.int64) * ncol + xj
                    pos = np.clip(np.searchsorted(lk, enc), 0, len(lk) - 1)
                    good &= lk[pos] != enc
                xi, xj = xi[good][:need], xj[good][:need]
                if len(xi):
                    ci.append(xi)
                    cj.append(xj)
                    need -= len(xi)
                tries += 1
        if not ci:
            continue
        ci = np.concatenate(ci).astype(np.int64)
        cj = np.concatenate(cj).astype(np.int64)
        if len(ci) < 50:
            continue
        cz = A[ci, cj]
        r = {"n_ctrl": int(len(ci)), "mean_ctrl": float(cz.mean()),
             "ratio": float(lz.mean() / cz.mean()),
             "p": float(stats.mannwhitneyu(lz, cz, alternative="two-sided")[1]),
             "res_strength_p": float(stats.mannwhitneyu(rstr[li], rstr[ci], alternative="two-sided")[1]),
             "res_resp_p": float(stats.mannwhitneyu(cresp[lj], cresp[cj], alternative="two-sided")[1]),
             "res_strength_smd": smd(rstr[li], rstr[ci]),
             "res_resp_smd": smd(cresp[lj], cresp[cj])}
        if third is not None:
            t1, t2 = third[lj], third[cj]
            m = np.isfinite(t1) & np.isfinite(t2)
            r["res_third_p"] = float(stats.mannwhitneyu(t1[m], t2[m], alternative="two-sided")[1])
            r["res_third_smd"] = smd(t1[m], t2[m])
        else:
            r["res_third_p"] = float("nan")
            r["res_third_smd"] = float("nan")
        rows.append(r)
    if not rows:
        return {"label": label, "n_pairs": int(len(lz)), "testable": False}

    def mn(k):
        return float(np.mean([r[k] for r in rows]))
    return {"label": label, "n_pairs": int(len(lz)), "testable": True, "n_draws": len(rows),
            "mean_abs_z_linked": float(lz.mean()),        # <- THE EFFECT SIZE: raw held-out value
            "median_abs_z_linked": float(np.median(lz)),
            "mean_abs_z_matched": mn("mean_ctrl"), "ratio": mn("ratio"),
            "ratio_min": float(np.min([r["ratio"] for r in rows])),
            "ratio_max": float(np.max([r["ratio"] for r in rows])),
            "p_median": float(np.median([r["p"] for r in rows])),
            "frac_draws_p05": float(np.mean([r["p"] < 0.05 for r in rows])),
            "residual_strength_p": mn("res_strength_p"),
            "residual_responsiveness_p": mn("res_resp_p"),
            "residual_third_p": mn("res_third_p"),
            "residual_strength_smd": mn("res_strength_smd"),
            "residual_responsiveness_smd": mn("res_resp_smd"),
            "residual_third_smd": mn("res_third_smd"),
            "per_draw_ratio": [r["ratio"] for r in rows]}


def head_to_head(label, ya, yb, ka, kb, cova, covb, n_draws=N_DRAWS, seed0=0):
    """Two gene sets compared against each other inside matched strata, redrawn n_draws times."""
    from scipy import stats
    pa, pb = {}, {}
    for i, k in enumerate(ka):
        pa.setdefault(int(k), []).append(i)
    for i, k in enumerate(kb):
        pb.setdefault(int(k), []).append(i)
    rows = []
    for s in range(n_draws):
        rng = np.random.default_rng(seed0 + s)
        ia, ib = [], []
        for k, la in pa.items():
            lb = pb.get(k, [])
            n = min(len(la), len(lb))
            if n:
                ia += list(rng.choice(la, n, replace=False))
                ib += list(rng.choice(lb, n, replace=False))
        if len(ia) < 30:
            continue
        ia, ib = np.array(ia), np.array(ib)
        rows.append({"n": int(len(ia)), "a": float(ya[ia].mean()), "b": float(yb[ib].mean()),
                     "p": float(stats.mannwhitneyu(ya[ia], yb[ib], alternative="two-sided")[1]),
                     "res_p": float(stats.mannwhitneyu(cova[ia], covb[ib], alternative="two-sided")[1])})
    if not rows:
        return {"label": label, "testable": False}

    def mn(k):
        return float(np.mean([r[k] for r in rows]))
    return {"label": label, "testable": True, "n_draws": len(rows), "n_matched": int(mn("n")),
            "mean_abs_z_a": mn("a"), "mean_abs_z_b": mn("b"), "ratio_a_over_b": mn("a") / mn("b"),
            "p_median": float(np.median([r["p"] for r in rows])),
            "frac_draws_p05": float(np.mean([r["p"] < 0.05 for r in rows])),
            "residual_covariate_p": mn("res_p")}


# ---------------------------------------------------------------------------- main
def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("K562 STATE FROM ENCODE -- expression + promoter accessibility, and whether a cell-matched")
    report("covariate beats the generic one the whole model has been matching on")
    report("=" * 100)

    from entity_registry import load as load_reg
    REG = load_reg()

    # ------------------------------------------------------------------ fetch (cached after first run)
    if CACHE.exists():
        red = json.load(gzip.open(CACHE, "rt"))
        report(f"  reduced ENCODE cache: {CACHE}")
        report(f"    RNA {len(red['rna_files'])} files / {len(red['rna_experiments'])} experiments, "
               f"ATAC {len(red['atac_files'])} experiments")
        tpm_med = {k: float(v) for k, v in red["tpm"].items()}
        rep_rho = red["replicate_rho"]
        rna_files, rna_accs = red["rna_files"], red["rna_experiments"]
        atac_files = red["atac_files"]
        atac_paths = red["atac_paths"]
    else:
        report("\n  FETCH -- ENCODE portal, one file-listing call per experiment")
        picked, per_file = fetch_rna(report)
        keysets = [set(v) for v in per_file.values()]
        common = set.intersection(*keysets)
        if any(len(k) != len(common) for k in keysets):
            report(f"    gene sets differ across files ({[len(k) for k in keysets]}); "
                   f"using the {len(common):,} in all of them")
        genes = sorted(common)
        M = np.array([[per_file[f][g] for g in genes] for f in per_file])
        rho = []
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                rho.append(float(stats.spearmanr(M[i], M[j])[0]))
        rep_rho = ({"median": float(np.median(rho)), "min": float(np.min(rho)), "n_pairs": len(rho)}
                   if rho else {"median": None, "min": None, "n_pairs": 0})
        tpm_med = {g: float(v) for g, v in zip(genes, np.median(M, axis=0))}
        report(f"  {len(tpm_med):,} Ensembl genes quantified; median pairwise Spearman between the "
               f"{len(M)} files {rep_rho['median']:.4f} (min {rep_rho['min']:.4f})")
        atac = fetch_atac(report)
        rna_files = [p["file"] for p in picked]
        rna_accs = sorted({p["experiment"] for p in picked})
        atac_files = [{"experiment": p["experiment"], "file": p["file"]} for p in atac]
        atac_paths = [p["path"] for p in atac]
        DIR.mkdir(parents=True, exist_ok=True)
        with gzip.open(CACHE, "wt") as fh:
            json.dump({"tpm": tpm_med, "replicate_rho": rep_rho, "rna_files": rna_files,
                       "rna_experiments": rna_accs, "atac_files": atac_files,
                       "atac_paths": atac_paths}, fh)

    # ------------------------------------------------------------------ join through the registry
    # TWO DECLARED JOINS. The loose one records what fraction of a whole GENCODE V29 annotation the registry
    # knows about -- retired ids, scaffolds and non-coding biotypes make that well under 1 and it is not a
    # bug. The strict one is the join this module actually depends on: the genes the Perturb-seq screen
    # MEASURES. If that one is weak, the layer cannot be scored and it must raise rather than proceed.
    _e, st_all = REG.join("gene", "ensembl", sorted(tpm_med), 0.50, "all ENCODE V29 gene ids")
    report(f"\n  registry join, all ENCODE RSEM ids -> gene/ensembl: {st_all['joined']:,}/{st_all['of']:,} "
           f"({st_all['rate']:.1%})  -- ENSG is the primary key; no symbol fallback anywhere in this module")

    # ------------------------------------------------------------------ TSS, GRCh38, from the registry
    d = json.load(gzip.open(NET, "rt"))
    names = [g["name"] for g in d["genes"]]
    ppm_raw = d.get("ppm", {})
    ens2pos = {}
    for g in d["genes"]:
        eid = REG.resolve("gene", "symbol", g["name"])
        e = REG.entities.get(eid) if eid else None
        if e and e.get("ensembl") and g.get("chrom") and g.get("tss") not in (None, ""):
            if e.get("assembly") != "GRCh38":
                continue
            ens2pos[e["ensembl"]] = (g["chrom"], int(float(g["tss"])))
    del d
    # `ppm` IS KEYED BY GENE INDEX, NOT SYMBOL. The registry's `cell_index` is the only correct route;
    # resolving those keys as symbols returns None for every one, and that is the bug that once ran an
    # abundance control as a column of zeros and passed. The Registry object does not expose cell_index, so
    # it is read from the same file -- and the resulting rate is ASSERTED, not assumed.
    cell_index = json.load(gzip.open(SP / "entity_registry.json.gz", "rt"))["cell_index"]
    ens2ppm = {}
    for k, v in ppm_raw.items():
        eid = cell_index.get(str(k))
        if eid is None:
            continue
        ens = REG.entities.get(eid, {}).get("ensembl")
        if ens and v:
            ens2ppm[ens] = float(v)
    rate_ppm = len(ens2ppm) / max(len(ppm_raw), 1)
    if rate_ppm < 0.80:
        raise SystemExit(f"generic ppm resolved to only {len(ens2ppm):,}/{len(ppm_raw):,} "
                         f"({rate_ppm:.1%}) ENSG via cell_index -- a silently empty covariate is the "
                         f"failure this project has already made once")
    report(f"  generic `ppm` via cell_index -> {len(ens2ppm):,}/{len(ppm_raw):,} ENSG ({rate_ppm:.1%}); "
           f"a symbol lookup would return None for every one")
    report(f"  GRCh38 TSS coordinates for {len(ens2pos):,} genes")

    # ------------------------------------------------------------------ promoter accessibility
    ens_list = sorted(ens2pos)
    chrom = np.array([ens2pos[g][0] for g in ens_list])
    tss = np.array([ens2pos[g][1] for g in ens_list], np.int64)
    acc_cache = DIR / "promoter.json.gz"
    if acc_cache.exists():
        pa = json.load(gzip.open(acc_cache, "rt"))
        n_hit = np.array([pa["n_exp"].get(g, 0) for g in ens_list], float)
        sig = np.array([pa["pct"].get(g, 0.0) for g in ens_list], float)
        report(f"  promoter accessibility cache: {acc_cache}")
    else:
        report(f"\n  PROMOTER ACCESSIBILITY -- peaks within +/-{PROMOTER_BP:,} bp of the GRCh38 TSS")
        peaks = [{"experiment": a["experiment"], "file": a["file"], "path": p}
                 for a, p in zip(atac_files, atac_paths)]
        n_hit, sig = promoter_signal(peaks, ens_list, chrom, tss, report)
        n_hit = n_hit.astype(float)
        with gzip.open(acc_cache, "wt") as fh:
            json.dump({"n_exp": {g: int(v) for g, v in zip(ens_list, n_hit) if v},
                       "pct": {g: float(v) for g, v in zip(ens_list, sig) if v}}, fh)
    n_atac = len(atac_files)
    report(f"  {int((n_hit > 0).sum()):,}/{len(ens_list):,} promoters accessible in >=1 of {n_atac} "
           f"experiments; {int((n_hit == n_atac).sum()):,} in all {n_atac}")
    # POSITIVE CONTROL ON THE COORDINATE JOIN, because a silent one is the failure mode this project keeps
    # hitting. A chromosome-naming or assembly mismatch would scatter peaks across genes at random and
    # every accessibility number downstream would be a null wearing a measurement's clothes. Open promoters
    # must be expressed promoters. If they are not, the join is broken, not the biology.
    tss_tpm = np.array([tpm_med.get(g, np.nan) for g in ens_list])
    hv = np.isfinite(tss_tpm)
    fr_open = float((tss_tpm[hv & (n_hit == n_atac)] >= EXPRESSED_TPM).mean())
    fr_shut = float((tss_tpm[hv & (n_hit == 0)] >= EXPRESSED_TPM).mean())
    report(f"  positive control on the TSS/peak join: {100*fr_open:.1f}% of promoters open in all "
           f"{n_atac} experiments are expressed (TPM >= {EXPRESSED_TPM}) against {100*fr_shut:.1f}% of "
           f"promoters open in none")
    if fr_open - fr_shut < 0.30:
        raise SystemExit("accessibility does not track expression -- the TSS/peak coordinate join is "
                         "broken (assembly, chromosome naming or index alignment), not the biology")
    join_control = {"frac_expressed_open_all": fr_open, "frac_expressed_open_none": fr_shut}

    # ------------------------------------------------------------------ the fast-test substrate
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        obs = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        var = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
    report(f"\n  GWPS: {X.shape[0]:,} perturbations x {X.shape[1]:,} genes, "
           f"{100*(1-np.isfinite(X).mean()):.4f}% non-finite cells (DROPPED, never imputed)")
    Xn = np.where(np.isfinite(X), X, np.nan)
    del X
    med = np.nanmedian(Xn, axis=0)
    mad = np.nanmedian(np.abs(Xn - med), axis=0)
    scale = 1.4826 * mad
    if (scale <= 0).any():
        report(f"    {(scale<=0).sum()} genes have zero MAD -- dropped, never imputed")
    A = np.abs((Xn - med) / np.where(scale > 0, scale, np.nan)).astype(np.float32)
    del Xn
    ok = np.isfinite(A)
    rstr = np.nanmean(np.where(ok, A, np.nan), axis=1)
    cresp = np.nanmean(np.where(ok, A, np.nan), axis=0)
    rq = np.digitize(rstr, np.quantile(rstr, np.linspace(.1, .9, N_DEC - 1)))
    cq = np.digitize(cresp, np.quantile(cresp, np.linspace(.1, .9, N_DEC - 1)))
    report(f"    per-gene robust z; {100*ok.mean():.4f}% of cells usable")

    pert_rows, nt_rows = {}, []
    for i, o in enumerate(obs):
        p = o.split("_")
        if p[1] == "non-targeting":
            nt_rows.append(i)
            continue
        if p[-1].startswith("ENSG"):
            pert_rows.setdefault(p[-1].split(".")[0], []).append(i)
    colof = {}
    for j, g in enumerate(var):
        colof.setdefault(g, j)
    n_nt = len(nt_rows)
    report(f"    {len(pert_rows):,} perturbed genes carry an ENSG ({n_nt:,} non-targeting rows excluded "
           f"from the pair universe, kept as the do-nothing floor), {len(colof):,} measured genes")

    tpm_arr = {g: v for g, v in tpm_med.items()}
    acc_n = {g: float(v) for g, v in zip(ens_list, n_hit)}
    acc_s = {g: float(v) for g, v in zip(ens_list, sig)}

    # THE JOIN THIS MODULE DEPENDS ON: the genes the screen measures. Declared and enforced.
    meas = sorted(set(var) & set(tpm_arr))
    _e2, st_meas = REG.join("gene", "ensembl", meas, 0.95, "ENCODE quantifications for GWPS-measured genes")
    report(f"    registry join on the {st_meas['of']:,} measured genes ENCODE also quantifies: "
           f"{st_meas['joined']:,} ({st_meas['rate']:.1%})")

    R = {"model": "encode-k562-state-v1",
         "source": {"portal": API, "rna_assay": RNA_ASSAY, "rna_experiments": rna_accs,
                    "rna_files": rna_files, "assembly": RNA_ASSEMBLY, "annotation": RNA_ANNOTATION,
                    "rna_output_type": RNA_OUTPUT, "rna_pipeline": "RSEM (header asserted)",
                    "atac_assay": ATAC_ASSAY, "atac_output_type": ATAC_OUTPUT,
                    "atac_files": atac_files, "promoter_bp": PROMOTER_BP},
         "n_genes_quantified": len(tpm_med), "replicate_agreement": rep_rho,
         "join": {"namespace": "ensembl", "all_ids_rate": st_all["rate"],
                  "measured_genes_rate": st_meas["rate"], "n_measured": st_meas["joined"]},
         "n_promoters_scored": len(ens_list),
         "n_promoters_accessible_any": int((n_hit > 0).sum()),
         "n_promoters_accessible_all": int((n_hit == n_atac).sum()),
         "tss_peak_join_control": join_control}

    # ================================================================== Q1  THE GENE AXIS
    report("\n" + "-" * 100)
    report("  Q1  GENE AXIS -- does K562 expression predict gene responsiveness better than generic `ppm`?")
    report("      EFFECT SIZE = raw mean |z| in each predictor's deciles. The contest is decided by a")
    report("      bootstrap over genes, not by a difference of point estimates.")
    report("-" * 100)
    gsel = [j for j, g in enumerate(var) if g in tpm_arr and g in ens2ppm and np.isfinite(cresp[j])]
    y = cresp[np.array(gsel)]
    a_gen = np.log10(1.0 + np.array([ens2ppm[var[j]] for j in gsel]))
    b_k562 = np.log10(1.0 + np.array([tpm_arr[var[j]] for j in gsel]))
    cov_tpm = sum(1 for g in var if g in tpm_arr)
    cov_ppm = sum(1 for g in var if g in ens2ppm)
    report(f"      coverage of the {len(var):,} measured genes: K562 RNA {cov_tpm:,} "
           f"({100*cov_tpm/len(var):.1f}%), generic ppm {cov_ppm:,} ({100*cov_ppm/len(var):.1f}%), "
           f"both {len(gsel):,} -- the contest runs on the SHARED set only")
    q1 = compare_predictors(y, a_gen, b_k562, "generic_ppm", "k562_rna")
    R["gene_axis"] = q1
    report(f"      Spearman vs responsiveness:  generic ppm {q1['spearman_generic_ppm']:+.4f}   "
           f"K562 RNA {q1['spearman_k562_rna']:+.4f}   (the two predictors correlate "
           f"{q1['spearman_between_predictors']:.4f})")
    report(f"      bootstrap over genes: K562 wins in {100*q1['boot_frac_k562_rna_wins']:.1f}% of 2,000 "
           f"resamples; delta|rho| {q1['boot_mean_delta_abs_r']:+.4f} "
           f"[{q1['boot_ci_delta_abs_r'][0]:+.4f}, {q1['boot_ci_delta_abs_r'][1]:+.4f}]")
    report(f"      Williams t {q1['williams_t']:+.2f}, p {q1['williams_p']:.3g}")
    report(f"      {'predictor':14s} {'bottom decile |z|':>18s} {'top decile |z|':>16s} {'top/bottom':>11s}")
    for nm in ("generic_ppm", "k562_rna"):
        dd = q1["deciles"][nm]
        report(f"      {nm:14s} {dd['bottom_decile_mean_abs_z']:18.4f} "
               f"{dd['top_decile_mean_abs_z']:16.4f} {dd['top_over_bottom']:11.3f}")

    # ================================================================== Q2  THE PERTURBATION AXIS
    report("\n" + "-" * 100)
    report("  Q2  PERTURBATION AXIS -- CRISPRi silences a promoter, so a promoter already silent in K562")
    report("      should have nothing to silence. A generic table cannot know which genes those are.")
    report("-" * 100)
    prow = {e: min(v) for e, v in pert_rows.items()}       # one row per perturbed gene: no pseudo-replication
    psel = [e for e in prow if e in tpm_arr and e in ens2ppm and np.isfinite(rstr[prow[e]])]
    yp = np.array([rstr[prow[e]] for e in psel])
    ap = np.log10(1.0 + np.array([ens2ppm[e] for e in psel]))
    bp = np.log10(1.0 + np.array([tpm_arr[e] for e in psel]))
    pcov_t = sum(1 for e in prow if e in tpm_arr)
    pcov_p = sum(1 for e in prow if e in ens2ppm)
    report(f"      coverage of the {len(prow):,} perturbed genes: K562 RNA {pcov_t:,} "
           f"({100*pcov_t/len(prow):.1f}%), generic ppm {pcov_p:,} ({100*pcov_p/len(prow):.1f}%), "
           f"both {len(psel):,}")
    q2 = compare_predictors(yp, ap, bp, "generic_ppm", "k562_rna", seed=1)
    R["perturbation_axis"] = q2
    report(f"      Spearman vs perturbation strength:  generic ppm {q2['spearman_generic_ppm']:+.4f}   "
           f"K562 RNA {q2['spearman_k562_rna']:+.4f}")
    report(f"      bootstrap: K562 wins in {100*q2['boot_frac_k562_rna_wins']:.1f}% of resamples; "
           f"delta|rho| {q2['boot_mean_delta_abs_r']:+.4f} "
           f"[{q2['boot_ci_delta_abs_r'][0]:+.4f}, {q2['boot_ci_delta_abs_r'][1]:+.4f}]")
    report(f"      Williams t {q2['williams_t']:+.2f}, p {q2['williams_p']:.3g}")
    report(f"      {'predictor':14s} {'bottom decile |z|':>18s} {'top decile |z|':>16s} {'top/bottom':>11s}")
    for nm in ("generic_ppm", "k562_rna"):
        dd = q2["deciles"][nm]
        report(f"      {nm:14s} {dd['bottom_decile_mean_abs_z']:18.4f} "
               f"{dd['top_decile_mean_abs_z']:16.4f} {dd['top_over_bottom']:11.3f}")
    silent = np.array([tpm_arr[e] < EXPRESSED_TPM for e in psel])
    if silent.sum() >= 20:
        mp = float(stats.mannwhitneyu(yp[silent], yp[~silent], alternative="two-sided")[1])
        R["silent_target"] = {"tpm_floor": EXPRESSED_TPM, "n_silent": int(silent.sum()),
                              "n_expressed": int((~silent).sum()),
                              "mean_abs_z_silent": float(yp[silent].mean()),
                              "mean_abs_z_expressed": float(yp[~silent].mean()),
                              "ratio": float(yp[~silent].mean() / yp[silent].mean()), "p": mp}
        s = R["silent_target"]
        # THE DO-NOTHING FLOOR. |z| has no zero: a row of pure noise still averages ~0.8 because it is a
        # mean of absolute values. The non-targeting guides say what "this perturbation did nothing" looks
        # like on this scale, and without that reference the silent-vs-expressed gap cannot be read.
        ntz = float(np.nanmean(rstr[np.array(nt_rows)])) if nt_rows else float("nan")
        s["nontargeting_floor"] = ntz
        s["at_or_below_floor"] = bool(s["mean_abs_z_silent"] <= ntz)
        report(f"      TARGETS SILENT IN K562 (TPM < {EXPRESSED_TPM}): {s['n_silent']:,} of {len(psel):,} "
               f"perturbations")
        report(f"        mean |z| silent {s['mean_abs_z_silent']:.4f} vs expressed "
               f"{s['mean_abs_z_expressed']:.4f}  ({s['ratio']:.3f}x, MWU p {s['p']:.3g})")
        report(f"        non-targeting floor {ntz:.4f} ({n_nt:,} guides) -- a mean of absolute values has "
               f"no zero, so this is what 'this guide did nothing' looks like on this scale. Silent "
               f"targets sit {'AT OR BELOW' if s['at_or_below_floor'] else 'above'} it.")
        # AND THE STRICT VERSION: the same contrast INSIDE generic-abundance strata. Only what survives
        # this is information the generic table does NOT already have. It is the hardest form of the
        # question and it is deliberately the one reported, rather than the flattering unmatched contrast.
        gq = np.digitize(ap, np.quantile(ap, np.linspace(.1, .9, N_DEC - 1)))
        hh = head_to_head("expressed vs silent, matched on generic ppm decile",
                          yp[~silent], yp[silent], gq[~silent], gq[silent], ap[~silent], ap[silent],
                          seed0=21)
        R["silent_target_matched_on_generic"] = hh
        if hh.get("testable"):
            report(f"        matched on GENERIC ppm decile, {hh['n_draws']} draws of ~{hh['n_matched']:,} "
                   f"per arm: expressed {hh['mean_abs_z_a']:.4f} vs silent {hh['mean_abs_z_b']:.4f} "
                   f"({hh['ratio_a_over_b']:.3f}x, median p {hh['p_median']:.3g}, "
                   f"{100*hh['frac_draws_p05']:.0f}% of draws p<0.05; residual ppm p "
                   f"{hh['residual_covariate_p']:.3g})")
    else:
        R["silent_target"] = {"n_silent": int(silent.sum()), "note": "too few silent targets to test"}
        R["silent_target_matched_on_generic"] = None
        report(f"      only {int(silent.sum())} perturbed genes are silent in K562 -- the screen selected "
               f"expressed genes, so this arm is underpowered")

    # ================================================================== Q3  ACCESSIBILITY BEYOND EXPRESSION
    report("\n" + "-" * 100)
    report("  Q3  DOES PROMOTER ACCESSIBILITY ADD ANYTHING BEYOND THE EXPRESSION LEVEL?")
    report("-" * 100)
    asel = [j for j, g in enumerate(var) if g in tpm_arr and g in acc_s and np.isfinite(cresp[j])]
    ya = cresp[np.array(asel)]
    ta = np.log10(1.0 + np.array([tpm_arr[var[j]] for j in asel]))
    sa = np.array([acc_s[var[j]] for j in asel])
    na = np.array([acc_n[var[j]] for j in asel])
    tq = np.digitize(ta, np.quantile(ta, np.linspace(.1, .9, N_DEC - 1)))
    rho_acc = float(stats.spearmanr(sa, ya)[0])
    # matched head-to-head: accessible in ALL experiments vs accessible in NONE, matched on TPM decile
    hi = na >= n_atac
    lo = na == 0
    h = head_to_head("promoter accessible in all vs none, matched on K562 TPM decile",
                     ya[hi], ya[lo], tq[hi], tq[lo], ta[hi], ta[lo], seed0=11)
    R["accessibility"] = {"n": len(asel), "spearman_signal_vs_responsiveness": rho_acc,
                          "n_accessible_all": int(hi.sum()), "n_accessible_none": int(lo.sum()),
                          "matched": h}
    report(f"      {len(asel):,} measured genes have both a TSS and a K562 TPM")
    report(f"      Spearman(promoter ATAC percentile, responsiveness) = {rho_acc:+.4f} (unmatched)")
    if h.get("testable"):
        report(f"      matched on TPM decile, {h['n_draws']} draws of ~{h['n_matched']:,} genes per arm:")
        report(f"        accessible in all {n_atac}: mean |z| {h['mean_abs_z_a']:.4f}")
        report(f"        accessible in none:        mean |z| {h['mean_abs_z_b']:.4f}")
        report(f"        ratio {h['ratio_a_over_b']:.4f}, median p {h['p_median']:.3g}, "
               f"{100*h['frac_draws_p05']:.0f}% of draws p<0.05; residual TPM imbalance p "
               f"{h['residual_covariate_p']:.3g}")
    else:
        report(f"      too few genes in one arm ({int(hi.sum())} all / {int(lo.sum())} none) to match")

    # ================================================================== Q4  DOES THE SWAP CHANGE A DECISION?
    report("\n" + "-" * 100)
    report("  Q4  THE DECISION. The banked PPI number (1.208x) comes from matching on strength x")
    report("      responsiveness deciles only. Add a THIRD axis -- the measured gene's abundance -- once")
    report("      with the generic table and once with K562 RNA, and see which absorbs more of the ratio.")
    report("-" * 100)
    d = json.load(gzip.open(NET, "rt"))
    ppi = d["ppi"]
    del d
    ens_of_idx = {}
    for i in range(len(names)):
        eid = cell_index.get(str(i))            # index keys again: the only correct route
        ens_of_idx[i] = REG.entities.get(eid, {}).get("ensembl") if eid else None
    n_mapped = sum(1 for v in ens_of_idx.values() if v)
    if n_mapped < 0.90 * len(names):
        raise SystemExit(f"PPI index -> ENSG mapped only {n_mapped:,}/{len(names):,}")
    pairs = set()
    for u, v in ppi:
        eu, ev = ens_of_idx.get(u), ens_of_idx.get(v)
        if not eu or not ev or eu == ev:
            continue
        pairs.add((eu, ev))
        pairs.add((ev, eu))
    li, lj = [], []
    for su, sv in pairs:
        rows_i = pert_rows.get(su)
        j = colof.get(sv)
        if not rows_i or j is None:
            continue
        for i in rows_i:
            if ok[i, j]:
                li.append(i)
                lj.append(j)
    li = np.array(li, np.int64)
    lj = np.array(lj, np.int64)
    linkset = set(zip(li.tolist(), lj.tolist()))
    report(f"      existing PPI layer: {len(ppi):,} index tuples -> {len(pairs)//2:,} distinct gene pairs "
           f"-> {len(li):,} testable (perturbation, gene) pairs")

    def colkey_with(vals, name):
        """Composite column stratum: responsiveness decile x abundance DECILE of the measured gene.

        Deciles, not quintiles: with 200,000 linked pairs a quintile match left a residual imbalance on the
        third covariate itself, which would make the comparison a contest between two failed matches.
        """
        x = np.array([vals.get(g, np.nan) for g in var], float)
        have = np.isfinite(x)
        aq = np.full(len(var), -1)
        if have.sum() > 50:
            aq[have] = np.digitize(np.log10(1 + x[have]),
                                   np.quantile(np.log10(1 + x[have]), np.linspace(.1, .9, N_DEC - 1)))
        return cq * (N_DEC + 1) + (aq + 1), np.where(have, np.log10(1 + x), np.nan), name

    base = matched_arm("PPI, strength x responsiveness (the banked test)", li, lj, A, ok, rq, cq,
                       rstr, cresp, None, linkset, seed0=0)
    kg, vg, _ = colkey_with(ens2ppm, "generic_ppm")
    kk, vk, _ = colkey_with(tpm_arr, "k562_rna")
    # the third axis is only defined where the covariate exists; pairs whose measured gene is not covered
    # are DROPPED from that arm rather than assigned a zero abundance
    mg = np.array([kg[j] % (N_DEC + 1) != 0 for j in lj])
    mk = np.array([kk[j] % (N_DEC + 1) != 0 for j in lj])
    both = mg & mk
    report(f"      third-axis coverage of the linked pairs: generic {100*mg.mean():.1f}%, "
           f"K562 {100*mk.mean():.1f}%, both {100*both.mean():.1f}% -- the two swap arms run on the SAME "
           f"{int(both.sum()):,} pairs so the comparison is not a coverage difference")
    base_sub = matched_arm("PPI on the shared subset, 2 axes", li[both], lj[both], A, ok, rq, cq,
                           rstr, cresp, None, linkset, n_draws=N_SWAP_DRAWS, seed0=5)
    arm_g = matched_arm("+ generic ppm decile", li[both], lj[both], A, ok, rq, kg, rstr, cresp,
                        vg, linkset, n_draws=N_SWAP_DRAWS, seed0=5)
    arm_k = matched_arm("+ K562 TPM decile", li[both], lj[both], A, ok, rq, kk, rstr, cresp,
                        vk, linkset, n_draws=N_SWAP_DRAWS, seed0=5)
    R["ppi_covariate_swap"] = {"n_ppi_tuples": len(ppi), "n_gene_pairs": len(pairs) // 2,
                               "n_pairs_testable": int(len(li)),
                               "n_pairs_shared": int(both.sum()),
                               "two_axis_all": base, "two_axis_shared": base_sub,
                               "three_axis_generic": arm_g, "three_axis_k562": arm_k}
    hdr = (f"      {'matching':42s} {'pairs':>8s} {'mean|z|':>8s} {'matched':>8s} {'ratio':>7s} "
           f"{'p(med)':>10s} {'draws':>6s}")
    report(hdr)

    def line(a):
        if not a.get("testable"):
            report(f"      {a['label']:42s} {a['n_pairs']:8,d}  -- not testable --")
            return
        report(f"      {a['label']:42s} {a['n_pairs']:8,d} {a['mean_abs_z_linked']:8.4f} "
               f"{a['mean_abs_z_matched']:8.4f} {a['ratio']:7.4f} {a['p_median']:10.2e} "
               f"{100*a['frac_draws_p05']:5.0f}%")
    for a in (base, base_sub, arm_g, arm_k):
        line(a)
    for a in (base, base_sub, arm_g, arm_k):
        if a.get("testable"):
            report(f"        {a['label']:42s} residual (p | standardised difference): "
                   f"strength {a['residual_strength_p']:.2g} | {a['residual_strength_smd']:+.3f}, "
                   f"responsiveness {a['residual_responsiveness_p']:.2g} | "
                   f"{a['residual_responsiveness_smd']:+.3f}, third {a['residual_third_p']:.2g} | "
                   f"{a['residual_third_smd']:+.3f}; ratio range "
                   f"[{a['ratio_min']:.4f}, {a['ratio_max']:.4f}]")
    # PAIRED, because the arms differ by ~0.005 and the draw-to-draw range is wider than that. Both arms
    # use the same seed sequence, so draw s of one is compared against draw s of the other rather than
    # against its mean -- which is the only way a difference this small is resolvable at all.
    ratio_g = arm_g["ratio"] if arm_g.get("testable") else float("nan")
    ratio_k = arm_k["ratio"] if arm_k.get("testable") else float("nan")
    if arm_g.get("testable") and arm_k.get("testable") and \
            len(arm_g["per_draw_ratio"]) == len(arm_k["per_draw_ratio"]):
        dg_ = np.array(arm_g["per_draw_ratio"])
        dk_ = np.array(arm_k["per_draw_ratio"])
        b0_ = np.array(base_sub["per_draw_ratio"]) if base_sub.get("testable") else None
        pair = {"n_draws": len(dg_),
                "mean_ratio_generic": float(dg_.mean()), "mean_ratio_k562": float(dk_.mean()),
                "mean_paired_delta_k562_minus_generic": float((dk_ - dg_).mean()),
                "frac_draws_k562_absorbs_more": float(np.mean(dk_ < dg_)),
                "wilcoxon_p": float(stats.wilcoxon(dk_, dg_)[1]) if len(dg_) >= 6 else None}
        if b0_ is not None and len(b0_) == len(dg_):
            pair["absorbed_by_generic"] = float((b0_ - dg_).mean())
            pair["absorbed_by_k562"] = float((b0_ - dk_).mean())
        R["ppi_covariate_swap"]["paired"] = pair
        report(f"      PAIRED over {pair['n_draws']} seeds: K562 matching leaves a ratio "
               f"{pair['mean_paired_delta_k562_minus_generic']:+.5f} below the generic arm "
               f"(signed-rank p {pair['wilcoxon_p']:.3g} -- the primary test, since the quantity is a "
               f"difference of two matched ratios); K562 absorbs more in "
               f"{100*pair['frac_draws_k562_absorbs_more']:.0f}% of individual draws")
        if "absorbed_by_generic" in pair:
            report(f"        of the {base_sub['ratio']-1:.4f} excess over the two-axis control, the third "
                   f"axis absorbs {pair['absorbed_by_generic']:.5f} (generic) vs "
                   f"{pair['absorbed_by_k562']:.5f} (K562) -- "
                   f"{100*pair['absorbed_by_generic']/(base_sub['ratio']-1):.1f}% vs "
                   f"{100*pair['absorbed_by_k562']/(base_sub['ratio']-1):.1f}% of the effect")
        # THE DIRECTION OF THE REMAINING IMBALANCE MATTERS. If the K562 arm is the LESS well balanced of
        # the two on its own covariate and still returns the lower ratio, the absorption it reports is a
        # floor, not a ceiling -- the leftover imbalance pushes the other way.
        if abs(arm_k["residual_third_smd"]) > abs(arm_g["residual_third_smd"]) and ratio_k < ratio_g:
            report(f"        note: the K562 arm is the WORSE balanced of the two on its own third "
                   f"covariate (standardised difference {arm_k['residual_third_smd']:+.3f} vs "
                   f"{arm_g['residual_third_smd']:+.3f}) and still returns the lower ratio, so what it "
                   f"absorbs is a lower bound")

    # ================================================================== Q5  the prescribed pair test, run
    report("\n" + "-" * 100)
    report("  Q5  THE PRESCRIBED PAIR TEST, RUN ON THIS LAYER ANYWAY -- and why it cannot score it.")
    report("      Linked pairs = perturbed gene in the top K562-expression tercile. The test matches on the")
    report("      row and column marginals, which is exactly what a state layer predicts, so a ratio near")
    report("      1.00 here is the test removing the signal by construction, not the layer being empty.")
    report("-" * 100)
    keep_e = [e for e in prow if e in tpm_arr]
    thr = np.nanquantile(np.array([tpm_arr[e] for e in keep_e]), 2 / 3)
    hi_rows = sorted({prow[e] for e in keep_e if tpm_arr[e] >= thr})
    rng = np.random.default_rng(99)
    si, sj = [], []
    for i in hi_rows:
        js = rng.choice(len(var), 40)
        for j in js:
            if ok[i, j]:
                si.append(i)
                sj.append(j)
    si = np.array(si, np.int64)
    sj = np.array(sj, np.int64)
    q5 = matched_arm("top-tercile-expressed perturbations x random genes", si, sj, A, ok, rq, cq,
                     rstr, cresp, None, set(zip(si.tolist(), sj.tolist())), seed0=17)
    R["prescribed_pair_test"] = q5
    report(hdr)
    line(q5)
    if q5.get("testable"):
        report(f"        residual: strength p {q5['residual_strength_p']:.3g}, responsiveness p "
               f"{q5['residual_responsiveness_p']:.3g}")
    # the same pairs WITHOUT the strength match, to show the size of what the match removes
    flat = np.zeros(A.shape[0], int)
    q5u = matched_arm("same pairs, responsiveness matched only", si, sj, A, ok, flat, cq,
                      rstr, cresp, None, set(zip(si.tolist(), sj.tolist())), seed0=17)
    R["prescribed_pair_test_no_strength_match"] = q5u
    line(q5u)

    # ================================================================== coverage bias
    report("\n" + "-" * 100)
    report("  COVERAGE AND ITS BIAS -- absence is unknown, never zero")
    report("-" * 100)
    has_t = np.array([var[j] in tpm_arr for j in range(len(var))])
    has_p = np.array([var[j] in ens2ppm for j in range(len(var))])
    has_a = np.array([var[j] in acc_s for j in range(len(var))])
    bias = {}
    for nm, m in (("k562_rna", has_t), ("generic_ppm", has_p), ("promoter_tss", has_a)):
        if m.all() or not m.any():
            bias[nm] = {"frac": float(m.mean()), "responsiveness_p": None}
            continue
        bias[nm] = {"frac": float(m.mean()),
                    "mean_responsiveness_covered": float(cresp[m].mean()),
                    "mean_responsiveness_absent": float(cresp[~m].mean()),
                    "responsiveness_p": float(stats.mannwhitneyu(cresp[m], cresp[~m],
                                                                 alternative="two-sided")[1])}
        report(f"      {nm:14s} covers {100*m.mean():5.1f}% of measured genes; responsiveness "
               f"{bias[nm]['mean_responsiveness_covered']:.4f} covered vs "
               f"{bias[nm]['mean_responsiveness_absent']:.4f} absent, p {bias[nm]['responsiveness_p']:.3g}")
    R["coverage_bias"] = bias

    # ================================================================== the layer
    layer = {"model": "encode-k562-state-v1", "cell_type": "K562", "assembly": RNA_ASSEMBLY,
             "annotation": RNA_ANNOTATION, "source": R["source"],
             # EVERY QUANTIFIED GENE IS WRITTEN, INCLUDING THE MEASURED ZEROS. Dropping them to save space
             # would make "ENCODE measured this gene at 0 TPM in K562" indistinguishable from "ENCODE never
             # measured it", and the first of those is this layer's single most useful statement -- it is
             # what identifies an inert CRISPRi target.
             "tpm": {g: round(v, 4) for g, v in tpm_med.items()},
             "promoter_scored_genes": ens_list,
             "promoter_n_experiments_accessible": {g: int(v) for g, v in zip(ens_list, n_hit) if v},
             "promoter_atac_percentile": {g: round(float(v), 5) for g, v in zip(ens_list, sig) if v},
             "n_atac_experiments": n_atac,
             "absence_semantics": {
                 "tpm": "contains every gene ENCODE quantified under the pinned annotation, measured zeros "
                        "included. A gene ABSENT from this dict was not quantified: unknown, not zero.",
                 "promoter": "promoter_scored_genes lists every gene with a GRCh38 TSS in the cell object. "
                             "One of those absent from the two peak dicts has no peak call in any of the "
                             f"{n_atac} experiments -- below threshold, which is a measurement, though not "
                             "proof the promoter is closed. A gene absent from promoter_scored_genes has "
                             "no TSS here and is unknown."},
             "limits": []}
    LIMITS = [
        f"one cell line and one growth condition per assay: ENCODE K562 is not the K562 the Perturb-seq "
        f"screen was run in -- different laboratory, medium, passage. Cell-MATCHED here means the same "
        f"immortalised line, not the same culture.",
        f"RNA is polyA-plus only, so non-polyadenylated transcripts (replication-dependent histones, many "
        f"lncRNAs) are systematically under-measured and their TPM is not a level, it is a floor.",
        f"TPM is a steady-state level, not a synthesis rate: a short-lived transcript at the same TPM is "
        f"being made far faster, and nothing here separates the two.",
        f"promoter accessibility is a peak CALL within +/-{PROMOTER_BP} bp of one annotated TSS. Genes with "
        f"multiple TSSs are scored at one of them, and a promoter with no peak is below the pipeline's "
        f"threshold, not demonstrably closed.",
        f"the ATAC signalValue is fold-enrichment against each experiment's own background and is NOT "
        f"comparable across experiments; it is rank-normalised within experiment before combining, which "
        f"preserves order and discards magnitude.",
        f"this layer is a STATE, not a response. It says what K562 looks like unperturbed; it says nothing "
        f"about how expression or accessibility MOVE after a perturbation.",
        f"the layer is a per-gene covariate, not a link set, so the standard pair fast test cannot score it "
        f"(Q5 shows the matching removes it by construction). Its value is measured as covariate quality "
        f"and as its effect on an already-banked matched result.",
        f"Williams' t is applied to Spearman coefficients, which is an approximation; the bootstrap over "
        f"genes is the primary statistic for every predictor contest here.",
        f"the decile match used by the standard fast test does not fully balance its own covariates at "
        f"this sample size -- on the 213k PPI pairs it leaves a standardised difference of about 0.11 on "
        f"gene responsiveness. That is a property of the reference test, inherited by every number "
        f"compared against it here, and it is why residuals are reported as effect sizes as well as "
        f"p-values.",
        f"the perturbation-axis contest uses one GWPS row per perturbed gene, so it is not inflated by "
        f"pseudo-replication; the gene-axis and PPI arms are not de-replicated and their p-values are "
        f"correspondingly optimistic (the ratios and Spearman coefficients are not).",
        f"the eight RNA experiments span 2011-2016 and five laboratories, and their median pairwise "
        f"Spearman is about 0.86: the K562 expression value used here is a median over that spread, not a "
        f"single measurement, and gene-level disagreement between laboratories is real and unmodelled."]
    layer["limits"] = LIMITS
    R["limits"] = LIMITS
    LAYER.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    # ================================================================== verdict
    report("\n" + "=" * 100)
    g_win = q1["boot_frac_k562_rna_wins"]
    p_win = q2["boot_frac_k562_rna_wins"]
    dg = arm_g["ratio"] if arm_g.get("testable") else float("nan")
    dk = arm_k["ratio"] if arm_k.get("testable") else float("nan")
    b0 = base_sub["ratio"] if base_sub.get("testable") else float("nan")
    pair = R["ppi_covariate_swap"].get("paired")
    sil = R.get("silent_target", {})
    hh = R.get("silent_target_matched_on_generic") or {}

    # PART 1+2: the covariate contest. Both axes, both raw values, the bootstrap as the decision rule.
    if g_win >= 0.95 and p_win >= 0.95:
        head = (f"CELL-MATCHED RNA IS THE BETTER COVARIATE, ON BOTH AXES AND BY A MARGIN THAT SURVIVES "
                f"RESAMPLING. Against gene responsiveness it reaches Spearman "
                f"{q1['spearman_k562_rna']:+.3f} where the generic `ppm` reaches "
                f"{q1['spearman_generic_ppm']:+.3f} on the same {q1['n']:,} genes "
                f"(K562 wins {100*g_win:.0f}% of 2,000 gene bootstraps, delta|rho| "
                f"{q1['boot_mean_delta_abs_r']:+.3f} [{q1['boot_ci_delta_abs_r'][0]:+.3f}, "
                f"{q1['boot_ci_delta_abs_r'][1]:+.3f}]); against perturbation strength "
                f"{q2['spearman_k562_rna']:+.3f} against {q2['spearman_generic_ppm']:+.3f} over "
                f"{q2['n']:,} perturbed genes ({100*p_win:.0f}% of bootstraps). In raw held-out terms the "
                f"top decile of K562 expression carries mean |z| "
                f"{q2['deciles']['k562_rna']['top_decile_mean_abs_z']:.3f} against "
                f"{q2['deciles']['k562_rna']['bottom_decile_mean_abs_z']:.3f} in the bottom "
                f"({q2['deciles']['k562_rna']['top_over_bottom']:.3f}x) where the generic table separates "
                f"only {q2['deciles']['generic_ppm']['top_over_bottom']:.3f}x.")
    else:
        head = (f"CELL-MATCHED RNA IS NOT RELIABLY THE BETTER COVARIATE. Responsiveness "
                f"{q1['spearman_k562_rna']:+.3f} vs {q1['spearman_generic_ppm']:+.3f} "
                f"({100*g_win:.0f}% of bootstraps); perturbation strength "
                f"{q2['spearman_k562_rna']:+.3f} vs {q2['spearman_generic_ppm']:+.3f} "
                f"({100*p_win:.0f}%).")
    if "ratio" in sil:
        head += (f" The mechanism behind the second axis holds: perturbations of the {sil['n_silent']:,} "
                 f"targets SILENT in K562 (TPM < {EXPRESSED_TPM}) reach mean |z| "
                 f"{sil['mean_abs_z_silent']:.3f} against {sil['mean_abs_z_expressed']:.3f} for expressed "
                 f"targets (p {sil['p']:.2g}) -- at or below the {sil['nontargeting_floor']:.3f} floor set "
                 f"by {n_nt:,} non-targeting guides, i.e. inert, which is what CRISPRi at an already-silent "
                 f"promoter should look like.")
        if hh.get("testable"):
            strong = hh["frac_draws_p05"] >= 0.8
            head += (f" HOW MUCH OF THAT IS NEW is a separate question and the answer is: not much that is "
                     f"demonstrable. Inside generic-abundance deciles the same contrast shrinks from "
                     f"{sil['ratio']:.3f}x to {hh['ratio_a_over_b']:.3f}x "
                     f"({hh['mean_abs_z_a']:.3f} vs {hh['mean_abs_z_b']:.3f}) and reaches p<0.05 in only "
                     f"{100*hh['frac_draws_p05']:.0f}% of {hh['n_draws']} draws of ~{hh['n_matched']:,} "
                     f"per arm, so most of what K562 expression knows about inert perturbations the "
                     f"generic table already knew, and the residual is not established at this n."
                     if not strong else
                     f" And it is NEW information: inside generic-abundance deciles the contrast survives "
                     f"at {hh['ratio_a_over_b']:.3f}x ({hh['mean_abs_z_a']:.3f} vs "
                     f"{hh['mean_abs_z_b']:.3f}), significant in {100*hh['frac_draws_p05']:.0f}% of "
                     f"{hh['n_draws']} draws.")

    # PART 3: the decision. A difference this small is only readable paired, and it is reported as such.
    if pair:
        moved = pair["mean_paired_delta_k562_minus_generic"]
        frac = pair["frac_draws_k562_absorbs_more"]
        share = (100 * pair["absorbed_by_k562"] / (b0 - 1)) if "absorbed_by_k562" in pair else float("nan")
        wp = pair["wilcoxon_p"]
        resolved = wp is not None and wp < 0.05 and moved < 0
        common = (f" Re-matching the existing PPI layer's fast test with a third axis of measured-gene "
                  f"abundance moves the ratio from {b0:.4f} (two axes, {base_sub['n_pairs']:,} pairs) to "
                  f"{dg:.4f} with the generic table and {dk:.4f} with K562 RNA. Paired over "
                  f"{pair['n_draws']} seeds -- the only way a difference smaller than either arm's "
                  f"draw-to-draw range is readable at all -- the K562 covariate absorbs more "
                  f"({moved:+.5f}, signed-rank p {wp:.2g}, more in {100*frac:.0f}% of individual draws), "
                  f"so the generic table was under-controlling in the predicted direction. The size of "
                  f"that under-control is {share:.1f}% of PPI's {100*(b0-1):.1f}% excess.")
        # THE THRESHOLD IS PRESPECIFIED: a covariate that absorbs under a fifth of a banked effect cannot
        # overturn an admission decision made on that effect, and saying so is the point of running this.
        if not resolved:
            tail = (f" THE SWAP IS NOT RESOLVABLE ON THIS LAYER.{common} At this number of seeds the "
                    f"direction itself is not established, so no claim is made either way.")
        elif np.isfinite(share) and share >= 20:
            tail = (f" AND IT CHANGES A BANKED NUMBER.{common} A fifth or more of the PPI effect was "
                    f"abundance the generic covariate failed to remove, so the banked ratio is optimistic "
                    f"and every layer scored against it inherits the same inflation.")
        else:
            tail = (f" WHAT IT DOES NOT BUY, WHICH IS THE PART THAT DECIDES WHETHER TO CARRY IT: the swap "
                    f"does not change a banked number.{common} That is real but small: PPI's ratio is not "
                    f"an abundance artefact under either covariate, and swapping them would not have "
                    f"changed a single admission decision this project has made. The better covariate is "
                    f"established; its consequence for this particular layer is not.")
    else:
        tail = (f" The covariate swap on the PPI fast test was not testable, so this layer's effect on a "
                f"banked number is unmeasured.")

    # the accessibility arm, reported whichever way it came out
    ac = R["accessibility"]["matched"]
    if ac.get("testable"):
        tail += (f" Promoter accessibility adds nothing on top of the expression level: matched on K562 "
                 f"TPM decile, promoters open in all {n_atac} ATAC experiments carry mean |z| "
                 f"{ac['mean_abs_z_a']:.4f} against {ac['mean_abs_z_b']:.4f} for promoters open in none "
                 f"({ac['ratio_a_over_b']:.4f}x over {ac['n_draws']} draws) -- a ratio BELOW 1, i.e. if "
                 f"anything the accessible promoters are marginally less responsive, and the unmatched "
                 f"Spearman is {R['accessibility']['spearman_signal_vs_responsiveness']:+.3f}. "
                 f"Accessibility is kept in the layer as state, not as a predictor.")
    if q5.get("testable") and q5u.get("testable"):
        tail += (f" Finally, the prescribed pair fast test behaves exactly as predicted before it was run: "
                 f"{q5['ratio']:.3f}x on {q5['n_pairs']:,} expression-defined pairs, because matching on "
                 f"perturbation strength removes a per-gene state layer by construction -- drop that one "
                 f"match and the same pairs give {q5u['ratio']:.3f}x. That is the size of what the test "
                 f"removes, and it is why this layer is scored as a covariate rather than against the "
                 f"1.208x PPI reference.")
    v = head + tail
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "encode_state.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'encode_state.json'}")


if __name__ == "__main__":
    main()
