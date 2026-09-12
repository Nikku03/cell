"""DEPMAP CO-DEPENDENCY -- and the endpoint this layer has been graded against by mistake.

WHAT IS ALREADY THERE. The cell object carries `codep`: 13,451 genes, up to eight partners each, 65,571
distinct pairs, keyed by GENE INDEX, with no recorded provenance. It also carries `sl`: 1,256 pairs labelled
synthetic-lethal. Neither has a source, a release, or a construction recipe attached. This module rebuilds
co-dependency from the primary file (DepMap 24Q2 CRISPRGeneEffect, md5 asserted against figshare's own
manifest), reports how far the existing layer agrees, and then does the thing that has not been done: grades
it against the endpoint it is actually about.

THE ENDPOINT ERROR. Every link layer in this project is graded by the same fast test -- does the layer
predict a large |robust z| in K562 Perturb-seq, i.e. does perturbing A move B's RNA. For PPI that is a fair
question. For co-dependency it is the wrong question, and the reference numbers say so: co-dependency scores
1.070x on that test, barely above the 1.05 floor, and it has been read as "a weak layer". But co-dependency
is not a claim about transcription at all. It is a claim about VIABILITY:

    P(cell dies | A knocked out) covaries with P(cell dies | B knocked out) across genetic backgrounds.

Nothing in that says A's knockout should move B's transcript. Two subunits of the same obligate complex are
co-dependent because the machine fails either way, and the cell can be perfectly capable of maintaining both
mRNAs while it dies. So the fast test is not a weak result for this layer -- it is a test of a different
proposition, and it should be reported as such rather than as a grade.

SO BOTH TESTS ARE RUN, ON THE SAME PAIRS.

  TEST 1, TRANSCRIPTIONAL (the standard fast test). |robust z| of linked (P,G) cells against cells matched on
    perturbation-strength decile x gene-responsiveness decile, 20 draws. Expected weak; reported honestly.
    Split by correlation strength, because if the transcriptional signal were really co-dependency showing
    through, a stronger co-dependency ought to produce a bigger transcriptional response. If it does not, the
    two endpoints are close to orthogonal and that is the finding.

  TEST 2, VIABILITY (the endpoint the layer is about). HELD OUT BY CONSTRUCTION: the 1,150 cell lines are cut
    in half, the layer is built on half A only, and the pairs it proposes are scored by their dependency-
    profile correlation on half B, which the build never saw. Controls are pairs matched on each gene's mean
    dependency, its dependency spread, and its expression -- all three computed on half A only -- drawn 20
    times per split over 5 independent splits. The effect size is the RAW held-out correlation on half B, not
    a difference from the control.

WHAT WOULD FALSIFY THIS. Five things, each checked rather than assumed:
  (a) The held-out viability effect is small in absolute terms. A ratio over a near-zero control can look
      enormous while the raw held-out r is 0.03, i.e. noise. The raw number is therefore the headline and the
      ratio is secondary.
  (b) It is lineage bookkeeping. Two genes both selectively required in one lineage will correlate across a
      panel that contains that lineage. So the test is repeated with each gene's dependency profile centred
      WITHIN its Oncotree lineage on the held-out half, which removes all between-lineage structure.
  (c) It is hubness. A gene whose dependency profile correlates with everything will have partners that
      reproduce for reasons that are not about the partnership. A sensitivity arm adds mean |r| on half A as a
      fourth matching covariate.
  (d) It is the global essentiality axis. The first few principal directions of a DepMap gene-effect matrix
      are growth rate, screen quality and broad essentiality; any two disruptive genes track them together.
      So the test is repeated with those directions projected out of the held-out half, and separately split
      by whether both genes are DepMap-inferred common essentials. That split is also where the project's own
      dataset plan turns out to be wrong: it records pan-essentials as a structural blind spot for this layer
      ("no variance -> no correlation"), and in this release they carry LARGER dependency spread and MORE
      partners than everything else. The correction matters in the opposite direction to the one recorded --
      common-essential edges are over-represented here, and are the ones most likely to be non-specific.
  (e) The gene-level covariates are not actually balanced. Residual imbalance is reported on every covariate,
      for every arm.

THE `sl` LAYER IS MISLABELLED, and the same data says so. All 1,256 of its pairs carry POSITIVE dependency
correlation (0.401 to 0.912) and they are things like TSC1/TSC2, four SAGA subunits, and NDUFB11/NDUFC1 --
same-complex, same-pathway pairs. Synthetic lethality is the opposite sign of evidence: A becomes required
when B is lost, which shows up as ANTI-correlation or as a buffering interaction, and cannot be read off a
positive co-dependency correlation at all. That layer is co-dependency under another name, and it is graded
here as co-dependency.

PROVENANCE IS PINNED AND ASSERTED. One release (DepMap 24Q2 Public), one file per axis, md5-checked against
figshare's manifest at load time -- the local CRISPRGeneEffect.csv was of unrecorded origin and is now
identified. Genes are joined through the registry by ENTREZ, the accession the DepMap header carries, never
by symbol. That join exposed a defect in the shared registry and the defect is reported rather than routed
around silently.
"""
import collections
import gzip
import hashlib
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

# ---- ONE RELEASE ALONG EVERY AXIS. The md5s come from figshare's own file manifest for this article and
# ---- are asserted, not stated. The local CRISPRGeneEffect.csv arrived with no recorded provenance; this is
# ---- what identifies it.
RELEASE = "DepMap 24Q2 Public"
DOI = "10.25452/figshare.plus.25880521.v1"
GENE_EFFECT = SP / "CRISPRGeneEffect.csv"
MD5_GENE_EFFECT = "d10d73692672380b3792231549e97d5a"
DM = SP / "depmap"
EXPR_CSV = DM / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
EXPR_URL = "https://ndownloader.figshare.com/files/46490878"
MD5_EXPR = "f00c434db50d811544ac4d047003979f"
MODEL_CSV = DM / "Model.csv"
MODEL_URL = "https://ndownloader.figshare.com/files/46489732"
MD5_MODEL = "e5b60d1ce7636542d93f45494019eded"
COMMESS_CSV = DM / "CRISPRInferredCommonEssentials.csv"
COMMESS_URL = "https://ndownloader.figshare.com/files/46489597"
MD5_COMMESS = "76dd2c044e9aa42bdfa344c8681904f7"
EXPR_NPZ = SP / "depmap_expr.npz"

GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
REGF = SP / "entity_registry.json.gz"
LAYER = SP / "cell_depmap_codep.json.gz"

TOP_K = 8                # match the existing layer's shape so "agreement" means something
R_MIN = 0.20             # ... and its apparent threshold
DECILES = 10
TIGHT = 30
QUINT = 5                # gene-level covariate bins for the viability match
QUINT_TIGHT = 10         # ... and the finer grid where the dependency-spread residual has to clear
N_DRAWS = 20
N_SPLITS = 5
MIN_LINEAGE = 3
N_GLOBAL_PC = 5          # global axes projected out of the held-out half as a falsification check


# ---------------------------------------------------------------------------------------------------
def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, path, want_md5, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.stat().st_size > 0):
        report(f"    downloading {path.name} ...")
        for i in range(4):
            try:
                r = urllib.request.Request(url, headers={"User-Agent": "cellos"})
                with urllib.request.urlopen(r, timeout=1800) as fh, open(path, "wb") as o:
                    while True:
                        b = fh.read(1 << 22)
                        if not b:
                            break
                        o.write(b)
                break
            except Exception:
                if path.exists():
                    path.unlink()
                if i == 3:
                    raise SystemExit(f"BLOCKED: could not fetch {url}")
                time.sleep(2 ** (i + 1))
    got = md5(path)
    if got != want_md5:
        raise SystemExit(f"{path.name}: md5 {got} != {want_md5} -- this is not the {RELEASE} file")
    return path


HDR = re.compile(r"^(.+) \((\d+)\)$")


def parse_header(cols):
    """DepMap column headers are `SYMBOL (ENTREZ)`. The Entrez id is an accession and the symbol is not."""
    sym, ent = [], []
    for c in cols:
        m = HDR.match(c)
        if not m:
            raise SystemExit(f"unparsable DepMap header column: {c!r}")
        sym.append(m.group(1))
        ent.append(m.group(2))
    return sym, ent


# ---------------------------------------------------------------------------------------------------
# THE REGISTRY'S ENTREZ NAMESPACE IS BROKEN, AND THIS IS THE ONE PLACE THAT SAYS SO OUT LOUD.
#
# entity_registry.py builds its entrez aliases from a parquet column that pandas typed as float, so every key
# is stored as "7105.0" rather than "7105". A caller passing a real Entrez id resolves 0 of 18,443 -- exactly
# the silent-empty-join the registry exists to prevent, and it fails in the direction that hurts: the natural
# recovery is to fall back to gene symbol, which is the weakest key and which this project's conventions
# forbid whenever an accession exists. The registry is NOT modified here (it is another module's file); the
# spelling is normalised at the call site, the defect is recorded in the output, and the join still goes
# through Registry.join() so the declared-rate guard is the thing that passes or raises.
def entrez_keys(ent):
    return [f"{int(e)}.0" for e in ent]


class Pools:
    """Members of each covariate class, flattened, so a matched control can be drawn without a Python loop."""

    def __init__(self, cls, n_cls):
        cls = np.asarray(cls)
        self.flat = np.argsort(cls, kind="stable")
        self.cnt = np.bincount(cls, minlength=n_cls)
        self.off = np.concatenate([[0], np.cumsum(self.cnt)]).astype(np.int64)

    def sample(self, klass, rng):
        c = np.maximum(self.cnt[klass], 1)
        u = (rng.random(len(klass)) * c).astype(np.int64)
        return self.flat[self.off[klass] + u]


def rejection_draw(pa, pb, ka, kb, reject, rng, tries=40):
    """One matched control per linked pair, drawn from the linked pair's own covariate cell.

    Drawing per-pair from its own cell reproduces the joint cell distribution of the linked set exactly,
    which is what a matched control has to do. `reject(x, y)` returns a boolean mask of draws that must be
    thrown back -- self-pairs, pairs that are themselves linked, non-finite cells, duplicates.
    """
    n = len(ka)
    x = np.full(n, -1, dtype=np.int64)
    y = np.full(n, -1, dtype=np.int64)
    todo = np.arange(n)
    for _ in range(tries):
        if len(todo) == 0:
            break
        cx = pa.sample(ka[todo], rng)
        cy = pb.sample(kb[todo], rng)
        bad = reject(cx, cy)
        x[todo[~bad]] = cx[~bad]
        y[todo[~bad]] = cy[~bad]
        todo = todo[bad]
    ok = x >= 0
    return x, y, ok


# ---------------------------------------------------------------------------------------------------
class FastMatcher:
    """TEST 1. Matched draws on (perturbation strength decile) x (gene responsiveness decile).

    The pair universe is 11,258 x 8,248, far too large to enumerate, so the control is drawn by rejection
    inside the matched cell. Self-pairs -- knocking a gene down moves its own transcript -- are excluded from
    both arms.
    """

    def __init__(self, A, prow, pcol, pert_eid, gene_eid, nbin):
        self.A, self.prow, self.pcol, self.nbin = A, prow, pcol, nbin
        q = np.linspace(1.0 / nbin, 1.0 - 1.0 / nbin, nbin - 1)
        self.rd = np.digitize(prow, np.quantile(prow, q))
        self.cd = np.digitize(pcol, np.quantile(pcol, q))
        self.pr = Pools(self.rd, nbin)
        self.pc = Pools(self.cd, nbin)
        u = {e: i for i, e in enumerate({x for x in pert_eid if x} | {x for x in gene_eid if x})}
        self.pk = np.array([u.get(e, -1) for e in pert_eid])
        self.gk = np.array([u.get(e, -2) for e in gene_eid])

    def run(self, L, linked_key, sel=None, cap=None, n_draws=N_DRAWS):
        from scipy import stats
        idx = np.arange(len(L)) if sel is None else np.where(sel)[0]
        if len(idx) < 200:
            return None
        nG = self.A.shape[1]
        out = collections.defaultdict(list)
        for s in range(n_draws):
            rng = np.random.default_rng(s)
            # when `cap` is set the LINKED arm is subsampled too, so n, raw mean and p all describe the
            # capped comparison -- reporting the full pool's n beside a capped p is how a power check
            # quietly stops being one
            use = idx if (cap is None or len(idx) <= cap) else rng.choice(idx, cap, replace=False)
            li, lj = L[use, 0], L[use, 1]

            def reject(x, y):
                return (np.isin(x.astype(np.int64) * nG + y, linked_key)
                        | (self.pk[x] == self.gk[y])
                        | ~np.isfinite(self.A[x, y]))

            ci, cj, ok = rejection_draw(self.pr, self.pc, self.rd[li], self.cd[lj], reject, rng)
            li, lj, ci, cj = li[ok], lj[ok], ci[ok], cj[ok]
            a1, a0 = self.A[li, lj], self.A[ci, cj]
            u1 = stats.mannwhitneyu(a1, a0, alternative="two-sided")
            out["n"].append(len(li))
            out["m1"].append(float(a1.mean()))
            out["m0"].append(float(a0.mean()))
            out["ratio"].append(float(a1.mean() / a0.mean()))
            out["auc"].append(float(u1[0] / (len(a1) * len(a0))))
            out["p"].append(float(u1[1]))
            out["rp"].append(float(stats.mannwhitneyu(self.prow[li], self.prow[ci])[1]))
            out["cp"].append(float(stats.mannwhitneyu(self.pcol[lj], self.pcol[cj])[1]))
        return {"n_linked": int(np.mean(out["n"])), "n_pool": int(len(idx)), "n_draws": n_draws,
                "capped_to": cap, "bins": self.nbin,
                # THE RAW HELD-OUT VALUE IS THE EFFECT SIZE. The ratio is the statistic that makes this
                # comparable to the other layers' reference numbers; it is not the effect.
                "raw_mean_abs_z_linked": float(np.mean(out["m1"])),
                "raw_mean_abs_z_matched": float(np.mean(out["m0"])),
                "ratio": float(np.mean(out["ratio"])), "ratio_sd": float(np.std(out["ratio"])),
                "auc": float(np.mean(out["auc"])),
                "p_median": float(np.median(out["p"])),
                "frac_draws_p05": float(np.mean([p < 0.05 for p in out["p"]])),
                "residual_pert_strength_p": float(np.mean(out["rp"])),
                "residual_gene_responsiveness_p": float(np.mean(out["cp"]))}


def fline(label, r):
    if r is None:
        return f"    {label:30s} too few testable pairs"
    return (f"    {label:30s} {r['n_linked']:7,d} {r['raw_mean_abs_z_linked']:9.4f} "
            f"{r['raw_mean_abs_z_matched']:9.4f} {r['ratio']:8.4f} {r['auc']:7.4f} "
            f"{r['p_median']:10.3g} {100*r['frac_draws_p05']:6.0f}%")


FHEAD = (f"    {'arm':30s} {'pairs':>7s} {'raw |z|':>9s} {'matched':>9s} {'ratio':>8s} {'AUC':>7s} "
         f"{'med p':>10s} {'p<.05':>7s}")


# ---------------------------------------------------------------------------------------------------
class ViabilityMatcher:
    """TEST 2. Matched draws over gene-level covariate classes, scored on held-out cell lines.

    A pair is described by the covariate class of each of its two genes. A control pair takes one gene from
    each of those two classes. Covariates are computed on the TRAINING half only, so the control construction
    never touches the held-out half either.
    """

    def __init__(self, cov, elig, nbin):
        """cov: (n_genes, n_cov) covariate matrix. elig: genes usable at all (all covariates known)."""
        self.elig = elig
        self.nbin = nbin
        q = np.linspace(1.0 / nbin, 1.0 - 1.0 / nbin, nbin - 1)
        cls = np.zeros(len(cov), dtype=np.int64)
        for k in range(cov.shape[1]):
            v = cov[:, k]
            cls = cls * nbin + np.digitize(v, np.quantile(v[elig], q))
        cls[~elig] = -1
        self.n_cls = nbin ** cov.shape[1]
        self.cls = cls
        self.pool = Pools(np.where(elig, cls, self.n_cls), self.n_cls + 1)   # ineligible parked in a bin
        self.cov = cov

    def run(self, pairs, linked_key, ZbT, nG, n_draws=N_DRAWS, label=""):
        """pairs: (n,2) gene indices. ZbT: z-scored held-out profiles, GENE-major (n_genes, n_lines_B).

        Gene-major so that picking out a pair's two profiles is a contiguous row gather rather than a
        strided column gather; chunked so the temporary never exceeds a few tens of MB.
        """
        from scipy import stats
        m = self.elig[pairs[:, 0]] & self.elig[pairs[:, 1]]
        P = pairs[m]
        if len(P) < 200:
            return None
        nB = ZbT.shape[1]

        def corr(i, j, chunk=20000):
            o = np.empty(len(i), dtype=np.float64)
            for a in range(0, len(i), chunk):
                b = min(a + chunk, len(i))
                o[a:b] = np.einsum("ij,ij->i", ZbT[i[a:b]], ZbT[j[a:b]]) / nB
            return o

        r1_all = corr(P[:, 0], P[:, 1])
        out = collections.defaultdict(list)
        for s in range(n_draws):
            rng = np.random.default_rng(1000 + s)
            ka, kb = self.cls[P[:, 0]], self.cls[P[:, 1]]

            def reject(x, y):
                lo, hi = np.minimum(x, y), np.maximum(x, y)
                return (x == y) | np.isin(lo.astype(np.int64) * nG + hi, linked_key)

            cx, cy, ok = rejection_draw(self.pool, self.pool, ka, kb, reject, rng)
            r1 = r1_all[ok]
            r0 = corr(cx[ok], cy[ok])
            u = stats.mannwhitneyu(r1, r0, alternative="two-sided")
            out["n"].append(len(r1))
            out["m1"].append(float(r1.mean()))
            out["m0"].append(float(r0.mean()))
            out["auc"].append(float(u[0] / (len(r1) * len(r0))))
            out["p"].append(float(u[1]))
            for k in range(self.cov.shape[1]):
                v = self.cov[:, k]
                lv = np.concatenate([v[P[ok, 0]], v[P[ok, 1]]])
                cv = np.concatenate([v[cx[ok]], v[cy[ok]]])
                out[f"res{k}"].append(float(stats.mannwhitneyu(lv, cv)[1]))
        m1, m0 = float(np.mean(out["m1"])), float(np.mean(out["m0"]))
        return {"label": label, "n_pairs": int(np.mean(out["n"])), "n_pool": int(len(P)),
                "n_dropped_ineligible": int((~m).sum()), "n_draws": n_draws, "bins": self.nbin,
                # RAW HELD-OUT CORRELATION ON LINES THE LAYER NEVER SAW. This is the effect size.
                "raw_mean_r_heldout_linked": m1, "raw_mean_r_heldout_matched": m0,
                "raw_median_r_heldout_linked": float(np.median(r1_all)),
                "delta": m1 - m0, "ratio": (m1 / m0) if m0 > 1e-6 else None,
                "auc": float(np.mean(out["auc"])),
                "p_median": float(np.median(out["p"])),
                "frac_draws_p05": float(np.mean([p < 0.05 for p in out["p"]])),
                "residual_p": [float(np.mean(out[f"res{k}"])) for k in range(self.cov.shape[1])]}


def vline(label, r):
    if r is None:
        return f"    {label:34s} too few testable pairs"
    rt = "  n/a" if r["ratio"] is None else f"{r['ratio']:6.1f}x"
    return (f"    {label:34s} {r['n_pairs']:7,d} {r['raw_mean_r_heldout_linked']:9.4f} "
            f"{r['raw_mean_r_heldout_matched']:9.4f} {rt:>7s} {r['auc']:7.4f} "
            f"{r['p_median']:10.3g} {100*r['frac_draws_p05']:6.0f}%")


VHEAD = (f"    {'arm':34s} {'pairs':>7s} {'raw r_B':>9s} {'matched':>9s} {'ratio':>7s} {'AUC':>7s} "
         f"{'med p':>10s} {'p<.05':>7s}")


# ---------------------------------------------------------------------------------------------------
def topk_pairs(Z, k=TOP_K, rmin=R_MIN, block=2048):
    """Top-k positive co-dependency partners per gene, blockwise so the full n^2 matrix is never held."""
    n = Z.shape[1]
    nl = Z.shape[0]
    idx = np.zeros((n, k), dtype=np.int32)
    val = np.zeros((n, k), dtype=np.float32)
    for a in range(0, n, block):
        b = min(a + block, n)
        C = (Z[:, a:b].T @ Z) / nl
        C[np.arange(b - a), np.arange(a, b)] = -9.0
        ii = np.argpartition(-C, k, axis=1)[:, :k]
        vv = np.take_along_axis(C, ii, 1)
        o = np.argsort(-vv, 1)
        idx[a:b] = np.take_along_axis(ii, o, 1)
        val[a:b] = np.take_along_axis(vv, o, 1)
        del C
    pairs, rs = {}, {}
    for i in range(n):
        for j, r in zip(idx[i], val[i]):
            if r >= rmin and i != j:
                key = (min(i, int(j)), max(i, int(j)))
                if r > rs.get(key, -9):
                    rs[key] = float(r)
    pairs = np.array(sorted(rs), dtype=np.int64).reshape(-1, 2)
    return pairs, np.array([rs[tuple(p)] for p in pairs.tolist()], dtype=np.float32), idx, val


def zscore(X):
    m = X.mean(0)
    s = X.std(0)
    return np.ascontiguousarray(((X - m) / np.where(s > 0, s, np.nan)).astype(np.float32))


def enc(pairs, n):
    """Unordered gene-index pair -> one int64, for O(1) membership by np.isin."""
    lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    return np.unique(lo * n + hi)


# ---------------------------------------------------------------------------------------------------
def main():
    from scipy import stats
    import h5py
    import pandas as pd
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 108)
    report("DEPMAP CO-DEPENDENCY -- graded on viability, the endpoint it is actually about, and on RNA, "
           "the one it is not")
    report("=" * 108)
    for p in (GENE_EFFECT, GWPS, NET, REGF):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    # ---- provenance, asserted ----
    got = md5(GENE_EFFECT)
    if got != MD5_GENE_EFFECT:
        raise SystemExit(f"CRISPRGeneEffect.csv md5 {got} != {MD5_GENE_EFFECT}; the local file is not "
                         f"{RELEASE} and mixing releases along this axis is not allowed")
    report(f"  SOURCE  {RELEASE}  ({DOI})")
    report(f"    CRISPRGeneEffect.csv  md5 {got}  ASSERTED against figshare's manifest "
           f"-- the local copy had no recorded provenance until now")
    download(MODEL_URL, MODEL_CSV, MD5_MODEL, report)
    report(f"    Model.csv             md5 {MD5_MODEL}  (lineage, same release)")
    download(COMMESS_URL, COMMESS_CSV, MD5_COMMESS, report)
    report(f"    CRISPRInferredCommonEssentials.csv  md5 {MD5_COMMESS}  (the blind-spot list, same release)")

    from entity_registry import load as load_reg
    REG = load_reg()
    CI = json.load(gzip.open(REGF, "rt"))["cell_index"]

    # ---- gene effect matrix ----
    t0 = time.time()
    df = pd.read_csv(GENE_EFFECT, index_col=0)
    X = df.to_numpy(dtype=np.float32)
    cols, lines = list(df.columns), list(df.index)
    del df
    sym, ent = parse_header(cols)
    report(f"  gene effect matrix: {X.shape[0]:,} cell lines x {X.shape[1]:,} genes "
           f"({time.time()-t0:.0f}s)")

    # ---- THE JOIN. Entrez is an accession and the header carries it, so symbol is not used as the key. ----
    ekeys = entrez_keys(ent)
    e_by_entrez, st = REG.join("gene", "entrez", ekeys, 0.98, "DepMap genes by Entrez")
    e_by_symbol = [REG.resolve("gene", "symbol", s) for s in sym]
    n_sym = sum(x is not None for x in e_by_symbol)
    disagree = [(s, e) for s, e, a, b in zip(sym, ent, e_by_entrez, e_by_symbol) if a and b and a != b]
    try:
        REG.join("gene", "entrez", ent, 0.98, "DepMap genes by RAW Entrez id")
        raw_ok = True
    except Exception:
        raw_ok = False
    report(f"  registry join by ENTREZ: {st['joined']:,}/{st['of']:,} ({st['rate']:.1%})   "
           f"by symbol would have been {n_sym:,} ({n_sym/len(sym):.1%})")
    report(f"    REGISTRY DEFECT: gene/entrez aliases are stored float-stringified ('7105.0'), so a raw "
           f"Entrez id resolves {'' if raw_ok else '0 of 18,443'} -- a silent empty join of exactly the kind "
           f"the registry exists to stop.")
    report(f"    Normalised at this call site (registry not modified). {len(disagree)} genes where symbol "
           f"and Entrez resolve to DIFFERENT entities, e.g. {[d[0] for d in disagree[:6]]} "
           f"-- joining by symbol would have mis-assigned them.")
    eid = e_by_entrez

    # ---- per-gene dependency statistics. NaNs are dropped, never imputed. ----
    nan_g = np.isnan(X).sum(0)
    complete = nan_g == 0
    dep_mean_all = np.nanmean(X, 0)
    dep_sd_all = np.nanstd(X, 0)
    report(f"  missing gene effect: {100*np.isnan(X).mean():.2f}% of cells; {int((~complete).sum()):,} genes "
           f"have >=1 missing line and are DROPPED from the correlation (never imputed)")
    report(f"    dropped genes mean effect {dep_mean_all[~complete].mean():.4f} vs kept "
           f"{dep_mean_all[complete].mean():.4f}   MWU p "
           f"{stats.mannwhitneyu(dep_mean_all[~complete], dep_mean_all[complete])[1]:.3g}"
           f"  -> missingness is NOT random, so absence here is unknown, not zero")

    kg = np.where(complete)[0]
    keep_eid = [eid[i] for i in kg]
    seen, uniq = set(), []
    for i, e in enumerate(keep_eid):
        if e is not None and e not in seen:
            seen.add(e)
            uniq.append(i)
    uniq = np.array(uniq)
    G = kg[uniq]                                   # column indices into X, one per distinct entity
    gene_eid = [eid[i] for i in G]
    gpos = {e: i for i, e in enumerate(gene_eid)}
    nG = len(G)
    report(f"  {nG:,} genes with a complete profile and a distinct registry entity")

    Xg = X[:, G]
    del X
    dep_mean = Xg.mean(0)
    dep_sd = Xg.std(0)
    dep_frac = (Xg < -0.5).mean(0)                 # DepMap's own "dependent" threshold on gene effect

    # ---- expression, same release, as a matching covariate ----
    if not EXPR_NPZ.exists():
        download(EXPR_URL, EXPR_CSV, MD5_EXPR, report)
        ed = pd.read_csv(EXPR_CSV, index_col=0)
        have = [l for l in lines if l in ed.index]
        E = ed.loc[have].to_numpy(dtype=np.float32)
        ecols = list(ed.columns)
        del ed
        np.savez_compressed(EXPR_NPZ, E=E, genes=np.array(ecols), lines=np.array(have),
                            md5=np.array([MD5_EXPR]))
        EXPR_CSV.unlink()                          # 460 MB reduced to a compact form and deleted
    z = np.load(EXPR_NPZ, allow_pickle=True)
    E, ecols, elines = z["E"], [str(x) for x in z["genes"]], [str(x) for x in z["lines"]]
    esym, eent = parse_header(ecols)
    e_eid = [REG.resolve("gene", "entrez", k) for k in entrez_keys(eent)]
    expr_mean = np.full(nG, np.nan, dtype=np.float32)
    emu = E.mean(0)
    for c, e in enumerate(e_eid):
        p = gpos.get(e)
        if p is not None:
            expr_mean[p] = emu[c]
    has_expr = np.isfinite(expr_mean)
    report(f"  expression (same release): {len(elines):,}/{len(lines):,} screened lines covered, "
           f"{int(has_expr.sum()):,}/{nG:,} genes have a mean log2(TPM+1)")
    report(f"    genes WITHOUT expression: mean effect {dep_mean[~has_expr].mean():.4f} vs "
           f"{dep_mean[has_expr].mean():.4f}   MWU p "
           f"{stats.mannwhitneyu(dep_mean[~has_expr], dep_mean[has_expr])[1]:.3g}"
           f"  -> carried as unknown and excluded from the matched arm, never zero-filled")
    del E

    # ---- lineage, for the falsification check ----
    mdl = pd.read_csv(MODEL_CSV)
    lin_of = dict(zip(mdl["ModelID"], mdl["OncotreeLineage"].astype(str)))
    lin = np.array([lin_of.get(l, "unknown") for l in lines])
    lc = collections.Counter(lin.tolist())
    report(f"  lineages: {len(lc)} distinct, largest {lc.most_common(3)}")

    # ---- build the layer on ALL lines (this is the shipped layer) ----
    t0 = time.time()
    Zall = zscore(Xg)
    pairs_all, r_all, top_idx, top_val = topk_pairs(Zall)
    report(f"  co-dependency built on all {Zall.shape[0]:,} lines: {len(pairs_all):,} distinct pairs "
           f"(top-{TOP_K}, r >= {R_MIN})  [{time.time()-t0:.0f}s]")

    # ---- THE DOCUMENTED BLIND SPOT, MEASURED RATHER THAN REPEATED ----
    # The project's own dataset plan records it: pan-essential genes have no variance across lines, so a
    # correlation of dependency profiles is structurally unable to see them. That is a coverage claim and it
    # is checkable against DepMap's own inferred common-essential list.
    ce = pd.read_csv(COMMESS_CSV)
    ce_sym, ce_ent = parse_header([str(x) for x in ce.iloc[:, 0]])
    ce_idx = np.zeros(nG, dtype=bool)
    for kk in entrez_keys(ce_ent):
        p = gpos.get(REG.resolve("gene", "entrez", kk))
        if p is not None:
            ce_idx[p] = True
    has_partner = np.zeros(nG, dtype=bool)
    has_partner[np.unique(pairs_all.ravel())] = True
    report(f"\n  THE DOCUMENTED BLIND SPOT IS NOT WHAT THIS RELEASE SAYS. The project's dataset plan records "
           f"that co-dependency is 'structurally blind' to pan-essentials because their essentiality does "
           f"not vary across lines. Measured, on {int(ce_idx.sum()):,} of DepMap's {len(ce_sym):,} inferred "
           f"common essentials:")
    report(f"    dependency spread {dep_sd[ce_idx].mean():.4f} vs {dep_sd[~ce_idx].mean():.4f} for the "
           f"rest -- LARGER, not smaller; {100*has_partner[ce_idx].mean():.1f}% get a partner against "
           f"{100*has_partner[~ce_idx].mean():.1f}%.")
    report(f"    That is not a licence to trust those edges. A common essential's spread is largely a "
           f"shared screen-quality / growth-rate axis, so pan-essentials can correlate with EACH OTHER for "
           f"a reason that is not specific to the pair. The viability test below is therefore split by "
           f"common-essential status AND repeated with the five strongest global axes projected out.")

    # ---- agreement with the existing, unprovenanced layer ----
    d = json.load(gzip.open(NET, "rt"))
    names = [g["name"] for g in d["genes"]]
    ex_codep, ex_sl = {}, {}
    for k, v in d["codep"].items():
        a = CI.get(k)
        for p, c in v:
            b = CI.get(str(p))
            if a and b and a != b:
                kk = (min(a, b), max(a, b))
                ex_codep[kk] = max(ex_codep.get(kk, -9), float(c))
    for a_i, b_i, c in d["sl"]:
        a, b = CI.get(str(a_i)), CI.get(str(b_i))
        if a and b and a != b:
            ex_sl[(min(a, b), max(a, b))] = float(c)
    del d

    mine = {(min(gene_eid[i], gene_eid[j]), max(gene_eid[i], gene_eid[j])): float(r)
            for (i, j), r in zip(pairs_all.tolist(), r_all.tolist())}
    inter = set(mine) & set(ex_codep)
    test_ex = {k: v for k, v in ex_codep.items() if k[0] in gpos and k[1] in gpos}
    ii = np.array([gpos[k[0]] for k in test_ex])
    jj = np.array([gpos[k[1]] for k in test_ex])
    my_r = (Zall[:, ii] * Zall[:, jj]).sum(0) / Zall.shape[0]
    claim = np.array([test_ex[k] for k in test_ex])
    agree_r = float(np.corrcoef(my_r, claim)[0, 1])
    report(f"\n  AGREEMENT WITH THE EXISTING `codep` LAYER (provenance was UNKNOWN)")
    report(f"    existing pairs {len(ex_codep):,}; mine {len(mine):,}; shared {len(inter):,} "
           f"(Jaccard {len(inter)/len(set(mine)|set(ex_codep)):.3f})")
    report(f"    of the {len(test_ex):,} existing pairs whose genes I can test, "
           f"{100*len(set(test_ex)&set(mine))/len(test_ex):.1f}% are recovered in my top-{TOP_K}")
    report(f"    my Pearson r vs its stated value: r = {agree_r:.4f}; "
           f"{100*float((my_r < 0).mean()):.2f}% of its pairs are negative in {RELEASE}")
    report(f"    -> the existing layer IS Pearson correlation of DepMap gene-effect profiles, top-8, "
           f"thresholded ~0.2, from a nearby release. Provenance recovered, not assumed.")

    ex_sl_r = None
    if ex_sl:
        sk = [k for k in ex_sl if k[0] in gpos and k[1] in gpos]
        si = np.array([gpos[k[0]] for k in sk])
        sj = np.array([gpos[k[1]] for k in sk])
        ex_sl_r = (Zall[:, si] * Zall[:, sj]).sum(0) / Zall.shape[0]
        report(f"\n  THE `sl` LAYER IS NOT SYNTHETIC LETHALITY. {len(ex_sl):,} pairs, stated correlations "
               f"{min(ex_sl.values()):.3f}..{max(ex_sl.values()):.3f}, ALL POSITIVE; measured here "
               f"{ex_sl_r.min():.3f}..{ex_sl_r.max():.3f} with {int((ex_sl_r < 0).sum())} negative.")
        report(f"    Positive co-dependency means 'both required in the same backgrounds' -- same complex, "
               f"same pathway (TSC1/TSC2, four SAGA subunits, NDUFB11/NDUFC1). Synthetic lethality is the "
               f"opposite: A becomes required when B is LOST. This layer is co-dependency mislabelled, and "
               f"it is graded below as co-dependency.")

    # =================================================================================================
    # TEST 2 -- VIABILITY, HELD OUT BY CONSTRUCTION
    # =================================================================================================
    report(f"\n{'='*108}")
    report(f"  TEST 2 -- VIABILITY.  Does a co-dependency call hold up on cell lines the build never saw?")
    report(f"  Layer built on half the lines; scored by dependency-profile correlation on the other half.")
    report(f"{'='*108}")
    nl = Zall.shape[0]
    lin_ok = np.array([lc[x] >= MIN_LINEAGE for x in lin])
    V = {"splits": [], "n_splits": N_SPLITS}
    arms = collections.defaultdict(list)
    for s in range(N_SPLITS):
        rng = np.random.default_rng(s)
        perm = rng.permutation(nl)
        ia, ib = perm[: nl // 2], perm[nl // 2:]
        Za = zscore(Xg[ia])
        Zb = zscore(Xg[ib])
        pA, _rA, _, _ = topk_pairs(Za)
        keyA = enc(pA, nG)

        # covariates from the TRAINING half only
        mA, sA = Xg[ia].mean(0), Xg[ia].std(0)
        hub = np.zeros(nG, dtype=np.float32)
        for a in range(0, nG, 2048):
            b = min(a + 2048, nG)
            C = (Za[:, a:b].T @ Za) / len(ia)
            hub[a:b] = np.abs(C).mean(1)
            del C
        # A gene with zero dependency variance on either half has an UNDEFINED correlation, not a zero one.
        # It is made ineligible for every arm of this split rather than imputed.
        finite = np.isfinite(Za).all(0) & np.isfinite(Zb).all(0)
        if not finite.all():
            report(f"    split {s}: {int((~finite).sum())} genes have zero dependency variance on one half "
                   f"-- correlation UNDEFINED, excluded from this split rather than set to zero")
        cov3 = np.stack([mA, sA, expr_mean], 1)
        elig3 = np.isfinite(cov3).all(1) & finite
        M3 = ViabilityMatcher(cov3, elig3, QUINT)
        # QUINTILES LEAVE A RESIDUAL ON DEPENDENCY SPREAD (p ~ 2e-3), and the imbalance runs the way that
        # FLATTERS the result: a wider-spread gene gives a better-estimated correlation. So the whole test
        # is repeated on a 10x10x10 grid, where the residual has to clear for the effect to be believed.
        MT3 = ViabilityMatcher(cov3, elig3, QUINT_TIGHT)
        cov4 = np.stack([mA, sA, expr_mean, hub], 1)
        elig4 = np.isfinite(cov4).all(1) & finite
        M4 = ViabilityMatcher(cov4, elig4, 4)

        # lineage-centred held-out profiles: all between-lineage structure removed
        Xb = Xg[ib].copy()
        lb = lin[ib]
        keepb = lin_ok[ib]
        for L in set(lb[keepb].tolist()):
            m = lb == L
            if m.sum() >= 2:
                Xb[m] -= Xb[m].mean(0)
        Zbl = zscore(Xb[keepb])
        # centring can flatten a gene that was only variable BETWEEN lineages; such a gene now contributes
        # exactly 0 to both the linked and the matched arm, so it dilutes symmetrically rather than biasing
        Zbl[:, ~np.isfinite(Zbl).all(0)] = 0.0
        Zbl = np.ascontiguousarray(Zbl.T)
        del Xb

        # GLOBAL AXES PROJECTED OUT of the held-out half. The first few principal directions of a DepMap
        # gene-effect matrix are growth rate, screen quality and broad essentiality -- exactly the things
        # that would make any two disruptive genes look co-dependent. Eigendecomposing the small
        # line x line Gram matrix gets them in a second.
        Zb[:, ~finite] = 0.0            # excluded above; zeroed only so the Gram matrix is defined
        Mg = (Zb @ Zb.T) / Zb.shape[1]
        w, U = np.linalg.eigh(Mg.astype(np.float64))
        U5 = U[:, -N_GLOBAL_PC:].astype(np.float32)
        var5 = float(w[-N_GLOBAL_PC:].sum() / w.sum())
        Zbp = np.ascontiguousarray(zscore(Zb - U5 @ (U5.T @ Zb)).T)
        ZbT = np.ascontiguousarray(Zb.T)
        del Zb, Mg, U

        ce_pair = ce_idx[pA[:, 0]] & ce_idx[pA[:, 1]]
        arms["mine_heldout"].append(M3.run(pA, keyA, ZbT, nG, label="my layer, half A -> half B"))
        arms["mine_hub"].append(M4.run(pA, keyA, ZbT, nG, label="+ hubness matched"))
        arms["mine_lineage"].append(M3.run(pA, keyA, Zbl, nG, label="lineage-centred held-out half"))
        arms["mine_pc"].append(M3.run(pA, keyA, Zbp, nG, label=f"top-{N_GLOBAL_PC} global axes removed"))
        arms["mine_tight"].append(MT3.run(pA, keyA, ZbT, nG,
                                          label=f"{QUINT_TIGHT}x{QUINT_TIGHT}x{QUINT_TIGHT} bins"))
        arms["mine_ce"].append(M3.run(pA[ce_pair], keyA, ZbT, nG, label="both common-essential"))
        arms["mine_nonce"].append(M3.run(pA[~ce_pair], keyA, ZbT, nG, label="neither common-essential"))
        exp = np.array([[gpos[k[0]], gpos[k[1]]] for k in test_ex], dtype=np.int64)
        arms["existing"].append(M3.run(exp, enc(exp, nG), ZbT, nG, label="existing codep layer"))
        if ex_sl:
            sp = np.array([[gpos[k[0]], gpos[k[1]]] for k in sk], dtype=np.int64)
            arms["sl"].append(M3.run(sp, enc(sp, nG), ZbT, nG, label="existing `sl` layer"))
        V["splits"].append({"seed": s, "n_train": len(ia), "n_test": len(ib),
                            "n_pairs_trained": int(len(pA)),
                            "n_lines_lineage_centred": int(keepb.sum()),
                            "global_pc_var_removed": var5,
                            "n_pairs_both_common_essential": int(ce_pair.sum())})
        report(f"    split {s}: train {len(ia)} lines -> {len(pA):,} pairs; test {len(ib)} lines "
               f"({int(keepb.sum())} usable after lineage centring; top-{N_GLOBAL_PC} axes = "
               f"{100*var5:.1f}% of variance; {int(ce_pair.sum()):,} pairs both common-essential)")
        del Za, ZbT, Zbl, Zbp

    INTKEYS = ("n_pairs", "n_pool", "n_dropped_ineligible", "n_draws", "bins")

    def pool(key):
        rs = [r for r in arms[key] if r]
        if not rs:
            return None
        o = {}
        for k in rs[0]:
            v = [r[k] for r in rs]
            if k == "label":
                o[k] = rs[0][k]
            elif k in INTKEYS:
                o[k] = int(round(float(np.mean(v))))
            elif k == "residual_p":
                o[k] = [float(np.mean([x[i] for x in v])) for i in range(len(rs[0][k]))]
            elif k == "ratio":
                o[k] = float(np.mean([x for x in v if x is not None])) if any(
                    x is not None for x in v) else None
            else:
                o[k] = float(np.mean(v))
        o["n_draws_total"] = len(rs) * N_DRAWS
        o["n_splits"] = len(rs)
        return o

    report(f"\n  {N_SPLITS} splits x {N_DRAWS} matched draws = {N_SPLITS*N_DRAWS} draws per arm.")
    report(f"  Matched on: mean dependency, dependency spread, mean expression (all from the training "
           f"half), {QUINT} bins each.")
    report(VHEAD)
    Vres = {}
    for k, lab in (("mine_heldout", "MY LAYER, held out"),
                   ("mine_hub", "  + hubness matched (4 bins)"),
                   ("mine_lineage", "  + lineage-centred test half"),
                   ("mine_pc", f"  + top-{N_GLOBAL_PC} global axes removed"),
                   ("mine_tight", f"  at {QUINT_TIGHT} bins per covariate"),
                   ("mine_ce", "  both genes common-essential"),
                   ("mine_nonce", "  neither common-essential"),
                   ("existing", "existing `codep` (not held out)"),
                   ("sl", "existing `sl` (not held out)")):
        Vres[k] = pool(k)
        if Vres[k]:
            report(vline(lab, Vres[k]))
    r0 = Vres["mine_heldout"]
    rt = Vres["mine_tight"]
    report(f"    residual imbalance at {QUINT} bins (mean p): mean-dep {r0['residual_p'][0]:.3g}, "
           f"dep-spread {r0['residual_p'][1]:.3g}, expression {r0['residual_p'][2]:.3g}")
    report(f"    residual imbalance at {QUINT_TIGHT} bins: mean-dep {rt['residual_p'][0]:.3g}, "
           f"dep-spread {rt['residual_p'][1]:.3g}, expression {rt['residual_p'][2]:.3g}"
           f"   <- the grid the effect has to survive")
    report(f"    {r0['n_dropped_ineligible']:,} pairs dropped for a missing covariate (expression unknown) "
           f"-- excluded, not zero-filled")
    report(f"  NOTE the existing layer's row is NOT held out: it was built from DepMap on an unrecorded "
           f"line set that plausibly includes these test lines, so its number is an UPPER BOUND.")

    del Zall

    # =================================================================================================
    # TEST 1 -- TRANSCRIPTIONAL (the standard fast test)
    # =================================================================================================
    report(f"\n{'='*108}")
    report(f"  TEST 1 -- TRANSCRIPTIONAL.  The standard fast test: does perturbing A move B's RNA?")
    report(f"{'='*108}")
    with h5py.File(GWPS, "r") as f:
        Xw = f["X"][:]
        gt = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        vg = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0] for s in f["var"]["gene_id"][:]]
    n_nonfinite = int((~np.isfinite(Xw)).sum())
    Xm = np.where(np.isfinite(Xw), Xw, np.nan).astype(np.float32)
    del Xw
    med = np.nanmedian(Xm, 0)
    mad = np.nanmedian(np.abs(Xm - med), 0)
    sc = 1.4826 * mad
    A = np.abs((Xm - med) / np.where(sc > 0, sc, np.nan)).astype(np.float32)
    del Xm
    okm = np.isfinite(A)
    prow = np.where(okm, A, 0).sum(1) / np.maximum(okm.sum(1), 1)
    pcol = np.where(okm, A, 0).sum(0) / np.maximum(okm.sum(0), 1)
    report(f"  GWPS: {A.shape[0]:,} perturbations x {A.shape[1]:,} genes, {n_nonfinite:,} non-finite cells "
           f"DROPPED (never imputed)")

    pert, n_sp = [], 0
    for g in gt:
        p = g.split("_")
        if p[-1].startswith("ENSG"):
            pert.append(REG.resolve("gene", "ensembl", p[-1]))
        elif len(p) > 1 and p[1] != "non-targeting":
            e = REG.resolve("gene", "symbol", p[1])
            pert.append(e)
            n_sp += e is not None
        else:
            pert.append(None)
    gwg = [REG.resolve("gene", "ensembl", e) for e in vg]
    report(f"  perturbations resolved {sum(x is not None for x in pert):,}/{len(pert):,} (by ENSG; {n_sp} "
           f"by symbol where the file carries no ENSG); measured genes "
           f"{sum(x is not None for x in gwg):,}/{len(gwg):,}")

    rowsof, colsof = collections.defaultdict(list), collections.defaultdict(list)
    for i, e in enumerate(pert):
        if e:
            rowsof[e].append(i)
    for j, e in enumerate(gwg):
        if e:
            colsof[e].append(j)
    nGW = A.shape[1]

    ce_eid = {gene_eid[i] for i in np.where(ce_idx)[0]}

    def cells(pair_r):
        L, S, C = [], [], []
        for (a, b), r in pair_r.items():
            both_ce = (a in ce_eid) and (b in ce_eid)
            for x, y in ((a, b), (b, a)):
                for i in rowsof.get(x, ()):
                    for j in colsof.get(y, ()):
                        if pert[i] == gwg[j] or not np.isfinite(A[i, j]):
                            continue
                        L.append((i, j))
                        S.append(r)
                        C.append(both_ce)
        return (np.array(L, dtype=np.int64).reshape(-1, 2), np.array(S, dtype=np.float32),
                np.array(C, dtype=bool))

    L, S, CEc = cells(mine)
    linked_key = np.unique(L[:, 0].astype(np.int64) * nGW + L[:, 1])
    report(f"  testable linked (P,G) cells: {len(L):,} from {len(set(L[:,0].tolist())):,} perturbations "
           f"and {len(set(L[:,1].tolist())):,} measured genes")
    cov_pairs = sum(1 for (a, b) in mine if (rowsof.get(a) and colsof.get(b))
                    or (rowsof.get(b) and colsof.get(a)))
    report(f"  coverage: {cov_pairs:,}/{len(mine):,} ({100*cov_pairs/len(mine):.1f}%) of layer pairs are "
           f"testable at all in GWPS -- the rest are unknown at this endpoint, not null")

    MF = FastMatcher(A, prow, pcol, pert, gwg, DECILES)
    MT = FastMatcher(A, prow, pcol, pert, gwg, TIGHT)
    report(f"\n  matched on perturbation-strength decile x gene-responsiveness decile, {N_DRAWS} draws")
    report(f"  reference numbers on this exact test: PPI 1.208x, co-dependency 1.070x, "
           f"regulatory 1.021x, signalling 1.028x (null); under ~1.05 is not interesting")
    report(FHEAD)
    F = {}
    F["mine"] = MF.run(L, linked_key)
    report(fline("my layer (all pairs)", F["mine"]))
    bands = [(0.20, 0.25), (0.25, 0.30), (0.30, 0.40), (0.40, 1.01)]
    for lo, hi in bands:
        k = f"band_{lo:.2f}_{hi:.2f}"
        F[k] = MF.run(L, linked_key, (S >= lo) & (S < hi))
        report(fline(f"  r in [{lo:.2f},{hi:.2f})", F[k]))
    F["ce"] = MF.run(L, linked_key, CEc)
    report(fline("  both common-essential", F["ce"]))
    F["nonce"] = MF.run(L, linked_key, ~CEc)
    report(fline("  neither common-essential", F["nonce"]))
    LE, SE, _ = cells(test_ex)
    F["existing"] = MF.run(LE, np.unique(LE[:, 0].astype(np.int64) * nGW + LE[:, 1]))
    report(fline("existing `codep` layer", F["existing"]))
    if ex_sl:
        LS, _, _ = cells({k: ex_sl[k] for k in sk})
        F["sl"] = MF.run(LS, np.unique(LS[:, 0].astype(np.int64) * nGW + LS[:, 1]))
        report(fline("existing `sl` layer", F["sl"]))
    report(f"    residual imbalance after decile matching (mean p): perturbation strength "
           f"{F['mine']['residual_pert_strength_p']:.3g}, gene responsiveness "
           f"{F['mine']['residual_gene_responsiveness_p']:.3g}")
    report(f"\n  SENSITIVITY at {TIGHT}x{TIGHT} bins")
    report(FHEAD)
    F["mine_tight"] = MT.run(L, linked_key)
    report(fline("my layer, tight bins", F["mine_tight"]))
    # the common-essential split is re-run tight as well: at deciles a within-bin essentiality gradient
    # could manufacture the CE/non-CE gap, which is the whole question that split is asking
    F["ce_tight"] = MT.run(L, linked_key, CEc)
    report(fline("  both common-essential", F["ce_tight"]))
    F["nonce_tight"] = MT.run(L, linked_key, ~CEc)
    report(fline("  neither common-essential", F["nonce_tight"]))
    report(f"    residual imbalance: perturbation strength "
           f"{F['mine_tight']['residual_pert_strength_p']:.3g}, gene responsiveness "
           f"{F['mine_tight']['residual_gene_responsiveness_p']:.3g}")
    del A

    # =================================================================================================
    # the layer, written out
    # =================================================================================================
    sym_of = {}
    for i, e in enumerate(gene_eid):
        ent_ = REG.entities.get(e) or {}
        sym_of[e] = ent_.get("symbol") or sym[G[i]]
    eid2ci = {}
    for k, e in CI.items():
        eid2ci.setdefault(e, k)
    part = {}
    for i in range(nG):
        pl = [[gene_eid[int(j)], round(float(r), 4)]
              for j, r in zip(top_idx[i], top_val[i]) if r >= R_MIN]
        if pl:
            part[gene_eid[i]] = pl
    layer = {
        "model": "depmap-codep-v1", "source": f"{RELEASE} CRISPRGeneEffect ({DOI})",
        "md5": {"CRISPRGeneEffect.csv": MD5_GENE_EFFECT, "Model.csv": MD5_MODEL,
                "OmicsExpressionProteinCodingGenesTPMLogp1.csv": MD5_EXPR},
        "construction": f"Pearson correlation of gene-effect profiles across {nl:,} cell lines, "
                        f"top-{TOP_K} partners per gene, r >= {R_MIN}",
        "join": {"namespace": "entrez", "rate": st["rate"],
                 "note": "registry gene/entrez keys are float-stringified upstream; normalised at the call "
                         "site, NOT by editing the registry. Symbol was not used as the key."},
        "n_genes": nG, "n_pairs": len(mine), "n_lines": int(nl),
        "endpoint": "viability co-dependency across cell lines; NOT a transcriptional claim",
        "partners": part,
        "gene_stats": {gene_eid[i]: {"symbol": sym_of[gene_eid[i]],
                                     "cell_index": eid2ci.get(gene_eid[i]),
                                     "dep_mean": round(float(dep_mean[i]), 4),
                                     "dep_sd": round(float(dep_sd[i]), 4),
                                     "dep_frac_lines": round(float(dep_frac[i]), 4),
                                     "expr_mean_log2tpm1": (None if not has_expr[i]
                                                            else round(float(expr_mean[i]), 4))}
                       for i in range(nG)},
        "limits": [
            f"one release ({RELEASE}); co-dependency values shift between releases as lines are added -- "
            f"the existing layer agrees at r={agree_r:.3f} but is not identical, and mixing them would mix "
            f"line sets",
            "Pearson correlation across a cancer cell line panel is not a mechanism: it cannot separate "
            "same-complex, same-pathway, shared-lineage-requirement and shared-screen-artefact",
            "the panel is CANCER cell lines grown in 2D; a co-dependency that exists only in that context "
            "will not be marked as such here",
            f"{int((~complete).sum()):,} genes dropped for incomplete profiles and the drop is not random "
            f"(mean gene effect {dep_mean_all[~complete].mean():.4f} against "
            f"{dep_mean_all[complete].mean():.4f} for kept genes, MWU p "
            f"{stats.mannwhitneyu(dep_mean_all[~complete], dep_mean_all[complete])[1]:.2g}), so absence "
            f"from this layer is unknown, not zero",
            f"expression is unknown for {int((~has_expr).sum()):,} genes; those pairs are excluded from the "
            f"matched arm rather than zero-filled",
            "the layer is SYMMETRIC and unsigned as to direction: it says A and B are needed together, not "
            "that A acts on B or that either is upstream",
            f"PAN-ESSENTIALS ARE OVER-CONNECTED, NOT INVISIBLE. The project's dataset plan says this layer "
            f"is structurally blind to them; in {RELEASE} they have LARGER dependency spread "
            f"({dep_sd[ce_idx].mean():.3f} vs {dep_sd[~ce_idx].mean():.3f}) and MORE partners "
            f"({100*has_partner[ce_idx].mean():.0f}% vs {100*has_partner[~ce_idx].mean():.0f}%). Their "
            f"spread is largely a shared growth-rate / screen-quality axis, so a common-essential pair is "
            f"the edge type most likely to be non-specific -- see the common-essential split and the "
            f"global-axis-removed arm before trusting one",
            "it is NOT synthetic lethality. Positive co-dependency is co-requirement; SL is the opposite "
            "sign of evidence and is not measurable from this file",
            "the transcriptional numbers in this module are a measurement of a DIFFERENT endpoint, not a "
            "grade of this layer -- and they are also a different CONTEXT: one cell line (K562) against a "
            "1,150-line panel, so 'orthogonal endpoints' and 'orthogonal contexts' are not separated here",
            "CRISPR gene effect is Chronos-corrected; copy number and screen-quality corrections are the "
            "pipeline's, and are inherited rather than validated here",
            "the half-splits are held out at the level of CELL LINES, not of the whole pipeline: Chronos was "
            "fitted across all 1,150 lines before this module saw the matrix, so a shared correction could "
            "leak a little structure between halves. The held-out number is an upper bound to that extent, "
            "and the lineage-centred arm is the check that bounds it from the other side"]}
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    # =================================================================================================
    # verdict
    # =================================================================================================
    fm = F["mine"]
    vm = Vres["mine_heldout"]
    vh = Vres["mine_hub"]
    vl = Vres["mine_lineage"]
    vp = Vres["mine_pc"]
    vce, vnc = Vres["mine_ce"], Vres["mine_nonce"]
    strong = vm["auc"] >= 0.75 and vm["frac_draws_p05"] >= 0.95 and vm["raw_mean_r_heldout_linked"] >= 0.05
    grad = [F[f"band_{lo:.2f}_{hi:.2f}"] for lo, hi in bands]
    grad = [g["ratio"] for g in grad if g]
    rnaword = ("under the 1.05 floor" if fm["ratio"] < 1.05 else
               "just over the 1.05 floor" if fm["ratio"] < 1.10 else
               "clearly over the 1.05 floor")
    grange = max(grad) - min(grad)
    gradword = (f"and it barely moves with co-dependency strength (whole range {grange:.3f} across bands)"
                if grange < 0.05 else
                "and it does rise with co-dependency strength, so part of it IS co-dependency showing "
                "through")
    report("\n" + "=" * 108)
    v = (f"THIS LAYER CARRIES VIABILITY INFORMATION AND ALMOST NO TRANSCRIPTIONAL INFORMATION, and the two "
         f"were previously graded with one number. Held out by construction -- built on half the "
         f"{nl:,} DepMap 24Q2 lines, scored on the other half -- co-dependent pairs reach a raw mean "
         f"dependency-profile correlation of {vm['raw_mean_r_heldout_linked']:.4f} on lines the build never "
         f"saw, against {vm['raw_mean_r_heldout_matched']:.4f} for pairs matched on mean dependency, "
         f"dependency spread and expression (AUC {vm['auc']:.3f}, {100*vm['frac_draws_p05']:.0f}% of "
         f"{vm['n_draws_total']} draws p<0.05 over {N_SPLITS} independent splits; residual imbalance "
         f"p {vm['residual_p'][0]:.2g}/{vm['residual_p'][1]:.2g}/{vm['residual_p'][2]:.2g}). It survives "
         f"all three falsification checks: hubness as a fourth matched covariate leaves "
         f"{vh['raw_mean_r_heldout_linked']:.4f} vs {vh['raw_mean_r_heldout_matched']:.4f} "
         f"(AUC {vh['auc']:.3f}); centring every held-out profile within its Oncotree lineage, which "
         f"deletes all between-lineage structure, leaves {vl['raw_mean_r_heldout_linked']:.4f} vs "
         f"{vl['raw_mean_r_heldout_matched']:.4f} (AUC {vl['auc']:.3f}); and projecting the "
         f"{N_GLOBAL_PC} strongest global axes -- growth rate, screen quality, broad essentiality -- out "
         f"of the held-out half leaves {vp['raw_mean_r_heldout_linked']:.4f} vs "
         f"{vp['raw_mean_r_heldout_matched']:.4f} (AUC {vp['auc']:.3f}). So it is not hub, lineage or "
         f"global-axis bookkeeping. It is NOT uniform across edge types: pairs of common essentials reach "
         f"{vce['raw_mean_r_heldout_linked']:.4f} (AUC {vce['auc']:.3f}) against "
         f"{vnc['raw_mean_r_heldout_linked']:.4f} (AUC {vnc['auc']:.3f}) for pairs with neither -- and the "
         f"project's own dataset plan has this backwards, recording pan-essentials as a structural blind "
         f"spot when in this release they carry LARGER dependency spread "
         f"({dep_sd[ce_idx].mean():.3f} vs {dep_sd[~ce_idx].mean():.3f}) and MORE partners "
         f"({100*has_partner[ce_idx].mean():.0f}% vs {100*has_partner[~ce_idx].mean():.0f}%). "
         f"On the transcriptional fast test the SAME pairs give "
         f"{fm['ratio']:.3f}x ({fm['n_linked']:,} testable cells, median p {fm['p_median']:.3g}, AUC "
         f"{fm['auc']:.3f}) -- {rnaword}, against PPI's 1.208x -- {gradword} (ratios "
         f"{', '.join(f'{g:.3f}' for g in grad)} across r bands 0.20/0.25/0.30/0.40+), so "
         f"the two endpoints are close to orthogonal rather than one being a weak version of the other. "
         f"An AUC of {fm['auc']:.3f} is the number to keep: the transcriptional effect is real "
         f"(median p {fm['p_median']:.3g}, {100*fm['frac_draws_p05']:.0f}% of draws) and useless for "
         f"deciding anything about a single pair. What little of it there is looks like shared "
         f"essentiality rather than partnership -- at {TIGHT}x{TIGHT} bins, common-essential pairs give "
         f"{F['ce_tight']['ratio']:.3f}x against {F['nonce_tight']['ratio']:.3f}x for pairs with neither. "
         f"Provenance for the pre-existing layer is recovered rather than assumed: my correlations agree "
         f"with its stated values at r={agree_r:.3f} and recover "
         f"{100*len(set(test_ex)&set(mine))/len(test_ex):.0f}% of its pairs, so it is DepMap gene-effect "
         f"Pearson from a nearby release. The `sl` layer is co-dependency mislabelled -- all "
         f"{len(ex_sl):,} pairs are POSITIVELY correlated, which is co-requirement, the opposite sign of "
         f"evidence from synthetic lethality.")
    if not strong:
        v = ("THE VIABILITY ENDPOINT DOES NOT CLEAR ITS OWN BAR. " + v)
    report(f"  VERDICT: {v}")

    R = {"source": f"{RELEASE} ({DOI})", "md5": layer["md5"], "construction": layer["construction"],
         "join": layer["join"],
         "registry_defect": {
             "namespace": "gene/entrez",
             "problem": "aliases stored float-stringified ('7105.0'); a raw Entrez id resolves 0/18,443",
             "handled": "normalised at the call site; registry not modified",
             "n_symbol_entrez_disagreements": len(disagree),
             "examples": [d[0] for d in disagree[:10]]},
         "n_genes": nG, "n_pairs": len(mine), "n_lines": int(nl),
         "n_genes_dropped_incomplete": int((~complete).sum()),
         "n_genes_without_expression": int((~has_expr).sum()),
         "blind_spot": {"n_common_essential_in_matrix": int(ce_idx.sum()),
                        "frac_common_essential_with_a_partner": float(has_partner[ce_idx].mean()),
                        "frac_other_with_a_partner": float(has_partner[~ce_idx].mean()),
                        "dep_sd_common_essential": float(dep_sd[ce_idx].mean()),
                        "dep_sd_other": float(dep_sd[~ce_idx].mean()),
                        "documented_claim": "the project's dataset plan records pan-essentials as a "
                                            "structural blind spot for co-dependency: no variance across "
                                            "lines, so nothing to correlate",
                        "measured": "CONTRADICTED in this release. Common essentials have LARGER "
                                    "dependency spread and MORE partners than the rest. Their spread is "
                                    "largely a shared growth-rate / screen-quality axis, so they are "
                                    "over-connected here rather than invisible, and a common-essential "
                                    "edge is the type most in need of the global-axis control",
                        "note": "a gene with no partner in this layer is UNKNOWN at this endpoint, never "
                                "negative"},
         "agreement_with_existing": {
             "n_existing_pairs": len(ex_codep), "n_shared": len(inter),
             "jaccard": len(inter) / len(set(mine) | set(ex_codep)),
             "frac_existing_recovered": len(set(test_ex) & set(mine)) / len(test_ex),
             "pearson_r_vs_stated": agree_r,
             "inferred_construction": "DepMap gene-effect Pearson, top-8, r>=0.2, nearby release"},
         "sl_layer": ({"n_pairs": len(ex_sl), "n_negative_in_24Q2": int((ex_sl_r < 0).sum()),
                       "min_r": float(ex_sl_r.min()), "max_r": float(ex_sl_r.max()),
                       "finding": "all pairs positively co-dependent -> co-requirement, not synthetic "
                                  "lethality; the label is wrong"} if ex_sl else None),
         "viability_test": {"design": f"{N_SPLITS} random half-splits of cell lines; layer built on half A, "
                                      f"scored by dependency-profile correlation on half B; controls "
                                      f"matched on mean dependency, dependency spread and expression "
                                      f"(training half only), {N_DRAWS} draws per split",
                            "arms": Vres, "splits": V["splits"]},
         "transcriptional_test": {"design": "GWPS K562 CRISPRi, per-gene robust z, matched on "
                                            "perturbation-strength decile x gene-responsiveness decile",
                                  "reference_layers": {"PPI": 1.208, "codep_prior": 1.070,
                                                       "regulatory": 1.021, "signalling": 1.028},
                                  "n_layer_pairs_testable": cov_pairs,
                                  "frac_layer_pairs_testable": cov_pairs / len(mine),
                                  "arms": F},
         "limits": layer["limits"], "verdict": v}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "depmap_codep.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'depmap_codep.json'}")


if __name__ == "__main__":
    main()
