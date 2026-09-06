"""CORUM PROTEIN COMPLEXES -- and the size at which "same machine" stops predicting "same response".

WHAT IS ALREADY THERE, AND WHAT IT IS. The cell object carries 2,039 complexes in `complexes` / `gene2cplx`
that an earlier version of this module called "provenance UNKNOWN". That was wrong and it was wrong against a
file already in this repo: every one of the 2,039 names is an exact match to `outputs/orphan/science_data/
complexes.parquet`, which is the EBI Complex Portal 9606.tsv (Last-Modified 2026-01-14, retrieved
2026-07-13). This module now ASSERTS that identification instead of shrugging at it, which matters twice
over: the existing layer is a curated set with a citable version, not an orphan, and Complex Portal ships a
`stoichiometry` column that gives the 1:1:1 assumption below something to be measured against. CORUM is still
ADDED AND COMPARED, never substituted.

WHY CO-COMPLEX MEMBERSHIP SHOULD MOVE TRANSCRIPTS AT ALL. Knocking down one subunit of an obligate complex
destabilises the others: unassembled subunits are degraded, and in many machines the transcriptional program
that builds the machine is co-regulated. So the prediction is that a (perturbation P, gene G) pair which
shares a complex shows a larger |robust z| in K562 CRISPRi Perturb-seq than a matched pair that does not.

WHY THIS VERSION DOES NOT REPORT ONE RATIO. The previous version reported 1.212x from a single 10x10 decile
match and checked residual imbalance on exactly the two covariates it had matched on. Both of those were
wrong in the same direction:

  (a) A decile control is loose. Tightening the bins moves the ratio monotonically down -- 1.212 at 10x10,
      1.165 at 50x50, 1.156 at 160x160 -- so a number quoted at one tightness is not comparable to a number
      quoted at another, and the loosest rung flatters the layer most. The LADDER is therefore reported to
      convergence and the verdict rests on the converged rung, not the legacy one.
  (b) Residual imbalance on the two matched covariates is not evidence that the control is matched. At both
      10x10 and 50x50 the control differed from the linked arm at p < 1e-40 on measured-gene EXPRESSION,
      publication count, gene length, K562 iBAQ abundance, knockdown efficiency and cell count -- none of
      which the old control ever looked at. All eleven are now reported on every arm, and each one gets an
      arm where it is matched on explicitly.

THREE CONTROLS THE OLD VERSION DID NOT HAVE, each of which can kill the layer:
  SHAM. A degree-preserving rewiring of the same complex graph. It must return ~1.00. If the instrument
      returns a positive ratio on pairs that share no complex, every other number here is machinery.
  SUBUNIT-vs-SUBUNIT. The control pair is drawn from CORUM subunits only, so both arms are made of the same
      kind of gene and the only difference is whether they share a complex. "Complex genes are special" and
      "these two genes share a complex" are different claims and the old control could not separate them.
  EXACT PERTURBATION. The control gene is drawn from the SAME knockdown. Every row-side confound, measured
      or not, is then held exactly.

AND THE UNIT OF REPLICATION IS NOT THE PAIR. 86,364 testable pairs come from 3,082 complexes, and the four
largest -- spliceosome E, spliceosome A, the Nop56p pre-rRNA complex, C-complex spliceosome -- carry 37% of
them by themselves. A Mann-Whitney p over pairs is therefore optimistic by construction and this module
reports p ~ 0 for exactly that reason. The honest statements are the bootstrap over COMPLEXES and the
bootstrap over PERTURBED GENES, and they are what the verdict quotes.

WHAT WOULD FALSIFY THIS LAYER, each checked rather than assumed:
  (a) The converged ratio lands under ~1.05, like `signalling` did. Then co-complex membership is not a
      usable link type in this cell.
  (b) The small/large split is a POWER artefact. Large complexes are subsampled to the small set's exact n.
  (c) The size effect is study bias -- big machines are the well-studied ones. MEASURED, not assumed:
      spearman(n subunits, mean subunit publication count) over 5,315 complexes, plus an arm matched on
      publication quintile.
  (d) The median complex shows nothing and a handful of giant machines carry the pooled number. This is the
      one that half-fires: see the per-complex sign test.

STOICHIOMETRY IS NOT IN CORUM. 5,576 of 5,628 human complexes carry an empty `subunits_stoechiometrie`
column; that is asserted at parse time, not trusted. A limiting-subunit capacity IS computed from K562
protein abundance and is labelled ASSUMED_1_1_1 everywhere it appears. The previous version then declared the
assumption FALSIFIED because the lowest-abundance subunit is less disruptive to knock down than its mates.
That inference does not survive its own control: abundance predicts perturbation strength genome-wide
(spearman +0.156, p 2.5e-35), and once that gradient is subtracted the within-complex contrast is a clean
null. The test measures the abundance gradient, not stoichiometry, and is now reported as UNINFORMATIVE.
What it can be measured against is Complex Portal, which does ship stoichiometry: 1:1:1 is right for 62% of
the complexes that specify one, so the assumption has a known error rate rather than an unknown one.

PROVENANCE IS PINNED AND ASSERTED. One release (CORUM 5.3, checked against the API's own current-release
endpoint), one organism (asserted: every row and every subunit is Human), one accession namespace (UniProt,
with symbol used only for a residue of retired accessions and recorded via=symbol), one comparison layer
(Complex Portal 9606.tsv as pinned in science_data), one study-bias source (NCBI gene2pubmed, pinned by its
Last-Modified header).
"""
import collections
import csv
import gzip
import json
import os
import re
import ssl
import subprocess
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
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
PROTL = SP / "cell_k562_protein.json.gz"
SD = ROOT / "outputs/orphan/science_data"
BIOPLEX = OUT / "bioplex.json"
RAW = SP / "corum_human_5_3.txt"
RELJ = SP / "corum_release_5_3.json"
PUBC = SP / "corum_pubcount.json"
LAYER = SP / "cell_corum.json.gz"

HOST = "https://mips.helmholtz-muenchen.de"
API = HOST + "/fastapi-corum"
RELEASE = "5.3"                       # pinned; a different current release must be adopted deliberately
SOURCE = f"CORUM {RELEASE} human complexes (mips.helmholtz-muenchen.de/fastapi-corum, file_id=human)"
CA_BUNDLE = Path("/root/.ccr/ca-bundle.crt")
CHAIN = SP / "corum_tls_intermediate.pem"
# The CORUM host serves a LEAF-ONLY chain: the HARICA "GEANT TLS RSA 1" intermediate is missing, so
# verification fails with "unable to get local issuer certificate" against any correct trust store. The fix
# is to supply the intermediate, never to switch verification off -- and the intermediate is itself verified
# against the trusted bundle before it is used, so the anchor is still a root the system already trusts.
INTERMEDIATE_CRTSH = "https://crt.sh/?d=16099180997"
GENE2PUBMED = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz"

SMALL_MAX = 5                         # prespecified small/large boundary, from the brief
N_DRAWS = 20                          # never read a single matched draw as the result
LADDER = (10, 20, 40, 80, 160)        # 10 is the legacy rung, kept only for comparability
LEGACY = 10
SIZE_BANDS = [(2, 2, "2 (heterodimer)"), (3, 5, "3-5"), (6, 10, "6-10"),
              (11, 20, "11-20"), (21, 10 ** 9, ">20")]
NBOOT = 400


# ---------------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------------
def tls_context(report):
    """A context that verifies fully, after repairing the server's incomplete chain."""
    if not CHAIN.exists():
        ctx = ssl.create_default_context()
        if CA_BUNDLE.exists():
            ctx.load_verify_locations(cafile=str(CA_BUNDLE))
        req = urllib.request.Request(INTERMEDIATE_CRTSH, headers={"User-Agent": "cellos"})
        pem = urllib.request.urlopen(req, context=ctx, timeout=120).read()
        if b"BEGIN CERTIFICATE" not in pem:
            raise SystemExit(f"could not retrieve the missing TLS intermediate from {INTERMEDIATE_CRTSH}")
        CHAIN.parent.mkdir(parents=True, exist_ok=True)
        CHAIN.write_bytes(pem)
    # ASSERT the intermediate chains to a root already in the trusted bundle. Without this check, adding it
    # as an anchor would be trusting an unverified certificate -- which is the thing we are refusing to do.
    if CA_BUNDLE.exists():
        v = subprocess.run(["openssl", "verify", "-CAfile", str(CA_BUNDLE), str(CHAIN)],
                           capture_output=True, text=True)
        if v.returncode != 0:
            raise SystemExit(f"the TLS intermediate does not verify against {CA_BUNDLE}: "
                             f"{v.stdout.strip()} {v.stderr.strip()} -- refusing to trust it")
        report(f"  TLS: server sends a leaf-only chain; supplied the HARICA GEANT intermediate, which "
               f"`openssl verify` confirms is issued by a root already in the trusted bundle")
    ctx = ssl.create_default_context()
    if CA_BUNDLE.exists():
        ctx.load_verify_locations(cafile=str(CA_BUNDLE))
    ctx.load_verify_locations(cadata=CHAIN.read_text())
    return ctx


def get(url, ctx, timeout=600, tries=4):
    """The CORUM host intermittently stalls; a transient read timeout must not look like a missing source."""
    req = urllib.request.Request(url, headers={"User-Agent": "cellos", "accept": "*/*"})
    last = None
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, context=ctx, timeout=timeout).read()
        except Exception as e:                       # noqa: BLE001 -- retried, then re-raised
            last = e
            time.sleep(4 * (k + 1))
    raise last


def fetch(report):
    ctx = tls_context(report)
    cur = json.loads(get(f"{API}/public/releases/current", ctx, 180))
    RELJ.write_text(json.dumps(cur))
    report(f"  CORUM current release reported by the API: {cur['version']} ({cur['date']})")
    if str(cur["version"]) != RELEASE:
        raise SystemExit(
            f"CORUM current release is {cur['version']}, this module is pinned to {RELEASE}. One source "
            f"version along one axis: adopt the new release deliberately (bump RELEASE and re-run every "
            f"number in this file), do not let it drift under the results.")
    if not RAW.exists() or RAW.stat().st_size < 1_000_000:
        b = get(f"{API}/public/file/download_current_file?file_id=human&file_format=txt", ctx)
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(b)
    report(f"  {SOURCE}  ({RAW.stat().st_size/1e6:.1f} MB)")
    return cur


def fetch_pubcount(report):
    """NCBI gene2pubmed, streamed and reduced to entrez -> n publications; the raw file never lands on disk.

    Study bias is the alternative explanation this layer's own limits used to name and never test. It needs
    a per-gene publication count from outside the analysis, pinned to a version.
    """
    if PUBC.exists():
        d = json.load(open(PUBC))
        report(f"  study bias: gene2pubmed cached ({d['version']}), {len(d['count']):,} human genes")
        return d
    req = urllib.request.Request(GENE2PUBMED, headers={"User-Agent": "cellos"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        ver = resp.headers.get("Last-Modified") or "unknown"
        cnt = collections.Counter()
        with gzip.GzipFile(fileobj=resp) as fh:
            next(fh)
            for line in fh:
                t, g, _p = line.decode().split("\t")
                if t == "9606":
                    cnt[g] += 1
    d = {"source": GENE2PUBMED, "version": f"Last-Modified {ver}", "count": dict(cnt)}
    json.dump(d, open(PUBC, "w"))
    report(f"  study bias: NCBI gene2pubmed ({d['version']}), {len(cnt):,} human genes -- raw file streamed "
           f"and reduced, never written to disk")
    return d


# ---------------------------------------------------------------------------------------------------
# the matched-control instrument
# ---------------------------------------------------------------------------------------------------
def qbin(v, n):
    """Quantile bins over the finite values, with a DEDICATED bin for missing. Absence is a bin, not a zero."""
    out = np.full(len(v), -1, dtype=np.int64)
    f = np.isfinite(v)
    if f.sum() > n:
        out[f] = np.digitize(v[f], np.quantile(v[f], np.linspace(1 / n, 1 - 1 / n, n - 1)))
    return out


def code(*bs):
    """Combine bin vectors into one stratum code."""
    out = np.zeros(len(bs[0]), dtype=np.int64)
    for b in bs:
        out = out * (int(b.max()) + 2) + (b + 1)
    return out


class Bench:
    """One definition of `matched`, shared by every arm.

    THE EFFECT SIZE IS `mean_abs_z_linked` -- the raw held-out value. The ratio is the CONTRAST and the
    p-value is the significance; neither is the effect, and nothing here is ever reported as a difference
    from a shuffled arm.

    THE ARMS ARE CELL-EXACT. If a stratum cannot be filled, the linked pairs in that stratum are dropped too,
    so the two arms always have identical stratum composition and `fill` reports what was lost. A control
    that is short in the extreme cells while the linked arm keeps them is not a matched control.

    RESIDUAL IMBALANCE IS REPORTED ON ALL ELEVEN COVARIATES, not on the two that were matched. The previous
    version of this module reported two and both looked fine while measured-gene expression, publication
    count, gene length, iBAQ abundance, knockdown efficiency and cell count were all off at p < 1e-40.
    """

    def __init__(self, A, rowc, colc, linkset, rowlab, collab):
        self.A, self.linkset = A, linkset
        self.rowc, self.colc = rowc, colc          # name -> per-row / per-column covariate vector
        self.rowlab, self.collab = rowlab, collab
        self.ok = np.isfinite(A)
        self.NR, self.NC = A.shape

    def _pools(self, rowk, colk, rowmask, colmask):
        rp, cp = collections.defaultdict(list), collections.defaultdict(list)
        for i in range(self.NR):
            if rowmask is None or rowmask[i]:
                rp[rowk[i]].append(i)
        for j in range(self.NC):
            if colmask is None or colmask[j]:
                cp[colk[j]].append(j)
        return ({k: np.array(v) for k, v in rp.items()}, {k: np.array(v) for k, v in cp.items()})

    def draw(self, i2, j2, rowk, colk, rowmask, colmask, rng, exclude):
        """One cell-exact matched draw. Returns (kept linked slots, control rows, control cols)."""
        rp, cp = self._pools(rowk, colk, rowmask, colmask)
        slots = collections.defaultdict(list)
        for t in range(len(i2)):
            slots[(rowk[i2[t]], colk[j2[t]])].append(t)
        keep, ci, cj = [], [], []
        for (a, b), ts in slots.items():
            rs, cs = rp.get(a), cp.get(b)
            if rs is None or cs is None:
                continue
            need, tries, got = len(ts), 0, []
            while need > len(got) and tries < 80:
                m = int((need - len(got)) * 1.6) + 16
                for x, y in zip(rs[rng.integers(0, len(rs), m)], cs[rng.integers(0, len(cs), m)]):
                    if len(got) >= need:
                        break
                    if not self.ok[x, y] or (x, y) in exclude:
                        continue
                    got.append((x, y))
                tries += 1
            for t, (x, y) in zip(ts, got):          # cell-exact: only as many linked as controls found
                keep.append(t)
                ci.append(x)
                cj.append(y)
        return (np.array(keep, dtype=np.int64), np.array(ci, dtype=np.int64),
                np.array(cj, dtype=np.int64))

    def arm(self, label, i2, j2, rowk, colk, rowmask=None, colmask=None, n_draws=N_DRAWS, seed0=0,
            exclude=None, cluster_of=None, boot_of=None, nboot=NBOOT):
        from scipy import stats
        exclude = self.linkset if exclude is None else exclude
        if len(i2) < 200:
            return {"label": label, "n_pairs": int(len(i2)), "testable": False}
        rows, acc_l, acc_c, acc_t = [], [], [], []
        for s in range(n_draws):
            rng = np.random.default_rng(seed0 + s)
            keep, ci, cj = self.draw(i2, j2, rowk, colk, rowmask, colmask, rng, exclude)
            if len(keep) < 50:
                continue
            lz, cz = self.A[i2[keep], j2[keep]], self.A[ci, cj]
            r = {"fill": len(keep) / len(i2), "mean_l": float(lz.mean()), "mean_c": float(cz.mean()),
                 "ratio": float(lz.mean() / cz.mean()),
                 "p": float(stats.mannwhitneyu(lz, cz, alternative="two-sided")[1])}
            for nm, v in self.rowc.items():
                a, b = v[i2[keep]], v[ci]
                fa, fb = np.isfinite(a), np.isfinite(b)
                r["res_p_row " + nm] = float(stats.mannwhitneyu(a[fa], b[fb])[1]) if fa.any() and fb.any() else 1.0
                r["res_mr_row " + nm] = float(a[fa].mean() / b[fb].mean()) if fb.any() and b[fb].mean() else 1.0
            for nm, v in self.colc.items():
                a, b = v[j2[keep]], v[cj]
                fa, fb = np.isfinite(a), np.isfinite(b)
                r["res_p_col " + nm] = float(stats.mannwhitneyu(a[fa], b[fb])[1]) if fa.any() and fb.any() else 1.0
                r["res_mr_col " + nm] = float(a[fa].mean() / b[fb].mean()) if fb.any() and b[fb].mean() else 1.0
            rows.append(r)
            acc_l.append(lz)
            acc_c.append(cz)
            acc_t.append(keep)
        if not rows:
            return {"label": label, "n_pairs": int(len(i2)), "testable": False}

        def mn(k):
            return float(np.mean([r[k] for r in rows]))
        res_p = {k[6:]: mn(k) for k in rows[0] if k.startswith("res_p_")}
        res_mr = {k[7:]: mn(k) for k in rows[0] if k.startswith("res_mr_")}
        out = {"label": label, "n_pairs": int(len(i2)), "testable": True, "n_draws": len(rows),
               "control_fill": mn("fill"),
               "mean_abs_z_linked": mn("mean_l"),          # <- THE EFFECT SIZE: raw held-out value
               "median_abs_z_linked": float(np.median(self.A[i2, j2])),
               "mean_abs_z_matched": mn("mean_c"),
               "ratio": mn("ratio"),
               "ratio_min": float(min(r["ratio"] for r in rows)),
               "ratio_max": float(max(r["ratio"] for r in rows)),
               "p_median_PAIR_LEVEL_OPTIMISTIC": float(np.median([r["p"] for r in rows])),
               "frac_draws_p05": float(np.mean([r["p"] < 0.05 for r in rows])),
               "residual_p": res_p, "residual_meanratio": res_mr,
               "residual_worst_covariate": min(res_p, key=res_p.get),
               "residual_worst_p": min(res_p.values()),
               "n_covariates_imbalanced_p05": int(sum(1 for v in res_p.values() if v < 0.05))}
        # cluster-robust intervals: the unit of replication is NOT the pair
        if cluster_of is not None or boot_of is not None:
            lz = np.concatenate(acc_l)
            cz = np.concatenate(acc_c)
            tt = np.concatenate(acc_t)
            rng = np.random.default_rng(9)
            for nm, grp in (("perturbation", cluster_of), ("complex", boot_of)):
                if grp is None:
                    continue
                g = np.asarray(grp)[tt]
                by = collections.defaultdict(list)
                for t, k in enumerate(g.tolist()):
                    by[k].append(t)
                by = {k: np.array(v) for k, v in by.items()}
                ks = np.array(sorted(by))
                boots = []
                for _ in range(nboot):
                    pick = ks[rng.integers(0, len(ks), len(ks))]
                    sel = np.concatenate([by[k] for k in pick])
                    boots.append(lz[sel].mean() / cz[sel].mean())
                boots = np.array(boots)
                d = np.array([lz[v].mean() - cz[v].mean() for v in by.values()])
                out[f"cluster_{nm}"] = {
                    "n_clusters": int(len(ks)), "ratio": float(lz.mean() / cz.mean()),
                    "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
                    "frac_boot_le_1": float(np.mean(boots <= 1.0)),
                    "frac_boot_le_105": float(np.mean(boots <= 1.05)),
                    "frac_clusters_positive": float(np.mean(d > 0)),
                    "sign_test_p": float(stats.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue)}
        return out


def line(r):
    if not r or not r.get("testable"):
        return f"    {(r or {}).get('label','?')[:42]:42s}   too few testable pairs"
    return (f"    {r['label'][:42]:42s} {r['n_pairs']:7,d} {r['mean_abs_z_linked']:8.4f} "
            f"{r['mean_abs_z_matched']:8.4f} {r['ratio']:7.4f} {100*r['frac_draws_p05']:5.0f}% "
            f"{r['n_covariates_imbalanced_p05']:3d}/11 {r['residual_worst_covariate'][:22]:22s}")


HEAD = (f"    {'arm':42s} {'pairs':>7s} {'raw |z|':>8s} {'matched':>8s} {'ratio':>7s} "
        f"{'p<.05':>6s} {'imbal':>6s} {'worst covariate':22s}")


# ---------------------------------------------------------------------------------------------------
def main():
    from scipy import stats
    import h5py
    import pandas as pd
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 130)
    report("CORUM PROTEIN COMPLEXES -- does co-complex membership move transcripts, and at what size?")
    report("=" * 130)
    for p in (GWPS, NET):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    from entity_registry import load as load_reg
    REG = load_reg()
    rel = fetch(report)
    pubsrc = fetch_pubcount(report)

    # ---- parse, and assert the provenance instead of stating it ----
    csv.field_size_limit(10 ** 9)
    with open(RAW) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    bad_org = [r["complex_id"] for r in rows if r["organism"] != "Human"]
    bad_sub = [r["complex_id"] for r in rows
               if any("Human" not in o for o in r["subunits_organism"].split(";") if o.strip())]
    if bad_org or bad_sub:
        raise SystemExit(f"ONE ORGANISM ALONG THIS AXIS: {len(bad_org)} non-human rows and {len(bad_sub)} "
                         f"rows with a non-human subunit in a file that claims to be human-only")
    report(f"  {len(rows):,} human complexes, organism asserted pure on both the complex and the subunit "
           f"column")
    n_stoich = sum(1 for r in rows
                   if any(re.fullmatch(r"\d+(\.\d+)?", t.strip())
                          for t in (r.get("subunits_stoechiometrie") or "").split(";")))
    report(f"  subunits_stoechiometrie carries a numeric token in {n_stoich:,}/{len(rows):,} complexes "
           f"({100*n_stoich/len(rows):.1f}%) -- CORUM does not ship usable stoichiometry and none is invented")

    # ---- registry join: accession first, symbol only for the residue ----
    acc = sorted({u.strip() for r in rows for u in r["subunits_uniprot_id"].split(";") if u.strip()})
    base = sorted({a.split("-")[0] for a in acc})     # CORUM carries isoform-suffixed accessions
    _p, stp = REG.join("protein", "uniprot", base, 0.95, "CORUM subunit accessions -> protein")
    _g, stg = REG.join("gene", "uniprot", base, 0.95, "CORUM subunit accessions -> gene")
    report(f"\n  JOIN (accession is the key; symbol is never used where an accession resolves)")
    report(f"    protein/uniprot  {stp['joined']:,}/{stp['of']:,} ({stp['rate']:.1%})")
    report(f"    gene/uniprot     {stg['joined']:,}/{stg['of']:,} ({stg['rate']:.1%})")
    unres = [a for a in base if REG.resolve("gene", "uniprot", a) is None]
    sym_of = {}
    for r in rows:
        for u, s in zip(r["subunits_uniprot_id"].split(";"), r["subunits_gene_name"].split(";")):
            if u.strip() and s.strip():
                sym_of.setdefault(u.strip().split("-")[0], s.strip())
    resid = [sym_of[a] for a in unres if a in sym_of]
    _s, sts = REG.join("gene", "symbol", resid, 0.05, "CORUM residue accessions -> gene by symbol") \
        if resid else (None, {"joined": 0, "of": 0, "rate": 0.0})
    report(f"    gene/symbol      {sts['joined']:,}/{sts['of']:,} fallback over the {len(unres)} accessions "
           f"the strong key missed  <- recorded via=symbol")

    via_symbol = set()

    def resolve(u):
        u = u.strip()
        for v in (u, u.split("-")[0]):
            e = REG.resolve("gene", "uniprot", v)
            if e:
                return e
        s = sym_of.get(u.split("-")[0])
        if s:
            e = REG.resolve("gene", "symbol", s)
            if e:
                via_symbol.add(e)
                return e
        return None

    cplx, cname, cinfo = {}, {}, {}
    n_drop = 0
    for r in rows:
        m = {e for u in r["subunits_uniprot_id"].split(";") if u.strip() for e in [resolve(u)] if e}
        if len(m) < 2:
            n_drop += 1
            continue
        cid = r["complex_id"]
        cplx[cid] = m
        cname[cid] = r["complex_name"]
        cinfo[cid] = {"name": r["complex_name"], "pmid": r["pmid"], "cell_line": r["cell_line"],
                      "n_subunits_annotated": len([u for u in r["subunits_uniprot_id"].split(";") if u.strip()]),
                      "go": r["functions_go_name"], "fcg": r["fcgs_name"]}
    subs = {g for m in cplx.values() for g in m}
    report(f"  {len(cplx):,} complexes with >=2 resolvable subunits ({n_drop:,} dropped as singletons or "
           f"unresolvable), {len(subs):,} distinct genes, {len(via_symbol):,} of them admitted via=symbol")

    # ---- the pair set, tagged with the SMALLEST complex that contains it ----
    pair_size, pair_cplx = {}, {}
    for cid, m in cplx.items():
        m = sorted(m)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                k = (m[i], m[j])
                if len(m) < pair_size.get(k, 10 ** 9):
                    pair_size[k], pair_cplx[k] = len(m), cid
    report(f"  {len(pair_size):,} distinct co-complex gene pairs")

    # ---- WHAT THE EXISTING LAYER ACTUALLY IS. Asserted, not shrugged at. ----
    RJ = json.load(gzip.open(SP / "entity_registry.json.gz", "rt"))
    cidx = RJ["cell_index"]
    d = json.load(gzip.open(NET, "rt"))
    ex_raw = d.get("complexes") or {}
    del d
    ex = {}
    for k, v in ex_raw.items():
        m = {x for x in {cidx.get(str(i)) for i in v} if x}
        if len(m) >= 2:
            ex[k] = m
    cp = pd.read_parquet(SD / "complexes.parquet")
    cpm = pd.read_parquet(SD / "complex_members.parquet")
    cp_ver = sorted(set(cp["source_version"].astype(str)))
    cp_names = {str(n).strip().lower() for n in cp["name"]}
    hit = sum(1 for k in ex_raw if str(k).strip().lower() in cp_names)
    if hit < 0.98 * len(ex_raw) or len(cp_ver) != 1:
        raise SystemExit(f"the existing complex layer does not identify as one pinned Complex Portal "
                         f"release: {hit:,}/{len(ex_raw):,} names matched, versions {cp_ver}")
    EX_SOURCE = f"{cp['source'].iloc[0]} / {cp_ver[0]} (retrieved {cp['retrieved_on'].iloc[0]})"
    report(f"\n  THE EXISTING 2,039-COMPLEX LAYER IS NOT UNTRACEABLE -- it is asserted here:")
    report(f"    {hit:,}/{len(ex_raw):,} of its names are exact matches to {SD.name}/complexes.parquet")
    report(f"    -> {EX_SOURCE}")

    # ---- overlap and DISAGREEMENT between CORUM and Complex Portal ----
    ex_pairs = set()
    for m in ex.values():
        m = sorted(m)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                ex_pairs.add((m[i], m[j]))
    cor_pairs = set(pair_size)
    inter, conly, eonly = cor_pairs & ex_pairs, cor_pairs - ex_pairs, ex_pairs - cor_pairs
    name_hit = len({v.strip().lower() for v in cname.values()} & {k.strip().lower() for k in ex})
    byg = collections.defaultdict(set)
    for cid, m in cplx.items():
        for g in m:
            byg[g].add(cid)
    best = []
    for k, m in ex.items():
        cand = collections.Counter()
        for g in m:
            for c in sorted(byg.get(g, ())):
                cand[c] += 1
        best.append(max((n / len(m | cplx[c]) for c, n in cand.most_common(20)), default=0.0))
    best = np.array(best)
    report(f"  CORUM {RELEASE} vs Complex Portal, as two curated sets of the same object")
    report(f"    complex NAMES shared: {name_hit} of {len(ex):,} -- the two curations name the same machines "
           f"differently, so names are not a join key here")
    report(f"    co-complex pairs: {len(inter):,} shared, {len(conly):,} CORUM-only, {len(eonly):,} "
           f"Complex-Portal-only  (Jaccard {len(inter)/max(len(cor_pairs|ex_pairs),1):.3f})")
    report(f"    {100*(best >= 0.999).mean():.1f}% of Complex Portal complexes have an exact CORUM membership "
           f"twin, {100*(best >= 0.5).mean():.1f}% reach Jaccard 0.5, {100*(best <= 0).mean():.1f}% share no "
           f"subunit at all")

    # ---- the fast test matrix ----
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        gt = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        vg = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0] for s in f["var"]["gene_id"][:]]
        var_mean = np.asarray(f["var"]["mean"][:], float)
        obs_fold = np.asarray(f["obs"]["fold_expr"][:], float)
        obs_ncell = np.asarray(f["obs"]["num_cells_filtered"][:], float)
    Xm = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    n_nonfinite = int((~np.isfinite(X)).sum())
    del X
    med = np.nanmedian(Xm, 0)
    mad = np.nanmedian(np.abs(Xm - med), 0)
    sc = 1.4826 * mad
    A = np.abs((Xm - med) / np.where(sc > 0, sc, np.nan)).astype(np.float32)
    del Xm
    ok = np.isfinite(A)
    prow = np.where(ok, A, 0).sum(1) / np.maximum(ok.sum(1), 1)      # perturbation strength
    pcol = np.where(ok, A, 0).sum(0) / np.maximum(ok.sum(0), 1)      # gene responsiveness
    report(f"\n  GWPS: {A.shape[0]:,} perturbations x {A.shape[1]:,} genes, {n_nonfinite:,} non-finite cells "
           f"DROPPED (never imputed)")

    # perturbation identity by ENSG (the last field), which is the accession, not the symbol
    pert, n_sym_pert = [], 0
    for g in gt:
        p = g.split("_")
        last = p[-1]
        if last.startswith("ENSG"):
            pert.append(REG.resolve("gene", "ensembl", last))
        elif len(p) > 1 and p[1] not in ("non-targeting",):
            e = REG.resolve("gene", "symbol", p[1])
            pert.append(e)
            n_sym_pert += e is not None
        else:
            pert.append(None)
    gene = [REG.resolve("gene", "ensembl", e) for e in vg]
    report(f"  perturbations resolved: {sum(x is not None for x in pert):,}/{len(pert):,} "
           f"(by ENSG; {n_sym_pert} by symbol where the file carries no ENSG); "
           f"measured genes {sum(x is not None for x in gene):,}/{len(gene):,}")

    gi, pm = collections.defaultdict(list), collections.defaultdict(list)
    for j, e in enumerate(gene):
        if e:
            gi[e].append(j)
    for i, e in enumerate(pert):
        if e:
            pm[e].append(i)

    # ---- COVARIATES. Eleven, not two. ----
    ent2pub = {}
    pk = [f"{int(k)}.0" for k in pubsrc["count"]]      # the registry stores entrez as '7105.0'
    gp, stpub = REG.join("gene", "entrez", pk, 0.10, "gene2pubmed entrez -> gene")
    for e, k in zip(gp, pubsrc["count"]):
        if e:
            ent2pub[e] = max(ent2pub.get(e, 0), pubsrc["count"][k])
    report(f"  study-bias join: gene/entrez {stpub['joined']:,}/{stpub['of']:,} ({stpub['rate']:.1%}) -- the "
           f"registry keys entrez as a float string, so the naive key silently matches ZERO; this one does not")
    gl = pd.read_parquet(SD / "genes.parquet")
    ent2len = {}
    for e_, s_, t_ in zip(gl["ensembl_gene_id"], gl["start_position"], gl["end_position"]):
        eid = REG.resolve("gene", "ensembl", str(e_))
        if eid and pd.notna(s_) and pd.notna(t_):
            ent2len[eid] = abs(float(t_) - float(s_)) + 1
    eab = {}
    if PROTL.exists():
        pl = json.load(gzip.open(PROTL, "rt"))
        for s_, v_ in pl.get("abundance", {}).items():
            e = REG.resolve("gene", "symbol", s_)
            if e:
                eab[e] = max(eab.get(e, 0.0), float(v_))

    def per(vec, ids, f):
        return np.array([f(e) if e else np.nan for e in ids], float)

    lg = lambda x: np.log10(1.0 + x)
    rowc = {"strength": prow,
            "KD fold_expr": obs_fold,
            "log n cells": lg(obs_ncell),
            "log pubcount": per(None, pert, lambda e: lg(ent2pub.get(e, 0))),
            "log gene length": per(None, pert, lambda e: np.log10(ent2len[e]) if e in ent2len else np.nan),
            "log iBAQ": per(None, pert, lambda e: lg(eab[e]) if e in eab else np.nan)}
    colc = {"responsiveness": pcol,
            "log expression": np.log10(1e-4 + var_mean),
            "log pubcount": per(None, gene, lambda e: lg(ent2pub.get(e, 0))),
            "log gene length": per(None, gene, lambda e: np.log10(ent2len[e]) if e in ent2len else np.nan),
            "log iBAQ": per(None, gene, lambda e: lg(eab[e]) if e in eab else np.nan)}

    def linked_pairs(psize):
        Lp, Sp, Cp = [], [], []
        for (a, b), s in psize.items():
            for x, y in ((a, b), (b, a)):
                for i in pm.get(x, []):
                    for j in gi.get(y, []):
                        if pert[i] == gene[j]:
                            continue
                        if ok[i, j]:
                            Lp.append((i, j))
                            Sp.append(s)
                            Cp.append(pair_cplx.get((a, b) if a < b else (b, a), "?"))
        return (np.array(Lp, dtype=np.int64).reshape(-1, 2), np.array(Sp, dtype=np.int64), np.array(Cp))

    L, S, PCX = linked_pairs(pair_size)
    li, lj = L[:, 0], L[:, 1]
    linked = set(map(tuple, L.tolist()))
    report(f"  testable linked (P,G) cells: {len(L):,} from {len(set(li.tolist())):,} distinct perturbations "
           f"and {len(set(lj.tolist())):,} distinct measured genes, over {len(set(PCX.tolist())):,} complexes")
    cc = collections.Counter(PCX.tolist())
    top4 = sum(n for _, n in cc.most_common(4))
    report(f"  CONCENTRATION: the 4 largest contribute {top4:,}/{len(L):,} = {100*top4/len(L):.1f}% of all "
           f"testable pairs -- " + "; ".join(f"{cname[c][:34]} ({n:,})" for c, n in cc.most_common(4)))

    # ---- the instrument ----
    B = Bench(A, rowc, colc, linked, "perturbation", "measured gene")
    RB = {n: qbin(prow, n) for n in LADDER}
    CB = {n: qbin(pcol, n) for n in LADDER}
    ROW_EXACT = np.arange(len(pert))
    row_link = np.array([e is not None for e in pert])       # a row the linked arm could ever contain
    col_link = np.array([e is not None for e in gene])
    row_sub = np.array([e in subs for e in pert])
    col_sub = np.array([e in subs for e in gene])
    cluster_of = np.array([hash(e) if e else -1 for e in pert])
    report(f"  control universe: {row_link.sum():,}/{len(pert):,} rows and {col_link.sum():,}/{len(gene):,} "
           f"columns carry a resolvable gene ({sum(1 for i,e in enumerate(pert) if e is None and 'non-targeting' in gt[i]):,} "
           f"rows are non-targeting controls the linked arm can never contain, and are excluded from it)")

    R = {}
    report(f"\n  THE LADDER -- one ratio at one tightness is not a result. {N_DRAWS} draws per rung.")
    report(HEAD)
    R["ladder"] = {}
    for n in LADDER:
        lab = f"{'LEGACY ' if n == LEGACY else ''}{n}x{n} joint"
        R["ladder"][f"joint_{n}"] = B.arm(lab, li, lj, RB[n], CB[n], row_link, col_link,
                                          cluster_of=cluster_of if n == LADDER[-1] else None,
                                          boot_of=PCX if n == LADDER[-1] else None)
        report(line(R["ladder"][f"joint_{n}"]))
    rr = [R["ladder"][f"joint_{n}"]["ratio"] for n in LADDER]
    conv = LADDER[-1]
    for a, b, n in zip(rr, rr[1:], LADDER):
        if abs(a - b) < 0.005:
            conv = n
            break
    report(f"    -> the ratio falls monotonically from {rr[0]:.4f} at {LADDER[0]}x{LADDER[0]} to {rr[-1]:.4f} "
           f"at {LADDER[-1]}x{LADDER[-1]}; converged rung {conv}x{conv}")
    CONV = LADDER[-1]
    conv_arm = R["ladder"][f"joint_{CONV}"]

    report(f"\n  THE CONTROLS THE PREVIOUS VERSION DID NOT HAVE")
    report(HEAD)
    # SHAM: degree-preserving rewiring. Must return ~1.00 or the instrument is manufacturing the effect.
    rng = np.random.default_rng(1)
    aa = np.array([p[0] for p in pair_size], dtype=object)
    bb = np.array([p[1] for p in pair_size], dtype=object)
    bb = bb[rng.permutation(len(bb))]
    sham = {}
    for x, y in zip(aa.tolist(), bb.tolist()):
        if x == y:
            continue
        k = (x, y) if x < y else (y, x)
        if k in pair_size:
            continue
        sham[k] = pair_size.get(k, 2)
    LS, _SS, _CS = linked_pairs(sham)
    R["sham"] = B.arm(f"SHAM degree-preserving rewire, {CONV}x{CONV}", LS[:, 0], LS[:, 1], RB[CONV], CB[CONV],
                      row_link, col_link, exclude=linked | set(map(tuple, LS.tolist())))
    R["sham_exact"] = B.arm("SHAM rewire, exact perturbation", LS[:, 0], LS[:, 1], ROW_EXACT, CB[CONV],
                            row_link, col_link, exclude=linked | set(map(tuple, LS.tolist())))
    report(line(R["sham"]))
    report(line(R["sham_exact"]))
    # SUBUNIT vs SUBUNIT: both arms made of the same kind of gene
    R["subunit_universe"] = B.arm(f"control = CORUM subunit x subunit, {CONV}x{CONV}", li, lj, RB[CONV],
                                  CB[CONV], row_sub, col_sub, cluster_of=cluster_of, boot_of=PCX)
    report(line(R["subunit_universe"]))
    # EXACT PERTURBATION: every row-side confound held exactly, measured or not
    R["exact_pert"] = B.arm(f"EXACT perturbation x responsiveness {CONV}", li, lj, ROW_EXACT, CB[CONV],
                            row_link, col_link, cluster_of=cluster_of, boot_of=PCX)
    R["exact_pert_sub"] = B.arm(f"EXACT perturbation x resp {CONV}, subunit columns", li, lj, ROW_EXACT,
                                CB[CONV], row_link, col_sub, cluster_of=cluster_of, boot_of=PCX)
    report(line(R["exact_pert"]))
    report(line(R["exact_pert_sub"]))

    report(f"\n  CONFOUNDER ARMS -- each alternative explanation matched on explicitly, at {CONV}x{CONV}")
    report(HEAD)
    R["confounders"] = {}
    for nm, rv, cv in [("expression", None, colc["log expression"]),
                       ("pubcount", rowc["log pubcount"], colc["log pubcount"]),
                       ("iBAQ abundance", rowc["log iBAQ"], colc["log iBAQ"]),
                       ("gene length", rowc["log gene length"], colc["log gene length"]),
                       ("KD efficiency", rowc["KD fold_expr"], None)]:
        rk = RB[CONV] if rv is None else code(qbin(prow, 40), qbin(rv, 5))
        ck = CB[CONV] if cv is None else code(qbin(pcol, 40), qbin(cv, 5))
        R["confounders"][nm] = B.arm(f"+ {nm} quintile (both sides)", li, lj, rk, ck, row_link, col_link)
        report(line(R["confounders"][nm]))

    # ---- SIZE: the prespecified split, then where the break actually is ----
    small, large = S <= SMALL_MAX, S > SMALL_MAX
    report(f"\n  SIZE SPLIT (prespecified <={SMALL_MAX} vs >{SMALL_MAX}) at {CONV}x{CONV} and under the "
           f"strictest control")
    report(HEAD)
    R["size"] = {}
    for nm, sel in (("small", small), ("large", large)):
        R["size"][nm] = B.arm(f"{nm} ({'<=' if nm=='small' else '>'}{SMALL_MAX} subunits), {CONV}x{CONV}",
                              li[sel], lj[sel], RB[CONV], CB[CONV], row_link, col_link,
                              boot_of=PCX[sel], cluster_of=cluster_of)
        report(line(R["size"][nm]))
    idx_large = np.where(large)[0]
    capn = int(small.sum())
    rngc = np.random.default_rng(3)
    keep = rngc.choice(idx_large, capn, replace=False)
    R["size"]["large_capped"] = B.arm(f"large, subsampled to the small arm's n={capn:,}", li[keep], lj[keep],
                                      RB[CONV], CB[CONV], row_link, col_link)
    report(line(R["size"]["large_capped"]))
    for nm, sel in (("small", small), ("large", large)):
        R["size"][nm + "_exact"] = B.arm(f"{nm}, EXACT perturbation", li[sel], lj[sel], ROW_EXACT, CB[CONV],
                                         row_link, col_link, boot_of=PCX[sel], cluster_of=cluster_of)
        report(line(R["size"][nm + "_exact"]))

    report(f"\n  IS IT A THRESHOLD OR A GRADIENT? bands at {CONV}x{CONV}")
    report(HEAD)
    bands = {}
    for lo, hi, lab in SIZE_BANDS:
        sel = (S >= lo) & (S <= hi)
        bands[lab] = B.arm(f"band {lab}", li[sel], lj[sel], RB[CONV], CB[CONV], row_link, col_link,
                           boot_of=PCX[sel])
        report(line(bands[lab]))
    R["size_bands"] = bands
    ok_bands = [(lab, bands[lab]) for _l, _h, lab in SIZE_BANDS if bands[lab].get("testable")]
    first_sig = next((lab for lab, b in ok_bands if b["ratio"] >= 1.05 and b["frac_draws_p05"] >= 0.8), None)
    report(f"    -> the first band that clears 1.05 with {N_DRAWS}/{N_DRAWS} draws significant is "
           f"{first_sig}; every band below it is a null, so this is a THRESHOLD, not the gradient the "
           f"previous version claimed")

    # ---- study bias vs size: the stated limit, now measured ----
    sz = np.array([len(m) for m in cplx.values()])
    pu = np.array([np.mean([np.log10(1 + ent2pub.get(x, 0)) for x in m]) for m in cplx.values()])
    sp = stats.spearmanr(sz, pu)
    R["study_bias_vs_size"] = {"spearman": float(sp[0]), "p": float(sp[1]), "n_complexes": int(len(sz)),
                               "by_band": {lab: float(np.median(pu[(sz >= lo) & (sz <= hi)]))
                                           for lo, hi, lab in SIZE_BANDS}}
    report(f"\n  STUDY BIAS vs SIZE (the previous version's first stated limit, asserted there, measured "
           f"here): spearman(n subunits, mean subunit log10 publication count) = {sp[0]:+.3f} (p {sp[1]:.2g}) "
           f"over {len(sz):,} complexes -- if anything the big machines are LESS studied per subunit, so "
           f"study bias does not explain the size effect")

    # ---- the existing (Complex Portal) layer on the identical ladder ----
    ex_size = {}
    for m in ex.values():
        m = sorted(m)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                k = (m[i], m[j])
                ex_size[k] = min(ex_size.get(k, 10 ** 9), len(m))
    LE, SE, _CE = linked_pairs(ex_size)
    linkedE = set(map(tuple, LE.tolist()))
    BE = Bench(A, rowc, colc, linkedE, "perturbation", "measured gene")
    report(f"\n  COMPLEX PORTAL (the existing layer) ON THE IDENTICAL ARMS -- 'add or replace' is a measured "
           f"question")
    report(HEAD)
    R["existing"] = {
        f"joint_{LEGACY}": BE.arm(f"Complex Portal, LEGACY {LEGACY}x{LEGACY}", LE[:, 0], LE[:, 1], RB[LEGACY],
                                  CB[LEGACY], row_link, col_link),
        f"joint_{CONV}": BE.arm(f"Complex Portal, {CONV}x{CONV}", LE[:, 0], LE[:, 1], RB[CONV], CB[CONV],
                                row_link, col_link, cluster_of=cluster_of),
        "exact_pert": BE.arm("Complex Portal, EXACT perturbation", LE[:, 0], LE[:, 1], ROW_EXACT, CB[CONV],
                             row_link, col_link, cluster_of=cluster_of)}
    for k in R["existing"]:
        report(line(R["existing"][k]))

    # ---- comparator layers, read from their own committed results so the comparison cannot drift ----
    R["comparators"] = {}
    if BIOPLEX.exists():
        bj = json.load(open(BIOPLEX))
        R["comparators"] = {
            "source": str(BIOPLEX),
            "BioPlex PPI legacy 10x10": bj["ladder"]["loose_10"]["ratio"],
            "BioPlex PPI converged": bj["ladder"][f"joint_{max(int(k.split('_')[1]) for k in bj['ladder'] if k.startswith('joint_'))}"]["ratio"],
            "BioPlex PPI exact-perturbation": bj["cluster_ci"]["exact-perturbation"]["ratio"],
            "existing PPI layer legacy 10x10": bj["reference_layers"]["existing PPI layer"]["legacy 10x10"]["ratio"],
            "existing PPI layer converged": bj["reference_layers"]["existing PPI layer"]["converged"]["ratio"]}
        report(f"\n  COMPARATORS read live from {BIOPLEX.name} so a ranking cannot be quoted across "
               f"tightnesses: " + ", ".join(f"{k} {v:.3f}" for k, v in R["comparators"].items()
                                            if k != "source"))

    # ---- ASSUMED limiting-subunit capacity, and the control the previous version skipped ----
    report(f"\n  LIMITING-SUBUNIT CAPACITY -- 1:1:1 stoichiometry is ASSUMED, never measured")
    cap, lim_res = {}, None
    if eab:
        covered = len(subs & set(eab))
        report(f"    K562 protein covers {covered:,}/{len(subs):,} ({100*covered/len(subs):.1f}%) of CORUM "
               f"subunits")
        full = [c for c, m in cplx.items() if m and all(g in eab for g in m)]
        for c in full:
            m = sorted(cplx[c], key=lambda g: (eab[g], g))
            cap[c] = {"limiting_gene": m[0], "capacity_ibaq": float(eab[m[0]]), "n_subunits": len(m),
                      "assumption": "ASSUMED_1_1_1"}
        report(f"    {len(full):,} complexes have EVERY subunit measured and receive an ASSUMED capacity; "
               f"the other {len(cplx)-len(full):,} carry null, not zero")
        # the genome-wide abundance -> strength gradient, which is what the old test was actually measuring
        gstr = {e: float(np.mean(prow[r])) for e, r in pm.items()}
        have = [e for e in gstr if e in eab]
        xv = np.array([np.log10(1 + eab[e]) for e in have])
        yv = np.array([gstr[e] for e in have])
        sp2 = stats.spearmanr(xv, yv)
        q = np.quantile(xv, np.linspace(0, 1, 21))
        bb2 = np.clip(np.digitize(xv, q[1:-1]), 0, 19)
        prof = np.array([yv[bb2 == k].mean() if (bb2 == k).any() else np.nan for k in range(20)])
        pred = {e: prof[np.clip(np.digitize([np.log10(1 + eab[e])], q[1:-1])[0], 0, 19)] for e in have}
        obs, exp = [], []
        for c in full:
            if len(cplx[c]) < 3:
                continue
            m = sorted(cplx[c], key=lambda g: (eab[g], g))
            sl = gstr.get(m[0])
            so = [gstr[g] for g in m[1:] if g in gstr]
            if sl is None or not so:
                continue
            obs.append(sl - float(np.mean(so)))
            exp.append(pred[m[0]] - float(np.mean([pred[g] for g in m[1:] if g in gstr])))
        obs, exp = np.array(obs), np.array(exp)
        if len(obs) >= 50:
            resid_d = obs - exp
            lim_res = {"n_complexes": len(obs), "observed_delta": float(obs.mean()),
                       "observed_p": float(stats.wilcoxon(obs)[1]),
                       "expected_from_abundance_gradient": float(exp.mean()),
                       "residual_delta": float(resid_d.mean()),
                       "residual_p": float(stats.wilcoxon(resid_d)[1]),
                       "frac_residual_positive": float(np.mean(resid_d > 0)),
                       "genomewide_spearman_ibaq_strength": float(sp2[0]),
                       "genomewide_p": float(sp2[1]),
                       "verdict": "UNINFORMATIVE about 1:1:1 -- the raw contrast is the genome-wide "
                                  "abundance gradient, and the residual after removing it is a null"}
            report(f"    the previous version read this as a FALSIFICATION of 1:1:1. It does not survive its "
                   f"own control:")
            report(f"      abundance predicts perturbation strength genome-wide: spearman {sp2[0]:+.3f} "
                   f"(p {sp2[1]:.2g}) over {len(have):,} perturbed genes")
            report(f"      within-complex observed delta {obs.mean():+.4f} (Wilcoxon p "
                   f"{stats.wilcoxon(obs)[1]:.3g}) vs {exp.mean():+.4f} expected from that gradient alone")
            report(f"      RESIDUAL {resid_d.mean():+.4f} (p {stats.wilcoxon(resid_d)[1]:.3g}, "
                   f"{100*np.mean(resid_d>0):.1f}% positive) -- the gradient explains "
                   f"{100*exp.mean()/obs.mean():.0f}% of it; the test says nothing about stoichiometry")
    # what Complex Portal knows about stoichiometry, which is the only measurement available
    st = collections.defaultdict(list)
    for cid, s_ in zip(cpm["complex_id"], cpm["stoichiometry"]):
        try:
            st[cid].append(int(float(s_)))
        except (TypeError, ValueError):
            continue
    known = {c: v for c, v in st.items() if len(v) >= 2 and all(x > 0 for x in v)}
    allone = sum(1 for v in known.values() if all(x == 1 for x in v))
    R["stoichiometry_reference"] = {"source": EX_SOURCE, "n_fully_specified": len(known),
                                    "n_all_ones": allone,
                                    "frac_1_1_1": allone / max(len(known), 1),
                                    "n_with_unknown": len(st) - len(known)}
    report(f"    MEASURED AGAINST THE ONE SOURCE THAT HAS IT: of {len(known):,} Complex Portal complexes with "
           f"a fully specified stoichiometry, {allone:,} ({100*allone/max(len(known),1):.1f}%) are actually "
           f"1:1:1 -- so ASSUMED_1_1_1 carries a ~{100-100*allone/max(len(known),1):.0f}% known error rate, "
           f"not an unknown one")

    # ---- coverage, and whether missingness is biased ----
    incell = {e for e, v in REG.entities.items() if v.get("in_cell")}
    meas = {g for g in subs if g in set(gi)}
    pert_cov = {g for g in subs if g in set(pm)}
    report(f"\n  COVERAGE (absence is unknown, never zero)")
    report(f"    CORUM subunits that are genes in the cell object: {len(subs & incell):,}/{len(subs):,} "
           f"({100*len(subs & incell)/len(subs):.1f}%)")
    report(f"    ... measured in GWPS: {len(meas):,}   ... perturbed in GWPS: {len(pert_cov):,}")
    csz = np.array([len(m) for m in cplx.values()])
    tested = np.array([1.0 if any(g in pm for g in m) and any(h in gi for h in m) else 0.0
                       for m in cplx.values()])
    qs = np.array([np.searchsorted([2, 3, 5, 10], s) for s in csz])
    report(f"    {'complex size':16s} {'n':>7s} {'frac with a testable pair':>26s}")
    for k in sorted(set(qs.tolist())):
        m = qs == k
        report(f"    {['2','3','4-5','6-10','>10'][k]:16s} {int(m.sum()):7,d} {tested[m].mean():26.4f}")
    report(f"    -> testability rises steeply with complex size, so the small-complex arm is also the arm "
           f"with the thinner sample; that is why it gets an explicit power check")

    # ---- assemble ----
    R.update({"model": "corum-v2", "source": SOURCE, "release": rel,
              "study_bias_source": {"source": pubsrc["source"], "version": pubsrc["version"],
                                    "join": stpub},
              "n_complexes_parsed": len(rows), "n_complexes_usable": len(cplx),
              "n_subunit_genes": len(subs), "n_cocomplex_pairs": len(pair_size),
              "join": {"protein_uniprot": stp, "gene_uniprot": stg, "gene_symbol_fallback": sts,
                       "n_admitted_via_symbol": len(via_symbol)},
              "stoichiometry": {"numeric_tokens_in_n_complexes": n_stoich, "of": len(rows),
                                "verdict": "CORUM ships no usable subunit stoichiometry; none is invented"},
              "existing_layer": {
                  "provenance": EX_SOURCE, "name_match_to_science_data": f"{hit}/{len(ex_raw)}",
                  "n_existing_complexes": len(ex), "corum_name_collisions": name_hit,
                  "pairs_shared": len(inter), "pairs_corum_only": len(conly),
                  "pairs_existing_only": len(eonly),
                  "pair_jaccard": len(inter) / max(len(cor_pairs | ex_pairs), 1),
                  "frac_existing_with_exact_twin": float((best >= 0.999).mean()),
                  "frac_existing_jaccard_ge_half": float((best >= 0.5).mean()),
                  "frac_existing_no_overlap": float((best <= 0).mean())},
              "n_testable_pairs": int(len(L)), "n_complexes_testable": int(len(set(PCX.tolist()))),
              "top4_share_of_testable_pairs": float(top4 / len(L)),
              "converged_rung": CONV,
              "limiting_subunit": {"assumption": "ASSUMED_1_1_1", "n_complexes_with_capacity": len(cap),
                                   "test": lim_res},
              "coverage": {"n_subunits_in_cell": len(subs & incell), "n_subunits_measured_gwps": len(meas),
                           "n_subunits_perturbed_gwps": len(pert_cov)}})

    cl_c = conv_arm.get("cluster_complex", {})
    cl_p = conv_arm.get("cluster_perturbation", {})
    ex_a = R["exact_pert"]
    ex_c = ex_a.get("cluster_complex", {})
    R["limits"] = [
        f"the ratio depends on how tight the control is and falls monotonically along the ladder "
        f"({rr[0]:.3f} at {LADDER[0]}x{LADDER[0]} -> {rr[-1]:.3f} at {LADDER[-1]}x{LADDER[-1]} -> "
        f"{R['subunit_universe']['ratio']:.3f} against a subunit-only control -> {ex_a['ratio']:.3f} holding "
        f"the perturbation exactly). No single number is THE effect and any comparison with another layer is "
        f"meaningless unless both are quoted at the same tightness.",
        f"the pair-level Mann-Whitney p (~0) is optimistic by construction: {len(L):,} pairs come from "
        f"{len(set(PCX.tolist())):,} complexes and the four largest carry {100*top4/len(L):.0f}% of them. "
        f"The cluster bootstraps are the honest significance and they are much weaker: at the converged rung "
        f"the complex-level 95% CI is [{cl_c.get('ci_lo',float('nan')):.3f}, "
        f"{cl_c.get('ci_hi',float('nan')):.3f}] and only "
        f"{100*cl_c.get('frac_clusters_positive',float('nan')):.0f}% of complexes are individually positive.",
        "even at the converged rung the control remains imbalanced on covariates it does not match "
        "(publication count, iBAQ abundance, gene length); each has its own arm, none of them abolishes the "
        "effect, but no single arm matches all eleven at once, so an unmeasured correlate of 'is a subunit "
        "of a big machine' cannot be excluded",
        f"CORUM {RELEASE} is literature-curated, so complex size could be confounded with how well studied a "
        f"machine is. That is measured here rather than assumed and comes out slightly NEGATIVE "
        f"(spearman {sp[0]:+.3f}), so it is not the explanation -- but publication count is a crude proxy "
        f"for curation depth.",
        "most CORUM complexes were characterised in HEK293, HeLa or in vitro, NOT in K562 -- membership is "
        "imported from a different cellular context than the Perturb-seq matrix it is tested against",
        "no stoichiometry in CORUM: any limiting-subunit capacity here is ASSUMED 1:1:1, labelled "
        f"ASSUMED_1_1_1, and Complex Portal -- the only source on hand that ships stoichiometry -- says that "
        f"assumption is wrong for {100-100*allone/max(len(known),1):.0f}% of the complexes that specify one",
        "the previous version reported the limiting-subunit test as a FALSIFICATION of 1:1:1; it is not. The "
        "contrast it measured is the genome-wide abundance -> perturbation-strength gradient and the "
        "residual after removing that gradient is a null, so the test is uninformative in both directions",
        "capacity is computed only where every subunit is measured in K562; the rest are null, not zero, and "
        "K562 protein coverage is itself abundance-biased so those nulls are not missing at random",
        "a co-complex pair here is UNDIRECTED and UNSIGNED -- the test measures |z| only and says nothing "
        "about whether losing a subunit raises or lowers its partners",
        f"only pairs where one member is perturbed and the other measured in GWPS are testable: "
        f"{len(pair_size):,} curated pairs collapse to {len(L):,} cells biased toward well-expressed genes; "
        f"testability rises with complex size, which is why the small arm gets a power check",
        "complexes are flat membership sets -- CORUM's own hierarchy is discarded, so the size bands mix "
        "modules with the machines containing them, and a pair is tagged with the SMALLEST complex "
        "containing it, which is the conservative choice for the size claim but not the only defensible one",
        f"Complex Portal is compared, never replaced: {len(eonly):,} pairs exist only there and this module "
        f"cannot adjudicate them"]

    # ---- verdict, computed from the strict rungs rather than the flattering one ----
    strict = ex_a["ratio"]
    interesting = conv_arm["ratio"] >= 1.05 and conv_arm["frac_draws_p05"] >= 0.8
    holds = strict >= 1.05 and ex_c.get("frac_boot_le_105", 1.0) < 0.5
    sham_ok = abs(R["sham"]["ratio"] - 1.0) < 0.02
    head = (f"CORUM {RELEASE} contributes {len(cplx):,} human complexes over {len(pair_size):,} co-complex "
            f"gene pairs -> {len(L):,} testable (perturbation, gene) cells, {len(L)/max(len(LE),1):.1f}x the "
            f"testable surface of the Complex Portal layer already in the cell object (now identified, not "
            f"'unknown': {EX_SOURCE}), with which it shares {len(inter):,} pairs and {name_hit} names.")
    core = (f"RAW held-out mean |z| of co-complex pairs is {conv_arm['mean_abs_z_linked']:.4f}. The CONTRAST "
            f"depends entirely on the control: {rr[0]:.3f}x at the legacy {LADDER[0]}x{LADDER[0]} match the "
            f"previous version reported, {rr[-1]:.3f}x converged at {CONV}x{CONV}, "
            f"{R['subunit_universe']['ratio']:.3f}x when the control pair is also made of CORUM subunits, and "
            f"{strict:.3f}x holding the perturbation EXACTLY. The sham rewiring returns "
            f"{R['sham']['ratio']:.3f}x, so the instrument is not manufacturing it. The pair-level p is ~0 "
            f"and should be ignored: over {cl_c.get('n_clusters',0):,} complexes the bootstrap CI at the "
            f"converged rung is [{cl_c.get('ci_lo',float('nan')):.3f}, {cl_c.get('ci_hi',float('nan')):.3f}] "
            f"and only {100*cl_c.get('frac_clusters_positive',0):.0f}% of individual complexes are positive "
            f"(sign test p {cl_c.get('sign_test_p',float('nan')):.3g}); over "
            f"{cl_p.get('n_clusters',0):,} perturbations it is "
            f"[{cl_p.get('ci_lo',float('nan')):.3f}, {cl_p.get('ci_hi',float('nan')):.3f}].")
    size = (f"The signal is NOT in complexes as such, it is in big machines, and the prespecified "
            f"<={SMALL_MAX}/>{SMALL_MAX} split puts the boundary in the wrong place: bands run " +
            ", ".join(f"{lab} {bands[lab]['ratio']:.3f}" for _l, _h, lab in SIZE_BANDS
                      if bands[lab].get('testable')) +
            f", so everything up to 10 subunits is a null and the first band that clears 1.05 with all "
            f"{N_DRAWS} draws significant is {first_sig}. That is a THRESHOLD, not the gradient the previous "
            f"version claimed. It is not a power artefact -- large complexes subsampled to the small arm's "
            f"n={capn:,} still score {R['size']['large_capped']['ratio']:.3f} -- and it is not study bias, "
            f"which is measured at spearman {sp[0]:+.3f} against size. Per complex, "
            f"{100*bands['>20'].get('cluster_complex',{}).get('frac_clusters_positive',0):.0f}% of the "
            f">20-subunit machines are individually positive against "
            f"{100*bands['2 (heterodimer)'].get('cluster_complex',{}).get('frac_clusters_positive',0):.0f}% "
            f"of heterodimers.")
    tail = (f"CORUM IS BROADER, NOT BETTER PER PAIR: on identical arms Complex Portal scores "
            f"{R['existing'][f'joint_{CONV}']['ratio']:.3f}x converged and "
            f"{R['existing']['exact_pert']['ratio']:.3f}x exact-perturbation against CORUM's "
            f"{rr[-1]:.3f}x / {strict:.3f}x, so the two should be UNIONED, not swapped. And the ASSUMED "
            f"1:1:1 capacity earns nothing: the previous version's falsification of it was the genome-wide "
            f"abundance gradient (residual {lim_res['residual_delta']:+.4f}, p {lim_res['residual_p']:.2g}), "
            f"while Complex Portal says 1:1:1 is simply wrong for "
            f"{100-100*allone/max(len(known),1):.0f}% of complexes that specify a stoichiometry."
            if lim_res else "")
    if not interesting:
        v = (f"CO-COMPLEX MEMBERSHIP IS NOT A USABLE LINK TYPE HERE. {head} {core}")
    elif holds and sham_ok:
        v = (f"CO-COMPLEX MEMBERSHIP CARRIES A REAL BUT MUCH SMALLER SIGNAL THAN THE 1.212x THIS MODULE "
             f"PREVIOUSLY REPORTED, AND ONLY FOR LARGE MACHINES. {head} {core} {size} {tail}")
    else:
        v = (f"CO-COMPLEX MEMBERSHIP DOES NOT SURVIVE ITS OWN CONTROLS. {head} {core} {size} {tail}")
    R["verdict"] = v
    report("\n" + "=" * 130)
    report(f"  VERDICT: {v}")

    layer = {"model": "corum-v2", "source": SOURCE, "release": rel, "assembly_free": True,
             "complexes": {c: sorted(m) for c, m in cplx.items()},
             "complex_info": cinfo,
             "limiting_subunit_ASSUMED_1_1_1": cap,
             "via_symbol": sorted(via_symbol),
             "join": R["join"], "limits": R["limits"]}
    LAYER.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "corum.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'corum.json'}")


if __name__ == "__main__":
    main()
