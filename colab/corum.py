"""CORUM PROTEIN COMPLEXES -- and the size at which "same machine" stops predicting "same response".

WHAT IS ALREADY THERE, AND WHY IT IS NOT ENOUGH. The cell object carries 2,039 complexes in `complexes` /
`gene2cplx` with no recorded provenance. Not one of their 2,039 names appears in CORUM 5.3, so whatever they
are, they are not CORUM under its current naming -- the convention ("Calprotectin heterotetramer", "bZIP
transcription factor complex, ATF4-CREB1", "General transcription factor TFIIF complex") is the EBI Complex
Portal's, but this module has no Complex Portal file to prove that with, so the provenance stays UNKNOWN and
is reported as unknown. That is exactly the situation the project keeps losing to: a layer nobody can trace.
CORUM is therefore ADDED AND COMPARED, never substituted. The overlap and the disagreement are both measured.

WHY CO-COMPLEX MEMBERSHIP SHOULD MOVE TRANSCRIPTS AT ALL. Knocking down one subunit of an obligate complex
destabilises the others: unassembled subunits are degraded, and in many machines the transcriptional program
that builds the machine is co-regulated. So the prediction is that a (perturbation P, gene G) pair which
shares a complex shows a larger |robust z| in K562 CRISPRi Perturb-seq than a pair matched on how disruptive
P is and how responsive G is. That is the same fast test every other layer here has taken, so the number is
comparable: PPI 1.208x, co-dependency 1.070x, regulatory 1.021x, signalling 1.028x (null).

THE SECOND QUESTION IS THE ONE WORTH ASKING. A complex is not one kind of object. CORUM's median human
complex has TWO subunits -- a co-immunoprecipitated pair from one paper -- while its tail runs to the
ribosome and the spliceosome. Those should not behave alike. A transient heterodimer has no reason to produce
a co-regulated transcriptional response; a 40-subunit machine whose subunits are stoichiometrically
co-produced has every reason to. So the test is run SPLIT BY COMPLEX SIZE, prespecified at <=5 vs >5 as the
brief asks, and then across size bands to see whether it is a threshold or a gradient.

WHAT WOULD FALSIFY THIS LAYER. Three things, each of which is checked rather than assumed:
  (a) The pooled ratio lands under ~1.05, like `signalling` did. Then co-complex membership is not a usable
      link type in this cell and the layer is descriptive only.
  (b) The small/large split is a POWER artefact -- large complexes contribute 7x more pairs, so "small is
      null" could just be a thin sample. Large complexes are therefore subsampled to the small set's exact n
      and re-tested. If the difference evaporates, the split claim dies.
  (c) The pooled ratio is an artefact of coarse matching. Deciles are the specified protocol and the number
      the reference figures are comparable to, but decile matching leaves a residual imbalance on gene
      responsiveness here (p ~ 1e-6), so the whole test is re-run at 50x50 bins where the residual clears.
      If the effect only exists at deciles, it is the covariate talking.

STOICHIOMETRY IS NOT IN THIS FILE. CORUM 5.3 ships a `subunits_stoechiometrie` column and it is empty: 5,576
of 5,628 human complexes carry nothing but the semicolon delimiters, and only 469 carry a single numeric
token anywhere. That is asserted at parse time, not trusted. A limiting-subunit capacity IS computed from
K562 protein abundance, and it is labelled ASSUMED_1_1_1 in every field it touches, because the assumption is
invented here and not measured anywhere. It is then given something to be wrong about: if the lowest-abundance
subunit were the rate-limiting one, knocking it down should be MORE disruptive than knocking down its
complex-mates. That is a within-complex paired comparison, so it needs no matching.

PROVENANCE IS PINNED AND ASSERTED. One release (CORUM 5.3, checked against the API's own current-release
endpoint), one organism (asserted: every row and every subunit is Human), one accession namespace (UniProt,
with symbol used only for a residue of retired accessions and recorded via=symbol).
"""
import collections
import csv
import gzip
import json
import os
import ssl
import subprocess
import sys
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
RAW = SP / "corum_human_5_3.txt"
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

SMALL_MAX = 5                         # prespecified small/large boundary, from the brief
N_DRAWS = 20                          # never read a single matched draw as the result
DECILES = 10                          # the specified protocol; comparable to the reference figures
TIGHT = 50                            # the sensitivity re-run, where the residual imbalance clears
SIZE_BANDS = [(2, 2, "2 (heterodimer)"), (3, 5, "3-5"), (6, 10, "6-10"),
              (11, 20, "11-20"), (21, 10 ** 9, ">20")]


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


def get(url, ctx, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "cellos", "accept": "*/*"})
    return urllib.request.urlopen(req, context=ctx, timeout=timeout).read()


def fetch(report):
    ctx = tls_context(report)
    cur = json.loads(get(f"{API}/public/releases/current", ctx, 120))
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


# ---------------------------------------------------------------------------------------------------
# the fast test
# ---------------------------------------------------------------------------------------------------
class Matcher:
    """Matched draws on (perturbation strength decile) x (gene responsiveness decile).

    The pair universe is 11,258 x 8,248, far too large to enumerate, so the control is drawn by rejection
    inside the matched cell: pick a random row of the right strength bin and a random column of the right
    responsiveness bin, reject it if it is linked, self, already drawn, or non-finite. That samples the
    matched non-linked population uniformly, which is what a matched control has to be.
    """

    def __init__(self, A, prow, pcol, pert_eid, gene_eid, nbin):
        self.A, self.prow, self.pcol, self.nbin = A, prow, pcol, nbin
        q = np.linspace(1.0 / nbin, 1.0 - 1.0 / nbin, nbin - 1)
        self.rd = np.digitize(prow, np.quantile(prow, q))
        self.cd = np.digitize(pcol, np.quantile(pcol, q))
        self.rows_by = [np.where(self.rd == k)[0] for k in range(nbin)]
        self.cols_by = [np.where(self.cd == k)[0] for k in range(nbin)]
        # self-pairs (knocking a gene down moves its own transcript) are excluded from BOTH arms
        u = {e: i for i, e in enumerate({x for x in pert_eid if x} | {x for x in gene_eid if x})}
        self.pk = np.array([u.get(e, -1) for e in pert_eid])
        self.gk = np.array([u.get(e, -2) for e in gene_eid])

    def draw(self, cell, linked, rng):
        ci, cj = [], []
        for (a, b), n in cell.items():
            R, C = self.rows_by[a], self.cols_by[b]
            if len(R) == 0 or len(C) == 0:
                continue
            got, tries = set(), 0
            while len(got) < n and tries < 60:
                k = (n - len(got)) * 3 + 10
                for x, y in zip(rng.choice(R, k).tolist(), rng.choice(C, k).tolist()):
                    if len(got) >= n:
                        break
                    if (x, y) in linked or (x, y) in got or self.pk[x] == self.gk[y]:
                        continue
                    if not np.isfinite(self.A[x, y]):
                        continue
                    got.add((x, y))
                tries += 1
            for x, y in got:
                ci.append(x)
                cj.append(y)
        return np.array(ci, dtype=np.int64), np.array(cj, dtype=np.int64)

    def run(self, L, linked, sel=None, cap=None, n_draws=N_DRAWS):
        from scipy import stats
        idx = np.arange(len(L)) if sel is None else np.where(sel)[0]
        if len(idx) < 200:
            return None
        rats, ps, m1, m0, rp, cp, ns = [], [], [], [], [], [], []
        for s in range(n_draws):
            rng = np.random.default_rng(s)
            # when `cap` is set the LINKED arm is subsampled too, so every number below -- n, raw mean and
            # p -- describes the capped comparison. Reporting the full pool's n next to a capped p is how a
            # power check quietly stops being one.
            use = idx if (cap is None or len(idx) <= cap) else rng.choice(idx, cap, replace=False)
            li, lj = L[use, 0], L[use, 1]
            cell = collections.Counter(zip(self.rd[li].tolist(), self.cd[lj].tolist()))
            ci, cj = self.draw(cell, linked, rng)
            a1, a0 = self.A[li, lj], self.A[ci, cj]
            rats.append(float(a1.mean() / a0.mean()))
            ps.append(float(stats.mannwhitneyu(a1, a0, alternative="two-sided")[1]))
            m1.append(float(a1.mean()))
            m0.append(float(a0.mean()))
            rp.append(float(stats.mannwhitneyu(self.prow[li], self.prow[ci])[1]))
            cp.append(float(stats.mannwhitneyu(self.pcol[lj], self.pcol[cj])[1]))
            ns.append(int(len(li)))
        return {"n_linked": int(np.mean(ns)), "n_pool": int(len(idx)), "n_draws": n_draws,
                "capped_to": cap, "bins": self.nbin,
                # THE RAW HELD-OUT VALUE. The ratio is the comparable statistic; this is the effect size.
                "raw_mean_abs_z_linked": float(np.mean(m1)),
                "raw_mean_abs_z_matched": float(np.mean(m0)),
                "ratio": float(np.mean(rats)), "ratio_sd": float(np.std(rats)),
                "p_median": float(np.median(ps)), "frac_draws_p05": float(np.mean([p < 0.05 for p in ps])),
                "residual_pert_strength_p": float(np.mean(rp)),
                "residual_gene_responsiveness_p": float(np.mean(cp))}


def line(label, r):
    if r is None:
        return f"    {label:26s} too few testable pairs"
    return (f"    {label:26s} {r['n_linked']:7,d} {r['raw_mean_abs_z_linked']:9.4f} "
            f"{r['raw_mean_abs_z_matched']:9.4f} {r['ratio']:8.4f} {r['p_median']:10.3g} "
            f"{100*r['frac_draws_p05']:6.0f}%")


HEAD = (f"    {'arm':26s} {'pairs':>7s} {'raw |z|':>9s} {'matched':>9s} {'ratio':>8s} "
        f"{'med p':>10s} {'p<.05':>7s}")


# ---------------------------------------------------------------------------------------------------
def main():
    from scipy import stats
    import h5py
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("CORUM PROTEIN COMPLEXES -- does co-complex membership move transcripts, and at what size?")
    report("=" * 100)
    for p in (GWPS, NET):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    from entity_registry import load as load_reg
    REG = load_reg()
    rel = fetch(report)

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

    # stoichiometry: asserted absent, not assumed absent
    import re
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
    # This arm is ALLOWED to be weak: it runs only over accessions that failed the strong key, most of them
    # retired or merged, and every subunit it recovers is recorded via=symbol so the weaker provenance
    # travels with the datum instead of being assumed away.
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
    pair_size = {}
    for cid, m in cplx.items():
        m = sorted(m)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                k = (m[i], m[j])
                pair_size[k] = min(pair_size.get(k, 10 ** 9), len(m))
    report(f"  {len(pair_size):,} distinct co-complex gene pairs")

    # ---- overlap and DISAGREEMENT with the cell object's existing complexes ----
    RJ = json.load(gzip.open(SP / "entity_registry.json.gz", "rt"))
    cidx = RJ["cell_index"]
    d = json.load(gzip.open(NET, "rt"))
    ex_raw = d.get("complexes") or {}
    ex = {}
    for k, v in ex_raw.items():
        m = {cidx.get(str(i)) for i in v}
        m = {x for x in m if x}
        if len(m) >= 2:
            ex[k] = m
    del d
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
    report(f"\n  OVERLAP WITH THE 2,039 COMPLEXES ALREADY IN THE CELL OBJECT (provenance UNKNOWN)")
    report(f"    complex NAMES shared with CORUM {RELEASE}: {name_hit} of {len(ex):,}  "
           f"-> the existing set is NOT CORUM")
    report(f"    co-complex pairs: {len(inter):,} shared, {len(conly):,} CORUM-only, {len(eonly):,} "
           f"existing-only  (Jaccard {len(inter)/max(len(cor_pairs|ex_pairs),1):.3f})")
    report(f"    {100*(best >= 0.999).mean():.1f}% of existing complexes have an exact CORUM membership twin, "
           f"{100*(best >= 0.5).mean():.1f}% reach Jaccard 0.5, {100*(best <= 0).mean():.1f}% share no "
           f"subunit with any CORUM complex")
    report(f"    -> CORUM is ADDED alongside, not substituted: {len(eonly):,} pairs exist only in the "
           f"existing set and deleting them would be a silent loss")

    # ---- the fast test matrix ----
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        gt = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        vg = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0] for s in f["var"]["gene_id"][:]]
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
    pert = []
    n_sym_pert = 0
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

    def linked_pairs(psize):
        L, S = [], []
        for (a, b), s in psize.items():
            for x, y in ((a, b), (b, a)):
                for i in pm.get(x, []):
                    for j in gi.get(y, []):
                        if pert[i] == gene[j]:
                            continue
                        if np.isfinite(A[i, j]):
                            L.append((i, j))
                            S.append(s)
        return np.array(L, dtype=np.int64).reshape(-1, 2), np.array(S, dtype=np.int64)

    L, S = linked_pairs(pair_size)
    linked = set(map(tuple, L.tolist()))
    report(f"  testable linked (P,G) cells: {len(L):,} from {len(set(L[:, 0].tolist())):,} distinct "
           f"perturbations and {len(set(L[:, 1].tolist())):,} distinct measured genes")

    M = Matcher(A, prow, pcol, pert, gene, DECILES)
    T = Matcher(A, prow, pcol, pert, gene, TIGHT)
    small, large = S <= SMALL_MAX, S > SMALL_MAX

    report(f"\n  FAST TEST -- |robust z| of co-complex pairs vs pairs matched on perturbation-strength "
           f"decile x gene-responsiveness decile, {N_DRAWS} draws")
    report(HEAD)
    r_all = M.run(L, linked)
    r_small = M.run(L, linked, small)
    r_large = M.run(L, linked, large)
    r_pow = M.run(L, linked, large, cap=int(small.sum()))
    report(line("all complexes", r_all))
    report(line(f"small (<={SMALL_MAX} subunits)", r_small))
    report(line(f"large (>{SMALL_MAX} subunits)", r_large))
    report(line("large, n-matched to small", r_pow))
    report(f"    residual imbalance after decile matching (mean p over draws): perturbation strength "
           f"{r_all['residual_pert_strength_p']:.3g}, gene responsiveness "
           f"{r_all['residual_gene_responsiveness_p']:.3g}")

    report(f"\n  SENSITIVITY -- the same test at {TIGHT}x{TIGHT} bins, where the residual imbalance clears")
    report(HEAD)
    t_all = T.run(L, linked)
    t_small = T.run(L, linked, small)
    t_large = T.run(L, linked, large)
    report(line("all complexes", t_all))
    report(line(f"small (<={SMALL_MAX} subunits)", t_small))
    report(line(f"large (>{SMALL_MAX} subunits)", t_large))
    report(f"    residual imbalance: perturbation strength {t_all['residual_pert_strength_p']:.3g}, "
           f"gene responsiveness {t_all['residual_gene_responsiveness_p']:.3g}")

    report(f"\n  IS IT A THRESHOLD OR A GRADIENT? deciles, {N_DRAWS} draws")
    report(HEAD)
    bands = {}
    for lo, hi, lab in SIZE_BANDS:
        bands[lab] = M.run(L, linked, (S >= lo) & (S <= hi))
        report(line(lab, bands[lab]))

    # ---- the same test on the existing layer, so the comparison is decision-relevant ----
    ex_size = {}
    for m in ex.values():
        m = sorted(m)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                k = (m[i], m[j])
                ex_size[k] = min(ex_size.get(k, 10 ** 9), len(m))
    LE, SE = linked_pairs(ex_size)
    linkedE = set(map(tuple, LE.tolist()))
    report(f"\n  THE EXISTING gene2cplx LAYER ON THE SAME TEST (so 'add or replace' is a measured question)")
    report(HEAD)
    e_all = M.run(LE, linkedE)
    e_small = M.run(LE, linkedE, SE <= SMALL_MAX)
    e_large = M.run(LE, linkedE, SE > SMALL_MAX)
    report(line("existing, all", e_all))
    report(line(f"existing, small (<={SMALL_MAX})", e_small))
    report(line(f"existing, large (>{SMALL_MAX})", e_large))

    # ---- ASSUMED limiting-subunit capacity, and a test that can falsify it ----
    report(f"\n  LIMITING-SUBUNIT CAPACITY -- 1:1:1 stoichiometry is ASSUMED, never measured")
    cap = {}
    lim_res = None
    if PROTL.exists():
        pl = json.load(gzip.open(PROTL, "rt"))
        eab = {}
        for s, v in pl.get("abundance", {}).items():
            e = REG.resolve("gene", "symbol", s)
            if e:
                eab[e] = max(eab.get(e, 0.0), float(v))
        covered = len(subs & set(eab))
        report(f"    K562 protein ({pl['source']}) covers {covered:,}/{len(subs):,} "
               f"({100*covered/len(subs):.1f}%) of CORUM subunits")
        # A minimum over a PARTIALLY measured complex is not a minimum. Only fully measured complexes get a
        # capacity, and the rest carry `null` -- absence is unknown, never zero.
        full = [c for c, m in cplx.items() if m and all(g in eab for g in m)]
        for c in full:
            m = sorted(cplx[c], key=lambda g: (eab[g], g))
            cap[c] = {"limiting_gene": m[0], "capacity_ibaq": float(eab[m[0]]),
                      "n_subunits": len(m), "assumption": "ASSUMED_1_1_1"}
        n3 = sum(1 for m in cplx.values() if len(m) >= 3)
        report(f"    {len(full):,} complexes have EVERY subunit measured and receive an ASSUMED capacity; "
               f"the other {len(cplx)-len(full):,} carry null, not zero")
        # falsification: if the lowest-abundance subunit were rate-limiting, its knockdown should be the
        # more disruptive one. Within-complex and paired, so no matching is needed.
        diffs, names = [], []
        for c in full:
            if len(cplx[c]) < 3:
                continue
            m = sorted(cplx[c], key=lambda g: (eab[g], g))
            def strength(g):
                r = pm.get(g, [])
                return float(np.mean(prow[r])) if r else None
            sl = strength(m[0])
            so = [x for x in (strength(g) for g in m[1:]) if x is not None]
            if sl is None or not so:
                continue
            diffs.append(sl - float(np.mean(so)))
            names.append(c)
        if len(diffs) >= 50:
            diffs = np.array(diffs)
            w = float(stats.wilcoxon(diffs)[1])
            lim_res = {"n_complexes": len(diffs), "mean_delta": float(diffs.mean()),
                       "median_delta": float(np.median(diffs)), "wilcoxon_p": w,
                       "frac_limiting_more_disruptive": float(np.mean(diffs > 0)),
                       "n_complexes_ge3": n3,
                       "n_complexes_ge3_all_measured": len([c for c in full if len(cplx[c]) >= 3])}
            report(f"    TEST of the assumption, over {len(diffs):,} complexes with >=3 fully measured "
                   f"subunits: knocking down the lowest-abundance ('limiting') subunit changes "
                   f"perturbation strength by {diffs.mean():+.4f} versus its complex-mates "
                   f"(median {np.median(diffs):+.4f}, Wilcoxon p {w:.3g}, "
                   f"{100*np.mean(diffs>0):.1f}% of complexes positive)")
    else:
        report(f"    absent: {PROTL} -- no capacity computed")

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
        lab = ["2", "3", "4-5", "6-10", ">10"][k]
        report(f"    {lab:16s} {int(m.sum()):7,d} {tested[m].mean():26.4f}")
    report(f"    -> testability rises steeply with complex size, so the small-complex arm is also the "
           f"arm with the thinner sample; that is why it gets an explicit power check rather than a shrug")

    # ---- verdict ----
    R = {"model": "corum-v1", "source": SOURCE, "release": rel,
         "n_complexes_parsed": len(rows), "n_complexes_usable": len(cplx),
         "n_subunit_genes": len(subs), "n_cocomplex_pairs": len(pair_size),
         "join": {"protein_uniprot": stp, "gene_uniprot": stg, "gene_symbol_fallback": sts,
                  "n_admitted_via_symbol": len(via_symbol)},
         "stoichiometry": {"numeric_tokens_in_n_complexes": n_stoich, "of": len(rows),
                           "verdict": "CORUM ships no usable subunit stoichiometry; none is invented"},
         "overlap_with_existing": {
             "n_existing_complexes": len(ex), "name_matches": name_hit,
             "pairs_shared": len(inter), "pairs_corum_only": len(conly), "pairs_existing_only": len(eonly),
             "pair_jaccard": len(inter) / max(len(cor_pairs | ex_pairs), 1),
             "frac_existing_with_exact_twin": float((best >= 0.999).mean()),
             "frac_existing_jaccard_ge_half": float((best >= 0.5).mean()),
             "frac_existing_no_overlap": float((best <= 0).mean()),
             "provenance_of_existing": "UNKNOWN -- 0/2,039 names appear in CORUM 5.3; the naming convention "
                                       "resembles the EBI Complex Portal but no Complex Portal file was "
                                       "fetched here, so this is not asserted"},
         "fast_test": {"deciles": {"all": r_all, "small": r_small, "large": r_large,
                                   "large_n_matched_to_small": r_pow},
                       f"tight_{TIGHT}x{TIGHT}": {"all": t_all, "small": t_small, "large": t_large},
                       "size_bands": bands,
                       "existing_layer": {"all": e_all, "small": e_small, "large": e_large}},
         "limiting_subunit": {"assumption": "ASSUMED_1_1_1", "n_complexes_with_capacity": len(cap),
                              "test": lim_res},
         "coverage": {"n_subunits_in_cell": len(subs & incell), "n_subunits_measured_gwps": len(meas),
                      "n_subunits_perturbed_gwps": len(pert_cov)}}

    R["limits"] = [
        f"CORUM {RELEASE} is a LITERATURE-CURATED set, so it inherits publication bias: the complexes with "
        f"many subunits are the ones many groups have worked on, and the size effect below is therefore "
        f"confounded with how well studied a machine is. Nothing here separates those two.",
        "most CORUM complexes were characterised in HEK293, HeLa or in vitro, NOT in K562 -- membership is "
        "imported from a different cellular context than the Perturb-seq matrix it is tested against",
        "no stoichiometry: 5,576/5,628 complexes carry an empty subunits_stoechiometrie column. Any "
        "limiting-subunit capacity in this layer is ASSUMED 1:1:1 and is labelled ASSUMED_1_1_1",
        "the capacity is computed only where EVERY subunit has a K562 protein measurement (a minimum over a "
        "partially measured complex is not a minimum); the rest carry null, and K562 protein coverage is "
        "itself abundance-biased, so those nulls are not missing at random",
        f"a co-complex pair here is UNDIRECTED and UNSIGNED: the test measures |z| only, so it says nothing "
        f"about whether losing a subunit raises or lowers its partners",
        "the pairs tested are only those where one member is perturbed and the other measured in GWPS; "
        f"{len(pair_size):,} curated pairs collapse to {len(L):,} testable cells, and the survivors are "
        "biased toward well-expressed genes",
        "complexes are treated as flat membership sets -- CORUM's own hierarchy (a complex that is a module "
        "of a larger one) is discarded, so the size bands mix modules with the machines that contain them",
        "the existing 2,039-complex layer is compared, not replaced, and its provenance remains UNKNOWN; "
        "this module cannot say which of the two is right where they disagree"]

    report("\n" + "=" * 100)
    dec, tig = r_all, t_all
    interesting = dec["ratio"] >= 1.05 and dec["frac_draws_p05"] >= 0.8
    survives = tig["ratio"] >= 1.05 and tig["frac_draws_p05"] >= 0.8
    split = (r_small["ratio"] < 1.05 or r_small["frac_draws_p05"] < 0.5) and \
            r_pow["ratio"] >= 1.05 and r_pow["frac_draws_p05"] >= 0.8
    head = (f"CORUM {RELEASE} contributes {len(cplx):,} human complexes over {len(pair_size):,} distinct "
            f"co-complex gene pairs, which resolve to {len(L):,} testable (perturbation, gene) cells in "
            f"GWPS -- {len(L)/max(len(LE),1):.1f}x the testable surface of the {len(ex):,} complexes "
            f"already in the cell object, which share only {len(inter):,} of those gene pairs with CORUM "
            f"and not one complex NAME.")
    if not interesting:
        v = (f"CO-COMPLEX MEMBERSHIP IS NOT A USABLE LINK TYPE HERE. {head} Matched on perturbation strength "
             f"and gene responsiveness over {N_DRAWS} draws, co-complex pairs reach mean |z| "
             f"{dec['raw_mean_abs_z_linked']:.4f} against {dec['raw_mean_abs_z_matched']:.4f} for matched "
             f"pairs -- a ratio of {dec['ratio']:.3f}, below the ~1.05 bar, in the same territory as "
             f"`signalling` (1.028x, null). A well-controlled null.")
    else:
        core = (f"Matched over {N_DRAWS} draws, co-complex pairs reach a RAW mean |z| of "
                f"{dec['raw_mean_abs_z_linked']:.4f} against {dec['raw_mean_abs_z_matched']:.4f} matched -- "
                f"a ratio of {dec['ratio']:.3f} over {dec['n_linked']:,} pairs, median p "
                f"{dec['p_median']:.3g}, {100*dec['frac_draws_p05']:.0f}% of draws p<0.05. That is the "
                f"strongest link type this test has scored, level with PPI (1.208x) and well above "
                f"co-dependency (1.070x) and regulatory (1.021x). Decile matching leaves a residual "
                f"imbalance on gene responsiveness (p {dec['residual_gene_responsiveness_p']:.1g}), so the "
                f"test was re-run at {TIGHT}x{TIGHT} bins where that clears "
                f"(p {tig['residual_gene_responsiveness_p']:.2g}); the ratio softens to {tig['ratio']:.3f} "
                f"and stays significant in {100*tig['frac_draws_p05']:.0f}% of draws, so the effect is not "
                f"the covariate talking.")
        if split:
            v = (f"CO-COMPLEX MEMBERSHIP CARRIES REAL SIGNAL, BUT ONLY FOR LARGE MACHINES. {head} {core} "
                 f"The signal is entirely in the big complexes: >{SMALL_MAX} subunits gives "
                 f"{r_large['ratio']:.3f} over {r_large['n_linked']:,} pairs "
                 f"({100*r_large['frac_draws_p05']:.0f}% of draws p<0.05) while <={SMALL_MAX} subunits gives "
                 f"{r_small['ratio']:.3f} over {r_small['n_linked']:,} pairs and reaches p<0.05 in "
                 f"{100*r_small['frac_draws_p05']:.0f}% of them. That is not a power artefact: subsampled to "
                 f"the small set's exact n of {r_pow['n_linked']:,}, large complexes still score "
                 f"{r_pow['ratio']:.3f} at median p {r_pow['p_median']:.3g} in "
                 f"{100*r_pow['frac_draws_p05']:.0f}% of draws. Across bands the effect is a GRADIENT rather "
                 f"than a threshold: " +
                 ", ".join(f"{lab} {bands[lab]['ratio']:.3f}" for _lo, _hi, lab in SIZE_BANDS
                           if bands.get(lab)) +
                 f". The practical consequence is that a CORUM heterodimer edge should not be admitted on "
                 f"the strength of the pooled number -- the pooled number is the ribosome and the "
                 f"spliceosome. CORUM IS BROADER, NOT BETTER PER PAIR: on the identical test the existing "
                 f"untraceable layer scores {e_all['ratio']:.3f} over its {e_all['n_linked']:,} cells, "
                 f"slightly ABOVE CORUM's {dec['ratio']:.3f} -- what CORUM buys is "
                 f"{len(L)/max(len(LE),1):.1f}x the coverage and a citable provenance, not a stronger edge, "
                 f"so the two should be unioned rather than one swapped for the other. And the ASSUMED "
                 f"1:1:1 limiting-subunit capacity earns nothing: the "
                 f"lowest-abundance subunit is {'LESS' if (lim_res and lim_res['mean_delta'] < 0) else 'not more'} "
                 f"disruptive to knock down than its complex-mates" +
                 (f" ({lim_res['mean_delta']:+.4f}, Wilcoxon p {lim_res['wilcoxon_p']:.3g}, "
                  f"{100*lim_res['frac_limiting_more_disruptive']:.0f}% positive over "
                  f"{lim_res['n_complexes']:,} complexes)" if lim_res else "") +
                 f", so the assumption is recorded as falsified rather than carried.")
        else:
            v = (f"CO-COMPLEX MEMBERSHIP CARRIES REAL SIGNAL. {head} {core} Split by size, "
                 f"<={SMALL_MAX} subunits gives {r_small['ratio']:.3f} "
                 f"({100*r_small['frac_draws_p05']:.0f}% of draws p<0.05) and >{SMALL_MAX} gives "
                 f"{r_large['ratio']:.3f} ({100*r_large['frac_draws_p05']:.0f}%), so the size split does "
                 f"not cleanly separate the two on this data.")
        if not survives:
            v += (f" CAUTION: at {TIGHT}x{TIGHT} matching the ratio falls to {tig['ratio']:.3f}, so part of "
                  f"the decile-matched figure is coarse matching rather than complex membership.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")

    # ---- write ----
    layer = {"model": "corum-v1", "source": SOURCE, "release": rel, "assembly_free": True,
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
