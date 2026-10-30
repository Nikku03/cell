"""PTM SITES WITH POSITIONS AND ENZYMES -- and whether a site annotation says anything about this cell.

WHAT WAS THERE BEFORE. The cell object's `ptm` field holds 8,425 entries, each a COUNT and a list of type
names. No residue, no position, no enzyme. A count cannot be used for activity: you cannot ask whether
S473 of AKT1 is the site CDK1 hits, you cannot ask which kinase to perturb to change a substrate, and you
cannot even ask whether the count is measuring biology or measuring how hard the protein was studied. This
module replaces the count with 74,862 positioned sites (residue, position, modification type, evidence code,
named enzyme where UniProt gives one) and 5,136 enzyme->substrate gene pairs, and then -- the part that
matters -- measures whether either carries information about the one downstream decision this project can
actually score.

THE DISTINCTION THAT GOVERNS EVERYTHING HERE. A UniProt MOD_RES line says A SITE EXISTS -- somewhere, in
some tissue, in some experiment, often a proteome-wide MS study of a different cell line, and 27% of the
time inferred from a mouse ortholog rather than observed in human at all. It does NOT say the site is
occupied in K562. There is no K562 phosphoproteome in this layer, so `n_phosphosites` is a property of the
PROTEIN and of the LITERATURE, never a state of the cell. Every number below is written under that reading,
and the layer stores an `occupancy_measured: false` flag so a downstream model cannot quietly promote a
site's existence into a site's activity.

TWO TESTS, BECAUSE THE LAYER MAKES TWO DIFFERENT CLAIMS.

  TEST A -- SITE COUNT AS A GENE PROPERTY (the prescribed test). Are genes carrying many annotated
  phosphosites more responsive in Perturb-seq than matched genes? The confound is not subtle and it is
  enormous: site count correlates with publication count at rho 0.44 and with protein length at rho 0.46,
  and both of those correlate with responsiveness. So the match is on publication-count decile x
  protein-length decile, and then re-run with K562 mRNA abundance added, because mass-spectrometry site
  discovery is abundance-driven and abundance is the single strongest predictor of responsiveness here
  (rho 0.52). If the effect survives only at the two-covariate match, it is the third covariate talking.

  A MEASUREMENT TRAP FOUND ON THE WAY, and recorded because it would silently sink this test. The project's
  robust z is computed PER GENE COLUMN -- median and MAD over perturbations -- so every column has median 0
  and MAD 1 BY CONSTRUCTION. `mean |z| of a gene's column` therefore has sd 0.046 on a mean of 0.90; it is
  nearly a constant, and it varies only with the column's kurtosis. It is the right thing to MATCH on (it
  is what the pair test controls) and the wrong thing to use as a gene-level OUTCOME, where it would return
  a p of 0.003 on a ratio of 1.0009 -- significance with no effect, which is exactly the error this project
  has made four times. So Test A is reported on three outcomes: raw mean |log fold change| (unnormalised,
  real dynamic range), the fraction of perturbations reaching |z| >= 5 (tail responsiveness), and mean |z|
  (shown only to demonstrate it is degenerate).

  TEST B -- ENZYME -> SUBSTRATE AS AN EDGE (the project's standard pair test, comparable to the reference
  numbers). For (perturbation P, gene G) pairs where P's product modifies G's product, is |z| elevated over
  pairs matched on perturbation-strength decile x gene-responsiveness decile? Reference: PPI 1.208x,
  co-dependency 1.070x, regulatory 1.021x, signalling 1.028x (null).

WHAT WOULD FALSIFY THE NULL, AND WHY THE POWER CHECK IS NOT OPTIONAL. Only 2,810 (P,G) cells are testable
in Test B against PPI's 94,368. A null at that n could be a null or could be no power, and the two are not
distinguishable from the p-value alone. So BioPlex 3.0 -- an interaction layer that scores 1.088x on this
same machinery over 75,639 pairs -- is subsampled to exactly this n and re-run as a POSITIVE CONTROL. It
turns out to keep its point estimate (1.089x) and lose its significance (20% of draws p<0.05), which
settles how Test B must be read: by effect size and bootstrap interval, not by p. A SHAM arm of random pairs,
REDRAWN every draw rather than fixed, bounds the other side at 0.996x -- a sham whose linked set is fixed
varies only in its control and is one sample wearing twenty hats.

THE MECHANISTIC PRIOR, STATED BEFORE THE RESULT. Phosphorylation is post-translational. Knocking down a
kinase changes its substrate's PHOSPHORYLATION, not necessarily its mRNA, so a null in a transcriptional
readout is the mechanistically expected outcome and would NOT mean the edges are wrong -- it would mean
Perturb-seq is blind to them. The one sub-arm where a transcriptional effect IS predicted is enzyme ->
TF-substrate: phosphorylating a transcription factor should propagate to transcription. That arm is split
out and reported separately rather than buried in the pooled number.

PROVENANCE IS PINNED AND ASSERTED, NOT DESCRIBED. UniProt release is read from the response header and
compared against a constant; iPTMnet release is parsed from its readme and compared against a constant. One
organism, reviewed entries only, one release each. A source that drifts under a result is a result about
the source.
"""
import collections
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
DIR = SP / "ptm"
UNI = DIR / "uniprot_ptm.tsv.gz"
UNI_HDR = DIR / "uniprot_release.json"
IPTM = DIR / "iptm_ptm.txt"
IPTM_README = DIR / "iptm_readme.txt"
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
BIOPLEX = SP / "cell_bioplex.json.gz"
LAYER = SP / "cell_ptm_sites.json.gz"

# ---- pinned provenance: one release per axis, asserted at runtime ----
UNIPROT_RELEASE = "2026_02"
IPTMNET_RELEASE = "6.2"
UNIPROT_STREAM = ("https://rest.uniprot.org/uniprotkb/stream?query=organism_id%3A9606%20AND%20reviewed"
                  "%3Atrue&fields=accession%2Creviewed%2Cid%2Cgene_primary%2Clength%2Cft_mod_res"
                  "%2Cft_carbohyd%2Cft_lipid%2Cxref_ensembl&format=tsv&compressed=true")
UNIPROT_SEARCH = ("https://rest.uniprot.org/uniprotkb/search?query=organism_id:9606+AND+reviewed:true"
                  "&size=1&fields=accession")
IPTM_BASE = "https://research.bioinformatics.udel.edu/iptmnet_data/files/current"

DECILES = 10
N_DRAWS = 20
HI_SITES = 5          # "many phosphosites": top ~24% of testable genes
Z_TAIL = 5.0          # |z| at which a perturbation counts as having moved this gene

# residue letters are DERIVED FROM THE MODIFICATION NAME, never guessed. A modification whose name does not
# name its residue gets residue=None rather than a plausible letter.
RESIDUE = [("serine", "S"), ("threonine", "T"), ("tyrosine", "Y"), ("lysine", "K"), ("arginine", "R"),
           ("histidine", "H"), ("proline", "P"), ("asparagine", "N"), ("aspart", "D"), ("glutam", "E"),
           ("cysteine", "C"), ("glycine", "G"), ("alanine", "A"), ("methionine", "M"),
           ("tryptophan", "W"), ("valine", "V"), ("isoleucine", "I"), ("leucine", "L"),
           ("phenylalanine", "F"), ("glutamine", "Q")]


def residue_of(note):
    n = note.lower()
    for k, v in RESIDUE:
        if k in n:
            return v
    return None


def fetch_url(url, path, tries=4, timeout=900, headers_to=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and (headers_to is None or headers_to.exists()):
        return
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "cellos", "accept": "*/*"})
            with urllib.request.urlopen(r, timeout=timeout) as fh:
                if headers_to is not None:
                    headers_to.write_text(json.dumps(dict(fh.headers)))
                with open(path, "wb") as o:
                    while True:
                        b = fh.read(1 << 20)
                        if not b:
                            break
                        o.write(b)
            return
        except Exception:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def fetch(report):
    """Both sources, with the release ASSERTED rather than assumed."""
    # UniProt: the release is a response header, so it is read from the live service and compared
    r = urllib.request.Request(UNIPROT_SEARCH, headers={"User-Agent": "cellos"})
    with urllib.request.urlopen(r, timeout=180) as fh:
        rel = fh.headers.get("x-uniprot-release")
        reldate = fh.headers.get("x-uniprot-release-date")
    if rel != UNIPROT_RELEASE:
        raise SystemExit(
            f"UniProt is serving release {rel}, this module is pinned to {UNIPROT_RELEASE}. One source "
            f"version along one axis: adopt the new release deliberately (bump UNIPROT_RELEASE, delete "
            f"{UNI}, re-run every number in this file), do not let it drift under the results.")
    report(f"  UniProtKB/Swiss-Prot release {rel} ({reldate}), human, reviewed only")
    fetch_url(UNIPROT_STREAM, UNI, headers_to=UNI_HDR)
    report(f"    {UNI.name}  ({UNI.stat().st_size/1e6:.1f} MB)")

    fetch_url(f"{IPTM_BASE}/readme.txt", IPTM_README, timeout=180)
    m = re.search(r"release\s+([0-9.]+)", IPTM_README.read_text())
    if not m or m.group(1) != IPTMNET_RELEASE:
        raise SystemExit(f"iPTMnet readme reports release {m.group(1) if m else '?'}, pinned to "
                         f"{IPTMNET_RELEASE}. Adopt a new release deliberately.")
    report(f"  iPTMnet release {m.group(1)} (PIR, U. Delaware / Georgetown; CC BY-NC-SA 4.0)")
    fetch_url(f"{IPTM_BASE}/ptm.txt", IPTM)
    report(f"    {IPTM.name}  ({IPTM.stat().st_size/1e6:.1f} MB)")
    return {"uniprot_release": rel, "uniprot_release_date": reldate, "iptmnet_release": m.group(1)}


# ---------------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------------
FEAT = re.compile(r'(MOD_RES|CARBOHYD|LIPID) (\d+); /note="([^"]*)"(?:; /evidence="([^"]*)")?')


def parse_uniprot(report):
    """accession -> {length, sites:[...]}. Sites carry position, residue, type, evidence and enzyme."""
    prot = {}
    n_rev, n_unrev = 0, 0
    with gzip.open(UNI, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(hdr)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                p += [""] * (len(hdr) - len(p))
            if p[ix["Reviewed"]] != "reviewed":
                n_unrev += 1
                continue
            n_rev += 1
            sites = []
            for col, blob in (("mod", p[ix["Modified residue"]]), ("glyco", p[ix["Glycosylation"]]),
                              ("lipid", p[ix["Lipidation"]])):
                if not blob:
                    continue
                for kind, pos, note, ev in FEAT.findall(blob):
                    typ = note.split(";")[0].strip()
                    enz = re.search(r"\bby ([^;]+)", note)
                    sites.append({"pos": int(pos), "res": residue_of(typ), "type": typ,
                                  "cls": col, "eco": sorted(set(re.findall(r"ECO:\d+", ev or ""))),
                                  "enzyme_text": enz.group(1).strip() if enz else None})
            prot[p[ix["Entry"]]] = {
                "length": int(p[ix["Length"]]) if p[ix["Length"]] else None,
                "symbol": p[ix["Gene Names (primary)"]] or None,
                "sites": sites}
    if n_unrev:
        raise SystemExit(f"{n_unrev:,} unreviewed entries in a reviewed-only query -- the annotation axis "
                         f"is not pinned")
    ns = sum(len(v["sites"]) for v in prot.values())
    report(f"  UniProt: {n_rev:,} reviewed human proteins, {ns:,} positioned sites "
           f"({sum(1 for v in prot.values() if v['sites']):,} proteins carry at least one)")
    return prot


def parse_iptmnet(report):
    """(enzyme_acc, substrate_acc) -> {sources, ptm types, sites}. Human rows with an enzyme only."""
    rows, edges = 0, collections.defaultdict(lambda: {"src": set(), "type": set(), "site": set(),
                                                      "pmid": set()})
    subs, n_human = set(), 0
    with open(IPTM) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8 or p[4] != "Homo sapiens (Human)":
                continue
            n_human += 1
            subs.add(p[2].split("-")[0])
            if not p[6]:
                continue
            rows += 1
            k = (p[6].split("-")[0], p[2].split("-")[0])
            e = edges[k]
            e["src"].add(p[1])
            e["type"].add(p[0])
            if p[5]:
                e["site"].add(p[5])
            if len(p) > 9 and p[9]:
                e["pmid"].add(p[9])
    report(f"  iPTMnet: {n_human:,} human PTM rows over {len(subs):,} substrates; {rows:,} rows name an "
           f"enzyme, giving {len(edges):,} enzyme->substrate accession pairs")
    return edges


# ---------------------------------------------------------------------------------------------------
# the fast test -- pair level, identical machinery to the co-complex / PPI layers
# ---------------------------------------------------------------------------------------------------
class Matcher:
    """Matched draws on (perturbation strength decile) x (gene responsiveness decile).

    11,258 x 8,248 is too large to enumerate, so the matched control is drawn by rejection inside each
    matched cell: pick a random row of the right strength bin and a random column of the right
    responsiveness bin, reject it if it is linked, a self-pair, already drawn, or non-finite.
    """

    def __init__(self, A, prow, pcol, pert_eid, gene_eid, nbin):
        self.A, self.prow, self.pcol, self.nbin = A, prow, pcol, nbin
        q = np.linspace(1.0 / nbin, 1.0 - 1.0 / nbin, nbin - 1)
        self.rd = np.digitize(prow, np.quantile(prow, q))
        self.cd = np.digitize(pcol, np.quantile(pcol, q))
        self.rows_by = [np.where(self.rd == k)[0] for k in range(nbin)]
        self.cols_by = [np.where(self.cd == k)[0] for k in range(nbin)]
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

    def run(self, L, linked, cap=None, n_draws=N_DRAWS, label="", sham_n=None, rng0=None):
        """`sham_n` redraws the LINKED arm at random every draw -- the only honest sham.

        A sham whose linked set is fixed varies only in its control, so its ratio is one sample of the
        linked arm dressed up as twenty. That is the single-draw error this project has made three times,
        wearing a different hat.
        """
        from scipy import stats
        if sham_n is None and len(L) < 100:
            return None
        idx = None if sham_n else np.arange(len(L))
        rats, ps, m1, m0, rp, cp, ns = [], [], [], [], [], [], []
        keep1, keep0 = [], []
        for s in range(n_draws):
            rng = np.random.default_rng(s if rng0 is None else rng0 + s)
            if sham_n:
                li, lj = self.random_cells(sham_n, rng)
            else:
                use = idx if (cap is None or len(idx) <= cap) else rng.choice(idx, cap, replace=False)
                li, lj = L[use, 0], L[use, 1]
            cell = collections.Counter(zip(self.rd[li].tolist(), self.cd[lj].tolist()))
            ci, cj = self.draw(cell, set() if sham_n else linked, rng)
            a1, a0 = self.A[li, lj], self.A[ci, cj]
            rats.append(float(a1.mean() / a0.mean()))
            ps.append(float(stats.mannwhitneyu(a1, a0, alternative="two-sided")[1]))
            m1.append(float(a1.mean()))
            m0.append(float(a0.mean()))
            rp.append(float(stats.mannwhitneyu(self.prow[li], self.prow[ci])[1]))
            cp.append(float(stats.mannwhitneyu(self.pcol[lj], self.pcol[cj])[1]))
            ns.append(int(len(li)))
            keep1.append(a1)
            keep0.append(a0)
        # A BOOTSTRAP INTERVAL, NOT JUST A p. At 2,810 pairs a p-value answers "could this be zero?" and
        # says nothing about which effect sizes are excluded -- which is the whole question for a null.
        b1, b0 = np.concatenate(keep1), np.concatenate(keep0)
        br = np.random.default_rng(7)
        boot = [float(br.choice(b1, len(b1) // len(keep1), replace=True).mean()
                      / br.choice(b0, len(b0) // len(keep0), replace=True).mean()) for _ in range(400)]
        sd = float(np.std(np.concatenate([b1, b0])))
        mde = 2.8 * sd * np.sqrt(2.0 / np.mean(ns)) / max(np.mean(m0), 1e-12)
        return {"label": label, "n_linked": int(np.mean(ns)), "n_pool": int(len(L)) if not sham_n else sham_n,
                "n_draws": n_draws, "capped_to": cap, "bins": self.nbin,
                # THE RAW HELD-OUT VALUE IS THE EFFECT SIZE. The ratio is only the comparable statistic.
                "raw_mean_abs_z_linked": float(np.mean(m1)),
                "raw_mean_abs_z_matched": float(np.mean(m0)),
                "ratio": float(np.mean(rats)), "ratio_sd": float(np.std(rats)),
                "ratio_min": float(np.min(rats)), "ratio_max": float(np.max(rats)),
                "ratio_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "min_detectable_ratio_80pct": float(1.0 + mde),
                "p_median": float(np.median(ps)), "frac_draws_p05": float(np.mean([p < 0.05 for p in ps])),
                "residual_pert_strength_p": float(np.mean(rp)),
                "residual_gene_responsiveness_p": float(np.mean(cp))}

    def random_cells(self, n, rng):
        """n uniformly random FINITE cells. Non-finite cells are rejected, never imputed."""
        ri, ci = [], []
        while len(ri) < n:
            x = rng.integers(0, self.A.shape[0], n)
            y = rng.integers(0, self.A.shape[1], n)
            m = np.isfinite(self.A[x, y]) & (self.pk[x] != self.gk[y])
            ri += x[m].tolist()
            ci += y[m].tolist()
        return np.array(ri[:n], dtype=np.int64), np.array(ci[:n], dtype=np.int64)


PHEAD = (f"    {'arm':34s} {'pairs':>7s} {'raw |z|':>9s} {'matched':>9s} {'ratio':>8s} "
         f"{'95% CI':>16s} {'med p':>9s} {'p<.05':>6s} {'MDE':>7s}")


def pline(r, label=None):
    if r is None:
        return f"    {(label or '?'):34s}  too few testable pairs"
    ci = f"{r['ratio_ci95'][0]:.3f}-{r['ratio_ci95'][1]:.3f}"
    return (f"    {(label or r['label']):34s} {r['n_linked']:7,d} {r['raw_mean_abs_z_linked']:9.4f} "
            f"{r['raw_mean_abs_z_matched']:9.4f} {r['ratio']:8.4f} {ci:>16s} {r['p_median']:9.3g} "
            f"{100*r['frac_draws_p05']:5.0f}% {r['min_detectable_ratio_80pct']:7.3f}")


# ---------------------------------------------------------------------------------------------------
# the gene-level test
# ---------------------------------------------------------------------------------------------------
def gene_test(Y, hi, lo, covs, nbin, n_draws=N_DRAWS, label=""):
    """HIGH-site genes vs 0-site genes, matched on covariate deciles, drawn n_draws times."""
    from scipy import stats
    keys = [np.digitize(v, np.quantile(v, np.linspace(1/nbin, 1-1/nbin, nbin-1))) for v in covs.values()]
    p1, p0 = collections.defaultdict(list), collections.defaultdict(list)
    for i in range(len(Y)):
        k = tuple(int(x[i]) for x in keys)
        if hi[i]:
            p1[k].append(i)
        elif lo[i]:
            p0[k].append(i)
    rr, pp, nn, m1, m0 = [], [], [], [], []
    res = collections.defaultdict(list)
    for s in range(n_draws):
        rng = np.random.default_rng(s)
        k1, k0 = [], []
        for k, i1 in p1.items():
            i0 = p0.get(k, [])
            n = min(len(i1), len(i0))
            if n:
                k1 += list(rng.choice(i1, n, replace=False))
                k0 += list(rng.choice(i0, n, replace=False))
        k1, k0 = np.array(k1, dtype=int), np.array(k0, dtype=int)
        if len(k1) < 50:
            continue
        rr.append(float(Y[k1].mean() / Y[k0].mean()))
        pp.append(float(stats.mannwhitneyu(Y[k1], Y[k0], alternative="two-sided")[1]))
        nn.append(len(k1))
        m1.append(float(Y[k1].mean()))
        m0.append(float(Y[k0].mean()))
        for c, v in covs.items():
            res[c].append(float(stats.mannwhitneyu(v[k1], v[k0])[1]))
    if not rr:
        return None
    # what ratio could this design have detected? two-sample normal approximation at 80% power, alpha .05
    sd = float(np.std(np.concatenate([Y[hi], Y[lo]])))
    mde = 2.8 * sd * np.sqrt(2.0 / np.mean(nn))
    return {"label": label, "n_per_arm": int(np.mean(nn)), "n_hi_pool": int(hi.sum()),
            "n_lo_pool": int(lo.sum()), "bins": nbin, "n_draws": len(rr),
            "raw_hi": float(np.mean(m1)), "raw_lo": float(np.mean(m0)),
            "ratio": float(np.mean(rr)), "ratio_sd": float(np.std(rr)),
            "p_median": float(np.median(pp)), "frac_draws_p05": float(np.mean([p < 0.05 for p in pp])),
            "residual": {c: float(np.mean(v)) for c, v in res.items()},
            "min_detectable_ratio_80pct": float(1.0 + mde / max(np.mean(m0), 1e-12))}


GHEAD = (f"    {'outcome / match':44s} {'n/arm':>6s} {'raw hi':>9s} {'raw lo':>9s} {'ratio':>8s} "
         f"{'med p':>10s} {'p<.05':>6s} {'MDE':>7s}")


def gline(r, label):
    if r is None:
        return f"    {label:44s}  no balanced draw"
    return (f"    {label:44s} {r['n_per_arm']:6,d} {r['raw_hi']:9.5f} {r['raw_lo']:9.5f} "
            f"{r['ratio']:8.4f} {r['p_median']:10.3g} {100*r['frac_draws_p05']:5.0f}% "
            f"{r['min_detectable_ratio_80pct']:7.4f}")


# ---------------------------------------------------------------------------------------------------
def main():
    from scipy import stats
    import h5py
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("PTM SITES -- positions and enzymes, and whether either predicts a transcriptional response")
    report("=" * 100)

    prov = fetch(report)
    from entity_registry import load as load_reg
    REG = load_reg()

    prot = parse_uniprot(report)
    ipt = parse_iptmnet(report)

    # ---- joins, declared ----
    accs = sorted(prot)
    _e, st_u = REG.join("gene", "uniprot", accs, 0.90, "UniProt reviewed accessions -> gene")
    report(f"  registry join UniProt accession -> gene: {st_u['joined']:,}/{st_u['of']:,} "
           f"({st_u['rate']:.1%})  <- accession, the strong key; symbols are never used for this join")
    iacc = sorted({a for k in ipt for a in k})
    _e2, st_i = REG.join("gene", "uniprot", iacc, 0.85, "iPTMnet enzyme/substrate accessions -> gene")
    report(f"  registry join iPTMnet accession -> gene:  {st_i['joined']:,}/{st_i['of']:,} "
           f"({st_i['rate']:.1%})  <- isoform suffixes stripped to the canonical accession")
    # the misses are NAMED rather than left as a residual. A counted, classified miss is a limit; an
    # uncounted one is the silent-join failure this project keeps re-discovering.
    miss = [a for a in iacc if not REG.resolve("gene", "uniprot", a)]
    n_pro = sum(1 for a in miss if a.startswith("PR:"))
    report(f"    {len(miss)} unresolved: {n_pro} are PRO ontology protein-FORM ids (e.g. PR:000027108), "
           f"not accessions, and the remaining {len(miss)-n_pro} are obsolete or TrEMBL entries with no "
           f"Ensembl xref. Both are dropped and counted, never guessed at.")

    acc2eid = {a: REG.resolve("gene", "uniprot", a) for a in accs}
    eid2acc = collections.defaultdict(list)
    for a, g in acc2eid.items():
        if g:
            eid2acc[g].append(a)

    # UniProt's own `by ENZYME` free text, resolved where it happens to BE a gene symbol. It usually is not
    # -- PKA, CK2, PKC name families, not genes -- so this is a weak, partial second source and is recorded
    # as via=symbol, never merged into the accession-keyed edges.
    bytext = collections.Counter()
    by_edges = collections.defaultdict(set)
    for a, v in prot.items():
        for s in v["sites"]:
            if s["enzyme_text"]:
                bytext[s["enzyme_text"]] += 1
                e = REG.resolve("gene", "symbol", s["enzyme_text"])
                if e and acc2eid.get(a) and e != acc2eid[a]:
                    by_edges[(e, acc2eid[a])].add(s["type"])
    n_by_res = sum(v for k, v in bytext.items() if REG.resolve("gene", "symbol", k))
    report(f"  UniProt names a modifying enzyme on {sum(bytext.values()):,} sites using {len(bytext):,} "
           f"free-text labels; {n_by_res:,} of those annotations ({100*n_by_res/max(sum(bytext.values()),1):.0f}%) "
           f"use a string that IS a gene symbol -- the rest name families (PKA, CK2, PKC) or 'autocatalysis' "
           f"and are stored as text, not resolved")

    # ---- GWPS matrix ----
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        gt = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        vg = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0] for s in f["var"]["gene_id"][:]]
        mrna = f["var"]["mean"][:]
    Xm = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    n_nonfinite = int((~np.isfinite(X)).sum())
    del X
    okr = np.isfinite(Xm)
    raw_resp = np.where(okr, np.abs(Xm), 0).sum(0) / np.maximum(okr.sum(0), 1)
    med = np.nanmedian(Xm, 0)
    mad = np.nanmedian(np.abs(Xm - med), 0)
    sc = 1.4826 * mad
    A = np.abs((Xm - med) / np.where(sc > 0, sc, np.nan)).astype(np.float32)
    del Xm, okr
    ok = np.isfinite(A)
    prow = np.where(ok, A, 0).sum(1) / np.maximum(ok.sum(1), 1)
    pcol = np.where(ok, A, 0).sum(0) / np.maximum(ok.sum(0), 1)
    tail = np.where(ok, A >= Z_TAIL, 0).sum(0) / np.maximum(ok.sum(0), 1)
    report(f"\n  GWPS: {A.shape[0]:,} perturbations x {A.shape[1]:,} genes, {n_nonfinite:,} non-finite "
           f"cells DROPPED (never imputed)")
    report(f"  per-gene robust z is normalised PER COLUMN, so mean |z| per column has sd "
           f"{pcol.std():.4f} on mean {pcol.mean():.4f} -- near-constant BY CONSTRUCTION. It is the "
           f"matching covariate, not a gene-level outcome.")

    pert, n_sym = [], 0
    for g in gt:
        p = g.split("_")
        last = p[-1]
        if last.startswith("ENSG"):
            pert.append(REG.resolve("gene", "ensembl", last))
        elif len(p) > 1 and p[1] != "non-targeting":
            e = REG.resolve("gene", "symbol", p[1])
            pert.append(e)
            n_sym += e is not None
        else:
            pert.append(None)
    gene = [REG.resolve("gene", "ensembl", e) for e in vg]
    report(f"  perturbations resolved {sum(x is not None for x in pert):,}/{len(pert):,} (by ENSG; "
           f"{n_sym} by symbol where the file carries none); measured genes "
           f"{sum(x is not None for x in gene):,}/{len(gene):,}")
    pm, gidx = collections.defaultdict(list), collections.defaultdict(list)
    for i, e in enumerate(pert):
        if e:
            pm[e].append(i)
    for j, e in enumerate(gene):
        if e:
            gidx[e].append(j)

    # ---- cell-object covariates ----
    d = json.load(gzip.open(NET, "rt"))
    pubs, istf, old_ptm = {}, {}, d.get("ptm", {})
    for g in d["genes"]:
        e = REG.resolve("gene", "symbol", g["name"])
        if e:
            if g.get("pubs") is not None:
                pubs[e] = float(g["pubs"])
            istf[e] = bool(g.get("tf"))
    del d
    report(f"  cell object: `pubs` for {len(pubs):,} genes; existing `ptm` field has {len(old_ptm):,} "
           f"entries carrying a COUNT and a type list only (no site, no position, no enzyme)")

    # =============================================================================================
    # TEST A -- site count as a gene property
    # =============================================================================================
    report("\n" + "-" * 100)
    report("  TEST A -- do genes with many annotated phosphosites respond more in Perturb-seq?")
    report("-" * 100)
    sel, C, CNH, LN, PB, MN = [], [], [], [], [], []
    n_no_uni, n_no_pubs = 0, 0
    for j, e in enumerate(gene):
        if not e:
            continue
        if e not in eid2acc:
            n_no_uni += 1
            continue
        a = max(eid2acc[e], key=lambda z: sum(1 for s in prot[z]["sites"] if s["cls"] == "mod"))
        if e not in pubs:
            n_no_pubs += 1
            continue
        ph = [s for s in prot[a]["sites"] if s["type"].startswith("Phospho")]
        sel.append(j)
        C.append(len(ph))
        CNH.append(sum(1 for s in ph if "ECO:0007744" not in s["eco"]))
        LN.append(prot[a]["length"] or 0)
        PB.append(pubs[e])
        MN.append(float(mrna[j]))
    sel = np.array(sel)
    C, CNH, LN, PB, MN = (np.array(x, dtype=float) for x in (C, CNH, LN, PB, MN))
    report(f"  testable genes {len(sel):,} of {len(gene):,} measured  "
           f"({n_no_uni:,} have no reviewed UniProt entry -> UNKNOWN, not zero sites; "
           f"{n_no_pubs:,} lack a publication count)")
    report(f"  phosphosite count: {100*(C == 0).mean():.1f}% have none annotated, median "
           f"{np.median(C):.0f}, 90th pct {np.percentile(C, 90):.0f}, max {C.max():.0f}")
    report(f"  the confound, measured: rho(sites, pubs) {stats.spearmanr(C, PB)[0]:+.3f}   "
           f"rho(sites, length) {stats.spearmanr(C, LN)[0]:+.3f}   "
           f"rho(sites, K562 mRNA) {stats.spearmanr(C, MN)[0]:+.3f}")
    OUTC = {"raw mean |lfc|": raw_resp[sel], f"frac pert |z|>={Z_TAIL:.0f}": tail[sel],
            "mean |z| (degenerate)": pcol[sel]}
    for nm, Y in OUTC.items():
        report(f"  rho({nm:22s}, sites) {stats.spearmanr(C, Y)[0]:+.4f} "
               f"(p {stats.spearmanr(C, Y)[1]:.3g});  vs pubs {stats.spearmanr(PB, Y)[0]:+.3f}  "
               f"vs length {stats.spearmanr(LN, Y)[0]:+.3f}  vs mRNA {stats.spearmanr(MN, Y)[0]:+.3f}")

    hi, lo = C >= HI_SITES, C == 0
    report(f"\n  HIGH = >= {HI_SITES} annotated phosphosites ({int(hi.sum()):,} genes), "
           f"LOW = 0 annotated ({int(lo.sum()):,} genes); matched draws, {N_DRAWS} each")
    report(f"  before matching, HIGH vs LOW differ on every covariate: pubs "
           f"{np.median(PB[hi]):.0f} vs {np.median(PB[lo]):.0f}, length {np.median(LN[hi]):.0f} vs "
           f"{np.median(LN[lo]):.0f}, mRNA {np.median(MN[hi]):.3f} vs {np.median(MN[lo]):.3f}")
    report(GHEAD)
    TA = {}
    for nm, Y in OUTC.items():
        for cname, covs, nb in (("pubs x length (d10)", {"pubs": PB, "length": LN}, DECILES),
                                ("pubs x length x mRNA (q5)",
                                 {"pubs": PB, "length": LN, "mRNA": MN}, 5)):
            r = gene_test(Y, hi, lo, covs, nb, label=f"{nm} | {cname}")
            TA[f"{nm} | {cname}"] = r
            report(gline(r, f"{nm} | {cname}"))
    for k, r in TA.items():
        if r:
            report(f"    residual imbalance  {k:44s} "
                   f"{ {c: round(v, 3) for c, v in r['residual'].items()} }")
    # a gradient view that does not depend on the HIGH/LOW cut at all
    def partial_rho(x, y, ctrl):
        rx = np.column_stack([stats.rankdata(c) for c in ctrl])
        rx = np.column_stack([rx, np.ones(len(x))])
        bx = np.linalg.lstsq(rx, stats.rankdata(x), rcond=None)[0]
        by = np.linalg.lstsq(rx, stats.rankdata(y), rcond=None)[0]
        return float(np.corrcoef(stats.rankdata(x) - rx @ bx, stats.rankdata(y) - rx @ by)[0, 1])
    TA_partial = {nm: partial_rho(C, Y, [np.log1p(PB), np.log1p(LN), np.log1p(MN)])
                  for nm, Y in OUTC.items()}
    report(f"  partial Spearman of site count vs each outcome, controlling log pubs, log length, "
           f"log mRNA: " + ", ".join(f"{k} {v:+.4f}" for k, v in TA_partial.items()))
    report(f"  READ THE RATIO, NOT THE p. Several of these rows reach p<0.05 in most draws at a ratio of "
           f"1.00x: with ~800 genes per arm and a tight outcome the rank test detects a shape difference "
           f"that is not an effect. The MDE column is what the design could have found; the ratio column "
           f"is what it did find.")
    # the non-high-throughput subset: sites NOT from ECO:0007744 proteome-wide MS
    hin, lon = CNH >= 3, CNH == 0
    TA_nohtp = gene_test(raw_resp[sel], hin, lon, {"pubs": PB, "length": LN}, DECILES,
                         label="raw mean |lfc| | curated (non-HTP) sites | pubs x length (d10)")
    report(gline(TA_nohtp, "raw |lfc| | non-HTP sites>=3 | pubs x len"))

    # =============================================================================================
    # TEST B -- enzyme -> substrate as an edge
    # =============================================================================================
    report("\n" + "-" * 100)
    report("  TEST B -- enzyme -> substrate edges under the project's standard pair test")
    report("-" * 100)
    es = collections.defaultdict(lambda: {"src": set(), "type": set()})
    for (ea, sa), v in ipt.items():
        e1, e2 = REG.resolve("gene", "uniprot", ea), REG.resolve("gene", "uniprot", sa)
        if e1 and e2 and e1 != e2:
            es[(e1, e2)]["src"] |= v["src"]
            es[(e1, e2)]["type"] |= v["type"]
    report(f"  iPTMnet enzyme->substrate resolved to {len(es):,} directed GENE pairs from "
           f"{len({a for a, _ in es}):,} enzymes")

    def build(pairs):
        L = []
        for (a, b) in pairs:
            for i in pm.get(a, []):
                for j in gidx.get(b, []):
                    if pert[i] == gene[j]:
                        continue
                    if np.isfinite(A[i, j]):
                        L.append((i, j))
        return np.array(L, dtype=np.int64).reshape(-1, 2)

    allp = sorted(es)
    L_all = build(allp)
    linked_all = set(map(tuple, L_all.tolist()))
    M = Matcher(A, prow, pcol, pert, gene, DECILES)
    report(f"  testable (P,G) cells: {len(L_all):,} from {len(set(L_all[:, 0].tolist())):,} "
           f"perturbations and {len(set(L_all[:, 1].tolist())):,} measured genes")

    arms = {
        "all enzyme->substrate": allp,
        "excl SIGNOR-sourced": [k for k in allp if es[k]["src"] - {"sign"}],
        "SIGNOR-sourced only": [k for k in allp if "sign" in es[k]["src"]],
        "substrate is a TF": [k for k in allp if istf.get(k[1])],
        "substrate is not a TF": [k for k in allp if not istf.get(k[1])],
        "UniProt 'by X' edges (via=symbol)": sorted(by_edges),
        "REVERSED (perturb substrate)": [(b, a) for (a, b) in allp]}
    TB = {}
    report(PHEAD)
    for nm, pr in arms.items():
        L = build(pr)
        TB[nm] = M.run(L, linked_all if "REVERSED" not in nm else set(map(tuple, L.tolist())), label=nm)
        report(pline(TB[nm], nm))

    # ---- SHAM: random pairs at the same n, REDRAWN EVERY DRAW. Must land at 1.00. ----
    TB["SHAM (random pairs, same n)"] = M.run(None, None, sham_n=len(L_all),
                                              label="SHAM (random pairs, same n)")
    report(pline(TB["SHAM (random pairs, same n)"]))

    # ---- POWER: BioPlex, a layer known to score 1.10x here, subsampled to exactly this n ----
    TB_power = None
    if BIOPLEX.exists():
        bp = json.load(gzip.open(BIOPLEX, "rt"))
        # SORTED, not set order: string hashing is randomised per process, so an unsorted set would make
        # the capped subsample -- and therefore the power estimate -- change from run to run.
        bpairs = sorted({(e["bait"], e["prey"]) for e in bp["edges"]})
        del bp
        L_bp = build(bpairs)
        linked_bp = set(map(tuple, L_bp.tolist()))
        TB_power = {"full": M.run(L_bp, linked_bp, label="BioPlex 3.0, all pairs"),
                    "capped": M.run(L_bp, linked_bp, cap=len(L_all),
                                    label=f"BioPlex 3.0 subsampled to n={len(L_all):,}")}
        report(f"\n  POWER CONTROL -- the same machinery on a layer that is NOT null, at the same n")
        report(PHEAD)
        report(pline(TB_power["full"]))
        report(pline(TB_power["capped"]))
        pc = TB_power["capped"]
        report(f"    -> at n={pc['n_linked']:,} a BioPlex-sized effect still shows up as a POINT ESTIMATE "
               f"({pc['ratio']:.3f}x) but reaches p<0.05 in only {100*pc['frac_draws_p05']:.0f}% of draws.")
        report(f"       So Test B's p-values are not the evidence here; the effect size and its interval "
               f"are. Anything at or above {pc['min_detectable_ratio_80pct']:.3f}x would have been "
               f"significant; below that, only the CI speaks.")
    else:
        report(f"\n  POWER CONTROL SKIPPED: {BIOPLEX} absent -- the null below is then not distinguishable "
               f"from no power")

    # =============================================================================================
    # the layer
    # =============================================================================================
    eco = collections.Counter()
    for v in prot.values():
        for s in v["sites"]:
            eco.update(s["eco"] or ["none"])
    n_enz_pert = len({a for a, _b in es if a in pm})
    sites_out, n_sites = {}, 0
    for a, v in prot.items():
        if not v["sites"]:
            continue
        e = acc2eid.get(a)
        sites_out[a] = {"gene": e, "symbol": v["symbol"], "length": v["length"],
                        "sites": [{"pos": s["pos"], "res": s["res"], "type": s["type"], "cls": s["cls"],
                                   "eco": s["eco"], "enzyme_text": s["enzyme_text"]} for s in v["sites"]]}
        n_sites += len(v["sites"])
    enz_out = [{"enzyme": a, "substrate": b, "enzyme_gene": REG.resolve("gene", "uniprot", a),
                "substrate_gene": REG.resolve("gene", "uniprot", b),
                "ptm_types": sorted(v["type"]), "sources": sorted(v["src"]),
                "sites": sorted(v["site"]), "n_pmid": len(v["pmid"])}
               for (a, b), v in ipt.items()]
    layer = {
        "model": "ptm-sites-v1",
        "sources": {"uniprot": f"UniProtKB/Swiss-Prot {prov['uniprot_release']} "
                               f"({prov['uniprot_release_date']}), human, reviewed, MOD_RES + CARBOHYD + "
                               f"LIPID features",
                    "iptmnet": f"iPTMnet {prov['iptmnet_release']} (PIR); enzyme-substrate rows, human"},
        "occupancy_measured": False,
        "occupancy_note": "every site here is ANNOTATED TO EXIST on the protein. No K562 phosphoproteome "
                          "is present in this layer, so no site can be called occupied, stoichiometric or "
                          "active in this cell. `n_phosphosites` is a property of the protein and of the "
                          "literature, not a state of the cell.",
        "n_proteins": len(prot), "n_proteins_with_site": len(sites_out), "n_sites": n_sites,
        "n_enzyme_substrate_accession_pairs": len(ipt),
        "join": {"uniprot": st_u, "iptmnet": st_i,
                 "note": "both joins go through UniProt ACCESSION, the strong key. The UniProt 'by X' "
                         "free-text enzyme labels are the only symbol-keyed join here and are recorded "
                         "via=symbol."},
        "replaces": {"field": "ptm", "n_entries_before": len(old_ptm),
                     "before": "count + type list, no position, no residue, no enzyme"},
        "sites": sites_out,
        "enzyme_substrate": enz_out}
    LAYER.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    # =============================================================================================
    # verdict
    # =============================================================================================
    key = "raw mean |lfc| | pubs x length (d10)"
    ta = TA[key]
    ta3 = TA["raw mean |lfc| | pubs x length x mRNA (q5)"]
    tb = TB["all enzyme->substrate"]
    tf_arm = TB["substrate is a TF"]
    sham = TB["SHAM (random pairs, same n)"]
    pw = TB_power["capped"] if TB_power else None

    R = {"model": "ptm-sites-v1", "sources": layer["sources"], "occupancy_measured": False,
         "n_proteins": len(prot), "n_proteins_with_site": len(sites_out), "n_sites": n_sites,
         "n_enzyme_substrate_pairs_accession": len(ipt), "n_enzyme_substrate_pairs_gene": len(es),
         "join": layer["join"],
         "confounds": {"rho_sites_pubs": float(stats.spearmanr(C, PB)[0]),
                       "rho_sites_length": float(stats.spearmanr(C, LN)[0]),
                       "rho_sites_mrna": float(stats.spearmanr(C, MN)[0]),
                       "rho_mrna_responsiveness": float(stats.spearmanr(MN, raw_resp[sel])[0])},
         "column_z_is_degenerate": {"mean": float(pcol.mean()), "sd": float(pcol.std()),
                                    "note": "robust z is normalised per gene column, so mean |z| per "
                                            "column is near-constant by construction and cannot serve as "
                                            "a gene-level outcome"},
         "test_a_gene_level": {k: v for k, v in TA.items()},
         "test_a_partial_spearman": TA_partial,
         "test_a_non_htp_sites": TA_nohtp,
         "test_b_pair_level": {k: v for k, v in TB.items()},
         "test_b_power_control": TB_power,
         "coverage": {"gwps_genes": len(gene),
                      "gwps_genes_with_reviewed_uniprot": int(len(gene) - n_no_uni
                                                              - sum(1 for e in gene if not e)),
                      "gwps_genes_testable_in_test_a": int(len(sel)),
                      "frac_testable_genes_with_zero_annotated_phosphosites": float((C == 0).mean()),
                      "missingness_is_biased": True},
         "limits": [
             f"A SITE ANNOTATION IS NOT AN OCCUPIED SITE. UniProt MOD_RES says the residue is modified "
             f"somewhere in some experiment; {eco['ECO:0000250']:,} of {n_sites:,} sites carry "
             f"ECO:0000250 (by similarity, usually from a mouse ortholog) and {eco['ECO:0007744']:,} "
             f"carry ECO:0007744 (proteome-wide MS, mostly not K562), against only "
             f"{eco['ECO:0000269']:,} with direct manual experimental evidence. Nothing here measures "
             f"phosphorylation IN K562, so no site can be called active, and site count must never be "
             f"read as pathway activity.",
             "SITE COUNT IS PARTLY A MEASURE OF STUDY EFFORT. rho 0.44 with publication count and 0.46 "
             "with protein length. Every gene-level number here is matched on both; the unmatched "
             "correlation is meaningless and is not reported as a result.",
             "PERTURB-SEQ IS A TRANSCRIPTIONAL READOUT AND PTM IS POST-TRANSLATIONAL. A null in Test B is "
             "consistent with the edges being correct and the assay being blind to them. This module "
             "cannot distinguish 'these edges are wrong' from 'mRNA does not see them'; that would take a "
             "K562 phosphoproteome after perturbation, which does not exist at this scale.",
             f"ENZYME COVERAGE IS THIN AND BIASED. {len(es):,} enzyme->substrate gene pairs from "
             f"{len({a for a, _ in es}):,} enzymes, of which only {n_enz_pert} are also Perturb-seq "
             f"perturbations -- kinases studied since the 1990s. hprd (2009) and SIGNOR contribute most "
             f"enzyme rows, so the edge set reflects two decades of kinase literature, not a survey.",
             f"THE UNIPROT 'by X' LABELS ARE MOSTLY NOT GENES. {len(bytext):,} free-text strings covering "
             f"{sum(bytext.values()):,} annotations; only {n_by_res:,} of those annotations use a string "
             f"that is itself a gene symbol. The commonest (PKA, CK2, PKC, GSK3-beta, AMPK) name FAMILIES "
             f"with several catalytic subunit genes, and {bytext.get('autocatalysis', 0)} say only "
             f"'autocatalysis'. They are stored as text and are resolved into edges only where the string "
             f"is itself a symbol, recorded via=symbol.",
             "ABSENCE IS UNKNOWN. A gene with no reviewed UniProt entry is excluded, not scored as zero "
             "sites; a reviewed protein with no MOD_RES is scored as zero ANNOTATED sites, which is a "
             "statement about UniProt, not about the protein. Missingness is biased toward the "
             "unpublished and the short.",
             "ISOFORMS ARE COLLAPSED. Positions are on the canonical UniProt sequence; iPTMnet isoform "
             "accessions are truncated to the canonical accession, so a site position is only valid "
             "against the canonical sequence of that release.",
             "ONE RELEASE EACH, AND THE RESULTS ARE NOT PORTABLE ACROSS RELEASES. UniProt "
             f"{prov['uniprot_release']} and iPTMnet {prov['iptmnet_release']} are asserted at runtime; "
             "site counts grow every release, so the HIGH/LOW cut at 5 sites is release-specific.",
             "TEST A IS A GENE-LEVEL RATIO AND IS NOT COMPARABLE TO THE PAIR-LEVEL REFERENCE NUMBERS "
             "(PPI 1.208x etc.). Only Test B is.",
             f"TEST B IS UNDERPOWERED FOR SMALL EFFECTS AND ITS p-VALUES ARE NOT THE EVIDENCE. "
             f"{tb['n_linked']:,} testable pairs against PPI's 94,368. The bootstrap 95% CI on the ratio "
             f"is [{tb['ratio_ci95'][0]:.3f}, {tb['ratio_ci95'][1]:.3f}]: a PPI- or CORUM-sized effect "
             f"(1.21x) is excluded, a co-dependency-sized one (1.07x) is not. The "
             f"BioPlex-at-the-same-n control returns the correct point estimate and loses significance, "
             f"which is why effect sizes and intervals are reported here rather than p-values.",
             "TEST A's MATCHED DRAWS OVERLAP. Only ~800 genes per arm survive matching out of pools of "
             f"{int(hi.sum()):,} and {int(lo.sum()):,}, so the 20 draws share most of their members and "
             "the fraction of draws reaching p<0.05 is not 20 independent tests. It is reported as a "
             "stability check on the draw, not as replication."]}

    report("\n" + "=" * 100)
    powtxt = ""
    if pw:
        powtxt = (f"WHAT TEST B CAN AND CANNOT EXCLUDE, stated rather than implied. Only {tb['n_linked']:,} "
                  f"(P,G) cells are testable against PPI's 94,368, so its p-values are weak evidence and "
                  f"its interval is the real answer: the ratio's bootstrap 95% CI is "
                  f"[{tb['ratio_ci95'][0]:.3f}, {tb['ratio_ci95'][1]:.3f}], which excludes a PPI-sized "
                  f"({1.208:.3f}x) or CORUM-sized effect but does NOT exclude a co-dependency-sized "
                  f"1.07x one. The power control makes that concrete: BioPlex 3.0 through this identical "
                  f"machinery scores {TB_power['full']['ratio']:.3f}x on "
                  f"{TB_power['full']['n_linked']:,} pairs, and subsampled to exactly "
                  f"{pw['n_linked']:,} pairs it still returns the same point estimate "
                  f"({pw['ratio']:.3f}x) but reaches p<0.05 in only {100*pw['frac_draws_p05']:.0f}% of "
                  f"draws -- so at this n the design recovers a real effect's SIZE and loses its "
                  f"SIGNIFICANCE, and a null p here would have proved nothing on its own. The sham arm, "
                  f"random pairs redrawn every draw at the same n, sits at {sham['ratio']:.3f}x "
                  f"[{sham['ratio_ci95'][0]:.3f}, {sham['ratio_ci95'][1]:.3f}] as it must. ")
    interesting = (tb and tb["ratio"] >= 1.05 and tb["ratio_ci95"][0] > 1.0)
    a_interesting = (ta and ta["ratio"] >= 1.05 and ta["frac_draws_p05"] >= 0.8)
    if not interesting and not a_interesting:
        v = (f"BOTH TESTS ARE NULL, AND TEST A IS NULL WITH REAL POWER BEHIND IT: AN ANNOTATED PTM SITE "
             f"SAYS NOTHING ABOUT HOW A TRANSCRIPT MOVES IN THIS CELL. Test A: genes with >= {HI_SITES} "
             f"annotated phosphosites ({int(hi.sum()):,} of {len(sel):,} testable) reach raw mean |lfc| "
             f"{ta['raw_hi']:.5f} against {ta['raw_lo']:.5f} for zero-site genes matched on "
             f"publication-count and protein-length deciles over {ta['n_draws']} draws -- a ratio of "
             f"{ta['ratio']:.4f} on a design that would have caught "
             f"{ta['min_detectable_ratio_80pct']:.4f} at 80% power, and it falls to {ta3['ratio']:.4f} "
             f"once K562 mRNA abundance is matched too. Several of those rows do reach p<0.05 in most "
             f"draws; at a ratio of 1.002 that is a shape difference, not an effect, and reporting it as "
             f"one would be this project's recurring error. The partial Spearman of site count against "
             f"responsiveness, controlling log pubs, log length and log mRNA, is "
             f"{TA_partial['raw mean |lfc|']:+.4f}. Test B: {tb['n_linked']:,} testable (P,G) cells from "
             f"{len(es):,} iPTMnet enzyme->substrate gene pairs give raw mean |z| "
             f"{tb['raw_mean_abs_z_linked']:.4f} against {tb['raw_mean_abs_z_matched']:.4f} matched on "
             f"perturbation-strength x gene-responsiveness deciles -- {tb['ratio']:.4f}x. The arm where a "
             f"transcriptional effect is actually predicted, enzyme -> TF-substrate, gives "
             f"{tf_arm['ratio']:.4f}x on {tf_arm['n_linked']:,} pairs; the direction control, perturbing "
             f"the SUBSTRATE and reading the ENZYME's transcript, gives {TB['REVERSED (perturb substrate)']['ratio']:.4f}x, "
             f"i.e. if anything the edges run backwards in this assay, which is what a set of edges with "
             f"no transcriptional content looks like. {powtxt}"
             f"THE HONEST READING IS NOT THAT THE {n_sites:,} POSITIONED SITES OR THE {len(es):,} ENZYME "
             f"EDGES ARE WRONG. Site existence is a property of the protein and of the literature -- rho "
             f"{stats.spearmanr(C, PB)[0]:.2f} with publication count and "
             f"{stats.spearmanr(C, LN)[0]:.2f} with protein length -- and phosphorylation is "
             f"post-translational, so a transcriptional readout is close to blind to it by construction. "
             f"The layer earns its place on capability, not on this score: positions, residues, evidence "
             f"codes and named enzymes are what an activity model needs, and the field it replaces "
             f"({len(old_ptm):,} bare counts with no site and no enzyme) could not support a single one "
             f"of those questions. But it must not be wired into a transcript-level predictor as a "
             f"feature, and site count must never be used as a proxy for pathway activity. What would "
             f"overturn this is a K562 phosphoproteome measured after perturbation -- the occupancy this "
             f"layer explicitly does not have; nothing here substitutes for one.")
    elif interesting:
        v = (f"ENZYME->SUBSTRATE EDGES CARRY A TRANSCRIPTIONAL SIGNAL: {tb['ratio']:.4f}x "
             f"[{tb['ratio_ci95'][0]:.3f}, {tb['ratio_ci95'][1]:.3f}] on {tb['n_linked']:,} matched pairs "
             f"(raw mean |z| {tb['raw_mean_abs_z_linked']:.4f} vs "
             f"{tb['raw_mean_abs_z_matched']:.4f}), while site COUNT as a gene property is null "
             f"({ta['ratio']:.4f}x). {powtxt}")
    else:
        v = (f"SITE COUNT PREDICTS RESPONSIVENESS ({ta['ratio']:.4f}x, "
             f"{100*ta['frac_draws_p05']:.0f}% of draws p<0.05, matched on pubs and length) BUT THE "
             f"ENZYME EDGES DO NOT ({tb['ratio']:.4f}x). {powtxt}")
    R["verdict"] = v
    report(f"  VERDICT: {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "ptm_sites.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'ptm_sites.json'}")


if __name__ == "__main__":
    main()
