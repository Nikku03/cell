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


# =================================================================================================
# the test
# =================================================================================================

def prepare(min_tpm=1.0):
    """log2(TPM+1), then CENTRE EACH GENE ACROSS THE PANEL, per replicate independently.

    Centring per replicate matters: doing it once on the pooled data would leak replicate 2 into
    the predictor built from replicate 1, which is the same class of defect this build order has
    caught in its own gates four times."""
    genes, tfs, r1, r2, sha = load_panel()
    L1 = np.log2(np.nan_to_num(r1, nan=0.0) + 1.0)
    L2 = np.log2(np.nan_to_num(r2, nan=0.0) + 1.0)
    expressed = (np.nanmean(np.nan_to_num(r1, nan=0.0), axis=0) >= min_tpm)
    D1 = L1 - L1.mean(axis=0, keepdims=True)
    D2 = L2 - L2.mean(axis=0, keepdims=True)
    return genes, tfs, D1, D2, expressed, sha


def regulator_index(genes, tfs):
    """For each carried gene, which perturbed factors TRRUST calls its regulators, and with what
    sign. A gene with none is not a test of anything the engine does."""
    from rem.atlas.humantransfer import symbol_map
    s2e, e2s, _ = symbol_map()
    tfi = {t: i for i, t in enumerate(tfs)}
    gsym = [e2s.get(g, "") for g in genes]
    E, _ = signed_edges()
    regs = collections.defaultdict(list)
    for u, v, m in E:
        if u in tfi:
            regs[v].append((tfi[u], m))
    out = {}
    for j, s in enumerate(gsym):
        if s and s in regs and len(regs[s]) >= 1:
            out[j] = regs[s]
    return out, gsym


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("DOES THE YEAST PROMOTER RESULT TRANSFER TO HUMAN CELLS?"); P_(RULE)
    genes, tfs, D1, D2, expressed, sha_h = prepare()
    regs, gsym = regulator_index(genes, tfs)
    P_(f"  ENCODE CRISPRi RNA-seq, K562. {len(tfs)} transcription factors silenced, each with two")
    P_(f"  biological replicates. HGNC map sha256[:32] {sha_h}.")
    E, _ = signed_edges()
    P_(f"  TRRUST provides the regulator sets; {len(genes)} of its genes are quantified here.")

    keep = {j: r for j, r in regs.items() if expressed[j] and len(r) >= 2}
    P_(f"  carried: {len(keep)} target genes that are EXPRESSED and have >= 2 of these factors as")
    P_(f"  TRRUST regulators; {sum(1 for r in keep.values() if len(r) >= 3)} have >= 3,"
       f" most-regulated has {max((len(r) for r in keep.values()), default=0)}.")

    pairs = [(j, ti, m) for j, r in keep.items() for ti, m in r]
    y1 = np.array([D1[ti, j] for j, ti, m in pairs])
    y2 = np.array([D2[ti, j] for j, ti, m in pairs])
    P_(f"  {len(pairs)} (gene, regulator) pairs measured twice each.")

    # ---- H1  NOISE FLOOR, FIRST ----------------------------------------------------------------
    P_("\n" + RULE); P_("H1  THE REPLICATE NOISE FLOOR, MEASURED BEFORE ANY MODEL"); P_(RULE)
    dif = y1 - y2
    sigma = float(np.std(dif)) / np.sqrt(2.0)
    P_(f"  RMS replicate-to-replicate difference in the centred response : {np.std(dif):.4f} log2")
    P_(f"  NOISE FLOOR, the noise in ONE replicate's response            : {sigma:.4f} log2")
    P_( "  The quantity every model below is scored against is a SINGLE replicate's response, so")
    P_( "  the floor is the single-replicate noise. Stated explicitly because the yeast module got")
    P_( "  exactly this wrong -- it scored against a two-replicate MEAN while dividing by the")
    P_( "  single-replicate figure -- and had to be corrected by an audit.")
    P_(f"  spread of the response itself: {float(np.std(y2)):.4f} log2"
       f"   (signal-to-noise {float(np.std(y2))/sigma:.2f})")

    def rmse(pred):
        return float(np.sqrt(np.mean((y2 - pred) ** 2)))

    # ---- H2  THE BASELINE THAT MUST FAIL -------------------------------------------------------
    P_("\n" + RULE); P_("H2  THE BASELINE THAT MUST FAIL"); P_(RULE)
    e_zero = rmse(np.zeros_like(y2))
    gmean = {}
    for j, r in keep.items():
        gmean[j] = float(np.mean([D1[ti, j] for ti, m in r]))
    e_blind = rmse(np.array([gmean[j] for j, ti, m in pairs]))
    e_ident = rmse(y1)
    P_(f"    {'model':<44} {'held-out RMSE':>14} {'floors vs identity':>19}")
    P_(f"    {'no information at all (predict zero)':<44} {e_zero:>14.4f}"
       f" {(e_zero-e_ident)/sigma:>19.2f}")
    P_(f"    {'the GENE only, blind to which regulator':<44} {e_blind:>14.4f}"
       f" {(e_blind-e_ident)/sigma:>19.2f}")
    P_(f"    {'the (gene, regulator) pair -- IDENTITY':<44} {e_ident:>14.4f} {0.0:>19.2f}")
    h2 = (e_zero - e_blind) > sigma
    P_(f"\n  H2: {'PASS -- knowing the gene beats knowing nothing, so these data carry regulation' if h2 else 'FAIL -- knowing the gene buys nothing; these data are not measuring regulation and nothing below is readable'}")
    P_( "  Note the identity model predicts one replicate from the other, so its error carries TWO")
    P_(f"  replicates' noise: sqrt(2)*sigma = {np.sqrt(2)*sigma:.4f}, which is what it measures")
    P_( "  against. It is a reference, not a fitted model, and no model here can beat it.")

    # ---- H3  THE HEAD TO HEAD ------------------------------------------------------------------
    P_("\n" + RULE); P_("H3  DOES IT MATTER WHICH REGULATOR YOU PERTURB?"); P_(RULE)
    gap = e_blind - e_ident
    P_(f"  regulator-blind RMSE  {e_blind:.4f}")
    P_(f"  identity RMSE         {e_ident:.4f}")
    P_(f"  identity advantage    {gap:+.4f} log2   against a noise floor of {sigma:.4f}")
    P_(f"  ratio to floor        {gap/sigma:.2f}")
    if gap > sigma:
        P_("\n  H3: THE YEAST RESULT TRANSFERS. In human cells, as in yeast, a gene's regulators are")
        P_(f"  NOT interchangeable -- knowing which one was perturbed is worth {gap/sigma:.2f} noise")
        P_("  floors. A count summary, which by construction cannot tell them apart, is refuted on")
        P_("  human data too.")
    else:
        P_("\n  H3: THE YEAST RESULT DOES NOT TRANSFER. On human data a gene's regulators are")
        P_(f"  interchangeable to within {gap/sigma:.2f} of the noise floor, which is what a count")
        P_("  summary assumes. Every accuracy figure in this build order rests on a yeast")
        P_("  experiment that does not describe this network, and the caps that depend on them")
        P_("  must be recomputed or withdrawn.")

    # ---- H4  THE CLASS LADDER ------------------------------------------------------------------
    P_("\n" + RULE); P_("H4  THE CLASS LADDER: HOW MANY CLASSES DOES HUMAN NEED?"); P_(RULE)
    P_("  The direct analogue of signed.py's S1. Partition the factors into C classes by their")
    P_("  fitted effect and predict from the class mean. Classes are fitted on replicate 1 only;")
    P_("  replicate 2 is never seen by the partition. C = 1 is regulator-blind, C = all is identity.")
    # each factor's fitted effect: its mean centred response across the genes TRRUST says it
    # regulates, taken from REPLICATE 1 ONLY, so replicate 2 never touches the partition.
    eff = np.zeros(len(tfs))
    for ti in range(len(tfs)):
        vals = [D1[ti, j] for j, r in keep.items() if any(t == ti for t, _ in r)]
        eff[ti] = float(np.mean(vals)) if vals else 0.0
    order = np.argsort(eff)
    P_(f"\n    {'C':>5} {'held-out RMSE':>14} {'floors from identity':>21} {'% of gap closed':>16}")
    ladder = []
    CS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, len(tfs)]
    for C in CS:
        if C < 1:
            continue
        cls = np.zeros(len(tfs), dtype=int)
        for b, grp in enumerate(np.array_split(order, min(C, len(tfs)))):
            cls[grp] = b
        cm = {}
        for j, r in keep.items():
            byc = collections.defaultdict(list)
            for ti, m in r:
                byc[cls[ti]].append(D1[ti, j])
            cm[j] = {c: float(np.mean(v)) for c, v in byc.items()}
        pred = np.array([cm[j].get(cls[ti], gmean[j]) for j, ti, m in pairs])
        e = rmse(pred)
        ladder.append((C, e))
        P_(f"    {C:>5} {e:>14.4f} {(e-e_ident)/sigma:>21.2f}"
           f" {100*(e_blind-e)/max(gap,1e-12):>15.1f}%")
    inside = [C for C, e in ladder if (e - e_ident) <= sigma]
    smallest = min(inside) if inside else None
    P_(f"\n  smallest C inside one noise floor of identity: {smallest if smallest else 'NONE of those tested'}")
    if smallest:
        P_(f"  as a FRACTION of the alphabet: {smallest}/{len(tfs)} = {100*smallest/len(tfs):.1f}%")
    P_(f"  yeast needed 64 of 404 = 15.8%. That fraction is the number that decides whether the")
    P_( "  caps in this build order transfer, because a cap is only meaningful at a stated accuracy.")

    # ---- H5  TRRUST's OWN SIGN AS THE CLASS ----------------------------------------------------
    P_("\n" + RULE); P_("H5  TRRUST's OWN SIGN AS THE CLASS -- THE ONE LABEL THAT IS NOT FITTED")
    P_(RULE)
    modes = sorted({m for j, ti, m in pairs})
    mcm = {}
    for j, r in keep.items():
        bym = collections.defaultdict(list)
        for ti, m in r:
            bym[m].append(D1[ti, j])
        mcm[j] = {m: float(np.mean(v)) for m, v in bym.items()}
    e_mode = rmse(np.array([mcm[j].get(m, gmean[j]) for j, ti, m in pairs]))
    P_(f"  modes present: {', '.join(f'{m} ({sum(1 for p in pairs if p[2]==m)})' for m in modes)}")
    P_(f"    {'regulator-blind':<40} {e_blind:>10.4f} {(e_blind-e_ident)/sigma:>8.2f} floors")
    P_(f"    {'TRRUST sign as the class':<40} {e_mode:>10.4f} {(e_mode-e_ident)/sigma:>8.2f} floors")
    P_(f"    {'identity':<40} {e_ident:>10.4f} {0.0:>8.2f} floors")
    P_(f"  the signed count was refuted on yeast at 3.70 floors; here the real annotation leaves")
    P_(f"  {(e_mode-e_ident)/sigma:.2f} floors, and it closes"
       f" {100*(e_blind-e_mode)/max(gap,1e-12):.1f}% of the gap identity opens.")

    # ---- H6 / H7 -------------------------------------------------------------------------------
    P_("\n" + RULE); P_("H6  WHAT DIFFERS BETWEEN THE TWO EXPERIMENTS"); P_(RULE)
    P_("  1. Yeast measured SITES designed into a promoter; this measures PERTURBATIONS of factors")
    P_("     in their native network. A knockdown effect includes everything downstream of that")
    P_("     factor, not only its direct action on the gene -- so this test is, if anything,")
    P_("     biased TOWARD regulators looking interchangeable, since indirect effects are shared.")
    P_("  2. Yeast varied the number of sites; here exactly one factor is perturbed at a time, so")
    P_("     the count-versus-identity question becomes which-versus-whether, which is the half")
    P_("     that matters for the engine's per-gene factor.")
    P_("  3. TRRUST's regulator sets are literature-curated and incomplete, so a gene's true")
    P_("     regulator set is larger than the one used here.")
    P_("  4. One cell line, one assay, and CRISPRi silences transcription rather than removing")
    P_("     protein, so factors acting post-translationally are under-represented.")
    P_("\n" + RULE); P_("H7  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  It settles whether a HUMAN gene's response depends on WHICH of its regulators is hit,")
    P_("  on this panel, which is the assumption every count-like summary in this build order")
    P_("  makes. It does not settle the temporal half of the engine's summary, it does not")
    P_("  validate any cap directly, and it cannot say anything about factors absent from the")
    P_("  panel or edges absent from TRRUST.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_humantransfer.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
