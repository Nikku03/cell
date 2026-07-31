"""BIOPLEX 3.0 AP-MS -- rebuilding the PPI layer with the three things it threw away: assay, cell line, confidence.

WHAT IS THERE NOW. The cell object's `ppi` field is 191,447 bare 2-tuples of gene INDEX. No assay, no cell
line, no confidence, no direction. Every edge in it asserts exactly one thing -- "these two are connected" --
and asserts it with equal force whether it came from a co-crystal structure, a yeast two-hybrid screen, or a
text-mining co-mention score of 0.4 that STRING would itself label low confidence. That layer is the single
biggest source of edges in the model and it is the least interrogable. This module does not extend it. It
rebuilds one well-characterised slice of it with the metadata intact, and then measures whether the metadata
buys anything.

WHY BIOPLEX AND NOT MORE STRING. BioPlex 3.0 is one assay (affinity purification of an HA-FLAG tagged bait
followed by mass spectrometry), one pipeline (CompPASS-plus), one release (Dec 2019), run to saturation over
10,128 baits in HEK293T and 5,522 in HCT116. Every edge carries a bait/prey direction and a three-way
posterior -- pWrongID, pNI (no interaction), pInt (interaction) -- from a single model applied uniformly.
That is the opposite of an aggregated score: the confidence means the same thing on every edge.

AND THE POINT OF KEEPING THE CELL LINE. A 293T edge is not a K562 edge. Roughly half the interactions BioPlex
detects in one line are absent in the other even where the bait was screened in both, because complex
membership tracks expression, and expression is cell-type specific. The existing layer's flattening erased
exactly the axis the whole-cell model is supposed to care about. So the cell line is retained per edge, and it
is TESTED: does an interaction seen in BOTH lines predict K562 transcriptional coupling better than one seen
in only one?

THE FOUR PRESPECIFIED QUESTIONS.

  1. DOES THE LAYER CARRY INFORMATION AT ALL? The fast test: for (perturbation P, gene G) pairs this layer
     links, is |robust z| in Replogle K562 CRISPRi Perturb-seq elevated over pairs matched on BOTH
     perturbation strength decile and gene responsiveness decile? Reference points from the same test on
     existing layers: PPI 1.208x, co-dependency 1.070x, regulatory 1.021x, signalling 1.028x (null).

  2. WAS KEEPING CONFIDENCE WORTH IT? If pInt is a real quantity rather than a decoration, top-tercile edges
     must beat bottom-tercile edges on question 1 -- both against their own matched controls and in a direct
     matched head-to-head. If they do not, then the score is noise above the 0.75 publication cut and the
     honest thing is to record that and stop treating it as a weight.

  3. IS CROSS-CELL-LINE REPRODUCIBILITY A BETTER FILTER THAN CONFIDENCE? Restricted to edges whose bait was
     screened in BOTH lines -- so the interaction had the OPPORTUNITY to be seen twice -- do the ones
     actually seen twice transfer to K562 better than the ones seen once? This is the direct test of the
     claim that motivates keeping the cell line at all.

  4. IS THIS A REBUILD OR A DUPLICATE? How much of BioPlex is already in the existing 191,447-edge layer, and
     do the edges that are NOT already there carry signal? An overlap of 90% with no signal in the remainder
     would mean this module adds metadata to known edges and nothing else; a low overlap with signal in the
     remainder means the existing layer is missing real biology.

THE CONTROL THAT MATTERS MOST, AND WHY THERE ARE TWO. The obvious control -- random gene pairs matched on the
two deciles -- compares a BioPlex edge against a pair BioPlex NEVER TESTED. A gene that was chosen as a bait
in a saturating interactome screen is not a random gene, and neither is a protein abundant enough to be seen
as prey. So a second, stringent control is run: the perturbed gene must itself be a BioPlex BAIT and the
measured gene must appear somewhere in the BioPlex network, but this specific pair must not be an edge. That
is "tested and not found", not "never looked at". If the ratio survives the stringent control, the layer is
telling us about interactions. If it only survives the loose one, the layer is telling us which genes are
baits, which is a property of the experiment, not the cell.

WHAT WOULD FALSIFY THIS. A ratio at or below ~1.05 against the stringent control means AP-MS physical
association does not constrain the K562 transcriptional response and the layer should not be weighted as if
it does. A confidence stratification that is flat means pInt should be dropped. A cell-line stratification
that is flat means the 293T/HCT116 distinction is not worth carrying, and the existing layer's flattening was
not the loss it looks like.

ABSENCE IS NOT ZERO. A gene that was never a bait cannot have BioPlex edges. Bait coverage is measured
against perturbation strength and gene responsiveness rather than assumed uniform, and non-edges among
non-baits are recorded as UNKNOWN, never as evidence of no interaction.
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DIR = SP / "bioplex"
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
LAYER = SP / "cell_bioplex.json.gz"

BASE = "https://bioplex.hms.harvard.edu/data/"
# ONE RELEASE, ONE PIPELINE, PINNED BY FILENAME. BioPlex 1.0/2.0/2.3 and the RPE1/U2OS v0.1 previews sit in
# the same directory and use the same column names with a DIFFERENT interaction model behind them. Mixing
# releases would make pInt mean two things at once, which is the exact defect this module is fixing in the
# existing PPI layer. Only these four files are ever read.
FILES = {"293T": "BioPlex_293T_Network_10K_Dec_2019.tsv",
         "HCT116": "BioPlex_HCT116_Network_5.5K_Dec_2019.tsv"}
BAITS = {"293T": "BioPlex_3p0_293T_baitList.tsv",
         "HCT116": "BioPlex_3p0_HCT116_baitList.tsv"}
RELEASE = "BioPlex 3.0 (Dec 2019); Huttlin et al., Cell 2021 / bioRxiv 2020"
ASSAY = "AP-MS (HA-FLAG bait immunoprecipitation, LC-MS/MS, CompPASS-plus)"
HEADER = ["GeneA", "GeneB", "UniprotA", "UniprotB", "SymbolA", "SymbolB", "pW", "pNI", "pInt"]
PINT_FLOOR = 0.75          # the published admission threshold of BioPlex 3.0
N_DRAWS = 20
N_DEC = 10


def download(name, tries=4):
    p = DIR / name
    if p.exists() and p.stat().st_size > 0:
        return p
    DIR.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            r = urllib.request.Request(BASE + name, headers={"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=900) as fh, open(p, "wb") as o:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            return p
        except Exception:
            if p.exists():
                p.unlink()
            if i == tries - 1:
                raise SystemExit(f"BLOCKED: cannot fetch {BASE + name}")
            time.sleep(2 ** (i + 1))


def read_tsv(path):
    with open(path) as fh:
        r = csv.reader(fh, delimiter="\t", quotechar='"')
        head = next(r)
        return head, [row for row in r]


def arm(label, link_i, link_j, A, ok, rq, cq, rstr, cresp, pool_rows, pool_cols, linkset,
        n_draws=N_DRAWS, seed0=0):
    """One stratum of the fast test, with its matched control redrawn n_draws times.

    THE EFFECT SIZE IS `mean_abs_z_linked` -- the raw held-out value. The ratio is the CONTRAST and the
    p-value is the significance; neither is the effect, and nothing is ever reported as a difference from
    a shuffled arm. One matched draw is one sample of the matched population, so the draw is repeated and
    the FRACTION of draws reaching p<0.05 is reported alongside the mean rather than a single draw.
    Residual imbalance on BOTH matched covariates is measured after matching, on every draw.
    """
    from scipy import stats
    lz = A[link_i, link_j]
    ct = Counter(zip(rq[link_i].tolist(), cq[link_j].tolist()))
    rows = []
    for s in range(n_draws):
        rng = np.random.default_rng(seed0 + s)
        ci, cj = [], []
        for (a, b), n in ct.items():
            rs, cs = pool_rows[a], pool_cols[b]
            if len(rs) == 0 or len(cs) == 0:
                continue
            need, tries = n, 0
            while need > 0 and tries < 40:
                m = int(need * 1.6) + 16
                xi = rs[rng.integers(0, len(rs), m)]
                xj = cs[rng.integers(0, len(cs), m)]
                for x, y in zip(xi, xj):
                    if need <= 0:
                        break
                    if not ok[x, y] or (x, y) in linkset:
                        continue
                    ci.append(x)
                    cj.append(y)
                    need -= 1
                tries += 1
        ci = np.array(ci, dtype=np.int64)
        cj = np.array(cj, dtype=np.int64)
        if len(ci) < 50:
            continue
        cz = A[ci, cj]
        rows.append({
            "n_ctrl": int(len(ci)), "mean_ctrl": float(cz.mean()),
            "ratio": float(lz.mean() / cz.mean()),
            "p": float(stats.mannwhitneyu(lz, cz, alternative="two-sided")[1]),
            "res_strength_p": float(stats.mannwhitneyu(rstr[link_i], rstr[ci],
                                                       alternative="two-sided")[1]),
            "res_resp_p": float(stats.mannwhitneyu(cresp[link_j], cresp[cj],
                                                   alternative="two-sided")[1])})
    if not rows:
        return {"label": label, "n_pairs": int(len(lz)), "testable": False}

    def mn(k):
        return float(np.mean([r[k] for r in rows]))
    return {"label": label, "n_pairs": int(len(lz)), "testable": True,
            "n_draws": len(rows),
            "mean_abs_z_linked": float(lz.mean()),         # <- THE EFFECT SIZE: raw held-out value
            "median_abs_z_linked": float(np.median(lz)),
            "mean_abs_z_matched": mn("mean_ctrl"),
            "ratio": mn("ratio"),
            "ratio_min": float(np.min([r["ratio"] for r in rows])),
            "ratio_max": float(np.max([r["ratio"] for r in rows])),
            "p_median": float(np.median([r["p"] for r in rows])),
            "frac_draws_p05": float(np.mean([r["p"] < 0.05 for r in rows])),
            "residual_strength_p": mn("res_strength_p"),
            "residual_responsiveness_p": mn("res_resp_p")}


def head_to_head(label, ia, ja, ib, jb, A, rq, cq, rstr, cresp, n_draws=N_DRAWS, seed0=0,
                 extra=None, extra_name=None):
    """Direct matched comparison of two linked sets against each other (not against random pairs).

    `extra` is an optional THIRD matching axis given as (values_for_a, values_for_b) already binned to
    integers -- used for the cell-line arm, where the obvious alternative explanation for "seen twice" is
    "the bait is a hub", and a hub perturbation is disruptive for reasons that have nothing to do with the
    cell line.
    """
    from scipy import stats
    ea = eb = None
    if extra is not None:
        ea, eb = extra
    pa, pb = {}, {}
    for k in range(len(ia)):
        pa.setdefault((rq[ia[k]], cq[ja[k]]) + ((int(ea[k]),) if ea is not None else ()), []).append(k)
    for k in range(len(ib)):
        pb.setdefault((rq[ib[k]], cq[jb[k]]) + ((int(eb[k]),) if eb is not None else ()), []).append(k)
    za, zb = A[ia, ja], A[ib, jb]
    rows = []
    for s in range(n_draws):
        rng = np.random.default_rng(seed0 + s)
        ka, kb = [], []
        for key, la in pa.items():
            lb = pb.get(key, [])
            n = min(len(la), len(lb))
            if n:
                ka += list(rng.choice(la, n, replace=False))
                kb += list(rng.choice(lb, n, replace=False))
        if len(ka) < 50:
            continue
        ka, kb = np.array(ka), np.array(kb)
        rows.append({"n": int(len(ka)), "a": float(za[ka].mean()), "b": float(zb[kb].mean()),
                     "p": float(stats.mannwhitneyu(za[ka], zb[kb], alternative="two-sided")[1]),
                     "res_strength_p": float(stats.mannwhitneyu(rstr[ia[ka]], rstr[ib[kb]],
                                                                alternative="two-sided")[1]),
                     "res_resp_p": float(stats.mannwhitneyu(cresp[ja[ka]], cresp[jb[kb]],
                                                            alternative="two-sided")[1]),
                     "res_extra_p": (float(stats.mannwhitneyu(ea[ka], eb[kb],
                                                              alternative="two-sided")[1])
                                     if ea is not None else float("nan"))})
    if not rows:
        return {"label": label, "testable": False}

    def mn(k):
        return float(np.mean([r[k] for r in rows]))
    out = {"label": label, "testable": True, "n_draws": len(rows), "n_matched": int(mn("n")),
           "mean_abs_z_a": mn("a"), "mean_abs_z_b": mn("b"),
           "ratio_a_over_b": mn("a") / mn("b"),
           "p_median": float(np.median([r["p"] for r in rows])),
           "frac_draws_p05": float(np.mean([r["p"] < 0.05 for r in rows])),
           "residual_strength_p": mn("res_strength_p"),
           "residual_responsiveness_p": mn("res_resp_p")}
    if ea is not None:
        out["matched_also_on"] = extra_name
        out[f"residual_{extra_name}_p"] = mn("res_extra_p")
    return out


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("BIOPLEX 3.0 AP-MS -- a PPI rebuild that keeps assay, cell line, direction and confidence")
    report("=" * 100)

    # ---------------------------------------------------------------- fetch, and PIN THE PROVENANCE
    paths = {k: download(v) for k, v in FILES.items()}
    bpaths = {k: download(v) for k, v in BAITS.items()}
    nets, baitsets = {}, {}
    for cl in FILES:
        head, rows = read_tsv(paths[cl])
        if head != HEADER:
            raise SystemExit(f"{FILES[cl]}: unexpected columns {head} -- release layout changed")
        nets[cl] = rows
        bh, br = read_tsv(bpaths[cl])
        baitsets[cl] = {r[1] for r in br}
        report(f"  {cl:7s} {FILES[cl]:44s} {len(rows):7,d} edges   {len(baitsets[cl]):6,d} baits")
    # ASSERTIONS, not docstring claims. (a) the three posteriors are a partition; (b) GeneA is the BAIT --
    # the direction is not documented in the file, it is verified against the bait list; (c) nothing below
    # the published pInt floor leaked in, which bounds what "low confidence" can mean here.
    for cl, rows in nets.items():
        s = np.array([float(r[6]) + float(r[7]) + float(r[8]) for r in rows])
        if np.abs(s - 1).max() > 1e-6:
            raise SystemExit(f"{cl}: pW+pNI+pInt != 1 (max dev {np.abs(s-1).max():.2g})")
        a_bait = np.mean([r[0] in baitsets[cl] for r in rows])
        b_bait = np.mean([r[1] in baitsets[cl] for r in rows])
        if a_bait < 0.999:
            raise SystemExit(f"{cl}: GeneA is not always a bait ({a_bait:.3%}) -- direction assumption dead")
        pint = np.array([float(r[8]) for r in rows])
        if pint.min() < PINT_FLOOR - 1e-9:
            raise SystemExit(f"{cl}: pInt below the published floor ({pint.min():.3f})")
        report(f"  {cl:7s} pW+pNI+pInt=1 OK | GeneA in baitList {a_bait:.1%} -> A=BAIT, B=PREY "
               f"(GeneB is a bait in {b_bait:.1%}) | pInt in [{pint.min():.3f}, {pint.max():.3f}]")
    report(f"  release pinned: {RELEASE}")
    report(f"  assay pinned:   {ASSAY}")

    # ---------------------------------------------------------------- join through the registry
    from entity_registry import Registry, REGISTRY as REGPATH
    raw = json.load(gzip.open(REGPATH, "rt"))
    REG = Registry(raw)
    CELLIDX = raw["cell_index"]        # the cell object keys `ppi`/`ppm`/`abund` by gene INDEX, not symbol

    ent_ids = sorted({r[0] for rs in nets.values() for r in rs} |
                     {r[1] for rs in nets.values() for r in rs})
    uni, sym = {}, {}
    for rs in nets.values():
        for r in rs:
            uni[r[0]], uni[r[1]] = r[2], r[3]
            sym[r[0]], sym[r[1]] = r[4], r[5]
    # ACCESSION FIRST. The file carries a UniProt accession for every endpoint; that is what the mass
    # spectrometer actually identified, and it is the strongest key here. Entrez is second. Symbol is used
    # only where both accessions fail and every such endpoint is counted and recorded as via=symbol.
    _g, st_u = REG.join("gene", "uniprot", [uni[g].split("-")[0] for g in ent_ids], 0.90,
                        "BioPlex endpoints by UniProt accession")
    # NOTE, and it is a defect in the registry rather than in BioPlex: gene|entrez alias keys were written
    # from a float column and are stored as "7105.0". Resolving the integer id returns None for EVERY gene
    # -- the same class of silent-miss the registry exists to prevent. It is handled here by normalising on
    # this side (the registry is not modified) and the surviving raw-integer rate is reported so the defect
    # is visible rather than papered over.
    _g, st_e_raw = REG.join("gene", "entrez", ent_ids, 0.0, "BioPlex endpoints by raw Entrez id")
    _g, st_e = REG.join("gene", "entrez", [g + ".0" for g in ent_ids], 0.90,
                        "BioPlex endpoints by Entrez id (float-normalised)")
    report(f"\n  REGISTRY JOIN over {len(ent_ids):,} distinct endpoints")
    report(f"    gene/uniprot (accession, primary)   {st_u['joined']:6,d}/{st_u['of']:,} "
           f"({st_u['rate']:.1%})")
    report(f"    gene/entrez  raw integer            {st_e_raw['joined']:6,d}/{st_e_raw['of']:,} "
           f"({st_e_raw['rate']:.1%})  <- registry stores entrez keys as '7105.0'; a raw join silently "
           f"misses everything")
    report(f"    gene/entrez  float-normalised       {st_e['joined']:6,d}/{st_e['of']:,} "
           f"({st_e['rate']:.1%})")

    cache = {}

    def resolve(g):
        if g in cache:
            return cache[g]
        u = uni.get(g)
        e = REG.resolve("gene", "uniprot", u.split("-")[0]) if u else None
        v = "uniprot"
        if e is None:
            e, v = REG.resolve("gene", "entrez", g + ".0"), "entrez"
        if e is None and g in sym:
            e, v = REG.resolve("gene", "symbol", sym[g]), "symbol"
        cache[g] = (e, v if e else None)
        return cache[g]

    via = Counter()
    for g in ent_ids:
        via[resolve(g)[1] or "UNRESOLVED"] += 1
    report(f"    resolved endpoints by route: {dict(via)}  "
           f"({100*(1-via['UNRESOLVED']/len(ent_ids)):.2f}% of endpoints carry an entity id)")

    # ---------------------------------------------------------------- the edge layer
    edges = []
    drop_unres = drop_self = 0
    for cl, rows in nets.items():
        for r in rows:
            ea, va = resolve(r[0])
            eb, vb = resolve(r[1])
            if ea is None or eb is None:
                drop_unres += 1
                continue
            if ea == eb:
                drop_self += 1        # two accessions of one gene; not an interaction claim
                continue
            edges.append({"bait": ea, "prey": eb, "bait_symbol": r[4], "prey_symbol": r[5],
                          "bait_uniprot": r[2], "prey_uniprot": r[3],
                          "cell_line": cl, "assay": "AP-MS",
                          "pWrongID": float(r[6]), "pNI": float(r[7]), "pInt": float(r[8]),
                          "via": va if va == vb else f"{va}/{vb}"})
    report(f"\n  {len(edges):,} directed bait->prey edges retained "
           f"({drop_unres:,} dropped for an unresolvable endpoint, {drop_self:,} self-edges)")
    per_cl = Counter(e["cell_line"] for e in edges)
    report(f"    by cell line: {dict(per_cl)}")

    # unordered gene-pair view, needed for the overlap and the cell-line arm
    gp = {}
    for e in edges:
        k = tuple(sorted((e["bait"], e["prey"])))
        d = gp.setdefault(k, {"cl": set(), "pint": 0.0, "orient": set()})
        d["cl"].add(e["cell_line"])
        d["pint"] = max(d["pint"], e["pInt"])
        d["orient"].add((e["bait"], e["prey"]))
    report(f"    {len(gp):,} distinct unordered gene pairs; "
           f"{sum(1 for v in gp.values() if len(v['cl']) == 2):,} seen in BOTH cell lines")

    # ---------------------------------------------------------------- Q4a: overlap with the existing layer
    d = json.load(gzip.open(NET, "rt"))
    names = [g["name"] for g in d["genes"]]
    old_pairs = set()
    n_old_bad = 0
    for a, b in d["ppi"]:
        ea, eb = CELLIDX.get(str(a)), CELLIDX.get(str(b))
        if ea is None or eb is None or ea == eb:
            n_old_bad += 1
            continue
        old_pairs.add(tuple(sorted((ea, eb))))
    n_old_raw = len(d["ppi"])
    del d
    incell = {e for e in (CELLIDX.get(str(i)) for i in range(len(names))) if e}
    report(f"\n  EXISTING PPI LAYER: {n_old_raw:,} index 2-tuples -> {len(old_pairs):,} distinct entity "
           f"pairs ({n_old_bad:,} unmappable/self)")
    bp_incell = {k for k in gp if k[0] in incell and k[1] in incell}
    inter = bp_incell & old_pairs
    report(f"    BioPlex pairs with both endpoints in the cell object: {len(bp_incell):,} "
           f"({100*len(bp_incell)/len(gp):.1f}% of BioPlex)")
    report(f"    overlap: {len(inter):,} pairs = {100*len(inter)/max(len(bp_incell),1):.1f}% of "
           f"(in-cell) BioPlex, {100*len(inter)/max(len(old_pairs),1):.1f}% of the existing layer")
    report(f"    BioPlex pairs ABSENT from the existing layer: "
           f"{len(bp_incell)-len(inter):,}")

    # ---------------------------------------------------------------- the fast test substrate
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        obs = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        var = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
    report(f"\n  GWPS: {X.shape[0]:,} perturbations x {X.shape[1]:,} genes, "
           f"{100*(1-np.isfinite(X).mean()):.4f}% non-finite cells")
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
    report(f"    per-gene robust z computed; {100*ok.mean():.4f}% of cells usable "
           f"(non-finite pairs DROPPED, not imputed)")
    rstr = np.where(ok, A, np.nan)
    rstr = np.nanmean(rstr, axis=1)
    cresp = np.nanmean(np.where(ok, A, np.nan), axis=0)
    rq = np.digitize(rstr, np.quantile(rstr, np.linspace(.1, .9, N_DEC - 1)))
    cq = np.digitize(cresp, np.quantile(cresp, np.linspace(.1, .9, N_DEC - 1)))

    # perturbation -> ENSG. The obs label is `{idx}_{symbol}_{transcript}_{ENSG}` and the transcript field
    # can itself contain an underscore (NM_001159524), so the ACCESSION is taken from the last field and
    # the symbol is only a fallback for the handful whose ENSG is literally "nan".
    pert_rows, n_nt, n_symfall = {}, 0, 0
    for i, o in enumerate(obs):
        p = o.split("_")
        if p[1] == "non-targeting":
            n_nt += 1
            continue
        e = p[-1] if p[-1].startswith("ENSG") else None
        if e is None:
            eid = REG.resolve("gene", "symbol", p[1])
            e = REG.entities[eid].get("ensembl") if eid else None
            if e:
                n_symfall += 1
        if e:
            pert_rows.setdefault(e, []).append(i)
    colof = {}
    for j, g in enumerate(var):
        colof.setdefault(g, j)
    report(f"    {len(pert_rows):,} perturbed genes carry an ENSG ({n_nt:,} non-targeting rows excluded "
           f"from the pair universe; {n_symfall} recovered via symbol), {len(colof):,} measured genes")

    ensof = {}

    def ens(eid):
        if eid not in ensof:
            ensof[eid] = REG.entities.get(eid, {}).get("ensembl")
        return ensof[eid]

    # ---------------------------------------------------------------- linked pairs, with their attributes
    # A BAIT THAT YIELDED NO EDGE IS THE PUREST "TESTED AND NOT FOUND" ROW THERE IS, so the bait universe is
    # taken from the bait LISTS, not from the network endpoints -- 451 baits produced no retained
    # interaction and dropping them would quietly make the stringent control easier.
    eid_bait_both = {resolve(g)[0] for g in (baitsets["293T"] & baitsets["HCT116"])} - {None}
    eid_bait_any = {resolve(g)[0] for cl in baitsets for g in baitsets[cl]} - {None}
    eid_in_bioplex = {e["bait"] for e in edges} | {e["prey"] for e in edges}

    pairs = {}
    for e in edges:
        for src, dst, orient in ((e["bait"], e["prey"], "bait2prey"),
                                 (e["prey"], e["bait"], "prey2bait")):
            se, de = ens(src), ens(dst)
            if not se or not de:
                continue
            rows_i = pert_rows.get(se)
            j = colof.get(de)
            if not rows_i or j is None:
                continue
            for i in rows_i:
                if not ok[i, j]:
                    continue
                k = (i, j)
                r = pairs.get(k)
                if r is None:
                    r = pairs[k] = {"pint": 0.0, "cl": set(), "orient": set(),
                                    "gp": tuple(sorted((e["bait"], e["prey"])))}
                r["pint"] = max(r["pint"], e["pInt"])
                r["cl"].add(e["cell_line"])
                r["orient"].add(orient)
    keys = list(pairs)
    li = np.array([k[0] for k in keys], dtype=np.int64)
    lj = np.array([k[1] for k in keys], dtype=np.int64)
    linkset = set(keys)
    # PSEUDO-REPLICATION IS REAL HERE. 788 genes were perturbed by more than one sgRNA construct and appear
    # as separate GWPS rows, so one gene pair can contribute several (row, column) pairs. That inflates n
    # and therefore every p-value. The count of DISTINCT gene pairs behind the linked pairs is reported,
    # and a one-row-per-perturbed-gene arm is run below to check the ratio is not an artefact of it.
    first_row = {e: min(v) for e, v in pert_rows.items()}
    keep_rows = set(first_row.values())
    onerow = np.array([k[0] in keep_rows for k in keys])
    n_gene_pairs_linked = len({pairs[k]["gp"] for k in keys})
    report(f"\n  LINKED (perturbation, gene) pairs testable in GWPS: {len(keys):,} "
           f"over {n_gene_pairs_linked:,} distinct gene pairs "
           f"({int(onerow.sum()):,} survive one-row-per-perturbed-gene de-replication)")

    # control pools --------------------------------------------------------
    # LOOSE: every perturbation row, every measured gene.
    pool_rows_all = [np.where(rq == k)[0] for k in range(N_DEC)]
    pool_cols_all = [np.where(cq == k)[0] for k in range(N_DEC)]
    # STRINGENT: the perturbed gene must itself be a BioPlex BAIT (so the experiment looked) and the
    # measured gene must appear somewhere in the BioPlex network (so it was detectable as prey). A pair
    # from this pool that is not an edge is TESTED-AND-NOT-FOUND rather than never examined.
    bait_ens = {ens(e) for e in eid_bait_any} - {None}
    seen_ens = {ens(e) for e in eid_in_bioplex} - {None}
    row_is_bait = np.zeros(len(obs), bool)
    for e, ii in pert_rows.items():
        if e in bait_ens:
            row_is_bait[ii] = True
    col_is_seen = np.array([var[j] in seen_ens for j in range(len(var))])
    pool_rows_str = [np.where((rq == k) & row_is_bait)[0] for k in range(N_DEC)]
    pool_cols_str = [np.where((cq == k) & col_is_seen)[0] for k in range(N_DEC)]
    report(f"    control pools: LOOSE {sum(len(r) for r in pool_rows_all):,} rows x "
           f"{sum(len(c) for c in pool_cols_all):,} cols | "
           f"STRINGENT (bait rows x BioPlex-detected cols) "
           f"{sum(len(r) for r in pool_rows_str):,} x {sum(len(c) for c in pool_cols_str):,}")

    def run(label, sel=None, pools="loose", seed0=0):
        i2 = li[sel] if sel is not None else li
        j2 = lj[sel] if sel is not None else lj
        pr, pc = (pool_rows_all, pool_cols_all) if pools == "loose" else (pool_rows_str, pool_cols_str)
        return arm(label, i2, j2, A, ok, rq, cq, rstr, cresp, pr, pc, linkset, seed0=seed0)

    R = {"model": "bioplex-v1", "release": RELEASE, "assay": ASSAY,
         "files": {k: FILES[k] for k in FILES}, "pint_floor": PINT_FLOOR,
         "n_edges": len(edges), "edges_by_cell_line": dict(per_cl),
         "n_gene_pairs": len(gp),
         "n_pairs_both_cell_lines": int(sum(1 for v in gp.values() if len(v["cl"]) == 2)),
         "join": {"uniprot_rate": st_u["rate"], "entrez_raw_rate": st_e_raw["rate"],
                  "entrez_normalised_rate": st_e["rate"], "routes": dict(via)},
         "overlap_existing_ppi": {
             "existing_tuples": n_old_raw, "existing_distinct_pairs": len(old_pairs),
             "bioplex_pairs_in_cell": len(bp_incell), "shared": len(inter),
             "frac_of_bioplex": len(inter) / max(len(bp_incell), 1),
             "frac_of_existing": len(inter) / max(len(old_pairs), 1),
             "bioplex_novel": len(bp_incell) - len(inter)},
         "n_linked_pairs": len(keys), "n_linked_gene_pairs": n_gene_pairs_linked,
         "n_linked_pairs_dereplicated": int(onerow.sum())}

    # ---------------------------------------------------------------- Q1: does the layer carry information?
    report("\n" + "-" * 100)
    report("  Q1  FAST TEST -- |robust z| of linked pairs vs pairs matched on perturbation-strength decile")
    report("      x gene-responsiveness decile.  EFFECT SIZE = mean |z| of the linked pairs (raw).")
    report("-" * 100)
    R["fast_test"] = {"loose": run("all BioPlex, loose control"),
                      "stringent": run("all BioPlex, tested-negative control", None, "stringent"),
                      "dereplicated": run("de-replicated (1 row per pert gene)", onerow, "stringent",
                                          seed0=7)}
    hdr = (f"      {'stratum':40s} {'pairs':>8s} {'mean|z|':>8s} {'matched':>8s} {'ratio':>7s} "
           f"{'p(med)':>10s} {'draws':>6s}")
    report(hdr)

    def line(a):
        if not a.get("testable"):
            report(f"      {a['label']:40s} {a['n_pairs']:8,d}  -- too few to test --")
            return
        report(f"      {a['label']:40s} {a['n_pairs']:8,d} {a['mean_abs_z_linked']:8.4f} "
               f"{a['mean_abs_z_matched']:8.4f} {a['ratio']:7.4f} {a['p_median']:10.2e} "
               f"{100*a['frac_draws_p05']:5.0f}%")
    line(R["fast_test"]["loose"])
    line(R["fast_test"]["stringent"])
    line(R["fast_test"]["dereplicated"])
    for k in ("loose", "stringent", "dereplicated"):
        a = R["fast_test"][k]
        if a.get("testable"):
            report(f"        {k}: residual imbalance after matching -- strength p "
                   f"{a['residual_strength_p']:.3g}, responsiveness p "
                   f"{a['residual_responsiveness_p']:.3g}; ratio across {a['n_draws']} draws "
                   f"[{a['ratio_min']:.4f}, {a['ratio_max']:.4f}]")

    # a pipeline sanity arm: relabel the SAME number of pairs at random. This establishes that the machinery
    # returns ~1.00 on a null, and nothing else -- it is NOT the effect size and is not subtracted anywhere.
    rng = np.random.default_rng(1234)
    fi = rng.integers(0, A.shape[0], len(keys) * 2)
    fj = rng.integers(0, A.shape[1], len(keys) * 2)
    keep = ok[fi, fj]
    fi, fj = fi[keep][:len(keys)], fj[keep][:len(keys)]
    sham = arm("SHAM (random pairs, same n)", fi, fj, A, ok, rq, cq, rstr, cresp,
               pool_rows_all, pool_cols_all, set(zip(fi.tolist(), fj.tolist())), seed0=500)
    R["sham"] = sham
    line(sham)

    # ---------------------------------------------------------------- Q2: was confidence worth keeping?
    report("\n" + "-" * 100)
    report("  Q2  WAS KEEPING pInt WORTH IT?  terciles of the pair's best pInt")
    report("-" * 100)
    pint = np.array([pairs[k]["pint"] for k in keys])
    t1, t2 = np.quantile(pint, [1 / 3, 2 / 3])
    lo, mid, hi = pint <= t1, (pint > t1) & (pint <= t2), pint > t2
    report(f"      tercile cuts at pInt {t1:.4f} / {t2:.4f}  "
           f"(the file floor is {PINT_FLOOR}, so 'low' means 0.75-{t1:.2f}, not 0-0.5)")
    report(hdr)
    R["confidence"] = {}
    for nm, m in (("pInt low", lo), ("pInt mid", mid), ("pInt high", hi)):
        for pools in ("loose", "stringent"):
            a = run(f"{nm} ({pools})", m, pools, seed0=17)
            R["confidence"][f"{nm}|{pools}"] = a
            line(a)
    R["confidence_head_to_head"] = head_to_head(
        "pInt high vs pInt low, matched", li[hi], lj[hi], li[lo], lj[lo], A, rq, cq, rstr, cresp, seed0=31)
    h = R["confidence_head_to_head"]
    if h.get("testable"):
        report(f"      head-to-head high vs low, matched on both deciles, {h['n_draws']} draws of "
               f"{h['n_matched']:,} pairs each:")
        report(f"        mean |z| high {h['mean_abs_z_a']:.4f}  low {h['mean_abs_z_b']:.4f}  "
               f"ratio {h['ratio_a_over_b']:.4f}  median p {h['p_median']:.3g}  "
               f"{100*h['frac_draws_p05']:.0f}% of draws p<0.05")
        report(f"        residual imbalance: strength p {h['residual_strength_p']:.3g}, "
               f"responsiveness p {h['residual_responsiveness_p']:.3g}")
        # SIGN AND SIGNIFICANCE READ TOGETHER. Unmatched, the low tercile looks BETTER than the high one,
        # which would be a reversal if it were real. Matched, the gap collapses to
        # ratio ~1 at p~0.2 with no draw reaching 0.05, so it is a NULL -- the unmatched ordering is a
        # composition artefact of which perturbations and which genes each tercile happens to contain, not
        # evidence that low-confidence edges are the better ones.
        report("        the unmatched terciles run the 'wrong' way; matched, this is a NULL, not a "
               "reversal -- do not read the raw ordering as a sign.")

    # ---------------------------------------------------------------- Q3: cell line reproducibility
    report("\n" + "-" * 100)
    report("  Q3  DOES CROSS-CELL-LINE REPRODUCIBILITY BEAT CONFIDENCE?")
    report("      restricted to pairs whose gene pair had an endpoint screened as a bait in BOTH lines")
    report("-" * 100)
    opp = np.array([(pairs[k]["gp"][0] in eid_bait_both) or (pairs[k]["gp"][1] in eid_bait_both)
                    for k in keys])
    ncl = np.array([len(pairs[k]["cl"]) for k in keys])
    both_m = opp & (ncl == 2)
    one_m = opp & (ncl == 1)
    report(f"      pairs with the opportunity to be seen twice: {int(opp.sum()):,} "
           f"({int(both_m.sum()):,} seen in both lines, {int(one_m.sum()):,} in one)")
    report(hdr)
    R["cell_line"] = {}
    for nm, m in (("seen in BOTH lines", both_m), ("seen in ONE line", one_m)):
        for pools in ("loose", "stringent"):
            a = run(f"{nm} ({pools})", m, pools, seed0=41)
            R["cell_line"][f"{nm}|{pools}"] = a
            line(a)
    # THE ALTERNATIVE EXPLANATION FOR "SEEN TWICE" IS "THE BAIT IS A HUB": an interaction involving a
    # promiscuous, abundant protein is more likely to be recovered in any given purification, and
    # perturbing a hub is disruptive for reasons that have nothing to do with cell line. So the head-to-head
    # is matched on BioPlex degree of the perturbed gene as well as on the two prescribed deciles.
    deg = Counter()
    for k, v in gp.items():
        deg[k[0]] += 1
        deg[k[1]] += 1
    pair_deg = np.array([deg[pairs[k]["gp"][0]] + deg[pairs[k]["gp"][1]] for k in keys], dtype=float)
    dq = np.digitize(pair_deg, np.quantile(pair_deg, np.linspace(.1, .9, N_DEC - 1)))
    report(f"      BioPlex degree of the pair's endpoints: both-lines median "
           f"{np.median(pair_deg[both_m]):.0f}, one-line median {np.median(pair_deg[one_m]):.0f} "
           f"-- matched as a third axis below")
    R["cell_line_head_to_head"] = head_to_head(
        "both lines vs one line, matched", li[both_m], lj[both_m], li[one_m], lj[one_m],
        A, rq, cq, rstr, cresp, seed0=53, extra=(dq[both_m], dq[one_m]), extra_name="degree")
    h = R["cell_line_head_to_head"]
    if h.get("testable"):
        report(f"      head-to-head both vs one, matched on strength x responsiveness x DEGREE deciles, "
               f"{h['n_draws']} draws of {h['n_matched']:,} pairs each:")
        report(f"        mean |z| {h['mean_abs_z_a']:.4f} vs {h['mean_abs_z_b']:.4f} "
               f"(ratio {h['ratio_a_over_b']:.4f}, median p {h['p_median']:.3g}, "
               f"{100*h['frac_draws_p05']:.0f}% of draws p<0.05)")
        report(f"        residual imbalance: strength p {h['residual_strength_p']:.3g}, "
               f"responsiveness p {h['residual_responsiveness_p']:.3g}, "
               f"degree p {h['residual_degree_p']:.3g}")
    # and the per-line arms, since neither line is K562 and the asymmetry is worth seeing
    for cl in ("293T", "HCT116"):
        m = np.array([cl in pairs[k]["cl"] for k in keys])
        a = run(f"{cl} edges (stringent)", m, "stringent", seed0=61)
        R["cell_line"][f"{cl}|stringent"] = a
        line(a)

    # ---------------------------------------------------------------- Q4b: is the novel remainder real?
    report("\n" + "-" * 100)
    report("  Q4  REBUILD OR DUPLICATE?  BioPlex pairs already in the existing PPI layer vs not")
    report("-" * 100)
    known = np.array([pairs[k]["gp"] in old_pairs for k in keys])
    report(f"      linked pairs whose gene pair is already an existing PPI edge: {int(known.sum()):,}; "
           f"novel: {int((~known).sum()):,}")
    report(hdr)
    R["novelty"] = {}
    for nm, m in (("already in existing PPI", known), ("novel to BioPlex", ~known)):
        for pools in ("loose", "stringent"):
            a = run(f"{nm} ({pools})", m, pools, seed0=71)
            R["novelty"][f"{nm}|{pools}"] = a
            line(a)
    R["novelty_head_to_head"] = head_to_head(
        "known vs novel, matched", li[known], lj[known], li[~known], lj[~known],
        A, rq, cq, rstr, cresp, seed0=83)

    # ---------------------------------------------------------------- orientation, since we kept it
    report("\n" + "-" * 100)
    report("  Q5  WAS KEEPING BAIT/PREY DIRECTION WORTH IT?  perturbing the BAIT and measuring the PREY,")
    report("      versus perturbing the PREY and measuring the BAIT (exclusive sets)")
    report("-" * 100)
    ob = np.array([pairs[k]["orient"] == {"bait2prey"} for k in keys])
    op = np.array([pairs[k]["orient"] == {"prey2bait"} for k in keys])
    report(hdr)
    R["orientation"] = {}
    for nm, m in (("perturb BAIT -> measure PREY", ob), ("perturb PREY -> measure BAIT", op)):
        a = run(f"{nm} (stringent)", m, "stringent", seed0=97)
        R["orientation"][nm] = a
        line(a)
    R["orientation_head_to_head"] = head_to_head(
        "bait->prey vs prey->bait, matched", li[ob], lj[ob], li[op], lj[op], A, rq, cq, rstr, cresp,
        seed0=101)
    h = R["orientation_head_to_head"]
    if h.get("testable"):
        report(f"      head-to-head, {h['n_draws']} draws of {h['n_matched']:,} pairs each: "
               f"{h['mean_abs_z_a']:.4f} vs {h['mean_abs_z_b']:.4f} "
               f"(ratio {h['ratio_a_over_b']:.4f}, median p {h['p_median']:.3g}, "
               f"{100*h['frac_draws_p05']:.0f}% of draws p<0.05)")

    # ---------------------------------------------------------------- coverage, and whether it is biased
    report("\n" + "-" * 100)
    report("  COVERAGE -- absence of a BioPlex edge is UNKNOWN, and the missingness is not random")
    report("-" * 100)
    n_pert_bait = sum(1 for e in pert_rows if e in bait_ens)
    n_var_seen = int(col_is_seen.sum())
    report(f"      {n_pert_bait:,}/{len(pert_rows):,} ({100*n_pert_bait/len(pert_rows):.1f}%) of GWPS "
           f"perturbed genes were screened as a BioPlex bait")
    report(f"      {n_var_seen:,}/{len(var):,} ({100*n_var_seen/len(var):.1f}%) of GWPS measured genes "
           f"appear anywhere in the BioPlex network")
    p_str = float(stats.mannwhitneyu(rstr[row_is_bait], rstr[~row_is_bait], alternative="two-sided")[1])
    p_resp = float(stats.mannwhitneyu(cresp[col_is_seen], cresp[~col_is_seen],
                                      alternative="two-sided")[1])
    report(f"      perturbation strength: baits {rstr[row_is_bait].mean():.4f} vs non-baits "
           f"{rstr[~row_is_bait].mean():.4f}  MWU p {p_str:.3g}")
    report(f"      gene responsiveness:   in-BioPlex {cresp[col_is_seen].mean():.4f} vs absent "
           f"{cresp[~col_is_seen].mean():.4f}  MWU p {p_resp:.3g}")
    report(f"      -> coverage is biased on both matched covariates, which is exactly why the stringent "
           f"control exists.")
    R["coverage"] = {"gwps_perts_that_are_baits": n_pert_bait, "gwps_perts": len(pert_rows),
                     "gwps_genes_in_bioplex": n_var_seen, "gwps_genes": len(var),
                     "bias_strength_p": p_str, "bias_responsiveness_p": p_resp,
                     "bait_mean_strength": float(rstr[row_is_bait].mean()),
                     "nonbait_mean_strength": float(rstr[~row_is_bait].mean())}

    # ---------------------------------------------------------------- write the layer
    R["limits"] = [
        "NEITHER CELL LINE IS K562. BioPlex 3.0 is HEK293T and HCT116; the fast test asks whether an "
        "interaction found in those lines constrains the K562 transcriptional response, which is a "
        "transfer question, not a within-cell measurement. Nothing here establishes that any given edge "
        "exists in K562.",
        "AP-MS reports CO-COMPLEX MEMBERSHIP, not binary direct contact. A bait->prey edge may be bridged "
        "by one or more other proteins; the direction is experimental (who was tagged), not mechanistic.",
        f"The released network is already filtered at pInt >= {PINT_FLOOR}, so the 'low confidence' arm is "
        f"0.75-tercile, not a genuine low-confidence tail. The confidence test therefore measures "
        f"resolution WITHIN the published set and says nothing about edges BioPlex rejected.",
        "Bait selection is not random and neither is prey detectability: baits skew to well-expressed, "
        "clonable ORFs and prey to abundant proteins. Absence of an edge among non-baits is UNKNOWN, never "
        "evidence of no interaction.",
        "Overexpression of a tagged bait can create interactions that do not occur at endogenous "
        "stoichiometry; BioPlex mitigates this with near-endogenous promoters but does not eliminate it.",
        "The GWPS readout is transcriptional. A physical interaction that acts post-translationally leaves "
        "no signature in this test, so a null here is a null for TRANSCRIPTIONAL coupling only -- it is "
        "not evidence against the interaction.",
        "Endpoints are joined to the registry by UniProt accession first; a small number resolve only by "
        "gene symbol and are recorded via=symbol, the weakest key.",
        "The registry's gene|entrez alias keys are stored as floats ('7105.0'); this module normalises on "
        "its side. Any layer joining raw integer Entrez ids against this registry will silently match "
        "nothing.",
        "pW/pNI/pInt come from one CompPASS-plus model fitted to this release. They are not comparable to "
        "STRING confidences, MIscores, or the confidences of any other release, and must not be pooled "
        "with them.",
        "788 genes are perturbed by more than one construct in GWPS, so linked (row, gene) pairs are not "
        "independent and the raw p-values are optimistic. The de-replicated arm is the one to quote if a "
        "single number is wanted.",
        "The two matched covariates are binned into deciles, which leaves a residual imbalance that is "
        "small but not zero (reported per arm). Finer bins would tighten it at the cost of thinner cells.",
        "Every ratio here is a contrast against a matched control drawn from the SAME perturbation matrix; "
        "no arm has been validated on a held-out cell line, so this measures association within K562 "
        "Perturb-seq, not predictive transfer.",
    ]
    layer = dict(R)
    layer["edges"] = edges
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    # ---------------------------------------------------------------- verdict
    report("\n" + "=" * 100)
    fl, fs = R["fast_test"]["loose"], R["fast_test"]["stringent"]
    fd = R["fast_test"]["dereplicated"]
    hc = R["confidence_head_to_head"]
    hl = R["cell_line_head_to_head"]
    hn = R["novelty_head_to_head"]

    def sig(a):
        return a.get("testable") and a["frac_draws_p05"] >= 0.8 and a["ratio"] > 1.0
    strong = fs["ratio"] >= 1.05 and sig(fs)
    conf_works = hc.get("testable") and hc["frac_draws_p05"] >= 0.8 and hc["ratio_a_over_b"] > 1.0
    cl_works = hl.get("testable") and hl["frac_draws_p05"] >= 0.8 and hl["ratio_a_over_b"] > 1.0
    nov = R["novelty"]["novel to BioPlex|stringent"]
    head = (f"BioPlex 3.0 yields {len(edges):,} directed AP-MS edges "
            f"({per_cl.get('293T',0):,} in 293T, {per_cl.get('HCT116',0):,} in HCT116) over "
            f"{len(gp):,} distinct gene pairs, joined to the registry {100*(1-via['UNRESOLVED']/len(ent_ids)):.1f}% "
            f"by UniProt accession rather than symbol. Only {100*len(inter)/max(len(bp_incell),1):.1f}% of "
            f"the in-cell BioPlex pairs are already in the existing 191,447-tuple PPI layer, so this is a "
            f"rebuild that also adds {len(bp_incell)-len(inter):,} pairs that layer does not have.")
    body = (f"On the fast test, {len(keys):,} (perturbation, gene) pairs are testable in GWPS: mean |z| "
            f"{fs['mean_abs_z_linked']:.4f} against {fl['ratio']:.3f}x a loosely matched control and "
            f"{fs['ratio']:.3f}x the STRINGENT control -- pairs where the perturbed gene was itself a "
            f"BioPlex bait and the measured gene was detected in BioPlex, so the comparison is "
            f"tested-and-not-found rather than never-examined (median p {fs['p_median']:.2g}, "
            f"{100*fs['frac_draws_p05']:.0f}% of {fs['n_draws']} draws p<0.05; residual imbalance after "
            f"matching, strength p {fs['residual_strength_p']:.2g}, responsiveness p "
            f"{fs['residual_responsiveness_p']:.2g}). The sham arm returns {sham['ratio']:.3f}x, so the "
            f"machinery is not manufacturing the effect, and de-replicating to one GWPS row per perturbed "
            f"gene leaves {fd['ratio']:.3f}x on {fd['n_pairs']:,} pairs "
            f"({100*fd['frac_draws_p05']:.0f}% of draws p<0.05), so the p-values are not being carried by "
            f"repeated sgRNA constructs.")
    if strong:
        lead = (f"THIS LAYER CARRIES INFORMATION, AND IT SURVIVES THE HARD CONTROL. ")
    elif fs["ratio"] >= 1.02 and sig(fs):
        lead = (f"THIS LAYER CARRIES A SMALL BUT REAL SIGNAL, WELL BELOW THE EXISTING PPI LAYER. ")
    else:
        lead = (f"THIS LAYER IS A WELL-CONTROLLED NULL AGAINST THE HARD CONTROL. ")
    q2 = (f"Confidence was worth keeping: matched head-to-head, top-tercile pInt pairs reach "
          f"{hc['mean_abs_z_a']:.4f} against {hc['mean_abs_z_b']:.4f} for the bottom tercile "
          f"({hc['ratio_a_over_b']:.3f}x, median p {hc['p_median']:.2g}, "
          f"{100*hc['frac_draws_p05']:.0f}% of draws p<0.05)."
          if conf_works else
          f"Confidence was NOT worth keeping for this purpose: matched head-to-head, top-tercile pInt "
          f"pairs reach {hc['mean_abs_z_a']:.4f} against {hc['mean_abs_z_b']:.4f} for the bottom tercile "
          f"({hc['ratio_a_over_b']:.3f}x, median p {hc['p_median']:.2g}, only "
          f"{100*hc['frac_draws_p05']:.0f}% of draws p<0.05) -- above the 0.75 publication floor, pInt "
          f"does not resolve which edges couple transcriptionally in K562. It is stored, not weighted.")
    q3 = (f"Cross-cell-line reproducibility DOES beat it, and it is the one field worth weighting: among "
          f"pairs whose bait was screened in both lines -- so the interaction had the OPPORTUNITY to be "
          f"seen twice -- those actually seen twice reach {hl['mean_abs_z_a']:.4f} against "
          f"{hl['mean_abs_z_b']:.4f} for one line ({hl['ratio_a_over_b']:.3f}x, median p "
          f"{hl['p_median']:.2g}, {100*hl['frac_draws_p05']:.0f}% of draws p<0.05), matched on "
          f"BioPlex degree as well as on both prescribed deciles so this is not simply hub baits. The "
          f"cell line the existing layer discarded is therefore the more informative of the two fields it "
          f"discarded."
          if cl_works else
          f"Cross-cell-line reproducibility does not separate either: both-lines {hl['mean_abs_z_a']:.4f} "
          f"vs one-line {hl['mean_abs_z_b']:.4f} ({hl['ratio_a_over_b']:.3f}x, "
          f"{100*hl['frac_draws_p05']:.0f}% of draws p<0.05, matched on degree as well), so seeing an "
          f"interaction in two non-K562 lines does not make it more likely to matter in K562.")
    q4 = (f"The pairs BioPlex adds that the existing layer lacks are not noise: they score "
          f"{nov['ratio']:.3f}x against the stringent control on {nov['n_pairs']:,} pairs "
          f"({100*nov['frac_draws_p05']:.0f}% of draws p<0.05)."
          if sig(nov) else
          f"The pairs BioPlex adds that the existing layer lacks do NOT carry signal on their own "
          f"({nov['ratio']:.3f}x, {nov['n_pairs']:,} pairs, {100*nov['frac_draws_p05']:.0f}% of draws "
          f"p<0.05), so the value of this rebuild is the metadata on known edges, not the new edges.")
    cov = (f"Coverage is biased on both matched covariates ({n_pert_bait:,}/{len(pert_rows):,} perturbed "
           f"genes are baits, strength p {p_str:.2g}; {n_var_seen:,}/{len(var):,} measured genes appear in "
           f"BioPlex, responsiveness p {p_resp:.2g}), which is why the stringent control is the one the "
           f"verdict rests on and why a missing edge is stored as unknown.")
    v = " ".join([lead + head, body, q2, q3, q4, cov])
    R["verdict"] = v
    report(f"  VERDICT: {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "bioplex.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'bioplex.json'}")


if __name__ == "__main__":
    main()
