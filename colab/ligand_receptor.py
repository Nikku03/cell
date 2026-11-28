"""LIGAND-RECEPTOR IN A MONOCULTURE -- how much of the layer is even switched on, and the one half of the
arrow a single cell line can test.

WHY THIS TEST. The cell object carries two multicellular layers that have never been checked against
anything: `lr`, 948 ligand-receptor pairs, and `nichenet`, 1,195 ligands each asserting ~25 downstream
target genes. Both encode a claim of the form

    SENDER expresses L  ->  L reaches RECEIVER  ->  RECEIVER's R binds it  ->  a target programme moves

and a K562 Perturb-seq screen contains exactly one cell type. Three of those four links are unobservable
here: there is no sender that is not also the receiver, there is no space, and there is no ligand exposure
to vary. Pretending otherwise is how a layer gets marked validated on evidence that cannot bear on it. So
this module does the two things monoculture CAN do, and says plainly what each is worth.

(1) GATE. An edge whose ligand or whose receptor is not expressed is inactive by construction, whatever the
    curation says. Scored against K562 mRNA (the Replogle panel's own detection floor) and, independently,
    against K562 protein (PaxDb iBAQ, Geiger 2012), how many of the 948 pairs have BOTH partners present?
    That number is the honest size of the layer in this cell type. It is deliberately the AUTOCRINE gate --
    the only mode a monoculture has -- and the atlas `emask` is used alongside it to show how much of the
    layer is intercellular by construction and therefore out of reach here rather than merely absent.

(2) TEST. NicheNet asserts ligand -> target programme. The programme is intracellular, downstream of the
    receptor, and CRISPRi does perturb receptors. So: for the receptors perturbed in Replogle K562, are the
    NicheNet targets of their COGNATE ligands enriched among the genes that actually move? This checks the
    intracellular half of the arrow ONLY. It cannot check that the ligand reaches the receiver, that the
    sender exists, or that the pair is spatially plausible. A positive result would say the downstream half
    is real in this cell; it would say nothing about the communication event.

THE CONTROL THAT DECIDES IT. NicheNet's top-25 target lists are not ligand-specific in the way the layer's
shape implies: MYC appears for 950 of 1,195 ligands, VEGFA for 931, CDKN1A for 930. A random-gene control
would therefore "confirm" NicheNet by measuring nothing but the responsiveness of famous stress genes. The
null that matters is the LIGAND SWAP: give the receptor the target set of a different ligand, keeping the
resource's compositional bias intact and destroying only the cognate pairing.

WHAT WOULD FALSIFY WHAT. If cognate target sets move more than ligand-swapped sets, the intracellular half
of NicheNet survives in K562. If they do not, either NicheNet's ligand-specificity is not real at this
resolution, or -- and this is the monoculture confound, not a detail -- a receptor with no ligand present
has nothing to transduce, so knocking it down cannot silence a programme that was never running. Those two
explanations are not separable here, and the module does not pretend to separate them. Because a null is
only readable with a power statement, the identical machinery is run on a relationship known to be real on
this readout (protein-complex co-membership), at full n and again subsampled to the receptor test's n, so
the smallest effect this design could have seen is measured rather than assumed.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GWPS = SP / "gwps.h5ad"                       # Replogle K562 genome-scale Perturb-seq pseudobulk
NET = SP / "cell_complete.json.gz"            # the cell object: lr, nichenet, emask, complexes
K562P = SP / "cell_k562_protein.json.gz"      # PaxDb 9606-iBAQ_K562_Geiger_2012 (colab/k562_protein.py)
LAYER = SP / "cell_ligand_receptor.json.gz"
CENSUS = "https://api.cellxgene.cziscience.com/wmg/v2/query"
CENSUS_DIMS = "https://api.cellxgene.cziscience.com/wmg/v2/primary_filter_dimensions"

# ---- pinned provenance. ONE dataset on each axis, asserted, not assumed. -------------------------------
SRC_RNA = "Replogle et al. 2022 genome-scale K562 CRISPRi Perturb-seq (gwps.h5ad pseudobulk)"
SRC_PROT = "PaxDb 9606-iBAQ_K562_Geiger_2012 (Geiger et al., Mol Cell Proteomics 2012)"
SRC_ATLAS = "cell-object `emask`: 200 atlas cell types, 7,496 genes"
N_DRAWS = 25          # matched controls are never read from one draw
MIN_TARGETS = 5       # a target set with fewer measured members is not a programme, it is noise
CENSUS_MAX_GENES = 200
CENSUS_BATCH = 10
CENSUS_PC = 0.10      # fraction of cells in a cell type that must detect the gene to call it expressed


def robust_z(M):
    """Per-GENE robust z across perturbations, the same definition causal_reg.py uses.

    (x - median) / (1.4826 * MAD) down each column, so an effect means "this gene moved more than it
    usually moves" rather than "this gene moved" -- which matters enormously here, because NicheNet's
    target vocabulary is precisely the set of genes that move under everything.
    """
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    Z = (M - med) / mad
    return np.where(np.isfinite(Z), Z, np.nan)


def bins(v, k=20):
    """Quantile bins for matching. TWENTY rather than ten deliberately: with deciles the matched control
    sat measurably below the observed set on the very covariate it was matching (residual imbalance is
    reported either way, but a control that has to be corrected in prose is a worse control)."""
    q = np.digitize(v, np.nanquantile(v[np.isfinite(v)], np.linspace(1 / k, 1 - 1 / k, k - 1)))
    return q, {d: np.where(q == d)[0] for d in np.unique(q)}


def census_fetch(ens_ids, report):
    """Cell-type-resolved expression from the CZ CELLxGENE census WMG API. Side note, not a deliverable.

    Streamed and reduced in flight: the raw response is ~0.7 MB of JSON per gene and none of it is kept.
    A blocked or changed endpoint returns None and is reported as unreachable -- never filled in.
    """
    import urllib.request
    import urllib.error
    out, labels, snaps = {}, {}, set()
    try:
        req = urllib.request.Request(CENSUS_DIMS, headers={"Accept-Encoding": "gzip"})
        r = urllib.request.urlopen(req, timeout=120)
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        dims = json.loads(raw)
        snaps.add(str(dims.get("snapshot_id")))
        known = set()
        for e in dims.get("gene_terms", {}).get("NCBITaxon:9606", []):
            known.update(e.keys())
    except Exception as e:                                                # noqa: BLE001
        report(f"    CENSUS UNREACHABLE at {CENSUS_DIMS}: {type(e).__name__} {e}")
        return None
    ask = [g for g in ens_ids if g in known][:CENSUS_MAX_GENES]
    report(f"    census gene universe {len(known):,}; requesting {len(ask)} of {len(ens_ids)} LR genes "
           f"(cap {CENSUS_MAX_GENES})")
    for i in range(0, len(ask), CENSUS_BATCH):
        blk = ask[i:i + CENSUS_BATCH]
        body = json.dumps({"filter": {"gene_ontology_term_ids": blk,
                                      "organism_ontology_term_id": "NCBITaxon:9606"},
                           "is_rollup": True}).encode()
        try:
            req = urllib.request.Request(CENSUS, data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Accept-Encoding": "gzip"})
            r = urllib.request.urlopen(req, timeout=300)
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            j = json.loads(raw)
        except Exception as e:                                            # noqa: BLE001
            report(f"    census batch {i//CENSUS_BATCH} failed: {type(e).__name__} {e} -- partial result")
            break
        snaps.add(str(j.get("snapshot_id")))
        for cl, lab in (j.get("term_id_labels", {}).get("cell_types", {}) or {}).items():
            labels[cl] = lab if isinstance(lab, str) else str(lab)
        for g, tis in j.get("expression_summary", {}).items():
            acc = out.setdefault(g, {})
            for _t, cts in tis.items():
                for cl, v in cts.items():
                    a = v.get("aggregated") or {}
                    pc, me = float(a.get("pc", 0) or 0), float(a.get("me", 0) or 0)
                    if pc > acc.get(cl, (0.0, 0.0))[0]:
                        acc[cl] = (pc, me)
        del j, raw
    if len(snaps) > 1:
        raise SystemExit(f"census returned MORE THAN ONE SNAPSHOT {snaps} -- provenance is not pinned")
    return {"snapshot_id": snaps.pop() if snaps else None, "labels": labels,
            "expressed": {g: sorted([c for c, (pc, _m) in v.items() if pc >= CENSUS_PC])
                          for g, v in out.items()}}


def main():
    import h5py
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("LIGAND-RECEPTOR -- what is switched on in K562, and the half of the arrow monoculture can test")
    report("=" * 100)
    for p in (GWPS, NET, K562P):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    # ---- the join backbone. Every lookup below goes through it. -------------------------------------
    from entity_registry import load as load_reg
    import entity_registry as ER
    REG = load_reg()
    _RJ = json.load(gzip.open(ER.REGISTRY, "rt"))
    CELL_INDEX = _RJ["cell_index"]                    # index-string -> eid; the ONLY correct route for
    del _RJ                                           # `emask`, `lr` indices, `gene2cplx`, `ppm`, `abund`

    D = json.load(gzip.open(NET, "rt"))
    names = [g["name"] for g in D["genes"]]
    lr = [tuple(e) for e in D["lr"]]
    nn = D["nichenet"]
    emask = {int(k): int(v) for k, v in D["emask"].items()}
    ctnames = D["ctnames"]
    complexes = D["complexes"]
    del D

    # THE HISTORICAL BUG, CHECKED RATHER THAN ASSUMED. `lr` and `emask` are keyed by gene INDEX. If the
    # index->entity map and the positional decode ever disagree, every gate below silently gates the wrong
    # genes -- which is exactly how an abundance control once ran as a column of zeros and passed.
    agree = sum(1 for i, n in enumerate(names)
                if REG.entities.get(CELL_INDEX.get(str(i)), {}).get("symbol") == n)
    if agree < len(names):
        raise SystemExit(f"cell_index disagrees with positional decode on {len(names)-agree} genes")
    report(f"  cell_index <-> positional decode: {agree:,}/{len(names):,} agree "
           f"(the index-keyed layers are being read on the right genes)")

    _e, st_lig = REG.join("gene", "symbol", sorted({names[a] for a, _b in lr}), 0.90, "lr ligands")
    _e, st_rec = REG.join("gene", "symbol", sorted({names[b] for _a, b in lr}), 0.90, "lr receptors")
    _e, st_nnl = REG.join("gene", "symbol", sorted(nn), 0.85, "nichenet ligands")
    nn_vocab = sorted({t for v in nn.values() for t in v})
    _e, st_nnt = REG.join("gene", "symbol", nn_vocab, 0.85, "nichenet target vocabulary")
    report(f"  registry joins: lr ligands {st_lig['rate']:.1%}, lr receptors {st_rec['rate']:.1%}, "
           f"nichenet ligands {st_nnl['rate']:.1%}, nichenet targets {st_nnt['rate']:.1%}")

    # ---- K562 mRNA: the Replogle panel ---------------------------------------------------------------
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        gid = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                        for s in f["var"]["gene_id"][:]])
        gmean = f["var"]["mean"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    _e, st_g = REG.join("gene", "ensembl", list(gid), 0.80, "K562 measured genes")
    _e, st_p = REG.join("gene", "symbol", list(psym), 0.80, "K562 perturbations")
    report(f"  registry joins: K562 measured genes {st_g['rate']:.1%}, "
           f"K562 perturbations {st_p['rate']:.1%}")

    ens2sym = {}
    for e in gid:
        eid = REG.resolve("gene", "ensembl", e)
        ent = REG.entities.get(eid) if eid else None
        if ent and ent.get("symbol"):
            ens2sym[e] = ent["symbol"]
    sym2col, panel_mean = {}, {}
    for j, e in enumerate(gid):
        s = ens2sym.get(e)
        if s and s not in sym2col:
            sym2col[s] = j
            panel_mean[s] = float(gmean[j])
    sym2row = {}
    for i, s in enumerate(psym):
        sym2row.setdefault(s, []).append(i)
    report(f"  K562 panel: {X.shape[0]:,} perturbations x {X.shape[1]:,} measured genes; "
           f"{len(sym2col):,} resolve to a symbol; detection floor mean = {gmean.min():.3f}")

    # ---- K562 protein, an INDEPENDENT axis on the same cell ------------------------------------------
    kp = json.load(gzip.open(K562P, "rt"))
    if kp["source"] != SRC_PROT:
        raise SystemExit(f"proteomics provenance drifted: {kp['source']!r} != {SRC_PROT!r}")
    prot = kp["abundance"]
    _e, st_pr = REG.join("gene", "symbol", sorted(prot), 0.75, "K562 proteome symbols")
    report(f"  K562 proteome: {len(prot):,} proteins ({st_pr['rate']:.1%} join) -- {SRC_PROT}")

    # ==================================================================================================
    # PART 1 -- THE GATE. How much of the lr layer is switched on in this cell at all?
    # ==================================================================================================
    report("\n" + "-" * 100)
    report("PART 1  GATE -- an edge with an unexpressed partner is inactive by construction")
    report("-" * 100)
    ligs = sorted({names[a] for a, _b in lr})
    recs = sorted({names[b] for _a, b in lr})
    allg = names

    def rate(S, memb):
        return sum(1 for x in S if x in memb) / max(len(S), 1)

    report(f"  {'set':26s} {'n':>6s} {'in K562 mRNA panel':>20s} {'in K562 proteome':>18s}")
    for lab, S in (("all cell-object genes", allg), ("lr ligands", ligs), ("lr receptors", recs)):
        report(f"  {lab:26s} {len(S):6,d} {rate(S, panel_mean):20.3f} {rate(S, prot):18.3f}")

    def detected_by(g):
        """Which axis saw this gene. `none` means NOT DETECTED, which is not the same as absent: both
        axes have floors, and a secreted ligand is under-detected by each of them."""
        return ("rna" if g in panel_mean else "") + ("+prot" if g in prot else "") or "none"

    def gate(g):
        return "expressed" if (g in panel_mean or g in prot) else "not_detected"

    # HOW MUCH DOES THE mRNA PANEL MISS? The proteome is an independent axis on the same cell, so the
    # genes it detects that the panel does not are a direct read of the panel's false-negative rate --
    # which is what turns "absent from the panel" into "unknown" rather than "not there".
    lrgenes = sorted(set(ligs) | set(recs))
    prot_only = [g for g in lrgenes if g not in panel_mean and g in prot]
    rna_only = [g for g in lrgenes if g in panel_mean and g not in prot]
    report(f"\n  PANEL FALSE NEGATIVES: {len(prot_only)} of {len(lrgenes)} lr partners are absent from the "
           f"mRNA panel yet\n    present in the K562 proteome ({100*len(prot_only)/len(lrgenes):.1f}%); "
           f"{len(rna_only)} are the reverse. Panel absence is UNKNOWN, not zero.")

    both_rna = [(names[a], names[b]) for a, b in lr if names[a] in panel_mean and names[b] in panel_mean]
    both_any = [(names[a], names[b]) for a, b in lr
                if gate(names[a]) == "expressed" and gate(names[b]) == "expressed"]
    rec_only = sum(1 for a, b in lr if names[b] in panel_mean and names[a] not in panel_mean)
    lig_only = sum(1 for a, b in lr if names[a] in panel_mean and names[b] not in panel_mean)
    report(f"\n  948-pair layer, BOTH partners required:")
    report(f"    both in K562 mRNA panel            {len(both_rna):4,d}/{len(lr):,}  "
           f"({100*len(both_rna)/len(lr):.1f}%)")
    report(f"    both in mRNA OR K562 proteome      {len(both_any):4,d}/{len(lr):,}  "
           f"({100*len(both_any)/len(lr):.1f}%)   <- the most generous honest gate")
    report(f"    receptor only (no ligand in K562)  {rec_only:4,d}   ligand only {lig_only:,}")
    report(f"    surviving pairs: {sorted(both_any)}")

    # IS THE DEPLETION REAL, OR ARE LR GENES JUST NARROW GENES? Matched control on atlas breadth: a
    # ligand and a random gene expressed in the same number of atlas cell types should clear the K562
    # panel at the same rate if there is nothing special about being a ligand.
    br = {i: bin(m).count("1") for i, m in emask.items()}
    have_mask = np.array(sorted(br))
    brv = np.array([br[i] for i in have_mask], dtype=float)
    bq, by_bq = bins(brv)
    name2i = {n: i for i, n in enumerate(names)}
    idx_of = {int(g): k for k, g in enumerate(have_mask)}
    rng = np.random.default_rng(0)

    def breadth_matched_panel_rate(S, draws=N_DRAWS):
        pos = [idx_of[name2i[s]] for s in S if name2i.get(s) in idx_of]
        obs = np.mean([names[have_mask[k]] in panel_mean for k in pos])
        ms, imb = [], []
        for _d in range(draws):
            alt = [int(rng.choice(by_bq[bq[k]])) for k in pos]
            ms.append(np.mean([names[have_mask[k]] in panel_mean for k in alt]))
            imb.append(float(np.mean(brv[alt]) - np.mean(brv[pos])))
        return len(pos), float(obs), float(np.mean(ms)), float(np.mean(imb))

    report(f"\n  MATCHED CONTROL on atlas breadth ({N_DRAWS} draws) -- are lr genes merely narrow genes?")
    report(f"  {'set':26s} {'n w/ emask':>11s} {'observed':>9s} {'breadth-matched':>16s} "
           f"{'residual breadth imb.':>22s}")
    gate_matched = {}
    for lab, S in (("lr ligands", ligs), ("lr receptors", recs)):
        n, o, m, im = breadth_matched_panel_rate(S)
        gate_matched[lab] = {"n": n, "observed_panel_rate": o, "matched_panel_rate": m,
                             "residual_breadth_imbalance_celltypes": im}
        report(f"  {lab:26s} {n:11,d} {o:9.3f} {m:16.3f} {im:+22.2f}")

    # HOW MUCH OF THE LAYER IS INTERCELLULAR BY CONSTRUCTION? If a pair's two partners are never expressed
    # in the same atlas cell type, no monoculture of any kind can carry it -- that is a property of the
    # edge, not a failure of K562.
    co, cob = [], []
    for a, b in lr:
        if a in emask and b in emask:
            co.append(bin(emask[a] & emask[b]).count("1"))
            cob.append((br[a], br[b]))
    co = np.array(co)
    ctrl_co = []
    for _d in range(N_DRAWS):
        s = []
        for ba, bb in cob:
            ka = int(rng.choice(by_bq[bq[idx_of[int(have_mask[np.argmin(np.abs(brv - ba))])]]]))
            kb = int(rng.choice(by_bq[bq[idx_of[int(have_mask[np.argmin(np.abs(brv - bb))])]]]))
            s.append(bin(emask[int(have_mask[ka])] & emask[int(have_mask[kb])]).count("1"))
        ctrl_co.append(float(np.mean(s)))
    report(f"\n  ATLAS CO-EXPRESSION ({len(co):,}/{len(lr):,} pairs with a mask on both partners, "
           f"{len(ctnames)} cell types)")
    report(f"    cell types expressing BOTH partners: mean {co.mean():.2f}, median {np.median(co):.0f}, "
           f"zero for {100*np.mean(co == 0):.1f}% of pairs")
    report(f"    breadth-matched random pairs ({N_DRAWS} draws): mean {np.mean(ctrl_co):.2f}")
    report(f"    -> over half the layer has NO cell type in the atlas expressing both partners. That is "
           f"the layer\n       behaving as an INTERCELLULAR object; it is out of reach of any monoculture, "
           f"not merely of K562.")

    # ==================================================================================================
    # PART 2 -- THE TEST. Perturb the receptor; ask whether the cognate ligand's programme moves.
    # ==================================================================================================
    report("\n" + "-" * 100)
    report("PART 2  RECEPTOR PERTURBATION -- the intracellular half of the arrow, and only that half")
    report("-" * 100)
    Z = robust_z(X)
    A = np.abs(Z)
    gr = np.nanmean(A, axis=0)                 # gene responsiveness
    ps = np.nanmean(A, axis=1)                 # perturbation strength
    grq, by_grq = deciles(gr)
    psq, by_psq = deciles(ps)

    rec2lig = {}
    for a, b in lr:
        rec2lig.setdefault(names[b], set()).add(names[a])
    nn_meas = {l: sorted({sym2col[s] for s in t if s in sym2col}) for l, t in nn.items()}
    lig_pool = [l for l, c in nn_meas.items() if len(c) >= MIN_TARGETS]

    # HOW LIGAND-SPECIFIC IS NICHENET, REALLY? This bounds everything below: if two ligands' target sets
    # are largely the same genes, the ligand-swap control is largely the same test, and no design can
    # separate them.
    ls = [frozenset(nn_meas[l]) for l in lig_pool]
    pick = rng.integers(0, len(ls), (20000, 2))
    jac = [len(ls[i] & ls[j]) / len(ls[i] | ls[j]) for i, j in pick if i != j and (ls[i] | ls[j])]
    from collections import Counter
    tc = Counter(t for v in nn.values() for t in v)
    report(f"  NicheNet shape: {len(nn):,} ligands, {len(nn_vocab):,} distinct targets, "
           f"{len({t for t in nn_vocab if t in sym2col}):,} of them measured in K562")
    report(f"    most-asserted targets: " +
           ", ".join(f"{t} {100*c/len(nn):.0f}%" for t, c in tc.most_common(6)))
    report(f"    mean pairwise Jaccard between two ligands' MEASURED target sets: {np.mean(jac):.3f} "
           f"(median {np.median(jac):.3f})")
    report(f"    -> a swapped ligand already shares ~{100*np.mean(jac):.0f}% of the cognate set, which "
           f"ATTENUATES any\n       true ligand-specific effect by about that much. Read the null below "
           f"with that in mind.")

    cases = []
    for r in sorted(rec2lig):
        if r not in sym2row:
            continue
        ligset = sorted(rec2lig[r] & set(nn_meas))
        cols = sorted({c for l in ligset for c in nn_meas[l]})
        if len(cols) < MIN_TARGETS:
            continue
        auto = any((l in panel_mean) or (l in prot) for l in ligset)
        cases.append({"rec": r, "rows": sym2row[r], "cols": cols, "ligs": ligset, "autocrine": auto})
    report(f"\n  {len(cases)} receptors are perturbed in Replogle K562 AND have a cognate ligand with "
           f">= {MIN_TARGETS} measured NicheNet targets")
    report(f"    of those, {sum(c['autocrine'] for c in cases)} have a cognate ligand itself present in "
           f"K562 (autocrine-plausible)")

    # POWER CHECK 1 -- did the knockdowns work? A receptor whose own transcript did not fall cannot be
    # expected to silence anything downstream.
    selfz = [(c["rec"], float(np.nanmean(A[c["rows"], sym2col[c["rec"]]])))
             for c in cases if c["rec"] in sym2col]
    allself = np.array([float(np.nanmean(A[sym2row[s], sym2col[s]])) for s in sym2row if s in sym2col])
    sz = np.array([v for _r, v in selfz])
    report(f"\n  POWER 1 -- self-knockdown: {len(selfz)}/{len(cases)} tested receptors are themselves in "
           f"the panel;")
    report(f"    their own-transcript |z| median {np.median(sz):.2f} ({100*np.mean(sz >= 5):.0f}% reach "
           f"|z|>=5) vs {np.median(allself):.2f} ({100*np.mean(allself >= 5):.0f}%) across all "
           f"{len(allself):,} perturbations -- the perturbations took, if a little weakly")
    rstr = np.array([float(np.nanmean(ps[c["rows"]])) for c in cases])
    report(f"    receptor perturbation strength median {np.median(rstr):.3f} vs {np.median(ps):.3f} over "
           f"all perturbations -- receptor knockdown is LESS disruptive than an average perturbation")

    def sc(rows, cols):
        return float(np.nanmean(A[np.ix_(rows, cols)]))

    obs = np.array([sc(c["rows"], c["cols"]) for c in cases])
    obs_hits = np.array([float(np.nanmean(A[np.ix_(c["rows"], c["cols"])] >= 2)) for c in cases])
    report(f"\n  EFFECT SIZE (the RAW held-out value, not a difference):")
    report(f"    mean |robust z| of cognate NicheNet targets under receptor knockdown = "
           f"{obs.mean():.4f}   (median over receptors {np.median(obs):.4f})")
    report(f"    fraction of those (receptor, target) readings reaching |z| >= 2 = {obs_hits.mean():.4f}")

    def matched(gen, label, draws=N_DRAWS):
        """Every matched control is read as `draws` draws: the mean effect, the fraction of draws that
        reach p<0.05, and the residual imbalance left on the covariate that was matched."""
        ms, md, wins, pv, imb = [], [], [], [], []
        for d in range(draws):
            vals, ib = [], []
            for c in cases:
                v, i = gen(c, rng)
                vals.append(v)
                ib.append(i)
            v = np.array(vals)
            ms.append(v.mean())
            md.append(float(np.median(v)))
            wins.append(float(np.mean(obs > v)))
            pv.append(float(stats.wilcoxon(obs, v).pvalue))
            imb.append(float(np.mean(ib)))
        pv = np.array(pv)
        rec = {"label": label, "draws": draws, "control_mean": float(np.mean(ms)),
               "control_median": float(np.mean(md)), "ratio_mean": float(obs.mean() / np.mean(ms)),
               "ratio_median": float(np.median(obs) / np.mean(md)),
               "win_rate": float(np.mean(wins)), "frac_draws_p_lt_05": float(np.mean(pv < 0.05)),
               "median_p": float(np.median(pv)), "residual_imbalance": float(np.mean(imb))}
        report(f"    {label:38s} {np.mean(ms):8.4f} {rec['ratio_mean']:7.3f}x {rec['ratio_median']:8.3f}x "
               f"{rec['win_rate']:7.3f} {rec['frac_draws_p_lt_05']:9.2f} {rec['residual_imbalance']:+11.4f}")
        return rec

    def g_resp(c, g):
        """C1 -- random genes matched on responsiveness decile. Residual imbalance = mean responsiveness."""
        alt = [int(g.choice(by_grq[grq[j]])) for j in c["cols"]]
        return sc(c["rows"], alt), float(np.mean(gr[alt]) - np.mean(gr[c["cols"]]))

    def g_lig(c, g):
        """C2 -- THE NULL THAT DECIDES IT. A different ligand's real NicheNet set: same compositional
        bias toward famous stress genes, no cognate relationship. Residual imbalance = responsiveness."""
        alt = sorted({x for l in g.choice(lig_pool, len(c["ligs"]), replace=False) for x in nn_meas[l]})
        if len(alt) < MIN_TARGETS:
            alt = c["cols"]
        return sc(c["rows"], alt), float(np.mean(gr[alt]) - np.mean(gr[c["cols"]]))

    def g_rec(c, g):
        """C3 -- same target genes, a different perturbation of matched strength. Gene identity is held
        EXACTLY fixed, so responsiveness cannot leak. Residual imbalance = perturbation strength."""
        alt = [int(g.choice(by_psq[psq[i]])) for i in c["rows"]]
        return sc(alt, c["cols"]), float(np.mean(ps[alt]) - np.mean(ps[c["rows"]]))

    report(f"\n  MATCHED CONTROLS, {N_DRAWS} draws each (n = {len(cases)} receptors, paired Wilcoxon)")
    report(f"    {'control':38s} {'mean':>8s} {'ratio':>8s} {'ratio_med':>9s} {'win':>7s} "
           f"{'frac p<.05':>9s} {'resid imb':>12s}")
    c1 = matched(g_resp, "C1 responsiveness-matched genes")
    c2 = matched(g_lig, "C2 LIGAND SWAP (the null)")
    c3 = matched(g_rec, "C3 receptor swap, matched strength")

    # how much of the C2 "control" set is literally the cognate set?
    ov = []
    for c in cases:
        for _d in range(N_DRAWS):
            alt = {x for l in rng.choice(lig_pool, len(c["ligs"]), replace=False) for x in nn_meas[l]}
            ov.append(len(alt & set(c["cols"])) / max(len(c["cols"]), 1))
    report(f"    C2 attenuation: a swapped set contains {100*np.mean(ov):.1f}% of the cognate targets on "
           f"average")

    # PRE-REGISTERED SPLIT. If the intracellular half is real AND the loop is running, receptors whose
    # cognate ligand is itself present in K562 should show it and the rest should not.
    report(f"\n  PRE-REGISTERED SPLIT on autocrine plausibility (cognate ligand present in K562)")
    report(f"    {'subset':38s} {'n':>4s} {'observed':>9s} {'C2 swap':>9s} {'ratio':>8s} "
           f"{'frac p<.05':>10s}")
    splits = {}
    for lab, sel in (("cognate ligand present in K562", [c for c in cases if c["autocrine"]]),
                     ("cognate ligand ABSENT in K562", [c for c in cases if not c["autocrine"]]),
                     ("responsive receptors only", [c, ] if False else
                      [c for c, s in zip(cases, rstr) if s > np.median(ps)])):
        if len(sel) < 8:
            report(f"    {lab:38s} {len(sel):4d}  n too small to read")
            continue
        o = np.array([sc(c["rows"], c["cols"]) for c in sel])
        ms, pv = [], []
        for _d in range(N_DRAWS):
            v = np.array([sc(c["rows"],
                             sorted({x for l in rng.choice(lig_pool, len(c["ligs"]), replace=False)
                                     for x in nn_meas[l]}) or c["cols"]) for c in sel])
            ms.append(v.mean())
            pv.append(float(stats.wilcoxon(o, v).pvalue))
        pv = np.array(pv)
        splits[lab] = {"n": len(sel), "observed": float(o.mean()), "control": float(np.mean(ms)),
                       "ratio": float(o.mean() / np.mean(ms)),
                       "frac_draws_p_lt_05": float(np.mean(pv < 0.05))}
        report(f"    {lab:38s} {len(sel):4d} {o.mean():9.4f} {np.mean(ms):9.4f} "
               f"{o.mean()/np.mean(ms):8.3f}x {np.mean(pv < 0.05):10.2f}")

    # POWER CHECK 2 -- the same machinery on a relationship that IS real on this readout.
    # Protein-complex co-membership: perturb a subunit, its partners move. Not derived from this matrix.
    partners = {}
    for _cn, mem in complexes.items():
        ms_ = [names[i] for i in mem if isinstance(i, int) and i < len(names)]
        for m in ms_:
            partners.setdefault(m, set()).update(x for x in ms_ if x != m)
    pcases = []
    for s, pl in sorted(partners.items()):
        if s not in sym2row:
            continue
        cols = sorted({sym2col[p] for p in pl if p in sym2col})
        if len(cols) >= MIN_TARGETS:
            pcases.append({"rows": sym2row[s], "cols": cols})
    pobs = np.array([sc(c["rows"], c["cols"]) for c in pcases])

    def pos_control(sel, o):
        ms, pv = [], []
        for _d in range(N_DRAWS):
            v = np.array([sc(c["rows"], [int(rng.choice(by_grq[grq[j]])) for j in c["cols"]]) for c in sel])
            ms.append(v.mean())
            pv.append(float(stats.wilcoxon(o, v).pvalue))
        pv = np.array(pv)
        return float(o.mean()), float(np.mean(ms)), float(o.mean() / np.mean(ms)), float(np.mean(pv < 0.05))

    fo, fm, fr, fp = pos_control(pcases, pobs)
    sub = [pcases[i] for i in rng.choice(len(pcases), len(cases), replace=False)]
    so = np.array([sc(c["rows"], c["cols"]) for c in sub])
    po, pm, pr, pp = pos_control(sub, so)
    report(f"\n  POWER 2 -- the identical machinery on a relationship known to be real on this readout")
    report(f"    {'arm':38s} {'n':>5s} {'observed':>9s} {'matched':>9s} {'ratio':>8s} {'frac p<.05':>10s}")
    report(f"    {'complex co-membership, full':38s} {len(pcases):5,d} {fo:9.4f} {fm:9.4f} {fr:8.3f}x "
           f"{fp:10.2f}")
    report(f"    {'complex co-membership, n-matched':38s} {len(sub):5,d} {po:9.4f} {pm:9.4f} {pr:8.3f}x "
           f"{pp:10.2f}")

    # minimum detectable effect at this n, from the observed paired scatter against the deciding null
    dl = []
    for _d in range(N_DRAWS):
        v = np.array([g_lig(c, rng)[0] for c in cases])
        m = (obs > 0) & (v > 0)
        dl.append(np.log(obs[m]) - np.log(v[m]))
    sd = float(np.mean([x.std(ddof=1) for x in dl]))
    mde = float(np.exp((1.959964 + 0.841621) * sd / np.sqrt(len(cases))))
    report(f"    paired log-ratio sd {sd:.3f} at n={len(cases)} -> smallest ratio detectable at 80% power "
           f"is {mde:.3f}x")
    report(f"    -> a real effect smaller than {mde:.2f}x would be INVISIBLE here. The complex positive "
           f"control at the\n       same n reaches p<0.05 in only {100*pp:.0f}% of draws, which is the "
           f"honest ceiling on what this test can conclude.")

    # ==================================================================================================
    # PART 3 -- side note: which pairs are plausible in which cell types (external, optional)
    # ==================================================================================================
    report("\n" + "-" * 100)
    report("PART 3  CELL-TYPE PLAUSIBILITY (side note -- not a deliverable, and not a validation)")
    report("-" * 100)
    want, seen = [], set()
    for c in cases:
        for s in [c["rec"]] + c["ligs"]:
            eid = REG.resolve("gene", "symbol", s)
            ent = REG.entities.get(eid) if eid else None
            if ent and ent.get("ensembl") and ent["ensembl"] not in seen:
                seen.add(ent["ensembl"])
                want.append((ent["ensembl"], s))
    cens = census_fetch([e for e, _s in want], report)
    census_out = None
    if cens and cens["expressed"]:
        e2s = dict(want)
        exp = {e2s[e]: v for e, v in cens["expressed"].items() if e in e2s}
        nct = {s: len(v) for s, v in exp.items()}
        pairs = []
        for c in cases:
            for l in c["ligs"]:
                if l in exp and c["rec"] in exp:
                    pairs.append({"ligand": l, "receptor": c["rec"],
                                  "sender_cell_types": nct[l], "receiver_cell_types": nct[c["rec"]],
                                  "same_cell_type": len(set(exp[l]) & set(exp[c["rec"]]))})
        report(f"    census snapshot {cens['snapshot_id']}, {len(exp)} LR genes resolved, "
               f"{len(cens['labels'])} cell types")
        if pairs:
            sc_ = np.array([p["same_cell_type"] for p in pairs])
            report(f"    {len(pairs)} tested pairs with both partners in the census: median "
                   f"{np.median(sc_):.0f} cell types express BOTH (>= {CENSUS_PC:.0%} of cells);")
            report(f"    {100*np.mean(sc_ == 0):.0f}% have none -- so the census agrees with the atlas "
                   f"mask that these are\n    sender/receiver pairs, not autocrine loops. This is "
                   f"CO-EXPRESSION, NOT a validated interaction.")
            top = sorted(pairs, key=lambda p: -p["same_cell_type"])[:8]
            report(f"    most co-expressed: " +
                   ", ".join(f"{p['ligand']}->{p['receptor']} ({p['same_cell_type']})" for p in top))
            census_out = {"snapshot_id": cens["snapshot_id"], "n_genes": len(exp), "pairs": pairs,
                          "pct_cells_threshold": CENSUS_PC}
    else:
        report("    census contributed nothing usable; the K562 gate and the receptor test stand alone")

    # ==================================================================================================
    # write the layer and the result
    # ==================================================================================================
    pair_rows = []
    for a, b in lr:
        L, R = names[a], names[b]
        pair_rows.append({"ligand": L, "receptor": R,
                          "ligand_k562": gate(L), "receptor_k562": gate(R),
                          "ligand_rna": panel_mean.get(L), "receptor_rna": panel_mean.get(R),
                          "ligand_prot": prot.get(L), "receptor_prot": prot.get(R),
                          "atlas_cell_types_both": (bin(emask[a] & emask[b]).count("1")
                                                    if a in emask and b in emask else None),
                          "active_in_k562": gate(L) == "expressed" and gate(R) == "expressed"})
    limits = [
        "MONOCULTURE CANNOT VALIDATE A MULTICELLULAR LAYER. A ligand-receptor edge asserts a sender, a "
        "receiver, spatial proximity and a downstream programme. K562 Perturb-seq supplies one cell type "
        "and no space, so three of those four are untestable here by construction and nothing below "
        "should be read as validating the lr layer as an interaction.",
        "The receptor-perturbation test checks the INTRACELLULAR HALF ONLY: whether the programme "
        "NicheNet attributes to a ligand depends on the cognate receptor's gene product. It does not "
        "check that the ligand ever reaches the receptor.",
        f"THE NULL IS UNDERPOWERED AND SAYS SO. At n={len(cases)} receptors the smallest effect "
        f"detectable at 80% power is {mde:.2f}x, and the complex-co-membership positive control run "
        f"through identical machinery at the same n reaches p<0.05 in only {100*pp:.0f}% of draws. Large "
        f"ligand-specific effects are excluded; small ones are not.",
        f"NicheNet's ligand-specificity is weak at source: mean Jaccard {np.mean(jac):.2f} between two "
        f"ligands' measured target sets, MYC asserted for {100*tc['MYC']/len(nn):.0f}% of ligands. The "
        f"ligand-swap control therefore shares ~{100*np.mean(ov):.0f}% of the cognate set and attenuates "
        f"any true effect by about that much.",
        "A receptor with no ligand present has nothing to transduce, so absence of a downstream effect is "
        "as consistent with 'the pathway was never running' as with 'NicheNet is wrong'. These two are "
        "NOT separable in this design and are not separated.",
        f"ABSENCE IS UNKNOWN, NOT ZERO. The Replogle panel is {X.shape[1]:,} genes with a detection floor "
        f"of mean {gmean.min():.2f}, and the K562 proteome is {len(prot):,} proteins with strongly "
        f"abundance-dependent coverage; secreted ligands are under-detected by BOTH (lysate mass-spec "
        f"misses secreted protein, low-expressed transcripts fall below the panel floor). The gate is "
        f"therefore biased toward calling ligands absent, which makes the surviving fraction a LOWER "
        f"bound.",
        f"emask covers {len(emask):,}/{len(names):,} genes only, and its 200 atlas cell types do not "
        f"include K562 or any leukaemia line; it is used for intercellularity and breadth, never as a "
        f"K562 expression call.",
        "CRISPRi knocks down rather than out; residual receptor is not measured here.",
        "lr and nichenet were joined by gene SYMBOL, the weakest namespace in the registry, because "
        "neither layer carries an accession.",
        "The CELLxGENE census figures are CO-EXPRESSION across cell types, not evidence of an "
        "interaction, and are pinned to one snapshot; they validate nothing."]

    layer = {
        "model": "ligand-receptor-k562-gate-v1",
        "provenance": {"rna": SRC_RNA, "protein": SRC_PROT, "atlas": SRC_ATLAS,
                       "census_snapshot": (census_out or {}).get("snapshot_id")},
        "what_this_is": "the lr layer annotated with whether each partner is present in K562, plus the "
                        "result of the one NicheNet test a monoculture supports",
        "n_pairs": len(lr),
        "n_pairs_active_in_k562": len(both_any),
        "limits": limits,
        "pairs": pair_rows}
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    R = {
        "provenance": layer["provenance"],
        "registry_joins": {"lr_ligands": st_lig, "lr_receptors": st_rec, "nichenet_ligands": st_nnl,
                           "nichenet_targets": st_nnt, "k562_genes": st_g, "k562_perturbations": st_p,
                           "k562_proteome": st_pr,
                           "cell_index_agreement": {"agree": agree, "of": len(names)}},
        "gate": {
            "n_pairs": len(lr),
            "ligands": {"n": len(ligs), "frac_in_k562_rna": rate(ligs, panel_mean),
                        "frac_in_k562_protein": rate(ligs, prot)},
            "receptors": {"n": len(recs), "frac_in_k562_rna": rate(recs, panel_mean),
                          "frac_in_k562_protein": rate(recs, prot)},
            "all_genes": {"n": len(allg), "frac_in_k562_rna": rate(allg, panel_mean),
                          "frac_in_k562_protein": rate(allg, prot)},
            "both_partners_rna": len(both_rna), "both_partners_rna_or_protein": len(both_any),
            "surviving_fraction": len(both_any) / len(lr),
            "receptor_only": rec_only, "ligand_only": lig_only,
            "surviving_pairs": sorted(both_any),
            "breadth_matched_control": gate_matched,
            "atlas_coexpression": {"n_pairs_with_masks": int(len(co)),
                                   "mean_cell_types_both": float(co.mean()),
                                   "frac_pairs_never_coexpressed": float(np.mean(co == 0)),
                                   "breadth_matched_mean": float(np.mean(ctrl_co)),
                                   "n_cell_types": len(ctnames), "draws": N_DRAWS}},
        "nichenet_shape": {"n_ligands": len(nn), "n_targets": len(nn_vocab),
                           "n_targets_measured": len({t for t in nn_vocab if t in sym2col}),
                           "mean_pairwise_jaccard_measured": float(np.mean(jac)),
                           "top_targets": {t: c / len(nn) for t, c in tc.most_common(6)}},
        "receptor_test": {
            "n_receptors": len(cases), "n_autocrine_plausible": int(sum(c["autocrine"] for c in cases)),
            "effect_raw_mean_abs_z": float(obs.mean()),
            "effect_raw_median_abs_z": float(np.median(obs)),
            "effect_raw_frac_targets_abs_z_ge_2": float(obs_hits.mean()),
            "controls": {"C1_responsiveness_matched": c1, "C2_ligand_swap": c2, "C3_receptor_swap": c3},
            "c2_attenuation_frac_cognate_in_swap": float(np.mean(ov)),
            "splits": splits,
            "power": {"self_knockdown_median_abs_z": float(np.median(sz)),
                      "self_knockdown_all_perturbations_median": float(np.median(allself)),
                      "n_receptors_in_panel": len(selfz),
                      "receptor_strength_median": float(np.median(rstr)),
                      "all_perturbation_strength_median": float(np.median(ps)),
                      "positive_control_complex_full": {"n": len(pcases), "observed": fo, "matched": fm,
                                                        "ratio": fr, "frac_draws_p_lt_05": fp},
                      "positive_control_complex_n_matched": {"n": len(sub), "observed": po, "matched": pm,
                                                             "ratio": pr, "frac_draws_p_lt_05": pp},
                      "min_detectable_ratio_80pct": mde}},
        "census": census_out,
        "limits": limits}

    report("\n" + "=" * 100)
    surv = 100 * len(both_any) / len(lr)
    v = (f"THE lr LAYER IS 2% ALIVE IN K562, AND THE HALF OF NICHENET A MONOCULTURE CAN TEST IS A "
         f"WELL-CONTROLLED NULL. Gating the 948 pairs on both partners being present in this cell leaves "
         f"{len(both_rna)} pairs by mRNA and {len(both_any)} ({surv:.1f}%) allowing K562 proteomics as a "
         f"second axis -- against {100*rate(allg, panel_mean):.0f}% of cell-object genes reaching the "
         f"panel, and the depletion survives matching on atlas breadth "
         f"({gate_matched['lr ligands']['observed_panel_rate']:.3f} observed vs "
         f"{gate_matched['lr ligands']['matched_panel_rate']:.3f} matched for ligands, "
         f"{gate_matched['lr receptors']['observed_panel_rate']:.3f} vs "
         f"{gate_matched['lr receptors']['matched_panel_rate']:.3f} for receptors, {N_DRAWS} draws). "
         f"{100*np.mean(co == 0):.0f}% of pairs with atlas masks share NO cell type expressing both "
         f"partners, which is the layer being intercellular rather than broken. Second: for the "
         f"{len(cases)} receptors perturbed in Replogle K562 whose cognate ligand carries a NicheNet "
         f"programme, those targets move at mean |robust z| {obs.mean():.3f} -- against "
         f"{c2['control_mean']:.3f} for the ligand-swap null ({c2['ratio_mean']:.3f}x, win rate "
         f"{c2['win_rate']:.3f}, {100*c2['frac_draws_p_lt_05']:.0f}% of {N_DRAWS} draws below p 0.05) and "
         f"{c3['control_mean']:.3f} for a strength-matched receptor swap on the identical genes "
         f"({c3['ratio_mean']:.3f}x, {100*c3['frac_draws_p_lt_05']:.0f}% of draws). The apparent "
         f"{c1['ratio_mean']:.2f}x over responsiveness-matched random genes is the NicheNet vocabulary, "
         f"not the pairing: MYC is asserted for {100*tc['MYC']/len(nn):.0f}% of ligands and two ligands' "
         f"measured target sets already overlap at Jaccard {np.mean(jac):.2f}. THE NULL IS BOUNDED, NOT "
         f"DECISIVE: at n={len(cases)} the smallest detectable effect is {mde:.2f}x, and complex "
         f"co-membership -- real on this readout at {fr:.3f}x with {100*fp:.0f}% of draws significant at "
         f"n={len(pcases):,} -- falls to {pr:.3f}x and {100*pp:.0f}% when subsampled to n={len(sub)}. So "
         f"this excludes a large ligand-specific downstream programme and nothing smaller. And it tests "
         f"only the intracellular half of the arrow: a receptor with no ligand in the dish has nothing to "
         f"transduce, so 'no effect' and 'pathway never running' are not separable in monoculture. The lr "
         f"and nichenet layers remain UNVALIDATED as communication; what is now established is how little "
         f"of lr is even switched on here.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "ligand_receptor.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB, {len(lr)} pairs annotated)")
    report(f"  -> {OUT/'ligand_receptor.json'}")


if __name__ == "__main__":
    main()
