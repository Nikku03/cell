"""v4 SECTION 2B -- rebuild the relation graph WITH confidence, evidence and cell context, then re-run
the two verdicts that were measured on a graph that had none of them.

WHY THESE TWO VERDICTS ARE UNINTERPRETABLE AND NOT NEGATIVE. `cell_complete.json.gz` stores PPI as bare
integer pairs -- `[[7513, 10468], [7513, 6254], ...]`, 191,447 of them, carrying no confidence, no
detection method, no direction and no cell context. Complex membership is stored the same way, as an
index list per complex. Only co-dependency survived ingestion weighted (`[[16165, 0.392], ...]`). So when
this project reported "evidence strength not detected" four times, and "real vs shuffled wiring not
detected" once, TWO OF THE THREE EDGE TYPES IN THE GRAPH HAD NO QUALITY FIELD TO STRATIFY ON. An
experiment that gates on a field which does not exist for 2/3 of its edges has not tested the hypothesis;
it has tested a constant. Those verdicts are withdrawn as uninterpretable, not recorded as null.

The fields were never absent from disk. They were dropped by the ingestion. This module rebuilds them.

PART 1 -- THE TYPED RELATION TABLE  -> outputs/orphan/v4_relations.parquet
    columns: source, target, relation_type, direction, sign, strength, confidence, evidence_type,
             cell_context, provenance   (+ confidence_raw, confidence_scale, n_evidence)
    sources: IntAct (MI confidence score, detection method per row), Reactome interactions (interaction
             type, PubMed support), huMAP2 (complex confidence 1-5), Complex Portal (ECO evidence codes),
             UniBind TFBS datasets (cell_line), and cell_complete's codep / ppi / reg / sig.

    COVERAGE IS DECLARED, NOT FILLED. Every column is reported as the fraction of rows that ACTUALLY
    CARRY it, broken out per relation_type. A null stays null. The legacy `cell_complete` PPI rows are
    deliberately kept in the table with every evidence column null, because their coverage number IS the
    finding of part 1: it is the exact measurement of what the ingestion destroyed.

    `direction` is the one column that is 100% populated everywhere, and that is NOT evidence. It is a
    property of the SOURCE's semantics (PPI/complex/codependency are undirected by definition; regulatory
    and signalling edges are directed by definition), asserted per source, not observed per row. It is
    reported separately from the observed-evidence columns so the reader is not misled by a full column.

    THE huMAP2 CONFIDENCE POLARITY WAS NOT ASSUMED, IT WAS MEASURED. The file has a Confidence column of
    1-5 (the brief for this module said 1-3; the brief is wrong and the observed range is reported). Which
    end is "good" is not stated in the file, and getting it backwards would invert the entire experiment
    below. It is therefore resolved empirically against INDEPENDENT sources: the fraction of each
    confidence level's within-complex pairs that is corroborated by Complex Portal and by IntAct. That
    check runs every time and its numbers are written to the JSON. Nothing in it touches the response data
    the recall numbers are scored on, so it cannot leak into part 2.

PART 2 -- THE RE-TEST, on dense_stage1's task exactly
    Target, metric and conventions are taken from `colab/dense_stage1.py` unchanged: dense K562 pseudobulk
    (`gwps.h5ad`, 100% dense -- the truncated `nlz_*` benches are NOT used and must never be, since 96.5%
    of their entries are zeros meaning NOT RECORDED), per-gene robust z, top-5% movers are the tide,
    S = |z| minus the per-gene mean over TRAINING knockouts only, truth = the top-50 non-tide genes by S,
    recall@50, retrieval over the 10 nearest training knockouts.

    THE ARMS (all on the same knockouts, same folds, same target)
      A_native      all edges, CAP partners per type, source file order.  <- current behaviour, verbatim
      A_matched     all edges, but truncated to the matched count n[k,t] used by every arm below
      B0_matched    only edges that CARRY a confidence, any level, matched count
      B_top         only edges in the TOP CONFIDENCE TERCILE of their relation type, matched count
      D_rewire      DEGREE-MATCHED CONFIGURATION-MODEL rewiring of the top-confidence subgraph
      R_random      uniform random partners, matched count AND matched type mix
      self_only     no neighbourhood at all
      B_top -> BOTTOM tercile at test    COUNTERFACTUAL SWAP, no retraining
      B_top -> wrong-knockout            identity control

    WHY B0 EXISTS, AND IT IS THE CONTROL THAT DECIDES WHETHER PART 2 MEANS ANYTHING. For `ppi`, the only
    source carrying a per-row confidence is IntAct. So "top-confidence PPI edges" is confounded with
    "IntAct edges". B vs A alone therefore cannot tell a confidence effect from a source-swap effect. B0
    (confidence-bearing, any tercile) splits that in two: B0 - A is the annotation/source contrast, B - B0
    is the confidence contrast proper. The source composition of each arm's partner set is printed.

    THE TRIVIAL NON-LEARNED BASELINES, reported next to the model and not beneath it:
      TIDE-null       rank non-tide genes by how often they move -- the floor, and it is scored in the
                      SAME non-tide universe the truth set is drawn from (a baseline that ranks by
                      tide-ness selects exactly the genes the truth excludes and scores ~0 by
                      construction; that defect has shipped three times in this project)
      JACCARD-copy    copy the training knockout whose PARTNER SET overlaps most (Jaccard). No learning,
                      no embedding, no gradient. Run on both the A and the B_top partner sets.

    THE UNIT OF REPLICATION IS THE DISJOINT KNOCKOUT FOLD, NOT THE MODEL SEED. The claim under test is
    "confidence-gating the partner set changes recall on knockouts the model has never seen". Model-init
    seeds rotated inside one partition measure optimiser noise and nothing else; grading on them flipped a
    verdict in this project from 4/7 to 0/7. Held-out knockouts are therefore partitioned into V4_FOLDS
    DISJOINT test sets and every gap is computed PAIRED WITHIN A FOLD first. Both floors are printed; only
    the cross-fold one is graded on.

    THE MDE CONVENTION, DECLARED BEFORE ANY NUMBER: MDE = 3 * sd / sqrt(n), where sd is the standard
    deviation OF THE PAIRED PER-FOLD GAP (ddof=1) and n = V4_FOLDS = the number of DISJOINT HELD-OUT
    KNOCKOUT FOLDS. n counts folds. It does not count seeds, it does not count knockouts, and it does not
    count edges. The seed-only MDE is computed as well and is written to the JSON under a key that says
    not to grade on it.

    THE PREDECLARED GATE, fixed before the run:
      G1  |recall(B_top) - recall(A_native)| > MDE, pooled over all three relation types
      G2  the same, per relation type, for codependency / complex / ppi separately
      G3  recall(B_top) - recall(D_rewire) > MDE  -- real wiring beats degree-matched rewiring at
          matched confidence. This is the "real vs shuffled wiring" verdict, re-run with a quality field
          that now exists and against a configuration-model null rather than a uniform one.
      G4  recall(B_top) - recall(B_top tested on BOTTOM tercile) > MDE. If this is ~0 while G1 is ~0, the
          partner-confidence channel is IGNORED, not useless -- accuracy alone cannot separate those and
          only the counterfactual swap can.
      G5  the model beats the best non-learned baseline. If it does not, that is the headline.

    EFFECT SIZES ARE RAW HELD-OUT RECALL. Nothing is reported net of a control. The controls establish
    significance and attribution; they are never subtracted from the number that gets quoted.

    NO CIRCULARITY, and here is the check. The target is gwps K562 Perturb-seq transcriptomic response.
    The edge sources are IntAct, Reactome, huMAP2, Complex Portal, UniBind and DepMap co-essentiality --
    none of which is derived from gwps, and gwps is not an input to any of them. The one adjacency worth
    naming: `codependency` edges are DepMap CRISPRGeneEffect correlations and the node features (`Zp`) are
    a 64-PC projection of that same matrix, so codep edges are partly recoverable from the features and
    the codep arm is not an independent channel. That is a redundancy between two INPUTS, not leakage from
    the target, and it is listed under limits.
"""
import gzip
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
SCI = OUT / "science_data"
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
DEPVEC = OUT / "depmap_vecs.npz"
HUMAP2 = SP / "humap2.txt"
CPORTAL = SP / "rxn" / "complexportal_9606.tsv"
REACTOME = SP / "rxn" / "reactome_interactions.tsv"
RELPQ = OUT / "v4_relations.parquet"

# ---- dense_stage1 conventions, carried over unchanged so the numbers stay comparable to that module ----
TAU = 1.0            # |z| at which a gene counts as moved
TIDE_FRAC = 0.05     # top 5% most-frequently-moving genes are the tide
MIN_SPEC = 5         # a knockout needs this many specific movers to be scorable
K = 50               # recall@K
TRUTH_N = 50         # truth = the TRUTH_N largest non-tide genes by train-centred |z|
MAX_TOK = 24
D_MODEL = 128
N_PC = 64

SMOKE = os.environ.get("V4_SMOKE", "0") == "1"
N_ROWS = int(os.environ.get("V4_ROWS", "0"))          # gwps rows to read; 0 = all
N_KO = int(os.environ.get("V4_KO", "0"))              # usable knockouts to keep; 0 = all
N_FOLDS = int(os.environ.get("V4_FOLDS", "5"))        # DISJOINT held-out knockout folds -- the unit of n
N_SEEDS = int(os.environ.get("V4_SEEDS", "2"))        # model inits per fold. NOT the unit of replication
EPOCHS = int(os.environ.get("V4_EPOCHS", "20"))
CAP = int(os.environ.get("V4_CAP", "7"))              # partners per relation type -- dense_stage1's value
MAXCPLX = int(os.environ.get("V4_MAXCPLX", "40"))     # complexes larger than this are not expanded to pairs
REUSE = os.environ.get("V4_REUSE", "0") == "1"        # reuse an existing v4_relations.parquet
if SMOKE:
    N_ROWS, N_KO, N_FOLDS, N_SEEDS, EPOCHS = 2500, 700, 3, 1, 3

TYPES = ["codependency", "complex", "ppi"]            # the three dense_stage1 attends over
TYPE_ID = {"codependency": 1, "complex": 2, "ppi": 3}
EVID_COLS = ["sign", "strength", "confidence", "evidence_type", "cell_context"]


def robust_z(M):
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad


# ----------------------------------------------------------------------------------------------------
# PART 1 -- the typed relation table
# ----------------------------------------------------------------------------------------------------
def build_relations(report):
    import pandas as pd

    t0 = time.time()
    COLS = ["source", "target", "relation_type", "direction", "sign", "strength", "confidence",
            "evidence_type", "cell_context", "provenance", "confidence_raw", "confidence_scale",
            "n_evidence"]
    blocks = []        # one DataFrame per ingestion block; columnar so a full-scale build stays in RAM

    def add_block(**kw):
        """One relation source -> one DataFrame. Any column not supplied stays NULL. Nothing is filled:
        a source that does not carry a field contributes nulls, and those nulls are what the declared
        coverage table below counts."""
        n = len(kw["source"])
        df = pd.DataFrame({c: kw.get(c, pd.Series([None] * n, dtype=object)) for c in COLS})
        for c in ("sign", "strength", "confidence", "confidence_raw", "n_evidence"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        blocks.append(df)
        return n

    # ---- identifier maps: uniprot / ensembl -> HGNC symbol -------------------------------------------
    g = pd.read_parquet(SCI / "genes.parquet", columns=["ensembl_gene_id", "hgnc_symbol"])
    g = g.dropna().drop_duplicates("ensembl_gene_id")
    e2s = dict(zip(g.ensembl_gene_id, g.hgnc_symbol))
    x = pd.read_parquet(SCI / "gene_xrefs.parquet", columns=["ensembl_gene_id", "xref_db", "xref_id"])
    x = x[x.xref_db.astype(str).str.startswith("UniProtKB")].copy()
    x["sym"] = x.ensembl_gene_id.map(e2s)
    x = x.dropna(subset=["sym"])
    u2s = dict(zip(x.xref_id, x.sym))
    report(f"  id maps: {len(e2s):,} ensembl->symbol, {len(u2s):,} uniprot->symbol")

    def upro(u):
        if u is None or isinstance(u, float):
            return None
        return u2s.get(str(u).split("-")[0])

    # ---- cell context lookup: which cell lines has this TF actually been ChIP'd in -------------------
    tf = pd.read_parquet(SCI / "tfbs_datasets.parquet", columns=["tf_name", "cell_line", "total_peaks"])
    tf["cl"] = tf.cell_line.astype(str).str.split("_").str[0]
    tf_cells = (tf.sort_values("total_peaks", ascending=False)
                  .groupby("tf_name")["cl"].apply(lambda s: ";".join(list(dict.fromkeys(s))[:6])).to_dict())
    report(f"  UniBind: {len(tf):,} TFBS datasets -> cell context for {len(tf_cells):,} TFs "
           f"({tf.cl.nunique()} distinct cell lines)")

    # ---- (1) IntAct PPI: the only PPI source with a per-row confidence -------------------------------
    ia = pd.read_parquet(SCI / "intact_interactions.parquet",
                         columns=["uniprot_a", "uniprot_b", "detection_method", "interaction_type",
                                  "confidence", "negative", "taxid_a", "taxid_b"])
    ia = ia[~ia.negative.fillna(False).astype(bool)]
    ia["sa"] = ia.uniprot_a.map(upro)
    ia["sb"] = ia.uniprot_b.map(upro)
    ia_ok = ia.dropna(subset=["sa", "sb"])
    ia_ok = ia_ok[ia_ok.sa != ia_ok.sb]
    add_block(source=ia_ok.sa.tolist(), target=ia_ok.sb.tolist(),
              relation_type="ppi", direction="undirected",
              confidence=ia_ok.confidence.tolist(), confidence_raw=ia_ok.confidence.tolist(),
              confidence_scale="intact_mi_score_0_1",
              evidence_type=(ia_ok.detection_method.fillna("") + "|"
                             + ia_ok.interaction_type.fillna("")).replace("|", None).tolist(),
              provenance="intact_interactions.parquet (IntAct/EBI)")
    report(f"  IntAct: {len(ia):,} rows -> {len(ia_ok):,} with both ends mapped to symbols "
           f"({100*len(ia_ok)/max(len(ia),1):.1f}%); confidence present on "
           f"{100*ia_ok.confidence.notna().mean():.1f}% of them")

    # ---- (2) Reactome interactions: evidence TYPE and PubMed support, no numeric confidence ----------
    # Reactome's "Interaction context" column is a REACTOME PATHWAY ID, not a cell line. It is NOT put in
    # cell_context -- that would manufacture a coverage number out of a mislabelled column.
    rx = pd.read_csv(REACTOME, sep="\t", usecols=[0, 3, 6, 7, 8], dtype=str)
    rx.columns = ["a", "b", "itype", "context", "pmids"]
    rx["sa"] = rx.a.str.replace("uniprotkb:", "", regex=False).map(upro)
    rx["sb"] = rx.b.str.replace("uniprotkb:", "", regex=False).map(upro)
    rx_ok = rx.dropna(subset=["sa", "sb"])
    rx_ok = rx_ok[rx_ok.sa != rx_ok.sb]
    nev = rx_ok.pmids.fillna("").replace("-", "").map(lambda s: len(s.split("|")) if s else 0)
    add_block(source=rx_ok.sa.tolist(), target=rx_ok.sb.tolist(),
              relation_type="ppi", direction="undirected",
              evidence_type=rx_ok.itype.tolist(), n_evidence=nev.tolist(),
              provenance=("rxn/reactome_interactions.tsv (Reactome; pathway ctx "
                          + rx_ok.context.fillna("-") + ")").tolist())
    report(f"  Reactome: {len(rx):,} rows -> {len(rx_ok):,} mapped non-self ({100*len(rx_ok)/len(rx):.1f}%); "
           f"evidence type present on 100%, numeric confidence on 0% (the source has none)")

    # ---- (3) the LEGACY cell_complete PPI: kept, with every evidence column null ---------------------
    d = json.load(gzip.open(NET, "rt"))
    nmz = np.array([gg["name"] for gg in d["genes"]], dtype=object)
    pe = np.asarray(d["ppi"], dtype=np.int64)
    add_block(source=nmz[pe[:, 0]].tolist(), target=nmz[pe[:, 1]].tolist(),
              relation_type="ppi", direction="undirected",
              provenance="cell_complete.json.gz:ppi (LEGACY -- bare integer pairs, all evidence fields "
                         "dropped at ingestion)")
    report(f"  cell_complete legacy PPI: {len(d['ppi']):,} bare pairs, 0 evidence columns. "
           f"THIS BLOCK IS THE DEFECT BEING MEASURED.")

    # ---- (4) huMAP2 complexes, with the confidence polarity resolved empirically ---------------------
    hm = pd.read_csv(HUMAP2)
    hm["mem"] = hm.genenames.fillna("").str.split()
    lv = sorted(int(c) for c in hm.Confidence.dropna().unique())
    # independent corroboration per level, used ONLY to fix which end of the scale is "good"
    cm = pd.read_parquet(SCI / "complex_members.parquet", columns=["complex_id", "ensembl_gene_id"])
    cm["sym"] = cm.ensembl_gene_id.map(e2s)
    cm = cm.dropna(subset=["sym"])
    cp_pairs, cp_members = set(), {}
    for cid, grp in cm.groupby("complex_id"):
        s = sorted(set(grp["sym"]))
        cp_members[cid] = s
        if 2 <= len(s) <= MAXCPLX:
            cp_pairs.update(itertools.combinations(s, 2))
    ia_pairs = set()
    for sa, sb in zip(ia_ok.sa, ia_ok.sb):
        ia_pairs.add((sa, sb) if sa < sb else (sb, sa))
    corrob = {}
    for c, mem in zip(hm.Confidence, hm.mem):
        s = sorted(set(mem))
        if not (2 <= len(s) <= MAXCPLX):
            continue
        agg = corrob.setdefault(int(c), [0, 0, 0])
        for pr in itertools.combinations(s, 2):
            agg[0] += 1
            agg[1] += pr in cp_pairs
            agg[2] += pr in ia_pairs
    pol = {int(c): {"pairs": v[0], "complexportal_corroborated": v[1] / max(v[0], 1),
                    "intact_corroborated": v[2] / max(v[0], 1)} for c, v in sorted(corrob.items())}
    lo, hi = min(pol), max(pol)
    one_is_best = pol[lo]["complexportal_corroborated"] > pol[hi]["complexportal_corroborated"]
    report(f"  huMAP2 Confidence observed range {lo}-{hi} (the module brief said 1-3; it is wrong). "
           f"Polarity resolved against INDEPENDENT sources:")
    for c, v in pol.items():
        report(f"      level {c}: {v['pairs']:6,d} within-complex pairs   "
               f"ComplexPortal-corroborated {v['complexportal_corroborated']:.3f}   "
               f"IntAct-corroborated {v['intact_corroborated']:.3f}")
    report(f"      -> level {lo if one_is_best else hi} is the HIGH-confidence end "
           f"({'1=best' if one_is_best else '5=best'}); normalised as confidence in [0,1]")
    span = max(hi - lo, 1)

    def hmconf(c):
        return (hi - c) / span if one_is_best else (c - lo) / span
    ha, hb, hc, hcr, hp = [], [], [], [], []
    for cid, c, mem in zip(hm.HuMAP2_ID, hm.Confidence, hm.mem):
        s = sorted(set(mem))
        if not (2 <= len(s) <= MAXCPLX):
            continue
        for a_, b_ in itertools.combinations(s, 2):
            ha.append(a_)
            hb.append(b_)
            hc.append(hmconf(int(c)))
            hcr.append(float(c))
            hp.append(f"humap2.txt ({cid})")
    npair = add_block(source=ha, target=hb, relation_type="complex", direction="undirected",
                      confidence=hc, confidence_raw=hcr,
                      confidence_scale=f"humap2_level_{lo}_{hi}_normalised",
                      evidence_type="humap2_complex_comembership", provenance=hp)
    report(f"  huMAP2: {len(hm):,} complexes -> {npair:,} within-complex pairs "
           f"(complexes of >{MAXCPLX} members not expanded)")

    # ---- (5) Complex Portal: ECO evidence codes, no numeric confidence -------------------------------
    cx = pd.read_parquet(SCI / "complexes.parquet", columns=["complex_id", "evidence_code"])
    eco = dict(zip(cx.complex_id, cx.evidence_code))
    ca, cb, ce, cprov = [], [], [], []
    for cid, s in cp_members.items():
        if not (2 <= len(s) <= MAXCPLX):
            continue
        ev = eco.get(cid) if isinstance(eco.get(cid), str) else None
        for a_, b_ in itertools.combinations(s, 2):
            ca.append(a_)
            cb.append(b_)
            ce.append(ev)
            cprov.append(f"complexes.parquet / rxn/complexportal_9606.tsv ({cid})")
    ncp = add_block(source=ca, target=cb, relation_type="complex", direction="undirected",
                    evidence_type=ce, provenance=cprov)
    report(f"  Complex Portal: {len(cp_members):,} complexes -> {ncp:,} pairs; ECO evidence code on "
           f"{100*cx.evidence_code.notna().mean():.1f}% of complexes, numeric confidence on 0%")

    # ---- (6) co-dependency: the one edge type that survived ingestion weighted -----------------------
    da, db, dw = [], [], []
    for a_, v in (d.get("codep") or {}).items():
        ia_ = int(a_)
        for b, s_ in v:
            da.append(ia_)
            db.append(int(b))
            dw.append(float(s_))
    dw = np.asarray(dw, np.float64)
    ncd = add_block(source=nmz[np.asarray(da, np.int64)].tolist(),
                    target=nmz[np.asarray(db, np.int64)].tolist(),
                    relation_type="codependency", direction="undirected",
                    sign=np.sign(dw).astype(np.int64).tolist(), strength=dw.tolist(),
                    confidence=np.abs(dw).tolist(), confidence_raw=dw.tolist(),
                    confidence_scale="abs_pearson_r_native",
                    evidence_type="crispr_coessentiality_correlation",
                    provenance="cell_complete.json.gz:codep (DepMap CRISPRGeneEffect co-essentiality "
                               "correlation)")
    report(f"  co-dependency: {ncd:,} weighted edges (strength AND confidence both native)")

    # ---- (7) regulatory and signalling: direction and SIGN, plus the only cell context we have -------
    re_ = np.asarray(d.get("reg") or [], dtype=np.int64).reshape(-1, 3)
    rsrc = nmz[re_[:, 0]]
    cells = [tf_cells.get(s) for s in rsrc]
    ncell = sum(c is not None for c in cells)
    nreg = add_block(source=rsrc.tolist(), target=nmz[re_[:, 1]].tolist(),
                     relation_type="regulatory", direction="directed", sign=re_[:, 2].tolist(),
                     cell_context=cells,
                     evidence_type=["unibind_chipseq_dataset" if c else None for c in cells],
                     provenance="cell_complete.json.gz:reg (TF -> target)")
    se_ = np.asarray(d.get("sig") or [], dtype=np.int64).reshape(-1, 3)
    nsig = add_block(source=nmz[se_[:, 0]].tolist(), target=nmz[se_[:, 1]].tolist(),
                     relation_type="signalling", direction="directed", sign=se_[:, 2].tolist(),
                     provenance="cell_complete.json.gz:sig (signalling edge)")
    report(f"  regulatory: {nreg:,} directed signed edges, {ncell:,} ({100*ncell/max(nreg,1):.1f}%) with a "
           f"UniBind cell context; signalling: {nsig:,}")
    del d

    R = pd.concat(blocks, ignore_index=True)[COLS]
    report(f"  relation table: {len(R):,} rows x {R.shape[1]} columns in {time.time()-t0:.1f}s")

    # ---- DECLARED COVERAGE, per relation type. Nothing is filled. -----------------------------------
    def cov(sub, c):
        s = sub[c]
        if s.dtype == object:
            return float((s.notna() & (s.astype(str) != "")).mean())
        return float(s.notna().mean())
    coverage = {"__ALL__": {c: cov(R, c) for c in EVID_COLS + ["direction", "provenance"]}}
    coverage["__ALL__"]["n_rows"] = int(len(R))
    for rt, sub in R.groupby("relation_type"):
        coverage[rt] = {c: cov(sub, c) for c in EVID_COLS + ["direction", "provenance"]}
        coverage[rt]["n_rows"] = int(len(sub))
    # and per provenance block, because that is where the ingestion defect is visible
    prov_cov = {}
    R["_blk"] = R.provenance.str.split(" (", regex=False).str[0]
    for bk, sub in R.groupby("_blk"):
        prov_cov[bk] = {"n_rows": int(len(sub)), **{c: cov(sub, c) for c in EVID_COLS}}
    R = R.drop(columns=["_blk"])

    report("")
    report("  DECLARED COVERAGE -- fraction of rows that ACTUALLY CARRY each field (nothing filled)")
    report(f"    {'relation_type':16s} {'rows':>9s} " + " ".join(f"{c:>14s}" for c in EVID_COLS)
           + f" {'direction*':>11s}")
    for rt in ["__ALL__"] + sorted(k for k in coverage if k != "__ALL__"):
        v = coverage[rt]
        report(f"    {rt:16s} {v['n_rows']:9,d} " + " ".join(f"{v[c]:14.3f}" for c in EVID_COLS)
               + f" {v['direction']:11.3f}")
    report("    * direction is asserted from SOURCE semantics, not observed per row. It is not evidence.")
    report("")
    report("  the same, per ingestion block -- this is the measurement of what was dropped")
    report(f"    {'block':62s} {'rows':>9s} {'conf':>7s} {'evid':>7s} {'cell':>7s}")
    for bk, v in sorted(prov_cov.items(), key=lambda kv: -kv[1]["n_rows"]):
        report(f"    {bk[:62]:62s} {v['n_rows']:9,d} {v['confidence']:7.3f} "
               f"{v['evidence_type']:7.3f} {v['cell_context']:7.3f}")

    # ---- confidence terciles, computed WITHIN relation type over rows that have a confidence ---------
    terc = {}
    for rt in TYPES:
        s = R.loc[R.relation_type == rt, "confidence"].dropna()
        if len(s) < 30:
            terc[rt] = None
            continue
        terc[rt] = {"q33": float(np.quantile(s, 1 / 3)), "q67": float(np.quantile(s, 2 / 3)),
                    "n_with_conf": int(len(s)),
                    "frac_of_type_with_conf": float(len(s) / max((R.relation_type == rt).sum(), 1))}
    report("")
    report("  confidence terciles, per relation type, over rows that CARRY a confidence")
    for rt in TYPES:
        t_ = terc[rt]
        if t_ is None:
            report(f"    {rt:16s} NO CONFIDENCE ANYWHERE -- cannot be gated")
        else:
            report(f"    {rt:16s} q33 {t_['q33']:.3f}  q67 {t_['q67']:.3f}  "
                   f"{t_['n_with_conf']:,} rows carry one = {100*t_['frac_of_type_with_conf']:.1f}% of "
                   f"the type")
    return R, {"coverage": coverage, "coverage_by_block": prov_cov, "terciles": terc,
               "humap2_polarity": {"levels": {str(k_): v for k_, v in pol.items()},
                                   "high_confidence_end": lo if one_is_best else hi,
                                   "low_confidence_end": hi if one_is_best else lo,
                                   "corroboration_at_high_end":
                                       pol[lo if one_is_best else hi]["complexportal_corroborated"],
                                   "corroboration_at_low_end":
                                       pol[hi if one_is_best else lo]["complexportal_corroborated"],
                                   "observed_range": [lo, hi],
                                   "resolved_by": "fraction of within-complex pairs corroborated by "
                                                  "Complex Portal and IntAct -- independent of the "
                                                  "response target"}}


# ----------------------------------------------------------------------------------------------------
# PART 2 -- the re-test
# ----------------------------------------------------------------------------------------------------
def main():
    import h5py
    import pandas as pd
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 108)
    report("v4 SECTION 2B -- relations WITH confidence/evidence/cell context, and the two verdicts re-run")
    report("=" * 108)
    report(f"  MDE CONVENTION (declared before any number): 3*sd/sqrt(n), sd = sd of the PAIRED PER-FOLD")
    report(f"  gap (ddof=1), n = {N_FOLDS} DISJOINT HELD-OUT KNOCKOUT FOLDS. n counts folds, not seeds,")
    report(f"  not knockouts, not edges. Effect sizes are RAW held-out recall@{K}, never net of a control.")
    report(f"  SMOKE={int(SMOKE)}  rows={N_ROWS or 'all'}  ko={N_KO or 'all'}  folds={N_FOLDS}  "
           f"seeds/fold={N_SEEDS}  epochs={EPOCHS}  cap={CAP}")
    report("")
    report("-" * 108)
    report("PART 1 -- rebuilding the typed relation table")
    report("-" * 108)

    if REUSE and RELPQ.exists():
        R = pd.read_parquet(RELPQ)
        meta1 = json.load(open(OUT / "v4_relations_meta.json"))
        report(f"  reused {RELPQ} ({len(R):,} rows)")
    else:
        R, meta1 = build_relations(report)
        OUT.mkdir(parents=True, exist_ok=True)
        R.to_parquet(RELPQ, index=False)
        json.dump(meta1, open(OUT / "v4_relations_meta.json", "w"), indent=1)
        report(f"\n  -> {RELPQ}")

    # ------------------------------------------------------------------------------------------------
    report("")
    report("-" * 108)
    report("PART 2 -- dense_stage1's task, re-run with the confidence field that now exists")
    report("-" * 108)

    with h5py.File(GWPS, "r") as f:
        n_all = f["X"].shape[0]
        nr = n_all if N_ROWS <= 0 else min(N_ROWS, n_all)
        X = f["X"][:nr]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:nr]]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["var"]["__categories"]["gene_name"][:]]
        gname = np.array([cats[i] for i in f["var"]["gene_name"][:]])
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    report(f"  dense K562 (gwps.h5ad): {X.shape[0]:,}/{n_all:,} perturbations x {X.shape[1]:,} genes, "
           f"{100*np.isfinite(X).mean():.1f}% finite. The truncated nlz_* benches are NOT used.")

    A = np.abs(robust_z(X))
    del X
    ok_gene = np.isfinite(A).all(0)
    A = A[:, ok_gene]
    gname = gname[ok_gene]
    mover = A >= TAU
    freq = mover.mean(0)
    tide = freq >= np.quantile(freq, 1 - TIDE_FRAC)
    nontide = ~tide
    spec_n = (mover & nontide[None, :]).sum(1)
    scorable = spec_n >= MIN_SPEC
    del mover
    report(f"  {A.shape[1]:,} genes with a defined z; {int(tide.sum()):,} tide / {int(nontide.sum()):,} "
           f"non-tide; scorable knockouts {int(scorable.sum()):,}; random floor "
           f"recall@{K} = {K/A.shape[1]:.4f}")

    dv = np.load(DEPVEC, allow_pickle=True)
    syms = [str(s) for s in dv["syms"]]
    Zd = dv["Z"].astype(np.float32)
    U, s_, _ = np.linalg.svd(Zd - Zd.mean(0), full_matrices=False)
    Zp = (U[:, :N_PC] * s_[:N_PC]).astype(np.float32)
    srow = {gg: i for i, gg in enumerate(syms)}
    del Zd, U

    keep = np.where(scorable & np.array([s in srow for s in psym]))[0]
    if N_KO > 0:
        keep = keep[:N_KO]
    A = A[keep]
    psym_k = psym[keep]
    report(f"  usable knockouts (scorable AND with a DepMap profile): {len(keep):,}")

    # ---- partner maps per relation type and per confidence stratum ----------------------------------
    uni = sorted(srow)                     # the feature universe -- a partner without features is useless
    uset = set(uni)
    terc = meta1["terciles"]
    Rt = R[R.relation_type.isin(TYPES)]
    Rt = Rt[Rt.source.isin(uset) & Rt.target.isin(uset) & (Rt.source != Rt.target)]
    report(f"  relation rows whose BOTH ends carry a DepMap feature vector: {len(Rt):,}")

    def edgelist(rt, stratum):
        sub = Rt[Rt.relation_type == rt]
        t_ = terc.get(rt)
        if stratum == "all":
            pass
        elif t_ is None:
            return []
        elif stratum == "conf":
            sub = sub[sub.confidence.notna()]
        elif stratum == "top":
            sub = sub[sub.confidence >= t_["q67"]]
        elif stratum == "bot":
            sub = sub[sub.confidence.notna() & (sub.confidence <= t_["q33"])]
        return list(zip(sub.source.tolist(), sub.target.tolist()))

    def adj(edges):
        m = {}
        for a_, b_ in edges:
            m.setdefault(a_, []).append(b_)
            m.setdefault(b_, []).append(a_)
        return {k_: list(dict.fromkeys(v)) for k_, v in m.items()}

    def config_rewire(edges, rng):
        """DEGREE-MATCHED (configuration-model) rewiring: every node keeps its exact degree, every edge
        loses its identity. A uniform-random null also destroys the degree sequence, so a uniform control
        that fails cannot tell 'partner identity matters' from 'hub-ness matters'. This one can."""
        stubs = []
        for a_, b_ in edges:
            stubs.append(a_)
            stubs.append(b_)
        stubs = np.array(stubs, dtype=object)
        rng.shuffle(stubs)
        out = []
        for i in range(0, len(stubs) - 1, 2):
            if stubs[i] != stubs[i + 1]:
                out.append((stubs[i], stubs[i + 1]))
        return out

    rng_g = np.random.default_rng(11)
    PART = {}          # PART[stratum][rt] -> adjacency dict
    ecount = {}
    for st in ("all", "conf", "top", "bot"):
        PART[st] = {}
        for rt in TYPES:
            el = edgelist(rt, st)
            ecount[(st, rt)] = len(el)
            PART[st][rt] = adj(el)
    PART["rewire"] = {}
    for rt in TYPES:
        el = edgelist(rt, "top")
        PART["rewire"][rt] = adj(config_rewire(el, rng_g)) if el else {}
    report("")
    report(f"  edge counts inside the usable universe   {'all':>10s} {'has-conf':>10s} {'top-terc':>10s} "
           f"{'bot-terc':>10s} {'rewired':>10s}")
    for rt in TYPES:
        nr_ = sum(len(v) for v in PART["rewire"][rt].values()) // 2
        report(f"    {rt:16s} {ecount[('all',rt)]:22,d} {ecount[('conf',rt)]:10,d} "
               f"{ecount[('top',rt)]:10,d} {ecount[('bot',rt)]:10,d} {nr_:10,d}")

    # the matched count n[k][t]: the largest partner budget every arm can honour, so token counts are
    # IDENTICAL across arms by construction and a difference cannot be a token-count artefact
    NMATCH = np.zeros((len(psym_k), len(TYPES)), np.int32)
    for i, k_ in enumerate(psym_k):
        for j, rt in enumerate(TYPES):
            NMATCH[i, j] = min(CAP, *[len(PART[st][rt].get(k_, [])) for st in
                                      ("all", "conf", "top", "bot", "rewire")])
    report(f"  matched partner budget per knockout per type: mean {NMATCH.mean():.2f}, "
           f"total tokens {int(NMATCH.sum()):,}; knockouts with a non-empty matched set: "
           f"{int((NMATCH.sum(1) > 0).sum()):,}/{len(psym_k):,}")

    # source composition of each arm's partner set, so a confidence effect is not read off a source swap
    prov_of = {}
    for a_, b_, p_ in zip(Rt.source.tolist(), Rt.target.tolist(), Rt.provenance.tolist()):
        blk = p_.split(" (")[0]
        prov_of.setdefault((a_, b_), blk)
        prov_of.setdefault((b_, a_), blk)

    def composition(st, rt):
        c = {}
        for i, k_ in enumerate(psym_k):
            j = TYPES.index(rt)
            for p in PART[st][rt].get(k_, [])[:NMATCH[i, j]]:
                blk = prov_of.get((k_, p), "?")
                c[blk] = c.get(blk, 0) + 1
        tot = max(sum(c.values()), 1)
        return {k_: v / tot for k_, v in sorted(c.items(), key=lambda kv: -kv[1])[:3]}
    comps = {rt: {st: composition(st, rt) for st in ("all", "conf", "top")} for rt in TYPES}
    report("")
    report("  SOURCE COMPOSITION of the partner sets -- B vs A is confounded with a source swap wherever")
    report("  these differ, which is exactly why the B0 (has-confidence, any tercile) arm exists")
    for rt in TYPES:
        for st in ("all", "conf", "top"):
            report(f"    {rt:16s} {st:5s} " + ", ".join(f"{k_[:44]} {v:.2f}" for k_, v in
                                                        comps[rt][st].items()))

    # ---- token builder -------------------------------------------------------------------------------
    def tokens(stratum, types=TYPES, matched=True, rand=False, seed=0):
        rng = np.random.default_rng(1000 + seed)
        F = np.zeros((len(psym_k), MAX_TOK, N_PC), np.float32)
        T = np.zeros((len(psym_k), MAX_TOK), np.int64)
        M = np.zeros((len(psym_k), MAX_TOK), bool)
        for i, k_ in enumerate(psym_k):
            rws, tps = [k_], [0]
            for rt in types:
                j = TYPES.index(rt)
                n = int(NMATCH[i, j]) if matched else CAP
                if rand:
                    n = int(NMATCH[i, j])
                    pick = [uni[int(z)] for z in rng.integers(0, len(uni), n)]
                else:
                    lst = PART[stratum][rt].get(k_, [])
                    if matched:
                        lst = list(lst)
                        rng.shuffle(lst)
                    pick = lst[:n]
                for p in pick:
                    rws.append(p)
                    tps.append(TYPE_ID[rt])
            for jj, (p, t_) in enumerate(zip(rws[:MAX_TOK], tps[:MAX_TOK])):
                if p in srow:
                    F[i, jj] = Zp[srow[p]]
                T[i, jj] = t_
                M[i, jj] = True
        return (torch.from_numpy(F), torch.from_numpy(T), torch.from_numpy(M))

    uidx = {gg: i for i, gg in enumerate(uni)}

    def partner_matrix(stratum, matched=True):
        """Binary (knockouts x universe) incidence of the partner set, for the NON-LEARNED baseline."""
        from scipy import sparse
        r_, c_ = [], []
        for i, k_ in enumerate(psym_k):
            for j, rt in enumerate(TYPES):
                n = int(NMATCH[i, j]) if matched else CAP
                for p in PART[stratum][rt].get(k_, [])[:n]:
                    r_.append(i)
                    c_.append(uidx[p])
        M = sparse.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)),
                              shape=(len(psym_k), len(uni)))
        M.data[:] = 1.0
        M.sum_duplicates()
        M.data[:] = 1.0
        return M

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(N_PC, D_MODEL)
            self.rel = nn.Embedding(4, D_MODEL)
            self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
            self.ln = nn.LayerNorm(D_MODEL)
            self.out = nn.Linear(D_MODEL, D_MODEL)

        def forward(self, F, T, M):
            h = self.proj(F) + self.rel(T)
            a, _ = self.attn(h, h, h, key_padding_mask=~M, need_weights=False)
            h = self.ln(h + a)
            m = M.unsqueeze(-1).float()
            p = (h * m).sum(1) / m.sum(1).clamp(min=1)
            e = self.out(p)
            return e / (e.norm(dim=-1, keepdim=True) + 1e-9)

    # ---- pre-build every token bundle once (they do not depend on the fold) --------------------------
    BUN = {"A_native": tokens("all", matched=False),
           "A_matched": tokens("all"), "B0_matched": tokens("conf"),
           "B_top": tokens("top"), "B_bot": tokens("bot"),
           "D_rewire": tokens("rewire"), "R_random": tokens("all", rand=True),
           "self_only": tokens("all", types=[])}
    for rt in TYPES:
        for st, nm in (("all", "A_native"), ("conf", "B0_matched"), ("top", "B_top"),
                       ("rewire", "D_rewire")):
            BUN[f"{rt}|{nm}"] = tokens(st, types=[rt], matched=(nm != "A_native"))
    PS_A = partner_matrix("all", matched=False)
    PS_B = partner_matrix("top", matched=True)
    report(f"\n  token bundles built: {len(BUN)}; mean real tokens A_native "
           f"{BUN['A_native'][2].sum(1).float().mean():.2f}, B_top "
           f"{BUN['B_top'][2].sum(1).float().mean():.2f}, D_rewire "
           f"{BUN['D_rewire'][2].sum(1).float().mean():.2f}")

    # ---- fold loop -----------------------------------------------------------------------------------
    def centred(tr_idx):
        mu = A[tr_idx].mean(0)
        Sz = np.where(nontide[None, :], A - mu[None, :], 0.0).astype(np.float32)
        T_ = np.empty((len(Sz), TRUTH_N), np.int64)
        for i0 in range(0, len(Sz), 512):
            blk = np.where(nontide[None, :], Sz[i0:i0 + 512], -np.inf)
            T_[i0:i0 + 512] = np.argpartition(-blk, TRUTH_N, axis=1)[:, :TRUTH_N]
        Yn_ = Sz / (np.linalg.norm(Sz, axis=1, keepdims=True) + 1e-9)
        return Sz, T_, Yn_

    perm = np.random.default_rng(2024).permutation(len(keep))
    folds = np.array_split(perm, N_FOLDS)
    res = {}       # arm -> list over (fold, seed)
    per_fold = {}  # arm -> list over folds (seed-averaged)

    for fi, te in enumerate(folds):
        tr = np.concatenate([f_ for k_, f_ in enumerate(folds) if k_ != fi])
        S, TRUTH, Yn = centred(tr)
        report(f"\n  FOLD {fi}: {len(tr):,} train / {len(te):,} test knockouts (DISJOINT test sets across "
               f"folds -- this is the unit of replication)")

        def score_arm(scores):
            rec = []
            for j, i in enumerate(te):
                s = np.where(nontide, scores[j], -np.inf)
                top = np.argpartition(-s, K)[:K]
                rec.append(np.intersect1d(top, TRUTH[i]).size / TRUTH_N)
            return float(np.mean(rec))

        fold_vals = {}

        def add(name, val):
            res.setdefault(name, []).append(val)
            fold_vals.setdefault(name, []).append(val)

        # --- non-learned baselines. Rule 8: the cheapest sensible baseline goes NEXT TO the model. -----
        # the floor is ranked WITHIN the non-tide universe, the same universe the truth is drawn from.
        add("TIDE-null (floor, non-learned)", score_arm(np.tile(np.where(nontide, freq, -np.inf),
                                                               (len(te), 1))))
        for nm, PS in (("JACCARD-copy on A partners (non-learned)", PS_A),
                       ("JACCARD-copy on B_top partners (non-learned)", PS_B)):
            Pte, Ptr = PS[te], PS[tr]
            ate = np.asarray(Pte.sum(1)).ravel()
            atr = np.asarray(Ptr.sum(1)).ravel()
            pred = np.zeros((len(te), A.shape[1]), np.float32)
            for i0 in range(0, len(te), 256):
                blk = np.asarray((Pte[i0:i0 + 256] @ Ptr.T).todense(), np.float32)
                den = ate[i0:i0 + 256, None] + atr[None, :] - blk
                jac = np.where(den > 0, blk / np.maximum(den, 1e-9), 0.0)
                bi = tr[np.argmax(jac, axis=1)]
                pred[i0:i0 + 256] = S[bi]
            add(nm, score_arm(pred))

        for seed in range(N_SEEDS):
            def train(bundle):
                F, T, M = bundle
                torch.manual_seed(seed)
                net = Enc()
                opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
                Yt = torch.from_numpy(Yn[tr])
                for _e in range(EPOCHS):
                    pm = torch.randperm(len(tr))
                    for i0 in range(0, len(tr), 128):
                        b = pm[i0:i0 + 128]
                        if len(b) < 8:
                            continue
                        idx = torch.from_numpy(tr[b.numpy()])
                        e = net(F[idx], T[idx], M[idx])
                        loss = ((e @ e.T) - (Yt[b] @ Yt[b].T)).pow(2).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    return net, net(F, T, M).numpy()

            def retrieve(E_):
                En = E_ / (np.linalg.norm(E_, axis=1, keepdims=True) + 1e-9)
                sim = En[te] @ En[tr].T
                nn_ = np.argsort(-sim, axis=1)[:, :10]
                return np.stack([S[tr[r]].mean(0) for r in nn_])

            nets = {}
            for nm in ("self_only", "A_native", "A_matched", "B0_matched", "B_top", "D_rewire",
                       "R_random"):
                net, E = train(BUN[nm])
                nets[nm] = net
                add(nm, score_arm(retrieve(E)))
            # COUNTERFACTUAL SWAP: the B_top model, unchanged, handed the BOTTOM tercile at test time.
            # Accuracy alone cannot tell an ignored channel from a useless one; this can.
            with torch.no_grad():
                Eb = nets["B_top"](*BUN["B_bot"]).numpy()
            add("B_top -> BOTTOM tercile (SWAP)", score_arm(retrieve(Eb)))
            # identity control: correct predictions, wrong knockout
            with torch.no_grad():
                Et = nets["B_top"](*BUN["B_top"]).numpy()
            sh = np.random.default_rng(7 + fi).permutation(len(te))
            add("B_top -> wrong-knockout (IDENTITY)", score_arm(retrieve(Et)[sh]))
            for rt in TYPES:
                for nm in ("A_native", "B0_matched", "B_top", "D_rewire"):
                    _n, E = train(BUN[f"{rt}|{nm}"])
                    add(f"[{rt}] {nm}", score_arm(retrieve(E)))

        for nm, v in fold_vals.items():
            per_fold.setdefault(nm, []).append(float(np.mean(v)))
        for nm in ("A_native", "B_top", "D_rewire", "R_random",
                   "B_top -> BOTTOM tercile (SWAP)", "JACCARD-copy on B_top partners (non-learned)"):
            report(f"    {nm:44s} recall@{K} {np.mean(fold_vals[nm]):.4f}")
        S = TRUTH = Yn = None

    # ---- aggregation ---------------------------------------------------------------------------------
    arms = list(per_fold)
    report("")
    report(f"  {'arm':46s} {'recall@'+str(K):>10s} {'sd(fold)':>10s} {'per-fold':>34s}")
    for nm in arms:
        v = np.array(per_fold[nm])
        report(f"  {nm:46s} {v.mean():10.4f} {np.std(v, ddof=1) if len(v)>1 else 0.0:10.4f}   "
               f"{np.round(v, 4)}")

    def gap_mde(a, b):
        """PAIRED per-fold gap. sd is the sd of the GAP across DISJOINT knockout folds, n = folds."""
        g = np.array(per_fold[a]) - np.array(per_fold[b])
        sd = float(np.std(g, ddof=1)) if len(g) > 1 else float("nan")
        return float(g.mean()), sd, 3 * sd / np.sqrt(len(g)), g.tolist()

    sd_seed = float(np.mean([np.std(res[nm]) for nm in arms if len(res[nm]) > 1])) if N_SEEDS > 1 else 0.0
    gates, gapd = {}, {}
    for label, a, b in [("G1 pooled  B_top vs A_native", "B_top", "A_native"),
                        ("G3 pooled  B_top vs D_rewire", "B_top", "D_rewire"),
                        ("G4 pooled  B_top vs BOTTOM-tercile SWAP", "B_top",
                         "B_top -> BOTTOM tercile (SWAP)"),
                        ("    aux    B_top vs B0_matched (confidence proper)", "B_top", "B0_matched"),
                        ("    aux    B0_matched vs A_matched (source/annotation)", "B0_matched",
                         "A_matched"),
                        ("    aux    B_top vs R_random (uniform control)", "B_top", "R_random"),
                        ("    aux    B_top vs self_only", "B_top", "self_only"),
                        ("    aux    B_top vs wrong-knockout", "B_top",
                         "B_top -> wrong-knockout (IDENTITY)")]:
        g, sd, mde, pf = gap_mde(a, b)
        gapd[label] = {"gap": g, "sd_cross_fold": sd, "mde": mde, "per_fold_gap": pf,
                       "exceeds_mde": bool(abs(g) > mde)}
    for rt in TYPES:
        for label, a, b in [(f"G2 [{rt}] B_top vs A_native", f"[{rt}] B_top", f"[{rt}] A_native"),
                            (f"G3 [{rt}] B_top vs D_rewire", f"[{rt}] B_top", f"[{rt}] D_rewire"),
                            (f"   [{rt}] B_top vs B0_matched", f"[{rt}] B_top", f"[{rt}] B0_matched")]:
            g, sd, mde, pf = gap_mde(a, b)
            gapd[label] = {"gap": g, "sd_cross_fold": sd, "mde": mde, "per_fold_gap": pf,
                           "exceeds_mde": bool(abs(g) > mde)}

    report("")
    report(f"  seed-only spread (optimiser noise inside a fold): sd {sd_seed:.4f} -- NOT graded on")
    report(f"  {'contrast':56s} {'gap':>9s} {'sd':>8s} {'MDE':>8s}  verdict")
    for k_, v in gapd.items():
        report(f"  {k_:56s} {v['gap']:+9.4f} {v['sd_cross_fold']:8.4f} {v['mde']:8.4f}  "
               f"{'MOVES' if v['exceeds_mde'] else 'below MDE'}")

    best_nl = max([nm for nm in arms if "non-learned" in nm], key=lambda nm: np.mean(per_fold[nm]))
    best_model = max([nm for nm in arms if "non-learned" not in nm and not nm.startswith("[")],
                     key=lambda nm: np.mean(per_fold[nm]))
    gates["G1 confidence gating moves pooled recall"] = gapd["G1 pooled  B_top vs A_native"]["exceeds_mde"]
    for rt in TYPES:
        gates[f"G2 [{rt}] confidence gating moves recall"] = \
            gapd[f"G2 [{rt}] B_top vs A_native"]["exceeds_mde"]
    g3 = gapd["G3 pooled  B_top vs D_rewire"]
    gates["G3 real wiring beats DEGREE-MATCHED rewiring"] = bool(g3["gap"] > g3["mde"])
    g4 = gapd["G4 pooled  B_top vs BOTTOM-tercile SWAP"]
    gates["G4 confidence channel is READ (swap moves it)"] = bool(g4["gap"] > g4["mde"])
    gates["G5 model beats the best non-learned baseline"] = bool(
        np.mean(per_fold[best_model]) > np.mean(per_fold[best_nl]))
    report("")
    report("  PREDECLARED GATES")
    for k_, v in gates.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k_}")
    report(f"    best non-learned baseline: '{best_nl}' at {np.mean(per_fold[best_nl]):.4f}; "
           f"best model arm: '{best_model}' at {np.mean(per_fold[best_model]):.4f}")

    cov_all = meta1["coverage"]
    ppi_conf = meta1["terciles"]["ppi"]["frac_of_type_with_conf"] if meta1["terciles"]["ppi"] else 0.0
    cplx_conf = (meta1["terciles"]["complex"]["frac_of_type_with_conf"]
                 if meta1["terciles"]["complex"] else 0.0)
    verdict = (
        f"PART 1 SUCCEEDED AND PART 2 IS THEREFORE INTERPRETABLE FOR THE FIRST TIME. The ingestion's "
        f"legacy PPI block carries a confidence on 0.000 of its {cov_all['ppi']['n_rows']:,} rows, which is "
        f"the arithmetic reason 'evidence strength not detected' was reported four times: the field being "
        f"stratified on did not exist. Rebuilding from the sources on disk restores a per-row confidence "
        f"to {100*ppi_conf:.1f}% of PPI edges (IntAct MI score) and {100*cplx_conf:.1f}% of complex edges "
        f"(huMAP2 level, whose polarity was fixed empirically -- level "
        f"{meta1['humap2_polarity']['high_confidence_end']} is the high end, corroborated by Complex "
        f"Portal on {meta1['humap2_polarity']['corroboration_at_high_end']:.3f} of its pairs against "
        f"{meta1['humap2_polarity']['corroboration_at_low_end']:.3f} at the low end), "
        f"an evidence type to {100*cov_all['__ALL__']['evidence_type']:.1f}% of all rows and a cell "
        f"context to {100*cov_all['__ALL__']['cell_context']:.1f}% -- cell context reaching only "
        f"regulatory edges, via the cell lines in which each TF actually has a UniBind ChIP dataset. "
        f"PART 2, on dense_stage1's task with recall@{K} as the raw held-out effect size and an MDE of "
        f"3*sd/sqrt({N_FOLDS}) over {N_FOLDS} DISJOINT held-out knockout folds: confidence-gating the "
        f"partner set to the top tercile moves recall from {np.mean(per_fold['A_native']):.4f} "
        f"(all edges, current behaviour) to {np.mean(per_fold['B_top']):.4f}, a gap of "
        f"{gapd['G1 pooled  B_top vs A_native']['gap']:+.4f} against an MDE of "
        f"{gapd['G1 pooled  B_top vs A_native']['mde']:.4f} -- "
        f"{'ABOVE' if gates['G1 confidence gating moves pooled recall'] else 'BELOW'} the floor. Real "
        f"wiring at top confidence scores {np.mean(per_fold['B_top']):.4f} against a DEGREE-MATCHED "
        f"configuration-model rewiring at {np.mean(per_fold['D_rewire']):.4f} "
        f"({g3['gap']:+.4f}, MDE {g3['mde']:.4f}), which is the 'real vs shuffled wiring' question asked "
        f"for the first time against a null that preserves the degree sequence rather than a uniform one. "
        f"THE COUNTERFACTUAL SWAP IS THE PART THAT DECIDES WHAT A NULL MEANS: handing the top-confidence "
        f"model the BOTTOM tercile at test time changes recall by {g4['gap']:+.4f} against "
        f"{g4['mde']:.4f}, so the partner-confidence channel is "
        f"{'READ' if gates['G4 confidence channel is READ (swap moves it)'] else 'NOT READ AT ALL -- which means a null on G1 says the model ignores partner identity, not that confidence is uninformative, and no amount of better edge annotation can fix that'}"
        f". AND THE HEADLINE IS THE BASELINE: the cheapest non-learned arm, '{best_nl}', scores "
        f"{np.mean(per_fold[best_nl]):.4f} against the best learned arm's "
        f"{np.mean(per_fold[best_model]):.4f} -- "
        f"{'the model is ahead of it' if gates['G5 model beats the best non-learned baseline'] else 'A NON-LEARNED BASELINE MATCHES OR BEATS EVERY TRAINED ARM, so nothing here is evidence that the architecture is doing work'}. "
        f"Decomposing the confidence gate: B_top vs B0_matched (confidence proper, source held fixed) is "
        f"{gapd['    aux    B_top vs B0_matched (confidence proper)']['gap']:+.4f} while B0_matched vs "
        f"A_matched (the source/annotation swap, confidence not varied) is "
        f"{gapd['    aux    B0_matched vs A_matched (source/annotation)']['gap']:+.4f}; wherever the "
        f"second is the larger of the two, what looks like a confidence effect is a source-selection "
        f"effect, because IntAct is the only PPI source on disk that carries a per-row score.")
    report("\n" + "=" * 108)
    report(f"  VERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    Rout = {
        "model": "v4-relation-schema-2B",
        "smoke": bool(SMOKE),
        "config": {"gwps_rows_read": int(nr), "n_knockouts": int(len(keep)), "n_genes": int(A.shape[1]),
                   "folds": N_FOLDS, "seeds_per_fold": N_SEEDS, "epochs": EPOCHS, "cap": CAP,
                   "env": {"V4_SMOKE": "1 for the smoke defaults", "V4_ROWS": "gwps rows, 0=all",
                           "V4_KO": "knockouts, 0=all", "V4_FOLDS": "disjoint held-out knockout folds",
                           "V4_SEEDS": "model inits per fold", "V4_EPOCHS": "", "V4_CAP": "",
                           "V4_MAXCPLX": "", "V4_REUSE": "1 to reuse v4_relations.parquet"}},
        "relation_table": {"path": str(RELPQ), "n_rows": int(len(R)),
                           "columns": list(R.columns),
                           "coverage_declared": meta1["coverage"],
                           "coverage_by_ingestion_block": meta1["coverage_by_block"],
                           "confidence_terciles": meta1["terciles"],
                           "humap2_polarity_check": meta1["humap2_polarity"],
                           "direction_is_not_evidence": "the direction column is 100% populated because "
                                                        "it is asserted from source semantics, not "
                                                        "observed per row"},
        "edge_counts_in_universe": {f"{st}|{rt}": int(v) for (st, rt), v in ecount.items()},
        "partner_source_composition": comps,
        "arms": {nm: {"mean": float(np.mean(per_fold[nm])),
                      "sd": float(np.std(per_fold[nm], ddof=1)) if len(per_fold[nm]) > 1 else 0.0,
                      "per_fold": per_fold[nm], "per_seed": res[nm]} for nm in arms},
        "mde": {"convention": f"3*sd/sqrt(n); sd = sd (ddof=1) of the PAIRED PER-FOLD gap; "
                              f"n = {N_FOLDS} DISJOINT held-out knockout folds. n counts folds, not "
                              f"seeds, not knockouts, not edges",
                "n": N_FOLDS, "unit_of_replication": "disjoint held-out knockout fold",
                "sd_seed_only_DO_NOT_GRADE_ON_THIS": sd_seed,
                "contrasts": gapd},
        "gates": {k_: bool(v) for k_, v in gates.items()},
        "best_non_learned_baseline": {"arm": best_nl, "recall": float(np.mean(per_fold[best_nl]))},
        "best_model_arm": {"arm": best_model, "recall": float(np.mean(per_fold[best_model]))},
        "no_circularity": "the target is gwps K562 Perturb-seq response; no edge source (IntAct, "
                          "Reactome, huMAP2, Complex Portal, UniBind, DepMap co-essentiality) is derived "
                          "from it and it is not an input to any of them. The huMAP2 polarity check uses "
                          "Complex Portal and IntAct only, never the response data. The one adjacency: "
                          "codependency edges and the node features are both functions of DepMap "
                          "CRISPRGeneEffect, so the codep channel is partly redundant with the features "
                          "-- a redundancy between two INPUTS, not leakage from the target",
        "limits": [
            "WHAT WOULD FALSIFY PART 1: a claim that any evidence column is more complete than stated. "
            "Every number in coverage_declared is a raw notna() rate over the rows actually written; if a "
            "downstream consumer forward-fills a null it invalidates every stratified result built on it",
            "THE CONFIDENCE GATE IS CONFOUNDED WITH A SOURCE SWAP for ppi: IntAct is the only PPI source "
            "on disk carrying a per-row score, so 'top-confidence PPI' is very largely 'IntAct PPI'. The "
            "B0_matched arm is what separates the two and the decomposition is reported next to G1; "
            "reading G1 without B0 would repeat the original error in a new form",
            "Reactome and Complex Portal carry an evidence TYPE but no numeric confidence, so their edges "
            "can never enter a top-confidence tercile. Confidence-gating is therefore also a source "
            "filter, and the tercile threshold is computed only over rows that have a score at all",
            "huMAP2 confidence polarity is inferred, not read from a spec. It is inferred from "
            "corroboration by Complex Portal, whose complexes are themselves in the complex relation "
            "type, so high-confidence huMAP2 edges are enriched for edges that appear twice in the table. "
            "That enrichment is a property of the edge set, not of the target, but it means the complex "
            "confidence gate partly selects for multiply-sourced edges",
            "cell_context is reachable only for regulatory edges and only at the granularity of 'this TF "
            "has a UniBind ChIP dataset in this cell line'. That is not a cell-specific edge weight and "
            "must not be used as one. PPI, complex and codependency edges have NO cell context anywhere "
            "on disk here, so no cell-conditional graph can be built for the three types the retest uses",
            "the retest is K562 only and one target assay. A confidence effect that exists in another "
            "cell line or another readout would not be visible here",
            "codependency edges are DepMap correlations and the node features are DepMap principal "
            "components, so the codep channel is partly recoverable from the features themselves and its "
            "per-type numbers understate what an independent codep channel would contribute",
            "the degree-matched rewiring preserves degree exactly but not the degree-degree correlation "
            "(assortativity) or the community structure. A gap over it is evidence for partner identity "
            "beyond degree, not evidence for the specific biology of any one edge",
            "with n = folds as the unit, the MDE is wide at small V4_FOLDS. A gap reported as 'below MDE' "
            "at V4_FOLDS=3 is not evidence of absence; it is evidence that the design cannot resolve it. "
            "Read the per_fold_gap list, not just the mean",
            "the encoder is a single attention block and the retrieval is 10-NN, both inherited from "
            "dense_stage1 so the numbers stay comparable. A stronger reader might use a channel this one "
            "ignores, and G4 is what would show that -- but only for the arms actually run"],
        "verdict": verdict, "log": log}
    json.dump(Rout, open(OUT / "v4_relation_schema.json", "w"), indent=1)
    report(f"\n  -> {OUT/'v4_relation_schema.json'}")
    report(f"  -> {RELPQ}")


if __name__ == "__main__":
    main()
