"""V4 BLOCK 8 -- a protein head from ECCITE, against the NAIVE TRANSLATOR. Does RNA buy anything?

WHAT WAS ASKED FOR AND WHAT IS ACTUALLY ON DISK, STATED FIRST BECAUSE THE PREMISE WAS WRONG.
`papalexi.h5ad` was handed to this module as "ECCITE-seq: paired RNA and surface-protein (ADT) in the same
cells". IT IS NOT. The file is RNA ONLY: a (20,729 cell x 18,649 gene) CSC count matrix, with `obsm`,
`layers`, `uns`, `obsp`, `varm` and `varp` ALL PRESENT AND ALL EMPTY, and an `obs` of 17 columns (hto,
guide_id, hto_barcode, gdo_barcode, perturbation, tissue_type, cell_line, cancer, disease,
perturbation_type, celltype, organism, nperts, ngenes, ncounts, percent_mito, percent_ribo). There is no
protein modality anywhere inside it. The four antibody-derived tags do appear in `var` -- but as the
TRANSCRIPTS CD86, CD274, PDCD1LG2 and HAVCR2, which is the naive translator's input, not the target.

The protein modality for these exact cells exists as a SEPARATE GEO file, GSE153056 / GSM4633615
`ECCITE_ADT_counts.tsv.gz`: 4 ADT tags x 20,729 cells. That is not a substitute dataset -- it is the other
half of the same experiment, on the same barcodes, and this project has already joined it once
(`mrna_vs_protein.py`) at a verified 20,729/20,729 exact positional agreement, which this module re-asserts
rather than assumes. FOUR PROTEINS IS THE BINDING CONSTRAINT OF THE ENTIRE MODULE and it is stated here,
before any number, rather than in a footnote: the complete scorable universe is 25 knockouts x 4 proteins =
100 values. Nothing below can be rescued by a better model.

THE QUESTION. mRNA was measured, in this project's own `mrna_protein_gate.py`, to explain only 15.9% of
protein variance. If that is the whole story, a protein head is not a modelling problem at all -- you read
the transcript and you are done. So: given a perturbation and its full transcriptional response, can
anything predict the perturbation-induced change in surface protein BETTER THAN (a) that protein's own
average behaviour and (b) that protein's own transcript? Those two are the only baselines that matter and
the gate is defined against both of them.

THE TARGET. For knockout p and ADT tag j, Y[p,j] = log2 of the ratio of the mean ADT-total-normalised
abundance of tag j in p's cells to the same quantity in the non-targeting cells. Log2 fold change of a
size-normalised mean, exactly the convention `mrna_vs_protein.py` used, so the numbers sit on the same axis.

THE FEATURE. For the same knockout, R[p,g] = log2 fold change of gene g's UMI-total-normalised mean against
the same non-targeting reference, over the genes detected in the NT cells. Per-gene z-scored and PCA'd
INSIDE each fold on the training knockouts only.

RNA AND PROTEIN ARE READ FROM DISJOINT CELLS. Every knockout's cells are split in half, stratified by HTO
replicate; the RNA half builds the feature, the ADT half builds the target, and no cell contributes to
both. This is not cosmetic. Per-cell technical factors -- capture efficiency, cell size, ambient
contamination, droplet quality -- move UMI counts and ADT counts TOGETHER in the same droplet, so a
same-cell RNA/protein correlation is partly a measurement of the droplet rather than of the biology. With
the halves disjoint that channel is closed by construction. The undivided-cell version is run as an
explicit SENSITIVITY arm so the size of the artefact is reported rather than assumed away.

    THIS IS RAW COUNT DATA AND A ZERO MEANS ZERO UMI OBSERVED, which is a real measurement at that
    sequencing depth. It is NOT the `nlz_*` situation, where 96.5% of the matrix is structural zeros
    meaning NOT RECORDED. The `nlz_*` benches are not touched by this module.

THE ARMS. Every arm emits the same object -- an (n_heldout, 4) matrix of predicted log2 protein changes --
and every arm is scored by the same function on the same held-out knockouts. No arm is scored in a
universe the truth is not drawn from.

    0-param  MARGINAL-protein          each protein's MEAN change over the TRAINING knockouts. The
                                       intercept-only model. Refit inside every fold.
    0-param  NAIVE TRANSLATOR, raw     protein j's change = the log2 mRNA change of j's OWN transcript,
                                       used unscaled. Reported to expose the scale mismatch -- mRNA changes
                                       here span +-2.4 and protein changes +-0.6 -- and NOT the graded
                                       baseline, because grading against an uncalibrated arm would be
                                       beating a strawman on units rather than on content.
    2-param  NAIVE TRANSLATOR, CALIBRATED   THE BASELINE THAT MATTERS. Per protein, a + b * (own mRNA
                                       log2FC), with a and b fit by least squares ON THE TRAINING KNOCKOUTS
                                       ONLY. This is the strongest honest form of "protein follows its own
                                       transcript" and it is what the gate is set against.
    2-param  RANDOM-TRANSCRIPT translator      CONTROL. Identical calibrated form, but reading a random
                                       gene matched to the true one's EXPRESSION DECILE in the NT cells.
                                       Matched on abundance so the comparison is "the right transcript" vs
                                       "any comparable transcript", not "a transcript" vs "a dropout".
                                       Averaged over several draws.
    MODEL      ridge, RNA + identity   RNA-response PCs concatenated with the knocked-out gene's DepMap
                                       gene-effect PCs. Alpha chosen by leave-one-out INSIDE the training
                                       knockouts.
    MODEL      MLP, RNA + identity     small torch MLP on the same inputs, fixed epochs.
    ablation   MODEL, RNA only         identity channel removed
    ablation   MODEL, identity only    RNA channel removed
    CONTROL    SHUFFLED-RNA, structure-preserving   whole RNA response ROWS permuted across knockouts, in
                                       train and test alike. Every knockout keeps a REAL response vector
                                       with its real magnitude and its real gene-gene correlation
                                       structure; only the pairing to the protein truth is destroyed. This
                                       is the configuration-model-style control -- the sharp one.
    CONTROL    SHUFFLED-RNA, gene-wise every gene permuted independently across knockouts. Destroys the
                                       pairing AND the internal structure, producing a chimeric vector no
                                       real knockout has. The loose control, reported next to the sharp one
                                       so the gap between them is visible.
    CONTROL    SWAP-RNA (COUNTERFACTUAL)   the TRAINED model is handed the correct knockout identity and
                                       ANOTHER knockout's RNA. Accuracy alone cannot distinguish "the RNA
                                       channel is useless" from "the RNA channel is IGNORED": both look
                                       like MODEL == identity-only. This can.
    CONTROL    SWAP-IDENTITY (COUNTERFACTUAL)   the mirror image: correct RNA, another knockout's identity.
    CONTROL    WRONG-KNOCKOUT          identity control. The model's per-knockout prediction vectors
                                       permuted across the held-out knockouts. Bounds the metric's own
                                       degeneracy.

    There is no partner-set channel in this module, so the "random partners matched on count and type mix"
    control has no referent here and is not faked. Its role -- a control matched on everything except the
    hypothesised content -- is carried by the EXPRESSION-MATCHED RANDOM TRANSCRIPT and by the
    STRUCTURE-PRESERVING row permutation, which are the two matched controls this design admits.

THE UNIT OF REPLICATION IS A HELD-OUT KNOCKOUT, NOT A GUIDE AND NOT A SEED. The claim under test is
"predicts the protein response of a perturbation it has never seen", so the error bar is computed ACROSS
KNOCKOUTS: leave-one-knockout-out, and an arm's replicates are its per-held-out-knockout scores. The four
guides against one gene are FOUR MEASUREMENTS OF THE SAME PERTURBATION, so a guide-level split leaves
sibling guides in the training set and is leakage. That wrong unit is nevertheless RUN, deliberately, and
reported as a DEMONSTRATION OF THE INFLATION IT PRODUCES -- it is never graded on. In this project the
identical substitution (seeds inside one partition instead of disjoint partitions) once flipped a verdict
from 4/7 to 0/7.

    MDE = 3 * sd / sqrt(n),  n = the number of HELD-OUT KNOCKOUTS (25 at full scale; each contributes one
    leave-one-out fold), sd = the across-knockout sd of an arm's per-knockout RMSE, pooled over arms
    (ddof=1). This is the project-standard unpaired form and it is what the gate uses.
    A second, sharper PAIRED MDE = 3 * sd_diff / sqrt(n) is also reported, sd_diff being the across-knockout
    sd of the per-knockout DIFFERENCE between two arms; the arms share held-out knockouts, so the unpaired
    form is the conservative one and the paired form is the powerful one. Both are printed. The verdict
    quotes the unpaired gate and says what the paired one would have concluded.

THE GATE, DECLARED BEFORE ANY NUMBER IS PRODUCED. Primary metric is per-knockout RMSE in log2 protein-change
units, LOWER IS BETTER, and it is RAW -- nothing is subtracted from it. Secondary is the WITHIN-PROTEIN
Pearson r between predicted and observed across held-out knockouts, averaged over the four proteins, with
BOTH sides expressed as deviations from that fold's TRAINING mean.

    That centring is not cosmetic and it is not netting out a control -- the graded metric is untouched.
    Under leave-one-out an intercept-only predictor is mean_{k!=i} Y[k,j] = (S_j - Y[i,j])/(n-1), an exactly
    DECREASING linear function of the value it is predicting, so its uncentred correlation with the truth is
    exactly -1 for arithmetic reasons that have nothing to do with the data, and every arm carrying an
    intercept inherits a share of that bias. The centred form makes the constant predictor score exactly 0,
    which is its true information content. The uncentred number is reported alongside for every arm so the
    artifact is visible rather than hidden.

    PRIMARY-A   the MODEL beats MARGINAL-protein by more than the MDE (RMSE lower by > MDE).
    PRIMARY-B   the MODEL beats the CALIBRATED NAIVE TRANSLATOR by more than the MDE.
                THE GATE IS BOTH. A protein head that cannot beat "read the transcript" is not a head.
    SIG         the MODEL beats the STRUCTURE-PRESERVING SHUFFLED-RNA control by more than the MDE. This
                establishes significance only. It is not the effect size and is never subtracted.
    TRIVIAL     the cheapest non-learned arm is reported NEXT TO the model, not beneath it. If a
                zero-parameter or two-parameter arm wins, THAT IS THE HEADLINE.
    AGREE       the within-protein Pearson reaches the same verdict as RMSE on the primary comparisons.
                Disagreement means metric artifact and the result is not reportable as biology.

EFFECT SIZES ARE RAW HELD-OUT VALUES. Every RMSE and every r quoted is the raw number an arm achieved on
knockouts it never saw. The shuffled and swapped arms establish significance and attribution; no arm is
ever reported net of a control. Where a control reproduces most of an arm's advantage, that is stated as
the finding.

NO CIRCULARITY, CHECKED AND STATED. (i) The RNA feature and the protein target come from the same
experiment but from DISJOINT CELLS, so no per-droplet technical factor can create the association; the
undivided-cell version is reported as a sensitivity so the size of that artefact is measured. (ii) The
knockout-identity features are DepMap CRISPR gene-effect PCs -- a different assay, different cell lines,
never derived from Papalexi -- so no feature is scored on the dataset it came from. (iii) The gene
detection filter is computed on the NON-TARGETING cells ONLY, which are never a held-out knockout, so the
feature universe carries no information about any test perturbation. (iv) Every target-side quantity -- the
per-protein marginal, the translator's a and b, the per-gene z-scores, the PCA basis, the ridge alpha, the
feature scaler -- is refit on the training knockouts INSIDE every fold.
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
H5 = SP / "papalexi.h5ad"
ADT = SP / "protein" / "GSM4633615_ECCITE_ADT_counts.tsv.gz"
ADT_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4633nnn/GSM4633615/suppl/"
           "GSM4633615_ECCITE_ADT_counts.tsv.gz")
CACHE = SP / "papalexi_pb_cache.npz"
DEPVEC = OUT / "depmap_vecs.npz"

# ADT tag -> the gene whose transcript encodes it. This is antibody annotation, not a fitted quantity.
TAG2GENE = {"CD86": "CD86", "PDL1": "CD274", "PDL2": "PDCD1LG2", "CD366": "HAVCR2"}
# DepMap has renamed the membrane-associated RING-CH family; without this alias MARCH8 loses its identity
# features and would silently be handed a zero vector.
ALIAS = {"MARCH8": "MARCHF8"}

EPS = 1e-6           # pseudocount in normalised-abundance units (~0.013 UMI at the median 13k depth)
NPROT = 4
MIN_CELLS_HALF = int(os.environ.get("VPH_MIN_CELLS", "20"))   # per knockout, per cell-half
MIN_NT_DETECT = float(os.environ.get("VPH_MIN_DETECT", "0.02"))  # gene detection floor, NT cells only
N_PC = int(os.environ.get("VPH_PCS", "8"))            # RNA-response PCs
N_DPC = int(os.environ.get("VPH_DEPMAP_PCS", "12"))   # DepMap identity PCs
MLP_EPOCHS = int(os.environ.get("VPH_MLP_EPOCHS", "300"))
MLP_HIDDEN = int(os.environ.get("VPH_MLP_HIDDEN", "16"))
RAND_TX_DRAWS = int(os.environ.get("VPH_RANDTX_DRAWS", "8"))
ZCLIP = 5.0
ALPHAS = np.logspace(-2, 4, 13)
SMOKE = os.environ.get("VPH_SMOKE", "0") == "1"
MAX_KO = int(os.environ.get("VPH_MAX_KO", "0"))       # 0 = every usable knockout
if SMOKE:
    # A SMOKE RUN IS A PLUMBING CHECK AND NOT A RESULT. Eight knockouts give the MDE seven degrees of
    # freedom and put every arm inside it; three PCs and a shortened MLP make the learned arms
    # deliberately under-fit. Nothing printed under smoke is a measurement of anything.
    # MAX_KO or 8: an explicitly set VPH_MAX_KO wins, so the full-knockout code paths stay reachable
    # under smoke for plumbing purposes; unset, smoke truncates to 8.
    MAX_KO, N_PC, N_DPC, MLP_EPOCHS, RAND_TX_DRAWS = (MAX_KO or 8), 3, 6, 40, 3


# ---------------------------------------------------------------------------------------------------
# pseudobulk cache. Built once at GUIDE x CELL-HALF resolution and summed up to knockout level on use, so
# the graded knockout-level analysis and the ungraded guide-level demonstration read the same numbers.
# ---------------------------------------------------------------------------------------------------
def build_cache(report):
    import gzip
    import h5py
    with h5py.File(H5, "r") as f:
        genes = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["var/gene_symbol"][:]])
        bc = [re.sub(r"-\d+$", "", x.decode() if isinstance(x, bytes) else str(x))
              for x in f["obs/_index"][:]]

        def cat(col):
            g = f["obs"][col]
            cs = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
            return np.array([cs[c] for c in g["codes"][:]])
        guide = cat("guide_id")
        pertc = cat("perturbation")
        hto = cat("hto")
        data = f["X/data"][:]
        indices = f["X/indices"][:]
        indptr = f["X/indptr"][:]
    ncell, ngene = len(bc), len(genes)
    tot = np.bincount(indices, weights=data, minlength=ncell)

    # NT cells are the ones whose perturbation label is 'control'; their guide_id is one of NTg*.
    ko = np.array([re.sub(r"g\d+$", "", p) if p != "control" else "NT" for p in pertc])
    guide = np.where(ko == "NT", "NT_" + guide, guide)

    # DISJOINT CELL HALVES, stratified by (knockout, HTO replicate) so both halves carry the same
    # replicate composition. half 0 -> RNA, half 1 -> ADT.
    rng = np.random.default_rng(0)
    half = np.zeros(ncell, np.int8)
    for k in np.unique(ko):
        for h in np.unique(hto):
            m = np.where((ko == k) & (hto == h))[0]
            if len(m) < 2:
                continue
            half[m[rng.permutation(len(m))[:len(m) // 2]]] = 1

    gnames = sorted(set(guide))
    gidx = {g: i for i, g in enumerate(gnames)}
    grp = np.array([gidx[g] for g in guide]) * 2 + half
    NG = len(gnames) * 2
    w = data / np.maximum(tot[indices], 1.0)
    Psum = np.zeros((NG, ngene))
    Pnz = np.zeros((NG, ngene))
    for g0 in range(0, ngene, 3000):
        g1 = min(g0 + 3000, ngene)
        s, e = int(indptr[g0]), int(indptr[g1])
        gl = np.repeat(np.arange(g1 - g0), np.diff(indptr[g0:g1 + 1]))
        flat = grp[indices[s:e]].astype(np.int64) * (g1 - g0) + gl
        Psum[:, g0:g1] = np.bincount(flat, weights=w[s:e],
                                     minlength=NG * (g1 - g0)).reshape(NG, g1 - g0)
        Pnz[:, g0:g1] = np.bincount(flat, minlength=NG * (g1 - g0)).reshape(NG, g1 - g0)
    Pn = np.bincount(grp, minlength=NG).astype(float)

    if not ADT.exists():
        raise SystemExit(
            f"ADT matrix absent at {ADT}. papalexi.h5ad carries NO protein modality (obsm/layers/uns all "
            f"empty); the protein half of this experiment is a separate GEO file. Fetch it with:\n"
            f"  mkdir -p {ADT.parent} && curl -sS -o {ADT} {ADT_URL}")
    with gzip.open(ADT, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        acells = [c.strip('"').split("_", 1)[1] for c in hdr[1:]]
        tags, A = [], []
        for line in fh:
            fl = line.rstrip("\n").split("\t")
            tags.append(fl[0].strip('"'))
            A.append([float(x) for x in fl[1:]])
    A = np.asarray(A, np.float64)
    # THE JOIN IS ASSERTED, NOT ASSUMED. Stripping only the lane prefix gives 98.61% and looks fine;
    # normalising both conventions gives exact agreement. A 98.6% join would have produced entirely
    # plausible numbers from ~288 mismatched cells.
    agree = int(sum(1 for a, b in zip(acells, bc) if a == b))
    report(f"  ADT<->RNA barcode join: {agree:,}/{len(bc):,} exact positional agreement "
           f"({100*agree/len(bc):.2f}%)")
    assert agree == len(bc), "barcode join is not exact -- refusing to interpret anything downstream"
    atot = A.sum(0)
    An = A / np.maximum(atot, 1.0)[None, :]
    Asum = np.zeros((NG, len(tags)))
    for gi in range(NG):
        m = grp == gi
        if m.any():
            Asum[gi] = An[:, m].sum(1)

    np.savez_compressed(CACHE, Psum=Psum.astype(np.float32), Pnz=Pnz.astype(np.float32), Pn=Pn,
                        Asum=Asum, genes=genes, guides=np.array(gnames), tags=np.array(tags),
                        ko_of_guide=np.array([re.sub(r"g\d+$", "", g) if not g.startswith("NT_") else "NT"
                                              for g in gnames]))
    report(f"  built pseudobulk cache -> {CACHE}")


def main():
    import torch
    torch.set_num_threads(1)          # two full-scale training runs own most of this box's 4 cores
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 110)
    report("V4 BLOCK 8 -- protein head from ECCITE vs the naive translator")
    report("=" * 110)
    report("  GATE, declared before the run: the MODEL is retained only if it beats BOTH the per-protein")
    report("  MARGINAL and the CALIBRATED NAIVE TRANSLATOR by more than the MDE = 3*sd/sqrt(n), where")
    report("  n counts HELD-OUT KNOCKOUTS (leave-one-knockout-out). Guides are not the unit: four guides")
    report("  against one gene are four measurements of the SAME perturbation. Effect sizes are RAW")
    report("  held-out values; shuffles and swaps establish significance only and are never subtracted.")
    if SMOKE:
        report("  *** SMOKE MODE: PLUMBING CHECK ONLY. Nothing below is a measurement. ***")

    # ------------------------------------------------------------------ what is actually in the file
    import h5py
    with h5py.File(H5, "r") as f:
        empty = {k: list(f[k].keys()) for k in ("obsm", "layers", "uns", "obsp", "varm", "varp")}
        shape = tuple(int(x) for x in f["X"].attrs["shape"])
        nnz = int(f["X/data"].shape[0])
        obs_cols = sorted(set(f["obs"].keys()) - {"_index"})
    report(f"\n  papalexi.h5ad INVENTORY -- the premise handed to this module was that it holds paired")
    report(f"  RNA and protein. IT DOES NOT. X = {shape[0]:,} cells x {shape[1]:,} genes, "
           f"{nnz:,} stored values ({100*nnz/(shape[0]*shape[1]):.1f}% dense), raw integer counts.")
    for k, v in empty.items():
        report(f"    {k:8s} present and {'EMPTY' if not v else str(v)}")
    report(f"    obs ({len(obs_cols)} columns): {', '.join(obs_cols)}")
    assert not any(empty.values()), "an unexpected modality appeared -- re-inspect before trusting anything"
    report(f"  => NO protein/ADT modality inside the h5ad. The four ADT tags exist in `var` only as the")
    report(f"     TRANSCRIPTS CD86/CD274/PDCD1LG2/HAVCR2, which is the translator's INPUT, not the target.")
    report(f"     The protein half of this same experiment is GSE153056/GSM4633615 (4 tags x 20,729 cells)")
    report(f"     and is joined here on barcodes with an asserted exact match. A zero in X is zero UMI")
    report(f"     observed -- a real measurement -- NOT the nlz_* case where zeros mean NOT RECORDED.")

    if not CACHE.exists():
        report("\n  pseudobulk cache absent; building (one-off, reads the full count matrix)")
        build_cache(report)
    C = np.load(CACHE, allow_pickle=True)
    Psum, Pnz, Pn, Asum = C["Psum"].astype(np.float64), C["Pnz"].astype(np.float64), C["Pn"], C["Asum"]
    genes = np.array([str(x) for x in C["genes"]])
    guides = np.array([str(x) for x in C["guides"]])
    tags = [str(x) for x in C["tags"]]
    ko_of_guide = np.array([str(x) for x in C["ko_of_guide"]])
    assert tags == list(TAG2GENE), f"unexpected ADT tag order {tags}"

    def agg(mask_guides):
        """Sum guide-level pseudobulk up to whatever grouping the mask defines. Returns per-half means."""
        out = {}
        for h in (0, 1):
            rows = np.where(mask_guides)[0] * 2 + h
            n = Pn[rows].sum()
            out[h] = (Psum[rows].sum(0) / max(n, 1.0), Asum[rows].sum(0) / max(n, 1.0),
                      Pnz[rows].sum(0) / max(n, 1.0), n)
        return out

    is_nt = ko_of_guide == "NT"
    NTd = agg(is_nt)
    kos = sorted(set(ko_of_guide) - {"NT"})

    # gene universe: detection measured on the NON-TARGETING cells ONLY. NT is never a held-out knockout,
    # so nothing about any test perturbation enters the feature universe.
    det_nt = NTd[0][2]
    keepg = det_nt >= MIN_NT_DETECT
    for g in TAG2GENE.values():                    # the four translator genes are kept unconditionally
        keepg[genes == g] = True
    gsel = np.where(keepg)[0]
    report(f"\n  gene universe: {len(gsel):,}/{len(genes):,} genes detected in >= {100*MIN_NT_DETECT:.0f}% "
           f"of the {NTd[0][3]:,.0f} NT RNA-half cells (filter fit on NT cells only -- no test knockout "
           f"informs it)")
    report(f"  translator transcripts, NT detection rate: "
           + ", ".join(f"{t}->{g} {100*det_nt[int(np.where(genes==g)[0][0])]:.1f}%"
                       for t, g in TAG2GENE.items()))

    # ---------------------------------------------------------------- knockout-level truth and features
    rows_ko, Ymat, Rmat, ncell_rna, ncell_adt = [], [], [], [], []
    ntR = NTd[0][0][gsel]
    ntA = NTd[1][1]
    for k in kos:
        d = agg(ko_of_guide == k)
        if d[0][3] < MIN_CELLS_HALF or d[1][3] < MIN_CELLS_HALF:
            continue
        rows_ko.append(k)
        Ymat.append(np.log2((d[1][1] + 1e-9) / (ntA + 1e-9)))          # ADT half -> protein truth
        Rmat.append(np.log2((d[0][0][gsel] + EPS) / (ntR + EPS)))      # RNA half -> feature
        ncell_rna.append(d[0][3])
        ncell_adt.append(d[1][3])
    if MAX_KO:
        rows_ko, Ymat, Rmat = rows_ko[:MAX_KO], Ymat[:MAX_KO], Rmat[:MAX_KO]
        ncell_rna, ncell_adt = ncell_rna[:MAX_KO], ncell_adt[:MAX_KO]
    Y = np.asarray(Ymat)
    R = np.asarray(Rmat)
    N = len(rows_ko)
    report(f"\n  usable knockouts (>= {MIN_CELLS_HALF} cells in BOTH halves): {N} of {len(kos)}; "
           f"proteins {NPROT}. THE ENTIRE SCORABLE UNIVERSE IS {N} x {NPROT} = {N*NPROT} VALUES.")
    report(f"  cells per knockout: RNA half median {np.median(ncell_rna):,.0f} "
           f"(min {min(ncell_rna):,.0f}), ADT half median {np.median(ncell_adt):,.0f} "
           f"(min {min(ncell_adt):,.0f}); NT reference {NTd[0][3]:,.0f} RNA / {NTd[1][3]:,.0f} ADT cells")
    report(f"  protein log2FC spread {Y.std():.3f} (range {Y.min():+.3f}..{Y.max():+.3f}); own-transcript "
           f"mRNA log2FC spread {R[:, [int(np.where(genes[gsel]==g)[0][0]) for g in TAG2GENE.values()]].std():.3f} "
           f"-- the arms are NOT on the same scale, which is why the translator is calibrated")

    # positive control: knock out the gene, its own protein should fall. If this fails nothing else is
    # interpretable -- it would mean the guides, the join or the normalisation is wrong.
    pos = []
    for t, g in TAG2GENE.items():
        if g in rows_ko:
            pos.append((g, t, float(Y[rows_ko.index(g), tags.index(t)])))
    report("\n  POSITIVE CONTROL -- knock out the gene, does its OWN protein fall?")
    for g, t, v in pos:
        report(f"    {g:10s} -> {t:6s} protein log2FC {v:+.4f}  {'OK' if v < 0 else 'WRONG DIRECTION'}")
    pos_ok = sum(1 for _, _, v in pos if v < 0)
    report(f"    {pos_ok}/{len(pos)} self-knockouts lower their own protein")

    # ---------------------------------------------------------------------------- identity features
    dv = np.load(DEPVEC, allow_pickle=True)
    dsyms = {str(s): i for i, s in enumerate(dv["syms"])}
    Zd = dv["Z"].astype(np.float32)
    Ud, sd_, _ = np.linalg.svd(Zd - Zd.mean(0), full_matrices=False)
    Zdp = (Ud[:, :N_DPC] * sd_[:N_DPC]).astype(np.float64)
    Zdp = Zdp / (Zdp.std(0, keepdims=True) + 1e-9)
    ID = np.zeros((N, N_DPC))
    miss = []
    for i, k in enumerate(rows_ko):
        j = dsyms.get(k, dsyms.get(ALIAS.get(k, k)))
        if j is None:
            miss.append(k)
        else:
            ID[i] = Zdp[j]
    report(f"\n  identity features: {N_DPC} DepMap CRISPR gene-effect PCs "
           f"({N-len(miss)}/{N} knockouts covered"
           + (f"; MISSING {miss} -> zero vector" if miss else "") + "). DepMap is a different assay in "
           f"different cell lines and is NOT derived from Papalexi, so no feature is scored on its own "
           f"source dataset.")

    # translator inputs, and the expression-matched random-transcript control
    gs = genes[gsel]
    gpos = {g: i for i, g in enumerate(gs)}
    tx_col = np.array([gpos[TAG2GENE[t]] for t in tags])
    expr_nt = ntR
    dec = np.clip((np.argsort(np.argsort(expr_nt)) * 10) // max(len(expr_nt), 1), 0, 9)
    rng_tx = np.random.default_rng(11)
    rand_cols = []
    for _ in range(RAND_TX_DRAWS):
        cols = []
        for c in tx_col:
            pool = np.where((dec == dec[c]) & (np.arange(len(gs)) != c))[0]
            cols.append(int(rng_tx.choice(pool)) if len(pool) else int(c))
        rand_cols.append(np.array(cols))
    report(f"  random-transcript control: {RAND_TX_DRAWS} draws, each gene matched to the true "
           f"transcript's NT EXPRESSION DECILE (so the contrast is 'the right transcript' vs 'a comparable "
           f"transcript', not 'a transcript' vs 'a dropout')")

    # SHUFFLED-RNA controls, fixed once so train and test both see the decoupled channel.
    rs = np.random.default_rng(3)
    R_shuf_row = R[rs.permutation(N)]                      # structure-preserving: whole rows moved
    R_shuf_gene = R.copy()                                 # gene-wise: each column independently permuted
    for j in range(R_shuf_gene.shape[1]):
        R_shuf_gene[:, j] = R_shuf_gene[rs.permutation(N), j]
    swap_src = np.array([(i + 1) % N for i in range(N)])    # deterministic non-identity derangement

    # -------------------------------------------------------------------------------- machinery
    def prep(tr, Rsrc):
        """Per-gene z-score then PCA, both fit on the TRAINING knockouts only."""
        mu, sg = Rsrc[tr].mean(0), Rsrc[tr].std(0) + 1e-9
        Zr = np.clip((Rsrc - mu) / sg, -ZCLIP, ZCLIP)
        Ztr = Zr[tr] - Zr[tr].mean(0)
        _u, _s, vt = np.linalg.svd(Ztr, full_matrices=False)
        V = vt[:min(N_PC, vt.shape[0])].T
        P = (Zr - Zr[tr].mean(0)) @ V
        return P / (P[tr].std(0) + 1e-9)

    def ridge_fit(Xtr, Ytr, alpha):
        xm, ym = Xtr.mean(0), Ytr.mean(0)
        Xc, Yc = Xtr - xm, Ytr - ym
        G = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
        W = np.linalg.solve(G, Xc.T @ Yc)
        return W, xm, ym

    def ridge_alpha(Xtr, Ytr):
        """Leave-one-out INSIDE the training knockouts. The held-out knockout never sees this."""
        best, ba = np.inf, ALPHAS[0]
        for a in ALPHAS:
            err = []
            for i in range(len(Xtr)):
                m = np.ones(len(Xtr), bool)
                m[i] = False
                W, xm, ym = ridge_fit(Xtr[m], Ytr[m], a)
                err.append(((Xtr[i] - xm) @ W + ym - Ytr[i]) ** 2)
            e = float(np.mean(err))
            if e < best:
                best, ba = e, a
        return ba

    def mlp_predict(Xtr, Ytr, Xte, seed):
        import torch.nn as nn
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], MLP_HIDDEN), nn.ReLU(),
                            nn.Linear(MLP_HIDDEN, NPROT))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
        xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor(Ytr, dtype=torch.float32)
        for _ in range(MLP_EPOCHS):
            opt.zero_grad()
            ((net(xt) - yt) ** 2).mean().backward()
            opt.step()
        with torch.no_grad():
            return net(torch.tensor(Xte, dtype=torch.float32)).numpy().astype(np.float64)

    def translator(tr, te, cols, Rsrc, calibrate=True):
        pred = np.zeros((len(te), NPROT))
        for j in range(NPROT):
            x, y = Rsrc[tr, cols[j]], Y[tr, j]
            if calibrate:
                b = float(np.cov(x, y, ddof=1)[0, 1] / (np.var(x, ddof=1) + 1e-12)) if len(tr) > 2 else 0.0
                a = float(y.mean() - b * x.mean())
            else:
                a, b = 0.0, 1.0
            pred[:, j] = a + b * Rsrc[te, cols[j]]
        return pred

    # ------------------------------------------------------------------------------------ the folds
    def run_cv(folds, Rsrc, arms_wanted, tag=""):
        """One leave-one-out pass. EVERY arm emits (n_te, 4) predicted log2 protein changes and is scored
        by the same function on the same held-out knockouts -- no arm is scored in a universe the truth is
        not drawn from, which is the defect that has shipped three times in this project."""
        acc = {}
        for te in folds:
            tr = np.setdiff1d(np.arange(N), te)
            Ytr = Y[tr]
            Pfull = prep(tr, Rsrc)
            Xf = np.hstack([Pfull, ID])
            preds = {}
            preds["MARGINAL-protein (0 param)"] = np.tile(Ytr.mean(0), (len(te), 1))
            preds["NAIVE TRANSLATOR raw, uncalibrated"] = translator(tr, te, tx_col, Rsrc, False)
            preds["NAIVE TRANSLATOR calibrated (2 param)"] = translator(tr, te, tx_col, Rsrc, True)
            preds["RANDOM-TRANSCRIPT calibrated (CONTROL)"] = np.mean(
                [translator(tr, te, rc, Rsrc, True) for rc in rand_cols], axis=0)
            if "model" in arms_wanted:
                a = ridge_alpha(Xf[tr], Ytr)
                W, xm, ym = ridge_fit(Xf[tr], Ytr, a)
                preds["MODEL ridge (RNA + identity)"] = (Xf[te] - xm) @ W + ym
                for nm, Xv in (("MODEL ridge, RNA only", Pfull), ("MODEL ridge, identity only", ID)):
                    a2 = ridge_alpha(Xv[tr], Ytr)
                    W2, xm2, ym2 = ridge_fit(Xv[tr], Ytr, a2)
                    preds[nm] = (Xv[te] - xm2) @ W2 + ym2
                # COUNTERFACTUAL SWAPS on the trained full model. Accuracy alone cannot separate "the
                # channel is useless" from "the channel is IGNORED"; only these can.
                Xs = Xf[te].copy()
                Xs[:, :Pfull.shape[1]] = Pfull[swap_src[te]]
                preds["SWAP-RNA (COUNTERFACTUAL)"] = (Xs - xm) @ W + ym
                Xs2 = Xf[te].copy()
                Xs2[:, Pfull.shape[1]:] = ID[swap_src[te]]
                preds["SWAP-IDENTITY (COUNTERFACTUAL)"] = (Xs2 - xm) @ W + ym
            if "mlp" in arms_wanted:
                preds["MODEL MLP (RNA + identity)"] = mlp_predict(Xf[tr], Ytr, Xf[te], 7)
            # the fold's TRAINING mean, kept so the secondary correlation can be taken on deviations from
            # it. It is identical to the MARGINAL arm's prediction and does not depend on the RNA source.
            preds["__trainmean__"] = np.tile(Ytr.mean(0), (len(te), 1))
            for nm, p in preds.items():
                p = np.asarray(p, np.float64)
                assert p.shape == (len(te), NPROT) and np.isfinite(p).all(), \
                    f"arm '{nm}' did not emit a finite ({len(te)},{NPROT}) prediction"
                acc.setdefault(nm, {"pred": [], "te": []})
                acc[nm]["pred"].append(p)
                acc[nm]["te"].append(te)
        for nm in acc:
            acc[nm]["pred"] = np.vstack(acc[nm]["pred"])
            acc[nm]["te"] = np.concatenate(acc[nm]["te"])
        return acc

    MODEL = "MODEL ridge (RNA + identity)"
    MARG_NAME = MARG = "MARGINAL-protein (0 param)"
    TRANS = "NAIVE TRANSLATOR calibrated (2 param)"
    SHUF = "SHUFFLED-RNA structure-preserving (CONTROL)"

    folds = [np.array([i]) for i in range(N)]     # leave-one-KNOCKOUT-out: the unit matches the claim
    acc = run_cv(folds, R, {"model", "mlp"})
    acc_sr = run_cv(folds, R_shuf_row, {"model"})
    acc_sg = run_cv(folds, R_shuf_gene, {"model"})
    acc["SHUFFLED-RNA structure-preserving (CONTROL)"] = acc_sr["MODEL ridge (RNA + identity)"]
    acc["SHUFFLED-RNA gene-wise (CONTROL)"] = acc_sg["MODEL ridge (RNA + identity)"]
    base = acc["MODEL ridge (RNA + identity)"]
    ordr = np.argsort(base["te"])
    acc["WRONG-KNOCKOUT (identity CONTROL)"] = {
        "pred": base["pred"][ordr][swap_src], "te": base["te"][ordr]}

    def score(a, tm):
        """Per-held-out-knockout RMSE in log2 protein-change units, plus two Pearsons.

        rmse_per_ko  THE GRADED REPLICATE, fully raw. n counts KNOCKOUTS.
        r_within     mean over the four proteins of the Pearson between PREDICTED and OBSERVED deviations
                     from that fold's TRAINING mean, across held-out knockouts. An intercept-only predictor
                     is identically zero here, which is its true within-protein information content.
        r_within_raw the same correlation WITHOUT that centring. Under leave-one-out the intercept-only arm
                     scores exactly -1 on it -- mean_{k!=i} Y[k,j] = (S_j - Y[i,j])/(n-1) is a decreasing
                     linear function of the very value it predicts -- and every arm with an intercept
                     inherits part of that bias. Reported so the artifact is visible, never graded on.
        r_pooled     Pearson over all n*4 raw pairs. Carries the offsets BETWEEN proteins as well as the
                     ordering within them, which is exactly why it is not the graded number either.
        """
        o = np.argsort(a["te"])
        P, T, M = a["pred"][o], Y[a["te"][o]], tm[o]
        rmse = np.sqrt(((P - T) ** 2).mean(1))
        Pd, Td = P - M, T - M
        rw, rr = [], []
        for j in range(NPROT):
            rw.append(0.0 if Pd[:, j].std() < 1e-12 or Td[:, j].std() < 1e-12
                      else float(np.corrcoef(Pd[:, j], Td[:, j])[0, 1]))
            rr.append(0.0 if P[:, j].std() < 1e-12 or T[:, j].std() < 1e-12
                      else float(np.corrcoef(P[:, j], T[:, j])[0, 1]))
        rp = float(np.corrcoef(P.ravel(), T.ravel())[0, 1]) if P.std() > 1e-12 else 0.0
        return rmse, float(np.mean(rw)), rp, rw, float(np.mean(rr)), Pd, Td

    def trainmean(a):
        o = np.argsort(a["__trainmean__"]["te"])
        return a["__trainmean__"]["pred"][o]

    TM = trainmean(acc)
    res = {}
    for nm, a in acc.items():
        if nm == "__trainmean__":
            continue
        rmse, rw, rp, rwv, rraw, Pd, Td = score(a, TM)
        res[nm] = {"rmse": rmse, "rmse_mean": float(rmse.mean()), "rmse_sd": float(rmse.std(ddof=1)),
                   "r_within": rw, "r_pooled": rp, "r_per_protein": rwv, "r_within_raw": rraw,
                   "_Pd": Pd, "_Td": Td}

    report(f"\n  {'arm':44s} {'RMSE':>8s} {'sd':>7s} {'r_within':>9s} {'(raw)':>8s} {'r_pooled':>9s}"
           f"   per-protein r (centred)")
    for nm, v in res.items():
        report(f"  {nm:44s} {v['rmse_mean']:8.4f} {v['rmse_sd']:7.4f} {v['r_within']:+9.4f} "
               f"{v['r_within_raw']:+8.4f} {v['r_pooled']:+9.4f}   " + " ".join(
                   f"{t}{x:+.2f}" for t, x in zip([s[:4] + ":" for s in tags], v["r_per_protein"])))
    report(f"  r_within is taken on deviations from the fold's TRAINING mean, so the MARGINAL arm is "
           f"exactly 0 -- its true within-protein content.")
    report(f"  the (raw) column shows the SAME correlation uncentred: the marginal arm sits at "
           f"{res[MARG_NAME]['r_within_raw']:+.4f} there, which is a leave-one-out arithmetic artifact "
           f"(the mean of the others decreases as the held-out value rises), not a property of the data. "
           f"Nothing is graded on it.")

    # ---------------------------------------------------------------------------------------- MDE
    sd_pool = float(np.mean([v["rmse_sd"] for v in res.values()]))
    MDE = 3 * sd_pool / np.sqrt(N)

    def gap(a, b):
        """RMSE is lower-is-better, so a POSITIVE gap means `a` is better than `b`."""
        return float(res[b]["rmse_mean"] - res[a]["rmse_mean"])

    def paired(a, b):
        d = res[b]["rmse"] - res[a]["rmse"]
        s = float(d.std(ddof=1))
        return float(d.mean()), s, 3 * s / np.sqrt(N)

    report(f"\n  MDE = 3*sd/sqrt(n), n = {N} HELD-OUT KNOCKOUTS (leave-one-knockout-out; the claim is")
    report(f"  generalisation to an unseen perturbation, so the error bar is computed ACROSS knockouts).")
    report(f"  pooled across-knockout sd of per-knockout RMSE {sd_pool:.4f} -> MDE {MDE:.4f}")
    report(f"  the point estimate is the same either way (a mean of differences IS a difference of means);")
    report(f"  only the error bar changes, because the arms share held-out knockouts. Unpaired is the")
    report(f"  conservative bar and is what the gate uses; paired is the powerful one and is reported next")
    report(f"  to it so no call can turn silently on that choice.")
    for a, b, nm in ((MODEL, MARG, "model vs marginal"), (MODEL, TRANS, "model vs translator"),
                     (MODEL, SHUF, "model vs shuffled-RNA"), (TRANS, MARG, "translator vs marginal")):
        d, s, pm = paired(a, b)
        report(f"    {nm:24s} gap {gap(a,b):+.4f}  vs UNPAIRED MDE {MDE:.4f} "
               f"({'DETECTED' if gap(a,b) > MDE else 'not detected'})  |  vs PAIRED MDE {pm:.4f} "
               f"({'DETECTED' if d > pm else 'not detected'})")

    gates = {
        "PRIMARY-A  MODEL beats MARGINAL-protein by > MDE": bool(gap(MODEL, MARG) > MDE),
        "PRIMARY-B  MODEL beats the CALIBRATED NAIVE TRANSLATOR by > MDE": bool(gap(MODEL, TRANS) > MDE),
        "SIG        MODEL beats structure-preserving SHUFFLED-RNA by > MDE": bool(gap(MODEL, SHUF) > MDE),
        "TRIVIAL    MODEL beats the best non-learned arm by > MDE": None,
        "AGREE      within-protein r reaches the same verdict as RMSE on PRIMARY-B": None,
    }
    cheap = [MARG, "NAIVE TRANSLATOR raw, uncalibrated", TRANS, "RANDOM-TRANSCRIPT calibrated (CONTROL)"]
    best_cheap = min(cheap, key=lambda x: res[x]["rmse_mean"])
    gates["TRIVIAL    MODEL beats the best non-learned arm by > MDE"] = bool(gap(MODEL, best_cheap) > MDE)
    # THE SECONDARY METRIC NEEDS ITS OWN ERROR BAR ON THE SAME UNIT, otherwise "agreement" is a threshold
    # test against a bare sign test and disagrees at chance. r_within is an aggregate over folds and has no
    # per-fold replicate, so its spread is obtained by a DELETE-ONE-KNOCKOUT JACKKNIFE -- the unit of
    # replication stays the held-out knockout, and the MDE keeps the same 3*sd/sqrt(n) form.
    def rw_masked(nm, keep):
        Pd, Td = res[nm]["_Pd"][keep], res[nm]["_Td"][keep]
        out = []
        for j in range(NPROT):
            out.append(0.0 if Pd[:, j].std() < 1e-12 or Td[:, j].std() < 1e-12
                       else float(np.corrcoef(Pd[:, j], Td[:, j])[0, 1]))
        return float(np.mean(out))

    def r_delta_mde(a, b):
        full = rw_masked(a, np.ones(N, bool)) - rw_masked(b, np.ones(N, bool))
        ps = []
        for i in range(N):
            m = np.ones(N, bool)
            m[i] = False
            ps.append(rw_masked(a, m) - rw_masked(b, m))
        s = float(np.std(ps, ddof=1))
        return full, s, 3 * s / np.sqrt(N)

    d_r, sd_r, MDE_R = r_delta_mde(MODEL, TRANS)
    r_agrees = bool(d_r > MDE_R) == bool(gap(MODEL, TRANS) > MDE)
    gates["AGREE      within-protein r reaches the same verdict as RMSE on PRIMARY-B"] = bool(r_agrees)
    report(f"\n  SECONDARY METRIC, same unit and same MDE form: model-minus-translator within-protein r = "
           f"{d_r:+.4f}")
    report(f"  against a delete-one-knockout jackknife MDE of {MDE_R:.4f} (3*sd/sqrt({N}), jackknife sd "
           f"{sd_r:.4f}) -> {'DETECTED' if d_r > MDE_R else 'not detected'}, which "
           f"{'AGREES with' if r_agrees else 'DISAGREES with'} the RMSE call.")
    report("\n  PREDECLARED GATES")
    for k, v in gates.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k}")
    report(f"    best non-learned arm = '{best_cheap}' at RMSE {res[best_cheap]['rmse_mean']:.4f}; "
           f"MODEL {res[MODEL]['rmse_mean']:.4f}; gap {gap(MODEL, best_cheap):+.4f}")

    # rule: a control that reproduces most of the effect IS the finding.
    g_over_marg = gap(MODEL, MARG)
    repro = {
        "the CALIBRATED TRANSLATOR reproduces this fraction of the model's gain over the marginal":
            gap(TRANS, MARG) / g_over_marg if abs(g_over_marg) > 1e-9 else float("nan"),
        "the EXPRESSION-MATCHED RANDOM TRANSCRIPT reproduces this fraction of it":
            gap("RANDOM-TRANSCRIPT calibrated (CONTROL)", MARG) / g_over_marg
            if abs(g_over_marg) > 1e-9 else float("nan"),
        "the STRUCTURE-PRESERVING SHUFFLED-RNA model reproduces this fraction of it":
            gap(SHUF, MARG) / g_over_marg if abs(g_over_marg) > 1e-9 else float("nan"),
        "the IDENTITY-ONLY model (no RNA at all) reproduces this fraction of it":
            gap("MODEL ridge, identity only", MARG) / g_over_marg
            if abs(g_over_marg) > 1e-9 else float("nan"),
    }
    report("\n  CONTROLS THAT REPRODUCE THE EFFECT (reported as the finding, not as a footnote)")
    for k, v in repro.items():
        report(f"    {100*v:7.1f}%  {k}")
    repro_meaningful = bool(g_over_marg > MDE)
    if not repro_meaningful:
        report(f"    ^ the model's gain over the marginal is {g_over_marg:+.4f}, itself below the "
               f"{MDE:.4f} MDE, so these are ratios of two undetected quantities and mean nothing. They "
               f"are printed rather than suppressed so the denominator stays visible.")
    report(f"\n  COUNTERFACTUALS -- 'ignored' vs 'useless', which accuracy alone cannot separate:")
    report(f"    handing the trained model another knockout's RNA changes RMSE by "
           f"{-gap('SWAP-RNA (COUNTERFACTUAL)', MODEL):+.4f}; another knockout's IDENTITY by "
           f"{-gap('SWAP-IDENTITY (COUNTERFACTUAL)', MODEL):+.4f}. A swap gap at or below the "
           f"{MDE:.4f} MDE means that channel is IGNORED, not merely uninformative.")

    # ------------------------------------------------------- SENSITIVITY 1: undivided cells (confound)
    Rj, Yj = [], []
    for k in rows_ko:
        d = agg(ko_of_guide == k)
        nrm = d[0][3] + d[1][3]
        mr = (d[0][0] * d[0][3] + d[1][0] * d[1][3]) / max(nrm, 1)
        ma = (d[0][1] * d[0][3] + d[1][1] * d[1][3]) / max(nrm, 1)
        Rj.append(np.log2((mr[gsel] + EPS) / (((NTd[0][0] * NTd[0][3] + NTd[1][0] * NTd[1][3])
                                               / (NTd[0][3] + NTd[1][3]))[gsel] + EPS)))
        Yj.append(np.log2((ma + 1e-9) / (((NTd[0][1] * NTd[0][3] + NTd[1][1] * NTd[1][3])
                                          / (NTd[0][3] + NTd[1][3])) + 1e-9)))
    Ysave = Y.copy()
    Y = np.asarray(Yj)
    acc_j = run_cv(folds, np.asarray(Rj), {"model"})
    TMj = trainmean(acc_j)
    sens = {nm: float(score(a, TMj)[0].mean()) for nm, a in acc_j.items() if nm != "__trainmean__"}
    sens_r = {nm: float(score(a, TMj)[1]) for nm, a in acc_j.items() if nm != "__trainmean__"}
    Y = Ysave
    report("\n  SENSITIVITY 1 -- SAME cells for RNA and protein (the confound the split-half design closes)")
    report(f"    {'arm':44s} {'RMSE split':>11s} {'RMSE same':>11s} {'r_w split':>10s} {'r_w same':>10s}")
    for nm in (MARG, TRANS, MODEL):
        report(f"    {nm:44s} {res[nm]['rmse_mean']:11.4f} {sens[nm]:11.4f} "
               f"{res[nm]['r_within']:+10.4f} {sens_r[nm]:+10.4f}")
    report("    a same-cell number that is much better than the split-half one is per-droplet technical "
           "coupling, not biology. The split-half figures above are the graded ones.")

    # ------------------------------- SENSITIVITY 2: the WRONG unit of replication, run to show inflation
    gu = np.where(~is_nt)[0]
    gk, gY, gR = [], [], []
    for i in gu:
        d = agg(np.arange(len(guides)) == i)
        if d[0][3] < MIN_CELLS_HALF or d[1][3] < MIN_CELLS_HALF or ko_of_guide[i] not in rows_ko:
            continue
        gk.append(ko_of_guide[i])
        gY.append(np.log2((d[1][1] + 1e-9) / (ntA + 1e-9)))
        gR.append(np.log2((d[0][0][gsel] + EPS) / (ntR + EPS)))
    guide_demo = {"n_guides": len(gk)}
    if len(gk) >= 6:
        Ng, Yg, Rg = len(gk), np.asarray(gY), np.asarray(gR)
        Ysave, Nsave, IDsave, ssave = Y, N, ID, swap_src
        idg = np.zeros((Ng, N_DPC))
        for i, k in enumerate(gk):
            j = dsyms.get(k, dsyms.get(ALIAS.get(k, k)))
            if j is not None:
                idg[i] = Zdp[j]
        Y, N, ID, swap_src = Yg, Ng, idg, np.array([(i + 1) % Ng for i in range(Ng)])
        ag = run_cv([np.array([i]) for i in range(Ng)], Rg, {"model"})
        TMg = trainmean(ag)
        gsc = {nm: score(a, TMg) for nm, a in ag.items() if nm != "__trainmean__"}
        sdg = float(np.mean([s[0].std(ddof=1) for s in gsc.values()]))
        mdeg = 3 * sdg / np.sqrt(Ng)
        gg = gsc[MARG][0].mean() - gsc[MODEL][0].mean()
        guide_demo = {"n_guides": int(Ng), "mde": float(mdeg), "model_minus_marginal": float(gg),
                      "would_pass": bool(gg > mdeg),
                      "model_rmse": float(gsc[MODEL][0].mean()),
                      "marginal_rmse": float(gsc[MARG][0].mean()),
                      "translator_rmse": float(gsc[TRANS][0].mean())}
        Y, N, ID, swap_src = Ysave, Nsave, IDsave, ssave
        report(f"\n  SENSITIVITY 2 -- THE WRONG UNIT, RUN ON PURPOSE AND NOT GRADED ON. Splitting by GUIDE")
        report(f"    ({Ng} guides) leaves sibling guides against the SAME gene in the training set, so the")
        report(f"    'unseen perturbation' is not unseen. n rises {Ng}/{N} = {Ng/N:.1f}x and the MDE falls")
        report(f"    to {mdeg:.4f} from {MDE:.4f}. Under that wrong unit the model-vs-marginal gap is "
               f"{gg:+.4f}")
        report(f"    -> it would be reported as {'A PASS' if gg > mdeg else 'a fail'}; the graded knockout-"
               f"level answer is {'a pass' if gap(MODEL,MARG) > MDE else 'A FAIL'}. This is the exact "
               f"substitution")
        report(f"    that once flipped a verdict in this project from 4/7 to 0/7.")

    # ------------------------------------------------------------------------------------- verdict
    pa = gates["PRIMARY-A  MODEL beats MARGINAL-protein by > MDE"]
    pb = gates["PRIMARY-B  MODEL beats the CALIBRATED NAIVE TRANSLATOR by > MDE"]
    passed = bool(pa and pb)
    d_tm, _s, pm_tm = paired(MODEL, TRANS)
    verdict = (
        (f"SMOKE RUN ({N} knockouts, {N_PC} RNA PCs, {MLP_EPOCHS} MLP epochs) -- THIS IS A PLUMBING CHECK "
         f"AND NOT A RESULT. The knockout set is truncated to {N}, so the MDE has {N-1} degrees of freedom, "
         f"every arm sits inside it, and the arms cannot be separated at this size. No sentence below is a "
         f"measurement and none of it should be quoted as a finding. Rerun with VPH_SMOKE unset. "
         if SMOKE else "") +
        f"FIRST, THE PREMISE WAS WRONG AND THAT IS PART OF THE RESULT: papalexi.h5ad contains NO protein "
        f"modality -- obsm, layers, uns, obsp, varm and varp are all present and all empty, and the file is "
        f"a {shape[0]:,} x {shape[1]:,} RNA count matrix. The protein half of the same ECCITE experiment is "
        f"a separate GEO file (GSE153056/GSM4633615, 4 ADT tags x {shape[0]:,} cells), joined here at an "
        f"asserted-exact {shape[0]:,}/{shape[0]:,} barcode match; it is the other half of this experiment, "
        f"not a substitute dataset. FOUR PROTEINS AND {N} KNOCKOUTS IS THE WHOLE UNIVERSE -- {N*NPROT} "
        f"scorable values -- and that ceiling, not the modelling, is what bounds everything here. "
        f"{'PROTEIN HEAD PASSES' if passed else 'PROTEIN HEAD FAILS'}: on held-out knockouts, with RNA read "
        f"from cells DISJOINT from the cells the protein was read from, the ridge model on RNA-response PCs "
        f"plus DepMap identity PCs reaches RMSE {res[MODEL]['rmse_mean']:.4f} in log2 protein-change units, "
        f"against {res[MARG]['rmse_mean']:.4f} for the per-protein MARGINAL and "
        f"{res[TRANS]['rmse_mean']:.4f} for the CALIBRATED NAIVE TRANSLATOR (protein follows its own "
        f"transcript, slope and intercept fit on training knockouts only). Those are gaps of "
        f"{gap(MODEL, MARG):+.4f} and {gap(MODEL, TRANS):+.4f} against a minimum detectable increment of "
        f"{MDE:.4f} (3*sd/sqrt(n), n = {N} HELD-OUT KNOCKOUTS -- not guides, because four guides against "
        f"one gene are four measurements of the same perturbation, and not seeds, because seeds measure "
        f"optimiser noise). The gate was declared as BOTH, and it is "
        f"{'met' if passed else ('met on neither' if not (pa or pb) else 'met on only one, which is not the gate')}. "
        f"The sharper PAIRED test on the same held-out knockouts gives model-minus-translator "
        f"{d_tm:+.4f} against a paired MDE of {pm_tm:.4f}, i.e. "
        f"{'still detected' if d_tm > pm_tm else 'still not detected'}, so the call does not turn on the "
        f"choice of unpaired versus paired error bar. THE CHEAPEST ARM IS REPORTED NEXT TO THE MODEL, NOT "
        f"BENEATH IT: the best non-learned arm is '{best_cheap}' at {res[best_cheap]['rmse_mean']:.4f}, "
        f"{'which the model beats' if gap(MODEL, best_cheap) > MDE else 'WHICH MATCHES OR BEATS THE TRAINED MODEL -- that is the headline, not a footnote'}. "
        f"Attribution: a STRUCTURE-PRESERVING permutation of the RNA rows (every knockout keeps a real "
        f"response vector with its real magnitude and gene-gene structure, only the pairing is destroyed) "
        f"gives {res[SHUF]['rmse_mean']:.4f}, a gene-wise permutation "
        f"{res['SHUFFLED-RNA gene-wise (CONTROL)']['rmse_mean']:.4f}, an expression-decile-matched RANDOM "
        f"TRANSCRIPT read through the same calibration "
        f"{res['RANDOM-TRANSCRIPT calibrated (CONTROL)']['rmse_mean']:.4f}, and the identity-only model "
        f"with no RNA at all {res['MODEL ridge, identity only']['rmse_mean']:.4f}. The counterfactual "
        f"swaps separate 'ignored' from 'useless': handing the trained model another knockout's RNA moves "
        f"RMSE by {-gap('SWAP-RNA (COUNTERFACTUAL)', MODEL):+.4f} and another knockout's identity by "
        f"{-gap('SWAP-IDENTITY (COUNTERFACTUAL)', MODEL):+.4f}, and a swap gap inside the MDE means the "
        f"channel is being IGNORED rather than being uninformative -- accuracy alone cannot tell those "
        f"apart. The secondary metric -- the within-protein Pearson taken on deviations from each fold's "
        f"TRAINING mean, on which an intercept-only predictor scores exactly 0 because that is its true "
        f"within-protein content -- gives model {res[MODEL]['r_within']:+.4f} and translator "
        f"{res[TRANS]['r_within']:+.4f}, a difference of {d_r:+.4f} against a delete-one-knockout jackknife "
        f"MDE of {MDE_R:.4f} on the SAME unit, and it {'AGREES' if r_agrees else 'DISAGREES'} with RMSE on "
        f"the primary comparison"
        f"{'.' if r_agrees else ', and when the two metrics disagree the result is a metric artifact and is not reportable as biology.'} "
        f" The UNCENTRED form of that same correlation is reported in the JSON and must not be read as a "
        f"result: under leave-one-out the intercept-only arm scores exactly "
        f"{res[MARG]['r_within_raw']:+.2f} on it, because the mean of the other knockouts is by "
        f"construction a decreasing linear function of the held-out value, and every arm carrying an "
        f"intercept inherits some of that bias. "
        f"The positive control holds at {pos_ok}/{len(pos)} self-knockouts lowering their own protein. All "
        f"effect sizes quoted are RAW held-out values; the shuffles, swaps and identity control establish "
        f"significance and attribution and nothing is reported net of any of them.")
    report("\n" + "=" * 110)
    report(f"  VERDICT: {verdict}")

    R_ = {
        "model": "v4-protein-head-v1",
        "question": "Can a protein head predict perturbation-induced surface-protein change better than "
                    "(a) that protein's own marginal and (b) that protein's own transcript? mRNA was "
                    "measured in this project to explain 15.9% of protein variance, so the naive "
                    "translator is the baseline that decides whether a head is worth building at all.",
        "dataset_inventory": {
            "file": str(H5),
            "shape_cells_x_genes": [int(shape[0]), int(shape[1])],
            "stored_values": int(nnz),
            "density": float(nnz / (shape[0] * shape[1])),
            "protein_modality_in_h5ad": False,
            "empty_slots": {k: v for k, v in empty.items()},
            "obs_columns": obs_cols,
            "n_perturbation_categories": 99,
            "n_guide_categories": 107,
            "n_knockout_genes": len(kos),
            "protein_source": "GSE153056 / GSM4633615 ECCITE_ADT_counts.tsv.gz -- the protein half of the "
                              "SAME experiment, same barcodes, NOT a substitute dataset",
            "protein_source_url": ADT_URL,
            "n_proteins": NPROT, "protein_tags": tags,
            "barcode_join": "asserted exact, 20,729/20,729 positional agreement",
            "zeros_mean": "zero UMI observed at that depth -- a real measurement. NOT the nlz_* case where "
                          "96.5% of entries are structural zeros meaning NOT RECORDED; nlz_* is untouched."},
        "design": {
            "target": "Y[p,j] = log2 ratio of ADT-total-normalised mean of tag j in knockout p's cells to "
                      "the same in non-targeting cells",
            "feature": "R[p,g] = log2 ratio of UMI-total-normalised mean of gene g, same reference",
            "disjoint_cells": "RNA is read from one half of each knockout's cells and protein from the "
                              "other half, stratified by HTO replicate, so no per-droplet technical factor "
                              "can create the association. Same-cell version reported as SENSITIVITY 1.",
            "gene_universe": int(len(gsel)),
            "gene_filter": f"detected in >= {100*MIN_NT_DETECT:.0f}% of NT cells; NT is never a held-out "
                           f"knockout so the filter leaks nothing about any test perturbation",
            "min_cells_per_half": MIN_CELLS_HALF,
            "n_knockouts": int(N), "scorable_universe": int(N * NPROT),
            "cells_per_knockout_rna_median": float(np.median(ncell_rna)),
            "cells_per_knockout_adt_median": float(np.median(ncell_adt)),
            "identity_features": f"{N_DPC} DepMap CRISPR gene-effect PCs",
            "rna_pcs": N_PC},
        "positive_control": {"self_knockouts_lowering_own_protein": int(pos_ok), "n": len(pos),
                             "detail": [{"ko": g, "tag": t, "protein_log2fc": v} for g, t, v in pos]},
        "arms": {nm: {"rmse_mean": v["rmse_mean"], "rmse_sd": v["rmse_sd"],
                      "rmse_per_heldout_knockout": [float(x) for x in v["rmse"]],
                      "r_within_protein_centred_on_fold_train_mean": v["r_within"],
                      "r_within_protein_UNCENTRED_LOO_ARTIFACT_DO_NOT_GRADE": v["r_within_raw"],
                      "r_pooled": v["r_pooled"],
                      "r_per_protein": {t: float(x) for t, x in zip(tags, v["r_per_protein"])}}
                 for nm, v in res.items()},
        "secondary_metric_note":
            "r_within is the Pearson between PREDICTED and OBSERVED deviations from the fold's TRAINING "
            "mean. The uncentred version is also stored and must not be graded on: under leave-one-out an "
            "intercept-only predictor equals (sum_of_all - held_out)/(n-1), a decreasing linear function "
            "of the value it predicts, so it scores exactly -1 for arithmetic reasons, and every arm with "
            "an intercept inherits part of that bias. The graded primary metric (RMSE) is unaffected and "
            "is fully raw.",
        "held_out_knockout_order": rows_ko,
        "unit_of_replication": f"a HELD-OUT KNOCKOUT (leave-one-knockout-out, n = {N}). The per-fold lists "
                               f"above hold one value per held-out knockout, NOT per guide and NOT per "
                               f"model-init seed. Four guides against one gene are four measurements of "
                               f"the same perturbation.",
        "mde": {"value": float(MDE), "convention": "3*sd/sqrt(n)", "n": int(N),
                "n_counts": "held-out knockouts, one per leave-one-out fold",
                "pooled_across_knockout_sd": float(sd_pool),
                "primary_metric": "per-held-out-knockout RMSE in log2 protein-change units, lower better",
                "secondary_metric_mde": {
                    "model_minus_translator_r_within": float(d_r), "jackknife_sd": float(sd_r),
                    "mde": float(MDE_R), "detected": bool(d_r > MDE_R),
                    "convention": "3*sd/sqrt(n) with sd from a DELETE-ONE-KNOCKOUT jackknife, so the "
                                  "secondary metric is thresholded on the same unit of replication as the "
                                  "primary. Without this the AGREE gate compares a threshold test against "
                                  "a bare sign test and disagrees at chance."},
                "paired": {k: {"gap": paired(a, b)[0], "sd": paired(a, b)[1], "mde": paired(a, b)[2],
                               "detected": bool(paired(a, b)[0] > paired(a, b)[2])}
                           for k, (a, b) in {"model_vs_marginal": (MODEL, MARG),
                                             "model_vs_translator": (MODEL, TRANS),
                                             "model_vs_shuffled_rna": (MODEL, SHUF),
                                             "translator_vs_marginal": (TRANS, MARG)}.items()}},
        "gates": {k: bool(v) for k, v in gates.items()},
        "gaps_positive_means_first_arm_better": {
            "model_minus_marginal": gap(MODEL, MARG),
            "model_minus_calibrated_translator": gap(MODEL, TRANS),
            "model_minus_best_non_learned": gap(MODEL, best_cheap),
            "best_non_learned_arm": best_cheap,
            "model_minus_shuffled_rna_structure_preserving": gap(MODEL, SHUF),
            "model_minus_shuffled_rna_genewise": gap(MODEL, "SHUFFLED-RNA gene-wise (CONTROL)"),
            "model_minus_swap_rna_counterfactual": gap(MODEL, "SWAP-RNA (COUNTERFACTUAL)"),
            "model_minus_swap_identity_counterfactual": gap(MODEL, "SWAP-IDENTITY (COUNTERFACTUAL)"),
            "model_minus_wrong_knockout": gap(MODEL, "WRONG-KNOCKOUT (identity CONTROL)"),
            "translator_minus_marginal": gap(TRANS, MARG),
            "translator_minus_random_transcript": gap(TRANS, "RANDOM-TRANSCRIPT calibrated (CONTROL)"),
            "mlp_minus_ridge": gap("MODEL MLP (RNA + identity)", MODEL)},
        "controls_reproducing_the_effect": {k: (float(v) if np.isfinite(v) else None)
                                            for k, v in repro.items()},
        "controls_reproducing_the_effect_are_meaningful": repro_meaningful,
        "controls_reproducing_the_effect_note":
            "fractions of the model's raw RMSE gain over the marginal. If that denominator is itself below "
            "the MDE the ratios are meaningless and the flag above is false.",
        "sensitivity_same_cells_for_both_modalities": {
            "rmse": sens, "r_within": sens_r,
            "note": "if the same-cell numbers are markedly better than the graded split-half numbers, the "
                    "difference is per-droplet technical coupling between UMI and ADT capture, not biology"},
        "sensitivity_wrong_unit_guide_level_NOT_GRADED": guide_demo,
        "circularity_check":
            "RNA feature and protein target come from the same experiment but DISJOINT CELLS, so no "
            "per-droplet technical factor can manufacture the association; the same-cell version is "
            "reported as a sensitivity rather than assumed away. Identity features are DepMap CRISPR "
            "gene-effect PCs -- a different assay in different cell lines, never derived from Papalexi. "
            "The gene detection filter is computed on non-targeting cells only, which are never a held-out "
            "knockout. Every target-side quantity (marginal, translator slope/intercept, per-gene "
            "z-scores, PCA basis, ridge alpha, feature scaler) is refit on the training knockouts inside "
            "every fold.",
        "smoke": bool(SMOKE), "n_pcs": N_PC, "mlp_epochs": MLP_EPOCHS,
        "limits": [
            f"FOUR PROTEINS. The scorable universe is {N} knockouts x {NPROT} proteins = {N*NPROT} values "
            f"and no model can enlarge it. Any null here is underpowered by construction, and any positive "
            f"rests on four surface markers of one immune axis (CD86, PD-L1, PD-L2, CD366) in one cell "
            f"line. This is the single binding constraint on the module and it is not fixable by analysis",
            f"n = {N} held-out knockouts gives the sd {N-1} degrees of freedom. A gap just under the MDE is "
            f"not evidence of equality. The cheapest falsification is more knockouts -- a CITE-seq screen "
            f"with a larger panel -- not more epochs",
            "the four proteins do not respond independently: PD-L1 falls where CD86 and CD366 rise across "
            "the IFN-axis knockouts (IFNGR1/2, JAK2, STAT1, IRF1). The effective dimensionality of the "
            "target is therefore well below four, so the per-knockout RMSE replicate is the honest unit "
            "and the 100 (knockout, protein) cells must never be treated as 100 independent observations",
            "RNA and protein come from disjoint cells but from the SAME knockout, so a knockout-level batch "
            "effect (unequal 10x lane or HTO composition) still couples the two channels. The split is "
            "stratified by HTO to reduce this and it is NOT eliminated. A cross-replicate design, RNA from "
            "one replicate and protein from another, would close it and was not run",
            "the calibrated translator gets a single per-protein slope and intercept fit on the training "
            "knockouts. A richer translator -- non-linear, or using the transcript's absolute level or its "
            "measurement precision -- is a different and probably stronger arm and was not run, so 'the "
            "model beats the translator' should be read as 'beats a LINEAR translator'",
            "the model's capacity is bounded by the data, not chosen: with 24 training knockouts a ridge "
            "on a handful of PCs is near the limit of what can be fit, and the MLP arm is expected to "
            "overfit. A finding that the MLP loses to the ridge is a statement about sample size, not "
            "about architecture",
            "the ADT readout is a 4-tag panel normalised by ADT total, so the four values are compositional "
            "-- they sum to one before the log. A perturbation that raises one tag mechanically depresses "
            "the others. A CLR normalisation or an isotype/spike-in reference would break that and neither "
            "is available in this file",
            "MYC and SPI1 contribute ~50 and ~25 cells per half. Their protein estimates are the noisiest "
            "in the set and they are retained rather than dropped; re-running with VPH_MIN_CELLS raised is "
            "the obvious robustness check and a verdict that flips under it is not a verdict",
            "DepMap identity features exist for 24 of 25 knockouts (MARCH8 via the MARCHF8 alias). A gene "
            "with no DepMap profile silently receives a zero identity vector, which is a real weakness of "
            "the identity channel and is reported rather than patched",
            "one cell line (THP-1 monocytes), one time point, CRISPR knockout only. Nothing here speaks to "
            "whether a protein head generalises across cell context, dose or perturbation modality"],
        "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R_, open(OUT / "v4_protein_head.json", "w"), indent=1)
    report(f"\n  -> {OUT/'v4_protein_head.json'}")


if __name__ == "__main__":
    main()
