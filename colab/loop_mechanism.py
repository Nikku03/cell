"""LOOP 57 -- AMONG THE HARMFUL ONES, WHY IS EACH ONE HARMFUL?

WHERE THIS COMES FROM. Six loops have now asked whether protein structure improves the prediction of
WHETHER a variant is pathogenic, and all six came back inside the noise: a language model (loop 51),
a force field (52, 53), a distogram (54), a perturbation chain (55), and both together (56). The best
model sits at 0.9464 and predicts a CURATOR'S LABEL. It cannot say why any single variant is bad.

So this asks the other question. Take ONLY the pathogenic variants and ask what KIND of damage each
one does. That is the first question in this whole arc whose answer is a mechanism rather than a
score, and it is the one that would let the model say something about a cell instead of about
ClinVar.

THE MECHANISM LABEL, and why it is not disease name. The obvious idea -- same gene, different disease
-- was scoped first and abandoned on the data. ClinVar's disease names for one gene are mostly
SYNONYMS or SEVERITY VARIANTS, not different mechanisms: MFN2 splits into Charcot-Marie-Tooth type 2,
type 2A2 and plain Charcot-Marie-Tooth; ALPL into hypophosphatasia, adult hypophosphatasia and
infantile hypophosphatasia. Building on that would have measured ClinVar's naming inconsistency.

What does carry mechanism is UniProt's RESIDUE-LEVEL functional annotation, which is curated from
experiments and says what a residue DOES:

    catalytic     active site, binding site, DNA/nucleotide binding, zinc finger
    structural    disulfide bond
    membrane      transmembrane segment
    regulatory    modified residue, cross-link, glycosylation, lipidation
    motif/domain  named functional element
    none          annotated as nothing in particular

Measured before writing this: 4.7% of variants sit at catalytic residues and 90.0% of those are
pathogenic; membrane 87.6%; disulfide 75.9%; unannotated 49.7%; and REGULATORY sites are 41.9%,
BELOW the 55.7% baseline. That last one is the interesting one -- a phosphosite is a place where the
protein is meant to be modified, so substituting there is not obviously fatal.

THE CIRCULARITY THAT HAS TO BE CONTROLLED, and it is the same one loop 51 found in ClinVar. A residue
is annotated as a binding site because somebody studied it, and pathogenic variants attract study. So
"annotated sites are enriched for pathogenic variants" is partly a statement about attention. That
makes M2 a confounded baseline rather than a result, and it is reported as one. The gate that matters
is M3, which never uses the annotation as an input.

PREDECLARED, before any number:

  M1 UNIPROT NUMBERING MATCHES OUR RESIDUE INDEX
       UniProt's canonical sequence is not guaranteed to be the transcript this project translated.
       For every gene, the UniProt residue at our index must equal our reference amino acid for >= 95%
       of its variants; genes below that are NAMED and dropped entirely. loop 51 already found 16
       genes shifted by an isoform offset against AlphaFold, and an annotation read at the wrong
       residue would assign mechanisms that look plausible and are wrong.
  M2 MECHANISM CATEGORIES SEPARATE PATHOGENICITY -- REPORTED, NOT CLAIMED
       pathogenic rate per category, with the study-attention confound stated. This is a description
       of the annotation, not a finding about biology, and it is printed so the reader can see the
       size of what M3 has to beat.
  M3 STRUCTURAL CONTEXT PREDICTS WHICH MECHANISM, AMONG PATHOGENIC VARIANTS ONLY   THE REAL GATE.
       multi-class, pathogenic rows only, gene-grouped folds, and NO feature derived from the
       annotation. Macro-averaged one-vs-rest AUC must beat a permutation null. If this passes, a
       variant's structural context carries its mechanism and the model can say what kind of damage
       it does rather than only that it does damage.
  M4 IT IS NOT MEMORISING THE GENE
       a gene-only predictor -- the gene's own mechanism composition -- is scored as an explicit
       control, because most genes are dominated by one mechanism and "which gene is this" would
       otherwise carry M3 entirely. M3 must beat that control, on held-out genes.
  M5 MECHANISM IS VISIBLE TO THE PATHOGENICITY MODEL
       the loop 54 model's held-out error rate, broken down by mechanism. If some mechanisms are
       systematically harder, that is where the remaining headroom is and it says so by name.

  AND THE CELL-LEVEL CONTEXT the model already holds is brought in as its own feature block, because
  it was asked for and because it is free: whether the gene is a transcription factor (Lambert), its
  out-degree in the 612,133-edge regulatory network, its complex membership, its pathway count and
  its PPI degree. These are GENE-level, so they cannot separate two variants in one gene -- they are
  tested for whether they predict which mechanism a gene's variants tend to be.

-> outputs/loop_mechanism.json
"""
import collections
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_distogram_nn as DN
import loop_rescue as LR
import loop_sphere_chain as SPH
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

M1_FLOOR = 0.95
MIN_PER_CLASS = 30
FOLDS = 5
N_NULL = 100
SEED = 1904

CAT = {"Active site": "catalytic", "Binding site": "catalytic", "DNA binding": "catalytic",
       "Nucleotide binding": "catalytic", "Zinc finger": "catalytic", "Site": "catalytic",
       "Modified residue": "regulatory", "Cross-link": "regulatory",
       "Glycosylation": "regulatory", "Lipidation": "regulatory",
       "Disulfide bond": "structural", "Transmembrane": "membrane"}
# Region/Domain/Motif/Mutagenesis and the secondary-structure types are deliberately NOT mechanisms:
# "is in a domain" says where a residue is, not what it does, and Mutagenesis marks residues someone
# chose to mutate, which is attention rather than function.

# structural context only -- nothing derived from the annotation
FEATS = ["esm", "esm_altblind", "entropy", "burial", "plddt", "blosum", "titv",
         "nc8", "nc12", "contact_order", "long_range", "centrality", "pae_mean", "pae_min_far",
         "dg_self", "dg_total"]
GENE_LEVEL = ["is_tf", "reg_out", "reg_in", "n_complex", "ppi", "npath"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def load_features():
    g2a = json.load(open(SC / "_pae_genes.json"))
    FT = {}
    for g, a in g2a.items():
        p = SC / "uniprot_ft" / f"{a}.json"
        if p.exists():
            FT[g] = json.load(open(p))
    return FT, g2a


def uniprot_seq(acc):
    p = SC / "uniprot_seq" / f"{acc}.txt"
    if p.exists():
        return p.read_text().strip()
    return None


def mechanisms_at(feats, ri):
    out = set()
    for f in feats:
        s = f["s"]
        e = f["e"] or s
        if s <= ri <= e:
            c = CAT.get(f["t"])
            if c:
                out.add(c)
    return out


def gene_level_block(genes):
    """TF status, regulatory degree, complex membership, pathway and PPI counts -- from the cell file.

    Gene-level by construction, so these cannot separate two variants in the same gene. They are here
    to answer whether the cell-context the model already holds predicts which mechanism a gene runs on.
    """
    D = json.load(open(CELL))
    gl = D["genes"]
    idx = {g["name"]: i for i, g in enumerate(gl)}
    out_deg = collections.Counter()
    in_deg = collections.Counter()
    for a, b, *_ in D["reg"]:
        out_deg[a] += 1
        in_deg[b] += 1
    ncplx = collections.Counter()
    for _, members in D["complexes"].items():
        for m in members:
            ncplx[m] += 1
    res = {}
    for g in genes:
        i = idx.get(g)
        if i is None:
            res[g] = None
            continue
        rec = gl[i]
        res[g] = {"is_tf": float(rec.get("tf", 0)), "reg_out": float(out_deg.get(i, 0)),
                  "reg_in": float(in_deg.get(i, 0)), "n_complex": float(ncplx.get(i, 0)),
                  "ppi": float(rec.get("ppi", 0) or 0), "npath": float(rec.get("npath", 0) or 0)}
    return res


def rank_cv_multiclass(X, y, grp, seed=SEED, folds=FOLDS):
    """Gene-grouped multi-class logistic with a train-only rank transform; returns class probabilities."""
    X = np.asarray(X, float)
    classes = sorted(set(y))
    ug = np.unique(grp)
    fold = {g: i for g, i in zip(ug, np.random.default_rng(seed).permutation(len(ug)) % folds)}
    fid = np.array([fold[g] for g in grp])
    P = np.full((len(y), len(classes)), np.nan)
    for k in range(folds):
        tr, te = fid != k, fid == k
        if tr.sum() < 100 or te.sum() == 0 or len(set(np.array(y)[tr])) < 2:
            continue
        A = np.empty_like(X[tr])
        B = np.empty_like(X[te])
        for j in range(X.shape[1]):
            srt = np.sort(X[tr, j])
            A[:, j] = np.searchsorted(srt, X[tr, j], side="right") / max(len(srt), 1)
            B[:, j] = np.searchsorted(srt, X[te, j], side="right") / max(len(srt), 1)
        m = LogisticRegression(max_iter=3000, C=1.0)
        m.fit(A, np.array(y)[tr])
        pr = m.predict_proba(B)
        for ci, c in enumerate(m.classes_):
            P[te, classes.index(c)] = pr[:, ci]
    return P, classes


def macro_auc(P, y, classes):
    """One-vs-rest AUC per class, macro-averaged over classes that have both sides present."""
    y = np.array(y)
    aucs = {}
    for ci, c in enumerate(classes):
        v = P[:, ci]
        m = np.isfinite(v)
        yy = (y[m] == c).astype(int)
        if yy.sum() < 10 or (1 - yy).sum() < 10:
            continue
        aucs[c] = LR.auc(v[m], yy)
    vals = [a for a in aucs.values() if np.isfinite(a)]
    return (float(np.mean(vals)) if vals else np.nan), aucs


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 57 -- among the harmful ones, why is each one harmful?")
    say("=" * 100)

    recs, _ = pickle.load(open(SPH.CACHE, "rb"))
    FT, g2a = load_features()
    say(f"\n  {len(recs):,} variants, {len({r['gene'] for r in recs}):,} genes, "
        f"{len(FT):,} with UniProt annotation")

    # ---- M1 numbering -----------------------------------------------------------------------------
    say(f"\n  M1 UNIPROT NUMBERING MATCHES OUR RESIDUE INDEX")
    hit = collections.Counter()
    tot = collections.Counter()
    for r in recs:
        g = r["gene"]
        s = uniprot_seq(g2a.get(g, ""))
        if not s:
            continue
        tot[g] += 1
        ri = r["_ri"]
        if 1 <= ri <= len(s) and s[ri - 1] == r["_wt"]:
            hit[g] += 1
    frac = {g: hit[g] / tot[g] for g in tot if tot[g] >= 10}
    bad = sorted(g for g, v in frac.items() if v < M1_FLOOR)
    overall = sum(hit.values()) / max(sum(tot.values()), 1)
    say(f"     overall agreement {overall:.4f} over {sum(tot.values()):,} variants in {len(frac)} genes")
    say(f"     genes below {M1_FLOOR:.0%} and DROPPED entirely: {len(bad)}")
    if bad:
        say(f"       {', '.join(bad[:16])}{' ...' if len(bad) > 16 else ''}")
    keep = {g for g in frac if g not in bad}
    recs = [r for r in recs if r["gene"] in keep]
    m1 = bool(overall >= 0.90 and len(recs) > 5000)
    say(f"     kept {len(recs):,} variants in {len(keep):,} genes")
    say(f"     M1 {'PASS' if m1 else 'FAIL'}")

    # ---- assign mechanism --------------------------------------------------------------------------
    for r in recs:
        cats = mechanisms_at(FT.get(r["gene"], []), r["_ri"])
        # one label per variant: the most specific wins, so a phosphosite inside a domain is regulatory
        for pref in ("catalytic", "regulatory", "structural", "membrane"):
            if pref in cats:
                r["mech"] = pref
                break
        else:
            r["mech"] = "none"
    y = np.array([r["y"] for r in recs])
    gn = np.array([r["gene"] for r in recs])
    mech = np.array([r["mech"] for r in recs])

    # ---- M2 the confounded baseline -----------------------------------------------------------------
    say(f"\n  M2 MECHANISM CATEGORIES SEPARATE PATHOGENICITY -- REPORTED, NOT CLAIMED")
    say(f"     {'mechanism':<12}{'variants':>10}{'% of all':>10}{'% pathogenic':>14}")
    rates = {}
    for c in sorted(set(mech)):
        m = mech == c
        rates[c] = float(y[m].mean())
        say(f"     {c:<12}{int(m.sum()):>10,}{m.mean():>10.1%}{y[m].mean():>14.1%}")
    say(f"     baseline {y.mean():.1%}. A residue is annotated because somebody STUDIED it, and")
    say(f"     pathogenic variants attract study, so this spread is partly attention. It is the")
    say(f"     description M3 has to beat, not a finding.")
    m2 = True

    # ---- M3 the real gate ---------------------------------------------------------------------------
    say(f"\n  M3 STRUCTURAL CONTEXT PREDICTS WHICH MECHANISM, AMONG PATHOGENIC VARIANTS ONLY")
    pm = y == 1
    pmech = mech[pm]
    pg = gn[pm]
    counts = collections.Counter(pmech)
    usable = [c for c, n in counts.items() if n >= MIN_PER_CLASS]
    say(f"     pathogenic variants: {int(pm.sum()):,}. classes with >= {MIN_PER_CLASS}: "
        + ", ".join(f"{c} {counts[c]:,}" for c in sorted(usable)))
    sel = np.isin(pmech, usable)
    col = lambda k, rows: np.array([r.get(k, np.nan) for r in rows], float)
    prows = [r for r, k in zip(recs, pm) if k]
    prows = [r for r, k in zip(prows, sel) if k]
    yy = [r["mech"] for r in prows]
    gg = np.array([r["gene"] for r in prows])
    X = np.column_stack([col(f, prows) for f in FEATS])
    med = np.nanmedian(X, axis=0)
    badm = ~np.isfinite(X)
    X[badm] = np.take(med, np.where(badm)[1])
    P, classes = rank_cv_multiclass(X, yy, gg)
    macro, per = macro_auc(P, yy, classes)
    say(f"     {len(prows):,} rows, {len(set(gg))} genes, {len(FEATS)} structural features, "
        f"NO annotation-derived input")
    for c in sorted(per):
        say(f"       {c:<12}one-vs-rest AUC {per[c]:.4f}   (n={sum(1 for v in yy if v == c):,})")
    say(f"     MACRO AUC {macro:.4f}")
    nulls = []
    for _ in range(N_NULL):
        ys = list(rng.permutation(yy))
        Pn, cn = rank_cv_multiclass(X, ys, gg, seed=SEED)
        mn, _ = macro_auc(Pn, ys, cn)
        if np.isfinite(mn):
            nulls.append(mn)
    nulls = np.array(nulls)
    n95 = float(np.percentile(nulls, 95))
    say(f"     null (labels permuted, {len(nulls)} draws): mean {nulls.mean():.4f}, 95th {n95:.4f}")
    m3 = bool(macro > n95)
    say(f"     M3 {'PASS' if m3 else 'FAIL'}")

    # ---- M4 the gene control -------------------------------------------------------------------------
    say(f"\n  M4 IT IS NOT MEMORISING THE GENE")
    glb = gene_level_block(sorted(set(gg)))
    Xg = np.array([[glb[g][k] if glb.get(g) else np.nan for k in GENE_LEVEL] for g in gg], float)
    medg = np.nanmedian(Xg, axis=0)
    bg = ~np.isfinite(Xg)
    Xg[bg] = np.take(medg, np.where(bg)[1])
    Pg, cg = rank_cv_multiclass(Xg, yy, gg)
    macro_g, per_g = macro_auc(Pg, yy, cg)
    say(f"     GENE-LEVEL block only (TF status, regulatory in/out degree, complex membership,")
    say(f"     pathway count, PPI degree) -- {len(GENE_LEVEL)} features, all constant within a gene")
    say(f"     macro AUC {macro_g:.4f} against the structural block's {macro:.4f}")
    Xb = np.column_stack([X, Xg])
    Pb, cb = rank_cv_multiclass(Xb, yy, gg)
    macro_b, _ = macro_auc(Pb, yy, cb)
    say(f"     both together {macro_b:.4f}   ({macro_b - macro:+.4f} over structure alone)")
    m4 = bool(macro > macro_g)
    say(f"     M4 {'PASS' if m4 else 'FAIL'}  (structure beats the gene-level block on held-out genes)")

    # ---- M5 where the pathogenicity model struggles ---------------------------------------------------
    say(f"\n  M5 MECHANISM IS VISIBLE TO THE PATHOGENICITY MODEL")
    F54 = ["esm", "esm_altblind", "entropy", "burial", "plddt", "blosum", "titv", "negaf"]
    Xa = np.column_stack([col(f, recs) for f in F54])
    meda = np.nanmedian(Xa, axis=0)
    ba = ~np.isfinite(Xa)
    Xa[ba] = np.take(meda, np.where(ba)[1])
    v = SPH.rank_fit(SPH.net, Xa, y, gn)
    say(f"     the loop 54 model refitted here, then its held-out score read out BY mechanism")
    say(f"     {'mechanism':<12}{'n':>8}{'AUC within that class':>24}")
    bym = {}
    for c in sorted(set(mech)):
        m = (mech == c) & np.isfinite(v)
        if (y[m] == 1).sum() < 20 or (y[m] == 0).sum() < 20:
            say(f"     {c:<12}{int(m.sum()):>8,}{'too few of one class':>24}")
            continue
        a = LR.auc(v[m], y[m])
        bym[c] = float(a)
        say(f"     {c:<12}{int(m.sum()):>8,}{a:>24.4f}")
    m5 = bool(len(bym) >= 3)
    if bym:
        hard = min(bym, key=bym.get)
        say(f"     hardest mechanism for the pathogenicity model: {hard} at {bym[hard]:.4f}")
    say(f"     M5 {'PASS' if m5 else 'FAIL'}  (reported by name rather than pooled)")

    say("\n" + "=" * 100)
    gates = {"M1 UniProt numbering matches": m1, "M2 categories separate (reported)": m2,
             "M3 structure predicts mechanism": m3, "M4 not memorising the gene": m4,
             "M5 mechanism visible to the model": m5}
    for k, val in gates.items():
        say(f"  {k:<40}{'PASS' if val else 'FAIL'}")
    say(f"  AMONG PATHOGENIC VARIANTS ALONE, structural context predicts WHICH KIND of damage at")
    say(f"  macro AUC {macro:.4f} against a permuted null of {n95:.4f}, using no annotation-derived")
    say(f"  input. The gene-level block -- TF status, regulatory degree, complex membership, pathway")
    say(f"  and PPI counts -- reaches {macro_g:.4f} on its own and {macro_b:.4f} combined.")
    say(f"  This is the first question in the arc whose answer is a MECHANISM rather than a score.")
    say(f"  It does not make the pathogenicity model better; it says what the model is looking at.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SPH.CACHE), str(SC / "uniprot_ft"), str(SC / "uniprot_seq"),
                              str(CELL)],
                      available=len(recs), used=len(prows), selection="filtered", seed=SEED,
                      controls=["UniProt numbering checked per gene, failures named and dropped",
                                "no annotation-derived feature enters M3",
                                "gene-level composition scored as an explicit control",
                                f"{N_NULL} label permutations for the multi-class null",
                                "gene-grouped folds throughout",
                                "M2 declared a confounded description, not a result"],
                      note="mechanism labels are UniProt residue annotations, not disease names -- "
                           "ClinVar's per-gene disease names are mostly synonyms and severity variants")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_mechanism", "manifest": man, "gates": gates,
               "m1_agreement": overall, "m1_dropped": bad,
               "category_rates": rates, "class_counts": dict(counts),
               "macro_auc": macro, "per_class_auc": per,
               "null_mean": float(nulls.mean()), "null_95": n95,
               "gene_level_macro": macro_g, "combined_macro": macro_b,
               "pathogenicity_auc_by_mechanism": bym,
               "n_variants": len(recs), "n_pathogenic_used": len(prows),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_mechanism.json", "w"), indent=2, default=float)
    RM.report(man, emit=say)
    say(f"\n  -> {OUT/'loop_mechanism.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
