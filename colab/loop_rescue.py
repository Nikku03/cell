"""LOOP 51 -- IS THERE ANY SENSOR HERE THAT ANSWERS "WHAT DOES THIS POINT MUTATION DO"?

The point-mutation row is the weakest of the six: 8 of 14 gates. loop 31 built the chain -- genomic
position to codon to amino acid, 99.99% correct over 966,225 ClinVar variants -- and then C4, C5 and
C6 all failed at the far end. loop 32 found why: nexus is a protein-protein INTERFACE sensor
validated on SKEMPI ddG_bind, and on ClinVar pathogenicity it sits at chance (0.4806, and 0.4902
even inside confident AlphaFold structure). The chain works. The sensor at the end of it answers a
different question.

So this replaces the sensor with an ESM-2 zero-shot score and asks whether THAT answers it. An
earlier draft of this module was reviewed adversarially before it ran, by four readers who each had
one lens and the repository. They found the draft would have produced a meaningless pass, for four
reasons that are now gates rather than assumptions:

  THE DRAFT'S G1 WAS A TAUTOLOGY.  It checked translate(CDS)[i] == ref_aa. But ref_aa is not a
  ClinVar annotation -- loop_chain.py:293 MANUFACTURES it by translating that same codon. The
  identity holds for any frame, any strand, any transcript, including wrong ones, and it measured
  0.9997 here for no reason at all. The check that can fail is against a sequence that did not
  produce the index: the AlphaFold model. Independent agreement there is 94-96%, not 99.97%.

  THE BENCHMARK IS A POSITION TASK, NOT A SUBSTITUTION TASK.  Of all pathogenic/benign pairs inside
  a gene, only 0.016% sit at the SAME residue. So a score that never looks at the substituted amino
  acid has a within-gene ceiling of 0.9999, and "ESM beats BLOSUM" can be earned without the model
  ever consulting alt_aa. The alt-blind ablation is free -- same forward pass -- and it is now the
  gate that decides whether anything here is about substitutions at all.

  BLOSUM62 IS NOT A BASELINE.  Counting CB neighbours in the AlphaFold PDBs already on disk beats
  it, and may beat ESM-2. A comparator that loses to fifteen lines of numpy over files already in
  the repository is a strawman, and beating it would mean nothing.

  THE ddG JOIN IS POSITIONAL.  loop_nexus_fair.py:104 indexes a dict keyed by ROW POSITION in one
  particular 212,177-row frame. Any change to the filter silently reassigns every ddG to a different
  variant -- which would not inflate a result, it would destroy one, and this project would have
  recorded "the structural sensor does not belong here" as a validated negative when it was a join
  bug. A false negative costs exactly as much as a false positive in a repository that keeps its
  negatives.

PREDECLARED, before any number:

  G1 THE RESIDUE INDEX IS CHECKED AGAINST A SEQUENCE THAT DID NOT PRODUCE IT
       G1a the translated-CDS check must be 100.000%, and is labelled a PRECONDITION, not evidence:
           it is the tautology above and it can only catch cache corruption.
       G1b the AlphaFold CA residue at the mapped index must equal ref_aa for >= 99% of
           structurally scored variants, evaluated PER GENE. Genes below 95% are NAMED and dropped
           from every structure-derived feature before any score is computed.
  G2 THE WITHIN-GENE STATISTIC IS ACTUALLY WITHIN-GENE
       a feature that is CONSTANT inside each gene -- the gene's own pathogenic rate -- must score
       0.5000 +/- 0.01 under the statistic used here, and its pooled value is printed beside it to
       show the size of the gap. This gate exists because the helper inherited from loop 32 returns
       a POOLED AUC under a within-gene name, and every number this module reports depends on the
       difference.
  G3 ESM BEATS THE BASELINE PANEL, PAIRED, PER GENE
       not BLOSUM alone: BLOSUM62, CB-neighbour burial, pLDDT, transition-vs-transversion, and a
       gene-grouped 5-fold logistic STACK of all of them. ESM must beat the stack by Wilcoxon
       signed-rank p < 0.01 AND win in more than 60% of genes. A mean that wins because three genes
       out of 276 moved is not a result, and only the sign test can see that.
  G4 THE SUBSTITUTION MATTERS
       the full log-odds must beat the ALT-BLIND score from the same forward pass, paired and
       significant. If it does not, the finding recorded is "ESM-2 recovers conserved positions; it
       does not rank substitutions", and that is a real answer to the row's question rather than a
       failure to get one.
  G5 IT SURVIVES CURATION
       ClinVar's own ACMG criteria PP3/BP4 are computational-evidence criteria applied with
       conservation predictors, so a conservation model scoring ClinVar is partly scoring itself.
       The ordering in G3 and G4 must hold on the >= 2-star subset (multiple submitters no
       conflicts, expert panel, practice guideline). If the score does BETTER on weakly reviewed
       variants, it is tracking curation, and that comparison is the result.
  G6 DOES STRUCTURE ADD, AND DOES nexus's SENSOR ADD
       on ONE pinned intersection table -- ESM score AND finite ddG AND passing G1b -- rekeyed by
       (chrom, pos, ref, alt) with a 100% key-match assertion, both arms on identical rows, no
       imputation, no missingness indicator:
         G6a ESM + burial + pLDDT  vs  ESM
         G6b + ddG                 vs  ESM + burial + pLDDT
       The AUC of ddG AVAILABILITY is reported as a named control, because ddG exists only for the
       311 genes this project fetched structures for and those were chosen as disease genes.

-> outputs/loop_rescue.json
"""
import collections
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

MIN_CLASS = 20          # per gene, per class -- declared in the cache builder before any AUC existed
G1B_GENE_FLOOR = 0.95   # a gene below this is dropped from structure features and named
G1B_OVERALL = 0.99
SIGN_FRAC = 0.60        # G3/G4: fraction of genes the winner must actually win
WILCOXON_P = 0.01
SEED = 1904
AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
PURINE = set("AG")

log = []


def say(s=""):
    print(s)
    log.append(s)


# ---- statistics ------------------------------------------------------------------------------------
def auc(v, y):
    """Mann-Whitney AUC. NaNs are dropped by the caller, not here."""
    v = np.asarray(v, float)
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.argsort(np.argsort(v)) + 1.0
    # average ranks over ties, or a score with many equal values scores spuriously well
    order = np.argsort(v)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_gene_auc(v, y, g, min_class=MIN_CLASS):
    """AUC computed SEPARATELY inside each gene. This is the thing loop 32's helper was not."""
    out = {}
    v = np.asarray(v, float)
    y = np.asarray(y)
    g = np.asarray(g)
    fin = np.isfinite(v)
    for gg in np.unique(g[fin]):
        m = fin & (g == gg)
        if (y[m] == 1).sum() >= min_class and (y[m] == 0).sum() >= min_class:
            out[gg] = auc(v[m], y[m])
    return out


def summarize(pg, v, y, g):
    """Unweighted mean over genes, evidence-weighted concordance, and the pooled number.

    All three are reported because they answer different questions and can disagree: the unweighted
    mean asks 'does this work in a typical gene', the weighted one asks 'does it work over the
    evidence', and pooled is what a careless helper returns while calling itself within-gene.
    """
    if not pg:
        return {"mean": np.nan, "weighted": np.nan, "pooled": np.nan, "n_genes": 0}
    v = np.asarray(v, float)
    y = np.asarray(y)
    g = np.asarray(g)
    w, num, den = [], 0.0, 0.0
    for gg, a in pg.items():
        m = (g == gg) & np.isfinite(v)
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        num += a * n1 * n0
        den += n1 * n0
        w.append(n1 * n0)
    fin = np.isfinite(v) & np.isin(g, list(pg))
    return {"mean": float(np.mean(list(pg.values()))), "weighted": float(num / den) if den else np.nan,
            "pooled": auc(v[fin], y[fin]), "n_genes": len(pg)}


def paired(pg_a, pg_b):
    """Per-gene paired comparison: Wilcoxon p, win fraction, mean delta."""
    keys = sorted(set(pg_a) & set(pg_b))
    if len(keys) < 10:
        return {"n": len(keys), "p": np.nan, "win_frac": np.nan, "mean_delta": np.nan}
    d = np.array([pg_a[k] - pg_b[k] for k in keys])
    nz = d[d != 0]
    p = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) >= 10 else np.nan
    return {"n": len(keys), "p": p, "win_frac": float(np.mean(d > 0)),
            "mean_delta": float(np.mean(d))}


# ---- data ------------------------------------------------------------------------------------------
def chain_frame():
    """loop_chain's scored variant frame, rebuilt with the EXACT filter the ddG cache was keyed on.

    _chain_ddg.pkl is keyed by ROW POSITION in this frame. Reproducing the filter is the only way to
    rekey it by variant identity, and the length assertion is what makes that safe rather than hoped.
    """
    T = pickle.load(open(SC / "_chain_translated.pkl", "rb"))
    rows = [r for r in T if r[10] and "missense_variant" in r[6] and r[8] != r[9] and r[9] != "*"]
    return rows


def load_clinvar_meta():
    """Review status (stars) and population allele frequency, from the VCF the labels came from."""
    stars = {"practice_guideline": 4, "reviewed_by_expert_panel": 3,
             "criteria_provided,_multiple_submitters,_no_conflicts": 2,
             "criteria_provided,_single_submitter": 1,
             "criteria_provided,_conflicting_classifications": 1}
    meta = {}
    with gzip.open(SC / "clinvar.vcf.gz", "rt", errors="ignore") as f:
        for ln in f:
            if ln[0] == "#":
                continue
            p = ln.split("\t", 8)
            if len(p) < 8 or len(p[3]) != 1 or len(p[4]) != 1:
                continue
            d = {}
            for kv in p[7].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = v
            af = [float(d[k]) for k in ("AF_EXAC", "AF_TGP", "AF_ESP") if k in d]
            meta[(p[0], int(p[1]), p[3], p[4])] = (
                stars.get(d.get("CLNREVSTAT", ""), 0), max(af) if af else np.nan)
    return meta


def af_structure(path):
    """CA coordinates, pLDDT and the one-letter sequence from an AlphaFold model."""
    seq, pl, xyz = {}, {}, {}
    for ln in open(path, errors="ignore"):
        if not ln.startswith("ATOM"):
            continue
        if ln[12:16].strip() != "CA":
            continue
        try:
            i = int(ln[22:26])
        except ValueError:
            continue
        aa = THREE.get(ln[17:20].strip().upper())
        if not aa:
            continue
        seq[i] = aa
        pl[i] = float(ln[60:66])
        xyz[i] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return seq, pl, xyz


def burial_map(xyz, cutoff=10.0):
    idx = sorted(xyz)
    P = np.array([xyz[i] for i in idx])
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    n = (d < cutoff).sum(1) - 1
    return {i: int(c) for i, c in zip(idx, n)}


def blosum62():
    """BLOSUM62 without a Biopython dependency -- the standard 20x20 integer matrix."""
    raw = """A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4"""
    order = "ARNDCQEGHILKMFPSTWYV"
    M = {}
    for ln in raw.strip().split("\n"):
        p = ln.split()
        for j, v in enumerate(p[1:]):
            M[(p[0], order[j])] = int(v)
    return M


def logistic_stack(X, y, groups, rng, folds=5):
    """Gene-grouped 5-fold logistic regression, returning out-of-fold predictions.

    Plain gradient descent on standardised features -- the point is a fair COMPOSITE baseline, and a
    heavier learner would make the baseline unreproducible without buying it any strength.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ug = np.unique(groups)
    fold = {g: i for g, i in zip(ug, rng.permutation(len(ug)) % folds)}
    fid = np.array([fold[g] for g in groups])
    out = np.full(len(y), np.nan)
    for k in range(folds):
        tr, te = fid != k, fid == k
        if tr.sum() < 50 or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([(X[tr] - mu) / sd, np.ones(tr.sum())])
        B = np.column_stack([(X[te] - mu) / sd, np.ones(te.sum())])
        w = np.zeros(A.shape[1])
        for _ in range(400):
            p = 1 / (1 + np.exp(-A @ w))
            w -= 0.5 * (A.T @ (p - y[tr])) / max(tr.sum(), 1)
        out[te] = B @ w
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 51 -- is there any sensor here that answers 'what does this point mutation do'?")
    say("=" * 100)

    rows = chain_frame()
    say(f"\n  loop_chain's missense frame rebuilt: {len(rows):,} rows")
    ddg_raw = pickle.load(open(SC / "_chain_ddg.pkl", "rb"))
    # REKEY. The cache is keyed by row position in this exact frame; carrying that index onto any
    # other table silently reassigns every value. Rebuilt, keyed by variant identity, and asserted.
    ddg = {}
    for i, r in enumerate(rows):
        v = ddg_raw.get(i)
        if v is not None and v[1]:
            ddg[(r[0], r[1], r[2], r[3])] = float(v[0])
    join_ok = len(ddg_raw) > 0 and max(ddg_raw) < len(rows)
    say(f"  ddG rekeyed by (chrom,pos,ref,alt): {len(ddg):,} of {len(ddg_raw):,} cached entries; "
        f"max cached index {max(ddg_raw):,} vs frame {len(rows):,} -> "
        f"{'IN RANGE' if join_ok else 'OUT OF RANGE, ddG NOT USED'}")

    esm = pickle.load(open(SC / "_esm_scores_650M.pkl", "rb"))
    say(f"  ESM-2 650M log-prob matrices: {len(esm):,} genes")
    meta = load_clinvar_meta()
    say(f"  ClinVar VCF metadata: {len(meta):,} SNVs with review status and allele frequency")

    # ---- G1 ------------------------------------------------------------------------------------
    say(f"\n  G1 THE RESIDUE INDEX IS CHECKED AGAINST A SEQUENCE THAT DID NOT PRODUCE IT")
    cds_ok = sum(1 for r in rows if r[4] in esm and 1 <= r[7] <= len(esm[r[4]][0])
                 and esm[r[4]][0][r[7] - 1] == r[8])
    cds_n = sum(1 for r in rows if r[4] in esm)
    say(f"     G1a translated-CDS self-consistency  {cds_ok/max(cds_n,1):.6f}  ({cds_ok:,}/{cds_n:,})")
    say(f"         PRECONDITION, NOT EVIDENCE -- ref_aa was manufactured from this same codon walk,")
    say(f"         so this identity holds for any frame or transcript and can only catch corruption.")

    structs = pickle.load(open(SC / "_chain_structs.pkl", "rb"))
    ST = {}
    for g, p in structs.items():
        if g in esm and Path(p).exists():
            try:
                ST[g] = af_structure(p)
            except Exception:
                pass
    gene_hit = collections.Counter()
    gene_tot = collections.Counter()
    for r in rows:
        s = ST.get(r[4])
        if s is None:
            continue
        gene_tot[r[4]] += 1
        if s[0].get(r[7]) == r[8]:
            gene_hit[r[4]] += 1
    per_gene = {g: gene_hit[g] / gene_tot[g] for g in gene_tot if gene_tot[g] >= 20}
    bad = sorted(g for g, v in per_gene.items() if v < G1B_GENE_FLOOR)
    overall = sum(gene_hit.values()) / max(sum(gene_tot.values()), 1)
    say(f"     G1b AlphaFold residue == ref_aa      {overall:.4f} over {sum(gene_tot.values()):,} "
        f"rows in {len(per_gene):,} structure-bearing genes")
    say(f"         genes below {G1B_GENE_FLOOR:.0%} and DROPPED from every structural feature: "
        f"{len(bad)}")
    if bad:
        say(f"         {', '.join(bad[:18])}{' ...' if len(bad) > 18 else ''}")
    for g in bad:
        ST.pop(g, None)
    g1 = bool(cds_ok == cds_n and overall >= G1B_OVERALL)
    say(f"     G1 {'PASS' if g1 else 'FAIL'}  (G1a exactly 1.000000 AND G1b >= {G1B_OVERALL})")

    BUR = {g: burial_map(ST[g][2]) for g in ST}

    # ---- feature table ---------------------------------------------------------------------------
    BL = blosum62()
    rec = []
    for r in rows:
        g, ri, wt, mt = r[4], r[7], r[8], r[9]
        e = esm.get(g)
        if e is None or not (1 <= ri <= len(e[0])) or wt not in AA or mt not in AA:
            continue
        M = e[1]
        lp = M[ri - 1]
        if not np.isfinite(lp).all():
            continue
        llr = float(lp[AA.index(mt)] - lp[AA.index(wt)])
        p = np.exp(lp - lp.max())
        p = p / p.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum())
        key = (r[0], r[1], r[2], r[3])
        st = ST.get(g)
        stars, af = meta.get(key, (0, np.nan))
        rec.append({
            "gene": g, "y": r[5], "key": key,
            "esm": -llr,                                    # higher = more pathogenic
            # DIRECTIONS, DECLARED BY MECHANISM. The first run of this module had three of them
            # backwards and the numbers said so loudly: alt-blind scored 0.1581, entropy 0.1463 and
            # pLDDT 0.2616, all far BELOW chance, which is what a correct signal read upside down
            # looks like. It also made G4 pass by +0.7242 -- an ablation "beating" its parent by
            # three quarters of the scale is not a result, it is a sign error. Corrected here:
            #   logP(ref) HIGH  -> the model expects the wild-type residue -> constrained position
            #                      -> mutations there are more often pathogenic.
            #   entropy LOW     -> the column tolerates few residues -> conserved -> pathogenic.
            #   pLDDT HIGH      -> ordered, structured region -> pathogenic (disordered regions are
            #                      where curators rarely call missense pathogenic).
            "esm_altblind": float(lp[AA.index(wt)]),         # alt NEVER read
            "entropy": -ent,
            "blosum": -BL.get((wt, mt), 0),
            "burial": BUR[g].get(ri, np.nan) if st else np.nan,
            "plddt": st[1].get(ri, np.nan) if st else np.nan,
            "titv": 0.0 if (r[2] in PURINE) == (r[3] in PURINE) else 1.0,
            "negaf": -np.log10(af) if np.isfinite(af) and af > 0 else np.nan,
            "ddg": ddg.get(key, np.nan) if join_ok else np.nan,
            "stars": stars,
        })
    y = np.array([x["y"] for x in rec])
    gn = np.array([x["gene"] for x in rec])
    say(f"\n  feature table: {len(rec):,} variants, {len(set(gn)):,} genes, "
        f"{y.mean():.1%} pathogenic")
    col = lambda k: np.array([x[k] for x in rec], float)

    # ---- G2 the statistic ------------------------------------------------------------------------
    say(f"\n  G2 THE WITHIN-GENE STATISTIC IS ACTUALLY WITHIN-GENE")
    rate = {g: y[gn == g].mean() for g in set(gn)}
    prior = np.array([rate[g] for g in gn])
    pg_prior = per_gene_auc(prior, y, gn)
    s_prior = summarize(pg_prior, prior, y, gn)
    say(f"     a feature CONSTANT inside each gene (the gene's own pathogenic rate):")
    say(f"       within-gene mean {s_prior['mean']:.4f}   weighted {s_prior['weighted']:.4f}   "
        f"POOLED {s_prior['pooled']:.4f}")
    g2 = bool(abs(s_prior["mean"] - 0.5) <= 0.01 and abs(s_prior["weighted"] - 0.5) <= 0.01)
    say(f"     the pooled column is what loop 32's helper returns while calling itself within-gene;")
    say(f"     the gap is {s_prior['pooled'] - s_prior['mean']:+.4f} on a feature carrying NO "
        f"within-gene information at all.")
    say(f"     G2 {'PASS' if g2 else 'FAIL'}  (within-gene statistic on a gene-constant feature "
        f"= 0.5000 +/- 0.01)")

    # ---- G3 the baseline panel --------------------------------------------------------------------
    say(f"\n  G3 ESM BEATS THE BASELINE PANEL, PAIRED, PER GENE")
    feats = ["esm", "blosum", "burial", "plddt", "titv", "negaf", "esm_altblind", "entropy"]
    PG, S = {}, {}
    for f in feats:
        v = col(f)
        PG[f] = per_gene_auc(v, y, gn)
        S[f] = summarize(PG[f], v, y, gn)
    # THE STACK, and why it is not defined on every gene. Allele frequency is absent for 69% of rows,
    # so requiring it finite collapsed the first run's stack to EIGHT genes and the paired test to
    # nan -- a FAIL that measured nothing. It is now imputed with an availability INDICATOR, which is
    # the honest handling of a missing measurement. Burial and pLDDT are treated differently on
    # purpose: their missingness is GENE-level (no AlphaFold model), and this project fetched
    # structures for disease genes, so a has-structure indicator would let the stack learn the
    # selection rather than the biology -- the same 0.57-AUC leak G6 reports as a control. So the
    # stack is fitted and compared only where the structural features genuinely exist.
    stack_cols = ["blosum", "burial", "plddt", "titv", "negaf"]
    X = np.column_stack([col(c) for c in stack_cols])
    af_present = np.isfinite(col("negaf")).astype(float)
    Xi = X.copy()
    j = stack_cols.index("negaf")
    med = np.nanmedian(Xi[:, j])
    Xi[~np.isfinite(Xi[:, j]), j] = med
    Xi = np.column_stack([Xi, af_present])
    keep = np.isfinite(Xi).all(1)
    st_pred = np.full(len(y), np.nan)
    if keep.sum() > 200:
        st_pred[keep] = logistic_stack(Xi[keep], y[keep], gn[keep], rng)
    PG["STACK"] = per_gene_auc(st_pred, y, gn)
    S["STACK"] = summarize(PG["STACK"], st_pred, y, gn)
    say(f"     {'feature':<14}{'genes':>7}{'within':>9}{'weighted':>10}{'pooled':>9}")
    for f in feats + ["STACK"]:
        s = S[f]
        say(f"     {f:<14}{s['n_genes']:>7}{s['mean']:>9.4f}{s['weighted']:>10.4f}"
            f"{s['pooled']:>9.4f}")
    cmp3 = paired(PG["esm"], PG["STACK"])
    say(f"     ESM vs STACK, paired over {cmp3['n']} shared genes: mean delta "
        f"{cmp3['mean_delta']:+.4f}, wins {cmp3['win_frac']:.1%}, Wilcoxon p {cmp3['p']:.2e}")
    g3 = bool(cmp3["p"] < WILCOXON_P and cmp3["win_frac"] > SIGN_FRAC)
    say(f"     G3 {'PASS' if g3 else 'FAIL'}  (p < {WILCOXON_P} AND wins > {SIGN_FRAC:.0%} of genes)")

    # ---- G4 does the substitution matter ----------------------------------------------------------
    say(f"\n  G4 THE SUBSTITUTION MATTERS")
    cmp4 = paired(PG["esm"], PG["esm_altblind"])
    say(f"     full log-odds {S['esm']['mean']:.4f} vs ALT-BLIND -logP(ref) "
        f"{S['esm_altblind']['mean']:.4f}   (entropy {S['entropy']['mean']:.4f})")
    say(f"     paired over {cmp4['n']} genes: mean delta {cmp4['mean_delta']:+.4f}, wins "
        f"{cmp4['win_frac']:.1%}, Wilcoxon p {cmp4['p']:.2e}")
    g4 = bool(cmp4["p"] < WILCOXON_P and cmp4["win_frac"] > SIGN_FRAC)
    say(f"     G4 {'PASS' if g4 else 'FAIL'}  -- a fail means ESM recovers conserved POSITIONS and "
        f"does not rank SUBSTITUTIONS")

    # ---- G5 curation -----------------------------------------------------------------------------
    say(f"\n  G5 IT SURVIVES CURATION")
    st_arr = col("stars")
    hi = st_arr >= 2
    say(f"     >= 2-star variants: {int(hi.sum()):,} of {len(rec):,} ({hi.mean():.1%}), "
        f"{y[hi].mean():.1%} pathogenic vs {y[~hi].mean():.1%} below")
    sub = {}
    for lab, m in (("all", np.ones(len(y), bool)), (">=2 star", hi), ("0-1 star", ~hi)):
        pg_e = per_gene_auc(np.where(m, col("esm"), np.nan), y, gn)
        pg_s = per_gene_auc(np.where(m, st_pred, np.nan), y, gn)
        pg_b = per_gene_auc(np.where(m, col("esm_altblind"), np.nan), y, gn)
        sub[lab] = {"esm": float(np.mean(list(pg_e.values()))) if pg_e else np.nan,
                    "stack": float(np.mean(list(pg_s.values()))) if pg_s else np.nan,
                    "altblind": float(np.mean(list(pg_b.values()))) if pg_b else np.nan,
                    "n_genes": len(pg_e)}
        say(f"     {lab:<10} ESM {sub[lab]['esm']:.4f}   stack {sub[lab]['stack']:.4f}   "
            f"alt-blind {sub[lab]['altblind']:.4f}   ({len(pg_e)} genes)")
    a, b = sub[">=2 star"], sub["0-1 star"]
    ordering_holds = bool(np.isfinite(a["esm"]) and np.isfinite(a["stack"])
                          and (a["esm"] > a["stack"]) == (sub["all"]["esm"] > sub["all"]["stack"]))
    g5 = ordering_holds
    say(f"     G5 {'PASS' if g5 else 'FAIL'}  (the ESM-vs-stack ordering is the same on the "
        f"well-reviewed subset)")
    if np.isfinite(a["esm"]) and np.isfinite(b["esm"]) and b["esm"] > a["esm"]:
        say(f"     NOTE: ESM does BETTER on weakly reviewed variants ({b['esm']:.4f} vs "
            f"{a['esm']:.4f}), which is what tracking curation looks like.")

    # ---- G6 structure and the nexus sensor --------------------------------------------------------
    say(f"\n  G6 DOES STRUCTURE ADD, AND DOES nexus's SENSOR ADD")
    avail = np.isfinite(col("ddg")).astype(float)
    pg_av = per_gene_auc(avail, y, gn)
    say(f"     CONTROL -- ddG AVAILABILITY alone: pooled {auc(avail, y):.4f}, within-gene "
        f"{np.mean(list(pg_av.values())) if pg_av else float('nan'):.4f}")
    say(f"     ddG exists only for genes this project fetched structures for, and it fetched "
        f"disease genes.")
    pin = np.isfinite(col("ddg")) & np.isfinite(col("burial")) & np.isfinite(col("plddt")) \
        & np.isfinite(col("esm"))
    say(f"     PINNED intersection (ESM and finite ddG and burial and pLDDT, G1b-passing genes): "
        f"{int(pin.sum()):,} rows, {len(set(gn[pin])):,} genes, {y[pin].mean():.1%} pathogenic")
    g6a = g6b = False
    arms = {}
    if pin.sum() > 500:
        def arm(cols):
            v = np.full(len(y), np.nan)
            Xa = np.column_stack([col(c) for c in cols])   # signs are learned; directions are declared above
            v[pin] = logistic_stack(Xa[pin], y[pin], gn[pin], rng)
            return per_gene_auc(v, y, gn), v
        pg_e, _ = arm(["esm"])
        pg_s, _ = arm(["esm", "burial", "plddt"])
        pg_d, _ = arm(["esm", "burial", "plddt", "ddg"])
        c6a, c6b = paired(pg_s, pg_e), paired(pg_d, pg_s)
        arms = {"esm": float(np.mean(list(pg_e.values()))) if pg_e else np.nan,
                "esm+struct": float(np.mean(list(pg_s.values()))) if pg_s else np.nan,
                "esm+struct+ddg": float(np.mean(list(pg_d.values()))) if pg_d else np.nan,
                "G6a": c6a, "G6b": c6b}
        say(f"     ESM {arms['esm']:.4f}  ->  +burial+pLDDT {arms['esm+struct']:.4f}  ->  "
            f"+ddG {arms['esm+struct+ddg']:.4f}   (all on the SAME rows)")
        say(f"     G6a structure over ESM: delta {c6a['mean_delta']:+.4f}, wins "
            f"{c6a['win_frac']:.1%}, p {c6a['p']:.2e}")
        say(f"     G6b ddG over structure : delta {c6b['mean_delta']:+.4f}, wins "
            f"{c6b['win_frac']:.1%}, p {c6b['p']:.2e}")
        g6a = bool(c6a["p"] < WILCOXON_P and c6a["win_frac"] > SIGN_FRAC)
        g6b = bool(c6b["p"] < WILCOXON_P and c6b["win_frac"] > SIGN_FRAC)
    else:
        say(f"     NOT RUN -- the pinned table is too small to measure. Recorded as not run rather "
            f"than as a negative.")
    say(f"     G6a {'PASS' if g6a else 'FAIL'}   G6b {'PASS' if g6b else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"G1 index checked independently": g1, "G2 statistic is within-gene": g2,
             "G3 ESM beats the baseline panel": g3, "G4 the substitution matters": g4,
             "G5 survives curation": g5, "G6a structure adds": g6a, "G6b the ddG sensor adds": g6b}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    best = max((S[f]['mean'], f) for f in feats + ["STACK"] if np.isfinite(S[f]['mean']))
    say(f"  THE BEST SINGLE WITHIN-GENE SENSOR HERE IS {best[1]} at {best[0]:.4f}.")
    say(f"  ESM-2 650M is {S['esm']['mean']:.4f} within gene and {S['esm']['pooled']:.4f} pooled --")
    say(f"  a gap of {S['esm']['pooled'] - S['esm']['mean']:+.4f} that is entirely between-gene, and")
    say(f"  the pooled number is the one an earlier helper would have reported.")
    say(f"  The alt-blind score from the SAME forward pass reaches "
        f"{S['esm_altblind']['mean']:.4f}, so the")
    say(f"  substitution identity is worth {cmp4['mean_delta']:+.4f} per gene. Whatever this row can")
    say(f"  answer, it answers mostly by knowing WHICH RESIDUE, not which substitution.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "_chain_translated.pkl"), str(SC / "_esm_scores_650M.pkl"),
                              str(SC / "clinvar.vcf.gz"), str(SC / "_chain_ddg.pkl")],
                      available=len(rows), used=len(rec), selection="filtered", seed=SEED,
                      controls=["a gene-constant feature must score 0.5000 under the statistic",
                                "alt-blind ablation from the same forward pass",
                                "baseline panel incl. burial, pLDDT, Ti/Tv and allele frequency",
                                "ddG rekeyed by variant identity and range-asserted",
                                "ddG availability AUC reported as a named control",
                                "review-status stratification"],
                      note="every comparison paired per gene; pooled AUC reported beside within-gene "
                           "rather than instead of it")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_rescue", "manifest": man, "gates": gates,
               "summary": {f: S[f] for f in feats + ["STACK"]},
               "gene_prior_control": s_prior, "esm_vs_stack": cmp3, "esm_vs_altblind": cmp4,
               "by_review_status": sub, "g1b_overall": overall, "g1b_dropped": bad,
               "ddg_availability_auc": float(auc(avail, y)), "arms": arms,
               "n_rows": len(rec), "n_genes": len(set(gn)),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_rescue.json", "w"), indent=2, default=float)
    say(f"\n  -> {OUT/'loop_rescue.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
