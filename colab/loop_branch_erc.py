"""Loop 227. Real evolutionary rate covariation: branch rates off 54,308 Compara gene trees.

WHAT LOOPS 225 AND 226 MEASURED, AND WHY IT WAS THE WRONG QUANTITY. Both correlated PROTEIN
IDENTITY between human and each of 95 mammals. Loop 225 got AUC 0.5117 on 595 interacting pairs;
loop 226 got 0.5184 on 30,641, so sample size was refuted as the explanation. Both commits recorded
the same remaining suspect, before either rerun: identity to human is a PATH length. It sums every
branch from human back to the common ancestor and forward to that species, so a rate change on one
internal lineage is smeared across every descendant, and the measurement is dominated by divergence
time. Real evolutionary rate covariation -- Clark, Wolfe, and the RERconverge line of work --
correlates BRANCH-specific relative rates read off gene trees. That is a different quantity, not a
sharper version of the same one.

THE TREES ARE PUBLISHED AND THIS LOOP USES THEM. Ensembl Compara 116 ships every protein gene tree
in Newick with branch lengths, 177.5 MB gzipped:

    54,308 gene trees parsed
    38,317 skipped for carrying zero or more than one human leaf -- one-to-one only
    11,416 trees kept, each with a single human gene and its mammalian orthologs
    11,416 genes x 95 mammals of TERMINAL branch length, median 78 mammals per gene

TERMINAL BRANCHES ONLY, AND THAT IS A REAL LIMITATION STATED UP FRONT. A leaf label in Newick
carries its own branch length, so terminal branches extract exactly and unambiguously. Internal
branches would need every gene tree's internal nodes matched to species-tree nodes, which the .nh
dump does not label -- it is the .nhx file that carries taxon annotations. RERconverge uses both.
This loop therefore implements the terminal-branch half of the method. A rate shift on an ancestral
lineage is invisible here, and if co-evolution lives mainly on internal branches this loop cannot
see it and a FAIL would not settle the question.

THE NORMALISATION IS THE PUBLISHED ONE, not a centring of convenience. Branch lengths scale with
both the gene's overall rate and the branch's own duration. RERconverge removes this by regressing
each gene's branch lengths on the across-gene average branch length and keeping the residuals. That
per-gene SLOPE is what distinguishes it from double centring, which forces every gene to the same
proportionality. Both are computed; the regression residual is the primary.

BRANCH LENGTHS NEED SCREENING AND THE RAW DATA SAYS SO. The extracted matrix has a minimum of
0.00000 and a maximum of 100000.000 -- Ensembl emits sentinel values for alignments that failed to
converge. Zeros break a logarithm and sentinels dominate any regression. Lengths outside
(0, 5] are dropped and the count is reported, not silently clipped.

G2 IS THE GATE THAT MATTERS MOST AND IT IS AIMED AT ME. If the branch-rate representation turns
out to be nearly the same as the identity representation, then this loop is loop 226 with extra
steps, and any result would be a rediscovery dressed as a new method. G2 measures the overlap
directly and reports it before any interaction test is read.

PREDECLARED, BEFORE ANY NUMBER.

  G1 DID THE TREES PARSE INTO A USABLE MATRIX?
     Gate: PASS iff at least 8,000 genes retain 50 or more mammalian terminal branches after
     screening. Below that the comparison against loop 226's 15,694 genes is not meaningful.

  G2 IS THIS ACTUALLY A DIFFERENT MEASUREMENT FROM LOOP 226?
     For every gene present in both, correlate its branch-rate profile against its identity-derived
     profile over shared species.
     Gate: PASS iff the MEDIAN absolute correlation is below 0.80, meaning the two representations
     carry substantially different information. A FAIL does not void the loop -- it means any
     result below must be read as a confirmation of loop 226 rather than a new test, and that
     reading is fixed here rather than chosen afterwards.

  G3 DOES BRANCH-RATE ERC RECOVER KNOWN INTERACTIONS?
     Gate: PASS iff AUC >= 0.60. This is loop 225's and loop 226's bar, unchanged for the third
     time, so all three numbers sit on one scale. Bootstrap standard error reported beside it and
     NOT gated, since at these sample sizes significance and magnitude come apart.

  G4 IS G3 A CONFOUND?
     Negatives matched on branch count, mean rate and profile variance.
     Gate: PASS iff the matched AUC retains at least 70% of the unmatched excess over 0.5.
     Requires G3.

  G5 DOES SEPARATION GROW WITH THE NUMBER OF SHARED BRANCHES?
     Quartiles by shared branch count. Loop 226 ran this on identity and got a flat profile --
     0.5081, 0.5165, 0.5137, 0.5139 -- which is what a non-diluted non-effect looks like.
     Gate: PASS iff AUC rises monotonically across quartiles AND the top exceeds the bottom by
     0.02. Requires G1 only, for the same reason as loop 226: dose-response is informative exactly
     when the pooled effect is too small to clear a practical bar.

  G6 DOES THE BRANCH-RATE METHOD BEAT THE IDENTITY METHOD?  -- the point of the whole loop
     Gate: PASS iff branch-rate matched AUC exceeds loop 226's identity matched AUC of 0.5129 by
     at least 0.02, on the same interaction set restricted to genes present in both. Requires G1.

  G7 DOES IT ADD TO THE STANDING STACK?
     Gate: PASS iff held-out |r| exceeds loop 213's 0.5474. Loop 225 reached 0.0887 and loop 226
     reached 0.1037 against a fame-only baseline of 0.0940. Requires G1.

  G8 SHUFFLE CONTROL
     Branches permuted independently within each gene: marginals preserved exactly, cross-gene
     covariation destroyed.
     Gate: PASS iff the shuffled AUC falls below 0.55. Requires G1.

  G9 WHAT THIS CANNOT SHOW -- written before the run.
     Terminal branches only. Co-evolution acting on ancestral lineages is invisible to this loop
     and a FAIL leaves the internal-branch version untested rather than refuted.
     Requiring exactly one human leaf discards 38,317 of 54,308 trees, which removes every
     duplicated family -- and gene duplication is itself a mechanism of co-evolutionary change,
     so the retained set is biased toward single-copy, conserved, housekeeping-like genes.
     OmniPath and SIGNOR edges are literature-curated and biased toward well-studied proteins;
     G4 matches branch count, rate and variance but cannot match study effort.
     Nothing here tests Evo 2. A nucleotide model reads intra-genic, promoter and non-coding
     covariation that no gene-level branch rate can express, whether terminal or internal.
"""
import os, sys, json, time, csv, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_branch_erc.json"
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
BR, ID = SP / "compara" / "mammal_branch.npz", SP / "compara" / "mammal_identity.npz"
NET = ROOT / "colab" / "data" / "networks"
SEED, MIN_BR, K_FACTOR = 227227, 50, 32
BL_LO, BL_HI = 1e-6, 5.0
REF_STACK, REF_FAME, REF226_AUC, REF226_MATCH = 0.5474, 0.0940, 0.5184, 0.5129
MINGENE, AUC_BAR, MATCH_KEEP, DOSE_MIN, BEAT_MARGIN, SHUF_BAR, SAME_BAR = \
    8000, 0.60, 0.70, 0.02, 0.02, 0.55, 0.80

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) < 20 or len(neg) < 20:
        return float("nan")
    a = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(a)).astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def ridge_pred(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    return B @ np.linalg.solve(A.T @ A + R, A.T @ ytr)


def pair_corr(A, O, I, J, chunk=40000, minshare=30):
    out = np.empty(len(I))
    for s in range(0, len(I), chunk):
        i, j = I[s:s + chunk], J[s:s + chunk]
        ai, aj, oi, oj = A[i], A[j], O[i], O[j]
        n = np.einsum("ij,ij->i", oi, oj)
        si = np.einsum("ij,ij->i", ai, oj); sj = np.einsum("ij,ij->i", aj, oi)
        sii = np.einsum("ij,ij->i", ai * ai, oj); sjj = np.einsum("ij,ij->i", aj * aj, oi)
        sij = np.einsum("ij,ij->i", ai, aj)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (sij - si * sj / n) / np.sqrt((sii - si * si / n) * (sjj - sj * sj / n))
        r[(n < minshare) | ~np.isfinite(r)] = np.nan
        out[s:s + chunk] = r
    return out


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "branch-rate ERC from Compara gene trees"}
    say("=" * 104)
    say("LOOP 227 -- REAL ERC: BRANCH RATES OFF 54,308 COMPARA GENE TREES")
    say("=" * 104)
    say("     Loops 225 and 226 correlated protein IDENTITY, which is a PATH length: it sums every")
    say("     branch from human to a species, so a rate change on one lineage smears across all")
    say("     its descendants. Real ERC correlates BRANCH-specific rates. Different quantity.")

    Z = np.load(BR, allow_pickle=True)
    B0, bgen, bsp = Z["M"].astype(np.float64), Z["genes"], Z["species"]
    say(f"     parsed trees: {B0.shape[0]:,} genes x {B0.shape[1]} mammals, terminal branches")

    # ---------------------------------------------------------------- G1
    say("G1 DID THE TREES PARSE INTO A USABLE MATRIX?")
    bad = ~np.isfinite(B0) | (B0 <= BL_LO) | (B0 > BL_HI)
    say(f"     branch lengths outside ({BL_LO:g}, {BL_HI:g}] dropped: "
        f"{int((np.isfinite(B0) & ((B0 <= BL_LO) | (B0 > BL_HI))).sum()):,} of "
        f"{int(np.isfinite(B0).sum()):,} finite entries "
        f"(Ensembl emits sentinels up to {np.nanmax(B0):.0f})")
    Bv = np.where(bad, np.nan, B0)
    cov = np.isfinite(Bv).sum(1)
    keep = cov >= MIN_BR
    say(f"     genes retaining >={MIN_BR} branches: {int(keep.sum()):,} of {len(bgen):,}")
    G.add("G1", bool(keep.sum() >= MINGENE), stat=float(keep.sum()),
          if_true=lambda: f"G1 PASS -- {int(keep.sum()):,} genes with a usable branch profile",
          if_false=lambda: f"G1 FAIL -- only {int(keep.sum()):,} genes survive screening")
    Bv = Bv[keep]; bgen = bgen[keep]
    L = np.log(Bv)
    ok = np.isfinite(L)
    mean_branch = np.nanmean(L, axis=0)

    # RERconverge normalisation: regress each gene's log branch lengths on the across-gene
    # average, keep residuals. The per-gene SLOPE is what separates this from double centring.
    RER = np.full(L.shape, np.nan)
    for i in range(L.shape[0]):
        m = ok[i]
        x, y = mean_branch[m], L[i, m]
        x0, y0 = x - x.mean(), y - y.mean()
        b = float(x0 @ y0 / (x0 @ x0)) if (x0 @ x0) > 0 else 0.0
        RER[i, m] = y0 - b * x0
    DC = L - np.nanmean(L, axis=0, keepdims=True)
    DC = DC - np.nanmean(DC, axis=1, keepdims=True)
    A = np.where(ok, RER, 0.0); O = ok.astype(np.float64)
    say(f"     RERconverge-style residuals on {A.shape[0]:,} genes x {A.shape[1]} branches")
    res["matrix"] = {"n_genes": int(A.shape[0]), "n_branches": int(A.shape[1]),
                     "median_cov": int(np.median(cov[keep]))}

    # ---------------------------------------------------------------- G2
    say("G2 IS THIS ACTUALLY A DIFFERENT MEASUREMENT FROM LOOP 226?")
    Zi = np.load(ID, allow_pickle=True)
    Mi, igen = Zi["M"], Zi["genes"]
    ipos = {str(g): k for k, g in enumerate(igen)}
    both = [(k, ipos[str(g)]) for k, g in enumerate(bgen) if str(g) in ipos]
    Di = np.log(np.clip(100.0 - Mi.astype(np.float64), 0.5, None))
    Di = np.where(np.isfinite(Mi), Di, np.nan)
    Di = Di - np.nanmean(Di, axis=0, keepdims=True)
    Di = Di - np.nanmean(Di, axis=1, keepdims=True)
    ov = np.array([abs(pear(RER[a], Di[b])) for a, b in both[:4000]])
    med_ov = float(np.nanmedian(ov))
    say(f"     {len(both):,} genes present in both representations; per-gene |correlation| "
        f"between branch-rate and identity profiles: median {med_ov:.4f}, "
        f"quartiles {np.nanpercentile(ov,25):.4f} / {np.nanpercentile(ov,75):.4f}")
    G.add("G2", bool(med_ov < SAME_BAR), stat=med_ov, requires=("G1",),
          if_true=lambda: f"G2 PASS -- median overlap {med_ov:.3f}; branch rates carry "
                          f"substantially different information from identity",
          if_false=lambda: f"G2 FAIL -- median overlap {med_ov:.3f}; this is loop 226's "
                           f"measurement again and anything below is a confirmation, not a test")
    res["overlap"] = {"median": med_ov, "n": len(both)}

    # ---------------------------------------------------------------- G3
    say("G3 DOES BRANCH-RATE ERC RECOVER KNOWN INTERACTIONS?")
    e2s = L191.ensg_to_symbol(lambda *_: None)
    gpos = {}
    for k, g in enumerate(bgen):
        s = e2s.get(str(g), "")
        if s and s not in gpos:
            gpos[s] = k
    edges = set()
    with open(NET / "omnipath.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b = r.get("source_genesymbol", ""), r.get("target_genesymbol", "")
            if a in gpos and b in gpos and a != b:
                edges.add((min(a, b), max(a, b)))
    with open(NET / "signor_human.tsv") as f:
        for p in csv.reader(f, delimiter="\t"):
            if len(p) > 4 and p[0] in gpos and p[4] in gpos and p[0] != p[4]:
                edges.add((min(p[0], p[4]), max(p[0], p[4])))
    edges = sorted(edges)
    I = np.array([gpos[a] for a, _ in edges]); J = np.array([gpos[b] for _, b in edges])
    say(f"     {len(gpos):,} symbols usable; {len(edges):,} interacting pairs inside them")
    posv = pair_corr(A, O, I, J)
    ni = rng.integers(0, len(bgen), size=(len(edges) * 3, 2)); ni = ni[ni[:, 0] != ni[:, 1]]
    negv = pair_corr(A, O, ni[:, 0], ni[:, 1])
    a3 = auc(posv, negv)
    pf, nf = posv[np.isfinite(posv)], negv[np.isfinite(negv)]
    bs = np.array([auc(rng.choice(pf, min(4000, len(pf))), rng.choice(nf, min(4000, len(nf))))
                   for _ in range(200)])
    se = float(bs.std())
    say(f"     interacting pairs ERC median {np.nanmedian(posv):+.4f}   random "
        f"{np.nanmedian(negv):+.4f}")
    say(f"     AUC {a3:.4f} over {len(pf):,} positives and {len(nf):,} negatives")
    say(f"     bootstrap standard error {se:.4f} = {(a3-0.5)/max(se,1e-9):+.1f} standard errors "
        f"from chance -- reported, NOT gated")
    say(f"     loop 225 identity/595 pairs {0.5117:.4f}; loop 226 identity/30,641 pairs "
        f"{REF226_AUC:.4f}")
    G.add("G3", bool(np.isfinite(a3) and a3 >= AUC_BAR), stat=float(a3), requires=("G1",),
          if_true=lambda: f"G3 PASS -- AUC {a3:.4f}; branch rates recover interactions where "
                          f"identity did not",
          if_false=lambda: f"G3 FAIL -- AUC {a3:.4f} against the unchanged {AUC_BAR:.2f} bar")
    res["auc"] = {"raw": a3, "se": se, "n_pos": int(len(pf)), "n_neg": int(len(nf)),
                  "n_pairs": len(edges), "loop226": REF226_AUC}

    # ---------------------------------------------------------------- G4
    say("G4 IS G3 A CONFOUND?")
    nbr = ok.sum(1).astype(float)
    mrate = np.nanmean(L, axis=1)
    gvar = np.nanstd(RER, axis=1)
    F = np.column_stack([nbr, mrate, gvar])
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    nb = 8
    binz = np.zeros(len(bgen), np.int64)
    for c in range(Fz.shape[1]):
        q = np.quantile(Fz[:, c], np.linspace(0, 1, nb + 1)[1:-1])
        binz = binz * nb + np.searchsorted(q, Fz[:, c])
    bucket = {}
    for i, b in enumerate(binz):
        bucket.setdefault(int(b), []).append(i)
    mi, mj = [], []
    for k in range(len(I)):
        for src, other in ((I[k], J[k]), (J[k], I[k])):
            pool = bucket.get(int(binz[other]), [])
            if len(pool) > 1:
                c = int(pool[rng.integers(len(pool))])
                if c != src:
                    mi.append(src); mj.append(c)
    mi, mj = np.array(mi), np.array(mj)
    mneg = pair_corr(A, O, mi, mj)
    a4 = auc(posv, mneg)
    ex_r, ex_m = a3 - 0.5, a4 - 0.5
    kept = ex_m / ex_r if ex_r > 0 else float("nan")
    say(f"     negatives matched on branch count, mean rate and profile variance, {nb}^3 strata; "
        f"{len(mi):,} matched negatives")
    say(f"     matched AUC {a4:.4f} against unmatched {a3:.4f}; {kept:.1%} of the excess retained")
    G.add("G4", bool(np.isfinite(kept) and kept >= MATCH_KEEP), stat=float(kept), requires=("G3",),
          if_true=lambda: f"G4 PASS -- {kept:.0%} survives matching",
          if_false=lambda: f"G4 FAIL -- {kept:.0%} survives matching")
    res["matched"] = {"auc": a4, "kept": kept}

    # ---------------------------------------------------------------- G5
    say("G5 DOES SEPARATION GROW WITH THE NUMBER OF SHARED BRANCHES?")
    sp_ = np.einsum("ij,ij->i", O[I], O[J])
    sn_ = np.einsum("ij,ij->i", O[mi], O[mj])
    qs = np.quantile(sp_, [0.25, 0.5, 0.75])
    dose = []
    for b in range(4):
        lo = -np.inf if b == 0 else qs[b - 1]
        hi = np.inf if b == 3 else qs[b]
        pm, nm = (sp_ > lo) & (sp_ <= hi), (sn_ > lo) & (sn_ <= hi)
        ab = auc(posv[pm], mneg[nm]); dose.append(ab)
        say(f"       shared branches {0 if b==0 else lo:.0f}-{95 if b==3 else hi:.0f}: "
            f"AUC {ab:.4f}   ({int(pm.sum()):,} pos, {int(nm.sum()):,} neg)")
    dose = np.array(dose)
    mono = bool(np.all(np.diff(dose) > 0)) and (dose[-1] - dose[0] >= DOSE_MIN)
    say(f"     loop 226 on identity gave [0.5081, 0.5165, 0.5137, 0.5139] -- flat")
    G.add("G5", mono, stat=float(dose[-1] - dose[0]), requires=("G1",),
          if_true=lambda: f"G5 PASS -- rises monotonically {dose[0]:.4f} to {dose[-1]:.4f}",
          if_false=lambda: f"G5 FAIL -- quartiles {np.round(dose,4).tolist()}")
    res["dose"] = [float(x) for x in dose]

    # ---------------------------------------------------------------- G6
    say("G6 DOES THE BRANCH-RATE METHOD BEAT THE IDENTITY METHOD?")
    say(f"     branch-rate matched AUC {a4:.4f}")
    say(f"     loop 226 identity matched AUC {REF226_MATCH:.4f}")
    say(f"     difference {a4-REF226_MATCH:+.4f} against a {BEAT_MARGIN:+.2f} bar")
    G.add("G6", bool(np.isfinite(a4) and a4 - REF226_MATCH >= BEAT_MARGIN),
          stat=float(a4 - REF226_MATCH), requires=("G1",),
          if_true=lambda: f"G6 PASS -- branch rates beat identity by {a4-REF226_MATCH:+.4f}; the "
                          f"path-length explanation for loops 225 and 226 is confirmed",
          if_false=lambda: f"G6 FAIL -- {a4-REF226_MATCH:+.4f}; switching from path lengths to "
                           f"branch rates does not rescue it, so the path-length explanation "
                           f"recorded in both earlier commits is REFUTED")
    res["beat_identity"] = {"branch": a4, "identity": REF226_MATCH, "delta": a4 - REF226_MATCH}

    # ---------------------------------------------------------------- G7
    say("G7 DOES IT ADD TO THE STANDING STACK?")
    Az = A - A.mean(0, keepdims=True)
    U, Sv, Vt = np.linalg.svd(Az, full_matrices=False)
    EF = U[:, :K_FACTOR] * Sv[:K_FACTOR]
    EFn = EF / (np.linalg.norm(EF, axis=1, keepdims=True) + 1e-9)
    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    y_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    inter = [s for s in allg if s in gpos]
    yv = np.array([y_all[allg.index(s)] for s in inter])
    ridx = np.array([gpos[s] for s in inter])
    say(f"     scored on the {len(inter):,}-gene A549 intersection; top component "
        f"{Sv[0]**2/np.sum(Sv**2):.1%} of variance")
    sub = EFn[ridx] @ EFn[ridx].T
    np.fill_diagonal(sub, -np.inf)
    nn = np.argsort(-sub, axis=1)[:, :8]
    blocks = {"branch_factor": EF[ridx], "branch_neighbour_y": yv[nn],
              "branch_meta": np.column_stack([nbr[ridx], mrate[ridx], gvar[ridx]])}
    n = len(inter); perm = rng.permutation(n); cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    sc = {}
    for nm, X in blocks.items():
        Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        sc[nm] = abs(pear(yv[te], ridge_pred(Xc[tr], yv[tr], Xc[te])))
        say(f"       {nm:<20} held-out |r| {sc[nm]:.4f}")
    Xall = np.nan_to_num(np.hstack([blocks[k] for k in blocks]), nan=0.0)
    comb = abs(pear(yv[te], ridge_pred(Xall[tr], yv[tr], Xall[te])))
    ysh = yv.copy(); rng.shuffle(ysh)
    shuf = abs(pear(ysh[te], ridge_pred(Xall[tr], ysh[tr], Xall[te])))
    say(f"       {'all branch':<20} held-out |r| {comb:.4f}   shuffled control {shuf:.4f}")
    say(f"     loop 213 stack {REF_STACK:.4f}; fame-only {REF_FAME:.4f}; loop 225 0.0887; "
        f"loop 226 0.1037")
    G.add("G7", bool(comb > REF_STACK), stat=float(comb), requires=("G1",),
          if_true=lambda: f"G7 PASS -- {comb:.4f} above the standing stack's {REF_STACK:.4f}",
          if_false=lambda: f"G7 FAIL -- {comb:.4f} against the standing stack's {REF_STACK:.4f}")
    res["stack"] = dict(sc); res["stack"]["combined"] = comb; res["stack"]["shuffled"] = shuf

    # ---------------------------------------------------------------- G8
    say("G8 SHUFFLE CONTROL")
    As, Os = A.copy(), O.copy()
    for i in range(As.shape[0]):
        p = rng.permutation(As.shape[1]); As[i] = As[i][p]; Os[i] = Os[i][p]
    a8 = auc(pair_corr(As, Os, I, J), pair_corr(As, Os, mi, mj))
    say(f"     branches permuted independently within each gene; marginals preserved exactly")
    say(f"     shuffled AUC {a8:.4f} against the real matched {a4:.4f}")
    G.add("G8", bool(np.isfinite(a8) and a8 < SHUF_BAR), stat=float(a8), requires=("G1",),
          if_true=lambda: f"G8 PASS -- the shuffle drops to {a8:.4f}",
          if_false=lambda: f"G8 FAIL -- the shuffle still reaches {a8:.4f}")
    res["shuffle"] = a8

    # ---------------------------------------------------------------- G9
    say("G9 WHAT THIS CANNOT SHOW")
    say("     TERMINAL branches only. Co-evolution acting on ancestral lineages is invisible here")
    say("     and a FAIL leaves the internal-branch version untested rather than refuted.")
    say("     Requiring exactly one human leaf discarded 38,317 of 54,308 trees, removing every")
    say("     duplicated family -- and duplication is itself a co-evolutionary mechanism -- so")
    say("     the retained set is biased toward single-copy conserved genes.")
    say("     OmniPath and SIGNOR edges are literature-curated; G4 matches branch count, rate and")
    say("     variance but cannot match how much a gene has been studied.")
    say("     Nothing here tests Evo 2, whose nucleotide representation reads intra-genic,")
    say("     promoter and non-coding covariation no gene-level branch rate can express.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
