"""Loop 226. The same co-evolution test at genome scale, because loop 225 was underpowered.

WHAT LOOP 225 DID AND WHY IT IS BEING REDONE. Loop 225 measured evolutionary rate covariation
across 95 mammals and its positive control failed: AUC 0.5117 against a predeclared 0.60 bar on
595 known interacting pairs. Three explanations were recorded at the time, before any rerun:
the positive set was too small, the rate proxy was protein identity rather than branch rates, and
95 species are not independent draws. Only the first is fixable with data already on disk, and it
was self-inflicted -- loop 225 scored on the 1,336-gene A549 dexamethasone responder set because
that is what the stack uses, when all 18,757 genes were already parsed and sitting in the same
file.

    loop 225      1,246 genes usable,      595 interacting pairs
    this loop    15,694 genes usable,   26,989 interacting pairs      45x more

THE BAR IS NOT MOVED. F2 keeps loop 225's 0.60 AUC bar, unchanged, because the whole point is to
find out whether 0.5117 was a sample-size artefact or a real absence. Changing the bar at the same
time as the sample would make the comparison uninterpretable.

AND A DISTINCTION THAT MATTERS AT THIS SAMPLE SIZE, stated before the numbers exist. With 26,989
positives the standard error on an AUC is roughly 0.004, so an AUC of 0.52 is more than four
standard errors above chance and would be overwhelmingly SIGNIFICANT while remaining practically
useless -- it would rank an interacting pair above a random one 52 times in 100. This loop reports
both, and F2 gates on the practical bar. A p-value is not permitted to stand in for an effect
here; loops 87 and 87b already produced "54% survival" and "6718% survival" from two numbers that
were both zero, and significance-without-magnitude is the same error wearing better clothes.

WHAT IS STILL NOT FIXED, and neither is discovered after the fact. The rate proxy is still protein
identity, because every dn and ds field in Ensembl Compara 116's default dump is NULL -- 0 of
1,446,407 mammalian rows carry both. Real evolutionary rate covariation correlates BRANCH-specific
relative rates read off gene trees; pairwise identity to human is dominated by species divergence
time, and the double centring removes the average of that but not its per-gene interaction with
tree topology. And 95 species remain 95 non-independent points per pair.

F4 IS THE NEW TEST AND IT IS THE INTERESTING ONE. If co-evolution signal exists but is buried in
per-pair noise, then pairs sharing MORE species should separate better, because each ERC value is
a correlation whose error shrinks with its own sample size. That is a dose-response prediction with
a direction fixed in advance, and it can detect a real effect that the pooled AUC cannot. It
deliberately does NOT require F2: a signal too weak to clear a practical bar is exactly the case
where dose-response is informative, and gating it behind F2 would throw that away.

PREDECLARED, BEFORE ANY NUMBER.

  F1 IS THE POSITIVE SET NOW LARGE ENOUGH TO SETTLE THE QUESTION?
     Gate: PASS iff at least 10,000 interacting pairs fall inside the usable gene set. Below that
     the rerun cannot distinguish itself from loop 225 and there is no point scoring it.

  F2 DOES CO-EVOLUTION RECOVER KNOWN INTERACTIONS AT GENOME SCALE?
     Gate: PASS iff AUC >= 0.60 -- loop 225's bar, unchanged. The permutation significance is
     reported alongside and is NOT part of the gate.

  F3 IS F2 A CONFOUND?
     Negatives matched to positives on ortholog count, mean divergence and gene-level variance.
     Gate: PASS iff the matched AUC retains at least 70% of the unmatched excess over 0.5.
     Requires F2.

  F4 DOES SEPARATION GROW WITH THE NUMBER OF SHARED SPECIES?  -- dose-response, not gated on F2
     Split pairs into quartiles by how many species both members have, and compute a matched AUC
     inside each quartile.
     Gate: PASS iff AUC increases monotonically across the four quartiles AND the top quartile
     exceeds the bottom by at least 0.02. A flat or falling profile says the pooled result is not
     a diluted real effect.

  F5 DOES A LEARNED EMBEDDING BEAT PAIRWISE ERC?
     FACTOR (rank-32 factorisation) and MLP (masked-reconstruction autoencoder, the Evo-2-style
     objective) trained on all 15,694 genes rather than 1,246.
     Gate: PASS iff the best learned arm exceeds ERC's matched AUC by at least 0.02. Requires F2.

  F6 DOES IT ADD TO THE STANDING STACK?
     The embeddings are now trained genome-wide but scored on the A549 intersection, where the
     target exists. That is the actual improvement over loop 225: same evaluation, better-estimated
     representation.
     Gate: PASS iff held-out |r| exceeds loop 213's 0.5474. Requires F1 only -- a block can be
     useful for prediction even if it does not recover curated interactions, and gating this
     behind F2 would confuse two different questions.

  F7 RIDGE AGAINST MLP, SHUFFLED CONTROL ON BOTH
     Loop 225 E6 found the MLP winning for the first time: 0.1222 against ridge 0.0887, margins
     over their own shuffled controls +0.1106 and +0.0667. Loop 211 had the opposite, ridge 0.4057
     against MLP-wide 0.2072. Repeated here on genome-wide-trained features.
     Gate: PASS iff the better arm beats its own shuffled-label control by at least 0.05.

  F8 SPECIES-SHUFFLE CONTROL
     Species permuted independently within each gene: marginals preserved exactly, cross-gene
     covariation destroyed.
     Gate: PASS iff the shuffled AUC falls below 0.55.

  F9 WHAT THIS CANNOT SHOW -- written before the run.
     A genome-wide FAIL would still be a statement about protein identity across 95 mammals, not
     about co-evolution as such and not about Evo 2, whose nucleotide-resolution representation
     reads intra-genic, promoter and non-coding covariation that gene-level rates cannot express.
     OmniPath edges are literature-curated, so both the edges and the genes carrying them are
     biased toward well-studied proteins; F3 matches on ortholog count, divergence and variance
     but cannot match on study effort.
     If F2 fails and F4 passes, the correct reading is a real but small effect, not a vindication.
     If both fail, the correct reading is that this proxy carries nothing, and the branch-rate
     version remains untested rather than refuted.
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
OUT = "outputs/loop_coevolution_genome.json"
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
EVO = SP / "compara" / "mammal_identity.npz"
NET = ROOT / "colab" / "data" / "networks"
SEED, MIN_ORTH, K_FACTOR = 226226, 50, 32
REF_STACK, REF_FAME, REF_225 = 0.5474, 0.0940, 0.5117
MINPAIR, AUC_BAR, MATCH_KEEP, DOSE_MIN, LEARN_MARGIN, SHUF_BAR, CTRL_MARGIN = \
    10000, 0.60, 0.70, 0.02, 0.02, 0.55, 0.05

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


def pair_erc(A, O, I, J, chunk=40000, minshare=30):
    """Exact masked Pearson between rows I and J of a residual matrix, vectorised."""
    out = np.empty(len(I))
    for s in range(0, len(I), chunk):
        i, j = I[s:s + chunk], J[s:s + chunk]
        ai, aj, oi, oj = A[i], A[j], O[i], O[j]
        n = np.einsum("ij,ij->i", oi, oj)
        si = np.einsum("ij,ij->i", ai, oj)
        sj = np.einsum("ij,ij->i", aj, oi)
        sii = np.einsum("ij,ij->i", ai * ai, oj)
        sjj = np.einsum("ij,ij->i", aj * aj, oi)
        sij = np.einsum("ij,ij->i", ai, aj)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sij - si * sj / n
            vi = sii - si * si / n
            vj = sjj - sj * sj / n
            r = cov / np.sqrt(vi * vj)
        r[(n < minshare) | ~np.isfinite(r)] = np.nan
        out[s:s + chunk] = r
    return out


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "genome-wide mammalian co-evolution"}
    say("=" * 104)
    say("LOOP 226 -- THE SAME CO-EVOLUTION TEST AT GENOME SCALE")
    say("=" * 104)
    say("     Loop 225 failed its positive control at AUC 0.5117 on 595 pairs. It scored on the")
    say("     1,336-gene A549 responder set when all 18,757 genes were already in the same file.")
    say("     The 0.60 bar is UNCHANGED so the comparison stays interpretable.")

    Z = np.load(EVO, allow_pickle=True)
    Mid, egen, esp = Z["M"], Z["genes"], Z["species"]
    cov_all = np.isfinite(Mid).sum(1)
    keep = cov_all >= MIN_ORTH
    e2s = L191.ensg_to_symbol(lambda *_: None)
    sym2row, row2sym = {}, {}
    for i, g in enumerate(egen):
        if not keep[i]:
            continue
        s = e2s.get(str(g), "")
        if s and s not in sym2row:
            sym2row[s] = i; row2sym[i] = s
    rows = np.array(sorted(sym2row.values()))
    gpos = {row2sym[r]: k for k, r in enumerate(rows)}
    say(f"     {int(keep.sum()):,} of {len(egen):,} genes have >={MIN_ORTH} mammalian orthologs; "
        f"{len(gpos):,} carry a unique symbol")

    R = Mid[rows].astype(np.float64)
    ok = np.isfinite(R)
    D = np.log(np.clip(100.0 - R, 0.5, None))
    D = np.where(ok, D, np.nan)
    D = D - np.nanmean(D, axis=0, keepdims=True)
    D = D - np.nanmean(D, axis=1, keepdims=True)
    A = np.where(ok, D, 0.0); O = ok.astype(np.float64)
    say(f"     doubly-centred residual matrix {A.shape[0]:,} genes x {A.shape[1]} species")

    # ---------------------------------------------------------------- F1
    say("F1 IS THE POSITIVE SET NOW LARGE ENOUGH TO SETTLE THE QUESTION?")
    edges = set()
    with open(NET / "omnipath.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b = r.get("source_genesymbol", ""), r.get("target_genesymbol", "")
            if a in gpos and b in gpos and a != b:
                edges.add((min(a, b), max(a, b)))
    n_omni = len(edges)
    with open(NET / "signor_human.tsv") as f:   # headerless; symbols sit in columns 0 and 4
        for p in csv.reader(f, delimiter="\t"):
            if len(p) > 4 and p[0] in gpos and p[4] in gpos and p[0] != p[4]:
                edges.add((min(p[0], p[4]), max(p[0], p[4])))
    edges = sorted(edges)
    I = np.array([gpos[a] for a, _ in edges]); J = np.array([gpos[b] for _, b in edges])
    say(f"     interacting pairs: {len(edges):,}  (OmniPath {n_omni:,}, SIGNOR adds "
        f"{len(edges)-n_omni:,})")
    say(f"     loop 225 had 595 on 1,246 genes; this is {len(edges)/595:.0f}x more")
    G.add("F1", bool(len(edges) >= MINPAIR), stat=float(len(edges)),
          if_true=lambda: f"F1 PASS -- {len(edges):,} pairs, enough to separate a real absence "
                          f"from an underpowered one",
          if_false=lambda: f"F1 FAIL -- only {len(edges):,} pairs")
    res["pairs"] = {"n": len(edges), "omnipath": n_omni, "loop225": 595,
                    "n_genes": len(gpos)}

    # ---------------------------------------------------------------- F2
    say("F2 DOES CO-EVOLUTION RECOVER KNOWN INTERACTIONS AT GENOME SCALE?")
    posv = pair_erc(A, O, I, J)
    ni = rng.integers(0, len(gpos), size=(len(edges) * 3, 2))
    ni = ni[ni[:, 0] != ni[:, 1]]
    negv = pair_erc(A, O, ni[:, 0], ni[:, 1])
    a2 = auc(posv, negv)
    nboot = 200
    bs = np.array([auc(rng.choice(posv[np.isfinite(posv)], 4000),
                       rng.choice(negv[np.isfinite(negv)], 4000)) for _ in range(nboot)])
    se = float(bs.std())
    say(f"     interacting pairs ERC median {np.nanmedian(posv):+.4f}   "
        f"random pairs {np.nanmedian(negv):+.4f}")
    say(f"     AUC {a2:.4f} over {np.isfinite(posv).sum():,} positives and "
        f"{np.isfinite(negv).sum():,} negatives")
    say(f"     bootstrap standard error {se:.4f}; that is {(a2-0.5)/max(se,1e-9):+.1f} standard "
        f"errors from chance -- SIGNIFICANCE, reported and NOT gated")
    say(f"     loop 225 on 595 pairs measured {REF_225:.4f}")
    G.add("F2", bool(np.isfinite(a2) and a2 >= AUC_BAR), stat=float(a2), requires=("F1",),
          if_true=lambda: f"F2 PASS -- AUC {a2:.4f} at genome scale; loop 225's failure was "
                          f"sample size",
          if_false=lambda: f"F2 FAIL -- AUC {a2:.4f} against the unchanged {AUC_BAR:.2f} bar, on "
                           f"{len(edges):,} pairs rather than 595. Sample size was not the reason")
    res["genome_auc"] = {"auc": a2, "se": se, "pos_med": float(np.nanmedian(posv)),
                         "neg_med": float(np.nanmedian(negv)), "loop225": REF_225}

    # ---------------------------------------------------------------- F3
    say("F3 IS F2 A CONFOUND?")
    northo = ok.sum(1).astype(float)
    meandiv = np.nanmean(np.log(np.clip(100.0 - R, 0.5, None)), axis=1)
    gvar = np.nanstd(D, axis=1)
    F = np.column_stack([northo, meandiv, gvar])
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    nb = 8
    binz = np.zeros(len(gpos), np.int64)
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
    mnegv = pair_erc(A, O, mi, mj)
    a3 = auc(posv, mnegv)
    ex_raw, ex_m = a2 - 0.5, a3 - 0.5
    kept = ex_m / ex_raw if ex_raw > 0 else float("nan")
    say(f"     negatives matched on ortholog count, mean divergence and profile variance, "
        f"{nb}^3 strata; {len(mi):,} matched negatives")
    say(f"     matched AUC {a3:.4f} against unmatched {a2:.4f}; excess {ex_m:+.4f} of "
        f"{ex_raw:+.4f} = {kept:.1%} retained")
    G.add("F3", bool(np.isfinite(kept) and kept >= MATCH_KEEP), stat=float(kept), requires=("F2",),
          if_true=lambda: f"F3 PASS -- {kept:.0%} survives matching",
          if_false=lambda: f"F3 FAIL -- {kept:.0%} survives matching")
    res["matched"] = {"auc": a3, "kept": kept, "n_neg": int(len(mi))}

    # ---------------------------------------------------------------- F4
    say("F4 DOES SEPARATION GROW WITH THE NUMBER OF SHARED SPECIES?")
    share_p = np.einsum("ij,ij->i", O[I], O[J])
    share_n = np.einsum("ij,ij->i", O[mi], O[mj])
    qs = np.quantile(share_p, [0.25, 0.5, 0.75])
    dose = []
    for b in range(4):
        lo = -np.inf if b == 0 else qs[b - 1]
        hi = np.inf if b == 3 else qs[b]
        pm = (share_p > lo) & (share_p <= hi)
        nm = (share_n > lo) & (share_n <= hi)
        ab = auc(posv[pm], mnegv[nm])
        dose.append(ab)
        say(f"       shared species {lo if b else 0:.0f}-{hi if b<3 else 95:.0f}: "
            f"AUC {ab:.4f}   ({int(pm.sum()):,} positives, {int(nm.sum()):,} negatives)")
    dose = np.array(dose)
    mono = bool(np.all(np.diff(dose) > 0)) and (dose[-1] - dose[0] >= DOSE_MIN)
    G.add("F4", mono, stat=float(dose[-1] - dose[0]), requires=("F1",),
          if_true=lambda: f"F4 PASS -- AUC rises monotonically from {dose[0]:.4f} to "
                          f"{dose[-1]:.4f}; a real effect diluted by per-pair noise",
          if_false=lambda: f"F4 FAIL -- quartile AUCs {np.round(dose,4).tolist()}; the pooled "
                           f"result is not a diluted real effect")
    res["dose"] = {"auc_by_quartile": [float(x) for x in dose]}

    # ---------------------------------------------------------------- F5
    say("F5 DOES A LEARNED EMBEDDING BEAT PAIRWISE ERC?")
    Az = A - A.mean(0, keepdims=True)
    U, Sv, Vt = np.linalg.svd(Az, full_matrices=False)
    EF = U[:, :K_FACTOR] * Sv[:K_FACTOR]
    EFn = EF / (np.linalg.norm(EF, axis=1, keepdims=True) + 1e-9)
    a_f = auc(np.einsum("ij,ij->i", EFn[I], EFn[J]), np.einsum("ij,ij->i", EFn[mi], EFn[mj]))
    say(f"     FACTOR rank-{K_FACTOR}, top component {Sv[0]**2/np.sum(Sv**2):.1%} of variance: "
        f"matched AUC {a_f:.4f}")
    a_m, note = float("nan"), ""
    try:
        from sklearn.neural_network import MLPRegressor
        mask = rng.random(Az.shape) < 0.15
        net = MLPRegressor(hidden_layer_sizes=(K_FACTOR,), activation="tanh", max_iter=150,
                           random_state=SEED)
        Xin = np.where(mask, 0.0, Az)
        net.fit(Xin, Az)
        H = np.tanh(Xin @ net.coefs_[0] + net.intercepts_[0])
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        a_m = auc(np.einsum("ij,ij->i", Hn[I], Hn[J]), np.einsum("ij,ij->i", Hn[mi], Hn[mj]))
        note = f"masked autoencoder, {K_FACTOR} units, 15% masked, train loss {net.loss_:.4f}"
    except Exception as e:
        note = f"did not run: {type(e).__name__}: {e}"
    say(f"     MLP matched AUC {a_m:.4f}   ({note})")
    say(f"     ERC matched AUC {a3:.4f}")
    best = max([x for x in (a_f, a_m) if np.isfinite(x)], default=float("nan"))
    G.add("F5", bool(np.isfinite(best) and best - a3 >= LEARN_MARGIN), stat=float(best),
          requires=("F2",),
          if_true=lambda: f"F5 PASS -- learned {best:.4f} against ERC {a3:.4f}",
          if_false=lambda: f"F5 FAIL -- best learned {best:.4f} against ERC {a3:.4f}")
    res["learned"] = {"factor": a_f, "mlp": a_m, "erc": a3,
                      "top_component": float(Sv[0] ** 2 / np.sum(Sv ** 2))}

    # ---------------------------------------------------------------- F6
    say("F6 DOES IT ADD TO THE STANDING STACK?")
    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    y_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    inter = [s for s in allg if s in gpos]
    yv = np.array([y_all[allg.index(s)] for s in inter])
    ridx = np.array([gpos[s] for s in inter])
    say(f"     embeddings trained on {len(gpos):,} genes, scored on the {len(inter):,}-gene "
        f"A549 intersection")
    Cf = EFn @ EFn.T
    np.fill_diagonal(Cf, -np.inf)
    sub = Cf[np.ix_(ridx, ridx)]
    nn = np.argsort(-sub, axis=1)[:, :8]
    blocks = {"coev_factor": EF[ridx],
              "coev_neighbour_y": yv[nn],
              "coev_meta": np.column_stack([northo[ridx], meandiv[ridx], gvar[ridx]])}
    n = len(inter); perm = rng.permutation(n); cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    sc = {}
    for nm, X in blocks.items():
        Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        sc[nm] = abs(pear(yv[te], ridge_pred(Xc[tr], yv[tr], Xc[te])))
        say(f"       {nm:<20} held-out |r| {sc[nm]:.4f}")
    Xall = np.nan_to_num(np.hstack([blocks[k] for k in blocks]), nan=0.0)
    comb = abs(pear(yv[te], ridge_pred(Xall[tr], yv[tr], Xall[te])))
    say(f"       {'all co-evolution':<20} held-out |r| {comb:.4f}")
    say(f"     loop 213's stack {REF_STACK:.4f}; fame-only {REF_FAME:.4f}; loop 225 got 0.0887")
    G.add("F6", bool(comb > REF_STACK), stat=float(comb), requires=("F1",),
          if_true=lambda: f"F6 PASS -- {comb:.4f} above the standing stack's {REF_STACK:.4f}",
          if_false=lambda: f"F6 FAIL -- {comb:.4f} against the standing stack's {REF_STACK:.4f}")
    res["stack"] = dict(sc); res["stack"]["combined"] = comb; res["stack"]["reference"] = REF_STACK

    # ---------------------------------------------------------------- F7
    say("F7 RIDGE AGAINST MLP, SHUFFLED CONTROL ON BOTH")
    ysh = yv.copy(); rng.shuffle(ysh)
    r_r = abs(pear(yv[te], ridge_pred(Xall[tr], yv[tr], Xall[te])))
    r_s = abs(pear(ysh[te], ridge_pred(Xall[tr], ysh[tr], Xall[te])))
    m_r = m_s = float("nan")
    try:
        from sklearn.neural_network import MLPRegressor
        mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-9
        Zt, Ze = (Xall[tr] - mu) / sd, (Xall[te] - mu) / sd
        for tag, yy in (("real", yv), ("shuf", ysh)):
            nn_ = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED,
                               early_stopping=True)
            nn_.fit(Zt, yy[tr])
            v = abs(pear(yy[te], nn_.predict(Ze)))
            if tag == "real": m_r = v
            else: m_s = v
    except Exception as e:
        say(f"     MLP did not run: {type(e).__name__}: {e}")
    say(f"       ridge real {r_r:.4f}  shuffled {r_s:.4f}  margin {r_r-r_s:+.4f}")
    say(f"       MLP   real {m_r:.4f}  shuffled {m_s:.4f}  margin {m_r-m_s:+.4f}")
    say(f"     loop 225 had ridge 0.0887/0.0221 and MLP 0.1222/0.0116; loop 211 had ridge 0.4057 "
        f"and MLP-wide 0.2072")
    cd = {"ridge": (r_r, r_s), "MLP": (m_r, m_s)}
    win = max((k for k in cd if np.isfinite(cd[k][0])), key=lambda k: cd[k][0], default=None)
    marg = (cd[win][0] - cd[win][1]) if win else float("nan")
    G.add("F7", bool(np.isfinite(marg) and marg >= CTRL_MARGIN), stat=float(marg), requires=("F1",),
          if_true=lambda: f"F7 PASS -- {win} wins at {cd[win][0]:.4f}, {marg:+.4f} over its own "
                          f"shuffled control",
          if_false=lambda: f"F7 FAIL -- best margin over shuffled control is only {marg:+.4f}")
    res["ridge_vs_mlp"] = {"ridge_real": r_r, "ridge_shuf": r_s, "mlp_real": m_r,
                           "mlp_shuf": m_s, "winner": win}

    # ---------------------------------------------------------------- F8
    say("F8 SPECIES-SHUFFLE CONTROL")
    As, Os = A.copy(), O.copy()
    for i in range(As.shape[0]):
        p = rng.permutation(As.shape[1])
        As[i] = As[i][p]; Os[i] = Os[i][p]
    a8 = auc(pair_erc(As, Os, I, J), pair_erc(As, Os, mi, mj))
    say(f"     species permuted independently within each gene; marginals preserved exactly")
    say(f"     shuffled AUC {a8:.4f} against the real matched {a3:.4f}")
    G.add("F8", bool(np.isfinite(a8) and a8 < SHUF_BAR), stat=float(a8), requires=("F1",),
          if_true=lambda: f"F8 PASS -- the shuffle drops to {a8:.4f}",
          if_false=lambda: f"F8 FAIL -- the shuffle still reaches {a8:.4f}")
    res["shuffle"] = {"auc": a8}

    # ---------------------------------------------------------------- F9
    say("F9 WHAT THIS CANNOT SHOW")
    say("     A genome-wide FAIL is a statement about protein IDENTITY across 95 mammals, not")
    say("     about co-evolution as such and not about Evo 2, whose nucleotide representation")
    say("     reads intra-genic, promoter and non-coding covariation a gene-level rate cannot.")
    say("     Every dn and ds field in Compara 116 is NULL, which is what forced identity; real")
    say("     ERC correlates BRANCH rates from gene trees, and that version remains untested.")
    say("     OmniPath and SIGNOR edges are literature-curated and biased toward studied genes;")
    say("     F3 matches ortholog count, divergence and variance but cannot match study effort.")

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
