"""Loop 252. The test this whole arc has been unable to run: hold out a CELL LINE and predict its
response to a knockdown.

WHY EVERY EARLIER LOOP FAILED THE SAME WAY. The datasets on disk fall into two useless shapes:

    Perturb-seq K562   11,258 knockdowns  x  ONE cell line   x  full transcriptome
    DepMap             17,916 knockdowns  x  1,178 lines     x  ONE NUMBER (fitness)

Deep in perturbations with no contexts, or deep in contexts with almost no readout. So loops 239
to 251 could hold out GENES, or hold out a cell line and predict a single fitness scalar, but never
hold out a cell line and predict what a knockdown DOES to it. Every negative in that stretch --
signed edges +0.0001, dose +0.047, chromatin +0.033, the five clauses +0.0009 -- was measured on
data that could not answer the question the model was being asked.

LINCS L1000 has both axes. Streamed out of the 21 GB GCTX (see colab/lincs/):

    191,713 shRNA signatures  x  20 cell lines  x  978 MEASURED landmark genes

Only the 978 landmarks are used. The 12,328-gene version is IMPUTED from those 978 by a fitted
model, so a result on the full matrix would partly be a result about Broad's imputation. That
choice costs coverage and buys interpretability, and it is made before any number is computed.

VERIFIED BEFORE ANYTHING WAS BUILT ON IT. Knocking down gene g gives g's OWN landmark measurement
a mean z of -2.2997 (median -2.07) against -0.0058 for random other landmarks in the same rows;
97.1% of the 946 knocked-down genes that are themselves landmarks have own-gene z below zero.

THE TRAP, AND IT IS THE SAME ONE LOOP 240 WALKED INTO. If most of a knockdown's effect is the same
in every cell line, then "predict the held-out line" is answered by "use the average of the other
lines" and a high score means nothing about understanding context. Loop 238 scored 0.9250 that way
and it took loop 240 to establish the number was near-vacuous. So E2 decomposes the variance into
gene, line and gene-by-line BEFORE any model is scored, and the interaction term is what every
later gate is really about.

ARMS, all held out BY CELL LINE -- train on eight lines, predict the ninth.

    A0 ZERO        predict no response. The floor, and not a trivial one: these are z-scores, so
                   zero is the honest null.
    A1 GENEMEAN    the gene's mean response across the eight TRAINING lines. "A knockdown does the
                   same thing everywhere." This is the arm to beat.
    A2 LINEMEAN    the held-out line's own mean response across all genes -- its generic reaction
                   to being perturbed at all, with no gene information.
    A3 ADDITIVE    A1 + A2. No interaction.
    A4 SIMILAR     weight the training lines by how similar their overall response profiles are to
                   the held-out line's, estimated from genes OTHER than the one being predicted.
    A5 RIDGE       gene identity crossed with line identity, fitted linearly.

PREDECLARED, BEFORE ANY NUMBER.

  E1 IS THE BLOCK WHAT IT CLAIMS TO BE?
     The own-gene knockdown check, rerun inside the loop so the number in the record is the loop's.
     Gate: PASS iff the mean own-gene z is below -1.0 while random landmarks in the same rows sit
     within +/-0.2. Everything requires this.

  E2 HOW MUCH OF THE RESPONSE IS ACTUALLY CONTEXT?      -- requires E1
     Variance of the 978-dim response decomposed into a gene main effect, a line main effect and a
     gene-by-line interaction.
     Gate: PASS iff all three components are strictly positive and the decomposition accounts for
     at least 90% of the total. The RATIO is reported, not gated -- but if the interaction is a
     small share, then E3's number is near-vacuous by construction and E4 carries the result. This
     is loop 240's X2 applied where it belongs.

  E3 HOW WELL DOES A KNOCKDOWN TRANSFER TO A CELL LINE NEVER SEEN?      -- requires E1
     A1_GENEMEAN scored on held-out lines, correlation across the 978 landmarks, averaged over
     (gene, held-out line) pairs.
     Gate: PASS iff it exceeds 0.20. This is the number this project has wanted since loop 224
     measured K562-to-RPE1 transfer at 0.2286 on a different quantity.

  E4 DOES KNOWING THE CELL LINE ADD ANYTHING?      -- requires E1. The real test.
     Best of A2, A3, A4, A5 against A1_GENEMEAN, paired over held-out (gene, line) pairs.
     Gate: PASS iff at least 0.02. A FAIL means a knockdown's effect is a property of the gene and
     the cell line is decoration -- which would be the cleanest possible statement of why every
     context-specific model in loops 239-251 failed.

  E5 A NEW GENE IN A NEW CELL LINE.      -- requires E4, VOID if E4 found nothing
     The held-out line's model fitted using only genes that are NOT the one being predicted, so
     both axes are held out at once.
     Gate: PASS iff at least half of E4's advantage survives.

  E6 CONTROL: THE WRONG CELL LINE.      -- requires E4, VOID if E4's margin is under 0.005
     The held-out line's identity swapped for another line's wherever the arm uses it.
     Gate: PASS iff E4's advantage collapses to under 25%.

  E7 WHAT THIS CANNOT SHOW -- written before the run.
     978 genes, not a transcriptome. Landmarks were chosen to span expression space for imputation,
     not to cover biology evenly, so a knockdown whose real effect misses all 978 reads as no
     effect.
     shRNA has seed-based off-target effects. A signature is partly the intended knockdown and
     partly the construct, which inflates apparent gene-specificity and therefore helps A1.
     Nine cancer lines from one screening platform, all grown in one facility. Cross-context here
     means across cancer lines, not across tissues or organisms.
     Level 5 signatures are already z-scored against a plate population, so a response common to
     every perturbation on a plate has been removed. That biases AGAINST A2_LINEMEAN by
     construction, and A2's score must be read with that in mind rather than as a measurement of
     how much generic response exists.
"""
import os, sys, json, time, warnings, collections
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_crosscontext.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
LX = SCR / "lincs"

SEED, KNN = 252252, 4
LINES = ["PC3", "MCF7", "VCAP", "A375", "HA1E", "A549", "HT29", "HEPG2", "HCC515"]
MIN_LINES = 6
E1_BAR, E2_ACC, E3_BAR, E4_BAR, E5_KEEP, E6_MAX = -1.0, 0.90, 0.20, 0.02, 0.50, 0.25

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "hold out a cell line, predict its response to a knockdown"}
    say("=" * 104)
    say("LOOP 252 -- HOLD OUT A CELL LINE, PREDICT WHAT A KNOCKDOWN DOES TO IT")
    say("=" * 104)
    say("     Loops 239-251 could hold out GENES, or hold out a line and predict one fitness")
    say("     number, but never hold out a line and predict what a knockdown DOES to it. Every")
    say("     negative in that stretch was measured on data that could not answer the question.")
    say("     978 MEASURED landmarks only; the 12,328-gene matrix is imputed from these.")

    X = np.load(LX / "shrna_landmark.npy", mmap_mode="r")
    S = np.load(LX / "select.npz", allow_pickle=True)
    gene = np.array([str(x) for x in S["gene"]])
    cell = np.array([str(x) for x in S["cell"]])
    lmids = np.array([str(x) for x in S["lm_gene_ids"]])
    say(f"     {X.shape[0]:,} shRNA signatures x {X.shape[1]} landmarks")

    keep = np.isin(cell, LINES)
    Xk, gk, ck = np.asarray(X[keep]), gene[keep], cell[keep]
    say(f"     restricted to {len(LINES)} well-covered lines: {len(gk):,} signatures")

    # collapse replicate signatures to one profile per (gene, line)
    key = collections.defaultdict(list)
    for i, (g, c) in enumerate(zip(gk, ck)): key[(g, c)].append(i)
    pairs = sorted(key)
    Pm = np.zeros((len(pairs), Xk.shape[1]), np.float32)
    for j, k_ in enumerate(pairs): Pm[j] = Xk[key[k_]].mean(0)
    pg = np.array([p[0] for p in pairs]); pc = np.array([p[1] for p in pairs])
    nl = collections.Counter(pg)
    good = np.array([nl[g] >= MIN_LINES for g in pg])
    Pm, pg, pc = Pm[good], pg[good], pc[good]
    genes = sorted(set(pg.tolist()))
    say(f"     collapsed to {len(pairs):,} (gene, line) profiles; "
        f"{len(genes):,} genes present in >= {MIN_LINES} lines -> {len(pg):,} profiles used")
    res["n_genes"] = len(genes); res["n_profiles"] = int(len(pg))

    # ---------------------------------------------------------------- E1
    say("E1 IS THE BLOCK WHAT IT CLAIMS TO BE?")
    import gzip
    sym = {}
    with gzip.open(LX / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt", errors="replace") as fh:
        h = fh.readline().rstrip("\n").split("\t"); ix = {k: i for i, k in enumerate(h)}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= len(h): sym[p[ix["pr_gene_id"]]] = p[ix["pr_gene_symbol"]]
    lmsym = np.array([sym.get(g, "?") for g in lmids])
    lpos = {s: i for i, s in enumerate(lmsym)}
    own, oth = [], []
    for g in genes:
        if g not in lpos: continue
        m = pg == g
        if m.sum() < 3: continue
        own.append(float(Pm[m, lpos[g]].mean()))
        oth.append(float(Pm[m][:, rng.integers(0, Pm.shape[1], 20)].mean()))
    om, tm = float(np.mean(own)), float(np.mean(oth))
    say(f"     {len(own)} knocked-down genes are themselves landmarks")
    say(f"     own-gene z when knocked down {om:+.4f}; random landmarks, same rows {tm:+.4f}")
    G.add("E1", bool(om <= E1_BAR and abs(tm) <= 0.2), stat=float(om),
          if_true=lambda: f"E1 PASS -- knockdowns lower their own gene by {om:+.3f} while other "
                          f"landmarks sit at {tm:+.3f}",
          if_false=lambda: f"E1 FAIL -- own-gene {om:+.3f}, others {tm:+.3f}")
    res["E1"] = {"own": om, "other": tm, "n": len(own)}

    # ---------------------------------------------------------------- E2
    say("E2 HOW MUCH OF THE RESPONSE IS ACTUALLY CONTEXT?")
    grand = Pm.mean(0)
    gm = {g: Pm[pg == g].mean(0) for g in genes}
    cm = {c: Pm[pc == c].mean(0) for c in LINES}
    Gm = np.stack([gm[g] for g in pg]); Cm = np.stack([cm[c] for c in pc])
    v_g = float(((Gm - grand) ** 2).mean())
    v_c = float(((Cm - grand) ** 2).mean())
    inter = Pm - Gm - Cm + grand
    v_i = float((inter ** 2).mean())
    v_t = float(((Pm - grand) ** 2).mean())
    acc = (v_g + v_c + v_i) / v_t
    say(f"     variance of the 978-dim response, per element:")
    say(f"       gene main effect        {v_g:.4f}   ({v_g / v_t:.1%})")
    say(f"       cell-line main effect   {v_c:.4f}   ({v_c / v_t:.1%})")
    say(f"       gene x line interaction {v_i:.4f}   ({v_i / v_t:.1%})")
    say(f"       total                   {v_t:.4f}   (decomposition accounts for {acc:.1%})")
    say(f"     the RATIO is reported, not gated. If interaction is a small share then E3 is")
    say(f"     near-vacuous by construction and E4 carries the result -- loop 240's X2, here.")
    G.add("E2", bool(v_g > 0 and v_c > 0 and v_i > 0 and acc >= E2_ACC), stat=float(v_i / v_t),
          requires=("E1",),
          if_true=lambda: f"E2 PASS -- all three components positive; interaction is "
                          f"{v_i / v_t:.0%} of the total",
          if_false=lambda: f"E2 FAIL -- decomposition accounts for {acc:.1%} or a component is "
                           f"not positive")
    res["E2"] = {"gene": v_g, "line": v_c, "interaction": v_i, "total": v_t,
                 "interaction_share": v_i / v_t, "accounted": acc}

    # ---------------------------------------------------------------- arms
    say("     scoring, leave-one-cell-line-out ...")
    gi = {g: i for i, g in enumerate(genes)}
    ARMS = ["A0_ZERO", "A1_GENEMEAN", "A2_LINEMEAN", "A3_ADDITIVE", "A4_SIMILAR", "A5_RIDGE"]

    def evaluate(shuffle_line=False, gene_holdout=False):
        sc = {a: [] for a in ARMS}
        for held in LINES:
            te = pc == held
            tr = ~te
            if te.sum() < 50: continue
            src = held
            if shuffle_line:
                src = str(rng.choice([l for l in LINES if l != held]))
            gmean_tr = {}
            for g in genes:
                m = tr & (pg == g)
                if m.sum(): gmean_tr[g] = Pm[m].mean(0)
            # the source line's generic response, and its similarity to each training line
            srcmask = pc == src
            lmean_src = Pm[srcmask].mean(0)
            simw = {}
            for l in LINES:
                if l == held: continue
                simw[l] = pear(Pm[pc == l].mean(0), lmean_src)
            # A5's two mixing coefficients are fitted on the TRAINING lines only. Fitting them
            # against the held-out profile -- which the first draft of this loop did -- is
            # leakage: it lets the arm see the answer it is being scored on.
            da, db = [], []
            for l in LINES:
                if l == held: continue
                lmv_l = Pm[pc == l].mean(0)
                for j2 in np.where((pc == l))[0][:400]:
                    g2 = pg[j2]
                    m2 = tr & (pg == g2) & (pc != l)
                    if not m2.sum(): continue
                    da.append(np.stack([Pm[m2].mean(0), lmv_l], 1)); db.append(Pm[j2])
            if da:
                Ad = np.concatenate(da, 0); bd = np.concatenate(db, 0)
                beta5 = np.linalg.lstsq(Ad, bd, rcond=None)[0]
            else:
                beta5 = np.array([1.0, 0.0])
            for j in np.where(te)[0]:
                g = pg[j]
                truth = Pm[j]
                if g not in gmean_tr: continue
                gmv = gmean_tr[g]
                if gene_holdout:
                    # rebuild the line-similarity weights WITHOUT this gene
                    o = (pg != g)
                    lm2 = Pm[srcmask & o].mean(0)
                    sw = {l: pear(Pm[(pc == l) & o].mean(0), lm2) for l in LINES if l != held}
                    lmv = lm2
                else:
                    sw, lmv = simw, lmean_src
                # a constant prediction has undefined correlation, so A0 enters as 0 by
                # definition rather than as a number produced by dividing by zero
                sc["A0_ZERO"].append(0.0)
                sc["A1_GENEMEAN"].append(pear(gmv, truth))
                sc["A2_LINEMEAN"].append(pear(lmv, truth))
                sc["A3_ADDITIVE"].append(pear(gmv + lmv, truth))
                ls = sorted(sw, key=lambda l: -sw[l])[:KNN]
                w = np.array([max(sw[l], 0) for l in ls])
                w = w / w.sum() if w.sum() > 0 else np.ones(len(ls)) / len(ls)
                acc_ = np.zeros_like(truth)
                for wi, l in zip(w, ls):
                    m = (pc == l) & (pg == g)
                    if m.sum(): acc_ = acc_ + wi * Pm[m].mean(0)
                sc["A4_SIMILAR"].append(pear(acc_, truth))
                sc["A5_RIDGE"].append(pear(np.stack([gmv, lmv], 1) @ beta5, truth))
        return {a: np.asarray(v) for a, v in sc.items()}

    R = evaluate()
    for a in ARMS:
        say(f"       {a:<13} {np.nanmean(R[a]):+.4f}  (sd {np.nanstd(R[a]):.4f}, "
            f"n={int(np.isfinite(R[a]).sum()):,})")
    res["arms"] = {a: float(np.nanmean(R[a])) for a in ARMS}

    # ---------------------------------------------------------------- E3
    say("E3 HOW WELL DOES A KNOCKDOWN TRANSFER TO A CELL LINE NEVER SEEN?")
    r3 = float(np.nanmean(R["A1_GENEMEAN"]))
    say(f"     A1_GENEMEAN on held-out lines: {r3:.4f}")
    say(f"     loop 224 measured K562-to-RPE1 transfer at 0.2286 on a different quantity")
    G.add("E3", bool(r3 >= E3_BAR), stat=float(r3), requires=("E1",),
          if_true=lambda: f"E3 PASS -- a knockdown's effect transfers to an unseen cell line at "
                          f"{r3:.4f}",
          if_false=lambda: f"E3 FAIL -- transfer is {r3:.4f} against a {E3_BAR} bar")
    res["E3"] = {"genemean": r3}

    # ---------------------------------------------------------------- E4
    say("E4 DOES KNOWING THE CELL LINE ADD ANYTHING?")
    cand = ["A2_LINEMEAN", "A3_ADDITIVE", "A4_SIMILAR", "A5_RIDGE"]
    best = max(cand, key=lambda a: np.nanmean(R[a]))
    d4, se4, z4 = paired(R[best], R["A1_GENEMEAN"])
    say(f"     best line-aware arm {best} {np.nanmean(R[best]):+.4f} vs A1_GENEMEAN {r3:+.4f}")
    say(f"     paired over {int(np.isfinite(R[best]).sum()):,} held-out (gene, line) pairs: "
        f"{d4:+.4f} +/- {se4:.4f}  ({z4:+.1f} se)")
    G.add("E4", bool(d4 >= E4_BAR), stat=float(d4), requires=("E1",),
          if_true=lambda: f"E4 PASS -- knowing the cell line adds {d4:+.4f}",
          if_false=lambda: f"E4 FAIL -- knowing the cell line adds {d4:+.4f} against a {E4_BAR} "
                           f"bar; a knockdown's effect is a property of the GENE and the line is "
                           f"decoration")
    res["E4"] = {"best": best, "delta": d4, "se": se4, "z": z4}

    # ---------------------------------------------------------------- E5
    say("E5 A NEW GENE IN A NEW CELL LINE")
    if d4 < E4_BAR:
        G.add("E5", False, stat=float(d4), requires=("E4",), void_if=True,
              void_reason=f"E4 found {d4:+.4f}; there is no advantage to carry to a double holdout")
    else:
        H = evaluate(gene_holdout=True)
        d5, se5, _ = paired(H[best], H["A1_GENEMEAN"])
        kf = d5 / d4 if abs(d4) > 1e-9 else float("nan")
        say(f"     line-similarity rebuilt without the scored gene: {d5:+.4f} against {d4:+.4f} "
            f"({kf:.0%} retained)")
        G.add("E5", bool(np.isfinite(kf) and kf >= E5_KEEP), stat=float(kf), requires=("E4",),
              if_true=lambda: f"E5 PASS -- {kf:.0%} survives with both axes held out",
              if_false=lambda: f"E5 FAIL -- only {kf:.0%} survives")
        res["E5"] = {"single": d4, "double": d5, "retained": kf}

    # ---------------------------------------------------------------- E6
    say("E6 CONTROL: THE WRONG CELL LINE")
    if d4 < 0.005:
        G.add("E6", False, stat=float(d4), requires=("E4",), void_if=True,
              void_reason=f"E4's margin is {d4:+.4f}; there is nothing to collapse")
    else:
        Sh = evaluate(shuffle_line=True)
        ds, _, _ = paired(Sh[best], Sh["A1_GENEMEAN"])
        f6 = ds / d4
        say(f"     held-out line's identity swapped: {ds:+.4f} against a real {d4:+.4f} ({f6:.0%})")
        G.add("E6", bool(f6 <= E6_MAX), stat=float(f6), requires=("E4",),
              if_true=lambda: f"E6 PASS -- collapses to {f6:.0%} with the wrong line",
              if_false=lambda: f"E6 FAIL -- {f6:.0%} survives with the wrong line's identity")
        res["E6"] = {"real": d4, "shuffled": ds, "fraction": f6}

    say("E7 WHAT THIS CANNOT SHOW")
    say("     978 genes, not a transcriptome. Landmarks were chosen to span expression space for")
    say("     imputation, not to cover biology evenly, so a knockdown whose real effect misses")
    say("     all 978 reads as no effect.")
    say("     shRNA carries seed-based off-target effects: a signature is partly the intended")
    say("     knockdown and partly the construct, which inflates gene-specificity and helps A1.")
    say("     Nine cancer lines from one platform in one facility. Cross-context here means")
    say("     across cancer lines, not across tissues or organisms.")
    say("     Level 5 is z-scored against a plate population, so a response common to every")
    say("     perturbation on a plate is already removed. That biases AGAINST A2_LINEMEAN by")
    say("     construction and its score must be read with that in mind.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
