"""Loop 246. State-dependent noise, against the counting statistics that produce it for free.

THE PROPOSAL. Replace additive constant noise

    dS = F(S,G,E,t) dt + sigma dW

with state-dependent noise plus a jump term

    dS = F(S,G,E,M) dt + Sigma(S,G,E,M) dW + J(S) dN

so that two genetically identical cells given the same perturbation diverge because their noise
distributions differ, and rare discrete events -- transcriptional bursts -- enter as jumps rather
than as Gaussian tails.

THE TRAP THIS LOOP IS BUILT AROUND, AND IT IS FATAL IF IGNORED. In count data, variance is ALREADY
a function of the mean before any biology: a Poisson variable has Var = mu exactly. So "noise
depends on state" is TRUE BY CONSTRUCTION of the measurement and would pass any naive test with a
large margin while saying nothing about cells. Loop 87's C6 and loop 94's N4 are the same shape --
a statistic that could not have come out otherwise. The claim only becomes biological if Sigma
depends on state BEYOND what counting requires, so every gate here is stated against the
counting-statistics null and never against zero.

The standard decomposition makes that precise. For counts with cell size factors s_c,

    Var(gene, condition)  =  mu  +  phi * mu^2

where mu is the Poisson part -- pure measurement -- and phi is the overdispersion, the part that a
purely technical process cannot produce. phi is the estimand of Sigma. Every gate below is about
phi, not about Var.

THE DATA. sci-Plex A549 single cells: 24,262 cells, four compounds at seven doses plus each drug's
own vehicle, raw integer counts. This is the only single-cell perturbation matrix on disk, and it
carries a dose axis, so E in Sigma(S,G,E,M) is a real variable here rather than a label. The 529
cells labelled perturbation == "control" are EXCLUDED: loop 244 established they are 100 uM
vorinostat from a single well (top_oligo SAHA_100_E09) mislabelled during harmonisation.

M is not estimated here. The memory term needs a timecourse and sci-Plex is one timepoint; the
dexamethasone WITHDRAWAL series (GSE144662, twelve timepoints from 0 to 12 h in triplicate) is the
right instrument for it and is the next loop, not this one. Saying so in advance stops this loop
from claiming to have tested the full equation.

PREDECLARED, BEFORE ANY NUMBER.

  V1 HOW MUCH OF THE VARIANCE IS COUNTING, AND HOW MUCH IS LEFT?
     The Poisson part against the overdispersed part, pooled over genes and conditions.
     Gate: PASS iff a median overdispersion phi > 0 survives with at least 1,000 (gene, condition)
     cells having phi estimable and strictly positive. A FAIL means the data are Poisson and there
     is no Sigma to model. Everything requires this.

  V2 DOES SIGMA DEPEND ON STATE, BEYOND WHAT COUNTING REQUIRES?      -- requires V1
     log phi predicted from state -- the gene's mean in that condition, its detection rate, the
     drug and the log dose -- held out BY GENE so no gene appears in both fitting and scoring.
     Gate: PASS iff the held-out correlation exceeds 0.20.

  V3 THE SHARP FORM: SAME GENE, SAME MEAN, DIFFERENT CONDITION -- DIFFERENT NOISE?    -- requires V1
     The proposal's actual content is that Sigma varies with the CONDITION, not merely across
     genes. A gene whose mean is unchanged between two conditions but whose phi differs is
     state-dependent noise with nothing else that could explain it. Pairs of conditions are matched
     so the gene's mean differs by under 10%, and the phi difference is compared against the
     difference expected from estimation noise alone, obtained by splitting the cells of ONE
     condition in half.
     Gate: PASS iff the across-condition phi difference exceeds the within-condition split-half
     difference by at least 3 standard errors.

  V4 IS THERE A JUMP COMPONENT BEYOND A CONTINUOUS MODEL?      -- requires V1
     Bursting predicts more probability in the far tail than a negative binomial with the same
     mean and dispersion allows, and it predicts the excess to be POSITIVE. The sign is stated
     here so the gate cannot be satisfied by a deviation in either direction.
     Gate: PASS iff the observed 99th percentile exceeds the fitted negative binomial's 99th
     percentile in at least 60% of well-measured (gene, condition) cells, and the median excess
     is positive.

  V5 CONTROL: CELLS REASSIGNED TO THE WRONG CONDITIONS.      -- requires V2, VOID if V2 is under 0.05
     Cell-to-condition labels permuted, every other step identical. This destroys any real
     condition-specific dispersion while leaving the count distribution, the depth distribution
     and the gene means almost unchanged.
     Gate: PASS iff V2's held-out correlation collapses to under 25% of its true value.

  V6 WHAT THIS CANNOT SHOW -- written before the run.
     One cell line, one timepoint, four compounds. Nothing here separates dose response from time
     response, and M is not estimated at all.
     Overdispersion is not the same thing as bursting. Doublets, ambient RNA, cell-cycle phase and
     unmodelled cell-type substructure all inflate phi, and none of them is Sigma in the sense the
     proposal means. V3 controls for the gene and the mean, not for cell-cycle composition.
     Method-of-moments phi is biased at low mean, which is why a mean floor is applied and stated.
     Nutlin at 100 uM has 86 cells and BMS 129; a dispersion estimated from that few cells is
     poor, and those conditions are excluded by the cell-count floor rather than silently kept.
     A jump process is one explanation of a heavy tail. Zero-inflation and a mixture of two cell
     states would also produce V4's signature, and this loop cannot separate them.
"""
import os, sys, json, time, collections, warnings
from pathlib import Path
import numpy as np
from scipy import sparse, stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_state_noise.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SP = SCR / "sciplex2.h5ad"

SEED, NFOLD = 246246, 10
MIN_CELLS_COND, MIN_CELLS_GENE, MIN_MEAN = 200, 300, 0.05
V1_MIN, V2_BAR, V3_SE, V4_FRAC, V5_MAX = 1000, 0.20, 3.0, 0.60, 0.25

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


def paired(d):
    d = np.asarray([x for x in np.ravel(d) if np.isfinite(x)], float)
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def dispersion(C, s):
    """Method-of-moments overdispersion per gene for one condition.

    Counts C (cells x genes), size factors s. Under Var = mu + phi*mu^2 with Poisson sampling on
    top of a gamma-distributed rate, phi is what a purely technical process CANNOT produce."""
    r = C.multiply(1.0 / s[:, None]).toarray() if sparse.issparse(C) else C / s[:, None]
    mu = r.mean(0)
    v = r.var(0, ddof=1)
    pois = mu * float(np.mean(1.0 / s))
    phi = (v - pois) / np.maximum(mu ** 2, 1e-12)
    return mu, v, pois, phi


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "state-dependent noise on sci-Plex single cells, against counting statistics"}
    say("=" * 104)
    say("LOOP 246 -- STATE-DEPENDENT NOISE, AGAINST THE COUNTING STATISTICS THAT GIVE IT FOR FREE")
    say("=" * 104)
    say("     In count data Var = mu holds for a Poisson variable before any biology, so 'noise")
    say("     depends on state' is true by construction of the measurement. Every gate here is")
    say("     about the OVERDISPERSION phi in Var = mu + phi*mu^2 -- the part counting cannot")
    say("     produce -- and never about Var itself.")
    say("     M is not estimated here: memory needs a timecourse and sci-Plex is one timepoint.")
    say("     The dexamethasone WITHDRAWAL series is the instrument for it, and is the next loop.")

    import h5py
    h = h5py.File(SP, "r")
    def dec(x): return x.decode() if isinstance(x, bytes) else str(x)
    def ocol(n):
        ob = h["obs"][n]
        if isinstance(ob, h5py.Group):
            c = np.array([dec(x) for x in ob["categories"][:]]); return c[ob["codes"][:]]
        a = ob[:]
        if "__categories" in h["obs"] and n in h["obs"]["__categories"]:
            c = np.array([dec(x) for x in h["obs"]["__categories"][n][:]]); return c[a]
        return a
    drug = ocol("perturbation"); dose = ocol("dose_value").astype(float)
    vb = h["var"]["gene_symbol"]
    sym = (np.array([dec(x) for x in vb["categories"][:]])[vb["codes"][:]]
           if isinstance(vb, h5py.Group) else np.array([dec(x) for x in vb[:]]))
    shape = tuple(int(x) for x in h["X"].attrs["shape"])
    X = sparse.csr_matrix((h["X"]["data"][:], h["X"]["indices"][:], h["X"]["indptr"][:]),
                          shape=shape)
    h.close()
    keep = drug != "control"
    say(f"     excluded {int((~keep).sum())} cells labelled 'control' -- loop 244 established they")
    say(f"     are 100 uM vorinostat from one well (top_oligo SAHA_100_E09)")
    X, drug, dose = X[keep], drug[keep], dose[keep]
    tot = np.asarray(X.sum(1)).ravel()
    good = tot > 0
    X, drug, dose, tot = X[good], drug[good], dose[good], tot[good]
    sfac = tot / np.median(tot)
    ncell_g = np.asarray((X > 0).sum(0)).ravel()
    gk = np.where(ncell_g >= MIN_CELLS_GENE)[0]
    X = X[:, gk]; sym = sym[gk]
    say(f"     {X.shape[0]:,} cells x {X.shape[1]:,} genes detected in {MIN_CELLS_GENE}+ cells")

    conds = [(d, v) for d in sorted(set(drug)) for v in sorted(set(dose))
             if ((drug == d) & (dose == v)).sum() >= MIN_CELLS_COND]
    say(f"     {len(conds)} conditions with {MIN_CELLS_COND}+ cells")

    PHI, MU, POIS, VAR, DET = {}, {}, {}, {}, {}
    for (d, v) in conds:
        m = (drug == d) & (dose == v)
        mu, vr, po, ph = dispersion(X[m], sfac[m])
        PHI[(d, v)] = ph; MU[(d, v)] = mu; POIS[(d, v)] = po; VAR[(d, v)] = vr
        DET[(d, v)] = np.asarray((X[m] > 0).mean(0)).ravel()

    # ---------------------------------------------------------------- V1
    say("V1 HOW MUCH OF THE VARIANCE IS COUNTING, AND HOW MUCH IS LEFT?")
    allmu = np.concatenate([MU[c] for c in conds])
    allph = np.concatenate([PHI[c] for c in conds])
    allvar = np.concatenate([VAR[c] for c in conds])
    allpo = np.concatenate([POIS[c] for c in conds])
    est = np.isfinite(allph) & (allmu >= MIN_MEAN)
    npos = int((est & (allph > 0)).sum())
    frac_pois = float(np.median(allpo[est] / np.maximum(allvar[est], 1e-12)))
    say(f"     {int(est.sum()):,} (gene, condition) cells with mean >= {MIN_MEAN} and phi estimable")
    say(f"     median Poisson share of the total variance: {frac_pois:.1%}")
    say(f"     median overdispersion phi: {np.median(allph[est]):.4f}   "
        f"strictly positive in {npos:,}")
    G.add("V1", bool(npos >= V1_MIN and np.median(allph[est]) > 0), stat=float(npos),
          if_true=lambda: f"V1 PASS -- {npos:,} cells carry positive overdispersion; counting "
                          f"explains {frac_pois:.0%} of the variance and there is a Sigma left over",
          if_false=lambda: f"V1 FAIL -- only {npos:,} positive; the data are consistent with pure "
                           f"counting and there is no Sigma to model")
    res["V1"] = {"n_estimable": int(est.sum()), "n_positive": npos,
                 "median_phi": float(np.median(allph[est])), "poisson_share": frac_pois}

    # ---------------------------------------------------------------- V2
    say("V2 DOES SIGMA DEPEND ON STATE, BEYOND WHAT COUNTING REQUIRES?")
    dl = sorted(set(drug)); rows, ys, gidx = [], [], []
    for (d, v) in conds:
        ph, mu, dt = PHI[(d, v)], MU[(d, v)], DET[(d, v)]
        m = np.isfinite(ph) & (ph > 0) & (mu >= MIN_MEAN)
        oh = np.zeros(len(dl)); oh[dl.index(d)] = 1.0
        for g in np.where(m)[0]:
            rows.append([np.log(mu[g]), dt[g], np.log10(v + 0.01)] + list(oh))
            ys.append(np.log(ph[g])); gidx.append(g)
    F = np.asarray(rows); y = np.asarray(ys); gidx = np.asarray(gidx)
    say(f"     {len(y):,} (gene, condition) observations, {F.shape[1]} state features")
    ug = np.unique(gidx); rng.shuffle(ug)
    gf = {g: i % NFOLD for i, g in enumerate(ug)}
    fold_of = np.array([gf[g] for g in gidx])
    pred = np.full(len(y), np.nan)
    for k in range(NFOLD):
        tr, te = fold_of != k, fold_of == k
        Z = F[tr]; mu_, sd_ = Z.mean(0), Z.std(0) + 1e-9
        Zs = np.concatenate([(Z - mu_) / sd_, np.ones((tr.sum(), 1))], 1)
        A = Zs.T @ Zs + 1e-2 * tr.sum() * np.eye(Zs.shape[1])
        b = np.linalg.solve(A, Zs.T @ y[tr])
        pred[te] = np.concatenate([(F[te] - mu_) / sd_, np.ones((te.sum(), 1))], 1) @ b
    r2 = float(np.mean([pear(pred[fold_of == k], y[fold_of == k]) for k in range(NFOLD)]))
    say(f"     log phi from state, held out BY GENE, {NFOLD} folds: r = {r2:.4f}")
    G.add("V2", bool(r2 >= V2_BAR), stat=float(r2), requires=("V1",),
          if_true=lambda: f"V2 PASS -- state predicts the overdispersion at {r2:.4f}",
          if_false=lambda: f"V2 FAIL -- {r2:.4f} against a {V2_BAR} bar")
    res["V2"] = {"r": r2, "n_obs": int(len(y))}

    # ---------------------------------------------------------------- V3
    say("V3 SAME GENE, SAME MEAN, DIFFERENT CONDITION -- DIFFERENT NOISE?")
    say("     the proposal's actual content: Sigma varies with the CONDITION, not just the gene.")
    diffs = []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            a, b_ = conds[i], conds[j]
            pa, pb = PHI[a], PHI[b_]; ma, mb = MU[a], MU[b_]
            m = (np.isfinite(pa) & np.isfinite(pb) & (pa > 0) & (pb > 0)
                 & (ma >= MIN_MEAN) & (mb >= MIN_MEAN)
                 & (np.abs(np.log(ma / np.maximum(mb, 1e-12))) < np.log(1.10)))
            if m.sum() < 20: continue
            diffs.append(np.abs(np.log(pa[m] / pb[m])))
    across = np.concatenate(diffs) if diffs else np.array([])
    within = []
    for (d, v) in conds:
        m = (drug == d) & (dose == v)
        idx = np.where(m)[0]; rng.shuffle(idx)
        h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
        _, _, _, p1 = dispersion(X[h1], sfac[h1])
        _, _, _, p2 = dispersion(X[h2], sfac[h2])
        mm = np.isfinite(p1) & np.isfinite(p2) & (p1 > 0) & (p2 > 0) & (MU[(d, v)] >= MIN_MEAN)
        if mm.sum(): within.append(np.abs(np.log(p1[mm] / p2[mm])))
    within = np.concatenate(within) if within else np.array([])
    say(f"     |log phi ratio| across conditions at matched mean (within 10%): "
        f"{np.median(across):.4f}  (n={len(across):,})")
    say(f"     the same statistic from splitting ONE condition's cells in half: "
        f"{np.median(within):.4f}  (n={len(within):,})")
    if len(across) > 10 and len(within) > 10:
        sa = across.std(ddof=1) / np.sqrt(len(across)); sw = within.std(ddof=1) / np.sqrt(len(within))
        d3 = float(np.median(across) - np.median(within))
        z3 = d3 / np.sqrt(sa ** 2 + sw ** 2)
        say(f"     difference {d3:+.4f}  ({z3:+.1f} se)")
        G.add("V3", bool(d3 > 0 and z3 >= V3_SE), stat=float(d3), requires=("V1",),
              if_true=lambda: f"V3 PASS -- noise differs between conditions by {d3:+.4f} more than "
                              f"estimation noise alone ({z3:+.1f} se)",
              if_false=lambda: f"V3 FAIL -- {d3:+.4f} ({z3:+.1f} se); the apparent condition "
                               f"dependence is within what splitting one condition already gives")
        res["V3"] = {"across": float(np.median(across)), "within": float(np.median(within)),
                     "delta": d3, "z": float(z3)}
    else:
        G.add("V3", False, stat=float("nan"), requires=("V1",), void_if=True,
              void_reason="too few matched-mean pairs to compare")

    # ---------------------------------------------------------------- V4
    say("V4 IS THERE A JUMP COMPONENT BEYOND A CONTINUOUS MODEL?")
    say("     bursting predicts the observed far tail to EXCEED a negative binomial with the same")
    say("     mean and dispersion. The sign is stated before the number.")
    exc, nn = [], 0
    for (d, v) in conds:
        m = np.where((drug == d) & (dose == v))[0]
        Cc = X[m].toarray()
        mu = Cc.mean(0); vr = Cc.var(0, ddof=1)
        ok = (mu >= 1.0) & (vr > mu)
        if ok.sum() == 0: continue
        phi = (vr[ok] - mu[ok]) / np.maximum(mu[ok] ** 2, 1e-12)
        n_r = 1.0 / np.maximum(phi, 1e-6)
        p_r = n_r / (n_r + mu[ok])
        q_nb = stats.nbinom.ppf(0.99, n_r, p_r)
        q_ob = np.percentile(Cc[:, ok], 99, axis=0)
        exc.append((q_ob - q_nb) / np.maximum(q_nb, 1.0)); nn += int(ok.sum())
    exc = np.concatenate(exc) if exc else np.array([])
    frac = float(np.mean(exc > 0)) if len(exc) else float("nan")
    say(f"     {nn:,} well-measured (gene, condition) cells; observed 99th percentile exceeds the")
    say(f"     negative binomial's in {frac:.1%}, median relative excess {np.median(exc):+.4f}")
    G.add("V4", bool(np.isfinite(frac) and frac >= V4_FRAC and np.median(exc) > 0),
          stat=float(frac), requires=("V1",),
          if_true=lambda: f"V4 PASS -- the far tail exceeds a negative binomial in {frac:.0%} of "
                          f"cells, median excess {np.median(exc):+.3f}",
          if_false=lambda: f"V4 FAIL -- {frac:.0%} exceed (bar {V4_FRAC:.0%}), median excess "
                           f"{np.median(exc):+.3f}")
    res["V4"] = {"fraction_exceeding": frac, "median_excess": float(np.median(exc)) if len(exc) else None,
                 "n": nn}

    # ---------------------------------------------------------------- V5
    say("V5 CONTROL: CELLS REASSIGNED TO THE WRONG CONDITIONS")
    if r2 < 0.05:
        G.add("V5", False, stat=float(r2), requires=("V2",), void_if=True,
              void_reason=f"V2 is {r2:.4f}; there is nothing to collapse")
    else:
        pdrug = drug.copy(); pdose = dose.copy()
        pi = rng.permutation(len(pdrug)); pdrug, pdose = pdrug[pi], pdose[pi]
        rows2, ys2, g2 = [], [], []
        for (d, v) in conds:
            m = (pdrug == d) & (pdose == v)
            if m.sum() < MIN_CELLS_COND: continue
            mu, vr, po, ph = dispersion(X[m], sfac[m])
            dt = np.asarray((X[m] > 0).mean(0)).ravel()
            oh = np.zeros(len(dl)); oh[dl.index(d)] = 1.0
            sel = np.isfinite(ph) & (ph > 0) & (mu >= MIN_MEAN)
            for g in np.where(sel)[0]:
                rows2.append([np.log(mu[g]), dt[g], np.log10(v + 0.01)] + list(oh))
                ys2.append(np.log(ph[g])); g2.append(g)
        F2, y2, g2 = np.asarray(rows2), np.asarray(ys2), np.asarray(g2)
        f2 = np.array([gf.get(g, 0) for g in g2])
        pr2 = np.full(len(y2), np.nan)
        for k in range(NFOLD):
            tr, te = f2 != k, f2 == k
            if tr.sum() < 50 or te.sum() < 5: continue
            Z = F2[tr]; mu_, sd_ = Z.mean(0), Z.std(0) + 1e-9
            Zs = np.concatenate([(Z - mu_) / sd_, np.ones((tr.sum(), 1))], 1)
            A = Zs.T @ Zs + 1e-2 * tr.sum() * np.eye(Zs.shape[1])
            b = np.linalg.solve(A, Zs.T @ y2[tr])
            pr2[te] = np.concatenate([(F2[te] - mu_) / sd_, np.ones((te.sum(), 1))], 1) @ b
        rs = float(np.nanmean([pear(pr2[f2 == k], y2[f2 == k]) for k in range(NFOLD)]))
        f5 = rs / r2
        say(f"     cell-to-condition labels permuted: r = {rs:.4f} against a real {r2:.4f} "
            f"({f5:.0%})")
        G.add("V5", bool(f5 <= V5_MAX), stat=float(f5), requires=("V2",),
              if_true=lambda: f"V5 PASS -- collapses to {f5:.0%} on permuted conditions",
              if_false=lambda: f"V5 FAIL -- {f5:.0%} survives; V2 is reading the gene, not the "
                               f"condition")
        res["V5"] = {"real": r2, "shuffled": rs, "fraction": f5}

    say("V6 WHAT THIS CANNOT SHOW")
    say("     One cell line, one timepoint, four compounds; M is not estimated at all.")
    say("     Overdispersion is not bursting. Doublets, ambient RNA, cell-cycle phase and")
    say("     unmodelled substructure all inflate phi and none of them is Sigma in the intended")
    say("     sense. V3 controls for the gene and the mean, not for cell-cycle composition.")
    say("     Method-of-moments phi is biased at low mean, hence the stated mean floor.")
    say("     A jump process is ONE explanation of a heavy tail. Zero-inflation and a two-state")
    say("     mixture produce the same V4 signature and this loop cannot separate them.")

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
