"""Loop 244. Dose as a continuous parameter, against the binary perturbation every loop assumed.

WHAT THIS PROJECT HAS ASSUMED UNTIL NOW. Every model in loops 206-243 treats a perturbation as
present or absent. sci-Plex is the only dataset here with a graded intervention axis: A549, four
compounds -- BMS-345541, dexamethasone, nutlin-3a, vorinostat (SAHA) -- at seven doses from 0.1 to
100 micromolar, each with its own vehicle. A binary model cannot express a saturating response,
and saturation is what a receptor does.

    T_p(d)  =  d^eta / (K^eta + d^eta)  *  v_p

with v_p the per-gene response amplitude and (K, eta) the sensitivity constants. K is the EC50 and
eta the Hill coefficient.

THE LABELLING TRAP, FOUND BEFORE ANY MODEL WAS FITTED AND RECORDED HERE BECAUSE IT WOULD HAVE
RUINED EVERY NUMBER SILENTLY. The harmonised file carries perturbation == "control" for 529 cells.
They are not vehicle. All 529 come from a SINGLE well, and their top_oligo is `SAHA_100_E09` --
they are 100 micromolar vorinostat mislabelled as control. Subtracting them as a baseline would
have subtracted a maximal drug response from every condition, including from vorinostat itself.
The real vehicle is each drug's own dose == 0 cells, which are spread across many wells
(Dex_0_AD12, Dex_0_BD12, Dex_0_AD04 ...). This loop uses the per-drug dose-0 cells as the baseline
and excludes the mislabelled group entirely.

    cells per (drug, dose)     0     0.1    0.5      1      5     10     50    100
        BMS                  980     930    739    512    452    298    143    129
        Dex                  914     918    845   1051   1011    898   1131   1296
        Nutlin              1002     849    792    892    829   1096    410     86
        SAHA                1005     782    718    664    536    540    749    536

THE TEST THAT MAKES A DOSE MODEL EARN ITS PARAMETERS: HOLD OUT A DOSE. A Hill curve has three free
parameters per gene against one for a binary model, and more parameters always fit better in
sample. So every arm is fitted on six doses and scored on the seventh, which is also the first
time this project has held out a CONDITION on a graded axis rather than a gene or a cell line.
Interior doses are interpolation and the two ends are extrapolation; they are reported separately
because they are different claims.

FOUR ARMS.
    D0 BLIND        the mean response over the training doses. Dose-blind, which is what every
                    earlier loop assumed. The floor.
    D1 LOGLIN       response linear in log10(dose): two parameters per gene.
    D2 HILL_SHARED  K and eta shared across all genes within a drug, amplitude per gene.
    D3 HILL_FULL    K, eta and amplitude all per gene.
D1 is the honest twin: a Hill curve must beat the SIMPLEST graded model, not merely the binary one,
or its extra parameters have bought nothing. This is loop 241's W3 in a different costume, and it
is here for the same reason.

PREDECLARED, BEFORE ANY NUMBER.

  S1 IS THE DOSE AXIS REAL, WITHOUT ASSUMING IT IS MONOTONE?
     Loop 234's R3 failed because it gated for a monotone dose increase across a range where the
     response was saturating -- it gated against its own hypothesis. Under a Hill model saturation
     is PREDICTED, so monotonicity is the wrong requirement.
     Gate: PASS iff, in at least 3 of the 4 drugs, the Spearman correlation between dose and the
     magnitude of the response is positive. Everything else requires this.

  S2 DOES A GRADED MODEL BEAT A DOSE-BLIND ONE?      -- requires S1
     Best graded arm against D0, on the held-out dose, paired over (drug, dose) units.
     Gate: PASS iff it exceeds D0 by at least 0.05.

  S3 DOES THE HILL CURVE BEAT A STRAIGHT LINE IN LOG DOSE?      -- requires S1
     The better Hill arm against D1 on the held-out dose.
     Gate: PASS iff it exceeds D1 by at least 0.02. A FAIL means the saturating FORM bought
     nothing over the simplest graded model and the gain in S2 is just "dose matters at all".

  S4 CAN IT EXTRAPOLATE PAST THE DOSES IT SAW?      -- requires S1
     The highest dose held out specifically, so the model must predict beyond its training range.
     Gate: PASS iff the best graded arm still beats D0 by at least 0.05 there. Reported separately
     from S2 because interpolation and extrapolation are different claims.

  S5 ARE THE FITTED CONSTANTS PHARMACOLOGY OR CURVE-FITTING ARTEFACTS?      -- requires S1
     Gate: PASS iff, among genes with a real response, at least half have a fitted EC50 inside the
     tested 0.1-100 micromolar window and a Hill coefficient between 0.3 and 5. Constants pinned
     at the edge of the grid are a fit that ran out of road, not a measured sensitivity.

  S6 CONTROL: DOSE LABELS PERMUTED WITHIN DRUG.      -- requires S2, VOID if S2's margin is under 0.01
     The same profiles, the same drug, the doses reassigned.
     Gate: PASS iff the graded arm's advantage over D0 collapses to under 25%.

  S7 WHAT THIS CANNOT SHOW -- written before the run.
     Four compounds. Dexamethasone is a nuclear receptor agonist, vorinostat an HDAC inhibitor,
     nutlin-3a an MDM2 antagonist and BMS-345541 an IKK inhibitor -- four mechanisms, not a
     sample from which anything general about drugs follows.
     Nutlin at 100 micromolar has 86 cells and BMS 129. Pseudobulk from that few cells is noisy,
     and the extrapolation gate S4 lands exactly there.
     One cell line and one timepoint. Nothing here separates a dose response from a time response.
     A Hill curve fitted to seven points cannot distinguish cooperativity from curvature caused by
     cell death at high dose, and three of these compounds are cytotoxic at 100 micromolar.
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
OUT = "outputs/loop_dose_hill.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SP = SCR / "sciplex2.h5ad"

SEED = 244244
MIN_CELLS_GENE = 200
KGRID = np.array([0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0])
EGRID = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
S1_MIN, S2_BAR, S3_BAR, S4_BAR, S5_FRAC, S6_MAX = 3, 0.05, 0.02, 0.05, 0.50, 0.25

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


def hill_basis(doses, K, eta):
    return (doses ** eta) / (K ** eta + doses ** eta)


def fit_shared(doses, Y):
    """One (K, eta) for all genes, amplitude per gene. Amplitude is linear given the basis, so the
    search is a 2-D grid with a closed form inside it -- no optimiser to fail."""
    best = (np.inf, KGRID[0], EGRID[0], None)
    for K in KGRID:
        for e in EGRID:
            b = hill_basis(doses, K, e)
            den = float(b @ b)
            if den <= 0: continue
            amp = (Y @ b) / den
            r = Y - np.outer(amp, b)
            ss = float((r * r).sum())
            if ss < best[0]: best = (ss, K, e, amp)
    return best[1], best[2], best[3]


def fit_full(doses, Y):
    """Per-gene (K, eta, amplitude), same grid, vectorised across genes."""
    G = Y.shape[0]
    bestss = np.full(G, np.inf); bK = np.zeros(G); bE = np.zeros(G); bA = np.zeros(G)
    for K in KGRID:
        for e in EGRID:
            b = hill_basis(doses, K, e)
            den = float(b @ b)
            if den <= 0: continue
            amp = (Y @ b) / den
            r = Y - np.outer(amp, b)
            ss = (r * r).sum(1)
            better = ss < bestss
            bestss[better] = ss[better]; bK[better] = K; bE[better] = e; bA[better] = amp[better]
    return bK, bE, bA


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "Hill dose-response on sci-Plex A549, held out by dose"}
    say("=" * 104)
    say("LOOP 244 -- DOSE AS A CONTINUOUS PARAMETER, HELD OUT BY DOSE")
    say("=" * 104)
    say("     THE LABELLING TRAP, found before any model was fitted: perturbation == 'control'")
    say("     is 529 cells from ONE well whose top_oligo is SAHA_100_E09 -- 100 uM vorinostat")
    say("     mislabelled as control. Using it as baseline would have subtracted a maximal drug")
    say("     response from every condition. The baseline here is each drug's own dose-0 cells.")

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
    def vcol(n):
        vb = h["var"][n]
        if isinstance(vb, h5py.Group):
            c = np.array([dec(x) for x in vb["categories"][:]]); return c[vb["codes"][:]]
        return np.array([dec(x) for x in vb[:]])
    sym = vcol("gene_symbol")
    shape = tuple(int(x) for x in h["X"].attrs["shape"])
    Xs = sparse.csr_matrix((h["X"]["data"][:], h["X"]["indices"][:], h["X"]["indptr"][:]),
                           shape=shape)
    h.close()
    say(f"     {shape[0]:,} cells x {shape[1]:,} genes, {Xs.nnz:,} non-zero counts")

    keepc = drug != "control"
    say(f"     dropped {int((~keepc).sum())} mislabelled 'control' cells (all from well E09)")
    Xs = Xs[keepc]; drug = drug[keepc]; dose = dose[keepc]

    tot = np.asarray(Xs.sum(1)).ravel()
    ok = tot > 0
    Xs = Xs[ok]; drug = drug[ok]; dose = dose[ok]; tot = tot[ok]
    ncell = np.asarray((Xs > 0).sum(0)).ravel()
    gkeep = np.where(ncell >= MIN_CELLS_GENE)[0]
    say(f"     {len(gkeep):,} genes detected in {MIN_CELLS_GENE}+ cells")
    Xs = Xs[:, gkeep]; sym = sym[gkeep]

    # CPM then log1p, then pseudobulk per (drug, dose)
    inv = sparse.diags(1e6 / tot)
    Xn = inv @ Xs
    Xn.data = np.log1p(Xn.data)
    drugs = sorted(set(drug) - {"control"})
    doses_all = np.array(sorted(set(dose)))
    nz = doses_all[doses_all > 0]
    say(f"     drugs {drugs}, non-zero doses {list(nz)}")

    PB = {}
    for dr in drugs:
        for dv in doses_all:
            m = (drug == dr) & (dose == dv)
            if m.sum() == 0: continue
            PB[(dr, dv)] = np.asarray(Xn[m].mean(0)).ravel()
    DELTA = {}
    for dr in drugs:
        base = PB.get((dr, 0.0))
        if base is None: continue
        for dv in nz:
            if (dr, dv) in PB: DELTA[(dr, dv)] = PB[(dr, dv)] - base
    say(f"     {len(DELTA)} (drug, dose) response profiles built against each drug's own vehicle")

    # ---------------------------------------------------------------- S1
    say("S1 IS THE DOSE AXIS REAL, WITHOUT ASSUMING IT IS MONOTONE?")
    say("     loop 234's R3 gated for a monotone increase across a saturating range -- it gated")
    say("     against its own hypothesis. Saturation is PREDICTED here, so the test is a positive")
    say("     rank correlation between dose and response magnitude, not monotonicity.")
    rho = {}
    for dr in drugs:
        dv = [d for d in nz if (dr, d) in DELTA]
        mag = [float(np.sqrt((DELTA[(dr, d)] ** 2).mean())) for d in dv]
        rho[dr] = float(stats.spearmanr(dv, mag).statistic)
        say(f"       {dr:<7} magnitudes " + " ".join(f"{m:.3f}" for m in mag) +
            f"   Spearman(dose, magnitude) {rho[dr]:+.3f}")
    npos = sum(1 for v in rho.values() if v > 0)
    G.add("S1", bool(npos >= S1_MIN), stat=float(npos),
          if_true=lambda: f"S1 PASS -- {npos} of {len(drugs)} drugs show a positive dose-magnitude "
                          f"rank correlation",
          if_false=lambda: f"S1 FAIL -- only {npos} of {len(drugs)} drugs; the dose axis is not "
                           f"behaving as an intervention axis")
    res["S1"] = {"spearman": rho, "n_positive": npos}

    # ---------------------------------------------------------------- held-out dose
    say("     fitting, leave-one-dose-out within each drug ...")
    units, S = [], collections.defaultdict(list)
    for dr in drugs:
        dv = np.array([d for d in nz if (dr, d) in DELTA])
        Yall = np.stack([DELTA[(dr, d)] for d in dv], 1)              # genes x doses
        for hi, hold in enumerate(dv):
            tr = np.array([j for j in range(len(dv)) if j != hi])
            dtr, Ytr = dv[tr], Yall[:, tr]
            truth = Yall[:, hi]
            p0 = Ytr.mean(1)
            lg = np.log10(dtr)
            A = np.stack([lg, np.ones_like(lg)], 1)
            coef = np.linalg.lstsq(A, Ytr.T, rcond=None)[0]
            p1 = coef[0] * np.log10(hold) + coef[1]
            Ks, es, amps = fit_shared(dtr, Ytr)
            p2 = amps * hill_basis(np.array([hold]), Ks, es)[0]
            bK, bE, bA = fit_full(dtr, Ytr)
            p3 = bA * ((hold ** bE) / (bK ** bE + hold ** bE))
            units.append((dr, float(hold)))
            S["D0_BLIND"].append(pear(p0, truth)); S["D1_LOGLIN"].append(pear(p1, truth))
            S["D2_HILL_SHARED"].append(pear(p2, truth)); S["D3_HILL_FULL"].append(pear(p3, truth))
            if hi == len(dv) - 1:
                res.setdefault("constants", {})[dr] = {
                    "K_shared": float(Ks), "eta_shared": float(es), "n_genes": int(len(bK)),
                    "note": "from the fold holding out the top dose"}
                res.setdefault("_full", {})[dr] = (bK, bE, bA, truth)
    S = {k: np.asarray(v) for k, v in S.items()}
    say(f"     {len(units)} held-out (drug, dose) units")
    for a in ["D0_BLIND", "D1_LOGLIN", "D2_HILL_SHARED", "D3_HILL_FULL"]:
        say(f"       {a:<16} {np.nanmean(S[a]):+.4f}  (sd {np.nanstd(S[a]):.4f})")
    res["arms"] = {a: float(np.nanmean(S[a])) for a in S}

    # ---------------------------------------------------------------- S2
    say("S2 DOES A GRADED MODEL BEAT A DOSE-BLIND ONE?")
    graded = ["D1_LOGLIN", "D2_HILL_SHARED", "D3_HILL_FULL"]
    best = max(graded, key=lambda a: np.nanmean(S[a]))
    d2, se2, z2 = paired(S[best], S["D0_BLIND"])
    say(f"     best graded arm {best} {np.nanmean(S[best]):+.4f} vs D0_BLIND "
        f"{np.nanmean(S['D0_BLIND']):+.4f}   paired {d2:+.4f} +/- {se2:.4f} ({z2:+.1f} se)")
    G.add("S2", bool(d2 >= S2_BAR), stat=float(d2), requires=("S1",),
          if_true=lambda: f"S2 PASS -- knowing the dose is worth {d2:+.4f} on a held-out dose",
          if_false=lambda: f"S2 FAIL -- knowing the dose is worth {d2:+.4f} against a {S2_BAR} bar")
    res["S2"] = {"best": best, "delta": d2, "se": se2, "z": z2}

    # ---------------------------------------------------------------- S3
    say("S3 DOES THE HILL CURVE BEAT A STRAIGHT LINE IN LOG DOSE?")
    bh = max(["D2_HILL_SHARED", "D3_HILL_FULL"], key=lambda a: np.nanmean(S[a]))
    d3, se3, z3 = paired(S[bh], S["D1_LOGLIN"])
    say(f"     {bh} {np.nanmean(S[bh]):+.4f} vs D1_LOGLIN {np.nanmean(S['D1_LOGLIN']):+.4f}   "
        f"paired {d3:+.4f} +/- {se3:.4f} ({z3:+.1f} se)")
    G.add("S3", bool(d3 >= S3_BAR), stat=float(d3), requires=("S1",),
          if_true=lambda: f"S3 PASS -- the saturating form adds {d3:+.4f} over a straight line",
          if_false=lambda: f"S3 FAIL -- the saturating form adds {d3:+.4f} over a straight line in "
                           f"log dose, against a {S3_BAR} bar; the S2 gain is 'dose matters', not "
                           f"'the curve is a Hill curve'")
    res["S3"] = {"best_hill": bh, "delta": d3, "se": se3, "z": z3}

    # ---------------------------------------------------------------- S4
    say("S4 CAN IT EXTRAPOLATE PAST THE DOSES IT SAW?")
    top = float(nz.max())
    ie = np.array([u[1] == top for u in units])
    inter = ~ie
    d4, se4, z4 = paired(S[best][ie], S["D0_BLIND"][ie])
    di, _, _ = paired(S[best][inter], S["D0_BLIND"][inter])
    say(f"     interior doses ({int(inter.sum())} units): {best} beats D0 by {di:+.4f}")
    say(f"     highest dose {top:g} uM ({int(ie.sum())} units, extrapolation): {d4:+.4f} "
        f"+/- {se4:.4f} ({z4:+.1f} se)")
    G.add("S4", bool(d4 >= S4_BAR), stat=float(d4), requires=("S1",),
          if_true=lambda: f"S4 PASS -- the advantage survives extrapolation at {d4:+.4f}",
          if_false=lambda: f"S4 FAIL -- outside its training range the advantage is {d4:+.4f} "
                           f"against {di:+.4f} inside it")
    res["S4"] = {"interior": di, "extrapolation": d4, "se": se4, "top_dose": top}

    # ---------------------------------------------------------------- S5
    say("S5 ARE THE FITTED CONSTANTS PHARMACOLOGY OR CURVE-FITTING ARTEFACTS?")
    inw, tot_g = 0, 0
    for dr, (bK, bE, bA, truth) in res.pop("_full").items():
        resp = np.abs(bA) > np.percentile(np.abs(bA), 90)
        good = resp & (bK >= nz.min()) & (bK <= nz.max()) & (bE >= 0.3) & (bE <= 5.0)
        inw += int(good.sum()); tot_g += int(resp.sum())
        say(f"       {dr:<7} shared K {res['constants'][dr]['K_shared']:g} uM, "
            f"eta {res['constants'][dr]['eta_shared']:g}   |   responsive genes "
            f"{int(resp.sum()):,}, with EC50 inside 0.1-100 uM and eta in [0.3,5]: "
            f"{good.sum() / max(resp.sum(), 1):.0%}")
    frac = inw / max(tot_g, 1)
    say(f"     pooled: {frac:.0%} of responsive genes have both constants off the grid edge")
    G.add("S5", bool(frac >= S5_FRAC), stat=float(frac), requires=("S1",),
          if_true=lambda: f"S5 PASS -- {frac:.0%} of responsive genes carry plausible constants",
          if_false=lambda: f"S5 FAIL -- only {frac:.0%}; most fits are pinned at the grid edge, "
                           f"which is a fit that ran out of road rather than a measured EC50")
    res["S5"] = {"fraction_plausible": frac, "n_responsive": tot_g}

    # ---------------------------------------------------------------- S6
    say("S6 CONTROL: DOSE LABELS PERMUTED WITHIN DRUG")
    if d2 < 0.01:
        G.add("S6", False, stat=float(d2), requires=("S2",), void_if=True,
              void_reason=f"the real advantage is {d2:+.4f}; there is nothing to collapse")
    else:
        sh = collections.defaultdict(list)
        for dr in drugs:
            dv = np.array([d for d in nz if (dr, d) in DELTA])
            Yall = np.stack([DELTA[(dr, d)] for d in dv], 1)
            for hi, hold in enumerate(dv):
                tr = np.array([j for j in range(len(dv)) if j != hi])
                dperm = rng.permutation(dv[tr])                 # profiles kept, doses reassigned
                Ytr, truth = Yall[:, tr], Yall[:, hi]
                sh["D0_BLIND"].append(pear(Ytr.mean(1), truth))
                if best == "D1_LOGLIN":
                    A = np.stack([np.log10(dperm), np.ones(len(dperm))], 1)
                    c = np.linalg.lstsq(A, Ytr.T, rcond=None)[0]
                    p = c[0] * np.log10(hold) + c[1]
                elif best == "D2_HILL_SHARED":
                    Ks, es, amps = fit_shared(dperm, Ytr)
                    p = amps * hill_basis(np.array([hold]), Ks, es)[0]
                else:
                    bK, bE, bA = fit_full(dperm, Ytr)
                    p = bA * ((hold ** bE) / (bK ** bE + hold ** bE))
                sh[best].append(pear(p, truth))
        ds, _, _ = paired(np.asarray(sh[best]), np.asarray(sh["D0_BLIND"]))
        f6 = ds / d2
        say(f"     doses reassigned within drug: {ds:+.4f} against a real {d2:+.4f}  ({f6:.0%})")
        G.add("S6", bool(f6 <= S6_MAX), stat=float(f6), requires=("S2",),
              if_true=lambda: f"S6 PASS -- collapses to {f6:.0%} when the doses are permuted",
              if_false=lambda: f"S6 FAIL -- {f6:.0%} survives permuted doses; the gain is not the "
                               f"dose")
        res["S6"] = {"real": d2, "shuffled": ds, "fraction": f6}

    say("S7 WHAT THIS CANNOT SHOW")
    say("     Four compounds with four mechanisms -- a nuclear receptor agonist, an HDAC")
    say("     inhibitor, an MDM2 antagonist and an IKK inhibitor. Nothing general about drugs")
    say("     follows from four of them.")
    say("     Nutlin at 100 uM has 86 cells and BMS 129. The extrapolation gate S4 lands exactly")
    say("     on the thinnest pseudobulk in the dataset.")
    say("     One cell line, one timepoint: nothing here separates a dose response from a time")
    say("     response.")
    say("     A Hill curve on seven points cannot tell cooperativity from curvature caused by")
    say("     cell death, and three of these compounds are cytotoxic at the top dose.")

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
