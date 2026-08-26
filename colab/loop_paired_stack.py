"""Loop 229. The surviving fixes, applied to the real stack, on a harness that can resolve them.

WHY THE HARNESS COMES FIRST. Loop 228 tried five fixes and could not adjudicate three of them,
because it scored every arm on ONE random split. Measured directly afterwards, the physics arm's
held-out |r| ranged from 0.0139 to 0.0736 across 30 different fold splits of identical data --
mean 0.0354, standard deviation 0.0169 -- while the effect it was trying to detect was +0.0307.
The effect was smaller than the noise in its own baseline. Two runs of the same loop differing
only in the split gave equilibrium 0.0278 and 0.0735, and flipped the sign of the fitted alpha
from -1.7783 to +1.7783. I reported that sign as a finding before checking it.

That is a defect in the measuring instrument, and loops 206, 209, 211, 213 and 228 all reported
single-split numbers from the same regime. This loop replaces the instrument.

    EVERY comparison below is PAIRED across the SAME 20 splits. The quantity tested is not
    "arm A scored higher than arm B" but the mean and standard error of (A - B) computed
    split by split. Split-to-split variance is common to both arms and cancels exactly in
    the pairing, which is why a paired test can resolve a 0.02 effect that an unpaired one
    cannot see through a 0.017 standard deviation.

WHAT SURVIVED LOOP 228 AND IS CARRIED HERE. Only what passed a control:

    MAGNITUDE OVER SIGNED MEAN.  H2b measured a denominator-free magnitude summary reproducing
    at split-half +0.1945 against the signed mean's +0.0921. Near zero the SIGN of a change is a
    coin flip while its SIZE is not, so squaring keeps what reproduces and discards what does not.
    The switching-fraction version of this is NOT carried: H2a's control showed it survives at
    +0.9626 with its numerator destroyed, so it was a shared denominator correlating with itself.

    RELIABILITY WEIGHTING.  Loop 224 X6 raised K562-to-RPE1 agreement from +0.22862 to +0.30025
    by weighting each gene by the reliability measured from 1,914,250 individual cells. That is
    the only fix in this arc that produced a gain with a passing control behind it.

    STACKING OVER CONCATENATION.  Loop 213 measured 0.5474 against 0.4345; loop 228 measured
    0.4252 against 0.4086. Directionally consistent twice, under bar once.

NOT CARRIED, and the reasons are recorded rather than the arms quietly dropped: local-concentration
mu (H3 lost to its own shuffled-density control, 0.1072 against 0.1001), driven non-equilibrium
occupancy (H4 passed but its effect sits inside its baseline's split noise), and cell-state
interaction terms (H5, 0.3115 against 0.3126 additive).

AND J2 ASKS WHETHER THE PROJECT'S HEADLINE NUMBER SURVIVES ITS OWN INSTRUMENT. 0.5474 has been
quoted as the standing stack score in eight commits. It was one split. If the repeated-split mean
lands far from it, that is a correction to the record, not a new result.

PREDECLARED, BEFORE ANY NUMBER.

  J1 IS THE HARNESS NOW RESOLVABLE?
     Rebuild the curated stack and score it on 20 independent splits.
     Gate: PASS iff the across-split standard deviation of the stacked score is below 0.03, so
     that a 0.02 effect measured PAIRED is detectable. Report mean, sd, min and max regardless.

  J2 IS 0.5474 INSIDE ITS OWN NOISE?
     Gate: PASS iff |0.5474 - repeated-split mean| < 2 standard deviations. A FAIL means the
     project's headline stack number was a favourable split and the record needs correcting.
     This gate can fire in either direction and neither is assumed.

  J3 DOES THE MAGNITUDE SUMMARY BEAT THE SIGNED MEAN?  -- paired
     The Perturb-seq block built from m^2 against the same block built from m, nothing else
     changed.
     Gate: PASS iff the paired mean difference exceeds 2 standard errors AND exceeds +0.01.
     Significance alone is not enough at 20 paired splits and the magnitude bar is stated here.

  J4 DOES RELIABILITY WEIGHTING HELP?  -- paired
     Perturb-seq features weighted by loop 224's per-gene reliability against unweighted.
     Gate: PASS iff the paired mean difference exceeds 2 standard errors AND exceeds +0.01.

  J5 RIDGE OR MLP -- SETTLED, PAIRED
     This project has three conflicting answers: loop 211 measured ridge 0.4057 against MLP-wide
     0.2072; loop 225 E6 measured MLP 0.1222 against ridge 0.0887 and I called it a finding;
     loop 226 F7 reversed it again, ridge 0.1037 against MLP 0.0487 with the MLP's margin over
     its own shuffled control at +0.0005. All three were single splits.
     Gate: PASS iff the paired difference between the two exceeds 2 standard errors in EITHER
     direction, so the question is answered rather than left open. The winner is reported; the
     gate is on resolvability, not on which one wins.

  J6 ALL SURVIVING FIXES TOGETHER  -- paired
     Gate: PASS iff the fixed stack beats the unfixed stack by more than 2 standard errors AND
     by more than +0.02, paired across the same 20 splits.

  J7 SHUFFLED CONTROL
     Gate: PASS iff the fixed stack exceeds its shuffled-target score by at least 0.10 on every
     one of the 20 splits, not merely on average.

  J8 WHAT THIS CANNOT SHOW -- written before the run.
     Pairing removes split variance from a COMPARISON. It does not make any single arm's absolute
     score more accurate, so J1's mean is still one estimate of one quantity on 1,336 genes.
     The target is the A549 dexamethasone plateau. Loop 216 measured the plateau's replicate
     ceiling at +0.83380 and the per-interval change at -0.54028, so this loop works on the half
     of the problem that is measurable and says nothing about the half that is not.
     The magnitude summary discards direction. A model that predicts how much a gene moves but
     not which way is less useful than its score suggests, and no gate here penalises that.
"""
import os, sys, json, gzip, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_paired_stack.json"
SP = L191.SP
CK = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
REL_F = ROOT / "outputs" / "loop224_reliability.npz"
TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
SEED, NSPLIT, NFOLD, K_PS = 229229, 20, 5, 24
REF_213, SD_BAR, MIN_GAIN, CTRL_GAIN = 0.5474, 0.03, 0.01, 0.10
CUR = (0, 20000)

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def cv_pred(X, y, folds, lam=1.0):
    p = np.zeros(len(y))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.hstack([(X[tr] - mu) / sd, np.ones((len(tr), 1))])
        R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y[tr])
        p[te] = np.hstack([(X[te] - mu) / sd, np.ones((len(te), 1))]) @ w
    return p


def stack_score(blocks, y, folds):
    P = np.column_stack([cv_pred(np.nan_to_num(v).reshape(len(y), -1), y, folds)
                         for v in blocks.values()])
    return abs(pear(y, cv_pred(P, y, folds)))


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(se), float(d.mean() / se if se > 0 else np.inf)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "surviving fixes on a paired repeated-split harness"}
    say("=" * 104)
    say("LOOP 229 -- THE SURVIVING FIXES, ON A HARNESS THAT CAN RESOLVE THEM")
    say("=" * 104)
    say("     Loop 228 could not adjudicate three of five fixes because it scored one split.")
    say("     Measured after: the physics arm ranged 0.0139 to 0.0736 across 30 splits of the")
    say("     SAME data, sd 0.0169, while the effect sought was +0.0307. Every comparison here")
    say("     is PAIRED across the same 20 splits, so split variance cancels instead of hiding.")

    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    y_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    gp = {s: k for k, s in enumerate(allg)}

    relz = np.load(REL_F, allow_pickle=True)
    ro_gene = np.array([str(x) for x in relz["gene"]])
    reliab = np.nan_to_num(relz["reliability"].astype(np.float64), nan=0.0)
    ro = {g: i for i, g in enumerate(ro_gene)}
    names = [s for s in allg if s in ro]
    y = np.array([y_all[gp[s]] for s in names])
    N = len(names)
    say(f"     {N:,} genes carrying an A549 plateau and a Perturb-seq readout")

    ck = np.load(CK / "loop224_accum.npz")
    S, Q, n = ck["S"], ck["Q"], ck["n"]
    okp = n >= 40
    cols = np.array([ro[s] for s in names])
    nn = n[okp][:, None].astype(np.float64)
    m_sub = (S[okp][:, cols].astype(np.float64)) / nn
    fin = np.isfinite(m_sub).all(1)
    m_sub = m_sub[fin]
    say(f"     Perturb-seq: {m_sub.shape[0]:,} finite perturbations x {N:,} genes")
    w_rel = reliab[cols]

    def embed(Xp, k=K_PS):
        Xc = Xp.T
        Xc = Xc - Xc.mean(0, keepdims=True)
        Xc = np.nan_to_num(Xc)
        keep = np.isfinite(Xc).all(0) & (Xc.std(0) > 0)
        U, s_, _ = np.linalg.svd(Xc[:, keep], full_matrices=False)
        kk = min(k, U.shape[1])
        E = np.zeros((U.shape[0], k)); E[:, :kk] = U[:, :kk] * s_[:kk]
        return E

    PS_mean = embed(m_sub)
    PS_mag = embed(m_sub ** 2)
    PS_mean_w = PS_mean * w_rel[:, None]
    PS_mag_w = PS_mag * w_rel[:, None]

    TR = {}
    for t in TRACKS:
        pt, PM = L191.promoter_track(t, [tssb.get(s) for s in sym], L191.PROM_PAD, lambda *_: None)
        TR[t] = PM[[int(np.where(pt == tt)[0][0]) for tt in grid]]
    CHIP = np.column_stack([np.column_stack([
        TR[t][:, gi].mean(0), TR[t][:, gi].max(0), TR[t][-1, gi] - TR[t][0, gi]])
        for t in TRACKS])
    CHIP = np.array([CHIP[gp[s]] for s in names])

    CH = json.load(open(SP / "_chromatin_features.json"))["features"]
    ch = np.array([[CH.get(s.upper(), {}).get("pc1", 0.0), CH.get(s.upper(), {}).get("ins", 0.0),
                    np.log1p(CH.get(s.upper(), {}).get("dens") or 0.0)] for s in names], float)
    ch = np.nan_to_num(ch)

    nb = json.load(gzip.open("colab/data/net_bundle.json.gz"))
    nidx = {n_.upper(): i for i, n_ in enumerate(nb["names"])}
    ppi = Counter()
    for a, b in nb["ppi"]:
        ppi[int(a)] += 1; ppi[int(b)] += 1
    outd, ind = Counter(), Counter()
    for s_, t_, g_ in nb["reg"]:
        outd[int(s_)] += 1; ind[int(t_)] += 1
    ncplx = Counter()
    for _, mem in nb["complexes"].items():
        for m_ in mem:
            ncplx[int(m_)] += 1
    nrx = {int(k): len(v) for k, v in nb["generxn"].items()}
    coex = nb["coexpr"]
    NET = []
    for s in names:
        i = nidx.get(s.upper(), -1)
        cx = coex.get(str(i), [])
        NET.append([np.log1p(ppi.get(i, 0)), np.log1p(outd.get(i, 0)), np.log1p(ind.get(i, 0)),
                    np.log1p(ncplx.get(i, 0)), np.log1p(nrx.get(i, 0)), len(cx),
                    float(np.mean([c[1] for c in cx])) if cx else 0.0,
                    float(np.max([c[1] for c in cx])) if cx else 0.0])
    NET = np.array(NET)

    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    T = {str(g["name"]).upper(): g for g in tab}
    def onehot(vals, top=12):
        keys = [k for k, _ in Counter(vals).most_common(top)]
        return np.array([[1.0 if v == k else 0.0 for k in keys] for v in vals])
    comp_oh = onehot([str(T.get(s.upper(), {}).get("comp") or "") for s in names])
    proc_oh = onehot([str(T.get(s.upper(), {}).get("proc") or "") for s in names])
    FUN = np.hstack([np.array([[float(T.get(s.upper(), {}).get("loeuf") or 1.0),
                                float(T.get(s.upper(), {}).get("cpg") or 0),
                                np.log1p(float(T.get(s.upper(), {}).get("enh") or 0)),
                                np.log1p(float(T.get(s.upper(), {}).get("ndis") or 0)),
                                float(T.get(s.upper(), {}).get("ess") or 0),
                                float(T.get(s.upper(), {}).get("dark") or 0)] for s in names]),
                     comp_oh])
    PATH = np.hstack([np.array([[np.log1p(float(T.get(s.upper(), {}).get("npath") or 0))]
                                for s in names]), proc_oh])
    FAME = np.array([[np.log1p(float(T.get(s.upper(), {}).get("pubs") or 0))] for s in names])
    say(f"     blocks: network {NET.shape[1]}, function {FUN.shape[1]}, pathways {PATH.shape[1]}, "
        f"chip {CHIP.shape[1]}, chromatin {ch.shape[1]}, perturbseq {K_PS}, fame 1")

    BASE = {"network": NET, "function": FUN, "pathways": PATH, "chip": CHIP,
            "chromatin": ch, "fame": FAME}
    splits = [np.random.default_rng(SEED + i).permutation(N) for i in range(NSPLIT)]
    FOLDS = [[p[k::NFOLD] for k in range(NFOLD)] for p in splits]

    def run(blocks):
        return np.array([stack_score(blocks, y, f) for f in FOLDS])

    # ---------------------------------------------------------------- J1
    say("J1 IS THE HARNESS NOW RESOLVABLE?")
    unfixed = dict(BASE); unfixed["perturbseq"] = PS_mean
    s_unf = run(unfixed)
    say(f"     stacked score over {NSPLIT} independent splits:")
    say(f"       mean {s_unf.mean():.4f}   sd {s_unf.std(ddof=1):.4f}   "
        f"min {s_unf.min():.4f}   max {s_unf.max():.4f}")
    G.add("J1", bool(s_unf.std(ddof=1) < SD_BAR), stat=float(s_unf.std(ddof=1)),
          if_true=lambda: f"J1 PASS -- across-split sd {s_unf.std(ddof=1):.4f} is below "
                          f"{SD_BAR}, so a paired {MIN_GAIN} effect is detectable",
          if_false=lambda: f"J1 FAIL -- sd {s_unf.std(ddof=1):.4f}; single-split numbers from "
                           f"this harness cannot be compared at all")
    res["harness"] = {"mean": float(s_unf.mean()), "sd": float(s_unf.std(ddof=1)),
                      "min": float(s_unf.min()), "max": float(s_unf.max()), "n": NSPLIT}

    # ---------------------------------------------------------------- J2
    say("J2 IS 0.5474 INSIDE ITS OWN NOISE?")
    dev = abs(REF_213 - s_unf.mean()) / max(s_unf.std(ddof=1), 1e-9)
    say(f"     loop 213 reported {REF_213:.4f} on one split")
    say(f"     this harness, {NSPLIT} splits: {s_unf.mean():.4f} +/- {s_unf.std(ddof=1):.4f}")
    say(f"     deviation {abs(REF_213-s_unf.mean()):+.4f} = {dev:.1f} standard deviations")
    say("     note the block set differs from loop 213's ten -- structure3d, reactions and")
    say("     tf_network are not rebuilt here -- so a gap is not necessarily a bad split")
    G.add("J2", bool(dev < 2.0), stat=float(dev),
          if_true=lambda: f"J2 PASS -- {REF_213:.4f} sits {dev:.1f} sd from the repeated-split "
                          f"mean; the headline number is consistent with its own noise",
          if_false=lambda: f"J2 FAIL -- {REF_213:.4f} is {dev:.1f} sd from the repeated-split "
                           f"mean {s_unf.mean():.4f}; on this block set it is not reproducible")
    res["headline"] = {"loop213": REF_213, "repeated_mean": float(s_unf.mean()), "sd_away": dev}

    # ---------------------------------------------------------------- J3
    say("J3 DOES THE MAGNITUDE SUMMARY BEAT THE SIGNED MEAN?  -- paired")
    fixed_mag = dict(BASE); fixed_mag["perturbseq"] = PS_mag
    s_mag = run(fixed_mag)
    d, se, z = paired(s_mag, s_unf)
    say(f"     signed mean  {s_unf.mean():.4f} +/- {s_unf.std(ddof=1):.4f}")
    say(f"     magnitude    {s_mag.mean():.4f} +/- {s_mag.std(ddof=1):.4f}")
    say(f"     PAIRED difference {d:+.4f} +/- {se:.4f}  ({z:+.1f} standard errors)")
    G.add("J3", bool(z > 2.0 and d > MIN_GAIN), stat=float(d),
          if_true=lambda: f"J3 PASS -- magnitude gains {d:+.4f} paired, {z:.1f} standard errors",
          if_false=lambda: f"J3 FAIL -- {d:+.4f} +/- {se:.4f}, {z:+.1f} standard errors, against "
                           f"a {MIN_GAIN:+.2f} bar")
    res["magnitude"] = {"mean_signed": float(s_unf.mean()), "mean_magnitude": float(s_mag.mean()),
                        "paired_delta": d, "se": se, "z": z}

    # ---------------------------------------------------------------- J4
    say("J4 DOES RELIABILITY WEIGHTING HELP?  -- paired")
    best_ps = PS_mag if d > 0 else PS_mean
    best_w = PS_mag_w if d > 0 else PS_mean_w
    a_un = run({**BASE, "perturbseq": best_ps})
    a_w = run({**BASE, "perturbseq": best_w})
    d4, se4, z4 = paired(a_w, a_un)
    say(f"     unweighted {a_un.mean():.4f}   reliability-weighted {a_w.mean():.4f}")
    say(f"     PAIRED difference {d4:+.4f} +/- {se4:.4f}  ({z4:+.1f} standard errors)")
    say(f"     loop 224 X6 measured weighting raising K562-to-RPE1 from 0.22862 to 0.30025")
    G.add("J4", bool(z4 > 2.0 and d4 > MIN_GAIN), stat=float(d4),
          if_true=lambda: f"J4 PASS -- weighting gains {d4:+.4f} paired, {z4:.1f} standard errors",
          if_false=lambda: f"J4 FAIL -- {d4:+.4f} +/- {se4:.4f}, {z4:+.1f} standard errors")
    res["weighting"] = {"unweighted": float(a_un.mean()), "weighted": float(a_w.mean()),
                        "paired_delta": d4, "se": se4, "z": z4}

    # ---------------------------------------------------------------- J5
    say("J5 RIDGE OR MLP -- SETTLED, PAIRED")
    say("     loop 211 ridge 0.4057 / MLP-wide 0.2072; loop 225 MLP 0.1222 / ridge 0.0887;")
    say("     loop 226 ridge 0.1037 / MLP 0.0487. All three were single splits.")
    BEST = {**BASE, "perturbseq": (best_w if d4 > 0 else best_ps)}
    Xall = np.nan_to_num(np.hstack([v.reshape(N, -1) for v in BEST.values()]))
    r_ridge, r_mlp = [], []
    try:
        from sklearn.neural_network import MLPRegressor
        for f in FOLDS:
            r_ridge.append(abs(pear(y, cv_pred(Xall, y, f))))
            pm = np.zeros(N)
            for te in f:
                tr = np.setdiff1d(np.arange(N), te)
                mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-9
                nnw = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED,
                                   early_stopping=True)
                nnw.fit((Xall[tr] - mu) / sd, y[tr])
                pm[te] = nnw.predict((Xall[te] - mu) / sd)
            r_mlp.append(abs(pear(y, pm)))
        r_ridge, r_mlp = np.array(r_ridge), np.array(r_mlp)
        d5, se5, z5 = paired(r_ridge, r_mlp)
        win = "ridge" if d5 > 0 else "MLP"
        say(f"     ridge {r_ridge.mean():.4f} +/- {r_ridge.std(ddof=1):.4f}")
        say(f"     MLP   {r_mlp.mean():.4f} +/- {r_mlp.std(ddof=1):.4f}")
        say(f"     PAIRED ridge - MLP {d5:+.4f} +/- {se5:.4f}  ({z5:+.1f} standard errors); "
            f"{win} wins")
        ok5 = bool(abs(z5) > 2.0)
    except Exception as e:
        d5 = se5 = z5 = float("nan"); win = None; ok5 = False
        say(f"     MLP arm did not run: {type(e).__name__}: {e}")
    G.add("J5", ok5, stat=float(z5) if np.isfinite(z5) else None,
          if_true=lambda: f"J5 PASS -- {win} wins by {abs(d5):.4f} paired at {abs(z5):.1f} "
                          f"standard errors; the three-way conflict is settled",
          if_false=lambda: f"J5 FAIL -- {abs(z5) if np.isfinite(z5) else float('nan'):.1f} "
                           f"standard errors; still unresolved")
    res["ridge_vs_mlp"] = {"ridge": float(np.mean(r_ridge)) if len(r_ridge) else None,
                           "mlp": float(np.mean(r_mlp)) if len(r_mlp) else None,
                           "paired_delta": d5, "se": se5, "z": z5, "winner": win}

    # ---------------------------------------------------------------- J6
    say("J6 ALL SURVIVING FIXES TOGETHER  -- paired")
    s_fix = run(BEST)
    d6, se6, z6 = paired(s_fix, s_unf)
    say(f"     unfixed  (signed mean, unweighted, stacked) {s_unf.mean():.4f} "
        f"+/- {s_unf.std(ddof=1):.4f}")
    say(f"     fixed    (magnitude, reliability-weighted)  {s_fix.mean():.4f} "
        f"+/- {s_fix.std(ddof=1):.4f}")
    say(f"     PAIRED difference {d6:+.4f} +/- {se6:.4f}  ({z6:+.1f} standard errors)")
    G.add("J6", bool(z6 > 2.0 and d6 > 0.02), stat=float(d6),
          if_true=lambda: f"J6 PASS -- the surviving fixes gain {d6:+.4f} paired at {z6:.1f} "
                          f"standard errors, reaching {s_fix.mean():.4f}",
          if_false=lambda: f"J6 FAIL -- {d6:+.4f} +/- {se6:.4f}, {z6:+.1f} standard errors, "
                           f"against a +0.02 bar")
    res["combined"] = {"unfixed": float(s_unf.mean()), "fixed": float(s_fix.mean()),
                       "paired_delta": d6, "se": se6, "z": z6}

    # ---------------------------------------------------------------- J7
    say("J7 SHUFFLED CONTROL")
    sh = []
    for i, f in enumerate(FOLDS):
        ysh = y.copy(); np.random.default_rng(SEED + 900 + i).shuffle(ysh)
        P = np.column_stack([cv_pred(np.nan_to_num(v).reshape(N, -1), ysh, f)
                             for v in BEST.values()])
        sh.append(abs(pear(ysh, cv_pred(P, ysh, f))))
    sh = np.array(sh)
    margins = s_fix - sh
    say(f"     shuffled target {sh.mean():.4f} +/- {sh.std(ddof=1):.4f}")
    say(f"     per-split margin: min {margins.min():+.4f}, mean {margins.mean():+.4f}")
    G.add("J7", bool(margins.min() >= CTRL_GAIN), stat=float(margins.min()),
          if_true=lambda: f"J7 PASS -- the fixed stack beats its shuffled control by at least "
                          f"{margins.min():.4f} on EVERY one of {NSPLIT} splits",
          if_false=lambda: f"J7 FAIL -- worst-split margin {margins.min():+.4f}")
    res["shuffled"] = {"mean": float(sh.mean()), "min_margin": float(margins.min())}

    # ---------------------------------------------------------------- J8
    say("J8 WHAT THIS CANNOT SHOW")
    say("     Pairing removes split variance from a COMPARISON. It does not make any single arm's")
    say("     absolute score more accurate, so J1's mean is one estimate on one gene set.")
    say("     Loop 216 measured the plateau's replicate ceiling at +0.83380 and the per-interval")
    say("     change at -0.54028. This loop works the measurable half and says nothing about the")
    say("     half that is not.")
    say("     The magnitude summary discards direction. A model that predicts how much a gene")
    say("     moves but not which way is less useful than its score suggests, and no gate here")
    say("     penalises that.")

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
