"""Loop 212. A requirement calibrated on real predictors, and every gene the blocks can reach.

TWO THINGS LOOP 210 AND 211 LEFT BROKEN, AND THEY ARE THE SAME PROBLEM SEEN TWICE.

FIRST, THE REQUIREMENT IS NOT A SPECIFICATION. Loop 206 measured "a set point must reach r 0.9081"
and loop 210 revised it to 0.2481 under a shrunk rate rule. Both numbers were produced the same
way: take the TRUE plateau, add independent Gaussian noise until the correlation falls to r, and
see where the relaxation crosses persistence. Loop 210's C6b showed that does not transfer. A real
ridge predictor at r 0.4761 -- nearly twice the 0.2481 the sweep says is needed -- cleared
persistence by +0.00046, essentially not at all.

The mechanism was measured, not guessed: predicted variance 0.1959 against a true plateau variance
of 0.6488, a ratio of 0.3020. A fitted predictor SHRINKS toward the mean, so it is UNDER-dispersed
and its errors are correlated with the target. Added noise is OVER-dispersed and its errors are
independent. Same correlation, opposite geometry, different usefulness. So the calibration
instrument was wrong, and this loop replaces it: build a FAMILY OF REAL PREDICTORS spanning a range
of quality and read the crossover off them.

SECOND, THE GENE SET IS THE BINDING CONSTRAINT AND IT WAS SELF-INFLICTED. Loop 211's learning curve
rose monotonically to the last point -- 0.1574, 0.2781, 0.3590, 0.3966, 0.4057 at 20 to 100 per
cent of training genes -- and its verdict was DATA-limited. But the 600 genes are the INTERSECTION
of three blocks, and requiring all three is a modelling choice, not a fact. Loop 208 measured that
only 49.6% of the A549 responder set carries a Perturb-seq readout. A gene missing one block is not
a gene with no information; it is a gene with two blocks. Handling block-wise missingness instead
of dropping the row should recover most of loop 198's 1,336.

PREDECLARED, BEFORE ANY NUMBER.

  E1 IS THE EXPANDED SET HONEST?
     Gate: PASS iff the expanded set is a strict superset of the 600-gene intersection, every gene
     carries at least one real block, and each block's missingness indicator is non-constant --
     a constant indicator would mean the imputation is doing nothing and the expansion is fake.

  E2 THE REQUIREMENT, CALIBRATED ON REAL PREDICTORS.
     Build a family of genuinely fitted predictors spanning a wide range of quality by varying the
     feature subset, the regularisation and the training size, so each has the error geometry of a
     real predictor rather than of a noised oracle. Fit the relaxation for each and find where
     held-out R2 crosses persistence.
     Gate: PASS iff the real-predictor crossover differs from the noised-oracle crossover (0.2481)
     by more than 0.05. A PASS confirms C6b with a proper instrument; a FAIL would mean C6b's
     single point was a fluke and the noise calibration was fine after all.

  E3 IS DISPERSION THE MECHANISM?
     For every predictor in the family, record its dispersion ratio var(pred)/var(true) alongside
     its r and its R2.
     Gate: PASS iff, among predictors matched on r to within 0.05, the more dispersed one scores
     the higher R2 in the majority of matched pairs. This tests C6b's stated mechanism directly
     rather than accepting it because it sounds right. Requires E2.

  E4 DOES VARIANCE CORRECTION RECOVER ANYTHING?
     Rescale each prediction to match the true plateau's variance before relaxing, and rescore.
     Gate: PASS iff the corrected set point beats the uncorrected one by more than 0.01 in R2.
     If dispersion is the mechanism this is the cheapest available fix; if it is not, this fails
     and E3's mechanism is wrong. Requires E3.

  E5 DOES THE EXPANDED SET HELP?
     Gate: PASS iff the expanded set carries at least 1.5x the genes AND its combined |r| is
     within 0.05 of the 600-gene 0.4761 -- that is, the added genes did not dilute the signal.

  E6 DOES ANYTHING FINALLY BEAT PERSISTENCE?
     The best available combination -- expanded set, shrunk rate, per-gene lambda, variance
     correction if E4 passed -- scored held-out in time.
     Gate: PASS iff held-out R2 exceeds persistence by more than 0.01. This is the bar loop 198
     set and nothing in this project has cleared.

  E7 IS ANY OF IT FAME OR MISSINGNESS?
     Gate: PASS iff the model beats publication count AND beats a model built from the missingness
     indicators alone -- because which genes have which blocks is itself informative and would be
     a trivial way to score.

  E8 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, sys, time, warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import SEQ_F, gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
PHYS_CACHE = ROOT / "colab" / "data" / "physics" / "motif_occupancy_1200.npz"
K562 = SP / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
LIFE = "outputs/orphan/cell_lifetimes.json"
OUT = "outputs/loop_honest_requirement.json"
N_TRAIN, SEED = 6, 212212
TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
NOISE_CROSS = 0.2481
R_600 = 0.4761

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def ridge(Xtr, ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    return B @ np.linalg.solve(A.T @ A + R, A.T @ ytr)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "honest requirement"}
    say("=" * 104)
    say("LOOP 212 -- A REQUIREMENT CALIBRATED ON REAL PREDICTORS, AND EVERY GENE THE BLOCKS REACH")
    say("=" * 104)

    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    S_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    pos = {s: k for k, s in enumerate(allg)}
    seqs = json.load(open(SEQ_F))
    TR = {}
    for name in TRACKS:
        pt, PM = L191.promoter_track(name, [tssb.get(s) for s in sym], L191.PROM_PAD,
                                     lambda *_: None)
        TR[name] = PM[[int(np.where(pt == t)[0][0]) for t in grid]]
    fk = h5py.File(K562, "r")
    cats = [x.decode() if isinstance(x, bytes) else str(x)
            for x in fk["var/__categories/gene_name"][:]]
    readout = np.array([cats[c] for c in fk["var/gene_name"][:]])
    ridx = {g: i for i, g in enumerate(readout)}
    Xk = fk["X"]
    Z = np.load(PHYS_CACHE, allow_pickle=True)
    cpos = {str(g): i for i, g in enumerate(Z["genes"])}

    # ---------------------------------------------------------------- E1
    say("E1 IS THE EXPANDED SET HONEST?")
    inter = [s for s in allg if s in seqs and s in ridx]
    names = [s for s in allg if (s in seqs) or (s in ridx)]
    say(f"     loop 198 scored set                 {len(allg):,}")
    say(f"     all three blocks (loops 210/211)    {len(inter):,}")
    say(f"     at least one block (this loop)      {len(names):,}")
    have_seq = np.array([s in cpos for s in names])
    have_ps = np.array([s in ridx for s in names])
    say(f"       with promoter/motif block  {int(have_seq.sum()):,} ({have_seq.mean():.1%})")
    say(f"       with Perturb-seq block     {int(have_ps.sum()):,} ({have_ps.mean():.1%})")
    say(f"       with ChIP block            {len(names):,} (100.0%, tracks cover the roster)")
    ok1 = (set(inter) <= set(names) and len(names) > len(inter)
           and 0 < have_seq.mean() < 1 and 0 < have_ps.mean() < 1)
    G.add("E1", ok1,
          if_true=lambda: f"E1 PASS -- {len(names):,} genes, a strict superset of the {len(inter):,} "
                          f"intersection, and both missingness indicators are non-constant",
          if_false=lambda: f"E1 FAIL -- {len(names)} vs {len(inter)}, seq {have_seq.mean():.2f}, "
                           f"ps {have_ps.mean():.2f}")

    y = np.array([S_all[pos[s]] for s in names])
    chip_full = np.column_stack([np.column_stack([
        TR[t][:N_TRAIN, gi].mean(0), TR[t][:N_TRAIN, gi].max(0),
        TR[t][N_TRAIN - 1, gi] - TR[t][0, gi]]) for t in TRACKS])
    Fchip = np.array([chip_full[pos[s]] for s in names])

    rng = np.random.default_rng(SEED)
    icols = np.array([ridx[s] for s in names if s in ridx])
    picked = []
    for p in rng.permutation(Xk.shape[0]):
        if len(picked) >= 200:
            break
        v = Xk[int(p), :][icols]
        if np.isfinite(v).all():
            picked.append(int(p))
    G_ps = np.full((len(names), len(picked)), np.nan)
    sub = Xk[np.array(sorted(picked)), :]
    order_map = {p: i for i, p in enumerate(sorted(picked))}
    for k, s in enumerate(names):
        if s in ridx:
            G_ps[k] = sub[[order_map[p] for p in picked], ridx[s]]
    P_phys = np.full((len(names), Z["F"].shape[1]), np.nan)
    for k, s in enumerate(names):
        if s in cpos:
            P_phys[k] = Z["F"][cpos[s]]

    def impute(Fb):
        m = np.isfinite(Fb).all(1)
        out = Fb.copy()
        out[~m] = np.nanmean(Fb[m], axis=0)
        return out, m.astype(float)
    Fgain, m_gain = impute(G_ps)
    Fphys, m_phys = impute(P_phys)
    MISS = np.column_stack([m_gain, m_phys])
    say(f"     feature blocks: chip {Fchip.shape}  gains {Fgain.shape}  physics {Fphys.shape}")

    folds = np.array_split(np.random.default_rng(SEED).permutation(len(y)), 5)

    def cvpred(Feat, lams=(10.0, 100.0, 1000.0, 10000.0, 100000.0), tr_frac=1.0, seed=SEED):
        best = (float("-inf"), None)
        rg = np.random.default_rng(seed)
        for lam in lams:
            Sp = np.zeros(len(y))
            for k in range(5):
                te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
                if tr_frac < 1.0:
                    tr = tr[rg.permutation(len(tr))[:max(30, int(tr_frac * len(tr)))]]
                Sp[te] = ridge(Feat[tr], y[tr], Feat[te], lam)
            r = pear(Sp, y)
            if np.isfinite(r) and abs(r) > best[0]:
                best = (abs(r), Sp)
        return best

    # stacked combination on the expanded set
    r_chip, P_chip = cvpred(Fchip)
    r_gain, P_gain = cvpred(Fgain)
    r_phys, P_phys2 = cvpred(Fphys)
    r_stack, S_stack = cvpred(np.column_stack([P_chip, P_gain, P_phys2, MISS]))
    say(f"     expanded-set solo: chip {r_chip:.4f}  gains {r_gain:.4f}  physics {r_phys:.4f}")
    say(f"     expanded-set stacked (with missingness indicators)  |r| {r_stack:.4f}")

    # ---------------------------------------------------------------- relaxation harness
    def rows(lo, hi):
        yv, prev, dts, gg = [], [], [], []
        for j in range(1, len(grid)):
            if not (lo <= j < hi):
                continue
            dt = grid[j] - grid[j - 1]
            for kk, s in enumerate(names):
                i = gi[pos[s]]
                yv.append(M[j, i] - M[j - 1, i]); prev.append(M[j - 1, i])
                dts.append(dt); gg.append(kk)
        return np.array(yv), np.array(prev), np.array(dts), np.array(gg)
    ytr, ptr, dtr, gtr = rows(1, N_TRAIN)
    yte, pte, dte, gte = rows(N_TRAIN, len(grid))
    pers = r2s(yte, np.zeros_like(yte))
    life = json.load(open(LIFE))["lifetimes"]
    lam_g = np.array([np.log(2) / life[s]["mrna_hl_h"] / 60.0
                      if s in life and life[s].get("mrna_hl_h") else np.nan for s in names])
    lam_g = np.where(np.isfinite(lam_g), lam_g, np.nanmedian(lam_g))
    say(f"     harness: train {len(ytr):,} rows, score {len(yte):,}, persistence {pers:+.5f}")

    def relax(Sp, per_gene=True, shrink=True, rescale=False):
        S = Sp.copy()
        if rescale:
            sd_p, sd_t = np.std(S), np.std(y)
            if sd_p > 0:
                S = S.mean() + (S - S.mean()) * (sd_t / sd_p)
        w = (pear(S, y) ** 2) if shrink else 1.0
        base = lam_g if per_gene else np.ones(len(names))
        d_tr = base[gtr] * dtr * (S[gtr] - ptr)
        d_te = base[gte] * dte * (S[gte] - pte)
        k = float(d_tr @ ytr / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
        return r2s(yte, w * k * d_te)

    # ---------------------------------------------------------------- E2
    say("E2 THE REQUIREMENT, CALIBRATED ON REAL PREDICTORS")
    fam = []
    blocks = {"chip": Fchip, "gains": Fgain, "physics": Fphys, "miss": MISS}
    for nm, Fb in blocks.items():
        for lam in (10.0, 1e3, 1e5, 1e7):
            for frac in (0.15, 0.35, 0.7, 1.0):
                rr, Sp = cvpred(Fb, lams=(lam,), tr_frac=frac, seed=SEED + int(frac * 100))
                if np.isfinite(rr):
                    fam.append((rr, relax(Sp), float(np.std(Sp) / np.std(y)), f"{nm}/{lam:.0e}/{frac}"))
    for lam in (10.0, 1e3, 1e5):
        for frac in (0.3, 0.6, 1.0):
            rr, Sp = cvpred(np.column_stack([P_chip, P_gain, P_phys2, MISS]),
                            lams=(lam,), tr_frac=frac, seed=SEED + 1)
            if np.isfinite(rr):
                fam.append((rr, relax(Sp), float(np.std(Sp) / np.std(y)), f"stack/{lam:.0e}/{frac}"))
    fam.sort()
    say(f"     built {len(fam)} REAL predictors spanning |r| {fam[0][0]:.3f} to {fam[-1][0]:.3f}")
    say("        |r|      R2        dispersion   source")
    for rr, v, disp, tag in fam[::max(1, len(fam)//10)]:
        say(f"       {rr:.4f}  {v:+.5f}   {disp:.3f}      {tag}")
    above = [rr for rr, v, _, _ in fam if v > pers]
    real_cross = min(above) if above else None
    say(f"     real-predictor crossover: "
        f"{'r >= %.4f' % real_cross if real_cross is not None else 'NONE CLEARS PERSISTENCE'}")
    say(f"     noised-oracle crossover (loop 210 C5): {NOISE_CROSS:.4f}")
    diff = abs((real_cross if real_cross is not None else 1.0) - NOISE_CROSS)
    G.add("E2", bool(diff > 0.05), stat=diff, requires=("E1",),
          if_true=lambda: f"E2 PASS -- the real-predictor crossover is "
                          f"{'%.4f' % real_cross if real_cross is not None else 'unreachable'}, "
                          f"differing from the noised-oracle {NOISE_CROSS:.4f} by {diff:.4f}. "
                          f"C6b is confirmed with a proper instrument",
          if_false=lambda: f"E2 FAIL -- real crossover {real_cross} against noised {NOISE_CROSS}; "
                           f"C6b's single point was a fluke and the noise calibration was fine")
    res["family"] = [{"r": a, "r2": b, "disp": c, "tag": d} for a, b, c, d in fam]
    res["crossover"] = {"real": real_cross, "noised": NOISE_CROSS, "persistence": pers}

    # ---------------------------------------------------------------- E3
    say("E3 IS DISPERSION THE MECHANISM?")
    pairs, wins = 0, 0
    for i in range(len(fam)):
        for j in range(i + 1, len(fam)):
            if abs(fam[i][0] - fam[j][0]) <= 0.05 and abs(fam[i][2] - fam[j][2]) > 0.05:
                pairs += 1
                hi = i if fam[i][2] > fam[j][2] else j
                lo = j if hi == i else i
                wins += fam[hi][1] > fam[lo][1]
    frac_w = wins / pairs if pairs else float("nan")
    say(f"     matched pairs (|dr| <= 0.05, |d dispersion| > 0.05): {pairs:,}")
    say(f"     the MORE dispersed predictor scores the higher R2 in {wins:,} = {frac_w:.1%}")
    G.add("E3", bool(frac_w > 0.5), stat=frac_w, requires=("E2",),
          if_true=lambda: f"E3 PASS -- dispersion decides at matched r in {frac_w:.0%} of pairs, "
                          f"so C6b's stated mechanism holds",
          if_false=lambda: f"E3 FAIL -- {frac_w:.0%}; dispersion is not the mechanism and C6b's "
                           f"explanation is wrong even though its observation stands")

    # ---------------------------------------------------------------- E4
    say("E4 DOES VARIANCE CORRECTION RECOVER ANYTHING?")
    plain = relax(S_stack, rescale=False)
    fixed = relax(S_stack, rescale=True)
    say(f"       uncorrected  R2 {plain:+.5f}   dispersion {np.std(S_stack)/np.std(y):.4f}")
    say(f"       rescaled     R2 {fixed:+.5f}   dispersion 1.0000")
    say(f"       persistence  R2 {pers:+.5f}")
    G.add("E4", bool(fixed - plain > 0.01), stat=fixed - plain, requires=("E3",),
          if_true=lambda: f"E4 PASS -- matching the variance buys {fixed-plain:+.5f}",
          if_false=lambda: f"E4 FAIL -- rescaling buys {fixed-plain:+.5f}. Under-dispersion is "
                           f"diagnostic of the problem but correcting it is not the fix")

    # ---------------------------------------------------------------- E5
    say("E5 DOES THE EXPANDED SET HELP?")
    ratio = len(names) / max(len(inter), 1)
    say(f"     genes {len(inter):,} -> {len(names):,}  ({ratio:.2f}x)")
    say(f"     combined |r| on 600 was {R_600:.4f}; on the expanded set {r_stack:.4f} "
        f"({r_stack-R_600:+.4f})")
    G.add("E5", bool(ratio >= 1.5 and abs(r_stack - R_600) <= 0.05), stat=ratio, requires=("E1",),
          if_true=lambda: f"E5 PASS -- {ratio:.2f}x the genes at |r| {r_stack:.4f}, undiluted",
          if_false=lambda: f"E5 FAIL -- {ratio:.2f}x genes, |r| {r_stack:.4f} against {R_600:.4f}")

    # ---------------------------------------------------------------- E6
    say("E6 DOES ANYTHING FINALLY BEAT PERSISTENCE?")
    best = max(plain, fixed)
    say(f"       best configuration  R2 {best:+.5f}   persistence {pers:+.5f}   "
        f"margin {best-pers:+.5f}")
    G.add("E6", bool(best - pers > 0.01), stat=best, requires=("E5",),
          if_true=lambda: f"E6 PASS -- clears persistence by {best-pers:+.5f}",
          if_false=lambda: f"E6 FAIL -- clears persistence by only {best-pers:+.5f}. Six loops of "
                           f"better set points, a correctly shrunk rate, per-gene half-lives and "
                           f"{len(names):,} genes, and the model still does not move")
    res["final"] = {"plain": plain, "rescaled": fixed, "persistence": pers, "margin": best - pers,
                    "r_stack": r_stack, "n_genes": len(names)}

    # ---------------------------------------------------------------- E7
    say("E7 IS ANY OF IT FAME OR MISSINGNESS?")
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    pubs = {str(g["name"]).upper(): float(g.get("pubs") or 0) for g in tab}
    r_fame, _ = cvpred(np.log1p(np.array([pubs.get(s, 0.0) for s in names])).reshape(-1, 1))
    r_miss, _ = cvpred(MISS)
    say(f"       publication count      |r| {r_fame:.4f}")
    say(f"       missingness indicators |r| {r_miss:.4f}")
    say(f"       stacked model          |r| {r_stack:.4f}")
    G.add("E7", bool(r_stack > r_fame and r_stack > r_miss), stat=r_miss, requires=("E5",),
          if_true=lambda: f"E7 PASS -- beats fame {r_fame:.4f} and missingness {r_miss:.4f}",
          if_false=lambda: f"E7 FAIL -- fame {r_fame:.4f}, missingness {r_miss:.4f}")

    say("E8 WHAT THIS CANNOT SHOW")
    say("     Mean-imputing a missing block gives that gene the population's answer for it. That")
    say("     is conservative for prediction and it is NOT the same as measuring the block, so")
    say("     the expanded set is larger and shallower than the intersection, by construction.")
    say("     The real-predictor family is built from THESE blocks, so its error geometry is")
    say("     representative of ridge fits on this feature set and not of predictors in general.")
    say("     A stronger predictor built some other way could sit off this curve.")
    say("     Everything is still one cell line, one perturbation and one channel. Loop 197")
    say("     established there is no second densely-sampled matched course in ENCODE, so none")
    say("     of this can be replicated on a second target.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
