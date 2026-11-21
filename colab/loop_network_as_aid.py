"""
LOOP 258 -- THE NETWORK AS AN AID, NOT AS A STAGE EVERYTHING MUST PASS THROUGH

Loop 256 measured the cost of the other design. Every prediction was forced through the
network, so the network had to re-derive the 978-landmark profile that the additive
baseline already carried, and it lost by -0.1517. Its LINEAR twin on identical inputs
also lost, by -0.0494, which says the framing was wrong rather than the capacity.

So here the additive baseline is the standing answer and the network only ever proposes a
CORRECTION to it, with a blend weight alpha that is fitted on a genuinely held-out cell
line. If the network does not transfer across lines, alpha goes to zero and the baseline
stands untouched. The network is consulted; it is not in the path.

    prediction = additive_baseline + alpha * correction

THE TRAP IN THIS DESIGN, STATED BEFORE ANY NUMBER:
    Fitting alpha makes an improvement nearly guaranteed. A blend weight fitted on any
    correction, including pure noise, buys a small gain for free, because alpha can always
    shrink a useless correction to almost nothing and can always exploit whatever chance
    alignment exists. So "0.4518 -> 0.46" would be a FAKE win and will not be reported as
    a real one. J2 below is therefore declared a MACHINERY check that cannot meaningfully
    fail, and is NOT counted as evidence for anything.

    The load-bearing gate is J3: the real correction against the SAME correction with its
    rows permuted, blended through the identical alpha-fitting pipeline. The permuted arm
    has the exact same magnitudes, the exact same distribution, and the wrong pairing. Any
    gain the real one has over that one is the only gain that means anything.

WHERE ALPHA IS FITTED, AND WHY IT IS NOT LEAKAGE:
    For each outer held-out line L, one of the eight remaining lines is set aside as the
    calibration line V. The network trains on the other SEVEN. Alpha is fitted on V, which
    the network never saw, and then applied to L, which neither the network nor alpha saw.
    Training on seven lines rather than eight makes the network slightly weaker than loop
    257's; that is the price of an honestly fitted blend weight and it is paid knowingly.

WHAT THIS LOOP IS ACTUALLY FOR:
    Loops 253 and 255 asked what explains the gene x line interaction and found nine nulls.
    Loop 256 asked whether a network beats arithmetic on average and it does not. This asks
    a different and better question: is there ANY stratum where the network adds real
    signal, measured against the permuted null. A correction that helps on 12% of pairs and
    abstains on the rest is a genuine result; an average is not the only way to win.

GATES, ALL DECLARED BEFORE THE RUN:

  J1 DOES THE HARNESS REPRODUCE THE BASELINE?
     The additive baseline here against loop 252's 0.4477.
     Gate: PASS iff within 0.02.

  J2 MACHINERY ONLY -- DOES THE AIDED MODEL AT LEAST NOT LOSE?      -- requires J1
     This gate CANNOT MEANINGFULLY FAIL and is not evidence. It exists to catch a broken
     alpha fit, nothing else. A PASS here says the plumbing works. It does not say the
     network helped, and it must never be quoted as if it did.
     Gate: PASS iff aided >= baseline - 0.002.

  J3 LOAD-BEARING -- DOES THE REAL CORRECTION BEAT ITS OWN PERMUTATION?   -- requires J1
     The identical correction matrix with its ROWS permuted, so magnitudes and
     distribution are exactly preserved and only the pairing to the target is destroyed,
     pushed through the same alpha fit on the same calibration line.
     Gate: PASS iff real exceeds permuted by at least 0.005.
     A FAIL means every gain came from fitting a blend weight, and the network contributed
     nothing that a random correction of the same size would not have contributed.

  J4 IS THE BLEND WEIGHT DISTINGUISHABLE FROM ZERO?                 -- requires J1
     Alpha across the 9 outer folds, mean against its own standard error.
     Gate: PASS iff |mean alpha| > 2 * se. A FAIL means the calibration line itself could
     not decide the network was worth listening to.

  J5 DOES THE NETWORK CORRECTION BEAT A LINEAR CORRECTION?          -- requires J1
     A ridge on the identical inputs, producing a correction, blended by the identical
     alpha pipeline. This isolates the network from the aid FRAMING: if the framing is
     what helps, the linear correction gets the same benefit.
     Gate: PASS iff the network exceeds the linear correction by at least 0.005.

  J6 IS THERE A STRATUM WHERE IT HELPS, EVEN IF THE AVERAGE IS FLAT?  -- requires J1
     Alpha fitted per decile of correction magnitude on V and clipped at zero, so the
     model ABSTAINS where the calibration line says the correction is not worth using.
     Reported against the permuted null stratified identically, because deciles fitted on
     noise also buy a free gain and that gain is the thing to beat.
     Gate: PASS iff stratified real exceeds stratified permuted by at least 0.005.

  J7 CONTROL: THE WRONG CELL LINE                                   -- requires J3
     VOID if J3 found no margin, because there is then nothing to collapse.
     Gate: PASS iff feeding the hypernetwork another line's measured properties retains at
     most 25% of J3's margin.

  J8 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, copy, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import lincs_harness as H
from gate_guard import Gates
from loop_gated_operator import GatedOperator, NPC_EXPR, NPC_DEP, KEXP, EPOCHS, PATIENCE, LR, BATCH

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_network_as_aid.json"
SEED, SEEDS = 258258, [0, 1, 2]
LOOP252_ADDITIVE = 0.4477
J1_TOL, J2_FLOOR, J3_BAR, J5_BAR, J6_BAR, J7_MAX = 0.02, -0.002, 0.005, 0.005, 0.005, 0.25
NDEC = 10
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def fit_alpha(Y, A, C):
    """Least squares blend weight: argmin ||Y - (A + aC)||^2.  Closed form, and fitted on
    the RESIDUAL rather than on the evaluation metric, so it cannot chase the score."""
    R = (Y - A).ravel()
    c = C.ravel()
    d = float(c @ c)
    return 0.0 if d <= 0 else float((R @ c) / d)


def fit_alpha_strata(Y, A, C, edges):
    """One alpha per decile of correction magnitude, clipped at zero so the model abstains
    rather than anti-predicts where the calibration line says the correction is useless."""
    mag = np.linalg.norm(C, axis=1)
    out = []
    for k in range(len(edges) - 1):
        m = (mag >= edges[k]) & (mag < edges[k + 1] if k < len(edges) - 2 else mag <= edges[k + 1])
        out.append(max(0.0, fit_alpha(Y[m], A[m], C[m])) if m.sum() >= 30 else 0.0)
    return np.array(out)


def apply_strata(C, alphas, edges):
    mag = np.linalg.norm(C, axis=1)
    a = np.zeros(len(C), np.float32)
    for k in range(len(edges) - 1):
        m = (mag >= edges[k]) & (mag < edges[k + 1] if k < len(edges) - 2 else mag <= edges[k + 1])
        a[m] = alphas[k]
    return C * a[:, None]


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "the network as an aid to the additive baseline, not a stage in the path"}
    say("=" * 104)
    say("LOOP 258 -- THE NETWORK AS AN AID, NOT A STAGE EVERYTHING MUST PASS THROUGH")
    say("=" * 104)
    say("     Loop 256 forced every prediction through the network and it lost by -0.1517;")
    say("     its LINEAR twin on identical inputs also lost, by -0.0494, so the framing was")
    say("     wrong rather than the capacity. Here the baseline is the standing answer and the")
    say("     network only proposes a correction, weighted by an alpha fitted on a held-out line.")
    say("     DECLARED BEFORE THE RUN: fitting alpha makes a small gain nearly automatic, so J2")
    say("     is machinery and NOT evidence. J3 -- real correction against its own row")
    say("     permutation, same magnitudes, wrong pairing -- is the only gate that means anything.")

    D = H.load()
    Pm, pg, pc, LINES, NL = D["Pm"], D["pg"], D["pc"], D["LINES"], D["NL"]
    say(f"     {len(pg):,} (gene, line) pairs, {len(D['genes']):,} genes, {NL} landmarks")

    lmap = json.load(open(SCR / "lincs" / "line_map.json"))
    ez = np.load(SCR / "depmap_expr_aligned.npz", allow_pickle=True)
    XE = ez["XE"]; el = np.array([str(x) for x in ez["lines"]])
    U, sv, _ = np.linalg.svd(XE - XE.mean(0), full_matrices=False)
    EPC = U[:, :NPC_EXPR] * sv[:NPC_EXPR]
    ge = np.load(SCR / "depmap" / "gene_effect.npz", allow_pickle=True)
    GE = np.nan_to_num(np.asarray(ge["E"], np.float32)); gl = np.array([str(x) for x in ge["lines"]])
    U2, sv2, _ = np.linalg.svd(GE - GE.mean(0), full_matrices=False)
    DPC = U2[:, :NPC_DEP] * sv2[:NPC_DEP]
    burden = {}
    with open(SCR / "depmap" / "OmicsSomaticMutationsMatrixDamaging.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            burden[row[0]] = float(sum(1 for v in row[1:] if v not in ("", "0", "0.0")))
    ep_ = {l: int(np.where(el == lmap[l])[0][0]) for l in LINES}
    dp_ = {l: int(np.where(gl == lmap[l])[0][0]) for l in LINES}
    LF = np.stack([np.concatenate([EPC[ep_[l]], DPC[dp_[l]],
                                   [np.log1p(burden.get(lmap[l], 0.0))]]) for l in LINES])
    LF = ((LF - LF.mean(0)) / (LF.std(0) + 1e-6)).astype(np.float32)
    li = {l: i for i, l in enumerate(LINES)}
    say(f"     context vector: {LF.shape[1]} measured dims, PCs fitted on the full DepMap panel")
    say(f"     alpha is fitted on a CALIBRATION line the network never trained on, then applied")
    say(f"     to the outer held-out line, which neither the network nor alpha ever saw")

    def build(fit_lines, target_mask, src=None):
        """Rows for a set of lines (training) or a mask (evaluation)."""
        gm, tr = {}, np.isin(pc, list(fit_lines))
        for g in D["genes"]:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        grand = Pm[tr].mean(0)
        lmean = {l: Pm[pc == l].mean(0) for l in LINES}

        def rows(mask, source=None):
            Xg, Xl, Y, A = [], [], [], []
            for j in np.where(mask)[0]:
                g = pg[j]
                if g not in gm: continue
                c = source if source else pc[j]
                Xg.append(gm[g]); Xl.append(LF[li[c]]); Y.append(Pm[j])
                # DEFECT I: the substitute line feeds the MODEL's inputs only. The standing
                # answer keeps the TRUE line, so the wrong-line control moves ONE thing.
                A.append(gm[g] + lmean[pc[j]] - grand)
            return tuple(np.stack(v).astype(np.float32) for v in (Xg, Xl, Y, A))
        return rows, tr

    def sc(P, Y):
        return np.array([H.pear(P[i], Y[i]) for i in range(len(Y))])

    def train_net(Xg, Xl, R, seed, shuffle_line=False):
        torch.manual_seed(seed)
        r2 = np.random.default_rng(seed)
        ip = r2.permutation(len(Xg)); nv = max(200, int(0.12 * len(Xg)))
        va, fi = ip[:nv], ip[nv:]
        net = GatedOperator(NL, LF.shape[1], NL, KEXP)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        tg, tl, tr_ = (torch.from_numpy(Xg[fi]), torch.from_numpy(Xl[fi]), torch.from_numpy(R[fi]))
        vg, vl, vr = (torch.from_numpy(Xg[va]), torch.from_numpy(Xl[va]), torch.from_numpy(R[va]))
        best, bad, bw = 9e9, 0, None
        for _ in range(EPOCHS):
            idx = torch.randperm(len(fi))
            for b0 in range(0, len(fi), BATCH):
                j = idx[b0:b0 + BATCH]
                opt.zero_grad()
                ((net(tg[j], tl[j]) - tr_[j]) ** 2).mean().backward(); opt.step()
            with torch.no_grad():
                v = float(((net(vg, vl) - vr) ** 2).mean())
            if v < best - 1e-8: best, bad, bw = v, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= PATIENCE: break
        if bw: net.load_state_dict(bw)
        net.eval()
        return net

    def predict(net, Xg, Xl):
        with torch.no_grad():
            return net(torch.from_numpy(Xg), torch.from_numpy(Xl)).numpy()

    def run(seed, shuffle_line=False):
        """One full leave-one-line-out sweep. Returns every arm's per-pair scores."""
        out = {k: [] for k in ("base", "aided", "perm", "strat", "strat_perm",
                               "lin", "forced")}
        alphas, per_line = [], {}
        rng = np.random.default_rng(seed)
        for oi, hold in enumerate(LINES):
            calib = LINES[(oi + 1) % len(LINES)]
            if calib == hold: calib = LINES[(oi + 2) % len(LINES)]
            train = [l for l in LINES if l not in (hold, calib)]
            rows, _ = build(train, None)
            Xg, Xl, Y, A = rows(np.isin(pc, train))
            Xgc, Xlc, Yc, Ac = rows(pc == calib)
            src = (str(rng.choice([l for l in LINES if l != hold])) if shuffle_line else None)
            Xgt, Xlt, Yt, At = rows(pc == hold, source=src)

            net = train_net(Xg, Xl, Y - A, seed)
            Cc, Ct = predict(net, Xgc, Xlc), predict(net, Xgt, Xlt)

            # linear correction on identical inputs, same pipeline
            Z = np.concatenate([Xg, Xl, np.ones((len(Xg), 1), np.float32)], 1)
            M = Z.T @ Z + 1e-2 * len(Z) * np.eye(Z.shape[1], dtype=np.float32)
            b = np.linalg.solve(M, Z.T @ (Y - A))
            Lc = np.concatenate([Xgc, Xlc, np.ones((len(Xgc), 1), np.float32)], 1) @ b
            Lt = np.concatenate([Xgt, Xlt, np.ones((len(Xgt), 1), np.float32)], 1) @ b

            # the permuted null: identical magnitudes and distribution, wrong pairing
            pc_ = rng.permutation(len(Cc)); pt_ = rng.permutation(len(Ct))
            Pc, Pt = Cc[pc_], Ct[pt_]

            a_real = fit_alpha(Yc, Ac, Cc)
            a_perm = fit_alpha(Yc, Ac, Pc)
            a_lin = fit_alpha(Yc, Ac, Lc)
            alphas.append(a_real)

            mag = np.linalg.norm(Cc, axis=1)
            edges = np.quantile(mag, np.linspace(0, 1, NDEC + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            s_real = fit_alpha_strata(Yc, Ac, Cc, edges)
            magp = np.linalg.norm(Pc, axis=1)
            ep = np.quantile(magp, np.linspace(0, 1, NDEC + 1)); ep[0], ep[-1] = -np.inf, np.inf
            s_perm = fit_alpha_strata(Yc, Ac, Pc, ep)

            arms = {"base": At,
                    "aided": At + a_real * Ct,
                    "perm": At + a_perm * Pt,
                    "strat": At + apply_strata(Ct, s_real, edges),
                    "strat_perm": At + apply_strata(Pt, s_perm, ep),
                    "lin": At + a_lin * Lt,
                    "forced": At + Ct}
            for k, P in arms.items():
                out[k].append(sc(P, Yt))
            per_line[hold] = {"alpha": a_real,
                              "base": float(np.nanmean(sc(At, Yt))),
                              "aided": float(np.nanmean(sc(arms["aided"], Yt))),
                              "strata": [float(x) for x in s_real]}
        return ({k: np.concatenate(v) for k, v in out.items()},
                np.array(alphas), per_line)

    say(f"     9 outer folds x {len(SEEDS)} seeds; network trains on 7 lines, alpha on the 8th ...")
    R, AL, PL = {}, {}, None
    for sd in SEEDS:
        r_, a_, pl_ = run(sd)
        R[sd], AL[sd] = r_, a_
        if PL is None: PL = pl_
        say(f"       seed {sd}: base {np.nanmean(r_['base']):.4f}  aided {np.nanmean(r_['aided']):.4f}"
            f"  permuted {np.nanmean(r_['perm']):.4f}  forced {np.nanmean(r_['forced']):.4f}"
            f"  [{time.time() - t0:.0f}s]")
    A0 = R[SEEDS[0]]
    say(f"       stratified real {np.nanmean(A0['strat']):.4f}   "
        f"stratified permuted {np.nanmean(A0['strat_perm']):.4f}   "
        f"linear correction {np.nanmean(A0['lin']):.4f}")
    res["arms"] = {k: float(np.nanmean(A0[k])) for k in A0}
    res["arms"]["aided_mean_over_seeds"] = float(np.mean([np.nanmean(R[s]["aided"]) for s in SEEDS]))
    res["alpha"] = {str(s): [float(x) for x in AL[s]] for s in SEEDS}
    res["per_line"] = PL

    say("J1 DOES THE HARNESS REPRODUCE THE BASELINE?")
    a1 = float(np.nanmean(A0["base"]))
    say(f"     additive here {a1:.4f} against loop 252's {LOOP252_ADDITIVE:.4f}")
    G.add("J1", bool(abs(a1 - LOOP252_ADDITIVE) <= J1_TOL), stat=a1,
          if_true=lambda: f"J1 PASS -- reproduces to {abs(a1 - LOOP252_ADDITIVE):.4f}",
          if_false=lambda: f"J1 FAIL -- {a1:.4f} against {LOOP252_ADDITIVE:.4f}")

    say("J2 MACHINERY ONLY -- DOES THE AIDED MODEL AT LEAST NOT LOSE?")
    d2, se2, z2 = H.paired(A0["aided"], A0["base"])
    say(f"     aided {np.nanmean(A0['aided']):.4f} vs baseline {a1:.4f}   {d2:+.4f} +/- {se2:.4f}")
    say(f"     THIS GATE IS NOT EVIDENCE. A fitted blend weight makes a small gain nearly")
    say(f"     automatic; it exists only to catch a broken alpha fit. J3 decides the science.")
    G.add("J2", bool(d2 >= J2_FLOOR), stat=float(d2), requires=("J1",),
          if_true=lambda: f"J2 PASS (machinery) -- the aid is worth {d2:+.4f}, inside the "
                          f"{J2_FLOOR} machinery floor. This says the plumbing works and says "
                          f"NOTHING about whether the network helped.",
          if_false=lambda: f"J2 FAIL (machinery) -- {d2:+.4f} below a floor a fitted alpha should "
                           f"make unreachable; the alpha fit is broken, not the science")
    res["J2"] = {"delta": d2, "se": se2, "z": z2, "machinery_only": True}

    say("J3 LOAD-BEARING -- DOES THE REAL CORRECTION BEAT ITS OWN ROW PERMUTATION?")
    d3, se3, z3 = H.paired(A0["aided"], A0["perm"])
    say(f"     real {np.nanmean(A0['aided']):.4f} vs permuted {np.nanmean(A0['perm']):.4f}   "
        f"{d3:+.4f} +/- {se3:.4f} ({z3:+.1f} se)")
    say(f"     the permuted arm has identical magnitudes and distribution and the wrong pairing,")
    say(f"     so whatever it scores is the free lunch from fitting a blend weight at all")
    G.add("J3", bool(d3 >= J3_BAR), stat=float(d3), requires=("J1",),
          if_true=lambda: f"J3 PASS -- the network's correction is worth {d3:+.4f} over a "
                          f"same-sized random one",
          if_false=lambda: f"J3 FAIL -- {d3:+.4f} over its own permutation. Every gain came from "
                           f"fitting alpha; the network contributed nothing a random correction "
                           f"of the same size would not have.")
    res["J3"] = {"delta": d3, "se": se3, "z": z3}

    say("J4 IS THE BLEND WEIGHT DISTINGUISHABLE FROM ZERO?")
    av = AL[SEEDS[0]]
    m4, s4 = float(av.mean()), float(av.std(ddof=1) / np.sqrt(len(av)))
    say(f"     alpha per fold: " + ", ".join(f"{x:+.3f}" for x in av))
    say(f"     mean {m4:+.4f} +/- {s4:.4f}")
    G.add("J4", bool(abs(m4) > 2 * s4), stat=m4, requires=("J1",),
          if_true=lambda: f"J4 PASS -- alpha is {abs(m4) / max(s4, 1e-9):.1f} se from zero",
          if_false=lambda: f"J4 FAIL -- alpha {m4:+.4f} +/- {s4:.4f}; the calibration line could "
                           f"not decide the network was worth listening to")
    res["J4"] = {"mean": m4, "se": s4}

    say("J5 DOES THE NETWORK CORRECTION BEAT A LINEAR CORRECTION?")
    d5, se5, z5 = H.paired(A0["aided"], A0["lin"])
    say(f"     network-aided {np.nanmean(A0['aided']):.4f} vs linear-aided "
        f"{np.nanmean(A0['lin']):.4f}   {d5:+.4f} +/- {se5:.4f} ({z5:+.1f} se)")
    G.add("J5", bool(d5 >= J5_BAR), stat=float(d5), requires=("J1",),
          if_true=lambda: f"J5 PASS -- the network is worth {d5:+.4f} over a ridge given the aid",
          if_false=lambda: f"J5 FAIL -- the network is worth {d5:+.4f} over a ridge; the AID "
                           f"framing is doing the work, not the network")
    res["J5"] = {"delta": d5, "se": se5, "z": z5}

    say("J6 IS THERE A STRATUM WHERE IT HELPS, EVEN IF THE AVERAGE IS FLAT?")
    d6, se6, z6 = H.paired(A0["strat"], A0["strat_perm"])
    used = [sum(1 for a in PL[l]["strata"] if a > 0) for l in LINES]
    say(f"     stratified real {np.nanmean(A0['strat']):.4f} vs stratified permuted "
        f"{np.nanmean(A0['strat_perm']):.4f}   {d6:+.4f} +/- {se6:.4f} ({z6:+.1f} se)")
    say(f"     deciles with a non-zero alpha, per fold: " + ", ".join(str(u) for u in used)
        + f"   (out of {NDEC})")
    G.add("J6", bool(d6 >= J6_BAR), stat=float(d6), requires=("J1",),
          if_true=lambda: f"J6 PASS -- abstaining where the calibration line says so is worth "
                          f"{d6:+.4f} over the same procedure run on noise",
          if_false=lambda: f"J6 FAIL -- {d6:+.4f} over deciles fitted on noise; per-stratum "
                           f"abstention does not find a subset the network genuinely helps on")
    res["J6"] = {"delta": d6, "se": se6, "z": z6, "deciles_used": used}

    say("J7 CONTROL: THE WRONG CELL LINE")
    if d3 < J3_BAR:
        G.add("J7", False, stat=float(d3), requires=("J3",), void_if=True,
              void_reason=f"J3's margin is {d3:+.4f}; there is nothing to collapse")
    else:
        sh, _, _ = run(SEEDS[0], shuffle_line=True)
        d7, _, _ = H.paired(sh["aided"], sh["perm"])
        f7 = d7 / d3
        say(f"     hypernetwork fed another line's properties: {d7:+.4f} against a real "
            f"{d3:+.4f} ({f7:.0%})")
        G.add("J7", bool(f7 <= J7_MAX), stat=float(f7), requires=("J3",),
              if_true=lambda: f"J7 PASS -- collapses to {f7:.0%} on the wrong line",
              if_false=lambda: f"J7 FAIL -- {f7:.0%} survives the wrong line's properties")
        res["J7"] = {"real": d3, "shuffled": d7, "fraction": f7}

    say("J8 WHAT THIS CANNOT SHOW")
    say("     J2 passing is not a result. A fitted blend weight buys a gain from any correction,")
    say("     including the permuted one, and the permuted arm's own score is printed above so")
    say("     that free lunch is visible rather than absorbed into a headline.")
    say("     The network trains on SEVEN lines here, not eight, so it is slightly weaker than")
    say("     loop 257's. That is the price of fitting alpha on a line it never saw.")
    say("     Nine lines means the hypernetwork still generalises from seven examples; J7 is the")
    say("     only thing standing between a positive and memorisation.")
    say("     A stratum where the network helps names WHERE the interaction is learnable. It does")
    say("     not name WHAT the interaction is, and nothing here recovers the mechanism loops")
    say("     253 and 255 failed to find.")
    say("     978 landmarks and shRNA, not a transcriptome and not a clean knockout.")

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
