"""
LOOP 259 -- A BRANCHING NETWORK: EVERY BRANCH CONTRIBUTES, NONE IS A BOTTLENECK

Loop 257 kept the network out of the OUTPUT path (it predicted baseline + correction, and
that recovered 0.30 -> 0.44). But it left a bottleneck INSIDE the correction: the gene
profile enters through 978 -> 256 -> 64 and everything the model can say has to fit
through those 64 numbers. Every correction it can express is rank <= 64 in the gene
direction, and there is no route from the gene profile to the output that bypasses them.

Here the correction is a SUM OF PARALLEL BRANCHES. Each branch reads its own view, each
writes its own full 978-dim contribution, and none has to carry any other branch's
information:

    prediction = additive_baseline + sum_b  alpha_b * Branch_b

Two of the five branches have NO bottleneck at all -- they are elementwise in 978 dims.
Gradients reach every branch directly rather than through a shared narrow layer.

  B1  DIAGONAL GAIN            g * xg                     978 params, no bottleneck
      "this landmark systematically over- or under-shoots what the gene mean predicts"
  B2  CONTEXT-SCALED GAIN      (W2 z_c) * xg              per-landmark scale that VARIES
      by cell line. Genuinely three-way (gene x landmark x line), which loop 255 showed
      is the only kind of feature a within-profile correlation can even detect.
  B3  LOW-RANK OPERATOR        sum_k g_k(z_c) (e M_k)     loop 257's model, kept intact so
      the branching version strictly CONTAINS the bottlenecked one.
  B4  TRIPLE PRODUCT           c * xg * xlm               978 params, no bottleneck. Gene
      deviation times line deviation, elementwise: interaction without any projection.
  B5  CONTEXT OFFSET           W5 z_c                     a per-landmark line effect beyond
      what the line mean already removed.

All branches are zero-initialised, so the model starts EXACTLY at the additive baseline.

ALPHA IS FITTED ON A HELD-OUT LINE, PER BRANCH.
Train on 7 lines, fit the five alphas by least squares on the 8th (the calibration line),
evaluate on the 9th. A branch that does not transfer across cell lines gets alpha near
zero and stops contributing. Alphas are NOT clipped: a negative alpha means the branch
anti-predicts out of sample, and that is a finding, not something to hide.

WHAT LOOP 257 MEASURED THAT CONSTRAINS WHAT THIS CAN DO -- STATED BEFORE ANY NUMBER:
The gated operator removed 19% of validation MSE and then LOST 0.030 on a held-out line.
That is negative transfer, not underfitting. Adding branches adds capacity, and capacity
was never the binding constraint. This design can help in exactly two ways and no others:
  (a) the elementwise branches can express corrections the rank-64 bottleneck destroyed;
  (b) per-branch alphas fitted out of sample can switch OFF whatever does not transfer.
It cannot manufacture context information that eight training lines do not contain. If
only the context-FREE branches (B1, B4 without its line term surviving) carry weight, the
honest conclusion is that there is a gene-level correction worth making and context still
contributes nothing -- and that is declared here as an acceptable outcome, not a failure
to be explained away.

THE SAME TRAP AS LOOP 258, RESTATED: fitting alphas makes a small gain nearly automatic.
K2 is MACHINERY ONLY and is not evidence. K3, against a row-permuted null pushed through
the identical alpha fit, is the gate that decides whether anything real happened.

GATES, ALL DECLARED BEFORE THE RUN:

  K1 DOES THE HARNESS REPRODUCE THE BASELINE?
     Gate: PASS iff the additive baseline is within 0.02 of loop 252's 0.4477.

  K2 MACHINERY ONLY -- DOES THE BRANCHING MODEL AT LEAST NOT LOSE?     -- requires K1
     CANNOT MEANINGFULLY FAIL. Catches a broken alpha fit and nothing else. Never quote it.
     Gate: PASS iff branching >= baseline - 0.002.

  K3 LOAD-BEARING -- DOES IT BEAT ITS OWN ROW PERMUTATION?             -- requires K1
     Every branch output with its ROWS permuted: identical magnitudes, identical
     distribution, wrong pairing, same alpha fit on the same calibration line.
     Gate: PASS iff real exceeds permuted by at least 0.005.

  K4 THE USER'S HYPOTHESIS -- DOES BRANCHING BEAT THE BOTTLENECK?      -- requires K1
     The full branching model against B3 ALONE, which is loop 257's bottlenecked operator,
     each TRAINED SEPARATELY and each given the identical alpha treatment. This is the
     gate that tests whether removing the bottleneck is what mattered.
     Gate: PASS iff branching exceeds B3-alone by at least 0.005.

  K5 WHICH BRANCHES SURVIVE OUT OF SAMPLE?                             -- requires K1
     Mean alpha per branch across the 9 folds against its own standard error.
     Gate: PASS iff at least one branch has |mean alpha| > 2 se. Reported per branch
     either way, including negative alphas.

  K6 DOES CONTEXT CONTRIBUTE, OR ONLY GENE-LEVEL STRUCTURE?            -- requires K1
     The context-reading branches (B2, B3, B5) against the context-free ones (B1, B4),
     both alpha-fitted. Loop 255 proved line identity's out-of-sample ridge coefficient is
     exactly zero and loops 253/255 found nine annotation nulls, so a FAIL here is the
     expected outcome and is recorded as confirmation, not as a disappointment.
     Gate: PASS iff adding the context branches to the context-free ones is worth 0.005.

  K7 CONTROL: THE WRONG CELL LINE                                      -- requires K3
     VOID if K3 found no margin. Gate: PASS iff at most 25% of K3's margin survives.

  K8 TRAINABILITY, AND WHAT THIS CANNOT SHOW
     Defect H: every arm reports the fraction of validation MSE it removed, and any gate
     whose comparison arm did not train is VOID rather than FAIL.
"""
import json, time, copy, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import lincs_harness as H
from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_branching.json"
SEED, SEEDS = 259259, [0, 1, 2]
NPC_EXPR, NPC_DEP, GDIM, KEXP, HID = 50, 50, 64, 8, 256
EPOCHS, PATIENCE, LR, BATCH = 50, 7, 1e-3, 256
LOOP252_ADDITIVE = 0.4477
K1_TOL, K2_FLOOR, K3_BAR, K4_BAR, K6_BAR, K7_MAX = 0.02, -0.002, 0.005, 0.005, 0.005, 0.25
TRAIN_FLOOR = 0.02
BNAMES = ["B1_diag_gain", "B2_ctx_gain", "B3_lowrank_op", "B4_triple", "B5_ctx_offset"]
CTX_BRANCHES = [1, 2, 4]
FREE_BRANCHES = [0, 3]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


class BranchNet(nn.Module):
    """Five parallel contributions summed. No branch passes through another, and B1/B4 have
    no bottleneck at all. Every branch is zero-initialised so the sum starts at zero and the
    model begins exactly at the additive baseline."""

    def __init__(s, nl, lin, use=None):
        super().__init__()
        s.use = list(range(5)) if use is None else list(use)
        s.b1 = nn.Parameter(torch.zeros(nl))
        s.b2 = nn.Linear(lin, nl); nn.init.zeros_(s.b2.weight); nn.init.zeros_(s.b2.bias)
        s.gene = nn.Sequential(nn.Linear(nl, HID), nn.ReLU(), nn.Linear(HID, GDIM))
        s.hyper = nn.Sequential(nn.Linear(lin, 64), nn.ReLU(), nn.Linear(64, KEXP))
        s.M = nn.Parameter(torch.zeros(KEXP, GDIM, nl)); nn.init.normal_(s.M, std=1e-3)
        s.b4 = nn.Parameter(torch.zeros(nl))
        s.b5 = nn.Linear(lin, nl); nn.init.zeros_(s.b5.weight); nn.init.zeros_(s.b5.bias)

    def branches(s, xg, xl, xlm):
        out = []
        for i in range(5):
            if i not in s.use:
                out.append(torch.zeros_like(xg)); continue
            if i == 0:
                out.append(s.b1 * xg)
            elif i == 1:
                out.append(s.b2(xl) * xg)
            elif i == 2:
                e = s.gene(xg)
                o = torch.einsum("bd,kdo->bko", e, s.M)
                g = torch.sigmoid(s.hyper(xl))
                out.append((o * g[:, :, None]).sum(1))
            elif i == 3:
                out.append(s.b4 * xg * xlm)
            else:
                out.append(s.b5(xl))
        return out

    def forward(s, xg, xl, xlm):
        return sum(s.branches(xg, xl, xlm))


def fit_alphas(Y, A, Bs, ridge=1e-6):
    """One weight per branch, least squares on the RESIDUAL so it cannot chase the score.
    Not clipped: a negative alpha means that branch anti-predicts out of sample, which is a
    finding rather than something to suppress."""
    R = (Y - A).ravel().astype(np.float64)
    X = np.stack([b.ravel().astype(np.float64) for b in Bs], 1)
    Gm = X.T @ X + ridge * np.eye(X.shape[1])
    try:
        return np.linalg.solve(Gm, X.T @ R)
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "a branching additive network: every branch contributes, none is a bottleneck"}
    say("=" * 104)
    say("LOOP 259 -- EVERY BRANCH CONTRIBUTES, NONE IS THE BOTTLENECK")
    say("=" * 104)
    say("     Loop 257 kept the network out of the OUTPUT path and recovered 0.30 -> 0.44, but")
    say("     left a bottleneck INSIDE the correction: 978 -> 256 -> 64, so every correction it")
    say("     could express was rank <= 64 with no route around those 64 numbers.")
    say("     Here the correction is a SUM of five parallel branches. Two of them (B1, B4) are")
    say("     elementwise in 978 dims and have no bottleneck at all.")
    say("     DECLARED BEFORE THE RUN: loop 257 removed 19% of validation MSE and still lost")
    say("     0.030 on a held-out line, so the binding constraint was TRANSFER, not capacity.")
    say("     This can only help by expressing what rank-64 destroyed, or by switching off what")
    say("     does not transfer. It cannot invent context information eight lines do not hold.")

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
    say(f"     context: {LF.shape[1]} measured dims; alphas fitted on a CALIBRATION line the")
    say(f"     network never trained on -- 7 lines train, the 8th weights, the 9th is scored")

    def build(fit_lines):
        gm, tr = {}, np.isin(pc, list(fit_lines))
        for g in D["genes"]:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        grand = Pm[tr].mean(0)
        lmean = {l: Pm[pc == l].mean(0) for l in LINES}

        def rows(mask, source=None):
            Xg, Xl, Xm, Y, A = [], [], [], [], []
            for j in np.where(mask)[0]:
                g = pg[j]
                if g not in gm: continue
                c = source if source else pc[j]
                # DEFECT I: the substitute line feeds the MODEL's inputs only. The standing
                # answer keeps the TRUE line, so the wrong-line control moves ONE thing.
                dg = gm[g] - grand
                dl_model = lmean[c] - grand          # what the MODEL is told
                dl_true = lmean[pc[j]] - grand       # what the BASELINE stands on
                Xg.append(dg); Xl.append(LF[li[c]]); Xm.append(dl_model)
                Y.append(Pm[j]); A.append(grand + dg + dl_true)
            return tuple(np.stack(v).astype(np.float32) for v in (Xg, Xl, Xm, Y, A))
        return rows

    def sc(P, Y): return np.array([H.pear(P[i], Y[i]) for i in range(len(Y))])

    def train(Xg, Xl, Xm, R, seed, use):
        torch.manual_seed(seed)
        r2 = np.random.default_rng(seed)
        ip = r2.permutation(len(Xg)); nv = max(200, int(0.12 * len(Xg)))
        va, fi = ip[:nv], ip[nv:]
        net = BranchNet(NL, Xl.shape[1], use=use)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        T = lambda a, i: torch.from_numpy(a[i])
        tg, tl, tm, tr_ = T(Xg, fi), T(Xl, fi), T(Xm, fi), T(R, fi)
        vg, vl, vm, vr = T(Xg, va), T(Xl, va), T(Xm, va), T(R, va)
        with torch.no_grad():
            v_init = float(((net(vg, vl, vm) - vr) ** 2).mean())
        best, bad, bw = 9e9, 0, None
        for _ in range(EPOCHS):
            idx = torch.randperm(len(fi))
            for b0 in range(0, len(fi), BATCH):
                j = idx[b0:b0 + BATCH]
                opt.zero_grad()
                ((net(tg[j], tl[j], tm[j]) - tr_[j]) ** 2).mean().backward(); opt.step()
            with torch.no_grad():
                v = float(((net(vg, vl, vm) - vr) ** 2).mean())
            if v < best - 1e-8: best, bad, bw = v, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= PATIENCE: break
        if bw: net.load_state_dict(bw)
        net.eval()
        return net, v_init, best

    def branch_out(net, Xg, Xl, Xm):
        with torch.no_grad():
            bs = net.branches(torch.from_numpy(Xg), torch.from_numpy(Xl), torch.from_numpy(Xm))
        return [b.numpy() for b in bs]

    def run(seed, use=None, shuffle_line=False, permute=False, subset=None):
        """One leave-one-line-out sweep. `use` selects which branches EXIST during training;
        `subset` selects which of the trained branches are allowed a non-zero alpha."""
        use = list(range(5)) if use is None else list(use)
        S, A_, ALL_A, VI = [], [], [], []
        rng = np.random.default_rng(seed)
        for oi, hold in enumerate(LINES):
            calib = LINES[(oi + 1) % len(LINES)]
            if calib == hold: calib = LINES[(oi + 2) % len(LINES)]
            trl = [l for l in LINES if l not in (hold, calib)]
            rows = build(trl)
            Xg, Xl, Xm, Y, A = rows(np.isin(pc, trl))
            Xgc, Xlc, Xmc, Yc, Ac = rows(pc == calib)
            src = (str(rng.choice([l for l in LINES if l != hold])) if shuffle_line else None)
            Xgt, Xlt, Xmt, Yt, At = rows(pc == hold, source=src)

            net, v0, vb = train(Xg, Xl, Xm, Y - A, seed, use)
            VI.append((v0, vb))
            Bc, Bt = branch_out(net, Xgc, Xlc, Xmc), branch_out(net, Xgt, Xlt, Xmt)
            if permute:
                Bc = [b[rng.permutation(len(b))] for b in Bc]
                Bt = [b[rng.permutation(len(b))] for b in Bt]
            keep = use if subset is None else [i for i in use if i in subset]
            al = np.zeros(5)
            if keep:
                a = fit_alphas(Yc, Ac, [Bc[i] for i in keep])
                for i, v in zip(keep, a): al[i] = v
            P = At + sum(al[i] * Bt[i] for i in keep) if keep else At
            S.append(sc(P, Yt)); A_.append(sc(At, Yt)); ALL_A.append(al)
        return np.concatenate(S), np.concatenate(A_), np.stack(ALL_A), VI

    def learned(vi):
        """Defect H: the fraction of validation MSE an arm actually removed. An arm that did
        not move cannot carry a gate, in either direction."""
        num = sum(max(0.0, a - b) for a, b in vi); den = sum(a for a, _ in vi)
        return num / den if den > 0 else 0.0

    say(f"     9 folds x {len(SEEDS)} seeds, 5 branches, plus a separately TRAINED B3-only arm ...")
    S, ADD, AL, VI = {}, None, {}, {}
    for sd in SEEDS:
        s_, a_, al_, vi_ = run(sd)
        S[sd], AL[sd], VI[sd] = s_, al_, vi_
        ADD = a_
        say(f"       seed {sd}: branching {np.nanmean(s_):.4f}   additive {np.nanmean(a_):.4f}"
            f"   val MSE removed {learned(vi_):.1%}   [{time.time() - t0:.0f}s]")
    full = S[SEEDS[0]]
    perm_s, _, _, _ = run(SEEDS[0], permute=True)
    b3_s, _, _, b3_vi = run(SEEDS[0], use=[2])
    free_s, _, _, free_vi = run(SEEDS[0], use=FREE_BRANCHES)
    say(f"       row-permuted null            {np.nanmean(perm_s):.4f}")
    say(f"       B3 alone (loop 257's bottleneck, trained separately) {np.nanmean(b3_s):.4f}"
        f"   val MSE removed {learned(b3_vi):.1%}")
    say(f"       context-FREE branches only (B1, B4)                  {np.nanmean(free_s):.4f}"
        f"   val MSE removed {learned(free_vi):.1%}")
    res["arms"] = {"branching": float(np.mean([np.nanmean(S[s]) for s in SEEDS])),
                   "additive": float(np.nanmean(ADD)),
                   "permuted": float(np.nanmean(perm_s)),
                   "b3_alone": float(np.nanmean(b3_s)),
                   "context_free_only": float(np.nanmean(free_s))}
    res["val_mse_removed"] = {"branching": learned(VI[SEEDS[0]]), "b3_alone": learned(b3_vi),
                              "context_free_only": learned(free_vi)}

    A0 = AL[SEEDS[0]]
    say("     alpha per branch, mean over the 9 folds (fitted on a line never trained on):")
    for i, nm in enumerate(BNAMES):
        m, s_ = float(A0[:, i].mean()), float(A0[:, i].std(ddof=1) / np.sqrt(A0.shape[0]))
        say(f"       {nm:16s} {m:+.4f} +/- {s_:.4f}   ({'SURVIVES' if abs(m) > 2 * s_ else 'not distinguishable from zero'})")
    res["alphas"] = {nm: {"mean": float(A0[:, i].mean()),
                          "se": float(A0[:, i].std(ddof=1) / np.sqrt(A0.shape[0])),
                          "per_fold": [float(x) for x in A0[:, i]]}
                     for i, nm in enumerate(BNAMES)}

    say("K1 DOES THE HARNESS REPRODUCE THE BASELINE?")
    a1 = float(np.nanmean(ADD))
    say(f"     additive here {a1:.4f} against loop 252's {LOOP252_ADDITIVE:.4f}")
    G.add("K1", bool(abs(a1 - LOOP252_ADDITIVE) <= K1_TOL), stat=a1,
          if_true=lambda: f"K1 PASS -- reproduces to {abs(a1 - LOOP252_ADDITIVE):.4f}",
          if_false=lambda: f"K1 FAIL -- {a1:.4f} against {LOOP252_ADDITIVE:.4f}")

    say("K2 MACHINERY ONLY -- DOES THE BRANCHING MODEL AT LEAST NOT LOSE?")
    d2, se2, z2 = H.paired(full, ADD)
    say(f"     branching {np.nanmean(full):.4f} vs baseline {a1:.4f}   {d2:+.4f} +/- {se2:.4f}")
    say(f"     NOT EVIDENCE. Fitting five alphas makes a small gain nearly automatic; this")
    say(f"     catches a broken alpha fit and nothing else. K3 decides whether anything real")
    say(f"     happened, and K4 decides whether removing the bottleneck is what did it.")
    G.add("K2", bool(d2 >= K2_FLOOR), stat=float(d2), requires=("K1",),
          if_true=lambda: f"K2 PASS (machinery) -- worth {d2:+.4f}, inside the {K2_FLOOR} "
                          f"machinery floor. Says the plumbing works and NOTHING about whether "
                          f"branching helped.",
          if_false=lambda: f"K2 FAIL (machinery) -- {d2:+.4f}, below a floor a fitted alpha "
                           f"should make unreachable; the alpha fit is broken, not the science")
    res["K2"] = {"delta": d2, "se": se2, "z": z2, "machinery_only": True}

    say("K3 LOAD-BEARING -- DOES IT BEAT ITS OWN ROW PERMUTATION?")
    d3, se3, z3 = H.paired(full, perm_s)
    say(f"     real {np.nanmean(full):.4f} vs permuted {np.nanmean(perm_s):.4f}   "
        f"{d3:+.4f} +/- {se3:.4f} ({z3:+.1f} se)")
    G.add("K3", bool(d3 >= K3_BAR), stat=float(d3), requires=("K1",),
          if_true=lambda: f"K3 PASS -- the branches carry {d3:+.4f} beyond same-sized noise",
          if_false=lambda: f"K3 FAIL -- {d3:+.4f} over its own row permutation; the gain came "
                           f"from fitting five weights, not from what the branches learned")
    res["K3"] = {"delta": d3, "se": se3, "z": z3}

    say("K4 THE HYPOTHESIS -- DOES BRANCHING BEAT THE BOTTLENECK?")
    lb3 = learned(b3_vi)
    d4, se4, z4 = H.paired(full, b3_s)
    say(f"     5 branches {np.nanmean(full):.4f} vs B3 alone {np.nanmean(b3_s):.4f}   "
        f"{d4:+.4f} +/- {se4:.4f} ({z4:+.1f} se)")
    say(f"     B3 alone is loop 257's rank-{GDIM} operator, TRAINED SEPARATELY and given the")
    say(f"     identical alpha treatment, so the only difference is the bottleneck.")
    if lb3 < TRAIN_FLOOR:
        G.add("K4", False, stat=float(lb3), requires=("K1",), void_if=True,
              void_reason=f"the B3-alone arm removed only {lb3:.1%} of validation MSE, so it "
                          f"never trained and cannot carry this gate (defect H)")
    else:
        G.add("K4", bool(d4 >= K4_BAR), stat=float(d4), requires=("K1",),
              if_true=lambda: f"K4 PASS -- removing the bottleneck is worth {d4:+.4f}; parallel "
                              f"branches beat one rank-{GDIM} path",
              if_false=lambda: f"K4 FAIL -- branching is worth {d4:+.4f} over the single "
                               f"bottlenecked operator; the bottleneck was not the constraint")
    res["K4"] = {"delta": d4, "se": se4, "z": z4, "b3_val_mse_removed": lb3}

    say("K5 WHICH BRANCHES SURVIVE OUT OF SAMPLE?")
    surv = [(nm, float(A0[:, i].mean()), float(A0[:, i].std(ddof=1) / np.sqrt(A0.shape[0])))
            for i, nm in enumerate(BNAMES)]
    alive = [nm for nm, m, s_ in surv if abs(m) > 2 * s_]
    say(f"     branches distinguishable from zero: {alive if alive else 'NONE'}")
    G.add("K5", bool(alive), stat=float(len(alive)), requires=("K1",),
          if_true=lambda: f"K5 PASS -- {len(alive)} of 5 branches survive: {', '.join(alive)}",
          if_false=lambda: f"K5 FAIL -- no branch's weight is distinguishable from zero on a "
                           f"line it was not trained on")
    res["K5"] = {"alive": alive}

    say("K6 DOES CONTEXT CONTRIBUTE, OR ONLY GENE-LEVEL STRUCTURE?")
    lfr = learned(free_vi)
    d6, se6, z6 = H.paired(full, free_s)
    say(f"     all 5 branches {np.nanmean(full):.4f} vs context-FREE only (B1, B4) "
        f"{np.nanmean(free_s):.4f}   {d6:+.4f} +/- {se6:.4f} ({z6:+.1f} se)")
    say(f"     loops 253 and 255 found nine annotation nulls and proved line identity's")
    say(f"     out-of-sample coefficient is exactly zero, so a FAIL here CONFIRMS that arc.")
    if lfr < TRAIN_FLOOR:
        G.add("K6", False, stat=float(lfr), requires=("K1",), void_if=True,
              void_reason=f"the context-free arm removed only {lfr:.1%} of validation MSE, so it "
                          f"never trained and cannot carry this gate (defect H)")
    else:
        G.add("K6", bool(d6 >= K6_BAR), stat=float(d6), requires=("K1",),
              if_true=lambda: f"K6 PASS -- the context-reading branches are worth {d6:+.4f} "
                              f"beyond gene-level structure alone",
              if_false=lambda: f"K6 FAIL -- context branches are worth {d6:+.4f} beyond "
                               f"gene-level structure; consistent with the nine nulls of loops "
                               f"253 and 255, and recorded as confirming them")
    res["K6"] = {"delta": d6, "se": se6, "z": z6, "free_val_mse_removed": lfr}

    say("K7 CONTROL: THE WRONG CELL LINE")
    if d3 < K3_BAR:
        G.add("K7", False, stat=float(d3), requires=("K3",), void_if=True,
              void_reason=f"K3's margin is {d3:+.4f}; there is nothing to collapse")
    else:
        sh, _, _, _ = run(SEEDS[0], shuffle_line=True)
        d7, _, _ = H.paired(sh, perm_s)
        f7 = d7 / d3
        say(f"     branches fed another line's properties: {d7:+.4f} against {d3:+.4f} ({f7:.0%})")
        G.add("K7", bool(f7 <= K7_MAX), stat=float(f7), requires=("K3",),
              if_true=lambda: f"K7 PASS -- collapses to {f7:.0%} on the wrong line",
              if_false=lambda: f"K7 FAIL -- {f7:.0%} survives the wrong line's properties")
        res["K7"] = {"real": d3, "shuffled": d7, "fraction": f7}

    say("K8 TRAINABILITY, AND WHAT THIS CANNOT SHOW")
    say(f"     validation MSE removed -- branching {learned(VI[SEEDS[0]]):.1%}, B3 alone "
        f"{lb3:.1%}, context-free {lfr:.1%}. Defect H: an arm that did not move cannot carry")
    say(f"     a gate, and K4/K6 VOID rather than FAIL when their comparison arm did not train.")
    say("     K2 passing is not a result. Five fitted weights buy a gain from any branches,")
    say("     including the permuted ones, whose score is printed so the free lunch is visible.")
    say("     Capacity was never the binding constraint: loop 257 removed 19% of validation MSE")
    say("     and still lost 0.030 across lines. Branches cannot invent context information")
    say("     that eight training lines do not contain.")
    say("     If only B1 and B4 survive, the honest reading is a gene-level correction with no")
    say("     context contribution, which was DECLARED in advance as an acceptable outcome.")
    say("     Nine lines, 978 landmarks, shRNA rather than a clean knockout, and a surviving")
    say("     branch names WHERE a correction helps, never WHAT the interaction is.")

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
