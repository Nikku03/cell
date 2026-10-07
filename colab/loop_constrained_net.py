"""Loop 251. The five-clause principle as five ablatable structural terms in one neural model.

THE PRINCIPLE BEING TESTED, and it is worth stating why it is testable at all: every clause names
a computable object rather than an attitude.

    closure      -> the dynamics have a fixed point the system returns to
    currencies   -> a conserved budget, so paying for one thing costs another
    memory       -> parameters carrying history, at the timescales loop 249 measured
    prediction   -> optimisation with an INFORMATION COST, so deviating from the prior is not free
    scaling      -> not testable here; one cell line, no size axis. Said plainly rather than faked.

THE ONE THING THAT MAKES THIS A TEST RATHER THAN A DEMONSTRATION. Loop 241 built two MLPs on 19.3M
rows and both LOST to a linear model given identical inputs, by -0.0155 and -0.0170. Loop 242's
signed GCN was the first architecture here to beat its own twin, and it did so by encoding a real
asymmetry (activator against inhibitor) rather than by adding capacity. So a constrained network is
only interesting if each constraint beats an UNCONSTRAINED NETWORK OF THE SAME CAPACITY, and the
whole thing is only interesting if it beats ridge. Both comparisons are gates here, not remarks.

THE BUDGET PREMISE, MEASURED BEFORE THE MODEL WAS WRITTEN, because a hard constraint on a quantity
that is not conserved would be a modelling error dressed as a principle. Over 11,258 K562
perturbations:

    corr(sum of up-regulation, sum of down-regulation)   0.6924
    median up/down ratio                                 1.1467
    median |net| / (up + down)                           0.0793

A perfect budget gives ratio 1.00 and |net| = 0. So the transcriptome is a SOFT budget: about 8%
net imbalance per perturbation. The currency clause therefore enters as a penalty with a fitted
weight, not as a hard projection, and that choice is forced by the data rather than by taste.

THE TASK. sci-fate A549: predict a cell's NEW-RNA profile -- what it is transcribing now, which is
the rate -- from its TOTAL-RNA profile, which is the state. Held out by cell. This is the same
quantity loop 250 used, where the state alone reached 0.6473 and time added +0.00001.

ARMS. Every neural arm has identical capacity and identical inputs; they differ only in structure.

    RIDGE        linear, same inputs. Has beaten every network in this project so far.
    MLP          unconstrained. The twin every constrained arm must beat.
    +BUDGET      a penalty tying the predicted profile's total to the cell's measured total new RNA
    +CLOSURE     the network predicts a DEVIATION from the population mean profile, so an average
                 cell receives zero deviation by construction -- a fixed point built in, not learned
    +MEMORY      time enters as the two decay bases loop 249 measured (tau 1.94 h and 5.25 h),
                 frozen, instead of as a raw number
    +INFOCOST    an L1 penalty on the deviation from the prior: departing from the population mean
                 costs, which is the Landauer clause in the only form this data can carry
    +ALL         all four together

PREDECLARED, BEFORE ANY NUMBER.

  A1 DOES THE UNCONSTRAINED NETWORK BEAT RIDGE?
     The loop 241 control. Gate: PASS iff MLP exceeds RIDGE by at least 0.005, paired over
     held-out cells. A FAIL is expected on this project's record and is not a reason to stop: it
     sets the baseline the constrained arms must actually clear.

  A2 DOES ANY SINGLE CONSTRAINT BEAT THE UNCONSTRAINED NETWORK?      -- the load-bearing gate
     Best of +BUDGET, +CLOSURE, +MEMORY, +INFOCOST against MLP, paired.
     Gate: PASS iff at least 0.005. Structure has to earn its place against the same capacity.

  A3 DOES THE FULL MODEL BEAT THE BEST SINGLE CONSTRAINT?      -- requires A2
     Gate: PASS iff +ALL exceeds the best single arm by at least 0.005. A FAIL means the clauses
     are redundant with each other rather than additive, which is a statement about the principle
     and not only about the fit.

  A4 DOES THE BEST ARM BEAT RIDGE AT ALL?
     Gate: PASS iff the best neural arm exceeds RIDGE by at least 0.005. This is the gate that
     decides whether any of this is worth preferring over the linear model that has won every
     previous comparison in this project.

  A5 IS THE ADVANTAGE BIGGER THAN THE SEED NOISE?      -- requires A2, VOID if A2 found nothing
     Three seeds. Gate: PASS iff A2's margin exceeds twice the across-seed standard deviation of
     the winning arm. Loop 225's MLP win was reversed twice by later loops whose across-split sd
     was 0.1394 against ridge's 0.0288, and loop 241's seed sd was 0.0125 against a 0.02 bar.

  A6 WHICH CLAUSE PAYS, AND DOES ANY HURT? -- reported, not gated.
     Each constraint's individual margin over MLP, with its sign. A constraint that makes the fit
     WORSE is evidence about the principle and is reported as such rather than dropped.

  A7 WHAT THIS CANNOT SHOW -- written before the run.
     The scaling clause is not tested. One cell line, no size axis, no cross-taxon comparison.
     Nothing here speaks to it and no number below should be read as if it did.
     "Information cost" here is an L1 penalty on deviation from a population mean. That is a
     regulariser with a thermodynamic story attached, not a measurement of joules. Landauer's
     exchange rate is not established by anything in this loop.
     A fixed point built into the architecture is an assumption, not a finding: +CLOSURE beating
     MLP would show the assumption helps prediction, not that the cell has that fixed point.
     Predicting the rate from the state at one instant is not the same as simulating the dynamics.
     A model can score well here and still diverge if iterated.
"""
import os, sys, json, time, warnings, copy
from pathlib import Path
import numpy as np
from scipy import sparse

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

import torch
import torch.nn as nn
torch.set_num_threads(4)

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_constrained_net.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SF = SCR / "scifate"

SEED, NFOLD, NPC, NOUT = 251251, 5, 60, 2000
SEEDS = [0, 1, 2]
HID, EPOCHS, PATIENCE, LR, BATCH = 256, 60, 8, 3e-3, 256
TAU1, TAU2 = 1.94, 5.25                     # loop 249, frozen
W_BUDGET, W_INFO = 0.30, 1e-3
A1_BAR, A2_BAR, A3_BAR, A4_BAR, A5_MULT = 0.005, 0.005, 0.005, 0.005, 2.0

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


class Net(nn.Module):
    """One architecture; the clauses switch on and off. CLOSURE changes what the output MEANS --
    a deviation from the prior rather than an absolute profile -- so an average cell gets zero
    deviation by construction rather than having to learn it."""
    def __init__(s, d, o, prior, closure):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(),
                            nn.Linear(HID, o))
        s.closure = closure
        s.register_buffer("prior", torch.from_numpy(prior.astype(np.float32)))
    def forward(s, x):
        y = s.f(x)
        return (s.prior[None, :] + y) if s.closure else y
    def deviation(s, x):
        y = s.f(x)
        return y if s.closure else (y - s.prior[None, :])


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "five-clause principle as ablatable structural terms"}
    say("=" * 104)
    say("LOOP 251 -- THE PRINCIPLE AS FIVE ABLATABLE TERMS, EACH AGAINST THE SAME CAPACITY")
    say("=" * 104)
    say("     Loop 241's two MLPs LOST to their linear twins by -0.0155 and -0.0170. So a")
    say("     constrained network is interesting only if each constraint beats an UNCONSTRAINED")
    say("     network of the same capacity (A2) and the whole thing beats ridge (A4).")
    say(f"     The budget premise, measured first: over 11,258 K562 perturbations the up/down")
    say(f"     ratio is 1.1467 with 7.9% net imbalance -- a SOFT budget, so the currency clause")
    say(f"     enters as a penalty with weight {W_BUDGET}, not as a hard projection.")
    say("     The scaling clause is NOT tested: one cell line, no size axis.")

    M = sparse.load_npz(SF / "total.npz").tocsr()
    N = sparse.load_npz(SF / "new.npz").tocsr()
    meta = np.load(SF / "meta.npz", allow_pickle=True)
    hours = np.array([float(str(x).replace("h", "")) for x in meta["time"]])
    dbl = np.asarray(meta["doublet"], float)
    umi = np.asarray(M.sum(1)).ravel()
    ok = (dbl <= 0.15) & (umi >= 1000)
    M, N, hours, umi = M[ok], N[ok], hours[ok], umi[ok]
    nc = np.asarray((M > 0).sum(0)).ravel()
    gk = np.where(nc >= 100)[0]
    M, N = M[:, gk], N[:, gk]
    say(f"     sci-fate: {M.shape[0]:,} cells x {M.shape[1]:,} genes, timepoints "
        f"{sorted(set(hours))} h")

    def ln(A):
        s_ = np.asarray(A.sum(1)).ravel()
        B = sparse.diags(1e4 / np.maximum(s_, 1)) @ A
        B = B.tocsr(); B.data = np.log1p(B.data)
        return np.asarray(B.todense(), np.float32)
    Sd, Nd = ln(M), ln(N)
    sel = np.argsort(-Nd.var(0))[:NOUT]
    Y = Nd[:, sel]
    say(f"     target: the {NOUT:,} most variable new-RNA genes")

    U, sv, _ = np.linalg.svd(Sd - Sd.mean(0), full_matrices=False)
    P = (U[:, :NPC] * sv[:NPC]); P = ((P - P.mean(0)) / (P.std(0) + 1e-9)).astype(np.float32)
    depth = ((np.log(umi) - np.log(umi).mean()) / np.log(umi).std()).astype(np.float32)
    traw = ((hours - hours.mean()) / hours.std()).astype(np.float32)
    tmem = np.stack([1 - np.exp(-hours / TAU1), 1 - np.exp(-hours / TAU2)], 1).astype(np.float32)
    tmem = (tmem - tmem.mean(0)) / (tmem.std(0) + 1e-9)
    say(f"     memory basis frozen from loop 249: tau = {TAU1} h and {TAU2} h")
    newtot = np.asarray(N.sum(1)).ravel()
    bud = ((np.log1p(newtot) - np.log1p(newtot).mean()) / np.log1p(newtot).std()).astype(np.float32)

    ARMS = {
        "MLP":       dict(mem=False, closure=False, budget=False, info=False),
        "+BUDGET":   dict(mem=False, closure=False, budget=True, info=False),
        "+CLOSURE":  dict(mem=False, closure=True, budget=False, info=False),
        "+MEMORY":   dict(mem=True, closure=False, budget=False, info=False),
        "+INFOCOST": dict(mem=False, closure=False, budget=False, info=True),
        "+ALL":      dict(mem=True, closure=True, budget=True, info=True),
    }
    order = rng.permutation(len(hours))
    folds = [order[i::NFOLD] for i in range(NFOLD)]

    def features(cfg):
        cols = [P, depth[:, None], (tmem if cfg["mem"] else traw[:, None])]
        return np.concatenate(cols, 1).astype(np.float32)

    def run(cfg, seed, ridge=False):
        F = features(cfg)
        sc = np.full(len(hours), np.nan)
        for te in folds:
            tr = np.setdiff1d(np.arange(len(hours)), te)
            prior = Y[tr].mean(0)
            if ridge:
                Z = np.concatenate([F[tr], np.ones((len(tr), 1), np.float32)], 1)
                A = Z.T @ Z + 1e-2 * len(tr) * np.eye(Z.shape[1], dtype=np.float32)
                B = np.linalg.solve(A, Z.T @ Y[tr])
                Pr = np.concatenate([F[te], np.ones((len(te), 1), np.float32)], 1) @ B
                for j, c in enumerate(te): sc[c] = pear(Pr[j], Y[c])
                continue
            torch.manual_seed(seed)
            r2 = np.random.default_rng(seed)
            ip = r2.permutation(len(tr)); nv = max(50, int(0.15 * len(tr)))
            va, fi = tr[ip[:nv]], tr[ip[nv:]]
            net = Net(F.shape[1], NOUT, prior, cfg["closure"])
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            xf, yf = torch.from_numpy(F[fi]), torch.from_numpy(Y[fi])
            bf = torch.from_numpy(bud[fi])
            xv, yv = torch.from_numpy(F[va]), torch.from_numpy(Y[va])
            best, bad, bw = 9e9, 0, None
            for ep in range(EPOCHS):
                idx = torch.randperm(len(fi))
                for b0 in range(0, len(fi), BATCH):
                    j = idx[b0:b0 + BATCH]
                    opt.zero_grad()
                    out = net(xf[j])
                    loss = ((out - yf[j]) ** 2).mean()
                    if cfg["budget"]:
                        tot = out.sum(1)
                        tot = (tot - tot.mean()) / (tot.std() + 1e-6)
                        loss = loss + W_BUDGET * ((tot - bf[j]) ** 2).mean()
                    if cfg["info"]:
                        loss = loss + W_INFO * net.deviation(xf[j]).abs().mean()
                    loss.backward(); opt.step()
                with torch.no_grad():
                    v = float(((net(xv) - yv) ** 2).mean())
                if v < best - 1e-7:
                    best, bad, bw = v, 0, copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                    if bad >= PATIENCE: break
            if bw: net.load_state_dict(bw)
            net.eval()
            with torch.no_grad():
                Pr = net(torch.from_numpy(F[te])).numpy()
            for j, c in enumerate(te): sc[c] = pear(Pr[j], Y[c])
        return sc

    say("     fitting: ridge, then six neural arms x 3 seeds ...")
    S = {}
    S["RIDGE"] = run(ARMS["MLP"], 0, ridge=True)
    say(f"       {'RIDGE':<11} {np.nanmean(S['RIDGE']):.4f}   [{time.time() - t0:.0f}s]")
    per_seed = {a: [] for a in ARMS}
    for a, cfg in ARMS.items():
        for sd in SEEDS:
            s_ = run(cfg, sd)
            per_seed[a].append(s_)
            if sd == SEEDS[0]: S[a] = s_
        say(f"       {a:<11} {np.mean([np.nanmean(x) for x in per_seed[a]]):.4f}   "
            f"(seeds {', '.join(f'{np.nanmean(x):.4f}' for x in per_seed[a])})   "
            f"[{time.time() - t0:.0f}s]")
    res["arms"] = {a: float(np.mean([np.nanmean(x) for x in per_seed[a]])) for a in ARMS}
    res["arms"]["RIDGE"] = float(np.nanmean(S["RIDGE"]))

    # ---------------------------------------------------------------- A1
    say("A1 DOES THE UNCONSTRAINED NETWORK BEAT RIDGE?")
    d1, se1, z1 = paired(S["MLP"], S["RIDGE"])
    say(f"     MLP {np.nanmean(S['MLP']):.4f} vs RIDGE {np.nanmean(S['RIDGE']):.4f}   "
        f"paired {d1:+.4f} +/- {se1:.4f} ({z1:+.1f} se)")
    G.add("A1", bool(d1 >= A1_BAR), stat=float(d1),
          if_true=lambda: f"A1 PASS -- the unconstrained network beats ridge by {d1:+.4f}",
          if_false=lambda: f"A1 FAIL -- the unconstrained network is {d1:+.4f} against ridge; the "
                           f"loop 241 pattern holds and the constrained arms must clear ridge too")
    res["A1"] = {"delta": d1, "se": se1, "z": z1}

    # ---------------------------------------------------------------- A6 first (feeds A2)
    say("A6 WHICH CLAUSE PAYS, AND DOES ANY HURT? -- reported, not gated")
    singles = ["+BUDGET", "+CLOSURE", "+MEMORY", "+INFOCOST"]
    marg = {}
    for a in singles:
        d, se, zz = paired(S[a], S["MLP"])
        marg[a] = (d, se, zz)
        say(f"       {a:<11} over MLP: {d:+.5f} +/- {se:.5f}  ({zz:+.1f} se)")
    res["A6"] = {a: {"delta": marg[a][0], "se": marg[a][1], "z": marg[a][2]} for a in marg}

    # ---------------------------------------------------------------- A2
    say("A2 DOES ANY SINGLE CONSTRAINT BEAT THE UNCONSTRAINED NETWORK?")
    bestc = max(singles, key=lambda a: marg[a][0])
    d2 = marg[bestc][0]
    say(f"     best single constraint is {bestc} at {d2:+.5f}")
    G.add("A2", bool(d2 >= A2_BAR), stat=float(d2),
          if_true=lambda: f"A2 PASS -- {bestc} beats the same capacity unconstrained by {d2:+.4f}",
          if_false=lambda: f"A2 FAIL -- the best constraint adds {d2:+.5f} over the same capacity "
                           f"unconstrained, against a {A2_BAR} bar")
    res["A2"] = {"best": bestc, "delta": d2}

    # ---------------------------------------------------------------- A3
    say("A3 DOES THE FULL MODEL BEAT THE BEST SINGLE CONSTRAINT?")
    d3, se3, z3 = paired(S["+ALL"], S[bestc])
    say(f"     +ALL {np.nanmean(S['+ALL']):.4f} vs {bestc} {np.nanmean(S[bestc]):.4f}   "
        f"paired {d3:+.5f} +/- {se3:.5f} ({z3:+.1f} se)")
    G.add("A3", bool(d3 >= A3_BAR), stat=float(d3), requires=("A2",),
          if_true=lambda: f"A3 PASS -- the clauses are additive: {d3:+.4f} over the best single",
          if_false=lambda: f"A3 FAIL -- combining all four adds {d3:+.5f} over {bestc} alone; the "
                           f"clauses are redundant with each other rather than additive")
    res["A3"] = {"delta": d3, "se": se3, "z": z3}

    # ---------------------------------------------------------------- A4
    say("A4 DOES THE BEST ARM BEAT RIDGE AT ALL?")
    bestall = max(list(ARMS), key=lambda a: np.nanmean(S[a]))
    d4, se4, z4 = paired(S[bestall], S["RIDGE"])
    say(f"     best neural arm {bestall} {np.nanmean(S[bestall]):.4f} vs RIDGE "
        f"{np.nanmean(S['RIDGE']):.4f}   paired {d4:+.4f} +/- {se4:.4f} ({z4:+.1f} se)")
    G.add("A4", bool(d4 >= A4_BAR), stat=float(d4),
          if_true=lambda: f"A4 PASS -- {bestall} beats ridge by {d4:+.4f}; structure earned the "
                          f"network its place",
          if_false=lambda: f"A4 FAIL -- the best neural arm is {d4:+.4f} against ridge; the linear "
                           f"model remains the thing to beat")
    res["A4"] = {"best": bestall, "delta": d4, "se": se4, "z": z4}

    # ---------------------------------------------------------------- A5
    say("A5 IS THE ADVANTAGE BIGGER THAN THE SEED NOISE?")
    sds = {a: float(np.std([np.nanmean(x) for x in per_seed[a]], ddof=1)) for a in ARMS}
    for a in ARMS: say(f"       {a:<11} across-seed sd {sds[a]:.5f}")
    if d2 < A2_BAR:
        G.add("A5", False, stat=float(d2), requires=("A2",), void_if=True,
              void_reason=f"A2 found {d2:+.5f}; there is no advantage for a seed spread to be "
                          f"compared against")
    else:
        G.add("A5", bool(d2 >= A5_MULT * sds[bestc]), stat=float(sds[bestc]), requires=("A2",),
              if_true=lambda: f"A5 PASS -- {d2:+.4f} is {d2 / max(sds[bestc], 1e-9):.1f}x the "
                              f"across-seed sd",
              if_false=lambda: f"A5 FAIL -- a {d2:+.4f} margin against an across-seed sd of "
                               f"{sds[bestc]:.5f}")
    res["A5"] = {"seed_sd": sds}

    say("A7 WHAT THIS CANNOT SHOW")
    say("     The scaling clause is not tested at all: one cell line, no size axis, no taxa.")
    say("     'Information cost' here is an L1 penalty on deviation from a population mean -- a")
    say("     regulariser with a thermodynamic story attached, not a measurement of joules.")
    say("     Landauer's exchange rate is not established by anything in this loop.")
    say("     A built-in fixed point is an assumption: +CLOSURE beating MLP would show the")
    say("     assumption helps prediction, not that the cell has that fixed point.")
    say("     Predicting the rate from the state at one instant is not simulating the dynamics.")
    say("     A model can score here and still diverge when iterated.")

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
