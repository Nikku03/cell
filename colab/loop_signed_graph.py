"""Loop 242. Signed directed edges against the unsigned graph every previous loop used.

THE COMPLAINT THIS ANSWERS, AND IT IS CORRECT. Every network result in this project -- loop 239's
partner expression, loop 240's A3_PARTNER, loop 241's pair features -- used BioGRID, which is
UNDIRECTED and UNSIGNED. A graph with no sign cannot distinguish an activator from a repressor, so
a model built on it is asked to predict the direction of an expression change from a structure that
does not encode direction. OmniPath does: 1,096,924 interactions, 1,054,024 directed, 131,694
stimulation and 42,160 inhibition, and this project has never used one of them.

WHAT WAS MEASURED BEFORE THIS FILE WAS WRITTEN, and it is reported here because it sets the scale
every gate below is calibrated to. Coverage against the 8,248-gene Perturb-seq readout: 3,105 of
the 9,867 perturbed genes have at least one directed measured target, 1,296 have five or more.
Then the first-principles direction check, with the prediction stated before the number was
computed -- knocking down an ACTIVATOR should push its targets DOWN, knocking down a REPRESSOR
should push its targets UP, each against the same genes' behaviour under other knockdowns:

    ACTIVATING   n=642   paired -0.0055 +/- 0.0022   (-2.5 se)     predicted negative: yes
    INHIBITING   n=219   paired +0.0028 +/- 0.0036   (+0.8 se)     predicted positive: yes, but
                                                                   not distinguishable from zero
    gap +0.0083, in the predicted direction

So the sign is real and it is SMALL. The activating arm carries it; the inhibiting arm on its own
does not clear noise. Any gate here that demanded a large effect would be a gate that could not
pass, which is the defect gate_guard exists to prevent, so the bars below are set against 0.008
and not against the 0.02 bars used in loops 240-241 on a different quantity.

THE MODEL. A knockdown of gene p removes p's outgoing regulation. If p activates j, removing p
should lower j; if p represses j, removing p should raise j. So the physically motivated one-hop
prediction is the NEGATED signed adjacency row:

    yhat_j  =  -S[p, j],        S[p,j] = +1 activate, -1 inhibit, 0 no edge

and two hops compose by multiplying signs, which is the whole content of the claim that sign
matters: an activator of a repressor of j should RAISE j, and an unsigned graph cannot express
that at all.

SEVEN ARMS, held out BY PERTURBATION.

    C0 MEAN            the mean delta profile over training perturbations. Knows nothing about
                       which gene was hit. The floor.
    C1 BIOGRID         one hop through the undirected unsigned graph. What loops 239-241 used.
    C2 OP_UNSIGNED     one hop through OmniPath's DIRECTED edges with the sign discarded. This
                       separates what DIRECTION is worth from what SIGN is worth, which C1 cannot.
    C3 OP_SIGNED       one hop, signed.
    C4 OP_SIGNED_2HOP  signed one hop plus signed two-hop paths, signs multiplied along the path,
                       with one fitted mixing weight.
    C5 SIGNED_RIDGE    every feature above as columns of one linear model.        <- the twin
    C6 SIGNED_GCN      the same features through separate learned activator and inhibitor weight
                       matrices with a nonlinearity, as proposed.

PREDECLARED, BEFORE ANY NUMBER.

  Q1 IS THERE ENOUGH SIGNED STRUCTURE TOUCHING THE READOUT TO TEST ANYTHING?
     Gate: PASS iff at least 500 perturbations have 5 or more signed directed targets among the
     genes that are finite in every row. Everything else requires this.

  Q2 DOES THE SIGN PREDICT THE DIRECTION OF THE CHANGE?
     The probe above, rerun inside the loop on the screened matrix so the number in the record is
     the loop's own. Gate: PASS iff the inhibiting-minus-activating gap is positive AND the
     activating arm is at least 2 standard errors below zero. Both halves, because a gap built
     from two indistinguishable numbers is loop 87's C6.

  Q3 DOES SIGN BEAT NO SIGN?      -- requires Q1
     C3 against C2, paired over held-out perturbations. Both use the same directed edge set and
     differ ONLY in whether the sign is kept, so this isolates the sign from the direction and
     from the choice of database.
     Gate: PASS iff C3 exceeds C2 by at least 0.005 paired, at 3 or more standard errors.

  Q4 DOES DIRECTION BEAT AN UNDIRECTED GRAPH?      -- requires Q1
     C2 against C1. Gate: PASS iff C2 exceeds C1 by at least 0.005 paired at 3 se.

  Q5 CONTROL: PERMUTED SIGNS.      -- requires Q3, VOID if the Q3 advantage is under 0.002
     The activate/inhibit labels shuffled across edges, the graph otherwise identical.
     Gate: PASS iff the C3-over-C2 advantage collapses to under 25% of its true value.

  Q6 CONTROL: REVERSED EDGES.      -- requires Q1
     Every directed edge reversed. A regulator's targets become its regulators.
     Gate: PASS iff C3's advantage over C0 falls by at least half. A FAIL means the graph is
     acting as an undirected similarity measure and the arrows are decorative.

  Q7 DID THE NETWORK HELP, OR THE FEATURES?      -- the loop 241 gate, requires Q1
     C6 against C5 on identical inputs, paired.
     Gate: PASS iff C6 exceeds C5 by at least 0.005. Loop 241 measured both its MLPs LOSING to
     their linear twins by -0.0155 and -0.0170; this gate exists so that result is not quietly
     repeated as a win.

  Q8 WHAT THIS CANNOT SHOW -- written before the run.
     OmniPath is a literature aggregation. Its edges are biased toward well-studied genes, and
     coverage correlates with how much a gene has been published on, so a coverage-driven result
     can look like a mechanism.
     The readout is CRISPRi knockdown in K562 at one timepoint. Two hops of transcriptional
     propagation may not have occurred by the time of measurement, which would penalise C4
     for a reason that is about assay timing rather than about graph structure.
     A sign in OmniPath is a consensus over sources and over contexts. An edge that activates in
     one cell type may repress in another, and this tests the consensus sign, not a K562 sign.
     8,175 of 8,248 genes are finite in every row and 8,917 of 11,258 rows are finite; the rest
     are screened out, so nothing here speaks for the perturbations that were dropped.
"""
import os, sys, json, time, csv, collections, warnings
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
OUT = "outputs/loop_signed_graph.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
K562 = SCR / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
OP = SCR / "reg" / "op_2022.tsv"
BG = SCR / "biogrid_hs_edges.tsv.gz"

SEED, NFOLD = 242242, 5
Q1_MIN, Q2_SE, Q3_BAR, Q3_SE, Q4_BAR, Q5_MAX, Q6_DROP, Q7_BAR = 500, 2.0, 0.005, 3.0, 0.005, 0.25, 0.50, 0.005
GCN_EPOCHS, GCN_PATIENCE, GCN_LR, GCN_DIM = 120, 15, 3e-3, 64

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear_cols(P, T):
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        a, b = P[:, j], T[:, j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5: continue
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        if d > 0: out[j] = float((a * b).sum() / d)
    return out


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


FEATS = ["self", "bg1", "d1", "a1", "i1", "a2", "i2", "deg_out", "deg_in"]
FI = {n: i for i, n in enumerate(FEATS)}
ARMS = {
    "C1_BIOGRID":       ["self", "bg1"],
    "C2_OP_UNSIGNED":   ["self", "d1"],
    "C3_OP_SIGNED":     ["self", "a1", "i1"],
    "C4_OP_SIGNED_2HOP": ["self", "a1", "i1", "a2", "i2"],
    "C5_SIGNED_RIDGE":  FEATS,
}


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "signed directed OmniPath vs unsigned BioGRID on K562 Perturb-seq"}
    say("=" * 104)
    say("LOOP 242 -- SIGNED DIRECTED EDGES, AGAINST THE UNSIGNED GRAPH EVERY EARLIER LOOP USED")
    say("=" * 104)
    say("     The sign is real and small: the pre-loop probe measured activating targets at")
    say("     -0.0055 +/- 0.0022 and inhibiting at +0.0028 +/- 0.0036. Bars below are set")
    say("     against 0.008, not against the 0.02 used in loops 240-241 on a different quantity.")

    import h5py
    f = h5py.File(K562, "r")
    cats = f["var"]["__categories"]["gene_name"][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
    gname = cats[f["var"]["gene_name"][:]]
    k = f["obs"].attrs.get("_index", "_index")
    k = k.decode() if isinstance(k, bytes) else k
    obs = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["obs"][k][:]])
    pert = np.array([o.split("_")[1] for o in obs])       # field 1 is the symbol (loop 224's fix)
    X = f["X"][:]
    f.close()

    gcol = np.isfinite(X).all(0)
    rrow = np.isfinite(X[:, gcol]).all(1)
    MG = gname[gcol]                                       # measured, finite-everywhere genes
    say(f"     screened: {gcol.sum():,} of {len(gcol):,} genes finite in every row, "
        f"{rrow.sum():,} of {len(rrow):,} rows finite across them")
    X = X[np.ix_(rrow, gcol)]
    pert = pert[rrow]
    mpos = {g: i for i, g in enumerate(MG)}

    # ---------------------------------------------------------------- graphs
    A_act, A_inh, A_dir = [collections.defaultdict(set) for _ in range(3)]
    with open(OP) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            s, t = row["source_genesymbol"], row["target_genesymbol"]
            if not s or not t or s == t or row["is_directed"] != "1": continue
            A_dir[s].add(t)
            if row["is_stimulation"] == "1": A_act[s].add(t)
            if row["is_inhibition"] == "1": A_inh[s].add(t)
    import gzip
    A_bg = collections.defaultdict(set)
    with open(BG, "rb") as fh:
        for ln in gzip.GzipFile(fileobj=fh):
            p = ln.decode().rstrip("\n").split("\t")
            if len(p) < 2 or p[0] == p[1]: continue
            A_bg[p[0]].add(p[1]); A_bg[p[1]].add(p[0])
    say(f"     OmniPath: {sum(len(v) for v in A_dir.values()):,} directed, "
        f"{sum(len(v) for v in A_act.values()):,} activating, "
        f"{sum(len(v) for v in A_inh.values()):,} inhibiting")
    say(f"     BioGRID:  {sum(len(v) for v in A_bg.values()):,} undirected entries")

    # node universe: everything that can carry a path between a perturbation and a measured gene
    nodes = sorted(set(MG.tolist()) | set(A_dir) | {t for v in A_dir.values() for t in v}
                   | set(A_bg) | {t for v in A_bg.values() for t in v})
    npos = {g: i for i, g in enumerate(nodes)}
    NN = len(nodes)
    say(f"     propagation universe: {NN:,} nodes")

    def spmat(d):
        r, c, v = [], [], []
        for s, ts in d.items():
            si = npos.get(s, -1)
            if si < 0: continue
            for t in ts:
                ti = npos.get(t, -1)
                if ti < 0: continue
                r.append(si); c.append(ti); v.append(1.0)
        return sparse.csr_matrix((v, (r, c)), shape=(NN, NN), dtype=np.float32)

    S_act, S_inh, S_dir, S_bg = spmat(A_act), spmat(A_inh), spmat(A_dir), spmat(A_bg)
    S_sign = (S_act - S_inh).tocsr()
    mcols = np.array([npos[g] for g in MG])
    NG = len(MG)

    usable = np.array([i for i in range(len(pert))
                       if pert[i] in npos and S_dir[npos[pert[i]]].nnz > 0])
    nsig5 = sum(1 for g in set(pert[usable])
                if len(set(A_act.get(g, set()) | A_inh.get(g, set())) & set(MG.tolist())) >= 5)
    say(f"     {len(usable):,} screened rows have a perturbed gene with outgoing directed edges")

    # ---------------------------------------------------------------- Q1
    say("Q1 IS THERE ENOUGH SIGNED STRUCTURE TOUCHING THE READOUT TO TEST ANYTHING?")
    say(f"     perturbations with 5 or more SIGNED targets among the {NG:,} screened genes: {nsig5:,}")
    G.add("Q1", bool(nsig5 >= Q1_MIN), stat=float(nsig5),
          if_true=lambda: f"Q1 PASS -- {nsig5:,} perturbations carry 5 or more signed measured "
                          f"targets",
          if_false=lambda: f"Q1 FAIL -- only {nsig5:,} against a bar of {Q1_MIN}; the signed graph "
                           f"barely touches this readout")
    res["coverage"] = {"usable_rows": int(len(usable)), "signed_ge5": int(nsig5), "genes": int(NG)}

    # ---------------------------------------------------------------- Q2
    say("Q2 DOES THE SIGN PREDICT THE DIRECTION OF THE CHANGE?")
    rows_of = collections.defaultdict(list)
    for i in usable: rows_of[pert[i]].append(i)

    def direction(d, label):
        real, ctrl = [], []
        for g, ts in d.items():
            if g not in rows_of: continue
            cols = [mpos[t] for t in ts if t in mpos]
            if len(cols) < 5: continue
            r = rows_of[g][0]
            others = rng.choice(usable, 8, replace=False)
            real.append(float(np.mean(X[r, cols])))
            ctrl.append(float(np.mean([np.mean(X[o, cols]) for o in others])))
        mu, se, z = paired(np.array(real), np.array(ctrl))
        say(f"     {label:<11} n={len(real):4d}  targets {np.mean(real):+.4f} vs the same genes "
            f"under other knockdowns {np.mean(ctrl):+.4f}   paired {mu:+.4f} +/- {se:.4f} "
            f"({z:+.1f} se)")
        return mu, se, z

    am, ase, az = direction(A_act, "ACTIVATING")
    im, ise, iz = direction(A_inh, "INHIBITING")
    gap = im - am
    say(f"     gap (inhibiting minus activating) {gap:+.4f}; the prediction requires it POSITIVE")
    say(f"     and requires the activating arm at or below {-Q2_SE:.0f} se, which is the half that")
    say(f"     stops a gap built from two indistinguishable numbers counting as evidence")
    G.add("Q2", bool(gap > 0 and az <= -Q2_SE), stat=float(gap),
          if_true=lambda: f"Q2 PASS -- gap {gap:+.4f} with the activating arm at {az:+.1f} se",
          if_false=lambda: f"Q2 FAIL -- gap {gap:+.4f}, activating arm {az:+.1f} se "
                           f"(bar {-Q2_SE:.0f} se)")
    res["direction"] = {"activating": [am, ase, az], "inhibiting": [im, ise, iz], "gap": gap}

    # ---------------------------------------------------------------- features
    deg_out = np.asarray(S_dir.sum(1)).ravel()
    deg_in = np.asarray(S_dir.sum(0)).ravel()

    def build(prow, Sa, Si, Sd, Sb):
        """(NG, len(FEATS)) for one perturbed gene. A knockdown REMOVES the regulation, so the
        one-hop prediction is the NEGATED signed row: an activator's targets fall, a repressor's
        targets rise. Two-hop signs multiply along the path, which is the whole content of the
        claim that sign matters."""
        p = npos[prow]
        F = np.zeros((NG, len(FEATS)), np.float32)
        F[:, FI["self"]] = (MG == prow).astype(np.float32)
        F[:, FI["bg1"]] = -np.asarray(Sb[p].todense()).ravel()[mcols]
        F[:, FI["d1"]] = -np.asarray(Sd[p].todense()).ravel()[mcols]
        a1 = np.asarray(Sa[p].todense()).ravel()
        i1 = np.asarray(Si[p].todense()).ravel()
        F[:, FI["a1"]] = -a1[mcols]
        F[:, FI["i1"]] = +i1[mcols]
        s1 = a1 - i1                                   # signed one-hop effect at every node
        Ssg = (Sa - Si).tocsr(); Sab = (Sa + Si).tocsr()
        s2 = Ssg.T @ s1                                # signs MULTIPLY along the path: an
        n2 = Sab.T @ np.abs(s1)                        # activator of a repressor RAISES the target
        F[:, FI["a2"]] = -s2[mcols] / np.maximum(n2[mcols], 1.0)
        F[:, FI["i2"]] = np.log1p(n2[mcols])           # two-hop connectivity, sign-free
        F[:, FI["deg_out"]] = np.log1p(deg_out[mcols])
        F[:, FI["deg_in"]] = np.log1p(deg_in[mcols])
        return F

    ACT_F = [FI["a1"], FI["a2"]]
    INH_F = [FI["i1"], FI["i2"]]
    SELF_F = [FI["self"], FI["bg1"], FI["d1"], FI["deg_out"], FI["deg_in"]]

    class SignedGCN(nn.Module):
        """h = sigma( W_act * (activator messages) + W_inh * (inhibitor messages)
                      + W_self * (self terms) ), which is the proposed layer.

        The weights must see DISJOINT feature groups. Applying W_act, W_inh and W_self all to the
        full vector -- which is what the first draft of this class did -- makes their sum a single
        linear map, so the architecture would have been a plain MLP wearing three names and Q7
        would have compared a network against itself."""
        def __init__(s):
            super().__init__()
            s.act = nn.Linear(len(ACT_F), GCN_DIM)
            s.inh = nn.Linear(len(INH_F), GCN_DIM)
            s.slf = nn.Linear(len(SELF_F), GCN_DIM)
            s.out = nn.Sequential(nn.ReLU(), nn.Linear(GCN_DIM, GCN_DIM), nn.ReLU(),
                                  nn.Linear(GCN_DIM, 1))
        def forward(s, x):
            return s.out(s.act(x[:, ACT_F]) + s.inh(x[:, INH_F])
                         + s.slf(x[:, SELF_F])).squeeze(-1)

    perts_u = np.array(sorted(set(pert[usable])))
    say(f"     {len(perts_u):,} distinct perturbed genes enter the held-out comparison")
    permp = rng.permutation(len(perts_u))
    folds = [perts_u[permp[i::NFOLD]] for i in range(NFOLD)]
    row_of = {g: rows_of[g][0] for g in perts_u}

    FEATCACHE = {}
    def feats(g, tag, Sa, Si, Sd, Sb):
        key = (g, tag)
        if key not in FEATCACHE: FEATCACHE[key] = build(g, Sa, Si, Sd, Sb)
        return FEATCACHE[key]

    def run(tag, Sa, Si, Sd, Sb, with_gcn=False):
        acc = {a: [] for a in list(ARMS) + ["C0_MEAN"] + (["C6_SIGNED_GCN"] if with_gcn else [])}
        for te in folds:
            tr = np.setdiff1d(perts_u, te)
            Ytr = X[[row_of[g] for g in tr]]
            mu = Ytr.mean(0)
            P = len(FEATS)
            XtX = np.zeros((P + 1, P + 1)); Xty = np.zeros(P + 1)
            Xs, Ys = [], []
            for g in tr:
                F = feats(g, tag, Sa, Si, Sd, Sb)
                y = X[row_of[g]] - mu
                Xa = np.concatenate([F, np.ones((NG, 1), np.float32)], 1).astype(np.float64)
                XtX += Xa.T @ Xa; Xty += Xa.T @ y.astype(np.float64)
                if with_gcn and len(Xs) < 400: Xs.append(F); Ys.append(y)
            lam = 1e-3 * np.trace(XtX) / (P + 1)
            beta = {}
            for a, names in ARMS.items():
                c = [FI[n] for n in names] + [P]
                A = XtX[np.ix_(c, c)] + lam * np.eye(len(c))
                beta[a] = (np.linalg.solve(A, Xty[c]), c)
            net = None
            if with_gcn:
                torch.manual_seed(0)
                net = SignedGCN()
                opt = torch.optim.Adam(net.parameters(), lr=GCN_LR)
                xt = torch.from_numpy(np.concatenate(Xs, 0))
                yt = torch.from_numpy(np.concatenate(Ys, 0))
                nv = int(0.15 * xt.shape[0])
                xv, yv, xf, yf = xt[:nv], yt[:nv], xt[nv:], yt[nv:]
                lf = nn.MSELoss(); best, bad, bw = 9e9, 0, None
                for ep in range(GCN_EPOCHS):
                    idx = torch.randperm(xf.shape[0])
                    for b0 in range(0, xf.shape[0], 8192):
                        j = idx[b0:b0 + 8192]
                        opt.zero_grad(); lf(net(xf[j]), yf[j]).backward(); opt.step()
                    with torch.no_grad(): v = float(lf(net(xv), yv))
                    if v < best - 1e-6: best, bad, bw = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= GCN_PATIENCE: break
                if bw: net.load_state_dict(bw)
                net.eval()
            for g in te:
                F = feats(g, tag, Sa, Si, Sd, Sb)
                truth = X[row_of[g]] - mu
                Fa = np.concatenate([F, np.ones((NG, 1), np.float32)], 1)
                acc["C0_MEAN"].append(0.0)   # the target IS the residual after the
                # training mean profile, so the mean predictor scores exactly 0 here
                for a in ARMS:
                    b, c = beta[a]
                    acc[a].append(_p(Fa[:, c] @ b, truth))
                if with_gcn:
                    with torch.no_grad():
                        acc["C6_SIGNED_GCN"].append(_p(net(torch.from_numpy(F)).numpy(), truth))
        return {a: np.asarray(v) for a, v in acc.items()}

    def _p(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5: return np.nan
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else np.nan

    say("     fitting ...")
    R = run("real", S_act, S_inh, S_dir, S_bg, with_gcn=True)
    say("     arm scores, correlation across the screened genes, held out by perturbation:")
    for a in ["C0_MEAN"] + list(ARMS) + ["C6_SIGNED_GCN"]:
        say(f"       {a:<20} {np.nanmean(R[a]):+.4f}  (sd {np.nanstd(R[a]):.4f})")
    res["arms"] = {a: float(np.nanmean(R[a])) for a in R}

    # ---------------------------------------------------------------- Q3
    say("Q3 DOES SIGN BEAT NO SIGN?")
    d3, se3, z3 = paired(R["C3_OP_SIGNED"], R["C2_OP_UNSIGNED"])
    say(f"     C3_OP_SIGNED {np.nanmean(R['C3_OP_SIGNED']):+.4f} vs C2_OP_UNSIGNED "
        f"{np.nanmean(R['C2_OP_UNSIGNED']):+.4f} -- SAME edges, sign kept or discarded")
    say(f"     paired {d3:+.4f} +/- {se3:.4f}  ({z3:+.1f} se)")
    G.add("Q3", bool(d3 >= Q3_BAR and z3 >= Q3_SE), stat=float(d3), requires=("Q1",),
          if_true=lambda: f"Q3 PASS -- keeping the sign is worth {d3:+.4f} ({z3:+.1f} se)",
          if_false=lambda: f"Q3 FAIL -- keeping the sign is worth {d3:+.4f} ({z3:+.1f} se) against "
                           f"a {Q3_BAR} bar at {Q3_SE} se")
    res["Q3"] = {"delta": d3, "se": se3, "z": z3}

    # ---------------------------------------------------------------- Q4
    say("Q4 DOES DIRECTION BEAT AN UNDIRECTED GRAPH?")
    d4, se4, z4 = paired(R["C2_OP_UNSIGNED"], R["C1_BIOGRID"])
    say(f"     C2_OP_UNSIGNED {np.nanmean(R['C2_OP_UNSIGNED']):+.4f} vs C1_BIOGRID "
        f"{np.nanmean(R['C1_BIOGRID']):+.4f}   paired {d4:+.4f} +/- {se4:.4f} ({z4:+.1f} se)")
    G.add("Q4", bool(d4 >= Q4_BAR and z4 >= Q3_SE), stat=float(d4), requires=("Q1",),
          if_true=lambda: f"Q4 PASS -- direction is worth {d4:+.4f} over the undirected graph",
          if_false=lambda: f"Q4 FAIL -- direction is worth {d4:+.4f} ({z4:+.1f} se)")
    res["Q4"] = {"delta": d4, "se": se4, "z": z4}

    # ---------------------------------------------------------------- Q5
    say("Q5 CONTROL: PERMUTED SIGNS")
    if d3 < 0.002:
        G.add("Q5", False, stat=float(d3), requires=("Q3",), void_if=True,
              void_reason=f"the real sign advantage is {d3:+.4f}; there is nothing to collapse")
    else:
        FEATCACHE.clear()
        alln = S_act.nnz + S_inh.nnz
        both = (S_act + S_inh).tocoo()
        both.data[:] = 1.0                             # an edge listed as both must not weigh two
        keepm = rng.random(both.nnz) < (S_act.nnz / max(alln, 1))
        Sa2 = sparse.csr_matrix((both.data[keepm], (both.row[keepm], both.col[keepm])), shape=(NN, NN))
        Si2 = sparse.csr_matrix((both.data[~keepm], (both.row[~keepm], both.col[~keepm])), shape=(NN, NN))
        Rs = run("shuf", Sa2, Si2, S_dir, S_bg)
        ds, _, _ = paired(Rs["C3_OP_SIGNED"], Rs["C2_OP_UNSIGNED"])
        f5 = ds / d3
        say(f"     signs permuted across the same edges: {ds:+.4f} against a real {d3:+.4f} "
            f"({f5:.0%})")
        G.add("Q5", bool(f5 <= Q5_MAX), stat=float(f5), requires=("Q3",),
              if_true=lambda: f"Q5 PASS -- collapses to {f5:.0%} with permuted signs",
              if_false=lambda: f"Q5 FAIL -- {f5:.0%} survives permuted signs; the gain is not the "
                               f"sign")
        res["Q5"] = {"real": d3, "shuffled": ds, "fraction": f5}

    # ---------------------------------------------------------------- Q6
    say("Q6 CONTROL: REVERSED EDGES")
    FEATCACHE.clear()
    Rr = run("rev", S_act.T.tocsr(), S_inh.T.tocsr(), S_dir.T.tocsr(), S_bg)
    dreal, _, _ = paired(R["C3_OP_SIGNED"], R["C0_MEAN"])
    drev, _, _ = paired(Rr["C3_OP_SIGNED"], Rr["C0_MEAN"])
    f6 = drev / dreal if abs(dreal) > 1e-9 else float("nan")
    say(f"     every arrow reversed: C3 over C0 falls from {dreal:+.4f} to {drev:+.4f}  ({f6:.0%})")
    G.add("Q6", bool(np.isfinite(f6) and f6 <= 1 - Q6_DROP), stat=float(f6), requires=("Q1",),
          if_true=lambda: f"Q6 PASS -- reversing the arrows removes {1 - f6:.0%}; the direction is "
                          f"load-bearing",
          if_false=lambda: f"Q6 FAIL -- {f6:.0%} survives with every arrow reversed; the graph is "
                           f"acting as an undirected similarity and the arrows are decorative")
    res["Q6"] = {"real": dreal, "reversed": drev, "fraction": f6}

    # ---------------------------------------------------------------- Q7
    say("Q7 DID THE NETWORK HELP, OR THE FEATURES?")
    d7, se7, z7 = paired(R["C6_SIGNED_GCN"], R["C5_SIGNED_RIDGE"])
    say(f"     C6_SIGNED_GCN {np.nanmean(R['C6_SIGNED_GCN']):+.4f} vs C5_SIGNED_RIDGE "
        f"{np.nanmean(R['C5_SIGNED_RIDGE']):+.4f}   paired {d7:+.4f} +/- {se7:.4f} ({z7:+.1f} se)")
    say(f"     loop 241 measured its two MLPs LOSING to their linear twins by -0.0155 and -0.0170")
    G.add("Q7", bool(d7 >= Q7_BAR), stat=float(d7), requires=("Q1",),
          if_true=lambda: f"Q7 PASS -- separate activator and inhibitor weights add {d7:+.4f} over "
                          f"the linear model on identical features",
          if_false=lambda: f"Q7 FAIL -- the signed network adds {d7:+.4f} over its linear twin, "
                           f"against a {Q7_BAR} bar")
    res["Q7"] = {"delta": d7, "se": se7, "z": z7}

    say("Q8 WHAT THIS CANNOT SHOW")
    say("     OmniPath is a literature aggregation: coverage tracks how much a gene has been")
    say("     published on, so a coverage-driven result can look like a mechanism.")
    say("     The readout is CRISPRi in K562 at one timepoint. Two hops of propagation may not")
    say("     have happened by then, which would penalise C4 for assay timing, not structure.")
    say("     An OmniPath sign is a consensus across sources and contexts. This tests the")
    say("     consensus sign, not a K562-specific sign.")
    say("     8,175 of 8,248 genes and 8,917 of 11,258 rows survive screening; nothing here")
    say("     speaks for what was dropped.")

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
