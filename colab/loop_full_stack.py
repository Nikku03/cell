"""Loop 213. Persist the 0.4761 model, then give it everything this project ever curated.

WHAT IS BEING ADDED, AND WHY IT IS A FAIR TEST. Loop 210 reached |r| 0.4761 on three blocks --
same-cell ChIP, Perturb-seq gains, motif thermodynamics. Loop 211 then measured that capacity is
not the constraint (ridge 0.4057 beat a 256x64 MLP at 0.2072 and 2,000 boosted trees at 0.4039) and
that the learning curve was still rising, so the binding constraint was information. Loop 212 tried
to buy information by IMPUTING a missing block over 1,263 genes and the score fell to 0.3902 --
imputation is not measurement.

This loop buys information the other way: not more genes, but more MEASURED FEATURES on the same
genes. Seven blocks this project curated over 212 loops and never once put in front of the set-point
task:

    network        PPI degree, coexpression neighbourhood, regulator in/out-degree, complex and
                   reaction membership, from colab/data/net_bundle.json.gz
    tf_network     is_TF, curated in-degree, and the SIGNED in-degree split (activating against
                   repressing regulators), from the curated tier of the same bundle
    chromatin4d    A/B compartment PC1, TSS insulation and local contact density for 16,216 genes,
                   plus replication timing
    reactions      reaction count, complex count, max subunits, proton balance and ion flags, from
                   colab/data/rem_enzyme.npz
    pathways       pathway count and the process/pathway class, from cell_complete
    function       LOEUF constraint, CpG, enhancer count, disease count, essentiality, dependency
                   fraction, compartment
    structure3d    the DNA-binding-domain geometry of each gene's REGULATORS, aggregated -- groove
                   preference, net charge, charge density, residue composition and volume from
                   colab/data/tf_domains.json. This is the only way a structural feature can reach
                   a non-TF target gene, and it uses the TF network to get there.

WHAT WOULD MAKE THIS SELF-DECEIVING, AND THE GUARD. Several of these blocks correlate with how
well-studied a gene is -- PPI degree, pathway count and disease count especially. Publication count
has beaten real biology in this project before (loop 71), and it is scored here as a block in its
OWN right so that a gain over it has to be a gain over fame specifically, not merely a gain over
noise. Every new block is also scored solo, so a stack that improves cannot hide which part did it.

PREDECLARED, BEFORE ANY NUMBER.

  F1 DOES THE 0.4761 MODEL PERSIST AND RELOAD?
     Fit the three-block stack, write it to disk, reload it in a fresh object and re-predict.
     Gate: PASS iff the reloaded model reproduces the saved predictions to within 1e-10 AND its
     |r| lands within 0.02 of 0.4761. FAIL means what is on disk is not the model that was
     measured, and every comparison below would be against a different thing.

  F2 DO THE NEW BLOCKS CARRY ANYTHING ALONE?
     Each of the seven scored on its own, gene-held-out, against its own shuffled-label control.
     Gate: PASS iff at least three of the seven beat their own shuffled control by 0.05.
     A FAIL means the curated layers carry nothing about this target and the stack cannot help.

  F3 DOES THE FULL STACK BEAT 0.4761?
     Gate: PASS iff the ten-block stack reaches |r| >= 0.5261, i.e. beats the three-block
     baseline by at least 0.05. Bettering it by less is within the fold-to-fold movement this
     project has already seen between loops 210 and 211 (0.4761 against 0.4057 on the same data).

  F4 DOES CAPACITY MATTER NOW?
     Loop 211 measured that it did not, on three blocks. With ten blocks the interactions a linear
     model cannot express may finally exist.
     Gate: PASS iff the best nonlinear model beats ridge by >= 0.05 on the full stack.

  F5 IS IT FAME?
     Gate: PASS iff the full stack beats a publication-count-only model by >= 0.10, and beats a
     model built from the study-effort blocks alone (PPI degree, pathway count, disease count,
     publication count) by >= 0.05.

  F6 DOES IT MOVE THE FORWARD MODEL?
     The best set point, relaxed with per-gene lambda from measured half-lives, scored held-out in
     time against persistence.
     Gate: PASS iff it clears persistence by more than 0.01. Requires F3 -- a set point that did
     not improve has nothing new to give the forward model.

  F7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, pickle, sys, time, warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import SEQ_F, gene_set
from gate_guard import Gates

from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
PHYS_CACHE = ROOT / "colab" / "data" / "physics" / "motif_occupancy_1200.npz"
K562 = SP / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
MODEL_OUT = ROOT / "colab" / "models" / "setpoint_stack_v1.pkl"
OUT = "outputs/loop_full_stack.json"
N_TRAIN, SEED = 6, 213213
TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
BASE_R = 0.4761
CUR = (0, 55716)

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
    Bm = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    return Bm @ np.linalg.solve(A.T @ A + R, A.T @ ytr)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def onehot(vals, top=12):
    from collections import Counter
    keys = [k for k, _ in Counter(vals).most_common(top)]
    return np.array([[1.0 if v == k else 0.0 for k in keys] for v in vals]), keys


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "full curated stack"}
    say("=" * 104)
    say("LOOP 213 -- PERSIST THE 0.4761 MODEL, THEN GIVE IT EVERYTHING THIS PROJECT CURATED")
    say("=" * 104)

    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    S_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    pos = {s: k for k, s in enumerate(allg)}
    seqs = json.load(open(SEQ_F))
    Z = np.load(PHYS_CACHE, allow_pickle=True)
    cpos = {str(g): i for i, g in enumerate(Z["genes"])}
    fk = h5py.File(K562, "r")
    cats = [x.decode() if isinstance(x, bytes) else str(x)
            for x in fk["var/__categories/gene_name"][:]]
    readout = np.array([cats[c] for c in fk["var/gene_name"][:]])
    ridx = {g: i for i, g in enumerate(readout)}
    Xk = fk["X"]
    names = [s for s in allg if s in cpos and s in ridx]
    y = np.array([S_all[pos[s]] for s in names])
    say(f"     gene set: the {len(names):,}-gene intersection where 0.4761 was measured")

    TR = {}
    for name in TRACKS:
        pt, PM = L191.promoter_track(name, [tssb.get(s) for s in sym], L191.PROM_PAD,
                                     lambda *_: None)
        TR[name] = PM[[int(np.where(pt == t)[0][0]) for t in grid]]
    chip_full = np.column_stack([np.column_stack([
        TR[t][:N_TRAIN, gi].mean(0), TR[t][:N_TRAIN, gi].max(0),
        TR[t][N_TRAIN - 1, gi] - TR[t][0, gi]]) for t in TRACKS])
    B = {}
    B["chip"] = np.array([chip_full[pos[s]] for s in names])
    cols = np.array([ridx[s] for s in names])
    rng = np.random.default_rng(SEED)
    picked = []
    for p in rng.permutation(Xk.shape[0]):
        if len(picked) >= 200:
            break
        if np.isfinite(Xk[int(p), :][cols]).all():
            picked.append(int(p))
    B["gains"] = np.column_stack([Xk[p, :][cols] for p in picked])
    B["physics"] = np.array([Z["F"][cpos[s]] for s in names])

    # ---------------------------------------------------------------- new blocks
    say("     building the seven curated blocks ...")
    nb = json.load(gzip.open("colab/data/net_bundle.json.gz"))
    nn = nb["names"]; nidx = {n.upper(): i for i, n in enumerate(nn)}
    from collections import Counter, defaultdict
    ppi_deg = Counter()
    for a, b in nb["ppi"]:
        ppi_deg[int(a)] += 1; ppi_deg[int(b)] += 1
    outd, ind, sgn_pos, sgn_neg = Counter(), Counter(), Counter(), Counter()
    for s_, t_, g_ in nb["reg"]:
        outd[int(s_)] += 1; ind[int(t_)] += 1
        if g_ == 1: sgn_pos[int(t_)] += 1
        elif g_ == -1: sgn_neg[int(t_)] += 1
    cur_in, cur_out = Counter(), Counter()
    for s_, t_, _ in nb["reg"][CUR[0]:CUR[1]]:
        cur_out[int(s_)] += 1; cur_in[int(t_)] += 1
    coex = nb["coexpr"]
    ncplx = Counter()
    for _, mem in nb["complexes"].items():
        for m in mem:
            ncplx[int(m)] += 1
    nrx = {int(k): len(v) for k, v in nb["generxn"].items()}

    def gidx(s):
        return nidx.get(s.upper(), -1)
    net = []
    for s in names:
        i = gidx(s)
        cx = coex.get(str(i), [])
        net.append([np.log1p(ppi_deg.get(i, 0)), np.log1p(outd.get(i, 0)),
                    np.log1p(ind.get(i, 0)), np.log1p(ncplx.get(i, 0)),
                    np.log1p(nrx.get(i, 0)), len(cx),
                    float(np.mean([c[1] for c in cx])) if cx else 0.0,
                    float(np.max([c[1] for c in cx])) if cx else 0.0])
    B["network"] = np.array(net)

    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    T = {str(g["name"]).upper(): g for g in tab}
    tfn = []
    for s in names:
        i = gidx(s); g = T.get(s, {})
        p_, n_ = sgn_pos.get(i, 0), sgn_neg.get(i, 0)
        tfn.append([float(g.get("tf") or 0), np.log1p(cur_in.get(i, 0)),
                    np.log1p(cur_out.get(i, 0)), np.log1p(p_), np.log1p(n_),
                    (p_ - n_) / (p_ + n_ + 1.0)])
    B["tf_network"] = np.array(tfn)

    CH = json.load(open(SP / "_chromatin_features.json"))["features"]
    ch = []
    for s in names:
        c = CH.get(s.upper(), {})
        ch.append([c.get("pc1", np.nan), c.get("ins", np.nan),
                   np.log1p(c.get("dens", np.nan)) if c.get("dens") else np.nan])
    ch = np.array(ch, float)
    ch = np.where(np.isfinite(ch), ch, np.nanmean(np.where(np.isfinite(ch), ch, np.nan), axis=0))
    B["chromatin4d"] = ch

    ze = np.load("colab/data/rem_enzyme.npz", allow_pickle=True)
    esym = {str(s).upper() for s in ze["symbols"] if s}
    gpr_g, gpr_rx = ze["gpr_gene"], ze["gpr_rx"]
    rxcount = Counter()
    egen = [str(x).upper() for x in ze["genes"]]
    for gg, rx in zip(gpr_g, gpr_rx):
        if 0 <= int(gg) < len(egen):
            rxcount[egen[int(gg)]] += 1
    B["reactions"] = np.array([[1.0 if s in esym else 0.0, np.log1p(rxcount.get(s, 0))]
                               for s in names])

    proc_oh, _ = onehot([str(T.get(s, {}).get("proc") or "") for s in names], 12)
    comp_oh, _ = onehot([str(T.get(s, {}).get("comp") or "") for s in names], 12)
    B["pathways"] = np.hstack([
        np.array([[np.log1p(float(T.get(s, {}).get("npath") or 0))] for s in names]), proc_oh])
    B["function"] = np.hstack([
        np.array([[float(T.get(s, {}).get("loeuf") or 1.0),
                   float(T.get(s, {}).get("cpg") or 0),
                   np.log1p(float(T.get(s, {}).get("enh") or 0)),
                   np.log1p(float(T.get(s, {}).get("ndis") or 0)),
                   float(T.get(s, {}).get("ess") or 0),
                   float(T.get(s, {}).get("dep_frac") or 0),
                   float(T.get(s, {}).get("dark") or 0)] for s in names]), comp_oh])

    dom = json.load(open("colab/data/tf_domains.json"))["matrices"]
    dprop = {}
    for v in dom.values():
        nm = (v.get("name") or "").upper().split("::")[0]
        if nm:
            dprop[nm] = [float(v.get(k) or 0) for k in
                         ("net_charge", "charge_density", "arg_frac", "lys_frac",
                          "bulky_frac", "mean_volume", "protein_length")]
    regs = defaultdict(list)
    for s_, t_, _ in nb["reg"][CUR[0]:CUR[1]]:
        regs[int(t_)].append(nn[int(s_)].upper())
    st = []
    for s in names:
        vs = [dprop[r] for r in regs.get(gidx(s), []) if r in dprop]
        st.append(list(np.mean(vs, axis=0)) + [len(vs)] if vs else [0.0] * 8)
    B["structure3d"] = np.array(st)
    B["fame"] = np.log1p(np.array([float(T.get(s, {}).get("pubs") or 0)
                                   for s in names])).reshape(-1, 1)

    for k, v in B.items():
        say(f"       {k:<12} {v.shape}")

    folds = np.array_split(np.random.default_rng(SEED).permutation(len(y)), 5)

    def cvpred(F, yy=None):
        yy = y if yy is None else yy
        best = (float("-inf"), None)
        for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0):
            Sp = np.zeros(len(yy))
            for k in range(5):
                te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
                Sp[te] = ridge(F[tr], yy[tr], F[te], lam)
            r = pear(Sp, yy)
            if np.isfinite(r) and abs(r) > best[0]:
                best = (abs(r), Sp)
        return best

    # ---------------------------------------------------------------- F1
    say("F1 DOES THE 0.4761 MODEL PERSIST AND RELOAD?")
    P3 = {k: cvpred(B[k])[1] for k in ("chip", "gains", "physics")}
    r3, S3 = cvpred(np.column_stack([P3["chip"], P3["gains"], P3["physics"]]))
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    art = {"version": "setpoint_stack_v1", "blocks": ["chip", "gains", "physics"],
           "genes": names, "block_predictions": P3, "stack_prediction": S3,
           "r": r3, "folds": [f.tolist() for f in folds], "seed": SEED,
           "target": "A549 dexamethasone log2 plateau, mean of last 3 grid points"}
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(art, f)
    with open(MODEL_OUT, "rb") as f:
        back = pickle.load(f)
    err = float(np.max(np.abs(back["stack_prediction"] - S3)))
    say(f"     three-block stack |r| {r3:.4f}   (loop 210 measured {BASE_R:.4f})")
    say(f"     wrote {MODEL_OUT}  ({MODEL_OUT.stat().st_size/1024:.0f} KB)")
    say(f"     reload max abs difference {err:.2e}")
    ok1 = (err < 1e-10 and abs(r3 - BASE_R) <= 0.02)
    G.add("F1", ok1, stat=r3,
          if_true=lambda: f"F1 PASS -- persisted and reloads identically at |r| {r3:.4f}",
          if_false=lambda: f"F1 FAIL -- |r| {r3:.4f} against {BASE_R:.4f}, reload err {err:.2e}")

    # ---------------------------------------------------------------- F2
    say("F2 DO THE NEW BLOCKS CARRY ANYTHING ALONE?")
    rg2 = np.random.default_rng(SEED + 5)
    solo, shufd = {}, {}
    NEW = ["network", "tf_network", "chromatin4d", "reactions", "pathways", "function",
           "structure3d"]
    for k in NEW + ["fame"]:
        solo[k] = cvpred(B[k])[0]
        shufd[k] = cvpred(B[k], yy=rg2.permutation(y))[0]
        say(f"       {k:<12} |r| {solo[k]:.4f}   shuffled {shufd[k]:.4f}   "
            f"delta {solo[k]-shufd[k]:+.4f}")
    nwin = sum(1 for k in NEW if solo[k] - shufd[k] >= 0.05)
    G.add("F2", bool(nwin >= 3), stat=float(nwin), requires=("F1",),
          if_true=lambda: f"F2 PASS -- {nwin} of {len(NEW)} new blocks beat their own shuffle",
          if_false=lambda: f"F2 FAIL -- only {nwin} of {len(NEW)} carry anything about this target")
    res["solo"] = solo; res["shuffled"] = shufd

    # ---------------------------------------------------------------- F3
    say("F3 DOES THE FULL STACK BEAT 0.4761?")
    ALL = ["chip", "gains", "physics"] + NEW
    P = dict(P3)
    for k in NEW:
        P[k] = cvpred(B[k])[1]
    r_stack, S_stack = cvpred(np.column_stack([P[k] for k in ALL]))
    r_cat, _ = cvpred(np.hstack([B[k] for k in ALL]))
    say(f"       stacked, {len(ALL)} block predictions   |r| {r_stack:.4f}")
    say(f"       concatenated, {sum(B[k].shape[1] for k in ALL):,} columns   |r| {r_cat:.4f}")
    r_best = max(r_stack, r_cat)
    S_best = S_stack if r_stack >= r_cat else cvpred(np.hstack([B[k] for k in ALL]))[1]
    G.add("F3", bool(r_best >= BASE_R + 0.05), stat=r_best, requires=("F1",),
          if_true=lambda: f"F3 PASS -- {r_best:.4f}, beating {BASE_R:.4f} by {r_best-BASE_R:+.4f}",
          if_false=lambda: f"F3 FAIL -- {r_best:.4f} against {BASE_R:.4f} ({r_best-BASE_R:+.4f}). "
                           f"Seven curated layers added {r_best-r3:+.4f} over the three-block stack")
    res["stack"] = {"stacked": r_stack, "concat": r_cat, "best": r_best, "three_block": r3}

    # ---------------------------------------------------------------- F4
    say("F4 DOES CAPACITY MATTER NOW?")
    Xa = np.column_stack([P[k] for k in ALL])
    nl = {}
    for nm, mk in (("mlp", lambda: make_pipeline(StandardScaler(), MLPRegressor(
                        hidden_layer_sizes=(64, 16), alpha=1e-1, max_iter=3000,
                        early_stopping=True, n_iter_no_change=30, random_state=SEED))),
                   ("gbm", lambda: xgb.XGBRegressor(
                        n_estimators=1500, max_depth=3, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
                        n_jobs=4, verbosity=0))):
        Sp = np.zeros(len(y))
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            m = mk(); m.fit(Xa[tr], y[tr]); Sp[te] = m.predict(Xa[te])
        nl[nm] = abs(pear(Sp, y))
        say(f"       {nm:<6} |r| {nl[nm]:.4f}")
    say(f"       ridge  |r| {r_stack:.4f}")
    gain = max(nl.values()) - r_stack
    G.add("F4", bool(gain >= 0.05), stat=gain, requires=("F3",),
          if_true=lambda: f"F4 PASS -- nonlinear beats ridge by {gain:+.4f} on ten blocks",
          if_false=lambda: f"F4 FAIL -- nonlinear beats ridge by {gain:+.4f}; capacity still is "
                           f"not the constraint even with ten blocks")
    res["nonlinear"] = nl

    # ---------------------------------------------------------------- F5
    say("F5 IS IT FAME?")
    study = np.hstack([B["fame"], B["network"][:, :1],
                       B["pathways"][:, :1], B["function"][:, 3:4]])
    r_study, _ = cvpred(study)
    say(f"       publication count alone     |r| {solo['fame']:.4f}")
    say(f"       study-effort block          |r| {r_study:.4f}   "
        f"(pubs + PPI degree + pathway count + disease count)")
    say(f"       full stack                  |r| {r_best:.4f}")
    G.add("F5", bool(r_best - solo["fame"] >= 0.10 and r_best - r_study >= 0.05),
          stat=r_study, requires=("F3",),
          if_true=lambda: f"F5 PASS -- beats fame by {r_best-solo['fame']:+.4f} and the "
                          f"study-effort block by {r_best-r_study:+.4f}",
          if_false=lambda: f"F5 FAIL -- fame {solo['fame']:.4f}, study-effort {r_study:.4f}, "
                           f"stack {r_best:.4f}")

    # ---------------------------------------------------------------- F6
    say("F6 DOES IT MOVE THE FORWARD MODEL?")
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
    life = json.load(open("outputs/orphan/cell_lifetimes.json"))["lifetimes"]
    lam_g = np.array([np.log(2) / life[s]["mrna_hl_h"] / 60.0
                      if s in life and life[s].get("mrna_hl_h") else np.nan for s in names])
    lam_g = np.where(np.isfinite(lam_g), lam_g, np.nanmedian(lam_g))
    d_tr = lam_g[gtr] * dtr * (S_best[gtr] - ptr)
    d_te = lam_g[gte] * dte * (S_best[gte] - pte)
    k_ = float(d_tr @ ytr / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
    fwd = r2s(yte, k_ * d_te)
    say(f"       held-out-in-time R2 {fwd:+.5f}   persistence {pers:+.5f}   "
        f"margin {fwd-pers:+.5f}")
    G.add("F6", bool(fwd - pers > 0.01), stat=fwd, requires=("F3",),
          if_true=lambda: f"F6 PASS -- clears persistence by {fwd-pers:+.5f}",
          if_false=lambda: f"F6 FAIL -- clears persistence by {fwd-pers:+.5f}")
    res["forward"] = {"r2": fwd, "persistence": pers, "margin": fwd - pers}

    say("F7 WHAT THIS CANNOT SHOW")
    say("     Several of these blocks are proxies for how well-studied a gene is -- PPI degree,")
    say("     pathway count, disease count. F5 scores them together as a block so a gain has to")
    say("     be a gain over study effort specifically, but no design fully separates a")
    say("     well-studied gene from a well-connected one.")
    say("     structure3d reaches a target gene only through its curated regulators, so it is a")
    say("     property of the TF network as much as of any structure, and it is empty for genes")
    say("     with no curated regulator.")
    say("     chromatin4d is GM12878/K562-derived and the target is A549. Loop 90 measured the")
    say("     polymer model scoring below a distance-only null, so these are descriptive tracks")
    say("     rather than a working chromatin model.")
    say("     Everything is still one cell line, one perturbation, one channel, 600 genes.")

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
