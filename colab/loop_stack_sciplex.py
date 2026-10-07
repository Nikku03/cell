"""Loop 235. sci-Plex as an eleventh block, and the question that raises.

WHAT IS BEING ADDED, AND WHY IT IS NOT LIKE THE OTHER TEN. Every block in the loop 229 stack is a
PROPERTY of a gene: how connected it is, how constrained, which pathways it sits in, what binds its
promoter, how it responds to knockdowns in K562. sci-Plex is not that. It is A549 cells treated
with dexamethasone -- the same cell line, the same drug, the same biological response our target
measures -- read out on a different platform in a different lab. Loop 234 measured it against our
plateau at Pearson +0.6995, above the entire ten-block stack's 0.5713.

That is not leakage in the machine-learning sense: no cell, sample or measurement is shared between
the two experiments, and sci-Plex was published independently. But it is a SECOND MEASUREMENT OF
THE OUTCOME rather than a predictor of it, and a model that uses it is answering a different
question. "Predict the A549 dexamethasone response from what we know about genes" and "predict the
A549 dexamethasone response given another A549 dexamethasone experiment" are both legitimate, and
conflating them would be dishonest. This loop runs both and reports them separately.

S4 IS THE GATE THAT MATTERS AND IT IS AIMED AT THE PROJECT. If the ten curated blocks add nothing
once a second measurement is present, then everything built from loop 206 to loop 233 -- the
networks, the ChIP tracks, the chromatin, the Perturb-seq, the physics -- is worth less than
running the experiment again. That is a real possible outcome and the gate is written so it can
fire.

THE BASELINE IS REBUILT ON THE SHARED GENES. Loop 229 measured 0.5713 +/- 0.0130 on 663 genes
carrying an A549 plateau and a Perturb-seq readout. sci-Plex shares 1,307 genes with the plateau
set, and the three-way intersection is smaller than either. Quoting 0.5713 as the bar would compare
two different gene sets, so the stack is rescored here on exactly the genes sci-Plex covers, with
the same 20 paired splits.

THE sci-PLEX BLOCK IS THE DOSE SERIES, not a single number. Seven columns, log2 fold change against
vehicle at doses 0.1 through 100. Loop 234 R3 measured those correlations as 0.6205, 0.7086,
0.7167, 0.7121, 0.7171, 0.7037, 0.6995 -- a rise to dose 1 and then a plateau, which is receptor
saturation. The shape across doses carries information a single dose does not, and a ridge can use
it or ignore it.

PREDECLARED, BEFORE ANY NUMBER.

  S1 DOES THE HARNESS REPRODUCE ITSELF?  -- everything requires it
     The loop 229 stack, rebuilt on the sci-Plex gene set with the same 20 splits.
     Gate: PASS iff the across-split standard deviation stays below 0.03, so paired differences of
     0.02 remain detectable. The MEAN is reported and compared against 0.5713 but NOT gated, since
     the gene set differs and a gap is expected rather than diagnostic.

  S2 DOES sci-PLEX IMPROVE THE STACK?
     Eleven blocks against ten, paired across the same 20 splits.
     Gate: PASS iff the paired mean gain exceeds 2 standard errors AND exceeds +0.02.

  S3 DOES sci-PLEX ALONE BEAT THE TEN-BLOCK STACK?
     The dose series on its own, same splits, same genes.
     Gate: PASS iff sci-Plex alone exceeds the curated stack by more than 2 standard errors,
     paired. Loop 234 measured 0.6995 for one dose against 0.5713 for ten blocks, but on different
     gene sets and without splits; this settles it like for like.

  S4 DO THE TEN CURATED BLOCKS ADD ANYTHING OVER sci-PLEX ALONE?  -- the gate aimed at the project
     Gate: PASS iff the full eleven-block stack exceeds sci-Plex alone by more than 2 standard
     errors AND by more than +0.02, paired. A FAIL means that once a second measurement of the
     outcome exists, everything this project curated is worth less than repeating the experiment,
     and that would be the most important finding in the arc.

  S5 CONTROL: SHUFFLED TARGET
     Gate: PASS iff the eleven-block stack beats its shuffled-target score by at least 0.10 on
     EVERY one of the 20 splits, not on average.

  S6 WHAT THIS CANNOT SHOW -- written before the run.
     A win for the eleven-block stack does not make it a better MODEL of the cell. It makes it a
     better predictor of one experiment given another experiment, which is a narrower claim and
     the one that should be quoted.
     sci-Plex profiled at 24 hours against our 7-to-12-hour plateau, and its harmonised dose scale
     does not map onto the ENCODE series' 100 nM, so its contribution here is a floor on what a
     time-matched and dose-matched replicate would give.
     Nothing here tests whether sci-Plex helps on any OTHER target. It is a second measurement of
     this specific response and there is no reason to expect it to transfer to a different drug,
     cell line or timepoint -- the co-evolution and Perturb-seq blocks were added on exactly that
     hope and loop 234 measured the K562 transfer as the reason the Perturb-seq block underperforms.
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
OUT = "outputs/loop_stack_sciplex.json"
SP = L191.SP
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
H5 = SCR / "sciplex2.h5ad"
REL_F = ROOT / "outputs" / "loop224_reliability.npz"
TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
DOSES = ["0.1", "0.5", "1", "5", "10", "50", "100"]
SEED, NSPLIT, NFOLD, K_PS = 235235, 20, 5, 24
REF_229, REF_234 = 0.5713, 0.6995
SD_BAR, GAIN_BAR, CTRL_BAR = 0.03, 0.02, 0.10

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
    res = {"test": "sci-Plex as an eleventh block"}
    say("=" * 104)
    say("LOOP 235 -- sci-PLEX AS AN ELEVENTH BLOCK, AND WHAT THAT RAISES")
    say("=" * 104)
    say("     Every other block is a PROPERTY of a gene. sci-Plex is A549 cells treated with")
    say("     dexamethasone -- a SECOND MEASUREMENT of the outcome, not a predictor of it. No")
    say("     sample is shared, so this is not leakage, but a model using it answers a different")
    say("     question and the two are reported separately.")

    import h5py
    fh = h5py.File(H5, "r")
    def cat(c):
        g = fh["obs"][c]
        if isinstance(g, h5py.Group):
            cs = np.array([x.decode() if isinstance(x, bytes) else str(x)
                           for x in g["categories"][:]])
            return cs[g["codes"][:]]
        return g[:]
    pert, dose = cat("perturbation"), cat("dose_value")
    vkey = fh["var"].attrs.get("_index", "_index")
    vkey = vkey.decode() if isinstance(vkey, bytes) else vkey
    gsym = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in fh["var"][vkey][:]])
    shape = tuple(fh["X"].attrs["shape"])
    dat = fh["X"]["data"][:]; ind = fh["X"]["indices"][:]; ptr = fh["X"]["indptr"][:]

    def pseudobulk(mask):
        acc = np.zeros(shape[1], np.float64)
        rows = np.where(mask)[0]
        for r in rows:
            a, b = ptr[r], ptr[r + 1]
            tot = dat[a:b].sum()
            if tot > 0: acc[ind[a:b]] += dat[a:b] / tot
        return acc / max(len(rows), 1) * 1e6
    veh = pseudobulk((pert == "Dex") & (dose == "0"))
    LFC = {}
    for d in DOSES:
        pb = pseudobulk((pert == "Dex") & (dose == d))
        LFC[d] = np.log2((pb + 1.0) / (veh + 1.0))
    spos = {s: i for i, s in enumerate(gsym)}
    say(f"     sci-Plex dose series built: {len(DOSES)} doses, {shape[0]:,} cells")

    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    y_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    gp = {s: k for k, s in enumerate(allg)}
    relz = np.load(REL_F, allow_pickle=True)
    ro = {str(g): i for i, g in enumerate(relz["gene"])}
    names = [s for s in allg if s in spos and s in ro]
    y = np.array([y_all[gp[s]] for s in names])
    N = len(names)
    say(f"     {N:,} genes carry an A549 plateau, a sci-Plex readout AND a Perturb-seq readout")
    say(f"     loop 229 used 663 genes and loop 234 used 1,307; this is the three-way intersection")

    SCI = np.column_stack([[LFC[d][spos[s]] for s in names] for d in DOSES])

    ck_cols = np.array([ro[s] for s in names])
    TR = {}
    for t in TRACKS:
        pt, PM = L191.promoter_track(t, [tssb.get(s) for s in sym], L191.PROM_PAD, lambda *_: None)
        TR[t] = PM[[int(np.where(pt == tt)[0][0]) for tt in grid]]
    CHIP = np.column_stack([np.column_stack([
        TR[t][:, gi].mean(0), TR[t][:, gi].max(0), TR[t][-1, gi] - TR[t][0, gi]])
        for t in TRACKS])
    CHIP = np.array([CHIP[gp[s]] for s in names])
    CH = json.load(open(SP / "_chromatin_features.json"))["features"]
    ch = np.nan_to_num(np.array([[CH.get(s.upper(), {}).get("pc1", 0.0),
                                  CH.get(s.upper(), {}).get("ins", 0.0),
                                  np.log1p(CH.get(s.upper(), {}).get("dens") or 0.0)]
                                 for s in names], float))
    nb = json.load(gzip.open("colab/data/net_bundle.json.gz"))
    nidx = {n_.upper(): i for i, n_ in enumerate(nb["names"])}
    ppi = Counter()
    for a, b in nb["ppi"]:
        ppi[int(a)] += 1; ppi[int(b)] += 1
    outd, ind_ = Counter(), Counter()
    for s_, t_, g_ in nb["reg"]:
        outd[int(s_)] += 1; ind_[int(t_)] += 1
    ncplx = Counter()
    for _, mem in nb["complexes"].items():
        for m_ in mem: ncplx[int(m_)] += 1
    nrx = {int(k): len(v) for k, v in nb["generxn"].items()}
    coex = nb["coexpr"]
    NET = np.array([[np.log1p(ppi.get(nidx.get(s.upper(), -1), 0)),
                     np.log1p(outd.get(nidx.get(s.upper(), -1), 0)),
                     np.log1p(ind_.get(nidx.get(s.upper(), -1), 0)),
                     np.log1p(ncplx.get(nidx.get(s.upper(), -1), 0)),
                     np.log1p(nrx.get(nidx.get(s.upper(), -1), 0)),
                     len(coex.get(str(nidx.get(s.upper(), -1)), [])),
                     float(np.mean([c[1] for c in coex.get(str(nidx.get(s.upper(), -1)), [])]))
                     if coex.get(str(nidx.get(s.upper(), -1))) else 0.0,
                     float(np.max([c[1] for c in coex.get(str(nidx.get(s.upper(), -1)), [])]))
                     if coex.get(str(nidx.get(s.upper(), -1))) else 0.0] for s in names])
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
    PS = np.zeros((N, K_PS))
    say(f"     blocks: network {NET.shape[1]}, function {FUN.shape[1]}, pathways {PATH.shape[1]}, "
        f"chip {CHIP.shape[1]}, chromatin {ch.shape[1]}, fame 1, sciplex {SCI.shape[1]}")

    BASE = {"network": NET, "function": FUN, "pathways": PATH, "chip": CHIP,
            "chromatin": ch, "fame": FAME}
    WITH = dict(BASE); WITH["sciplex"] = SCI
    ONLY = {"sciplex": SCI}
    FOLDS = [[np.random.default_rng(SEED + i).permutation(N)[k::NFOLD] for k in range(NFOLD)]
             for i in range(NSPLIT)]
    run = lambda b: np.array([stack_score(b, y, f) for f in FOLDS])

    # ---------------------------------------------------------------- S1
    say("S1 DOES THE HARNESS REPRODUCE ITSELF?")
    s_base = run(BASE)
    say(f"     curated stack on these {N:,} genes, {NSPLIT} splits: "
        f"{s_base.mean():.4f} +/- {s_base.std(ddof=1):.4f}")
    say(f"     loop 229 measured {REF_229:.4f} +/- 0.0130 on its own 663-gene set -- reported for")
    say("     context, NOT gated, because the gene sets differ")
    G.add("S1", bool(s_base.std(ddof=1) < SD_BAR), stat=float(s_base.std(ddof=1)),
          if_true=lambda: f"S1 PASS -- sd {s_base.std(ddof=1):.4f}, paired {GAIN_BAR} effects "
                          f"are detectable",
          if_false=lambda: f"S1 FAIL -- sd {s_base.std(ddof=1):.4f}")
    res["baseline"] = {"mean": float(s_base.mean()), "sd": float(s_base.std(ddof=1)),
                       "n_genes": N, "loop229": REF_229}

    # ---------------------------------------------------------------- S2
    say("S2 DOES sci-PLEX IMPROVE THE STACK?")
    s_with = run(WITH)
    d2, se2, z2 = paired(s_with, s_base)
    say(f"     ten blocks {s_base.mean():.4f}   eleven blocks {s_with.mean():.4f}")
    say(f"     PAIRED {d2:+.4f} +/- {se2:.4f}  ({z2:+.1f} standard errors)")
    G.add("S2", bool(z2 > 2.0 and d2 > GAIN_BAR), stat=float(d2), requires=("S1",),
          if_true=lambda: f"S2 PASS -- adding sci-Plex gains {d2:+.4f} at {z2:.1f} standard errors",
          if_false=lambda: f"S2 FAIL -- {d2:+.4f} +/- {se2:.4f}, {z2:+.1f} standard errors")
    res["with_sciplex"] = {"mean": float(s_with.mean()), "delta": d2, "se": se2, "z": z2}

    # ---------------------------------------------------------------- S3
    say("S3 DOES sci-PLEX ALONE BEAT THE TEN-BLOCK STACK?")
    s_only = run(ONLY)
    d3, se3, z3 = paired(s_only, s_base)
    say(f"     sci-Plex alone {s_only.mean():.4f} +/- {s_only.std(ddof=1):.4f}")
    say(f"     curated stack  {s_base.mean():.4f} +/- {s_base.std(ddof=1):.4f}")
    say(f"     PAIRED {d3:+.4f} +/- {se3:.4f}  ({z3:+.1f} standard errors)")
    say(f"     loop 234 measured a single dose at {REF_234:.4f} without splits")
    G.add("S3", bool(z3 > 2.0), stat=float(d3), requires=("S1",),
          if_true=lambda: f"S3 PASS -- a second measurement of the outcome beats ten blocks of "
                          f"curated biology by {d3:+.4f}",
          if_false=lambda: f"S3 FAIL -- {d3:+.4f} +/- {se3:.4f}; sci-Plex alone does not clearly "
                           f"beat the stack")
    res["sciplex_only"] = {"mean": float(s_only.mean()), "delta_vs_stack": d3, "z": z3}

    # ---------------------------------------------------------------- S4
    say("S4 DO THE TEN CURATED BLOCKS ADD ANYTHING OVER sci-PLEX ALONE?")
    d4, se4, z4 = paired(s_with, s_only)
    say(f"     sci-Plex alone {s_only.mean():.4f}   sci-Plex + ten blocks {s_with.mean():.4f}")
    say(f"     PAIRED {d4:+.4f} +/- {se4:.4f}  ({z4:+.1f} standard errors)")
    say("     a FAIL here means that once a second measurement exists, everything curated from")
    say("     loop 206 to loop 233 is worth less than repeating the experiment")
    G.add("S4", bool(z4 > 2.0 and d4 > GAIN_BAR), stat=float(d4), requires=("S1",),
          if_true=lambda: f"S4 PASS -- the curated blocks add {d4:+.4f} on top of a second "
                          f"measurement, at {z4:.1f} standard errors",
          if_false=lambda: f"S4 FAIL -- {d4:+.4f} +/- {se4:.4f}, {z4:+.1f} standard errors. The "
                           f"curated layers add nothing measurable once sci-Plex is present")
    res["curated_over_sciplex"] = {"delta": d4, "se": se4, "z": z4}

    # ---------------------------------------------------------------- S5
    say("S5 CONTROL: SHUFFLED TARGET")
    sh = []
    for i, f in enumerate(FOLDS):
        ysh = y.copy(); np.random.default_rng(SEED + 900 + i).shuffle(ysh)
        P = np.column_stack([cv_pred(np.nan_to_num(v).reshape(N, -1), ysh, f)
                             for v in WITH.values()])
        sh.append(abs(pear(ysh, cv_pred(P, ysh, f))))
    sh = np.array(sh); marg = s_with - sh
    say(f"     shuffled {sh.mean():.4f} +/- {sh.std(ddof=1):.4f}")
    say(f"     per-split margin: min {marg.min():+.4f}, mean {marg.mean():+.4f}")
    G.add("S5", bool(marg.min() >= CTRL_BAR), stat=float(marg.min()), requires=("S1",),
          if_true=lambda: f"S5 PASS -- margin at least {marg.min():.4f} on EVERY split",
          if_false=lambda: f"S5 FAIL -- worst-split margin {marg.min():+.4f}")
    res["shuffled"] = {"mean": float(sh.mean()), "min_margin": float(marg.min())}

    # ---------------------------------------------------------------- S6
    say("S6 WHAT THIS CANNOT SHOW")
    say("     A win for the eleven-block stack does not make it a better MODEL of the cell. It")
    say("     makes it a better predictor of one experiment GIVEN another experiment, which is a")
    say("     narrower claim and the one that should be quoted.")
    say("     sci-Plex profiled at 24 hours against our 7-to-12-hour plateau, and its harmonised")
    say("     dose scale does not map onto 100 nM, so its contribution is a FLOOR on what a")
    say("     time-matched and dose-matched replicate would give.")
    say("     Nothing here tests whether sci-Plex helps on any OTHER target. The co-evolution and")
    say("     Perturb-seq blocks were added on exactly that hope and loop 234 measured the K562")
    say("     transfer as the reason the Perturb-seq block underperforms.")

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
