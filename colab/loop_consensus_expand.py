"""Loop 237. Eleven more arms, and an honest accounting of how independent they are not.

WHERE THIS STARTS. Loop 236 reached 0.9281 +/- 0.0003 by averaging three independent A549
dexamethasone experiments -- sci-Plex, GSE229084 and the sign-flipped washout GSE144662 -- against
sci-Plex alone at 0.7474 on the same genes and splits. Loop 232's noise law says error falls as
n^(-0.59) when measurements are independent, so more arms should help. The question is how many are
really available and how independent they really are.

WHAT WAS FOUND. GSE144660 is the third series in the same causal-network family as GSE144662:

    A549 cells stimulated with dexamethasone, profiled at 00h, 01h, 04h, 08h and 12h, in
    ELEVEN separate arms -- a control vector plus ten transcription-factor overexpressions
    (CEBPB, CEBPD, FOSL2, FOXO1, FOXO3, KLF15, KLF6, KLF9, OCT4, TFCP2L1). 19,815
    protein-coding genes as corrected log TPM, three replicates per timepoint.

The control arm is the closest match to our target that exists anywhere: A549, dexamethasone, a
12-hour window against our plateau's 7-to-12 hours, unperturbed. The ten overexpression arms are the
same experiment with one transcription factor forced up, so each still contains the dexamethasone
response measured independently.

AND HERE IS THE PROBLEM WITH SIMPLY ADDING THEM. GSE144660 and GSE144662 are the SAME STUDY: same
laboratory, same cell stock, same library preparation, same processing pipeline down to the
surrogate-variable correction in the filename. Eleven arms from one study are not eleven independent
experiments. They share every systematic error the protocol has, and loop 232's n^(-0.59) law
applies only to INDEPENDENT noise. Averaging correlated measurements buys far less, and claiming
otherwise by counting arms would be the same error as counting significance instead of magnitude.

V4 MEASURES THAT RATHER THAN ASSERTING IT. If arms from one study agree with each other much more
than they agree with arms from other studies, the extra arms are partly redundant, and the gap
between within-study and between-study agreement is the size of the shared systematic component.
That number bounds what the expansion can deliver, and it is computed before the consensus is
scored.

PREDECLARED, BEFORE ANY NUMBER.

  V1 DO THE NEW ARMS PARSE AND COVER OUR GENES?  -- everything requires it
     Gate: PASS iff all eleven arms load and at least 500 genes are shared with the A549 plateau
     set.

  V2 DOES THE MATCHED 12-HOUR TIME COURSE RECOVER OUR PLATEAU?
     GSE144660 control arm, 12h against 00h.
     Gate: PASS iff Pearson exceeds +0.60. The bar is set high deliberately: this is the closest
     match available -- same cell line, same drug, and a time window overlapping our plateau's --
     so a weak correlation would mean something is wrong with the pairing rather than with the
     biology. Loop 236 measured GSE229084's 2 h arm at +0.6292 and its 18 h arm at +0.8975.

  V3 DO THE TEN OVEREXPRESSION ARMS RECOVER IT TOO?
     Each arm's 12h-against-00h contrast against our plateau.
     Gate: PASS iff at least 8 of 10 exceed +0.50. Every arm contains dexamethasone, so every arm
     should carry the response; an arm that does not would mean its overexpressed factor blocks
     the programme, which is a finding rather than a failure and would be reported as such.

  V4 HOW INDEPENDENT ARE THEY REALLY?  -- the honest accounting
     Median pairwise agreement WITHIN the GSE144660 family against median agreement BETWEEN
     studies (sci-Plex, GSE229084, GSE144662 family).
     Gate: PASS iff within-study agreement EXCEEDS between-study agreement, which would confirm a
     shared systematic component and mean the arms must be discounted. The gate is written so that
     the expected answer FAILS to support naive arm-counting, and both numbers are reported.

  V5 DOES THE EXPANDED CONSENSUS BEAT LOOP 236's 0.9281?
     All arms, same 20 paired splits, scored on the genes common to everything.
     Gate: PASS iff the paired gain over the loop 236 three-arm consensus exceeds 2 standard
     errors.

  V6 DOES IT REACH 0.95?  -- the target asked for
     Gate: PASS iff the expanded consensus mean exceeds 0.95.

  V7 CONTROL: SHUFFLED TARGET ON EVERY SPLIT
     Gate: PASS iff the expanded consensus beats its shuffled-target score by at least 0.10 on
     EVERY one of the 20 splits. With eighteen correlated arms and 5-fold cross-validation this is
     the gate that asks whether the ridge is fitting the target or the noise.

  V8 WHAT THIS CANNOT SHOW -- written before the run.
     Every arm is a MEASUREMENT of the outcome, not a prediction of it. Reaching 0.95 would mean
     the A549 dexamethasone response is almost perfectly reproducible across laboratories. It would
     say nothing new about whether the cell can be modelled -- loop 235's S4 measured that at
     +0.0081 for ten blocks of curated biology on top of one such measurement.
     Eleven of the arms come from one study and share its systematic errors, so the effective
     number of independent measurements is smaller than the arm count and V4 measures by how much.
     The overexpression arms have a transcription factor forced up, so their dexamethasone response
     is measured in a perturbed background. That makes them independent in their noise but not
     unbiased in their signal, and a consensus including them is a consensus over perturbed states.
"""
import os, sys, json, gzip, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_consensus_expand.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAT = SCR / "matched"
OE = ["ctrl", "CEBPB", "CEBPD", "FOSL2", "FOXO1", "FOXO3", "KLF15", "KLF6", "KLF9",
      "OCT4", "TFCP2L1"]
DOSES = ["0.1", "0.5", "1", "5", "10", "50", "100"]
SEED, NSPLIT, NFOLD = 237237, 20, 5
REF_236 = 0.9281
MIN_GENES, V2_BAR, V3_BAR, TARGET = 500, 0.60, 0.50, 0.95

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


def read_tsv(path):
    with gzip.open(path, "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        cols = hdr[1:] if len(hdr) > 1 else hdr
        names, rows = [], []
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            names.append(p[0])
            rows.append([float(x) if x not in ("", "NA", "NaN") else np.nan for x in p[1:]])
    return np.array(names), np.array(cols), np.array(rows, float)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "expanded matched consensus"}
    say("=" * 104)
    say("LOOP 237 -- ELEVEN MORE ARMS, AND HOW INDEPENDENT THEY ARE NOT")
    say("=" * 104)
    say("     GSE144660: A549 + dexamethasone at 00h, 01h, 04h, 08h, 12h in eleven arms -- a")
    say("     control vector plus ten TF overexpressions. The control arm is the closest match to")
    say("     our target that exists: same cell line, same drug, 12 h against our plateau's 7-12 h.")
    say("     But GSE144660 and GSE144662 are the SAME STUDY. Eleven arms from one study are not")
    say("     eleven independent experiments, and V4 measures by how much rather than asserting it.")

    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    plateau = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    gp = {s: k for k, s in enumerate(allg)}
    e2s = L191.ensg_to_symbol(lambda *_: None)

    # ---------------------------------------------------------------- V1
    say("V1 DO THE NEW ARMS PARSE AND COVER OUR GENES?")
    ARM = {}
    sym144 = None
    for a in OE:
        p = MAT / f"OE_{a}.txt.gz"
        if not p.exists():
            say(f"       {a}: MISSING"); continue
        n, c, X = read_tsv(p)
        if sym144 is None:
            sym144 = np.array([e2s.get(str(g).split(".")[0], "") for g in n])
            i144 = {}
            for i, s in enumerate(sym144):
                if s and s not in i144: i144[s] = i
        t0i = [i for i, cc in enumerate(c) if ".00h." in cc]
        t12 = [i for i, cc in enumerate(c) if ".12h." in cc]
        if not t0i or not t12:
            say(f"       {a}: no 00h/12h columns"); continue
        ARM[a] = X[:, t12].mean(1) - X[:, t0i].mean(1)
    shared = [s for s in allg if s in i144]
    say(f"     {len(ARM)} of {len(OE)} arms loaded; {len(shared):,} genes shared with the plateau")
    G.add("V1", bool(len(ARM) == len(OE) and len(shared) >= MIN_GENES), stat=float(len(ARM)),
          if_true=lambda: f"V1 PASS -- all {len(ARM)} arms, {len(shared):,} shared genes",
          if_false=lambda: f"V1 FAIL -- {len(ARM)} arms, {len(shared):,} shared genes")
    y_sh = np.array([plateau[gp[s]] for s in shared])
    res["coverage"] = {"n_arms": len(ARM), "n_genes": len(shared)}

    # ---------------------------------------------------------------- V2
    say("V2 DOES THE MATCHED 12-HOUR TIME COURSE RECOVER OUR PLATEAU?")
    xc = np.array([ARM["ctrl"][i144[s]] for s in shared])
    rc = pear(xc, y_sh)
    say(f"     GSE144660 control arm, 12h against 00h: Pearson {rc:+.4f}")
    say(f"     loop 236 measured GSE229084 at +0.6292 (2 h) and +0.8975 (18 h)")
    G.add("V2", bool(rc > V2_BAR), stat=float(rc), requires=("V1",),
          if_true=lambda: f"V2 PASS -- {rc:+.4f}; the closest available match behaves like one",
          if_false=lambda: f"V2 FAIL -- {rc:+.4f} against a +{V2_BAR:.2f} bar")
    res["ctrl_arm"] = {"r": rc}

    # ---------------------------------------------------------------- V3
    say("V3 DO THE TEN OVEREXPRESSION ARMS RECOVER IT TOO?")
    oe_r = {}
    for a in OE:
        if a == "ctrl" or a not in ARM: continue
        xv = np.array([ARM[a][i144[s]] for s in shared])
        oe_r[a] = pear(xv, y_sh)
        say(f"       OE {a:<9} Pearson {oe_r[a]:+.4f}")
    npass = sum(1 for v in oe_r.values() if v > V3_BAR)
    say(f"     {npass} of {len(oe_r)} exceed +{V3_BAR:.2f}")
    G.add("V3", bool(npass >= 8), stat=float(npass), requires=("V1",),
          if_true=lambda: f"V3 PASS -- {npass}/{len(oe_r)} arms carry the response",
          if_false=lambda: f"V3 FAIL -- only {npass}/{len(oe_r)} exceed +{V3_BAR:.2f}; the arms "
                           f"below it name factors whose overexpression blunts the programme")
    res["oe_arms"] = dict(oe_r)

    # ---------------------------------------------------------------- build every arm
    import h5py
    fh = h5py.File(SCR / "sciplex2.h5ad", "r")
    def cat(c):
        g = fh["obs"][c]
        cs = np.array([x.decode() if isinstance(x, bytes) else str(x)
                       for x in g["categories"][:]])
        return cs[g["codes"][:]]
    pert, dose = cat("perturbation"), cat("dose_value")
    vk = fh["var"].attrs.get("_index", "_index")
    vk = vk.decode() if isinstance(vk, bytes) else vk
    gsym = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in fh["var"][vk][:]])
    shp = tuple(fh["X"].attrs["shape"])
    dt, ix, pt = fh["X"]["data"][:], fh["X"]["indices"][:], fh["X"]["indptr"][:]
    def pb(mask):
        acc = np.zeros(shp[1])
        for r in np.where(mask)[0]:
            a, b = pt[r], pt[r + 1]
            s_ = dt[a:b].sum()
            if s_ > 0: acc[ix[a:b]] += dt[a:b] / s_
        return acc / max(mask.sum(), 1) * 1e6
    veh = pb((pert == "Dex") & (dose == "0"))
    SCI = {d: np.log2((pb((pert == "Dex") & (dose == d)) + 1) / (veh + 1)) for d in DOSES}
    spos = {s: i for i, s in enumerate(gsym)}

    n229, c229, X229 = read_tsv(MAT / "GSE229084_geneCounts.tab.gz")
    i229 = {g: i for i, g in enumerate(n229)}
    def lfc229(tp, cp):
        ti = [i for i, c in enumerate(c229) if tp in c]
        ci = [i for i, c in enumerate(c229) if cp in c]
        t = X229[:, ti].mean(1); c = X229[:, ci].mean(1)
        t = t / max(t.sum(), 1) * 1e6; c = c / max(c.sum(), 1) * 1e6
        return np.log2((t + 1.0) / (c + 1.0))
    L2H = lfc229("16h_DMSO__2h_DEX+DMSO", "18h_DMSO")
    L18 = lfc229("2h_DEX__16h_DEX+DMSO", "18h_DMSO")

    n62, c62, X62 = read_tsv(MAT / ("GSE144662_A549.rna_seq.dedex.featurecounts.genes.TPM."
                                    "selected_samples.surrogate_variables_reduced.corrected_ln."
                                    "protein_coding.txt.gz"))
    s62 = np.array([e2s.get(str(g).split(".")[0], "") for g in n62])
    i62 = {}
    for i, s in enumerate(s62):
        if s and s not in i62: i62[s] = i
    t0b = [i for i, c in enumerate(c62) if ".00h." in c]
    t12b = [i for i, c in enumerate(c62) if ".12h." in c]
    WASH = -(X62[:, t12b].mean(1) - X62[:, t0b].mean(1))

    common = [s for s in allg if s in spos and s in i229 and s in i144 and s in i62]
    y = np.array([plateau[gp[s]] for s in common])
    N = len(common)
    say(f"     {N:,} genes present in EVERY arm and our plateau")

    A_SCI = {f"sciplex_{d}": np.array([SCI[d][spos[s]] for s in common]) for d in DOSES}
    A_229 = {"gse229084_2h": np.array([L2H[i229[s]] for s in common]),
             "gse229084_18h": np.array([L18[i229[s]] for s in common])}
    A_662 = {"gse144662_washout": np.array([WASH[i62[s]] for s in common])}
    A_660 = {f"gse144660_{a}": np.array([ARM[a][i144[s]] for s in common]) for a in ARM}
    FAMILY = {**A_660, **A_662}
    OTHER = {**A_SCI, **A_229}

    # ---------------------------------------------------------------- V4
    say("V4 HOW INDEPENDENT ARE THEY REALLY?")
    fk = list(FAMILY); ok_ = list(OTHER)
    within = [abs(pear(FAMILY[a], FAMILY[b])) for i, a in enumerate(fk) for b in fk[i + 1:]]
    between = [abs(pear(FAMILY[a], OTHER[b])) for a in fk for b in ok_]
    mw, mb = float(np.median(within)), float(np.median(between))
    say(f"     within the GSE144660/144662 family ({len(fk)} arms): median |r| {mw:.4f}")
    say(f"     between that family and other studies ({len(ok_)} arms): median |r| {mb:.4f}")
    say(f"     gap {mw-mb:+.4f} -- the size of the shared systematic component")
    G.add("V4", bool(mw > mb), stat=float(mw - mb), requires=("V1",),
          if_true=lambda: f"V4 PASS -- within-study {mw:.3f} exceeds between-study {mb:.3f}, so "
                          f"the eleven arms share systematic error and must be discounted; arm "
                          f"count is NOT independent-measurement count",
          if_false=lambda: f"V4 FAIL -- within {mw:.3f} against between {mb:.3f}; the arms are as "
                           f"independent of each other as they are of other studies")
    res["independence"] = {"within": mw, "between": mb, "gap": mw - mb,
                           "n_family": len(fk), "n_other": len(ok_)}

    # ---------------------------------------------------------------- V5
    say("V5 DOES THE EXPANDED CONSENSUS BEAT LOOP 236's 0.9281?")
    FOLDS = [[np.random.default_rng(SEED + i).permutation(N)[k::NFOLD] for k in range(NFOLD)]
             for i in range(NSPLIT)]
    THREE = np.column_stack([*A_SCI.values(), A_229["gse229084_18h"],
                             A_662["gse144662_washout"], A_229["gse229084_2h"]])
    ALL = np.column_stack([*A_SCI.values(), *A_229.values(), *A_662.values(), *A_660.values()])
    s3 = np.array([abs(pear(y, cv_pred(THREE, y, f))) for f in FOLDS])
    sa = np.array([abs(pear(y, cv_pred(ALL, y, f))) for f in FOLDS])
    d5 = sa - s3
    se5 = d5.std(ddof=1) / np.sqrt(len(d5))
    z5 = d5.mean() / se5 if se5 > 0 else np.inf
    say(f"     loop 236 arms rebuilt here ({THREE.shape[1]} columns): {s3.mean():.4f} +/- "
        f"{s3.std(ddof=1):.4f}")
    say(f"     all arms ({ALL.shape[1]} columns):                    {sa.mean():.4f} +/- "
        f"{sa.std(ddof=1):.4f}")
    say(f"     PAIRED {d5.mean():+.4f} +/- {se5:.4f}  ({z5:+.1f} standard errors)")
    G.add("V5", bool(z5 > 2.0), stat=float(d5.mean()), requires=("V1",),
          if_true=lambda: f"V5 PASS -- the expansion gains {d5.mean():+.4f} at {z5:.1f} standard "
                          f"errors",
          if_false=lambda: f"V5 FAIL -- {d5.mean():+.4f} +/- {se5:.4f}")
    res["expansion"] = {"three": float(s3.mean()), "all": float(sa.mean()),
                        "delta": float(d5.mean()), "z": float(z5), "n_cols": int(ALL.shape[1])}

    # ---------------------------------------------------------------- V6
    say("V6 DOES IT REACH 0.95?")
    say(f"     expanded consensus {sa.mean():.4f} +/- {sa.std(ddof=1):.4f}   target {TARGET:.2f}")
    say(f"     loop 236 reached {REF_236:.4f} on its own gene set")
    G.add("V6", bool(sa.mean() > TARGET), stat=float(sa.mean()), requires=("V1",),
          if_true=lambda: f"V6 PASS -- {sa.mean():.4f}, past {TARGET:.2f}",
          if_false=lambda: f"V6 FAIL -- {sa.mean():.4f} against {TARGET:.2f}")
    res["target"] = {"reached": float(sa.mean()), "bar": TARGET, "loop236": REF_236}

    # ---------------------------------------------------------------- V7
    say("V7 CONTROL: SHUFFLED TARGET ON EVERY SPLIT")
    sh = []
    for i, f in enumerate(FOLDS):
        ysh = y.copy(); np.random.default_rng(SEED + 900 + i).shuffle(ysh)
        sh.append(abs(pear(ysh, cv_pred(ALL, ysh, f))))
    sh = np.array(sh); marg = sa - sh
    say(f"     shuffled {sh.mean():.4f} +/- {sh.std(ddof=1):.4f}")
    say(f"     per-split margin: min {marg.min():+.4f}, mean {marg.mean():+.4f}")
    say(f"     with {ALL.shape[1]} correlated columns this is the gate that asks whether the ridge")
    say("     is fitting the target or the noise")
    G.add("V7", bool(marg.min() >= 0.10), stat=float(marg.min()), requires=("V1",),
          if_true=lambda: f"V7 PASS -- margin at least {marg.min():.4f} on EVERY split",
          if_false=lambda: f"V7 FAIL -- worst-split margin {marg.min():+.4f}")
    res["shuffled"] = {"mean": float(sh.mean()), "min_margin": float(marg.min())}

    # ---------------------------------------------------------------- V8
    say("V8 WHAT THIS CANNOT SHOW")
    say("     Every arm is a MEASUREMENT of the outcome, not a prediction of it. Reaching 0.95")
    say("     means the A549 dexamethasone response is almost perfectly reproducible across")
    say("     laboratories. It says nothing new about whether the cell can be modelled -- loop 235")
    say("     S4 measured that at +0.0081 for ten blocks of curated biology.")
    say("     Eleven arms come from one study and share its systematic errors, so the effective")
    say("     number of independent measurements is smaller than the arm count; V4 measures it.")
    say("     The overexpression arms have a transcription factor forced up, so their response is")
    say("     measured in a PERTURBED background -- independent in their noise, not unbiased in")
    say("     their signal, and a consensus including them is a consensus over perturbed states.")

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
