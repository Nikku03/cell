"""Loop 236. Three more independent A549 dexamethasone experiments, and a predicted sign flip.

WHY MORE OF THESE. Loop 234 found that an independent A549 dexamethasone experiment (sci-Plex 2)
recovers our plateau at +0.6995, and loop 235 measured that the dose series alone reaches 0.7709
against the whole ten-block curated stack's 0.5205 -- and that adding those ten blocks on top buys
+0.0081. For this target, a second measurement is worth about thirty times everything curated. The
obvious question is whether that was luck with one dataset, and the obvious test is more of them.

WHAT GEO HAS, after searching A549 against dexamethasone, glucocorticoid and NR3C1:

    GSE229084   A549 + dex at 100 nM -- OUR EXACT DOSE -- for 2 h and for 18 h, against DMSO,
                AND with three separate GR blockers: mifepristone, CORT113176, and KH-103, a
                GR-PROTAC that degrades the receptor outright. 35,943 genes by symbol.
    GSE144662   A549 stimulated with dex for 12 h, then the drug WASHED OUT, profiled every hour
                for the 12 h that follow. 12 timepoints x 3 replicates, 19,815 protein-coding
                genes as corrected log TPM.
    GSE304966   A549 + dex 2 h with photo-activated GR degraders. UNUSABLE: the counts matrix
                has 36 header fields and 36 data fields with the first field numeric, so it
                carries no gene identifiers at all, and the alternative is an .rds this container
                cannot read. Recorded rather than silently dropped.

GSE144662 IS THE INTERESTING ONE AND IT LETS ME PREDICT A SIGN BEFORE LOOKING. Our target is what
happens when dexamethasone is ADDED. GSE144662 measures what happens when it is TAKEN AWAY. If both
are reading the same glucocorticoid programme, then genes our plateau puts UP must come DOWN as the
drug clears, and the correlation between the washout trajectory and our plateau must be NEGATIVE.
A positive or absent correlation would mean one of the two is not measuring the glucocorticoid
response. Predicting the direction in advance is a much stronger test than predicting a magnitude,
because noise has no sign.

AND THE BLOCKER ARMS ARE A SHARPER CONTROL THAN LOOP 234 HAD. There, the control was three
different drugs -- if Nutlin predicted our dex plateau, the signal was generic stress. Here the drug
is the SAME and the RECEPTOR is removed. Mifepristone and CORT113176 antagonise GR; KH-103 degrades
it. If the dex signature survives with GR blocked, it was never GR-mediated. That is a mechanistic
control, not a contrast control.

PREDECLARED, BEFORE ANY NUMBER.

  U1 DO THE FILES PARSE AND COVER OUR GENES?  -- everything requires it
     Gate: PASS iff both usable datasets map at least 500 genes onto the A549 plateau set.

  U2 DOES DEX AT 100 nM RECOVER OUR PLATEAU?
     GSE229084, 2 h dex + DMSO against 18 h DMSO, on shared genes.
     Gate: PASS iff Pearson exceeds +0.30 -- POSITIVE, since this is dexamethasone added, the same
     direction as our target. A negative correlation here would refute the pairing outright.

  U3 DOES THE 18-HOUR ARM ALSO RECOVER IT, AND BETTER?
     The same dataset's 18 h dex arm. Our plateau is 7-12 h, so 18 h should sit closer to it than
     2 h does.
     Gate: PASS iff the 18 h correlation exceeds the 2 h correlation. This is a prediction about
     which timepoint matches better, made before either is computed.

  U4 DOES WASHOUT RUN THE OTHER WAY?  -- the sign prediction
     GSE144662, 12 h after removal against the moment of removal.
     Gate: PASS iff Pearson is BELOW -0.30. The sign is the test. A value near zero or above it
     means these are not the same programme.

  U5 GR BLOCKADE CONTROL
     The same 2 h dex contrast run under mifepristone, CORT113176 and KH-103.
     Gate: PASS iff every blocked arm falls below HALF the unblocked correlation. Same drug, same
     cells, receptor removed -- if the signature survives, it was never GR-mediated and U2 means
     something other than what it appears to.

  U6 DO THE INDEPENDENT EXPERIMENTS AGREE WITH EACH OTHER?
     sci-Plex, GSE229084 and the sign-flipped GSE144662, correlated pairwise on shared genes.
     Gate: PASS iff the median pairwise |correlation| exceeds 0.40. Agreement between experiments
     that share no samples, platform or laboratory is the strongest evidence available here that
     the response itself is reproducible.

  U7 DOES A CONSENSUS BEAT sci-PLEX ALONE?
     All available matched measurements averaged after sign correction, scored against our plateau
     on the same 20 paired splits loop 235 used.
     Gate: PASS iff the consensus exceeds sci-Plex alone at 0.7709 by more than 2 standard errors,
     paired. Loop 232's noise law says averaging independent measurements should help; this is that
     law applied across experiments rather than across cells.

  U8 WHAT THIS CANNOT SHOW -- written before the run.
     Every arm here is a MEASUREMENT of the outcome, not a prediction of it. A consensus that
     reaches 0.85 would mean the response is highly reproducible across laboratories, not that we
     can model it. Loop 235's S4 already measured what the curated biology adds on top of one such
     measurement: +0.0081.
     GSE229084's blocker arms differ from its dex arm in more than GR occupancy -- mifepristone and
     CORT113176 have their own off-target effects and KH-103 removes a protein the cell uses for
     other things. A collapse under blockade is consistent with GR-dependence but does not isolate
     it.
     GSE144662's washout begins after 12 h of dex, so its 00h timepoint is a fully induced state
     rather than a naive one. The decay it measures is of an established response, and the genes
     with the slowest turnover will look least affected regardless of how strongly they responded.
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
OUT = "outputs/loop_matched_consensus.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAT = SCR / "matched"
G229 = MAT / "GSE229084_geneCounts.tab.gz"
G144 = MAT / ("GSE144662_A549.rna_seq.dedex.featurecounts.genes.TPM.selected_samples."
              "surrogate_variables_reduced.corrected_ln.protein_coding.txt.gz")
SCIPLEX = SCR / "sciplex2.h5ad"
DOSES = ["0.1", "0.5", "1", "5", "10", "50", "100"]
SEED, NSPLIT, NFOLD = 236236, 20, 5
REF_SCIPLEX = 0.7709
MIN_GENES, R_BAR, AGREE_BAR = 500, 0.30, 0.40

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
    res = {"test": "independent matched A549 dexamethasone experiments"}
    say("=" * 104)
    say("LOOP 236 -- MORE MATCHED EXPERIMENTS, AND A SIGN PREDICTED BEFORE LOOKING")
    say("=" * 104)
    say("     Our target is dexamethasone ADDED. GSE144662 measures dexamethasone TAKEN AWAY.")
    say("     If both read the same programme, the correlation must be NEGATIVE. The sign is the")
    say("     test, and noise has no sign.")
    say("     GSE304966 was found and is UNUSABLE: its counts matrix has 36 header fields, 36 data")
    say("     fields and a numeric first field, so it carries no gene identifiers.")

    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    plateau = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    gp = {s: k for k, s in enumerate(allg)}

    # ---------------------------------------------------------------- U1
    say("U1 DO THE FILES PARSE AND COVER OUR GENES?")
    n229, c229, X229 = read_tsv(G229)
    say(f"     GSE229084: {X229.shape[0]:,} genes x {X229.shape[1]} samples")
    n144, c144, X144 = read_tsv(G144)
    e2s = L191.ensg_to_symbol(lambda *_: None)
    s144 = np.array([e2s.get(str(g).split(".")[0], "") for g in n144])
    say(f"     GSE144662: {X144.shape[0]:,} genes x {X144.shape[1]} samples, "
        f"{int((s144!='').sum()):,} mapped to symbols")
    sh229 = [s for s in allg if s in set(n229)]
    sh144 = [s for s in allg if s in set(s144)]
    say(f"     shared with the A549 plateau set: GSE229084 {len(sh229):,}, "
        f"GSE144662 {len(sh144):,}")
    G.add("U1", bool(len(sh229) >= MIN_GENES and len(sh144) >= MIN_GENES),
          stat=float(min(len(sh229), len(sh144))),
          if_true=lambda: f"U1 PASS -- {len(sh229):,} and {len(sh144):,} genes shared",
          if_false=lambda: f"U1 FAIL -- {len(sh229):,} and {len(sh144):,} genes shared")
    res["coverage"] = {"gse229084": len(sh229), "gse144662": len(sh144)}

    def cols_like(cols, pat, exclude=()):
        return [i for i, c in enumerate(cols)
                if pat in c and not any(e in c for e in exclude)]

    def lfc229(treat_pat, ctrl_pat, excl=()):
        ti = cols_like(c229, treat_pat, excl); ci = cols_like(c229, ctrl_pat, excl)
        if not ti or not ci: return None, 0, 0
        t = X229[:, ti].mean(1); c = X229[:, ci].mean(1)
        t = t / max(t.sum(), 1) * 1e6; c = c / max(c.sum(), 1) * 1e6
        return np.log2((t + 1.0) / (c + 1.0)), len(ti), len(ci)

    i229 = {g: i for i, g in enumerate(n229)}
    y229 = np.array([plateau[gp[s]] for s in sh229])

    # ---------------------------------------------------------------- U2
    say("U2 DOES DEX AT 100 nM RECOVER OUR PLATEAU?")
    lf2h, nt, nc = lfc229("16h_DMSO__2h_DEX+DMSO", "18h_DMSO")
    x2h = np.array([lf2h[i229[s]] for s in sh229])
    r2h = pear(x2h, y229)
    say(f"     2 h dex + DMSO ({nt} samples) against 18 h DMSO ({nc}): Pearson {r2h:+.4f}")
    say(f"     GSE229084 used dexamethasone at 100 nM, the same dose as the ENCODE A549 series")
    G.add("U2", bool(r2h > R_BAR), stat=float(r2h), requires=("U1",),
          if_true=lambda: f"U2 PASS -- {r2h:+.4f}, positive as required",
          if_false=lambda: f"U2 FAIL -- {r2h:+.4f} against a +{R_BAR:.2f} bar")
    res["dex_2h"] = {"r": r2h, "n_treat": nt, "n_ctrl": nc}

    # ---------------------------------------------------------------- U3
    say("U3 DOES THE 18-HOUR ARM RECOVER IT BETTER?")
    lf18, nt2, nc2 = lfc229("2h_DEX__16h_DEX+DMSO", "18h_DMSO")
    x18 = np.array([lf18[i229[s]] for s in sh229])
    r18 = pear(x18, y229)
    say(f"     18 h dex ({nt2} samples) against 18 h DMSO: Pearson {r18:+.4f}")
    say(f"     our plateau is 7-12 h, so 18 h should sit closer than 2 h -- predicted before")
    say(f"     either was computed")
    G.add("U3", bool(r18 > r2h), stat=float(r18 - r2h), requires=("U1",),
          if_true=lambda: f"U3 PASS -- 18 h {r18:+.4f} above 2 h {r2h:+.4f}; the closer timepoint "
                          f"matches better",
          if_false=lambda: f"U3 FAIL -- 18 h {r18:+.4f} against 2 h {r2h:+.4f}")
    res["dex_18h"] = {"r": r18}

    # ---------------------------------------------------------------- U4
    say("U4 DOES WASHOUT RUN THE OTHER WAY?")
    t0i = [i for i, c in enumerate(c144) if ".00h." in c]
    t12 = [i for i, c in enumerate(c144) if ".12h." in c]
    d144 = X144[:, t12].mean(1) - X144[:, t0i].mean(1)
    i144 = {}
    for i, s in enumerate(s144):
        if s and s not in i144: i144[s] = i
    xw = np.array([d144[i144[s]] for s in sh144])
    yw = np.array([plateau[gp[s]] for s in sh144])
    rw = pear(xw, yw)
    say(f"     12 h after removal ({len(t12)} samples) minus the moment of removal ({len(t0i)}): "
        f"Pearson {rw:+.4f}")
    say("     the SIGN was predicted before this was computed: dex added goes up, dex removed")
    say("     comes back down, so a shared programme must give a NEGATIVE correlation")
    G.add("U4", bool(rw < -R_BAR), stat=float(rw), requires=("U1",),
          if_true=lambda: f"U4 PASS -- {rw:+.4f}, negative as predicted; the washout reverses our "
                          f"plateau",
          if_false=lambda: f"U4 FAIL -- {rw:+.4f} against a -{R_BAR:.2f} bar; the predicted sign "
                           f"did not appear")
    res["washout"] = {"r": rw}

    # ---------------------------------------------------------------- U5
    say("U5 GR BLOCKADE CONTROL")
    blocked = {}
    for nm, pat in (("mifepristone", "16h_MIF___2h_DEX+MIF"),
                    ("CORT113176", "16h_Cort113__2h_DEX+Cort113"),
                    ("KH-103 (GR-PROTAC)", "16h_KH103__2h_DEX+KH103")):
        lf, a, b = lfc229(pat, "18h_DMSO")
        if lf is None: continue
        xv = np.array([lf[i229[s]] for s in sh229])
        blocked[nm] = pear(xv, y229)
        say(f"       {nm:<22} {a} samples   Pearson {blocked[nm]:+.4f}")
    say(f"       {'unblocked (DMSO)':<22} {nt} samples   Pearson {r2h:+.4f}")
    worst = max(blocked.values()) if blocked else float("nan")
    say("     same drug, same cells, receptor antagonised or degraded")
    G.add("U5", bool(np.isfinite(worst) and worst < 0.5 * r2h), stat=float(worst),
          requires=("U2",),
          if_true=lambda: f"U5 PASS -- every blocked arm falls below half the unblocked "
                          f"correlation, worst {worst:+.4f} against {r2h:+.4f}",
          if_false=lambda: f"U5 FAIL -- a blocked arm reaches {worst:+.4f} against the unblocked "
                           f"{r2h:+.4f}; the signature survives GR removal and U2 is not what it "
                           f"appears to be")
    res["blockade"] = dict(blocked); res["blockade"]["unblocked"] = r2h

    # ---------------------------------------------------------------- U6
    say("U6 DO THE INDEPENDENT EXPERIMENTS AGREE WITH EACH OTHER?")
    import h5py
    fh = h5py.File(SCIPLEX, "r")
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
        acc = np.zeros(shp[1]); 
        for r in np.where(mask)[0]:
            a, b = pt[r], pt[r + 1]
            s_ = dt[a:b].sum()
            if s_ > 0: acc[ix[a:b]] += dt[a:b] / s_
        return acc / max(mask.sum(), 1) * 1e6
    veh = pb((pert == "Dex") & (dose == "0"))
    sci = np.log2((pb((pert == "Dex") & (dose == "100")) + 1) / (veh + 1))
    spos = {s: i for i, s in enumerate(gsym)}
    common = [s for s in allg if s in spos and s in i229 and s in i144]
    say(f"     {len(common):,} genes present in all three experiments and our plateau")
    A = np.array([sci[spos[s]] for s in common])
    B = np.array([lf18[i229[s]] for s in common])
    C = -np.array([d144[i144[s]] for s in common])          # sign-corrected washout
    prs = {"sciPlex vs GSE229084": pear(A, B), "sciPlex vs GSE144662": pear(A, C),
           "GSE229084 vs GSE144662": pear(B, C)}
    for k, v in prs.items(): say(f"       {k:<26} {v:+.4f}")
    med = float(np.median(list(prs.values())))
    say(f"     median pairwise {med:+.4f}; no samples, platform or laboratory shared")
    G.add("U6", bool(med > AGREE_BAR), stat=med, requires=("U1",),
          if_true=lambda: f"U6 PASS -- median pairwise agreement {med:+.4f} across three "
                          f"independent experiments",
          if_false=lambda: f"U6 FAIL -- median pairwise {med:+.4f} against a {AGREE_BAR:.2f} bar")
    res["agreement"] = dict(prs)

    # ---------------------------------------------------------------- U7
    say("U7 DOES A CONSENSUS BEAT sci-PLEX ALONE?")
    y = np.array([plateau[gp[s]] for s in common])
    N = len(common)
    FOLDS = [[np.random.default_rng(SEED + i).permutation(N)[k::NFOLD] for k in range(NFOLD)]
             for i in range(NSPLIT)]
    SCI_ONLY = np.column_stack([[np.log2((pb((pert == "Dex") & (dose == d)) + 1) / (veh + 1))
                                 [spos[s]] for s in common] for d in DOSES])
    CONS = np.column_stack([SCI_ONLY, B, C, np.array([x2h[sh229.index(s)] if s in sh229 else 0.0
                                                      for s in common])])
    s_sci = np.array([abs(pear(y, cv_pred(SCI_ONLY, y, f))) for f in FOLDS])
    s_con = np.array([abs(pear(y, cv_pred(CONS, y, f))) for f in FOLDS])
    d7 = s_con - s_sci
    se7 = d7.std(ddof=1) / np.sqrt(len(d7))
    z7 = d7.mean() / se7 if se7 > 0 else np.inf
    say(f"     sci-Plex dose series alone {s_sci.mean():.4f} +/- {s_sci.std(ddof=1):.4f}")
    say(f"     consensus of all three      {s_con.mean():.4f} +/- {s_con.std(ddof=1):.4f}")
    say(f"     PAIRED {d7.mean():+.4f} +/- {se7:.4f}  ({z7:+.1f} standard errors)")
    say(f"     loop 235 measured sci-Plex alone at {REF_SCIPLEX:.4f} on its own gene set")
    G.add("U7", bool(z7 > 2.0), stat=float(d7.mean()), requires=("U1",),
          if_true=lambda: f"U7 PASS -- the consensus gains {d7.mean():+.4f} at {z7:.1f} standard "
                          f"errors; averaging independent experiments helps as loop 232's law "
                          f"predicts",
          if_false=lambda: f"U7 FAIL -- {d7.mean():+.4f} +/- {se7:.4f}, {z7:+.1f} standard errors")
    res["consensus"] = {"sciplex": float(s_sci.mean()), "consensus": float(s_con.mean()),
                        "delta": float(d7.mean()), "se": float(se7), "z": float(z7),
                        "n_genes": N}

    # ---------------------------------------------------------------- U8
    say("U8 WHAT THIS CANNOT SHOW")
    say("     Every arm here is a MEASUREMENT of the outcome, not a prediction of it. A consensus")
    say("     reaching 0.85 would mean the response is highly reproducible across laboratories,")
    say("     not that we can model it. Loop 235 S4 already measured what the curated biology adds")
    say("     on top of one such measurement: +0.0081.")
    say("     The blocker arms differ from the dex arm in more than GR occupancy -- mifepristone")
    say("     and CORT113176 have off-target effects and KH-103 removes a protein used elsewhere.")
    say("     A collapse under blockade is consistent with GR-dependence without isolating it.")
    say("     GSE144662's washout starts after 12 h of dex, so its 00h is a fully induced state.")
    say("     It measures decay of an established response, and slow-turnover genes will look")
    say("     least affected however strongly they responded.")

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
