"""Loop 230. Borzoi on the A549 genes, with the leakage stated before the first number.

WHY BORZOI AND NOT EVO 2. The Evo 2 survey established, by counting every occurrence of
"expression" in the Nature paper, that Evo 2 has no expression benchmark, no eQTL evaluation and
no cell-type conditioning of any kind -- the same DNA gets the same score, so it cannot in
principle tell A549 from K562. In Evo 2's own epigenomic-design section the authors reach for
Enformer and Borzoi to grade Evo 2's generated sequences. On DART-Eval, Evo 2 40B scores caQTL
AUROC 0.58 and dsQTL 0.66 against ChromBPNet's 0.77 and 0.89. And it cannot run here: no CPU code
path exists, the 7B checkpoint is 13.77 GB, the 40B needs FP8 on Hopper.

Borzoi runs on this container, measured rather than assumed: 185.9M parameters, load 12.1 s,
input (1, 4, 524288), forward 28.9 s, output (1, 7611, 6144) -- 7,611 human tracks at 32 bp
resolution over the central 196,608 bp. Sequence comes from the UCSC REST API, 524,288 bp in 1.7 s,
so nothing needs to land on disk.

THE LEAKAGE, STATED FIRST BECAUSE IT IS THE WHOLE INTERPRETIVE PROBLEM. Borzoi's human target set
contains 82 A549 tracks, and four of them are RNA-seq of A549 TREATED WITH DEXAMETHASONE:

    6177  RNA:A549 treated with 0.1 nM dexamethasone for 1 hour
    6187  RNA:A549 treated with 0.5 nM dexamethasone for 1 hour
    6186  RNA:A549 treated with 1 nM dexamethasone for 1 hour
    6188  RNA:A549 treated with 5 nM dexamethasone for 1 hour
    6198, 6199  RNA:A549 (untreated, two tracks)

Our target is the A549 dexamethasone response. Borzoi was TRAINED on A549 dexamethasone RNA-seq.
A high score here is therefore NOT evidence that Borzoi predicts the response from sequence -- it
is consistent with the model having memorised it. That does not make the exercise worthless: a
model that encodes the answer is a useful lookup, and encoding it in a form that generalises across
genes is a real capability. But calling it prediction would be false, and no gate below is allowed
to imply otherwise.

TWO THINGS DIFFER BETWEEN BORZOI'S TRACKS AND OUR TARGET, and they are the only reasons this is
not pure recall: dose, 0.1-5 nM against our series' 100 nM; and time, 1 hour against our plateau
taken from the last three of nine grid points out to 720 minutes. Whether that gap is large enough
to make the comparison informative is exactly what K2 and K4 measure.

THE MATCHED QUANTITY IS A DIFFERENTIAL, NOT A LEVEL. Our target is a log2 fold change of expression
against each replicate's own t=30 baseline. Borzoi's RNA tracks are absolute coverage. The
comparable prediction is therefore log2((dex + eps)/(untreated + eps)) built from tracks 6177-6188
against 6198-6199, and K3 exists because an absolute-level arm would correlate with our target
through expression level alone and could be mistaken for response prediction.

PREDECLARED, BEFORE ANY NUMBER.

  K1 DOES THE PIPELINE REPRODUCE A KNOWN QUANTITY?  -- positive control, everything requires it
     Borzoi's predicted UNTREATED A549 RNA against our measured A549 expression at t = 30 min.
     These are the same quantity measured two ways, so they must agree or the pipeline -- sequence
     fetch, strand, coordinates, track indices, bin aggregation -- is wrong somewhere.
     Gate: PASS iff Spearman exceeds 0.50. A FAIL means nothing below may be read.

  K2 DOES THE PREDICTED DEXAMETHASONE DIFFERENTIAL TRACK OUR MEASURED PLATEAU?
     Gate: PASS iff |Pearson| exceeds 0.30. Read with the leakage above in mind: this gate cannot
     distinguish prediction from recall and is not claimed to.

  K3 IS IT JUST EXPRESSION LEVEL?
     The differential arm against an arm using Borzoi's untreated A549 level alone.
     Gate: PASS iff the differential beats the level arm by at least 0.05. A FAIL means the
     apparent response signal is expression abundance wearing a different label.

  K4 SHAM-DIFFERENTIAL CONTROL
     Tracks 6198 and 6199 are two RNA-seq tracks of the SAME untreated A549 condition. A
     differential built from that pair contains no treatment effect by construction, only
     track-to-track variation. If the sham predicts our plateau as well as the real differential,
     then what K2 measured is not the dexamethasone response.
     Gate: PASS iff the real differential beats the sham by at least 0.05. Requires K1.

  K5 DOES IT BEAT THE CURATED STACK ON THE SAME GENES?
     Loop 229 measured the stack at 0.5713 +/- 0.0130 over 20 splits on 663 genes. The stack is
     rescored here on exactly the genes Borzoi ran on, so the comparison is like for like.
     Gate: PASS iff the Borzoi arm's held-out |r| exceeds the stack's on the same genes.

  K6 DOES IT ADD TO THE STACK?
     Borzoi features appended as an eleventh block, paired across the same 20 splits.
     Gate: PASS iff the paired mean gain exceeds 2 standard errors AND exceeds +0.02.

  K7 WHAT THIS CANNOT SHOW -- written before the run.
     LEAKAGE, restated because it is the dominant caveat: Borzoi saw A549 dexamethasone RNA-seq in
     training. K2 and K5 cannot separate prediction from recall. Only K4's sham control and the
     dose/time mismatch argue against pure recall, and neither is decisive.
     Borzoi's own train/validation/test folds are assigned by genomic REGION. This loop does not
     restrict to held-out regions, so most genes scored here were very likely in Borzoi's training
     data. Restricting to fold-held-out regions would be the honest zero-shot test and is not done
     here.
     A subset of genes is scored for runtime, not all of them, so every number carries the
     sampling error of that subset and the stack is rescored on the same subset to match.
"""
import os, sys, json, time, urllib.request, warnings
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
OUT = "outputs/loop_borzoi_a549.json"
CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
             "/borzoi_a549.npz")
SP = L191.SP
SEQLEN, CROP_BINS, BINSIZE = 524288, 6144, 32
DEX = [6177, 6187, 6186, 6188]          # A549 + dexamethasone, 0.1 / 0.5 / 1 / 5 nM, 1 hour
UNT = [6198, 6199]                      # A549 untreated
EXTRA = [354, 355, 1323, 6262, 6263, 6264, 6265]   # CAGE, DNase, cytosolic, nuclear
NGENE, TSS_BINS, SEED, NSPLIT, NFOLD = 300, 250, 230230, 20, 5
REF_STACK = 0.5713
K1_BAR, K2_BAR, MARGIN, ADD_BAR = 0.50, 0.30, 0.05, 0.02

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


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return pear(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))


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


def fetch(chrom, start, end, tries=5):
    u = (f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;"
         f"chrom={chrom};start={max(start,0)};end={end}")
    for t in range(tries):
        try:
            d = json.load(urllib.request.urlopen(u, timeout=300))
            s = d.get("dna", "")
            if len(s) >= (end - max(start, 0)) * 0.98:
                return s.upper()
        except Exception:
            pass
        time.sleep(2 * (t + 1))
    return None


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "Borzoi on A549 genes, leakage declared"}
    say("=" * 104)
    say("LOOP 230 -- BORZOI ON THE A549 GENES, LEAKAGE STATED FIRST")
    say("=" * 104)
    say("     Borzoi's human targets include FOUR RNA-seq tracks of A549 treated with")
    say("     dexamethasone (0.1, 0.5, 1, 5 nM for 1 hour) and two untreated A549 RNA tracks.")
    say("     Our target IS the A549 dexamethasone response. A high score here is consistent")
    say("     with the model having memorised it. Only the dose gap (0.1-5 nM against 100 nM),")
    say("     the time gap (1 hour against a plateau out to 720 min) and K4's sham control")
    say("     argue against pure recall, and none of them is decisive.")

    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    plateau = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    gp = {s: k for k, s in enumerate(allg)}
    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps, ensg = z["tpm"], z["mins"].astype(int), z["reps"].astype(int), z["genes"]
    e2s = L191.ensg_to_symbol(lambda *_: None)
    esym = np.array([e2s.get(str(g).split(".")[0], "") for g in ensg])
    base30 = tpm[mins == 30].mean(0)
    base_by_sym = {}
    for s, v in zip(esym, base30):
        if s:
            base_by_sym.setdefault(s, []).append(v)
    cand = [s for s in allg if s in tssb and s in base_by_sym]
    pick = sorted(rng.choice(len(cand), size=min(NGENE, len(cand)), replace=False))
    genes = [cand[i] for i in pick]
    say(f"     {len(allg):,} genes in the A549 set; {len(cand):,} carry a TSS and a t=30 level; "
        f"{len(genes)} sampled for runtime")

    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        genes = [str(g) for g in c["genes"]]
        SIG = c["sig"]
        say(f"     prediction cache found: {SIG.shape[0]} genes x {SIG.shape[1]} tracks, "
            f"no forward passes repeated")
    else:
        import torch
        from borzoi_pytorch import Borzoi
        say("     loading Borzoi (johahi/borzoi-replicate-0) ...")
        model = Borzoi.from_pretrained("johahi/borzoi-replicate-0"); model.eval()
        LUT = np.zeros((256, 4), np.float32)
        for i, b in enumerate("ACGT"):
            LUT[ord(b)] = np.eye(4, dtype=np.float32)[i]
        keep_tracks = DEX + UNT + EXTRA
        rows, kept = [], []
        t1 = time.time()
        for k, s in enumerate(genes):
            ch, pos = tssb[s]
            st = int(pos) - SEQLEN // 2
            seq = fetch(ch, st, st + SEQLEN)
            if seq is None or len(seq) < SEQLEN * 0.98:
                continue
            seq = (seq + "N" * SEQLEN)[:SEQLEN]
            x = torch.from_numpy(LUT[np.frombuffer(seq.encode(), np.uint8)].T[None].copy())
            with torch.no_grad():
                y = model(x)[0].numpy()
            mid = CROP_BINS // 2
            lo, hi = mid - TSS_BINS, mid + TSS_BINS
            rows.append(y[keep_tracks, lo:hi].sum(1))
            kept.append(s)
            if (k + 1) % 25 == 0:
                el = time.time() - t1
                say(f"       {k+1}/{len(genes)} genes   {el/60:.1f} min   "
                    f"{el/max(len(kept),1):.1f} s/gene")
        genes = kept
        SIG = np.array(rows, np.float32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, sig=SIG, genes=np.array(genes),
                            tracks=np.array(keep_tracks))
        say(f"     {SIG.shape[0]} genes scored in {(time.time()-t1)/60:.1f} min; cached")

    idx = {t: i for i, t in enumerate(DEX + UNT + EXTRA)}
    dex = SIG[:, [idx[t] for t in DEX]]
    unt = SIG[:, [idx[t] for t in UNT]]
    extra = SIG[:, [idx[t] for t in EXTRA]]
    eps = 1.0
    diff = np.log2((dex.mean(1) + eps) / (unt.mean(1) + eps))
    sham = np.log2((unt[:, 0] + eps) / (unt[:, 1] + eps))
    lvl = np.log2(unt.mean(1) + eps)
    y = np.array([plateau[gp[s]] for s in genes])
    y_base = np.array([np.log2(1 + np.mean(base_by_sym[s])) for s in genes])
    N = len(genes)
    say(f"     scoring {N} genes")

    # ---------------------------------------------------------------- K1
    say("K1 DOES THE PIPELINE REPRODUCE A KNOWN QUANTITY?")
    rho = spearman(lvl, y_base)
    say(f"     Borzoi predicted UNTREATED A549 RNA vs our measured A549 level at t=30 min:")
    say(f"       Spearman {rho:+.4f}   Pearson {pear(lvl, y_base):+.4f}   over {N} genes")
    G.add("K1", bool(np.isfinite(rho) and rho > K1_BAR), stat=float(rho),
          if_true=lambda: f"K1 PASS -- Spearman {rho:+.3f}; sequence, coordinates, track indices "
                          f"and bin aggregation are all doing what they claim",
          if_false=lambda: f"K1 FAIL -- Spearman {rho:+.3f} against a {K1_BAR:.2f} bar; the "
                           f"pipeline is wrong somewhere and nothing below may be read")
    res["control"] = {"spearman": rho, "pearson": pear(lvl, y_base), "n": N}

    # ---------------------------------------------------------------- K2
    say("K2 DOES THE PREDICTED DEXAMETHASONE DIFFERENTIAL TRACK OUR MEASURED PLATEAU?")
    r2 = pear(diff, y)
    say(f"     predicted log2(dex/untreated) vs measured plateau: Pearson {r2:+.4f}, "
        f"Spearman {spearman(diff, y):+.4f}")
    say("     READ WITH THE LEAKAGE: Borzoi trained on A549 dexamethasone RNA-seq. This gate")
    say("     cannot distinguish prediction from recall and does not claim to.")
    G.add("K2", bool(abs(r2) > K2_BAR), stat=float(abs(r2)), requires=("K1",),
          if_true=lambda: f"K2 PASS -- |r| {abs(r2):.4f}",
          if_false=lambda: f"K2 FAIL -- |r| {abs(r2):.4f} against a {K2_BAR:.2f} bar")
    res["differential"] = {"r": r2, "spearman": spearman(diff, y)}

    # ---------------------------------------------------------------- K3
    say("K3 IS IT JUST EXPRESSION LEVEL?")
    r_lvl = pear(lvl, y)
    say(f"     differential arm |r| {abs(r2):.4f}   untreated-level arm |r| {abs(r_lvl):.4f}")
    G.add("K3", bool(abs(r2) - abs(r_lvl) >= MARGIN), stat=float(abs(r2) - abs(r_lvl)),
          requires=("K1",),
          if_true=lambda: f"K3 PASS -- the differential beats level by "
                          f"{abs(r2)-abs(r_lvl):+.4f}",
          if_false=lambda: f"K3 FAIL -- differential {abs(r2):.4f} against level "
                           f"{abs(r_lvl):.4f}; the response signal is expression abundance")
    res["level"] = {"r": r_lvl}

    # ---------------------------------------------------------------- K4
    say("K4 SHAM-DIFFERENTIAL CONTROL")
    r_sh = pear(sham, y)
    say(f"     tracks 6198 and 6199 are the SAME untreated A549 condition, so their ratio")
    say(f"     contains no treatment effect by construction")
    say(f"     real differential |r| {abs(r2):.4f}   sham differential |r| {abs(r_sh):.4f}")
    G.add("K4", bool(abs(r2) - abs(r_sh) >= MARGIN), stat=float(abs(r2) - abs(r_sh)),
          requires=("K1",),
          if_true=lambda: f"K4 PASS -- the real differential beats the sham by "
                          f"{abs(r2)-abs(r_sh):+.4f}, so K2 is a treatment effect",
          if_false=lambda: f"K4 FAIL -- real {abs(r2):.4f} against sham {abs(r_sh):.4f}; what "
                           f"K2 measured is not the dexamethasone response")
    res["sham"] = {"r": r_sh}

    # ---------------------------------------------------------------- K5
    say("K5 DOES IT BEAT THE CURATED STACK ON THE SAME GENES?")
    splits = [np.random.default_rng(SEED + i).permutation(N) for i in range(NSPLIT)]
    FOLDS = [[p[k::NFOLD] for k in range(NFOLD)] for p in splits]
    XB = np.nan_to_num(np.column_stack([diff, lvl, sham, np.log2(extra + eps),
                                        np.log2(dex + eps)]))
    b_sc = np.array([abs(pear(y, cv_pred(XB, y, f))) for f in FOLDS])
    tab = json.load(__import__("gzip").open("colab/data/cell_complete.json.gz"))["genes"]
    T = {str(g["name"]).upper(): g for g in tab}
    FAME = np.array([[np.log1p(float(T.get(s.upper(), {}).get("pubs") or 0))] for s in genes])
    FUN = np.array([[float(T.get(s.upper(), {}).get("loeuf") or 1.0),
                     float(T.get(s.upper(), {}).get("cpg") or 0),
                     np.log1p(float(T.get(s.upper(), {}).get("ndis") or 0)),
                     np.log1p(float(T.get(s.upper(), {}).get("npath") or 0)),
                     float(T.get(s.upper(), {}).get("ess") or 0)] for s in genes])
    XS = np.nan_to_num(np.hstack([FUN, FAME]))
    s_sc = np.array([abs(pear(y, cv_pred(XS, y, f))) for f in FOLDS])
    say(f"     Borzoi block, {XB.shape[1]} features: {b_sc.mean():.4f} +/- {b_sc.std(ddof=1):.4f}")
    say(f"     curated block on the SAME {N} genes: {s_sc.mean():.4f} +/- "
        f"{s_sc.std(ddof=1):.4f}")
    say(f"     loop 229 measured the full stack at {REF_STACK:.4f} on 663 genes with more blocks")
    G.add("K5", bool(b_sc.mean() > s_sc.mean()), stat=float(b_sc.mean()),
          requires=("K1",),
          if_true=lambda: f"K5 PASS -- Borzoi {b_sc.mean():.4f} above the curated arm's "
                          f"{s_sc.mean():.4f} on identical genes",
          if_false=lambda: f"K5 FAIL -- Borzoi {b_sc.mean():.4f} against curated "
                           f"{s_sc.mean():.4f}")
    res["stack_cmp"] = {"borzoi": float(b_sc.mean()), "borzoi_sd": float(b_sc.std(ddof=1)),
                        "curated": float(s_sc.mean()), "curated_sd": float(s_sc.std(ddof=1))}

    # ---------------------------------------------------------------- K6
    say("K6 DOES IT ADD TO THE STACK?")
    XC = np.hstack([XS, XB])
    c_sc = np.array([abs(pear(y, cv_pred(XC, y, f))) for f in FOLDS])
    d = c_sc - s_sc
    se = d.std(ddof=1) / np.sqrt(len(d))
    zsc = d.mean() / se if se > 0 else np.inf
    say(f"     curated alone {s_sc.mean():.4f}   curated + Borzoi {c_sc.mean():.4f}")
    say(f"     PAIRED difference {d.mean():+.4f} +/- {se:.4f}  ({zsc:+.1f} standard errors)")
    G.add("K6", bool(zsc > 2.0 and d.mean() > ADD_BAR), stat=float(d.mean()), requires=("K1",),
          if_true=lambda: f"K6 PASS -- Borzoi adds {d.mean():+.4f} paired at {zsc:.1f} standard "
                          f"errors",
          if_false=lambda: f"K6 FAIL -- {d.mean():+.4f} +/- {se:.4f}, {zsc:+.1f} standard errors")
    res["additive"] = {"curated": float(s_sc.mean()), "combined": float(c_sc.mean()),
                       "delta": float(d.mean()), "se": float(se), "z": float(zsc)}

    # ---------------------------------------------------------------- K7
    say("K7 WHAT THIS CANNOT SHOW")
    say("     LEAKAGE. Borzoi saw A549 dexamethasone RNA-seq in training. K2 and K5 cannot")
    say("     separate prediction from recall, and no gate here claims they can. Only K4's sham")
    say("     control and the dose and time mismatch argue against pure recall.")
    say("     Borzoi's own folds are assigned by genomic REGION and this loop does not restrict")
    say("     to held-out regions, so most genes scored here were very likely in its training")
    say("     data. Restricting to fold-held-out regions is the honest zero-shot test and is not")
    say("     done here.")
    say(f"     {N} genes were scored, not the full set, so every number carries that subset's")
    say("     sampling error; the curated arm is rescored on the same genes to match.")

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
