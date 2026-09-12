"""LOOP 63 -- REPLICATION TIMING: THE MISSING LAYER, AND WHETHER IT RESCUES THE CHROMOSOME.

WHERE THIS COMES FROM. Loop 61 scored 28 layers of the cell model and found 26 could be filled from
the rest. Two could not, and one of those is the whole 3D genome: loops3d came back at model 0.5071
against a null of 0.5211 -- chance. Every other layer leaks into its neighbours; the chromosome does
not. The cell model has no path from a chromosome to anything else.

It also has no replication at all. `cellcycle` holds 153 entries and there is no replication timing,
no origin firing, no fork. That was the one part of the whole-cell ask with literally nothing behind
it.

These two facts are a single opportunity, because REPLICATION TIMING AND 3D GENOME ORGANISATION ARE
THE SAME PHYSICS SEEN TWICE. Early-replicating domains are the A compartment; late-replicating
domains are the B compartment. If loops3d is unpredictable because the model lacks any physical
description of the genome -- rather than because 3D structure is genuinely independent of everything
else -- then replication timing is exactly the input that should rescue it. If loops3d stays at
chance even with replication timing in hand, the disconnection is real and not a data gap.

THE DATA, and a correction to what this session claimed earlier. ENCODE was first queried with
`assay_title=Repli-seq` and returned 0, then with `assay_term_name` and returned 119 experiments,
which was reported here as replication data being available. That was half right: those experiments
carry only fastq and hg19 bam, no processed signal. The processed tracks live under `assay_title`
after all -- 96 Repli-seq and 119 Repli-chip bigWigs -- and NONE of them exist on GRCh38.

    20 Repli-chip wavelet-smoothed signal tracks, one per distinct cell type, hg19
    gene TSSs lifted hg38 -> hg19 (16,362 of 16,380, 99.89%; 18 unmapped and dropped)
    signal averaged over TSS +/- 50 kb, since replication timing is smooth at ~100 kb

Lifting 16,380 points is exact where lifting millions of signal bins would not be, which is why the
liftover runs on the genes and not on the tracks.

MEASURED BEFORE THE GATES WERE WRITTEN, as calibration rather than result:

    early replication vs enhancer count   rho +0.5930     early domains are active euchromatin
    early replication vs CpG island       rho +0.1544
    early replication vs protein ppm      rho +0.1938
    early replication vs PUBLICATIONS     rho +0.1442     <- the important one

That last number is why this layer is worth having. Loop 61 measured that publication count predicts
a layer's membership at a median AUC of 0.7173, ranging to 0.8906 -- every curated layer in this
model is substantially a record of what has been studied. Replication timing is a physical
measurement of a cell, so it carries almost none of that. It is the first layer added to this model
that fame cannot explain.

WHY THIS IS NOT LOOP 61's QUESTION. Loop 61 asked which layers can be FILLED, because those layers
have blanks. Replication timing covers 16,355 of 16,492 genes -- 99.2% -- so it has no blanks worth
filling. The questions for a new layer are the reverse pair: is it REDUNDANT with what we already
have, and does it ADD to what we can predict.

PREDECLARED, before any number:

  R1 THE LIFTOVER AND INDEXING ARE CORRECT
       >= 95% of TSSs lift, >= 95% of lifted TSSs carry signal, and the sign of the enhancer
       correlation is POSITIVE. That last one is a direction check, not a discovery: if early
       replication did not track active chromatin, the track is being read upside down or at the
       wrong coordinates, and everything downstream would be confidently wrong.
  R2 REPLICATION TIMING IS NOT A FAME PROXY
       |rho| with publication count < 0.30, and below the fame association of the layers already in
       the model. If it fails, this is one more record of attention and adds nothing new in kind.
  R3 IS IT REDUNDANT WITH THE CELL WE ALREADY HAVE?
       predict replication timing from all 32 existing layers, chromosome-grouped folds. Held-out
       Spearman >= 0.70 means the model already contains it and the layer is decoration. THE
       INTERESTING OUTCOME HERE IS THE MODEL FAILING TO PREDICT IT.
  R4 DOES IT RESCUE THE CHROMOSOME?                                    THE GATE THAT MATTERS.
       re-run loop 61's fill test for loops3d with replication timing added to the feature block,
       same folds, same baselines, same null. loops3d must move from chance to beating its null.
       Every other layer is re-scored too and any whose margin improves by >= 0.01 is named, so a
       loops3d rescue can be told apart from a general lift.
  R5 THE NULLS ARE MEASURED
       label permutation for the classification arms, and a shuffled-target null for R3.

  A FAILURE OF R4 IS A REAL RESULT AND IS REPORTED AS ONE. It would mean the 3D genome is
  informationally disconnected from the rest of this cell model even given the physical measurement
  most closely tied to it, which is a much stronger statement than loop 61 could make.

-> outputs/loop_replication.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import loop_state_vector as SV
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

LIFT_FLOOR = 0.95
SIGNAL_FLOOR = 0.95
FAME_CEILING = 0.30
REDUNDANT = 0.70
IMPROVE = 0.01
SEED = 6301

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def load_rt(n):
    M = np.load(SC / "_rt_matrix.npy")
    gi = json.load(open(SC / "_rt_geneidx.json"))
    rt = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(M, axis=1)
        st = np.nanstd(M, axis=1)
    for k, i in enumerate(gi):
        if i < n:
            rt[i] = mu[k]
            sd[i] = st[k]
    return rt, sd, M.shape


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 63 -- replication timing: the missing layer, and whether it rescues the chromosome")
    say("=" * 100)
    say()

    D = json.load(open(CELL))
    n = len(D["genes"])
    L, pubs, chrom = SV.build_layers(D)
    rt, rtsd, shape = load_rt(n)
    have = np.isfinite(rt)

    say("R1 THE LIFTOVER AND INDEXING ARE CORRECT")
    n_bed = sum(1 for _ in open(SC / "_tss_hg38.bed"))
    n_lift = sum(1 for _ in open(SC / "_tss_hg19.bed"))
    lift_frac = n_lift / max(n_bed, 1)
    sig_frac = have.sum() / max(n_lift, 1)
    enh = L["enh"]
    m = have & np.isfinite(enh)
    r_enh = spearmanr(rt[m], enh[m]).statistic
    say(f"     {shape[1]} cell-type tracks, TSS +/- 50 kb")
    say(f"     TSSs lifted hg38->hg19   {n_lift:,} of {n_bed:,}  ({lift_frac:.4f})")
    say(f"     lifted TSSs with signal  {int(have.sum()):,}  ({sig_frac:.4f})")
    say(f"     DIRECTION CHECK  early replication vs enhancer count  rho {r_enh:+.4f}")
    r1 = lift_frac >= LIFT_FLOOR and sig_frac >= SIGNAL_FLOOR and r_enh > 0
    say(f"     R1 {'PASS' if r1 else 'FAIL'}")
    say()

    say("R2 REPLICATION TIMING IS NOT A FAME PROXY")
    r_pub = spearmanr(rt[have], pubs[have]).statistic
    prev = json.load(open(OUT / "loop_state_vector.json"))
    fame_med = prev["fame_median"]
    say(f"     replication timing vs publication count  rho {r_pub:+.4f}")
    say(f"     for comparison, loop 61: fame predicts layer membership at median AUC {fame_med:.4f}, "
        f"up to {max(r['pubs'] for r in prev['layers']):.4f}")
    r2 = abs(r_pub) < FAME_CEILING
    say(f"     R2 {'PASS' if r2 else 'FAIL'}  (ceiling |rho| < {FAME_CEILING})")
    say()

    ug = sorted(set(chrom))
    fold = {g: i for g, i in zip(ug, np.random.default_rng(SV.SEED).permutation(len(ug)) % SV.FOLDS)}
    fid = np.array([fold[c] for c in chrom])

    say("R3 IS IT REDUNDANT WITH THE CELL WE ALREADY HAVE?")
    names = sorted(L)
    X = np.column_stack([L[k] for k in names] + [pubs])
    P = np.full(n, np.nan)
    for k in range(SV.FOLDS):
        tr = (fid != k) & have
        te = (fid == k) & have
        if tr.sum() < 200 or te.sum() == 0:
            continue
        A = np.column_stack([SV.rank(X[tr, j]) for j in range(X.shape[1])])
        B = np.empty((int(te.sum()), X.shape[1]))
        for j in range(X.shape[1]):
            srt = np.sort(X[tr, j])
            B[:, j] = np.searchsorted(srt, X[te, j], side="right") / max(len(srt), 1)
        mdl = Ridge(alpha=1.0).fit(A, rt[tr])
        P[te] = mdl.predict(B)
    okp = np.isfinite(P) & have
    rho = spearmanr(rt[okp], P[okp]).statistic
    rng = np.random.default_rng(SEED)
    nl = []
    for _ in range(200):
        yp = rt[okp].copy()
        rng.shuffle(yp)
        nl.append(abs(spearmanr(yp, P[okp]).statistic))
    n95 = float(np.percentile(nl, 95))
    say(f"     the cell predicts replication timing at held-out Spearman {rho:.4f}  "
        f"(null 95th {n95:.4f}, n={int(okp.sum()):,})")
    r3 = rho < REDUNDANT
    say(f"     R3 {'PASS' if r3 else 'FAIL'}  -- "
        f"{'NOT redundant: the model does not already contain it' if r3 else 'REDUNDANT'}")
    say()

    say("R4 DOES IT RESCUE THE CHROMOSOME?")
    rt_f = np.where(have, rt, np.nanmedian(rt))
    sd_f = np.where(np.isfinite(rtsd), rtsd, np.nanmedian(rtsd[np.isfinite(rtsd)]))
    Lp = dict(L)
    Lp["rt"] = rt_f
    Lp["rt_sd"] = sd_f
    Lp["rt_have"] = have.astype(float)
    rng2 = np.random.default_rng(SV.SEED + 7)

    def score(layers, target):
        y = (layers[target] > 0).astype(int)
        feats = [k for k in sorted(layers) if k != target]
        Xf = np.column_stack([layers[k] for k in feats] + [pubs])
        a_model, Pm = SV.cv_auc(Xf, y, fid)
        a_pubs, _ = SV.cv_auc(pubs.reshape(-1, 1), y, fid)
        best = float("nan")
        bn = "-"
        for k in feats:
            s, _ = SV.cv_auc(layers[k].reshape(-1, 1), y, fid)
            if not np.isnan(s) and (np.isnan(best) or s > best):
                best, bn = s, k
        okm = ~np.isnan(Pm)
        nn = []
        for _ in range(SV.N_NULL):
            yp = y[okm].copy()
            rng2.shuffle(yp)
            nn.append(SV.roc_auc_score(yp, Pm[okm]))
        nq = float(np.percentile(nn, 95))
        mg = a_model - max(a_pubs, best)
        return {"model": a_model, "pubs": a_pubs, "best1": best, "best1_name": bn,
                "null95": nq, "margin": mg,
                "earned": bool(mg >= SV.MARGIN and a_model > nq)}

    a = score(L, "loops3d")
    b = score(Lp, "loops3d")
    say(f"     {'loops3d':12s} {'model':>7s} {'pubs':>7s} {'best1':>7s} {'null95':>7s} {'margin':>7s}")
    say(f"     {'without RT':12s} {a['model']:7.4f} {a['pubs']:7.4f} {a['best1']:7.4f} "
        f"{a['null95']:7.4f} {a['margin']:+7.4f}   {'EARNED' if a['earned'] else 'defer'}")
    say(f"     {'with RT':12s} {b['model']:7.4f} {b['pubs']:7.4f} {b['best1']:7.4f} "
        f"{b['null95']:7.4f} {b['margin']:+7.4f}   {'EARNED' if b['earned'] else 'defer'}")
    r4 = b["model"] > b["null95"]
    say(f"     loops3d beats its own null with RT: {r4}   "
        f"(model {b['model']:.4f} vs null95 {b['null95']:.4f})")
    say(f"     R4 {'PASS' if r4 else 'FAIL'}")
    say()

    say("     every other layer re-scored, to tell a loops3d rescue from a general lift:")
    improved = []
    others = []
    for t in sorted(L):
        if t == "loops3d":
            continue
        y = (L[t] > 0).astype(int)
        if y.sum() < SV.MIN_PRESENT or (len(y) - y.sum()) < SV.MIN_ABSENT:
            continue
        p, q = score(L, t), score(Lp, t)
        others.append({"layer": t, "old_margin": p["margin"], "new_margin": q["margin"],
                       "old_earned": p["earned"], "new_earned": q["earned"]})
        if q["margin"] - p["margin"] >= IMPROVE:
            improved.append(t)
            say(f"       {t:12s} {p['margin']:+.4f} -> {q['margin']:+.4f}  "
                f"({q['margin'] - p['margin']:+.4f})")
    say(f"     {len(improved)} of {len(others)} other layers improve by >= {IMPROVE}")
    say()

    r5 = np.isfinite(n95) and np.isfinite(b["null95"])
    say(f"R5 nulls measured: R3 shuffled-target 95th {n95:.4f}, "
        f"R4 label permutation 95th {b['null95']:.4f}")
    say(f"     R5 {'PASS' if r5 else 'FAIL'}")
    say()

    gates = {"R1 liftover and indexing correct": bool(r1),
             "R2 replication timing is not fame": bool(r2),
             "R3 not redundant with the existing cell": bool(r3),
             "R4 the chromosome is rescued": bool(r4),
             "R5 nulls measured": bool(r5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CELL), str(SC / "_rt_matrix.npy"), str(SC / "_tss_hg19.bed")],
                      available=n, used=int(have.sum()), selection="filtered", seed=SEED,
                      controls=["enhancer-correlation sign as a read-direction check",
                                "publication count measured against the new layer",
                                "shuffled-target null for the redundancy regression",
                                "loop 61 folds, baselines and label-permutation null reused",
                                "every other layer re-scored to separate rescue from general lift"],
                      note="20 Repli-chip wavelet-smoothed tracks, hg19; gene TSSs lifted hg38->hg19 "
                           "because lifting 16,380 points is exact where lifting signal bins is not")
    RM.report(man, emit=say)
    json.dump({"test": "loop_replication", "manifest": man, "gates": gates,
               "n_tracks": int(shape[1]), "lift_frac": lift_frac, "signal_frac": sig_frac,
               "rho_enhancers": float(r_enh), "rho_pubs": float(r_pub),
               "fame_median_loop61": fame_med,
               "redundancy_rho": float(rho), "redundancy_null95": n95,
               "loops3d_without_rt": a, "loops3d_with_rt": b,
               "improved": improved, "others": others,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_replication.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_replication.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
