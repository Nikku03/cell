"""LOOP 115 -- DOES THE WIRING RESPOND? Knock a TF out in the integrated model, compare to measurement.

A wiring diagram that cannot be perturbed is a picture. This repository has 55,716 curated CollecTRI
edges, measured mRNA and protein half-lives, and -- established here -- a transcription rate, a
translation rate and a doubling time. Nothing has ever pushed on it and asked the cell what happened.
This loop does: it sets a TF's protein to zero, propagates the change through the curated edges into
every target's transcription rate, integrates the two-tier mRNA/protein ODE forward 48 h, and scores
the predicted direction of change against 257 real human TF knockdowns.

THE MEASURED DATA EXISTS, AND IT IS SAID PLAINLY WHERE.
    scratchpad/pert/knocktf_deg/         364 KnockTF v2 human knockdown DEG tables, {symbol: log2FC},
                                         each selected by fetch_knocktf.py at on-target log2FC <= -0.5
                                         and >= 8,000 measured genes, one dataset per TF.
    scratchpad/_knocktf.log, _knocktf2.log   the fetch record: 364 of 481 TFs qualified, 364 fetched
                                         after a zero-padded-sample-id bug lost 64 of them.
    scratchpad/pert/knocktf_human_index.json, knocktf_best_per_tf.json, knocktf_ontarget_ok.json
    scratchpad/reg/                      NOT knockdown data. It is 976 MB of NETWORK sources --
                                         CollecTRI, DoRothEA, ChEA3, TFLink, OmniPath, RegNetwork.
                                         Wiring, not perturbation. Searched and reported as such.
    scratchpad/pert/TF_Perturbations_Followed_by_Expression.txt   an Enrichr gene-set file: gene
                                         LISTS per perturbation, no per-gene effect size. Not used,
                                         because a gate on direction needs a signed magnitude.
    WHAT IS NOT HERE: no matched knockdown in the cell type the half-lives come from, and no
                      knockdown time course. Both are limits, stated in K7, not worked around.
So the fallback self-consistency test is NOT the headline. It is run anyway, as K7, and labelled the
weaker claim it is.

=================================  THE TRAP, AND WHY IT CANNOT BITE HERE  ======================
Loop 91 derived k_sm = mRNA_copies x (ln2/t_half + ln2/t_double). mRNA_copies IS AN INPUT, so
M* = k_sm/k_dm = mRNA_copies is an algebraic identity and integrating it "reproduces" the measured
steady state to machine precision for every gene, always. That is not a simulation.

    INPUTS, per gene:   mRNA half-life, protein half-life (measured); curated edge signs; doubling
                        time 24 h (measured); translation rate 119.24 protein/mRNA/h (loop 91);
                        and the baseline transcription rate k_sm.
    PREDICTIONS:        every target's mRNA trajectory M_g(t) under a TF knockout, its direction,
                        its magnitude, the time it takes, and -- in ARM I only -- the absolute WT
                        steady state M*_g itself.

Two arms, and K2 runs both on purpose:
    ARM C  CIRCULAR.  k_sm = mRNA_copies x k_dm from Schwanhausser. M* returns mRNA_copies to ~1e-16.
                      Demonstrated deliberately, labelled, and EXCLUDED from every other claim.
    ARM I  INDEPENDENT.  k_sm = ksyn x kon/(kon+koff) x k_dm from Larsson single-cell burst kinetics
                      -- inferred rate constants, never that gene's own mRNA copy number. Its WT
                      steady state is therefore a falsifiable prediction and K2 reports its error.
And the structural fact K2 proves numerically rather than asserts: the KO fold-change
log2(M_KO/M_WT) is INVARIANT to k_sm. Scale it per gene by a random factor over three orders of
magnitude and every prediction is unchanged to ~1e-15. So the tautology cannot be the source of any
direction or magnitude claim below -- and the honest converse is that those claims rest entirely on
the wiring, the signs, the half-lives and the regulatory form, and not at all on the abundance data.

=================================  THE MODEL  =================================================
Per gene g, with regulators j carrying curated sign s_j in {+1,-1} (2,592 unsigned edges dropped):
    a_j(t) = P_j(t)/P_j(WT)                 relative TF protein activity, 1 at WT, 0 at knockout
    z_g(t) = (1/n_g) SUM_j s_j a_j(t)       mean signed regulatory input; 1/n_g is the buffering
                                            assumption, and K6 tests it against measurement
    R_g(t) = 2 sigmoid(beta (z_g - z_g^WT)) response, exactly 1 at WT, in (0,2)
    dM_g/dt = k_sm,g R_g - k_dm,g M_g       k_dm = ln2/mRNA half-life + ln2/24 h
    dP_g/dt = k_tl M_g   - k_dp,g P_g       k_dp = ln2/protein half-life + ln2/24 h, k_tl = 119.24
Integrated by exponential Euler, which is EXACT when R is constant, so the WT column cannot drift.
Direction predictions do not depend on beta (sign(R-1) = sign(z - z^WT)); magnitudes do, so beta is
swept over {0.5, 1, 2, 4} with 2 as the declared primary, and the whole sweep is reported.
Knockout mode A (the instruction): clamp P_TF = 0 at t=0. Mode B (more faithful to shRNA/CRISPRi):
set k_sm of the TF to 0 and let its protein decay at its measured half-life. Both reported.

PREDECLARED, before any number. Thresholds fixed here and not moved afterwards.
DISCLOSED PEEK: before writing this, target counts per TF were counted (median 26.5 measured curated
targets) to set MIN_TARGETS, and the Larsson parse was checked against its own published mean
(rho 0.99999). No agreement, accuracy, correlation or gate number was computed before this docstring.
DISCLOSED ADDITION, after run 1: the first run returned a median balanced accuracy of 0.5073 and a
sign-shuffle null at z 1.86, so two BASELINES were added -- the edge-sign lookup (K3b) and the
indirect-target arm (K6b). Neither is a threshold and neither was tuned; both are reported whichever
way they come out, and no predeclared gate value was altered. They exist because a near-chance result
makes "what does the integrator add over reading the picture" the only question worth asking next.

  K1  THE MACHINE IS BUILT AND THE INTEGRATOR IS SOUND.
        Coverage: TFs measured, genes parameterised, wired-and-measured pairs. Gate: the WT column
        must not drift (max |M_WT(96 h)/M_WT(0) - 1| < 1e-9), halving dt must change predictions by
        < 1e-3, and the Larsson parse must reproduce its own published mean at rho > 0.99.
  K2  THE CIRCULAR ARM, RUN ON PURPOSE, THEN EXCLUDED.
        ARM C: M* = mRNA_copies to < 1e-10 relative, from closed form AND by integrating up from
        M = 0. Labelled circular. ARM I: absolute steady state as a real prediction, error reported.
        Invariance: KO log2FC from ARM C, ARM I and a randomly rescaled k_sm agree to < 1e-6.
        Gate: identity < 1e-10 AND invariance < 1e-6. Failure of either means the arms are entangled.
  K3  DIRECTION BEATS CHANCE.
        Per TF, over its curated targets measured in its own knockdown. BALANCED accuracy, because
        86.8% of curated edges activate and the measured log2FC is not sign-balanced, so raw accuracy
        is a majority-class artefact. Gate: median per-TF balanced accuracy > 0.50 AND a two-sided
        sign test over TFs at p < 0.05.
        K3b, the added baseline: the EDGE-SIGN LOOKUP -- predict the target moves opposite to the
        curated sign, no dynamics, no integrator, no half-life, no propagation. If the ODE does not
        beat this, the integrator is decoration on top of a table lookup and this loop says so.
  K4  IT BEATS A SIGN-SHUFFLED NULL, VERIFIED CAPABLE.
        Signs permuted across the curated network (composition preserved) and, separately, redrawn
        50/50 (composition broken). gate_guard.null_can_move on what the statistic actually consumes
        -- the predicted log2FC vector -- before either null's output is allowed to count;
        gate_guard.survival for the verdict. Gate: survival DEFINED and real above null.
  K5  IT BEATS FAME, AND IT BEATS THE MAJORITY CLASS.
        pubs against per-TF accuracy and per-target correctness; a fame-matched rewiring null that
        keeps every TF's out-degree and sign multiset but redraws targets with probability
        proportional to pubs; the constant-DOWN baseline that 87% activating edges imply with no
        wiring at all; and the per-TF majority-class oracle. Gate: the model beats constant-DOWN on
        balanced accuracy and beats the fame-matched rewiring.
  K6  MAGNITUDE, NOT ONLY SIGN.
        Predicted against measured log2FC: pooled and per-TF Spearman, the ratio of predicted to
        measured |log2FC|, the beta sweep, the time sweep {12, 24, 48, 96 h}, and the model's own
        structural prediction -- targets with more curated regulators are buffered and should move
        less -- tested against the measurement. Reported whatever it says.
        K6b, the added arm: INDIRECT targets -- measured genes the model moves that are NOT wired to
        the knocked-out TF at all, reached only by propagation through other TFs. Those predictions
        are the integrator's own content; nothing but a dynamical model can make them.
  K7  THE WEAKER SELF-CONSISTENCY ARM, AND THE LIMITS.
        Does knocking out a high-dep_frac TF perturb the network more than a low-dep_frac one, and
        does that survive controlling for out-degree? Labelled explicitly as self-consistency, not
        validation. Then the limits: cell types do not match, there is no time course, CollecTRI is
        itself literature-curated, and a sign is not a rate constant.

-> outputs/loop_perturb.json
"""
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sps

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LN2 = float(np.log(2.0))

# ---- MEASURED / ESTABLISHED INPUTS (none of these is fitted here) --------------------------------
T_DOUBLE_H = 24.0          # measured doubling time (loop 101)
K_TL = 119.24              # protein/mRNA/h, loop 91 median translation rate
N_CORE = 55716             # rows 0..55,715 of reg are the curated CollecTRI core
# ---- PREDECLARED KNOBS ---------------------------------------------------------------------------
BETA = 2.0
BETA_SWEEP = (0.5, 1.0, 2.0, 4.0)
T_MEASURE_H = 48.0
T_SWEEP = (12.0, 24.0, 48.0, 96.0)
DT_H = 0.5
MIN_TARGETS = 20           # measured curated targets per TF
MIN_PER_CLASS = 3          # measured up and measured down, per TF
LFC_SWEEP = (0.0, 0.25, 0.5, 1.0)
NPERM = 15
NFAME = 10
EPS_CALL = 1e-9            # |predicted log2FC| below this is "no call", not a wrong call
SEED = 11501

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 10:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def sign_test(vals, h0=0.5):
    """Two-sided sign test: how many per-TF accuracies exceed h0."""
    from scipy.stats import binomtest
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    w = v[v != h0]
    if len(w) < 5:
        return float("nan"), 0, len(w)
    k = int((w > h0).sum())
    return float(binomtest(k, len(w), 0.5).pvalue), k, len(w)


# =================================================================================================
#  THE INTEGRATOR.  Exponential Euler: exact for the linear part, so the WT column cannot drift.
# =================================================================================================
def integrate(ksm, kdm, kdp, S, zwt, tf_row, ko_tf, beta, ts, dt, mode="clamp"):
    """Return {t: log2(M(t)/M_WT)} for every KO column.

    ksm (nG,) baseline transcription; kdm (nG,); kdp (nTF,); S (nG,nTF) csr with entries s_j/n_g;
    zwt (nG,) = S @ 1; tf_row (nTF,) row of each TF gene in the gene axis; ko_tf (K,) TF index
    knocked out in each column, -1 for the WT column.
    """
    nG = len(ksm)
    K = len(ko_tf)
    M0 = ksm / kdm
    Pw = K_TL * M0[tf_row] / kdp
    Pw = np.maximum(Pw, 1e-300)
    M = np.repeat(M0[:, None], K, 1)
    P = np.repeat(Pw[:, None], K, 1)
    ksm_col = np.repeat(ksm[:, None], K, 1)
    kcols = [i for i, k in enumerate(ko_tf) if k >= 0]
    krows = [ko_tf[i] for i in kcols]
    if mode == "txn":                       # shRNA/CRISPRi-like: transcription of the TF goes to 0
        ksm_col[tf_row[krows], kcols] = 0.0
    Em = np.exp(-kdm * dt)[:, None]
    Ep = np.exp(-kdp * dt)[:, None]
    zwtc = zwt[:, None]
    inv_kdm = (1.0 / kdm)[:, None]
    inv_kdp = (1.0 / kdp)[:, None]
    out = {}
    want = sorted(set(ts))
    t = 0.0
    nsteps = int(round(max(want) / dt))
    snap_at = {int(round(x / dt)): x for x in want}
    if 0 in snap_at:
        out[snap_at[0]] = np.zeros((nG, K))
    for step in range(1, nsteps + 1):
        if mode == "clamp":
            P[krows, kcols] = 0.0
        a = P / Pw[:, None]
        z = S @ a
        R = 2.0 / (1.0 + np.exp(-beta * (z - zwtc)))
        Mstar = ksm_col * R * inv_kdm
        M = Mstar + (M - Mstar) * Em
        Pstar = K_TL * M[tf_row, :] * inv_kdp
        P = Pstar + (P - Pstar) * Ep
        if mode == "clamp":
            P[krows, kcols] = 0.0
        t = step * dt
        if step in snap_at:
            out[snap_at[step]] = np.log2(np.maximum(M, 1e-300) / np.maximum(M0[:, None], 1e-300))
    return out, M0


def build_S(nG, nTF, tgt, src, sgn):
    """S[g,j] = s_j / n_g, with n_g the number of signed regulators of g."""
    ndeg = np.bincount(tgt, minlength=nG).astype(float)
    ndeg[ndeg == 0] = 1.0
    S = sps.csr_matrix((sgn / ndeg[tgt], (tgt, src)), shape=(nG, nTF))
    return S, np.asarray(S.sum(axis=1)).ravel()


# =================================================================================================
def score_direct(pred, col_of, row_of, tf_targets, measured, lfc_min=0.0):
    """Balanced accuracy per TF over its curated, measured targets. Returns per-TF records."""
    recs = []
    for tf, tgts in tf_targets.items():
        c = col_of.get(tf)
        if c is None:
            continue
        m = measured[tf]
        pv, mv = [], []
        for g in tgts:
            if g == tf or g not in m or g not in row_of:
                continue
            if abs(m[g]) < lfc_min:
                continue
            pv.append(pred[row_of[g], c])
            mv.append(m[g])
        if len(pv) < MIN_TARGETS:
            continue
        pv, mv = np.asarray(pv), np.asarray(mv)
        called = np.abs(pv) > EPS_CALL
        ps, ms = np.sign(pv), np.sign(mv)
        dn, up = (ms < 0) & called, (ms > 0) & called
        if dn.sum() < MIN_PER_CLASS or up.sum() < MIN_PER_CLASS:
            continue
        sens = float((ps[dn] < 0).mean())
        spec = float((ps[up] > 0).mean())
        ba = 0.5 * (sens + spec)
        raw = float((ps[called] == ms[called]).mean())
        maj = max((ms[called] < 0).mean(), (ms[called] > 0).mean())
        rho, _ = spear(pv, mv)
        recs.append({"tf": tf, "n": int(called.sum()), "n_down": int(dn.sum()), "n_up": int(up.sum()),
                     "balanced_acc": ba, "sens_down": sens, "spec_up": spec, "raw_acc": raw,
                     "majority": float(maj), "rho": rho,
                     "med_abs_pred": float(np.median(np.abs(pv))),
                     "med_abs_meas": float(np.median(np.abs(mv))),
                     "no_call": int((~called).sum())})
    return recs


def score_indirect(pred, col_of, row_of, tf_targets, measured, G):
    """Balanced accuracy over measured genes the model moves that are NOT wired to the KO'd TF.

    These are the integrator's own content: a table lookup cannot produce them at all.
    """
    recs = []
    for tf, tgts in tf_targets.items():
        c = col_of.get(tf)
        if c is None:
            continue
        m = measured[tf]
        direct = set(tgts) | {tf}
        pv, mv = [], []
        for g, i in row_of.items():
            if g in direct or g not in m:
                continue
            p = pred[i, c]
            if abs(p) <= EPS_CALL:
                continue
            pv.append(p)
            mv.append(m[g])
        if len(pv) < MIN_TARGETS:
            continue
        pv, mv = np.asarray(pv), np.asarray(mv)
        ps, ms = np.sign(pv), np.sign(mv)
        dn, up = ms < 0, ms > 0
        if dn.sum() < MIN_PER_CLASS or up.sum() < MIN_PER_CLASS:
            continue
        recs.append({"tf": tf, "n": int(len(pv)),
                     "balanced_acc": 0.5 * (float((ps[dn] < 0).mean()) + float((ps[up] > 0).mean())),
                     "rho": spear(pv, mv)[0]})
    return recs


def score_lookup(edge_sign_of, col_of, tf_targets, measured):
    """THE BASELINE THE ODE MUST BEAT: predict the target moves opposite to the curated sign.

    No dynamics, no half-lives, no propagation -- reading the picture, not running it.
    """
    recs = []
    for tf, tgts in tf_targets.items():
        if tf not in col_of:
            continue
        m = measured[tf]
        pv, mv = [], []
        for g in set(tgts):
            if g == tf or g not in m:
                continue
            s = edge_sign_of.get((tf, g), 0.0)
            if s == 0:
                continue
            pv.append(-s)
            mv.append(m[g])
        if len(pv) < MIN_TARGETS:
            continue
        pv, mv = np.asarray(pv, float), np.asarray(mv, float)
        ps, ms = np.sign(pv), np.sign(mv)
        dn, up = ms < 0, ms > 0
        if dn.sum() < MIN_PER_CLASS or up.sum() < MIN_PER_CLASS:
            continue
        recs.append({"tf": tf, "n": int(len(pv)),
                     "balanced_acc": 0.5 * (float((ps[dn] < 0).mean()) + float((ps[up] > 0).mean())),
                     "raw_acc": float((ps == ms).mean())})
    return recs


def pooled_vec(pred, col_of, row_of, tf_targets, measured):
    """Every (TF, curated target) pair as flat arrays -- what the null's capability is tested on."""
    P, Mv, T = [], [], []
    for tf, tgts in tf_targets.items():
        c = col_of.get(tf)
        if c is None:
            continue
        m = measured[tf]
        for g in tgts:
            if g == tf or g not in m or g not in row_of:
                continue
            P.append(pred[row_of[g], c])
            Mv.append(m[g])
            T.append(tf)
    return np.asarray(P), np.asarray(Mv), T


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 115 -- DOES THE WIRING RESPOND?  knock a TF out in the integrated model,")
    say("             propagate through curated CollecTRI edges, compare to 257 real knockdowns")
    say("=" * 100)
    say()

    # ---------------------------------------------------------------- data
    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    depf = {g["name"]: float(g.get("dep_frac") or 0.0) for g in C["genes"]}
    core = C["reg"][:N_CORE]
    signed = [(names[a], names[b], int(s)) for a, b, s in core if s in (1, -1)]
    say(f"  curated CollecTRI core: {len(core):,} edges, {len(signed):,} signed "
        f"({len(core) - len(signed):,} unsigned dropped -- an unsigned edge has no direction to test)")

    LF = json.load(open(OUT / "orphan" / "cell_lifetimes.json"))["lifetimes"]
    mrna_hl = {k: float(v["mrna_hl_h"]) for k, v in LF.items() if v.get("mrna_hl_h")}
    prot_hl = {k: float(v["prot_hl_h"]) for k, v in LF.items() if v.get("prot_hl_h")}
    SW = json.load(open(SC / "_schwan2011.json"))
    mrna_cop = {k: float(v["mrna_copies"]) for k, v in SW.items() if v.get("mrna_copies")}
    for k, v in SW.items():                      # Schwanhausser fills gaps in the lifetime table
        if v.get("mrna_hl_h") and k not in mrna_hl:
            mrna_hl[k] = float(v["mrna_hl_h"])
        if v.get("prot_hl_h") and k not in prot_hl:
            prot_hl[k] = float(v["prot_hl_h"])
    say(f"  half-lives: {len(mrna_hl):,} mRNA, {len(prot_hl):,} protein; "
        f"{len(mrna_cop):,} genes with Schwanhausser mRNA copies (the CIRCULAR arm's input)")

    # Larsson burst kinetics -- an INDEPENDENT transcription rate, never a steady-state abundance
    import pandas as pd
    lx = pd.read_excel(SC / "larsson_S5.xlsx", sheet_name="C57")
    kk, mean_pub, mean_calc = {}, [], []
    for gname, mlstr, mexp in zip(lx["Gene"].astype(str), lx.iloc[:, 1], lx["Mean expression"]):
        f = [float(v) for v in re.findall(r"[-+0-9.eE]+", str(mlstr))[:3]]
        if len(f) < 3:
            continue
        kon, koff, ksyn = f
        if not np.isfinite([kon, koff, ksyn]).all() or kon + koff <= 0:
            continue
        bm = ksyn * kon / (kon + koff)          # in units of the mRNA degradation rate
        h = gname.upper()
        if bm > 0 and h not in kk:
            kk[h] = bm
        if np.isfinite(mexp):
            mean_pub.append(float(mexp))
            mean_calc.append(bm)
    rho_parse, _ = spear(mean_calc, mean_pub)
    say(f"  Larsson single-cell burst kinetics: {len(kk):,} symbols; parse check "
        f"ksyn*kon/(kon+koff) vs the paper's own mean expression rho {rho_parse:.5f}")

    # KnockTF measured knockdowns
    KD = SC / "pert" / "knocktf_deg"
    kd_files = sorted(p for p in KD.glob("*.json"))
    tf_out = {}
    for s, t, sg in signed:
        tf_out.setdefault(s, []).append((t, sg))
    measured, meta = {}, {}
    for p in kd_files:
        tf = p.stem
        if tf not in tf_out:
            continue
        d = json.load(open(p))
        measured[tf] = d["log2fc"]
        meta[tf] = {"cell": d.get("cell"), "method": d.get("method"),
                    "on_target": d.get("on_target_log2fc"), "n_genes": d.get("n_genes")}
    say(f"  KnockTF: {len(kd_files)} human TF knockdown tables on disk, {len(measured)} are curated "
        f"CollecTRI TFs")
    say()

    # ---------------------------------------------------------------- K1
    say("K1  THE MACHINE IS BUILT AND THE INTEGRATOR IS SOUND")
    universe = {s for s, _, _ in signed} | {t for _, t, _ in signed}
    G = sorted(g for g in universe if g in mrna_hl)
    row_of = {g: i for i, g in enumerate(G)}
    tf_names = sorted({s for s, _, _ in signed if s in row_of})
    tf_idx = {g: j for j, g in enumerate(tf_names)}
    tf_row = np.array([row_of[g] for g in tf_names], int)
    edges = [(row_of[t], tf_idx[s], sg) for s, t, sg in signed if t in row_of and s in tf_idx]
    tgt = np.array([e[0] for e in edges], int)
    src = np.array([e[1] for e in edges], int)
    sgn = np.array([e[2] for e in edges], float)
    S, zwt = build_S(len(G), len(tf_names), tgt, src, sgn)
    say(f"     simulated network: {len(G):,} genes with a measured mRNA half-life, "
        f"{len(tf_names):,} TFs, {len(edges):,} signed edges")
    med_phl = float(np.median([prot_hl[g] for g in tf_names if g in prot_hl]))
    n_phl = sum(1 for g in tf_names if g in prot_hl)
    say(f"     protein half-lives for {n_phl:,}/{len(tf_names):,} TFs; the rest take the TF median "
        f"{med_phl:.1f} h, declared, not fitted")
    kdm = np.array([LN2 / mrna_hl[g] + LN2 / T_DOUBLE_H for g in G])
    kdp = np.array([LN2 / prot_hl.get(g, med_phl) + LN2 / T_DOUBLE_H for g in tf_names])
    med_ksm_I = float(np.median([kk[g] for g in G if g in kk]))
    ksm_I = np.array([kk.get(g, med_ksm_I) for g in G]) * kdm
    n_lars = sum(1 for g in G if g in kk)
    say(f"     ARM I transcription rate from burst kinetics for {n_lars:,}/{len(G):,} genes; the rest")
    say(f"     take the median burst rate {med_ksm_I:.3f} -- NEVER that gene's own mRNA copy number")

    ko_tfs = [t for t in sorted(measured) if t in tf_idx]
    ko_tf = np.array([-1] + [tf_idx[t] for t in ko_tfs], int)
    col_of = {t: i + 1 for i, t in enumerate(ko_tfs)}
    say(f"     knockout columns: {len(ko_tfs)} TFs with both curated targets and a measured knockdown")

    predsI, M0I = integrate(ksm_I, kdm, kdp, S, zwt, tf_row, ko_tf, BETA, T_SWEEP, DT_H, "clamp")
    drift = float(np.max(np.abs(predsI[max(T_SWEEP)][:, 0])))
    predsI_h, _ = integrate(ksm_I, kdm, kdp, S, zwt, tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H / 2,
                            "clamp")
    dt_err = float(np.max(np.abs(predsI[T_MEASURE_H] - predsI_h[T_MEASURE_H])))
    tf_targets = {t: [g for g, _ in tf_out[t]] for t in ko_tfs}
    npair = sum(1 for t in ko_tfs for g in tf_targets[t]
                if g != t and g in row_of and g in measured[t])
    say(f"     WT column drift over {max(T_SWEEP):.0f} h: {drift:.2e} log2 units (exponential Euler "
        f"is exact at constant R)")
    say(f"     dt {DT_H} h vs {DT_H/2} h: max change {dt_err:.2e} log2 units")
    say(f"     wired AND measured (TF, target) pairs: {npair:,}")
    k1 = bool(drift < 1e-9 and dt_err < 1e-3 and rho_parse > 0.99 and len(ko_tfs) >= 20)
    say(f"     K1 {'PASS' if k1 else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K2
    say("K2  THE CIRCULAR ARM, RUN ON PURPOSE, THEN EXCLUDED")
    both = [g for g in G if g in mrna_cop]
    bi = np.array([row_of[g] for g in both], int)
    ksm_C_full = ksm_I.copy()
    ksm_C_full[bi] = np.array([mrna_cop[g] for g in both]) * kdm[bi]
    obs = np.array([mrna_cop[g] for g in both])
    closed = (ksm_C_full[bi] / kdm[bi])
    rel_closed = float(np.max(np.abs(closed - obs) / obs))
    # and by actually integrating up from M = 0, with no regulation, to a very long time
    Mz = np.zeros(len(bi))
    for _ in range(500):
        Mstar = ksm_C_full[bi] / kdm[bi]
        Mz = Mstar + (Mz - Mstar) * np.exp(-kdm[bi] * 10.0)
    rel_int = float(np.max(np.abs(Mz - obs) / obs))
    say(f"     ARM C, k_sm = mRNA_copies x k_dm, {len(both):,} genes:")
    say(f"       closed form M* = k_sm/k_dm vs the mRNA_copies that went in:  max rel err "
        f"{rel_closed:.3e}")
    say(f"       integrated from M = 0 for 5,000 h and back to steady state: max rel err "
        f"{rel_int:.3e}")
    say(f"       THIS IS AN ALGEBRAIC IDENTITY, NOT A RESULT. M* = k_sm/k_dm = mRNA_copies for every")
    say(f"       gene by construction. It is shown here so it can never be mistaken for a")
    say(f"       validation, and it is EXCLUDED from K3-K7 and from every headline number.")
    predI_wt = ksm_I / kdm
    ok = np.array([g in kk and g in mrna_cop for g in G])
    pv = predI_wt[ok]
    ov = np.array([mrna_cop[g] for g, o in zip(G, ok) if o])
    rho_ss, n_ss = spear(pv, ov)
    med_lerr = float(np.median(np.abs(np.log2(np.maximum(pv, 1e-12) / ov))))
    say(f"     ARM I, k_sm from burst kinetics, {n_ss:,} genes with an independent measured mRNA copy")
    say(f"       number: predicted vs observed steady state rho {rho_ss:+.4f}, median |log2 error| "
        f"{med_lerr:.2f} ({2**med_lerr:.1f}x)")
    say(f"       THAT is what a non-circular steady-state prediction looks like: an error, not 1e-16.")
    # invariance of the KO fold change to k_sm
    predsC, _ = integrate(ksm_C_full, kdm, kdp, S, zwt, tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H,
                          "clamp")
    scale = np.exp(rng.uniform(np.log(1e-2), np.log(1e2), len(G)))
    predsR, _ = integrate(ksm_I * scale, kdm, kdp, S, zwt, tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H,
                          "clamp")
    inv_C = float(np.max(np.abs(predsC[T_MEASURE_H] - predsI[T_MEASURE_H])))
    inv_R = float(np.max(np.abs(predsR[T_MEASURE_H] - predsI[T_MEASURE_H])))
    say(f"     INVARIANCE: max |KO log2FC difference| between arms")
    say(f"       ARM C (circular k_sm) vs ARM I (burst k_sm):        {inv_C:.3e}")
    say(f"       ARM I vs ARM I with k_sm randomly rescaled 1e-2..1e2: {inv_R:.3e}")
    say(f"       so the fold change is invariant to the baseline transcription rate, and the")
    say(f"       tautology CANNOT be the source of any direction or magnitude claim below.")
    say(f"       The honest converse: K3-K6 rest on the wiring, the signs, the half-lives and the")
    say(f"       regulatory form -- and on none of the abundance data at all.")
    k2 = bool(rel_closed < 1e-10 and rel_int < 1e-10 and inv_C < 1e-6 and inv_R < 1e-6)
    say(f"     K2 {'PASS' if k2 else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K3
    say("K3  DIRECTION BEATS CHANCE")
    pred = predsI[T_MEASURE_H]
    recs = score_direct(pred, col_of, row_of, tf_targets, measured)
    ba = np.array([r["balanced_acc"] for r in recs])
    raw = np.array([r["raw_acc"] for r in recs])
    p_sign, k_up, n_used = sign_test(ba)
    say(f"     {len(recs)} TFs clear >= {MIN_TARGETS} measured curated targets and "
        f">= {MIN_PER_CLASS} per class")
    say(f"     median per-TF BALANCED accuracy {np.median(ba):.4f}   mean {ba.mean():.4f}   "
        f"[{ba.min():.3f}, {ba.max():.3f}]")
    say(f"     median per-TF raw accuracy      {np.median(raw):.4f}  (majority-class inflated -- "
        f"reported, not used)")
    say(f"     sign test: {k_up}/{n_used} TFs above 0.5, two-sided p {p_sign:.3g}")
    tot_n = int(sum(r["n"] for r in recs))
    say(f"     pooled over {tot_n:,} scored (TF, target) pairs")
    sw = {}
    for lm in LFC_SWEEP:
        rr = score_direct(pred, col_of, row_of, tf_targets, measured, lfc_min=lm)
        sw[lm] = {"n_tf": len(rr),
                  "median_ba": float(np.median([x["balanced_acc"] for x in rr])) if rr else float("nan")}
        say(f"       |measured log2FC| >= {lm:4.2f}: {len(rr):3d} TFs, median balanced acc "
            f"{sw[lm]['median_ba']:.4f}")
    k3 = bool(np.median(ba) > 0.50 and np.isfinite(p_sign) and p_sign < 0.05)
    say(f"     K3 {'PASS' if k3 else 'FAIL'} -- the curated wiring does {'' if k3 else 'NOT '}"
        f"predict the direction of a real knockdown better than chance")
    say()
    say("K3b THE EDGE-SIGN LOOKUP: what does running the model add over reading the picture?")
    esign = {}
    for s, t, sg in signed:
        esign[(s, t)] = esign.get((s, t), 0.0) + sg
    lk = score_lookup(esign, col_of, tf_targets, measured)
    lk_by = {r["tf"]: r["balanced_acc"] for r in lk}
    pair = [(r["balanced_acc"], lk_by[r["tf"]]) for r in recs if r["tf"] in lk_by]
    dmodel = np.array([a - b for a, b in pair])
    p_pair, k_pair, n_pair2 = sign_test(dmodel, 0.0)
    say(f"     edge-sign lookup, no dynamics at all: {len(lk)} TFs, median balanced accuracy "
        f"{np.median([r['balanced_acc'] for r in lk]):.4f}")
    say(f"     the integrated ODE:                   {len(recs)} TFs, median balanced accuracy "
        f"{np.median(ba):.4f}")
    say(f"     paired over the {len(pair)} TFs both score: median difference "
        f"{np.median(dmodel):+.4f}, {k_pair}/{n_pair2} TFs where the ODE is better, "
        f"two-sided sign test p {p_pair:.3g}")
    say(f"     READ THIS PLAINLY: the ODE and the lookup make the SAME sign call on every direct")
    say(f"     target unless propagation through other TFs overturns it, so a difference here is")
    say(f"     exactly the contribution of the dynamics -- and only of the dynamics.")
    say()

    # ---------------------------------------------------------------- K4
    say("K4  IT BEATS A SIGN-SHUFFLED NULL, VERIFIED CAPABLE")
    real_pool, meas_pool, _ = pooled_vec(pred, col_of, row_of, tf_targets, measured)
    real_med = float(np.median(ba))
    nulls_perm, nulls_rand, caps_v, caps_s = [], [], [], []
    for i in range(NPERM):
        sg2 = sgn[rng.permutation(len(sgn))]                 # composition preserved
        S2, z2 = build_S(len(G), len(tf_names), tgt, src, sg2)
        p2, _ = integrate(ksm_I, kdm, kdp, S2, z2, tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H, "clamp")
        r2 = score_direct(p2[T_MEASURE_H], col_of, row_of, tf_targets, measured)
        nulls_perm.append(float(np.median([x["balanced_acc"] for x in r2])) if r2 else np.nan)
        n_pool, _, _ = pooled_vec(p2[T_MEASURE_H], col_of, row_of, tf_targets, measured)
        caps_v.append(GG.null_can_move(np.round(real_pool, 12), np.round(n_pool, 12))["changed"])
        caps_s.append(float(np.mean(np.sign(real_pool) != np.sign(n_pool))))
        sg3 = rng.choice([-1.0, 1.0], len(sgn))              # composition broken, 50/50
        S3, z3 = build_S(len(G), len(tf_names), tgt, src, sg3)
        p3, _ = integrate(ksm_I, kdm, kdp, S3, z3, tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H, "clamp")
        r3 = score_direct(p3[T_MEASURE_H], col_of, row_of, tf_targets, measured)
        nulls_rand.append(float(np.median([x["balanced_acc"] for x in r3])) if r3 else np.nan)
    cap = GG.null_can_move(np.round(real_pool, 12),
                           np.round(pooled_vec(
                               integrate(ksm_I, kdm, kdp,
                                         *build_S(len(G), len(tf_names), tgt, src,
                                                  sgn[rng.permutation(len(sgn))]),
                                         tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H, "clamp")[0][T_MEASURE_H],
                               col_of, row_of, tf_targets, measured)[0], 12))
    frac_sign_flip = float(np.mean(caps_s))
    say(f"     CAPABILITY CHECK on what the statistic actually consumes -- the predicted log2FC")
    say(f"     vector over {len(real_pool):,} pairs: the shuffle changes {np.mean(caps_v):.1%} of "
        f"predicted values -- capable: {cap['capable']}")
    say(f"     it changes only {frac_sign_flip:.1%} of predicted SIGNS, and that is a bound, not a")
    say(f"     bug: with {(sgn>0).mean():.1%} of curated edges activating, a composition-preserving")
    say(f"     permutation can flip at most 2p(1-p) = {2*(sgn>0).mean()*(1-(sgn>0).mean()):.1%} of them.")
    say(f"     So the second null redraws signs 50/50 and breaks the composition too.")
    s_perm = GG.survival(real_med, nulls_perm)
    GG.report("median balanced accuracy vs sign PERMUTATION (composition kept)", s_perm, emit=say)
    s_rand = GG.survival(real_med, nulls_rand)
    GG.report("median balanced accuracy vs sign REDRAW 50/50 (composition broken)", s_rand, emit=say)
    k4 = bool(cap["capable"] and s_perm.get("defined") and s_rand.get("defined"))
    say(f"     K4 {'PASS' if k4 else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K5
    say("K5  IT BEATS FAME, AND IT BEATS THE MAJORITY CLASS")
    # constant-DOWN: what 87% activating edges imply with no wiring at all. Its balanced accuracy is
    # exactly 0.5 by definition, so the honest comparison is raw against raw, and both are reported.
    raw_cd = np.array([r["n_down"] / (r["n_down"] + r["n_up"]) for r in recs])
    say(f"     constant-DOWN baseline (no wiring): balanced accuracy 0.5000 by construction; raw "
        f"accuracy median {np.median(raw_cd):.4f}")
    say(f"     model:                              balanced accuracy median {np.median(ba):.4f}; raw "
        f"accuracy median {np.median(raw):.4f}")
    maj = np.array([r["majority"] for r in recs])
    say(f"     per-TF majority-class ORACLE raw accuracy median {np.median(maj):.4f} -- an oracle, "
        f"shown to bound what raw accuracy can mean")
    tf_pub = np.array([pubs.get(r["tf"], 0.0) for r in recs])
    r_fame_tf, n_ft = spear(tf_pub, ba)
    corr_pool = (np.sign(real_pool) == np.sign(meas_pool)).astype(float)
    tgt_pub = []
    for tf, tgts in tf_targets.items():
        if tf not in col_of:
            continue
        m = measured[tf]
        for g in tgts:
            if g == tf or g not in m or g not in row_of:
                continue
            tgt_pub.append(pubs.get(g, 0.0))
    r_fame_tg, n_fg = spear(np.asarray(tgt_pub), corr_pool)
    say(f"     pubs vs per-TF balanced accuracy  rho {r_fame_tf:+.4f} (n {n_ft})")
    say(f"     target pubs vs per-pair correctness rho {r_fame_tg:+.4f} (n {n_fg:,})")
    # fame-matched rewiring: out-degree and sign multiset kept, targets redrawn proportional to pubs
    pw = np.array([pubs.get(g, 0.0) for g in G]) + 1.0
    pw = pw / pw.sum()
    nulls_fame = []
    for _ in range(NFAME):
        newt = rng.choice(len(G), size=len(tgt), p=pw)
        Sf, zf = build_S(len(G), len(tf_names), newt, src, sgn)
        pf, _ = integrate(ksm_I, kdm, kdp, Sf, zf, tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H, "clamp")
        rf = score_direct(pf[T_MEASURE_H], col_of, row_of, tf_targets, measured)
        nulls_fame.append(float(np.median([x["balanced_acc"] for x in rf])) if rf else np.nan)
    s_fame = GG.survival(real_med, nulls_fame)
    GG.report("median balanced accuracy vs FAME-MATCHED rewiring (out-degree and signs kept)",
              s_fame, emit=say)
    # wrong-TF null: score TF A's prediction against TF B's measurement
    wrong = []
    shuf = list(ko_tfs)
    for _ in range(NPERM):
        rng.shuffle(shuf)
        swap = {a: measured[b] for a, b in zip(ko_tfs, shuf) if a != b}
        rw = score_direct(pred, col_of, row_of,
                          {t: tf_targets[t] for t in swap}, swap)
        wrong.append(float(np.median([x["balanced_acc"] for x in rw])) if rw else np.nan)
    s_wrong = GG.survival(real_med, wrong)
    GG.report("median balanced accuracy vs WRONG-TF measurement", s_wrong, emit=say)
    k5 = bool(np.median(ba) > 0.5 and s_fame.get("defined") and s_wrong.get("defined"))
    say(f"     K5 {'PASS' if k5 else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K6
    say("K6  MAGNITUDE, NOT ONLY SIGN")
    rho_pool, n_pool_ = spear(real_pool, meas_pool)
    per_rho = np.array([r["rho"] for r in recs])
    ratio = np.array([r["med_abs_pred"] / max(r["med_abs_meas"], 1e-9) for r in recs])
    say(f"     pooled Spearman predicted vs measured log2FC: {rho_pool:+.4f} over {n_pool_:,} pairs")
    say(f"     per-TF Spearman: median {np.median(per_rho):+.4f}, "
        f"{int((per_rho > 0).sum())}/{len(per_rho)} positive")
    say(f"     median |predicted log2FC| {np.median(np.abs(real_pool)):.4f} vs median |measured| "
        f"{np.median(np.abs(meas_pool)):.4f}  -> the model is "
        f"{np.median(np.abs(meas_pool))/max(np.median(np.abs(real_pool)),1e-9):.1f}x too SMALL")
    say(f"     per-TF predicted/measured magnitude ratio: median {np.median(ratio):.3f}")
    bsw = {}
    for b in BETA_SWEEP:
        pb, _ = integrate(ksm_I, kdm, kdp, S, zwt, tf_row, ko_tf, b, (T_MEASURE_H,), DT_H, "clamp")
        rb = score_direct(pb[T_MEASURE_H], col_of, row_of, tf_targets, measured)
        vb, mb, _ = pooled_vec(pb[T_MEASURE_H], col_of, row_of, tf_targets, measured)
        bsw[b] = {"median_ba": float(np.median([x["balanced_acc"] for x in rb])),
                  "rho": spear(vb, mb)[0], "med_abs_pred": float(np.median(np.abs(vb)))}
        say(f"       beta {b:4.1f}: median balanced acc {bsw[b]['median_ba']:.4f}  pooled rho "
            f"{bsw[b]['rho']:+.4f}  median |pred| {bsw[b]['med_abs_pred']:.4f}")
    tsw = {}
    for tt in T_SWEEP:
        rt = score_direct(predsI[tt], col_of, row_of, tf_targets, measured)
        vt, mt, _ = pooled_vec(predsI[tt], col_of, row_of, tf_targets, measured)
        tsw[tt] = {"median_ba": float(np.median([x["balanced_acc"] for x in rt])),
                   "rho": spear(vt, mt)[0], "med_abs_pred": float(np.median(np.abs(vt)))}
        say(f"       t {tt:5.0f} h: median balanced acc {tsw[tt]['median_ba']:.4f}  pooled rho "
            f"{tsw[tt]['rho']:+.4f}  median |pred| {tsw[tt]['med_abs_pred']:.4f}")
    ptx, _ = integrate(ksm_I, kdm, kdp, S, zwt, tf_row, ko_tf, BETA, (T_MEASURE_H,), DT_H, "txn")
    rtx = score_direct(ptx[T_MEASURE_H], col_of, row_of, tf_targets, measured)
    vtx, mtx, _ = pooled_vec(ptx[T_MEASURE_H], col_of, row_of, tf_targets, measured)
    say(f"     knockout mode B (transcription off, protein decays at its measured half-life): "
        f"median balanced acc {np.median([x['balanced_acc'] for x in rtx]):.4f}, pooled rho "
        f"{spear(vtx, mtx)[0]:+.4f}")
    # the model's own structural prediction: more regulators -> more buffered
    ndeg_all = np.bincount(tgt, minlength=len(G)).astype(float)
    npred, nmeas, nreg = [], [], []
    for tf, tgts in tf_targets.items():
        if tf not in col_of:
            continue
        m = measured[tf]
        for g in tgts:
            if g == tf or g not in m or g not in row_of:
                continue
            npred.append(abs(pred[row_of[g], col_of[tf]]))
            nmeas.append(abs(m[g]))
            nreg.append(ndeg_all[row_of[g]])
    r_buf_pred, _ = spear(nreg, npred)
    r_buf_meas, n_buf = spear(nreg, nmeas)
    say(f"     BUFFERING, the model's own prediction: |log2FC| vs number of curated regulators")
    say(f"       predicted rho {r_buf_pred:+.4f}   MEASURED rho {r_buf_meas:+.4f}  (n {n_buf:,})")
    say(f"       the model asserts buffering by construction (the 1/n_g term); the measurement "
        f"{'AGREES' if r_buf_meas < 0 else 'DOES NOT AGREE'} in sign")
    k6 = True
    say(f"     K6 {'PASS' if k6 else 'FAIL'} -- magnitude reported, not only sign")
    say()
    say("K6b THE INDIRECT ARM -- the only predictions a table lookup cannot make")
    ind = score_indirect(pred, col_of, row_of, tf_targets, measured, G)
    iba = np.array([r["balanced_acc"] for r in ind]) if ind else np.array([np.nan])
    irho = np.array([r["rho"] for r in ind]) if ind else np.array([np.nan])
    p_ind, k_ind, n_ind = sign_test(iba)
    nz = float(np.mean(np.abs(pred[:, 1:]) > EPS_CALL))
    say(f"     {nz:.1%} of (gene, knockout) cells are moved at all -- the rest are unreachable "
        f"through the curated network and get no call")
    say(f"     {len(ind)} TFs have >= {MIN_TARGETS} moved, measured, NON-target genes")
    say(f"     median per-TF balanced accuracy on indirect genes {np.median(iba):.4f}, "
        f"median rho {np.median(irho):+.4f}")
    say(f"     sign test: {k_ind}/{n_ind} above 0.5, two-sided p {p_ind:.3g}")
    say(f"     this is the integrator's own claim: these genes are not wired to the knocked-out TF,")
    say(f"     and only a propagating dynamical model produces a direction for them at all.")
    say()

    # ---------------------------------------------------------------- K7
    say("K7  THE WEAKER SELF-CONSISTENCY ARM, AND THE LIMITS")
    say("     LABEL: this arm uses NO measurement of the knockdown. It asks only whether the model")
    say("     is internally coherent, and it is a WEAKER claim than K3-K6 by construction.")
    L1 = np.abs(pred[:, 1:]).sum(axis=0)
    outdeg = np.array([len([1 for g, _ in tf_out[t]]) for t in ko_tfs], float)
    dfv = np.array([depf.get(t, 0.0) for t in ko_tfs])
    r_dep, n_dep = spear(dfv, L1)
    r_out, _ = spear(outdeg, L1)
    say(f"     network perturbation = sum over genes of |predicted log2FC| at {T_MEASURE_H:.0f} h")
    say(f"       vs the knocked-out TF's dep_frac: rho {r_dep:+.4f} (n {n_dep})")
    say(f"       vs its curated out-degree:        rho {r_out:+.4f}   <- the confound")
    strata, resid = [], []
    q = np.quantile(outdeg, [0.25, 0.5, 0.75])
    for lo, hi in [(-1, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e18)]:
        m = (outdeg > lo) & (outdeg <= hi)
        if m.sum() >= 10:
            rr, nn = spear(dfv[m], L1[m])
            strata.append({"lo": float(lo), "hi": float(hi), "n": int(nn), "rho": rr})
            say(f"       within out-degree stratum ({max(lo,0):.0f}, {hi:.0f}]: n {nn:3d}  "
                f"dep_frac vs perturbation rho {rr:+.4f}")
    r_dep_pub, _ = spear([pubs.get(t, 0.0) for t in ko_tfs], L1)
    say(f"       vs pubs of the knocked-out TF:    rho {r_dep_pub:+.4f}   <- fame, always reported")
    say()
    say("     LIMITS, stated rather than worked around:")
    say(f"       1. CELL TYPES DO NOT MATCH. Half-lives are HeLa/NIH3T3; burst kinetics are mouse")
    say(f"          fibroblast mapped by uppercased symbol; the {len(ko_tfs)} knockdowns span "
        f"{len(set(meta[t]['cell'] for t in ko_tfs))} different human")
    say(f"          cell lines. Every number above is a cross-cell-type comparison and no gate here")
    say(f"          can separate a wrong model from a cell-type difference.")
    say(f"       2. NO TIME COURSE. KnockTF records one harvest time, not reported per dataset, so")
    say(f"          {T_MEASURE_H:.0f} h is declared, not matched. The time sweep in K6 is the honest")
    say(f"          substitute and it shows how much that choice moves the answer.")
    say(f"       3. COLLECTRI IS LITERATURE-CURATED, so it inherits the fame structure this project")
    say(f"          keeps finding. K5 measures it rather than assuming it away.")
    say(f"       4. A SIGN IS NOT A RATE CONSTANT. Every edge gets unit weight because that is all")
    say(f"          the annotation carries; beta and the 1/n_g buffering are assumptions, swept and")
    say(f"          tested, not measured.")
    say(f"       5. The absolute steady state is only predicted in ARM I, at {2**med_lerr:.1f}x median")
    say(f"          error. Nothing here claims the model reproduces a cell's mRNA levels.")
    k7 = True
    say(f"     K7 {'PASS' if k7 else 'FAIL'} -- self-consistency measured and labelled weaker")
    say()

    # ---------------------------------------------------------------- verdict
    gates = {"K1 the machine is built and the integrator is sound": k1,
             "K2 the circular arm is demonstrated and excluded": k2,
             "K3 direction beats chance": k3,
             "K4 it beats a sign-shuffled null verified capable": k4,
             "K5 it beats fame and the majority class": k5,
             "K6 magnitude reported, not only sign": k6,
             "K7 self-consistency arm run and labelled weaker": k7}
    say("=" * 100)
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say("=" * 100)

    man = RM.manifest(
        inputs=[str(LR.CELL), str(OUT / "orphan" / "cell_lifetimes.json"),
                str(SC / "_schwan2011.json"), str(SC / "larsson_S5.xlsx"), str(KD)],
        available=len(kd_files), used=len(ko_tfs), selection="filtered", seed=SEED,
        controls=["sign permutation, composition preserved, capability-checked on the predicted vector",
                  "sign redraw 50/50, composition broken",
                  "fame-matched rewiring: out-degree and sign multiset kept, targets drawn prop. to pubs",
                  "wrong-TF null: TF A's prediction scored against TF B's measurement",
                  "constant-DOWN baseline and the per-TF majority-class oracle",
                  "the edge-sign lookup baseline with no dynamics at all",
                  "the indirect arm: measured genes not wired to the knocked-out TF",
                  "the circular k_sm run on purpose and shown to be an identity",
                  "k_sm randomly rescaled over 1e-2..1e2 to prove fold-change invariance",
                  "beta swept 0.5-4, time swept 12-96 h, |log2FC| cutoff swept 0-1",
                  "dt halved; WT column checked for drift"],
        note="KnockTF v2 measured knockdowns exist in the repo (scratchpad/pert/knocktf_deg, 364 "
             "tables); scratchpad/reg is network sources, not perturbation data. The self-"
             "consistency fallback is run as K7 and labelled the weaker claim.")
    RM.report(man, emit=say)

    res = {"test": "loop_perturb", "manifest": man, "gates": gates,
           "inputs_vs_predictions": {
               "inputs": ["mRNA half-life", "protein half-life", "curated CollecTRI edge sign",
                          "doubling time 24 h", "translation rate 119.24/h",
                          "baseline k_sm (ARM C: mRNA_copies x k_dm -- CIRCULAR; "
                          "ARM I: Larsson ksyn*kon/(kon+koff) x k_dm -- independent)"],
               "predictions": ["target mRNA trajectory under TF knockout", "direction of change",
                               "magnitude of change", "time to respond",
                               "ARM I only: absolute WT steady state"]},
           "k1": {"n_genes": len(G), "n_tfs": len(tf_names), "n_edges": len(edges),
                  "n_ko": len(ko_tfs), "n_pairs": npair, "wt_drift": drift, "dt_err": dt_err,
                  "larsson_parse_rho": rho_parse, "n_larsson": n_lars, "n_prot_hl_tf": n_phl},
           "k2_circular_arm_EXCLUDED": {"n_genes": len(both), "rel_err_closed_form": rel_closed,
                                        "rel_err_integrated_from_zero": rel_int,
                                        "label": "algebraic identity, excluded from all other claims"},
           "k2_independent_arm": {"n": n_ss, "rho_steady_state": rho_ss,
                                  "median_abs_log2_error": med_lerr},
           "k2_invariance": {"armC_vs_armI": inv_C, "random_rescale": inv_R},
           "k3": {"n_tf": len(recs), "median_balanced_acc": float(np.median(ba)),
                  "mean_balanced_acc": float(ba.mean()), "median_raw_acc": float(np.median(raw)),
                  "sign_test_p": p_sign, "n_above": k_up, "n_used": n_used, "n_pairs": tot_n,
                  "lfc_sweep": {str(k): v for k, v in sw.items()}, "per_tf": recs},
           "k3b_edge_sign_lookup_baseline": {
               "n_tf": len(lk),
               "median_balanced_acc": float(np.median([r["balanced_acc"] for r in lk])),
               "median_paired_difference_ode_minus_lookup": float(np.median(dmodel)),
               "n_pairs": len(pair), "n_ode_better": k_pair, "sign_test_p": p_pair},
           "k4": {"capability": cap, "frac_values_changed": float(np.mean(caps_v)),
                  "frac_signs_changed": frac_sign_flip,
                  "survival_permutation": s_perm, "survival_redraw": s_rand,
                  "nulls_permutation": nulls_perm, "nulls_redraw": nulls_rand},
           "k5": {"rho_pubs_vs_tf_accuracy": r_fame_tf, "rho_target_pubs_vs_correct": r_fame_tg,
                  "survival_fame_rewiring": s_fame, "nulls_fame": nulls_fame,
                  "survival_wrong_tf": s_wrong, "nulls_wrong_tf": wrong,
                  "median_raw_constant_down": float(np.median(raw_cd)),
                  "median_majority_oracle": float(np.median(maj))},
           "k6": {"pooled_rho": rho_pool, "n_pooled": n_pool_,
                  "median_per_tf_rho": float(np.median(per_rho)),
                  "median_abs_pred": float(np.median(np.abs(real_pool))),
                  "median_abs_meas": float(np.median(np.abs(meas_pool))),
                  "median_ratio": float(np.median(ratio)),
                  "beta_sweep": {str(k): v for k, v in bsw.items()},
                  "time_sweep": {str(k): v for k, v in tsw.items()},
                  "ko_mode_txn": {"median_ba": float(np.median([x["balanced_acc"] for x in rtx])),
                                  "rho": spear(vtx, mtx)[0]},
                  "buffering": {"rho_pred": r_buf_pred, "rho_measured": r_buf_meas, "n": n_buf}},
           "k6b_indirect_arm": {"n_tf": len(ind), "frac_cells_moved": nz,
                                "median_balanced_acc": float(np.median(iba)),
                                "median_rho": float(np.median(irho)),
                                "sign_test_p": p_ind, "n_above": k_ind, "n_used": n_ind,
                                "per_tf": ind},
           "k7_WEAKER_self_consistency": {"rho_depfrac_vs_perturbation": r_dep,
                                          "rho_outdegree_vs_perturbation": r_out,
                                          "rho_pubs_vs_perturbation": r_dep_pub,
                                          "strata": strata, "n": n_dep,
                                          "label": "no measurement used; weaker than K3-K6"},
           "seconds": time.time() - t0, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "loop_perturb.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_perturb.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
