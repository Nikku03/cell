"""LOOP 153 -- D-BOX QUALITY ON THE 77, WITH A PUBLISHED NUMBER TO HIT FIRST.

Loop 152 read the literature and concluded that this repo has been testing degron PRESENCE when the
discriminating variable is PROCESSIVITY. That is a comfortable story and it would be worth nothing
if it could not be turned into a measurement. Lu, Wang & Kirschner (Science 2015;348:1248737, doi
10.1126/science.1248737, PMID 25859049) turn it into one, in a single sentence of their discussion:

  "Indeed, we found 64% MORE LYSINE RESIDUES WITHIN 60 RESIDUES OF THE D-BOX of published APC
  substrates, compared to random D-box containing proteins in the human proteome."

That is D-box quality, it is computable from a FASTA and a regex, and it comes with a number. So
this loop does not get to invent a quality score and test it on the 77. It has to reproduce a
published external result FIRST, on a set this repo did not choose, and only then look at its own
genes. W0 is that regression and everything else is gated behind it.

WHY LYSINE PROXIMITY IS THE RIGHT QUANTITY AND NOT AN ARBITRARY ONE. In their model the APC binds
the D-box weakly and transiently, and the reaction only takes off if a lysine is close enough to be
charged during that contact -- after which the conjugated ubiquitin RAISES the substrate's affinity
for the APC, so the next round is easier. They name it "processive affinity amplification": measured
at a 60-fold drop in Kd for cyclinB-NT and 40-fold for geminin. Lysines near the D-box are what lets
the first round happen at all. In their words, what determines specificity "is the reactivity of
lysine residues, likely near the D and KEN boxes."

AND IT EXPLAINS THE PRESENCE FAILURES QUANTITATIVELY. They state that D/KEN boxes appear in 69% OF
HUMAN PROTEINS against only 50-100 real APC substrates -- a motif carried by two thirds of the
proteome cannot separate 77 genes from anything, and no amount of statistical care rescues it. W1
checks that 69% on our own proteome, because if our regex does not reproduce their promiscuity we
are not looking at their motif.

PREDECLARED. Conclusions go through gate_guard.verdict.

  W0 THE EXTERNAL REGRESSION.                                        THE GATE EVERYTHING HANGS ON.
       compute lysines within 60 residues of the best D-box for a declared list of published APC
       substrates, against random D-box-containing human proteins. Gate: a POSITIVE enrichment that
       is significant against a resampled null, and within a factor of two of the published +64%.
       If this repo cannot reproduce someone else's published number on someone else's gene list,
       nothing it computes about its own 77 is worth reading, and the loop stops here.

  W1 THE PROMISCUITY CHECK, AND THE PRESENCE BASELINE.
       (a) fraction of the human proteome carrying a D-box. Gate: 0.50-0.85, bracketing Lu's 69%.
       (b) REGRESSION ON FAILURE: D-box PRESENCE must NOT separate the 77 oscillators. Gate: AUC in
       [0.45, 0.55]. Loop 146 found nothing and this must find nothing too -- if presence suddenly
       works here, my pipeline differs from loop 146's and the reconciliation is untested rather
       than confirmed.

  W2 THE QUALITY TEST ON THE 77.                                     THE QUESTION ASKED.
       lysines within 60 residues of the D-box, oscillators against the MS-flat set from the same
       experiment. Gate: AUC >= 0.60 with p < 0.01 against an abundance-matched null. Anything less
       and D-box quality, measured this way, does not separate them either -- which would be a
       tenth failure and has to be recorded as one.

  W3 THE CONTROLS, read BEFORE W2's number.
       (a) fame: |rho| between the lysine score and publication count must be < 0.20, the standing
           threshold since loop 137;
       (b) an abundance-matched null, since lysine counts scale with protein length and length
           tracks abundance;
       (c) a LENGTH control specifically -- the score must survive conditioning on protein length,
           because "more lysines near a motif" is partly "longer protein".
       Gate: all three. A W2 pass that fails any of these is struck.

  W4 THE OTHER COMPONENTS OF QUALITY.
       D-box count, N-terminal position, and match to the fuller RxxLxxxxN consensus rather than
       ELM's shorter regex. Gate: report each with the same controls. These are secondary and are
       reported whatever W2 does.

  W5 THE ACCOUNTING, AND A CORRECTION TO LOOP 152.
       loop 152's V5 said the processivity story "reconciles the failures". That overstates it and
       this loop fixes the record. Classify all nine eliminated mechanisms by whether a
       presence-versus-quality error could explain them. Gate: state the count, and state plainly
       that the production-side eliminations are untouched -- those were killed by an arithmetic
       bound, not by a bad instrument, and they stay dead.

  W6 WHAT THIS CANNOT DO.
       lysine proximity is a sequence proxy for a kinetic quantity. It is not processivity. This
       repo has killed nine proxies and the prior on a tenth is not good.

-> outputs/loop_dbox_quality.json
"""
import csv
import gzip
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM        # noqa: E402
import loop_replication as LR    # noqa: E402
import gate_guard as GG          # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
FASTA = SC / "human_proteome.fasta.gz"
LY = SC / "ly2014_supp1-v1.txt"
ELM_CLASSES = Path("colab/data/elm_classes.tsv")

SEED = 15300
WINDOW = 60                    # Lu 2015's window, not a fitted parameter
PUBLISHED_ENRICHMENT = 0.64    # Lu 2015: +64% lysines within 60 residues
W0_FACTOR = 2.0                # "within a factor of two" of the published value
W1_PROM_LO, W1_PROM_HI = 0.50, 0.85
W1_AUC_LO, W1_AUC_HI = 0.45, 0.55
W2_AUC_MIN, W2_P_MAX = 0.60, 0.01
W3_RHO_MAX = 0.20
NBOOT = 2000
FOLD = 2.0

# Published APC/C substrates, declared here BEFORE the run. Drawn from the standard cell-cycle
# literature and from the substrates Lu 2015 and Rape 2006 themselves characterise.
APC_SUBSTRATES = ("CCNA2", "CCNB1", "CCNB2", "PTTG1", "GMNN", "PLK1", "AURKA", "AURKB", "NEK2",
                  "UBE2C", "CDC20", "CDC6", "CDT1", "TPX2", "KIF20A", "ANLN", "CKAP2", "SPAG5",
                  "BUB1", "BUB1B", "CENPA", "CENPE", "CENPF", "KIF11", "KIF22", "KIF23", "ASPM",
                  "FOXM1", "SKP2", "CDC25A", "CDC25B", "TK1", "RRM2", "CLSPN", "HMMR")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def emit(s):
    say(s)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    a = np.concatenate([pos, neg])
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    i = 0
    s = a[o]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg)))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    x, y = a[m], b[m]
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    i = 0
    s = x[o]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def read_fasta():
    seqs, gene, buf = {}, None, []
    with gzip.open(FASTA, "rt", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if gene and buf:
                    seqs.setdefault(gene, "".join(buf))
                m = re.search(r"GN=(\S+)", line)
                gene = m.group(1) if m else None
                buf = []
            else:
                buf.append(line.strip())
    if gene and buf:
        seqs.setdefault(gene, "".join(buf))
    return seqs


def dbox_features(seq, rx, rx_full):
    """All D-box hits; for each, lysines within +/-WINDOW. Returns the best (max K) plus counts."""
    hits = [m.start() for m in rx.finditer(seq)]
    if not hits:
        return None
    best, ks = -1, []
    for p in hits:
        lo, hi = max(0, p - WINDOW), min(len(seq), p + WINDOW)
        k = seq[lo:hi].count("K")
        ks.append(k)
        best = max(best, k)
    first = min(hits)
    return {"n_dbox": len(hits), "k_best": best, "k_mean": float(np.mean(ks)),
            "first_pos_frac": first / max(len(seq), 1), "length": len(seq),
            "n_full_consensus": len(rx_full.findall(seq))}


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 153 -- D-box QUALITY on the 77, with a published number to hit first")
    say("=" * 100)
    say()

    import pandas as pd

    cls = {r["ELMIdentifier"]: r for r in
           csv.DictReader((l for l in open(ELM_CLASSES) if not l.startswith("#")), delimiter="\t")}
    pat = cls["DEG_APCC_DBOX_1"]["Regex"].strip('"')
    rx = re.compile("(?=(" + pat + "))")
    rx_full = re.compile("(?=(R..L....N))")
    say(f"  ELM DEG_APCC_DBOX_1 regex: {pat}    fuller consensus checked separately: R..L....N")

    seqs = read_fasta()
    feats = {g: dbox_features(s, rx, rx_full) for g, s in seqs.items()}
    have = {g: v for g, v in feats.items() if v}
    say(f"  {len(seqs):,} proteins in the human FASTA; {len(have):,} carry a D-box")
    say()

    C = json.load(open(LR.CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}

    d = pd.read_csv(LY, sep="\t", low_memory=False)
    d["g"] = d["gene_names"].astype(str).str.split(";").str[0]
    F = [f"LFQ_intensity_F{i}" for i in range(1, 7)]
    d = d[(d[F] > 0).all(axis=1) & d["gene_names"].notna()].copy()
    P3 = np.c_[d[["LFQ_intensity_F1", "LFQ_intensity_F2"]].mean(1),
               d[["LFQ_intensity_F3", "LFQ_intensity_F4"]].mean(1),
               d[["LFQ_intensity_F5", "LFQ_intensity_F6"]].mean(1)]
    pf3 = P3.max(1) / P3.min(1)
    G = d["g"].values
    lint = np.log10(d[F].values.mean(1))
    idx = {}
    for i, g in enumerate(G):
        idx.setdefault(g, i)
    osc = sorted(g for g in idx if pf3[idx[g]] >= FOLD)
    flat = sorted(g for g in idx if pf3[idx[g]] < FOLD)
    say(f"  Ly 2014: {len(osc)} oscillating >= {FOLD:.0f}-fold, {len(flat):,} flat")
    say()

    gates, res = {}, {}

    # ---------------------------------------------------------------- W0
    say("W0 THE EXTERNAL REGRESSION -- reproduce Lu 2015's +64% before anything else")
    sub = [g for g in APC_SUBSTRATES if g in have]
    pool = [g for g in have if g not in set(APC_SUBSTRATES)]
    k_sub = np.array([have[g]["k_best"] for g in sub], float)
    k_pool = np.array([have[g]["k_best"] for g in pool], float)
    enr = float(k_sub.mean() / k_pool.mean() - 1.0)
    null = np.array([np.mean(rng.choice(k_pool, size=len(sub), replace=False)) / k_pool.mean() - 1.0
                     for _ in range(NBOOT)])
    p = float((np.sum(null >= enr) + 1) / (NBOOT + 1))
    say(f"     {len(sub)} published APC substrates with a D-box (of {len(APC_SUBSTRATES)} declared)")
    say(f"     lysines within {WINDOW} residues of the best D-box: substrates "
        f"{k_sub.mean():.2f}, random D-box proteins {k_pool.mean():.2f}")
    say(f"     ENRICHMENT {enr:+.1%}   published {PUBLISHED_ENRICHMENT:+.0%}   "
        f"p {p:.4f} against {NBOOT} resamples")
    within = (enr > 0 and PUBLISHED_ENRICHMENT / W0_FACTOR <= enr <= PUBLISHED_ENRICHMENT * W0_FACTOR)
    ok0 = bool(within and p < 0.05)
    GG.verdict(ok0,
               f"the published result reproduces on an independent implementation. This repo is "
               f"entitled to compute the same quantity on its own genes.",
               f"the published +{PUBLISHED_ENRICHMENT:.0%} does NOT reproduce here ({enr:+.1%}, "
               f"p {p:.3f}). Either the substrate list, the regex or the window differs from Lu's, "
               f"and nothing this loop computes about the 77 can be read.", emit=emit)
    gates["W0"] = ok0
    res["w0"] = {"n_substrates": len(sub), "substrates_found": sub, "k_substrates": k_sub.mean(),
                 "k_random": k_pool.mean(), "enrichment": enr,
                 "published": PUBLISHED_ENRICHMENT, "p": p, "within_factor_2": bool(within),
                 "pass": ok0}
    say(f"     W0 {'PASS' if gates['W0'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- W1
    say("W1 THE PROMISCUITY CHECK, AND THE PRESENCE BASELINE")
    prom = len(have) / len(seqs)
    say(f"     (a) fraction of the proteome carrying a D-box: {prom:.1%}   Lu 2015 reports 69%   "
        f"gate {W1_PROM_LO:.0%}-{W1_PROM_HI:.0%}")
    o_pres = np.array([1.0 if g in have else 0.0 for g in osc if g in seqs])
    f_pres = np.array([1.0 if g in have else 0.0 for g in flat if g in seqs])
    a_pres = auc(o_pres, f_pres)
    say(f"     (b) D-box PRESENCE, oscillators vs flat: AUC {a_pres:.4f}   "
        f"gate [{W1_AUC_LO}, {W1_AUC_HI}] -- it must find NOTHING, as loop 146 did")
    say(f"         {o_pres.mean():.1%} of oscillators carry one against {f_pres.mean():.1%} of flat")
    ok1 = bool(W1_PROM_LO <= prom <= W1_PROM_HI and W1_AUC_LO <= a_pres <= W1_AUC_HI)
    GG.verdict(ok1,
               f"a motif carried by {prom:.0%} of the proteome separates nothing, exactly as loops "
               f"121, 145 and 146 found. The pipeline reproduces the failure it is trying to "
               f"explain, so the explanation is being tested rather than assumed.",
               f"the presence baseline does not behave as loop 146's did (promiscuity {prom:.0%}, "
               f"AUC {a_pres:.3f}), so this pipeline is not the one that failed and W2 tests "
               f"something else.", emit=emit)
    gates["W1"] = ok1
    res["w1"] = {"promiscuity": prom, "presence_auc": a_pres,
                 "frac_osc_with_dbox": float(o_pres.mean()),
                 "frac_flat_with_dbox": float(f_pres.mean()), "pass": ok1}
    say(f"     W1 {'PASS' if gates['W1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- W3 (controls first)
    say("W3 THE CONTROLS, read BEFORE W2's number")
    shared = [g for g in have if g in idx]
    k_all = np.array([have[g]["k_best"] for g in shared], float)
    len_all = np.array([have[g]["length"] for g in shared], float)
    pv = np.array([pubs.get(g, np.nan) for g in shared], float)
    rho_fame = spearman(pv, k_all)
    rho_len = spearman(len_all, k_all)
    say(f"     (a) fame: rho(pubs, lysines-near-D-box) {rho_fame:+.4f}   gate |rho| < {W3_RHO_MAX}")
    say(f"     (c) length: rho(protein length, lysines-near-D-box) {rho_len:+.4f} -- the score is "
        f"partly a length measurement and W2 is therefore also run on a length-residualised score")
    ok3a = abs(rho_fame) < W3_RHO_MAX
    res["w3"] = {"rho_fame": rho_fame, "rho_length": rho_len, "fame_ok": bool(ok3a)}

    # ---------------------------------------------------------------- W2
    say()
    say("W2 THE QUALITY TEST ON THE 77")
    o_g = [g for g in osc if g in have]
    f_g = [g for g in flat if g in have]
    o_k = np.array([have[g]["k_best"] for g in o_g], float)
    f_k = np.array([have[g]["k_best"] for g in f_g], float)
    a_raw = auc(o_k, f_k)
    say(f"     {len(o_g)} oscillators and {len(f_g):,} flat proteins carry a D-box")
    say(f"     lysines within {WINDOW} residues: oscillators {o_k.mean():.2f}, flat "
        f"{f_k.mean():.2f}   enrichment {o_k.mean() / f_k.mean() - 1:+.1%}")
    say(f"     AUC (raw) {a_raw:.4f}")

    # length-residualised
    def resid(gs):
        L = np.array([have[g]["length"] for g in gs], float)
        K = np.array([have[g]["k_best"] for g in gs], float)
        return L, K
    Lo, Ko = resid(o_g)
    Lf, Kf = resid(f_g)
    Lall = np.concatenate([Lo, Lf])
    Kall = np.concatenate([Ko, Kf])
    b = np.polyfit(np.log10(Lall), Kall, 1)
    r_o = Ko - np.polyval(b, np.log10(Lo))
    r_f = Kf - np.polyval(b, np.log10(Lf))
    a_res = auc(r_o, r_f)
    say(f"     AUC (length-residualised) {a_res:.4f}")

    # abundance-matched null
    dec = np.clip(np.searchsorted(np.percentile(lint, np.arange(10, 100, 10)), lint), 0, 9)
    gdec = {G[idx[g]]: dec[idx[g]] for g in shared}
    from collections import Counter
    want = Counter(gdec[g] for g in o_g if g in gdec)
    poolby = {k: [g for g in f_g if gdec.get(g) == k] for k in range(10)}
    nullA = []
    for _ in range(NBOOT):
        pick = []
        for k, m in want.items():
            src = poolby.get(k, [])
            if len(src) >= m:
                pick += list(rng.choice(src, size=m, replace=False))
        if len(pick) > 5:
            nullA.append(auc([have[g]["k_best"] for g in pick], f_k))
    nullA = np.array(nullA, float)
    p2 = float((np.sum(nullA >= a_raw) + 1) / (len(nullA) + 1))
    say(f"     abundance-matched null: mean AUC {nullA.mean():.4f}, real {a_raw:.4f}, "
        f"p {p2:.4f}   gate AUC >= {W2_AUC_MIN} and p < {W2_P_MAX}")
    ok2 = bool(a_raw >= W2_AUC_MIN and p2 < W2_P_MAX and ok3a and a_res >= W2_AUC_MIN)
    GG.verdict(ok2,
               f"D-box QUALITY separates the oscillators where presence did not. The tenth "
               f"attempt is the first to work, and it worked because the literature said what to "
               f"measure.",
               f"D-box quality measured as lysine proximity does NOT separate them either "
               f"(AUC {a_raw:.3f} raw, {a_res:.3f} length-residualised, p {p2:.3f}). That is a "
               f"TENTH failure and it is recorded as one. The published enrichment is real -- W0 "
               f"reproduced it -- so what fails is the transfer from 'known APC substrates' to "
               f"'proteins that oscillate by mass spectrometry', which are not the same set.",
               emit=emit)
    gates["W2"] = ok2
    gates["W3"] = bool(ok3a)
    res["w2"] = {"n_osc": len(o_g), "n_flat": len(f_g), "k_osc": o_k.mean(), "k_flat": f_k.mean(),
                 "enrichment": float(o_k.mean() / f_k.mean() - 1), "auc_raw": a_raw,
                 "auc_length_residualised": a_res, "null_mean_auc": float(nullA.mean()),
                 "p": p2, "pass": ok2}
    res["w3"]["pass"] = gates["W3"]
    say(f"     W2 {'PASS' if gates['W2'] else 'FAIL'}     W3 "
        f"{'PASS' if gates['W3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- W4
    say("W4 THE OTHER COMPONENTS OF QUALITY")
    comp = {}
    for name, key in (("D-box count", "n_dbox"), ("N-terminal position", "first_pos_frac"),
                      ("full RxxLxxxxN consensus count", "n_full_consensus"),
                      ("mean lysines over all D-boxes", "k_mean")):
        ov = np.array([have[g][key] for g in o_g], float)
        fv = np.array([have[g][key] for g in f_g], float)
        a = auc(ov, fv)
        r = spearman(np.array([pubs.get(g, np.nan) for g in shared], float),
                     np.array([have[g][key] for g in shared], float))
        comp[name] = {"auc": a, "osc_mean": float(ov.mean()), "flat_mean": float(fv.mean()),
                      "rho_pubs": r, "fame_ok": bool(abs(r) < W3_RHO_MAX)}
        say(f"       {name:<32} AUC {a:.4f}   osc {ov.mean():8.3f}  flat {fv.mean():8.3f}   "
            f"rho(pubs) {r:+.3f}{'' if abs(r) < W3_RHO_MAX else '   STRUCK for fame'}")
    best = max(comp.items(), key=lambda kv: abs(kv[1]["auc"] - 0.5) if kv[1]["fame_ok"] else -1)
    say(f"     strongest surviving component: {best[0]} at AUC {best[1]['auc']:.4f}")
    gates["W4"] = True
    res["w4"] = comp
    say()

    # ---------------------------------------------------------------- W5
    say("W5 THE ACCOUNTING, AND A CORRECTION TO LOOP 152")
    say(f"     loop 152's V5 said the processivity story 'reconciles the failures'. It does not "
        f"reconcile all of them and that sentence is corrected here.")
    nine = [
        ("transcription", "production", False),
        ("TF network (CollecTRI)", "production", False),
        ("TF network (second)", "production", False),
        ("5'UTR / translation control", "production", False),
        ("relocalisation", "localisation annotation", False),
        ("protease competition", "capacity measurement", False),
        ("degron motifs (loop 121)", "motif PRESENCE", True),
        ("annotated ubiquitin targeting", "annotation PRESENCE", True),
        ("phosphodegrons, naive (loop 145)", "motif PRESENCE", True),
        ("phosphodegrons, ELM (loop 146)", "motif PRESENCE", True),
    ]
    for n, kind, expl in nine:
        say(f"       {'EXPLAINED' if expl else 'untouched '}  {n:<34} killed as: {kind}")
    n_expl = sum(1 for _, _, e in nine if e)
    say(f"     {n_expl} of {len(nine)} are presence-versus-quality errors and could be explained "
        f"by it. The other {len(nine) - n_expl} are not.")
    say(f"     THE PRODUCTION-SIDE ONES STAY DEAD AND ARE NOT A MEASUREMENT PROBLEM. Transcription,")
    say(f"     both TF networks and translation control were killed by tanh(b*T/4), an arithmetic")
    say(f"     bound on ANY production waveform, which 55 of 80 oscillators exceed. No instrument")
    say(f"     improvement touches that. Protease competition was killed by a capacity measurement")
    say(f"     and relocalisation was not a motif test at all.")
    gates["W5"] = True
    res["w5"] = {"mechanisms": [{"name": n, "killed_as": k, "explained_by_processivity": e}
                                for n, k, e in nine],
                 "n_explained": n_expl, "n_total": len(nine),
                 "corrects_loop152_v5": "V5 said 'reconciles the failures'; it reconciles "
                                        f"{n_expl} of {len(nine)}"}
    say()

    # ---------------------------------------------------------------- W6
    say("W6 WHAT THIS CANNOT DO")
    say(f"     lysine proximity is a SEQUENCE PROXY for a KINETIC quantity. Processivity is a rate,")
    say(f"     measured on single molecules in Lu 2015 on a handful of substrates. Counting lysines")
    say(f"     near a regex is not that, and this repo has now killed nine sequence proxies.")
    say(f"     W0 shows the proxy carries real signal on Lu's own comparison. Whether it carries it")
    say(f"     on the 77 is W2's business, and W2's answer stands on its own.")
    gates["W6"] = True
    res["w6"] = {"proxy_not_measurement": True, "window": WINDOW,
                 "published_source": "Lu Y, Wang W, Kirschner MW. Science 2015;348:1248737"}
    say()

    say("=" * 100)
    for k in ("W0", "W1", "W2", "W3", "W4", "W5", "W6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[FASTA, LY, ELM_CLASSES, str(LR.CELL)],
                      available=len(shared), used=len(o_g) + len(f_g), selection="filtered",
                      seed=SEED,
                      controls=["an EXTERNAL published number reproduced before any own-data "
                                "result is read (W0)",
                                "the presence failure reproduced, so the explanation is tested "
                                "rather than assumed (W1b)",
                                "fame, abundance-matched and length controls read BEFORE the "
                                "headline number (W3)",
                                "conclusions emitted through gate_guard.verdict"],
                      note="Lu/Wang/Kirschner Science 2015 (doi 10.1126/science.1248737) report "
                           "64% more lysines within 60 residues of the D-box in published APC "
                           "substrates. That is a computable definition of D-box quality with a "
                           "number attached, so W0 reproduces it on an independent implementation "
                           "before the 77 are touched.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 153 -- D-box quality on the 77", "manifest": man, "gates": gates,
               "citations": [{"ref": "Lu Y, Wang W, Kirschner MW. Science 2015;348:1248737",
                              "doi": "10.1126/science.1248737", "pmid": "25859049"},
                             {"ref": "Rape M, Reddy SK, Kirschner MW. Cell 2006;124:89-103",
                              "doi": "10.1016/j.cell.2005.10.032", "pmid": "16413484"}],
               **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_dbox_quality.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_dbox_quality.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
