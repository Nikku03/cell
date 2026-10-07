"""LOOP 122 -- THE LAST TERM: regulated translation, fetched, and the bound that applies to all of them.

WHERE THE ARC STANDS. dP/dt = k_sp*M - b*P has three time-varying things in it and this repository
has now tested all but one.

    M(t)      loop 119   ELIMINATED. 362 proteins oscillate whose mRNA is flat, against 38 the
                         other way, p = 2e-67. The filter forces the opposite, at 100.0000% of
                         4,190 genes.
    the wiring loop 120  ELIMINATED. Real edge signs score AUC 0.5465 against SHUFFLED signs at
                         0.5494. Regulator count alone beats every network score.
    b(t)      loop 121   WRONG GENES. Degron density ranks both-oscillate > transcript-only >
                         protein-only for all three motifs. The proteins that need a post-
                         transcriptional mechanism carry the LEAST degron signal.
    k_sp(t)   THIS LOOP  the remaining half of the space, and it was untestable because the
                         signature lives in the 5' UTR and this repository had no 5' UTR.

WHAT WAS FETCHED, AND HOW IT WAS BUILT. There is no 5'-UTR-only file to download. Ensembl publishes
the full transcript (cdna) and the coding sequence (cds) separately, so the 5' UTR is what remains
when the CDS is subtracted -- exact, not predicted. The isoform is not chosen by this loop: NCBI's
MANE Select names one transcript per gene as the reference, which is what fixes the transcription
start site, and the TSS is the whole ballgame for a cap-proximal motif.

    Ensembl GRCh38 cdna.all + cds.all -> 382,428 matched transcripts
    MANE Select v1.5 -> 19,363 genes, one transcript each
    5' UTR recovered for 19,293 genes, median 133 nt

MEASURED DURING CONSTRUCTION AND THEREFORE DISCLOSED, NOT CLAIMED AS AN UNSEEN RESULT: the 5'TOP
positive control. 71.6% of the 81 ribosomal proteins carry C[CT]{4,14} at position 0, against 4.4%
of the other 19,212 genes -- 16.3x. That is the textbook answer for the textbook TOP mRNAs, and it
says the TSS annotation is right. U1 keeps the check but names it as pre-measured, and adds a
SECOND positive control that has not been run: the Kozak -3 purine fraction, which validates the
CDS boundary independently of the TSS.

THE PREDICTION, STATED BEFORE ANY NUMBER. Cap-dependent initiation falls roughly two-fold in
mitosis. The mRNAs that keep being translated through it are the ones that do not need an
unobstructed cap-proximal scan: long 5' UTRs, GC-rich and therefore structured, and uORF-bearing.
So the 362 protein-only oscillators -- the ones with no transcriptional source by construction --
should have LONGER, MORE GC-RICH, MORE uORF-CONTAINING 5' UTRs than both the imaged non-CCD
controls and the transcript-only group. That is a direction, committed in advance, and loop 121's
equivalent prediction came out backwards.

PREDECLARED:

  U1 THE INSTRUMENT IS CORRECT                                      THE PREREQUISITE.
       (a) PRE-MEASURED, disclosed above: 5'TOP in ribosomal proteins >= 5x the rest of the
           proteome. Retained because the gene subset here is not the one it was measured on.
       (b) NOT YET RUN: the Kozak -3 purine fraction genome-wide must be >= 0.70. Human start
           codons sit in a purine at -3 in the large majority; if this comes out near 0.5 the CDS
           boundary is misplaced and every feature in the loop is measuring the wrong window.
  U2 THE FEATURES ARE NOT FAME PROXIES, ONE BY ONE                  LOOP 121's S1, FIXED.
       |Spearman(feature, publication count)| < 0.15, per feature. loop 121 gated this all-or-
       nothing and lost the whole gate to two motifs out of five. Here each feature passes or
       fails on its own, and a feature that fails is EXCLUDED from U3-U5 rather than quietly used.
       Gate: at least three features survive, so the loop still has something to test with.
  U3 CCD PROTEINS DIFFER FROM THE IMAGED CONTROLS                   THE MEASUREMENT.
       748 against 776, per surviving feature, permutation p with Bonferroni across features, and
       a composition-preserving sequence shuffle for the motif features. Gate: at least one
       survivor significant after correction.
  U4 THE DISCRIMINATING TEST                                        THE ONE THAT MATTERS.
       protein-only (362) against transcript-only (38) and against neither (642), in the direction
       predicted above. Gate: predicted direction AND permutation p < 0.05 on at least one
       surviving feature. This is exactly the gate loop 121 failed backwards; the same standard
       applies and the same reversal is possible.
  U5 TRANSLATION AGAINST DEGRADATION AGAINST GEOMETRY, HEAD TO HEAD  THE COMPARISON.
       the best 5' UTR feature against loop 121's best degron feature (D-box+, AUC 0.5350) and its
       post-hoc relocalisation feature (dual localisation, AUC 0.5992), all recomputed on the same
       genes so the numbers are commensurable. Gate: the 5' UTR feature must beat the degron
       feature. Losing to relocalisation is reported, not gated -- that hypothesis is post-hoc and
       same-instrument, and gating on it would launder it into a result.
  U6 THE FILTER BOUNDS EVERY MECHANISM EQUALLY                      THE STRUCTURAL RESULT.
       cell_assembled.integrate_tl drives k_sp; integrate_deg drives b. Linearised, both give
       relative amplitude beta * gain(bbar, T) -- the SAME first-order filter with the SAME corner
       frequency, driven from opposite sides. Gate, three parts:
         (a) beta = 0 reduces exactly for the translation integrator, < 1e-9;
         (b) the two integrators agree on relative amplitude to within 1% per gene;
         (c) therefore the fraction of measured CCD proteins that NO mechanism inside this equation
             can explain is computable, and is reported.
       If (b) holds, then arguing about which term carries the cell cycle is beside the point for
       every protein above the half-life threshold: none of them can, and the equation itself is
       what has to change.

-> outputs/loop_utr5.json
"""
import csv
import gzip
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
HPA = LR.SC / "proteinatlas.tsv"
CDNA = LR.SC / "Homo_sapiens.GRCh38.cdna.all.fa.gz"
CDS = LR.SC / "Homo_sapiens.GRCh38.cds.all.fa.gz"
MANE = LR.SC / "MANE.summary.txt.gz"
SEED = 12200
NPERM = 2000
NSEQ = 200
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H
T_CYCLE = 24.0
AMP_MIN = 0.20

U1_TOP = 5.0
U1_KOZAK = 0.70
U2_RHO = 0.15
U2_MIN_SURVIVORS = 3
U6_TOL = 1e-9
U6_AGREE = 0.01

TOP = re.compile(r"^C[CT]{4,14}")
MOTIF_FEATURES = ("5'TOP motif", "uORF count", "uORF density/100nt")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sv, r = allv[order], np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def perm_p(a, b, rng, n=NPERM):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    k = len(a)
    null = np.array([(lambda s: s[:k].mean() - s[k:].mean())(rng.permutation(pool))
                     for _ in range(n)])
    return float(obs), float(np.mean(np.abs(null) >= abs(obs)))


def read_fa(path):
    out, tid, buf = {}, None, []
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln[0] == ">":
                if tid:
                    out[tid] = "".join(buf)
                buf, tid = [], ln[1:].split()[0].split(".")[0]
            else:
                buf.append(ln.strip())
    if tid:
        out[tid] = "".join(buf)
    return out


def features(u, cd, ci):
    """Every feature computed from the 5' UTR sequence and the start-codon context. Nothing else."""
    n = len(u)
    gc = (u.count("G") + u.count("C")) / n if n else float("nan")
    cap = u[:40]
    gc40 = ((cap.count("G") + cap.count("C")) / len(cap)) if cap else float("nan")
    uorf = u.count("ATG")
    koz = cd[ci - 3] if ci >= 3 else ""
    return {"5'UTR length": float(n),
            "log10 length": float(np.log10(n + 1)),
            "GC content": gc,
            "cap-proximal GC (first 40 nt)": gc40,
            "uORF count": float(uorf),
            "uORF density/100nt": 100.0 * uorf / n if n else float("nan"),
            "5'TOP motif": float(bool(TOP.match(u))),
            "_kozak_m3": koz}


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 122 -- the last term: regulated translation, and the bound on all of them")
    say("=" * 100)
    say()

    sel = {}
    r = csv.reader(gzip.open(MANE, "rt"), delimiter="\t")
    h = next(r)
    i_s, i_e, i_st = h.index("symbol"), h.index("Ensembl_nuc"), h.index("MANE_status")
    for x in r:
        if "Select" in x[i_st]:
            sel[x[i_s]] = x[i_e].split(".")[0]
    cdna, cds = read_fa(CDNA), read_fa(CDS)
    utr, cdseq, cdidx = {}, {}, {}
    for g, t in sel.items():
        if t in cdna and t in cds:
            i = cdna[t].find(cds[t])
            if i >= 0:
                utr[g], cdseq[g], cdidx[g] = cdna[t][:i], cdna[t], i
    say(f"  Ensembl GRCh38 cdna+cds: {len(cdna):,} / {len(cds):,} transcripts; "
        f"MANE Select v1.5: {len(sel):,} genes")
    say(f"  5' UTR recovered by CDS subtraction for {len(utr):,} genes, "
        f"median {int(np.median([len(v) for v in utr.values()]))} nt")

    D = CA.load()
    pubs = D["pubs"]
    with open(HPA, newline="") as f:
        rr = csv.reader(f, delimiter="\t")
        hh = next(rr)
        jG, jP, jT = hh.index("Gene"), hh.index("CCD Protein"), hh.index("CCD Transcript")
        jM, jA = hh.index("Subcellular main location"), hh.index("Subcellular additional location")
        rows = [(x[jG], x[jP], x[jT], x[jM], x[jA]) for x in rr]
    cp = {g: v for g, v, _, _, _ in rows if v in ("Yes", "No")}
    ct = {g: v for g, _, v, _, _ in rows if v in ("Yes", "No")}
    locn = {g: (m, a) for g, _, _, m, a in rows}

    called = sorted(g for g in cp if g in utr and len(utr[g]) > 0)
    F = {g: features(utr[g], cdseq[g], cdidx[g]) for g in called}
    FEATS = [k for k in F[called[0]] if not k.startswith("_")]
    lab = np.array([cp[g] == "Yes" for g in called])
    say(f"  CCD called AND with a 5' UTR: {len(called):,}  "
        f"({int(lab.sum())} Yes / {int((~lab).sum())} No)")
    say()

    gates = {}

    # ---------------------------------------------------------------- U1
    say("U1 THE INSTRUMENT IS CORRECT")
    rp = [g for g in utr if re.match(r"^(RPL|RPS)\d+[A-Z]?$|^RPLP\d$|^RPSA$", g)]
    oth = [g for g in utr if g not in set(rp)]
    f_rp = float(np.mean([bool(TOP.match(utr[g])) for g in rp]))
    f_ot = float(np.mean([bool(TOP.match(utr[g])) for g in oth]))
    ratio = f_rp / max(f_ot, 1e-12)
    say(f"     (a) PRE-MEASURED and disclosed: 5'TOP motif in ribosomal proteins {f_rp:.1%} "
        f"({len(rp)}) vs {f_ot:.1%} ({len(oth):,}) = {ratio:.1f}x  gate >= {U1_TOP}x")
    koz = [F[g]["_kozak_m3"] for g in called if F[g]["_kozak_m3"]]
    pur = float(np.mean([c in "AG" for c in koz]))
    kozall = [cdseq[g][cdidx[g] - 3] for g in utr if cdidx[g] >= 3]
    purall = float(np.mean([c in "AG" for c in kozall]))
    say(f"     (b) NOT PRE-RUN: Kozak -3 purine, genome-wide {purall:.1%} of {len(kozall):,}; "
        f"on the called set {pur:.1%}  gate >= {U1_KOZAK:.0%}")
    say(f"         a misplaced CDS boundary would give ~50%; this validates the window the "
        f"features are measured in")
    gates["U1"] = bool(ratio >= U1_TOP and purall >= U1_KOZAK)
    say(f"     U1 {'PASS' if gates['U1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- U2
    say("U2 THE FEATURES ARE NOT FAME PROXIES, ONE BY ONE")
    allg = [g for g in utr if g in pubs and len(utr[g]) > 0]
    pv = np.array([pubs[g] for g in allg])
    Fa = {g: features(utr[g], cdseq[g], cdidx[g]) for g in allg}
    rhos, survivors = {}, []
    for k in FEATS:
        rhos[k] = spearman([Fa[g][k] for g in allg], pv)
        ok = abs(rhos[k]) < U2_RHO
        survivors += [k] if ok else []
        say(f"     {k:<32} rho vs pubs {rhos[k]:+.4f}   {'kept' if ok else 'EXCLUDED'}")
    gates["U2"] = bool(len(survivors) >= U2_MIN_SURVIVORS)
    say(f"     {len(survivors)} of {len(FEATS)} features survive; gate >= {U2_MIN_SURVIVORS}")
    say(f"     U2 {'PASS' if gates['U2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- U3
    say("U3 CCD PROTEINS DIFFER FROM THE IMAGED CONTROLS")
    bonf = max(1, len(survivors))
    u3, ok3 = {}, False
    for k in survivors:
        v = np.array([F[g][k] for g in called])
        obs, p = perm_p(v[lab], v[~lab], rng)
        a = auc(v[lab], v[~lab])
        u3[k] = {"ccd": float(np.nanmean(v[lab])), "nonccd": float(np.nanmean(v[~lab])),
                 "difference": obs, "p": p, "p_bonferroni": min(1.0, p * bonf), "auc": a}
        say(f"     {k:<32} CCD {np.nanmean(v[lab]):>9.4f}  non-CCD {np.nanmean(v[~lab]):>9.4f}  "
            f"diff {obs:+.4f}  p {p:.4f}  p*{bonf} {min(1.0, p * bonf):.4f}  AUC {a:.4f}")
        if p * bonf < 0.05:
            ok3 = True
    # composition-preserving shuffle for the motif features that survived
    mot = [k for k in survivors if k in MOTIF_FEATURES]
    if mot:
        say(f"     composition-preserving 5' UTR shuffle, {NSEQ} rounds, for {mot}")
        arrs = {g: np.frombuffer(utr[g].encode(), dtype="S1") for g in called}
        nulls = {k: [] for k in mot}
        capb = None
        for i in range(NSEQ):
            sd = {}
            for g in called:
                s = b"".join(rng.permutation(arrs[g]).tolist()).decode()
                if i == 0 and g == called[0]:
                    capb = GG.null_can_move(list(utr[g]), list(s))
                n = max(len(s), 1)
                sd[g] = {"5'TOP motif": float(bool(TOP.match(s))),
                         "uORF count": float(s.count("ATG")),
                         "uORF density/100nt": 100.0 * s.count("ATG") / n}
            for k in mot:
                v = np.array([sd[g][k] for g in called])
                nulls[k].append(float(v[lab].mean() - v[~lab].mean()))
        for k in mot:
            s = GG.survival(u3[k]["difference"], nulls[k], z_min=2.0)
            u3[k]["seq_null"] = s
            GG.report(f"     {k} vs composition shuffle", s, emit=say)
        say(f"     the shuffle changes {capb['changed']:.1%} of {capb['achievable']:.1%} "
            f"achievable -- {'capable' if capb['capable'] else 'INERT'}")
    gates["U3"] = bool(ok3)
    say(f"     U3 {'PASS' if gates['U3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- U4
    say("U4 THE DISCRIMINATING TEST")
    say("     PREDICTED IN ADVANCE: protein-only should be LONGER, MORE GC-RICH, MORE uORF-BEARING")
    bc = [g for g in called if g in ct]
    grp = {"both": [g for g in bc if cp[g] == "Yes" and ct[g] == "Yes"],
           "protein only": [g for g in bc if cp[g] == "Yes" and ct[g] == "No"],
           "transcript only": [g for g in bc if cp[g] == "No" and ct[g] == "Yes"],
           "neither": [g for g in bc if cp[g] == "No" and ct[g] == "No"]}
    PREDICTED_UP = {"5'UTR length", "log10 length", "GC content",
                    "cap-proximal GC (first 40 nt)", "uORF count", "uORF density/100nt"}
    u4, ok4 = {}, False
    for k in survivors:
        vals = {gn: float(np.nanmean([F[g][k] for g in gs])) for gn, gs in grp.items()}
        a = np.array([F[g][k] for g in grp["protein only"]])
        b = np.array([F[g][k] for g in grp["transcript only"]])
        obs, p = perm_p(a, b, rng)
        u4[k] = {"means": vals, "protein_minus_transcript": obs, "p": p,
                 "predicted_up": k in PREDICTED_UP}
        say(f"     {k}")
        say("       " + "   ".join(f"{gn} {vals[gn]:.4f} (n={len(grp[gn])})" for gn in
                                   ("both", "protein only", "transcript only", "neither")))
        say(f"       protein-only minus transcript-only: {obs:+.4f}, p = {p:.4f}"
            f"   predicted {'UP' if k in PREDICTED_UP else 'n/a'}")
        if k in PREDICTED_UP and obs > 0 and p < 0.05:
            ok4 = True
    gates["U4"] = bool(ok4)
    say(f"     U4 {'PASS' if gates['U4'] else 'FAIL'} -- the post-transcriptional group "
        f"{'differs as predicted' if gates['U4'] else 'does NOT differ as predicted'}")
    say()

    # ---------------------------------------------------------------- U5
    say("U5 TRANSLATION AGAINST DEGRADATION AGAINST GEOMETRY, HEAD TO HEAD")
    seqp = {}
    import gzip as _gz
    prot, nm, buf = {}, None, []
    with _gz.open(LR.SC / "human_proteome.fasta.gz", "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if nm and buf and len("".join(buf)) > len(prot.get(nm, "")):
                    prot[nm] = "".join(buf)
                nm, buf = None, []
                for p in ln.split():
                    if p.startswith("GN="):
                        nm = p[3:]
                        break
            else:
                buf.append(ln.strip())
    if nm and buf and len("".join(buf)) > len(prot.get(nm, "")):
        prot[nm] = "".join(buf)
    dbx = re.compile("R..L..[LIVM]")
    inter = [g for g in called if g in prot and g in locn]
    li = np.array([cp[g] == "Yes" for g in inter])
    dv = np.array([100.0 * len(dbx.findall(prot[g])) / max(len(prot[g]), 1) for g in inter])
    mv = np.array([1.0 if len([p for p in (locn[g][0] + "," + locn[g][1]).split(",") if p.strip()])
                   > 1 else 0.0 for g in inter])
    best_u = max(survivors, key=lambda k: abs(u3[k]["auc"] - 0.5))
    uv = np.array([F[g][best_u] for g in inter])
    a_u, a_d, a_m = auc(uv[li], uv[~li]), auc(dv[li], dv[~li]), auc(mv[li], mv[~li])
    say(f"     on the same {len(inter):,} genes ({int(li.sum())} CCD / {int((~li).sum())} not):")
    say(f"       5' UTR   best surviving feature '{best_u}'   AUC {a_u:.4f}")
    say(f"       degron   D-box+ R..L..[LIVM] density          AUC {a_d:.4f}   (loop 121)")
    say(f"       geometry dual localisation                    AUC {a_m:.4f}   (loop 121 post-hoc)")
    gates["U5"] = bool(abs(a_u - 0.5) > abs(a_d - 0.5))
    say(f"     U5 {'PASS' if gates['U5'] else 'FAIL'} -- the 5' UTR "
        f"{'beats' if gates['U5'] else 'does NOT beat'} the degron feature")
    say(f"     geometry is reported, NOT gated: it is post-hoc and comes from the same images as "
        f"the CCD call, and gating on it would launder that into a result")
    say()

    # ---------------------------------------------------------------- U6
    say("U6 THE FILTER BOUNDS EVERY MECHANISM EQUALLY")
    st = CA.state_vector(D)
    ksp, Mbar, bbar = st["k_sp"], st["M"], st["k_loss_prot"]
    r0, m0 = CA.integrate_tl(ksp, Mbar, bbar, T_CYCLE, beta=0.0, ncyc=3, nstep=100)
    ss = ksp * Mbar / bbar
    dev = float(np.max(np.abs(m0 - ss) / ss))
    say(f"     (a) beta = 0: max relative deviation {dev:.3e}, residual {r0.max():.3e}")
    rt, _ = CA.integrate_tl(ksp, Mbar, bbar, T_CYCLE, beta=1.0)
    rd, _ = CA.integrate_deg(ksp, Mbar, bbar, T_CYCLE, beta=1.0)
    w = 2.0 * np.pi / T_CYCLE
    ana = 1.0 / np.sqrt(1.0 + (w / bbar) ** 2)
    agree = float(np.median(np.abs(rt - rd) / np.maximum(rt, 1e-12)))
    say(f"     (b) translation-driven median amplitude {np.median(rt):.4f}; "
        f"degradation-driven {np.median(rd):.4f}; analytic gain(bbar,T) {np.median(ana):.4f}")
    say(f"         median per-gene disagreement {agree:.4%}   gate < {U6_AGREE:.0%}")
    say(f"         they are the SAME first-order filter driven from opposite sides")
    lo, hi = 0.01, 5000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 / np.sqrt(1.0 + (w / (LN2 / mid + MU)) ** 2) > AMP_MIN:
            lo = mid
        else:
            hi = mid
    thr = 0.5 * (lo + hi)
    S = D["schwan"]
    hl = {g: S[g]["prot_hl_h"] for g in S if S[g].get("prot_hl_h")}
    ccd_hl = [hl[g] for g in called if cp[g] == "Yes" and g in hl]
    above = float(np.mean(np.array(ccd_hl) >= thr))
    say(f"     (c) no mechanism inside dP/dt = k_sp*M - b*P can exceed gain(bbar,T), whichever")
    say(f"         term it drives. At {AMP_MIN:.0%} amplitude that needs a half-life under "
        f"{thr:.1f} h.")
    say(f"         {above:.1%} of the {len(ccd_hl)} measured CCD proteins with a half-life are "
        f"ABOVE it.")
    say(f"         For those, the answer is not which term -- it is that the EQUATION is wrong, or")
    say(f"         that the measurement is not measuring copy number.")
    gates["U6"] = bool(dev < U6_TOL and r0.max() < U6_TOL and agree < U6_AGREE)
    say(f"     U6 {'PASS' if gates['U6'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- verdict
    say("=" * 100)
    for k in ("U1", "U2", "U3", "U4", "U5", "U6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)
    say()

    # ------------------------------------------------------------- AFTER THE FACT
    # ADDED AFTER THE RUN, GATED ON NOTHING. Four loops have now eliminated every mechanism this
    # equation contains, and U6 says why the elimination had to happen: they are all the same
    # filter. But "the measurement is not measuring copy number" cannot be the whole story either,
    # and there is a specific reason, so the population gets split rather than dismissed.
    say("AFTER THE FACT -- splitting the population instead of dismissing it")
    say()
    say("  (i) THE GEOMETRY HYPOTHESIS CANNOT BE THE WHOLE ANSWER, and loop 119 is why.")
    say("      Protein HALF-LIFE predicts the CCD call at AUC 0.5999 -- measured by SILAC mass")
    say("      spectrometry in mouse fibroblasts, an instrument with no camera in it and no")
    say("      connection to the imaging that made the call. A pure localisation artefact has no")
    say("      reason to track turnover in a different species by a different method. So genuine")
    say("      abundance change is part of this population.")
    say()
    say("  (ii) SO SPLIT IT AT THE BOUND. Above the half-life threshold no mechanism in the")
    say(f"      equation can reach {AMP_MIN:.0%}; below it, they all can.")
    fast = [g for g in called if cp[g] == "Yes" and g in hl and hl[g] < thr]
    slow = [g for g in called if cp[g] == "Yes" and g in hl and hl[g] >= thr]
    ctrl = [g for g in called if cp[g] == "No" and g in hl]
    say(f"      CCD, half-life < {thr:.1f} h  (explainable): {len(fast)}")
    say(f"      CCD, half-life >= {thr:.1f} h (NOT explainable): {len(slow)}")
    say(f"      imaged controls with a half-life: {len(ctrl)}")

    def frac(gs, f):
        v = [f(g) for g in gs if f(g) is not None]
        return float(np.mean(v)) if v else float("nan"), len(v)

    def dbox(g):
        return 100.0 * len(dbx.findall(prot[g])) / max(len(prot[g]), 1) if g in prot else None

    def dual(g):
        if g not in locn:
            return None
        return 1.0 if len([p for p in (locn[g][0] + "," + locn[g][1]).split(",")
                           if p.strip()]) > 1 else 0.0
    split = {}
    for nm2, gs in (("CCD fast (explainable)", fast), ("CCD slow (NOT explainable)", slow),
                    ("imaged controls", ctrl)):
        d, nd = frac(gs, dbox)
        m, nm3 = frac(gs, dual)
        split[nm2] = {"dbox": d, "n_dbox": nd, "dual": m, "n_dual": nm3, "n": len(gs)}
        say(f"        {nm2:<28} D-box+ {d:.4f} (n={nd})   dual-localised {m:.1%} (n={nm3})")
    df, pf = perm_p([dbox(g) for g in fast if dbox(g) is not None],
                    [dbox(g) for g in ctrl if dbox(g) is not None], rng)
    ds, ps = perm_p([dbox(g) for g in slow if dbox(g) is not None],
                    [dbox(g) for g in ctrl if dbox(g) is not None], rng)
    mf, pmf = perm_p([dual(g) for g in fast if dual(g) is not None],
                     [dual(g) for g in ctrl if dual(g) is not None], rng)
    ms, pms = perm_p([dual(g) for g in slow if dual(g) is not None],
                     [dual(g) for g in ctrl if dual(g) is not None], rng)
    say(f"      vs the controls:")
    say(f"        D-box+          fast {df:+.4f} p {pf:.4f}    slow {ds:+.4f} p {ps:.4f}")
    say(f"        dual-localised  fast {mf:+.1%} p {pmf:.4f}    slow {ms:+.1%} p {pms:.4f}")
    say()
    say("  (iii) WHAT THE WHOLE ARC ADDS UP TO, in one sentence per candidate:")
    say("        M(t) transcription       ELIMINATED  loop 119, p = 2e-67 in the wrong direction")
    say("        the regulatory wiring    ELIMINATED  loop 120, shuffled signs score higher")
    say("        b(t) timed destruction   PARTIAL     loop 121, signal on the already-explained")
    say("                                             genes; and bounded by the same filter")
    say("        k_sp(t) translation      NO SIGNAL   this loop, U3/U4/U5 all fail")
    say("        the equation itself      IMPLICATED  U6: all of the above share one bound, and")
    say(f"                                             {above:.0%} of CCD proteins sit above it")
    say("        measurement geometry     LEADING     AUC 0.5991, the largest number in four")
    say("                                             loops -- and same-instrument, so unproven")
    say()
    posthoc = {"threshold_h": thr, "split": split,
               "dbox_fast_vs_ctrl": [df, pf], "dbox_slow_vs_ctrl": [ds, ps],
               "dual_fast_vs_ctrl": [mf, pmf], "dual_slow_vs_ctrl": [ms, pms],
               "note": "added after the run, gated on nothing; the half-life association from an "
                       "independent instrument (loop 119, AUC 0.5999) rules out a pure imaging "
                       "artefact, so the population is split at the bound rather than dismissed"}

    man = RM.manifest(inputs=[CDNA, CDS, MANE, HPA, LR.CELL],
                      available=len(cp), used=len(called), selection="filtered", seed=SEED,
                      controls=["ribosomal-protein 5'TOP as a pre-measured positive control",
                                "Kozak -3 purine as an unseen CDS-boundary control",
                                "publication count per feature, excluding the ones that fail",
                                "776 imaged non-CCD proteins as the assay-matched control",
                                "composition-preserving 5' UTR shuffle for the motif features",
                                "the degron and geometry features recomputed on the same genes"],
                      note="5' UTR obtained by subtracting the CDS from the cDNA of the MANE "
                           "Select transcript; exact, not predicted, and the isoform is NCBI's "
                           "choice rather than this loop's")
    RM.report(man, emit=say)
    json.dump({"test": "loop_utr5", "manifest": man, "gates": gates,
               "build": {"cdna": len(cdna), "cds": len(cds), "mane_select": len(sel),
                         "utr_recovered": len(utr),
                         "median_len": float(np.median([len(v) for v in utr.values()]))},
               "n": {"called_with_utr": len(called), "yes": int(lab.sum()),
                     "no": int((~lab).sum()), "groups": {k: len(v) for k, v in grp.items()}},
               "u1": {"top_ribosomal": f_rp, "top_other": f_ot, "ratio": ratio,
                      "kozak_purine_all": purall, "kozak_purine_called": pur},
               "u2": {"rho_vs_pubs": rhos, "survivors": survivors},
               "u3": u3, "u4": u4,
               "u5": {"best_utr_feature": best_u, "auc_utr": a_u, "auc_degron": a_d,
                      "auc_geometry": a_m, "n": len(inter)},
               "posthoc": posthoc,
               "u6": {"max_rel_deviation": dev, "median_disagreement": agree,
                      "median_amp_translation": float(np.median(rt)),
                      "median_amp_degradation": float(np.median(rd)),
                      "threshold_h": thr, "ccd_above_threshold": above, "n_ccd_hl": len(ccd_hl)},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_utr5.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_utr5.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
