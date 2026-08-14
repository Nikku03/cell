"""LOOP 121 -- WHERE THE OSCILLATION ACTUALLY COMES FROM, once transcription and the TF network are out.

THE QUESTION, NARROWED BY TWO LOOPS OF ELIMINATION. loop 119: 362 proteins oscillate over the cell
cycle whose mRNA does not, against 38 the other way, p = 2e-67. loop 120: the regulatory network
cannot pick out even the 126 that ARE transcriptional -- real edge signs score 0.5465 against
shuffled signs at 0.5494, and regulator count alone beats every network score. So the signal is not
in transcription, and it is not in the wiring above transcription.

The protein equation has exactly two other terms:

    dP/dt = k_sp * M - b * P

With M flat, a protein can only oscillate if k_sp oscillates or b oscillates. That is not a theory,
it is the whole of the remaining space, and this loop tests which half of it the data sits in.

    b(t)     REGULATED DEGRADATION. Timed destruction by APC/C and SCF, the canonical cell-cycle
             mechanism. It leaves a signature IN THE AMINO ACID SEQUENCE: a destruction box or a
             KEN box, which is what the ubiquitin ligase physically binds.
    k_sp(t)  REGULATED TRANSLATION. Also real, also cell-cycle-relevant -- and NOT TESTABLE HERE.
             Its signature lives in the 5' UTR, which this repository does not have for any gene.
             Stated as a gap rather than quietly folded into the alternative.

WHY A SEQUENCE MOTIF IS THE RIGHT INSTRUMENT, AND WHY IT IS THE FIRST ONE THIS SESSION THAT CANNOT
LOSE TO FAME. Publication count has beaten or matched the biology in seven loops. It does so because
every feature tried so far was an ANNOTATION -- edges, GO terms, phosphosites, complex membership --
and annotation is a record of what has been studied. A motif count divided by protein length is
computed from the amino acid sequence and from nothing else. UniProt does not sequence a protein
more thoroughly because it is famous. S1 gates on that rather than assuming it.

THE MOTIFS, and both are the physical binding site of a ubiquitin ligase adaptor:
    D-box    R..L        the destruction box core, bound by CDC20/CDH1 to recruit APC/C
    D-box+   R..L..[LIVM]        the extended form
    KEN-box  KEN         the second APC/C degron, CDH1-specific
    CDK site [ST]P and [ST]P.[KR]        included as the phosphorylation counter-hypothesis

R..L occurs 3.4 times in an average protein by chance, which is exactly why the composition-
preserving null below is not optional: an arginine-rich protein has more R..L for reasons that have
nothing to do with APC/C.

PREDECLARED, before any number:

  S1 THE SEQUENCE FEATURES ARE NOT FAME PROXIES                     THE PREREQUISITE.
       |Spearman(feature, publication count)| < 0.15 for every motif density, over every gene with
       a sequence. This is the claim that makes the loop worth running, so it is gated first and
       the loop reports itself broken if it fails. Densities, not counts, because protein LENGTH
       does correlate with study.
  S2 DEGRON DENSITY IS HIGHER IN CELL-CYCLE PROTEINS                THE MEASUREMENT.
       748 CCD proteins against the 776 imaged non-CCD controls, above TWO nulls, both required:
         (a) the CCD label shuffled across the called set, capability-checked;
         (b) EACH PROTEIN'S OWN SEQUENCE SHUFFLED, preserving amino acid composition exactly and
             destroying position. This is the null that separates a degron from an R-rich protein.
       Gate: real above both at z >= 2.
  S3 THE PROTEIN-ONLY GROUP IS THE ONE THAT CARRIES IT              THE DISCRIMINATING TEST.
       loop 119's 362 protein-only oscillators need a post-transcriptional mechanism BY
       CONSTRUCTION. The 38 transcript-only ones do not -- they already have a source. So degron
       density should be higher in the protein-only group than the transcript-only group. Gate:
       higher AND a permutation p < 0.05 on the difference. loop 120 passed two gates on margins
       inside their own noise; every gate here carries its significance test from the start.
  S4 THE HALF-LIFE THRESHOLD                                        THE ARITHMETIC, NOT NEW EVIDENCE.
       a loss rate swinging 0 to 2*bbar gives relative amplitude gain(bbar,T) = 1/sqrt(1+(w/bbar)^2)
       -- the SAME statistic loop 119's C3 already reported at AUC 0.5999, re-expressed. So this
       gate claims no new evidence and says so. What it adds is the number that matters for a
       model: the half-life above which timed destruction CANNOT produce a 20% swing, and the
       fraction of measured CCD proteins that sit below it.
  S5 FAME, AND THE ANNOTATION-BASED ALTERNATIVES                    THE CONTROL THAT KEEPS WINNING.
       every sequence feature and every annotation feature (publication count, phosphosite count
       from the model's ptm layer, GO ubiquitin-catabolism membership) scored on the same genes.
       Gate: the best SEQUENCE feature must beat publication count. If the annotation features win
       and pubs wins over them in turn, that is the same seven-loop story and it gets recorded
       again rather than dressed up.
  S6 THE MODEL RUNS WITH THE DEGRADATION TERM DRIVEN                THE WIRING.
       cell_assembled.integrate_deg holds M flat and puts the cycle in b(t). Gate, three parts:
         (a) beta = 0 reproduces k_sp*Mbar/bbar to < 1e-9 -- an addition, not a replacement;
         (b) it produces protein oscillation with FLAT mRNA, which loop 119 proved the existing
             equation cannot do at any parameter;
         (c) the fraction of the 748 measured CCD proteins reaching >= 20% amplitude exceeds the
             same fraction among the 776 controls.

WHAT A PASS WOULD AND WOULD NOT MEAN. Finding a degron signature says the mechanism is timed
destruction. It does not say WHEN each protein is destroyed, because a motif has no phase -- the
timing lives in the ligase's own activity, which is not in this repository. A pass moves the cell
cycle from "unexplained" to "explained by a mechanism the model does not yet contain but now has a
term for". That is smaller than a clock and larger than nothing.

-> outputs/loop_degron.json
"""
import gzip
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import csv
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
FASTA = LR.SC / "human_proteome.fasta.gz"
SEED = 12100
NPERM = 2000
NSEQ = 100                      # composition-preserving sequence shuffles
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H
T_CYCLE = 24.0
S1_RHO = 0.15
S2_Z = 2.0
S3_P = 0.05
S6_TOL = 1e-9
AMP_MIN = 0.20                  # the swing a protein must reach to be called oscillating

MOTIFS = {"D-box R..L": re.compile("R..L"),
          "D-box+ R..L..[LIVM]": re.compile("R..L..[LIVM]"),
          "KEN-box": re.compile("KEN"),
          "CDK [ST]P": re.compile("[ST]P"),
          "CDK [ST]P.[KR]": re.compile("[ST]P.[KR]")}
DEGRONS = ("D-box R..L", "D-box+ R..L..[LIVM]", "KEN-box")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
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


def load_fasta(path):
    seq, nm, buf = {}, None, []
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if nm and buf:
                    s = "".join(buf)
                    if len(s) > len(seq.get(nm, "")):
                        seq[nm] = s
                nm, buf = None, []
                for p in ln.split():
                    if p.startswith("GN="):
                        nm = p[3:]
                        break
            else:
                buf.append(ln.strip())
    if nm and buf:
        s = "".join(buf)
        if len(s) > len(seq.get(nm, "")):
            seq[nm] = s
    return seq


def densities(s):
    """Motif hits per 100 residues. Density, not count, so protein length cannot carry the signal."""
    n = max(len(s), 1)
    return {k: 100.0 * len(p.findall(s)) / n for k, p in MOTIFS.items()}


def perm_p(a, b, rng, n=NPERM):
    """Two-sided permutation p on a difference of means, with the observed difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    k = len(a)
    null = np.array([(lambda s: s[:k].mean() - s[k:].mean())(rng.permutation(pool))
                     for _ in range(n)])
    return float(obs), float(np.mean(np.abs(null) >= abs(obs))), null


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 121 -- where the oscillation comes from, once transcription and the network are out")
    say("=" * 100)
    say()

    D = CA.load()
    seq = load_fasta(FASTA)
    say(f"  proteome: {len(seq):,} sequences with a gene name")

    with open(HPA, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        iG, iCP, iCT = h.index("Gene"), h.index("CCD Protein"), h.index("CCD Transcript")
        rows = [(x[iG], x[iCP], x[iCT]) for x in r]
    cp = {g: v for g, v, _ in rows if v in ("Yes", "No")}
    ct = {g: v for g, _, v in rows if v in ("Yes", "No")}

    called = sorted(g for g in cp if g in seq)
    yes = [g for g in called if cp[g] == "Yes"]
    no = [g for g in called if cp[g] == "No"]
    say(f"  CCD called AND sequenced: {len(called):,}  ({len(yes)} Yes / {len(no)} No)")
    dens = {g: densities(seq[g]) for g in called}
    say(f"  motifs: " + ", ".join(MOTIFS))
    say()

    gates = {}

    # ---------------------------------------------------------------- S1
    say("S1 THE SEQUENCE FEATURES ARE NOT FAME PROXIES")
    pubs = D["pubs"]
    allg = [g for g in seq if g in pubs]
    pv = np.array([pubs[g] for g in allg])
    lens = np.array([len(seq[g]) for g in allg])
    say(f"     over {len(allg):,} genes with a sequence and a publication count")
    say(f"     protein LENGTH vs pubs:  rho {spearman(lens, pv):+.4f}   "
        f"(this is why the features are densities)")
    rhos = {}
    for k in MOTIFS:
        dv = np.array([100.0 * len(MOTIFS[k].findall(seq[g])) / max(len(seq[g]), 1) for g in allg])
        rhos[k] = spearman(dv, pv)
        say(f"     {k:>22} density vs pubs:  rho {rhos[k]:+.4f}")
    gates["S1"] = bool(all(abs(v) < S1_RHO for v in rhos.values()))
    say(f"     S1 {'PASS' if gates['S1'] else 'FAIL'} -- every motif density is under "
        f"|rho| {S1_RHO}: a sequence count cannot be a record of study effort")
    say()

    # ---------------------------------------------------------------- S2
    say("S2 DEGRON DENSITY IS HIGHER IN CELL-CYCLE PROTEINS")
    s2 = {}
    lab = np.array([cp[g] == "Yes" for g in called])
    for k in DEGRONS:
        v = np.array([dens[g][k] for g in called])
        real = float(v[lab].mean() - v[~lab].mean())
        # null (a): shuffle which proteins are CCD
        na, capa = [], None
        for i in range(NPERM):
            sh = rng.permutation(lab)
            if i == 0:
                capa = GG.null_can_move(lab.astype(int), sh.astype(int))
            na.append(float(v[sh].mean() - v[~sh].mean()))
        sa = GG.survival(real, na, z_min=S2_Z)
        s2[k] = {"real": real, "label_null": sa, "label_capability": capa}
        say(f"     {k}")
        say(f"       CCD {v[lab].mean():.4f} / 100 aa   non-CCD {v[~lab].mean():.4f}   "
            f"difference {real:+.4f}")
        GG.report("       vs label shuffle", sa, emit=say)
    # null (b): shuffle each protein's own residues, preserving composition exactly
    say(f"     composition-preserving sequence shuffle, {NSEQ} rounds over {len(called):,} proteins")
    arrs = {g: np.frombuffer(seq[g].encode("ascii", "ignore"), dtype="S1") for g in called}
    nb = {k: [] for k in DEGRONS}
    capb = None
    for i in range(NSEQ):
        sd = {}
        for g in called:
            a = arrs[g]
            s = b"".join(rng.permutation(a).tolist()).decode("ascii")
            if i == 0 and g == called[0]:
                capb = GG.null_can_move(list(seq[g]), list(s))
            n = max(len(s), 1)
            sd[g] = {k: 100.0 * len(MOTIFS[k].findall(s)) / n for k in DEGRONS}
        for k in DEGRONS:
            v = np.array([sd[g][k] for g in called])
            nb[k].append(float(v[lab].mean() - v[~lab].mean()))
    ok2 = True
    for k in DEGRONS:
        sb = GG.survival(s2[k]["real"], nb[k], z_min=S2_Z)
        s2[k]["seq_null"] = sb
        GG.report(f"     {k} vs composition shuffle", sb, emit=say)
        if not (s2[k]["label_null"].get("z", 0) >= S2_Z and sb.get("z", 0) >= S2_Z):
            ok2 = False
    say(f"     the composition shuffle changes {capb['changed']:.1%} of residue positions "
        f"(achievable {capb['achievable']:.1%}) -- {'capable' if capb['capable'] else 'INERT'}")
    gates["S2"] = bool(ok2 and capb["capable"] and all(s2[k]["label_capability"]["capable"]
                                                       for k in DEGRONS))
    say(f"     S2 {'PASS' if gates['S2'] else 'FAIL'} -- a degron signature "
        f"{'survives both nulls' if gates['S2'] else 'does NOT survive both nulls'}")
    say()

    # ---------------------------------------------------------------- S3
    say("S3 THE PROTEIN-ONLY GROUP IS THE ONE THAT CARRIES IT")
    both_c = [g for g in called if g in ct]
    grp = {"both": [g for g in both_c if cp[g] == "Yes" and ct[g] == "Yes"],
           "protein only": [g for g in both_c if cp[g] == "Yes" and ct[g] == "No"],
           "transcript only": [g for g in both_c if cp[g] == "No" and ct[g] == "Yes"],
           "neither": [g for g in both_c if cp[g] == "No" and ct[g] == "No"]}
    s3 = {}
    ok3 = True
    for k in DEGRONS:
        say(f"     {k}")
        for gn, gs in grp.items():
            say(f"       {gn:>16}  {np.mean([dens[g][k] for g in gs]):.4f} / 100 aa  (n={len(gs)})")
        a = np.array([dens[g][k] for g in grp["protein only"]])
        b = np.array([dens[g][k] for g in grp["transcript only"]])
        obs, p, _ = perm_p(a, b, rng)
        s3[k] = {"protein_only": float(a.mean()), "transcript_only": float(b.mean()),
                 "difference": obs, "p": p,
                 "n_protein_only": len(a), "n_transcript_only": len(b)}
        say(f"       protein-only minus transcript-only: {obs:+.4f}, permutation p = {p:.4f}")
        if not (obs > 0 and p < S3_P):
            ok3 = False
    gates["S3"] = bool(ok3)
    say(f"     S3 {'PASS' if gates['S3'] else 'FAIL'} -- the post-transcriptional group "
        f"{'carries the degron signal' if gates['S3'] else 'is NOT distinguishable from the transcriptional one'}")
    say()

    # ---------------------------------------------------------------- S4
    say("S4 THE HALF-LIFE THRESHOLD  (arithmetic, NOT new evidence -- same statistic as loop 119 C3)")
    S = D["schwan"]
    hl = {g: S[g]["prot_hl_h"] for g in S if S[g].get("prot_hl_h")}
    w = 2.0 * np.pi / T_CYCLE

    def g_of(t):
        b = LN2 / t + MU
        return 1.0 / np.sqrt(1.0 + (w / b) ** 2)
    lo, hi = 0.01, 5000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g_of(mid) > AMP_MIN:
            lo = mid
        else:
            hi = mid
    thr = 0.5 * (lo + hi)
    say(f"     a loss rate swinging 0 to 2*bbar reaches {AMP_MIN:.0%} relative amplitude only for")
    say(f"     proteins with a half-life below {thr:.1f} h (at a {T_CYCLE:.0f} h cycle)")
    fy = [hl[g] for g in yes if g in hl]
    fn = [hl[g] for g in no if g in hl]
    py_ = float(np.mean(np.array(fy) < thr))
    pn_ = float(np.mean(np.array(fn) < thr))
    obs4, p4, _ = perm_p((np.array(fy) < thr).astype(float),
                         (np.array(fn) < thr).astype(float), rng)
    say(f"     below it: CCD {py_:.1%} of {len(fy)}   non-CCD {pn_:.1%} of {len(fn)}   "
        f"difference {obs4:+.1%}, permutation p = {p4:.4f}")
    say(f"     and {1 - py_:.1%} of measured CCD proteins are ABOVE the threshold -- timed "
        f"destruction alone cannot make those oscillate, so it is not the whole answer either")
    gates["S4"] = bool(obs4 > 0 and p4 < 0.05)
    say(f"     S4 {'PASS' if gates['S4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- S5
    say("S5 FAME, AND THE ANNOTATION-BASED ALTERNATIVES")
    idx = {n: i for i, n in enumerate(D["names"])}
    ptm = D["model"].get("ptm") or {}
    feats = {}
    for k in MOTIFS:
        feats[f"SEQ  {k}"] = np.array([dens[g][k] for g in called])
    feats["ANN  publication count"] = np.array([pubs.get(g, 0.0) for g in called])
    feats["ANN  phosphosite count"] = np.array(
        [float((ptm.get(str(idx.get(g, -1))) or {}).get("n", 0)) for g in called])
    go = D["model"].get("go") or {}
    ubq = np.array([float(any("ubiquitin" in t.lower() or "proteasom" in t.lower()
                              for t in (go.get(g) or {}).get("P", []))) for g in called])
    feats["ANN  GO ubiquitin/proteasome"] = ubq
    best_seq, best_ann = ("", 0.0), ("", 0.0)
    s5 = {}
    for k, v in sorted(feats.items()):
        a = auc(v[lab], v[~lab])
        s5[k] = a
        say(f"     {k:<32} AUC {a:.4f}   |AUC-0.5| {abs(a - 0.5):.4f}")
        if k.startswith("SEQ") and abs(a - 0.5) > best_seq[1]:
            best_seq = (k, abs(a - 0.5))
        if k.startswith("ANN") and abs(a - 0.5) > best_ann[1]:
            best_ann = (k, abs(a - 0.5))
    fame = abs(s5["ANN  publication count"] - 0.5)
    say(f"     best sequence feature: {best_seq[0]} at {best_seq[1]:.4f}")
    say(f"     best annotation feature: {best_ann[0]} at {best_ann[1]:.4f}")
    say(f"     publication count: {fame:.4f}")
    gates["S5"] = bool(best_seq[1] > fame)
    say(f"     S5 {'PASS' if gates['S5'] else 'FAIL'} -- sequence "
        f"{'beats fame' if gates['S5'] else 'DOES NOT beat fame, for the eighth time'}")
    say()

    # ---------------------------------------------------------------- S6
    say("S6 THE MODEL RUNS WITH THE DEGRADATION TERM DRIVEN")
    st = CA.state_vector(D)
    ksp, Mbar, bbar = st["k_sp"], st["M"], st["k_loss_prot"]
    rel0, mean0 = CA.integrate_deg(ksp, Mbar, bbar, T_CYCLE, beta=0.0, ncyc=3, nstep=100)
    ss = ksp * Mbar / bbar
    dv = float(np.max(np.abs(mean0 - ss) / ss))
    say(f"     beta = 0: max relative deviation {dv:.3e}, residual oscillation {rel0.max():.3e}")
    rel1, _ = CA.integrate_deg(ksp, Mbar, bbar, T_CYCLE, beta=1.0)
    say(f"     beta = 1, mRNA held FLAT: {np.mean(rel1 >= AMP_MIN):.1%} of {len(rel1):,} genes "
        f"reach {AMP_MIN:.0%} amplitude, median {np.median(rel1):.4f}, max {rel1.max():.4f}")
    say(f"     loop 119's equation with flat mRNA gives EXACTLY zero protein oscillation, at "
        f"every parameter -- this is the term that was missing")
    sg = {g: i for i, g in enumerate(st["genes"])}
    iy = [sg[g] for g in yes if g in sg]
    inn = [sg[g] for g in no if g in sg]
    ry = float(np.mean(rel1[iy] >= AMP_MIN)) if iy else float("nan")
    rn = float(np.mean(rel1[inn] >= AMP_MIN)) if inn else float("nan")
    obs6, p6, _ = perm_p((rel1[iy] >= AMP_MIN).astype(float),
                         (rel1[inn] >= AMP_MIN).astype(float), rng)
    say(f"     of the measured calls with a state: CCD {ry:.1%} of {len(iy)} reach it, "
        f"non-CCD {rn:.1%} of {len(inn)}   difference {obs6:+.1%}, p = {p6:.4f}")
    gates["S6"] = bool(dv < S6_TOL and rel0.max() < S6_TOL
                       and np.mean(rel1 >= AMP_MIN) > 0 and obs6 > 0 and p6 < 0.05)
    say(f"     S6 {'PASS' if gates['S6'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- verdict
    say("=" * 100)
    for k in ("S1", "S2", "S3", "S4", "S5", "S6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[FASTA, HPA, LR.CELL, LR.SC / "_schwan2011.json"],
                      available=len(cp), used=len(called), selection="filtered", seed=SEED,
                      controls=["776 imaged non-CCD proteins as the assay-matched control",
                                "CCD label shuffled, capability-checked",
                                "EACH PROTEIN'S SEQUENCE SHUFFLED, composition preserved exactly",
                                "loop 119's four-way table used as internal positive and negative "
                                "classes",
                                "publication count, phosphosite count and GO membership as the "
                                "annotation-based alternatives",
                                "the degradation-driven integrator gated on exact reduction at "
                                "beta = 0"],
                      note="regulated translation k_sp(t) is the untested half of the remaining "
                           "space: its signature is in the 5' UTR, which this repository has for "
                           "no gene")
    RM.report(man, emit=say)
    json.dump({"test": "loop_degron", "manifest": man, "gates": gates,
               "n": {"called_and_sequenced": len(called), "yes": len(yes), "no": len(no),
                     "groups": {k: len(v) for k, v in grp.items()}},
               "s1": {"rho_vs_pubs": rhos, "rho_length_vs_pubs": spearman(lens, pv)},
               "s2": {k: {kk: vv for kk, vv in v.items()} for k, v in s2.items()},
               "s3": s3,
               "s4": {"threshold_h": thr, "ccd_below": py_, "nonccd_below": pn_,
                      "difference": obs4, "p": p4, "amp_min": AMP_MIN},
               "s5": s5, "s5_best_sequence": best_seq[0], "s5_best_annotation": best_ann[0],
               "s6": {"max_rel_deviation": dv, "residual": float(rel0.max()),
                      "frac_reaching": float(np.mean(rel1 >= AMP_MIN)),
                      "median_amplitude": float(np.median(rel1)),
                      "ccd_reaching": ry, "nonccd_reaching": rn, "difference": obs6, "p": p6},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_degron.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_degron.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
