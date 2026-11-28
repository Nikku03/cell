"""The fair version of rawchip.py: sequence-specific transcription factors, chosen before looking.

WHY. rawchip.py's rule ("most movers in their own knockdown") chose mostly general machinery --
cohesin, basal transcription, MAX -- bound at thousands of promoters, where binding should be least
specific. Recorded there as a flaw. This run asks the same question of the factors where binding
SHOULD mean regulation.

SELECTION, FIXED NOW (no outcome looked at): every factor that is
  (1) a sequence-specific DNA-binding TF in Lambert et al. 2018, "The Human Transcription Factors"
      (Cell; humantfs.ccbr.utoronto.ca v1.01, "Is TF?" = Yes),
  (2) the target of a released ENCODE K562 TF ChIP-seq experiment with GRCh38 IDR peaks
      (rawchip.fetch_peaks: first experiment by accession, conservative IDR else IDR),
  (3) knocked down in K562 genome-wide Perturb-seq with >= 20 movers (a measurable phenotype),
  (4) bound at >= 50 promoters in that peak file (else the file is uninformative; counted).
If more than 40 qualify, the first 40 by SHA-256 of the symbol.

GATES: rawchip.py's T1-T4, unchanged (permutation null over factor <-> binding derangements among
the chosen factors; per-factor Fisher enrichment; Spearman dose-response with a sign test across
factors; OmniPath sign control, verdict only with >= 20 pairs). DISCOVERY only if T1 passes.
"""

from __future__ import annotations
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import time
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_rawchip2.txt")
ART = R("outputs", "rawchip2.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("rawchip", os.path.join(HERE, "rawchip.py"))
rc = importlib.util.module_from_spec(_s); _s.loader.exec_module(rc)
cd = rc.cd
LAMBERT = os.path.join(HERE, "_cache", "lambert", "DatabaseExtract_v_1.01.csv")
CAP, MIN_MOV, MIN_BOUND, NPERM = 40, 20, 50, 1000


def encode_k562_targets():
    req = urllib.request.Request(f"{rc.ENC}/search/?type=Experiment&assay_title=TF+ChIP-seq&biosample_ontology.term_name=K562"
                                 f"&status=released&format=json&limit=all&field=target.label",
                                 headers={"Accept": "application/json"})
    g = json.load(urllib.request.urlopen(req, timeout=300))["@graph"]
    return {e["target"]["label"] for e in g if isinstance(e.get("target"), dict)}


def main():
    from scipy.stats import fisher_exact, spearmanr
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    rng = np.random.default_rng(20261008)
    X, kos, genes, ep, fold = cd.load_gwps()
    gidx = {g: j for j, g in enumerate(genes)}; kidx = {k: i for i, k in enumerate(kos)}
    zmov = 3.0 / 6.85
    lam = {r["HGNC symbol"] for r in csv.DictReader(open(LAMBERT)) if r["Is TF?"] == "Yes"}
    enc = encode_k562_targets()
    nmov = {k: int(np.nansum(np.abs(X[kidx[k]]) >= zmov)) for k in kos}
    pool = sorted([t for t in lam & enc & set(kos) if nmov[t] >= MIN_MOV], key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    P_(RULE); P_("SEQUENCE-SPECIFIC FACTORS, CHOSEN BEFORE LOOKING: DOES PROMOTER BINDING PREDICT RESPONSE?"); P_(RULE)
    P_(f"  Lambert TFs {len(lam):,}; ENCODE K562 ChIP targets {len(enc)}; knocked down {len(kos):,}; "
       f"all three with >= {MIN_MOV} movers: {len(pool)}")
    tss = rc.tss_table()
    chosen, peaks, meta, dropped = [], {}, [], collections.Counter()
    for tf in pool:
        if len(chosen) >= CAP:
            break
        m = rc.fetch_peaks(tf)
        if m is None:
            dropped["no GRCh38 IDR peak file"] += 1; continue
        b = rc.bound_genes(m["path"], tss)
        if len(b) < MIN_BOUND:
            dropped[f"< {MIN_BOUND} promoters bound"] += 1; continue
        chosen.append(tf); peaks[tf] = b; meta.append(m)
    P_(f"  chosen {len(chosen)}; dropped {dict(dropped)}")
    P_("  " + ", ".join(f"{t} ({nmov[t]} movers, {len(peaks[t]):,} promoters)" for t in chosen))
    omni = cd.load_omnipath()
    omni_tx = {(s, t): sg for s, t, sg, ty, _ in omni if ty == "transcriptional"}
    omni_all = {(s, t) for s, t, _, _, _ in omni}
    D = json.load(gzip.open(cd.dc.ENCY)); bn = [r["name"] for r in D["genes"]]
    bank_pairs = {(s, t) for s, t, _, _ in cd.cb.facts_table(D, bn)[0]}
    meas = set(genes)
    mv = {t: {g for g in genes if g != t and np.isfinite(X[kidx[t], gidx[g]]) and abs(X[kidx[t], gidx[g]]) >= zmov} for t in chosen}
    bset = {t: set(peaks[t]) & meas for t in chosen}
    real = sum(len(mv[t] & bset[t]) for t in chosen)
    null = []
    for _ in range(NPERM):
        perm = list(chosen)
        while True:
            rng.shuffle(perm)
            if all(a != b for a, b in zip(chosen, perm)):
                break
        null.append(sum(len(mv[t] & (bset[o] - {t})) for t, o in zip(chosen, perm)))
    p1 = (1 + sum(n >= real for n in null)) / (NPERM + 1)
    v1 = "SHARP ENOUGH" if p1 < 0.05 else "NOT SHARP ENOUGH"
    P_("\n" + RULE); P_("RESULTS"); P_(RULE)
    P_(f"  T1 triangulated pairs {real:,} vs null {np.mean(null):,.1f} +/- {np.std(null):.1f}; permutation p {p1:.4f} -> {v1}")
    ors, sig, rows_ = [], 0, []
    for t in chosen:
        a = len(mv[t] & bset[t]); b = len(bset[t] - mv[t] - {t}); c = len(mv[t] - bset[t]); d = len(meas) - a - b - c - 1
        orr, p = fisher_exact([[a, b], [c, d]])
        ors.append(orr); sig += int(orr > 1 and p < 0.05); rows_.append((t, len(mv[t]), len(bset[t]), a, orr, p))
    for t, m_, b_, a, orr, p in sorted(rows_, key=lambda r: r[5]):
        P_(f"     {t:<8} movers {m_:>5}  bound {b_:>6}  both {a:>4}  odds ratio {orr:>6.2f}  p {p:.2g}")
    w2 = sum(o > 1 for o in ors)
    P_(f"  T2 factors with significant enrichment (OR > 1, p < 0.05): {sig}/{len(chosen)}; OR > 1 in {w2}/{len(chosen)}; median OR {np.median(ors):.2f}")
    rhos = []
    for t in chosen:
        gs = [g for g in peaks[t] if g in gidx and g != t and np.isfinite(X[kidx[t], gidx[g]])]
        if len(gs) >= 20:
            rhos.append(spearmanr([peaks[t][g] for g in gs], [abs(X[kidx[t], gidx[g]]) for g in gs]).correlation)
    w = sum(x > 0 for x in rhos); n_ = len(rhos)
    p3 = min(1.0, 2 * sum(math.comb(n_, i) for i in range(0, min(w, n_ - w) + 1)) / 2 ** n_) if n_ else 1.0
    v3 = "STRONGER BINDING, BIGGER EFFECT" if p3 < 0.05 and w > n_ - w else "NO DOSE-RESPONSE"
    P_(f"  T3 dose-response: positive in {w}/{n_} factors (median rho {np.median(rhos):+.3f}), p {p3:.3g} -> {v3}")
    tri = [(t, g, float(X[kidx[t], gidx[g]])) for t in chosen for g in mv[t] & bset[t]]
    agr = [(-np.sign(z) == omni_tx[(t, g)]) for t, g, z in tri if (t, g) in omni_tx]
    if len(agr) >= 20:
        k_ = int(sum(agr)); pa = sum(math.comb(len(agr), i) for i in range(k_, len(agr) + 1)) / 2 ** len(agr)
        P_(f"  T4 sign control: agreement with OmniPath {k_}/{len(agr)} = {k_ / len(agr):.3f}, p {pa:.3g}")
    else:
        P_(f"  T4 sign control: {int(sum(agr))}/{len(agr)} (fewer than 20: no verdict)")
    res = dict(chosen=chosen, real=real, null_mean=float(np.mean(null)), p=p1, verdict=v1, sig_factors=sig,
               or_above_1=w2, median_or=float(np.median(ors)), dose_pos=w, dose_n=n_, dose_p=p3, dose_verdict=v3,
               sign_agree=int(sum(agr)), sign_n=len(agr))
    if p1 < 0.05:
        novel = sorted([(t, g, z) for t, g, z in tri if (t, g) not in omni_all and (t, g) not in bank_pairs], key=lambda x: -abs(x[2]))
        fdr = float(np.mean(null)) / max(real, 1)
        P_(f"  DISCOVERY: {len(novel)} candidate direct links in neither OmniPath nor the bank; estimated FDR {fdr:.2f}")
        for t, g, z in novel[:30]:
            P_(f"     {t:>8} {'activates' if z < 0 else 'represses'} {g:<10} z-equiv {z * 6.85:+.1f}, peak strength {peaks[t][g]:.0f}")
        res.update(novel=len(novel), fdr=fdr, novel_list=[(t, g, round(z * 6.85, 2), round(peaks[t][g], 1)) for t, g, z in novel])
    else:
        P_("  DISCOVERY: none listed -- T1 did not pass.")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"files": meta, "results": res}, open(ART, "w"), indent=1, default=float)
    P_(f"\n  artifact: outputs/rawchip2.json   runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
