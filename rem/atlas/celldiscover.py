"""Give the memory bank the selected data and ask what it can discover -- every discovery method
validated on known biology first, every discovery list priced against a null.

THE DATA ADDED (all fetched this session; raw files in the gitignored cache, only derived results
committed; licences recorded):
  K562 GENOME-WIDE Perturb-seq  Replogle et al. 2022, figshare 20029387, CC BY 4.0
        K562_gwps_normalized_bulk_01.h5ad: 11,258 CRISPRi perturbations x 8,248 genes (z-scores)
  OmniPath (commercial-licence subset, directed; includes SIGNOR, CollecTRI, DoRothEA, ...)
        217,587 interactions; sign from consensus_stimulation / consensus_inhibition
  ENCODE TF ChIP-seq, K562 sets (via the Enrichr ENCODE_TF_ChIP-seq_2015 library; ENCODE data are
        unrestricted; the Enrichr packaging's terms are academic -- flagged)
  DepMap 24Q4 CRISPRGeneEffect, figshare 27993248, CC BY 4.0: 1,178 models x 17,916 genes
  HGNC gene groups (repository copy) -- the answer key for gene-function validation
A knockdown's per-gene profile is the perturbation with the strongest on-target knockdown (lowest
fold_expr). Sign of a measured effect X -> Y is -sign(value) (CRISPRi lowers X).
SCALE AMENDMENT, BEFORE THE FIRST RUN (found by a structure check, no discovery outcome computed):
the gwps "normalized bulk" values are pseudobulk means of single-cell z-scores, ~7x smaller than the
essential-screen z-scores used in cellbank2.py (all-entry median |value| 0.059; own-gene median
-0.465). The conversion is MEASURED at run time on entries shared with the essential-screen file
(median |z_ess| / |value| over entries with |z_ess| > 3), and every threshold is the cellbank2
threshold divided by it: MOVER |value| >= 3 / scale; G0 own-gene median <= -1 / scale. Gate: the
scale must lie in [3, 15].
SECOND AMENDMENT, AFTER THE FIRST RUN (its output is committed in history): the matrix holds 6,300
non-finite entries (73 readout genes with zero control variance). The first run counted infinite
values as movers (they topped the D1 list) and they turned the D4 knockdown-profile similarity
into NaN (AUROC nan -> D4 FAIL). Fix: non-finite entries are MISSING -- never movers, never scored;
the 73 affected readout columns are dropped from profile similarities. Every question re-runs
under its unchanged gates. Values are printed as z-equivalents (value x scale).

=================================================================================================
GATES, PREDECLARED
=================================================================================================
G0 DATA, BLOCKING. Own-gene median <= -1/scale over knockdowns whose gene is measured; K562 present in
   DepMap; >= 20 ENCODE K562 TFs that were also knocked down; >= 10,000 signed OmniPath edges.

D1 NEW DIRECT REGULATORY LINKS IN K562 (two independent assays + absent from every database).
   For TF X (ENCODE K562 ChIP AND knocked down): TRIANGULATED pairs = Y moved (|z| >= 3) AND Y in
   X's K562 ChIP set. Sign = -sign(z).
   D1a POSITIVE CONTROL, BLOCKING FOR CLAIMS: among triangulated pairs that OmniPath already signs,
       agreement with OmniPath >= 0.60 with binomial p < 0.05 (>= 20 pairs). Else every D1 list is
       UNVALIDATED.
   D1b ENRICHMENT: share of OmniPath-known pairs among triangulated vs moved-only vs bound-only vs
       neither (descriptive; triangulated must be the highest for D1 to be called informative).
   D1c NULL: X's knockdown paired with ANOTHER TF's ChIP set (100 random derangements) -> expected
       triangulated count by chance; estimated FDR = null mean / observed.
   OUTPUT: triangulated pairs in neither OmniPath nor the bank -- CANDIDATE new direct links.
D2 WHERE THE LITERATURE AND K562 DISAGREE. Signed transcriptional OmniPath edges X -> Y with X
   knocked down and Y a mover: agreement rate; by ChIP support; by curation-effort tertile (top vs
   bottom, Fisher exact p < 0.05 -> "evidence predicts correctness"). The same for the bank's own
   signed regulatory edges. OUTPUT: strongest ChIP-supported contradictions.
D3 DOES BETTER INFORMATION FIX THE DIRECTION PROBLEM? cellbank2's engine on three fact tables:
       v2   the bank's literature facts (cellbank2.py)
       v2+O plus OmniPath signed facts
       v3   plus K562-MEASURED facts from TRAIN knockdowns (X -> Y for movers, sign -sign(z))
   TEST knockdowns (cellbank2's hash split, first 400 by hash) never contribute their own facts.
   Paired per-knockdown sign tests on direction accuracy (>= 5 scored movers), p < 0.05; fair
   pooled enrichment (observed vs knockdown-specific expected movers) per hop.
D4 WHAT DO UNCHARACTERISED GENES DO? Two independent assays: knockdown-profile similarity (genes
   with a transcriptional phenotype, energy-test p < 0.05) and DepMap co-essentiality (Pearson
   over 1,178 lines). VALIDATION, BLOCKING FOR CLAIMS: AUROC for "same HGNC gene group" (groups of
   2-50 genes); the combined score must beat each assay alone and exceed 0.60.
   DISCOVERY: an uncharacterised gene (HGNC name: open reading frame / uncharacterized / family
   with sequence similarity / KIAA) is assigned a group when >= 3 of its top-5 characterised
   partners share it. NULL: the same with the knockdown-profile labels permuted -> estimated FDR.
D5 WHAT THIS IS NOT. Candidates, not findings: one cell line, steady state, CRISPRi; ChIP binding is
   not regulation; HGNC groups are an imperfect answer key. Nothing here is validated at the bench.
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
import re
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CA = os.path.join(HERE, "_cache")
OUT = os.path.join(HERE, "RESULTS_celldiscover.txt")
ART = R("outputs", "celldiscover.json")
BANK3 = R("memory_bank", "cell_v3")
RULE = "=" * 97
ZMOV = None          # set from the measured scale in main()
_s = importlib.util.spec_from_file_location("cellbank2", os.path.join(HERE, "cellbank2.py"))
cb = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb)
dc = cb.dc


def load_gwps():
    import h5py
    f = h5py.File(os.path.join(CA, "perturbseq", "K562_gwps_normalized_bulk_01.h5ad"), "r")
    X = f["X"][()]
    gt = [s.decode() for s in f["obs/gene_transcript"][()]]
    fold = f["obs/fold_expr"][()]; core = f["obs/core_control"][()]; ep = f["obs/energy_test_p_value"][()]
    cats = [s.decode() for s in f["var/__categories/gene_name"][()]]
    genes = [cats[c] for c in f["var/gene_name"][()]]
    best = {}
    for i, g in enumerate(gt):
        parts = g.split("_")
        sym = parts[1] if len(parts) > 1 else g
        if core[i] or "non-targeting" in g.lower() or sym.lower().startswith("non"):
            continue
        fv = fold[i] if np.isfinite(fold[i]) else 1.0
        if sym not in best or fv < best[sym][1]:
            best[sym] = (i, fv)
    kos = sorted(best)
    rows = [best[k][0] for k in kos]
    Xs = X[rows].astype(np.float32)
    Xs[~np.isfinite(Xs)] = np.nan
    return Xs, kos, genes, np.array([ep[r] for r in rows]), np.array([best[k][1] for k in kos])


def load_omnipath():
    out = []
    with open(os.path.join(CA, "omnipath", "interactions_commercial.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            st, inh = r["consensus_stimulation"] == "True", r["consensus_inhibition"] == "True"
            if st == inh:
                continue
            out.append((r["source_genesymbol"], r["target_genesymbol"], 1 if st else -1, r["type"],
                        int(r["curation_effort"] or 0)))
    return out


def load_encode():
    sets = collections.defaultdict(set)
    for line in open(os.path.join(CA, "encode", "ENCODE_TF_ChIP-seq_2015.gmt")):
        p = line.rstrip("\n").split("\t")
        if " K562 " not in p[0] + " ":
            continue
        tf = p[0].split()[0].replace("eGFP-", "").upper()
        sets[tf] |= {x.upper() for x in p[2:] if x}
    return sets


def load_depmap():
    import pandas as pd
    df = pd.read_csv(os.path.join(CA, "depmap", "CRISPRGeneEffect.csv"), index_col=0)
    df.columns = [c.split(" (")[0] for c in df.columns]
    return df


def load_hgnc():
    groups, names = {}, {}
    with open(os.path.join(HERE, "hgnc_complete_set.txt")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["locus_group"] != "protein-coding gene":
                continue
            names[r["symbol"]] = r["name"]
            groups[r["symbol"]] = {g for g in r["gene_group"].split("|") if g}
    return groups, names


def binom_p_greater(k, n, p0=0.5):
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1)))


def fisher_2x2(a, b, c, d):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, b], [c, d]])[1])


def auroc_pairs(scores_pos, scores_neg):
    s = np.concatenate([scores_pos, scores_neg]); y = np.r_[np.ones(len(scores_pos)), np.zeros(len(scores_neg))]
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - len(scores_pos) * (len(scores_pos) + 1) / 2) / (len(scores_pos) * len(scores_neg)))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    rng = np.random.default_rng(20261006)
    P_(RULE); P_("WHAT CAN THE MEMORY BANK DISCOVER WITH THE NEW DATA?"); P_(RULE)
    X, kos, genes, ep, fold = load_gwps()
    gidx = {g: j for j, g in enumerate(genes)}; kidx = {k: i for i, k in enumerate(kos)}
    omni = load_omnipath(); enc = load_encode(); dep = load_depmap(); groups, names = load_hgnc()
    D = json.load(gzip.open(dc.ENCY)); bnames = [r["name"] for r in D["genes"]]
    bank_rows, bank_by_src, _, _ = cb.facts_table(D, bnames)
    bank_pairs = {(s, t) for s, t, _, _ in bank_rows}
    bank_sign = {}
    for s, t, sg, l in bank_rows:
        if l == 0:
            bank_sign[(s, t)] = sg
    global ZMOV
    import pyarrow.parquet as pq
    ess = pq.read_table(os.path.join(CA, "perturbseq", "k562ess_perturbation_zscores.parquet")).to_pandas()
    sk = [k for k in ess.index if k in kidx]; sg_ = [g for g in ess.columns if g in gidx]
    Ae = ess.loc[sk, sg_].values; Bg = X[np.ix_([kidx[k] for k in sk], [gidx[g] for g in sg_])]
    m_ = np.isfinite(Ae) & np.isfinite(Bg) & (np.abs(Ae) > 3) & (np.abs(Bg) > 1e-6)
    scale = float(np.median(np.abs(Ae[m_]) / np.abs(Bg[m_])))
    ZMOV = 3.0 / scale
    cb.ZMOV = ZMOV
    P_(f"  measured scale (essential z / gwps value) {scale:.2f} over {int(m_.sum()):,} shared large entries -> mover |value| >= {ZMOV:.3f}")
    own = [X[kidx[k], gidx[k]] for k in kos if k in gidx and np.isfinite(X[kidx[k], gidx[k]])]
    tfs = sorted(set(enc) & set(kos))
    omni_tx = {(s, t): (sg, ce) for s, t, sg, ty, ce in omni if ty == "transcriptional"}
    omni_all = {(s, t) for s, t, _, _, _ in omni}
    P_(f"  K562 genome-wide: {len(kos):,} knocked-down genes x {len(genes):,} readouts; own-gene z median {np.median(own):+.2f}")
    P_(f"  OmniPath signed {len(omni):,} (transcriptional {len(omni_tx):,}); ENCODE K562 TFs {len(enc)} ({len(tfs)} also knocked down); "
       f"DepMap {dep.shape[0]:,} models x {dep.shape[1]:,} genes; HGNC protein-coding {len(names):,}")
    g0 = 3 <= scale <= 15 and np.median(own) <= -1 / scale and "ACH-000551" in dep.index and len(tfs) >= 20 and len(omni) >= 10000
    P_(f"  G0: {'PASS' if g0 else 'FAIL'}")
    if not g0:
        open(OUT, "w").write("\n".join(out) + "\n"); return
    res = {}

    # =========================================================================== D1 ============
    P_("\n" + RULE); P_("D1  NEW DIRECT REGULATORY LINKS IN K562 (knockdown x ChIP x not in any database)"); P_(RULE)
    meas = set(genes)
    tri, cats = [], collections.Counter()
    known_in = collections.Counter(); totals = collections.Counter()
    for tf in tfs:
        z = X[kidx[tf]]
        bound = enc[tf] & meas
        for j, g in enumerate(genes):
            if g == tf:
                continue
            mv = abs(z[j]) >= ZMOV; bd = g in bound
            cat = "triangulated" if mv and bd else "moved only" if mv else "bound only" if bd else "neither"
            totals[cat] += 1
            if (tf, g) in omni_tx:
                known_in[cat] += 1
            if cat == "triangulated":
                tri.append((tf, g, float(z[j])))
    agree = [(-np.sign(zz) == omni_tx[(tf, g)][0]) for tf, g, zz in tri if (tf, g) in omni_tx]
    k_, n_ = int(sum(agree)), len(agree)
    pa = binom_p_greater(k_, n_)
    d1a = n_ >= 20 and k_ / max(n_, 1) >= 0.60 and pa < 0.05
    P_(f"  triangulated pairs {len(tri):,} over {len(tfs)} TFs")
    P_(f"  D1a positive control: agreement with OmniPath's sign {k_}/{n_} = {k_ / max(n_, 1):.3f}, p {pa:.3g} -> {'PASS' if d1a else 'FAIL'}")
    P_("  D1b share of OmniPath-known pairs by category:")
    for cat in ("triangulated", "moved only", "bound only", "neither"):
        P_(f"     {cat:<13} {known_in[cat]:>6,} / {totals[cat]:>9,} = {known_in[cat] / max(totals[cat], 1):.4f}")
    d1b = all(known_in["triangulated"] / max(totals["triangulated"], 1) > known_in[c] / max(totals[c], 1)
              for c in ("moved only", "bound only", "neither"))
    P_(f"  D1b triangulated is the most enriched category: {d1b}")
    nulls = []
    for _ in range(100):
        perm = list(tfs)
        while True:
            rng.shuffle(perm)
            if all(a != b for a, b in zip(tfs, perm)):
                break
        c = 0
        for tf, other in zip(tfs, perm):
            z = X[kidx[tf]]
            bound = (enc[other] & meas) - {tf}
            c += sum(1 for g in bound if abs(z[gidx[g]]) >= ZMOV)
        nulls.append(c)
    fdr = float(np.mean(nulls)) / max(len(tri), 1)
    P_(f"  D1c null (another TF's ChIP set): {np.mean(nulls):,.0f} +/- {np.std(nulls):.0f} expected vs {len(tri):,} observed -> estimated FDR {fdr:.2f}")
    novel = sorted([(tf, g, zz) for tf, g, zz in tri if (tf, g) not in omni_all and (tf, g) not in bank_pairs],
                   key=lambda x: -abs(x[2]))
    P_(f"  CANDIDATE NEW DIRECT LINKS (in neither OmniPath nor the bank): {len(novel):,}"
       + ("" if d1a and d1b else "   [UNVALIDATED -- D1a/D1b did not pass]"))
    for tf, g, zz in novel[:25]:
        P_(f"     {tf:>8} {'activates' if zz < 0 else 'represses'} {g:<10} (knockdown z-equiv {zz * scale:+.1f}; bound in K562 ChIP)")
    per_tf = collections.Counter(tf for tf, _, _ in novel)
    P_(f"  by TF: {dict(per_tf.most_common(10))}")
    res["D1"] = dict(triangulated=len(tri), tfs=len(tfs), control_agree=k_, control_n=n_, control_p=pa, d1a=d1a, d1b=d1b,
                     null_mean=float(np.mean(nulls)), fdr=fdr, novel=len(novel), known_share={c: known_in[c] / max(totals[c], 1) for c in totals})

    # =========================================================================== D2 ============
    P_("\n" + RULE); P_("D2  WHERE THE LITERATURE AND K562 DISAGREE"); P_(RULE)
    obs = []
    for (s, t), (sg, ce) in omni_tx.items():
        if s in kidx and t in gidx and s != t:
            zz = X[kidx[s], gidx[t]]
            if abs(zz) >= ZMOV:
                obs.append((s, t, sg, ce, float(zz), s in enc and t in enc[s]))
    ag = np.array([(-np.sign(o[4]) == o[2]) for o in obs])
    P_(f"  OmniPath transcriptional signed edges testable in K562 (source knocked down, target moved): {len(obs):,}")
    P_(f"  agreement overall {ag.mean():.3f} (p {binom_p_greater(int(ag.sum()), len(ag)):.3g} for > 0.5)")
    chip = np.array([o[5] for o in obs])
    for lab, msk in (("ChIP-supported", chip), ("no ChIP support", ~chip)):
        if msk.sum():
            P_(f"     {lab:<16} {ag[msk].mean():.3f} over {int(msk.sum()):,}")
    ce = np.array([o[3] for o in obs])
    lo, hi = np.percentile(ce, 33.3), np.percentile(ce, 66.7)
    top, bot = ce > hi, ce <= lo
    pf = fisher_2x2(int(ag[top].sum()), int((~ag[top]).sum()), int(ag[bot].sum()), int((~ag[bot]).sum())) if top.sum() and bot.sum() else 1.0
    P_(f"  by curation effort: top tertile (> {hi:.0f}) {ag[top].mean():.3f} (n {int(top.sum())}) vs bottom (<= {lo:.0f}) "
       f"{ag[bot].mean():.3f} (n {int(bot.sum())}); Fisher p {pf:.3g} -> "
       + ("EVIDENCE PREDICTS CORRECTNESS" if pf < 0.05 and ag[top].mean() > ag[bot].mean() else "evidence does not predict correctness"))
    bobs = [(s, t, sg) for (s, t), sg in bank_sign.items() if s in kidx and t in gidx and abs(X[kidx[s], gidx[t]]) >= ZMOV and s != t]
    bag = np.array([(-np.sign(X[kidx[s], gidx[t]]) == sg) for s, t, sg in bobs])
    P_(f"  the bank's own signed regulatory edges: agreement {bag.mean():.3f} over {len(bag):,} "
       f"(p {binom_p_greater(int(bag.sum()), len(bag)):.3g})")
    contra = sorted([o for o in obs if o[5] and (-np.sign(o[4]) != o[2])], key=lambda o: -(abs(o[4]) * (1 + o[3])))
    P_("  strongest ChIP-supported contradictions (literature sign vs what K562 shows):")
    for s, t, sg, cev, zz, _ in contra[:12]:
        P_(f"     {s:>8} -> {t:<10} literature {'activates' if sg > 0 else 'represses'} (curation {cev}); in K562 knockdown z-equiv {zz * scale:+.1f}")
    res["D2"] = dict(n=len(obs), agree=float(ag.mean()), chip_agree=float(ag[chip].mean()) if chip.sum() else None,
                     nochip_agree=float(ag[~chip].mean()) if (~chip).sum() else None, top_tertile=float(ag[top].mean()),
                     bottom_tertile=float(ag[bot].mean()), fisher_p=pf, bank_agree=float(bag.mean()), bank_n=len(bag))

    # =========================================================================== D3 ============
    P_("\n" + RULE); P_("D3  DOES BETTER INFORMATION FIX THE DIRECTION PROBLEM?"); P_(RULE)
    protein = {bnames[int(k)]: v for k, v in D["ppm"].items() if k.isdigit()}
    dbd_dead = {}
    ij = R("outputs", "isoform_edge_bounds.json")
    if os.path.exists(ij):
        for g, v in json.load(open(ij)).items():
            dbd_dead[g] = len(v.get("dbd_disrupted", [])) / max(v.get("n_isoforms") or 1, 1)
    split = lambda k: "TRAIN" if int(hashlib.sha256(k.encode()).hexdigest(), 16) % 2 == 0 else "TEST"
    train_k = [k for k in kos if split(k) == "TRAIN"]
    omni_rows = [(s, t, sg, 0 if ty == "transcriptional" else 1) for s, t, sg, ty, _ in omni]
    meas_rows = []
    for k in train_k:
        z = X[kidx[k]]
        for j in np.where(np.abs(z) >= ZMOV)[0]:
            if genes[j] != k:
                meas_rows.append((k, genes[j], int(-np.sign(z[j])), 0))

    def table(rows_):
        by = collections.defaultdict(list)
        for i, r in enumerate(rows_):
            by[r[0]].append(i)
        return rows_, by
    T2 = table(list(bank_rows)); T2O = table(list(bank_rows) + omni_rows); T3 = table(list(bank_rows) + omni_rows + meas_rows)
    P_(f"  facts: v2 {len(T2[0]):,}   v2+OmniPath {len(T2O[0]):,}   v3 (+ K562-measured from {len(train_k):,} TRAIN knockdowns) {len(T3[0]):,}")
    mk = lambda: dc.Cell(bnames, protein, {}, dbd_dead, {})
    test_k = sorted([k for k in kos if split(k) == "TEST" and k in T2[1]], key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    elig = []
    for k in test_k:
        C = mk(); pr = cb.new_engine(C, T2[0], T2[1], {k: -1})
        if sum(1 for g in pr if g in gidx and g != k) >= 10:
            elig.append(k)
        if len(elig) >= 400:
            break
    P_(f"  TEST knockdowns scored: {len(elig)} (first 400 eligible by hash)")
    scores = {}
    for nm, T in (("v2", T2), ("v2+O", T2O), ("v3", T3)):
        sc = {}
        for k in elig:
            C = mk(); pr = cb.new_engine(C, T[0], T[1], {k: -1})
            sc[k] = cb.score(pr, X[kidx[k]].astype(float), gidx, k)
        scores[nm] = sc
    for nm in ("v2", "v2+O", "v3"):
        line = f"  {nm:<5}"
        for h in (1, 2, 3):
            o_ = sum(s[h]["movers"] for s in scores[nm].values()); e_ = sum(s[h]["n"] * s[h]["base"] for s in scores[nm].values())
            hi_ = sum(s[h]["hits"] for s in scores[nm].values())
            line += f"   hop {h}: enrichment {o_ / max(e_, 1e-9):.2f}x ({o_:,} movers), direction {hi_ / max(o_, 1):.3f}"
        P_(line)
    accf = lambda s: (sum(s[h]["hits"] for h in (1, 2, 3)) / max(sum(s[h]["movers"] for h in (1, 2, 3)), 1)
                      if sum(s[h]["movers"] for h in (1, 2, 3)) >= 5 else None)
    for a_, b_ in (("v2+O", "v2"), ("v3", "v2+O"), ("v3", "v2")):
        d = [accf(scores[a_][k]) - accf(scores[b_][k]) for k in elig if accf(scores[a_][k]) is not None and accf(scores[b_][k]) is not None]
        w, l, p = cb.sign_test(d)
        v = "BETTER" if p < 0.05 and w > l else "WORSE" if p < 0.05 else "NO DIFFERENCE"
        res[f"D3_{a_}_vs_{b_}"] = dict(better=w, worse=l, p=p, n=len(d), mean=float(np.mean(d)) if d else None, verdict=v)
        P_(f"  direction accuracy {a_} vs {b_}: better {w}, worse {l} of {len(d)}, mean {np.mean(d) if d else float('nan'):+.3f}, p {p:.3g} -> {v}")

    # =========================================================================== D4 ============
    P_("\n" + RULE); P_("D4  WHAT DO UNCHARACTERISED GENES DO? (knockdown profiles x DepMap co-essentiality)"); P_(RULE)
    G = [k for k, e in zip(kos, ep) if np.isfinite(e) and e < 0.05 and k in dep.columns and k in names]
    okc = np.all(np.isfinite(X), 0)
    A = X[[kidx[g] for g in G]][:, okc].astype(np.float64)
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    SA = (A @ A.T) / A.shape[1]
    Bm = dep[G].values.astype(np.float64)
    Bm = np.where(np.isfinite(Bm), Bm, np.nanmean(Bm, 0, keepdims=True))
    Bm = (Bm - Bm.mean(0)) / (Bm.std(0) + 1e-9)
    SB = (Bm.T @ Bm) / Bm.shape[0]
    fz = lambda S: np.arctanh(np.clip(S, -0.999, 0.999))
    SC = (fz(SA) + fz(SB)) / 2
    gsize = collections.Counter(gr for g in G for gr in groups.get(g, ()))
    ok_groups = {gr for gr, n in gsize.items() if 2 <= n <= 50}
    gg = [groups.get(g, set()) & ok_groups for g in G]
    n = len(G)
    iu = np.triu_indices(n, 1)
    glist = sorted(ok_groups); gpos = {gr: c for c, gr in enumerate(glist)}
    Mg = np.zeros((n, len(glist)), dtype=np.float32)
    for i, s_ in enumerate(gg):
        for gr in s_:
            Mg[i, gpos[gr]] = 1.0
    same = ((Mg @ Mg.T) > 0)[iu]
    pos = np.where(same)[0]; neg = rng.choice(np.where(~same)[0], size=min(200000, int((~same).sum())), replace=False)
    au = {nm: auroc_pairs(S[iu][pos], S[iu][neg]) for nm, S in (("knockdown profile", SA), ("co-essentiality", SB), ("combined", SC))}
    P_(f"  genes with a phenotype and DepMap data: {n:,}; same-group pairs {len(pos):,} ({len(ok_groups)} groups of 2-50)")
    for nm, a in au.items():
        P_(f"     AUROC {nm:<18} {a:.3f}")
    d4 = au["combined"] > max(au["knockdown profile"], au["co-essentiality"]) and au["combined"] > 0.60
    P_(f"  D4 validation: {'PASS' if d4 else 'FAIL'}")
    pat = re.compile(r"open reading frame|uncharacterized|family with sequence similarity|kiaa", re.I)
    unk = [i for i, g in enumerate(G) if pat.search(names.get(g, ""))]
    known_i = [i for i, g in enumerate(G) if gg[i]]

    def assign(S):
        hits = []
        for i in unk:
            cand = [j for j in known_i if j != i]
            top = sorted(cand, key=lambda j: -S[i, j])[:5]
            c = collections.Counter(gr for j in top for gr in gg[j])
            if c:
                gr, cnt = c.most_common(1)[0]
                if cnt >= 3:
                    hits.append((G[i], gr, [G[j] for j in top], float(np.mean([S[i, j] for j in top]))))
        return hits
    hits = assign(SC)
    nul = []
    for _ in range(20):
        p_ = rng.permutation(n)
        SAp = SA[np.ix_(p_, p_)]
        nul.append(len(assign((fz(SAp) + fz(SB)) / 2)))
    fdr4 = float(np.mean(nul)) / max(len(hits), 1)
    P_(f"  uncharacterised genes with both assays: {len(unk)}; assigned a group (>= 3 of top-5 partners): {len(hits)}; "
       f"null {np.mean(nul):.1f} +/- {np.std(nul):.1f} -> estimated FDR {fdr4:.2f}" + ("" if d4 else "   [UNVALIDATED]"))
    for g, gr, top, sc_ in sorted(hits, key=lambda h: -h[3])[:20]:
        P_(f"     {g:<10} ({names[g][:42]:<42}) -> {gr[:48]:<48} partners {', '.join(top[:5])}")
    res["D4"] = dict(n_genes=n, auroc=au, validated=d4, n_unk=len(unk), assigned=len(hits), null_mean=float(np.mean(nul)), fdr=fdr4)

    # =========================================================================== write ==========
    os.makedirs(BANK3, exist_ok=True)
    json.dump({"schema": "cell_v2 engine + OmniPath (commercial) + K562-measured facts; discoveries with evidence",
               "sources": {"perturbseq": "Replogle 2022 K562 gwps, figshare 20029387, CC BY 4.0",
                           "omnipath": "commercial-licence subset", "encode": "ENCODE TF ChIP-seq K562 via Enrichr 2015 library",
                           "depmap": "24Q4 CRISPRGeneEffect, figshare 27993248, CC BY 4.0"},
               "n_facts_v3": len(T3[0]), "results": res},
              open(os.path.join(BANK3, "bank.json"), "w"), indent=1, default=float)
    with open(os.path.join(BANK3, "candidate_direct_links_K562.tsv"), "w") as fh:
        fh.write("tf\ttarget\tdirection\tknockdown_z\tevidence\n")
        for tf, g, zz in novel:
            fh.write(f"{tf}\t{g}\t{'activates' if zz < 0 else 'represses'}\t{zz:.2f}\tK562 CRISPRi + K562 ChIP; not in OmniPath or bank\n")
    with open(os.path.join(BANK3, "literature_contradictions_K562.tsv"), "w") as fh:
        fh.write("source\ttarget\tliterature\tcuration_effort\tknockdown_z\tchip_bound\n")
        for s, t, sg, cev, zz, cb_ in sorted([o for o in obs if -np.sign(o[4]) != o[2]], key=lambda o: -abs(o[4])):
            fh.write(f"{s}\t{t}\t{'activates' if sg > 0 else 'represses'}\t{cev}\t{zz:.2f}\t{cb_}\n")
    with open(os.path.join(BANK3, "uncharacterised_gene_assignments.tsv"), "w") as fh:
        fh.write("gene\tname\tpredicted_group\ttop5_partners\tmean_score\n")
        for g, gr, top, sc_ in sorted(hits, key=lambda h: -h[3]):
            fh.write(f"{g}\t{names[g]}\t{gr}\t{','.join(top)}\t{sc_:.3f}\n")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(res, open(ART, "w"), indent=1, default=float)
    P_(f"\n  bank written to memory_bank/cell_v3/ ; artifact outputs/celldiscover.json ; runtime {time.time() - t0:.0f}s")
    P_("  D5: candidates, not findings -- one cell line, CRISPRi, steady state; nothing validated at the bench.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
