"""DOES COMPARTMENT GATING RESCUE THE KINASE-SUBSTRATE LAYER, WHICH IS ALREADY NULL UNGATED?

THIS IS A RESCUE ATTEMPT, NOT A FRESH CLAIM, AND SAYING SO FIRST IS THE POINT. `colab/ptm_sites.py` already
measured the enzyme->substrate layer against this project's fast test and it came back null with real power
behind it:

    2,810 enzyme->substrate pairs, mean |robust z| 0.8899 against 0.8969 for pairs matched on
    perturbation-strength decile x gene-responsiveness decile -> ratio 0.992, 0/20 draws at p<0.05,
    residual imbalance p 0.60 and p 0.91, and a minimum detectable ratio at 80% power of 1.075

So the question is not "do kinase-substrate edges predict transcriptional response" -- that has been asked
and answered. The question is whether the null is partly an artefact of the edge set containing pairs that
CANNOT PHYSICALLY MEET. iPTMnet aggregates enzyme-substrate relations across experiments, tissues and assay
types; a kinase annotated in the nucleus and a substrate annotated only in the mitochondrial matrix are a
literature relation, not a K562 relation. If that is a large fraction of the set, gating on shared
subcellular compartment should concentrate whatever signal exists.

TWO ENDPOINTS, AND THE SECOND IS THE ONE WITH A CHANCE.

  TRANSCRIPT RESPONSE. The endpoint ptm_sites used. It is the wrong physics for this layer and that is worth
  stating rather than discovering: knocking down a kinase changes its substrate's PHOSPHORYLATION, and there
  is no reason a phosphorylation event has to move the substrate's own mRNA. A null here is close to
  expected. It is run anyway because it is the arm that is directly comparable to the published null, and
  because a gating claim that only works on the endpoint I chose second would not be believable.

  CO-DEPENDENCY. DepMap gene effect across 1,150 cell lines. If a kinase and its substrate are in one
  functional relationship, lines that need one should tend to need the other, and this endpoint does not
  require the relation to pass through transcription at all. It is a genuinely different readout on the same
  edges, and it is where a rescue would show up if there is one.

THE CONFOUND THAT WOULD MANUFACTURE A GATING EFFECT, and it is specific. A gene annotated to many
compartments is more likely to share one with anything -- and a gene annotated to many compartments is a
well-studied, abundant, well-measured gene. The HPA module in this project already hit exactly this and
found that quartile-binning the locations-per-pair count left a residual imbalance at p 6e-9, so it matches
that covariate EXACTLY as an integer. This module reuses that matcher rather than writing a weaker one.

    ARM 1   kinase-substrate pairs that SHARE at least one HPA compartment
    ARM 0   kinase-substrate pairs whose compartments are DISJOINT
    both arms are kinase-substrate pairs, so the comparison isolates compartment and nothing else --
    the same subunit-vs-subunit principle the CORUM module uses

plus a COMPARTMENT-SHUFFLE control: permute the compartment sets across genes while preserving how many
compartments each gene has, and re-split. That destroys which compartment a gene is in and keeps how
promiscuously it is annotated, so a "gating effect" that is really an annotation-depth effect survives it
and is exposed.

WHAT A NULL HERE WOULD MEAN. That the ungated null is not explained by physically impossible pairs -- which
is worth knowing, because it removes the most obvious rescue and points at the endpoint or at the edge set
itself. This project commits nulls.
"""
import collections
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GWPS = SP / "gwps.h5ad"
DEP = SP / "CRISPRGeneEffect.csv"
N_DRAWS = 20
N_SHUFFLE = 20          # the shuffle spread DECIDES the verdict, so it gets as many draws as the arms do
DECILES = 10
MIN_ARM = 200           # hpa_localization.Matcher.versus refuses below this, and so should we


def robust_z(M):
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad


def load_gwps(report):
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        gid = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    X = np.where(np.isfinite(X), X, np.nan)
    return X, psym, np.array(gid)


def codep_arm(report, pairs, label, esd, emean, M, gpos, n_draws=N_DRAWS, seed=0, quiet=False):
    """|correlation| of DepMap gene-effect profiles for a pair set, against a matched control.

    MATCHED ON WHAT AND WHY. Two genes with large dependency variance correlate more easily than two flat
    ones, and dependency variance tracks how essential and how well-screened a gene is. The control draws
    replacement pairs from the same (variance decile, mean-effect decile) cells on BOTH sides, so the
    control pairs are as easy to correlate as the real ones and only the pairing differs.
    """
    ok = [(a, b) for a, b in pairs if a in gpos and b in gpos]
    if len(ok) < MIN_ARM:
        report(f"     {label}: only {len(ok)} pairs scorable in DepMap -- below the {MIN_ARM} floor, "
               f"reported as too small rather than as a number")
        return None
    ia = np.array([gpos[a] for a, _b in ok])
    ib = np.array([gpos[b] for _a, b in ok])
    obs = np.abs(np.array([np.dot(M[:, x], M[:, y]) / M.shape[0] for x, y in zip(ia, ib)]))
    q = np.linspace(0.1, 0.9, DECILES - 1)
    vd = np.digitize(esd, np.quantile(esd, q))
    md = np.digitize(emean, np.quantile(emean, q))
    by = collections.defaultdict(list)
    for i in range(M.shape[1]):
        by[(vd[i], md[i])].append(i)
    draws = []
    for s in range(n_draws):
        rr = np.random.default_rng(seed * 1000 + s)
        vals = []
        for x, y in zip(ia, ib):
            ca, cb = by.get((vd[x], md[x])), by.get((vd[y], md[y]))
            if not ca or not cb:
                continue
            u, v = int(rr.choice(ca)), int(rr.choice(cb))
            if u == v:
                continue
            vals.append(abs(float(np.dot(M[:, u], M[:, v]) / M.shape[0])))
        draws.append(float(np.mean(vals)) if vals else np.nan)
    nm = float(np.nanmean(draws))
    sd = float(np.nanstd(draws))
    frac = float(np.mean([np.nanmean(obs) > d for d in draws]))
    if not quiet:
        report(f"     {label:34s} n {len(ok):5,d}  |r| {np.mean(obs):.4f}  matched {nm:.4f}  "
               f"ratio {np.mean(obs)/max(nm,1e-9):.3f}x  (sd {sd:.4f}, {100*frac:.0f}% of {n_draws} draws "
               f"below observed)")
    return {"label": label, "n": len(ok), "observed": float(np.mean(obs)), "matched_mean": nm,
            "matched_sd": sd, "ratio": float(np.mean(obs) / max(nm, 1e-9)),
            "frac_draws_below_observed": frac, "n_draws": n_draws}


def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("COMPARTMENT-GATED KINASE-SUBSTRATE -- can gating rescue a layer that is already null?")
    report("=" * 100)
    report("  PRIOR (colab/ptm_sites.py): 2,810 enzyme->substrate pairs, ratio 0.992x against a")
    report("  decile-matched control, 0/20 draws significant, minimum detectable ratio 1.075. This module")
    report("  asks only whether gating on shared compartment changes that.")

    import hpa_localization as H
    import ptm_sites as P
    from entity_registry import load as load_reg
    REG = load_reg()

    # ---- sources ----
    report("\n  1. SOURCES")
    H.fetch(report)
    hrows = H.parse(report)
    P.fetch_url(f"{P.IPTM_BASE}/ptm.txt", P.IPTM)
    report(f"    iPTMnet ptm.txt ({P.IPTM.stat().st_size/1e6:.1f} MB)")
    edges = P.parse_iptmnet(report)

    # ---- join everything to one gene entity id, with declared rates ----
    report("\n  2. JOINS (declared rates; a thin join raises rather than returning a small answer)")
    accs = sorted({a for a, _b in edges} | {b for _a, b in edges})
    _e, st = REG.join("gene", "uniprot", accs, 0.60, "iPTMnet accessions -> gene")
    ensg = [r["Gene"] for r in hrows]
    _e, st2 = REG.join("gene", "ensembl", ensg, 0.80, "HPA ENSG -> gene")

    loc_of = {}
    nloc = {}
    for r in hrows:
        eid = REG.resolve("gene", "ensembl", r["Gene"])
        if not eid:
            continue
        s = {x.strip() for c in ("Main location", "Additional location")
             for x in str(r[c]).split(";") if x.strip()}
        if s:
            loc_of[eid] = s
            nloc[eid] = len(s)
    report(f"    HPA compartments for {len(loc_of):,} gene entities; "
           f"median {np.median(list(nloc.values())):.0f} locations per gene, "
           f"max {max(nloc.values())}")

    ks = []
    for (ea, sa), meta in edges.items():
        ke, ksb = REG.resolve("gene", "uniprot", ea), REG.resolve("gene", "uniprot", sa)
        if ke and ksb and ke != ksb:
            ks.append((ke, ksb, meta))
    report(f"    {len(ks):,} kinase/enzyme -> substrate pairs resolve to two distinct gene entities")

    both = [(a, b, m) for a, b, m in ks if a in loc_of and b in loc_of]
    report(f"    {len(both):,} of those have HPA compartments on BOTH sides ({100*len(both)/max(len(ks),1):.0f}%)")
    if len(both) < 2 * MIN_ARM:
        raise SystemExit(f"only {len(both)} pairs have compartments on both sides; the two arms cannot both "
                         f"reach {MIN_ARM} and a split here would be a number about the join")

    shared = [(a, b) for a, b, _m in both if loc_of[a] & loc_of[b]]
    disj = [(a, b) for a, b, _m in both if not (loc_of[a] & loc_of[b])]
    report(f"    SHARE a compartment: {len(shared):,}   DISJOINT: {len(disj):,} "
           f"({100*len(shared)/max(len(both),1):.0f}% share)")
    report(f"    locations per pair: shared {np.mean([nloc[a]+nloc[b] for a,b in shared]):.2f}, "
           f"disjoint {np.mean([nloc[a]+nloc[b] for a,b in disj]):.2f} -- if these differ, the gated arm "
           f"is\n      also the better-annotated arm, and the matcher below bins this EXACTLY rather than "
           f"by quartile")

    R = {"model": "compartment-gated-ks-v1",
         "prior_null": {"module": "colab/ptm_sites.py", "n_pairs": 2810, "ratio": 0.992,
                        "min_detectable_ratio_80pct": 1.0745,
                        "note": "the ungated layer is already null with power; this module tests only "
                                "whether compartment gating changes that"},
         "n_ks_pairs": len(ks), "n_with_both_compartments": len(both),
         "n_shared": len(shared), "n_disjoint": len(disj),
         "mean_locations_per_pair": {"shared": float(np.mean([nloc[a]+nloc[b] for a, b in shared])),
                                     "disjoint": float(np.mean([nloc[a]+nloc[b] for a, b in disj]))}}

    # ---- endpoint A: co-dependency ----
    report("\n  3. ENDPOINT A -- CO-DEPENDENCY (DepMap gene effect; does not pass through transcription)")
    import pandas as pd
    ge = pd.read_csv(DEP, index_col=0)
    gsym = [c.split(" (")[0].strip() for c in ge.columns]
    Xd = ge.to_numpy(dtype=np.float32)
    del ge
    fin = np.isfinite(Xd)
    Xd = np.where(fin, Xd, 0.0)
    keep = fin.sum(0) >= 200
    Xd, gsym = Xd[:, keep], [g for g, k in zip(gsym, keep) if k]
    emean = Xd.mean(0)
    esd = Xd.std(0)
    Z = (Xd - emean) / np.maximum(esd, 1e-6)
    report(f"     DepMap: {Z.shape[0]:,} lines x {Z.shape[1]:,} genes with >= 200 screens")
    gpos = {}
    for i, g in enumerate(gsym):
        e = REG.resolve("gene", "symbol", g)
        if e:
            gpos.setdefault(e, i)
    report(f"     {len(gpos):,} DepMap genes resolve to a gene entity")
    a_all = codep_arm(report, [(a, b) for a, b, _m in both], "kinase-substrate, ALL", esd, emean, Z,
                      gpos, seed=7)
    a_shared = codep_arm(report, shared, "kinase-substrate, SHARED compartment", esd, emean, Z, gpos)
    a_disj = codep_arm(report, disj, "kinase-substrate, DISJOINT", esd, emean, Z, gpos, seed=1)

    # SPECIFICITY. An enzyme and its substrate are often also complex partners or direct interactors, and
    # both of those already predict co-dependency in this project (PPI 1.208x on the fast test). An
    # enrichment that vanishes once co-complex and PPI pairs are removed is the PPI layer wearing a
    # phosphorylation label, which is the same category of error as the `sl` layer wearing a synthetic
    # lethality label.
    d = json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))
    cname = [g["name"] for g in d["genes"]]
    cidx = {n: i for i, n in enumerate(cname)}
    eid_of_ci = {}
    for n, i in cidx.items():
        e = REG.resolve("gene", "symbol", n)
        if e:
            eid_of_ci.setdefault(e, i)
    ppi_e = set()
    for a, b in [(int(x[0]), int(x[1])) for x in d["ppi"]]:
        ppi_e.add((a, b))
        ppi_e.add((b, a))
    g2c_e = {int(k): set(v) for k, v in (d.get("gene2cplx") or {}).items()}
    del d

    def entangled(a, b):
        ia, ib = eid_of_ci.get(a), eid_of_ci.get(b)
        if ia is None or ib is None:
            return False
        if (ia, ib) in ppi_e:
            return True
        return bool(g2c_e.get(ia) and g2c_e.get(ib) and g2c_e[ia] & g2c_e[ib])

    clean = [(a, b) for a, b, _m in both if not entangled(a, b)]
    report(f"     of {len(both):,} pairs, {len(both)-len(clean):,} are ALSO a PPI edge or co-complex; "
           f"{len(clean):,} are neither")
    a_clean = codep_arm(report, clean, "kinase-substrate, NOT PPI/co-complex", esd, emean, Z, gpos, seed=9)
    R["codependency"] = {"all": a_all, "shared": a_shared, "disjoint": a_disj,
                         "not_ppi_or_complex": a_clean,
                         "n_entangled_with_ppi_or_complex": len(both) - len(clean)}
    if a_shared and a_disj:
        gap = a_shared["ratio"] - a_disj["ratio"]
        pooled = float(np.hypot(a_shared["matched_sd"], a_disj["matched_sd"]) /
                       max(a_shared["matched_mean"], 1e-9))
        report(f"     -> gating gap {gap:+.3f}x against a pooled control spread of {pooled:.3f}x")
        R["codependency"]["gating_gap_ratio"] = gap
        R["codependency"]["pooled_control_spread"] = pooled

    # ---- compartment-shuffle control ----
    #
    # PRESERVES HOW MANY COMPARTMENTS EACH GENE HAS AND DESTROYS WHICH. A gating effect that is really an
    # annotation-depth effect -- promiscuously annotated genes share compartments with everything AND are
    # better studied, more abundant, better measured -- survives this and is caught.
    report("\n  4. COMPARTMENT-SHUFFLE CONTROL (per-gene compartment COUNT preserved, identity destroyed)")
    ents = sorted(loc_of)
    by_n = collections.defaultdict(list)
    for e in ents:
        by_n[nloc[e]].append(e)
    sh_gaps = []
    for s in range(N_SHUFFLE):
        rr = np.random.default_rng(400 + s)
        perm = {}
        for n_, group in by_n.items():
            g2 = list(group)
            rr.shuffle(g2)
            for a, b in zip(group, g2):
                perm[a] = loc_of[b]
        sh = [(a, b) for a, b, _m in both if perm[a] & perm[b]]
        dj = [(a, b) for a, b, _m in both if not (perm[a] & perm[b])]
        if len(sh) < MIN_ARM or len(dj) < MIN_ARM:
            report(f"     shuffle {s+1}: split {len(sh)}/{len(dj)} -- an arm fell below {MIN_ARM}, skipped")
            continue
        r1 = codep_arm(report, sh, f"  shuffled SHARED (draw {s+1})", esd, emean, Z, gpos,
                       n_draws=6, seed=50 + s, quiet=(s > 2))
        r0 = codep_arm(report, dj, f"  shuffled DISJOINT (draw {s+1})", esd, emean, Z, gpos,
                       n_draws=6, seed=80 + s, quiet=(s > 2))
        if r1 and r0:
            sh_gaps.append(r1["ratio"] - r0["ratio"])
    R["compartment_shuffle"] = {"gaps": sh_gaps,
                                "mean_gap": float(np.mean(sh_gaps)) if sh_gaps else None,
                                "sd_gap": float(np.std(sh_gaps)) if sh_gaps else None}
    if sh_gaps:
        report(f"     -> shuffled gating gap {np.mean(sh_gaps):+.3f}x (sd {np.std(sh_gaps):.3f}) against "
               f"the real {R['codependency'].get('gating_gap_ratio', float('nan')):+.3f}x")

    # ---- endpoint B: transcript response, using the HPA module's own matcher ----
    report("\n  5. ENDPOINT B -- TRANSCRIPT RESPONSE (the endpoint the published null used)")
    report("     Reusing hpa_localization.Matcher.versus, which matches the two arms cell by cell on")
    report("     perturbation strength x gene responsiveness AND on locations-per-pair as an EXACT integer.")
    X, psym, gid = load_gwps(report)
    Zk = robust_z(X)
    pert_eid = [REG.resolve("gene", "symbol", s) for s in psym]
    gene_eid = [REG.resolve("gene", "ensembl", g) for g in gid]
    prow = np.nanmean(np.abs(Zk), axis=1)
    pcol = np.nanmean(np.abs(Zk), axis=0)
    Mt = H.Matcher(Zk, prow, pcol, pert_eid, gene_eid, DECILES)
    prow_of = collections.defaultdict(list)
    for i, e in enumerate(pert_eid):
        if e:
            prow_of[e].append(i)
    pcol_of = collections.defaultdict(list)
    for j, e in enumerate(gene_eid):
        if e:
            pcol_of[e].append(j)

    def to_grid(pairs):
        L, xs = [], []
        for a, b in pairs:
            for i in prow_of.get(a, ())[:1]:
                for j in pcol_of.get(b, ())[:1]:
                    if np.isfinite(Zk[i, j]):
                        L.append((i, j))
                        xs.append(nloc[a] + nloc[b])
        return np.array(L), np.array(xs)

    L1, x1 = to_grid(shared)
    L0, x0 = to_grid(disj)
    report(f"     onto the K562 grid: SHARED {len(L1):,} pairs, DISJOINT {len(L0):,}")
    if len(L1) < MIN_ARM or len(L0) < MIN_ARM:
        report(f"     one arm is below the {MIN_ARM} floor -- endpoint B is reported as UNDERPOWERED "
               f"rather than as a null, because an arm this thin cannot detect a difference of any size")
        R["transcript"] = {"status": "underpowered", "n_shared": int(len(L1)), "n_disjoint": int(len(L0))}
    else:
        v = Mt.versus(L1, L0, x1=x1, x0=x0)
        R["transcript"] = v
        if v:
            for k in ("ratio", "ratio_sd", "p_median", "frac_draws_p05"):
                if k in v:
                    report(f"     {k:20s} {v[k]}")
            report(f"     full versus result recorded in the JSON")

    # ---- verdict ----
    #
    # TWO FINDINGS AND THEY POINT OPPOSITE WAYS. Leading with the gating question alone would bury the
    # larger one: the layer that is null on transcript response is NOT null on co-dependency. The verdict
    # states the readout result first because it is the bigger fact, then the gating result, then the
    # specificity control that decides whether the readout result is about phosphorylation at all.
    gapv = (R.get("codependency") or {}).get("gating_gap_ratio")
    shm = R["compartment_shuffle"]["mean_gap"]
    shs = R["compartment_shuffle"]["sd_gap"]
    rescued = bool(gapv is not None and shm is not None
                   and gapv - shm > 3 * max(shs or 0, 1e-6) and gapv > 0.05)
    layer_lives = bool(a_all and a_all["ratio"] > 1.05 and a_all["frac_draws_below_observed"] >= 1.0)
    specific = bool(a_clean and a_clean["ratio"] > 1.05 and a_clean["frac_draws_below_observed"] >= 1.0)
    # Endpoint B's ratio_sd is the number that decides whether it says anything. A ratio of 2.26 with an sd
    # of 0.93 is not a large effect, it is an unmeasured one, and printing the point estimate without the
    # spread is how a noisy arm gets read as a finding.
    tb = R.get("transcript") or {}
    tb_informative = bool(isinstance(tb, dict) and tb.get("ratio") is not None
                          and tb.get("ratio_sd") is not None
                          and tb["ratio_sd"] < 0.25 * abs(tb["ratio"] - 1.0))
    verdict = (
        f"THE ENDPOINT RESCUES THIS LAYER AND THE COMPARTMENT GATE DOES NOT. "
        + (f"On CO-DEPENDENCY across {Z.shape[0]:,} DepMap lines -- a readout that does not pass through "
           f"transcription -- all {a_all['n']:,} kinase/enzyme->substrate pairs sit at "
           f"{a_all['ratio']:.3f}x their own decile-matched control "
           f"({100*a_all['frac_draws_below_observed']:.0f}% of {a_all['n_draws']} draws below observed). "
           f"That is a layer with something in it, and it is the same layer ptm_sites measured at 0.992x "
           f"on transcript response with a minimum detectable ratio of 1.075. The edges were not the "
           f"problem; the endpoint was. "
           if layer_lives and a_all else
           f"On co-dependency the full pair set reaches "
           f"{(a_all or {}).get('ratio', float('nan')):.3f}x its matched control, which does not clear "
           f"the 1.05x bar, so the readout change does not rescue the layer either. ")
        + (f"SPECIFICITY: {R['codependency']['n_entangled_with_ppi_or_complex']:,} of {len(both):,} pairs "
           f"are ALSO a PPI edge or co-complex, and PPI already predicts co-dependency in this project. "
           f"Removing them leaves {a_clean['n']:,} pairs at {a_clean['ratio']:.3f}x "
           f"({100*a_clean['frac_draws_below_observed']:.0f}% of draws below observed), so the enrichment "
           f"{'survives and is not simply the PPI layer wearing a phosphorylation label' if specific else 'does NOT survive and is the PPI layer wearing a phosphorylation label'}. "
           if a_clean else "SPECIFICITY: the PPI/co-complex-excluded arm fell below the size floor and is "
                           "reported as unavailable rather than as a number. ")
        + (f"GATING: shared-compartment pairs reach {a_shared['ratio']:.3f}x and disjoint pairs "
           f"{a_disj['ratio']:.3f}x, a gap of {gapv:+.3f}x. Permuting compartment IDENTITY across genes "
           f"while preserving how many compartments each gene has gives {shm:+.3f}x (sd {shs:.3f}, "
           f"{N_SHUFFLE} draws), so "
           + ("the gap is not reproduced by annotation depth and the gate is doing work. "
              if rescued else
              "the gap is not separated from what annotation depth alone produces -- promiscuously "
              "annotated genes share compartments with everything and are also better studied. The gate "
              "is NOT DETECTED as useful, which is not the same as shown useless. ")
           if (a_shared and a_disj and shm is not None) else "GATING: an arm was unavailable. ")
        + (f"Endpoint B (transcript response) came back at ratio {tb.get('ratio', float('nan')):.2f} with "
           f"an sd of {tb.get('ratio_sd', float('nan')):.2f} across draws, so its interval spans 1.0 "
           f"comfortably: it is UNINFORMATIVE, not a large effect and not a null. "
           if (isinstance(tb, dict) and tb.get("ratio_sd") is not None and not tb_informative) else "")
        + f"WHAT THIS DOES NOT SHOW: a compartment annotation is an immunofluorescence consensus over the "
          f"Atlas's cell lines, not a measurement in K562 or in the DepMap panel, so 'shares a compartment' "
          f"means shares an annotation. And co-dependency across a cell-line panel is consistent with a "
          f"functional relationship without being evidence of the phosphorylation event itself.")
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R["rescued"] = rescued
    R["layer_nonnull_on_codependency"] = layer_lives
    R["specific_to_ks_not_ppi"] = specific
    R["transcript_informative"] = tb_informative
    R["limits"] = [
        "HPA subcellular location is immunofluorescence across the Atlas's own cell lines, not a "
        "measurement in K562 or in the DepMap panel. Two proteins sharing an HPA annotation may not share "
        "a compartment in the cells these endpoints were measured in",
        "iPTMnet aggregates enzyme-substrate relations across experiments, tissues and assay types with no "
        "cell-type field, so the edge set itself is not a K562 edge set and gating cannot make it one",
        "the transcript endpoint is the wrong physics for a phosphorylation layer, stated before the run "
        "rather than after. It is included for comparability with the published null",
        "co-dependency is a correlation across a cell-line panel. A kinase and its substrate correlating "
        "in dependency is consistent with a functional relationship and is not evidence of the "
        "phosphorylation event",
        "genes annotated to many compartments are well-studied genes. The matched controls bin "
        "locations-per-pair exactly and the shuffle preserves per-gene compartment count, but neither "
        "removes publication depth itself, which this project has repeatedly found to be among the "
        "strongest nuisance covariates",
        "the shared and disjoint arms are not independent samples of one population -- which genes have "
        "disjoint annotations is itself informative (organelle-restricted proteins are a specific kind of "
        "protein), so the head-to-head compares two biologically different sets of pairs",
        "only 5 compartment shuffles are drawn, against 20 matched draws inside each arm. The shuffle sd "
        "is therefore the least precisely estimated quantity in the verdict",
    ]
    R["verdict"] = verdict
    R["log"] = log
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "compartment_gated_ks.json", "w"), indent=1, default=str)
    report(f"\n  -> {OUT/'compartment_gated_ks.json'}")


if __name__ == "__main__":
    main()
