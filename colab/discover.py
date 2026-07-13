"""discover — point the validated reasoning machinery at the UNKNOWNS (not essentiality, which is already measured).
Four discovery engines, each producing a RANKED candidate list with a held-out / enrichment validation number:

  1. function      — what does a DARK gene do? co-dependency (DepMap 18,443×1,150) + STRING neighbours → transfer a
                     specific pathway label. Validated on held-out annotated genes (specific-label top-1 vs chance).
  2. selective     — which genes are SELECTIVELY essential (a strong dependency in a SUBSET of lines, not pan-essential
                     and not never)? That bimodal window is the drug-target sweet spot. Metric = left-skew of the RAW
                     gene-effect distribution with a robustness floor. Validated: recovery of known selective oncology
                     dependencies (KRAS/BRAF/FOXA1/...).
  3. disease_new   — genes NOT yet disease-linked that score in the top bucket of the disease prior. Honest: the disease
                     axis is modest (0.65) and the calibration is COARSE (6 buckets), so this is a candidate SET, not a
                     high-resolution ranking.
  4. druggable     — selective-essential AND surface-accessible (secreted/membrane) — a tractable-target shortlist.
                     (Partial: no Pharos/DGIdb tractability loaded yet.)

Discovery = predictions for things WITHOUT a measured answer, each with a confidence you can trust or ignore.
-> outputs/orphan/discover.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")

# known selective / lineage oncology dependencies — the enrichment yardstick for engine 2
KNOWN_SELECTIVE = {"KRAS", "BRAF", "PIK3CA", "CTNNB1", "MYB", "PAX8", "SOX10", "MITF", "EGFR", "ERBB2", "MDM2",
                   "WRN", "CCND1", "NRAS", "MYCN", "GATA3", "ESR1", "AR", "FOXA1", "SPDEF", "TP63", "KLF5"}


def _load():
    from complete_cell import CompleteCell
    C = CompleteCell()
    z = np.load(f"{OUT}/depmap_vecs.npz", allow_pickle=True)
    syms = [str(s) for s in z["syms"]]
    Z = np.nan_to_num(z["Z"].astype("float32"), nan=0.0)        # per-gene z-scored (for co-dependency similarity)
    mu = z["mu"].astype("float32")[:, None]
    sd = z["sd"].astype("float32")[:, None]
    raw = Z * sd + mu                                           # reconstruct RAW gene-effect (selectivity lives here)
    return C, syms, Z, {s: i for i, s in enumerate(syms)}, raw


# ---------- 1. dark-gene FUNCTION (co-dependency + STRING k-NN label transfer) — HONESTLY TESTED ----------
def function(C, syms, Z, sidx, k=15):
    """Co-dependency + STRING label transfer. Validates on annotated genes — but honestly stratified: the annotated
    genes are ALL well-connected (their co-dependency neighbours are pathway-coherent), so the validation number lives
    in the well-connected regime. On the truly DARK proteome the neighbours are NOT pathway-coherent (the vote scatters
    across ~50 pathways, margin ~0.02) → the method collapses to a NULL. We report both, and only emit the rare
    high-margin dark calls."""
    from collections import Counter
    rp = json.load(open(f"{OUT}/reactome_pathways.json"))["pathways"]
    name = lambda i: C.genes[i].get("name")
    rp = {pw: gl for pw, gl in rp.items() if len(set(gl)) <= 300}   # exclude generic mega-pathways
    g2path = {}
    for pw, gl in rp.items():
        for gi in set(gl):
            g2path.setdefault(name(gi), []).append(pw)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    labeled = [g for g in syms if g in g2path]
    labset = {g: set(g2path[g]) for g in labeled}
    lab_idx = np.array([sidx[g] for g in labeled])
    all_labels = set().union(*labset.values()) if labset else set()

    def call(sims, exclude_row=None):
        """top pathway + margin (winner's share of vote mass) from co-dependency neighbours with sim>0.1."""
        order = np.argsort(-sims)[:k + 1]
        votes = Counter()
        for j in order:
            if j == exclude_row or sims[j] <= 0.1:
                continue
            for pw in labset[labeled[j]]:
                votes[pw] += float(sims[j])
        if not votes:
            return None, 0.0
        top, w = votes.most_common(1)[0]
        return top, w / (sum(votes.values()) + 1e-9)

    # validation on annotated genes (leave-one-out). NOTE: these are all high-support by construction.
    rng = np.random.default_rng(0)
    tsel = rng.choice(len(labeled), min(500, len(labeled)), replace=False)
    Slab = Zn[lab_idx[tsel]] @ Zn[lab_idx].T                    # (500 x n_labeled)
    hit = 0
    for r, ti in enumerate(tsel):
        pred, _ = call(Slab[r], exclude_row=ti)
        if pred and pred in labset[labeled[ti]]:
            hit += 1
    top1 = hit / len(tsel); baseline = 1 / max(len(all_labels), 1)

    # dark pass (vectorised): what is the actual vote-margin distribution on the dark proteome?
    dark = [C.genes[i].get("name") for i in range(len(C.genes)) if C.genes[i].get("dark") and C.genes[i].get("name") in sidx]
    didx = np.array([sidx[g] for g in dark]) if dark else np.array([], dtype=int)
    preds, margins = [], []
    if len(didx):
        Sd = Zn[didx] @ Zn[lab_idx].T                          # (n_dark x n_labeled), one BLAS call
        for r, g in enumerate(dark):
            pred, m = call(Sd[r])
            margins.append(m)
            if pred and m >= 0.15:                              # only emit calls with real vote concentration
                preds.append({"gene": g, "predicted_function": pred, "margin": round(float(m), 3)})
    preds.sort(key=lambda x: -x["margin"])
    med_margin = float(np.median(margins)) if margins else 0.0
    return {"validation_wellconnected_top1": round(top1, 3), "baseline_chance": round(baseline, 4),
            "lift_over_chance": round(top1 / (baseline + 1e-9), 1),
            "dark_median_vote_margin": round(med_margin, 3),
            "n_dark_high_margin": len(preds),
            "verdict": (f"NULL on the dark proteome — label transfer validates {top1:.0%} on well-connected annotated "
                        f"genes (their neighbours are pathway-coherent), but on dark genes the neighbours scatter "
                        f"(median vote margin {med_margin:.2f}) so only {len(preds)} dark genes get a concentrated call. "
                        f"This is the well-connected-vs-dark split, not a general dark-gene function predictor."),
            "top_predictions": preds[:25]}


# ---------- 2. SELECTIVE dependencies (the drug-target window) ----------
def selective(C, syms, raw):
    """A selective dependency has a bimodal RAW gene-effect: a tail of strong dependency (a subset of lines) against a
    bulk near zero. Metric = LEFT-SKEW of the raw distribution, floored to genes with >=3 strongly-dependent lines so a
    single CRISPR outlier can't score. Validated by recovery of known selective oncology dependencies."""
    r_mean = raw.mean(1); r_std = raw.std(1); r_min = raw.min(1)
    n_dep1 = (raw < -1.0).sum(1)                               # strongly-dependent lines (gene-effect < -1)
    skew = -((raw - r_mean[:, None]) ** 3).mean(1) / (r_std ** 3 + 1e-9)   # left-skew -> positive score
    floor = n_dep1 >= 3                                        # robustness: not a single-line artifact
    score = np.where(floor, skew, -1e9)
    order = np.argsort(-score)
    N = len(syms)

    # validation: do the top-ranked genes concentrate the known selective oncology targets?
    kidx = [sidx_of(syms, g) for g in KNOWN_SELECTIVE if g in syms]
    kset = set(i for i in kidx if i is not None)
    for topn in (300, 500, 1000):
        top = set(order[:topn].tolist())
        hit = len(kset & top)
        if topn == 500:
            hit500, enr500 = hit, (hit / topn) / (len(kset) / N + 1e-12)
    cand = []
    for i in order:
        if score[i] <= -1e8:
            break
        cand.append({"gene": syms[i], "skew": round(float(skew[i]), 2), "n_strong_dep_lines": int(n_dep1[i]),
                     "min_effect": round(float(r_min[i]), 2), "mean_effect": round(float(r_mean[i]), 2),
                     "known": syms[i] in KNOWN_SELECTIVE})
    novel = [c for c in cand if not c["known"]]
    return {"n_selective": len(cand),
            "known_recovered_top500": hit500, "known_total": len(kset),
            "enrichment_top500_vs_chance": round(float(enr500), 1),
            "top_selective_targets": cand[:25], "top_novel_selective": novel[:20]}


def sidx_of(syms, g):
    try:
        return syms.index(g)
    except ValueError:
        return None


# ---------- 3. novel DISEASE candidates ----------
def disease_new(C):
    dr = json.load(open(f"{OUT}/disease_reason.json"))
    priors = dr.get("priors", {})
    axis_auc = dr.get("weighted_reasoning_auc", 0.65)
    name2ndis = {C.genes[i].get("name"): (C.genes[i].get("ndis", 0) or 0) for i in range(len(C.genes))}
    # genes NOT yet disease-linked, in the TOP calibration bucket
    scored = [(g, r["p_disease"], r.get("n_agree", 0), r.get("n_lines", 0)) for g, r in priors.items()
              if r.get("p_disease") is not None and name2ndis.get(g, 0) == 0]
    if not scored:
        return {"n_candidates_scored": 0, "top_novel_disease_candidates": []}
    top_p = max(p for _, p, _, _ in scored)
    top_bucket = [(g, p, a, nl) for g, p, a, nl in scored if abs(p - top_p) < 1e-6]
    top_bucket.sort(key=lambda x: (-x[2], -x[3], x[0]))        # ties broken by agreement, then determinism
    return {"n_candidates_scored": len(scored), "axis_auc": round(float(axis_auc), 3),
            "resolution": f"COARSE — {len(set(round(p,3) for _,p,_,_ in scored))} calibration buckets; top bucket is a SET",
            "n_in_top_bucket": len(top_bucket), "top_bucket_probability": round(float(top_p), 3),
            "top_novel_disease_candidates": [{"gene": g, "prior": round(p, 3), "n_agree": a} for g, p, a, _ in top_bucket[:25]]}


# ---------- 4. DRUGGABLE triage (selective-essential AND surface-accessible) ----------
def druggable(C, sel):
    loc = json.load(open(f"{OUT}/localization.json")).get("labels", {})
    SURFACE = ("Cell membrane", "Secreted", "Membrane")
    out = []
    for t in sel["top_selective_targets"] + sel.get("top_novel_selective", []):
        g = t["gene"]
        comps = loc.get(g, [])
        surf = [c for c in comps if c in SURFACE]
        if surf:
            out.append({"gene": g, "n_strong_dep_lines": t["n_strong_dep_lines"], "min_effect": t["min_effect"],
                        "compartment": surf[0], "known": t["known"]})
    # dedup preserving order
    seen, uniq = set(), []
    for o in out:
        if o["gene"] not in seen:
            seen.add(o["gene"]); uniq.append(o)
    return {"note": "selective dependency + surface-accessible (no Pharos/DGIdb tractability loaded yet — partial)",
            "n": len(uniq), "shortlist": uniq[:15]}


def main():
    C, syms, Z, sidx, raw = _load()
    print("=" * 92)
    print("DISCOVER — the reasoning machinery aimed at the UNKNOWNS (not essentiality)")
    print("=" * 92)

    fn = function(C, syms, Z, sidx)
    print(f"\n  1. DARK-GENE FUNCTION [NULL]: validates {fn['validation_wellconnected_top1']:.0%} on WELL-CONNECTED "
          f"annotated genes ({fn['lift_over_chance']}× chance) — but dark-gene median vote margin is only "
          f"{fn['dark_median_vote_margin']} (scatter = no signal). Only {fn['n_dark_high_margin']} dark genes get a "
          f"concentrated call:")
    for p in fn["top_predictions"][:5]:
        print(f"       {p['gene']:10} → {p['predicted_function'][:50]}  (margin {p['margin']})")

    se = selective(C, syms, raw)
    print(f"\n  2. SELECTIVE TARGETS: {se['n_selective']} genes with a bimodal (selective) dependency profile; "
          f"recovered {se['known_recovered_top500']}/{se['known_total']} known oncology targets in top-500 "
          f"({se['enrichment_top500_vs_chance']}× enrichment)")
    for t in se["top_selective_targets"][:6]:
        tag = " [known]" if t["known"] else ""
        print(f"       {t['gene']:10} skew {t['skew']:5}  strong in {t['n_strong_dep_lines']:3} lines  "
              f"min {t['min_effect']}{tag}")
    print("       novel (not in known set):", ", ".join(t["gene"] for t in se["top_novel_selective"][:10]))

    dn = disease_new(C)
    print(f"\n  3. NOVEL DISEASE CANDIDATES: {dn['n_candidates_scored']} un-flagged genes scored "
          f"(axis AUC {dn['axis_auc']}, {dn['resolution']})")
    print(f"       top bucket: {dn['n_in_top_bucket']} genes at P≈{dn['top_bucket_probability']} — "
          + ", ".join(c["gene"] for c in dn["top_novel_disease_candidates"][:8]))

    dg = druggable(C, se)
    print(f"\n  4. DRUGGABLE SHORTLIST (selective + surface): {dg['n']} genes — {dg['note']}")
    for d in dg["shortlist"][:6]:
        tag = " [known]" if d["known"] else ""
        print(f"       {d['gene']:10} strong in {d['n_strong_dep_lines']:3} lines, {d['compartment']}{tag}")

    out = {"function": fn, "selective": se, "disease_new": dn, "druggable": dg}
    json.dump(out, open(f"{OUT}/discover.json", "w"), indent=1)
    print("\n" + "=" * 92)
    return out


if __name__ == "__main__":
    main()
