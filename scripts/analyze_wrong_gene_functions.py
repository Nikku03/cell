"""Functional + universality breakdown of FP/FN errors.

Answers two questions on top of analyze_loo_errors.py:

  1. How many of the wrongly predicted genes are "universal" -- present
     across most organisms (family_n_organisms high) AND consistently
     essential across organisms that have them?

  2. How many of the wrongs are core-housekeeping function genes
     (ribosome, RNA pol, DNA replication, aminoacyl-tRNA synthetases,
     translation factors, ATP synthase, cell division, etc.)?

The expectation, if the model is doing its job:
  - Wrongs concentrate in the long tail of organism-specific accessory
    genes, not in the universal-essential core.
  - If we are wrong on rps*/rpl*/dnaA/rpoB/ileS-class genes, that is
    a serious failure -- those are the genes the family prior should
    nail.

Inputs:
  loo_predictions.csv  (from analyze_loo_errors.py)
  labels.csv           (for gene_name)

Outputs:
  reliability_function_breakdown.json
    per-category and per-universality FP/FN counts and rates
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
PRED_CSV   = DATA_DIR / "loo_predictions.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
OUT_JSON   = DATA_DIR / "reliability_function_breakdown.json"


# Gene-name -> functional category. Order matters (first match wins).
# Patterns are case-insensitive, matched against gene_name as a whole word
# or as a prefix (locked by "starts-with" semantics below).
CATEGORIES: list[tuple[str, list[str]]] = [
    ("ribosome_LSU",          [r"^rpl[a-z]?\d*$", r"^rpm[a-z]\d*$"]),
    ("ribosome_SSU",          [r"^rps[a-z]?\d*$"]),
    ("rRNA",                  [r"^rrf$", r"^rrl$", r"^rrs$", r"^rrn[a-z]$"]),
    ("tRNA_synthetase",       [r"^(ile|leu|val|ala|gly|asp|asn|gln|glu|"
                                r"lys|arg|his|pro|tyr|trp|phe|met|ser|"
                                r"thr|cys)[srtq]?[s]$",
                                r"^pheT$", r"^glyQ$", r"^metG$"]),
    ("RNA_polymerase",        [r"^rpo[a-z]\d*$"]),
    ("DNA_replication",       [r"^dna[a-z]$", r"^pol[abc]$", r"^lig[a-z]$",
                                r"^pri[a-z]$", r"^ssb$", r"^rnh[a-z]$",
                                r"^holA$", r"^holB$", r"^holC$"]),
    ("topoisomerase",         [r"^gyr[ab]$", r"^par[cdef]$", r"^topo?[abc]$",
                                r"^topB$"]),
    ("translation_factor",    [r"^tuf[a-z]?$", r"^tsf$", r"^fus[a-z]?$",
                                r"^inf[abc]$", r"^prf[abcl]$", r"^frr$",
                                r"^efp$", r"^lepA$", r"^typA$"]),
    ("transcription_factor",  [r"^nus[abeg]$", r"^gre[ab]$", r"^mfd$",
                                r"^rho$"]),
    ("cell_division",         [r"^fts[a-z]$", r"^zip[a-z]$", r"^mre[a-z]$",
                                r"^min[a-z]$", r"^mip[a-z]$", r"^mur[a-g]$",
                                r"^damX$"]),
    ("cell_wall_PG",          [r"^mra[yj]$", r"^pbp[a-z]\d*$", r"^lpx[a-z]$",
                                r"^lpp$", r"^ddl[ab]$", r"^glm[susm]$",
                                r"^bam[a-e]$", r"^lpt[a-g]$"]),
    ("chaperone_export",      [r"^gro[esl]l?$", r"^dna[kj]$", r"^grpE$",
                                r"^hsl[uv]$", r"^lon$", r"^clp[a-z]$",
                                r"^htp[a-z]$", r"^sec[a-z]\d*$", r"^ffh$",
                                r"^fts[yk]$", r"^yidC$", r"^tig$"]),
    ("ATP_synthase",          [r"^atp[a-z]$"]),
    ("nucleotide_metab",      [r"^pur[a-z]$", r"^pyr[a-z]$", r"^gua[ab]$",
                                r"^gmk$", r"^ndk$", r"^adk$", r"^dut$",
                                r"^thy[a-z]$", r"^fol[a-z]$", r"^cmk$",
                                r"^tmk$"]),
    ("FA_lipid_biosynth",     [r"^fab[a-z]$", r"^acc[a-d]$", r"^acp[psh]?$",
                                r"^pls[bcxy]$", r"^lgt$", r"^lspA$",
                                r"^lol[a-e]$", r"^plsB$"]),
    ("glycolysis_central",    [r"^pgi$", r"^pfk[ab]$", r"^fba[ab]$",
                                r"^tpi[a-z]$", r"^gap[a-z]$", r"^pgk$",
                                r"^gpm[ab]$", r"^eno$", r"^pyk[a-z]$"]),
]


def classify(name: str | None) -> str:
    if not isinstance(name, str) or not name:
        return "uncategorized"
    n = name.strip().lower()
    for cat, pats in CATEGORIES:
        for p in pats:
            if re.match(p, n):
                return cat
    return "other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv",   type=Path, default=PRED_CSV)
    p.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    p.add_argument("--out",        type=Path, default=OUT_JSON)
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    print("Loading ...")
    pred   = pd.read_csv(args.pred_csv)
    labels = pd.read_csv(args.labels_csv,
                         usecols=["organism", "locus_tag", "gene_name"])
    print(f"  predictions: {len(pred):>7d}")
    print(f"  labels:      {len(labels):>7d}")

    df = pred.merge(labels, on=["organism", "locus_tag"], how="left")
    n_named = int(df.gene_name.notna().sum())
    print(f"  joined:      {len(df):>7d}  ({n_named} with gene_name, "
          f"{len(df)-n_named} unnamed)")

    df["category"] = df.gene_name.apply(classify)
    df["error_class"] = df.apply(
        lambda r: "TP" if (r.y_pred==1 and r.y_true==1) else
                  "FN" if (r.y_pred==0 and r.y_true==1) else
                  "FP" if (r.y_pred==1 and r.y_true==0) else "TN", axis=1)

    # ---- universality bands ----
    def uni_bin(n):
        if n <= 1:  return "1 org (organism-specific)"
        if n <= 5:  return "2-5 orgs (narrow)"
        if n <= 15: return "6-15 orgs (clade-level)"
        if n <= 30: return "16-30 orgs (broad)"
        return "31+ orgs (universal)"
    df["universality"] = df.family_n_organisms.apply(uni_bin)

    # ============ Q1: universality of wrong predictions ============
    print(f"\n=== Q1: how universal are the wrongly-predicted genes? ===")
    print(f"{'band':<30s}  {'n':>8s}  {'FP':>6s}  {'FN':>6s}  "
          f"{'FP%row':>7s}  {'FN%row':>7s}  {'%of_FP':>7s}  {'%of_FN':>7s}")
    by_uni = {}
    total_fp = int((df.error_class=="FP").sum())
    total_fn = int((df.error_class=="FN").sum())
    order = ["1 org (organism-specific)", "2-5 orgs (narrow)",
             "6-15 orgs (clade-level)", "16-30 orgs (broad)",
             "31+ orgs (universal)"]
    for band in order:
        sub = df[df.universality == band]
        cm  = sub.error_class.value_counts().to_dict()
        fp  = cm.get("FP", 0); fn = cm.get("FN", 0)
        tp  = cm.get("TP", 0); tn = cm.get("TN", 0)
        fp_rate = 100*fp / max(fp+tn, 1)
        fn_rate = 100*fn / max(fn+tp, 1)
        share_fp = 100*fp / max(total_fp, 1)
        share_fn = 100*fn / max(total_fn, 1)
        print(f"  {band:<28s}  {len(sub):>8d}  {fp:>6d}  {fn:>6d}  "
              f"{fp_rate:>6.1f}%  {fn_rate:>6.1f}%  "
              f"{share_fp:>6.1f}%  {share_fn:>6.1f}%")
        by_uni[band] = {"n":len(sub), "TP":tp, "FP":fp, "TN":tn, "FN":fn,
                          "FP_rate":fp_rate/100, "FN_rate":fn_rate/100,
                          "share_of_total_FP":share_fp/100,
                          "share_of_total_FN":share_fn/100}

    # ---- "is this a TRULY universal essential?" ----
    truly_uni = df[(df.family_n_organisms >= 31) &
                   (df.family_frac_essential_leakfree >= 0.8)]
    tu_cm = truly_uni.error_class.value_counts().to_dict()
    print(f"\n  TRULY UNIVERSAL ESSENTIAL  (>=31 orgs & frac>=0.8): "
          f"n={len(truly_uni)}")
    print(f"    TP={tu_cm.get('TP',0):>6d}  FP={tu_cm.get('FP',0):>4d}  "
          f"TN={tu_cm.get('TN',0):>4d}  FN={tu_cm.get('FN',0):>4d}")
    print(f"    recall on these = "
          f"{100*tu_cm.get('TP',0)/max(tu_cm.get('TP',0)+tu_cm.get('FN',0),1):.1f}%")

    # ============ Q2: by functional category ============
    print(f"\n=== Q2: FP/FN by functional category ===")
    print(f"{'category':<24s}  {'n':>7s}  {'n_ess':>7s}  "
          f"{'FP':>5s}  {'FN':>5s}  {'recall':>7s}  {'precision':>9s}")
    by_cat = {}
    cat_order = [c for c,_ in CATEGORIES] + ["other", "uncategorized"]
    for cat in cat_order:
        sub = df[df.category == cat]
        if len(sub) == 0: continue
        cm  = sub.error_class.value_counts().to_dict()
        tp  = cm.get("TP", 0); fp = cm.get("FP", 0)
        tn  = cm.get("TN", 0); fn = cm.get("FN", 0)
        n_ess = int(sub.y_true.sum())
        rec  = tp / max(tp+fn, 1)
        prec = tp / max(tp+fp, 1)
        print(f"  {cat:<22s}  {len(sub):>7d}  {n_ess:>7d}  "
              f"{fp:>5d}  {fn:>5d}  {rec:>6.1%}  {prec:>8.1%}")
        by_cat[cat] = {"n":len(sub), "n_essential":n_ess,
                         "TP":tp, "FP":fp, "TN":tn, "FN":fn,
                         "recall":rec, "precision":prec}

    # ---- Aggregate: "core housekeeping" = ribosome + tRNA-syn + RNA pol +
    #      DNA replication + topoisomerase + translation + ATP synthase ----
    CORE = {"ribosome_LSU", "ribosome_SSU", "rRNA", "tRNA_synthetase",
            "RNA_polymerase", "DNA_replication", "topoisomerase",
            "translation_factor", "ATP_synthase"}
    core = df[df.category.isin(CORE)]
    cm = core.error_class.value_counts().to_dict()
    print(f"\n  CORE HOUSEKEEPING (ribosome+tRNAsyn+RNApol+DNArep+topo+"
          f"transl+ATPsyn):")
    print(f"    n={len(core)}  n_essential={int(core.y_true.sum())}")
    print(f"    TP={cm.get('TP',0)}  FP={cm.get('FP',0)}  "
          f"TN={cm.get('TN',0)}  FN={cm.get('FN',0)}")
    rec  = cm.get('TP',0) / max(cm.get('TP',0)+cm.get('FN',0), 1)
    prec = cm.get('TP',0) / max(cm.get('TP',0)+cm.get('FP',0), 1)
    print(f"    recall={rec:.1%}  precision={prec:.1%}")

    # ============ Q3: function x universality cross-tab on errors ============
    print(f"\n=== Q3: where do FNs hide? (FN count by category x universality) ===")
    fn_df = df[df.error_class == "FN"]
    pivot = fn_df.pivot_table(index="category", columns="universality",
                                values="locus_tag", aggfunc="count",
                                fill_value=0)
    # reorder cols
    pivot = pivot[[c for c in order if c in pivot.columns]]
    print(pivot.to_string())

    print(f"\n=== Q3': where do FPs come from? (FP count by category x universality) ===")
    fp_df = df[df.error_class == "FP"]
    pivot_fp = fp_df.pivot_table(index="category", columns="universality",
                                  values="locus_tag", aggfunc="count",
                                  fill_value=0)
    pivot_fp = pivot_fp[[c for c in order if c in pivot_fp.columns]]
    print(pivot_fp.to_string())

    # ---- Save ----
    out = {
        "n_predicted":    len(df),
        "n_with_gene_name": n_named,
        "by_universality": by_uni,
        "by_category":     by_cat,
        "core_housekeeping_summary": {
            "n": len(core), "n_essential": int(core.y_true.sum()),
            "TP": cm.get("TP",0), "FP": cm.get("FP",0),
            "TN": cm.get("TN",0), "FN": cm.get("FN",0),
            "recall": rec, "precision": prec,
        },
        "truly_universal_essential": {
            "n": len(truly_uni),
            "TP": int(tu_cm.get("TP",0)), "FP": int(tu_cm.get("FP",0)),
            "TN": int(tu_cm.get("TN",0)), "FN": int(tu_cm.get("FN",0)),
            "recall": tu_cm.get("TP",0) /
                       max(tu_cm.get("TP",0)+tu_cm.get("FN",0), 1),
        },
        "fn_pivot_cat_x_universality": pivot.to_dict(),
        "fp_pivot_cat_x_universality": pivot_fp.to_dict(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
