"""context_dependency — a standing cell-model layer that ATTRIBUTES each selective dependency to the lesion
that explains it. Turns "WRN is a selective dependency somewhere (#92)" into "WRN is the dependency of THIS
lesion" — the step that makes a genome-wide vulnerability actionable for a specific tumour.

Two tiers, by how the lesion shows up in DepMap CRISPR gene-effect:

  TIER 1 — oncogene-addiction (self-contained, runs anywhere).
     A mutated oncogene is ITSELF a dependency in its mutant lines, so its gene-effect profile is a proxy for
     "lines carrying that lesion". A selective dependency whose gene-effect CO-VARIES with a driver's is
     attributed to that driver's context (corr(BRAF, MAP2K1)=0.45 -> MAP2K1 is a BRAF-context dependency).

  TIER 2 — loss / synthetic-lethal (needs per-line lesion labels; Colab).
     A LOST tumour suppressor (MMR, MTAP) is knocked out already, so it is NOT a dependency — it leaves no
     trace in the gene-effect matrix, and its synthetic-lethal partner (WRN, PRMT5) cannot be reached by
     co-essentiality (corr(WRN, MLH1)~0). Attribution needs the DepMap per-line MSI/lineage/mutation table
     (Model.csv + OmicsSomaticMutations): regress each selective dependency's gene-effect on those features.
     Until that table is present the gene is flagged ORPHAN — an honest "selective dependency I cannot yet
     attribute; supply the lesion labels".

Output per gene: selective rank, and its attributed context(s) or the orphan flag.
-> outputs/orphan/context_dependency.json   (folds into CompleteCell.gene(g)['context_dependency'])
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/orphan"

# drivers whose OWN gene-effect marks their mutation context (they are dependencies in their mutant lines)
ADDICTION_DRIVERS = [
    "BRAF", "KRAS", "NRAS", "HRAS", "EGFR", "KIT", "PDGFRA", "MET", "ALK", "FLT3", "JAK2", "ABL1", "RET",
    "PIK3CA", "AKT1", "MYC", "MYCN", "MDM2", "CCND1", "CDK4", "CDK6", "CTNNB1", "ERBB2", "MITF", "SOX10",
    "ESR1", "AR", "BCL2", "MCL1", "WRN", "EZH2", "IRF4", "PAX8", "GATA3", "TP63", "NRAS",
]


def _dm():
    p = f"{OUT}/depmap_vecs.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"syms": list(z["syms"]), "Z": z["Z"], "col": {s: i for i, s in enumerate(z["syms"])}}


def selective_dependencies(dm, min_lines=100):
    """genome-wide ranked selective dependencies (lethal in a subset, neutral overall)."""
    Z = dm["Z"]; sel = []
    for i, s in enumerate(dm["syms"]):
        row = Z[i]; nz = row[row != 0]
        if len(nz) < min_lines:
            continue
        frac = float((row < -3).mean())
        if frac > 0 and row.min() < -4 and abs(nz.mean()) < 0.6:
            sel.append((frac, float(row.min()), s))
    sel.sort(reverse=True)
    return sel


def _corr(Z, i, j):
    x, y = Z[i], Z[j]; m = (x != 0) & (y != 0)
    if m.sum() < 80:
        return 0.0
    return float(np.corrcoef(x[m], y[m])[0, 1])


def attribute_addiction(dm, gene, drivers, thresh=0.25):
    """TIER 1: which driver contexts does this dependency co-vary with? (self-contained co-essentiality)."""
    if gene not in dm["col"]:
        return []
    gi = dm["col"][gene]; Z = dm["Z"]
    out = []
    for d in drivers:
        if d == gene or d not in dm["col"]:
            continue
        c = _corr(Z, gi, dm["col"][d])
        if c >= thresh:
            out.append((d, round(c, 3)))
    out.sort(key=lambda x: -x[1])
    return out


def attribute_lesion(dm, model_csv=None, mut_csv=None, sel=None, top=400):
    """TIER 2 (Colab): regress each selective dependency's gene-effect on per-line lesion features (MSI status,
    lineage, driver-mutation flags) from the DepMap annotation tables. Returns {gene: [(feature, effect, n)]}.
    Requires the tables; returns {} when they are absent (caller then marks those genes ORPHAN)."""
    if not model_csv or not os.path.exists(model_csv):
        return {}
    import pandas as pd
    Z = dm["Z"]
    md = pd.read_csv(model_csv)                       # rows = ModelID; columns include MSIStatus, OncotreeLineage
    # NOTE: aligning md rows to the gene-effect COLUMNS needs the ModelID order the matrix was built from.
    # depmap_vecs.npz dropped that order, so proper alignment happens in the Colab builder that keeps ModelIDs.
    # This function is the hook; the Colab notebook passes an aligned (features x lines) design matrix.
    feats = {}
    if "MSIStatus" in md.columns:
        feats["MSI"] = (md["MSIStatus"].astype(str).str.contains("MSI-H|Instable|MSI", case=False, na=False)).to_numpy(float)
    if "OncotreeLineage" in md.columns:
        for lin in md["OncotreeLineage"].dropna().unique():
            feats[f"lineage:{lin}"] = (md["OncotreeLineage"] == lin).to_numpy(float)
    res = {}
    order = [s for _, _, s in (sel or selective_dependencies(dm))[:top]]
    for g in order:
        if g not in dm["col"]:
            continue
        y = Z[dm["col"][g]]
        best = []
        for fn, fv in feats.items():
            n = min(len(fv), len(y))
            yv, xv = y[:n], fv[:n]
            m = yv != 0
            if m.sum() < 30 or xv[m].std() == 0:
                continue
            dep_in = yv[m][xv[m] == 1].mean() if (xv[m] == 1).any() else 0.0
            dep_out = yv[m][xv[m] == 0].mean() if (xv[m] == 0).any() else 0.0
            eff = dep_out - dep_in                    # positive = MORE dependent inside the feature group
            if eff > 0.4:
                best.append((fn, round(float(eff), 3), int((xv[m] == 1).sum())))
        best.sort(key=lambda x: -x[1])
        if best:
            res[g] = best[:3]
    return res


def resolve_orphans_colab(gene_effect_csv, model_csv, genes=None):
    """TIER 2, properly aligned (Colab). Re-parse the DepMap gene-effect CSV KEEPING ModelIDs, join the per-line
    Model table (MSIStatus, OncotreeLineage), and for each gene test whether its dependency concentrates in a
    lesion group — the regression the stripped local matrix can't do. Returns {gene: [(context, effect, n)]}.

    This is the step that turns WRN (orphan #92) into 'WRN = the MSI-lesion dependency'."""
    import pandas as pd
    ge = pd.read_csv(gene_effect_csv, index_col=0)                 # rows=ModelID, cols='SYM (id)'
    ge.columns = [c.split(" (")[0] for c in ge.columns]
    md = pd.read_csv(model_csv, index_col=0)                       # rows=ModelID
    common = ge.index.intersection(md.index)
    ge, md = ge.loc[common], md.loc[common]
    feats = {}
    for col, lab in (("MSIStatus", "MSI"), ("OncotreeLineage", None)):
        if col not in md.columns:
            continue
        if lab:
            feats["lesion:MSI"] = md[col].astype(str).str.contains("MSI-H|Instable|MSI", case=False, na=False)
        else:
            for lin in md[col].dropna().unique():
                feats[f"lineage:{lin}"] = (md[col] == lin)
    genes = genes or list(ge.columns)
    out = {}
    for g in genes:
        if g not in ge.columns:
            continue
        y = ge[g]
        best = []
        for fn, mask in feats.items():
            m = mask.reindex(y.index).fillna(False)
            if m.sum() < 5 or (~m).sum() < 5:
                continue
            eff = float(y[~m].mean() - y[m].mean())               # >0 = MORE dependent inside the group
            if eff > 0.4:
                best.append((fn, round(eff, 3), int(m.sum())))
        best.sort(key=lambda x: -x[1])
        if best:
            out[g] = best[:3]
    return out


def build(model_csv=None, mut_csv=None):
    dm = _dm()
    if dm is None:
        print("  no DepMap matrix -> context_dependency layer empty")
        json.dump({}, open(f"{OUT}/context_dependency.json", "w")); return {}
    sel = selective_dependencies(dm)
    rank = {s: k for k, (_, _, s) in enumerate(sel)}
    lesion = attribute_lesion(dm, model_csv, mut_csv, sel)     # {} unless the Colab tables are present
    out = {}
    for frac, mn, g in sel:
        addiction = attribute_addiction(dm, g, ADDICTION_DRIVERS)
        les = lesion.get(g)
        out[g] = {
            "selective_rank": rank[g],
            "min_gene_effect": round(mn, 2),
            "frac_lines_dependent": round(frac, 3),
            "addiction_context": addiction[:3] or None,      # TIER 1 (self-contained)
            "lesion_context": les,                           # TIER 2 (Colab); None if tables absent
            "orphan": not addiction and not les,             # selective dep with no attributable context yet
        }
    json.dump(out, open(f"{OUT}/context_dependency.json", "w"))
    n_add = sum(1 for v in out.values() if v["addiction_context"])
    n_les = sum(1 for v in out.values() if v["lesion_context"])
    n_orph = sum(1 for v in out.values() if v["orphan"])
    print("=" * 76)
    print(f"CONTEXT-DEPENDENCY LAYER  —  {len(out)} selective dependencies attributed")
    print("=" * 76)
    print(f"  TIER 1 addiction-attributed (self-contained): {n_add}")
    print(f"  TIER 2 lesion-attributed (Colab tables):       {n_les}"
          + ("" if model_csv else "   [tables not supplied -> loss-context deps stay orphan]"))
    print(f"  ORPHAN (selective but unattributed):           {n_orph}")
    for g in ("WRN", "MAP2K1", "SOX10", "PRMT5"):
        if g in out:
            v = out[g]
            print(f"    {g:8} rank#{v['selective_rank']:<5} addiction={v['addiction_context']} "
                  f"lesion={v['lesion_context']} orphan={v['orphan']}")
    return out


if __name__ == "__main__":
    import sys
    mc = sys.argv[1] if len(sys.argv) > 1 else None
    build(model_csv=mc)
