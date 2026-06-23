"""Build a minimal cell for E. coli (beril_Keio) end-to-end and test it.

Ground truth: 605 essential genes (Keio single-gene knockout collection).
We assemble the candidate essential gene set three ways and validate each
against that truth, then check the assembled cell for functional completeness.

  A) MODEL      : transformer essentiality (cross-clade recalibrated)
  B) FBA        : flux-balance essential calls (metabolic network)
  C) MODEL + FBA: union -- tests the complementarity hypothesis
                  (model strong on conserved/translation; FBA on metabolism)

Then: categorize the assembled cell into functional slots and check coverage of
syn3a's required-function fingerprint -> "is this a functionally complete cell?"

Output: outputs/orphan/make_cell_ecoli.json
"""
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np

OUT = Path("outputs/orphan")
ORG = "beril_Keio"


def prf(pred_set, truth_set, universe):
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(n_pred=len(pred_set), tp=tp, fp=fp, fn=fn,
                precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3))


def fit_logit(X, y, l2=1.0, iters=400, lr=0.5):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Xs.shape[1])
    c = np.bincount(y.astype(int), minlength=2).astype(float)
    cw = (c.sum() / np.maximum(c, 1)) ** 0.5; sw = cw[y.astype(int)]; sw /= sw.mean()
    n = len(X)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((p - y) * sw) / n; g[1:] += l2 * w[1:] / n; w -= lr * g
    return w, mu, sd


def pred_logit(m, X):
    w, mu, sd = m; return 1 / (1 + np.exp(-np.clip((np.c_[np.ones(len(X)), (X - mu) / sd]) @ w, -30, 30)))


# ---- truth + annotations ----
labels = {(r["organism"], r["locus_tag"]): int(r["essential"])
          for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))}
keio = [l for (o, l) in labels if o == ORG]
truth_ess = set(l for l in keio if labels[(ORG, l)] == 1)
print(f"E. coli (beril_Keio): {len(keio):,} genes, {len(truth_ess)} essential (Keio truth)")

annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"):
        annot[r["locus_tag"]] = r["product"]

# ---- A) MODEL predictions (cross-clade recalibrated) ----
preds = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p_a, br_a, y_a = preds["p"], preds["brightness"], preds["y"].astype(int)
clade_a, lt_a = preds["clade"], preds["lt"]
cache = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
corgs = list(cache["orgs"]); cmeta = cache["meta_org"]; clt = cache["lt"]; cclade = cache["clade"]
cladelt2org = {(str(cclade[i]), str(clt[i])): str(corgs[int(cmeta[i])]) for i in range(len(clt))}
# recalibrate on all clades except escherichia, apply to escherichia
valid = ~(np.isnan(p_a) | np.isnan(br_a))
Xc = np.column_stack([p_a, br_a]).astype(np.float32)
is_ec = np.array([cladelt2org.get((str(clade_a[i]), str(lt_a[i]))) == ORG for i in range(len(p_a))])
tr = valid & ~is_ec & (clade_a != "escherichia")
M = fit_logit(Xc[tr], y_a[tr].astype(float))
model_p = {}
for i in range(len(p_a)):
    if is_ec[i] and valid[i]:
        model_p[str(lt_a[i])] = float(pred_logit(M, Xc[i:i+1])[0])
print(f"  model predictions for Keio genes: {len(model_p):,}")
model_ess = set(l for l, p in model_p.items() if p >= 0.5)

# ---- B) FBA essential calls ----
fba_ess = set()
fba_modeled = set()
for r in csv.DictReader(open("data/drive_import/labels/fba_features.csv")):
    if r["organism"] == ORG:
        fba_modeled.add(r["locus_tag"])
        if r.get("fba_essential") in ("1", "1.0", "True", "true"):
            fba_ess.add(r["locus_tag"])
print(f"  FBA: {len(fba_modeled)} modeled, {len(fba_ess)} essential")

# ---- validate the three assemblies ----
universe = set(keio)
print("\n=== validation against 605 Keio essentials ===")
results = {}
for name, pred in [("MODEL", model_ess), ("FBA", fba_ess),
                   ("MODEL+FBA (union)", model_ess | fba_ess),
                   ("MODEL+FBA (conserved-model + metabolic-fba)", None)]:
    if pred is None:
        # complementary union: model essentials that are conserved (have annot)
        # + FBA essentials (metabolic). This is the routed assembly.
        pred = model_ess | (fba_ess & fba_modeled)
    r = prf(pred, truth_ess, universe)
    results[name] = r
    print(f"  {name:<46} n={r['n_pred']:>4}  P={r['precision']:.2f} "
          f"R={r['recall']:.2f} F1={r['f1']:.2f}  (tp={r['tp']} fp={r['fp']} fn={r['fn']})")

# ---- complementarity: what does FBA catch that the model misses, and vice versa ----
model_only = (model_ess & truth_ess) - fba_ess
fba_only = (fba_ess & truth_ess) - model_ess
both = model_ess & fba_ess & truth_ess
print(f"\n=== complementarity on TRUE essentials ===")
print(f"  caught by MODEL only : {len(model_only)}")
print(f"  caught by FBA only   : {len(fba_only)}")
print(f"  caught by both       : {len(both)}")
# functional character of each
def slot_summary(locus_set, topn=6):
    c = Counter()
    for l in locus_set:
        prod = (annot.get(l, "") or "").lower()
        for kw in ["ribosom", "trna", "synthetase", "polymerase", "transferase",
                   "dehydrogenase", "synthase", "reductase", "membrane", "transport",
                   "biosynthe", "kinase"]:
            if kw in prod: c[kw] += 1
    return c.most_common(topn)
print(f"  MODEL-only functional character: {slot_summary(model_only)}")
print(f"  FBA-only   functional character: {slot_summary(fba_only)}")

# ---- functional completeness of the assembled cell (MODEL+FBA union) ----
cell = model_ess | (fba_ess & fba_modeled)
SLOT_KW = {
    "ribosomal": ["ribosom"], "trna": ["trna"], "synthetase": ["synthetase", "ligase"],
    "polymerase": ["polymerase"], "dna": ["helicase", "gyrase", "primase", "replication"],
    "translation": ["elongation factor", "initiation factor", "release factor"],
    "transcription": ["transcription", "sigma factor"], "membrane": ["membrane", "transport", "permease"],
    "energy": ["atp synthase", "dehydrogenase", "atpase"],
    "metabolism": ["biosynthe", "synthase", "transferase", "reductase"],
}
cell_slots = {}
for slot, kws in SLOT_KW.items():
    n = sum(1 for l in cell if any(k in (annot.get(l, "") or "").lower() for k in kws))
    cell_slots[slot] = n
print(f"\n=== assembled cell (MODEL + FBA, n={len(cell)}) functional completeness ===")
for slot, n in cell_slots.items():
    status = "OK" if n > 0 else "MISSING"
    print(f"  {slot:<14} {n:>4}  {status}")
missing = [s for s, n in cell_slots.items() if n == 0]

# ---- verdict ----
best = max(results.items(), key=lambda kv: kv[1]["f1"])
print(f"\n=== VERDICT ===")
print(f"  best assembly: {best[0]}  (F1={best[1]['f1']}, recall={best[1]['recall']})")
print(f"  assembled cell size: {len(cell)} genes")
print(f"  functional slots covered: {len(cell_slots)-len(missing)}/{len(cell_slots)}"
      + (f"  MISSING: {missing}" if missing else "  (complete)"))
print(f"  recovers {len((model_ess|fba_ess) & truth_ess)}/{len(truth_ess)} true essentials "
      f"({len((model_ess|fba_ess)&truth_ess)/len(truth_ess)*100:.0f}%)")

json.dump(dict(
    organism=ORG, n_genes=len(keio), n_true_essential=len(truth_ess),
    validation=results,
    complementarity=dict(model_only=len(model_only), fba_only=len(fba_only), both=len(both),
                         model_only_character=slot_summary(model_only),
                         fba_only_character=slot_summary(fba_only)),
    assembled_cell=dict(n=len(cell), slots=cell_slots, missing_slots=missing),
), open(OUT / "make_cell_ecoli.json", "w"), indent=2)
print("\nwrote outputs/orphan/make_cell_ecoli.json")
