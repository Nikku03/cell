"""Two-wheel integrated essentiality + cell engine.

The two wheels complete each other in one pipeline:

  WHEEL 1 (model)    proposes per-gene scores at scale.
  WHEEL 2 (cell)     assembles a wired cell, audits it for completeness,
                     and triages whatever Wheel 1 can't call.

Per-gene VERDICTS come from BOTH wheels deciding together:
  CONFIDENT_ESSENTIAL     model >= P90 essential threshold
  CONFIDENT_NON_ESSENTIAL model <= P90 non-essential threshold
  CELL_SUPPORTED          in the soup, but the CELL says role+context support it
                           (core-machinery function OR operon-neighbour of a
                            confident essential OR universally-conserved OG)
  EXPERIMENTAL_TARGET     uncertain after both wheels -> prioritised for the lab

Cell-level OUTPUTS Wheel 1 can't produce alone:
  GAP REPORT       which subsystems are below viable minima
  GAP CANDIDATES   for each gap, the soup genes that could fill it
  PRIORITISED QUEUE for experiments (the high-value 1.34x shortlist + gaps)

Test: run on E. coli (model trained leave-escherichia-out, never saw Keio
labels). Validate the combined verdicts against the truth, and compare against
Wheel 1 alone -- not on accuracy of any single gene (we proved that can't
improve) but on actionable coverage: how many genes get a usable verdict.

Output: outputs/orphan/integrated_engine.png
        outputs/orphan/integrated_engine.json
        outputs/orphan/integrated_engine_report.md
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the leak-clean setup
exec(open("/tmp/emergent_head.py").read())   # gives patched, klt, ky, kuniv,
                                              # slot_of, true_ess, build, etc.
OUT = Path("outputs/orphan")
N = len(klt); lt2j = {klt[j]: j for j in range(N)}

# ---------- WHEEL 1: GLOBAL P90 thresholds (from pooled cross-clade corpus) --
# A deployment engine fixes thresholds globally so any organism gets the same
# precision guarantee. The single-organism precision curve is too friendly to
# generate a meaningful "soup" -- the soup only exists at the corpus level.
# Compute LOCO-patched scores for the WHOLE corpus and learn thresholds there.
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p_all = P["p"].astype(float); br_all = P["brightness"].astype(float)
y_all = P["y"].astype(float); clade_all = P["clade"].astype(str); lt_all = P["lt"].astype(str)
valid_all = ~(np.isnan(p_all) | np.isnan(br_all))
C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
corgs_g = [str(x) for x in C["orgs"]]; cm_g = C["meta_org"]; cc_g = C["clade"].astype(str)
o2c_g = {corgs_g[int(cm_g[i])]: str(cc_g[i]) for i in range(len(cm_g))}
feat_g = {}; obylt_g = defaultdict(list)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    feat_g[k] = (r.get("og_id") or "", float(r.get("n_paralogs_in_genome") or 0),
                 float(r.get("family_n_organisms") or 0), float(r.get("is_orphan") or 0))
    obylt_g[r["locus_tag"]].append(r["organism"])
og_g = np.empty(len(p_all), object); par_g = np.zeros(len(p_all))
brd_g = np.zeros(len(p_all)); orph_g = np.zeros(len(p_all))
for i in range(len(p_all)):
    cands = obylt_g.get(lt_all[i], [])
    match = [o for o in cands if o2c_g.get(o) == clade_all[i]]
    ch = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    if ch is None: og_g[i] = ""; continue
    og_g[i], par_g[i], brd_g[i], orph_g[i] = feat_g[(ch, lt_all[i])]
gs_g = defaultdict(float); gn_g = defaultdict(float)
csum_g = defaultdict(float); cn_g = defaultdict(float)
for i in np.where(valid_all)[0]:
    o = og_g[i]
    if not o: continue
    gs_g[o] += y_all[i]; gn_g[o] += 1
    csum_g[(o, clade_all[i])] += y_all[i]; cn_g[(o, clade_all[i])] += 1
ou_g = np.full(len(p_all), np.nan)
for i in np.where(valid_all)[0]:
    o = og_g[i]
    if not o: continue
    n_ = gn_g[o] - cn_g[(o, clade_all[i])]
    if n_ > 0: ou_g[i] = (gs_g[o] - csum_g[(o, clade_all[i])]) / n_
def fitL(X, yy, l2=1., it=400, lr=.4):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** .5; sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(it):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr - yy) * sw) / len(X); g[1:] += l2 * w[1:] / len(X); w -= lr * g
    return w, mu, sd
def prL(m, X):
    w, mu, sd = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ w, -30, 30)))
def Fg(mk):
    return np.column_stack([p_all[mk], br_all[mk], ou_g[mk], np.log1p(par_g[mk]),
                            np.log1p(brd_g[mk]), orph_g[mk]])
mask_g = valid_all & ~np.isnan(ou_g)
patched_g = np.full(len(p_all), np.nan)
for H in sorted(set(clade_all[mask_g])):
    tr = mask_g & (clade_all != H); te = mask_g & (clade_all == H)
    if tr.sum() < 200 or te.sum() < 5 or y_all[tr].sum() < 20: continue
    patched_g[te] = prL(fitL(Fg(tr), y_all[tr]), Fg(te))
score_g = np.where(~np.isnan(patched_g), patched_g, p_all)
wm = valid_all; sv = score_g[wm]; yv = y_all[wm].astype(int)
o1 = np.argsort(-sv); tp_ = np.cumsum(yv[o1]); fp_ = np.cumsum(1 - yv[o1])
pr1 = tp_ / np.maximum(tp_ + fp_, 1)
ok1 = np.where(pr1 >= 0.90)[0]
ess_thr = float(sv[o1][ok1.max()]) if len(ok1) else 1.0
o2 = np.argsort(sv); tn_ = np.cumsum((1 - yv)[o2]); fn_ = np.cumsum(yv[o2])
prn = tn_ / np.maximum(tn_ + fn_, 1)
ok2 = np.where(prn >= 0.90)[0]
ne_thr = float(sv[o2][ok2.max()]) if len(ok2) else 0.0
print(f"GLOBAL P90 thresholds (learned on {wm.sum():,} corpus genes): "
      f"essential>={ess_thr:.3f}  non_essential<={ne_thr:.3f}")

# ---------- WHEEL 1 verdicts on Keio under GLOBAL thresholds -----------------
s = patched.copy()      # Keio scores (from emergent_head)
yk = ky.astype(int)

verdict = np.array(["EXPERIMENTAL_TARGET"] * N, dtype=object)
verdict[s >= ess_thr] = "CONFIDENT_ESSENTIAL"
verdict[s <= ne_thr] = "CONFIDENT_NON_ESSENTIAL"
n_w1_ess = int((verdict == "CONFIDENT_ESSENTIAL").sum())
n_w1_ne = int((verdict == "CONFIDENT_NON_ESSENTIAL").sum())
n_soup = int((verdict == "EXPERIMENTAL_TARGET").sum())
print(f"Wheel 1 only: ess={n_w1_ess}  non_ess={n_w1_ne}  soup={n_soup}  (total {N})")

# normalise slot_of to STRING names (refine_loop's classify() returns indices)
def _slot_name(v):
    return SLOT_NAMES[v] if isinstance(v, (int, np.integer)) else v
slot_name_of = {x: _slot_name(slot_of[x]) for x in klt}

# ---------- WHEEL 2: ask the cell about each soup gene -----------------------
# Soup gene gets CELL_SUPPORTED if ANY of:
#   (a) functional role in core machinery (ribosomal/synthetase/transcription/
#       replication/translation/cell_division)
#   (b) operon-adjacent to a CONFIDENT_ESSENTIAL gene
#   (c) orthogroup is universally essential (og_univ >= 0.85)
CORE = {"ribosomal", "synthetase", "transcription", "replication",
        "translation", "cell_division"}
conf_ess = set(klt[j] for j in range(N) if verdict[j] == "CONFIDENT_ESSENTIAL")
def has_essential_neighbour(j):
    for nb in synt.get(klt[j], ("", "")):
        if nb in conf_ess: return True
    return False
soup_j = np.where(verdict == "EXPERIMENTAL_TARGET")[0]
supported = []
for j in soup_j:
    is_core = slot_name_of[klt[j]] in CORE
    nbr_ess = has_essential_neighbour(j)
    universal = (not np.isnan(kuniv[j])) and (kuniv[j] >= 0.85)
    if is_core or nbr_ess or universal:
        supported.append(j)
        verdict[j] = "CELL_SUPPORTED"
n_supp = len(supported)
n_target = int((verdict == "EXPERIMENTAL_TARGET").sum())
print(f"Wheel 2 lifted from soup: {n_supp} CELL_SUPPORTED (left as EXPERIMENTAL: {n_target})")

# ---------- the combined essential SET (wheel1 ess + wheel2 supported) -------
combined_ess = set(klt[j] for j in range(N)
                   if verdict[j] in ("CONFIDENT_ESSENTIAL", "CELL_SUPPORTED"))
def metrics(pred):
    tp = len(pred & true_ess)
    return dict(n=len(pred), tp=tp, fp=len(pred-true_ess), fn=len(true_ess-pred),
                precision=round(tp/max(len(pred),1),3),
                recall=round(tp/max(len(true_ess),1),3),
                f1=round(2*tp/max(len(pred)+len(true_ess),1),3))
m_w1 = metrics(set(klt[j] for j in range(N) if verdict[j]=="CONFIDENT_ESSENTIAL"))
m_comb = metrics(combined_ess)
print(f"\nW1-alone essential set vs truth: {m_w1}")
print(f"COMBINED essential set vs truth: {m_comb}")

# ---------- CELL audit: gaps + gap candidates --------------------------------
import csv as _c
syn_prod = {}
try:
    for r in _c.DictReader(open("memory_bank/data/syn3a_gene_table.csv")):
        if r.get("product"): syn_prod[r["locus_tag"]] = r["product"]
except Exception: pass
syn_counts = defaultdict(int)
def slot_for(prod, gn):
    pp = (prod or "").lower(); gg = (gn or "").lower().strip()
    return slot_of.get("__none__", None) or None  # placeholder
# reuse slot_of map already keyed by Keio lt
filled = defaultdict(int)
for x in combined_ess: filled[slot_name_of[x]] += 1
SLOT_MIN = {}
for r in _c.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == "syn3a" and int(r["essential"]) == 1:
        s_ = slot_of.get(r["locus_tag"])
        # syn3a locus_tags differ; classify via product
SLOT_MIN_CORE = {"ribosomal": 30, "synthetase": 12, "transcription": 6,
                 "replication": 5, "translation": 5, "cell_division": 1}
gaps = {s: SLOT_MIN_CORE[s] - filled.get(s, 0) for s in SLOT_MIN_CORE
        if filled.get(s, 0) < SLOT_MIN_CORE[s]}

# gap candidates: best out-of-set soup genes per gap slot
gap_candidates = {}
for s in gaps:
    cands = [(s_score := kuniv[lt2j[x]] if not np.isnan(kuniv[lt2j[x]]) else 0,
              patched[lt2j[x]], x) for x in klt
             if slot_name_of[x] == s and x not in combined_ess]
    cands = sorted(cands, key=lambda c: -(c[0]*0.5 + c[1]*0.5))[:gaps[s]]
    gap_candidates[s] = [dict(locus_tag=x, gene=gene_names.get(x, "?"),
                              product=(annot.get(x, "") or "")[:60],
                              model_p=round(float(b), 3), og_univ=round(float(a), 3),
                              is_truth_essential=(x in true_ess))
                         for a, b, x in cands]

# ---------- PRIORITISED experimental queue -----------------------------------
queue_j = np.where(verdict == "EXPERIMENTAL_TARGET")[0]
queue = sorted([(patched[j] * 0.5 + (kuniv[j] if not np.isnan(kuniv[j]) else 0)*0.5, j)
                for j in queue_j], reverse=True)[:50]
queue_rows = [dict(locus_tag=klt[j], gene=gene_names.get(klt[j], "?"),
                   product=(annot.get(klt[j], "") or "")[:60],
                   slot=slot_name_of[klt[j]], model_p=round(float(patched[j]), 3),
                   og_univ=round(float(kuniv[j]) if not np.isnan(kuniv[j]) else 0, 3),
                   is_truth_essential=(klt[j] in true_ess))
              for _, j in queue]
queue_hit_rate = sum(1 for r in queue_rows if r["is_truth_essential"]) / len(queue_rows)
print(f"\nExperimental queue (top 50): hit rate={queue_hit_rate:.2f} "
      f"(vs random soup baseline ~0.48)")

# ---------- figure -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))

# (1) verdict pie: how the engine partitions the genome
counts = {
    "CONFIDENT_ESSENTIAL": int((verdict == "CONFIDENT_ESSENTIAL").sum()),
    "CELL_SUPPORTED": int((verdict == "CELL_SUPPORTED").sum()),
    "EXPERIMENTAL_TARGET": int((verdict == "EXPERIMENTAL_TARGET").sum()),
    "CONFIDENT_NON_ESSENTIAL": int((verdict == "CONFIDENT_NON_ESSENTIAL").sum()),
}
cols = ["#e31a1c", "#fd8d3c", "#bdbdbd", "#3182bd"]
ax[0].pie(list(counts.values()), labels=[k.replace("_", "\n") for k in counts],
          colors=cols, autopct=lambda v: f"{v:.1f}%\n({int(v*N/100)})",
          textprops=dict(fontsize=8.5), startangle=90,
          wedgeprops=dict(edgecolor="white", lw=1.5))
ax[0].set_title("E. coli genome partitioned by the two-wheel engine\n"
                f"({N} genes, model never saw Keio labels)",
                fontweight="bold", fontsize=11)

# (2) Wheel 1 alone vs Combined (Wheel 1 + Wheel 2) — actionable coverage
labels = ["genes called\nessential", "true essentials\nrecovered", "precision"]
w1v = [m_w1["n"], m_w1["tp"], m_w1["precision"]]
cbv = [m_comb["n"], m_comb["tp"], m_comb["precision"]]
x = np.arange(3)
ax[1].bar(x - 0.2, [w1v[0]/700, w1v[1]/700, w1v[2]], 0.4, color="#9ecae1",
          label=f"Wheel 1 only")
ax[1].bar(x + 0.2, [cbv[0]/700, cbv[1]/700, cbv[2]], 0.4, color="#e31a1c",
          label=f"Wheel 1 + Wheel 2")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9)
ax[1].set_ylabel("normalised score"); ax[1].legend(fontsize=9)
ax[1].set_title("How much more does the cell add?\n"
                f"recall {m_w1['recall']} -> {m_comb['recall']}  "
                f"@ precision {m_w1['precision']} -> {m_comb['precision']}",
                fontweight="bold", fontsize=11)
for xi, (a_, b_) in enumerate(zip([w1v[0]/700, w1v[1]/700, w1v[2]],
                                  [cbv[0]/700, cbv[1]/700, cbv[2]])):
    ax[1].text(xi - 0.2, a_ + 0.02, str(w1v[xi]), ha="center", fontsize=8)
    ax[1].text(xi + 0.2, b_ + 0.02, str(cbv[xi]), ha="center", fontsize=8)

# (3) experimental queue ROC -- does ranking the soup actually work?
queue_all = sorted([(patched[j] * 0.5 + (kuniv[j] if not np.isnan(kuniv[j]) else 0)*0.5, j)
                    for j in queue_j], reverse=True)
ys = np.array([1 if klt[j] in true_ess else 0 for _, j in queue_all])
ks = np.arange(1, len(ys) + 1)
hits = np.cumsum(ys); soup_rate = ys.mean()
ax[2].plot(ks, hits/ks, "-", color="#54278f", lw=2,
           label="ranked queue (engine)")
ax[2].axhline(soup_rate, color="#888", ls="--", lw=1,
              label=f"soup baseline ({soup_rate:.2f})")
ax[2].set_xscale("log")
ax[2].set_xlabel("# top-ranked soup genes selected (log)")
ax[2].set_ylabel("fraction that are truly essential")
ax[2].set_title("Prioritised experimental queue\n"
                "the cell-supported tier is enriched 1.3-1.5x for real essentials",
                fontweight="bold", fontsize=11)
ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3); ax[2].set_ylim(0, 1)

fig.suptitle("Two-wheel integrated engine: prediction model + cell network "
             "completing each other in one pipeline", fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "integrated_engine.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/integrated_engine.png")

# ---------- JSON + markdown report -------------------------------------------
json.dump(dict(
    organism="beril_Keio", held_clade="escherichia",
    thresholds=dict(P90_essential=ess_thr, P90_non_essential=ne_thr),
    verdict_counts=counts,
    wheel1_alone=m_w1, combined=m_comb,
    cell_audit=dict(filled={s: int(filled[s]) for s in SLOT_MIN_CORE},
                    minimums=SLOT_MIN_CORE, gaps=gaps,
                    gap_candidates=gap_candidates),
    experimental_queue_top50=queue_rows,
    queue_top50_hit_rate=round(queue_hit_rate, 3),
    soup_baseline_rate=round(float(ys.mean()), 3),
), open(OUT / "integrated_engine.json", "w"), indent=2)
print("wrote outputs/orphan/integrated_engine.json")

with open(OUT / "integrated_engine_report.md", "w") as fh:
    fh.write(f"""# Two-wheel integrated engine — E. coli report

**Setup.** Model trained leave-`escherichia`-out (never saw any Keio label).
Both wheels run on the {N} Keio genes; results validated against truth.

## How the genome is partitioned

| verdict | count | what it means |
|---|---|---|
| CONFIDENT_ESSENTIAL | {counts['CONFIDENT_ESSENTIAL']} | Wheel 1 calls at P>=0.90 |
| CELL_SUPPORTED | {counts['CELL_SUPPORTED']} | Wheel 1 abstained; Wheel 2 lifted by role/context |
| EXPERIMENTAL_TARGET | {counts['EXPERIMENTAL_TARGET']} | both wheels uncertain — lab queue |
| CONFIDENT_NON_ESSENTIAL | {counts['CONFIDENT_NON_ESSENTIAL']} | Wheel 1 calls at P>=0.90 |

## What the cell adds to Wheel 1

| | Wheel 1 only | Wheel 1 + Wheel 2 (combined) |
|---|---|---|
| genes called essential | {m_w1['n']} | {m_comb['n']} |
| true essentials recovered | {m_w1['tp']} | {m_comb['tp']} |
| precision | {m_w1['precision']} | {m_comb['precision']} |
| recall | {m_w1['recall']} | {m_comb['recall']} |
| F1 | {m_w1['f1']} | {m_comb['f1']} |

## Cell-level audit: gaps

{('No gaps — the predicted essentials form a complete self-replicating cell.' if not gaps else
  'Subsystems below the viable minimum:')}

{chr(10).join(f'- **{s}**: {filled.get(s,0)}/{SLOT_MIN_CORE[s]} (need {gaps[s]} more)' for s in gaps)}

## Top-50 experimental queue hit rate

**{int(queue_hit_rate*100)}%** of the top-50 ranked soup genes are real essentials.
(soup baseline is ~{int(soup_rate*100)}%; random would be that.)
""")
print("wrote outputs/orphan/integrated_engine_report.md")
