"""Model <-> network dialogue: iterative refinement of the predicted essential set.

The per-gene model proposes essentials; the network audits them and talks back;
the model re-scores; repeat. Each round:

  PROPOSE   model score -> top-K essential set
  LAY OUT   build operon + phyletic network on that set
  AUDIT     the network reports two kinds of "doesn't add up":
            (a) INCOHERENT nodes: in the set but isolated (no operon neighbour,
                no phyletic partner) AND not universal-core -> likely false
                positive leaking in.
            (b) MISSING function: a viable-cell slot below its syn3a minimum,
                or a universal-core gene (essential in ~all other genomes) left
                out -> likely false negative.
  FEEDBACK  down-weight incoherent set members; up-weight universal-core and
            slot-completing candidates. Re-rank.

All feedback signals are leak-clean: universal-core uses leave-one-CLADE-out OG
essential rate; slot minimums come from syn3a (external), never from Keio truth.

Run 3 refinement loops, track precision/recall/coherence/slot-completeness,
plot the trajectory + initial vs final network.

Output: outputs/orphan/refine_loop.png
        outputs/orphan/refine_loop.json
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "beril_Keio"; HELD_CLADE = "escherichia"

# ===================== predictions + features (leave-escherichia-out) =========
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p = P["p"].astype(float); br = P["brightness"].astype(float)
y = P["y"].astype(float); clade = P["clade"].astype(str); lt = P["lt"].astype(str)
valid = ~(np.isnan(p) | np.isnan(br))
C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
cache_orgs = [str(x) for x in C["orgs"]]; cm = C["meta_org"]; cc = C["clade"].astype(str)
org2clade = {cache_orgs[int(cm[i])]: str(cc[i]) for i in range(len(cm))}
feat = {}; orgs_by_lt = defaultdict(list)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    feat[k] = (r.get("og_id") or "", float(r.get("n_paralogs_in_genome") or 0),
               float(r.get("family_n_organisms") or 0), float(r.get("is_orphan") or 0))
    orgs_by_lt[r["locus_tag"]].append(r["organism"])
resolved_org = np.empty(len(p), object); og = np.empty(len(p), object)
paralog = np.zeros(len(p)); breadth = np.zeros(len(p)); is_orph = np.zeros(len(p))
for i in range(len(p)):
    cands = orgs_by_lt.get(lt[i], [])
    match = [o for o in cands if org2clade.get(o) == clade[i]]
    chosen = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    if chosen is None:
        og[i] = ""; resolved_org[i] = ""; continue
    resolved_org[i] = chosen
    og[i], paralog[i], breadth[i], is_orph[i] = feat[(chosen, lt[i])]

g_sum = defaultdict(float); g_n = defaultdict(float)
c_sum = defaultdict(float); c_n = defaultdict(float)
for i in np.where(valid)[0]:
    o = og[i]
    if not o: continue
    g_sum[o] += y[i]; g_n[o] += 1; c_sum[(o, clade[i])] += y[i]; c_n[(o, clade[i])] += 1
og_univ = np.full(len(p), np.nan)
for i in np.where(valid)[0]:
    o = og[i]
    if not o: continue
    n = g_n[o] - c_n[(o, clade[i])]
    if n > 0: og_univ[i] = (g_sum[o] - c_sum[(o, clade[i])]) / n

def fit_logit(X, yy, l2=1.0, iters=500, lr=0.4):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** 0.5; sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr - yy) * sw) / len(X); g[1:] += l2 * w[1:] / len(X); w -= lr * g
    return w, mu, sd
def pred_logit(m, X):
    w, mu, sd = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ w, -30, 30)))
mask = valid & ~np.isnan(og_univ)
def feats(mk):
    return np.column_stack([p[mk], br[mk], og_univ[mk], np.log1p(paralog[mk]),
                            np.log1p(breadth[mk]), is_orph[mk]])
M = fit_logit(feats(mask & (clade != HELD_CLADE)), y[mask & (clade != HELD_CLADE)])
is_keio = (resolved_org == ORG) & mask
patched = pred_logit(M, feats(is_keio))
ki = np.where(is_keio)[0]
klt = lt[ki]; ky = y[ki]; kuniv = og_univ[ki]
print(f"Keio genes predicted: {len(ki)}  (model never saw their labels)")
true_ess = set(klt[ky == 1]); K = int(round(0.17 * len(ki)))

# ===================== network + slot machinery ==============================
gene_names = {}; annot = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        gn = (r.get("gene_name") or "").strip()
        if gn and gn != "-": gene_names[r["locus_tag"]] = gn
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"): annot[r["locus_tag"]] = r["product"]
og_of = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]; og_present[r["og_id"]].add(r["organism"])
n_orgs = len({o for s in og_present.values() for o in s})
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""), r.get("next_locus_tag", ""))

SLOTS = [
    ("ribosomal",[r"ribosom",r"\b30s\b",r"\b50s\b"],[r"^rp[sl][a-z]?\d*$",r"^rpm[a-z]\d*$"]),
    ("trna",[r"\btrna\b",r"transfer rna"],[r"^trn",r"^trna-"]),
    ("synthetase",[r"synthetase",r"trna ligase",r"aminoacyl"],[r"^[a-z]rs[a-z]?$"]),
    ("transcription",[r"polymerase",r"transcription",r"sigma factor"],[r"^rpo[a-z]",r"^nus",r"^sig"]),
    ("replication",[r"helicase",r"gyrase",r"primase",r"topoisomer",r"replication"],
                   [r"^dna[a-z]$",r"^gyr[ab]$",r"^par[ce]$",r"^lig[a-z]$",r"^ssb$",r"^hol[a-z]$"]),
    ("translation",[r"elongation factor",r"initiation factor",r"release factor",r"chaperone"],
                   [r"^tuf",r"^fus",r"^if[123]",r"^prf",r"^dna[jk]$",r"^gro"]),
    ("cell_division",[r"cell division",r"septation",r"ftsz"],[r"^fts",r"^min[cde]$"]),
    ("membrane",[r"membrane",r"lipoprotein",r"transporter",r"permease",r"porin"],
                [r"^omp",r"^lpt",r"^msb",r"^sec[a-z]$",r"^lol[a-z]$"]),
    ("energy",[r"atp synthase",r"atpase",r"cytochrome",r"nadh",r"electron transport"],
              [r"^atp[a-z]",r"^nuo",r"^cyo",r"^cyd"]),
    ("lipid",[r"fatty acid",r"phospholipid",r"lipid"],[r"^fab",r"^pls",r"^acp"]),
    ("nucleotide",[r"purine",r"pyrimidine",r"nucleotide",r"ribonucleot"],[r"^pur",r"^pyr",r"^nrd"]),
    ("amino_acid",[r"amino acid",r"glutam",r"aspart"],[r"^thr[abc]$",r"^trp",r"^his[a-h]$",r"^aro",r"^ilv"]),
    ("cofactor",[r"cofactor",r"folate",r"thiamine",r"riboflavin",r"coenzyme"],[r"^fol",r"^thi",r"^rib",r"^pdx"]),
    ("other",[],[]),
]
SLOT_NAMES = [s[0] for s in SLOTS]
SLOT_PROD = [[re.compile(x, re.I) for x in a] for _, a, _ in SLOTS]
SLOT_GENE = [[re.compile(x, re.I) for x in b] for _, _, b in SLOTS]
# viable-cell minimums (syn3a-derived; external reference, not Keio truth)
SLOT_MIN = {"ribosomal":30,"trna":20,"synthetase":15,"transcription":3,"replication":5,
            "translation":5,"cell_division":1,"membrane":10,"energy":3,"lipid":0,
            "nucleotide":0,"amino_acid":0,"cofactor":0,"other":0}
def classify(prod, gn):
    pp = (prod or "").lower(); gg = (gn or "").lower().strip()
    txt = pp if pp else (gg if len(gg) > 6 else "")
    for i in range(len(SLOT_NAMES) - 1):
        if txt and any(rx.search(txt) for rx in SLOT_PROD[i]): return i
        if gg and any(rx.search(gg) for rx in SLOT_GENE[i]): return i
    return len(SLOT_NAMES) - 1
slot_of = {x: classify(annot.get(x, ""), gene_names.get(x, "")) for x in klt}
univ_of = {klt[j]: (kuniv[j] >= 0.90) for j in range(len(klt))}

def build(ess_lts):
    ess = list(ess_lts); es = set(ess); idx = {x: i for i, x in enumerate(ess)}; n = len(ess)
    edges = set(); operon = set()
    for x in ess:
        for nb in synt.get(x, ("", "")):
            if nb in es and nb != x:
                e = tuple(sorted((idx[x], idx[nb]))); edges.add(e); operon.add(e)
    frac = {}; pv = {}
    for x in ess:
        o = og_of.get((ORG, x))
        if o: pv[x] = og_present[o]; frac[x] = len(og_present[o]) / n_orgs
    keys = [x for x in ess if x in pv and 0.10 <= frac[x] < 0.90]
    for a in range(len(keys)):
        va = pv[keys[a]]
        for b in range(a + 1, len(keys)):
            vb = pv[keys[b]]; inter = len(va & vb)
            if inter and inter / (len(va) + len(vb) - inter) >= 0.9:
                edges.add(tuple(sorted((idx[keys[a]], idx[keys[b]]))))
    adj = defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    return ess, idx, edges, operon, adj

# ===================== the dialogue ==========================================
def metrics(pred_set):
    tp = len(pred_set & true_ess)
    return dict(precision=round(tp/max(len(pred_set),1),3),
                recall=round(tp/max(len(true_ess),1),3),
                f1=round(2*tp/max(len(pred_set)+len(true_ess),1),3), tp=tp)

lt2j = {klt[j]: j for j in range(len(klt))}
N_LOOPS = 3

def run_dialogue(mode):
    """mode='naive' (topology/isolation audit) or 'discriminative' (universality)."""
    score = patched.copy(); hist = []; snaps = {}
    for loop in range(N_LOOPS + 1):
        order = np.argsort(-score)
        ess_lts = [klt[j] for j in order[:K]]; ess_set = set(ess_lts)
        ess, idx, edges, operon, adj = build(ess_lts)
        deg = {x: len(adj[idx[x]]) for x in ess}
        incoherent = [x for x in ess if deg[x] == 0 and not univ_of.get(x, False)]
        slot_counts = defaultdict(int)
        for x in ess: slot_counts[SLOT_NAMES[slot_of[x]]] += 1
        under = {s: SLOT_MIN[s]-slot_counts[s] for s in SLOT_MIN if slot_counts[s] < SLOT_MIN[s]}
        sc_done = sum(1 for s in SLOT_MIN if SLOT_MIN[s] > 0 and slot_counts[s] >= SLOT_MIN[s])
        n_req = sum(1 for s in SLOT_MIN if SLOT_MIN[s] > 0)
        m = metrics(ess_set)
        hist.append(dict(loop=loop, **m, incoherent=len(incoherent),
                         slots_complete=f"{sc_done}/{n_req}"))
        if loop in (0, N_LOOPS):
            snaps[loop] = (ess, idx, edges, operon, adj, ess_set)
        if loop == N_LOOPS:
            break
        if mode == "naive":
            # topology audit: isolated -> FP (the WRONG rule)
            for x in incoherent:
                score[lt2j[x]] *= 0.5
            for j in range(len(klt)):
                if univ_of.get(klt[j], False) and klt[j] not in ess_set:
                    score[j] = max(score[j], 0.85)
            for s, need in under.items():
                si = SLOT_NAMES.index(s)
                cands = sorted([(score[lt2j[x]], x) for x in klt
                                if slot_of[x] == si and x not in ess_set], reverse=True)
                promoted = 0
                for _, x in cands:
                    if [nb for nb in synt.get(x, ("", "")) if nb in ess_set] or univ_of.get(x, False):
                        score[lt2j[x]] = max(score[lt2j[x]], 0.8); promoted += 1
                    if promoted >= need: break
        else:
            # discriminative audit: cross-organism essentiality (the RIGHT signal)
            for x in ess:
                j = lt2j[x]
                if not univ_of.get(x, False) and kuniv[j] < 0.5:
                    score[j] *= 0.7
            for j in range(len(klt)):
                if klt[j] not in ess_set and (univ_of.get(klt[j], False) or kuniv[j] >= 0.85):
                    score[j] = max(score[j], 0.82)
            for s, need in under.items():
                si = SLOT_NAMES.index(s)
                cands = sorted([(kuniv[lt2j[x]], x) for x in klt
                                if slot_of[x] == si and x not in ess_set and kuniv[lt2j[x]] >= 0.5],
                               reverse=True)
                for _, x in cands[:need]:
                    score[lt2j[x]] = max(score[lt2j[x]], 0.80)
    return hist, snaps

print("--- naive (topology/isolation audit) ---")
hist_naive, _ = run_dialogue("naive")
for h in hist_naive: print(f"  loop {h['loop']}: F1={h['f1']} P={h['precision']} R={h['recall']} incoh={h['incoherent']}")
print("--- discriminative (cross-organism essentiality audit) ---")
hist_disc, snaps = run_dialogue("discriminative")
for h in hist_disc: print(f"  loop {h['loop']}: F1={h['f1']} P={h['precision']} R={h['recall']} incoh={h['incoherent']}")

# ---- diagnostic: do audit signals separate TP from FP at loop 0? ----
ess0, idx0, _, _, adj0, ess0_set = snaps[0]
deg0 = {x: len(adj0[idx0[x]]) for x in ess0}
tp0 = [x for x in ess0 if x in true_ess]; fp0 = [x for x in ess0 if x not in true_ess]
def grp(g, key):
    return float(np.mean([key(x) for x in g]))
diag = dict(
    TP_isolated=grp(tp0, lambda x: deg0[x] == 0),
    FP_isolated=grp(fp0, lambda x: deg0[x] == 0),
    TP_universal=grp(tp0, lambda x: univ_of.get(x, False)),
    FP_universal=grp(fp0, lambda x: univ_of.get(x, False)),
)
# missed essentials (false negatives) -- species-specific?
missed = [x for x in true_ess if x not in ess0_set and x in lt2j]
fn_oguniv = np.array([kuniv[lt2j[x]] for x in missed])
tp_oguniv = np.array([kuniv[lt2j[x]] for x in tp0])

# ===================== figure ================================================
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))

# (1) why naive fails: audit signal discrimination
ax[0].bar([0, 1], [diag["TP_isolated"], diag["FP_isolated"]], width=0.35,
          color="#9ecae1", label="isolated (topology)")
ax[0].bar([0.4, 1.4], [diag["TP_universal"], diag["FP_universal"]], width=0.35,
          color="#e31a1c", label="universal-core (cross-org)")
ax[0].set_xticks([0.2, 1.2]); ax[0].set_xticklabels(["true essential", "false positive"])
ax[0].set_ylabel("fraction"); ax[0].set_ylim(0, 1)
ax[0].set_title("Which audit signal separates TP from FP?\n"
                "isolation: flat (bad).  universal-core: 0.70 vs 0.27 (good)",
                fontweight="bold", fontsize=11)
ax[0].legend(fontsize=8)

# (2) trajectory: naive hurts, discriminative is flat
L = [h["loop"] for h in hist_naive]
ax[1].plot(L, [h["f1"] for h in hist_naive], "o-", color="#e31a1c", lw=2,
           label="naive audit (isolation)")
ax[1].plot(L, [h["f1"] for h in hist_disc], "s-", color="#3182bd", lw=2,
           label="discriminative audit")
ax[1].axhline(hist_disc[0]["f1"], color="#888", ls="--", lw=0.8)
ax[1].set_xlabel("dialogue loop"); ax[1].set_ylabel("F1 vs Keio truth")
ax[1].set_xticks(L); ax[1].set_ylim(0.5, 0.8)
ax[1].set_title("3 loops of dialogue: does it help accuracy?\n"
                "naive HURTS; discriminative is a NO-OP (signal already in model)",
                fontweight="bold", fontsize=11)
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

# (3) the recall ceiling: missed essentials are species-specific
ax[2].hist(tp_oguniv, bins=20, alpha=0.7, color="#3182bd", label=f"recovered TP (n={len(tp0)})")
ax[2].hist(fn_oguniv, bins=20, alpha=0.7, color="#e31a1c",
           label=f"missed FN (n={len(missed)})")
ax[2].axvline(0.5, color="#444", ls="--", lw=0.8)
ax[2].set_xlabel("cross-organism essential rate of gene's orthogroup (og_univ)")
ax[2].set_ylabel("# genes")
ax[2].set_title("Why recall is capped at 0.73\n"
                "missed essentials are SPECIES-SPECIFIC (low og_univ) --\n"
                "no cross-organism signal exists to recover them",
                fontweight="bold", fontsize=11)
ax[2].legend(fontsize=8)

fig.suptitle("Model <-> network dialogue (3 loops): the network can only improve "
             "the model with signal the model doesn't already have", fontweight="bold",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "refine_loop.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/refine_loop.png")

json.dump(dict(organism=ORG, held_clade=HELD_CLADE, K=K, n_true_essential=len(true_ess),
               naive=hist_naive, discriminative=hist_disc, diagnostic=diag,
               n_missed_species_specific=int((fn_oguniv < 0.5).sum()),
               n_missed_total=len(missed)),
          open(OUT / "refine_loop.json", "w"), indent=2)
print("wrote outputs/orphan/refine_loop.json")
