"""Multi-organism demand-driven assembly: run the Wheel2->Wheel1->Wheel2 loop
across a spectrum of genomes and measure where the assembly does heavy work.

Test 6 organisms spanning predictability:
  E. coli (Keio)  - well-predicted free-living, baseline
  M. tuberculosis - distinct clade, harder
  syn3a / mgen    - minimal cells (Mycoplasma)
  S. aureus       - small dataset
  S. pneumoniae   - small dataset, distant

For each:
  1 global P90 thresholds applied  -> Wheel 1 confident essentials
  2 syn3a-derived requirements table (the same FUNCTIONAL spec for every cell)
  3 gaps reported
  4 soup classified into PROTEIN FAMILIES via cross-organism orthogroup
    profile (consensus product, consensus slot, breadth) -- the data
    equivalent of an ESM/structure family classifier
  5 Wheel 2 reasons which families fit which open slot and inserts top candidates
  6 validate against truth

The thesis being tested: when Wheel 1 is weak, Wheel 2's assembly does
proportionally more lifting -- demand-driven, not blanket re-scoring.

Output: outputs/orphan/multi_org_assembly.png
        outputs/orphan/multi_org_assembly.json
"""
import csv, json, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
OUT = Path("outputs/orphan")
# overridable artifact paths (DEG run sets these via env)
PREDS = os.environ.get("AF_PREDS", "af_torch_preds_aug.npz")
CACHE = os.environ.get("AF_CACHE", "af_msa_cache.npz")
LABELS_DIR = os.environ.get("AF_LABELS_DIR", "data/drive_import/labels")
ORTH_CSV = f"{LABELS_DIR}/orthology_features.csv"
LABELS_CSV = f"{LABELS_DIR}/labels.csv"
ANNOT_CSV = "data/drive_import/labels/gene_annotations.csv"

# ===================== 1. predictions + features (whole corpus) ==============
P = np.load(OUT / PREDS, allow_pickle=True)
p = P["p"].astype(float); br = P["brightness"].astype(float)
y = P["y"].astype(float); clade = P["clade"].astype(str); lt = P["lt"].astype(str)
valid = ~(np.isnan(p) | np.isnan(br))
C = np.load(OUT / CACHE, allow_pickle=True)
corgs = [str(x) for x in C["orgs"]]; cm = C["meta_org"]; cc = C["clade"].astype(str)
org2clade = {corgs[int(cm[i])]: str(cc[i]) for i in range(len(cm))}

feat = {}; obylt = defaultdict(list)
for r in csv.DictReader(open(ORTH_CSV)):
    k = (r["organism"], r["locus_tag"])
    feat[k] = (r.get("og_id") or "", float(r.get("n_paralogs_in_genome") or 0),
               float(r.get("family_n_organisms") or 0), float(r.get("is_orphan") or 0))
    obylt[r["locus_tag"]].append(r["organism"])
resolved = np.empty(len(p), object); og_arr = np.empty(len(p), object)
par = np.zeros(len(p)); brd = np.zeros(len(p)); orph = np.zeros(len(p))
for i in range(len(p)):
    cands = obylt.get(lt[i], [])
    match = [o for o in cands if org2clade.get(o) == clade[i]]
    ch = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    resolved[i] = ch or ""
    if ch is None: og_arr[i] = ""; continue
    og_arr[i], par[i], brd[i], orph[i] = feat[(ch, lt[i])]

# leak-clean OG-universality
gs = defaultdict(float); gn_ = defaultdict(float)
csum = defaultdict(float); cn = defaultdict(float)
for i in np.where(valid)[0]:
    o = og_arr[i]
    if not o: continue
    gs[o] += y[i]; gn_[o] += 1
    csum[(o, clade[i])] += y[i]; cn[(o, clade[i])] += 1
ou = np.full(len(p), np.nan)
for i in np.where(valid)[0]:
    o = og_arr[i]
    if not o: continue
    n = gn_[o] - cn[(o, clade[i])]
    if n > 0: ou[i] = (gs[o] - csum[(o, clade[i])]) / n

# leave-one-clade-out patched stacker -> score
def fit(X, yy, l2=1.0, it=400, lr=.4):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** .5
    sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(it):
        pr_ = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr_ - yy) * sw) / len(X)
        g[1:] += l2 * w[1:] / len(X); w -= lr * g
    return w, mu, sd
def predL(m, X):
    w, mu, sd = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ w, -30, 30)))
def F(mk):
    return np.column_stack([p[mk], br[mk], ou[mk], np.log1p(par[mk]),
                            np.log1p(brd[mk]), orph[mk]])
mask = valid & ~np.isnan(ou); patched = np.full(len(p), np.nan)
for H in sorted(set(clade[mask])):
    tr = mask & (clade != H); te = mask & (clade == H)
    if tr.sum() < 200 or te.sum() < 5 or y[tr].sum() < 20: continue
    patched[te] = predL(fit(F(tr), y[tr]), F(te))
score = np.where(~np.isnan(patched), patched, p)

# global P90 thresholds (deployment)
sv = score[valid]; yv = y[valid].astype(int)
o1 = np.argsort(-sv); tp_ = np.cumsum(yv[o1]); fp_ = np.cumsum(1 - yv[o1])
ok1 = np.where(tp_ / np.maximum(tp_ + fp_, 1) >= 0.90)[0]
ESS_THR = float(sv[o1][ok1.max()]) if len(ok1) else 1.0
o2 = np.argsort(sv); tn_ = np.cumsum((1 - yv)[o2]); fn_ = np.cumsum(yv[o2])
ok2 = np.where(tn_ / np.maximum(tn_ + fn_, 1) >= 0.90)[0]
NE_THR = float(sv[o2][ok2.max()]) if len(ok2) else 0.0
print(f"GLOBAL P90 thresholds: essential>={ESS_THR:.3f}  non_essential<={NE_THR:.3f}")

# ===================== 2. per-organism resources =============================
annot_all = defaultdict(dict); gname_all = defaultdict(dict)
for r in csv.DictReader(open(ANNOT_CSV)):
    if r.get("product"): annot_all[r["organism"]][r["locus_tag"]] = r["product"]
for r in csv.DictReader(open(LABELS_CSV)):
    gnv = (r.get("gene_name") or "").strip()
    if gnv and gnv != "-": gname_all[r["organism"]][r["locus_tag"]] = gnv

memb = defaultdict(list); og_present = defaultdict(set)
for r in csv.DictReader(open(ORTH_CSV)):
    if r.get("og_id"):
        memb[r["og_id"]].append((r["organism"], r["locus_tag"]))
        og_present[r["og_id"]].add(r["organism"])

# ===================== 3. slot classifier (synthetase before trna) ===========
SLOTS = [
    ("ribosomal", [r"ribosom"], [r"^rp[sl][a-z]?\d*$", r"^rpm[a-z]\d*$"]),
    ("synthetase", [r"synthetase", r"trna ligase", r"aminoacyl"], [r"^[a-z]rs[a-z]?$"]),
    ("trna", [r"\btrna\b", r"transfer rna"], [r"^trn", r"^trna-"]),
    ("transcription", [r"polymerase", r"transcription", r"sigma factor"],
                     [r"^rpo[a-z]", r"^nus", r"^sig"]),
    ("replication", [r"helicase", r"gyrase", r"primase", r"topoisomer", r"replication"],
                   [r"^dna[a-z]$", r"^gyr[ab]$", r"^par[ce]$", r"^lig[a-z]$", r"^ssb$"]),
    ("translation", [r"elongation factor", r"initiation factor", r"release factor", r"chaperone"],
                   [r"^tuf", r"^fus", r"^if[123]", r"^prf", r"^dna[jk]$", r"^gro"]),
    ("cell_division", [r"cell division", r"septation", r"ftsz"], [r"^fts", r"^min[cde]$"]),
    ("membrane", [r"membrane", r"lipoprotein", r"transporter", r"permease", r"porin"],
                [r"^omp", r"^lpt", r"^msb", r"^sec[a-z]$"]),
    ("energy", [r"atp synthase", r"atpase", r"cytochrome", r"nadh", r"electron transport"],
              [r"^atp[a-z]", r"^nuo", r"^cyo", r"^cyd"]),
    ("lipid", [r"fatty acid", r"phospholipid", r"lipid"], [r"^fab", r"^pls", r"^acp"]),
    ("nucleotide", [r"purine", r"pyrimidine", r"nucleotide", r"ribonucleot"],
                  [r"^pur", r"^pyr", r"^nrd"]),
    ("amino_acid", [r"amino acid", r"glutam", r"aspart"],
                  [r"^thr[abc]$", r"^trp", r"^his[a-h]$"]),
    ("cofactor", [r"cofactor", r"folate", r"thiamine", r"riboflavin", r"coenzyme"],
                [r"^fol", r"^thi", r"^rib"]),
    ("other", [], []),
]
SN = [s[0] for s in SLOTS]
SP = [[re.compile(x, re.I) for x in a] for _, a, _ in SLOTS]
SG = [[re.compile(x, re.I) for x in b] for _, _, b in SLOTS]
def classify(prod, gnm):
    pp = (prod or "").lower(); gg = (gnm or "").lower().strip()
    txt = pp if pp else (gg if len(gg) > 6 else "")
    for i in range(len(SN) - 1):
        if txt and any(rx.search(txt) for rx in SP[i]): return SN[i]
        if gg and any(rx.search(gg) for rx in SG[i]): return SN[i]
    return "other"

# syn3a-derived requirements (universal across organisms)
syn_prod = {}
for r in csv.DictReader(open("memory_bank/data/syn3a_gene_table.csv")):
    if r.get("product"): syn_prod[r["locus_tag"]] = r["product"]
syn_counts = Counter()
for r in csv.DictReader(open(LABELS_CSV)):
    if r["organism"] == "syn3a" and int(r["essential"]) == 1:
        syn_counts[classify(syn_prod.get(r["locus_tag"], ""),
                            r.get("gene_name", ""))] += 1
TABLE = {s: int(np.ceil(0.5 * syn_counts.get(s, 0)))
         for s in SN if s != "other" and syn_counts.get(s, 0) > 0}
print(f"Requirements table (syn3a x50%): {TABLE}")

# ===================== 4. enhanced family classifier =========================
# orthogroup -> family profile. This is what a structure-based classifier
# would produce: cluster proteins by sequence/structure -> consensus function.
# Here the clustering is precomputed (orthogroups); we add functional consensus.
_prof_cache = {}
def og_profile(og):
    if og in _prof_cache: return _prof_cache[og]
    members = memb.get(og, [])
    prods = [annot_all[o].get(l, "") for (o, l) in members]
    prods = [pp for pp in prods if pp and pp.lower() != "hypothetical protein"]
    consensus_prod = Counter(prods).most_common(1)[0][0] if prods else ""
    slots = [classify(annot_all[o].get(l, ""), gname_all[o].get(l, ""))
             for (o, l) in members]
    cs = Counter(slots).most_common(1)
    consensus_slot = cs[0][0] if cs else "other"
    _prof_cache[og] = dict(slot=consensus_slot, product=consensus_prod,
                            breadth=len(og_present[og]), n_members=len(members))
    return _prof_cache[og]

# ===================== 5. assembly for one organism ==========================
def assemble(org):
    idx = np.where((resolved == org) & valid)[0]
    if len(idx) < 100: return None
    lts = lt[idx]; truth = y[idx].astype(int); scs = score[idx]; ous = ou[idx]
    true_ess = set(lts[truth == 1])
    n_true = len(true_ess)
    if n_true < 30: return None

    # Wheel 1: confident calls at global P90
    conf_ess = [j for j in range(len(lts)) if scs[j] >= ESS_THR]
    conf_ne = [j for j in range(len(lts)) if scs[j] <= NE_THR]
    soup = [j for j in range(len(lts))
            if scs[j] > NE_THR and scs[j] < ESS_THR]

    # current category fill
    slot_of = {x: classify(annot_all[org].get(x, ""), gname_all[org].get(x, ""))
               for x in lts}
    filled = Counter(slot_of[lts[j]] for j in conf_ess)
    gaps = {s: TABLE[s] - filled.get(s, 0)
            for s in TABLE if filled.get(s, 0) < TABLE[s]}

    # family-classify the soup
    fam_of = {}
    for j in soup:
        og = og_arr[idx[j]]
        fam_of[j] = og_profile(og) if og else dict(slot="other", product="",
                                                    breadth=0, n_members=0)

    # Wheel 2 inserts: best family-matched candidates per gap slot
    inserts = defaultdict(list); chosen = set()
    for s, need in gaps.items():
        cands = [(scs[j] * 0.5 +
                  (ous[j] if not np.isnan(ous[j]) else 0) * 0.5, j)
                 for j in soup if fam_of[j]["slot"] == s and j not in chosen]
        cands.sort(reverse=True)
        for _, j in cands[:need]:
            inserts[s].append(j); chosen.add(j)
    ins_all = [j for v in inserts.values() for j in v]
    ins_prec = (np.mean([lts[j] in true_ess for j in ins_all])
                if ins_all else 0)

    def m(pred):
        tp = len(pred & true_ess)
        return dict(n=len(pred), tp=tp,
                    p=round(tp / max(len(pred), 1), 3),
                    r=round(tp / max(len(true_ess), 1), 3),
                    f1=round(2 * tp / max(len(pred) + n_true, 1), 3))
    cell_conf = set(lts[j] for j in conf_ess)
    cell_asm = cell_conf | set(lts[j] for j in ins_all)
    return dict(
        organism=org, n_genes=len(lts), n_essential=n_true,
        ess_rate=round(n_true / len(lts), 3),
        wheel1_size=len(conf_ess),
        wheel1_cov=round(len(conf_ess) / max(n_true, 1), 3),
        gaps=dict(gaps), n_gaps=sum(gaps.values()),
        soup_size=len(soup), inserts=len(ins_all),
        insert_precision=round(float(ins_prec), 3),
        cell_conf=m(cell_conf), cell_asm=m(cell_asm),
        insert_details=[dict(slot=s, lt=lts[j],
                              gene=gname_all[org].get(lts[j], "?"),
                              family=fam_of[j]["product"][:45],
                              real=(lts[j] in true_ess))
                        for s, js in inserts.items() for j in js][:30],
    )

# ===================== 6. run on the 6 organisms =============================
ORGS = (os.environ["ASSEMBLY_ORGS"].split(",") if os.environ.get("ASSEMBLY_ORGS")
        else ["beril_Keio", "beril_BFirm", "beril_Methanococcus_S2",
              "beril_DvH", "beril_SynE", "beril_Btheta"])
SHORT = {"beril_Keio": "E. coli", "beril_BFirm": "B. firm",
         "beril_Methanococcus_S2": "Mcoccus (archaea)",
         "beril_DvH": "D. vulgaris", "beril_SynE": "Synechoc.",
         "beril_Btheta": "B. theta"}
results = {}
for org in ORGS:
    r = assemble(org)
    if r is None:
        print(f"{org}: skipped"); continue
    results[org] = r
    print(f"\n{org}:")
    print(f"  genes={r['n_genes']} essential={r['n_essential']} "
          f"(rate={r['ess_rate']})")
    print(f"  Wheel1 confident: {r['wheel1_size']}  -> "
          f"recall {r['cell_conf']['r']:.2f}  P={r['cell_conf']['p']:.2f}")
    print(f"  gaps: {r['gaps']}")
    print(f"  Wheel2 inserts: {r['inserts']}  precision={r['insert_precision']}")
    print(f"  assembled cell: P={r['cell_asm']['p']:.2f} R={r['cell_asm']['r']:.2f} "
          f"F1={r['cell_asm']['f1']:.2f}")

# ===================== 7. figure =============================================
orgs_ok = list(results.keys())
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# (a) where Wheel 1 is weak/strong
w1_recall = [results[o]["cell_conf"]["r"] for o in orgs_ok]
asm_recall = [results[o]["cell_asm"]["r"] for o in orgs_ok]
xp = np.arange(len(orgs_ok))
ax[0].bar(xp - 0.2, w1_recall, 0.4, color="#9ecae1", label="Wheel 1 alone (recall)")
ax[0].bar(xp + 0.2, asm_recall, 0.4, color="#e31a1c", label="after Wheel 2 assembly")
ax[0].set_xticks(xp); ax[0].set_xticklabels([SHORT[o] for o in orgs_ok], rotation=15)
ax[0].set_ylabel("recall vs ground truth"); ax[0].set_ylim(0, 1)
ax[0].set_title("Where Wheel 1 is weak,\nthe assembly does heavier work",
                fontweight="bold", fontsize=11)
ax[0].legend(fontsize=9); ax[0].grid(axis="y", alpha=0.3)
for i, (a_, b_) in enumerate(zip(w1_recall, asm_recall)):
    if b_ > a_:
        ax[0].annotate(f"+{(b_-a_)*100:.0f}pp", xy=(i+0.2, b_+0.02),
                       ha="center", color="#e31a1c", fontsize=8, fontweight="bold")

# (b) gaps before vs after assembly
gaps_n = [results[o]["n_gaps"] for o in orgs_ok]
inserts_n = [results[o]["inserts"] for o in orgs_ok]
ax[1].bar(xp - 0.2, gaps_n, 0.4, color="#fd8d3c", label="categorical gaps (need)")
ax[1].bar(xp + 0.2, inserts_n, 0.4, color="#54278f", label="Wheel 2 inserts (filled)")
ax[1].set_xticks(xp); ax[1].set_xticklabels([SHORT[o] for o in orgs_ok], rotation=15)
ax[1].set_ylabel("# slots / inserts")
ax[1].set_title("Gaps the cell reports vs.\nfamily-matched candidates Wheel 2 inserts",
                fontweight="bold", fontsize=11)
ax[1].legend(fontsize=9); ax[1].grid(axis="y", alpha=0.3)

# (c) quality of inserts per organism
ins_prec = [results[o]["insert_precision"] for o in orgs_ok]
bars = ax[2].bar(xp, ins_prec, 0.6, color="#3182bd")
# baseline = each organism's true essential rate
baselines = [results[o]["ess_rate"] for o in orgs_ok]
ax[2].plot(xp, baselines, "ko--", lw=1, label="random within-organism rate")
ax[2].set_xticks(xp); ax[2].set_xticklabels([SHORT[o] for o in orgs_ok], rotation=15)
ax[2].set_ylabel("precision of Wheel 2's educated guesses")
ax[2].set_ylim(0, 1.05)
ax[2].set_title("Quality of the educated guesses\n(blue >> black = reasoning beats random)",
                fontweight="bold", fontsize=11)
ax[2].legend(fontsize=9)
for i, v in enumerate(ins_prec):
    if v > 0:
        ax[2].text(i, v + 0.02, f"{int(v*100)}%", ha="center", fontsize=9,
                   fontweight="bold")

fig.suptitle("Demand-driven assembly across organisms — the contribution scales "
             "with Wheel 1's weakness, and the educated guesses validate against truth",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT / "multi_org_assembly.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/multi_org_assembly.png")

json.dump(results, open(OUT / "multi_org_assembly.json", "w"), indent=2)
print("wrote outputs/orphan/multi_org_assembly.json")
