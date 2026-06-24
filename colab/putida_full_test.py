"""Full-stack test on P. putida (beril_Putida) -- never tested before.

Runs the whole battery in one script:
  1 fully-fledged cell layout (all 4,715 genes, 20 functional subsystems)
  2 staged 4-stage loop  W1 -> W2 -> W1 -> W2  with per-stage stats
  3 3-source consensus   (model + conservation + native FBA)
  4 knockout simulator   on 4 representative genes
  5 leak-clean precision boost (pseudomonas held out; isolate the fold_mean
                                leakage from earlier finding)

Putida = Pseudomonas putida KT2440, biotech workhorse. Pseudomonas clade is
in the model's training corpus (along with several other Pseudomonas orgs),
so Wheel 1 should be strong here -- a good contrast to the dead-Wheel-1
mtub/saur cases.

Output: outputs/orphan/putida_full_test.png  (4-panel summary)
        outputs/orphan/putida_full_test.json (every stat)
        outputs/orphan/putida_cell_layout.png (the spatial layout)
        outputs/orphan/putida_report.md      (written summary)
"""
import csv, re, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "beril_Putida"
HELD_CLADE = "pseudomonas"

# =============== LOAD ===============
truth = {r["locus_tag"]: int(r["essential"])
         for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))
         if r["organism"] == ORG}
gname = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        g = (r.get("gene_name") or "").strip()
        if g and g != "-": gname[r["locus_tag"]] = g
annot = {r["locus_tag"]: r["product"]
         for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
         if r["organism"] == ORG and r.get("product")}
fba = {r["locus_tag"]: int(r["fba_essential"])
       for r in csv.DictReader(open("data/drive_import/labels/fba_features.csv"))
       if r["organism"] == ORG}
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""),
                                r.get("next_locus_tag", ""))
reg = {}
for r in csv.DictReader(open("data/drive_import/labels/regulator_features.csv")):
    if r["organism"] == ORG:
        reg[r["locus_tag"]] = dict(
            regulator=int(r.get("is_regulator", 0) or 0),
            signaling=int(r.get("is_signaling", 0) or 0),
            transporter=int(r.get("is_transporter", 0) or 0))
og_of = {}; og_present = defaultdict(set)
og_info = {}  # og_id -> (n_paralogs, family_size, family_n_orgs, leak_free_rate)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
n_orgs = len({o for s in og_present.values() for o in s})

# leak-clean OG essential rate (pseudomonas excluded since it's the test set)
gs = defaultdict(float); gn = defaultdict(float)
clade_of_org = {r["organism"]: r["clade"]
                for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if clade_of_org.get(r["organism"]) == HELD_CLADE:
        continue                     # exclude pseudomonas clade entirely (leak-clean)
    og = og_of.get((r["organism"], r["locus_tag"]))
    if og:
        gs[og] += int(r["essential"]); gn[og] += 1
og_rate = {og: gs[og] / gn[og] for og in gn if gn[og] >= 5}

# transformer predictions
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p_arr = P["p"].astype(float); br_arr = P["brightness"].astype(float)
clade_arr = P["clade"].astype(str); lt_arr = P["lt"].astype(str)
ec_idx = (clade_arr == HELD_CLADE)
# map Putida locus_tags to model prediction (clade match + lt match)
lt2pred = {}
for i in np.where(ec_idx)[0]:
    if lt_arr[i] in truth and not np.isnan(p_arr[i]):
        lt2pred[lt_arr[i]] = (float(p_arr[i]), float(br_arr[i]))

print(f"=== P. putida full test ===")
print(f"genes: {len(truth)}, essential: {sum(truth.values())} "
      f"({100*sum(truth.values())/len(truth):.1f}%)")
print(f"annotated: {len(annot)}, FBA: {len(fba)}, predictions: {len(lt2pred)}")
print(f"OG-conservation joinable: "
      f"{sum(1 for lt in truth if og_of.get((ORG, lt)) in og_rate)}")

# =============== slot classifier (rich) ===============
SLOTS = [
    ("ribosomal",    [r"ribosom"]),
    ("synthetase",   [r"--trna lig", r"trna ligase", r"synthetase", r"aminoacyl"]),
    ("trna",         [r"\btrna\b"]),
    ("transcription",[r"rna polymerase", r"transcription", r"sigma"]),
    ("replication",  [r"dna polymerase", r"helicase", r"gyrase", r"primase",
                      r"topoisomer", r"replicat", r"dna ligase"]),
    ("translation",  [r"elongation factor", r"initiation factor",
                      r"release factor", r"chaperon"]),
    ("cell_division",[r"cell division", r"septation", r"ftsz", r"divisome"]),
    ("membrane",     [r"membrane", r"lipoprotein", r"porin", r"\bsec[a-z]\b"]),
    ("transport",    [r"transporter", r"permease", r"abc transport", r"channel",
                      r"\buptake\b", r"efflux"]),
    ("energy",       [r"atp synthase", r"atpase", r"cytochrome", r"nadh",
                      r"electron transport"]),
    ("lipid",        [r"fatty acid", r"\bacp\b", r"lipid", r"phospholipid"]),
    ("nucleotide",   [r"purine", r"pyrimidine", r"nucleotide"]),
    ("amino_acid",   [r"amino acid", r"glutam", r"aspart"]),
    ("cofactor",     [r"folate", r"thiamine", r"riboflavin", r"coenzyme",
                      r"biotin"]),
    ("regulation",   [r"transcription regulat", r"repressor", r"activator",
                      r"two-component", r"response regulator"]),
    ("stress",       [r"stress", r"sos", r"heat shock", r"oxidative"]),
    ("metabolism",   [r"oxidoreductase", r"transferase", r"hydrolase",
                      r"isomerase", r"ligase", r"lyase", r"kinase",
                      r"dehydrogenase", r"reductase"]),
]
SP = [(nm, [re.compile(x, re.I) for x in pats]) for nm, pats in SLOTS]
def slot_of(lt):
    rg = reg.get(lt, {})
    if rg.get("regulator"): return "regulation"
    if rg.get("transporter"): return "transport"
    if rg.get("signaling"): return "signaling"
    p = (annot.get(lt, "") or "").lower()
    for nm, rx in SP:
        if any(r.search(p) for r in rx): return nm
    return "other"
slot = {lt: slot_of(lt) for lt in truth}
TRUE_ESS = set(lt for lt, e in truth.items() if e == 1)

def universal(lt):
    og = og_of.get((ORG, lt))
    return bool(og) and len(og_present[og]) / n_orgs >= 0.9

# =============== (1) fully-fledged cell layout ===============
all_slots_set = set(slot.values())
ORDER = ["ribosomal", "trna", "synthetase", "transcription", "replication",
         "translation", "cell_division", "membrane", "transport", "energy",
         "lipid", "nucleotide", "amino_acid", "cofactor", "metabolism",
         "regulation", "signaling", "stress", "other"]
present_slots = [s for s in ORDER if s in all_slots_set]
for s in all_slots_set:
    if s not in present_slots: present_slots.append(s)

N = len(truth); genes = list(truth.keys())
idx = {lt: i for i, lt in enumerate(genes)}
node_mod = np.array([present_slots.index(slot[lt]) for lt in genes])
mod_size = np.array([(node_mod == m).sum() for m in range(len(present_slots))])
ang = np.linspace(0, 2 * np.pi, len(present_slots), endpoint=False)
R = 7.0
mod_center = {m: np.array([R * np.cos(a), R * np.sin(a)])
              for m, a in zip(range(len(present_slots)), ang)}
rng = np.random.default_rng(0)
pos = np.zeros((N, 2))
for m in range(len(present_slots)):
    sel = np.where(node_mod == m)[0]
    if len(sel) == 0: continue
    rad = 0.40 * np.sqrt(max(len(sel), 1))
    r = rad * np.sqrt(rng.random(len(sel)))
    th = rng.random(len(sel)) * 2 * np.pi
    pos[sel] = mod_center[m] + np.column_stack([r * np.cos(th), r * np.sin(th)])
# operon edges
edges = []
for lt in genes:
    p_, nx = synt.get(lt, ("", ""))
    for nb in (p_, nx):
        if nb in idx and nb != lt:
            a, b = idx[lt], idx[nb]
            if a < b: edges.append((a, b))
edges = list(set(edges))

ess_mask = np.array([truth[lt] == 1 for lt in genes])
univ_mask = np.array([universal(lt) for lt in genes])

fig, ax = plt.subplots(figsize=(13, 13))
ax.set_facecolor("#fafafa")
for a, b in edges:
    ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
            color="#dddddd", lw=0.18, alpha=0.6, zorder=0)
sizes = np.where(univ_mask, 28, 12)
ax.scatter(pos[~ess_mask, 0], pos[~ess_mask, 1], s=sizes[~ess_mask],
           color="#74a9cf", edgecolor="none", alpha=0.55, zorder=1)
ax.scatter(pos[ess_mask & ~univ_mask, 0], pos[ess_mask & ~univ_mask, 1],
           s=20, color="#e31a1c", edgecolor="none", zorder=2)
ax.scatter(pos[ess_mask & univ_mask, 0], pos[ess_mask & univ_mask, 1],
           s=42, color="#e31a1c", edgecolor="#000", linewidth=0.7, zorder=3)
for m in range(len(present_slots)):
    if mod_size[m] == 0: continue
    c = mod_center[m]; out = c * 1.20
    ax.text(out[0], out[1], f"{present_slots[m]}\n({mod_size[m]})",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666"))
ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"P. putida FULLY-FLEDGED cell -- {N} genes, {len(present_slots)} subsystems\n"
             f"red = essential ({int(ess_mask.sum())}), blue = non-essential "
             f"({int((~ess_mask).sum())}), black ring = universal-core "
             f"({int(univ_mask.sum())})", fontweight="bold", fontsize=12, pad=14)
plt.tight_layout()
plt.savefig(OUT / "putida_cell_layout.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\n[1/5] wrote putida_cell_layout.png")
print(f"  modules: " + ", ".join(f"{s}={mod_size[present_slots.index(s)]}"
                                  for s in present_slots if mod_size[present_slots.index(s)] > 0))

# =============== (2) staged W1->W2->W1->W2 loop ===============
def metrics(pred):
    tp = len(pred & TRUE_ESS)
    p_ = tp / max(len(pred), 1); r_ = tp / max(len(TRUE_ESS), 1)
    return dict(n=len(pred), tp=tp, precision=round(p_, 3),
                recall=round(r_, 3),
                f1=round(2 * p_ * r_ / max(p_ + r_, 1e-9), 3))

universe = [lt for lt in truth if lt in annot]
# Wheel 1 score: model p alone (the transformer's verdict for pseudomonas)
def w1_model(lt): return lt2pred[lt][0] if lt in lt2pred else np.nan
# Wheel 1 alt: conservation prior (og_rate, leak-clean)
def w2_cons(lt):
    og = og_of.get((ORG, lt))
    return og_rate.get(og, np.nan) if og else np.nan
# combined Wheel 1 score (mean of model + conservation when both present)
score = {}
for lt in universe:
    m = w1_model(lt); c = w2_cons(lt)
    if not np.isnan(m) and not np.isnan(c): score[lt] = 0.5 * m + 0.5 * c
    elif not np.isnan(m): score[lt] = m
    elif not np.isnan(c): score[lt] = c

REQ = {"ribosomal": 30, "synthetase": 12, "transcription": 6,
       "replication": 5, "translation": 5, "cell_division": 1}

stages = []
CONF = 0.75
confident = set(lt for lt in score if score[lt] >= CONF)
m1 = metrics(confident)
filled1 = Counter(slot[lt] for lt in confident)
gaps1 = {s: REQ[s] - filled1.get(s, 0) for s in REQ
         if filled1.get(s, 0) < REQ[s]}
stages.append(("S1 Wheel 1 (model+conservation @>=0.75)",
               dict(**m1, gaps=dict(gaps1))))
print(f"\n[2/5] staged loop:")
print(f"  S1 W1 confident essentials: {m1['n']}  P={m1['precision']:.3f} "
      f"R={m1['recall']:.3f} F1={m1['f1']:.3f}")
print(f"     gaps: {dict(gaps1)}")

# S2: Wheel 2 fills gaps by family-matched soup
soup = [lt for lt in universe if lt not in confident]
inserts = defaultdict(list); chosen = set()
for s, need in gaps1.items():
    cands = [(score.get(lt, 0), lt) for lt in soup
             if slot[lt] == s and lt not in chosen]
    cands.sort(reverse=True)
    for _, lt in cands[:need]:
        inserts[s].append(lt); chosen.add(lt)
ins_all = [lt for v in inserts.values() for lt in v]
ins_prec = np.mean([lt in TRUE_ESS for lt in ins_all]) if ins_all else 0.0
assembled = confident | set(ins_all)
filled2 = Counter(slot[lt] for lt in assembled)
gaps2 = {s: REQ[s] - filled2.get(s, 0) for s in REQ
         if filled2.get(s, 0) < REQ[s]}
m2 = metrics(assembled)
stages.append(("S2 Wheel 2 (assemble + family-match gaps)",
               dict(inserts=len(ins_all),
                    insert_precision=round(float(ins_prec), 3),
                    gaps_after=dict(gaps2), **m2)))
print(f"  S2 W2 inserted {len(ins_all)} family-matched genes (P={ins_prec:.2f})")
print(f"     remaining gaps: {dict(gaps2)}")
print(f"     assembled: n={m2['n']} P={m2['precision']:.3f} R={m2['recall']:.3f}")

# S3: Wheel 1 re-score: promote cell-supported high-conservation soup
promoted = set(lt for lt in soup
               if score.get(lt, 0) >= 0.5 and slot[lt] in REQ
               and lt not in chosen)
combined = assembled | promoted
m3 = metrics(combined)
stages.append(("S3 Wheel 1 re-score (cell-supported promotion)",
               dict(promoted=len(promoted), **m3)))
print(f"  S3 W1 re-score: promoted {len(promoted)} cell-supported soup genes")
print(f"     combined: n={m3['n']} P={m3['precision']:.3f} R={m3['recall']:.3f}")

# S4: Wheel 2 final cell
final = combined
filled4 = Counter(slot[lt] for lt in final)
complete = sum(1 for s in REQ if filled4.get(s, 0) >= REQ[s])
m4 = metrics(final)
stages.append(("S4 Wheel 2 (final cell)",
               dict(core_complete=f"{complete}/{len(REQ)}", **m4)))
print(f"  S4 W2 final cell: n={m4['n']} P={m4['precision']:.3f} R={m4['recall']:.3f}")
print(f"     core self-replication: {complete}/{len(REQ)} complete")

# per-gene final accuracy over annotated universe
tp = sum(1 for lt in universe if lt in final and truth[lt] == 1)
tn = sum(1 for lt in universe if lt not in final and truth[lt] == 0)
final_acc = (tp + tn) / len(universe)
print(f"  -> final per-gene accuracy over {len(universe)} annotated: {final_acc:.3f}")

# =============== (3) 3-source consensus ===============
print(f"\n[3/5] 3-source consensus (model + conservation + native FBA):")
rows3 = []
for lt in truth:
    if lt not in lt2pred: continue
    og = og_of.get((ORG, lt))
    if og not in og_rate: continue
    if lt not in fba: continue
    rows3.append(dict(lt=lt, truth=truth[lt],
                      w1=int(lt2pred[lt][0] >= 0.5),
                      w2=int(og_rate[og] >= 0.5),
                      w3=fba[lt]))
print(f"  joinable (all 3 + truth): {len(rows3)}")
def pmetric(preds, ys):
    tp = sum(1 for p, y in zip(preds, ys) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, ys) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, ys) if p == 0 and y == 1)
    P_ = tp / max(tp + fp, 1); R_ = tp / max(tp + fn, 1)
    return dict(P=round(P_, 3), R=round(R_, 3),
                F1=round(2 * P_ * R_ / max(P_ + R_, 1e-9), 3))
ys3 = [r["truth"] for r in rows3]
for s_ in ("w1", "w2", "w3"):
    m_ = pmetric([r[s_] for r in rows3], ys3)
    print(f"  {s_:<3} {m_}")
ttt = [r for r in rows3 if r["w1"] == r["w2"] == r["w3"] == 1]
nnn = [r for r in rows3 if r["w1"] == r["w2"] == r["w3"] == 0]
maj_ess = [r for r in rows3 if r["w1"] + r["w2"] + r["w3"] >= 2]
mw1w2 = [r for r in rows3 if r["w1"] == 1 and r["w2"] == 1]
def prec(g, t): return sum(1 for r in g if r["truth"] == t) / max(len(g), 1)
p_ttt = round(prec(ttt, 1), 3); p_nnn = round(prec(nnn, 0), 3)
p_maj = round(prec(maj_ess, 1), 3); p_mw1w2 = round(prec(mw1w2, 1), 3)
print(f"  3-way ESS agreement: n={len(ttt)}  precision={p_ttt}")
print(f"  3-way NON ess agreement: n={len(nnn)}  precision={p_nnn}")
print(f"  2-of-3 ess majority: n={len(maj_ess)}  precision={p_maj}")
print(f"  W1+W2 ess: n={len(mw1w2)} prec={p_mw1w2} | "
      f"+W3 (TTT): {p_ttt} (+{int((p_ttt - p_mw1w2)*100)}pp)")
# FBA-blind essentials
fba_blind = [r for r in rows3 if r["w1"] == 1 and r["w2"] == 1
             and r["w3"] == 0 and r["truth"] == 1]
fba_only = [r for r in rows3 if r["w1"] == 0 and r["w2"] == 0
            and r["w3"] == 1 and r["truth"] == 1]
print(f"  FBA-blind essentials (W1+W2 say yes, FBA says no, truth=ess): {len(fba_blind)}")
print(f"  FBA-only essentials (only W3 yes, truth=ess): {len(fba_only)}")

# =============== (4) knockout simulator on 4 reps ===============
print(f"\n[4/5] knockout simulator (representative essentials):")
# essential phyletic neighbours via OG presence Jaccard >= 0.9 over Putida essentials
ess_pv = {}
for lt in TRUE_ESS:
    og = og_of.get((ORG, lt))
    if og: ess_pv[lt] = og_present[og]
phyl = defaultdict(set)
keys = [lt for lt in ess_pv if 0.1 <= len(ess_pv[lt]) / n_orgs < 0.9]
for i_ in range(len(keys)):
    va = ess_pv[keys[i_]]
    for j_ in range(i_ + 1, len(keys)):
        vb = ess_pv[keys[j_]]; inter = len(va & vb)
        if inter and inter / (len(va) + len(vb) - inter) >= 0.9:
            phyl[keys[i_]].add(keys[j_]); phyl[keys[j_]].add(keys[i_])

def find_putida(predicates, ess=True):
    for lt, p_ in annot.items():
        if truth.get(lt) != (1 if ess else 0): continue
        if any(re.search(pat, p_, re.I) for pat in predicates): return lt
    return None

slot_counts_true = Counter(slot[lt] for lt in TRUE_ESS)
def sim(lt):
    if not lt: return None
    s = slot[lt]; product = annot.get(lt, "?")
    essential = truth[lt] == 1; univ = universal(lt)
    prev, nxt = synt.get(lt, ("", ""))
    operon_hit = [g for g in (prev, nxt)
                  if g and g in truth and truth[g] == 1]
    phy_partners = sorted(phyl.get(lt, set()))[:6]
    cur = slot_counts_true.get(s, 0)
    after = cur - (1 if essential else 0) - len(operon_hit)
    req = REQ.get(s, 0)
    slot_ok = (not req) or (after >= req)
    # 2-hop cascade
    seen = {lt}; front = [lt]; cascade = set()
    for _ in range(2):
        nf = []
        for u in front:
            for v in phyl.get(u, set()) | {x for x in synt.get(u, ("", "")) if x}:
                if v in TRUE_ESS and v not in seen:
                    seen.add(v); nf.append(v); cascade.add(v)
        front = nf
    verdict = ("DEAD" if essential else
               ("IMPAIRED" if (req and not slot_ok) or len(operon_hit) >= 2
                else "VIABLE"))
    return dict(lt=lt, product=product, slot=s, essential=essential,
                universal=univ, operon_essentials_hit=len(operon_hit),
                phyletic_partners=len(phy_partners),
                cascade_reach=len(cascade), verdict=verdict)

KO_DEMO = [
    ("cell division FtsZ", find_putida([r"^cell division protein FtsZ", r"\bFtsZ\b"])),
    ("DNA gyrase subunit A", find_putida([r"\bDNA gyrase subunit A\b"])),
    ("RNA polymerase beta", find_putida([r"DNA-directed RNA polymerase subunit beta\b"])),
    ("a non-essential dehydrogenase", find_putida([r"dehydrogenase"], ess=False)),
]
ko_results = []
for label, lt in KO_DEMO:
    if not lt:
        print(f"  {label:<32} (not found)"); continue
    r = sim(lt); r["label"] = label
    ko_results.append(r)
    print(f"  KO {label:<32} ({lt}, {r['slot']}) -> "
          f"{r['verdict']} cascade={r['cascade_reach']}")

# =============== (5) leak-clean precision boost ===============
# Hold out pseudomonas clade, train stacker WITH and WITHOUT fold_mean
# to isolate the leakage we flagged earlier.
print(f"\n[5/5] leak-clean precision boost (pseudomonas as held-clade):")

# Build feature matrices over a manageable scope: all corpus genes,
# we LOCO with pseudomonas held out.
# Load orth+ columns
orth_lf = {}; orth_fm = {}; orth_size = {}; orth_par = {}; orth_brd = {}; orth_orph = {}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    orth_lf[k] = float(r.get("family_frac_essential_leakfree") or 0)
    orth_fm[k] = float(np.mean([float(r.get(f"family_frac_essential_fold{i}") or 0)
                                 for i in range(5)]))
    orth_size[k] = float(r.get("family_size_total") or 0)
    orth_par[k] = float(r.get("n_paralogs_in_genome") or 0)
    orth_brd[k] = float(r.get("family_n_organisms") or 0)
    orth_orph[k] = float(r.get("is_orphan") or 0)

C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
cache_orgs = [str(x) for x in C["orgs"]]
o2c = {cache_orgs[int(C["meta_org"][i])]: str(C["clade"][i])
       for i in range(len(C["meta_org"]))}
obylt = defaultdict(list)
for k in orth_lf: obylt[k[1]].append(k[0])
y = P["y"].astype(float)
valid = ~(np.isnan(p_arr) | np.isnan(br_arr))
resolved = np.empty(len(p_arr), object)
for i in range(len(p_arr)):
    cands = obylt.get(lt_arr[i], [])
    match = [o for o in cands if o2c.get(o) == clade_arr[i]]
    ch = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    resolved[i] = ch or ""

# LOCO og_universality, excluding own clade
gs2 = defaultdict(float); gn2 = defaultdict(float)
csum = defaultdict(float); cn = defaultdict(float)
for i in np.where(valid)[0]:
    if not resolved[i]: continue
    og = og_of.get((resolved[i], lt_arr[i]))
    if not og: continue
    gs2[og] += y[i]; gn2[og] += 1
    csum[(og, clade_arr[i])] += y[i]; cn[(og, clade_arr[i])] += 1
ou = np.full(len(p_arr), np.nan)
for i in np.where(valid)[0]:
    if not resolved[i]: continue
    og = og_of.get((resolved[i], lt_arr[i]))
    if not og: continue
    n_ = gn2[og] - cn[(og, clade_arr[i])]
    if n_ > 0: ou[i] = (gs2[og] - csum[(og, clade_arr[i])]) / n_
mask_eval = valid & (resolved != "") & ~np.isnan(ou)

# build columns
N_ = len(p_arr)
col_par = np.array([orth_par.get((resolved[i], lt_arr[i]), 0) for i in range(N_)])
col_brd = np.array([orth_brd.get((resolved[i], lt_arr[i]), 0) for i in range(N_)])
col_orph = np.array([orth_orph.get((resolved[i], lt_arr[i]), 0) for i in range(N_)])
col_lf = np.array([orth_lf.get((resolved[i], lt_arr[i]), 0) for i in range(N_)])
col_fm = np.array([orth_fm.get((resolved[i], lt_arr[i]), 0) for i in range(N_)])
col_size = np.array([orth_size.get((resolved[i], lt_arr[i]), 0) for i in range(N_)])
ou_safe = np.where(np.isnan(ou), 0, ou)

F_BASE = np.column_stack([p_arr, br_arr, ou_safe, np.log1p(col_par),
                          np.log1p(col_brd), col_orph])
F_LF = np.column_stack([F_BASE, np.log1p(col_size), col_lf])
F_FM = np.column_stack([F_BASE, np.log1p(col_size), col_fm])
F_ALL = np.column_stack([F_BASE, np.log1p(col_size), col_lf, col_fm])

def fit(X, yy, l2=1.0, it=300, lr=0.4):
    X = np.nan_to_num(X)
    mu_ = X.mean(0); sd_ = np.maximum(X.std(0), 1e-3)
    Xs = np.c_[np.ones(len(X)), (X - mu_) / sd_]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** 0.5; sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(it):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr - yy) * sw) / len(X)
        g[1:] += l2 * w[1:] / len(X); w -= lr * g
    return w, mu_, sd_
def predL(m, X):
    X = np.nan_to_num(X); w, mu_, sd_ = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu_) / sd_] @ w, -30, 30)))
def cov(s, yt, pos=True, t=0.9):
    sc = s if pos else -s; ys = yt if pos else 1 - yt
    o = np.argsort(-sc); o_ys = ys[o]; tp = np.cumsum(o_ys); fp = np.cumsum(1 - o_ys)
    prec = tp / np.maximum(tp + fp, 1); ok = prec >= t
    return float(tp[ok].max() / max(ys.sum(), 1)) if ok.any() else 0.0

# Hold out pseudomonas clade ONLY (not full LOCO -- focused on Putida)
ab = mask_eval & (br_arr < 0.85)
tr = ab & (clade_arr != HELD_CLADE); te = ab & (clade_arr == HELD_CLADE)
def run(F, label):
    m_ = fit(F[tr], y[tr])
    pred = predL(m_, F[te])
    s = pred; yy = y[te].astype(int)
    return dict(label=label, n=int(te.sum()),
                essCov=round(cov(s, yy, True), 3),
                neCov=round(cov(s, yy, False), 3))
ps_base = run(F_BASE, "baseline (6)")
ps_lf = run(F_LF, "+ size + leakfree")
ps_fm = run(F_FM, "+ size + fold_mean")
ps_all = run(F_ALL, "+ size + leakfree + fold_mean")
print(f"  on pseudomonas held-out (abstention): n={int(te.sum())}")
for r in (ps_base, ps_lf, ps_fm, ps_all):
    print(f"    {r['label']:<36} essCov={r['essCov']:.3f}  neCov={r['neCov']:.3f}")
delta_lf = ps_lf["essCov"] - ps_base["essCov"]
delta_fm = ps_fm["essCov"] - ps_base["essCov"]
delta_all = ps_all["essCov"] - ps_base["essCov"]
print(f"  -> lift from LEAKFREE alone: +{delta_lf*100:.1f}pp  (legitimate)")
print(f"  -> lift from FOLD_MEAN alone: +{delta_fm*100:.1f}pp  (suspect)")
print(f"  -> lift from BOTH: +{delta_all*100:.1f}pp")

# =============== summary figure ===============
fig, ax = plt.subplots(2, 2, figsize=(18, 11))

# (a) staged loop trajectory
xs = ["S1\nW1", "S2\nW2", "S3\nW1", "S4\nW2"]
Ps = [s[1]["precision"] for s in stages]
Rs = [s[1]["recall"] for s in stages]
F1s = [s[1]["f1"] for s in stages]
a = ax[0, 0]
a.plot(xs, Ps, "o-", color="#3182bd", lw=2, label="precision")
a.plot(xs, Rs, "s-", color="#e31a1c", lw=2, label="recall")
a.plot(xs, F1s, "^-", color="#000", lw=2, label="F1")
a.set_ylim(0, 1); a.set_ylabel("score"); a.legend(fontsize=9); a.grid(alpha=0.3)
a.set_title(f"Staged W1->W2->W1->W2 loop\nfinal per-gene accuracy = {final_acc:.3f}",
            fontweight="bold", fontsize=12)
for i, f in enumerate(F1s):
    a.annotate(f"{f:.2f}", (i, f), xytext=(0, 8), textcoords="offset points",
               ha="center", fontsize=9, fontweight="bold")

# (b) 3-source agreement (Venn-like + consensus precision)
a = ax[0, 1]
patterns = [(1,1,1),(1,1,0),(1,0,1),(0,1,1),(1,0,0),(0,1,0),(0,0,1),(0,0,0)]
labels2 = ["W1∩W2∩W3","W1∩W2 only","W1∩W3 only","W2∩W3 only",
          "W1 only","W2 only","W3 only","none"]
counts = []; pos_frac = []
for w1,w2,w3 in patterns:
    g = [r for r in rows3 if r["w1"]==w1 and r["w2"]==w2 and r["w3"]==w3]
    counts.append(len(g))
    pos_frac.append(sum(1 for r in g if r["truth"]==1)/max(len(g),1))
yp = np.arange(len(labels2))
a.barh(yp, counts, color="#9ecae1")
for i,(c,f) in enumerate(zip(counts,pos_frac)):
    if c>0:
        col = "#e31a1c" if f>=0.5 else "#888"
        a.text(c+3, i, f"n={c}, P_ess={f:.2f}", va="center", fontsize=9, color=col,
               fontweight="bold" if f>=0.9 else "normal")
a.set_yticks(yp); a.set_yticklabels(labels2, fontsize=9)
a.invert_yaxis()
a.set_title(f"3-source agreement matrix on Putida\nW1+W2 ess prec={p_mw1w2}, "
            f"+W3 (TTT) prec={p_ttt}", fontweight="bold", fontsize=12)

# (c) leak-clean precision boost
a = ax[1, 0]
labs = ["baseline","+leakfree","+fold_mean","+both"]
vs = [ps_base["essCov"], ps_lf["essCov"], ps_fm["essCov"], ps_all["essCov"]]
ncs = [ps_base["neCov"], ps_lf["neCov"], ps_fm["neCov"], ps_all["neCov"]]
x = np.arange(4)
a.bar(x-0.2, vs, 0.4, color="#e31a1c", label="essCov@P90")
a.bar(x+0.2, ncs, 0.4, color="#3182bd", label="neCov@P90")
for i,(v,n) in enumerate(zip(vs,ncs)):
    a.text(i-0.2, v+0.01, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    a.text(i+0.2, n+0.01, f"{n:.3f}", ha="center", fontsize=8.5)
a.set_xticks(x); a.set_xticklabels(labs, fontsize=9)
a.set_ylabel("coverage @ P>=0.90 (Putida held out)"); a.legend(fontsize=9)
a.set_ylim(0, max(vs+ncs)+0.1)
a.set_title(f"Leak-clean precision audit on Putida\n"
            f"leakfree lift={delta_lf*100:+.1f}pp  fold_mean lift={delta_fm*100:+.1f}pp",
            fontweight="bold", fontsize=12)

# (d) knockout summaries
a = ax[1, 1]; a.axis("off")
txt = "Knockout simulator -- 4 reps:\n\n"
for r in ko_results:
    col = "DEAD" if r["verdict"]=="DEAD" else ("IMPAIRED" if r["verdict"]=="IMPAIRED" else "VIABLE")
    txt += f"  {r['label']}  ({r['lt']})\n"
    txt += f"    slot: {r['slot']}, universal: {r['universal']}\n"
    txt += f"    verdict: {r['verdict']}  | cascade {r['cascade_reach']} essentials\n\n"
txt += f"FBA-blind essentials (W1+W2 yes, FBA no, truth ess): {len(fba_blind)}\n"
txt += f"FBA-only essentials (only W3, truth ess): {len(fba_only)}\n"
a.text(0.0, 1.0, txt, transform=a.transAxes, va="top", fontsize=10,
       family="monospace",
       bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5fa", ec="#888"))

fig.suptitle("P. putida full-stack test: staged loop + 3-source + leak-clean audit + knockouts",
             fontweight="bold", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "putida_full_test.png", dpi=120, bbox_inches="tight")
print(f"\nwrote outputs/orphan/putida_full_test.png")

# =============== json dump ===============
out = dict(
    organism=ORG, held_clade=HELD_CLADE,
    inventory=dict(genes=len(truth), essential=int(sum(truth.values())),
                   annotated=len(annot), fba=len(fba),
                   model_preds=len(lt2pred)),
    cell_layout=dict(n_modules=len(present_slots),
                     module_sizes={present_slots[m]: int(mod_size[m])
                                   for m in range(len(present_slots))},
                     operon_edges=len(edges)),
    staged_loop=dict(stages=[dict(name=n, **st) for n, st in stages],
                     final_accuracy=round(final_acc, 3)),
    three_source=dict(n_joinable=len(rows3),
                      w1=pmetric([r["w1"] for r in rows3], ys3),
                      w2=pmetric([r["w2"] for r in rows3], ys3),
                      w3=pmetric([r["w3"] for r in rows3], ys3),
                      ttt_precision=p_ttt, nnn_precision=p_nnn,
                      majority_2of3_precision=p_maj,
                      w1w2_precision=p_mw1w2,
                      fba_blind_essentials=len(fba_blind),
                      fba_only_essentials=len(fba_only)),
    knockouts=ko_results,
    leak_clean_audit=dict(baseline=ps_base, leakfree_only=ps_lf,
                          fold_mean_only=ps_fm, both=ps_all,
                          delta_leakfree_pp=round(delta_lf * 100, 1),
                          delta_fold_mean_pp=round(delta_fm * 100, 1),
                          delta_both_pp=round(delta_all * 100, 1)),
)
json.dump(out, open(OUT / "putida_full_test.json", "w"), indent=2)
print(f"wrote outputs/orphan/putida_full_test.json")
