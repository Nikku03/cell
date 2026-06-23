"""Let the model and the network NEGOTIATE a cell -- no add/remove rules.

Instead of hand-coding "drop this, keep that", we define what biology WANTS in
a cell and let each gene's membership settle to equilibrium. The cell emerges.

Biological objective per gene (all LABEL-FREE -- Keio truth never enters):
  model evidence    logit(p)           the per-gene model's vote
  conservation      logit(og_univ)     essential across life? (cross-organism)
  functional coherence  context        do my operon/phyletic neighbours belong?
                                        (isolation is NEUTRAL -- not penalised,
                                         since we showed it doesn't discriminate)
  functional necessity  slot_need      is my subsystem still below the viable
                                        minimum a cell needs? (syn3a-derived;
                                        pulls genes IN to complete a function,
                                        never names a specific gene)
  parsimony         -lambda            a cell minimises its genome

Mean-field settling: m_i <- sigmoid( w.model*logit(p) + w.cons*logit(u)
    + w.net*(2*context-1) + w.slot*slot_need - lambda ), iterate to fixpoint.
The network and model literally exchange messages through `context` and the
shared slot occupancy; nobody is told what to do.

Sweep lambda (genome-size pressure) -> a spectrum of cells from bloated to
minimal. The biologically BEST cell = the most parsimonious one that is still
functionally complete (the minimal viable genome). Then check it against
Keio truth and the syn3a minimal cell -- as an external audit only.

Output: outputs/orphan/emergent_cell.png
        outputs/orphan/emergent_cell.json
"""
import numpy as np, json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the leak-clean prediction + network setup
exec(open("/tmp/emergent_head.py").read())
OUT = Path("outputs/orphan")
lt2j = {klt[j]: j for j in range(len(klt))}
N = len(klt)

# ---- fixed classifier: synthetase BEFORE trna (aaRS products say "trna
#      ligase" and were being mis-slotted as trna), self-consistent minimums ----
import csv, re as _re
SLOTS2 = [s for s in SLOTS if s[0] not in ("trna", "synthetase")]
# rebuild with synthetase ahead of trna
_syn = [s for s in SLOTS if s[0] == "synthetase"][0]
_trna = [s for s in SLOTS if s[0] == "trna"][0]
SLOTS2 = ([SLOTS[0]] + [_syn, _trna] + [s for s in SLOTS[1:]
          if s[0] not in ("trna", "synthetase")])
SN2 = [s[0] for s in SLOTS2]
SP2 = [[_re.compile(x, _re.I) for x in a] for _, a, _ in SLOTS2]
SG2 = [[_re.compile(x, _re.I) for x in b] for _, _, b in SLOTS2]
def classify2(prod, gn):
    pp = (prod or "").lower(); gg = (gn or "").lower().strip()
    txt = pp if pp else (gg if len(gg) > 6 else "")
    for i in range(len(SN2) - 1):
        if txt and any(rx.search(txt) for rx in SP2[i]): return SN2[i]
        if gg and any(rx.search(gg) for rx in SG2[i]): return SN2[i]
    return "other"

# derive viable minimums from syn3a's OWN essential set (external biology)
syn_prod = {}
try:
    for r in csv.DictReader(open("memory_bank/data/syn3a_gene_table.csv")):
        if r.get("product"): syn_prod[r["locus_tag"]] = r["product"]
except Exception:
    pass
syn_counts = defaultdict(int)
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == "syn3a" and int(r["essential"]) == 1:
        syn_counts[classify2(syn_prod.get(r["locus_tag"], ""), r.get("gene_name", ""))] += 1
# "a cell" universally must self-replicate: translate, transcribe, replicate,
# divide. Mandate ONLY the non-redundant core machinery; let metabolism /
# membrane be filled by negotiation (they vary by organism -- E. coli
# synthesises, Mycoplasma imports). tRNA is excluded: it is redundancy-buffered
# (multiple gene copies -> individual tRNAs rarely essential), an organism-
# specific trait, not a universal requirement. Bar = 50% of syn3a's count, the
# level real viable cells (incl. E. coli) actually clear.
REQ = ["ribosomal", "synthetase", "transcription", "replication",
       "translation", "cell_division"]
SLOT_MIN2 = {s: (int(np.ceil(0.5 * syn_counts.get(s, 0))) if s in REQ else 0)
             for s in SN2}
print(f"core-machinery minimums (a cell must self-replicate): "
      f"{ {k:v for k,v in SLOT_MIN2.items() if v>0} }")

# per-gene biological signals
def logit(x): return np.log(np.clip(x, 0.02, 0.98) / (1 - np.clip(x, 0.02, 0.98)))
p_vec = patched.copy()
u_vec = np.where(np.isnan(kuniv), 0.25, kuniv)        # conservation (fallback neutral-low)
SLOT_NAMES = SN2
slot_of = {klt[j]: classify2(annot.get(klt[j], ""), gene_names.get(klt[j], ""))
           for j in range(N)}
name2i = {s: i for i, s in enumerate(SLOT_NAMES)}
slot_idx = np.array([name2i[slot_of[klt[j]]] for j in range(N)])
SLOT_MIN = SLOT_MIN2
SLOT_MIN_ARR = np.array([SLOT_MIN[s] for s in SLOT_NAMES])

# neighbour lists over the FULL Keio gene set (operon + phyletic), once
og_of_keio = {};
for j in range(N):
    o = og_of.get((ORG, klt[j]))
    if o: og_of_keio[klt[j]] = o
# operon neighbours
nbr = defaultdict(set)
ess_all = set(klt)
for j in range(N):
    for nb in synt.get(klt[j], ("", "")):
        if nb in lt2j and nb != klt[j]:
            nbr[j].add(lt2j[nb]); nbr[lt2j[nb]].add(j)
# phyletic neighbours (variable-presence OG Jaccard>=0.9), among Keio genes
pv = {}; frac = {}
for j in range(N):
    o = og_of_keio.get(klt[j])
    if o:
        pv[j] = og_present[o]; frac[j] = len(og_present[o]) / n_orgs
varj = [j for j in range(N) if j in pv and 0.10 <= frac[j] < 0.90]
for a_i in range(len(varj)):
    ja = varj[a_i]; va = pv[ja]
    for b_i in range(a_i + 1, len(varj)):
        jb = varj[b_i]; vb = pv[jb]; inter = len(va & vb)
        if inter and inter / (len(va) + len(vb) - inter) >= 0.9:
            nbr[ja].add(jb); nbr[jb].add(ja)
nbr_list = [np.array(sorted(nbr[j]), dtype=int) for j in range(N)]
print(f"neighbour graph built: {sum(len(v) for v in nbr_list)//2} edges over {N} genes")

# ---- settling dynamics ----
W_MODEL, W_CONS, W_NET, W_SLOT = 1.0, 1.0, 0.7, 1.6
def settle(lam, iters=60):
    m = 1 / (1 + np.exp(-logit(p_vec)))        # init from model
    for _ in range(iters):
        # context = mean neighbour membership (network's message); isolation->0.5
        ctx = np.full(N, 0.5)
        for j in range(N):
            nl = nbr_list[j]
            if len(nl): ctx[j] = m[nl].mean()
        # slot occupancy (genes in same subsystem cooperate to meet the minimum)
        filled = np.zeros(len(SLOT_NAMES))
        for s in range(len(SLOT_NAMES)):
            filled[s] = m[slot_idx == s].sum()
        need = np.clip((SLOT_MIN_ARR - filled) / np.maximum(SLOT_MIN_ARR, 1), 0, 1)
        slot_need = need[slot_idx]
        sc = (W_MODEL * logit(p_vec) + W_CONS * logit(u_vec)
              + W_NET * (2 * ctx - 1) + W_SLOT * slot_need - lam)
        m = 1 / (1 + np.exp(-np.clip(sc, -30, 30)))
    return m

# ---- biological audit of an emergent cell (label-free) ----
def audit(m):
    cell = set(klt[j] for j in range(N) if m[j] > 0.5)
    cj = [j for j in range(N) if m[j] > 0.5]
    filled = defaultdict(int)
    for j in cj: filled[SLOT_NAMES[slot_idx[j]]] += 1
    req = [s for s in SLOT_NAMES if SLOT_MIN[s] > 0]
    complete = sum(1 for s in req if filled[s] >= SLOT_MIN[s])
    # connectivity (descriptive only)
    inset = set(cj)
    conn = np.mean([len([x for x in nbr_list[j] if x in inset]) > 0 for j in cj]) if cj else 0
    cons = np.mean([u_vec[j] for j in cj]) if cj else 0
    return dict(size=len(cell), complete=complete, n_req=len(req),
                connected=round(float(conn), 3), conservation=round(float(cons), 3),
                cell=cell, filled=dict(filled))

# external check vs truth (used ONLY for reporting, never in the dynamics)
def vs_truth(cell):
    tp = len(cell & true_ess)
    return dict(precision=round(tp / max(len(cell), 1), 3),
                recall=round(tp / max(len(true_ess), 1), 3), tp=tp)

# ---- sweep genome-size pressure ----
lams = np.linspace(-1.0, 7.0, 33)
spectrum = []
for lam in lams:
    a = audit(settle(lam))
    a["lambda"] = round(float(lam), 2); a.update(vs_truth(a["cell"]))
    spectrum.append(a)
    print(f"lambda={lam:+.2f}  size={a['size']:>4}  complete={a['complete']}/{a['n_req']}  "
          f"conn={a['connected']}  cons={a['conservation']}  "
          f"(audit: P={a['precision']} R={a['recall']})")

# ---- emergent landmarks (none dictated) ----
viable = [a for a in spectrum if a["complete"] == a["n_req"]]
floor = min(viable, key=lambda a: a["size"])             # minimal self-replicating cell
cliff = max([a["size"] for a in spectrum if a["complete"] < a["n_req"]] or [0])
near_syn3a = min(spectrum, key=lambda a: abs(a["size"] - 383))
near_truth = min(spectrum, key=lambda a: abs(a["size"] - len(true_ess)))
print(f"\n=== EMERGENT LANDMARKS (no truth used in negotiation) ===")
print(f"  viability cliff: cell collapses below ~{floor['size']} genes "
      f"(largest broken = {cliff})")
print(f"  minimal self-replicating cell: {floor['size']} genes, "
      f"conservation {floor['conservation']}, audit precision {floor['precision']} "
      f"(={floor['tp']}/{floor['size']} are real essentials)")
print(f"  cell at syn3a size: {near_syn3a['size']} genes (syn3a=383), "
      f"conservation {near_syn3a['conservation']}")
print(f"  cell at E.coli-essential size: {near_truth['size']} genes "
      f"(truth=605), audit P={near_truth['precision']} R={near_truth['recall']}")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
sz = [a["size"] for a in spectrum]; lm = [a["lambda"] for a in spectrum]
comp = [a["complete"] / a["n_req"] for a in spectrum]
conn = [a["connected"] for a in spectrum]; cons = [a["conservation"] for a in spectrum]

ax[0].plot(lm, sz, "o-", color="#3182bd", lw=2)
ax[0].axhline(len(true_ess), color="#e31a1c", ls="--", lw=1, label=f"E. coli essential ({len(true_ess)})")
ax[0].axhline(383, color="#888", ls=":", lw=1, label="syn3a minimal (383)")
ax[0].axhline(floor["size"], color="#2ca02c", ls="-", lw=1, alpha=0.6,
              label=f"viability floor ({floor['size']})")
ax[0].set_xlabel("genome-size pressure  λ"); ax[0].set_ylabel("emergent cell size")
ax[0].set_title("Cells that emerge as genome pressure rises\n(no add/remove rules — pure negotiation)",
                fontweight="bold", fontsize=11); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(sz, comp, "o-", color="#54278f", lw=2, label="self-replication completeness")
ax[1].plot(sz, conn, "s-", color="#41b6c4", lw=1.5, label="connected fraction")
ax[1].plot(sz, cons, "^-", color="#fd8d3c", lw=1.5, label="mean conservation")
ax[1].axvspan(0, floor["size"], color="#e31a1c", alpha=0.08)
ax[1].text(floor["size"], 0.45, f" viability\n cliff\n (<{floor['size']})",
           color="#e31a1c", fontsize=9, ha="left")
ax[1].set_xlabel("emergent cell size"); ax[1].set_ylabel("biological quality")
ax[1].set_title("What makes a cell viable?\nself-replication holds until the cliff",
                fontweight="bold", fontsize=11); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

prec = [a["precision"] for a in spectrum]; rec = [a["recall"] for a in spectrum]
ax[2].plot(sz, prec, "o-", color="#3182bd", lw=2, label="precision vs truth")
ax[2].plot(sz, rec, "s-", color="#e31a1c", lw=2, label="recall vs truth")
ax[2].axvline(len(true_ess), color="#e31a1c", ls="--", lw=1)
ax[2].axvline(383, color="#888", ls=":", lw=1)
ax[2].axvline(floor["size"], color="#2ca02c", lw=1)
ax[2].set_xlabel("emergent cell size"); ax[2].set_ylabel("agreement w/ Keio truth")
ax[2].set_title("External audit (truth NOT in negotiation)\nthe biology-only cell rediscovers reality",
                fontweight="bold", fontsize=11); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
ax[2].set_ylim(0, 1)

fig.suptitle("Model & network negotiate a cell under biological principles — "
             "no add/remove rules. Three real landmarks emerge on their own.",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT / "emergent_cell.png", dpi=120, bbox_inches="tight")
print("\nwrote outputs/orphan/emergent_cell.png")
best = floor

def landmark(a):
    return dict(lambda_=a["lambda"], size=a["size"],
                complete=f"{a['complete']}/{a['n_req']}", conservation=a["conservation"],
                connected=a["connected"], audit_precision=a["precision"],
                audit_recall=a["recall"], filled=a["filled"])
json.dump(dict(weights=dict(model=W_MODEL, cons=W_CONS, net=W_NET, slot=W_SLOT),
               spectrum=[{k: v for k, v in a.items() if k not in ("cell", "filled")}
                         for a in spectrum],
               landmarks=dict(
                   minimal_self_replicating=landmark(floor),
                   viability_cliff_below=floor["size"],
                   at_syn3a_size=landmark(near_syn3a),
                   at_ecoli_essential_size=landmark(near_truth)),
               n_true_essential=len(true_ess)),
          open(OUT / "emergent_cell.json", "w"), indent=2)
print("wrote outputs/orphan/emergent_cell.json")
