"""Correction: robustness is a whole-genome property, not an essential-only one.

The earlier "robust-yet-fragile" Q3 plot was wrong in spirit. Essentiality is
an AND gate: the cell is viable iff EVERY essential is present. Remove one
essential and the cell dies -- the still-connected sub-graph is a corpse.

This script:
  1) Reframes Q3 at the WHOLE-GENOME level (essential + non-essential). This
     IS where the robust-yet-fragile property actually lives: random knockout
     hits a non-essential ~83% of the time, so the cell tolerates random loss.
     Targeted hub knockout almost always lands on an essential -> death.
  2) Shows the literal AND-gate viability curve: P(cell viable | k random
     knockouts of any gene) drops as 1 - (1 - p_ess)^k roughly.
  3) Re-interprets what the essential-graph topology I plotted DOES mean
     (dependency structure, not survivability).

Output: outputs/orphan/and_gate.png
        outputs/orphan/and_gate.json
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "beril_Keio"

# ---- whole-genome labels ----
labels = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        labels[r["locus_tag"]] = int(r["essential"])
all_genes = list(labels.keys())
ess = [g for g, e in labels.items() if e == 1]
non = [g for g, e in labels.items() if e == 0]
N = len(all_genes); E = len(ess); NE = len(non)
print(f"E. coli BW25113: {N} genes, {E} essential, {NE} non-essential "
      f"(p_ess={E/N:.3f})")

# ---- (1) viability under random k-knockouts: AND gate ----
# P(cell viable | knock out k random genes) = P(none of them essential)
#   = C(NE, k) / C(N, k)
# For large N this is well-approximated by (1 - p_ess)^k, but we compute it
# exactly via cumulative log-ratios so the curve is faithful.
def p_viable(k):
    # exact hypergeometric: choosing k from N, none from ess
    if k > NE:
        return 0.0
    logp = 0.0
    for i in range(k):
        logp += np.log((NE - i) / (N - i))
    return float(np.exp(logp))

ks = np.arange(0, 50)
viab_random = [p_viable(int(k)) for k in ks]
# half-life of a cell under random knockouts
half = next((int(k) for k in ks if viab_random[k] <= 0.5), None)
# 95th-percentile death
p05 = next((int(k) for k in ks if viab_random[k] <= 0.05), None)

# ---- (2) whole-genome attack: random vs targeted (essentials first) ----
# Targeted = adversary knows essentiality and removes essentials first.
# Random = uniform over the genome.
rng = np.random.default_rng(0)
fracs = np.linspace(0, 0.30, 16)
# random: expected viable = (1-f)^E ish; compute by sampling
def random_viable_curve(trials=200):
    out = []
    for f in fracs:
        k = int(f * N)
        v = 0
        for _ in range(trials):
            idx = rng.choice(N, k, replace=False)
            # cell dies if ANY essential is in idx
            hits_ess = np.isin(idx, np.arange(N)[np.array([labels[g]==1 for g in all_genes])])
            if not hits_ess.any():
                v += 1
        out.append(v / trials)
    return out
ess_mask = np.array([labels[g] == 1 for g in all_genes])
def random_viable_fast(trials=400):
    out = []
    for f in fracs:
        k = int(f * N)
        v = 0
        for _ in range(trials):
            idx = rng.choice(N, k, replace=False)
            if not ess_mask[idx].any():
                v += 1
        out.append(v / trials)
    return out
viab_curve_rand = random_viable_fast()
# targeted: as long as at least one essential removed -> 0 viability;
# for the first essential the adversary picks ANY essential, so
# viability drops to 0 immediately. To make a more informative attack curve,
# show "% of essential subsystems lost" instead.
# Pre-target attack: remove top-degree essentials (from the network),
# the cell dies after the first one; show cumulative essential subsystems hit.

# Simpler & more honest curve: fraction of essentials lost vs k.
ess_lost_rand = []
for f in fracs:
    k = int(f * N)
    e_hit = 0
    for _ in range(200):
        idx = rng.choice(N, k, replace=False)
        e_hit += ess_mask[idx].sum() / E
    ess_lost_rand.append(e_hit / 200)
ess_lost_targ = [min(1.0, (f * N) / E) for f in fracs]

# ---- (3) the essential-network topology means dependency, not survival ----
# Compute average path length among essentials in the giant component, to show
# the network density of mechanistic links the cell relies on.
# (Reuse the network edges from lay_network.)
import re
og_of = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
n_orgs = len({o for s in og_present.values() for o in s})
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""), r.get("next_locus_tag", ""))
ess_set = set(ess); idx = {lt: i for i, lt in enumerate(ess)}; n = len(ess)
edges = set()
for lt in ess:
    for nb in synt.get(lt, ("", "")):
        if nb in ess_set and nb != lt:
            edges.add(tuple(sorted((idx[lt], idx[nb]))))
frac_pres = {}; pv = {}
for lt in ess:
    og = og_of.get((ORG, lt))
    if og:
        pv[lt] = og_present[og]; frac_pres[lt] = len(og_present[og]) / n_orgs
keys = [lt for lt in ess if lt in pv and 0.10 <= frac_pres[lt] < 0.90]
for a in range(len(keys)):
    va = pv[keys[a]]
    for b in range(a + 1, len(keys)):
        vb = pv[keys[b]]; inter = len(va & vb)
        if inter and inter / (len(va) + len(vb) - inter) >= 0.9:
            edges.add(tuple(sorted((idx[keys[a]], idx[keys[b]]))))
adj = defaultdict(set)
for a, b in edges:
    adj[a].add(b); adj[b].add(a)
deg = np.array([len(adj[i]) for i in range(n)])
# giant component
seen = set(); comps = []
for s in range(n):
    if s in seen: continue
    stack = [s]; seen.add(s); c = [s]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen: seen.add(v); stack.append(v); c.append(v)
    comps.append(c)
gc = max(comps, key=len)
# BFS average shortest path in giant
gc_set = set(gc); gc_list = list(gc)
sample = rng.choice(len(gc_list), min(60, len(gc_list)), replace=False)
spls = []
for s_idx in sample:
    s = gc_list[s_idx]
    dist = {s: 0}; q = [s]
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    spls.extend(d for d in dist.values() if d > 0)
avg_path = float(np.mean(spls))

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))

# (a) the AND-gate
ax[0].plot(ks, viab_random, color="#e31a1c", lw=2.4)
ax[0].fill_between(ks, viab_random, alpha=0.18, color="#e31a1c")
ax[0].axhline(0.5, color="#888", ls="--", lw=0.8)
ax[0].axvline(half, color="#888", ls="--", lw=0.8)
ax[0].text(half + 0.4, 0.53, f"50% dead at k = {half}", fontsize=9, color="#444")
ax[0].text(p05 + 0.4, 0.08, f"95% dead at k = {p05}", fontsize=9, color="#444")
ax[0].set_xlabel("# random gene knockouts")
ax[0].set_ylabel("P(cell viable)")
ax[0].set_title("The AND-gate of essentiality\n"
                f"viable iff none of the {E} essentials hit",
                fontsize=11, fontweight="bold")
ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3)

# (b) why the genome (not the essentials) is robust-yet-fragile
ax[1].plot(fracs * 100, ess_lost_rand, "o-", color="#3182bd", lw=2,
           label=f"random (over all {N} genes)")
ax[1].plot(fracs * 100, ess_lost_targ, "s-", color="#e31a1c", lw=2,
           label=f"targeted (essentials first)")
ax[1].set_xlabel("% of GENOME removed")
ax[1].set_ylabel("fraction of essentials lost")
ax[1].set_title("Robust-yet-fragile lives in the WHOLE genome\n"
                "(cell dies the moment this curve > 0)",
                fontsize=11, fontweight="bold")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
ax[1].set_ylim(0, 1.02)
ax[1].axhline(0.0001, color="#000", lw=0.7)
ax[1].text(15, 0.04, "any value above 0 = dead cell", fontsize=8, color="#444")

# (c) what the essential-network topology actually means
hd = ax[2].hist(deg, bins=30, color="#54278f", edgecolor="white")
ax[2].set_xlabel("dependency-degree (operon + phyletic partners)")
ax[2].set_ylabel("# essential genes")
ax[2].set_title("Essential-network topology = dependency density,\n"
                f"NOT survivability. Mean shortest path = {avg_path:.2f} hops",
                fontsize=11, fontweight="bold")
ax[2].grid(alpha=0.3)
ax[2].text(0.97, 0.93,
           "long tail = a few hub essentials\n"
           "wire many others; failing one\n"
           "cascades through tight chains.\n"
           "It still kills the cell --\n"
           "topology says HOW, not IF.",
           transform=ax[2].transAxes, ha="right", va="top",
           fontsize=8.5, color="#333",
           bbox=dict(boxstyle="round", fc="#f4eaff", ec="#a892c8"))

plt.tight_layout()
plt.savefig(OUT / "and_gate.png", dpi=130, bbox_inches="tight")
print("wrote outputs/orphan/and_gate.png")

json.dump(dict(
    organism=ORG,
    n_genes=N, n_essential=E, p_essential=E / N,
    viability_curve=dict(k=ks.tolist(), P_viable=[round(v, 4) for v in viab_random]),
    half_dead_at_k=half, p95_dead_at_k=p05,
    essentials_lost_random=[round(x, 3) for x in ess_lost_rand],
    essentials_lost_targeted=[round(x, 3) for x in ess_lost_targ],
    avg_shortest_path_essentials=avg_path,
    note=("Cell viability is an AND over essentials. Earlier 'graph-survival' "
          "Q3 was a category error -- the residual giant component is a corpse. "
          "Robust-yet-fragile lives at the WHOLE-GENOME level."),
), open(OUT / "and_gate.json", "w"), indent=2)
print("wrote outputs/orphan/and_gate.json")
print(f"\nhalf-dead at k={half} random knockouts; 95%-dead at k={p05}")
print(f"avg shortest path among essentials in giant component: {avg_path:.2f} hops")
