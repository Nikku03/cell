"""Replication geometry: where essential genes sit on the replisome's map.

If a gene is essential the cell must guarantee it gets copied -- on time, in the
right dosage, and without colliding with the replisome. We test five
predictions at once on GMI1000 chromosome 1 (and the 6-bacterium pool):

  P1 GENE-DOSE         essentials cluster near oriC (more copies during multifork
                       replication = higher transcript dosage for the same promoter)
  P2 REPLICHORE SYM    essentials are roughly mirror-symmetric across oriC<->terC
                       (both replichores need their core machinery)
  P3 CO-ORIENTATION    essentials avoid head-on encounters: leading-strand bias,
                       worse for LONG essentials (collision risk scales with length)
  P4 TERMINUS HAZARD   the terminus is special: convergent forks, dif/XerCD,
                       chromosome-dimer resolution; essential density should
                       AVOID the terminal third
  P5 MACHINERY ROSTER  the replication+division machinery itself should be
                       essential (dnaA/B/G/E, gyrA/B, topA, ssb, primase, helicase,
                       ftsZ/A/K/W/I, parA/B, minC/D/E, mukB, smc, dif)

Outputs:
  outputs/orphan/replication_geometry.png
  outputs/orphan/replication_geometry_machinery.csv
  outputs/orphan/replication_geometry_summary.json
"""
import re, json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")

BACT = {
    "beril_RalstoniaGMI1000":  "GCF_000009125.1",
    "beril_RalstoniaBSBF1503": "GCF_001587155.1",
    "beril_RalstoniaPSI07":    "GCF_000283475.1",
    "beril_Dda3937":           "GCF_000147055.1",
    "beril_HerbieS":           "GCF_000143225.1",
    "beril_Magneto":           "GCF_000009985.1",
}
ORG_FOCAL = "beril_RalstoniaGMI1000"

# replication + division machinery (product/gene-name regexes, case-insensitive)
MACHINERY = {
    "REPLICATION_initiation": r"\b(dnaA|chromosomal replication init|DnaA)\b",
    "REPLICATION_polymerase": r"DNA polymerase III|DnaE|DnaN|DnaQ|DnaX|HolA|HolB|HolC|HolD|HolE",
    "REPLICATION_helicase":   r"replicative DNA helicase|DnaB|DnaC",
    "REPLICATION_primase":    r"DNA primase|DnaG",
    "REPLICATION_ssb":        r"\bSSB\b|single-strand(?:ed)? DNA-binding",
    "REPLICATION_ligase":     r"DNA ligase",
    "REPLICATION_gyrase":     r"DNA gyrase|GyrA|GyrB|topoisomerase II",
    "REPLICATION_topoIV":     r"topoisomerase IV|ParC|ParE",
    "REPLICATION_topoI":      r"topoisomerase I\b|TopA",
    "TERMINUS_dif_xer":       r"XerC|XerD|tyrosine recombinase|chromosomal dimer",
    "TERMINUS_FtsK":          r"FtsK|DNA translocase",
    "TERMINUS_Tus":           r"replication terminator|Tus\b",
    "DIVISION_FtsZ":          r"\bFtsZ\b|cell division protein FtsZ",
    "DIVISION_FtsA":          r"\bFtsA\b",
    "DIVISION_FtsI_W":        r"FtsI|FtsW|penicillin-binding protein 3",
    "DIVISION_MinCDE":        r"\bMinC\b|\bMinD\b|\bMinE\b|septum site-determining",
    "DIVISION_MreB":          r"MreB|rod shape",
    "PARTITION_ParAB":        r"\bParA\b|\bParB\b|chromosome partitioning",
    "PARTITION_SMC_MukB":     r"\bSMC\b|\bMukB\b|MukE|MukF|condensin",
}

# labels
labels = defaultdict(dict)
for r in csv.DictReader(open(DATA / "labels" / "labels.csv")):
    labels[r["organism"]][r["locus_tag"]] = int(r["essential"])


def read_contigs(acc):
    d = {}; cur = None; buf = []
    for line in (DATA / "genome_cache" / acc / "genome.fna").read_text().splitlines():
        if line.startswith(">"):
            if cur: d[cur] = "".join(buf).upper()
            cur = line[1:].split()[0]; buf = []
        else: buf.append(line.strip())
    if cur: d[cur] = "".join(buf).upper()
    return d


def oriC_terC(seq, win=10000):
    n = len(seq); sk = []
    for i in range(0, n, win):
        w = seq[i:i+win]; g, c = w.count("G"), w.count("C")
        sk.append((g - c) / max(1, g + c))
    cum = np.cumsum(sk)
    return int(np.argmin(cum)) * win, int(np.argmax(cum)) * win


def parse_cds(acc):
    cds = defaultdict(list)
    for line in (DATA / "genome_cache" / acc / "genomic.gff").read_text().splitlines():
        if "\tCDS\t" not in line: continue
        c = line.split("\t")
        m = re.search(r"locus_tag=([^;\n]+)", c[8])
        if not m: continue
        prod = re.search(r"product=([^;\n]+)", c[8])
        g = re.search(r"gene=([^;\n]+)", c[8])
        cds[c[0]].append((int(c[3]), int(c[4]), c[6], m.group(1),
                          prod.group(1) if prod else "",
                          g.group(1) if g else ""))
    for k in cds: cds[k].sort()
    return cds


def per_gene_geom(org, acc):
    """Return list of dicts (one per CDS on the longest contig)."""
    contigs = read_contigs(acc)
    ctg = max(contigs, key=lambda c: len(contigs[c])); seq = contigs[ctg]; L = len(seq)
    oriC, terC = oriC_terC(seq)
    cds = parse_cds(acc)[ctg]
    rows = []
    for s, e, st, lt, prod, gene in cds:
        if lt not in labels[org]: continue
        mid = (s + e) // 2
        d = abs(mid - oriC) % L; d = min(d, L - d)
        ori_frac = d / (L / 2)
        delta = (mid - oriC) % L
        leading = int((st == "+") == (delta < L / 2))
        replichore = "R" if delta < L / 2 else "L"
        dter = abs(mid - terC) % L; dter = min(dter, L - dter)
        ter_frac = dter / (L / 2)
        # 1 at oriC, -1 at terC -- the dosage axis
        dose = 1 - 2 * ori_frac
        rows.append(dict(organism=org, contig=ctg, locus_tag=lt,
                         gene=gene, product=prod, essential=labels[org][lt],
                         start=s, end=e, strand=st, mid=mid, L=L,
                         oriC_pos=oriC, terC_pos=terC,
                         ori_frac=ori_frac, ter_frac=ter_frac,
                         dose=dose, leading=leading, replichore=replichore,
                         length_aa=(e - s + 1)//3))
    return rows


print("computing per-gene replication geometry for 6 bacteria ...")
all_rows = []
for org, acc in BACT.items():
    r = per_gene_geom(org, acc)
    e = sum(x["essential"] for x in r)
    print(f"  {org:<26} genes={len(r):<5} essential={e}")
    all_rows += r
df = pd.DataFrame(all_rows)
df_focal = df[df.organism == ORG_FOCAL].copy()

# ---- P1 dose effect: essential rate vs ori_frac (binned) ----
print("\nP1 GENE DOSE -- essential rate by ori_frac bin (pooled 6 bacteria):")
df["ori_bin"] = pd.cut(df.ori_frac, bins=np.linspace(0,1,11),
                       include_lowest=True, labels=False)
dose = df.groupby("ori_bin").agg(n=("essential","size"),
                                 ess=("essential","mean")).reset_index()
for _, r in dose.iterrows():
    print(f"  ori_frac {r['ori_bin']*0.1:.1f}-{(r['ori_bin']+1)*0.1:.1f}: "
          f"n={int(r['n']):<5} ess_rate={r['ess']:.3f}")

# ---- P2 replichore symmetry (focal) ----
sym = df_focal.groupby("replichore").agg(
    n=("essential","size"), ess=("essential","mean")).reset_index()
print(f"\nP2 REPLICHORE SYMMETRY ({ORG_FOCAL}):")
for _, r in sym.iterrows():
    print(f"  {r['replichore']}-arm n={int(r['n'])} ess_rate={r['ess']:.3f}")

# ---- P3 co-orientation x length (focal) ----
print(f"\nP3 CO-ORIENTATION x length ({ORG_FOCAL}):")
print(f"  {'class':<18} n   ess_rate    mean_len  head-on penalty?")
fc = df_focal.copy()
fc["lenq"] = pd.qcut(fc.length_aa, 4, labels=["Q1 short","Q2","Q3","Q4 long"])
for s in [1, 0]:
    for q in ["Q1 short","Q2","Q3","Q4 long"]:
        sub = fc[(fc.leading == s) & (fc.lenq == q)]
        if len(sub) < 30: continue
        tag = "leading" if s == 1 else "lagging"
        print(f"  {tag:<8} {q:<10} {len(sub):<5} {sub.essential.mean():.3f}      {sub.length_aa.mean():.0f}")

# ---- P4 terminus hazard ----
print(f"\nP4 TERMINUS HAZARD ({ORG_FOCAL}): essential rate in oriC-third vs middle vs ter-third")
df_focal["zone"] = np.where(df_focal.ori_frac < 1/3, "oriC-third",
                    np.where(df_focal.ori_frac > 2/3, "ter-third", "middle"))
zone = df_focal.groupby("zone").agg(n=("essential","size"),
                                    ess=("essential","mean")).reset_index()
for _, r in zone.iterrows():
    print(f"  {r['zone']:<12} n={int(r['n']):<5} ess_rate={r['ess']:.3f}")

# ---- P5 machinery roster (focal) ----
print(f"\nP5 MACHINERY ROSTER ({ORG_FOCAL}):")
print(f"  {'category':<26} {'n_found':>7} {'ess/n':>8} {'mean_ori_frac':>14}")
mach_rows = []
for cat, pat in MACHINERY.items():
    rx = re.compile(pat, re.I)
    hits = df_focal[df_focal.apply(lambda r: bool(rx.search(r["product"] or "") or
                                                  rx.search(r["gene"] or "")), axis=1)]
    if len(hits) == 0:
        print(f"  {cat:<26} {'0':>7}")
        continue
    e = int(hits.essential.sum())
    print(f"  {cat:<26} {len(hits):>7} {e}/{len(hits):>4}   {hits.ori_frac.mean():>14.2f}")
    for _, h in hits.iterrows():
        mach_rows.append(dict(category=cat, locus_tag=h["locus_tag"],
                              gene=h["gene"], product=h["product"],
                              essential=h["essential"], strand=h["strand"],
                              leading=h["leading"], ori_frac=round(h["ori_frac"], 3),
                              ter_frac=round(h["ter_frac"], 3),
                              length_aa=int(h["length_aa"])))
pd.DataFrame(mach_rows).to_csv(
    OUT / "replication_geometry_machinery.csv", index=False)

# ---- figure ----
fig, axes = plt.subplots(2, 3, figsize=(19, 11))

# A: circular chromosome map of essentials, focal organism
ax = axes[0, 0]
ax.remove()
ax = fig.add_subplot(2, 3, 1, projection="polar")
fc = df_focal
# polar coords: angle = (mid - oriC)/L * 2pi; r small=lagging, large=leading
L = int(fc.L.iloc[0]); oriC = int(fc.oriC_pos.iloc[0]); terC = int(fc.terC_pos.iloc[0])
theta = ((fc.mid - oriC) % L) / L * 2 * np.pi
r = np.where(fc.leading == 1, 1.0, 0.7)
nemask = fc.essential == 0
ax.scatter(theta[nemask], r[nemask], s=2, c="#BDC3C7", alpha=0.4)
ax.scatter(theta[~nemask], r[~nemask], s=4, c="#C0392B", alpha=0.7)
ax.set_yticklabels([]); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.set_rmax(1.15); ax.set_rmin(0.5)
# mark oriC at angle 0, terC at angle of (terC - oriC)%L / L * 2pi
ter_ang = ((terC - oriC) % L) / L * 2 * np.pi
ax.plot([0,0],[0.5,1.15],color="green",lw=1.2)
ax.plot([ter_ang,ter_ang],[0.5,1.15],color="red",lw=1.2)
ax.text(0, 1.2, "oriC", ha="center", color="green", fontsize=10)
ax.text(ter_ang, 1.2, "terC", ha="center", color="red", fontsize=10)
ax.set_title(f"A. Replichore map (focal) -- {ORG_FOCAL.replace('beril_','')}\n"
             f"outer=leading inner=lagging, red=essential", fontsize=10, pad=20)

# B: gene-dose curve (P1)
ax = axes[0, 1]
xs = (dose.ori_bin.astype(float) + 0.5) * 0.1
ax.plot(xs, dose.ess, "o-", color="#C0392B", lw=2)
ax.axhline(df.essential.mean(), color="gray", ls="--", lw=0.8,
           label=f"pool mean {df.essential.mean():.2f}")
ax.set_xlabel("normalized distance from oriC (0=origin, 1=terminus)")
ax.set_ylabel("essential rate")
ax.set_title("B. P1 GENE DOSE -- essentials cluster near oriC\n(pooled 6 bacteria)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# C: replichore symmetry (P2)
ax = axes[0, 2]
arms = sym.replichore.tolist(); vals = sym.ess.tolist(); ns = sym.n.tolist()
bars = ax.bar(arms, vals, color=["#3498DB","#E67E22"], alpha=0.85)
for i,(v,n) in enumerate(zip(vals,ns)):
    ax.text(i, v, f" n={n}\n {v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(vals)*1.4)
ax.set_ylabel("essential rate (focal)")
ax.set_title(f"C. P2 REPLICHORE SYMMETRY ({ORG_FOCAL.replace('beril_','')})")

# D: co-orientation x length (P3) -- focal
ax = axes[1, 0]
fcq = df_focal.copy()
fcq["lenq"] = pd.qcut(fcq.length_aa, 4, labels=["Q1 short","Q2","Q3","Q4 long"])
labelsLQ = ["Q1 short","Q2","Q3","Q4 long"]
lead = [fcq[(fcq.leading==1)&(fcq.lenq==q)].essential.mean() for q in labelsLQ]
lagg = [fcq[(fcq.leading==0)&(fcq.lenq==q)].essential.mean() for q in labelsLQ]
x = np.arange(4); w = 0.35
ax.bar(x-w/2, lead, w, color="#27AE60", label="leading (co-oriented)")
ax.bar(x+w/2, lagg, w, color="#C0392B", label="lagging (head-on)")
ax.set_xticks(x); ax.set_xticklabels(labelsLQ, fontsize=9)
ax.set_ylabel("essential rate"); ax.set_ylim(0, max(lead+lagg)*1.3)
ax.set_title("D. P3 CO-ORIENTATION x length\nhead-on penalty grows with length?")
ax.legend(fontsize=8)

# E: terminus hazard (P4)
ax = axes[1, 1]
zorder = ["oriC-third","middle","ter-third"]
vals = [zone[zone.zone==z].ess.iloc[0] for z in zorder]
ns = [int(zone[zone.zone==z].n.iloc[0]) for z in zorder]
ax.bar(zorder, vals, color=["#27AE60","#7F8C8D","#C0392B"], alpha=0.85)
for i,(v,n) in enumerate(zip(vals,ns)):
    ax.text(i, v, f" n={n}\n {v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(vals)*1.3)
ax.set_ylabel("essential rate")
ax.set_title("E. P4 TERMINUS HAZARD -- ter-third vs oriC-third (focal)")

# F: machinery roster heatmap
ax = axes[1, 2]
mr = pd.DataFrame(mach_rows)
if len(mr):
    cat_summary = mr.groupby("category").agg(
        n=("locus_tag","size"), ess=("essential","mean")).reset_index()
    cats = list(MACHINERY); ess_arr = []; n_arr = []
    for c in cats:
        sub = cat_summary[cat_summary.category == c]
        if len(sub):
            ess_arr.append(float(sub.ess.iloc[0])); n_arr.append(int(sub.n.iloc[0]))
        else:
            ess_arr.append(np.nan); n_arr.append(0)
    yp = np.arange(len(cats))[::-1]
    colors = ["#C0392B" if (e is not None and not np.isnan(e) and e >= 0.5)
              else "#7F8C8D" if (e is not None and not np.isnan(e) and e > 0)
              else "#ECF0F1" for e in ess_arr]
    ax.barh(yp, [e if (e is not None and not np.isnan(e)) else 0 for e in ess_arr],
            color=colors, alpha=0.85)
    for i, (e, n) in enumerate(zip(ess_arr, n_arr)):
        y = yp[i]
        if n == 0: txt = "  n/a"
        elif np.isnan(e): txt = f"  n={n}"
        else: txt = f"  {e:.2f}  (n={n})"
        ax.text(0, y, txt, va="center", fontsize=8)
    ax.set_yticks(yp); ax.set_yticklabels([c.replace("_"," ") for c in cats], fontsize=8)
    ax.set_xlim(0, 1.05); ax.axvline(df_focal.essential.mean(), color="gray", ls="--", lw=0.8)
    ax.set_xlabel("fraction essential (focal)")
    ax.set_title("F. P5 MACHINERY ROSTER -- are the copying tools themselves essential?")

fig.suptitle("Replication geometry: essentials and the copying / division machinery they ride with",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "replication_geometry.png", dpi=140, bbox_inches="tight")

(OUT / "replication_geometry_summary.json").write_text(json.dumps({
    "focal_organism": ORG_FOCAL,
    "n_bacteria_pool": len(BACT),
    "P1_dose_curve_essential_rate_by_ori_bin": [
        dict(ori_frac_bin=round(float(b)*0.1+0.05, 2), n=int(n), ess_rate=round(float(e),3))
        for b, n, e in zip(dose.ori_bin, dose.n, dose.ess)],
    "P2_replichore_symmetry_focal": [
        dict(arm=str(a), n=int(n), ess_rate=round(float(e),3))
        for a, n, e in zip(sym.replichore, sym.n, sym.ess)],
    "P4_zone_essential_rate_focal": [
        dict(zone=str(z), n=int(n), ess_rate=round(float(e),3))
        for z, n, e in zip(zone.zone, zone.n, zone.ess)],
    "P5_machinery_csv": "outputs/orphan/replication_geometry_machinery.csv",
    "P5_machinery_categories_found": int(pd.DataFrame(mach_rows).category.nunique()
                                          if mach_rows else 0),
    "P5_machinery_total_genes": int(len(mach_rows)),
}, indent=2))
print("\nwrote replication_geometry.png + machinery csv + summary")
