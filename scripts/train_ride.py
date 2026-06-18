"""The genome train ride.

Ride each chromosome 5' -> 3' like a train. Every CDS is a platform. As we
roll between platforms, a conductor with no idea which genes are essential
takes notes whenever something feels off:

   DEJA_VU         seen this family before in this genome (paralog of a past stop)
   WRONG_SIDE      strand opposite the running majority (last 20 genes)
   EMPTY_PLATFORM  long upstream gap (>300 bp) -- operon boundary / solitary
   EXPRESS         tiny upstream gap (<10 bp), same strand -- deep operon interior
   FOREIGN_SIGN    GC differs from 50-kb local window by >0.05 -- HGT island feel
   TINY            very short gene (<100 aa)
   GRAND_CENTRAL   very long gene (>700 aa)
   HEADWIND        head-on with replication (lagging strand)
   SCRAPYARD       within 5 kb of a transposase / IS / phage
   LONE_STAR       far from any mobile element (>50 kb)
   NEW_LINE        first or last gene on a contig (replicon boundary)
   HOMETOWN        within 10% of oriC
   REMOTE          within 10% of terminus
   DEEP_OPERON     both neighbors same strand within 50 bp (operon interior)
   GC_NORMAL       gene GC within 0.01 of local window
   SAME_STRAND_RUN >=10 consecutive same-strand genes in a row including this one

Then at the end, the labels arrive, and for each note we ask:
   among genes flagged with this note, what is the essential rate vs the base?

Outputs:
  outputs/orphan/train_ride.png
  outputs/orphan/train_ride_log.txt          first 40 stations on org 1, prose log
  outputs/orphan/train_ride_summary.json
"""
import re, json, csv
from pathlib import Path
from collections import defaultdict, Counter, deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")

ORGS = {
    "beril_RalstoniaGMI1000":  "GCF_000009125.1",
    "beril_RalstoniaBSBF1503": "GCF_001587155.1",
    "beril_RalstoniaPSI07":    "GCF_000283475.1",
    "beril_Dda3937":           "GCF_000147055.1",
    "beril_HerbieS":           "GCF_000143225.1",
    "beril_Magneto":           "GCF_000009985.1",
    "mtub":                    "GCF_000195955.2",
    "saur_NCTC8325_biotradis": "GCF_000013425.1",
    "beril_Methanococcus_S2":  "GCF_000011585.1",
    "beril_Methanococcus_JJ":  "GCF_002945325.1",
}
MOB = re.compile(r"transposase|integrase|insertion sequence|IS\d|phage|"
                 r"recombinase|mobile element", re.I)

labels = defaultdict(dict)
for r in csv.DictReader(open(DATA / "labels" / "labels.csv")):
    labels[r["organism"]][r["locus_tag"]] = int(r["essential"])
orth = defaultdict(dict)
for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
    orth[r["organism"]][r["locus_tag"]] = r.get("og_id", "")


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
    n = len(seq); skew = []
    for i in range(0, n, win):
        w = seq[i:i+win]; g, c = w.count("G"), w.count("C")
        skew.append((g - c) / max(1, g + c))
    cum = np.cumsum(skew)
    return int(np.argmin(cum)) * win, int(np.argmax(cum)) * win


def parse_cds(acc):
    """return dict[contig] = list of (start, end, strand, locus, product) sorted."""
    cds = defaultdict(list)
    for line in (DATA / "genome_cache" / acc / "genomic.gff").read_text().splitlines():
        if "\tCDS\t" not in line: continue
        c = line.split("\t")
        m = re.search(r"locus_tag=([^;\n]+)", c[8])
        if not m: continue
        prod = re.search(r"product=([^;\n]+)", c[8])
        cds[c[0]].append((int(c[3]), int(c[4]), c[6], m.group(1),
                          prod.group(1) if prod else ""))
    for k in cds: cds[k].sort()
    return cds


def gc_of(seq):
    if not seq: return 0
    return (seq.count("G") + seq.count("C")) / len(seq)


NOTES = ["DEJA_VU","WRONG_SIDE","EMPTY_PLATFORM","EXPRESS","FOREIGN_SIGN",
         "TINY","GRAND_CENTRAL","HEADWIND","SCRAPYARD","LONE_STAR",
         "NEW_LINE","HOMETOWN","REMOTE","DEEP_OPERON","GC_NORMAL",
         "SAME_STRAND_RUN"]


def ride_organism(org, acc, prose_log=None):
    """Return list of dicts (one per CDS) with binary note columns."""
    contigs = read_contigs(acc)
    cds_all = parse_cds(acc)
    rows = []
    for contig, lst in cds_all.items():
        seq = contigs.get(contig)
        if not seq or len(seq) < 20000: continue
        L = len(seq)
        oriC, terC = oriC_terC(seq)
        mobs = sorted({(a+b)//2 for a,b,st,lt,p in lst if MOB.search(p or "")})
        # rolling state
        recent_strand = deque(maxlen=20)
        seen_families = set()
        run_strand = None; run_len = 0
        for i, (s, e, strand, lt, prod) in enumerate(lst):
            mid = (s + e) // 2
            length_aa = (e - s + 1) // 3
            gene_gc = gc_of(seq[s-1:e])
            # local 50-kb window GC excluding this gene
            ws, we_ = max(0, mid - 25000), min(L, mid + 25000)
            local_gc = gc_of(seq[ws:s-1] + seq[e:we_])
            # gaps
            up_gap = s - lst[i-1][1] if i > 0 else None
            dn_gap = lst[i+1][0] - e if i < len(lst)-1 else None
            up_strand_same = (i > 0 and lst[i-1][2] == strand)
            dn_strand_same = (i < len(lst)-1 and lst[i+1][2] == strand)
            # replication leading/lagging
            delta = (mid - oriC) % L
            leading = (strand == "+") == (delta < L / 2)
            # oriC fractional distance
            dori = abs(mid - oriC) % L; dori = min(dori, L - dori)
            ori_frac = dori / (L / 2)
            dter = abs(mid - terC) % L; dter = min(dter, L - dter)
            ter_frac = dter / (L / 2)
            # mobile distance
            if mobs:
                md = min(min(abs(mid-mp), L-abs(mid-mp)) for mp in mobs) / 1e3
            else:
                md = 1000.0
            # family / paralog déjà vu
            fam = orth[org].get(lt, "")
            deja = bool(fam) and fam in seen_families
            if fam: seen_families.add(fam)
            # strand run
            if strand == run_strand: run_len += 1
            else: run_strand, run_len = strand, 1
            # majority strand of last 20
            recent_strand.append(strand)
            maj = Counter(recent_strand).most_common(1)[0][0]
            wrong_side = (strand != maj and len(recent_strand) >= 8)

            notes = dict(
                DEJA_VU         = int(deja),
                WRONG_SIDE      = int(wrong_side),
                EMPTY_PLATFORM  = int(up_gap is not None and up_gap > 300),
                EXPRESS         = int(up_gap is not None and -20 < up_gap < 10 and up_strand_same),
                FOREIGN_SIGN    = int(abs(gene_gc - local_gc) > 0.05),
                TINY            = int(length_aa < 100),
                GRAND_CENTRAL   = int(length_aa > 700),
                HEADWIND        = int(not leading),
                SCRAPYARD       = int(md < 5),
                LONE_STAR       = int(md > 50),
                NEW_LINE        = int(i == 0 or i == len(lst)-1),
                HOMETOWN        = int(ori_frac < 0.10),
                REMOTE          = int(ter_frac < 0.10),
                DEEP_OPERON     = int(up_strand_same and dn_strand_same and
                                      (up_gap is not None and up_gap < 50) and
                                      (dn_gap is not None and dn_gap < 50)),
                GC_NORMAL       = int(abs(gene_gc - local_gc) < 0.01),
                SAME_STRAND_RUN = int(run_len >= 10),
            )
            ess = labels[org].get(lt)
            rows.append(dict(organism=org, locus_tag=lt, contig=contig,
                             start=s, end=e, strand=strand,
                             length_aa=length_aa, product=prod,
                             essential=ess, **notes))
            # prose log for first 40 stations of the first contig
            if prose_log is not None and len(prose_log) < 40 and contig == list(cds_all)[0]:
                hits = [k for k,v in notes.items() if v]
                tag = ", ".join(hits) if hits else "—"
                prose_log.append(f"  station #{i+1:3d} {lt:<14} ({length_aa:>4} aa, {strand}) "
                                 f"{prod[:40]:<40}  notes: {tag}")
    return rows


print("riding 10 organisms ...")
all_rows = []
prose = []
for i, (org, acc) in enumerate(ORGS.items()):
    log = prose if i == 0 else None   # only log first org
    r = ride_organism(org, acc, prose_log=log)
    e = sum(1 for x in r if x["essential"] == 1)
    n = sum(1 for x in r if x["essential"] is not None)
    print(f"  {org:<26} stations={len(r):<5} labeled={n} essential={e}")
    all_rows += r
df = pd.DataFrame(all_rows)
print(f"\ntotal: {len(df):,} stations across {df.organism.nunique()} organisms")
df.to_parquet(OUT / "train_ride_stations.parquet", index=False)

# Write the prose log
(OUT / "train_ride_log.txt").write_text(
    "GENOME TRAIN RIDE -- conductor's notebook\n"
    f"first 40 stations of {list(ORGS)[0]}:\n\n" + "\n".join(prose) + "\n"
)

# ----- compare notes to essentiality -----
df_lab = df[df.essential.notna()].copy()
df_lab["essential"] = df_lab["essential"].astype(int)
base = df_lab["essential"].mean()
print(f"\nBASE essential rate (10 orgs pooled): {base:.3f}")
print(f"\n{'note':<18}  flagged   ess-rate  enrich   org-consistent?")
print(f"  (number of orgs where ess-rate > org base)")
print("-"*78)

per_note = []
for note in NOTES:
    sub = df_lab[df_lab[note] == 1]
    if len(sub) < 50: continue
    r = sub.essential.mean()
    enr = r / base
    consistent = sum(
        1 for o in ORGS
        if ((df_lab.organism == o) & (df_lab[note] == 1)).sum() >= 20
        and df_lab[(df_lab.organism == o) & (df_lab[note] == 1)].essential.mean()
            > df_lab[df_lab.organism == o].essential.mean()
    )
    per_note.append((note, len(sub), r, enr, consistent))
    print(f"  {note:<16}  {len(sub):>6}   {r:.3f}     {enr:+.2f}×    {consistent}/10")

per_note.sort(key=lambda x: -x[3])
print("\nranked by enrichment (essential rate / base):")
for n, k, r, e, c in per_note:
    bar = "█" * int(max(0, (e - 1) * 30))
    star = "  ★" if c >= 7 else ""
    print(f"  {e:+.2f}× {n:<16} (n={k:5d}, p_ess={r:.2f}) {bar}{star}")

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
ax = axes[0]
labels_n = [x[0] for x in per_note]
enrichs  = [x[3] for x in per_note]
flagged  = [x[1] for x in per_note]
consist  = [x[4] for x in per_note]
colors = ["#27AE60" if e>=1.15 else "#E74C3C" if e<=0.85 else "#7F8C8D" for e in enrichs]
y = np.arange(len(per_note))
ax.barh(y, [e-1 for e in enrichs], color=colors, alpha=0.85)
for i,(e,k,c) in enumerate(zip(enrichs, flagged, consist)):
    star = " ★" if c>=7 else ""
    ax.text(e-1 + (0.02 if e>=1 else -0.02), i, f" {e:.2f}×  (n={k}, {c}/10){star}",
            va="center", ha="left" if e>=1 else "right", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labels_n, fontsize=9)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("essential enrichment over base (essential_rate / base - 1)")
ax.set_title(f"Conductor's notes vs essentiality (10 organisms, base={base:.2f})\n"
             "green: enriched for essentials  | red: depleted  | ★ consistent in ≥7/10 orgs",
             fontsize=11)
ax.grid(alpha=0.3, axis="x")

# Panel B: one-organism "train timetable" along the chromosome -- show GMI1000
ax = axes[1]
g = df[df.organism == list(ORGS)[0]].copy()
g = g[g.contig == g.contig.value_counts().idxmax()].reset_index(drop=True)
mid = (g.start + g.end) / 2 / 1e6
ess = g.essential.fillna(-1).to_numpy()
col_ess = np.where(ess == 1, "#C0392B", np.where(ess == 0, "#BDC3C7", "#7F8C8D"))
# scatter
ax.scatter(mid, np.full(len(g), 0.5), c=col_ess, s=4, alpha=0.6, label="essentials (red)")
# overlay top-3 enriching notes
for note, color, yoff in [("DEEP_OPERON","#2980B9", 0.7),
                          ("DEJA_VU","#E67E22", 0.85),
                          ("LONE_STAR","#27AE60", 1.0)]:
    mask = g[note] == 1
    ax.scatter(mid[mask], np.full(mask.sum(), yoff), c=color, s=6, alpha=0.7, label=note)
ax.set_xlabel("position along chromosome (Mb)")
ax.set_yticks([0.5, 0.7, 0.85, 1.0])
ax.set_yticklabels(["essentials","DEEP_OPERON","DEJA_VU","LONE_STAR"])
ax.set_ylim(0.3, 1.15)
ax.set_title(f"B. Train timetable -- {list(ORGS)[0].replace('beril_','')} chromosome 1")
ax.grid(alpha=0.3, axis="x")
ax.legend(fontsize=8, loc="upper right")

fig.suptitle("The genome train ride: do anomaly notes (taken blind) predict the essential platforms?",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT / "train_ride.png", dpi=140, bbox_inches="tight")

# Summary
(OUT / "train_ride_summary.json").write_text(json.dumps({
    "n_organisms": len(ORGS),
    "n_stations": int(len(df)),
    "n_labeled": int(len(df_lab)),
    "base_essential_rate": round(float(base), 3),
    "notes_ranked_by_enrichment": [
        {"note": n, "n_flagged": int(k), "essential_rate": round(float(r), 3),
         "enrichment_over_base": round(float(e), 3),
         "orgs_consistent": int(c)}
        for n,k,r,e,c in per_note
    ],
}, indent=2))

print(f"\nwrote train_ride.png + train_ride_log.txt + train_ride_summary.json")
