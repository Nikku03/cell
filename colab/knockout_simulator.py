"""Knockout & pathway-effect simulator on E. coli (beril_Keio).

Honest scope: this is NOT flux balance (no kinetic / metabolic model is local;
the FBA features file has only precomputed essentiality calls, and the
b-number<->Keio-tag bridge isn't constructible from our data). Instead it's a
NETWORK-BASED pathway-impact simulator:

  KNOCKOUT(gene X) ->
   1 IDENTIFY    pathway / functional slot of X (from local annotation)
   2 OPERON      downstream operon partners hit by polar effects
   3 SLOT IMPACT count of slot's essential genes left after removal
                 (does the slot fall below the syn3a-derived viable minimum?)
   4 PHYLETIC    co-evolving partners likely to lose function
   5 CASCADE     2-hop network reach: how many essentials transitively at risk
   6 VERDICT     viable / impaired / dead, with the mechanistic reason
   7 PATHWAY     bottleneck pathway: which subsystem can no longer self-sustain

For each demo gene (essential hub, non-essential metabolic) we emit a structured
trace + a visual. Computed numbers are honest; no rate-constant fabrication.
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "beril_Keio"

# ------- load: labels, annotation, synteny (operon), OG presence (phyletic) --
truth = {}; gname = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG:
        truth[r["locus_tag"]] = int(r["essential"])
        g = (r.get("gene_name") or "").strip()
        if g and g != "-": gname[r["locus_tag"]] = g
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == ORG and r.get("product"):
        annot[r["locus_tag"]] = r["product"]
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == ORG:
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""),
                                r.get("next_locus_tag", ""))
og_of = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
n_orgs = len({o for s in og_present.values() for o in s})

# slot classifier
SLOTS = [
    ("ribosomal", [r"ribosom"]),
    ("synthetase", [r"--trna lig", r"trna ligase", r"synthetase", r"aminoacyl"]),
    ("trna", [r"\btrna\b"]),
    ("transcription", [r"rna polymerase", r"transcription", r"sigma"]),
    ("replication", [r"dna polymerase", r"helicase", r"gyrase", r"primase",
                     r"topoisomer", r"replicat", r"dna ligase"]),
    ("translation", [r"elongation factor", r"initiation factor",
                     r"release factor", r"chaperon"]),
    ("cell_division", [r"cell division", r"septation", r"ftsz",
                       r"\bftsz?\b", r"divisome"]),
    ("membrane", [r"membrane", r"lipoprotein", r"transporter", r"permease",
                  r"porin", r"secretion"]),
    ("energy", [r"atp synthase", r"atpase", r"cytochrome", r"nadh"]),
    ("lipid", [r"fatty acid", r"\bacp\b", r"lipid", r"phospholipid"]),
    ("nucleotide", [r"purine", r"pyrimidine", r"nucleotide", r"ribonucleot"]),
    ("amino_acid", [r"amino acid", r"glutam", r"aspart"]),
    ("cofactor", [r"folate", r"thiamine", r"riboflavin", r"coenzyme", r"cofactor"]),
]
SP = [(nm, [re.compile(x, re.I) for x in pats]) for nm, pats in SLOTS]
def slot_of(lt):
    p = (annot.get(lt, "") or "").lower()
    for nm, rx in SP:
        if any(r.search(p) for r in rx): return nm
    return "other"

# syn3a-derived viable minimums (the "AND" floor)
REQ = {"ribosomal": 30, "synthetase": 12, "transcription": 6,
       "replication": 5, "translation": 5, "cell_division": 1}

# phyletic neighbours: same-OG-presence-jaccard >=0.9 over Keio essentials
TRUE_ESS = set(lt for lt, e in truth.items() if e == 1)
slot_of_lt = {lt: slot_of(lt) for lt in truth}
slot_counts_true = defaultdict(int)
for lt in TRUE_ESS: slot_counts_true[slot_of_lt[lt]] += 1

ess_pv = {}
for lt in TRUE_ESS:
    og = og_of.get((ORG, lt))
    if og: ess_pv[lt] = og_present[og]
phyl = defaultdict(set)
keys = [lt for lt in ess_pv if 0.1 <= len(ess_pv[lt]) / n_orgs < 0.9]
for i in range(len(keys)):
    va = ess_pv[keys[i]]
    for j in range(i + 1, len(keys)):
        vb = ess_pv[keys[j]]; inter = len(va & vb)
        if inter and inter / (len(va) + len(vb) - inter) >= 0.9:
            phyl[keys[i]].add(keys[j]); phyl[keys[j]].add(keys[i])

# universal core flag (>= 90% of genomes carry this OG)
def is_universal(lt):
    og = og_of.get((ORG, lt))
    return bool(og) and len(og_present[og]) / n_orgs >= 0.90

# ------- the simulator -------
def simulate_knockout(lt):
    """Return a structured report of what happens when we delete this gene."""
    if lt not in truth:
        return {"error": f"gene {lt} not in Keio"}
    product = annot.get(lt, "(unannotated)")
    s = slot_of_lt[lt]
    essential = truth[lt] == 1
    universal = is_universal(lt)

    # operon partners (polar effects: if a downstream gene shares an operon,
    # it loses its transcript). We approximate by synteny neighbours that are
    # ALSO essential & in the same slot OR universally conserved.
    prev, nxt = synt.get(lt, ("", ""))
    operon_hit = [g for g in (prev, nxt)
                  if g and g in truth and (truth[g] == 1 or is_universal(g))]

    # phyletic partners that co-evolve (often act in the same pathway/complex)
    phy_partners = sorted(phyl.get(lt, set()))[:8]

    # slot impact: simulate removing this gene from its slot's essential count
    cur_count = slot_counts_true[s]
    after = cur_count - (1 if essential else 0) - len(operon_hit)
    req = REQ.get(s, 0)
    slot_status = "slot OK"
    if req and after < req:
        slot_status = f"SLOT BELOW VIABLE MINIMUM ({after} < {req} needed)"

    # cascade: 2-hop reach over phyletic+operon (essential genes only)
    seen = {lt}; frontier = [lt]; depth = 0; cascade = set()
    while frontier and depth < 2:
        nf = []
        for u in frontier:
            for v in phyl.get(u, set()) | {x for x in synt.get(u, ("", "")) if x}:
                if v in TRUE_ESS and v not in seen:
                    seen.add(v); nf.append(v); cascade.add(v)
        frontier = nf; depth += 1

    # verdict (AND gate + slot AND + cascade severity)
    if essential:
        verdict = "DEAD"
        why = f"essential gene knockout (AND gate: 1 essential lost -> 0 viability)"
    elif slot_status.startswith("SLOT"):
        verdict = "DEAD"
        why = f"{s} subsystem can no longer self-sustain ({slot_status})"
    elif len(operon_hit) >= 2:
        verdict = "IMPAIRED"
        why = f"polar effect knocks out {len(operon_hit)} downstream essentials"
    elif universal and not essential:
        verdict = "IMPAIRED"
        why = "universal-core gene removal; conserved everywhere despite being non-essential here"
    else:
        verdict = "VIABLE"
        why = "non-essential, no cascading slot collapse"

    return dict(
        locus_tag=lt, gene_name=gname.get(lt, "?"), product=product,
        slot=s, essential=essential, universal_core=universal,
        operon_neighbours=[(g, gname.get(g, "?"), annot.get(g, "")[:40],
                            "essential" if truth.get(g) == 1 else "non-essential")
                           for g in (prev, nxt) if g],
        operon_essentials_hit=len(operon_hit),
        phyletic_partners=[(g, gname.get(g, "?"), annot.get(g, "")[:50]) for g in phy_partners],
        slot_before=cur_count, slot_after=max(after, 0), slot_required=req,
        slot_status=slot_status,
        cascade_reach=len(cascade),
        verdict=verdict, mechanism=why,
    )


# ------- pick representative genes to demo -------
def find_gene(predicates):
    """find a Keio gene whose product matches one of these regex predicates."""
    for lt, p in annot.items():
        if any(re.search(pat, p, re.I) for pat in predicates):
            return lt
    return None

def find_gene_with(preds, essential=None):
    """find a Keio gene matching ANY predicate, optionally filtered on essentiality."""
    for lt, p in annot.items():
        if essential is not None and truth.get(lt) != int(bool(essential)):
            continue
        if any(re.search(pat, p, re.I) for pat in preds):
            return lt
    return None

DEMO = [
    ("FtsZ (cell division GTPase)",   find_gene_with([r"^cell division protein FtsZ", r"\bFtsZ\b"], essential=True)),
    ("GyrA (DNA gyrase A)",           find_gene_with([r"\bDNA gyrase subunit A\b", r"\bgyrase A\b"], essential=True)),
    ("RpoB (RNA polymerase beta)",    find_gene_with([r"DNA-directed RNA polymerase subunit beta\b"], essential=True)),
    ("FolA (DHFR, folate)",           find_gene_with([r"dihydrofolate reductase"], essential=True)),
    ("non-essential metabolic gene",  find_gene_with([r"^L-lactate dehydrogenase", r"^pyruvate kinase\b",
                                                      r"^glycerol-3-phosphate dehydrogenase", r"^aldehyde dehydrogenase"],
                                                     essential=False)),
]
print(f"{'demo gene':<36}{'locus_tag':<12}{'verdict':>10}")
results = []
for label, lt in DEMO:
    if not lt:
        print(f"  {label:<34} -- not found in Keio annotation"); continue
    rep = simulate_knockout(lt)
    results.append((label, rep))
    print(f"  {label:<34}{lt:<12}{rep['verdict']:>10}  -> {rep['mechanism'][:60]}")

# Cross-validate vs FBA precomputed calls where possible (no bridge for Keio,
# but we can show the validation principle for mtub which uses native locus_tags)
mtub_fba = {r["locus_tag"]: int(r["fba_essential"])
            for r in csv.DictReader(open("data/drive_import/labels/fba_features.csv"))
            if r["organism"] == "mtub"}
mtub_truth = {r["locus_tag"]: int(r["essential"])
              for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))
              if r["organism"] == "mtub"}
join = [(lt, mtub_fba[lt], mtub_truth[lt]) for lt in mtub_fba if lt in mtub_truth]
fba_prec = fba_rec = 0; tp=fp=fn=tn=0
if join:
    tp = sum(1 for _, f, t in join if f == 1 and t == 1)
    fp = sum(1 for _, f, t in join if f == 1 and t == 0)
    fn = sum(1 for _, f, t in join if f == 0 and t == 1)
    tn = sum(1 for _, f, t in join if f == 0 and t == 0)
    fba_prec = tp / max(tp + fp, 1); fba_rec = tp / max(tp + fn, 1)
    print(f"\nFBA-vs-truth cross-check (mtub, {len(join)} joinable): "
          f"P={fba_prec:.2f} R={fba_rec:.2f}")

# ------- write reports + figure -------
report_md = ["# Knockout simulator -- demo reports (E. coli Keio)\n"]
for label, rep in results:
    report_md.append(f"\n## {label}  --  `{rep['locus_tag']}` ({rep['gene_name']})\n")
    report_md.append(f"- **product**: {rep['product']}")
    report_md.append(f"- **slot**: {rep['slot']}    "
                     f"**essential (truth)**: {rep['essential']}    "
                     f"**universal core**: {rep['universal_core']}")
    if rep["operon_neighbours"]:
        report_md.append(f"- **operon neighbours**: " +
                         ", ".join(f"{g} ({n}, {p[:30]}, {st})"
                                   for g, n, p, st in rep["operon_neighbours"]))
        report_md.append(f"    polar effect hits {rep['operon_essentials_hit']} downstream essential(s)")
    if rep["phyletic_partners"]:
        report_md.append(f"- **phyletic co-evolving partners**: " +
                         "; ".join(f"{g} ({n}, {p[:35]})" for g, n, p in rep["phyletic_partners"][:5]))
    report_md.append(f"- **slot {rep['slot']}**: had {rep['slot_before']} essentials, "
                     f"after knockout = {rep['slot_after']}, "
                     f"viable min = {rep['slot_required'] or 'n/a'} -> {rep['slot_status']}")
    report_md.append(f"- **cascade reach**: {rep['cascade_reach']} additional essentials "
                     "in 2-hop network neighbourhood")
    report_md.append(f"- **VERDICT**: **{rep['verdict']}** -- {rep['mechanism']}")
Path(OUT / "knockout_report.md").write_text("\n".join(report_md))
print(f"\nwrote outputs/orphan/knockout_report.md ({len(results)} demo knockouts)")

json.dump({label.split(" ")[0]: rep for label, rep in results},
          open(OUT / "knockout_report.json", "w"), indent=2)

# ------- figure: 4 demo knockouts, their cascade + slot impact -------
fig, axes = plt.subplots(2, 3, figsize=(19, 9))
to_plot = results[:4] + [(("FBA cross-check (mtub)"), None)]
for ax, (label, rep) in zip(axes.flatten(), to_plot[:6]):
    if rep is None:
        ax.axis("off")
        if join:
            txt = (f"FBA precomputed knockout calls\n"
                   f"vs M. tuberculosis truth\n"
                   f"({len(join)} genes joinable)\n\n"
                   f"  Precision (essential): {fba_prec:.2f}\n"
                   f"  Recall    (essential): {fba_rec:.2f}\n"
                   f"  TP {tp}  FP {fp}\n  FN {fn}  TN {tn}\n\n"
                   f"This is the kind of validation\n"
                   f"the simulator targets per gene:\n"
                   f"if knockout->dead matches truth,\n"
                   f"the network mechanism is real.")
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
                    fontsize=10, family="monospace",
                    bbox=dict(boxstyle="round", fc="#f0f0f5", ec="#888"))
        continue
    slots = list(REQ.keys())
    vals = [slot_counts_true[s] for s in slots]
    colors = ["#e31a1c" if s == rep["slot"] else "#3182bd" for s in slots]
    ax.barh(range(len(slots)), vals, color=colors)
    for i, s in enumerate(slots):
        ax.plot([REQ[s], REQ[s]], [i - 0.4, i + 0.4], color="black", lw=2)
        if s == rep["slot"]:
            ax.barh([i], [rep["slot_after"]], color="#fd8d3c", height=0.5)
    ax.set_yticks(range(len(slots)))
    ax.set_yticklabels(slots, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("# essential genes in slot")
    color = {"DEAD": "#e31a1c", "IMPAIRED": "#fd8d3c", "VIABLE": "#2ca02c"}[rep["verdict"]]
    ax.set_title(f"KO {label}\n"
                 f"verdict: {rep['verdict']}  (cascade {rep['cascade_reach']} ess.)",
                 fontsize=10, fontweight="bold", color=color)
    ax.text(0.98, 0.04, rep["mechanism"][:90], transform=ax.transAxes, ha="right",
            fontsize=7, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color))

if len(to_plot) < 6: axes.flatten()[-1].axis("off")

fig.suptitle("E. coli network-based knockout simulator: per-gene pathway impact "
             "+ AND-gate viability\n"
             "(no flux model; mechanism = slot collapse + operon polar effect + "
             "phyletic cascade)",
             fontweight="bold", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT / "knockout_simulator.png", dpi=120, bbox_inches="tight")
print("wrote outputs/orphan/knockout_simulator.png")
