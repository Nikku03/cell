"""cell_levels — hand the software the new pathway-LEVEL machinery (tier-1 metabolic position + tier-2 signaling
tier) and let it label the whole cell, completing every gene it honestly can and abstaining where it can't.

Runs the two validated labelers at full scale and merges them into one per-gene table:
  - tier-1 METABOLIC (pathway_position.py): a step level for enzymes in ordered substrate chains (validated:
    glycolysis 0.99). Loaded from the committed pathway_position.json (validated pathways).
  - tier-2 SIGNALING (pathway_tier.py): an upstream→downstream tier for the ~10k membership genes via the SIGNOR
    directed graph + SCC feedback handling; feed-forward = trustworthy, feedback module = flagged, no-edge = abstain.

Output = cell_levels.json: for every gene, its level (0=upstream .. 1=downstream), a coarse bucket, a status
(metabolic-step / signaling-tier / feedback-module / no-level), and the pathway it came from. The point is to see
how much of the cell the software can now complete on its own — and to have it SAY where it can't.
-> outputs/orphan/cell_levels.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def _bucket(x):
    return "upstream" if x < 0.34 else "downstream" if x > 0.66 else "midstream"


def build():
    from complete_cell import CompleteCell
    from pathway_tier import Tierer
    C = CompleteCell()
    T = Tierer(C)
    name = lambda i: C.genes[i].get("name")

    labels = {}     # gene name -> record

    # ---- tier-1: metabolic step levels, SCALED across all KEGG metabolic maps (metabolic_levels.py) ----
    n_metab = n_metab_cyc = n_metab_ctx = 0
    ml_path = f"{OUT}/metabolic_levels.json"
    cellgenes = set(name(i) for i in range(len(C.genes)))     # real genes only — drop KEGG aliases (PK1, PKR, …)
    if os.path.exists(ml_path):
        mlab = json.load(open(ml_path)).get("labels", {})
        for g, r in mlab.items():
            if g not in cellgenes:
                continue
            st = r["status"]
            if st == "metabolic-step":
                labels[g] = {"status": "metabolic-step", "level": r["level"], "bucket": r["bucket"],
                             "pathway": r.get("map"), "detail": r["detail"], "source": "KEGG topology"}
                n_metab += 1
            elif st == "metabolic-cycle":
                labels[g] = {"status": "metabolic-cycle", "level": r["level"], "bucket": r["bucket"],
                             "pathway": r.get("map"), "detail": r["detail"], "source": "KEGG topology"}
                n_metab_cyc += 1
            else:                                          # context-dependent across maps
                labels[g] = {"status": "context-dependent", "level": None, "bucket": None,
                             "pathway": r.get("map"), "detail": r["detail"], "source": "KEGG topology"}
                n_metab_ctx += 1
    else:                                                  # fallback: the 2 validated pathways
        pp = json.load(open(f"{OUT}/pathway_position.json")).get("pathways", {})
        for pname, r in pp.items():
            for row in r.get("per_enzyme", []):
                if row.get("fused_pos") is None:
                    continue
                labels[row["gene"]] = {"status": "metabolic-step", "level": round(float(row["fused_pos"]), 2),
                                       "bucket": _bucket(row["fused_pos"]), "pathway": pname,
                                       "detail": f"step {row['true_step']}", "source": "KEGG topology + literature"}
                n_metab += 1

    # ---- tier-2: signaling tiers over ALL Reactome pathways (SIGNOR SCC) ----
    rp = json.load(open(f"{OUT}/reactome_pathways.json"))["pathways"]
    # per gene, collect feed-forward normalized levels + any feedback/no-context placement
    ff = {}          # name -> list of (norm_level, pathway, pathway_size)
    fb = {}          # name -> pathway (in a feedback module somewhere)
    ctx = set()      # names that appear in any >=3 pathway (candidate for a tier)
    for pname, gl in rp.items():
        ids = sorted(set(gl))
        if len(ids) < 3:
            continue
        for i in ids:
            ctx.add(name(i))
        tier, loop = T.tier(ids)
        if not tier:
            continue
        mx = max(tier.values()) or 1
        for i, t in tier.items():
            g = name(i)
            if loop[i]:
                fb.setdefault(g, pname)
            else:
                ff.setdefault(g, []).append((t / mx, pname, len(ids)))

    membership = set(name(i) for gl in rp.values() for i in set(gl))
    n_sig = n_fb = n_ctxdep = 0
    for g in membership:
        if g in labels:                                    # metabolic already labeled (higher priority)
            continue
        if g in ff and ff[g]:
            # a gene's level is PATHWAY-RELATIVE, so aggregate across ALL its pathways; a gene that sits early in
            # some and late in others has no single level → flag it context-dependent instead of faking a number
            lv = np.array([x[0] for x in ff[g]])
            mean, spread = float(lv.mean()), float(lv.std())
            pw = min(ff[g], key=lambda x: x[2])[1]
            if len(lv) >= 3 and spread > 0.3:
                labels[g] = {"status": "context-dependent", "level": None, "bucket": None, "pathway": pw,
                             "detail": f"level varies across {len(lv)} pathways (mean {mean:.2f} ± {spread:.2f}) — "
                                       f"no single level", "source": "SIGNOR directed graph"}
                n_ctxdep += 1
            else:
                labels[g] = {"status": "signaling-tier", "level": round(mean, 2), "bucket": _bucket(mean),
                             "pathway": pw, "detail": f"{_bucket(mean)} across {len(lv)} pathway(s) (±{spread:.2f})",
                             "source": "SIGNOR directed graph"}
                n_sig += 1
        elif g in fb:
            labels[g] = {"status": "feedback-module", "level": None, "bucket": None, "pathway": fb[g],
                         "detail": "in a feedback loop — no linear level", "source": "SIGNOR SCC"}
            n_fb += 1

    # ---- abstain: membership genes with no tier, + note the rest of the genome ----
    n_abstain = 0
    for g in membership:
        if g not in labels:
            labels[g] = {"status": "no-level", "level": None, "bucket": None, "pathway": None,
                         "detail": "pathway member but no orderable context (abstain)", "source": None}
            n_abstain += 1

    trustworthy = n_metab + n_metab_cyc + n_sig
    census = {"total_genome": len(C.genes), "pathway_membership_genes": len(membership),
              "total_genes_labeled": len(labels),
              "metabolic_step_labeled": n_metab, "metabolic_cycle_labeled": n_metab_cyc,
              "signaling_tier_labeled": n_sig,
              "context_dependent_flagged": n_ctxdep + n_metab_ctx, "feedback_module_flagged": n_fb,
              "no_level_abstain": n_abstain, "trustworthy_level_total": trustworthy}
    return labels, census


def main():
    labels, census = build()
    print("=" * 88)
    print("CELL LEVELS — the software labels every gene's pathway level it honestly can")
    print("=" * 88)
    print(f"  genome                         {census['total_genome']:,}   labeled {census['total_genes_labeled']:,}")
    print(f"    → metabolic STEP level        {census['metabolic_step_labeled']:,}  (tier-1, KEGG ordered chains)")
    print(f"    → metabolic CYCLE (conventional) {census['metabolic_cycle_labeled']:,}  (tier-1, cyclic maps — flagged)")
    print(f"    → signaling upstream/downstream tier {census['signaling_tier_labeled']:,}  (tier-2, feed-forward)")
    print(f"    → context-dependent (flagged) {census['context_dependent_flagged']:,}  (level differs by pathway/map — honest)")
    print(f"    → feedback module (flagged)   {census['feedback_module_flagged']:,}  (no linear level — honest)")
    print(f"    → no orderable context (abstain) {census['no_level_abstain']:,}")
    print(f"  TRUSTWORTHY level for {census['trustworthy_level_total']:,} genes "
          f"(metabolic step+cycle {census['metabolic_step_labeled']+census['metabolic_cycle_labeled']:,} + "
          f"signaling tier {census['signaling_tier_labeled']:,}); the rest FLAGGED or abstained.")
    # a few example labels
    print("\n  examples:")
    for g in ("FASN", "PFKL", "CS", "RELA", "CASP3", "MAPK1"):
        if g in labels:
            r = labels[g]
            print(f"    {g:7} {r['status']:16} level={r['level']}  {r['detail']}  [{r.get('pathway')}]")
    json.dump({"labels": labels, "census": census,
               "note": "per-gene pathway LEVEL completed by the software: metabolic step (tier-1) + signaling tier "
                       "(tier-2, SIGNOR SCC); feedback modules flagged, unorderable genes abstained"},
              open(f"{OUT}/cell_levels.json", "w"), indent=1)
    print(f"\n  → wrote {len(labels):,} gene level labels to cell_levels.json")
    return census


if __name__ == "__main__":
    main()
