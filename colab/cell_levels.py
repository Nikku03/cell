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

    # ---- tier-1: metabolic step levels from the validated pathway_position.json ----
    pp = json.load(open(f"{OUT}/pathway_position.json")).get("pathways", {})
    n_metab = 0
    for pname, r in pp.items():
        rows = r.get("per_enzyme", [])
        n = len(rows)
        for row in rows:
            fp = row.get("fused_pos")
            if fp is None:
                continue
            g = row["gene"]
            labels[g] = {"status": "metabolic-step", "level": round(float(fp), 2), "bucket": _bucket(fp),
                         "pathway": pname, "detail": f"step {row['true_step']}/{n}", "source": "KEGG topology + literature"}
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

    census = {"total_genome": len(C.genes), "pathway_membership_genes": len(membership),
              "metabolic_step_labeled": n_metab, "signaling_tier_labeled": n_sig,
              "context_dependent_flagged": n_ctxdep, "feedback_module_flagged": n_fb, "no_level_abstain": n_abstain,
              "completed_frac_of_membership": round((n_metab + n_sig) / len(membership), 3)}
    return labels, census


def main():
    labels, census = build()
    print("=" * 88)
    print("CELL LEVELS — the software labels every gene's pathway level it honestly can")
    print("=" * 88)
    print(f"  genome                         {census['total_genome']:,}")
    print(f"  pathway-membership genes       {census['pathway_membership_genes']:,}")
    print(f"    → metabolic step level        {census['metabolic_step_labeled']:,}  (tier-1, ordered chains)")
    print(f"    → signaling upstream/downstream tier {census['signaling_tier_labeled']:,}  (tier-2, feed-forward)")
    print(f"    → context-dependent (flagged) {census['context_dependent_flagged']:,}  (level differs by pathway — honest)")
    print(f"    → feedback module (flagged)   {census['feedback_module_flagged']:,}  (no linear level — honest)")
    print(f"    → no orderable context (abstain) {census['no_level_abstain']:,}")
    print(f"  COMPLETED a trustworthy level for {census['completed_frac_of_membership']:.0%} of membership genes; "
          f"the rest it FLAGS or abstains on.")
    # a few example labels
    print("\n  examples:")
    for g in ("PKM", "TCF7", "RELA", "CASP3", "MAPK1"):
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
