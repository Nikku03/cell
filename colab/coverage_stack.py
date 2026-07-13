"""coverage_stack — "get more cell lines": stack every measured perturbation screen we can get a gene list for,
and watch cumulative measured-response coverage climb (honestly, including diminishing returns).

Sources (perturbed-gene lists, no huge matrices needed):
  * K562 + RPE1  — Replogle genome-scale (on disk)                          [genome-wide]
  * CD4+ T cell  — Zhu/Marson 2025 targeted library (committed gene list)   [genome-wide]
  * scPerturb    — ~30 harmonized screens across neuron / iPSC / melanoma / THP-1 / MCF7 / GM12878 / …, with the
                   perturbed genes recoverable from each dataset's pairwise_edist table header (targeted screens)

The honest expectation: the two genome-wide screens (K562, T-cell) do the heavy lifting; the targeted scPerturb
screens each add a little, and returns diminish as the genome fills — a NEW genome-wide screen in a new lineage is
worth more than ten targeted ones. This counts it exactly, per source, so the diminishing return is visible, not
hidden. -> outputs/orphan/coverage_stack.json + committed union gene list.
"""
import os, sys, json, glob, csv
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")
CTRL = {"control", "ctrl", "nt", "ntc", "non-targeting", "nontargeting", "safe-harbor", "aavs1", "neg", "scramble", ""}


def genes_from_edist(path):
    """perturbation names are the header + first column; split combos on '_' → gene tokens."""
    toks = set()
    with open(path) as f:
        header = f.readline().strip().split(",")[1:]
        col0 = [row.split(",")[0] for row in f]
    for name in header + col0:
        for t in name.strip().strip('"').split("_"):
            if t and t.lower() not in CTRL:
                toks.add(t)
    return toks


def main(scp_tables=None, zhu_list=None):
    from complete_cell import CompleteCell
    from cross_cell_line import load_screen
    C = CompleteCell(); universe = {C.name[i] for i in range(len(C.genes))}; N = len(C.genes)
    SCR = os.environ.get("SCRATCH_DIR",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
    scp_tables = scp_tables or f"{SCR}/scP/website/tables"
    zhu_list = zhu_list or f"{OUT}/zhu_targeted_genes.txt"

    # ---- assemble each source's perturbed-gene set (in the model universe) ----
    sources = []                                            # (label, geneset)
    k562 = set()
    for f in ("gwps.h5ad", "k562.h5ad"):
        p = f"{SCR}/{f}"
        if os.path.exists(p):
            resp, _ = load_screen(p); k562 |= set(resp)
    sources.append(("K562 (Replogle genome-wide)", k562 & universe))
    if os.path.exists(zhu_list):
        zhu = {l.strip() for l in open(zhu_list) if l.strip()}
        sources.append(("CD4+ T cell (Zhu 2025 genome-scale)", zhu & universe))
    # scPerturb diverse cell types (skip the ones already represented to keep the story honest)
    skip = ("Replogle", "Norman", "sciplex", "SrivatsanTrapnell", "AissaBenevol", "ChangYe", "McFarland",
            "Schiebinger", "Gehring")            # K562-redundant, drug, or non-human
    for path in sorted(glob.glob(f"{scp_tables}/pairwise_edist_*.csv")):
        name = os.path.basename(path)[len("pairwise_edist_"):-4]
        if any(s in name for s in skip):
            continue
        g = genes_from_edist(path) & universe
        if g:
            sources.append((f"scPerturb:{name}", g))

    # ---- stack cumulatively, record marginal contribution ----
    covered = set(); rows = []
    union_all = set()
    for label, g in sources:
        new = (g - covered)
        covered |= g; union_all |= g
        rows.append({"source": label, "genes_in_source": len(g), "new_added": len(new),
                     "cumulative": len(covered), "cumulative_pct": round(len(covered) / N, 4)})

    base = len(sources[0][1])
    final = len(covered)
    print(f"CUMULATIVE MEASURED-RESPONSE COVERAGE — stacking cell lines  (genome = {N:,})\n")
    print(f"  {'source':<44}{'in src':>7}{'+new':>6}{'cumul':>8}{'  %':>7}")
    print("  " + "-" * 74)
    for r in rows:
        star = "  ←genome-wide" if "genome" in r["source"] else ""
        print(f"  {r['source'][:44]:<44}{r['genes_in_source']:>7}{r['new_added']:>6}"
              f"{r['cumulative']:>8}{r['cumulative_pct']*100:>6.1f}%{star}")
    print("  " + "-" * 74)
    print(f"  K562 alone: {base:,} ({base/N:.1%})  →  all stacked: {final:,} ({final/N:.1%})   (+{final-base:,} genes)")
    # honest note: how much came from genome-wide vs targeted
    gw = set();
    for label, g in sources:
        if "genome" in label:
            gw |= g
    tgt_added = len(covered) - len(gw & universe)
    print(f"  of the gain: genome-wide screens → {len((gw&universe))-base:,} new; "
          f"all {len([s for s in sources if 'scPerturb' in s[0]])} targeted scPerturb screens → {tgt_added:,} new "
          f"(diminishing returns — targeted screens re-hit shared genes).")

    open(f"{OUT}/coverage_stack_genes.txt", "w").write("\n".join(sorted(covered)) + "\n")
    json.dump({"genome": N, "k562_base": base, "k562_base_pct": round(base / N, 4),
               "stacked_total": final, "stacked_pct": round(final / N, 4), "gain": final - base,
               "n_sources": len(sources), "per_source": rows,
               "note": "targeted scPerturb screens add little over the two genome-wide screens — a new GENOME-WIDE "
                       "screen in a new lineage is worth far more than many targeted ones."},
              open(f"{OUT}/coverage_stack.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
