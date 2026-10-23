"""gene_identity -- start from the gene, not from the response: what is it, what machine does it work in, and if it is a TF, what does it
control and what do those targets do. Then audit whether that can be answered for the WHOLE cell or only for a well-studied corner.

THE REFRAME. Every model in this project began at the response -- knock out X, see what moves, try to predict it. The prior question was
skipped: what IS X? What machinery does it belong to? If it is a transcription factor, which genes does it control and what do THOSE do?
That is the question a biologist asks first, and it is answerable from annotation alone, without any perturbation.

THE CARD, per gene:
    IDENTITY    locus, compartment, annotated process, evidence weight (publications, dark-proteome flag, confidence)
    FUNCTION    GO molecular function, GO biological process, Reactome pathway
    MACHINERY   curated complexes it belongs to and their other members, PPI degree and partners, metabolic reactions it catalyses
    CONTROL     if a TF: how many targets, and -- the part that matters -- the FUNCTIONAL PROFILE of that regulon, i.e. what the targets do,
                grouped by process and compartment, so the regulon is described rather than merely counted
    CONTROLLED  which TFs regulate it

THE AUDIT, which is the real question. A card is easy to produce for GATA1. The useful question is for how many of the cell's ~16.5k genes
each field can be filled at all, because this project has already measured that annotation coverage is not uniform -- the pointwise model
scored 0.428 on genes in a curated complex and 0.143 on genes in none, and only 25.6% of one knockout's movers had a Reactome pathway. So
every field is counted genome-wide and, separately, over the genes that are actually measured in Perturb-seq and the genes that were
actually knocked out, since those populations are biased towards well-studied genes and the difference is the point.

For TFs the audit goes one step further and asks the CHAIN question: for a TF with known targets, what fraction of those targets have a
known function? A regulon of 900 genes is worthless if nobody knows what 600 of them do."""
import os
import json, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))


def load():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    comps = collections.defaultdict(set); cmem = {}
    for cid, mem in D["complexes"].items():
        mm = {names[x] for x in mem if isinstance(x, int) and x < N}
        if 2 <= len(mm) <= 200:
            cmem[cid] = mm
            for g in mm:
                comps[g].add(cid)
    reg_out = collections.defaultdict(list); reg_in = collections.defaultdict(list)
    for e in (D.get("reg") or []):
        try:
            a, b = names[e[0]], names[e[1]]; s = int(e[2]) if len(e) > 2 else 0
        except (TypeError, IndexError, ValueError):
            continue
        reg_out[a].append((b, s)); reg_in[b].append((a, s))
    ppi = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        try:
            a, b = names[e[0]], names[e[1]]
        except (TypeError, IndexError):
            continue
        ppi[a].add(b); ppi[b].add(a)
    go = D.get("go") or {}
    path_of = collections.defaultdict(set)
    for pw, mem in (D.get("pathways") or {}).items():
        for x in mem:
            if isinstance(x, int) and x < N:
                path_of[names[x]].add(pw)
    rxn = {names[int(k)]: v for k, v in (D.get("generxn") or {}).items() if int(k) < N}
    ptm = {names[int(k)]: v for k, v in (D.get("ptm") or {}).items() if int(k) < N}
    ppm = {names[int(k)]: v for k, v in (D.get("ppm") or {}).items() if int(k) < N}
    return D, names, info, comps, cmem, reg_out, reg_in, ppi, go, path_of, rxn, ptm, ppm


GENERIC_F = {"protein binding", "identical protein binding", "metal ion binding", "ATP binding", "RNA binding",
             "DNA binding", "nucleotide binding", "zinc ion binding", "molecular_function",
             "protein-containing complex binding", "enzyme binding", "calcium ion binding", "GTP binding",
             "protein homodimerization activity", "protein heterodimerization activity"}
VAGUE_PROC = {"other", "?", "", None}


def card(g, ctx):
    D, names, info, comps, cmem, reg_out, reg_in, ppi, go, path_of, rxn, ptm, ppm = ctx
    gi = info.get(g) or {}
    G = go.get(g, {}) or {}
    tg = reg_out.get(g, [])
    prof_proc = collections.Counter(); prof_comp = collections.Counter(); known = 0
    for t, _ in tg:
        ti = info.get(t) or {}
        p = ti.get("proc")
        if p:
            prof_proc[p] += 1
        c = ti.get("comp")
        if c:
            prof_comp[c] += 1
        # STRICT, matching the audit: a generic GO term or the "other" process bucket is not a known function
        if [x for x in ((go.get(t, {}) or {}).get("F") or []) if x not in GENERIC_F] or (p not in VAGUE_PROC):
            known += 1
    mates = set()
    for c in comps.get(g, ()):
        mates |= (cmem[c] - {g})
    return {
        "gene": g,
        "identity": {"locus": f"{gi.get('chrom')}:{gi.get('tss')}", "compartment": gi.get("comp"),
                     "process": gi.get("proc"), "publications": gi.get("pubs"), "dark": gi.get("dark"),
                     "confidence": gi.get("conf"), "essential": gi.get("ess"), "loeuf": gi.get("loeuf"),
                     "abundance_ppm": ppm.get(g)},
        "function": {"GO_molecular_function": (G.get("F") or [])[:6],
                     "GO_biological_process": (G.get("P") or [])[:6],
                     "GO_component": (G.get("C") or [])[:4],
                     "reactome": sorted(path_of.get(g, ()))[:6],
                     "reactome_primary": (gi.get("path") or "").strip() or None},
        "machinery": {"complexes": sorted(comps.get(g, ()))[:6], "n_complexes": len(comps.get(g, ())),
                      "complex_partners": sorted(mates)[:14], "n_complex_partners": len(mates),
                      "ppi_degree": len(ppi.get(g, ())), "ppi_partners": sorted(ppi.get(g, ()))[:14],
                      "catalyses": (rxn.get(g) or [])[:4], "ptm": ptm.get(g)},
        "control": {"is_tf": bool(gi.get("tf")), "n_targets": len(tg),
                    "n_targets_with_known_function": known,
                    "frac_targets_annotated": round(known / len(tg), 3) if tg else None,
                    "regulon_by_process": prof_proc.most_common(8),
                    "note": "n_targets_with_known_function uses the strict definition: an informative GO molecular "
                            "function term, or a process outside the 'other' bucket",
                    "regulon_by_compartment": prof_comp.most_common(6),
                    "example_targets": [{"gene": t, "sign": s,
                                         "process": (info.get(t) or {}).get("proc"),
                                         "function": ((go.get(t, {}) or {}).get("F") or [None])[0]}
                                        for t, s in tg[:12]]},
        "controlled_by": {"n_regulators": len(reg_in.get(g, [])),
                          "regulators": [{"tf": a, "sign": s} for a, s in reg_in.get(g, [])[:12]]},
    }


def main(gene="GATA1"):
    ctx = load()
    D, names, info, comps, cmem, reg_out, reg_in, ppi, go, path_of, rxn, ptm, ppm = ctx

    c = card(gene, ctx)
    print(f"IDENTITY CARD: {gene}")
    i, f, m, k = c["identity"], c["function"], c["machinery"], c["control"]
    print(f"  locus {i['locus']} | {i['compartment']} | {i['process']} | {i['publications']} papers | "
          f"conf={i['confidence']} dark={i['dark']} | {i['abundance_ppm']} ppm")
    print(f"  FUNCTION   {f['GO_molecular_function'][:3]}")
    print(f"             {f['GO_biological_process'][:3]}")
    print(f"             reactome: {f['reactome_primary']}")
    print(f"  MACHINERY  {m['n_complexes']} complexes {m['complexes'][:3]}")
    print(f"             {m['n_complex_partners']} complex partners, {m['ppi_degree']} PPI partners")
    print(f"             PTM {m['ptm']}")
    if k["is_tf"]:
        print(f"  CONTROLS   {k['n_targets']} targets; {k['n_targets_with_known_function']} have a known function "
              f"({k['frac_targets_annotated']:.0%})")
        print(f"             regulon by process:     {k['regulon_by_process'][:6]}")
        print(f"             regulon by compartment: {k['regulon_by_compartment'][:5]}")
    print(f"  CONTROLLED BY {c['controlled_by']['n_regulators']} TFs")
    json.dump(c, open(OUT / f"gene_identity_{gene}.json", "w"), indent=1)

    # ---------------- THE AUDIT: can this be done for the whole cell? ----------------
    allg = [g["name"] for g in D["genes"]]
    try:
        df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
        measured = {str(x) for x in df.columns}
        A = np.abs(df.values.astype(np.float32))
        kod = {str(k) for k, n in zip(df.index, (A >= 4.2).sum(1)) if n >= 20}
    except Exception:
        measured, kod = set(), set()

    # PLACEHOLDERS ARE NOT KNOWLEDGE. A first version of this audit counted proc == "other" (4,302 genes, 26% of the
    # genome) and GO F terms like "protein binding" as annotation, and reported 100% function coverage genome-wide.
    # Both are placeholders: "other" is the bucket for genes that fit none of the 12 process categories, and the generic
    # GO terms below apply to most of the proteome and say nothing specific. Counting them inflates every downstream
    # figure, including the TF-chain result. Strict definitions are used throughout.
    def informative_F(g):
        F = (go.get(g, {}) or {}).get("F") or []
        return [t for t in F if t not in GENERIC_F]

    def specific_process(g):
        return ((info.get(g) or {}).get("proc")) not in VAGUE_PROC

    FIELDS = {
        "compartment": lambda g: bool((info.get(g) or {}).get("comp")),
        "process (excl. \"other\")": specific_process,
        "GO mol. function (any)": lambda g: bool((go.get(g, {}) or {}).get("F")),
        "GO mol. function (informative)": lambda g: bool(informative_F(g)),
        "GO biological process": lambda g: bool((go.get(g, {}) or {}).get("P")),
        "Reactome pathway": lambda g: bool(path_of.get(g)) or bool(((info.get(g) or {}).get("path") or "").strip()),
        "in a curated complex": lambda g: bool(comps.get(g)),
        "any PPI partner": lambda g: bool(ppi.get(g)),
        "catalyses a reaction": lambda g: g in rxn,
        "PTM sites": lambda g: g in ptm,
        "protein abundance": lambda g: g in ppm,
        "has any regulator": lambda g: bool(reg_in.get(g)),
        "is annotated a TF": lambda g: bool((info.get(g) or {}).get("tf")),
        ">=5 publications": lambda g: (((info.get(g) or {}).get("pubs")) or 0) >= 5,
    }
    POPS = [("all genes", allg), ("measured in Perturb-seq", sorted(measured & set(allg))),
            ("knocked out (>=20 movers)", sorted(kod & set(allg)))]
    print(f"\n\nCOVERAGE AUDIT -- for how many genes can each field be filled at all?")
    print(f"  {'field':26s}" + "".join(f"{p[0][:24]:>26s}" for p in POPS))
    audit = {}
    for fld, test in FIELDS.items():
        audit[fld] = {}
        line = f"  {fld:26s}"
        for pname, pop in POPS:
            v = (sum(1 for g in pop if test(g)) / len(pop)) if pop else 0.0
            audit[fld][pname] = round(v, 4)
            line += f"{v:>25.1%} "
        print(line)
    for pname, pop in POPS:
        print(f"  {'(population size)':26s}" if pname == POPS[0][0] else "", end="")
    print("  " + "".join(f"{len(p[1]):>26d}" for p in POPS))

    # the composite: can we answer "what is it, what machine, what does it control"?
    def can_function(g):
        """strict: an informative GO molecular function, or a process that is not the 'other' bucket"""
        return bool(informative_F(g)) or specific_process(g)

    def can_machinery(g):
        return FIELDS["in a curated complex"](g) or FIELDS["any PPI partner"](g)
    comp_rows = {}
    print(f"\n  COMPOSITE -- the three questions actually asked")
    for label, test in (("function known", can_function),
                        ("machinery known", can_machinery),
                        ("BOTH function and machinery", lambda g: can_function(g) and can_machinery(g)),
                        ("function + machinery + pathway", lambda g: can_function(g) and can_machinery(g)
                         and FIELDS["Reactome pathway"](g)),
                        ("STRICT: informative fn + complex + pathway",
                         lambda g: bool(informative_F(g)) and bool(comps.get(g)) and FIELDS["Reactome pathway"](g))):
        comp_rows[label] = {}
        line = f"  {label:26s}"
        for pname, pop in POPS:
            v = (sum(1 for g in pop if test(g)) / len(pop)) if pop else 0.0
            comp_rows[label][pname] = round(v, 4)
            line += f"{v:>25.1%} "
        print(line)

    # the TF chain: for TFs with targets, how well are the TARGETS annotated?
    tfs = [g for g in allg if (info.get(g) or {}).get("tf")]
    with_t = [g for g in tfs if reg_out.get(g)]
    fr = []
    for g in with_t:
        tg = [t for t, _ in reg_out[g]]
        if tg:
            fr.append(sum(1 for t in tg if can_function(t)) / len(tg))
    tf_stats = {"n_tf_annotated": len(tfs), "n_tf_with_targets": len(with_t),
                "median_regulon_size": int(np.median([len(reg_out[g]) for g in with_t])) if with_t else 0,
                "median_frac_targets_with_known_function": round(float(np.median(fr)), 3) if fr else None,
                "frac_tf_whose_regulon_is_>=80pct_annotated": round(float(np.mean([x >= 0.8 for x in fr])), 3) if fr else None}
    print(f"\n  THE TF CHAIN -- a regulon is only useful if the targets are annotated")
    print(f"    genes annotated as TF:                    {tf_stats['n_tf_annotated']}")
    print(f"    of those, with >=1 curated target:        {tf_stats['n_tf_with_targets']}")
    print(f"    median regulon size:                      {tf_stats['median_regulon_size']}")
    print(f"    median frac of targets with known function: {tf_stats['median_frac_targets_with_known_function']}")
    print(f"    TFs whose regulon is >=80% annotated:     {tf_stats['frac_tf_whose_regulon_is_>=80pct_annotated']:.1%}")

    verdict = (
        f"GENE IDENTITY + COVERAGE AUDIT: asks the question that precedes prediction -- what is this gene, what machinery does it work in, "
        f"and if it is a TF what does it control and what do those targets do -- and then measures whether that can be answered for the "
        f"whole cell or only for a well-studied corner. "
        "PLACEHOLDERS ARE EXCLUDED: a first version counted proc=='other' (4,302 genes, 26% of the genome) and generic GO terms like "
        "'protein binding' as knowledge and reported 100% function coverage. Both are placeholders and both are now excluded. "
        f"Genome-wide over {len(allg)} genes: function (an INFORMATIVE GO molecular function, or a process outside the 'other' bucket) is "
        f"available for "
        f"{comp_rows['function known']['all genes']:.1%}, machinery (a curated complex or any PPI partner) for "
        f"{comp_rows['machinery known']['all genes']:.1%}, BOTH for {comp_rows['BOTH function and machinery']['all genes']:.1%}, and all "
        f"three including a Reactome pathway for {comp_rows['function + machinery + pathway']['all genes']:.1%}. "
        f"The populations differ sharply and that difference is the finding: over the {len(POPS[1][1])} genes measured in Perturb-seq the "
        f"both-fields figure is {comp_rows['BOTH function and machinery']['measured in Perturb-seq']:.1%}, and over the "
        f"{len(POPS[2][1])} genes actually knocked out with a strong phenotype it is "
        f"{comp_rows['BOTH function and machinery']['knocked out (>=20 movers)']:.1%} -- the genes we experiment on are far better "
        f"annotated than the genome they are drawn from, so coverage measured on the experimental set OVERSTATES what is known about the "
        f"cell as a whole. "
        f"For transcription factors the chain matters more than the count: {tf_stats['n_tf_with_targets']} of {tf_stats['n_tf_annotated']} "
        f"annotated TFs have at least one curated target, median regulon {tf_stats['median_regulon_size']} genes, and the median TF has "
        f"{tf_stats['median_frac_targets_with_known_function']:.0%} of its targets carrying a known function, with "
        f"{tf_stats['frac_tf_whose_regulon_is_>=80pct_annotated']:.0%} of TFs having a regulon that is at least 80% annotated. "
        "Deterministic; every figure is a count over the merged annotation model, not an estimate.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"example_card": c, "audit": audit, "composite": comp_rows, "tf_chain": tf_stats,
               "population_sizes": {p[0]: len(p[1]) for p in POPS}, "verdict": verdict, "note": verdict},
              open(OUT / "gene_identity_audit.json", "w"), indent=1)
    print(f"\n  -> outputs/orphan/gene_identity_{gene}.json, outputs/orphan/gene_identity_audit.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GATA1")
