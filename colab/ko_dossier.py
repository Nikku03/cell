"""ko_dossier -- for ONE knockout, label everything: what moved, whether it is a direct TF target, and each affected protein's full journey.

WHAT THIS IS. Not a predictor and not an evaluation. It is an assembly: take a knockout that was actually performed in K562, take the genes
that actually moved, and attach every layer of annotation this project has to each one -- following a protein from its gene through
transcription, translation, modification, transport, interaction and reaction to its functional end state. The point is to see the measured
effect and the mechanistic annotation side by side for a real perturbation, at full detail rather than in aggregate.

THE JOURNEY, per affected protein, using only layers present in cell_complete.json:
    gene            chromosome, TSS, CpG island, enhancer count, 3D contact loops
    transcription   which TFs regulate it (signed), and whether the knocked-out gene is among them -- this is the DIRECT vs INDIRECT call
    translation     protein abundance (ppm)
    modification    PTM types and counts
    transport       assigned subcellular compartment
    interaction     PPI partners, curated complex memberships, signalling edges, ligand-receptor role
    reaction        metabolic reactions the protein catalyses
    end state       annotated process, GO terms (function / process / component), Reactome pathways

HONESTY ABOUT COVERAGE. Every layer is sparse in a different way, so the dossier reports, for each layer, how many of the affected genes
actually carry that annotation. A field that is blank for most movers is stated as such rather than quietly omitted -- the gaps are part of
the picture, and this project has already shown (0.428 vs 0.143 by complex membership) that annotation coverage is not uniform.

DIRECT vs INDIRECT is the one inferential claim here and it is a claim about the DATABASE, not about the cell: "direct" means a curated
TF->target edge exists, which the ChIP-vs-Perturb-seq work in this repo showed is a weak guide to actual regulation (0.7% of bound genes
move). It is labelled as database evidence, not as mechanism."""
import os
import json, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
T = 4.2
TIDE_FRAC = 0.05
DETAIL_N = 40          # movers given the full journey treatment in the readable output; all of them go into the JSON


def main(ko="GATA1"):
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]; gset = {g: i for i, g in enumerate(genes)}
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    if ko not in kidx:
        print(f"{ko} was not knocked out in this dataset"); return
    Z = df.values.astype(np.float32); A = np.abs(Z)
    tide = (A >= T).mean(0) >= TIDE_FRAC
    tide_genes = {genes[i] for i in np.where(tide)[0]}

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    nidx = {n: i for i, n in enumerate(names)}
    info = {g["name"]: g for g in D["genes"]}

    def nm(i):
        return names[i] if isinstance(i, int) and 0 <= i < N else None

    # ---- regulatory edges: TF -> target, signed ----
    reg_out = collections.defaultdict(list); reg_in = collections.defaultdict(list)
    for e in (D.get("reg") or []):
        try:
            a, b = nm(e[0]), nm(e[1]); s = int(e[2]) if len(e) > 2 else 0
        except (TypeError, ValueError, IndexError):
            continue
        if a and b:
            reg_out[a].append((b, s)); reg_in[b].append((a, s))
    ppi = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        a, b = nm(e[0]) if len(e) > 0 else None, nm(e[1]) if len(e) > 1 else None
        if a and b:
            ppi[a].add(b); ppi[b].add(a)
    sig = collections.defaultdict(list)
    for e in (D.get("sig") or []):
        a, b = nm(e[0]) if len(e) > 0 else None, nm(e[1]) if len(e) > 1 else None
        if a and b:
            sig[a].append((b, int(e[2]) if len(e) > 2 else 0))
    lr_pairs = set()
    for e in (D.get("lr") or []):
        a, b = nm(e[0]) if len(e) > 0 else None, nm(e[1]) if len(e) > 1 else None
        if a and b:
            lr_pairs.add((a, b))
    cplx_of = collections.defaultdict(set); cplx_mem = {}
    for cid, mem in D["complexes"].items():
        mm = {nm(x) for x in mem if isinstance(x, int) and x < N} - {None}
        if 2 <= len(mm) <= 200:
            cplx_mem[cid] = mm
            for g in mm:
                cplx_of[g].add(cid)
    path_of = collections.defaultdict(set)
    for pw, mem in (D.get("pathways") or {}).items():
        for x in mem:
            g = nm(x)
            if g:
                path_of[g].add(pw)
    GO = D.get("go") or {}
    PTM = {nm(int(k)): v for k, v in (D.get("ptm") or {}).items() if nm(int(k))}
    PPM = {nm(int(k)): v for k, v in (D.get("ppm") or {}).items() if nm(int(k))}
    RXN = {nm(int(k)): v for k, v in (D.get("generxn") or {}).items() if nm(int(k))}
    LOOP = {nm(int(k)): v for k, v in (D.get("loops3d") or {}).items() if nm(int(k))}
    CC = {nm(int(k)): v for k, v in (D.get("cellcycle") or {}).items() if nm(int(k))}
    enz = collections.defaultdict(list)
    for r in (D.get("reactions") or []):
        if r.get("enz"):
            enz[r["enz"]].append(r)

    # ---------------- PART 1: the knocked-out gene ----------------
    gi = info.get(ko, {}) or {}
    row = Z[kidx[ko]]
    moved = [(genes[i], float(row[i])) for i in np.where(A[kidx[ko]] >= T)[0]
             if genes[i] != ko and genes[i] not in tide_genes]
    moved.sort(key=lambda x: -abs(x[1]))
    mset = {m for m, _ in moved}
    tgt_direct = {b for b, _ in reg_out.get(ko, [])}

    def journey(g):
        gj = info.get(g, {}) or {}
        regs = reg_in.get(g, [])
        go = GO.get(g, {}) or {}
        return {
            "gene": {"chrom": gj.get("chrom"), "tss": gj.get("tss"), "cpg_island": gj.get("cpg"),
                     "enhancers": gj.get("enh"), "loops3d": LOOP.get(g), "loeuf": gj.get("loeuf"),
                     "essential": gj.get("ess"), "dep_frac": gj.get("dep_frac")},
            "transcription": {"n_regulators": len(regs),
                              "regulated_by_the_knockout": ko in {a for a, _ in regs} or g in tgt_direct,
                              "regulators": [{"tf": a, "sign": s} for a, s in regs[:12]]},
            "translation": {"abundance_ppm": PPM.get(g)},
            "modification": PTM.get(g),
            "transport": {"compartment": gj.get("comp"), "cell_cycle_phase": CC.get(g)},
            "interaction": {"n_ppi": len(ppi.get(g, ())), "ppi_partners": sorted(ppi.get(g, ()))[:12],
                            "complexes": sorted(cplx_of.get(g, ()))[:8],
                            "signals_to": [b for b, _ in sig.get(g, [])][:8],
                            "ligand_receptor": [f"{a}->{b}" for (a, b) in lr_pairs if a == g or b == g][:4]},
            "reaction": {"metabolic": (RXN.get(g) or [])[:4],
                         "enzyme_of": [f"{r['sub']} -> {r['prod']} ({r['pathway']})" for r in enz.get(g, [])][:4]},
            "end_state": {"process": gj.get("proc"), "pathways": sorted(path_of.get(g, ()))[:8],
                          "reactome_primary": (gj.get("path") or "").strip() or None,
                          "GO_function": (go.get("F") or [])[:5], "GO_process": (go.get("P") or [])[:5],
                          "GO_component": (go.get("C") or [])[:5]},
        }

    dossier = {
        "knockout": ko,
        "cell_line": "K562",
        "threshold": T,
        "self": {"identity": {k: gi.get(k) for k in ("chrom", "tss", "comp", "proc", "tf", "ess", "dep_frac",
                                                     "loeuf", "cpg", "enh", "pubs", "dark", "conf", "npath", "master")},
                 "journey": journey(ko),
                 "n_curated_targets": len(tgt_direct)},
        "effect": {"n_moved_specific": len(moved),
                   "n_up": sum(1 for _, z in moved if z > 0), "n_down": sum(1 for _, z in moved if z < 0),
                   "n_direct_curated_targets_among_moved": len(mset & tgt_direct),
                   "curated_targets_total": len(tgt_direct),
                   "frac_of_curated_targets_that_moved": round(len(mset & tgt_direct) / max(len(tgt_direct), 1), 4)},
        "moved": [],
    }
    for g, z in moved:
        dossier["moved"].append({"gene": g, "z": round(z, 2), "direction": "up" if z > 0 else "down",
                                 "direct_curated_target": g in tgt_direct, "journey": journey(g)})

    # ---------------- PART 4: interconnection among the affected set ----------------
    pc = collections.Counter()
    for g in mset:
        for p in path_of.get(g, ()):
            pc[p] += 1
    comp_c = collections.Counter((info.get(g, {}) or {}).get("comp") or "?" for g in mset)
    proc_c = collections.Counter((info.get(g, {}) or {}).get("proc") or "?" for g in mset)
    intra = [(a, b) for a in mset for b in ppi.get(a, ()) if b in mset and a < b]
    shared_cplx = collections.Counter()
    for cid, mm in cplx_mem.items():
        k = len(mm & mset)
        if k >= 2:
            shared_cplx[cid] = k
    # second-order: affected genes that are themselves TFs regulating other affected genes
    cascade = []
    for g in mset:
        if (info.get(g, {}) or {}).get("tf"):
            downstream = {b for b, _ in reg_out.get(g, [])} & mset
            if downstream:
                cascade.append({"tf": g, "n_downstream_also_moved": len(downstream),
                                "examples": sorted(downstream)[:8]})
    cascade.sort(key=lambda d: -d["n_downstream_also_moved"])
    cov = {lay: round(sum(1 for g in mset if test(g)) / max(len(mset), 1), 3) for lay, test in (
        ("compartment", lambda g: (info.get(g, {}) or {}).get("comp")),
        ("process", lambda g: (info.get(g, {}) or {}).get("proc")),
        ("any_regulator_known", lambda g: bool(reg_in.get(g))),
        ("abundance", lambda g: g in PPM), ("ptm", lambda g: g in PTM),
        ("any_ppi", lambda g: bool(ppi.get(g))), ("in_a_complex", lambda g: bool(cplx_of.get(g))),
        ("reactome_pathway", lambda g: bool(path_of.get(g))), ("metabolic_reaction", lambda g: g in RXN),
        ("GO_terms", lambda g: bool(GO.get(g))))}
    dossier["interconnection"] = {
        "pathways_shared_by_movers": pc.most_common(15),
        "compartments": comp_c.most_common(), "processes": proc_c.most_common(),
        "ppi_edges_among_movers": len(intra), "ppi_examples": sorted(intra)[:20],
        "complexes_with_2plus_movers": shared_cplx.most_common(15),
        "tf_cascade_within_movers": cascade[:15],
    }
    dossier["annotation_coverage_over_movers"] = cov

    json.dump(dossier, open(OUT / f"ko_dossier_{ko}.json", "w"), indent=1)

    # ---------------- readable summary ----------------
    e = dossier["effect"]
    print(f"KNOCKOUT DOSSIER: {ko} in K562   (|z| >= {T}, tide-masked)")
    print(f"\nTHE GENE ITSELF")
    idn = dossier["self"]["identity"]
    print(f"  {ko}: {idn['chrom']}:{idn['tss']} | compartment {idn['comp']} | process {idn['proc']} | "
          f"TF={idn['tf']} | essential={idn['ess']} | LOEUF {idn['loeuf']} | {idn['pubs']} papers")
    sj = dossier["self"]["journey"]
    print(f"  gene:          CpG island={sj['gene']['cpg_island']}, {sj['gene']['enhancers']} enhancers, "
          f"dep_frac {sj['gene']['dep_frac']}")
    print(f"  transcription: regulated by {sj['transcription']['n_regulators']} TFs; itself regulates "
          f"{dossier['self']['n_curated_targets']} curated targets")
    print(f"  translation:   abundance {sj['translation']['abundance_ppm']} ppm")
    print(f"  modification:  {sj['modification']}")
    print(f"  interaction:   {sj['interaction']['n_ppi']} PPI partners, complexes {sj['interaction']['complexes']}")
    print(f"  end state:     {sj['end_state']['reactome_primary']} | GO-P {sj['end_state']['GO_process'][:3]}")

    print(f"\nMEASURED EFFECT")
    print(f"  {e['n_moved_specific']} specific movers ({e['n_up']} up, {e['n_down']} down)")
    print(f"  curated TF->target edges for {ko}: {e['curated_targets_total']}; of those, "
          f"{e['n_direct_curated_targets_among_moved']} actually moved "
          f"({e['frac_of_curated_targets_that_moved']:.1%})")
    print(f"  movers that are curated direct targets: {e['n_direct_curated_targets_among_moved']}/"
          f"{e['n_moved_specific']} = {e['n_direct_curated_targets_among_moved']/max(e['n_moved_specific'],1):.1%} "
          f"-- the rest are indirect or uncurated")

    print(f"\nTOP MOVERS, WITH JOURNEY")
    print(f"  {'gene':10s} {'z':>7s} {'dir':>5s} {'direct':>7s} {'compartment':14s} {'process':20s} "
          f"{'ppi':>4s} {'ppm':>8s} pathway")
    for m in dossier["moved"][:DETAIL_N]:
        j = m["journey"]
        print(f"  {m['gene']:10s} {m['z']:>7.1f} {m['direction']:>5s} {str(m['direct_curated_target']):>7s} "
              f"{str(j['transport']['compartment'])[:14]:14s} {str(j['end_state']['process'])[:20]:20s} "
              f"{j['interaction']['n_ppi']:>4d} {str(j['translation']['abundance_ppm'] or '-'):>8s} "
              f"{(j['end_state']['reactome_primary'] or '-')[:34]}")

    ic = dossier["interconnection"]
    print(f"\nINTERCONNECTION AMONG THE {len(mset)} MOVERS")
    print(f"  compartments: {ic['compartments'][:6]}")
    print(f"  processes:    {ic['processes'][:6]}")
    print(f"  shared Reactome pathways (top): {ic['pathways_shared_by_movers'][:6]}")
    print(f"  PPI edges among movers: {ic['ppi_edges_among_movers']}")
    print(f"  complexes containing >=2 movers: {len(ic['complexes_with_2plus_movers'])}, "
          f"top {ic['complexes_with_2plus_movers'][:4]}")
    print(f"  TF cascade (movers that are TFs regulating other movers): {len(ic['tf_cascade_within_movers'])}")
    for c in ic["tf_cascade_within_movers"][:5]:
        print(f"     {c['tf']:10s} -> {c['n_downstream_also_moved']:3d} downstream movers  {c['examples'][:5]}")

    print(f"\nANNOTATION COVERAGE OVER THE MOVERS  (what fraction carry each layer at all)")
    for k, v in cov.items():
        print(f"  {k:22s} {v:.1%}")
    print(f"\n  -> outputs/orphan/ko_dossier_{ko}.json  (all {len(moved)} movers, full journey each)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GATA1")
