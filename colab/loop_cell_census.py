"""Loop 190. The census: which layer covers which gene, and what is missing.

WHY A CENSUS RATHER THAN ANOTHER PREDICTOR. This project now holds several layers that were each
validated on their own -- a metabolic reaction network, a TF regulatory network with signs, a
protein-interaction graph, motif and structural descriptions of transcription factors, an enhancer
model, and chromatin marks. Each has a number attached to it. What has never been written down is
which GENES each layer actually reaches, and where they overlap. Without that, "we have a cell map"
is a claim about the number of files rather than about coverage.

WHAT SET THIS OFF. The gene table files 3,783 genes under the process label "other". UniProt has a
reviewed FUNCTION paragraph, an EC number, or a GO molecular-function term for 96.5% of them. Two
consequences are itemised rather than summarised: 1,029 of those carry a DNA-binding domain or a
transcription keyword while the table flags them tf = 0, so every TF-layer result in this arc ran
on a roster missing about a thousand factors; and 647 carry an EC number while being absent from
Human-GEM, which is a LIST of unmodelled enzymatic genes rather than an estimate of how many there
might be.

THE TWO NUMBERS THIS LOOP EXISTS TO PRODUCE.

  THE COVERAGE HISTOGRAM -- how many genes sit in zero layers, one layer, two, and so on. A gene in
  zero layers is invisible to everything built here regardless of how well studied it is.

  THE COUPLING MATRIX -- how many genes are in two layers AT ONCE. This is the number that decides
  whether these are one map or several maps of different territories, and it is the question a
  coverage percentage cannot answer. A cell model that knows the chemistry of one set of genes and
  the regulation of a disjoint set does not know how regulation changes chemistry, no matter how
  good either half is.

PREDECLARED, BEFORE ANY NUMBER.

  D1 IS THE ROSTER FAIR? The gene table against UniProt's reviewed human proteome.
     Gate: PASS iff at least 95% of the table's symbols match a reviewed entry. Below that the
     census is counting a roster with its own biases rather than the proteome.

  D2 THE COVERAGE HISTOGRAM. Descriptive; the distribution is the output.

  D3 THE COUPLING, and it is the gate that matters. Of the genes with a modelled reaction, how many
     also carry a curated TF regulator.
     Gate: PASS iff more than half of them do. Below half, the mechanism layer and the regulation
     layer are joined on a minority of their own members and the honest description is "separate
     maps", which is then said in the verdict rather than in a footnote.

  D4 THE LEDGER. Itemised incompleteness, each as a count and a list rather than a percentage:
     enzymes with an EC number outside the metabolic model; reactions with no gene at all;
     transcription factors with no motif; transcription factors absent from the regulatory network;
     genes with no regulator; genes in no layer at all.
     Descriptive; the items are the output.

  D5 WHAT THE MAP CAN ANSWER TODAY. Each question type this project has measured, with the number
     that was measured and the loop it came from, so the capability statement cannot drift from the
     evidence.

  D6 WHAT THIS CANNOT SHOW.

-> outputs/loop_cell_census.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_cell_census.json"
BUNDLE = Path("colab/data/net_bundle.json.gz")
TABLE = Path("colab/data/cell_complete.json.gz")
ANNO = Path("colab/data/all_gene_annotation.json")
CUR = (0, 55716)
MIN_MATCH = 0.95
MIN_COUPLE = 0.50

# what this project has measured, and where. Kept here so D5 cannot drift from the record.
MEASURED = [
    ("which metabolite completes a reaction", "hit@1 0.8506", "loop 170"),
    ("...at 50% coverage, when confident", "precision 0.9986", "loop 168"),
    ("which enzyme catalyses a reaction", "0.825", "loop 163d"),
    ("is this DNA an enhancer", "AUC 0.8506", "loop 177"),
    ("which gene does an enhancer control", "R@1 0.6734 vs distance 0.5930", "loop 185"),
    ("what makes a TF bind an enhancer", "co-binding 0.8455 > accessibility 0.7902 > motif 0.6228",
     "loop 184"),
    ("do feedforward loops exist beyond chance", "z = +0.8, no", "loop 187"),
    ("do feedback loops", "z = +43.8, yes", "loop 187"),
    ("does 3D contact say which gene", "three instruments, all fail the stranger-swap",
     "loops 181, 186"),
]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 190  THE CENSUS: which layer covers which gene, and what is missing")
    say("=" * 104)
    say(f"  PREDECLARED: at least {MIN_MATCH:.0%} of the table's symbols must match a reviewed")
    say("  UniProt entry for the census to be about the proteome rather than about this roster;")
    say(f"  and more than {MIN_COUPLE:.0%} of the genes with a modelled reaction must carry a")
    say("  curated TF regulator, or the mechanism and regulation layers are separate maps and the")
    say("  verdict says so.")
    say()

    tab = json.load(gzip.open(TABLE))["genes"]
    sym = [str(g["name"]).upper() for g in tab]
    idx = {s: i for i, s in enumerate(sym)}
    anno = json.load(open(ANNO))["classification"]
    A = {k.upper(): v for k, v in anno.items()}
    say(f"    gene table: {len(tab):,} symbols")

    # ---- D1 ------------------------------------------------------------------------------------
    say()
    say("D1 IS THE ROSTER FAIR?")
    matched = sum(1 for s in sym if A.get(s, {}).get("evidence") != "no reviewed entry"
                  and s in A)
    known = sum(1 for s in sym if A.get(s, {}).get("known"))
    say(f"     matched a reviewed UniProt entry: {matched:,}/{len(sym):,} ({matched/len(sym):.1%})")
    say(f"     with a FUNCTION paragraph, EC number or GO molecular-function term: "
        f"{known:,} ({known/len(sym):.1%})")
    say(f"     with none of those: {len(sym)-known:,} ({1-known/len(sym):.1%})")
    d1 = bool(matched / len(sym) >= MIN_MATCH)
    GG.verdict(d1, emit=say,
               if_true=f"D1 PASS -- {matched/len(sym):.1%} of the roster is a reviewed human "
                       f"protein, so the counts below are about the proteome",
               if_false=f"D1 FAIL -- only {matched/len(sym):.1%} matched; the census would be "
                        f"describing this table's idiosyncrasies")

    # ---- layers ----------------------------------------------------------------------------
    z = np.load("colab/data/rem_enzyme.npz", allow_pickle=True)
    met = {str(s).upper() for s in z["symbols"] if s}
    grx, gg = z["gpr_rx"], z["gpr_gene"]
    n_rx = len(z["reactions"])
    rx_with_gene = len(set(grx.tolist()))

    nb = json.load(gzip.open(BUNDLE))
    names, reg, ppi = nb["names"], nb["reg"], nb["ppi"]
    nidx = {n.upper(): i for i, n in enumerate(names)}
    cur = reg[CUR[0]:CUR[1]]
    regulators = {int(r[0]) for r in cur}
    targets = defaultdict(int)
    for r in cur:
        targets[int(r[1])] += 1
    ppi_deg = Counter()
    for a, b in ppi:
        ppi_deg[int(a)] += 1
        ppi_deg[int(b)] += 1

    dom = json.load(open("colab/data/tf_domains.json"))["matrices"]
    motif_names = {(v.get("name") or "").upper().split("::")[0] for v in dom.values()}

    L = {}
    L["reaction"] = np.array([s in met for s in sym])
    L["enzyme_EC"] = np.array([bool(A.get(s, {}).get("has_ec")) for s in sym])
    L["is_TF"] = np.array([bool(A.get(s, {}).get("is_tf")) for s in sym])
    L["TF_in_network"] = np.array([nidx.get(s, -1) in regulators for s in sym])
    L["has_regulator"] = np.array([targets.get(nidx.get(s, -1), 0) > 0 for s in sym])
    L["has_PPI"] = np.array([ppi_deg.get(nidx.get(s, -1), 0) > 0 for s in sym])
    L["has_motif"] = np.array([s in motif_names for s in sym])
    L["essential_measured"] = np.array([g.get("ess_src") == "measured" for g in tab])
    L["function_known"] = np.array([bool(A.get(s, {}).get("known")) for s in sym])

    say()
    say("D2 THE COVERAGE HISTOGRAM")
    for k, v in L.items():
        say(f"     {k:20} {int(v.sum()):7,}  {v.mean():6.1%}")
    core = ["reaction", "TF_in_network", "has_regulator", "has_PPI", "has_motif"]
    n_layers = np.sum([L[k] for k in core], axis=0)
    say(f"     genes in N of the {len(core)} MECHANISTIC layers "
        f"(reaction, TF, regulated, PPI, motif):")
    for k in range(len(core) + 1):
        n = int((n_layers == k).sum())
        say(f"       {k} layers  {n:7,}  {n/len(sym):6.1%}")
    orphan = [sym[i] for i in range(len(sym)) if n_layers[i] == 0]
    say(f"     in NO mechanistic layer: {len(orphan):,}; "
        f"of those, {sum(1 for s in orphan if A.get(s, {}).get('known')):,} have a known function")

    # ---- D3 ------------------------------------------------------------------------------------
    say()
    say("D3 THE COUPLING")
    m = L["reaction"]
    both = int((m & L["has_regulator"]).sum())
    say(f"     genes with a modelled reaction:              {int(m.sum()):,}")
    say(f"       of those, with a curated TF regulator:     {both:,} ({both/max(int(m.sum()),1):.1%})")
    say(f"       of those, that are themselves TFs:         {int((m & L['is_TF']).sum()):,}")
    say(f"     genes that are TFs with a motif here:        {int((L['is_TF'] & L['has_motif']).sum()):,}")
    pairs = {}
    for a in core:
        for b in core:
            if a < b:
                pairs[f"{a} & {b}"] = int((L[a] & L[b]).sum())
    say("     pairwise overlaps between mechanistic layers:")
    for k, v in sorted(pairs.items(), key=lambda x: -x[1]):
        say(f"       {k:44} {v:7,}")
    d3 = bool(both / max(int(m.sum()), 1) > MIN_COUPLE)
    GG.verdict(d3, emit=say,
               if_true=f"D3 PASS -- {both/max(int(m.sum()),1):.1%} of the modelled enzymes carry a "
                       f"curated regulator, so the two layers are joined on a majority of their "
                       f"shared members",
               if_false=f"D3 FAIL -- only {both/max(int(m.sum()),1):.1%} of modelled enzymes have "
                        f"a curated regulator. Mechanism and regulation are SEPARATE MAPS here, "
                        f"and no amount of accuracy in either closes that")

    # ---- D4 ------------------------------------------------------------------------------------
    say()
    say("D4 THE LEDGER -- itemised incompleteness")
    ec_out = [sym[i] for i in range(len(sym)) if L["enzyme_EC"][i] and not L["reaction"][i]]
    tf_nomotif = [sym[i] for i in range(len(sym)) if L["is_TF"][i] and not L["has_motif"][i]]
    tf_nonet = [sym[i] for i in range(len(sym)) if L["is_TF"][i] and not L["TF_in_network"][i]]
    noreg = [sym[i] for i in range(len(sym)) if not L["has_regulator"][i]]
    unk = [sym[i] for i in range(len(sym)) if not L["function_known"][i]]
    items = [
        ("enzymes with an EC number OUTSIDE the metabolic model", ec_out),
        ("transcription factors with NO motif in this project", tf_nomotif),
        ("transcription factors absent from the regulatory network", tf_nonet),
        ("genes with NO curated TF regulator", noreg),
        ("genes in NO mechanistic layer at all", orphan),
        ("genes with no UniProt function evidence", unk),
    ]
    for label, lst in items:
        say(f"     {label:58} {len(lst):7,}  e.g. {', '.join(lst[:4])}")
    say(f"     reactions with NO gene at all                              {n_rx-rx_with_gene:7,} "
        f"of {n_rx:,}")
    d4 = True
    say(f"     D4 {'PASS' if d4 else 'FAIL'} (descriptive)")

    # ---- D5 ------------------------------------------------------------------------------------
    say()
    say("D5 WHAT THE MAP CAN ANSWER TODAY, with the loop each number came from")
    for q, val, src in MEASURED:
        say(f"     {q:48} {val:44} {src}")
    d5 = True
    say(f"     D5 {'PASS' if d5 else 'FAIL'}")

    say()
    say("D6 WHAT THIS CANNOT SHOW")
    say("     A gene being in a layer is not the same as that layer predicting anything about it.")
    say("     Coverage is a lower bound on ignorance, never an upper bound on knowledge.")
    say("     The curated regulatory tier is literature-derived, so 'has a regulator' partly means")
    say("     'has been studied'. The binding tier would raise D3's number and would mean less.")
    say("     Nothing here is dynamic. Every layer counted is a steady-state description, and no")
    say("     count of parts becomes a trajectory.")
    say("     The roster is this project's gene table, which is 16,492 of roughly 20,000 protein-")
    say("     coding genes, so about 3,500 genes are outside the census entirely.")
    d6 = True
    say(f"     D6 {'PASS' if d6 else 'FAIL'}")

    gates = {"D1": d1, "D2": True, "D3": d3, "D4": d4, "D5": d5, "D6": d6}
    man = RM.manifest(inputs=[TABLE, BUNDLE, ANNO, Path("colab/data/rem_enzyme.npz")],
                      available=len(sym), used=len(sym), selection="the whole gene table", seed=0,
                      controls=["UniProt reviewed entries as the roster check",
                                "the curated regulatory tier only, with the binding tier's effect "
                                "on D3 stated rather than used"],
                      note="which layer covers which gene, and the itemised incompleteness")
    out = dict(test="cell census", gates=gates, n_genes=len(sym),
               uniprot_matched=matched, function_known=known,
               layers={k: int(v.sum()) for k, v in L.items()},
               histogram={int(k): int((n_layers == k).sum()) for k in range(len(core) + 1)},
               pairwise=pairs,
               coupling=dict(modelled_enzymes=int(m.sum()), with_regulator=both,
                             fraction=both / max(int(m.sum()), 1)),
               ledger={label: dict(n=len(lst), examples=lst[:40]) for label, lst in items},
               reactions_total=n_rx, reactions_with_gene=rx_with_gene,
               measured=[dict(question=q, value=v, source=s) for q, v, s in MEASURED],
               manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    out["log"] = log
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
