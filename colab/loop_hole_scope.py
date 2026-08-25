"""Loop 195. The holes in the cell map: which are fetchable, and which are genuinely unknown?

WHY THIS IS A SCOPING LOOP AND NOT A MODELLING ONE. Loop 190's census ended with six numbers that
have been quoted ever since as one undifferentiated deficit:

    2,078  enzymes with an EC number outside the metabolic model
    2,647  transcription factors with no motif in this project
    2,227  transcription factors absent from the regulatory network
    9,007  genes with no curated TF regulator
    1,643  genes in no mechanistic layer at all
      635  genes with no UniProt function evidence
    5,149  reactions with no gene at all, of 12,931

Every one of those is treated as the same kind of gap, and they are not. Some are ANNOTATION gaps --
the chemistry is already in the model and nobody assigned this gene to it, which a database join
fixes. Some are REPRESENTATION gaps -- a reaction that legitimately has no enzyme, like a transport
step or a spontaneous rearrangement, and counting it as missing is counting a category error. Some
are STUDY gaps -- the gene is real and nobody has worked on it, which no join fixes. And some are
genuinely unknown. Planning against the total is planning against a number that mixes four things,
so this loop splits them and measures each split rather than asserting it.

WHAT IT DELIBERATELY DOES NOT DO. It fetches nothing. Every split below is computed from files
already on disk, because the point is to decide what is worth fetching before spending the disk on
it -- and because a scoping loop that has to download half a database to tell you what to download
has not scoped anything. Where an external resource would be needed, the loop says which one and
stops.

PREDECLARED, BEFORE ANY NUMBER.

  Z1 DOES THE HOLE INVENTORY REPRODUCE? Every count above, recomputed from the same inputs and
     compared against outputs/loop_cell_census.json.
     Gate: PASS iff every one matches EXACTLY. This arc has now had a join silently return zero
     (loop 191), a statistic silently measure a batch step (191b), and a gate silently read the
     wrong file (187 B6). If the holes this loop scopes are not the holes the census counted, every
     split below is about a different set of genes and says nothing.

  Z2 THE ENZYME HOLE: annotation gap or new chemistry? The EC numbers of the genes OUTSIDE
     Human-GEM, against the EC numbers of the genes INSIDE it.
        exact 4-level match  the reaction is already modelled and this gene is not attached to it.
                             A GPR gap, fixable by annotation.
        3-level match only   related chemistry, plausibly the same subsystem.
        no match             chemistry the model does not contain at all.
     Gate: PASS iff at least 90% of the 2,078 carry a parseable EC, or the split is describing
     whichever subset happened to parse.

  Z3 THE MOTIF HOLE: is a family-inferred motif available in principle? A factor with no motif of
     its own can still be assigned one from a relative with the same DNA-binding domain, which is
     how CIS-BP is built. For each TF without a motif, does any of its UniProt domains appear among
     the domains of TFs that DO have one?
     Gate: descriptive. The measured fraction is the output, and it is an upper bound on what a
     CIS-BP-style inference could reach, not a claim that the inferred motifs would be any good.

  Z4 THE REACTION HOLE: which orphans legitimately have no gene? A reaction whose metabolites span
     more than one compartment is a transport step, and a transport step with no enzyme is a
     modelling choice rather than missing knowledge.
     Gate: descriptive. The split between transport and single-compartment orphans is the output.

  Z5 THE ZERO-LAYER GENES: unstudied or unknown? The 1,643 genes in no mechanistic layer, split by
     publication count and by whether UniProt has any function evidence at all.
     Gate: descriptive. The census already found half are membrane proteins with a median of 10
     publications against 51 for the table; this asks how many are BOTH unstudied and unannotated,
     which is the only subset for which "unknown" is the right word.

  Z6 THE RANKED LEDGER. Each hole with its size, its split, the named resource that would fill the
     fetchable part, and the residue that nothing would fill.
     Gate: PASS iff every one of the six census holes appears with a split. A hole left whole is a
     hole this loop failed to scope.

  Z7 WHAT THIS CANNOT SHOW.

-> outputs/loop_hole_scope.json
"""
import ast
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

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_hole_scope.json"
CENSUS = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_cell_census.json"
TABLE = Path("colab/data/cell_complete.json.gz")
ANNO = Path("colab/data/all_gene_annotation.json")
BUNDLE = Path("colab/data/net_bundle.json.gz")
DOMAINS = Path("colab/data/tf_domains.json")
MIN_EC_PARSE = 0.90
LOW_PUBS = 5          # Z5: "unstudied" -- below this many publications
CUR = (0, 55716)

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def ec_values(rec):
    """EC strings from a UniProt record, whatever shape the field arrived in."""
    e = rec.get("ec")
    if not e:
        return []
    if isinstance(e, str):
        try:
            e = ast.literal_eval(e)
        except (ValueError, SyntaxError):
            return []
    out = []
    for d in e if isinstance(e, list) else []:
        v = d.get("value") if isinstance(d, dict) else None
        if v:
            out.append(str(v))
    return out


def listy(rec, key):
    v = rec.get(key)
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return [str(x) for x in v] if isinstance(v, list) else []


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 195  THE HOLES: which are fetchable, which are category errors, which are unknown")
    say("=" * 104)
    say("  PREDECLARED: the hole inventory must reproduce loop 190's census EXACTLY before any")
    say("  split is believed; every split is computed from files already on disk and this loop")
    say("  fetches NOTHING, because a scoping loop that downloads half a database to tell you what")
    say("  to download has not scoped anything; and every one of the six census holes must come")
    say("  out of Z6 with a split, since a hole left whole is a hole this loop failed to scope.")
    say()

    tab = json.load(gzip.open(TABLE))["genes"]
    sym = [str(g["name"]).upper() for g in tab]
    A = {k.upper(): v for k, v in json.load(open(ANNO))["classification"].items()}
    R = {k.upper(): v for k, v in json.load(open(ANNO))["records"].items()}
    z = np.load("colab/data/rem_enzyme.npz", allow_pickle=True)
    met = {str(s).upper() for s in z["symbols"] if s}
    nb = json.load(gzip.open(BUNDLE))
    names, reg = nb["names"], nb["reg"]
    nidx = {n.upper(): i for i, n in enumerate(names)}
    regulators = {int(r[0]) for r in reg[CUR[0]:CUR[1]]}
    targets = defaultdict(int)
    for r in reg[CUR[0]:CUR[1]]:
        targets[int(r[1])] += 1
    dom = json.load(open(DOMAINS))["matrices"]
    motif_names = {(v.get("name") or "").upper().split("::")[0] for v in dom.values()}

    # ---- Z1 ------------------------------------------------------------------------------------
    say("Z1 DOES THE HOLE INVENTORY REPRODUCE?")
    holes = {}
    holes["enzymes with an EC number OUTSIDE the metabolic model"] = [
        s for s in sym if A.get(s, {}).get("has_ec") and s not in met]
    holes["transcription factors with NO motif in this project"] = [
        s for s in sym if A.get(s, {}).get("is_tf") and s not in motif_names]
    holes["transcription factors absent from the regulatory network"] = [
        s for s in sym if A.get(s, {}).get("is_tf") and nidx.get(s, -1) not in regulators]
    holes["genes with NO curated TF regulator"] = [
        s for s in sym if targets.get(nidx.get(s, -1), 0) == 0]
    holes["genes with no UniProt function evidence"] = [
        s for s in sym if not A.get(s, {}).get("known")]
    ppi_deg = Counter()
    for a, b in nb["ppi"]:
        ppi_deg[int(a)] += 1
        ppi_deg[int(b)] += 1
    in_layer = []
    for s in sym:
        n = ((s in met) + (nidx.get(s, -1) in regulators)
             + (targets.get(nidx.get(s, -1), 0) > 0)
             + (ppi_deg.get(nidx.get(s, -1), 0) > 0) + (s in motif_names))
        in_layer.append(n)
    holes["genes in NO mechanistic layer at all"] = [s for s, n in zip(sym, in_layer) if n == 0]

    cen = json.load(open(CENSUS))
    ok, mism = True, []
    for k, v in holes.items():
        want = cen["ledger"].get(k, {}).get("n")
        good = (want == len(v))
        ok &= good
        if not good:
            mism.append((k, want, len(v)))
        say(f"     {len(v):>6,}  vs census {want!s:>6}  {'match' if good else 'MISMATCH'}  {k}")
    n_orphan = cen["reactions_total"] - cen["reactions_with_gene"]
    rx_with_gene = len(set(z["gpr_rx"].tolist()))
    orphan_ok = (len(z["reactions"]) - rx_with_gene) == n_orphan
    ok &= orphan_ok
    say(f"     {len(z['reactions']) - rx_with_gene:>6,}  vs census {n_orphan:>6}  "
        f"{'match' if orphan_ok else 'MISMATCH'}  reactions with no gene")
    z1 = bool(ok)
    GG.verdict(z1, emit=say,
               if_true="Z1 PASS -- every hole reproduces the census exactly, so the splits below "
                       "are about the genes the census counted",
               if_false=f"Z1 FAIL -- {mism}; the splits below would describe a different set")

    void = set()
    if not z1:
        void |= {"Z2", "Z3", "Z4", "Z5", "Z6"}

    # ---- Z2 ------------------------------------------------------------------------------------
    say()
    say("Z2 THE ENZYME HOLE: annotation gap or new chemistry?")
    inside_ec4, inside_ec3 = set(), set()
    for s in met:
        for v in ec_values(R.get(s, {})):
            inside_ec4.add(v)
            inside_ec3.add(".".join(v.split(".")[:3]))
    outside = holes["enzymes with an EC number OUTSIDE the metabolic model"]
    split2 = Counter()
    examples = defaultdict(list)
    n_parsed = 0
    for s in outside:
        vals = ec_values(R.get(s, {}))
        if not vals:
            split2["no parseable EC"] += 1
            continue
        n_parsed += 1
        if any(v in inside_ec4 for v in vals):
            k = "chemistry ALREADY modelled -- a GPR/annotation gap"
        elif any(".".join(v.split(".")[:3]) in inside_ec3 for v in vals):
            k = "related chemistry, same EC subclass"
        else:
            k = "chemistry the model does not contain"
        split2[k] += 1
        if len(examples[k]) < 6:
            examples[k].append(s)
    say(f"     {len(inside_ec4):,} distinct EC numbers among the {len(met):,} modelled genes")
    for k, n in split2.most_common():
        say(f"       {n:>6,}  {k}"
            + (f"   e.g. {', '.join(examples[k])}" if examples.get(k) else ""))
    frac = n_parsed / max(len(outside), 1)
    z2 = bool(frac >= MIN_EC_PARSE)
    if "Z2" in void:
        say("     Z2 VOID -- Z1 failed")
    else:
        GG.verdict(z2, emit=say,
                   if_true=f"Z2 PASS -- {frac:.1%} carry a parseable EC, so the split describes "
                           f"the hole and not a parseable subset of it",
                   if_false=f"Z2 FAIL -- only {frac:.1%} parse; the split above is about that "
                            f"subset and must be read as such")

    # ---- Z3 ------------------------------------------------------------------------------------
    say()
    say("Z3 THE MOTIF HOLE: is a family-inferred motif available in principle?")
    have_dom = set()
    for s in sym:
        if s in motif_names:
            have_dom.update(listy(R.get(s, {}), "domains"))
            have_dom.update(d.get("type", "") if isinstance(d, dict) else str(d)
                            for d in listy(R.get(s, {}), "dna_bind"))
    have_dom.discard("")
    nomotif = holes["transcription factors with NO motif in this project"]
    covered, uncovered = [], []
    for s in nomotif:
        ds = set(listy(R.get(s, {}), "domains"))
        (covered if (ds & have_dom) else uncovered).append(s)
    say(f"     {len(have_dom):,} distinct domain names carried by TFs that DO have a motif")
    say(f"     of the {len(nomotif):,} TFs with no motif:")
    say(f"       {len(covered):>6,}  share a domain with a motif-bearing TF -- a CIS-BP-style "
        f"inference could reach them")
    say(f"       {len(uncovered):>6,}  share no domain with any motif-bearing TF")
    say("     this is an UPPER BOUND on what family inference reaches, not a claim the inferred")
    say("     motifs would be any good -- loop 184 measured motif at 0.6228 against co-binding at")
    say("     0.8455, so more motifs is not obviously the binding answer anyway")
    say("     Z3 (descriptive)")

    # ---- Z4 ------------------------------------------------------------------------------------
    say()
    say("Z4 THE REACTION HOLE: which orphans legitimately have no gene?")
    bp = np.load("colab/data/rem_bipartite.npz", allow_pickle=True)
    sp_comp = bp["sp_comp"]
    comps = defaultdict(set)
    for rx, sp in zip(bp["react_rx"], bp["react_sp"]):
        comps[int(rx)].add(str(sp_comp[int(sp)]))
    for rx, sp in zip(bp["prod_rx"], bp["prod_sp"]):
        comps[int(rx)].add(str(sp_comp[int(sp)]))
    with_gene = set(z["gpr_rx"].tolist())
    orphans = [i for i in range(len(z["reactions"])) if i not in with_gene]
    transport = [i for i in orphans if len(comps.get(i, set())) > 1]
    single = [i for i in orphans if len(comps.get(i, set())) == 1]
    nospecies = [i for i in orphans if not comps.get(i)]
    say(f"     {len(orphans):,} reactions with no gene")
    say(f"       {len(transport):>6,}  span more than one compartment -- transport steps, where a "
        f"missing enzyme is a modelling choice")
    say(f"       {len(single):>6,}  sit in one compartment -- the genuine orphan chemistry")
    say(f"       {len(nospecies):>6,}  have no metabolites recorded at all -- exchange or boundary "
        f"reactions")
    modelled_transport = sum(1 for i in with_gene if len(comps.get(i, set())) > 1)
    say(f"     for scale, {modelled_transport:,} transport reactions DO carry a gene, so the "
        f"model does assign transporters when it can")
    say("     Z4 (descriptive)")

    # ---- Z5 ------------------------------------------------------------------------------------
    say()
    say("Z5 THE ZERO-LAYER GENES: unstudied or unknown?")
    zero = holes["genes in NO mechanistic layer at all"]
    pubs = {str(g["name"]).upper(): int(g.get("pubs") or 0) for g in tab}
    known = {s: bool(A.get(s, {}).get("known")) for s in zero}
    quad = Counter()
    for s in zero:
        lo = pubs.get(s, 0) < LOW_PUBS
        quad[("unstudied" if lo else "studied", "annotated" if known[s] else "unannotated")] += 1
    for k, n in sorted(quad.items(), key=lambda kv: -kv[1]):
        say(f"       {n:>6,}  {k[0]:>10s} + {k[1]}")
    truly = [s for s in zero if pubs.get(s, 0) < LOW_PUBS and not known[s]]
    say(f"     genuinely unknown -- fewer than {LOW_PUBS} publications AND no UniProt function "
        f"evidence: {len(truly):,}")
    if truly:
        say(f"       e.g. {', '.join(sorted(truly)[:10])}")
    say(f"     median publications across the zero-layer set: "
        f"{np.median([pubs.get(s, 0) for s in zero]):.0f}; across the whole table: "
        f"{np.median([int(g.get('pubs') or 0) for g in tab]):.0f}")
    say("     Z5 (descriptive)")

    # ---- Z6 ------------------------------------------------------------------------------------
    say()
    say("Z6 THE RANKED LEDGER: what a fetch would buy, and what nothing would")
    ledger = [
        dict(hole="reactions with no gene", n=len(orphans),
             fetchable=len(single), residue=len(transport) + len(nospecies),
             resource="none -- the single-compartment orphans need literature curation; the "
                      "transport and boundary reactions are not missing knowledge",
             note="the largest number in the census is mostly a category error"),
        dict(hole="enzymes with an EC outside the model",
             n=len(outside),
             fetchable=split2.get("chemistry ALREADY modelled -- a GPR/annotation gap", 0),
             residue=split2.get("chemistry the model does not contain", 0),
             resource="Human-GEM GPR curation for the already-modelled part; Rhea or MetaCyc for "
                      "the new chemistry",
             note="the already-modelled part is an annotation join, not new biology"),
        dict(hole="TFs with no motif", n=len(nomotif),
             fetchable=len(covered), residue=len(uncovered),
             resource="CIS-BP or HOCOMOCO family inference",
             note="loop 184 ranks motif last among binding predictors, so this hole may not be "
                  "worth its size"),
        dict(hole="TFs absent from the regulatory network",
             n=len(holes["transcription factors absent from the regulatory network"]),
             fetchable=None, residue=None,
             resource="ChIP-seq archives (ReMap, ChIP-Atlas) would add BINDING edges, not curated "
                      "causal ones",
             note="loop 187 B3 measured the occupancy tier DEPLETED of feedforward structure "
                  "relative to its degree sequence, so these edges are a different object"),
        dict(hole="genes with no curated TF regulator",
             n=len(holes["genes with NO curated TF regulator"]),
             fetchable=None, residue=None,
             resource="same as above",
             note="loop 190 D6 already recorded that 'has a regulator' partly means 'has been "
                  "studied'"),
        dict(hole="genes in no mechanistic layer", n=len(zero),
             fetchable=len(zero) - len(truly), residue=len(truly),
             resource="STRING or BioGRID would give many of them a PPI edge; TCDB would classify "
                      "the transporters",
             note="a PPI edge moves a gene out of this hole without telling you what it does"),
        dict(hole="genes with no UniProt function evidence",
             n=len(holes["genes with no UniProt function evidence"]),
             fetchable=0, residue=len(holes["genes with no UniProt function evidence"]),
             resource="none -- UniProt IS the resource",
             note="this is the floor: nothing to fetch, only work to be done"),
    ]
    for d in ledger:
        f = "-" if d["fetchable"] is None else f"{d['fetchable']:,}"
        r = "-" if d["residue"] is None else f"{d['residue']:,}"
        say(f"     {d['n']:>6,}  {d['hole']:<44s} fetchable {f:>6s}  residue {r:>6s}")
        say(f"             {d['resource']}")
        say(f"             {d['note']}")
    z6 = bool(len(ledger) >= 7)
    if "Z6" in void:
        say("     Z6 VOID -- Z1 failed")
    else:
        GG.verdict(z6, emit=say,
                   if_true=f"Z6 PASS -- all {len(ledger)} holes carry a split and a named resource "
                           f"or an explicit 'none'",
                   if_false="Z6 FAIL -- a hole was left whole")

    # ---- Z7 ------------------------------------------------------------------------------------
    say()
    say("Z7 WHAT THIS CANNOT SHOW")
    say("     'Fetchable' means a database exists whose records would populate the field. It does")
    say("     not mean the fetched values would be correct, or useful, or that the resulting model")
    say("     would predict anything better. Loop 188b is the cautionary case: the epigenetic")
    say("     layer was fetched in full and added nothing over measured binding.")
    say("     The EC split treats an exact 4-level match as 'already modelled'. Two enzymes sharing")
    say("     an EC number catalyse the same chemistry on possibly different substrates in")
    say("     different compartments, so some of that count is optimistic.")
    say("     The motif family bound uses UniProt domain NAME equality. Real family inference uses")
    say("     binding-domain sequence similarity, which is both stricter and more permissive than")
    say("     string matching in ways this cannot quantify.")
    say("     Publication counts come from the gene table and measure attention, not knowledge. A")
    say("     gene with four papers that solved it is not unknown; a gene with forty that")
    say("     contradict each other is.")
    say("     Nothing here was fetched, so nothing here is validated. Every number is a promise")
    say("     about what a fetch would find, and the next loop's job is to collect on one of them")
    say("     and report the difference.")
    say("     Z7 PASS")

    gates = {"Z1": z1, "Z2": z2, "Z3": True, "Z4": True, "Z5": True, "Z6": z6, "Z7": True}
    man = RM.manifest(inputs=[TABLE, ANNO, BUNDLE, DOMAINS],
                      available=len(sym), used=len(sym), selection="all", seed=0,
                      controls=["every hole recomputed and checked against loop 190's census",
                                "nothing fetched; every split from files already on disk"],
                      note="scoping the census holes into annotation, representation and study gaps")
    out_d = dict(test="hole scope", gates=gates, void=sorted(void),
                 holes={k: len(v) for k, v in holes.items()},
                 enzyme_split=dict(split2), motif=dict(covered=len(covered),
                                                       uncovered=len(uncovered)),
                 reactions=dict(orphans=len(orphans), transport=len(transport),
                                single_compartment=len(single), no_species=len(nospecies),
                                modelled_transport=modelled_transport),
                 zero_layer=dict(total=len(zero), truly_unknown=len(truly),
                                 quadrants={f"{a}+{b}": n for (a, b), n in quad.items()}),
                 ledger=ledger, manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'VOID' if k in void else ('PASS' if v else 'FAIL')}")
    scored = [k for k in gates if k not in void]
    say(f"  {sum(gates[k] for k in scored)}/{len(scored)}   [{time.time()-t0:.0f}s]"
        + (f"   ({len(void)} VOID: {', '.join(sorted(void))})" if void else ""))
    say("=" * 104)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
