"""The expression machinery layer: transcription, translation and protein maturation. A cache, no gates.

WHY THIS IS A SEPARATE SOURCE. Human-GEM has no transcription and no translation. It has 2 mentions
of the words and zero polymerase or ribosome reactions; the machinery appears only as the DIFFERENCE
between two biomass pseudo-reactions -- MAR09932 is literally "Biomass maintenance reaction without
replication, transcription, and translation". A metabolic model lumps expression into a coefficient
on purpose. So the expression layer has to come from Reactome, which carries 15,597 human reactions
with 13,164 protein and 18,612 complex participants against Human-GEM's 2,848 genes.

THE JOIN IS THE INTERESTING PART, AND IT IS MEASURED HERE RATHER THAN ASSUMED. The two sources share
only 1,132 genes out of a 10,610 union, so they are almost disjoint at the protein level. They can
still meet through METABOLITES -- translation consumes amino acids and GTP, transcription consumes
NTPs, and those are Human-GEM species. The question a "full cell model" turns on is whether they
meet anywhere ELSE, because if the only bridge is ATP and GTP then the two graphs are glued at the
currency hubs and the assembly is two models in a trenchcoat. This builder records the bridge
species so that can be counted rather than claimed.

PROCESS CLASSES, AND THE INDIRECTION THAT WAS FORCED ON THEM. The intent was to class each reaction
by ancestor closure in the Reactome pathway hierarchy. That is not possible with these files: all
15,597 steps appear in NCBI2Reactome_PE_Reactions column 3, but ReactomePathwaysRelation contains
only 2,854 nodes and NONE of the 15,597 events appears as a child of any of them -- the leaf events
sit below the hierarchy the relation file describes. So the link runs through genes instead:

    event --(PE file col0 -> col3)--> NCBI gene --(All_Levels)--> pathways --(closure)--> class

That is weaker than a direct containment, because genes are pleiotropic and a gene in Translation
also appears in many other pathways. The SUPPORT -- what fraction of an event's genes carry the
class assigned to it -- is therefore stored per reaction, so any downstream use can demand a
threshold instead of trusting the label. Reactions with no gene at all cannot be classed and are
counted separately rather than dropped into "other" silently.

WHAT IS KNOWN TO BE WRONG WITH THIS SOURCE. 46,450 of the repo's Reactome participants are
UNRESOLVED -- an entity the extraction could not type. That is 42% and it is recorded per reaction
here so any downstream claim can be restricted to the resolved part.

-> colab/data/rem_expression.npz
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loop_replication as LR  # noqa: E402

SC = LR.SC
NET = Path("outputs/orphan/reaction_network.json")
PE = SC / "NCBI2Reactome_PE_Reactions.txt"
REL = SC / "ReactomePathwaysRelation.txt"
PATH = SC / "ReactomePathways.txt"
ALL = SC / "NCBI2Reactome_All_Levels.txt"
OUT = Path("colab/data/rem_expression.npz")

ROOTS = {
    "R-HSA-74160": "transcription",          # Gene expression (Transcription)
    "R-HSA-72766": "translation",
    "R-HSA-597592": "maturation",            # Post-translational protein modification
    "R-HSA-8953854": "maturation",           # Metabolism of RNA -> reassigned below if also txn
    "R-HSA-392499": "protein_metabolism",     # parent of translation + PTM
    "R-HSA-1430728": "metabolism",
    "R-HSA-162582": "signalling",
    "R-HSA-1640170": "cell_cycle",
    "R-HSA-73894": "dna_repair",
    "R-HSA-69306": "dna_replication",
}
PRIORITY = ["transcription", "translation", "maturation", "dna_replication", "dna_repair",
            "cell_cycle", "signalling", "metabolism", "protein_metabolism"]


def build(say=print):
    names = {}
    for ln in open(PATH, encoding="utf-8", errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3 and f[2] == "Homo sapiens":
            names[f[0]] = f[1]
    say(f"  {len(names):,} human Reactome pathways")

    kids = defaultdict(list)
    for ln in open(REL, encoding="utf-8", errors="replace"):
        a, _, b = ln.rstrip("\n").partition("\t")
        if a.startswith("R-HSA") and b.startswith("R-HSA"):
            kids[a].append(b)

    # descendant closure of each root, so a reaction's pathway can be classed by ancestry
    cls_of_pathway = {}
    for root, cls in ROOTS.items():
        stack, seen = [root], set()
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(kids.get(p, ()))
        for p in seen:
            cur = cls_of_pathway.get(p)
            if cur is None or PRIORITY.index(cls) < PRIORITY.index(cur):
                cls_of_pathway[p] = cls
    say(f"  {len(cls_of_pathway):,} pathways classed under {len(set(ROOTS.values()))} roots")

    # PE file: col0 NCBI gene, col1 physical entity, col3 EVENT (the id the repo's steps carry)
    rx_genes = defaultdict(set)
    n = 0
    for ln in open(PE, encoding="utf-8", errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 8 or f[7] != "Homo sapiens":
            continue
        n += 1
        rx_genes[f[3]].add(f[0])
    say(f"  {n:,} human PE rows -> {len(rx_genes):,} distinct events with at least one gene")

    gene_paths = defaultdict(set)
    for ln in open(ALL, encoding="utf-8", errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 6 or f[5] != "Homo sapiens" or not f[1].startswith("R-HSA"):
            continue
        gene_paths[f[0]].add(f[1])
    say(f"  {len(gene_paths):,} genes with all-levels pathway membership")

    def class_of(rid):
        """Most specific class supported by the event's genes, plus the fraction supporting it."""
        gs = rx_genes.get(rid, ())
        if not gs:
            return None, 0.0
        # Every class each gene touches gets a vote, and the MAJORITY wins. An earlier version
        # took each gene's highest-PRIORITY class first, which handed transcription every tie
        # because it sits at priority 0 -- that assigned 6,932 of 15,597 reactions (44%) to
        # transcription, which is not credible for a Reactome dump. PRIORITY now only breaks an
        # exact vote tie.
        votes = Counter()
        ngene = 0
        for g in gs:
            cs = {cls_of_pathway[p] for p in gene_paths.get(g, ()) if p in cls_of_pathway}
            if cs:
                ngene += 1
                for c in cs:
                    votes[c] += 1
        if not votes or not ngene:
            return None, 0.0
        best = max(votes, key=lambda c: (votes[c], -PRIORITY.index(c)))
        return best, votes[best] / ngene

    net = json.load(open(NET))
    steps = [s for s in net["steps"] if s.get("source") == "reactome"]
    say(f"  {len(steps):,} reactome steps in the repo's network")

    rxid, cls_id, nunres, ncat, nrev, support = [], [], [], [], [], []
    ent_name, ent_type = {}, {}
    e_rx_in, e_sp_in, e_rx_out, e_sp_out = [], [], [], []
    cat_rx, cat_gene = [], []
    genes = {}
    CLASSES = ["other"] + PRIORITY + ["unclassifiable"]

    def eid(p):
        k = str(p.get("id"))
        if k not in ent_name:
            ent_name[k] = p.get("name") or k
            ent_type[k] = p.get("type") or "OTHER"
        return k

    ents = {}
    for j, s in enumerate(steps):
        rxid.append(s["id"])
        c, sup = class_of(s["id"])
        cls_id.append(CLASSES.index(c) if c else (CLASSES.index("unclassifiable")
                                                  if s["id"] not in rx_genes else 0))
        support.append(sup)
        nrev.append(1 if s.get("reversible") else 0)
        u = 0
        for side, rxs, sps in (("in", e_rx_in, e_sp_in), ("out", e_rx_out, e_sp_out)):
            for p in s.get(side, ()):
                k = eid(p)
                if k not in ents:
                    ents[k] = len(ents)
                if (p.get("type") or "") == "UNRESOLVED":
                    u += 1
                rxs.append(j)
                sps.append(ents[k])
        nunres.append(u)
        cats = s.get("catalysts") or []
        ncat.append(len(cats))
        for g in cats:
            if g not in genes:
                genes[g] = len(genes)
            cat_rx.append(j)
            cat_gene.append(genes[g])

    cls_id = np.array(cls_id, np.int8)
    say("  process classes:")
    for i, c in enumerate(CLASSES):
        k = int((cls_id == i).sum())
        if k:
            say(f"     {c:<20s} {k:>6,}")
    sup = np.array(support)
    say(f"     support: median {np.median(sup[sup > 0]):.2f}, "
        f"{int((sup >= 0.5).sum()):,} reactions where at least half the genes agree")
    say(f"  {len(ents):,} entities | {len(genes):,} distinct catalyst genes")
    say(f"  reactions with no catalyst listed: {int((np.array(ncat) == 0).sum()):,}")
    say(f"  UNRESOLVED participants: {int(np.sum(nunres)):,} over "
        f"{int((np.array(nunres) > 0).sum()):,} reactions")

    ent_keys = sorted(ents, key=ents.get)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        reactions=np.array(rxid), rx_class=cls_id, classes=np.array(CLASSES),
        reversible=np.array(nrev, np.int8), n_unresolved=np.array(nunres, np.int32),
        class_support=np.array(support, np.float32),
        entities=np.array(ent_keys),
        ent_name=np.array([ent_name[k] for k in ent_keys]),
        ent_type=np.array([ent_type[k] for k in ent_keys]),
        in_rx=np.array(e_rx_in, np.int32), in_ent=np.array(e_sp_in, np.int32),
        out_rx=np.array(e_rx_out, np.int32), out_ent=np.array(e_sp_out, np.int32),
        genes=np.array(sorted(genes, key=genes.get)),
        cat_rx=np.array(cat_rx, np.int32), cat_gene=np.array(cat_gene, np.int32))
    say(f"  -> {OUT}")


if __name__ == "__main__":
    build()
