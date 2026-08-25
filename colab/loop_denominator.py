"""Loop 200. The denominator: how much of a human cell's chemistry is in Human-GEM at all?

THE QUESTION NINETEEN LOOPS OF METABOLISM NEVER ASKED. Loops 160-172 built a metabolite completer
(hit@1 0.8506) and loops 163b-d built an enzyme assigner (0.8065 held out). Both are scored the same
way: take a reaction FROM HUMAN-GEM, blank a participant, see if it comes back. Loop 195 then scoped
the holes -- 5,149 reactions with no gene, 2,078 enzymes with an EC outside the model -- and every
one of those numbers is also measured INSIDE Human-GEM, against Human-GEM's own contents.

So the whole arc has been answering "given a reaction this model already contains, can we complete
it?" and reporting it as "can we complete a cell's reactions?". Those are the same question only if
Human-GEM is the set of reactions a human cell runs. Nothing in this project has ever tested that,
and Human-GEM is a RECONSTRUCTION -- a curated, incomplete, human-assembled model, not an
observation of a cell.

This loop measures the denominator, or rather the part of it that is measurable.

WHAT CANNOT BE MEASURED, SAID FIRST SO IT IS NOT SMUGGLED IN LATER. The true count of reactions a
human cell performs is not in any database. Every source is curation, and curation is incomplete by
construction -- an uncatalogued reaction looks exactly like a reaction that does not happen. So no
number below is "the fraction of cellular chemistry we have". Each is a fraction against ANOTHER
CURATED SET, and the only honest claim available is a comparison of two incomplete catalogues.

WHY RHEA IS THE RIGHT SECOND CATALOGUE. Rhea is expert-curated reaction chemistry, cross-referenced
to reviewed UniProt entries, and assembled independently of the Recon/Human-GEM reconstruction
lineage. Its human subset -- Rhea reactions annotated to reviewed Homo sapiens entries -- is a set of
reactions human enzymes are curated to catalyse, built without reference to any metabolic model. It
is the closest thing available to a held-out test set for a reconstruction.

WHAT IS DELIBERATELY NOT COUNTED AS INDEPENDENT SUPPORT, declared before any counting. Human-GEM
carries VMH and BiGG identifiers on nearly every reaction. VMH and BiGG are the Recon lineage that
Human-GEM was built from. A VMH identifier is therefore provenance, not corroboration, and counting
it as external evidence would inflate W4 to ~100% by construction. "Outside support" below means
Rhea, KEGG reaction, EC code, or MetaNetX reaction only.

PREDECLARED, BEFORE ANY NUMBER.

  W1 IS THE INSTRUMENT READING THE FILE CORRECTLY?
     Gate: PASS iff the parse returns exactly 12,931 reactions, 8,461 species and 2,848 gene
     products; every reaction has at least one participant; and every participant id resolves to a
     declared species. FAIL means every number below is about a different file than the one the
     rest of this project used, so nothing downstream may be read.

  W2 HOW MUCH DISTINCT CHEMISTRY IS IN THE 12,931?
     Human-GEM carries the same reaction separately in each compartment it occurs in. The count of
     reactions is therefore not the count of chemistries, and this project has quoted 12,931 as if
     it were.
     Predeclared: the 12,931 reactions contain FEWER than 12,931 distinct chemistries once
     compartment tags are stripped.
     CONTROL, because a signature collapse can be an id artifact rather than shared chemistry:
     among collapse groups with two or more EC-annotated members, the members must agree on an EC
     code MORE OFTEN than randomly drawn groups of the same sizes do.
     Gate: PASS iff n_distinct < 12,931 AND the EC-agreement rate exceeds the random rate.
     FAIL on the control means the signature is grouping unrelated reactions and its count is
     meaningless.

  W3 DOES HUMAN-GEM'S OWN CONFIDENCE FIELD SEPARATE ANYTHING?
     Every reaction carries a "Confidence Level" note. If that field is informative, reactions at
     a higher level should carry outside identifiers at a higher rate than level-0 reactions.
     Gate: PASS iff rate(level > 0) > rate(level 0). FAIL means the field is inert and nothing in
     this project may use it as evidence -- which matters because it is the only per-reaction
     evidence annotation the model ships.

  W4 WHAT FRACTION OF THE MODEL HAS SUPPORT FROM OUTSIDE ITS OWN LINEAGE?
     Outside = Rhea, KEGG reaction, EC code, or MetaNetX reaction (see the exclusion above).
     Predeclared bar: PASS iff MORE THAN HALF of the 12,931 carry at least one.
     This can fail in either direction and I am not predicting which.

  W5 THE REVERSE RECALL -- WHAT DOES INDEPENDENT CURATION HAVE THAT THE MODEL LACKS?
     Build the Rhea human set: Rhea master reactions with at least one reviewed human UniProt
     accession. Ask what fraction appear in Human-GEM by Rhea master id.
     Predeclared bar: PASS iff >= 60% are present.
     NEGATIVE CONTROL, declared before the number: the same computation over Rhea master reactions
     that have reviewed accessions but NONE of them human. A human-specific model should recover
     the human set at a HIGHER rate than the non-human-only set. If it does not, the join is
     matching on something other than human biology and W5's number carries no information --
     in that case W5 is VOID, not FAIL, because the test did not run. Magnitude, not sign, decides
     "higher" (weakened_by), so this gate cannot assume the direction of its own answer.

  W6 HOW MANY CATALYTICALLY-ANNOTATED HUMAN ENZYMES ARE OUTSIDE THE MODEL?
     Reviewed human UniProt accessions carrying at least one Rhea reaction, split by whether the
     accession appears among Human-GEM's gene products.
     Predeclared: FEWER are absent than are present.
     FAIL means the model contains a minority of the human enzymes that curation says catalyse a
     known reaction, and every coverage number this project has quoted is against a minority.

  W7 IS EITHER CATALOGUE A SUBSET OF THE OTHER?
     Gate: PASS iff Rhea-human contains reactions Human-GEM lacks AND Human-GEM contains Rhea-
     annotated reactions Rhea-human lacks -- i.e. neither contains the other.
     A PASS here is the finding that BOTH catalogues are incomplete and no denominator is available
     from either. A FAIL would mean one contains the other, and the larger would be the denominator
     -- which would be good news and is what this gate exists to give a chance to happen.

  W8 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, random, re, sys, time
from collections import defaultdict
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates, weakened_by

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEM = os.path.join(ROOT, "HumanGEM.xml")
DATA = os.path.join(ROOT, "colab", "data", "denominator")
RHEA2UP = os.path.join(DATA, "rhea2uniprot_sprot.tsv")
HUMAN = os.path.join(DATA, "human_reviewed.txt")
OUT = os.path.join(ROOT, "outputs", "loop_denominator.json")

SBML = "{http://www.sbml.org/sbml/level3/version1/core}"
FBC = "{http://www.sbml.org/sbml/level3/version1/fbc/version2}"

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def parse_gem(path):
    """Stream the SBML once. Returns species, reactions with participants and xrefs, genes."""
    species, reactions, genes = {}, [], []
    xref = re.compile(r"identifiers\.org/([a-zA-Z0-9._-]+)/([^\"]+)")
    conf = re.compile(r"Confidence Level:\s*([0-9]+)")
    for _, el in ET.iterparse(path, events=("end",)):
        tag = el.tag
        if tag == SBML + "species":
            species[el.get("id")] = el.get("compartment")
            el.clear()
        elif tag == FBC + "geneProduct":
            genes.append({"id": el.get(FBC + "id"), "label": el.get(FBC + "label")})
            el.clear()
        elif tag == SBML + "reaction":
            raw = ET.tostring(el, encoding="unicode")
            subs, prods = [], []
            for side, bag in ((SBML + "listOfReactants", subs), (SBML + "listOfProducts", prods)):
                lst = el.find(side)
                if lst is not None:
                    for sr in lst.findall(SBML + "speciesReference"):
                        bag.append((sr.get("species"), float(sr.get("stoichiometry") or 1)))
            refs = defaultdict(set)
            for ns, val in xref.findall(raw):
                refs[ns].add(val)
            m = conf.search(raw)
            reactions.append({
                "id": el.get("id"),
                "subs": subs, "prods": prods,
                "ec": sorted(refs.get("ec-code", ())),
                "rhea": sorted(refs.get("rhea", ())),
                "kegg": sorted(refs.get("kegg.reaction", ())),
                "mnx": sorted(refs.get("metanetx.reaction", ())),
                "vmh": sorted(refs.get("vmhreaction", ())),
                "bigg": sorted(refs.get("bigg.reaction", ())),
                "uniprot": sorted(refs.get("uniprot", ())),
                "conf": int(m.group(1)) if m else None,
            })
            el.clear()
    return species, reactions, genes


def base_id(sid):
    """Strip the trailing compartment letter from a Human-GEM species id (MAM00001c -> MAM00001)."""
    return sid[:-1] if re.fullmatch(r"MAM\d+[a-z]", sid) else sid


def signature(r):
    """Direction-normalised multiset signature over compartment-stripped participants."""
    s = tuple(sorted((base_id(i), c) for i, c in r["subs"]))
    p = tuple(sorted((base_id(i), c) for i, c in r["prods"]))
    return (s, p) if s <= p else (p, s)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "the denominator"}

    say("=" * 104)
    say("LOOP 200 -- THE DENOMINATOR: how much of a cell's chemistry is in Human-GEM at all?")
    say("=" * 104)

    # ---------------------------------------------------------------- W1
    say("W1 IS THE INSTRUMENT READING THE FILE CORRECTLY?")
    species, rxns, genes = parse_gem(GEM)
    declared = set(species)
    empty = [r["id"] for r in rxns if not r["subs"] and not r["prods"]]
    unresolved = sorted({i for r in rxns for i, _ in r["subs"] + r["prods"] if i not in declared})
    say(f"     reactions {len(rxns):,}   species {len(species):,}   gene products {len(genes):,}")
    say(f"     reactions with no participant  {len(empty)}")
    say(f"     participant ids not declared   {len(unresolved)}")
    ok1 = (len(rxns) == 12931 and len(species) == 8461 and len(genes) == 2848
           and not empty and not unresolved)
    G.add("W1", ok1,
          if_true="W1 PASS -- the parse matches the file this project has been using throughout",
          if_false=lambda: f"W1 FAIL -- {len(rxns)} reactions / {len(species)} species / "
                           f"{len(genes)} genes, {len(empty)} empty, {len(unresolved)} unresolved; "
                           f"nothing below may be read")
    res["counts"] = {"reactions": len(rxns), "species": len(species), "genes": len(genes),
                     "empty": len(empty), "unresolved": len(unresolved)}

    # ---------------------------------------------------------------- W2
    say("W2 HOW MUCH DISTINCT CHEMISTRY IS IN THE 12,931?")
    groups = defaultdict(list)
    for r in rxns:
        groups[signature(r)].append(r)
    n_distinct = len(groups)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    say(f"     reactions                 {len(rxns):,}")
    say(f"     distinct chemistries      {n_distinct:,}")
    say(f"     groups with >1 member     {len(multi):,}   "
        f"covering {sum(len(v) for v in multi.values()):,} reactions")

    ec_groups = [v for v in groups.values() if sum(1 for r in v if r["ec"]) >= 2]
    def agrees(members):
        sets = [set(r["ec"]) for r in members if r["ec"]]
        return bool(set.intersection(*sets)) if len(sets) >= 2 else False
    real_agree = sum(agrees(v) for v in ec_groups)
    real_rate = real_agree / len(ec_groups) if ec_groups else float("nan")

    rng = random.Random(200)
    ec_pool = [r for r in rxns if r["ec"]]
    sizes = [sum(1 for r in v if r["ec"]) for v in ec_groups]
    ctrl_hits, n_draw = 0, 0
    for _ in range(20):
        for k in sizes:
            ctrl_hits += agrees(rng.sample(ec_pool, k)); n_draw += 1
    ctrl_rate = ctrl_hits / n_draw if n_draw else float("nan")
    say(f"     EC-annotated groups of >=2 members   {len(ec_groups):,}")
    say(f"     members share an EC code             real {real_rate:.4f}   "
        f"random-group control {ctrl_rate:.4f}   ({n_draw:,} draws)")
    ok2 = bool(n_distinct < len(rxns) and real_rate > ctrl_rate)
    G.add("W2", ok2, stat=real_rate, requires=("W1",),
          if_true=lambda: f"W2 PASS -- {len(rxns):,} reactions are {n_distinct:,} distinct "
                          f"chemistries ({len(rxns)/max(n_distinct,1):.2f} compartment copies each), "
                          f"and the collapse groups reactions that genuinely share an EC "
                          f"({real_rate:.4f} vs {ctrl_rate:.4f} random)",
          if_false=lambda: f"W2 FAIL -- distinct {n_distinct:,} vs {len(rxns):,}; EC agreement "
                           f"{real_rate:.4f} vs random {ctrl_rate:.4f}")
    res["distinct"] = {"reactions": len(rxns), "distinct": n_distinct,
                       "groups_gt1": len(multi), "ec_groups": len(ec_groups),
                       "ec_agree_real": real_rate, "ec_agree_control": ctrl_rate}

    # ---------------------------------------------------------------- W3/W4
    OUTSIDE = ("rhea", "kegg", "ec", "mnx")
    def has_outside(r):
        return any(r[k] for k in OUTSIDE)

    say("W3 DOES HUMAN-GEM'S OWN CONFIDENCE FIELD SEPARATE ANYTHING?")
    lv = defaultdict(list)
    for r in rxns:
        lv[r["conf"]].append(r)
    for k in sorted(lv, key=lambda x: (x is None, x)):
        v = lv[k]
        say(f"     level {str(k):>4}   n {len(v):>6,}   with outside support "
            f"{sum(map(has_outside, v))/len(v):.4f}")
    lo = lv.get(0, [])
    hi = [r for k, v in lv.items() if k is not None and k > 0 for r in v]
    rate_lo = sum(map(has_outside, lo)) / len(lo) if lo else float("nan")
    rate_hi = sum(map(has_outside, hi)) / len(hi) if hi else float("nan")
    G.add("W3", bool(rate_hi > rate_lo), stat=rate_hi, requires=("W1",),
          if_true=lambda: f"W3 PASS -- the confidence field separates: level>0 {rate_hi:.4f} vs "
                          f"level 0 {rate_lo:.4f}",
          if_false=lambda: f"W3 FAIL -- level>0 {rate_hi:.4f} is not above level 0 {rate_lo:.4f} "
                           f"on {len(hi):,} vs {len(lo):,} reactions. The only per-reaction "
                           f"evidence field the model ships is inert and must not be used as "
                           f"evidence anywhere in this project")
    res["confidence"] = {str(k): {"n": len(v), "outside": sum(map(has_outside, v)) / len(v)}
                         for k, v in lv.items()}

    say("W4 WHAT FRACTION OF THE MODEL HAS SUPPORT FROM OUTSIDE ITS OWN LINEAGE?")
    for k, label in (("rhea", "Rhea"), ("kegg", "KEGG reaction"), ("ec", "EC code"),
                     ("mnx", "MetaNetX"), ("vmh", "VMH (lineage, NOT counted)"),
                     ("bigg", "BiGG (lineage, NOT counted)")):
        say(f"     {label:<28} {sum(1 for r in rxns if r[k]):>6,}   "
            f"{sum(1 for r in rxns if r[k])/len(rxns):.4f}")
    n_out = sum(map(has_outside, rxns))
    frac_out = n_out / len(rxns)
    say(f"     ANY outside identifier       {n_out:>6,}   {frac_out:.4f}")
    G.add("W4", bool(frac_out > 0.5), stat=frac_out, requires=("W1",),
          if_true=lambda: f"W4 PASS -- {frac_out:.1%} of the model carries an identifier from "
                          f"outside its own reconstruction lineage",
          if_false=lambda: f"W4 FAIL -- only {frac_out:.1%} ({n_out:,} of {len(rxns):,}) carries "
                           f"outside support; the rest is corroborated only by the lineage it "
                           f"came from")
    res["outside"] = {"n": n_out, "fraction": frac_out,
                      "by_source": {k: sum(1 for r in rxns if r[k])
                                    for k in ("rhea", "kegg", "ec", "mnx", "vmh", "bigg")}}

    # ---------------------------------------------------------------- W5
    say("W5 THE REVERSE RECALL -- WHAT DOES INDEPENDENT CURATION HAVE THAT THE MODEL LACKS?")
    human_acc = {l.strip() for l in open(HUMAN) if l.strip()}
    master_acc = defaultdict(set)
    with open(RHEA2UP) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                master_acc[p[2]].add(p[3])
    rhea_human = {m for m, a in master_acc.items() if a & human_acc}
    rhea_other = {m for m, a in master_acc.items() if a and not (a & human_acc)}
    gem_rhea = {v for r in rxns for v in r["rhea"]}
    say(f"     Rhea master reactions with a reviewed accession   {len(master_acc):,}")
    say(f"     ...with a reviewed HUMAN accession                {len(rhea_human):,}")
    say(f"     ...reviewed but NO human accession (control)      {len(rhea_other):,}")
    say(f"     distinct Rhea ids carried by Human-GEM            {len(gem_rhea):,}")
    hit_h = len(rhea_human & gem_rhea) / len(rhea_human) if rhea_human else float("nan")
    hit_o = len(rhea_other & gem_rhea) / len(rhea_other) if rhea_other else float("nan")
    say(f"     Human-GEM recovers  human set  {len(rhea_human & gem_rhea):,}/{len(rhea_human):,}"
        f" = {hit_h:.4f}")
    say(f"     Human-GEM recovers  non-human  {len(rhea_other & gem_rhea):,}/{len(rhea_other):,}"
        f" = {hit_o:.4f}   (control)")
    sep = weakened_by(hit_h, hit_o)
    say(f"     control check: |human| {abs(hit_h):.4f} vs |non-human| {abs(hit_o):.4f} -> "
        f"{'SEPARATES' if sep['weakened'] else 'DOES NOT SEPARATE'}")
    G.add("W5", bool(hit_h >= 0.60), stat=hit_h, requires=("W1",),
          void_if=not sep["weakened"],
          void_reason=(f"the non-human control is recovered at {hit_o:.4f} against the human set's "
                       f"{hit_h:.4f}, so the join is not selecting human biology and this recall "
                       f"carries no information"),
          if_true=lambda: f"W5 PASS -- Human-GEM contains {hit_h:.1%} of the independently curated "
                          f"human reaction set",
          if_false=lambda: f"W5 FAIL -- Human-GEM contains {hit_h:.1%} of the "
                           f"{len(rhea_human):,} Rhea reactions curated to reviewed human enzymes. "
                           f"{len(rhea_human - gem_rhea):,} curated human reactions are absent "
                           f"from the model this project has treated as the universe")
    res["reverse_recall"] = {"rhea_master_with_reviewed": len(master_acc),
                             "rhea_human": len(rhea_human), "rhea_nonhuman": len(rhea_other),
                             "gem_rhea_ids": len(gem_rhea),
                             "recovered_human": len(rhea_human & gem_rhea),
                             "recovered_nonhuman": len(rhea_other & gem_rhea),
                             "hit_human": hit_h, "hit_nonhuman": hit_o,
                             "missing_human": len(rhea_human - gem_rhea), "control": sep}

    # ---------------------------------------------------------------- W6
    say("W6 HOW MANY CATALYTICALLY-ANNOTATED HUMAN ENZYMES ARE OUTSIDE THE MODEL?")
    gem_up = {v for r in rxns for v in r["uniprot"]}
    catalytic_human = {a for m, accs in master_acc.items() for a in accs if a in human_acc}
    inside = catalytic_human & gem_up
    outside_e = catalytic_human - gem_up
    say(f"     reviewed human accessions                       {len(human_acc):,}")
    say(f"     ...with at least one curated Rhea reaction      {len(catalytic_human):,}")
    say(f"     ...present among Human-GEM's gene products      {len(inside):,}")
    say(f"     ...ABSENT from Human-GEM                        {len(outside_e):,}")
    G.add("W6", bool(len(outside_e) < len(inside)), stat=float(len(outside_e)), requires=("W1",),
          if_true=lambda: f"W6 PASS -- {len(inside):,} of {len(catalytic_human):,} catalytic human "
                          f"enzymes are in the model and only {len(outside_e):,} are outside it",
          if_false=lambda: f"W6 FAIL -- {len(outside_e):,} of {len(catalytic_human):,} human "
                           f"enzymes with curated reaction chemistry are ABSENT from Human-GEM, "
                           f"against {len(inside):,} present. Every coverage number this project "
                           f"has quoted is measured against a minority of the cell's known enzymes")
    res["enzymes"] = {"human_reviewed": len(human_acc), "catalytic": len(catalytic_human),
                      "in_model": len(inside), "outside_model": len(outside_e),
                      "gem_uniprot": len(gem_up)}

    # ---------------------------------------------------------------- W7
    say("W7 IS EITHER CATALOGUE A SUBSET OF THE OTHER?")
    only_rhea = rhea_human - gem_rhea
    only_gem = gem_rhea - rhea_human
    say(f"     in Rhea-human, not in Human-GEM   {len(only_rhea):,}")
    say(f"     in Human-GEM, not in Rhea-human   {len(only_gem):,}")
    G.add("W7", bool(only_rhea and only_gem), requires=("W1",),
          if_true=lambda: f"W7 PASS -- neither contains the other ({len(only_rhea):,} and "
                          f"{len(only_gem):,} exclusive). Both catalogues are incomplete and "
                          f"NO denominator is available from either of them",
          if_false=lambda: f"W7 FAIL -- one catalogue contains the other "
                           f"({len(only_rhea):,} / {len(only_gem):,} exclusive), so the larger "
                           f"is a usable denominator")
    res["subset"] = {"only_rhea": len(only_rhea), "only_gem": len(only_gem)}

    # ---------------------------------------------------------------- W8
    say("W8 WHAT THIS CANNOT SHOW")
    say("     The true number of reactions a human cell performs is not in any database, and this")
    say("     loop does not produce it. Rhea is curation, so the missing set below is a LOWER")
    say("     bound of unknown tightness -- reactions absent from both catalogues are invisible to")
    say("     this design and are not counted anywhere in it.")
    say("     The Rhea join is by reaction identity, so a reaction Human-GEM contains under")
    say("     different chemistry but no Rhea id counts as missing here. That biases W5 DOWN.")
    say("     Only 1,774 of 12,931 Human-GEM reactions carry a Rhea id at all, so W5 and W7 speak")
    say("     about the Rhea-annotated slice, not about the whole model.")
    say("     Nothing here says the missing reactions matter: an absent reaction may be rare,")
    say("     tissue-specific, or irrelevant to any question this project asks.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
