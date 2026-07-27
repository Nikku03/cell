"""LESION REPORT: for one knockout -- which reactions break, where each sits in its pathway, and what each affected
protein does and acts on.

THE THREE QUESTIONS, ANSWERED IN ORDER, FROM DATA ALREADY BUILT:

  1  WHICH REACTIONS ARE AFFECTED       from the corrected ablation. Two tiers, kept apart because they are not
                                        the same claim: DIRECT (the knockout is the sole catalyst, or a substrate
                                        nothing else makes) and DOWNSTREAM (a reaction that consumed a product the
                                        direct lesion stopped producing). Reactions that merely contain the gene
                                        but survive its removal are NOT listed -- BUFFERED and REDUNDANT are the
                                        whole point of the ablation.

  2  WHICH PATHWAY, AND AT WHICH STEP   pathway by gene overlap, step by position in the pathway's own reaction
                                        DAG. Both are DERIVED, not curated -- see the two caveats below.

  3  WHAT EACH AFFECTED PROTEIN DOES    UniProt function text, subcellular location, and -- separately -- the
                                        substrates it consumes and products it makes, taken from the reactions it
                                        actually catalyses rather than from prose.

TWO THINGS ARE INFERRED AND MUST NOT BE READ AS CURATED:

  PATHWAY ASSIGNMENT. Reactome's reaction-to-pathway table is not in this repo, so a reaction is assigned to the
  pathway that (a) contains at least half its participant genes and (b) is the SMALLEST such pathway. The
  smallest-wins rule matters: without it every reaction lands in "Metabolism" or "Signal Transduction", which is
  true and useless. Coverage is printed, and a reaction with no qualifying pathway is reported as unassigned
  rather than being forced into one.

  STEP NUMBER. There is no curated step index either, so it is computed: within a pathway's own reactions, draw an
  edge from reaction A to reaction B when a non-currency product of A is a substrate of B, then take the longest
  path to each reaction. Step 1 means nothing else in that pathway feeds it. Currency species are excluded or
  every reaction would feed every other through ATP and the depth would be meaningless. Cycles are broken by
  capping the relaxation, and any reaction inside a cycle is marked so, because "step 4 of 9" is a false precision
  when the pathway loops.
"""
import collections
import json
import sys
from pathlib import Path

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAXPART = 60
MINFRAC = 0.5      # a pathway must contain at least this fraction of a reaction's genes to claim it
MAXPWGENES = 600   # pathways bigger than this are containers ("Metabolism", "Immune System"), never a step index


def load():
    net = json.load(open(OUT / "reaction_network.json"))
    steps = {s["id"]: s for s in net["steps"]}
    comp = json.loads((OUT / "compartments.json").read_text())["steps"]
    func = json.loads((OUT / "dark_function_table.json").read_text())["table"]
    # dark_function_table covers the 4,993 DARK genes only, so KARS1 and every other well-studied protein came
    # back as "(no curated function text)". cell_complete.json carries structured fields for all of them; they are
    # not prose, and are labelled as the fields they are rather than dressed up as a function description.
    for g in json.load(open(OUT / "cell_complete.json"))["genes"]:
        r = func.setdefault(g["name"], {})
        r.setdefault("card", {k: g.get(k) for k in ("comp", "proc", "path", "ess", "ess_src", "dep_frac", "pubs")})
    pw = {}
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            pw[p[0]] = set(p[2:])
    return steps, comp, func, pw


def rxn_genes(s):
    gs = set(s["catalysts"])
    for p in s["in"] + s["out"]:
        gs |= set(p["genes"])
    return gs


def assign_pathways(steps, pw):
    """Smallest pathway containing >= MINFRAC of a reaction's genes. Smallest-wins is the load-bearing rule."""
    small = {k: v for k, v in pw.items() if 2 <= len(v) <= MAXPWGENES}
    bygene = collections.defaultdict(list)
    for name, gs in small.items():
        for g in gs:
            bygene[g].append(name)
    out = {}
    for sid, s in steps.items():
        gs = rxn_genes(s)
        if not gs or len(gs) > MAXPART:
            continue
        hit = collections.Counter()
        for g in gs:
            for name in bygene.get(g, ()):
                hit[name] += 1
        # rank by how much of the REACTION the pathway explains, and only then by pathway specificity. Ranking
        # by size alone let a 5-gene pathway that covered half a reaction beat a 60-gene one that covered all of
        # it -- which is how ubiquitin steps came back labelled "Maturation of protein E" and a VHL/HIF step came
        # back as "Inactivation of CSF3 (G-CSF) signaling".
        # AT LEAST TWO genes must be shared. One gene cannot place a reaction: every gene belongs to many
        # pathways, so a single-gene reaction gets whichever tiny pathway happens to contain it -- which is how a
        # methionyl-tRNA charging step came back labelled "Platelet degranulation" and another
        # "Aggregated beta-amyloid interacts with fibrinogen". Single-gene reactions are now left unassigned,
        # which is the truthful answer, at the cost of coverage.
        best = [(-c / len(gs), len(small[n]), n) for n, c in hit.items()
                if c >= 2 and c >= MINFRAC * len(gs)]
        if best:
            out[sid] = sorted(best)[0][2]
    return out


def step_index(steps, rxn2pw):
    """Step position within each pathway, computed on the CONDENSATION of its reaction graph.

    Edge A->B when a non-currency product of A is a substrate of B. Currency is excluded or every reaction feeds
    every other through ATP and the depth is meaningless.

    Cycles are collapsed to a single step rather than unrolled. The first version relaxed longest-paths with an
    iteration cap, which reported things like "step 80 of 82" for a mutually-feeding ubiquitination cluster -- a
    number that looks like a position in a sequence and is really the iteration count where relaxation was cut
    off. Reactions that share a cycle genuinely have NO order between them, so they now share one step index and
    are flagged; the acyclic part of the same pathway keeps a real index.
    """
    bypw = collections.defaultdict(list)
    for sid, name in rxn2pw.items():
        bypw[name].append(sid)
    depth, total, cyclic = {}, {}, set()
    for name, sids in bypw.items():
        makes = collections.defaultdict(set)
        for sid in sids:
            for p in steps[sid]["out"]:
                if not p.get("currency"):
                    makes[p["id"]].add(sid)
        succ = {sid: set() for sid in sids}
        for sid in sids:
            for p in steps[sid]["in"]:
                if not p.get("currency"):
                    for q in makes.get(p["id"], set()) - {sid}:
                        succ[q].add(sid)

        # --- Tarjan SCC (iterative: some pathways are large enough to blow the recursion limit) ---
        idx, low, on, stack, comp_of, comps, counter = {}, {}, set(), [], {}, [], [0]
        for root in sids:
            if root in idx:
                continue
            work = [(root, iter(succ[root]))]
            idx[root] = low[root] = counter[0]; counter[0] += 1
            stack.append(root); on.add(root)
            while work:
                v, it = work[-1]
                adv = False
                for w in it:
                    if w not in idx:
                        idx[w] = low[w] = counter[0]; counter[0] += 1
                        stack.append(w); on.add(w)
                        work.append((w, iter(succ[w])))
                        adv = True
                        break
                    if w in on:
                        low[v] = min(low[v], idx[w])
                if adv:
                    continue
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == idx[v]:
                    c = []
                    while True:
                        w = stack.pop(); on.discard(w); c.append(w)
                        comp_of[w] = len(comps)
                        if w == v:
                            break
                    comps.append(c)
        for c in comps:
            if len(c) > 1:
                cyclic |= set(c)

        # --- longest path over the condensation, which is a DAG, so this terminates exactly ---
        cedge = collections.defaultdict(set)
        for v in sids:
            for w in succ[v]:
                if comp_of[v] != comp_of[w]:
                    cedge[comp_of[v]].add(comp_of[w])
        indeg = collections.Counter()
        for a, bs in cedge.items():
            for b in bs:
                indeg[b] += 1
        d = {i: 1 for i in range(len(comps))}
        q = [i for i in range(len(comps)) if not indeg[i]]
        while q:
            a = q.pop()
            for b in cedge.get(a, ()):
                d[b] = max(d[b], d[a] + 1)
                indeg[b] -= 1
                if not indeg[b]:
                    q.append(b)
        mx = max(d.values()) if d else 1
        for sid in sids:
            depth[sid], total[sid] = d[comp_of[sid]], mx
    return depth, total, cyclic


def tier(p):
    """How much to trust "removing this gene removes this entity".

    THE UNRESOLVED LIMITATION. Entities were flattened to a gene list when the network was built, so a COMPLEX whose
    own components are DefinedSets -- a nucleosome, whose H2A slot accepts any of a dozen paralogs -- looks like a
    complex in which every one of those paralogs is required. That is why H2AX comes out as the widest cascade in
    the network after the DefinedSet fix, exactly as RPS27A did before it: ubiquitin and core histones are the two
    places where a large paralog family hides inside a complex.

    Distinguishing them properly needs the entity tree, which is not in this repo's network file. Rather than
    guess, the claim is tiered by how much flattening could have hidden:
        HIGH    a sole catalyst, or a single-protein substrate -- one gene, one entity, nothing to flatten
        MEDIUM  a small complex (<=4 genes) -- plausibly all required
        LOW     a large complex (>4 genes) -- a paralog family may be masquerading as required components
    """
    n = len(p.get("genes") or [])
    if p["type"] == "PROTEIN" or n == 1:
        return "HIGH"
    return "MEDIUM" if n <= 4 else "LOW"


# AMBIGUOUS: the gene is one of several annotated catalysts, and this representation cannot say whether those are
# ALTERNATIVE ENZYMES (removing one leaves the reaction running) or SUBUNITS OF ONE ENZYME (removing one stops it).
# PMPCA/PMPCB are the mitochondrial processing peptidase -- an obligate heterodimer -- and were being silently
# discarded as "redundant", which is why PMPCB, a knockout whose mitochondrial phenotype this project predicts
# correctly by other means, came back with no lesion at all.
#
# A PPI TEST FOR THIS WAS TRIED AND REJECTED. The idea: two catalysts that physically interact are subunits, two
# that do not are alternatives. Checked against known cases first, and it fails in both directions -- CYP2C8/CYP2C9,
# UGT1A1/UGT1A4 and MAFF/MAFG (all genuine alternatives, the last one literally a Reactome DefinedSet) all carry
# PPI edges, while CUL1/SKP1, an obligate complex, carries none. Paralogues attract spurious interactions, so the
# signal is anti-correlated with what is needed. Recorded so the same idea is not retried.
#
# What WOULD resolve it: the schemaClass of the CatalystActivity's physicalEntity (Complex vs DefinedSet), one
# field per reaction from Reactome. 5,556 reactions have more than one catalyst -- 19.5% of the network.


def consequences(steps):
    """Direct lesions and downstream starvation per gene, using the corrected currency + addressability rules."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reaction_ablation import CURRENCY, norm

    def iscur(p):
        return norm(p["name"]) in CURRENCY

    def keys(p):
        k = [p["id"]]
        if p["type"] in ("METABOLITE", "DRUG", "OTHER"):
            k.append("nm:" + norm(p["name"]))
        return k

    produced_by, consumed_by = collections.defaultdict(set), collections.defaultdict(set)
    for sid, s in steps.items():
        for p in s["out"]:
            if not iscur(p):
                for k in keys(p):
                    produced_by[k].add(sid)
        for p in s["in"]:
            if not iscur(p):
                for k in keys(p):
                    consumed_by[k].add(sid)

    def others(p, sid, tbl):
        g = set()
        for k in keys(p):
            g |= tbl.get(k, set())
        return g - {sid}

    con = collections.defaultdict(lambda: {"direct": {}, "downstream": {}})
    for sid, s in steps.items():
        gs = rxn_genes(s)
        if len(gs) > MAXPART:
            continue
        cats = list(s["catalysts"])
        breakers = {}
        if len(cats) == 1:
            breakers[cats[0]] = ("sole catalyst", "HIGH")
        else:
            for g in cats:
                breakers[g] = (f"one of {len(cats)} annotated catalysts", "AMBIGUOUS")
        for p in s["in"]:
            if iscur(p) or not p["genes"] or others(p, sid, produced_by):
                continue
            if p["type"] == "SET" and len(p["genes"]) > 1:
                continue      # DefinedSet members are ALTERNATIVES: removing one leaves the entity intact
            for g in p["genes"]:
                breakers.setdefault(g, (f"only source of {p['name'][:34]}", tier(p)))
        if not breakers:
            continue
        lost = [q for q in s["out"] if not iscur(q) and not others(q, sid, produced_by)]
        starved = {}
        for q in lost:
            for t in others(q, sid, consumed_by):
                starved.setdefault(t, q["name"])
        for g, (why, tr) in breakers.items():
            con[g]["direct"][sid] = {"why": why, "tier": tr, "lost": [q["name"] for q in lost]}
            for t, viaq in starved.items():
                con[g]["downstream"].setdefault(t, viaq)
    return con


COMPACT = {"cytosol": "c", "nucleus": "n", "mitochondrion": "m", "endoplasmic reticulum": "r", "golgi": "g",
           "lysosome": "l", "peroxisome": "x", "extracellular": "e", "plasma membrane": "pm", "endosome": "en"}


def eq(s, w=30):
    """Participant names carry a compartment tag. Without it an antiport reads `glycine + valine -> glycine +
    valine`, because the only thing that changed is which side of the membrane each is on."""
    def tag(p):
        c = (p.get("compartment") or "").lower()
        c = COMPACT.get(c, c[:3])
        return f"{p['name'][:w]}[{c}]" if c else p["name"][:w]

    def side(xs):
        v = [tag(p) for p in xs if not p.get("currency")]
        return " + ".join(v[:4]) or "(none)"
    arrow = f"--[{','.join(s['catalysts'])[:26]}]-->" if s["catalysts"] else "-->"
    return f"{side(s['in'])}  {arrow}  {side(s['out'])}"


def report(gene, steps, comp, func, rxn2pw, depth, total, cyclic, con, acts, innet):
    c = con.get(gene)
    print("\n" + "=" * 108)
    print(f"KNOCKOUT  {gene}")
    f = func.get(gene, {})
    if f.get("function"):
        print(f"  {f.get('name') or ''}")
        print(f"  function: {f['function'][:200]}")
    print("=" * 108)
    if not c or not c["direct"]:
        # These are two different findings and an earlier version reported both with the same sentence, which
        # asserted buffering for genes that are simply not in the reaction network. PMPCB appears in zero
        # reactions -- Reactome annotates the mitochondrial processing peptidase through PMPCA only.
        if gene not in innet:
            print("  This gene appears in NO reaction in the network -- not as a catalyst, not as a participant.")
            print("  Nothing can be said about it from this view; the absence is in the annotation, not the cell.")
        else:
            print("  The gene is in the network, but no reaction stops when it is removed: every reaction it takes")
            print("  part in has a second catalyst, or a second producer for the substrate it supplies.")
        return
    if all(d["tier"] == "AMBIGUOUS" for d in c["direct"].values()):
        print("  Every lesion below is AMBIGUOUS: this gene is always one of several annotated catalysts, and the")
        print("  network does not record whether those are alternative enzymes or subunits of one enzyme.")

    tn = collections.Counter(d["tier"] for d in c["direct"].values())
    print(f"\n1. REACTIONS AFFECTED  --  {len(c['direct'])} stop directly, "
          f"{len(c['downstream'])} starve downstream")
    print(f"   confidence that the knockout really removes the required entity: "
          f"{dict(tn.most_common())}\n")
    print(f"   {'reaction':<16} {'step':>10}  {'pathway':<46} why")
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "AMBIGUOUS": 3}
    for sid, d in sorted(c["direct"].items(),
                         key=lambda kv: (order[kv[1]["tier"]], -len(kv[1]["lost"])))[:14]:
        name = rxn2pw.get(sid)
        st = (f"{depth[sid]}/{total[sid]}" + ("*" if sid in cyclic else "")) if name else "-"
        print(f"   {sid:<16} {st:>10}  {(name or '(unassigned)')[:46]:<46} [{d['tier']}] {d['why']}")
        print(f"      {eq(steps[sid])}")
        if d["lost"]:
            print(f"      nothing else makes: {', '.join(d['lost'][:3])[:88]}")
    if len(c["direct"]) > 14:
        print(f"   ... and {len(c['direct'])-14} more direct lesions")

    if c["downstream"]:
        print(f"\n2. DOWNSTREAM  --  reactions that lose a substrate the lesion stopped making\n")
        print(f"   {'reaction':<16} {'step':>10}  {'pathway':<46} starved of")
        for sid, viaq in sorted(c["downstream"].items())[:12]:
            name = rxn2pw.get(sid)
            st = (f"{depth[sid]}/{total[sid]}" + ("*" if sid in cyclic else "")) if name else "-"
            print(f"   {sid:<16} {st:>10}  {(name or '(unassigned)')[:46]:<46} {viaq[:30]}")
        if len(c["downstream"]) > 12:
            print(f"   ... and {len(c['downstream'])-12} more")

    aff = collections.Counter()
    for sid in list(c["direct"]) + list(c["downstream"]):
        for g in rxn_genes(steps[sid]):
            if g != gene:
                aff[g] += 1
    print(f"\n3. AFFECTED PROTEINS  --  {len(aff)} distinct, ranked by how many affected reactions they are in\n")
    for g, n in aff.most_common(12):
        fg = func.get(g, {})
        fn = (fg.get("function") or "").replace("\n", " ")
        loc = (fg.get("location") or "").replace("SUBCELLULAR LOCATION: ", "")
        print(f"   {g:<12} in {n:>2} affected rxn   {(fg.get('name') or '')[:56]}")
        if fn:
            print(f"      function : {fn[:150]}")
        else:
            cd = fg.get("card") or {}
            bits = [f"{k}={cd[k]}" for k in ("comp", "proc", "path") if cd.get(k)]
            e = f"essential={cd.get('ess')}({cd.get('ess_src')})" if cd.get("ess") else ""
            print(f"      fields   : {'; '.join(bits)[:130] or '(nothing recorded)'}  {e}")
        if loc:
            print(f"      located  : {loc[:100]}")
        a = acts.get(g)
        if a:
            print(f"      acts on  : {', '.join(sorted(a['sub'])[:4])[:70]}  ->  "
                  f"{', '.join(sorted(a['prod'])[:4])[:70]}")
    if len(aff) > 12:
        print(f"   ... and {len(aff)-12} more")


def main():
    steps, comp, func, pw = load()
    rxn2pw = assign_pathways(steps, pw)
    depth, total, cyclic = step_index(steps, rxn2pw)
    con = consequences(steps)
    # what each protein acts on, from the reactions it CATALYSES -- measured substrate/product, not prose
    acts = collections.defaultdict(lambda: {"sub": set(), "prod": set()})
    for sid, s in steps.items():
        for g in s["catalysts"]:
            for p in s["in"]:
                if not p.get("currency"):
                    acts[g]["sub"].add(p["name"][:30])
            for p in s["out"]:
                if not p.get("currency"):
                    acts[g]["prod"].add(p["name"][:30])

    print(f"reaction -> pathway assigned for {len(rxn2pw):,}/{len(steps):,} reactions "
          f"({len(rxn2pw)/len(steps):.1%}); {len(set(rxn2pw.values())):,} distinct pathways used")
    dd = collections.Counter(total[s] for s in rxn2pw)
    print(f"  pathway lengths: median {sorted(total[s] for s in rxn2pw)[len(rxn2pw)//2]} steps, "
          f"single-step pathways {dd.get(1,0):,}; reactions inside a cycle {len(cyclic & set(rxn2pw)):,}")
    tt = collections.Counter(d["tier"] for v in con.values() for d in v["direct"].values())
    hi = sum(1 for v in con.values() if any(d["tier"] == "HIGH" for d in v["direct"].values()))
    print(f"  genes with at least one direct lesion: {sum(1 for v in con.values() if v['direct']):,} "
          f"(with a HIGH-confidence one: {hi:,})")
    print(f"  direct lesions by confidence: {dict(tt.most_common())}")

    want = sys.argv[1:]
    if not want:
        # default: the widest cascade, a sealed-cohort mitochondrial case, and one with no lesion at all
        # rank by HIGH-confidence lesions only: ranking on the raw count surfaces whichever gene sits inside the
        # biggest flattened paralog family, which is an artefact of the entity representation, not a cascade
        widest = max(con, key=lambda g: sum(1 for d in con[g]["direct"].values() if d["tier"] == "HIGH")
                     + len(con[g]["downstream"]) * 0.01)
        want = [widest, "PMPCB", "MARS1"]
    innet = set()
    for s_ in steps.values():
        innet |= rxn_genes(s_)
    print(f"  genes appearing in at least one reaction: {len(innet):,}")
    for g in want:
        report(g, steps, comp, func, rxn2pw, depth, total, cyclic, con, acts, innet)

    dump = {g: {"direct": {k: v for k, v in c["direct"].items()},
                "downstream": c["downstream"],
                "n_direct": len(c["direct"]), "n_downstream": len(c["downstream"])}
            for g, c in con.items()}
    json.dump({"n_genes": len(dump), "n_rxn_with_pathway": len(rxn2pw),
               "rxn_pathway": rxn2pw, "rxn_step": {k: [depth[k], total[k]] for k in rxn2pw},
               "cyclic": sorted(cyclic & set(rxn2pw)), "lesions": dump},
              open(OUT / "lesion_report.json", "w"))
    print(f"\n  -> {OUT/'lesion_report.json'} "
          f"({(OUT/'lesion_report.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
