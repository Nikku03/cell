"""ORIENTING PPI EDGES FROM WITHIN THE REACTION: why input->output fails and catalyst->output is the right relation.

THE USER'S REASONING, WHICH IS CORRECT AND SHARPER THAN WHAT WAS BUILT. build_directed_network.py orients {A,B} by a
GLOBAL question -- does some reaction containing A precede some reaction containing B -- and A and B may interact in
a completely different reaction. The local question is better: find the reaction where A and B ACTUALLY interact and
read the arrow off that reaction's own curated structure.

MEASURED FIRST, BEFORE BUILDING ANYTHING, and it reshapes the whole idea:

    PPI edges whose endpoints share a reaction   36,415  (19.02% of the interactome)  <- good reach
    ...that SPLIT across input and output            95  (0.3% of those)              <- almost nothing
    ...that are BOTH INPUTS                       1,593
    ...that are BOTH OUTPUTS                        458

WITHIN A REACTION, AN INTERACTION HAS NO DIRECTION. That is not a data gap, it is what a binding event is: the
reaction is A + B -> AB, so A and B are both INPUTS and they are peers. Asking which of two binding partners comes
"first" inside the step where they bind is asking a question the step does not answer. Reactome says so in its own
vocabulary -- reactions carry a `category` field, and the value for these is literally "binding". This module counts
that breakdown rather than asserting it.

SO THE RELATION THAT CARRIES DIRECTION IS CATALYSIS, NOT CONSUMPTION. In a phosphorylation reaction Reactome
records input = {substrate, ATP}, output = {phospho-substrate, ADP}, and the KINASE as a CATALYST -- not an input.
An input->output rule therefore yields substrate -> phospho-substrate, which is the same protein, and never recovers
kinase -> substrate, which is exactly the arrow SIGNOR records. Catalyst -> output is the relation that matches.

THREE THINGS ABOUT THE DATA HAD TO BE ESTABLISHED BY PROBING, AND EACH ONE WOULD HAVE SILENTLY WRECKED THE RESULT.

1. THE BULK ENDPOINT DOES CARRY catalystActivity -- an earlier version of this module concluded it did not. The probe
   used R-HSA-109639, which is a `binding` reaction and genuinely HAS no catalyst, so `catalystActivity: null` was
   read as "the field is never returned". Checking a reaction that actually has a catalyst (R-HSA-8848436, "PTK6
   phosphorylates CDKN1B") shows the field present in the bulk response. The lesson is that a null on one record is
   not evidence about a schema.

2. THE CATALYST IS USUALLY THE WHOLE ENZYME:SUBSTRATE COMPLEX, NOT THE ENZYME. For R-HSA-8848436 the
   catalystActivity's physicalEntity is `p-Y342-PTK6:CDKN1B:(CDK4:CCND1,(CDK2:CCNE1))` -- which CONTAINS the
   substrate CDKN1B. Expanding it and firing catalyst->output would emit PTK6->CDKN1B and CDKN1B->PTK6 together, so
   the edge would come out ambiguous and be discarded: the signal cancels itself. Reactome solves this with
   `activeUnit`, which here is exactly `p-Y342-PTK6`. Preferring activeUnit over physicalEntity is what makes the
   rule mean "the enzyme acted on the product" instead of "these things were in a complex together".
   This also kills the filter an earlier draft had -- "require that the catalyst is not itself an input". In this
   canonical idiom the catalyst complex IS an input, so that filter would have thrown away the true positives.

3. THE EXISTING sym2pe.json IS THE WRONG MAP AND WOULD HAVE FAILED QUIETLY. Its keys are Reactome DISPLAY NAMES, not
   gene symbols -- `p-Y342-PTK6` is a key, distinct from `PTK6` -- so every modified protein form would have missed
   the PPI gene index. And no entity in it maps to more than one gene, meaning complexes are never expanded, so any
   complex input or output contributes nothing. This module builds its own map by recursively expanding
   `hasComponent`/`hasMember` down to EntityWithAccessionedSequence and reading `referenceEntity.geneName`, which is
   the authoritative UniProt symbol.

WHAT IS MEASURED, against independently curated SIGNOR, with the degree rule computed ON THE SAME EDGES as the
control -- because a random walk's direction on an undirected graph is provably the degree ratio, and an annotation
could be selecting edges where degree already happens to work.
"""
import collections
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
UA = {"User-Agent": "Mozilla/5.0 (compatible; cell-network-research/1.0)"}
BATCH = 20              # measured silent cap on the bulk endpoint: posting 25/50/150 ids all return 20
MAXDEPTH = 8            # complex nesting is a few levels deep; this is a runaway guard, not a real limit
CACHE = Path(SP) / "catalyst_arrows_cache.json"     # NOT catalysts.json -- another process writes that name
P1 = Path(SP) / "car_pass1_reactions.json"          # per-pass caches so a crash late in the fetch is cheap
P2 = Path(SP) / "car_pass2_catalysts.json"
PAUSE = 1.2             # deliberate inter-request delay. Reactome 429s under sustained load, and a fetch that
                        # trips the limiter spends longer in backoff than the delay ever costs.


def post(ids, tries=8):
    """POST a batch, treating HTTP 429 as "slow down" rather than as a failure to shrug off.

    THE BUG THIS REPLACES WAS THE DANGEROUS KIND. The old version retried four times on 1-8s backoff and then
    returned None, and the caller did `if not d: continue` -- so a rate-limited batch was skipped SILENTLY. Reactome
    does start returning 429 under sustained load (confirmed: a direct probe during a long fetch returned 429), so
    the fetch would "finish" with an unknown fraction of reactions simply absent, and every downstream count would be
    confidently wrong with nothing to indicate it. A crash would have been better. Now 429 gets a long,
    Retry-After-aware backoff, and any batch that still fails is counted and surfaced by the caller.
    """
    for t in range(tries):
        try:
            req = urllib.request.Request("https://reactome.org/ContentService/data/query/ids",
                                         data=",".join(str(i) for i in ids).encode(),
                                         headers=dict(UA, **{"Content-Type": "text/plain"}))
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After") or 0) or min(120, 15 * (t + 1)))
                continue
            if t == tries - 1:
                return None
            time.sleep(2 ** t)
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(2 ** t)
    return None


def ref(x):
    """Key for an entity reference. Reactome's JSON is not uniform: the same field comes back sometimes as a nested
    object with a stId, sometimes as a bare integer dbId. Both are accepted as ids by the bulk endpoint, so both are
    usable as keys -- as long as the entity map is later keyed by BOTH, which fetch() does."""
    if isinstance(x, dict):
        return x.get("stId") or (str(x["dbId"]) if x.get("dbId") else None)
    if isinstance(x, int):
        return str(x)
    return None


def bulk(ids, handle, label, ckpt=None, state=None, done=None, on_save=None):
    """POST ids in batches of 20, calling handle(obj) on each returned object.

    CHECKPOINTED, BECAUSE THE HOST THROTTLES AND THE FETCH IS LONGER THAN THE PATIENCE OF ANY ONE RUN. Reactome
    starts returning 429 under sustained load, which drops throughput to ~0.2 req/s -- an hour for one pass. Without
    checkpointing every interruption restarts from zero, which is how a fetch never finishes. Ids already recorded in
    `done` are skipped, and progress is flushed to `ckpt` periodically, so re-running converges on a complete fetch
    across as many restarts as it takes.

    Returns the number of batches that failed outright. The caller refuses to declare a pass complete if any failed:
    a fetch that quietly drops batches is worse than one that crashes, because the numbers look finished.
    """
    ids = [i for i in ids if not done or str(i) not in done]
    t0 = time.time()
    failed = 0
    for i in range(0, len(ids), BATCH):
        d = post(ids[i:i + BATCH])
        if not d:
            failed += 1
            continue
        for o in d:
            handle(o)
        if PAUSE:
            time.sleep(PAUSE)
        if (i // BATCH) % 50 == 0 and i:
            if on_save is not None:
                on_save()
            elif ckpt is not None and state is not None:
                ckpt.write_text(json.dumps(state))
        if (i // BATCH) % 100 == 0:
            el = time.time() - t0
            rate = (i // BATCH + 1) / max(el, 1)
            left = (len(ids) - i) / BATCH / max(rate, 1e-6) / 60
            print(f"    {label} {i//BATCH}/{len(ids)//BATCH}  "
                  f"({rate:.2f} req/s, {failed} failed, ~{left:.0f} min left)", flush=True)
    if on_save is not None:
        on_save()
    elif ckpt is not None and state is not None:
        ckpt.write_text(json.dumps(state))
    if failed:
        print(f"    !! {label}: {failed} batches FAILED after retries -- pass is INCOMPLETE, re-run to resume",
              flush=True)
    return failed


def fetch():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    cls = json.loads((Path(SP) / "reaction_io.json").read_text())["className"]
    rxn = sorted(k for k, v in cls.items() if v == "Reaction")

    # pass 1 -- reactions: category, inputs, outputs, catalystActivity dbIds
    print(f"pass 1: {len(rxn):,} reactions -> category + catalystActivity", flush=True)
    rx = {}

    def h1(o):
        sid = o.get("stId")
        if not sid:
            return
        rx[sid] = {
            "cat": o.get("category"),
            "in": [k for k in (ref(e) for e in (o.get("input") or [])) if k],
            "out": [k for k in (ref(e) for e in (o.get("output") or [])) if k],
            "ca": [c["dbId"] if isinstance(c, dict) else c
                   for c in (o.get("catalystActivity") or [])
                   if isinstance(c, int) or (isinstance(c, dict) and c.get("dbId"))],
        }
    # EACH PASS IS CACHED SEPARATELY. Pass 1 is ~780 requests and 13 minutes; a crash in pass 2 or 3 used to throw
    # all of it away, which is how a schema surprise turns into an hour of refetching.
    if P1.exists():
        rx = json.loads(P1.read_text())
        print(f"  resuming from {P1.name}: {len(rx):,}/{len(rxn):,} reactions already fetched", flush=True)
    if len(rx) < len(rxn):
        if bulk(rxn, h1, "rxn", ckpt=P1, state=rx, done=set(rx)):
            P1.write_text(json.dumps(rx))
            raise SystemExit(f"pass 1 incomplete ({len(rx):,}/{len(rxn):,}) -- progress saved. Re-run to resume; "
                             "already-fetched ids are skipped.")
        P1.write_text(json.dumps(rx))
    ncat = sum(1 for v in rx.values() if v["ca"])
    print(f"  {len(rx):,} reactions retrieved, {ncat:,} with a catalyst", flush=True)

    # pass 2 -- CatalystActivity -> physicalEntity and (preferably) activeUnit
    allca = sorted({c for v in rx.values() for c in v["ca"]})
    print(f"pass 2: {len(allca):,} CatalystActivity objects -> physicalEntity / activeUnit", flush=True)
    ca = {}

    def h2(o):
        if not o.get("dbId"):
            return
        au = [k for k in (ref(u) for u in (o.get("activeUnit") or [])) if k]
        ca[str(o["dbId"])] = {"pe": ref(o.get("physicalEntity")), "au": au}
    if P2.exists():
        ca = json.loads(P2.read_text())
        print(f"  resuming from {P2.name}: {len(ca):,}/{len(allca):,} activities already fetched", flush=True)
    if len(ca) < len(allca):
        if bulk(allca, h2, "ca", ckpt=P2, state=ca, done=set(ca)):
            P2.write_text(json.dumps(ca))
            raise SystemExit(f"pass 2 incomplete ({len(ca):,}/{len(allca):,}) -- progress saved, re-run to resume.")
        P2.write_text(json.dumps(ca))
    nau = sum(1 for v in ca.values() if v["au"])
    print(f"  {len(ca):,} resolved, {nau:,} have an explicit activeUnit", flush=True)

    # pass 3 -- recursively expand every physical entity to gene symbols
    need = set()
    for v in rx.values():
        need |= set(v["in"]) | set(v["out"])
    for v in ca.values():
        if v["pe"]:
            need.add(v["pe"])
        need |= set(v["au"])
    # pass 3 checkpoint holds the union of what has been expanded so far, keyed by entity id
    P3 = Path(SP) / "car_pass3_entities.json"
    ETREE = Path(SP) / "entity_tree.json"
    if P3.exists():
        _p3 = json.loads(P3.read_text())
        gene, comp, seen = _p3["gene"], _p3["comp"], set(_p3["seen"])
        print(f"  resuming from {P3.name}: {len(seen):,} entities already expanded", flush=True)
    elif ETREE.exists():
        # SEED FROM THE EXISTING ENTITY TREE INSTEAD OF REFETCHING IT. entity_tree.json holds 54,310 entities with
        # their components and UniProt refs -- strictly more than this pass would retrieve, from the same endpoint.
        # Re-fetching them under a 429-throttled ~0.3 req/s would cost roughly an hour to obtain data already on
        # disk. Whatever it does not cover still gets fetched below, so this is a head start, not a substitution.
        _t = json.loads(ETREE.read_text())
        _u2g = {}
        for _line in (Path(SP) / "reactome/up2gene.tsv").read_text().splitlines():
            _p = _line.split("\t")
            if len(_p) >= 2 and _p[0] and _p[1]:
                _u2g[_p[0]] = _p[1]
        gene, comp, seen = {}, {}, set(_t)
        for k, d in _t.items():
            g = _u2g.get(d.get("up"))
            if g:
                gene[k] = [g]
            if d.get("kids"):
                comp[k] = d["kids"]
        print(f"  seeded pass 3 from {ETREE.name}: {len(seen):,} entities, {len(gene):,} gene-mapped, "
              f"{len(comp):,} with components", flush=True)
    else:
        gene, comp, seen = {}, {}, set()
    depth = 0
    while need and depth < MAXDEPTH:
        todo = sorted(need - seen)
        if not todo:
            break
        # `seen` IS UPDATED FROM WHAT ACTUALLY ARRIVED, not from what was requested. Marking the whole batch seen
        # up front is correct only if the fetch never fails -- and under 429 throttling it does, so a resume would
        # skip entities that were never retrieved and silently leave holes in the complex expansion.
        print(f"pass 3.{depth}: expanding {len(todo):,} entities", flush=True)
        nxt = set()     # accumulate with .update(), never `|=` -- augmented assignment inside the nested
                        # handler would make the name local to it and raise UnboundLocalError

        def h3(o):
            # KEY BY BOTH stId AND dbId. Reactome refers to the same entity by either, depending on the field, so a
            # map keyed only by stId silently misses every reference that arrived as a bare integer.
            keys = [k for k in (o.get("stId"), str(o["dbId"]) if o.get("dbId") else None) if k]
            if not keys:
                return
            seen.update(keys)       # retrieved, therefore genuinely done
            re_ = o.get("referenceEntity")
            gn = (re_.get("geneName") if isinstance(re_, dict) else None) or []
            if gn:
                for k in keys:
                    gene[k] = gn if isinstance(gn, list) else [gn]
            # EntitySet hasCandidate is deliberately NOT followed: candidates are Reactome's explicit
            # "might be this one" annotation, and expanding them would manufacture arrows from uncertainty.
            kids = []
            for f in ("hasComponent", "hasMember"):
                for c in (o.get(f) or []):
                    k = ref(c)
                    if k:
                        kids.append(k)
            if kids:
                for k in keys:
                    comp[k] = kids
                nxt.update(kids)
        st = {"gene": gene, "comp": comp, "seen": []}

        def save():
            st["seen"] = sorted(seen)
            P3.write_text(json.dumps(st))

        failed = bulk(todo, h3, f"pe{depth}", ckpt=P3, state=st, done=seen, on_save=save)
        save()
        if failed:
            raise SystemExit(f"pass 3.{depth} incomplete ({len(seen):,} entities expanded) -- progress saved, "
                             "re-run to resume.")
        need = nxt
        depth += 1

    out = {"rx": rx, "ca": ca, "gene": gene, "comp": comp}
    CACHE.write_text(json.dumps(out))
    print(f"  cached -> {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)", flush=True)
    return out


def build_pe2sym(gene, comp):
    """PE stId -> set of gene symbols, resolving complexes/sets through their components."""
    memo = {}

    def resolve(pe, stack=()):
        if pe in memo:
            return memo[pe]
        if pe in stack:            # cycle guard; Reactome complexes are a DAG but do not assume it
            return set()
        s = set(gene.get(pe, ()))
        for c in comp.get(pe, ()):
            s |= resolve(c, stack + (pe,))
        memo[pe] = s
        return s

    for pe in list(gene) + list(comp):
        resolve(pe)
    return memo


def evaluate(cnt, label, gt, deg, n_ppi):
    orient = {k: v.most_common(1)[0][0] for k, v in cnt.items() if len(v) == 1}
    amb = len(cnt) - len(orient)
    test = [(k, orient[k]) for k in orient if k in gt]
    n = len(test)
    acc = sum(1 for k, d in test if gt[k] == d) / n if n else float("nan")
    se = float(np.sqrt(acc * (1 - acc) / n)) if n else float("nan")
    # MATCHED CONTROL: point from the low-degree protein to the high-degree one, on exactly these edges.
    dacc = (sum(1 for k, _ in test if deg.get(gt[k][0], 0) < deg.get(gt[k][1], 0)) / n) if n else float("nan")
    print(f"\n  {label}")
    print(f"    PPI edges oriented        {len(orient):,}  ({len(orient)/n_ppi:.2%} of the interactome)")
    print(f"    ambiguous (both ways)     {amb:,}")
    print(f"    checkable against SIGNOR  {n:,}")
    if n:
        print(f"    accuracy                  {acc:.4f} +/- {se:.4f}   (chance 0.500)")
        print(f"    MATCHED degree baseline   {dacc:.4f}          <- same edges, degree rule")
        print(f"    delta over degree         {acc - dacc:+.4f}")
    return {"orient": orient, "amb": amb, "n": n, "acc": acc, "se": se, "deg": dacc}


def main():
    F = fetch()
    rx, ca, gene, comp = F["rx"], F["ca"], F["gene"], F["comp"]
    pe2sym = build_pe2sym(gene, comp)
    nmulti = sum(1 for v in pe2sym.values() if len(v) > 1)
    print(f"\n{len(pe2sym):,} physical entities mapped to genes; {nmulti:,} map to >1 gene (complexes/sets)")
    print(f"{sum(1 for v in rx.values() if v['ca']):,} of {len(rx):,} reactions have a catalyst; "
          f"{sum(1 for v in ca.values() if v['au']):,} catalyst activities name an activeUnit", flush=True)

    # ---- PPI edges and the SIGNOR ground truth ----
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    ppi = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            ppi.add((names[min(a, b)], names[max(a, b)]))
    edge_of = collections.defaultdict(set)
    for a, b in ppi:
        edge_of[a].add(b)
        edge_of[b].add(a)
    S = set()
    for e in D.get("sig", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            S.add((names[s_], names[t_]))
    # ground truth = SIGNOR pairs that are PPI edges AND unambiguous (no reciprocal arrow)
    gt = {frozenset(p): p for p in S if (p[1], p[0]) not in S and p[1] in edge_of.get(p[0], ())}
    deg = {g: len(v) for g, v in edge_of.items()}
    print(f"PPI edges {len(ppi):,}; SIGNOR-orientable {len(gt):,}", flush=True)

    # ---- Reactome's own verdict on whether a shared reaction can carry direction ----
    cats = collections.Counter()
    for r, v in rx.items():
        syms = set()
        for p in v["in"] + v["out"]:
            syms |= pe2sym.get(p, set())
        if any(b in edge_of.get(a, ()) for a in syms for b in syms if a != b):
            cats[v["cat"] or "none"] += 1
    print(f"\n  reaction `category` among reactions containing at least one PPI edge: "
          f"{dict(cats.most_common())}")

    # ---- catalyst -> output on PPI edges ----
    # ALL   catalyst entity = activeUnit when Reactome names one, else the whole catalyst physicalEntity
    # AU    only reactions where an activeUnit IS named -- Reactome pinned which subunit acts, so the arrow
    #       means "this enzyme acted", not "these proteins were in a complex together"
    # AUMONO  AU, and the output entity is a single protein rather than a complex. When the output is a complex,
    #       the naive rule fires catalyst -> EVERY member of it, including subunits the enzyme never touched.
    arrows, au_arrows, aumono = (collections.defaultdict(collections.Counter) for _ in range(3))
    n_out_single = n_out_complex = 0
    for r, v in rx.items():
        if not v["ca"]:
            continue
        cat_syms, cat_syms_au = set(), set()
        for c in v["ca"]:
            d = ca.get(str(c))
            if not d:
                continue
            if d["au"]:
                s = set().union(*(pe2sym.get(u, set()) for u in d["au"]))
                cat_syms |= s
                cat_syms_au |= s
            elif d["pe"]:
                cat_syms |= pe2sym.get(d["pe"], set())
        if not cat_syms:
            continue
        for p in v["out"]:
            syms = pe2sym.get(p, set())
            if not syms:
                continue
            single = len(syms) == 1
            n_out_single += single
            n_out_complex += not single
            for a in cat_syms:
                nb = edge_of.get(a)
                if not nb:
                    continue
                for b in syms:
                    if a == b or b not in nb:
                        continue
                    arrows[frozenset((a, b))][(a, b)] += 1
                    if a in cat_syms_au:
                        au_arrows[frozenset((a, b))][(a, b)] += 1
                        if single:
                            aumono[frozenset((a, b))][(a, b)] += 1

    A = evaluate(arrows, "CATALYST -> OUTPUT  (ALL)", gt, deg, len(ppi))
    U = evaluate(au_arrows, "CATALYST -> OUTPUT  (AU: Reactome names the active subunit)", gt, deg, len(ppi))
    M = evaluate(aumono, "CATALYST -> OUTPUT  (AU + output is a single protein)", gt, deg, len(ppi))
    print(f"\n    output entities seen: {n_out_single:,} single-protein, {n_out_complex:,} complex/multi-gene")
    print(f"      reference points: degree/Markov 0.5664 | global precedence 0.6547 | curated-vs-curated 0.783")

    # ---- does it orient DIFFERENT edges than the global precedence rule? ----
    DN = json.load(open(OUT / "directed_network.json"))
    po = {frozenset((a, b)): (a, b) for a, b, p, t in DN["edges"] if p == "pathway_order"}
    oriented = A["orient"]
    both = set(oriented) & set(po)
    agree = sum(1 for k in both if oriented[k] == po[k]) if both else 0
    uniq = len(set(oriented) - set(po))
    print(f"\n    overlap with the global pathway_order rule: {len(both):,} edges"
          + (f", agreeing {agree/len(both):.1%}" if both else ""))
    print(f"    edges catalyst->output orients that pathway_order does NOT: {uniq:,}")

    # THE VERDICT KEYS OFF THE MATCHED CONTROL, not off beating a number measured on a different edge set.
    usable = [(n, r) for n, r in (("ALL", A), ("AU", U), ("AU+MONO", M)) if r["n"] >= 30]
    best_name, best = max(usable, key=lambda x: x[1]["acc"] - 2 * x[1]["se"]) if usable else ("ALL", A)
    beats_deg = bool(usable) and best["acc"] - 2 * best["se"] > best["deg"]
    beats_glob = bool(usable) and best["acc"] - 2 * best["se"] > 0.6547

    verdict = (
        f"WITHIN-REACTION ARROWS: THE PREMISE WAS RIGHT, THE FIRST RULE WAS WRONG, AND THE FIX IS CATALYSIS. "
        f"Orienting {{A,B}} from the reaction where A and B ACTUALLY interact is a better question than the global "
        f"'does some reaction containing A precede some reaction containing B'. But the obvious local rule -- A is "
        f"an input, B is an output -- reaches almost nothing: of the 36,415 PPI edges (19.02%) whose endpoints share "
        f"a reaction, only 95 split across input and output, while 1,593 are BOTH INPUTS. That is not a gap in the "
        f"data, it is what binding IS: the reaction is A + B -> AB, both partners go in, and inside the step where "
        f"two proteins bind, neither comes first. Reactome agrees in its own vocabulary -- the `category` breakdown "
        f"of reactions containing a PPI edge is {dict(cats.most_common(4))}. "
        f"THE RELATION THAT CARRIES DIRECTION IS CATALYSIS. In a phosphorylation Reactome records input = "
        f"{{substrate, ATP}}, output = {{phospho-substrate, ADP}} and the kinase as a CATALYST, so input->output "
        f"yields substrate -> phospho-substrate (the same protein) and never recovers kinase -> substrate, which is "
        f"the arrow SIGNOR actually records. "
        f"TWO CLAIMS IN AN EARLIER DRAFT OF THIS FILE ARE CORRECTED BY ITS OWN NUMBERS. First, preferring "
        f"activeUnit was said to be what makes the rule work; measured, AU scores BELOW ALL "
        f"({U['acc']:.4f} vs {A['acc']:.4f}), so naming the active subunit narrows coverage without improving "
        f"accuracy except when combined with a single-protein output. Second, 0.783 was described as a "
        f"'curated-vs-curated ceiling'; it is not a ceiling and that description is withdrawn -- it was the "
        f"agreement rate between two specific sources over shared edges, where disagreement is dominated by TF "
        f"pairs whose two arrows are both true. "
        f"TWO DATA FACTS HAD TO BE PROBED, AND EITHER WOULD HAVE WRECKED THIS SILENTLY. First, the catalyst entity "
        f"is usually the whole ENZYME:SUBSTRATE complex, so expanding it fires both directions at once and the edge "
        f"cancels to 'ambiguous'; Reactome's `activeUnit` names the subunit that actually acts, and preferring it is "
        f"what makes the rule mean 'the enzyme acted on the product'. Second, the existing sym2pe map is keyed by "
        f"DISPLAY NAME ('p-Y342-PTK6', not 'PTK6') and never expands complexes, so it was rebuilt here by recursing "
        f"through hasComponent/hasMember to UniProt geneName. "
        f"MEASURED: ALL orients {len(A['orient']):,} PPI edges ({len(A['orient'])/len(ppi):.2%}) at "
        f"{A['acc']:.4f} +/- {A['se']:.4f} on n={A['n']:,}; AU {len(U['orient']):,} edges at {U['acc']:.4f} +/- "
        f"{U['se']:.4f} on n={U['n']:,}; AU+MONO {len(M['orient']):,} edges at {M['acc']:.4f} +/- {M['se']:.4f} on "
        f"n={M['n']:,}. "
        f"THE CONTROL THAT DECIDES IT is the degree rule ON THE SAME EDGES, because a random walk's direction on an "
        f"undirected graph is provably the degree ratio and the annotation could simply be selecting edges where "
        f"degree already works: {A['deg']:.4f} / {U['deg']:.4f} / {M['deg']:.4f} respectively. "
        + (f"The best stratum, {best_name}, is {best['acc']:.4f} against a matched {best['deg']:.4f}, so catalysis "
           f"carries direction the graph's own shape does not. " if beats_deg else
           f"The best stratum, {best_name}, is {best['acc']:.4f} against a matched degree baseline of "
           f"{best['deg']:.4f}, and the gap is inside two standard errors -- so on this evidence catalyst->output is "
           f"NOT shown to beat simply pointing from the low-degree protein to the high-degree one. ")
        + (f"It also clears the global precedence rule's 0.6547. " if beats_glob else
           f"It does not clearly clear the global precedence rule's 0.6547 at this sample size. ")
        + f"It orients {uniq:,} edges the global rule does not, so the two are complementary rather than redundant. "
        f"WHAT THIS CANNOT SAY. Reactome describes canonical human biology, not K562. Reactome and SIGNOR curators "
        f"read overlapping literature, so agreement between them is not fully independent validation. And 'the "
        f"direction of a PPI edge' is not single-valued, because protein-level and "
        f"transcriptional direction on the same pair legitimately oppose. Coverage, not accuracy, remains the "
        f"binding constraint.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"strata": {n: {k: v for k, v in r.items() if k != "orient"} | {"n_oriented": len(r["orient"])}
                          for n, r in (("ALL", A), ("AU", U), ("AU+MONO", M))},
               "n_ppi": len(ppi), "n_gt": len(gt), "categories": dict(cats),
               "best": best_name, "beats_degree": beats_deg,
               "overlap_with_pathway_order": len(both), "unique": uniq, "verdict": verdict,
               "edges": [[d[0], d[1]] for d in U["orient"].values()]},
              open(OUT / "catalyst_arrows.json", "w"))
    print(f"\n  -> {OUT/'catalyst_arrows.json'}")


if __name__ == "__main__":
    main()
