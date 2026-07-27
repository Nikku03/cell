"""ENTITY LOGIC: decide whether the several genes attached to a reaction are ALTERNATIVES or ALL REQUIRED.

THE GAP THIS CLOSES. The reaction network flattens every entity to a list of gene symbols, which destroys the
distinction between AND and OR. Two consequences ran through everything built on top of it:

  1  A reaction with two annotated catalysts was called REDUNDANT, conflating "two alternative isozymes" with "two
     subunits of one enzyme". PMPCA/PMPCB are the mitochondrial processing peptidase, an obligate heterodimer, and
     were being discarded as redundant. 5,556 reactions (19.5% of the network) are affected.
  2  A COMPLEX whose own components are DefinedSets -- a nucleosome, whose H2A slot accepts any of a dozen
     paralogues -- looks like a complex in which every paralogue is required. 3,957 Reactome complexes carry more
     than four genes and are LOW-confidence for this reason.

Both are the same missing fact: the boolean structure over the genes. Recovered here from two sources that need
completely different treatment.

  HUMAN-GEM (4,408 of the 5,556 multi-catalyst reactions -- 79%): the SBML already carries it. Every reaction has an
  `fbc:geneProductAssociation` holding a nested and/or tree, and it is exact, not inferred. No network access.
  Two traps, both previously hit in this project: `fbc:listOfGeneProducts` appears AFTER `</listOfReactions>`, so
  labels must be resolved in a second pass, and cobra drops `fbc:label` entirely, so the file is parsed directly.

  REACTOME (1,148 reactions + 3,957 complexes): needs the entity tree from the ContentService. `schemaClass` gives
  the operator -- Complex -> AND over hasComponent, DefinedSet/CandidateSet -> OR over hasMember/hasCandidate,
  EntityWithAccessionedSequence -> leaf.

THE TEST IS THE SAME ONE USED EVERYWHERE ELSE IN THIS PROJECT: single-participant ablation. A gene is REQUIRED for a
reaction exactly when setting that gene false, with every other gene true, makes the rule evaluate false. That
definition handles arbitrary nesting for free, needs no special cases for "isozyme" or "subunit", and is the same
question reaction_ablation.py asks of substrates.

WHAT THIS DOES NOT FIX. Reactome's CandidateSet means "one of these, but which is uncertain" -- treated as OR here,
which is the right structural reading but records a real annotation uncertainty rather than resolving it.
"""
import collections
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SBML = SP / "HumanGEM.xml"
CACHE = OUT / "entity_logic_cache.json"
API = "https://reactome.org/ContentService/data/query/ids"
RXCACHE = OUT / "entity_logic_rxn_cache.json"
FBC = "{http://www.sbml.org/sbml/level3/version1/fbc/version2}"
BATCH = 20        # the bulk endpoint SILENTLY caps at 20 results; asking for more loses the remainder without error
# REACTOME REJECTS THE DEFAULT PYTHON USER-AGENT WITH 403. Measured directly: identical POST with no UA -> 403
# Forbidden, with any ordinary UA -> 200 and 8,082 bytes; curl to the same host through the same proxy also returns
# 200, so the host is permitted and this is user-agent filtering, not an egress-policy denial. catalyst_arrows.py
# already sets this header. Omitting it here cost a 30-minute run that retried every batch six times, wrote no
# checkpoint, and produced nothing.
UA = {"User-Agent": "Mozilla/5.0 (compatible; cell-network-research/1.0)"}

AND_CLASSES = {"Complex"}
OR_CLASSES = {"DefinedSet", "CandidateSet", "EntitySet", "OpenSet"}


# ---------------------------------------------------------------------------------------------------------------
# the rule language: ("and"|"or", [children]) or ("gene", symbol)
# ---------------------------------------------------------------------------------------------------------------
def evaluate(rule, false_gene):
    """Is the rule satisfied when `false_gene` is absent and everything else is present?"""
    op = rule[0]
    if op == "gene":
        return rule[1] != false_gene
    vals = [evaluate(c, false_gene) for c in rule[1]]
    if not vals:
        return True
    return all(vals) if op == "and" else any(vals)


def genes_of(rule, acc=None):
    acc = set() if acc is None else acc
    if rule[0] == "gene":
        acc.add(rule[1])
    else:
        for c in rule[1]:
            genes_of(c, acc)
    return acc


def required_genes(rule):
    """Genes whose individual removal breaks the rule. Empty rule -> nothing required."""
    gs = genes_of(rule)
    if not gs or not evaluate(rule, None):
        return set()
    return {g for g in gs if not evaluate(rule, g)}


# ---------------------------------------------------------------------------------------------------------------
# HUMAN-GEM
# ---------------------------------------------------------------------------------------------------------------
def parse_sbml_gpr():
    """reaction id -> rule tree, with gene symbols resolved from fbc:label."""
    label = {}
    rules = {}
    raw = {}
    for ev, el in ET.iterparse(str(SBML), events=("end",)):
        tag = el.tag
        if tag == f"{FBC}geneProduct":
            gid = el.get(f"{FBC}id")
            lab = el.get(f"{FBC}label")
            if gid:
                label[gid] = lab or gid
            el.clear()
        elif tag.endswith("}reaction"):
            rid = el.get("id")
            gpa = el.find(f"{FBC}geneProductAssociation")
            if rid and gpa is not None:
                kids = list(gpa)
                if kids:
                    raw[rid] = _node(kids[0])
            el.clear()
    # SECOND PASS BY NECESSITY: listOfGeneProducts sits after listOfReactions in this file, so labels are not
    # known while the reactions are being read. Resolving inline is what once gave all 12,931 metabolic reactions
    # zero catalysts, silently.
    for rid, r in raw.items():
        rules[rid] = _relabel(r, label)
    return rules


def _node(el):
    t = el.tag
    if t == f"{FBC}geneProductRef":
        return ("gene", el.get(f"{FBC}geneProduct"))
    op = "and" if t == f"{FBC}and" else "or"
    return (op, [_node(c) for c in el])


def _relabel(r, label):
    if r[0] == "gene":
        return ("gene", label.get(r[1], r[1]))
    return (r[0], [_relabel(c, label) for c in r[1]])


# ---------------------------------------------------------------------------------------------------------------
# REACTOME
# ---------------------------------------------------------------------------------------------------------------
def ref(x):
    """Key for an entity reference. Reactome's JSON is NOT uniform: the same field comes back sometimes as a nested
    object with a stId, sometimes as a bare integer dbId. Measured on a 5-record sample of hasComponent: 7 dicts
    and 1 int. This exact non-uniformity was found and fixed in catalyst_arrows.py earlier and not carried over,
    which crashed this module with 'int object has no attribute get'."""
    if isinstance(x, dict):
        return x.get("stId") or (str(x["dbId"]) if x.get("dbId") else None)
    if isinstance(x, int):
        return str(x)
    return None


_seen_err = set()


def post(ids, tries=6):
    """Fetch a batch. Returns [] for "nothing there", None for FAILED -- never conflates the two.

    Errors are PRINTED ONCE per kind rather than silently retried. The previous version caught every exception and
    slept, so a 403 on every single request was indistinguishable from a slow network: the run sat for half an
    hour, retried 1,148 reactions six times each, and wrote nothing. A retry loop that cannot say what it is
    retrying is the same silent-failure pattern this project has now hit four times.
    """
    body = ",".join(ids).encode()
    for t in range(tries):
        try:
            rq = urllib.request.Request(API, data=body,
                                        headers=dict(UA, **{"Content-Type": "text/plain",
                                                            "Accept": "application/json"}))
            with urllib.request.urlopen(rq, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After") or 0) or min(120, 15 * (t + 1)))
                continue
            if e.code in (404, 400):
                return []
            if e.code not in _seen_err:
                _seen_err.add(e.code)
                print(f"    HTTP {e.code} from the ContentService -- reported, not silently retried")
            if e.code in (403, 407):
                return None          # a filter or policy denial: retrying cannot help, so fail fast and loudly
            time.sleep(2 * (t + 1))
        except Exception as e:
            k = type(e).__name__
            if k not in _seen_err:
                _seen_err.add(k)
                print(f"    {k}: {str(e)[:120]}")
            time.sleep(2 * (t + 1))
    return None


def fetch_entities(seed):
    """Walk hasComponent/hasMember/hasCandidate to closure, recording schemaClass and children.

    Failures are counted, not swallowed. An earlier fetch in this project cached an incomplete pass as if complete
    and reported 46% of participants as type OTHER for weeks.
    """
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    need = {s for s in seed if s and s not in cache}
    failed = 0
    rounds = 0
    while need and rounds < 12:
        rounds += 1
        todo = sorted(need)
        print(f"  round {rounds}: {len(todo):,} entities to fetch")
        got_any = set()
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            d = post(chunk)
            if d is None:
                failed += len(chunk)
                continue
            for rec in d:
                sid = rec.get("stId") or str(rec.get("dbId"))
                kids = []
                for f in ("hasComponent", "hasMember", "hasCandidate"):
                    for c in rec.get(f) or []:
                        k = ref(c)
                        if k:
                            kids.append(k)
                rec_out = {"cls": rec.get("schemaClass"), "kids": kids,
                              "gene": (rec.get("referenceEntity") or {}).get("geneName", [None])[0]
                              if isinstance((rec.get("referenceEntity") or {}).get("geneName"), list)
                              else (rec.get("referenceEntity") or {}).get("geneName"),
                              "name": (rec.get("displayName") or "")[:80]}
                # keyed by BOTH ids: children arrive as stIds in some records and bare dbIds in others, so a
                # single-key cache misses on exactly the records that motivated ref()
                cache[sid] = rec_out
                if rec.get("dbId"):
                    cache[str(rec["dbId"])] = rec_out
                got_any.add(sid)
            if i and i % (BATCH * 10) == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"    {i:,}/{len(todo):,} (checkpoint)")
        # anything requested but not returned is marked so it is not retried forever
        for s in todo:
            if s not in cache:
                cache[s] = {"cls": None, "kids": [], "gene": None, "name": "", "unresolved": True}
        json.dump(cache, open(CACHE, "w"))
        need = {k for s in got_any for k in cache[s]["kids"] if k not in cache}
    if failed:
        print(f"  WARNING: {failed:,} entities failed to fetch; results below are incomplete and are labelled so")
    return cache, failed


def entity_rule(sid, cache, depth=0, seen=None):
    """Turn an entity id into an and/or rule over gene symbols.

    DEPTH AND CYCLES. Reactome complexes nest much deeper than first assumed -- the closure over 4,434 seeds
    reached 19,882 records with 11,228 of them Complexes averaging 3.57 children, i.e. complexes inside complexes
    inside complexes. A depth cap of 8 silently truncated those to None, which reads as "no rule" and would have
    dropped exactly the largest assemblies this is meant to resolve. The cap is now 24, and a `seen` set makes the
    walk safe against a self-referential entity rather than relying on the cap to stop it.
    """
    seen = set() if seen is None else seen
    rec = cache.get(sid)
    if not rec or depth > 24 or sid in seen:
        return None
    seen = seen | {sid}
    cls = rec.get("cls")
    if rec.get("gene"):
        return ("gene", rec["gene"])
    kids = [entity_rule(k, cache, depth + 1, seen) for k in rec.get("kids", [])]
    kids = [k for k in kids if k]
    if not kids:
        return None
    if cls in AND_CLASSES:
        return ("and", kids)
    if cls in OR_CLASSES:
        return ("or", kids)
    return ("and", kids) if len(kids) > 1 else kids[0]


def main():
    net = json.load(open(OUT / "reaction_network.json"))
    steps = {s["id"]: s for s in net["steps"]}

    # ---------------- HUMAN-GEM ----------------
    print("HUMAN-GEM: parsing GPR and/or trees from SBML (no network access)")
    gpr = parse_sbml_gpr()
    print(f"  {len(gpr):,} reactions carry a gene-product association")
    hg = {}
    stat = collections.Counter()
    for rid, rule in gpr.items():
        if rid not in steps:
            continue
        gs = genes_of(rule)
        req = required_genes(rule)
        hg[rid] = {"n_genes": len(gs), "required": sorted(req), "n_required": len(req)}
        if len(gs) <= 1:
            stat["single gene"] += 1
        elif not req:
            stat["pure alternatives (all OR)"] += 1
        elif req == gs:
            stat["all required (all AND)"] += 1
        else:
            stat["mixed and/or"] += 1
    print(f"  {len(hg):,} of them are in the network")
    for k, v in stat.most_common():
        print(f"    {k:<28} {v:>7,}")
    multi = [r for r in hg.values() if r["n_genes"] > 1]
    was_redundant = len(multi)
    now_required = sum(1 for r in multi if r["n_required"])
    print(f"  OF THE {was_redundant:,} MULTI-GENE REACTIONS previously all labelled REDUNDANT:")
    print(f"    at least one gene is genuinely REQUIRED : {now_required:,}  ({now_required/max(was_redundant,1):.1%})")
    print(f"    genuinely alternatives                  : {was_redundant-now_required:,}")

    json.dump({"humangem": hg, "stats": dict(stat)}, open(OUT / "entity_logic_humangem.json", "w"))
    print(f"  -> {OUT/'entity_logic_humangem.json'}")

    # ---------------- REACTOME ----------------
    # The flattened network keeps catalysts as gene symbols and drops the entity, so the catalyst entity id has to
    # come back from the reaction record. Only reactions with MORE THAN ONE catalyst gene are ambiguous; a single
    # catalyst is unambiguous whatever its class.
    want_rxn = sorted(sid for sid, s_ in steps.items()
                      if s_["source"] == "reactome" and len(s_["catalysts"]) > 1)
    print(f"\nREACTOME: {len(want_rxn):,} multi-catalyst reactions need their catalyst entity")
    rc = json.loads(RXCACHE.read_text()) if RXCACHE.exists() else {}
    todo = [r for r in want_rxn if r not in rc]
    # HOP 1: reaction -> CatalystActivity dbIds. The bulk endpoint returns a SHALLOW catalystActivity carrying only
    # dbId/displayName/className/schemaClass -- physicalEntity and activeUnit are NOT in it. Reading them straight
    # off the reaction record recovered the catalyst entity for 0 of 1,148 reactions while looking like success.
    ca_of = {}
    for i in range(0, len(todo), BATCH):
        d = post(todo[i:i + BATCH])
        if d is None:
            continue
        for rec in d:
            sid = rec.get("stId") or str(rec.get("dbId"))
            ca_of[sid] = [str(c["dbId"]) for c in (rec.get("catalystActivity") or [])
                          if isinstance(c, dict) and c.get("dbId")]
        if i and i % (BATCH * 10) == 0:
            print(f"    hop1 reactions {i:,}/{len(todo):,}")
    allca = sorted({c for v in ca_of.values() for c in v})
    print(f"  hop 1: {len(allca):,} CatalystActivity records referenced by {len(ca_of):,} reactions")
    # HOP 2: CatalystActivity -> physicalEntity / activeUnit
    ca_ent = {}
    for i in range(0, len(allca), BATCH):
        d = post(allca[i:i + BATCH])
        if d is None:
            continue
        for rec in d:
            cid = str(rec.get("dbId"))
            ents = []
            for f in ("activeUnit", "physicalEntity"):
                v = rec.get(f)
                vs = v if isinstance(v, list) else ([v] if v else [])
                refs = [ref(x) for x in vs]
                refs = [r_ for r_ in refs if r_]
                if refs:
                    ents = refs         # activeUnit is the catalytic part; prefer it over the whole complex
                    break
            ca_ent[cid] = ents
        if i and i % (BATCH * 10) == 0:
            print(f"    hop2 activities {i:,}/{len(allca):,}")
    for sid, cids in ca_of.items():
        rc[sid] = sorted({e for c in cids for e in ca_ent.get(c, [])})
    json.dump(rc, open(RXCACHE, "w"))
    n_res = sum(1 for r in want_rxn if rc.get(r))
    print(f"  catalyst entity recovered for {n_res:,}/{len(want_rxn):,} reactions")

    # entity closure: catalyst entities + every COMPLEX participant carrying >4 genes (the paralogue-inside-complex
    # problem, item 3)
    big = {p["id"] for s_ in steps.values() for p in s_["in"] + s_["out"]
           if p["type"] == "COMPLEX" and len(p["genes"]) > 4}
    seed = {e for r in want_rxn for e in rc.get(r, [])} | big
    print(f"  entity closure seeded with {len(seed):,} ids "
          f"({len(big):,} large complexes + catalyst entities)")
    cache, failed = fetch_entities(seed)
    print(f"  entity cache holds {len(cache):,} records")
    cls = collections.Counter(v.get("cls") for v in cache.values())
    print(f"  schemaClass distribution: {dict(cls.most_common(8))}")

    # ---- ITEM 2: are the multiple catalysts alternatives or subunits? ----
    rx = {}
    verdict = collections.Counter()
    for r in want_rxn:
        ents = rc.get(r) or []
        rules = [entity_rule(e, cache) for e in ents]
        rules = [x for x in rules if x]
        if not rules:
            verdict["unresolved"] += 1
            continue
        rule = rules[0] if len(rules) == 1 else ("or", rules)   # several catalyst activities = alternative routes
        gs = genes_of(rule)
        req = required_genes(rule)
        rx[r] = {"n_genes": len(gs), "required": sorted(req), "n_required": len(req)}
        if not gs:
            verdict["unresolved"] += 1
        elif not req:
            verdict["alternatives (OR)"] += 1
        elif req == gs:
            verdict["all subunits required (AND)"] += 1
        else:
            verdict["mixed"] += 1
    print(f"\n  ITEM 2 -- Reactome multi-catalyst reactions, previously ALL called REDUNDANT:")
    for k, v in verdict.most_common():
        print(f"    {k:<32} {v:>6,}")
    nreq = sum(1 for v in rx.values() if v["n_required"])
    print(f"    at least one catalyst genuinely REQUIRED: {nreq:,}/{max(len(rx),1):,}")

    # ---- ITEM 3: do large complexes hide alternatives? ----
    comp = {}
    hidden = 0
    for e in sorted(big):
        rule = entity_rule(e, cache)
        if not rule:
            continue
        gs = genes_of(rule)
        req = required_genes(rule)
        comp[e] = {"n_genes": len(gs), "n_required": len(req), "required": sorted(req)[:40]}
        if len(gs) > len(req):
            hidden += 1
    print(f"\n  ITEM 3 -- large COMPLEX participants (>4 genes), resolved {len(comp):,}/{len(big):,}:")
    print(f"    contain at least one gene that is NOT required (a set hiding inside): {hidden:,} "
          f"({hidden/max(len(comp),1):.1%})")
    tot_g = sum(v["n_genes"] for v in comp.values())
    tot_r = sum(v["n_required"] for v in comp.values())
    print(f"    gene slots: {tot_g:,} total, {tot_r:,} genuinely required "
          f"({tot_r/max(tot_g,1):.1%}) -- the rest were being treated as single points of failure")

    json.dump({"humangem": hg, "stats": dict(stat), "reactome_rxn": rx, "reactome_verdict": dict(verdict),
               "complexes": comp, "n_failed_fetch": failed},
              open(OUT / "entity_logic.json", "w"))
    print(f"\n  -> {OUT/'entity_logic.json'}")


if __name__ == "__main__":
    main()
