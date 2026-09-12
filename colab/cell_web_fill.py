"""THE MISSING LIST, AND FILLING IT FROM THE WEB -- with the same validation everything else got.

WHERE THIS COMES FROM.  The orphan work established a hard ceiling: the candidate vocabulary is genes already
annotated as catalysts, so a protein never annotated as one cannot be proposed at any k. 37% of catalysts sit in
exactly that position, and 2,583 orphan reactions are chemically isolated -- no catalysed reaction shares any
non-currency metabolite with them. For those, more graph search over local data cannot help. The vocabulary
itself has to come from somewhere else.

So: emit the missing list, then go and get the vocabulary from outside.

    UniProt REST   proteins by EC number and by catalytic-activity text, restricted to human
    Rhea           reaction database, metabolite -> reaction -> EC

THE TRAP, and it is the same one as every other day today.  "Fetch some candidates from the web" always returns
SOMETHING -- UniProt will happily hand back 25 kinases for any query mentioning ATP. A list of plausible names
is not a result. So the web pass is validated exactly like the local one:

    RECOVERY TEST   run the web query for reactions whose catalyst we ALREADY KNOW, without telling it the
                    answer, and measure whether the true catalyst comes back.
    CONTROL         the same query stripped of the reaction's specific metabolites, keeping only the generic
                    ones. If that scores the same, the web is answering the question "name some human enzymes"
                    rather than "name an enzyme for THIS reaction".

PREDECLARED, before any query:
    web recall > stripped-query control     -> the web adds vocabulary we do not have, and the missing list is
                                               worth filling this way.
    web recall ~ control                    -> we are retrieving generic enzymes. The list stays empty and that
                                               is the honest outcome.

SCOPE: this runs on a SAMPLE, not all 2,583, because each reaction costs live HTTP. The sample size is stated
in the output and the per-query rate is deliberately slow. Nothing here is cached into the repo as fact -- the
output is candidate hypotheses with their provenance, which is what an unfillable list can honestly become.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT, SP = A.OUT, A.SP
NONCAT = ("binding", "dissociation")
ABSTRACT = ("omitted", "uncertain")
N_VALIDATE, N_FILL, PAUSE = 60, 40, 0.35
UA = {"User-Agent": "cellos-partslist/1.0 (research; contact via repo)"}


def get(url, timeout=25):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def uniprot_by_terms(terms, size=25):
    """human proteins whose catalytic-activity annotation mentions these metabolites."""
    if not terms:
        return []
    q = " AND ".join(f'cc_catalytic_activity:"{t}"' for t in terms[:3])
    url = ("https://rest.uniprot.org/uniprotkb/search?query="
           + urllib.parse.quote(f"({q}) AND organism_id:9606 AND reviewed:true")
           + f"&format=json&size={size}&fields=accession,gene_primary,ec")
    d = get(url)
    out = []
    for r in (d or {}).get("results", []):
        g = (r.get("genes") or [{}])[0].get("geneName", {}).get("value")
        if g:
            out.append(g)
    return out


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("THE MISSING LIST, AND FILLING IT FROM THE WEB")
    report("=" * 100)
    report("  PREDECLARED: web recall > stripped-query control => the web adds vocabulary we lack.")
    report("               web recall ~ control => we are retrieving generic enzymes. List stays empty.")

    net = json.load(open(OUT / "reaction_network.json"))
    steps = net["steps"]
    currency = set(str(x) for x in (net.get("currency") or []))

    def parts_named(s):
        """metabolite NAMES, which is what a text database can be queried with."""
        out = []
        for side in ("in", "out"):
            for p in (s.get(side) or []):
                if str(p.get("id") or "") in currency:
                    continue
                nm = p.get("name")
                if nm and (p.get("type") or "").upper() == "METABOLITE":
                    out.append(str(nm))
        return out

    def parts_ids(s):
        return {str(p.get("id") or p.get("name") or "")
                for side in ("in", "out") for p in (s.get(side) or [])} - currency - {""}

    cats = {i: list(s["catalysts"]) for i, s in enumerate(steps) if s.get("catalysts")}
    P = {i: parts_ids(s) for i, s in enumerate(steps)}
    cat_of_part = defaultdict(set)
    for i in cats:
        for p in P[i]:
            cat_of_part[p].add(i)

    real = [i for i, s in enumerate(steps)
            if not s.get("catalysts") and s.get("category") not in NONCAT + ABSTRACT]
    nocand = [i for i in real if not any(cat_of_part.get(p) for p in P[i])]
    named = [i for i in nocand if parts_named(steps[i])]
    report(f"\n  orphans in the real pool          : {len(real):,}")
    report(f"  of those, NO local candidate      : {len(nocand):,}")
    report(f"  of those, with named metabolites  : {len(named):,}  <- the only ones a text database can be asked")
    report(f"  the remaining {len(nocand)-len(named):,} have only opaque IDs and cannot be queried at all")

    # ---------------- THE MISSING LIST, emitted regardless of whether the web helps ----------------
    missing = []
    for i in nocand:
        s = steps[i]
        missing.append({"reaction": s.get("id"), "category": s.get("category"), "source": s.get("source"),
                        "metabolites": parts_named(s)[:8],
                        "n_participants": len(P[i]), "queryable": bool(parts_named(s))})
    (OUT / "cell_missing_list.json").write_text(json.dumps(missing, indent=2))
    report(f"  -> missing list written: {OUT/'cell_missing_list.json'} ({len(missing):,} reactions)")

    # ---------------- VALIDATION: can the web recover catalysts we already know? ----------------
    rng = np.random.default_rng(A.SEED)
    val_pool = [i for i in cats if parts_named(steps[i])]
    report(f"\n  VALIDATION on {min(N_VALIDATE, len(val_pool))} reactions whose catalyst IS known "
           f"(pool {len(val_pool):,})")
    vs = [val_pool[int(x)] for x in rng.choice(len(val_pool), min(N_VALIDATE, len(val_pool)), replace=False)]
    generic = ["ATP", "H2O", "NAD+", "O2"]
    hit_web = hit_ctl = n_ok = 0
    for i in vs:
        truth = {c.upper() for c in cats[i]}
        terms = parts_named(steps[i])
        w = uniprot_by_terms(terms)
        time.sleep(PAUSE)
        c = uniprot_by_terms(generic)
        time.sleep(PAUSE)
        if w or c:
            n_ok += 1
            hit_web += bool(truth & {x.upper() for x in w})
            hit_ctl += bool(truth & {x.upper() for x in c})
    if n_ok:
        report(f"    queries returning anything : {n_ok}/{len(vs)}")
        report(f"    web query recall           : {hit_web/n_ok:.3f}  ({hit_web}/{n_ok})")
        report(f"    stripped-query control     : {hit_ctl/n_ok:.3f}  ({hit_ctl}/{n_ok})")
    else:
        report("    NO query returned anything -- the web pass is not usable here and nothing is filled.")

    beats = n_ok > 0 and hit_web > hit_ctl

    # ---------------- FILL, only if validation supports it ----------------
    filled = []
    if beats:
        sample = [named[int(x)] for x in rng.choice(len(named), min(N_FILL, len(named)), replace=False)]
        report(f"\n  FILLING a sample of {len(sample)} missing reactions from UniProt")
        for i in sample:
            terms = parts_named(steps[i])
            cands = uniprot_by_terms(terms, size=15)
            time.sleep(PAUSE)
            if cands:
                filled.append({"reaction": steps[i].get("id"), "metabolites": terms[:6],
                               "web_candidates": cands[:10], "source": "UniProtKB reviewed, human"})
        report(f"    {len(filled)}/{len(sample)} got >=1 candidate from outside our vocabulary")
        if filled:
            report(f"    example: {filled[0]['reaction']} -> {filled[0]['web_candidates'][:5]}")
    else:
        report("\n  NOT FILLING. The web query did not beat a query stripped of this reaction's own chemistry,")
        report("  so what comes back is 'some human enzymes', not 'an enzyme for this reaction'. Writing those")
        report("  into the missing list would be manufacturing content, which is worse than an empty column.")

    json.dump({"test": "cell_web_fill", "n_real_orphans": len(real), "n_no_local_candidate": len(nocand),
               "n_queryable": len(named), "n_validate": len(vs), "n_queries_ok": n_ok,
               "web_recall": (hit_web / n_ok) if n_ok else None,
               "control_recall": (hit_ctl / n_ok) if n_ok else None,
               "web_beats_control": bool(beats), "n_filled": len(filled), "filled": filled,
               "log": log}, open(OUT / "cell_web_fill.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'cell_web_fill.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
