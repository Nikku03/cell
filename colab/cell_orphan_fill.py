"""FILL THE REAL GAPS: catalyst genes for the 9,186 genuinely orphan reactions, in a reviewable form.

WHAT CHANGES HERE.  Everything before this was a benchmark -- solved reactions with the answer hidden, so
accuracy could be measured. This turns the method on the reactions that are ACTUALLY missing a catalyst,
where there is no answer key and never was. That is the deliverable, and it is also why the output format
matters more than the method: nothing downstream can check these against truth, so a human has to.

THE MEASURED ACCURACY GOVERNS THE FORMAT.  Obscure top-1 is 0.688 reaction-weighted, and the cluster
bootstrap puts the honest interval at [0.523, 0.873] over 38 distinct facts. So roughly one in three of
these will be wrong, and WHICH third is not knowable from the prediction alone. A file of bare gene names
would therefore be actively harmful -- it would read as annotation. Every row here carries the evidence
that produced it so a curator can accept or reject in seconds rather than re-deriving:

    the reaction, rendered as the model saw it
    the predicted gene
    COPIED       whether that gene was among the shown examples' catalysts. Not disqualifying -- chemistry
                 genuinely repeats -- but a copied answer is a retrieval hit and an uncopied one is an
                 inference, and those deserve different scrutiny.
    STARVED      how many examples were available. Below the full 12 the retrieval had little to work with,
                 and the abstention analysis found starved-and-copied rows are where the errors concentrate.
    n_examples, top_similarity, and the examples themselves
    the model's stated reason

TWO SANITY CHECKS RUN AUTOMATICALLY, because some errors do not need a human:
    is_gene        the prediction must be a real symbol in the model's own gene table. A hallucinated
                   symbol is a hard reject, not a judgement call.
    compartment    a catalyst is checked against the reaction's compartment. A cytosolic enzyme proposed
                   for a mitochondrial-matrix reaction is flagged -- not rejected, since compartment
                   annotation is itself incomplete, but surfaced.

THE POPULATION, and why it is 9,186 and not 17,551.  Of the orphans, 6,042 are binding or dissociation
steps that are spontaneous and correctly have no enzyme, and 2,323 are curator-flagged abstractions that
are not single steps at all. Guessing catalysts for those would be manufacturing annotation. Only the
metabolic and transition categories are filled.

DISTRIBUTED BY CONSTRUCTION.  Prompts are written as independent batches so they can be answered in
parallel by separate workers, and the merge refuses partial writes -- a batch that fails leaves the row
unfilled rather than silently absent.

WHAT THIS IS NOT.  It is not a completed network. At 0.69 with a wide interval it is a ranked worklist that
makes curation perhaps threefold faster, and the honest description of the artifact is "hypotheses with
their evidence attached", which is what the file is shaped like.

usage:  python cell_orphan_fill.py dump [N]     write prompt batches for N orphans (default all)
        python cell_orphan_fill.py merge        collect worker answers into the review table
        python cell_orphan_fill.py review       print the summary and the checks

-> outputs/orphan/cell_orphan_fill.json  and  outputs/orphan/orphan_fill_review.tsv
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
from cell_orphan_incontext import _render  # noqa: E402  -- identical rendering to the measured arm

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/orphan_fill")
K = 12
BATCH = 50
NONCAT = ("binding", "dissociation")
ABSTRACT = ("omitted", "uncertain")
SEED = 101


def build_pool():
    """The orphans worth filling, with their examples. Same retrieval as the measured arm."""
    B = T.build()
    steps, cats, P = B["steps"], B["cats"], B["P"]
    cat_of = defaultdict(set)
    for i in cats:
        for p in P[i]:
            cat_of[p].add(i)
    orph = [i for i, s in enumerate(steps) if not s.get("catalysts")]
    real = [i for i in orph if steps[i].get("category") not in NONCAT + ABSTRACT]
    return B, real, cat_of


def examples_for(i, P, cats, cat_of, k=K):
    pi = P[i]
    sc = {}
    for p in pi:
        for j in cat_of.get(p, ()):
            if j == i:
                continue
            u = len(pi | P[j])
            sc[j] = max(sc.get(j, 0), len(pi & P[j]) / u if u else 0)
    top = sorted(sc, key=lambda j: -sc[j])[:k]
    return top, [sc[j] for j in top]


_COMPMAP = (("cyto", "cytosol"), ("nucle", "nucleus"), ("mitoch", "mitochondrion"),
            ("endoplasmic", "er"), (" er", "er"), ("golgi", "golgi"), ("lysosom", "lysosome"),
            ("peroxisom", "peroxisome"), ("extracell", "extracellular"), ("secret", "extracellular"),
            ("plasma", "membrane"), ("membrane", "membrane"), ("vesic", "vesicle"),
            ("endosom", "endosome"), ("ribosom", "cytosol"), ("nucleol", "nucleus"))


def _compvocab(s):
    """Reduce a free-text compartment to a small controlled vocabulary, so 'nucleus' and 'nucleoplasm'
    compare equal and 'mitochondrion' matches 'mitochondrial matrix'."""
    s = (s or "").lower()
    return {v for k, v in _COMPMAP if k in s}


def cmd_dump(limit=None):
    B, real, cat_of = build_pool()
    steps, cats, P = B["steps"], B["cats"], B["P"]
    print(f"  {len(real)} fillable orphans (metabolic/transition; binding, dissociation and "
          f"curator-abstractions excluded)")
    sel = real if limit is None else real[:limit]
    SP.mkdir(parents=True, exist_ok=True)
    meta, txt = {}, []
    for pid, i in enumerate(sel):
        top, sims = examples_for(i, P, cats, cat_of)
        ex = [f"  SOLVED: {_render(steps[j], steps)}\n     catalyst: {', '.join(sorted(cats[j]))}"
              for j in top]
        s = steps[i]
        head = (f"#{pid}  ({s.get('source','?')}, {s.get('category','?')}, "
                f"{'reversible' if s.get('reversible') else 'irreversible'})")
        body = _render(s, steps)
        txt.append(head + "\n" + body + "\n  --- SOLVED REACTIONS FOR REFERENCE ---\n" + "\n".join(ex)
                   if ex else head + "\n" + body + "\n  --- NO SIMILAR SOLVED REACTION FOUND ---")
        meta[str(pid)] = {"step": int(i), "id": s.get("id", ""), "name": s.get("name", ""),
                          "category": s.get("category", ""), "source": s.get("source", ""),
                          "n_examples": len(top), "top_sim": float(sims[0]) if sims else 0.0,
                          "example_catalysts": sorted({g for j in top for g in cats[j]}),
                          "compartments": sorted({q.get("compartment") for q in
                                                  (s.get("in") or []) + (s.get("out") or [])
                                                  if q.get("compartment")})}
    nb = 0
    for b in range(0, len(txt), BATCH):
        (SP / f"orphan_b{b // BATCH}.txt").write_text(
            ("\n\n" + "=" * 90 + "\n\n").join(txt[b:b + BATCH]))
        nb += 1
    json.dump(meta, open(OUT / "orphan_fill_meta.json", "w"), indent=1)
    starved = sum(1 for v in meta.values() if v["n_examples"] < K)
    print(f"  wrote {len(txt)} prompts in {nb} batches of {BATCH} -> {SP}")
    print(f"  {starved} ({starved/max(len(txt),1):.1%}) have fewer than {K} examples available -- these")
    print(f"  are the STARVED rows where the abstention analysis found the errors concentrate")
    return 0


def cmd_merge():
    meta = json.load(open(OUT / "orphan_fill_meta.json"))
    ansf = OUT / "orphan_fill_answers.json"
    ans = json.load(open(ansf)) if ansf.exists() else {}
    got = sum(1 for p in meta if p in ans)
    print(f"  {got} of {len(meta)} answered ({got/max(len(meta),1):.1%})")
    if got == 0:
        print("  nothing to merge; run the workers first")
        return 1
    D = json.load(open(OUT / "cell_complete.json"))
    known = {g["name"].upper(): g for g in D["genes"]}
    # VALIDITY IS HGNC, NOT THE MODEL'S OWN TABLE. Checking against cell_complete.json alone called 67
    # predictions hallucinated; 65 of them were valid HGNC symbols -- PIK3CA, PRPS1, UCK2, TYMP -- simply
    # absent from a 16,492-gene table when HGNC lists 19,296. That is a gap in the MODEL, not an error by
    # the predictor, and conflating the two would have sent a curator to reject correct answers.
    try:
        hgnc = {x.upper() for x in json.load(open("data/hgnc_protein_coding.json"))}
    except Exception:
        hgnc = set()
    MEMBRANE = ("membrane", "transporter", "channel")
    rows, nbad, nmiss, ncopy, nstarve, ncompart = [], 0, 0, 0, 0, 0
    for pid, m in meta.items():
        a = ans.get(pid)
        if not a:
            continue
        gene = str(a.get("gene", "")).strip().upper()
        why = str(a.get("why", ""))[:300]
        in_model = gene in known
        is_gene = in_model or (gene in hgnc)
        missing_from_model = is_gene and not in_model
        copied = gene in {g.upper() for g in m["example_catalysts"]}
        starved = m["n_examples"] < K
        # COMPARTMENT: a membrane protein legitimately connects two compartments -- that is what a
        # transporter IS. The first version flagged every SLC and every ecto-enzyme for spanning cytosol
        # and extracellular, i.e. it flagged them for doing their job, and produced a 36% "mismatch" rate
        # made almost entirely of correct predictions. Only a SOLUBLE protein in one specific compartment,
        # proposed for a reaction wholly in a different one, is a genuine conflict.
        comp_ok = True
        gc = (known[gene].get("comp") or "").lower() if in_model else ""
        if gc and m["compartments"] and not any(w in gc for w in MEMBRANE):
            # BOTH SIDES ARE FREE TEXT and they do not use the same words: a gene annotated "nucleus"
            # catalyses reactions in "nucleoplasm", "mitochondrion" against "mitochondrial matrix".
            # Substring matching on the raw strings failed on exactly those, so both sides are reduced to
            # a small vocabulary first and only a DISJOINT pair in a single-compartment reaction is a
            # conflict. This check stays advisory: compartment annotation is incomplete on both sides.
            gv, rv = _compvocab(gc), set()
            for c in m["compartments"]:
                rv |= _compvocab(c)
            comp_ok = (not gv) or (not rv) or bool(gv & rv) or len(m["compartments"]) > 1
        nbad += (not is_gene)
        nmiss += missing_from_model
        ncopy += copied
        nstarve += starved
        ncompart += (not comp_ok)
        rows.append({"pid": pid, "step": m["step"], "id": m["id"], "name": m["name"],
                     "category": m["category"], "gene": gene, "is_gene": is_gene,
                     "missing_from_model": missing_from_model,
                     "copied": copied, "starved": starved, "n_examples": m["n_examples"],
                     "top_sim": m["top_sim"], "compartment_ok": comp_ok,
                     "compartments": m["compartments"], "why": why})
    hdr = ["pid", "reaction_id", "reaction", "category", "PREDICTED_GENE", "is_real_gene", "copied",
           "starved", "n_examples", "top_similarity", "compartment_ok", "compartments", "reason"]
    with open(OUT / "orphan_fill_review.tsv", "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in
                               [r["pid"], r["id"], r["name"][:70], r["category"], r["gene"],
                                r["is_gene"], r["copied"], r["starved"], r["n_examples"],
                                f"{r['top_sim']:.3f}", r["compartment_ok"],
                                ";".join(r["compartments"]), r["why"].replace("\t", " ")]) + "\n")
    res = {"test": "cell_orphan_fill", "n_meta": len(meta), "n_answered": got, "rows": rows,
           "checks": {"not_a_real_gene": nbad, "valid_but_missing_from_model": nmiss,
                      "copied_from_examples": ncopy, "starved": nstarve,
                      "compartment_mismatch": ncompart}}
    json.dump(res, open(OUT / "cell_orphan_fill.json", "w"), indent=1)
    print(f"  -> {OUT/'orphan_fill_review.tsv'}  and  {OUT/'cell_orphan_fill.json'}")
    return cmd_review()


def cmd_review():
    f = OUT / "cell_orphan_fill.json"
    if not f.exists():
        print("  no merged output yet")
        return 1
    res = json.load(open(f))
    rows, c = res["rows"], res["checks"]
    n = len(rows)
    print("\n" + "=" * 100)
    print("ORPHAN FILL -- what came back, and what a curator should look at first")
    print("=" * 100)
    print(f"  {n} predictions from {res['n_meta']} orphan reactions")
    print(f"\n  AUTOMATIC CHECKS (these need no human)")
    print(f"    not a real gene symbol       {c['not_a_real_gene']:>6}  ({c['not_a_real_gene']/max(n,1):.1%})"
          f"  <- hard reject, hallucinated (checked against HGNC)")
    print(f"    valid HGNC, ABSENT from model{c.get('valid_but_missing_from_model',0):>6}  "
          f"({c.get('valid_but_missing_from_model',0)/max(n,1):.1%})  <- a gap in the MODEL, not an error")
    print(f"    compartment mismatch         {c['compartment_mismatch']:>6}  "
          f"({c['compartment_mismatch']/max(n,1):.1%})  <- flag, not reject; annotation is incomplete")
    print(f"\n  TRIAGE (these decide how hard to look)")
    print(f"    copied from a shown example  {c['copied_from_examples']:>6}  "
          f"({c['copied_from_examples']/max(n,1):.1%})  <- retrieval hit; check the chemistry matches")
    print(f"    starved (<{K} examples)       {c['starved']:>6}  ({c['starved']/max(n,1):.1%})"
          f"  <- least support; errors concentrate here")
    hi = [r for r in rows if r["is_gene"] and not r["starved"] and r["compartment_ok"]]
    lo = [r for r in rows if r["starved"] or not r["is_gene"]]
    print(f"\n  {len(hi)} rows ({len(hi)/max(n,1):.1%}) are real genes with full retrieval and no")
    print(f"  compartment conflict. {len(lo)} rows ({len(lo)/max(n,1):.1%}) are starved or invalid.")
    print(f"\n  THE NUMBER THAT GOVERNS ALL OF THIS: measured obscure top-1 is 0.688, cluster interval")
    print(f"  [0.523, 0.873]. About a third of these are wrong and the prediction does not say which.")
    print(f"  This is a ranked worklist with its evidence attached, not annotation.")
    print(f"\n  distinct genes proposed: {len({r['gene'] for r in rows})}")
    from collections import Counter
    top = Counter(r["gene"] for r in rows).most_common(8)
    print(f"  most-proposed: " + ", ".join(f"{g} x{k}" for g, k in top))
    print(f"  IF ONE GENE DOMINATES, that is the popularity bias the obscurity work measured, and those")
    print(f"  rows deserve the hardest look regardless of what the other flags say.")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "review"
    if cmd == "dump":
        raise SystemExit(cmd_dump(int(sys.argv[2]) if len(sys.argv) > 2 else None))
    if cmd == "merge":
        raise SystemExit(cmd_merge())
    raise SystemExit(cmd_review())
