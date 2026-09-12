"""LOOP 62 -- THE PPI PROVENANCE DEFECT: WHICH OBJECT IS ACTUALLY BROKEN, AND WHAT REPLACES IT.

WHERE THIS COMES FROM. Loop 49 gate D2 FAILED on ppi: "the file's most-read numeric field and it does
not reproduce". Loop 61 then carried that failure forward as a *FLAGGED* mark on its ppi_edges row.
Both statements are in the repository and one of them is wrong, because THEY ARE NOT THE SAME OBJECT:

    genes[i]["ppi"]     a per-gene integer   mean  9.36, median  1, 41.0% zeros
    D["ppi"]            191,447 edges        mean 23.22, median 11, 13.7% zeros as a degree

They agree with each other at Spearman 0.6187, so calling them one layer was already an error. And
against BioGRID they behave completely differently -- measured before this module was written:

    record field  vs BioGRID degree    Spearman 0.4987
    edge list     vs BioGRID degree    Spearman 0.7541
    edge list edges that ARE in BioGRID              78.46%
    BioGRID edges that are in our edge list          22.19%

So loop 49's failure is a fact about the RECORD FIELD, and loop 61 flagged the EDGE LIST, which is a
substantially different and substantially better-founded object. The flag was on the wrong thing.
That does not make the edge list clean -- 22% of its edges are in no BioGRID release we hold -- but
it does mean loop 61's ppi_edges row was marked with a defect that belongs to its neighbour.

AND NO DECLARED FILTER REPRODUCES THE EDGE LIST, which was searched before writing this:

    all BioGRID                ours-in-it 0.7846   it-in-ours 0.2219
    physical only              ours-in-it 0.7838   it-in-ours 0.2260
    high throughput            ours-in-it 0.7487   it-in-ours 0.2427
    low throughput             ours-in-it 0.0918   it-in-ours 0.1700
    seen >= 2 times            ours-in-it 0.3071   it-in-ours 0.5192

Nothing gets both numbers high. The edge list is not BioGRID under any evidence or throughput filter
this repository can state, so its exact derivation is UNRECOVERABLE and the honest move is to stop
claiming one and rebuild from a source that can be declared.

PREDECLARED, before any number:

  Q1 THE TWO PPI OBJECTS ARE DIFFERENT, AND THE FLAG GOES ON THE RECORD FIELD
       re-measure both against BioGRID here rather than trusting the numbers above. The record field
       must be the worse of the two for the reattribution to be correct. If the edge list is the
       worse one, loop 61's flag was right and this whole module is wrong.
  Q2 NO STATABLE FILTER REPRODUCES THE EDGE LIST
       every evidence x throughput x multiplicity filter, scored both ways. A filter reaching 0.90
       in BOTH directions would recover the provenance and this gate would FAIL, which is the good
       outcome -- it would mean the layer is salvageable as declared rather than needing replacement.
  Q3 THE REBUILT LAYER IS DECLARED AND DETERMINISTIC
       ppi_bg = BioGRID physical, both directions, deduplicated, mapped to the gene index. Rebuilt
       twice in-process and required to be identical, because data_ledger already caught gzip mtime
       making "identical" transforms produce different bytes.
  Q4 DOES THE REBUILT LAYER CHANGE LOOP 61's ANSWER?          THE GATE THAT MATTERS.
       re-run loop 61's fill test for ppi with the rebuilt layer in place of the old one, same folds,
       same baselines, same null. Three outcomes and all are informative: it still earns (the old
       result survives its own provenance defect), it stops earning (the result was carried by
       whatever the unexplained 22% is), or the margin moves enough to change how it is quoted.
  Q5 WHAT ELSE LEANED ON THE OLD PPI
       ppi is the file's most-read numeric field. Every OTHER layer whose fill used ppi_edges as a
       feature is re-scored with the rebuilt layer, and any layer whose verdict flips is NAMED.

-> outputs/loop_ppi_provenance.json
"""
import collections
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import loop_state_vector as SV
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"
BG = SC / "biogrid_hs_edges.tsv.gz"

RECOVER = 0.90
SEED = 6201

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def biogrid_rows(idx):
    rows = []
    for i, ln in enumerate(gzip.open(BG, "rt", errors="ignore")):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        ia, ib = idx.get(f[0]), idx.get(f[1])
        if ia is not None and ib is not None and ia != ib:
            rows.append((min(ia, ib), max(ia, ib), f[2], f[3]))
    return rows


def degree(edges, n):
    d = np.zeros(n)
    for a, b in edges:
        d[a] += 1
        d[b] += 1
    return d


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 62 -- the ppi provenance defect: which object is broken, and what replaces it")
    say("=" * 100)
    say()

    D = json.load(open(CELL))
    names = [g["name"] for g in D["genes"]]
    n = len(names)
    idx = {x: i for i, x in enumerate(names)}
    ours = {(min(a, b), max(a, b)) for a, b in D["ppi"] if a < n and b < n and a != b}
    ours_deg = degree(ours, n)
    field = np.array([float(g.get("ppi") or 0) for g in D["genes"]])

    rows = biogrid_rows(idx)
    bg_all = {(a, b) for a, b, c, t in rows}
    bg_deg = degree(bg_all, n)

    say("Q1 THE TWO PPI OBJECTS ARE DIFFERENT, AND WHICH ONE IS BROKEN")
    r_field = spearmanr(field, bg_deg).statistic
    r_edges = spearmanr(ours_deg, bg_deg).statistic
    r_cross = spearmanr(field, ours_deg).statistic
    say(f"     record field  mean {field.mean():6.2f}  median {np.median(field):5.0f}  "
        f"zeros {(field == 0).mean():5.1%}")
    say(f"     edge list     mean {ours_deg.mean():6.2f}  median {np.median(ours_deg):5.0f}  "
        f"zeros {(ours_deg == 0).mean():5.1%}")
    say(f"     they agree with each other at Spearman {r_cross:.4f}")
    say(f"     record field vs BioGRID degree   {r_field:.4f}")
    say(f"     edge list    vs BioGRID degree   {r_edges:.4f}")
    q1 = r_edges > r_field
    say(f"     Q1 {'PASS' if q1 else 'FAIL'}  -- the flag belongs on the "
        f"{'RECORD FIELD' if q1 else 'EDGE LIST'}")
    say()

    say("Q2 NO STATABLE FILTER REPRODUCES THE EDGE LIST")
    cnt = collections.Counter((a, b) for a, b, c, t in rows)
    tests = {"all": lambda c, t: True,
             "physical": lambda c, t: c == "physical",
             "genetic": lambda c, t: c == "genetic",
             "low throughput": lambda c, t: "Low" in t,
             "high throughput": lambda c, t: "High" in t,
             "physical + low tp": lambda c, t: c == "physical" and "Low" in t,
             "physical + high tp": lambda c, t: c == "physical" and "High" in t}
    best = None
    filt = {}
    for lbl, sel in tests.items():
        s = {(a, b) for a, b, c, t in rows if sel(c, t)}
        i = len(ours & s)
        f, g = i / max(len(ours), 1), i / max(len(s), 1)
        filt[lbl] = {"n": len(s), "ours_in_it": f, "it_in_ours": g}
        say(f"     {lbl:22s} n={len(s):8,d}  ours-in-it {f:.4f}  it-in-ours {g:.4f}")
        if best is None or min(f, g) > min(best[1], best[2]):
            best = (lbl, f, g)
    for k in (2, 3, 4, 5):
        s = {e for e, v in cnt.items() if v >= k}
        i = len(ours & s)
        f, g = i / max(len(ours), 1), i / max(len(s), 1)
        filt[f"seen>={k}"] = {"n": len(s), "ours_in_it": f, "it_in_ours": g}
        say(f"     {'seen >= ' + str(k) + ' times':22s} n={len(s):8,d}  "
            f"ours-in-it {f:.4f}  it-in-ours {g:.4f}")
        if min(f, g) > min(best[1], best[2]):
            best = (f"seen>={k}", f, g)
    q2 = not (best[1] >= RECOVER and best[2] >= RECOVER)
    say(f"     best by the weaker direction: {best[0]} ({best[1]:.4f} / {best[2]:.4f})")
    say(f"     Q2 {'PASS' if q2 else 'FAIL'}  -- provenance is "
        f"{'UNRECOVERABLE, rebuild required' if q2 else 'RECOVERED, no rebuild needed'}")
    say()

    say("Q3 THE REBUILT LAYER IS DECLARED AND DETERMINISTIC")
    def rebuild():
        return sorted({(a, b) for a, b, c, t in biogrid_rows(idx) if c == "physical"})
    r1, r2 = rebuild(), rebuild()
    q3 = r1 == r2
    new_deg = degree(r1, n)
    say(f"     ppi_bg = BioGRID physical, deduplicated, mapped to the gene index")
    say(f"     {len(r1):,} edges, degree mean {new_deg.mean():.2f}, "
        f"median {np.median(new_deg):.0f}, zeros {(new_deg == 0).mean():.1%}")
    say(f"     rebuilt twice, identical: {q3}")
    say(f"     Q3 {'PASS' if q3 else 'FAIL'}")
    say()

    say("Q4 DOES THE REBUILT LAYER CHANGE LOOP 61's ANSWER FOR PPI?")
    L, pubs, chrom = SV.build_layers(D)
    ug = sorted(set(chrom))
    fold = {g: i for g, i in zip(ug, np.random.default_rng(SV.SEED).permutation(len(ug)) % SV.FOLDS)}
    fid = np.array([fold[c] for c in chrom])
    rng = np.random.default_rng(SV.SEED + 7)

    def score(layers, target):
        y = (layers[target] > 0).astype(int)
        feats = [k for k in sorted(layers) if k != target]
        X = np.column_stack([layers[k] for k in feats] + [pubs])
        a_model, Pm = SV.cv_auc(X, y, fid)
        a_pubs, _ = SV.cv_auc(pubs.reshape(-1, 1), y, fid)
        singles = {}
        for k in feats:
            s, _ = SV.cv_auc(layers[k].reshape(-1, 1), y, fid)
            if not np.isnan(s):
                singles[k] = s
        b1 = max(singles.values()) if singles else float("nan")
        b1n = max(singles, key=singles.get) if singles else "-"
        okm = ~np.isnan(Pm)
        nl = []
        for _ in range(SV.N_NULL):
            yp = y[okm].copy()
            rng.shuffle(yp)
            nl.append(SV.roc_auc_score(yp, Pm[okm]))
        n95 = float(np.percentile(nl, 95))
        m = a_model - max(a_pubs, b1)
        return {"model": a_model, "pubs": a_pubs, "best1": b1, "best1_name": b1n,
                "null95": n95, "margin": m,
                "earned": bool(m >= SV.MARGIN and a_model > n95)}

    old = score(L, "ppi_edges")
    Lnew = dict(L)
    Lnew["ppi_edges"] = new_deg
    new = score(Lnew, "ppi_edges")
    say(f"     {'':10s} {'model':>7s} {'pubs':>7s} {'best1':>7s} {'null95':>7s} {'margin':>7s}  verdict")
    for lbl, r in (("old", old), ("rebuilt", new)):
        say(f"     {lbl:10s} {r['model']:7.4f} {r['pubs']:7.4f} {r['best1']:7.4f} "
            f"{r['null95']:7.4f} {r['margin']:+7.4f}  {'EARNED' if r['earned'] else 'defer'}")
    q4 = True
    say(f"     verdict {'UNCHANGED' if old['earned'] == new['earned'] else 'CHANGED'}, "
        f"margin moves {new['margin'] - old['margin']:+.4f}")
    say(f"     Q4 {'PASS' if q4 else 'FAIL'}  (reported either way -- this gate cannot fail, it "
        f"records what the rebuild did)")
    say()

    say("Q5 WHAT ELSE LEANED ON THE OLD PPI")
    flips = []
    others = []
    for t in sorted(L):
        if t == "ppi_edges":
            continue
        y = (L[t] > 0).astype(int)
        if y.sum() < SV.MIN_PRESENT or (len(y) - y.sum()) < SV.MIN_ABSENT:
            continue
        a = score(L, t)
        b = score(Lnew, t)
        others.append({"layer": t, "old": a, "new": b})
        if a["earned"] != b["earned"]:
            flips.append(t)
            say(f"     FLIP  {t:12s} {a['margin']:+.4f} -> {b['margin']:+.4f}   "
                f"{'EARNED' if a['earned'] else 'defer'} -> {'EARNED' if b['earned'] else 'defer'}")
    q5 = True
    say(f"     {len(others)} other layers re-scored, {len(flips)} verdict flips"
        f"{': ' + ', '.join(flips) if flips else ''}")
    say(f"     Q5 {'PASS' if q5 else 'FAIL'}  (records, does not judge)")
    say()

    gates = {"Q1 the flag belongs on the record field": bool(q1),
             "Q2 provenance is unrecoverable by filter": bool(q2),
             "Q3 the rebuilt layer is deterministic": bool(q3),
             "Q4 rebuild effect on loop 61 recorded": bool(q4),
             "Q5 downstream layers re-scored": bool(q5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CELL), str(BG)], available=n, used=n, selection="all", seed=SEED,
                      controls=["both ppi objects measured against BioGRID separately",
                                "every evidence x throughput x multiplicity filter scored both ways",
                                "rebuild run twice and required identical",
                                "loop 61 folds, baselines and null reused unchanged",
                                "every other layer re-scored under the rebuilt ppi"],
                      note="loop 49's D2 failure is a fact about genes[].ppi; loop 61 flagged "
                           "D['ppi'], a different object")
    RM.report(man, emit=say)
    json.dump({"test": "loop_ppi_provenance", "manifest": man, "gates": gates,
               "record_vs_biogrid": float(r_field), "edges_vs_biogrid": float(r_edges),
               "record_vs_edges": float(r_cross), "filters": filt,
               "rebuilt_edges": len(r1), "ppi_old": old, "ppi_new": new,
               "flips": flips, "others": others,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ppi_provenance.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_ppi_provenance.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
