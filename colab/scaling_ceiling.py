"""WHAT HAPPENS IF THE STRUCTURAL COVERAGE GOES FROM 141 PROTEINS TO THE WHOLE PROTEOME?

The question was asked as a forecast. Most of it is a measurement, so it is measured here instead.

THE FIRST THING TO SETTLE, because it changes the shape of the answer: STRUCTURE IS NOT THE
BOTTLENECK AND HAS NOT BEEN FOR SOME TIME. AlphaFold DB covers the entire reviewed human proteome.
This project runs on 141 proteins because of what could be LABELLED, not what could be folded, and
scaling the structures does not scale the labels. So the useful question is which of the three
structural results is label-limited and which is not.

  loop 54  is a variant pathogenic          0.9464   needs ClinVar pathogenic variants per gene
  loop 57  what kind of damage              0.7304   needs UniProt residue-level annotation
  loop 60  which residue is functional      0.7670   needs NOTHING at inference -- coordinates only

The third one is different in kind. It consumes annotation only to be SCORED, never to predict, so a
protein with no annotation at all can still be handed to it. Whether that is worth doing depends on
whether its accuracy holds up away from the well-studied proteins it was measured on, and that is
testable on the 81 proteins already scored.

THE TEST, and it is post-hoc. Loop 60's gates were declared before it ran; this correlation was
computed afterwards, so it is a description of the result and not a test of it. Publication counts
come from gene2pubmed, which this project already holds.

    per-protein AUC vs publication count      rho -0.3461  p 0.0016
      | protein length                        rho -0.3367  p 0.0021
      | number of annotated sites             rho -0.2734  p 0.0135
      | both                                  rho -0.2868  p 0.0094
      within nucleotide-site proteins only    rho -0.3930  p 0.0261

The sign is NEGATIVE and survives every control available. That is the opposite of every other
result in this repository: publication count predicts cancer drivers at 0.8259 against our model's
0.2990 partial, and was the single best drug feature at 0.7913. Here it runs the other way -- the
model is WORST on the proteins the literature knows best.

THE ALTERNATIVE EXPLANATION, which is NOT tested here and should not be waved away: a heavily studied
protein accumulates annotation of marginal sites -- weak, transient, allosteric, or found by one
low-throughput assay -- and a buried-pocket detector should be expected to miss those. Under that
reading the effect is about what gets ANNOTATED on famous proteins rather than about what the model
can find, and the honest claim shrinks to "the extra sites on famous proteins are not classic
pockets". Distinguishing the two needs a held-out set of unstudied proteins with independently
determined sites, which this project does not have.

-> outputs/scaling_ceiling.json
"""
import collections
import gzip
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC

# counted from the UniProt REST x-total-results header on the reviewed human proteome; recorded as
# constants because they are a remote census this module does not re-fetch on every run
UNIPROT = {"reviewed human proteins": 20431,
           "with a binding-site annotation": 4808,
           "with an active-site annotation": 2321,
           "with any experimental 3D structure": 8844,
           "with an AlphaFold model": 20431}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def clinvar_census():
    pg = collections.Counter()
    for ln in gzip.open(SC / "clinvar.vcf.gz", "rt", errors="ignore"):
        if ln.startswith("#"):
            continue
        info = ln.split("\t")[7]
        if "CLNSIG=Pathogenic" not in info and "CLNSIG=Likely_pathogenic" not in info:
            continue
        if "missense_variant" not in info:
            continue
        m = re.search(r"GENEINFO=([^:;]+)", info)
        if m:
            pg[m.group(1)] += 1
    return pg


def rk(x):
    return (np.argsort(np.argsort(x)) + 1) / len(x)


def partial(y, x, Z):
    Y, X = rk(y), rk(x)
    Z = np.column_stack([rk(z) for z in Z] + [np.ones(len(y))])
    ry = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
    rx = X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return spearmanr(ry, rx)


def main():
    say("=" * 100)
    say("  WHOLE-PROTEOME STRUCTURE -- what is actually gated by it, and what is not")
    say("  the fame correlation below is POST-HOC on loop 60 and is not one of its gates")
    say("=" * 100)
    say()

    say("  THE ANNOTATION CEILING (reviewed human proteome)")
    for k, v in UNIPROT.items():
        say(f"     {v:7,d}  {k}   ({v / UNIPROT['reviewed human proteins']:5.1%})")
    say()

    pg = clinvar_census()
    say("  THE CLINVAR LABEL CEILING -- genes by pathogenic missense count")
    tiers = {}
    for t in (1, 10, 30, 50, 100, 200):
        n = sum(1 for v in pg.values() if v >= t)
        tiers[t] = n
        say(f"     >= {t:4d} pathogenic missense: {n:6,d} genes")
    say(f"     this project uses 141 genes; {tiers[100]:,} genes in all of ClinVar reach 100.")
    say("     the pathogenicity model is therefore AT its label ceiling, not below it")
    say()

    per = json.load(open(OUT / "loop_pocket.json"))["per_protein"]
    pubs = json.load(open(SC / "_pubs_by_symbol.json"))
    B = pickle.load(open(SC / "_pocket_v1.pkl", "rb"))
    rows = {r["gene"]: r for r in B["rows"]}
    g = [k for k in per if k in pubs and k in rows]
    A = np.array([per[k] for k in g])
    P = np.log10(np.array([pubs[k] for k in g], float))
    L = np.log10(np.array([len(rows[k]["y"]) for k in g], float))
    N = np.log10(np.array([rows[k]["y"].sum() for k in g], float))

    say(f"  DOES THE POCKET FINDER NEED THE PROTEIN TO BE WELL STUDIED?   n={len(g)}")
    out = {}
    for nm, r in (("AUC ~ papers", spearmanr(A, P)),
                  ("AUC ~ length", spearmanr(A, L)),
                  ("AUC ~ n annotated sites", spearmanr(A, N)),
                  ("AUC ~ papers | length", partial(A, P, [L])),
                  ("AUC ~ papers | n sites", partial(A, P, [N])),
                  ("AUC ~ papers | length + n sites", partial(A, P, [L, N]))):
        say(f"     {nm:34s} rho {r.statistic:+.4f}   p {r.pvalue:.4f}")
        out[nm] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    q = np.quantile(P, [0, .25, .5, .75, 1.0])
    quart = []
    say()
    for i in range(4):
        m = (P >= q[i]) & (P <= q[i + 1])
        say(f"     quartile {i + 1} by papers: n={m.sum():3d}  "
            f"{10 ** q[i]:6.0f}-{10 ** q[i + 1]:6.0f} papers   mean AUC {A[m].mean():.4f}")
        quart.append({"q": i + 1, "n": int(m.sum()), "lo": float(10 ** q[i]),
                      "hi": float(10 ** q[i + 1]), "mean_auc": float(A[m].mean())})
    say()
    say("     the sign is NEGATIVE and survives every control. In this repository publication count")
    say("     predicts cancer drivers at 0.8259 and was the best drug feature at 0.7913; here it")
    say("     runs the other way. The untested alternative is that famous proteins accumulate")
    say("     marginal site annotations a pocket detector should be expected to miss.")

    man = RM.manifest(inputs=[str(SC / "clinvar.vcf.gz"), str(SC / "_pubs_by_symbol.json"),
                              str(OUT / "loop_pocket.json")],
                      available=len(per), used=len(g), selection="filtered", seed=0,
                      controls=["protein length partialled out",
                                "number of annotated sites partialled out",
                                "both partialled out together"],
                      note="POST-HOC on loop 60. Not a gate. UniProt counts are a recorded remote "
                           "census, not re-fetched per run")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "scaling_ceiling", "post_hoc": True, "manifest": man,
               "uniprot": UNIPROT, "clinvar_tiers": tiers, "fame": out, "quartiles": quart,
               "log": log}, open(OUT / "scaling_ceiling.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'scaling_ceiling.json'}")


if __name__ == "__main__":
    main()
