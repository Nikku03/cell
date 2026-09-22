"""Validate structure-based function transfer (AlphaFold + Foldseek) — is it accurate, and does it see what
sequence methods can't?

Two claims, both checked on proteins whose function is KNOWN (so we have ground truth), blind to that
annotation (the pipeline only gets the structure):

  1. ACCURACY — the transferred function keyword-matches the protein's real function.
  2. STRUCTURE > SEQUENCE — among the confident structural neighbours, a meaningful fraction sit in the
     sequence 'twilight zone' (seqId < 30%): homologs a sequence method (BLAST/ESM) would miss but that
     Foldseek recovers from the fold. This is the user's hypothesis, made measurable.

Network: AlphaFold EBI + Foldseek web API. -> outputs/orphan/foldseek_function_validation.json
"""
import json, re
from pathlib import Path
from foldseek_function import predict_function

# resolve outputs/orphan relative to the repo root (robust to cwd)
_here = Path(__file__).resolve().parent
OUT = next((p / "outputs" / "orphan" for p in [_here.parent, Path.cwd()]
            if (p / "outputs" / "orphan").exists()), Path("outputs/orphan"))

# known-function proteins spanning fold classes; `keywords` = accepted matches for the transferred annotation.
BENCH = [
    dict(gene="LDHA",  uniprot="P00338", keywords=["lactate dehydrogenase", "dehydrogenase", "oxidoreductase"]),
    dict(gene="CA2",   uniprot="P00918", keywords=["carbonic anhydrase", "anhydrase"]),
    dict(gene="SOD1",  uniprot="P00441", keywords=["superoxide dismutase", "dismutase"]),
    dict(gene="MB",    uniprot="P02144", keywords=["myoglobin", "globin", "oxygen"]),
    dict(gene="CTSD",  uniprot="P07339", keywords=["cathepsin", "peptidase", "protease", "aspartic"]),
]


def _match(pred, keywords):
    p = (pred or "").lower()
    return any(k.lower() in p for k in keywords)


def main(bench=None):
    bench = bench or BENCH
    rows = []
    for b in bench:
        r = predict_function(b["gene"], self_uniprot=b.get("uniprot"))
        if r.get("abstain"):
            rows.append(dict(gene=b["gene"], ok=False, reason=r["reason"])); continue
        hit = _match(r["prediction"], b["keywords"])
        # twilight-zone structural homologs: confident (prob>=0.9) but low sequence identity (<30%),
        # scanned over ALL hits (predict_function.twilight_homologs), carrying the RIGHT function.
        twil = r.get("twilight_homologs", [])
        twil_right = [h for h in twil if _match(h.get("desc"), b["keywords"])]
        rows.append(dict(gene=b["gene"], ok=True, prediction=r["prediction"], correct=bool(hit),
                         best_seqId=r.get("homolog_seqId"), confidence=r["confidence"],
                         n_twilight=len(twil), n_twilight_correct=len(twil_right),
                         twilight_example=(dict(desc=twil_right[0].get("desc"), seqId=twil_right[0].get("seqId"))
                                           if twil_right else None)))
    ran = [r for r in rows if r.get("ok")]
    n_correct = sum(1 for r in ran if r.get("correct"))
    n_twil = sum(1 for r in ran if r.get("n_twilight_correct", 0) > 0)
    passed = (len(ran) >= 4 and n_correct / max(1, len(ran)) >= 0.8 and n_twil >= 2)
    res = dict(passed=bool(passed),
               summary=dict(n_ran=len(ran), n_correct=n_correct,
                            accuracy=round(n_correct / max(1, len(ran)), 3),
                            n_with_twilight_homolog=n_twil),
               proteins=rows,
               bar="function keyword-match accuracy>=0.8 on>=4 proteins & >=2 with a correct twilight-zone (<30% seqId) structural homolog")
    json.dump(res, open(OUT / "foldseek_function_validation.json", "w"), indent=2)
    print("=" * 78)
    print("FOLDSEEK FUNCTION TRANSFER — AlphaFold structure -> function  —  %s" % ("PASS" if passed else "FAIL"))
    print("=" * 78)
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['gene']:6} ABSTAIN ({r['reason'][:50]})"); continue
        mark = "OK " if r["correct"] else "MISS"
        print(f"  {r['gene']:6} [{mark}] '{r['prediction'][:42]}'  (best seqId {r['best_seqId']})")
        if r.get("twilight_example"):
            t = r["twilight_example"]
            print(f"          structure>sequence: correct fn from a homolog at seqId {t['seqId']} — '{t['desc'][:40]}'")
    s = res["summary"]
    print(f"\n  accuracy {s['n_correct']}/{s['n_ran']} = {s['accuracy']} | "
          f"{s['n_with_twilight_homolog']} proteins recovered correct function via a <30% seqId structural homolog")
    print("\n-> outputs/orphan/foldseek_function_validation.json")
    return res


if __name__ == "__main__":
    main()
