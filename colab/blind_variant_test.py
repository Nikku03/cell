"""blind_variant_test — a GENUINE held-out test: does the variant-effect engine separate pathogenic from benign
variants it has NEVER seen, for genes that were NOT in the model until this session?

Why this is honestly blind:
  - the genes (PIK3CA, PAH, GBA, TYK2) were ABSENT from the original model — added in P0 with independent STRING
    edges, so no curated disease/role knowledge about them was ever loaded;
  - ESM-2 is ZERO-SHOT — it scores each substitution from evolutionary statistics; it never saw these variants
    or their labels;
  - the pathogenic/benign labels (ClinVar via UniProt) are withheld from the engine and used ONLY to score.

If functional_score separates pathogenic from benign (AUC > 0.5) on genes and variants the model never saw,
that is generalisation, not recall — the first thing in this whole project that is genuinely held-out.
-> outputs/orphan/blind_variant_test.json
"""
import os, sys, json, urllib.request, ssl, random
sys.path.insert(0, os.path.dirname(__file__))
import molecular_engine as me

OUT = "outputs/orphan"
_CA = "/root/.ccr/ca-bundle.crt"
_1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
      "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
      "TYR": "Y", "VAL": "V"}


def _ctx():
    return ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None


def fetch_labeled_variants(acc):
    """UniProt variation -> [(pos, wt, mut, label in {path,benign})] for single-aa missense with a clinical call."""
    d = json.load(urllib.request.urlopen(f"https://www.ebi.ac.uk/proteins/api/variation?accession={acc}",
                                         timeout=60, context=_ctx()))
    feats = d["features"] if isinstance(d, dict) else d[0]["features"]
    out = []
    for f in feats:
        wt = f.get("wildType", ""); mut = f.get("alternativeSequence", "")
        if len(wt) != 1 or len(mut) != 1 or mut in ("*", "-"):
            continue
        try:
            pos = int(f.get("begin"))
        except (TypeError, ValueError):
            continue
        cs = str(f.get("clinicalSignificances", "")).lower()
        lab = "path" if "pathogenic" in cs else "benign" if "benign" in cs else None
        if lab:
            out.append((pos, wt, mut, lab))
    return out


def auc(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return None
    c = sum(1 for p in pos_scores for n in neg_scores if p > n)
    t = sum(1 for p in pos_scores for n in neg_scores if p == n)
    return (c + 0.5 * t) / (len(pos_scores) * len(neg_scores))


def test_gene(gene, acc, max_each=18, seed=0):
    random.seed(seed)
    vs = fetch_labeled_variants(acc)
    ben = [v for v in vs if v[3] == "benign"]
    pat = [v for v in vs if v[3] == "path"]
    if len(ben) < 3 or len(pat) < 3:
        return {"gene": gene, "skipped": f"too few labels (path {len(pat)}, benign {len(ben)})"}
    # balance: all benign + an equal-ish sample of pathogenic (keep CPU bounded)
    random.shuffle(pat)
    pat = pat[:max(max_each, len(ben))][:max_each]
    ben = ben[:max_each]
    pscore, nscore = [], []
    for pos, wt, mut, _ in pat:
        r = me.variant_effect(gene, pos, wt, mut, acc=acc)
        if r["functional_score"] is not None:
            pscore.append(r["functional_score"])
    for pos, wt, mut, _ in ben:
        r = me.variant_effect(gene, pos, wt, mut, acc=acc)
        if r["functional_score"] is not None:
            nscore.append(r["functional_score"])
    a = auc(pscore, nscore)
    return {"gene": gene, "n_pathogenic": len(pscore), "n_benign": len(nscore),
            "mean_func_pathogenic": round(sum(pscore) / len(pscore), 3) if pscore else None,
            "mean_func_benign": round(sum(nscore) / len(nscore), 3) if nscore else None,
            "AUC_pathogenic_vs_benign": round(a, 3) if a else None}


def run():
    # genes ADDED this session (absent from the original model) — the honest 'never seen' set
    genes = [("PIK3CA", "P42336"),   # cancer (added P0)
             ("PAH", "P00439"),      # phenylketonuria (added P0)
             ("GBA", "P04062"),      # Gaucher / Parkinson (added P0)
             ("TYK2", "P29597"),     # autoimmune (added P0)
             ("HBB", "P68871")]      # sickle / thalassaemia (added P0)
    results = []
    for g, a in genes:
        try:
            r = test_gene(g, a)
        except Exception as e:
            r = {"gene": g, "error": str(e)[:60]}
        results.append(r); print("  ", r)
    good = [r for r in results if r.get("AUC_pathogenic_vs_benign")]
    macro = round(sum(r["AUC_pathogenic_vs_benign"] for r in good) / len(good), 3) if good else None
    out = {"results": results, "macro_AUC": macro,
           "verdict": ("engine separates pathogenic from benign on unseen variants of genes it never had -> "
                       "genuine generalisation" if (macro and macro > 0.65) else
                       "weak/no separation on held-out variants" if macro else "insufficient labels")}
    json.dump(out, open(f"{OUT}/blind_variant_test.json", "w"), indent=2)
    print("=" * 70)
    print(f"BLIND VARIANT TEST — genes added this session, variants never seen")
    print(f"  macro AUC (pathogenic vs benign): {macro}")
    print(f"  -> {out['verdict']}")
    return out


if __name__ == "__main__":
    run()
