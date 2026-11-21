"""What is actually known about the genes this project could not label.

THE FINDING THAT SHAPES THIS MODULE, measured before it was written. 3,783 of the 16,492 genes in
this project's table carry the process label "other" and sit outside the metabolic model. The
obvious reading is that they are obscure. They are not:

    median publications            42   (against 54 for the labelled genes)
    fraction flagged dark        54.2%  (against 23.6%)
    essential in some cell line  10.9%  (against 8.9% -- MORE, not less)
    LOEUF, loss-of-function       0.67  (against 0.78 -- MORE constrained, not less)
    pathway memberships              0   (median; the labelled median is 1)
    compartment              49% cytoplasm, 45% nucleus

These are well-published, unusually constrained, slightly MORE essential genes with no pathway
membership. That combination says the gap is in THIS PROJECT'S annotation pipeline, not in human
knowledge, and a loop that treated them as unknown biology would be measuring its own bookkeeping.

So this module asks the literature first and predicts second. It pulls, for every one of the 3,783,
what UniProt already records: a reviewed function statement, an EC number, GO molecular-function
terms, DNA-binding and zinc-finger features, named domains, keywords, and UniProt's own annotation
score. Only the residue with no molecular-function evidence of any kind is a candidate for
prediction, and how large that residue is, is the loop's first gate.

WHY UNIPROT AND NOT A GO DUMP. A GO term can be inferred electronically from a single homolog, so a
gene can carry twenty GO terms and still have nobody who has ever measured what it does. UniProt
separates reviewed from unreviewed and attaches an annotation score, and it carries the FEATURE
table -- DNA_BIND, ZN_FING, named domains -- which is a structural claim rather than a similarity
claim. Both distinctions matter for deciding what counts as unknown.

Output: colab/data/dark_gene_annotation.json
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "dark_gene_annotation.json"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = ("accession,id,gene_names,protein_name,cc_function,ec,go_f,ft_dna_bind,ft_zn_fing,"
          "ft_domain,keyword,length,annotation_score,reviewed,cc_subcellular_location")
CHUNK = 40

# UniProt keyword and domain vocabulary that asserts a DNA-binding role structurally rather than by
# similarity. Kept explicit so what counts as "is a transcription factor" is auditable.
TF_KEYWORDS = {"DNA-binding", "Transcription", "Transcription regulation", "Activator",
               "Repressor", "Zinc-finger", "Homeobox", "Bromodomain", "Chromatin regulator"}
DBD_WORDS = ["bzip", "bhlh", "hmg box", "homeobox", "fork-head", "forkhead", "ets", "t-box",
             "mads", "rel", "irf", "stat", "runt", "sand", "tea", "cut", "myb", "arid",
             "nuclear receptor", "paired", "pou-specific", "hsf", "gcm", "ap-2", "sox",
             "wrky", "znf", "c2h2"]


def get(url, tries=4, timeout=180):
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers={"accept": "application/json",
                                                     "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=timeout))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def fetch(symbols, report=print, chunk=CHUNK):
    """gene symbol -> the UniProt record, preferring a reviewed human entry."""
    out = {}
    syms = sorted(set(symbols))
    t0 = time.time()
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        q = "(" + " OR ".join(f"gene_exact:{s}" for s in part) + \
            ") AND organism_id:9606 AND reviewed:true"
        url = f"{UNIPROT}?query={urllib.parse.quote(q)}&format=json&size=500&fields={FIELDS}"
        try:
            d = get(url)
        except Exception:
            report(f"      chunk {i}-{i+len(part)} failed")
            continue
        for e in d.get("results", []):
            g = e.get("genes") or [{}]
            name = (g[0].get("geneName") or {}).get("value")
            if not name:
                continue
            key = name.upper()
            if key in out:
                continue
            feats = [(f["type"], str(f.get("description", "")))
                     for f in e.get("features", [])]
            out[key] = dict(
                acc=e.get("primaryAccession"),
                protein=((e.get("proteinDescription") or {}).get("recommendedName") or {})
                        .get("fullName", {}).get("value"),
                score=e.get("annotationScore"),
                length=(e.get("sequence") or {}).get("length"),
                function=next((c["texts"][0]["value"]
                               for c in e.get("comments", [])
                               if c.get("commentType") == "FUNCTION" and c.get("texts")), None),
                ec=[x for x in ((e.get("proteinDescription") or {})
                                .get("recommendedName", {}).get("ecNumbers") or [])],
                keywords=[k.get("name") for k in e.get("keywords", [])],
                go_f=[x.get("id") for x in e.get("uniProtKBCrossReferences", [])
                      if x.get("database") == "GO" and any(
                          p.get("value", "").startswith("F:")
                          for p in x.get("properties", []))],
                dna_bind=[f for f in feats if f[0] == "DNA binding"],
                zn_fing=[f for f in feats if f[0] == "Zinc finger"],
                domains=[f[1] for f in feats if f[0] == "Domain"],
            )
        if (i // chunk + 1) % 10 == 0:
            el = time.time() - t0
            report(f"      {min(i+chunk, len(syms))}/{len(syms)} symbols  [{el:.0f}s]")
    report(f"    UniProt: {len(out):,}/{len(syms):,} symbols matched a reviewed human entry")
    return out


def classify(rec):
    """What UniProt already asserts, split so the loop can ask what is left over.

    `has_function` is the strict test: a reviewed FUNCTION paragraph, an EC number, or a GO
    molecular-function term. Keywords alone do not count -- 'Phosphoprotein' is a keyword and says
    nothing about what a protein does."""
    if rec is None:
        return dict(known=False, is_tf=False, is_enzyme=False, evidence="no reviewed entry")
    ec = bool(rec.get("ec"))
    fn = bool(rec.get("function"))
    go = bool(rec.get("go_f"))
    kw = set(rec.get("keywords") or [])
    dom = " ".join(rec.get("domains") or []).lower()
    is_tf = bool(rec.get("dna_bind") or rec.get("zn_fing")
                 or (kw & {"DNA-binding", "Transcription regulation"})
                 or any(w in dom for w in DBD_WORDS))
    return dict(known=bool(fn or ec or go), is_tf=is_tf, is_enzyme=ec,
                has_function_text=fn, has_ec=ec, has_go_f=go,
                score=rec.get("score"),
                evidence=("FUNCTION" if fn else "") + (" EC" if ec else "") + (" GO-F" if go else ""))


def build(symbols, report=print, force=False):
    if OUT.exists() and not force:
        d = json.load(open(OUT))
        if len(d.get("records", {})) and set(d.get("queried", [])) == set(symbols):
            report(f"    dark-gene annotation from cache: {OUT.name}")
            return d
    recs = fetch(symbols, report)
    cls = {s: classify(recs.get(s.upper())) for s in symbols}
    DATA.mkdir(parents=True, exist_ok=True)
    d = dict(queried=sorted(symbols), records=recs, classification=cls,
             source="UniProtKB reviewed human entries")
    json.dump(d, open(OUT, "w"), indent=1)
    report(f"    -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    return d


if __name__ == "__main__":
    syms = json.load(open("/tmp/unlabelled_genes.json"))
    print("=" * 100)
    print(f"WHAT UNIPROT ALREADY KNOWS ABOUT THE {len(syms):,} GENES THIS PROJECT COULD NOT LABEL")
    print("=" * 100)
    build(syms, print)
