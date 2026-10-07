"""gene_universe — P0: close the gene-coverage gap. The model held 16,492 of ~19,300 HGNC protein-coding genes
(~3,000 missing, including disease-causal ones: globins, TYK2, PIK3CA, SMN1/2, PAH). A cell model that lacks a
disease's causal gene cannot model that disease — Bucket A proved it.

Two functions:
  coverage_report()  — diff the model index against the HGNC approved protein-coding set (alias-aware); enumerate
                       the truly-missing genes and tag every model gene fully-wired / sparse / isolated. Lets the
                       model report its OWN coverage instead of silently returning a partial answer.
  build_additions()  — add missing PRIORITY genes as real nodes with INDEPENDENT edges (STRING physical partners
                       that already exist in the model) — never curated to any answer, so downstream tests stay
                       fair. Writes gene_additions.json, folded into CompleteCell so the genes become queryable
                       and perturbable. Extensible to all ~3,000 in Colab.

-> outputs/orphan/gene_coverage.json , outputs/orphan/gene_additions.json
"""
import os, sys, json, csv, urllib.request, ssl
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/orphan"
_CA = "/root/.ccr/ca-bundle.crt"
HGNC = "data/hgnc.txt"

# disease-priority missing genes to add first (causal genes for major diseases the model currently can't touch)
PRIORITY = [
    "HBB", "HBA1", "HBA2", "HBG1", "HBG2", "HBD", "HBE1",           # haemoglobinopathies (sickle, thalassaemia)
    "TYK2", "PIK3CA",                                              # autoimmune / cancer signalling (Bucket A + MCF7)
    "SMN1", "SMN2",                                                # spinal muscular atrophy
    "PAH", "PKLR", "G6PD",                                         # PKU / pyruvate-kinase / G6PD deficiency
    "LPA", "APOA5",                                                # cardiovascular (Lp(a))
    "F8", "F9",                                                    # haemophilia
    "GBA", "SMPD1", "GLA", "IDUA",                                 # lysosomal storage (Gaucher/Fabry/…)
    "HEXA", "ASPA", "GALC",                                        # neurodegenerative metabolic
    "MYH7", "MYBPC3", "TNNT2",                                     # cardiomyopathy
    "RHO", "ABCA4", "USH2A",                                       # retinal
]


def _ctx():
    return ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None


def _hgnc_sets():
    approved = set(); alias2app = {}
    with open(HGNC, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") != "Approved" or "protein-coding" not in row.get("locus_group", ""):
                continue
            s = row["symbol"]; approved.add(s)
            for fld in ("alias_symbol", "prev_symbol"):
                for a in (row.get(fld) or "").split("|"):
                    if a:
                        alias2app[a] = s
    return approved, alias2app


def coverage_report(C):
    approved, alias2app = _hgnc_sets()
    model = set(C.name)
    missing = []
    for s in approved:
        if s in model:
            continue
        if any(app == s and a in model for a, app in alias2app.items()):
            continue
        missing.append(s)
    # tag every model gene by wiring
    tags = {"fully_wired": 0, "sparse": 0, "isolated": 0}
    for i, n in enumerate(C.name):
        deg = len(C.ppi_adj.get(i, ())) + len(C.causal_out.get(i, [])) + len(getattr(C, "sig_out", {}).get(i, []))
        tags["fully_wired" if deg >= 8 else "sparse" if deg > 0 else "isolated"] += 1
    rep = {"hgnc_protein_coding": len(approved), "in_model": len(approved & model),
           "missing": len(missing), "coverage_pct": round(100 * len(approved & model) / len(approved), 1),
           "wiring": tags, "priority_missing": [g for g in PRIORITY if g in missing]}
    json.dump({"summary": rep, "missing_genes": sorted(missing)}, open(f"{OUT}/gene_coverage.json", "w"))
    return rep


def string_partners(sym, limit=40):
    """INDEPENDENT edges: STRING physical interaction partners (score>=400) for a gene. Not curated to any answer."""
    try:
        url = (f"https://string-db.org/api/tsv/interaction_partners?identifiers={sym}"
               f"&species=9606&limit={limit}&required_score=400")
        d = urllib.request.urlopen(url, timeout=45, context=_ctx()).read().decode().splitlines()
        out = []
        for ln in d[1:]:
            p = ln.split("\t")
            if len(p) >= 6:
                out.append((p[3], float(p[5])))        # partner preferredName, combined score
        return out
    except Exception as e:
        print(f"    STRING fail {sym}: {str(e)[:40]}")
        return []


def uniprot_acc(sym):
    try:
        url = (f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{sym}+AND+organism_id:9606+AND+"
               f"reviewed:true&fields=accession&format=tsv&size=1")
        d = urllib.request.urlopen(url, timeout=30, context=_ctx()).read().decode().splitlines()
        return d[1] if len(d) > 1 else None
    except Exception:
        return None


def build_additions(C, genes=None):
    genes = genes or PRIORITY
    model = set(C.name)
    add = {}
    for g in genes:
        if g in model:
            continue
        partners = [(p, s) for p, s in string_partners(g) if p in model and p != g]
        add[g] = {"uniprot": uniprot_acc(g), "string_partners": partners[:30]}
        print(f"  + {g:8} uniprot={add[g]['uniprot']}  {len(partners)} partners in model")
    json.dump({"genes": add}, open(f"{OUT}/gene_additions.json", "w"))
    return add


if __name__ == "__main__":
    from complete_cell import CompleteCell
    C = CompleteCell()
    print("=" * 66); print("GENE-UNIVERSE COVERAGE"); print("=" * 66)
    rep = coverage_report(C)
    for k, v in rep.items():
        print(f"  {k:20} {v}")
    print("\nadding priority disease genes with independent STRING edges ...")
    build_additions(C)
