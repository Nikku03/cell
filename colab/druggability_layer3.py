"""LAYER 3 — structural druggability. For each candidate target from Layers 1-2, fetch protein family +
subcellular location + structure (UniProt REST; the model itself has almost no structure data), then decide
whether the REQUIRED direction (disable / activate) is achievable by that protein family's drug modality.

Druggability logic (family -> modality -> can we achieve the required direction):
  secreted cytokine / ligand      -> neutralizing antibody / trap        -> DISABLE achievable   (HIGH)
  cell-surface receptor           -> antagonist antibody or small molec  -> DISABLE achievable   (HIGH)
  kinase / enzyme                 -> small-molecule inhibitor            -> DISABLE achievable   (HIGH)
  nuclear receptor                -> inverse agonist (small molecule)    -> DISABLE achievable   (MED-HIGH)
  transcription factor (other)    -> no ligand pocket                    -> DISABLE hard         (LOW)

Reads outputs/orphan/psoriasis_target_layer12.json -> writes psoriasis_target_layer3.json + final call.
"""
import json, urllib.request, urllib.parse
from pathlib import Path

OUT = Path("outputs/orphan")
UA = {"User-Agent": "Mozilla/5.0 (cell-model research)"}


def uniprot(gene):
    q = f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true"
    url = ("https://rest.uniprot.org/uniprotkb/search?query=" + urllib.parse.quote(q) +
           "&fields=accession,protein_name,cc_function,cc_subcellular_location,keyword,"
           "cc_similarity,xref_pdb,xref_alphafolddb&format=json&size=1")
    try:
        req = urllib.request.Request(url, headers=UA)
        d = json.load(urllib.request.urlopen(req, timeout=40))
        if not d.get("results"):
            return None
        r = d["results"][0]
        kws = [k["name"] for k in r.get("keywords", [])]
        fam = ""
        loc = []
        for c in r.get("comments", []):
            if c.get("commentType") == "SIMILARITY":
                fam = c["texts"][0]["value"]
            if c.get("commentType") == "SUBCELLULAR LOCATION":
                for s in c.get("subcellularLocations", []):
                    loc.append(s.get("location", {}).get("value", ""))
        pdb = [x["id"] for x in r.get("uniProtKBCrossReferences", []) if x["database"] == "PDB"]
        return dict(acc=r["primaryAccession"], keywords=kws, family=fam,
                    location=sorted(set(loc)), n_pdb=len(pdb), pdb=pdb[:4])
    except Exception as e:
        return dict(error=str(e))


def classify(u):
    """family/location -> (protein_class, modality, direction_class druggable for DISABLE)."""
    kw = set(k.lower() for k in u.get("keywords", []))
    loc = " ".join(u.get("location", [])).lower()
    fam = (u.get("family") or "").lower()
    secreted = "secreted" in loc or "cytokine" in kw or "il-6 superfamily" in fam or "il-17 family" in fam
    membrane_rec = ("receptor" in kw and ("membrane" in loc or "cell membrane" in loc)) or "receptor" in fam
    kinase = "kinase" in kw or "transferase" in kw or "tyrosine-protein kinase" in fam
    nuc_recept = "nuclear receptor" in fam or ("receptor" in kw and "nucleus" in loc) or "ror" in fam
    tf = ("transcription" in " ".join(kw) or "dna-binding" in kw or "activator" in kw) and "nucleus" in loc
    if secreted:
        return ("secreted cytokine/ligand", "neutralizing antibody / ligand trap", "HIGH")
    if membrane_rec:
        return ("cell-surface receptor", "antagonist antibody or small molecule", "HIGH")
    if kinase:
        return ("kinase/enzyme", "small-molecule inhibitor", "HIGH")
    if nuc_recept:
        return ("nuclear receptor", "inverse agonist (small molecule)", "MED-HIGH")
    if tf:
        return ("transcription factor", "no ligand pocket (PROTAC/PPI only)", "LOW")
    return ("other/unclear", "unknown", "MED")


def main():
    L12 = json.load(open(OUT / "psoriasis_target_layer12.json"))
    if not L12.get("ok"):
        print("Layer 1-2 result not ok; run disease_target_pipeline.py first."); return
    cands = L12["candidates"]
    rows = []
    print("=" * 92)
    print("LAYER 3 — structural druggability of the Layer-1/2 candidates (UniProt family + structure)")
    print("=" * 92)
    print(f"{'gene':8} {'rescue':>6} {'dir':>8} {'class':26} {'modality':38} drug-dir")
    for c in cands:
        u = uniprot(c["gene"])
        if not u or u.get("error"):
            rows.append(dict(c, layer3_error=(u or {}).get("error", "no uniprot")))
            print(f"{c['gene']:8} {c['rescue']:>6.2f} {c['best_dir']:>8}  [uniprot fetch failed: {(u or {}).get('error','')[:40]}]")
            continue
        pclass, modality, drugdir = classify(u)
        # actionable = strong rescue AND the required direction is druggable
        actionable = (c["rescue"] >= 0.6) and (drugdir in ("HIGH", "MED-HIGH")) and (c["best_dir"] == "disable")
        row = dict(gene=c["gene"], rescue=c["rescue"], best_dir=c["best_dir"], influence=c["influence"],
                   protein_class=pclass, modality=modality, family=u["family"],
                   location=u["location"], n_pdb=u["n_pdb"], druggable_for_direction=drugdir,
                   actionable_target=actionable)
        rows.append(row)
        print(f"{c['gene']:8} {c['rescue']:>6.2f} {c['best_dir']:>8} {pclass:26} {modality:38} {drugdir}"
              + ("  <== ACTIONABLE" if actionable else ""))
    # final call
    actionable = [r for r in rows if r.get("actionable_target")]
    actionable.sort(key=lambda r: -r["rescue"])
    print("\nFINAL TARGET CALL (strong phenotype rescue AND druggable in the required direction):")
    for r in actionable:
        print(f"   {r['gene']:8} rescue {r['rescue']:.2f} · {r['best_dir']} · {r['protein_class']} "
              f"· {r['modality']} · {r['n_pdb']} PDB structures")
    json.dump(dict(disease=L12["disease"], layer3=rows,
                   final_targets=[r["gene"] for r in actionable]),
              open(OUT / "psoriasis_target_layer3.json", "w"), indent=2)
    print("\n-> outputs/orphan/psoriasis_target_layer3.json")


if __name__ == "__main__":
    main()
