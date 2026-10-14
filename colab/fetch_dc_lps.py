"""FETCH the ENCODE dendritic-cell LPS time course -- the replication set for loop 191d's clock.

WHY THIS SERIES AND NOT ANOTHER. Loop 191d measured that promoter accessibility reaches half its
plateau 48 minutes before the mRNA does, over 1,310 responding genes at p 6.4e-58, surviving the
magnitude control in all three terciles. That is the only result in this project with a fourth
dimension in it, and it rests on one drug, one receptor and one cell line. A search of every
released ENCODE experiment carrying a treatment duration returns four systems with at least four
matched accessibility and RNA timepoints, and this one is the fair test:

    A549 + dexamethasone      11 shared points   the series 191d already used
    dendritic cell + LPS       4 shared points   THIS ONE
    CD4 T cell activation      4 shared points   DNase 8 points but RNA only 4, and 7-14 DAY spacing
    K562 + 7 chromatin drugs   4 shared points   see below

The K562 panel is tempting because K562 is where this project's entire enhancer arc lives, and it
even ships a DMSO vehicle arm. It is rejected as the PRIMARY test because six of its seven drugs --
Vorinostat, Panobinostat, JQ1, MB-3, GSK J4 and Galeterone -- are HDAC, BET, acetyltransferase and
demethylase inhibitors. They act ON chromatin directly, so "accessibility moves before
transcription" is close to tautological there: it restates the drug's mechanism. Its 4/12/24/48
HOUR grid is also far too coarse to resolve a 48-minute lead. It is worth a later loop as a CAUSAL
test -- force the chromatin layer and see whether transcription follows -- but it cannot replicate
an observational claim.

What makes the dendritic-cell series the right one:

    a NATURAL stimulus. LPS through TLR4 is an innate immune response, not a nuclear receptor and
    not a chromatin drug. Nothing about it presupposes the answer.
    a DIFFERENT cell type. Primary dendritic cells against A549 lung carcinoma.
    a DIFFERENT assay. ATAC-seq against DNase-seq, so a positive is not an artefact of one
    accessibility protocol.
    a COMPARABLE window. 1, 2, 4 and 6 hours sits inside 191d's usable 30-720 minute window, which
    matters because a 48-minute lead is invisible on the K562 4-48 hour grid.
    ONE LAB. All 59 experiments are Manuel Garber's, UMass. Loop 191c had to discard the first 25
    minutes of the A549 series because of a batch discontinuity between submissions, so
    single-lab provenance is checked here BEFORE anything is downloaded rather than discovered
    afterwards.

WHAT IS TAKEN. ATAC-seq peaks and total RNA-seq gene quantifications, one experiment per timepoint,
with the same rules the A549 fetcher enforces: GRCh38 only, a single peak output_type across the
whole course (a mixed one puts a processing difference on the axis being measured), and gene_id/TPM
read by column NAME rather than position. The RNA TSVs are parsed and deleted; only the TPM matrix
is kept.

ATAC also carries a 30-minute point that RNA does not. It is fetched anyway and the loop decides
what to do with it, because an accessibility measurement before the first expression measurement is
exactly the kind of asymmetry that could manufacture a lead, and the loop must be able to see it.

-> {scratchpad}/dclps/rna.npz and {scratchpad}/dclps/ATAC/*.bed.gz
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_gr_timecourse as G              # noqa: E402

CELL = "dendritic cell"
DRUG = "lipopolysaccharide"
LAB = "Manuel Garber, UMass"
DIR = G.SP / "dclps"


def index():
    """Released experiments for this cell and treatment, with the duration in minutes.

    Provenance is filtered here and the count of what was dropped is printed, so a second lab
    appearing in this series later becomes visible rather than silently averaged in."""
    q = ("/search/?type=Experiment&status=released&limit=all&format=json"
         f"&biosample_ontology.term_name={CELL.replace(' ', '+')}"
         "&field=accession&field=assay_title&field=target.label&field=lab.title"
         "&field=replicates.library.biosample.treatments.duration"
         "&field=replicates.library.biosample.treatments.duration_units"
         "&field=replicates.library.biosample.treatments.treatment_term_name")
    rows, dropped = [], 0
    for r in G.api(q).get("@graph", []):
        lab = (r.get("lab") or {}).get("title", "")
        durs = set()
        for rep in r.get("replicates", []):
            for t in ((rep.get("library") or {}).get("biosample") or {}).get("treatments", []) or []:
                if t.get("duration") and DRUG.lower() in str(t.get("treatment_term_name", "")).lower():
                    durs.add(float(t["duration"]) * G.UNITS.get(t.get("duration_units"), 0.0))
        if not durs:
            continue
        if lab != LAB:
            dropped += 1
            continue
        rows.append({"acc": r["accession"], "assay": r.get("assay_title"),
                     "target": (r.get("target") or {}).get("label"),
                     "min": min(durs), "lab": lab})
    print(f"  provenance filter: kept {len(rows)} experiments from '{LAB}', "
          f"dropped {dropped} from other labs")
    return rows


def main():
    G.DIR = DIR                      # the shared helpers write under G.DIR
    DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("FETCH -- ENCODE dendritic cell + LPS time course (the replication set for loop 191d)")
    print("=" * 100)
    rows = index()
    if not rows:
        raise SystemExit("no experiments matched -- the search fields may have changed")
    seen = defaultdict(set)
    for r in rows:
        seen[r["assay"]].add(int(r["min"]))
    for a in sorted(seen):
        print(f"  {a:22s} timepoints {sorted(seen[a])}")

    man = {"cell": CELL, "drug": DRUG, "lab": LAB, "assembly": "GRCh38", "peaks": {}}
    atac = G.fetch_peaks(rows, "ATAC-seq", None, "ATAC")
    man["peaks"]["ATAC"] = atac
    mb = sum(v["size"] for v in atac.values()) / 1e6
    print(f"  ATAC    {len(atac):2d} timepoints  {sorted(atac)}  ({mb:.1f} MB)")

    print("\n  total RNA-seq gene quantifications (TPM kept, TSVs deleted)")
    rna_rows = [r for r in rows if r["assay"] == "total RNA-seq"]
    cols, genes = G.fetch_rna(rna_rows, assay="total RNA-seq")
    if not cols:
        raise SystemExit("no RNA-seq gene quantifications retrieved")
    gidx = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(cols), len(genes)), dtype=np.float32)
    for j, c in enumerate(cols):
        for g, v in c["tpm"].items():
            i = gidx.get(g)
            if i is not None:
                M[j, i] = v
    np.savez_compressed(DIR / "rna.npz", tpm=M, genes=np.array(genes),
                        mins=np.array([c["min"] for c in cols]),
                        reps=np.array([c["rep"] for c in cols]),
                        exps=np.array([c["exp"] for c in cols]),
                        files=np.array([c["file"] for c in cols]))
    man["rna"] = {"n_columns": len(cols), "n_genes": len(genes),
                  "timepoints": sorted({c["min"] for c in cols}),
                  "columns": [{k: c[k] for k in ("min", "exp", "file", "rep")} for c in cols]}
    json.dump(man, open(DIR / "manifest.json", "w"), indent=1)

    tp = sorted({c["min"] for c in cols})
    print(f"\n  {len(cols)} RNA columns x {len(genes):,} genes -> {DIR/'rna.npz'}")
    print(f"  RNA timepoints:  {tp}")
    print(f"  ATAC timepoints: {sorted(atac)}")
    shared = sorted(set(int(t) for t in tp) & set(atac))
    print(f"  SHARED (both assays): {len(shared)}  {shared}")
    print(f"  ATAC-only points (fetched, the loop decides): "
          f"{sorted(set(atac) - set(int(t) for t in tp))}")
    print(f"  -> {DIR/'manifest.json'}")


if __name__ == "__main__":
    main()
