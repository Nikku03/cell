"""ADD STRAND AND A DECLARED GENOME BUILD TO cell_complete.json.

WHY.  The integration assessment found that three of five proposed chromatin-to-biology couplings die at
the same two absences. `strand` is absent from all 16,492 gene records, which kills any SIGNED mechanism --
twin-domain supercoiling is entirely about the asymmetry between behind and ahead of a polymerase, and
without strand there is no behind. And no genome build is declared anywhere in the file that carries the
16,380 coordinates, so those coordinates cannot be joined to an external track with confidence. Both are
cheap to fix and both gate work that is otherwise blocked.

THE SOURCE.  outputs/orphan/science_data/genes.parquet, 86,411 rows, every one carrying strand,
chromosome_name, start_position and end_position, and -- unusually for this repo -- carrying its own
provenance: source "Ensembl BioMart (mcp-biomart connector)", source_version "Ensembl human genes
GRCh38.p14", plus a retrieved_on date.

WHAT CAN AND CANNOT BE CLAIMED ABOUT THE BUILD.  This distinction is the whole point of the file.

    genes.parquet's build is DECLARED. It says GRCh38.p14 in its own source_version field.
    cell_complete's build is INFERRED. No module in this repository writes `tss` -- every one of them
        reads it -- so its provenance is upstream and unrecorded, and the only available evidence is
        whether its coordinates are CONSISTENT with a GRCh38 source.

So the field written is `genome.build_evidence`, not a bare `genome.build`, and it carries the numbers the
inference rests on. Calling it a declaration would be asserting something this repository cannot support.

THE EVIDENCE, and why 96 bp settles it.  Chromosome agrees for 16,355 of 16,357 symbol-matched genes
(100.0%) and the median |TSS - annotated TSS| is 96 bp with 91.8% inside 10 kb. A build mismatch does not
look like that: lifting hg19 to GRCh38 moves genes by 1e4 to 1e7 bp, so a 96 bp median rules it out. What
96 bp DOES indicate is a different transcript annotation -- gene start against canonical-transcript TSS --
which is why only 6.7% match exactly and why `ens_tss` is written as a SEPARATE field rather than
overwriting `tss`.

TWO CONTROLS, because a join that cannot fail proves nothing:
    PERMUTED JOIN   shuffle the symbol-to-coordinate mapping and recompute the agreement. If a random
                    pairing agrees as well as the true one, the join is meaningless and so is everything
                    built on it.
    INDEPENDENT STRAND  outputs/orphan/invivo/crispr_egpairs.tsv carries strandGene for 10,412 of 10,412
                    pairs from a completely different source. Strand assigned from the parquet is checked
                    against it on the overlap. This is the doctrine's "two independent computations" rule
                    applied to a data patch.

PREDECLARED, before any number:
    permuted chromosome agreement collapses and independent strand agrees at >99%
        -> the join is real and the patch can be trusted.
    permuted agreement is comparable to the true join
        -> symbols are matching by coincidence, the patch is invalid, and nothing is written.
    independent strand disagrees on more than 1% of the overlap
        -> one of the two sources is wrong about direction and the patch is withheld until it is known
           which. A wrong strand is worse than no strand, because a signed mechanism would then be
           silently backwards.

-> outputs/orphan/cell_coords_audit.json   (and patches outputs/orphan/cell_complete.json in place)
"""
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = OUT / "orphan" / "cell_complete.json"
GENES = OUT / "orphan" / "science_data" / "genes.parquet"
CRISPR = OUT / "orphan" / "invivo" / "crispr_egpairs.tsv"
SEED = 8891
NEAR = 10_000          # bp; a real build mismatch moves genes far further than this


def load_parquet():
    import pyarrow.parquet as pq
    t = pq.read_table(GENES).to_pydict()
    by, meta = {}, {}
    for i in range(len(t["ensembl_gene_id"])):
        ch = t["chromosome_name"][i]
        if ch is None or len(str(ch)) > 2:      # primary assemblies only; scaffolds have long names
            continue
        rec = {"chrom": f"chr{ch}", "start": int(t["start_position"][i]),
               "end": int(t["end_position"][i]), "strand": t["strand"][i],
               "ensembl": t["ensembl_gene_id"][i], "biotype": t["gene_biotype"][i]}
        for key in (t["hgnc_symbol"][i], t["external_gene_name"][i]):
            if key and key not in by:
                by[key] = rec
        if not meta:
            meta = {"source": t["source"][i], "source_version": t["source_version"][i],
                    "retrieved_on": str(t["retrieved_on"][i])}
    return by, meta, len(t["ensembl_gene_id"])


def agreement(genes, by, rng=None):
    """Chromosome agreement and TSS offset for a symbol join. With rng, the mapping is PERMUTED, which is
    the control: a join that survives shuffling is matching on nothing."""
    keys = list(by)
    if rng is not None:
        vals = [by[k] for k in keys]
        rng.shuffle(vals)
        by = dict(zip(keys, vals))
    hit = chrom_ok = exact = near = 0
    diffs = []
    for g in genes:
        nm, tss, ch = g.get("name"), g.get("tss"), g.get("chrom")
        if nm not in by or tss in (None, ""):
            continue
        try:
            tss = int(float(tss))
        except (TypeError, ValueError):
            continue
        r = by[nm]
        hit += 1
        if ch and str(ch) == r["chrom"]:
            chrom_ok += 1
        exp = r["start"] if r["strand"] == "+" else r["end"]
        d = abs(tss - exp)
        if d == 0:
            exact += 1
        if d <= NEAR:
            near += 1
        diffs.append(d)
    return {"matched": hit, "chrom_agree": chrom_ok, "exact_tss": exact, "within_10kb": near,
            "median_offset_bp": float(statistics.median(diffs)) if diffs else float("nan"),
            "chrom_frac": chrom_ok / max(hit, 1), "near_frac": near / max(hit, 1)}


def independent_strand(by):
    """crispr_egpairs.tsv carries strandGene from a different source entirely. Check the overlap."""
    if not CRISPR.exists():
        return None
    seen = {}
    with open(CRISPR) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            g, s = row.get("measuredGeneSymbol"), row.get("strandGene")
            if g and s in ("+", "-"):
                seen.setdefault(g, s)
    agree = dis = 0
    bad = []
    for g, s in seen.items():
        if g in by:
            if by[g]["strand"] == s:
                agree += 1
            else:
                dis += 1
                bad.append(g)
    return {"overlap": agree + dis, "agree": agree, "disagree": dis,
            "frac": agree / max(agree + dis, 1), "examples": sorted(bad)[:8],
            "source_genes": len(seen)}


def main():
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("ADD STRAND AND A GENOME BUILD TO cell_complete.json")
    report("=" * 100)
    report("  strand is absent from all 16,492 gene records, which kills every SIGNED mechanism -- there")
    report("  is no 'behind the polymerase' without it. And no build is declared in the file that carries")
    report("  the coordinates, so they cannot be joined to an external track with confidence.")

    by, meta, n_parquet = load_parquet()
    report(f"\n  SOURCE  {GENES.relative_to(ROOT)}")
    report(f"    {n_parquet} rows, {len(by)} symbols on primary assemblies")
    for k, v in meta.items():
        report(f"    {k:<16}{v}")

    D = json.load(open(CELL))
    genes = D["genes"]
    report(f"\n  TARGET  {CELL.relative_to(ROOT)}   {len(genes)} gene records")

    true = agreement(genes, by)
    perm = agreement(genes, by, rng=rng)
    report(f"\n  CONTROL A -- is the symbol join real? permuted mapping must collapse")
    report(f"    {'':4}{'statistic':<26}{'TRUE':>12}{'PERMUTED':>12}")
    report(f"    {'':4}{'matched by symbol':<26}{true['matched']:>12}{perm['matched']:>12}")
    report(f"    {'':4}{'chromosome agrees':<26}{true['chrom_frac']:>11.1%}{perm['chrom_frac']:>12.1%}")
    report(f"    {'':4}{'TSS within 10 kb':<26}{true['near_frac']:>11.1%}{perm['near_frac']:>12.1%}")
    report(f"    {'':4}{'median |offset| bp':<26}{true['median_offset_bp']:>12,.0f}"
           f"{perm['median_offset_bp']:>12,.0f}")
    join_real = true["chrom_frac"] > 0.95 and perm["chrom_frac"] < 0.30

    ind = independent_strand(by)
    report(f"\n  CONTROL B -- strand against an INDEPENDENT source (crispr_egpairs.tsv strandGene)")
    if ind is None:
        report("    source absent; strand cannot be cross-checked and the patch is withheld")
        strand_ok = False
    else:
        report(f"    {ind['overlap']} genes overlap; agree {ind['agree']}, disagree {ind['disagree']} "
               f"({ind['frac']:.2%})")
        if ind["examples"]:
            report(f"    disagreements: {', '.join(ind['examples'])}")
        strand_ok = ind["frac"] >= 0.99

    report(f"\n  THE BUILD -- what can be claimed")
    report(f"    genes.parquet DECLARES its build: {meta.get('source_version')}")
    report(f"    cell_complete does NOT. No module in this repo writes `tss`; every one reads it, so its")
    report(f"    provenance is upstream and unrecorded. The build is therefore INFERRED from consistency:")
    report(f"      chromosome agrees {true['chrom_frac']:.1%}, median TSS offset "
           f"{true['median_offset_bp']:,.0f} bp, {true['near_frac']:.1%} within 10 kb")
    report(f"    A build mismatch moves genes by 1e4-1e7 bp. {true['median_offset_bp']:,.0f} bp does not")
    report(f"    look like that; it looks like a different transcript annotation (gene start vs canonical")
    report(f"    TSS), which is why only {true['exact_tss']/max(true['matched'],1):.1%} match exactly and")
    report(f"    why the annotated TSS is written as a SEPARATE field rather than overwriting `tss`.")

    if not (join_real and strand_ok):
        report(f"\n  GATE FAILED -- join_real={join_real}, strand_ok={strand_ok}. NOTHING IS WRITTEN.")
        report("  A wrong strand is worse than no strand: a signed mechanism would run silently backwards.")
        json.dump({"test": "cell_coords", "written": False, "true": true, "permuted": perm,
                   "independent_strand": ind, "source_meta": meta, "log": log},
                  open(OUT / "orphan" / "cell_coords_audit.json", "w"), indent=2)
        return 1

    # ---- the patch, additive only --------------------------------------------------------------------
    added = miss = 0
    for g in genes:
        r = by.get(g.get("name"))
        if not r:
            g["strand"] = None
            miss += 1
            continue
        g["strand"] = r["strand"]
        g["gene_start"] = r["start"]
        g["gene_end"] = r["end"]
        g["ens_tss"] = r["start"] if r["strand"] == "+" else r["end"]
        added += 1
    D["genome"] = {
        "build_evidence": "GRCh38",
        "status": "INFERRED from consistency with a declared-GRCh38 source, not declared by this file",
        "declared_by": str(GENES.relative_to(ROOT)),
        "declared_build": meta.get("source_version"),
        "source": meta.get("source"), "retrieved_on": meta.get("retrieved_on"),
        "evidence": {"symbol_matched": true["matched"], "chrom_agree_frac": true["chrom_frac"],
                     "median_tss_offset_bp": true["median_offset_bp"],
                     "within_10kb_frac": true["near_frac"],
                     "exact_tss_frac": true["exact_tss"] / max(true["matched"], 1),
                     "permuted_chrom_agree_frac": perm["chrom_frac"]},
        "caveat": ("`tss` is NOT from this source and its provenance is unrecorded; `ens_tss` is the "
                   "GRCh38.p14 annotated TSS and the two differ by a median of "
                   f"{true['median_offset_bp']:,.0f} bp"),
        "fields_added": ["strand", "gene_start", "gene_end", "ens_tss"],
        "coverage": {"with_strand": added, "without": miss, "total": len(genes)},
    }
    json.dump(D, open(CELL, "w"))
    report(f"\n  PATCHED  strand + gene_start + gene_end + ens_tss on {added} of {len(genes)} records "
           f"({added/len(genes):.1%}); {miss} left null rather than guessed")
    report(f"  PATCHED  top-level `genome` block with the evidence above")

    man = RM.manifest(inputs=[str(GENES), str(CRISPR)], available=len(genes), used=added,
                      selection="filtered", seed=SEED,
                      controls=["permuted symbol join", "independent strand (crispr_egpairs)"],
                      note="genes matched to GRCh38.p14 by HGNC symbol then Ensembl external name")
    RM.report(man, emit=report)
    json.dump({"test": "cell_coords", "written": True, "manifest": man, "true": true, "permuted": perm,
               "independent_strand": ind, "source_meta": meta, "genome": D["genome"], "log": log},
              open(OUT / "orphan" / "cell_coords_audit.json", "w"), indent=2)
    report(f"\n  -> {OUT/'orphan'/'cell_coords_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
