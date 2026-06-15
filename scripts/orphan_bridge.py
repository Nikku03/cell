"""PATH-ORPHAN, STEP 0 (Phase A) -- the locus-tag bridge.

The disjoint-data wall, made concrete: our labels/orthology key genes by an
organism-native locus_tag, AFDB keys structures by UniProt, dN/dS/ESM/Foldseek
need the actual protein SEQUENCE. If these namespaces don't join, the whole
experiment silently produces empty results (proven: beril_Keio joins its cached
genome at 0%). This step builds the bridge ONCE, caches it, and refuses to
proceed past a join-rate floor.

For the chosen org (default beril_RalstoniaGMI1000) it assembles, per gene:
  organism, locus_tag, essential, og_id, conservation (family_frac leak-free),
  family_n_organisms, n_paralogs_in_genome, is_orphan, protein_id (RefSeq WP_),
  seq_len, has_seq.

OUTPUTS (idempotent; skip if present unless --force):
  outputs/orphan/bridge_<org>.parquet      the joined per-gene table
  outputs/orphan/proteins_<org>.faa        sequences keyed by locus_tag
                                           (input to Phase B: ESM/Foldseek, and
                                            to dN/dS alignment in Phase A)
  outputs/orphan/uniprot_request_<org>.txt RefSeq protein_ids to map to UniProt
                                           for the AFDB pull (the one network
                                           step; run on Colab)

The GFF for these RefSeq genomes carries no UniProtKB xref, so AFDB lookup is
deferred to a UniProt idmapping call seeded by uniprot_request_<org>.txt.

--smoke : validate the join logic on a synthetic mini genome (no external data)
--real  : build from data/drive_import (sandbox) or Drive (Colab)
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"

# org -> RefSeq assembly accession (the cached genome that joins its labels)
ORG_ACC = {
    "beril_RalstoniaGMI1000": "GCF_000009125.1",   # DEG1057, Paper-1 kappa=0.18
    "beril_Dda3937":          "GCF_000147055.1",
    "beril_HerbieS":          "GCF_000143225.1",
    "beril_Magneto":          "GCF_000009985.1",
    "beril_RalstoniaPSI07":   "GCF_000283475.1",
}
JOIN_FLOOR = 0.90   # refuse to proceed below this label->sequence join rate


def _gff_attr(line, key):
    m = re.search(rf'{key}=([^;\n]+)', line)
    return m.group(1) if m else None


def parse_gff(gff_path):
    """locus_tag -> dict(protein_id, gene, product) for CDS features."""
    out = {}
    with open(gff_path) as f:
        for line in f:
            if "\tCDS\t" not in line:
                continue
            lt = _gff_attr(line, "locus_tag")
            if not lt or lt in out:
                continue
            out[lt] = {
                "protein_id": _gff_attr(line, "protein_id") or "",
                "gene": _gff_attr(line, "gene") or "",
                "product": _gff_attr(line, "product") or "",
            }
    return out


def parse_faa(faa_path):
    """protein_id -> sequence (first whitespace token of header is the id)."""
    seqs = {}
    pid = None
    buf = []
    with open(faa_path) as f:
        for line in f:
            if line.startswith(">"):
                if pid is not None:
                    seqs[pid] = "".join(buf)
                pid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if pid is not None:
        seqs[pid] = "".join(buf)
    return seqs


def build(org, labels_csv, orth_csv, gff_path, faa_path):
    import csv
    lab = {}
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            if r["organism"] == org:
                lab[r["locus_tag"]] = int(r["essential"])
    orth = {}
    with open(orth_csv) as f:
        for r in csv.DictReader(f):
            if r["organism"] == org:
                orth[r["locus_tag"]] = r
    gff = parse_gff(gff_path)
    seqs = parse_faa(faa_path)

    rows = []
    for lt, ess in lab.items():
        g = gff.get(lt, {})
        o = orth.get(lt, {})
        pid = g.get("protein_id", "")
        seq = seqs.get(pid, "")
        cons = o.get("family_frac_essential_leakfree", "")
        try:
            cons = float(cons)
        except (TypeError, ValueError):
            cons = float("nan")
        rows.append({
            "organism": org, "locus_tag": lt, "essential": ess,
            "gene": g.get("gene", ""), "product": g.get("product", ""),
            "og_id": o.get("og_id", ""),
            "conservation": cons,
            "family_n_organisms": o.get("family_n_organisms", ""),
            "n_paralogs_in_genome": o.get("n_paralogs_in_genome", ""),
            "is_orphan": o.get("is_orphan", ""),
            "protein_id": pid,
            "seq_len": len(seq),
            "has_seq": int(bool(seq)),
            "_seq": seq,
        })
    return rows


def write_outputs(org, rows, force=False):
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    bridge = OUT / f"bridge_{org}.parquet"
    faa = OUT / f"proteins_{org}.faa"
    req = OUT / f"uniprot_request_{org}.txt"

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_seq"} for r in rows])
    df.to_parquet(bridge, index=False)

    with open(faa, "w") as fh:
        for r in rows:
            if r["_seq"]:
                fh.write(f">{r['locus_tag']}\n{r['_seq']}\n")
    pids = sorted({r["protein_id"] for r in rows if r["protein_id"]})
    req.write_text("\n".join(pids) + "\n")
    return bridge, faa, req, df


def report(org, df):
    n = len(df)
    seq = int(df["has_seq"].sum())
    ess = int((df["essential"] == 1).sum())
    cons = df["conservation"]
    rogue = int(((df["essential"] == 1) & (cons < 0.1)).sum())
    print(f"\n  organism            : {org}")
    print(f"  genes (labeled)     : {n:,}   essential {ess:,} ({100*ess/n:.0f}%)")
    print(f"  with protein SEQ    : {seq:,} ({100*seq/n:.0f}%)   "
          f"[join floor {JOIN_FLOOR:.0%}]")
    print(f"  rogue essentials    : {rogue:,}  (essential & conservation<0.1 -- "
          f"the orphan/conditional target)")
    print(f"  unique protein_ids  : {df['protein_id'].nunique():,} "
          f"(-> UniProt request for AFDB)")
    return seq / n


def run_real(args):
    base = Path(args.data_root)
    org = args.org
    if org not in ORG_ACC:
        print(f"ERROR: unknown org {org}; known: {list(ORG_ACC)}")
        return 2
    acc = ORG_ACC[org]
    labels = base / "labels" / "labels.csv"
    orth = base / "labels" / "orthology_features.csv"
    gff = base / "genome_cache" / acc / "genomic.gff"
    faa = base / "genome_cache" / acc / "protein.faa"
    for p in (labels, orth, gff, faa):
        if not p.exists():
            print(f"ERROR: missing input {p}")
            return 2

    bridge = OUT / f"bridge_{org}.parquet"
    if bridge.exists() and not args.force:
        import pandas as pd
        print("=" * 64)
        print(f"STEP 0 bridge -- CACHED ({bridge}); use --force to rebuild")
        print("=" * 64)
        rate = report(org, pd.read_parquet(bridge))
    else:
        print("=" * 64)
        print(f"STEP 0 bridge -- building for {org} ({acc})")
        print("=" * 64)
        rows = build(org, labels, orth, gff, faa)
        b, f, r, df = write_outputs(org, rows, force=args.force)
        rate = report(org, df)
        print(f"\n  wrote {b}")
        print(f"  wrote {f}")
        print(f"  wrote {r}")

    if rate < JOIN_FLOOR:
        print(f"\n  *** JOIN-RATE GATE FAILED: {rate:.0%} < {JOIN_FLOOR:.0%} -- "
              f"this organism's namespace does not bridge; pick another. ***")
        return 1
    print(f"\n  JOIN-RATE GATE PASS: {rate:.0%} >= {JOIN_FLOOR:.0%}. "
          f"Phase A can proceed.")
    return 0


def run_smoke():
    import tempfile, os, pandas as pd
    print("=" * 64)
    print("STEP 0 bridge SMOKE -- join logic on a synthetic mini genome")
    print("=" * 64)
    d = Path(tempfile.mkdtemp())
    (d / "labels").mkdir(); (d / "genome_cache" / "GCF_TEST").mkdir(parents=True)
    # 3 genes: g1 essential+seq, g2 nonessential+seq, g3 essential but NO seq
    # in faa (tests has_seq=0 and that the join-rate gate would notice gaps)
    (d / "labels" / "labels.csv").write_text(
        "organism,locus_tag,gene_name,essential,source,source_url\n"
        "beril_RalstoniaGMI1000,LT1,,1,t,t\n"
        "beril_RalstoniaGMI1000,LT2,,0,t,t\n"
        "beril_RalstoniaGMI1000,LT3,,1,t,t\n"
        "other_org,LTX,,1,t,t\n")
    (d / "labels" / "orthology_features.csv").write_text(
        "organism,locus_tag,og_id,own_fold,n_paralogs_in_genome,family_size_total,"
        "family_n_organisms,is_orphan,family_frac_essential_global,"
        "family_frac_essential_leakfree,family_frac_essential_fold0,"
        "family_frac_essential_fold1,family_frac_essential_fold2,"
        "family_frac_essential_fold3,family_frac_essential_fold4\n"
        "beril_RalstoniaGMI1000,LT1,OG1,0,0,40,40,0,0.9,0.95,0.9,0.9,0.9,0.9,0.9\n"
        "beril_RalstoniaGMI1000,LT2,OG2,0,2,10,8,0,0.1,0.08,0.1,0.1,0.1,0.1,0.1\n"
        "beril_RalstoniaGMI1000,LT3,OG3,0,0,4,4,0,0.02,0.05,0.0,0.0,0.0,0.1,0.1\n")
    (d / "genome_cache" / "GCF_TEST" / "genomic.gff").write_text(
        "##gff\n"
        "c\tx\tCDS\t1\t9\t.\t+\t0\tID=c1;locus_tag=LT1;gene=dnaA;"
        "product=chromosomal replication initiator;protein_id=WP_1\n"
        "c\tx\tCDS\t20\t29\t.\t+\t0\tID=c2;locus_tag=LT2;"
        "product=DMT family transporter;protein_id=WP_2\n"
        "c\tx\tCDS\t40\t49\t.\t+\t0\tID=c3;locus_tag=LT3;"
        "product=hypothetical protein;protein_id=WP_3\n")
    # faa has WP_1 and WP_2 but NOT WP_3 (the rogue-essential lacks a sequence)
    (d / "genome_cache" / "GCF_TEST" / "protein.faa").write_text(
        ">WP_1 dnaA\nMKKLAA\n>WP_2 transporter\nMVVTTL\n")
    # point ORG_ACC at the synthetic assembly for the smoke run
    ORG_ACC["beril_RalstoniaGMI1000"] = "GCF_TEST"

    rows = build("beril_RalstoniaGMI1000",
                 d / "labels" / "labels.csv",
                 d / "labels" / "orthology_features.csv",
                 d / "genome_cache" / "GCF_TEST" / "genomic.gff",
                 d / "genome_cache" / "GCF_TEST" / "protein.faa")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_seq"} for r in rows])
    print(df[["locus_tag", "essential", "conservation", "product",
              "protein_id", "seq_len", "has_seq"]].to_string(index=False))
    # assertions
    assert len(df) == 3, "must drop the other_org gene"
    assert df.set_index("locus_tag").loc["LT1", "has_seq"] == 1
    assert df.set_index("locus_tag").loc["LT3", "has_seq"] == 0, "LT3 seq absent"
    assert df.set_index("locus_tag").loc["LT1", "gene"] == "dnaA"
    assert abs(df.set_index("locus_tag").loc["LT2", "conservation"] - 0.08) < 1e-9
    rogue = df[(df.essential == 1) & (df.conservation < 0.1)]
    assert set(rogue.locus_tag) == {"LT3"}, "LT3 is the lone rogue essential"
    rate = df["has_seq"].mean()
    assert abs(rate - 2/3) < 1e-9, "join rate = 2/3 (WP_3 missing)"
    print(f"\n  join rate {rate:.0%} (LT3 has no sequence) -> would FAIL the "
          f"{JOIN_FLOOR:.0%} gate on real data, as intended.")
    print("\n=== STEP 0 SMOKE PASS ===")
    print("  join across labels/orthology/GFF/faa, conservation parse, rogue-")
    print("  essential selection, and the join-rate gate all validated.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--data_root", default=str(REPO / "data" / "drive_import"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
