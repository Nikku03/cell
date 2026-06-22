"""Stage 3a (REWRITE) -- extract protein sequences from feba.db itself.

The previous version of this script tried to pull sequences from
data/drive_import/genome_cache/<accession>/protein.faa.  That failed for two
independent reasons:

  1. Most beril orgs in the manifest have no accession (`no_accession`) and
     the few that have one are flagged `low_join_wrong_assembly` -- the RefSeq
     locus_tags don't match the beril locus_tags anyway.
  2. The genome_cache fastas use NCBI protein IDs in headers, not the
     [locus_tag=...] tag the parser was hunting for.

The right source is **feba.db itself**.  feba's locusId is exactly what is in
cond_pairs (both come from feba), so the join is trivial; the only question is
which column in which feba table holds the amino-acid sequence.  We probe at
runtime since the schema varies slightly across releases.

Output: data/drive_import/cond/proteins.fasta (one record per cond_pairs gene)
        data/drive_import/cond/protein_index.json (gene_idx -> {org, locus_tag, len, resolved})
"""
import argparse, json, sqlite3, sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def discover_sequence_column(con):
    """Return (table, locus_col, seq_col, org_col_or_None) for whichever feba
    table actually holds protein sequences."""
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    # tables in priority order: Gene first (most releases), then GeneSeqs, etc.
    candidates = [t for t in ("Gene", "GeneSeqs", "Sequence", "ProteinSequence",
                              "Proteins", "AASeq") if t in tables]
    seq_aliases = ("seq", "aaseq", "sequence", "protein_seq", "AAseq", "aa")
    loc_aliases = ("locusId", "locus_id", "sysName", "name", "Gene")
    org_aliases = ("orgId", "organism", "org_id")
    for t in candidates:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in cur.fetchall()]
        seq = next((c for c in cols if c in seq_aliases), None)
        loc = next((c for c in cols if c in loc_aliases), None)
        org = next((c for c in cols if c in org_aliases), None)
        if seq and loc:
            return t, loc, seq, org
    # last resort: scan every table for a seq-like column
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in cur.fetchall()]
        seq = next((c for c in cols if c in seq_aliases), None)
        loc = next((c for c in cols if c in loc_aliases), None)
        org = next((c for c in cols if c in org_aliases), None)
        if seq and loc:
            return t, loc, seq, org
    return None


def main(cond_npz, cache_path, feba_db, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Z = np.load(cond_npz, allow_pickle=True)
    gene_idx = Z["gene_idx"]
    cache = np.load(cache_path, allow_pickle=True)
    cache_orgs = list(cache["orgs"]); meta_org = cache["meta_org"]; lt = cache["lt"]
    org_by_idx = {gi: (str(cache_orgs[int(meta_org[gi])]), str(lt[gi]))
                  for gi in set(gene_idx.tolist())}
    print(f"unique genes to extract: {len(org_by_idx):,}")

    # build the inverse mapping our_org -> feba orgId via the extractor's map
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_feba_conditions import MANUAL_MAP, map_feba_to_our_orgs

    con = sqlite3.connect(str(feba_db))
    probe = discover_sequence_column(con)
    if probe is None:
        print("!! could not find a (locusId, sequence) column pair in feba.db")
        print("   tables:", [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()])
        sys.exit("schema probe failed")
    seq_table, loc_col, seq_col, org_col = probe
    print(f"sequence source: {seq_table}({loc_col}, {seq_col}"
          f"{', '+org_col if org_col else ''})")

    # collect feba orgIds that map to our orgs
    cur = con.cursor()
    cur.execute(f"SELECT DISTINCT {org_col} FROM {seq_table}" if org_col
                else "SELECT DISTINCT orgId FROM GeneFitness")
    feba_orgs = [r[0] for r in cur.fetchall() if r[0]]
    our_orgs_in_need = set(o for (o, _) in org_by_idx.values())
    org_map = map_feba_to_our_orgs(feba_orgs, our_orgs_in_need)
    inv_map = {ours: fb for fb, ours in org_map.items()}    # our -> feba
    print(f"\nour-org -> feba orgId map ({len(inv_map)} orgs):")
    for ours, fb in sorted(inv_map.items()):
        print(f"  {ours:<26} <- {fb}")

    # pull sequences in bulk per org
    found = {}                                    # (our_org, locus_tag) -> seq
    by_org = defaultdict(set)
    for gi, (org, lt_) in org_by_idx.items():
        by_org[org].add(lt_)
    for our, loci in by_org.items():
        fb = inv_map.get(our)
        if fb is None:
            print(f"  {our:<26}: no feba mapping; skipped")
            continue
        if org_col:
            rows = cur.execute(
                f"SELECT {loc_col}, {seq_col} FROM {seq_table} "
                f"WHERE {org_col} = ?", (fb,)).fetchall()
        else:
            rows = cur.execute(
                f"SELECT {loc_col}, {seq_col} FROM {seq_table}").fetchall()
        hit = 0
        for lid, seq in rows:
            if lid in loci and seq:
                found[(our, str(lid))] = seq
                hit += 1
        print(f"  {our:<26} feba={fb:<14}  target {len(loci):>5}  "
              f"got {hit:>5}  ({100*hit/max(1,len(loci)):>5.1f}%)")

    # emit fasta + index
    out_fa = out_dir / "proteins.fasta"
    index = {}
    with open(out_fa, "w") as f:
        for gi, (org, lt_) in sorted(org_by_idx.items()):
            seq = (found.get((org, lt_), "") or "").replace("*", "")[:1022]
            index[gi] = dict(organism=org, locus_tag=lt_,
                             len=len(seq), resolved=bool(seq))
            if seq:
                f.write(f">{gi}|{org}|{lt_}\n{seq}\n")
    json.dump(index, open(out_dir / "protein_index.json", "w"))
    n_res = sum(1 for v in index.values() if v["resolved"])
    print(f"\nresolved {n_res:,} / {len(index):,} ({100*n_res/len(index):.1f}%)")
    print(f"wrote {out_fa} + protein_index.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond_npz", default="data/drive_import/cond/cond_pairs.npz")
    ap.add_argument("--cache", default="outputs/orphan/af_msa_cache.npz")
    ap.add_argument("--feba_db", default="/tmp/feba.db")
    ap.add_argument("--out_dir", default="data/drive_import/cond")
    a = ap.parse_args()
    main(a.cond_npz, a.cache, a.feba_db, a.out_dir)
