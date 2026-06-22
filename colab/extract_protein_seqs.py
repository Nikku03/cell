"""Stage 3a -- extract protein sequences for the genes in cond_pairs.

For each unique (organism, locus_tag) in cond_pairs.npz, locate its protein.faa
in data/drive_import/genome_cache/ (via genome_cache_manifest.csv), parse the
[locus_tag=...] tag in the FASTA header, and emit a deterministic per-gene
sequence file used by Stage 3b (ESM-2 embedding).

Output: data/drive_import/cond/proteins.fasta
        data/drive_import/cond/protein_index.json     # gene_idx -> {org, locus_tag, len}

Genes whose protein cannot be resolved get an empty sequence and are flagged in
the index (`resolved: False`) -- the model will fall back to a zero embedding
for those rows.
"""
import argparse, csv, json, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_faa(faa_path, target_loci):
    """Return {locus_tag: sequence} for locus_tags in target_loci."""
    out = {}
    cur_lt, cur_seq = None, []
    re_lt = re.compile(r"\[locus_tag=([^\]]+)\]")
    with open(faa_path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_lt is not None and cur_lt in target_loci and cur_seq:
                    out[cur_lt] = "".join(cur_seq)
                m = re_lt.search(line)
                if m:
                    cur_lt = m.group(1)
                else:
                    # fallback: take the first whitespace token after >
                    tok = line[1:].split()[0]
                    cur_lt = tok.split("|")[-1]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_lt is not None and cur_lt in target_loci and cur_seq:
            out[cur_lt] = "".join(cur_seq)
    return out


def main(cond_npz, genome_cache, manifest_csv, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Z = np.load(cond_npz, allow_pickle=True)
    gene_idx = Z["gene_idx"]
    org_arr = Z["org"]
    # we need to recover the locus_tag per gene_idx; load it from the cache npz
    cache_path = Path("outputs/orphan/af_msa_cache.npz")
    cache = np.load(cache_path, allow_pickle=True)
    cache_orgs = list(cache["orgs"]); meta_org = cache["meta_org"]; lt = cache["lt"]
    org_by_idx = {gi: (str(cache_orgs[int(meta_org[gi])]), str(lt[gi]))
                  for gi in set(gene_idx.tolist())}
    print(f"unique genes to extract: {len(org_by_idx):,}")

    # group by organism
    targets = defaultdict(set)
    for gi, (org, lt_) in org_by_idx.items():
        targets[org].add(lt_)
    print(f"unique organisms: {len(targets)}")

    # manifest: org -> accession -> genome_cache/<accession>/protein.faa
    org_to_accession = {}
    if Path(manifest_csv).exists():
        for r in csv.DictReader(open(manifest_csv)):
            if r.get("accession"):
                org_to_accession[r["organism"]] = r["accession"]

    # also look for protein.faa next to genomic.gff in any subdir
    found_sequences = {}
    for org, loci in targets.items():
        acc = org_to_accession.get(org)
        candidates = []
        if acc:
            candidates.append(Path(genome_cache) / acc / "protein.faa")
        # fallback: also try any subdir matching the org name
        for d in Path(genome_cache).iterdir():
            if d.is_dir():
                faa = d / "protein.faa"
                if faa.exists() and (acc is None or d.name == acc):
                    candidates.append(faa)
        seqs = {}
        for c in candidates:
            if c.exists():
                seqs.update(parse_faa(c, loci))
                if seqs:
                    break
        found_sequences[org] = seqs
        print(f"  {org:<28} target {len(loci):>5}  found {len(seqs):>5}  "
              f"({100*len(seqs)/max(1,len(loci)):>5.1f}%)")

    # emit fasta + index
    out_fa = out_dir / "proteins.fasta"
    index = {}
    with open(out_fa, "w") as f:
        for gi, (org, lt_) in sorted(org_by_idx.items()):
            seq = found_sequences.get(org, {}).get(lt_, "")
            seq = seq.replace("*", "")[:1022]   # ESM-2 max len 1024; leave headroom
            index[gi] = dict(organism=org, locus_tag=lt_,
                             len=len(seq), resolved=bool(seq))
            if seq:
                f.write(f">{gi}|{org}|{lt_}\n{seq}\n")
    json.dump(index, open(out_dir / "protein_index.json", "w"))
    n_resolved = sum(1 for v in index.values() if v["resolved"])
    print(f"\nresolved {n_resolved:,} / {len(index):,} ({100*n_resolved/len(index):.1f}%)")
    print(f"wrote {out_fa} + protein_index.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond_npz", default="data/drive_import/cond/cond_pairs.npz")
    ap.add_argument("--genome_cache", default="data/drive_import/genome_cache")
    ap.add_argument("--manifest", default="data/drive_import/labels/genome_cache_manifest.csv")
    ap.add_argument("--out_dir", default="data/drive_import/cond")
    a = ap.parse_args()
    main(a.cond_npz, a.genome_cache, a.manifest, a.out_dir)
