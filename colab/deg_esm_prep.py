"""Adapter: per-organism DEG proteomes -> the combined fasta + index JSON that
embed_proteins_esm.py expects (header '>{gene_idx}|org|locus_tag', index maps
gene_idx -> 'org|locus_tag').

In : colab/work/proteomes/<org>.faa   (>locus_tag\\nSEQ)
Out: colab/work/deg_proteins.fasta
     colab/work/deg_protein_index.json
Then on Colab:
  python colab/embed_proteins_esm.py --fasta colab/work/deg_proteins.fasta \\
      --index colab/work/deg_protein_index.json \\
      --out outputs/orphan/esm_embeddings.npy
Sandbox-side, esm_embeddings.npy + the index map idx -> (org, locus_tag) so the
Stage-5 family classifier can use real protein embeddings (kNN to gap-slot
centroids) instead of orthogroup consensus.
"""
import json, glob, argparse
from pathlib import Path

DEG_ORGS = {"mtub", "syn3a", "mgen", "saur_NCTC8325_biotradis", "spneT4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteomes", default="colab/work/proteomes")
    ap.add_argument("--fasta", default="colab/work/deg_proteins.fasta")
    ap.add_argument("--index", default="colab/work/deg_protein_index.json")
    ap.add_argument("--only_deg", action="store_true",
                    help="restrict to the 5 labelled DEG cells")
    args = ap.parse_args()

    idx = {}; gid = 0
    with open(args.fasta, "w") as out:
        for faa in sorted(glob.glob(f"{args.proteomes}/*.faa")):
            org = Path(faa).stem
            if args.only_deg and org not in DEG_ORGS:
                continue
            lt = None; seq = []
            def flush():
                nonlocal gid, lt, seq
                if lt and seq:
                    out.write(f">{gid}|{org}|{lt}\n{''.join(seq)}\n")
                    idx[gid] = f"{org}|{lt}"; gid += 1
                lt, seq = None, []
            for line in open(faa):
                line = line.rstrip("\n")
                if line.startswith(">"):
                    flush(); lt = line[1:].split()[0]
                else:
                    seq.append(line)
            flush()
    json.dump(idx, open(args.index, "w"))
    print(f"wrote {args.fasta} + {args.index}: {gid} proteins from "
          f"{len({v.split('|')[0] for v in idx.values()})} organisms")


if __name__ == "__main__":
    main()
