"""Compute ESM-2 protein embeddings for the cleanly-joined organisms.

Pilot to answer the one open scientific question: do sequence-derived
features (which capture structure/function orthogonal to phylogeny)
move MCC where six orthology-correlated feature classes did not?

Mapping chain per organism:
  our locus_tag --(GFF CDS protein_id=)--> protein_id
  protein_id    --(protein.faa header)--> amino-acid sequence
  sequence      --(ESM-2, mean-pooled)--> embedding vector

Only organisms with a clean locus_tag join (manifest join_rate >= the
threshold) are processed -- for those, BERIL locus_tag == RefSeq
locus_tag, so the GFF mapping is direct. We embed only genes that
appear in labels.csv for that organism (limits compute, guarantees the
embedding joins to a training row).

Model: facebook/esm2_t6_8M_UR50D by default (320-dim, fast). The
larger esm2_t33_650M (1280-dim) is available via --model if the pilot
shows signal and we want max quality.

Output: esm2_embeddings.parquet  (organism, locus_tag, emb_0..emb_{D-1})
"""
from __future__ import annotations
import argparse, gzip, sys, time
from pathlib import Path


def parse_gff_locus_to_protein(gff_path: Path) -> dict[str, str]:
    """{locus_tag: protein_id} from CDS features."""
    out = {}
    op = gzip.open if str(gff_path).endswith(".gz") else open
    with op(gff_path, "rt") as f:
        for line in f:
            if line.startswith("#") or "\tCDS\t" not in line:
                continue
            attr = line.rstrip("\n").split("\t")[-1]
            d = {}
            for kv in attr.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1); d[k] = v
            lt = d.get("locus_tag"); pid = d.get("protein_id")
            if lt and pid and lt not in out:
                out[lt] = pid
    return out


def parse_faa(faa_path: Path) -> dict[str, str]:
    """{protein_id: sequence}. NCBI .faa headers start '>WP_xxxx.1 desc'."""
    out = {}
    cur = None; buf = []
    op = gzip.open if str(faa_path).endswith(".gz") else open
    with op(faa_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    out[cur] = "".join(buf)
                cur = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        out[cur] = "".join(buf)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",   type=Path, required=True,
                   help="genome_cache dir (holds <accession>/ subdirs)")
    p.add_argument("--manifest",    type=Path, required=True,
                   help="genome_cache_manifest.csv")
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--out",         type=Path, required=True)
    p.add_argument("--model",       default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--min-join",    type=float, default=0.5,
                   help="only embed organisms with manifest join_rate >= this")
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--max-len",     type=int, default=1022)
    p.add_argument("--only",        nargs="*", default=None)
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        print(f"ERROR: {e}\n  pip install torch transformers pandas pyarrow",
              file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   model: {args.model}")

    labels = pd.read_csv(args.labels_csv)
    mf = pd.read_csv(args.manifest)
    # clean ours = kind ours with join >= threshold
    clean = mf[(mf.kind == "ours") & (mf.join_rate.notna())
               & (mf.join_rate >= args.min_join)].copy()
    if args.only:
        clean = clean[clean.organism.isin(args.only)]
    print(f"clean organisms to embed: {len(clean)}")
    for _, r in clean.iterrows():
        print(f"  {r.organism:<30s} {r.accession}  join={r.join_rate}")
    if len(clean) == 0:
        print("ERROR: no clean organisms in manifest", file=sys.stderr); return 1

    # ---- load model ----
    print("loading ESM-2 ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    emb_dim = model.config.hidden_size
    print(f"  embedding dim: {emb_dim}")

    @torch.no_grad()
    def embed_batch(seqs: list[str]) -> "np.ndarray":
        enc = tok(seqs, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state           # (B, L, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # mean-pool over real tokens (excludes padding; includes BOS/EOS,
        # negligible for mean over length)
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        return (summed / counts).cpu().numpy()

    all_rows = []
    for _, r in clean.iterrows():
        org = r.organism; acc = r.accession
        gdir = args.cache_dir / acc
        gff = gdir / "genomic.gff"; faa = gdir / "protein.faa"
        if not gff.exists() or not faa.exists():
            print(f"  {org}: missing gff/faa in {gdir}, skip"); continue
        lt2pid = parse_gff_locus_to_protein(gff)
        pid2seq = parse_faa(faa)
        # restrict to labelled genes for this organism
        org_lts = set(labels[labels.organism == org].locus_tag.astype(str))
        pairs = []
        for lt in org_lts:
            pid = lt2pid.get(lt)
            if pid and pid in pid2seq:
                seq = pid2seq[pid]
                if 10 <= len(seq) <= 5000:
                    pairs.append((lt, seq))
        print(f"  {org}: {len(pairs)}/{len(org_lts)} labelled genes have "
              f"protein seq  -> embedding")
        # batch through ESM-2
        t0 = time.time()
        for i in range(0, len(pairs), args.batch_size):
            chunk = pairs[i:i+args.batch_size]
            embs = embed_batch([s for _, s in chunk])
            for (lt, _), e in zip(chunk, embs):
                row = {"organism": org, "locus_tag": lt}
                for j in range(emb_dim):
                    row[f"emb_{j}"] = float(e[j])
                all_rows.append(row)
            if (i // args.batch_size) % 20 == 0:
                print(f"    {i+len(chunk)}/{len(pairs)}  "
                      f"({(i+len(chunk))/max(time.time()-t0,1e-9):.0f} prot/s)")
        print(f"    done {org} in {time.time()-t0:.0f}s")

    if not all_rows:
        print("ERROR: no embeddings produced", file=sys.stderr); return 1
    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df):,} genes x {emb_dim}-dim, "
          f"{df.organism.nunique()} organisms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
