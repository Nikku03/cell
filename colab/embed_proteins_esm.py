"""Stage 3b -- ESM-2 embeddings for the proteins extracted in 3a.

Loads `facebook/esm2_t12_35M_UR50D` (35 M params, ~140 MB) via HuggingFace
transformers, runs CLS-token (mean-pool over residues) embeddings, saves a
[n_genes, 480] float16 matrix indexed by gene_idx (matching cond_pairs).

Genes that didn't resolve in 3a get zero embeddings (model still works; their
protein branch contributes nothing).

  python colab/embed_proteins_esm.py     # ~15-30 min on A100 for ~70k proteins
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def main(fasta_path, index_path, out_path, model_name, max_batch, max_len):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; model={model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    hid = mdl.config.hidden_size
    print(f"hidden size: {hid}")

    index = {int(k): v for k, v in json.load(open(index_path)).items()}
    n_genes = max(index) + 1
    emb = np.zeros((n_genes, hid), np.float16)

    # group sequences by length bucket for efficient batching
    seqs = []
    cur_gid, cur_lines = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_gid is not None and cur_lines:
                    seqs.append((cur_gid, "".join(cur_lines)))
                cur_gid = int(line[1:].split("|")[0])
                cur_lines = []
            else:
                cur_lines.append(line)
        if cur_gid is not None and cur_lines:
            seqs.append((cur_gid, "".join(cur_lines)))
    print(f"sequences to embed: {len(seqs):,}")

    # sort by length descending for efficient batching (less padding waste)
    seqs.sort(key=lambda x: -len(x[1]))
    t0 = time.time()
    done = 0
    with torch.no_grad():
        i = 0
        while i < len(seqs):
            batch = []
            longest = len(seqs[i][1])
            target_n = max(1, min(max_batch, max_batch * 256 // max(longest, 1)))
            while len(batch) < target_n and i + len(batch) < len(seqs):
                batch.append(seqs[i + len(batch)])
            gids = [g for g, _ in batch]
            seqstr = [s[:max_len] for _, s in batch]
            enc = tok(seqstr, padding=True, return_tensors="pt").to(device)
            out = mdl(**enc).last_hidden_state          # [B, L, H]
            mask = enc.attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = pooled.cpu().numpy().astype(np.float16)
            for k, g in enumerate(gids):
                emb[g] = pooled[k]
            i += len(batch); done += len(batch)
            if done % 2000 < len(batch):
                dt = time.time() - t0
                rate = done / max(dt, 1)
                eta = (len(seqs) - done) / max(rate, 1)
                print(f"  {done:>6}/{len(seqs)}  ({rate:.0f}/s; ETA {eta/60:.1f} min)")
    np.save(out_path, emb)
    print(f"wrote {out_path}  shape={emb.shape}  dtype={emb.dtype}  "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/drive_import/cond/proteins.fasta")
    ap.add_argument("--index", default="data/drive_import/cond/protein_index.json")
    ap.add_argument("--out", default="data/drive_import/cond/protein_emb.npy")
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--max_batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1022)
    a = ap.parse_args()
    main(a.fasta, a.index, a.out, a.model, a.max_batch, a.max_len)
