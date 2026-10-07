"""Build a SEQUENCE source side: one ESM-2 embedding per knockout gene, cached to disk.

WHY THIS EXISTS.  The model's source side is 696 curated annotation channels, and 99 of 200 cohort-1 knockouts
have an EXACT channel twin -- another knockout with a byte-identical annotation vector.  For those the model is
being asked to distinguish inputs it cannot tell apart.  A protein sequence exists for every gene, differs between
twins, and -- unlike a GO term -- was not written down by anyone: ESM-2's representation is learned from ~250M
sequences.  That makes it the natural test of whether the ceiling is the architecture or the vocabulary.

WHAT THIS SCRIPT DOES, and nothing more: gene symbol -> canonical protein sequence -> mean-pooled ESM-2
representation -> `esm_<model>.npz`.  The scoring lives in adrn_esm_source.py so that the (slow) embedding is
paid once and the (fast) evaluation can be re-run and re-gated freely.

CHOICES, stated because they bound what the test can show:
  * REVIEWED (SwissProt) human proteins only.  TrEMBL would raise coverage but at the cost of predicted-only
    entries whose sequences are not curated.  Coverage is reported rather than papered over.
  * Where a gene symbol maps to several entries, the LONGEST sequence is taken as a canonical-isoform proxy.
  * Sequences are truncated to 1022 residues (ESM-2's context is 1024 including BOS/EOS).  The count of
    truncated proteins is reported -- for long proteins the embedding sees only the N-terminal region.
  * Mean pooling over residues.  A per-residue representation carries more, but the source side needs one
    fixed-length vector per gene and mean pooling is the standard, unfitted choice.
"""
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

SP = A.SP
FASTA = SP / "human_reviewed.fasta.gz"
MODEL = os.environ.get("ESM_MODEL", "esm2_t12_35M_UR50D")
LAYERS = {"esm2_t12_35M_UR50D": 12, "esm2_t30_150M_UR50D": 30, "esm2_t33_650M_UR50D": 33}
MAXLEN = 1022
N_BENCH = int(os.environ.get("ESM_BENCH", "0"))     # >0: embed only this many and report the rate, then stop
# SHARDING.  One 4-thread process reached 1,266 ms/protein (~105 min for 4,995).  Transformer inference on CPU
# scales better across PROCESSES than across threads -- intra-op parallelism on small matmuls leaves cores idle --
# so the run is split into NSHARD single-thread processes and merged.  Shards are interleaved (i::n) rather than
# contiguous so that each gets the same length distribution and they finish together; contiguous blocks of a
# length-sorted list would leave one shard with every long protein.
SHARD = int(os.environ.get("ESM_SHARD", "0"))
NSHARD = int(os.environ.get("ESM_NSHARD", "1"))


def load_fasta():
    """gene symbol -> longest reviewed sequence."""
    seqs, gene, buf = {}, None, []

    def flush():
        if gene and buf:
            s = "".join(buf)
            if gene not in seqs or len(s) > len(seqs[gene]):
                seqs[gene] = s

    for line in gzip.open(FASTA, "rt"):
        if line.startswith(">"):
            flush()
            buf = []
            gene = None
            for tok in line.split():
                if tok.startswith("GN="):
                    gene = tok[3:]
                    break
        else:
            buf.append(line.strip())
    flush()
    return seqs


def main():
    print("=" * 100, flush=True)
    print(f"ESM-2 SEQUENCE SOURCE SIDE -- model {MODEL}", flush=True)
    print("=" * 100, flush=True)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = sorted({str(k) for k in z["kos"]})
    seqs = load_fasta()
    print(f"  reviewed human proteins with a gene symbol: {len(seqs):,}", flush=True)

    have = [g for g in kos if g in seqs]
    miss = [g for g in kos if g not in seqs]
    print(f"  knockout genes: {len(kos):,} | sequence found {len(have):,} ({len(have)/len(kos):.1%}) "
          f"| missing {len(miss):,}", flush=True)
    if miss[:8]:
        print(f"  first missing: {', '.join(miss[:8])}", flush=True)
    ln = np.array([len(seqs[g]) for g in have])
    print(f"  length median {int(np.median(ln))}, mean {ln.mean():.0f}, "
          f"truncated at {MAXLEN}: {int((ln > MAXLEN).sum()):,}", flush=True)

    torch.set_num_threads(1 if NSHARD > 1 else max(1, os.cpu_count() - 1))
    print(f"  loading {MODEL} on cpu ({torch.get_num_threads()} threads) ...", flush=True)
    import esm
    model, alphabet = getattr(esm.pretrained, MODEL)()
    model.eval()
    bc = alphabet.get_batch_converter()
    layer = LAYERS[MODEL]

    todo = have[:N_BENCH] if N_BENCH else have
    # length-sorted so batches are homogeneous and padding is not paid for, THEN interleaved into shards so
    # every shard sees the same length distribution
    todo = sorted(todo, key=lambda g: len(seqs[g]))
    if NSHARD > 1:
        todo = todo[SHARD::NSHARD]
        print(f"  shard {SHARD + 1}/{NSHARD}: {len(todo):,} proteins", flush=True)
    dim = model.embed_dim
    E = np.zeros((len(todo), dim), np.float32)
    t0 = time.time()
    i = 0
    while i < len(todo):
        L = min(len(seqs[todo[i]]), MAXLEN)
        bs = max(1, min(16, int(16000 / max(L, 1))))          # keep tokens/batch roughly constant
        chunk = todo[i:i + bs]
        data = [(g, seqs[g][:MAXLEN]) for g in chunk]
        _, _, toks = bc(data)
        with torch.no_grad():
            rep = model(toks, repr_layers=[layer])["representations"][layer]
        for b, g in enumerate(chunk):
            n = min(len(seqs[g]), MAXLEN)
            E[i + b] = rep[b, 1:n + 1].mean(0).numpy()        # drop BOS/EOS, mean over residues
        i += bs
        if (i // 200) != ((i - bs) // 200) or i >= len(todo):
            el = time.time() - t0
            print(f"    {i}/{len(todo)}  {el:.0f}s  ({el/max(i,1)*1000:.0f} ms/protein, "
                  f"~{(len(todo)-i)*el/max(i,1)/60:.1f} min left)", flush=True)

    el = time.time() - t0
    print(f"  embedded {len(todo):,} proteins in {el/60:.1f} min ({el/max(len(todo),1)*1000:.0f} ms each)",
          flush=True)
    if N_BENCH:
        print(f"  BENCHMARK ONLY -- full run of {len(have):,} would take "
              f"~{len(have)*el/max(len(todo),1)/60:.0f} min. Nothing written.", flush=True)
        return 0

    out = SP / (f"esm_{MODEL}.shard{SHARD}of{NSHARD}.npz" if NSHARD > 1 else f"esm_{MODEL}.npz")
    np.savez_compressed(out, genes=np.array(todo), E=E, missing=np.array(miss))
    print(f"  -> {out}  ({E.shape})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
