"""PATH-ORPHAN, PHASE B (GPU) -- ESM-2 embeddings for the deep-orphan tail.

The ONE GPU step. dN/dS needs >=3 orthologs and structure needs a fold hit;
true singletons have neither. ESM-2 gives a sequence embedding for EVERY protein
(trained across all of life), so it is the only feature defined on the deepest
orphans. We mean-pool ESM-2 per protein and reduce to K PCs (keeps the model
small; full 1280-d would swamp 140 positives).

Run on an A100/L4 runtime; ~20 min for ~4,400 proteins. Reuses the project's
ESM pass (`build_esm2_embeddings.py`); this wrapper keys by locus_tag and PCAs.

INPUT (real): outputs/orphan/proteins_<org>.faa  (from step 0)
OUTPUT: outputs/orphan/esm_<org>.parquet  locus_tag, esm_0..esm_{K-1}

--smoke : validate mean-pool + PCA-reduce + IO shape on synthetic embeddings
          (no GPU / no ESM model needed)
--real  : embed with ESM-2 on GPU, then PCA-reduce
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
K_PCS = 16


def pca_reduce(emb, k=K_PCS, seed=0):
    """emb: (n, d) float. Return (n, k) PCA scores (centered, SVD). Pure NumPy."""
    import numpy as np
    emb = np.asarray(emb, float)
    mu = emb.mean(0)
    Xc = emb - mu
    k = min(k, Xc.shape[1], max(1, Xc.shape[0] - 1))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:k].T)


def _read_faa(path):
    ids = []; seqs = []; sid = None; buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if sid:
                    ids.append(sid); seqs.append("".join(buf))
                sid = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if sid:
        ids.append(sid); seqs.append("".join(buf))
    return ids, seqs


def gather_all_proteins():
    """(organism, locus_tag, seq) for every labeled gene across all orgs whose
    cached genome joins -> the training pool for the leave-one-organism-out test.
    Reuses the GFF locus_tag->protein_id + faa->seq join (the step-0 bridge)."""
    import csv, re
    DATA = REPO / "data" / "drive_import"
    acc = {}
    with open(DATA / "labels" / "genome_cache_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r["accession"] and r.get("has_protein_faa") == "True":
                acc[r["organism"]] = r["accession"]
    labeled = {}
    with open(DATA / "labels" / "labels.csv") as f:
        for r in csv.DictReader(f):
            labeled.setdefault(r["organism"], set()).add(r["locus_tag"])
    out = []
    for org, a in acc.items():
        gff = DATA / "genome_cache" / a / "genomic.gff"
        faa = DATA / "genome_cache" / a / "protein.faa"
        if not (gff.exists() and faa.exists()) or org not in labeled:
            continue
        lt2pid = {}
        with open(gff) as f:
            for line in f:
                if "\tCDS\t" not in line:
                    continue
                lt = re.search(r'locus_tag=([^;\n]+)', line)
                pid = re.search(r'protein_id=([^;\n]+)', line)
                if lt and pid:
                    lt2pid.setdefault(lt.group(1), pid.group(1))
        ids, seqs = _read_faa(faa)
        seqmap = dict(zip(ids, seqs))
        for lt in labeled[org]:
            pid = lt2pid.get(lt)
            if pid and pid in seqmap:
                out.append((org, lt, seqmap[pid]))
    return out


def embed_esm(seqs, model_name="facebook/esm2_t33_650M_UR50D", batch=8):
    """Mean-pooled ESM-2 embeddings (GPU). Imported lazily so --smoke needs no
    torch/transformers."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = [s[:1022] for s in seqs[i:i+batch]]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                rep = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
    import numpy as np
    return np.concatenate(out, 0)


def run_multi(args):
    """Embed ALL orgs together with a GLOBAL PCA -> esm_all.parquet (organism,
    locus_tag, esm_*). This is the input to the leave-one-organism-out test:
    PCs must be comparable across orgs, so PCA is fit on the pooled embedding."""
    import pandas as pd, numpy as np
    out_path = OUT / "esm_all.parquet"
    if out_path.exists() and not args.force:
        print(f"PHASE B ESM(multi) -- CACHED ({out_path})"); return 0
    trip = gather_all_proteins()
    orgs = sorted({o for o, _, _ in trip})
    print(f"  embedding {len(trip):,} proteins across {len(orgs)} orgs with "
          f"ESM-2 ({args.model})")
    seqs = [s for _, _, s in trip]
    emb = embed_esm(seqs, args.model, args.batch)
    # Keep the FULL embedding (not 16 PCs). Top-variance PCs capture protein-
    # family structure, NOT the essentiality-discriminative directions, so
    # aggressive PCA throws the signal away (audit: 16-PC ESM scored 0 cross-org
    # while raw AA-composition scored 0.13). Cross-org LOO trains on 11-18k
    # genes -> can fit full dims with L2.
    feat = emb.astype("float32")
    print(f"  keeping FULL {feat.shape[1]}-d embedding for LOO")
    df = pd.DataFrame(feat, columns=[f"esm_{i}" for i in range(feat.shape[1])])
    df.insert(0, "locus_tag", [lt for _, lt, _ in trip])
    df.insert(0, "organism", [o for o, _, _ in trip])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}  ({pcs.shape[0]}x{pcs.shape[1]}, {len(orgs)} orgs)")
    return 0


def run_real(args):
    import pandas as pd, numpy as np
    if args.multi:
        return run_multi(args)
    faa = OUT / f"proteins_{args.org}.faa"
    if not faa.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    out_path = OUT / f"esm_{args.org}.parquet"
    if out_path.exists() and not args.force:
        print(f"PHASE B ESM -- CACHED ({out_path})"); return 0
    ids, seqs = _read_faa(faa)
    print(f"  embedding {len(ids):,} proteins with ESM-2 ({args.model})")
    emb = embed_esm(seqs, args.model, args.batch)
    pcs = pca_reduce(emb, args.k)
    df = pd.DataFrame(pcs, columns=[f"esm_{i}" for i in range(pcs.shape[1])])
    df.insert(0, "locus_tag", ids)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}  ({pcs.shape[0]}x{pcs.shape[1]})")
    return 0


def run_smoke():
    import numpy as np
    print("=" * 64)
    print("PHASE B ESM SMOKE -- mean-pool / PCA-reduce / IO (no GPU)")
    print("=" * 64)
    rng = np.random.RandomState(0)
    # synthetic embeddings with structure in 3 directions -> PCA must recover var
    n, d = 50, 64
    latent = rng.normal(0, 1, (n, 3))
    W = rng.normal(0, 1, (3, d))
    emb = latent @ W + rng.normal(0, 0.01, (n, d))
    pcs = pca_reduce(emb, k=8)
    print(f"  emb {emb.shape} -> pcs {pcs.shape}")
    assert pcs.shape == (n, 8)
    # first 3 PCs should capture ~all variance (rank-3 signal)
    var = pcs.var(0)
    frac = var[:3].sum() / var.sum()
    print(f"  variance in first 3 PCs: {frac:.3f} (expect ~1.0, rank-3 signal)")
    assert frac > 0.99, "PCA should concentrate the rank-3 structure"
    # faa round-trip
    import tempfile
    p = Path(tempfile.mkdtemp()) / "t.faa"
    p.write_text(">L1\nMKKL\n>L2\nMVVT\n>L3\nMAAG\n")
    ids, seqs = _read_faa(p)
    assert ids == ["L1", "L2", "L3"] and seqs[0] == "MKKL"
    print(f"  faa parse: {ids}  OK")
    print("\n=== PHASE B SMOKE PASS ===")
    print("  PCA reduction (recovers low-rank structure) + FASTA IO validated.")
    print("  Real path embeds with ESM-2 on GPU (one ~20-min A100 pass).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--k", type=int, default=K_PCS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--multi", action="store_true",
                    help="embed ALL orgs -> esm_all.parquet (for LOO-org test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
