"""STAGE 2: ESM-2 protein-language-model features for the conditional model.

WHY: conservation is a gene-level constant -- it cannot interact with condition.
ESM-2 embeddings encode protein FUNCTION at high resolution, which CAN interact
with condition (a reductase matters under oxidative stress; a transporter under
nutrient limit). This is the biggest single lever to lift the conditional-
fitness model above the conservation baseline (today AUPRC 0.19).

SEQUENCE SOURCE: feba's aaseqs (figshare 25236931, ~43 MB) -- every gene's
amino-acid sequence keyed by feba (orgId, locusId), the SAME key as our
training frame. Bridge: organism = "beril_"+orgId, locus_tag = locusId.
Full coverage, per-gene granularity, no OG-rep 44% limit, no disjoint-data wall.

OUTPUT: outputs/esm_embeddings.parquet -> columns organism, locus_tag, e0..eN
(mean-pooled last-layer ESM-2 embedding per gene).

STAGES:
  --stage download  fetch + gunzip aaseqs from figshare
  --stage dryrun    parse aaseqs, report header format + JOIN RATE to the
                    training frame  (do this BEFORE paying for GPU embedding)
  --stage embed     run ESM-2 (default esm2_t30_150M_UR50D, 640-dim), cache
                    per-organism, write the embeddings parquet
--smoke : validate aaseqs parsing + the (organism,locus_tag) bridge on synthetic
          data (no download, no GPU).
"""
from __future__ import annotations
import argparse, gzip, json, sys, urllib.request, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
GTDB = REPO / "data" / "gtdb"          # reuse for cache dir
SEQ = REPO / "data" / "feba_seqs"
FRAME_CANDIDATES = [
    OUT / "step2_training_frame.parquet",
    Path("/content/drive/MyDrive/step2_training_frame.parquet"),
]
ZENODO_ARTICLE = "25236931"            # feba figshare release (has aaseqs.gz)
DEFAULT_MODEL = "esm2_t30_150M_UR50D"  # 640-dim; bump to t33_650M (1280) to scale
MAX_AA = 1022


def _stream(url, dest, expected=0, chunk=1 << 20, retries=4):
    dest = Path(dest)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            return
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(dest, "ab" if have else "wb") as f:
                if have and r.status == 200:
                    f.close(); dest.unlink(); f = open(dest, "wb")
                while True:
                    b = r.read(chunk)
                    if not b: break
                    f.write(b)
            return
        except Exception as e:
            print(f"  dl error {e}; retry {attempt}"); time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed {url}")


def _find_aaseqs():
    import urllib.request, json as _json
    SEQ.mkdir(parents=True, exist_ok=True)
    local = SEQ / "aaseqs"
    if local.exists() and local.stat().st_size > 1_000_000:
        return local
    api = (f"https://api.figshare.com/v2/articles/{ZENODO_ARTICLE}/files"
           f"?page_size=1000")
    req = urllib.request.Request(api, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        files = _json.loads(r.read())
    cand = [f for f in files if f["name"].lower().startswith("aaseqs")]
    if not cand:
        raise RuntimeError(f"no aaseqs in figshare {ZENODO_ARTICLE}: "
                           f"{[f['name'] for f in files[:10]]}")
    f = cand[0]
    url = f.get("download_url") or \
        f"https://ndownloader.figshare.com/files/{f['id']}"
    gz = SEQ / f["name"]
    print(f"  downloading {f['name']} ({f.get('size',0)/1e6:.1f} MB)")
    _stream(url, gz)
    if str(gz).endswith(".gz"):
        print("  gunzip ...")
        with gzip.open(gz, "rt") as fi, open(local, "w") as fo:
            for line in fi:
                fo.write(line)
        return local
    return gz


def parse_aaseqs(path, max_aa=MAX_AA):
    """feba aaseqs FASTA. Header is 'orgId:locusId' (':' separated). Yields
    (organism, locus_tag, seq) with organism = 'beril_'+orgId."""
    org, lt, buf = None, None, []
    out = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if org is not None:
                    out.append((org, lt, "".join(buf)[:max_aa]))
                head = line[1:].split()[0]
                if ":" in head:
                    o, l = head.split(":", 1)
                    org, lt = "beril_" + o, l
                else:
                    org, lt = None, head  # unknown org layout
                buf = []
            else:
                buf.append(line.strip())
        if org is not None:
            out.append((org, lt, "".join(buf)[:max_aa]))
    return out


def stage_download():
    p = _find_aaseqs()
    print(f"  aaseqs ready at {p} ({p.stat().st_size/1e6:.1f} MB)")


def stage_dryrun():
    """Parse aaseqs + report header format and JOIN RATE to the frame.
    Run this BEFORE embedding -- it's the cheap go/no-go for coverage."""
    import pandas as pd
    p = _find_aaseqs()
    seqs = parse_aaseqs(p)
    print(f"  parsed {len(seqs):,} sequences")
    bad_org = sum(1 for o, _, _ in seqs if o is None)
    print(f"  header parse: {bad_org} with unknown org "
          f"(expect 0 if 'orgId:locusId' format)")
    sdf = pd.DataFrame(seqs, columns=["organism", "locus_tag", "seq"])
    sdf = sdf.dropna(subset=["organism"])
    print(f"  organisms in aaseqs: {sdf.organism.nunique()}  "
          f"sample: {sdf.organism.head(3).tolist()}")
    print(f"  seq length: median {sdf.seq.str.len().median():.0f}  "
          f">{MAX_AA}aa (truncated): {(sdf.seq.str.len() >= MAX_AA).sum()}")
    frame = next((q for q in FRAME_CANDIDATES if q.exists()), None)
    if frame is None:
        print("  (training frame not found; skip join check)"); return
    f = pd.read_parquet(frame, columns=["organism", "locus_tag"])
    fk = set(map(tuple, f[["organism", "locus_tag"]].drop_duplicates().values))
    sk = set(map(tuple, sdf[["organism", "locus_tag"]].values))
    hit = len(fk & sk)
    print(f"  FRAME (organism,locus_tag) keys: {len(fk):,}")
    print(f"  JOIN RATE: {hit:,}/{len(fk):,} = {100*hit/len(fk):.1f}% of frame "
          f"genes have an aaseqs sequence")
    if hit / len(fk) < 0.6:
        print("  WARNING: <60% join -- header format or org bridge may be off")


def stage_embed(args):
    import pandas as pd, numpy as np, torch, esm
    out = OUT / "esm_embeddings.parquet"
    if out.exists() and not args.force:
        print(f"  have {out}"); return
    p = _find_aaseqs()
    seqs = [(o, l, s) for o, l, s in parse_aaseqs(p) if o and s]
    # restrict to genes that are actually in the frame (saves embedding cost)
    frame = next((q for q in FRAME_CANDIDATES if q.exists()), None)
    if frame is not None:
        f = pd.read_parquet(frame, columns=["organism", "locus_tag"])
        keep = set(map(tuple, f[["organism", "locus_tag"]]
                       .drop_duplicates().values))
        seqs = [(o, l, s) for o, l, s in seqs if (o, l) in keep]
    print(f"  embedding {len(seqs):,} sequences with {args.model}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = getattr(esm.pretrained, args.model)()
    bc = alphabet.get_batch_converter()
    model.eval().to(dev)
    nlayer = model.num_layers
    rows = []
    B = args.batch
    # SPEED: sort by length so each batch pads to ~uniform length (a single
    # long seq in a mixed batch otherwise forces every member through a long
    # forward pass). Output carries its own (o,l) key, so order is irrelevant.
    seqs = sorted(seqs, key=lambda x: len(x[2]))
    # SPEED: bf16 autocast on GPU (L4/A100 native). ~1.5-2x, no accuracy cost
    # for mean-pooled features. CPU stays fp32.
    use_amp = dev == "cuda"
    amp_dt = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) \
        else torch.float16
    t0 = time.time()
    for i in range(0, len(seqs), B):
        chunk = [(f"{o}|{l}", s) for o, l, s in seqs[i:i + B]]
        _, _, toks = bc(chunk)
        toks = toks.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dt,
                                              enabled=use_amp):
            rep = model(toks, repr_layers=[nlayer])["representations"][nlayer]
        rep = rep.float()
        for j, (o, l, s) in enumerate(seqs[i:i + B]):
            L = min(len(s), MAX_AA)
            v = rep[j, 1:L + 1].mean(0).cpu().numpy()
            rows.append((o, l, *v.tolist()))
        if (i // B) % 50 == 0:
            el = time.time() - t0
            print(f"    {i+len(chunk):,}/{len(seqs):,}  "
                  f"{(i+len(chunk))/max(el,1):.0f} seq/s")
    d = len(rows[0]) - 2
    cols = ["organism", "locus_tag"] + [f"e{k}" for k in range(d)]
    df = pd.DataFrame(rows, columns=cols)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out}: {len(df):,} genes x {d}-dim ESM embeddings")


def run_smoke():
    import pandas as pd, tempfile, os
    print("=" * 60)
    print("STAGE 2 SMOKE -- aaseqs parsing + frame bridge")
    print("=" * 60)
    fa = "\n".join([">ANA3:11530891", "MKAT", ">ANA3:11530892", "GGGG",
                    ">MR1:999", "MMMM"]) + "\n"
    d = tempfile.mkdtemp(); p = os.path.join(d, "aaseqs")
    open(p, "w").write(fa)
    seqs = parse_aaseqs(p)
    print("  parsed:", seqs)
    assert seqs[0] == ("beril_ANA3", "11530891", "MKAT"), "header parse wrong"
    assert seqs[2] == ("beril_MR1", "999", "MMMM")
    # bridge to a synthetic frame
    frame = pd.DataFrame({"organism": ["beril_ANA3", "beril_MR1"],
                          "locus_tag": ["11530891", "999"]})
    sk = set(map(tuple, pd.DataFrame(seqs, columns=["o", "l", "s"])
                 [["o", "l"]].values))
    fk = set(map(tuple, frame.values))
    hit = len(fk & sk)
    print(f"  frame join: {hit}/{len(fk)}")
    assert hit == 2, "bridge join failed"
    print("\n=== STAGE 2 SMOKE PASS ===")
    print("  aaseqs 'orgId:locusId' parsing + beril_ bridge validated.")
    print("  On Colab: --stage download, then --stage dryrun (check join%),")
    print("  then --stage embed (GPU).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", choices=["download", "dryrun", "embed"],
                    default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage is None and not args.smoke:
        args.smoke = True
    if args.smoke:
        return run_smoke()
    if args.stage == "download":
        stage_download()
    elif args.stage == "dryrun":
        stage_dryrun()
    elif args.stage == "embed":
        stage_embed(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
