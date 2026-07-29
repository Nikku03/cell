"""BORZOI IN-SILICO DELETION vs the CRISPR benchmark -- sequence grammar as a scalar, not an embedding.

WHY DELETION RATHER THAN EMBEDDINGS. The obvious use of a sequence model is to take its trunk activations
at each locus as features. With 569 positives and a 1536-dim trunk that is 3,072 columns for two loci, and
the model would fit noise. In-silico deletion returns ONE number per pair -- the predicted change in the
gene's expression when the element's sequence is removed -- which is the right size for this data AND is a
direct analogue of the CRISPRi experiment the benchmark measures. It is also how Enformer and Borzoi are
validated in the literature, rather than a use we invented.

WHY THIS IS NOT THE TIER-2 MISTAKE. Tier-2 trained our own sequence CNN and lost to a bisect over a BED
file; the lesson recorded was "you cannot out-predict a measurement". Borzoi is different in kind: it is
pretrained on orders of magnitude more sequence than this project has labels, and it is used FROZEN. The
question is not "can we learn sequence grammar from 569 positives" -- it is "does grammar learned elsewhere
carry information our measured tracks do not".

THE RECEPTIVE FIELD IS THE BINDING CONSTRAINT, and it decides the experiment. Borzoi sees 524,288 bp, so an
element must lie within 262,144 bp of the TSS to be deletable at all. That is 3,960 of 10,331 pairs --
but those hold 483 of 569 positives (84.9%), because positives concentrate at short range. The subset's base
rate is 0.1220 against 0.0551 overall, so **AUPRC here is not comparable to the 0.6881 headline** and the
baseline is rescored on exactly the same pairs, the same way extrude.py handles its simulated subset.

TWO CONTROLS, because one is not enough:
  shuffled-delta   permute the Borzoi scores across pairs -- is it an extra-column effect?
  sham-deletion    delete a RANDOM span of the same size at the same distance on the other side of the TSS.
                   Deleting any few hundred bp near a gene lowers predicted expression somewhat; this asks
                   whether deleting THIS element lowers it more than deleting an arbitrary one.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TWOBIT = SP / "hg38.2bit"
SEQLEN = 524_288                  # Borzoi input
HALF = SEQLEN // 2
OUT_BINS, BIN_BP = 6_144, 32      # Borzoi output: 196,608 bp centred
MODEL_ID = os.environ.get("BORZOI_MODEL", "johahi/borzoi-replicate-0")
NSUB = int(os.environ.get("BORZOI_NSUB", "0"))     # 0 = all addressable pairs
SEEDS = (0, 1, 2, 3, 4)
BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def load_pairs():
    from contact_gate import load_ctcf, contact_features
    from competition import competition_features, load as load_comp
    from ubiquitous import load_ubiquitous
    df, Xid, Cr = load_comp()
    CF = competition_features(df)
    U = np.isin(df["gene"].values, list(load_ubiquitous())).astype(float)[:, None]
    X = np.hstack([Xid, Cr, CF, U])
    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            k = (f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", row["measuredGeneSymbol"])
            try:
                tss[k] = (row["chrom"], int(row["chromStart"]), int(row["chromEnd"]), int(row["startTSS"]))
            except ValueError:
                continue
    loci = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]
    assert len(loci) == len(df), "element-gene join lost rows"
    return df, X, loci


def one_hot(seq):
    x = np.zeros((4, len(seq)), np.float32)
    for i, b in enumerate(seq.upper()):
        j = BASES.get(b)
        if j is not None:
            x[j, i] = 1.0
    return x


def window(tb, chrom, centre):
    """524 kb centred on the TSS, zero-padded at contig edges rather than shifted -- shifting would move
    the TSS off the output centre and silently change which bins are read."""
    lo, hi = centre - HALF, centre + HALF
    clen = tb.chroms().get(chrom)
    if clen is None:
        return None, None
    s, e = max(lo, 0), min(hi, clen)
    if e - s < SEQLEN // 4:
        return None, None
    seq = tb.sequence(chrom, s, e)
    x = np.zeros((4, SEQLEN), np.float32)
    x[:, s - lo:s - lo + (e - s)] = one_hot(seq)
    return x, lo


def main():
    import py2bit
    import torch
    from crispr_gate import _cv_auprc, _seeded_folds

    df, X, loci = load_pairs()
    y, ch = df["crispr_hit"].values, df["chromosome"].values
    dist = np.array([abs(t - (s + e) // 2) for _, s, e, t in loci])
    addressable = dist < HALF
    print("=" * 100)
    print("BORZOI IN-SILICO DELETION vs the CRISPR benchmark")
    print("=" * 100)
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}")
    print(f"  within Borzoi's +/-{HALF:,} bp receptive field: {addressable.sum():,} pairs "
          f"({100*addressable.mean():.1f}%) holding {int(y[addressable].sum())} positives "
          f"({100*y[addressable].sum()/y.sum():.1f}%), base rate {y[addressable].mean():.4f}")
    print("  -> AUPRC below is on that subset and is NOT comparable to the full-set 0.6881")

    idx = np.where(addressable)[0]
    if NSUB and NSUB < len(idx):
        idx = np.sort(np.random.default_rng(0).choice(idx, NSUB, replace=False))
        print(f"  BORZOI_NSUB set: sampling {len(idx):,} of the addressable pairs")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("\n  WARNING: no GPU. Borzoi on CPU is ~100x slower; this will not finish in a sane time.")
    from borzoi_pytorch import Borzoi
    model = Borzoi.from_pretrained(MODEL_ID).to(dev).eval()
    print(f"  model {MODEL_ID} on {dev}")

    tb = py2bit.open(str(TWOBIT))
    by_gene = {}
    for i in idx:
        by_gene.setdefault(df["gene"].values[i], []).append(i)

    delta = np.full(len(y), np.nan)
    sham = np.full(len(y), np.nan)
    t0 = __import__("time").time()
    done = 0
    with torch.no_grad():
        for gi, (gene, rowsi) in enumerate(by_gene.items()):
            chrom, _, _, t = loci[rowsi[0]]
            ref, lo = window(tb, chrom, t)
            if ref is None:
                continue
            # TSS sits at the centre of the cropped output; read a small window of bins around it
            c = OUT_BINS // 2
            def express(batch):
                o = model(torch.tensor(np.stack(batch), device=dev))
                if o.ndim == 3 and o.shape[1] < o.shape[2]:
                    o = o.transpose(1, 2)                  # -> (B, bins, tracks)
                return o[:, c - 4:c + 4, :].mean((1, 2)).float().cpu().numpy()

            base = express([ref])[0]
            batch, tags = [], []
            for i in rowsi:
                _, s, e, _ = loci[i]
                a, b = s - lo, e - lo
                if a < 0 or b > SEQLEN or b <= a:
                    continue
                m = ref.copy(); m[:, a:b] = 0.0                       # real deletion
                batch.append(m); tags.append((i, "real"))
                mirror = 2 * (t - lo) - b                              # same distance, other side
                if 0 <= mirror and mirror + (b - a) <= SEQLEN:
                    sh = ref.copy(); sh[:, mirror:mirror + (b - a)] = 0.0
                    batch.append(sh); tags.append((i, "sham"))
            for k in range(0, len(batch), 4):
                vals = express(batch[k:k + 4])
                for (i, kind), v in zip(tags[k:k + 4], vals):
                    lfc = float(np.log2((base + 1e-6) / (v + 1e-6)))
                    if kind == "real":
                        delta[i] = lfc
                    else:
                        sham[i] = lfc
            done += len(rowsi)
            if gi % 50 == 0:
                el = __import__("time").time() - t0
                print(f"    gene {gi}/{len(by_gene)}  {done}/{len(idx)} pairs  {el:.0f}s  "
                      f"~{el/max(done,1)*(len(idx)-done)/60:.1f} min left", flush=True)
    tb.close()

    ok = np.isfinite(delta)
    print(f"\n  scored {ok.sum():,} pairs in {(__import__('time').time()-t0)/60:.1f} min")
    print(f"  median predicted log2 drop: real {np.nanmedian(delta):+.5f}  "
          f"sham {np.nanmedian(sham):+.5f}")
    if not ok.any():
        raise SystemExit("no pair produced a prediction -- do not interpret anything below")

    sub = np.where(ok)[0]
    ys, chs, Xs = y[sub], ch[sub], X[sub]
    D = delta[sub][:, None]
    S = np.nan_to_num(sham[sub], nan=float(np.nanmedian(sham)))[:, None]

    def sc(M):
        return np.array([_cv_auprc(M, ys, _seeded_folds(chs, s)) for s in SEEDS])

    b = sc(Xs)
    d = sc(np.hstack([Xs, D]))
    rng = np.random.default_rng(0)
    c = sc(np.hstack([Xs, D[rng.permutation(len(D))]]))
    sh = sc(np.hstack([Xs, D - S]))          # real minus sham: the grammar-specific part
    print(f"\n  (all arms on the SAME {len(sub):,} pairs, base rate {ys.mean():.4f})")
    for tag, v in (("full stack [rescored on subset]", b), ("+ Borzoi deletion", d),
                   ("+ shuffled-delta control", c), ("+ (real - sham) deletion", sh)):
        print(f"    {tag:36s} {v.mean():.4f}  [{' '.join(f'{x:.3f}' for x in v)}]")
    net = (d - b).mean() - (c - b).mean()
    se = float(np.sqrt((d - b).var(ddof=1) / len(b) + (c - b).var(ddof=1) / len(b)))
    print(f"\n  Borzoi on top of the stack : {(d-b).mean():+.4f}")
    print(f"  shuffled-delta control     : {(c-b).mean():+.4f}")
    print(f"  NET OF CONTROL             : {net:+.4f} +/- {se:.4f}   z = {net/max(se,1e-9):+.2f}   "
          f"sign consistent: {bool(np.all((d-b)>0))}")
    print(f"  sham-corrected arm         : {(sh-b).mean():+.4f}")
    R = {"n_pairs_full": int(len(y)), "n_addressable": int(addressable.sum()),
         "n_scored": int(ok.sum()), "subset_base_rate": float(ys.mean()),
         "model": MODEL_ID, "baseline_on_subset": float(b.mean()), "plus_borzoi": float(d.mean()),
         "shuffled_control": float(c.mean()), "sham_corrected": float(sh.mean()),
         "delta": float((d - b).mean()), "delta_control": float((c - b).mean()),
         "net": float(net), "se": se, "z": float(net / max(se, 1e-9)),
         "sign_consistent": bool(np.all((d - b) > 0)),
         "median_lfc_real": float(np.nanmedian(delta)), "median_lfc_sham": float(np.nanmedian(sham))}
    R["verdict"] = ("GO -- frozen sequence grammar adds signal the measured tracks do not carry."
                    if net >= 0.01 and net > 2 * se and R["sign_consistent"] else
                    "NO-GO -- Borzoi in-silico deletion does not beat the measured-track stack.")
    print(f"\n  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "borzoi_gate.json", "w"), indent=1)
    print(f"\n  -> {OUT/'borzoi_gate.json'}")


if __name__ == "__main__":
    main()
