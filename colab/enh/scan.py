"""The sequence layer for the enhancer task: what is at each element, in the physical terms the
plan asked for -- binding sites, groove shape, groove charge, and how hard the duplex is to open.

WHAT THIS BUILDS, AND THE ONE IDEA IT IS ORGANISED AROUND. For every element the CRISPR benchmark
tested, and for every gene promoter it tested against, this scans all 743 JASPAR vertebrate motifs
and records four things per (motif, sequence) pair:

    MX   the best log-odds score anywhere in the sequence, either strand -- the classic PWM answer
    Z    the PARTITION FUNCTION, sum of exp(score) over every position on both strands
    NS   how many positions clear 80% of the motif's own maximum possible score
    SH   the Boltzmann-weighted average SHAPE over all those positions -- minor groove width,
         major groove width, propeller twist, roll, helix twist, and minor-groove electrostatic
         potential, each from the pentamer table

Z RATHER THAN MX IS THE POINT, and it is the correction that makes the plan's arithmetic work. A
6 bp motif in a 4 Mb window has about 4x10^6 x 2 / 4^6 = 1,953 expected matches, which is what the
plan predicted, and essentially all of them are non-functional -- >99.9% of genomic motif matches
are (Wasserman & Sandelin, Nat Rev Genet 2004). So a site's best score says almost nothing on its
own. What decides whether a protein is actually there is the site's score RELATIVE TO EVERY OTHER
SITE COMPETING FOR THE SAME LIMITED PROTEIN. That is a ratio of Boltzmann weights, and Z is its
numerator. The denominator is assembled by the loop, per gene, from the candidate pool plus a
genome-background term measured here. Keeping both MX and Z lets the loop ASK whether the
competition normalisation matters rather than assume it.

SHAPE IS AVERAGED OVER THE SAME WEIGHTS, not read off the single best site. A best-site shape is a
statement about one hit that may carry 1% of the occupancy; weighting by exp(score) makes the shape
the one the protein actually experiences, and it costs nothing extra because the weights are
already computed. The per-base shape tracks come from the pentamer table, so a position's value
depends on its two flanking bases on each side, which is exactly the flanking-sequence dependence
a PWM cannot express.

DUPLEX OPENING is the SantaLucia (1998) unified nearest-neighbour free energies, one value per
dinucleotide step. The plan asked for "the energy for opening the strand" and for the total to come
out net negative. Every stable binding event has a negative free energy, so net-negative does not
discriminate anything; what can discriminate is the BALANCE -- binding energy against the cost of
locally destabilising the duplex -- so both terms are carried separately and their difference is
handed to the loop as a feature rather than used as a filter.

THE DINUCLEOTIDE SHUFFLE, which is here because it is the control that decides whether any of this
is real. Element sequences are re-shuffled preserving every dinucleotide frequency exactly
(Altschul & Erikson 1985, the Eulerian-path construction) and the whole stack is rebuilt on the
shuffled sequences. GC content, CpG content and dinucleotide composition survive the shuffle;
binding sites do not. If the shuffled stack scores as well as the real one, the sequence features
are reading base composition and the motif story is decoration.

CACHING. The full scan is ~750 motifs over ~6 Mbp on both strands and takes minutes, so the result
is written to the scratchpad keyed by a digest of the inputs. Deleting the cache file is the only
way to make it recompute; nothing here mutates a cached array in place.
"""
import csv
import gzip
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from enh import genome as GEN               # noqa: E402
from enh import shape_table as ST           # noqa: E402

OUT_DIR = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = GEN.SP
CR = SP / "crispr"
CACHE = SP / "enh_scan"
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"

POWER_COL = "PowerAtEffectSize25"
MIN_POWER = 0.8              # an unpowered non-significant pair is NOT DETECTED, not a negative
PROMOTER_PAD = 500           # TSS +/- this, so a 1 kb promoter window
BG_WINDOWS, BG_WIDTH = 4000, 500      # 2 Mb of genome background, for the competition denominator
BG_SEED = 173173
SHUF_SEED = 20250823
REL_THRESH = 0.80            # "a site" = 80% of the motif's own maximum attainable log-odds
SHAPES = ["mgw", "mgrw", "prot", "roll", "helt", "ep"]
TRACKS = SHAPES + ["dg"]   # the six shape variables plus the duplex-opening free energy

# SantaLucia (1998) unified nearest-neighbour dG37, kcal/mol per dinucleotide step. Initiation and
# terminal-penalty terms are omitted: this is used as a per-bp mean over 500 bp windows, where they
# contribute a constant of order 0.002 kcal/mol/bp.
NN_DG = {"AA": -1.00, "TT": -1.00, "AT": -0.88, "TA": -0.58,
         "CA": -1.45, "TG": -1.45, "GT": -1.44, "AC": -1.44,
         "CT": -1.28, "AG": -1.28, "GA": -1.30, "TC": -1.30,
         "CG": -2.17, "GC": -2.24, "GG": -1.84, "CC": -1.84}


# ---------------------------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------------------------
def load_benchmark(report=print):
    rows = []
    for f, only_k562 in ((TRAINING, False), (HELDOUT, True)):
        p = CR / f
        if not p.exists():
            raise SystemExit(f"{p} missing -- fetch the EP CRISPR benchmark first")
        r = list(csv.DictReader(gzip.open(p, "rt"), delimiter="\t"))
        if only_k562:
            r = [x for x in r if x["CellType"] == "K562"]
        else:
            for x in r:
                x["CellType"] = "K562"
        report(f"    {f}: {len(r):,} K562 pairs")
        rows += r
    valid = [r for r in rows if r["ValidConnection"] in ("TRUE", "True", "true")]
    powered = [r for r in valid if float(r[POWER_COL] or 0) >= MIN_POWER]
    report(f"    {len(rows):,} -> {len(valid):,} valid -> {len(powered):,} powered "
           f"at {POWER_COL} >= {MIN_POWER}")
    return powered


# ---------------------------------------------------------------------------------------------
# motifs
# ---------------------------------------------------------------------------------------------
def load_motifs(report=print):
    """JASPAR matrix id -> log-odds (L,4) against a uniform background, plus the maximum attainable
    score. Duplicated matrix ids across gene keys collapse to one entry."""
    raw = json.load(open(OUT_DIR / "tf_motifs.json"))["motifs"]
    out = {}
    for rec in raw.values():
        mid = rec["id"]
        if mid in out:
            continue
        M = np.asarray(rec["pwm"], dtype=np.float64)       # (4, L) counts, rows ACGT
        tot = M.sum(0) + 1.0
        p = (M + 0.25) / tot
        lo = np.log(p / 0.25).T.astype(np.float32)         # (L, 4)
        out[mid] = lo
    ids = sorted(out)
    widths = np.array([out[m].shape[0] for m in ids])
    report(f"    {len(ids)} distinct motifs, width {widths.min()}-{widths.max()} "
           f"(median {int(np.median(widths))})")
    return ids, out


# ---------------------------------------------------------------------------------------------
# per-base shape tracks
# ---------------------------------------------------------------------------------------------
def shape_tracks(codes, tab):
    """Per-base shape, from the pentamer centred on each position. The first two and last two bases
    of a contiguous run have no pentamer and come back as NaN; so does any pentamer containing an N.
    Returns {name: float32 array the same length as codes}."""
    n = len(codes)
    c = codes.astype(np.int64)
    bad = c > 3
    c = np.where(bad, 0, c)
    k = np.zeros(n - 4, dtype=np.int64)
    b = np.zeros(n - 4, dtype=bool)
    for i in range(5):
        k = k * 4 + c[i:n - 4 + i]
        b |= bad[i:n - 4 + i]
    out = {}
    for name in SHAPES:
        col = {"roll": "roll1", "helt": "helt1"}.get(name, name)
        v = tab[col][k].astype(np.float32)
        if name in ("roll", "helt"):                     # the two central steps, averaged
            v = 0.5 * (v + tab[{"roll": "roll2", "helt": "helt2"}[name]][k].astype(np.float32))
        v[b] = np.nan
        full = np.full(n, np.nan, dtype=np.float32)
        full[2:n - 2] = v
        out[name] = full
    return out


def dg_track(codes):
    """Per-step SantaLucia dG37. Length n-1, NaN where either base is not ACGT."""
    n = len(codes)
    a, b = codes[:-1].astype(np.int64), codes[1:].astype(np.int64)
    bad = (a > 3) | (b > 3)
    tab = np.zeros(16, dtype=np.float32)
    B = "ACGT"
    for i, x in enumerate(B):
        for j, y in enumerate(B):
            tab[i * 4 + j] = NN_DG[x + y]
    v = tab[np.where(bad, 0, a) * 4 + np.where(bad, 0, b)]
    v[bad] = np.nan
    return v


def all_tracks(codes, tab):
    """The six pentamer shape tracks plus the per-step duplex opening energy, all the same length
    as `codes` so one rolling-window routine serves every one of them."""
    t = shape_tracks(codes, tab)
    dg = dg_track(codes)
    t["dg"] = np.concatenate([dg, [np.nan]]).astype(np.float32)
    return t


# ---------------------------------------------------------------------------------------------
# the scanner
# ---------------------------------------------------------------------------------------------
def _win_mean(track, L):
    """Rolling mean of `track` over windows of width L, NaN-propagating. Length len(track)-L+1."""
    x = np.nan_to_num(track, nan=0.0).astype(np.float64)
    ok = (~np.isnan(track)).astype(np.float64)
    cx = np.concatenate([[0.0], np.cumsum(x)])
    co = np.concatenate([[0.0], np.cumsum(ok)])
    s = cx[L:] - cx[:-L]
    c = co[L:] - co[:-L]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / np.where(c > 0, c, np.nan)).astype(np.float32)


def scan_set(cat, starts, ids, mots, tracks, report=print, want_shape=True, label=""):
    """One pass over a concatenated sequence set.

    `cat` is the concatenation of the sequences with N separators wider than the widest motif, so
    a window that straddles two sequences contains an N and scores -inf; `starts` are the segment
    offsets, and every reduceat therefore aggregates exactly one sequence plus a tail of -inf.

    Returns MX, LZ, NS (each (n_motifs, n_seq)) and, if asked, SH (n_shape, n_motifs, n_seq)."""
    nseg = len(starts)
    nm = len(ids)
    MX = np.full((nm, nseg), -np.inf, dtype=np.float32)
    LZ = np.full((nm, nseg), -np.inf, dtype=np.float32)
    NS = np.zeros((nm, nseg), dtype=np.int32)
    SH = np.full((len(TRACKS), nm, nseg), np.nan, dtype=np.float32) if want_shape else None
    wmean_cache = {}
    t0 = time.time()
    # width-sorted, so the rolling-window means for a given motif width are computed once and
    # reused across every motif of that width instead of being rebuilt 736 times
    order = sorted(range(nm), key=lambda i: mots[ids[i]].shape[0])
    for done, mi in enumerate(order):
        mid = ids[mi]
        lo = mots[mid]
        L = lo.shape[0]
        n = len(cat) - L + 1
        if n <= 0:
            continue
        thr = REL_THRESH * float(lo.max(1).sum())
        sf = np.zeros(n, dtype=np.float32)
        sr = np.zeros(n, dtype=np.float32)
        bad = np.zeros(n, dtype=bool)
        lor = lo[::-1, ::-1]
        for p in range(L):
            c = cat[p:p + n]
            bad |= c > 3
            idx = np.minimum(c, 3)
            sf += lo[p][idx]
            sr += lor[p][idx]
        sf[bad] = -np.inf
        sr[bad] = -np.inf
        MX[mi] = np.maximum(np.maximum.reduceat(sf, starts),
                            np.maximum.reduceat(sr, starts))
        w = np.exp(sf.astype(np.float64)) + np.exp(sr.astype(np.float64))
        zs = np.add.reduceat(w, starts)
        with np.errstate(divide="ignore"):
            LZ[mi] = np.log(zs)
        NS[mi] = (np.add.reduceat(((sf >= thr) | (sr >= thr)).astype(np.int32), starts))
        if want_shape:
            for si, name in enumerate(TRACKS):
                key = (name, L)
                if key not in wmean_cache:
                    if len(wmean_cache) >= len(TRACKS):
                        wmean_cache.clear()          # widths arrive in order, so the old width is done
                    wmean_cache[key] = _win_mean(tracks[name], L)
                v = wmean_cache[key][:n]
                vv = np.nan_to_num(v, nan=0.0).astype(np.float64)
                ok = (~np.isnan(v)).astype(np.float64)
                num = np.add.reduceat(w * vv, starts)
                den = np.add.reduceat(w * ok, starts)
                with np.errstate(invalid="ignore", divide="ignore"):
                    SH[si, mi] = np.where(den > 0, num / den, np.nan)
        if (done + 1) % 100 == 0:
            el = time.time() - t0
            report(f"      {label} motif {done+1}/{nm} (width {L})  "
                   f"[{el:.0f}s, eta {el/(done+1)*(nm-done-1):.0f}s]")
    return MX, LZ, NS, SH


def concat(seqs, sep):
    pad = np.full(sep, GEN.N_CODE, dtype=np.uint8)
    parts, starts, off = [], [], 0
    for s in seqs:
        starts.append(off)
        parts.append(s)
        parts.append(pad)
        off += len(s) + sep
    return np.concatenate(parts), np.array(starts, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# dinucleotide-preserving shuffle (Altschul & Erikson 1985)
# ---------------------------------------------------------------------------------------------
def dinuc_shuffle(seq, rng):
    """seq: uint8 array. Returns a permutation with every dinucleotide count preserved exactly."""
    s = list(seq)
    if len(s) < 4:
        return np.asarray(s, dtype=np.uint8)
    last = s[-1]
    edges = defaultdict(list)
    for a, b in zip(s[:-1], s[1:]):
        edges[a].append(b)
    verts = list(edges)
    for _ in range(200):
        lastedge = {}
        for v in verts:
            if v != last:
                lastedge[v] = edges[v][rng.randrange(len(edges[v]))]
        ok = True
        for v in list(lastedge):
            seen, u = set(), v
            while u != last:
                if u in seen or u not in lastedge:
                    ok = False
                    break
                seen.add(u)
                u = lastedge[u]
            if not ok:
                break
        if ok:
            break
    else:
        return np.asarray(s, dtype=np.uint8)          # could not find a valid Eulerian ordering
    out_edges = {}
    for v in verts:
        e = list(edges[v])
        if v in lastedge:
            e.remove(lastedge[v])
            rng.shuffle(e)
            e.append(lastedge[v])
        else:
            rng.shuffle(e)
        out_edges[v] = e
    res = [s[0]]
    ptr = defaultdict(int)
    cur = s[0]
    for _ in range(len(s) - 1):
        nxt = out_edges[cur][ptr[cur]]
        ptr[cur] += 1
        res.append(nxt)
        cur = nxt
    return np.asarray(res, dtype=np.uint8)


# ---------------------------------------------------------------------------------------------
def background_regions(g, rng, report=print):
    """BG_WINDOWS random windows of BG_WIDTH from the available hg19 chromosomes, weighted by
    chromosome length so the sample is uniform over the genome rather than over chromosomes."""
    chroms = g.available()
    lens = {}
    for c in chroms:
        p = g._path(c)
        lens[c] = int(p.stat().st_size)          # gz size is a fine proxy for relative length
    tot = sum(lens.values())
    regions = []
    for c in chroms:
        k = max(1, int(round(BG_WINDOWS * lens[c] / tot)))
        # hg19 chromosome lengths are not known here without reading; sample within a conservative
        # bound and let out-of-range windows come back as N, which the scan already excludes.
        span = int(lens[c] / 73_773_666 * 249_250_621)
        for _ in range(k):
            s = rng.randrange(1_000_000, max(1_000_001, span - 1_000_000))
            regions.append((c, s, s + BG_WIDTH))
    report(f"    background: {len(regions):,} windows x {BG_WIDTH} bp "
           f"= {len(regions)*BG_WIDTH/1e6:.2f} Mb over {len(chroms)} chromosomes")
    return regions


# ---------------------------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------------------------
def _composition(seq):
    n = len(seq)
    ok = seq <= 3
    m = int(ok.sum())
    gc = float(((seq == 1) | (seq == 2)).sum()) / max(m, 1)
    cg = int(((seq[:-1] == 1) & (seq[1:] == 2)).sum())
    return dict(width=n, n_frac=1.0 - m / max(n, 1), gc=gc,
                cpg=cg / max(m - 1, 1) / max(gc * gc / 4 + 1e-9, 1e-9) if gc > 0 else 0.0,
                cpg_raw=cg / max(m - 1, 1))


def build(report=print, force=False):
    t0 = time.time()
    tab = ST.load()
    ids, mots = load_motifs(report)
    maxw = max(m.shape[0] for m in mots.values())
    rows = load_benchmark(report)

    # ---- lift to hg19 --------------------------------------------------------------------------
    lo = GEN.LiftOver()
    el_key = sorted({(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows})
    gn_key = sorted({(r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"]) for r in rows})
    el19, el_ok = {}, 0
    for k in el_key:
        v = lo.lift_interval(k[0], k[1], k[2])
        el19[k] = v
        el_ok += v is not None
    gn19, gn_ok = {}, 0
    for k in gn_key:
        v = lo.lift(k[0], k[1])
        gn19[k] = v
        gn_ok += v is not None
    report(f"    lifted {el_ok:,}/{len(el_key):,} elements ({el_ok/len(el_key):.4f}) and "
           f"{gn_ok:,}/{len(gn_key):,} TSSs ({gn_ok/len(gn_key):.4f}) hg38 -> hg19")

    rows = [r for r in rows
            if el19[(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))] is not None
            and gn19[(r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])] is not None]
    el_key = sorted({(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows})
    gn_key = sorted({(r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"]) for r in rows})
    ei = {k: i for i, k in enumerate(el_key)}
    gi = {k: i for i, k in enumerate(gn_key)}
    y = np.array([1 if r["Significant"] in ("TRUE", "True", "true") else 0 for r in rows], np.int8)
    e_idx = np.array([ei[(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))] for r in rows])
    g_idx = np.array([gi[(r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])] for r in rows])
    dist = np.array([max(abs(float(r["distanceToTSS"])), 1.0) for r in rows])
    chrom = np.array([r["chrTSS"] for r in rows])
    dataset = np.array([r["Dataset"] for r in rows])
    report(f"    ANALYSIS SET: {len(rows):,} pairs, {int(y.sum())} positives "
           f"(base rate {y.mean():.4f}), {len(el_key):,} elements, {len(gn_key):,} genes, "
           f"{len(set(chrom))} chromosomes")

    # ---- sequence ------------------------------------------------------------------------------
    g = GEN.Genome()
    el_reg = [(k[0],) + el19[k] for k in el_key]
    pr_reg = [(k[0], gn19[k] - PROMOTER_PAD, gn19[k] + PROMOTER_PAD) for k in gn_key]
    rng = random.Random(BG_SEED)
    bg_reg = background_regions(g, rng, report)
    report("    extracting element sequence")
    el_seq = g.extract_cached(el_reg, "elem", report)
    report("    extracting promoter sequence")
    pr_seq = g.extract_cached(pr_reg, "prom", report)
    report("    extracting background sequence")
    bg_seq = g.extract_cached(bg_reg, "bg", report)
    el_nf = np.array([float((s > 3).mean()) for s in el_seq])
    report(f"    element N fraction: median {np.median(el_nf):.4f}, "
           f"{int((el_nf > 0.05).sum())} elements above 5%")

    # ---- shuffled elements ---------------------------------------------------------------------
    srng = random.Random(SHUF_SEED)
    sh_seq = [dinuc_shuffle(s, srng) for s in el_seq]
    d_ok = all(_dinuc_equal(a, b) for a, b in zip(el_seq[:200], sh_seq[:200]))
    report(f"    dinucleotide shuffle: composition preserved on the first 200 elements: {d_ok}")

    # ---- scan ----------------------------------------------------------------------------------
    out = {}
    for tag, seqs, want in (("el", el_seq, True), ("sh", sh_seq, True),
                            ("pr", pr_seq, False), ("bg", bg_seq, False)):
        cat, starts = concat(seqs, maxw)
        tracks = all_tracks(cat, tab) if want else None
        report(f"    scanning {tag}: {len(seqs):,} sequences, {len(cat):,} bp")
        MX, LZ, NS, SH = scan_set(cat, starts, ids, mots, tracks, report, want, tag)
        out[tag] = dict(MX=MX, LZ=LZ, NS=NS)
        if want:
            out[tag]["SH"] = SH
        del cat, tracks

    # ---- element-wide (not site-specific) composition and shape --------------------------------
    comp = {k: [] for k in ("width", "n_frac", "gc", "cpg", "cpg_raw")}
    shp = {k: [] for k in TRACKS}
    shp_sh = {k: [] for k in TRACKS}
    for s in el_seq:
        c = _composition(s)
        for k in comp:
            comp[k].append(c[k])
        t = all_tracks(s, tab)
        for k in TRACKS:
            shp[k].append(float(np.nanmean(t[k])) if np.isfinite(t[k]).any() else np.nan)
    for s in sh_seq:
        t = all_tracks(s, tab)
        for k in TRACKS:
            shp_sh[k].append(float(np.nanmean(t[k])) if np.isfinite(t[k]).any() else np.nan)

    payload = dict(
        y=y, e_idx=e_idx, g_idx=g_idx, dist=dist,
        chrom=np.array(chrom, dtype=object), dataset=np.array(dataset, dtype=object),
        motif_ids=np.array(ids, dtype=object),
        motif_width=np.array([mots[m].shape[0] for m in ids], np.int32),
        motif_maxscore=np.array([float(mots[m].max(1).sum()) for m in ids], np.float32),
        el_key=np.array([f"{a}:{b}-{c}" for a, b, c in el_key], dtype=object),
        gn_key=np.array([f"{a}:{b}:{c}" for a, b, c in gn_key], dtype=object),
        bg_bp=np.int64(sum(int((s <= 3).sum()) for s in bg_seq)),
        tracks=np.array(TRACKS, dtype=object),
    )
    for k, v in comp.items():
        payload["el_" + k] = np.asarray(v, np.float32)
    for k in TRACKS:
        payload["elmean_" + k] = np.asarray(shp[k], np.float32)
        payload["shmean_" + k] = np.asarray(shp_sh[k], np.float32)
    for tag in out:
        for k, v in out[tag].items():
            payload[f"{tag}_{k}"] = v

    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "enh_scan.npz"
    np.savez_compressed(p, **{k: v for k, v in payload.items()})
    report(f"    -> {p} ({p.stat().st_size/1e6:.1f} MB)  [{time.time()-t0:.0f}s]")
    return payload


def _dinuc_equal(a, b):
    def counts(x):
        d = defaultdict(int)
        for p, q in zip(x[:-1], x[1:]):
            d[(int(p), int(q))] += 1
        return dict(d)
    return counts(a) == counts(b)


def load(report=print):
    p = CACHE / "enh_scan.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run `python colab/enh/scan.py` first")
    z = np.load(p, allow_pickle=True)
    report(f"    scan cache: {p.name} ({p.stat().st_size/1e6:.1f} MB)")
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    print("=" * 100)
    print("SEQUENCE LAYER: motifs, partition functions, groove shape and charge, duplex opening")
    print("=" * 100)
    build()


# ---------------------------------------------------------------------------------------------
# motif CLUSTERING, which the per-motif summaries above cannot express
# ---------------------------------------------------------------------------------------------
"""MX, LZ and NS say how strong a motif's best site is, how much total weight it carries, and how
many sites clear threshold. None of them says where those sites are relative to each other, and
that is the one thing about an enhancer that has been reproducible for twenty years: functional
regulatory elements carry CLUSTERS -- several copies of one motif close together (homotypic), and
several different motifs close together (heterotypic). A count of 40 sites spread over 500 bp and a
count of 40 sites piled into 60 bp are the same number in NS and completely different objects.

Two families of statistic are collected in one extra pass over the same score arrays:

  HOMOTYPIC   for each motif, the largest number of its own above-threshold sites inside any window
              of CLUSTER_BP. Reduced immediately to per-sequence summaries -- the best such count
              over all motifs, and how many motifs manage two or more -- because keeping the full
              (motif x sequence) matrix would be a third copy of an array this loop already holds
              twice.
  HETEROTYPIC a per-position count of how many DISTINCT motifs have a site starting there,
              accumulated across all motifs, then summarised per sequence as the densest window at
              three scales and the mean. This is the quantity a "regulatory module" is usually
              defined by, and it cannot be recovered from any per-motif column.

Thresholds are the same REL_THRESH used everywhere else, so "a site" means one thing in this file.
"""
CLUSTER_BP = (50, 100, 200)
HOMO_BP = 100
MIN_HOMO = 2


def _roll_max(x, w, starts, n):
    """Per-segment maximum of the rolling sum of `x` over windows of width w."""
    c = np.concatenate([[0], np.cumsum(x.astype(np.int32))])
    s = (c[w:] - c[:-w]).astype(np.int32)
    st = np.minimum(starts, len(s) - 1)
    return np.maximum.reduceat(s, st)


def scan_clusters(cat, starts, ids, mots, report=print, label=""):
    """Homotypic and heterotypic clustering statistics, per sequence."""
    nseg = len(starts)
    dens = np.zeros(len(cat), dtype=np.int16)
    homo_best = np.zeros(nseg, dtype=np.int32)
    homo_n = np.zeros(nseg, dtype=np.int32)
    t0 = time.time()
    order = sorted(range(len(ids)), key=lambda i: mots[ids[i]].shape[0])
    for done, mi in enumerate(order):
        lo = mots[ids[mi]]
        L = lo.shape[0]
        n = len(cat) - L + 1
        if n <= 0:
            continue
        thr = REL_THRESH * float(lo.max(1).sum())
        sf = np.zeros(n, dtype=np.float32)
        sr = np.zeros(n, dtype=np.float32)
        bad = np.zeros(n, dtype=bool)
        lor = lo[::-1, ::-1]
        for p in range(L):
            c = cat[p:p + n]
            bad |= c > 3
            idx = np.minimum(c, 3)
            sf += lo[p][idx]
            sr += lor[p][idx]
        hit = ((sf >= thr) | (sr >= thr)) & ~bad
        dens[:n] += hit.astype(np.int16)
        m = _roll_max(hit, HOMO_BP, starts, n)
        homo_best = np.maximum(homo_best, m)
        homo_n += (m >= MIN_HOMO).astype(np.int32)
        if (done + 1) % 200 == 0:
            el = time.time() - t0
            report(f"      {label} cluster motif {done+1}/{len(ids)}  "
                   f"[{el:.0f}s, eta {el/(done+1)*(len(ids)-done-1):.0f}s]")
    out = dict(homo_best=homo_best.astype(np.float32), homo_n=homo_n.astype(np.float32))
    ends = np.r_[starts[1:], len(cat)]
    for w in CLUSTER_BP:
        c = np.concatenate([[0], np.cumsum(dens.astype(np.int64))])
        s = (c[w:] - c[:-w]).astype(np.int32)
        st = np.minimum(starts, len(s) - 1)
        out[f"het_max_{w}"] = np.maximum.reduceat(s, st).astype(np.float32)
    tot = np.add.reduceat(dens.astype(np.int64), starts).astype(np.float64)
    out["het_mean"] = (tot / np.maximum(ends - starts, 1)).astype(np.float32)
    report(f"    clusters: homotypic best median {np.median(out['homo_best']):.0f} sites in "
           f"{HOMO_BP} bp, {np.median(out['homo_n']):.0f} motifs with >= {MIN_HOMO}; "
           f"heterotypic density median {np.median(out['het_mean']):.2f} motifs/bp")
    return out
