"""Loop 224. All 1,914,250 perturbed cells, one cell at a time, and the noise floor measured
rather than assumed.

WHY THIS IS THE RIGHT NEXT MOVE AND NOT A SIDE QUEST. Every loop from 216 to 223 has been blocked
by the same thing: the A549 dexamethasone series has FOUR replicates, so the noise floor on a
per-interval change can only be estimated from six pairwise numbers, and loop 220 showed two of
those four share a component the other two lack. There is no way to measure the noise floor
properly with four samples. Loop 222 therefore had to GUESS at reliability from within-{1,2,3}
agreement, and loop 223 inherits that guess.

Perturb-seq removes the guess. Replogle et al. 2022 measured 1,989,578 individual K562 cells across
9,867 CRISPRi perturbations, a median of 178 cells per perturbation. Every pseudobulk value this
project used in loop 208 -- and reported at 0.2785 -- is a MEAN OF ABOUT 178 CELLS whose spread was
thrown away before we ever saw it. This loop reads the cells back.

    1,989,578 cells total
       75,328 non-targeting control cells, EXCLUDED from every accumulator below
    1,914,250 perturbed cells, all of them, at 8,248 genes each

THE FILE IS 65.83 GB AND IS NOT DOWNLOADED. There is 5.1 GB of writable space in this container,
so the matrix is streamed over HTTP range requests and never lands on disk. This is possible only
because of a property of the file that was checked before this loop was written and not assumed:
/X is (1989578, 8248) float32, chunks=None, compression=None, stored contiguously at byte offset
2048. A contiguous uncompressed dataset means the bytes of cells [i, i+m) are exactly
2048 + i*8248*4 up to 2048 + (i+m)*8248*4, so any run of cells is one range request. Measured
throughput at 16 workers is 82 MB/s, so the full pass is about 13 minutes of transfer.

RAW OFFSET ARITHMETIC IS THE ONE THING THAT COULD SILENTLY CORRUPT EVERYTHING. If the offset is
wrong by one row the accumulators fill with the wrong cells and every number below is garbage that
looks fine. X1 therefore reads the same cells twice by two independent routes -- through h5py, and
through the hand-computed byte offset -- and requires them to be bit-identical, before any other
gate is allowed to run.

WHAT IS ACCUMULATED IN ONE PASS, over perturbed cells only:
    n_p          cells per perturbation
    S_p, Q_p     per-perturbation sum and sum of squares, per gene
    S_pA, n_pA   the same for the even-indexed half of each perturbation's cells, which makes a
                 split-half estimate available without a second pass
    S_pX         the same under a fixed shuffle of the cell-to-perturbation assignment, which is
                 the control for X2 and X3 and costs one extra accumulator rather than a second
                 13-minute pass

PREDECLARED, BEFORE ANY NUMBER.

  X1 IS THE MATRIX BEING READ CORRECTLY?  -- the integrity gate, and everything requires it
     Two checks. First, cells read by hand-computed byte offset must be bit-identical to the same
     cells read through h5py, max absolute difference exactly 0.0. Second, the pseudobulk
     recomputed from streamed cells must reproduce the published bulk file already on disk.
     Gate: PASS iff the offset check is exactly 0.0 AND the recomputed pseudobulk correlates
     above 0.99 with the published values with median absolute difference below 0.01.
     A FAIL means the read is wrong and NOTHING below may be read.

  X2 WHAT IS THE MEASURED RELIABILITY OF A PSEUDOBULK VALUE?
     Decompose per gene: within-perturbation variance from the cells, between-perturbation
     variance from the perturbation means, and reliability of a mean of n cells as
     between / (between + within/n).
     Gate: PASS iff the decomposition is WELL-POSED -- both variance components strictly positive
     for at least 90% of genes, so the ratio has a denominator. It deliberately does NOT gate on
     reliability being high or low. This project has twice written gates that could only pass on
     the answer the author expected, and the reliability of a Perturb-seq pseudobulk value is
     exactly the kind of number where that would be invisible. The value is reported, not gated.

  X3 DOES THE MEASURED NOISE MODEL ACTUALLY PREDICT REPRODUCIBILITY?  -- the falsifiable one
     Split each perturbation's cells into two halves, form two independent pseudobulk estimates,
     and measure their real agreement per gene. Compare that against the agreement PREDICTED by
     X2's variance decomposition for a half-sized sample. If the decomposition is right the two
     agree; if the noise is structured rather than independent across cells, predicted
     reliability will exceed observed and the model is wrong.
     Gate: PASS iff Pearson between predicted and observed split-half reliability across genes is
     at least 0.80 AND the median absolute difference is below 0.10. Requires X1.

  X4 ARE THE PER-CELL RESPONSES UNIMODAL?  -- what a mean throws away
     A pseudobulk mean is the right summary only if the cells form one population. Take the 200
     strongest perturbation-by-gene effects, re-stream only those cells, and test each real
     per-cell distribution against a Gaussian with matched n.
     Gate: PASS iff the CONTROL is calibrated -- simulated Gaussian samples with matched n must be
     flagged non-Gaussian less than 10% of the time. The gate is on whether the test can tell the
     difference; the observed fraction is the answer and is reported, not gated. Requires X1.

  X5 CONTROL: DOES RELIABILITY COLLAPSE UNDER SHUFFLED ASSIGNMENT?
     The same decomposition on the shuffled cell-to-perturbation map, where between-perturbation
     variance must be nothing but sampling noise.
     Gate: PASS iff median shuffled reliability is below 0.05 while the real one is above it.
     Without this, a high reliability could be an artefact of the estimator rather than a fact
     about the data. Requires X1.

  X6 DOES THE MEASURED FLOOR IMPROVE CROSS-CELL-LINE TRANSFER?
     Loop 208 measured K562-to-RPE1 agreement on pseudobulk values weighted equally. Reweight by
     the reliability measured here and rescore.
     Gate: PASS iff reliability-weighted agreement exceeds the unweighted value computed in the
     same run on the same genes. VOID if the RPE1 file will not load. Requires X1 and X2.

  X7 WHAT THIS CANNOT SHOW -- written before the run.
     These are K562 CRISPRi knockdowns, not A549 dexamethasone. A noise floor measured here is a
     property of this assay and transfers to the A549 series only as an order of magnitude.
     The X matrix is the NORMALIZED one: already z-scored against non-targeting controls within
     gem group. Some cell-to-cell variance has therefore already been removed by the depositors,
     which biases the within-perturbation variance DOWNWARD and reliability UPWARD. The direction
     of that bias is stated here so a high reliability is not read as a clean result.
     A split-half test shares the same library prep, the same day and the same gem group between
     halves, so it measures reproducibility of sampling, not of the experiment.
"""
import os, sys, json, time, io, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_singlecell_floor.json"
URL = "https://ndownloader.figshare.com/files/35774440"
FSIZE = 65830941948
NCELL, NGENE, XOFF = 1989578, 8248, 2048
BLOCK, WORKERS, SEED = 2048, 16, 224224
BULK = Path(os.environ.get("SCRATCH", "/tmp/claude-0/-home-user-cell/"
                           "0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")) / "perturbseq"
NSTRONG, MINCELL = 200, 40

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def fetch_cells(lo, hi, tries=10):
    """Raw byte range for cells [lo, hi) of the contiguous float32 matrix.

    The first run of this loop lost a completed 24-minute stream because a single two-cell
    request failed six times and the exception propagated. Retries are now longer and capped
    rather than linear, and callers coalesce small requests instead of issuing them one per run.
    """
    b0 = XOFF + lo * NGENE * 4
    b1 = XOFF + hi * NGENE * 4 - 1
    last = ""
    for t in range(tries):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={b0}-{b1}"}, timeout=300,
                             allow_redirects=True)
            if r.status_code in (200, 206) and len(r.content) == (b1 - b0 + 1):
                return np.frombuffer(r.content, dtype="<f4").reshape(hi - lo, NGENE)
            last = f"status {r.status_code}, {len(r.content)} of {b1-b0+1} bytes"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(2.0 * (t + 1), 15.0))
    raise IOError(f"cells {lo}-{hi} failed after {tries} tries ({last})")


def h5_index(grp):
    """anndata names its index in an ATTRIBUTE, not a dataset called 'index'.

    The published bulk file carries obs.attrs['_index'] = 'gene_transcript' and
    var.attrs['_index'] = 'gene_id'. The first run of this loop looked for datasets named
    '_index' or 'index', found neither, and voided the integrity gate on a KeyError after the
    whole matrix had already been read.
    """
    key = grp.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    arr = grp[key][:]
    if arr.dtype.kind in "iu" and "__categories" in grp and key in grp["__categories"]:
        arr = np.asarray(grp["__categories"][key][:])[arr]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "single-cell noise floor over all perturbed cells"}
    rng = np.random.default_rng(SEED)
    say("=" * 104)
    say("LOOP 224 -- ALL 1,914,250 PERTURBED CELLS, AND THE NOISE FLOOR MEASURED NOT ASSUMED")
    say("=" * 104)
    say("     The A549 series has four replicates, so its noise floor can only be guessed.")
    say("     Perturb-seq has 1,989,578 cells. Every pseudobulk value loop 208 used at 0.2785 is")
    say("     a mean of about 178 cells whose spread was discarded before we saw it.")
    say(f"     The 65.83 GB matrix is STREAMED over HTTP range requests; {os.statvfs('.').f_bavail*os.statvfs('.').f_frsize/1e9:.1f} GB free on disk.")

    sys.path.insert(0, "/tmp")
    from rangefile import RangeFile
    import h5py
    rf = RangeFile(URL, size=FSIZE, block=8 << 20)
    hf = h5py.File(io.BufferedReader(rf, buffer_size=1 << 20), "r")
    cats = np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in hf["obs/__categories/gene"][:]])
    code = hf["obs/gene"][:].astype(np.int32)
    gcode = hf["var/gene_name"][:].astype(np.int32)
    gcats = np.array([x.decode() if isinstance(x, bytes) else str(x)
                      for x in hf["var/__categories/gene_name"][:]])
    gname = gcats[gcode]
    gid = np.array([x.decode() if isinstance(x, bytes) else str(x)
                    for x in hf["var/gene_id"][:]])
    NT = int(np.where(cats == "non-targeting")[0][0])
    npert = len(cats)
    say(f"     {NCELL:,} cells, {npert:,} perturbation labels, {NGENE:,} genes")
    say(f"     non-targeting is label {NT}; its {int((code==NT).sum()):,} cells are EXCLUDED")

    # ---------------------------------------------------------------- X1 part one
    say("X1 IS THE MATRIX BEING READ CORRECTLY?")
    a_h5 = np.asarray(hf["X"][1000:1016, :], dtype=np.float32)
    a_raw = fetch_cells(1000, 1016)
    off_err = float(np.max(np.abs(a_h5 - a_raw)))
    say(f"     cells 1000-1015 via h5py vs via hand-computed byte offset: "
        f"max absolute difference {off_err:.1e}")

    keep = code != NT
    say(f"     perturbed cells to stream: {int(keep.sum()):,}")
    code_sh = code.copy()
    code_sh[keep] = rng.permutation(code_sh[keep])

    CKPT = BULK.parent / "loop224_accum.npz"
    if CKPT.exists():
        say(f"     accumulator checkpoint found at {CKPT.name}; the 23.7-minute stream is not "
            f"repeated")
        _c = np.load(CKPT)
        S, Q, SA, SX = _c["S"], _c["Q"], _c["SA"], _c["SX"]
        n, nA, nX, code_sh = _c["n"], _c["nA"], _c["nX"], _c["code_sh"]
        keep = code != NT
        tstart = time.time()
    else:
        S = np.zeros((npert, NGENE), np.float32)
        Q = np.zeros((npert, NGENE), np.float32)
        SA = np.zeros((npert, NGENE), np.float32)
        SX = np.zeros((npert, NGENE), np.float32)
        n = np.zeros(npert, np.int64); nA = np.zeros(npert, np.int64)
        nX = np.zeros(npert, np.int64)

        def accum(A, cds, dest, cnt):
            o = np.argsort(cds, kind="stable")
            cs = cds[o]; Ao = A[o]
            b = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
            sums = np.add.reduceat(Ao, b, axis=0)
            u = cs[b]
            dest[u] += sums
            cnt += np.bincount(cs, minlength=len(cnt))

        blocks = [(i, min(i + BLOCK, NCELL)) for i in range(0, NCELL, BLOCK)]
        say(f"     streaming {len(blocks):,} blocks of {BLOCK:,} cells with {WORKERS} workers")
        say(f"     bounded sliding window of {2*WORKERS} in-flight blocks, a "
            f"{2*WORKERS*BLOCK*NGENE*4/1e9:.1f} GB ceiling. The first attempt used an unbounded "
            f"executor map, which submits all {len(blocks):,} blocks at once and holds every finished "
            f"one until the consumer reaches it; at 67.6 MB per block that reached the container "
            f"limit and the run was killed with no traceback.")
        done = [0]; tstart = time.time()
        from collections import deque
        with ThreadPoolExecutor(WORKERS) as ex:
            pend, it = deque(), iter(blocks)
            for _ in range(2 * WORKERS):
                b0 = next(it, None)
                if b0 is None:
                    break
                pend.append((b0, ex.submit(fetch_cells, *b0)))
            while pend:
                (lo, hi), fut = pend.popleft()
                A = fut.result()
                nxt = next(it, None)
                if nxt is not None:
                    pend.append((nxt, ex.submit(fetch_cells, *nxt)))
                m = keep[lo:hi]
                if not m.any():
                    continue
                A = A[m].astype(np.float32, copy=False)
                c = code[lo:hi][m]
                accum(A, c, S, n)
                accum(A * A, c, Q, np.zeros(npert, np.int64))
                half = (np.arange(lo, hi)[m] % 2) == 0
                if half.any():
                    accum(A[half], c[half], SA, nA)
                accum(A, code_sh[lo:hi][m], SX, nX)
                done[0] += 1
                if done[0] % 120 == 0:
                    el = time.time() - tstart
                    say(f"       {done[0]:,}/{len(blocks):,} blocks   {el/60:.1f} min   "
                        f"{done[0]*BLOCK*NGENE*4/1e6/el:.0f} MB/s")
        say(f"     stream complete in {(time.time()-tstart)/60:.1f} min, "
            f"{int(n.sum()):,} cells accumulated")
        np.savez(CKPT, S=S, Q=Q, SA=SA, SX=SX, n=n, nA=nA, nX=nX, code_sh=code_sh)
        say(f"     accumulators checkpointed to {CKPT.name} so a later gate failure cannot cost "
            f"the stream again")

    ok = (n >= MINCELL) & (np.arange(npert) != NT)
    M = np.zeros_like(S); M[ok] = S[ok] / n[ok, None]
    say(f"     {int(ok.sum()):,} perturbations with at least {MINCELL} cells")

    # ---------------------------------------------------------------- X1 part two
    bulkf = BULK / "K562_gwps_normalized_bulk_01.h5ad"
    cmp_r, cmp_d = float("nan"), float("nan")
    try:
        bh = h5py.File(bulkf, "r")
        bidx = h5_index(bh["obs"])
        bg = h5_index(bh["var"])
        bX = bh["X"]
        # index entries look like "0_A1BG_P1_ENSG00000121410": {row}_{symbol}_{promoter}_{ensg}.
        # The first field is a row counter, not an identifier; the perturbation name is field 1.
        bkey = np.array([(s.split("_")[1] if s.count("_") >= 3 else s) for s in bidx])
        # 789 symbols (8.0%) carry SEPARATE P1 and P2 rows in the published file, 2,180 rows in
        # total. Accumulation here is keyed on obs/gene, which is the symbol, so a symbol-level
        # mean spans both promoters while a published row covers one. A dict built over all rows
        # silently keeps the last, comparing an average of both against one -- which is what drove
        # the previous run's r to +0.98233. Measured directly: single-promoter symbols reproduce at
        # Pearson +1.00000 with median |difference| 0.00000, multi-promoter symbols at +0.62670.
        # The read is exact; the comparison was not. Restricted to the unambiguous symbols.
        uk, kc = np.unique(bkey, return_counts=True)
        multi = set(uk[kc > 1])
        pos = {k: i for i, k in enumerate(bkey) if k not in multi}
        say(f"     {len(multi):,} symbols have multiple promoter rows and are excluded from the "
            f"comparison; a symbol-level mean is not comparable to a single-promoter row")
        rows = [(i, pos[cats[i]]) for i in np.where(ok)[0] if cats[i] in pos]
        gmap = {g: i for i, g in enumerate(bg)}
        gsel = [(j, gmap[gid[j]]) for j in range(NGENE) if gid[j] in gmap]
        say(f"     bulk index columns: obs '{bh['obs'].attrs.get('_index')}', "
            f"var '{bh['var'].attrs.get('_index')}'; matched {len(gsel):,} genes by gene_id")
        if len(rows) >= 50 and len(gsel) >= 500:
            sub = rows[:: max(1, len(rows) // 300)][:300]
            gj = np.array([x[0] for x in gsel]); gb = np.array([x[1] for x in gsel])
            mine = np.array([M[i, gj] for i, _ in sub])
            theirs = np.array([np.asarray(bX[b, :])[gb] for _, b in sub])
            cmp_r = pear(mine, theirs)
            # ~17.8% of Perturb-seq rows carry non-finite values (loop 208 A4). pear() screens
            # them; the median did not, and returned nan while the correlation was fine.
            dif = np.abs(mine - theirs)
            dif = dif[np.isfinite(dif)]
            cmp_d = float(np.median(dif)) if dif.size else float("nan")
            say(f"     finite value pairs in the comparison: {dif.size / mine.size:.1%}")
            say(f"     recomputed pseudobulk vs published bulk, {len(sub)} perturbations x "
                f"{len(gj):,} genes: Pearson {cmp_r:+.5f}, median |difference| {cmp_d:.5f}")
        bh.close()
    except Exception as e:
        say(f"     published-bulk comparison could not run: {type(e).__name__}: {e}")
    ok1 = bool(off_err == 0.0 and np.isfinite(cmp_r) and cmp_r > 0.99 and cmp_d < 0.01)
    G.add("X1", ok1, stat=float(cmp_r),
          if_true=lambda: f"X1 PASS -- byte offsets exact and the recomputed pseudobulk matches "
                          f"the published file at r {cmp_r:+.4f}, median |diff| {cmp_d:.4f}",
          if_false=lambda: f"X1 FAIL -- offset error {off_err:.1e}, pseudobulk r {cmp_r:+.4f}, "
                           f"median |diff| {cmp_d:.4f}; nothing below may be read")
    res["integrity"] = {"offset_err": off_err, "bulk_r": cmp_r, "bulk_med_absdiff": cmp_d,
                        "cells_used": int(n.sum()), "perturbations": int(ok.sum())}

    # ---------------------------------------------------------------- X2
    say("X2 WHAT IS THE MEASURED RELIABILITY OF A PSEUDOBULK VALUE?")
    nn = n[ok][:, None].astype(np.float64)
    Sk, Qk, Mk = S[ok].astype(np.float64), Q[ok].astype(np.float64), M[ok].astype(np.float64)
    within = np.sum(Qk - Sk * Sk / nn, axis=0) / max(float(nn.sum() - ok.sum()), 1.0)
    between = np.var(Mk, axis=0, ddof=1) - within * np.mean(1.0 / nn[:, 0])
    nbar = float(np.median(n[ok]))
    wp = (within > 0) & np.isfinite(within)
    bp = (between > 0) & np.isfinite(between)
    relg = np.where(wp & bp, between / (between + within / nbar), np.nan)
    frac_ok = float(np.mean(wp & bp))
    say(f"     median cells per perturbation {nbar:.0f}")
    say(f"     per-gene within-perturbation variance: median {np.nanmedian(within):.4f}")
    say(f"     per-gene between-perturbation variance: median {np.nanmedian(between):.5f}")
    say(f"     both components strictly positive for {frac_ok:.1%} of genes")
    say(f"     RELIABILITY of a {nbar:.0f}-cell pseudobulk value: median "
        f"{np.nanmedian(relg):.4f}, quartiles {np.nanpercentile(relg,25):.4f} / "
        f"{np.nanpercentile(relg,75):.4f}")
    G.add("X2", bool(frac_ok >= 0.90), stat=float(frac_ok), requires=("X1",),
          if_true=lambda: f"X2 PASS -- the decomposition is well-posed for {frac_ok:.0%} of genes; "
                          f"median reliability {np.nanmedian(relg):.3f} is reported, not gated",
          if_false=lambda: f"X2 FAIL -- only {frac_ok:.0%} of genes have both variance components "
                           f"positive, so the ratio has no denominator for the rest")
    res["decomposition"] = {"n_bar": nbar, "within_med": float(np.nanmedian(within)),
                            "between_med": float(np.nanmedian(between)),
                            "frac_wellposed": frac_ok,
                            "reliability_med": float(np.nanmedian(relg)),
                            "reliability_q25": float(np.nanpercentile(relg, 25)),
                            "reliability_q75": float(np.nanpercentile(relg, 75))}

    # ---------------------------------------------------------------- X3
    say("X3 DOES THE MEASURED NOISE MODEL ACTUALLY PREDICT REPRODUCIBILITY?")
    two = ok & (nA >= MINCELL // 2) & ((n - nA) >= MINCELL // 2)
    MA = SA[two] / nA[two, None]
    MB = (S[two] - SA[two]) / (n[two] - nA[two])[:, None]
    obs = np.array([pear(MA[:, j], MB[:, j]) for j in range(NGENE)])
    nh = float(np.median(nA[two]))
    pred = np.where(wp & bp, between / (between + within / nh), np.nan)
    m3 = np.isfinite(obs) & np.isfinite(pred)
    rp = pear(pred[m3], obs[m3])
    md = float(np.median(np.abs(pred[m3] - obs[m3])))
    say(f"     {int(two.sum()):,} perturbations split into halves of median {nh:.0f} cells")
    say(f"     observed split-half reliability across genes: median {np.nanmedian(obs[m3]):.4f}")
    say(f"     predicted from the variance decomposition:    median {np.nanmedian(pred[m3]):.4f}")
    say(f"     agreement Pearson {rp:+.4f}   median |predicted - observed| {md:.4f}")
    G.add("X3", bool(rp >= 0.80 and md < 0.10), stat=float(rp), requires=("X1",),
          if_true=lambda: f"X3 PASS -- the noise model predicts real split-half reproducibility at "
                          f"r {rp:+.3f}, median error {md:.3f}; the floor is a measured quantity",
          if_false=lambda: f"X3 FAIL -- r {rp:+.3f} against a 0.80 bar, median error {md:.3f} "
                           f"against 0.10; cell-to-cell noise is not independent as assumed")
    res["splithalf"] = {"r": float(rp), "med_abs_err": md, "n_pert": int(two.sum()),
                        "obs_med": float(np.nanmedian(obs[m3])),
                        "pred_med": float(np.nanmedian(pred[m3]))}

    # ---------------------------------------------------------------- X4
    say("X4 ARE THE PER-CELL RESPONSES UNIMODAL?")
    Mo = np.where(np.isfinite(Mk), Mk, 0.0)
    flat = np.argsort(-np.abs(Mo).ravel())[:NSTRONG]
    pi, gi = np.unravel_index(flat, Mo.shape)
    pidx = np.where(ok)[0][pi]
    def bimod(v):
        v = v[np.isfinite(v)]
        if len(v) < 20: return np.nan
        v = v - v.mean(); s = v.std()
        if s <= 0: return np.nan
        v = v / s
        m3_, m4_ = np.mean(v ** 3), np.mean(v ** 4)
        return float((m3_ ** 2 + 1) / max(m4_, 1e-9))
    need = {}
    for p, gg in zip(pidx, gi):
        need.setdefault(int(p), []).append(int(gg))
    cells_of = {p: np.where(code == p)[0] for p in need}
    say(f"     re-streaming cells for {len(need)} perturbations carrying the {NSTRONG} strongest "
        f"effects")
    vals, GAP, failed = {}, 64, 0
    for p, idxs in cells_of.items():
        runs, st, prev = [], idxs[0], idxs[0]
        for b in idxs[1:]:
            if b - prev > GAP:
                runs.append((st, prev + 1)); st = b
            prev = b
        runs.append((st, prev + 1))
        try:
            got = {}
            for a, b in runs:
                blk = fetch_cells(a, b)
                for c in idxs[(idxs >= a) & (idxs < b)]:
                    got[int(c)] = blk[int(c) - a]
            rows = np.vstack([got[int(c)] for c in idxs])
        except Exception as e:
            failed += 1
            say(f"       perturbation {p}: {type(e).__name__}: {e} -- skipped")
            continue
        for gg in need[p]:
            vals[(p, gg)] = rows[:, gg].astype(np.float64)
    say(f"     {len(vals)} of {NSTRONG} effects recovered "
        f"({failed} perturbations unreachable, coalescing runs with a {GAP}-cell gap tolerance)")
    BIM = 0.555
    have = [(int(p), int(gg)) for p, gg in zip(pidx, gi) if (int(p), int(gg)) in vals]
    real = np.array([bimod(vals[k]) for k in have])
    ctrl = np.array([bimod(rng.normal(size=len(vals[k]))) for k in have])
    fr_real = float(np.nanmean(real > BIM)); fr_ctrl = float(np.nanmean(ctrl > BIM))
    say(f"     bimodality coefficient above {BIM} (the uniform-distribution reference):")
    say(f"       real per-cell distributions        {fr_real:.1%} of {len(have)}")
    say(f"       matched Gaussian samples, same n   {fr_ctrl:.1%}  <- false-positive rate")
    say(f"     real median coefficient {np.nanmedian(real):.4f}, "
        f"Gaussian control median {np.nanmedian(ctrl):.4f}")
    G.add("X4", bool(fr_ctrl < 0.10), stat=float(fr_ctrl), requires=("X1",),
          if_true=lambda: f"X4 PASS -- the control is calibrated at {fr_ctrl:.1%} false positives, "
                          f"so the observed {fr_real:.1%} is readable as an answer",
          if_false=lambda: f"X4 FAIL -- the Gaussian control itself is flagged {fr_ctrl:.1%} of "
                           f"the time, so the test cannot distinguish and {fr_real:.1%} means "
                           f"nothing")
    res["modality"] = {"frac_real": fr_real, "frac_control": fr_ctrl,
                       "real_med": float(np.nanmedian(real)), "n_recovered": len(have),
                       "ctrl_med": float(np.nanmedian(ctrl)), "n_tested": NSTRONG}

    # ---------------------------------------------------------------- X5
    say("X5 CONTROL: DOES RELIABILITY COLLAPSE UNDER SHUFFLED ASSIGNMENT?")
    okx = (nX >= MINCELL) & (np.arange(npert) != NT)
    MX = SX[okx] / nX[okx, None].astype(np.float64)
    betX = np.var(MX, axis=0, ddof=1) - within * np.mean(1.0 / nX[okx].astype(np.float64))
    relx = np.where(wp & (betX > 0), betX / (betX + within / nbar), 0.0)
    say(f"     shuffled: median reliability {np.nanmedian(relx):.5f}   "
        f"real: {np.nanmedian(relg):.5f}")
    G.add("X5", bool(np.nanmedian(relx) < 0.05 < np.nanmedian(relg)), stat=float(np.nanmedian(relx)),
          requires=("X1",),
          if_true=lambda: f"X5 PASS -- shuffling cell-to-perturbation assignment collapses "
                          f"reliability to {np.nanmedian(relx):.4f} against the real "
                          f"{np.nanmedian(relg):.4f}",
          if_false=lambda: f"X5 FAIL -- shuffled reliability {np.nanmedian(relx):.4f} against real "
                           f"{np.nanmedian(relg):.4f}; the estimator produces structure from "
                           f"nothing and X2 cannot be trusted")
    res["shuffle"] = {"rel_shuffled_med": float(np.nanmedian(relx)),
                      "rel_real_med": float(np.nanmedian(relg))}

    # ---------------------------------------------------------------- X6
    say("X6 DOES THE MEASURED FLOOR IMPROVE CROSS-CELL-LINE TRANSFER?")
    uw, ww = float("nan"), float("nan")
    try:
        rh = h5py.File(BULK / "rpe1_normalized_bulk_01.h5ad", "r")
        ridx = h5_index(rh["obs"])
        rg = h5_index(rh["var"])
        rkey = np.array([(s.split("_")[1] if s.count("_") >= 3 else s) for s in ridx])
        rpos = {k: i for i, k in enumerate(rkey)}
        shared_p = [(i, rpos[cats[i]]) for i in np.where(ok)[0] if cats[i] in rpos]
        rgm = {g: i for i, g in enumerate(rg)}
        shared_g = [(j, rgm[gid[j]]) for j in range(NGENE) if gid[j] in rgm]
        say(f"     shared perturbations {len(shared_p):,}, shared genes {len(shared_g):,}")
        if len(shared_p) >= 100 and len(shared_g) >= 500:
            sp = shared_p[:: max(1, len(shared_p) // 400)][:400]
            gj = np.array([x[0] for x in shared_g]); gr = np.array([x[1] for x in shared_g])
            K = np.array([M[i, gj] for i, _ in sp])
            R = np.array([np.asarray(rh["X"][b, :])[gr] for _, b in sp])
            w = np.nan_to_num(relg[gj], nan=0.0); w = np.clip(w, 0, None)
            uw = pear(K, R)
            fm = np.isfinite(K) & np.isfinite(R)
            Wm = np.broadcast_to(w, K.shape)[fm]
            a, b_ = K[fm], R[fm]
            am = np.sum(Wm * a) / np.sum(Wm); bm = np.sum(Wm * b_) / np.sum(Wm)
            ww = float(np.sum(Wm * (a - am) * (b_ - bm)) /
                       np.sqrt(np.sum(Wm * (a - am) ** 2) * np.sum(Wm * (b_ - bm) ** 2)))
            say(f"     K562 vs RPE1 on {len(sp)} shared perturbations x {len(gj):,} genes:")
            say(f"       unweighted                 {uw:+.5f}")
            say(f"       reliability-weighted       {ww:+.5f}   delta {ww-uw:+.5f}")
        rh.close()
    except Exception as e:
        say(f"     RPE1 comparison could not run: {type(e).__name__}: {e}")
    G.add("X6", bool(np.isfinite(ww) and np.isfinite(uw) and ww > uw), stat=float(ww),
          requires=("X1", "X2"), void_if=(not np.isfinite(ww)),
          void_reason="the RPE1 file did not yield a comparable matrix",
          if_true=lambda: f"X6 PASS -- weighting by measured reliability raises cross-line "
                          f"agreement from {uw:+.4f} to {ww:+.4f}",
          if_false=lambda: f"X6 FAIL -- weighted {ww:+.4f} against unweighted {uw:+.4f}")
    res["crossline"] = {"unweighted": uw, "weighted": ww}

    # ---------------------------------------------------------------- X7
    say("X7 WHAT THIS CANNOT SHOW")
    say("     These are K562 CRISPRi knockdowns, not A549 dexamethasone. A floor measured here")
    say("     transfers to the A549 series as an order of magnitude, not as a number.")
    say("     The X matrix is the NORMALIZED one -- already z-scored against non-targeting")
    say("     controls within gem group -- so some cell-to-cell variance was removed before we")
    say("     saw it. That biases within-perturbation variance DOWN and reliability UP.")
    say("     A split-half shares library prep, day and gem group between halves, so X3 measures")
    say("     reproducibility of sampling, not of the experiment.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed("outputs/loop224_reliability.npz", gene=gname,
                        reliability=relg.astype(np.float32),
                        within=within.astype(np.float32), between=between.astype(np.float32))
    say(f"     written {OUT} and outputs/loop224_reliability.npz")


if __name__ == "__main__":
    main()
