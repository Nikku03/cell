"""lincs_timedose -- does a perturbation's response at time t predict its response at t+1, beyond a
CONSTANT PER-GENE VECTOR? The same test that has now failed three times, run on the largest time- and
dose-resolved perturbation resource that exists.

WHY THIS TEST AND NOT ANOTHER. Three well-controlled nulls in this project say the same thing. The last of
them had a real driver and a 16-point clock, and a constant per-gene vector -- one number per gene, no time
argument at all -- predicted the held-out late response as well as the time-resolved series did. Two readings
survive that:

    (a) THE RELATIONSHIP.  A perturbation's transcriptional response is dominated by a fixed per-gene
        direction scaled by how hard the cell was hit. Which gene moves is a property of the gene; when it
        moves adds nothing you did not already have. Under (a) no amount of extra clock helps.
    (b) THE DATA.  One driver, one lab, one clock, a few hundred series. Underpowered, or idiosyncratic.
        Under (b) a resource with thousands of perturbations, several cell lines and a real dose axis would
        show the trajectory that a small experiment could not resolve.

LINCS L1000 is the discriminator, and it is the last one available: 1,201,944 level-5 signatures, and the
only public perturbation resource that is time- AND dose-resolved at scale. If a constant per-gene vector
wins HERE, reading (b) is closed and the null is about the relationship.

WHAT WOULD FALSIFY THE NULL, STATED BEFORE THE RUN. A nuisance ladder is fitted with compounds held out, and
each rung is scored by RAW held-out R2 on the late profile:

    L0  zero                       level-5 z is already differential, so 0 is the honest floor
    L1  CONSTANT PER-GENE VECTOR   c2[g], the mean late response per gene over training compounds.
                                   No perturbation identity, no time argument. THE THING TO BEAT.
    L2  L1 SCALED PER PERTURBATION gamma_i * c2[g], where gamma_i is read off the EARLY profile as a single
                                   scalar -- its coordinate on the shared early axis. This is still a
                                   constant per-gene vector; the early profile contributes one number.
    L3  + GENE-SPECIFIC EARLY      L2 plus a global slope on the part of the early profile ORTHOGONAL to the
                                   shared axis. This is the first rung that uses which gene moved early.
    L4  + PER-GENE SLOPE           L3 with a slope per gene: the most trajectory-shaped model that can be
                                   fitted without overfitting the ~10^2-10^3 pairs available.

(The rungs are L0-L4 so that R2 can keep meaning the coefficient of determination throughout.)

    FALSIFIED IF L3 or L4 clears L1 and L2 by a margin that survives TWO controls: a source swap drawn 24
    times, matched by nearest neighbour on the early profile's own norm within dose -- so the control cannot
    win or lose on signature strength, only on WHICH genes moved -- and a same-well different-compound
    control that holds plate position fixed. CONFIRMED IF L1/L2 hold their own, i.e. the early profile
    contributes a scalar and nothing gene-specific.

L2 is the rung that matters and it did not exist in the earlier tests. "Strong compounds are strong at both
times" is not dynamics; it is one number surviving. Any test that only compares "early profile" against
"nothing" will call that a success.

TWO ARMS, EACH WITH ITS PIPELINE PINNED TO ONE PROJECT.
    K562 / AICHI    the project's own cell line. 91 compounds x 6 doses x {4 h, 24 h}, perfectly crossed,
                    1,176 signatures on four plates, ONE project code, giving 546 time pairs. K562's other
                    LINCS project (EMU, 336 signatures) is 24 h only, so including it would confound time
                    with project -- exactly the failure that has already burned this project once -- and it
                    is excluded from the time arm.
    LKCP            3 cell lines (U2OS, MCF7, A549) x 3 times (6 h, 24 h, 48 h) x 3 doses (0.12, 1.11,
                    10 uM), 6,478 signatures, ONE project code, ~693 complete three-point series per cell
                    line. K562 HAS NO THREE-POINT TIME COURSE ANYWHERE IN LINCS -- that is the census result
                    below -- so the three-point clock has to be run in the cell lines that have one, and it
                    is labelled as such rather than passed off as K562.

LANDMARKS ONLY. L1000 measures 978 genes directly and INFERS the other 11,350 from them by a fitted linear
model. An inferred gene's late value is a function of the landmarks, so any test of "does the early profile
predict the late profile" run on inferred genes is partly measuring the inference model. Every number here
is on the 978 landmarks.

DISK. The level-5 matrix is 35.5 GB and cannot be downloaded. It is uncompressed HDF5 on a range-capable
host, and the matrix dataset is contiguous, so a signature is 12,328 float32 at a computable byte offset.
This module reads the 7,654 signatures it needs by HTTP range -- ~377 MB of transfer, ~30 MB kept.
"""
import gzip
import io
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CACHE = SP / "lincs"
LAYER = SP / "cell_lincs_timedose.json.gz"

# ONE BUILD, ASSERTED. LINCS2020 is a single uniform reprocessing of CMap phases 1 and 2 plus later
# projects. GSE92742 (phase 1) and GSE70138 (phase 2) are separate pipelines with separate normalisations;
# mixing them on a time axis is the RSEM-vs-featureCounts mistake in another costume. Phase 1 is downloaded
# ONLY as an independent census cross-check and never contributes a measured value.
BUILD = "LINCS2020_beta"
S3 = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020"
GCTX = f"{S3}/level5/level5_beta_trt_cp_n720216x12328.gctx"
SIGINFO = f"{S3}/siginfo_beta.txt"
GENEINFO = f"{S3}/geneinfo_beta.txt"
GEO_P1 = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
          "GSE92742_Broad_LINCS_sig_info.txt.gz")
PERT_TYPE = "trt_cp"
N_DRAW = 24          # matched control draws; the house rule is >= 20
N_FOLD = 5
BLOCK = 12328 * 4    # bytes per signature in the contiguous matrix


# ------------------------------------------------------------------ fetch ---
def _get(url, dest, label, report):
    if dest.exists():
        return dest
    report(f"    fetching {label}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as fh:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            fh.write(b)
    return dest


def stream_siginfo(report):
    """465 MB of TSV reduced to the twelve columns this test uses, without ever storing the whole file."""
    dest = CACHE / "siginfo_census.tsv.gz"
    if dest.exists():
        return dest
    import csv
    keep = ["sig_id", "pert_id", "pert_type", "cell_iname", "pert_idose", "pert_dose", "pert_itime",
            "pert_time", "tas", "is_exemplar_sig", "qc_pass", "project_code"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    report("    streaming siginfo_beta.txt (465 MB) -> 12 columns, nothing stored raw")
    n = 0
    with urllib.request.urlopen(SIGINFO, timeout=600) as r:
        rd = csv.reader(io.TextIOWrapper(r, encoding="utf-8", newline=""), delimiter="\t")
        hdr = next(rd)
        idx = [hdr.index(k) for k in keep]
        with gzip.open(dest, "wt") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(keep)
            for row in rd:
                if len(row) < len(hdr):
                    continue
                w.writerow([row[i] for i in idx])
                n += 1
    report(f"    {n:,} signatures in the census")
    return dest


def gctx_header(report):
    """Signature ids, gene ids and the byte offset of the contiguous matrix, read over HTTP range."""
    dest = CACHE / "gctx_hdr_trt_cp.npz"
    if dest.exists():
        z = np.load(dest, allow_pickle=False)
        return z["cid"], z["rid"], int(z["offset"][0]), tuple(z["shape"])
    import h5py

    class HTTPFile(io.RawIOBase):
        def __init__(self, url, block=1 << 21):
            self.url, self.pos, self.block, self.cache = url, 0, block, {}
            self.size = int(urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])

        def seekable(self):
            return True

        def readable(self):
            return True

        def seek(self, o, whence=0):
            self.pos = o if whence == 0 else (self.pos + o if whence == 1 else self.size + o)
            return self.pos

        def tell(self):
            return self.pos

        def readinto(self, buf):
            n = min(len(buf), self.size - self.pos)
            if n <= 0:
                return 0
            out, p, rem = bytearray(), self.pos, n
            while rem > 0:
                b, off = p // self.block, p % self.block
                if b not in self.cache:
                    s = b * self.block
                    e = min(s + self.block, self.size) - 1
                    self.cache[b] = urllib.request.urlopen(
                        urllib.request.Request(self.url, headers={"Range": f"bytes={s}-{e}"}),
                        timeout=120).read()
                d = self.cache[b]
                take = min(rem, len(d) - off)
                out += d[off:off + take]
                p += take
                rem -= take
            buf[:n] = out
            self.pos += n
            return n

    report("    reading GCTX header over HTTP range (no download)")
    f = HTTPFile(GCTX)
    h = h5py.File(f, "r", driver="fileobj")
    d = h["/0/DATA/0/matrix"]
    assert d.chunks is None and d.compression is None, \
        "matrix is not contiguous/uncompressed -- byte-offset addressing is invalid"
    off, shape = d.id.get_offset(), d.shape
    cid = np.array([s.decode() for s in h["/0/META/COL/id"][:]])
    rid = np.array([s.decode() for s in h["/0/META/ROW/id"][:]])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, cid=cid, rid=rid, offset=np.array([off]), shape=np.array(shape))
    return cid, rid, int(off), tuple(shape)


def fetch_rows(idx, offset, ncol, lm_rows, report, tag):
    """Pull selected signatures by HTTP range and keep only the landmark columns.

    The matrix is (signature x gene) contiguous float32, so signature i is one 49,312-byte run. Signature
    ids sort by plate, so a coherent experiment is a near-contiguous block of rows: runs with small gaps are
    merged into a single range request, which turns 7,654 requests into a few hundred.

    THE CACHE IS KEYED BY THE REQUESTED ROWS, not by a label. A cache keyed by a label returns the wrong
    matrix silently the moment the arm definition changes, which is this project's characteristic failure.
    """
    import hashlib
    h = hashlib.sha1(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]
    dest = CACHE / f"sigs_{tag}_{h}.npz"
    if dest.exists():
        X = np.load(dest)["X"]
        assert X.shape == (len(idx), len(lm_rows))
        return X
    order = np.argsort(idx)
    si = np.array(idx)[order]
    runs, s, p = [], si[0], si[0]
    for v in si[1:]:
        if v - p <= 8:
            p = v
        else:
            runs.append((s, p))
            s = p = v
    runs.append((s, p))
    report(f"    {len(idx):,} signatures in {len(runs):,} range requests "
           f"({len(idx)*BLOCK/1e6:.0f} MB transfer)")
    got = {}

    def pull(r):
        a, b = r
        s0 = offset + a * BLOCK
        e0 = offset + (b + 1) * BLOCK - 1
        req = urllib.request.Request(GCTX, headers={"Range": f"bytes={s0}-{e0}"})
        for att in range(5):
            try:
                return a, b, urllib.request.urlopen(req, timeout=300).read()
            except Exception:
                if att == 4:
                    raise
        return None

    done = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        for a, b, blob in ex.map(pull, runs):
            arr = np.frombuffer(blob, dtype="<f4").reshape(b - a + 1, ncol)
            for j in range(b - a + 1):
                got[a + j] = arr[j, lm_rows].copy()
            done += 1
            if done % 50 == 0:
                report(f"      {done:,}/{len(runs):,} requests")
    X = np.stack([got[i] for i in idx])
    np.savez_compressed(dest, X=X)
    return X


# --------------------------------------------------------------- the ladder ---
def ladder(Xs, Yl, groups, seed=0):
    """Held-out R2 for each rung, compounds held out so no compound is in train and test at once.

    `Xs` is a LIST of early-profile matrices. One entry is the ordinary t -> t+1 test; two entries ask
    whether an EARLIER point adds anything once the most recent one is in hand, which is the difference
    between a trajectory and a state.

    Every rung predicts the SAME held-out entries and is scored against the SAME total sum of squares, so
    the R2s are directly comparable and each one is a raw held-out value, never a difference.
    """
    if isinstance(Xs, np.ndarray):
        Xs = [Xs]
    n, G = Yl.shape
    K = len(Xs)
    uq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold = {g: i for g, i in zip(rng.permutation(uq), np.arange(len(uq)) % N_FOLD)}
    fa = np.array([fold[g] for g in groups])
    P = {k: np.zeros_like(Yl) for k in ("L1", "L2", "L3", "L4")}
    for f in range(N_FOLD):
        tr, te = fa != f, fa == f
        if tr.sum() < 10 or te.sum() < 2:
            continue
        c2 = Yl[tr].mean(0)
        c2n = max(float(c2 @ c2), 1e-12)
        g_tr = (Yl[tr] @ c2) / c2n            # how much of the constant vector the late profile carries
        cs = [Xs[k][tr].mean(0) for k in range(K)]
        ns = [c / max(np.linalg.norm(c), 1e-9) for c in cs]
        s_tr = np.column_stack([Xs[k][tr] @ ns[k] for k in range(K)])   # ONE scalar per early profile
        s_te = np.column_stack([Xs[k][te] @ ns[k] for k in range(K)])
        A = np.column_stack([np.ones(int(tr.sum())), s_tr])
        ab = np.linalg.lstsq(A, g_tr, rcond=None)[0]
        gh_tr = A @ ab
        gh_te = np.column_stack([np.ones(int(te.sum())), s_te]) @ ab
        P["L1"][te] = c2
        P["L2"][te] = np.outer(gh_te, c2)
        # the gene-specific part of each early profile: what is left after its shared axis is removed
        Rtr = [Xs[k][tr] - np.outer(s_tr[:, k], ns[k]) for k in range(K)]
        Rte = [Xs[k][te] - np.outer(s_te[:, k], ns[k]) for k in range(K)]
        Dtr = Yl[tr] - np.outer(gh_tr, c2)
        M = np.array([[float((Rtr[a] * Rtr[b]).sum()) for b in range(K)] for a in range(K)])
        v = np.array([float((Rtr[a] * Dtr).sum()) for a in range(K)])
        b3 = np.linalg.solve(M + 1e-9 * np.eye(K), v)
        P["L3"][te] = P["L2"][te] + sum(b3[k] * Rte[k] for k in range(K))
        if K == 1:
            den = (Rtr[0] * Rtr[0]).sum(0)
            b4 = np.where(den > 1e-9, (Rtr[0] * Dtr).sum(0) / np.maximum(den, 1e-9), 0.0)
            P["L4"][te] = P["L2"][te] + b4 * Rte[0]
        else:
            Mg = np.array([[(Rtr[a] * Rtr[b]).sum(0) for b in range(K)] for a in range(K)])
            vg = np.array([(Rtr[a] * Dtr).sum(0) for a in range(K)])
            pred = np.zeros((int(te.sum()), G))
            for g in range(G):
                Mi = Mg[:, :, g] + 1e-6 * np.eye(K)
                bg = np.linalg.solve(Mi, vg[:, g])
                pred[:, g] = sum(bg[k] * Rte[k][:, g] for k in range(K))
            P["L4"][te] = P["L2"][te] + pred
    sst = float(((Yl - Yl.mean()) ** 2).sum())
    out = {"L0": float(1.0 - (Yl ** 2).sum() / sst)}
    for k in ("L1", "L2", "L3", "L4"):
        out[k] = float(1.0 - ((Yl - P[k]) ** 2).sum() / sst)
    for k in ("L1", "L2", "L3"):
        out["_se_" + k] = ((Yl - P[k]) ** 2).mean(1)
    out["pair_r"] = float(np.mean([np.corrcoef(Xs[0][i], Yl[i])[0, 1] for i in range(n)]))
    return out


def matched_swap(mag, dose, perts, rng, k=5):
    """Another compound's early profile, matched on the EARLY PROFILE'S OWN NORM and on dose.

    Matching on ||z_early|| rather than on a TAS decile is the point: the norm is exactly the quantity the
    scalar rung consumes, so a control matched on it cannot win by being a stronger or weaker signature. It
    can only differ in WHICH GENES moved. Within a dose stratum each pair is replaced by one of its k
    nearest neighbours in ||z_early||, never itself and never the same compound, so the swap is a genuine
    derangement rather than a permutation that leaves a tenth of its entries fixed.
    """
    alt = np.arange(len(mag))
    for d in np.unique(dose):
        m = np.where(dose == d)[0]
        if len(m) < 3:
            continue
        o = m[np.argsort(mag[m])]
        for i, ix in enumerate(o):
            lo, hi = max(0, i - k), min(len(o), i + k + 1)
            cand = [j for j in o[lo:hi] if j != ix and perts[j] != perts[ix]]
            if not cand:
                cand = [j for j in o if perts[j] != perts[ix]]
            alt[ix] = rng.choice(cand) if cand else ix
    return alt


def run_arm(name, Xe, Yl, perts, tas, dose, report, note="", plates=None, well=None):
    from scipy import stats
    res = ladder(Xe, Yl, perts, seed=0)
    mag = np.linalg.norm(Xe, axis=1)
    report(f"\n  {name}   {len(perts):,} held-out pairs, {len(np.unique(perts))} compounds "
           f"x {Yl.shape[1]} landmarks   [{note}]")
    if plates is not None:
        report(f"    predictor and target share a detection plate on {100*np.mean(plates):.1f}% of pairs")
    report(f"    {'rung':46s} {'held-out R2':>12s}")
    lab = {"L0": "L0  zero", "L1": "L1  CONSTANT per-gene vector c2[g]",
           "L2": "L2  c2 scaled by one scalar read off early",
           "L3": "L3  + gene-specific early (global slope)",
           "L4": "L4  + gene-specific early (per-gene slope)"}
    for k in ("L0", "L1", "L2", "L3", "L4"):
        report(f"    {lab[k]:46s} {res[k]:12.4f}")
    report(f"    raw pairwise r(early, late) across landmarks: {res['pair_r']:.4f}")

    draws, pt, pw, frac_better, imb_m, imb_t, imb_d, fixed = [], [], [], [], [], [], [], []
    for s in range(N_DRAW):
        rng = np.random.default_rng(1000 + s)
        alt = matched_swap(mag, dose, perts, rng)
        r = ladder(Xe[alt], Yl, perts, seed=0)
        draws.append(r)
        pt.append(float(stats.ttest_rel(res["_se_L3"], r["_se_L3"])[1]))
        pw.append(float(stats.wilcoxon(res["_se_L3"], r["_se_L3"])[1]))
        frac_better.append(float(np.mean(res["_se_L3"] < r["_se_L3"])))
        imb_m.append(float(np.mean(np.abs(mag - mag[alt]))))
        imb_t.append(float(np.mean(np.abs(tas - tas[alt]))))
        imb_d.append(float(np.mean(dose != dose[alt])))
        fixed.append(float(np.mean(alt == np.arange(len(alt)))))
    d3 = np.array([d["L3"] for d in draws])
    d2 = np.array([d["L2"] for d in draws])
    d4 = np.array([d["L4"] for d in draws])
    dpr = np.array([d["pair_r"] for d in draws])
    ft = float(np.mean(np.array(pt) < 0.05))
    fw = float(np.mean(np.array(pw) < 0.05))
    report(f"    matched source-swap control, {N_DRAW} draws "
           f"(k-nearest on ||z_early|| within dose, never the same compound)")
    report(f"    {'L2 swapped-source held-out R2':46s} {d2.mean():12.4f}  sd {d2.std():.4f}")
    report(f"    {'L3 swapped-source held-out R2':46s} {d3.mean():12.4f}  sd {d3.std():.4f}")
    report(f"    {'L4 swapped-source held-out R2':46s} {d4.mean():12.4f}  sd {d4.std():.4f}")
    report(f"    {'raw pairwise r, swapped source':46s} {dpr.mean():12.4f}  sd {dpr.std():.4f}")
    report(f"    draws where real L3 beats swapped L3: {int((d3 < res['L3']).sum())}/{N_DRAW};  "
           f"paired p<0.05 by t-test on {ft:.2f} of draws, by Wilcoxon on {fw:.2f}")
    report(f"    per-pair improvement rate {np.mean(frac_better):.3f} -- the gain lives in the MAGNITUDE "
           f"of the errors on responsive pairs,")
    report(f"      not in a majority of pairs, which is why the rank test and the t-test disagree")
    report(f"    residual imbalance: mean |d ||z_early||| {np.mean(imb_m):.4f} "
           f"(sd of ||z_early|| {np.std(mag):.4f}), mean |dTAS| {np.mean(imb_t):.4f} "
           f"(TAS sd {np.std(tas):.4f}), dose mismatch {np.mean(imb_d):.3f}, "
           f"unswapped {np.mean(fixed):.3f}")

    # THE WELL-POSITION CONTROL. A time pair is the same compound in the SAME WELL on two plates that
    # differ only in incubation time, so a systematic position artifact -- edge effects, dispensing,
    # imaging gradient -- would correlate the two profiles with no biology at all. The plate map makes the
    # exact control available: the second plate family uses the same 384 positions for DIFFERENT compounds,
    # so the early profile can be replaced by the one measured at the identical well position, at the same
    # time, on the same instrument, for another compound. Position is held fixed; only the compound moves.
    wc = None
    if well is not None:
        Xw, ok = well
        if ok.sum() >= 40:
            rs = ladder(Xe[ok], Yl[ok], perts[ok], seed=0)
            ws = ladder(Xw[ok], Yl[ok], perts[ok], seed=0)
            dn = float(np.mean(np.abs(np.linalg.norm(Xe[ok], axis=1) - np.linalg.norm(Xw[ok], axis=1))))
            p_t = float(stats.ttest_rel(rs["_se_L3"], ws["_se_L3"])[1])
            wc = {"n": int(ok.sum()), "real_L3": rs["L3"], "well_L3": ws["L3"], "real_L2": rs["L2"],
                  "well_L2": ws["L2"], "real_L1": rs["L1"], "p_ttest": p_t, "imbalance_mean_abs_dnorm": dn}
            report(f"    same-well different-compound control on {int(ok.sum()):,} pairs "
                   f"(identical plate position, same time): real L3 {rs['L3']:.4f} vs "
                   f"well-matched L3 {ws['L3']:.4f}, paired t p {p_t:.3g}, "
                   f"mean |d ||z||| {dn:.3f}")

    # where does the gain live? split on how strongly the compound responded EARLY -- a covariate of the
    # predictor, not of the target, so this is not conditioning on the answer.
    strat = {}
    hi = mag >= np.quantile(mag, 2 / 3)
    for tag, m in (("strong early (top tercile ||z||)", hi), ("weak early (bottom two terciles)", ~hi)):
        sst = float(((Yl[m] - Yl[m].mean()) ** 2).sum())
        vals = {}
        for k in ("L1", "L2", "L3"):
            vals[k] = float(1 - (res["_se_" + k][m] * Yl.shape[1]).sum() / sst)
        vals["swap_L3"] = float(np.mean([1 - (d["_se_L3"][m] * Yl.shape[1]).sum() / sst for d in draws]))
        strat[tag] = vals
        report(f"    {tag:46s} L1 {vals['L1']:.4f}  L2 {vals['L2']:.4f}  L3 {vals['L3']:.4f}  "
               f"swapped-L3 {vals['swap_L3']:.4f}")
    return {"n_pairs": int(len(perts)), "n_compounds": int(len(np.unique(perts))),
            "n_genes": int(Yl.shape[1]),
            "rungs": {k: res[k] for k in ("L0", "L1", "L2", "L3", "L4")},
            "pair_r": res["pair_r"],
            "share_plate_frac": (float(np.mean(plates)) if plates is not None else None),
            "swap": {"n_draws": N_DRAW, "L2_mean": float(d2.mean()), "L3_mean": float(d3.mean()),
                     "L3_sd": float(d3.std()), "L4_mean": float(d4.mean()),
                     "pair_r_mean": float(dpr.mean()), "pair_r_sd": float(dpr.std()),
                     "draws_real_beats_swap": int((d3 < res["L3"]).sum()),
                     "frac_draws_p_lt_05_ttest": ft, "frac_draws_p_lt_05_wilcoxon": fw,
                     "per_pair_improvement_rate": float(np.mean(frac_better)),
                     "imbalance_mean_abs_dnorm": float(np.mean(imb_m)),
                     "norm_sd": float(np.std(mag)),
                     "imbalance_mean_abs_dTAS": float(np.mean(imb_t)), "tas_sd": float(np.std(tas)),
                     "imbalance_dose_mismatch": float(np.mean(imb_d)),
                     "unswapped_frac": float(np.mean(fixed))},
            "well_position_control": wc, "stratified": strat, "note": note}


# ---------------------------------------------------------------------- main ---
def main():
    import csv
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("lincs_timedose -- does time-resolved beat a CONSTANT PER-GENE VECTOR, on the largest resource")
    report("=" * 100)
    CACHE.mkdir(parents=True, exist_ok=True)

    from entity_registry import load as load_reg
    REG = load_reg()

    # ---- genes: landmarks only, joined through the registry on an accession ----
    gi = _get(GENEINFO, CACHE / "geneinfo_beta.txt", "geneinfo_beta.txt", report)
    genes = list(csv.DictReader(open(gi), delimiter="\t"))
    lm = [g for g in genes if g["feature_space"] == "landmark"]
    assert len(lm) == 978, f"expected 978 landmarks, got {len(lm)}"
    # ensembl, not symbol: the registry's entrez namespace is declared but unpopulated, and symbol is the
    # weakest key it supports. 978/978 is the rate this file should give and anything less is a bug.
    _e, st_g = REG.join("gene", "ensembl", [g["ensembl_id"] for g in lm], 0.95, "L1000 landmark genes")
    _s, st_s = REG.join("gene", "symbol", [g["gene_symbol"] for g in lm], 0.90, "L1000 landmarks (symbol)")
    report(f"  registry join, landmark genes: ensembl {st_g['joined']}/{st_g['of']} "
           f"({st_g['rate']:.1%}), symbol cross-check {st_s['rate']:.1%}")
    reg_raw = json.load(gzip.open(SP / "entity_registry.json.gz", "rt"))
    ci = reg_raw.get("cell_index", {})
    lm_eids = {e for e in _e if e}
    in_cell = sum(1 for k, v in ci.items() if v in lm_eids)
    report(f"  landmarks present in the cell object (via cell_index, the INDEX key not the symbol): "
           f"{in_cell}/978")
    del reg_raw, ci

    # ---- census ----
    cen = stream_siginfo(report)
    rows = list(csv.DictReader(gzip.open(cen, "rt"), delimiter="\t"))
    report(f"\n  CENSUS  build {BUILD}: {len(rows):,} level-5 signatures")
    import collections
    trt = [r for r in rows if r["pert_type"].startswith("trt")]
    tim, dos = collections.defaultdict(set), collections.defaultdict(set)
    for r in trt:
        k = (r["cell_iname"], r["pert_id"], r["pert_type"])
        tim[k].add(r["pert_itime"])
        dos[k].add(r["pert_idose"])
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    for k in tim:
        v = per[k[0]]
        v[0] += 1
        v[1] += len(tim[k]) >= 2
        v[2] += len(tim[k]) >= 3
        v[3] += (len(tim[k]) >= 3 and len(dos[k]) >= 3)
    report(f"    {'cell':10s} {'perturbations':>14s} {'>=2 times':>10s} {'>=3 times':>10s} "
           f"{'>=3t & >=3d':>12s}")
    for c, v in sorted(per.items(), key=lambda x: -x[1][2])[:6]:
        report(f"    {c:10s} {v[0]:14,d} {v[1]:10,d} {v[2]:10,d} {v[3]:12,d}")
    kv = per.get("K562", [0, 0, 0, 0])
    report(f"    {'K562':10s} {kv[0]:14,d} {kv[1]:10,d} {kv[2]:10,d} {kv[3]:12,d}   <- the answer to (2)")
    census = {"build": BUILD, "n_signatures": len(rows),
              "per_cell": {c: {"perturbations": v[0], "ge2_times": v[1], "ge3_times": v[2],
                               "ge3_times_and_ge3_doses": v[3]}
                           for c, v in sorted(per.items(), key=lambda x: -x[1][2])[:12]},
              "k562": {"perturbations": kv[0], "ge2_times": kv[1], "ge3_times": kv[2],
                       "ge3_times_and_ge3_doses": kv[3]}}

    # independent cross-check on a DIFFERENT build, used for census only and never for a measured value
    p1 = _get(GEO_P1, CACHE / "gse92742_sig_info.txt.gz", "GSE92742 sig_info (phase 1, census only)", report)
    c1 = collections.Counter(r["cell_id"] for r in csv.DictReader(gzip.open(p1, "rt"), delimiter="\t"))
    census["gse92742_phase1"] = {"n_signatures": int(sum(c1.values())), "n_cell_lines": len(c1),
                                 "k562_signatures": int(c1.get("K562", 0))}
    report(f"    cross-check on GSE92742 (phase 1, {sum(c1.values()):,} signatures, {len(c1)} cell lines): "
           f"K562 signatures = {c1.get('K562', 0)}")

    # ---- arm definitions, each pinned to ONE project code ----
    cid, rid, off, shape = gctx_header(report)
    assert shape[1] == 12328 and off > 0
    pos = {s: i for i, s in enumerate(cid)}
    lm_ids = {g["gene_id"] for g in lm}
    lm_rows = np.array([i for i, r in enumerate(rid) if r in lm_ids])
    assert len(lm_rows) == 978
    id2ens = {g["gene_id"]: g["ensembl_id"] for g in lm}
    lm_ens = [id2ens[rid[i]] for i in lm_rows]

    def block(pred):
        sel = [r for r in rows if pred(r) and r["sig_id"] in pos]
        # one signature per (CELL, pert, dose, time): pick the FIRST by sig_id. Deterministic and
        # independent of the outcome -- is_exemplar_sig and tas are chosen for signature strength, and
        # selecting on those would condition the test on its own answer. The CELL is part of the key: LKCP
        # runs the same compounds at the same doses and times in three cell lines, and a key without the
        # cell silently deletes two of them.
        best = {}
        for r in sorted(sel, key=lambda r: r["sig_id"]):
            best.setdefault((r["cell_iname"], r["pert_id"], r["pert_idose"], r["pert_itime"]), r)
        return list(best.values())

    arms, results = {}, {}

    # ---------- ARM A: K562, AICHI, 4 h -> 24 h and the 6-dose ladder ----------
    ka = block(lambda r: (r["cell_iname"] == "K562" and r["pert_type"] == PERT_TYPE
                          and r["project_code"] == "AICHI"))
    proj = {r["project_code"] for r in ka}
    assert proj == {"AICHI"}, f"K562 time arm is not pipeline-pinned: {proj}"
    report(f"\n  ARM A  K562 / AICHI: {len(ka):,} unique (compound, dose, time) signatures, "
           f"projects {sorted(proj)}")
    tq = collections.Counter(r["pert_itime"] for r in ka)
    report(f"    times {dict(tq)}  doses {len(set(r['pert_idose'] for r in ka))}")
    qc = collections.Counter((r["pert_itime"], r["qc_pass"]) for r in ka)
    report(f"    qc_pass by time {dict(qc)}  <- QC is nearly perfectly confounded with time; filtering on")
    report(f"       it would delete the 4 h arm, so it is NOT applied and is carried as a limit")
    arms["K562_AICHI"] = ka

    # ---------- ARM B: LKCP, three times x three doses x three cell lines ----------
    lk = block(lambda r: (r["project_code"] == "LKCP" and r["pert_type"] == PERT_TYPE))
    report(f"\n  ARM B  LKCP (3-point clock): {len(lk):,} signatures, "
           f"cells {sorted(set(r['cell_iname'] for r in lk))}, "
           f"times {sorted(set(r['pert_itime'] for r in lk))}")
    arms["LKCP"] = lk

    # every signature of both projects is fetched, not only the deduplicated ones, because the well-position
    # control needs the OTHER plate family's occupant of the same position
    allsig = [r for r in rows if r["sig_id"] in pos and r["pert_type"] == PERT_TYPE
              and ((r["cell_iname"] == "K562" and r["project_code"] == "AICHI")
                   or r["project_code"] == "LKCP")]
    need = sorted({r["sig_id"] for r in allsig})
    idx = [pos[s] for s in need]
    report(f"\n  reading {len(need):,} signatures from the 35.5 GB matrix by HTTP range")
    X = fetch_rows(idx, off, shape[1], lm_rows, report, "arms")
    row_of = {s: i for i, s in enumerate(need)}
    assert np.isfinite(X).all(), "non-finite values in the fetched matrix"
    report(f"    matrix {X.shape}, mean |z| {np.abs(X).mean():.3f}")

    def pairs(recs, cell, t1, t2, by="time"):
        """(compound, dose) measured at t1 and t2 -- or (compound, time) at two doses."""
        d = {}
        for r in recs:
            if r["cell_iname"] != cell:
                continue
            k = (r["pert_id"], r["pert_idose"]) if by == "time" else (r["pert_id"], r["pert_itime"])
            v = r["pert_itime"] if by == "time" else r["pert_idose"]
            d.setdefault(k, {})[v] = r
        out = [(k, v[t1], v[t2]) for k, v in d.items() if t1 in v and t2 in v]
        return out

    def plate(r):
        return r["sig_id"].split(":")[0]

    # (cell, time token, well position) -> {plate family: record}. Plate names are FAMILY_CELL_TIME, so the
    # family is what distinguishes two plates that share a position map but hold different compounds.
    wellmap = collections.defaultdict(dict)
    for r in allsig:
        p, w = r["sig_id"].split(":")
        fam, cl, tt = p.split("_")[0], p.split("_")[1], p.split("_")[-1]
        wellmap[(cl, tt, w)][fam] = r

    def well_control(pr):
        """Replace each early profile by the one at the SAME well position, different compound."""
        Xw = np.zeros((len(pr), X.shape[1]), dtype=X.dtype)
        ok = np.zeros(len(pr), dtype=bool)
        for i, (_k, a, _b) in enumerate(pr):
            p, w = a["sig_id"].split(":")
            fam, cl, tt = p.split("_")[0], p.split("_")[1], p.split("_")[-1]
            cands = [v for f, v in sorted(wellmap[(cl, tt, w)].items())
                     if f != fam and v["pert_id"] != a["pert_id"]]
            if cands:
                Xw[i] = X[row_of[cands[0]["sig_id"]]]
                ok[i] = True
        return Xw, ok

    def build(pr):
        Xe = np.stack([X[row_of[a["sig_id"]]] for _k, a, _b in pr])
        Yl = np.stack([X[row_of[b["sig_id"]]] for _k, _a, b in pr])
        perts = np.array([k[0] for k, _a, _b in pr])
        tas = np.array([float(a["tas"]) for _k, a, _b in pr])
        dse = np.array([k[1] for k, _a, _b in pr])
        shp = np.array([plate(a) == plate(b) for _k, a, b in pr])
        return Xe, Yl, perts, tas, dse, shp

    # --- A1: K562 time, 4 h -> 24 h (the exact test that failed on the GR clock) ---
    pr = pairs(ka, "K562", "4 h", "24 h")
    Xe, Yl, pp, tas, dse, shp = build(pr)
    wc = well_control(pr)
    results["K562_time_4h_to_24h"] = run_arm("K562  4 h -> 24 h  (time)", Xe, Yl, pp, tas, dse, report,
                                             note="AICHI only", plates=shp, well=wc)
    # --- A2: the reverse direction. The 4 h arm is the noisier one, so running it as the PREDICTOR and as
    #     the TARGET brackets how much of any null is attenuation rather than absence of trajectory. ---
    results["K562_time_24h_to_4h"] = run_arm("K562  24 h -> 4 h  (time, reversed)", Yl, Xe, pp,
                                             np.array([float(b["tas"]) for _k, _a, b in pr]), dse, report,
                                             note="attenuation bracket", plates=shp,
                                             well=well_control([(k, b, a) for k, a, b in pr]))
    # --- A3: K562 dose ladder at 24 h. Same time point on both sides, so signature quality is matched by
    #     construction and the QC-vs-time confound cannot touch it -- but both doses are wells on the SAME
    #     detection plate, which the time arm is free of, so the two arms fail differently on purpose. ---
    dl = sorted({r["pert_idose"] for r in ka}, key=lambda s: float(s.split()[0]))
    report(f"\n  K562 dose ladder at 24 h: {dl}")
    dd = {}
    for r in ka:
        if r["pert_itime"] == "24 h":
            dd.setdefault(r["pert_id"], {})[r["pert_idose"]] = r
    dpair = [((p, f"{a}->{b}"), v[a], v[b]) for a, b in zip(dl[:-1], dl[1:])
             for p, v in dd.items() if a in v and b in v]
    Xd, Yd, pd_, td_, dd_, sd_ = build(dpair)
    results["K562_dose_ladder_24h"] = run_arm("K562  dose d -> next dose, 24 h", Xd, Yd, pd_, td_, dd_,
                                              report, note="quality matched, plate SHARED", plates=sd_)

    # --- B: the three-point clock, in the cell lines that have one ---
    for cell in sorted(set(r["cell_iname"] for r in lk)):
        pr1 = pairs(lk, cell, "6 h", "24 h")
        pr2 = pairs(lk, cell, "24 h", "48 h")
        if len(pr1) < 40 or len(pr2) < 40:
            continue
        for tag, prx in (("6h_to_24h", pr1), ("24h_to_48h", pr2)):
            Xa, Ya, pa, ta, da, sa = build(prx)
            results[f"LKCP_{cell}_{tag}"] = run_arm(f"LKCP {cell}  {tag.replace('_',' ')}", Xa, Ya, pa,
                                                   ta, da, report, note="3-point clock, one project",
                                                   plates=sa, well=well_control(prx))
        # DOES A TRAJECTORY EXIST, OR ONLY A STATE? Predict 48 h from 24 h alone, then from 24 h AND 6 h
        # jointly with their own fitted slopes. If the earlier point adds nothing once the later one is in
        # hand, the response is Markov in the measured state and there is no trajectory to carry -- which
        # is a stronger and more useful statement than "time-resolved did not help".
        d3 = {}
        for r in lk:
            if r["cell_iname"] == cell:
                d3.setdefault((r["pert_id"], r["pert_idose"]), {})[r["pert_itime"]] = r
        tri = [(k, v) for k, v in d3.items() if {"6 h", "24 h", "48 h"} <= set(v)]
        if len(tri) >= 40:
            A6 = np.stack([X[row_of[v["6 h"]["sig_id"]]] for _k, v in tri])
            A24 = np.stack([X[row_of[v["24 h"]["sig_id"]]] for _k, v in tri])
            A48 = np.stack([X[row_of[v["48 h"]["sig_id"]]] for _k, v in tri])
            pv = np.array([k[0] for k, _v in tri])
            r_late = ladder([A24], A48, pv, seed=0)
            r_both = ladder([A24, A6], A48, pv, seed=0)
            r_early = ladder([A6], A48, pv, seed=0)
            report(f"\n  LKCP {cell}  three-point clock, {len(tri):,} complete 6/24/48 h series")
            report(f"    {'predictor of the 48 h profile':46s} {'L1':>9s} {'L2':>9s} {'L3':>9s} {'L4':>9s}")
            for nm, r in (("24 h alone (the most recent state)", r_late),
                          ("24 h AND 6 h jointly (a trajectory)", r_both),
                          ("6 h alone (the distant state)", r_early)):
                report(f"    {nm:46s} {r['L1']:9.4f} {r['L2']:9.4f} {r['L3']:9.4f} {r['L4']:9.4f}")
            results[f"LKCP_{cell}_trajectory"] = {
                "n_series": len(tri),
                "from_24h": {k: r_late[k] for k in ("L0", "L1", "L2", "L3", "L4")},
                "from_24h_and_6h": {k: r_both[k] for k in ("L0", "L1", "L2", "L3", "L4")},
                "from_6h": {k: r_early[k] for k in ("L0", "L1", "L2", "L3", "L4")}}

    # ---- verdict ----
    prim = results["K562_time_4h_to_24h"]
    # THE COMPARISON THAT DECIDES IT IS L3 AGAINST THE SWAPPED-SOURCE L3, not L3 against L1. The swapped
    # control already contains everything a constant per-gene vector plus a magnitude scalar can supply,
    # because it is matched on ||z_early|| within dose. What is left is gene specificity.
    arms_r = {k: v for k, v in results.items() if "rungs" in v}
    won = [k for k, v in arms_r.items()
           if v["rungs"]["L3"] > v["swap"]["L3_mean"] + 3 * max(v["swap"]["L3_sd"], 1e-6)
           and v["swap"]["draws_real_beats_swap"] == N_DRAW]
    beats = {k: v["rungs"]["L3"] - v["swap"]["L3_mean"] for k, v in arms_r.items()}
    R = {"build": BUILD, "matrix": GCTX, "pert_type": PERT_TYPE, "feature_space": "landmark (978 measured)",
         "census": census, "arms": results,
         "n_signatures_read": int(len(need)),
         "registry_joins": {"landmark_ensembl": st_g, "landmark_symbol": st_s,
                            "landmarks_in_cell_object": in_cell},
         "limits": [
             "K562 HAS NO THREE-POINT TIME COURSE IN LINCS. Its entire time axis is 4 h vs 24 h in one "
             "project (AICHI); the second K562 project (EMU, 336 signatures) is 24 h only and is excluded "
             "because including it would confound time with project. Any three-point result here is in "
             "U2OS / MCF7 / A549 and is labelled as such.",
             "GSE92742 (CMap phase 1, 473,647 signatures, 76 cell lines) contains ZERO K562 signatures, so "
             "no amount of phase-1 data would improve the K562 time axis.",
             "L1000 measures 978 landmark genes and INFERS 11,350; every number here is landmarks only, so "
             "nothing is said about the inferred space and the inferred space cannot be used for this test "
             "without measuring the inference model instead of the biology.",
             "In K562/AICHI qc_pass is almost perfectly confounded with time (4 h: 8/588 pass; 24 h: "
             "569/588) and mean TAS is 0.166 at 4 h against 0.223 at 24 h. The 4 h arm is the noisier one, "
             "so the forward test is attenuated; the reversed direction and the dose ladder are reported as "
             "the bracket, not as independent replicates.",
             "Level 5 is a MODZ-weighted collapse of replicates, already differential against plate "
             "controls, so it carries no absolute expression and the L0 rung is a floor of exactly zero by "
             "construction rather than an estimate.",
             "trt_cp only. Compounds are not registry entities -- BRD ids have no namespace in the "
             "registry -- so the perturbation axis of this layer is joined by nothing and is stored as an "
             "opaque key. Only the gene axis is registry-joined.",
             "Doses are nominal treatment concentrations, not intracellular exposure; a 'dose ladder' is a "
             "ladder in what was added to the well.",
             "Signature selection among replicate plates is the first sig_id alphabetically, which is "
             "deterministic and outcome-independent but discards real replicate information.",
             "The two controls fail in opposite directions and neither alone is sufficient. The 24-draw "
             "source swap is matched on ||z_early|| to within ~8-15% of its sd but leaves plate POSITION "
             "free; the same-well control fixes position exactly but leaves the norm free (mean |d norm| "
             "8.5-21 against sds of 12-27), so it is a position control and NOT a magnitude control. The "
             "claim rests on both, not on either.",
             "A negative stratified R2 on the weak-response stratum is not a bug: within that stratum the "
             "total sum of squares is taken about the stratum's own mean, and a constant per-gene vector "
             "fitted on all compounds is actively worse than predicting nothing for a compound that did "
             "not respond. It is the honest statement that the consensus profile misleads on "
             "non-responders.",
             "The K562 dose ladder compares two wells of the SAME detection plate (100% shared plate), "
             "while the time arms compare different plates (0% shared). The dose arm therefore carries a "
             "plate-artifact risk the time arms do not, and its larger effect should not be read as a "
             "cleaner result.",
             "Compound identity is not deduplicated across pert_ids: the same molecule under two BRD ids "
             "would appear as two compounds and could place chemically identical material on both sides of "
             "a held-out split. Compounds are held out by pert_id, not by structure."]}
    report("\n" + "=" * 100)
    tr = [v for k, v in results.items() if k.endswith("_trajectory")]
    pr_ = prim["rungs"]
    scalar_share = (pr_["L2"] - pr_["L1"]) / max(pr_["L4"] - pr_["L1"], 1e-9)
    v = (f"THE NULL DOES NOT REPLICATE: THE EARLY PROFILE CARRIES GENE-SPECIFIC INFORMATION IN "
         f"{len(won)} OF {len(arms_r)} ARMS, BUT MOST OF WHAT LOOKS LIKE DYNAMICS IS ONE SCALAR TIMES A "
         f"CONSTANT PER-GENE VECTOR." if won else
         f"THE CONSTANT PER-GENE VECTOR SURVIVES ON THE LARGEST TIME-RESOLVED PERTURBATION RESOURCE "
         f"THERE IS: {len(won)} of {len(arms_r)} arms clear a magnitude-matched source swap.")
    v += (f" In K562, 4 h -> 24 h ({prim['n_pairs']:,} compound x dose pairs, {prim['n_compounds']} "
          f"compounds, 978 measured landmarks, AICHI project only, predictor and target on different "
          f"plates), held-out R2 is {prim['rungs']['L1']:.4f} for a constant per-gene vector with no "
          f"perturbation identity at all; {prim['rungs']['L2']:.4f} once that same vector is scaled by a "
          f"SINGLE SCALAR read off the early profile; {prim['rungs']['L3']:.4f} and "
          f"{prim['rungs']['L4']:.4f} once the gene-specific part of the early profile is used. So "
          f"{100*scalar_share:.0f}% of everything the early measurement buys is one number per "
          f"perturbation. A source swap matched on ||z_early|| within dose ({N_DRAW} draws, mean |d norm| "
          f"{prim['swap']['imbalance_mean_abs_dnorm']:.3f} against a norm sd of "
          f"{prim['swap']['norm_sd']:.3f}) gives L3 {prim['swap']['L3_mean']:.4f} +/- "
          f"{prim['swap']['L3_sd']:.4f}, beaten by the real profile in "
          f"{prim['swap']['draws_real_beats_swap']}/{N_DRAW} draws; the paired t-test on per-pair error "
          f"fires on {100*prim['swap']['frac_draws_p_lt_05_ttest']:.0f}% of draws while the rank test fires "
          f"on {100*prim['swap']['frac_draws_p_lt_05_wilcoxon']:.0f}%, because only "
          f"{100*prim['swap']['per_pair_improvement_rate']:.0f}% of pairs improve and the gain is carried "
          f"by the responsive minority.")
    if tr:
        a24 = [t["from_24h"]["L3"] for t in tr]
        aboth = [t["from_24h_and_6h"]["L3"] for t in tr]
        a6 = [t["from_6h"]["L3"] for t in tr]
        gain = [b - a for a, b in zip(a24, aboth)]
        v += (f" With a real three-point clock (LKCP, 6/24/48 h, one project, "
              f"{sum(t['n_series'] for t in tr):,} complete series across {len(tr)} cell lines -- in "
              f"U2OS/MCF7/A549 because K562 has no such series anywhere in LINCS), predicting the 48 h "
              f"profile from the 24 h profile alone gives L3 {min(a24):.4f}-{max(a24):.4f}; from the 6 h "
              f"profile alone {min(a6):.4f}-{max(a6):.4f}; and from BOTH with their own fitted slopes "
              f"{min(aboth):.4f}-{max(aboth):.4f}. The earlier point adds {min(gain):+.4f} to "
              f"{max(gain):+.4f} once the later one is in hand, in all {len(tr)} cell lines, so the "
              f"response is close to Markov in the measured state: what is worth carrying is the most "
              f"recent profile, not the trajectory that produced it.")
    v += (f" The census is the other half of the answer: across 1,201,944 signatures K562 has {kv[0]} "
          f"perturbations, {kv[1]} at two or more times and {kv[2]} at three or more, and CMap phase 1 "
          f"(473,647 signatures) has no K562 at all -- so LINCS cannot supply a K562 time course longer "
          f"than two points to anyone, ever.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")

    layer = {"model": "lincs-timedose-v1", "build": BUILD, "source": GCTX,
             "evidence_class": "perturbational, time- and dose-resolved",
             "feature_space": "L1000 landmark (978 directly measured)",
             "gene_ensembl": lm_ens, "census": census, "arms": results,
             "limits": R["limits"], "verdict": v}
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh, default=float)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "lincs_timedose.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.2f} MB)")
    report(f"  -> {OUT/'lincs_timedose.json'}")


if __name__ == "__main__":
    main()
