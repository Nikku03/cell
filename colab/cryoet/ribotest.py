"""RIBOTEST (G14) -- is the machine built? Count assembled ribosomes inside human cells we model (Jurkat, RPE1) in
cryo-electron tomograms, after proving the ribosome finder on human tomograms with hand-placed ribosomes.

Asked by the user: "do the two partner checks, then make the colab notebook" (to count ribosomes in the Jurkat and RPE1
pictures of the CryoET Data Portal as a first direct test of "is the machine built").
DATA (CryoET Data Portal, public bucket s3://cryoet-data-portal-public, read over HTTPS at run time; Ermel et al. 2024
Nat Methods; no data licence stated, so nothing is redistributed): HSC-3 unroofed-cell tomograms with ribosomes picked
by hand (dataset 10426, 6 tomograms, "ground truth"); Jurkat cryo-sections (10169-10171, 4 tomograms each); RPE1 cell
periphery (10002, 3 tomograms); HeLa FIB lamellae, untreated (10476, 6 tomograms; reported only). Membrane masks: the
portal's MemBrain-seg predictions (else its "membrane" masks). The list, paths and the DEV / EXAM split of the six
hand-picked tomograms (3 smallest sha256('cellbrain-cryoet:' + run) = DEV) are in colab/cryoet/bundle.json.
FINDER (template matching, fixed before any run): template = human 80S ribosome EMD-54265 (HEK cells, 3.0 A),
low-passed to 50 A and resampled to 25 A voxels (box 40 nm, spherical mask radius 15 nm); 300 uniformly random
rotations (seed) plus identity; each rotated template gets the tomogram's missing wedge (tilt axis y, the run's tilt
range, +-60 degrees where the portal records none -- Jurkat and RPE1); locally normalised cross-correlation by FFT; tomograms resampled to 25 A, density sign chosen on the DEV
tomograms (the sign with the higher DEV F1); score map = maximum over rotations, in robust z units (median, 1.4826 x
MAD over the interior); peaks = local maxima at least 20 nm apart, 20 nm from the border; threshold = the z that
maximises F1 on the DEV tomograms (grid 2.0 .. 12.0, step 0.25), then used unchanged everywhere. A pick matches a
hand-placed ribosome within 15 nm (one-to-one); picks are scored only inside the box the hand-picked ribosomes of
that tomogram span, widened by 15 nm (some tomograms are labelled only in part, e.g. one slice). CONTROL TEMPLATE: the same template with its Fourier phases
randomised (same power spectrum, same mask), run through the identical pipeline (its own DEV threshold).
PREDICTION (frozen in bundle.json before any run; built by rem/atlas/cryoetbundle.py): share of ribosomes making ER
proteins = sum over nuclear-encoded genes with a signal peptide or transmembrane segment of HPA nTPM x protein length
/ the same sum over all genes, from HPA RNA of each line.

CLAIMS (ids G14-*):
 G14-R1 THE FINDER FINDS RIBOSOMES. On the three EXAM hand-picked HSC-3 tomograms, F1 >= 0.50. Control: the
        random-phase template must stay below 0.50.
 G14-R2 THE CELL'S MEMBRANE-BOUND RIBOSOMES MATCH WHAT ITS RNA PREDICTS (Jurkat). Share of found ribosomes whose
        centre lies within 25 nm of a membrane, pooled over the Jurkat tomograms that have a
        membrane mask (8 of 12), is within 0.10 of the predicted
        share of ribosomes making ER proteins. Controls: the random-phase template must find at most a third as many
        particles in Jurkat as the real one; R1 must pass (otherwise the count cannot be trusted).
 REPORTED: per line (Jurkat, RPE1, HeLa): ribosomes found per tomogram and per cubic micrometre of tomogram, the
        membrane-near share, and the predicted share (RPE1); the portal's own automated ribosome picks scored against
        the hand-picked ones (context for R1).
NOT: template matching at 25 A also picks other large dense particles; cryo-sections (Jurkat) are compressed and old
(2011); RPE1 tomograms show the thin cell edge, not a whole cell; "within 25 nm of a membrane" includes plasma
membrane, mitochondria and vesicles, not only ER; the predicted share assumes ribosomes per mRNA scale with protein
length.

Run (Colab GPU): python ribotest.py --bundle bundle.json --out OUT [--trial]
"""

from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = "https://cryoet-data-portal-public.s3.us-west-2.amazonaws.com/"
EMD_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-54265/map/emd_54265.map.gz"
VOX, LOWPASS, BOX_NM, MASK_NM = 25.0, 50.0, 40.0, 15.0
MATCH_NM, MINSEP_NM, BORDER_NM, MEM_NM = 15.0, 20.0, 20.0, 25.0
N_ROT, SEED = 300, 20261125
ZGRID = np.arange(2.0, 12.0001, 0.25)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "ribotest.py"), "rb").read()); h.update(open(os.path.join(HERE, "bundle.json"), "rb").read())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ data access
def get(url, tries=5):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cellbrain-ribotest/1.0"}), timeout=300).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * 2 ** k)


def s3_keys(prefix):
    keys, tok = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if tok:
            q["continuation-token"] = tok
        x = get(B + "?" + urllib.parse.urlencode(q)).decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]*)<", x)
        if not m:
            return keys
        tok = m.group(1)


def zarr_shape(prefix):
    return json.loads(get(B + prefix + ".zarray"))["shape"]


def read_zarr_level(prefix, start=None, size=None):
    """one level of an OME-Zarr v2 array (blosc chunks, '/' separator) -> numpy array; optionally only a box
    (start, size), fetching just the chunks it touches."""
    from numcodecs import get_codec
    meta = json.loads(get(B + prefix + ".zarray")); shape, ch, dt = meta["shape"], meta["chunks"], np.dtype(meta["dtype"])
    codec = get_codec(meta["compressor"]) if meta.get("compressor") else None
    lo = np.array(start if start is not None else [0, 0, 0]); n = np.array(size if size is not None else shape); hi = lo + n
    out = np.zeros(n, dt)
    for k in s3_keys(prefix):
        rest = k[len(prefix):]
        if rest.startswith(".") or not rest:
            continue
        idx = np.array([int(x) for x in rest.split("/")]); c0 = idx * np.array(ch); c1 = np.minimum(c0 + ch, shape)
        a0, a1 = np.maximum(c0, lo), np.minimum(c1, hi)
        if (a1 <= a0).any():
            continue
        raw = get(B + k); buf = codec.decode(raw) if codec else raw
        a = np.frombuffer(buf, dt).reshape(ch)
        out[tuple(slice(x - l, y - l) for x, y, l in zip(a0, a1, lo))] = a[tuple(slice(x - c, y - c) for x, y, c in zip(a0, a1, c0))]
    return out


def read_points(path, vox_in):
    """ndjson points (voxels at vox_in A) -> array of (z, y, x) in A."""
    pts = []
    for line in get(B + path).decode().splitlines():
        if line.strip():
            l = json.loads(line)["location"]; pts.append((l["z"] * vox_in, l["y"] * vox_in, l["x"] * vox_in))
    return np.array(pts, float).reshape(-1, 3)


def read_mrc_gz(data):
    raw = gzip.decompress(data); h = np.frombuffer(raw[:1024], np.int32); f = np.frombuffer(raw[:1024], np.float32)
    nx, ny, nz, mode, nsym = int(h[0]), int(h[1]), int(h[2]), int(h[3]), int(h[23])
    assert mode == 2, f"MRC mode {mode}"
    px = float(f[10]) / nx
    vol = np.frombuffer(raw[1024 + nsym:1024 + nsym + nx * ny * nz * 4], np.float32).reshape(nz, ny, nx).copy()
    return vol, px


# ------------------------------------------------------------------------------------------------ volumes on the GPU
def dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_vox(vol, vox_in, dtype=None, order=1):
    """resample to VOX with an anti-alias Gaussian (FFT) then trilinear (order 1) or nearest (order 0)."""
    import torch
    import torch.nn.functional as F
    t = torch.tensor(vol, dtype=torch.float32, device=dev())
    if order == 1 and VOX > vox_in:
        s = 0.5 * VOX / vox_in                                     # sigma in input voxels
        Ff = torch.fft.rfftn(t); k = [torch.fft.fftfreq(n, device=t.device) for n in t.shape[:2]] + [torch.fft.rfftfreq(t.shape[2], device=t.device)]
        g = torch.exp(-2 * (math.pi * s) ** 2 * (k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2))
        t = torch.fft.irfftn(Ff * g, s=t.shape); del Ff, g
    size = [max(8, int(round(n * vox_in / VOX))) for n in vol.shape]
    t = (F.interpolate(t[None, None], size=size, mode="trilinear", align_corners=False) if order == 1 else F.interpolate(t[None, None], size=size, mode="nearest"))[0, 0]
    return t


def template(phase_random=False, rng=None):
    """EMD-54265 -> 25 A, low-pass 50 A, box 40 nm, spherical mask 15 nm; optional random Fourier phases (control)."""
    from scipy import ndimage
    cache = os.path.join(os.environ.get("RIBO_CACHE", "."), "emd_54265.map.gz")
    if not os.path.exists(cache):
        open(cache, "wb").write(get(EMD_URL))
    vol, px = read_mrc_gz(open(cache, "rb").read())
    vol = ndimage.gaussian_filter(vol, sigma=(LOWPASS / 2.355) / px)       # FWHM = low-pass length
    t = ndimage.zoom(vol, px / VOX, order=1)
    b = int(round(BOX_NM * 10 / VOX)); b += b % 2
    c = np.array(t.shape) // 2; t = t[c[0] - b // 2:c[0] + b // 2, c[1] - b // 2:c[1] + b // 2, c[2] - b // 2:c[2] + b // 2]
    if phase_random:
        F_ = np.fft.rfftn(t); ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F_.shape)); ph[0, 0, 0] = 1
        t = np.fft.irfftn(np.abs(F_) * ph, s=t.shape)
    z, y, x = np.mgrid[:b, :b, :b] - (b - 1) / 2; mask = (np.sqrt(z ** 2 + y ** 2 + x ** 2) * VOX <= MASK_NM * 10).astype(np.float32)
    return t.astype(np.float32), mask


def rotations(n, seed):
    from scipy.spatial.transform import Rotation
    return np.concatenate([np.eye(3)[None], Rotation.random(n, random_state=seed).as_matrix()])


def rotated_templates(t, mask, Rs, tilt_max):
    """rotate (about the box centre), apply the missing wedge (tilt axis y), mask, zero-mean, unit norm."""
    from scipy import ndimage
    b = t.shape[0]; c = (np.array(t.shape) - 1) / 2
    kz, ky, kx = np.meshgrid(np.fft.fftfreq(b), np.fft.fftfreq(b), np.fft.fftfreq(b), indexing="ij")
    wedge = (np.abs(kz) <= np.abs(kx) * math.tan(math.radians(tilt_max))) | ((kx == 0) & (kz == 0))
    out = np.zeros((len(Rs), b, b, b), np.float32); m = mask > 0
    for i, Rm in enumerate(Rs):
        r = ndimage.affine_transform(t, Rm, offset=c - Rm @ c, order=1)
        r = np.real(np.fft.ifftn(np.fft.fftn(r) * wedge))
        r = r - r[m].mean(); r = r * mask; r /= np.sqrt((r ** 2).sum()) + 1e-12
        out[i] = r
    return out


def match(V, T, mask, batch):
    """locally normalised cross-correlation of V with each template; max over templates -> score map (torch)."""
    import torch
    d = V.device; shp = V.shape; b = T.shape[-1]; n = float(mask.sum())
    def pad(x):
        P = torch.zeros(x.shape[:-3] + shp, device=d, dtype=torch.float32)
        P[..., :b, :b, :b] = x; return torch.roll(P, shifts=(-(b // 2),) * 3, dims=(-3, -2, -1))
    Vf = torch.fft.rfftn(V); Mf = torch.fft.rfftn(pad(torch.tensor(mask, device=d)))
    mu = torch.fft.irfftn(Vf * torch.conj(Mf), s=shp) / n
    m2 = torch.fft.irfftn(torch.fft.rfftn(V * V) * torch.conj(Mf), s=shp) / n
    sd = torch.sqrt(torch.clamp(m2 - mu * mu, min=1e-12)); del Mf, mu, m2
    best = torch.full(shp, -1e9, device=d)
    Tt = torch.tensor(T, device=d)
    for s in range(0, len(T), batch):
        Tf = torch.fft.rfftn(pad(Tt[s:s + batch]), dim=(-3, -2, -1))
        cc = torch.fft.irfftn(Vf[None] * torch.conj(Tf), s=shp, dim=(-3, -2, -1)); del Tf
        best = torch.maximum(best, cc.max(0).values); del cc
    return best / (sd * math.sqrt(n)), sd


def peaks(S, sd):
    """robust z of the score map; local maxima >= MINSEP apart, away from the border; returns (coords A, z)."""
    import torch
    import torch.nn.functional as F
    e = int(round(BORDER_NM * 10 / VOX)); inner = S[e:-e, e:-e, e:-e]; ok = sd[e:-e, e:-e, e:-e] > 1e-6
    v = inner[ok]
    if v.numel() > 2_000_000:
        v = v[torch.randint(0, v.numel(), (2_000_000,), device=v.device)]
    med = v.median(); mad = (v - med).abs().median() * 1.4826 + 1e-12
    Z = (S - med) / mad; k = 2 * int(round(MINSEP_NM * 10 / VOX)) + 1
    mx = F.max_pool3d(Z[None, None], k, stride=1, padding=k // 2)[0, 0]
    keep = (Z == mx) & (Z > ZGRID[0]); keep[:e] = False; keep[-e:] = False; keep[:, :e] = False; keep[:, -e:] = False; keep[:, :, :e] = False; keep[:, :, -e:] = False
    idx = torch.nonzero(keep); z = Z[keep]
    return idx.cpu().numpy() * VOX, z.cpu().numpy()


def in_labelled(pred, gt):
    """picks inside the box spanned by the hand-picked ribosomes, widened by the match distance (partial labelling)."""
    if len(gt) == 0:
        return pred[:0]
    lo, hi = gt.min(0) - MATCH_NM * 10, gt.max(0) + MATCH_NM * 10
    return pred[((pred >= lo) & (pred <= hi)).all(1)]


def f1(pred, gt):
    from scipy.optimize import linear_sum_assignment
    pred = in_labelled(pred, gt)
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0, 0
    d = np.sqrt(((pred[:, None, :] - gt[None, :, :]) ** 2).sum(-1)); cost = np.where(d <= MATCH_NM * 10, d, 1e9)
    r, c = linear_sum_assignment(cost); tp = int((cost[r, c] < 1e9).sum())
    p, rc = tp / len(pred), tp / len(gt); return (2 * p * rc / (p + rc) if tp else 0.0), p, rc, tp


# ------------------------------------------------------------------------------------------------ driver
def run(a):
    import torch
    os.makedirs(a.out, exist_ok=True); os.environ.setdefault("RIBO_CACHE", a.out)
    bun = json.load(open(a.bundle)); sha = code_sha(); t0 = time.time(); rng = np.random.default_rng(SEED)
    if a.trial:                                                   # one DEV and one EXAM hand-picked run, one run per line
        gtr = bun["ground_truth"]; bun["ground_truth"] = [next(e for e in gtr if e["split"] == "dev"), next(e for e in gtr if e["split"] == "exam")]
        seen = set(); bun["cells"] = [e for e in bun["cells"] if not (e["line"] in seen or seen.add(e["line"]))]
    log(f"ribosome test G14, code {sha}, device {dev()}{' (TRIAL: tiny crops, few rotations, only controls shown)' if a.trial else ''}")
    n_rot = 12 if a.trial else N_ROT
    tpl = {"real": template(), "control": template(True, np.random.default_rng(SEED + 1))}
    Rs = rotations(n_rot, SEED)
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    batch = 16 if gpu_gb > 30 else 8 if gpu_gb > 14 else 2
    found = {}                                                   # (run, kind, sign) -> (coords, z)

    def load(e, centre_A=None):
        if a.trial:
            shp = np.array(zarr_shape(e["tomo_prefix"])); h = np.minimum(shp // 2, [50, 120, 120])
            c = shp // 2 if centre_A is None else np.clip(np.round(np.array(centre_A) / e["vox"]).astype(int), h, shp - h)
            e["crop"] = (c - h).tolist(); e["crop_size"] = (2 * h).tolist(); vol = read_zarr_level(e["tomo_prefix"], e["crop"], e["crop_size"])
        else:
            vol = read_zarr_level(e["tomo_prefix"])
        V = to_vox(vol, e["vox"]); V = (V - V.mean()) / (V.std() + 1e-12); return V

    def find(e, V, kind, sign):
        T = rotated_templates(*tpl[kind], Rs, e["tilt_max"])
        S, sd = match(sign * V, T, tpl[kind][1], batch); return peaks(S, sd)

    # 1) hand-picked HSC-3: DEV picks the sign and the thresholds, EXAM scores once
    gt = {}
    for e in bun["ground_truth"]:
        g = read_points(e["gt_path"], e["gt_vox"]); V = load(e, np.median(g, 0) if len(g) else None)
        if a.trial:
            lo = np.array(e["crop"]) * e["vox"]; hi = lo + np.array(V.shape) * VOX; g = g[((g >= lo) & (g < hi)).all(1)] - lo
        gt[e["run"]] = g
        for kind in ("real", "control"):
            for sign in ((-1, 1) if e["split"] == "dev" else ()):
                found[(e["run"], kind, sign)] = find(e, V, kind, sign)
        e["_V"] = V if e["split"] == "exam" else None
        log(f"  {e['run']} ({e['split']}): searched")
    dev_runs = [e["run"] for e in bun["ground_truth"] if e["split"] == "dev"]

    def best_z(kind, sign):
        f = []
        for z in ZGRID:
            tp = npred = ngt = 0
            for r in dev_runs:
                c, s = found[(r, kind, sign)]; p = in_labelled(c[s > z], gt[r]); _, _, _, t = f1(p, gt[r]); tp += t; npred += len(p); ngt += len(gt[r])
            pr, rc = tp / max(npred, 1), tp / max(ngt, 1); f.append(2 * pr * rc / (pr + rc) if tp else 0.0)
        i = int(np.argmax(f)); return float(ZGRID[i]), float(f[i])
    choice = {}
    for kind in ("real", "control"):
        opts = {sign: best_z(kind, sign) for sign in (-1, 1)}; sign = max(opts, key=lambda s: opts[s][1])
        choice[kind] = dict(sign=sign, z=opts[sign][0], dev_f1=opts[sign][1], dev_f1_other_sign=opts[-sign][1])
    if not a.trial:
        log("DEV choices: " + json.dumps(choice))
    exam = {}
    for kind in ("real", "control"):
        tp = npred = ngt = 0
        for e in bun["ground_truth"]:
            if e["split"] != "exam":
                continue
            c, s = find(e, e["_V"], kind, choice[kind]["sign"]); found[(e["run"], kind, choice[kind]["sign"])] = (c, s)
            p = in_labelled(c[s > choice[kind]["z"]], gt[e["run"]]); _, _, _, t = f1(p, gt[e["run"]]); tp += t; npred += len(p); ngt += len(gt[e["run"]])
        pr, rc = tp / max(npred, 1), tp / max(ngt, 1); exam[kind] = dict(f1=(2 * pr * rc / (pr + rc) if tp else 0.0), precision=pr, recall=rc, picks=npred, hand_picked=ngt)
    portal = {}
    if not a.trial:
        tp = npred = ngt = 0
        for e in bun["ground_truth"]:
            if e["split"] == "exam" and e.get("portal_auto_path"):
                p = in_labelled(read_points(e["portal_auto_path"], e["gt_vox"]), gt[e["run"]]); _, _, _, t = f1(p, gt[e["run"]]); tp += t; npred += len(p); ngt += len(gt[e["run"]])
        pr, rc = tp / max(npred, 1), tp / max(ngt, 1); portal = dict(f1=(2 * pr * rc / (pr + rc) if tp else 0.0), precision=pr, recall=rc)
    r1_real = exam["real"]["f1"] >= 0.50; r1_ctl = exam["control"]["f1"] < 0.50
    res = {"G14-R1": dict(exam=exam["real"], control=exam["control"], dev=choice, reported_portal_automated_vs_hand=portal,
                          gate=dict(real=bool(r1_real), controls_ok=bool(r1_ctl), verdict=("PASS" if r1_real else "FAIL") if r1_ctl else "VOID (controls)"))}
    log(f"G14-R1 scored ({(time.time() - t0) / 60:.1f} min)")
    # 2) cells we model (and HeLa): ribosomes found with the DEV threshold; membrane-near share
    lines = {}
    from scipy import ndimage
    for e in bun["cells"]:
        V = load(e); out = {}
        for kind in ("real", "control"):
            c, s = find(e, V, kind, choice[kind]["sign"]); out[kind] = c[s > choice[kind]["z"]]
        near = None
        if e.get("membrane_prefix") and len(out["real"]):
            m = read_zarr_level(e["membrane_prefix"], e.get("crop"), e.get("crop_size")) if a.trial else read_zarr_level(e["membrane_prefix"])
            M = (to_vox((m > 0).astype(np.float32), e["vox"], order=0) > 0.5).cpu().numpy()
            dist = ndimage.distance_transform_edt(~M) * VOX
            ijk = np.clip(np.round(out["real"] / VOX).astype(int), 0, np.array(M.shape) - 1)
            near = int((dist[ijk[:, 0], ijk[:, 1], ijk[:, 2]] <= MEM_NM * 10).sum())
        vol_um3 = float(np.prod(V.shape) * (VOX / 1e4) ** 3)
        L = lines.setdefault(e["line"], dict(tomograms=0, found=0, control_found=0, membrane_near=0, with_mask=0, volume_um3=0.0, per_tomogram=[]))
        L["tomograms"] += 1; L["found"] += len(out["real"]); L["control_found"] += len(out["control"]); L["volume_um3"] += vol_um3
        if near is not None:
            L["membrane_near"] += near; L["with_mask"] += len(out["real"])
        L["per_tomogram"].append(dict(run=e["run"], found=len(out["real"]), control=len(out["control"]), membrane_near=near, volume_um3=round(vol_um3, 4)))
        log(f"  {e['line']} {e['run']}: searched" if a.trial else f"  {e['line']} {e['run']}: {len(out['real'])} ribosomes (control template {len(out['control'])})")
    for k, L in lines.items():
        L["per_um3"] = L["found"] / max(L["volume_um3"], 1e-9); L["membrane_near_share"] = L["membrane_near"] / max(L["with_mask"], 1)
        L["control_ratio"] = L["control_found"] / max(L["found"], 1); L["predicted_er_share"] = bun["prediction"].get(k)
    J = lines.get("Jurkat", {}); pred = bun["prediction"]["Jurkat"]
    r2_real = bool(J and abs(J["membrane_near_share"] - pred) <= 0.10); r2_ctl = bool(J and J["control_ratio"] <= 1 / 3 and r1_real and r1_ctl)
    res["G14-R2"] = dict(jurkat=J, predicted=pred, difference=(J["membrane_near_share"] - pred) if J else None,
                         gate=dict(real=r2_real, controls_ok=r2_ctl, verdict=("PASS" if r2_real else "FAIL") if r2_ctl else "VOID (controls)"))
    res["reported_lines"] = lines
    res["meta"] = dict(code_sha=sha, trial=a.trial, device=dev(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                       rotations=int(len(Rs)), minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if a.trial:
        ctl = {k: {x: res[k]["gate"][x] for x in ("controls_ok",)} for k in ("G14-R1", "G14-R2")}
        json.dump(dict(meta=res["meta"], controls=ctl), open(os.path.join(a.out, "G14_trial_controls.json"), "w"), indent=1)
        log("trial ran; outcomes not shown; controls:", json.dumps(ctl)); return
    json.dump(res, open(os.path.join(a.out, "G14_results.json"), "w"), indent=1, default=float)
    r1, r2 = res["G14-R1"], res["G14-R2"]
    print("\n" + "=" * 100)
    print(f"G14 RESULTS (code {sha}; {res['meta']['gpu']}; {res['meta']['minutes']} min)")
    print(f" R1 the finder finds ribosomes: {r1['gate']['verdict']}. Held-out hand-picked tomograms: F1 {r1['exam']['f1']:.2f} "
          f"(precision {r1['exam']['precision']:.2f}, recall {r1['exam']['recall']:.2f}); scrambled template F1 {r1['control']['f1']:.2f}.")
    print(f" R2 Jurkat membrane-bound ribosomes vs RNA prediction: {r2['gate']['verdict']}. Measured {J['membrane_near_share']:.2f}, predicted {pred:.2f}.")
    for k, L in lines.items():
        print(f"   {k}: {L['found']} ribosomes in {L['tomograms']} tomograms ({L['per_um3']:.0f} per cubic micrometre), {L['membrane_near_share']:.0%} near a membrane; scrambled template {L['control_found']}")
    print("=" * 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", default=os.path.join(HERE, "bundle.json")); ap.add_argument("--out", default="ribo_out")
    ap.add_argument("--trial", action="store_true"); run(ap.parse_args())
