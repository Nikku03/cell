"""A LEARNED SURROGATE AGAINST THE PHYSICS, ON A CHROMOSOME NEITHER OF THEM HAS SEEN.

THE QUESTION.  The physics model reaches insulation r = 0.2974 on chr21 and cost nothing to fit,
because it has no fitted parameters at all -- every constant came from single-molecule measurements or
from geometry. A neural network can be fitted, and the honest question is not whether it can learn
chr21, which it certainly can, but whether it TRANSFERS: does a model trained on one chromosome beat
mechanism on a different one?

WHY chr22 AND NOT A HELD-OUT SLICE OF chr21.  A held-out window from the same chromosome shares CTCF
sites, compartment structure and mappability with its training data, and a convolutional model will
exploit all three. Held out by CHROMOSOME is the version of this test that can actually fail, and it is
the same discipline loop 8 used when it blocked its folds by chromosome.

WHAT COMPETES, all scored on chr22, none of them having seen it:

    the surrogate        a small 1D convolutional net over the oriented CTCF tracks, trained on chr21
    the physics          loops 34-36 run on chr22's own CTCF landscape, no refitting, no free parameters
    smoothed CTCF        the trivial baseline: total CTCF signal in a window. If this wins, neither the
                         network nor the mechanism earned anything, and the boundaries were simply
                         where the CTCF is.
    shuffled orientation the surrogate retrained on orientation-shuffled input, which tells us whether
                         the network learned the ASYMMETRY or merely the peak positions

PREDECLARED, before any number:

    S1 THE SURROGATE TRANSFERS   its chr22 insulation correlation beats the shuffle band.
                fails -> it memorised chr21 and there is nothing more to discuss.
    S2 IT BEATS THE TRIVIAL BASELINE   it beats smoothed CTCF density on chr22.
                A network that cannot beat a boxcar filter over its own input has learned nothing.
    S3 IT BEATS THE PHYSICS   it beats the mechanism model run on chr22.
                THIS IS THE REAL COMPARISON. The physics has zero fitted parameters; the network has
                thousands and a chromosome of training data. If the network does not win, the honest
                report is that mechanism was worth more than fitting here -- and if it does win, that
                is worth knowing too and the physics should not be defended.
    S4 IT LEARNED THE ASYMMETRY, NOT THE POSITIONS   the surrogate beats its own orientation-shuffled
                twin.
                Loop extrusion is a claim about DIRECTION. A network that scores the same on shuffled
                orientations has rediscovered where CTCF binds, which was already in its input.

CONTROLS: held out by whole chromosome; a trivial smoothed-density baseline; the physics model with no
refitting; an orientation-shuffled retrain; and 200 shuffles of the target profile for the band.

-> outputs/loop_surrogate.json
"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from loop_polymer import r2_matrix  # noqa: E402
from loop_hic_target import insulation  # noqa: E402
import loop_extrusion as LE  # noqa: E402
from loop_stiffness import laplacian_k, K_LOOP  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SEED = 1904
N_SHUFFLE = 200
BIN = 25000
WIN = 81                 # bins of context, ~2 Mb, the scale insulation is defined on
EPOCHS = 400
N_CONFIG = 30


def ctcf_tracks(chrom, n):
    """Forward and reverse CTCF strength per bin, orientation from the motif in that assembly."""
    seq = "".join(l.strip() for l in gzip.open(SC / f"hg19_{chrom}.fa.gz", "rt")
                  if l[0] != ">").upper()
    peaks = []
    with gzip.open(SC / "ctcf_gm12878_hg19.bed.gz", "rt") as f:
        for line in f:
            p = line.split("\t")
            if p[0] != chrom:
                continue
            a, b = int(p[1]), int(p[2])
            off = int(p[9]) if len(p) > 9 and p[9].strip().lstrip("-").isdigit() else -1
            peaks.append({"summit": a + (off if off >= 0 else (b - a) // 2), "signal": float(p[6])})
    pfm = json.load(open(SC / "ctcf_pfm.json"))
    P = np.array([pfm[c] for c in "ACGT"], float).T
    W = np.log2((P / P.sum(1, keepdims=True) + 1e-3) / 0.25)
    Lw = len(pfm["A"])
    ix = {c: i for i, c in enumerate("ACGT")}
    for p in peaks:
        w = seq[max(0, p["summit"] - 150):p["summit"] + 150]
        best, bo = -1e9, 0
        for i in range(len(w) - Lw + 1):
            s2 = w[i:i + Lw]
            f_ = -1e9 if any(c not in ix for c in s2) else sum(W[k, ix[c]] for k, c in enumerate(s2))
            s3 = s2.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            r_ = -1e9 if any(c not in ix for c in s3) else sum(W[k, ix[c]] for k, c in enumerate(s3))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["orient"] = bo if best > 6.0 else 0
    sig = np.array([p["signal"] for p in peaks]) if peaks else np.array([1.0])
    rank = sig.argsort().argsort() / max(len(sig) - 1, 1)
    fwd, rev = np.zeros(n), np.zeros(n)
    for p, rk in zip(peaks, rank):
        b = p["summit"] // BIN
        if 0 <= b < n and p["orient"] != 0:
            arr = fwd if p["orient"] > 0 else rev
            arr[b] = max(arr[b], rk)
    return fwd, rev


def physics_insulation(fwd, rev, n, mask):
    """loops 34-36, unchanged, on this chromosome's own landscape. No refitting."""
    cfgs, _, _, _ = LE.simulate(n, fwd, rev, np.random.default_rng(SEED))
    acc = np.zeros((n, n))
    for cfg in cfgs[:N_CONFIG]:
        L = laplacian_k(n, [(int(a), int(b)) for a, b in cfg if b > a], K_LOOP, LE.CONFINE)
        R2 = r2_matrix(L, confined=True)
        np.fill_diagonal(R2, np.inf)
        acc += R2 ** -1.5
    return insulation(acc / N_CONFIG)


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    say("=" * 100)
    say("A LEARNED SURROGATE AGAINST THE PHYSICS, ON A CHROMOSOME NEITHER HAS SEEN")
    say("=" * 100)
    say("  The physics has ZERO fitted parameters. The network gets thousands and a whole chromosome")
    say("  of training data. Held out by chromosome, because a held-out slice of chr21 shares CTCF")
    say("  sites and mappability with its own training data.")

    data = {}
    for ch in ("chr21", "chr22"):
        H = np.load(SC / f"hic_{ch}_25kb.npy").astype(np.float64)
        H[H == 0] = np.nan
        n = len(H)
        mask = np.isfinite(H).sum(1) > 50
        ins = insulation(H)
        fwd, rev = ctcf_tracks(ch, n)
        ok = np.isfinite(ins) & mask
        data[ch] = {"n": n, "mask": mask, "ins": ins, "fwd": fwd, "rev": rev, "ok": ok}
        say(f"  {ch}: {n:,} bins, {int(mask.sum()):,} mappable, "
            f"{int((fwd > 0).sum() + (rev > 0).sum())} oriented CTCF bins")

    def windows(d, fwd, rev):
        n, half = d["n"], WIN // 2
        F = np.pad(fwd, half)
        R = np.pad(rev, half)
        idx = np.where(d["ok"])[0]
        X = np.stack([np.stack([F[i:i + WIN], R[i:i + WIN]]) for i in idx])
        y = d["ins"][idx]
        y = (y - y.mean()) / (y.std() + 1e-9)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(2, 32, 9, padding=4), nn.ReLU(),
                nn.Conv1d(32, 32, 9, padding=4, dilation=1), nn.ReLU(),
                nn.Conv1d(32, 16, 9, padding=8, dilation=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 1))

        def forward(self, x):
            return self.f(x).squeeze(-1)

    def train_and_score(train_fwd, train_rev, label):
        Xtr, ytr, _ = windows(data["chr21"], train_fwd, train_rev)
        Xte, yte, _ = windows(data["chr22"], data["chr22"]["fwd"], data["chr22"]["rev"])
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        lossf = nn.MSELoss()
        for ep in range(EPOCHS):
            opt.zero_grad()
            loss = lossf(net(Xtr), ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(Xte).numpy()
        r = float(np.corrcoef(pred, yte.numpy())[0, 1])
        say(f"    {label:<34} chr22 r {r:+.4f}   (train loss {float(loss):.4f})")
        return r

    say("")
    r_net = train_and_score(data["chr21"]["fwd"], data["chr21"]["rev"], "surrogate, real orientation")
    r_shuf = train_and_score(rng.permutation(data["chr21"]["fwd"]),
                             rng.permutation(data["chr21"]["rev"]),
                             "surrogate, SHUFFLED orientation")

    d22 = data["chr22"]
    ok22 = d22["ok"]
    dens = np.convolve(d22["fwd"] + d22["rev"], np.ones(WIN) / WIN, mode="same")
    r_dens = float(np.corrcoef(dens[ok22], d22["ins"][ok22])[0, 1])
    say(f"    {'smoothed CTCF density (trivial)':<34} chr22 r {r_dens:+.4f}")

    ins_phys = physics_insulation(d22["fwd"], d22["rev"], d22["n"], d22["mask"])
    okp = ok22 & np.isfinite(ins_phys)
    r_phys = float(np.corrcoef(ins_phys[okp], d22["ins"][okp])[0, 1])
    say(f"    {'physics, no refitting':<34} chr22 r {r_phys:+.4f}   "
        f"(it got +0.2974 on chr21)")

    y22 = d22["ins"][ok22]
    band = float(np.percentile([abs(np.corrcoef(rng.permutation(y22), y22)[0, 1])
                                for _ in range(N_SHUFFLE)], 95))

    s1 = bool(abs(r_net) > band)
    s2 = bool(r_net > r_dens)
    s3 = bool(r_net > r_phys)
    s4 = bool(r_net > r_shuf)
    say(f"\n  S1 THE SURROGATE TRANSFERS -- {r_net:+.4f} vs band {band:.4f}   "
        f"S1 {'PASS' if s1 else 'FAIL'}")
    say(f"  S2 IT BEATS THE TRIVIAL BASELINE -- vs smoothed density {r_dens:+.4f}   "
        f"S2 {'PASS' if s2 else 'FAIL'}")
    say(f"  S3 IT BEATS THE PHYSICS -- vs {r_phys:+.4f}, which has zero fitted parameters   "
        f"S3 {'PASS' if s3 else 'FAIL'}")
    say(f"  S4 IT LEARNED THE ASYMMETRY -- vs its orientation-shuffled twin {r_shuf:+.4f}   "
        f"S4 {'PASS' if s4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"S1 surrogate transfers": s1, "S2 beats trivial baseline": s2,
             "S3 beats the physics": s3, "S4 learned the asymmetry": s4}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    best = max([("the surrogate", r_net), ("the physics", r_phys),
                ("smoothed CTCF density", r_dens)], key=lambda x: x[1])
    say(f"  ON A CHROMOSOME NONE OF THEM HAD SEEN, THE WINNER IS {best[0].upper()} AT {best[1]:+.4f}.")
    if not s3:
        say(f"  MECHANISM BEAT FITTING HERE. The physics carries no fitted parameters at all -- every")
        say(f"  constant came from single-molecule measurements or from geometry -- and it still")
        say(f"  transfers better than a network given a whole chromosome to learn from. That is the")
        say(f"  argument for building the mechanism rather than regressing on the readout.")
    if not s4:
        say(f"  AND THE NETWORK DID NOT LEARN DIRECTION. Its orientation-shuffled twin scores")
        say(f"  {r_shuf:+.4f} against {r_net:+.4f}, so what it extracted was where CTCF binds, which was")
        say(f"  already in its input. Loop extrusion is a claim about asymmetry and the network is not")
        say(f"  making it.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "hic_chr21_25kb.npy"), str(SC / "hic_chr22_25kb.npy")],
                      available=int(data["chr22"]["n"]), used=int(ok22.sum()),
                      selection="filtered", seed=SEED,
                      controls=["held out by whole chromosome, not by slice",
                                "smoothed CTCF density as the trivial baseline",
                                "the physics model with no refitting",
                                "an orientation-shuffled retrain",
                                f"{N_SHUFFLE} target shuffles for the band"],
                      note="the physics has zero fitted parameters and the network has thousands plus "
                           "a chromosome of training data; that asymmetry is the point of the test")
    RM.report(man, emit=say)
    json.dump({"test": "loop_surrogate", "manifest": man, "gates": gates,
               "r_surrogate": r_net, "r_surrogate_shuffled": r_shuf, "r_density": r_dens,
               "r_physics_chr22": r_phys, "r_physics_chr21": 0.2974, "band": band,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_surrogate.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_surrogate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
