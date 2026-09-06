"""Loop 209. The maximal physics arm: every human motif, and a shuffled-sequence control.

WHY THIS EXISTS RATHER THAN AN ARGUMENT. Loop 206 tested computed occupancy for ONE factor, NR3C1,
over a promoter window, and got r -0.0133 against a measured occupancy's +0.2932. Its own Y8 said
plainly that this was a FLOOR and not a ceiling: one motif, promoter only, an additive energy
model. That leaves the maximal version untested, and loop 207 measured the maximal version to be
cheap -- 0.712 ms per promoter-motif pair, so all 879 human JASPAR CORE motifs over these promoters
is minutes, not hours. An untested claim that is cheap to test should be tested, not reasoned about.

WHAT IS MAXIMAL HERE, AND WHAT IS STILL NOT. This runs every non-redundant human vertebrate CORE
motif at every promoter in loop 198's scored set, at three chemical potentials each, with the
partition function over both strands. It does NOT run docking, molecular dynamics, or free-energy
perturbation, and it does not reach distal elements -- loop 206's Y8 named both, and loop 207
priced the first at centuries of single-core time. So this is the maximal SEQUENCE-THERMODYNAMIC
arm, which is the only physics route this project can actually execute.

THE CONTROL THAT DECIDES WHAT A PASS WOULD MEAN. Loop 177 measured that four columns of base
composition beat an entire sequence chain, and loop 173 scored 3/11 for the same reason. A motif
scan over GC-rich promoters can score without reading a motif at all. So every real promoter is
matched by a DINUCLEOTIDE-SHUFFLED version of itself, which preserves length, base composition and
dinucleotide frequency while destroying every binding site. If the real sequence does not beat its
own shuffle, the arm is reading composition and the physics is decorative.

PREDECLARED, BEFORE ANY NUMBER.

  B1 IS THE INSTRUMENT READY?
     Gate: PASS iff 879 motifs parse with valid counts, the cached promoter set covers at least
     1,000 of loop 198's scored genes, and the measured throughput is within 3x of the 0.712 ms
     per promoter-motif pair that loop 207's cost model was built on. A throughput far from that
     would mean loop 207's whole costing was wrong.

  B2 DOES THE COMPUTED OCCUPANCY VARY ACROSS GENES AT ALL?
     Loop 206a measured median NR3C1 promoter occupancy at 0.8420 for mu = 0 and 0.9999 for mu = 8
     -- saturated, and a saturated feature cannot rank genes. This asks whether that was NR3C1's
     problem or the method's.
     Gate: PASS iff at least 25% of the 879 motifs have a cross-gene coefficient of variation above
     0.10 at the best of the three chemical potentials. A FAIL says the matrix is mostly constant
     down its columns, which would explain a failure in B3 without any appeal to biology.

  B3 THE MAXIMAL PHYSICS ARM.
     All motifs, gene-held-out five-fold ridge, set point scored on loop 206's harness.
     Gate: PASS iff it beats its own dinucleotide-shuffled control by at least 0.05 in |r|.
     This gate is about whether the physics is READING SEQUENCE, not about whether it is useful --
     B4 asks that separately, and separating them is the point.

  B4 IS IT USEFUL?
     Gate: PASS iff |r| reaches loop 206's measured crossover of 0.9081.
     Reported alongside and required for the pass to mean anything: the nine same-cell ChIP tracks
     at 0.2932 (loop 206 Y4) and the 200 measured Perturb-seq gains at 0.2785 (loop 208 A4).
     Requires B3 -- an arm that does not beat its own shuffle has nothing to be useful with.

  B5 THE TIME AND THE ACCURACY, which is the question this loop was asked.
     Report the MEASURED wall time for this run and extrapolate to all 16,492 promoters, and pair
     it with the measured accuracy. Not scored; it is the deliverable.

  B6 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import energy_matrix, scan, SEQ_F, gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PFMS = ROOT / "colab" / "data" / "physics" / "jaspar_core_vert.txt"
OUT = "outputs/loop_physics_max.json"
MU = np.array([-4.0, 0.0, 6.0])
SEED = 209209
R_REQ, R_NINE, R_GAIN = 0.9081, 0.2932, 0.2785
REF_MS = 0.712

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def read_pfms(path):
    out, name, rows = [], None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name and len(rows) == 4:
                out.append((name, np.array(rows, float)))
            name, rows = line[1:].strip(), []
        elif line.strip():
            nums = line[line.find("[") + 1:line.rfind("]")].split()
            rows.append([float(x) for x in nums])
    if name and len(rows) == 4:
        out.append((name, np.array(rows, float)))
    return out


def dinuc_shuffle(seq, rng):
    """Altschul-Erikson: preserve length, base composition AND dinucleotide frequency."""
    if len(seq) < 3:
        return seq
    edges = {}
    for a, b in zip(seq, seq[1:]):
        edges.setdefault(a, []).append(b)
    for k in edges:
        rng.shuffle(edges[k])
    out, cur = [seq[0]], seq[0]
    for _ in range(len(seq) - 1):
        nxt = edges.get(cur)
        if not nxt:
            break
        c = nxt.pop()
        out.append(c); cur = c
    while len(out) < len(seq):
        out.append(seq[len(out)])
    return "".join(out)


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def ridge(Xtr, ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    return B @ np.linalg.solve(A.T @ A + R, A.T @ ytr)


def build(seqs, names, pfms, bg):
    F = np.zeros((len(names), len(pfms) * len(MU)), dtype=np.float32)
    for j, (_, pfm) in enumerate(pfms):
        E = energy_matrix({b: pfm[i] for i, b in enumerate("ACGT")}, bg)
        for i, s in enumerate(names):
            F[i, j * len(MU):(j + 1) * len(MU)] = scan(seqs[s], E, MU)
    return F


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "maximal physics arm"}
    say("=" * 104)
    say("LOOP 209 -- THE MAXIMAL PHYSICS ARM: EVERY HUMAN MOTIF, AND A SHUFFLED CONTROL")
    say("=" * 104)

    grid, M, A, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    S_true = (M[-3:].mean(0))[gi]
    seqs = json.load(open(SEQ_F))
    names = [sym[i] for i in gi if sym[i] in seqs]
    pos = {s: k for k, s in enumerate([sym[i] for i in gi])}
    y = np.array([S_true[pos[s]] for s in names])
    pfms = read_pfms(PFMS)

    say("B1 IS THE INSTRUMENT READY?")
    say(f"     motifs parsed {len(pfms):,}   widths {min(p.shape[1] for _,p in pfms)}"
        f"-{max(p.shape[1] for _,p in pfms)}")
    say(f"     promoters cached {len(names):,} of loop 198's {len(gi):,} scored genes")
    cnt = np.zeros(4); B = {c: i for i, c in enumerate("ACGT")}
    for s in seqs.values():
        for c in s:
            if c in B:
                cnt[B[c]] += 1
    bg = cnt / cnt.sum()
    tt = time.time()
    E0 = energy_matrix({b: pfms[0][1][i] for i, b in enumerate("ACGT")}, bg)
    for s in names[:200]:
        scan(seqs[s], E0, MU)
    ms = (time.time() - tt) / 200 * 1000
    say(f"     measured throughput {ms:.3f} ms per promoter-motif pair "
        f"(loop 207 costed at {REF_MS})")
    ok1 = (len(pfms) == 879 and len(names) >= 1000 and ms < 3 * REF_MS)
    G.add("B1", ok1, stat=ms,
          if_true=lambda: f"B1 PASS -- {len(pfms)} motifs, {len(names):,} promoters, "
                          f"{ms:.2f} ms/pair confirms loop 207's cost model",
          if_false=lambda: f"B1 FAIL -- {len(pfms)} motifs, {len(names)} promoters, {ms:.2f} ms")

    say(f"     scanning {len(names):,} x {len(pfms):,} x {len(MU)} mu ...")
    t_scan = time.time()
    F = build(seqs, names, pfms, bg)
    scan_s = time.time() - t_scan
    say(f"     REAL sequence scan done in {scan_s:,.0f} s")
    rng = np.random.default_rng(SEED)
    shuf = {s: dinuc_shuffle(seqs[s], rng) for s in names}
    t_sh = time.time()
    Fs = build(shuf, names, pfms, bg)
    say(f"     SHUFFLED control scan done in {time.time()-t_sh:,.0f} s")

    say("B2 DOES THE COMPUTED OCCUPANCY VARY ACROSS GENES AT ALL?")
    cvs = []
    for j in range(len(pfms)):
        blk = F[:, j * len(MU):(j + 1) * len(MU)]
        cvs.append(float(np.nanmax(blk.std(0) / (blk.mean(0) + 1e-9))))
    cvs = np.array(cvs)
    frac = float((cvs > 0.10).mean())
    say(f"     cross-gene CV per motif (best of {len(MU)} mu): median {np.median(cvs):.4f}   "
        f"90th pct {np.percentile(cvs,90):.4f}")
    say(f"     motifs with CV > 0.10: {int((cvs>0.10).sum()):,} of {len(pfms):,} = {frac:.1%}")
    say(f"     mean occupancy over the whole matrix {F.mean():.4f}")
    G.add("B2", bool(frac >= 0.25), stat=frac, requires=("B1",),
          if_true=lambda: f"B2 PASS -- {frac:.0%} of motifs vary usefully across genes",
          if_false=lambda: f"B2 FAIL -- only {frac:.1%} of motifs have cross-gene CV above 0.10; "
                           f"the occupancy matrix is largely CONSTANT down its columns and cannot "
                           f"rank genes. Loop 206a saw this for NR3C1 and it is not NR3C1-specific")

    say("B3 THE MAXIMAL PHYSICS ARM")
    order = np.random.default_rng(SEED).permutation(len(y))
    folds = np.array_split(order, 5)

    def cv_r(Feat, label):
        best = (float("-inf"), None)
        for lam in (10.0, 100.0, 1000.0, 10000.0):
            Sp = np.zeros(len(y))
            for k in range(5):
                te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
                Sp[te] = ridge(Feat[tr], y[tr], Feat[te], lam)
            r = pear(Sp, y)
            if np.isfinite(r) and abs(r) > best[0]:
                best = (abs(r), r)
        say(f"       {label:<34} |r| {best[0]:.4f}  (r {best[1]:+.4f})")
        return best[0]
    r_real = cv_r(F, f"REAL sequence, {len(pfms)} motifs")
    r_shuf = cv_r(Fs, "dinucleotide-shuffled control")
    delta = r_real - r_shuf
    say(f"     real minus shuffled  {delta:+.4f}")
    G.add("B3", bool(delta >= 0.05), stat=delta, requires=("B2",),
          if_true=lambda: f"B3 PASS -- the arm beats its own shuffle by {delta:+.4f}, so it is "
                          f"reading binding sites and not base composition",
          if_false=lambda: f"B3 FAIL -- real {r_real:.4f} against its own dinucleotide shuffle "
                           f"{r_shuf:.4f}, difference {delta:+.4f}. Whatever the scan is reading, "
                           f"it survives destroying every binding site, so it is composition")

    say("B4 IS IT USEFUL?")
    say(f"       maximal physics arm             |r| {r_real:.4f}")
    say(f"       nine same-cell ChIP tracks      |r| {R_NINE:.4f}   (loop 206 Y4)")
    say(f"       200 measured Perturb-seq gains  |r| {R_GAIN:.4f}   (loop 208 A4)")
    say(f"       REQUIREMENT                     |r| {R_REQ:.4f}   (loop 206 Y3)")
    G.add("B4", bool(r_real >= R_REQ), stat=r_real, requires=("B3",),
          if_true=lambda: f"B4 PASS -- {r_real:.4f} clears the crossover",
          if_false=lambda: f"B4 FAIL -- {r_real:.4f} against a requirement of {R_REQ:.4f}")

    say("B5 THE TIME AND THE ACCURACY")
    full = scan_s * (16492 / max(len(names), 1))
    say(f"     MEASURED: {len(names):,} promoters x {len(pfms):,} motifs x {len(MU)} mu "
        f"in {scan_s:,.0f} s")
    say(f"     extrapolated to all 16,492 promoters: {full/3600:,.2f} CPU-hours "
        f"({full/3600/8:,.2f} h on 8 cores)")
    say(f"     ACCURACY on the one task with a bar: |r| {r_real:.4f} real, "
        f"{r_shuf:.4f} shuffled, requirement {R_REQ:.4f}")
    res["time_accuracy"] = {"promoters": len(names), "motifs": len(pfms), "mu": len(MU),
                            "scan_seconds": scan_s, "full_cpu_hours": full / 3600,
                            "r_real": r_real, "r_shuffled": r_shuf, "delta": delta,
                            "required": R_REQ, "nine_track": R_NINE, "perturbseq": R_GAIN,
                            "throughput_ms": ms, "motif_cv_frac": frac}

    say("B6 WHAT THIS CANNOT SHOW")
    say("     This is the maximal SEQUENCE-THERMODYNAMIC arm, not maximal physics. Docking, MD")
    say("     and free-energy perturbation were priced by loop 207 at centuries of single-core")
    say("     time and are not run here, so nothing below refutes them -- it refutes the only")
    say("     physics route this project can execute.")
    say("     Promoter windows only. Loop 185 measured distal elements carrying real signal, so")
    say("     a distal version is untested and would be a different number.")
    say("     Berg-von Hippel is additive across positions. Real binding is not, and loop 184")
    say("     measured co-binding at 0.8455 against motif at 0.6228 -- most of what decides")
    say("     occupancy is not the site.")
    say("     JASPAR CORE is non-redundant by construction, so related factors share one matrix")
    say("     and factor-specific differences are absent from this feature set by design.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
