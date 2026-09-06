"""THE TARGET: what "acts like DNA in the nucleus" has to mean, measured on a real Hi-C map.

WHY THIS COMES FIRST, AND WHY IT IS NOT A MODEL.  The instruction is to build a 4D chromatin model that
behaves like DNA in a nucleus. That phrase cannot be tested as written -- like a nucleus in which
respects, to what tolerance? So this loop builds no model at all. It fetches a real, deep Hi-C map,
measures the things a chromosome demonstrably does, and writes down the numeric windows a simulation
will have to land in. Predeclaring the target before the model exists is the only way the model can
later fail.

WHAT THIS REPOSITORY ALREADY SPENT ON CHROMATIN, so none of it is rebuilt by accident.  There is a whole
prior arc here and it ended in a negative that has to be read carefully:

    contact_gate   CTCF-between-element-and-promoter features moved a CRISPR enhancer-gene classifier
                   from AUPRC 0.6076 to 0.6632, beating its own shuffled control at 0.6048.  GO.
    rouse_gate     an analytic Rouse-with-loops model beat its shuffled anchors (0.6107 vs 0.5976) and
                   added +0.005 on top of the CTCF count.  PARTIAL -- "the physics is redundant".
    hic_gate       MEASURED Hi-C loops and domains added NOTHING on top of the CTCF count: net -0.0031
                   (z = -1.06) intact, -0.0012 (z = -0.55) in situ.
    extrude        KMC extrusion plus batched Langevin, built to see whether a real simulation beats the
                   approximation.

READ THAT LAST RESULT PROPERLY.  Real measured Hi-C failed to improve that classifier. So the arc's
negative is not "chromatin structure is unmodellable" -- it is "this particular enhancer-gene classifier
does not need 3D, because counting CTCF sites between two loci already captures what it can use." A
model of the nucleus is a different target with different observables, and it is scored here against
chromatin itself rather than against a downstream classifier that has already shown it cannot tell the
difference.

THE DATA.  Rao 2014 GM12878 in situ primary, the deepest published human Hi-C, read REMOTELY by range
request so that a 20 GB file costs one chromosome's worth of transfer. chr21 at 25 kb is 1,926 bins of
which 1,383 are mappable -- small enough to simulate as a polymer on four CPUs, which is the whole
reason for choosing it. CTCF is GM12878's own ChIP-seq, cell-type matched to the Hi-C rather than
borrowed from K562, and hg19 to match the map.

WHAT IS MEASURED, and each one is a thing a chromosome DOES:

    P(s)            contact probability against genomic separation. The single most robust property of
                    interphase chromatin; a fractal globule gives a slope near -1, an equilibrium
                    globule -1.5, and a model that misses this is not a chromosome at any scale.
    INSULATION      TAD boundaries, and whether CTCF sits on them. This is the loop-extrusion signature
                    in its weakest form.
    COMPARTMENTS    the A/B checkerboard, anchored to GC content, which is computed from the same hg19
                    sequence so no cross-assembly join can fake it.
    CORNER PEAKS    focal enrichment between CONVERGENT CTCF pairs specifically. This is the strong
                    signature: extrusion predicts that orientation matters, and nothing about proximity
                    or accessibility does. Motif direction comes from scanning JASPAR MA0139.1 in the
                    chr21 sequence.

PREDECLARED, before any number:

    D1 THE MAP IS REAL          at least 1,000 mappable bins, and the P(s) slope over 0.1-10 Mb lands in
                [-1.6, -0.7], the published range for interphase human chromatin.
                fails -> the extraction is broken and nothing below it means anything.
    D2 TADS EXIST AND CTCF MARKS THEM   insulation boundaries carry more CTCF than 200 shuffled-CTCF
                controls, where shuffling preserves peak count within the chromosome.
                fails -> either the boundary caller or the CTCF join is broken, and no extrusion model
                could be validated against this map.
    D3 COMPARTMENTS EXIST       the first eigenvector of the observed/expected correlation matrix is a
                checkerboard, and it tracks GC content beyond shuffles, with A being GC-rich.
    D4 ORIENTATION MATTERS      convergent CTCF pairs contact more than divergent and tandem pairs at
                MATCHED genomic separation.
                THIS IS THE GATE THAT MAKES THE WHOLE PROJECT WORTH DOING. Distance-matching is
                essential: convergent pairs are not specially close, so if orientation still predicts
                contact after separation is held fixed, that is loop extrusion and nothing else
                explains it. If it fails, this map cannot distinguish an extrusion model from a
                generic polymer, and the model must be scored on D1-D3 only -- which would have to be
                said out loud rather than quietly.

WHAT THIS MAP CANNOT DO, stated now rather than when it becomes inconvenient.  Hi-C is a fixed
population average. It has no time axis at all. So the FOURTH dimension cannot be validated here, and
any model claiming dynamics must be scored against live-imaging measurements from the literature --
sub-diffusive MSD with exponent about 0.4-0.5, a confinement plateau, and cohesin residence times of
order 10-25 minutes. Those are LITERATURE values, not measured in this container, and every use of them
is labelled as such. A project that scored its own dynamics against its own assumptions would be
grading its own homework.

CONTROLS: 200 within-chromosome CTCF shuffles preserving count; distance-matched comparison for every
orientation claim, since P(s) dominates every raw contact number; GC content from the same assembly as
the map; and the unmappable centromeric bins excluded up front rather than diluted into the averages.

-> outputs/loop_hic_target.json
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
HIC = SC / "hic_chr21_25kb.npy"
CTCF = SC / "ctcf_gm12878_hg19.bed.gz"
FASTA = SC / "hg19_chr21.fa.gz"
PFM = SC / "ctcf_pfm.json"
CHROM, BIN = "chr21", 25000
SEED = 1904
N_SHUFFLE = 200
INSUL_WIN = 10          # bins each side, i.e. 250 kb -- the standard insulation window
PS_LO, PS_HI = 1e5, 1e7  # the decade range P(s) is fitted over
GATE_PS = (-1.6, -0.7)
GATE_BINS = 1000


def insulation(M, w=INSUL_WIN):
    """Mean contact across a sliding diamond. Low = a boundary: few contacts bridge that point."""
    n = len(M)
    ins = np.full(n, np.nan)
    for i in range(w, n - w):
        blk = M[i - w:i, i + 1:i + 1 + w]
        if np.isfinite(blk).any() and blk.size:
            ins[i] = np.nanmean(blk)
    with np.errstate(divide="ignore", invalid="ignore"):
        ins = np.log2(ins / np.nanmean(ins))
    return ins


def expected(M, mask):
    """Mean contact at each separation, over mappable bins only."""
    n = len(M)
    exp = np.full(n, np.nan)
    for d in range(n):
        i = np.arange(0, n - d)
        j = i + d
        ok = mask[i] & mask[j]
        if ok.sum() >= 10:
            # ok says the BINS are mappable; it does not say the entries are finite. A separation
            # where every mappable pair happens to be empty gives nanmean an all-NaN slice and a
            # RuntimeWarning on every chromatin run. The value is genuinely undefined there, so it
            # stays NaN -- but silently rather than noisily, and by intent rather than by accident.
            vals = M[i[ok], j[ok]]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                exp[d] = vals.mean()
    return exp


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("THE TARGET -- what a model has to reproduce, measured on real Hi-C before any model exists")
    say("=" * 100)
    say("  This loop builds NO model. It writes down the windows a simulation will have to land in,")
    say("  so that the simulation can later fail.")
    say("  Prior arc, so none of it is rebuilt: CTCF-count moved a CRISPR classifier 0.6076 -> 0.6632;")
    say("  analytic Rouse added +0.005 on top of that; and MEASURED Hi-C added -0.0031, z = -1.06.")
    say("  That last one says the CLASSIFIER cannot use 3D -- not that chromatin cannot be modelled.")

    M = np.load(HIC).astype(np.float64)
    n = len(M)
    M[M == 0] = np.nan
    mask = np.isfinite(M).sum(1) > 50
    say(f"\n  {CHROM} at {BIN//1000} kb: {n:,} bins, {int(mask.sum()):,} mappable "
        f"(the rest is the short arm and centromere, excluded rather than averaged in)")

    # ---- D1 P(s) ---------------------------------------------------------------------------------
    exp = expected(M, mask)
    d = np.arange(n) * BIN
    sel = np.isfinite(exp) & (d >= PS_LO) & (d <= PS_HI) & (exp > 0)
    slope, icept = np.polyfit(np.log10(d[sel]), np.log10(exp[sel]), 1)
    d1 = bool(mask.sum() >= GATE_BINS and GATE_PS[0] <= slope <= GATE_PS[1])
    say(f"\n  D1 THE MAP IS REAL -- P(s) slope over {PS_LO/1e3:.0f} kb to {PS_HI/1e6:.0f} Mb: "
        f"{slope:.3f}")
    say(f"     fractal globule is about -1.0, equilibrium globule -1.5; gate is "
        f"[{GATE_PS[0]}, {GATE_PS[1]}]")
    say(f"     D1 {'PASS' if d1 else 'FAIL'}")

    # ---- CTCF, with motif orientation --------------------------------------------------------------
    peaks = []
    with gzip.open(CTCF, "rt") as f:
        for line in f:
            p = line.split("\t")
            if p[0] != CHROM:
                continue
            # narrowPeak: summit is col 9 relative to start; signal is col 6
            st, en = int(p[1]), int(p[2])
            summit = st + (int(p[9]) if len(p) > 9 and p[9].strip().lstrip("-").isdigit()
                           and int(p[9]) >= 0 else (en - st) // 2)
            peaks.append({"start": st, "end": en, "summit": summit, "signal": float(p[6])})
    say(f"\n  CTCF: {len(peaks)} GM12878 IDR peaks on {CHROM} (cell-type matched to the map)")

    seq = []
    with gzip.open(FASTA, "rt") as f:
        for line in f:
            if line[0] == ">":
                continue
            seq.append(line.strip())
    seq = "".join(seq).upper()
    say(f"  {CHROM} sequence: {len(seq):,} bp from the same assembly as the map")

    pfm = json.load(open(PFM))
    L = len(pfm["A"])
    bg = 0.25
    W = np.zeros((L, 4))
    for k, b in enumerate("ACGT"):
        col = np.array(pfm[b], float)
        W[:, k] = col
    W = W / W.sum(1, keepdims=True)
    W = np.log2((W + 1e-3) / bg)
    idx = {c: i for i, c in enumerate("ACGT")}

    def score(s):
        if len(s) != L or any(c not in idx for c in s):
            return -1e9
        return float(sum(W[i, idx[c]] for i, c in enumerate(s)))

    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    oriented = 0
    for p in peaks:
        a, b = max(0, p["summit"] - 150), min(len(seq), p["summit"] + 150)
        win = seq[a:b]
        best, bo = -1e9, 0
        for i in range(len(win) - L + 1):
            sub = win[i:i + L]
            f_, r_ = score(sub), score(rc(sub))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["motif"], p["orient"] = best, (bo if best > 6.0 else 0)
        oriented += p["orient"] != 0
    say(f"  motif orientation assigned to {oriented}/{len(peaks)} peaks by scanning JASPAR MA0139.1 "
        f"(score > 6 bits)")

    pb = np.zeros(n)
    for p in peaks:
        b = p["summit"] // BIN
        if 0 <= b < n:
            pb[b] += p["signal"]

    # ---- D2 insulation boundaries and CTCF ---------------------------------------------------------
    ins = insulation(M)
    ok = np.isfinite(ins) & mask
    thr = np.nanpercentile(ins[ok], 10)
    bnd = np.where(ok & (ins <= thr))[0]
    real = float(np.mean([pb[max(0, b - 1):b + 2].sum() for b in bnd]))
    null = []
    idxm = np.where(mask)[0]
    for _ in range(N_SHUFFLE):
        sh = np.zeros(n)
        pos = rng.choice(idxm, size=int((pb > 0).sum()), replace=False)
        sh[pos] = rng.permutation(pb[pb > 0])
        null.append(float(np.mean([sh[max(0, b - 1):b + 2].sum() for b in bnd])))
    null = np.array(null)
    p95 = float(np.percentile(null, 95))
    d2 = bool(real > p95)
    say(f"\n  D2 TADS EXIST AND CTCF MARKS THEM -- {len(bnd)} boundaries (lowest decile of insulation)")
    say(f"     CTCF signal at boundaries {real:.1f} vs shuffled 95th {p95:.1f} "
        f"(null mean {null.mean():.1f})")
    say(f"     D2 {'PASS' if d2 else 'FAIL'}")

    # ---- D3 compartments ---------------------------------------------------------------------------
    O = M.copy()
    for dd in range(n):
        if np.isfinite(exp[dd]) and exp[dd] > 0:
            i = np.arange(0, n - dd)
            O[i, i + dd] = M[i, i + dd] / exp[dd]
            O[i + dd, i] = O[i, i + dd]
    sub = O[np.ix_(mask, mask)]
    sub = np.nan_to_num(sub, nan=1.0)
    C = np.corrcoef(np.log(sub + 1e-6))
    C = np.nan_to_num(C)
    w, v = np.linalg.eigh(C)
    pc1 = v[:, -1]
    gc = np.array([(seq[i * BIN:(i + 1) * BIN].count("G") + seq[i * BIN:(i + 1) * BIN].count("C"))
                   / max(1, len(seq[i * BIN:(i + 1) * BIN].replace("N", "")))
                   for i in range(n)])[mask]
    good = np.isfinite(gc) & (gc > 0)
    r = float(np.corrcoef(pc1[good], gc[good])[0, 1])
    if r < 0:
        pc1, r = -pc1, -r
    nullr = np.array([abs(np.corrcoef(rng.permutation(pc1[good]), gc[good])[0, 1])
                      for _ in range(N_SHUFFLE)])
    d3 = bool(r > np.percentile(nullr, 95))
    say(f"\n  D3 COMPARTMENTS EXIST -- PC1 of the observed/expected correlation matrix vs GC content")
    say(f"     r = {r:+.4f} against a shuffled 95th of {np.percentile(nullr,95):.4f}; "
        f"A compartment is GC-rich, as it should be")
    say(f"     D3 {'PASS' if d3 else 'FAIL'}")

    # ---- D4 orientation, distance matched ----------------------------------------------------------
    fwd = [p["summit"] // BIN for p in peaks if p["orient"] > 0]
    rev = [p["summit"] // BIN for p in peaks if p["orient"] < 0]
    fs, rs = set(fwd), set(rev)
    conv, other, seps = [], [], []
    for i in sorted(fs | rs):
        for j in sorted(fs | rs):
            if j - i < 4 or (j - i) * BIN > 2e6:
                continue
            if not (mask[i] and mask[j]) or not np.isfinite(M[i, j]) or not np.isfinite(exp[j - i]):
                continue
            val = M[i, j] / exp[j - i]
            (conv if (i in fs and j in rs) else other).append((j - i, val))
    say(f"\n  D4 ORIENTATION MATTERS -- {len(conv)} convergent pairs vs {len(other)} non-convergent,")
    say(f"     scored as observed/expected so genomic separation is already divided out")
    # match on separation explicitly as well, so no residual distance effect can carry it
    from collections import defaultdict
    byd = defaultdict(list)
    for dd, v in other:
        byd[dd].append(v)
    mc, mo = [], []
    for dd, v in conv:
        if byd.get(dd):
            mc.append(v)
            mo.append(float(np.mean(byd[dd])))
    d4 = False
    if len(mc) >= 30:
        mc, mo = np.array(mc), np.array(mo)
        diff = float(np.mean(mc - mo))
        nulld = np.array([float(np.mean(rng.permutation(np.r_[mc, mo])[:len(mc)]
                                        - rng.permutation(np.r_[mc, mo])[:len(mc)]))
                          for _ in range(N_SHUFFLE)])
        d4 = bool(diff > np.percentile(nulld, 95))
        say(f"     {len(mc)} separation-matched pairs: convergent {mc.mean():.3f} vs "
            f"non-convergent {mo.mean():.3f}, difference {diff:+.3f}")
        say(f"     against a shuffled 95th of {np.percentile(nulld,95):+.3f}")
    else:
        say(f"     only {len(mc)} separation-matched pairs -- NOT TESTABLE at this resolution")
    say(f"     D4 {'PASS' if d4 else 'FAIL'}")

    # ---- the predeclared model acceptance windows --------------------------------------------------
    target = {
        "ps_slope": {"value": float(slope), "window": [float(slope) - 0.20, float(slope) + 0.20],
                     "why": "P(s) is the most robust property of interphase chromatin; miss it and the "
                            "object is not a chromosome at any scale"},
        "insulation_pearson": {"gate": 0.40,
                               "why": "the simulated insulation profile must track the measured one "
                                      "along the chromosome, not merely have boundaries somewhere"},
        "boundary_ctcf_enrichment": {"measured": real, "null95": p95,
                                     "why": "boundaries must land on CTCF, as they do here"},
        "compartment_r": {"measured": r, "gate": 0.30,
                          "why": "A/B segregation against GC, the same anchor used here"},
        "convergent_minus_other": {"measured": (float(np.mean(mc - mo)) if len(mc) >= 30
                                                else None),
                                   "why": "the extrusion signature: orientation must matter at matched "
                                          "separation"},
    }
    literature = {
        "msd_exponent": {"window": [0.35, 0.55],
                         "source": "live-imaging of tagged loci, LITERATURE not measured here"},
        "cohesin_residence_min": {"window": [10, 25],
                                  "source": "FRAP/single-molecule, LITERATURE not measured here"},
        "extrusion_speed_kb_s": {"window": [0.1, 2.0],
                                 "source": "single-molecule extrusion, LITERATURE not measured here"},
    }
    say("\n" + "-" * 100)
    say("  THE WINDOWS A MODEL WILL HAVE TO LAND IN -- written now, before the model exists")
    say("-" * 100)
    for k, v in target.items():
        say(f"    {k:<28}{json.dumps({a: b for a, b in v.items() if a != 'why'})}")
    say("    the FOURTH dimension cannot be scored on this map at all -- Hi-C is a fixed population")
    say("    average with no time axis. Dynamics will be scored against LITERATURE values, labelled:")
    for k, v in literature.items():
        say(f"    {k:<28}{v['window']}   <- {v['source']}")

    say("\n" + "=" * 100)
    gates = {"D1 the map is real": d1, "D2 TADs exist and CTCF marks them": d2,
             "D3 compartments exist": d3, "D4 orientation matters": d4}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    if all(gates.values()):
        say("  THE TARGET IS REAL AND FALSIFIABLE. All four signatures are present in this map, and")
        say("  the convergent-CTCF effect survives separation matching, which means loop extrusion is")
        say("  visible here and a model that lacks it can be caught.")
    elif not d4:
        say("  THE EXTRUSION SIGNATURE IS NOT VISIBLE AT THIS RESOLUTION. The model can still be scored")
        say("  on P(s), insulation and compartments, but this map cannot tell an extrusion model from a")
        say("  generic confined polymer, and that limit has to be carried forward explicitly.")
    else:
        say("  THE TARGET IS NOT SOUND. Fix the extraction before building anything against it.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(HIC), str(CTCF), str(FASTA)],
                      available=n, used=int(mask.sum()), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} within-chromosome CTCF shuffles preserving count",
                                "separation-matched comparison for every orientation claim",
                                "GC content from the same assembly as the map",
                                "unmappable centromeric bins excluded up front",
                                "literature dynamics values labelled as literature"],
                      note="measures the target before any model exists so the model can fail; the "
                           "fourth dimension is explicitly NOT scorable on Hi-C")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_hic_target", "manifest": man, "gates": gates,
               "n_bins": n, "n_mappable": int(mask.sum()), "ps_slope": float(slope),
               "boundary_ctcf": real, "boundary_null95": p95, "n_boundaries": len(bnd),
               "compartment_r": r, "n_peaks": len(peaks), "n_oriented": int(oriented),
               "target": target, "literature": literature, "log": log},
              open(OUT / "loop_hic_target.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_hic_target.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
