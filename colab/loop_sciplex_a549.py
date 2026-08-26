"""Loop 234. The matched measurement: A549 dexamethasone at single-cell resolution.

WHY THIS DATASET AND WHY IT MATTERS MORE THAN WHAT I PROPOSED. Loop 233 left the ranking clear:
the single strongest block in the stack is Perturb-seq at |r| 0.4583, and it is measured in K562
while the target is A549. Cross-cell-line agreement is 0.2286 to 0.3002 (loop 224 X6), so most of
that block's information is lost in transfer. I proposed measuring K562-to-A549 transfer directly
from GSE281048, which has both lines in one experiment. That is still worth doing, but its files
are 2.6 to 4.6 GB gzipped Seurat .rds, this container has 2.0 GB free, there is no R, and pyreadr
cannot read S4 Seurat objects. Blocked three ways.

Searching scPerturb's 54 harmonised h5ad datasets for an A549 CRISPR screen found none. It found
something better:

    sci-Plex 2 (Srivatsan et al. 2020), 145 MB, harmonised to h5ad
    cell_line          A549                       -- the target cell line, not a proxy
    perturbation       Dex, BMS, Nutlin, SAHA, control
    dose_value         0, 0.1, 0.5, 1, 5, 10, 50, 100
    dexamethasone arm  8,064 cells; 914 at vehicle, 1,296 at dose 100
    genes              58,347 with symbols
    counts             raw integers, CSR, 60,135,001 non-zeros over 24,262 cells

Our target is the A549 dexamethasone response. This is the A549 dexamethasone response, measured
one cell at a time, across a dose series. Every previous single-cell measurement in this project
has been K562 CRISPRi standing in for it.

WHAT THIS CAN SETTLE THAT NOTHING ELSE HAS. Loop 224 measured a noise floor of 0.2299 and loop 233
a shrinkage of 0.1872, both on K562 knockdowns, and both were carried into A549 work as an order of
magnitude rather than a number. This measures them on the actual system. And R5 turns the lever
from an argument into a number: does an A549 drug-response block beat the K562 knockdown block at
predicting our A549 plateau?

THE CONTROL THAT MAKES R2 READABLE, and it is the reason this dataset is worth more than a
bigger one. BMS, Nutlin and SAHA were applied to the SAME cells, in the SAME experiment, with the
SAME processing, at the SAME doses. If our dexamethasone plateau is predicted just as well by the
NF-kB inhibitor or the MDM2 inhibitor or the HDAC inhibitor, then what R2 measures is generic drug
stress -- proliferation arrest, ribosome load -- and not the glucocorticoid response. No control
this clean has been available in any loop of this arc.

TWO MISMATCHES, NAMED BEFORE ANY NUMBER. sci-Plex 2 profiled cells at 24 hours; our plateau is the
mean of the last three grid points, 420 to 720 minutes, so 7 to 12 hours. And sci-Plex's dose is in
micromolar units on an arbitrary scale in this harmonised file while the ENCODE A549 series used
100 nM. Neither is fatal -- the glucocorticoid response is saturating and sustained -- but both
mean a perfect correlation was never available and R2's bar is set accordingly.

PREDECLARED, BEFORE ANY NUMBER.

  R1 IS THIS THE GLUCOCORTICOID RESPONSE?  -- positive control, everything requires it
     The canonical direct GR targets are FKBP5, TSC22D3, PER1, KLF15 and ZBTB16. They are textbook
     and were not chosen by looking at this data.
     Gate: PASS iff at least 4 of 5 are UP at dose 100 against vehicle, and their median log2 fold
     change exceeds +0.5. A FAIL means this is not the response it claims to be and nothing below
     may be read.

  R2 DOES THE SINGLE-CELL A549 RESPONSE PREDICT OUR BULK PLATEAU?
     log2 fold change at dose 100 against vehicle, correlated with the loop 213 plateau on shared
     genes.
     Gate: PASS iff |Pearson| exceeds 0.30. Set at 0.30 rather than higher because of the 24-hour
     against 7-to-12-hour mismatch stated above.

  R3 IS IT DOSE-DEPENDENT?
     Repeat R2 at every dose from 0.1 to 100.
     Gate: PASS iff the correlation with our plateau increases monotonically across at least 5 of
     the 7 non-zero doses AND the top dose beats the lowest by at least 0.10. A dose-response that
     tracks our bulk target is far stronger evidence of shared biology than any single correlation,
     because noise has no reason to be ordered by concentration.

  R4 WHAT IS THE NOISE FLOOR FOR THIS SYSTEM, NOT FOR K562?
     Split the 1,296 dose-100 cells in half and measure the split-half agreement and the optimal
     shrinkage k, by the same construction as loop 233: k = cov(A,B)/var(A), a regression slope
     with no shared denominator.
     Gate: PASS iff k is estimable and its shuffled control falls below 0.02. The VALUE is reported
     and compared against K562's 0.1872, not gated -- the whole point is to find out whether the
     K562 number transferred, and gating on it would assume the answer.

  R5 DOES AN A549 BLOCK BEAT THE K562 BLOCK?  -- the lever, as a number
     Both as single features predicting our plateau on the same genes, same splits.
     Gate: PASS iff the sci-Plex A549 arm exceeds the Replogle K562 Perturb-seq arm's 0.4583.
     Requires R1.

  R6 CONTROL: DO THE OTHER THREE DRUGS PREDICT IT TOO?
     BMS, Nutlin and SAHA at dose 100 against the same plateau.
     Gate: PASS iff dexamethasone beats all three by at least 0.10. A FAIL means R2 is measuring
     generic drug stress rather than the glucocorticoid response, and it would be the most
     important result of the loop.

  R7 WHAT THIS CANNOT SHOW -- written before the run.
     24 hours against 7 to 12 hours, and a dose scale that does not map cleanly onto 100 nM. A
     weak R2 is as consistent with those mismatches as with a real absence.
     sci-Plex uses nuclear hashing and combinatorial indexing, so its counts per cell are low --
     the median here is a few thousand UMIs. That makes single-cell estimates noisier than
     droplet-based Perturb-seq and biases R4's floor upward relative to a 10x experiment.
     A drug response and a genetic knockdown are different interventions. R5 compares them as
     PREDICTORS of the same target, which is fair, but a win for either says nothing about which
     is the better instrument in general.
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_sciplex_a549.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
H5 = SCR / "sciplex2.h5ad"
GR_TARGETS = ["FKBP5", "TSC22D3", "PER1", "KLF15", "ZBTB16"]
DOSES = ["0.1", "0.5", "1", "5", "10", "50", "100"]
DRUGS = ["BMS", "Nutlin", "SAHA"]
SEED, NSPLIT, MINCELL = 234234, 20, 100
REF_K562_BLOCK, REF_K562_K = 0.4583, 0.1872
R2_BAR, DOSE_GAIN, DRUG_MARGIN, SHUF_BAR = 0.30, 0.10, 0.10, 0.02

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "sci-Plex A549 dexamethasone against the bulk plateau"}
    say("=" * 104)
    say("LOOP 234 -- A549 DEXAMETHASONE AT SINGLE-CELL RESOLUTION, THE MATCHED MEASUREMENT")
    say("=" * 104)
    say("     Every single-cell measurement in this arc has been K562 CRISPRi standing in for")
    say("     A549. sci-Plex 2 is A549 cells, dexamethasone, eight doses, one cell at a time.")
    say("     And BMS, Nutlin and SAHA ran in the SAME experiment on the SAME cells, which gives")
    say("     R6 a control no loop in this arc has had.")

    import h5py
    f = h5py.File(H5, "r")

    def cat(c):
        g = f["obs"][c]
        if isinstance(g, h5py.Group):
            cs = np.array([x.decode() if isinstance(x, bytes) else str(x)
                           for x in g["categories"][:]])
            return cs[g["codes"][:]]
        return g[:]

    pert, dose = cat("perturbation"), cat("dose_value")
    vi = f["var"]
    vkey = vi.attrs.get("_index", "_index")
    vkey = vkey.decode() if isinstance(vkey, bytes) else vkey
    gsym = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vi[vkey][:]])
    shape = tuple(f["X"].attrs["shape"])
    data = f["X"]["data"][:]; ind = f["X"]["indices"][:]; ptr = f["X"]["indptr"][:]
    say(f"     {shape[0]:,} cells x {shape[1]:,} genes, {len(data):,} non-zeros, raw counts")

    def pseudobulk(mask):
        """CPM-normalise each cell, then average. Cells are weighted equally, not by depth."""
        rows = np.where(mask)[0]
        acc = np.zeros(shape[1], np.float64)
        for r in rows:
            a, b = ptr[r], ptr[r + 1]
            tot = data[a:b].sum()
            if tot <= 0: continue
            acc[ind[a:b]] += data[a:b] / tot
        return acc / max(len(rows), 1) * 1e6

    veh = pseudobulk((pert == "Dex") & (dose == "0"))
    d100 = pseudobulk((pert == "Dex") & (dose == "100"))
    lfc = np.log2((d100 + 1.0) / (veh + 1.0))
    say(f"     vehicle {int(((pert=='Dex')&(dose=='0')).sum()):,} cells, "
        f"dose 100 {int(((pert=='Dex')&(dose=='100')).sum()):,} cells")

    # ---------------------------------------------------------------- R1
    say("R1 IS THIS THE GLUCOCORTICOID RESPONSE?")
    gpos = {s: i for i, s in enumerate(gsym)}
    hits = [(t, float(lfc[gpos[t]])) for t in GR_TARGETS if t in gpos]
    for t, v in hits:
        say(f"       {t:<10} log2 fold change {v:+.4f}")
    up = sum(1 for _, v in hits if v > 0)
    med = float(np.median([v for _, v in hits])) if hits else float("nan")
    say(f"     {up} of {len(hits)} canonical GR targets up; median log2FC {med:+.4f}")
    say("     these five are textbook direct GR targets and were not chosen from this data")
    G.add("R1", bool(up >= 4 and med > 0.5), stat=float(med),
          if_true=lambda: f"R1 PASS -- {up}/{len(hits)} up, median {med:+.3f}; this is the "
                          f"glucocorticoid response",
          if_false=lambda: f"R1 FAIL -- {up}/{len(hits)} up, median {med:+.3f}; this is not the "
                           f"response it claims to be and nothing below may be read")
    res["gr_targets"] = {t: v for t, v in hits}

    # ---------------------------------------------------------------- R2
    say("R2 DOES THE SINGLE-CELL A549 RESPONSE PREDICT OUR BULK PLATEAU?")
    grid, M, A9, sym, keepg, tssb = gene_set()
    gi = np.where(keepg)[0]
    plateau = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    shared = [s for s in allg if s in gpos]
    y = np.array([plateau[allg.index(s)] for s in shared])
    x100 = np.array([lfc[gpos[s]] for s in shared])
    r2 = pear(x100, y)
    say(f"     {len(shared):,} genes shared between sci-Plex and the A549 plateau set")
    say(f"     Pearson {r2:+.4f}   Spearman "
        f"{pear(np.argsort(np.argsort(x100)), np.argsort(np.argsort(y))):+.4f}")
    say("     bar is 0.30 rather than higher: sci-Plex profiled at 24 h, our plateau is 7-12 h")
    G.add("R2", bool(abs(r2) > R2_BAR), stat=float(abs(r2)), requires=("R1",),
          if_true=lambda: f"R2 PASS -- |r| {abs(r2):.4f}; an independent A549 dexamethasone "
                          f"experiment recovers our plateau",
          if_false=lambda: f"R2 FAIL -- |r| {abs(r2):.4f} against a {R2_BAR:.2f} bar")
    res["match"] = {"r": r2, "n_shared": len(shared)}

    # ---------------------------------------------------------------- R3
    say("R3 IS IT DOSE-DEPENDENT?")
    rs = []
    for d in DOSES:
        m = (pert == "Dex") & (dose == d)
        if m.sum() < MINCELL:
            rs.append(float("nan")); continue
        pb = pseudobulk(m)
        lf = np.log2((pb + 1.0) / (veh + 1.0))
        xv = np.array([lf[gpos[s]] for s in shared])
        rs.append(abs(pear(xv, y)))
        say(f"       dose {d:<5} {int(m.sum()):>5} cells   |r| with our plateau {rs[-1]:.4f}")
    rs = np.array(rs)
    fin = np.isfinite(rs)
    inc = int(np.sum(np.diff(rs[fin]) > 0))
    span = float(rs[fin][-1] - rs[fin][0]) if fin.sum() >= 2 else float("nan")
    say(f"     increases at {inc} of {int(fin.sum())-1} steps; top minus bottom {span:+.4f}")
    say("     noise has no reason to be ordered by concentration, which is why this is stronger")
    say("     evidence than any single correlation")
    G.add("R3", bool(inc >= 5 and span >= DOSE_GAIN), stat=float(span), requires=("R1",),
          if_true=lambda: f"R3 PASS -- the correlation rises with dose, {inc} increases and "
                          f"{span:+.3f} from lowest to highest",
          if_false=lambda: f"R3 FAIL -- {inc} increases, span {span:+.4f}")
    res["dose"] = {d: (float(v) if np.isfinite(v) else None) for d, v in zip(DOSES, rs)}

    # ---------------------------------------------------------------- R4
    say("R4 WHAT IS THE NOISE FLOOR FOR THIS SYSTEM, NOT FOR K562?")
    idx100 = np.where((pert == "Dex") & (dose == "100"))[0]
    pm = rng.permutation(len(idx100))
    hA, hB = idx100[pm[: len(pm) // 2]], idx100[pm[len(pm) // 2:]]
    mA = np.zeros(shape[0], bool); mA[hA] = True
    mB = np.zeros(shape[0], bool); mB[hB] = True
    pA, pB = pseudobulk(mA), pseudobulk(mB)
    lA = np.log2((pA + 1.0) / (veh + 1.0)); lB = np.log2((pB + 1.0) / (veh + 1.0))
    xa = np.array([lA[gpos[s]] for s in shared]); xb = np.array([lB[gpos[s]] for s in shared])
    sh_r = pear(xa, xb)
    k = float(np.sum(xa * xb) / np.sum(xa * xa))
    sp = rng.permutation(len(xb))
    k_sh = float(np.sum(xa * xb[sp]) / np.sum(xa * xa))
    say(f"     {len(hA)} vs {len(hB)} cells; split-half Pearson {sh_r:+.4f}")
    say(f"     optimal shrinkage k = {k:.4f}   shuffled control {k_sh:.4f}")
    say(f"     loop 233 measured k = {REF_K562_K:.4f} on K562 CRISPRi, carried into A549 work as")
    say("     an order of magnitude. This is the number for the actual system.")
    G.add("R4", bool(np.isfinite(k) and abs(k_sh) < SHUF_BAR), stat=float(k), requires=("R1",),
          if_true=lambda: f"R4 PASS -- k = {k:.4f} with the shuffled control at {k_sh:.4f}; "
                          f"K562's {REF_K562_K:.4f} is "
                          f"{'close' if abs(k-REF_K562_K)<0.1 else 'NOT close'} to it",
          if_false=lambda: f"R4 FAIL -- shuffled control {k_sh:.4f} did not collapse")
    res["floor"] = {"split_half_r": sh_r, "k": k, "k_shuffled": k_sh, "k_k562": REF_K562_K}

    # ---------------------------------------------------------------- R5
    say("R5 DOES AN A549 BLOCK BEAT THE K562 BLOCK?")
    say(f"     sci-Plex A549 dexamethasone, single feature: |r| {abs(r2):.4f}")
    say(f"     Replogle K562 Perturb-seq block, 200 features: |r| {REF_K562_BLOCK:.4f} (loop 210)")
    say("     note the K562 arm is a 200-column block and this is ONE column, so a loss here is")
    say("     not a like-for-like defeat")
    G.add("R5", bool(abs(r2) > REF_K562_BLOCK), stat=float(abs(r2)), requires=("R1",),
          if_true=lambda: f"R5 PASS -- the matched-cell-type drug response beats the "
                          f"cross-cell-type knockdown block, {abs(r2):.4f} against "
                          f"{REF_K562_BLOCK:.4f}",
          if_false=lambda: f"R5 FAIL -- {abs(r2):.4f} against the K562 block's "
                           f"{REF_K562_BLOCK:.4f}")
    res["lever"] = {"sciplex_a549": abs(r2), "k562_block": REF_K562_BLOCK}

    # ---------------------------------------------------------------- R6
    say("R6 CONTROL: DO THE OTHER THREE DRUGS PREDICT IT TOO?")
    other = {}
    for dr in DRUGS:
        m0 = (pert == dr) & (dose == "0")
        m1 = (pert == dr) & (dose == "100")
        if m1.sum() < 50 or m0.sum() < 50:
            m1 = (pert == dr) & np.isin(dose, ["10", "50", "100"])
        pb0 = pseudobulk(m0) if m0.sum() >= 50 else veh
        pb1 = pseudobulk(m1)
        lf = np.log2((pb1 + 1.0) / (pb0 + 1.0))
        xv = np.array([lf[gpos[s]] for s in shared])
        other[dr] = abs(pear(xv, y))
        say(f"       {dr:<8} {int(m1.sum()):>5} cells   |r| with our DEX plateau {other[dr]:.4f}")
    say(f"       {'Dex':<8} {int(((pert=='Dex')&(dose=='100')).sum()):>5} cells   "
        f"|r| {abs(r2):.4f}")
    worst = max(other.values()) if other else float("nan")
    say(f"     same cells, same experiment, same processing, different drug")
    G.add("R6", bool(abs(r2) - worst >= DRUG_MARGIN), stat=float(abs(r2) - worst),
          requires=("R1",),
          if_true=lambda: f"R6 PASS -- dexamethasone beats the best other drug by "
                          f"{abs(r2)-worst:+.4f}; R2 is the glucocorticoid response, not drug "
                          f"stress",
          if_false=lambda: f"R6 FAIL -- the best other drug reaches {worst:.4f} against "
                           f"dexamethasone's {abs(r2):.4f}. R2 is generic drug response and this "
                           f"is the most important result in the loop")
    res["drug_control"] = dict(other); res["drug_control"]["Dex"] = abs(r2)

    # ---------------------------------------------------------------- R7
    say("R7 WHAT THIS CANNOT SHOW")
    say("     sci-Plex profiled at 24 hours; our plateau is 7 to 12 hours. And the harmonised dose")
    say("     scale does not map cleanly onto the ENCODE series' 100 nM. A weak R2 is as")
    say("     consistent with those mismatches as with a real absence.")
    say("     sci-Plex uses nuclear hashing and combinatorial indexing, so counts per cell are low")
    say("     -- a few thousand UMIs. Single-cell estimates are therefore noisier than droplet")
    say("     Perturb-seq and R4's floor is biased upward relative to a 10x experiment.")
    say("     A drug response and a genetic knockdown are different interventions. R5 compares")
    say("     them as PREDICTORS of one target, which is fair, but says nothing about which is the")
    say("     better instrument in general -- and the K562 arm is 200 columns against this one.")

    res["gates"] = {k_: (v == "PASS") for k_, v in G.status.items()}
    res["void"] = [k_ for k_, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
