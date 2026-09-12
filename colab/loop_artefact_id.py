"""Loop 221. Naming the artefact, and testing whether removing it rescues the dataset.

WHAT LOOP 220 ESTABLISHED. The first principal component of the replicate-2-and-3 average change
explains 27.63% and 31.01% of those replicates' change variance and only 3.91% and 3.06% of
replicates 1 and 4 -- a ratio of 8.41x. Two of four replicates share a large component the other
two lack, and it is what loops 217, 218 and 219 were each finding and being defeated by.

Loop 220 named it a shared artefact and stopped there, because naming a component is not explaining
it. This loop makes an explicit, falsifiable claim about WHAT it is, and tests it.

THE CLAIM, STATED BEFORE ANY TEST. Read off the component's top loadings, it is dominated by:

    snoRNAs        SNORA7A, SNORA50A, SNORA44, SNORD3A, SNORD3B-1, SNORD3B-2, SNORD88C,
                   SNORD116-3, SNORD116-9, SNORD10
    snRNA          RNU5E-1
    snoRNA hosts   SNHG25
    processed      RPLP0P9, RPL3P4, RPL21P16, RPS26P19, DHFRP1, PRELID1P1, PRELID1P6,
    pseudogenes    SNRPEP4, HNRNPCP2, SNRPGP2, SUMO1P3

These are precisely the classes whose MEASURED abundance depends on library chemistry rather than
on transcription: short structured non-coding RNAs are captured or lost according to polyA
selection efficiency and fragmentation, and processed pseudogenes are multi-mapping against their
parent genes so their apparent counts move with alignment and with ribosomal contamination.

And the component's time profile does not look like a drug response. Across the eight intervals it
runs -0.2248, +0.6536, -0.1574, -0.2712, +0.0529, +0.1296, +0.2964, -0.5626 -- it changes sign five
times. A dexamethasone response is a rise or a fall to a plateau; it does not oscillate.

    HYPOTHESIS: the component is a LIBRARY-PREPARATION artefact, and it is carried by a small,
    identifiable set of gene biotypes. Removing those biotypes should remove the component and
    make all four replicates agree.

    IF THE HYPOTHESIS IS WRONG, Q2 or Q3 fails and Q4 must not be read as a rescue.

BIOTYPE ASSIGNMENT IS BY NAME PATTERN AND THAT IS A REAL LIMITATION, declared here rather than in a
footnote: SNOR*/SNHG*/RNU*/RNVU*/SCARNA* for small non-coding RNAs, and a trailing P followed by
digits for processed pseudogenes. This is crude, it will miss unnamed ENSG entries, and Q1 measures
how crude by checking the classes it assigns against their share of the whole roster.

PREDECLARED, BEFORE ANY NUMBER.

  Q1 IS THE BIOTYPE CALL USABLE?
     Gate: PASS iff the pattern assigns a class to between 2% and 40% of the retained genes -- a
     rule matching almost nothing cannot be tested and one matching almost everything is not a
     class. Report the counts so the reader can judge the rule rather than trust it.

  Q2 IS THE COMPONENT CARRIED BY THOSE BIOTYPES?  -- the hypothesis, tested
     Gate: PASS iff the flagged biotypes are enriched at least 3x among the top 200 loadings
     relative to their share of all retained genes, with a hypergeometric p below 1e-6.
     A FAIL refutes the claim above and Q4 must not be read.

  Q3 IS THE TIME PROFILE NON-BIOLOGICAL?
     Compare the component's sign changes and lag-1 autocorrelation against the same statistics for
     the 200 genes with the largest measured plateau -- genes that unambiguously respond to the drug.
     Gate: PASS iff the component has MORE sign changes and LOWER autocorrelation than the 95th
     percentile of real responders. Magnitude comparison against a real distribution, not against
     an assumption about what biology looks like.

  Q4 DOES REMOVING THE BIOTYPES RESCUE REPLICATE AGREEMENT?  -- the whole loop
     Recompute all six pairwise agreements on the per-interval change with the flagged biotypes
     dropped.
     Gate: PASS iff the mean of the five currently-disagreeing pairs rises by more than 0.15 in
     Pearson. Loop 218 measured them at 1v2 +0.1245, 1v3 +0.1195, 1v4 +0.1361, 2v4 +0.1616,
     3v4 +0.1481 against 2v3 +0.6007.
     Requires Q2.

  Q5 DOES THE CEILING MOVE?
     Gene-level and module-level replicate ceilings after filtering.
     Gate: PASS iff the gene-level ceiling rises above -0.10, from loop 220's -0.33410 at genome
     scale. Requires Q4.

  Q6 DO THE OTHER DATASETS CARRY A SHARED COMPONENT TOO?
     Apply the same decomposition to the Perturb-seq matrices across cell lines (K562 against
     RPE1) and to the ChIP tracks across factors.
     Gate: PASS iff at least one of them shows a dominant component explaining more than 20% of
     variance, which would say the pattern is general rather than specific to this RNA series.

  Q7 ARE THE ANNOTATION BLOCKS ONE THING OR MANY?
     Loop 213 stacked ten blocks to |r| 0.5474, of which network alone reached 0.4567 and function
     0.4007. Decompose the six annotation blocks jointly and measure how much one component
     explains.
     Gate: PASS iff the first component explains LESS than 50% of the joint variance. A FAIL means
     the ten-block stack is measuring one latent thing and loop 213's gain is thinner than it looks.

  Q8 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, pickle, re, sys, time, warnings
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
OUT = "outputs/loop_artefact_id.json"
GRID = [30, 60, 120, 180, 240, 420, 480, 600, 720]
MIN_TPM, SEED = 1.0, 221221
NC = re.compile(r"^(SNOR|SNHG|RNU|RNVU|SCARNA|RNY|RN7|VTRNA|MIR)", re.I)
PSG = re.compile(r"P\d+$")
REF_PAIRS = {"1v2": 0.1245, "1v3": 0.1195, "1v4": 0.1361,
             "2v3": 0.6007, "2v4": 0.1616, "3v4": 0.1481}
REF_GENE_CEIL = -0.33410

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "naming the artefact"}
    say("=" * 104)
    say("LOOP 221 -- NAMING THE ARTEFACT, AND TESTING WHETHER REMOVING IT RESCUES THE DATASET")
    say("=" * 104)
    say("     CLAIM, made before testing: the loop-220 component is a LIBRARY-PREPARATION")
    say("     artefact carried by small non-coding RNAs and processed pseudogenes -- the classes")
    say("     whose measured abundance depends on polyA selection, fragmentation and")
    say("     multi-mapping rather than on transcription.")

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps, ensg = z["tpm"], z["mins"].astype(int), z["reps"].astype(int), z["genes"]
    g = np.array(GRID, float)
    base = {r: tpm[(mins == 30) & (reps == r)].mean(0) for r in (1, 2, 3, 4)}
    sel = np.where(np.all([base[r] >= MIN_TPM for r in (1, 2, 3, 4)], axis=0))[0]
    e2s = L191.ensg_to_symbol(lambda *_: None)
    sym = np.array([e2s.get(str(x).split(".")[0], "") for x in ensg[sel]])
    D = {}
    for r in (1, 2, 3, 4):
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (r,), g)
        D[r] = np.array([Mi[j, sel] - Mi[j - 1, sel] for j in range(1, len(g))])
    ngen = len(sel)

    # ---------------------------------------------------------------- Q1
    say("Q1 IS THE BIOTYPE CALL USABLE?")
    is_nc = np.array([bool(NC.match(s)) for s in sym])
    is_ps = np.array([bool(PSG.search(s)) and not bool(NC.match(s)) for s in sym])
    flag = is_nc | is_ps
    say(f"     retained genes {ngen:,}")
    say(f"       small non-coding by name pattern  {int(is_nc.sum()):,} ({is_nc.mean():.2%})")
    say(f"       processed pseudogene by pattern   {int(is_ps.sum()):,} ({is_ps.mean():.2%})")
    say(f"       flagged in total                  {int(flag.sum()):,} ({flag.mean():.2%})")
    say(f"       unnamed (no symbol), not flagged  {int((sym=='').sum()):,}")
    ok1 = 0.02 <= flag.mean() <= 0.40
    G.add("Q1", ok1, stat=float(flag.mean()),
          if_true=lambda: f"Q1 PASS -- the rule flags {flag.mean():.1%} of retained genes",
          if_false=lambda: f"Q1 FAIL -- the rule flags {flag.mean():.1%}")

    A = np.mean([D[2], D[3]], axis=0); A = A - A.mean(1, keepdims=True)
    U, S_, Vt = np.linalg.svd(A, full_matrices=False)
    pc1 = Vt[0]
    ve = float(S_[0] ** 2 / np.sum(S_ ** 2))
    say(f"     component variance explained {ve:.2%}")

    # ---------------------------------------------------------------- Q2
    say("Q2 IS THE COMPONENT CARRIED BY THOSE BIOTYPES?")
    top = np.argsort(-np.abs(pc1))[:200]
    hit = int(flag[top].sum())
    bg = float(flag.mean())
    enr = (hit / 200) / bg if bg > 0 else float("inf")
    from scipy.stats import hypergeom
    p = float(hypergeom.sf(hit - 1, ngen, int(flag.sum()), 200))
    say(f"     top 200 loadings: {hit} flagged ({hit/200:.1%}) against a background of {bg:.2%}")
    say(f"     enrichment {enr:.2f}x   hypergeometric p {p:.3e}")
    G.add("Q2", bool(enr >= 3.0 and p < 1e-6), stat=enr, requires=("Q1",),
          if_true=lambda: f"Q2 PASS -- {enr:.1f}x enrichment, p {p:.1e}. The claim holds: the "
                          f"component is carried by small non-coding RNAs and pseudogenes",
          if_false=lambda: f"Q2 FAIL -- {enr:.2f}x enrichment, p {p:.1e}. My naming of the "
                           f"component is REFUTED and Q4 must not be read as a rescue")
    res["enrichment"] = {"hits": hit, "background": bg, "fold": enr, "p": p, "var_explained": ve}

    # ---------------------------------------------------------------- Q3
    say("Q3 IS THE TIME PROFILE NON-BIOLOGICAL?")
    prof = U[:, 0]
    def sgn_ch(v):
        return int(np.sum(np.diff(np.sign(v)) != 0))
    def ac1(v):
        v = v - v.mean()
        return float(np.sum(v[:-1] * v[1:]) / (np.sum(v * v) + 1e-12))
    Mall, _ = L191.rep_trajectories(tpm, mins, reps, (2, 3), g)
    pl = Mall[-3:, sel].mean(0)
    resp = np.argsort(-np.abs(pl))[:200]
    Dm = np.mean([D[2], D[3]], axis=0)
    rc = np.array([sgn_ch(Dm[:, i]) for i in resp])
    ra = np.array([ac1(Dm[:, i]) for i in resp])
    say(f"     component: {sgn_ch(prof)} sign changes, lag-1 autocorrelation {ac1(prof):+.4f}")
    say(f"     200 strongest real responders: sign changes median {int(np.median(rc))}, "
        f"95th pct {int(np.percentile(rc,95))}")
    say(f"                                    autocorrelation median {np.median(ra):+.4f}, "
        f"5th pct {np.percentile(ra,5):+.4f}")
    ok3 = bool(sgn_ch(prof) >= np.percentile(rc, 95) and ac1(prof) <= np.percentile(ra, 5))
    G.add("Q3", ok3, stat=float(ac1(prof)), requires=("Q1",),
          if_true=lambda: f"Q3 PASS -- the profile oscillates more and correlates less than 95% of "
                          f"real responders; it does not look like a drug response",
          if_false=lambda: f"Q3 FAIL -- {sgn_ch(prof)} sign changes and autocorrelation "
                           f"{ac1(prof):+.3f} sit inside the range of real responders")
    res["profile"] = {"sign_changes": sgn_ch(prof), "ac1": ac1(prof),
                      "resp_sign_p95": float(np.percentile(rc, 95)),
                      "resp_ac_p5": float(np.percentile(ra, 5)),
                      "trace": prof.tolist()}

    # ---------------------------------------------------------------- Q4
    say("Q4 DOES REMOVING THE BIOTYPES RESCUE REPLICATE AGREEMENT?")
    kp = ~flag
    say(f"     dropping {int(flag.sum()):,} flagged genes, {int(kp.sum()):,} remain")
    now = {}
    for a, b in combinations((1, 2, 3, 4), 2):
        v = float(np.corrcoef(D[a][:, kp].ravel(), D[b][:, kp].ravel())[0, 1])
        now[f"{a}v{b}"] = v
        say(f"       {a} vs {b}   before {REF_PAIRS[f'{a}v{b}']:+.4f}   after {v:+.4f}   "
            f"delta {v-REF_PAIRS[f'{a}v{b}']:+.4f}")
    dis = ["1v2", "1v3", "1v4", "2v4", "3v4"]
    before = float(np.mean([REF_PAIRS[k] for k in dis]))
    after = float(np.mean([now[k] for k in dis]))
    say(f"     five disagreeing pairs: mean before {before:+.4f}   after {after:+.4f}   "
        f"delta {after-before:+.4f}")
    G.add("Q4", bool(after - before > 0.15), stat=after - before, requires=("Q2",),
          if_true=lambda: f"Q4 PASS -- removing the flagged biotypes raises the five disagreeing "
                          f"pairs by {after-before:+.4f}. The dataset is rescued",
          if_false=lambda: f"Q4 FAIL -- the five disagreeing pairs move by {after-before:+.4f}. "
                           f"The biotypes carry the component but removing them does not restore "
                           f"agreement, so they are a marker of the artefact and not its whole "
                           f"substance")
    res["pairs_after"] = now

    # ---------------------------------------------------------------- Q5
    say("Q5 DOES THE CEILING MOVE?")
    c_after = r2s(D[3][:, kp].ravel(), D[2][:, kp].ravel())
    say(f"     gene-level ceiling, rep2 predicts rep3:  before {REF_GENE_CEIL:+.5f}   "
        f"after {c_after:+.5f}")
    G.add("Q5", bool(c_after > -0.10), stat=c_after, requires=("Q4",),
          if_true=lambda: f"Q5 PASS -- the gene-level ceiling rises to {c_after:+.4f}",
          if_false=lambda: f"Q5 FAIL -- {c_after:+.4f}")
    res["ceiling_after"] = c_after

    # ---------------------------------------------------------------- Q6
    say("Q6 DO THE OTHER DATASETS CARRY A SHARED COMPONENT TOO?")
    shares = {}
    try:
        fk = h5py.File(SP / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad", "r")
        Xs = fk["X"][:1500, :]
        Xs = Xs[np.isfinite(Xs).all(1)]
        Xc = Xs - Xs.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False)
        shares["perturbseq_K562"] = float(s[0] ** 2 / np.sum(s ** 2))
        say(f"       Perturb-seq K562, 1,500 perturbations: first component "
            f"{shares['perturbseq_K562']:.2%} of variance")
    except Exception as e:
        say(f"       Perturb-seq: {type(e).__name__}")
    grid6, M6, A9, sym6, keep6, tssb = gene_set()
    gi6 = np.where(keep6)[0]
    TRACKS = ["NR3C1", "EP300", "JUN", "JUNB", "CEBPB", "FOSL2", "DNase", "CTCF", "RAD21"]
    cols = []
    for t in TRACKS:
        pt, PM = L191.promoter_track(t, [tssb.get(s) for s in sym6], L191.PROM_PAD,
                                     lambda *_: None)
        cols.append(PM[[int(np.where(pt == v)[0][0]) for v in grid6]][:, gi6].mean(0))
    Ch = np.column_stack(cols)
    Cc = Ch - Ch.mean(0)
    s2 = np.linalg.svd(Cc, compute_uv=False)
    shares["chip_9tracks"] = float(s2[0] ** 2 / np.sum(s2 ** 2))
    say(f"       ChIP, 9 A549 tracks: first component {shares['chip_9tracks']:.2%} of variance")
    mx = max(shares.values())
    G.add("Q6", bool(mx > 0.20), stat=mx, requires=("Q1",),
          if_true=lambda: f"Q6 PASS -- at least one other dataset carries a dominant component "
                          f"({mx:.1%}); the pattern is general",
          if_false=lambda: f"Q6 FAIL -- the largest first component elsewhere is {mx:.1%}")
    res["other_datasets"] = shares

    # ---------------------------------------------------------------- Q7
    say("Q7 ARE THE ANNOTATION BLOCKS ONE THING OR MANY?")
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    T = {str(x["name"]).upper(): x for x in tab}
    nb = json.load(gzip.open("colab/data/net_bundle.json.gz"))
    nn = nb["names"]; nidx = {n.upper(): i for i, n in enumerate(nn)}
    from collections import Counter
    ppi = Counter()
    for a, b in nb["ppi"]:
        ppi[int(a)] += 1; ppi[int(b)] += 1
    ind, outd = Counter(), Counter()
    for s_, t_, _ in nb["reg"]:
        outd[int(s_)] += 1; ind[int(t_)] += 1
    gs = [s for s in sym6[gi6] if s in T]
    F = np.array([[np.log1p(ppi.get(nidx.get(s, -1), 0)),
                   np.log1p(ind.get(nidx.get(s, -1), 0)),
                   np.log1p(outd.get(nidx.get(s, -1), 0)),
                   np.log1p(float(T[s].get("npath") or 0)),
                   np.log1p(float(T[s].get("ndis") or 0)),
                   float(T[s].get("loeuf") or 1.0),
                   float(T[s].get("tf") or 0),
                   np.log1p(float(T[s].get("enh") or 0)),
                   np.log1p(float(T[s].get("pubs") or 0))] for s in gs])
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    s3 = np.linalg.svd(Fz - Fz.mean(0), compute_uv=False)
    v1 = float(s3[0] ** 2 / np.sum(s3 ** 2))
    ldg = np.linalg.svd(Fz - Fz.mean(0), full_matrices=False)[2][0]
    nmz = ["ppi_deg", "in_deg", "out_deg", "n_path", "n_dis", "loeuf", "is_tf", "n_enh", "pubs"]
    say(f"     nine annotation columns over {len(gs):,} genes")
    say(f"     first component explains {v1:.2%} of joint variance")
    say(f"     its loadings: " + "  ".join(f"{n} {l:+.2f}" for n, l in zip(nmz, ldg)))
    G.add("Q7", bool(v1 < 0.50), stat=v1, requires=("Q1",),
          if_true=lambda: f"Q7 PASS -- the first component explains {v1:.1%}, so the annotation "
                          f"blocks are not one latent thing",
          if_false=lambda: f"Q7 FAIL -- one component explains {v1:.1%} of the annotation blocks. "
                           f"Loop 213's ten-block stack is measuring fewer things than it counts")
    res["annotation_pc1"] = {"var": v1, "loadings": dict(zip(nmz, ldg.tolist()))}

    say("Q8 WHAT THIS CANNOT SHOW")
    say("     Biotype is assigned by NAME PATTERN, not by GENCODE. The rule will miss unnamed")
    say("     ENSG entries entirely -- and unnamed entries are 4 of the top 10 loadings -- so Q2")
    say("     UNDERSTATES the enrichment and Q4 removes less than the full class.")
    say("     Naming a component's carriers is not proving a mechanism. Library preparation is")
    say("     the best explanation for these biotypes moving together, but a genuine biological")
    say("     difference in snoRNA processing between two cultures would look identical from")
    say("     here, and ENCODE publishes no per-replicate batch metadata for this series.")
    say("     If Q4 fails, the honest reading is that the biotypes MARK the artefact rather than")
    say("     constitute it, and the component is broader than the classes that flag it.")

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
