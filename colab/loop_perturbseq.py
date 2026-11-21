"""Loop 208. Measured regulatory gains: is 0.91 reachable, and was "not fetchable" wrong?

TWO CLAIMS UNDER TEST, AND THEY ARE INDEPENDENT.

FIRST, A CLAIM IN THIS PROJECT'S OWN ARCHITECTURE NOTE. NOTES_rem_cell.md section 6 lists the
612,133 regulatory edge gains under "Blocked and not fetchable", described as "a curation and
biology gap, not a download". That claim decides the whole staged plan: if the gains cannot be
obtained, the propagation chain cannot be built and REM-Cell is finished. Genome-scale Perturb-seq
is a direct measurement of exactly that quantity -- knock a gene down, read the transcriptome, and
the change IS the gain -- and it is public. A2 tests the claim by counting.

SECOND, THE QUESTION THE PLAN ACTUALLY TURNS ON. Loop 206 measured that a set point must reach
Pearson r >= 0.9081 before relaxation beats persistence, that nine ChIP/DNase tracks measured in
the same cells reach 0.2932, and that computed thermodynamic occupancy reaches -0.0133. Measured
gains are the one input class never tried. A4 asks whether they close the gap.

THE DATA. Replogle et al. 2022, genome-scale Perturb-seq (figshare 20029387):
    K562 gwps pseudobulk   11,258 CRISPRi perturbations x 8,248 readout genes, z-normalised
                           against non-targeting controls. 9,867 distinct perturbed genes, 8,246
                           distinct readouts, 7,684 genes appearing on both axes.
    RPE1 pseudobulk        the same protocol in a second, non-cancer cell line.
    That is 81.4 million measured perturbation-response values in 470 MB.

WHY THE SECOND CELL LINE MATTERS AND IS NOT A BONUS. The set-point target is A549. The gains are
K562 and RPE1. Nothing here is measured in A549, so A4 is a CROSS-CELL-TYPE transfer and could fail
for that reason alone rather than because gains are useless. A5 bounds it directly: if a gain does
not even reproduce between K562 and RPE1, then no cross-cell-type transfer can work and A4's
failure would carry no information about gains as such. A5 therefore runs whether or not A4 passes.

SIGN IS NOT ASSUMED ANYWHERE. Dexamethasone ACTIVATES NR3C1; Perturb-seq KNOCKS IT DOWN. The two
signatures should oppose, but this project has scored a control backwards before by writing
`real > control` on a negative association (loop 199's Q5), so every comparison below is on
MAGNITUDE via gate_guard.weakened_by, and the sign is reported rather than gated.

PREDECLARED, BEFORE ANY NUMBER.

  A1 IS THE DATA WHAT IT CLAIMS TO BE?
     Gate: PASS iff the matrix is 11,258 x 8,248 with 9,867 distinct perturbed genes, AND the
     GATA1 positive control behaves -- GATA1 knockdown in K562 must raise myeloid markers, which
     is the best-known result in this cell line and is not a fit to anything here.
     FAIL means the file is not what the paper describes and nothing below may be read.

  A2 HOW MANY OF THE 612,133 GAINS DOES THIS SUPPLY?
     Count the network's directed edges whose regulator was perturbed and whose target was read.
     Gate: PASS iff more than 10% of the 612,133 edges now carry a measured value.
     A PASS refutes the "not fetchable" line in this project's own architecture note, and the note
     must be corrected rather than quietly left standing.

  A3 THE DIRECT TEST, and it is the sharpest one available.
     Dexamethasone acts through NR3C1. NR3C1 was perturbed in this dataset. So the measured NR3C1
     knockdown signature is a direct, independently-measured prediction of which genes the A549
     course should move.
     Gate: PASS iff |Pearson r| between the NR3C1 signature and the A549 plateau exceeds the 95th
     percentile of 1,000 random perturbation rows drawn from the same matrix. The null is other
     real perturbations, not shuffled values, so it controls for the matrix's own structure.

  A4 DO MEASURED GAINS CLOSE THE GAP?
     Set point predicted from measured gains, gene-held-out, scored on loop 206's harness.
     Gate: PASS iff r >= 0.9081, loop 206's measured crossover.
     Secondary bars reported and required: beat the nine-track arm at 0.2932, and beat the fame
     floor. This gate is the whole point and it is allowed to fail.

  A5 DO GAINS TRANSFER BETWEEN CELL TYPES AT ALL?
     For genes perturbed in both K562 and RPE1, correlate the two signatures.
     Gate: PASS iff the median cross-cell-type correlation exceeds the median correlation between
     DIFFERENT genes' signatures within K562. That within-line null is the right one: it asks
     whether a gain is more like itself in another cell type than it is like a different gain in
     the same one.
     A FAIL bounds A4 from below and must be reported alongside it, because it would mean A4 could
     not have succeeded regardless of how good gains are.

  A6 IS ANY OF IT FAME?
     Gate: PASS iff the gain-based arm beats publication count on the same folds.

  A7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, sys, time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from gate_guard import Gates, weakened_by

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
PS = SP / "perturbseq"
K562 = PS / "K562_gwps_normalized_bulk_01.h5ad"
RPE1 = PS / "rpe1_normalized_bulk_01.h5ad"
OUT = "outputs/loop_perturbseq.json"

T_MIN, REPS, MIN_TPM, MIN_PLATEAU = 30.0, (1, 2, 3), 1.0, 0.5
PROM_PAD = L191.PROM_PAD
N_TRAIN, SEED = 6, 208208
R_REQ, R_NINE = 0.9081, 0.2932
MYELOID = ["TYROBP", "LST1", "CSF3R", "CTSC", "CFD", "SAT1", "FCN1", "LYZ"]

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def load_h5ad(path):
    f = h5py.File(path, "r")
    gt = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs/gene_transcript"][:]]
    pert = np.array([g.split("_")[1] for g in gt])
    cats = [x.decode() if isinstance(x, bytes) else str(x)
            for x in f["var/__categories/gene_name"][:]]
    readout = np.array([cats[c] for c in f["var/gene_name"][:]])
    return f, f["X"], pert, readout


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def ridge(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    return B @ np.linalg.solve(A.T @ A + R, A.T @ ytr)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "perturb-seq gains"}
    say("=" * 104)
    say("LOOP 208 -- MEASURED REGULATORY GAINS: IS 0.91 REACHABLE, AND WAS 'NOT FETCHABLE' WRONG?")
    say("=" * 104)

    # ---------------------------------------------------------------- A1
    say("A1 IS THE DATA WHAT IT CLAIMS TO BE?")
    fk, Xk, pert, readout = load_h5ad(K562)
    say(f"     K562 gwps matrix {Xk.shape}   perturbed genes {len(set(pert)):,}   "
        f"readouts {len(set(readout)):,}   on both axes {len(set(pert)&set(readout)):,}")
    ridx = {g: i for i, g in enumerate(readout)}
    g1 = np.where(pert == "GATA1")[0]
    v = Xk[int(g1[0]), :] if len(g1) else np.zeros(Xk.shape[1])
    my = [ridx[m] for m in MYELOID if m in ridx]
    my_med = float(np.median(v[my])) if my else float("nan")
    all_med = float(np.median(v))
    say(f"     GATA1 knockdown: median z at {len(my)} myeloid markers {my_med:+.3f}   "
        f"all genes {all_med:+.3f}")
    say(f"       top effects " + ", ".join(
        f"{readout[i]} {v[i]:+.1f}" for i in np.argsort(-v)[:5]))
    ok1 = (Xk.shape == (11258, 8248) and len(set(pert)) == 9867 and my_med > all_med + 1.0)
    G.add("A1", ok1, stat=my_med,
          if_true=lambda: f"A1 PASS -- shape and counts match the paper, and GATA1 knockdown "
                          f"raises myeloid markers to {my_med:+.2f} against {all_med:+.2f} overall",
          if_false=lambda: f"A1 FAIL -- shape {Xk.shape}, perturbed {len(set(pert))}, "
                           f"myeloid {my_med:+.3f} vs {all_med:+.3f}")

    # ---------------------------------------------------------------- A2
    say("A2 HOW MANY OF THE 612,133 GAINS DOES THIS SUPPLY?")
    nb = json.load(gzip.open("colab/data/net_bundle.json.gz"))
    names = nb["names"]
    P, R = set(pert), set(readout)
    covered = sum(1 for s, t, _ in nb["reg"]
                  if names[s].upper() in P and names[t].upper() in R)
    total = len(nb["reg"])
    frac = covered / total
    say(f"     network edges {total:,}")
    say(f"     regulator perturbed AND target read out: {covered:,}  = {frac:.2%}")
    say(f"     the matrix itself holds {len(set(pert))*len(set(readout)):,} measured "
        f"perturbation-response values, in 470 MB")
    G.add("A2", bool(frac > 0.10), stat=frac,
          if_true=lambda: f"A2 PASS -- {frac:.1%} of the 612,133 gains now carry a MEASURED value. "
                          f"NOTES_rem_cell.md lists these under 'blocked and not fetchable, a "
                          f"curation and biology gap, not a download'. That line is wrong and the "
                          f"note must be corrected",
          if_false=lambda: f"A2 FAIL -- only {frac:.1%} covered; the note's line stands")
    res["coverage"] = {"edges": total, "covered": covered, "fraction": frac,
                       "matrix_values": len(set(pert)) * len(set(readout))}

    # ---------------------------------------------------------------- harness
    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm = z["tpm"]
    ensg = np.array([str(g).split(".")[0] for g in z["genes"]])
    mins, reps = z["mins"].astype(int), z["reps"].astype(int)
    allt = sorted(set(mins.tolist()))
    comp = {t: set(reps[mins == t].tolist()) for t in allt}
    grid = np.array([t for t in allt if set(REPS) <= comp[t] and t >= T_MIN], dtype=float)
    M, _ = L191.rep_trajectories(tpm, mins, reps, REPS, grid)
    e2s = L191.ensg_to_symbol(lambda *_: None)
    sym = np.array([e2s.get(g, "") for g in ensg])
    base = tpm[(mins == int(grid[0])) & np.isin(reps, REPS)].mean(0)
    pl = M[-3:].mean(0)
    resp = (base >= MIN_TPM) & (np.abs(pl) >= MIN_PLATEAU)
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    tssb, pubs = {}, {}
    for g in tab:
        pubs[str(g["name"]).upper()] = float(g.get("pubs") or 0)
    for line in open(SP / "_tss_hg38.bed"):
        q = line.split()
        if len(q) >= 4 and q[3].startswith("G"):
            i = int(q[3][1:])
            if i < len(tab):
                tssb[str(tab[i]["name"]).upper()] = (q[0], int(q[2]))
    pt, PM = L191.promoter_track("DNase", [tssb.get(s) for s in sym], PROM_PAD, lambda *_: None)
    A = PM[[int(np.where(pt == t)[0][0]) for t in grid]]
    keep = resp & (A > 0).any(0)
    gi = np.where(keep)[0]
    S_true = pl[gi]
    tgt = np.array([sym[i] for i in gi])
    say(f"     A549 target set: {len(gi):,} genes")

    # ---------------------------------------------------------------- A3
    say("A3 THE DIRECT TEST: does the measured NR3C1 knockdown signature predict the A549 plateau?")
    n_row = np.where(pert == "NR3C1")[0]
    have = np.array([t in ridx for t in tgt])
    cols = np.array([ridx[t] for t in tgt if t in ridx])
    say(f"     A549 genes with a Perturb-seq readout: {int(have.sum()):,} of {len(gi):,} "
        f"({have.mean():.1%})")
    if not len(n_row):
        G.add("A3", None, void_if=True, void_reason="NR3C1 was not perturbed in this dataset")
        r_nr = float("nan")
    else:
        sig = Xk[int(n_row[0]), :][cols]
        r_nr = pear(sig, S_true[have])
        rng = np.random.default_rng(SEED)
        pick = rng.choice(Xk.shape[0], 1000, replace=False)
        null = np.array([abs(pear(Xk[int(p), :][cols], S_true[have])) for p in pick])
        p95 = float(np.percentile(null, 95))
        say(f"     NR3C1 signature vs A549 plateau   r {r_nr:+.4f}   |r| {abs(r_nr):.4f}")
        say(f"     null of 1,000 OTHER real perturbations: median |r| {np.median(null):.4f}   "
            f"95th pct {p95:.4f}   max {null.max():.4f}")
        say(f"     sign: dexamethasone ACTIVATES NR3C1 and Perturb-seq KNOCKS IT DOWN, so an "
            f"opposing sign is the expected direction; it is reported, not gated")
        G.add("A3", bool(abs(r_nr) > p95), stat=abs(r_nr), requires=("A1",),
              if_true=lambda: f"A3 PASS -- |r| {abs(r_nr):.4f} exceeds the 95th percentile "
                              f"{p95:.4f} of other real perturbations",
              if_false=lambda: f"A3 FAIL -- |r| {abs(r_nr):.4f} does not exceed {p95:.4f}. The "
                               f"drug's own receptor, knocked down and measured directly, does not "
                               f"stand out among 1,000 unrelated perturbations")
        res["nr3c1"] = {"r": r_nr, "null_median": float(np.median(null)), "null_p95": p95,
                        "n_genes": int(have.sum())}

    # ---------------------------------------------------------------- A4
    say("A4 DO MEASURED GAINS CLOSE THE GAP?")
    tf_rows = [i for i, p in enumerate(pert) if p in {"NR3C1", "EP300", "JUN", "JUNB", "CEBPB",
                                                      "FOSL2", "CTCF", "RAD21", "MYC", "SPI1"}]
    say(f"     using {len(tf_rows)} perturbation signatures as per-gene features")
    F = np.column_stack([Xk[int(r), :][cols] for r in tf_rows])
    y = S_true[have]
    order = np.random.default_rng(SEED).permutation(len(y))
    folds = np.array_split(order, 5)

    def cv(Feat, label):
        Sp = np.zeros(len(y))
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Sp[te] = ridge(Feat[tr], y[tr], Feat[te])
        rr = pear(Sp, y)
        say(f"       {label:<34} r {rr:+.4f}")
        return rr
    r_gain = cv(F, f"measured gains ({len(tf_rows)} perturbations)")
    # NON-FINITE GUARD, and the reason this loop was rerun. The first run drew 200 perturbations
    # at random and the arm returned nan: 17.8% of rows in this matrix carry non-finite values
    # (measured -- 89 of 500 sampled rows, typically 1-3 cells of 8,248). A single nan propagates
    # through the ridge and through the correlation, and max() over a nan is not defined behaviour,
    # so the gate scored on the surviving arm by luck rather than by design. This is gate_guard's
    # finite() case, committed by a loop written after that module. Rows are now screened over the
    # columns actually used, and the count of what was dropped is reported.
    rng2 = np.random.default_rng(SEED + 1)
    cand = rng2.permutation(Xk.shape[0])
    picked, dropped = [], 0
    for p in cand:
        if len(picked) >= 200:
            break
        v = Xk[int(p), :][cols]
        if np.isfinite(v).all():
            picked.append(int(p))
        else:
            dropped += 1
    say(f"       screened for non-finite rows: kept {len(picked)}, dropped {dropped} "
        f"({dropped/(len(picked)+dropped):.1%} of those examined)")
    Fbig = np.column_stack([Xk[p, :][cols] for p in picked])
    r_big = cv(Fbig, f"{len(picked)} random perturbations")
    pv = np.log1p(np.array([pubs.get(t, 0.0) for t in tgt[have]])).reshape(-1, 1)
    r_fame = cv(pv, "publication count (fame)")
    finite_arms = [x for x in (r_gain, r_big) if np.isfinite(x)]
    best = max(abs(x) for x in finite_arms) if finite_arms else float("nan")
    if len(finite_arms) < 2:
        say(f"     WARNING: {2-len(finite_arms)} of 2 arms returned a non-finite r and are "
            f"excluded from the maximum")
    say(f"     best gain-based set point |r| {best:.4f}")
    say(f"       requirement (loop 206 Y3)        {R_REQ:.4f}")
    say(f"       nine same-cell tracks (206 Y4)   {R_NINE:.4f}")
    G.add("A4", bool(best >= R_REQ) if np.isfinite(best) else None, stat=best, requires=("A1",),
          if_true=lambda: f"A4 PASS -- measured gains reach {best:.4f}, clearing the {R_REQ:.4f} "
                          f"crossover. 0.91 is reachable",
          if_false=lambda: f"A4 FAIL -- measured gains reach {best:.4f} against a requirement of "
                           f"{R_REQ:.4f}. They "
                           f"{'DO' if best > R_NINE else 'do NOT'} beat the nine same-cell tracks "
                           f"({R_NINE:.4f})")
    res["setpoint"] = {"r_gains": r_gain, "r_random200": r_big, "r_fame": r_fame,
                       "best": best, "required": R_REQ, "nine_track": R_NINE}

    # ---------------------------------------------------------------- A5
    say("A5 DO GAINS TRANSFER BETWEEN CELL TYPES AT ALL?")
    fr, Xr, pert_r, readout_r = load_h5ad(RPE1)
    say(f"     RPE1 matrix {Xr.shape}   perturbed {len(set(pert_r)):,}")
    shared_r = [g for g in set(readout) & set(readout_r)]
    ck = {g: i for i, g in enumerate(readout)}
    cr = {g: i for i, g in enumerate(readout_r)}
    ck_i = np.array([ck[g] for g in shared_r]); cr_i = np.array([cr[g] for g in shared_r])
    both = sorted(set(pert) & set(pert_r))
    rng3 = np.random.default_rng(SEED + 2)
    samp = [both[i] for i in rng3.choice(len(both), min(300, len(both)), replace=False)]
    pk = {g: i for i, g in enumerate(pert)}
    pr = {g: i for i, g in enumerate(pert_r)}
    same, diff = [], []
    for g in samp:
        a = Xk[int(pk[g]), :][ck_i]
        b = Xr[int(pr[g]), :][cr_i]
        same.append(pear(a, b))
        h = samp[int(rng3.integers(len(samp)))]
        if h != g:
            diff.append(pear(a, Xk[int(pk[h]), :][ck_i]))
    same, diff = np.array(same), np.array(diff)
    say(f"     genes perturbed in BOTH lines: {len(both):,}   shared readouts {len(shared_r):,}")
    say(f"     SAME gene, K562 vs RPE1        median r {np.median(same):+.4f}   "
        f"n {len(same)}")
    say(f"     DIFFERENT genes, within K562   median r {np.median(diff):+.4f}   "
        f"n {len(diff)}   (the null)")
    cmp5 = weakened_by(float(np.median(same)), float(np.median(diff)))
    G.add("A5", bool(cmp5["weakened"]), stat=float(np.median(same)), requires=("A1",),
          if_true=lambda: f"A5 PASS -- a gain resembles itself across cell types "
                          f"({np.median(same):+.4f}) more than it resembles a different gain in "
                          f"the same one ({np.median(diff):+.4f}), so cross-cell-type transfer is "
                          f"not hopeless in principle",
          if_false=lambda: f"A5 FAIL -- {np.median(same):+.4f} against {np.median(diff):+.4f}. A "
                           f"gain does NOT transfer between cell types, so A4 could not have "
                           f"succeeded regardless of how good gains are, and its failure carries "
                           f"no information about gains as such")
    res["transfer"] = {"n_both": len(both), "median_same": float(np.median(same)),
                       "median_diff": float(np.median(diff)), "compare": cmp5}

    # ---------------------------------------------------------------- A6
    say("A6 IS ANY OF IT FAME?")
    G.add("A6", bool(best > abs(r_fame)), stat=r_fame, requires=("A4",),
          if_true=lambda: f"A6 PASS -- {best:.4f} beats publication count {abs(r_fame):.4f}",
          if_false=lambda: f"A6 FAIL -- fame {abs(r_fame):.4f} is not beaten")

    say("A7 WHAT THIS CANNOT SHOW")
    say("     The gains are K562 and RPE1; the target is A549. A5 bounds the transfer but does")
    say("     not remove it, and no Perturb-seq of A549 at this scale exists on this disk.")
    say("     CRISPRi knockdown is not the inverse of ligand activation. Knocking down a receptor")
    say("     and activating it with its ligand differ in magnitude, kinetics and in which")
    say("     cofactors are present, so A3 is a lower bound on what the signature could carry.")
    say("     Pseudobulk averages over cells, so anything that depends on cell state is gone.")
    say("     A measured gain is a STEADY-STATE response about 6 days after knockdown. The A549")
    say("     plateau is 12 hours after a drug. These are different timescales and nothing here")
    say("     corrects for that.")
    say("     A2 counts an edge as covered when the regulator was perturbed and the target read.")
    say("     That is coverage of the EDGE, not proof the measured value is that edge's gain --")
    say("     a knockdown effect propagates through the network and is not purely direct.")

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
