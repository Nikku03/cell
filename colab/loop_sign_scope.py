"""Loop 243. The sign, scored where it makes a claim, and tested where its prediction reverses.

WHY THIS LOOP EXISTS: LOOP 242's Q3 ASKED A GLOBAL QUESTION OF A LOCAL CLAIM. Q3 compared a signed
graph against an unsigned one by correlation across all 8,175 screened genes and got +0.0001. That
number is not evidence about the sign, because of this, measured after the run:

    genes scored per perturbation                     8,175
    signed targets the graph speaks about             median 3   (mean 8.1, p90 23)
    fraction of the scored profile the sign touches   0.037%
    share of the response's energy on those targets   0.146%

A correlation over 8,175 genes cannot move on 3 of them. The gate was arithmetically incapable of
detecting what it was built to detect, which is the same family as loop 119's inert null and loop
240's variance-defined gene set. Loop 242's Q2 had ALREADY shown the sign works where it speaks
(+0.0072 gap, activating arm -3.0 se); Q3 then scored it somewhere else and called it worthless.

AND THE OBVIOUS REPAIR DOES NOT WORK EITHER, which is why this is not simply loop 242 rerun on a
subset. Restricting the correlation to a perturbation's signed targets makes the UNSIGNED arm
degenerate: every target gets the same predicted value, the prediction has zero variance, and its
correlation is undefined. Swapping one vacuous comparison for another is not a repair. So the
instrument changes: the sign is tested as a TWO-GROUP CONTRAST WITHIN a single perturbation --
that perturbation's activated targets against its own inhibited targets. Every perturbation-level
confound (how strong the knockdown was, how many cells, batch) is common to both groups and
cancels exactly, which is the loop 229 pairing argument applied one level down.

THE PART THAT IS NEW EVIDENCE RATHER THAN A REPAIR. The matched A549 experiments are ten
transcription factors OVER-EXPRESSED, with a matched empty-vector control, across five timepoints
shared with that control. Perturb-seq is loss of function; this is gain of function. So the signed
graph makes a prediction whose sign is REVERSED:

    knock down an activator  ->  its targets fall
    OVER-EXPRESS an activator ->  its targets RISE

Different cell line, different laboratory, different assay, opposite perturbation direction, and a
prediction that cannot be satisfied by any generic "perturbed genes and their neighbours move
together" effect -- because such an effect has no sign to flip. R5 is the strongest test of the
signed graph available in this project, and it is only possible because the matched experiments
exist.

OmniPath signed coverage of the ten over-expressed factors, counted before the run:
    CEBPB 83 act / 20 inh    FOXO1 63/26    FOXO3 62/12    OCT4 33/16    CEBPD 17/1
    FOSL2 16/1   KLF6 12/3   KLF9 2/3       KLF15 1/2      TFCP2L1 0/0
The last three cannot carry the test and are reported as excluded rather than silently dropped.

PREDECLARED, BEFORE ANY NUMBER.

  R1 IS THE LOCAL TEST EVEN AVAILABLE?
     Gate: PASS iff at least 200 K562 perturbations have 3 or more activating AND 3 or more
     inhibiting measured targets, which is what a within-perturbation contrast requires.
     Everything else requires this.

  R2 DOES THE SIGN SEPARATE UP FROM DOWN WITHIN ONE PERTURBATION?
     For each qualifying perturbation, the mean change of its ACTIVATED targets minus the mean
     change of its own INHIBITED targets, paired across perturbations.
     Gate: PASS iff that difference is NEGATIVE -- activated targets fall further than inhibited
     ones when the regulator is removed -- by at least 3 standard errors.

  R3 CONTROL: LABELS PERMUTED INSIDE EACH PERTURBATION.      -- requires R2
     The activate/inhibit labels shuffled among that perturbation's OWN targets, so the target set
     and its size are identical and only the assignment changes.
     Gate: PASS iff the effect collapses to under 25% of its true magnitude.

  R4 CONTROL: ARROWS REVERSED.      -- requires R1
     A gene's REGULATORS in place of its targets. Loop 242's Q6 found 102% of the global effect
     surviving reversal, so this asks whether the local effect is any more directional.
     Gate: PASS iff the effect falls by at least half.

  R5 THE MATCHED EXPERIMENTS: DOES THE PREDICTION REVERSE UNDER OVER-EXPRESSION?    -- requires R2
     The same contrast on the A549 over-expression arms, where the prediction flips sign.
     Gate: PASS iff the A549 contrast is POSITIVE (activated targets rise) while R2's K562
     contrast is NEGATIVE, and the A549 side is at least 2 standard errors from zero. A sign
     agreement rather than a reversal would mean both are reading a direction-free artefact.

  R6 DO THE MATCHED EXPERIMENTS HELP TRAINING?      -- requires R1
     Predict a held-out over-expression arm's whole change profile, leaving out that arm entirely
     (all five of its timepoints). Trained on the K562 signed model alone, against the same model
     plus the other nine arms.
     Three predictors: the factor's own K562 knockdown profile NEGATED, the mean of the other
     nine arms, and both combined. The gate is the combination against the BEST SINGLE of the two,
     not against K562 alone -- every A549 arm shares a dexamethasone timecourse, so "the other
     arms resemble this one" is shared batch structure and beating K562 with it would prove
     nothing about whether the matched experiments ADD information.
     Gate: PASS iff the combination beats the better single source by at least 0.02, paired over
     the held-out (factor, timepoint) units.

  R7 WHAT THIS CANNOT SHOW -- written before the run.
     Seven usable factors is a small number. R5's power comes from the target genes inside them,
     not from the count of factors, so a single unusual factor can move it; the per-factor numbers
     are reported for that reason.
     Over-expression is not the inverse of knockdown. A transcription factor pushed far above its
     normal level can act on sites it never normally occupies, so a failed reversal has a
     biological reading as well as a graph-quality reading.
     The A549 arms sit in a dexamethasone timecourse. Every arm shares that treatment, and the
     control is matched timepoint by timepoint, so it divides out of the contrast -- but it is not
     absent from the biology.
     OmniPath signs are a consensus across sources and cell types. This tests the consensus sign.
"""
import os, sys, json, time, csv, gzip, re, collections, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_sign_scope.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
K562 = SCR / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
OP = SCR / "reg" / "op_2022.tsv"
MATCH = SCR / "matched"
E2S = SCR / "ens2sym.npz"

SEED = 243243
MIN_EACH = 3
R1_MIN, R2_SE, R3_MAX, R4_DROP, R5_SE, R6_BAR = 200, 3.0, 0.25, 0.50, 2.0, 0.02
OE_TFS = ["CEBPB", "CEBPD", "FOSL2", "FOXO1", "FOXO3", "KLF15", "KLF6", "KLF9", "OCT4", "TFCP2L1"]
ALIAS = {"OCT4": "POU5F1"}

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def paired(d):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "signed edges scored within perturbation, and reversed under over-expression"}
    say("=" * 104)
    say("LOOP 243 -- THE SIGN, SCORED WHERE IT MAKES A CLAIM")
    say("=" * 104)
    say("     Loop 242's Q3 compared signed against unsigned over all 8,175 genes and got +0.0001.")
    say("     The signed graph speaks about a MEDIAN OF 3 of those genes -- 0.037% of the profile,")
    say("     carrying 0.146% of the response's energy. That gate could not have detected what it")
    say("     was built to detect. Restricting the correlation instead makes the unsigned arm")
    say("     degenerate (one value for every target, zero variance), so the instrument changes to")
    say("     a two-group contrast WITHIN each perturbation, where every perturbation-level")
    say("     confound cancels exactly.")

    # ---------------------------------------------------------------- graphs
    act, inh = collections.defaultdict(set), collections.defaultdict(set)
    with open(OP) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s, t = r["source_genesymbol"], r["target_genesymbol"]
            if not s or not t or s == t or r["is_directed"] != "1": continue
            if r["is_stimulation"] == "1": act[s].add(t)
            if r["is_inhibition"] == "1": inh[s].add(t)
    ract, rinh = collections.defaultdict(set), collections.defaultdict(set)
    for s, ts in act.items():
        for t in ts: ract[t].add(s)
    for s, ts in inh.items():
        for t in ts: rinh[t].add(s)
    say(f"     OmniPath: {sum(len(v) for v in act.values()):,} activating, "
        f"{sum(len(v) for v in inh.values()):,} inhibiting directed edges")

    # ---------------------------------------------------------------- K562
    import h5py
    f = h5py.File(K562, "r")
    cats = f["var"]["__categories"]["gene_name"][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
    gname = cats[f["var"]["gene_name"][:]]
    k = f["obs"].attrs.get("_index", "_index")
    k = k.decode() if isinstance(k, bytes) else k
    obs = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["obs"][k][:]])
    pert = np.array([o.split("_")[1] for o in obs])
    X = f["X"][:]
    f.close()
    gcol = np.isfinite(X).all(0)
    rrow = np.isfinite(X[:, gcol]).all(1)
    MG = gname[gcol]; X = X[np.ix_(rrow, gcol)]; pert = pert[rrow]
    mpos = {g: i for i, g in enumerate(MG)}
    rows = collections.defaultdict(list)
    for i, g in enumerate(pert): rows[g].append(i)
    say(f"     K562: {X.shape[0]:,} screened rows x {X.shape[1]:,} genes finite everywhere")

    def contrast(Y, rowidx, gsym, A, I, pos, permute=False):
        """Mean change of a regulator's ACTIVATED targets minus its own INHIBITED targets.
        Both groups come from the same row, so anything that scales the whole row cancels."""
        a = [pos[t] for t in A.get(gsym, ()) if t in pos]
        b = [pos[t] for t in I.get(gsym, ()) if t in pos]
        if len(a) < MIN_EACH or len(b) < MIN_EACH: return None, len(a), len(b)
        if permute:
            both = np.array(a + b); rng.shuffle(both)
            a, b = both[:len(a)], both[len(a):]
        y = Y[rowidx]
        return float(np.mean(y[a]) - np.mean(y[b])), len(a), len(b)

    # ---------------------------------------------------------------- R1
    say("R1 IS THE LOCAL TEST EVEN AVAILABLE?")
    qual = [g for g in rows
            if len([t for t in act.get(g, ()) if t in mpos]) >= MIN_EACH
            and len([t for t in inh.get(g, ()) if t in mpos]) >= MIN_EACH]
    say(f"     K562 perturbations with {MIN_EACH}+ activating AND {MIN_EACH}+ inhibiting measured "
        f"targets: {len(qual):,}")
    G.add("R1", bool(len(qual) >= R1_MIN), stat=float(len(qual)),
          if_true=lambda: f"R1 PASS -- {len(qual):,} perturbations can carry a within-perturbation "
                          f"contrast",
          if_false=lambda: f"R1 FAIL -- only {len(qual):,} against a bar of {R1_MIN}")
    res["qualifying"] = len(qual)

    # ---------------------------------------------------------------- R2
    say("R2 DOES THE SIGN SEPARATE UP FROM DOWN WITHIN ONE PERTURBATION?")
    real = [contrast(X, rows[g][0], g, act, inh, mpos)[0] for g in qual]
    m2, se2, z2 = paired(real)
    say(f"     n={sum(1 for x in real if x is not None and np.isfinite(x)):,}  "
        f"activated minus inhibited, within perturbation: {m2:+.4f} +/- {se2:.4f}  ({z2:+.1f} se)")
    say(f"     removing an activator should lower its activated targets MORE than its inhibited")
    say(f"     ones, so the prediction is a NEGATIVE number")
    G.add("R2", bool(m2 < 0 and z2 <= -R2_SE), stat=float(m2), requires=("R1",),
          if_true=lambda: f"R2 PASS -- {m2:+.4f} at {z2:+.1f} se, the predicted direction",
          if_false=lambda: f"R2 FAIL -- {m2:+.4f} at {z2:+.1f} se against a bar of {-R2_SE:.0f} se")
    res["R2"] = {"mean": m2, "se": se2, "z": z2}

    # ---------------------------------------------------------------- R3
    say("R3 CONTROL: LABELS PERMUTED INSIDE EACH PERTURBATION")
    if abs(m2) < 1e-4:
        G.add("R3", False, stat=float(m2), requires=("R2",), void_if=True,
              void_reason=f"the real contrast is {m2:+.4f}; there is nothing to collapse")
    else:
        perm = []
        for _ in range(5):
            perm.append(np.nanmean([contrast(X, rows[g][0], g, act, inh, mpos, permute=True)[0]
                                    for g in qual]))
        mp = float(np.mean(perm))
        f3 = abs(mp) / abs(m2)
        say(f"     labels shuffled among each perturbation's OWN targets, 5 draws: {mp:+.4f} "
            f"against a real {m2:+.4f}  ({f3:.0%} of the magnitude)")
        G.add("R3", bool(f3 <= R3_MAX), stat=float(f3), requires=("R2",),
              if_true=lambda: f"R3 PASS -- collapses to {f3:.0%} when only the labels move",
              if_false=lambda: f"R3 FAIL -- {f3:.0%} survives permuting the labels")
        res["R3"] = {"real": m2, "permuted": mp, "fraction": f3}

    # ---------------------------------------------------------------- R4
    say("R4 CONTROL: ARROWS REVERSED")
    rev = [contrast(X, rows[g][0], g, ract, rinh, mpos)[0] for g in rows
           if len([t for t in ract.get(g, ()) if t in mpos]) >= MIN_EACH
           and len([t for t in rinh.get(g, ()) if t in mpos]) >= MIN_EACH]
    m4, se4, z4 = paired(rev)
    f4 = abs(m4) / abs(m2) if abs(m2) > 1e-9 else float("nan")
    say(f"     a gene's REGULATORS in place of its targets, n={len(rev):,}: {m4:+.4f} +/- {se4:.4f}"
        f"  ({f4:.0%} of the real magnitude)")
    say(f"     loop 242's Q6 found 102% of the GLOBAL effect surviving reversal")
    G.add("R4", bool(np.isfinite(f4) and f4 <= 1 - R4_DROP), stat=float(f4), requires=("R1",),
          if_true=lambda: f"R4 PASS -- reversal removes {1 - f4:.0%}; the local effect is "
                          f"directional where the global one was not",
          if_false=lambda: f"R4 FAIL -- {f4:.0%} survives reversal; even locally the arrows are "
                           f"not load-bearing")
    res["R4"] = {"reversed": m4, "fraction": f4, "n": len(rev)}

    # ---------------------------------------------------------------- matched A549
    say("     loading the matched A549 over-expression arms ...")
    e2s = np.load(E2S, allow_pickle=True)
    ens2sym = dict(zip([str(x) for x in e2s["ensembl"]], [str(x) for x in e2s["symbol"]]))

    def read_oe(path):
        with gzip.open(path, "rt") as fh:
            cols = fh.readline().rstrip("\n").split("\t")[1:]
            idx, vals = [], []
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                idx.append(p[0].split(".")[0])
                vals.append([float(x) if x else np.nan for x in p[1:]])
        tp = [re.search(r"dex\.([0-9a-z]+)\.rep", c).group(1) for c in cols]
        return np.array(idx), np.asarray(vals, np.float32), np.array(tp)

    ci, cv, ct = read_oe(MATCH / "OE_ctrl.txt.gz")
    cmean = {t: cv[:, ct == t].mean(1) for t in set(ct)}
    arms = {}
    for tf in OE_TFS:
        ai, av, at = read_oe(MATCH / f"OE_{tf}.txt.gz")
        if not np.array_equal(ai, ci):
            say(f"     {tf}: gene order differs from the control, skipped"); continue
        shared = sorted(set(at) & set(ct))
        arms[tf] = {t: av[:, at == t].mean(1) - cmean[t] for t in shared}
    syms = np.array([ens2sym.get(e, "") for e in ci])
    opos = {}
    for i, s in enumerate(syms):
        if s and s not in opos: opos[s] = i
    say(f"     {len(arms)} arms loaded, {len(ci):,} genes, "
        f"{len(opos):,} resolved to a symbol, timepoints "
        f"{sorted(set(ct))}")

    # ---------------------------------------------------------------- R5
    say("R5 THE MATCHED EXPERIMENTS: DOES THE PREDICTION REVERSE UNDER OVER-EXPRESSION?")
    say("     over-expressing an ACTIVATOR should RAISE its targets, so R5 must be POSITIVE while")
    say("     R2 is negative. A sign AGREEMENT would mean both read a direction-free artefact.")
    per_tf, oe_vals, excluded = {}, [], []
    for tf, tps in arms.items():
        g = ALIAS.get(tf, tf)
        na = len([t for t in act.get(g, ()) if t in opos])
        ni = len([t for t in inh.get(g, ()) if t in opos])
        if na < MIN_EACH or ni < MIN_EACH:
            excluded.append(f"{tf} ({na} act / {ni} inh)"); continue
        v = []
        for t, y in tps.items():
            c, _, _ = contrast(y[None, :], 0, g, act, inh, opos)
            if c is not None and np.isfinite(c): v.append(c); oe_vals.append(c)
        per_tf[tf] = (float(np.mean(v)), na, ni, len(v))
    for tf, (v, na, ni, n) in sorted(per_tf.items()):
        say(f"       {tf:<9} {v:+.4f}   ({na} activating / {ni} inhibiting targets, {n} timepoints)")
    if excluded:
        say(f"     excluded for too few signed targets: {', '.join(excluded)}")
    m5, se5, z5 = paired(oe_vals)
    say(f"     pooled over {len(oe_vals)} (factor, timepoint) units: {m5:+.4f} +/- {se5:.4f} "
        f"({z5:+.1f} se)")
    G.add("R5", bool(m5 > 0 and m2 < 0 and z5 >= R5_SE), stat=float(m5), requires=("R2",),
          if_true=lambda: f"R5 PASS -- the prediction REVERSES: {m5:+.4f} under over-expression "
                          f"against {m2:+.4f} under knockdown, {z5:+.1f} se",
          if_false=lambda: f"R5 FAIL -- over-expression gives {m5:+.4f} ({z5:+.1f} se) against a "
                           f"knockdown {m2:+.4f}; "
                           f"{'no reversal' if m5 * m2 >= 0 else 'reversal not above noise'}")
    res["R5"] = {"mean": m5, "se": se5, "z": z5, "per_tf": per_tf, "excluded": excluded,
                 "knockdown": m2}

    # ---------------------------------------------------------------- R6
    say("R6 DO THE MATCHED EXPERIMENTS HELP TRAINING?")
    common = [s for s in opos if s in mpos]
    cpos_o = np.array([opos[s] for s in common]); cpos_k = np.array([mpos[s] for s in common])
    say(f"     {len(common):,} genes measured in BOTH K562 and the A549 arms")
    units, k_scores, j_scores, o_scores = [], [], [], []
    tf_list = sorted(arms)
    for held in tf_list:
        g = ALIAS.get(held, held)
        # K562-only predictor: this factor's own K562 knockdown profile, NEGATED -- the
        # sign-reversal hypothesis used as a predictor rather than tested as a contrast
        kp = -X[rows[g][0]][cpos_k] if g in rows else None
        # joint predictor: the same, plus the mean profile of the OTHER nine arms
        others = [np.nanmean(np.stack(list(arms[o].values())), 0)[cpos_o]
                  for o in tf_list if o != held]
        om = np.nanmean(np.stack(others), 0)
        for t, y in arms[held].items():
            truth = y[cpos_o]
            if kp is None: continue
            a = pear(kp, truth)
            c = pear(om, truth)
            zk = (kp - np.nanmean(kp)) / (np.nanstd(kp) + 1e-9)
            zo = (om - np.nanmean(om)) / (np.nanstd(om) + 1e-9)
            b = pear(zk + zo, truth)
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
                units.append((held, t)); k_scores.append(a); j_scores.append(b); o_scores.append(c)
    say(f"     leave-one-factor-out over {len(set(u[0] for u in units))} factors, "
        f"{len(units)} (factor, timepoint) units")
    ks, os_, js = np.array(k_scores), np.array(o_scores), np.array(j_scores)
    say(f"     K562 knockdown profile alone, negated:   {np.nanmean(ks):+.4f}")
    say(f"     the other nine matched arms alone:       {np.nanmean(os_):+.4f}")
    say(f"     both together:                           {np.nanmean(js):+.4f}")
    say("     the gate is JOINT against the BEST SINGLE arm, not against K562 alone. Every A549")
    say("     arm shares a dexamethasone timecourse, so 'the other arms resemble this one' is")
    say("     shared batch structure; beating K562-alone with it would prove nothing about")
    say("     whether the matched experiments ADD anything.")
    single = "K562" if np.nanmean(ks) >= np.nanmean(os_) else "OTHER ARMS"
    base = ks if np.nanmean(ks) >= np.nanmean(os_) else os_
    m6, se6, z6 = paired(js - base)
    say(f"     best single arm is {single}; joint minus it: {m6:+.4f} +/- {se6:.4f} ({z6:+.1f} se)")
    G.add("R6", bool(m6 >= R6_BAR), stat=float(m6), requires=("R1",),
          if_true=lambda: f"R6 PASS -- combining adds {m6:+.4f} over the best single source "
                          f"({single})",
          if_false=lambda: f"R6 FAIL -- combining adds {m6:+.4f} over {single} alone, against a "
                           f"{R6_BAR} bar")
    res["R6"] = {"k562_only": float(np.nanmean(ks)), "others_only": float(np.nanmean(os_)),
                 "joint": float(np.nanmean(js)), "best_single": single,
                 "delta": m6, "se": se6, "z": z6, "n_units": len(units)}

    say("R7 WHAT THIS CANNOT SHOW")
    say("     Seven usable factors is few. R5's power comes from the target genes inside them,")
    say("     not from the count of factors, so one unusual factor can move it -- which is why")
    say("     the per-factor numbers are printed above rather than only the pooled one.")
    say("     Over-expression is not the inverse of knockdown. A factor pushed far above its")
    say("     normal level can occupy sites it never normally occupies, so a failed reversal has")
    say("     a biological reading as well as a graph-quality one.")
    say("     The A549 arms sit in a dexamethasone timecourse shared with their control and")
    say("     matched timepoint by timepoint, so it divides out of the contrast -- but it is not")
    say("     absent from the biology.")
    say("     OmniPath signs are a consensus across sources and cell types.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
