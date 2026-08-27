"""Loop 256. Stop hand-writing features. Learn the representation, and hold it to the same bar.

WHAT LOOP 255 ACTUALLY SHOWED, AND IT IS NOT WHAT IT LOOKED LIKE. Seven candidate explanations for
the LINCS gene x line interaction all came back null -- mutation, hotspot, copy number, dependency,
paralogues, lineage, volatility. But three of the seven agents independently found the same defect,
and the defect was MINE: a feature that is constant across landmarks within a (gene, line) pair
contributes only an offset, and a within-profile correlation is invariant to offset and scale. Such
a feature set predicts ONE FIXED DIRECTION per held-out line, which is exactly what the residual
already subtracts. Those arms could not have expressed an interaction even if one existed. The
bottleneck was never model capacity. It was that I was choosing the representation by hand.

TWO CHANGES, AND EACH ONE IS A DIRECT ANSWER TO A MEASURED FAILURE.

  1. LEARN the representation instead of specifying it. A gene tower reads the gene's own mean
     response over TRAINING lines; a line tower reads the line's MEASURED properties; a decoder
     sees both at once and can therefore express gene x line interaction, which no hand-written
     per-pair scalar could.

  2. Predict the RAW response, not the residual. Loop 255's structural finding means the residual
     target is blind to any per-line direction, and a per-line direction may be real signal rather
     than a nuisance. Scoring on the raw profile removes that blindness. Loop 252's A3_ADDITIVE
     (gene mean + line mean) reached 0.4477 on exactly this target and is the number to beat.

WHY THE LINE TOWER MAY ONLY USE MEASURED PROPERTIES. Loop 255 proved, numerically, that under
leave-one-cell-line-out the held-out line's one-hot column is identically zero in the training
design, so ridge assigns it a coefficient of exactly zero. Cell-line IDENTITY carries nothing
out-of-sample and never can. So the line tower reads expression (19,193 genes), CRISPR dependency
(17,916 genes) and damaging-mutation burden -- all present for all nine lines, none of them a
label. The principal components are fitted on the full DepMap panel of 1,178 lines, which never
sees a LINCS response, so the basis is unsupervised with respect to the target.

WHAT THIS BORROWS FROM EVO 2 AND WHAT IT DOES NOT. Evo 2 is StripedHyena 2 trained on next-
nucleotide prediction over 9.3 trillion tokens; its published benchmarks are sequence tasks and it
carries no cell-type conditioning. Its ARCHITECTURE does not transfer to predicting a knockdown
response in a named cell line -- that is a different problem with different inputs. What transfers
is the principle: learn representations from data at scale rather than hand-specifying them. This
loop takes the principle and not the formalism, and says so rather than implying more.

THE BAR IS UNCHANGED, AND THIS IS THE POINT. Loop 241's two MLPs LOST to their linear twins on
identical inputs by -0.0155 and -0.0170. Loop 251's four constrained networks lost to ridge by
-0.0088. Loop 242's signed GCN is the only architecture in this project ever to beat its own twin,
by +0.0043, and it still failed its bar. A learned representation is not automatically better than
a hand-written one; H3 is the gate that decides, and it compares the network against a LINEAR MODEL
GIVEN THE IDENTICAL TOWER INPUTS.

PREDECLARED, BEFORE ANY NUMBER.

  H1 DOES THE HARNESS REPRODUCE LOOP 252?
     A3_ADDITIVE recomputed here, held out by cell line, on the raw response.
     Gate: PASS iff it lands within 0.02 of loop 252's 0.4477. Everything requires it; without it
     no comparison to loop 252 is valid.

  H2 DOES THE TWO-TOWER MODEL BEAT THE ADDITIVE BASELINE?      -- requires H1
     Gate: PASS iff it exceeds A3_ADDITIVE by at least 0.02, paired over held-out (gene, line)
     pairs. Loop 252's best line-aware arm managed +0.0406 over gene-mean; this must beat the
     additive model, which is a harder reference.

  H3 DID THE NETWORK HELP, OR THE FEATURES?      -- requires H1. The gate this loop turns on.
     The same tower inputs through a linear model instead of the two-tower network.
     Gate: PASS iff the network exceeds its linear twin by at least 0.01. Every network in this
     project has failed this except loop 242's signed GCN.

  H4 DOES THE LINE TOWER DO ANYTHING?      -- requires H1
     The identical network with the line embedding zeroed at both train and test time, so the
     model keeps its capacity and loses only its knowledge of which cell line it is in.
     Gate: PASS iff the full model exceeds the ablated one by at least 0.01. A FAIL means the
     gain, if any, is the gene tower and the cell line is still decoration.

  H5 IS THE ADVANTAGE LARGER THAN THE SEED NOISE?      -- requires H2, VOID if H2 found nothing
     Three seeds. Gate: PASS iff H2's margin exceeds twice the across-seed standard deviation.
     Loop 225's MLP win was reversed twice by later loops; loop 241's seed sd was 0.0125.

  H6 CONTROL: THE WRONG CELL LINE.      -- requires H4, VOID if H4's margin is under 0.005
     The line tower fed another line's measured properties, everything else identical.
     Gate: PASS iff H4's advantage collapses to under 25%.

  H7 WHAT THIS CANNOT SHOW -- written before the run.
     Nine cell lines is nine points for the line tower to generalise over. A tower that must map
     measured properties to a 32-dimensional embedding from nine examples is in a regime where
     almost any positive result should be suspected of memorising the eight training lines, which
     is what H6 exists to catch and why H5 uses three seeds.
     978 landmark genes, not a transcriptome, and shRNA rather than a clean genetic knockout.
     A learned gene embedding is built from the gene's own response in other lines, so this model
     cannot predict a gene never perturbed anywhere -- it generalises across CONTEXTS, not across
     genes. Loop 252's E5 tested the double holdout; this loop does not.
     Beating the additive baseline would show the interaction is partly learnable. It would not
     say what the interaction IS, and nothing here recovers the mechanism that loops 253 and 255
     failed to find.
"""
import os, sys, json, time, csv, copy, warnings, collections
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates
import lincs_harness as H

import torch
import torch.nn as nn
torch.set_num_threads(4)

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_two_tower.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")

SEED = 256256
SEEDS = [0, 1, 2]
NPC_EXPR, NPC_DEP, GDIM, LDIM, HID = 50, 50, 128, 32, 256
EPOCHS, PATIENCE, LR, BATCH = 60, 8, 1e-3, 256
LOOP252_ADDITIVE = 0.4477
H1_TOL, H2_BAR, H3_BAR, H4_BAR, H5_MULT, H6_MAX = 0.02, 0.02, 0.01, 0.01, 2.0, 0.25

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


class TwoTower(nn.Module):
    """gene tower reads what this knockdown usually does; line tower reads what this cell IS;
    the decoder sees both at once, which is the only way an interaction can be expressed."""
    def __init__(s, gin, lin, nout):
        super().__init__()
        s.g = nn.Sequential(nn.Linear(gin, HID), nn.ReLU(), nn.Linear(HID, GDIM))
        s.l = nn.Sequential(nn.Linear(lin, 64), nn.ReLU(), nn.Linear(64, LDIM))
        s.d = nn.Sequential(nn.Linear(GDIM + LDIM, HID), nn.ReLU(),
                            nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, nout))
    def forward(s, xg, xl, drop_line=False):
        e = s.g(xg)
        z = s.l(xl)
        if drop_line: z = torch.zeros_like(z)
        return s.d(torch.cat([e, z], 1))


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "two-tower factorization on LINCS, held out by cell line"}
    say("=" * 104)
    say("LOOP 256 -- LEARN THE REPRESENTATION, HELD TO THE SAME BAR")
    say("=" * 104)
    say("     Loop 255's seven nulls were partly MY defect: hand-written per-pair scalars cannot")
    say("     express a gene x line interaction, because a within-profile correlation ignores")
    say("     offset and scale. So the representation is learned here, and the target is the RAW")
    say("     response rather than the residual, which loop 255 showed is blind to per-line")
    say("     directions. Loop 252's additive baseline reached 0.4477 on this target.")
    say("     Line identity is NOT an input: loop 255 proved its out-of-sample coefficient is")
    say("     exactly zero. The line tower reads measured properties only.")

    D = H.load()
    Pm, pg, pc, LINES = D["Pm"], D["pg"], D["pc"], D["LINES"]
    NL = D["NL"]
    say(f"     {len(pg):,} (gene, line) pairs, {len(D['genes']):,} genes, {NL} landmarks")

    # ---------------------------------------------------------------- line tower inputs
    lmap = json.load(open(SCR / "lincs" / "line_map.json"))
    ez = np.load(SCR / "depmap_expr_aligned.npz", allow_pickle=True)
    XE = ez["XE"]; el = np.array([str(x) for x in ez["lines"]])
    U, sv, _ = np.linalg.svd(XE - XE.mean(0), full_matrices=False)
    EPC = (U[:, :NPC_EXPR] * sv[:NPC_EXPR])
    ge = np.load(SCR / "depmap" / "gene_effect.npz", allow_pickle=True)
    GE = np.nan_to_num(np.asarray(ge["E"], np.float32)); gl = np.array([str(x) for x in ge["lines"]])
    U2, sv2, _ = np.linalg.svd(GE - GE.mean(0), full_matrices=False)
    DPC = (U2[:, :NPC_DEP] * sv2[:NPC_DEP])
    say(f"     line-tower basis fitted on the FULL DepMap panel -- {XE.shape[0]:,} lines of "
        f"expression, {GE.shape[0]:,} of dependency -- so it never sees a LINCS response")
    burden = {}
    with open(SCR / "depmap" / "OmicsSomaticMutationsMatrixDamaging.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            try: burden[row[0]] = float(np.sum([1.0 for v in row[1:] if v not in ("", "0", "0.0")]))
            except Exception: pass
    ep = {l: int(np.where(el == lmap[l])[0][0]) for l in LINES}
    dp = {l: int(np.where(gl == lmap[l])[0][0]) for l in LINES}
    LF = np.stack([np.concatenate([EPC[ep[l]], DPC[dp[l]],
                                   [np.log1p(burden.get(lmap[l], 0.0))]]) for l in LINES])
    LF = ((LF - LF.mean(0)) / (LF.std(0) + 1e-6)).astype(np.float32)
    li = {l: i for i, l in enumerate(LINES)}
    say(f"     line tower input: {LF.shape[1]} measured dims "
        f"({NPC_EXPR} expression + {NPC_DEP} dependency + 1 mutation burden)")

    gidx = {g: i for i, g in enumerate(D["genes"])}

    def fold_data(hold):
        """Gene tower input is the gene's mean response over TRAINING lines only."""
        tr = pc != hold
        gm = {}
        for g in D["genes"]:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        lmv = Pm[pc == hold].mean(0)
        grand = Pm[tr].mean(0)
        def rows(mask, src=None):
            Xg, Xl, Y, add = [], [], [], []
            for j in np.where(mask)[0]:
                g = pg[j]
                if g not in gm: continue
                c = src if src else pc[j]
                Xg.append(gm[g]); Xl.append(LF[li[c]]); Y.append(Pm[j])
                # DEFECT I: the substitute line feeds the MODEL's inputs only. The standing
                # answer keeps the TRUE line, so the wrong-line control moves ONE thing.
                add.append(gm[g] + Pm[pc == pc[j]].mean(0) - grand)
            return (np.stack(Xg).astype(np.float32), np.stack(Xl).astype(np.float32),
                    np.stack(Y).astype(np.float32), np.stack(add).astype(np.float32))
        return rows, tr

    def score_rows(P, Y):
        return np.array([H.pear(P[i], Y[i]) for i in range(len(Y))])

    def run(seed, drop_line=False, shuffle_line=False, linear=False):
        allsc, alladd = [], []
        for hold in LINES:
            rows, tr = fold_data(hold)
            Xg, Xl, Y, _ = rows(tr)
            src = None
            if shuffle_line:
                src = str(np.random.default_rng(seed + li[hold]).choice(
                    [l for l in LINES if l != hold]))
            Xgt, Xlt, Yt, addt = rows(pc == hold, src=src)
            if linear:
                Z = np.concatenate([Xg, Xl if not drop_line else Xl * 0,
                                    np.ones((len(Xg), 1), np.float32)], 1)
                A = Z.T @ Z + 1e-2 * len(Z) * np.eye(Z.shape[1], dtype=np.float32)
                b = np.linalg.solve(A, Z.T @ Y)
                Zt = np.concatenate([Xgt, Xlt if not drop_line else Xlt * 0,
                                     np.ones((len(Xgt), 1), np.float32)], 1)
                allsc.append(score_rows(Zt @ b, Yt)); alladd.append(score_rows(addt, Yt)); continue
            torch.manual_seed(seed)
            r2 = np.random.default_rng(seed)
            ip = r2.permutation(len(Xg)); nv = max(200, int(0.12 * len(Xg)))
            va, fi = ip[:nv], ip[nv:]
            net = TwoTower(NL, LF.shape[1], NL)
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            tg, tl, ty = (torch.from_numpy(Xg[fi]), torch.from_numpy(Xl[fi]),
                          torch.from_numpy(Y[fi]))
            vg, vl, vy = (torch.from_numpy(Xg[va]), torch.from_numpy(Xl[va]),
                          torch.from_numpy(Y[va]))
            best, bad, bw = 9e9, 0, None
            for ep in range(EPOCHS):
                idx = torch.randperm(len(fi))
                for b0 in range(0, len(fi), BATCH):
                    j = idx[b0:b0 + BATCH]
                    opt.zero_grad()
                    loss = ((net(tg[j], tl[j], drop_line) - ty[j]) ** 2).mean()
                    loss.backward(); opt.step()
                with torch.no_grad():
                    v = float(((net(vg, vl, drop_line) - vy) ** 2).mean())
                if v < best - 1e-7: best, bad, bw = v, 0, copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                    if bad >= PATIENCE: break
            if bw: net.load_state_dict(bw)
            net.eval()
            with torch.no_grad():
                P = net(torch.from_numpy(Xgt), torch.from_numpy(Xlt), drop_line).numpy()
            allsc.append(score_rows(P, Yt)); alladd.append(score_rows(addt, Yt))
        return np.concatenate(allsc), np.concatenate(alladd)

    say("     training: 9 leave-one-cell-line-out folds x 3 seeds ...")
    S, ADD = {}, None
    for sd in SEEDS:
        s_, a_ = run(sd)
        S[sd] = s_; ADD = a_
        say(f"       seed {sd}: two-tower {np.nanmean(s_):.4f}   additive {np.nanmean(a_):.4f}   "
            f"[{time.time() - t0:.0f}s]")
    full = S[SEEDS[0]]
    lin_s, _ = run(SEEDS[0], linear=True)
    abl_s, _ = run(SEEDS[0], drop_line=True)
    say(f"       linear twin (same tower inputs) {np.nanmean(lin_s):.4f}")
    say(f"       line tower ablated              {np.nanmean(abl_s):.4f}")
    res["arms"] = {"two_tower": float(np.mean([np.nanmean(S[s]) for s in SEEDS])),
                   "additive": float(np.nanmean(ADD)), "linear_twin": float(np.nanmean(lin_s)),
                   "line_ablated": float(np.nanmean(abl_s))}

    # ---------------------------------------------------------------- H1
    say("H1 DOES THE HARNESS REPRODUCE LOOP 252?")
    a1 = float(np.nanmean(ADD))
    say(f"     A3_ADDITIVE here {a1:.4f} against loop 252's {LOOP252_ADDITIVE:.4f}")
    G.add("H1", bool(abs(a1 - LOOP252_ADDITIVE) <= H1_TOL), stat=float(a1),
          if_true=lambda: f"H1 PASS -- reproduces to {abs(a1 - LOOP252_ADDITIVE):.4f}",
          if_false=lambda: f"H1 FAIL -- {a1:.4f} here against {LOOP252_ADDITIVE:.4f}")
    res["H1"] = {"additive": a1, "loop252": LOOP252_ADDITIVE}

    # ---------------------------------------------------------------- H2
    say("H2 DOES THE TWO-TOWER MODEL BEAT THE ADDITIVE BASELINE?")
    d2, se2, z2 = H.paired(full, ADD)
    say(f"     two-tower {np.nanmean(full):.4f} vs additive {a1:.4f}")
    say(f"     paired over {int(np.isfinite(full).sum()):,} pairs: {d2:+.4f} +/- {se2:.4f} "
        f"({z2:+.1f} se)")
    G.add("H2", bool(d2 >= H2_BAR), stat=float(d2), requires=("H1",),
          if_true=lambda: f"H2 PASS -- learning the representation adds {d2:+.4f} over the "
                          f"additive model",
          if_false=lambda: f"H2 FAIL -- the two-tower model adds {d2:+.4f} against a {H2_BAR} bar")
    res["H2"] = {"delta": d2, "se": se2, "z": z2}

    # ---------------------------------------------------------------- H3
    say("H3 DID THE NETWORK HELP, OR THE FEATURES?")
    d3, se3, z3 = H.paired(full, lin_s)
    say(f"     two-tower {np.nanmean(full):.4f} vs its LINEAR TWIN on identical tower inputs "
        f"{np.nanmean(lin_s):.4f}")
    say(f"     paired {d3:+.4f} +/- {se3:.4f}  ({z3:+.1f} se)")
    say(f"     loop 241's MLPs lost this by -0.0155 and -0.0170; loop 251's by -0.0088")
    G.add("H3", bool(d3 >= H3_BAR), stat=float(d3), requires=("H1",),
          if_true=lambda: f"H3 PASS -- the network beats its linear twin by {d3:+.4f}",
          if_false=lambda: f"H3 FAIL -- the network is {d3:+.4f} against a linear model given the "
                           f"identical tower inputs")
    res["H3"] = {"delta": d3, "se": se3, "z": z3}

    # ---------------------------------------------------------------- H4
    say("H4 DOES THE LINE TOWER DO ANYTHING?")
    d4, se4, z4 = H.paired(full, abl_s)
    say(f"     full {np.nanmean(full):.4f} vs line embedding zeroed {np.nanmean(abl_s):.4f}")
    say(f"     paired {d4:+.4f} +/- {se4:.4f}  ({z4:+.1f} se)")
    G.add("H4", bool(d4 >= H4_BAR), stat=float(d4), requires=("H1",),
          if_true=lambda: f"H4 PASS -- knowing what the cell line IS adds {d4:+.4f}",
          if_false=lambda: f"H4 FAIL -- the line embedding is worth {d4:+.4f}"
                           + (f"; it is not decoration, it is ACTIVELY HARMFUL -- zeroing it "
                              f"IMPROVES the model by {-d4:.4f}" if d4 < 0 else
                              f"; below the {H4_BAR} bar, so the cell line is decoration"))
    res["H4"] = {"delta": d4, "se": se4, "z": z4}

    # ---------------------------------------------------------------- H5
    say("H5 IS THE ADVANTAGE LARGER THAN THE SEED NOISE?")
    sds = float(np.std([np.nanmean(S[s]) for s in SEEDS], ddof=1))
    say(f"     across {len(SEEDS)} seeds: " + ", ".join(f"{np.nanmean(S[s]):.4f}" for s in SEEDS)
        + f"   sd {sds:.5f}")
    if d2 < H2_BAR:
        G.add("H5", False, stat=float(d2), requires=("H2",), void_if=True,
              void_reason=f"H2 found {d2:+.4f}; there is no advantage for a seed spread to be "
                          f"compared against")
    else:
        G.add("H5", bool(d2 >= H5_MULT * sds), stat=float(sds), requires=("H2",),
              if_true=lambda: f"H5 PASS -- {d2:+.4f} is {d2 / max(sds, 1e-9):.1f}x the seed sd",
              if_false=lambda: f"H5 FAIL -- {d2:+.4f} against a seed sd of {sds:.5f}")
    res["H5"] = {"seed_sd": sds, "per_seed": [float(np.nanmean(S[s])) for s in SEEDS]}

    # ---------------------------------------------------------------- H6
    say("H6 CONTROL: THE WRONG CELL LINE")
    if d4 < 0.005:
        G.add("H6", False, stat=float(d4), requires=("H4",), void_if=True,
              void_reason=f"H4's margin is {d4:+.4f}; there is nothing to collapse")
    else:
        sh, _ = run(SEEDS[0], shuffle_line=True)
        d6, _, _ = H.paired(sh, abl_s)
        f6 = d6 / d4
        say(f"     line tower fed another line's properties: advantage over ablated {d6:+.4f} "
            f"against a real {d4:+.4f}  ({f6:.0%})")
        G.add("H6", bool(f6 <= H6_MAX), stat=float(f6), requires=("H4",),
              if_true=lambda: f"H6 PASS -- collapses to {f6:.0%} on the wrong line",
              if_false=lambda: f"H6 FAIL -- {f6:.0%} survives the wrong line's properties")
        res["H6"] = {"real": d4, "shuffled": d6, "fraction": f6}

    say("H7 WHAT THIS CANNOT SHOW")
    say("     Nine cell lines is nine points for the line tower to generalise over. Almost any")
    say("     positive here should be suspected of memorising the eight training lines, which is")
    say("     what H6 catches and why H5 uses three seeds.")
    say("     978 landmarks, not a transcriptome, and shRNA rather than a clean knockout.")
    say("     The gene tower reads the gene's own response in other lines, so this cannot predict")
    say("     a gene never perturbed anywhere: it generalises across CONTEXTS, not across genes.")
    say("     Beating the additive baseline would show the interaction is partly learnable. It")
    say("     would NOT say what the interaction is, and nothing here recovers the mechanism")
    say("     loops 253 and 255 failed to find.")

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
