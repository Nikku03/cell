"""Loop 165. Atom-level chemistry as a PAIRWISE score, and whether the shortlist rule works.

THE PROPOSAL. Only N, O and F form hydrogen bonds; carbon bonds covalently and selectively; so use
atom types to shorten the candidate list. Two separate claims live in that, and they do not stand
or fall together:

  (a) THE FILTER. Measured before anything is built: 8,266 of 8,428 non-currency candidates contain
      at least one N, O or F -- 98.1%. Atom PRESENCE removes 1.9% of the list and is not a
      shortlist. Nearly every biological metabolite carries a heteroatom.

  (b) THE COMPLEMENTARITY. Atom COUNTS and RATIOS do vary: median 7 H-bond-capable atoms with a long
      tail, heteroatom fraction averaging 0.318, sulfur in 2,062 candidates and phosphorus in 2,077,
      charge splitting 4,599 negative / 3,313 neutral / 516 positive. And matching those against
      what a protein's SURFACE presents is a property of the PAIR.

WHY (b) IS DIFFERENT FROM EVERYTHING IN THIS ARC SO FAR. Every block up to loop 164 -- sequence,
geometry, electrostatics, sterics -- is a vector describing a PROTEIN, handed to a k-NN that finds
similar proteins and copies their answers. None of them can express "this protein and this candidate
fit each other", because k-NN never sees a candidate. A complementarity term scores the (protein,
candidate) pair directly, which is a kind of information the existing merge structurally cannot
contain. That is the reason to expect it to be independent, and P4 measures whether it is.

THE TERMS, each a chemical statement rather than a fitted parameter:
  charge          -(surface net charge) x (candidate charge): opposite signs attract
  hydrogen bonds  surface donor capacity x candidate acceptor count, both log-scaled
  hydrophobic     surface hydrophobic fraction x candidate carbon fraction
  size / sterics  candidate heavy-atom count against the protein's largest cavity volume, scored by
                  how well the ligand FITS rather than by either being large
  sulfur          surface Cys/Met fraction x candidate sulfur count, since S-S and S-metal chemistry
                  is specific and only a quarter of candidates carry sulfur at all

PREDECLARED, before any number is looked at.

  P1 THE FILTER CLAIM, MEASURED. What fraction of candidates survive an atom-presence rule, and
     what fraction of TRUE answers survive it.
     Gate: PASS iff a presence filter removes more than 20% of candidates while retaining more than
     99% of true answers. This gate is expected to FAIL on the numbers already seen, and it is
     written so the proposal's literal form is tested rather than quietly replaced by the version
     that works.

  P2 DOES COMPLEMENTARITY SCORE ABOVE CHANCE, alone, with no k-NN and no learning?
     Gate: more than 3 sem above the popularity floor on the frequency-matched contests.

  P3 IS IT INDEPENDENT of the four existing blocks? Spearman against each.
     Gate: passes on being reported. Loop 163d established that independence, not solo strength,
     decides whether a block earns a place -- electrostatics scored worst alone and was
     load-bearing, sterics scored better and was droppable.

  P4 DOES IT ADD TO THE FOUR-BLOCK MERGE? Score-space fusion at a weight fitted on one half of the
     enzymes and scored on the other, both ways round, against loop 163d's frozen configuration.
     Gate: more than 3 sem.

  P5 WHICH TERM CARRIES IT. Each of the five complementarity terms alone, and the regret of dropping
     each from the full complementarity score.
     Gate: passes on all five being reported.

  P6 WHAT THIS CANNOT SHOW. Elemental formulas carry no connectivity, so an N in an amine and an N
     in a nitrile are the same atom here. Surface composition is a whole-protein average, not the
     composition of a binding site. And a fitted weight in P4 is fitted on DEV enzymes, not on the
     locked test split, which this arc has still never touched.

-> outputs/loop_atom_complementarity.json
"""
import gzip
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                              # noqa: E402
import run_manifest as RM                            # noqa: E402
import loop_replication as LR                        # noqa: E402
from rem.harness import REM                          # noqa: E402
from loop_struct_vs_seq import homology_folds, knn_scores, NFOLD, SEED  # noqa: E402

SEQF = Path("colab/data/ml/esm_enzymes.npz")
STRF = Path("colab/data/ml/struct_enzymes.npz")
ESF = Path("colab/data/ml/elecster_enzymes.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_atom_complementarity.json"
NEG_PER_POS, TOL = 40, 0.02
W163D = (0.1, 0.1, 0.1)
DONOR = list("STYNQKRWH")
ACCEPT = list("DENQSTY")
PHOBIC = list("AVLIMFW")
P1_SHRINK, P1_RECALL = 0.20, 0.99

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def contest_auc(sv, mini):
    if not mini:
        return np.nan
    return float(np.mean([float((sv[n] < sv[p]).sum() + 0.5 * (sv[n] == sv[p]).sum()) / len(n)
                          for p, n in mini]))


def main():
    t0 = time.time()
    say("=" * 104)
    say("  ATOM-LEVEL COMPLEMENTARITY: a score of the PAIR, not of the protein")
    say("=" * 104)
    say()

    R = REM()
    els = list(map(str, R.elements))
    ei = {e: k for k, e in enumerate(els)}
    E = R.E[R.noncur].astype(float)
    qc = R.charge[R.noncur].astype(float)
    heavy = E.sum(1) - (E[:, ei["H"]] if "H" in ei else 0.0)
    hb = E[:, [ei[e] for e in ("N", "O", "F") if e in ei]].sum(1)
    Cc = E[:, ei["C"]] if "C" in ei else np.zeros(len(E))
    Sc = E[:, ei["S"]] if "S" in ei else np.zeros(len(E))
    with np.errstate(divide="ignore", invalid="ignore"):
        het_frac = np.where(heavy > 0, hb / heavy, 0.0)
        c_frac = np.where(heavy > 0, Cc / heavy, 0.0)
    say(f"     {len(E):,} non-currency candidates | median {np.median(hb):.0f} H-bond atoms, "
        f"heteroatom fraction {het_frac.mean():.3f}, {int((Sc > 0).sum()):,} carry sulfur")

    S = np.load(SEQF, allow_pickle=False)
    T = np.load(STRF, allow_pickle=False)
    ES = np.load(ESF, allow_pickle=False)
    common = sorted(set(map(str, S["accs"])) & set(map(str, T["accs"])) & set(map(str, ES["accs"])))
    ix = [{a: i for i, a in enumerate(map(str, z["accs"]))} for z in (S, T, ES)]
    E35 = S["esm35"][[ix[0][a] for a in common]]
    GEO = T["X"][[ix[1][a] for a in common]]
    ELE = ES["elec"][[ix[2][a] for a in common]]
    STE = ES["steric"][[ix[2][a] for a in common]]
    gname = list(map(str, T["names"]))
    sname = list(map(str, ES["steric_names"]))
    ename = list(map(str, ES["elec_names"]))

    def gcol(nm):
        return GEO[:, gname.index(nm)]
    surf_don = sum(gcol(f"surf_{a}") for a in DONOR if f"surf_{a}" in gname)
    surf_acc = sum(gcol(f"surf_{a}") for a in ACCEPT if f"surf_{a}" in gname)
    surf_pho = sum(gcol(f"surf_{a}") for a in PHOBIC if f"surf_{a}" in gname)
    surf_s = sum(gcol(f"surf_{a}") for a in ("C", "M") if f"surf_{a}" in gname)
    surf_q = ELE[:, ename.index("surface_net_charge")]
    cav = STE[:, sname.index("log_cavity_volume")]

    R2 = REM()
    Z = np.load("colab/data/rem_enzyme.npz", allow_pickle=False)
    sym = list(map(str, Z["symbols"]))
    grx = defaultdict(set)
    for j, g in zip(Z["gpr_rx"], Z["gpr_gene"]):
        grx[sym[int(g)]].add(int(j))
    a2g, seqs, acc, buf = {}, {}, None, []
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if acc and buf:
                    seqs[acc] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                g = re.search(r"GN=(\S+)", ln)
                acc, buf = (m.group(1) if m else None), []
                if acc and g:
                    a2g[acc] = g.group(1)
            else:
                buf.append(ln.strip())
    if acc and buf:
        seqs[acc] = "".join(buf)

    Y = np.zeros((len(common), len(R.noncur)), np.float32)
    for i, a in enumerate(common):
        for j in grx.get(a2g.get(a, ""), ()):
            for m in (R2.react_of[j] | R2.prod_of[j]) - R2.currency:
                Y[i, R2.ncmap[int(m)]] = 1.0
    keep = Y.sum(1) > 0
    accs = [a for a, k in zip(common, keep) if k]
    E35, GEO, ELE, STE, Y = E35[keep], GEO[keep], ELE[keep], STE[keep], Y[keep]
    surf_don, surf_acc, surf_pho, surf_s = (surf_don[keep], surf_acc[keep],
                                            surf_pho[keep], surf_s[keep])
    surf_q, cav = surf_q[keep], cav[keep]
    pop = Y.mean(0)

    # ------------------------------------------------------------------ P1
    say()
    say("P1 THE FILTER CLAIM, MEASURED")
    for thr in (1, 2, 4):
        surv = hb >= thr
        kept_true = float((Y[:, surv].sum()) / max(Y.sum(), 1))
        say(f"     >= {thr} H-bond atom(s): keeps {surv.mean():6.1%} of candidates, "
            f"{kept_true:6.2%} of true answers")
    surv1 = hb >= 1
    shrink = 1 - surv1.mean()
    recall = float(Y[:, surv1].sum() / max(Y.sum(), 1))
    p1 = bool(shrink > P1_SHRINK and recall > P1_RECALL)
    say(f"     the literal rule (needs >= 1 N/O/F): removes {shrink:.1%} of candidates, "
        f"retains {recall:.2%} of answers")
    GG.verdict(p1, emit=say, if_true=(
        f"the presence filter is a real shortlist: it removes {shrink:.1%} while keeping "
        f"{recall:.2%} of the answers."), if_false=(
        f"the presence filter is NOT a shortlist. It removes {shrink:.1%} of candidates, far under "
        f"the {P1_SHRINK:.0%} bar, because 98.1% of biological metabolites already contain N or O. "
        f"Atom PRESENCE does not discriminate; atom COUNTS and ratios might, and P2 tests those."))
    say(f"     P1 {'PASS' if p1 else 'FAIL'}")

    # ---------------------------------------------------------------- contests
    order = np.argsort(pop, kind="stable")
    posn = np.empty(len(pop), int)
    posn[order] = np.arange(len(pop))
    half = NEG_PER_POS // 2
    cand, ndrop = [], 0
    for i in range(len(accs)):
        mini = []
        for p in np.where(Y[i] > 0)[0]:
            b, a_ = [], []
            k = posn[p] - 1
            while k >= 0 and len(b) < half:
                if Y[i, order[k]] == 0:
                    b.append(order[k])
                k -= 1
            k = posn[p] + 1
            while k < len(order) and len(a_) < half:
                if Y[i, order[k]] == 0:
                    a_.append(order[k])
                k += 1
            if len(b) == half and len(a_) == half:
                mini.append((p, np.array(b + a_)))
            else:
                ndrop += 1
        cand.append(mini)
    fold, ncl, _ = homology_folds(seqs, accs)
    say(f"     {sum(len(m) for m in cand):,} mini-contests | {ncl:,} homology clusters")

    # ------------------------------------------------------------- the terms
    z = lambda v: (v - v.mean()) / max(v.std(), 1e-9)  # noqa: E731
    lhb, lheavy, lS = np.log1p(hb), np.log1p(heavy), np.log1p(Sc)
    TERMS = {
        "charge": lambda i: -z(surf_q)[i] * z(qc),
        "hbond": lambda i: z(surf_don)[i] * z(lhb) + z(surf_acc)[i] * z(lhb),
        "hydrophobic": lambda i: z(surf_pho)[i] * z(c_frac),
        "size_fit": lambda i: -np.abs(z(cav)[i] - z(lheavy)),
        "sulfur": lambda i: z(surf_s)[i] * z(lS),
    }

    def comp_score(i, use=None):
        use = use or list(TERMS)
        return sum(TERMS[t](i) for t in use)

    A = {}
    A["complementarity"] = np.array([contest_auc(comp_score(i), cand[i])
                                     for i in range(len(accs))])
    A["popularity"] = np.array([contest_auc(pop, cand[i]) for i in range(len(accs))])

    def zs(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    B = {"sequence": zs(E35), "geometry": zs(GEO), "electrostatics": zs(ELE), "sterics": zs(STE)}
    P = {k: np.zeros_like(Y) for k in B}
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        for k, X in B.items():
            P[k][te] = knn_scores(X[tr], Y[tr], X[te])
        say(f"     fold {f} arms computed [{time.time()-t0:.0f}s]")
    for k in B:
        A[k] = np.array([contest_auc(P[k][i], cand[i]) for i in range(len(accs))])
    ok = np.isfinite(A["sequence"]) & np.isfinite(A["complementarity"])

    def mn(a):
        return float(np.nanmean(a[ok]))

    def pdiff(a, b):
        d = a[ok] - b[ok]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ P2
    d2, s2 = pdiff(A["complementarity"], A["popularity"])
    p2 = bool(d2 > 3 * s2)
    say()
    say("P2 DOES COMPLEMENTARITY SCORE ABOVE CHANCE, alone?")
    for k in sorted(A, key=lambda x: -mn(A[x])):
        say(f"     {k:<18s} {mn(A[k]):.4f}")
    say(f"     complementarity minus popularity: {d2:+.4f} sem {s2:.4f} = {d2/s2:+.1f} sem")
    GG.verdict(p2, emit=say, if_true=(
        "a pairwise chemical score with no learning and no graph beats chance."), if_false=(
        "the complementarity terms do not beat chance; the pair-level chemistry as formulated here "
        "carries nothing."))
    say(f"     P2 {'PASS' if p2 else 'FAIL'}")

    # ------------------------------------------------------------------ P3
    say()
    say("P3 IS IT INDEPENDENT of the four existing blocks?")
    rhos = {}
    for k in B:
        r = float(stats.spearmanr(A["complementarity"][ok], A[k][ok]).statistic)
        rhos[k] = r
        say(f"     complementarity vs {k:<16s} {r:+.4f}")
    p3 = True
    GG.verdict(max(abs(v) for v in rhos.values()) < 0.30, emit=say, if_true=(
        "it is close to orthogonal to everything already in the merge, which is what loop 163d "
        "showed decides whether a block earns its place."), if_false=(
        "it overlaps an existing block more than electrostatics did (+0.211 with geometry), so "
        "less of it is new than it looks."))
    say(f"     P3 {'PASS' if p3 else 'FAIL'}")

    # ------------------------------------------------------------------ P4
    say()
    say("P4 DOES IT ADD TO THE FOUR-BLOCK MERGE? weight held out on halves")

    def nmax(v):
        m = v.max()
        return v / m if m > 0 else v
    base = np.array([contest_auc(
        nmax(P["sequence"][i]) + W163D[0] * nmax(P["geometry"][i])
        + W163D[1] * nmax(P["electrostatics"][i]) + W163D[2] * nmax(P["sterics"][i]),
        cand[i]) for i in range(len(accs))])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(accs))
    H = [perm[:len(accs) // 2], perm[len(accs) // 2:]]
    GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    rows = []
    for a, b in ((0, 1), (1, 0)):
        fit, test = H[a], H[b]
        best, bv = None, -1
        for w in GRID:
            v = np.nanmean([contest_auc(
                nmax(P["sequence"][i]) + W163D[0] * nmax(P["geometry"][i])
                + W163D[1] * nmax(P["electrostatics"][i]) + W163D[2] * nmax(P["sterics"][i])
                + w * nmax(comp_score(i) - comp_score(i).min()), cand[i]) for i in fit])
            if v > bv:
                bv, best = v, w
        held = np.array([contest_auc(
            nmax(P["sequence"][i]) + W163D[0] * nmax(P["geometry"][i])
            + W163D[1] * nmax(P["electrostatics"][i]) + W163D[2] * nmax(P["sterics"][i])
            + best * nmax(comp_score(i) - comp_score(i).min()), cand[i]) for i in test])
        d = held - base[test]
        d = d[np.isfinite(d)]
        rows.append({"w": best, "fused": float(np.nanmean(held)), "delta": float(d.mean()),
                     "sem": float(d.std() / np.sqrt(len(d)))})
        say(f"       fold {a}->{b}: w={best} | fused {np.nanmean(held):.4f} vs 163d "
            f"{np.nanmean(base[test]):.4f} | delta {d.mean():+.4f} sem "
            f"{d.std()/np.sqrt(len(d)):.4f}")
    d4 = float(np.mean([r["delta"] for r in rows]))
    s4 = float(np.mean([r["sem"] for r in rows]))
    p4 = bool(d4 > 3 * s4)
    GG.verdict(p4, emit=say, if_true=f"complementarity adds {d4:+.4f} on top of the four blocks.",
               if_false=f"complementarity adds {d4:+.4f} against a 3-sem bar of {3*s4:.4f}.")
    say(f"     P4 {'PASS' if p4 else 'FAIL'}")

    # ------------------------------------------------------------------ P5
    say()
    say("P5 WHICH TERM CARRIES IT")
    solo, reg = {}, {}
    for t in TERMS:
        v = np.array([contest_auc(TERMS[t](i), cand[i]) for i in range(len(accs))])
        solo[t] = mn(v)
        others = [x for x in TERMS if x != t]
        w = np.array([contest_auc(comp_score(i, others), cand[i]) for i in range(len(accs))])
        d, s = pdiff(A["complementarity"], w)
        reg[t] = {"solo": mn(v), "without": mn(w), "regret": d, "sem": s}
        say(f"     {t:<14s} alone {mn(v):.4f} | without it {mn(w):.4f} | regret {d:+.4f} "
            f"({'load-bearing' if d > 3 * s else 'droppable'})")
    p5 = True
    say(f"     P5 {'PASS' if p5 else 'FAIL'}")

    say()
    say("P6 WHAT THIS CANNOT SHOW")
    say("     Elemental formulas carry no connectivity: an N in an amine and an N in a nitrile are")
    say("     the same atom here, so 'can hydrogen bond' is an upper bound on every candidate.")
    say("     Surface composition is a whole-protein average, not the composition of a binding site.")
    say("     The P4 weight is fitted on DEV enzymes; the locked test split is still untouched.")
    p6 = True
    say(f"     P6 {'PASS' if p6 else 'FAIL'}")

    gates = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6}
    man = RM.manifest(inputs=[SEQF, STRF, ESF], available=len(accs), used=int(ok.sum()),
                      selection="all", seed=SEED,
                      controls=["the literal presence-filter claim gated separately from the count-based one",
                                "frequency-matched contests, popularity reported alongside",
                                "independence measured against every existing block",
                                "the fusion weight fitted on one half of the enzymes and scored on the other",
                                "per-term regret, so a passing total cannot hide four dead terms"],
                      note="atom-level pairwise complementarity between protein surface and candidate")
    out = {"test": "atom complementarity", "gates": gates,
           "arms": {k: mn(A[k]) for k in A}, "spearman": rhos,
           "filter": {"shrink": shrink, "recall": recall},
           "p4_folds": rows, "solo_terms": solo, "term_regret": reg,
           "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
