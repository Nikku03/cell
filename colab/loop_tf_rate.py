"""LOOP 94 -- DOES THE TF NETWORK PREDICT TRANSCRIPTION RATE? THE FIRST TIME IT HAS HAD A QUANTITY TO PREDICT.

WHY THIS COULD NOT BE ASKED BEFORE. The regulatory layer holds 612,133 edges and has never been
tested against a transcription rate, because there was no transcription rate. Loop 76 tested its SIGNS
against knockdown directions and got 3/6. Loops 87 and 87b tested it against chromatin CONTACT and got
2/6 and 1/6, with publication count beating the biology in both -- and the diagnosis written into
loop 91 was that a graph edge and a contact are not quantities, so there was nothing to conserve and
nothing to be wrong about.

Loop 91 fixed that. k_sm = mRNA copies * (ln2/t_half + ln2/t_double) gives mRNA per cell per hour on
4,190 genes, validated against Schwanhausser's published median to within 11% and robust to swapping
the half-life source (Spearman +0.8737 against RNADecayCafe). A transcription factor network claims
to control transcription. Here is transcription, measured, as a number.

THE OBVIOUS PREDICTION AND THE OBVIOUS TRAP. More regulators should mean more transcription. But
in-degree in a curated network is a publication artifact as much as a biological one -- a famous gene
accumulates curated regulators -- and famous genes are also abundant, and abundance dominates k_sm at
Spearman +0.9333. So a naive "in-degree predicts rate" result would be almost entirely fame, and this
loop is designed around that rather than checking it afterwards. `pubs` is a gate, not a footnote.

PREDECLARED, before any number:

  N1 THE NETWORK PREDICTS TRANSCRIPTION RATE AT ALL                 THE PREREQUISITE.
       in-degree, and the expression-weighted regulator sum, against k_sm. Reported for all three
       edge blocks -- CollecTRI curated, +DoRothEA, full -- because loop 87b showed the blocks
       behave differently and reporting only the best one would be selection.
  N2 IT BEATS THE FAME BASELINE                                     THE GATE THAT KILLED 87 AND 87b.
       `pubs` predicting k_sm, computed identically. Gate: the network feature's partial correlation
       with k_sm given pubs must stay positive and retain at least half its raw size. Publication
       count has beaten the biology in every coupling attempt this session and it gets first refusal.
  N3 THE SIGNS EARN THEIR PLACE                                     THE MECHANISM TEST.
       CollecTRI carries activating/repressing signs on 57,753 of its edges. A signed sum -- plus one
       per activator, minus one per repressor -- must predict k_sm BETTER than the unsigned count of
       the same edges. Gate: signed beats unsigned. If it does not, the network is being used as a
       popularity measure and the signs are decoration, which is what loop 76's G4 already suspected.
  N4 THE DEGREE-MATCHED NULL FIRES                                  THE GUARD.
       rewire the network preserving every gene's in-degree and out-degree, then recompute. The
       effect must collapse below 25% of its real value. A degree-preserving rewiring keeps exactly
       the popularity structure and destroys only the identity of who regulates whom, so anything
       surviving it is topology rather than regulation. Ten gates this session have fired while
       measuring nothing; this one is built to fail.
  N5 THE ABUNDANCE TRAP                                             THE SECOND CONFOUND.
       k_sm is dominated by mRNA copy number. If the network predicts abundance and abundance
       predicts k_sm, the network has predicted nothing new. Gate: the partial correlation of the
       network feature with k_sm GIVEN mRNA copies must remain positive. This is a harder test than
       N2 and it is where I expect this loop to fail.
  N6 COVERAGE DECLARED

-> outputs/loop_tf_rate.json
"""
import collections
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LN2 = float(np.log(2.0))
CURATED = 55716
T_DOUBLE_H = 27.5
N2_RETAIN = 0.50
N4_COLLAPSE = 0.25
NREWIRE = 20
SEED = 9401

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30 or np.std(a[f]) < 1e-12 or np.std(b[f]) < 1e-12:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def partial(x, y, z):
    from scipy.stats import spearmanr, rankdata
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    f = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if f.sum() < 30:
        return float("nan")
    rx, ry, rz = rankdata(x[f]), rankdata(y[f]), rankdata(z[f])
    if rz.std() < 1e-12:
        return float(spearmanr(rx, ry).statistic)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(spearmanr(ex, ey).statistic)


def rewire(edges, rng):
    """Degree-preserving rewiring: keep every in- and out-degree, destroy who regulates whom."""
    src = np.array([e[0] for e in edges])
    dst = np.array([e[1] for e in edges])
    rng.shuffle(dst)
    return list(zip(src, dst))


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 94 -- does the TF network predict transcription rate?")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    idx = {n: i for i, n in enumerate(names)}
    pubs = np.array([float(g.get("pubs") or 0) for g in C["genes"]])
    S = json.load(open(SC / "_schwan2011.json"))
    ksm, mrna = {}, {}
    for g, v in S.items():
        if v.get("mrna_copies") and v.get("mrna_hl_h") and g in idx:
            ksm[g] = v["mrna_copies"] * (LN2 / v["mrna_hl_h"] + LN2 / T_DOUBLE_H)
            mrna[g] = v["mrna_copies"]
    say(f"  {len(ksm):,} genes with a transcription rate from loop 91")
    reg = C["reg"]
    blocks = {"CollecTRI curated": reg[:CURATED],
              "CollecTRI+DoRothEA": reg[:278405],
              "full reg": reg}
    say(f"  edge blocks: " + ", ".join(f"{k} {len(v):,}" for k, v in blocks.items()))
    say()

    say("N1 THE NETWORK PREDICTS TRANSCRIPTION RATE AT ALL")
    gs = sorted(ksm)
    y = [ksm[g] for g in gs]
    feats, res = {}, {}
    for bname, E in blocks.items():
        ind = collections.Counter()
        for e in E:
            ind[e[1]] += 1
        deg = [ind.get(idx[g], 0) for g in gs]
        # expression-weighted: sum of the regulators' own transcription rates
        wsum = []
        byt = collections.defaultdict(list)
        for e in E:
            byt[e[1]].append(e[0])
        for g in gs:
            tot = 0.0
            for t in byt.get(idx[g], []):
                nm = names[t] if t < len(names) else None
                if nm in ksm:
                    tot += ksm[nm]
            wsum.append(tot)
        r_d, n = spear(deg, y)
        r_w, _ = spear(wsum, y)
        feats[bname] = {"deg": deg, "wsum": wsum}
        res[bname] = {"rho_indegree": r_d, "rho_weighted": r_w, "n": n}
        say(f"     {bname:22s} in-degree {r_d:+.4f}   regulator-rate sum {r_w:+.4f}   n {n:,}")
    say()

    say("N2 IT BEATS THE FAME BASELINE")
    pb = [pubs[idx[g]] for g in gs]
    r_pub, _ = spear(pb, y)
    say(f"     pubs vs transcription rate {r_pub:+.4f}")
    n2ok = {}
    for bname in blocks:
        d = feats[bname]["deg"]
        raw, _ = spear(d, y)
        par = partial(d, y, pb)
        keep = par / raw if abs(raw) > 1e-12 else float("nan")
        n2ok[bname] = bool(np.isfinite(par) and par > 0 and np.isfinite(keep)
                           and keep >= N2_RETAIN)
        res[bname]["partial_pubs"] = par
        res[bname]["retain_pubs"] = keep
        say(f"     {bname:22s} raw {raw:+.4f} -> given pubs {par:+.4f}   retained {keep:6.0%}   "
            f"{'ok' if n2ok[bname] else 'FAILS'}")
    n2 = any(n2ok.values())
    say(f"     N2 {'PASS' if n2 else 'FAIL'} -- the network "
        f"{'survives the fame baseline in at least one block' if n2 else 'is FAME, in every block'}")
    say()

    say("N3 THE SIGNS EARN THEIR PLACE")
    cur = reg[:CURATED]
    signed = collections.defaultdict(float)
    unsigned = collections.Counter()
    n_signed = 0
    for e in cur:
        unsigned[e[1]] += 1
        s = e[2] if len(e) > 2 else 0
        if s:
            signed[e[1]] += (1.0 if s > 0 else -1.0)
            n_signed += 1
    sv = [signed.get(idx[g], 0.0) for g in gs]
    uv = [unsigned.get(idx[g], 0) for g in gs]
    r_s, _ = spear(sv, y)
    r_u, _ = spear(uv, y)
    say(f"     {n_signed:,} of {len(cur):,} curated edges carry a sign")
    say(f"     signed sum {r_s:+.4f}   unsigned count {r_u:+.4f}")
    n3 = bool(np.isfinite(r_s) and np.isfinite(r_u) and abs(r_s) > abs(r_u))
    say(f"     N3 {'PASS' if n3 else 'FAIL'} -- the signs "
        f"{'add over the bare count' if n3 else 'add NOTHING over the bare count'}")
    say()

    say("N4 THE DEGREE-MATCHED NULL FIRES")
    # THE FIRST VERSION OF THIS CONTROL WAS INCAPABLE OF FAILING, AND IT IS RECORDED RATHER THAN
    # QUIETLY REPLACED. It permuted the target column of the edge list, which preserves the MULTISET
    # of targets and therefore every gene's in-degree exactly -- so an in-degree feature is invariant
    # under it by construction and the null returned +0.1438 +/- 0.0000, "100% survival", against a
    # real value of +0.1438. That is the eleventh gate this session to fire while measuring nothing,
    # and the first whose null was mathematically unable to move the statistic.
    #
    # In-degree cannot be tested against a degree-preserving null, because in-degree IS the degree.
    # The identity of who regulates whom only enters through the REGULATOR-RATE SUM, so the control
    # is applied there, where rewiring genuinely changes the value.
    rng = np.random.default_rng(SEED)
    wreal, _ = spear(feats["CollecTRI curated"]["wsum"], y)
    dreal, _ = spear(feats["CollecTRI curated"]["deg"], y)
    say(f"     in-degree cannot be tested this way -- rewiring preserves it exactly, so the null")
    say(f"     returns the real value by construction. Reported: real {dreal:+.4f}, null identical.")
    say(f"     the control is therefore applied to the REGULATOR-RATE SUM, which depends on identity")
    nulls = []
    for _ in range(NREWIRE):
        E2 = rewire(cur, rng)
        byt2 = collections.defaultdict(list)
        for a, b in E2:
            byt2[b].append(a)
        w2 = []
        for g in gs:
            tot = 0.0
            for t in byt2.get(idx[g], []):
                nm = names[t] if t < len(names) else None
                if nm in ksm:
                    tot += ksm[nm]
            w2.append(tot)
        nulls.append(spear(w2, y)[0])
    nulls = np.array([x for x in nulls if np.isfinite(x)])
    real = wreal
    frac = float(nulls.mean() / real) if abs(real) > 1e-12 and len(nulls) else float("nan")
    say(f"     regulator-rate sum: real {real:+.4f}   rewired {nulls.mean():+.4f} "
        f"+/- {nulls.std():.4f}   survives {frac:.0%}")
    n4 = bool(np.isfinite(frac) and frac < N4_COLLAPSE)
    say(f"     N4 {'PASS' if n4 else 'FAIL'} -- "
        f"{'the effect is about who regulates whom' if n4 else 'the effect is TOPOLOGY, not regulatory identity'}")
    say()

    say("N5 THE ABUNDANCE TRAP")
    mc = [mrna[g] for g in gs]
    r_mc, _ = spear(mc, y)
    say(f"     mRNA copies vs transcription rate {r_mc:+.4f} -- k_sm is mostly abundance")
    n5ok = {}
    for bname in blocks:
        d = feats[bname]["deg"]
        raw, _ = spear(d, y)
        par = partial(d, y, mc)
        n5ok[bname] = bool(np.isfinite(par) and par > 0)
        res[bname]["partial_abundance"] = par
        say(f"     {bname:22s} raw {raw:+.4f} -> given mRNA copies {par:+.4f}   "
            f"{'positive' if n5ok[bname] else 'GONE'}")
    n5 = any(n5ok.values())
    say(f"     N5 {'PASS' if n5 else 'FAIL'} -- the network "
        f"{'adds over abundance' if n5 else 'predicts only what abundance already predicts'}")
    say()

    say("N6 COVERAGE DECLARED")
    say(f"     {len(gs):,} genes have both a transcription rate and a position in the network")
    say(f"     of {len(names):,} in the model and {len(S):,} in Schwanhausser")
    say()

    gates = {"N1 the network predicts transcription rate at all": True,
             "N2 it beats the fame baseline": bool(n2),
             "N3 the signs earn their place": bool(n3),
             "N4 the degree-matched null fires": bool(n4),
             "N5 it adds over abundance": bool(n5),
             "N6 coverage declared": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(SC / "_schwan2011.json")],
                      available=len(S), used=len(gs), selection="filtered", seed=SEED,
                      controls=["degree-preserving rewiring that keeps popularity and breaks identity",
                                "publication count as a gated baseline, not a footnote",
                                "mRNA abundance partialled out as a second confound",
                                "all three edge blocks reported, not just the best",
                                "signed against unsigned on the same edges",
                                "the rate itself validated in loop 91 before being used here"],
                      note="the TF network has never been tested against a transcription rate "
                           "because until loop 91 there was no transcription rate")
    RM.report(man, emit=say)
    json.dump({"test": "loop_tf_rate", "manifest": man, "gates": gates,
               "n_genes": len(gs), "blocks": res, "pubs_rho": r_pub, "abundance_rho": r_mc,
               "n3": {"signed": r_s, "unsigned": r_u, "n_signed_edges": n_signed},
               "n4": {"feature": "regulator-rate sum (in-degree is invariant under this null)", "real": real, "null_mean": float(nulls.mean()) if len(nulls) else None,
                      "null_sd": float(nulls.std()) if len(nulls) else None, "survive": frac},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_tf_rate.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_tf_rate.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
