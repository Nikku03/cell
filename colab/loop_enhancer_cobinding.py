"""Loop 185. Can measured co-binding and accessibility close stage two?

THE HONEST ANSWER BEFORE THE LOOP RUNS, because two of the three ideas here are already bounded by
measurements this arc has made, and saying so afterwards would be cheating.

  ACCESSIBILITY IS A PROPERTY OF THE ELEMENT. So is the number of factors bound at it. Loop 176's
  M5 and loop 178's P1 fixed what any element-intrinsic score can do on this task: an oracle that
  has seen every label but must give each element ONE score shared by all its genes reaches
  within-gene R@1 0.4422, BELOW distance-only at 0.5930. An element-intrinsic column cannot reorder
  a gene's own candidates any better than that, however well measured it is. Loop 178's P3 showed
  the practical version -- handing stage two an explicit "how good an enhancer is this" score cost
  0.1256 R@1. Z6 runs accessibility and the bound-factor count anyway, and predeclares that they
  should move pooled AUPRC and not R@1, because that is the only channel open to them.

  WHAT IS NOT BOUNDED is the overlap. An element's measured bound-factor set is a property of the
  element; a promoter's measured bound-factor set is a property of the gene; the OVERLAP between
  them is a property of the pair, and it is gene-varying, which loop 176 showed is where all the
  signal on this task lives. That is the idea worth testing and Z3 is its gate.

  AND LOOP 183'S BLOCK CAN NOW BE DONE PROPERLY. Loop 183 asked whether an element carries sites
  for the factors that regulate this gene, using MOTIF occupancy as the stand-in for "the factor is
  here". Its W5 showed the block was permutable at almost no cost -- it was reading how much
  occupancy the element carried, not who was there. Loop 184 then measured why: a motif predicts a
  factor's real binding at a median AUC of 0.6228, worse than accessibility (0.7902) and worse than
  simply counting the neighbours (0.8455). With ENCODE ChIP the same question can be asked with
  measured presence instead of predicted presence, and Z2 is that head-to-head.

WHAT IS BUILT. The same 191-factor roster scored twice: at the 4,482 benchmark elements (already
done, 13.1% of cells bound) and at the 2,205 promoters in a 1 kb window, so the two sets are
indexed column for column and their overlap means something.

THE CONTROL THAT DECIDES Z3. Validated enhancers carry a median of 40 bound factors against 15
elsewhere (loop 184's Y7), and an expressed gene's promoter is busy too. So a raw shared-factor
count rises with both marginals and would "work" for reasons that have nothing to do with the pair.
Z4 therefore requires the increment to survive when the overlap is expressed as pointwise mutual
information -- observed overlap over the overlap expected from the two degrees alone -- and Z5
permutes which promoter's factor set belongs to which gene, holding every count exactly.

NOTE ON WHICH CONTROL APPLIES. The dinucleotide shuffle that gated every earlier loop is a
SEQUENCE control and has no meaning for a ChIP measurement; shuffling the DNA does not move a peak
call. The corresponding control here is Z5's permutation, and it is the one that has to hold.

PREDECLARED, BEFORE ANY NUMBER.

  Z1 DOES THE PROMOTER SIDE JOIN? Factors bound per promoter, and how many promoters have any.
     Gate: PASS iff at least 90% of the benchmark's promoters carry at least one bound factor from
     the roster.

  Z2 MEASURED PRESENCE OR PREDICTED PRESENCE? Loop 183's regulator-match block recomputed with
     ChIP occupancy in place of motif occupancy, against loop 183's own version.
     Gate: paired per-seed R@1 positive in >= 4/5 and past 3 sem, AND paired AUPRC >= +0.01 in
     >= 4/5 -- loop 173's E3 bar, unchanged since loop 173.

  Z3 DOES THE ELEMENT-PROMOTER SHARED-FACTOR OVERLAP HELP? The overlap block over the base stack.
     Gate: same bar.

  Z4 IS IT THE OVERLAP OR THE TWO MARGINALS? The same block with the overlap expressed as pointwise
     mutual information against the degree-expected overlap, and with both degrees entered as
     separate columns so the model can use them directly.
     Gate: same bar, applied to the PMI form. A pass on the raw count and a fail here means the
     block reads busy-ness at both ends.

  Z5 WHOSE PROMOTER? The promoter factor sets permuted across genes, every count held exactly.
     Gate: PASS iff real beats permuted on R@1 in >= 4/5 seeds past 3 sem.

  Z6 THE ELEMENT-INTRINSIC ARM, run because it was asked for and predeclared as bounded.
     Accessibility and the bound-factor count over the base stack.
     Gate: PASS iff it clears the AUPRC bar and does NOT clear the R@1 bar. That is the pattern an
     element-intrinsic column must show given loop 178's 0.4422 ceiling; clearing R@1 as well would
     contradict a measurement this arc already made and would need explaining, not celebrating.

  Z7 THE DECISIVE ONE. The best arm against distance alone, identical folds.
     Gate: same bar as Z2. This is the gate loops 173, 175, 178, 179, 181, 182 and 183 were held
     to, and which only 183's binding-tier arm has cleared.

  Z8 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_cobinding.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import chip as CH                   # noqa: E402
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402
import loop_enhancer_tfnet as L183           # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_cobinding.json"
SEEDS = L173.SEEDS
NFOLD = 5
MIN_SEEDS = 4
MIN_PROM_COVER = 0.90
L173_DIST_R1 = 0.5930

OVERLAP = ["ov_n", "ov_jac", "ov_frac_el", "ov_frac_pr"]
PMI = ["ov_pmi", "ov_pmi_max", "log_deg_el", "log_deg_pr"]
INTRINSIC = ["acc_dhs", "acc_h3k", "log_n_bound_el"]
REGCHIP = ["chipreg_n", "chipreg_frac", "chipreg_idf"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def run(X, y, chrom, g_idx, jitter, tag, report=print):
    r1, ap = [], []
    for s in SEEDS:
        fold = L173.folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if te.sum() == 0 or y[tr].sum() == 0:
                continue
            m = gbm(s)
            m.fit(np.nan_to_num(X[tr]), y[tr])
            sc[te] = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
        r1.append(L173.within_gene(sc, y, g_idx, jitter)[0])
        ap.append(average_precision_score(y, sc))
    r1, ap = np.array(r1), np.array(ap)
    report(f"    {tag:42} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"AUPRC {ap.mean():.4f}")
    return dict(r1=r1, ap=ap, mrr=np.zeros(len(SEEDS)))


def overlap_features(Be, Bp, e_idx, g_idx, perm=None):
    """Be: (T, n_el) measured binding at elements. Bp: (T, n_prom) at promoters.
    `perm` reassigns which promoter's factor set a gene gets -- Z5's control."""
    T = Be.shape[0]
    de = Be.sum(0).astype(float)
    dp = Bp.sum(0).astype(float)
    n = len(e_idx)
    F = {k: np.zeros(n) for k in OVERLAP + PMI}
    # per-factor breadth, so a factor bound everywhere contributes little to a "shared" count
    breadth = Be.mean(1) + 1e-6
    for i in range(n):
        e = int(e_idx[i])
        g = int(g_idx[i]) if perm is None else int(perm[int(g_idx[i])])
        a, b = Be[:, e], Bp[:, g]
        both = a & b
        k = float(both.sum())
        F["ov_n"][i] = k
        uni = float((a | b).sum())
        F["ov_jac"][i] = k / uni if uni else 0.0
        F["ov_frac_el"][i] = k / de[e] if de[e] else 0.0
        F["ov_frac_pr"][i] = k / dp[g] if dp[g] else 0.0
        exp = de[e] * dp[g] / max(T, 1)
        F["ov_pmi"][i] = np.log2((k + 0.5) / (exp + 0.5))
        F["ov_pmi_max"][i] = (float(np.log2(1.0 / breadth[both]).max()) if k else 0.0)
        F["log_deg_el"][i] = np.log10(1.0 + de[e])
        F["log_deg_pr"][i] = np.log10(1.0 + dp[g])
    for k in F:
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    return F


def chip_regulator_features(Be, tf_names, sets, grow, e_idx, g_idx, tf_targets):
    """Loop 183's block with MEASURED presence: is a factor that regulates this gene actually
    bound at this element?"""
    pos = {t.upper(): i for i, t in enumerate(tf_names)}
    n = len(e_idx)
    F = {k: np.zeros(n) for k in REGCHIP}
    ntot = max(len(tf_targets), 1)
    for i in range(n):
        e, g = int(e_idx[i]), int(g_idx[i])
        gr = grow[g]
        d = sets["bind"].get(gr, {}) if gr >= 0 else {}
        rows, idf = [], []
        for name in d:
            j = pos.get(name)
            if j is not None:
                rows.append(j)
                idf.append(np.log(ntot / max(tf_targets.get(name, 0) + 1, 1)))
        if not rows:
            continue
        rows = np.array(rows)
        hit = Be[rows, e]
        F["chipreg_n"][i] = float(hit.sum())
        F["chipreg_frac"][i] = float(hit.sum()) / len(rows)
        F["chipreg_idf"][i] = float(np.array(idf)[hit].sum()) if hit.any() else 0.0
    for k in F:
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    return F


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 185  CAN MEASURED CO-BINDING AND ACCESSIBILITY CLOSE STAGE TWO?")
    say("=" * 104)
    say(f"  PREDECLARED: >= {MIN_PROM_COVER:.0%} of promoters must carry a bound factor; every arm")
    say(f"  on loop 173's E3 bar -- paired R@1 positive in >= {MIN_SEEDS}/5 past 3 sem AND paired")
    say(f"  AUPRC >= +0.01 in >= {MIN_SEEDS}/5; the overlap must survive being expressed as")
    say("  pointwise mutual information against its own two degrees; the promoter factor sets must")
    say("  not be permutable at no cost; the element-intrinsic arm is predeclared to move AUPRC and")
    say(f"  NOT R@1, because loop 178 put that ceiling at 0.4422; and the winner must clear")
    say(f"  distance-only R@1 {L173_DIST_R1}.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))
    dom = TD.load()
    names = sorted({(v.get("name") or "").upper().split("::")[0]
                    for v in dom.values() if v.get("name")})

    Be, tfs = CH.build(S["el_key"], names, say)
    prom_key = []
    for k in S["gn_key"]:
        c, p, _ = str(k).split(":")
        prom_key.append(f"{c}:{max(0, int(p) - CH.PROMOTER_PAD)}-{int(p) + CH.PROMOTER_PAD}")
    Bp, tfs_p = CH.build(prom_key, names, say, cache=CH.PROM_CACHE, pad=0)
    if tfs_p != tfs:
        common = sorted(set(tfs) & set(tfs_p))
        ie = [tfs.index(t) for t in common]
        ip = [tfs_p.index(t) for t in common]
        Be, Bp, tfs = Be[ie], Bp[ip], common
        say(f"    rosters differed; restricted to the {len(common)} factors measured on both sides")

    # ---- Z1 ------------------------------------------------------------------------------------
    say()
    say("Z1 DOES THE PROMOTER SIDE JOIN?")
    dp = Bp.sum(0)
    cover = float((dp > 0).mean())
    say(f"     {Bp.shape[0]} factors x {Bp.shape[1]:,} promoters, {100*Bp.mean():.1f}% of cells bound")
    say(f"     factors per promoter: median {np.median(dp):.0f} "
        f"(IQR {np.percentile(dp,25):.0f}-{np.percentile(dp,75):.0f}); "
        f"{cover:.1%} of promoters carry at least one")
    z1 = bool(cover >= MIN_PROM_COVER)
    GG.verdict(z1, emit=say,
               if_true=f"Z1 PASS -- {cover:.1%} of promoters carry a measured factor, so the "
                       f"overlap is defined for essentially every pair",
               if_false=f"Z1 FAIL -- only {cover:.1%} of promoters carry one; the overlap would be "
                        f"zero for most pairs by absence rather than by biology")

    # ---- features -------------------------------------------------------------------------
    say()
    say("   building features")
    OV = overlap_features(Be, Bp, e_idx, g_idx)
    sets, grow, midf, gom = L183.network(S, lambda *_: None)
    import gzip
    nb = json.load(gzip.open(L183.BUNDLE))
    tgt_count = Counter()
    for r in nb["reg"]:
        tgt_count[nb["names"][int(r[0])].upper()] += 1
    RC = chip_regulator_features(Be, tfs, sets, grow, e_idx, g_idx, tgt_count)

    rows = SC.load_benchmark(lambda *_: None)
    dhs, h3k = {}, {}
    for r in rows:
        k = f"{r['chrom']}:{r['chromStart']}-{r['chromEnd']}"
        try:
            dhs[k] = float(r.get("DHS.RPM") or 0)
            h3k[k] = float(r.get("H3K27ac.RPM") or 0)
        except ValueError:
            pass
    ek = [str(k) for k in S["el_key"]]
    IN = {"acc_dhs": np.array([np.log10(1 + dhs.get(ek[int(i)], 0.0)) for i in e_idx]),
          "acc_h3k": np.array([np.log10(1 + h3k.get(ek[int(i)], 0.0)) for i in e_idx]),
          "log_n_bound_el": np.log10(1.0 + Be.sum(0))[e_idx].astype(float)}

    E, FAM, _ = L178.element_frame(S, "el", lambda *_: None)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    for c in P:
        P[c] = np.nan_to_num(P[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xbase = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    Xd = np.column_stack([P["log_dist"]])
    occ = np.exp(L173.occupancy(S["el_LZ"], e_idx, g_idx, len(S["gn_key"]),
                                np.exp(L173._logsumexp(S["bg_LZ"].astype(np.float64), axis=1))
                                / float(S["bg_bp"])))
    N183 = L183.net_features(S, occ, sets, grow, midf, lambda *_: None)
    say(f"    base {Xbase.shape[1]} columns; overlap {len(OVERLAP)}; pmi {len(PMI)}; "
        f"chip-regulator {len(REGCHIP)}; intrinsic {len(INTRINSIC)}")

    def cols(d, names_):
        return [d[c] for c in names_]

    res = {}
    res["distance"] = run(Xd, y, chrom, g_idx, jitter, "distance", say)
    res["base"] = run(Xbase, y, chrom, g_idx, jitter, "base stack", say)
    res["l183_motif"] = run(np.column_stack([Xbase] + cols(N183, L183.BIND_COLS)),
                            y, chrom, g_idx, jitter, "loop 183 block (MOTIF presence)", say)
    res["chip_reg"] = run(np.column_stack([Xbase] + cols(RC, REGCHIP)),
                          y, chrom, g_idx, jitter, "same block, MEASURED presence", say)
    res["+overlap"] = run(np.column_stack([Xbase] + cols(OV, OVERLAP)),
                          y, chrom, g_idx, jitter, "+ element-promoter shared factors", say)
    res["+pmi"] = run(np.column_stack([Xbase] + cols(OV, OVERLAP + PMI)),
                      y, chrom, g_idx, jitter, "+ shared factors as PMI over both degrees", say)
    res["+intrinsic"] = run(np.column_stack([Xbase] + cols(IN, INTRINSIC)),
                            y, chrom, g_idx, jitter, "+ accessibility and bound-factor count", say)
    res["ALL"] = run(np.column_stack([Xbase] + cols(OV, OVERLAP + PMI) + cols(RC, REGCHIP)
                                     + cols(IN, INTRINSIC)),
                     y, chrom, g_idx, jitter, "everything measured", say)

    # ---- Z2..Z7 --------------------------------------------------------------------------------
    def gate(tag, a, b, title, if_t, if_f, use_ap=True):
        d = L173.paired(res[a], res[b])
        say()
        say(title)
        say(f"     {a} vs {b}   {L173.fmt(d)}")
        ok = L173.gate_pair(d, use_ap=use_ap)
        GG.verdict(ok, emit=say, if_true=f"{tag} PASS -- {if_t}", if_false=f"{tag} FAIL -- {if_f}")
        return ok, d

    z2, d2 = gate("Z2", "chip_reg", "l183_motif",
                  "Z2 MEASURED PRESENCE OR PREDICTED PRESENCE?",
                  "asking whether the gene's regulators are ACTUALLY bound beats asking whether "
                  "their motifs are present",
                  "measured presence does no better than predicted presence, even though a motif "
                  "only predicts real binding at AUC 0.62")
    z3, d3 = gate("Z3", "+overlap", "base",
                  "Z3 DOES THE ELEMENT-PROMOTER SHARED-FACTOR OVERLAP HELP?",
                  "sharing measured factors with this gene's own promoter adds over the sequence "
                  "stack",
                  "the shared-factor overlap adds nothing")
    z4, d4 = gate("Z4", "+pmi", "base",
                  "Z4 IS IT THE OVERLAP OR THE TWO MARGINALS?",
                  "the overlap survives being normalised by the two degrees it could have come "
                  "from",
                  "once the overlap is expressed against what the two degrees predict, the block "
                  "stops helping -- it was busy-ness at both ends")

    say()
    say("Z5 WHOSE PROMOTER?")
    pr1, pap = [], []
    for s in SEEDS:
        perm = np.random.default_rng(8000 + s).permutation(Bp.shape[1])
        OVp = overlap_features(Be, Bp, e_idx, g_idx, perm=perm)
        Xp = np.column_stack([Xbase] + cols(OVp, OVERLAP + PMI))
        fold = L173.folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if te.sum() == 0 or y[tr].sum() == 0:
                continue
            m = gbm(s)
            m.fit(np.nan_to_num(Xp[tr]), y[tr])
            sc[te] = m.predict_proba(np.nan_to_num(Xp[te]))[:, 1]
        pr1.append(L173.within_gene(sc, y, g_idx, jitter)[0])
        pap.append(average_precision_score(y, sc))
    res["+pmi_perm"] = dict(r1=np.array(pr1), ap=np.array(pap), mrr=np.zeros(len(SEEDS)))
    say(f"    {'+pmi, promoter sets permuted':42} R@1 {np.mean(pr1):.4f}   AUPRC {np.mean(pap):.4f}")
    d5 = L173.paired(res["+pmi"], res["+pmi_perm"])
    say(f"     real vs permuted   {L173.fmt(d5)}")
    z5 = L173.gate_pair(d5, use_ap=False)
    GG.verdict(z5, emit=say,
               if_true="Z5 PASS -- it matters WHOSE promoter the element shares factors with",
               if_false="Z5 FAIL -- any promoter's factor set works as well, so the block reads how "
                        "many factors are around and not which gene they belong to")

    say()
    say("Z6 THE ELEMENT-INTRINSIC ARM (predeclared as bounded)")
    d6 = L173.paired(res["+intrinsic"], res["base"])
    say(f"     +intrinsic vs base   {L173.fmt(d6)}")
    r1_ok = L173.gate_pair(d6, use_ap=False)
    ap_ok = d6["n_ap_pass"] >= MIN_SEEDS
    z6 = bool(ap_ok and not r1_ok)
    GG.verdict(z6, emit=say,
               if_true="Z6 PASS -- accessibility and crowding move the pooled ranking and not the "
                       "within-gene choice, which is exactly what an element-intrinsic column can "
                       "do given loop 178's 0.4422 ceiling",
               if_false=f"Z6 FAIL -- AUPRC bar {'cleared' if ap_ok else 'missed'}, R@1 bar "
                        f"{'cleared' if r1_ok else 'missed'}; the pattern is not the one an "
                        f"element-intrinsic column should show and needs explaining")

    say()
    say("Z7 THE DECISIVE ONE")
    best = max((k for k in res if k not in ("distance", "+pmi_perm")),
               key=lambda k: res[k]["r1"].mean())
    d7 = L173.paired(res[best], res["distance"])
    say(f"     best arm {best} at R@1 {res[best]['r1'].mean():.4f} / AUPRC "
        f"{res[best]['ap'].mean():.4f} against distance {res['distance']['r1'].mean():.4f} / "
        f"{res['distance']['ap'].mean():.4f}")
    say(f"     {L173.fmt(d7)}")
    say(f"     for reference, loop 183's binding-tier arm reached R@1 0.6271 / AUPRC 0.3377")
    z7 = L173.gate_pair(d7)
    GG.verdict(z7, emit=say,
               if_true=f"Z7 PASS -- {best} clears the bar every stage-two loop has been held to",
               if_false="Z7 FAIL -- stage two is still distance")

    say()
    say("Z8 WHAT THIS CANNOT SHOW")
    say("     ChIP is crosslinked proximity in a population. A factor tethered indirectly at an")
    say("     element looks identical to one that binds it, and the overlap block cannot tell a")
    say("     shared direct contact from two independent tetherings.")
    say("     191 of 723 factor names have a K562 track and ENCODE's roster is enriched for the")
    say("     factors people study, so the overlap is computed over a biased slice of the proteome.")
    say("     The dinucleotide shuffle that gated every sequence loop has no meaning here --")
    say("     shuffling DNA does not move a peak call -- so Z5's permutation is the control that")
    say("     carries the weight, and it tests only the gene axis.")
    say("     Accessibility, crowding and validated-enhancer status are all properties of the same")
    say("     4,482 elements the screens pre-selected for looking regulatory.")
    z8 = True
    say(f"     Z8 {'PASS' if z8 else 'FAIL'}")

    gates = {"Z1": z1, "Z2": z2, "Z3": z3, "Z4": z4, "Z5": z5, "Z6": z6, "Z7": z7, "Z8": z8}
    man = RM.manifest(inputs=[Path("colab/data/tf_domains.json")],
                      available=int(len(y)), used=int(len(y)), selection="loop 173's pairs",
                      seed=L173.TIE_SEED,
                      controls=["the overlap re-expressed as PMI against both degrees",
                                "promoter factor sets permuted across genes, counts held exactly",
                                "the element-intrinsic arm gated on the pattern loop 178 predicts",
                                "measured presence put head to head with motif presence"],
                      note="measured co-binding and accessibility on the stage-two task")
    out = dict(test="enhancer co-binding", gates=gates,
               n_factors=len(tfs), promoter_cover=cover,
               frac_cells_bound_elements=float(Be.mean()),
               frac_cells_bound_promoters=float(Bp.mean()),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "ap")}
                     for k, v in res.items()},
               deltas={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                           for kk, vv in d.items()}
                       for k, d in (("Z2", d2), ("Z3", d3), ("Z4", d4), ("Z5", d5),
                                    ("Z6", d6), ("Z7", d7))},
               best_arm=best, manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    out["log"] = log
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
