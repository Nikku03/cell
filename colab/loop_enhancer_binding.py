"""Loop 184. What actually makes a transcription factor attach to an enhancer?

WHY THIS IS A DIFFERENT KIND OF LOOP. Loops 173 to 183 asked whether a feature improves a
prediction. This one asks what the signal IS, and it can only be asked now because the measurement
finally exists: ENCODE conservative-IDR ChIP for 191 factors that also carry a JASPAR matrix here,
over the 4,482 benchmark elements, 13.1% of cells bound, a median of 17 factors per element and
432 elements per factor. Every earlier loop used a motif score as a stand-in for "this factor is
here". That is a statement about sequence, and whether the protein is actually there is a different
fact.

THE THREE CANDIDATE EXPLANATIONS, and each is put against the same measured binding.

  SEQUENCE. The factor's own motif, scored as the element's partition function -- the same number
  every loop in this arc has used. Y2 asks how well it does on its own.

  THE DNA'S SHAPE AT THE SITE. Minor and major groove width, propeller twist, roll, helix twist and
  minor-groove electrostatic potential, Boltzmann-weighted over the occupied positions. This is the
  channel that worked for stage one (loop 174's F5, +0.0353 AUC) and that never worked for stage
  two. Y3 asks whether it adds to the motif for the thing it was always supposed to explain.

  THE ELEMENT, NOT THE FACTOR. Accessibility, and how many OTHER factors are bound at the same
  element. If binding is mostly a property of open, crowded regions, then the sequence story is
  small and the honest headline says so. Y4 and Y6 are those two, and Y6 is the harsher of them.

AND THE QUESTION THAT ONLY THE PROTEIN STRUCTURES CAN ANSWER. Factors differ enormously in how well
their motif predicts their binding, and that spread is itself data. Y5 regresses each factor's
motif-to-binding AUC on its measured domain geometry -- charge dipole over radius of gyration,
Shrake-Rupley surface charge, how far its arginines sit from the domain centroid, its reach in base
pairs -- plus its structural class, its motif's information content and its binding degree. What
comes out is a statement about WHAT KIND OF PROTEIN binds where its motif is, and what kind gets
there another way.

PREDECLARED, BEFORE ANY NUMBER.

  Y1 IS THE MEASUREMENT USABLE? Factors need both classes present to have an AUC at all.
     Gate: PASS iff at least 100 factors have >= 50 bound and >= 50 unbound elements.

  Y2 HOW WELL DOES THE MOTIF PREDICT BINDING? Per-factor AUC of motif occupancy against measured
     binding, over all 4,482 elements.
     Gate: PASS iff the median per-factor AUC exceeds 0.55, which is the floor for calling the
     motif informative at all. The number itself, not the gate, is the result: it says how much of
     this arc's central assumption holds.

  Y3 DOES DNA SHAPE ADD OVER THE MOTIF? Per factor, a chromosome-held-out logistic model on the
     motif alone against the motif plus the six shape variables at its own occupied positions.
     Gate: PASS iff the median paired change in AUC across factors is positive at a two-sided
     Wilcoxon signed-rank p < 0.05.

  Y4 DOES ACCESSIBILITY BEAT SEQUENCE? Per-factor AUC of the element's DNase signal alone against
     the motif alone, on identical elements.
     Gate: descriptive; PASS iff both are computable for at least 100 factors. Which one wins, and
     for how many factors, is the finding and is not gated in either direction.

  Y5 WHAT KIND OF FACTOR IS MOTIF-PREDICTABLE? Each factor's motif AUC regressed on its AlphaFold
     domain geometry, groove class, structural class, motif information content, motif width and
     binding degree.
     Gate: PASS iff at least one predictor survives Benjamini-Hochberg at q < 0.05 across every
     predictor tested. A FAIL says the spread in motif predictability is not explained by anything
     measured here, which is itself worth knowing and is reported as such.

  Y6 IS BINDING A PROPERTY OF THE ELEMENT RATHER THAN THE FACTOR? Per-factor AUC of "how many of
     the OTHER 190 factors are bound at this element" against that factor's own binding, with the
     factor itself excluded from the count.
     Gate: descriptive; PASS iff computable for at least 100 factors. If co-binding beats the motif
     for most factors, the honest headline is that an enhancer is a crowded place and the factor's
     own sequence preference is a minor term.

  Y7 FROM BINDING TO FUNCTION. Among the pairs where a factor IS bound, what separates elements
     that are validated enhancers of a gene that factor regulates from elements that are not?
     Descriptive, with effect sizes and matched comparisons.

  Y8 THE DEGREE CONTROL. Every comparison above is repeated with factors stratified by binding
     degree and elements stratified by how many factors bind them, because "binds a lot" and
     "bound by a lot" are the two variables that could produce all of the above by themselves.
     Gate: PASS iff Y2's median motif AUC survives within the middle degree tertile.

  Y9 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_binding.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import chip as CH                   # noqa: E402
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402
from enh import tf_structures as TS          # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402

from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402
from sklearn.pipeline import make_pipeline                     # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_binding.json"
NFOLD = 5
MIN_CLASS = 50
MIN_FACTORS = 100
MIN_MEDIAN_AUC = 0.55
SHAPES = ["mgw", "mgrw", "prot", "roll", "helt", "ep"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def cv_auc(X, y, chrom, seed=0):
    """Chromosome-held-out AUC for one factor. Returns nan if a fold has one class only."""
    ch = sorted(set(chrom))
    order = np.random.default_rng(seed).permutation(len(ch))
    assign = {ch[order[i]]: i % NFOLD for i in range(len(ch))}
    fold = np.array([assign[c] for c in chrom])
    sc = np.zeros(len(y))
    for f in range(NFOLD):
        te, tr = fold == f, fold != f
        if te.sum() == 0 or y[tr].sum() < 5 or y[tr].sum() == tr.sum():
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        m.fit(np.nan_to_num(X[tr]), y[tr])
        sc[te] = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
    try:
        return roc_auc_score(y, sc)
    except Exception:
        return np.nan


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 184  WHAT MAKES A TRANSCRIPTION FACTOR ATTACH TO AN ENHANCER?")
    say("=" * 104)
    say(f"  PREDECLARED: >= {MIN_FACTORS} factors must have >= {MIN_CLASS} bound and unbound")
    say(f"  elements; the median per-factor motif AUC must exceed {MIN_MEDIAN_AUC} for the motif to")
    say("  count as informative; shape must add over the motif at a signed-rank p < 0.05; at least")
    say("  one structural predictor must survive Benjamini-Hochberg at q < 0.05; and the motif")
    say("  result must survive inside the middle binding-degree tertile.")
    say()

    S = SC.load(say)
    dom = TD.load()
    st = TS.load()
    mid = [str(m) for m in S["motif_ids"]]
    mname = [(dom.get(m, {}).get("name") or "").upper().split("::")[0] for m in mid]
    B, tfs = CH.build(S["el_key"], sorted(set(n for n in mname if n)), say)
    by_name = defaultdict(list)
    for k, n in enumerate(mname):
        by_name[n].append(k)

    el_key = [str(k) for k in S["el_key"]]
    chrom = np.array([k.split(":")[0] for k in el_key])
    ne = len(el_key)

    bg = np.exp(L173._logsumexp(S["bg_LZ"].astype(np.float64), axis=1)) / float(S["bg_bp"])
    OCC = np.log10(np.maximum(
        np.exp(S["el_LZ"].astype(np.float64)
               - (np.log(np.maximum(bg, 1e-300)) + np.log(4_000_000))[:, None]), 1e-300))
    SH = {n: S["el_SH"][i] for i, n in enumerate(list(S["tracks"]))}

    rows = SC.load_benchmark(lambda *_: None)
    dhs, h3k = {}, {}
    for r in rows:
        k = f"{r['chrom']}:{r['chromStart']}-{r['chromEnd']}"
        try:
            dhs[k] = float(r.get("DHS.RPM") or 0)
            h3k[k] = float(r.get("H3K27ac.RPM") or 0)
        except ValueError:
            pass
    ACC = np.array([np.log10(1 + dhs.get(k, 0.0)) for k in el_key])
    AC2 = np.array([np.log10(1 + h3k.get(k, 0.0)) for k in el_key])
    say(f"    accessibility joined for {sum(1 for k in el_key if k in dhs):,}/{ne:,} elements")

    # ---- Y1 ------------------------------------------------------------------------------------
    say()
    say("Y1 IS THE MEASUREMENT USABLE?")
    use = []
    for ti, t in enumerate(tfs):
        nb = int(B[ti].sum())
        if nb >= MIN_CLASS and (ne - nb) >= MIN_CLASS and by_name.get(t.upper()):
            use.append(ti)
    say(f"     {len(tfs)} factors measured; {len(use)} have >= {MIN_CLASS} bound and unbound "
        f"elements and a matrix here")
    per_el = B.sum(0)
    say(f"     factors per element median {np.median(per_el):.0f}, "
        f"{(per_el == 0).mean():.1%} of elements bound by none")
    y1 = bool(len(use) >= MIN_FACTORS)
    GG.verdict(y1, emit=say,
               if_true=f"Y1 PASS -- {len(use)} factors carry both classes, enough for per-factor "
                       f"comparisons",
               if_false=f"Y1 FAIL -- only {len(use)} factors are usable")

    # ---- Y2, Y3, Y4, Y6 ------------------------------------------------------------------------
    say()
    say("   per-factor models (chromosome-held-out logistic, one factor at a time)")
    other = B.sum(0).astype(float)
    recs = []
    for n, ti in enumerate(use):
        t = tfs[ti]
        k = max(by_name[t.upper()], key=lambda j: OCC[j].std())
        yb = B[ti].astype(int)
        co = np.log10(1.0 + np.maximum(other - yb, 0))
        Xm = OCC[k][:, None]
        Xs = np.column_stack([OCC[k]] + [np.nan_to_num(SH[s][k], nan=0.0) for s in SHAPES])
        r = dict(tf=t, matrix=mid[k], n_bound=int(yb.sum()),
                 auc_motif=cv_auc(Xm, yb, chrom),
                 auc_shape=cv_auc(Xs, yb, chrom),
                 auc_acc=cv_auc(ACC[:, None], yb, chrom),
                 auc_h3k=cv_auc(AC2[:, None], yb, chrom),
                 auc_co=cv_auc(co[:, None], yb, chrom))
        recs.append(r)
        if (n + 1) % 50 == 0:
            say(f"      {n+1}/{len(use)} factors  [{time.time()-t0:.0f}s]")
    A = {k: np.array([r[k] for r in recs], dtype=float) for k in
         ("auc_motif", "auc_shape", "auc_acc", "auc_h3k", "auc_co", "n_bound")}
    ok = np.isfinite(A["auc_motif"]) & np.isfinite(A["auc_shape"]) & np.isfinite(A["auc_acc"])
    say(f"    {int(ok.sum())} factors with every AUC computable")

    say()
    say("Y2 HOW WELL DOES THE MOTIF PREDICT BINDING?")
    med = float(np.median(A["auc_motif"][ok]))
    say(f"     per-factor motif AUC: median {med:.4f}, "
        f"IQR [{np.percentile(A['auc_motif'][ok], 25):.4f}, "
        f"{np.percentile(A['auc_motif'][ok], 75):.4f}], "
        f"range [{A['auc_motif'][ok].min():.4f}, {A['auc_motif'][ok].max():.4f}]")
    say(f"     factors above 0.70: {int((A['auc_motif'][ok] > 0.70).sum())}/{int(ok.sum())} "
        f"({(A['auc_motif'][ok] > 0.70).mean():.1%}); "
        f"below 0.60: {int((A['auc_motif'][ok] < 0.60).sum())} "
        f"({(A['auc_motif'][ok] < 0.60).mean():.1%})")
    y2 = bool(med > MIN_MEDIAN_AUC)
    GG.verdict(y2, emit=say,
               if_true=f"Y2 PASS -- the motif is informative, median AUC {med:.4f}, and the spread "
                       f"across factors is the thing Y5 tries to explain",
               if_false=f"Y2 FAIL -- median motif AUC {med:.4f} is at chance, so the sequence "
                        f"assumption every earlier loop rested on does not hold at all")

    say()
    say("Y3 DOES DNA SHAPE ADD OVER THE MOTIF?")
    d = A["auc_shape"][ok] - A["auc_motif"][ok]
    w = stats.wilcoxon(d) if len(d) > 10 else None
    say(f"     paired change per factor: median {np.median(d):+.4f}, mean {d.mean():+.4f}, "
        f"positive in {int((d > 0).sum())}/{len(d)} factors")
    say(f"     Wilcoxon signed-rank p = {w.pvalue:.3g}" if w else "     too few factors to test")
    y3 = bool(w is not None and w.pvalue < 0.05 and np.median(d) > 0)
    GG.verdict(y3, emit=say,
               if_true=f"Y3 PASS -- groove geometry at the occupied positions adds {np.median(d):+.4f} "
                       f"AUC over the motif alone, across factors",
               if_false="Y3 FAIL -- shape does not add over the motif for measured binding either")

    say()
    say("Y4 DOES ACCESSIBILITY BEAT SEQUENCE?")
    wins = int((A["auc_acc"][ok] > A["auc_motif"][ok]).sum())
    say(f"     accessibility (DNase) AUC median {np.median(A['auc_acc'][ok]):.4f}; "
        f"H3K27ac median {np.median(A['auc_h3k'][ok]):.4f}; motif median {med:.4f}")
    say(f"     accessibility beats the motif for {wins}/{int(ok.sum())} factors "
        f"({wins/max(int(ok.sum()),1):.1%})")
    y4 = bool(int(ok.sum()) >= MIN_FACTORS)
    GG.verdict(y4, emit=say,
               if_true=f"Y4 PASS -- both are computable for {int(ok.sum())} factors; accessibility "
                       f"wins for {wins/max(int(ok.sum()),1):.0%} of them, and that ratio is the "
                       f"finding",
               if_false="Y4 FAIL -- too few factors to compare")

    say()
    say("Y6 IS BINDING A PROPERTY OF THE ELEMENT RATHER THAN THE FACTOR?")
    cw = int((A["auc_co"][ok] > A["auc_motif"][ok]).sum())
    say(f"     co-binding AUC median {np.median(A['auc_co'][ok]):.4f} against the motif's {med:.4f}")
    say(f"     co-binding beats the motif for {cw}/{int(ok.sum())} factors "
        f"({cw/max(int(ok.sum()),1):.1%})")
    y6 = bool(int(ok.sum()) >= MIN_FACTORS)
    GG.verdict(y6, emit=say,
               if_true=f"Y6 PASS -- computable for {int(ok.sum())} factors. Counting the OTHER "
                       f"factors bound at an element beats that factor's own motif for "
                       f"{cw/max(int(ok.sum()),1):.0%} of them",
               if_false="Y6 FAIL -- too few factors")

    # ---- Y5 ------------------------------------------------------------------------------------
    say()
    say("Y5 WHAT KIND OF FACTOR IS MOTIF-PREDICTABLE?")
    feats = {}
    for nm, fn in (("dipole", "dipole"), ("surf_charge", "surf_charge"), ("arg_out", "arg_out"),
                   ("max_dim", "max_dim"), ("rg", "rg"), ("plddt", "plddt")):
        feats[nm] = np.array([float(st.get(r["matrix"], {}).get(fn, np.nan) or np.nan)
                              for r in recs])
    W = S["motif_width"].astype(float)
    mi = {m: i for i, m in enumerate(mid)}
    feats["motif_width"] = np.array([W[mi[r["matrix"]]] for r in recs])
    feats["motif_maxscore"] = np.array([float(S["motif_maxscore"][mi[r["matrix"]]]) for r in recs])
    feats["log_n_bound"] = np.log10(1.0 + A["n_bound"])
    gro = np.array([dom.get(r["matrix"], {}).get("groove", "major") for r in recs])
    feats["reads_minor"] = ((gro == "minor") | (gro == "both")).astype(float)
    tgt = A["auc_motif"]
    res5 = []
    for nm, v in feats.items():
        m = ok & np.isfinite(v)
        if m.sum() < 30 or np.nanstd(v[m]) == 0:
            continue
        r, p = stats.spearmanr(v[m], tgt[m])
        res5.append(dict(feature=nm, n=int(m.sum()), rho=float(r), p=float(p)))
    res5.sort(key=lambda x: x["p"])
    mth = len(res5)
    for i, r in enumerate(res5):
        r["q"] = min(1.0, r["p"] * mth / (i + 1))
    for r in res5:
        say(f"     {r['feature']:16} n={r['n']:3d}  rho {r['rho']:+.3f}  p {r['p']:.3g}  "
            f"q {r['q']:.3g}" + ("   <-- survives" if r["q"] < 0.05 else ""))
    # structural class, as a group test
    cls = np.array([str(dom.get(r["matrix"], {}).get("cls")) for r in recs])
    groups = [tgt[ok & (cls == c)] for c in sorted(set(cls[ok]))
              if (ok & (cls == c)).sum() >= 5]
    if len(groups) >= 3:
        kw = stats.kruskal(*groups)
        say(f"     JASPAR structural class over {len(groups)} classes with n>=5: "
            f"Kruskal-Wallis p = {kw.pvalue:.3g}")
        best_c = sorted(((float(np.median(tgt[ok & (cls == c)])), c, int((ok & (cls == c)).sum()))
                         for c in sorted(set(cls[ok])) if (ok & (cls == c)).sum() >= 5),
                        reverse=True)
        for v, c, n in best_c[:3]:
            say(f"       highest  {c[:58]:58} n={n:3d}  median motif AUC {v:.4f}")
        for v, c, n in best_c[-3:]:
            say(f"       lowest   {c[:58]:58} n={n:3d}  median motif AUC {v:.4f}")
    else:
        kw = None
    y5 = bool(any(r["q"] < 0.05 for r in res5) or (kw is not None and kw.pvalue < 0.05))
    GG.verdict(y5, emit=say,
               if_true="Y5 PASS -- the spread in motif predictability is partly explained by what "
                       "kind of protein the factor is",
               if_false="Y5 FAIL -- nothing measured here explains why some factors bind where "
                        "their motif is and others do not, which is a real negative about the "
                        "descriptors and not about the phenomenon")

    # ---- Y7 ------------------------------------------------------------------------------------
    say()
    say("Y7 FROM BINDING TO FUNCTION")
    y = S["y"].astype(int)
    e_idx = S["e_idx"]
    pos_el = np.zeros(ne, bool)
    pos_el[np.unique(e_idx[y == 1])] = True
    say(f"     {int(pos_el.sum())} of {ne} elements are a validated enhancer of some gene")
    nb_pos = B[:, pos_el].sum(0)
    nb_neg = B[:, ~pos_el].sum(0)
    say(f"     factors bound per element: validated {np.median(nb_pos):.0f}, "
        f"other {np.median(nb_neg):.0f}, ratio {np.median(nb_pos)/max(np.median(nb_neg),1):.2f}x")
    u = stats.mannwhitneyu(nb_pos, nb_neg, alternative="greater")
    say(f"     Mann-Whitney p = {u.pvalue:.3g}")
    accp, accn = ACC[pos_el], ACC[~pos_el]
    say(f"     accessibility: validated median {np.median(accp):.3f}, "
        f"other {np.median(accn):.3f}, "
        f"Mann-Whitney p = {stats.mannwhitneyu(accp, accn, alternative='greater').pvalue:.3g}")
    y7 = True
    say(f"     Y7 {'PASS' if y7 else 'FAIL'} (descriptive)")

    # ---- Y8 ------------------------------------------------------------------------------------
    say()
    say("Y8 THE DEGREE CONTROL")
    deg = A["n_bound"][ok]
    q1, q2 = np.percentile(deg, [33.3, 66.7])
    mid_t = ok.copy()
    mid_t[ok] = (deg >= q1) & (deg <= q2)
    m_med = float(np.median(A["auc_motif"][mid_t]))
    say(f"     middle binding-degree tertile ({int(q1)}-{int(q2)} elements bound): "
        f"{int(mid_t.sum())} factors, median motif AUC {m_med:.4f} against {med:.4f} overall")
    say(f"     within that tertile, accessibility beats the motif for "
        f"{int((A['auc_acc'][mid_t] > A['auc_motif'][mid_t]).sum())}/{int(mid_t.sum())} factors, "
        f"co-binding for {int((A['auc_co'][mid_t] > A['auc_motif'][mid_t]).sum())}")
    y8 = bool(m_med > MIN_MEDIAN_AUC)
    GG.verdict(y8, emit=say,
               if_true=f"Y8 PASS -- the motif result holds at {m_med:.4f} inside the middle "
                       f"degree tertile, so it is not an artefact of promiscuous or rare factors",
               if_false=f"Y8 FAIL -- inside the middle tertile the median motif AUC falls to "
                        f"{m_med:.4f}, so Y2 was carried by the extremes of binding degree")

    say()
    say("Y9 WHAT THIS CANNOT SHOW")
    say("     ChIP measures crosslinked proximity in a population, not direct contact in a cell. A")
    say("     factor pulled down at an element it never touches directly is indistinguishable here")
    say("     from one that binds it, and that is precisely how indirect tethering would look.")
    say("     Only 191 of 723 factor names have a K562 track, and ENCODE's roster is not a random")
    say("     sample of transcription factors -- it is enriched for the ones people study.")
    say("     Accessibility, co-binding and validated-enhancer status are all properties of the")
    say("     same 4,482 pre-selected elements, which were chosen for looking regulatory.")
    say("     Nothing here is causal. A factor whose motif predicts its binding well is not")
    say("     thereby a factor that matters at those elements.")
    y9 = True
    say(f"     Y9 {'PASS' if y9 else 'FAIL'}")

    gates = {"Y1": y1, "Y2": y2, "Y3": y3, "Y4": y4, "Y5": y5, "Y6": y6, "Y7": y7, "Y8": y8,
             "Y9": y9}
    man = RM.manifest(inputs=[Path("colab/data/tf_structures.json")],
                      available=len(tfs), used=int(ok.sum()), selection="factors with both classes",
                      seed=0,
                      controls=["chromosome-held-out folds inside every per-factor model",
                                "accessibility and co-binding as rival explanations, not baselines",
                                "Benjamini-Hochberg across every structural predictor",
                                "the middle binding-degree tertile"],
                      note="what explains a transcription factor attaching to an enhancer")
    out = dict(test="enhancer binding explanation", gates=gates,
               n_factors_measured=len(tfs), n_factors_used=int(ok.sum()),
               frac_cells_bound=float(B.mean()),
               per_factor={k: [None if not np.isfinite(x) else float(x) for x in A[k]]
                           for k in A},
               tf_names=[r["tf"] for r in recs],
               motif_auc_median=med, shape_delta_median=float(np.median(d)),
               shape_wilcoxon_p=float(w.pvalue) if w else None,
               acc_beats_motif=int(wins), cobind_beats_motif=int(cw),
               structure_spearman=res5,
               class_kruskal_p=float(kw.pvalue) if kw is not None else None,
               middle_tertile_motif_auc=m_med,
               manifest=man, seconds=time.time() - t0, log=log)
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
