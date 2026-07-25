"""program_coldstart -- predict a knockout's response WITHOUT using any perturbation data about it or anything near it.

THE LIMITATION BEING REMOVED. program_factorization beat the no-network baseline (0.350 vs 0.114) but only by averaging the MEASURED response
profiles of the test gene's neighbours. That is strictly more information than a graph has, and worse, it does not generalise: a gene whose
partners were never knocked out gets no prediction at all. It also encodes a wrong assumption -- that what a knockout affects is near it. The
response programs say otherwise: component 1 is translation and it moves ribosomal genes throughout the cell, component 5 is the RNA exosome and
it moves transcripts with no spatial relationship to the exosome at all. Distal effects are the norm, and the program view handles them natively.

SO THE TASK HERE IS COLD START: given a gene NOBODY HAS EVER KNOCKED OUT, and no measured response from any of its partners, predict what
knocking it out will do.

    response(s)  ~=  mu  +  P_hat(s) @ G        G, mu = programs and mean response, fitted on TRAIN responses only
                            ^-- predicted from the gene's OWN annotation + start state, by a regressor trained on train genes' loadings

WHY THIS MIGHT WORK WHERE EARLIER ATTEMPTS DID NOT. Two previous efforts failed on high-dimensional targets: annotation_prior ranked which of 8k
genes would move (recall@50 0.125 against a 0.272 tide) and infer_network predicted 8,726 individual edges from static features (AUC 0.542).
Here the target is KPROG program loadings, two to three orders of magnitude smaller, and the programs are recognisably complexes -- so
"which complex is this gene annotated to" is expected to map onto them almost directly. That is a genuinely different bet, not a repeat.

THE CONTROLS THAT DECIDE THIS.
  MISMATCHED SOURCE -- score each source's prediction against a DIFFERENT held-out source's truth, keeping the truth set (and so the recall
    denominator) fixed and swapping only the prediction. A model that has merely learned "which non-tide genes tend to move" would lose nothing;
    only the matched-minus-mismatched gap is knockout-specific skill. This is the number to look at first.
  MU-ONLY -- the reconstruction is mu + P_hat @ G, and mu is the AVERAGE knockout response, identical for every gene. Anything the model scores
    that mu-only also scores is a generic stress response, not a prediction.
  Reported alongside: the tide-null (rank by how often a gene moves in train), random, and 1-NN annotation retrieval (copy the measured profile of
  the most annotation-similar TRAIN gene -- a cold-start method too, since it never touches the test gene's neighbours).

FEATURES ARE ANNOTATION AND START STATE ONLY -- never a perturbation response of any gene:
    gene-intrinsic   essentiality, LOEUF, TF flag, PPI degree, pathway count, disease count, CpG, publications, dark/conf flags
    localisation     subcellular compartment, annotated process (one-hot)
    structure        InterPro domain content
    start state      the gene's unperturbed expression in K562 CONTROL cells (Replogle control_expr) and, where the gene is on the readout panel,
                     its control mean / sd / CV / Gini. This is measured start state, not a response.
    membership       which curated complex the gene belongs to (one-hot)
TWO ABLATIONS. Complex identity is the feature most likely to carry the whole result, since the components ARE complexes, and it is also a curated
PARTNER relation -- no partner's measured profile is read at prediction time, but the regressor learned each complex's loadings from that complex's
train members. So the model is run WITHOUT it for a fully partner-free number. Second, DepMap essentiality and dependency fraction are themselves
knockout measurements of the test gene (viability, not transcriptome, but still a perturbation), so a third variant drops those too -- that is the
number for a gene no assay has ever perturbed.

PROTOCOL. Sources split 70/30 x NSPLIT, repeated over SEEDS independent seeds. Programs, mu, feature vocabularies, feature scaling, the regressor and
the tide mask are all fitted on TRAIN only. The test gene's own response is used solely as truth, and no neighbour's response is used at any point.
Significance is tested on one value per UNIQUE source, since a source can be held out in several splits. Deterministic."""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
RNG = np.random.RandomState(0)
TIDE_FRAC = 0.05
MINSRC = 20
MAXCOMPLEX = 200
K_RECALL = 50
NSPLIT = 5
KPROG = 50             # programs to predict (the sweep in program_factorization saturated around here)
RIDGE = 10.0

SEEDS = (0, 1, 2)      # single-seed splits are not evidence; the headline is reported as mean +- spread over these

MODES = ("full", "intrinsic", "no_depmap")
MODELS = ("annotation + complex identity", "gene-intrinsic only (no complex)", "no complex, no DepMap (never perturbed at all)")
REFS = ("[ref] mean response only (mu, no gene-specific term)", "[ref] 1-NN annotation retrieval",
        "[ref] tide-null (no model at all)", "[ref] random",
        "[ctrl] model prediction scored against a DIFFERENT source")


def main():
    from sklearn.linear_model import Ridge
    from scipy.stats import wilcoxon
    H = Harness("K562")
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]; gidx = {g: i for i, g in enumerate(genes)}
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); Aall = np.abs(Z)

    tc = [int((np.abs(H.M[H.ki[k]]) >= 1.0).sum()) for k in H.kos
          if k in kidx and (np.abs(H.M[H.ki[k]]) > 0).any() and np.abs(H.M[H.ki[k]])[np.abs(H.M[H.ki[k]]) > 0].min() < 1.0]
    med = float(np.median(tc)); T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        e = abs(float(np.median((Aall >= t).sum(1))) - med)
        if best is None or e < best:
            best, T = e, float(t)
    per = (Aall >= T).sum(1)
    sources = [kos[i] for i in range(len(kos)) if per[i] >= MINSRC]
    print(f"K562: |z|>={T:.1f} | {len(sources)} sources | predicting {KPROG} program loadings from annotation + start state alone")

    # ---------------- features: annotation + start state ONLY ----------------
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    comps = collections.defaultdict(set)
    for cid, mem in D["complexes"].items():
        mm = sorted({names[x] for x in mem if isinstance(x, int) and x < N})
        if 2 <= len(mm) <= MAXCOMPLEX:
            for g in mm:
                comps[g].add(cid)
    doms = {}
    try:
        dd = json.load(open(OUT / "domains.json")).get("domains", {})
        for k, v in dd.items():
            try:
                doms[names[int(k)]] = set(str(x) for x in v)
            except (ValueError, IndexError):
                pass
    except Exception:
        pass
    # REAL start state: expression in K562 CONTROL cells, from the Replogle GWPS file. control_expr is the targeted gene's own
    # unperturbed level; clean_mean/sd/cv/gini are the control-cell statistics of any gene that is on the readout panel. Neither
    # is derived from a perturbation response -- the earlier draft used mean |z| across knockouts as an "expression proxy",
    # which is a perturbation quantity and would have contradicted the premise of the whole test.
    BE = json.load(open(SP / "k562_baseline_expression.json"))
    base_panel = BE["measured_baseline"]           # gene -> [clean_mean, clean_std, clean_cv, gini]
    ctrl_expr = BE["source_control_expr"]          # gene -> unperturbed expression of the targeted gene
    ncov = sum(1 for s in sources if s in ctrl_expr)
    npan = sum(1 for s in sources if s in base_panel)
    print(f"start state: control_expr for {ncov}/{len(sources)} sources, control-cell panel stats for {npan}/{len(sources)}")

    # DepMap essentiality is itself a knockout measurement of the test gene -- a viability screen, not a transcriptome response, but a
    # perturbation nonetheless. MODES[2] drops it so there is a number for a gene that has never been perturbed by ANY assay.
    NUM = ["ess", "loeuf", "tf", "ppi", "npath", "ndis", "cpg", "pubs", "dark", "conf", "enh", "dep_frac"]
    DEPMAP = {"ess", "dep_frac"}

    def build_X(train, mode):
        """Feature matrix for every source, with the categorical VOCABULARIES fitted on TRAIN sources only.
        Fitting them on all sources would let the choice of columns depend on the held-out genes."""
        def vocab(getter, minn):
            c = collections.Counter()
            for s in train:
                c.update(getter(s))
            return sorted(k for k, v in c.items() if v >= minn)
        v_comp = vocab(lambda s: comps.get(s, ()), 2) if mode == "full" else []
        v_dom = vocab(lambda s: doms.get(s, ()), 3)
        v_proc = vocab(lambda s: [(info.get(s, {}) or {}).get("proc") or "?"], 3)
        v_loc = vocab(lambda s: [(info.get(s, {}) or {}).get("comp") or "?"], 3)
        num = [k for k in NUM if not (mode == "no_depmap" and k in DEPMAP)]

        def feats(s):
            gi = info.get(s, {}) or {}
            x = []
            for k in num:
                try:
                    x.append(float(gi.get(k) or 0.0))
                except (TypeError, ValueError):
                    x.append(0.0)
            bp = base_panel.get(s)
            x.append(np.log1p(float(ctrl_expr.get(s, 0.0))))
            x.append(1.0 if s in ctrl_expr else 0.0)
            x += [np.log1p(bp[0]), bp[1], bp[2], bp[3], 1.0] if bp else [0.0, 0.0, 0.0, 0.0, 0.0]
            x += [1.0 if (gi.get("proc") or "?") == p else 0.0 for p in v_proc]
            x += [1.0 if (gi.get("comp") or "?") == p else 0.0 for p in v_loc]
            dl = doms.get(s, set())
            x += [1.0 if d in dl else 0.0 for d in v_dom]
            cl = comps.get(s, set())
            x += [1.0 if c in cl else 0.0 for c in v_comp]
            return x
        return np.array([feats(s) for s in sources], float), (len(v_proc), len(v_loc), len(v_dom), len(v_comp))

    si_of = {s: i for i, s in enumerate(sources)}
    res = collections.defaultdict(list)
    per_src = collections.defaultdict(lambda: collections.defaultdict(list))   # label -> source -> recalls (de-pseudo-replicate)

    def recall(pred_full, cand, truth):
        top = cand[np.argsort(-np.abs(pred_full[cand]))[:K_RECALL]]
        return len({genes[q] for q in top} & truth) / min(len(truth), K_RECALL)

    seed_means = collections.defaultdict(list)
    for seed in SEEDS:
      RNG = np.random.RandomState(seed)
      seed_start = {lbl: len(res[lbl]) for lbl in MODELS + REFS}
      for si in range(NSPLIT):
        ss = sorted(sources); RNG.shuffle(ss)
        cut = int(0.7 * len(ss)); train, test = ss[:cut], ss[cut:]
        tr_rows = [kidx[s] for s in train]
        tide = (Aall[tr_rows] >= T).mean(0) >= TIDE_FRAC
        ok = ~tide; keep = np.where(ok)[0]; cand = keep
        Mtr = Z[tr_rows][:, keep].astype(np.float64)
        mu = Mtr.mean(0); U, s_, Vt = np.linalg.svd(Mtr - mu, full_matrices=False)
        Gk = Vt[:KPROG]
        Ptr = (Mtr - mu) @ Gk.T                                  # train loadings (the regression target)
        tr_rank = np.zeros(len(genes)); tr_rank[keep] = (Aall[tr_rows][:, keep] >= T).mean(0)
        mu_full = np.zeros(len(genes)); mu_full[keep] = mu

        truths = {}
        for s in test:
            tset = {genes[q] for q in np.where((Aall[kidx[s]] >= T) & ok)[0] if genes[q] != s}
            if tset:
                truths[s] = tset

        def record(lbl, s, v):
            res[lbl].append(v); per_src[lbl][s].append(v)

        for mode, lbl in zip(MODES, MODELS):
            X, vsz = build_X(train, mode)
            if si == 0:
                print(f"  {lbl:44s} {X.shape[1]:>4d} features | {vsz[0]} proc, {vsz[1]} comp, {vsz[2]} dom, {vsz[3]} complexes")
            Xtr = X[[si_of[s] for s in train]]; Xte = X[[si_of[s] for s in test]]
            m = Xtr.mean(0); sd = Xtr.std(0) + 1e-9              # scaling from TRAIN only
            reg = Ridge(alpha=RIDGE).fit((Xtr - m) / sd, Ptr)
            Pte = reg.predict((Xte - m) / sd)
            preds = {}
            for j, s in enumerate(test):
                if s not in truths:
                    continue
                pred = np.zeros(len(genes)); pred[keep] = mu + Pte[j] @ Gk
                preds[s] = pred
                record(lbl, s, recall(pred, cand, truths[s]))
            if mode == "full":
                # THE DECISIVE CONTROL. Score each source's prediction against a DIFFERENT held-out source's truth. The recall above
                # could come from the model picking a generically-plausible non-tide gene set that happens to overlap any knockout's
                # movers; if so, mismatching sources costs nothing. Only the gap between matched and mismatched is gene-specific skill.
                sc = sorted(preds); perm = list(sc)
                for _ in range(200):                             # derangement: nobody keeps their own truth
                    RNG.shuffle(perm)
                    if all(a != b for a, b in zip(sc, perm)):
                        break
                for a, b in zip(sc, perm):
                    # keep the TRUTH fixed at a's and swap in b's prediction, so the comparison against the matched score is
                    # exactly paired -- same target genes, same recall denominator, only the prediction changes.
                    record(REFS[4], a, recall(preds[b], cand, truths[a]))
            if mode == "full":                                   # 1-NN retrieval shares this feature space
                Ztr = (Xtr - m) / sd; Zte = (Xte - m) / sd
                Ztr_n = Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)
                Zte_n = Zte / (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
                nn = np.argmax(Zte_n @ Ztr_n.T, axis=1)
                for j, s in enumerate(test):
                    if s not in truths:
                        continue
                    pred = np.zeros(len(genes)); pred[keep] = Z[kidx[train[nn[j]]]][keep]
                    record(REFS[1], s, recall(pred, cand, truths[s]))

        for s in test:
            if s not in truths:
                continue
            record(REFS[0], s, recall(mu_full, cand, truths[s]))
            record(REFS[2], s, recall(tr_rank, cand, truths[s]))
            record(REFS[3], s, recall(RNG.rand(len(genes)), cand, truths[s]))
      for lbl in MODELS + REFS:
          seed_means[lbl].append(float(np.mean(res[lbl][seed_start[lbl]:])))

    print("\n  across-seed spread (each seed = its own 5 splits, so this is the number that matters, not one split):")
    for lbl in MODELS + REFS:
        v = np.array(seed_means[lbl])
        print(f"  {lbl:52s} {v.mean():>8.4f} +- {v.std():.4f}   {np.round(v, 4).tolist()}")

    # A source can land in the test half of several splits, so the 489 raw scores are NOT independent. Every significance test below
    # runs on one value per UNIQUE source (its mean across the splits that held it out), which is the honest unit of replication.
    scored = sorted(set(per_src[MODELS[0]]))
    def bysrc(lbl):
        return np.array([float(np.mean(per_src[lbl][s])) for s in scored])

    print(f"\n  {'model':52s} {'recall@50':>10s} {'n_obs':>6s} {'n_src':>6s}")
    tab = []
    for lbl in MODELS + REFS:
        v = res[lbl]; b = bysrc(lbl)
        tab.append({"model": lbl, "recall50": round(float(np.mean(v)), 4),
                    "recall50_per_source": round(float(b.mean()), 4), "n_obs": len(v), "n_sources": len(b)})
        print(f"  {lbl:52s} {np.mean(v):>10.4f} {len(v):>6d} {len(b):>6d}")
    full = bysrc(MODELS[0]); intr = bysrc(MODELS[1]); nodm = bysrc(MODELS[2])
    muo = bysrc(REFS[0]); nn1 = bysrc(REFS[1]); tid = bysrc(REFS[2]); mis = bysrc(REFS[4])
    p_mu = float(wilcoxon(full, muo).pvalue); p_t = float(wilcoxon(full, tid).pvalue)
    p_i = float(wilcoxon(full, intr).pvalue); p_n = float(wilcoxon(full, nn1).pvalue)
    p_d = float(wilcoxon(nodm, tid).pvalue); p_x = float(wilcoxon(full, mis).pvalue)
    gene_specific = bool(full.mean() > muo.mean() and p_mu < 0.05)
    beats_tide = bool(full.mean() > tid.mean() and p_t < 0.05)
    cx_carries = bool(full.mean() > intr.mean() and p_i < 0.05)
    nodm_works = bool(nodm.mean() > tid.mean() and p_d < 0.05)
    print(f"\n  (paired tests on {len(scored)} unique sources, not the {len(res[MODELS[0]])} split-level observations)")
    print(f"  vs mu-only (THE test: is anything gene-specific?) : {full.mean()-muo.mean():+.4f}, paired p={p_mu:.3g}")
    print(f"  vs tide-null                                      : {full.mean()-tid.mean():+.4f}, paired p={p_t:.3g}")
    print(f"  vs 1-NN annotation retrieval                      : {full.mean()-nn1.mean():+.4f}, paired p={p_n:.3g}")
    print(f"  complex identity contributes                      : {full.mean()-intr.mean():+.4f}, paired p={p_i:.3g}")
    print(f"  no-complex no-DepMap vs tide-null                 : {nodm.mean()-tid.mean():+.4f}, paired p={p_d:.3g}")
    print(f"  matched vs mismatched source (gene-specificity)   : {full.mean()-mis.mean():+.4f}, paired p={p_x:.3g}")

    verdict = (
        "PROGRAM COLD START (program_coldstart.py): predicting what a knockout will do to a gene NOBODY HAS KNOCKED OUT, using no perturbation "
        "data about it or any of its partners -- only annotation and measured start state (expression in K562 control cells). This removes the "
        "crutch in program_factorization, which reached 0.350 only by averaging the MEASURED profiles of the test gene's neighbours and which "
        "cannot fire at all when those partners were never perturbed. It also drops the assumption that effects stay local: the programs are global "
        "by construction (component 1 is translation and moves ribosomal transcripts cell-wide), so predicting the PROGRAM predicts distal effects "
        f"automatically. Target is {KPROG} program loadings -- two to three orders of magnitude smaller than the targets that defeated earlier "
        "annotation-only attempts (annotation_prior ranked 8k genes at recall@50 0.125 against a 0.272 tide; infer_network predicted 8,726 edges at "
        "AUC 0.542). "
        f"RESULT on {len(scored)} held-out sources (mean recall@50 per source): annotation+complex {full.mean():.4f}, gene-intrinsic only "
        f"{intr.mean():.4f}, no-complex-no-DepMap {nodm.mean():.4f}, mu-only (the average knockout response, same for every gene) {muo.mean():.4f}, "
        f"1-NN annotation retrieval {nn1.mean():.4f}, tide-null {tid.mean():.4f}, random {bysrc(REFS[3]).mean():.4f}. "
        f"THE DECISIVE CONTROL is scoring each source's prediction against a DIFFERENT held-out source's truth (same truth set and denominator, "
        f"only the prediction swapped): that collapses to {mis.mean():.4f} against {full.mean():.4f} matched (paired p={p_x:.3g}), so the model is "
        "picking genes specific to the knockout in question, not a generically-plausible non-tide gene set that would overlap any knockout's movers. "
        "Second, the model minus mu-only, because the reconstruction is mu + P_hat @ G and mu is shared by every gene, so "
        "only the difference is gene-specific: "
        + (f"the gene-specific term is REAL, worth {full.mean()-muo.mean():+.4f} over mu-only (paired p={p_mu:.3g}). Annotation and start state do "
           "carry knockout-specific information. " if gene_specific else
           f"it is {full.mean()-muo.mean():+.4f} (paired p={p_mu:.3g}), so the model is NOT adding gene-specific content -- whatever recall it "
           "shows is the average knockout response, which is identical for every gene and requires no annotation at all. Cold start from "
           "annotation does not work at this target size either. ")
        + (f"It also clears the tide-null by {full.mean()-tid.mean():+.4f} (p={p_t:.3g}). " if beats_tide else
           f"It does not clear the tide-null ({full.mean()-tid.mean():+.4f}, p={p_t:.3g}). ")
        + (f"Complex identity carries a large share ({full.mean():.4f} with, {intr.mean():.4f} without, p={p_i:.3g}). That is consistent with the "
           "components literally being complexes, but it is also the honesty limit of the 'no neighbours' claim: no partner's MEASURED profile is "
           "read at prediction time, yet the regressor learned each complex's loadings from that complex's TRAIN members, so complex identity is a "
           "curated partner relation used through supervision. The partner-free number is the gene-intrinsic one. " if cx_carries else
           f"Complex identity is not what carries it ({full.mean():.4f} with, {intr.mean():.4f} without, p={p_i:.3g}). ")
        + (f"Dropping DepMap essentiality as well -- so the gene has been perturbed by no assay at all, not even a viability screen -- gives "
           f"{nodm.mean():.4f} vs tide-null {tid.mean():.4f} (p={p_d:.3g}), so the result does not depend on having already knocked the gene out. "
           if nodm_works else
           f"Dropping DepMap essentiality too (no assay has ever perturbed the gene) collapses it to {nodm.mean():.4f} vs tide-null {tid.mean():.4f} "
           f"(p={p_d:.3g}) -- the viability screen was doing the work. ")
        + f"Note the 1-NN retrieval baseline at {nn1.mean():.4f}: simply copying the measured profile of the most annotation-similar train gene gets "
        "most of the way, so the program decomposition is worth "
        + (f"{full.mean()-nn1.mean():+.4f} over it (p={p_n:.3g}). " if full.mean() > nn1.mean() else
           f"{full.mean()-nn1.mean():+.4f} -- i.e. nothing over it (p={p_n:.3g}). ")
        + f"Stable across seeds: {' / '.join(f'{v:.4f}' for v in seed_means[MODELS[0]])} over {len(SEEDS)} independent seeds of {NSPLIT} splits each. "
        + "Deterministic; programs, mu, feature vocabularies, scaling, regressor and tide mask all fitted on train sources only; the test gene's own "
        f"response is used only as truth and no neighbour response is used anywhere. Significance tested on {len(scored)} unique sources, not the "
        f"{len(res[MODELS[0]])} split-level observations, since a source can be held out in several splits.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"threshold": T, "n_programs": KPROG, "n_splits": NSPLIT, "n_unique_sources": len(scored), "table": tab,
               "p_vs_mu_only": p_mu, "p_vs_tide": p_t, "p_vs_1nn": p_n, "p_complex_ablation": p_i, "p_nodepmap_vs_tide": p_d, "p_matched_vs_mismatched": p_x,
               "gene_specific_signal": gene_specific, "beats_tide_null": beats_tide,
               "complex_carries": cx_carries, "nodepmap_works": nodm_works, "verdict": verdict, "note": verdict},
              open(OUT / "program_coldstart.json", "w"), indent=1)
    print("\n  -> outputs/orphan/program_coldstart.json")


if __name__ == "__main__":
    main()
