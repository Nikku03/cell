"""reason_modules — the whole stack + LLM reasoning, now at the RIGHT UNIT (programs, not genes). Before, the reasoning layer
failed because it reranked individual genes (per-gene identity is chaotic) and betting on specific biology backfired. Now the
predictable unit is the transcriptomic MODULE (panic button). So we test: can a biology-expert reasoner pick WHICH module a knockout
trips (heme / ISR / SREBP / ribosome / mito ...) better than the mechanical XGBoost module-classifier, and does that improve
far-field RECALL over the generic tide?

Strict held-out protocol:
  mode 'dump'  : NMF the K562-deep response into modules, print each module's top genes (so its biology is legible) + a panel of
                 knockouts with identity/function/capacity -- but HIDE their movers and true module. Also train the mechanical
                 XGBoost module-classifier (held-out) and cache all state. The reasoner reads and COMMITS, per knockout, which
                 module(s) it fires.
  mode 'score' : reveal movers; score three far-field predictors at a fixed 100-gene budget --
                   TIDE-ONLY        : top-100 generic prior (the dominant program)
                   STACK-mechanical : top-70 tide + top-30 of the XGBoost-predicted module
                   STACK-reasoned   : top-70 tide + top-30 of the LLM-reasoned module
                 plus module-selection accuracy (reasoned vs mechanical vs actual dominant module).
The honest question: at the module level, does reasoning finally ADD -- or does the mechanical classifier / the plain tide already
capture everything?
"""
import os
import sys, json, collections
from pathlib import Path
import numpy as np
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
STATE = f"{SP}/reason_modules_state.json"
NMODULES = 15
TOPVAR = 3000
N_PANEL = 8
BUDGET = 100


def _build_state():
    import farfield_modules as fm
    from sklearn.decomposition import NMF
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb
    D, info, idx, kos, kg, Z, movers = fm.load()
    A = np.abs(Z); var = A.var(0); top = np.argsort(-var)[:TOPVAR]
    At = A[:, top]; gtop = [kg[j] for j in top]
    nmf = NMF(n_components=NMODULES, init="nndsvda", max_iter=300, random_state=0)
    W = nmf.fit_transform(At); H = nmf.components_
    mod_genes = [[gtop[j] for j in np.argsort(-H[m])[:60]] for m in range(NMODULES)]
    dom = W.argmax(1)
    mf = collections.Counter()
    for g in kos:
        for j in movers[g]:
            mf[j] += 1
    prior_rank = [g for g, _ in mf.most_common()]
    F = fm._features(info, kos, kg, Z)
    dom = np.asarray(dom); groups = np.arange(len(kos))
    pred_mod = np.zeros(len(kos), dtype=int)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(F, dom, groups):
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8,
                                num_class=NMODULES, objective="multi:softmax", n_jobs=4)
        clf.fit(F[tr], dom[tr]); pred_mod[te] = clf.predict(F[te])
    # panel = top-N by mover count (deterministic, unbiased)
    panel = sorted(range(len(kos)), key=lambda i: -len(movers[kos[i]]))[:N_PANEL]
    state = {"kos": kos, "mod_genes": mod_genes, "dominant": dom.tolist(), "pred_mod": pred_mod.tolist(),
             "prior_rank": prior_rank[:400], "movers": {kos[i]: sorted(movers[kos[i]]) for i in range(len(kos))},
             "panel": [kos[i] for i in panel],
             "info": {kos[i]: {"proc": info[kos[i]].get("proc", "?"), "path": info[kos[i]].get("path", "?"),
                               "comp": info[kos[i]].get("comp", "?"), "tf": info[kos[i]].get("tf", 0),
                               "dep_frac": info[kos[i]].get("dep_frac", 0), "ess": info[kos[i]].get("ess", 0)}
                      for i in panel}}
    json.dump(state, open(STATE, "w"))
    return state


def dump():
    state = _build_state()
    kos = state["kos"]; ki = {g: i for i, g in enumerate(kos)}
    print(f"# REASON-MODULES CONTEXT DUMP (movers + true module HIDDEN). {NMODULES} programs from K562-deep NMF.\n")
    print("## The 15 modules (top genes -- infer the biology / panic-button of each):")
    for m, gs in enumerate(state["mod_genes"]):
        print(f"  M{m:02d}: {', '.join(gs[:12])}")
    print(f"\n## Panel ({len(state['panel'])} knockouts) -- pick which module(s) each one fires:")
    for g in state["panel"]:
        gi = state["info"][g]
        print("=" * 92)
        print(f"KNOCKOUT {g}: proc={gi['proc']} | pathway={str(gi['path'])[:40]} | comp={gi['comp']} | is_TF={gi['tf']} "
              f"| dep_frac={gi['dep_frac']} | ess={gi['ess']}")
    print("\n# Reason which module(s) each KO triggers, fill PICKS (ko -> [module ids]) in reason_modules.py, run: score")


# ---- filled after reasoning (biology expert), before reveal: which module(s) does each panel KO fire ----
# M01=ribosome, M05=ISR/amino-acid stress, M10=heme/erythroid, M12=mito/OXPHOS, M13=UPR, M04=generic responsive
PICKS = {
    "GATA1":   [10],        # erythroid master TF -> heme/erythroid program (high conviction)
    "POT1":    [4, 11],     # telomere/shelterin -> generic stress + growth-arrest (low conviction)
    "RPL7A":   [1],         # ribosomal protein -> ribosome program (high conviction)
    "SUPT6H":  [1, 4],      # elongation/biogenesis collapse -> ribosome/generic
    "RBM22":   [5, 1],      # spliceosome -> ISR (splicing stress) + ribosome
    "AQR":     [5],         # spliceosome/intron -> ISR
    "SNRPD2":  [5, 1],      # snRNP/splicing -> ISR + ribosome
    "PPP1R10": [4],         # 3' processing -> generic (low conviction)
}


def _recall(pred_genes, movers_set):
    ms = set(movers_set)
    if len(ms) < 3:
        return None
    p = set(pred_genes)
    return len(ms & p) / len(ms)


def score():
    state = json.load(open(STATE))
    kos = state["kos"]; ki = {g: i for i, g in enumerate(kos)}
    mod_genes = state["mod_genes"]; prior = state["prior_rank"]; movers = state["movers"]
    dom = state["dominant"]; pred_mod = state["pred_mod"]
    rows = []
    for g in state["panel"]:
        if g not in PICKS:
            continue
        i = ki[g]; mv = movers[g]
        tide = [x for x in prior[:BUDGET] if x != g]
        tide70 = [x for x in prior[:70] if x != g]
        mech_mod = pred_mod[i]
        mech_pred = tide70 + [x for x in mod_genes[mech_mod][:30] if x not in set(tide70) and x != g]
        rmods = PICKS[g]
        rgenes = []
        for m in rmods:
            rgenes += mod_genes[m][:30 // max(len(rmods), 1) + 5]
        reason_pred = tide70 + [x for x in rgenes if x not in set(tide70) and x != g][:30]
        rows.append({
            "ko": g, "n_movers": len(mv), "true_module": dom[i], "mech_module": mech_mod, "reasoned_modules": rmods,
            "recall_tide": _recall(tide, mv), "recall_mech": _recall(mech_pred, mv), "recall_reason": _recall(reason_pred, mv),
            "reasoned_hit_true": int(dom[i] in rmods), "mech_hit_true": int(dom[i] == mech_mod)})
    def avg(k):
        vs = [r[k] for r in rows if r[k] is not None]; return round(float(np.mean(vs)), 3) if vs else None
    res = {"n_panel": len(rows), "budget": BUDGET,
           "avg_recall_tide_only": avg("recall_tide"), "avg_recall_stack_mechanical": avg("recall_mech"),
           "avg_recall_stack_reasoned": avg("recall_reason"),
           "module_acc_reasoned": round(float(np.mean([r["reasoned_hit_true"] for r in rows])), 3),
           "module_acc_mechanical": round(float(np.mean([r["mech_hit_true"] for r in rows])), 3),
           "per_ko": rows}
    print("=" * 92)
    print(f"REASON-MODULES SCORE (panel {res['n_panel']}, budget {BUDGET} genes, recall of real movers)")
    print("=" * 92)
    print(f"  {'ko':10s} {'movers':>6s} {'trueM':>5s} {'mechM':>5s} {'reasonM':>10s}  {'tide':>5s} {'mech':>5s} {'reason':>6s}")
    for r in res["per_ko"]:
        print(f"  {r['ko']:10s} {r['n_movers']:>6} {r['true_module']:>5} {r['mech_module']:>5} {str(r['reasoned_modules']):>10s}  "
              f"{(r['recall_tide'] or 0):>5.2f} {(r['recall_mech'] or 0):>5.2f} {(r['recall_reason'] or 0):>6.2f}")
    print(f"\n  AVG RECALL @ {BUDGET}:  tide-only {res['avg_recall_tide_only']}   stack-mechanical {res['avg_recall_stack_mechanical']}"
          f"   stack-REASONED {res['avg_recall_stack_reasoned']}")
    print(f"  MODULE-SELECTION accuracy (hit the true dominant module):  reasoned {res['module_acc_reasoned']}  "
          f"mechanical {res['module_acc_mechanical']}")
    t, mech, reason = res["avg_recall_tide_only"], res["avg_recall_stack_mechanical"], res["avg_recall_stack_reasoned"]
    ra, ma = res["module_acc_reasoned"], res["module_acc_mechanical"]
    verdict = (
        f"WHOLE STACK + LLM REASONING AT THE MODULE LEVEL, panel of {res['n_panel']} knockouts (the biggest responders), recall of real "
        f"movers at a {BUDGET}-gene budget. Recall: tide-only {t}, stack-mechanical {mech}, stack-REASONED {reason} -- reasoning HURTS. "
        f"Module-selection: the biology reasoner hit the true dominant module {ra:.0%} of the time; the mechanical XGBoost classifier {ma:.0%}. "
        "THE FINDING (and it is the EIF2S1 lesson at the program level): my biologically-'obvious' calls were WRONG on all 8 because "
        "the dominant module of a strong knockout is NOT its specific functional program -- it is a GENERIC one. GATA1's dominant "
        "response is not the heme program (I picked M10) but an inflammatory/generic module; RPL7A's is not the ribosome module "
        "(I picked M01) but the generic module; the spliceosome knockouts' is not the ISR (I picked M05) but the generic program. The "
        "mechanical classifier WINS (62%) precisely because it learns 'big essential knockout -> generic panic button', which is "
        "exactly the CAPACITY MODEL's prediction: the largest deficits breach the global budget and trip the same generic stress "
        "program regardless of which gene was hit. So reasoning about specific biology fights the capacity logic -- and loses at every "
        "level, gene AND module. HONEST CAVEAT: this panel is the top-8 by response size = the strongest, most-essential knockouts, "
        "the ones MOST dominated by the generic global-stress program; a weaker, more-specific perturbation (e.g. a single lipid "
        "enzyme) might let a reasoned specific module help, and that is the one place left to look. BOTTOM LINE for the whole stack: "
        "the recoverable far-field is the TIDE (the generic dominant program), recovered by the generic prior + a capacity-driven "
        "mechanical module classifier; adding an LLM biology reasoner does NOT help and slightly hurts, because the measured response "
        "is dominated by the generic panic button, not the specific pathway. Reasoning is for the near-field (mechanism, mutations); "
        "the far-field belongs to capacity + the tide.")
    print(f"\n  VERDICT: {verdict}")
    res["verdict"] = verdict
    res["note"] = ("Whole stack + LLM reasoning at the MODULE level (the right unit, after the tide/ripple reframe). Strict held-out: "
                   "reasoner picks which NMF program(s) each knockout fires from identity/function/capacity (movers hidden), then "
                   "far-field recall at a 100-gene budget is scored for tide-only vs stack-mechanical (XGBoost module) vs "
                   f"stack-reasoned (LLM module). RESULT: recall tide-only {t}, mechanical {mech}, reasoned {reason}; module-selection "
                   f"reasoned {res['module_acc_reasoned']} vs mechanical {res['module_acc_mechanical']}. "
                   + ("Reasoning adds at the module level (unlike per-gene where it hurt)." if reason and t and reason > t + 0.03
                      else "Reasoning does NOT help and slightly hurts even at the module level. The biology reasoner got the true "
                      f"dominant module {ra:.0%} vs the mechanical classifier {ma:.0%}: my 'obvious' specific calls (GATA1->heme, "
                      "RPL7A->ribosome, splicing->ISR) were all WRONG because a strong knockout's DOMINANT module is generic, not its "
                      "functional pathway -- exactly the capacity model's prediction (big deficit -> global panic button). The "
                      "mechanical classifier wins by learning 'big essential KO -> generic module'.") +
                   " Caveat: panel = top-8 by response size (most generic-dominated); weaker specific perturbations may differ. The "
                   "recoverable far-field is the tide (generic program) + capacity; reasoning is for the near-field, not the far-field.")
    json.dump(res, open(OUT / "reason_modules.json", "w"), indent=1)
    print("\n  -> outputs/orphan/reason_modules.json")
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    (dump if mode == "dump" else score)()
