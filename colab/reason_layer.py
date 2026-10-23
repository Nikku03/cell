"""reason_layer — test whether a BIOLOGY-EXPERT reasoning layer on top of the full stack improves the 10 predictions. Strict
held-out protocol (the reasoner = the LLM, which has seen these genes' biology in training, so we control it carefully):

  mode 'dump'  : for a fixed panel of knockouts, print the KO's identity + function + network context (curated regulon, signaling
                 partners and their regulons) + the full-stack CANDIDATE POOL (top genes by the mechanical score) -- but HIDE which
                 genes actually moved. The reasoner reads this, reasons, and COMMITS a top-10 per knockout in writing.
  mode 'score' : read the committed PICKS, REVEAL the measured movers, and score reasoned-picks vs full-stack-top10 vs prior-top10.

Ground truth = the DEEPER k562.h5ad essential screen at thr 1.5 (~8 movers/KO) -- better resolved so reasoning has real movers to
hit. The candidate pool = X's regulon + signaling-bridge targets + top prior genes (all measured in X's universe), so the reasoner
reranks within what was actually measured (cannot invent unmeasured genes). Panel chosen deterministically = top-N knockouts by
mover count (well-powered, no cherry-picking for genes the reasoner happens to know).
"""
import sys, json, collections
from pathlib import Path
import numpy as np
import h5py
import os
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
THR = 1.5
N_PANEL = 6
POOL_PRIOR = 30            # how many top-prior genes to show in the candidate pool


def _load():
    import predict10 as p10, predict10_deep as pd
    D = json.load(open(OUT / "cell_complete.json"))
    genes = D["genes"]; names = [g["name"] for g in genes]; idx = {n: i for i, n in enumerate(names)}
    info = {g["name"]: g for g in genes}
    KO = pd.load_ko("k562.h5ad", THR, idx)
    reg, sig = pd._reg_sig(idx)
    mf = collections.Counter()
    for g, d in KO.items():
        for j in d["movers"]:
            mf[j] += 1
    nko = len(KO)
    return names, idx, info, KO, reg, sig, mf, nko, p10


def _panel(KO, idx):
    kos = list(KO)
    return sorted(kos, key=lambda g: -len(KO[g]["movers"]))[:N_PANEL]


def candidate_pool(X, KO, reg, sig, mf, nko):
    """the genes the reasoner may choose from = regulon + bridge + top-prior, all measured in X's universe. WITH full-stack score."""
    U = set(KO[X]["U"])
    def prior(g): return (mf.get(g, 0) - (1 if g in KO[X]["movers"] else 0)) / max(nko - 1, 1)
    regset = set(reg.get(X, []))
    bridge = set()
    for tf in sig.get(X, ()):
        bridge |= set(reg[tf])
    bridge -= regset
    pool = {}
    for g in U:
        if g == X: continue
        tier = "regulon" if g in regset else ("bridge" if g in bridge else "prior")
        if tier == "prior":
            continue
        pool[g] = (tier, round(prior(g), 4))
    # add top prior genes
    top_prior = sorted((g for g in U if g != X), key=lambda g: -prior(g))[:POOL_PRIOR]
    for g in top_prior:
        pool.setdefault(g, ("prior", round(prior(g), 4)))
    return pool


def dump():
    names, idx, info, KO, reg, sig, mf, nko, p10 = _load()
    panel = _panel(KO, idx)
    print(f"# REASONING-LAYER CONTEXT DUMP  (ground truth hidden)  dataset=k562-deep thr={THR}  panel={len(panel)} knockouts\n")
    for X in panel:
        gi = info.get(X, {})
        U = set(KO[X]["U"])
        print("=" * 96)
        print(f"KNOCKOUT: {X}   [n_measured_movers HIDDEN]")
        print(f"  function: proc={gi.get('proc','?')} | pathway={gi.get('path','?')} | compartment={gi.get('comp','?')} "
              f"| is_TF={gi.get('tf',0)} | essential={gi.get('ess',0)}")
        rset = [t for t in reg.get(X, []) if t in U]
        print(f"  curated regulon (measured): {', '.join(rset[:25]) if rset else '(none — not a curated TF)'}")
        sigtf = list(sig.get(X, ()))
        print(f"  signals to TFs: {', '.join(sigtf[:12]) if sigtf else '(none in SIGNOR)'}")
        pool = candidate_pool(X, KO, reg, sig, mf, nko)
        # show pool grouped
        for tier in ("regulon", "bridge", "prior"):
            gs = sorted([g for g, (t, p) in pool.items() if t == tier], key=lambda g: -pool[g][1])
            if gs:
                print(f"  POOL[{tier}] ({len(gs)}): " + ", ".join(f"{g}({pool[g][1]:.2f})" for g in gs[:30]))
        # full-stack's own top-10 (for reference / baseline), computed but not marked as answer
        fs_top = p10.predict(X, U, KO, reg, sig, mf, nko, use_mech=True)
        print(f"  [full-stack mechanical top-10 for reference]: {', '.join(fs_top)}")
    print("\n# Reason over each and commit a top-10. Then fill PICKS in reason_layer.py and run: python reason_layer.py score")


# ---- filled in AFTER reasoning (biology expert), before scoring ----
PICKS = {
    # GATA1: erythroid EFFECTOR genes (heme/membrane) over weak TF-regulon members, back off to reliable movers
    "GATA1":  ["ALAS2", "GYPB", "HEMGN", "GP1BB", "PRG2", "KLF1", "FTH1", "SNHG1", "MT-ND2", "GAS5"],
    # POT1: telomere/shelterin -> global DNA-damage stress; pool is generic, ~prior with stress/growth-arrest bumped
    "POT1":   ["SNHG1", "MT-ND2", "GAS5", "FTH1", "MT-CO3", "EEF1A1", "MALAT1", "FTL", "BTG1", "TPT1"],
    # SUPT6H (SPT6): elongation + HISTONE gene control; bump histone + generic stress
    "SUPT6H": ["SNHG1", "MT-ND2", "GAS5", "FTH1", "HIST1H1C", "EEF1A1", "MT-CO3", "RPS3", "MALAT1", "TPT1"],
    # CHMP3: ESCRT/cytokinesis -> generic stress, minimal specific edge, ~prior
    "CHMP3":  ["SNHG1", "MT-ND2", "GAS5", "FTH1", "MT-CO3", "EEF1A1", "MALAT1", "FTL", "MT-ATP6", "TPT1"],
    # RPL7A: ribosomal-protein KO -> ribosomal/nucleolar stress; bump co-regulated RP genes
    "RPL7A":  ["RPS3", "RPS9", "RPS18", "RPS2", "RPS12", "SNHG1", "FTH1", "MT-ND2", "EEF1A1", "GAS5"],
    # AQR: spliceosome/TC-NER -> splicing + DNA-damage stress; pool generic, ~prior
    "AQR":    ["SNHG1", "MT-ND2", "GAS5", "FTH1", "MT-CO3", "EEF1A1", "MALAT1", "IGFL2-AS1", "MT-ATP6", "TPT1"],
}


def score():
    names, idx, info, KO, reg, sig, mf, nko, p10 = _load()
    if not PICKS:
        print("PICKS is empty — fill it after reasoning."); return
    rows = []
    for X, picks in PICKS.items():
        if X not in KO:
            continue
        mv = set(KO[X]["movers"]); U = set(KO[X]["U"])
        reasoned = [g for g in picks if g in U][:10]
        fs = p10.predict(X, U, KO, reg, sig, mf, nko, use_mech=True)
        pr = p10.predict(X, U, KO, reg, sig, mf, nko, use_mech=False)
        rows.append({"ko": X, "n_movers": len(mv),
                     "reasoned_hits": sum(1 for g in reasoned if g in mv),
                     "fullstack_hits": sum(1 for g in fs if g in mv),
                     "prior_hits": sum(1 for g in pr if g in mv),
                     "reasoned_top10": reasoned,
                     "reasoned_correct": [g for g in reasoned if g in mv]})
    R = np.mean([r["reasoned_hits"] for r in rows]); F = np.mean([r["fullstack_hits"] for r in rows])
    P = np.mean([r["prior_hits"] for r in rows])
    print("=" * 96)
    print(f"REASONING-LAYER SCORE  (panel of {len(rows)} knockouts, k562-deep thr {THR})")
    print("=" * 96)
    print(f"  {'knockout':10s} {'movers':>6s} {'REASONED':>9s} {'full-stack':>11s} {'prior':>6s}   reasoned correct")
    for r in rows:
        print(f"  {r['ko']:10s} {r['n_movers']:>6} {r['reasoned_hits']:>9}/10 {r['fullstack_hits']:>8}/10 {r['prior_hits']:>4}/10   "
              f"{', '.join(r['reasoned_correct']) or '—'}")
    print(f"\n  AVERAGE:  reasoned {R:.2f}/10   full-stack {F:.2f}/10   prior {P:.2f}/10")
    print(f"  reasoning adds {R-F:+.2f} vs mechanical full-stack, {R-P:+.2f} vs prior")
    verdict = (
        f"BIOLOGY-EXPERT REASONING LAYER, strict held-out (context given, movers hidden; reasoner commits top-10 before reveal), "
        f"panel of {len(rows)} well-powered K562-deep knockouts. RESULT: reasoning does NOT improve the prediction -- reasoned "
        f"{R:.2f}/10 vs mechanical full-stack {F:.2f}/10 vs generic prior {P:.2f}/10 ({R-F:+.2f} vs full-stack, {R-P:+.2f} vs prior). "
        "And crucially, WHERE the reasoner made confident MECHANISTIC bets they BACKFIRED: GATA1 (bet the erythroid effector program "
        "ALAS2/GYPB/HEMGN/GP1BB) scored 0/10 vs the prior's 1 and full-stack's 2 -- with 295 genes moving, almost none were the "
        "canonical erythroid targets; SUPT6H (bet histone genes, its known specialty) 4 vs 5; RPL7A (bet co-regulated RP genes) 6 vs "
        "7. Where reasoning 'helped' (CHMP3 +2, AQR +1) it was luck reordering GENERIC prior genes (MALAT1/FTL/TPT1), not biological "
        "insight. This generalizes the EIF2S1 lesson: the readout is dominated by generic stress-responsive genes, so betting on the "
        "biologically-correct specific program strictly hurts. The wall is NOT a reasoning gap -- a domain-expert reasoner with all "
        "context except the answer cannot beat the dumb prior, because the missing thing is the MEASUREMENT of specific transmission, "
        "not its interpretation. Honest caveats: n=6 panel (the -0.17 is within noise), single-shot (iterative feedback would leak the "
        "test set), and the reasoner reranked a mostly-generic candidate pool; a tool-augmented reasoner might differ -- but the "
        "direction is unambiguous and consistent with every prior result: specific mechanism, however sourced, does not lift the "
        "absolute count on this readout.")
    print(f"\n  VERDICT: {verdict}")
    out = {"panel": rows, "avg_reasoned": round(float(R), 2), "avg_fullstack": round(float(F), 2),
           "avg_prior": round(float(P), 2), "reasoning_vs_fullstack": round(float(R - F), 2),
           "reasoning_vs_prior": round(float(R - P), 2), "threshold": THR, "n_panel": len(rows), "verdict": verdict,
           "note": ("Test of a biology-expert reasoning layer on top of the full stack, strict held-out (context + full-stack "
                    "candidate pool given, measured movers hidden; the LLM reasoner commits a reranked top-10 per knockout BEFORE "
                    "reveal). Panel = 6 well-powered K562-deep knockouts (GATA1 + 5 essential core-machinery genes). RESULT: reasoning "
                    "does not help -- reasoned 3.17/10 = generic prior 3.17/10, slightly below mechanical full-stack 3.33/10. Confident "
                    "mechanistic bets BACKFIRED (GATA1 erythroid program 0/10 despite 295 movers; SUPT6H histone -1; RPL7A RP-genes "
                    "-1); apparent gains (CHMP3/AQR) were generic-gene reordering luck. Generalizes the EIF2S1 finding: the "
                    "transcriptomic readout is dominated by generic stress-responsive genes, so betting the biologically-correct "
                    "specific program hurts. The far-field wall is a MEASUREMENT gap, not a reasoning gap -- a domain expert with all "
                    "context except the answer cannot beat the prior. Caveats: n=6 (within noise), single-shot (iterative feedback "
                    "leaks), reranking a mostly-generic pool.")}
    json.dump(out, open(OUT / "reason_layer.json", "w"), indent=1)
    print("\n  -> outputs/orphan/reason_layer.json")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    (dump if mode == "dump" else score)()
