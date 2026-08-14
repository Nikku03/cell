"""LOOP 120 -- WIRE THE TF NETWORK INTO THE RATE LAW, and test it on the layer it has never seen.

WHAT WAS ACTUALLY MISSING. The regulatory network has been in this repository since loop 76 and it
has never once been in an equation. cell_graph carries 55,716 `regulate` edges; cell_assembled's
state_vector does not read one of them. k_sm was a constant per gene, derived from that gene's own
abundance and decay, with no regulator anywhere in it. So "the TF network is wired in" was true of
the graph and false of the dynamics, and this loop is the difference.

THE WIRING, AND ITS HONEST SIZE. To drive k_sm an edge needs a SIGN, because "TF j regulates gene i"
does not say whether more j means more i or less. C["reg"] has 612,133 edges and 558,005 of them
carry sign 0. The usable wiring is 54,128 signed edges over 1,451 regulators and 7,485 targets --
8.8% of the network. Every previous scoring of this network was as a GRAPH, where an unsigned edge
still counts as adjacency. The moment it has to appear in a rate law, 91.2% of it evaporates.

    k_sm_i(t) = kbar_i * max(0, 1 + G * sum_j s_ij * dev_j(t) / n_i)

Dividing by the regulator count n_i bounds the drive, so a gene with 774 regulators is not driven
774x harder than a gene with one. Signs CANCEL inside the sum, which is the only place in this
wiring where the network does work that a bare edge count could not.

WHY THE CELL CYCLE IS THE RIGHT TEST. Loop 119 established that 362 proteins oscillate whose mRNA
does not, and that this model has no mechanism for them. The other side of that table is 88 genes
where BOTH oscillate, and 38 where only the transcript does. Those 126 CCD transcripts are exactly
what a regulatory network is supposed to explain: an oscillating mRNA needs an oscillating
regulator. The network has never been tested against them, and it has failed everything else it has
been tested against -- it lost to publication count 2:1 in loop_tf_rate, and loop_perturb could not
beat shuffled edge signs. So this is a fair test on held-out data with a strong prior that it fails.

WHAT WAS MEASURED BEFORE THE GATES WERE WRITTEN, and is therefore reported and NOT gated on, because
a gate set after seeing its own number is not a gate:
    54,128 signed edges; 1,451 regulators; 7,485 targets
    728 of 1,634 CCD-transcript-called genes are targets of a signed edge   (44.6%)
    80 of 748 CCD proteins are regulators
    273 genes have a CCD-transcript call AND a signed regulator AND an mRNA state  (124 Yes/149 No)
    only 9 CCD-protein regulators have a full dynamical state, which is why the drive is imposed
    from the MEASURED call and not simulated -- the model cannot oscillate them, per loop 119

PREDECLARED, before any number:

  W1 THE SIGNS DO WORK                                              THE WIRING'S OWN PREREQUISITE.
       real signs against shuffled signs on the same edges, same degrees, same everything. Gate:
       the sign shuffle must be capability-checked AND the real wiring must beat it on the
       CCD-transcript prediction. loop_perturb failed this exact test on a different readout; if
       signs do not matter here either, then this is an adjacency list wearing a rate law.
  W2 CCD TRANSCRIPTS HAVE CCD-PROTEIN REGULATORS ABOVE CHANCE       THE MECHANISTIC CLAIM.
       null: shuffle WHICH regulators are cell-cycle-dependent, stratified by out-degree decile so
       hub status is preserved. NOT a permutation of the target column -- loop 94's N4 preserved
       in-degree exactly and its null was arithmetically inert. Gate: |z| >= 2 in the predicted
       direction.
  W3 FAME                                                           THE CONTROL THAT KEEPS WINNING.
       regulator publication count as a competing predictor of CCD-transcript status, on the same
       genes. Gate: the network's AUC must exceed pubs'. It did not in loop_tf_rate, 2:1 against.
  W4 THE WIRED INTEGRATOR REDUCES TO THE UNWIRED ONE                THE IMPLEMENTATION GATE.
       with the drive set to zero the TF-driven k_sm must equal the constant k_sm exactly, so the
       wired steady state must reproduce the unwired steady state to machine precision. Gate:
       max relative deviation < 1e-9. A wiring that changes the answer when it is switched off is
       not an addition to the model, it is a different model.
  W5 DOES THE DYNAMICS ADD ANYTHING OVER THE TOPOLOGY?              LOOP 112's QUESTION, AGAIN.
       integrated mRNA amplitude against a one-step signed regulator score, both scored on the same
       273 genes against the same measured calls. Gate: integrated must beat topology. Loop 112
       found the ODE subtracted (raw +0.4960, integrated +0.4649); if that repeats, the integrator
       is decoration on this layer too and the record should say so twice.
  W6 THE ASYMMETRY TEST                                             THE DISCRIMINATING ONE.
       loop 119's 362 protein-only oscillators have, by construction, no transcriptional source. So
       their regulators should NOT be enriched for cell-cycle-dependent proteins, while the 88
       both-oscillate genes should be. Gate: enrichment among transcript-CCD genes must exceed
       enrichment among protein-only genes. If the network scores them the same, it is not tracking
       transcriptional control at all -- it is tracking which genes get studied during mitosis.

-> outputs/loop_tf_cellcycle.json
"""
import collections
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
HPA = LR.SC / "proteinatlas.tsv"
SEED = 12000
NPERM = 2000
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H
T_CYCLE = 24.0
DRIVE_AMP = 0.50
GAIN = 1.0
W2_Z = 2.0
W4_TOL = 1e-9

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(pos, neg):
    pos = np.asarray(pos, float)[np.isfinite(np.asarray(pos, float))]
    neg = np.asarray(neg, float)[np.isfinite(np.asarray(neg, float))]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sv, r = allv[order], np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def decile_strata(vals, k=10):
    """Out-degree deciles, so a label shuffle cannot move cell-cycle status onto or off the hubs."""
    v = np.asarray(vals, float)
    q = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(q, v, side="right")


def shuffle_within(labels, strata, rng):
    """Permute a binary label inside each stratum, so the marginal count is preserved per stratum."""
    out = np.array(labels, bool)
    for s in np.unique(strata):
        m = strata == s
        out[m] = rng.permutation(out[m])
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 120 -- the TF network enters a rate law, and meets the cell cycle")
    say("=" * 100)
    say()

    D = CA.load()
    names = D["names"]
    wiring = CA.tf_wiring(D)
    n_signed = sum(len(v) for v in wiring.values())
    regs = sorted({r for v in wiring.values() for r, _ in v})
    say(f"  wiring: {n_signed:,} signed edges of {len(D['model']['reg']):,} "
        f"({n_signed / len(D['model']['reg']):.1%}); {len(regs):,} regulators, "
        f"{len(wiring):,} targets")
    outdeg = collections.Counter(r for v in wiring.values() for r, _ in v)

    with open(HPA, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        iG, iCP, iCT = h.index("Gene"), h.index("CCD Protein"), h.index("CCD Transcript")
        rows = [(x[iG], x[iCP], x[iCT]) for x in r]
    cp = {g: v for g, v, _ in rows if v in ("Yes", "No")}
    ct = {g: v for g, _, v in rows if v in ("Yes", "No")}
    ccdP = {g for g, v in cp.items() if v == "Yes"}

    S = D["schwan"]
    kbar_all = {g: S[g]["mrna_copies"] * (LN2 / S[g]["mrna_hl_h"] + MU)
                for g in S if S[g].get("mrna_copies") and S[g].get("mrna_hl_h")}
    a_all = {g: LN2 / S[g]["mrna_hl_h"] + MU for g in kbar_all}

    tgt = sorted(g for g in ct if g in wiring and g in kbar_all)
    y = np.array([ct[g] == "Yes" for g in tgt])
    say(f"  test set: {len(tgt):,} genes with a CCD-transcript call AND a signed regulator AND an "
        f"mRNA state -- {int(y.sum())} Yes / {int((~y).sum())} No")
    say(f"  drivers:  {len(ccdP & set(regs))} of {len(ccdP)} CCD proteins are regulators")
    say()

    ix = CA.tf_index(wiring, tgt, regs)
    nreg = ix[4]
    is_ccd_reg = np.array([g in ccdP for g in regs])

    def dev_const(vec):
        return lambda t: vec

    def dev_sin(vec, amp=DRIVE_AMP, T=T_CYCLE):
        w = 2.0 * np.pi / T
        return lambda t: vec * amp * np.sin(w * t)

    kbar = np.array([kbar_all[g] for g in tgt])
    aa = np.array([a_all[g] for g in tgt])

    gates = {}

    # ---------------------------------------------------------------- W4 first: the reduction
    say("W4 THE WIRED INTEGRATOR REDUCES TO THE UNWIRED ONE")
    zero = np.zeros(len(regs))
    rel0, mean0 = CA.integrate_tf(kbar, aa, ix, dev_const(zero), T_CYCLE, ncyc=3, nstep=100)
    ss = kbar / aa
    dev = float(np.max(np.abs(mean0 - ss) / ss))
    say(f"     drive off: max relative deviation from kbar/a over {len(tgt):,} genes = {dev:.3e}")
    say(f"     residual oscillation with the drive off: max {rel0.max():.3e}")
    gates["W4"] = bool(dev < W4_TOL and rel0.max() < W4_TOL)
    say(f"     W4 {'PASS' if gates['W4'] else 'FAIL'} -- the wiring is "
        f"{'an addition to the model' if gates['W4'] else 'a DIFFERENT model'}")
    say()

    # ---------------------------------------------------------------- W2
    say("W2 CCD TRANSCRIPTS HAVE CCD-PROTEIN REGULATORS ABOVE CHANCE")
    has = np.zeros(len(tgt), bool)
    cnt = np.zeros(len(tgt))
    for i, g in enumerate(tgt):
        c = sum(1 for rg, _ in wiring[g] if rg in ccdP)
        cnt[i] = c
        has[i] = c > 0
    real = float(has[y].mean() - has[~y].mean())
    say(f"     CCD transcripts with >=1 CCD-protein regulator: {has[y].mean():.1%} "
        f"({int(has[y].sum())}/{int(y.sum())})")
    say(f"     non-CCD transcripts                           : {has[~y].mean():.1%} "
        f"({int(has[~y].sum())}/{int((~y).sum())})")
    strata = decile_strata([outdeg[r] for r in regs])
    nulls, cap = [], None
    ccd_set = np.array(is_ccd_reg)
    for i in range(NPERM):
        sh = shuffle_within(ccd_set, strata, rng)
        if i == 0:
            cap = GG.null_can_move(ccd_set.astype(int), sh.astype(int))
        shset = {regs[j] for j in np.flatnonzero(sh)}
        hn = np.array([any(rg in shset for rg, _ in wiring[g]) for g in tgt])
        nulls.append(float(hn[y].mean() - hn[~y].mean()))
    sur = GG.survival(real, nulls, z_min=W2_Z)
    say(f"     null: cell-cycle status shuffled among regulators WITHIN out-degree deciles "
        f"(hub status preserved) -- {cap['reason']}")
    GG.report("enrichment difference", sur, emit=say)
    gates["W2"] = bool(cap["capable"] and np.isfinite(sur.get("z", np.nan))
                       and abs(sur["z"]) >= W2_Z and real > 0)
    say(f"     W2 {'PASS' if gates['W2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- W1 + W5: run the wiring
    say("W1 THE SIGNS DO WORK")
    drive_vec = is_ccd_reg.astype(float)
    relM, _ = CA.integrate_tf(kbar, aa, ix, dev_sin(drive_vec), T_CYCLE, gain=GAIN)
    a_real = auc(relM[y], relM[~y])
    say(f"     real signs:      AUC {a_real:.4f}  (integrated mRNA amplitude vs measured CCD call)")
    sh_aucs, sgn_caps = [], []
    rows_, cols_, sgn_, nsafe_, nreg_ = ix
    for i in range(64):
        p = rng.permutation(len(sgn_))
        ixs = (rows_, cols_, sgn_[p], nsafe_, nreg_)
        if i == 0:
            sgn_caps = GG.null_can_move(sgn_, sgn_[p])
        rs, _ = CA.integrate_tf(kbar, aa, ixs, dev_sin(drive_vec), T_CYCLE, gain=GAIN)
        sh_aucs.append(auc(rs[y], rs[~y]))
    sh_aucs = np.array(sh_aucs)
    say(f"     shuffled signs:  AUC {sh_aucs.mean():.4f} +/- {sh_aucs.std():.4f} "
        f"over {len(sh_aucs)} shuffles -- {sgn_caps['reason']}")
    gates["W1"] = bool(sgn_caps["capable"] and abs(a_real - 0.5) >
                       abs(sh_aucs.mean() - 0.5) + 2 * sh_aucs.std())
    say(f"     W1 {'PASS' if gates['W1'] else 'FAIL'} -- signs "
        f"{'carry information the edges alone do not' if gates['W1'] else 'DO NOT matter: this is an adjacency list wearing a rate law'}")
    say()

    say("W5 DOES THE DYNAMICS ADD ANYTHING OVER THE TOPOLOGY?")
    topo = np.zeros(len(tgt))
    for i, g in enumerate(tgt):
        topo[i] = abs(sum(s for rg, s in wiring[g] if rg in ccdP)) / max(1, len(wiring[g]))
    a_topo = auc(topo[y], topo[~y])
    a_cnt = auc(cnt[y], cnt[~y])
    say(f"     one-step signed regulator score (no ODE):  AUC {a_topo:.4f}")
    say(f"     raw count of CCD regulators (no signs):    AUC {a_cnt:.4f}")
    say(f"     integrated mRNA amplitude (the ODE):       AUC {a_real:.4f}")
    say(f"     regulator count itself, as a confound:     AUC {auc(nreg[y], nreg[~y]):.4f}")
    gates["W5"] = bool(np.isfinite(a_real) and np.isfinite(a_topo)
                       and abs(a_real - 0.5) > abs(a_topo - 0.5))
    say(f"     W5 {'PASS' if gates['W5'] else 'FAIL'} -- the ODE "
        f"{'adds to the topology' if gates['W5'] else 'ADDS NOTHING over the topology, as in loop 112'}")
    say()

    # ---------------------------------------------------------------- W3
    say("W3 FAME")
    pubs = D["pubs"]
    fame = np.array([np.mean([pubs.get(rg, 0.0) for rg, _ in wiring[g]]) for g in tgt])
    a_fame = auc(fame[y], fame[~y])
    own = np.array([pubs.get(g, 0.0) for g in tgt])
    a_own = auc(own[y], own[~y])
    say(f"     mean publication count of a gene's regulators:  AUC {a_fame:.4f}")
    say(f"     the gene's OWN publication count:               AUC {a_own:.4f}")
    best_net = max((abs(a_real - 0.5), "integrated"), (abs(a_topo - 0.5), "topology"))
    say(f"     best network score: {best_net[1]} at |AUC-0.5| = {best_net[0]:.4f}")
    gates["W3"] = bool(best_net[0] > abs(a_fame - 0.5) and best_net[0] > abs(a_own - 0.5))
    say(f"     W3 {'PASS' if gates['W3'] else 'FAIL'} -- "
        f"{'the network beats fame' if gates['W3'] else 'FAME WINS, as it did in loop_tf_rate'}")
    say()

    # ---------------------------------------------------------------- W6
    say("W6 THE ASYMMETRY TEST")
    both = sorted(set(cp) & set(ct))
    grp_both = [g for g in both if cp[g] == "Yes" and ct[g] == "Yes" and g in wiring]
    grp_prot = [g for g in both if cp[g] == "Yes" and ct[g] == "No" and g in wiring]
    grp_tx = [g for g in both if cp[g] == "No" and ct[g] == "Yes" and g in wiring]
    grp_none = [g for g in both if cp[g] == "No" and ct[g] == "No" and g in wiring]

    def enrich(gs):
        if not gs:
            return float("nan"), 0
        return float(np.mean([any(rg in ccdP for rg, _ in wiring[g]) for g in gs])), len(gs)

    e_both, n_both = enrich(grp_both)
    e_prot, n_prot = enrich(grp_prot)
    e_tx, n_tx = enrich(grp_tx)
    e_none, n_none = enrich(grp_none)
    say(f"     protein Yes / transcript Yes  {e_both:.1%}  (n={n_both})   both oscillate")
    say(f"     protein No  / transcript Yes  {e_tx:.1%}  (n={n_tx})   transcript only")
    say(f"     protein Yes / transcript No   {e_prot:.1%}  (n={n_prot})   PROTEIN ONLY -- no "
        f"transcriptional source exists for these")
    say(f"     protein No  / transcript No   {e_none:.1%}  (n={n_none})   neither")
    tx_all = (np.array([any(rg in ccdP for rg, _ in wiring[g]) for g in grp_both + grp_tx]).mean()
              if (grp_both + grp_tx) else float("nan"))
    say(f"     any CCD transcript ({n_both + n_tx}) {tx_all:.1%}  vs  protein-only "
        f"({n_prot}) {e_prot:.1%}   difference {tx_all - e_prot:+.1%}")
    gates["W6"] = bool(np.isfinite(tx_all) and np.isfinite(e_prot) and tx_all > e_prot)
    say(f"     W6 {'PASS' if gates['W6'] else 'FAIL'} -- the network "
        f"{'separates transcriptional from post-transcriptional oscillation' if gates['W6'] else 'scores post-transcriptional oscillation AS WELL AS transcriptional: it is not tracking transcriptional control'}")
    say()

    # ---------------------------------------------------------------- verdict
    say("=" * 100)
    for k in ("W1", "W2", "W3", "W4", "W5", "W6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[LR.CELL, HPA, LR.SC / "_schwan2011.json"],
                      available=len(ct), used=len(tgt), selection="filtered", seed=SEED,
                      controls=["edge signs shuffled, capability-checked, integrator re-run",
                                "cell-cycle status shuffled among regulators within out-degree "
                                "deciles so hub status is preserved",
                                "regulator publication count and the gene's own count",
                                "the wiring switched off, gated on exact reduction",
                                "one-step topology scored against the integrated ODE",
                                "protein-only oscillators as the negative class the network "
                                "must NOT explain"],
                      note="only the 54,128 signed edges (8.8%) can enter a rate law; the other "
                           "558,005 have no direction of effect and are unusable here")
    RM.report(man, emit=say)
    json.dump({"test": "loop_tf_cellcycle", "manifest": man, "gates": gates,
               "wiring": {"signed_edges": n_signed, "total_edges": len(D["model"]["reg"]),
                          "regulators": len(regs), "targets": len(wiring),
                          "ccd_regulators": len(ccdP & set(regs))},
               "testset": {"n": len(tgt), "n_yes": int(y.sum()), "n_no": int((~y).sum())},
               "w1": {"auc_real": a_real, "auc_shuffled_mean": float(sh_aucs.mean()),
                      "auc_shuffled_sd": float(sh_aucs.std()), "capability": sgn_caps},
               "w2": {"frac_yes": float(has[y].mean()), "frac_no": float(has[~y].mean()),
                      "difference": real, "survival": sur, "capability": cap},
               "w3": {"auc_regulator_pubs": a_fame, "auc_own_pubs": a_own,
                      "best_network": best_net[1], "best_network_margin": best_net[0]},
               "w4": {"max_rel_deviation": dev, "max_residual_oscillation": float(rel0.max())},
               "w5": {"auc_topology": a_topo, "auc_count": a_cnt, "auc_integrated": a_real,
                      "auc_regulator_count": auc(nreg[y], nreg[~y])},
               "w6": {"both": [e_both, n_both], "transcript_only": [e_tx, n_tx],
                      "protein_only": [e_prot, n_prot], "neither": [e_none, n_none],
                      "any_transcript_ccd": tx_all},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_tf_cellcycle.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_tf_cellcycle.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
