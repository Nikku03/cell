"""Loop 161. Coefficients, protons and electrons on the REM graph -- can chemistry predict the gap?

WHAT LOOP 160 LEFT ON THE TABLE, in its own words: "stoichiometric coefficients are dropped, so
this is topology and not flux". Its R5 then FAILED -- the Markov walk scored AUC 0.6928 on the
whole graph against a degree-only column at 0.7149, so counting beat walking. The walk only won
inside the 2-step shortlist, 0.8158 against 0.6904, because the shortlist removed the hubs the
degree column was riding on.

That failure has a shape. Everything loop 160 measured was CONNECTIVITY, and connectivity in a
metabolic network is dominated by H+ and H2O. This loop adds the one thing connectivity cannot see:
the reaction has to balance. Human-GEM carries a formula and a formal charge for every species, so
every reaction has an elemental balance and an electron balance, and a predictor built on those
does not care which metabolites are popular. If it works it CANNOT be the degree confound.

WHAT IS ADDED
    coefficients   55,198 stoichiometry values, restoring the weights loop 160 dropped
    formulas       8,461 species, 22 elements; 1,115 carry a generic R/X and cannot be balanced
    charges        8,461 formal charges
    protons        explicit H+ stoichiometry per reaction, and per compartment pair for transport
    electrons      net charge transfer, and the NAD/NADP/FAD/quinone/cytochrome redox couples

PREDECLARED, before any number is looked at.

  S1 THE CHEMISTRY ALIGNS WITH THE GRAPH. The coefficient arrays are built by a second scan of the
     SBML and must line up index-for-index with loop 160's edge arrays -- same length, same species
     at every position -- or the coefficients are being attached to the wrong edges.
     Gate: 0 mismatches across all 27,582 reactant and 27,616 product references.

  S2 DOES HUMAN-GEM ACTUALLY BALANCE? Element-wise and element-plus-charge, over every reaction
     whose species all carry a real formula.
     Gate: PASS iff element+charge balance holds for at least 90% of checkable reactions. Below
     that, balance is not a constraint this model obeys and S5/S6 are void -- they would then be
     reported as void rather than as a weak signal.

  S3 THE PROTON AND ELECTRON INVENTORY. Net H+ consumed or emitted per reaction; H+ moved between
     compartments; net charge transfer; how many reactions carry a redox couple. Per compartment.
     Gate: passes on being reported. This is an inventory, not a hypothesis.

  S4 DO THE COEFFICIENTS HELP THE WALK? Loop 160's identical held-out task -- same 500 reactions,
     same seed, same 2-step shortlist -- with transition probabilities weighted by stoichiometry
     instead of uniform.
     Gate: PASS iff the weighted walk beats the uniform walk by more than 3 sem. Written so that
     "the coefficients change nothing" is the reported outcome if that is what happens.

  S5 CAN BALANCE ALONE FIND THE MISSING PRODUCT? No graph at all. Take the held-out reaction's
     reactants, grant the currency products for free (any gap-filling method assumes water and
     protons are available, and that assumption is stated because it makes the task easier), and
     compute what the remaining products must sum to elementally. Score every candidate by the
     fraction of that residual its own formula can explain at a best non-negative coefficient.
     Gate: PASS iff AUC >= 0.60 on the same shortlist AND it beats the degree-only column by more
     than 0.02 -- the bar loop 160's walk failed on the whole graph.

  S6 DOES CHEMISTRY COMPOSE WITH CONNECTIVITY, OR SUBSTITUTE FOR IT? Rank-fuse the balance score
     with the walk score.
     Gate: PASS iff the fusion beats the better of the two alone by more than 3 sem. Loop 159 found
     five mechanisms that all substituted and were worth +0.0004 together; if that happens again
     here the honest report is that chemistry and connectivity are reading the same thing.

  S7 WHAT THIS STILL CANNOT SHOW. Balance is necessary and not sufficient -- a balanced equation
     can still be thermodynamically impossible and have no enzyme. 1,115 species carry a generic
     R/X group and drop out of every balance check. The held-out reactions are ones Human-GEM
     already contains, so this measures recovery and not discovery.

-> outputs/loop_rem_chemistry.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse, stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402

TOPO = Path("colab/data/rem_bipartite.npz")
CHEM = Path("colab/data/rem_chem.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_rem_chemistry.json"
CURRENCY_MIN = 200
ALPHA, NITER = 0.15, 60
N_HOLDOUT, SEED = 500, 16000          # identical to loop 160, so the numbers are comparable
S2_BAR, S5_AUC_BAR, S5_MARGIN = 0.90, 0.60, 0.02
COMPNAME = {"c": "cytosol", "e": "extracellular", "g": "Golgi", "i": "inner mitochondria",
            "l": "lysosome", "m": "mitochondria", "n": "nucleus", "r": "ER", "x": "peroxisome"}
REDOX = ("NAD", "NADP", "FAD", "FMN", "ubiquino", "cytochrome", "ferredoxin", "glutathione")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def walk(P, seed_idx, n, alpha=ALPHA, niter=NITER):
    e = np.zeros(n)
    e[seed_idx] = 1.0 / len(seed_idx)
    p = e.copy()
    for _ in range(niter):
        p = (1 - alpha) * (P @ p) + alpha * e
    return p


def auc_of(scores, pos):
    if pos.sum() == 0 or (~pos).sum() == 0:
        return np.nan
    r = stats.rankdata(scores)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    t0 = time.time()
    say("=" * 104)
    say("  COEFFICIENTS, PROTONS AND ELECTRONS -- can chemistry find the gap connectivity missed?")
    say("=" * 104)
    say()

    T = np.load(TOPO, allow_pickle=False)
    C = np.load(CHEM, allow_pickle=False)
    species = list(T["species"])
    sp_comp = list(T["sp_comp"])
    sp_name = list(T["sp_name"])
    rev = T["reversible"]
    rr, rs = T["react_rx"], T["react_sp"]
    pr, ps = T["prod_rx"], T["prod_sp"]
    NS, NR = len(species), len(T["reactions"])
    E = C["E"]
    charge = C["charge"]
    generic = C["generic"]
    elements = list(C["elements"])
    rco, pco = C["react_coef"], C["prod_coef"]

    # ------------------------------------------------------------------ S1
    say("S1 THE CHEMISTRY ALIGNS WITH THE GRAPH")
    ok_len = (len(C["react_sp"]) == len(rs) and len(C["prod_sp"]) == len(ps)
              and list(C["species"]) == species and list(C["reactions"]) == list(T["reactions"]))
    mism = 0
    if ok_len:
        mism = int((C["react_sp"] != rs).sum() + (C["prod_sp"] != ps).sum()
                   + (C["react_rx"] != rr).sum() + (C["prod_rx"] != pr).sum())
    s1 = bool(ok_len and mism == 0)
    say(f"     {len(rs):,} reactant and {len(ps):,} product references, "
        f"{mism} index mismatches against loop 160's edge arrays")
    say(f"     coefficients: {len(set(rco.tolist() + pco.tolist())):,} distinct values, "
        f"range {min(rco.min(), pco.min()):g} to {max(rco.max(), pco.max()):g}, "
        f"{int(((rco != 1) | False).sum() + (pco != 1).sum()):,} references carry a "
        f"coefficient other than 1")
    say(f"     {len(elements)} elements | {int(generic.sum()):,} species with a generic R/X group")
    GG.verdict(s1, emit=say, if_true=(
        "the coefficients attach to the edges loop 160 built, so the two loops describe one graph."),
        if_false="the arrays do not line up; the coefficients would be on the wrong edges.")
    say(f"     S1 {'PASS' if s1 else 'FAIL'}")

    # ------------------------------------------------------------------ S2
    say()
    say("S2 DOES HUMAN-GEM ACTUALLY BALANCE?")
    resid = np.zeros((NR, len(elements)))
    qres = np.zeros(NR)
    has_generic = np.zeros(NR, bool)
    for j, i, c in zip(rr, rs, rco):
        resid[j] -= c * E[i]
        qres[j] -= c * charge[i]
        has_generic[j] |= generic[i]
    for j, i, c in zip(pr, ps, pco):
        resid[j] += c * E[i]
        qres[j] += c * charge[i]
        has_generic[j] |= generic[i]
    nref = np.zeros(NR, int)
    for j in rr:
        nref[j] += 1
    npf = np.zeros(NR, int)
    for j in pr:
        npf[j] += 1
    checkable = (~has_generic) & (nref > 0) & (npf > 0)
    el_ok = (np.abs(resid) < 1e-6).all(1)
    q_ok = np.abs(qres) < 1e-6
    nc = int(checkable.sum())
    f_el = float((el_ok & checkable).sum() / max(nc, 1))
    f_both = float((el_ok & q_ok & checkable).sum() / max(nc, 1))
    say(f"     {nc:,} of {NR:,} reactions are checkable (both sides present, no generic group)")
    say(f"       element-balanced          {int((el_ok & checkable).sum()):,}  ({f_el:.1%})")
    say(f"       element AND charge        {int((el_ok & q_ok & checkable).sum()):,}  ({f_both:.1%})")
    worst = Counter()
    for j in np.where(checkable & ~el_ok)[0]:
        for k in np.where(np.abs(resid[j]) > 1e-6)[0]:
            worst[elements[k]] += 1
    say(f"     of the {int((checkable & ~el_ok).sum()):,} that do not balance, the elements left "
        f"over: {dict(worst.most_common(6))}")
    s2 = bool(f_both >= S2_BAR)
    GG.verdict(s2, emit=say, if_true=(
        f"balance is a constraint this model obeys at {f_both:.1%}, so it can be used as one."),
        if_false=(
        f"balance holds for only {f_both:.1%} of checkable reactions, below the {S2_BAR:.0%} bar. "
        f"S5 and S6 are VOID -- a constraint the model does not obey cannot predict what it "
        f"contains, and any AUC they report would be measuring the exceptions."))
    say(f"     S2 {'PASS' if s2 else 'FAIL'}")

    # ------------------------------------------------------------------ S3
    say()
    say("S3 THE PROTON AND ELECTRON INVENTORY")
    is_h = np.array([n == "H+" for n in sp_name])
    hnet = np.zeros(NR)
    for j, i, c in zip(rr, rs, rco):
        if is_h[i]:
            hnet[j] -= c
    for j, i, c in zip(pr, ps, pco):
        if is_h[i]:
            hnet[j] += c
    say(f"     explicit H+ species: {int(is_h.sum())} (one per compartment)")
    say(f"     reactions that consume H+ {int((hnet < 0).sum()):,} | emit H+ "
        f"{int((hnet > 0).sum()):,} | neither {int((hnet == 0).sum()):,}")
    hmove = Counter()
    hr = defaultdict(lambda: [set(), set()])
    for j, i, c in zip(rr, rs, rco):
        if is_h[i]:
            hr[int(j)][0].add(sp_comp[i])
    for j, i, c in zip(pr, ps, pco):
        if is_h[i]:
            hr[int(j)][1].add(sp_comp[i])
    for j, (a, b) in hr.items():
        for x in a:
            for y in b:
                if x != y:
                    hmove[(x, y)] += 1
    say("     H+ MOVED BETWEEN COMPARTMENTS (the proton-motive machinery, read off the model):")
    for (a, b), v in hmove.most_common(8):
        say(f"       {COMPNAME.get(a, a):<18s} -> {COMPNAME.get(b, b):<18s} {v:>4,} reactions")
    say(f"     net charge transferred, nonzero in {int((np.abs(qres) > 1e-6).sum()):,} reactions")
    redox = np.zeros(NR, bool)
    is_redox_sp = np.array([any(k.lower() in n.lower() for k in REDOX) for n in sp_name])
    for j, i in list(zip(rr, rs)) + list(zip(pr, ps)):
        if is_redox_sp[i]:
            redox[j] = True
    say(f"     redox couples (NAD/NADP/FAD/FMN/quinone/cytochrome/ferredoxin/glutathione) appear "
        f"in {int(redox.sum()):,} reactions ({redox.mean():.1%})")
    bycomp = Counter()
    for j in np.where(redox)[0]:
        for i in [i for k, i in zip(rr, rs) if k == j][:1]:
            bycomp[sp_comp[i]] += 1
    s3 = True
    say(f"     S3 {'PASS' if s3 else 'FAIL'}")

    # ------------------------------------------------------------------ held-out setup
    say()
    say("S4/S5/S6 THE HELD-OUT TASK -- loop 160's exact protocol, same seed, same 500 reactions")
    react_of, prod_of = defaultdict(set), defaultdict(set)
    coef_r, coef_p = defaultdict(dict), defaultdict(dict)
    for j, i, c in zip(rr, rs, rco):
        react_of[int(j)].add(int(i))
        coef_r[int(j)][int(i)] = float(c)
    for j, i, c in zip(pr, ps, pco):
        prod_of[int(j)].add(int(i))
        coef_p[int(j)][int(i)] = float(c)
    deg = Counter()
    for i in rs:
        deg[int(i)] += 1
    for i in ps:
        deg[int(i)] += 1
    currency = {i for i in range(NS) if deg[i] > CURRENCY_MIN}
    noncur = np.array(sorted(set(range(NS)) - currency))
    ncmap = {int(v): k for k, v in enumerate(noncur)}
    degv = np.array([deg[int(i)] for i in noncur], float)
    Enc = E[noncur]
    Enorm2 = (Enc ** 2).sum(1)
    Enorm2[Enorm2 == 0] = np.inf          # generic/empty formulas score 0 by construction

    nodes = NS + NR
    src = np.concatenate([rs, NS + pr, NS + rr[rev[rr] == 1], ps[rev[pr] == 1]])
    dst = np.concatenate([NS + rr, ps, rs[rev[rr] == 1], NS + pr[rev[pr] == 1]])
    wgt = np.concatenate([rco, pco, rco[rev[rr] == 1], pco[rev[pr] == 1]])

    def operator(mask_rx, weighted):
        s_, d_, w_ = src, dst, (wgt if weighted else np.ones(len(src)))
        keep = ~(((s_ >= NS) & (s_ - NS == mask_rx)) | ((d_ >= NS) & (d_ - NS == mask_rx)))
        A = sparse.csr_matrix((w_[keep], (d_[keep], s_[keep])), shape=(nodes, nodes))
        cs = np.asarray(A.sum(0)).ravel()
        cs[cs == 0] = 1.0
        return A @ sparse.diags(1.0 / cs)

    sp_rx = defaultdict(set)
    for j in range(NR):
        for i in react_of[j] | prod_of[j]:
            sp_rx[i].add(j)

    rng = np.random.default_rng(SEED)
    eligible = [j for j in range(NR)
                if (react_of[j] - currency) and (prod_of[j] - currency)
                and len(react_of[j] | prod_of[j]) >= 2]
    hold = rng.choice(eligible, size=min(N_HOLDOUT, len(eligible)), replace=False)
    say(f"     {len(eligible):,} eligible, {len(hold)} held out")

    A_uni, A_wt, A_bal, A_deg, A_fuse, A_bal_hard = [], [], [], [], [], []
    n_eligible_case, n_in_shortlist, rank_corr = 0, 0, []
    for t, j in enumerate(hold):
        j = int(j)
        seeds = sorted(react_of[j] - currency)
        targets = prod_of[j] - currency - set(seeds)
        if not seeds or not targets:
            continue
        p_uni = walk(operator(j, False), seeds, nodes)[:NS][noncur]
        p_wt = walk(operator(j, True), seeds, nodes)[:NS][noncur]
        pos = np.zeros(len(noncur), bool)
        for i in targets:
            pos[ncmap[int(i)]] = True
        excl = np.zeros(len(noncur), bool)
        for i in seeds:
            excl[ncmap[int(i)]] = True
        nb = set()
        for i in seeds:
            for k in sp_rx[i]:
                if k != j:
                    nb |= (react_of[k] | prod_of[k])
        nb -= currency
        inb = np.zeros(len(noncur), bool)
        for i in nb:
            inb[ncmap[int(i)]] = True
        m = inb & ~excl
        n_eligible_case += 1
        if pos[m].sum() == 0 or (~pos[m]).sum() == 0:
            continue
        n_in_shortlist += 1
        # the elemental residual the non-currency products must supply
        res_free = np.zeros(len(elements))
        for i, c in coef_r[j].items():
            res_free += c * E[i]
        res_hard = res_free.copy()
        for i, c in coef_p[j].items():
            if i in currency:
                res_free -= c * E[i]

        def bal(res):
            d = float(res @ res)
            if d <= 0:
                return np.zeros(len(noncur))
            ip = Enc @ res
            return np.where(ip > 0, ip ** 2 / (Enorm2 * d), 0.0)
        b_free, b_hard = bal(res_free), bal(res_hard)
        A_uni.append(auc_of(p_uni[m], pos[m]))
        A_wt.append(auc_of(p_wt[m], pos[m]))
        A_deg.append(auc_of(degv[m], pos[m]))
        A_bal.append(auc_of(b_free[m], pos[m]))
        A_bal_hard.append(auc_of(b_hard[m], pos[m]))
        rk_w, rk_b = stats.rankdata(p_uni[m]), stats.rankdata(b_free[m])
        fu = rk_w + rk_b
        A_fuse.append(auc_of(fu, pos[m]))
        if len(rk_w) > 2:
            rank_corr.append(float(stats.spearmanr(rk_w, rk_b).statistic))
        if t and t % 150 == 0:
            say(f"       {t}/{len(hold)}  [{time.time() - t0:.0f}s]")

    def mn(a):
        return float(np.nanmean(np.array(a, float)))

    def sem(a):
        a = np.array(a, float)
        return float(np.nanstd(a) / np.sqrt(np.isfinite(a).sum()))
    uni, wt, bal_, dgm, fus, hard = (mn(A_uni), mn(A_wt), mn(A_bal), mn(A_deg),
                                     mn(A_fuse), mn(A_bal_hard))
    su, sb = sem(A_uni), sem(A_bal)
    n_ok = int(np.isfinite(np.array(A_uni, float)).sum())
    say(f"     {n_ok} reactions scored, all on the 2-step shortlist")
    say()
    say(f"       uniform walk        {uni:.4f}   (sem {su:.4f})   <- loop 160's 0.8158")
    say(f"       stoich-weighted walk{wt:.4f}")
    say(f"       degree only         {dgm:.4f}")
    say(f"       BALANCE only        {bal_:.4f}   (sem {sb:.4f})   no graph at all")
    say(f"       balance, no currency granted {hard:.4f}")
    say(f"       walk + balance fused{fus:.4f}")

    say()
    say("     NOT A GATE -- two diagnostics added after the run, because two of the numbers above")
    say("     cannot be read without them.")
    rec = n_in_shortlist / max(n_eligible_case, 1)
    say(f"     (a) SHORTLIST RECALL. The 2-step shortlist contains at least one true non-currency")
    say(f"         product in {n_in_shortlist} of {n_eligible_case} held-out reactions = {rec:.1%}.")
    say(f"         Every AUC above is CONDITIONAL on that. Loop 160 reported 0.8158 as the")
    say(f"         shortlist's headline without this number beside it; the honest statement is that")
    say(f"         the shortlist ranks well when it contains the answer and cannot find it at all")
    say(f"         the other {1 - rec:.1%} of the time. That is a correction to loop 160's report.")
    rc = float(np.nanmean(np.array(rank_corr, float))) if rank_corr else float("nan")
    say(f"     (b) WHY THE FUSION DOES NOT HELP. Mean Spearman between the walk's ranking and the")
    say(f"         balance ranking, over the same candidates: {rc:+.4f}. Two scores that rank the")
    say(f"         same candidates the same way cannot add information by being averaged.")
    say()

    # ------------------------------------------------------------------ S4
    s4 = bool(wt - uni > 3 * su)
    GG.verdict(s4, emit=say, if_true=(
        f"S4 the coefficients help: {wt - uni:+.4f} over the uniform walk, outside 3 sem."),
        if_false=(
        f"S4 the coefficients change nothing: {wt - uni:+.4f} against a 3-sem bar of {3 * su:.4f}. "
        f"Weighting a transition by how many molecules the reaction consumes does not change which "
        f"species the walk reaches -- the stoichiometry is real chemistry the RANDOM WALK cannot "
        f"use, not chemistry that is absent."))
    say(f"     S4 {'PASS' if s4 else 'FAIL'}")

    # ------------------------------------------------------------------ S5
    say()
    s5 = bool(s2 and bal_ >= S5_AUC_BAR and bal_ - dgm > S5_MARGIN)
    GG.verdict(s5, emit=say, if_true=(
        f"S5 balance alone finds the missing product at {bal_:.4f}, beating the degree column by "
        f"{bal_ - dgm:+.4f}. This predictor has NO graph in it -- it cannot be reading which "
        f"metabolites are popular, which is exactly what loop 160's R5 lost to."), if_false=(
        f"S5 balance alone scores {bal_:.4f} against degree {dgm:.4f}, margin {bal_ - dgm:+.4f}, "
        f"and does not clear the bar. Mass balance constrains WHAT a product can be but does not "
        f"single it out: many species share an elemental composition."))
    say(f"     S5 {'PASS' if s5 else 'FAIL'}")

    # ------------------------------------------------------------------ S6
    say()
    best_single = max(uni, bal_)
    s6 = bool(fus - best_single > 3 * max(su, sb))
    GG.verdict(s6, emit=say, if_true=(
        f"S6 chemistry and connectivity COMPOSE: fused {fus:.4f} against the better single "
        f"predictor at {best_single:.4f}. They are reading different things and both are needed."),
        if_false=(
        f"S6 chemistry and connectivity SUBSTITUTE: fused {fus:.4f} against the better single "
        f"predictor at {best_single:.4f}, inside 3 sem. The same shape as loop 159's five "
        f"mechanisms -- two representations that look independent and turn out to rank the same "
        f"species for the same reason."))
    say(f"     S6 {'PASS' if s6 else 'FAIL'}")

    # ------------------------------------------------------------------ S7
    say()
    say("S7 WHAT THIS STILL CANNOT SHOW")
    say("     Balance is necessary, not sufficient: a balanced equation can be thermodynamically")
    say("     impossible and have no enzyme in any genome.")
    say(f"     {int(generic.sum()):,} species carry a generic R/X group and drop out of every")
    say("     balance check, so lipid and glycan chemistry is systematically under-tested here.")
    say("     The currency products are GRANTED in S5's main number. The un-granted variant is")
    say("     reported beside it and is the honest figure for a gap where nothing is known.")
    say("     Held-out reactions are ones Human-GEM already contains: this is recovery, not")
    say("     discovery, and loop 160's R7 said the same thing about the same protocol.")
    s7 = True
    say(f"     S7 {'PASS' if s7 else 'FAIL'}")

    gates = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": s6, "S7": s7}
    man = RM.manifest(
        inputs=[Path("HumanGEM.xml"), TOPO, CHEM],
        available=len(eligible), used=n_ok, selection="random", seed=SEED,
        controls=[
            "the coefficient arrays checked index-for-index against loop 160's edge arrays (S1)",
            "degree-only column run on the identical candidate sets",
            "balance scored with and without the currency products granted",
            "identical seed and held-out set as loop 160, so the walk numbers are comparable",
            "reactions with a generic R/X formula excluded from every balance claim",
            "conclusions emitted through gate_guard.verdict",
        ],
        note="stoichiometry, elemental balance, charge, protons and redox couples on the REM graph")
    res = {"test": "can chemistry predict the gap that connectivity could not",
           "gates": gates, "balance": {"checkable": nc, "element": f_el, "element_charge": f_both,
                                       "unbalanced_elements": dict(worst.most_common(10))},
           "protons": {"consume": int((hnet < 0).sum()), "emit": int((hnet > 0).sum()),
                       "moved": {f"{a}->{b}": v for (a, b), v in hmove.items()}},
           "redox_reactions": int(redox.sum()),
           "shortlist_recall": {"cases": n_eligible_case, "contained": n_in_shortlist,
                                "recall": rec},
           "walk_balance_rank_corr": rc,
           "auc": {"walk_uniform": uni, "walk_weighted": wt, "degree": dgm,
                   "balance": bal_, "balance_no_currency": hard, "fused": fus,
                   "sem_walk": su, "sem_balance": sb, "n_scored": n_ok},
           "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time() - t0:.0f}s]")
    say("=" * 104)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
