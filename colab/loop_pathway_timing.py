"""Loop 194. Is the regulatory input to metabolism time-structured?

WHAT THIS IS AND, MORE IMPORTANTLY, WHAT IT IS NOT. Loop 190's census ended by saying every layer it
counted is a steady-state description. The metabolic layer is the worst offender and not by
accident: flux balance analysis is steady-state BY CONSTRUCTION. It solves for a flux distribution
that satisfies dx/dt = 0. There is no version of FBA that has a fourth dimension, so "metabolism in
4D" is not something this or any loop can deliver from Human-GEM.

What CAN be asked is one link upstream. Loop 190's D3 measured that 1,448 of 2,568 modelled enzymes
(56.4%) carry a curated TF regulator, so the chain TF -> enzyme transcript -> flux capacity is wired
for a majority of the metabolic layer. If that regulatory input is time-structured -- if enzymes
that work on the same chemistry are switched on together rather than independently -- then the
metabolic layer has a temporal organisation even though its solver cannot represent one. If it is
not, then transcriptional control of metabolism in this system is gene-by-gene and the steady-state
description loses nothing by ignoring time.

THE LIMIT, STATED BEFORE THE NUMBER AND NOT AFTER. There is no measured flux time course anywhere on
this disk. Transcript abundance is not flux: an enzyme can be transcribed without being translated,
translated without being active, and active without carrying flux, and metabolic control analysis
exists precisely because flux is usually not controlled by any single enzyme's level. A pass here
means the REGULATORY INPUT to metabolism is coordinated in time. It does not mean metabolic flux is,
and this loop's verdict must not be quoted as though it did.

THE GROUPING COMES FROM THE MODEL, NOT FROM AN ANNOTATION. The gene table carries one pathway string
per gene, which is thin and externally curated. Human-GEM itself gives a better grouping for free:
two enzymes are coupled if their reactions share a metabolite. That is the actual chemistry rather
than somebody's naming of it, and it needs no external file.

    THE HUB PROBLEM, and it is the whole methodological risk. H+ appears in 2,722 reactions, H2O in
    2,020, ATP in 885. Coupling through those makes almost every enzyme a neighbour of almost every
    other, and a near-complete graph tests nothing -- every "within-group" statistic becomes the
    global statistic and the null becomes the observation. 126 of 8,461 metabolites carry degree
    above 50. They are excluded, the threshold is declared here rather than tuned, V2 gates on the
    resulting graph not being near-complete, and V4 repeats the whole test at other thresholds so
    the answer cannot depend on this one choice.

PREDECLARED, BEFORE ANY NUMBER.

  V1 DOES THE METABOLIC LAYER RESPOND AT ALL? Human-GEM genes among the dexamethasone responders,
     on the post-boundary grid loop 191c established is the usable half of that series.
     Gate: PASS iff at least 100 modelled enzymes respond. Below that nothing downstream has power
     and a null would be about the sample.

  V2 IS THE COUPLING GRAPH SANE? Edges between metabolic genes sharing a non-hub metabolite.
     Gate: PASS iff the graph has at least 500 edges AND a density below 0.10. A near-complete
     graph is the failure mode this whole construction risks, and a test run on one would report
     the global mean as though it were a within-group mean.

  V3 IS TIMING COHERENT ACROSS COUPLED ENZYMES? The mean absolute difference in response time over
     coupled pairs, against a null that shuffles response times among metabolic responders while
     holding the graph fixed.
     Gate: PASS iff z > 3.0, where coherence means a SMALLER mean difference than the null, so
     z = (null mean - observed) / null sd.

  V4 IS IT ROBUST TO THE HUB THRESHOLD? V3 repeated at hub degrees 25, 50 and 100.
     Gate: PASS iff z > 3.0 at every threshold tested. An answer that depends on where the hub line
     is drawn is an answer about the line.

  V5 DOES THE CURATED GROUPING AGREE? The same test with the gene table's pathway strings instead
     of metabolite sharing.
     Gate: PASS iff z > 3.0. Two independent groupings agreeing is worth more than either alone,
     and if they disagree that is a finding about which grouping carries the structure.

  V6 IS IT JUST EXPRESSION LEVEL? Coupled enzymes may simply be co-expressed, and genes at similar
     abundance may share apparent timing for measurement reasons rather than regulatory ones. V3
     repeated against a null that permutes response times only WITHIN baseline-expression deciles.
     Gate: PASS iff z > 3.0 against that stratified null. A pass in V3 that dies here is an
     abundance effect wearing a coordination costume -- the same shape as loop 191d's magnitude
     confound, which is why it gets its own gate rather than a footnote.

  V7 WHAT THIS CANNOT SHOW.

-> outputs/loop_pathway_timing.json
"""
import gzip
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
import loop_response_timing_d as L191        # noqa: E402
import loop_timing_statistic as L196         # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_pathway_timing.json"
SP = L191.SP
A549 = SP / "grtc"
TABLE = Path("colab/data/cell_complete.json.gz")

HUB_DEG = 50              # primary; V4 repeats at these
HUB_LEVELS = (25, 50, 100)
MIN_RESP = 100            # V1
MIN_EDGES = 500           # V2
MAX_DENSITY = 0.10        # V2 -- a near-complete graph tests nothing
N_PERM = 1000
Z_BAR = 3.0
N_DECILES = 10
MIN_PLATEAU = 0.5
MIN_TPM = 1.0
T_MIN = 30.0              # the batch boundary loop 191c established
REPS = (1, 2, 3)
SEED = 194194

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def coupling_edges(hub_deg, gene_of_rx, report=print):
    """Metabolic gene pairs sharing a non-hub metabolite.

    Returns the edge set and the number of metabolites excluded as hubs. Excluding by DEGREE rather
    than by a hand-written list of cofactors is deliberate: a curated cofactor list is an opinion
    about which chemistry is 'currency', and a degree cut is a statement about this model."""
    b = np.load("colab/data/rem_bipartite.npz", allow_pickle=True)
    deg = Counter(b["react_sp"].tolist()) + Counter(b["prod_sp"].tolist())
    hubs = {sp for sp, n in deg.items() if n > hub_deg}
    by_sp = defaultdict(set)
    for arr_rx, arr_sp in (("react_rx", "react_sp"), ("prod_rx", "prod_sp")):
        for rx, sp in zip(b[arr_rx], b[arr_sp]):
            if int(sp) in hubs:
                continue
            for g in gene_of_rx.get(int(rx), ()):
                by_sp[int(sp)].add(g)
    edges = set()
    for sp, gs in by_sp.items():
        gs = sorted(gs)
        if len(gs) > 200:            # a non-hub metabolite touching 200 genes is still a hub here
            continue
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                edges.add((gs[i], gs[j]))
    report(f"     hub degree > {hub_deg}: {len(hubs):,} metabolites excluded; "
           f"{len(edges):,} gene pairs coupled")
    return edges, len(hubs)


def coherence(edges, t, idx, rng, n_perm=N_PERM, strata=None):
    """Mean |delta response time| over edges, against a permutation null holding the graph fixed.

    `strata` restricts the permutation to within-stratum swaps, which is how V6 asks whether the
    coherence is really about abundance."""
    e = np.array([(idx[a], idx[b]) for a, b in edges if a in idx and b in idx], dtype=int)
    if len(e) < 10:
        return None
    obs = float(np.mean(np.abs(t[e[:, 0]] - t[e[:, 1]])))
    draws = np.empty(n_perm)
    order = np.arange(len(t))
    for k in range(n_perm):
        if strata is None:
            p = rng.permutation(order)
        else:
            p = order.copy()
            for s in np.unique(strata):
                m = np.where(strata == s)[0]
                p[m] = rng.permutation(m)
        tt = t[p]
        draws[k] = np.mean(np.abs(tt[e[:, 0]] - tt[e[:, 1]]))
    mu, sd = float(draws.mean()), float(draws.std(ddof=1))
    z = (mu - obs) / sd if sd > 0 else float("nan")
    return dict(n_edges=int(len(e)), observed=obs, null_mean=mu, null_sd=sd, z=float(z))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 194  IS THE REGULATORY INPUT TO METABOLISM TIME-STRUCTURED?")
    say("=" * 104)
    say("  PREDECLARED: flux balance is steady-state BY CONSTRUCTION, so this loop does not and")
    say("  cannot put metabolism in 4D. It asks one link upstream, whether enzymes working on the")
    say("  same chemistry are switched on together in time. There is NO measured flux time course")
    say("  on this disk and transcript is not flux, so a pass means the regulatory INPUT is")
    say("  coordinated and must not be quoted as though flux were. The grouping comes from the")
    say(f"  model's own metabolite sharing with hubs excluded at degree > {HUB_DEG}; V2 gates on")
    say("  the graph not being near-complete, V4 repeats at other thresholds, V5 checks a curated")
    say("  grouping independently, and V6 repeats everything against a null stratified by baseline")
    say(f"  expression. Every coherence gate is z > {Z_BAR} against {N_PERM} permutations.")
    say()

    z = np.load("colab/data/rem_enzyme.npz", allow_pickle=True)
    met_sym = np.array([str(s).upper() for s in z["symbols"]])
    gene_of_rx = defaultdict(set)
    for rx, g in zip(z["gpr_rx"], z["gpr_gene"]):
        gene_of_rx[int(rx)].add(str(met_sym[int(g)]))
    say(f"    Human-GEM: {len(z['reactions']):,} reactions, {len(met_sym):,} genes, "
        f"{len(gene_of_rx):,} reactions with a gene")

    # ---- the A549 response times, on the usable half of the series ----------------------------
    rz = np.load(A549 / "rna.npz", allow_pickle=True)
    tpm = rz["tpm"]
    ensg = np.array([str(g).split(".")[0] for g in rz["genes"]])
    mins, reps = rz["mins"].astype(int), rz["reps"].astype(int)
    allt = sorted(set(mins.tolist()))
    comp = {t: set(reps[mins == t].tolist()) for t in allt}
    grid = np.array([t for t in allt if set(REPS) <= comp[t] and t >= T_MIN], dtype=float)
    say(f"    A549 grid (post-boundary, constant composition): {[int(x) for x in grid]}")
    M, nrep = L191.rep_trajectories(tpm, mins, reps, REPS, grid)
    e2s = L191.ensg_to_symbol(say)
    sym = np.array([e2s.get(g, "") for g in ensg])
    base = tpm[(mins == int(grid[0])) & np.isin(reps, REPS)].mean(0)
    th, pl = L196.st_half_plateau(M, grid)
    responder = (base >= MIN_TPM) & (np.abs(pl) >= MIN_PLATEAU) & np.isfinite(th)

    # ---- V1 ------------------------------------------------------------------------------------
    say()
    say("V1 DOES THE METABOLIC LAYER RESPOND AT ALL?")
    metset = set(met_sym.tolist())
    is_met = np.array([s in metset for s in sym])
    m_resp = responder & is_met
    say(f"     {int(responder.sum()):,} responders overall; {int(m_resp.sum()):,} are modelled "
        f"enzymes ({int(is_met.sum()):,} Human-GEM genes present in the RNA table)")
    say(f"     median response time, metabolic {np.median(th[m_resp]):.0f} min vs "
        f"non-metabolic {np.median(th[responder & ~is_met]):.0f} min")
    v1 = bool(m_resp.sum() >= MIN_RESP)
    GG.verdict(v1, emit=say,
               if_true=f"V1 PASS -- {int(m_resp.sum()):,} modelled enzymes respond, above the "
                       f"{MIN_RESP} floor",
               if_false=f"V1 FAIL -- {int(m_resp.sum()):,} against a floor of {MIN_RESP}")

    void = set()
    if not v1:
        void |= {"V2", "V3", "V4", "V5", "V6"}

    # one row per responding metabolic SYMBOL
    order = [i for i in np.where(m_resp)[0]]
    seen, keep = set(), []
    for i in order:
        if sym[i] not in seen:
            seen.add(sym[i])
            keep.append(i)
    keep = np.array(keep)
    gidx = {sym[i]: k for k, i in enumerate(keep)}
    tvec = th[keep]
    bvec = base[keep]
    say(f"     {len(keep):,} distinct responding metabolic symbols carried forward")

    # ---- V2 ------------------------------------------------------------------------------------
    say()
    say("V2 IS THE COUPLING GRAPH SANE?")
    edges, n_hub = coupling_edges(HUB_DEG, gene_of_rx, say)
    inedges = [(a, b) for a, b in edges if a in gidx and b in gidx]
    n = len(gidx)
    dens = 2.0 * len(inedges) / max(n * (n - 1), 1)
    deg = Counter()
    for a, b in inedges:
        deg[a] += 1
        deg[b] += 1
    say(f"     among the {n:,} responding metabolic genes: {len(inedges):,} edges, "
        f"density {dens:.4f}, median degree {np.median(list(deg.values()) or [0]):.0f}")
    v2 = bool(len(inedges) >= MIN_EDGES and dens < MAX_DENSITY)
    if "V2" in void:
        say("     V2 VOID -- V1 failed")
    else:
        GG.verdict(v2, emit=say,
                   if_true=f"V2 PASS -- {len(inedges):,} edges at density {dens:.4f}, so a "
                           f"within-group statistic is not the global statistic in disguise",
                   if_false=f"V2 FAIL -- {len(inedges):,} edges at density {dens:.4f}; "
                            f"a graph this {'sparse' if len(inedges) < MIN_EDGES else 'dense'} "
                            f"cannot support the test")
    if not v2:
        void |= {"V3", "V4", "V5", "V6"}

    rng = np.random.default_rng(SEED)

    # ---- V3 ------------------------------------------------------------------------------------
    say()
    say("V3 IS TIMING COHERENT ACROSS COUPLED ENZYMES?")
    d3 = None
    v3 = False
    if "V3" in void:
        say("     V3 VOID -- see above")
    else:
        d3 = coherence(inedges, tvec, gidx, rng)
        say(f"     observed mean |delta t| over {d3['n_edges']:,} coupled pairs: "
            f"{d3['observed']:.1f} min")
        say(f"     null {d3['null_mean']:.1f} +/- {d3['null_sd']:.1f} over {N_PERM} permutations "
            f"-> z {d3['z']:+.1f}")
        v3 = bool(np.isfinite(d3["z"]) and d3["z"] > Z_BAR)
        GG.verdict(v3, emit=say,
                   if_true="V3 PASS -- enzymes sharing chemistry respond closer together in time "
                           "than the same response times redistributed over the same graph",
                   if_false=f"V3 FAIL -- z {d3['z']:+.1f}; coupled enzymes are no closer in time "
                            f"than chance, so transcriptional control of metabolism here is "
                            f"gene-by-gene and the steady-state description loses nothing")

    # ---- V4 ------------------------------------------------------------------------------------
    say()
    say("V4 IS IT ROBUST TO THE HUB THRESHOLD?")
    d4 = {}
    if "V4" in void:
        say("     V4 VOID -- see above")
    else:
        for h in HUB_LEVELS:
            ee, _ = coupling_edges(h, gene_of_rx, lambda *_: None)
            ie = [(a, b) for a, b in ee if a in gidx and b in gidx]
            dd = 2.0 * len(ie) / max(n * (n - 1), 1)
            r = coherence(ie, tvec, gidx, np.random.default_rng(SEED + h), n_perm=300)
            d4[h] = dict(edges=len(ie), density=dd, **(r or {}))
            say(f"     hub>{h:>3d}: {len(ie):>7,} edges  density {dd:.4f}  "
                + (f"z {r['z']:+.1f}" if r else "too few edges"))
        zs = [v.get("z") for v in d4.values() if v.get("z") is not None]
        v4 = bool(zs and all(np.isfinite(x) and x > Z_BAR for x in zs))
        GG.verdict(v4, emit=say,
                   if_true="V4 PASS -- the direction holds at every hub threshold, so the answer "
                           "is not about where the hub line was drawn",
                   if_false="V4 FAIL -- the result depends on the hub threshold, which makes it an "
                            "answer about the threshold rather than about the chemistry")
    v4 = bool(d4) and all(v.get("z") is not None and np.isfinite(v["z"]) and v["z"] > Z_BAR
                          for v in d4.values())

    # ---- V5 ------------------------------------------------------------------------------------
    say()
    say("V5 DOES THE CURATED GROUPING AGREE?")
    d5, v5 = None, False
    if "V5" in void:
        say("     V5 VOID -- see above")
    else:
        tab = json.load(gzip.open(TABLE))["genes"]
        path_of = {str(g["name"]).upper(): (g.get("path") or "").strip() for g in tab}
        by_path = defaultdict(list)
        for s in gidx:
            p = path_of.get(s, "")
            if p:
                by_path[p].append(s)
        pedges = []
        for p, gs in by_path.items():
            if 1 < len(gs) <= 200:
                for i in range(len(gs)):
                    for j in range(i + 1, len(gs)):
                        pedges.append((gs[i], gs[j]))
        say(f"     {len(by_path):,} curated pathways cover {sum(len(v) for v in by_path.values()):,} "
            f"of the {n:,} responding metabolic genes; {len(pedges):,} within-pathway pairs")
        d5 = coherence(pedges, tvec, gidx, np.random.default_rng(SEED + 1))
        if d5 is None:
            void.add("V5")
            say("     V5 VOID -- too few within-pathway pairs")
        else:
            say(f"     observed {d5['observed']:.1f} min, null {d5['null_mean']:.1f} "
                f"+/- {d5['null_sd']:.1f} -> z {d5['z']:+.1f}")
            v5 = bool(np.isfinite(d5["z"]) and d5["z"] > Z_BAR)
            GG.verdict(v5, emit=say,
                       if_true="V5 PASS -- an independently curated grouping finds the same "
                               "structure the model's own chemistry does",
                       if_false="V5 FAIL -- the curated pathway labels do not carry the timing "
                                "structure. If V3 passed, the structure is in the chemistry and "
                                "not in the naming")

    # ---- V6 ------------------------------------------------------------------------------------
    say()
    say("V6 IS IT JUST EXPRESSION LEVEL?")
    d6, v6 = None, False
    if "V6" in void:
        say("     V6 VOID -- see above")
    else:
        q = np.quantile(bvec, np.linspace(0, 1, N_DECILES + 1))
        q[-1] += 1e-9
        strata = np.clip(np.searchsorted(q, bvec, side="right") - 1, 0, N_DECILES - 1)
        say(f"     permuting response times only WITHIN {N_DECILES} baseline-expression deciles")
        d6 = coherence(inedges, tvec, gidx, np.random.default_rng(SEED + 2), strata=strata)
        say(f"     observed {d6['observed']:.1f} min, stratified null {d6['null_mean']:.1f} "
            f"+/- {d6['null_sd']:.1f} -> z {d6['z']:+.1f}")
        v6 = bool(np.isfinite(d6["z"]) and d6["z"] > Z_BAR)
        GG.verdict(v6, emit=say,
                   if_true="V6 PASS -- the coherence survives a null that can only swap genes of "
                           "similar abundance, so it is coordination and not an abundance effect",
                   if_false="V6 FAIL -- against an abundance-matched null the coherence goes. "
                            "Coupled enzymes are co-expressed and co-expressed genes look "
                            "co-timed; V3's result is about abundance")

    # ---- V7 ------------------------------------------------------------------------------------
    say()
    say("V7 WHAT THIS CANNOT SHOW")
    say("     TRANSCRIPT IS NOT FLUX, and this is the limit that matters. An enzyme can be")
    say("     transcribed without being translated, translated without being active, and active")
    say("     without carrying flux. Metabolic control analysis exists because flux is usually not")
    say("     set by any one enzyme's level. Nothing here licenses a claim about flux dynamics.")
    say("     Flux balance remains steady-state by construction. Whatever this loop finds, the")
    say("     solver in this project still cannot represent a trajectory, so a pass describes the")
    say("     INPUT to a layer that has no time axis of its own.")
    say("     One stimulus in one cell line, and a glucocorticoid response is not a metabolic")
    say("     perturbation. Enzymes may be coordinated here because GR happens to hit them")
    say("     together, which is a fact about dexamethasone rather than about metabolism.")
    say("     Metabolite sharing is not pathway membership. Two enzymes joined by an uncommon")
    say("     metabolite may sit in different subsystems, and the degree cut that removes H+ and")
    say("     ATP is a threshold, not a definition of what counts as currency.")
    say("     The response time is loop 196's incumbent statistic, which that loop measured to be")
    say("     sound on this eleven-point grid and unusable below about eight points. Nothing here")
    say("     extends to a coarser series.")
    say("     V7 PASS")

    gates = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6, "V7": True}
    man = RM.manifest(inputs=[A549 / "rna.npz", Path("colab/data/rem_enzyme.npz"),
                              Path("colab/data/rem_bipartite.npz")],
                      available=int(len(sym)), used=int(len(keep)),
                      selection="filtered", seed=SEED,
                      controls=[f"{N_PERM} permutations holding the coupling graph fixed",
                                "three hub thresholds",
                                "an independently curated grouping",
                                "a null stratified by baseline expression decile"],
                      note="timing coherence among enzymes sharing chemistry, A549 dexamethasone")
    out_d = dict(test="pathway timing coherence", gates=gates, void=sorted(void),
                 grid=[int(x) for x in grid], n_metabolic_responders=int(m_resp.sum()),
                 n_symbols=len(keep), hub_deg=HUB_DEG, n_hubs=n_hub,
                 graph=dict(edges=len(inedges), density=dens),
                 v3=d3, v4={str(k): v for k, v in d4.items()}, v5=d5, v6=d6,
                 manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'VOID' if k in void else ('PASS' if v else 'FAIL')}")
    scored = [k for k in gates if k not in void]
    say(f"  {sum(gates[k] for k in scored)}/{len(scored)}   [{time.time()-t0:.0f}s]"
        + (f"   ({len(void)} VOID: {', '.join(sorted(void))})" if void else ""))
    say("=" * 104)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
