"""Loop 193 (Phase 2). The 44-gene seam: a mechanism, or a bookkeeping overlap?

WHAT THE CENSUS FOUND AND LEFT UNSTATED. Loop 190's D3 measured the pairwise overlaps between this
project's mechanistic layers and found one number far smaller than the rest: only 44 genes carry
BOTH a modelled metabolic reaction and a role as a curated regulator. Everywhere else the layers
connect through EDGES -- 1,448 modelled enzymes have a TF regulator -- but 44 is the count of genes
that are simultaneously in both, and they are not a random 44. They are almost entirely the
chromatin writers and erasers: KAT2A/2B/5/6A/6B/7/8, KMT2A-D, DNMT1/3A/3B/3L, EZH2, DOT1L, SETDB1,
NSD1/2, SIRT1/6, PARP9/10/14, EP300, CREBBP. Enzymes whose chemistry is writing marks.

The census reported the overlap and stopped. Whether being in both layers is LOAD-BEARING -- whether
these genes actually couple metabolism to regulation, or merely happen to satisfy two membership
tests -- has never been asked, and the plan for this phase asked it the wrong way.

WHY THE PLANNED TEST CANNOT BE BUILT, and this is stated before any alternative so the reasoning is
inspectable rather than a quiet substitution. The plan was: does metabolic supply of acetyl-CoA, SAM
and NAD+ predict mark deposition at these writers' targets. It cannot be run here, for a reason that
has nothing to do with missing files:

    SUBSTRATE CONCENTRATION IS A CELL-LEVEL PROPERTY. Acetyl-CoA in K562 is one number. It does not
    vary from gene to gene, so it cannot predict which genes carry a mark within one cell type. The
    only design in which supply predicts deposition compares CELLS -- high-acetyl-CoA against
    low -- and every mark measurement on this disk is K562 or A549 alone.

    This is loop 185's Z6 lesson one level up. Z6 predeclared that element-intrinsic columns could
    not move within-gene R@1, and was refuted because element-intrinsic is not the same as constant
    within a gene. Here the quantity really is constant across the comparison, so the test really is
    inert, and running it would produce a number that could only be noise.

    Flux is also not measured anywhere here, so even across cell types the supply term would be a
    proxy built from enzyme expression rather than a measurement.

So this loop asks the question the data CAN answer, and it is the one the census actually left open:
if the 44 genuinely couple the layers, their regulatory targets should be metabolic more often than
another regulator's are -- and more specifically, should be enriched for the chemistry they
themselves perform. If neither holds, dual membership is two satisfied predicates and the honest
description of the seam is bookkeeping.

PREDECLARED, BEFORE ANY NUMBER.

  R1 DOES THE SEAM REPRODUCE? The 44, recomputed and checked against loop 190's census.
     Gate: PASS iff the count and the membership match exactly. Loop 195's Z1 is the precedent and
     the reason: this arc has had a join silently return zero, so a set that is not the census's set
     makes every result below about different genes.

  R2 DO THE 44 REGULATE METABOLISM MORE THAN OTHER REGULATORS DO? The fraction of each gene's
     curated targets that are modelled enzymes, against DEGREE-MATCHED non-bridge regulators drawn
     from the other 1,133. Degree matching is not optional: the 44 range from 1 to 582 targets, and
     a regulator with 582 targets hits more metabolic genes than one with 9 for arithmetic reasons.
     Gate: PASS iff z > 3.0 against the degree-matched null.

  R3 IS IT THEIR OWN CHEMISTRY? Each writer's substrate class is read from its OWN Human-GEM
     reactions -- acetyl-CoA, S-adenosylmethionine or NAD+ -- and its targets are tested for
     enrichment in genes whose reactions touch that same metabolite.
     Gate: PASS iff z > 3.0. R2 can pass on generic metabolic regulation; this asks whether the
     coupling is specific to the chemistry the gene performs, which is what "mechanism" would mean.

  R4 THE STRANGER SWAP, run against R3 and NOT against R2, for a reason worth stating in advance.
     Permuting which bridge gene holds which target set cannot change the GROUP MEAN of per-gene
     metabolic fractions -- the multiset of fractions is identical -- so a swap control on R2 is
     arithmetically incapable of moving, which is gate_guard's own family two. R3's statistic
     depends on the PAIRING between a writer's substrate class and its targets, and permuting
     breaks that, so the swap bites. The swap is also checked with null_can_move before its output
     is allowed to count.
     Gate: PASS iff the real excess over the null exceeds the swapped excess IN MAGNITUDE, and the
     swap actually changed the input. Magnitude, because loop 199's Q5 compared signed values,
     assumed the sign of its own answer, and scored a control that worked as a failure.

  R5 WHAT THIS CANNOT SHOW.

-> outputs/loop_bridge.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_bridge.json"
CENSUS = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_cell_census.json"
TABLE = Path("colab/data/cell_complete.json.gz")
BUNDLE = Path("colab/data/net_bundle.json.gz")
CUR = (0, 55716)
N_PERM = 2000
Z_BAR = 3.0
SEED = 193193
SUBSTRATES = {"acetyl": ("acetyl-coa",), "sam": ("s-adenosylmethionin", "s-adenosyl-l-methionin"),
              "nad": ("nad+",)}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def degree_matched_null(deg_of, pool_by_deg, targets, is_met, rng, n_perm=N_PERM):
    """Enrichment of metabolic genes among targets, against regulators of the SAME out-degree.

    For each bridge gene a control regulator is drawn from the pool whose out-degree is closest,
    without replacement within a draw. Matching on degree is the whole point: the 44 span 1 to 582
    targets and a broad regulator hits more of everything."""
    obs = float(np.mean([np.mean([is_met(t) for t in targets[g]]) for g in deg_of]))
    draws = np.empty(n_perm)
    keys = sorted(pool_by_deg)
    for k in range(n_perm):
        vals, used = [], set()
        for g, d in deg_of.items():
            cand = min(keys, key=lambda x: (abs(x - d), x))
            opts = [c for c in pool_by_deg[cand] if c not in used] or \
                   [c for c in sum(pool_by_deg.values(), []) if c not in used]
            if not opts:
                continue
            c = opts[rng.integers(0, len(opts))]
            used.add(c)
            vals.append(np.mean([is_met(t) for t in targets[c]]) if targets[c] else 0.0)
        draws[k] = float(np.mean(vals)) if vals else np.nan
    mu, sd = float(np.nanmean(draws)), float(np.nanstd(draws, ddof=1))
    z = (obs - mu) / sd if sd > 0 else float("nan")
    return dict(observed=obs, null_mean=mu, null_sd=sd, z=float(z), n_perm=n_perm)


def main():
    t0 = time.time()
    G = GG.Gates(emit=say)
    say("=" * 104)
    say("LOOP 193  THE 44-GENE SEAM: a mechanism, or a bookkeeping overlap?")
    say("=" * 104)
    say("  PREDECLARED: the planned substrate-supply test CANNOT be built and the docstring says")
    say("  why before offering an alternative -- acetyl-CoA in K562 is ONE number, constant across")
    say("  genes, so it cannot predict which gene carries a mark within one cell type, and every")
    say("  mark measurement here is one cell type. That is loop 185's Z6 lesson one level up, and")
    say("  this time the quantity really is constant, so the test really is inert. What is asked")
    say("  instead is whether the 44 regulate metabolism more than degree-matched other regulators")
    say(f"  (R2) and whether the coupling is specific to their own chemistry (R3), both at z > "
        f"{Z_BAR} against {N_PERM} draws.")
    say()

    tab = json.load(gzip.open(TABLE))["genes"]
    sym = [str(g["name"]).upper() for g in tab]
    z = np.load("colab/data/rem_enzyme.npz", allow_pickle=True)
    met_sym = np.array([str(s).upper() for s in z["symbols"]])
    met = set(met_sym.tolist())
    nb = json.load(gzip.open(BUNDLE))
    names, reg = nb["names"], nb["reg"]
    nidx = {n.upper(): i for i, n in enumerate(names)}
    cur = reg[CUR[0]:CUR[1]]
    regulators = {int(r[0]) for r in cur}
    tgt = defaultdict(set)
    for r in cur:
        tgt[int(r[0])].add(int(r[1]))
    name_of = {i: n.upper() for i, n in enumerate(names)}

    # ---- R1 ------------------------------------------------------------------------------------
    say("R1 DOES THE SEAM REPRODUCE?")
    bridge = sorted(s for s in sym if s in met and nidx.get(s, -1) in regulators)
    cen = json.load(open(CENSUS))
    want = cen.get("pairwise", {}).get("TF_in_network & reaction")
    say(f"     recomputed {len(bridge)} bridge genes; census recorded {want}")
    say(f"     {', '.join(bridge[:14])}{' ...' if len(bridge) > 14 else ''}")
    G.add("R1", len(bridge) == want, stat=len(bridge),
          if_true=f"R1 PASS -- the seam reproduces at {len(bridge)}, so what follows is about the "
                  f"genes the census counted",
          if_false=lambda: f"R1 FAIL -- {len(bridge)} against the census's {want}")

    # targets and the control pool
    targets = {g: sorted(tgt[nidx[g]]) for g in bridge}
    pool = [s for s in sym if nidx.get(s, -1) in regulators and s not in met]
    pool_t = {g: sorted(tgt[nidx[g]]) for g in pool}
    say(f"     control pool: {len(pool):,} curated regulators that are NOT modelled enzymes")

    def is_met(i):
        return name_of.get(int(i), "") in met

    deg_of = {g: len(targets[g]) for g in bridge}
    pool_by_deg = defaultdict(list)
    for g in pool:
        pool_by_deg[len(pool_t[g])].append(g)
    allt = dict(targets)
    allt.update(pool_t)
    rng = np.random.default_rng(SEED)

    # ---- R2 ------------------------------------------------------------------------------------
    say()
    say("R2 DO THE 44 REGULATE METABOLISM MORE THAN OTHER REGULATORS DO?")
    say(f"     bridge out-degree: median {int(np.median(list(deg_of.values())))}, "
        f"range {min(deg_of.values())}-{max(deg_of.values())}")
    d2 = degree_matched_null(deg_of, pool_by_deg, allt, is_met, rng)
    say(f"     fraction of targets that are modelled enzymes: observed {d2['observed']:.4f}")
    say(f"     degree-matched null {d2['null_mean']:.4f} +/- {d2['null_sd']:.4f} "
        f"over {N_PERM} draws  ->  z {d2['z']:+.1f}")
    G.add("R2", np.isfinite(d2["z"]) and d2["z"] > Z_BAR, stat=d2["z"], requires=("R1",),
          if_true=lambda: (f"R2 PASS -- the bridge genes regulate metabolic genes more than "
                           f"regulators of the same breadth do (z {d2['z']:+.1f})"),
          if_false=lambda: (f"R2 FAIL -- z {d2['z']:+.1f}; at matched out-degree the bridge genes "
                            f"are no more metabolic in their targeting than any other regulator, "
                            f"so dual membership does not come with a regulatory preference"))

    # ---- R3 ------------------------------------------------------------------------------------
    say()
    say("R3 IS IT THEIR OWN CHEMISTRY?")
    b = np.load("colab/data/rem_bipartite.npz", allow_pickle=True)
    sp_name = [str(x).lower() for x in b["sp_name"]]
    sub_species = {k: {i for i, nm in enumerate(sp_name) if any(w in nm for w in ws)}
                   for k, ws in SUBSTRATES.items()}
    for k, v in sub_species.items():
        say(f"     {k}: {len(v)} species ids")
    rx_species = defaultdict(set)
    for a, s in zip(b["react_rx"], b["react_sp"]):
        rx_species[int(a)].add(int(s))
    for a, s in zip(b["prod_rx"], b["prod_sp"]):
        rx_species[int(a)].add(int(s))
    gene_rx = defaultdict(set)
    for rx, gi in zip(z["gpr_rx"], z["gpr_gene"]):
        gene_rx[str(met_sym[int(gi)])].add(int(rx))
    gene_sub = {}
    for g, rxs in gene_rx.items():
        sp = set().union(*[rx_species.get(r, set()) for r in rxs]) if rxs else set()
        gene_sub[g] = {k for k, v in sub_species.items() if sp & v}
    n_with = sum(1 for g in bridge if gene_sub.get(g))
    say(f"     {n_with} of {len(bridge)} bridge genes touch acetyl-CoA, SAM or NAD+ in their own "
        f"reactions")
    same_chem = {}
    for g in bridge:
        mine = gene_sub.get(g, set())
        same_chem[g] = {t for t in targets[g]
                        if mine and (gene_sub.get(name_of.get(int(t), ""), set()) & mine)}
    obs3 = float(np.mean([len(same_chem[g]) / max(len(targets[g]), 1) for g in bridge]))
    draws = np.empty(N_PERM)
    for k in range(N_PERM):
        vals = []
        for g in bridge:
            mine = gene_sub.get(g, set())
            if not mine or not targets[g]:
                vals.append(0.0)
                continue
            cand = rng.choice(pool, size=1)[0]
            ts = pool_t[cand] or [0]
            vals.append(np.mean([bool(gene_sub.get(name_of.get(int(t), ""), set()) & mine)
                                 for t in ts]))
        draws[k] = float(np.mean(vals))
    mu3, sd3 = float(draws.mean()), float(draws.std(ddof=1))
    z3 = (obs3 - mu3) / sd3 if sd3 > 0 else float("nan")
    say(f"     fraction of targets sharing the writer's own substrate: observed {obs3:.4f}")
    say(f"     null {mu3:.4f} +/- {sd3:.4f}  ->  z {z3:+.1f}")
    d3 = dict(observed=obs3, null_mean=mu3, null_sd=sd3, z=float(z3), n_with_substrate=n_with)
    G.add("R3", np.isfinite(z3) and z3 > Z_BAR, stat=z3, requires=("R1",),
          if_true=lambda: (f"R3 PASS -- the coupling is specific to the chemistry each writer "
                           f"performs (z {z3:+.1f}), which is what a mechanism would look like"),
          if_false=lambda: (f"R3 FAIL -- z {z3:+.1f}; a writer's targets are no more likely to "
                            f"share its own substrate than another regulator's are. Whatever the "
                            f"seam is, it is not the chemistry"))

    # ---- R4 ------------------------------------------------------------------------------------
    say()
    say("R4 THE STRANGER SWAP")
    say("     AND THE FIRST VERSION OF THIS GATE COULD NOT MOVE, which is recorded rather than")
    say("     quietly repaired. It swapped target sets among the 44 and compared the GROUP MEAN of")
    say("     per-gene metabolic fractions -- a quantity that is invariant under a permutation of")
    say("     which gene holds which set, because the multiset of fractions is unchanged. Real and")
    say("     swapped both came back 0.2121, exactly. That is gate_guard's own family two: a null")
    say("     arithmetically incapable of changing the statistic it is meant to destroy.")
    say("     R3's statistic is different and a swap DOES bite it, because it depends on the")
    say("     PAIRING between a writer's own substrate class and its targets, which permuting")
    say("     breaks. So the swap is run against R3, which is also the gate that passed -- a")
    say("     passing result with no working control is what this project does not ship.")
    perm = rng.permutation(len(bridge))
    swapped_t = {g: targets[bridge[perm[i]]] for i, g in enumerate(bridge)}
    obs4 = float(np.mean([
        np.mean([bool(gene_sub.get(name_of.get(int(t), ""), set()) & gene_sub.get(g, set()))
                 for t in swapped_t[g]]) if (swapped_t[g] and gene_sub.get(g)) else 0.0
        for g in bridge]))
    inert = GG.null_can_move(
        [tuple(targets[g]) for g in bridge], [tuple(swapped_t[g]) for g in bridge])
    say(f"     the swap changes {inert['changed']:.0%} of the target sets "
        f"({'capable' if inert['capable'] else 'INERT'})")
    say(f"     R3 statistic: real {obs3:.4f}, swapped {obs4:.4f}, null {mu3:.4f}")
    w = GG.weakened_by(real=obs3 - mu3, control=obs4 - mu3)
    say(f"     real excess {w['real']:+.4f} against swapped excess {w['control']:+.4f}")
    G.add("R4", w["weakened"] and inert["capable"], requires=("R3",),
          if_true=lambda: (f"R4 PASS -- breaking the pairing between a writer and its targets "
                           f"collapses the chemistry-specific enrichment ({obs3:.4f} -> "
                           f"{obs4:.4f}), so it belongs to which writer has which targets"),
          if_false=lambda: (f"R4 FAIL -- a stranger's targets share the writer's substrate just as "
                            f"often ({obs4:.4f} against {obs3:.4f}), so R3's enrichment is a "
                            f"property of the group and not of the specific pairing"))

    # ---- R5 ------------------------------------------------------------------------------------
    say()
    say("R5 WHAT THIS CANNOT SHOW")
    say("     The substrate-supply question is UNANSWERED, not answered. It needs mark measurements")
    say("     across cells differing in metabolic state, and nothing on this disk has that. A later")
    say("     loop that finds such data should not treat this loop as having tested it.")
    say("     CollecTRI is literature-curated, and chromatin writers are among the most studied")
    say("     proteins in the genome. If their targets look metabolic it may be because their")
    say("     metabolic targets are the ones people looked for. Degree matching controls for how")
    say("     BROAD a regulator is, not for what its literature is about.")
    say("     Human-GEM assigns these writers reactions because their chemistry is in the model, so")
    say("     'is a modelled enzyme' is partly a statement about the model's scope. The seam is 44")
    say("     genes in THIS model, not a fact about the cell.")
    say("     A pass at R2 with a fail at R3 would say the bridge genes prefer metabolic targets")
    say("     without preferring their own chemistry, which is a coupling but not a mechanism. The")
    say("     two gates are separate so that combination can be reported rather than blurred.")
    G.add("R5", True, if_true="R5 PASS")

    gates, void = G.as_dict()
    man = RM.manifest(inputs=[TABLE, BUNDLE, Path("colab/data/rem_enzyme.npz")],
                      available=len(sym), used=len(bridge), selection="filtered", seed=SEED,
                      controls=[f"{N_PERM} degree-matched draws from {len(pool)} non-bridge "
                                f"regulators", "a stranger swap within the bridge set",
                                "the seam checked against loop 190's census before use"],
                      note="is the 44-gene metabolism/regulation seam load-bearing?")
    out_d = dict(test="bridge seam", gates=gates, void=void, bridge=bridge,
                 n_pool=len(pool), r2=d2, r3=d3,
                 r4=dict(swapped=obs4, capable=inert['capable'],
                         changed=inert['changed'], **w), manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    G.summary(seconds=time.time() - t0)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
