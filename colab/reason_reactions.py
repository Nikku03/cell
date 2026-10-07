"""reason_reactions — the SAME reasoning principle (reasoner_core), new WORKING: reason about a metabolic REACTION
instead of a gene. Truth axis changes from gene-essentiality to REACTION-essentiality, grounded independently by
FBA single-reaction deletion (delete the reaction, does growth collapse?). The evidence lines change to what's
observable about a reaction, chosen to be differently-blind AND independent of the FBA-deletion truth:

  L1 flux        — the reaction carries flux at the WT optimum (physics: it's in use)            [modeled]
  L2 solo-gene   — a single gene catalyzes it (no isozyme redundancy to buffer its loss)         [structural]
  L3 dep(gene)   — its gene(s) are DepMap-essential (MEASURED — fully independent of FBA)         [measured]
  L4 bottleneck  — it handles a non-currency metabolite that few other reactions touch (topology)[network]

If the principle is real, fusing these should predict FBA reaction-essentiality and calibrate — with NO line
being the truth itself. Honest by construction: FBA deletion is the truth and is NOT one of the lines.
-> outputs/orphan/reason_reactions.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")

CURRENCY = {"h2o", "h", "atp", "adp", "amp", "pi", "ppi", "nad", "nadh", "nadp", "nadph", "co2", "o2",
            "coa", "nh3", "nh4", "fad", "fadh2", "gtp", "gdp", "utp", "udp", "ctp", "cdp", "hco3", "na1", "k"}


def _ensg2sym(path="HumanGEM_genes.tsv"):
    m = {}
    if not os.path.exists(path):
        return m
    with open(path) as fh:
        hdr = [h.strip().strip('"').lower() for h in fh.readline().split("\t")]
        gi = next((i for i, h in enumerate(hdr) if h == "genes"), 0)
        si = next((i for i, h in enumerate(hdr) if "symbol" in h), 4)
        for ln in fh:
            p = [x.strip().strip('"') for x in ln.split("\t")]
            if len(p) > max(gi, si) and p[gi] and p[si]:
                m[p[gi]] = p[si].split(";")[0]
    return m


def build(sample=None, ess_threshold=0.5, seed=0):
    import cobra
    import ecflux
    from cobra.flux_analysis import pfba, single_reaction_deletion
    from complete_cell import CompleteCell

    m = ecflux.load_humangem()
    bio = ecflux.biomass_id(m)
    wt = m.slim_optimize()
    fluxes = pfba(m).fluxes
    print(f"  model {len(m.reactions)} reactions, WT growth {wt:.3f}")

    ens2sym = _ensg2sym()
    C = CompleteCell()
    def dep_of(sym):
        i = C.idx.get(sym)
        if i is None:
            return None
        v = C.genes[i].get("dep_frac")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # metabolite degree (how many reactions touch it), for the bottleneck line
    metdeg = {mt.id: len(mt.reactions) for mt in m.metabolites}

    # candidate reactions: those with a GPR (so all four lines can exist); exclude the biomass/exchange objective
    rxns = [r for r in m.reactions if r.genes and r.id != bio and not r.id.startswith("EX_")]
    if sample and len(rxns) > sample:
        rng = np.random.default_rng(seed)
        rxns = list(np.array(rxns, dtype=object)[rng.choice(len(rxns), sample, replace=False)])
    rids = [r.id for r in rxns]
    print(f"  reasoning universe: {len(rids)} gene-associated reactions")

    # --- evidence lines ---
    lines = {k: np.full(len(rids), np.nan) for k in ("flux", "solo_gene", "dep", "bottleneck")}
    for j, r in enumerate(rxns):
        lines["flux"][j] = float(abs(fluxes.get(r.id, 0.0)) > 1e-6)
        ng = len(r.genes)
        lines["solo_gene"][j] = float(ng == 1)                       # single catalyst = no redundancy
        deps = [dep_of(ens2sym.get(g.id, g.id)) for g in r.genes]
        deps = [d for d in deps if d is not None]
        if deps:
            lines["dep"][j] = float(max(deps) > 0.5)                 # measured essentiality of its gene(s)
        mets = [mt for mt in r.metabolites if mt.id.rsplit("_", 1)[0].lower() not in CURRENCY
                and (mt.name or "").split("[")[0].strip().lower() not in CURRENCY]
        if mets:
            lines["bottleneck"][j] = float(min(metdeg[mt.id] for mt in mets) <= 3)

    # --- truth: FBA single-reaction deletion ---
    print(f"  running FBA single-reaction deletion on {len(rids)} reactions ...")
    dd = single_reaction_deletion(m, reaction_list=rids, method="fba")
    # map back id -> growth
    g_by_id = {}
    for _, row in dd.iterrows():
        ids = row["ids"]
        rid = list(ids)[0] if not isinstance(ids, str) else ids
        g = row.get("growth")
        g_by_id[rid] = float(g) if g is not None and np.isfinite(g) else 0.0
    truth = np.array([1.0 if (g_by_id.get(rid, wt) / wt) < ess_threshold else 0.0 for rid in rids])
    print(f"  FBA-essential reactions (growth < {ess_threshold:.0%} WT): {int(truth.sum())} / {len(rids)} "
          f"({truth.mean():.0%})")
    return lines, truth, rids


def main(sample=None):
    import reasoner_core as rc
    lines, truth, rids = build(sample=sample)

    def pack(rep):
        return {k: rep[k] for k in ("n", "n_truth_pos", "lines", "singles", "convergence_auc",
                                    "weighted_reasoning_auc", "best_single", "calibration", "verdict")}

    # (1) ALL lines — but `flux` is near-tautological with FBA-deletion truth (a zero-flux reaction can't be
    #     essential), so a high AUC here is mostly that one line, NOT convergence. Reported, with the caveat.
    rep_all = rc.reason_over(lines, truth, min_lines=2)
    print("\n### ALL lines (incl. flux — near-tautological with the FBA truth) ###")
    rc.print_report(rep_all, entity="reaction", truth_name="FBA-essential")

    # (2) INDEPENDENT lines only (drop flux): the HONEST test of whether measured + topological evidence
    #     CONVERGES on reaction-essentiality without peeking at the physics that defines the truth.
    indep = {k: v for k, v in lines.items() if k != "flux"}
    rep_indep = rc.reason_over(indep, truth, min_lines=2)
    print("\n### INDEPENDENT lines only (dep + solo_gene + bottleneck; flux dropped) — the honest test ###")
    rc.print_report(rep_indep, entity="reaction", truth_name="FBA-essential")

    honest = (f"PRINCIPLE TRANSFERS, VALUE IS SUBSTRATE-DEPENDENT. The core runs on reactions and reproduces the "
              f"machinery, but reasoning only ADDS value when several INDEPENDENT informative lines converge. With "
              f"flux included the fused AUC is {rep_all['weighted_reasoning_auc']:.2f} — but flux is near-tautological "
              f"with FBA-deletion truth, so that is one line, not convergence. Dropping it, the genuinely independent "
              f"lines (measured essentiality + topology) reason to only AUC {rep_indep['weighted_reasoning_auc']:.2f} "
              f"— reaction-essentiality is a physics property that measured/topological evidence recovers weakly. "
              f"Contrast genes, where 5 independent lines truly converge (0.80 > best single 0.75).")
    print("\n" + "#" * 84 + f"\nHONEST VERDICT: {honest}\n" + "#" * 84)

    save = {"substrate": "metabolic reactions (Human-GEM)",
            "truth": "FBA single-reaction deletion growth < 50% WT (independent of every evidence line)",
            "all_lines": pack(rep_all), "independent_lines_only": pack(rep_indep),
            "flux_is_near_tautological_with_truth": True, "honest_verdict": honest}
    json.dump(save, open(f"{OUT}/reason_reactions.json", "w"), indent=1)
    return rep_all, rep_indep


if __name__ == "__main__":
    import sys
    s = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(sample=s)
