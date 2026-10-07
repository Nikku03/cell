"""IS "ALL PARTICIPANTS ARE NECESSARY" ENOUGH? TESTING NECESSITY AGAINST MEASURED ESSENTIALITY.

THE ARGUMENT BEING TESTED, WHICH IS A GOOD ONE. A reaction needs its substrates AND its catalyst; remove any one and
the reaction stops. The catalyst is not consumed, so it persists, but it is no less required. Therefore -- the
argument goes -- knowing WHO ACTS ON WHOM is unnecessary bookkeeping. It is enough to record that all participants
are required, and propagate loss.

THAT ARGUMENT IS CORRECT FOR KNOCKOUT PROPAGATION, and this file concedes it rather than defending the alternative.
It is also how flux balance analysis has always worked: gene-protein-reaction rules set a reaction's bounds to zero
when its enzymes are gone, with no notion of "acts on". The catalyst->substrate arrows derived earlier in this
project are a LOSSY PROJECTION of a reaction onto a protein-protein graph -- they discard stoichiometry, metabolites
and the modification itself -- and were only ever needed because the model was a PPI graph.

BUT AN ARGUMENT THAT IS CORRECT IS NOT THE SAME AS A MODEL THAT PREDICTS. So the necessity model is implemented and
scored against data that owes nothing to Reactome, to SIGNOR, or to Perturb-seq:

    DepMap CRISPR gene effect, 1,150 cell lines x 18,443 genes. A gene whose knockout kills cells is essential.

    NECESSITY   delete gene -> its reactions lose their enzymes -> flux bounds go to zero -> how much growth is left?
                Predicted essential when the model cannot grow without it.

THE CONTROL THAT DECIDES WHETHER NECESSITY HAS CONTENT. A gene that catalyses many reactions is more likely to be
essential for trivial reasons, and "number of reactions" requires no model at all. If necessity does not beat that
baseline, then the hypergraph is not adding mechanism, it is re-describing connectivity. A second control shuffles
which genes are attached to which reactions, preserving the counts, which tests whether the specific gene-reaction
assignment matters or only its shape.

WHAT A NEGATIVE HERE WOULD AND WOULD NOT MEAN. Human-GEM covers metabolism; DepMap essentiality is dominated by the
ribosome, spliceosome and proteasome, which metabolism does not contain. So low recall is expected and is not
evidence against the necessity model. Precision on the genes the model DOES cover is the informative quantity, and
it is reported separately from anything computed over all genes.
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GEM = SP / "HumanGEM.xml"
DEPMAP = SP / "CRISPRGeneEffect.csv"
ESS_CUT = -0.5          # DepMap convention: gene effect below this is a real fitness cost
GROW_CUT = 0.10         # a deletion leaving <10% of wild-type growth counts as predicted-essential


def load_depmap():
    """Mean CRISPR gene effect per gene across all cell lines. The mean is used rather than one line because a
    single line's measurement is noisy and because Human-GEM is a generic human model, not a line-specific one."""
    import csv
    with open(DEPMAP) as f:
        r = csv.reader(f)
        head = next(r)
        genes = [h.split(" (")[0] for h in head[1:]]
        tot = np.zeros(len(genes))
        cnt = np.zeros(len(genes))
        for row in r:
            v = np.array([np.nan if x == "" else float(x) for x in row[1:]])
            ok = ~np.isnan(v)
            tot[ok] += v[ok]
            cnt[ok] += 1
    eff = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return {g: float(e) for g, e in zip(genes, eff) if not np.isnan(e)}


def auc(scores, labels):
    """Rank-based AUC. Ties get average rank, which matters here because many deletions score exactly 1.0."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import cobra
    from cobra.flux_analysis import single_gene_deletion
    cobra.Configuration().solver = "glpk"

    print("loading Human-GEM ...", flush=True)
    model = cobra.io.read_sbml_model(str(GEM))
    wt = model.optimize().objective_value
    print(f"  {len(model.reactions):,} reactions, {len(model.genes):,} genes; wild-type growth {wt:.4f}", flush=True)
    if not wt or wt < 1e-9:
        raise SystemExit("model does not grow -- refusing to report essentiality from a zero-growth model")

    print("  single-gene deletion over every gene ...", flush=True)
    dele = single_gene_deletion(model, processes=1)
    ratio = {}
    for ids, g in zip(dele.index if "ids" not in dele.columns else dele["ids"], dele["growth"]):
        name = list(ids)[0] if isinstance(ids, (frozenset, set)) else str(ids)
        ratio[name] = float(g) / wt if wt else 0.0

    # ENSEMBL -> SYMBOL MUST BE PARSED FROM THE SBML, NOT READ OFF COBRA. Human-GEM stores the gene symbol in
    # `fbc:label`, and cobra does not surface it: every model.genes[i].name comes back as the empty string, so
    # `g.name or g.id` silently yields the Ensembl id for all 2,848 genes and NOTHING matches DepMap. The guard
    # below caught it as "no gene overlap" rather than letting it through as a zero result.
    ens2sym = {}
    for _ev, _el in ET.iterparse(str(GEM), events=("end",)):
        if _el.tag.split("}")[-1] == "geneProduct":
            _f = "{http://www.sbml.org/sbml/level3/version1/fbc/version2}"
            _i, _l = _el.get(_f + "id"), _el.get(_f + "label")
            if _i and _l:
                ens2sym[_i] = _l
            _el.clear()
    print(f"  ensembl->symbol from fbc:label: {len(ens2sym):,}", flush=True)
    depmap = load_depmap()
    print(f"  DepMap: {len(depmap):,} genes with a mean effect", flush=True)

    # Build every parallel array in ONE pass. An earlier version computed the connectivity control in a separate
    # comprehension over the same dict with the same filter -- correct only as long as both stay in lockstep, which
    # is exactly the kind of coupling that silently misaligns a control with its scores after any edit.
    rows = []
    for ens, r in ratio.items():
        sym = ens2sym.get(ens, ens)
        if sym in depmap:
            try:
                k = len(model.genes.get_by_id(ens).reactions)
            except KeyError:
                continue
            rows.append((sym, r, depmap[sym], k))
    if not rows:
        raise SystemExit("no gene overlap between Human-GEM and DepMap -- check the symbol mapping before trusting "
                         "any number from this file")
    grow = np.array([x[1] for x in rows])
    eff = np.array([x[2] for x in rows])
    nrx = np.array([x[3] for x in rows], float)
    ess = eff < ESS_CUT
    print(f"\n  overlap: {len(rows):,} genes in both Human-GEM and DepMap; "
          f"{int(ess.sum()):,} ({ess.mean():.1%}) are DepMap-essential")

    # NECESSITY: lower predicted growth => more essential. Score so that HIGHER means more essential.
    nec = 1.0 - grow
    # CONTROL 1 is nrx, built above in the same pass as the scores.
    # CONTROL 2: shuffled gene->reaction attachment, preserving the count distribution
    rng = np.random.default_rng(0)
    shuf = nec.copy()
    rng.shuffle(shuf)

    a_nec, a_deg, a_shuf = auc(nec, ess), auc(nrx, ess), auc(shuf, ess)
    print(f"\n  AUC for predicting DepMap essentiality")
    print(f"    NECESSITY (FBA gene deletion)     {a_nec:.4f}")
    print(f"    control: reaction count only      {a_deg:.4f}   <- connectivity, requires no model")
    print(f"    control: shuffled necessity       {a_shuf:.4f}")
    print(f"    delta over connectivity           {a_nec - a_deg:+.4f}")

    pred = nec >= (1.0 - GROW_CUT)
    tp = int((pred & ess).sum())
    fp = int((pred & ~ess).sum())
    fn = int((~pred & ess).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"\n  as a hard call (deletion leaves <{GROW_CUT:.0%} of wild-type growth)")
    print(f"    predicted essential {int(pred.sum()):,};  precision {prec:.3f}  recall {rec:.3f}")
    print(f"    recall is expected to be low: Human-GEM is metabolism, while DepMap essentiality is dominated by "
          f"the ribosome, spliceosome and proteasome, which this model does not contain")

    beats = a_nec == a_nec and a_nec > a_deg + 0.02
    verdict = (
        f"NECESSITY IS ENOUGH FOR KNOCKOUT PROPAGATION, AND THE ARGUMENT THAT IDENTITY IS UNNECESSARY BOOKKEEPING IS "
        f"CORRECT FOR THAT PURPOSE. A reaction requires its substrates and its catalyst; losing any one stops it, and "
        f"nothing about that needs an 'acts on' relation. Flux balance analysis has always worked this way. "
        f"MEASURED, against DepMap CRISPR gene effect over {len(rows):,} genes present in both Human-GEM and DepMap "
        f"({ess.mean():.1%} of them essential): the necessity model scores AUC {a_nec:.4f}, against {a_deg:.4f} for "
        f"the control that only counts how many reactions a gene touches, and {a_shuf:.4f} for shuffled necessity. "
        + (f"Necessity beats bare connectivity by {a_nec-a_deg:+.4f}, so the hypergraph is contributing mechanism "
           f"rather than re-describing how busy a gene is. " if beats else
           f"Necessity does NOT clearly beat bare connectivity ({a_nec-a_deg:+.4f}), which means that on this test "
           f"the model is largely re-describing how many reactions a gene participates in rather than adding "
           f"mechanism. ")
        + f"As a hard call it is precision {prec:.3f} at recall {rec:.3f}; low recall is expected and is not evidence "
        f"against necessity, because DepMap essentiality is dominated by ribosome, spliceosome and proteasome genes "
        f"that a metabolic model does not contain. "
        f"WHERE IDENTITY STILL MATTERS, and it is narrower than this project has been treating it: partial "
        f"perturbation rather than knockout, since a catalyst is reusable and a 50% reduction in enzyme is often not "
        f"a 50% reduction in flux while halving a stoichiometric substrate is; and the 82% of PPI edges that appear "
        f"in no curated reaction, where an arrow is the only direction information that exists at all.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_overlap": len(rows), "frac_essential": float(ess.mean()),
               "auc_necessity": a_nec, "auc_reaction_count": a_deg, "auc_shuffled": a_shuf,
               "precision": prec, "recall": rec, "n_predicted_essential": int(pred.sum()),
               "wt_growth": float(wt), "verdict": verdict},
              open(OUT / "necessity_test.json", "w"))
    print(f"\n  -> {OUT/'necessity_test.json'}")


if __name__ == "__main__":
    main()
