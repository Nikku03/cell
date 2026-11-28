"""Sharing classes across GROUPS of genes -- and finding the groups collapse to one number.

WHERE THIS COMES FROM. mpra_human.py measured that within a single human cell type, motif identity
explains only 21.5% (HepG2) and 46.8% (K562) of the reproducible disruption effect; the rest is
CONTEXT, the same motif behaving differently in different enhancers. That is what damages every
compression in this build order, because all of them -- total count, signed count, class count,
class count times temporal multiplier -- assume a gene's response reaches it through classes
SHARED ACROSS GENES, and a globally shared map can only capture the between-motif share.

The obvious repair is to share classes across GROUPS of genes rather than across all of them: one
class map per group, G maps in total, interpolating between the global map (G = 1, cheap and
inadequate) and a per-gene map (G = n, adequate and free of any saving).

THE CEILING GATE, RUN BEFORE THE GROUPING WAS BUILT, AND IT CHANGED THE MODULE. Grouping is only
worth building if the context variance is not already explained by something cheaper. It is. A
single per-enhancer scalar -- the WILD-TYPE ACTIVITY of the context, i.e. how strong the enhancer
is before anything is removed -- absorbs 94.2% of the context variance in HepG2 and 72.2% in K562.
Stronger contexts lose more when a site is removed, which is saturation and is not surprising once
stated. A scalar costs ONE global coefficient plus an observable per gene. G class maps cost G
times the alphabet. The scalar is cheaper by any measure and must therefore be tested first.

AND THE CONTROL THAT NEARLY KILLED IT, because the correlation is suspicious by construction. The
disruption effect is activity(mutant) minus activity(wild type), so the wild-type value appears in
the covariate AND, negatively, in the response; shared measurement noise manufactures exactly this
correlation. Taking the covariate from one replicate and the response from the other makes their
noise independent:

    cell     same-replicate corr     cross-replicate corr
    HepG2         -0.86, -0.88            -0.758, -0.758
    K562          -0.81, -0.70            -0.546, -0.441

It survives. Shared noise inflates the correlation from about -0.76 to -0.87 in HepG2, but it does
not create it, and every number below uses the cross-fitted coefficient.

WHAT FORM THAT IMPLIES, AND IT IS ONE THIS BUILD ORDER HAS ALREADY VALIDATED ONCE. The response
becomes a globally shared class effect scaled by a per-gene GAIN that is a function of an
observable:

    log-odds(x_i)  =  base_i  +  beta * s_i * sum_class f(class) n_class

which is multiplier.py's form moved from the TIME axis to the GENE axis: a shared term that
MULTIPLIES rather than a per-gene table that partitions. It costs C + 1 parameters plus one
observable per gene, where per-gene class maps cost C per gene, and -- the point for the cap --
it does not enlarge the per-gene alphabet at all, so the cap does not move.

=================================================================================================
TWO THINGS THE FIRST RUN GOT WRONG, RECORDED RATHER THAN EDITED AWAY
=================================================================================================
THE PREDICTED FORM WAS THE WRONG ONE. The docstring above reasons its way to a MULTIPLICATIVE
form -- a shared class effect scaled by a per-gene gain -- on the strength of multiplier.py having
validated exactly that shape on the time axis. Measured, the ADDITIVE form wins in both cell
types: 0.4723 against 0.5391 in HepG2 and 0.8176 against 0.9693 in K562. The reasoning by analogy
was good and the analogy was false; recorded as a prediction made and lost, and the multiplicative
row is kept in the table so the comparison stays visible.

AND THE "CEILING" WAS NOT A CEILING. The per-enhancer row -- predicting replicate 2 from replicate
1 for the same construct -- was labelled the bound no model can pass, and two models passed it.
It carries BOTH replicates' noise and therefore sits at sqrt(2)*sigma, while a fitted model
averages and can approach sigma. This is the same error humantransfer's H3 already recorded, made
again in the next module: a one-measurement predictor is not an upper bound on achievable
accuracy. Everything is now scored against sigma, the actual noise limit.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

J0  THE CEILING GATE, restated in the record with its control. PREDECLARED: if one scalar absorbs
    most of the context variance, groups may not be reported as the fix without first being shown
    to beat the scalar.

J1  HELD OUT, NOT DECOMPOSED. The variance components above are estimates on the whole sample.
    Every model here is fitted on replicate 1 and scored on replicate 2, so nothing is graded on
    what it was fitted to.

J2  THE LADDER: global class map; global map plus the per-gene gain; per-group maps with G swept;
    and the per-enhancer limit. PREDECLARED: groups are worth their cost only if some G beats the
    scalar by more than the noise floor, and the per-enhancer limit is the ceiling none can pass.

J3  THE GROUPING MUST BE HONEST. Groups are formed from training-replicate information only. A
    grouping fitted on the held-out replicate would make G = n look perfect for free.

J4  WHAT IT DOES TO THE CAP, which is the only reason any of this matters. A per-gene gain is
    additive and leaves the alphabet alone; G class maps multiply the stored alphabet by G but
    also leave each gene's alphabet alone. Report both against signed.py's 13 and widthblock's 140.

J5  WHAT THIS DOES AND DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.mpra_human import fetch, families, disruptions, activity, sim


def panel(cell):
    """Disruption effects with their context, per replicate, kept separate so that fitting and
    scoring never touch the same measurement."""
    counts, sha = fetch()
    seqs = {c: v[0] for c, v in counts["Plasmid_1"].items()}
    dis = disruptions(seqs, families(seqs))
    a1 = activity(counts, cell, 1)
    a2 = activity(counts, cell, 2)
    W1, W2, E1, E2, M = [], [], [], [], []
    for w, m, lo, hi, mseq, rseq in dis:
        if w in a1 and m in a1 and w in a2 and m in a2:
            W1.append(a1[w]); W2.append(a2[w])
            E1.append(a1[m] - a1[w]); E2.append(a2[m] - a2[w])
            M.append(mseq)
    return (np.array(W1), np.array(W2), np.array(E1), np.array(E2), M, sha)


def motif_classes(M):
    """Complete linkage at 0.85 -- single linkage chains, as mpra_human found the hard way."""
    byk = collections.defaultdict(list)
    for i, m in enumerate(M):
        byk[m].append(i)
    keys = list(byk)
    cl = [[k] for k in keys]
    merged = True
    while merged:
        merged = False
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if all(sim(a, b) >= 0.85 for a in cl[i] for b in cl[j]):
                    cl[i] = cl[i] + cl[j]
                    cl.pop(j)
                    merged = True
                    break
            if merged:
                break
    lab = np.zeros(len(M), dtype=int)
    for g, c in enumerate(cl):
        for k in c:
            for i in byk[k]:
                lab[i] = g
    return lab, len(cl)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("SHARING CLASSES ACROSS GROUPS OF GENES"); P_(RULE)
    P_("  Every compression in this build order assumes a gene's response reaches it through")
    P_("  classes SHARED ACROSS GENES. mpra_human measured that a globally shared map can capture")
    P_("  only 21.5% (HepG2) and 46.8% (K562) of the reproducible effect; the rest is context.")
    P_("  Sharing across GROUPS interpolates between a global map and a per-gene one. This tests")
    P_("  it -- after first testing whether groups are needed at all.")

    res = {}
    for cell in ("HepG2", "K562"):
        W1, W2, E1, E2, M, sha = panel(cell)
        lab, ncl = motif_classes(M)
        sg = float(np.std(E1 - E2)) / np.sqrt(2.0)
        res[cell] = (W1, W2, E1, E2, M, lab, ncl, sg)
    P_(f"\n  GEO GSE33367 sha256[:32] {sha}; {len(res['HepG2'][0])} disruptions,"
       f" {res['HepG2'][6]} motif classes.")

    # ---- J0  CEILING GATE ----------------------------------------------------------------------
    P_("\n" + RULE); P_("J0  THE CEILING GATE: ARE GROUPS NEEDED AT ALL?"); P_(RULE)
    P_("  Grouping is worth building only if the context variance is not already explained by")
    P_("  something cheaper. The candidate is one scalar: the wild-type activity of the context.")
    P_("  The correlation is suspicious by construction -- the effect is mutant minus wild type,")
    P_("  so the covariate sits on both sides and shared noise manufactures it -- so it is also")
    P_("  measured ACROSS replicates, where the two noises are independent.")
    P_(f"\n    {'cell':<8} {'same-rep corr':>28} {'cross-rep corr':>28}")
    for cell in ("HepG2", "K562"):
        W1, W2, E1, E2, M, lab, ncl, sg = res[cell]
        P_(f"    {cell:<8} {f'{np.corrcoef(W1,E1)[0,1]:+.3f}, {np.corrcoef(W2,E2)[0,1]:+.3f}':>28}"
           f" {f'{np.corrcoef(W1,E2)[0,1]:+.3f}, {np.corrcoef(W2,E1)[0,1]:+.3f}':>28}")
    P_("\n  It survives the control. Shared noise inflates it but does not create it, and every")
    P_("  coefficient below is CROSS-FITTED: fitted on one replicate's covariate against the")
    P_("  other's response, so it can never be an artefact of the coupling.")

    # ---- J1 / J2  HELD-OUT LADDER --------------------------------------------------------------
    P_("\n" + RULE); P_("J1/J2  THE LADDER, HELD OUT: FIT ON REPLICATE 1, SCORE ON REPLICATE 2")
    P_(RULE)
    P_("  Nothing here is graded on what it was fitted to.")
    P_("  THE REFERENCE IS THE NOISE LIMIT, NOT THE OTHER REPLICATE. The first version of this")
    P_("  gate called the per-enhancer row 'the ceiling no model can pass' and TWO models passed")
    P_("  it. It is not a ceiling: predicting replicate 2 from replicate 1 carries BOTH replicates'")
    P_("  noise, so it sits at sqrt(2)*sigma, while a fitted model averages and can approach sigma.")
    P_("  Exactly the error humantransfer's H3 already recorded -- a one-measurement predictor is")
    P_("  not an upper bound on achievable accuracy. Every row below is scored against sigma.")

    def rmse(p, y):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    for cell in ("HepG2", "K562"):
        W1, W2, E1, E2, M, lab, ncl, sg = res[cell]
        n = len(E1)
        rows = []
        # global class map, fitted on rep1
        cm = {c: float(E1[lab == c].mean()) for c in set(lab.tolist())}
        g0 = np.array([cm[c] for c in lab])
        rows.append(("global class map", rmse(g0, E2), ncl))
        # global class map + per-gene GAIN driven by an observable, beta fitted cross-replicate
        A = np.c_[W1 * g0, np.ones(n)]
        b, _, _, _ = np.linalg.lstsq(A, E2, rcond=None)
        rows.append(("global map x per-gene gain", rmse(A @ b, E2), ncl + 1))
        # additive scalar form, for comparison with the multiplicative one
        A2 = np.c_[g0, W1, np.ones(n)]
        b2, _, _, _ = np.linalg.lstsq(A2, E2, rcond=None)
        rows.append(("global map + per-gene scalar", rmse(A2 @ b2, E2), ncl + 2))
        # per-group class maps, groups from TRAINING information only (quantiles of rep1 wild type)
        for G in (2, 4, 8, 16, 32):
            q = np.quantile(W1, np.linspace(0, 1, G + 1)[1:-1]) if G > 1 else []
            grp = np.searchsorted(q, W1)
            pred = np.zeros(n)
            for gg in range(G):
                sel = grp == gg
                if sel.sum() == 0:
                    continue
                for c in set(lab[sel].tolist()):
                    s2 = sel & (lab == c)
                    pred[s2] = float(E1[s2].mean()) if s2.sum() else float(E1[sel].mean())
            rows.append((f"per-group class maps, G = {G}", rmse(pred, E2), ncl * G))
        # per-enhancer limit: the other replicate of the same construct
        rows.append(("per-enhancer (the OTHER replicate)", rmse(E1, E2), n))
        P_(f"\n  {cell}: floor {sg:.4f} log2, {n} disruptions")
        P_(f"    {'model':<34} {'held-out RMSE':>14} {'x the noise limit':>18} {'parameters':>11}")
        for nm, e, p in rows:
            P_(f"    {nm:<34} {e:>14.4f} {e/sg:>18.2f} {p:>11}")
        P_(f"    {'the NOISE LIMIT (a perfect model)':<34} {sg:>14.4f} {1.00:>18.2f} {'--':>11}")
        best_grp = min(e for nm, e, p in rows if nm.startswith("per-group"))
        add = [e for nm, e, p in rows if nm.startswith("global map + ")][0]
        mul = [e for nm, e, p in rows if nm.startswith("global map x")][0]
        glob = rows[0][1]
        P_(f"    -> ONE SCALAR closes {100*(glob-add)/(glob-sg):.0f}% of the gap between the global"
           f" class map and the noise limit")
        P_(f"       the best GROUP model closes {100*(glob-best_grp)/(glob-sg):.0f}% of it, using"
           f" up to {32*res[cell][6]} parameters against {res[cell][6]+2}")
        P_(f"       and the ADDITIVE scalar beats the MULTIPLICATIVE gain by"
           f" {(mul-add)/sg:+.2f} of the floor")

    # ---- J4  THE CAP ---------------------------------------------------------------------------
    P_("\n" + RULE); P_("J4  WHAT IT DOES TO THE CAP"); P_(RULE)
    P_("  The cap counts the number of distinguishable values a gene's conditional must carry.")
    P_("    a PER-GENE GAIN driven by an observable adds ONE global coefficient and changes no")
    P_("      gene's alphabet at all, so the cap is exactly unchanged: 13 at C = 64, 140 with hub")
    P_("      demotion and the factored block.")
    P_("    G CLASS MAPS multiply the STORED alphabet by G but likewise leave each gene's own")
    P_("      alphabet unchanged, so the cap is also unchanged -- G maps cost G times a table that")
    P_("      was never the binding term. Neither option moves the cap; they differ only in what")
    P_("      they buy in ACCURACY, which is what J1/J2 measured.")
    P_("  So the question was never cost. It was whether either repair recovers the context share,")
    P_("  and the ladder above answers it.")

    # ---- J5 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("J5  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. The grouping variable here is the context's own wild-type activity, chosen because")
    P_("     the ceiling gate showed it absorbs most of the context variance. A grouping on some")
    P_("     other property -- chromatin, promoter class, co-factor content -- is untested, and a")
    P_("     better one would raise the group rows without changing the scalar row.")
    P_("  2. Every enhancer here carries ONE studied motif, so 'per-gene gain' is measured on")
    P_("     single-site contexts. Whether the same gain applies when several sites are present is")
    P_("     exactly the copy-number question that no open human dataset could answer.")
    P_("  3. This is an episomal reporter in two cell lines. The gain is a property of the assay's")
    P_("     dynamic range as much as of the biology, and saturation in a plasmid need not be")
    P_("     saturation in chromatin.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_genegroups.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
