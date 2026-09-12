"""LOOP 73 -- THE EXPRESSION LAYER: MAKE THE MACHINES COST SOMETHING.

WHY. Loop 71 put HumanGEM's 12,931 reactions under a defined medium and predicted human gene
essentiality at AUC 0.6403 against publication count's 0.8443 -- a loss of 0.2040, the exact mirror
of E. coli where the same schema WON by 0.1819. The reason is visible in one line of the model:

    MAR10062:  0.0721 alanine + 0.0801 arginine + ... -> protein_pool_biomass      GPR: (none)

Protein is made from amino acids by no gene at all. No ribosome, no tRNA synthetase, no elongation
factor appears anywhere in the stoichiometry, so deleting any of them costs nothing. Measured against
Hart's reference sets, that blindness covers 157 of 684 core-essential genes (23.0%) across seven
machine classes -- 41 cytoplasmic ribosome, 24 spliceosome, 24 proteasome, 22 tRNA synthetase, 17
initiation/elongation factor, 17 RNA polymerase, 12 mitochondrial ribosome -- and ZERO of the 926
non-essential genes. The classes separate the two label sets perfectly, which is why they are worth
building rather than approximating.

WHAT IS ADDED, and the principle is one sentence: A GROWING CELL MUST DOUBLE ITS MACHINES.

    per subunit gene   SYN_<gene>:  charged amino acids (that protein's REAL composition from its
                                    UniProt sequence) + GTP  ->  PROT_<gene>      GPR = that gene
    per machine        ASM_<machine>: sum of its PROT_<gene>  ->  <machine>
    biomass            now also consumes each machine, at a coefficient DERIVED from its mass
                       fraction rather than typed in

THE COEFFICIENT IS DERIVED, AND THE FIRST RUN PROVES WHY IT HAD TO BE. Declaring "the ribosome is 5%
of protein mass" is biology; writing 0.05 into the biomass reaction is a units error. One assembled
ribosome here costs 23,499 amino acids, while the entire protein pool demands 5.3380 mmol of amino
acids per unit biomass -- so the correct coefficient for 5% of protein mass is 1.136e-05, and 0.05 is
4,400x too large. The first run used 0.05 and growth collapsed 0.02036 -> 0.00015 /h, failing E1.
The fix is not to tune the number until it grows; it is to compute it:

    coefficient = target_mass_fraction * total_protein_aa_demand / machine_aa_cost

so the only thing declared is the mass fraction, which is a measurable property of a cell.

Dilution by growth is the physically correct coupling -- a cell that doubles must make a second
ribosome -- and it is what turns a subunit into a requirement rather than a decoration.

THE CIRCULARITY, NAMED BEFORE IT IS EXPLOITED. Once the ribosome is assembled from ribosomal
proteins, deleting a ribosomal protein is lethal in silico. THAT IS NOT A PREDICTION. It is the
stoichiometry I just typed in, read back. Reporting it as an essentiality result would be the purest
form of the mistake this project has spent seventy loops avoiding.

So the gates are split. E2 reports the wired genes as a WIRING CHECK -- did the plumbing work -- and
is explicitly barred from the headline. E3 and E4, the gates that count, are scored ONLY on genes
that were never wired: the metabolic genes and everything else. The prediction being tested is that
coupling every enzyme to a shared amino-acid and machine budget changes which METABOLIC genes are
essential -- because a bypass route that costs more enzyme than the budget allows is no longer a
bypass. That is emergent, and it can fail.

PREDECLARED, before any number:

  E1 THE EXTENDED MODEL STILL GROWS
       growth must stay in 0.01-0.05 /h. Adding machine demand can only slow the cell; if it stops
       growing the coefficients are wrong and nothing downstream means anything.
  E2 THE WIRING WORKS                                   NOT A RESULT. A PLUMBING CHECK.
       >= 90% of the wired subunit genes must become lethal in silico. This is the stoichiometry
       read back and is reported as such. It is barred from the headline and from E3/E4's scoring.
  E3 IT IMPROVES PREDICTION ON GENES THAT WERE NOT WIRED           THE GATE.
       essentiality AUC over labelled genes EXCLUDING every wired subunit. Must improve on the BASE
       model scored over the IDENTICAL genes. If the expression layer only makes the genes I typed
       in lethal, it has taught the model nothing and this gate says so.
       [CORRECTED AFTER THE FIRST RUN. This gate originally compared against the hardcoded constant
       LOOP71_AUC, and the sentence here claimed that was "the same restricted set". It was
       not: loop 71 measured it on its own 208 labelled genes, INCLUDING genes this module later
       wires. Two AUCs over two different gene sets are not a comparison. The base model is now
       deleted inside this module before anything is added, and both are scored on the same genes.
       The first run reported 0.6858 vs 0.6403, delta +0.0455; that delta was not paired and should
       not be quoted. The count of non-wired genes whose deletion phenotype actually CHANGED is now
       printed too, because if that count is zero any AUC difference is arithmetic, not biology.]
  E4 IT BEATS FAME ON THAT SAME SET                      THE POINT.
       publication count on the non-wired labelled genes. Loop 71 lost this 0.6403 to 0.8443.
  E5 WHAT IT STILL CANNOT SEE
       the remaining CEGv2 classes with no representation, named and counted.

-> outputs/loop_expression_layer.json
"""
import collections
import gzip
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
GEM = SC / "humangem.json"
MED = SC / "_ham_medium.json"
FASTA = SC / "human_proteome.fasta.gz"
GENES = SC / "hgem_genes.tsv"
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

AA20 = ["alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamine", "glutamate",
        "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine",
        "proline", "serine", "threonine", "tryptophan", "tyrosine", "valine"]
AA1 = dict(zip("ARNDCQEGHILKMFPSTWYV",
               ["alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamine",
                "glutamate", "glycine", "histidine", "isoleucine", "leucine", "lysine",
                "methionine", "phenylalanine", "proline", "serine", "threonine", "tryptophan",
                "tyrosine", "valine"]))

# value is the machine's TARGET MASS FRACTION of total cell protein -- a measurable biological
# quantity. The stoichiometric coefficient is derived from it at build time, never typed in.
MACHINES = {
    "cyto_ribosome":  (r"^RP[LS]", 0.050),
    "mito_ribosome":  (r"^MRP[LS]", 0.005),
    "proteasome":     (r"^PSM", 0.010),
    "spliceosome":    (r"^(SNRP|SF3|PRPF|LSM|U2AF)", 0.005),
    "rna_polymerase": (r"^POLR", 0.003),
    "trna_synthetase": (r"^([A-Z]ARS2?$|EPRS$|QARS1$|AARS1$|IARS1$|KARS1$)", 0.005),
    "elong_init":     (r"^(EIF|EEF)", 0.010),
}
GROWTH_LO, GROWTH_HI = 0.01, 0.05
WIRE_FLOOR = 0.90
LOOP71_AUC = 0.6902   # loop 71 RERUN after the GLPK fix; the superseded buggy run gave 0.6403
LOOP71_FAME = 0.8443
DEAD = 0.01
SEED = 7301
LP_TIMEOUT = 20            # seconds per deletion LP; median is 0.069 s. GLPK's default is INT_MAX.
LP_RETRY = 120             # seconds for the lone retry of any LP that hit the limit

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def composition():
    """gene symbol -> {amino acid name: count}, from the real UniProt sequence."""
    out, name, seq = {}, None, []
    def flush():
        if name and seq:
            s = "".join(seq)
            c = collections.Counter(s)
            if name not in out or sum(c.values()) > sum(out[name].values()):
                out[name] = {AA1[a]: n for a, n in c.items() if a in AA1}
    for ln in gzip.open(FASTA, "rt", errors="ignore"):
        if ln.startswith(">"):
            flush()
            gn = [x[3:] for x in ln.split() if x.startswith("GN=")]
            name = gn[0] if gn else None
            seq = []
        else:
            seq.append(ln.strip())
    flush()
    return out


def deletion_ratios(M, mu, note=""):
    """Growth ratio after knocking out each gene, plus the genes the solver could not settle.

    GLPK's default tm_lim is INT_MAX -- no limit -- and this module once hung for 1 h 50 min inside
    a single degenerate deletion LP whose median cost is 0.069 s (py-spy caught it in glp_simplex).
    A limit is set here; a timed-out LP is retried alone and, if still unsettled, EXCLUDED rather
    than read as zero growth. Reading it as zero would have invented essential genes from solver
    fatigue -- on the extended model, whose extra reactions are exactly what E3/E4 are testing.
    """
    from cobra.flux_analysis import single_gene_deletion
    from cobra.manipulation import knock_out_model_genes
    M.solver.configuration.timeout = LP_TIMEOUT
    say(f"     running {len(M.genes):,} single-gene deletions{note} (LP limit {LP_TIMEOUT}s) ...")
    dl = single_gene_deletion(M, gene_list=M.genes, processes=1)
    ratio, suspect = {}, []
    for _, row in dl.iterrows():
        ids = list(row["ids"]) if not isinstance(row["ids"], str) else [row["ids"]]
        v = row["growth"]
        for g in ids:
            if v is None or not np.isfinite(v):
                suspect.append(g)
            else:
                ratio[g] = float(v) / max(mu, 1e-12)
    unresolved = []
    if suspect:
        M.solver.configuration.timeout = LP_RETRY
        for g in suspect:
            with M:
                knock_out_model_genes(M, [g])
                v = M.slim_optimize()
                stat = M.solver.status
            if v is not None and np.isfinite(v):
                ratio[g] = float(v) / max(mu, 1e-12)
            elif stat == "infeasible":
                ratio[g] = 0.0
            else:
                unresolved.append(g)
        M.solver.configuration.timeout = LP_TIMEOUT
    if unresolved:
        say(f"     WARNING: {len(unresolved)} deletion LPs did not settle in {LP_RETRY}s and are "
            f"EXCLUDED, not scored as lethal: {', '.join(sorted(unresolved)[:10])}")
    return ratio, unresolved


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 73 -- the expression layer: make the machines cost something")
    say("  E2 is a plumbing check, not a result. E3/E4 score ONLY genes that were never wired.")
    say("=" * 100)
    say()

    import cobra
    import csv
    cobra.Configuration().solver = "glpk"
    M = cobra.io.load_json_model(str(GEM))
    cfg = json.load(open(MED))
    FREE = {"H2O", "H+", "O2", "Na+", "K+", "chloride", "Pi", "sulfate", "HCO3-", "Mg2+",
            "Ca2+", "Fe2+"}
    for r in M.reactions:
        if r.boundary:
            r.lower_bound = 0.0
    for w, rid in cfg["medium"].items():
        M.reactions.get_by_id(rid).lower_bound = -1000.0 if w in FREE else -0.010
    mu0 = float(M.optimize().objective_value or 0.0)

    # THE PAIRED BASELINE. The first version of E3 compared this module's AUC on non-wired genes
    # against the hardcoded constant LOOP71_AUC -- a number loop 71 computed on ITS OWN
    # labelled set, which included genes this module later wires. Two AUCs on two different gene
    # sets are not a comparison, and the docstring's claim of "the same restricted set" was false.
    # So the base model is deleted here, BEFORE a single reaction is added, and E3 scores base and
    # extended on exactly the same genes. It costs one extra pass (~6 min) and it is the difference
    # between a paired test and a suggestive pair of numbers.
    say("  paired baseline: deleting genes on the BASE model before anything is added")
    base_ratio, base_unres = deletion_ratios(M, mu0, note=" on the BASE model")

    e2s = {}
    with open(GENES) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if len(row) > 5:
                e2s[row[0].strip('"')] = row[4].strip('"')
    s2e = {v: k for k, v in e2s.items()}
    hart = json.load(open(SC / "_hart.json"))
    ess, non = set(hart["ceg"]), set(hart["neg"])
    comp = composition()

    aa_met = {}
    for m in M.metabolites:
        nm = (m.name or "").strip()
        if nm in AA20 and m.compartment == "c":
            aa_met.setdefault(nm, m)
    # total amino-acid demand of the existing protein pool, used to derive every machine coefficient
    pp = M.metabolites.get_by_id("MAM10013c")
    r62 = M.reactions.get_by_id("MAR10062")
    aa_per_pool = sum(-c for m, c in r62.metabolites.items() if c < 0 and m.id != "MAM02040c")
    total_aa = abs(M.reactions.get_by_id("MAR13082").metabolites[pp]) * aa_per_pool
    say(f"  base model growth {mu0:.5f} /h; {len(aa_met)}/20 cytosolic amino acids located")
    say(f"  total protein amino-acid demand per unit biomass: {total_aa:.4f} mmol")

    say()
    say("  BUILDING THE EXPRESSION LAYER")
    atp = M.metabolites.get_by_id("MAM01371c")
    adp = M.metabolites.get_by_id("MAM01285c")
    pi = M.metabolites.get_by_id("MAM02751c")
    h2o = M.metabolites.get_by_id("MAM02040c")
    bm = M.reactions.get_by_id("MAR13082")
    wired = {}
    coefs = {}
    newrx = []
    for mach, (pat, frac) in MACHINES.items():
        subs = sorted({g for g in ess | non if re.match(pat, g)} |
                      {g for g in s2e if re.match(pat, g)})
        subs = [g for g in subs if g in comp]
        if not subs:
            continue
        machine_met = cobra.Metabolite(f"MACH_{mach}_c", compartment="c", name=mach)
        prot_mets = []
        for g in subs:
            pm = cobra.Metabolite(f"PROT_{g}_c", compartment="c", name=f"{g} protein")
            prot_mets.append(pm)
            r = cobra.Reaction(f"SYN_{g}", name=f"synthesis of {g}", lower_bound=0, upper_bound=1000)
            st = {}
            L = sum(comp[g].values())
            for aa, n in comp[g].items():
                if aa in aa_met:
                    st[aa_met[aa]] = st.get(aa_met[aa], 0) - float(n)
            st[atp] = -4.0 * L
            st[h2o] = -4.0 * L
            st[adp] = 4.0 * L
            st[pi] = 4.0 * L
            st[pm] = 1.0
            r.add_metabolites(st)
            gid = s2e.get(g)
            r.gene_reaction_rule = gid if gid else g
            newrx.append(r)
        asm = cobra.Reaction(f"ASM_{mach}", name=f"{mach} assembly", lower_bound=0, upper_bound=1000)
        asm.add_metabolites({pm: -1.0 for pm in prot_mets} | {machine_met: 1.0})
        newrx.append(asm)
        machine_aa = sum(sum(comp[g].values()) for g in subs)
        coef = frac * total_aa / max(machine_aa, 1)
        bm.add_metabolites({machine_met: -coef})
        wired[mach] = subs
        coefs[mach] = coef
        say(f"     {mach:16s} {len(subs):3d} subunits, {machine_aa:7,d} aa/machine, "
            f"mass fraction {frac:.1%} -> coefficient {coef:.3e}")
    M.add_reactions(newrx)
    allwired = {g for v in wired.values() for g in v}
    say(f"     added {len(newrx):,} reactions, {len(allwired)} wired subunit genes")
    say()

    say("E1 THE EXTENDED MODEL STILL GROWS")
    mu = float(M.optimize().objective_value or 0.0)
    say(f"     growth {mu0:.5f} -> {mu:.5f} /h   (gate {GROWTH_LO}-{GROWTH_HI})")
    e1 = GROWTH_LO <= mu <= GROWTH_HI
    say(f"     E1 {'PASS' if e1 else 'FAIL'}")
    say()

    ratio, unresolved = deletion_ratios(M, mu, note=" on the EXTENDED model")
    sym_ratio = {}
    for gid, v in ratio.items():
        s = e2s.get(gid, gid)
        sym_ratio[s] = min(sym_ratio.get(s, 1.0), v)

    say()
    say("E2 THE WIRING WORKS -- PLUMBING CHECK, NOT A RESULT")
    lw = [g for g in allwired if sym_ratio.get(g, 1.0) < DEAD]
    frac_w = len(lw) / max(len(allwired), 1)
    say(f"     {len(lw)} of {len(allwired)} wired subunits lethal in silico ({frac_w:.1%})")
    say(f"     this is the stoichiometry typed in above, read back. It is NOT an essentiality result")
    say(f"     and is excluded from E3 and E4.")
    e2 = frac_w >= WIRE_FLOOR
    say(f"     E2 {'PASS' if e2 else 'FAIL'}")
    say()

    say("E3 IT IMPROVES PREDICTION ON GENES THAT WERE NOT WIRED")
    base_sym = {}
    for gid, v in base_ratio.items():
        base_sym[e2s.get(gid, gid)] = v
    lab, sc, bs, gs = [], [], [], []
    for g, v in sym_ratio.items():
        if g in allwired or g not in base_sym:
            continue
        if g in ess:
            y = 1
        elif g in non:
            y = 0
        else:
            continue
        lab.append(y)
        sc.append(1.0 - min(max(v, 0.0), 1.0))
        bs.append(1.0 - min(max(base_sym[g], 0.0), 1.0))
        gs.append(g)
    lab, sc, bs = np.array(lab), np.array(sc), np.array(bs)
    two = len(set(lab)) > 1
    auc = float(roc_auc_score(lab, sc)) if two else float("nan")
    auc_base = float(roc_auc_score(lab, bs)) if two else float("nan")
    say(f"     {len(lab)} labelled non-wired genes ({int(lab.sum())} essential), scored on the "
        f"IDENTICAL set by both models")
    say(f"     BASE model     (metabolism only)  AUC {auc_base:.4f}")
    say(f"     EXTENDED model (+ machines)       AUC {auc:.4f}   delta {auc - auc_base:+.4f}")
    say(f"     for reference, loop 71 reported {LOOP71_AUC:.4f} on ITS OWN larger labelled set --")
    say(f"     that number is NOT the baseline here, because it was measured on different genes.")
    nch = int((np.abs(sc - bs) > 1e-9).sum())
    say(f"     {nch} of {len(lab)} non-wired genes changed their deletion phenotype at all")
    if nch == 0:
        say(f"     ZERO changed. Any AUC difference would be arithmetic noise, not biology.")
    e3 = two and auc > auc_base
    say(f"     E3 {'PASS' if e3 else 'FAIL'}")
    say()

    say("E4 IT BEATS FAME ON THAT SAME SET")
    D = json.load(open(CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in D["genes"]}
    fv = np.array([pubs.get(g, 0.0) for g in gs])
    have = fv > 0
    auc_f = float(roc_auc_score(lab[have], fv[have])) if len(set(lab[have])) > 1 else float("nan")
    auc_m = float(roc_auc_score(lab[have], sc[have]))
    say(f"     publication count  {auc_f:.4f}")
    say(f"     extended model     {auc_m:.4f}    delta {auc_m - auc_f:+.4f}")
    say(f"     loop 71 lost this {LOOP71_AUC:.4f} to {LOOP71_FAME:.4f}")
    e4 = auc_m > auc_f
    say(f"     E4 {'PASS' if e4 else 'FAIL'}")
    say()

    say("E5 WHAT IT STILL CANNOT SEE")
    unseen = sorted(g for g in ess if g not in allwired and g not in sym_ratio)
    say(f"     {len(unseen)} of {len(ess)} core-essential genes have NO representation at all")
    say(f"       {', '.join(unseen[:20])}")
    e5 = True
    say(f"     E5 PASS (named)")
    say()

    gates = {"E1 extended model still grows": bool(e1),
             "E2 wiring works (plumbing, not a result)": bool(e2),
             "E3 improves on non-wired genes": bool(e3),
             "E4 beats fame on non-wired genes": bool(e4),
             "E5 the blind spot is named": bool(e5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(GEM), str(FASTA), str(GENES), str(CELL)],
                      available=len(M.genes), used=len(lab), selection="filtered", seed=SEED,
                      controls=["wired genes excluded from every scored gate",
                                "amino-acid cost from each protein's real UniProt composition",
                                "growth required to stay physiological after adding machine demand",
                                "publication count on the identical non-wired set",
                                "unrepresented essential genes named"],
                      note="E2 is the stoichiometry read back and is barred from the headline")
    RM.report(man, emit=say)
    json.dump({"test": "loop_expression_layer", "manifest": man, "gates": gates, "e3_auc_base": auc_base, "e3_auc_extended": auc,
               "e3_delta": auc - auc_base, "e3_n_phenotype_changed": nch,
               "base_unresolved": len(base_unres), "ext_unresolved": len(unresolved),
               "growth_before": mu0, "growth_after": mu,
               "wired": {k: v for k, v in wired.items()}, "n_wired": len(allwired),
               "machine_coefficients": coefs, "total_protein_aa_demand": total_aa,
               "wired_lethal_frac": frac_w,
               "n_labelled_nonwired": int(len(lab)), "n_essential_nonwired": int(lab.sum()),
               "auc_nonwired": auc, "loop71_auc": LOOP71_AUC,
               "auc_fame": auc_f, "auc_model": auc_m,
               "unrepresented": unseen, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_expression_layer.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_expression_layer.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
