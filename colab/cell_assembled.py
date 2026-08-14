"""THE ASSEMBLED CELL: every layer this repository has, in one object, with its provenance attached.

WHAT THIS IS. Not a new result. An assembly of what loops 33-115 built, so that "how much of a cell
works" stops being a matter of opinion and becomes a matter of counting. Every field carries where it
came from and whether it was validated, because the difference between a cell model and a database
is whether you can say which parts of it have been tested.

THE THREE HONEST DENOMINATORS, and they disagree, which is the point:
    by GENE            16,492 in the model -- the number people quote
    by PROTEOME MASS   what fraction of the actual protein in the cell is covered
    by REACTION        12,931 Human-GEM reactions with stoichiometry

A layer covering 24% of genes can cover 93% of mass, because a few abundant proteins are most of the
cell. Quoting only the flattering denominator is how model coverage gets overstated, so all three are
returned for every layer.

STATUS VOCABULARY, applied per layer and defined once:
    RUNS       advances a state or solves a system, and has been validated against held-out data
    CLOSES     a physical budget that balances, with no fitted parameter
    STATIC     real data, correctly loaded, but nothing computes from it
    FAILED     built and tested and did not survive its own gate
    ABSENT     not present at all
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_replication as LR  # noqa: E402

LN2 = float(np.log(2.0))
MU = LN2 / 24.0
LIFE = Path(LR.CELL).parent / "cell_lifetimes.json"


def load():
    """Every layer, with its provenance. Nothing here computes; it assembles."""
    C = json.load(open(LR.CELL))
    S = json.load(open(LR.SC / "_schwan2011.json"))
    life = json.load(open(LIFE))["lifetimes"] if LIFE.exists() else {}
    names = [g["name"] for g in C["genes"]]
    return {"model": C, "names": names, "idx": {n: i for i, n in enumerate(names)},
            "schwan": S, "lifetimes": life,
            "pubs": {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}}


def state_vector(D):
    """The cell's dynamical state: mRNA and protein copy number per gene, where both are knowable.

    This is the object a whole-cell model must have and this repository did not have until loop 112.
    It is small: a state needs a copy number AND both half-lives, and only Schwanhausser supplies all
    three from one cell type -- the constraint loop 92 established when mixing sources produced a
    translation rate twelve times the published value.
    """
    S = D["schwan"]
    ok = [g for g, v in S.items()
          if v.get("mrna_copies") and v.get("prot_copies")
          and v.get("mrna_hl_h") and v.get("prot_hl_h")]
    M = np.array([S[g]["mrna_copies"] for g in ok])
    P = np.array([S[g]["prot_copies"] for g in ok])
    a = np.array([LN2 / S[g]["mrna_hl_h"] + MU for g in ok])
    b = np.array([LN2 / S[g]["prot_hl_h"] + MU for g in ok])
    return {"genes": ok, "M": M, "P": P, "k_loss_mrna": a, "k_loss_prot": b,
            "k_sm": M * a, "k_sp": P * b / M}


def integrate(st, hours=2000.0, dt=0.25):
    """Forward-march dM/dt = k_sm - a*M and dP/dt = k_sp*M - b*P from zero, exponential stepper."""
    n = len(st["genes"])
    M = np.zeros(n)
    P = np.zeros(n)
    a, b, ks, kp = st["k_loss_mrna"], st["k_loss_prot"], st["k_sm"], st["k_sp"]
    ea, eb = np.exp(-a * dt), np.exp(-b * dt)
    for _ in range(int(hours / dt)):
        Mn = M * ea + (ks / a) * (1 - ea)
        P = P * eb + (kp * M / b) * (1 - eb)
        M = Mn
    return M, P


def mass_fraction(D, genes):
    """What share of measured protein MASS a gene subset carries -- copies x residues."""
    S = D["schwan"]
    import gzip
    L, nm, c = {}, None, 0
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if nm and c:
                    L[nm] = max(L.get(nm, 0), c)
                c, nm = 0, None
                for p in ln.split():
                    if p.startswith("GN="):
                        nm = p[3:]
                        break
            else:
                c += len(ln.strip())
    if nm and c:
        L[nm] = max(L.get(nm, 0), c)
    tot = sum(S[g]["prot_copies"] * L[g] for g in S
              if S[g].get("prot_copies") and g in L)
    sub = sum(S[g]["prot_copies"] * L[g] for g in genes
              if g in S and S[g].get("prot_copies") and g in L)
    return sub / tot if tot else float("nan")


# THE AUDIT TABLE. Every entry's numbers come from a committed outputs/*.json; nothing is asserted.
LAYERS = [
    ("chromatin folding", "RUNS", "loop_second/85/86",
     "1 s timestep, 25 kb, P(s) -1.05 genome-wide; orientation +0.3965 vs measured +0.3788",
     "map rho 0.8555 vs distance-null 0.8283 -- 19% of headroom; 1 of 23 chromosomes"),
    ("metabolic network", "RUNS", "cell_sim/93",
     "12,931 reactions, 96.2% one component, traceable 8 steps to biomass, DepMap-scored",
     "steady state only; growth 712x too fast from the medium's uptake bounds"),
    ("ribosome budget", "CLOSES", "loop_translation",
     "demand 30.76 Gcodons/h vs capacity 137.75 Gcodons/h, 22.3% utilisation",
     "4,179 genes"),
    ("proteasome budget", "CLOSES", "loop_proteostasis",
     "demand inside capacity at the worst swept corner; 1.38M particles",
     "4,821 genes"),
    ("doubling-time budget", "CLOSES", "loop_doubling",
     "13.28 h predicted from protein budget alone, nothing fitted; measured 24 h",
     "bootstrap [11.24, 15.57] h; says translation does NOT bind"),
    ("transcription->mRNA->protein", "RUNS", "loop_integrator",
     "mRNA predicted from independent burst kinetics rho +0.4649; protein +0.5593",
     "the ODE itself adds nothing: raw input scores +0.4960, integrated +0.4649"),
    ("protein->flux->growth", "RUNS", "loop_growth_loop",
     "iterated fixed point, stable to start, growth within 3x of measured",
     "does not converge"),
    ("signalling stoichiometry", "STATIC", "loop_signalling_cost",
     "13,195 balanced reactions; ATP budget puts a 5-minute floor under phosphosite lifetime",
     "no kinetics; phospho-forms share a node so the chain is chemically inert"),
    ("complex stoichiometry", "STATIC", "loop_complex_stoich",
     "obligate complexes 2.2x tighter than chance on real abundance",
     "the average annotated complex is not a machine; fame clusters more than abundance"),
    ("chromatin->transcription", "FAILED", "loop_chromatin_to_rate",
     "forward emission fails held-out prediction, fails fame, fails its own null",
     "loop 95's +0.037 correlation does not survive being made generative"),
    ("regulation->transcription", "FAILED", "loop_tf_rate/loop_perturb",
     "network loses to fame 2:1; perturbation no better than shuffled edge signs",
     "612,133 edges that encode which genes were studied together"),
    ("reaction+graph fusion", "FAILED", "loop_fusion_linear",
     "combined 0.166 BELOW graph-only; 90% of what remains survives degree-preserving rewiring",
     "the metabolic bridge is absent for 84% of genes"),
    ("cell cycle", "FAILED", "loop_cellcycle",
     "canonical phase order recovered but 176/500 shuffles do it too",
     "no phase-resolved layer exists"),
    ("replication as process", "FAILED", "loop_replication_time",
     "a Gaussian blur of the origin map (R2 0.320) beats the fork simulation (0.153)",
     "fork speed is unconstrained by the data it was fitted to"),
    ("expression noise", "FAILED", "loop_noise",
     "predicted CV tracks measured at +0.4837 but is cross-dataset abundance agreement to 0.8%",
     "against the non-abundance component it points the wrong way"),
    # CORRECTED. I recorded transport as ABSENT and it is not. Human-GEM is compartment-resolved
    # and 36.7% of its reactions move a metabolite across a membrane. Measured, not assumed.
    ("compartments (metabolic)", "RUNS", "cell_sim",
     "9 compartments: cytosol 3,339 metabolites, extracellular 1,660, ER 971, mitochondrion 939, "
     "peroxisome 507, lysosome 452, Golgi 361, nucleus 212, inner-mito 20",
     "compartments of METABOLITES; proteins are not compartment-resolved in the same object"),
    ("transport (metabolic)", "RUNS", "cell_sim",
     "4,742 reactions span >1 compartment (36.7% of the model), 2,427 with gene rules; "
     "714 carry flux at the optimum = 31.0% of all active reactions; "
     "c<->e 436, c<->r 101, c<->m 87, c<->x 21, c<->l 18, c<->n 16",
     "stoichiometric transport with no rate law, no membrane, no concentration gradient"),
    ("compartments (gene labels)", "STATIC", "loop_compartment_mass",
     "12 buckets: nucleus 4,007, cytoplasm 3,581, plasma membrane 2,782, membrane 1,462, "
     "extracellular 1,455, mitochondrion 952, ER 787, cytoskeleton 619, Golgi 435, "
     "endosome 193, lysosome 157, peroxisome 62",
     "AGREES WITH THE GEM COMPARTMENT OF ITS OWN REACTION ONLY 40.3% OF THE TIME "
     "(9,356 of 23,225 gene-reaction pairs) -- two compartment systems that do not line up"),
    ("protein structure", "STATIC", "struct/fold + the nexus arm",
     "13 genes in `struct` with residue counts and variant tallies; 1 gene in `fold` (SOD1 A4V)",
     "the nexus/dock arm repeatedly failed its own baselines; there is no structure for the "
     "other 16,479 genes and nothing computes from the 13 that have one"),
    ("spatial organisation", "ABSENT", "-",
     "no volumes, no membranes, no coordinates -- compartments are set membership, not geometry",
     "-"),
    ("diffusion", "ABSENT", "-",
     "transport reactions move material between compartment POOLS with no distance or gradient",
     "-"),
    ("cell division", "ABSENT", "-", "no cycle advances; the cell accumulates forever", "-"),
    ("enzyme kinetics", "ABSENT", "-",
     "no measured k_cat anywhere; loop 93 inverted it from flux and got median 0.1/s", "-"),
]
