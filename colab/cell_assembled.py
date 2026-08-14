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


def tf_wiring(D, signed_only=True):
    """The regulatory network as a RATE LAW needs it: signed edges, grouped by the gene they act on.

    THE 91% THAT CANNOT DRIVE ANYTHING. C["reg"] carries 612,133 edges and 558,005 of them have
    sign 0 -- a binding event with no recorded direction of effect. An unsigned edge cannot enter a
    rate law, because "TF j regulates gene i" does not say whether more j means more i or less.
    Every previous use of this network scored it as a graph, where an unsigned edge still counts as
    adjacency; the moment it has to drive k_sm, 91.2% of it becomes unusable. That number is stated
    here rather than hidden behind a gene count, because it is the honest size of the wiring.

    Returns {target gene name: [(regulator gene name, sign), ...]} over the 54,128 signed edges,
    1,451 distinct regulators and 7,485 distinct targets.
    """
    names = D["names"]
    by = {}
    for s, t, g in D["model"]["reg"]:
        if signed_only and int(g) == 0:
            continue
        by.setdefault(names[t], []).append((names[s], int(g)))
    return by


def tf_index(wiring, targets, regulators):
    """Flatten the wiring onto integer indices so the drive is one segment-sum, not a Python loop."""
    ti = {g: i for i, g in enumerate(targets)}
    ri = {g: i for i, g in enumerate(regulators)}
    rows, cols, sgn = [], [], []
    for g in targets:
        for r, s in wiring.get(g, ()):
            if r in ri:
                rows.append(ti[g])
                cols.append(ri[r])
                sgn.append(float(s))
    n = np.zeros(len(targets))
    for i in rows:
        n[i] += 1.0
    return (np.array(rows, int), np.array(cols, int), np.array(sgn),
            np.maximum(n, 1.0), n.astype(int))


def tf_drive(ix, dev):
    """Net signed drive per target: sum_j s_ij * dev_j / n_i. Bounded in [-1, 1] whenever dev is.

    Dividing by the regulator count is not cosmetic -- it is what makes the drive a fraction of
    k_sm rather than an unbounded sum, so a gene with 774 regulators cannot be driven 774x harder
    than a gene with one. Signs CANCEL inside the sum, which is the only place in this wiring where
    the network does work a bare edge count could not.
    """
    rows, cols, sgn, nsafe, _ = ix
    out = np.zeros(len(nsafe))
    np.add.at(out, rows, sgn * dev[cols])
    return out / nsafe


def integrate_tf(kbar, a, ix, dev_at, T, ncyc=12, nstep=400, gain=1.0):
    """dM/dt = k_sm(t) - a*M with k_sm(t) = kbar * max(0, 1 + gain*drive(t)).

    With dev == 0 the drive is zero and k_sm is exactly kbar, so the wired integrator reduces to
    the unwired one identically -- the wiring is an addition to the equation, not a replacement of
    it, and loop 120's W4 gates on that reduction holding to machine precision.
    """
    dt = T / nstep
    ea = np.exp(-a * dt)
    M = kbar / a
    total = ncyc * nstep
    tr = np.zeros((nstep, len(M)))
    for s in range(total):
        k_t = kbar * np.maximum(0.0, 1.0 + gain * tf_drive(ix, dev_at(s * dt)))
        M = M * ea + (k_t / a) * (1 - ea)
        if s >= total - nstep:
            tr[s - (total - nstep)] = M
    mean = tr.mean(0)
    rel = (tr.max(0) - tr.min(0)) / (2.0 * np.maximum(mean, 1e-300))
    return rel, mean


def integrate_deg(ksp, Mbar, bbar, T, beta=1.0, ncyc=12, nstep=400):
    """dP/dt = k_sp*Mbar - b(t)*P with b(t) = bbar*(1 + beta*sin(wt)) and the mRNA held FLAT.

    THE OTHER TERM. loop 119 showed that with b constant, protein oscillation is forced to be
    smaller than mRNA oscillation and therefore rarer -- while the measurement has 362 proteins
    oscillating whose mRNA does not. This integrator is the alternative: leave M at its steady
    value and put the cycle in the loss rate instead. beta = 1 is a loss rate swinging between 0
    and 2*bbar over the cycle, which is the size of swing an APC/C or SCF substrate experiences.

    beta = 0 must reproduce k_sp*Mbar/bbar exactly, and loop 121's S6 gates on that.
    """
    dt = T / nstep
    w = 2.0 * np.pi / T
    P = ksp * Mbar / bbar
    total = ncyc * nstep
    tr = np.zeros((nstep, len(P)))
    for s in range(total):
        b_t = np.maximum(bbar * (1.0 + beta * np.sin(w * (s * dt))), 1e-12)
        eb = np.exp(-b_t * dt)
        P = P * eb + (ksp * Mbar / b_t) * (1 - eb)
        if s >= total - nstep:
            tr[s - (total - nstep)] = P
    mean = tr.mean(0)
    rel = (tr.max(0) - tr.min(0)) / (2.0 * np.maximum(mean, 1e-300))
    return rel, mean


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
    # ADDED by loop 119, and it is a falsification of the layer directly above rather than a new one.
    ("protein dynamics from transcription alone", "FAILED", "loop_cellcycle_axis/119",
     "dP/dt = k_sp*M - b*P is a first-order filter, so protein swing is FORCED below mRNA swing -- "
     "measured at 100.0000% of 4,190 genes, median attenuation 0.16, 81.9% damped below a tenth of "
     "the drive. Protein oscillation can only ever be RARER than transcript oscillation here",
     "THE MEASUREMENT POINTS THE OTHER WAY: 362 proteins oscillate whose transcript does not, "
     "against 38 the other way (exact binomial p = 2e-67). 80.4% of cell-cycle protein dynamics "
     "has no transcriptional source, and no parameter choice in this equation can supply one"),
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
    # ADDED by loop 120. The wiring MACHINERY works and is now part of this module -- tf_wiring,
    # tf_index, tf_drive, integrate_tf put the network into k_sm for the first time, and switching
    # the drive off reproduces the unwired steady state to 2.8e-15. The NETWORK inside it does not.
    ("TF network inside the rate law", "RUNS", "cell_assembled.integrate_tf/120",
     "k_sm_i(t) = kbar_i * max(0, 1 + G*sum_j s_ij*dev_j/n_i); with the drive off it reduces to "
     "the constant-k_sm integrator to 2.8e-15 relative, so the wiring is an addition and not a "
     "replacement. 54,128 signed edges, 1,451 regulators, 7,485 targets",
     "ONLY 8.8% OF THE NETWORK CAN ENTER A RATE LAW. The other 558,005 edges have sign 0 -- a "
     "binding event with no direction of effect, unusable in any equation"),
    ("TF network as a predictor of transcript dynamics", "FAILED", "loop_tf_cellcycle/120",
     "on 273 genes with a measured CCD-transcript call, a signed regulator and an mRNA state: "
     "real edge signs AUC 0.5465 against SHUFFLED signs 0.5494 +/- 0.0079 -- signs carry nothing; "
     "CCD-regulator enrichment +9.9 points against an out-degree-preserving null at +4.9 +/- 7.9, "
     "z = 0.63; the gene's own publication count (0.5536) beats every network score",
     "AND THE CONFOUND IS BIGGER THAN THE SIGNAL: regulator count alone scores AUC 0.5806, higher "
     "than the ODE, the signed topology, the raw count and both fame measures. Holding it fixed by "
     "quintile collapses integrated to 0.5348, topology to 0.5320 and raw count to 0.4985. The "
     "network was scoring how many regulators a gene has been ASSIGNED, which is annotation effort"),
    ("reaction+graph fusion", "FAILED", "loop_fusion_linear",
     "combined 0.166 BELOW graph-only; 90% of what remains survives degree-preserving rewiring",
     "the metabolic bridge is absent for 84% of genes"),
    # CORRECTED by loop 119. The stated cause -- "no phase-resolved layer exists" -- was true of
    # this repository and not of the literature. Mahdessian 2021 was fetched and the layer is now
    # here: 748 cell-cycle-dependent proteins against 776 imaged non-dependent controls, plus 530
    # CCD transcripts. What changed is the diagnosis, not the verdict.
    ("cell cycle (measured layer)", "STATIC", "loop_cellcycle_axis/119",
     "748 CCD proteins, 776 matched non-CCD controls, 530 CCD transcripts (Mahdessian 2021 via "
     "HPA); abundance does not explain the call (AUC 0.4648) and neither does fame (AUC 0.5063); "
     "CCD proteins carry 5.49% of measured proteome mass",
     "the calls are BINARY. There is still no per-gene phase and no per-cell pseudotime, so "
     "loop 99's ordering test cannot be redone and nothing advances a clock"),
    ("cell cycle as a process", "FAILED", "loop_cellcycle/99",
     "canonical phase order recovered but 176/500 shuffles do it too",
     "no cycle advances; the failure is now known to be the wiring, not the data -- see below"),
    # ADDED by loop 121, which asked where the oscillation comes from once transcription and the
    # regulatory network are both eliminated, and got a NEGATIVE that is more useful than its
    # passes: the group that needs a post-transcriptional mechanism has the LEAST degron signal.
    ("regulated degradation b(t)", "RUNS", "cell_assembled.integrate_deg/121",
     "dP/dt = k_sp*Mbar - b(t)*P with the mRNA held flat; beta=0 reduces exactly (2.8e-15). With "
     "b swinging 0 to 2*bbar, 30.1% of 4,190 genes reach a 20% swing -- the existing equation "
     "gives exactly zero at every parameter, so this is the term that was missing. Threshold: a "
     "half-life under 24.5 h at a 24 h cycle",
     "62.6% of measured CCD proteins have half-lives ABOVE that threshold, so timed destruction "
     "cannot be the whole answer even where it applies"),
    ("degron motifs as the mechanism", "FAILED", "loop_degron/121",
     "D-box R..L and D-box+ do not clear z=2 against either a label shuffle or a composition-"
     "preserving sequence shuffle. KEN-box clears both at z +2.6 -- and is one of the two motif "
     "densities that FAILED the fame check (rho +0.2429 with publication count)",
     "AND THE DISCRIMINATING TEST CAME OUT BACKWARDS. Degron density ranks both-oscillate > "
     "transcript-only > protein-only for every motif. The 359 proteins that oscillate WITHOUT "
     "their mRNA -- the ones that need a post-transcriptional mechanism by construction -- carry "
     "the LEAST degron signal. Timed destruction is the mechanism for the classic APC/C "
     "substrates that were already transcriptionally periodic, not for the unexplained majority"),
    ("relocalisation as the mechanism", "STATIC", "loop_degron/121 post-hoc",
     "88.9% of CCD proteins are annotated to more than one compartment against 69.0% of the "
     "imaged controls (+19.8 points, p < 1e-4, AUC 0.5992) -- larger than every sequence feature, "
     "every annotation feature and publication count, and it survives stratifying by antibody "
     "reliability (0.5953)",
     "NOT A DEMONSTRATED MECHANISM. Post-hoc, gated on nothing, and the CCD call and the location "
     "call come from the SAME immunofluorescence images. The specific relocalisation signature -- "
     "nucleus AND cytosol -- points the WRONG WAY (-4.7%). What separates the groups is the COUNT "
     "of compartments, which is also what an imaging-confidence confound looks like"),
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
