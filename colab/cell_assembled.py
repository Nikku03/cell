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


# LOOP 128 measured what a human rebuild costs, and the answer is that the two builds are NOT
# interchangeable. Itzhak HeLa copies with Mathieson human half-lives cover 5,595 genes against
# this function's 4,190 and 90.53% of proteome mass against 79.89% -- but the DERIVED protein
# production rate comes out 5.06x the self-consistent mouse one (Spearman +0.7518, so the ranking
# survives and only the scale moves). Each build closes its own ribosome budget (32.8% against
# 18.3%) because the ribosome count scales with the protein content, so the discrepancy is largely
# cell size and turnover rate, not error. What it forbids is SWAPPING: anything calibrated on one
# cannot take numbers from the other. This function stays on Schwanhausser for that reason.
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
            # DEGRADATION WITHOUT THE DILUTION TERM, for integrate_cell. Loop 125 made division
            # explicit, so a model that halves its contents must NOT also leak at mu -- that would
            # count the same event twice. Both forms are returned because the older integrators
            # still use the combined one and their recorded results depend on it.
            "k_loss_mrna_deg": a - MU, "k_loss_prot_deg": b - MU,
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


def integrate_tl(ksp, Mbar, bbar, T, beta=1.0, ncyc=12, nstep=400):
    """dP/dt = k_sp(t)*Mbar - bbar*P with k_sp(t) = ksp*(1 + beta*sin(wt)) and the mRNA held FLAT.

    THE THIRD AND LAST TERM. loop 119 eliminated M(t), loop 120 eliminated the wiring above it,
    loop 121 tested b(t) and found the degron signal sitting on the wrong genes. This is regulated
    translation, the remaining half of the space -- cap-dependent initiation drops in mitosis, and
    which mRNAs keep going is decided in the 5' UTR.

    loop 122's U6 gates on the fact that this and integrate_deg reach the SAME relative amplitude:
    both are the same first-order filter with the same corner frequency bbar, driven from opposite
    sides. If that holds, no mechanism inside this equation can exceed gain(bbar, T), whichever
    term it acts on -- which is a much stronger statement than either loop alone could make.
    """
    dt = T / nstep
    w = 2.0 * np.pi / T
    eb = np.exp(-bbar * dt)
    P = ksp * Mbar / bbar
    total = ncyc * nstep
    tr = np.zeros((nstep, len(P)))
    for s in range(total):
        k_t = np.maximum(ksp * (1.0 + beta * np.sin(w * (s * dt))), 0.0)
        P = P * eb + (k_t * Mbar / bbar) * (1 - eb)
        if s >= total - nstep:
            tr[s - (total - nstep)] = P
    mean = tr.mean(0)
    rel = (tr.max(0) - tr.min(0)) / (2.0 * np.maximum(mean, 1e-300))
    return rel, mean


def integrate_division(k, b_deg, T, ncyc=200, nstep=2000):
    """dP/dt = k - b_deg*P with the contents HALVED at every t = T. The cell finally divides.

    THE TERM THAT WAS ALREADY THERE. Every loss rate in this repository is ln2/t_half + mu, and mu
    is ln2/t_double -- the growth-dilution term. So the model has been ASSUMING division in every
    rate it computes while never performing one, and treating a continuous leak as a stand-in for a
    discrete event nobody had checked.

    b_deg here is DEGRADATION ONLY, with no mu, because dilution is now explicit.

    The periodic steady state has a closed form, which loop 125's D1 gates the integrator against:
        x = exp(-b*T),  P(0+) = (k/b)(1-x)/(2-x),  P(T-) = 2*P(0+)
    so max/min is exactly 2 for a perfectly stable protein, and the cycle mean is
        k/b + (P(0+) - k/b)(1-x)/(b*T)

    Returns (trajectory sampled at t = 0, dt, ..., T-dt ; the value at T- just before halving).
    The trajectory is recorded at the START of each substep, so tr[0] is P(0+) exactly. Recording
    at the END instead cost loop 125 its D1 gate on the first run: for a 0.1 h half-life a single
    step moves the protein 4.7%, so tr[0] was P(dt) and the comparison against the closed form
    missed by that much -- an off-by-one-step, not a model error, and the gate was right to catch it.
    """
    dt = T / nstep
    e = np.exp(-b_deg * dt)
    ss = k / np.maximum(b_deg, 1e-300)
    P = ss.copy() if hasattr(ss, "copy") else np.atleast_1d(ss).astype(float)
    tr = np.zeros((nstep, len(np.atleast_1d(k))))
    for c in range(ncyc):
        for s in range(nstep):
            if c == ncyc - 1:
                tr[s] = P
            P = P * e + ss * (1 - e)
        if c == ncyc - 1:
            return tr, P
        P = P / 2.0
    return tr, P


def window_means(tr, nwin):
    """Average a trajectory into nwin equal windows -- what a fractionation experiment measures.

    Elutriation does not sample an instant, it pools a slice of the cycle. Comparing an
    instantaneous max/min against a fraction-averaged one overstates every amplitude, so any
    comparison with Ly's six fractions has to average the same way they did.
    """
    n = tr.shape[0]
    edges = np.linspace(0, n, nwin + 1).astype(int)
    return np.stack([tr[edges[i]:edges[i + 1]].mean(0) for i in range(nwin)])


def integrate_cell(st, T, ix=None, dev_at=None, beta_deg=0.0, beta_tl=0.0,
                   divide=True, ncyc=40, nstep=400, gain=1.0):
    """EVERY WIRING AT ONCE, which is the only way to find out whether they compose.

        dM/dt = k_sm * (1 + gain*TF_drive(t))      - a_deg * M
        dP/dt = k_sp * (1 + beta_tl*sin(wt)) * M   - b_deg * (1 + beta_deg*sin(wt)) * P
        and both are HALVED at every t = T when divide is True.

    a_deg and b_deg are DEGRADATION ONLY -- the mu that stood in for division is dropped, because
    division is now performed rather than approximated. With every drive at zero and divide=True
    this must reproduce loop 125's closed form exactly, and cell_run gates on that: a set of
    wirings that do not reduce to the unwired model are not wirings, they are a different model.

    Returns (M trajectory, P trajectory, M at T-, P at T-), the trajectories sampled at the start
    of each substep so index 0 is the post-division state.
    """
    n = len(st["genes"])
    a = st["k_loss_mrna_deg"]
    b = st["k_loss_prot_deg"]
    ks, kp = st["k_sm"], st["k_sp"]
    dt = T / nstep
    w = 2.0 * np.pi / T
    ea = np.exp(-a * dt)
    M = ks / a
    P = kp * M / b
    trM = np.zeros((nstep, n))
    trP = np.zeros((nstep, n))
    for c in range(ncyc):
        last = (c == ncyc - 1)
        for s in range(nstep):
            if last:
                trM[s] = M
                trP[s] = P
            t = s * dt
            k_t = ks if ix is None else ks * np.maximum(0.0, 1.0 + gain * tf_drive(ix, dev_at(t)))
            b_t = b * (1.0 + beta_deg * np.sin(w * t))
            b_t = np.maximum(b_t, 1e-12)
            eb = np.exp(-b_t * dt)
            kp_t = kp * (1.0 + beta_tl * np.sin(w * t))
            Mn = M * ea + (k_t / a) * (1 - ea)
            P = P * eb + (kp_t * M / b_t) * (1 - eb)
            M = Mn
        if last:
            return trM, trP, M, P
        if divide:
            M, P = M / 2.0, P / 2.0
    return trM, trP, M, P


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
    ("CollecTRI as a replacement network", "FAILED", "loop_collectri/130",
     "loop 120's protocol run unchanged on literature-curated CollecTRI. Marginally better on "
     "identical genes -- AUC 0.5717 against the existing network's 0.5610 -- and it is the first "
     "network in this arc to beat fame (0.5717 against the gene's own publication count 0.5586). "
     "That is the whole of the good news",
     "IT FAILS THE SAME THREE GATES. Real signs AUC 0.5717 against SHUFFLED signs 0.5732 +/- "
     "0.0064 -- the shuffle scores HIGHER, exactly as it did for the existing network. "
     "CCD-regulator enrichment z 1.28, up from 0.63 and still short of 2. And regulator count "
     "alone scores 0.5856, above CollecTRI, with stratification collapsing it to 0.5491. Curated "
     "signs from the literature do not help, so the problem is not the network SOURCE"),
    ("reaction+graph fusion", "RUNS", "loop_fusion_fix/140 + loop_fusion_margin/141",
     "THE RECORDED FAILURE WAS AN ARTEFACT OF EARLY FUSION. loop 96 embedded the MERGED adjacency "
     "and scored 0.3935, below graph-only's 0.4898. Embedding the channels separately and "
     "concatenating scores 0.5358 against graph-only's 0.5275 on identical folds: +0.0084 [+0.0032, "
     "+0.0138], an interval that excludes zero. It also beats its own noise control decisively -- a "
     "random block of the same shape, scale and sparsity pattern scores -0.0091 [-0.0152, -0.0027], "
     "so noise does not merely fail to help, it hurts. Late over early fusion is +0.1364 [+0.1139, "
     "+0.1598], 16x larger",
     "THE MECHANISM AND TWO LIMITS. Merging adjacencies puts metabolite-mediated edges into one "
     "Laplacian with protein interactions; a single common metabolite links hundreds of genes, so "
     "hub structure dominates the leading eigenvectors. Early fusion survives degree-preserving "
     "rewiring at 90.1% -- it became a DEGREE detector -- while late fusion survives 34.2% against "
     "graph-only's 36.0%, keeping the specific-edge signal. LIMIT ONE: the biological gain is +1.59% "
     "of a 0.5275 baseline. LIMIT TWO, and it is a gate I got wrong: loop 141 M4 was written as a "
     "comparison of two point estimates and passed, but the bridge-gene effect's own interval is "
     "[-0.0233, +0.0265] and includes zero, 5.9x wider than the effect it tests. With 1,008 genes it "
     "is underpowered roughly six-fold, so it is NOT established that the gain comes from "
     "metabolism-specific structure rather than from a group-level split between genes that have "
     "reaction edges and genes that do not"),
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
    # ADDED by loop 122, which fetched the 5' UTR and closed the last term. The two production/loss
    # integrators are the SAME first-order filter driven from opposite sides -- measured, not
    # argued: median amplitude 0.1647 vs 0.1644 against an analytic gain(bbar,T) of 0.1647, median
    # per-gene disagreement 0.22%. So there is ONE bound over every mechanism the equation contains.
    ("regulated translation k_sp(t)", "FAILED", "loop_utr5/122",
     "5' UTR built by subtracting the CDS from the MANE Select cDNA -- exact, 19,293 genes, median "
     "133 nt, validated by 71.6% 5'TOP in ribosomal proteins (16.3x) and 86.1% Kozak -3 purine. "
     "U3: no feature survives Bonferroni. U4: the predicted direction fails on all five surviving "
     "features (best p = 0.12). U5: best 5' UTR AUC 0.5311, below the degron feature's 0.5345",
     "AND THE 5'TOP MOTIF IS A FAME PROXY (rho +0.3296 with publication count) -- a positional "
     "sequence motif, excluded by the same check that caught KEN-box in loop 121. TOP mRNAs are "
     "ribosomal proteins and translation factors, which are among the most-studied genes"),
    ("the amplitude bound on every mechanism", "CLOSES", "loop_utr5/122 U6",
     "driving k_sp or driving b gives the SAME relative amplitude beta*gain(bbar,T): 0.1647 vs "
     "0.1644, analytic 0.1647, median disagreement 0.22%. Nothing inside dP/dt = k_sp*M - b*P can "
     "exceed that bound, whichever term it acts on. At 20% amplitude it needs a half-life under "
     "24.5 h at a 24 h cycle",
     "62.6% of measured CCD proteins are ABOVE the threshold. For those the question is not which "
     "term carries the cycle -- none of them can, and the equation itself is what has to change"),
    # ADDED by loop 123, which fetched the measurement loop 122 said would settle it: Ly et al.
    # 2014 (eLife 3:e01630), NB4 cells by centrifugal elutriation -- no drug, no synchronisation --
    # with label-free MS and RNA-Seq on the SAME cells in the SAME table. Two instruments, one
    # experiment, neither of them a camera.
    ("cell cycle, measured without a camera", "STATIC", "loop_ms_cellcycle/123",
     "6,470 genes quantified by MS across six elutriation fractions, 5,553 with matched RNA-Seq. "
     "Not a cell-size artefact: peaks spread over all six fractions (largest 21.9%), 57.6% rising "
     "and 42.4% falling. The protein-without-mRNA asymmetry REPRODUCES -- 575:135 at 1.5x "
     "(p 1.8e-65), 139:78 at 2x (p 4.2e-05). Half-life predicts MS oscillation at AUC 0.6373 "
     "(median 29.5 h vs 50.3 h, p 0.0000), beating publication count at 0.5785",
     "IT IS FIVE TIMES WEAKER THAN THE CAMERA SAID: 1.8:1 by mass spectrometry against 9.5:1 by "
     "imaging, and it vanishes at 3x where only 74 genes remain discordant. The imaging call is "
     "real (9.6x enriched over controls) but OVER-CALLS SEVENFOLD -- MS confirms 42 of 315"),
    ("the production ceiling tanh(b*T/4)", "CLOSES", "loop_ms_cellcycle/123",
     "for dP/dt = k(t) - b*P with k(t) >= 0 of ANY waveform, max relative amplitude is exactly "
     "tanh(b*T/4). No drive parameter, no fitting, depends only on the measured half-life and the "
     "cycle length. Degradation is NOT bounded by it, because b(t) has no upper limit -- so an "
     "oscillation above the ceiling cannot be produced by transcription or translation at all. "
     "55 of 80 MS-oscillating genes with a measured half-life (68.8%) exceed it",
     "SUPERSEDES the sinusoid-specific beta*gain(bbar,T) used in loops 121-122, which understated "
     "what production can do because a real destruction switch is a pulse and not a sine. And the "
     "prediction it makes FAILS ITS OWN FOLLOW-UP: the over-ceiling genes carry LESS D-box+ than "
     "the under-ceiling ones (0.1517 vs 0.2018, p 0.273), so either that motif is a poor proxy for "
     "regulated degradation or the cross-species half-lives are wrong for these genes"),
    ("relocalisation as the mechanism", "STATIC", "loop_degron/121 post-hoc",
     "88.9% of CCD proteins are annotated to more than one compartment against 69.0% of the "
     "imaged controls (+19.8 points, p < 1e-4, AUC 0.5992) -- larger than every sequence feature, "
     "every annotation feature and publication count, and it survives stratifying by antibody "
     "reliability (0.5953)",
     "NOT A DEMONSTRATED MECHANISM. Post-hoc, gated on nothing, and the CCD call and the location "
     "call come from the SAME immunofluorescence images. The specific relocalisation signature -- "
     "nucleus AND cytosol -- points the WRONG WAY (-4.7%). What separates the groups is the COUNT "
     "of compartments, which is also what an imaging-confidence confound looks like"),
    # THE SYNTHESIS, from loop 122's post-hoc split. Splitting the CCD proteins at the amplitude
    # bound separates TWO signals that behave completely differently, which is the closest thing to
    # an answer four loops of elimination have produced.
    # RETIRED by loop 123's V4, which is what a decisive test is for. The geometry hypothesis
    # predicted dual localisation would be higher among imaging calls the mass spectrometer
    # DISCONFIRMS. It is 88.6% against 88.1%, p = 1.0000 -- dead flat. Dual localisation separates
    # imaging-CCD from imaging-control (+19.5%, p < 1e-4) but not true calls from false ones, so it
    # is a property of what the imaging pipeline scores, not a mechanism converting movement into
    # apparent abundance. Kept in the record because a retired hypothesis that vanishes is a
    # hypothesis nobody can check.
    ("relocalisation, RETIRED", "FAILED", "loop_ms_cellcycle/123 V4",
     "predicted: dual localisation higher in the MS-disconfirmed imaging calls. Measured: "
     "disconfirmed 88.6% (n=273) vs confirmed 88.1% (n=42), difference +0.5%, permutation p 1.0000",
     "the +19.8-point association loop 121 found is real and is NOT a relocalisation mechanism"),
    # ADDED by loop 137: the fourth elimination, and the first evidence the 362 are real.
    ("annotated ubiquitin targeting as the mechanism", "FAILED", "loop_ubiq_dynamics/137",
     "curator-annotated ubiquitin cross-links and the Ubl-conjugation keyword from UniProt, on the "
     "exact 88/362/38/642 split loop 119 recorded. NOT enriched among the 362: odds ratio 1.28 "
     "against the 642 that oscillate in neither, Fisher p = 0.098, and 0.87 against the 38 with p = "
     "0.71. This is the fourth candidate mechanism eliminated and the first tested on curator "
     "annotation rather than on a sequence motif",
     "THE FAME CONTROL RETURNED SOMETHING WORTH MORE THAN THE TEST IT POLICED. Publication count "
     "predicts the ubiquitin annotation at AUC 0.7169 and membership of the 362 at 0.4760. Fame "
     "drives the ANNOTATION hard and the PHENOTYPE not at all, so after loops 121 and 122 were both "
     "found to be partly measuring publication count, this is the first direct evidence that loop "
     "119's 362 are a real phenomenon and not an artefact of which proteins have been studied. Two "
     "gates caught my own errors first: U1 refused a run that tested 20,151 genes instead of 1,130 "
     "because the string 'NA' is truthy, and U4's AUC reported 0.7183 for a binary predictor that "
     "cannot exceed 0.52 until ties were given midranks"),
    ("the cell-cycle signal, decomposed", "STATIC", "loop_utr5/122 post-hoc",
     "split at the 24.5 h bound: 79 CCD proteins the equation CAN explain, 132 it cannot, 265 "
     "imaged controls. D-box+ degron density is FLAT across all three (fast +0.0035 p 0.89, slow "
     "-0.0138 p 0.52). Dual localisation is +21.3% (p 0.0005) in the fast half and +21.9% "
     "(p 0.0000) in the slow half -- IDENTICAL, so it is independent of turnover entirely. And "
     "loop 119's half-life association (AUC 0.5999) comes from SILAC mass spectrometry in mouse "
     "fibroblasts, an instrument with no camera in it, so real abundance change is also present",
     "TWO SIGNALS ARE MIXED IN ONE CALL. One tracks turnover and the equation can produce it for "
     "the 79 fast proteins. One tracks compartment count, is equally strong on both sides of the "
     "bound, and the equation has NO TERM FOR IT AT ALL. Separating them needs a cell-cycle "
     "abundance measurement that is not made with the same microscope"),
    ("replication as process", "FAILED", "loop_replication_time",
     "a Gaussian blur of the origin map (rho 0.5657 at sigma 400 kb) beats the fork simulation "
     "(0.4434). loop 138 V2 shows why the comparison was never a test of a fork model: sweeping fork "
     "speed over 0.25-14.0 kb/min, a factor of 56, moves rho by 0.0222, while sweeping a nuisance "
     "smoothing width over 50-1600 kb moves it by 0.1190 -- 5.4x more. The 'best' speed is 14.0 "
     "kb/min against a physiological 1-2",
     "MISLABELLED, AND loop 138 V3 SAYS WHAT IT ACTUALLY IS. The fork speed is UNIDENTIFIABLE: the "
     "data cannot see the model's central parameter, so no fork model was ever tested. What was "
     "tested is a distance-to-origin model with a free knob. And the blur that beat it is that "
     "model's own analytic limit -- 400 kb against a 840 kb fork traverse over S phase, a ratio of "
     "0.476. The simulation lost to its own closed form, which is a statement about the "
     "implementation and not about replication"),
    # ADDED by loop 139: not a rescue of the layer above, a different and smaller claim.
    ("replication timing from measured constants", "CLOSES", "loop_measured_constants/139 W1-W3",
     "sigma = v*T_S/2 with v the literature fork speed and T_S the recorded S phase. At v = 1.75 "
     "kb/min this gives sigma 420 kb and rho 0.5647 against measured Repli-chip RT, versus 0.5641 "
     "for the width FITTED on the same data -- 100.1% of the fitted optimum with NOTHING tuned. "
     "Robust across the whole literature range: v = 1.5 gives 0.5622 and v = 2.0 gives 0.5659, both "
     "within 0.4% of the fit",
     "THIS IS A SMALLER CLAIM THAN 'REPLICATION WORKS' AND THE DISTINCTION MATTERS. The fork "
     "SIMULATION remains FAILED: loop 138 V2 showed its speed parameter is invisible to the data, "
     "moving rho by 0.0222 across a 56x sweep while a nuisance smoothing width moves it by 0.1190. "
     "What closes here is the smoothed-origin model, which is that simulation's analytic limit, "
     "evaluated at a constant nobody fitted. It predicts WHERE replication is early or late; it does "
     "not simulate a fork"),
    ("expression noise", "FAILED", "loop_noise",
     "predicted CV tracks measured at +0.4837, against a cross-dataset abundance agreement ceiling "
     "of +0.8860. loop 138 V4 locates the failure precisely: partitioning and Poisson noise both "
     "give CV ~ N^(-1/2), so the slope of log10 CV on log10 N must be -0.5. Using OUR copy numbers "
     "it is -0.2055; using LARSSON'S OWN mean on the same genes it is -0.4063",
     "MISLABELLED, AND loop 139 W5 NAMES THE UNIT ERROR. The noise physics is closer to right than "
     "the record said -- within one dataset the exponent is -0.4063 against a predicted -0.5, and "
     "loop 139 W4 shows imposing the physical -0.5 costs 0.0937 of exponent while buying back a free "
     "parameter. Two separate faults remain and they are different sizes. The SMALL one is diploidy: "
     "my burst size is 0.5199 of the reported value against an expected 0.5000, because Larsson's "
     "parameters are per ALLELE and copy numbers are per CELL. That is a unit error, fixed by "
     "halving copies. The LARGE one is the cross-dataset join, which inflates b by 6.24x and which "
     "no choice of exponent or unit repairs"),
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
    # CORRECTED. This said "no volumes" and that stopped being true at loop 116, which gave every
    # compartment a literature volume fraction and swept the cell volume, and at loop 118, which
    # measured the cytosolic protein concentration at 163.3 mg/mL -- inside the real 100-400
    # range, from measured copy numbers and measured pool fractions rather than a label. A stale
    # ABSENT is worse than a FAILED: it tells the reader not to look.
    ("spatial organisation", "STATIC", "loop_geometry/116 + loop_spatial_proteome/118",
     "7 compartments with literature volume fractions (cytosol 0.45-0.60, nucleus 0.06-0.12, "
     "mitochondrion 0.08-0.22, ER 0.08-0.15, Golgi 0.02-0.04, lysosome 0.005-0.015, peroxisome "
     "0.002-0.008), cell volume swept at 2000/3000/4000 um3, and measured concentrations: cytosol "
     "163.3 mg/mL, whole cell 178.0 mg/mL, both inside the physically possible range",
     "VOLUMES AND CONCENTRATIONS, NOT GEOMETRY. Still no coordinates, no membrane surfaces, no "
     "distances -- a compartment is a well-stirred box with a size. And loop 116's V1 FAILED: an "
     "explicit label-to-compartment mapping could not beat the naive 40.3% agreement between a "
     "gene's annotated address and the compartment of its own reaction"),
    # CORRECTED by loop 126, and the correction is the result. The layer is still not implemented --
    # there are no coordinates and no Fickian solver -- but the approximation it would replace turns
    # out to be JUSTIFIED, measured rather than assumed. An approximation is only a defect when it
    # is violated, and this one is not, by four orders of magnitude.
    ("diffusion / well-stirred pools", "CLOSES", "loop_diffusion/126",
     "Damkohler number Da = (L^2/6D)(kcat*[E]/K_M) on 751 genes with a kcat, a K_M and a MEASURED "
     "copy number: median 3.45e-04 to 7.25e-04 across 3 cell volumes x 3 crowding factors, with "
     "2.7-3.2% of enzymes above 1. Diffusion constants from Stokes-Einstein at 37 C validated "
     "against GFP to 0.3% (117 um^2/s computed, 116.5 from the literature 87 at 25 C scaled by "
     "T/eta). Crossover at a compartment radius of 320 um, 36x the radius of the whole cell",
     "no coordinates, no membranes, no gradients -- the finding is that the model does not NEED "
     "them at this scale, not that it has them. And 2.7-3.2% of enzymes DO exceed Da = 1; those "
     "reactions are diffusion-limited and the well-stirred treatment is wrong for them"),
    ("the diffusion limit as a physics audit", "CLOSES", "loop_diffusion/126 F3-F5",
     "k_cat/K_M IS a bimolecular rate constant and cannot exceed Smoluchowski. Measured values: "
     "1 of 235 above the limit (0.43%, ANGEL2 at 1.1x). Predicted values: 2 of 751 (0.27%). Both "
     "respect a bound that owes nothing to any dataset",
     "BUT F5 FAILED AND IT IS THE SHARPEST RESULT HERE. The canonical catalytically perfect "
     "enzymes should sit at the top of the k_cat/K_M distribution -- that is why they are famous. "
     "The bundle ranks SOD1 at the 28th percentile and SOD2 at the 30th, predicting 1.43e4 for an "
     "enzyme whose real value is about 2e9, low by five orders of magnitude. Loop 124 measured "
     "this as a 12.95x fold-error against held-out data; F5 sees the same failure with no "
     "held-out data at all -- the predictions do not know which enzymes are fast"),
    # CORRECTED by loop 125. Recorded ABSENT, but division has been inside every rate in this
    # repository since loop 92 as mu = ln2/t_double -- a continuous leak standing in for a discrete
    # event nobody had tested. Present as an untested assumption is a worse place for it than absent.
    ("cell division", "CLOSES", "cell_assembled.integrate_division/125",
     "dP/dt = k - b*P with the contents HALVED at every T, matching its closed form P(0+) = "
     "(k/b)(1-x)/(2-x) to 1.3e-11 over half-lives from 0.1 h to 1000 h. AND IT BOUNDS THE "
     "ASSUMPTION THE WHOLE MODEL RESTS ON: the continuous-dilution stand-in understates the cycle "
     "mean by at most 3.97%, worst case at the most stable proteins, recovering the analytic "
     "1.5/1.4427 limit to 6.5e-07. Every rate here is now accurate to a proved four percent "
     "rather than a hoped-for one",
     "the cycle has no PHASES -- one period, one halving, no G1/S/G2/M and no checkpoint. It "
     "divides on a timer, not on a state"),
    ("division as a cell-cycle confound", "RUNS", "loop_division/125",
     "bare division produces a real sawtooth: six-window amplitude 1.088 to 1.770 across 4,821 "
     "measured half-lives, tracking half-life rather than uniform. But the ceiling is ANALYTIC -- "
     "P(T-) = 2*P(0+) for every b, which averages to 1.7698 over six windows and cannot be "
     "exceeded by any parameter. Loop 123's threshold is 2.0, a guaranteed 1.130x margin, and 0 of "
     "4,821 genes reach it. LFQ has already divided out ~86% of the sawtooth (10.82% residual "
     "against the 77.0% a per-cell quantification would show)",
     "I DRAFTED THE OPPOSITE CONCLUSION -- that division is common-mode and therefore normalises "
     "away -- and struck it. The amplitude spans 63% and is half-life dependent. Loop 123 survives "
     "on the analytic ceiling, not on uniformity"),
    ("partitioning noise", "CLOSES", "loop_division/125 D6",
     "each molecule goes to one daughter with p = 1/2, so a daughter gets Binomial(N, 1/2) and no "
     "gene can be quieter than CV = 1/sqrt(N). Parameter-free, from measured copy numbers: protein "
     "median 0.485% (42,566 copies), mRNA median 24.0% (17 copies) -- a 49-fold difference arising "
     "entirely from copy number",
     "no measured single-cell CV was joined, so the floor is a physical statement rather than a "
     "validated prediction; the Larsson join is still to do"),
    # CORRECTED. This said "no measured k_cat anywhere" and that was wrong -- I repeated it to the
    # user before checking. colab/data/kinetics_bundle.json.gz is committed and carries a kcat for
    # 8,184 reactions and 2,549 genes, from a measured base of 2,437 human records. The real problem
    # is not absence, it is that three quarters of it is machine-learning prediction that loop 124
    # then showed does not beat a constant. Second stale ABSENT found this session.
    ("enzyme kinetics", "STATIC", "kinetics_bundle + loop_kcat_audit/124",
     "8,184 reaction kcats and 2,549 gene kcats, committed. Tiers: 22.1% human-EC median over 362 "
     "distinct EC numbers, 72.6% CatPred prediction, 3.7% a constant 1.85/s, 1.6% any-organism EC. "
     "1,309 human proteins fetched from UniProt with literature-curated kinetics -- 242 kcat, "
     "1,275 KM, PubMed-cited, parser validated on PNP/FH/TK1",
     "THE PREDICTIONS DO NOT BEAT A CONSTANT. On 66 held-out genes the bundle's kcat scores 12.95x "
     "median fold-error against 9.42x for a flat 1.85/s (p 0.55). The claimed tier ordering breaks: "
     "human-EC 3.83x < GLOBAL MEDIAN 9.42x < CatPred 13.29x, so tier 2 is worse than the null it "
     "outranks. And the audit set is selected -- measured genes have 82 median publications against "
     "64 (p 0.0105) -- so this describes the enzymes people study"),
    ("kinetics filtered to the noise floor", "RUNS", "loop_kcat_floor/127",
     "THE EXPERIMENTAL NOISE FLOOR IS NOW MEASURED, NOT ASSUMED: predicting one human kcat record "
     "from the median of the others in its own EC gives 2.85x median leave-one-out error over "
     "2,221 records and 204 ECs. The proposed 4x operating point sits at the 60th percentile of "
     "that, so it is defensible. Shipped filter: keep a tier-1 EC median when its EC is shared by "
     "<= 3 genes, else use the constant 1.85/s FLAGGED AS A CONSTANT. 1,130 reactions measured, "
     "7,054 constant. Validated on 70 held-out UniProt genes: 2.82x accepted against 43.31x "
     "rejected, p 0.0035 (Bonferroni 0.014). -> colab/data/kinetics_filtered.json.gz",
     "THE SELECTOR WAS CHOSEN POST HOC and the artefact records that. The predeclared one -- each "
     "EC's own self-consistency -- did NOT predict accuracy (rho +0.10, p 0.39); EC BREADTH did. "
     "n = 70 is every gene with both a UniProt kcat and an EC carrying replicates, and the "
     "rejected-group medians swing from 6.7x to 43.3x across thresholds, which is what n = 70 "
     "looks like. A lead with a number on it, not an established filter. LOOP 129 TRIED TO "
     "RETEST IT ON DLKcat AND COULD NOT: 160 of 169 candidate genes are already tier 'measured' "
     "in the bundle because ec_kcat_medians was BUILT FROM DLKcat, leaving 9 genuinely held out. "
     "The filter is untested, not disproven, and loop 127's own UniProt validation remains the "
     "only evidence for it"),
    ("the true experimental noise floor", "CLOSES", "fetch_gaps + loop_kcat_floor2/129",
     "same protein AND same substrate, independently measured: median 1.15x over 101 values from "
     "49 pairs. This is reproducibility, and it needs no model -- it is a property of the "
     "measurements alone. Loop 127's 2.85x was leave-one-out WITHIN AN EC, which is "
     "enzyme-to-enzyme and substrate-to-substrate variation, roughly three times too large",
     "the 4x operating point survives but is about three times LOOSER than what can be measured, "
     "not close to the limit as loop 127 claimed"),
    ("replicate count as a quality signal", "FAILED", "loop_kcat_floor/127",
     "spread RISES with the number of measurements behind an EC: 1.54x at 3-4 records, 2.24x at "
     "5-9, 2.55x at 10-19, 2.62x at 20+",
     "'trust the well-measured ones' is exactly backwards -- a heavily measured EC is a BROAD "
     "class covering many enzymes and conditions, not a well-pinned one. Any filter built on "
     "record count would have selected for breadth, which is the thing that predicts error"),
    ("kcat from sequence, learned", "RUNS", "loop_ml_kcat/131 + loop_ml_audit/132",
     "17,004 measurements, 7,856 sequences, split by MinHash 5-mer cluster with the leakage "
     "audited at 0.371 max test-train Jaccard. Gradient boosting on ESM2-8M + Morgan fingerprints "
     "+ composition scores RMSE 1.3768 log10 against a constant's 1.5091 and the best lookup "
     "baseline (EC median) at 1.5015. CLUSTER bootstrap on the gain: +0.1323, 95% [+0.1003, "
     "+0.1668], all five folds positive. A properly tuned MLP ties it at 1.3867, 95% CI on the "
     "difference [-0.0143, +0.0345]. Every homology baseline scores WORSE than a constant, which "
     "is the split working. 0.000% of predictions violate the Smoluchowski limit",
     "IT CLOSES 9.5% OF THE DISTANCE to the 0.061 experimental floor, cutting typical fold error "
     "from 32x to 23x. And the ceiling is much closer than that: 86.0% of the variance is BETWEEN "
     "proteins, so a PERFECT per-protein predictor would score 0.5625 -- this model is 2.4x worse "
     "than a method-free bound on ignoring the substrate entirely. The substrate channel is real "
     "but small: shuffling substrates within a protein costs only +0.0191 of the +0.1323 gain"),
    ("what the kcat model actually learned", "FAILED", "loop_ml_probe/133 B3-B4",
     "the ESM embedding is a FAMILY DETECTOR, not a catalysis model. Same 320 numbers, same folds: "
     "EC top-level class accuracy 0.779 against a 0.394 majority, log10 sequence length R2 +0.7631, "
     "log10 kcat R2 +0.1381. loop 133's B4 reached this by refitting on the EC-median residual (RMSE "
     "1.5386 vs a residual sd of 1.4849), and loop 134 C2 showed that test was broken three ways: "
     "its training residual was built from a complement CONTAINING the test fold (+0.0312), its "
     "baseline was an in-sample sd (-0.0024), and a residual is the wrong instrument for the "
     "question. loop 134 C3 settles it properly and without a residual -- permuting the embedding "
     "among records SHARING an EC number costs +0.0046 against a paired interval of 0.0488, with C4 "
     "confirming first that 73.4% of records did receive a different sequence",
     "PROTEIN IDENTITY IS WORTH +0.0046, WHICH IS NOTHING. But 'everything is in the EC number' is "
     "wrong as stated and loop 134 C1 is why: the EC number explains 0.7% of the variance, while "
     "sequence-only scores 1.3890 against a constant's 1.5069. What the model uses is FAMILY "
     "STRUCTURE at a resolution the ESM embedding captures and the EC string does not -- swapping a "
     "protein for a class-mate is free, but the class itself is identified far better by the "
     "embedding than by the label. The +0.1323 gain loop 132 confirmed as real is real, and it is "
     "family-level, not per-protein"),
    ("what a kcat model would need instead", "STATIC", "loop_ml_probe/133 B1, B5, B6",
     "B1: 18,595 point-mutant pairs (equal length, 1-5 residues) differ by a median 0.649 log10 -- "
     "4.5x turnover from five residues -- and a mean-pooled embedding cannot see them, worth 0.947 "
     "log10 of irreducible RMSE. B5: within (protein, substrate) pairs measured twice the residual "
     "sd is 0.5137 log10 = 3.26x, against loop 129's 1.15x for identical conditions, so unrecorded "
     "temperature, pH and mutation status cost the difference. B6: the enzymes the cell model "
     "leans on are EASIER than average -- reaction-weighted RMSE 1.1895 against 1.3072 unweighted",
     "all three point the same way: family-level information is exhausted, and the only route left "
     "is resolution WITHIN a family -- active-site residues, mutation status, and the assay "
     "conditions that are absent from the file"),
    ("Michaelis constants K_M", "RUNS", "kinetics_bundle + loop_kcat_audit/124 K4",
     "the one kinetic parameter in this model that carries real information: predicted KM scores "
     "3.54x median fold-error against 5.77x for a constant 43.6 uM, on 628 genes with a measured "
     "counterpart, p = 0.0000. 30.1% land within 2x",
     "every KM in the bundle is tagged catpred+ECprior -- there is no measured-KM tier, so these "
     "are predictions that happen to be good, not measurements. The 1,275 UniProt values are "
     "fetched but not yet substituted in"),
]
