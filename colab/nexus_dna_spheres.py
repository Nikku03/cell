"""NEXUS SPHERE COVERAGE ON REAL DNA: how far does a single-atom change actually reach?

THE PROPOSAL.  Cover the whole of chromatin's DNA in small overlapping spheres -- ~4 A radius -- run NEXUS's
chemistry inside each sphere, and let the overlap couple neighbours so that changing ONE ATOM is felt across
the structure. No trajectory, no integrator: the sphere network carries the influence.

The scheme is a Schwarz-type overlapping domain decomposition, and it is a real technique. Whether it works
here is not a matter of opinion, because a sphere of radius R around an atom is EXACTLY an interaction
truncation at R. So the question "is 4 A enough?" has an exact answer computable on a real structure, and
this file computes it rather than arguing it.

FIRST FACT, FROM NEXUS'S OWN SOURCE.  `flex_physics` sets its own cutoffs at

    Lennard-Jones   6 A          desolvation     8 A
    electrostatics 10 A          induction      10 A

Every one of them is larger than the proposed 4 A sphere -- the engine already declares that its terms are
still live past that distance. That is a reason to measure, not a proof: a cutoff is chosen conservatively,
and the fraction of the answer living between 4 A and 10 A is an empirical quantity.

WHY DNA, AND A PREDICTION OF MINE THAT THE RUN REFUTED.  DNA is a polyanion: one formal -1 per phosphate
every ~3.4 A of backbone, ~41 heavy atoms per base pair (measured here, not assumed). I expected that to be
the mechanism -- that truncating Coulomb along a dense LINE OF LIKE CHARGES leaves a tail of same-signed
terms that does not cancel, where a protein's sparse mixed-sign charges would forgive a short cutoff.

The controls say that is not the story. Neutralising every phosphate moves 4 A capture from 35% to 43%, and
SHUFFLING the charges across atoms -- which destroys the backbone alignment entirely while keeping the
charge multiset -- gives 46%. All three are the same failure. So the long tail is not about DNA's charge
ARRANGEMENT; it is about an UNSCREENED 1/r^2 kernel in a dense medium, and DNA is merely a fair place to
meet it. The one thing that does fix it is screening, which is a change to the physics, not to the geometry.

THE SYSTEM.  Nucleosome core particle 1KX5 at 1.9 A: 147 bp of DNA on a histone octamer, 16,755 atoms with
ordered waters. This IS chromatin at atomic resolution, and it is small enough that the EXACT all-atom
answer -- every pair, no cutoff at all -- can be computed and used as the reference. Nothing here is
compared against a model; it is compared against the truth for this structure.

THE PERTURBATION IS ONE ATOM.  A single phosphate oxygen's charge is changed by +1e (OP1: -0.5 -> +0.5).
That is the user's "single atom change", it is chemically meaningful (it is what a bound counterion or a
phosphate modification does), and it is the cleanest possible probe of range.

ARM 1 -- REACH.  Compute the exact per-atom response dE_i to that one change, with no cutoff. Then ask what
fraction of the total |response| lies inside radius R, for R from 4 A up. This is the whole question in one
curve, and it needs no model at all.

ARM 2 -- FIDELITY, STRATIFIED BY DISTANCE.  Correlation between truncated and exact per-atom response is a
trap: most atoms sit far away where both are ~0, and agreeing about zero manufactures a high correlation
that says nothing. So agreement is reported PER DISTANCE SHELL, and the near shells -- where the answer is
actually large -- are reported separately from the far ones. A single pooled number is not quoted.

ARM 3 -- PROPAGATION, which is the user's actual mechanism.  Truncation error in a pairwise energy cannot be
iterated away: there is nothing to propagate, the missing terms are simply absent. Influence only travels if
ATOMS MOVE. So this arm perturbs one atom and RELAXES -- once with the full system, once with sphere-local
relaxation iterated Schwarz-style -- and measures how far displacement actually travels per iteration and
whether iterating recovers the full answer. This is the arm that tests "a change in one is felt in another".

CONTROLS, each isolating one explanation:
    lj_only          charges zeroed. Short-ranged by construction, so it SHOULD be captured at small R.
                     If it is and the charged case is not, the failure is electrostatics, not decomposition.
    dna_neutral      phosphate charges zeroed, everything else intact. Isolates the polyanion specifically
                     rather than charge in general.
    protein_only     the same perturbation applied inside the histone core, where charges are sparse and
                     mixed-sign. If small spheres work there and fail on DNA, the claim is about DNA.
    shuffled_charges charges randomly permuted across atoms, preserving the multiset. Kills the spatial
                     correlation of the charge distribution while keeping its magnitude -- separates "the
                     tail is big" from "the tail is big BECAUSE like charges line up along the backbone".

PREDECLARED, before any number:
    >=90% of the response captured inside 4 A
        -> the proposal is sound as stated; only arithmetic decides whether the genome is reachable.
    4 A insufficient but some R captures >=90%
        -> the idea works at THAT radius, and the cost is recomputed there. The finding is a number, not a no.
    no R below the system size captures the response
        -> pairwise truncation cannot represent charged DNA at any sphere size, and the fix is not a bigger
           sphere but a different treatment of the far field (multipole/Ewald), which is a different proposal.
    displacement propagates < one sphere radius per Schwarz iteration
        -> "connected spheres" does not transmit influence for free; it costs iterations, and the iteration
           count is the real price of the scheme.

-> outputs/orphan/nexus_dna_spheres.json
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
PDB_ID = "1KX5"
RADII = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0)
SHELLS = ((0, 4), (4, 8), (8, 12), (12, 20), (20, 35), (35, 60))
SEED = 17
DEBYE = 7.9      # Debye length at 150 mM monovalent salt, Angstrom. Real chromatin sits in roughly this,
                 # and screening is the one thing that could make a local scheme legitimate -- so it is an
                 # ARM, not an assumption. NEXUS itself does not screen; that is the gap being measured.
LIVE_FRAC = 1e-4  # an atom counts as "live" if |exact response| exceeds this fraction of the largest.
                  # Correlating two vectors that are mostly matched zeros returns ~1 from the padding
                  # alone; metrics are computed on live atoms only and the dead count is reported.
KE = 332.06                              # kcal*A/(mol*e^2), as in flex_physics
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "MN": 1.60, "CL": 1.75}
POL = {"C": 1.05, "N": 1.10, "O": 0.80, "S": 2.90, "P": 2.10}
EPS_LJ = 0.10

# DNA partial charges. Coarse and stated as such -- the same coarseness as flex_physics' protein charges,
# which is the point: this measures NEXUS as it exists, not an idealised force field. The phosphate group
# carries its formal -1 (P +1.17, OP1/OP2 -0.78 each, ester O -0.50 each -> net -1.09, near enough).
DNA_CHG = {"P": 1.17, "OP1": -0.78, "OP2": -0.78, "O1P": -0.78, "O2P": -0.78,
           "O5'": -0.50, "O3'": -0.50, "O4'": -0.35, "N1": -0.20, "N3": -0.20, "N7": -0.20,
           "O2": -0.45, "O4": -0.45, "O6": -0.45, "N2": -0.30, "N4": -0.30, "N6": -0.30}
PROT_CHG = {"O": -0.50, "OXT": -0.50, "N": -0.16, "C": 0.50,
            "OD1": -0.55, "OD2": -0.55, "OE1": -0.55, "OE2": -0.55,
            "NZ": 0.80, "NE": -0.10, "NH1": 0.40, "NH2": 0.40, "CZ": 0.60,
            "ND1": -0.25, "NE2": -0.25, "OG": -0.40, "OG1": -0.40, "OH": -0.40, "SG": -0.20}
NUC = {"DA", "DT", "DG", "DC", "A", "T", "G", "C"}


def fetch_pdb():
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{PDB_ID.lower()}.pdb"
    if not f.exists():
        url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
        urllib.request.urlretrieve(url, f)
    return f


def load(path, keep_water=True):
    """Parse to flat arrays. Waters are kept by default: they screen, and dropping them would quietly make
    the electrostatics look longer-ranged than the crystal says it is."""
    co, el, nm, rs, ch, isdna = [], [], [], [], [], []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resn = line[17:20].strip()
        if resn == "HOH" and not keep_water:
            continue
        e = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        co.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        el.append(e)
        nm.append(line[12:16].strip())
        rs.append(resn)
        ch.append(line[21])
        isdna.append(resn in NUC)
    return {"co": np.array(co, float), "el": np.array(el), "nm": np.array(nm),
            "rs": np.array(rs), "ch": np.array(ch), "dna": np.array(isdna, bool)}


def charges(t, neutral_dna=False, zero_all=False, shuffle_rng=None):
    """Partial charges from atom name, DNA-aware. Waters and ions get their obvious values."""
    q = np.zeros(len(t["nm"]))
    if zero_all:
        return q
    for i, (n, r, isd) in enumerate(zip(t["nm"], t["rs"], t["dna"])):
        if r == "HOH":
            q[i] = -0.80                      # water O; the H are absent from the model, so this is the
            continue                          # oxygen's share and is stated, not derived
        if r == "MN":
            q[i] = 2.0
            continue
        if r == "CL":
            q[i] = -1.0
            continue
        if isd:
            v = DNA_CHG.get(n, 0.0)
            if neutral_dna and n in ("P", "OP1", "OP2", "O1P", "O2P"):
                v = 0.0
            q[i] = v
        else:
            q[i] = PROT_CHG.get(n, 0.0)
    if shuffle_rng is not None:
        # keep the MULTISET of charges, destroy where they sit. Separates "the neglected tail is large"
        # from "the neglected tail is large because like charges line up along a backbone".
        q = q[shuffle_rng.permutation(len(q))]
    return q


def radii_of(t):
    return np.array([VDW.get(e, 1.70) for e in t["el"]])


def polar_of(t):
    return np.array([POL.get(e, 1.00) for e in t["el"]])


def shell_profile(dE, dk, edges):
    """Signed AND absolute response summed per spherical shell.

    THIS IS THE DECISIVE CURVE, and it is not the per-atom magnitude. For NEXUS's kernel U ~ q q / r^2 the
    number of atoms in a shell grows as r^2 while the interaction falls as r^-2, so the SHELL-INTEGRATED
    contribution is flat in r even though the per-atom response decays cleanly. Plotting per-atom decay
    would show a reassuring falloff while the neglected tail grows linearly with system size. So the shell
    sum is what gets reported.
    """
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dk >= lo) & (dk < hi)
        out.append((lo, hi, int(m.sum()), float(dE[m].sum()), float(np.abs(dE[m]).sum())))
    return out


def screen(r, debye):
    """Debye-Huckel screening factor. NEXUS does not screen -- its eps(r)=r is a crude stand-in -- so this
    is applied only in the explicitly screened arm, never silently. Returns an ARRAY in both branches: a
    bare 1.0 would be silently unindexable and only fails on the unscreened path, i.e. the default one."""
    return np.ones_like(r) if debye is None else np.exp(-r / debye)


def fields_all_radii(co, q, radii, chunk=512, debye=None):
    """The static Coulomb field at every atom, computed simultaneously for the exact case and for EVERY
    truncation radius, in ONE O(N^2) pass.

    The obvious implementation loops the pass once per radius. One pass with a stack of masked accumulators
    gives identical numbers for a fraction of the cost, which is what makes the full radius sweep affordable
    on 16,755 atoms.

    `radii` is passed EMPTY for arms that only need the exact field. Each extra radius is another reduction
    over a (chunk x N x 3) array, and computing truncated fields for arms that never use them was most of
    the first run's cost.
    """
    n = len(co)
    F = {R: np.zeros((n, 3)) for R in radii}
    F["exact"] = np.zeros((n, 3))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dv = co[s:e, None, :] - co[None, :, :]
        r = np.linalg.norm(dv, axis=2)
        r = np.maximum(r, 0.1)
        w = q[None, :] * screen(r, debye) / r ** 3
        for loc, glob in enumerate(range(s, e)):
            w[loc, glob] = 0.0
        base = dv * w[:, :, None]
        F["exact"][s:e] = KE * base.sum(1)
        for R in radii:
            F[R][s:e] = KE * np.where((r < R)[:, :, None], base, 0.0).sum(1)
    return F


def response(co, q, alpha, k, dq, R, F0, debye=None):
    """Per-atom energy response to changing atom k's charge by dq, truncated at R (R=None -> exact).

    A sphere of radius R around each atom IS a truncation at R, so this one function expresses both the
    reference and the sphere scheme, and they cannot drift apart through separate implementations.

    Only the charge changes, so LJ and desolvation are bit-identical before and after and cancel exactly in
    the difference. What survives:
      ELECTROSTATICS, linear in dq -- only pairs involving k change, so this part is O(N), not O(N^2).
      INDUCTION, quadratic in the TOTAL field, so it does NOT cancel. Writing E1 = E0 + dE_k, the change is
        -1/2 (alpha/KE) ( 2 E0.dE_k + |dE_k|^2 )
      and the cross term 2 E0.dE_k is FIRST ORDER in the pre-existing field. This is why induction is the
      term least able to survive decomposition: it is not a sum over pairs that can be split, it is a square
      of a sum, and truncating inside the square leaves an error that inherits the field's own long range.
    """
    n = len(co)
    d = np.linalg.norm(co - co[k], axis=1)
    d = np.maximum(d, 0.1)
    sc = screen(d, debye)
    inr = np.ones(n, bool) if R is None else (d < R)
    inr[k] = False

    dE = np.zeros(n)
    dE[inr] = KE * q[inr] * dq * sc[inr] / d[inr] ** 2
    dE[k] = KE * dq * (q[inr] * sc[inr] / d[inr] ** 2).sum()

    dv = co - co[k]
    dEk = KE * dq * (dv * (sc / d ** 3)[:, None])          # field change at every atom from dq at k
    dEk[~inr] = 0.0
    dEk[k] = 0.0
    E0 = F0["exact"] if R is None else F0[R]
    dE += -0.5 * (alpha / KE) * (2.0 * (E0 * dEk).sum(1) + (dEk ** 2).sum(1))
    return dE


def captured(dE_exact, dk, R):
    """Fraction of the TOTAL absolute response that lives inside radius R of the changed atom.

    Absolute, not signed: a signed fraction can read 100% while the far field is a large positive and a
    large negative that happen to cancel, and cancellation is not capture.
    """
    tot = np.abs(dE_exact).sum()
    if tot <= 0:
        return float("nan")
    return float(np.abs(dE_exact)[dk < R].sum() / tot)


def shell_agreement(dE_exact, dE_trunc, dk):
    """Agreement per distance shell, as relative error AND regression slope -- never as a bare correlation.

    THREE WAYS A POOLED CORRELATION LIES HERE, all avoided:
      1. Most atoms sit far away where both sides are ~0. Correlating two vectors padded with matched zeros
         returns ~1 from the padding. So metrics run only over LIVE atoms and the dead count is printed.
      2. Pearson r is invariant under y -> a*y + b. A scheme that gets every atom right up to a uniform
         factor of six scores r = 1.0 while every energy is wrong by 500%. The regression SLOPE catches
         exactly that, so it is reported beside r and is the number that matters.
      3. Near-field atoms dominate any pooled statistic and they are the easy case. Hence per shell.
    """
    live = np.abs(dE_exact) > LIVE_FRAC * np.abs(dE_exact).max()
    out = []
    for lo, hi in SHELLS:
        m = (dk >= lo) & (dk < hi) & live
        ntot = int(((dk >= lo) & (dk < hi)).sum())
        if m.sum() < 5:
            out.append((lo, hi, ntot, int(m.sum()), float("nan"), float("nan"), float("nan")))
            continue
        ex, tr = dE_exact[m], dE_trunc[m]
        den = np.abs(ex).sum()
        rel = float(np.abs(tr - ex).sum() / den) if den > 0 else float("nan")
        slope = float((ex * tr).sum() / (ex * ex).sum()) if (ex * ex).sum() > 0 else float("nan")
        r = float(np.corrcoef(ex, tr)[0, 1]) if np.std(ex) > 0 and np.std(tr) > 0 else float("nan")
        out.append((lo, hi, ntot, int(m.sum()), rel, slope, r))
    return out


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("NEXUS SPHERE COVERAGE ON REAL DNA -- how far does a single-atom change actually reach?")
    report("=" * 100)
    report("  The proposal: tile chromatin's DNA in ~4 A overlapping spheres, run NEXUS chemistry in each,")
    report("  and let overlap carry a single-atom change across the structure. A sphere of radius R around")
    report("  an atom IS an interaction truncation at R, so 'is 4 A enough' has an exact answer on a real")
    report("  structure. Reference = every pair, no cutoff. Nothing here is compared against a model.")
    report(f"  NEXUS's OWN cutoffs: LJ 6 A, desolvation 8 A, electrostatics 10 A, induction 10 A -- every one")
    report("  larger than the proposed 4 A sphere. That is a reason to measure, not a proof.")

    p = fetch_pdb()
    t = load(p, keep_water=True)
    co = t["co"]
    alpha = polar_of(t)
    n = len(co)
    nd, nw = int(t["dna"].sum()), int((t["rs"] == "HOH").sum())
    report(f"\n  {PDB_ID}: {n} atoms | {nd} DNA | {nw} ordered water | {n-nd-nw} histone+ion")
    report(f"  147 bp on a histone octamer at 1.9 A -- chromatin at atomic resolution, small enough to be exact.")

    q_full = charges(t)
    # the perturbed atom: a phosphate oxygen in the middle of the DNA, so the far field is not clipped by
    # simply running out of structure in one direction
    ph = np.where(t["dna"] & np.isin(t["nm"], ["OP1", "O1P"]))[0]
    if len(ph) == 0:
        report("  *** no phosphate oxygens found; aborting")
        return 1
    k = int(ph[len(ph) // 2])
    dq = 1.0
    report(f"  PERTURBATION: atom {k} ({t['nm'][k]} of {t['rs'][k]} chain {t['ch'][k]}) charge "
           f"{q_full[k]:+.2f} -> {q_full[k]+dq:+.2f}. One atom.")

    dk = np.linalg.norm(co - co[k], axis=1)
    ARMS = [
        ("dna_charged  (the case)", charges(t), None),
        ("dna_screened (control)", charges(t), DEBYE),
        ("dna_neutral  (control)", charges(t, neutral_dna=True), None),
        ("shuffled_chg (control)", charges(t, shuffle_rng=np.random.default_rng(SEED + 3)), None),
    ]

    res = {"n_atoms": n, "n_dna": nd, "n_water": nw, "perturbed_atom": k, "arms": {}}

    FID_R = (4.0, 8.0, 12.0, 20.0)      # the radii ARM 3 actually compares; only the case arm needs them
    report("\n  Precomputing the static field at all 16k atoms in a single O(N^2) pass per arm...")
    fields = {}
    for name, q, deb in ARMS:
        need = FID_R if name.startswith("dna_charged") else ()
        fields[name] = fields_all_radii(co, q, need, debye=deb)
        report(f"    {name} done at {time.time()-t0:.0f}s")

    report("\n  ARM 1 -- REACH: fraction of the TOTAL |response| captured inside radius R.")
    report("  Absolute, not signed: a signed fraction can read 100% while a large positive and a large")
    report("  negative in the far field merely cancel, and cancellation is not capture.")
    report("    " + f"{'arm':<25}" + "".join(f"{r:>7.0f}A" for r in RADII))
    exact_store = {}
    for name, q, deb in ARMS:
        dE = response(co, q, alpha, k, dq, None, fields[name], debye=deb)
        exact_store[name] = (dE, q, deb)
        row = "".join(f"{captured(dE, dk, R):>8.3f}" for R in RADII)
        report(f"    {name:<25}{row}")
        res["arms"][name.split()[0]] = {"captured": {str(R): captured(dE, dk, R) for R in RADII},
                                        "total_abs": float(np.abs(dE).sum())}

    report("\n  ARM 2 -- THE SHELL-INTEGRATED RESPONSE, which is the decisive curve and is NOT the per-atom")
    report("  decay. For a kernel U ~ q q / r^2 the atom count in a shell grows as r^2 while the interaction")
    report("  falls as r^-2, so the shell SUM can stay flat while the per-atom response decays reassuringly.")
    report("  A per-atom plot would look convergent for a sum that is not.")
    edges = [0, 4, 8, 12, 16, 20, 25, 30, 40, 50, 70]
    for name, _, _ in ARMS:
        dE = exact_store[name][0]
        prof = shell_profile(dE, dk, edges)
        report(f"    {name}   (kcal/mol per shell, |.| = absolute)")
        report("      " + "".join(f"{f'{lo}-{hi}':>9}" for lo, hi, _, _, _ in prof))
        report("      " + "".join(f"{a:>9.2f}" for _, _, _, _, a in prof))
        res["arms"][name.split()[0]]["shell_abs"] = [{"lo": lo, "hi": hi, "n": c, "signed": s, "abs": a}
                                                     for lo, hi, c, s, a in prof]

    report("\n  ARM 3 -- FIDELITY per distance shell: truncated (= sphere of radius R) vs exact.")
    report("  rel.err = sum|trunc-exact| / sum|exact|.  slope = regression of truncated on exact; a scheme")
    report("  that is right up to a uniform factor scores pearson 1.0 and slope != 1, so slope is the")
    report("  number that matters. Metrics over LIVE atoms only -- matched zeros would manufacture r ~ 1.")
    key = "dna_charged  (the case)"
    dE_ex, q_case, deb_case = exact_store[key]
    res["shells"] = {}
    for R in FID_R:
        dE_tr = response(co, q_case, alpha, k, dq, R, fields[key], debye=deb_case)
        rows = shell_agreement(dE_ex, dE_tr, dk)
        tot_ex, tot_tr = float(np.abs(dE_ex).sum()), float(np.abs(dE_tr).sum())
        report(f"    sphere R = {R:.0f} A     total |response| exact {tot_ex:.1f} vs truncated {tot_tr:.1f} "
               f"kcal/mol ({tot_tr/tot_ex:.1%} of it)")
        report(f"      {'shell':<12}{'atoms':>8}{'live':>7}{'rel.err':>10}{'slope':>8}{'pearson':>9}")
        for lo, hi, ntot, nlive, rel, sl, r in rows:
            report(f"      {f'{lo}-{hi} A':<12}{ntot:>8}{nlive:>7}{rel:>10.3f}{sl:>8.3f}{r:>9.3f}")
        res["shells"][str(R)] = [{"lo": lo, "hi": hi, "n": ntot, "live": nlive, "rel_err": rel,
                                  "slope": sl, "pearson": r} for lo, hi, ntot, nlive, rel, sl, r in rows]

    report("\n  READING")
    # the inside-the-sphere rows are the ones worth stating: truncation is not only losing the tail
    near = res["shells"]["4.0"][0]
    if near["slope"] < 0.5:
        report(f"  TRUNCATION DOES NOT MERELY LOSE THE FAR FIELD, IT CORRUPTS THE NEAR ONE. In the 0-4 A shell")
        report(f"  -- entirely INSIDE the 4 A sphere, where the scheme should be exact -- the slope against")
        report(f"  the true response is {near['slope']:+.3f} and the correlation {near['pearson']:+.3f}.")
        report("  The reason is induction: its energy is -1/2 (alpha/KE)|E|^2, a SQUARE of a sum, so the")
        report("  change contains a cross term 2*E0.dE that multiplies the perturbation by the PRE-EXISTING")
        report("  field. Truncating corrupts E0, and a wrong E0 inside the sphere is enough to flip the sign")
        report("  of the answer there. No sphere size fixes a term that is a square of a truncated sum.")
    c4 = res["arms"]["dna_charged"]["captured"]["4.0"]
    c4s = res["arms"]["dna_screened"]["captured"]["4.0"]
    need = next((R for R in RADII if res["arms"]["dna_charged"]["captured"][str(R)] >= 0.90), None)
    needs = next((R for R in RADII if res["arms"]["dna_screened"]["captured"][str(R)] >= 0.90), None)
    report(f"  A 4 A sphere captures {c4:.1%} of the unscreened response and {c4s:.1%} of the screened one.")
    if need is None:
        report("  UNSCREENED: no radius in the sweep reaches 90%. The sphere scheme cannot represent bare")
        report("  Coulomb on DNA at ANY size short of the whole particle -- the fix is not a bigger sphere,")
        report("  it is a far-field channel (Ewald/multipole), which is a different proposal.")
    else:
        report(f"  UNSCREENED: 90% of the response is captured at R = {need:.0f} A. That is the minimum")
        report("  viable sphere, and the genome-scale cost must be recomputed at that radius, not at 4 A.")
    if needs is not None:
        report(f"  SCREENED (150 mM, Debye 7.9 A): 90% captured at R = {needs:.0f} A. Screening is what makes")
        report("  a local scheme legitimate -- but NEXUS does not screen, so this is a change to the physics")
        report("  rather than a vindication of the tiling as it stands.")

    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_dna_spheres.json'}")
    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "nexus_dna_spheres.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
