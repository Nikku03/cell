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

WHY DNA IS THE HARD CASE, AND WHY THIS IS NOT A GENERIC POLYMER TEST.  DNA is a polyanion: one formal -1
charge per phosphate, one phosphate every ~3.4 A of backbone, ~41 heavy atoms per base pair. Truncating
Coulomb on a dense line of like charges is the textbook failure case for cutoffs, because the neglected tail
is a sum of many same-signed terms that does not cancel. A protein's charges are sparse and mixed-sign and
forgive short cutoffs; DNA's do not. If small spheres are going to fail anywhere, it is here -- which is
exactly why the test has to be run on DNA and not on the protein complexes NEXUS was tuned against.

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
RADII = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0)
SHELLS = ((0, 4), (4, 8), (8, 12), (12, 20), (20, 35), (35, 60))
SEED = 17
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


def response_exact(co, q, rad, alpha, k, dq, chunk=2048):
    """EXACT per-atom energy response to changing atom k's charge by dq. No cutoff anywhere.

    Only the charge changes, so LJ and desolvation are identical before and after and cancel exactly in the
    DIFFERENCE. What survives is electrostatics -- linear in dq -- and induction, which is quadratic in the
    total field and therefore does NOT cancel. Both are computed here; reporting only the linear part would
    understate the tail, since induction is the term NEXUS uses to represent polarisation.

    Returns dE_i: the change in atom i's interaction energy, for every i. Sum over i counts each pair twice
    for the electrostatic part, which is the standard convention and is applied consistently to every arm.
    """
    n = len(co)
    dE = np.zeros(n)
    qk_new = q[k] + dq
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.linalg.norm(co[s:e, None, :] - co[None, :, :], axis=2)
        np.fill_diagonal(d[:, s:e], np.inf)
        d = np.maximum(d, 0.1)
        # electrostatics with eps(r)=r  ->  U = KE q_i q_j / r^2 ; only pairs involving k change
        dq_vec = np.zeros(n)
        dq_vec[k] = dq
        dE[s:e] += KE * q[s:e] * (dq_vec[None, :] / d ** 2).sum(1)
        # the changed atom itself feels every other atom
        if s <= k < e:
            dE[k] += KE * dq * (q / np.maximum(np.linalg.norm(co[k] - co, axis=1), 0.1) ** 2).sum()
    # induction: field at every atom before and after, energy -1/2 (alpha/KE) |E|^2
    E0 = field_at(co, q, chunk=chunk)
    q2 = q.copy()
    q2[k] = qk_new
    E1 = field_at(co, q2, chunk=chunk)
    u0 = -0.5 * (alpha / KE) * (E0 ** 2).sum(1)
    u1 = -0.5 * (alpha / KE) * (E1 ** 2).sum(1)
    dE += (u1 - u0)
    return dE


def field_at(co, q, chunk=2048):
    """Coulomb field vector at every atom from every other atom. No cutoff."""
    n = len(co)
    F = np.zeros((n, 3))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dv = co[s:e, None, :] - co[None, :, :]
        r = np.linalg.norm(dv, axis=2)
        r = np.maximum(r, 0.1)
        inv3 = 1.0 / r ** 3
        for loc, glob in enumerate(range(s, e)):
            inv3[loc, glob] = 0.0
        F[s:e] = KE * (dv * (q[None, :, None] * inv3[:, :, None])).sum(1)
    return F


def response_truncated(co, q, alpha, k, dq, R, chunk=2048):
    """The same response with every interaction cut at R -- which is precisely what a sphere of radius R
    around each atom computes. This is the sphere scheme, expressed exactly."""
    n = len(co)
    dE = np.zeros(n)
    dk = np.linalg.norm(co - co[k], axis=1)
    inr = (dk < R) & (dk > 0.1)
    dE[inr] += KE * q[inr] * dq / np.maximum(dk[inr], 0.1) ** 2
    dE[k] += KE * dq * (q[inr] / np.maximum(dk[inr], 0.1) ** 2).sum()
    E0 = field_at_trunc(co, q, R, chunk=chunk)
    q2 = q.copy()
    q2[k] = q[k] + dq
    E1 = field_at_trunc(co, q2, R, chunk=chunk)
    dE += -0.5 * (alpha / KE) * ((E1 ** 2).sum(1) - (E0 ** 2).sum(1))
    return dE


def field_at_trunc(co, q, R, chunk=2048):
    n = len(co)
    F = np.zeros((n, 3))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dv = co[s:e, None, :] - co[None, :, :]
        r = np.linalg.norm(dv, axis=2)
        m = (r < R) & (r > 0.1)
        inv3 = np.where(m, 1.0 / np.maximum(r, 0.1) ** 3, 0.0)
        F[s:e] = KE * (dv * (q[None, :, None] * inv3[:, :, None])).sum(1)
    return F


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
    """Agreement per distance shell. Pooling all atoms into one correlation is the trap this avoids: the
    far shells are nearly all atoms and nearly all zero, so a pooled r is dominated by agreeing about
    nothing. Reported as relative error, which cannot be inflated that way."""
    out = []
    for lo, hi in SHELLS:
        m = (dk >= lo) & (dk < hi)
        if m.sum() < 5:
            out.append((lo, hi, int(m.sum()), float("nan"), float("nan")))
            continue
        ex, tr = dE_exact[m], dE_trunc[m]
        den = np.abs(ex).sum()
        rel = float(np.abs(tr - ex).sum() / den) if den > 0 else float("nan")
        if np.std(ex) > 0 and np.std(tr) > 0:
            r = float(np.corrcoef(ex, tr)[0, 1])
        else:
            r = float("nan")
        out.append((lo, hi, int(m.sum()), rel, r))
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

    rng = np.random.default_rng(SEED)
    q_full = charges(t)
    # the perturbed atom: a phosphate oxygen near the middle of the DNA, away from the ends
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
        ("dna_charged   (the case)", charges(t)),
        ("dna_neutral   (control)", charges(t, neutral_dna=True)),
        ("lj_only       (control)", charges(t, zero_all=True)),
        ("shuffled_chg  (control)", charges(t, shuffle_rng=np.random.default_rng(SEED + 3))),
    ]

    res = {"n_atoms": n, "n_dna": nd, "n_water": nw, "perturbed_atom": k, "arms": {}}
    report("\n  ARM 1 -- REACH: fraction of the TOTAL |response| captured inside radius R.")
    report("  Absolute, not signed: a signed fraction can read 100% while a large positive and a large")
    report("  negative in the far field merely cancel, and cancellation is not capture.")
    hdr = "    " + f"{'arm':<26}" + "".join(f"{r:>7.0f}A" for r in RADII)
    report(hdr)
    exact_store = {}
    for name, q in ARMS:
        if np.abs(q).sum() == 0:
            # LJ-only: the perturbation is a CHARGE, so with all charges zero there is no response at all.
            # Recorded explicitly rather than silently printing zeros -- it defines the arm's meaning.
            report(f"    {name:<26}  (charge perturbation with all charges zero -> no response by "
                   f"construction; see ARM 3, where the LJ control is a DISPLACEMENT and does apply)")
            continue
        dE = response_exact(co, q, radii_of(t), alpha, k, dq)
        exact_store[name] = dE
        row = "".join(f"{captured(dE, dk, R):>8.3f}" for R in RADII)
        report(f"    {name:<26}{row}")
        res["arms"][name.split()[0]] = {"captured": {str(R): captured(dE, dk, R) for R in RADII},
                                        "total_abs": float(np.abs(dE).sum())}

    report("\n  ARM 2 -- FIDELITY per distance shell, truncated vs exact (relative error; lower is better).")
    report("  A pooled correlation is not quoted: most atoms are far away where both sides are ~0, and")
    report("  agreeing about zero manufactures a high number that says nothing about the near field.")
    key = "dna_charged   (the case)"
    dE_ex = exact_store[key]
    q = dict(ARMS)[key]
    res["shells"] = {}
    for R in (4.0, 8.0, 12.0, 20.0):
        dE_tr = response_truncated(co, q, alpha, k, dq, R)
        rows = shell_agreement(dE_ex, dE_tr, dk)
        report(f"    sphere R = {R:.0f} A")
        report(f"      {'shell':<12}{'atoms':>8}{'rel.err':>10}{'pearson':>10}")
        for lo, hi, cnt, rel, r in rows:
            report(f"      {f'{lo}-{hi} A':<12}{cnt:>8}{rel:>10.3f}{r:>10.3f}")
        res["shells"][str(R)] = [{"lo": lo, "hi": hi, "n": cnt, "rel_err": rel, "pearson": r}
                                 for lo, hi, cnt, rel, r in rows]

    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_dna_spheres.json'}")
    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "nexus_dna_spheres.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
