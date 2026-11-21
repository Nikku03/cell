"""DNA ROTATION AND WHERE THE ENERGY GOES -- two channels the displacement field does not show.

WHAT IS MISSING FROM WHAT HAS BEEN BUILT.  The mark machinery solves K u = f and reports displacement: how
far each atom moves. That is one projection of the response and it hides two things that matter more.

ROTATION.  DNA is a twisted rod, and twist is a degree of freedom that translation does not capture. It also
propagates on a completely different length scale: displacement from a methyl was measured to fall ~15x
within 15 A, but TORSIONAL stress in a rod travels along its length -- that is the mechanism of supercoiling,
and it is why transcribing polymerase generates positive supercoils kilobases ahead of itself. So the
question is not whether twist exists but whether it carries the mark's effect FURTHER than displacement did.
If it does, every reach number in this line so far was measured on the short-range channel.

ENERGY, AND SPECIFICALLY WHERE IT IS STORED.  Displacement says what moved. Elastic energy says what is
STRAINED, which is the quantity that couples to stability. A nucleosome holds DNA by roughly 14 contact
points, and unwrapping is a free-energy competition. If a mark's deformation energy localises at the
DNA-histone interface, the mark is mechanically coupled to how tightly that DNA is held -- which IS
accessibility, and would be the first measured bridge in this project from atomistic structure to a
phenotype-relevant quantity. If instead the energy spreads into the bulk, the mark is mechanically real and
biologically inert, and that is worth knowing before anything is wired into the cell model.

    displacement   how far things moved          measured already, short-ranged
    twist          how far things ROTATED        this file; expected longer-ranged
    energy         where the strain is STORED    this file; the interface fraction is the biology

HOW TWIST IS COMPUTED, and the approximation is stated rather than buried.  Base pairs are identified by
pairing C1' atoms across the two DNA strands (Watson-Crick C1'-C1' is ~10.5 A, which separates pairs
cleanly from non-pairs). For pair i, v_i is the C1'->C1' vector and c_i the pair centroid. The local helical
axis at step i is c_{i+1} - c_i, and the twist of that step is the angle between v_i and v_{i+1} measured in
the plane perpendicular to that axis. This is the standard construction reduced to the two atoms that define
it; it is not a full base-frame calculation (no propeller, roll or tilt) and no claim is made beyond twist.

Applying the solved displacement u to the coordinates and recomputing gives the CHANGE in twist per step,
which is then plotted against sequence separation from the mark -- in BASE PAIRS, because that is the axis
along which torsion travels, not Angstroms.

HOW ENERGY IS DECOMPOSED.  Every spring stores 0.5 k (e . (u_i - u_j))^2. Each is classified by what it
connects -- DNA-DNA, histone-histone, or the DNA-histone INTERFACE -- and the totals are reported as
fractions. The interface fraction is compared against the fraction of springs that ARE interface springs,
because a channel holding 10% of the energy while being 10% of the network has done nothing.

CONTROLS
    baseline_twist   the twist profile of the unperturbed structure, so a change is measured against the
                     real helix rather than against an idealised 10.5 bp/turn that this crystal need not obey
    shuffled_force   the same force magnitudes on randomly chosen atoms. If twist propagates the same way,
                     the propagation is a property of the rod and not of where the mark sits.
    energy_share     each compartment's share of the SPRINGS as well as of the energy. Only the ratio is
                     evidence; a raw fraction is not.
    two marks        5mC (steric, on the DNA itself) and acetyl (charge, on a histone tail). A DNA mark and
                     a protein mark should load the interface differently, and if they do not, the
                     decomposition is not sensitive to anything.

PREDECLARED, before any number:
    twist change extends over more base pairs than displacement extends in Angstroms
        -> rotation is a genuinely longer-range channel, and reach conclusions drawn from displacement
           alone understate how far a mark is felt.
    twist change is as local as displacement
        -> the rod picture does not apply at nucleosome scale, where DNA is clamped every ~10 bp by the
           histone core, and twist adds a degree of freedom but no new range.
    interface energy fraction exceeds the interface spring fraction
        -> marks load the DNA-histone contact preferentially, which is the mechanical route to accessibility
           and the first thing in this line that would justify connecting to the cell model.
    interface energy fraction tracks the spring fraction
        -> strain distributes with the network's own connectivity, the mark is not selectively loading the
           grip, and the accessibility bridge is not supported by this evidence.

-> outputs/orphan/chromatin_twist_energy.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load, VDW  # noqa: E402
from nexus_methyl_propagate import (elastic_network, rigid_basis, exact_solve, RC, K_BOND,  # noqa: E402
                                    K_NB, methyl_position, methyl_forces)
from chromatin_mark_cost import charges_of, charge_forces, DEBYE  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
PAIR_MIN, PAIR_MAX = 8.0, 13.0     # Watson-Crick C1'-C1' separation window, Angstrom
SEED = 113
BP_BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 80))


def base_pairs(t):
    """Pair C1' atoms across the two DNA strands. Geometry, not residue numbering, because the numbering
    convention of a PDB is not something to depend on."""
    dna = t["dna"]
    c1 = np.where(dna & (t["nm"] == "C1'"))[0]
    ch = t["ch"][c1]
    chains = sorted(set(ch))
    if len(chains) < 2:
        return []
    A = c1[ch == chains[0]]
    Bc = c1[ch == chains[1]]
    co = t["co"]
    pairs = []
    used = set()
    for a in A:
        d = np.linalg.norm(co[Bc] - co[a], axis=1)
        j = int(np.argmin(d))
        if PAIR_MIN < d[j] < PAIR_MAX and Bc[j] not in used:
            pairs.append((int(a), int(Bc[j])))
            used.add(Bc[j])
    return pairs


def twist_profile(co, pairs):
    """Twist per base-pair step, in degrees. See the module docstring for the construction and its limits."""
    v = np.array([co[b] - co[a] for a, b in pairs])
    c = np.array([0.5 * (co[a] + co[b]) for a, b in pairs])
    out = np.full(len(pairs) - 1, np.nan)
    for i in range(len(pairs) - 1):
        ax = c[i + 1] - c[i]
        na = np.linalg.norm(ax)
        if na < 1e-6:
            continue
        ax = ax / na
        p = v[i] - np.dot(v[i], ax) * ax
        q = v[i + 1] - np.dot(v[i + 1], ax) * ax
        np_, nq = np.linalg.norm(p), np.linalg.norm(q)
        if np_ < 1e-6 or nq < 1e-6:
            continue
        cth = np.clip(np.dot(p, q) / (np_ * nq), -1.0, 1.0)
        s = np.sign(np.dot(np.cross(p, q), ax))
        out[i] = np.degrees(np.arccos(cth)) * (1.0 if s >= 0 else -1.0)
    return out


def spring_energy(co, u, t, rc=RC):
    """Per-spring elastic energy 0.5 k (e.(u_i - u_j))^2, classified by what the spring connects.

    Rebuilt with the same rule as the Hessian so the classification cannot drift from the physics: the
    stiffness and cutoff here must match `elastic_network` exactly or the totals describe a different model.
    """
    n = len(co)
    U = u.reshape(n, 3)
    dna = t["dna"]
    tot = {"dna": 0.0, "histone": 0.0, "interface": 0.0}
    cnt = {"dna": 0, "histone": 0, "interface": 0}
    chunk = 1024
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = co[s:e, None, :] - co[None, :, :]
        r = np.linalg.norm(d, axis=2)
        ii, jj = np.where((r < rc) & (r > 1e-6))
        keep = jj > (ii + s)
        ii, jj = ii[keep], jj[keep]
        if not len(ii):
            continue
        gi = ii + s
        rr = r[ii, jj]
        ev = d[ii, jj] / rr[:, None]
        k = np.where(rr < 1.8, K_BOND, K_NB)
        du = U[gi] - U[jj]
        el = (ev * du).sum(1)
        en = 0.5 * k * el ** 2
        a, b = dna[gi], dna[jj]
        m_i = a & b
        m_h = (~a) & (~b)
        m_x = ~(m_i | m_h)
        for key, m in (("dna", m_i), ("histone", m_h), ("interface", m_x)):
            tot[key] += float(en[m].sum())
            cnt[key] += int(m.sum())
    return tot, cnt


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("DNA ROTATION AND WHERE THE ENERGY GOES")
    report("=" * 100)
    report("  Displacement is one projection of the response and it hides two things. TWIST is a degree of")
    report("  freedom translation does not capture, and it travels along a rod rather than through space --")
    report("  the mechanism of supercoiling -- so it may carry a mark much further than the ~15 A")
    report("  displacement reach measured earlier. ENERGY says what is STRAINED rather than what moved, and")
    report("  strain at the DNA-histone interface is mechanical coupling to how tightly the DNA is held,")
    report("  which is the closest thing to accessibility this line has produced.")

    t = load(fetch_pdb(), keep_water=False)
    co = t["co"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    q = charges_of(t)
    pairs = base_pairs(t)
    report(f"\n  1KX5 no waters: {n} atoms, {npair} springs | {len(pairs)} base pairs resolved by C1'-C1'")
    if len(pairs) < 20:
        report("  *** too few base pairs resolved; the twist arm cannot run")
        return 1

    tw0 = twist_profile(co, pairs)
    ok = np.isfinite(tw0)
    report(f"  baseline twist: {np.nanmean(tw0):.2f} deg/step (sd {np.nanstd(tw0):.2f}) over {ok.sum()} steps")
    report(f"  = {360.0/max(abs(np.nanmean(tw0)),1e-9):.1f} bp per turn, against ~10.5 for ideal B-DNA.")
    report("  Reported as the control it is: change is measured against THIS helix, not against the ideal.")

    # --- the marks ---------------------------------------------------------------------------------------
    dc5 = [i for i in range(n) if t["rs"][i] == "DC" and t["nm"][i] == "C5"]
    i_c5 = dc5[len(dc5) // 2]
    same = np.linalg.norm(co - co[i_c5], axis=1) < 3.0
    imap = {t["nm"][j]: j for j in np.where(same)[0]}
    imap["C5"] = i_c5
    lysnz = [i for i in range(n) if t["rs"][i] == "LYS" and t["nm"][i] == "NZ"]
    i_nz = lysnz[len(lysnz) // 2]
    rng = np.random.default_rng(SEED)
    f_me = methyl_forces(co, rad, methyl_position(co, imap), i_c5)
    f_ac = charge_forces(co, q, i_nz, -0.80, debye=DEBYE)
    fs = np.zeros_like(f_me)
    hit = np.where(np.linalg.norm(f_me, axis=1) > 0)[0]
    pick = rng.choice(n, size=len(hit), replace=False)
    fs[pick] = f_me[hit]
    fs -= fs.mean(0)
    MARKS = [("5mC      on DNA", f_me, i_c5), ("acetyl   on histone", f_ac, i_nz),
             ("shuffled control", fs, i_c5)]

    # where is the mark, in base-pair coordinates
    def nearest_bp(idx):
        cen = np.array([0.5 * (co[a] + co[b]) for a, b in pairs])
        return int(np.argmin(np.linalg.norm(cen - co[idx], axis=1)))

    res = {"n_pairs": len(pairs), "baseline_twist_mean": float(np.nanmean(tw0)), "marks": {}}
    report("\n  TWIST CHANGE by SEQUENCE separation from the mark (degrees, RMS over steps in each bin).")
    report("  Separation is in BASE PAIRS, not Angstroms, because torsion travels along the helix.")
    report("    " + f"{'mark':<22}" + "".join(f"{f'{a}-{b}bp':>10}" for a, b in BP_BINS))
    for name, f, site in MARKS:
        u, info, resid = exact_solve(K, Q, f)
        if info != 0 or resid > 1e-6:
            report(f"    {name:<22} exact solve failed; skipped")
            continue
        co2 = co + u.reshape(n, 3)
        tw1 = twist_profile(co2, pairs)
        dtw = tw1 - tw0
        b0 = nearest_bp(site)
        sep = np.abs(np.arange(len(dtw)) - b0)
        row, prof = [], []
        for lo, hi in BP_BINS:
            m = (sep >= lo) & (sep < hi) & np.isfinite(dtw)
            v = float(np.sqrt(np.nanmean(dtw[m] ** 2))) if m.sum() else float("nan")
            row.append(v)
            prof.append({"lo": lo, "hi": hi, "n": int(m.sum()), "rms_deg": v})
        report(f"    {name:<22}" + "".join(f"{v:>10.4f}" for v in row))
        en, cn = spring_energy(co, u, t)
        tote = sum(en.values()) or 1.0
        totc = sum(cn.values()) or 1
        res["marks"][name.split()[0]] = {"twist": prof, "bp_site": b0,
                                         "energy": en, "springs": cn,
                                         "energy_frac": {k: en[k] / tote for k in en},
                                         "spring_frac": {k: cn[k] / totc for k in cn}}

    report("\n  WHERE THE STRAIN IS STORED. `share` is the compartment's fraction of the springs -- only")
    report("  the RATIO is evidence, since a channel holding 20% of the energy while being 20% of the")
    report("  network has done nothing selective.")
    report("    " + f"{'mark':<22}{'compartment':<12}{'energy %':>10}{'springs %':>11}{'ratio':>8}")
    for nm, d in res["marks"].items():
        for c in ("dna", "histone", "interface"):
            ef, sf = d["energy_frac"][c], d["spring_frac"][c]
            report(f"    {nm:<22}{c:<12}{100*ef:>10.2f}{100*sf:>11.2f}{ef/max(sf,1e-9):>8.2f}")

    report("\n  READING")
    for nm, d in res["marks"].items():
        pr = [p["rms_deg"] for p in d["twist"]]
        near, far = pr[0], next((v for v in pr[::-1] if np.isfinite(v)), float("nan"))
        ratio = near / far if far and np.isfinite(far) and far > 0 else float("nan")
        report(f"    {nm}: twist falls {ratio:.1f}x from 0-2 bp to the farthest bin; interface energy "
               f"ratio {d['energy_frac']['interface']/max(d['spring_frac']['interface'],1e-9):.2f}")
    me = res["marks"].get("5mC")
    sh = res["marks"].get("shuffled")
    if me and sh:
        m_far = me["twist"][-1]["rms_deg"]
        s_far = sh["twist"][-1]["rms_deg"]
        if np.isfinite(m_far) and np.isfinite(s_far) and m_far > 2 * s_far:
            report("    TWIST CARRIES THE MARK FURTHER THAN THE SHUFFLED CONTROL, so rotation is a real")
            report("    channel and not just the rod responding to any push.")
        else:
            report("    TWIST DOES NOT BEAT THE SHUFFLED CONTROL AT RANGE: whatever rotation appears far")
            report("    from the mark is what the rod does for any force, not something the mark carries.")
        r_me = me["energy_frac"]["interface"] / max(me["spring_frac"]["interface"], 1e-9)
        if r_me > 1.5:
            report(f"    AND THE DNA MARK LOADS THE INTERFACE {r_me:.1f}x ITS SHARE OF THE NETWORK. Strain")
            report("    concentrates where the histone grips the DNA, which is the mechanical route to")
            report("    accessibility and the first thing here that would justify a cell-model connection.")
        else:
            report(f"    INTERFACE LOADING IS {r_me:.2f}x ITS SHARE -- strain distributes with the network's")
            report("    own connectivity rather than concentrating on the grip, so the accessibility bridge")
            report("    is NOT supported by this evidence and should not be assumed downstream.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "chromatin_twist_energy.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_twist_energy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
