"""ENERGY PER BOND AND PER ATOM -- because the compartment totals hid the only interesting question.

WHAT THE AGGREGATE MISSED.  `chromatin_twist_energy` reported strain in three buckets: DNA, histone, and
the DNA-histone interface. A methyl put 99.94% of its energy in DNA-DNA springs and 0.04% at the interface,
which reads as "marks do not touch the grip". But that is a total over 264,000 springs, and a total cannot
distinguish two completely different structures:

    energy spread thinly over every DNA bond          -> the mark is mechanically diffuse, nothing to see
    energy concentrated in a handful of bonds         -> the mark strains specific chemistry, and WHICH
                                                         bonds is the mechanism

Those have identical bucket totals. Worse, a 0.04% interface share is entirely compatible with the single
MOST strained bond in the structure being an interface contact -- 264,000 springs means a share that small
is still hundreds of times the energy of a typical bond. Share and tail are different questions and only
the tail carries mechanism.

SO THIS RESOLVES EVERY SPRING AND EVERY ATOM.

    per bond   0.5 k (e . (u_i - u_j))^2 for each spring, kept individually rather than summed
    per atom   half of each incident bond's energy assigned to each end, the standard partition

AND ASKS THE QUESTIONS A TOTAL CANNOT
    concentration   what fraction of the strain sits in the top 1% and top 10% of bonds. If a mark's energy
                    lives in a few hundred bonds out of 264,000, the mark is a local chemical event with a
                    definite location, and that location is reportable.
    the tail test   among the most-strained bonds, what fraction are DNA-histone contacts, against the
                    5.19% they are of the network. THIS is the accessibility question asked properly:
                    not "do interface bonds hold most of the energy" (they cannot, there are few of them)
                    but "are they over-represented where the strain actually concentrates".
    named bonds     the top bonds printed with their residues and atom names, so the answer is a chemical
                    statement -- which contact, on which base, to which histone residue -- rather than a
                    number. This is the form a wet-lab follow-up can act on.
    covalent vs non-bonded   springs under 1.8 A are stiffened 20x to keep the chain a chain, so they store
                    more energy at equal strain. Reported separately, because a mark that strains covalent
                    bonds and one that strains contacts are different physical claims and mixing them would
                    let stiffness masquerade as mechanism.

CONTROLS
    shuffled_force  the same force magnitudes on random atoms. Concentration is partly a property of ANY
                    point perturbation in a network, so the mark must beat this to have shown anything.
    two marks       a DNA mark and a histone mark. If the top-bond lists are indistinguishable, the
                    analysis is reading the network's stiffness pattern rather than the mark.

PREDECLARED, before any number:
    interface bonds over-represented in the top-strained tail, and more so than for shuffled forces
        -> marks DO selectively load the DNA-histone grip, the aggregate was hiding it, and the
           accessibility bridge is back on the table with a named set of contacts to test.
    interface bonds at or below their network share in the tail
        -> the negative from the aggregate survives at full resolution, which is a much stronger negative
           than the aggregate alone could support, and the bridge should be abandoned rather than retried.
    energy as concentrated for shuffled forces as for real marks
        -> concentration is a property of the network, not of the mark, and none of this is about chemistry.

-> outputs/orphan/chromatin_bond_energy.json  and  outputs/orphan/bond_energy_top.tsv
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load, VDW  # noqa: E402
from nexus_methyl_propagate import (elastic_network, rigid_basis, exact_solve, RC,  # noqa: E402
                                    K_BOND, K_NB, methyl_position, methyl_forces)
from chromatin_mark_cost import charges_of, charge_forces, DEBYE  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SEED = 127
TOPN = 200
TAIL_FRACS = (0.001, 0.01, 0.10)


def all_bonds(co, rc=RC):
    """Every spring in the network, as arrays. Same rule as `elastic_network` -- if these two ever disagree
    the energies describe a different model from the one that was solved."""
    n = len(co)
    I, J, KK, EV = [], [], [], []
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
        I.append(gi)
        J.append(jj)
        KK.append(np.where(rr < 1.8, K_BOND, K_NB))
        EV.append(d[ii, jj] / rr[:, None])
    return (np.concatenate(I), np.concatenate(J), np.concatenate(KK), np.concatenate(EV))


def bond_energy(u, I, J, KK, EV, n):
    U = u.reshape(n, 3)
    el = (EV * (U[I] - U[J])).sum(1)
    return 0.5 * KK * el ** 2


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("ENERGY PER BOND AND PER ATOM")
    report("=" * 100)
    report("  The compartment totals said a methyl puts 0.04% of its strain at the DNA-histone interface,")
    report("  which reads as 'marks do not touch the grip'. But that is a total over 264,000 springs, and")
    report("  a total cannot tell energy spread thinly over every bond from energy concentrated in a few.")
    report("  Those have identical bucket sums and completely different meanings. A 0.04% share is also")
    report("  entirely compatible with the single most-strained bond being an interface contact.")

    t = load(fetch_pdb(), keep_water=False)
    co, nm, rs, ch = t["co"], t["nm"], t["rs"], t["ch"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    q = charges_of(t)
    I, J, KK, EV = all_bonds(co)
    dna = t["dna"]
    iface = dna[I] != dna[J]
    coval = KK > (K_NB + 1e-9)
    report(f"\n  {n} atoms, {len(I)} springs | interface {iface.mean():.2%} of springs | "
           f"covalent {coval.mean():.2%}")

    dc5 = [i for i in range(n) if rs[i] == "DC" and nm[i] == "C5"]
    i_c5 = dc5[len(dc5) // 2]
    same = np.linalg.norm(co - co[i_c5], axis=1) < 3.0
    imap = {nm[j]: j for j in np.where(same)[0]}
    imap["C5"] = i_c5
    lysnz = [i for i in range(n) if rs[i] == "LYS" and nm[i] == "NZ"]
    i_nz = lysnz[len(lysnz) // 2]
    rng = np.random.default_rng(SEED)
    f_me = methyl_forces(co, rad, methyl_position(co, imap), i_c5)
    f_ac = charge_forces(co, q, i_nz, -0.80, debye=DEBYE)
    fs = np.zeros_like(f_me)
    hit = np.where(np.linalg.norm(f_me, axis=1) > 0)[0]
    fs[rng.choice(n, size=len(hit), replace=False)] = f_me[hit]
    fs -= fs.mean(0)
    MARKS = [("5mC", f_me, i_c5), ("acetyl", f_ac, i_nz), ("shuffled", fs, i_c5)]

    res = {"n_atoms": n, "n_bonds": int(len(I)), "iface_share": float(iface.mean()), "marks": {}}
    rows_out = []
    report("\n  CONCENTRATION -- what fraction of the total strain sits in the most-strained bonds")
    report(f"    {'mark':<12}{'top 0.1%':>10}{'top 1%':>10}{'top 10%':>10}{'bonds for 90%':>16}")
    store = {}
    for name, f, site in MARKS:
        u, info, resid = exact_solve(K, Q, f)
        if info != 0 or resid > 1e-6:
            report(f"    {name}: solve failed; skipped")
            continue
        E = bond_energy(u, I, J, KK, EV, n)
        store[name] = (u, E)
        tot = E.sum()
        o = np.argsort(E)[::-1]
        cs = np.cumsum(E[o]) / max(tot, 1e-30)
        fr = [float(cs[max(int(len(E) * x) - 1, 0)]) for x in TAIL_FRACS]
        n90 = int(np.searchsorted(cs, 0.90) + 1)
        report(f"    {name:<12}{fr[0]:>10.3f}{fr[1]:>10.3f}{fr[2]:>10.3f}{n90:>16,}")
        res["marks"][name] = {"top_fracs": dict(zip([str(x) for x in TAIL_FRACS], fr)),
                              "bonds_for_90pct": n90, "total_energy": float(tot)}

    report("\n  THE TAIL TEST -- are DNA-histone contacts OVER-REPRESENTED where strain concentrates?")
    report(f"    Interface bonds are {iface.mean():.2%} of the network. A ratio above 1 means the mark")
    report("    selectively loads the grip; at or below 1 the aggregate negative survives at full")
    report("    resolution, which is a far stronger negative than the bucket totals could support.")
    report(f"    {'mark':<12}{'top 100':>10}{'top 1000':>10}{'top 1%':>10}{'ratio @1%':>12}")
    for name, (u, E) in store.items():
        o = np.argsort(E)[::-1]
        r100 = float(iface[o[:100]].mean())
        r1000 = float(iface[o[:1000]].mean())
        k1 = max(int(len(E) * 0.01), 1)
        r1 = float(iface[o[:k1]].mean())
        report(f"    {name:<12}{r100:>10.3f}{r1000:>10.3f}{r1:>10.3f}"
               f"{r1/max(iface.mean(),1e-9):>12.2f}")
        res["marks"][name].update({"iface_top100": r100, "iface_top1000": r1000,
                                   "iface_top1pct": r1,
                                   "iface_ratio_top1pct": r1 / max(float(iface.mean()), 1e-9)})

    report("\n  COVALENT vs NON-BONDED share of the strain (covalent springs are stiffened 20x, so equal")
    report("  strain stores more energy there; separated so stiffness cannot masquerade as mechanism)")
    report(f"    {'mark':<12}{'covalent %':>12}{'nonbonded %':>13}")
    for name, (u, E) in store.items():
        tot = E.sum() or 1.0
        report(f"    {name:<12}{100*E[coval].sum()/tot:>12.2f}{100*E[~coval].sum()/tot:>13.2f}")
        res["marks"][name]["covalent_frac"] = float(E[coval].sum() / tot)

    # --- per-atom energy ---------------------------------------------------------------------------------
    report("\n  PER-ATOM STRAIN, decay with distance from the mark (mean energy per atom in each shell)")
    SH = ((0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 40), (40, 70))
    report(f"    {'mark':<12}" + "".join(f"{f'{a}-{b}A':>11}" for a, b in SH))
    for name, (u, E) in store.items():
        A = np.zeros(n)
        np.add.at(A, I, 0.5 * E)
        np.add.at(A, J, 0.5 * E)
        site = dict((m[0], m[2]) for m in MARKS)[name]
        d = np.linalg.norm(co - co[site], axis=1)
        row = []
        for a, b in SH:
            m = (d >= a) & (d < b)
            row.append(float(A[m].mean()) if m.sum() else float("nan"))
        report(f"    {name:<12}" + "".join(f"{v:>11.3e}" for v in row))
        res["marks"][name]["atom_shell_energy"] = row
        if name == "5mC":
            np.save(OUT / "bond_energy_per_atom_5mC.npy", A)

    # --- named top bonds, so the answer is chemistry ------------------------------------------------------
    def lab(i):
        return f"{rs[i]}{'' if not ch[i] else ':' + ch[i]}.{nm[i]}"
    report(f"\n  THE {TOPN} MOST-STRAINED BONDS for 5mC are written to bond_energy_top.tsv, with residues")
    report("  and atom names, so the result is a chemical statement rather than a number.")
    if "5mC" in store:
        u, E = store["5mC"]
        o = np.argsort(E)[::-1][:TOPN]
        with open(OUT / "bond_energy_top.tsv", "w") as fh:
            fh.write("rank\tenergy\tatom_i\tatom_j\ttype\tis_interface\tdist_from_mark_A\n")
            for k, b in enumerate(o):
                i_, j_ = int(I[b]), int(J[b])
                dm = float(min(np.linalg.norm(co[i_] - co[i_c5]), np.linalg.norm(co[j_] - co[i_c5])))
                fh.write(f"{k+1}\t{E[b]:.6g}\t{lab(i_)}\t{lab(j_)}\t"
                         f"{'covalent' if coval[b] else 'contact'}\t{bool(iface[b])}\t{dm:.1f}\n")
        report("    top 5:")
        for k, b in enumerate(o[:5]):
            i_, j_ = int(I[b]), int(J[b])
            report(f"      {k+1}. {lab(i_):<18} -- {lab(j_):<18} {E[b]:.4g}  "
                   f"{'covalent' if coval[b] else 'contact'}{'  INTERFACE' if iface[b] else ''}")

    report("\n  READING")
    for name in ("5mC", "acetyl"):
        if name not in res["marks"]:
            continue
        d = res["marks"][name]
        sh = res["marks"].get("shuffled", {})
        report(f"    {name}: top 1% of bonds hold {d['top_fracs']['0.01']:.1%} of the strain "
               f"(shuffled {sh.get('top_fracs',{}).get('0.01', float('nan')):.1%}); interface ratio in that "
               f"tail {d['iface_ratio_top1pct']:.2f}")
    r5 = res["marks"].get("5mC", {}).get("iface_ratio_top1pct", 0.0)
    rs_ = res["marks"].get("shuffled", {}).get("iface_ratio_top1pct", 0.0)
    if r5 > 1.5 and r5 > 1.5 * rs_:
        report("    INTERFACE CONTACTS ARE OVER-REPRESENTED WHERE THE STRAIN CONCENTRATES, and more so than")
        report("    for a shuffled force. The aggregate was hiding a real effect: marks do load the grip,")
        report("    just not with much of their TOTAL energy. The named bonds are the testable claim.")
    else:
        report("    INTERFACE CONTACTS ARE NOT OVER-REPRESENTED IN THE TAIL EITHER. The negative from the")
        report("    bucket totals survives at full bond resolution, which is a much stronger result than")
        report("    the aggregate alone: it is not that the effect is small, it is that the strain does not")
        report("    reach the grip at any resolution. The accessibility bridge should be abandoned on this")
        report("    route rather than retried with a bigger model.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "chromatin_bond_energy.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_bond_energy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
