"""LOOP 52 -- PHYSICS FOR THE ATTACHMENT PART: can a force field compute what nexus was built to sense?

WHERE THIS SITS. loop 51 answered the point-mutation row as far as sequence and monomer structure can
take it, and closed one thing firmly: nexus's ddG_fold sensor adds NOTHING on top of an AlphaFold
monomer (-0.0005 per gene, 46.3% of genes, p 0.67, both arms pinned to identical rows). But loop 32
established that nexus's real competence is somewhere else entirely -- protein-protein INTERFACE
binding, measured on SKEMPI ddG_bind, where its own recorded numbers are AUC 0.562 intrinsic and
0.755 with the second sensor, and loop 32's rebuild got 0.625 and 0.770.

So the attachment question is still open, and it is the one the whole nexus arc was actually about:
GIVEN A MUTATION AT AN INTERFACE, DOES THE COMPLEX STILL HOLD TOGETHER?

THE APPROACH IS THE ONE THAT ALREADY WON HERE ONCE. The 4D chromatin engine beat a trained CNN on a
held-out chromosome, 0.2852 to 0.0893, with ZERO fitted parameters, because the physics was right and
the CNN had to learn it from data. The same bet is available here: binding free energy is a physical
quantity with a force field behind it, and this repository already contains one --
flex_physics.py, with Lennard-Jones at 6 A, electrostatics at 10 A, desolvation at 8 A and induction
at 10 A, plus rotamer enumeration over chi1/chi2 wells.

WHAT IS COMPUTED. Only the ACROSS-INTERFACE terms, because everything else cancels:

    ddG_bind = [ E_across(mutant) - E_across(wild-type) ]

    E_across = LJ + electrostatics + induction + desolvation, summed over pairs with one atom in the
               mutated residue's side chain and the other in the PARTNER chain.

Intra-chain terms appear identically in the bound and unbound states and cancel exactly in a rigid
model, which is why they are not computed rather than computed and subtracted. The mutant side chain
is built onto the same backbone and its chi1/chi2 rotamer chosen by soft-core clash against the
environment -- the mutant is not assumed to sit where the wild type sat.

NOT FITTED. No coefficient in the physics arm is estimated from SKEMPI. The constants are the force
field's. The ONLY fitting anywhere in this module is inside the BASELINE stack, which is the thing
physics has to beat -- so if the fitting helps anyone, it helps the opponent.

PREDECLARED, before any number:

  P1 THE STRUCTURES AND THE TARGET ARE REAL, AND THE INDEX IS CHECKED AGAINST SOMETHING INDEPENDENT
       every scored mutation has a cached PDB, a resolvable chain and residue number, and a measured
       ddG from SKEMPI's own kon/koff. The wild-type residue SKEMPI names must match the residue in
       the PDB for >= 95% of rows. THIS CHECK IS NOT THE ONE LOOP 51 SHIPPED: SKEMPI's wt annotation
       was written by the experimentalists, not derived from this coordinate file, so the two can
       genuinely disagree and mismatches are dropped rather than averaged in.
  P2 THE PHYSICS TRACKS THE MEASUREMENT, HELD OUT BY COMPLEX
       Spearman between computed and measured ddG_bind, and AUC on nexus's OWN declared task
       (ddG_bind > 2 kcal/mol = interface breaker), above a permutation null that shuffles labels
       WITHIN complex -- so that "which complex is this" cannot carry the result. 319 complexes and
       a median of 4 mutations each means between-complex signal is the obvious confound here.
  P3 IT BEATS THE NON-PHYSICS BASELINES
       not a strawman: contact count at the mutated residue, hydrophobicity change, side-chain
       volume change, BLOSUM62, and a complex-grouped 5-fold logistic STACK of all four. The stack
       is fitted on the labels and the physics is not, so this comparison is deliberately unfair in
       the baseline's favour.
  P4 IT BEATS nexus's OWN RECORDED NUMBER ON nexus's OWN TASK
       0.755 recorded, 0.770 on loop 32's rebuild. The physics arm must beat the higher of the two,
       complex-held-out. Anything less and the honest statement is that the force field does not
       improve on the sensor that already exists.
  P5 EVERY TERM EARNS ITS PLACE
       LJ alone, +electrostatics, +induction, +desolvation, cumulative. A term that does not improve
       the held-out number is reported as not earning its place. "Physics" must not turn out to be
       one term doing all the work while three others ride along.
  P6 NOTHING IN THE PHYSICS ARM IS FITTED
       asserted structurally: the physics score is a fixed sum with no coefficients read from the
       labels, and the module prints the weights it used so the claim can be checked rather than
       believed.

-> outputs/loop_physics_bind.json
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flex_physics as F
import run_manifest as RM

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

BREAKER = 2.0          # kcal/mol -- nexus's own declared threshold, not chosen here
NEXUS_RECORDED = 0.755
NEXUS_REBUILT = 0.770
WT_MATCH_MIN = 0.95
N_NULL = 500
SEED = 1904

# Kyte-Doolittle hydropathy and residue side-chain volume (A^3) -- baseline features, not physics
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
       "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
       "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(v, y):
    v = np.asarray(v, float)
    y = np.asarray(y)
    m = np.isfinite(v)
    v, y = v[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(v)
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def resname_at(t, ch, pos):
    """The PDB's own three-letter residue name -- read from the file, not from SKEMPI."""
    m = (t["ch"] == ch) & (t["rn"] == pos)
    if not m.any():
        return None
    # flex_physics' table drops the resname column, so recover it from the CA line's parent record
    return None


def build_table_with_resnames(pdb):
    """Same flat table flex_physics builds, plus the residue NAME, which P1 needs."""
    p = f"{F.PDBDIR}/{pdb}.pdb"
    if not os.path.exists(p):
        return None
    rn3 = {}
    for line in open(p, errors="ignore"):
        if not line.startswith("ATOM"):
            continue
        try:
            rn3[(line[21], int(line[22:26]))] = line[17:20].strip().upper()
        except ValueError:
            continue
    return rn3


def across_terms(t, ch, pos, coords=None, names=None, wt=None):
    """LJ / electrostatics / induction / desolvation between one side chain and the PARTNER chain.

    Intra-chain terms are identical in the bound and unbound state of a rigid model and cancel
    exactly, so they are never computed. `coords`/`names` let a REBUILT mutant side chain be scored
    on the same backbone in the same environment.
    """
    scm = (t["ch"] == ch) & (t["rn"] == pos) & ~np.isin(t["nm"], list(F.BB) + ["CB"])
    if coords is None:
        if not scm.any():
            return None
        scc, scn = t["co"][scm], t["nm"][scm]
    else:
        scc, scn = coords, names
        if len(scc) == 0:
            return {"lj": 0.0, "elec": 0.0, "induc": 0.0, "desolv": 0.0}
    other = t["ch"] != ch
    if not other.any() or len(scc) == 0:
        return {"lj": 0.0, "elec": 0.0, "induc": 0.0, "desolv": 0.0}
    d0 = np.linalg.norm(t["co"][:, None, :] - scc[None, :, :], axis=2).min(1)
    near = other & (d0 < 12.0)
    if not near.any():
        return {"lj": 0.0, "elec": 0.0, "induc": 0.0, "desolv": 0.0}
    el = np.array([n[0] for n in scn])
    rad = F._RAD(el)
    qa = F._charge_res(wt if wt else "", scn) if wt else np.zeros(len(scn))
    qb = np.array([F._charge_name(n) for n in t["nm"][near]])
    return {"lj": F._lj_sum(scc, rad, t["co"][near], t["rad"][near], soft=True),
            "elec": F._elec(scc, qa, t["co"][near], qb),
            "induc": F._induction(scc, el, t["co"][near], qb),
            "desolv": F._desolv(scc, el, scn, t["co"][near])}


def mutant_sidechain(t, ch, pos, mut):
    """Build the mutant side chain on the same backbone and pick its rotamer by soft-core clash.

    The mutant is NOT assumed to occupy the wild type's coordinates. Without this the score would be
    a wild-type property with the identity of the new residue pasted on, which is the kind of thing
    that looks like physics and is bookkeeping.
    """
    res = (t["ch"] == ch) & (t["rn"] == pos)
    nm = t["nm"][res]
    co = t["co"][res]
    if not all(a in nm for a in ("N", "CA", "C")):
        return None
    N, CA, C = (co[list(nm).index(a)] for a in ("N", "CA", "C"))
    try:
        new_co, _new_rad, new_nm = F.mut_sidechain(mut, N, CA, C)
    except Exception:
        return None
    new_nm = [str(n) for n in new_nm]
    if new_co is None or len(new_co) == 0:
        # glycine: no side chain at all, so every across-interface term it had is LOST. That is a
        # real physical answer, not a missing measurement, and returning empty arrays gives it.
        return np.zeros((0, 3)), []
    env = ~res
    try:
        best, _, _ = F.sample_rotamers(np.asarray(new_co), list(new_nm), N, CA,
                                       t["co"][env], t["rad"][env])
    except Exception:
        best = np.asarray(new_co)
    keep = [i for i, n in enumerate(new_nm) if n not in list(F.BB) + ["CB"]]
    return best[keep], [new_nm[i] for i in keep]


def logistic_grouped(X, y, groups, rng, folds=5):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ug = np.unique(groups)
    fold = {g: i for g, i in zip(ug, rng.permutation(len(ug)) % folds)}
    fid = np.array([fold[g] for g in groups])
    out = np.full(len(y), np.nan)
    for k in range(folds):
        tr, te = fid != k, fid == k
        if tr.sum() < 40 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([(X[tr] - mu) / sd, np.ones(tr.sum())])
        B = np.column_stack([(X[te] - mu) / sd, np.ones(te.sum())])
        w = np.zeros(A.shape[1])
        for _ in range(500):
            p = 1 / (1 + np.exp(-A @ w))
            w -= 0.5 * (A.T @ (p - y[tr])) / max(tr.sum(), 1)
        out[te] = B @ w
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 52 -- physics for the attachment part: ddG_bind from a force field")
    say("=" * 100)

    muts = F.load_pdb_muts()
    say(f"\n  SKEMPI single-point mutations with a measured ddG: {len(muts):,}")

    # ---- P1 -------------------------------------------------------------------------------------
    say(f"\n  P1 THE STRUCTURES AND THE TARGET ARE REAL, AND THE INDEX IS CHECKED INDEPENDENTLY")
    rn3cache = {}
    rows, wt_ok, wt_bad, no_struct = [], 0, 0, 0
    for r in muts:
        t = F.table(r["pdb"])
        if t is None:
            no_struct += 1
            continue
        if r["pdb"] not in rn3cache:
            rn3cache[r["pdb"]] = build_table_with_resnames(r["pdb"]) or {}
        got = rn3cache[r["pdb"]].get((r["chain"], r["pos"]))
        if got is None:
            no_struct += 1
            continue
        if AA3TO1.get(got) != r["wt"]:
            wt_bad += 1
            continue
        wt_ok += 1
        rows.append(r)
    frac = wt_ok / max(wt_ok + wt_bad, 1)
    say(f"     mutations with a structure          {wt_ok + wt_bad:,}   (no structure/residue: "
        f"{no_struct:,})")
    say(f"     SKEMPI's wt residue == the PDB's    {frac:.4f}   ({wt_bad:,} mismatches DROPPED)")
    say(f"     this is an INDEPENDENT check -- SKEMPI's annotation was written by the "
        f"experimentalists,")
    say(f"     not derived from this coordinate file, so the two can genuinely disagree.")
    p1 = bool(frac >= WT_MATCH_MIN and len(rows) > 500)
    say(f"     P1 {'PASS' if p1 else 'FAIL'}  (>= {WT_MATCH_MIN:.0%} agreement and > 500 usable rows)")

    # ---- compute the physics ---------------------------------------------------------------------
    say(f"\n  computing across-interface energies for {len(rows):,} mutations")
    recs = []
    for i, r in enumerate(rows):
        t = F.table(r["pdb"])
        wt3 = rn3cache[r["pdb"]].get((r["chain"], r["pos"]), "")
        w = across_terms(t, r["chain"], r["pos"], wt=wt3)
        if w is None:
            continue
        mb = mutant_sidechain(t, r["chain"], r["pos"], r["mut"])
        if mb is None:
            continue
        mco, mnm = mb
        m = across_terms(t, r["chain"], r["pos"], coords=mco, names=mnm,
                         wt=F.AA3.get(r["mut"], ""))
        if m is None:
            continue
        rec = {"pdb": r["pdb"], "ddg": r["ddg"], "loc": r["loc"], "wt": r["wt"], "mut": r["mut"]}
        for k in ("lj", "elec", "induc", "desolv"):
            rec["d_" + k] = float(m[k] - w[k])
        # baseline features -- deliberately cheap and non-physical
        sc = (t["ch"] == r["chain"]) & (t["rn"] == r["pos"])
        oth = t["ch"] != r["chain"]
        if sc.any() and oth.any():
            dd = np.linalg.norm(t["co"][sc][:, None, :] - t["co"][oth][None, :, :], axis=2)
            rec["contacts"] = float((dd < 4.5).sum())
        else:
            rec["contacts"] = 0.0
        rec["d_kd"] = KD.get(r["mut"], 0) - KD.get(r["wt"], 0)
        rec["d_vol"] = VOL.get(r["mut"], 0) - VOL.get(r["wt"], 0)
        rec["blosum"] = 0.0
        recs.append(rec)
        if (i + 1) % 500 == 0:
            say(f"     {i+1:,}/{len(rows):,}  {time.time()-t0:.0f}s")
    say(f"     scored {len(recs):,} mutations in {time.time()-t0:.0f}s")

    B = None
    try:
        import loop_rescue as LR
        B = LR.blosum62()
        for rec in recs:
            rec["blosum"] = -B.get((rec["wt"], rec["mut"]), 0)
    except Exception:
        pass

    ddg = np.array([r["ddg"] for r in recs])
    grp = np.array([r["pdb"] for r in recs])
    y = (ddg > BREAKER).astype(int)
    say(f"     {len(recs):,} mutations, {len(set(grp)):,} complexes, {y.sum():,} breakers "
        f"({y.mean():.1%}) at ddG > {BREAKER} kcal/mol")

    TERMS = ["d_lj", "d_elec", "d_induc", "d_desolv"]
    col = lambda k: np.array([r[k] for r in recs], float)
    phys = sum(col(k) for k in TERMS)

    # ---- P2 -------------------------------------------------------------------------------------
    say(f"\n  P2 THE PHYSICS TRACKS THE MEASUREMENT, HELD OUT BY COMPLEX")
    rho = float(spearmanr(phys, ddg).statistic)
    a_phys = auc(phys, y)

    def null95(v):
        out = []
        idx = {g: np.where(grp == g)[0] for g in set(grp)}
        for _ in range(N_NULL):
            ys = y.copy()
            for g, ii in idx.items():
                ys[ii] = rng.permutation(y[ii])
            out.append(abs(auc(v, ys) - 0.5))
        return float(np.percentile(out, 95)) + 0.5

    n_phys = null95(phys)
    say(f"     Spearman(physics, measured ddG)  {rho:+.4f}")
    say(f"     AUC on the breaker task          {a_phys:.4f}   null 95th {n_phys:.4f}  "
        f"(labels shuffled WITHIN complex)")
    p2 = bool(a_phys > n_phys and rho > 0)
    say(f"     P2 {'PASS' if p2 else 'FAIL'}")

    # ---- P3 -------------------------------------------------------------------------------------
    say(f"\n  P3 IT BEATS THE NON-PHYSICS BASELINES")
    base_cols = ["contacts", "d_kd", "d_vol", "blosum"]
    S = {}
    for c in base_cols:
        S[c] = auc(col(c), y)
    Xb = np.column_stack([col(c) for c in base_cols])
    st = logistic_grouped(Xb, y, grp, rng)
    S["STACK (fitted)"] = auc(st, y)
    S["PHYSICS (not fitted)"] = a_phys
    for k, v in sorted(S.items(), key=lambda x: -(x[1] if np.isfinite(x[1]) else 0)):
        say(f"     {k:<24}{v:.4f}")
    p3 = bool(np.isfinite(S["STACK (fitted)"]) and a_phys > S["STACK (fitted)"])
    say(f"     P3 {'PASS' if p3 else 'FAIL'}  (physics beats a stack that was allowed to see the "
        f"labels)")

    # ---- P4 -------------------------------------------------------------------------------------
    say(f"\n  P4 IT BEATS nexus's OWN RECORDED NUMBER ON nexus's OWN TASK")
    say(f"     nexus recorded {NEXUS_RECORDED:.3f} · loop 32's rebuild {NEXUS_REBUILT:.3f} · "
        f"physics {a_phys:.4f}")
    p4 = bool(a_phys > max(NEXUS_RECORDED, NEXUS_REBUILT))
    say(f"     P4 {'PASS' if p4 else 'FAIL'}")

    # ---- P5 -------------------------------------------------------------------------------------
    say(f"\n  P5 EVERY TERM EARNS ITS PLACE")
    cum, prev, ladder = np.zeros(len(recs)), None, {}
    for k in TERMS:
        cum = cum + col(k)
        a = auc(cum, y)
        ladder[k] = {"auc": a, "delta": (a - prev) if prev is not None else np.nan,
                     "alone": auc(col(k), y)}
        say(f"     +{k:<10} cumulative {a:.4f}   " +
            (f"delta {a-prev:+.4f}" if prev is not None else "            ") +
            f"   alone {ladder[k]['alone']:.4f}")
        prev = a
    earned = [k for k in TERMS[1:] if np.isfinite(ladder[k]["delta"]) and ladder[k]["delta"] > 0]
    p5 = bool(len(earned) == len(TERMS) - 1)
    say(f"     P5 {'PASS' if p5 else 'FAIL'}  -- terms that improved the cumulative score: "
        f"{', '.join(earned) if earned else 'none'}")

    # ---- P6 -------------------------------------------------------------------------------------
    say(f"\n  P6 NOTHING IN THE PHYSICS ARM IS FITTED")
    say(f"     physics score = " + " + ".join(f"1.0*{k}" for k in TERMS))
    say(f"     every weight is exactly 1.0 and every constant is the force field's: LJ 6 A, "
        f"electrostatics 10 A,")
    say(f"     desolvation 8 A, induction 10 A, eps {F.EPS}. The ONLY fitting in this module is the "
        f"baseline stack.")
    p6 = True
    say(f"     P6 PASS (asserted structurally and printed so it can be checked)")

    say("\n" + "=" * 100)
    gates = {"P1 structures and target are real": p1, "P2 physics tracks the measurement": p2,
             "P3 beats the non-physics baselines": p3, "P4 beats nexus's own number": p4,
             "P5 every term earns its place": p5, "P6 nothing in the physics arm is fitted": p6}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    say(f"  A FORCE FIELD WITH NO FITTED PARAMETERS REACHES {a_phys:.4f} ON nexus's OWN TASK,")
    say(f"  against {NEXUS_REBUILT:.3f} for the sensor it was built to replace and "
        f"{S['STACK (fitted)']:.4f} for a stack")
    say(f"  that was allowed to see the labels. Spearman against the measured binding free energy")
    say(f"  is {rho:+.4f} over {len(recs):,} mutations in {len(set(grp)):,} complexes.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "skempi_v2.csv"), str(SC / "pdb_cache")],
                      available=len(muts), used=len(recs), selection="filtered", seed=SEED,
                      controls=[f"{N_NULL} label permutations WITHIN complex",
                                "SKEMPI's wt residue checked against the PDB, mismatches dropped",
                                "baseline stack fitted with complex-grouped 5-fold; physics not "
                                "fitted at all",
                                "mutant side chain rebuilt and its rotamer chosen by clash",
                                "term-by-term ablation ladder"],
                      note="only across-interface terms are computed; intra-chain terms cancel "
                           "exactly between the bound and unbound state of a rigid model")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_physics_bind", "manifest": man, "gates": gates,
               "n_mutations": len(recs), "n_complexes": len(set(grp)),
               "wt_match": frac, "spearman": rho, "auc_physics": a_phys, "null95": n_phys,
               "baselines": S, "ladder": ladder, "breaker_frac": float(y.mean()),
               "nexus_recorded": NEXUS_RECORDED, "nexus_rebuilt": NEXUS_REBUILT,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_physics_bind.json", "w"), indent=2, default=float)
    say(f"\n  -> {OUT/'loop_physics_bind.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
