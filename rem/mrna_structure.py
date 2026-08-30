"""Does local mRNA secondary structure predict ribosome occupancy along a transcript?

WHY THIS TEST AND NOT THE PREVIOUS ONE. M4 tested tRNA-abundance dwell and failed; M4b
localised the failure to the INPUT PHYSICS rather than the solver. A per-gene label such as
gene function or protein family cannot repair that, and not for want of signal: it is
CONSTANT within a gene, while the quantity being scored is the rank order of occupancy
ALONG a gene. Measured, the split of log P-site density variance is 26.4% between genes and
73.6% within, and adding any per-gene constant leaves the within-gene Spearman bit-identical.
So the only candidates that can move this test are ones that VARY along the axis. Local
folding energy is the cheapest such candidate that has not been tried here.

THE MECHANISM. The ribosome must unwind duplex mRNA entering its entry channel. A window
just downstream of the current codon that folds into a stable structure is harder to unwind,
so the ribosome dwells longer and occupancy is higher. Primary definition, fixed here:
    dG_i = MFE of the 30-nt window starting at the first nucleotide of codon i,
    folded with rem.rna's nearest-neighbour StackingModel.
More negative dG means more structure means longer dwell.

WHY THE RANK TEST NEEDS NO FREE PARAMETER. Spearman depends only on rank order, and any
monotone map of dG gives the same ranking. So the raw test -- does -dG rank-correlate with
P-site density -- has no scale, no temperature and no fitted constant in it. The REM-
propagated version does introduce a scale, and is run separately at calibrated density.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  S1  THE FOLDER IS RIGHT. A designed hairpin must fold far below a poly-A window of the
      same length. GATE: dG(hairpin) < -10 kcal/mol and dG(poly-A) > -0.5.
  S2  THE SIGNAL VARIES ALONG THE AXIS -- the property gene function lacks. GATE: median
      within-gene standard deviation of dG > 1.0 kcal/mol. If structure were near-constant
      within a gene it would be as incapable of moving this test as a per-gene label, and
      that must be checked rather than assumed.
  S3  THE TEST, on the same bar M4 was held to. Per gene, Spearman of -dG against measured
      P-site density, versus the SAME quantity with the windows shuffled within that gene --
      which preserves the gene's dG distribution and destroys only position. GATE: paired
      difference positive on significantly more than half of genes, binomial p < 1e-6.
  S4  SOLVER VERSUS INPUT, the M4b split. Feed -dG/RT as the dwell log-weight through the
      exact hard-rod occupancy at calibrated density and compare with the raw correlation.
      GATE: |rho(REM) - rho(raw)| < 0.01, i.e. the solver is not introducing the result.
  S5  HEAD TO HEAD against tAI on the SAME genes, and the two combined by rank-average.
      Reported, not gated: whichever wins, the comparison is only meaningful on one gene set.
  S6  THE SIGN, reported explicitly rather than assumed, as M4b did: both -dG and +dG.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.ribosome import (load_cds, load_ribo, gene_profile, trna_weights,
                          codon_logweights, occupancy_exact, _spearman, _binom_p,
                          _solve_fugacity, FOOTPRINT)

WINDOW_NT = 30           # entry-channel window, in nucleotides
RT37 = 0.6156            # kcal/mol at 37 C


def window_energies(cds: str, window: int = WINDOW_NT, offset: int = 0
                    ) -> np.ndarray:
    """MFE of the `window`-nt stretch starting at each codon's first nucleotide.

    offset shifts the window relative to the codon: 0 puts its 5' end at the codon (the
    entry-channel choice), a negative offset centres it. Positions where the window would
    run off the 3' end are filled with the gene's median, so length is always n_codons.
    """
    from rem.rna import mfe
    n = len(cds) // 3
    out = np.full(n, np.nan)
    seq = cds.upper().replace("T", "U")
    for i in range(n):
        a = 3 * i + offset
        b = a + window
        if a < 0 or b > len(seq):
            continue
        w = seq[a:b]
        if set(w) - set("ACGU"):
            continue
        out[i] = mfe(w).energy
    ok = np.isfinite(out)
    if ok.any():
        out[~ok] = float(np.median(out[ok]))
    else:
        out[:] = 0.0
    return out


def verify(n_genes: int = 250, min_codons: int = 200, min_counts: float = 200.0,
           window: int = WINDOW_NT, rho: float = 0.01, ell: int = FOOTPRINT,
           verbose: bool = True, seed: int = 0) -> dict:
    """Run S1-S6. Bars are fixed in the module docstring, above, before any number."""
    from rem.rna import mfe
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    out: Dict[str, object] = {}
    rng = np.random.default_rng(seed)

    # ---- S1: the folder is right ----------------------------------------------------
    hp = mfe(("GGGGGAAAACCCCC" * 3)[:window]).energy
    pa = mfe("A" * window).energy
    out["S1_hairpin"], out["S1_polyA"] = float(hp), float(pa)
    out["S1"] = bool(hp < -10.0 and pa > -0.5)
    say(f"  S1 folder sanity: hairpin {hp:+.2f}, poly-A {pa:+.2f} kcal/mol   "
        f"{'PASS' if out['S1'] else 'FAIL'}")

    cds = load_cds(); W = trna_weights(); h = load_ribo()
    genes = [g for g in h.keys() if g in cds and len(cds[g]) // 3 >= min_codons]
    genes.sort(); rng.shuffle(genes)

    say(f"\n  folding {window}-nt windows along each transcript ...")
    items, t0 = [], time.perf_counter()
    for g in genes:
        if len(items) >= n_genes:
            break
        n = len(cds[g]) // 3
        prof = gene_profile(h, g, n)
        if prof is None or prof.sum() < min_counts:
            continue
        dg = window_energies(cds[g], window)
        lw_tai, _ = codon_logweights(cds[g], W)
        items.append((g, n, prof, dg, lw_tai))
        if len(items) % 50 == 0:
            say(f"      {len(items)} genes, {time.perf_counter()-t0:.0f}s")
    h.close()
    say(f"      {len(items)} genes folded in {time.perf_counter()-t0:.0f}s")

    # ---- S2: does it vary along the axis? -------------------------------------------
    sds = np.array([float(np.std(dg)) for _, _, _, dg, _ in items])
    out["S2_median_sd"] = float(np.median(sds))
    out["S2"] = bool(np.median(sds) > 1.0)
    say(f"\n  S2 within-gene sd of dG: median {np.median(sds):.3f} kcal/mol "
        f"(bar > 1.0)   {'PASS' if out['S2'] else 'FAIL'}")
    say(f"      dG overall: mean {np.mean([dg.mean() for _,_,_,dg,_ in items]):+.2f}, "
        f"range {min(dg.min() for _,_,_,dg,_ in items):+.1f} to "
        f"{max(dg.max() for _,_,_,dg,_ in items):+.1f}")

    # ---- S3 / S6: the test, and the sign --------------------------------------------
    r_raw, r_shuf, r_flip = [], [], []
    for g, n, prof, dg, _ in items:
        a = _spearman(-dg, prof)
        idx = rng.permutation(n)
        b = _spearman(-dg[idx], prof)
        c = _spearman(dg, prof)
        if np.isfinite(a) and np.isfinite(b):
            r_raw.append(a); r_shuf.append(b); r_flip.append(c)
    r_raw, r_shuf, r_flip = map(np.array, (r_raw, r_shuf, r_flip))
    d = r_raw - r_shuf
    k, nt = int((d > 0).sum()), len(d)
    p = _binom_p(k, nt)
    out.update({"S3_median_raw": float(np.median(r_raw)),
                "S3_median_shuf": float(np.median(r_shuf)),
                "S3_paired": float(np.median(d)), "S3_win": k / nt, "S3_p": p})
    out["S3"] = bool(p < 1e-6 and k > nt / 2)
    say(f"\n  S3 -dG vs measured P-site density, {nt} genes")
    say(f"      median Spearman  real {np.median(r_raw):+.4f}   window-shuffled "
        f"{np.median(r_shuf):+.4f}   paired {np.median(d):+.4f}")
    say(f"      beats its own shuffle on {k}/{nt} ({100*k/nt:.1f}%), binomial p {p:.3e}")
    say(f"      S3 {'PASS' if out['S3'] else 'FAIL'}  (bar p < 1e-6 and > 50%)")
    out["S6_median_flip"] = float(np.median(r_flip))
    say(f"  S6 sign check: -dG {np.median(r_raw):+.4f}   +dG "
        f"{np.median(r_flip):+.4f}  (mirror images, as they must be)")

    # ---- S4: solver vs input --------------------------------------------------------
    r_rem = []
    for g, n, prof, dg, _ in items:
        lw = -dg / RT37
        lw = lw - lw.mean()
        z = _solve_fugacity(lw, rho, ell)
        p_occ, _ = occupancy_exact(lw + z, ell)
        v = _spearman(p_occ, prof)
        if np.isfinite(v):
            r_rem.append(v)
    r_rem = np.array(r_rem)
    gap = abs(float(np.median(r_rem)) - float(np.median(r_raw)))
    out["S4_rem"], out["S4_gap"] = float(np.median(r_rem)), gap
    out["S4"] = bool(gap < 0.01)
    say(f"\n  S4 REM-propagated at rho={rho}: {np.median(r_rem):+.4f}  vs raw "
        f"{np.median(r_raw):+.4f}   |gap| {gap:.4f} (bar < 0.01)   "
        f"{'PASS' if out['S4'] else 'FAIL'}")

    # ---- S5: head to head against tAI, same genes ------------------------------------
    r_tai, r_both = [], []
    for g, n, prof, dg, lw_tai in items:
        t = _spearman(np.exp(lw_tai), prof)
        rank = lambda x: np.argsort(np.argsort(x)).astype(float)
        comb = rank(-dg) + rank(np.exp(lw_tai))
        c = _spearman(comb, prof)
        if np.isfinite(t):
            r_tai.append(t)
        if np.isfinite(c):
            r_both.append(c)
    out["S5_tai"], out["S5_both"] = float(np.median(r_tai)), float(np.median(r_both))
    say(f"\n  S5 head to head on the same {nt} genes (median Spearman)")
    say(f"      structure (-dG)      {np.median(r_raw):+.4f}")
    say(f"      tAI dwell (1/W)      {np.median(r_tai):+.4f}")
    say(f"      rank-average of both {np.median(r_both):+.4f}")

    gates = ["S1", "S2", "S3", "S4"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    verify()
