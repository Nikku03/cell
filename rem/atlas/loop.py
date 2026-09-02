"""The propose-price-test loop, with the model in the proposing seat and blinded to the answer.

THE CLAIM BEING TESTED. A large model has breadth and no access to 1e-12; an exact solver has
exactness and no imagination. The proposed division of labour is:

    1. the model reads the data and proposes a missing mechanism
    2. the exact solver prices what that mechanism costs in the tail
    3. the gap detector tests the augmented model on HELD-OUT CONDITIONS, not held-out samples
    4. loop

That is a claim about a workflow, so it has to be run, not argued. This module is the harness.

THE BLINDING, WHICH IS THE WHOLE EXPERIMENT. `generate()` picks ONE hidden mechanism using
os.urandom -- not a seed anyone chose -- writes the truth to a file the proposer must not open,
and writes ONLY summary observables for the proposer to read. The proposal is committed to git
BEFORE the truth file is read. Git history is the evidence that the order was respected; without
it this is a demonstration rather than a test.

WHAT COUNTS AS A PASS, PREDECLARED:
  L1  the proposer names the correct mechanism family from observables alone
  L2  the augmented model, refitted, reproduces HELD-OUT conditions the base model cannot
  L3  the solver prices the mechanism in the tail, and the price is large -- if a missing
      mechanism costs nothing in the tail, finding it was not worth the loop
  L4  the gap detector fires on the base model and goes quiet on the augmented one. Both halves
      are required: a detector that fires on everything has found nothing.

WHAT WOULD FALSIFY THE WORKFLOW: the proposer names the wrong family, or names the right one
but the detector cannot tell the two apart on held-out conditions. Either outcome is reported.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

MECHANISMS = ["bursty_tx", "refractory_promoter", "erlang_protein_decay",
              "negative_feedback", "constitutive_leak", "gene_dosage",
              "saturating_protein_decay"]

BASE_RATES = dict(k_on=0.6, k_off=1.4, k_tx=14.0, k_mdeg=1.0, k_tl=1.6, k_pdeg=0.35)
CONDITIONS = [0.35, 0.55, 0.8, 1.0, 1.45, 2.1]        # multiplies k_tx: an induction series


def _solve(rows, cols, vals, n, shape):
    r = np.array(rows); c = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n)
    A = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([c, np.arange(n)]),
                        np.concatenate([r, np.arange(n)]))), shape=(n, n)).tolil()
    A[0, :] = 1.0
    b = np.zeros(n); b[0] = 1.0
    p = np.maximum(spl.spsolve(A.tocsr(), b), 0.0)
    p /= p.sum()
    return p.reshape(shape)


def model(mech: str, rates: Dict[str, float], induction: float,
          cap_m: int = 26, cap_p: int = 90, strength: float = 1.0) -> np.ndarray:
    """Joint (promoter-state, mRNA, protein) -> returns (mRNA marginal, protein marginal).

    `mech` is None/'base' for the model a modeller would assume; anything else adds one
    mechanism on top of the same base.
    """
    k_on, k_off = rates["k_on"], rates["k_off"]
    k_tx = rates["k_tx"] * induction
    k_md, k_tl, k_pd = rates["k_mdeg"], rates["k_tl"], rates["k_pdeg"]
    ng = 3 if mech == "refractory_promoter" else 2
    nph = 4 if mech == "erlang_protein_decay" else 1
    nm, npr = cap_m + 1, cap_p + 1
    n = ng * nm * npr * nph
    idx = lambda g, m, p, ph: ((g * nm + m) * npr + p) * nph + ph
    rows, cols, vals = [], [], []

    def add(i, j, r):
        if r > 0:
            rows.append(i); cols.append(j); vals.append(r)

    burst_q = 0.0
    if mech == "bursty_tx":
        bmean = 1.0 + 2.5 * strength
        burst_q = bmean / (1.0 + bmean)

    for g in range(ng):
        for m in range(nm):
            for p in range(npr):
                for ph in range(nph):
                    i = idx(g, m, p, ph)
                    # ---- promoter ----
                    if mech == "refractory_promoter":
                        # OFF(0) -> REFRACTORY(1) -> ON(2) -> OFF: ON waiting time non-exponential
                        if g == 0: add(i, idx(1, m, p, ph), k_on * 2.0)
                        elif g == 1: add(i, idx(2, m, p, ph), k_on * 2.0)
                        else: add(i, idx(0, m, p, ph), k_off)
                        active = (g == 2)
                    else:
                        add(i, idx(1 - g, m, p, ph), k_on if g == 0 else k_off)
                        active = (g == 1)
                    # ---- transcription ----
                    rate_tx = k_tx if active else 0.0
                    if mech == "negative_feedback":
                        rate_tx = rate_tx / (1.0 + (p / (18.0 / max(strength, 1e-9))) ** 2)
                    if mech == "gene_dosage" and g == (ng - 1):
                        rate_tx *= (1.0 + strength)
                    if mech == "constitutive_leak" and not active:
                        rate_tx = k_tx * 0.12 * strength
                    if rate_tx > 0:
                        if mech == "bursty_tx":
                            for j in range(1, min(12, nm - m)):
                                w = (1 - burst_q) * burst_q ** (j - 1)
                                add(i, idx(g, m + j, p, ph), rate_tx * w / (1.0 / (1 - burst_q)))
                        elif m + 1 < nm:
                            add(i, idx(g, m + 1, p, ph), rate_tx)
                    if m > 0:
                        add(i, idx(g, m - 1, p, ph), k_md * m)
                    if p + 1 < npr and m > 0:
                        add(i, idx(g, m, p + 1, ph), k_tl * m)
                    # ---- protein decay ----
                    if p > 0:
                        if mech == "erlang_protein_decay":
                            r = k_pd * p * nph
                            if ph < nph - 1: add(i, idx(g, m, p, ph + 1), r)
                            else: add(i, idx(g, m, p - 1, 0), r)
                        elif mech == "saturating_protein_decay":
                            km = 22.0 / max(strength, 1e-9)
                            add(i, idx(g, m, p - 1, ph), k_pd * p * km / (km + p))
                        else:
                            add(i, idx(g, m, p - 1, ph), k_pd * p)
    P = _solve(rows, cols, vals, n, (ng, nm, npr, nph))
    return P.sum(axis=(0, 2, 3)), P.sum(axis=(0, 1, 3))


def summarise(pm: np.ndarray, pp: np.ndarray) -> Dict[str, float]:
    def mom(p):
        x = np.arange(len(p)); m = float((x * p).sum())
        v = float((x * x * p).sum() - m * m)
        sd = math.sqrt(max(v, 1e-30))
        sk = float((((x - m) ** 3) * p).sum()) / max(sd ** 3, 1e-30)
        return m, v / max(m, 1e-12), sk
    mm, mf, ms = mom(pm)
    pmn, pf, ps = mom(pp)
    c = np.cumsum(pp)
    q = {f"p_q{int(t*100)}": float(np.searchsorted(c, t)) for t in (0.1, 0.5, 0.9, 0.99)}
    return {"mrna_mean": mm, "mrna_fano": mf, "mrna_skew": ms,
            "prot_mean": pmn, "prot_fano": pf, "prot_skew": ps, **q}


def generate(out_obs: str, out_truth: str) -> str:
    """Pick a hidden mechanism with os.urandom, emit observables, seal the truth."""
    pick = MECHANISMS[int.from_bytes(os.urandom(2), "big") % len(MECHANISMS)]
    strength = 0.7 + (int.from_bytes(os.urandom(2), "big") % 100) / 100.0
    obs = []
    for ind in CONDITIONS:
        pm, pp = model(pick, BASE_RATES, ind, strength=strength)
        obs.append({"induction": ind, **summarise(pm, pp)})
    with open(out_obs, "w") as fh:
        json.dump({"conditions": obs, "note": "protein and mRNA summary statistics only"},
                  fh, indent=1)
    with open(out_truth, "w") as fh:
        json.dump({"mechanism": pick, "strength": strength, "rates": BASE_RATES}, fh, indent=1)
    return out_obs
