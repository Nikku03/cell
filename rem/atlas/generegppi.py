"""GENEREGPPI -- the user's rule: proteins that interact in one cell interact in another cell if both are made there and
sit in the same room (compartment). Tested on knockdown data with Monte Carlo, and used to predict.

Asked by the user: "if protein A interacts with protein B in one cell type, and both proteins are in the same room in
another cell, they will interact. Use that and add it to the Monte Carlo and see what we get."
Readout: proteins that work together give similar knockdown effects (knock down either, the same genes move). So in a
cell type, the rule says ON pairs should have similar knockdown patterns there; OFF pairs should not.
Data (cellbrain bank): interactions -- HPA consensus (PPI_HPA), OpenCell IP-MS measured in HEK293T (PPI_OPENCELL; a
cell type not among the four tested), Complex Portal co-membership (SETS_COMPLEX, complexes <= 60 members); rooms --
HPA subcellular locations (SETS_LOCATION) and OpenCell localisations (grades 3+2), mapped to 11 shared rooms (ROOMS
below); made = HPA cell-line RNA nTPM >= 1 in that cell line (protein made; RNA as the proxy, the only per-cell-type
measure here); knockdown effects -- XCELL_* (K562, RPE1, HepG2, Jurkat essential screens; K562 genome-wide), links and
cis masking as in generegx. TIMING is not in this data (no time courses): not tested.
Pattern of a knockdown in a cell type = its log2 fold changes minus its strength x that cell type's generic profile
(generegp split; the shared stress response removed, else any two strong knockdowns look alike). Similarity of two
knockdowns = Pearson r of their patterns over measured genes. Pairs whose TSS are within 100 kb are dropped (CRISPRi
of one can silence the other). "Clear effect" = >= 20 links in that cell type.
RULE in cell type X: ON = interacting, both made in X, share a room; OFF-ROOM = interacting, both made, no shared room;
OFF-MADE = interacting, one not made in X.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G4-*):
 G4-R1  SAME ROOM MATTERS. Among interacting pairs with clear effects in X: median r(ON) - median r(OFF-ROOM) >= 0.02
        and Monte Carlo p < 0.05 (ON/OFF labels permuted 10,000 times within strength quintiles), in each of the four
        essential cell types (K562 genome-wide reported).
 G4-R2  THE INTERACTION CARRIES OVER FROM ANOTHER CELL TYPE. OpenCell pairs (measured in HEK293T) that are ON in X vs
        random non-interacting pairs that are also made and share a room (each pair's partner swapped for a random gene
        in a room it shares, same strength quintile, not cis), 1,000 Monte Carlo draws: median r gain >= 0.02 and
        p_MC < 0.05, in each of the four essential cell types.
 G4-R3  NOT MADE -> NO PARTNERSHIP. Pooled over the five screens (all knockdowns): median r(ON) - median r(OFF-MADE)
        >= 0.02, Monte Carlo p < 0.05 (labels permuted 10,000 times within screen).
 G4-P1  THE RULE HELPS PREDICT (on top of G3). A knockdown's profile in X predicted by the G3-T1 model (other cell
        types, strength + patterns) plus its ON partners' mean pattern measured in X (partners with clear effects;
        never its own data in X). Fit on DEV knockdowns, scored once on EXAM knockdowns with a clear effect and >= 1
        ON partner. CLAIM in each of the four cell types: BETTER than the G3-T1 model (house rule) AND Monte Carlo
        p < 0.05 against 200 draws where the partners are swapped for random genes made in X, sharing a room, with
        clear effects (same number).
 G4-P2  PREDICT A GENE NEVER MEASURED ANYWHERE, FROM ITS PARTNERS ALONE. Model = generic + ON partners' mean pattern
        in X. Same split, scoring and Monte Carlo; BETTER than generic and p_MC < 0.05 in each of the four.
Controls: R1/R2/R3 negative = one random relabelling must not pass, planted = ON similarities + 0.05 must pass;
P1/P2 negative = the first random-partner draw must not be BETTER, planted = partner pattern replaced by the
knockdown's own pattern in X plus noise of its lfcSE must be BETTER.
NOT: similarity of knockdown effects is a readout of working together, not proof of binding; rooms come from other
cell lines (HPA, HEK293T) and are assumed to hold; RNA stands in for protein; essential genes are made almost
everywhere, so OFF-MADE is rare (tens to hundreds of pairs per screen).

Run: python3 -I rem/atlas/generegppi.py declare | test [--trial]
"""

from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
TRIAL = "--trial" in sys.argv
_argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
_s = importlib.util.spec_from_file_location("generegp", os.path.join(HERE, "generegp.py"))
gp = importlib.util.module_from_spec(_s); _s.loader.exec_module(gp)
sys.argv = _argv
gx, cbb, Fact = gp.gx, gp.cbb, gp.Fact

LEDGER = gx.LEDGER
RUN = os.path.join(cbb.HOME, "generegppi_trial" if TRIAL else "generegppi")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "genereg")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_generegppi.txt")
CLAIMS = ["G4-R1", "G4-R2", "G4-R3", "G4-P1", "G4-P2"]
ESS, GW, CELL = gx.ESS, gx.GW, gx.CELL
MIN_LINKS, CIS_PAIR, SEED = 20, 100_000, 20261124
N_R1, N_R2, N_R3, N_P = (200, 50, 200, 5) if TRIAL else (10_000, 1_000, 10_000, 200)
MADE = float(np.log1p(1.0))
ROOMS = {
    "nucleoplasm": ["Nucleoplasm", "Nuclear bodies", "Nuclear speckles", "Kinetochore", "Mitotic chromosome", "nucleoplasm", "nuclear_punctae", "chromatin"],
    "nucleolus": ["Nucleoli", "Nucleoli fibrillar center", "Nucleoli rim", "nucleolus_fc_dfc", "nucleolus_gc"],
    "nuclear_membrane": ["Nuclear membrane", "nuclear_membrane"],
    "cytosol": ["Cytosol", "Cytoplasmic bodies", "Aggresome", "Rods & Rings", "cytoplasmic", "big_aggregates"],
    "mitochondria": ["Mitochondria", "mitochondria"],
    "er": ["Endoplasmic reticulum", "er"],
    "golgi": ["Golgi apparatus", "golgi"],
    "vesicles": ["Vesicles", "Endosomes", "Lysosomes", "Peroxisomes", "Lipid droplets", "vesicles"],
    "plasma_membrane": ["Plasma membrane", "Cell Junctions", "Focal adhesion sites", "membrane", "cell_contact", "focal_adhesions"],
    "cytoskeleton": ["Actin filaments", "Intermediate filaments", "Microtubules", "Microtubule ends", "Mitotic spindle", "Cleavage furrow",
                     "Cytokinetic bridge", "Midbody", "Midbody ring", "cytoskeleton"],
    "centrosome": ["Centrosome", "Centriolar satellite", "Basal body", "Primary cilium", "Primary cilium tip", "Primary cilium transition zone", "centrosome"],
}


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "generegppi.py"), "rb").read()); h.update(gp.code_sha().encode())
    for n in ("PPI_HPA", "PPI_OPENCELL", "SETS_COMPLEX", "SETS_LOCATION", "SETS_OPENCELL_LOCATION"):
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ knowledge
def knowledge():
    ppi = collections.defaultdict(set)
    z = gx.npz("PPI_HPA")
    for a, b in zip(z["a"].astype(str), z["b"].astype(str)):
        if a != b:
            ppi[frozenset((a, b))].add("HPA")
    z = gx.npz("PPI_OPENCELL")
    for a, b in zip(z["a"].astype(str), z["b"].astype(str)):
        if a != b and a.startswith("ENSG") and b.startswith("ENSG"):
            ppi[frozenset((a, b))].add("OpenCell")
    for d in json.load(open(gx.L("SETS_COMPLEX"))).values():
        m = d["members"]
        if len(m) <= 60:
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    ppi[frozenset((m[i], m[j]))].add("Complex")
    inv = {lab: room for room, labs in ROOMS.items() for lab in labs}; rooms = collections.defaultdict(set)
    for k, d in json.load(open(gx.L("SETS_LOCATION"))).items():
        if k in inv:
            for g in d["members"]:
                rooms[g].add(inv[k])
    for k, d in json.load(open(gx.L("SETS_OPENCELL_LOCATION"))).items():
        if k in inv:
            for g in d["members"]:
                rooms[g].add(inv[k])
    return ppi, rooms


def made_in(K, s, known_not=False):
    """made in that cell line (nTPM >= 1); with known_not, instead: measured in HPA and below 1 (missing data is neither)."""
    e = gx.hpa_expr(K)[:, ESS.index(s) if s in ESS else 0]
    return (np.isfinite(e) & (e < MADE)) if known_not else (e >= MADE)


def cis_pair(T, a, b):
    ta, tb = T.get(a), T.get(b)
    return bool(ta and tb and ta[0] == tb[0] and abs(ta[1] - tb[1]) <= CIS_PAIR)


def patterns(S, K, rows, fit_mask=None):
    """genes x K patterns (strength x generic removed) and standardised columns for r; strength; links."""
    P = gx.align(S, rows, K, "lfc"); links = gx.align(S, rows, K, "sig").sum(0)
    m = (links >= MIN_LINKS) if fit_mask is None else fit_mask
    a, E, g = gp.split(P, m)
    Z = E - np.nanmean(E, 0); sd = np.sqrt(np.nanmean(Z ** 2, 0)); Z = np.nan_to_num(Z / np.where(sd > 0, sd, np.nan))
    return P, E, Z, a, links


def pair_r(Z, ia, ib, chunk=20000):
    out = np.empty(len(ia), np.float32); n = Z.shape[0]
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = (Z[:, ia[s:s + chunk]] * Z[:, ib[s:s + chunk]]).sum(0) / n
    return out


def strat_perm_p(x, lab, strata, n, rng, shift=0.0):
    """median(x[lab]) + shift - median(x[~lab]) and its Monte Carlo p (labels permuted within strata)."""
    x = x.copy(); x[lab] += shift
    real = float(np.median(x[lab]) - np.median(x[~lab])); ge = 0
    groups = [np.nonzero(strata == k)[0] for k in np.unique(strata)]
    for _ in range(n):
        L_ = lab.copy()
        for gi in groups:
            L_[gi] = L_[gi][rng.permutation(len(gi))]
        ge += (np.median(x[L_]) - np.median(x[~L_])) >= real
    return real, float((1 + ge) / (1 + n))


# ------------------------------------------------------------------------------------------------ R1, R2, R3
def screen_pairs(S, s, ppi, rooms, T, clear_only=True):
    K = list(S["cols"]); rows = list(S["rows"]); ki = {k: i for i, k in enumerate(K)}
    P, E, Z, a, links = patterns(S, K, rows)
    made = made_in(K, s); notmade = made_in(K, s, known_not=True); clear = links >= MIN_LINKS
    pairs = []
    for p, src in ppi.items():
        x, y = tuple(p)
        if x in ki and y in ki and not cis_pair(T, x, y):
            i, j = ki[x], ki[y]
            if clear_only and not (clear[i] and clear[j]):
                continue
            pairs.append((i, j, "OpenCell" in src))
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 3)
    ia, ib, oc = pairs[:, 0], pairs[:, 1], pairs[:, 2].astype(bool)
    has = np.array([bool(rooms.get(K[i])) and bool(rooms.get(K[j])) for i, j in zip(ia, ib)], bool)
    share = np.array([bool(rooms.get(K[i], set()) & rooms.get(K[j], set())) for i, j in zip(ia, ib)], bool)
    both = made[ia] & made[ib]
    q = np.searchsorted(np.quantile(np.abs(a[clear]), [0.2, 0.4, 0.6, 0.8]), np.abs(a)) if clear.any() else np.zeros(len(K), int)
    return dict(K=K, Z=Z, a=a, made=made, notmade=notmade, clear=clear, q=q, ia=ia, ib=ib, oc=oc, has=has, share=share, both=both, r=pair_r(Z, ia, ib))


def run_r1(D, rng):
    out = {}
    for s, d in D.items():
        sel = d["has"] & d["both"]; on = d["share"][sel]; x = d["r"][sel]
        st = np.minimum(d["q"][d["ia"]], d["q"][d["ib"]])[sel]
        real, p = strat_perm_p(x, on, st, N_R1, rng)
        L_ = on[rng.permutation(len(on))]; neg = float(np.median(x[L_]) - np.median(x[~L_]))
        pl, ppl = strat_perm_p(x, on, st, max(200, N_R1 // 50), rng, shift=0.05)
        out[CELL[s]] = dict(on=int(on.sum()), off_room=int((~on).sum()), median_r_on=float(np.median(x[on])), median_r_off_room=float(np.median(x[~on])),
                            diff=real, p_mc=p, passed=bool(real >= 0.02 and p < 0.05), negative_diff=neg, negative_passed=bool(neg >= 0.02),
                            planted_diff=pl, planted_passed=bool(pl >= 0.02 and ppl < 0.05))
    ok = all(out[CELL[s]]["passed"] for s in ESS); ctrl = all(not v["negative_passed"] and v["planted_passed"] for v in out.values())
    return dict(per_screen=out, gate=dict(real=ok, controls_ok=ctrl, verdict="PASS" if ok and ctrl else "FAIL"))


def run_r2(D, ppi, rooms, T, rng):
    out = {}
    for s in ESS:
        d = D[s]; K = d["K"]; nK = len(K)
        sel = d["oc"] & d["share"] & d["both"]; ia, ib = d["ia"][sel], d["ib"][sel]; real_r = d["r"][sel]
        partners = collections.defaultdict(set)
        for p in ppi:
            x, y = tuple(p); partners[x].add(y); partners[y].add(x)
        R_ = np.zeros((nK, len(ROOMS)), bool); rn = list(ROOMS)
        for i, k in enumerate(K):
            for rr in rooms.get(k, ()):
                R_[i, rn.index(rr)] = True
        ok_gene = d["clear"] & d["made"]
        cand = {}
        for i in np.unique(ia):
            sh = (R_ & R_[i]).any(1) & ok_gene; sh[i] = False
            for j in np.nonzero(sh)[0]:
                if K[j] in partners[K[i]] or cis_pair(T, K[i], K[j]):
                    sh[j] = False
            idx = np.nonzero(sh)[0]; cand[i] = {qq: idx[d["q"][idx] == qq] for qq in range(5)}; cand[i]["all"] = idx
        pools = [cand[i][d["q"][j]] if len(cand[i][d["q"][j]]) else cand[i]["all"] for i, j in zip(ia, ib)]
        keep = np.array([len(pl) > 0 for pl in pools]); ia, ib, real_r = ia[keep], ib[keep], real_r[keep]; pools = [pl for pl, k in zip(pools, keep) if k]
        lens = np.array([len(pl) for pl in pools]); flat = np.concatenate(pools); off = np.concatenate([[0], np.cumsum(lens)[:-1]])
        real = float(np.median(real_r)); nulls = np.empty(N_R2)
        for b in range(N_R2):
            pick = flat[off + (rng.random(len(lens)) * lens).astype(np.int64)]
            nulls[b] = np.median(pair_r(d["Z"], ia, pick))
        gain = real - float(np.median(nulls)); p = float((1 + (nulls >= real).sum()) / (1 + N_R2))
        out[CELL[s]] = dict(pairs=int(len(ia)), median_r_opencell_on=real, median_r_random_same_room=float(np.median(nulls)),
                            null_95=float(np.percentile(nulls, 95)), gain=gain, p_mc=p, passed=bool(gain >= 0.02 and p < 0.05),
                            negative_gain=float(nulls[0] - np.median(nulls[1:])), negative_passed=bool(nulls[0] - np.median(nulls[1:]) >= 0.02 and
                                                                                             (1 + (nulls[1:] >= nulls[0]).sum()) / N_R2 < 0.05),
                            planted_gain=gain + 0.05, planted_passed=bool(real + 0.05 - np.median(nulls) >= 0.02 and (1 + (nulls >= real + 0.05).sum()) / (1 + N_R2) < 0.05))
    ok = all(v["passed"] for v in out.values()); ctrl = all(not v["negative_passed"] and v["planted_passed"] for v in out.values())
    return dict(per_cell=out, gate=dict(real=ok, controls_ok=ctrl, verdict="PASS" if ok and ctrl else "FAIL"))


def run_r3(SS, ppi, rooms, T, rng):
    xs, labs, st, per = [], [], [], {}
    for k_, s in enumerate(ESS + [GW]):
        d = screen_pairs(SS[s], s, ppi, rooms, T, clear_only=False)
        on = d["both"] & d["share"]; off = d["notmade"][d["ia"]] | d["notmade"][d["ib"]]; sel = on | off
        xs.append(d["r"][sel]); labs.append(on[sel]); st.append(np.full(int(sel.sum()), k_))
        per[CELL[s]] = dict(on=int(on.sum()), off_made=int(off.sum()), median_r_on=float(np.median(d["r"][on])) if on.any() else None,
                            median_r_off_made=float(np.median(d["r"][off])) if off.any() else None)
    x, lab, stv = np.concatenate(xs), np.concatenate(labs), np.concatenate(st)
    real, p = strat_perm_p(x, lab, stv, N_R3, rng)
    L_ = lab.copy()
    for k in np.unique(stv):
        g = np.nonzero(stv == k)[0]; L_[g] = L_[g][rng.permutation(len(g))]
    neg = float(np.median(x[L_]) - np.median(x[~L_])); pl, ppl = strat_perm_p(x, lab, stv, 200, rng, shift=0.05)
    out = dict(per_screen=per, diff=real, p_mc=p, passed=bool(real >= 0.02 and p < 0.05), negative_diff=neg, planted_diff=pl)
    ctrl = (neg < 0.02) and (pl >= 0.02 and ppl < 0.05)
    out["gate"] = dict(real=out["passed"], controls_ok=bool(ctrl), verdict="PASS" if out["passed"] and ctrl else "FAIL")
    return out


# ------------------------------------------------------------------------------------------------ P1, P2
def fit_ne(F, O, m):
    A = np.stack([np.nan_to_num(f)[m] for f in F], 1); return np.linalg.solve(A.T @ A + 1e-9 * np.eye(A.shape[1]), A.T @ O[m])


def run_p(SS, ppi, rooms, T, rng):
    from scipy import sparse
    T4 = sorted(set.intersection(*(set(SS[s]["rows"]) for s in ESS))); K4 = sorted(set.intersection(*(set(SS[s]["cols"]) for s in ESS)))
    Pall = {s: gx.align(SS[s], T4, K4, "lfc") for s in ESS}; ki = {k: i for i, k in enumerate(K4)}; nK = len(K4)
    partners = collections.defaultdict(set)
    for p in ppi:
        x, y = tuple(p); partners[x].add(y); partners[y].add(x)
    R_ = np.zeros((nK, len(ROOMS)), bool); rn = list(ROOMS)
    for i, k in enumerate(K4):
        for rr in rooms.get(k, ()):
            R_[i, rn.index(rr)] = True
    out1, out2 = {}, {}
    for X in ESS:
        O = Pall[X]; links = gx.align(SS[X], T4, K4, "sig").sum(0); clear = links >= MIN_LINKS; made = made_in(K4, X)
        tr, sc = gp.roles(K4); tr &= clear; sc &= clear
        gen = np.nanmean(O[:, tr], 1)[:, None] * np.ones((1, nK), np.float32)
        Fb, _ = gp.features(gen, [Pall[s] for s in ESS if s != X], tr)
        _, EX, _ = gp.split(O, tr); EX = np.nan_to_num(EX)
        on = [[] for _ in range(nK)]
        for i, k in enumerate(K4):
            if not made[i]:
                continue
            for y in partners.get(k, ()):
                j = ki.get(y)
                if j is not None and j != i and clear[j] and made[j] and (R_[i] & R_[j]).any() and not cis_pair(T, k, y):
                    on[i].append(j)
        npart = np.array([len(v) for v in on]); tr2 = tr & (npart > 0); sc2 = sc & (npart > 0)

        def feat(lists):
            rr, cc, vv = [], [], []
            for i, v in enumerate(lists):
                for j in v:
                    rr.append(j); cc.append(i); vv.append(1.0 / len(v))
            M = sparse.csr_matrix((vv, (rr, cc)), shape=(nK, nK)); return np.asarray((sparse.csr_matrix(EX) @ M).todense(), np.float32)
        Pp = feat(on)
        mtr = np.isfinite(O) & tr2[None, :]
        w_b = fit_ne(Fb, O, mtr); r_b = gp.colr(gp.pred([f[:, sc2] for f in Fb], w_b), O[:, sc2])
        w_1 = fit_ne(Fb + [Pp], O, mtr); r_1 = gp.colr(gp.pred([f[:, sc2] for f in Fb + [Pp]], w_1), O[:, sc2])
        w_g = fit_ne([gen], O, mtr); r_g = gp.colr(gen[:, sc2], O[:, sc2])
        w_2 = fit_ne([gen, Pp], O, mtr); r_2 = gp.colr(gp.pred([gen[:, sc2], Pp[:, sc2]], w_2), O[:, sc2])
        g1, g2 = gp.better(r_1 - r_b), gp.better(r_2 - r_g); m1, m2 = g1["median"], g2["median"]
        # Monte Carlo: partners swapped for random genes made in X, in a shared room, with clear effects (same number)
        pool = clear & made; shared = (R_.astype(np.int32) @ R_.T.astype(np.int32)) > 0
        cands = [np.nonzero(shared[i] & pool & (np.arange(nK) != i))[0] if npart[i] else np.array([], int) for i in range(nK)]
        n1, n2 = np.empty(N_P), np.empty(N_P)
        for b in range(N_P):
            lists = [list(rng.choice(c, min(len(c), npart[i]), replace=False)) if npart[i] and len(c) else [] for i, c in enumerate(cands)]
            Pr = feat(lists)
            wr = fit_ne(Fb + [Pr], O, mtr); n1[b] = np.nanmedian(gp.colr(gp.pred([f[:, sc2] for f in Fb + [Pr]], wr), O[:, sc2]) - r_b)
            wr2 = fit_ne([gen, Pr], O, mtr); n2[b] = np.nanmedian(gp.colr(gp.pred([gen[:, sc2], Pr[:, sc2]], wr2), O[:, sc2]) - r_g)
            if b == 0:
                neg1 = gp.better(gp.colr(gp.pred([f[:, sc2] for f in Fb + [Pr]], wr), O[:, sc2]) - r_b)
                neg2 = gp.better(gp.colr(gp.pred([gen[:, sc2], Pr[:, sc2]], wr2), O[:, sc2]) - r_g)
        Pl = EX + rng.standard_normal(EX.shape).astype(np.float32) * np.nan_to_num(gx.align(SS[X], T4, K4, "se"), nan=0.3)
        wp = fit_ne(Fb + [Pl], O, mtr); pl1 = gp.better(gp.colr(gp.pred([f[:, sc2] for f in Fb + [Pl]], wp), O[:, sc2]) - r_b)
        wp2 = fit_ne([gen, Pl], O, mtr); pl2 = gp.better(gp.colr(gp.pred([gen[:, sc2], Pl[:, sc2]], wp2), O[:, sc2]) - r_g)
        p1 = float((1 + (n1 >= m1).sum()) / (1 + N_P)); p2 = float((1 + (n2 >= m2).sum()) / (1 + N_P))
        base = dict(knockdowns_fit=int(tr2.sum()), knockdowns_scored=int(sc2.sum()), median_on_partners=float(np.median(npart[sc2])) if sc2.any() else 0)
        out1[CELL[X]] = dict(**base, r_g3_median=float(np.nanmedian(r_b)), r_with_partners_median=float(np.nanmedian(r_1)), better=g1,
                             random_partner_gain_95=float(np.percentile(n1, 95)), p_mc=p1, passed=bool(g1["passed"] and p1 < 0.05),
                             weight_partners=float(w_1[-1]), negative=neg1, planted=pl1)
        out2[CELL[X]] = dict(**base, r_generic_median=float(np.nanmedian(r_g)), r_partners_only_median=float(np.nanmedian(r_2)), better=g2,
                             random_partner_gain_95=float(np.percentile(n2, 95)), p_mc=p2, passed=bool(g2["passed"] and p2 < 0.05),
                             weight_partners=float(w_2[-1]), negative=neg2, planted=pl2)
    res = []
    for o in (out1, out2):
        ok = all(v["passed"] for v in o.values()); ctrl = all(not v["negative"]["passed"] and v["planted"]["passed"] for v in o.values())
        res.append(dict(per_cell=o, gate=dict(real=ok, controls_ok=ctrl, verdict="PASS" if ok and ctrl else "FAIL")))
    return res


# ------------------------------------------------------------------------------------------------ ledger
def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G4-")[0].split("Controls:")[0].strip()[:1200], trial=TRIAL),
                               sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G4 already tested")
    t0 = time.time(); os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True); rng = np.random.default_rng(SEED)
    T = gx.tss(); keep = None
    if TRIAL:
        tr = np.random.default_rng(1); ess = sorted({str(e) for e in gx.npz("XCELL_K562ESS")["cols_ens"] if str(e)})
        gw = sorted({str(e) for e in gx.npz(f"XCELL_{GW}")["cols_ens"] if str(e)})
        keep = set(tr.choice(ess, 600, replace=False)) | set(tr.choice(gw, 600, replace=False))
    SS = {s: gx.load(s, T, keep) for s in ESS + [GW]}
    ppi, rooms = knowledge(); print(f"knowledge: {len(ppi):,} interacting pairs, {len(rooms):,} genes with rooms ({time.time() - t0:.0f}s)", flush=True)
    D = {s: screen_pairs(SS[s], s, ppi, rooms, T) for s in ESS + [GW]}
    out = {}
    out["G4-R1"] = run_r1(D, rng); print("G4-R1 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G4-R2"] = run_r2(D, ppi, rooms, T, rng); print("G4-R2 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G4-R3"] = run_r3(SS, ppi, rooms, T, rng); print("G4-R3 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G4-P1"], out["G4-P2"] = run_p(SS, ppi, rooms, T, rng); print("G4-P done", f"({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        ctl = {c: out[c]["gate"]["controls_ok"] for c in CLAIMS}
        print("trial controls ok:", ctl, "; real outcomes not shown"); return
    for cid in CLAIMS:
        print(cid, json.dumps(out[cid], default=float)[:1800], flush=True)
    json.dump(out, open(os.path.join(OUTD, "ppi_rule_summary.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "generegppi:G4", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    n = cbb.bank().add_many([Fact(f"CLAIM-{cid}", "ppi_rule_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
    r1, r2, r3, p1, p2 = (out[c] for c in CLAIMS)
    Ls = ["=" * 97, "GENEREGPPI -- interact in one cell + made + same room in another -> work together there? (with Monte Carlo)", "=" * 97,
          "  G4-R1 same room matters (median r ON vs OFF-ROOM): " + ", ".join(f"{c} {v['median_r_on']:.3f} vs {v['median_r_off_room']:.3f} (p {v['p_mc']:.2g})"
                                                                         for c, v in r1["per_screen"].items()) + f" -> {r1['gate']['verdict']}",
          "  G4-R2 interaction from HEK293T carries over (OpenCell ON vs random same-room): " +
          ", ".join(f"{c} {v['median_r_opencell_on']:.3f} vs {v['median_r_random_same_room']:.3f} (p {v['p_mc']:.2g})" for c, v in r2["per_cell"].items()) +
          f" -> {r2['gate']['verdict']}",
          f"  G4-R3 not made -> no partnership: diff {r3['diff']:.3f}, p {r3['p_mc']:.2g} -> {r3['gate']['verdict']}",
          "  G4-P1 partners help prediction on top of G3: " + ", ".join(f"{c} {v['r_g3_median']:.3f}->{v['r_with_partners_median']:.3f} (p_MC {v['p_mc']:.2g})"
                                                                        for c, v in p1["per_cell"].items()) + f" -> {p1['gate']['verdict']}",
          "  G4-P2 partners alone predict an unmeasured gene: " + ", ".join(f"{c} {v['r_generic_median']:.3f}->{v['r_partners_only_median']:.3f} (p_MC {v['p_mc']:.2g})"
                                                                           for c, v in p2["per_cell"].items()) + f" -> {p2['gate']['verdict']}",
          f"  Facts written {n}"]
    open(RES, "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
