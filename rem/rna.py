"""RNA secondary structure: the McCaskill partition function and MFE folding.

WHY THIS IS A REM MODULE. A *nested* (pseudoknot-free) secondary structure is exactly a
non-crossing matching, and a non-crossing matching has a tree decomposition whose bags are
triples of sequence positions. So the governing law applies with the *sequence index* as
the variable:

    d = n (a position takes one of n values)      cost = d ** (treewidth + 1) = n ** 3

Concretely, every term of the dynamic program below mentions at most THREE free indices --
e.g. the multiloop split  QB[i,j] <- QM[i+1,h-1] * QM1[h,j-1]  mentions (i, h, j) -- so the
largest intermediate table is n^3 and the elimination width is 2 in the `rem.factorgraph`
sense (`largest_table = d^(width+1)`). `verify()` MEASURES the exponent by timing the fold
at several lengths and fitting a log-log slope; it comes out at ~3.

This is also where the wall is. Allowing crossing pairs (pseudoknots) destroys the
non-crossing tree decomposition: the simplest pseudoknot class needs O(n^6) (width 5), and
general pseudoknotted folding is NP-hard. Nothing here is a heuristic dodge of that -- the
model is restricted to nested structures and says so.

SEMIRINGS, following `rem.factorgraph`. The DP is built ONCE as an explicit hypergraph
whose hyperedges carry a real energy E in kcal/mol. Then

    "sum" semiring:  value = log sum exp( -E_total / RT )   -> log Z   (McCaskill)
    "min" semiring:  value = min E_total                    -> MFE     (Zuker)

One structure, two semirings, no duplicated recursion to keep in sync. Base-pair
probabilities come from a GENERIC inside-outside pass over the same hypergraph
(P(i,j) = inside(QB_ij) * outside(QB_ij) / Z), so the only thing that has to be right is
the recursion itself -- which `verify()` checks against explicit enumeration.

LOG SPACE everywhere in the sum semiring. A 76 nt tRNA has Z ~ e^30 and long sequences
overflow float64 quickly; nothing here ever exponentiates a raw Boltzmann weight.

ENERGY MODELS (both documented in their class docstrings):
  BasePairModel  -- E = sum over pairs of a per-pair-type energy. No stacking, no loops.
  StackingModel  -- Turner-style loop model: nearest-neighbour stacking, hairpin/bulge/
                    internal loop initiation tables, Ninio asymmetry, linear multiloop,
                    terminal AU/GU penalty. Approximations are listed in the docstring.

The energy of a *given* structure is computed by `structure_energy()`, which decomposes a
pair list into loops with its own stack walk and never touches the DP. That function is the
scoring half of the brute force; the enumeration half (`enumerate_structures`) lists every
nested structure explicitly. Together they are an algorithmically independent reference.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

# ------------------------------------------------------------------ physical constants
R_KCAL = 1.98717e-3           # gas constant, kcal / (mol K)
T37 = 310.15                  # 37 C in kelvin
RT37 = R_KCAL * T37           # 0.61633 kcal/mol

NEG_INF = float("-inf")
INF = float("inf")

CANONICAL: Set[Tuple[str, str]] = {("A", "U"), ("U", "A"), ("G", "C"),
                                   ("C", "G"), ("G", "U"), ("U", "G")}
WOBBLE: Set[Tuple[str, str]] = {("G", "U"), ("U", "G")}


def clean_seq(seq: str) -> str:
    """Upper-case, DNA->RNA, anything non-ACGU becomes N (never pairs)."""
    s = "".join(seq.split()).upper().replace("T", "U")
    return "".join(c if c in "ACGU" else "N" for c in s)


# ------------------------------------------------------------------ structure notation
def pairs_to_db(pairs: Iterable[Tuple[int, int]], n: int) -> str:
    s = ["."] * n
    for i, j in pairs:
        s[i] = "("
        s[j] = ")"
    return "".join(s)


def db_to_pairs(db: str) -> List[Tuple[int, int]]:
    stack, out = [], []
    for k, c in enumerate(db):
        if c == "(":
            stack.append(k)
        elif c == ")":
            if not stack:
                raise ValueError("unbalanced dot-bracket")
            out.append((stack.pop(), k))
    if stack:
        raise ValueError("unbalanced dot-bracket")
    return sorted(out)


def compare_structures(pred: Iterable[Tuple[int, int]],
                       ref: Iterable[Tuple[int, int]]) -> dict:
    """Sensitivity = TP/|ref|, PPV = TP/|pred|, the standard RNA-folding accuracy pair."""
    P, R = set(map(tuple, pred)), set(map(tuple, ref))
    tp = len(P & R)
    sens = tp / len(R) if R else float("nan")
    ppv = tp / len(P) if P else float("nan")
    f1 = 0.0 if (sens + ppv) == 0 else 2 * sens * ppv / (sens + ppv)
    return {"tp": tp, "n_pred": len(P), "n_ref": len(R),
            "sensitivity": sens, "ppv": ppv, "f1": f1}


# ================================================================== energy models
class EnergyModel:
    """Interface. Every method returns kcal/mol; the DP and the independent structure
    scorer call exactly these primitives, so a model cannot disagree with itself.

    A nested structure decomposes uniquely into loops. Each base pair (i,j) closes exactly
    one loop, classified by how many pairs sit immediately inside it:
        0 pairs  -> hairpin(i, j)
        1 pair   -> interior(i, j, k, l)     (covers stack, bulge and internal loop)
        >=2      -> ml_closing(i,j) + sum ml_branch(k,l) + ml_unpaired * (unpaired bases)
    plus the exterior loop:  sum ext_branch(k,l) + ext_unpaired * (unpaired bases).
    """
    name = "abstract"
    max_loop = 30          # max unpaired bases in an interior loop (Vienna's default)

    def can_pair(self, a: str, b: str) -> bool:
        return (a, b) in CANONICAL

    def hairpin(self, seq, i, j) -> float: raise NotImplementedError
    def interior(self, seq, i, j, k, l) -> float: raise NotImplementedError
    def ml_closing(self, seq, i, j) -> float: raise NotImplementedError
    def ml_branch(self, seq, k, l) -> float: raise NotImplementedError
    def ml_unpaired(self) -> float: raise NotImplementedError
    def ext_branch(self, seq, k, l) -> float: raise NotImplementedError
    def ext_unpaired(self) -> float: raise NotImplementedError


class BasePairModel(EnergyModel):
    """The simplest model that is still an energy model, not a pair count.

        E(structure) = sum over base pairs (i,j) of  e[type(i,j)]

    with e(GC) = -3, e(AU) = -2, e(GU) = -1 kcal/mol by default. No stacking, no loop
    entropy, no multiloop penalty -- context is completely ignored. It is here as the
    *negative control* for the stacking model: same search space, same DP, different
    scoring function, so any accuracy difference is attributable to the scoring function
    alone.

    Implementation note: the closing pair's energy is charged to the loop that pair closes
    (hairpin / interior / multiloop closing), which is why all three return the same value
    and the branch/unpaired terms are zero. Summing over loops then reproduces the sum over
    pairs exactly.
    """
    name = "base-pair"

    def __init__(self, gc=-3.0, au=-2.0, gu=-1.0, max_loop=30):
        self.e = {("G", "C"): gc, ("C", "G"): gc,
                  ("A", "U"): au, ("U", "A"): au,
                  ("G", "U"): gu, ("U", "G"): gu}
        self.max_loop = max_loop

    def pair_energy(self, a, b) -> float:
        return self.e.get((a, b), INF)

    def hairpin(self, seq, i, j):            return self.pair_energy(seq[i], seq[j])
    def interior(self, seq, i, j, k, l):     return self.pair_energy(seq[i], seq[j])
    def ml_closing(self, seq, i, j):         return self.pair_energy(seq[i], seq[j])
    def ml_branch(self, seq, k, l):          return 0.0
    def ml_unpaired(self):                   return 0.0
    def ext_branch(self, seq, k, l):         return 0.0
    def ext_unpaired(self):                  return 0.0


# ---- Turner-style parameters (Delta G at 37 C, kcal/mol) ---------------------------
# Watson-Crick nearest-neighbour stacks. Key (a, b, c, d) = outer pair (a,b) = (seq[i],
# seq[j]) stacked on inner pair (c,d) = (seq[i+1], seq[j-1]), i.e. the duplex
#       5'- a c -3'
#       3'- b d -5'
# The ten independent values are the standard Turner/Freier set written in XY/X'Y' form:
#   AA/UU -0.93  AU/UA -1.10  UA/AU -1.33  CU/GA -2.08  CA/GU -2.11
#   GU/CA -2.24  GA/CU -2.35  CG/GC -2.36  GG/CC -3.26  GC/CG -3.42
_WC_NN: Dict[Tuple[str, str, str, str], float] = {
    ("A", "U", "A", "U"): -0.93,
    ("A", "U", "U", "A"): -1.10,
    ("U", "A", "A", "U"): -1.33,
    ("C", "G", "U", "A"): -2.08,
    ("C", "G", "A", "U"): -2.11,
    ("G", "C", "U", "A"): -2.24,
    ("G", "C", "A", "U"): -2.35,
    ("C", "G", "G", "C"): -2.36,
    ("G", "C", "G", "C"): -3.26,
    ("G", "C", "C", "G"): -3.42,
}
STACK37: Dict[Tuple[str, str, str, str], float] = {}
for _k, _v in _WC_NN.items():
    _a, _b, _c, _d = _k
    STACK37[(_a, _b, _c, _d)] = _v
    STACK37[(_d, _c, _b, _a)] = _v          # reading the duplex from the other side

# Hairpin loop initiation by number of unpaired bases (Turner 2004 table, 3..30).
HAIRPIN37 = [INF, INF, INF, 5.40, 5.60, 5.70, 5.40, 6.00, 5.50, 6.40, 6.50,
             6.60, 6.70, 6.78, 6.86, 6.94, 7.01, 7.07, 7.13, 7.19, 7.25,
             7.30, 7.35, 7.40, 7.44, 7.49, 7.53, 7.57, 7.61, 7.65, 7.69]
# Bulge loop initiation by size.
BULGE37 = [INF, 3.80, 2.80, 3.20, 3.60, 4.00, 4.40, 4.59, 4.70, 4.80, 4.90,
           5.00, 5.10, 5.18, 5.26, 5.34, 5.41, 5.47, 5.53, 5.59, 5.65,
           5.70, 5.75, 5.80, 5.84, 5.89, 5.93, 5.97, 6.01, 6.05, 6.09]
# Internal loop initiation by total number of unpaired bases (n1 + n2 >= 2).
# Sizes 2 and 3 (1x1 and 1x2 loops) really use the Turner int11/int21 lookup tables, which
# are not implemented here; 1.70 / 1.80 stand in for them.
INTERNAL37 = [INF, INF, 1.70, 1.80, 1.10, 2.00, 2.00, 2.10, 2.30, 2.40, 2.50,
              2.60, 2.70, 2.78, 2.86, 2.94, 3.01, 3.07, 3.13, 3.18, 3.23,
              3.28, 3.33, 3.38, 3.42, 3.46, 3.50, 3.54, 3.58, 3.61, 3.65]

# Linear multiloop parameters  E = a + b * (branches incl. closing) + c * (unpaired).
# BOTH published sets are shipped, because which one you pick decides the tRNA answer
# (see benchmark_trna) and neither is "the" right answer in a model this coarse.
#   turner1999 -- ViennaRNA 1.8 defaults: ML_closing37 340, ML_intern37 40, ML_BASE37 0
#   turner2004 -- ViennaRNA 2.x defaults: cc 930, ci -90, cu 0
MULTILOOP_PRESETS = {"turner1999": (3.40, 0.40, 0.00),
                     "turner2004": (9.30, -0.90, 0.00)}


def _extrapolate(table: List[float], size: int, ref: int) -> float:
    """Jacobson-Stockmayer log extrapolation past the tabulated range."""
    if size < len(table):
        return table[size]
    return table[ref] + 1.75 * RT37 * math.log(size / ref)


class StackingModel(EnergyModel):
    """Turner-style nearest-neighbour loop model at 37 C.

    Terms actually implemented
      * stacking       -- the full 4x4 Watson-Crick nearest-neighbour table (10 unique
                          parameters, symmetrised). THIS is the term the simple base-pair
                          model is missing and the reason helices form at all.
      * hairpin        -- initiation table by loop size + terminal AU/GU penalty.
      * bulge          -- initiation table; a size-1 bulge additionally KEEPS the stack of
                          its flanking pairs (the standard special case).
      * internal loop  -- initiation by total size + Ninio asymmetry min(3.0, 0.6*|n1-n2|)
                          + terminal AU/GU penalty on both closing pairs.
      * hairpin terminal mismatch (optional, ON by default) -- a COARSE stand-in for the
                          Turner mismatch_hairpin table: -1.4 for a G-C closing pair,
                          -0.9 otherwise, plus the Turner-1999 first-mismatch bonuses
                          (U.U or G.A -0.9, G.G -0.8). When it is on the hairpin terminal
                          AU/GU penalty is dropped, as in Turner.
      * multiloop      -- linear:  a + b*(branches incl. closing) + c*(unpaired), from
                          MULTILOOP_PRESETS, plus terminal AU/GU per branch.
      * exterior loop  -- free unpaired bases, terminal AU/GU penalty per branch.
      * terminal AU/GU penalty 0.45 kcal/mol on any helix end that is not G-C.

    DELIBERATE APPROXIMATIONS, stated so nobody mistakes this for ViennaRNA
      * G-U wobble stacks are NOT the full Turner table. A stack containing exactly one
        G-U pair gets GU_SINGLE = -1.3, two adjacent G-U pairs get GU_TANDEM = -0.5.
        Real values range roughly -0.5 to -2.5, and the special 5'GG3'/3'UU5' tandem motif
        is not modelled.
      * The hairpin terminal mismatch is 2 numbers + 2 bonuses, not the 150-entry Turner
        table. Internal loops get no terminal mismatch at all.
      * No dangling-end energies (equivalent to ViennaRNA -d0) and no coaxial stacking.
        Coaxial stacking is worth several kcal/mol at a tRNA four-way junction, and its
        absence is the main reason the multiloop parameters below are load-bearing.
      * No special hairpin tables (tetraloop bonuses, all-C penalty, GU closure bonus).
      * No 1x1 / 1x2 / 2x2 internal loop lookup tables; those use the generic initiation.
    Each omission costs accuracy on real sequences; `benchmark_trna()` reports what is
    left, and its parameter sweep shows how little slack there is.
    """
    TERM_AU = 0.45
    GU_SINGLE = -1.30
    GU_TANDEM = -0.50
    MM_GC = -1.40         # hairpin terminal mismatch, G-C closing pair
    MM_AU = -0.90         # hairpin terminal mismatch, A-U / G-U closing pair
    MM_UU_GA = -0.90      # Turner-1999 first-mismatch bonus
    MM_GG = -0.80
    NINIO = 0.60
    NINIO_MAX = 3.00

    def __init__(self, max_loop: int = 30, terminal_mismatch: bool = True,
                 multiloop: str = "turner1999"):
        self.max_loop = max_loop
        self.terminal_mismatch = bool(terminal_mismatch)
        if multiloop not in MULTILOOP_PRESETS:
            raise ValueError(f"multiloop must be one of {sorted(MULTILOOP_PRESETS)}")
        self.multiloop = multiloop
        self.ML_A, self.ML_B, self.ML_C = MULTILOOP_PRESETS[multiloop]
        # per-instance copies so a sweep can perturb one number without global effects
        self.hairpin_init = list(HAIRPIN37)
        self.bulge_init = list(BULGE37)
        self.internal_init = list(INTERNAL37)
        self.name = ("stacking+mismatch" if self.terminal_mismatch else "stacking")
        self.name += f"/{multiloop}"

    # ---- helpers
    def terminal(self, a: str, b: str) -> float:
        """Terminal AU/GU penalty: helix ends that are not G-C cost 0.45."""
        return 0.0 if (a, b) in (("G", "C"), ("C", "G")) else self.TERM_AU

    def stack(self, a, b, c, d) -> float:
        v = STACK37.get((a, b, c, d))
        if v is not None:
            return v
        n_wob = ((a, b) in WOBBLE) + ((c, d) in WOBBLE)
        if n_wob == 2:
            return self.GU_TANDEM
        if n_wob == 1:
            return self.GU_SINGLE
        return INF          # non-canonical: should never be reached

    # ---- loop energies
    def hairpin(self, seq, i, j) -> float:
        size = j - i - 1
        if size < 3:
            return INF
        e = _extrapolate(self.hairpin_init, size, 30)
        if not self.terminal_mismatch:
            return e + self.terminal(seq[i], seq[j])
        # Turner replaces the AU penalty by a terminal mismatch for hairpins.
        e += self.MM_GC if (seq[i], seq[j]) in (("G", "C"), ("C", "G")) else self.MM_AU
        x, y = seq[i + 1], seq[j - 1]
        if (x, y) in (("U", "U"), ("G", "A")):
            e += self.MM_UU_GA
        elif (x, y) == ("G", "G"):
            e += self.MM_GG
        return e

    def interior(self, seq, i, j, k, l) -> float:
        n1, n2 = k - i - 1, j - l - 1
        a, b, c, d = seq[i], seq[j], seq[k], seq[l]
        if n1 == 0 and n2 == 0:                       # stack
            return self.stack(a, b, c, d)
        m = n1 + n2
        if n1 == 0 or n2 == 0:                        # bulge
            if m == 1:
                return _extrapolate(self.bulge_init, 1, 30) + self.stack(a, b, c, d)
            return (_extrapolate(self.bulge_init, m, 30)
                    + self.terminal(a, b) + self.terminal(c, d))
        asym = min(self.NINIO_MAX, self.NINIO * abs(n1 - n2))
        return (_extrapolate(self.internal_init, m, 30) + asym
                + self.terminal(a, b) + self.terminal(c, d))

    def ml_closing(self, seq, i, j) -> float:
        return self.ML_A + self.ML_B + self.terminal(seq[i], seq[j])

    def ml_branch(self, seq, k, l) -> float:
        return self.ML_B + self.terminal(seq[k], seq[l])

    def ml_unpaired(self) -> float:
        return self.ML_C

    def ext_branch(self, seq, k, l) -> float:
        return self.terminal(seq[k], seq[l])

    def ext_unpaired(self) -> float:
        return 0.0


# ================================================================== independent scorer
def _loop_children(partner: List[int], a: int, b: int):
    """Pairs immediately inside [a,b] and the number of unpaired bases there.

    Plain left-to-right walk over the partner array -- no DP, no recursion tables. This is
    the scoring half of the brute-force reference."""
    ch, unpaired, k = [], 0, a
    while k <= b:
        p = partner[k]
        if p == -1:
            unpaired += 1
            k += 1
        elif p > k:
            if p > b:
                raise ValueError("crossing pair -- structure is not nested")
            ch.append((k, p))
            k = p + 1
        else:
            raise ValueError("crossing pair -- structure is not nested")
    return ch, unpaired


def structure_energy(seq: str, pairs: Iterable[Tuple[int, int]],
                     model: EnergyModel, min_hairpin: int = 3) -> float:
    """Free energy of ONE explicit secondary structure, by direct loop decomposition.

    Returns +inf if the structure is not representable under the model (illegal pair,
    hairpin shorter than min_hairpin, interior loop longer than model.max_loop) so that
    brute-force enumeration and the DP agree on the search space exactly."""
    seq = clean_seq(seq)
    n = len(seq)
    partner = [-1] * n
    for i, j in pairs:
        i, j = int(i), int(j)
        if not (0 <= i < j < n):
            raise ValueError(f"bad pair {(i, j)} for length {n}")
        if partner[i] != -1 or partner[j] != -1:
            raise ValueError(f"base used twice at {(i, j)}")
        partner[i], partner[j] = j, i
    plist = sorted((i, j) for i, j in ((int(a), int(b)) for a, b in pairs))
    for i, j in plist:
        if j - i - 1 < min_hairpin:
            return INF
        if not model.can_pair(seq[i], seq[j]):
            return INF

    E = 0.0
    ch, unp = _loop_children(partner, 0, n - 1)          # exterior loop
    E += model.ext_unpaired() * unp
    for k, l in ch:
        E += model.ext_branch(seq, k, l)
    for i, j in plist:                                    # one loop per base pair
        ch, unp = _loop_children(partner, i + 1, j - 1)
        if not ch:
            E += model.hairpin(seq, i, j)
        elif len(ch) == 1:
            k, l = ch[0]
            if (k - i - 1) + (j - l - 1) > model.max_loop:
                return INF
            E += model.interior(seq, i, j, k, l)
        else:
            E += model.ml_closing(seq, i, j) + model.ml_unpaired() * unp
            for k, l in ch:
                E += model.ml_branch(seq, k, l)
        if E == INF:
            return INF
    return E


def enumerate_structures(seq: str, model: EnergyModel,
                         min_hairpin: int = 3) -> List[Tuple[Tuple[int, int], ...]]:
    """EVERY nested secondary structure of `seq`, listed explicitly.

    Memoised split recursion over intervals: position i is either unpaired or paired with
    some l, recurse on the two sides. It produces structures, not partition functions, and
    shares no code with the McCaskill hypergraph -- that is what makes it an independent
    reference. Exponential; n <= 14 only."""
    seq = clean_seq(seq)
    n = len(seq)
    memo: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], ...]]] = {}

    def rec(i: int, j: int):
        if i >= j:
            return [()]
        key = (i, j)
        hit = memo.get(key)
        if hit is not None:
            return hit
        out = [s for s in rec(i + 1, j)]                 # i unpaired
        for l in range(i + min_hairpin + 1, j + 1):
            if not model.can_pair(seq[i], seq[l]):
                continue
            inner = rec(i + 1, l - 1)
            outer = rec(l + 1, j)
            p = (i, l)
            for a in inner:
                for b in outer:
                    out.append((p,) + a + b)
        memo[key] = out
        return out

    return [tuple(sorted(s)) for s in rec(0, n - 1)]


def brute_force(seq: str, model: EnergyModel, min_hairpin: int = 3) -> dict:
    """logZ, base-pair probabilities and MFE from explicit enumeration. Reference only."""
    seq = clean_seq(seq)
    n = len(seq)
    structs = enumerate_structures(seq, model, min_hairpin)
    kept, energies = [], []
    for s in structs:
        E = structure_energy(seq, s, model, min_hairpin)
        if E < INF:
            kept.append(s)
            energies.append(E)
    if not kept:
        raise ValueError("no representable structure (not even the open chain?)")
    w = np.asarray([-E / RT37 for E in energies])
    m = float(w.max())
    logZ = m + float(np.log(np.exp(w - m).sum()))
    p = np.exp(w - logZ)
    P = np.zeros((n, n))
    for s, pi in zip(kept, p):
        for i, j in s:
            P[i, j] += pi
            P[j, i] += pi
    k = int(np.argmin(energies))
    return {"logZ": logZ, "bpp": P, "mfe": float(energies[k]),
            "mfe_pairs": list(kept[k]), "n_structures": len(kept)}


# ================================================================== the DP hypergraph
_QB, _QM1, _QM, _QE = 0, 1, 2, 3
_KIND_NAME = {_QB: "QB", _QM1: "QM1", _QM: "QM", _QE: "QE"}


@dataclass
class DPGraph:
    """Terms are stored flat: term t belongs to item x iff start[x] <= t < end[x].

    Each term carries an ENERGY e[t] and at most two children (c1, c2; -1 = none).
    Items are created in topological order (children always have a smaller index), which
    is what lets the inside and outside passes be single linear sweeps."""
    n: int
    kind: List[int] = field(default_factory=list)
    ij: List[Tuple[int, int]] = field(default_factory=list)
    start: List[int] = field(default_factory=list)
    end: List[int] = field(default_factory=list)
    e: List[float] = field(default_factory=list)
    c1: List[int] = field(default_factory=list)
    c2: List[int] = field(default_factory=list)
    qb: Dict[Tuple[int, int], int] = field(default_factory=dict)
    root: int = -1

    def n_items(self) -> int:
        return len(self.kind)

    def n_terms(self) -> int:
        return len(self.e)

    def max_children(self) -> int:
        return max((int(a >= 0) + int(b >= 0) for a, b in zip(self.c1, self.c2)),
                   default=0)


def build_graph(seq: str, model: EnergyModel, min_hairpin: int = 3) -> DPGraph:
    """The McCaskill / Zuker recursion, written once, as data.

    QB[i,j]   (i,j) is a pair; everything inside plus the loop it closes.
    QM1[i,j]  exactly one multiloop branch in [i,j], starting AT i, rest unpaired.
    QM[i,j]   one or more multiloop branches in [i,j].
    QE[j]     exterior loop over the prefix of length j.   Z = QE[n].

    Every decomposition is by a UNIQUE landmark (the last exterior branch, the start of the
    last multiloop branch, the single pair inside an interior loop), so each structure is
    generated exactly once -- which is what the enumeration check in verify() confirms."""
    seq = clean_seq(seq)
    n = len(seq)
    L = model.max_loop
    g = DPGraph(n=n)
    kind, ij, start, end = g.kind, g.ij, g.start, g.end
    E, C1, C2 = g.e, g.c1, g.c2

    can = [[False] * n for _ in range(n)]
    for i in range(n):
        si = seq[i]
        for j in range(i + min_hairpin + 1, n):
            can[i][j] = model.can_pair(si, seq[j])

    idx_qb: Dict[Tuple[int, int], int] = g.qb
    idx_qm1: Dict[Tuple[int, int], int] = {}
    idx_qm: Dict[Tuple[int, int], int] = {}
    idx_qe: Dict[int, int] = {}

    def new_item(k, a, b) -> int:
        kind.append(k)
        ij.append((a, b))
        start.append(len(E))
        end.append(len(E))
        return len(kind) - 1

    def add(w, x1=-1, x2=-1):
        E.append(w)
        C1.append(x1)
        C2.append(x2)

    def close(x):
        end[x] = len(E)

    def live(x):
        return x is not None and x >= 0 and end[x] > start[x]

    mlu = model.ml_unpaired()

    for span in range(1, n + 1):
        for i in range(0, n - span + 1):
            j = i + span - 1
            # ---------------------------------------------------------------- QB[i,j]
            if can[i][j]:
                x = new_item(_QB, i, j)
                add(model.hairpin(seq, i, j))
                kmax = min(j - 2, i + 1 + L)
                for k in range(i + 1, kmax + 1):
                    n1 = k - i - 1
                    lmin = max(k + min_hairpin + 1, j - 1 - (L - n1))
                    for l in range(lmin, j):
                        y = idx_qb.get((k, l))
                        if y is None:
                            continue
                        add(model.interior(seq, i, j, k, l), y)
                mlc = model.ml_closing(seq, i, j)
                for h in range(i + 2, j):
                    a = idx_qm.get((i + 1, h - 1))
                    b = idx_qm1.get((h, j - 1))
                    if live(a) and live(b):
                        add(mlc, a, b)
                close(x)
                idx_qb[(i, j)] = x
            # --------------------------------------------------------------- QM1[i,j]
            x = new_item(_QM1, i, j)
            for l in range(i + min_hairpin + 1, j + 1):
                y = idx_qb.get((i, l))
                if y is None:
                    continue
                add(model.ml_branch(seq, i, l) + mlu * (j - l), y)
            close(x)
            idx_qm1[(i, j)] = x
            # ---------------------------------------------------------------- QM[i,j]
            x = new_item(_QM, i, j)
            for h in range(i, j + 1):
                b = idx_qm1.get((h, j))
                if not live(b):
                    continue
                add(mlu * (h - i), b)                      # i..h-1 all unpaired
                if h > i:
                    a = idx_qm.get((i, h - 1))
                    if live(a):
                        add(0.0, a, b)                     # h starts the LAST branch
            close(x)
            idx_qm[(i, j)] = x

    # ------------------------------------------------------------------- exterior loop
    x = new_item(_QE, 0, 0)
    add(0.0)
    close(x)
    idx_qe[0] = x
    eu = model.ext_unpaired()
    for m in range(1, n + 1):
        x = new_item(_QE, 0, m)
        add(eu, idx_qe[m - 1])                             # base m-1 unpaired
        for h in range(0, m - min_hairpin - 1):
            y = idx_qb.get((h, m - 1))
            if y is None:
                continue
            add(model.ext_branch(seq, h, m - 1), idx_qe[h], y)
        close(x)
        idx_qe[m] = x
    g.root = idx_qe[n]
    return g


def graph_info(g: DPGraph) -> dict:
    """Cost bookkeeping. `treewidth` is in the rem.factorgraph sense: the largest
    intermediate table is d^(treewidth+1) with d = n sequence positions."""
    return {"n": g.n, "n_items": g.n_items(), "n_terms": g.n_terms(),
            "max_children_per_term": g.max_children(),
            "max_free_indices_per_term": 3,
            "treewidth": 2, "d": g.n, "cost_exponent": 3,
            "nested_only": True}


# ------------------------------------------------------------------ semiring passes
def _lae(a: float, b: float) -> float:
    if b > a:
        a, b = b, a
    if b == NEG_INF:
        return a
    return a + math.log1p(math.exp(b - a))


def inside_sum(g: DPGraph, RT: float = RT37) -> List[float]:
    """log sum exp(-E/RT) over everything each item derives. Log space throughout."""
    N = g.n_items()
    val = [NEG_INF] * N
    Ee, C1, C2, S, T = g.e, g.c1, g.c2, g.start, g.end
    for x in range(N):
        s0, s1 = S[x], T[x]
        if s1 == s0:
            continue
        best = NEG_INF
        terms = []
        for t in range(s0, s1):
            w = Ee[t]
            if w == INF:
                continue
            v = -w / RT
            a = C1[t]
            if a >= 0:
                va = val[a]
                if va == NEG_INF:
                    continue
                v += va
            b = C2[t]
            if b >= 0:
                vb = val[b]
                if vb == NEG_INF:
                    continue
                v += vb
            terms.append(v)
            if v > best:
                best = v
        if not terms or best == NEG_INF:
            continue
        acc = 0.0
        for v in terms:
            acc += math.exp(v - best)
        val[x] = best + math.log(acc)
    return val


def outside_sum(g: DPGraph, val: List[float], RT: float = RT37) -> List[float]:
    """Generic outside pass over the same hypergraph.

    out[x] = log sum over ways of deriving the root WITH a hole at x, of the weight of
    everything except x's own subderivation. Then P(x) = exp(in[x] + out[x] - logZ).
    Written once and reused, rather than a hand-derived per-quantity outside recursion --
    the usual source of McCaskill bugs."""
    N = g.n_items()
    out = [NEG_INF] * N
    out[g.root] = 0.0
    Ee, C1, C2, S, T = g.e, g.c1, g.c2, g.start, g.end
    for x in range(N - 1, -1, -1):
        ox = out[x]
        if ox == NEG_INF:
            continue
        for t in range(S[x], T[x]):
            w = Ee[t]
            if w == INF:
                continue
            base = ox - w / RT
            a, b = C1[t], C2[t]
            if a >= 0 and b >= 0:
                va, vb = val[a], val[b]
                if va == NEG_INF or vb == NEG_INF:
                    continue
                out[a] = _lae(out[a], base + vb)
                out[b] = _lae(out[b], base + va)
            elif a >= 0:
                if val[a] == NEG_INF:
                    continue
                out[a] = _lae(out[a], base)
    return out


def inside_min(g: DPGraph) -> Tuple[List[float], List[int]]:
    """Min-plus semiring on the SAME hypergraph: minimum total energy, plus a backpointer."""
    N = g.n_items()
    val = [INF] * N
    back = [-1] * N
    Ee, C1, C2, S, T = g.e, g.c1, g.c2, g.start, g.end
    for x in range(N):
        best, bt = INF, -1
        for t in range(S[x], T[x]):
            v = Ee[t]
            if v == INF:
                continue
            a = C1[t]
            if a >= 0:
                va = val[a]
                if va == INF:
                    continue
                v += va
            b = C2[t]
            if b >= 0:
                vb = val[b]
                if vb == INF:
                    continue
                v += vb
            if v < best:
                best, bt = v, t
        val[x], back[x] = best, bt
    return val, back


def traceback_min(g: DPGraph, back: List[int]) -> List[Tuple[int, int]]:
    pairs = []
    stack = [g.root]
    while stack:
        x = stack.pop()
        if g.kind[x] == _QB:
            pairs.append(g.ij[x])
        t = back[x]
        if t < 0:
            continue
        a, b = g.c1[t], g.c2[t]
        if a >= 0:
            stack.append(a)
        if b >= 0:
            stack.append(b)
    return sorted(pairs)


# ================================================================== public API
class MFEResult(NamedTuple):
    energy: float
    structure: str
    pairs: List[Tuple[int, int]]
    info: dict


def mccaskill(seq: str, energy_model: Optional[EnergyModel] = None,
              min_hairpin: int = 3, temperature: float = T37,
              return_info: bool = False):
    """McCaskill partition function.

    Returns (logZ, P) where P[i, j] = P[j, i] = probability that bases i and j are paired,
    marginalising over EVERY nested secondary structure. With return_info=True returns
    (logZ, P, info) where info carries the cost bookkeeping (treewidth, table sizes).

    logZ is the natural log of Z = sum over structures of exp(-E/RT); the ensemble free
    energy is -RT * logZ."""
    seq = clean_seq(seq)
    model = energy_model if energy_model is not None else StackingModel()
    RT = R_KCAL * temperature
    t0 = time.perf_counter()
    g = build_graph(seq, model, min_hairpin)
    val = inside_sum(g, RT)
    logZ = val[g.root]
    out = outside_sum(g, val, RT)
    n = g.n
    P = np.zeros((n, n))
    for (i, j), x in g.qb.items():
        vi, oi = val[x], out[x]
        if vi == NEG_INF or oi == NEG_INF:
            continue
        p = math.exp(vi + oi - logZ)
        P[i, j] = p
        P[j, i] = p
    info = graph_info(g)
    info.update({"model": model.name, "RT": RT, "logZ": logZ,
                 "seconds": time.perf_counter() - t0,
                 "ensemble_free_energy": -RT * logZ})
    if return_info:
        return logZ, P, info
    return logZ, P


def mfe(seq: str, energy_model: Optional[EnergyModel] = None,
        min_hairpin: int = 3) -> MFEResult:
    """Minimum free energy structure -- the min-plus counterpart of `mccaskill`.

    Returns (energy, dot_bracket, pairs, info). Same hypergraph, different semiring."""
    seq = clean_seq(seq)
    model = energy_model if energy_model is not None else StackingModel()
    t0 = time.perf_counter()
    g = build_graph(seq, model, min_hairpin)
    val, back = inside_min(g)
    pairs = traceback_min(g, back)
    info = graph_info(g)
    info.update({"model": model.name, "seconds": time.perf_counter() - t0})
    return MFEResult(float(val[g.root]), pairs_to_db(pairs, g.n), pairs, info)


def probability_of_structure(seq: str, pairs, energy_model=None,
                             min_hairpin: int = 3, temperature: float = T37) -> float:
    """Boltzmann probability of one specific structure: exp(-E/RT - logZ)."""
    model = energy_model if energy_model is not None else StackingModel()
    RT = R_KCAL * temperature
    E = structure_energy(seq, pairs, model, min_hairpin)
    if E == INF:
        return 0.0
    logZ, _ = mccaskill(seq, model, min_hairpin, temperature)
    return math.exp(-E / RT - logZ)


# ================================================================== benchmarks
# Yeast (Saccharomyces cerevisiae) cytoplasmic tRNA-Phe, 76 nt -- the classic RNA
# structure benchmark (Holbrook et al., Nature 1978; PDB 1EHZ, Shi & Moore, RNA 2000).
# Modified bases in the natural molecule (Gm34, yW37, T54, psi55, ...) are written as
# their unmodified parents, which is standard practice for secondary-structure
# benchmarking and is itself a source of error the model cannot see.
YEAST_TRNA_PHE = ("GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUC"
                  "CACAGAAUUCGCACCA")
# The accepted cloverleaf, 21 pairs (1-based):
#   acceptor stem 1-72 .. 7-66      D-arm      10-25 .. 13-22
#   anticodon arm 27-43 .. 31-39    T-arm      49-65 .. 53-61
# with anticodon GAA at 34-36 and the CCA tail at 74-76 unpaired.
YEAST_TRNA_PHE_DB = ("(((((((..((((........)))).(((((.......))))).....(((((.......))"
                     "))))))))))....")

# Synthetic POSITIVE CONTROLS. Each is a designed sequence whose intended fold is written
# next to it; they exist so that "the simple model gets 33% on tRNA" can be distinguished
# from "the pipeline is broken". Strong G-C helices separated by A-rich linkers.
PLANTED = [
    # 6 bp G-C stem, 4 nt loop
    ("hairpin",
     "GGCGCG" "AAAA" "CGCGCC",
     "((((((" "...." "))))))"),
    # two independent hairpins sitting side by side in the exterior loop
    ("two hairpins",
     "GGCGCG" "AAAA" "CGCGCC" "AAAA" "GGGAGG" "AAAA" "CCUCCC",
     "((((((" "...." "))))))" "...." "((((((" "...." "))))))"),
    # a genuine three-way junction: outer stem closing two hairpins -> multiloop
    ("three-way junction",
     "GGCAGC" "AA" "GGCGCG" "AAAA" "CGCGCC" "AA" "GGGAGG" "AAAA" "CCUCCC" "AA" "GCUGCC",
     "((((((" ".." "((((((" "...." "))))))" ".." "((((((" "...." "))))))" ".." "))))))"),
]


def fold_and_score(seq: str, ref_db: str, model: EnergyModel,
                   min_hairpin: int = 3, with_bpp: bool = True) -> dict:
    """Fold with `model`, score the MFE structure against a reference dot-bracket."""
    seq = clean_seq(seq)
    ref = db_to_pairs(ref_db)
    r = mfe(seq, model, min_hairpin)
    sc = compare_structures(r.pairs, ref)
    sc.update({"model": model.name, "mfe": r.energy, "structure": r.structure,
               "energy_of_reference": structure_energy(seq, ref, model, min_hairpin),
               "seconds": r.info["seconds"]})
    if with_bpp:
        logZ, P = mccaskill(seq, model, min_hairpin)
        bpp_ref = [P[i, j] for i, j in ref]
        sc.update({"logZ": logZ,
                   "mean_bpp_of_true_pairs": float(np.mean(bpp_ref)),
                   "min_bpp_of_true_pairs": float(np.min(bpp_ref)),
                   "p_of_reference": math.exp(-sc["energy_of_reference"] / RT37 - logZ)
                   if sc["energy_of_reference"] < INF else 0.0})
    return sc


def trna_model_ladder() -> List[EnergyModel]:
    """The four scoring functions the tRNA benchmark compares. Same DP, same search space
    (max_loop = 30), same min_hairpin -- ONLY the scoring function changes."""
    return [BasePairModel(),
            StackingModel(terminal_mismatch=False, multiloop="turner1999"),
            StackingModel(terminal_mismatch=True, multiloop="turner1999"),
            StackingModel(terminal_mismatch=True, multiloop="turner2004")]


def dinucleotide_shuffle(seq: str, rng: random.Random) -> str:
    """Altschul-Erikson-style shuffle: preserves mononucleotide composition exactly and
    dinucleotide composition approximately. Used as the NEGATIVE control -- if a shuffled
    tRNA also scores well against the cloverleaf, the benchmark is meaningless."""
    s = list(clean_seq(seq))
    rng.shuffle(s)
    return "".join(s)


def benchmark_trna(seq: str = YEAST_TRNA_PHE, ref_db: str = YEAST_TRNA_PHE_DB,
                   verbose: bool = True, sweep: bool = True,
                   n_shuffles: int = 8, seed: int = 0) -> dict:
    """Fold a real tRNA and score against the accepted structure, with and without
    stacking, plus a parameter sweep and a shuffled-sequence null."""
    seq = clean_seq(seq)
    ref = db_to_pairs(ref_db)
    rows = [fold_and_score(seq, ref_db, m) for m in trna_model_ladder()]
    out = {"n": len(seq), "n_ref_pairs": len(ref), "reference": ref_db, "rows": rows}

    if sweep:
        # How much slack is there? Move ONE parameter at a time around the default.
        best = StackingModel(terminal_mismatch=True, multiloop="turner1999")
        sw = {"MM_GC": [], "ML_A": [], "ML_B": []}
        for v in (-0.6, -1.0, -1.2, -1.4, -1.8, -2.2):
            m = StackingModel(); m.MM_GC = v; m.MM_AU = v + 0.5
            sw["MM_GC"].append((v, compare_structures(mfe(seq, m).pairs,
                                                      ref)["sensitivity"]))
        for v in (1.4, 2.4, 3.4, 4.4, 6.4, 9.3):
            m = StackingModel(); m.ML_A = v
            sw["ML_A"].append((v, compare_structures(mfe(seq, m).pairs,
                                                     ref)["sensitivity"]))
        for v in (-0.9, -0.4, 0.0, 0.4, 0.8, 1.2):
            m = StackingModel(); m.ML_B = v
            sw["ML_B"].append((v, compare_structures(mfe(seq, m).pairs,
                                                     ref)["sensitivity"]))
        # INTERNAL37[4] is 1.10 in ViennaRNA's generic Turner-2004 `interior` table, but
        # real 2x2 loops there use the int22 lookup whose mean is nearer 1.9. Without
        # int22 the choice is a genuine modelling fork, so it gets swept like a parameter.
        sw["internal4"] = []
        for v in (0.8, 1.1, 1.4, 1.7, 2.0, 2.3):
            m = StackingModel(); m.internal_init[4] = v
            sw["internal4"].append((v, compare_structures(mfe(seq, m).pairs,
                                                          ref)["sensitivity"]))
        out["sweep"] = sw
        del best

    if n_shuffles:
        rng = random.Random(seed)
        m = StackingModel()
        null = []
        for _ in range(n_shuffles):
            sh = dinucleotide_shuffle(seq, rng)
            null.append(compare_structures(mfe(sh, m).pairs, ref)["sensitivity"])
        out["shuffled_sensitivity"] = {"mean": float(np.mean(null)),
                                       "max": float(np.max(null)), "n": n_shuffles}

    if verbose:
        print(f"  tRNA benchmark: yeast tRNA-Phe, n = {len(seq)}, "
              f"{len(ref)} accepted pairs")
        print(f"    reference           {ref_db}")
        for r in rows:
            print(f"    {r['model']:<26} sens {r['sensitivity']:.3f}  "
                  f"PPV {r['ppv']:.3f}  F1 {r['f1']:.3f}   "
                  f"({r['tp']}/{r['n_ref']} true pairs, {r['n_pred']} predicted)   "
                  f"MFE {r['mfe']:7.2f}   E(ref) {r['energy_of_reference']:7.2f}")
            print(f"    {'':<26} {r['structure']}")
            print(f"    {'':<26} P(reference) {r['p_of_reference']:.3e}   "
                  f"mean P(true pair) {r['mean_bpp_of_true_pairs']:.3f}   "
                  f"min {r['min_bpp_of_true_pairs']:.3f}")
        if sweep:
            print("    one-parameter sweep around stacking+mismatch/turner1999 "
                  "(sensitivity):")
            for k, vals in out["sweep"].items():
                print(f"      {k:<6} " + "  ".join(f"{v:+.2f}->{s:.2f}"
                                                   for v, s in vals))
        if n_shuffles:
            sh = out["shuffled_sensitivity"]
            print(f"    null: {sh['n']} shuffled sequences folded with the same model "
                  f"score sens mean {sh['mean']:.3f}, max {sh['max']:.3f} "
                  f"against the cloverleaf")
    return out


def benchmark_planted(verbose: bool = True) -> dict:
    """POSITIVE CONTROL. Designed sequences with an unambiguous intended fold.

    If the pipeline cannot recover a planted structure, a low tRNA score says nothing
    about the energy model. Both energy models are run."""
    out = []
    for name, seq, db in PLANTED:
        assert len(seq) == len(db), (name, len(seq), len(db))
        for model in (BasePairModel(), StackingModel()):
            r = fold_and_score(seq, db, model, with_bpp=True)
            r["case"] = name
            out.append(r)
            if verbose:
                print(f"    {name:<34} {model.name:<26} "
                      f"sens {r['sensitivity']:.3f} PPV {r['ppv']:.3f}  "
                      f"P(planted) {r['p_of_reference']:.3f}")
    return {"rows": out}


# ================================================================== verification
def _random_seq(rng: random.Random, n: int, gc: float = 0.5) -> str:
    out = []
    for _ in range(n):
        out.append(rng.choice("GC") if rng.random() < gc else rng.choice("AU"))
    return "".join(out)


def _has_multiloop(seq: str, pairs) -> bool:
    n = len(seq)
    partner = [-1] * n
    for i, j in pairs:
        partner[i], partner[j] = j, i
    for i, j in pairs:
        ch, _ = _loop_children(partner, i + 1, j - 1)
        if len(ch) >= 2:
            return True
    return False


# Designed three-way junctions, short enough to enumerate completely. Their point is
# MULTILOOP COVERAGE: a random 12-mer cannot even contain a multiloop (a closing pair plus
# two branches needs >= 12 nt), so without these the multiloop recursion would go untested.
ENUM_CASES = [
    "GCGAAGCAAAAGCAAGCAAAAGCAAGC",
    "GCAAGGCAAAAGCCAAGGCAAAAGCCAAGC",
    "GCGAAGCGAAAACGCAAGCGAAAACGCAACGC",
]


def multiloop_stress_model() -> "StackingModel":
    """A deliberately distorted model (free multiloop closing, bonus per branch) whose
    optima ARE multiloops. Without it the MFE traceback through the multiloop recursion is
    only ever checked on structures the model rejects."""
    m = StackingModel()
    m.ML_A, m.ML_B = 0.0, -0.5
    m.name = "stacking/multiloop-stress"
    return m


def verify(seed: int = 0, n_seqs: int = 8, verbose: bool = True) -> dict:
    """Check McCaskill against EXPLICIT ENUMERATION of every nested structure.

    The reference shares no code path with the DP. `enumerate_structures` lists structures
    by a split recursion; `structure_energy` scores each one by walking its partner array
    and classifying loops; the partition function is then a plain logsumexp over that list
    and the base-pair probabilities are a plain weighted count. The DP instead builds a
    hypergraph of McCaskill recursions and runs generic inside/outside over it. The only
    thing the two share is the EnergyModel, which is the model definition, not the
    algorithm."""
    rng = random.Random(seed)
    models = [BasePairModel(),
              StackingModel(terminal_mismatch=False),
              StackingModel(terminal_mismatch=True),
              StackingModel(terminal_mismatch=True, multiloop="turner2004"),
              multiloop_stress_model()]
    err = {m.name: {"logZ": 0.0, "bpp": 0.0, "mfe": 0.0, "mfe_self": 0.0}
           for m in models}
    n_struct_total, n_ml_struct, lens = 0, 0, []
    n_ml_mfe: Dict[str, int] = {}
    seqs = [_random_seq(rng, rng.randint(9, 14), gc=0.55) for _ in range(n_seqs)]
    seqs += [_random_seq(rng, rng.randint(18, 22), gc=0.55) for _ in range(3)]
    seqs += ENUM_CASES
    for seq in seqs:
        lens.append(len(seq))
        for model in models:
            ref = brute_force(seq, model)
            n_struct_total += ref["n_structures"]
            logZ, P = mccaskill(seq, model)
            e = err[model.name]
            e["logZ"] = max(e["logZ"], abs(logZ - ref["logZ"]))
            e["bpp"] = max(e["bpp"], float(np.max(np.abs(P - ref["bpp"]))))
            r = mfe(seq, model)
            e["mfe"] = max(e["mfe"], abs(r.energy - ref["mfe"]))
            # the structure the traceback returns must actually have the reported energy
            e_self = structure_energy(seq, r.pairs, model)
            e["mfe_self"] = max(e["mfe_self"], abs(e_self - r.energy))
            if _has_multiloop(seq, r.pairs):
                n_ml_mfe[model.name] = n_ml_mfe.get(model.name, 0) + 1
        # coverage: how many enumerated structures contain a genuine multiloop?
        for s in enumerate_structures(seq, models[0]):
            if _has_multiloop(seq, s):
                n_ml_struct += 1

    # ---- cost scaling: the law says n^3. Two measurements, because timing alone is
    # noisy: the SIZE of the hypergraph is exact combinatorics, the time is what it costs.
    scal = []
    model = StackingModel()
    for n in (40, 60, 90, 130):
        s = _random_seq(rng, n, gc=0.5)
        t0 = time.perf_counter()
        gg = build_graph(s, model)
        inside_min(gg)
        scal.append((n, time.perf_counter() - t0, gg.n_terms()))
    logn = np.log([a for a, _, _ in scal])
    slope = float(np.polyfit(logn, np.log([b for _, b, _ in scal]), 1)[0])
    slope_terms = float(np.polyfit(logn, np.log([c for _, _, c in scal]), 1)[0])

    gi = graph_info(build_graph(YEAST_TRNA_PHE, StackingModel()))
    planted = benchmark_planted(verbose=False)
    junction_ok = all(r["sensitivity"] == 1.0 and r["ppv"] == 1.0
                      for r in planted["rows"])
    trna = benchmark_trna(verbose=False)

    res = {"errors": err, "n_sequences": len(seqs),
           "lengths": (min(lens), max(lens)),
           "n_structures_enumerated": n_struct_total,
           "n_multiloop_structures_enumerated": n_ml_struct,
           "n_multiloop_mfe_checked": n_ml_mfe,
           "planted_all_recovered": bool(junction_ok),
           "scaling": scal, "measured_cost_exponent": slope,
           "measured_size_exponent": slope_terms,
           "graph": gi, "trna": trna, "planted": planted}

    if verbose:
        print("  rem.rna.verify")
        print(f"    (a) {len(seqs)} sequences, length {min(lens)}-{max(lens)}, "
              f"{n_struct_total:,} nested structures enumerated "
              f"({n_ml_struct:,} of them contain a multiloop)")
        for m in models:
            e = err[m.name]
            print(f"        {m.name:<26} "
                  f"max|logZ - enum| {e['logZ']:.3e}   "
                  f"max|P(i,j) - enum| {e['bpp']:.3e}")
        print("    (b) MFE vs the minimum over the same enumeration "
              f"({sum(n_ml_mfe.values())} of the checked optima are multiloops: "
              + ", ".join(f"{k} {v}" for k, v in sorted(n_ml_mfe.items())) + ")")
        for m in models:
            e = err[m.name]
            print(f"        {m.name:<26} "
                  f"max|MFE - min over enum| {e['mfe']:.3e}   "
                  f"max|E(returned structure) - MFE| {e['mfe_self']:.3e}")
        print(f"    cost: nested structure -> treewidth {gi['treewidth']} with d = n, "
              f"largest table n^{gi['cost_exponent']} "
              f"({gi['max_free_indices_per_term']} free indices per DP term)")
        print("        measured: "
              + ", ".join(f"n={a} {c:,} terms {b*1e3:.0f} ms" for a, b, c in scal))
        print(f"        log-log slope: hypergraph size n^{slope_terms:.2f}, "
              f"time n^{slope:.2f}   (predicted 3)")
        print(f"        tRNA hypergraph: {gi['n_items']:,} items, "
              f"{gi['n_terms']:,} terms")
        print("    positive control (planted structures, must be recovered):")
        benchmark_planted(verbose=True)
        print("    (c) real benchmark")
        benchmark_trna(verbose=True)
    return res


if __name__ == "__main__":
    verify()
