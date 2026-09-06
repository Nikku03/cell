"""Phylogenetics: Felsenstein pruning IS tree elimination.

THE GOVERNING LAW, and why a phylogeny is the easy case:

    cost = d ** treewidth        d = states per variable (d = 4 for DNA)

A tree has treewidth 1. Every clique in the elimination is a single edge (parent, child),
so the largest intermediate table is d^(treewidth+1) = d^2 = 16 numbers, no matter how many
taxa there are. The likelihood of an alignment is a sum over d^(internal nodes) ancestral
state assignments -- 4^2047 for a 2048-tip tree, a number with 1233 digits -- and bucket
elimination on the tree does it EXACTLY in time linear in the number of tips. That is the
whole content of Felsenstein's 1981 pruning algorithm, and it is elimination on a
treewidth-1 factor graph. Nothing is approximated and nothing is sampled.

REM is not a quantum computer; the saving here comes entirely from the tree structure. Put
a reticulation in (hybridisation, recombination) and the treewidth rises, and the cost
rises with it as d^treewidth exactly as the law says.

TREE REPRESENTATION (explicit nested tuples; a Newick parser is also provided).

    a LEAF          is a str, the taxon name                       "human"
    an INTERNAL node is a tuple of (child_subtree, branch_length) pairs

        (("a", 0.1), ("b", 0.2))                    # cherry, two tips
        (((("a", .1), ("b", .2)), 0.3), ("c", .4))  # ((a,b),c)

    Multifurcations are allowed (a node may have any number of children). Branch lengths
    are measured in EXPECTED SUBSTITUTIONS PER SITE; every model here is normalised so
    that -sum_i pi_i Q_ii = 1.

    The forms are unambiguous: a node is a tuple whose every element is a
    (subtree, number) pair; a pair's second element is a number, never a subtree.

    Nodes are labelled canonically by `_flatten`: leaves keep their taxon names, internal
    nodes get "n0" (the root), "n1", ... in the order a left-to-right depth-first walk
    creates them. Those labels are the FactorGraph variable names and the keys accepted by
    the `branch_lengths` override.

LOG SPACE EVERYWHERE. Pruning in linear probability space underflows: a 2048-tip tree has
log-likelihood about -2800, and exp(-2800) is 0 in float64. `felsenstein_loglik` carries
log partial likelihoods and combines children with logsumexp, so it never underflows.
`verify()` measures exactly where the naive linear-space version dies.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rem.factorgraph import FactorGraph, logsumexp

# ----------------------------------------------------------------------------- alphabet
DNA = "ACGT"
_IDX = {c: i for i, c in enumerate(DNA)}
IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
    "N": "ACGT", "-": "ACGT", "?": "ACGT", "X": "ACGT",
}


def tip_vector(state, d: int = 4) -> np.ndarray:
    """Likelihood vector of an observation at a tip: P(observation | tip state = k).

    Accepts an int index, a single IUPAC character (ambiguity gives a 0/1 vector over the
    compatible bases, gaps and N give all-ones = missing data), or an explicit length-d
    vector."""
    if isinstance(state, (int, np.integer)):
        v = np.zeros(d)
        v[int(state)] = 1.0
        return v
    if isinstance(state, str):
        if len(state) != 1:
            raise ValueError(f"tip state {state!r} is not a single character")
        c = state.upper()
        if c not in IUPAC:
            raise ValueError(f"unknown IUPAC code {state!r}")
        if d != 4:
            raise ValueError("character tip states require d = 4")
        v = np.zeros(4)
        for b in IUPAC[c]:
            v[_IDX[b]] = 1.0
        return v
    v = np.asarray(state, dtype=float)
    if v.shape != (d,):
        raise ValueError(f"tip vector must have shape ({d},), got {v.shape}")
    return v


# ------------------------------------------------------------------- substitution models
def _expm_taylor(A: np.ndarray, terms: int = 40) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring + Taylor series.

    Deliberately naive and completely independent of the closed forms below and of the
    eigen-decomposition used by GTR; verify() uses it as ground truth for every P(t)."""
    A = np.asarray(A, dtype=float)
    nrm = float(np.abs(A).sum(axis=1).max())
    s = 0 if nrm <= 0.5 else int(np.ceil(np.log2(nrm))) + 4
    B = A / (2.0 ** s)
    n = A.shape[0]
    X = np.eye(n)
    term = np.eye(n)
    for k in range(1, terms + 1):
        term = term @ B / k
        X = X + term
    for _ in range(s):
        X = X @ X
    return X


def jukes_cantor(t: float, d: int = 4) -> np.ndarray:
    """JC69 transition matrix P(t), t = expected substitutions per site.

    P[i, j] = Prob(state j at the child | state i at the parent).
    Closed form for the equal-rate model:  P_ii = 1/d + (d-1)/d e,  P_ij = 1/d - 1/d e,
    with e = exp(-d t / (d-1)); for d = 4 that is the textbook exp(-4t/3)."""
    t = float(t)
    if t < 0:
        raise ValueError(f"branch length must be non-negative, got {t}")
    e = math.exp(-d * t / (d - 1.0))
    off = (1.0 - e) / d
    P = np.full((d, d), off)
    np.fill_diagonal(P, off + e)
    return P


def jukes_cantor_Q(d: int = 4) -> np.ndarray:
    """Rate matrix behind jukes_cantor, normalised to one substitution per unit time."""
    Q = np.full((d, d), 1.0 / (d - 1.0))
    np.fill_diagonal(Q, -1.0)
    return Q


def kimura_2p(t: float, kappa: float = 2.0) -> np.ndarray:
    """K80 transition matrix. kappa = transition/transversion rate ratio.

    Order A, C, G, T; purines A,G and pyrimidines C,T, so the transitions are A<->G and
    C<->T. kappa = 1 reduces exactly to Jukes-Cantor (checked in verify())."""
    t = float(t)
    if t < 0:
        raise ValueError(f"branch length must be non-negative, got {t}")
    beta = 1.0 / (kappa + 2.0)              # normalised: 1 substitution per unit time
    alpha = kappa * beta
    e1 = math.exp(-4.0 * beta * t)
    e2 = math.exp(-2.0 * (alpha + beta) * t)
    same = 0.25 + 0.25 * e1 + 0.5 * e2
    ti = 0.25 + 0.25 * e1 - 0.5 * e2
    tv = 0.25 - 0.25 * e1
    P = np.full((4, 4), tv)
    np.fill_diagonal(P, same)
    P[0, 2] = P[2, 0] = P[1, 3] = P[3, 1] = ti
    return P


def kimura_2p_Q(kappa: float = 2.0) -> np.ndarray:
    beta = 1.0 / (kappa + 2.0)
    alpha = kappa * beta
    Q = np.full((4, 4), beta)
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = alpha
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


_GTR_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # AC AG AT CG CT GT


def gtr_Q(exchangeabilities: Sequence[float], pi: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """General time-reversible rate matrix, normalised to one substitution per unit time."""
    S = np.zeros((4, 4))
    ex = list(exchangeabilities)
    if len(ex) != 6:
        raise ValueError("GTR needs 6 exchangeabilities (AC AG AT CG CT GT)")
    for (i, j), s in zip(_GTR_PAIRS, ex):
        if s <= 0:
            raise ValueError("exchangeabilities must be positive")
        S[i, j] = S[j, i] = float(s)
    p = np.asarray(pi, dtype=float)
    if np.any(p <= 0):
        raise ValueError("base frequencies must be positive")
    p = p / p.sum()
    Q = S * p[None, :]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    mu = float(-np.sum(p * np.diag(Q)))
    return Q / mu, p


def _expm_reversible(Q: np.ndarray, pi: np.ndarray, t: float) -> np.ndarray:
    """exp(Qt) for a reversible Q, via the symmetric similarity transform.

    B = diag(sqrt pi) Q diag(1/sqrt pi) is symmetric when pi_i Q_ij = pi_j Q_ji, so one
    eigh gives P(t) for every t. Independent of _expm_taylor, which verify() compares to."""
    r = np.sqrt(pi)
    B = (r[:, None] * Q) / r[None, :]
    B = 0.5 * (B + B.T)                      # kill round-off asymmetry
    w, V = np.linalg.eigh(B)
    P = (V * np.exp(w * float(t))[None, :]) @ V.T
    P = (P / r[:, None]) * r[None, :]
    P = np.clip(P, 0.0, None)
    return P


def gtr(t: float, exchangeabilities: Sequence[float], pi: Sequence[float]) -> np.ndarray:
    Q, p = gtr_Q(exchangeabilities, pi)
    return _expm_reversible(Q, p, t)


@dataclass
class SubstModel:
    """A substitution model: a stationary distribution and t -> P(t)."""
    name: str
    pi: np.ndarray
    P: Callable[[float], np.ndarray]
    Q: Optional[np.ndarray] = None

    def __call__(self, t: float) -> np.ndarray:
        return self.P(t)

    @property
    def d(self) -> int:
        return len(self.pi)


JC69 = SubstModel("JC69", np.full(4, 0.25), jukes_cantor, jukes_cantor_Q())


def K80(kappa: float = 2.0) -> SubstModel:
    return SubstModel(f"K80(kappa={kappa:g})", np.full(4, 0.25),
                      lambda t, k=kappa: kimura_2p(t, k), kimura_2p_Q(kappa))


def GTR(exchangeabilities: Sequence[float], pi: Sequence[float]) -> SubstModel:
    Q, p = gtr_Q(exchangeabilities, pi)
    return SubstModel("GTR", p, lambda t, Q=Q, p=p: _expm_reversible(Q, p, t), Q)


def _as_model(model, pi=None) -> Tuple[Callable[[float], np.ndarray], np.ndarray]:
    if isinstance(model, SubstModel):
        Pfun = model.P
        p = model.pi if pi is None else np.asarray(pi, dtype=float)
    elif callable(model):
        Pfun = model
        if pi is None:
            d = int(np.asarray(Pfun(0.1)).shape[0])
            p = np.full(d, 1.0 / d)
        else:
            p = np.asarray(pi, dtype=float)
    else:
        raise TypeError("model must be a SubstModel or a callable t -> P(t)")
    p = np.asarray(p, dtype=float)
    if abs(float(p.sum()) - 1.0) > 1e-9:
        raise ValueError(f"stationary distribution must sum to 1, got {p.sum()}")
    return Pfun, p


# ------------------------------------------------------------------------ tree structure
def is_leaf(x) -> bool:
    return isinstance(x, str)


def is_node(x) -> bool:
    """Shallow, deliberately NOT recursive: a caterpillar tree of 3000 tips is 3000 deep
    and a recursive predicate would blow the interpreter stack before _flatten (which is
    iterative) ever ran. Deep validation happens inside _flatten's stack walk."""
    return (isinstance(x, (tuple, list)) and len(x) > 0
            and all(isinstance(e, (tuple, list)) and len(e) == 2
                    and isinstance(e[1], (int, float, np.integer, np.floating))
                    and isinstance(e[0], (str, tuple, list)) for e in x))


@dataclass
class FlatTree:
    """A tree flattened to arrays. Node 0 is the root; every parent index is < its
    children's indices, so a single reversed sweep visits children before parents."""
    parent: List[int]
    brlen: np.ndarray
    label: List[str]
    leaf_name: List[Optional[str]]
    leaves: List[int]
    internal: List[int]

    @property
    def n(self) -> int:
        return len(self.parent)

    @property
    def n_tips(self) -> int:
        return len(self.leaves)

    @property
    def n_internal(self) -> int:
        return len(self.internal)


def _flatten(tree, prefix: str = "n") -> FlatTree:
    if not (is_leaf(tree) or is_node(tree)):
        raise ValueError("tree must be a taxon name (str) or a tuple of "
                         "(subtree, branch_length) pairs")
    parent: List[int] = []
    brlen: List[float] = []
    leaf_name: List[Optional[str]] = []
    stack = [(tree, -1, 0.0)]
    while stack:
        sub, par, bl = stack.pop()
        idx = len(parent)
        parent.append(par)
        brlen.append(float(bl))
        if is_leaf(sub):
            leaf_name.append(sub)
        else:
            if not is_node(sub):
                raise ValueError(f"malformed subtree {sub!r}: an internal node must be a "
                                 "tuple of (subtree, branch_length) pairs")
            leaf_name.append(None)
            for child, length in reversed(list(sub)):   # left-to-right exploration
                if float(length) < 0:
                    raise ValueError(f"negative branch length {length}")
                stack.append((child, idx, float(length)))
    leaves = [i for i, nm in enumerate(leaf_name) if nm is not None]
    internal = [i for i, nm in enumerate(leaf_name) if nm is None]
    names = [leaf_name[i] for i in leaves]
    if len(set(names)) != len(names):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"duplicate taxon names: {dup}")
    while any(nm is not None and nm.startswith(prefix) and nm[len(prefix):].isdigit()
              for nm in leaf_name):
        prefix = "_" + prefix
    label: List[Optional[str]] = [None] * len(parent)
    k = 0
    for i in range(len(parent)):
        if leaf_name[i] is None:
            label[i] = f"{prefix}{k}"
            k += 1
        else:
            label[i] = leaf_name[i]
    return FlatTree(parent, np.asarray(brlen, dtype=float), label, leaf_name,
                    leaves, internal)


def node_labels(tree, prefix: str = "n") -> List[str]:
    return _flatten(tree, prefix).label


def taxa(tree) -> List[str]:
    f = _flatten(tree)
    return [f.leaf_name[i] for i in f.leaves]


def _branch_lengths(flat: FlatTree, override, scale: float) -> np.ndarray:
    b = flat.brlen.copy()
    if override is not None:
        if isinstance(override, dict):
            pos = {lab: i for i, lab in enumerate(flat.label)}
            for lab, val in override.items():
                if lab not in pos:
                    raise KeyError(f"no node labelled {lab!r}; labels are {flat.label}")
                if pos[lab] == 0:
                    raise KeyError("the root has no parent branch")
                b[pos[lab]] = float(val)
        else:
            arr = np.asarray(override, dtype=float)
            if arr.shape != (flat.n - 1,):
                raise ValueError(f"branch_lengths needs {flat.n - 1} values "
                                 f"(one per edge), got {arr.shape}")
            b[1:] = arr
    if np.any(b[1:] < 0):
        raise ValueError("negative branch length after override")
    return b * float(scale)


# ---------------------------------------------------------------------------- newick I/O
def _newick_tokens(s: str):
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "(),:;":
            yield c
            i += 1
        elif c.isspace():
            i += 1
        elif c == "'":
            j = s.index("'", i + 1)
            yield ("NAME", s[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and s[j] not in "(),:;" and not s[j].isspace():
                j += 1
            yield ("NAME", s[i:j])
            i = j


def parse_newick(s: str):
    """Parse a Newick string into the nested-tuple representation.

    Iterative (no recursion limit), tolerates whitespace, quoted names and internal node
    labels (which are discarded). Missing branch lengths default to 0."""
    toks = list(_newick_tokens(s))
    stack: List[List] = []
    pending = None
    plen = 0.0
    just_closed = False
    k = 0
    while k < len(toks):
        t = toks[k]
        if t == "(":
            stack.append([])
            pending, plen, just_closed = None, 0.0, False
        elif t == ",":
            if pending is None or not stack:
                raise ValueError("malformed newick: empty subtree before ','")
            stack[-1].append((pending, plen))
            pending, plen, just_closed = None, 0.0, False
        elif t == ")":
            if pending is None or not stack:
                raise ValueError("malformed newick: unbalanced ')'")
            stack[-1].append((pending, plen))
            pending = tuple(stack.pop())
            plen, just_closed = 0.0, True
        elif t == ":":
            k += 1
            if k >= len(toks) or not isinstance(toks[k], tuple):
                raise ValueError("malformed newick: ':' without a length")
            plen = float(toks[k][1])
            just_closed = False
        elif t == ";":
            break
        else:
            if just_closed:
                just_closed = False              # internal node label, ignored
            else:
                pending = t[1]
        k += 1
    if stack:
        raise ValueError("malformed newick: unclosed '('")
    if pending is None:
        raise ValueError("empty newick string")
    return pending


def to_newick(tree, fmt: str = "%.6g") -> str:
    """Write the nested-tuple form back out as Newick (recursive; small trees)."""
    def rec(sub):
        if is_leaf(sub):
            return sub
        return "(" + ",".join(rec(c) + ":" + (fmt % float(l)) for c, l in sub) + ")"
    return rec(tree) + ";"


# ------------------------------------------------------------------------ tree generators
def random_tree(n_tips: int, rng, min_len: float = 0.02, max_len: float = 0.5,
                names: Optional[Sequence[str]] = None):
    """Random rooted binary tree by repeatedly joining two random subtrees."""
    if n_tips < 1:
        raise ValueError("n_tips must be >= 1")
    items = list(names) if names is not None else [f"t{i}" for i in range(n_tips)]
    if len(items) != n_tips:
        raise ValueError("names must have n_tips entries")
    while len(items) > 1:
        i, j = sorted(rng.choice(len(items), size=2, replace=False))
        b = items.pop(j)
        a = items.pop(i)
        items.append(((a, float(rng.uniform(min_len, max_len))),
                      (b, float(rng.uniform(min_len, max_len)))))
    return items[0]


def balanced_tree(n_tips: int, brlen: float = 0.1, rng=None,
                  min_len: float = 0.02, max_len: float = 0.3):
    """Perfectly balanced rooted binary tree; n_tips must be a power of two."""
    if n_tips < 1 or (n_tips & (n_tips - 1)) != 0:
        raise ValueError("balanced_tree needs a power of two")
    level = [f"t{i}" for i in range(n_tips)]
    draw = (lambda: brlen) if rng is None else (lambda: float(rng.uniform(min_len, max_len)))
    while len(level) > 1:
        level = [((level[i], draw()), (level[i + 1], draw()))
                 for i in range(0, len(level), 2)]
    return level[0]


def caterpillar_tree(n_tips: int, brlen: float = 0.1):
    """Fully unbalanced ladder tree -- the worst case for any recursive implementation.
    Everything here is iterative, so this parses and prunes fine at thousands of tips."""
    if n_tips < 2:
        raise ValueError("n_tips must be >= 2")
    t = ((f"t0", brlen), (f"t1", brlen))
    for i in range(2, n_tips):
        t = ((t, brlen), (f"t{i}", brlen))
    return t


# ---------------------------------------------------------------- Felsenstein pruning
def _tip_log_matrix(flat: FlatTree, tip_states, d: int) -> np.ndarray:
    """(n_nodes, n_sites, d) array of LOG tip likelihoods; internal nodes start at log 1."""
    per_tip = {}
    n_sites = None
    for i in flat.leaves:
        nm = flat.leaf_name[i]
        if nm not in tip_states:
            raise KeyError(f"no state given for tip {nm!r}")
        st = tip_states[nm]
        if isinstance(st, str) and len(st) > 1:
            vecs = np.stack([tip_vector(c, d) for c in st])
        elif isinstance(st, np.ndarray) and st.ndim == 2:
            vecs = np.asarray(st, dtype=float)
        else:
            vecs = tip_vector(st, d)[None, :]
        if n_sites is None:
            n_sites = vecs.shape[0]
        elif vecs.shape[0] != n_sites:
            raise ValueError(f"tip {nm!r} has {vecs.shape[0]} sites, expected {n_sites}")
        per_tip[i] = vecs
    L = np.zeros((flat.n, n_sites, d))
    with np.errstate(divide="ignore"):
        for i, v in per_tip.items():
            L[i] = np.log(v)
    return L


def _prune_log(flat: FlatTree, L: np.ndarray, brlen: np.ndarray,
               Pfun, logpi: np.ndarray) -> np.ndarray:
    """The whole algorithm. One reversed sweep, one d x d logsumexp per edge.

    logL[parent][j] += log sum_k P(t)[j, k] exp(logL[child][k])

    which is exactly eliminating the child variable out of the (parent, child) factor in
    the sum-product semiring. Cost O(edges * d^2) = O(n_tips * d^(treewidth+1))."""
    L = L.copy()
    cache: Dict[float, np.ndarray] = {}
    for i in range(flat.n - 1, 0, -1):
        t = float(brlen[i])
        logP = cache.get(t)
        if logP is None:
            with np.errstate(divide="ignore"):
                logP = np.log(np.asarray(Pfun(t), dtype=float))
            cache[t] = logP
        msg = logsumexp(logP[None, :, :] + L[i][:, None, :], axis=2)   # (sites, d)
        L[flat.parent[i]] += msg
    return logsumexp(logpi[None, :] + L[0], axis=1)


def felsenstein_site_logliks(tree, tip_states, model=JC69, branch_lengths=None,
                             pi=None, scale: float = 1.0,
                             prefix: str = "n") -> np.ndarray:
    """Per-site log likelihoods. tip_states maps taxon -> character / int / string /
    (n_sites, d) likelihood matrix."""
    flat = _flatten(tree, prefix)
    if flat.n_internal == 0:
        raise ValueError("a single-tip tree has no internal nodes")
    Pfun, p = _as_model(model, pi)
    d = len(p)
    b = _branch_lengths(flat, branch_lengths, scale)
    L = _tip_log_matrix(flat, tip_states, d)
    with np.errstate(divide="ignore"):
        logpi = np.log(p)
    return _prune_log(flat, L, b, Pfun, logpi)


def felsenstein_loglik(tree, tip_states, model=JC69, branch_lengths=None,
                       pi=None, scale: float = 1.0, prefix: str = "n") -> float:
    """Exact log likelihood of the data at the tips, summed over ALL d^(internal nodes)
    ancestral state assignments, in time linear in the number of tips.

    tree            nested tuples (see the module docstring) or parse_newick output
    tip_states      {taxon: state}; a state is an int, an IUPAC character, a string of
                    characters (an alignment: the per-site log likelihoods are summed),
                    or an explicit likelihood vector / matrix
    model           SubstModel (JC69, K80(kappa), GTR(...)) or a bare callable t -> P(t)
    branch_lengths  optional override: {node_label: length} or a length-(n_nodes-1)
                    sequence in flatten order. None uses the lengths stored in the tree.
    scale           multiplies every branch length (used to fit overall tree length)
    """
    return float(np.sum(felsenstein_site_logliks(
        tree, tip_states, model, branch_lengths, pi, scale, prefix)))


def alignment_loglik(tree, alignment: Dict[str, str], model=JC69, branch_lengths=None,
                     pi=None, scale: float = 1.0, prefix: str = "n") -> float:
    """Total log likelihood of an alignment {taxon: "ACGT..."} (sites independent)."""
    return felsenstein_loglik(tree, alignment, model, branch_lengths, pi, scale, prefix)


def _prune_linear_naive(tree, tip_states, model=JC69, scale: float = 1.0,
                        pi=None, prefix: str = "n") -> float:
    """Pruning in LINEAR probability space -- the textbook version, kept only so that
    verify() can measure the tree size at which it underflows to 0. Do not use."""
    flat = _flatten(tree, prefix)
    Pfun, p = _as_model(model, pi)
    d = len(p)
    b = _branch_lengths(flat, None, scale)
    L = np.ones((flat.n, d))
    for i in flat.leaves:
        L[i] = tip_vector(tip_states[flat.leaf_name[i]], d)
    for i in range(flat.n - 1, 0, -1):
        L[flat.parent[i]] *= np.asarray(Pfun(float(b[i]))) @ L[i]
    return float(p @ L[0])


# ------------------------------------------------------------- the same thing, as REM
def to_factorgraph(tree, tip_states, model=JC69, branch_lengths=None, pi=None,
                   scale: float = 1.0, prefix: str = "n") -> FactorGraph:
    """Build the factor graph whose eliminate("sum") IS the Felsenstein likelihood.

    One variable per node with d states. One pairwise factor log P(t)[parent, child] per
    edge, one unary log pi at the root, one unary log(tip likelihood) per tip (an observed
    base is a -inf clamp on the three other states). The graph is a tree, so its treewidth
    is 1 and elimination costs d^2 per edge. Single site only -- for an alignment the
    graph is the same and only the tip clamps change."""
    flat = _flatten(tree, prefix)
    Pfun, p = _as_model(model, pi)
    d = len(p)
    b = _branch_lengths(flat, branch_lengths, scale)
    g = FactorGraph()
    for lab in flat.label:
        g.add_var(lab, d)
    with np.errstate(divide="ignore"):
        g.add_factor([flat.label[0]], np.log(p))
        for i in range(1, flat.n):
            g.add_factor([flat.label[flat.parent[i]], flat.label[i]],
                         np.log(np.asarray(Pfun(float(b[i])), dtype=float)))
        for i in flat.leaves:
            g.add_factor([flat.label[i]],
                         np.log(tip_vector(tip_states[flat.leaf_name[i]], d)))
    return g


def factorgraph_loglik(tree, tip_states, model=JC69, branch_lengths=None, pi=None,
                       scale: float = 1.0, prefix: str = "n") -> Tuple[float, dict]:
    """Log likelihood via rem.FactorGraph.eliminate("sum"). Returns (loglik, info);
    info["treewidth"] is 1 for any tree, and info["largest_table"] is d^2."""
    g = to_factorgraph(tree, tip_states, model, branch_lengths, pi, scale, prefix)
    val, _, info = g.eliminate("sum")
    return float(val), info


def tree_info(tree, d: int = 4, prefix: str = "n") -> dict:
    """Structural report: tips, internal nodes, treewidth, and the size of the sum that
    pruning does not have to enumerate."""
    flat = _flatten(tree, prefix)
    g = FactorGraph()
    for lab in flat.label:
        g.add_var(lab, d)
    for i in range(1, flat.n):
        g.add_factor([flat.label[flat.parent[i]], flat.label[i]], np.zeros((d, d)))
    tw = g.treewidth()
    return {"n_tips": flat.n_tips, "n_internal": flat.n_internal, "n_nodes": flat.n,
            "n_edges": flat.n - 1, "treewidth": int(tw), "d": d,
            "clique_table": d ** (int(tw) + 1),
            "search_space_log10": flat.n_internal * math.log10(d),
            "max_depth": _max_depth(flat)}


def _max_depth(flat: FlatTree) -> int:
    depth = [0] * flat.n
    for i in range(1, flat.n):
        depth[i] = depth[flat.parent[i]] + 1
    return max(depth)


# ------------------------------------------------------------------------ brute force
def brute_force_loglik(tree, tip_states, model=JC69, pi=None) -> float:
    """REFERENCE. Explicit summation over all d^(internal nodes) ancestral assignments.

    Genuinely independent of everything above: it builds its own edge list by direct
    recursion on the nested tuples (no _flatten, no FlatTree), multiplies probabilities in
    LINEAR space (no logsumexp), and enumerates itertools.product over the internal nodes.
    Nothing it calls is on the pruning code path except tip_vector, which only decodes the
    input characters. Exponential -- verification only."""
    Pfun, p = _as_model(model, pi)
    d = len(p)
    edges: List[Tuple[Tuple[str, object], Tuple[str, object], float]] = []
    internals: List[Tuple[str, int]] = []

    def walk(sub):
        if is_leaf(sub):
            return ("tip", sub)
        key = ("int", len(internals))
        internals.append(key)
        for child, length in sub:
            ckey = walk(child)
            edges.append((key, ckey, float(length)))
        return key

    root = walk(tree)
    if root[0] != "int":
        raise ValueError("a single-tip tree has no internal nodes")
    Pc: Dict[float, np.ndarray] = {}
    tipvec = {}
    total = 0.0
    for _, ckey, _ in edges:
        if ckey[0] == "tip":
            tipvec[ckey[1]] = tip_vector(tip_states[ckey[1]], d)
    for combo in itertools.product(range(d), repeat=len(internals)):
        st = {key: combo[i] for i, key in enumerate(internals)}
        prob = float(p[st[root]])
        for a, b, length in edges:
            M = Pc.get(length)
            if M is None:
                M = np.asarray(Pfun(length), dtype=float)
                Pc[length] = M
            if b[0] == "int":
                prob *= float(M[st[a], st[b]])
            else:
                prob *= float(M[st[a]] @ tipvec[b[1]])
            if prob == 0.0:
                break
        total += prob
    return math.log(total) if total > 0 else -math.inf


# ------------------------------------------------------------------------- simulation
def simulate_alignment(tree, model=JC69, n_sites: int = 100, rng=None, pi=None,
                       scale: float = 1.0, prefix: str = "n") -> Dict[str, str]:
    """Sample sequences down the tree: root from pi, then each child from P(t)[parent]."""
    if rng is None:
        rng = np.random.default_rng(0)
    flat = _flatten(tree, prefix)
    Pfun, p = _as_model(model, pi)
    d = len(p)
    b = _branch_lengths(flat, None, scale)
    states = np.zeros((flat.n, n_sites), dtype=int)
    states[0] = rng.choice(d, size=n_sites, p=p)
    for i in range(1, flat.n):
        P = np.asarray(Pfun(float(b[i])), dtype=float)
        cum = np.cumsum(P, axis=1)
        u = rng.random(n_sites)
        parent_state = states[flat.parent[i]]
        states[i] = (u[:, None] > cum[parent_state]).sum(axis=1)
    if d != 4:
        raise ValueError("simulate_alignment writes DNA characters; needs d = 4")
    return {flat.leaf_name[i]: "".join(DNA[s] for s in states[i]) for i in flat.leaves}


def _golden_section(f, a: float, b: float, tol: float = 1e-4, maxit: int = 200) -> float:
    """Minimise a unimodal f on [a, b]. No scipy dependency."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, dd = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(dd)
    for _ in range(maxit):
        if b - a < tol:
            break
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, dd, fd
            dd = a + gr * (b - a)
            fd = f(dd)
    return 0.5 * (a + b)


def fit_branch_scale(tree, alignment, model=JC69, pi=None, lo: float = 0.02,
                     hi: float = 12.0, tol: float = 1e-4) -> Tuple[float, float]:
    """ML estimate of a single multiplier on every branch length. Returns (scale, loglik)."""
    def neg(s):
        return -alignment_loglik(tree, alignment, model, pi=pi, scale=s)
    s = _golden_section(neg, lo, hi, tol=tol)
    return float(s), float(-neg(s))


# ---------------------------------------------------------------------------- verify
def _random_tip_states(rng, names, d=4):
    return {nm: int(rng.integers(0, d)) for nm in names}


def _models_for_verify(rng):
    ex = list(rng.uniform(0.4, 2.5, size=6))
    p = rng.uniform(0.15, 0.35, size=4)
    p = p / p.sum()
    return [("JC69", JC69), ("K80(2.7)", K80(2.7)), ("GTR", GTR(ex, p))]


def verify(seed: int = 0, verbose: bool = True) -> dict:
    """Check Felsenstein pruning against brute force, closed forms and REM elimination."""
    rng = np.random.default_rng(seed)
    out: dict = {}

    # ---- 0. transition matrices vs an independent matrix exponential -----------------
    e_expm, e_row, e_stat, e_ck, e_jc = 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(12):
        t = float(rng.uniform(0.001, 3.0))
        kappa = float(rng.uniform(0.5, 6.0))
        ex = list(rng.uniform(0.3, 3.0, size=6))
        p = rng.uniform(0.15, 0.35, size=4)
        p = p / p.sum()
        Qg, pg = gtr_Q(ex, p)
        for P, Q, stat in ((jukes_cantor(t), jukes_cantor_Q(), np.full(4, 0.25)),
                           (kimura_2p(t, kappa), kimura_2p_Q(kappa), np.full(4, 0.25)),
                           (gtr(t, ex, p), Qg, pg)):
            ref = _expm_taylor(Q * t)
            e_expm = max(e_expm, float(np.max(np.abs(P - ref))))
            e_row = max(e_row, float(np.max(np.abs(P.sum(axis=1) - 1.0))))
            e_stat = max(e_stat, float(np.max(np.abs(stat @ P - stat))))
        # Chapman-Kolmogorov P(s)P(t) = P(s+t), a property no closed form is fitted to
        s = float(rng.uniform(0.001, 2.0))
        e_ck = max(e_ck, float(np.max(np.abs(
            kimura_2p(s, kappa) @ kimura_2p(t, kappa) - kimura_2p(s + t, kappa)))))
        e_ck = max(e_ck, float(np.max(np.abs(
            gtr(s, ex, p) @ gtr(t, ex, p) - gtr(s + t, ex, p)))))
        e_jc = max(e_jc, float(np.max(np.abs(kimura_2p(t, 1.0) - jukes_cantor(t)))))
    out.update(max_err_expm=e_expm, max_err_rowsum=e_row, max_err_stationary=e_stat,
               max_err_chapman_kolmogorov=e_ck, max_err_k80_kappa1_is_jc=e_jc)

    # ---- (a) pruning vs brute force over 4^(internal nodes) --------------------------
    e_bf, e_fg, e_lin = 0.0, 0.0, 0.0
    widths, tables, spaces = set(), set(), []
    rows = []
    for trial in range(24):
        n_tips = int(rng.integers(3, 8))                 # <= 7 tips  ->  <= 6 internal
        tree = random_tree(n_tips, rng)
        names = taxa(tree)
        st = _random_tip_states(rng, names)
        mname, model = _models_for_verify(rng)[int(rng.integers(0, 3))]
        info = tree_info(tree)
        widths.add(info["treewidth"])
        tables.add(info["clique_table"])
        spaces.append(info["n_internal"])
        got = felsenstein_loglik(tree, st, model)
        ref = brute_force_loglik(tree, st, model)
        e_bf = max(e_bf, abs(got - ref))
        fg, fginfo = factorgraph_loglik(tree, st, model)
        e_fg = max(e_fg, abs(got - fg))
        widths.add(fginfo["treewidth"])
        tables.add(fginfo["largest_table"])
        lin = _prune_linear_naive(tree, st, model)
        e_lin = max(e_lin, abs(got - math.log(lin)))
        rows.append((n_tips, mname, got, ref))
    out.update(max_err_vs_bruteforce=e_bf, max_err_factorgraph=e_fg,
               max_err_linear_pruning=e_lin,
               treewidths=sorted(widths), largest_tables=sorted(tables),
               max_internal_nodes=max(spaces))

    # ---- ambiguity / missing data ----------------------------------------------------
    # An N at a tip must equal the sum of the four resolved likelihoods, and R = A + G.
    # Looped, because on any single instance the two float paths often land on bit-
    # identical values and a lone 0.0 would say nothing.
    e_amb = 0.0
    for _ in range(20):
        n_tips = int(rng.integers(3, 7))
        tree = random_tree(n_tips, rng)
        names = taxa(tree)
        st = _random_tip_states(rng, names)
        model = _models_for_verify(rng)[int(rng.integers(0, 3))][1]
        target = names[int(rng.integers(0, n_tips))]
        parts = [felsenstein_loglik(tree, dict(st, **{target: k}), model)
                 for k in range(4)]
        e_amb = max(e_amb, abs(felsenstein_loglik(tree, dict(st, **{target: "N"}), model)
                               - float(logsumexp(np.array(parts), 0))))
        e_amb = max(e_amb, abs(felsenstein_loglik(tree, dict(st, **{target: "R"}), model)
                               - float(logsumexp(np.array([parts[0], parts[2]]), 0))))
    out["max_err_ambiguity"] = e_amb

    # ---- normalisation: sum over ALL 4^tips data patterns must be exactly 1 ----------
    e_norm = 0.0
    for _ in range(4):
        n_tips = int(rng.integers(3, 6))
        tree = random_tree(n_tips, rng)
        names = taxa(tree)
        model = _models_for_verify(rng)[int(rng.integers(0, 3))][1]
        tot = 0.0
        for pat in itertools.product(range(4), repeat=n_tips):
            tot += math.exp(felsenstein_loglik(tree, dict(zip(names, pat)), model))
        e_norm = max(e_norm, abs(tot - 1.0))
    out["max_err_normalisation"] = e_norm

    # ---- two-tip analytic identity ---------------------------------------------------
    # Root a cherry: L = sum_r pi_r P(t1)[r,a] P(t2)[r,b]. Reversibility collapses this to
    # pi_a P(t1+t2)[a,b] -- pure algebra, no summation code at all.
    e_two = 0.0
    for _ in range(8):
        t1, t2 = float(rng.uniform(0.01, 1.5)), float(rng.uniform(0.01, 1.5))
        a, b = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        kappa = float(rng.uniform(0.5, 5.0))
        m = K80(kappa)
        got = felsenstein_loglik((("x", t1), ("y", t2)), {"x": a, "y": b}, m)
        ref = math.log(0.25 * kimura_2p(t1 + t2, kappa)[a, b])
        e_two = max(e_two, abs(got - ref))
    out["max_err_two_tip_analytic"] = e_two

    # ---- root position must not matter for a reversible model ------------------------
    # Same UNROOTED tree (a,b | c,d joined by an internal branch L), rooted at u, at v,
    # and at five points along the middle branch. Every likelihood must be identical.
    e_root = 0.0
    for _ in range(6):
        t1, t2, t3, t4, L = [float(rng.uniform(0.02, 0.9)) for _ in range(5)]
        u_sub = (("a", t1), ("b", t2))
        v_sub = (("c", t3), ("d", t4))
        st = {k: int(rng.integers(0, 4)) for k in "abcd"}
        model = _models_for_verify(rng)[int(rng.integers(0, 3))][1]
        vals = [felsenstein_loglik(((u_sub, f * L), (v_sub, (1 - f) * L)), st, model)
                for f in (0.0, 0.25, 0.5, 0.77, 1.0)]
        vals.append(felsenstein_loglik((("a", t1), ("b", t2), (v_sub, L)), st, model))
        vals.append(felsenstein_loglik((("c", t3), ("d", t4), (u_sub, L)), st, model))
        e_root = max(e_root, max(vals) - min(vals))
    out["max_err_rerooting"] = e_root

    # ---- (c) measured scaling, and where linear space dies ---------------------------
    scale_rows = []
    rng2 = np.random.default_rng(seed + 99)
    for n_tips in (16, 32, 64, 128, 256, 512, 1024, 2048):
        tree = balanced_tree(n_tips, rng=rng2)
        names = taxa(tree)
        st = _random_tip_states(rng2, names)
        info = tree_info(tree)
        reps = max(1, int(4000 // n_tips))
        t0 = time.perf_counter()
        for _ in range(reps):
            ll = felsenstein_loglik(tree, st, JC69)
        dt = (time.perf_counter() - t0) / reps
        fg_ms = float("nan")
        if n_tips <= 256:
            t0 = time.perf_counter()
            fgv, fginfo = factorgraph_loglik(tree, st, JC69)
            fg_ms = (time.perf_counter() - t0) * 1e3
            if abs(fgv - ll) > 1e-9:
                raise AssertionError(f"factorgraph disagrees at {n_tips} tips")
        lin = _prune_linear_naive(tree, st, JC69)
        scale_rows.append({"n_tips": n_tips, "n_internal": info["n_internal"],
                           "treewidth": info["treewidth"],
                           "log10_search_space": info["search_space_log10"],
                           "ms": dt * 1e3, "us_per_tip": dt * 1e6 / n_tips,
                           "fg_ms": fg_ms, "loglik": ll,
                           "linear_space_value": lin})
    ns = np.array([r["n_tips"] for r in scale_rows], dtype=float)
    ts = np.array([r["ms"] for r in scale_rows], dtype=float)
    slope = float(np.polyfit(np.log(ns), np.log(ts), 1)[0])
    per_tip = np.array([r["us_per_tip"] for r in scale_rows])
    out["scaling"] = scale_rows
    out["scaling_exponent"] = slope
    out["us_per_tip_spread"] = float(per_tip.max() / per_tip.min())
    first_underflow = next((r["n_tips"] for r in scale_rows
                            if r["linear_space_value"] == 0.0), None)
    out["linear_space_underflows_at_tips"] = first_underflow

    if verbose:
        print("  rem.phylo.verify   Felsenstein pruning = elimination on a treewidth-1 graph")
        print(f"    substitution models vs independent Taylor expm      {e_expm:.3e}")
        print(f"    row sums of P(t) minus 1                            {e_row:.3e}")
        print(f"    stationarity  pi P(t) - pi                          {e_stat:.3e}")
        print(f"    Chapman-Kolmogorov P(s)P(t) - P(s+t)                {e_ck:.3e}")
        print(f"    K80(kappa=1) - JC69                                 {e_jc:.3e}")
        print(f"  (a) 24 random trees, 3-7 tips, up to "
              f"4^{out['max_internal_nodes']} = "
              f"{4 ** out['max_internal_nodes']:,} ancestral assignments")
        print(f"    max |pruning - BRUTE FORCE enumeration|             {e_bf:.3e}")
        print(f"    max |pruning - linear-space pruning|                {e_lin:.3e}")
        print(f"  (b) same instances as a rem.FactorGraph, eliminate('sum')")
        print(f"    max |pruning - FactorGraph elimination|             {e_fg:.3e}")
        print(f"    treewidths seen {sorted(widths)}   largest table "
              f"{sorted(tables)}  (d^(tw+1) = 16)")
        print(f"    max |N at a tip - sum of 4 resolved|, 20 trees      {e_amb:.3e}")
        print(f"    max |sum over all 4^tips data patterns - 1|         {e_norm:.3e}")
        print(f"    max |cherry likelihood - pi_a P(t1+t2)[a,b]|        {e_two:.3e}")
        print(f"    max spread over 7 root placements (reversibility)   {e_root:.3e}")
        print("  (c) measured scaling, balanced trees, 1 site, JC69")
        print("      tips  internal  tw   4^internal      pruning ms  us/tip   "
              "FactorGraph ms   linear-space value")
        for r in scale_rows:
            fg = "     -" if math.isnan(r["fg_ms"]) else f"{r['fg_ms']:8.2f}"
            print(f"      {r['n_tips']:5d} {r['n_internal']:8d} {r['treewidth']:3d}   "
                  f"10^{r['log10_search_space']:8.1f}   {r['ms']:9.3f}  "
                  f"{r['us_per_tip']:6.2f}   {fg}         "
                  f"{r['linear_space_value']:.3e}")
        print(f"    fitted log-log slope of time vs tips  {slope:.3f}   (1.000 = linear)")
        print(f"    us/tip spread over the table          {out['us_per_tip_spread']:.2f}x")
        print(f"    naive LINEAR-space pruning underflows to 0.0 at "
              f"{first_underflow} tips; log space stays finite "
              f"(loglik {scale_rows[-1]['loglik']:.1f} at 2048 tips)")
    return out


def verify_recovers_planted_scale(seed: int = 0, n_tips: int = 12, n_sites: int = 600,
                                  true_scale: float = 1.0, verbose: bool = True) -> dict:
    """POSITIVE CONTROL. Simulate an alignment down a known tree at a known branch scale,
    then recover that scale by maximising the pruning likelihood. The NEGATIVE control is
    the same pipeline on i.i.d. random sequences, which carry no phylogenetic signal and
    must push the estimate to the top of the bracket."""
    rng = np.random.default_rng(seed)
    tree = random_tree(n_tips, rng)
    names = taxa(tree)
    info = tree_info(tree)
    aln = simulate_alignment(tree, JC69, n_sites, rng, scale=true_scale)
    hi = 12.0
    s_hat, ll_hat = fit_branch_scale(tree, aln, JC69, hi=hi)
    ll_true = alignment_loglik(tree, aln, JC69, scale=true_scale)

    noise = {nm: "".join(DNA[k] for k in rng.integers(0, 4, size=n_sites)) for nm in names}
    s_null, _ = fit_branch_scale(tree, noise, JC69, hi=hi)

    res = {"n_tips": n_tips, "n_sites": n_sites, "treewidth": info["treewidth"],
           "true_scale": true_scale, "fitted_scale": s_hat,
           "rel_error": abs(s_hat - true_scale) / true_scale,
           "loglik_at_fit": ll_hat, "loglik_at_truth": ll_true,
           "fitted_scale_on_noise": s_null, "noise_bracket_top": hi}
    if verbose:
        print(f"  rem.phylo positive control: {n_tips} tips, {n_sites} sites, "
              f"treewidth {info['treewidth']}")
        print(f"    planted branch scale {true_scale:.3f}  ->  ML estimate "
              f"{s_hat:.4f}   ({100 * res['rel_error']:.1f}% off)")
        print(f"    loglik at the fit {ll_hat:.3f}  vs at the truth {ll_true:.3f} "
              f"(fit must be >=)")
        print(f"    NEGATIVE control, i.i.d. random sequences: fitted scale "
              f"{s_null:.3f} against a bracket top of {hi:.1f} "
              f"-- no signal, so the tree is stretched to saturation")
    return res


if __name__ == "__main__":
    verify()
    print()
    verify_recovers_planted_scale()
