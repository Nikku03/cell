"""HP-lattice folding against full enumeration of self-avoiding walks.

The reference in every accuracy test is `rem.hp.enumerate_bruteforce`, which LISTS every
self-avoiding walk by naive DFS (`_walks`) and SCORES each one with `hp_energy`, an O(n^2)
scan over all residue pairs. It has no bound, no incremental contact counter, no move
ordering and no symmetry reduction; the branch-and-bound search shares none of that code.
The one thing in common is `hp_energy`, which is the DEFINITION of the model rather than
an algorithm -- if the two sides did not share it they would be scoring different physics.

Two further audits do not trust either side:
  * test_bound_is_admissible enumerates EVERY self-avoiding prefix and EVERY completion of
    it, and demands the pruning bound never fall below the contacts actually still to be
    gained. That is the property whose failure would let branch and bound silently lose
    the optimum.
  * test_hand_counted_energy and test_all_H_square_optimum_is_hand_derivable pin numbers
    that were worked out by hand, so a shared bug in hp_energy would still be caught.
"""
import os
import random

import pytest

from rem import hp
from rem.hp import (fold_hp, fold_hp_full, enumerate_bruteforce, enumerate_bruteforce_full,
                    hp_energy, validate_conformation, structure_info,
                    max_contacts_upper_bound, verify_bound_admissible,
                    verify_bound_matches_search)


# --------------------------------------------------------------------------- benchmarks
# The classic 2D HP benchmark sequences of Unger & Moult, "Genetic algorithms for protein
# folding simulations", J. Mol. Biol. 231:75-81 (1993), Table 1 -- the same three
# sequences used as the standard yardstick by Yue & Dill (CHCC, PNAS 92:146, 1995),
# Beutler & Dill (Protein Sci. 5:2037, 1996), Lesh, Mitzenmacher & Whitesides (RECOMB
# 2003), Cebrian et al. (2008) and essentially every HP paper since. The energies are the
# published optima for the SQUARE lattice, -1 per non-bonded H-H contact.
BENCH_20 = ("HPHPPHHPHPPHPHHPPHPH", -9)              # (HP)2PH(HP)2(PH)2HP(PH)2
BENCH_24 = ("HHPPHPPHPPHPPHPPHPPHPPHH", -9)          # H2P2(HP2)6H2
BENCH_25 = ("PPHPPHHPPPPHHPPPPHHPPPPHH", -8)         # P2HP2(H2P4)3H2
BENCH_36 = ("PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", -14)  # P3H2P2H2P5H7P2H2P4H2P2HP2

SLOW = bool(os.environ.get("REM_SLOW"))


def _rand_seq(rng, n):
    return "".join(rng.choice("HP") for _ in range(n))


# ------------------------------------------------------ (a) exactness vs full enumeration
@pytest.mark.parametrize("seed", range(14))
def test_square_matches_enumeration(seed):
    rng = random.Random(1000 + seed)
    seq = _rand_seq(rng, rng.randint(5, 10))
    e_bb, conf = fold_hp(seq, "square")
    e_bf, conf_bf = enumerate_bruteforce(seq, "square")
    assert e_bb == e_bf, f"{seq}: branch-and-bound {e_bb} != enumeration {e_bf}"
    assert validate_conformation(seq, conf, "square", e_bb)["ok"]
    assert validate_conformation(seq, conf_bf, "square", e_bf)["ok"]


@pytest.mark.parametrize("seed", range(8))
def test_cubic_matches_enumeration(seed):
    rng = random.Random(2000 + seed)
    seq = _rand_seq(rng, rng.randint(4, 7))
    e_bb, conf = fold_hp(seq, "cubic")
    e_bf, _ = enumerate_bruteforce(seq, "cubic")
    assert e_bb == e_bf, f"{seq}: branch-and-bound {e_bb} != enumeration {e_bf}"
    assert validate_conformation(seq, conf, "cubic", e_bb)["ok"]


@pytest.mark.parametrize("seq", ["HHHHHHHHH", "PPPPPPPP", "HPHPHPHPHP", "HHPPHHPPHH",
                                 "PHHPPHHPPH", "HHHPPPHHHP", "H", "HH", "HP", "HHH"])
def test_designed_sequences_match_enumeration(seq):
    e_bb, conf = fold_hp(seq, "square")
    e_bf, _ = enumerate_bruteforce(seq, "square")
    assert e_bb == e_bf
    assert validate_conformation(seq, conf, "square", e_bb)["ok"]


def test_enumeration_symmetry_reduction_is_energy_preserving():
    """Fixing the first step is a global rotation, so it must not change the optimum."""
    for seq in ["HPHPPHHPHP", "HHPPHHPPHH", "PHHPHPPHHP"]:
        a = enumerate_bruteforce(seq, "square", fix_first_step=False)[0]
        b = enumerate_bruteforce(seq, "square", fix_first_step=True)[0]
        assert a == b


def test_walk_counts_are_the_known_saw_numbers():
    """Guards the enumerator itself: the number of self-avoiding walks on Z^2 with
    n steps is 4, 12, 36, 100, 284, 780, 2172 (OEIS A001411)."""
    known = {1: 4, 2: 12, 3: 36, 4: 100, 5: 284, 6: 780, 7: 2172}
    for steps, want in known.items():
        got = enumerate_bruteforce_full("H" * (steps + 1), "square")["n_walks"]
        assert got == want, f"{steps} steps: enumerated {got}, known {want}"
    # Z^3: 6, 30, 150, 726, 3534 (OEIS A001412)
    for steps, want in {1: 6, 2: 30, 3: 150, 4: 726, 5: 3534}.items():
        got = enumerate_bruteforce_full("H" * (steps + 1), "cubic")["n_walks"]
        assert got == want


# ------------------------------------------------------------- (b) published benchmarks
@pytest.mark.parametrize("seq,published,label", [
    (BENCH_20[0], BENCH_20[1], "20-mer"),
    (BENCH_24[0], BENCH_24[1], "24-mer"),
    (BENCH_25[0], BENCH_25[1], "25-mer"),
])
def test_published_2d_benchmarks(seq, published, label):
    r = fold_hp_full(seq, "square")
    assert r["proved_optimal"], f"{label}: search did not run to completion"
    assert r["energy"] == published, (
        f"{label}: branch-and-bound proved {r['energy']}, published optimum {published}")
    chk = validate_conformation(seq, r["conformation"], "square", r["energy"])
    assert chk["ok"], chk["problems"]


@pytest.mark.skipif(not SLOW, reason="~2 minutes; set REM_SLOW=1 to run")
def test_published_36mer():
    r = fold_hp_full(BENCH_36[0], "square")
    assert r["proved_optimal"] and r["energy"] == BENCH_36[1]
    assert validate_conformation(BENCH_36[0], r["conformation"], "square",
                                 r["energy"])["ok"]


def test_module_benchmark_table_matches_this_file():
    """The module keeps its own copy of the benchmark set; the two must not drift."""
    mine = {BENCH_20, BENCH_24, BENCH_25, BENCH_36}
    theirs = {(s, e) for _, s, e in hp.HP_BENCHMARKS_2D}
    assert mine == theirs


def test_published_optima_respect_the_static_upper_bound():
    """A published optimum below the module's own upper bound would prove the bound wrong."""
    for seq, published in (BENCH_20, BENCH_24, BENCH_25, BENCH_36):
        ub = max_contacts_upper_bound(seq, "square")
        assert -published <= ub, f"{seq}: published {published} beats the bound -{ub}"


# --------------------------------------------------------- (c) conformation validation
def test_hand_counted_energy():
    """A 2x2 square walk of four H's: sites (0,0)(1,0)(1,1)(0,1). The only non-bonded
    pair at unit distance is residue 0 with residue 3, so E = -1, counted by hand."""
    conf = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert hp_energy("HHHH", conf) == -1
    assert hp_energy("HPPH", conf) == -1
    assert hp_energy("HPPP", conf) == 0
    # a straight rod has no contacts at all
    assert hp_energy("HHHH", [(0, 0), (1, 0), (2, 0), (3, 0)]) == 0


def test_all_H_square_optimum_is_hand_derivable():
    """9 H's fold into the 3x3 square. That polyomino has 12 lattice adjacencies and the
    walk spends 8 of them on chain bonds, so exactly 4 are contacts: E = -4. 12 is also
    the Harary-Harborth maximum floor(2m - 2 sqrt(m)) for m = 9 cells, so no other shape
    can do better."""
    e, conf = fold_hp("HHHHHHHHH", "square")
    assert e == -4
    assert validate_conformation("HHHHHHHHH", conf, "square", -4)["ok"]
    assert enumerate_bruteforce("HHHHHHHHH", "square")[0] == -4


def test_all_H_16mer_matches_the_polyomino_maximum():
    """16 H's: the 4x4 grid graph has 24 edges, a Hamiltonian path spends 15, leaving 9
    contacts. floor(2*16 - 2*sqrt(16)) = 24 is the maximum adjacency count for any
    16-cell polyomino, so E = -9 is ground truth, not a guess."""
    r = fold_hp_full("H" * 16, "square")
    assert r["proved_optimal"] and r["energy"] == -9
    assert validate_conformation("H" * 16, r["conformation"], "square", -9)["ok"]


def test_validator_rejects_broken_conformations():
    assert not validate_conformation("HHH", [(0, 0), (1, 0), (1, 0)], "square")["ok"]
    assert not validate_conformation("HHH", [(0, 0), (2, 0), (2, 1)], "square")["ok"]
    good = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert validate_conformation("HHHH", good, "square", -1)["ok"]
    assert not validate_conformation("HHHH", good, "square", -2)["ok"]
    assert not validate_conformation("HHHH", good, "cubic")["ok"]


def test_every_returned_conformation_is_valid():
    rng = random.Random(99)
    for _ in range(12):
        seq = _rand_seq(rng, rng.randint(6, 14))
        e, conf = fold_hp(seq, "square")
        chk = validate_conformation(seq, conf, "square", e)
        assert chk["ok"], (seq, chk["problems"])
        assert hp_energy(seq, conf) == e


# ----------------------------------------------------------------- bound / search audits
def test_bound_is_admissible():
    r = verify_bound_admissible(verbose=False)
    assert r["max_deficit_tier1"] <= 0
    assert r["max_deficit_tier2"] <= 0
    assert r["tier2_never_looser_than_tier1"]
    assert r["prefixes_checked"] > 5000


def test_search_bound_equals_the_audited_formula():
    r = verify_bound_matches_search(verbose=False)
    assert r["mismatches"] == 0
    assert r["bound_evaluations"] > 500


def test_root_bound_is_an_upper_bound_on_contacts():
    rng = random.Random(5)
    for _ in range(10):
        seq = _rand_seq(rng, rng.randint(5, 10))
        e, _ = fold_hp(seq, "square")
        assert -e <= max_contacts_upper_bound(seq, "square")


def test_budget_exhaustion_warns_and_does_not_claim_optimality():
    with pytest.warns(UserWarning):
        e, conf = fold_hp(BENCH_36[0], "square", node_limit=2000, warmup_nodes=1000)
    assert validate_conformation(BENCH_36[0], conf, "square", e)["ok"]
    r = fold_hp_full(BENCH_36[0], "square", node_limit=2000, warmup_nodes=1000)
    assert not r["proved_optimal"]
    assert r["energy"] >= BENCH_36[1]           # only an upper bound on the true optimum


def test_structure_info_reports_the_governing_law():
    si = structure_info(BENCH_20[0], "square")
    assert si["treewidth"] == len(BENCH_20[0]) - 1
    assert si["d_states_per_variable"] == (2 * 20 + 1) ** 2
    assert si["log10_elimination_cost"] > 50


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        fold_hp("HPX", "square")
    with pytest.raises(ValueError):
        fold_hp("HP", "hexagonal")
    with pytest.raises(ValueError):
        hp_energy("HHH", [(0, 0), (1, 0)])


def test_verify_runs_and_reports_zero_error():
    res = hp.verify(verbose=False, n_random=4)
    assert res["max_err_vs_enumeration"] == 0
    assert res["invalid_conformations"] == []
    assert res["benchmarks_all_match_published"]
    assert res["bound_audit"]["admissible"]
