"""RNA folding against explicit enumeration of every nested secondary structure.

The reference in every accuracy test is `rem.rna.brute_force`, which LISTS structures
(`enumerate_structures`, a split recursion) and SCORES each one by walking its partner
array (`structure_energy`). Neither touches the McCaskill hypergraph or the inside/outside
passes under test. The only shared object is the EnergyModel, which is the definition of
the physics rather than an algorithm -- if it were not shared the two sides would be
computing different quantities.
"""
import math
import random

import numpy as np
import pytest

from rem import rna
from rem.rna import (BasePairModel, StackingModel, mccaskill, mfe, brute_force,
                     enumerate_structures, structure_energy, db_to_pairs,
                     pairs_to_db, compare_structures, build_graph, graph_info,
                     YEAST_TRNA_PHE, YEAST_TRNA_PHE_DB)

TOL = 1e-10

MODELS = [BasePairModel(),
          StackingModel(terminal_mismatch=False),
          StackingModel(terminal_mismatch=True),
          StackingModel(terminal_mismatch=True, multiloop="turner2004")]


def _seq(seed, n=None, gc=0.55):
    rng = random.Random(seed)
    n = n or rng.randint(9, 13)
    return rna._random_seq(rng, n, gc)


# --------------------------------------------------------------- exactness vs enumeration
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("mi", range(len(MODELS)))
def test_logZ_and_bpp_match_enumeration(seed, mi):
    model = MODELS[mi]
    seq = _seq(seed)
    ref = brute_force(seq, model)
    logZ, P = mccaskill(seq, model)
    assert abs(logZ - ref["logZ"]) < TOL
    assert np.max(np.abs(P - ref["bpp"])) < TOL


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("mi", range(len(MODELS)))
def test_mfe_matches_enumeration(seed, mi):
    model = MODELS[mi]
    seq = _seq(seed)
    ref = brute_force(seq, model)
    r = mfe(seq, model)
    assert abs(r.energy - ref["mfe"]) < TOL
    # and the traceback must return a structure that really has that energy
    assert abs(structure_energy(seq, r.pairs, model) - r.energy) < TOL


@pytest.mark.parametrize("seq", rna.ENUM_CASES)
def test_multiloop_optima_match_enumeration(seq):
    """A random 12-mer cannot contain a multiloop at all (closing pair + two branches needs
    >= 12 nt), so the multiloop recursion needs its own designed cases. The stress model
    makes the OPTIMUM a multiloop, which also exercises the min-semiring traceback."""
    model = rna.multiloop_stress_model()
    ref = brute_force(seq, model)
    logZ, P = mccaskill(seq, model)
    r = mfe(seq, model)
    assert abs(logZ - ref["logZ"]) < TOL
    assert np.max(np.abs(P - ref["bpp"])) < TOL
    assert abs(r.energy - ref["mfe"]) < TOL
    assert rna._has_multiloop(rna.clean_seq(seq), r.pairs), "case does not test multiloops"


def test_enumeration_actually_covers_multiloops():
    seq = rna.ENUM_CASES[1]
    structs = enumerate_structures(seq, MODELS[0])
    n_ml = sum(1 for s in structs if rna._has_multiloop(rna.clean_seq(seq), s))
    assert n_ml > 1000, f"only {n_ml} multiloop structures enumerated"


# --------------------------------------------------------------- independence of scorer
def test_base_pair_energy_is_literally_the_sum_over_pairs():
    """`structure_energy` reaches this number through a loop decomposition
    (hairpin / interior / multiloop). For BasePairModel the answer must equal the trivial
    sum over pairs -- an independent check that the decomposition itself is right."""
    model = BasePairModel()
    for seed in range(6):
        seq = rna.clean_seq(_seq(100 + seed))
        for s in enumerate_structures(seq, model)[:200]:
            direct = sum(model.pair_energy(seq[i], seq[j]) for i, j in s)
            assert abs(structure_energy(seq, s, model) - direct) < TOL


def test_stacking_energy_matches_a_hand_computation():
    """Close the last loophole in the independence argument: `structure_energy` and the DP
    share the EnergyModel, so a wrong PARAMETER would cancel out of every enumeration test.
    Here the energy of a designed hairpin is written out by hand from the published tables.

        GGCGCG AAAA CGCGCC   with pairs (0,15)(1,14)(2,13)(3,12)(4,11)(5,10)
        stacks GC/GC -3.26, GC/CG -3.42, CG/GC -2.36, GC/CG -3.42, CG/GC -2.36
        hairpin of 4 closed by G-C: init 5.60 + terminal mismatch -1.40 (A.A, no bonus)
        exterior branch G-C: 0.00
    """
    seq = "GGCGCG" "AAAA" "CGCGCC"
    pairs = [(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10)]
    by_hand = (-3.26 - 3.42 - 2.36 - 3.42 - 2.36) + (5.60 - 1.40) + 0.0
    assert structure_energy(seq, pairs, StackingModel()) == pytest.approx(by_hand, abs=1e-9)
    assert mfe(seq, StackingModel()).energy == pytest.approx(by_hand, abs=1e-9)


def test_partition_function_is_a_probability_distribution():
    for seed in range(5):
        seq = _seq(200 + seed, n=24)
        model = StackingModel()
        logZ, P = mccaskill(seq, model)
        assert np.allclose(P, P.T)
        assert P.min() >= -1e-12
        assert P.max() <= 1 + 1e-9
        assert P.sum(axis=1).max() <= 1 + 1e-9      # a base pairs with at most one partner
        # -RT logZ is the ensemble free energy and can never be above the MFE
        assert -rna.RT37 * logZ <= mfe(seq, model).energy + 1e-9


def test_probability_of_mfe_structure_matches_bpp_of_its_pairs():
    seq = _seq(7, n=20)
    model = StackingModel()
    r = mfe(seq, model)
    logZ, P = mccaskill(seq, model)
    p_mfe = math.exp(-r.energy / rna.RT37 - logZ)
    for i, j in r.pairs:
        assert P[i, j] >= p_mfe - 1e-12          # the MFE is one structure containing (i,j)


# --------------------------------------------------------------- log space
def test_log_space_is_load_bearing():
    """At 5 K the partition function is e^1069. A naive implementation that stores raw
    Boltzmann weights overflows float64 here; this one still matches enumeration exactly."""
    seq = "GGCGCGAAAACGCGCC"
    model = StackingModel()
    logZ, P = mccaskill(seq, model, temperature=5.0)
    ref = brute_force(seq, model, temperature=5.0)
    assert logZ > 709.78, "test no longer reaches the overflow regime"
    with pytest.raises(OverflowError):          # float64 cannot hold Z itself
        math.exp(logZ)
    assert abs(logZ - ref["logZ"]) < 1e-9
    assert np.max(np.abs(P - ref["bpp"])) < 1e-9


# --------------------------------------------------------------- notation and constraints
def test_dot_bracket_roundtrip():
    for _, seq, db in rna.PLANTED:
        assert pairs_to_db(db_to_pairs(db), len(db)) == db


def test_min_hairpin_is_respected():
    seq = "GGGGCCCC"
    for mh in (3, 4, 5):
        r = mfe(seq, StackingModel(), min_hairpin=mh)
        for i, j in r.pairs:
            assert j - i - 1 >= mh
        _, P = mccaskill(seq, StackingModel(), min_hairpin=mh)
        for i in range(len(seq)):
            for j in range(i + 1, min(i + mh + 1, len(seq))):
                assert P[i, j] == 0.0


def test_predictions_are_nested():
    for seed in range(5):
        seq = _seq(300 + seed, n=40)
        r = mfe(seq, StackingModel())
        # structure_energy raises on any crossing pair
        assert structure_energy(seq, r.pairs, StackingModel()) < rna.INF


# --------------------------------------------------------------- cost / the governing law
def test_dp_terms_have_at_most_two_children():
    """Three free sequence indices per term is the whole cost claim: n^3 = d^(tw+1)
    with d = n and tw = 2."""
    g = build_graph(YEAST_TRNA_PHE, StackingModel())
    assert g.max_children() == 2
    assert graph_info(g)["treewidth"] == 2


def test_cost_is_cubic_in_sequence_length():
    import time
    model = StackingModel()
    rng = random.Random(5)
    rows = []
    for n in (60, 90, 130):
        s = rna._random_seq(rng, n, 0.5)
        t0 = time.perf_counter()
        g = build_graph(s, model)
        rna.inside_min(g)
        rows.append((n, time.perf_counter() - t0, g.n_terms()))
    x = np.log([a for a, _, _ in rows])
    size_slope = np.polyfit(x, np.log([c for _, _, c in rows]), 1)[0]
    time_slope = np.polyfit(x, np.log([b for _, b, _ in rows]), 1)[0]
    assert 2.5 < size_slope < 3.5, f"hypergraph size grows as n^{size_slope:.2f}"
    assert time_slope < 4.0, f"time grows as n^{time_slope:.2f}"


# --------------------------------------------------------------- the tRNA benchmark
def test_trna_reference_structure_is_self_consistent():
    """Guard the hardcoded benchmark: every accepted pair must be canonical and the arms
    must sit where the literature says they do (1-based positions in the comment)."""
    seq = rna.clean_seq(YEAST_TRNA_PHE)
    assert len(seq) == 76
    ref = db_to_pairs(YEAST_TRNA_PHE_DB)
    assert len(YEAST_TRNA_PHE_DB) == 76
    assert len(ref) == 21
    for i, j in ref:
        assert (seq[i], seq[j]) in rna.CANONICAL, (i + 1, j + 1, seq[i], seq[j])
    one = {(i + 1, j + 1) for i, j in ref}
    assert (1, 72) in one and (7, 66) in one          # acceptor stem
    assert (10, 25) in one and (13, 22) in one        # D-arm
    assert (27, 43) in one and (31, 39) in one        # anticodon arm
    assert (49, 65) in one and (53, 61) in one        # T-arm
    assert seq[33:36] == "GAA"                        # anticodon, positions 34-36
    assert seq[73:76] == "CCA"                        # CCA tail


def test_trna_stacking_beats_base_pair_model():
    """The measured claim: stacking does not change how well the SEARCH works (both are
    exact), it changes where the optimum sits. Recorded numbers at the time of writing:
    base-pair E(accepted) - MFE = +14.00 kcal/mol, P(accepted) = 9.3e-15;
    stacking+mismatch = +0.23 kcal/mol, P(accepted) = 2.6e-2."""
    b = rna.fold_and_score(YEAST_TRNA_PHE, YEAST_TRNA_PHE_DB, BasePairModel())
    s = rna.fold_and_score(YEAST_TRNA_PHE, YEAST_TRNA_PHE_DB, StackingModel())
    gap_b = b["energy_of_reference"] - b["mfe"]
    gap_s = s["energy_of_reference"] - s["mfe"]
    assert gap_b > 10.0, gap_b
    assert gap_s < 1.0, gap_s
    assert s["p_of_reference"] > 1e10 * b["p_of_reference"]
    assert s["mean_bpp_of_true_pairs"] > 2 * b["mean_bpp_of_true_pairs"]
    assert s["ppv"] >= b["ppv"]


def test_trna_accuracy_is_recorded_not_asserted_away():
    """Honest lock-in of what this energy model actually achieves on yeast tRNA-Phe.
    Sensitivity is 7/21 = 0.333 for every tier of the model as shipped; the accepted
    structure is only 0.23 kcal/mol above the optimum, and single-parameter changes of a
    few tenths of a kcal flip it to 21/21. That is a scoring-function limit, not a search
    limit -- the search is exact to 1e-15 (tests above)."""
    out = rna.benchmark_trna(verbose=False, sweep=True, n_shuffles=4)
    by = {r["model"]: r for r in out["rows"]}
    assert by["base-pair"]["sensitivity"] == pytest.approx(7 / 21, abs=1e-9)
    assert by["stacking+mismatch/turner1999"]["sensitivity"] == pytest.approx(7 / 21,
                                                                             abs=1e-9)
    # the acceptor stem (7 pairs) is what every tier gets right
    assert by["stacking+mismatch/turner1999"]["tp"] == 7
    # a small change in any one of four parameters recovers the full cloverleaf
    flips = {k: max(s for _, s in v) for k, v in out["sweep"].items()}
    assert all(v == 1.0 for v in flips.values()), flips
    # ... and the null says 21/21 is not something a shuffled sequence stumbles into
    assert out["shuffled_sensitivity"]["max"] < 0.5


def test_planted_structures_are_recovered_positive_control():
    """Before believing 'the model is wrong on tRNA', check the pipeline recovers a planted
    signal. The stacking model must get every designed case exactly right."""
    rows = rna.benchmark_planted(verbose=False)["rows"]
    for r in rows:
        if r["model"].startswith("stacking"):
            assert r["sensitivity"] == 1.0 and r["ppv"] == 1.0, r["case"]
            assert r["p_of_reference"] > 0.5, (r["case"], r["p_of_reference"])
    # the base-pair model fails the two-hairpin case -- a real scoring-function failure,
    # visible on a designed sequence where the intended answer is not in doubt
    bp = [r for r in rows if r["model"] == "base-pair" and r["case"] == "two hairpins"]
    assert bp and bp[0]["sensitivity"] < 1.0


def test_shuffled_sequences_do_not_fold_into_the_cloverleaf():
    """Negative control for the benchmark itself."""
    rng = random.Random(0)
    ref = db_to_pairs(YEAST_TRNA_PHE_DB)
    model = StackingModel()
    sens = []
    for _ in range(6):
        sh = rna.dinucleotide_shuffle(YEAST_TRNA_PHE, rng)
        sens.append(compare_structures(mfe(sh, model).pairs, ref)["sensitivity"])
    assert np.mean(sens) < 0.2, sens


# --------------------------------------------------------------- module verify()
def test_verify_summary_is_exact():
    r = rna.verify(seed=1, n_seqs=3, verbose=False)
    for name, e in r["errors"].items():
        assert e["logZ"] < TOL, (name, e)
        assert e["bpp"] < TOL, (name, e)
        assert e["mfe"] < TOL, (name, e)
        assert e["mfe_self"] < TOL, (name, e)
    assert r["n_multiloop_structures_enumerated"] > 1000
    assert sum(r["n_multiloop_mfe_checked"].values()) >= 3
    assert r["planted_all_recovered"] is True
    assert 2.5 < r["measured_size_exponent"] < 3.5


def test_outside_pass_at_trna_scale_without_enumeration():
    """sum_j P(i,j) = 1 - Z[i forced unpaired]/Z, with the right-hand side computed by a
    separate inside-only run on a mutated sequence. Tests the outside recursion at n = 76,
    where enumeration is impossible."""
    r = rna.verify_bpp_by_constraint(verbose=False)
    assert r["n"] == 76
    assert r["max_err"] < 1e-10, r["max_err"]
    assert len(r["rows"]) == 8
    with pytest.raises(ValueError):        # the identity is invalid with mismatch terms on
        rna.verify_bpp_by_constraint(model=StackingModel(terminal_mismatch=True))
