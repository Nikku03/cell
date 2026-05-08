"""Smoke test for the SBML species-graph builder.

Builds a tiny synthetic COBRA model (3 metabolites, 2 reactions) plus a
row list that matches all 3 SBML species and adds 2 unmatched rows.
Verifies edge counts, that unmatched rows get only self-loops, and that
the saved/loaded graph round-trips.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))   # repo root onto path

from cell_sim.lgnn.data.species_graph import (
    build_species_graph, graph_summary, load_species_graph,
    save_species_graph,
)


def write_fake_cobra(path: Path):
    """3 metabolites (A, B, C), 2 reactions (R1: A→B, R2: B+C→A)."""
    model = {
        'metabolites': [
            {'id': 'A', 'compartment': 'c'},
            {'id': 'B', 'compartment': 'c'},
            {'id': 'C', 'compartment': 'c'},
        ],
        'reactions': [
            {'id': 'R1', 'metabolites': {'A': -1.0, 'B': 1.0}},
            {'id': 'R2', 'metabolites': {'B': -1.0, 'C': -1.0, 'A': 1.0}},
        ],
    }
    path.write_text(json.dumps(model))


def test_graph_basic_topology(tmp_path):
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    # 5 rows: 3 SBML matches + 2 unmatched. PM_dummy has alpha locus
    # so it won't trigger the simulator-edge regex; cost_paid has no
    # underscore-digit pattern at all.
    rows = ['A', 'B', 'C', 'PM_dummy', 'cost_paid']
    g = build_species_graph(rows, cobra, include_simulator_edges=False)

    # R1 has 2 species → 2 directed pairs (A,B) (B,A)
    # R2 has 3 species → 6 directed pairs (B,C) (B,A) (C,B) (C,A) (A,B) (A,C)
    # Total SBML edges = 2 + 6 = 8
    assert g.n_sbml_edges == 8, g.n_sbml_edges
    # 5 self-loops, one per node
    assert (g.edge_kind == 1).sum() == 5
    # 3 of 5 rows are SBML-matched
    assert g.sbml_match.tolist() == [True, True, True, False, False]


def test_unmatched_rows_have_only_self_loops(tmp_path):
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    rows = ['A', 'B', 'C', 'PM_dummy', 'cost_paid']
    g = build_species_graph(rows, cobra, include_simulator_edges=False)
    # Find node index of PM_dummy
    idx_pm = rows.index('PM_dummy')
    # All edges where dst == idx_pm
    inc = (g.edge_index[1] == idx_pm)
    # Should be exactly 1 (the self-loop)
    assert inc.sum() == 1
    assert g.edge_kind[inc].tolist() == [1]
    # And src == dst for that edge
    assert g.edge_index[0][inc].item() == idx_pm


def test_edge_attributes_carry_stoichiometry(tmp_path):
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    rows = ['A', 'B', 'C']
    g = build_species_graph(rows, cobra, include_simulator_edges=False)
    # Find the (A→B) edge from R1.  Stoich A=-1, B=+1 → product = -1, sign=-1.
    a_idx, b_idx = 0, 1
    mask = (g.edge_index[0] == a_idx) & (g.edge_index[1] == b_idx) & (g.edge_kind == 0)
    rows_match = mask.nonzero().flatten()
    # R1 contributes one (A,B) edge; R2 contributes another (A,B) edge.
    # So we expect at least 2.
    assert len(rows_match) >= 1
    # Pick the R1 one — it has stoich (-1, +1) → attr [-1,+1,-1,1,0]
    # All matching edges should have that exact attr set if they're from R1;
    # R2's (A,B) is (+1, -1).
    attrs = g.edge_attr[rows_match].tolist()
    expected_set = {(-1.0, 1.0, -1.0, 1.0, 0.0),
                     (1.0, -1.0, -1.0, 1.0, 0.0)}
    for a in attrs:
        assert tuple(a) in expected_set, a


def test_save_and_load_roundtrip(tmp_path):
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    rows = ['A', 'B', 'C', 'PM_dummy', 'cost_paid']
    g = build_species_graph(rows, cobra, include_simulator_edges=False)
    pth = tmp_path / 'graph.pt'
    save_species_graph(g, pth)
    g2 = load_species_graph(pth)
    assert torch.equal(g.edge_index, g2.edge_index)
    assert torch.equal(g.edge_attr,  g2.edge_attr)
    assert torch.equal(g.edge_kind,  g2.edge_kind)
    assert torch.equal(g.sbml_match, g2.sbml_match)
    assert g.row_names   == g2.row_names
    assert g.reaction_id == g2.reaction_id


def test_graph_summary_reports_expected_stats(tmp_path):
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    rows = ['A', 'B', 'C', 'PM_dummy']
    g = build_species_graph(rows, cobra, include_simulator_edges=False)
    s = graph_summary(g)
    assert s['n_nodes'] == 4
    assert s['n_sbml_match'] == 3
    assert s['n_sbml_unmatched'] == 1
    assert s['edges_per_kind']['SBML'] == 8
    assert s['edges_per_kind']['SELF_LOOP'] == 4


def test_simulator_edges_link_central_dogma(tmp_path):
    """A locus with G_, R_, and PM_ rows must produce transcription
    + translation simulator edges between them."""
    from cell_sim.lgnn.data.species_graph import (
        build_simulator_edges, simulator_edge_summary, EdgeKind,
    )
    rows = ['G_0001', 'R_0001', 'P_0001', 'RP_0001', 'RPM_0001',
             'PM_0001', 'DM_0001', 'D_0001',
             'M_atp_c',                # metabolite — no locus
             'chromosome',             # outside the template
             'cost_paid']              # outside the template
    edges = build_simulator_edges(rows)
    # 8 patterns × 1 pair each × 2 directions = 16 edges (1 locus, 1 row each)
    assert len(edges) == 16, len(edges)
    # All EdgeKinds in the result must be valid simulator kinds
    kinds_seen = {k for _, _, k, _ in edges}
    assert EdgeKind.TRANSCRIPTION in kinds_seen
    assert EdgeKind.TRANSLATION in kinds_seen
    assert EdgeKind.DEGRADATION in kinds_seen
    assert EdgeKind.TRANSLOCATION in kinds_seen
    # Spot-check: G_0001 and RP_0001 are linked (transcription)
    g_idx = rows.index('G_0001')
    rp_idx = rows.index('RP_0001')
    edge_pairs = {(s, d) for s, d, _, _ in edges}
    assert (g_idx, rp_idx) in edge_pairs
    assert (rp_idx, g_idx) in edge_pairs

    # summary fn agrees
    s = simulator_edge_summary(rows)
    assert s['n_loci_seen'] == 1
    assert s['n_loci_with_G_R_P_all'] == 1
    assert s['total_simulator_edges'] == 16


def test_simulator_edges_skip_metabolites_with_short_digit_runs(tmp_path):
    """M_3pg_c, M_2pg_c etc. must NOT match the locus regex."""
    from cell_sim.lgnn.data.species_graph import build_simulator_edges
    rows = ['M_3pg_c', 'M_2pg_c', 'M_atp_c', 'M_13dpg_c']
    edges = build_simulator_edges(rows)
    assert edges == []                    # nothing should link


def test_full_graph_with_simulator_edges(tmp_path):
    """Build a graph with both SBML metabolite edges AND simulator
    central-dogma edges. Counts should add (no overlap)."""
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    rows = ['A', 'B', 'C', 'G_0001', 'RP_0001', 'R_0001']
    g = build_species_graph(rows, cobra, include_simulator_edges=True)
    s = graph_summary(g)
    # SBML edges from R1 (A-B pair = 2) + R2 (A,B,C triplet = 6) = 8
    assert s['edges_per_kind']['SBML'] == 8
    # 6 self-loops (one per row)
    assert s['edges_per_kind']['SELF_LOOP'] == 6
    # Simulator: G↔RP, RP↔R = 2 patterns × 1 pair × 2 dirs = 4 transcription edges
    assert s['edges_per_kind']['TRANSCRIPTION'] == 4


def test_flux_edges_couple_F_rxn_to_species(tmp_path):
    """For each SBML reaction r whose F_<r> row is present, the graph
    must contain bidirectional FLUX_COUPLING edges between F_<r> and
    every species in r."""
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    # F_R1 is present (R1: A↔B → 2 species → 4 directed edges)
    # F_R2_end is present (R2: A,B,C → 3 species → 6 directed edges)
    # F_NOT_A_REACTION is present but has no SBML reaction → 0 edges
    rows = ['A', 'B', 'C', 'F_R1', 'F_R2_end', 'F_NOT_A_REACTION']
    g = build_species_graph(rows, cobra, include_simulator_edges=False)
    s = graph_summary(g)
    assert s['edges_per_kind']['FLUX_COUPLING'] == 4 + 6, \
        s['edges_per_kind']
    # Hand-verify: F_R1 ↔ A, F_R1 ↔ B should both exist
    flux_idx = rows.index('F_R1')
    a_idx = rows.index('A')
    edges = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    assert (flux_idx, a_idx) in edges
    assert (a_idx, flux_idx) in edges
    # F_NOT_A_REACTION should have no flux edges (only self-loop)
    nar_idx = rows.index('F_NOT_A_REACTION')
    nar_kind_set = {g.edge_kind[i].item()
                    for i in range(g.n_edges)
                    if g.edge_index[0][i].item() == nar_idx
                       or g.edge_index[1][i].item() == nar_idx}
    assert nar_kind_set == {1}     # only SELF_LOOP


def test_flux_edges_summary_counts_match(tmp_path):
    """flux_edge_summary should agree with what build_species_graph
    actually emits."""
    from cell_sim.lgnn.data.species_graph import flux_edge_summary
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    rows = ['A', 'B', 'C', 'F_R1', 'F_R2_end']
    rep = flux_edge_summary(rows, cobra)
    assert rep['n_sbml_reactions'] == 2
    assert rep['n_with_F_avg_row'] == 1       # F_R1 only
    assert rep['n_with_F_end_row'] == 1       # F_R2_end only
    assert rep['n_reactions_covered'] == 2    # both reactions covered


def test_sbml_xml_parser_matches_cobra_format(tmp_path):
    """SBML XML output must come back in the same shape as the COBRA
    parser so build_species_graph can dispatch by extension."""
    xml_text = """<?xml version="1.0"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core">
  <model id="test">
    <listOfSpecies>
      <species id="M_atp_c"/>
      <species id="M_adp_c"/>
      <species id="M_pi_c"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="R_test">
        <listOfReactants>
          <speciesReference species="M_atp_c" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="M_adp_c" stoichiometry="1"/>
          <speciesReference species="M_pi_c" stoichiometry="1"/>
        </listOfProducts>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
    p = tmp_path / 'mini.xml'
    p.write_text(xml_text)
    from cell_sim.lgnn.data.species_graph import parse_sbml_reactions
    species, rxns = parse_sbml_reactions(p)
    assert species == ['M_atp_c', 'M_adp_c', 'M_pi_c']
    assert set(rxns) == {'R_test'}
    # Reactants signed negative, products positive
    assert rxns['R_test'] == {'M_atp_c': -1.0, 'M_adp_c': 1.0, 'M_pi_c': 1.0}


def test_alias_matching_handles_M_prefix(tmp_path):
    """Rows like `atp_c` (no M_ prefix) should match SBML's `M_atp_c`."""
    xml_text = """<?xml version="1.0"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core">
  <model id="test">
    <listOfSpecies><species id="M_atp_c"/><species id="M_adp_c"/></listOfSpecies>
    <listOfReactions>
      <reaction id="R_x">
        <listOfReactants><speciesReference species="M_atp_c" stoichiometry="1"/></listOfReactants>
        <listOfProducts><speciesReference species="M_adp_c" stoichiometry="1"/></listOfProducts>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
    p = tmp_path / 'mini.xml'
    p.write_text(xml_text)
    # Rows use bare convention
    rows = ['atp_c', 'adp_c', 'unrelated_thing']
    g = build_species_graph(rows, p)
    # 2 SBML edges (atp_c ↔ adp_c via R_x) + 3 self-loops
    assert g.n_sbml_edges == 2
    assert (g.edge_kind == 1).sum() == 3
    # Both metabolite rows should be marked matched
    assert g.sbml_match.tolist() == [True, True, False]


def test_diagnose_reports_sensible_stats(tmp_path):
    cobra = tmp_path / 'mini.json'
    write_fake_cobra(cobra)
    rows = ['A', 'B', 'C', 'PM_dummy', 'cost_paid']
    from cell_sim.lgnn.data.species_graph import diagnose_row_matching
    rep = diagnose_row_matching(rows, cobra)
    assert rep['n_rows'] == 5
    assert rep['n_total_match'] == 3
    assert rep['matches_per_rule']['direct'] == 3
    assert 'PM' in rep['top_prefixes']


def test_real_local_sbml_parses_and_aliases(tmp_path):
    """Sanity check on the actual Syn3A_updated.xml in the repo —
    it must give nonzero matches for canonical metabolite names like
    M_atp_c regardless of which prefix convention the rows use.
    """
    sbml = (Path(__file__).resolve().parent.parent
            / 'data' / 'Minimal_Cell_ComplexFormation'
            / 'input_data' / 'Syn3A_updated.xml')
    assert sbml.exists(), f'{sbml} missing — repo state changed'
    # Both prefix conventions should match
    rows_M = ['M_atp_c', 'M_adp_c', 'M_pi_c', 'PM_garbage']
    g = build_species_graph(rows_M, sbml, include_simulator_edges=False)
    assert g.sbml_match[:3].all()
    assert not g.sbml_match[3]
    assert g.n_sbml_edges > 0           # ATP/ADP/Pi all co-occur in many reactions

    rows_bare = ['atp_c', 'adp_c', 'pi_c', 'PM_garbage']
    g2 = build_species_graph(rows_bare, sbml, include_simulator_edges=False)
    assert g2.sbml_match[:3].all()
    assert g2.n_sbml_edges == g.n_sbml_edges


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_graph_basic_topology(tp)
        test_unmatched_rows_have_only_self_loops(tp)
        test_edge_attributes_carry_stoichiometry(tp)
        test_save_and_load_roundtrip(tp)
        test_graph_summary_reports_expected_stats(tp)
        test_simulator_edges_link_central_dogma(tp)
        test_simulator_edges_skip_metabolites_with_short_digit_runs(tp)
        test_full_graph_with_simulator_edges(tp)
        test_flux_edges_couple_F_rxn_to_species(tp)
        test_flux_edges_summary_counts_match(tp)
        test_sbml_xml_parser_matches_cobra_format(tp)
        test_alias_matching_handles_M_prefix(tp)
        test_diagnose_reports_sensible_stats(tp)
        test_real_local_sbml_parses_and_aliases(tp)
    print('OK: species-graph smoke tests passed.')
