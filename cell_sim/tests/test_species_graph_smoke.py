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
    # 5 rows: 3 SBML matches + 2 unmatched
    rows = ['A', 'B', 'C', 'PM_dummy', 'cost_paid']
    g = build_species_graph(rows, cobra)

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
    g = build_species_graph(rows, cobra)
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
    g = build_species_graph(rows, cobra)
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
    g = build_species_graph(rows, cobra)
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
    g = build_species_graph(rows, cobra)
    s = graph_summary(g)
    assert s['n_nodes'] == 4
    assert s['n_sbml_match'] == 3
    assert s['n_sbml_unmatched'] == 1
    assert s['unique_reactions'] == 2


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_graph_basic_topology(tp)
        test_unmatched_rows_have_only_self_loops(tp)
        test_edge_attributes_carry_stoichiometry(tp)
        test_save_and_load_roundtrip(tp)
        test_graph_summary_reports_expected_stats(tp)
    print('OK: species-graph smoke tests passed.')
