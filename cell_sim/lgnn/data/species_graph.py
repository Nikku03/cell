"""Build the species×species reaction graph from iMB155.

Reads the COBRA-format JSON (`lsdata.get('cobra_imb155')`) — much easier
than parsing SBML XML and identical content. For every reaction r we
enumerate all unordered pairs (s_i, s_j) of metabolites that participate
in r and emit a directed edge in each direction with attributes
`(stoich_i, stoich_j, sign, |stoich_i*stoich_j|)`.

The 8572 rows of counts_and_fluxes are a superset of the SBML
metabolites — proteins, mRNAs, complexes, and cost accumulators are
not in iMB155. Unmatched rows get a self-loop with `is_self=1` so
downstream message passing falls back to a per-node MLP for them
(same behaviour as the v0 baseline). This makes the graph prior
helpful where it can be helpful and harmless where it can't.

Output is a torch dict savable via `torch.save`:
    {
      'edge_index'  : (2, E)   long, [src; dst]
      'edge_attr'   : (E, 5)   float, [stoich_i, stoich_j, sign, mag, is_self]
      'edge_kind'   : (E,)     long, 0=sbml, 1=self_loop
      'row_names'   : list[str], length = n_nodes
      'sbml_match'  : (n_nodes,) bool, True where the row matched an SBML id
      'reaction_id' : list[str|None], length E   -- None for self-loops
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SpeciesGraph:
    edge_index: torch.Tensor       # (2, E) long
    edge_attr: torch.Tensor        # (E, 5) float
    edge_kind: torch.Tensor        # (E,) long; 0=sbml, 1=self_loop
    row_names: List[str]
    sbml_match: torch.Tensor       # (n_nodes,) bool
    reaction_id: List[Optional[str]]

    @property
    def n_nodes(self) -> int:
        return len(self.row_names)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def n_sbml_edges(self) -> int:
        return int((self.edge_kind == 0).sum().item())


def parse_cobra_reactions(cobra_json_path: Path) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Return (metabolite_ids, reaction_id -> {met_id: stoich})."""
    with open(cobra_json_path) as f:
        model = json.load(f)
    met_ids = [m['id'] for m in model.get('metabolites', [])]
    reactions = {}
    for r in model.get('reactions', []):
        reactions[r['id']] = dict(r.get('metabolites', {}))
    return met_ids, reactions


def build_species_graph(
    row_names: List[str],
    cobra_json_path: Path,
) -> SpeciesGraph:
    """Construct the SpeciesGraph for the given row order.

    `row_names` is the species list of one counts_and_fluxes replicate
    (length 8572 in the Luthey-Schulten data). The function maps each
    row name to an SBML metabolite id via exact string match. If your
    naming differs you'll need an aliasing pass before calling this.
    """
    met_ids, reactions = parse_cobra_reactions(Path(cobra_json_path))
    name_to_idx = {n: i for i, n in enumerate(row_names)}
    sbml_set = set(met_ids)

    src_list, dst_list = [], []
    attrs: List[List[float]] = []
    kinds: List[int] = []
    rxn_ids: List[Optional[str]] = []

    # SBML edges from co-occurrence in reactions
    for rxn_id, stoich in reactions.items():
        species_in_rxn = [(s, st) for s, st in stoich.items()
                          if s in name_to_idx and s in sbml_set]
        for a in range(len(species_in_rxn)):
            for b in range(len(species_in_rxn)):
                if a == b:
                    continue
                sid_a, st_a = species_in_rxn[a]
                sid_b, st_b = species_in_rxn[b]
                i = name_to_idx[sid_a]
                j = name_to_idx[sid_b]
                src_list.append(i); dst_list.append(j)
                prod = float(st_a) * float(st_b)
                attrs.append([float(st_a), float(st_b),
                              float(np.sign(prod)),
                              float(abs(prod)),
                              0.0])                       # is_self=0
                kinds.append(0)
                rxn_ids.append(rxn_id)

    # Self-loop on every node (SBML-matched and unmatched alike)
    for i in range(len(row_names)):
        src_list.append(i); dst_list.append(i)
        attrs.append([0.0, 0.0, 0.0, 0.0, 1.0])           # is_self=1
        kinds.append(1)
        rxn_ids.append(None)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attrs, dtype=torch.float32)
    edge_kind  = torch.tensor(kinds, dtype=torch.long)
    sbml_match = torch.tensor([n in sbml_set for n in row_names],
                               dtype=torch.bool)
    return SpeciesGraph(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_kind=edge_kind,
        row_names=list(row_names),
        sbml_match=sbml_match,
        reaction_id=rxn_ids,
    )


def save_species_graph(g: SpeciesGraph, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'edge_index':  g.edge_index,
        'edge_attr':   g.edge_attr,
        'edge_kind':   g.edge_kind,
        'row_names':   g.row_names,
        'sbml_match':  g.sbml_match,
        'reaction_id': g.reaction_id,
    }, path)
    return path


def load_species_graph(path: Path) -> SpeciesGraph:
    obj = torch.load(Path(path), map_location='cpu', weights_only=False)
    return SpeciesGraph(
        edge_index=obj['edge_index'],
        edge_attr=obj['edge_attr'],
        edge_kind=obj['edge_kind'],
        row_names=obj['row_names'],
        sbml_match=obj['sbml_match'],
        reaction_id=obj['reaction_id'],
    )


def graph_summary(g: SpeciesGraph) -> dict:
    """Quick stats for sanity-checking a freshly-built graph."""
    sbml_edges = (g.edge_kind == 0).sum().item()
    self_edges = (g.edge_kind == 1).sum().item()
    matched_nodes = int(g.sbml_match.sum().item())
    # Per-node SBML degree (excluding self-loops)
    sbml_mask = g.edge_kind == 0
    if sbml_edges > 0:
        dst_sbml = g.edge_index[1][sbml_mask]
        deg = torch.bincount(dst_sbml, minlength=g.n_nodes)
    else:
        deg = torch.zeros(g.n_nodes, dtype=torch.long)
    matched_deg = deg[g.sbml_match]
    return {
        'n_nodes':            g.n_nodes,
        'n_sbml_match':       matched_nodes,
        'n_sbml_unmatched':   g.n_nodes - matched_nodes,
        'n_sbml_edges':       sbml_edges,
        'n_self_loops':       self_edges,
        'mean_sbml_degree':   float(deg.float().mean()),
        'matched_node_mean_degree': (float(matched_deg.float().mean())
                                      if matched_nodes > 0 else 0.0),
        'matched_node_max_degree':  (int(matched_deg.max())
                                      if matched_nodes > 0 else 0),
        'unique_reactions':   len({r for r in g.reaction_id if r is not None}),
    }
