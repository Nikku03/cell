"""Build the species×species reaction graph from iMB155 (COBRA JSON) or
the local Syn3A_updated.xml (SBML XML). The Luthey-Schulten simulator
emits trajectories using `M_atp_c`-style metabolite IDs; iMB155's COBRA
JSON sometimes uses bare `atp_c` (no `M_` prefix). Multiple alias
conventions are tried so the graph builder works regardless of which
SBML source is supplied.

For every reaction r we enumerate all unordered pairs (s_i, s_j) of
metabolites that participate in r and emit a directed edge in each
direction with attributes `(stoich_i, stoich_j, sign, |stoich_i*stoich_j|)`.

The 8572 rows of counts_and_fluxes are a superset of the SBML
metabolites — proteins, mRNAs, complexes, and cost accumulators are
not in any flux-balance model. Unmatched rows get a self-loop with
`is_self=1` so downstream message passing falls back to a per-node
MLP for them (same behaviour as the v0 baseline).

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
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_cobra_reactions(
    cobra_json_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """COBRA JSON -> (metabolite_ids, {rxn_id: {met_id: signed_stoich}})."""
    with open(cobra_json_path) as f:
        model = json.load(f)
    met_ids = [m['id'] for m in model.get('metabolites', [])]
    reactions = {}
    for r in model.get('reactions', []):
        reactions[r['id']] = dict(r.get('metabolites', {}))
    return met_ids, reactions


def parse_sbml_reactions(
    sbml_xml_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """SBML XML -> (species_ids, {rxn_id: {species_id: signed_stoich}}).

    Reactants get negative sign, products positive, matching COBRA. Uses
    only the stdlib xml.etree.ElementTree so libsbml isn't required.
    """
    tree = ET.parse(sbml_xml_path)
    root = tree.getroot()
    m = re.match(r'\{(.*)\}', root.tag)
    ns = m.group(1) if m else ''
    def t(name: str) -> str:
        return f'{{{ns}}}{name}' if ns else name

    species_ids = [s.get('id') for s in root.iter(t('species'))]
    reactions: Dict[str, Dict[str, float]] = {}
    for rxn in root.iter(t('reaction')):
        rid = rxn.get('id')
        if rid is None:
            continue
        stoich: Dict[str, float] = {}
        for child in rxn:
            cname = child.tag.split('}')[-1]
            if cname not in ('listOfReactants', 'listOfProducts'):
                continue
            sign = -1.0 if cname == 'listOfReactants' else +1.0
            for sref in child:
                sid = sref.get('species')
                if sid is None or sid == 'None':
                    continue
                try:
                    s = float(sref.get('stoichiometry', '1'))
                except (TypeError, ValueError):
                    s = 1.0
                stoich[sid] = sign * s
        if stoich:
            reactions[rid] = stoich
    return species_ids, reactions


def parse_reactions_auto(
    path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Dispatch by file extension. .json -> COBRA, .xml -> SBML."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in ('.json',):
        return parse_cobra_reactions(p)
    if suffix in ('.xml', '.sbml'):
        return parse_sbml_reactions(p)
    raise ValueError(f'unrecognized reaction model file: {p}')


# ---------------------------------------------------------------------------
# Alias matching — handle the M_*_c vs *_c convention divergence
# ---------------------------------------------------------------------------
def _alias_candidates(row_name: str):
    """Yield possible SBML IDs for a given row name. Order matters —
    earlier candidates win in a tie."""
    yield row_name                                  # direct
    if not row_name.startswith('M_'):
        yield f'M_{row_name}'                       # add prefix
    if row_name.startswith('M_'):
        yield row_name[2:]                          # strip prefix


def _build_alias_map(
    row_names: List[str], sbml_set: set,
) -> Dict[str, str]:
    """For each row, pick the first alias candidate that hits sbml_set.
    Returns {row_name: matched_sbml_id}."""
    out: Dict[str, str] = {}
    for r in row_names:
        for cand in _alias_candidates(r):
            if cand in sbml_set:
                out[r] = cand
                break
    return out


# ---------------------------------------------------------------------------
# Diagnostic — what conventions appear in this row list?
# ---------------------------------------------------------------------------
def diagnose_row_matching(
    row_names: List[str], reaction_model_path: Path,
) -> dict:
    """Run the multi-convention match against the model and return a
    structured report. Useful when n_sbml_match comes back small and
    you don't know whether the convention or the data is at fault."""
    import collections
    sbml_ids, _ = parse_reactions_auto(reaction_model_path)
    sbml_set = set(sbml_ids)
    prefixes = collections.Counter(r.split('_')[0] for r in row_names)
    alias_map = _build_alias_map(row_names, sbml_set)
    examples = {p: [r for r in row_names if r.startswith(p + '_') or r == p][:3]
                for p in [k for k, _ in prefixes.most_common(15)]}
    by_alias_rule = {
        'direct':      sum(1 for r in row_names if r in sbml_set),
        'add_M_':      sum(1 for r in row_names
                           if (not r.startswith('M_')) and f'M_{r}' in sbml_set),
        'strip_M_':    sum(1 for r in row_names
                           if r.startswith('M_') and r[2:] in sbml_set),
    }
    return {
        'n_rows':            len(row_names),
        'n_sbml_species':    len(sbml_set),
        'n_total_match':     len(alias_map),
        'matches_per_rule':  by_alias_rule,
        'top_prefixes':      dict(prefixes.most_common(15)),
        'examples_by_prefix': examples,
        'sample_sbml_ids':   sbml_ids[:10],
        'sample_unmatched':  [r for r in row_names
                              if r not in alias_map][:10],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
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


def build_species_graph(
    row_names: List[str],
    reaction_model_path: Path,
) -> SpeciesGraph:
    """Construct a SpeciesGraph for the given row order.

    `reaction_model_path` may be a COBRA JSON (.json) or an SBML XML
    (.xml/.sbml). Multi-convention aliasing handles M_*_c vs *_c
    divergence. Unmatched rows get a self-loop only.
    """
    sbml_ids, reactions = parse_reactions_auto(Path(reaction_model_path))
    sbml_set = set(sbml_ids)
    alias = _build_alias_map(row_names, sbml_set)        # row -> sbml_id

    # Reverse: sbml_id -> row index. Multiple rows could alias to the
    # same sbml_id in pathological cases; keep the first.
    sbml_to_row_idx: Dict[str, int] = {}
    for i, r in enumerate(row_names):
        sid = alias.get(r)
        if sid is not None and sid not in sbml_to_row_idx:
            sbml_to_row_idx[sid] = i

    src_list, dst_list = [], []
    attrs: List[List[float]] = []
    kinds: List[int] = []
    rxn_ids: List[Optional[str]] = []

    # SBML edges from co-occurrence in reactions
    for rxn_id, stoich in reactions.items():
        species_in_rxn = [(sid, st) for sid, st in stoich.items()
                          if sid in sbml_to_row_idx]
        for a in range(len(species_in_rxn)):
            for b in range(len(species_in_rxn)):
                if a == b:
                    continue
                sid_a, st_a = species_in_rxn[a]
                sid_b, st_b = species_in_rxn[b]
                i = sbml_to_row_idx[sid_a]
                j = sbml_to_row_idx[sid_b]
                src_list.append(i); dst_list.append(j)
                prod = float(st_a) * float(st_b)
                attrs.append([float(st_a), float(st_b),
                              float(np.sign(prod)),
                              float(abs(prod)),
                              0.0])                       # is_self=0
                kinds.append(0)
                rxn_ids.append(rxn_id)

    # Self-loop on every node (matched and unmatched alike)
    for i in range(len(row_names)):
        src_list.append(i); dst_list.append(i)
        attrs.append([0.0, 0.0, 0.0, 0.0, 1.0])           # is_self=1
        kinds.append(1)
        rxn_ids.append(None)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attrs, dtype=torch.float32)
    edge_kind  = torch.tensor(kinds, dtype=torch.long)
    sbml_match = torch.tensor([r in alias for r in row_names],
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
