"""Build the species×species reaction graph from iMB155 (COBRA JSON) or
the local Syn3A_updated.xml (SBML XML), plus *simulator-template edges*
that link gene/mRNA/protein/complex rows along the central-dogma chain
defined in the Luthey-Schulten Lattice Microbes simulator.

The Luthey-Schulten simulator emits trajectories using `M_atp_c`-style
metabolite IDs; iMB155's COBRA JSON sometimes uses bare `atp_c`. Multiple
alias conventions are tried so the graph builder works regardless of
which SBML source is supplied.

For every reaction r we enumerate all unordered pairs (s_i, s_j) of
metabolites that participate in r and emit a directed edge in each
direction with attributes `(stoich_i, stoich_j, sign, |stoich_i*stoich_j|)`.

The 8572 rows of counts_and_fluxes are a superset of the SBML
metabolites — proteins, mRNAs, complexes, and cost accumulators are
not in any flux-balance model. Two extensions handle this:

  1. Unmatched rows that share a locus number with other rows along
     the simulator's central-dogma template (G_xxxx, R_xxxx, P_xxxx,
     RP_xxxx, RPM_xxxx, PM_xxxx, DM_xxxx, D_xxxx) get *simulator
     edges* between adjacent stages of the chain. Pattern source:
     cell_sim/data/Minimal_Cell_ComplexFormation/programs/rxns_CME.py.
  2. Anything still unmatched gets a self-loop with `is_self=1` so
     downstream message passing falls back to a per-node MLP.

Output is a torch dict savable via `torch.save`:
    {
      'edge_index'  : (2, E)   long, [src; dst]
      'edge_attr'   : (E, 5)   float, [stoich_i, stoich_j, sign, mag, is_self]
      'edge_kind'   : (E,)     long; values defined in EdgeKind below
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
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class EdgeKind(IntEnum):
    """Routing tag for the GNN's hetero-edge message MLPs."""
    SBML            = 0
    SELF_LOOP       = 1
    TRANSCRIPTION   = 2
    TRANSLATION     = 3
    TRANSLOCATION   = 4
    DEGRADATION     = 5
    FLUX_COUPLING   = 6      # F_<rxn> ↔ each species in rxn — see communicate.py:216


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
    reaction_model_path: Optional[Path] = None,
    include_simulator_edges: bool = True,
) -> SpeciesGraph:
    """Construct a SpeciesGraph for the given row order.

    `reaction_model_path` may be a COBRA JSON (.json) or an SBML XML
    (.xml/.sbml). When None, only simulator + self-loop edges are
    emitted. Multi-convention aliasing handles M_*_c vs *_c divergence.

    `include_simulator_edges=True` adds central-dogma chain edges
    (transcription/translation/translocation/degradation) between rows
    that share a locus number along the simulator's template — this is
    what gives the ~6900 protein/mRNA/complex rows real connectivity.
    """
    if reaction_model_path is not None:
        sbml_ids, reactions = parse_reactions_auto(Path(reaction_model_path))
        sbml_set = set(sbml_ids)
    else:
        sbml_ids, reactions, sbml_set = [], {}, set()
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
                kinds.append(int(EdgeKind.SBML))
                rxn_ids.append(rxn_id)

    # Simulator central-dogma edges (transcription/translation/etc)
    if include_simulator_edges:
        for i, j, kind, label in build_simulator_edges(row_names):
            src_list.append(i); dst_list.append(j)
            attrs.append([0.0, 0.0, 0.0, 0.0, 0.0])       # no stoich
            kinds.append(int(kind))
            rxn_ids.append(label)

    # Flux↔species coupling edges. The simulator writes one flux row
    # per metabolic reaction r as F_<r> (period-averaged) and an
    # optional F_<r>_end (end-of-step). For every SBML reaction whose
    # flux row is present, link the flux row to each participating
    # species (both directions, same EdgeKind for now). Directionality
    # discussion: a flux is a rate so flux→species is the causal
    # direction (rate determines next species count); species→flux is
    # the rate-law direction (upstream concentrations set the rate).
    # Both are real and the GNN's hetero-MLP can learn the asymmetry
    # from this single edge type. Split into two EdgeKinds later if a
    # directed message-passing variant is added.
    name_to_idx = {n: i for i, n in enumerate(row_names)}
    for rxn_id, stoich in reactions.items():
        for flux_name in (f'F_{rxn_id}', f'F_{rxn_id}_end'):
            flux_idx = name_to_idx.get(flux_name)
            if flux_idx is None:
                continue
            for sid in stoich:
                spec_idx = sbml_to_row_idx.get(sid)
                if spec_idx is None or spec_idx == flux_idx:
                    continue
                # flux → species
                src_list.append(flux_idx); dst_list.append(spec_idx)
                attrs.append([0.0, 0.0, 0.0, 0.0, 0.0])
                kinds.append(int(EdgeKind.FLUX_COUPLING))
                rxn_ids.append(f'flux:{rxn_id}')
                # species → flux
                src_list.append(spec_idx); dst_list.append(flux_idx)
                attrs.append([0.0, 0.0, 0.0, 0.0, 0.0])
                kinds.append(int(EdgeKind.FLUX_COUPLING))
                rxn_ids.append(f'flux:{rxn_id}')

    # Self-loop on every node (matched and unmatched alike)
    for i in range(len(row_names)):
        src_list.append(i); dst_list.append(i)
        attrs.append([0.0, 0.0, 0.0, 0.0, 1.0])           # is_self=1
        kinds.append(int(EdgeKind.SELF_LOOP))
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


# ---------------------------------------------------------------------------
# Simulator-template edges — central-dogma chain from rxns_CME.py
# ---------------------------------------------------------------------------
# Pattern: <prefix>_<locus>[_<state>] e.g., G_0001, RP_0001_C1.
# Locus must be 3+ digits so we don't false-match metabolites like
# M_3pg_c (3-phosphoglycerate) where the digit is part of the chemical
# name, not a JCVISYN3A locus tag.
_LOCUS_RE = re.compile(r'^([A-Za-z][A-Za-z]*?)_(\d{3,})(_.+)?$')

# Each tuple: (from_prefix, to_prefix, EdgeKind, human_label).
# Both directions are emitted per pair.
_CENTRAL_DOGMA_PATTERNS = [
    ('G',   'RP',  EdgeKind.TRANSCRIPTION, 'sim:rnap_binding'),
    ('RP',  'R',   EdgeKind.TRANSCRIPTION, 'sim:transcription_elongation'),
    ('R',   'RPM', EdgeKind.TRANSLATION,   'sim:ribosome_binding'),
    ('RPM', 'P',   EdgeKind.TRANSLATION,   'sim:translation_elongation'),
    ('RPM', 'PM',  EdgeKind.TRANSLATION,   'sim:translation_elongation_membrane'),
    ('P',   'PM',  EdgeKind.TRANSLOCATION, 'sim:membrane_insertion'),
    ('R',   'DM',  EdgeKind.DEGRADATION,   'sim:degradosome_binding'),
    ('DM',  'D',   EdgeKind.DEGRADATION,   'sim:mrna_degradation'),
]


def extract_locus_groups(
    row_names: List[str],
) -> Dict[str, Dict[str, List[int]]]:
    """{locus: {prefix: [row_index, ...]}}.

    Multiple states per (prefix, locus) pair (e.g. RP_0001 + RP_0001_C1)
    accumulate in the same list.
    """
    groups: Dict[str, Dict[str, List[int]]] = {}
    for i, r in enumerate(row_names):
        m = _LOCUS_RE.match(r)
        if m is None:
            continue
        prefix, locus = m.group(1), m.group(2)
        groups.setdefault(locus, {}).setdefault(prefix, []).append(i)
    return groups


def build_simulator_edges(row_names: List[str]):
    """(src, dst, EdgeKind, label) tuples for every central-dogma pair
    between rows that share a locus tag. Symmetric — both directions."""
    groups = extract_locus_groups(row_names)
    out = []
    for locus, prefix_to_indices in groups.items():
        for from_p, to_p, kind, label in _CENTRAL_DOGMA_PATTERNS:
            from_indices = prefix_to_indices.get(from_p, [])
            to_indices   = prefix_to_indices.get(to_p, [])
            for i in from_indices:
                for j in to_indices:
                    if i == j:
                        continue
                    out.append((i, j, kind, label))
                    out.append((j, i, kind, label))
    return out


def simulator_edge_summary(row_names: List[str]) -> dict:
    """Per-pattern edge counts + locus stats."""
    groups = extract_locus_groups(row_names)
    by_pattern = {label: 0 for *_, label in _CENTRAL_DOGMA_PATTERNS}
    for locus, prefix_to_indices in groups.items():
        for from_p, to_p, kind, label in _CENTRAL_DOGMA_PATTERNS:
            n_from = len(prefix_to_indices.get(from_p, []))
            n_to   = len(prefix_to_indices.get(to_p, []))
            by_pattern[label] += n_from * n_to * 2     # both directions
    n_loci = len(groups)
    n_loci_with_full_chain = sum(
        1 for _, p2i in groups.items()
        if all(p in p2i for p in ('G', 'R', 'P'))
    )
    return {
        'n_loci_seen':              n_loci,
        'n_loci_with_G_R_P_all':    n_loci_with_full_chain,
        'edges_per_pattern':        by_pattern,
        'total_simulator_edges':    sum(by_pattern.values()),
        'sample_loci':              sorted(groups)[:10],
    }


def flux_edge_summary(
    row_names: List[str],
    reaction_model_path: Path,
) -> dict:
    """How many SBML reactions have an F_<rxn_id> row? Diagnostic for
    the flux-coupling layer — if low, the metabolic outputs are still
    orphan in the graph."""
    name_set = set(row_names)
    _, reactions = parse_reactions_auto(Path(reaction_model_path))
    n_avg, n_end, n_with_either, samples = 0, 0, 0, []
    for rxn_id in reactions:
        has_avg = f'F_{rxn_id}' in name_set
        has_end = f'F_{rxn_id}_end' in name_set
        n_avg += int(has_avg)
        n_end += int(has_end)
        if has_avg or has_end:
            n_with_either += 1
            if len(samples) < 8:
                samples.append((rxn_id, has_avg, has_end))
    return {
        'n_sbml_reactions':       len(reactions),
        'n_with_F_avg_row':       n_avg,
        'n_with_F_end_row':       n_end,
        'n_reactions_covered':    n_with_either,
        'sample_with_flux_rows':  samples,
        'n_unmatched_reactions':  len(reactions) - n_with_either,
    }


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
    """Edge counts per EdgeKind + per-node connectivity stats."""
    n_per_kind = {k.name: int((g.edge_kind == int(k)).sum().item())
                  for k in EdgeKind}
    matched_nodes = int(g.sbml_match.sum().item())
    # Non-self-loop in-degree per node
    non_self_mask = g.edge_kind != int(EdgeKind.SELF_LOOP)
    if non_self_mask.any():
        dst = g.edge_index[1][non_self_mask]
        deg = torch.bincount(dst, minlength=g.n_nodes)
    else:
        deg = torch.zeros(g.n_nodes, dtype=torch.long)
    n_orphans = int((deg == 0).sum().item())          # only self-loop
    return {
        'n_nodes':                  g.n_nodes,
        'n_sbml_match':             matched_nodes,
        'n_sbml_unmatched':         g.n_nodes - matched_nodes,
        'edges_per_kind':           n_per_kind,
        'n_total_edges':            g.n_edges,
        'mean_in_degree_excl_self': float(deg.float().mean()),
        'max_in_degree_excl_self':  int(deg.max()) if g.n_nodes > 0 else 0,
        'n_orphan_nodes':           n_orphans,
        'unique_reactions':         len({r for r in g.reaction_id
                                          if r is not None}),
    }
