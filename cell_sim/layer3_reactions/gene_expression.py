"""
Priority 2: Gene expression events.

Implements the central dogma as event-driven rules:
  - Transcription: Gene + RNAP → mRNA (consumes NTP)
  - Translation: mRNA + Ribosome → Protein (consumes amino acids / GTP)
  - mRNA degradation: mRNA + Degradosome → NMPs
  - Protein degradation: Protein → amino acids (slow, first-order)

Rate parameters from the Gene Expression sheet of kinetic_params.xlsx:
  - Transcription elongation: 85 nt/s
  - Translation elongation: 12 aa/s
  - mRNA degradation: 88 nt/s
  - RNAP binding rate to mRNA gene: 2100 /M/s × promoter strength S
  - Ribosome binding rate: 890,000 /M/s
  - Degradosome binding rate: 140,000 /M/s

Simplifications (and why):
  1. Transcription is a single "complete mRNA" event. Real model tracks
     elongation step-by-step. We precompute event time from gene_length/k_cat
     and fire atomically. Equivalent at steady state, less visible at short
     timescales.
  2. NTP pool lumped: we consume from a single "ntp" pool (really the sum
     of ATP+UTP+CTP+GTP), decrementing count = gene_length * stoich_ntp.
     Real model tracks each separately.
  3. Amino acid pool lumped the same way for translation.
  4. Ribosomes and RNAP are not physical ProteinInstances — we use effective
     counts scaled from proteomics. This avoids needing ribosome biogenesis
     (which is Priority 3+).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from layer2_field.dynamics import (
    CellState, TransitionRule,
)
from layer3_reactions.coupled import (
    get_species_count, update_species_count, AVOGADRO,
    count_to_mM, mM_to_count,
)


DATA_ROOT = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data'


# ============================================================================
# Gene expression parameters (from kinetic_params.xlsx Gene Expression sheet)
# ============================================================================
TRANSCRIPTION_ELONGATION_NT_PER_S = 85.0    # nt/s, tRNA/rRNA
TRANSLATION_ELONGATION_AA_PER_S = 12.0      # aa/s
MRNA_DEGRADATION_NT_PER_S = 88.0            # nt/s
PROTEIN_HALF_LIFE_S = 25 * 60               # ~25 min — typical bacterial, ~105 min cell cycle

# These are rate constants k (binding) in /M/s from the paper
K_BIND_RNAP_MRNA_BASE = 2100.0              # ×S (promoter strength)
K_BIND_RIBOSOME_MRNA = 890_000.0
K_BIND_DEGRADOSOME_MRNA = 140_000.0

# Effective counts (from Luthey-Schulten 2022 — roughly scaled by proteomics)
# We'll load these from the mRNA sheet too
DEFAULT_RNAP_COUNT = 140       # ~140 per cell (minus a few for rRNA)
DEFAULT_RIBOSOME_COUNT = 500   # ~500 per cell
DEFAULT_DEGRADOSOME_COUNT = 120


# ============================================================================
# mRNA State — added to CellState
# ============================================================================
def load_initial_mrna_counts(scale_factor: float = 1.0) -> Dict[str, float]:
    """
    Load initial mRNA counts per gene from the mRNA Count sheet.

    Returns {locus_tag: total_mRNA_count}. Values are typically < 3 per gene
    (sparse — most genes have fractional or zero mRNAs on average).
    """
    df = pd.read_excel(DATA_ROOT / 'initial_concentrations.xlsx', sheet_name='mRNA Count')
    out = {}
    for _, row in df.iterrows():
        locus = str(row.get('LocusTag', '')).strip()
        if not locus.startswith('JCVISYN3A'):
            continue
        try:
            total = float(row.get('total', 0))
            out[locus] = total * scale_factor
        except (ValueError, TypeError):
            pass
    return out


def initialize_gene_expression_state(
    state: CellState,
    scale_factor: float = 0.1,
    rnap_count: int = DEFAULT_RNAP_COUNT,
    ribosome_count: int = DEFAULT_RIBOSOME_COUNT,
    degradosome_count: int = DEFAULT_DEGRADOSOME_COUNT,
    seed: int = 42,
):
    """
    Add mRNA counts, shared-pool counts (NTP, AA), and machinery counts
    to the state.

    `seed` controls the stochastic rounding of fractional mRNA counts. Must
    match the EventSimulator seed for bit-identical reproducibility across
    runs. The previous behaviour (np.random.random()) pulled from the
    global RNG and caused ~0.1% drift in Priority 2 totals between runs
    with the same sim seed.
    """
    rng = np.random.default_rng(seed)

    # mRNA counts per gene (iterate in sorted order for determinism)
    state.mrna_counts: Dict[str, int] = {}
    raw_counts = load_initial_mrna_counts(scale_factor=scale_factor)
    total_mrnas = 0
    for locus in sorted(state.spec.proteins):
        avg = raw_counts.get(locus, 0.0)
        # Round stochastically, deterministically
        count = int(np.floor(avg)) + (1 if rng.random() < (avg - np.floor(avg)) else 0)
        state.mrna_counts[locus] = count
        total_mrnas += count

    # Machinery
    state.rnap_free = int(rnap_count * scale_factor) or 1
    state.ribosome_free = int(ribosome_count * scale_factor) or 1
    state.degradosome_free = int(degradosome_count * scale_factor) or 1

    # Shared pools as simple counts
    # Pool NTP from the 4 individual ones
    # ATP alone is ~70k molecules at scale 0.02 — shared with metabolism already
    state.ntp_pool_shared = True  # means we use metabolite counts (ATP, GTP, CTP, UTP)
    state.aa_pool_shared = True

    # Compute promoter strength per gene (proportional to protein count at steady state)
    # S = total_ptn_abundance_of_gene / average_ptn_abundance
    # Approximated from initial protein instance counts
    avg_count = max(1, np.mean([
        len(state.proteins_by_gene.get(g, set()))
        for g in state.spec.proteins
    ]))
    state.promoter_strength: Dict[str, float] = {}
    for g in state.spec.proteins:
        n_ptn = len(state.proteins_by_gene.get(g, set()))
        state.promoter_strength[g] = n_ptn / avg_count if n_ptn > 0 else 0.1

    return {
        'mrnas': total_mrnas,
        'rnap_free': state.rnap_free,
        'ribosome_free': state.ribosome_free,
        'degradosome_free': state.degradosome_free,
    }


# ============================================================================
# Transcription rule
# ============================================================================
def make_transcription_rule(
    gene_id: str,
    gene_length_nt: int,
) -> TransitionRule:
    """
    Transcription of one gene → one mRNA.

    Completion rate per gene depends on:
      - Promoter strength S (approximates RNAP binding affinity)
      - Free RNAP count
      - Elongation time = gene_length_nt / 85 nt/s
      - NTP availability (each mRNA costs ~gene_length_nt ATP-equivalents)

    Effective completion rate ≈ k_bind * RNAP_count * S / elongation_time
    """
    name = f'transcribe:{gene_id}'
    length_nt = max(100, gene_length_nt)
    elongation_time = length_nt / TRANSCRIPTION_ELONGATION_NT_PER_S
    # Completions per second ≈ 1/elongation_time × (fraction of RNAPs bound here)
    # Simplified: rate = k_bind_M_per_s × S × RNAP_count / cell_volume, adjusted for completion
    # For the Gillespie rule, we use an effective rate per event

    # The rate is derived from: binding flux × probability of completing elongation
    # We simplify to: rate_per_event = promoter_strength * RNAP_free / elongation_time / scale_denom
    # scale_denom tuned so total mRNA production ~balances degradation at steady state

    def can_fire(state):
        if state.rnap_free <= 0:
            return []
        # Need NTPs to transcribe — check ATP as proxy
        ntp_cost = length_nt // 4  # a rough 1/4 of NTPs are ATP
        if get_species_count(state, 'M_atp_c') < ntp_cost:
            return []
        S = state.promoter_strength.get(gene_id, 0.1)
        # Propensity tokens proportional to S * free RNAP
        return [('tx', S)] * max(1, int(S * state.rnap_free))

    def apply(state, cands, rng):
        if state.rnap_free <= 0:
            return
        # Check NTPs
        ntp_cost_each = length_nt // 4
        atp = get_species_count(state, 'M_atp_c')
        if atp < ntp_cost_each:
            return
        # Consume NTPs (approximate — hit each NTP pool equally)
        for pool in ['M_atp_c', 'M_gtp_c', 'M_ctp_c', 'M_utp_c']:
            if get_species_count(state, pool) >= ntp_cost_each:
                update_species_count(state, pool, -ntp_cost_each)
        # Produce NMPs from NTP hydrolysis (2 of 3 phosphates released)
        # Simplified: each NTP → NMP + PPi, PPi → 2 Pi by PPA
        update_species_count(state, 'M_pi_c', +length_nt * 2)
        # Increment mRNA
        state.mrna_counts[gene_id] = state.mrna_counts.get(gene_id, 0) + 1
        # RNAP released
        state.log_event(
            f'transcribe:{gene_id}', [gene_id],
            f'mRNA({gene_id}) transcribed ({length_nt}nt, cost ~{length_nt} NTP)')

    # Rate: events per sec per "token". Each token represents one active RNAP-gene pair.
    # Per-token completion rate = 1 / elongation_time.
    rate = 1.0 / elongation_time
    return TransitionRule(
        name=name,
        participants=[gene_id],
        rate=rate,
        rate_source='literature_kozo',
        can_fire=can_fire,
        apply=apply,
    )


# ============================================================================
# Translation rule
# ============================================================================
def make_translation_rule(
    gene_id: str,
    protein_length_aa: int,
) -> TransitionRule:
    """
    Translation of one mRNA → one protein.

    Rate limited by:
      - Free ribosomes
      - mRNA count for this gene
      - Amino acid availability
      - Elongation time = protein_length / 12 aa/s
    """
    name = f'translate:{gene_id}'
    length_aa = max(10, protein_length_aa)
    elongation_time = length_aa / TRANSLATION_ELONGATION_AA_PER_S

    def can_fire(state):
        n_mrna = state.mrna_counts.get(gene_id, 0)
        if n_mrna <= 0 or state.ribosome_free <= 0:
            return []
        # Need amino acids — we use aa pool proxy via glycine
        # (proper would be all 20; use total AA budget via alanine)
        if get_species_count(state, 'M_ala__L_c') < length_aa // 20:
            return []
        # Tokens ≈ min(mRNA * ribosome)
        return [('tl',)] * min(n_mrna * state.ribosome_free, 100)

    def apply(state, cands, rng):
        n_mrna = state.mrna_counts.get(gene_id, 0)
        if n_mrna <= 0 or state.ribosome_free <= 0:
            return
        # Consume amino acids (evenly from all 20; approximate by taking from alanine + glycine)
        aa_per_type = max(1, length_aa // 20)
        for aa in ['M_ala__L_c', 'M_gly_c', 'M_ser__L_c', 'M_leu__L_c']:
            c = get_species_count(state, aa)
            if c >= aa_per_type:
                update_species_count(state, aa, -aa_per_type)
        # Consume GTP (2 per peptide bond — simplified)
        gtp_cost = length_aa * 2
        if get_species_count(state, 'M_gtp_c') >= gtp_cost:
            update_species_count(state, 'M_gtp_c', -gtp_cost)
            update_species_count(state, 'M_gdp_c', +gtp_cost)
        # Create a new protein instance
        state.new_protein(gene_id, conformation='unfolded')  # starts unfolded
        # mRNA stays (ribosomes can re-initiate)
        state.log_event(
            f'translate:{gene_id}', [gene_id],
            f'protein({gene_id}) synthesized ({length_aa}aa, cost ~{gtp_cost} GTP)')

    rate = 1.0 / elongation_time
    return TransitionRule(
        name=name,
        participants=[gene_id],
        rate=rate,
        rate_source='literature_kozo',
        can_fire=can_fire,
        apply=apply,
    )


# ============================================================================
# mRNA degradation rule
# ============================================================================
def make_mrna_degradation_rule(gene_id: str, gene_length_nt: int) -> TransitionRule:
    """mRNA → NMPs. Per-mRNA first-order decay at k = 88 nt/s / gene_length."""
    name = f'degrade_mrna:{gene_id}'
    length_nt = max(100, gene_length_nt)
    k_degrade = MRNA_DEGRADATION_NT_PER_S / length_nt

    def can_fire(state):
        n = state.mrna_counts.get(gene_id, 0)
        if n <= 0 or state.degradosome_free <= 0:
            return []
        return [('deg',)] * min(n * state.degradosome_free, 50)

    def apply(state, cands, rng):
        n = state.mrna_counts.get(gene_id, 0)
        if n <= 0:
            return
        state.mrna_counts[gene_id] = n - 1
        # Recycle NMPs (we just increment Pi as aggregate)
        update_species_count(state, 'M_pi_c', +length_nt)
        state.log_event(
            f'degrade_mrna:{gene_id}', [gene_id],
            f'mRNA({gene_id}) degraded ({length_nt}nt recycled)')

    return TransitionRule(
        name=name,
        participants=[gene_id],
        rate=k_degrade,
        rate_source='literature',
        can_fire=can_fire,
        apply=apply,
    )


# ============================================================================
# Protein degradation rule
# ============================================================================
def make_protein_degradation_rule(gene_id: str, half_life_s: float = PROTEIN_HALF_LIFE_S):
    """Protein → amino acids. First-order decay."""
    name = f'degrade_protein:{gene_id}'
    k_decay = np.log(2) / half_life_s  # per molecule per second

    def can_fire(state):
        # Only degrade native or unfolded, not bound
        natives = state.proteins_by_state.get(f'{gene_id}:native', set())
        unfolded = state.proteins_by_state.get(f'{gene_id}:unfolded', set())
        return list(natives) + list(unfolded)

    def apply(state, cands, rng):
        if not cands:
            return
        pid = cands[rng.integers(0, len(cands))]
        if pid not in state.proteins:
            return
        gene = state.proteins[pid].gene_id
        state.remove_protein(pid)
        # Recycle amino acids (rough — add to alanine pool as proxy)
        length_aa = state.spec.proteins.get(gene, type('', (), {'length': 100})()).length or 100
        update_species_count(state, 'M_ala__L_c', +length_aa // 20)
        state.log_event(
            f'degrade_protein:{gene}', [pid],
            f'protein({gene}) degraded (AAs recycled)')

    return TransitionRule(
        name=name,
        participants=[gene_id],
        rate=k_decay,
        rate_source='literature_median',
        can_fire=can_fire,
        apply=apply,
    )


# ============================================================================
# Build all gene expression rules
# ============================================================================
def build_gene_expression_rules(spec, max_genes: Optional[int] = None,
                                 include_degradation: bool = True) -> List[TransitionRule]:
    """
    Build gene expression rules for all (or first max_genes) genes.

    Returns list of rules: transcription + translation + mRNA degradation +
    protein degradation.
    """
    rules = []
    gene_list = list(spec.proteins.items())
    if max_genes:
        gene_list = gene_list[:max_genes]

    for gene_id, protein in gene_list:
        length_aa = max(10, protein.length or 100)
        length_nt = length_aa * 3 + 100  # crude: 3 nt per aa + 100 UTR

        rules.append(make_transcription_rule(gene_id, length_nt))
        rules.append(make_translation_rule(gene_id, length_aa))
        if include_degradation:
            rules.append(make_mrna_degradation_rule(gene_id, length_nt))
            rules.append(make_protein_degradation_rule(gene_id))

    return rules


# Add a helper to the dynamics module so protein degradation can find proteins
def _patch_cellstate():
    """Add remove_protein method to CellState if missing."""
    from layer2_field.dynamics import CellState as CS

    if hasattr(CS, 'remove_protein'):
        return

    def remove_protein(self, inst_id: str):
        if inst_id not in self.proteins:
            return
        p = self.proteins[inst_id]
        gene = p.gene_id
        conf = p.conformation
        # Remove from indices
        if gene in self.proteins_by_gene and inst_id in self.proteins_by_gene[gene]:
            self.proteins_by_gene[gene].discard(inst_id)
        key = f'{gene}:{conf}'
        if key in self.proteins_by_state and inst_id in self.proteins_by_state[key]:
            self.proteins_by_state[key].discard(inst_id)
        # Remove
        del self.proteins[inst_id]

    CS.remove_protein = remove_protein


_patch_cellstate()


# ============================================================================
# Demo
# ============================================================================
if __name__ == "__main__":
    import io, contextlib, time
    from layer0_genome.syn3a_real import build_real_syn3a_cellspec
    from layer2_field.dynamics import EventSimulator
    from layer2_field.real_syn3a_rules import (
        populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
    )
    from layer3_reactions.sbml_parser import parse_sbml
    from layer3_reactions.kinetics import load_all_kinetics, load_medium
    from layer3_reactions.reversible import build_reversible_catalysis_rules, initialize_medium
    from layer3_reactions.coupled import initialize_metabolites

    print('=' * 60)
    print('Priority 2: Gene expression (transcription/translation)')
    print('=' * 60)

    print('\nLoading...')
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()
    sbml = parse_sbml(Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data' / 'Syn3A_updated.xml')
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=0.02, max_per_gene=10)
    initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    # Initialize gene expression
    gex_stats = initialize_gene_expression_state(state, scale_factor=0.1)
    print(f'\n  Gene expression initial state:')
    print(f'    mRNAs:          {gex_stats["mrnas"]}')
    print(f'    Free RNAP:      {gex_stats["rnap_free"]}')
    print(f'    Free ribosomes: {gex_stats["ribosome_free"]}')
    print(f'    Free degradosomes: {gex_stats["degradosome_free"]}')

    # Build rules — only for a subset of genes to keep tractable
    # Focus on the top-expressed genes
    top_genes = sorted(state.spec.proteins.keys(),
                        key=lambda g: state.mrna_counts.get(g, 0),
                        reverse=True)[:50]
    print(f'  Focusing gene expression on top-{len(top_genes)} expressed genes')

    # Build just rules for these
    gex_rules = []
    for g in top_genes:
        p = spec.proteins[g]
        length_aa = max(10, p.length or 100)
        length_nt = length_aa * 3 + 100
        gex_rules.append(make_transcription_rule(g, length_nt))
        gex_rules.append(make_translation_rule(g, length_aa))
        gex_rules.append(make_mrna_degradation_rule(g, length_nt))
        gex_rules.append(make_protein_degradation_rule(g))

    print(f'  Built {len(gex_rules)} gene expression rules ({len(top_genes)} genes × 4)')

    all_rules = (
        [make_folding_rule(20.0)]
        + rev_rules
        + make_complex_formation_rules(complexes, base_rate_per_s=0.05)
        + gex_rules
    )
    print(f'  Total rules: {len(all_rules)}')

    # Snapshot initial proteins per gene
    initial_proteins = {g: len(state.proteins_by_gene.get(g, set())) for g in top_genes}
    initial_mrnas = dict(state.mrna_counts)

    # Run for 10 seconds of simulated time — gene expression needs more time
    sim_time = 10.0
    print(f'\nRunning {sim_time}s simulated time (long enough to see translation)...')
    sim = EventSimulator(state, all_rules, mode='gillespie', seed=42)
    t0 = time.time()
    run_stats = sim.run_until(t_end=sim_time, max_events=2_000_000)
    wall = time.time() - t0
    print(f'  {run_stats["n_events"]:,} events in {wall:.1f}s wall '
          f'({run_stats["events_per_wall_sec"]:.0f} events/sec)')
    print(f'  Simulated: {state.time:.2f}s')

    # Breakdown
    from collections import Counter
    event_types = Counter()
    for e in state.events:
        parts = e.rule_name.split(':')
        event_types[parts[0]] += 1
    print('\n  Event breakdown:')
    for t, n in event_types.most_common():
        print(f'    {t}: {n:,}')

    # Protein changes
    print(f'\n  Protein count changes (top 10 expressed genes):')
    print(f'  {"Gene":18s}  {"Initial":>8s}  {"Final":>8s}  {"Δ":>8s}  mRNA(i→f)')
    for g in top_genes[:10]:
        init = initial_proteins[g]
        final = len(state.proteins_by_gene.get(g, set()))
        mi = initial_mrnas.get(g, 0)
        mf = state.mrna_counts.get(g, 0)
        gn = spec.proteins[g].annotations.get('gene_name', '')[:10]
        print(f'    {g}/{gn:10s}  {init:>8d}  {final:>8d}  {final-init:>+8d}  '
              f'{mi:>4.0f}→{mf:<4d}')

    # Total protein count
    total_init = sum(initial_proteins.values())
    total_final = sum(len(state.proteins_by_gene.get(g, set())) for g in top_genes)
    print(f'\n  Total tracked proteins: {total_init} → {total_final}  '
          f'(Δ{total_final-total_init:+d})')

    # Events
    for evt in state.events[:5]:
        print(f'  {evt.time*1000:.3f}ms: {evt.description[:80]}')
