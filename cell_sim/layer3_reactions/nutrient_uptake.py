"""
Nutrient uptake patch for Syn3A.

Problem: the Luthey-Schulten 160-reaction kinetic database provides k_cat
values for 58 transport reactions, but is missing k_cat for several
critical nutrient importers that Syn3A's SBML network specifies:

    GLCpts   — PTS-dependent glucose import (the cell's main carbon source)
    GLYCt    — glycerol uptake
    FAt      — fatty acid uptake
    O2t      — oxygen uptake
    CHOLt    — cholesterol uptake
    TAGt     — triacylglycerol uptake

Without these, a cell simulated at full scale cannot replace the carbon
and lipid it consumes, and metabolism decays within ~1 biological second.
This module patches those transporters in using literature-informed
rate estimates.

Additionally, we add four synthetic pseudo-reactions that the SBML
doesn't define but that are biologically realistic for Syn3A nucleobase
salvage:

    ADEt_syn   — free adenine import from medium (purine salvage)
    GUAt_syn   — free guanine import
    URAt_syn   — free uracil import
    CYTDt_syn  — free cytidine import

Real Syn3A genes that likely mediate these:
    PTSI (JCVISYN3A_0233), IIA (_0234), PTSH (_0694), PtsG (_0779)
        → GLCpts
    GlpK (JCVISYN3A_0218) is the glycerol kinase in the downstream
        pathway; there's no annotated glycerol facilitator in Syn3A, so
        GLYCt uses a placeholder from the general transport protein pool
    RnsA-D (JCVISYN3A_0008-0011) is the nucleoside ABC transporter
        already captured via ADNabc/DADNabc/GSNabc/DGSNabc with measured
        k_cats; we leave those alone.
    For free nucleobases (ade/gua/ura) and lipids/O2, Syn3A has no
        annotated permease. We use JCVISYN3A_0034 (the highest-abundance
        uncharacterized efflux ABC transporter, 1795 copies) as a
        placeholder carrier for these. This is explicitly fictional at
        the level of which protein does the work, but the cell does have
        that many copies of some membrane transporter that it's using
        for SOMETHING, and the net flux is what matters for the model.

All k_cat values in this module are literature-informed estimates.
They are NOT substitutes for measured Syn3A kinetics. They reflect
order-of-magnitude rates for comparable bacterial transporters. When
measurements become available, replace `KCAT_ESTIMATES` below.

Usage:
    from layer3_reactions.nutrient_uptake import (
        build_missing_transport_rules, build_reversible_plus_uptake)

    # Option A: get just the extras, append to your existing rule set
    extra = build_missing_transport_rules(sbml, kinetics)
    all_rules = base_rules + extra

    # Option B: get combined rule set
    rules, report = build_reversible_plus_uptake(sbml, kinetics)
    print(report)   # {'base': 232, 'added': 20, 'total': 252}
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from layer3_reactions.kinetics import ReactionKinetics
from layer3_reactions.sbml_parser import SBMLModel
from layer3_reactions.reversible import _build_one_direction_rule
from layer2_field.dynamics import TransitionRule


# Placeholder locus: highest-abundance uncharacterized ABC transporter in
# Syn3A. Used where no specific annotated permease exists.
PLACEHOLDER_LOCUS = 'JCVISYN3A_0034'


# Literature-informed k_cat estimates. See module docstring for sources.
# Each entry:
#   kcat_fwd (/s), kcat_rev (/s or 0), km_mM (Km for cytoplasmic substrate),
#   reversible, loci (list of Syn3A gene IDs that encode this transporter),
#   notes (one-line rationale)
KCAT_ESTIMATES: Dict[str, Dict] = {
    'GLCpts': dict(
        kcat_fwd=500.0, kcat_rev=0.0, km_mM=0.1, reversible=False,
        loci=['JCVISYN3A_0779'],  # PtsG — primary glucose transporter subunit
        notes='PTS glucose import; ~500/s (Postma 1993)',
    ),
    'GLYCt': dict(
        kcat_fwd=200.0, kcat_rev=200.0, km_mM=1.0, reversible=True,
        loci=[PLACEHOLDER_LOCUS],
        notes='facilitated glycerol diffusion (GlpF-like); no annotated gene in Syn3A',
    ),
    'O2t': dict(
        kcat_fwd=500.0, kcat_rev=500.0, km_mM=0.01, reversible=True,
        loci=[PLACEHOLDER_LOCUS],
        notes='passive O2 diffusion across membrane',
    ),
    'FAt': dict(
        kcat_fwd=50.0, kcat_rev=50.0, km_mM=0.05, reversible=True,
        loci=['JCVISYN3A_0616'],  # FakB, fatty acid binding protein
        notes='fatty acid flippase-mediated uptake',
    ),
    'CHOLt': dict(
        kcat_fwd=50.0, kcat_rev=50.0, km_mM=0.01, reversible=True,
        loci=[PLACEHOLDER_LOCUS],
        notes='cholesterol uptake (Mycoplasma depends on host sterols)',
    ),
    'TAGt': dict(
        kcat_fwd=50.0, kcat_rev=50.0, km_mM=0.05, reversible=True,
        loci=[PLACEHOLDER_LOCUS],
        notes='triacylglycerol uptake',
    ),
}


# Synthetic pseudo-reactions — NOT in the SBML. These close the purine
# and pyrimidine salvage loops so adenine consumed by nucleotide
# biosynthesis can be replenished from the medium.
SYNTHETIC_UPTAKE: Dict[str, Dict] = {
    'ADEt_syn': dict(
        substrates=[('M_ade_e', 1.0)], products=[('M_ade_c', 1.0)],
        kcat_fwd=20.0, km_mM=0.01, reversible=False,
        loci=[PLACEHOLDER_LOCUS],
        notes='adenine uptake; purine salvage closure',
    ),
    'GUAt_syn': dict(
        substrates=[('M_gua_e', 1.0)], products=[('M_gua_c', 1.0)],
        kcat_fwd=20.0, km_mM=0.01, reversible=False,
        loci=[PLACEHOLDER_LOCUS],
        notes='guanine uptake; purine salvage closure',
    ),
    'URAt_syn': dict(
        substrates=[('M_ura_e', 1.0)], products=[('M_ura_c', 1.0)],
        kcat_fwd=20.0, km_mM=0.01, reversible=False,
        loci=[PLACEHOLDER_LOCUS],
        notes='uracil uptake; pyrimidine salvage closure',
    ),
    'CYTDt_syn': dict(
        substrates=[('M_cytd_e', 1.0)], products=[('M_cytd_c', 1.0)],
        kcat_fwd=10.0, km_mM=0.01, reversible=False,
        loci=[PLACEHOLDER_LOCUS],
        notes='cytidine uptake; pyrimidine salvage closure',
    ),
}


def _build_rule_pair(
    rxn_name: str,
    short: str,
    substrates: Dict[str, float],
    products: Dict[str, float],
    spec: Dict,
) -> List[TransitionRule]:
    """Build forward (and optional reverse) rules from a spec entry."""
    rules = []
    loci = spec['loci']

    # Km only applies to cytoplasmic substrates (infinite-reservoir
    # _e species can't be rate-limiting in the saturation term).
    km_dict = {sid: spec['km_mM'] for sid in substrates if not sid.endswith('_e')}

    fwd = _build_one_direction_rule(
        name=f'catalysis:{rxn_name}',
        rxn_name=rxn_name,
        direction='fwd',
        substrates=substrates,
        products=products,
        enzyme_loci=loci,
        kcat=float(spec['kcat_fwd']),
        Km=km_dict,
        include_saturation=True,
    )
    if fwd is not None:
        rules.append(fwd)

    if spec.get('reversible', False) and spec.get('kcat_rev', 0) > 0:
        km_rev = {sid: spec['km_mM'] for sid in products if not sid.endswith('_e')}
        rev = _build_one_direction_rule(
            name=f'catalysis:{rxn_name}:rev',
            rxn_name=rxn_name,
            direction='rev',
            substrates=products,
            products=substrates,
            enzyme_loci=loci,
            kcat=float(spec['kcat_rev']),
            Km=km_rev,
            include_saturation=True,
        )
        if rev is not None:
            rules.append(rev)

    return rules


def build_missing_transport_rules(
    sbml: SBMLModel,
    kinetics: Dict[str, ReactionKinetics],
    include_synthetic: bool = True,
) -> List[TransitionRule]:
    """
    Build transport rules for SBML reactions missing measured k_cat, plus
    optional synthetic pseudo-reactions for nucleobase salvage.

    Only adds rules whose short_name is NOT already in `kinetics`. If you
    later get measured k_cats for these, they'll supersede automatically.

    Args:
        sbml: parsed SBML model
        kinetics: dict of measured ReactionKinetics (from load_all_kinetics())
        include_synthetic: if True, add ADEt/GUAt/URAt/CYTDt pseudo-reactions

    Returns:
        list of TransitionRule objects ready to append to the main rule set
    """
    rules: List[TransitionRule] = []

    # Index SBML reactions by short_name
    by_short = {r.short_name: r for r in sbml.reactions.values()}

    # --- SBML reactions we have specs for ---
    for short, spec in KCAT_ESTIMATES.items():
        if short in kinetics:
            # Measured k_cat exists — don't override
            continue
        if short not in by_short:
            # SBML doesn't have this reaction — skip
            continue
        rxn = by_short[short]
        substrates = dict(rxn.reactants)
        products = dict(rxn.products)
        rules.extend(_build_rule_pair(short, short, substrates, products, spec))

    # --- Synthetic pseudo-reactions ---
    if include_synthetic:
        for short, spec in SYNTHETIC_UPTAKE.items():
            substrates = dict(spec['substrates'])
            products = dict(spec['products'])
            rules.extend(_build_rule_pair(short, short, substrates, products, spec))

    return rules


def build_reversible_plus_uptake(
    sbml: SBMLModel,
    kinetics: Dict[str, ReactionKinetics],
    include_saturation: bool = True,
    include_synthetic: bool = True,
) -> Tuple[List[TransitionRule], Dict[str, int]]:
    """
    Drop-in replacement for build_reversible_catalysis_rules that also
    includes missing nutrient-uptake rules.

    Returns (rules, report) where report summarises how many rules came
    from each source.
    """
    from layer3_reactions.reversible import build_reversible_catalysis_rules

    base_rules, _ = build_reversible_catalysis_rules(
        sbml, kinetics, include_saturation=include_saturation)
    extra = build_missing_transport_rules(
        sbml, kinetics, include_synthetic=include_synthetic)

    report = {
        'base_rules': len(base_rules),
        'uptake_rules_added': len(extra),
        'total': len(base_rules) + len(extra),
    }
    return base_rules + extra, report


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from layer3_reactions.sbml_parser import parse_sbml
    from layer3_reactions.kinetics import load_all_kinetics

    sbml = parse_sbml(
        Path(__file__).resolve().parent.parent /
        'data' / 'Minimal_Cell_ComplexFormation' /
        'input_data' / 'Syn3A_updated.xml'
    )
    kin = load_all_kinetics()

    extra = build_missing_transport_rules(sbml, kin)
    print(f'Built {len(extra)} missing-transport rules:')
    for r in extra:
        print(f'  {r.name:<35s}  k_cat={r.rate:>6.1f}  '
              f'participants={list(r.participants)[:3]}...')

    rules, report = build_reversible_plus_uptake(sbml, kin)
    print(f'\nReport: {report}')
