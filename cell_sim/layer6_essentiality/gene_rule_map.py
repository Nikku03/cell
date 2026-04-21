"""Extract a gene -> set-of-rule-names map from built simulator rules.

The per-rule event-count detector needs to know, for each gene, which
``catalysis:*`` rules its product catalyses. That mapping lives inside
each rule's ``compiled_spec['enzyme_loci']`` attribute, which
:func:`cell_sim.layer3_reactions.reversible.build_reversible_catalysis_rules`
populates from the SBML gene-protein-reaction annotations + the
kinetic_params table.

Usage::

    from cell_sim.layer6_essentiality.gene_rule_map import build_gene_to_rules
    gene_to_rules = build_gene_to_rules(rev_rules + extra_rules)
    # gene_to_rules["JCVISYN3A_0445"] == {"catalysis:PGI", "catalysis:PGI:rev"}

Only rules carrying a ``compiled_spec`` with ``kind == "mm"`` and a
non-empty ``enzyme_loci`` entry are considered. Folding rules and
complex-formation rules don't have an enzyme locus in this sense and
are skipped.
"""
from __future__ import annotations

from typing import Iterable


def build_gene_to_rules(rules: Iterable) -> dict[str, set[str]]:
    """Return ``{locus_tag: {rule_name, ...}}`` for every gene that
    catalyses at least one Michaelis-Menten rule in ``rules``.

    Genes that produce non-catalytic proteins (ribosomal proteins, tRNA
    synthetases outside the MM stack, etc.) are absent from the returned
    map. That is *correct*: the per-rule detector has no direct signal
    for them and must return ``NONE`` for their KOs.
    """
    mapping: dict[str, set[str]] = {}
    for rule in rules:
        spec = getattr(rule, "compiled_spec", None)
        if not spec:
            continue
        if spec.get("kind") != "mm":
            continue
        for locus in spec.get("enzyme_loci") or ():
            if not locus:
                continue
            mapping.setdefault(locus, set()).add(rule.name)
    return mapping


def summarise(mapping: dict[str, set[str]]) -> dict:
    """Small summary stats for the gene->rules map, useful for logging
    and for writing measured-MCC fact files."""
    if not mapping:
        return {"genes_with_rules": 0, "total_rule_pairs": 0,
                "min_rules": 0, "avg_rules": 0.0, "max_rules": 0}
    counts = [len(v) for v in mapping.values()]
    return {
        "genes_with_rules": len(mapping),
        "total_rule_pairs": sum(counts),
        "min_rules": min(counts),
        "avg_rules": round(sum(counts) / len(counts), 2),
        "max_rules": max(counts),
    }
