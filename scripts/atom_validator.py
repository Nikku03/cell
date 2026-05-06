"""atom_validator — chemistry-plausibility check for v15 essentiality predictions.

Use this AFTER you have a v15 (or any predictor's) essentiality call for
a Syn3A gene. The validator does NOT predict essentiality. It answers:

    "Is the chemistry that this gene catalyses plausible at atomic
     resolution?"

How it works
-----------
1. For each input gene, look up its catalysis rules via the v15 simulator
   (`build_gene_to_rules`).
2. For each rule, list the reactant/product metabolites + their elemental
   bond pairs (derived from chemical formulas).
3. Cross-reference against atom_engine's verdict from the full-panel run
   (`outputs/atom_engine_full_panel_results.json`):
     - essential pairs at 8 ps: HH, CH, HN, HO, CC, CO
     - non-essential pairs:     HP, HS, CN, CP, CS, NN, NO, OO, OP, OS, SS
4. Per gene, output:
     - reactions catalysed
     - bond pairs touched
     - per-pair atom_engine status
     - "chemistry plausibility": fraction of touched pairs that
        atom_engine confirms ARE bondable in a reactive soup at all
        (i.e. ANY bondable pair, regardless of essential/non-essential
        — both are physically plausible bond classes)
     - "active-chemistry density": fraction of touched pairs that are
        in the essential-bond set (heavy bond-formation activity)

When this is useful
-------------------
- Sanity-check a v15 prediction: if v15 says "gene X is essential" but
  the validator finds 0 plausible bond pairs in any of X's rules, the
  GPR annotation is probably wrong.
- Spot-check uncharacterised genes: if the v15 model assigns generic
  catalysis rules to an "is_putative" gene, the validator gives an
  honest physical signal about whether that's plausible.
- Filter unreliable cross-org transfers: an ortholog-transferred
  prediction whose underlying chemistry doesn't validate is suspicious.

What this is NOT
----------------
- NOT an essentiality predictor (use v15 + cascade for that;
  atom_engine on Syn3A bridge produced MCC = 0.190 — see
  `outputs/atom_engine_on_syn3a_results.json`)
- NOT a chemistry simulator (the verdicts come from the cached
  `outputs/atom_engine_full_panel_results.json`; the actual
  atomic simulation runs once via `run_atom_essentiality.py`)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

ATOM_PANEL_JSON = ROOT / 'outputs/atom_engine_full_panel_results.json'


def _load_atom_verdict() -> tuple[set[str], set[str]]:
    """Load atom_engine essential / non-essential bond-pair sets from
    the cached full-panel run."""
    if not ATOM_PANEL_JSON.exists():
        raise FileNotFoundError(
            f'Run scripts/run_atom_essentiality.py first to produce '
            f'{ATOM_PANEL_JSON.relative_to(ROOT)}')
    j = json.loads(ATOM_PANEL_JSON.read_text())
    ess = set(p.replace('pair_', '') for p in j['essential_pairs'])
    non = set(p.replace('pair_', '') for p in j['nonessential_pairs'])
    return ess, non


def parse_formula(formula: str) -> set[str]:
    """Return the set of elements present in a chemical formula."""
    if not formula: return set()
    elts = set()
    for elt, _ in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if elt: elts.add(elt)
    return elts & {'H', 'C', 'N', 'O', 'P', 'S'}


def bond_pairs(formula: str) -> set[str]:
    """Bond-pair labels (alphabetical) between elements present in a
    metabolite formula."""
    elts = sorted(parse_formula(formula))
    return {a + b for i, a in enumerate(elts) for b in elts[i:]}


def _build_v15_lookups():
    """Set up the v15 simulator and return (gene_to_rules, rule_metabs,
    metabolite_formulas)."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig)
    from cell_sim.layer6_essentiality.gene_rule_map import build_gene_to_rules

    cfg = RealSimulatorConfig(
        scale_factor=0.05, seed=42, use_rust_backend=False,
        enable_metabolite_sinks=False, enable_imb155_patches=True,
        enable_gene_expression=False, enable_bioelectric=False,
    )
    sim = RealSimulator(cfg)
    sim._ensure_setup()
    rules = list(sim._rev_rules or []) + list(sim._extra_rules or [])
    gene_to_rules = build_gene_to_rules(rules)

    rule_metabs: dict[str, dict] = {}
    for r in rules:
        spec = getattr(r, 'compiled_spec', None)
        if not spec or spec.get('kind') != 'mm':
            continue
        rule_metabs[r.name] = {
            'reactants': [s for s, _ in (spec.get('reactants') or ())],
            'products':  [s for s, _ in (spec.get('products')  or ())],
        }

    # Build metabolite formula table (libsbml fbc plugin)
    formulas: dict[str, str] = {}
    sbml_path = (ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/'
                       'input_data/Syn3A_updated.xml')
    if sbml_path.exists():
        try:
            from libsbml import SBMLReader
            doc = SBMLReader().readSBMLFromFile(str(sbml_path))
            model = doc.getModel()
            for i in range(model.getNumSpecies()):
                sp = model.getSpecies(i)
                fbc = sp.getPlugin('fbc')
                if fbc is not None:
                    f = fbc.getChemicalFormula()
                    if f: formulas[sp.getId()] = f
        except Exception:
            pass
    return gene_to_rules, rule_metabs, formulas


def validate_gene(locus_tag: str, gene_to_rules, rule_metabs,
                    formulas, ess_pairs, non_pairs) -> dict:
    rules = sorted(gene_to_rules.get(locus_tag, set()))
    rules_report = []
    all_pairs_touched = set()
    has_data = False
    for rn in rules:
        info = rule_metabs.get(rn, {})
        mets = info.get('reactants', []) + info.get('products', [])
        rule_pairs = set()
        rule_metabs_with_formula = []
        for m in mets:
            f = formulas.get(m)
            if f:
                rule_pairs |= bond_pairs(f)
                rule_metabs_with_formula.append({'metabolite': m, 'formula': f})
        if rule_pairs:
            has_data = True
            all_pairs_touched |= rule_pairs
            essential_hit = sorted(rule_pairs & ess_pairs)
            nonessential_hit = sorted(rule_pairs & non_pairs)
            rules_report.append({
                'rule_name': rn,
                'reactants': info.get('reactants', []),
                'products': info.get('products', []),
                'metabolites_with_formula': rule_metabs_with_formula,
                'bond_pairs_touched': sorted(rule_pairs),
                'atom_essential_pairs_present': essential_hit,
                'atom_nonessential_pairs_present': nonessential_hit,
                'has_essential_chemistry': bool(essential_hit),
            })
        else:
            rules_report.append({
                'rule_name': rn,
                'reactants': info.get('reactants', []),
                'products': info.get('products', []),
                'metabolites_with_formula': [],
                'bond_pairs_touched': [],
                'atom_essential_pairs_present': [],
                'atom_nonessential_pairs_present': [],
                'has_essential_chemistry': False,
                'note': 'no metabolite formula data for this rule',
            })

    if not all_pairs_touched:
        return {
            'locus_tag': locus_tag,
            'has_catalysis_rules': bool(rules),
            'n_rules': len(rules),
            'n_rules_with_chemistry_data': 0,
            'rules': rules_report,
            'plausibility_summary': (
                'NO_SIGNAL: no metabolite formula data for any of this '
                'gene\'s rules, OR gene has no catalysis rules. '
                'Validator cannot say anything about chemistry plausibility.'),
            'chemistry_plausibility': None,
            'active_chemistry_density': None,
        }
    n_rules_with_data = sum(1 for r in rules_report if r['bond_pairs_touched'])
    plaus = len(all_pairs_touched & (ess_pairs | non_pairs)) / max(
                  1, len(all_pairs_touched))
    active = (len(all_pairs_touched & ess_pairs) /
                  max(1, len(all_pairs_touched)))
    return {
        'locus_tag': locus_tag,
        'has_catalysis_rules': True,
        'n_rules': len(rules),
        'n_rules_with_chemistry_data': n_rules_with_data,
        'all_bond_pairs_touched': sorted(all_pairs_touched),
        'atom_essential_pairs_in_chemistry': sorted(
            all_pairs_touched & ess_pairs),
        'atom_nonessential_pairs_in_chemistry': sorted(
            all_pairs_touched & non_pairs),
        'rules': rules_report,
        'chemistry_plausibility': float(plaus),
        'active_chemistry_density': float(active),
        'plausibility_summary': (
            f'{n_rules_with_data}/{len(rules)} rules have chemistry data; '
            f'{int(plaus*100)}% of {len(all_pairs_touched)} touched bond '
            f'pairs are atom-engine-classified; '
            f'{int(active*100)}% are in the active-chemistry set.'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gene', action='append', default=[],
                     help='locus_tag (e.g. JCVISYN3A_0445); pass multiple '
                          'times to validate several genes')
    ap.add_argument('--predictions', type=Path,
                     help='optional CSV with a locus_tag column; validates '
                          'every locus listed there (or filter by --top)')
    ap.add_argument('--top', type=int, default=0,
                     help='if --predictions, validate only the top N rows')
    ap.add_argument('--out', type=Path,
                     default=ROOT / 'outputs/atom_validator_report.json',
                     help='where to write the JSON report')
    args = ap.parse_args()

    # Resolve gene list
    genes: list[str] = list(args.gene)
    if args.predictions:
        df = pd.read_csv(args.predictions)
        col = 'locus_tag' if 'locus_tag' in df.columns else df.columns[0]
        loci = df[col].astype(str).tolist()
        if args.top: loci = loci[:args.top]
        genes.extend(loci)
    genes = list(dict.fromkeys(genes))   # dedupe, preserve order
    if not genes:
        ap.error('Pass --gene <locus> or --predictions <csv>')

    print(f'[1] loading atom_engine verdict + v15 lookups...')
    ess, non = _load_atom_verdict()
    print(f'  atom essential pairs:     {sorted(ess)}')
    print(f'  atom non-essential pairs: {sorted(non)}')

    gene_to_rules, rule_metabs, formulas = _build_v15_lookups()
    print(f'  v15 gene_to_rules: {len(gene_to_rules)} genes; '
          f'{len(rule_metabs)} rules with metabolites; '
          f'{len(formulas)} metabolites with formulas')

    print(f'\n[2] validating {len(genes)} genes...')
    reports = []
    for g in genes:
        r = validate_gene(g, gene_to_rules, rule_metabs, formulas,
                            ess, non)
        reports.append(r)
        plaus = r.get('chemistry_plausibility')
        active = r.get('active_chemistry_density')
        plaus_s = f'{plaus:.2f}' if plaus is not None else '  -'
        active_s = f'{active:.2f}' if active is not None else '  -'
        print(f'  {g:<22s}  rules={r["n_rules"]:>2d}  '
              f'with_data={r["n_rules_with_chemistry_data"]:>2d}  '
              f'plaus={plaus_s}  active={active_s}  '
              f'pairs={r.get("all_bond_pairs_touched", "[]")}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        'method': 'atom_engine_chemistry_validator',
        'description': 'NOT an essentiality predictor; validates that the '
                        'chemistry catalysed by a gene is recognized by '
                        'atom_engine\'s bond-pair physics.',
        'atom_essential_pairs': sorted(ess),
        'atom_nonessential_pairs': sorted(non),
        'n_genes_validated': len(reports),
        'reports': reports,
    }, indent=2, default=str))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
