"""Bridge atom_engine's chemistry-pair essentiality to Syn3A's 455 genes.

Direct atomic simulation of Syn3A is infeasible by ~10 orders of magnitude
(femtosecond steps × seconds-of-biology). This script does the only
tractable mapping:

  1. Build the v15 gene -> reaction map (which catalysis rules each gene
     enables).
  2. For each catalysis rule, look up its reactants + products (chemical
     formulas) and enumerate the elemental bond pairs involved
     (e.g. acetyl-CoA + oxaloacetate -> citrate + CoA touches C-C, C-O,
     C-H, C-N pairs).
  3. atom_engine's PerRule detector verdict at 8 ps gives us:
       essential bond pairs = {pair_HH, pair_CH, pair_HN, pair_HO,
                                  pair_CC, pair_CO}
     non-essential bond pairs = {pair_HP, pair_HS, pair_CN, pair_CP,
                                   pair_CS, pair_NN, pair_NO, pair_OO,
                                   pair_OP, pair_OS, pair_SS}
  4. Vote: a Syn3A gene is "atom-engine essential" iff every one of its
     catalysis rules involves at least one essential bond pair.
     Counter-vote: gene is "atom-engine non-essential" iff none of its
     rules involve essential bond pairs.

  5. Compare to Breuer 2019 ground truth, compute MCC.

This will likely produce a weak MCC because atom_engine's essential set
covers most organic chemistry — most genes will be voted essential by
this rule. That is the honest finding and is the point of running it.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# atom_engine verdict at 8 ps (from outputs/atom_engine_full_panel_results.json)
AE_ESSENTIAL_PAIRS = frozenset({'HH', 'CH', 'HN', 'HO', 'CC', 'CO'})
AE_NONESSENTIAL_PAIRS = frozenset({'HP', 'HS', 'CN', 'CP', 'CS',
                                       'NN', 'NO', 'OO', 'OP', 'OS', 'SS'})


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula like 'C6H12O6' or 'H2O' into element counts."""
    if not formula:
        return {}
    out = {}
    for elt, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if not elt: continue
        out[elt] = out.get(elt, 0) + (int(count) if count else 1)
    return out


def bond_pairs_in_metabolite(formula: str) -> set[str]:
    """Enumerate bond-pair labels (sorted, no Hill-order prefix) present in
    a metabolite's elemental composition. We don't know the molecule's
    exact bond graph from formula alone — we approximate by saying every
    pair of elements that BOTH appear in the formula is a candidate."""
    elts = set(parse_formula(formula).keys())
    # atom_engine only knows H/C/N/O/P/S
    elts = elts & {'H', 'C', 'N', 'O', 'P', 'S'}
    pairs = set()
    elts_l = sorted(elts)
    for i, a in enumerate(elts_l):
        for b in elts_l[i:]:   # include self-pairs (CC, NN, OO, ...)
            pairs.add(a + b if a == b else (a + b))
    # Normalize to atom_engine's pair labels (alphabetical)
    return {''.join(sorted(p)) for p in pairs}


def main():
    print('[1] loading v15 simulator + building gene_to_rules + rule_metabolites...')
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig)
    from cell_sim.layer6_essentiality.gene_rule_map import (
        build_gene_to_rules, build_rule_products, build_metabolite_producers)
    from cell_sim.layer6_essentiality.labels import (
        binary_labels, load_breuer2019_labels)

    cfg = RealSimulatorConfig(
        scale_factor=0.05, seed=42, use_rust_backend=False,
        enable_metabolite_sinks=False, enable_imb155_patches=True,
        enable_gene_expression=False, enable_bioelectric=False,
    )
    sim = RealSimulator(cfg)
    sim._ensure_setup()
    rules = list(sim._rev_rules or []) + list(sim._extra_rules or [])
    print(f'  built {len(rules)} catalysis rules')

    gene_to_rules = build_gene_to_rules(rules)
    rule_products = build_rule_products(rules)
    print(f'  {len(gene_to_rules)} genes have catalysis rules')
    print(f'  {len(rule_products)} rules have products')

    # Build rule -> reactants, rule -> products by walking the simulator's
    # compiled rule specs directly so we capture both sides.
    rule_metabs: dict[str, list[str]] = {}
    for r in rules:
        spec = getattr(r, 'compiled_spec', None)
        if not spec or spec.get('kind') != 'mm':
            continue
        mets = []
        for s, _ in (spec.get('reactants') or ()): mets.append(s)
        for s, _ in (spec.get('products')  or ()): mets.append(s)
        rule_metabs[r.name] = mets
    print(f'  {len(rule_metabs)} rules have reactants+products')

    # Build metabolite_id -> formula via the simulator's species table
    print('\n[2] building metabolite -> chemical formula map...')
    formulas = {}
    species_meta = getattr(sim, '_species_meta', None) or {}
    if species_meta:
        for mid, meta in species_meta.items():
            f = meta.get('formula') if isinstance(meta, dict) else getattr(meta, 'formula', None)
            if f: formulas[mid] = f
    # Fall back to parsing from the SBML XML directly
    if len(formulas) < 50:
        sbml_path = (ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/'
                          'input_data/Syn3A_updated.xml')
        if sbml_path.exists():
            try:
                from libsbml import SBMLReader  # noqa
                reader = SBMLReader()
                doc = reader.readSBMLFromFile(str(sbml_path))
                model = doc.getModel()
                for i in range(model.getNumSpecies()):
                    sp = model.getSpecies(i)
                    sid = sp.getId()
                    # libSBML stores formula as a SBO/notes annotation; try plugin
                    plug = sp.getPlugin('fbc')
                    if plug is not None:
                        f = plug.getChemicalFormula()
                        if f: formulas[sid] = f
            except Exception as e:
                print(f'  libsbml fallback failed: {e}')
    print(f'  formulas captured: {len(formulas)}')

    # 3. Apply atom_engine verdict per gene
    print('\n[3] applying atom_engine verdict per gene...')
    gene_predictions = {}
    gene_evidence = {}
    for gene, rule_names in gene_to_rules.items():
        all_pairs = set()
        n_rules_with_data = 0
        for rn in rule_names:
            mets = rule_metabs.get(rn, [])
            for m in mets:
                f = formulas.get(m)
                if f:
                    pairs = bond_pairs_in_metabolite(f)
                    all_pairs.update(pairs)
            if any(formulas.get(m) for m in mets):
                n_rules_with_data += 1
        # Vote: gene essential if ANY of its bond pairs is in atom_engine
        # essential set (lenient OR rule)
        ess_hit = all_pairs & AE_ESSENTIAL_PAIRS
        if not all_pairs:
            gene_predictions[gene] = None   # no data
            gene_evidence[gene] = 'no_metabolite_formulas'
        elif ess_hit:
            gene_predictions[gene] = 1
            gene_evidence[gene] = f'essential_pairs:{sorted(ess_hit)}'
        else:
            gene_predictions[gene] = 0
            gene_evidence[gene] = f'all_nonessential_pairs:{sorted(all_pairs)}'

    # 4. Strict variant: gene essential iff EVERY rule involves an
    # essential pair (more selective)
    gene_strict = {}
    for gene, rule_names in gene_to_rules.items():
        rule_hits = []
        for rn in rule_names:
            mets = rule_metabs.get(rn, [])
            rule_pairs = set()
            for m in mets:
                f = formulas.get(m)
                if f: rule_pairs |= bond_pairs_in_metabolite(f)
            if rule_pairs:
                rule_hits.append(bool(rule_pairs & AE_ESSENTIAL_PAIRS))
        if not rule_hits:
            gene_strict[gene] = None
        elif all(rule_hits):
            gene_strict[gene] = 1
        else:
            gene_strict[gene] = 0

    # 5. Compare to Breuer ground truth
    print('\n[4] comparing to Breuer 2019 ground truth on Syn3A panel...')
    labels = load_breuer2019_labels()
    binary = binary_labels(labels)

    rows = []
    for locus in sorted(binary.keys()):
        truth = int(binary[locus])
        pred_lenient = gene_predictions.get(locus)
        pred_strict = gene_strict.get(locus)
        rows.append({
            'locus_tag': locus,
            'true_essential': truth,
            'has_catalysis_rules': locus in gene_to_rules,
            'n_rules': len(gene_to_rules.get(locus, set())),
            'ae_pred_lenient': pred_lenient,
            'ae_pred_strict': pred_strict,
            'evidence': gene_evidence.get(locus, 'no_rules'),
        })
    df = pd.DataFrame(rows)

    # MCC computation — for genes WITHOUT rules, default to non-essential (0)
    df['ae_pred_lenient_filled'] = df.ae_pred_lenient.fillna(0).astype(int)
    df['ae_pred_strict_filled']  = df.ae_pred_strict.fillna(0).astype(int)
    y = df.true_essential.values

    print(f'\n  panel: {len(df)} genes  ({df.true_essential.sum()} essential, '
          f'{(df.true_essential==0).sum()} non-essential)')
    print(f'  with catalysis rules:        {df.has_catalysis_rules.sum()}')
    print(f'  with metabolite formula data: {df.ae_pred_lenient.notna().sum()}')

    for col, name in [('ae_pred_lenient_filled', 'LENIENT (any-pair-hit, full panel)'),
                       ('ae_pred_strict_filled',  'STRICT (all-rules-hit,  full panel)')]:
        pred = df[col].values
        if len(set(pred)) < 2:
            print(f'  {name:<50s}  degenerate: predicts only one class')
            continue
        mcc = matthews_corrcoef(y, pred)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        prec = tp/(tp+fp) if (tp+fp) else float('nan')
        rec = tp/(tp+fn) if (tp+fn) else float('nan')
        print(f'  {name:<50s}  MCC={mcc:+.4f}  TP={tp} FP={fp} TN={tn} FN={fn}  '
              f'prec={prec:.3f}  rec={rec:.3f}')

    # Subset MCC: only on genes WHERE atom_engine has signal
    mask = df.ae_pred_lenient.notna()
    if mask.sum() >= 5:
        for col, name in [('ae_pred_lenient', 'LENIENT (subset, has atom signal)'),
                           ('ae_pred_strict',  'STRICT (subset, has atom signal)')]:
            sub = df[mask].copy()
            sub_y = sub.true_essential.values
            sub_pred = sub[col].astype(int).values
            if len(set(sub_pred)) < 2:
                print(f'  {name:<50s}  n={len(sub):>3d}  degenerate')
                continue
            mcc = matthews_corrcoef(sub_y, sub_pred)
            tn, fp, fn, tp = confusion_matrix(sub_y, sub_pred, labels=[0,1]).ravel()
            print(f'  {name:<50s}  n={len(sub):>3d}  MCC={mcc:+.4f}  '
                  f'TP={tp} FP={fp} TN={tn} FN={fn}')

    # Distribution of votes
    print(f'\n  vote distribution (lenient):')
    print(f'    predict essential:    {(df.ae_pred_lenient==1).sum()}')
    print(f'    predict non-essential: {(df.ae_pred_lenient==0).sum()}')
    print(f'    no signal (defaulted to 0): {df.ae_pred_lenient.isna().sum()}')

    out_csv = ROOT / 'outputs/atom_engine_on_syn3a_per_gene.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')

    out_json = ROOT / 'outputs/atom_engine_on_syn3a_results.json'
    summary = {
        'method': 'atom_engine_chemistry_pair_verdict_via_gene_to_rules_bridge',
        'atom_engine_essential_pairs': sorted(AE_ESSENTIAL_PAIRS),
        'atom_engine_nonessential_pairs': sorted(AE_NONESSENTIAL_PAIRS),
        'panel_size': int(len(df)),
        'n_essential': int(df.true_essential.sum()),
        'n_with_catalysis_rules': int(df.has_catalysis_rules.sum()),
        'n_with_atom_signal': int(df.ae_pred_lenient.notna().sum()),
        'lenient_full_panel_mcc': float(matthews_corrcoef(
            y, df.ae_pred_lenient_filled.values))
            if len(set(df.ae_pred_lenient_filled.values)) > 1 else None,
        'strict_full_panel_mcc': float(matthews_corrcoef(
            y, df.ae_pred_strict_filled.values))
            if len(set(df.ae_pred_strict_filled.values)) > 1 else None,
        'honest_caveat': (
            'This is the most generous bridge possible: it maps Syn3A genes '
            'to atom_engine bond-pair verdicts via the metabolite chemical '
            'formulas in the v15 simulator. The expected weakness is that '
            'atom_engine\'s essential pairs (HH/CH/HN/HO/CC/CO) cover the '
            'elemental composition of nearly every metabolite in biological '
            'chemistry, so the bridge produces near-trivial predictions.'
        ),
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f'wrote {out_json}')


if __name__ == '__main__':
    main()
