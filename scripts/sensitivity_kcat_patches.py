"""Static + dynamic sensitivity analysis for the patched nutrient-uptake kcats.

Honest answer to the postdoc question: "Are any of your essentiality
predictions sensitive to the patches?"

This script answers it concretely by:
  1. Static analysis: which Syn3A genes' v15 essentiality call DIRECTLY
     depends on a patched-kcat rule (i.e., gene_to_rules contains the
     rule name)?
  2. Dynamic analysis: for each directly-sensitive gene, perturb the
     relevant kcat by ±50% and re-run the v15 KO simulation. Does the
     essentiality call flip?

Static caveat: indirect sensitivity (gene X's KO depletes a metabolite
that flows through a patched rule) is NOT quantified here. The only
clean way to capture indirect effects is a full sweep at perturbed
kcats, which is ~50 min on Colab. This script reports the direct
first-order sensitivity, which is what the placeholder-assignment
question is really asking about.

Outputs
-------
  outputs/kcat_patch_sensitivity.csv
    Per-gene: locus_tag, gene_name, true_essential (Breuer 2019),
    v15_call (cached), v15_confidence, n_total_rules,
    n_patched_rules, patched_fraction, patched_rule_names,
    sensitivity_label (direct/indirect/none).

  outputs/kcat_patch_sensitivity.json
    Summary statistics + per-perturbation MCC if dynamic run completes.

Usage
-----
  python scripts/sensitivity_kcat_patches.py [--dynamic]

  --dynamic : also run perturbed-kcat KO simulations for genes flagged
              direct-sensitive. Adds ~5-30 minutes depending on count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cell_sim'))

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / 'outputs/kcat_patch_sensitivity.csv'
OUT_JSON = ROOT / 'outputs/kcat_patch_sensitivity.json'

# v15 cached predictions on Syn3A's 455 genes
V15_CSV = (ROOT / 'outputs' /
           'predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v15_round2_priors.csv')


# The 15 rule names produced by build_missing_transport_rules() in
# nutrient_uptake.py. Every catalysis rule whose kcat is set by our
# literature-informed patches.
PATCHED_RULE_NAMES = frozenset({
    'catalysis:GLCpts',           # PtsG glucose import (sole catalyst: 0779)
    'catalysis:GLYCt', 'catalysis:GLYCt:rev',
    'catalysis:O2t',   'catalysis:O2t:rev',
    'catalysis:FAt',   'catalysis:FAt:rev',     # sole catalyst: 0616 (FakB)
    'catalysis:CHOLt', 'catalysis:CHOLt:rev',
    'catalysis:TAGt',  'catalysis:TAGt:rev',
    'catalysis:ADEt_syn',
    'catalysis:GUAt_syn',
    'catalysis:URAt_syn',
    'catalysis:CYTDt_syn',
})


def load_v15_cache() -> pd.DataFrame:
    """Load cached v15 per-gene predictions on Syn3A."""
    df = pd.read_csv(V15_CSV)
    df = df.rename(columns={'essential': 'v15_call', 'confidence': 'v15_conf'})
    df['true_essential'] = df.true_class.map(
        {'Essential': 1, 'Quasiessential': 1, 'Nonessential': 0}).astype(int)
    return df[['locus_tag', 'gene_name', 'v15_call', 'v15_conf',
                 'true_essential', 'failure_mode']]


def build_simulator_and_map():
    """Build the simulator's rule list + gene_to_rules. We don't run any
    KOs here — just build the structure once."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    from cell_sim.layer6_essentiality.gene_rule_map import build_gene_to_rules

    cfg = RealSimulatorConfig(
        scale_factor=0.05, seed=42,
        use_rust_backend=False,
        enable_metabolite_sinks=False,
        enable_imb155_patches=True,    # patched: 0034/0228/0732 cleared
        enable_gene_expression=False,
        enable_bioelectric=False,
    )
    sim = RealSimulator(cfg)
    sim._ensure_setup()
    rules = list(sim._rev_rules or []) + list(sim._extra_rules or [])
    gene_to_rules = build_gene_to_rules(rules)
    return sim, rules, gene_to_rules


def static_analysis(gene_to_rules: dict, v15_df: pd.DataFrame) -> pd.DataFrame:
    """For each Syn3A gene, count how many of its catalysis rules are
    patched. Cross-reference with v15 cached calls."""
    rows = []
    for r in v15_df.itertuples(index=False):
        rules = gene_to_rules.get(r.locus_tag, set())
        n_total = len(rules)
        patched = rules & PATCHED_RULE_NAMES
        n_patched = len(patched)
        if n_total == 0:
            label = 'no_catalysis_rules'
        elif n_patched == 0:
            label = 'no_patched_dependency'
        elif n_patched == n_total:
            label = 'fully_patched'
        else:
            label = 'partially_patched'
        rows.append({
            'locus_tag': r.locus_tag,
            'gene_name': r.gene_name,
            'true_essential': r.true_essential,
            'v15_call': r.v15_call,
            'v15_conf': r.v15_conf,
            'failure_mode': r.failure_mode,
            'n_total_rules': n_total,
            'n_patched_rules': n_patched,
            'patched_fraction': (n_patched / n_total) if n_total > 0 else 0.0,
            'patched_rule_names': '|'.join(sorted(patched)) if patched else '',
            'sensitivity_label': label,
        })
    return pd.DataFrame(rows)


def dynamic_perturbation(sim, sensitive_loci: list, perturbations: dict
                            ) -> list[dict]:
    """For each sensitive gene, KO it and run the live simulation under
    each kcat perturbation. Compare the v15 call vs the cached one."""
    from cell_sim.layer6_essentiality.harness import FailureMode
    from cell_sim.layer6_essentiality.complex_assembly_detector import ComplexAssemblyDetector
    from cell_sim.layer6_essentiality.annotation_class_detector import AnnotationClassDetector
    from cell_sim.layer6_essentiality.per_rule_detector import PerRuleDetector
    from cell_sim.layer6_essentiality.composed_detector import ComposedDetector

    T_END_S = 0.5
    DT_S = 0.1

    results = []

    for perturb_name, kcat_overrides in perturbations.items():
        # Build a fresh simulator with the perturbation
        sim_p = _build_perturbed_simulator(kcat_overrides)
        wt = sim_p.run([], t_end_s=T_END_S, sample_dt_s=DT_S)
        gene_to_rules = sim_p.build_gene_to_rules_map()
        pr = PerRuleDetector(wt=wt, gene_to_rules=gene_to_rules,
                                min_wt_events=20)
        detector = ComposedDetector(
            structural=ComplexAssemblyDetector(),
            annotation=AnnotationClassDetector(),
            trajectory=pr,
        )

        for locus in sensitive_loci:
            t0 = time.time()
            try:
                ko = sim_p.run([locus], t_end_s=T_END_S, sample_dt_s=DT_S)
                mode, t_fail, conf, ev = detector.detect_for_gene(locus, ko)
            except Exception as e:
                results.append({
                    'perturbation': perturb_name, 'locus_tag': locus,
                    'failed': True, 'error': str(e)[:200],
                })
                continue
            results.append({
                'perturbation': perturb_name,
                'locus_tag': locus,
                'failed': False,
                'v15_call': int(mode != FailureMode.NONE),
                'v15_conf': float(conf or 0.0),
                'failure_mode': mode.value if hasattr(mode, 'value') else str(mode),
                'wall_time_s': round(time.time() - t0, 2),
            })
    return results


def _build_perturbed_simulator(kcat_overrides: dict):
    """Re-import nutrient_uptake with kcat overrides applied to the dict."""
    # We need to mutate the kcat dict before the simulator builds rules.
    # The cleanest path: import the module, mutate, build sim, then
    # restore. (This script runs on a single thread so no concurrency
    # issue.)
    from cell_sim.layer3_reactions import nutrient_uptake as nu
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )

    # Save originals
    saved_kcat = {k: dict(v) for k, v in nu.KCAT_ESTIMATES.items()}
    saved_synth = {k: dict(v) for k, v in nu.SYNTHETIC_UPTAKE.items()}

    try:
        for short, factor in kcat_overrides.items():
            if short in nu.KCAT_ESTIMATES:
                nu.KCAT_ESTIMATES[short]['kcat_fwd'] *= factor
                if nu.KCAT_ESTIMATES[short].get('kcat_rev', 0) > 0:
                    nu.KCAT_ESTIMATES[short]['kcat_rev'] *= factor
            elif short in nu.SYNTHETIC_UPTAKE:
                nu.SYNTHETIC_UPTAKE[short]['kcat_fwd'] *= factor

        cfg = RealSimulatorConfig(
            scale_factor=0.05, seed=42,
            use_rust_backend=False,
            enable_metabolite_sinks=False,
            enable_imb155_patches=True,
            enable_gene_expression=False,
            enable_bioelectric=False,
        )
        sim = RealSimulator(cfg)
        return sim
    finally:
        # Restore original kcats so subsequent calls aren't tainted
        for k, v in saved_kcat.items():
            nu.KCAT_ESTIMATES[k] = v
        for k, v in saved_synth.items():
            nu.SYNTHETIC_UPTAKE[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dynamic', action='store_true',
                     help='Run perturbed-kcat KO sims on directly-sensitive genes')
    args = ap.parse_args()

    print('[1] loading v15 cached predictions...')
    v15_df = load_v15_cache()
    print(f'  {len(v15_df)} genes from cached v15 sweep')

    print('\n[2] building simulator + gene_to_rules map...')
    t0 = time.time()
    sim, rules, gene_to_rules = build_simulator_and_map()
    print(f'  {len(rules)} total rules, '
          f'{len(gene_to_rules)} genes have at least one catalysis rule '
          f'({time.time()-t0:.1f}s)')

    # Sanity: how many rules in the simulator's set carry a patched name?
    actual_patched_in_rules = {r.name for r in rules if r.name in PATCHED_RULE_NAMES}
    print(f'  patched rules present in built rule list: '
          f'{len(actual_patched_in_rules)}/{len(PATCHED_RULE_NAMES)}')
    # The ones the imb155 patch cleared (placeholder-only) won't appear here:
    cleared = sorted(PATCHED_RULE_NAMES - actual_patched_in_rules)
    if cleared:
        print(f'  patched rules NOT in built list (cleared by imb155): '
              f'{len(cleared)}')
        for r in cleared[:6]:
            print(f'    - {r}')
        if len(cleared) > 6:
            print(f'    ... and {len(cleared)-6} more')

    print('\n[3] static sensitivity analysis...')
    df = static_analysis(gene_to_rules, v15_df)
    df.to_csv(OUT_CSV, index=False)
    print(f'  wrote {OUT_CSV}')

    # Summary
    by_label = df['sensitivity_label'].value_counts()
    print(f'\n  sensitivity label distribution:')
    for label, n in by_label.items():
        print(f'    {label:25s} {n:>4d} genes')

    flagged = df[df.n_patched_rules > 0]
    print(f'\n  genes with patched-rule dependency (direct first-order): '
          f'{len(flagged)}')
    if len(flagged) > 0:
        print(f'\n  detail:')
        print(f'  {"locus_tag":<20s}  {"gene_name":<10s}  {"true":>4s}  '
              f'{"v15":>3s}  {"conf":>5s}  {"n_pat":>5s}/{"tot":<3s}  '
              f'{"frac":>4s}  patched_rules')
        for _, r in flagged.iterrows():
            print(f'  {r.locus_tag:<20s}  {r.gene_name[:10]:<10s}  '
                  f'{int(r.true_essential):>4d}  '
                  f'{int(r.v15_call):>3d}  {r.v15_conf:>5.2f}  '
                  f'{int(r.n_patched_rules):>5d}/{int(r.n_total_rules):<3d}  '
                  f'{r.patched_fraction:>4.2f}  {r.patched_rule_names}')

    # Cross-tab: are flagged genes' v15 calls correct?
    if len(flagged) > 0:
        y = flagged.true_essential.values; p = flagged.v15_call.values
        n_correct = int((y == p).sum())
        print(f'\n  v15 accuracy on flagged genes: {n_correct}/{len(flagged)} '
              f'({100*n_correct/len(flagged):.0f}%)')

    out_summary = {
        'method': 'kcat_patch_sensitivity_static',
        'n_genes_total': len(df),
        'n_patched_rule_names': len(PATCHED_RULE_NAMES),
        'n_patched_rules_in_built_list': len(actual_patched_in_rules),
        'n_patched_rules_cleared_by_imb155': len(cleared),
        'cleared_rules': cleared,
        'sensitivity_label_counts': by_label.to_dict(),
        'flagged_loci': flagged.locus_tag.tolist(),
        'flagged_loci_per_rule': {
            rname: flagged[flagged.patched_rule_names.str.contains(
                rname, regex=False)].locus_tag.tolist()
            for rname in actual_patched_in_rules
        },
    }

    if args.dynamic and len(flagged) > 0:
        print(f'\n[4] dynamic perturbation on {len(flagged)} flagged gene(s)...')
        perturbations = {
            'baseline_1.0x': {},
            'kcat_0.5x': {short: 0.5 for short in [
                'GLCpts', 'GLYCt', 'O2t', 'FAt', 'CHOLt', 'TAGt',
                'ADEt_syn', 'GUAt_syn', 'URAt_syn', 'CYTDt_syn']},
            'kcat_1.5x': {short: 1.5 for short in [
                'GLCpts', 'GLYCt', 'O2t', 'FAt', 'CHOLt', 'TAGt',
                'ADEt_syn', 'GUAt_syn', 'URAt_syn', 'CYTDt_syn']},
        }
        sensitive_loci = flagged.locus_tag.tolist()
        dyn_results = dynamic_perturbation(sim, sensitive_loci, perturbations)

        # Summarize: did any call flip?
        print(f'\n  dynamic results:')
        print(f'  {"locus":<20s}  {"baseline":>10s}  '
              f'{"0.5x":>10s}  {"1.5x":>10s}  flip?')
        by_locus = {}
        for r in dyn_results:
            by_locus.setdefault(r['locus_tag'], {})[r['perturbation']] = r
        for locus in sensitive_loci:
            row = by_locus.get(locus, {})
            base = row.get('baseline_1.0x', {}).get('v15_call', '?')
            half = row.get('kcat_0.5x', {}).get('v15_call', '?')
            onehalf = row.get('kcat_1.5x', {}).get('v15_call', '?')
            calls = {base, half, onehalf} - {'?'}
            flip = 'YES' if len(calls) > 1 else 'no'
            print(f'  {locus:<20s}  {str(base):>10s}  '
                  f'{str(half):>10s}  {str(onehalf):>10s}  {flip}')

        out_summary['dynamic_results'] = dyn_results

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_summary, indent=2, default=str))
    print(f'\nwrote {OUT_JSON}')


if __name__ == '__main__':
    main()
