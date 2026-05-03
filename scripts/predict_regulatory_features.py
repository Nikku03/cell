"""Phase 6 — predict regulatory features (RBS, -10, -35, TF binding sites) in
all 16 organisms by training motif scanners on Salmonella's RegulonDB-derived
GenBank annotation (the only one of our 16 that has /regulatory and
/protein_bind features).

Pipeline
--------

1. Parse Salmonella LT2 GenBank for labeled regulatory positions:
     - regulatory_class=ribosome_binding_site (3965 sites)
     - regulatory_class=minus_10_signal (256 sites)
     - regulatory_class=minus_35_signal (256 sites)
     - protein_bind features with bound_moiety = TF name (397 sites)
   Also extract the 6-15bp sequence at each labeled position.

2. Build PWMs from the labeled sequences using Bio.motifs:
     - RBS PWM (Shine-Dalgarno consensus learned from labeled positions)
     - -10 PWM
     - -35 PWM
     - Per-TF PWM for major regulators (CRP, FNR, IHF, Lrp, NarL, ArcA, ArgR)

3. Validate on Salmonella's own genes: for each Salmonella CDS, scan its
   upstream 200bp with each PWM, get the best score. Compare to whether
   that gene actually had a labeled regulatory feature. Compute per-feature
   precision / recall to know how trustworthy the predictions are.

4. Apply the PWM scanners to upstream regions of every CDS in all 16
   organisms (uniformly). Generate per-gene features:
     - rbs_score, rbs_present (binary @ threshold)
     - minus10_score, minus10_present
     - minus35_score, minus35_present
     - has_minus35_minus10_pair (both within 17±3bp spacing)
     - tf_score_<TF> for each major TF
     - tf_count (sum of binary TF presences)

5. Save augmented feature matrix.

Output: outputs/multiorg_per_gene_features_with_regulatory.csv
"""
from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO, motifs
from Bio.Seq import Seq

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
GB_DIR = ROOT / 'memory_bank/data/multiorg_essentiality/raw/multi_organism_genbank'
INP_FEATURES = ROOT / 'outputs/multiorg_per_gene_features.csv'
OUT_FEATURES = ROOT / 'outputs/multiorg_per_gene_features_with_regulatory.csv'
OUT_VALIDATION = ROOT / 'outputs/regulatory_motif_validation.json'

GB_PATHS = {
    'ccrescentus': GB_DIR / 'ccrescentus_NA1000_CP001340.1.gb',
    'saureus': GB_DIR / 'saureus_N315_BA000018.3.gb',
    'mtuberculosis': GB_DIR / 'mtuberculosis_H37Rv_AL123456.3.gb',
    'styphimurium': GB_DIR / 'styphimurium_LT2_AE006468.2.gb',
    'abaylyi': GB_DIR / 'abaylyi_ADP1_CR543861.1.gb',
    'mgenitalium': GB_DIR / 'mgenitalium_G37_L43967.2.gb',
    'mpne': ROOT / 'memory_bank/data/multiorg_essentiality/raw/mpneumoniae_M129_NC_000912.gb',
    'syn3a': ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb',
}

# Major TFs to score (top in Salmonella's annotation by site count)
TARGET_TFS = ['CRP', 'FNR', 'IHF', 'Lrp', 'NarL', 'ArcA', 'ArgR',
              'OmpR', 'PurR', 'GlpR', 'TyrR', 'AraC', 'FIS', 'LexA']


# ─────────────── Step 1 + 2: Extract labeled sequences + build PWMs ───────────


def extract_motif_sequences(gb_path: Path) -> dict[str, list[str]]:
    """Pull labeled regulatory sequences from Salmonella GenBank.
    Returns {motif_class: list_of_sequences}."""
    rec = next(SeqIO.parse(str(gb_path), 'genbank'))
    seqs = defaultdict(list)
    for f in rec.features:
        if f.type == 'regulatory':
            cls = f.qualifiers.get('regulatory_class', [''])[0]
            try:
                s = str(f.extract(rec.seq)).upper()
                if 5 <= len(s) <= 30 and set(s) <= set('ACGT'):
                    seqs[cls].append(s)
            except Exception:
                pass
        elif f.type == 'protein_bind':
            tf = f.qualifiers.get('bound_moiety', [''])[0]
            try:
                s = str(f.extract(rec.seq)).upper()
                if 8 <= len(s) <= 40 and set(s) <= set('ACGT'):
                    seqs[f'TF_{tf}'].append(s)
            except Exception:
                pass
    return seqs


def build_pwm_from_seqs(seqs: list[str], target_len: int = None) -> 'motifs.Motif':
    """Build a PWM from labeled sequences. Pad/trim to target_len if specified."""
    if not seqs:
        return None
    if target_len is None:
        # Use the most common length
        target_len = Counter(len(s) for s in seqs).most_common(1)[0][0]
    aligned = [s for s in seqs if len(s) == target_len]
    if len(aligned) < 5:
        # Try with truncation/padding
        aligned = []
        for s in seqs:
            if len(s) >= target_len:
                aligned.append(s[:target_len])
        if len(aligned) < 5:
            return None
    instances = [Seq(s) for s in aligned]
    m = motifs.create(instances)
    m.pseudocounts = 0.5  # for log-odds calculation
    return m


def score_seq_with_pwm(seq: str, pwm: 'motifs.Motif') -> tuple[float, int]:
    """Slide the PWM along seq, return (best_score, best_position).
    Score is log-odds vs uniform background."""
    if pwm is None or len(seq) < pwm.length:
        return -1e9, -1
    pssm = pwm.pssm
    seq_obj = Seq(seq)
    best_score = -1e9; best_pos = -1
    try:
        for pos, score in pssm.search(seq_obj, threshold=-1e9):
            if pos >= 0 and score > best_score:
                best_score = float(score); best_pos = int(pos)
    except Exception:
        # Brute-force if pssm.search fails
        L = pwm.length
        for i in range(len(seq) - L + 1):
            sub = seq[i:i+L]
            if set(sub) - set('ACGT'): continue
            s = float(pssm.calculate(Seq(sub)))
            if s > best_score:
                best_score = s; best_pos = i
    return best_score, best_pos


# ─────────────── Step 3: Validate on Salmonella ground truth ───────────


def get_upstream_seq(rec, feature, length=200) -> str:
    """Return the `length`-bp upstream sequence (5' UTR side) of a CDS."""
    strand = int(feature.location.strand) if feature.location.strand else 1
    if strand == 1:
        start = max(0, int(feature.location.start) - length)
        end = int(feature.location.start)
        return str(rec.seq[start:end]).upper()
    else:
        start = int(feature.location.end)
        end = min(len(rec.seq), int(feature.location.end) + length)
        return str(rec.seq[start:end].reverse_complement()).upper()


def labeled_regulatory_feature_per_cds(salmonella_path: Path):
    """For each Salmonella CDS, return whether each regulatory class has at
    least one labeled feature within 200bp upstream. This becomes the
    validation ground truth."""
    rec = next(SeqIO.parse(str(salmonella_path), 'genbank'))
    cds_index = {}
    for f in rec.features:
        if f.type != 'CDS': continue
        gene = (f.qualifiers.get('gene', [''])[0] or '').strip()
        locus = f.qualifiers.get('locus_tag', [None])[0]
        key = gene if gene else locus
        if not key: continue
        cds_index[key] = (f, get_upstream_seq(rec, f, 200))

    # Walk the regulatory + protein_bind features and assign them to nearest
    # upstream CDS (by gene/locus_tag if present in the feature qualifiers).
    has_label = defaultdict(set)  # cds_key -> set of motif_classes
    for f in rec.features:
        if f.type == 'regulatory':
            cls = f.qualifiers.get('regulatory_class', [''])[0]
        elif f.type == 'protein_bind':
            tf = f.qualifiers.get('bound_moiety', [''])[0]
            cls = f'TF_{tf}'
        else:
            continue
        gene = (f.qualifiers.get('gene', [''])[0] or '').strip()
        locus = f.qualifiers.get('locus_tag', [None])[0]
        key = gene if gene else locus
        if key:
            has_label[key].add(cls)
    return cds_index, has_label, rec


# ─────────────── Step 4: Apply scanners to all 16 organisms ───────────


def per_gene_regulatory_features(rec, organism: str, pwms: dict, thresholds: dict
                                   ) -> dict:
    """Scan upstream of every CDS in `rec`, score with each PWM, return
    {locus_tag (or gene_symbol depending on org policy): feature_dict}.
    Locus-tag policy matches build_multiorg_features.py for join compatibility."""
    out = {}
    for f in rec.features:
        if f.type != 'CDS': continue
        # Per-organism locus_tag policy
        locus = f.qualifiers.get('locus_tag', [None])[0]
        gene_sym = (f.qualifiers.get('gene', [''])[0] or '').strip()
        if organism == 'mpne':
            ot = f.qualifiers.get('old_locus_tag', [None])[0]
            if ot: locus = ot
        elif organism == 'saureus':
            locus = gene_sym if gene_sym else locus
        elif organism == 'styphimurium':
            locus = gene_sym if gene_sym else locus
        if not locus: continue

        ups = get_upstream_seq(rec, f, 200)
        # Score with each PWM
        feats = {}
        for cls, pwm in pwms.items():
            if pwm is None: continue
            score, pos = score_seq_with_pwm(ups, pwm)
            feats[f'{cls}_score'] = float(score) if np.isfinite(score) else 0.0
            t = thresholds.get(cls, 0.0)
            feats[f'{cls}_present'] = float(score >= t)
            feats[f'{cls}_dist'] = float(len(ups) - pos) if pos >= 0 else 200.0
        # Combine: -35/-10 pair check (proper 17±3bp spacing)
        if 'minus_35_signal' in pwms and 'minus_10_signal' in pwms:
            s35, p35 = score_seq_with_pwm(ups, pwms['minus_35_signal'])
            s10, p10 = score_seq_with_pwm(ups, pwms['minus_10_signal'])
            if p35 >= 0 and p10 >= 0:
                gap = (p10 - (p35 + pwms['minus_35_signal'].length))
                feats['has_promoter_pair'] = float(14 <= gap <= 20 and
                                                     s35 >= thresholds.get('minus_35_signal', -1e9) and
                                                     s10 >= thresholds.get('minus_10_signal', -1e9))
            else:
                feats['has_promoter_pair'] = 0.0
        # Aggregate TF count
        tf_count = sum(feats.get(f'TF_{tf}_present', 0.0) for tf in TARGET_TFS)
        feats['predicted_tf_count'] = float(tf_count)
        out[locus] = feats
    return out


# ─────────────── Main ───────────


def main():
    print('[1] extracting labeled regulatory sequences from Salmonella LT2...')
    sal_path = GB_PATHS['styphimurium']
    sal_seqs = extract_motif_sequences(sal_path)
    print('  motif class counts:')
    for cls, sl in sorted(sal_seqs.items(), key=lambda x: -len(x[1])):
        if cls in ('ribosome_binding_site', 'minus_10_signal', 'minus_35_signal') \
                or cls.startswith('TF_'):
            print(f'    {cls:35s} {len(sl):5d} sequences  '
                  f'sample: {sl[0][:20] if sl else ""}')

    print('\n[2] building PWMs...')
    target_classes = ['ribosome_binding_site', 'minus_10_signal', 'minus_35_signal']
    target_classes += [f'TF_{tf}' for tf in TARGET_TFS]
    pwms = {}
    for cls in target_classes:
        seqs = sal_seqs.get(cls, [])
        if len(seqs) < 5:
            print(f'  {cls:35s} skipped — only {len(seqs)} labeled sequences')
            continue
        pwm = build_pwm_from_seqs(seqs)
        pwms[cls] = pwm
        if pwm:
            print(f'  {cls:35s} PWM built (length={pwm.length}, n={len(seqs)})')

    print('\n[3] validating PWMs on Salmonella own labels...')
    # For each Salmonella CDS that HAS a labeled regulatory feature for class C,
    # check whether scanning its upstream 200bp with PWM_C gives a high score.
    cds_index, has_label, sal_rec = labeled_regulatory_feature_per_cds(sal_path)

    # Calibrate thresholds: pick threshold per class such that 80% of labeled
    # CDSes score above threshold. This calibrates "presence" calls.
    thresholds = {}
    validation_stats = {}
    for cls, pwm in pwms.items():
        if pwm is None: continue
        labeled_cds = [k for k, classes in has_label.items() if cls in classes
                       and k in cds_index]
        if not labeled_cds:
            continue
        # Score upstream of each labeled CDS
        scores = []
        for k in labeled_cds:
            _, ups = cds_index[k]
            s, _ = score_seq_with_pwm(ups, pwm)
            scores.append(s)
        scores = sorted([s for s in scores if np.isfinite(s)])
        # Threshold = 20th percentile (so 80% recall)
        if len(scores) >= 5:
            t = scores[len(scores) // 5]
        else:
            t = scores[0] if scores else 0.0
        thresholds[cls] = float(t)

        # Compute precision/recall on Salmonella
        # For all CDSes: predict positive if upstream score >= t
        all_cds_keys = list(cds_index.keys())
        tp = fp = tn = fn = 0
        for k in all_cds_keys:
            _, ups = cds_index[k]
            s, _ = score_seq_with_pwm(ups, pwm)
            pred = s >= t
            true = (cls in has_label.get(k, set()))
            if pred and true: tp += 1
            elif pred and not true: fp += 1
            elif not pred and not true: tn += 1
            else: fn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        validation_stats[cls] = {
            'threshold': t, 'n_labeled': len(labeled_cds),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall,
        }
        print(f'  {cls:35s}  t={t:6.2f}  P={precision:.3f}  R={recall:.3f}  '
              f'(labeled n={len(labeled_cds)})')

    print('\n[4] applying PWM scanners to all 16 organisms...')
    all_organisms_to_score = ['ccrescentus', 'saureus', 'mtuberculosis',
                                'styphimurium', 'abaylyi', 'mgenitalium',
                                'mpne', 'syn3a']
    organism_features = {}
    for org in all_organisms_to_score:
        gb_path = GB_PATHS[org]
        if not gb_path.exists():
            print(f'  {org}: GenBank missing'); continue
        print(f'  scanning {org}...')
        rec = next(SeqIO.parse(str(gb_path), 'genbank'))
        feats = per_gene_regulatory_features(rec, org, pwms, thresholds)
        organism_features[org] = feats
        print(f'    {len(feats)} CDSes scored, sample feature keys: '
              f'{sorted(list(feats.values())[0].keys())[:6] if feats else "none"}')

    print('\n[5] joining onto v25 feature matrix...')
    base = pd.read_csv(INP_FEATURES)
    print(f'  base rows: {len(base)}')

    # Add regulatory features to each row
    new_cols = set()
    for k, fd in next(iter(organism_features.values())).items() if organism_features else []:
        new_cols.add(k); break
    # Get all possible feature keys from any organism's first gene
    new_cols = set()
    for org_feats in organism_features.values():
        for fd in org_feats.values():
            new_cols.update(fd.keys())
            break

    for c in new_cols:
        base[c] = 0.0

    n_matched = 0
    for idx, row in base.iterrows():
        org_feats = organism_features.get(row['organism'])
        if not org_feats: continue
        gene_feats = org_feats.get(row['locus_tag'])
        if not gene_feats: continue
        for k, v in gene_feats.items():
            base.at[idx, k] = v
        n_matched += 1
    print(f'  matched {n_matched}/{len(base)} rows with predicted regulatory features')

    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(OUT_FEATURES, index=False)
    print(f'  wrote {OUT_FEATURES}')

    # Validation summary
    import json
    OUT_VALIDATION.write_text(json.dumps(
        {'pwm_validation_on_salmonella_self': validation_stats,
         'thresholds': thresholds,
         'pwm_lengths': {c: pwms[c].length for c in pwms if pwms[c]},
         'new_feature_keys': sorted(new_cols),
         'n_genes_with_regulatory_features': n_matched},
        indent=2))
    print(f'  wrote {OUT_VALIDATION}')


if __name__ == '__main__':
    main()
