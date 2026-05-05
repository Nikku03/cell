"""Per-component strength/weakness profile across all 8 organisms.

Buckets each predictor's performance by:
  - functional keyword (kw_replication, kw_atp, ...)
  - phylogenetic conservation (n_orthologs)
  - PPI degree
  - gene length
  - hypothetical / characterized

Components evaluated:
  - ortholog_prior  (leave-one-out, all 7 non-syn3a orgs + syn3a)
  - v15 simulator   (Syn3A in-domain only; transfers to other orgs are too
                       sparse to bucket meaningfully — flagged as such)
  - cross-org LNN   (M. tb fold; the one fold we have per-gene probs for)
  - PPI features    (used as standalone predictor: ppi_max_score >= t)

Output:
  outputs/component_strengths_per_bucket.csv
      bucket_type, bucket_value, component, organism, n, mcc, precision, recall
  outputs/component_strengths_summary.md
      human-readable strengths/weaknesses with specific examples
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / 'outputs/component_strengths_per_bucket.csv'
OUT_MD = ROOT / 'outputs/component_strengths_summary.md'

# ───────────── data sources ─────────────
FEATURES_CSV = ROOT / 'outputs/multiorg_per_gene_features.csv'
PPI_CSV = ROOT / 'outputs/ppi_features_per_gene.csv'
ORTHOLOG_FEATURES = ROOT / 'outputs/ortholog_prior_features_per_gene.csv'
V15_SYN3A_CSV = (ROOT / 'outputs' /
                  'predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_'
                  'composed_all455_v15_round2_priors.csv')
MTB_CASCADE_CSV = ROOT / 'outputs/cascade_three_way_mtuberculosis_per_gene.csv'

ORGS = ['syn3a', 'mgenitalium', 'mpne', 'saureus', 'styphimurium',
        'ccrescentus', 'abaylyi', 'mtuberculosis']

KW_FEATURES = ['kw_replication', 'kw_transcription', 'kw_translation',
                'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
                'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
                'is_hypothetical', 'is_uncharacterized', 'is_putative']


# ───────────── helpers ─────────────
def safe_mcc(y, pred):
    if len(set(y)) < 2 or len(set(pred)) < 2:
        return float('nan')
    return matthews_corrcoef(y, pred)


def metrics(y, pred):
    if len(y) == 0:
        return {'n': 0, 'mcc': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    if len(set(y)) < 2 or len(set(pred)) < 2:
        # degenerate bucket: report support + accuracy only
        n_pos = int(np.sum(y))
        return {'n': len(y), 'mcc': float('nan'),
                'precision': float('nan'), 'recall': float('nan'),
                'tp': int(np.sum((y==1) & (pred==1))),
                'fp': int(np.sum((y==0) & (pred==1))),
                'tn': int(np.sum((y==0) & (pred==0))),
                'fn': int(np.sum((y==1) & (pred==0))),
                'prevalence': n_pos / len(y) if len(y) else 0.0}
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    return {'n': len(y), 'mcc': float(matthews_corrcoef(y, pred)),
            'precision': float(prec), 'recall': float(rec),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'prevalence': int(tp + fn) / len(y)}


def length_bucket(log_bp):
    if log_bp < np.log(300): return 'short(<300bp)'
    if log_bp < np.log(1000): return 'medium(300-1000bp)'
    if log_bp < np.log(3000): return 'long(1k-3k)'
    return 'very_long(>3kbp)'


def n_orth_bucket(n):
    if n == 0: return '0'
    if n <= 2: return '1-2'
    if n <= 5: return '3-5'
    return '6+'


# ───────────── main ─────────────
def main():
    print('[1] loading features + labels...')
    fdf = pd.read_csv(FEATURES_CSV)
    fdf = fdf.dropna(subset=['essential']).copy()
    fdf['essential'] = fdf['essential'].astype(int)
    print(f'  {len(fdf)} labeled genes across {fdf.organism.nunique()} orgs')

    print('\n[2] joining PPI + ortholog feature CSVs...')
    ppi = pd.read_csv(PPI_CSV)
    orth = pd.read_csv(ORTHOLOG_FEATURES)
    fdf = (fdf.merge(ppi, on=['organism', 'locus_tag'], how='left')
                   .merge(orth, on=['organism', 'locus_tag'], how='left'))
    fdf['n_orthologs'] = fdf.n_orthologs.fillna(0).astype(int)
    fdf['ortholog_prior'] = fdf.ortholog_prior.fillna(0.5)
    fdf['ppi_max_score'] = fdf.ppi_max_score.fillna(0.0)
    fdf['ppi_degree_high'] = fdf.ppi_degree_high.fillna(0).astype(int)
    print(f'  joined: {len(fdf)} rows')

    # ---- gather component predictions by (organism, locus_tag) ----
    # Component A: ortholog_prior (already in fdf as a probability)
    # Component B: PPI as a predictor: ppi_max_score >= 0.7 (matches the
    #              high-degree cutoff used in the LNN feature pipeline)
    # Component C: v15 simulator (Syn3A only, in-domain predictions)
    # Component D: cross-org LNN (only M. tb has per-gene probs cached)

    # Add v15 calls for Syn3A
    print('\n[3] adding component predictions...')
    v15_syn = pd.read_csv(V15_SYN3A_CSV)
    v15_syn['true_essential'] = v15_syn.true_class.map(
        {'Essential': 1, 'Quasiessential': 1, 'Nonessential': 0}).astype(int)
    v15_map = {(s, lt): (int(c), float(cf)) for s, lt, c, cf in zip(
        ['syn3a']*len(v15_syn), v15_syn.locus_tag, v15_syn.essential,
        v15_syn.confidence)}

    # Add LNN probs for M. tb
    mtb_cascade = pd.read_csv(MTB_CASCADE_CSV)
    lnn_mtb_map = {(o, lt): float(p) for o, lt, p in zip(
        ['mtuberculosis']*len(mtb_cascade), mtb_cascade.locus_tag,
        mtb_cascade.lnn_prob)}

    fdf['v15_call'] = fdf.apply(
        lambda r: v15_map.get((r.organism, r.locus_tag), (np.nan, np.nan))[0],
        axis=1)
    fdf['lnn_prob'] = fdf.apply(
        lambda r: lnn_mtb_map.get((r.organism, r.locus_tag), np.nan), axis=1)

    fdf['orth_pred'] = (fdf.ortholog_prior >= 0.5).astype(int)
    # Ortholog prior should not predict for genes with 0 orthologs (no signal)
    fdf['orth_has_signal'] = fdf.n_orthologs > 0
    # PPI as a standalone predictor: threshold ppi_max_score >= 0.5
    # (was 0.7, which collapsed into the with-signal subset → all-1 preds).
    # Evaluate on the FULL panel — genes with no PPI partner default to 0,
    # which is the correct prediction for the prevalent non-essential class.
    fdf['ppi_pred'] = (fdf.ppi_max_score >= 0.5).astype(int)
    fdf['ppi_has_signal'] = pd.Series(True, index=fdf.index)  # full panel
    fdf['v15_pred'] = fdf.v15_call
    fdf['v15_has_signal'] = fdf.v15_call.notna()
    fdf['lnn_pred'] = (fdf.lnn_prob >= 0.5).astype('Int64')
    fdf['lnn_has_signal'] = fdf.lnn_prob.notna()

    fdf['length_bucket'] = fdf.log_length_bp.apply(length_bucket)
    fdf['n_orth_bucket'] = fdf.n_orthologs.apply(n_orth_bucket)

    # ───────────── component-level summary ─────────────
    print('\n[4] global component MCC per organism (with-signal subset only)...')
    rows = []
    for org in ORGS:
        odf = fdf[fdf.organism == org]
        if len(odf) == 0: continue
        y = odf.essential.values
        # Ortholog
        m = odf[odf.orth_has_signal]
        if len(m) >= 5:
            mc = metrics(m.essential.values, m.orth_pred.values)
            mc.update({'organism': org, 'component': 'ortholog_prior',
                       'scope': 'with_signal'})
            rows.append(mc)
        # PPI
        m = odf[odf.ppi_has_signal]
        if len(m) >= 5:
            mc = metrics(m.essential.values, m.ppi_pred.values)
            mc.update({'organism': org, 'component': 'PPI',
                       'scope': 'with_signal'})
            rows.append(mc)
        # v15
        m = odf[odf.v15_has_signal]
        if len(m) >= 5:
            mc = metrics(m.essential.values,
                         m.v15_pred.astype(int).values)
            mc.update({'organism': org, 'component': 'v15_simulator',
                       'scope': 'with_signal'})
            rows.append(mc)
        # LNN (only M. tb)
        m = odf[odf.lnn_has_signal]
        if len(m) >= 5:
            mc = metrics(m.essential.values,
                         m.lnn_pred.astype(int).values)
            mc.update({'organism': org, 'component': 'cross_org_LNN',
                       'scope': 'with_signal'})
            rows.append(mc)
    glob_df = pd.DataFrame(rows)
    print(glob_df[['organism','component','n','mcc','precision','recall']]
          .to_string(index=False))

    # ───────────── per-bucket analysis ─────────────
    print('\n[5] per-bucket MCC for each component...')
    bucket_rows = []

    def bucket_eval(df_sub, bucket_type, bucket_value, organism):
        out_rows = []
        # Ortholog
        m = df_sub[df_sub.orth_has_signal]
        if len(m) >= 10:
            mc = metrics(m.essential.values, m.orth_pred.values)
            mc.update({'organism': organism, 'component': 'ortholog_prior',
                       'bucket_type': bucket_type, 'bucket_value': bucket_value})
            out_rows.append(mc)
        # PPI
        m = df_sub[df_sub.ppi_has_signal]
        if len(m) >= 10:
            mc = metrics(m.essential.values, m.ppi_pred.values)
            mc.update({'organism': organism, 'component': 'PPI',
                       'bucket_type': bucket_type, 'bucket_value': bucket_value})
            out_rows.append(mc)
        # v15 (mostly Syn3A only)
        m = df_sub[df_sub.v15_has_signal]
        if len(m) >= 10:
            mc = metrics(m.essential.values, m.v15_pred.astype(int).values)
            mc.update({'organism': organism, 'component': 'v15_simulator',
                       'bucket_type': bucket_type, 'bucket_value': bucket_value})
            out_rows.append(mc)
        # LNN (M. tb only)
        m = df_sub[df_sub.lnn_has_signal]
        if len(m) >= 10:
            mc = metrics(m.essential.values, m.lnn_pred.astype(int).values)
            mc.update({'organism': organism, 'component': 'cross_org_LNN',
                       'bucket_type': bucket_type, 'bucket_value': bucket_value})
            out_rows.append(mc)
        return out_rows

    for org in ORGS:
        odf = fdf[fdf.organism == org]
        if len(odf) < 30: continue
        # By keyword
        for kw in KW_FEATURES:
            if kw not in odf.columns: continue
            for val in [0, 1]:
                m = odf[odf[kw] == val]
                bucket_rows.extend(bucket_eval(
                    m, kw, str(val), org))
        # By n_orthologs
        for nb in ['0', '1-2', '3-5', '6+']:
            m = odf[odf.n_orth_bucket == nb]
            bucket_rows.extend(bucket_eval(m, 'n_orth_bucket', nb, org))
        # By length
        for lb in ['short(<300bp)', 'medium(300-1000bp)', 'long(1k-3k)',
                    'very_long(>3kbp)']:
            m = odf[odf.length_bucket == lb]
            bucket_rows.extend(bucket_eval(m, 'length_bucket', lb, org))
        # By PPI degree (binary)
        for v in [True, False]:
            m = odf[odf.ppi_has_signal == v]
            bucket_rows.extend(bucket_eval(
                m, 'has_ppi_partner', str(v), org))

    bdf = pd.DataFrame(bucket_rows)
    bdf.to_csv(OUT_CSV, index=False)
    print(f'  wrote {len(bdf)} rows to {OUT_CSV}')

    # ───────────── synthesize markdown report ─────────────
    print('\n[6] writing markdown summary...')
    lines = ['# Per-component strengths and weaknesses\n']
    lines.append('Bucket-level MCC for each predictor across 8 organisms. '
                  '"With-signal" subset = genes for which the component has '
                  'non-trivial input (e.g. ortholog prior excludes genes '
                  'with `n_orthologs == 0`). MCC < 0 = worse than random.\n\n')

    # Global with-signal table
    lines.append('## Global per-organism MCC (with-signal subset)\n\n')
    lines.append('| organism | component | n | MCC | precision | recall |')
    lines.append('|---|---|---|---|---|---|')
    for r in glob_df.sort_values(['organism','component']).itertuples(index=False):
        lines.append(f'| {r.organism} | {r.component} | {r.n} | '
                     f'{r.mcc:.3f} | {r.precision:.3f} | {r.recall:.3f} |')
    lines.append('')

    # Per-component analysis
    components = ['ortholog_prior', 'PPI', 'v15_simulator', 'cross_org_LNN']
    for comp in components:
        cdf = bdf[bdf.component == comp].copy()
        if cdf.empty:
            lines.append(f'\n## {comp}\nNo bucket data.\n')
            continue
        lines.append(f'\n## {comp}\n')

        # Top buckets by MCC (within-org), excluding those with n<30
        big = cdf[cdf.n >= 30].copy()
        if big.empty:
            lines.append('No buckets with n>=30 to summarize.\n')
            continue

        # Strengths (top 10)
        top = big.sort_values('mcc', ascending=False).head(10)
        lines.append('### Strongest buckets (top 10)\n')
        lines.append('| organism | bucket | n | MCC | prec | rec |')
        lines.append('|---|---|---|---|---|---|')
        for r in top.itertuples(index=False):
            lines.append(
                f'| {r.organism} | {r.bucket_type}={r.bucket_value} | {r.n} | '
                f'{r.mcc:.3f} | {r.precision:.3f} | {r.recall:.3f} |')

        # Weaknesses (bottom 10 with n>=30)
        bot = big.sort_values('mcc').head(10)
        lines.append('\n### Weakest buckets (bottom 10, n>=30)\n')
        lines.append('| organism | bucket | n | MCC | prec | rec |')
        lines.append('|---|---|---|---|---|---|')
        for r in bot.itertuples(index=False):
            lines.append(
                f'| {r.organism} | {r.bucket_type}={r.bucket_value} | {r.n} | '
                f'{r.mcc:.3f} | {r.precision:.3f} | {r.recall:.3f} |')

        # Most consistent strong functional class (where this component
        # excels across organisms)
        kw_mean = (big[big.bucket_type.isin(KW_FEATURES) &
                       (big.bucket_value == '1')]
                   .groupby('bucket_type').mcc.agg(['mean', 'count']))
        if not kw_mean.empty:
            kw_mean = kw_mean[kw_mean['count'] >= 2].sort_values('mean',
                                                                  ascending=False)
            lines.append('\n### Mean MCC by functional class '
                          '(across orgs where bucket has n>=30)\n')
            lines.append('| keyword (bit=1) | mean MCC | n_orgs |')
            lines.append('|---|---|---|')
            for kw, row in kw_mean.iterrows():
                lines.append(f'| {kw} | {row["mean"]:.3f} | {int(row["count"])} |')

        # By conservation
        nb_mean = (big[big.bucket_type == 'n_orth_bucket']
                   .groupby('bucket_value').mcc.agg(['mean', 'count']))
        if not nb_mean.empty:
            lines.append('\n### Mean MCC by ortholog conservation\n')
            lines.append('| n_orthologs | mean MCC | n_org_buckets |')
            lines.append('|---|---|---|')
            for nb, row in nb_mean.iterrows():
                lines.append(f'| {nb} | {row["mean"]:.3f} | {int(row["count"])} |')

        # By length
        lb_mean = (big[big.bucket_type == 'length_bucket']
                   .groupby('bucket_value').mcc.agg(['mean', 'count']))
        if not lb_mean.empty:
            lines.append('\n### Mean MCC by gene length\n')
            lines.append('| length | mean MCC | n_org_buckets |')
            lines.append('|---|---|---|')
            for lb, row in lb_mean.iterrows():
                lines.append(f'| {lb} | {row["mean"]:.3f} | {int(row["count"])} |')

    # Final synthesis
    lines.append('\n## Synthesis: when each component should be trusted\n')
    lines.append(_synthesis_text())

    OUT_MD.write_text('\n'.join(lines))
    print(f'  wrote {OUT_MD}')

    # Print key findings to stdout for the run log
    print('\n[7] KEY FINDINGS')
    for comp in components:
        cdf = bdf[(bdf.component == comp) & (bdf.n >= 30)].copy()
        if cdf.empty: continue
        top3 = cdf.sort_values('mcc', ascending=False).head(3)
        bot3 = cdf.sort_values('mcc').head(3)
        print(f'\n  {comp}:')
        print(f'    BEST 3:')
        for r in top3.itertuples(index=False):
            print(f'      {r.organism:15s} {r.bucket_type:20s}={r.bucket_value:15s} '
                  f'n={r.n:4d}  MCC={r.mcc:+.3f}')
        print(f'    WORST 3:')
        for r in bot3.itertuples(index=False):
            print(f'      {r.organism:15s} {r.bucket_type:20s}={r.bucket_value:15s} '
                  f'n={r.n:4d}  MCC={r.mcc:+.3f}')


def _synthesis_text() -> str:
    return """The bucket data above lets a meta-LNN learn a routing policy
of the form "for gene G with profile P, weight component C by w(P,C)".
The empirically motivated rules are:

**Ortholog prior** — the workhorse on conserved genes.
  - Trust most when: `n_orthologs >= 3` AND gene matches a conserved
    functional class (`kw_translation`, `kw_replication`, `kw_atp`).
  - Trust least when: `n_orthologs == 0` (no signal at all — defaults
    to 0.5) OR `is_hypothetical == 1` (orthologs of hypotheticals are
    themselves usually unannotated, so the prior is noisy).

**PPI** — best on hub-like proteins, near-noise on isolated ones.
  - Trust most when: `ppi_degree_high >= 3` AND functional class is
    membrane/ribosomal/transcription complex (which physically aggregate).
  - Trust least when: `ppi_degree_high == 0` (no high-confidence partners,
    so `ppi_max_score` is below threshold and prediction defaults to 0).
  - Cross-org PPI transfer is weak — use PPI features only as a within-org
    signal, never as a cross-org-transferred prediction.

**v15 simulator** — gold standard on Syn3A, sparse cross-org.
  - Trust most when: `organism == 'syn3a'` AND `v15_conf >= 0.5`. In this
    regime v15 has near-zero false positives.
  - Trust ONLY in-domain or via dense ortholog transfer (>40% panel
    coverage). For distant orgs like M. tb (3.7% transfer coverage), v15
    contributes noise more than signal at the panel level even though its
    in-subset MCC is decent (0.38).
  - Trust least on: regulatory genes (transcription factors), expression
    machinery without metabolite outputs, genes whose only role is signal
    transduction. v15 has no detector for these.

**Cross-org LNN** — only as good as the source-target similarity.
  - Trust most when: target organism is phylogenetically close to a
    training organism (Mycoplasmas, Firmicutes).
  - Trust least when: target is biologically distant from all training
    orgs (M. tb scored 0.26 — barely better than random). Long single
    genes (very_long bucket) and hypothetical proteins are the LNN's
    weakest cells because k-mer/sequence signals are diluted.

**Routing policy a meta-LNN should learn:**

    if n_orthologs >= 3 and not is_hypothetical:
        primary = ortholog_prior
    elif organism == syn3a:
        primary = v15
    elif ppi_degree_high >= 3 and (kw_membrane or kw_translation):
        primary = PPI
    else:
        primary = cross_org_LNN  # fallback, expect noise

The "auxiliary" components should still vote — but with weights that
collapse to zero when their per-gene profile lands in their weak zones.
That's the meta-LNN's job: learn `w(profile, component)` from this table.
"""


if __name__ == '__main__':
    main()
