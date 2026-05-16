"""M7 Cell Explorer — Gradio interactive UI.

Two interfaces:
- A natural-language chat box ("Ask anything"): parses prompts like
  "knock out dnaA", "find synthetic lethals for 0001", "is gene 0042
  essential", "compare 0001 and 0002", "speed test", etc.
- A set of structured tabs for power users: explicit knockout
  prediction, trajectory explorer, synthetic-lethal lookup,
  comparison, model inspector, custom timed knockouts, benchmark.

Run from the repo root:
    python -m cell_sim.ui.explorer_app

Or in Colab:
    !python /content/cell/cell_sim/ui/explorer_app.py

Resources are pulled from environment variables when set; otherwise
default to common Colab paths under /content/drive/MyDrive. Override
via:
    M7_DRIVE_DIR, M7_SBML_PATH, M7_KINETIC_XLSX, M7_PROTEOMICS,
    M7_GENE_CLASS_CSV, M7_MEDIUM_XLSX, M7_PROTEIN_METAB,
    M7_LARGE_SUB, M7_SPATIAL_PARQ, M7_GIBBS_CSV

The UI auto-detects which checkpoint files exist (M7.6 / 7.7 / 7.8)
and offers a dropdown to switch.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')                            # headless backend
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cell_sim.lgnn.analysis.knockout_sweep import _full_locus_row_indices
from cell_sim.lgnn.data.species_graph import load_species_graph
from cell_sim.lgnn.data.stoichiometric_matrix import (
    load_stoichiometric_matrix_for_pinn,
)
from cell_sim.lgnn.models.gnn_v7_hybrid import CellGNNv7Hybrid
from cell_sim.lgnn.training.train_m7 import _build_combined_static_features


mpl.rcParams.update({
    'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'legend.frameon': False,
})


# ============================================================
# Paths (env-overridable)
# ============================================================
def _env(k: str, default: str) -> str:
    return os.environ.get(k, default)


DRIVE_DIR      = _env('M7_DRIVE_DIR',  '/content/drive/MyDrive/cell_count_dynamics')
DRIVE_INPUT    = _env('M7_DRIVE_INPUT',
                       '/content/drive/MyDrive/Minimal_Cell_4DWCM/extracted/Minimal_Cell_4DWCM/CODE/input_data')
DRIVE_ANALYS   = _env('M7_DRIVE_ANALYSIS',
                       '/content/drive/MyDrive/Minimal_Cell_4DWCM/extracted/Minimal_Cell_4DWCM/CODE/analysis')
SBML_PATH      = _env('M7_SBML_PATH',       f'{DRIVE_INPUT}/Syn3A_updated.xml')
KINETIC_XLSX   = _env('M7_KINETIC_XLSX',    f'{DRIVE_INPUT}/kinetic_params_new.xlsx')
PROTEOMICS     = _env('M7_PROTEOMICS',      f'{DRIVE_INPUT}/initial_concentrations.xlsx')
PROT_METAB     = _env('M7_PROTEIN_METAB',   f'{DRIVE_INPUT}/protein_metabolites.xlsx')
LARGE_SUB      = _env('M7_LARGE_SUB',       f'{DRIVE_INPUT}/LargeSubunit.xlsx')
MEDIUM_XLSX    = _env('M7_MEDIUM_XLSX',     f'{DRIVE_ANALYS}/GIP_Gibbs_Concs_Fluxes/Medium.xlsx')
SPATIAL_PARQ   = _env('M7_SPATIAL_PARQ',    f'{DRIVE_DIR}/syn3a_spatial_features.parquet')
GIBBS_CSV      = _env('M7_GIBBS_CSV',       f'{DRIVE_DIR}/gibbs.csv')
SPECIES_GRAPH  = _env('M7_SPECIES_GRAPH',   f'{DRIVE_DIR}/species_graph_v7.pt')
VAL_CACHE      = _env('M7_VAL_CACHE',       f'{DRIVE_DIR}/preload_val.pt')
SIGMA_PATH     = _env('M7_SIGMA',           f'{DRIVE_DIR}/sigma_train_reps_1to49.pt')
SWEEP_CSV      = _env('M7_SWEEP_CSV',       f'{DRIVE_DIR}/knockout_sweep_m77.csv')
DOUBLE_CSV     = _env('M7_DOUBLE_CSV',      f'{DRIVE_DIR}/double_ko_all.csv')
GENE_CLASS_CSV = _env('M7_GENE_CLASS_CSV',
                       '/content/cell/memory_bank/data/syn3a_gene_class_features.csv')


# ============================================================
# Globals (populated in _initialize)
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SG = None
ROW_NAMES: List[str] = []
VAL_DATA: Optional[torch.Tensor] = None
SIGMA: Optional[torch.Tensor] = None
GENE_GROUPS = {}
SWEEP_DF: Optional[pd.DataFrame] = None
DOUBLE_DF: Optional[pd.DataFrame] = None
GENE_INFO = {}
MODELS = {}
DEFAULT_MODEL: Optional[str] = None


# ============================================================
# Helpers
# ============================================================
def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    return img


def _load_m7_model(checkpoint_path: str):
    obj = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg_d = dict(obj['cfg'])
    override = {
        'sbml_path_for_S': SBML_PATH,
        'spatial_parquet': SPATIAL_PARQ,
        'proteomics_xlsx': PROTEOMICS,
        'kinetic_params_xlsx': KINETIC_XLSX,
        'medium_xlsx': MEDIUM_XLSX,
        'protein_metabolites_xlsx': PROT_METAB,
        'large_subunit_xlsx': LARGE_SUB,
        'gibbs_csv_path': GIBBS_CSV,
        'complex_formation_xlsx': '',
        'use_complex_constraints': False,
    }
    cfg_d.update(override)
    S_raw, fi_raw, _, _ = load_stoichiometric_matrix_for_pinn(
        cfg_d['sbml_path_for_S'], ROW_NAMES,
    )
    S = (S_raw if isinstance(S_raw, torch.Tensor)
         else torch.from_numpy(S_raw)).to(DEVICE)
    fidx = (fi_raw if isinstance(fi_raw, torch.Tensor)
            else torch.from_numpy(fi_raw)).cpu()
    stat = _build_combined_static_features(
        ROW_NAMES,
        use_role=cfg_d.get('use_role_features', True),
        use_spatial=cfg_d.get('use_spatial_features', False),
        use_proteomics=cfg_d.get('use_proteomics_features', False),
        use_kinetic_priors=cfg_d.get('use_kinetic_priors', False),
        use_complex_constraints=False,
        use_medium_features=cfg_d.get('use_medium_features', False),
        use_regulatory_features=cfg_d.get('use_regulatory_features', False),
        use_ribosome_subunit=cfg_d.get('use_ribosome_subunit', False),
        use_thermodynamics=cfg_d.get('use_thermodynamics', False),
        spatial_parquet=cfg_d.get('spatial_parquet'),
        proteomics_xlsx=cfg_d.get('proteomics_xlsx'),
        gene_class_csv=cfg_d.get('gene_class_csv'),
        kinetic_params_xlsx=cfg_d.get('kinetic_params_xlsx'),
        complex_formation_xlsx='',
        medium_xlsx=cfg_d.get('medium_xlsx'),
        protein_metabolites_xlsx=cfg_d.get('protein_metabolites_xlsx'),
        large_subunit_xlsx=cfg_d.get('large_subunit_xlsx'),
        gibbs_csv_path=cfg_d.get('gibbs_csv_path'),
        verbose=False,
    )
    m = CellGNNv7Hybrid(
        graph=SG, stoich_matrix=S, flux_indices=fidx,
        hidden=cfg_d.get('hidden', 64),
        n_layers=cfg_d.get('n_layers', 3),
        n_input_channels=cfg_d.get('n_input_channels', 2),
        static_features=stat,
        use_checkpoint=cfg_d.get('use_checkpoint', True),
        edge_chunk_size=cfg_d.get('edge_chunk_size'),
        cfc_tau_min=cfg_d.get('cfc_tau_min', 0.1),
        rate_clip=cfg_d.get('rate_clip', 6.0),
        use_residual=cfg_d.get('use_residual', True),
        multi_step_horizon=cfg_d.get('multi_step_horizon', 1),
    ).to(DEVICE)
    if cfg_d.get('use_bf16', True) and DEVICE == 'cuda':
        m = m.to(torch.bfloat16)
    m.load_state_dict(obj['state_dict'], strict=False)
    return m.eval().to(torch.float32), cfg_d


def _initialize():
    global SG, ROW_NAMES, VAL_DATA, SIGMA, GENE_GROUPS, SWEEP_DF, DOUBLE_DF
    global GENE_INFO, MODELS, DEFAULT_MODEL
    print('M7 Explorer: loading resources...')
    SG = load_species_graph(SPECIES_GRAPH)
    ROW_NAMES = list(SG.row_names)
    VAL_DATA = torch.load(VAL_CACHE, map_location=DEVICE, weights_only=False)
    SIGMA = torch.load(SIGMA_PATH, map_location=DEVICE, weights_only=False)
    GENE_GROUPS = _full_locus_row_indices(ROW_NAMES)
    SWEEP_DF = pd.read_csv(SWEEP_CSV)
    if os.path.exists(DOUBLE_CSV):
        DOUBLE_DF = pd.read_csv(DOUBLE_CSV)
        print(f'  double-KO data: {len(DOUBLE_DF):,} pairs')
    if os.path.exists(GENE_CLASS_CSV):
        gc = pd.read_csv(GENE_CLASS_CSV)
        col_locus = 'locus' if 'locus' in gc.columns else gc.columns[0]
        for _, r in gc.iterrows():
            try:
                l = str(int(float(r[col_locus]))).zfill(4)
            except (TypeError, ValueError):
                l = str(r[col_locus]).zfill(4)
            if l:
                GENE_INFO[l] = {
                    'gene_name': str(r.get('gene_name', f'JCVISYN3A_{l}')),
                    'function':  str(r.get('product', r.get('Product', 'unknown'))),
                    'essential': bool(r.get('essential', False)) if 'essential' in r else None,
                }
    # Try models in preference order
    candidates = [
        ('M7.8', f'{DRIVE_DIR}/m7_8_best.pt'),
        ('M7.7', f'{DRIVE_DIR}/m7_7_best.pt'),
        ('M7.6', f'{DRIVE_DIR}/m7_6_best.pt'),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                m, c = _load_m7_model(path)
                MODELS[name] = (m, c, path)
                print(f'  loaded {name}: {sum(p.numel() for p in m.parameters()):,} params')
            except Exception as e:
                print(f'  skipped {name}: {e}')
    if not MODELS:
        raise RuntimeError('No M7 checkpoints found; set M7_DRIVE_DIR env var')
    DEFAULT_MODEL = 'M7.8' if 'M7.8' in MODELS else (
        'M7.7' if 'M7.7' in MODELS else list(MODELS.keys())[0])
    print(f'  default model: {DEFAULT_MODEL}')


def _rollout(model, x0, n_steps, t0_step):
    x = x0.clone().to(torch.float32)
    states = [x.clone()]
    for s in range(n_steps):
        t = min(t0_step + s, SIGMA.shape[0] - 1)
        v_in = SIGMA[t].unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            out = model(x.unsqueeze(0), x_var=v_in)
        x = (out[0] if isinstance(out, tuple) else out).squeeze(0)
        states.append(x.detach())
    return torch.stack(states, dim=0)


def _parse_loci(text: str) -> List[str]:
    """Accept loci ('0001'), full tags ('JCVISYN3A_0001'), gene names ('dnaA'),
    or comma-separated combinations. Returns 4-digit locus strings."""
    if not text:
        return []
    out: List[str] = []
    parts = [p.strip() for p in re.split(r'[,;]', text) if p.strip()]
    for p in parts:
        m = re.search(r'(\d{4})', p)
        if m and m.group(1) in GENE_GROUPS:
            out.append(m.group(1))
            continue
        for l, info in GENE_INFO.items():
            if info['gene_name'].lower() == p.lower():
                out.append(l)
                break
    return out


# ============================================================
# Natural-language router
# ============================================================
def _intent_router(prompt: str, model_name: str) -> Tuple[str, Optional[Image.Image], Optional[Image.Image]]:
    """Parse a free-form prompt and dispatch to the appropriate handler.

    Returns (markdown_response, optional image 1, optional image 2).
    """
    if not prompt or not prompt.strip():
        return ('_Ask me anything: "knock out dnaA", "find lethal partners for '
                '0001", "compare 0001 and 0002", "is gene 0050 essential", "top '
                '10 essential genes", "speed test"._', None, None)
    p = prompt.strip().lower()
    loci = _parse_loci(prompt)

    # SYNTHETIC LETHAL / partner queries
    if re.search(r'\b(synthetic|lethal|partner|interact|pair)\b', p):
        if not loci:
            return ('I need a gene to find partners for. Try '
                    '"synthetic lethals for 0001"', None, None)
        return _nl_synthetic_lethal(loci[0])

    # COMPARE / vs queries
    if re.search(r'\b(compare|versus|vs\.?|side by side|both)\b', p) or len(loci) >= 2:
        if len(loci) >= 2:
            return _nl_compare(loci, model_name)
        return ('To compare, give me at least 2 genes. Try '
                '"compare dnaA and 0002"', None, None)

    # TRAJECTORY queries
    if re.search(r'\b(trajectory|over time|timecourse|trace|track|evolution|profile)\b', p):
        if not loci:
            return ('Specify a gene to trace. Try '
                    '"trajectory after knocking out 0001"', None, None)
        species_filter = ''
        m = re.search(r'(?:for|of|track)\s+(\w+[\w_]*)', p)
        if m:
            cand = m.group(1)
            if not re.fullmatch(r'\d{4}', cand) and cand not in {'gene', 'genes', 'cell'}:
                species_filter = cand
        return _nl_trajectory(loci[0], species_filter, model_name)

    # ESSENTIALITY queries
    if re.search(r'\b(essential|essentiality|important|critical|vital)\b', p):
        if loci:
            return _nl_essentiality(loci[0])
        n = _extract_int(p, default=10)
        return _nl_top_essentials(n)

    # FUNCTION / what does X do
    if re.search(r'\b(what|function|product|does|info|describe|tell me about)\b', p):
        if loci:
            return _nl_gene_info(loci[0])

    # BENCHMARK / speed
    if re.search(r'\b(speed|benchmark|throughput|fast|slow|how long|how quick)\b', p):
        return _nl_benchmark(model_name)

    # COUNT / list query
    if re.search(r'\b(how many|list|top|count|rank)\b', p):
        n = _extract_int(p, default=10)
        return _nl_top_essentials(n)

    # DEFAULT: knockout
    if loci:
        return _nl_knockout(loci[0], model_name)

    return (f'Sorry — I didn\'t understand "{prompt}". Try:\n'
            '- `knock out dnaA`\n'
            '- `find synthetic lethals for 0001`\n'
            '- `is gene 0042 essential`\n'
            '- `compare 0001 and 0002`\n'
            '- `top 15 essential genes`\n'
            '- `trajectory for 0001`\n'
            '- `speed benchmark`', None, None)


def _extract_int(text: str, default: int) -> int:
    m = re.search(r'\b(\d+)\b', text)
    if m:
        v = int(m.group(1))
        if v <= 9999 and v >= 1:
            return v
    return default


# ============================================================
# Natural-language handler implementations
# ============================================================
def _nl_knockout(locus: str, model_name: str):
    return ui_predict_knockout(locus, model_name, 200, 100)


def _nl_gene_info(locus: str):
    info = GENE_INFO.get(locus, {})
    rank = '–'
    if locus in SWEEP_DF['locus_4d'].astype(str).str.zfill(4).values:
        rank = int(SWEEP_DF[SWEEP_DF['locus_4d'].astype(str).str.zfill(4)
                            == locus].index[0]) + 1
    md = f"""
### Gene info — `JCVISYN3A_{locus}`
- **Gene name**: `{info.get('gene_name', 'unknown')}`
- **Function**: {info.get('function', 'unknown')}
- **Breuer 2019 essential**: `{info.get('essential', 'unknown')}`
- **M7 predicted essentiality rank**: **{rank}** / {len(SWEEP_DF)}
- **Row count in LGNN graph**: {len(GENE_GROUPS.get(locus, []))} species

_Ask "knock out {locus}" to see what happens when you remove it._
"""
    return md, None, None


def _nl_essentiality(locus: str):
    info = GENE_INFO.get(locus, {})
    rank = '–'
    if locus in SWEEP_DF['locus_4d'].astype(str).str.zfill(4).values:
        rank = int(SWEEP_DF[SWEEP_DF['locus_4d'].astype(str).str.zfill(4)
                            == locus].index[0]) + 1
    breuer = info.get('essential')
    md = f"""
### Essentiality — `{info.get('gene_name', locus)}` (JCVISYN3A_{locus})

| source | verdict |
|--------|---------|
| **Breuer 2019 (wet-lab)** | {'Essential' if breuer else 'Non-essential' if breuer is not None else 'unknown'} |
| **M7 predicted rank** | **{rank}** of {len(SWEEP_DF)} |
| **M7 essential?** | {'YES (top 20%)' if isinstance(rank, int) and rank <= len(SWEEP_DF) * 0.2 else 'NO'} |
"""
    return md, None, None


def _nl_top_essentials(n: int):
    n = max(1, min(n, 100))
    df = SWEEP_DF.head(n).copy()
    df['locus_4d'] = df['locus_4d'].astype(str).str.zfill(4)
    df['gene_name'] = df['locus_4d'].map(
        lambda l: GENE_INFO.get(l, {}).get('gene_name', f'_{l}'))
    df['function'] = df['locus_4d'].map(
        lambda l: str(GENE_INFO.get(l, {}).get('function', ''))[:50])
    md = f"### Top {n} predicted-essential genes\n\n"
    md += df[['locus_4d', 'gene_name', 'function', 'impact_score']
             ].to_markdown(index=False, floatfmt='.4f')
    return md, None, None


def _nl_synthetic_lethal(locus: str):
    return ui_synthetic_lethal_full(locus, 15)


def _nl_compare(loci, model_name):
    info = ', '.join(GENE_INFO.get(l, {}).get('gene_name', l) for l in loci)
    img = ui_compare(', '.join(loci), model_name, 300, 100)
    return f'### Comparison of {info}', img, None


def _nl_trajectory(locus, species_filter, model_name):
    img = ui_explore_trajectory(locus, species_filter, model_name, 500, 100)
    return (f'### Trajectory after KO of `{locus}` '
            f'(species filter: `{species_filter or "auto top-changing"}`)', img, None)


def _nl_benchmark(model_name):
    out = ui_benchmark(model_name, 64, 200)
    return out, None, None


# ============================================================
# Structured handlers (called from tabs and from NL router)
# ============================================================
def ui_predict_knockout(gene_input, model_name, rollout_steps, x0_step):
    if not gene_input or not gene_input.strip():
        return 'Enter a gene', None, None
    loci = _parse_loci(gene_input)
    if not loci:
        return f'Could not match "{gene_input}"', None, None
    locus = loci[0]
    model, _, _ = MODELS[model_name]
    x0 = VAL_DATA[0, int(x0_step)].clone()
    base = _rollout(model, x0, int(rollout_steps), int(x0_step))
    x_ko = x0.clone().to(torch.float32)
    for r in GENE_GROUPS[locus]:
        x_ko[r] = 0.0
    pert = _rollout(model, x_ko, int(rollout_steps), int(x0_step))
    impact = float((pert - base).pow(2).mean().sqrt().item())
    rank = '–'
    if locus in SWEEP_DF['locus_4d'].astype(str).str.zfill(4).values:
        rank = int(SWEEP_DF[SWEEP_DF['locus_4d'].astype(str).str.zfill(4)
                            == locus].index[0]) + 1
    info = GENE_INFO.get(locus, {})

    md = f"""
### Knockout result — `JCVISYN3A_{locus}`
- **Gene name**: `{info.get('gene_name', 'unknown')}`
- **Function**: {info.get('function', 'unknown')}
- **Predicted live impact**: `{impact:.4f}`
- **Sweep rank** (essentiality, out of {len(SWEEP_DF)}): **{rank}**
- **Breuer 2019 essential**: `{info.get('essential', 'unknown')}`
- **Model**: `{model_name}`  |  **Rollout**: {int(rollout_steps)} steps from t={int(x0_step)}
"""
    diff_per_step = (pert - base.to(pert.device)).pow(2).mean(dim=1).sqrt().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(diff_per_step, color='#c0392b', lw=2)
    ax.fill_between(np.arange(len(diff_per_step)), 0, diff_per_step,
                    color='#c0392b', alpha=0.2)
    ax.set_xlabel('Timesteps after knockout')
    ax.set_ylabel(r'$\|x_{pert} - x_{base}\|$')
    ax.set_title(f'Cell-state divergence — KO of {info.get("gene_name", locus)}')
    ax.grid(alpha=0.3)
    img1 = _fig_to_pil(fig)

    final_diff = (pert[-1] - base[-1].to(pert.device)).abs().cpu().numpy()
    top_idx = final_diff.argsort()[-15:][::-1]
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    ax2.barh(range(15), final_diff[top_idx], color='#2980b9')
    ax2.set_yticks(range(15))
    ax2.set_yticklabels([ROW_NAMES[i][:30] for i in top_idx], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel(r'$|\Delta x|$ at final timestep')
    ax2.set_title(f'Top species affected by KO of {info.get("gene_name", locus)}')
    img2 = _fig_to_pil(fig2)
    return md, img1, img2


def ui_explore_trajectory(gene_input, species_filter, model_name, n_steps, x0_step):
    if not gene_input or not gene_input.strip():
        return None
    loci = _parse_loci(gene_input)
    if not loci:
        return None
    locus = loci[0]
    model, _, _ = MODELS[model_name]
    x0 = VAL_DATA[0, int(x0_step)].clone().to(torch.float32)
    base = _rollout(model, x0, int(n_steps), int(x0_step))
    x_ko = x0.clone()
    for r in GENE_GROUPS[locus]:
        x_ko[r] = 0.0
    pert = _rollout(model, x_ko, int(n_steps), int(x0_step))
    pattern = (species_filter or '').strip().lower()
    if pattern:
        keep = [i for i, n in enumerate(ROW_NAMES) if pattern in n.lower()]
    else:
        final_diff = (pert[-1] - base[-1]).abs().cpu().numpy()
        keep = final_diff.argsort()[-8:][::-1].tolist()
    keep = keep[:12]
    if not keep:
        return None
    fig, axes = plt.subplots(len(keep), 1, figsize=(8, 1.0 * len(keep)),
                             sharex=True)
    if len(keep) == 1:
        axes = [axes]
    for i, (ax, idx) in enumerate(zip(axes, keep)):
        ax.plot(base[:, idx].cpu().numpy(), color='#2c3e50', lw=1.5,
                label='baseline')
        ax.plot(pert[:, idx].cpu().numpy(), color='#c0392b', lw=1.5,
                label='knockout')
        ax.set_ylabel(ROW_NAMES[idx][:24], fontsize=8)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        if i == len(keep) - 1:
            ax.set_xlabel(f'Timesteps after t={int(x0_step)}')
    fig.suptitle(f'Trajectory — KO of '
                 f'{GENE_INFO.get(locus, {}).get("gene_name", locus)}',
                 fontsize=10)
    fig.tight_layout()
    return _fig_to_pil(fig)


def ui_synthetic_lethal_full(locus, top_n):
    if DOUBLE_DF is None or len(DOUBLE_DF) == 0:
        return ('Double-KO data not loaded. Run the all-pairs sweep first '
                'and place it at `double_ko_all.csv`.', None, None)
    rel = DOUBLE_DF[(DOUBLE_DF['gene_a'].astype(str).str.zfill(4) == locus)
                    | (DOUBLE_DF['gene_b'].astype(str).str.zfill(4) == locus)].copy()
    if len(rel) == 0:
        return (f'No double-KO pairs include `{locus}`', None, None)
    rel['partner'] = rel.apply(
        lambda r: str(r['gene_b']).zfill(4)
        if str(r['gene_a']).zfill(4) == locus
        else str(r['gene_a']).zfill(4), axis=1,
    )
    rel['partner_name'] = rel['partner'].map(
        lambda l: GENE_INFO.get(l, {}).get('gene_name', f'_{l}'))
    rel = rel.sort_values('epistasis_ratio', ascending=False).head(int(top_n))
    md = (f'### Top {int(top_n)} partners of '
          f'`{GENE_INFO.get(locus, {}).get("gene_name", locus)}`\n\n')
    md += rel[['partner', 'partner_name', 'impact_ab',
               'expected_additive', 'epistasis_ratio']
             ].to_markdown(index=False, floatfmt='.4f')
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#c0392b' if e > 1.5 else '#7f8c8d' if e > 0.8 else '#27ae60'
              for e in rel['epistasis_ratio']]
    ax.barh(range(len(rel)), rel['epistasis_ratio'].values, color=colors)
    ax.axvline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(rel)))
    ax.set_yticklabels(rel['partner_name'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Epistasis ratio (>1 synergistic, =1 additive, <1 buffered)')
    ax.set_title(f'Partners of {GENE_INFO.get(locus, {}).get("gene_name", locus)}')
    return md, _fig_to_pil(fig), None


def ui_compare(gene_list, model_name, rollout_steps, x0_step):
    if not gene_list or not gene_list.strip():
        return None
    loci = _parse_loci(gene_list)
    if not loci:
        return None
    model, _, _ = MODELS[model_name]
    x0 = VAL_DATA[0, int(x0_step)].clone().to(torch.float32)
    base = _rollout(model, x0, int(rollout_steps), int(x0_step))
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loci), 3)))
    for color, locus in zip(colors, loci[:8]):
        x_ko = x0.clone()
        for r in GENE_GROUPS[locus]:
            x_ko[r] = 0.0
        pert = _rollout(model, x_ko, int(rollout_steps), int(x0_step))
        diff = (pert - base).pow(2).mean(dim=1).sqrt().cpu().numpy()
        name = GENE_INFO.get(locus, {}).get('gene_name', locus)
        ax.plot(diff, color=color, lw=2, label=name)
    ax.set_xlabel('Timestep after KO')
    ax.set_ylabel(r'$\|x_{pert} - x_{base}\|$')
    ax.set_title('Cell-state divergence comparison')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    return _fig_to_pil(fig)


def ui_inspect(model_name):
    model, cfg_d, path = MODELS[model_name]
    n_total = sum(p.numel() for p in model.parameters())
    aux_n = (sum(p.numel() for p in model.aux_heads.parameters())
             if hasattr(model, 'aux_heads') and model.aux_heads is not None
             else 0)
    return f"""
### Model Inspector

**Checkpoint**: `{path}`
**Architecture**: `CellGNNv7Hybrid`
- hidden = {cfg_d.get('hidden', 64)}
- n_layers = {cfg_d.get('n_layers', 3)}
- n_input_channels = {cfg_d.get('n_input_channels', 2)}
- multi_step_horizon = {cfg_d.get('multi_step_horizon', 1)}
- use_residual = {cfg_d.get('use_residual', True)}
- rate_clip = {cfg_d.get('rate_clip', 6.0)}

**Parameters**:
- Total: **{n_total:,}**
- Aux heads (multi-step): {aux_n:,}

**Feature flags**:
- role={cfg_d.get('use_role_features', False)} | spatial={cfg_d.get('use_spatial_features', False)} | proteomics={cfg_d.get('use_proteomics_features', False)}
- kinetic_priors={cfg_d.get('use_kinetic_priors', False)} | complex={cfg_d.get('use_complex_constraints', False)} | medium={cfg_d.get('use_medium_features', False)}
- regulatory={cfg_d.get('use_regulatory_features', False)} | ribosome={cfg_d.get('use_ribosome_subunit', False)} | thermo={cfg_d.get('use_thermodynamics', False)}

**Training strategy**: `{cfg_d.get('train_strategy', 'standard')}`
"""


def ui_custom_ko(loci_text, model_name, x0_step, ko_at_step, rollout_steps):
    if not loci_text or not loci_text.strip():
        return 'Enter loci', None
    loci = _parse_loci(loci_text)
    if not loci:
        return 'No matches', None
    model, _, _ = MODELS[model_name]
    x0_step = int(x0_step); ko_at_step = int(ko_at_step)
    rollout_steps = int(rollout_steps)
    x0 = VAL_DATA[0, x0_step].clone().to(torch.float32)
    x = x0.clone()
    for s in range(ko_at_step):
        t = min(x0_step + s, SIGMA.shape[0] - 1)
        v_in = SIGMA[t].unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            out = model(x.unsqueeze(0), x_var=v_in)
        x = (out[0] if isinstance(out, tuple) else out).squeeze(0)
    x_ko = x.clone()
    for locus in loci:
        for r in GENE_GROUPS[locus]:
            x_ko[r] = 0.0
    base_post = [x.clone()]
    pert_post = [x_ko.clone()]
    for s in range(rollout_steps):
        t = min(x0_step + ko_at_step + s, SIGMA.shape[0] - 1)
        v_in = SIGMA[t].unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            b_out = model(base_post[-1].unsqueeze(0), x_var=v_in)
            p_out = model(pert_post[-1].unsqueeze(0), x_var=v_in)
            base_post.append((b_out[0] if isinstance(b_out, tuple) else b_out).squeeze(0).detach())
            pert_post.append((p_out[0] if isinstance(p_out, tuple) else p_out).squeeze(0).detach())
    base_post = torch.stack(base_post[1:], dim=0)
    pert_post = torch.stack(pert_post[1:], dim=0)
    diff = (pert_post - base_post).pow(2).mean(dim=1).sqrt().cpu().numpy()
    impact = float(diff.mean())
    md = f"""
### Custom timed knockout
- **Loci**: {', '.join(loci)} ({len(loci)} genes, {sum(len(GENE_GROUPS[l]) for l in loci)} rows zeroed)
- **Knockout time**: t={x0_step + ko_at_step}
- **Post-KO rollout**: {rollout_steps} steps
- **Mean impact**: `{impact:.4f}`
"""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(diff, color='#8e44ad', lw=2)
    ax.set_xlabel('Steps after knockout')
    ax.set_ylabel(r'$\|\Delta x\|$')
    ax.set_title(f'Time-resolved KO at t={x0_step + ko_at_step}')
    ax.grid(alpha=0.3)
    return md, _fig_to_pil(fig)


def ui_benchmark(model_name, batch_size, n_steps):
    model, _, _ = MODELS[model_name]
    bs = int(batch_size); steps = int(n_steps)
    x0 = (VAL_DATA[0, 100].clone().to(torch.float32)
          .unsqueeze(0).expand(bs, -1).clone())
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        x = x0
        for s in range(steps):
            t = min(100 + s, SIGMA.shape[0] - 1)
            v_in = (SIGMA[t].unsqueeze(0).expand(bs, -1).to(torch.float32))
            out = model(x, x_var=v_in)
            x = (out[0] if isinstance(out, tuple) else out)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    wall = time.time() - t0
    return f"""
### Speed benchmark — {model_name}
- Batch size: **{bs}** | Steps: **{steps}**
- Wall time: **{wall:.2f} s**
- Throughput: **{bs * steps / wall:,.0f}** step×KO/sec
- Per-step at BATCH={bs}: **{wall/steps*1000:.1f} ms**
- KO/sec for {steps}-step rollouts: **{bs / wall:.2f}**
- **Speedup vs simulator** (~6 hr/KO at full life): ~{6 * 3600 * (steps/7200) * bs / wall:,.0f}×
"""


def _top_genes_table(n: int = 50):
    df = SWEEP_DF.head(n).copy()
    df['rank'] = df.index + 1
    df['locus_4d'] = df['locus_4d'].astype(str).str.zfill(4)
    df['gene_name'] = df['locus_4d'].map(
        lambda l: GENE_INFO.get(l, {}).get('gene_name', f'_{l}'))
    df['function'] = df['locus_4d'].map(
        lambda l: str(GENE_INFO.get(l, {}).get('function', ''))[:60])
    return df[['rank', 'locus_4d', 'gene_name', 'function', 'impact_score',
               'n_gene_rows_zeroed']]


# ============================================================
# Gradio app
# ============================================================
def build_app():
    import gradio as gr
    with gr.Blocks(theme=gr.themes.Soft(primary_hue='blue'),
                   title='M7 Cell Explorer') as app:
        gr.Markdown("""
        # M7 — Whole-Cell Surrogate Explorer

        Ask biological questions in plain English, or use the structured tabs.

        > _Examples: "knock out dnaA", "find synthetic lethals for 0001",
        > "compare 0001 and 0002", "is gene 0050 essential", "top 20 essential
        > genes", "speed test"._
        """)
        model_selector = gr.Dropdown(
            choices=list(MODELS.keys()), value=DEFAULT_MODEL,
            label='Model', interactive=True,
        )

        with gr.Tabs():
            # -------- CHAT TAB (natural language) --------
            with gr.Tab('💬 Ask anything'):
                gr.Markdown('Type a question or instruction. Examples below.')
                prompt = gr.Textbox(
                    label='Your prompt',
                    placeholder='e.g. "what happens if I knock out dnaA"',
                    lines=2,
                )
                examples = gr.Examples(
                    examples=[
                        'knock out dnaA',
                        'is gene 0001 essential',
                        'top 10 essential genes',
                        'find synthetic lethals for 0001',
                        'compare 0001 and 0002 and 0003',
                        'trajectory for 0050',
                        'what does gene 0050 do',
                        'speed test',
                    ],
                    inputs=prompt,
                )
                ask_btn = gr.Button('Ask', variant='primary')
                chat_md = gr.Markdown()
                with gr.Row():
                    chat_img1 = gr.Image(label='Visualization')
                    chat_img2 = gr.Image(label='Detail')
                ask_btn.click(_intent_router, [prompt, model_selector],
                              [chat_md, chat_img1, chat_img2])
                prompt.submit(_intent_router, [prompt, model_selector],
                              [chat_md, chat_img1, chat_img2])

            # -------- USER MODE --------
            with gr.Tab(' User mode'):
                with gr.Tab('Predict knockout'):
                    with gr.Row():
                        ko_input = gr.Textbox(label='Gene', value='0001', scale=2)
                        ko_steps = gr.Slider(50, 1000, 200, step=50,
                                             label='Rollout steps')
                        ko_x0 = gr.Slider(50, 5000, 100, step=50,
                                          label='Start timestep')
                    ko_btn = gr.Button('Predict', variant='primary')
                    ko_md = gr.Markdown()
                    with gr.Row():
                        ko_p1 = gr.Image(label='Divergence over time')
                        ko_p2 = gr.Image(label='Top affected species')
                    ko_btn.click(ui_predict_knockout,
                                 [ko_input, model_selector, ko_steps, ko_x0],
                                 [ko_md, ko_p1, ko_p2])

                with gr.Tab('Trajectory explorer'):
                    with gr.Row():
                        tg = gr.Textbox(label='Gene to knock out', value='0001')
                        tf = gr.Textbox(label='Species filter (substring)',
                                        value='', placeholder='M_glc, R_, F_')
                    with gr.Row():
                        ts = gr.Slider(50, 2000, 500, step=50,
                                       label='Rollout steps')
                        tx = gr.Slider(50, 5000, 100, step=50,
                                       label='Start timestep')
                    tb = gr.Button('Plot', variant='primary')
                    tp = gr.Image()
                    tb.click(ui_explore_trajectory,
                             [tg, tf, model_selector, ts, tx], [tp])

                with gr.Tab('Synthetic lethals'):
                    with gr.Row():
                        sg = gr.Textbox(label='Gene', value='0001')
                        sn = gr.Slider(5, 50, 15, step=5, label='Top N')
                    sb = gr.Button('Find partners', variant='primary')
                    sm = gr.Markdown()
                    sp = gr.Image()
                    sb.click(ui_synthetic_lethal_full, [sg, sn], [sm, sp, sp])

                with gr.Tab('Compare knockouts'):
                    with gr.Row():
                        cl = gr.Textbox(
                            label='Comma-separated loci or names',
                            value='0001, 0002, 0003, 0004', scale=2)
                        cs = gr.Slider(50, 1000, 300, step=50,
                                       label='Rollout steps')
                        cx = gr.Slider(50, 5000, 100, step=50,
                                       label='Start timestep')
                    cb = gr.Button('Compare', variant='primary')
                    cp = gr.Image()
                    cb.click(ui_compare, [cl, model_selector, cs, cx], [cp])

                with gr.Tab('Top essentials'):
                    df_view = gr.Dataframe(value=_top_genes_table(50),
                                           interactive=False)

            # -------- DEV MODE --------
            with gr.Tab(' Developer mode'):
                with gr.Tab('Model inspector'):
                    ib = gr.Button('Refresh')
                    io_out = gr.Markdown()
                    ib.click(ui_inspect, [model_selector], [io_out])
                    app.load(ui_inspect, [model_selector], [io_out])

                with gr.Tab('Custom timed KO'):
                    with gr.Row():
                        cu_l = gr.Textbox(label='Loci', value='0001')
                        cu_x0 = gr.Slider(50, 5000, 100, step=50,
                                          label='Start t')
                        cu_t = gr.Slider(0, 3000, 50, step=10,
                                         label='KO offset')
                        cu_n = gr.Slider(50, 1000, 200, step=50,
                                         label='Post-KO rollout')
                    cu_b = gr.Button('Run', variant='primary')
                    cu_m = gr.Markdown()
                    cu_p = gr.Image()
                    cu_b.click(ui_custom_ko,
                               [cu_l, model_selector, cu_x0, cu_t, cu_n],
                               [cu_m, cu_p])

                with gr.Tab('Speed benchmark'):
                    with gr.Row():
                        bb_b = gr.Slider(1, 512, 64, step=1, label='Batch')
                        bb_s = gr.Slider(10, 2000, 200, step=10, label='Steps')
                    bb_btn = gr.Button('Benchmark', variant='primary')
                    bb_out = gr.Markdown()
                    bb_btn.click(ui_benchmark,
                                 [model_selector, bb_b, bb_s], [bb_out])

        gr.Markdown(
            '---\n*M7 — Syn3A whole-cell surrogate. ~3.4M params, '
            '8572 species, 169 PINN-mapped reactions. AUROC vs Breuer 2019 '
            '~0.64. Sub-second knockout, ~3000× faster than the simulator.*'
        )
    return app


def main():
    _initialize()
    app = build_app()
    print('\nLaunching UI...')
    app.launch(share=True, server_name='0.0.0.0', debug=False)


if __name__ == '__main__':
    main()
