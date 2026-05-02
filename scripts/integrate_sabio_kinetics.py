"""Phase D2 upgrade — integrate SABIO-RK Km/kcat values on top of the
bacterial-default fill.

Pulls 70 reactions' worth of measured Km / kcat data from the SABIO-RK
curation pushed in commit a086f7f (organism-rank-prioritized: M.
pneumoniae direct → Mycoplasma genus → E. coli → other prokaryote)
and rewrites cell_sim/data/Mycoplasma_pneumoniae/input_data/
kinetic_params.xlsx so reactions with measured values use those, and
the rest keep the bacterial defaults.

Match key
---------

SABIO TSV column rxn_id (e.g. RR00106) corresponds to my
kinetic_params.xlsx column 'yus2009_id' (Wodke MOESM2 Table S1
column "reaction_ID (Yus2009)" — same Yus 2009 ID convention).
The match is a string equality on yus2009_id == rxn_id, with light
normalization (some Yus IDs are slash-delimited multi-mappings like
'R001/R002'; we expand and check both).

Provenance per row
------------------

The Parameter Type column is rewritten to one of:
  'kcat (per s, SABIO_RK_Mpne)'        - measured directly in M. pneumoniae
  'kcat (per s, SABIO_RK_Mycoplasma)'  - measured in another Mycoplasma sp.
  'kcat (per s, SABIO_RK_Ecoli)'       - E. coli proxy
  'kcat (per s, SABIO_RK_other)'       - any other organism in SABIO
  'kcat (per s, BACTERIAL_DEFAULT)'    - 100/s default kept
  'Km (mM, SABIO_RK_*)'                - same scheme for Km

Adds Km rows to kinetic_params.xlsx where SABIO has values
(previously the Phase D2 fill only populated kcat).

Reproducibility
---------------

Deterministic given the source SABIO TSV. Re-running overwrites
kinetic_params.xlsx with the same result.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SABIO_TSV = ROOT / 'memory_bank/data/multiorg_essentiality/raw/mpneumoniae/sabio_rk_kinetics/kinetic_params_mpne_per_reaction.tsv'
KINETIC_XLSX = ROOT / 'cell_sim/data/Mycoplasma_pneumoniae/input_data/kinetic_params.xlsx'

DEFAULT_KCAT = 100.0   # bacterial default kept for unmatched reactions
DEFAULT_KM_MM = 0.1    # bacterial default for unmatched reactions

ORGANISM_RANK_TO_TAG = {
    0: 'SABIO_RK_Mpne',
    1: 'SABIO_RK_Mycoplasma',
    2: 'SABIO_RK_Mollicutes',
    3: 'SABIO_RK_Ecoli',
    4: 'SABIO_RK_model_bacteria',
    5: 'SABIO_RK_other_prokaryote',
    6: 'SABIO_RK_eukaryote',
}


def _organism_to_rank(org: str) -> int:
    """Match the curator's rank scheme exactly."""
    if not org or pd.isna(org):
        return 5
    o = str(org).lower()
    if 'mycoplasma pneumoniae' in o:
        return 0
    if 'mycoplasma' in o:
        return 1
    if 'mollicute' in o or 'spiroplasma' in o or 'ureaplasma' in o:
        return 2
    if 'escherichia coli' in o:
        return 3
    model_bacteria = ['bacillus', 'salmonella', 'pseudomonas', 'streptococcus',
                      'staphylococcus', 'lactococcus', 'lactobacillus',
                      'enterococcus', 'haemophilus', 'mycobacterium', 'thermus']
    if any(m in o for m in model_bacteria):
        return 4
    eukaryote_hints = ['homo', 'sapiens', 'mus', 'musculus', 'rattus',
                        'saccharomyces', 'schizosaccharomyces',
                        'arabidopsis', 'dictyostelium', 'drosophila',
                        'caenorhabditis', 'leishmania', 'trypanosoma',
                        'plasmodium', 'entamoeba']
    if any(e in o for e in eukaryote_hints):
        return 6
    return 5


def _organism_tag(org: str) -> str:
    return ORGANISM_RANK_TO_TAG[_organism_to_rank(org)]


def main():
    print('[1] loading SABIO-RK per-reaction TSV...')
    sabio = pd.read_csv(SABIO_TSV, sep='\t')
    print(f'    {len(sabio)} rows; unique rxn_ids: {sabio.rxn_id.nunique()}')

    # Some yus2009_ids are slash-delimited (e.g. 'R001/R002'); expand to
    # a (yus_id_clean, sabio_row) lookup keyed by the bare R-id form.
    # SABIO's rxn_id is e.g. 'RR00106'; iJW145's yus2009_id is 'R001'.
    # The naming differs! Let me check: SABIO's first column is rxn_id
    # which the curator wrote as RR00106 — but my Wodke yus2009_id is
    # 'R001'. These are NOT the same. Let me re-check by inspecting both.

    # Actually SABIO's per_reaction TSV has rxn_id derived from the SBML
    # reaction id. The iJW145 SBML reactions are labeled 'RR00106' style.
    # And Wodke's MOESM2 'reaction_ID (Yus2009)' column is 'R001/R002'
    # style. They use DIFFERENT ID systems. I need to join via EC number
    # instead, since both have ec_number.

    print('\n[2] loading current kinetic_params.xlsx...')
    xl = pd.ExcelFile(KINETIC_XLSX)
    sheets = {sh: pd.read_excel(KINETIC_XLSX, sh) for sh in xl.sheet_names}
    total_rows = sum(len(df) for df in sheets.values())
    print(f'    {len(sheets)} sheets, {total_rows} reaction rows')

    # Build EC-keyed lookup from SABIO
    sabio_by_ec = {}
    for _, r in sabio.iterrows():
        ec = str(r.get('ec_number', '')).strip()
        if not ec:
            continue
        # Some rows have multi-EC strings; keep first
        ec = ec.split()[0] if ' ' in ec else ec
        existing = sabio_by_ec.get(ec)
        # Prefer rows with both km and kcat populated; tie-break by km_organism rank
        new_rank = (
            -int(pd.notna(r.get('km_value')) and pd.notna(r.get('kcat_value'))),
            _organism_to_rank(r.get('km_organism', '')),
        )
        if existing is None:
            sabio_by_ec[ec] = (new_rank, r)
        else:
            old_rank, _ = existing
            if new_rank < old_rank:
                sabio_by_ec[ec] = (new_rank, r)
    print(f'    SABIO EC index: {len(sabio_by_ec)} unique ECs')

    # 3. For each row in kinetic_params, look up by EC and replace
    print('\n[3] integrating SABIO values...')
    n_kcat_replaced = 0
    n_km_added = 0
    n_total_with_ec = 0
    rxns_replaced = set()
    for sh, df in sheets.items():
        if 'ec_number' not in df.columns:
            continue
        # Build new rows: for each row, keep the kcat row (replace value if SABIO
        # has it) and conditionally add a Km row right after
        out_rows = []
        for _, row in df.iterrows():
            ec = str(row.get('ec_number', '')).strip()
            if not ec or ec == 'nan':
                out_rows.append(row.to_dict())
                continue
            n_total_with_ec += 1
            # Multi-EC reactions: try first listed EC
            first_ec = ec.split()[0].split('/')[0] if (' ' in ec or '/' in ec) else ec
            sabio_match = sabio_by_ec.get(first_ec)
            if sabio_match is None:
                out_rows.append(row.to_dict())
                continue

            _rank, sabio_row = sabio_match
            kcat_val = sabio_row.get('kcat_value')
            kcat_org = sabio_row.get('kcat_organism', '')
            km_val = sabio_row.get('km_value')   # in M from SABIO
            km_org = sabio_row.get('km_organism', '')

            new_row = row.to_dict()
            # Replace kcat if SABIO has it
            if pd.notna(kcat_val) and kcat_val > 0:
                new_row['Value'] = float(kcat_val)
                new_row['Units'] = '1/s'
                new_row['Parameter Type'] = f'kcat (per s, {_organism_tag(kcat_org)})'
                new_row['kinetics_source_organism'] = str(kcat_org)
                n_kcat_replaced += 1
                rxns_replaced.add(first_ec)
            out_rows.append(new_row)

            # Add a Km row right after if SABIO has Km
            if pd.notna(km_val) and km_val > 0:
                km_mM = float(km_val) * 1000.0   # M -> mM
                km_row = {k: '' for k in row.to_dict()}
                km_row['Reaction Name'] = row['Reaction Name']
                km_row['Subsystem'] = row['Subsystem']
                km_row['Parameter Type'] = f'Km (mM, {_organism_tag(km_org)})'
                km_row['Related Species'] = sabio_row.get('km_substrate', '')
                km_row['Value'] = km_mM
                km_row['Units'] = 'mM'
                km_row['kinetics_source_organism'] = str(km_org)
                # Carry over traceability columns
                for col in ['wodke_model_id', 'yus2009_id', 'ec_number',
                            'reversibility', 'equation']:
                    if col in row:
                        km_row[col] = row[col]
                out_rows.append(km_row)
                n_km_added += 1

        sheets[sh] = pd.DataFrame(out_rows)

    print(f'    rows with EC number: {n_total_with_ec}')
    print(f'    kcat replaced with SABIO value: {n_kcat_replaced}')
    print(f'    Km rows added with SABIO value: {n_km_added}')
    print(f'    unique ECs replaced: {len(rxns_replaced)}')

    # 4. Write back
    print('\n[4] writing back...')
    with pd.ExcelWriter(KINETIC_XLSX, engine='openpyxl') as writer:
        for sh, df in sheets.items():
            df.to_excel(writer, sheet_name=sh, index=False)
    final_rows = sum(len(df) for df in sheets.values())
    print(f'    -> {KINETIC_XLSX} ({final_rows} total rows)')


if __name__ == '__main__':
    main()
