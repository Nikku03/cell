"""Phase D2.5: populate complex_formation.xlsx 'Init. Count' column for
M. pneumoniae using the limiting-subunit heuristic.

For each complex, init_count = floor(min(member abundance) / max stoichiometry)
where member abundances come from
memory_bank/data/multiorg_essentiality/raw/mpneumoniae/maier_2011_proteomics/
44320_2011_BFMSB201138_MOESM3_ESM.zip [Table S2: Protein abundances all conditions].

If a complex has zero matched members in Maier S2, Init. Count remains None.

This is the same heuristic Luthey-Schulten uses for Syn3A: the smallest
member-protein population caps the complex copy count.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'memory_bank/data/multiorg_essentiality/raw/mpneumoniae'
COMPLEX_XLSX = ROOT / 'cell_sim/data/Mycoplasma_pneumoniae/input_data/complex_formation.xlsx'

# 1. Load Maier 2011 Table S2 (Protein abundances all conditions)
print('[1] loading Maier 2011 Table S2 protein abundances...')
zip_path = RAW / 'maier_2011_proteomics/44320_2011_BFMSB201138_MOESM3_ESM.zip'
with zipfile.ZipFile(zip_path) as z:
    for fn in z.namelist():
        if 'S2 ' in fn and fn.endswith('.xlsx'):
            with z.open(fn) as fh:
                data = fh.read()
            break
    else:
        raise RuntimeError('Maier S2 xlsx not found')

abund = pd.read_excel(io.BytesIO(data))
abund_col = 'Control (copies/cell)'
abund = abund[['MPN ID', abund_col]].dropna()
abund['MPN ID'] = abund['MPN ID'].astype(str).str.strip()
abund_map = dict(zip(abund['MPN ID'], pd.to_numeric(abund[abund_col], errors='coerce').fillna(0)))
print(f'    Maier S2 abundance map: {len(abund_map)} proteins')

# 2. Load existing complex_formation.xlsx
print('[2] loading existing complex_formation.xlsx...')
xl = pd.ExcelFile(COMPLEX_XLSX)
sheets = {sh: pd.read_excel(COMPLEX_XLSX, sheet_name=sh) for sh in xl.sheet_names}
df = sheets['Complexes']
print(f'    Complexes sheet: {len(df)} rows')

# 3. Compute Init. Count for each row
print('[3] computing Init. Count via limiting-subunit heuristic...')
init_counts = []
no_match = 0
for _, r in df.iterrows():
    members_str = str(r['Genes Products']) if pd.notna(r['Genes Products']) else ''
    stoichs_str = str(r['Stoichiometries']) if pd.notna(r['Stoichiometries']) else ''
    members = [m.strip() for m in members_str.split(',') if m.strip()]
    stoichs = [s.strip() for s in stoichs_str.split(',') if s.strip()]
    if not members:
        init_counts.append(None)
        no_match += 1
        continue

    # Pad stoichs if short (default to 1)
    if len(stoichs) < len(members):
        stoichs = stoichs + ['1'] * (len(members) - len(stoichs))

    # For each member, abundance / stoichiometry = max possible copies of the complex
    member_caps = []
    for m, s in zip(members, stoichs):
        ab = abund_map.get(m)
        if ab is None or ab == 0:
            continue
        try:
            stoich_val = int(s)
        except ValueError:
            stoich_val = 1   # 'multiple' or non-numeric
        if stoich_val <= 0:
            stoich_val = 1
        member_caps.append(ab / stoich_val)

    if not member_caps:
        init_counts.append(None)
        no_match += 1
        continue

    init_counts.append(int(np.floor(min(member_caps))))

df['Init. Count'] = init_counts

n_with = sum(1 for x in init_counts if x is not None)
print(f'    populated Init. Count for {n_with}/{len(df)} complexes')
print(f'    {no_match} complexes have no Maier abundance for any member (Init. Count None)')
print()
print('=== sample populated rows ===')
populated = df[df['Init. Count'].notna()]
for _, r in populated.head(8).iterrows():
    print(f"  {r['Name'][:50]:50s} -> {int(r['Init. Count']):6d} copies")

# 4. Write back, preserving other sheets
print('\n[4] writing back...')
sheets['Complexes'] = df
with pd.ExcelWriter(COMPLEX_XLSX, engine='openpyxl') as writer:
    for sh, sub in sheets.items():
        sub.to_excel(writer, sheet_name=sh, index=False)
print(f'    -> {COMPLEX_XLSX}')
