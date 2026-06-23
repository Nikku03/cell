# Demand-driven cell assembly — E. coli

Model trained leave-`escherichia`-out (never saw Keio labels). Protein family =
orthogroup (where a real ESM/AlphaFold family classifier would plug in).

## The loop
1. **Wheel 2 requirements table** (core machinery a cell must have): `{'ribosomal': 31, 'synthetase': 12, 'trna': 8, 'transcription': 9, 'replication': 8, 'translation': 5, 'cell_division': 1, 'membrane': 25, 'energy': 7, 'lipid': 5, 'nucleotide': 4, 'amino_acid': 1, 'cofactor': 3}`
2. **Wheel 1 confident fill** (P>=0.90): 333 essentials, precision 0.901
3. **Gaps reported**: `{'trna': 1, 'membrane': 2, 'lipid': 1}`
4. **Search space** after removing placed + confident non-essential: 872 genes
5. **Wheel 1 families**: leftovers grouped into orthogroup families
6. **Wheel 2 inserts** 4 family-matched genes; **precision of the inserts = 0.75**

## Result

| | confident-only cell | assembled cell |
|---|---|---|
| genes | 333 | 337 |
| true essentials | 300 | 303 |
| precision | 0.901 | 0.899 |
| recall | 0.51 | 0.515 |

Gaps remaining after assembly: `none — table complete`
