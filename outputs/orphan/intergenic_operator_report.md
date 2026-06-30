# Remove coding DNA, search operators in the intergenic ~11%

User's positional-prior lever, made literal: drop the coding genome, scan only
intergenic DNA for operators. E. coli / RegulonDB, leak-free.

## Genome composition
E. coli is gene-dense: **88.8% coding, 11.2% intergenic.**

## Effect of dropping coding (63 TFs)

| metric | value |
|---|---|
| operator hits that fell in coding (decoys removed) | **78%** |
| precision (site → true target), whole-genome scan | 0.071 |
| precision, **intergenic only** | **0.139** (≈ **2×**) |

Removing coding eliminates ~78% of all operator matches — essentially all
spurious — and roughly **doubles** precision. The idea is correct and necessary.

## But the two-regime split persists (sharper)

| specific TF | intergenic precision | | global TF | intergenic precision |
|---|---|---|---|---|
| torR / trpR / hypT | 1.00 | | CRP | 0.099 |
| metJ | 0.57 | | FNR | 0.054 |
| qseB / argP / tyrR | 0.50 | | H-NS | 0.051 |
| | | | IHF | 0.040 |

- **Specific operators become reliably findable** in the intergenic space
  (precision 0.5–1.0).
- **Degenerate / global regulators stay hopeless**: CRP still yields ~70,000
  intergenic hits (precision 0.10). Their decoys live *inside* the intergenic
  region next to the real sites, so there is no coding to remove that would help.

## Conclusion
Removing coding is **necessary and ~2× helpful, but not sufficient.** It cleans
the search space and makes the specific-operator TFs solid; it cannot rescue
degenerate operators. Same boundary throughout: **specific TFs → sequence works
(strongly, in the intergenic 11%); global TFs → functional edge (co-expression +
co-fitness).**

Files: colab/intergenic_operator.py, outputs/orphan/intergenic_operator.json.
