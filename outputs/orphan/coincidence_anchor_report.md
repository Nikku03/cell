# Computing global-TF targets via the coincidence (AND-gate): proof of principle

Test the cell's mechanism computationally: for (global TF G, specific co-regulator
S) pairs sharing >=6 targets, predict the SHARED targets with G-motif alone,
S-motif alone, and the coincidence z(G)+z(S). Leak-free. 95 pairs, E. coli/RegulonDB.

## Result
| | mean AUC |
|---|---|
| global motif alone | 0.500 (chance) |
| specific partner alone | 0.529 |
| coincidence | 0.516 |
coincidence beats global-alone in 63/95 pairs.

## The real finding: super-additive for genuine pairs
| pair | G | S | coincidence |
|---|---|---|---|
| H-NS + RcsB | 0.781 | 0.718 | **0.843** |
| H-NS + PhoP | 0.690 | 0.731 | 0.741 |
| FNR + PhoB | 0.652 | 0.745 | 0.738 |
| CRP + FliZ | 0.620 | 0.633 | **0.681** |
| Fur + SoxS | 0.591 | 0.496 | **0.643** |

The AND-gate works and EXCEEDS both components for genuine co-regulator pairs --
the cell's coincidence mechanism is computationally emulable.

## But not universal (mean 0.516) because
- many "specific" partners' own motifs are weak,
- shared-target sets are tiny (6-16) -> noisy,
- the PAIRING must be known; wrong/weak partners give no gain.

## Why the cell beats us
The cell has the RIGHT partner physically present at the promoter -- it never
guesses. We must know which partner co-regulates which global-TF target; most
guesses are wrong, so the average washes out. The cell RUNS the AND-gate in
parallel; we have to SEARCH for it, and weak components compound false positives.

## Answer
Yes, conditionally: the global-TF coincidence is computable and super-additive
when anchored on a genuinely informative co-regulator with a known pairing
(H-NS+RcsB 0.84). Not a blanket solution -- the general case still needs the
pairing supplied from data. The mechanism computes; the coverage is limited.

Files: colab/coincidence_anchor.py, outputs/orphan/coincidence_anchor.json.
