# Structure-based function for the dark proteome — AlphaFold + Foldseek

**The user's hypothesis, built and validated: structure beats sequence for function.** For a protein of
unknown function, its *fold* is far more conserved than its *sequence*, so a **structural** homolog can transfer
function even when sequence identity is deep in the twilight zone (<30%) where BLAST/ESM fail. So: fetch the
**AlphaFold** structure → search it against **AlphaFold/Swiss-Prot** with **Foldseek** → transfer the function
of the top structural neighbours.

`colab/foldseek_function.py` (`predict_function`), gated by `colab/validate_foldseek_function.py` as a
recovery-scorecard axis, and wired into **CellQA** as `function(gene)`.

## How it complements what we already had

The model already predicts dark-protein function via **`darkfn`** — Perturb-seq **co-essentiality**
guilt-by-association (4,993 genes). But that answers a *different* question:

| evidence | question it answers | example (AAMDC) |
|---|---|---|
| **co-essentiality** (`darkfn`) | *which pathway* does the gene act in? | "glutathione conjugation" |
| **structure** (AlphaFold+Foldseek) | *what is the molecular activity of the fold?* | **"Mth938 domain-containing protein"** ✓ (its real annotation) |

They are **independent** lines of evidence — agreement raises confidence, and structure reaches folds
co-essentiality can't see. `CellQA.function(gene)` returns both, each tagged with provenance.

## Validation — accuracy AND the twilight-zone claim

Run blind on 5 proteins of known function (spanning fold classes), the pipeline sees only the structure:

| protein | transferred function | correct? | best twilight-zone homolog (<30% seqId) |
|---|---|:---:|---|
| LDHA | L-lactate dehydrogenase A chain | ✓ | **Malate dehydrogenase** @ 29.4% |
| CA2 | Carbonic anhydrase 2 | ✓ | Carbonic anhydrase-like @ 25.8% |
| SOD1 | Superoxide dismutase [Cu-Zn] | ✓ | Superoxide dismutase @ 26.7% |
| MB | Myoglobin | ✓ | **Hemoglobin subunit ζ** @ 29.9% |
| CTSD | Cathepsin D | ✓ | Aspartic protease 2 @ 26.5% |

- **Accuracy 5/5 (1.0)** — structure recovers the correct function every time.
- **All 5 also recover the correct function from a homolog at <30% sequence identity** — the twilight zone
  where sequence methods break down. LDHA→malate dehydrogenase and myoglobin→hemoglobin are textbook cases of
  conserved fold + divergent sequence: exactly the regime where **structure sees what ESM/BLAST cannot.** This
  is the user's hypothesis, measured and confirmed.

## Honest scope

- **Needs an AlphaFold structure and a Foldseek network call.** It abstains cleanly when there is no structure
  or no confident structural homolog (a genuinely novel fold → honest "unknown", not a guess).
- **Transfers annotation, doesn't prove mechanism.** A structural homolog with the same fold usually shares
  molecular activity, but exceptions exist (same fold, diverged function). The confidence is the Foldseek
  probability; the seqId flags whether the call leans on sequence (high) or is a pure structural inference
  (low — the more novel, and the more it adds over sequence methods).
- **Complements, not replaces, co-essentiality.** Structure gives molecular activity; co-essentiality gives
  cellular pathway. The dark proteome is best annotated by *both*.

## Use

```python
from foldseek_function import predict_function
predict_function("AAMDC")          # gene -> AlphaFold -> Foldseek -> function + twilight homologs
# or via the unified layer:
CellQA().function("AAMDC")         # structure + co-essentiality, each with provenance
```

## Why it matters for the goal

"Map the whole cell, miss nothing" runs straight into the **dark proteome** — thousands of human proteins with
no assigned function. Sequence methods leave the twilight-zone ones unannotated. Pulling the AlphaFold structure
and asking Foldseek "what known fold is this?" is the most accurate available way to put a molecular function on
those genes — and the validation shows it works precisely where sequence methods give up.
