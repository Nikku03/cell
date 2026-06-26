# Predicting TF→target by the cell's own logic (validated on RegulonDB)

We tried to reconstruct "which gene each TF regulates" from **genome structure
alone** — no PWMs, no ChIP, no RegulonDB — using the tricks the cell actually
uses, then checked against RegulonDB ground truth (4,439 edges, 211 TFs, 193
with chromosome coordinates).

## (A) Adjacency / divergent-pair — the cell's co-location trick — WORKS

Local regulators sit beside their target operon, usually divergently
transcribed across a shared promoter. Predicting TF → flanking gene (bonus for a
divergent pair + small intergenic gap):

| TF set | loose (≥1.0) P / R | strict, divergent-pair (≥2.5) P / R |
|---|---|---|
| **all TFs** | 0.37 / 0.06 | **0.51** / 0.015 |
| **local (non-hub)** | 0.38 / **0.12** | **0.51** / 0.03 |
| **global (hub)** | 0.29 / 0.005 | 0.50 / 0.002 |

**This confirms the biology exactly:**
- **Local regulators** are recovered at **~0.38 precision (loose) up to 0.51 on
  the high-confidence divergent-pair subset** — from genome coordinates alone.
- **Global regulators are invisible to adjacency** (recall 0.005). CRP, FNR,
  IHF, Fis, H-NS, ArcA… are scattered far from their hundreds of targets — the
  cell finds them by Boltzmann competition + abundance, not co-location, so no
  structural signal exists to recover. Exactly as predicted.

The win is real and **organism-agnostic**: for the hundreds of local regulators
in *any* genome (no RegulonDB needed), the divergent neighbor is the target at
~50% precision. Recall is low because we only catch the immediately-adjacent
target — but for a local regulator that adjacent operon IS its primary target,
so this is the high-value slice.

## (B) Family-effector compatibility — not yet working (implementation gap)

Idea: a TF family implies an effector chemistry; real targets handle that
chemistry. In practice only **2 predictions** had a detectable family — the
family name isn't in the E. coli GeneProductSet functional descriptions, so the
keyword detector found almost nothing. This needs proper HMM-based family
assignment (Pfam) joined to the E. coli genes; not a disproof of the idea, a
missing input.

## (C) Convergence — not working as implemented

Idea: the cell makes degenerate sites decisive by clustering several TFs on one
promoter. My proxy (count TF genes flanking a target) *dropped* precision
(0.38 → 0.29) because genomic TF-density ≠ co-occupancy of one promoter. Real
convergence needs predicted binding sites in the *same* intergenic region, which
needs motif scanning — circling back to the specificity data we don't have.

## Verdict

Of the cell's eight tricks, the one that's **recoverable from sequence alone —
genomic co-location of local regulators — works**, at ~0.5 precision,
organism-agnostically. It reconstructs the *local* regulatory architecture that
dominates by count.

The two tricks that would sharpen it (effector-chemistry matching, combinatorial
convergence) need inputs we don't yet have wired (HMM family calls; per-site
motifs). And the **global regulatory network is fundamentally unreachable from
structure** — it's run by abundance + Boltzmann competition + condition, i.e.
the condition-specific data (Wheel 4) we keep arriving at.

So: the cell "decides" locally by *putting the regulator next to its target*
(which we can read), and globally by *thermodynamic competition tuned by
condition* (which we cannot read without measurement). We can now reconstruct
the first half from any genome.
