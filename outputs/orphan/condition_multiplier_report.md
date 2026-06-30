# Global condition multiplier = master-regulator program activities (built + verified)

The in-vivo multiplier on intrinsic promoter strength is NOT one scalar (3%) but a
low-rank vector of master-regulator program activities. Built on PRECISE +
RegulonDB + metadata, verified.

## (a) The programs reconstruct the global condition variance
15 master-regulator regulon activities (CRP, FNR, Fur, ArcA, Fis, IHF, Lrp, SoxS,
OxyR, NarL, PhoB, OmpR, Nac, LexA + ribosomal/growth) reconstruct **R^2 = 0.571**
of all cross-condition expression variance. The multiplier IS these programs.

## (b) Effectors predict program activity (verified, correct directions)
| effector -> program | result |
|---|---|
| growth rate -> ribosomal/growth | corr +0.46 (n=195) -- growth law; growth rate is FBA-computable |
| iron limitation (DPD) -> Fur | iron-limited +0.48 vs other -0.01 (strong) |
| carbon source -> CRP | non-glucose +0.20 vs glucose -0.08 (correct) |
| anaerobic -> FNR | n=2 anaerobic in PRECISE -> underpowered (data limit) |

## Recipe (now closed end-to-end, verified)
in-vivo beta(gene,condition) = intrinsic beta(promoter sequence; R^2 0.6-0.97)
                               x  SUM master-program activities(condition effectors; 57%)
- intrinsic beta: Urtecho MPRA model (solved)
- multiplier: master-regulator programs, computable from FBA growth + conserved
  effector logic (CRP/Fur/FNR/ArcA are universal across bacteria)
Residual ~43%: finer/organism-specific programs + supercoiling -> calibration tail
(a few RNA-seq samples), and FNR-type axes need datasets with those conditions.

Files: colab/condition_multiplier.py, outputs/orphan/condition_multiplier.json.
