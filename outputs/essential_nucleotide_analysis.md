# Essential vs non-essential CDS: NUCLEOTIDE-level analysis

**30,332 CDSs** from 9 organisms (beril_Dda3937, beril_HerbieS, beril_Magneto, beril_Methanococcus_JJ, beril_Methanococcus_S2, beril_RalstoniaBSBF1503, beril_RalstoniaGMI1000, beril_RalstoniaPSI07, mtub). Each per-gene feature is computed directly from the A/C/G/T sequence of the CDS (extracted from GFF coords + genome.fna, reverse-complemented where strand=='-').

## Pooled comparison (sorted by |AUC-0.5|)

| feature | essential mean | non-essential mean | Cohen's d | AUC |
|---|---:|---:|---:|---:|
| TpA_obs_exp | 0.4444 | 0.5244 | -0.427 | 0.384 |
| AT_skew | 0.0412 | 0.0106 | +0.234 | 0.565 |
| GC2 | 0.4386 | 0.4494 | -0.150 | 0.436 |
| length_nt | 903.2022 | 1035.9497 | -0.189 | 0.441 |
| T_frac | 0.1796 | 0.1876 | -0.169 | 0.456 |
| homopolymer_max | 4.9181 | 5.0747 | -0.151 | 0.457 |
| GC3 | 0.7746 | 0.7601 | +0.073 | 0.536 |
| A_frac | 0.2006 | 0.1964 | +0.059 | 0.533 |
| ends_with_TAA | 0.3237 | 0.2702 | +0.119 | 0.527 |
| GC1 | 0.6460 | 0.6385 | +0.084 | 0.516 |
| ends_with_TAG | 0.1289 | 0.1595 | -0.086 | 0.485 |
| ends_with_TGA | 0.5465 | 0.5697 | -0.047 | 0.488 |
| starts_with_ATG | 0.8814 | 0.8659 | +0.046 | 0.508 |
| CpG_obs_exp | 1.1236 | 1.1286 | -0.031 | 0.496 |
| GC | 0.6197 | 0.6160 | +0.034 | 0.497 |
| GC_skew | 0.0149 | 0.0158 | -0.011 | 0.498 |
| C_frac | 0.3076 | 0.3057 | +0.029 | 0.502 |
| G_frac | 0.3122 | 0.3103 | +0.037 | 0.499 |

## Within-organism (cross-org GC confound stripped)

| feature | beril_Dda3937 | beril_HerbieS | beril_Magneto | beril_Methanococcus_JJ | beril_Methanococcus_S2 | beril_RalstoniaBSBF1503 | beril_RalstoniaGMI1000 | beril_RalstoniaPSI07 | mtub |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| length_nt | 0.412 | 0.384 | 0.451 | 0.485 | 0.555 | 0.387 | 0.395 | 0.393 | 0.623 |
| A_frac | 0.582 | 0.593 | 0.511 | 0.532 | 0.531 | 0.574 | 0.569 | 0.609 | 0.517 |
| C_frac | 0.409 | 0.436 | 0.513 | 0.644 | 0.612 | 0.456 | 0.441 | 0.451 | 0.474 |
| G_frac | 0.491 | 0.450 | 0.519 | 0.647 | 0.637 | 0.487 | 0.473 | 0.454 | 0.540 |
| T_frac | 0.496 | 0.484 | 0.449 | 0.275 | 0.296 | 0.463 | 0.496 | 0.452 | 0.460 |
| GC | 0.429 | 0.411 | 0.522 | 0.684 | 0.660 | 0.454 | 0.437 | 0.432 | 0.512 |
| GC1 | 0.551 | 0.478 | 0.536 | 0.659 | 0.645 | 0.517 | 0.499 | 0.480 | 0.500 |
| GC2 | 0.423 | 0.392 | 0.466 | 0.638 | 0.622 | 0.387 | 0.381 | 0.368 | 0.427 |
| GC3 | 0.408 | 0.457 | 0.537 | 0.561 | 0.550 | 0.497 | 0.481 | 0.500 | 0.590 |
| GC_skew | 0.566 | 0.505 | 0.499 | 0.480 | 0.502 | 0.528 | 0.529 | 0.510 | 0.540 |
| AT_skew | 0.558 | 0.572 | 0.536 | 0.679 | 0.659 | 0.578 | 0.557 | 0.609 | 0.532 |
| CpG_obs_exp | 0.546 | 0.601 | 0.581 | 0.398 | 0.438 | 0.552 | 0.540 | 0.573 | 0.479 |
| TpA_obs_exp | 0.416 | 0.439 | 0.393 | 0.394 | 0.408 | 0.363 | 0.368 | 0.367 | 0.432 |
| starts_with_ATG | 0.498 | 0.510 | 0.486 | 0.511 | 0.520 | 0.506 | 0.504 | 0.510 | 0.493 |
| ends_with_TAA | 0.538 | 0.546 | 0.511 | 0.529 | 0.529 | 0.547 | 0.543 | 0.558 | 0.496 |
| ends_with_TAG | 0.490 | 0.484 | 0.499 | 0.485 | 0.492 | 0.478 | 0.479 | 0.467 | 0.489 |
| ends_with_TGA | 0.472 | 0.470 | 0.490 | 0.485 | 0.479 | 0.475 | 0.478 | 0.475 | 0.516 |
| homopolymer_max | 0.470 | 0.448 | 0.439 | 0.418 | 0.486 | 0.419 | 0.440 | 0.433 | 0.553 |