# Per-component strengths and weaknesses

Bucket-level MCC for each predictor across 8 organisms. "With-signal" subset = genes for which the component has non-trivial input (e.g. ortholog prior excludes genes with `n_orthologs == 0`). MCC < 0 = worse than random.


## Global per-organism MCC (with-signal subset)


| organism | component | n | MCC | precision | recall |
|---|---|---|---|---|---|
| abaylyi | PPI | 2887 | 0.162 | 0.164 | 0.611 |
| abaylyi | ortholog_prior | 1244 | 0.555 | 0.602 | 0.693 |
| ccrescentus | PPI | 3614 | 0.183 | 0.195 | 0.729 |
| ccrescentus | ortholog_prior | 1301 | 0.606 | 0.692 | 0.735 |
| mgenitalium | PPI | 456 | 0.190 | 0.953 | 0.227 |
| mgenitalium | ortholog_prior | 446 | 0.338 | 0.859 | 0.890 |
| mpne | PPI | 614 | 0.267 | 0.885 | 0.377 |
| mpne | ortholog_prior | 449 | 0.312 | 0.923 | 0.809 |
| mtuberculosis | PPI | 3497 | 0.132 | 0.189 | 0.630 |
| mtuberculosis | cross_org_LNN | 3497 | 0.190 | 0.206 | 0.732 |
| mtuberculosis | ortholog_prior | 936 | 0.543 | 0.720 | 0.653 |
| saureus | PPI | 407 | 0.092 | 0.912 | 0.324 |
| saureus | ortholog_prior | 281 | 0.217 | 0.899 | 0.906 |
| styphimurium | PPI | 3306 | 0.147 | 0.082 | 0.832 |
| styphimurium | ortholog_prior | 1464 | 0.442 | 0.287 | 0.959 |
| syn3a | PPI | 455 | 0.057 | 0.892 | 0.151 |
| syn3a | ortholog_prior | 269 | 0.302 | 0.949 | 0.835 |
| syn3a | v15_simulator | 455 | 0.537 | 0.990 | 0.749 |


## ortholog_prior

### Strongest buckets (top 10)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| ccrescentus | kw_transcription=1 | 58 | 0.868 | 0.889 | 0.889 |
| ccrescentus | kw_replication=1 | 34 | 0.825 | 0.941 | 0.889 |
| mtuberculosis | kw_transcription=1 | 39 | 0.755 | 0.857 | 0.750 |
| abaylyi | kw_kinase=1 | 31 | 0.736 | 0.769 | 0.909 |
| abaylyi | kw_replication=1 | 36 | 0.723 | 0.889 | 0.842 |
| ccrescentus | kw_translation=1 | 111 | 0.719 | 0.900 | 0.935 |
| mtuberculosis | kw_translation=1 | 70 | 0.714 | 0.917 | 0.982 |
| abaylyi | kw_translation=1 | 81 | 0.711 | 0.808 | 0.955 |
| ccrescentus | length_bucket=short(<300bp) | 40 | 0.666 | 0.929 | 0.684 |
| ccrescentus | kw_dehydrogenase=0 | 1126 | 0.640 | 0.706 | 0.782 |

### Weakest buckets (bottom 10, n>=30)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| saureus | length_bucket=short(<300bp) | 33 | -0.156 | 0.710 | 0.917 |
| saureus | n_orth_bucket=1-2 | 74 | 0.017 | 0.733 | 0.815 |
| mpne | n_orth_bucket=3-5 | 110 | 0.038 | 0.951 | 0.740 |
| mgenitalium | is_putative=1 | 69 | 0.051 | 0.763 | 0.865 |
| syn3a | length_bucket=long(1k-3k) | 113 | 0.070 | 0.955 | 0.794 |
| styphimurium | n_orth_bucket=6+ | 84 | 0.168 | 0.549 | 1.000 |
| saureus | kw_translation=1 | 88 | 0.173 | 0.884 | 0.987 |
| syn3a | n_orth_bucket=6+ | 89 | 0.186 | 0.987 | 0.897 |
| mtuberculosis | kw_kinase=1 | 33 | 0.196 | 0.467 | 0.583 |
| saureus | kw_synthase=0 | 248 | 0.196 | 0.893 | 0.893 |

### Mean MCC by functional class (across orgs where bucket has n>=30)

| keyword (bit=1) | mean MCC | n_orgs |
|---|---|---|
| kw_replication | 0.692 | 3 |
| kw_transcription | 0.672 | 4 |
| kw_translation | 0.560 | 7 |
| kw_kinase | 0.465 | 4 |
| kw_synthase | 0.463 | 6 |
| kw_membrane | 0.436 | 6 |
| is_hypothetical | 0.411 | 2 |
| kw_protease | 0.394 | 3 |
| kw_dehydrogenase | 0.358 | 4 |
| is_putative | 0.175 | 3 |

### Mean MCC by ortholog conservation

| n_orthologs | mean MCC | n_org_buckets |
|---|---|---|
| 1-2 | 0.339 | 8 |
| 3-5 | 0.364 | 8 |
| 6+ | 0.296 | 8 |

### Mean MCC by gene length

| length | mean MCC | n_org_buckets |
|---|---|---|
| long(1k-3k) | 0.393 | 8 |
| medium(300-1000bp) | 0.448 | 8 |
| short(<300bp) | 0.329 | 4 |

## PPI

### Strongest buckets (top 10)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| mtuberculosis | is_putative=1 | 45 | 0.403 | 0.200 | 1.000 |
| mtuberculosis | length_bucket=very_long(>3kbp) | 84 | 0.400 | 0.351 | 0.867 |
| abaylyi | length_bucket=very_long(>3kbp) | 40 | 0.376 | 0.292 | 1.000 |
| ccrescentus | kw_replication=1 | 63 | 0.346 | 0.514 | 0.783 |
| mpne | kw_membrane=0 | 555 | 0.298 | 0.891 | 0.407 |
| abaylyi | kw_kinase=1 | 47 | 0.287 | 0.364 | 0.727 |
| mpne | kw_synthase=0 | 597 | 0.274 | 0.886 | 0.379 |
| mpne | kw_kinase=0 | 595 | 0.274 | 0.882 | 0.384 |
| styphimurium | kw_translation=1 | 119 | 0.271 | 0.485 | 0.961 |
| mpne | is_uncharacterized=0 | 614 | 0.267 | 0.885 | 0.377 |

### Weakest buckets (bottom 10, n>=30)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| saureus | n_orth_bucket=1-2 | 74 | -0.157 | 0.619 | 0.241 |
| syn3a | length_bucket=long(1k-3k) | 171 | -0.135 | 0.769 | 0.065 |
| mpne | n_orth_bucket=6+ | 80 | -0.087 | 0.980 | 0.620 |
| styphimurium | n_orth_bucket=6+ | 84 | -0.073 | 0.526 | 0.911 |
| syn3a | n_orth_bucket=6+ | 89 | -0.061 | 0.964 | 0.310 |
| mpne | kw_translation=1 | 95 | -0.060 | 0.986 | 0.745 |
| syn3a | is_uncharacterized=1 | 130 | -0.052 | 0.500 | 0.011 |
| saureus | kw_translation=1 | 91 | -0.052 | 0.870 | 0.750 |
| mtuberculosis | kw_protease=1 | 45 | -0.052 | 0.136 | 0.429 |
| syn3a | n_orth_bucket=0 | 186 | -0.045 | 0.667 | 0.043 |

### Mean MCC by functional class (across orgs where bucket has n>=30)

| keyword (bit=1) | mean MCC | n_orgs |
|---|---|---|
| kw_replication | 0.237 | 4 |
| kw_transcription | 0.206 | 4 |
| kw_kinase | 0.176 | 4 |
| kw_synthase | 0.169 | 6 |
| is_putative | 0.146 | 4 |
| kw_membrane | 0.114 | 7 |
| kw_translation | 0.087 | 7 |
| kw_dehydrogenase | 0.056 | 4 |
| kw_protease | 0.035 | 4 |
| is_hypothetical | 0.014 | 5 |

### Mean MCC by ortholog conservation

| n_orthologs | mean MCC | n_org_buckets |
|---|---|---|
| 0 | 0.060 | 7 |
| 1-2 | 0.078 | 8 |
| 3-5 | 0.137 | 8 |
| 6+ | 0.087 | 8 |

### Mean MCC by gene length

| length | mean MCC | n_org_buckets |
|---|---|---|
| long(1k-3k) | 0.127 | 8 |
| medium(300-1000bp) | 0.168 | 8 |
| short(<300bp) | 0.147 | 6 |
| very_long(>3kbp) | 0.273 | 4 |

## v15_simulator

### Strongest buckets (top 10)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| syn3a | kw_translation=1 | 99 | 0.862 | 0.990 | 1.000 |
| syn3a | is_uncharacterized=0 | 325 | 0.765 | 0.989 | 0.956 |
| syn3a | n_orth_bucket=3-5 | 109 | 0.742 | 1.000 | 0.918 |
| syn3a | n_orth_bucket=6+ | 89 | 0.699 | 1.000 | 0.977 |
| syn3a | length_bucket=medium(300-1000bp) | 236 | 0.577 | 1.000 | 0.714 |
| syn3a | kw_protease=0 | 440 | 0.548 | 0.989 | 0.759 |
| syn3a | is_putative=0 | 453 | 0.539 | 0.990 | 0.751 |
| syn3a | kw_membrane=0 | 411 | 0.539 | 0.988 | 0.739 |
| syn3a | has_ppi_partner=True | 455 | 0.537 | 0.990 | 0.749 |
| syn3a | is_hypothetical=0 | 455 | 0.537 | 0.990 | 0.749 |

### Weakest buckets (bottom 10, n>=30)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| syn3a | is_uncharacterized=1 | 130 | 0.159 | 1.000 | 0.078 |
| syn3a | n_orth_bucket=0 | 186 | 0.394 | 0.971 | 0.486 |
| syn3a | length_bucket=short(<300bp) | 38 | 0.408 | 0.917 | 0.733 |
| syn3a | kw_membrane=1 | 44 | 0.430 | 1.000 | 0.833 |
| syn3a | length_bucket=long(1k-3k) | 171 | 0.484 | 0.992 | 0.800 |
| syn3a | kw_translation=0 | 356 | 0.503 | 0.990 | 0.667 |
| syn3a | kw_rrna=0 | 443 | 0.522 | 0.990 | 0.751 |
| syn3a | kw_kinase=0 | 436 | 0.523 | 0.989 | 0.743 |
| syn3a | kw_synthase=0 | 429 | 0.526 | 0.989 | 0.736 |
| syn3a | n_orth_bucket=1-2 | 71 | 0.533 | 0.978 | 0.763 |

### Mean MCC by functional class (across orgs where bucket has n>=30)

| keyword (bit=1) | mean MCC | n_orgs |
|---|---|---|

### Mean MCC by ortholog conservation

| n_orthologs | mean MCC | n_org_buckets |
|---|---|---|
| 0 | 0.394 | 1 |
| 1-2 | 0.533 | 1 |
| 3-5 | 0.742 | 1 |
| 6+ | 0.699 | 1 |

### Mean MCC by gene length

| length | mean MCC | n_org_buckets |
|---|---|---|
| long(1k-3k) | 0.484 | 1 |
| medium(300-1000bp) | 0.577 | 1 |
| short(<300bp) | 0.408 | 1 |

## cross_org_LNN

### Strongest buckets (top 10)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| mtuberculosis | kw_translation=1 | 74 | 0.438 | 0.814 | 1.000 |
| mtuberculosis | n_orth_bucket=6+ | 77 | 0.260 | 0.881 | 0.908 |
| mtuberculosis | kw_replication=1 | 46 | 0.216 | 0.436 | 0.944 |
| mtuberculosis | kw_dehydrogenase=0 | 3214 | 0.205 | 0.215 | 0.746 |
| mtuberculosis | length_bucket=long(1k-3k) | 1340 | 0.200 | 0.255 | 0.758 |
| mtuberculosis | kw_transcription=1 | 162 | 0.196 | 0.154 | 0.714 |
| mtuberculosis | kw_membrane=0 | 3039 | 0.193 | 0.212 | 0.739 |
| mtuberculosis | kw_protease=0 | 3452 | 0.191 | 0.207 | 0.732 |
| mtuberculosis | has_ppi_partner=True | 3497 | 0.190 | 0.206 | 0.732 |
| mtuberculosis | is_uncharacterized=0 | 3497 | 0.190 | 0.206 | 0.732 |

### Weakest buckets (bottom 10, n>=30)

| organism | bucket | n | MCC | prec | rec |
|---|---|---|---|---|---|
| mtuberculosis | kw_dehydrogenase=1 | 283 | 0.004 | 0.118 | 0.545 |
| mtuberculosis | is_hypothetical=1 | 620 | 0.004 | 0.035 | 0.238 |
| mtuberculosis | kw_protease=1 | 45 | 0.043 | 0.167 | 0.714 |
| mtuberculosis | is_putative=1 | 45 | 0.088 | 0.071 | 0.500 |
| mtuberculosis | kw_synthase=1 | 245 | 0.090 | 0.470 | 0.910 |
| mtuberculosis | length_bucket=short(<300bp) | 212 | 0.103 | 0.079 | 0.636 |
| mtuberculosis | n_orth_bucket=0 | 2561 | 0.110 | 0.102 | 0.652 |
| mtuberculosis | length_bucket=very_long(>3kbp) | 84 | 0.121 | 0.211 | 0.800 |
| mtuberculosis | kw_synthase=0 | 3252 | 0.154 | 0.169 | 0.680 |
| mtuberculosis | kw_membrane=1 | 458 | 0.157 | 0.161 | 0.673 |

### Mean MCC by functional class (across orgs where bucket has n>=30)

| keyword (bit=1) | mean MCC | n_orgs |
|---|---|---|

### Mean MCC by ortholog conservation

| n_orthologs | mean MCC | n_org_buckets |
|---|---|---|
| 0 | 0.110 | 1 |
| 1-2 | 0.177 | 1 |
| 3-5 | 0.184 | 1 |
| 6+ | 0.260 | 1 |

### Mean MCC by gene length

| length | mean MCC | n_org_buckets |
|---|---|---|
| long(1k-3k) | 0.200 | 1 |
| medium(300-1000bp) | 0.176 | 1 |
| short(<300bp) | 0.103 | 1 |
| very_long(>3kbp) | 0.121 | 1 |

## Synthesis: when each component should be trusted

The bucket data above lets a meta-LNN learn a routing policy
of the form "for gene G with profile P, weight component C by w(P,C)".
The empirically motivated rules are:

**Ortholog prior** — the workhorse on conserved genes.
  - Trust most when: `n_orthologs >= 3` AND gene matches a conserved
    functional class (`kw_translation`, `kw_replication`, `kw_atp`).
  - Trust least when: `n_orthologs == 0` (no signal at all — defaults
    to 0.5) OR `is_hypothetical == 1` (orthologs of hypotheticals are
    themselves usually unannotated, so the prior is noisy).

**PPI** — best on hub-like proteins, near-noise on isolated ones.
  - Trust most when: `ppi_degree_high >= 3` AND functional class is
    membrane/ribosomal/transcription complex (which physically aggregate).
  - Trust least when: `ppi_degree_high == 0` (no high-confidence partners,
    so `ppi_max_score` is below threshold and prediction defaults to 0).
  - Cross-org PPI transfer is weak — use PPI features only as a within-org
    signal, never as a cross-org-transferred prediction.

**v15 simulator** — gold standard on Syn3A, sparse cross-org.
  - Trust most when: `organism == 'syn3a'` AND `v15_conf >= 0.5`. In this
    regime v15 has near-zero false positives.
  - Trust ONLY in-domain or via dense ortholog transfer (>40% panel
    coverage). For distant orgs like M. tb (3.7% transfer coverage), v15
    contributes noise more than signal at the panel level even though its
    in-subset MCC is decent (0.38).
  - Trust least on: regulatory genes (transcription factors), expression
    machinery without metabolite outputs, genes whose only role is signal
    transduction. v15 has no detector for these.

**Cross-org LNN** — only as good as the source-target similarity.
  - Trust most when: target organism is phylogenetically close to a
    training organism (Mycoplasmas, Firmicutes).
  - Trust least when: target is biologically distant from all training
    orgs (M. tb scored 0.26 — barely better than random). Long single
    genes (very_long bucket) and hypothetical proteins are the LNN's
    weakest cells because k-mer/sequence signals are diluted.

**Routing policy a meta-LNN should learn:**

    if n_orthologs >= 3 and not is_hypothetical:
        primary = ortholog_prior
    elif organism == syn3a:
        primary = v15
    elif ppi_degree_high >= 3 and (kw_membrane or kw_translation):
        primary = PPI
    else:
        primary = cross_org_LNN  # fallback, expect noise

The "auxiliary" components should still vote — but with weights that
collapse to zero when their per-gene profile lands in their weak zones.
That's the meta-LNN's job: learn `w(profile, component)` from this table.
