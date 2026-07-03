# Overnight autonomous run — status log

Working the INTEGRATION_PLAN phases + tissue + transformer. **Sandbox limit:** heavy data runs
(ARCHS4 59 GB, full pipeline, Perturb-seq) execute on **Colab**; here I write + unit-test + wire each
processor and run everything that fits in the sandbox. "sandbox-validated" = logic tested on real
formats/synthetic data; "pending Colab" = needs the big data + GPU run.

| phase | status | evidence |
|---|---|---|
| **P1 — ARCHS4 co-expression** | ✅ code-complete, sandbox-validated | `compute_coexpr.py`; recovers planted modules 29/29; chunked FUSE-friendly reader; wired into build + notebook (`WITH_COEXPR`). Real 59 GB run pending Colab. |
| **P2 — Convergence engine** | ✅ **built + run on real data** | `compute_convergence.py`; 4,031 convergent / 1,204 novel / 2,118 known-control; live PubMed novelty ranking; top novel links (HIRA-TLK2, DDX24-RBM28, PKP2-SCN5A) verified sensible. |
| P1+P2 combined | ✅ verified | co-expression correctly becomes a 3rd lens (n_lens=3). |
| **P3 — Causal regulome (ReMap × Perturb-seq)** | ✅ code-complete, sandbox-validated | `compute_causal_reg.py` + `perturbseq_targets.json`; signs/intersection verified; upgrades reg edges. |
| **P4 — GTEx + DoRothEA∪TRRUST** | ✅ built (union run on real data) | GTEx URL verified live; reg union 45,499 → **278,387** edges. NicheNet already wired (tissue). |
| **P5 — ncRNA (miRTarBase + LncTarD)** | ✅ code-complete, sandbox-validated | `compute_ncrna.py`; weak-evidence drop + reverse map verified. miRTarBase URL release-specific (env). |
| Self-progression analysis | ✅ | `SELF_PROGRESSION.md` |
| Tissue model (check → plan → phases) | 🟡 checked + planned | `TISSUE_PLAN.md` |
| Transformer (running notes + final architecture) | ✅ plan + notes | `TRANSFORMER_PLAN.md`, `TRANSFORMER_NOTES.md`, `HOMEWORK_BRIEF.md` |

**Homework brief (research workflow):** ARCHS4 layout confirmed, co-expression validation via EGAD
(target GO-AUROC 0.70-0.75), GTEx URL, ncRNA DBs, full transformer review — `HOMEWORK_BRIEF.md`.

## Runbook (what the user runs on Colab)
1. `WITH_COEXPR=1` in `[2b]` reads ARCHS4 from `virtual_cell_data/expression_geo/` in place (no 59 GB copy) → `coexpr_neighbors.json`.
2. Assemble runs build → model4 → coexpr fold-in → build → **convergence** (`CONV_PUBMED=1`) → `convergence.json`.
3. New outputs to grab: `convergence.json` (the novel-links asset), plus the usual `cell_complete.json.gz`.

## Honest notes
- Model 4 stays **weak** (9.2× lift but negative R²) — the value pivoted to convergence across measured lenses, exactly as planned.
- Convergence **de-risks, doesn't validate** — outputs are credible novel hypotheses, not proven facts.
