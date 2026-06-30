# Interactive E. coli cell explorer (full stack)

Single-file HTML app (outputs/orphan/cell_app.html, self-contained, opens offline).
Backed by build_cell_app_data.py which fuses every method into one model.

## Full-stack essentiality (vs Keio, full genome)
| signal | precision | recall |
|---|---|---|
| FBA metabolic | 0.455 | 0.163 |
| essential core (informational) | 0.803 | 0.101 |
| conservation >=85% | 0.546 | 0.213 |
| feba fitness | 0.376 | 0.683 |
| FUSED (FBA|core|conservation) | 0.514 | 0.359 |
| FUSED + fitness | 0.363 | 0.688 |
Fusion ~doubles recall over FBA alone (0.16->0.36) at precision ~0.51.

## The app (verified in-browser, zero JS errors)
- Genome panel: search 4741 genes; click -> detail (essential/predicted/metabolic/
  TF class/regulon/conservation), editable sequence (clear = loss-of-function).
- Perturbation: knock out / mutate any genes + pick carbon source -> live growth,
  viability, and the regulatory cascade (condition-gated). e.g. KO crp viable on
  glucose, LETHAL on arabinose (araBAD-type targets forced off).
- Dynamics: SOS response live ODE with draggable DNA-damage time + play.
- Network: click a TF -> radial regulon graph (activate/repress).

Data: 4741 genes, 211 TFs, KO x 6-media FBA lookups, regulatory net, SOS params.
Files: colab/build_cell_app_data.py, outputs/orphan/cell_app.html (+template).
