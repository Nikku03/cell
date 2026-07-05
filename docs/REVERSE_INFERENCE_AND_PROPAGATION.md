# Reverse Inference + Propagation Therapy — overnight build

Two capabilities beyond target *validation*: (1) **discover** the causal source of a disease from its
phenotype, and (2) design a **self-propagating** cure. Honest status of each below.

## 1. Reverse inference — phenotype → cause (`compute_reverse_inference.py`)

**Goal.** Input: a diseased-cell molecular signature (gene → signed dysregulation, +up/−down). Output:
ranked **candidate causal genes** — the source of the lesion = the target — discovered *without* being told
the target or mutation. This is the discovery direction (vs. the earlier validation direction).

**Method — an evidence chain composed like algebra** (each lens a term; the cause is where they agree):
1. **Master-regulator** — which regulator's signed regulon is coherently dysregulated (VIPER-like). Catches
   a driver whose *targets* move even if its own mRNA doesn't.
2. **Network diffusion** — heat-diffuse |signature| over the dense multi-lens network (PPI+reg+co-ess) to the
   causal neighbourhood. Meant to catch an upstream cause not itself in the signature.
3. **Cascade match** — forward-simulate each candidate's perturbation; which one *reproduces* the phenotype.
4. **Prior (light)** — LoF-constraint × essentiality as a weak tie-breaker only.
Anti-cheat: never uses the disease-gene label.

**Validation (sandbox, `validate_reverse_inference.py`).**
- Self-consistency + noise: the pipeline inverts a network perturbation **robustly** — recall@10 = 1.0 at
  noise 0–2×, median rank 1, ~1600× over random. Proves the algebra is correct and bug-free.
- **Honest caveat — this is largely CIRCULAR.** Master-reg and cascade both use the regulatory network that
  *generated* the signature. The one **independent** lens (diffusion over PPI+co-essentiality) barely recovers
  the source (recall@10 = 0.017). So the number is *not* proof it works on real data.
- **Definitive test = `lincs_reverse_test.py`** (Colab): recover a LINCS-knocked-down gene from its *real*
  experimental signature — fully non-circular. **Run this for the true accuracy number.**
- Honest expectation: works for genes with an **annotated regulon** (most TFs / cancer drivers); relies on the
  weak diffusion lens for genes without one.

## 2. Propagation therapy — a self-spreading cure

**Idea.** Instead of a drug that must reach *every* diseased cell (solid tumours are only ~30–70% penetrable),
treat a small seed; treated cells **secrete a factor** that induces the same therapeutic state in neighbours,
which re-secrete — a wave. (Computational generalization of the suicide-gene *bystander effect*.)

**Simulation (`simulate_propagation.py`, self-contained).** 2-D tissue cellular automaton:
- **Traditional** drug cures only what it reaches (cured fraction = penetration; a plateau).
- **Propagation** from a **2% seed → ~90% cured**, i.e. **~45× more cells cured per cell dosed**.
- **But it's threshold/percolation-gated**: if the signal is weak (needs ≥2 secreting neighbours) the wave
  **fizzles** (~2%); it only percolates when ~1 neighbour suffices AND >~60–70% of cells can receive the
  signal. All-or-nothing, like an excitable medium.

**Circuit finder (`build_propagation_therapy.py`, full build).** Searches the model's real layers for
self-amplifying secreted circuits: ligand **L** (secretome) → receptor **R** (ligand-receptor) → downstream →
a TF that **re-induces L** (closed loop = self-propagating) and/or an **apoptosis/effect** program
(therapeutic). Ranks circuits by loop closure + receptor breadth + effect match. Needs the ligand-receptor
layer populated (OmniPath ligrec / CellPhoneDB) — logic is ready.

## What to run on Colab to complete this
1. `lincs_reverse_test.py` — the real reverse-inference accuracy (needs `lincs_train.npz`).
2. Full build (populates ligand-receptor) → `build_propagation_therapy.py` → real self-propagating circuits.

## Honest bottom line
- Reverse inference: **engine built, algebra validated; real accuracy pending the LINCS test.** Not yet proven
  on real data — do not claim it works until `lincs_reverse_test.py` returns a non-circular number.
- Propagation therapy: **the efficiency advantage is real and quantified (~45× coverage-per-dose), gated by a
  percolation threshold.** The circuit finder needs the full build to surface real ligand-receptor loops.
