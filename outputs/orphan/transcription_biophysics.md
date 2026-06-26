# Transcription biophysics for the model: RNAP, promoters, TFs

What's an exact, knowable number vs what's a *framework with parameter ranges*.
I will not invent per-sequence affinities — where they exist they come from
specific Sort-Seq experiments, not a universal table. Companion to
*An Introduction to Systems Biology* (Alon) ch. on input functions, and the
thermodynamic models of Bintu et al. (2005) and Phillips' *Physical Biology of
the Cell*.

---

## 1. The σ70 promoter: the sequences RNAP holoenzyme reads

E. coli RNAP holoenzyme = core (α₂ββ′ω) + σ factor. Housekeeping σ⁷⁰ recognizes:

| element | consensus (5'→3') | position | recognized by | role |
|---|---|---|---|---|
| **UP element** | AT-rich (proximal/distal subsites) | ~ −40 to −60 | α-CTD | boosts up to ~30× |
| **−35 box** | **TTGACA** | centered −35 | σ region 4.2 | initial recognition |
| **spacer** | (non-specific) | 17 ± 1 bp | — | geometry; length itself tunes strength |
| **extended −10** | **TG**n | just upstream of −10 | σ region 3.0 | can rescue a weak −35 |
| **−10 box (Pribnow)** | **TATAAT** | centered −10 | σ region 2.4 | melted to open the bubble |
| **discriminator** | GC-content varies | −10 to +1 | σ1.2 / β | sets ppGpp/DksA sensitivity |
| **+1 TSS** | purine (A > G) | +1 | — | first nucleotide |

Most-conserved single positions: the **A at −11 and T at −7** in the −10 box
(these flip out during melting). Fully-consensus synthetic promoters are
*pathologically strong* — natural promoters are deliberately suboptimal so TFs
have room to regulate them.

**Two kinetic steps set "strength"** (not one affinity):
1. **Closed complex** — reversible binding, equilibrium constant K_B (≈ Kd⁻¹).
2. **Open complex** — isomerization (melting ~13 bp bubble), forward rate k_f,
   largely irreversible. Strong promoters can be limited by *either* step.

---

## 2. The forces, and their energy scales (in kBT; 1 kBT ≈ 0.6 kcal/mol)

DNA–protein recognition is a sum of:

| force | nature | scale | sequence-specific? |
|---|---|---|---|
| **H-bonds (direct readout)** | side-chain ↔ base edges in major groove | ~1–2 kcal/mol (~2–3 kBT) each | **yes** — the base-specific term |
| **Electrostatic** | Arg/Lys ↔ phosphate backbone | several kBT total | no — anchoring |
| **van der Waals / shape** | packing complementarity | small per contact, large summed | partly |
| **Hydrophobic / water release** | entropy gain on interface burial | several kBT | no |
| **Indirect readout (DNA deformability)** | bending/twisting cost; A-tracts narrow the minor groove (read by Arg) | bending a promoter tens of degrees ≈ several kBT (persistence length ~50 nm / 150 bp) | **yes** — shape depends on sequence |
| **Open-complex melting** | breaking ~13 bp of dsDNA | large; lowered by A/T-rich −10 | **yes** |

The practical consequence: **each base deviation from the −10/−35 consensus
costs roughly ΔΔG ≈ 1–3 kBT**, additive to good approximation. That additivity
is exactly what a *position energy matrix* encodes.

---

## 3. Sequence → binding energy → occupancy → transcription rate

This is the part your Alon reading formalizes. The chain:

**(a) Sequence → energy.** Energy matrix ε(base, position); a site's total
mismatch energy ΔΔG = Σ_positions ε. (Additive, ~10–20% error vs reality.)

**(b) Energy → occupancy (Boltzmann / thermodynamic model).** For RNAP at a
promoter, in the simplest 2-state picture:

```
p_bound = (P/N_NS)·e^(−βΔε_RNAP) / (1 + (P/N_NS)·e^(−βΔε_RNAP))
```
P = # RNAP, N_NS = # non-specific sites (~5×10⁶ in E. coli), β = 1/kBT.

**(c) Occupancy → rate.** Transcription rate ∝ p_bound · k_f. This is Alon's
**input function**.

**Regulators modify (b):**
- **Simple repression** (repressor occludes RNAP): fold-change =
  `1 / (1 + (R/N_NS)·e^(−βΔε_R))`. LacI gives up to ~**1000×** repression.
- **Simple activation** (recruitment: activator helps RNAP bind via a
  protein–protein contact of energy ε_AP ≈ −2 to −3 kBT): fold-change rises with
  both activator occupancy *and* e^(−βε_AP). CRP gives up to ~**50×** activation.
- These combine multiplicatively → the MWC / thermodynamic regulatory algebra
  (Bintu 2005; Razo-Mejia/Phillips 2018 for quantitative LacI numbers).

**Honest parameter ranges (literature):**
- RNAP–promoter Kd: ~**1–100 nM** (strong) up to **μM** (weak).
- per-base ΔΔG in −10/−35: **~1–3 kBT**.
- activator–RNAP recruitment energy: **−2 to −3 kBT** (CRP class).
- Exact per-base energy matrices exist for **specific** promoters from
  **Sort-Seq**: Kinney et al. PNAS 2010 (lac: RNAP + CRP matrices in kBT);
  Belliveau et al. PNAS 2018 (~dozens of E. coli promoters/TFs); Brewster et
  al. 2014 (activation). **There is no genome-wide "every sequence → Kd" table**
  — that's a measurement, done promoter-by-promoter.

---

## 4. Transcription factors: what we have (and plotted)

"Every known TF → what gene → what result" exists for E. coli as RegulonDB, and
we have it: **211 TFs, 1,847 target genes, 4,670 edges.** What's in our table:
TF, target, **effect (activator/repressor/dual)**, evidence, confidence. What's
**not** in it: per-site binding *position* and *sequence* and *affinity* — those
live in RegulonDB's `BindingSiteSet.txt`/`PromoterSet.txt` (proxy-blocked here)
and in the Sort-Seq papers above. ("Strength" in our file is *evidence
confidence*, NOT Kd — don't confuse them.)

**Plotted (`regulon_plots.png`):**
- **Top global regulators** — CRP (531 targets), FNR (302), Fis (233), IHF
  (230), H-NS (194), ArcA (180), Fur (130)… each split into activated vs
  repressed targets. CRP is the canonical hub.
- **Out-degree is heavy-tailed** — a few global hubs, a long tail of specific
  TFs (median 7 targets). This is the scale-free signature Alon discusses.
- **Convergence** — mean 2.5 TFs per gene, up to 15; genes integrate multiple
  inputs (combinatorial logic).
- **Essentiality vs connectivity** — genome base essential rate 0.17; the
  highest-connectivity TFs (51+ targets, n=11) sit at **0.45 essential** —
  global regulators are ~2.7× more likely essential. (Caveat from earlier: these
  regulator labels are partly "via essential_families", so read as a trend, not
  gospel.)

---

## 5. What this means for the model (the valve idea, made quantitative)

A regulator-as-valve is, precisely, a **multiplicative fold-change on its
targets' input functions** — not a binary on/off. To do this properly we would
replace the hard "sole-activator-off → target-off" rule (which over-predicted)
with the thermodynamic fold-change: target activity = basal × Π(regulator
fold-changes). That needs, per edge, the **sign (have it)** and the **strength
(don't have it cleanly)**. The missing piece is exactly per-site affinity /
energy matrices — i.e., condition-resolved binding, which is the same gap that
points at Sort-Seq data or measured fold-changes (and, for genome-scale, at the
expression/fitness data in feba.db / PRECISE).
