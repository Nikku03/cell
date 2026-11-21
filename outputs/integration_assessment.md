# Integration Assessment: cell model × nexus × 4d chromatin

**Ground truth on the architecture.** `outputs/subsystem_links.json`: 17 code links `4d chromatin -> nexus`, 5 `cell model -> nexus`, 8 `nexus -> cell model`, and `missing_pairs = [["4d chromatin","cell model"]]` — zero links, both directions. Exactly **one data link exists in the entire system** (`holdout_meta.json`, cell model → nexus). Every other coupling is an `import`: compile-time, carrying nothing measured. Nothing below changes that today.

---

## 1. THE SPINE — BUILD NOW

None of the five candidate couplings is build-now. What *is* build-now is a four-item dependency chain on the 4d-chromatin↔nexus axis plus one runnable scoring harness. Each item is infrastructure; only the last one can grade a claim.

### S1. The unit bridge (already built; make it explicit) — no dependencies

`nexus_methyl_propagate.elastic_network` (`K_NB = 1.0`, `K_BOND = 20.0`, `colab/nexus_methyl_propagate.py:96-97,133`) is imported by 12 of the 19 4d-chromatin modules. `colab/chromatin_timescale.py:59` imports it and pins one number against crystallographic B-factors (mean **51.55682075402366 Å²** over 13607 heavy atoms): **spring_scale = 3.6752943622332355 kcal/mol/Å²** (`outputs/chromatin_timescale.json`).

**Binding constraint:** anchored to a single structure (`1kx5.pdb`) and the network is soft — the covalent spring is 73.5 kcal/mol/Å² against a real 300–700, so every derived time is an upper bound.

**Why it is load-bearing:** it is the only thing making a 4d-chromatin energy comparable with a nexus affinity. It converts acetyl's whole-particle strain **0.5769154796125926 model units → 2.120334209705245 kcal/mol** (`outputs/orphan/chromatin_bond_energy.json`) against `dG_bind_wt = 9.0` (`colab/nexus.py:26`) — and that single division is what kills candidate 4 (below). A bridge whose first act is to reject a coupling is a working bridge.

### S2. The nucleosome solver as a reusable service — depends on S1

Two-level modal Schwarz vs one-level, on the same exact-solve reference (`outputs/orphan/chromatin_twolevel.json`): 5mC **64 iters / 2.0436909198760986 s / 4.271078528e9 flops** vs **1785 / 50.24176216125488 s / 6.237e10** — 27.9× fewer iterations, 14.6× fewer flops; acetyl 63 vs 1598. Both `coarse_only` arms hit the 4000-iteration cap (rel err 0.939, NaN), which is the control proving the fine level is not redundant. Green's route: factorise once **24.98612141609192 s** (16.0× fill-in), then **88.98566166559856 ms/column** — 281× reuse — and **82.908034324646 µs per superposition event** (`outputs/chromatin_greens.json`).

**Binding constraint:** the modal coarse space costs a **79.56912612915039 s** eigensolve and only amortises because the structure repeats ~15.5M times genome-wide (`outputs/chromatin_cost_model.json`: genome = 2.111875e11 atoms). Second and harder: **superposition is a declared CONTROL, not a test** — `outputs/physics_doctrine.json` rule 2 lists `linearity -> superposition` as "unfalsifiable by construction". The licence is operator-level only.

### S3. The Lk ledger as the topology API — depends on S2

`outputs/chromatin_rod.json` `refine_chain`: worst **Lk drift 4.440892098500626e-16** over 8 successive ops, 1600→100 bp/bead and back (n = 48→96→192→384→768→384→192→96→48), Lk = 3.0000000000000004 throughout. Torque slope **measured 5.87027377274637e-22 = predicted 2πC/L, ratio 1.0**. All five rod gates PASS.

**Binding constraint:** every one of those eight operations changes the **entire chain**. No mixed-resolution configuration exists anywhere on disk. So this licenses the *sequential* refinement schedule — 1.39e9 bead-steps total for 1 Mb, coarse level 99.8% of it, against 100.3 days for uniform 200 bp, a 124798× difference — but it does **not** license a standing fine/coarse boundary, which is the cost model's own named engineering risk.

### S4. The AMR cost ledger as the shared budget — depends on S3

I decomposed rather than trusting the headline. From `outputs/chromatin_cost_model.json`: 86679.26691682274 s / 1.733585338336455e12 bead-steps = **4.999999999999999e-08 s per bead-step on 4 cores**; 303.030303030303 s of biological time at dt 8.740617335369522e-09 s = **3.4669e10 fine steps per 200 bp bead** → **1733.46 s per fine bead on 4 cores, 27.09 s on 256**. This reproduces all four recorded AMR rows (predicted 86679.96 / 86741.67 / 87365.72 / 103937.60 vs recorded 86679.27 / 86741.67 / 87365.72 / 103937.60).

**The two facts every coupling must be priced against:** domain size is nearly free (100 kb → chr1 249 Mb is 2490× the domain for **+19.9%** cost, 86679.27 → 103937.60 s), and resolved window is strictly linear (2 kb 4.8 h, 10 kb 1.0 d, 50 kb 5.0 d, 100 kb 10.0 d on 4 cores). **Affordable envelope: one day on 256 cores = 3190 fine beads = 638 kb of resolved window = 63 windows at 10 kb, or 319 at 2 kb.**

### S5. The one scoring bar that actually runs — independent of S1–S4

The CRISPR enhancer–gene ladder is fully re-runnable in this container: `outputs/orphan/crispr_features_compendium.csv` (10331 rows, 14 columns), `outputs/orphan/invivo/compendium_tf.json`, `outputs/orphan/invivo/crispr_egpairs.tsv` (10412 rows) all present. Bars: base epi+TF **0.6075574090157142**, CTCF-count **0.6631813437390706**, shuffled-anchor control **0.5975510116521655** (`outputs/orphan/rouse_gate.json`).

**Note, because it matters below:** `crispr_egpairs.tsv` carries `strandGene` for **10412 of 10412** pairs (5612 `+`, 4800 `−`) over 2146 genes. Strand exists here and nowhere else.

**Contrast — the transcription bar does not run.** `colab/nexus_txn.py:49,65,93,98` reads `rnadecay/AvgKdegs.csv`, `k562_atac.bed.gz`, `_abcfeat.json`, `lambert_tf_symbols.json`. A filesystem-wide `find` returns **zero hits for all four**. `outputs/orphan/nexus_txn.json` stores only summary rows (r, r2, within_2x, n) — no per-gene predictions, no residuals, no split indices, and splits are regenerated by `RandomState(0)` shuffling a gene list that requires the missing target (`colab/nexus_txn.py:149`).

---

## 2. REDUCED — buildable only in cut-down form

### R1. Coordinate handoff, capped at the measured budget (candidate 5, reduced)

**Buildable:** a ranked BED of **≤63 intervals at 10 kb** (or ≤319 at 2 kb) emitted from the **16380 of 16492** gene records that carry both `chrom` and `tss` (counted directly in `outputs/orphan/cell_complete.json`). This would be the first data object ever to cross the 4d chromatin ↔ cell model boundary.

**The reduction, exactly:** the ranking is **not** measured k_syn and **not** promoter ATAC — both source files are absent. It is nexus_txn's **L1 layer**, which is entirely repo-resident: `cpg, log1p(enh), loeuf, ess, dep_frac, log1p(pubs), dark` (`colab/nexus_txn.py:118-120`), all present for all 16492 genes.

**Cost of the honesty, stated on the artifact:** L1 alone scores **r2 = 0.1582** against k_syn (`outputs/orphan/nexus_txn.json`), and that target itself agrees with direct TT-seq at **r = 0.36055946235014674 / Spearman 0.34699068010258566** on n = 7462, with `cell_specific = false` and cross-line HEK293T scoring **0.3717 > K562 0.3606** (`outputs/orphan/rate_ground_truth.json`). Label it "coordinates ranked by a weak proxy", never "measured rate".

**Hard cap, not a preference:** the unstated version — 7631 genes × 10 kb — is **6.614e8 s = 20.97 years on 4 cores / 119.6 days on 256**, 121× over the one-day-on-256 envelope. At 2 kb it is still 4.19 years on 4 cores.

### R2. The closed-form torque table (candidate 1, reduced)

**Buildable, essentially free:** τ(σ, L) = 2πC·σ·Lk₀/L with C = 4.066e-28 J m, validated at **ratio 1.0** (`outputs/chromatin_rod.json`).

**The reduction:** indexed by (σ, L), **not by gene**. σ is a hardcoded *input*, not an output — `sigmas = (0.0, -0.02, -0.04, -0.06)`, four values model-wide (`colab/chromatin_supercoil.py:204`) — and **0 of 16492** gene records carry `sigma`, `domain`, or `tad` (union of gene keys is exactly: chrom, comp, conf, cpg, dark, dep_frac, enh, ess, ess_prob, ess_src, flag, loeuf, master, name, ndis, npath, path, ppi, proc, pubs, tf, ti, tss).

**Mandatory caveat on every value it emits:** torque here is `C_TWIST * 2π * twist() / L` (`colab/chromatin_rod.py:214-218`) — a function of **twist**, not Lk. The measured twist share is **1 − 0.9656831950986892 = 0.03431680490131084** against a literature 0.25–0.30, so the table is a **lower bound, low by 7.29×–8.74×** (`outputs/chromatin_supercoil.json`, gate `writhe_fraction = false`).

### R3. Per-mark displacement fields (candidate 4, reduced) — buildable, no consumer

**Affordable:** chr22 histone marks 2.54e6 × 3.105 s at a 40 Å region = **7.89e6 core-s = 91.3 core-days**; genome ≈122× = **30.5 core-years, or 43.5 days on 256 cores** (`outputs/orphan/chromatin_mark_cost.json` log).

**The reduction:** it delivers a 3N displacement field and stops. There is no ΔΔG. And it must be re-run first, because `colab/chromatin_mark_cost.py` selects each site with `most_buried()` — every published radius, peak |u| and cost is a buried-site number.

**Flagged honestly: nothing downstream can read a 3N vector.** The only place it could enter is nexus_txn's L4, which is four integer graph degrees — `log1p(len(regs)), log1p(len(tfr)), len(tfr)/max(len(regs),1), 1.0 if regs else 0.0` (`colab/nexus_txn.py:133-136`) — with no affinity term, worth **0.0117 R2** (0.1878 → 0.1995), and unrunnable here.

---

## 3. BLOCKED

### C4 — charge mark → ΔΔG → occupancy → rate. **Blocked on energy, by a factor of 4.2.**

Acetyl's total strain over the whole 13625-atom particle is 0.5769154796125926 model units = **2.120334209705245 kcal/mol** at the repo's own spring scale. Feeding *all* of it in as ΔΔG_bind gives `bound_fraction = 0.9999863370562986` — an occupancy change of **1.37e-5**. Halving occupancy needs 9.0 kcal/mol. The mark does not possess it. And the routing is worse than the total suggests: **99.99999996%** of acetyl's energy sits in the histone compartment, **2.6595283397276796e-09** at the DNA–histone interface (`outputs/orphan/chromatin_twist_energy.json`) = 5.6e-9 kcal/mol, with `iface_top100 = 0.0` and `iface_top1000 = 0.0`. Independently: the u → ΔΔG map has never been built (no `ddg` string in any `colab/chromatin_*.py`), and `bind_ddg`'s features need a WT/MUT residue pair and a partner chain — `1kx5.pdb` has 10 chains, all histone or DNA. Even 5mC's larger 7.974712005371745 kcal/mol is 99.94% in the DNA compartment.

### C6 — replication-fork Lk ledger as topoisomerase demand. **Blocked on the failed writhe gate and on timescale, by 721×.**

The ledger is the best-tested object in the repo (drift 4.44e-16). The signal is not. Twist share **0.03431680490131084 vs 0.25–0.30** means the transported torque is **7.29×–8.74× too small**, and `outputs/chromatin_partition_probes.json` states the consequence in application words: *"too little dLk stays as TWIST, so torque is UNDER-estimated, and torque is what drives melting and topoisomerase action."* The bias is not a divisible constant — across the repo's own probes the partition runs 0.9010902300590073 (C = 55 nm) to 0.9693489686729915, a **3.22× spread** in the correction factor. Separately, at the proposed 100 kb–1 Mb domains, timescale separation is **1.3868 and 0.013868** against the reducer's threshold of **1000** (`outputs/chromatin_timescale.json` × `outputs/physics_reduce.json`) — short by **721×** and **72109×**. And there is no topoisomerase model: three prose mentions (`chromatin_rod.py:3`, `chromatin_supercoil.py:42,251`), no rate.

### C2 — superhelical compaction as a contact ingredient. **Blocked by two measured negatives on the destination side.**

The minimal form is already built and already null: `same_domain` is `FEATNAMES[5]` in `colab/hic_gate.py:155,166`, computed against the same 4904 K562 intact domains — coverage 0.0796, **net −0.003137298046080472, se 0.0029724251157528282, z = −1.0554674798883492, sign_consistent = false** (`outputs/orphan/hic_gate.json`). Above it, real cell-type-matched Hi-C contacts add **hic_over_ctcf = −0.007619271491370094** LOCO on top of CTCF counting (`outputs/orphan/contact_decompose.json`). Hi-C sees every contact regardless of physical cause, torsional ones included; the aggregate channel has been measured and is negative. Scale kills it independently: largest supercoiled system ever equilibrated is **4000 bp / N=121** (`outputs/chromatin_partition_probes.json`); median enhancer–gene span is **370,392 bp**, with **82.59% > 100 kb and only 2.40% ≤ 10 kb** (computed from `log_dist`, n = 10331). That is **92.6×**, with no Rg(σ, L) scaling law measured anywhere.

### C3 — per-gene k_syn as the event clock. **Blocked on a category error, with no headroom in the right direction.**

`colab/chromatin_cost_model.py:65` and `colab/chromatin_timescale.py:73` both label the constant as **elongation** ("33 bp/s", "s between polymerase steps"). 30.303 ms is how often *one* polymerase perturbs *one base*; k_syn is a flux in RPKM/h. Even granting the swap: flipping the tightest holding verdict (100 kb Rouse, margin **0.030303030303030304 / 0.021851328219019688 = 1.38678×**) requires **>45.76 events/s at one locus**, faster than the 33/s elongation clock; 10 kb requires 4576 /s. A per-gene initiation interval is *longer* than the per-base elongation interval, so the substitution can only move verdicts toward HOLD. Also: the contour-length axis — the only axis on which the Rouse/Zimm verdict varies — cannot be built. `cell_complete.json` has `tss` but **0 of 16492** records carry `strand`, `start` or `end`, and `loops3d` holds 767 single anchor strings (`'3:138640kb'`), not spans.

### C1 — twin-domain torque as a transcription prior. **Blocked at rank zero.**

σ is an input (`colab/chromatin_supercoil.py:204`); torque is its exact affine image (`dLk = σ·Lk₀` exactly: −6.857142857142857 / 342.85714285714283 = −0.02), so `[torque_pNnm, sigma_domain]` is rank 1, and with the only available σ constant across genes it is **rank 0** — a zero-variance column contributes exactly 0.0000 R2 against the repo's own `abc_gain_r2 = 0.0211` bar. Twin-domain is a *signed* mechanism and `strand` is absent from all 16492 records (nexus_txn's symmetric `PROM = 2000` window confirms nothing downstream knows direction). The escape hatch — "σ of the topological domain" — needs a domain assignment that does not exist. Deriving σ from elongation flux is circular against the prediction target, the exact failure mode `nexus_txn.py` declined by name for the IDR β and for k_on/k_off/k_ini.

### C5 — accessibility/rate map as AMR window placement. **Blocked on missing data and on 20.97 years.**

All three score inputs absent (`AvgKdegs.csv`, `k562_atac.bed.gz`, `_abcfeat.json` — all listed under nexus `lost` in `outputs/subsystem_map.json`; `spatac/` does not exist). Cost as stated is **6.614e8 s = 20.97 years on 4 cores**, 121× the affordable envelope. The chromVAR fallback has no coordinates (TF-level z-scores only) and its own self-drop validation passed for **4 of 25** TFs (`outputs/orphan/perturb_atac.json`). Physics-side, a *persistent* fine window is a locality assumption, and locality by norm is the repo's cleanest rejection: **quantity 1.0 against threshold 0.25**, with 28 of 28 rod beads carrying the norm at R = 10, 20, 30, 50, 70 and 100 (`outputs/physics_reduce_rod.json`). Reduced form survives as R1.

---

## 4. THE ONE THING THAT WOULD CHANGE THE MOST

**Acquire a per-locus superhelical density track for K562 (GapR-seq, psoralen/bTMP, or TOP1/TOP2 CC-seq) mapped to genomic coordinates — and test it as a single feature column before running any physics.**

Argued from the numbers:

- **It is the missing index variable for three of five candidates.** C1, C2 and C6 all terminate at the same place: σ(locus) does not exist. `0 of 16492` gene records carry `sigma`/`domain`/`tad`; the 14-column CRISPR compendium has no torsional variable; `outputs/orphan/invivo/` holds only ChIP/DNase/H3K27ac/ABC BEDs; a filesystem search for GapR/psoralen/TMP-seq/bTMP across `colab/` returns **nothing**. Everything else on the blocked list is downstream of this one absence.
- **The decisive test costs an afternoon, not 689 core-days.** The CRISPR harness runs *today* (S5). Add the raw track as one column to the 0.6631813437390706 CTCF-count arm and grade it against that arm and its own shuffled twin at 0.5975510116521655. If a *directly measured* torsion track cannot beat that bar, no simulated multiplier derived from it can — and that verdict is bought for zero simulation, against a full-physics route priced at **689 core-days** (N^2.64 fit from the two MC timing points: 0.1244 s/sweep at N=61, 0.7591 s/sweep at N=121) or **3020 core-days** at the code's own N³, *per configuration at one σ and one length*.
- **It beats the two runner-up interventions on their own numbers.** Fixing the writhe partition is cheap (87.08218145370483 s per 700-sweep 2 kb probe) and both source logs demand it — but a perfect fix still leaves C1, C2 and C6 with no index, and `chromatin_resolution.json` already tested and rejected the predeclared explanation (`falls`, `bending_costlier`, `reaches_literature` all false; *"the overshoot lives in the model"*), while `chromatin_partition_probes.json` measured every available knob as far too weak (C −42% buys −0.036; excluded diameter +60% buys −0.013; stiffness 20× buys −0.031, against a 0.216 gap). Restoring `AvgKdegs.csv` gives back a bar whose target's own reliability is **r = 0.3606 with `cell_specific = false`** and a cross-cell-line control at 0.3717 that beats it — a bar worth restoring, but not one worth building physics against first.

**The nearly-free companion fix, worth doing in the same pass:** add `strand` and a gene-end coordinate to `cell_complete.json`. Strand already exists in this container — `crispr_egpairs.tsv` carries it for 10412/10412 pairs — and its absence from the 16492 gene records is what independently kills C1's sign and C3's contour axis. Also declare the genome build: `cell_complete.json` contains no `hg38`/`hg19`/`GRCh38` string anywhere, so the 16380 coordinates cannot currently be joined to anything with confidence.

---

## 5. WHAT IS NOT MEASURED

Stated plainly — the repo has no evidence either way on these.

- **Rg(σ, L) scaling.** Every supercoiled configuration on disk is 2000, 3600 or 4000 bp. No length-scaling law for Rg was ever measured (the one length probe, 2000→4000 bp, measured only the writhe partition: 0.9370 → 0.9427).
- **Rg error bars.** No `rg_sd` field on any of the four rows in `outputs/chromatin_supercoil.json`.
- **Mixed-resolution Lk conservation.** `refine_chain` is uniform whole-chain only. A standing or moving fine/coarse boundary — the thing an AMR window requires — has never been tested.
- **Finite-amplitude linearity, on any system.** Rod probe returned **NOT TESTABLE** (deviation 0.8555143567608874 vs threshold 0.001), and the nucleosome result is a declared control (`physics_doctrine.json` rule 2). System-level superposition is unmeasured everywhere.
- **Rod null-space reproducibility.** `n_null` is 3 at h = 1e-9 and 6 at h = 1e-10; `count stable: False` (`outputs/physics_reduce_rod.json`).
- **The u → ΔΔG map.** Never attempted in any form.
- **Marks at reader-accessible sites.** All mark numbers come from `most_buried()` sites (acetyl atom 9037, phospho 6868, 5mC 5093). The solvent-exposed distribution is unmeasured.
- **Field accuracy away from a mark.** Convergence is judged on a fixed `EVAL_R = 10.0 Å` set. Accuracy at a docking surface further out: not measured.
- **`iters_warm = 5`** — the cost model calls it "the softest number in this file". `iters_cold = 64` is measured; the warm figure, on which every event-stepping estimate rests, is assumed.
- **The origin of the writhe overshoot.** `chromatin_resolution.json` recorded **1 of 3** predeclared bead counts (only N=120, so its reported change compares a row with itself); the excluded-volume sweep was killed after **3 of 5** rows (`note_excl`).
- **Topoisomerase relaxation rate, polymerase Lk-injection rate, replication timing, origin firing, fork velocity.** None present.
- **Whether derived k_syn and TT-seq are two noisy views of one quantity or two different quantities.** `rate_ground_truth.json`'s own ceiling model failed its own prediction (features scored 0.174 against the consensus where truth-plus-independent-noise required 0.261).
- **Genome build of the cell model's 16380 coordinates.** Not declared in the file that carries them.