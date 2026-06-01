"""Colab cell: v13 cell-state emulator - speedup pass (smaller model + bigger batch).

v12 -> v13:
  Three lightweight changes for ~3x wall-clock speedup, no architectural risk:
    1. LGNN_HIDDEN 64 -> 32: half the per-node params (CfC's A/B were the
       biggest term).  ~2x faster forward.  May dip R² slightly; set back
       to 64 for v12-quality.
    2. BATCH 16 -> 32: doubled on Blackwell's 96 GB.  More throughput per
       step; STEPS dropped 2000 → 1500 to compensate.
    3. USE_TORCH_COMPILE flag (default off, experimental): wraps the model
       in torch.compile for an extra ~1.5x.  Risky with checkpoint+autocast,
       falls back to eager on failure.

  Estimated time on RTX 6000 Pro Blackwell: ~17-25 min total
  (vs v12's ~50 min, v11's ~120 min).

  Sub-hour at K_MAX=7000: still hard.  Best honest run on v13 with USE_TORCH_COMPILE=True:
    K_MAX=7000, STEPS=500 → ~75-90 min (was ~6 hours in v11).
  To genuinely fit K=7000 in <60 min would need FP8 or graph-rewriting we haven't done.

v11 -> v12:
  1. TRUNCATED BPTT  K_MAX raised from 64 -> 256 (4x longer horizon = 4 min
     biological at 1s stride).  Memory bounded by TBPTT_CHUNK=64 — gradients
     flow back 64 steps at a time, state .detach()'d between chunks.  Without
     this, K=256 would need ~280 GB of activations (4x v11's already-tight
     budget) — TBPTT keeps memory at v11 levels but covers 4x more horizon.
     Set K_MAX=7000 for full cell cycle coverage (~5-8 hours on Blackwell).
  2. BF16 AUTOCAST  ~2x speedup on Blackwell / Hopper / A100 — both training
     and inference paths.  Cast back to fp32 between rollout iterations to
     keep state numerically clean.  Set USE_BF16=False to disable.
  3. STEPS halved (8000 -> 2000) since each step now does ~4x more work,
     keeping total gradient signal roughly constant.

  Estimated time on RTX 6000 Pro Blackwell: ~50 min total (vs v11's ~120 min).

v10 -> v11:
  Adds three more 4DWCM input files into the graph + KnownRules:
    - protein_metabolites.xlsx → regulatory P↔M binding edges (a real
      causation path: when protein knocks out, the metabolites it regulates
      drift).
    - gibbs.csv → per-reaction ΔG° (thermodynamics).  Reported in KnownRules;
      could later weight edges by |ΔG| or constrain PINN flux direction.
    - LargeSubunit.xlsx → 50S ribosome assembly stages (subunit→intermediate
      edges).  Niche but exact-physics.
  ESM-2 protein embeddings (the heaviest M7/M8 idea, ~2.5 GB model download
  for 650M parameter version) intentionally deferred — provide a separate
  utility if R²/MCC justifies the cost.

v9 -> v10 (Tier C):
  1. TIME_STRIDE=1  full simulator resolution (7,200 timesteps/traj vs 120).
     ~60x more starting positions for training; sees fast-cascade dynamics
     that the 60s stride averaged over.
  2. RICHER GRAPH  build_full_graph adds central-dogma edges per gene,
     enzyme->flux edges (from kinetics' 160 reaction->enzyme links), and
     subunit->complex edges, on top of the SBML-reaction co-occurrence.
     Without this, ~5,800 non-SBML species had no graph path to anything
     except themselves — perturbations couldn't propagate.
  3. STEPS=8,000  4x training-step budget to cover the larger data manifold.
  4. KO_N_STEPS=300  knockout rollout = 5 min biological at 1s stride,
     instead of 30s — gives fast cascades time to register.
  5. LENS_TIME_STRIDE=60  the pattern-discovery lenses subsample to keep the
     pairwise-correlation / SVD memory at v9 scale.
  Hardware: this run needs ~80 GB GPU and ~30 GB system RAM (Colab Pro+).
  Wall-clock: ~80-100 min on H100 / A100 80GB.

v8 -> v9:
  1. PINN HEAD  hardwired mass-balance for SBML-covered species.  GNN predicts
     per-reaction log-space fluxes v_log;  Δx = S · signed_expm1(v_log) is
     mass-conserving by construction (log-space bridge handles the wide
     dynamic range).  Ported from M7's pinn_head.py.
  2. STOCHASTIC HEAD + NLL LOSS  per-species log_sigma output; training uses
     Gaussian NLL instead of MSE so the model can express uncertainty
     where it can't predict precisely - breaks the deterministic noise floor.
     Ported from M8 upgrade 1/5.
  3. VARIANCE-WEIGHTED R²  honest metric: median R² over the top-K (200)
     highest-variance species.  Strips out the thousands of near-constant
     species that drag mean R² toward zero artificially.
  4. KNOCKOUT SWEEP vs BREUER 2019  the biology metric: zero each gene's
     P/R/RP/G species, roll forward 30 steps, rank by trajectory deviation.
     Compare top-N predicted-essential to Breuer 2019 experimental essentiality
     calls;  report Matthews correlation coefficient.

v7 -> v8:
  1. ARCHITECTURE: replace the transformer with a Liquid Graph Neural Network,
     porting the core M7 ideas from the parallel claude/build-m7-surrogate-Dt8w7
     branch into our scaffolding:
       - SBML-derived species graph (edges between co-occurring species + self-loops)
       - CfC (closed-form continuous-time) node update with per-species learned A,B
       - Degree-normalised message passing
     Single-step model (no context window — CfC handles temporal structure via tau).
     The PINN mass-balance head from M7 is intentionally deferred: it tangles with
     our (x-lo)/span normalisation and only covers ~2.5% of species; cleaner to
     get the CfC graph in first and add the PINN head later if it pays off.
  2. BOUNDS FIX  the v7.1 count-space cap punished species that failed validation
     by dropping them onto the loose global CLAMP (which inflated peaks from 3.5M
     to 4.8M).  Bounds now always fit val data by construction (lo<=val_min,
     hi>=val_max), so no species ever falls back to the loose clamp.

v6 -> v7:
  1. STARTUP SKIP  drop the first decimated step (t=0->t=60); the simulator's
     startup transient is non-physical, training and rollout now start from t=60.
  2. INGEST ALL INPUT FILES  one knowledge phase before training parses every
     staged 4DWCM input file:
       Syn3A_updated.xml           - SBML reaction network                (v6)
       kinetic_params.xlsx         - rate constants, reaction->enzyme map (NEW)
       initial_concentrations.xlsx - protein/mRNA/metabolite initials     (NEW)
       complex_formation.xlsx      - protein-complex assembly stoich      (NEW)
       syn3a_gene_table.csv        - gene-type labels                     (v5)
  3. SHARP TYPE SEPARATION  KnownRules (input-file facts, deterministic) and
     DiscoveredPatterns (trajectory regularities, empirical) are DIFFERENT
     KINDS of objects.  Both feed validation but stay conceptually distinct.
  4. MULTI-LENS PATTERN DISCOVERY  each lens looks for a different KIND:
       1D     - per-species monotonicity, bounds                          (v6)
       2D     - pairwise correlation on deltas (couplings)                (NEW)
       low-d  - SVD: lowest-variance directions = conservation candidates (NEW)
       time   - FFT: dominant frequency per species (periodicities)       (NEW)
       chain  - per-gene central-dogma channel cross-correlation lags     (NEW)
  5. "PHD" SUMMARY  one comprehensive printout of what the system knows.
  6. CROSS-VALIDATION  KnownRules vs trajectory: discrepancies surface as
     missing-info flags (e.g. trajectory t=0 vs xlsx initial counts).
  7. TWO-TIER (unchanged from v6): Tier 1 enforced as hard guardrails - only
     the subset that is BOTH validated AND safely enforceable (monotone, bounds).
     Tier 2 reported - everything else (conservation, couplings, periodicities).
     "It shouldn't be wrong": enforced is strictly less than discovered.

Run on Colab with Drive mounted, GPU runtime (~11 min).
"""

import glob
import os
import re
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# ── config ───────────────────────────────────────────────────────────────────
PARQUET_DIR         = ""
# v15.8.2: when True, main() runs only data-load + knowledge phase + the three
# reframe diagnostics, then exits.  ~10–15 min instead of 80–110 min.  Use this
# to verify the reframes (R1 algorithm / R2 constraints / R3 low-dim) without
# spending an hour on a full training run.  Can also be flipped via
# REFRAMES_ONLY=1 env var (so you don't have to edit cell 3).
REFRAMES_ONLY       = (os.environ.get("REFRAMES_ONLY", "0") == "1")
# v15.8.3: gate hard-projection conservation pairs to MECHANISM-BACKED only.
# When True (safe default): cpairs that don't trace to SBML interconversion,
# complex subunit-sharing, LargeSubunit assembly, ribosomal _open-state pairs,
# or known tRNA charging are demoted to Tier 2 SOFT (penalty in loss, not a
# hard projection).  This avoids enforcing trajectory-only "conservation"
# laws that might be artifacts of the specific 50 training trajectories.
# Set to False to restore the v15.5–v15.8.2 behavior of enforcing ALL 520
# discovered pairs (useful for ablation: "did strict mechanism gating help?").
CPAIR_STRICT_MECHANISM = True
TIME_STRIDE         = 60           # v15.4 [4/6]: was 30. Halves the rollout horizon (240->120 steps) so less compounding error, and K_MAX=90 now covers ~86% of the cycle (vs half at stride 30). Compare to v9 (60s -> rollout R^2 ~0.64). NEEDS a fresh training run (RESUME=False) — v15.0 ckpt is stride-30.
LENS_TIME_STRIDE    = 60           # v10: subsample for the lens phase (memory)
# Drop t=0 and t=1 of the simulator (the unnatural startup transient).  At
# TIME_STRIDE=1 that's the first 2 decimated steps; at TIME_STRIDE=60 it's the
# first 1 step (no choice, the stride already swallows seconds 1-59).
SKIP_STARTUP_STEPS  = max(1, 2 // TIME_STRIDE)
CONTEXT             = 8
D_MODEL             = 256
D_TYPE_EMBED        = 16
N_LAYERS            = 3
N_HEADS             = 8
DROPOUT             = 0.1
N_TRAIN_TRAJ        = 40
STEPS               = 3000         # v15.0: was 1500 — doubled for the LGNN+Transformer hybrid (more capacity → more steps to converge)
K_MAX               = 90           # v15.4 [4/6]: was 120. At stride 60 the trajectory is ~120 steps, so K_MAX must be < T-1 or training indexes out of bounds. 90 steps = 90 min biological (~86% of the 105-min cycle).
TBPTT_CHUNK         = 64           # v12: gradient flows back this many rollout steps at a time
USE_BF16            = True         # v12: BF16 autocast on Blackwell/Hopper/A100 — ~2x speedup
BATCH               = 32           # v13: was 16 — doubled on Blackwell 96GB; more throughput per step
LR                  = 3e-4
WEIGHT_DECAY        = 1e-5
LAMBDA_1STEP        = 1.0
LAMBDA_HYP          = 0.01         # v13.8: weight for Tier-2 hypothesis aux loss
CLAMP_LO, CLAMP_HI  = -0.2, 1.2
MONO_EPS            = 1e-4
RULE_COMPLIANCE     = 0.999
CONSERVE_DRIFT      = 0.02
PAIRWISE_TOP_K      = 50           # NEW v7
PAIRWISE_THRESHOLD  = 0.85         # NEW v7
CONSERVATION_K      = 8            # NEW v7
PERIODICITY_TOP_K   = 20           # NEW v7
GENE_LAG_MAX        = 5            # NEW v7
COUNT_BOUND_SLACK   = 0.1          # NEW v7.1: count-space high-side cap headroom
# v15.5 PhD-knowledge upgrades:
CONSPAIR_CV_MAX     = 0.02         # conservation-pair lens: max coeff-of-variation of (A+B) over the cycle to call it conserved (2% drift)
CONSPAIR_ANTICORR   = -0.90        # only test anti-correlated pairs (r < this) as conservation candidates
MI_LENS_TOP_K       = 30           # nonlinear lens: top-K mutual-information pairs to report
MI_LENS_BINS        = 8            # histogram bins for the MI estimate
COMPOSE_MIN_LENSES  = 2            # lens-composition: a species flagged by ≥ this many lenses is "mechanistically central"
# v15.7 Move B: deeper pattern lenses
GRANGER_TOP_K       = 20           # Granger-causality lens: top directed pairs to report
GRANGER_MAX_LAG     = 3            # AR lag order for the Granger test
NTUPLE_MAX_TERMS    = 3            # stoichiometric N-tuple conservation: search up to this many species
NTUPLE_TOP_K        = 20           # max N-tuples to report
REGIME_N_PHASES     = 3            # regime-conditional lens: split cycle into this many phases
MI_CORR_CEIL        = 0.7          # v15.7 fix: MI lens only keeps pairs with |corr| < this (genuinely nonlinear)
CONSPAIR_MEAN_FLOOR = 0.5          # v15.7 fix: was 1.0 — catch single-copy gene pairs that flicker during replication
LGNN_HIDDEN         = 64           # v14.9: bumped 32 → 64 to attack mode collapse (capacity-limited)
LGNN_N_LAYERS       = 3            # NEW v8: number of CfC graph layers
LGNN_CFC_TAU_MIN    = 0.1          # NEW v8: CfC time-constant minimum
LGNN_N_TYPE_EMBED   = 4            # NEW v8: gene-type embed dim
# v15.0: TemporalContext — small transformer over the past T_CTX_WINDOW states.
# Provides a global "trajectory context" vector that gets broadcast-added to
# the LGNN's hidden state. Attacks late-rollout mode collapse (attention back
# to t=0), gives phase awareness (positional encoding within window), and
# damps compounding rollout error by injecting global state info each step.
USE_TEMPORAL_CONTEXT = False       # v15.1: reverted — v15.0 added 1.1M params and 2× wall-clock but only moved rollout R² by +0.008 (within noise).  Diagnosis: history buffer fills with model predictions, transformer attends over its own errors.  Code retained, gated off.
T_CTX_WINDOW        = 8            # past states attended over (includes current)
T_CTX_HIDDEN        = 32           # per-token embed dim — keeps S→ctx projection cheap
T_CTX_LAYERS        = 2            # transformer encoder depth
T_CTX_HEADS         = 4            # multi-head attention heads
T_CTX_FF            = 64           # feed-forward dim within each encoder layer
USE_PINN_HEAD       = True         # NEW v9: hardwire mass-balance for SBML species
USE_METABOLISM_CORE = True         # NEW v13.9: replace PINN's neural flux with the bi-bi rate law for ~160 SBML reactions
USE_VOLUME_CORE     = True         # NEW v13.9: dynamic cell volume from membrane-lipid count (vs constant)
USE_CENTRAL_DOGMA   = True         # v14.3: re-enabled with per-gene rate calibration from initial counts
USE_ASSEMBLY_CORE   = True         # v15.4 [6/6]: re-enabled via SAFE additive blend (ASM_BLEND_WEIGHT=0.1) + safety cap. Was disabled v14.1 (override regressions). Only ~2/24 complexes wire — low impact; revert by setting False.
USE_KO_AUGMENTATION = True         # NEW v13.9 module 7: random species-zero during training, teaches KO response
KO_AUG_PROB         = 0.5          # v14.9: bumped 0.3 → 0.5 — more KO exposure, sharpens mid-rank essentiality scores
GIBBS_DG_THRESHOLD_KJ = 10.0       # NEW v14 day 1: ΔG° (kJ/mol) below which a reaction is forced irreversible-forward
USE_ATP_LEDGER      = True         # NEW v14 day 3: soft penalty when net ATP production rate falls below maintenance floor
ATP_SPECIES_NAME    = "M_atp_c"    # which species to track for the energy ledger
ATP_MAINTENANCE_RATE = 4.0e5       # ATP molecules per second per cell — NGAM floor (~5 mmol/g/hr × Syn3A mass)
LAMBDA_ATP          = 0.01         # auxiliary loss weight for ATP-deficit penalty
USE_SIGMA_ANCHOR    = False        # v15.4: DROPPED — post-hoc recalibration showed alpha=0.925 (sigma-head already calibrated by NLL). The anchor was solving a non-problem; the metric it "targeted" measured mode-collapse, not sigma.
LAMBDA_SIGMA_ANCHOR = 0.15         # v15.1: bumped 0.05 → 0.15 — v15.0 σ was over-confident by 10× (log10 pred/true = -0.93), σ-anchor needs more weight
LAMBDA_TRAJ_VAR     = 0.0          # v14.8: disabled — didn't budge mode collapse in v14.7, dropped to avoid masking
SAMPLE_NOISE_SCALE  = 0.0          # v14.8: DISABLED — was the main cause of v14.7's rollout R² crash from 0.43 → -0.05
                                   #         (model converged to "predict no change" under noisy training inputs)
# v15.3 step 5: stochastic rollout for distributional eval.  Post-hoc
# σ recalibration fits α[s] s.t. α·predicted_σ matches empirical residual
# std; rollout then samples from N(pred, α·σ) instead of using pred.
# Makes the σ-calibration / W₂ / paper-validation distributional metrics
# read against a fair (noise-aware) comparison.  Per-trajectory R² is
# typically slightly lower for stochastic rollout — that's expected.
USE_STOCHASTIC_ROLLOUT_EVAL = True

# v15.6: Breuer-as-training-signal — pairwise hinge loss on KO impact magnitudes.
# Trains the model so essential-gene KOs produce LARGER trajectory deviation than
# nonessential-gene KOs. This is the "past the simulator" move — Breuer 2019 is
# real-experiment data the simulator was never validated against per-gene. We've
# been scoring against it (KO MCC at eval); now we LEARN from it.
USE_BREUER_LOSS         = True
# v15.7.1: bumped from v15.6 defaults (which gave breuer EMA pinned at margin
# floor — zero gradient flow). Diagnostic showed impact range 2e-4 to 3e-3,
# so the old margin 1e-4 was satisfied trivially at random init.
LAMBDA_BREUER           = 0.20    # was 0.05 — bigger so it competes with main NLL (~-1.5)
BREUER_LOSS_START_FRAC  = 0.5     # activate after this fraction of STEPS (latter half)
BREUER_LOSS_EVERY       = 5       # run every N training steps (amortise the extra rollouts)
BREUER_HORIZON          = 15      # was 10 — give KO effect more time to develop
BREUER_KO_PER_LABEL     = 3       # was 2 — more samples per call for less noise
BREUER_MARGIN           = 5e-3    # was 1e-4 — above typical-impact floor so hinge bites

# v15.7 Move A: training-signal upgrades (close the measurement→training loop)
USE_MULTITASK_LOSS  = True        # A1: aux MSE on derived totals (lipid/ATP/ribosome/ori/ter/protein)
LAMBDA_MULTITASK    = 0.10        # weight on the derived-quantity aux loss
USE_PER_CLASS_LOSS_W = True       # A2: weight main loss by species variance (align with honest top-K metric)
PER_CLASS_W_FLOOR   = 0.2         # min weight for low-variance species (so they aren't ignored)
PER_CLASS_W_CEIL    = 5.0         # max weight for high-variance species
USE_FW_TRAIN_LOSS   = False       # A5: Friedlin-Wentzell action as a (small) training loss — OFF by default (needs drift/diff buffers)
LAMBDA_FW           = 0.01

# v15.7 Move C: self-supervised pretraining (forces the graph layer to be used)
USE_MASKED_PRETRAIN  = True       # C1: BERT-style masked-species prediction before main training
MASK_PRETRAIN_STEPS  = 300        # pretraining steps (short — just to warm the graph)
MASK_FRAC            = 0.20        # fraction of species masked per example
MASK_PRETRAIN_LR     = 5e-4       # pretraining LR (higher than main; quick warmup)

# v15.7 Move D: better use of discarded data (diagnostic + constraint extraction)
USE_DATA_RICHNESS_REPORT = True   # D: surface what the active-species filter discards

# v15.1: refinement-pass training in the last 10% of steps.  Attacks the
# rollout-vs-1step gap (v15.0: 1-step R²=0.814, rollout R²=0.586, 0.22 lost to
# compounding error).  Mechanism: after computing pred at step k, do an EXTRA
# forward pass on pred.detach() and check that the result lands at true(t+k+2).
# Trains the model to be robust to its own (imperfect) outputs as input.
USE_REFINEMENT      = True
LAMBDA_REFINE       = 0.15         # weight on refinement MSE term
REFINE_START_FRAC   = 0.9          # activate refinement at this fraction of STEPS (0.9 = last 10%)
USE_STOCHASTIC_HEAD = True         # NEW v9: per-species log_sigma + NLL loss
USE_TORCH_COMPILE   = True         # v13.4: was False — enable torch.compile by default (falls back to eager on failure)
RESUME_FROM_CHECKPOINT  = False    # v13.6: load existing cell_emulator_v13.pt if compatible
SKIP_TRAINING_IF_LOADED = False    # v13.6: if RESUME loaded weights, skip training (eval-only run)
CHECKPOINT_EVERY    = 250          # v13.7: write a rolling mid-training checkpoint every N steps (0 to disable)
PINN_RATE_CLIP      = 6.0          # NEW v9: log-space rate clip (prevents expm1 blow-up)
VAR_R2_TOP_K        = 200          # NEW v9: top-K species (by variance) for the honest R²
KO_N_STEPS          = 60           # v15.4: 60 steps × 60 s stride = 60 min biological
KO_BATCH_SIZE       = 32           # NEW v9: parallel knockouts per batch
# v15.9: static essentiality classifier (keyword + pipeline features) + synthetic
# lethality screen.  The LGNN's dynamic KO sweep tops out at MCC ~0.06 because
# essentiality is a STATIC property (gene annotation + network position), not a
# dynamic one.  These two diagnostics test that directly.
USE_ESSENTIALITY_XGB = True        # static-feature essentiality classifier + 5-fold MCC
USE_SYNTHETIC_LETHALITY = True     # double-KO super-additivity screen
SL_MAX_PAIRS        = 600          # cap on candidate gene pairs (compute bound)
SL_HORIZON          = 30           # rollout steps per double-KO (30 = 30 min biological)
ESS_N_FOLDS         = 5            # stratified CV folds for the MCC test
BREUER_PATH         = "memory_bank/data/syn3a_essentiality_breuer2019.csv"
SEED                = 0
SAVE_DIR            = "/content/drive/MyDrive"
GENE_TABLE_PATH     = "memory_bank/data/syn3a_gene_table.csv"
SBML_PATH           = "Syn3A_updated.xml"
KINETICS_PATH       = "kinetic_params.xlsx"             # NEW v7
INITIAL_CONC_PATH   = "initial_concentrations.xlsx"     # NEW v7
COMPLEXES_PATH      = "complex_formation.xlsx"          # NEW v7
PROTEIN_METABOLITES_PATH = "protein_metabolites.xlsx"   # NEW v11: P↔M regulatory binding
GIBBS_PATH          = "gibbs.csv"                       # NEW v11: per-reaction ΔG°
LARGESUBUNIT_PATH   = "LargeSubunit.xlsx"               # NEW v11: 50S ribosome assembly

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SEED RULES  -  the rules we KNOW.  Edit / extend this block freely.      ║
# ║  Everything here is still validated on held-out data before enforcement.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
SEED_MONOTONE_UP_CHANNELS = {"RPM", "PM", "DM"}
SEED_NONNEG               = True


# ── species-name parsing ──────────────────────────────────────────────────────

_CD_PREFIXES = sorted([
    "RB_pe", "RB_cp", "RB_p", "RP_f", "P_TC",
    "RPM", "R_d", "C_P",
    "RB", "RP", "DM", "PM", "DT",
    "G", "R", "P", "S", "D",
], key=len, reverse=True)

CHAN_NAMES = ["G", "R", "R_d", "RP", "RP_f", "RB", "RB_p", "RB_pe", "RB_cp",
              "P", "C_P", "P_TC", "S", "D", "DT", "RPM", "PM", "DM"]
N_CHAN = len(CHAN_NAMES)
CHAN_IDX = {c: i for i, c in enumerate(CHAN_NAMES)}

GTYPE_PROTEIN, GTYPE_TRNA, GTYPE_RRNA, GTYPE_OTHER, GTYPE_GLOBAL = 0, 1, 2, 3, 4
N_GTYPES = 5


def parse_species(name):
    """'RPM_0001' -> ('RPM', '0001');  'M_atp_c' -> (None, 'M_atp_c')."""
    for p in _CD_PREFIXES:
        if name.startswith(p + "_"):
            return p, name[len(p) + 1:]
    return None, name


def _locus_key(locus):
    """'0412_C1' -> '0412'.  Strips chromosome-copy / variant suffixes."""
    m = re.match(r"\d+", locus)
    return m.group(0) if m else locus


def load_breuer_essentiality(path):
    """Load Breuer 2019 essentiality labels — {locus_num: 'Essential'|'Quasiessential'|'Nonessential'}."""
    if not HAS_PANDAS:
        print("[breuer] pandas unavailable - skipping knockout sweep")
        return {}
    try:
        df = pd.read_csv(path)
        out = {}
        for _, row in df.iterrows():
            tag = str(row.get("locus_tag", ""))
            ess = str(row.get("essentiality", ""))
            if "_" in tag and ess in ("Essential", "Quasiessential", "Nonessential"):
                out[tag.split("_")[1]] = ess
        print(f"[breuer] loaded {len(out)} essentiality labels")
        return out
    except Exception as e:
        print(f"[breuer] load failed ({e}) - skipping knockout sweep")
        return {}


def load_gene_types(csv_path):
    """Return {locus_num_str: int_type_code} from syn3a_gene_table.csv."""
    if not HAS_PANDAS:
        print("[gene_types] pandas unavailable - GTYPE_OTHER for all genes")
        return {}
    try:
        df = pd.read_csv(csv_path)
        fmap = {"CDS": GTYPE_PROTEIN, "tRNA": GTYPE_TRNA, "rRNA": GTYPE_RRNA}
        out = {}
        for _, row in df.iterrows():
            tag = str(row.get("locus_tag", ""))
            ft  = str(row.get("feature_type", ""))
            if "_" in tag:
                out[tag.split("_")[1]] = fmap.get(ft, GTYPE_OTHER)
        print(f"[gene_types] loaded {len(out)} locus entries")
        return out
    except Exception as e:
        print(f"[gene_types] load failed ({e}) - GTYPE_OTHER for all genes")
        return {}


def build_gene_index(species_names, gene_type_map):
    """Per-species gene-type labelling. Returns (species_type_ids, locus_list)."""
    locus_to_idx, locus_list = {}, []
    for name in species_names:
        prefix, locus = parse_species(name)
        if prefix is not None:
            key = _locus_key(locus)
            if key not in locus_to_idx:
                locus_to_idx[key] = len(locus_list)
                locus_list.append(key)

    gene_type_ids = np.array(
        [gene_type_map.get(loc, GTYPE_OTHER) for loc in locus_list], dtype=np.int32)
    S = len(species_names)
    species_type_ids = np.full(S, GTYPE_GLOBAL, dtype=np.int32)
    n_global = 0
    for i, name in enumerate(species_names):
        prefix, locus = parse_species(name)
        if prefix is not None:
            species_type_ids[i] = gene_type_ids[locus_to_idx[_locus_key(locus)]]
        else:
            n_global += 1
    typed = int((species_type_ids != GTYPE_OTHER).sum()) - n_global
    print(f"[gene_index] {len(locus_list)} genes  {n_global} global species  "
          f"{typed} species type-labelled  "
          f"example '{species_names[0]}' -> {parse_species(species_names[0])}")
    return species_type_ids, locus_list


# ── SBML parsing ──────────────────────────────────────────────────────────────

def _formula_atoms(formula):
    atoms = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        atoms[el] = atoms.get(el, 0) + (int(n) if n else 1)
    return atoms


def parse_sbml(path):
    """Parse SBML L3 (+fbc). Returns dict or None on failure."""
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        print(f"[sbml] read failed ({e}) - SBML rules skipped")
        return None
    def local(tag): return tag.rsplit("}", 1)[-1]
    def attr(e, n):
        for k, v in e.attrib.items():
            if local(k) == n:
                return v
        return None
    species, reactions = {}, []
    for elem in root.iter():
        ln = local(elem.tag)
        if ln == "species":
            sid = attr(elem, "id")
            f = attr(elem, "chemicalFormula") or ""
            species[sid] = {"formula": f, "atoms": _formula_atoms(f)}
        elif ln == "reaction":
            rxn = {"id": attr(elem, "id"),
                   "reversible": attr(elem, "reversible") == "true",
                   "reactants": [], "products": []}
            for child in elem:
                lc = local(child.tag)
                bucket = ("reactants" if lc == "listOfReactants"
                          else "products" if lc == "listOfProducts" else None)
                if bucket is None:
                    continue
                for sr in child:
                    if local(sr.tag) == "speciesReference":
                        rxn[bucket].append((attr(sr, "species"),
                                            float(attr(sr, "stoichiometry") or 1.0)))
            reactions.append(rxn)
    print(f"[sbml] parsed {len(species)} species, {len(reactions)} reactions "
          f"({sum(r['reversible'] for r in reactions)} reversible)")
    return {"species": species, "reactions": reactions}


def sbml_monotone_candidates(sbml, species_names):
    if sbml is None:
        return set(), set()
    produced, consumed = set(), set()
    for rxn in sbml["reactions"]:
        for sid, _ in rxn["reactants"]:
            consumed.add(sid)
            if rxn["reversible"]: produced.add(sid)
        for sid, _ in rxn["products"]:
            produced.add(sid)
            if rxn["reversible"]: consumed.add(sid)
    name_to_idx = {n: i for i, n in enumerate(species_names)}
    up   = {name_to_idx[s] for s in (produced - consumed) if s in name_to_idx}
    down = {name_to_idx[s] for s in (consumed - produced) if s in name_to_idx}
    print(f"[sbml] structural monotone candidates: {len(up)} up, {len(down)} down")
    return up, down


def element_balances(sbml, species_names, raw_counts):
    if sbml is None:
        return []
    cols, atoms_list = [], []
    for i, name in enumerate(species_names):
        if name in sbml["species"] and sbml["species"][name]["atoms"]:
            cols.append(i)
            atoms_list.append(sbml["species"][name]["atoms"])
    if not cols:
        print("[sbml] no SBML metabolites matched trajectory species")
        return []
    elements = sorted({e for a in atoms_list for e in a})
    E = np.array([[a.get(e, 0) for e in elements] for a in atoms_list], dtype=np.float64)
    counts = raw_counts[:, :, cols].astype(np.float64)
    Q = counts @ E
    out = []
    for j, el in enumerate(elements):
        q = Q[:, :, j]
        mean = q.mean(axis=1)
        rng = q.max(axis=1) - q.min(axis=1)
        drift = float(np.mean(rng / np.clip(np.abs(mean), 1e-9, None)))
        out.append({"element": el, "n_species": len(cols),
                    "drift_frac": drift, "conserved": drift < CONSERVE_DRIFT})
    print(f"[sbml] element balances over {len(cols)} metabolites: "
          + ", ".join(f"{d['element']} {d['drift_frac']*100:.1f}%" for d in out))
    return out


# ── NEW v7: xlsx parsers for the rest of the input files ─────────────────────

def parse_kinetics(path):
    """Parse kinetic_params.xlsx -> {enzymes, n_params, subsystems, params} or None.

    v13.9: extended to extract the full per-reaction bi-bi parameter table.
      params[rxn_id] = {kcat_fwd, kcat_rev, km: {species: value},
                        enzyme, gpr, subsys}
    Backward-compat: `enzymes` and `n_params` keys preserved for graph code.
    """
    if not HAS_PANDAS:
        return None
    try:
        sheets = ["Central", "Nucleotide", "Lipid", "Cofactor", "Transport"]
        enzymes, n_params, subsys = {}, 0, {}
        params = {}
        for sn in sheets:
            try:
                df = pd.read_excel(path, sheet_name=sn)
            except Exception:
                continue
            subsys[sn] = len(df)
            n_params += len(df)
            for _, row in df.iterrows():
                rxn = str(row.get("Reaction Name", "")).strip()
                if not rxn or rxn == "nan":
                    continue
                pt = str(row.get("Parameter Type", "")).strip()
                val = row.get("Value")
                sp = row.get("Related Species")
                p = params.setdefault(rxn, {
                    "kcat_fwd": None, "kcat_rev": None, "km": {},
                    "enzyme": None, "gpr": None, "subsys": sn})

                if pt == "Substrate Catalytic Rate Constant":
                    try: p["kcat_fwd"] = float(val)
                    except (ValueError, TypeError): pass
                elif pt == "Product Catalytic Rate Constant":
                    try: p["kcat_rev"] = float(val)
                    except (ValueError, TypeError): pass
                elif pt == "Michaelis Menten Constant" and sp is not None:
                    sp_str = str(sp).strip()
                    if sp_str and sp_str != "nan":
                        try: p["km"][sp_str] = float(val)
                        except (ValueError, TypeError): pass
                elif pt == "Eff Enzyme Count":
                    val_str = str(val).strip() if val is not None else ""
                    if val_str and val_str != "nan":
                        p["enzyme"] = val_str
                        enzymes[rxn] = val_str
                elif pt == "GPR rule":
                    p["gpr"] = str(val) if val is not None else None

        with_kcat = sum(1 for p in params.values() if p["kcat_fwd"] is not None)
        with_enz  = sum(1 for p in params.values() if p["enzyme"] is not None)
        with_km   = sum(1 for p in params.values() if p["km"])
        print(f"[kinetics] {n_params} parameter rows across {len(subsys)} subsystems, "
              f"{len(enzymes)} reaction->enzyme mappings")
        print(f"[kinetics] per-reaction coverage: {with_kcat}/{len(params)} k_cat_fwd, "
              f"{with_enz}/{len(params)} enzymes, {with_km}/{len(params)} K_m sets")
        return {"enzymes": enzymes, "n_params": n_params,
                "subsystems": subsys, "params": params}
    except Exception as e:
        print(f"[kinetics] parse failed ({e}) - skipping kinetics")
        return None


def parse_initial_concentrations(path):
    """Parse initial_concentrations.xlsx -> {proteins, mRNAs, metabolites, medium}.

    Handles both the ComplexFormation version (has 'mRNA Count' + 'Protein
    Metabolites' sheets) and the 4DWCM version (lacks those — they were
    refactored into separate files).  Each sheet parsed independently so
    one missing sheet doesn't kill the rest.
    """
    if not HAS_PANDAS:
        return None
    proteins, mRNAs, metabolites, medium = {}, {}, {}, {}

    # Proteins (Comparative Proteomics).  4DWCM uses "Exp. Ptn Cnt" instead of
    # the ComplexFormation version's "Sim. Initial Ptn Cnt".
    try:
        df = pd.read_excel(path, sheet_name="Comparative Proteomics")
        cnt_col = next((c for c in ("Sim. Initial Ptn Cnt", "Exp. Ptn Cnt")
                        if c in df.columns), None)
        if cnt_col:
            df["_cnt"] = pd.to_numeric(df[cnt_col], errors="coerce")
            for _, r in df.iterrows():
                tag = str(r.get("Locus Tag", ""))
                if tag.startswith("JCVISYN3A_") and pd.notna(r["_cnt"]):
                    proteins[tag] = float(r["_cnt"])
    except Exception:
        pass

    try:
        df = pd.read_excel(path, sheet_name="mRNA Count")
        for _, r in df.iterrows():
            tag = str(r.get("LocusTag", ""))
            tot = r.get("total")
            if tag.startswith("JCVISYN3A_") and pd.notna(tot):
                mRNAs[tag] = float(tot)
    except Exception:
        pass   # 4DWCM version doesn't have this sheet — silently skip

    try:
        df = pd.read_excel(path, sheet_name="Intracellular Metabolites")
        metabolites = {f"M_{r['Met ID']}": float(r["Init Conc (mM)"])
                       for _, r in df.iterrows()
                       if pd.notna(r.get("Met ID")) and pd.notna(r.get("Init Conc (mM)"))}
    except Exception:
        pass

    try:
        df = pd.read_excel(path, sheet_name="Simulation Medium")
        medium = {f"M_{r['Met ID']}": float(r["Conc (mM)"])
                  for _, r in df.iterrows()
                  if pd.notna(r.get("Met ID")) and pd.notna(r.get("Conc (mM)"))}
    except Exception:
        pass

    if not (proteins or mRNAs or metabolites or medium):
        print(f"[initial_conc] no recognised sheets in {path} — skipping")
        return None
    print(f"[initial_conc] {len(proteins)} proteins, {len(mRNAs)} mRNAs, "
          f"{len(metabolites)} intracellular metabolites, "
          f"{len(medium)} medium components")
    return {"proteins": proteins, "mRNAs": mRNAs,
            "metabolites": metabolites, "medium": medium}


def parse_complex_formation(path):
    """Parse complex_formation.xlsx -> {complexes, predefined} or None."""
    if not HAS_PANDAS:
        return None
    try:
        df = pd.read_excel(path, sheet_name="Complexes")
        complexes = []
        for _, r in df.iterrows():
            name = str(r.get("Name", ""))
            genes = str(r.get("Genes Products", ""))
            stois = str(r.get("Stoichiometries", ""))
            if not name or genes in ("nan", ""):
                continue
            try:
                gids = [g.strip() for g in genes.split(";")]
                svs  = [int(s.strip()) for s in stois.split(";")]
                if len(gids) != len(svs):
                    continue
                subunits = list(zip(gids, svs))
            except Exception:
                continue
            ic = r.get("Init. Count")
            complexes.append({"name": name, "subunits": subunits,
                              "init_count": float(ic) if pd.notna(ic) else None})

        df = pd.read_excel(path, sheet_name="Predefined Complexes")
        predefined = {str(r["Name"]): float(r["Init. Count"])
                      for _, r in df.iterrows()
                      if pd.notna(r.get("Name")) and pd.notna(r.get("Init. Count"))}
        print(f"[complexes] {len(complexes)} assembly rules, "
              f"{len(predefined)} predefined complexes")
        return {"complexes": complexes, "predefined": predefined}
    except Exception as e:
        print(f"[complexes] parse failed ({e}) - skipping complex rules")
        return None


# ── NEW v11: parsers for additional 4DWCM data sources ────────────────────────

def parse_protein_metabolites(path):
    """Parse protein_metabolites.xlsx → list of {protein, metabolites, reactions}.

    Each row: a protein and the list of metabolites it regulates (binds /
    catalyses).  Adds P↔M edges in the graph.  Also tries the 'Protein
    Metabolites' sheet of initial_concentrations.xlsx as a fallback source.
    """
    if not HAS_PANDAS:
        return []
    try:
        # try standalone file first, then the sheet in initial_concentrations.xlsx
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_excel(INITIAL_CONC_PATH, sheet_name="Protein Metabolites")
        pairs = []
        for _, row in df.iterrows():
            prot = str(row.get("Protein", "")).strip()
            mets_s = str(row.get("Metabolite IDs", "")).strip()
            if not prot or prot in ("nan", "") or mets_s in ("nan", ""):
                continue
            mets = [m.strip() for m in mets_s.split(",") if m.strip()]
            if mets:
                pairs.append({"protein": prot, "metabolites": mets,
                              "reactions": str(row.get("Reactions", ""))})
        print(f"[prot_metab] {len(pairs)} protein-metabolite binding records")
        return pairs
    except Exception as e:
        print(f"[prot_metab] load failed ({e}) - skipping P-M edges")
        return []


def parse_gibbs(path):
    """Parse gibbs.csv → {reaction_id: ΔG° (float)}.

    Tries common column names (reaction/rxn/id for the key; dG/gibbs/delta
    for the value).  Defensive — returns empty dict if file missing or
    format unexpected.
    """
    if not HAS_PANDAS:
        return {}
    # v15.7.1: gibbs.csv is user-produced (eQuilibrator) and easy to lose between
    # Colab sessions.  If not at the primary path, fall back to a Drive-wide glob
    # — finding it anywhere on Drive is way better than silently losing 194
    # ΔG° values (which collapses MetabolismCore from 115 → 92 wired reactions
    # and disables the entire TUR diagnostic).
    import glob as _glob, os as _os
    if not _os.path.exists(path):
        for cand_glob in ("/content/drive/MyDrive/**/gibbs.csv",
                          "/content/**/gibbs.csv",
                          "/content/cell/**/gibbs.csv"):
            hits = _glob.glob(cand_glob, recursive=True)
            if hits:
                path = hits[0]
                print(f"[gibbs] primary path missing — recovered from {path}")
                break
        else:
            print(f"[gibbs] ⚠ NOT FOUND at {path} or anywhere on Drive — "
                  f"ΔG° clamps + TUR diagnostic disabled (MetabolismCore will be "
                  f"weaker; expect 20-30% fewer wired reactions)")
    try:
        df = pd.read_csv(path)
        rxn_col = next((c for c in df.columns
                        if any(k in c.lower() for k in ("reaction", "rxn", "id"))),
                       df.columns[0])
        dg_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ("dg", "gibbs", "delta", "energy"))
                       and pd.api.types.is_numeric_dtype(df[c])),
                      None)
        if dg_col is None:
            for c in df.columns:
                if c != rxn_col and pd.api.types.is_numeric_dtype(df[c]):
                    dg_col = c
                    break
        if dg_col is None:
            print(f"[gibbs] couldn't find ΔG° numeric column in {list(df.columns)}")
            return {}
        out = {}
        for _, row in df.iterrows():
            r = str(row[rxn_col]).strip()
            try:
                dg = float(row[dg_col])
                if r and not np.isnan(dg):
                    out[r] = dg
            except (TypeError, ValueError):
                continue
        if out:
            vals = list(out.values())
            print(f"[gibbs] loaded ΔG° for {len(out)} reactions  "
                  f"(median {np.median(vals):+.1f}, range "
                  f"[{min(vals):+.1f}, {max(vals):+.1f}])")
        return out
    except Exception as e:
        print(f"[gibbs] load failed ({e}) - skipping thermodynamic priors")
        return {}


def parse_largesubunit(path):
    """Parse LargeSubunit.xlsx → list of (substrate, intermediate, product) tuples
    for 50S ribosome assembly.  Adds substrate→product and intermediate→product
    edges in the graph (the assembly path).

    4DWCM layout: two sheets ('parameters' with Protein/Rate, and 'reactions'
    with substrate/intermediate/product).  Try 'reactions' first; fall back
    to the first sheet for legacy single-sheet variants.
    """
    if not HAS_PANDAS:
        return []
    try:
        try:
            df = pd.read_excel(path, sheet_name="reactions")
        except Exception:
            df = pd.read_excel(path)               # first sheet for legacy layouts
        tuples = []
        for _, row in df.iterrows():
            sub   = str(row.get("substrate", "")).strip()
            inter = str(row.get("intermediate", "")).strip()
            prod  = str(row.get("product", "")).strip()
            if (sub and inter and prod
                    and all(v not in ("nan", "") for v in (sub, inter, prod))):
                tuples.append((sub, inter, prod))
        print(f"[largesubunit] {len(tuples)} 50S ribosome assembly steps")
        return tuples
    except Exception as e:
        print(f"[largesubunit] load failed ({e}) - skipping 50S assembly edges")
        return []


# ── NEW v7: KnownRules (input-file facts) ────────────────────────────────────

class KnownRules:
    """Facts from input files (deterministic, mechanism-derived).

    Distinct from DiscoveredPatterns: these are NOT empirical regularities
    mined from data, they are explicit structural facts read from model files.
    """

    def __init__(self, sbml=None, kinetics=None, initial=None, complexes=None,
                 protein_metabolites=None, gibbs=None, largesubunit=None):
        self.sbml = sbml
        self.kinetics = kinetics
        self.initial = initial
        self.complexes = complexes
        # v11 additions:
        self.protein_metabolites = protein_metabolites or []
        self.gibbs = gibbs or {}
        self.largesubunit = largesubunit or []

    def has_anything(self):
        return any(v is not None for v in
                   (self.sbml, self.kinetics, self.initial, self.complexes)) or \
               bool(self.protein_metabolites or self.gibbs or self.largesubunit)

    def summary(self):
        s = ["KnownRules (from input files):"]
        if self.sbml is not None:
            nr = len(self.sbml["reactions"])
            nrv = sum(r["reversible"] for r in self.sbml["reactions"])
            ns = sum(1 for v in self.sbml["species"].values() if v["atoms"])
            s.append(f"    SBML reactions          : {nr}  "
                     f"({nrv} reversible, {nr-nrv} irreversible)")
            s.append(f"    SBML species (w/ formula): {ns}")
        if self.complexes is not None:
            s.append(f"    Complex assembly rules  : {len(self.complexes['complexes'])}")
            s.append(f"    Predefined complexes    : {len(self.complexes['predefined'])}")
        if self.initial is not None:
            s.append(f"    Initial protein counts  : {len(self.initial['proteins'])}")
            s.append(f"    Initial mRNA counts     : {len(self.initial['mRNAs'])}")
            s.append(f"    Intracellular metab. init: {len(self.initial['metabolites'])}")
            s.append(f"    Simulation medium       : {len(self.initial['medium'])}")
        if self.kinetics is not None:
            s.append(f"    Kinetic parameter rows  : {self.kinetics['n_params']}"
                     f"  ({len(self.kinetics['enzymes'])} reaction->enzyme links)")
        # v11: new sources
        if self.protein_metabolites:
            n_edges = sum(len(r["metabolites"]) for r in self.protein_metabolites)
            s.append(f"    Protein↔metabolite reg. : {len(self.protein_metabolites)} proteins, "
                     f"{n_edges} regulatory bindings")
        if self.gibbs:
            vals = list(self.gibbs.values())
            n_neg = sum(1 for v in vals if v < 0)
            s.append(f"    Gibbs ΔG° (reactions)   : {len(self.gibbs)}  "
                     f"({n_neg} exergonic, {len(vals)-n_neg} endergonic)")
        if self.largesubunit:
            s.append(f"    50S ribosome assembly   : {len(self.largesubunit)} stages")
        if not self.has_anything():
            s.append("    (no input files staged)")
        return "\n".join(s)


# ── NEW v7: multi-dimensional pattern lenses ─────────────────────────────────

def lens_monotone(d_tr, d_va, mono_eps=MONO_EPS):
    """1D: per-species monotonicity. Returns held-out compliance + empirical candidates."""
    n_steps_tr = d_tr.shape[0] * d_tr.shape[1]
    n_steps_va = d_va.shape[0] * d_va.shape[1]
    dec_tr = (d_tr < -mono_eps).sum(axis=(0, 1)) / n_steps_tr
    inc_tr = (d_tr >  mono_eps).sum(axis=(0, 1)) / n_steps_tr
    dec_va = (d_va < -mono_eps).sum(axis=(0, 1)) / n_steps_va
    inc_va = (d_va >  mono_eps).sum(axis=(0, 1)) / n_steps_va
    ok_up_va, ok_down_va = 1.0 - dec_va, 1.0 - inc_va
    emp_up   = set(np.where(dec_tr < (1.0 - RULE_COMPLIANCE))[0].tolist())
    emp_down = set(np.where(inc_tr < (1.0 - RULE_COMPLIANCE))[0].tolist())
    return ok_up_va, ok_down_va, emp_up, emp_down


def lens_bounds(tr, va, slack=0.1, lo=None, span=None,
                raw_counts_tr=None, count_slack=COUNT_BOUND_SLACK):
    """1D: per-species bounds. Validated if ALL held-out points are within [lo,hi].

    If lo, span and raw_counts_tr are provided, additionally tighten the HIGH
    side so the un-normalised count cannot exceed train_max * (1 + count_slack).
    Without this, the 0.1 normalised slack expm1's to a 3-4x count-space
    inflation (e.g. 110k -> 400k+) — which is what was producing the 2.8M / 3.5M
    overshoot we saw in v6/v7.
    """
    S = tr.shape[-1]
    tr_flat = tr.reshape(-1, S)
    va_flat = va.reshape(-1, S)
    tr_min  = tr_flat.min(axis=0)
    tr_max  = tr_flat.max(axis=0)
    va_min  = va_flat.min(axis=0)
    va_max  = va_flat.max(axis=0)
    # Base: train range + normalised slack, but never less generous than val
    # (so bounds always *fit* the val data — no species ever falls back to
    # the loose global CLAMP).  The v7.0 'validation gate' fallback was
    # punitive: species whose val barely exceeded the tight cap got bumped
    # to CLAMP_HI=1.2 which un-normalises to ~9M counts; net effect: count
    # peaks went UP, not down (3.5M -> 4.8M on the real run).
    lo_cand = np.minimum(tr_min - slack, va_min - 0.01)
    hi_cand = np.maximum(tr_max + slack, va_max + 0.01)
    if lo is not None and span is not None and raw_counts_tr is not None:
        count_max = raw_counts_tr.max(axis=tuple(range(raw_counts_tr.ndim - 1)))
        count_cap = count_max * (1.0 + count_slack)
        sl_cap    = np.sign(count_cap) * np.log1p(np.abs(count_cap))
        hi_count_cap = (sl_cap - lo) / np.where(span > 1e-9, span, 1.0)
        # Take the tighter of (normalised hi_cand, count-space cap) but
        # never below val_max + 0.01 - keeps bounds val-fitting always.
        capped  = np.minimum(hi_cand, hi_count_cap)
        hi_cand = np.maximum(capped, va_max + 0.01)
        n_tight = int((hi_cand < tr_max + slack).sum())
        print(f"[lens_bounds] count-space cap tightened {n_tight}/{S} species "
              f"({count_slack*100:.0f}% headroom over train max, val-fitting)")
    # By construction lo_cand <= val_min and hi_cand >= val_max — bound_ok
    # is True for every species, no fallback to loose CLAMP.
    ok = np.ones(S, dtype=bool)
    return lo_cand, hi_cand, ok


def lens_pairwise(d_tr, d_va, top_k=PAIRWISE_TOP_K, threshold=PAIRWISE_THRESHOLD):
    """2D: pairwise correlation on deltas. Returns top |r| pairs.

    Returns list of (i, j, corr_train, corr_val).  Pairs where either species
    has near-zero variance are skipped (correlation undefined).
    """
    S = d_tr.shape[-1]
    Xtr = d_tr.reshape(-1, S).astype(np.float32)
    Xva = d_va.reshape(-1, S).astype(np.float32)
    sd_tr = Xtr.std(axis=0)
    sd_va = Xva.std(axis=0)
    valid = (sd_tr > 1e-6) & (sd_va > 1e-6)
    if valid.sum() < 2:
        return []
    Ztr = (Xtr - Xtr.mean(axis=0)) / (sd_tr + 1e-9)
    Zva = (Xva - Xva.mean(axis=0)) / (sd_va + 1e-9)
    Ctr = (Ztr.T @ Ztr) / Xtr.shape[0]
    np.fill_diagonal(Ctr, 0.0)
    # restrict to valid pairs
    Ctr_v = Ctr.copy()
    Ctr_v[~valid, :] = 0.0
    Ctr_v[:, ~valid] = 0.0
    abs_C = np.abs(Ctr_v)
    n_pick = min(2 * top_k * 4, abs_C.size)
    flat_idx = np.argpartition(abs_C.ravel(), -n_pick)[-n_pick:]
    order = np.argsort(-abs_C.ravel()[flat_idx])
    seen, pairs = set(), []
    for fi in flat_idx[order]:
        i, j = int(fi // S), int(fi % S)
        if i >= j or (i, j) in seen:
            continue
        seen.add((i, j))
        c_tr = float(Ctr[i, j])
        if abs(c_tr) < threshold:
            break
        c_va = float((Zva[:, i] * Zva[:, j]).mean())
        pairs.append((i, j, c_tr, c_va))
        if len(pairs) >= top_k:
            break
    return pairs


def lens_conservation(tr_counts, va_counts, n_candidates=CONSERVATION_K):
    """Low-d: SVD of standardized counts; smallest singular directions = candidates.

    Returns list of dicts with {singular_value, std_train, std_val, top_species_*}.
    """
    n_tr, T, S = tr_counts.shape
    X_tr = tr_counts.reshape(-1, S).astype(np.float32)
    X_va = va_counts.reshape(-1, S).astype(np.float32)
    sd = X_tr.std(axis=0) + 1e-9
    mu = X_tr.mean(axis=0)
    Ztr = (X_tr - mu) / sd
    Zva = (X_va - mu) / sd
    try:
        _, s, Vh = np.linalg.svd(Ztr, full_matrices=False)
    except Exception as e:
        print(f"[lens] conservation SVD failed ({e})")
        return []
    smallest = np.argsort(s)[:n_candidates]
    out = []
    for idx in smallest:
        v = Vh[idx]
        proj_tr = Ztr @ v
        proj_va = Zva @ v
        top = np.argsort(-np.abs(v))[:3]
        out.append({
            "singular_value": float(s[idx]),
            "std_train":      float(proj_tr.std()),
            "std_val":        float(proj_va.std()),
            "top_species_idx":    [int(j) for j in top],
            "top_species_weight": [float(v[j]) for j in top],
        })
    return out


def lens_periodicity(tr, top_k=PERIODICITY_TOP_K):
    """Time: FFT power spectrum per species. Returns top species by peak/total ratio."""
    n_tr, T, S = tr.shape
    x = tr - tr.mean(axis=1, keepdims=True)
    fft_vals = np.fft.rfft(x, axis=1)
    power = (np.abs(fft_vals) ** 2).mean(axis=0)    # (T//2+1, S)
    if power.shape[0] <= 1:
        return []
    peak_idx = power[1:].argmax(axis=0) + 1
    peak_power = power[peak_idx, np.arange(S)]
    total = power[1:].sum(axis=0)
    rel = peak_power / (total + 1e-12)
    order = np.argsort(-rel)[:top_k]
    return [(int(i), int(peak_idx[i]), float(rel[i])) for i in order]


def lens_gene_chain(tr_active, species_names, max_lag=GENE_LAG_MAX):
    """Chain: per-gene central-dogma cross-channel lag estimation.

    For each consecutive pair in (G, RP, R, RB, P), find the lag (within
    [-max_lag, +max_lag]) that maximises cross-correlation, averaged across
    genes that have both channels.
    """
    n_tr, T, S = tr_active.shape
    chain_pairs = [("G", "RP"), ("RP", "R"), ("R", "RB"), ("RB", "P")]
    gene_chans = {}
    for i, name in enumerate(species_names):
        pre, loc = parse_species(name)
        if pre in {"G", "R", "RP", "RB", "P"}:
            gene_chans.setdefault(_locus_key(loc), {})[pre] = i

    result = {}
    for cf, ct in chain_pairs:
        lags = []
        for chans in gene_chans.values():
            if cf not in chans or ct not in chans:
                continue
            xs = tr_active[:, :, chans[cf]].astype(np.float64).mean(axis=0)
            ys = tr_active[:, :, chans[ct]].astype(np.float64).mean(axis=0)
            xs = xs - xs.mean(); ys = ys - ys.mean()
            best_corr, best_lag = -np.inf, 0
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    a, b = xs[:T - lag], ys[lag:]
                else:
                    a, b = xs[-lag:], ys[:T + lag]
                if len(a) < 3:
                    continue
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-9 or nb < 1e-9:
                    continue
                c = float((a * b).sum() / (na * nb))
                if c > best_corr:
                    best_corr, best_lag = c, lag
            lags.append(best_lag)
        if lags:
            result[f"{cf}->{ct}"] = {
                "n_genes":    len(lags),
                "mean_lag":   float(np.mean(lags)),
                "median_lag": float(np.median(lags)),
                "std_lag":    float(np.std(lags)),
            }
    return result


def lens_conservation_pairs(tr_counts, va_counts, species_names,
                            anticorr=CONSPAIR_ANTICORR, cv_max=CONSPAIR_CV_MAX):
    """v15.5 PhD upgrade 1: find anti-correlated species pairs whose COUNT SUM
    is conserved over the cell cycle — A + B ≈ const.

    Mechanism-over-correlation: an anti-correlation r≈-1 between A and B is
    *explained* (not just fit) when A and B are two states of one conserved
    pool (free gene G vs ribosome-bound RP; charged vs uncharged tRNA).  Those
    are HARD constraints (A+B=const), not soft patterns to learn.

    For each pair with train-Δ correlation < anticorr, test whether
    (A_t + B_t) has coefficient-of-variation < cv_max on BOTH train and
    held-out val (so the conservation generalises).  Returns the validated
    conserved pairs with their conserved sum level.

    Counts are in raw (linear) space — conservation is a count identity, not a
    normalised-space one.
    """
    n_tr, T, S = tr_counts.shape
    Xtr = tr_counts.reshape(-1, S).astype(np.float64)
    Xva = va_counts.reshape(-1, S).astype(np.float64)
    # Δ-correlation to find anti-correlated candidates (reuse pairwise math)
    d_tr = np.diff(tr_counts, axis=1).reshape(-1, S).astype(np.float64)
    sd = d_tr.std(axis=0)
    valid = sd > 1e-9
    if valid.sum() < 2:
        return []
    Z = np.where(valid, (d_tr - d_tr.mean(axis=0)) / (sd + 1e-12), 0.0)
    C = (Z.T @ Z) / d_tr.shape[0]
    np.fill_diagonal(C, 0.0)
    C[~valid, :] = 0.0; C[:, ~valid] = 0.0
    # candidate anti-correlated pairs
    cand = np.argwhere(C < anticorr)
    out = []
    seen = set()
    for i, j in cand:
        if i >= j or (i, j) in seen:
            continue
        seen.add((i, j))
        s_tr = Xtr[:, i] + Xtr[:, j]
        s_va = Xva[:, i] + Xva[:, j]
        m_tr = s_tr.mean()
        if m_tr < CONSPAIR_MEAN_FLOOR:       # v15.7: was 1.0 — catch flickering single-copy genes
            continue
        cv_tr = s_tr.std() / (abs(m_tr) + 1e-9)
        cv_va = s_va.std() / (abs(s_va.mean()) + 1e-9)
        if cv_tr < cv_max and cv_va < cv_max:
            out.append({
                "i": int(i), "j": int(j),
                "sum_level": float(m_tr),
                "cv_train": float(cv_tr), "cv_val": float(cv_va),
                "corr": float(C[i, j]),
            })
    # strongest (lowest drift) first
    out.sort(key=lambda d: d["cv_val"])
    return out


def lens_mutual_information(tr, top_k=MI_LENS_TOP_K, bins=MI_LENS_BINS):
    """v15.5 PhD upgrade 3a: nonlinear coupling via mutual information on Δ.

    Correlation only sees linear coupling.  MI sees ANY statistical dependence
    — conditional activation, threshold switches, saturation.  We restrict to
    the most-variable species (MI on 5,900² pairs is infeasible) and report the
    top-K pairs whose MI is high but linear |corr| is LOW — i.e. couplings the
    pairwise lens is BLIND to.

    Returns list of (i, j, mi_nats, abs_corr).
    """
    n_tr, T, S = tr.shape
    d = np.diff(tr, axis=1).reshape(-1, S).astype(np.float64)
    var = d.var(axis=0)
    # restrict to a tractable set of the most-variable species
    k_keep = min(120, S)
    keep = np.argsort(-var)[:k_keep]
    dk = d[:, keep]
    n = dk.shape[0]
    if n < 50:
        return []
    # discretise each kept species into `bins` quantile bins
    disc = np.zeros_like(dk, dtype=np.int32)
    for c in range(dk.shape[1]):
        col = dk[:, c]
        qs = np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1]) if col.std() > 1e-12 else None
        disc[:, c] = np.digitize(col, qs) if qs is not None else 0
    # linear corr among kept (to find MI-high / corr-low pairs)
    sd = dk.std(axis=0) + 1e-12
    Z = (dk - dk.mean(axis=0)) / sd
    Ck = np.abs((Z.T @ Z) / n)
    results = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            xa, xb = disc[:, a], disc[:, b]
            # joint + marginal histograms → MI in nats
            pj = np.histogram2d(xa, xb, bins=bins)[0] / n
            pa = pj.sum(axis=1, keepdims=True)
            pb = pj.sum(axis=0, keepdims=True)
            ac = float(Ck[a, b])
            if ac >= MI_CORR_CEIL:        # v15.7: skip linearly-coupled pairs — keep only genuinely nonlinear
                continue
            nz = pj > 0
            mi = float((pj[nz] * np.log(pj[nz] / (pa @ pb)[nz] + 1e-12)).sum())
            results.append((int(keep[a]), int(keep[b]), mi, ac))
    # rank by MI (already filtered to |corr| < ceil, so these are nonlinear-only)
    results.sort(key=lambda r: -r[2])
    return results[:top_k]


def lens_granger(tr, species_names, top_k=GRANGER_TOP_K, max_lag=GRANGER_MAX_LAG):
    """v15.7 Move B: directed causality via pairwise Granger test on the most-
    variable species.  A 'Granger-causes' B if past(A) improves the AR
    prediction of B beyond past(B) alone.  Reports directed pairs ranked by the
    variance-reduction ratio (how much A's history cuts B's residual variance).

    Distinct from lens_gene_chain (fixed G->R->P chain, cross-correlation only):
    this is data-driven *direction* discovery across arbitrary species pairs.
    """
    n_tr, T, S = tr.shape
    # mean trajectory per species (Granger on the ensemble-mean dynamics)
    x = tr.mean(axis=0).astype(np.float64)                 # (T, S)
    var = x.var(axis=0)
    keep = np.argsort(-var)[:min(60, S)]                   # tractable subset
    xs = x[:, keep]
    L = max_lag
    if T - L < 8:
        return []

    def _resid_var(target_col, predictor_cols):
        # AR(L) regression of target on lagged predictors; return residual variance
        y = xs[L:, target_col]
        feats = []
        for c in predictor_cols:
            for lag in range(1, L + 1):
                feats.append(xs[L - lag:T - lag, c])
        A = np.column_stack(feats + [np.ones(len(y))])
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(((y - A @ coef) ** 2).mean())
        except Exception:
            return float(y.var())

    out = []
    for bi, b in enumerate(keep):
        rv_self = _resid_var(bi, [bi])
        if rv_self < 1e-12:
            continue
        for ai, a in enumerate(keep):
            if ai == bi:
                continue
            rv_both = _resid_var(bi, [bi, ai])
            ratio = 1.0 - rv_both / rv_self          # fraction of B's residual var A explains
            if ratio > 0.05:
                out.append((int(a), int(b), float(ratio)))
    out.sort(key=lambda r: -r[2])
    return out[:top_k]


def lens_ntuple_conservation(tr_counts, va_counts, species_names,
                              max_terms=NTUPLE_MAX_TERMS, top_k=NTUPLE_TOP_K,
                              cv_max=CONSPAIR_CV_MAX):
    """v15.7 Move B: search for N-way conserved sums (A+B+C = const), the
    generalisation of the pairwise conservation lens.  Restricted to the
    most-variable species and built greedily: start from a high-variance seed,
    add the species that most reduces the running-sum's coefficient of
    variation, stop at max_terms or when CV < cv_max.

    Returns validated N-tuples (CV < cv_max on BOTH train and val)."""
    n_tr, T, S = tr_counts.shape
    Xtr = tr_counts.reshape(-1, S).astype(np.float64)
    Xva = va_counts.reshape(-1, S).astype(np.float64)
    var = Xtr.var(axis=0)
    seeds = np.argsort(-var)[:min(40, S)]                  # only seed from high-variance species
    out, used = [], set()
    for seed in seeds:
        if seed in used or var[seed] < 1e-9:
            continue
        members = [int(seed)]
        s_tr = Xtr[:, seed].copy()
        for _ in range(max_terms - 1):
            best_j, best_cv = None, np.inf
            cand = np.argsort(-var)[:min(60, S)]
            for j in cand:
                if j in members:
                    continue
                trial = s_tr + Xtr[:, j]
                mt = trial.mean()
                if mt < CONSPAIR_MEAN_FLOOR:
                    continue
                cv = trial.std() / (abs(mt) + 1e-9)
                if cv < best_cv:
                    best_cv, best_j = cv, int(j)
            if best_j is None:
                break
            members.append(best_j)
            s_tr = s_tr + Xtr[:, best_j]
            if best_cv < cv_max:
                break
        if len(members) >= 2:
            sv = Xva[:, members].sum(axis=1)
            cv_tr = s_tr.std() / (abs(s_tr.mean()) + 1e-9)
            cv_va = sv.std() / (abs(sv.mean()) + 1e-9)
            if cv_tr < cv_max and cv_va < cv_max and len(members) >= 3:
                out.append({"members": members, "sum_level": float(s_tr.mean()),
                            "cv_train": float(cv_tr), "cv_val": float(cv_va)})
                used.update(members)
    out.sort(key=lambda d: d["cv_val"])
    return out[:top_k]


def lens_regime_conditional(tr, species_names, n_phases=REGIME_N_PHASES,
                              top_k=PAIRWISE_TOP_K, threshold=PAIRWISE_THRESHOLD):
    """v15.7 Move B: pairwise couplings that exist in ONE cell-cycle phase but
    not others.  Splits each trajectory into n_phases time-segments, computes
    Δ-correlation per phase, and reports pairs whose correlation swings by more
    than `threshold` between phases — regime-dependent structure the global
    pairwise lens averages away.

    Returns list of (i, j, phase_of_max, corr_range)."""
    n_tr, T, S = tr.shape
    seg = T // n_phases
    if seg < 3:
        return []
    var = np.diff(tr, axis=1).reshape(-1, S).var(axis=0)
    keep = np.argsort(-var)[:min(80, S)]
    phase_corr = []                                          # (n_phases, k, k)
    for p in range(n_phases):
        sl = tr[:, p * seg:(p + 1) * seg, :]
        d = np.diff(sl, axis=1).reshape(-1, S)[:, keep].astype(np.float64)
        sd = d.std(axis=0) + 1e-12
        Z = (d - d.mean(axis=0)) / sd
        phase_corr.append((Z.T @ Z) / d.shape[0])
    phase_corr = np.stack(phase_corr)                        # (P, k, k)
    rng = phase_corr.max(axis=0) - phase_corr.min(axis=0)    # (k, k) swing
    np.fill_diagonal(rng, 0.0)
    out, seen = [], set()
    flat = np.argsort(-rng.ravel())
    K = len(keep)
    for fi in flat[:top_k * 6]:
        a, b = fi // K, fi % K
        if a >= b or (a, b) in seen:
            continue
        seen.add((a, b))
        swing = float(rng[a, b])
        if swing < threshold:
            break
        pmax = int(np.argmax(np.abs(phase_corr[:, a, b])))
        out.append((int(keep[a]), int(keep[b]), pmax, swing))
        if len(out) >= top_k:
            break
    return out


class DiscoveredPatterns:
    """Statistical regularities mined from trajectories via multiple lenses.

    Distinct from KnownRules: these are empirical, observational.  Each lens
    looks for a different KIND of pattern (1D / 2D / low-dim / time / chain).
    """

    def __init__(self):
        self.up_frac_va = self.down_frac_va = None
        self.emp_up    = set()
        self.emp_down  = set()
        self.lo_cand   = self.hi_cand = self.bound_ok = None
        self.pairwise     = []
        self.conservation = []
        self.periodicity  = []
        self.gene_chain   = {}
        # v15.5 PhD upgrades:
        self.conservation_pairs = []   # upgrade 1: validated A+B=const pairs
        self.mutual_info        = []   # upgrade 3a: nonlinear couplings
        self.central_species    = {}   # upgrade 2: species → which lenses flagged it
        # v15.7 Move B: deeper lenses
        self.granger            = []   # directed causality (a, b, ratio)
        self.ntuples            = []   # N-way conservation A+B+C=const
        self.regime_pairs       = []   # phase-dependent couplings

    @classmethod
    def from_trajectories(cls, train_X, val_X, train_counts, val_counts, species_names,
                          lo=None, span=None):
        tr, va = train_X.numpy(), val_X.numpy()
        d_tr   = np.diff(tr, axis=1)
        d_va   = np.diff(va, axis=1)
        p = cls()
        print("[patterns] 1D monotone lens ...")
        p.up_frac_va, p.down_frac_va, p.emp_up, p.emp_down = lens_monotone(d_tr, d_va)
        print("[patterns] 1D bounds lens ...")
        p.lo_cand, p.hi_cand, p.bound_ok = lens_bounds(
            tr, va, lo=lo, span=span, raw_counts_tr=train_counts)
        print("[patterns] 2D pairwise lens ...")
        p.pairwise = lens_pairwise(d_tr, d_va)
        print("[patterns] low-d conservation lens ...")
        p.conservation = lens_conservation(train_counts, val_counts)
        print("[patterns] time/FFT periodicity lens ...")
        p.periodicity = lens_periodicity(tr)
        print("[patterns] gene-chain lag lens ...")
        p.gene_chain = lens_gene_chain(tr, species_names)
        # v15.5 upgrade 1: conservation-pair lens (mechanism over correlation)
        print("[patterns] conservation-pair lens (A+B=const) ...")
        p.conservation_pairs = lens_conservation_pairs(
            train_counts, val_counts, species_names)
        # v15.5 upgrade 3a: nonlinear mutual-information lens
        print("[patterns] nonlinear mutual-information lens ...")
        p.mutual_info = lens_mutual_information(tr)
        # v15.7 Move B: deeper lenses
        print("[patterns] Granger-causality lens ...")
        p.granger = lens_granger(tr, species_names)
        print("[patterns] N-tuple conservation lens (A+B+C=const) ...")
        p.ntuples = lens_ntuple_conservation(train_counts, val_counts, species_names)
        print("[patterns] regime-conditional lens (phase-dependent couplings) ...")
        p.regime_pairs = lens_regime_conditional(tr, species_names)
        # v15.5 upgrade 2: lens composition — connect the dots across lenses
        print("[patterns] lens-composition (connecting the dots) ...")
        p.central_species = p._compose_lenses(species_names)
        return p

    def _compose_lenses(self, species_names):
        """v15.5 upgrade 2: cross-reference lens outputs.  A species flagged by
        multiple INDEPENDENT lenses is mechanistically central — e.g. a species
        that is both in an SVD conservation direction AND periodic is likely a
        conserved oscillator (cell-cycle clock candidate)."""
        from collections import defaultdict
        flags = defaultdict(set)
        for c in self.conservation:
            for idx in c["top_species_idx"]:
                flags[idx].add("svd-conservation")
        for cp in self.conservation_pairs:
            flags[cp["i"]].add("pair-conservation")
            flags[cp["j"]].add("pair-conservation")
        for (i, j, *_ ) in self.pairwise:
            flags[i].add("pairwise"); flags[j].add("pairwise")
        for (i, _f, _r) in self.periodicity:
            flags[i].add("periodic")
        for (i, j, _mi, _c) in self.mutual_info:
            flags[i].add("nonlinear-MI"); flags[j].add("nonlinear-MI")
        for (a, b, _r) in self.granger:
            flags[a].add("granger-cause"); flags[b].add("granger-effect")
        for nt in self.ntuples:
            for idx in nt["members"]:
                flags[idx].add("ntuple-conservation")
        for (a, b, _p, _s) in self.regime_pairs:
            flags[a].add("regime-dependent"); flags[b].add("regime-dependent")
        # keep only multiply-flagged species
        central = {idx: sorted(s) for idx, s in flags.items()
                   if len(s) >= COMPOSE_MIN_LENSES}
        return central

    def summary(self, species_names):
        S = len(species_names)
        nb = int(self.bound_ok.sum()) if self.bound_ok is not None else 0
        return "\n".join([
            "DiscoveredPatterns (multi-lens analysis of trajectories):",
            f"    Monotone-up candidates      : {len(self.emp_up)}  (1D lens)",
            f"    Monotone-down candidates    : {len(self.emp_down)}  (1D lens)",
            f"    Per-species bounds validated: {nb}/{S}  (1D lens)",
            f"    Pairwise couplings (|r|>{PAIRWISE_THRESHOLD}): {len(self.pairwise)}  (2D lens)",
            f"    Conservation candidates     : {len(self.conservation)}  (low-d / SVD lens)",
            f"    Periodic species (top)      : {len(self.periodicity)}  (FFT lens)",
            f"    Gene-chain lag pairs        : {len(self.gene_chain)}  (chain lens)",
            f"    Conserved A+B=const pairs   : {len(self.conservation_pairs)}  "
            f"(v15.5 conservation-pair lens — HARD constraints)",
            f"    Nonlinear MI couplings      : {len(self.mutual_info)}  "
            f"(v15.5 mutual-information lens)",
            f"    Mechanistically-central spp : {len(self.central_species)}  "
            f"(v15.5 lens-composition — flagged by ≥{COMPOSE_MIN_LENSES} lenses)",
            f"    Granger directed pairs      : {len(self.granger)}  "
            f"(v15.7 causality lens)",
            f"    N-tuple conservation        : {len(self.ntuples)}  "
            f"(v15.7 — A+B+C=const)",
            f"    Regime-dependent couplings  : {len(self.regime_pairs)}  "
            f"(v15.7 phase-conditional lens)",
        ])


# ── enforcement: Tier 1 RuleSet + Tier 2 Hypotheses ──────────────────────────

class RuleSet:
    """Tier 1: validated rules, enforced as hard rollout guardrails."""

    def __init__(self):
        self.mono_up_mask = self.mono_down_mask = None
        self.mono_up = self.mono_down = None
        self.lo_bound = self.hi_bound = None
        self.n_seed = self.n_sbml = self.n_empirical = 0
        # v15.5 upgrade 1: conservation-pair hard constraints.  Buffers in
        # NORMALISED space so projection happens in the same domain as `project`.
        self.cpair_i = self.cpair_j = None      # (P,) long species indices
        self.cpair_lo = self.cpair_span = None  # (P,2) per-pair normalisation params
        self.n_cpair = 0

    def to(self, dev):
        for a in ("mono_up_mask", "mono_down_mask", "mono_up", "mono_down",
                  "lo_bound", "hi_bound", "cpair_i", "cpair_j",
                  "cpair_lo", "cpair_span"):
            t = getattr(self, a, None)
            if t is not None:
                setattr(self, a, t.to(dev))
        return self

    def project(self, prev, nxt):
        if self.mono_up_mask is not None:
            nxt = torch.where(self.mono_up_mask.unsqueeze(0),
                              torch.maximum(nxt, prev), nxt)
        if self.mono_down_mask is not None:
            nxt = torch.where(self.mono_down_mask.unsqueeze(0),
                              torch.minimum(nxt, prev), nxt)
        if self.lo_bound is not None:
            nxt = torch.clamp(nxt, self.lo_bound.unsqueeze(0),
                              self.hi_bound.unsqueeze(0))
        # v15.5 upgrade 1: enforce A+B = const for validated conservation pairs.
        # Conservation is a COUNT identity, so convert the two species to counts,
        # rescale so their sum matches the conserved level from prev, convert back.
        if self.cpair_i is not None:
            nxt = self._project_conservation_pairs(prev, nxt)
        return nxt

    def _project_conservation_pairs(self, prev, nxt):
        # de-normalise the pair species to counts via stored lo/span (signed-log)
        i, j = self.cpair_i, self.cpair_j
        lo_i, lo_j     = self.cpair_lo[:, 0],   self.cpair_lo[:, 1]
        span_i, span_j = self.cpair_span[:, 0], self.cpair_span[:, 1]

        def to_count(x_norm, lo, span):
            sl = x_norm * span + lo
            return torch.sign(sl) * torch.expm1(torch.abs(sl))

        def to_norm(cnt, lo, span):
            sl = torch.sign(cnt) * torch.log1p(torch.abs(cnt))
            return ((sl - lo) / span).clamp(CLAMP_LO, CLAMP_HI)

        a_prev = to_count(prev[:, i], lo_i, span_i).clamp(min=0)
        b_prev = to_count(prev[:, j], lo_j, span_j).clamp(min=0)
        a_next = to_count(nxt[:, i],  lo_i, span_i).clamp(min=0)
        b_next = to_count(nxt[:, j],  lo_j, span_j).clamp(min=0)
        target_sum = a_prev + b_prev                          # conserved level
        pred_sum   = (a_next + b_next).clamp(min=1e-6)
        scale      = (target_sum / pred_sum).unsqueeze(0) if target_sum.dim() == 1 else target_sum / pred_sum
        a_fix = a_next * (target_sum / pred_sum)
        b_fix = b_next * (target_sum / pred_sum)
        nxt = nxt.clone()
        nxt[:, i] = to_norm(a_fix, lo_i, span_i).to(nxt.dtype)
        nxt[:, j] = to_norm(b_fix, lo_j, span_j).to(nxt.dtype)
        return nxt

    def summary(self):
        nu = int(self.mono_up_mask.sum())   if self.mono_up_mask   is not None else 0
        nd = int(self.mono_down_mask.sum()) if self.mono_down_mask is not None else 0
        nb = "yes" if self.lo_bound is not None else "no"
        return (f"Tier 1 RuleSet: {nu} monotone-up + {nd} monotone-down + "
                f"per-species bounds ({nb}) + {self.n_cpair} conservation-pairs  "
                f"[provenance: {self.n_seed} seed, {self.n_sbml} SBML-backed, "
                f"{self.n_empirical} trajectory-only]")


class Hypotheses:
    """Tier 2: candidates that did NOT pass validation, or are not enforceable.

    Tracked, confidence-scored and REPORTED.  v13.8: also optionally fed back
    to training as a SOFT loss (weight LAMBDA_HYP, small) — preserves the
    "can be wrong" semantics because training data dominates ties, but stops
    the model from being completely blind to what the discovery lenses found.
    """

    def __init__(self):
        self.items = []
        # v13.8: soft-loss tensors, populated by build_tensors().  None until then.
        self._mono_up_idx = None
        self._mono_down_idx = None
        self._pair_i = None
        self._pair_j = None
        self._pair_r = None
        self._delta_mean = None
        self._delta_std = None
        # Mono loss lives in normalised [0,1] space (≈ same scale as main loss).
        # Pair loss lives in z-score delta space, naturally ~100× larger; the 0.01
        # weight rescales it to the same effective magnitude per element.
        self._kind_weights = {"mono": 1.0, "pair": 0.01}

    def add(self, kind, detail, score, source, data=None):
        self.items.append({"kind": kind, "detail": detail,
                           "score": score, "source": source, "data": data})

    def summary(self, max_show=20):
        if not self.items:
            return "Tier 2 Hypotheses: none"
        lines = [f"Tier 2 Hypotheses: {len(self.items)} (reported, NOT enforced)"]
        for h in sorted(self.items, key=lambda x: -x["score"])[:max_show]:
            lines.append(f"    [{h['source']:10s}] {h['kind']:22s} "
                         f"{h['detail']}  (score {h['score']:.3f})")
        if len(self.items) > max_show:
            lines.append(f"    ... and {len(self.items)-max_show} more")
        return "\n".join(lines)

    def build_tensors(self, train_X_norm, device):
        """v13.8: compile items into tensors for the auxiliary soft loss.

        train_X_norm: (N, T, S) normalised training tensor on CPU.

        Soft losses (each ~0 when prediction is consistent with the hypothesis):
          monotone-up  : F.relu(prev - nxt) on stored indices
          monotone-down: F.relu(nxt - prev) on stored indices
          pairwise     : standardised-delta residual after the implied
                         linear relation, (dzi - r * dzj)^2

        Conservation, periodicity, gene-chain-lag are skipped for now —
        they're nonlinear in normalised space or hard to evaluate
        instantaneously.  Add in a follow-up if mono+pair shows traction.
        """
        N, T, S = train_X_norm.shape
        n_pairs = N * (T - 1)
        if n_pairs > 50_000:
            g = torch.Generator().manual_seed(SEED + 3)
            idx_n = torch.randint(0, N, (50_000,), generator=g)
            idx_t = torch.randint(0, T - 1, (50_000,), generator=g)
            deltas = train_X_norm[idx_n, idx_t + 1] - train_X_norm[idx_n, idx_t]
        else:
            deltas = (train_X_norm[:, 1:, :] - train_X_norm[:, :-1, :]).reshape(-1, S)
        self._delta_mean = deltas.mean(dim=0).to(device)
        self._delta_std = (deltas.std(dim=0) + 1e-6).to(device)

        mono_up, mono_down, pair_i, pair_j, pair_r = [], [], [], [], []
        for h in self.items:
            d = h.get("data")
            if d is None:
                continue
            kind = h["kind"]
            if kind == "monotone-up":
                mono_up.append(int(d["i"]))
            elif kind == "monotone-down":
                mono_down.append(int(d["i"]))
            elif kind.startswith("pairwise"):
                pair_i.append(int(d["i"]))
                pair_j.append(int(d["j"]))
                # Use the TRAIN correlation, not the held-out one — using r_va
                # would leak test-set statistics into the training loss.
                pair_r.append(float(d["r_tr"]))

        if mono_up:
            self._mono_up_idx = torch.tensor(mono_up, dtype=torch.long, device=device)
        if mono_down:
            self._mono_down_idx = torch.tensor(mono_down, dtype=torch.long, device=device)
        if pair_i:
            self._pair_i = torch.tensor(pair_i, dtype=torch.long, device=device)
            self._pair_j = torch.tensor(pair_j, dtype=torch.long, device=device)
            self._pair_r = torch.tensor(pair_r, dtype=torch.float32, device=device)
        n_total = len(mono_up) + len(mono_down) + len(pair_i)
        print(f"[hyp] aux-loss tensors: {len(mono_up)} mono-up + "
              f"{len(mono_down)} mono-down + {len(pair_i)} pair "
              f"(total {n_total} soft constraints, lambda={LAMBDA_HYP})")
        return n_total

    def has_aux_loss(self):
        return (self._mono_up_idx is not None
                or self._mono_down_idx is not None
                or self._pair_i is not None)

    def soft_loss(self, prev, nxt):
        """Returns (scalar aux loss, per-kind dict of DETACHED TENSORS).

        prev, nxt: (B, S) normalised state tensors (any dtype, BF16-safe).
        The per-kind values are GPU tensors — the caller is responsible for
        any .item()/float() conversion (deferred to avoid GPU sync in the
        hot training loop).
        """
        prev = prev.to(nxt.dtype)        # BF16 safety
        per_kind = {}
        total = torch.zeros((), device=nxt.device, dtype=nxt.dtype)
        w_mono = self._kind_weights["mono"]
        w_pair = self._kind_weights["pair"]

        if self._mono_up_idx is not None:
            v = F.relu(prev[:, self._mono_up_idx] - nxt[:, self._mono_up_idx])
            l = (v * v).mean()
            per_kind["mono_up"] = l.detach()
            total = total + w_mono * l

        if self._mono_down_idx is not None:
            v = F.relu(nxt[:, self._mono_down_idx] - prev[:, self._mono_down_idx])
            l = (v * v).mean()
            per_kind["mono_down"] = l.detach()
            total = total + w_mono * l

        if self._pair_i is not None:
            dx = nxt - prev
            dz = (dx - self._delta_mean.to(dx.dtype)) / self._delta_std.to(dx.dtype)
            dzi = dz[:, self._pair_i]
            dzj = dz[:, self._pair_j]
            r = self._pair_r.to(dx.dtype)
            res = dzi - r.unsqueeze(0) * dzj
            l = (res * res).mean()
            per_kind["pair"] = l.detach()
            total = total + w_pair * l

        return total, per_kind


def _build_cpair_classifier(known, species_names):
    """v15.8.3: classify a conservation-pair (i, j) by mechanistic backing.

    Returns a closure `classify(i, j) -> (kind, source)` where `kind` is one of:
      - 'ribosome_open_state'     — name patterns differ only by '_open'
      - 'sbml_interconversion'    — same SBML reaction, opposite-sign stoich
      - 'complex_subunits'        — both species are subunits of same complex
      - 'largesubunit_assembly'   — both in LargeSubunit assembly path
      - 'trna_charging'           — pattern: M_<aa>trna_c + M_f<aa>trna_c
      - 'trajectory_only'         — no mechanism backing found

    Anything that is NOT 'trajectory_only' is considered safe to project as a
    hard Tier 1 constraint.  Trajectory-only pairs are mathematically valid
    on the 50 training trajectories but could be artifacts of those specific
    simulator runs — they should be Tier 2 SOFT until validated on perturbed
    conditions (knockouts, parameter sweeps).
    """
    rxn_membership = {}                # species_id -> [(rxn_id, signed_stoich)]
    if known is not None and known.sbml is not None:
        for rxn in known.sbml.get("reactions", []):
            rid = rxn["id"]
            for sid, st in rxn.get("reactants", []):
                rxn_membership.setdefault(sid, []).append((rid, -float(st)))
            for sid, st in rxn.get("products", []):
                rxn_membership.setdefault(sid, []).append((rid, +float(st)))

    complex_membership = {}            # species_id -> set of complex names
    if known is not None and known.complexes is not None:
        for cx in known.complexes.get("complexes", []):
            for sub_id, _ in cx.get("subunits", []):
                complex_membership.setdefault(sub_id, set()).add(cx["name"])
        # also include predefined complex names (species that ARE the complex)
        for cx_name in (known.complexes.get("predefined", {}) or {}):
            complex_membership.setdefault(cx_name, set()).add(cx_name)

    largesubunit_species = set()
    if known is not None:
        for (sub, inter, prod) in (known.largesubunit or []):
            for s in (sub, inter, prod):
                if s and s != "nan":
                    largesubunit_species.add(s)

    def classify(i, j):
        name_i = species_names[i] if i < len(species_names) else ""
        name_j = species_names[j] if j < len(species_names) else ""
        # 1. ribosome _open-state pattern (e.g. 'RP_0068_c1_2' + 'RP_0068_c1_open_2')
        if "_open_" in name_i and name_i.replace("_open_", "_") == name_j:
            return ("ribosome_open_state", f"pattern:{name_j}")
        if "_open_" in name_j and name_j.replace("_open_", "_") == name_i:
            return ("ribosome_open_state", f"pattern:{name_i}")
        # Also handle 'open' as suffix without trailing index (rare)
        if name_i.endswith("_open") and name_i[:-5] == name_j:
            return ("ribosome_open_state", f"pattern:{name_j}")
        if name_j.endswith("_open") and name_j[:-5] == name_i:
            return ("ribosome_open_state", f"pattern:{name_i}")
        # 1b. v15.8.4: central-dogma gene occupancy — G_<locus>_C<n> + RP_<locus>_C<n>.
        # Each chromosome copy has a fixed gene count, and the gene is either free
        # (G_) or RNA-polymerase-occupied (RP_).  Their sum is a hard biological
        # constraint (chromosome stoichiometry).  Pattern: same locus and chromosome
        # suffix, prefix swapped G_<X> <-> RP_<X>.
        if name_i.startswith("G_") and name_j.startswith("RP_") and name_i[2:] == name_j[3:]:
            return ("central_dogma_gene_occupancy", f"pattern:G/RP_{name_i[2:]}")
        if name_j.startswith("G_") and name_i.startswith("RP_") and name_j[2:] == name_i[3:]:
            return ("central_dogma_gene_occupancy", f"pattern:G/RP_{name_j[2:]}")
        # 2. SBML interconversion (same reaction, opposite stoich)
        for (rid_i, st_i) in rxn_membership.get(name_i, []):
            for (rid_j, st_j) in rxn_membership.get(name_j, []):
                if rid_i == rid_j and abs(st_i + st_j) < 1e-9 and abs(st_i) > 1e-9:
                    return ("sbml_interconversion", f"reaction:{rid_i}")
        # 3. Complex co-subunits
        shared = complex_membership.get(name_i, set()) & complex_membership.get(name_j, set())
        if shared:
            return ("complex_subunits", f"complex:{next(iter(shared))}")
        # 4. LargeSubunit assembly co-membership
        if name_i in largesubunit_species and name_j in largesubunit_species:
            return ("largesubunit_assembly", "LargeSubunit.xlsx")
        # 5. tRNA charging pattern: M_<aa>trna_c + M_f<aa>trna_c (formyl-methionine
        #    style — generalises to any aminoacyl/formyl pair)
        if (name_i.startswith("M_") and name_j.startswith("M_")
                and "trna" in name_i.lower() and "trna" in name_j.lower()):
            # canonical form: stripping leading 'f' from one matches the other
            ni, nj = name_i[2:], name_j[2:]
            if ni == "f" + nj or nj == "f" + ni:
                return ("trna_charging", "biology_pattern:f_aminoacyl")
        return ("trajectory_only", None)

    return classify


def build_enforcement(known, patterns, species_names, lo=None, span=None):
    """Sort the validated subset of KnownRules+DiscoveredPatterns into Tier 1.
    Everything else (failed monotone, all conservation/pairwise/etc.) -> Tier 2.

    v15.5: validated conservation pairs (A+B=const) become HARD Tier-1
    constraints when lo/span are provided (needed to convert between normalised
    and count space).

    Returns (RuleSet, Hypotheses).
    """
    rs, hyp = RuleSet(), Hypotheses()
    S = len(species_names)

    seed_up = {i for i, nm in enumerate(species_names)
               if parse_species(nm)[0] in SEED_MONOTONE_UP_CHANNELS}
    sbml_up, sbml_down = sbml_monotone_candidates(known.sbml, species_names)
    emp_up, emp_down = patterns.emp_up, patterns.emp_down

    # ── monotone-up ──
    cand_up = seed_up | sbml_up | emp_up
    val_up = set()
    for i in cand_up:
        if patterns.up_frac_va[i] >= RULE_COMPLIANCE:
            val_up.add(i)
        else:
            src = ("seed" if i in seed_up else
                   "sbml" if i in sbml_up else "trajectory")
            hyp.add("monotone-up", f"'{species_names[i]}' fails held-out "
                    f"({patterns.up_frac_va[i]*100:.2f}% compliant)",
                    patterns.up_frac_va[i], src, data={"i": int(i)})

    # ── monotone-down ──
    cand_down = sbml_down | emp_down
    val_down = set()
    for i in cand_down:
        if patterns.down_frac_va[i] >= RULE_COMPLIANCE and i not in val_up:
            val_down.add(i)
        elif i not in val_up:
            src = "sbml" if i in sbml_down else "trajectory"
            hyp.add("monotone-down", f"'{species_names[i]}' fails held-out "
                    f"({patterns.down_frac_va[i]*100:.2f}% compliant)",
                    patterns.down_frac_va[i], src, data={"i": int(i)})

    # ── bounds ──
    lo, hi, ok = patterns.lo_cand, patterns.hi_cand, patterns.bound_ok
    rs.lo_bound = torch.from_numpy(np.where(ok, lo, CLAMP_LO).astype(np.float32))
    rs.hi_bound = torch.from_numpy(np.where(ok, hi, CLAMP_HI).astype(np.float32))

    up_sorted, down_sorted = sorted(val_up), sorted(val_down)
    rs.mono_up   = torch.tensor(up_sorted,   dtype=torch.long)
    rs.mono_down = torch.tensor(down_sorted, dtype=torch.long)
    um = torch.zeros(S, dtype=torch.bool); dm = torch.zeros(S, dtype=torch.bool)
    if up_sorted:   um[rs.mono_up]   = True
    if down_sorted: dm[rs.mono_down] = True
    rs.mono_up_mask, rs.mono_down_mask = um, dm
    rs.n_seed      = len(val_up & seed_up)
    rs.n_sbml      = len((val_up & sbml_up) | (val_down & sbml_down))
    rs.n_empirical = len((val_up | val_down) - seed_up - sbml_up - sbml_down)

    # ── Tier 2: report-only patterns ──
    for c in patterns.conservation:
        names_top = [species_names[i] for i in c["top_species_idx"]]
        wt        = [f"{w:+.2f}" for w in c["top_species_weight"]]
        detail = "+".join(f"{w} {n}" for w, n in zip(wt, names_top))
        hyp.add("conservation",
                f"sigma={c['std_val']:.3f} ≈ {detail}",
                1.0 / (1.0 + c["std_val"]), "svd")
    for i, j, c_tr, c_va in patterns.pairwise:
        kind = "anti-corr" if c_tr < 0 else "corr"
        hyp.add(f"pairwise-{kind}",
                f"'{species_names[i]}' <-> '{species_names[j]}' "
                f"(r_tr={c_tr:+.2f} r_va={c_va:+.2f})",
                abs(c_va), "trajectory",
                data={"i": int(i), "j": int(j),
                      "r_tr": float(c_tr), "r_va": float(c_va)})
    for i, freq_idx, rel in patterns.periodicity[:8]:
        hyp.add("periodicity",
                f"'{species_names[i]}' peak bin {freq_idx} (rel power {rel:.2f})",
                rel, "fft")
    for pair, info in patterns.gene_chain.items():
        hyp.add("gene-chain-lag",
                f"{pair}: mean lag {info['mean_lag']:+.1f} steps "
                f"(n={info['n_genes']} genes, std {info['std_lag']:.1f})",
                1.0 / (1.0 + info["std_lag"]),
                "trajectory")

    # ── v15.5 upgrade 1, v15.8.3 strict-mechanism gating: conservation pairs ──
    cpairs = getattr(patterns, "conservation_pairs", [])
    if cpairs:
        # v15.8.3: classify each pair by mechanistic backing.  Only mechanism-
        # backed pairs become HARD Tier 1 projections.  Trajectory-only pairs
        # demote to SOFT Tier 2 (loss penalty, no projection).
        classifier = _build_cpair_classifier(known, species_names)
        prov_counts = {}
        mech_backed, traj_only = [], []
        for cp in cpairs:
            kind, src = classifier(cp["i"], cp["j"])
            cp["_provenance"]     = kind
            cp["_provenance_src"] = src
            prov_counts[kind] = prov_counts.get(kind, 0) + 1
            (mech_backed if kind != "trajectory_only" else traj_only).append(cp)

        if CPAIR_STRICT_MECHANISM:
            enforce_list, soft_list = mech_backed, traj_only
        else:
            enforce_list, soft_list = cpairs, []

        if enforce_list and lo is not None and span is not None:
            ci, cj, c_lo, c_span = [], [], [], []
            for cp in enforce_list:
                i, j = cp["i"], cp["j"]
                ci.append(i); cj.append(j)
                c_lo.append([float(lo[i]), float(lo[j])])
                c_span.append([float(span[i]), float(span[j])])
            rs.cpair_i    = torch.tensor(ci, dtype=torch.long)
            rs.cpair_j    = torch.tensor(cj, dtype=torch.long)
            rs.cpair_lo   = torch.tensor(c_lo,   dtype=torch.float32)
            rs.cpair_span = torch.tensor(c_span, dtype=torch.float32).clamp(min=1e-6)
            rs.n_cpair    = len(enforce_list)
            print(f"[cpair-provenance] breakdown of {len(cpairs)} discovered pairs:")
            for kind, n in sorted(prov_counts.items(), key=lambda kv: -kv[1]):
                print(f"    {kind:<28s} {n:>4d}")
            mode = "STRICT" if CPAIR_STRICT_MECHANISM else "ALL"
            print(f"[cpair-provenance] {mode} mode: {len(enforce_list)} pairs "
                  f"enforced as HARD Tier 1; {len(soft_list)} demoted to SOFT Tier 2")
            for cp in enforce_list[:12]:
                print(f"[rules]   conserved pair: '{species_names[cp['i']]}' + "
                      f"'{species_names[cp['j']]}' = {cp['sum_level']:.1f} "
                      f"(cv_val {cp['cv_val']*100:.2f}%)  "
                      f"[{cp.get('_provenance', '?')}]")
        elif cpairs and (lo is None or span is None):
            # Can't build projection without lo/span — report everything as Tier 2.
            soft_list = cpairs

        # SOFT: trajectory-only pairs become Tier 2 hypotheses (reported, used
        # as soft loss when hyp.soft_loss is wired in training).
        for cp in soft_list:
            hyp.add("conservation-pair-soft",
                    f"'{species_names[cp['i']]}' + '{species_names[cp['j']]}' "
                    f"= {cp['sum_level']:.1f} (cv {cp['cv_val']*100:.2f}%) "
                    f"[trajectory-only, no mechanism backing]",
                    1.0 / (1.0 + cp["cv_val"]), "conspair-traj")

    # ── v15.5 upgrade 3a: nonlinear MI couplings → Tier 2 (reported) ──
    for (i, j, mi, ac) in getattr(patterns, "mutual_info", [])[:12]:
        hyp.add("nonlinear-MI",
                f"'{species_names[i]}' ~ '{species_names[j]}' "
                f"(MI={mi:.3f} nats, |corr|={ac:.2f})",
                mi, "mutual-info")
    # v15.7 Move B: report deeper-lens findings as Tier 2
    for (a, b, ratio) in getattr(patterns, "granger", [])[:12]:
        hyp.add("granger-cause",
                f"'{species_names[a]}' → '{species_names[b]}' "
                f"(resid-var cut {ratio*100:.0f}%)", ratio, "granger")
    for nt in getattr(patterns, "ntuples", [])[:8]:
        names = " + ".join(species_names[m] for m in nt["members"])
        hyp.add("ntuple-conservation",
                f"{names} = {nt['sum_level']:.1f} (cv {nt['cv_val']*100:.2f}%)",
                1.0 / (1.0 + nt["cv_val"]), "ntuple")
    for (a, b, pmax, swing) in getattr(patterns, "regime_pairs", [])[:8]:
        hyp.add("regime-coupling",
                f"'{species_names[a]}' ~ '{species_names[b]}' "
                f"(phase {pmax}, corr swing {swing:.2f})", swing, "regime")

    for i in seed_up - val_up:
        print(f"[rules] WARNING: seed monotone '{species_names[i]}' failed validation "
              f"({patterns.up_frac_va[i]*100:.2f}%) - moved to Tier 2")

    print(f"[rules] Tier 1: {len(val_up)} mono-up, {len(val_down)} mono-down, "
          f"{int(ok.sum())}/{S} bounded, {rs.n_cpair} conserved-pairs")
    print(f"[rules] Tier 2: {len(hyp.items)} hypotheses (reported, not enforced)")
    return rs, hyp


# ── cross-validation: KnownRules vs trajectory ───────────────────────────────

def cross_validate_known(known, raw_counts, species_names):
    """Compare KnownRules against trajectory. Returns dict with discrepancy info."""
    report = {}
    if known.initial is not None:
        t0 = raw_counts[:, 0, :].mean(0)        # mean state at trajectory t=0 (post-startup)
        loc_to = {"P": {}, "R": {}}
        for i, nm in enumerate(species_names):
            pre, loc = parse_species(nm)
            if pre in loc_to:
                loc_to[pre].setdefault(_locus_key(loc), i)

        def ratios(target_pre, source_dict):
            out = []
            for tag, expected in source_dict.items():
                key = tag.split("_")[1] if "_" in tag else tag
                i = loc_to[target_pre].get(key)
                if i is not None and t0[i] > 0 and expected > 1e-6:
                    out.append(t0[i] / expected)
            return out

        rp = ratios("P", known.initial["proteins"])
        rm = ratios("R", known.initial["mRNAs"])
        if rp:
            report["proteins"] = {"n": len(rp),
                                  "median": float(np.median(rp)),
                                  "p25": float(np.percentile(rp, 25)),
                                  "p75": float(np.percentile(rp, 75))}
        if rm:
            report["mRNAs"]    = {"n": len(rm),
                                  "median": float(np.median(rm)),
                                  "p25": float(np.percentile(rm, 25)),
                                  "p75": float(np.percentile(rm, 75))}

    if known.complexes is not None:
        loc_to_p = {}
        for i, nm in enumerate(species_names):
            pre, loc = parse_species(nm)
            if pre == "P":
                loc_to_p.setdefault(_locus_key(loc), i)
        tracked = 0
        for cx in known.complexes["complexes"]:
            all_in = all(g.strip() in loc_to_p for g, _ in cx["subunits"])
            if all_in:
                tracked += 1
        report["complex_tracking"] = {
            "n_total":   len(known.complexes["complexes"]),
            "n_tracked": tracked,
        }
    return report


# ── PhD summary ──────────────────────────────────────────────────────────────

def phd_summary(known, patterns, ruleset, hyp, cross, species_names):
    """One comprehensive printout of everything the knowledge phase produced."""
    print()
    print("#" * 72)
    print("#  PHD KNOWLEDGE SUMMARY  -  JCVI-Syn3A whole-cell emulator")
    print("#" * 72)
    print()
    print(known.summary())
    print()
    print(patterns.summary(species_names))
    print()
    if cross:
        print("Cross-validation (KnownRules vs trajectory t=0, post-startup):")
        if "proteins" in cross:
            r = cross["proteins"]
            print(f"    Protein initial counts  : {r['n']} matched, "
                  f"traj/xlsx median {r['median']:.2f} (IQR {r['p25']:.2f}-{r['p75']:.2f})")
        if "mRNAs" in cross:
            r = cross["mRNAs"]
            print(f"    mRNA initial counts     : {r['n']} matched, "
                  f"traj/xlsx median {r['median']:.2f} (IQR {r['p25']:.2f}-{r['p75']:.2f})")
        if "complex_tracking" in cross:
            c = cross["complex_tracking"]
            print(f"    Complex subunit tracking: {c['n_tracked']}/{c['n_total']} "
                  "complexes have all subunits in the trajectory")
    # v15.5 upgrade 2: lens-composition — connecting the dots across lenses
    central = getattr(patterns, "central_species", {})
    if central:
        print()
        print("Lens composition (species flagged by multiple lenses — "
              "mechanistically central):")
        ranked = sorted(central.items(), key=lambda kv: -len(kv[1]))[:12]
        for idx, lenses in ranked:
            print(f"    {species_names[idx]:24s}  {len(lenses)} lenses: "
                  f"{', '.join(lenses)}")
    print()
    print(ruleset.summary())
    print()
    print(hyp.summary())
    print("#" * 72)


# ── model ─────────────────────────────────────────────────────────────────────
#
# v8: Liquid Graph Neural Network.  Ports the core M7 architectural ideas from
# the parallel claude/build-m7-surrogate-Dt8w7 branch — SBML-derived species
# graph + Liquid (CfC) node dynamics + per-species learned time constants —
# into our scaffolding.  The PINN mass-balance head from M7 is deferred: it
# tangles with our (x-lo)/span normalisation and only covers ~2.5% of species
# (the SBML-mapped subset), so the value-vs-risk is worse than getting the CfC
# graph in cleanly first.
#
# Model is single-step: takes the current state, returns the next state.  The
# context window from v7 is dropped — the CfC's continuous-time formulation
# carries temporal structure through the per-node τ.
# ─────────────────────────────────────────────────────────────────────────────


def build_sbml_graph(sbml, species_names):
    """Build an SBML-derived species graph.

    Two species are connected if they co-occur in any SBML reaction (so the
    GNN can propagate information through reactions).  Self-loops added so
    species with no SBML edges still update themselves.

    Returns:
        edge_index : (2, E) long
        edge_weight: (E,) float
    """
    n = len(species_names)
    edges = set()
    if sbml is not None:
        name_to_idx = {nm: i for i, nm in enumerate(species_names)}
        for rxn in sbml["reactions"]:
            cols = [name_to_idx[s] for s, _ in rxn["reactants"] + rxn["products"]
                    if s in name_to_idx]
            for i in cols:
                for j in cols:
                    edges.add((i, j))
    for i in range(n):                     # always include self-loops
        edges.add((i, i))
    edge_list = list(edges)
    edge_index  = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


def build_full_graph(sbml, kinetics, complexes, species_names,
                     protein_metabolites=None, largesubunit=None):
    """v10 Tier C + v11: SBML + central dogma + enzyme→flux + subunit→complex
    + protein↔metabolite regulation + 50S ribosome assembly.

    Seven edge sources, all bidirectional in the GNN:
      1. SBML reaction co-occurrence — species in the same reaction
      2. Central dogma per gene — G ↔ R ↔ P ↔ RP ↔ RB
      3. Enzyme → flux — P_xxxx ↔ F_yyyy (kinetic_params' reaction→enzyme map)
      4. Subunit → complex — subunit protein ↔ complex species
      5. NEW v11: Protein ↔ metabolite — regulatory binding (protein_metabolites.xlsx)
      6. NEW v11: 50S assembly — ribosomal subunits ↔ assembled intermediates
      7. Self-loops on every node

    Defensive: every edge is only added if both endpoints exist in species_names.
    """
    n = len(species_names)
    name_to_idx = {nm: i for i, nm in enumerate(species_names)}
    edges = set()
    n_sbml = n_cd = n_enz = n_cplx = n_pm = n_lsu = 0

    # 1. SBML reaction co-occurrence
    if sbml is not None:
        for rxn in sbml["reactions"]:
            cols = [name_to_idx[s] for s, _ in rxn["reactants"] + rxn["products"]
                    if s in name_to_idx]
            for i in cols:
                for j in cols:
                    if (i, j) not in edges:
                        edges.add((i, j)); n_sbml += 1

    # 2. Central dogma per gene (G, R, R_d, RP, RP_f, RB, RB_p, RB_pe, RB_cp, P, C_P)
    CD_CHANNELS = {"G", "R", "R_d", "RP", "RP_f",
                   "RB", "RB_p", "RB_pe", "RB_cp", "P", "C_P"}
    gene_cd = {}
    for i, name in enumerate(species_names):
        pre, loc = parse_species(name)
        if pre in CD_CHANNELS:
            gene_cd.setdefault(_locus_key(loc), []).append(i)
    for locus, members in gene_cd.items():
        for i in members:
            for j in members:
                if i != j and (i, j) not in edges:
                    edges.add((i, j)); n_cd += 1

    # 3. Enzyme → flux (from kinetics' reaction→enzyme map)
    if kinetics is not None and "enzymes" in kinetics:
        for rxn_id, enz in kinetics["enzymes"].items():
            flux_name = f"F_{rxn_id}"
            if enz in name_to_idx and flux_name in name_to_idx:
                a, b = name_to_idx[enz], name_to_idx[flux_name]
                if (a, b) not in edges:
                    edges.add((a, b)); n_enz += 1
                if (b, a) not in edges:
                    edges.add((b, a)); n_enz += 1

    # 4. Subunit → complex (from complex_formation)
    if complexes is not None:
        for cx in complexes["complexes"]:
            cname = cx["name"]
            if cname not in name_to_idx:
                continue
            cidx = name_to_idx[cname]
            for gene_id, _stoi in cx["subunits"]:
                for cand in (f"P_{gene_id}", f"P_{gene_id.zfill(4)}", f"P_{gene_id.lstrip('0')}"):
                    if cand in name_to_idx:
                        a = name_to_idx[cand]
                        if (a, cidx) not in edges:
                            edges.add((a, cidx)); n_cplx += 1
                        if (cidx, a) not in edges:
                            edges.add((cidx, a)); n_cplx += 1
                        break

    # 5. NEW v11: Protein ↔ metabolite regulatory binding
    if protein_metabolites:
        for rec in protein_metabolites:
            prot = rec["protein"]
            if prot not in name_to_idx:
                continue
            i_p = name_to_idx[prot]
            for met in rec["metabolites"]:
                if met not in name_to_idx:
                    continue
                i_m = name_to_idx[met]
                if (i_p, i_m) not in edges:
                    edges.add((i_p, i_m)); n_pm += 1
                if (i_m, i_p) not in edges:
                    edges.add((i_m, i_p)); n_pm += 1

    # 6. NEW v11: 50S ribosome assembly (substrate + intermediate → product)
    if largesubunit:
        for sub, inter, prod in largesubunit:
            for src_name in (sub, inter):
                if src_name in name_to_idx and prod in name_to_idx:
                    a, b = name_to_idx[src_name], name_to_idx[prod]
                    if (a, b) not in edges:
                        edges.add((a, b)); n_lsu += 1
                    if (b, a) not in edges:
                        edges.add((b, a)); n_lsu += 1

    # 7. Self-loops
    n_self = 0
    for i in range(n):
        if (i, i) not in edges:
            edges.add((i, i)); n_self += 1

    edge_list = list(edges)
    edge_index  = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    print(f"[graph] full graph: {edge_index.shape[1]:,} edges  "
          f"(SBML {n_sbml:,} + central-dogma {n_cd:,} + enzyme-flux {n_enz:,} + "
          f"subunit-complex {n_cplx:,} + prot-metab {n_pm:,} + LSU {n_lsu:,} + "
          f"self {n_self:,})")
    return edge_index, edge_weight


def build_stoich_matrix(sbml, species_names):
    """Build the stoichiometric matrix S restricted to SBML species present
    in our trajectory.  Used by the PINN head's mass-balance bridge.

    Returns:
        sbml_mask     : (S,) bool   - True where the species is in S
        sbml_indices  : (n_sbml,)   - full indices of SBML species
        stoich_matrix : (n_sbml, n_rxn) float — Δx_sbml = S @ v
    Or (None, None, None) if no SBML overlap.
    """
    if sbml is None:
        return None, None, None
    name_to_idx = {nm: i for i, nm in enumerate(species_names)}
    present = [s for s in sbml["species"].keys() if s in name_to_idx]
    if not present:
        return None, None, None
    sbml_idx_map = {s: i for i, s in enumerate(present)}
    full_indices = [name_to_idx[s] for s in present]
    n_sbml, n_rxn = len(present), len(sbml["reactions"])
    S_mat = torch.zeros(n_sbml, n_rxn, dtype=torch.float32)
    for j, rxn in enumerate(sbml["reactions"]):
        for sid, stoi in rxn["reactants"]:
            if sid in sbml_idx_map:
                S_mat[sbml_idx_map[sid], j] -= stoi
        for sid, stoi in rxn["products"]:
            if sid in sbml_idx_map:
                S_mat[sbml_idx_map[sid], j] += stoi
    mask = torch.zeros(len(species_names), dtype=torch.bool)
    mask[torch.tensor(full_indices)] = True
    print(f"[pinn] stoich matrix: {n_sbml} SBML species × {n_rxn} reactions  "
          f"({mask.sum().item()}/{len(species_names)} species mass-balanced)")
    return mask, torch.tensor(full_indices, dtype=torch.long), S_mat


# ── v13.9 MetabolismCore: bi-bi rate law on SBML reactions ───────────────────

NA_AVOGADRO       = 6.02214076e23
SYN3A_VOLUME_L    = 3.35e-17   # v14.4: corrected per Thornburg 2026 — sphere of 200 nm radius
                               #        (was 2.0e-16, ~6× too large).  Doubles to ~6.55e-17 at division.


def build_metabolism_tensors(sbml, kinetics, species_names,
                             volume_l=SYN3A_VOLUME_L,
                             gibbs=None,
                             gibbs_threshold_kj=GIBBS_DG_THRESHOLD_KJ):
    """Pre-compute the tensors MetabolismCore needs.

    Wires every SBML reaction that has (a) a k_cat_fwd, (b) an enzyme listed
    AND present in the trajectory, (c) at least one substrate present in the
    trajectory.  Missing K_m values fall back to the global median K_m so the
    rate law is still defined; track how many fell back vs measured.

    v14 day 1: when `gibbs` (a dict reaction_id -> ΔG° in kJ/mol) is provided,
    any reaction with ΔG° < -gibbs_threshold_kj gets its k_cat_rev forced to 0
    — strongly exergonic reactions become irreversible-forward by construction.
    Endergonic reactions are left as-is (they may run forward via energetic
    coupling we're not modelling explicitly).

    Returns dict with the tensor buffers, the species coverage mask, and the
    wired / skipped reaction lists.  None if nothing wirable.
    """
    def lookup_gibbs(rxn_id):
        if not gibbs:
            return None
        if rxn_id in gibbs:
            return gibbs[rxn_id]
        bare = rxn_id[2:] if rxn_id.startswith("R_") else rxn_id
        return gibbs.get(bare)
    if sbml is None or kinetics is None or "params" not in kinetics:
        return None

    name_to_idx = {nm: i for i, nm in enumerate(species_names)}
    S = len(species_names)

    all_kms = [v for p in kinetics["params"].values()
               for v in p["km"].values() if v and v > 0]
    median_km = float(np.median(all_kms)) if all_kms else 1.0

    wired, skipped = [], []
    for rxn in sbml["reactions"]:
        rid = rxn["id"]
        # SBML reaction IDs are 'R_PGI', kinetic_params.xlsx uses bare 'PGI'.
        kp = (kinetics["params"].get(rid)
              or kinetics["params"].get(rid[2:] if rid.startswith("R_") else rid))
        if kp is None:
            skipped.append((rid, "no_kinetics_row"))
            continue
        if kp["kcat_fwd"] is None:
            skipped.append((rid, "no_kcat_fwd"))
            continue
        if kp["enzyme"] is None:
            skipped.append((rid, "no_enzyme"))
            continue
        if kp["enzyme"] not in name_to_idx:
            skipped.append((rid, f"enzyme_not_in_traj:{kp['enzyme']}"))
            continue
        subs = [(sid, st) for sid, st in rxn["reactants"] if sid in name_to_idx]
        prods = [(sid, st) for sid, st in rxn["products"] if sid in name_to_idx]
        if not subs:
            skipped.append((rid, "no_substrates_in_traj"))
            continue
        wired.append((rxn, kp, subs, prods))

    if not wired:
        print("[metabcore] no reactions wirable — kinetics data too sparse")
        return None

    R = len(wired)
    MAX_S = max(len(s) for _, _, s, _ in wired)
    MAX_P = max((len(p) for _, _, _, p in wired), default=1) or 1

    enzyme_idx  = torch.zeros(R, dtype=torch.long)
    kcat_fwd    = torch.zeros(R, dtype=torch.float32)
    kcat_rev    = torch.zeros(R, dtype=torch.float32)
    sub_idx     = torch.zeros(R, MAX_S, dtype=torch.long)
    sub_km      = torch.full((R, MAX_S), float("inf"), dtype=torch.float32)
    sub_stoich  = torch.zeros(R, MAX_S, dtype=torch.float32)
    prod_idx    = torch.zeros(R, MAX_P, dtype=torch.long)
    prod_km     = torch.full((R, MAX_P), float("inf"), dtype=torch.float32)
    prod_stoich = torch.zeros(R, MAX_P, dtype=torch.float32)

    n_km_measured, n_km_fallback = 0, 0
    n_gibbs_clamped = 0           # reactions forced irreversible by ΔG° sign
    n_gibbs_unknown = 0           # wired reactions with no ΔG° available
    for j, (rxn, kp, subs, prods) in enumerate(wired):
        enzyme_idx[j] = name_to_idx[kp["enzyme"]]
        kcat_fwd[j]   = float(kp["kcat_fwd"])
        kcat_rev[j]   = float(kp["kcat_rev"]) if kp["kcat_rev"] is not None else 0.0
        # v14 day 1: thermodynamic sign clamp
        dg = lookup_gibbs(rxn["id"])
        if dg is None:
            n_gibbs_unknown += 1
        elif dg < -gibbs_threshold_kj and kcat_rev[j] > 0:
            kcat_rev[j] = 0.0       # strongly exergonic → irreversible forward
            n_gibbs_clamped += 1
        for k, (sid, st) in enumerate(subs):
            sub_idx[j, k]    = name_to_idx[sid]
            sub_stoich[j, k] = float(st)
            km = kp["km"].get(sid)
            if km is not None and km > 0:
                sub_km[j, k] = km; n_km_measured += 1
            else:
                sub_km[j, k] = median_km; n_km_fallback += 1
        for k, (sid, st) in enumerate(prods):
            prod_idx[j, k]    = name_to_idx[sid]
            prod_stoich[j, k] = float(st)
            km = kp["km"].get(sid)
            if km is not None and km > 0:
                prod_km[j, k] = km; n_km_measured += 1
            else:
                prod_km[j, k] = median_km; n_km_fallback += 1

    # Stoichiometric matrix (S × R): Δstate = stoich @ v
    stoich = torch.zeros(S, R, dtype=torch.float32)
    for j, (rxn, _, subs, prods) in enumerate(wired):
        for sid, st in subs:
            stoich[name_to_idx[sid], j] -= float(st)
        for sid, st in prods:
            stoich[name_to_idx[sid], j] += float(st)
    coverage_mask = stoich.abs().sum(dim=1) > 0

    print(f"[metabcore] wired {R}/{len(sbml['reactions'])} reactions; "
          f"skipped {len(skipped)} "
          f"(top reasons: {dict((r, sum(1 for _, x in skipped if x.startswith(r))) for r in set(x.split(':')[0] for _, x in skipped))})")
    print(f"[metabcore] K_m: {n_km_measured} measured, "
          f"{n_km_fallback} fallback (median={median_km:.3g})")
    print(f"[metabcore] ΔG° clamp: {n_gibbs_clamped} reactions forced irreversible-forward "
          f"({n_gibbs_unknown} have no ΔG° data, left reversible)")
    print(f"[metabcore] species coverage: {int(coverage_mask.sum())}/{S}")

    # v14 day 3: locate the ATP species for the energy ledger
    atp_idx = name_to_idx.get(ATP_SPECIES_NAME)
    if atp_idx is not None and coverage_mask[atp_idx]:
        net_atp_per_rxn = stoich[atp_idx]   # (R,) net ATP per reaction
        n_atp_producers = int((net_atp_per_rxn > 0).sum())
        n_atp_consumers = int((net_atp_per_rxn < 0).sum())
        print(f"[metabcore] ATP ledger: {n_atp_producers} producing + "
              f"{n_atp_consumers} consuming reactions of {R} wired "
              f"(net stoich coefficients available)")
    else:
        atp_idx = None
        print(f"[metabcore] ATP ledger: '{ATP_SPECIES_NAME}' not in covered species "
              f"— ledger disabled, ATP loss won't fire")

    return {
        "enzyme_idx": enzyme_idx,
        "kcat_fwd": kcat_fwd, "kcat_rev": kcat_rev,
        "sub_idx": sub_idx, "sub_km": sub_km, "sub_stoich": sub_stoich,
        "prod_idx": prod_idx, "prod_km": prod_km, "prod_stoich": prod_stoich,
        "stoich_matrix": stoich, "coverage_mask": coverage_mask,
        "wired_reactions": [r["id"] for r, _, _, _ in wired],
        "skipped_reactions": skipped,
        "median_km": median_km, "volume_l": volume_l,
        "atp_idx": atp_idx,
    }


class MetabolismCore(nn.Module):
    """Bi-bi (random-order MM) rate law for SBML reactions — direct
    replacement for the PINN head's neural flux prediction.

    All k_cat, K_m, and enzyme assignments come from kinetic_params.xlsx as
    frozen buffers; no learnable parameters by default.  Optionally
    `learnable_rates=True` makes log(k_cat) and log(K_m) trainable for a
    fine-tune-around-measured-values mode (not used in initial integration).

    Forward signature: state (B, S) [counts] -> (delta_state, fluxes)
      delta_state: (B, S) per-second count change for wired species
      fluxes:      (B, R) net flux per wired reaction (mM/s)
    """

    def __init__(self, tensors, learnable_rates=False, atp_idx=None):
        super().__init__()
        self.register_buffer("enzyme_idx",    tensors["enzyme_idx"])
        self.register_buffer("sub_idx",       tensors["sub_idx"])
        self.register_buffer("sub_stoich",    tensors["sub_stoich"])
        self.register_buffer("prod_idx",      tensors["prod_idx"])
        self.register_buffer("prod_stoich",   tensors["prod_stoich"])
        self.register_buffer("stoich_matrix", tensors["stoich_matrix"])
        self.register_buffer("coverage_mask", tensors["coverage_mask"])
        self.register_buffer("volume_l",
                             torch.tensor(tensors["volume_l"], dtype=torch.float32))
        # v14 day 3: ATP stoichiometry vector — net ATP yield per wired reaction
        if atp_idx is not None:
            self.register_buffer("atp_stoich", self.stoich_matrix[atp_idx].clone())
            self.has_atp = True
        else:
            self.has_atp = False

        if learnable_rates:
            self.log_kcat_fwd = nn.Parameter(torch.log(tensors["kcat_fwd"].clamp(min=1e-9)))
            self.log_kcat_rev = nn.Parameter(torch.log(tensors["kcat_rev"].clamp(min=1e-9)))
            self.log_sub_km   = nn.Parameter(torch.log(tensors["sub_km"].clamp(min=1e-9)))
            self.log_prod_km  = nn.Parameter(torch.log(tensors["prod_km"].clamp(min=1e-9)))
        else:
            self.register_buffer("kcat_fwd", tensors["kcat_fwd"])
            self.register_buffer("kcat_rev", tensors["kcat_rev"])
            self.register_buffer("sub_km",   tensors["sub_km"])
            self.register_buffer("prod_km",  tensors["prod_km"])
        self.learnable_rates = learnable_rates
        self.R = tensors["kcat_fwd"].shape[0]
        # v15.2: stash wired reaction names so post-hoc diagnostics (TUR, F-W
        # action, provenance audit) can align ΔG° / per-reaction stats.
        self.wired_reactions = list(tensors.get("wired_reactions", []))

    def _params(self):
        if self.learnable_rates:
            return (torch.exp(self.log_kcat_fwd), torch.exp(self.log_kcat_rev),
                    torch.exp(self.log_sub_km),  torch.exp(self.log_prod_km))
        return self.kcat_fwd, self.kcat_rev, self.sub_km, self.prod_km

    def forward(self, state, dt=1.0, volume_l=None):
        """state: (B, S) raw molecular counts.  dt: seconds.
        volume_l: scalar or (B,) tensor of cell volumes in litres.
                  None → constant SYN3A_VOLUME_L buffer (back-compat).
        Returns (delta_state, fluxes)."""
        kcat_fwd, kcat_rev, sub_km, prod_km = self._params()
        B = state.shape[0]

        # Volume → (B, 1) for broadcast over species
        if volume_l is None:
            v_l = self.volume_l.expand(B).unsqueeze(-1)
        elif volume_l.ndim == 0:
            v_l = volume_l.expand(B).unsqueeze(-1)
        else:
            v_l = volume_l.unsqueeze(-1)

        # Count → mM concentration (NA · V_l in litres → concentration in mol/L; ×1e3 to mM)
        conc = state / (NA_AVOGADRO * v_l) * 1e3

        sub_conc  = conc[:, self.sub_idx].clamp(min=0.0)    # (B, R, MAX_S)
        prod_conc = conc[:, self.prod_idx].clamp(min=0.0)   # (B, R, MAX_P)
        enz_conc  = conc[:, self.enzyme_idx].clamp(min=0.0) # (B, R)

        sub_ratio  = sub_conc / sub_km
        prod_ratio = prod_conc / prod_km

        # Numerator: Π(S/Km)^stoich (padding has stoich=0 → ratio^0 = 1)
        num_fwd = sub_ratio.clamp(min=1e-30).pow(self.sub_stoich).prod(dim=-1)
        num_rev = prod_ratio.clamp(min=1e-30).pow(self.prod_stoich).prod(dim=-1)

        # Denominator: Π(1+S/Km)^stoich + Π(1+P/Km)^stoich - 1
        den_sub  = (1.0 + sub_ratio).pow(self.sub_stoich).prod(dim=-1)
        den_prod = (1.0 + prod_ratio).pow(self.prod_stoich).prod(dim=-1)
        den = den_sub + den_prod - 1.0

        v = enz_conc * (kcat_fwd * num_fwd - kcat_rev * num_rev) / den.clamp(min=1e-12)

        # Δconc = stoich @ v → (B, S); back to count via the same V_l
        delta_conc  = torch.einsum("sr,br->bs", self.stoich_matrix, v)
        delta_state = delta_conc * dt * (NA_AVOGADRO * v_l) / 1e3

        return delta_state, v

    def compute_atp_rate(self, fluxes, volume_l=None):
        """v14 day 3: net ATP production rate over wired reactions.

        fluxes:    (B, R) bi-bi fluxes from forward() (mM/s)
        volume_l:  scalar or (B,) volume; None → use self.volume_l buffer
        Returns:   (B,) net ATP rate in molecules/s.  None if ATP not tracked.
        """
        if not self.has_atp:
            return None
        atp_rate_mm = torch.einsum("br,r->b", fluxes, self.atp_stoich.to(fluxes.dtype))
        B = fluxes.shape[0]
        if volume_l is None:
            v_l = self.volume_l.expand(B)
        elif volume_l.ndim == 0:
            v_l = volume_l.expand(B)
        else:
            v_l = volume_l
        return atp_rate_mm * (NA_AVOGADRO * v_l) / 1e3


# ── v13.9 VolumeCore: dynamic cell volume from membrane-lipid count ──────────

# Heuristic: any SBML species whose name starts with one of these is a
# membrane component for our purposes. Catches phosphatidic acids, PE, PG,
# PC, PS, cardiolipins, cholesterol, glycerol-phosphate intermediates.
LIPID_PREFIXES = ("M_pa_", "M_pe_", "M_pg_", "M_pc_", "M_ps_",
                  "M_clpn", "M_chsterol", "M_1ag3p", "M_dag", "M_pgp",
                  "M_cdpdag", "M_glyc3p")


def build_volume_core(species_names, raw_counts_t0,
                      base_volume_l=SYN3A_VOLUME_L):
    """Build a VolumeCore from the species list + t=0 counts.

    Returns a VolumeCore module, or None if no lipid species are present
    in this trajectory (caller falls back to constant volume in MetabolismCore).
    """
    lipid_idx = [i for i, n in enumerate(species_names)
                 if any(n.startswith(p) for p in LIPID_PREFIXES)]
    if not lipid_idx:
        print(f"[volumecore] no lipid species found in {len(species_names)} "
              f"species — falling back to constant V={base_volume_l:.2e} L")
        return None
    # initial total: mean over training trajectories at t=0
    arr = np.asarray(raw_counts_t0)
    initial_total = float(arr[:, lipid_idx].sum(axis=1).mean())
    if initial_total <= 0:
        print(f"[volumecore] {len(lipid_idx)} lipid species but t=0 total=0 — "
              f"falling back to constant V={base_volume_l:.2e} L")
        return None
    print(f"[volumecore] tracking {len(lipid_idx)} lipid species, "
          f"t=0 mean total = {initial_total:.3g}, base V = {base_volume_l:.2e} L")
    return VolumeCore(lipid_idx, initial_total, base_volume_l)


class VolumeCore(nn.Module):
    """Proxies cell volume from membrane-lipid count:
        V_L(t) = V_0 * (lipid_total(t) / lipid_total_0)

    Crude — upstream's V follows surface area which follows lipid synthesis.
    A linear scaling skips the SA→V geometry but captures the doubling.
    """

    def __init__(self, lipid_indices, initial_lipid_total, base_volume_l):
        super().__init__()
        self.register_buffer("lipid_indices",
                             torch.as_tensor(lipid_indices, dtype=torch.long))
        self.register_buffer("initial_total",
                             torch.tensor(float(initial_lipid_total),
                                          dtype=torch.float32))
        self.register_buffer("base_volume_l",
                             torch.tensor(float(base_volume_l),
                                          dtype=torch.float32))

    def forward(self, state):
        """state: (B, S) counts → (B,) volume in litres."""
        lipid_total = state[:, self.lipid_indices].sum(dim=1).clamp(min=1.0)
        return self.base_volume_l * (lipid_total / self.initial_total)


# ── v13.9 CentralDogmaCore: first-order tx / tl / mRNA-deg / protein-deg ─────

# Literature defaults — tuned to give plausible steady states (mRNA ~10/gene,
# protein ~1000/mRNA).  Per-gene refinement (from syn3A.gb gene lengths)
# is a future improvement.
K_TX_DEFAULT      = 0.06               # transcription initiation rate per gene copy (/s)
K_TL_DEFAULT      = 2.0e-3             # translation initiation rate per mRNA (/s) — fallback when no ribosome pool found
T_HALF_MRNA_S     = 120.0              # mRNA half-life (~2 min, Syn3A literature)
T_HALF_PROTEIN_S  = 36_000.0           # protein half-life (~10 h)

# v14 day 2: shared-ribosome-pool model
RIBOSOME_PREFIXES = ("RB_", "RPM_", "C_ribosome", "ribosome")
K_M_TOTAL_MRNA    = 100.0              # half-saturation of the total-mRNA bottleneck (legacy; superseded by ribosome scaling)
K_PER_RIBO        = 1.5e-3             # translation initiation rate per ribosome (/s) (legacy)

# v14.3: per-gene calibration (Thornburg 2026 § "Transcription of mRNA and tRNA")
MIN_PROMOTER_STRENGTH = 0.05           # floor for genes with no observed expression — still get some transcription
AVG_PROTEIN_FALLBACK  = 180.0          # paper-stated average across 455 mRNA-coding genes

# v14.6: global tuning knobs for CD rates.  Per-gene calibration sets steady-state
# to match initial counts, but the LGNN often over-predicts gene replication
# (G doubling), which CD then amplifies through mRNA and protein.  These knobs
# scale the per-gene rates so we match upstream's observed fold-change.
CD_TRANSCRIPTION_SCALE = 1.0           # multiplier on every k_tx_per_gene
CD_TRANSLATION_SCALE   = 0.85          # v14.8: was 0.7 (overshot — protein fold dropped to 1.00× when target was ~2.07×).
                                       # 0.85 should land protein fold closer to upstream without freezing it.
CD_BLEND_WEIGHT        = 0.1           # v14.9: was implicit-1.0 (full override).  Now CD adds residual:
                                       #   x_next += BLEND × (x_cd - x_input)
                                       # 0.1 = LGNN trusted to predict trajectory, CD nudges 10%.


# v15.4 [5/6]: per-class translation scales.  One global CD_TRANSLATION_SCALE
# cannot fit ribosomal vs membrane vs metabolic proteins (provenance audit:
# 8/15 worst-predicted species are CD-covered; ribosome fold 1.27 vs upstream
# 1.85 roots in ribosomal-protein under-production).  Each gene's k_tl is
# multiplied by its class scale, classified from the gene-table `product` text.
USE_CD_CLASS_SCALES = True
CD_CLASS_TL_SCALES  = {
    "ribosomal":  1.30,   # ribosomal proteins — under-produced, scale up
    "membrane":   0.85,   # membrane / transport proteins
    "metabolic":  0.85,   # enzymes
    "other":      0.85,   # default / unclassified (== old global scale)
}


def load_gene_products(csv_path):
    """v15.4 [5/6]: {locus_num_str: product_str} from syn3a_gene_table.csv."""
    if not HAS_PANDAS:
        return {}
    try:
        df = pd.read_csv(csv_path)
        out = {}
        for _, row in df.iterrows():
            tag = str(row.get("locus_tag", ""))
            if "_" in tag:
                out[tag.split("_")[1]] = str(row.get("product", ""))
        return out
    except Exception as e:
        print(f"[gene_products] load failed ({e})")
        return {}


def _classify_gene_for_cd(product):
    """v15.4 [5/6]: map a gene product string to a CD translation class."""
    p = (product or "").lower()
    if not p:
        return "other"
    if ("ribosomal protein" in p or "ribosomal subunit" in p
            or "30s " in p or "50s " in p):
        return "ribosomal"
    if ("membrane" in p or "transporter" in p or "permease" in p
            or "translocase" in p or "abc transport" in p):
        return "membrane"
    if any(k in p for k in ("synthase", "synthetase", "kinase", "transferase",
            "dehydrogenase", "reductase", "hydrolase", "ligase", "polymerase",
            "phosphatase", "isomerase", "lyase", "oxidase", "carboxylase",
            "aminotransferase", "nuclease", "phosphorylase")):
        return "metabolic"
    return "other"


def build_central_dogma_tensors(species_names, initial=None, raw_counts_t0=None,
                                product_map=None,
                                t_half_mrna=T_HALF_MRNA_S,
                                t_half_prot=T_HALF_PROTEIN_S):
    """Pair (G, R, P) species by gene locus, with v14.3 per-gene calibration.

    For each locus that has *all three* of G_<locus>, R_<locus>, P_<locus>
    species in the trajectory (preferring the _C1 chromosome-copy variant
    where multiple exist), pack the indices.

    v14.3: per-gene k_tx_g and k_tl_g are derived from observed initial
    counts (Thornburg 2026 promoter-strength formulation, S = P_init / 180):
      - k_tx_g = R_init_g * k_deg_mrna     (steady-state R_ss = R_init_g when G=1)
        if observed mRNA xlsx is available; else
      - k_tx_g = (P_init_g / 180) * k_deg_mrna  (promoter-strength proxy)
      - k_tl_g = P_init_g * k_deg_prot / max(R_init_g, 1) so steady-state P_ss = P_init_g
    Floors apply (MIN_PROMOTER_STRENGTH * k_deg_mrna) when no observed counts.

    Ribosome species are auto-identified by prefix.  CentralDogmaCore scales
    all translation by (ribo_total / initial_ribo_total) so ribosome KO
    halts all translation.

    Returns dict with index tensors + per-gene rate tensors + coverage_mask.
    """
    S = len(species_names)
    by_locus = {}
    for i, name in enumerate(species_names):
        pre, full_locus = parse_species(name)
        if pre not in ("G", "R", "P") or not full_locus:
            continue
        locus = _locus_key(full_locus)
        if not locus:
            continue
        # Prefer _C1 variant; first match otherwise.  Don't overwrite a
        # _C1 with a non-_C1 hit.
        d = by_locus.setdefault(locus, {})
        existing = d.get(pre)
        if existing is None:
            d[pre] = (i, full_locus)
        elif full_locus.endswith("_C1") and not existing[1].endswith("_C1"):
            d[pre] = (i, full_locus)

    triples = [(d["G"][0], d["R"][0], d["P"][0], locus)
               for locus, d in by_locus.items()
               if all(k in d for k in ("G", "R", "P"))]
    if not triples:
        print(f"[cdcore] no genes with G+R+P species found in {S} species — disabled")
        return None

    gene_idx = torch.tensor([t[0] for t in triples], dtype=torch.long)
    mrna_idx = torch.tensor([t[1] for t in triples], dtype=torch.long)
    prot_idx = torch.tensor([t[2] for t in triples], dtype=torch.long)
    cov_mask = torch.zeros(S, dtype=torch.bool)
    cov_mask[mrna_idx] = True
    cov_mask[prot_idx] = True

    # v14 day 2: find ribosome species for the shared translation pool
    ribo_idx = [i for i, n in enumerate(species_names)
                if any(n.startswith(p) for p in RIBOSOME_PREFIXES)]
    initial_ribo_total = 1.0
    if ribo_idx and raw_counts_t0 is not None:
        arr = np.asarray(raw_counts_t0)
        initial_ribo_total = max(1.0, float(arr[:, ribo_idx].sum(axis=1).mean()))

    print(f"[cdcore] {len(triples)} genes wired (G+R+P all present); "
          f"coverage = {int(cov_mask.sum())} species (mRNA + protein, "
          f"genes left to LGNN)")
    if ribo_idx:
        print(f"[cdcore] ribosome pool: {len(ribo_idx)} species "
              f"(prefixes {RIBOSOME_PREFIXES}); t=0 mean total = "
              f"{initial_ribo_total:.3g} — translation scales with ribo/ribo_0")
    else:
        print(f"[cdcore] no ribosome species found — translation uncoupled from ribosomes")

    # v14.3: per-gene calibration from initial conditions
    k_deg_mrna_s = np.log(2) / float(t_half_mrna)
    k_deg_prot_s = np.log(2) / float(t_half_prot)
    proteins = (initial or {}).get("proteins", {})
    mRNAs    = (initial or {}).get("mRNAs", {})
    avg_protein = (float(np.mean([v for v in proteins.values() if v > 0]))
                   if proteins else AVG_PROTEIN_FALLBACK)
    if avg_protein <= 0:
        avg_protein = AVG_PROTEIN_FALLBACK

    k_tx_per_gene = np.zeros(len(triples), dtype=np.float32)
    k_tl_per_gene = np.zeros(len(triples), dtype=np.float32)
    n_direct, n_proxy, n_default = 0, 0, 0
    for j, (_, _, _, locus) in enumerate(triples):
        tag = f"JCVISYN3A_{locus}"
        r_init = mRNAs.get(tag)
        p_init = proteins.get(tag)
        # transcription rate: prefer direct mRNA xlsx, else promoter-strength proxy
        if r_init is not None and r_init > 0:
            k_tx_per_gene[j] = float(r_init) * k_deg_mrna_s
            n_direct += 1
            r_target = float(r_init)
        elif p_init is not None and p_init > 0:
            strength = max(p_init / avg_protein, MIN_PROMOTER_STRENGTH)
            k_tx_per_gene[j] = strength * k_deg_mrna_s
            n_proxy += 1
            r_target = strength
        else:
            k_tx_per_gene[j] = MIN_PROMOTER_STRENGTH * k_deg_mrna_s
            n_default += 1
            r_target = MIN_PROMOTER_STRENGTH
        # translation rate: calibrate so steady-state P_ss = P_init given R_ss = r_target
        if p_init is not None and p_init > 0:
            k_tl_per_gene[j] = float(p_init) * k_deg_prot_s / max(r_target, 1e-6)
        else:
            k_tl_per_gene[j] = K_TL_DEFAULT

    # v14.6: apply global scaling knobs to compensate for LGNN over-prediction
    # of gene replication that CD amplifies through mRNA → protein.
    k_tx_per_gene = k_tx_per_gene * CD_TRANSCRIPTION_SCALE
    # v15.4 [5/6]: per-class translation scaling (falls back to the global scale)
    if USE_CD_CLASS_SCALES and product_map:
        tl_scales = np.empty(len(triples), dtype=np.float32)
        cls_counts = {}
        for j, (_, _, _, locus) in enumerate(triples):
            cls = _classify_gene_for_cd(product_map.get(locus, ""))
            tl_scales[j] = CD_CLASS_TL_SCALES.get(cls, CD_TRANSLATION_SCALE)
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
        k_tl_per_gene = k_tl_per_gene * tl_scales
        print("[cdcore] per-class k_tl scales: " + ", ".join(
            f"{k}x{CD_CLASS_TL_SCALES.get(k, CD_TRANSLATION_SCALE)}({v})"
            for k, v in sorted(cls_counts.items())))
    else:
        k_tl_per_gene = k_tl_per_gene * CD_TRANSLATION_SCALE

    print(f"[cdcore] per-gene calibration: {n_direct} from mRNA xlsx, "
          f"{n_proxy} from promoter-strength proxy (P/{avg_protein:.0f}), "
          f"{n_default} default-floor")
    print(f"[cdcore]   scales: k_tx × {CD_TRANSCRIPTION_SCALE}, "
          f"k_tl × {CD_TRANSLATION_SCALE}  "
          f"(tune CD_TRANSLATION_SCALE to match protein fold-change)")
    print(f"[cdcore]   k_tx /s: median {np.median(k_tx_per_gene):.3e}, "
          f"range [{k_tx_per_gene.min():.3e}, {k_tx_per_gene.max():.3e}]")
    print(f"[cdcore]   k_tl /s: median {np.median(k_tl_per_gene):.3e}, "
          f"range [{k_tl_per_gene.min():.3e}, {k_tl_per_gene.max():.3e}]")

    return {
        "gene_idx": gene_idx, "mrna_idx": mrna_idx, "prot_idx": prot_idx,
        "coverage_mask": cov_mask, "loci": [t[3] for t in triples],
        "n_genes": len(triples),
        "ribosome_idx": torch.tensor(ribo_idx, dtype=torch.long) if ribo_idx
                        else torch.zeros(0, dtype=torch.long),
        "initial_ribo_total": initial_ribo_total,
        "k_tx_per_gene": torch.as_tensor(k_tx_per_gene, dtype=torch.float32),
        "k_tl_per_gene": torch.as_tensor(k_tl_per_gene, dtype=torch.float32),
    }


class CentralDogmaCore(nn.Module):
    """First-order tx / tl / decay per gene locus.

    For each gene g with all of (G_g, R_g, P_g) species in the trajectory:
        d(R_g)/dt = k_tx · G_g       - k_deg_mRNA · R_g
        d(P_g)/dt = k_tl · R_g       - k_deg_prot · P_g

    Overrides the LGNN's prediction for mRNA + protein species (genes
    themselves are left to the LGNN — DNA replication is ReplicationCore
    later).  Frozen-buffer rate constants from literature defaults; can
    later be made per-gene from syn3A.gb gene lengths.

    NOTE: this is a deliberate simplification of upstream's GIP_rates.py
    which couples to NTP/aa/ribosome pools.  We're skipping that coupling
    in v1 — the model has to live with literature-average rates for now.
    """

    def __init__(self, tensors,
                 t_half_mrna=T_HALF_MRNA_S, t_half_prot=T_HALF_PROTEIN_S):
        super().__init__()
        self.register_buffer("gene_idx",      tensors["gene_idx"])
        self.register_buffer("mrna_idx",      tensors["mrna_idx"])
        self.register_buffer("prot_idx",      tensors["prot_idx"])
        self.register_buffer("coverage_mask", tensors["coverage_mask"])
        self.register_buffer("k_deg_mrna",
                             torch.tensor(np.log(2) / float(t_half_mrna),
                                          dtype=torch.float32))
        self.register_buffer("k_deg_prot",
                             torch.tensor(np.log(2) / float(t_half_prot),
                                          dtype=torch.float32))
        # v14.3: per-gene calibrated rates from initial-condition observation
        self.register_buffer("k_tx_per_gene", tensors["k_tx_per_gene"])
        self.register_buffer("k_tl_per_gene", tensors["k_tl_per_gene"])
        # Ribosome pool: translation scales linearly with (ribo_total / ribo_total_initial),
        # so KO of all ribosomes -> exactly zero translation.
        ribo_idx = tensors.get("ribosome_idx", torch.zeros(0, dtype=torch.long))
        self.register_buffer("ribosome_idx", ribo_idx)
        self.register_buffer("initial_ribo_total",
                             torch.tensor(float(tensors.get("initial_ribo_total", 1.0)),
                                          dtype=torch.float32))
        self.has_ribosome_cap = ribo_idx.numel() > 0
        self.n_genes = tensors["n_genes"]

    def forward(self, state, dt=1.0):
        """state: (B, S) counts.  Returns delta_state (B, S) counts."""
        B = state.shape[0]
        G = state[:, self.gene_idx].clamp(min=0.0)
        R = state[:, self.mrna_idx].clamp(min=0.0)
        P = state[:, self.prot_idx].clamp(min=0.0)

        # v14.3: per-gene calibrated transcription rate
        dR = (self.k_tx_per_gene * G - self.k_deg_mrna * R) * dt

        # v14.3: per-gene calibrated translation, scaled by ribosome availability
        if self.has_ribosome_cap:
            ribo_total = state[:, self.ribosome_idx].sum(dim=1).clamp(min=0.0)   # (B,)
            ribo_factor = (ribo_total / self.initial_ribo_total).unsqueeze(-1)    # (B, 1)
            r_tl = self.k_tl_per_gene * R * ribo_factor                           # (B, n_genes)
        else:
            r_tl = self.k_tl_per_gene * R
        dP = (r_tl - self.k_deg_prot * P) * dt

        delta = torch.zeros_like(state)
        delta = delta.scatter_add(
            1, self.mrna_idx.unsqueeze(0).expand(B, -1), dR)
        delta = delta.scatter_add(
            1, self.prot_idx.unsqueeze(0).expand(B, -1), dP)
        return delta


# ── v13.9 AssemblyCore: mass-action complex assembly ──────────────────────────

ASSEMBLY_K_ON_DEFAULT = 1.0e-5     # /s per molecule^stoich, ballpark for protein-protein
ASSEMBLY_SAFETY_FRAC  = 0.5        # cap rate so it can't drain >50% of smallest pool/dt
ASM_BLEND_WEIGHT      = 0.1           # v15.4 [6/6]: additive-residual weight for AssemblyCore (mirrors CD_BLEND_WEIGHT). NOTE: only ~2/24 complexes wire (complex product must be a tracked species) so impact is limited; validate with a training run.


def _subunit_locus(gene_id):
    """'JCVISYN3A_0445' -> '0445'.  Returns None for unparseable."""
    if not gene_id or gene_id in ("nan", ""):
        return None
    return gene_id.rsplit("_", 1)[-1] if "_" in gene_id else gene_id


def build_assembly_tensors(complexes_data, species_names,
                           k_on=ASSEMBLY_K_ON_DEFAULT, lsu_chain=None):
    """Wire complex-assembly reactions from complex_formation.xlsx + optionally
    the 50S ribosome assembly chain from LargeSubunit.xlsx.

    From complex_formation.xlsx:
        Σ stoich_i · subunit_i  ->  1 · complex,
        rate = k_on · Π subunit_i^stoich_i

    From LargeSubunit.xlsx (3-tuples of substrate, intermediate, product):
        1 · substrate + 1 · intermediate  ->  1 · product
        rate = k_on · substrate · intermediate
    (these are sequential 50S biogenesis steps, each producing the next
    intermediate.)

    Subunit name resolution tries several `P_<locus>` variants; complex name
    resolution tries the bare name + `C_` prefix variants; LSU intermediates
    are looked up by exact name.
    """
    S = len(species_names)
    name_to_idx = {n: i for i, n in enumerate(species_names)}

    def find_complex(name):
        for cand in (name, f"C_{name}", name.replace("-", "_"),
                     f"C_{name.replace('-', '_')}"):
            if cand in name_to_idx:
                return name_to_idx[cand]
        return None

    def find_subunit(gene_id):
        loc = _subunit_locus(gene_id)
        if loc is None:
            return None
        for cand in (f"P_{loc}", f"P_{loc}_C1", f"P_{loc}_C2"):
            if cand in name_to_idx:
                return name_to_idx[cand]
        return None

    wired, skipped = [], []
    if complexes_data is not None and complexes_data.get("complexes"):
        for cx in complexes_data["complexes"]:
            nm = cx["name"]
            c_idx = find_complex(nm)
            if c_idx is None:
                skipped.append((nm, "complex_not_in_traj"))
                continue
            sub_idxs, sub_sts = [], []
            ok = True
            for gid, st in cx["subunits"]:
                si = find_subunit(gid)
                if si is None:
                    ok = False; break
                sub_idxs.append(si); sub_sts.append(float(st))
            if not ok or not sub_idxs:
                skipped.append((nm, "subunit_missing"))
                continue
            wired.append((c_idx, sub_idxs, sub_sts, nm))
    n_complex_wired = len(wired)

    # v13.9: 50S assembly chain — each step is substrate + intermediate → product
    n_lsu_wired = 0
    if lsu_chain:
        for sub, inter, prod in lsu_chain:
            si = name_to_idx.get(sub)
            ii = name_to_idx.get(inter)
            pi = name_to_idx.get(prod)
            if si is None or ii is None or pi is None:
                skipped.append((f"LSU:{sub}+{inter}→{prod}", "lsu_species_missing"))
                continue
            # 1 of each substrate, 1 product
            wired.append((pi, [si, ii], [1.0, 1.0], f"LSU:{prod}"))
            n_lsu_wired += 1

    if not wired:
        n_total = (len(complexes_data["complexes"]) if complexes_data else 0) + len(lsu_chain or [])
        print(f"[asmcore] no assembly reactions wirable "
              f"(0/{n_total}; e.g. skipped reasons: "
              f"{set(r for _, r in skipped[:5])})")
        return None

    R = len(wired)
    MAX_SUB = max(len(s) for _, s, _, _ in wired)
    complex_idx = torch.zeros(R, dtype=torch.long)
    sub_idx     = torch.zeros(R, MAX_SUB, dtype=torch.long)
    sub_stoich  = torch.zeros(R, MAX_SUB, dtype=torch.float32)
    k_on_t      = torch.full((R,), float(k_on), dtype=torch.float32)

    for j, (cidx, sidxs, ssts, _) in enumerate(wired):
        complex_idx[j] = cidx
        for k, (si, st) in enumerate(zip(sidxs, ssts)):
            sub_idx[j, k]    = si
            sub_stoich[j, k] = st

    # Stoichiometric matrix (S × R): -stoich on subunits, +1 on the complex
    stoich = torch.zeros(S, R, dtype=torch.float32)
    for j, (cidx, sidxs, ssts, _) in enumerate(wired):
        for si, st in zip(sidxs, ssts):
            stoich[si, j] -= st
        stoich[cidx, j] += 1.0
    coverage_mask = stoich.abs().sum(dim=1) > 0

    print(f"[asmcore] wired {R} assembly reactions "
          f"({n_complex_wired} from complex_formation, "
          f"{n_lsu_wired} from LargeSubunit chain); coverage = "
          f"{int(coverage_mask.sum())} species")
    return {
        "complex_idx": complex_idx, "sub_idx": sub_idx, "sub_stoich": sub_stoich,
        "k_on": k_on_t, "stoich_matrix": stoich, "coverage_mask": coverage_mask,
        "complex_names": [w[3] for w in wired],
        "skipped": skipped,
    }


class AssemblyCore(nn.Module):
    """Mass-action complex assembly.

    rate_j = k_on_j · Π (subunit_i^stoich_i)_j
    Δstate = stoich_matrix · rate · dt

    Includes a per-reaction rate cap so that no subunit pool can be drained
    by more than ASSEMBLY_SAFETY_FRAC per timestep — protects against Euler
    overshoot for fast reactions on a long (30s) step.
    """

    def __init__(self, tensors, safety_frac=ASSEMBLY_SAFETY_FRAC):
        super().__init__()
        self.register_buffer("complex_idx",   tensors["complex_idx"])
        self.register_buffer("sub_idx",       tensors["sub_idx"])
        self.register_buffer("sub_stoich",    tensors["sub_stoich"])
        self.register_buffer("k_on",          tensors["k_on"])
        self.register_buffer("stoich_matrix", tensors["stoich_matrix"])
        self.register_buffer("coverage_mask", tensors["coverage_mask"])
        self.safety_frac = float(safety_frac)
        self.R = tensors["k_on"].shape[0]

    def forward(self, state, dt=1.0):
        """state: (B, S) counts -> delta_state (B, S) counts."""
        B = state.shape[0]
        sub_counts = state[:, self.sub_idx].clamp(min=0.0)        # (B, R, MAX_SUB)

        # Mass-action rate: padding has stoich=0 → x^0=1 (kept), no padding flag needed.
        sub_terms = sub_counts.clamp(min=1e-30).pow(self.sub_stoich)
        rates = self.k_on * sub_terms.prod(dim=-1)                 # (B, R)

        # Safety cap: rate · dt · stoich ≤ safety_frac · count
        # ⇒ rate ≤ safety_frac · count / (dt · stoich)  per substrate slot.
        # Padding slots (stoich=0) must NOT constrain — set them to +inf.
        ratio = self.safety_frac * sub_counts / (dt * self.sub_stoich.clamp(min=1e-30))
        max_per_slot = torch.where(self.sub_stoich > 0, ratio,
                                    torch.full_like(ratio, float("inf")))
        max_per_rxn = max_per_slot.min(dim=-1).values
        rates = torch.minimum(rates, max_per_rxn).clamp(min=0.0)

        delta = torch.einsum("sr,br->bs", self.stoich_matrix, rates) * dt
        return delta


class _CfCGraphLayer(nn.Module):
    """Message-passing graph layer with CfC (closed-form continuous-time) update.

    Each step: aggregate neighbour messages → linear self-update → CfC gating.
    The CfC update form (h_new = σ(-gate)·A + σ(gate)·B with per-node A, B and
    bounded W via cfc_tau_min) is the canonical Liquid update from Hasani et
    al. and matches gnn_v2.py:_CfCAttentionGNNLayer line 167 in the M7 branch.
    """

    def __init__(self, hidden, n_nodes, cfc_tau_min=0.1):
        super().__init__()
        self.hidden = hidden
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.self_lin = nn.Linear(hidden, hidden)
        self.W_proj   = nn.Linear(hidden, hidden)
        self.b_proj   = nn.Linear(hidden, hidden)
        self.cfc_A    = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)
        self.cfc_B    = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)
        self.cfc_tau_min = cfc_tau_min
        self.norm     = nn.LayerNorm(hidden)

    def _forward_impl(self, h, edge_index, edge_weight):
        # h: (B, N, H);  edge_index: (2, E);  edge_weight: (E,)
        B, N, H = h.shape
        src, dst = edge_index[0], edge_index[1]
        h_src = h.index_select(1, src)        # (B, E, H)
        h_dst = h.index_select(1, dst)
        msg = self.msg_mlp(torch.cat([h_src, h_dst], dim=-1))
        # v13.3: cast back to h.dtype.  Under BF16 autocast msg is BF16, but
        # edge_weight is an fp32 buffer, so `msg * edge_weight` gets promoted
        # to fp32 — the index_add into a BF16 agg below then errors.
        msg = (msg * edge_weight.unsqueeze(0).unsqueeze(-1)).to(h.dtype)
        agg = torch.zeros(B, N, H, device=h.device, dtype=h.dtype)
        agg = agg.index_add(1, dst, msg)      # out-of-place: autograd-safe
        # Degree-normalise so high-degree nodes don't blow up
        ones = torch.ones_like(dst, dtype=h.dtype)
        deg  = torch.zeros(N, device=h.device, dtype=h.dtype)
        deg  = deg.index_add(0, dst, ones).clamp(min=1.0)
        agg  = agg / deg.unsqueeze(0).unsqueeze(-1)
        combined = agg + self.self_lin(h)
        # CfC gating (Hasani et al. Liquid form)
        W = self.W_proj(combined)
        W_clip = 1.0 / max(self.cfc_tau_min, 1e-6)
        gate = W.clamp(-W_clip, W_clip) + self.b_proj(combined)
        A = self.cfc_A.unsqueeze(0)            # (1, N, H)
        B_p = self.cfc_B.unsqueeze(0)
        h_new = torch.sigmoid(-gate) * A + torch.sigmoid(gate) * B_p
        return self.norm(h + h_new)

    def forward(self, h, edge_index, edge_weight):
        # v9.1: gradient checkpointing during training cuts memory ~2x by
        # re-computing the layer's intermediates during backward.  Important
        # for the K=64 rollout BPTT: without this, the (B, E=9395, H=64)
        # message tensor × 3 layers × 48 unrolled steps eats >30 GB and OOMs.
        if self.training and torch.is_grad_enabled():
            return ckpt.checkpoint(self._forward_impl, h, edge_index, edge_weight,
                                   use_reentrant=False)
        return self._forward_impl(h, edge_index, edge_weight)


class PINNHead(nn.Module):
    """v9: hardwired mass-balance head for SBML-covered species.

    The GNN's hidden state at SBML species is pooled to predict per-reaction
    log-space fluxes v_log.  The stoichiometric matrix S then maps v to
    species deltas via the log-space bridge:

        v_lin       = signed_expm1(v_log)             (linear-space rates)
        x_lin       = signed_expm1(x_signedlog)       (linear counts)
        Δx_lin      = S @ v_lin                       (mass balance, exact)
        x_next_lin  = clamp(x_lin + Δx_lin, min=0)
        x_next_sl   = signed_log1p(x_next_lin)        (back to signed-log)
        x_next_norm = (x_next_sl - lo) / span         (back to normalised)

    Mass conservation is guaranteed by construction for SBML species —
    no matter how badly v is predicted, S·v respects the stoichiometric
    ratios the simulator was built from.  Ported from the M7 branch's
    pinn_head.py with our (x-lo)/span normalisation accounted for.
    """

    def __init__(self, hidden, sbml_mask, sbml_indices, stoich_matrix,
                 lo_norm, span_norm, rate_clip=6.0):
        super().__init__()
        self.register_buffer("sbml_mask",     sbml_mask)              # (S,) bool
        self.register_buffer("sbml_indices",  sbml_indices)           # (n_sbml,) long
        self.register_buffer("stoich_matrix", stoich_matrix.float())  # (n_sbml, n_rxn)
        # CPU-side indexing for the per-SBML lo/span: sbml_indices may already
        # have been .to(cuda)'d by the caller, but lo_norm/span_norm are
        # numpy/CPU. Force CPU for this one-shot index, then register so
        # model.to(device) moves them along with everything else.
        si_cpu = sbml_indices.detach().cpu()
        lo_t   = torch.as_tensor(lo_norm,   dtype=torch.float32)
        span_t = torch.as_tensor(span_norm, dtype=torch.float32)
        self.register_buffer("lo_sbml",   lo_t[si_cpu])
        self.register_buffer("span_sbml", span_t[si_cpu].clamp(min=1e-6))
        n_rxn = stoich_matrix.shape[1]
        self.rate_head = nn.Linear(hidden, n_rxn)
        # Initialise to predict v_log ~ 0 (no change) so PINN starts as identity-ish
        nn.init.zeros_(self.rate_head.bias)
        nn.init.normal_(self.rate_head.weight, std=0.01)
        self.rate_clip = rate_clip

    def forward(self, h, x_norm):
        # h: (B, S, hidden);  x_norm: (B, S) normalised
        h_pool = h[:, self.sbml_mask].mean(dim=1)                          # (B, hidden)
        v_log  = self.rate_head(h_pool).clamp(-self.rate_clip, self.rate_clip)
        v_lin  = t_signed_expm1(v_log)                                     # (B, n_rxn)
        x_norm_sbml = x_norm[:, self.sbml_mask]                            # (B, n_sbml)
        x_sl_sbml   = x_norm_sbml * self.span_sbml + self.lo_sbml
        x_lin_sbml  = t_signed_expm1(x_sl_sbml)
        dx_lin      = v_lin @ self.stoich_matrix.T                         # (B, n_sbml)
        x_next_lin  = (x_lin_sbml + dx_lin).clamp(min=0.0)
        x_next_sl   = t_signed_log1p(x_next_lin)
        x_next_norm = (x_next_sl - self.lo_sbml) / self.span_sbml
        return x_next_norm                                                 # (B, n_sbml)


class StochasticHead(nn.Module):
    """v9: per-species log_sigma output, for NLL training.

    Trained jointly with the mean head, allows the model to express
    uncertainty (high log_sigma) where it can't predict precisely - which
    breaks the deterministic MSE noise floor.  Ported from M8 upgrade 1/5
    in the parallel M7 branch.
    """

    def __init__(self, hidden):
        super().__init__()
        self.head = nn.Linear(hidden, 1)
        # Init so initial sigma ~ 0.1 (log(0.1) ≈ -2.3)
        nn.init.constant_(self.head.bias, -2.3)
        nn.init.zeros_(self.head.weight)

    def forward(self, h):                                                  # (B, S, H) -> (B, S)
        return self.head(h).squeeze(-1).clamp(-6.0, 2.0)                  # sigma in [exp(-6), exp(2)]


class TemporalContext(nn.Module):
    """v15.0: small transformer that summarises the recent state history.

    Given a window of the past W states (B, W, S), produces a context vector
    (B, ctx_dim) used to bias the LGNN at the next forward.  The "two-cortex"
    half of the v15.0 hybrid — LGNN handles local species-graph dynamics, the
    transformer handles global trajectory shape (phase awareness, drift
    detection, anti-mode-collapse through attention back to t=0).

    Architecture:
        state(t-W+1..t) ─► Linear(S, D) per-step embed
                       + learned positional encoding (W, D)
                       ─► TransformerEncoder (L layers, H heads, FF dim)
                       ─► last-position output (B, D)

    Caller maintains the rolling history buffer; module is stateless so it
    plays nicely with torch.compile.  When fewer than W states are available
    (early in a rollout), the caller pads on the left with copies of the
    oldest state — produces a smooth transient that converges to the steady
    behaviour by step W.
    """

    def __init__(self, S, ctx_dim=T_CTX_HIDDEN, n_heads=T_CTX_HEADS,
                 n_layers=T_CTX_LAYERS, ff_dim=T_CTX_FF, window=T_CTX_WINDOW):
        super().__init__()
        self.S = S
        self.ctx_dim = ctx_dim
        self.window = window
        self.state_proj = nn.Linear(S, ctx_dim)
        # Learnable positional encoding (last position = "now")
        self.pos_embed = nn.Parameter(torch.randn(window, ctx_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.0, activation="gelu",
            norm_first=True,
        )
        # enable_nested_tensor=False suppresses an info-only PyTorch warning that
        # fires whenever norm_first=True (which we want for pre-LN stability).
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        # Init output near zero so the residual injection starts as a no-op
        # — model recovers v14.9 behaviour at step 0, transformer wakes up
        # as training adjusts these weights.
        self.out_proj = nn.Linear(ctx_dim, ctx_dim)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def forward(self, history):
        # history: (B, T_w, S) — last T_w states in normalised space.
        # If T_w < window: pad left with copies of the oldest entry.
        # If T_w >= window: take the rightmost window slots.
        B, T_w, _ = history.shape
        if T_w >= self.window:
            hist = history[:, -self.window:]
        else:
            pad_left = history[:, :1].expand(B, self.window - T_w, -1)
            hist = torch.cat([pad_left, history], dim=1)
        emb = self.state_proj(hist) + self.pos_embed.unsqueeze(0)            # (B, W, D)
        out = self.encoder(emb)                                              # (B, W, D)
        return self.out_proj(out[:, -1])                                     # (B, D)


class DynamicsModel(nn.Module):
    """v8 Liquid Graph Neural Network + v9 PINN head + v9 stochastic head.

    Single-step: takes current state (B, S), returns predicted next state.
    When the stochastic head is enabled, returns (next_state, log_sigma).
    """

    def __init__(self, S, hidden, n_layers, species_type_ids,
                 edge_index, edge_weight, cfc_tau_min=0.1, n_type_embed=4,
                 # v9 heads:
                 use_pinn=False, sbml_mask=None, sbml_indices=None,
                 stoich_matrix=None, lo_norm=None, span_norm=None,
                 pinn_rate_clip=PINN_RATE_CLIP, use_stochastic=False,
                 # v13.9: equation-wired metabolism
                 metab_tensors=None, metab_dt=1.0, volume_core=None,
                 cd_tensors=None, asm_tensors=None,
                 # v14 day 5: per-species empirical log σ (sigma calibration anchor)
                 target_log_sigma=None,
                 # v15.0: temporal-context transformer (two-cortex hybrid)
                 use_temporal_context=False,
                 t_ctx_window=T_CTX_WINDOW, t_ctx_hidden=T_CTX_HIDDEN,
                 t_ctx_heads=T_CTX_HEADS, t_ctx_layers=T_CTX_LAYERS,
                 t_ctx_ff=T_CTX_FF,
                 # backward-compat v7 kwargs (ignored)
                 d_model=None, n_heads=None, context=None, dropout=None,
                 d_type=None):
        super().__init__()
        self.S = S
        self.hidden = hidden
        self.n_layers = n_layers
        self.register_buffer("edge_index",  edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.type_embed = nn.Embedding(N_GTYPES, n_type_embed)
        self.register_buffer("stype",
                             torch.tensor(species_type_ids, dtype=torch.long))
        self.in_proj = nn.Linear(1 + n_type_embed, hidden)
        self.layers  = nn.ModuleList([
            _CfCGraphLayer(hidden, S, cfc_tau_min=cfc_tau_min)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)
        # v14.9: PINN and MetabolismCore coexist.  PINN runs first and covers
        # all 137 SBML species via neural flux + stoich; MetabolismCore then
        # overrides on its ~125 species with the bi-bi (measured-kinetics) rate.
        # Net result: all 356 SBML reactions get mass-balance enforcement,
        # MetabolismCore wins on overlap where bi-bi math is more trustworthy.
        self.metab_core = (MetabolismCore(metab_tensors,
                                          atp_idx=metab_tensors.get("atp_idx"))
                           if metab_tensors is not None else None)
        self.metab_dt = float(metab_dt)
        # v13.9: VolumeCore tracks dynamic cell volume; passed to MetabolismCore each step.
        self.volume_core = volume_core
        # v13.9: CentralDogmaCore — v14.9 additive (residual) instead of override
        self.cd_core = CentralDogmaCore(cd_tensors) if cd_tensors is not None else None
        # v13.9: AssemblyCore — mass-action complex assembly
        self.asm_core = AssemblyCore(asm_tensors) if asm_tensors is not None else None
        # v9: PINN head (optional).  v14.9: no longer mutually exclusive with MetabolismCore.
        self.use_pinn = bool(use_pinn and sbml_mask is not None)
        if self.use_pinn:
            self.pinn_head = PINNHead(hidden, sbml_mask, sbml_indices,
                                      stoich_matrix, lo_norm, span_norm,
                                      pinn_rate_clip)
        # v9: stochastic head (optional)
        self.use_stochastic = bool(use_stochastic)
        if self.use_stochastic:
            self.stochastic_head = StochasticHead(hidden)
        # Equation-wired cores all need per-species lo/span for normalised <-> count conversion.
        # Also used by KnockoutAugmentation in the training loop to compute the
        # normalised value of count=0 per species (zero_norm = -lo/span).
        if lo_norm is not None and span_norm is not None:
            lo_t   = torch.as_tensor(lo_norm,   dtype=torch.float32)
            span_t = torch.as_tensor(span_norm, dtype=torch.float32).clamp(min=1e-6)
            self.register_buffer("metab_lo",   lo_t)
            self.register_buffer("metab_span", span_t)
            zero_norm = (-lo_t / span_t).clamp(CLAMP_LO, CLAMP_HI)
            self.register_buffer("zero_norm", zero_norm)
        # v14 day 5: σ-anchor — pulls predicted log σ toward observed std,
        # prevents the NLL-shrinkage mode collapse where σ → 0 to game the loss.
        if target_log_sigma is not None:
            self.register_buffer("target_log_sigma",
                                  torch.as_tensor(target_log_sigma,
                                                  dtype=torch.float32))
        # v15.0: TemporalContext (two-cortex hybrid).  Out-projection initialised
        # near zero so the model starts from v14.9 behaviour and the transformer
        # wakes up gradually during training.
        self.use_temporal_context = bool(use_temporal_context)
        if self.use_temporal_context:
            self.temporal_ctx = TemporalContext(
                S=S, ctx_dim=t_ctx_hidden, n_heads=t_ctx_heads,
                n_layers=t_ctx_layers, ff_dim=t_ctx_ff, window=t_ctx_window,
            )
            self.ctx_to_hidden = nn.Linear(t_ctx_hidden, hidden)
            nn.init.zeros_(self.ctx_to_hidden.bias)
            nn.init.normal_(self.ctx_to_hidden.weight, std=0.01)
            self.t_ctx_window = t_ctx_window

    def forward(self, x, state_history=None):  # x: (B, S) normalised
        # state_history: optional (B, T_w, S) — past states including the
        # current one as the last row.  When None, falls back to using just
        # `x` as a single-step history (TemporalContext pads).
        # v15.9 speed: force contiguous input.  Rollout/eval slices like
        # trajs[:, t] hand us a non-contiguous (B, S) view (stride S*T at dim 1),
        # which made torch.compile re-specialise on stride and blow past the
        # recompile_limit (the "x_norm stride mismatch ... actual 120" warning).
        # One .contiguous() at the entry stabilises the compiled graph.
        x = x.contiguous()
        B, S = x.shape
        te  = self.type_embed(self.stype).unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([x.unsqueeze(-1), te], dim=-1)
        h   = self.in_proj(inp)
        # v15.0: inject the temporal-context vector as a per-batch bias to h
        # BEFORE the CfC layers run.  Each species gets the same context added
        # → it propagates through the graph layers and out to the prediction.
        if self.use_temporal_context:
            hist = state_history if state_history is not None else x.unsqueeze(1)
            c    = self.temporal_ctx(hist)                 # (B, ctx_dim)
            c_h  = self.ctx_to_hidden(c).to(h.dtype)       # (B, hidden)
            h    = h + c_h.unsqueeze(1)                    # broadcast over S
        for layer in self.layers:
            h = layer(h, self.edge_index, self.edge_weight)
        h = self.out_norm(h)
        delta  = self.out_head(h).squeeze(-1)
        x_next = x + delta
        # v14.9 head ordering (priority increases down the list):
        #   1. PINN — neural flux × stoichiometry for all 137 SBML species
        #   2. MetabolismCore — bi-bi rate law for ~125 species, overrides PINN where they overlap
        #   3. CentralDogmaCore — ADDITIVE residual (blend × CD_delta) on ~910 species
        #   4. AssemblyCore — override for complex species (currently off)
        # The reordering means PINN's mass-balance covers the 241 reactions
        # MetabolismCore doesn't have kinetics for, while MetabolismCore's
        # measured-kinetics wins on its 115 reactions.

        # 1. PINN head: mass-balance for all 137 SBML species
        if self.use_pinn:
            x_next_sbml = self.pinn_head(h, x)                              # (B, n_sbml)
            x_pinn_full = torch.zeros_like(x_next)
            x_pinn_full = x_pinn_full.index_copy(1, self.pinn_head.sbml_indices,
                                                  x_next_sbml)
            mask = self.pinn_head.sbml_mask.unsqueeze(0).expand(B, -1)
            x_next = torch.where(mask, x_pinn_full, x_next)

        # 2. MetabolismCore: bi-bi override (takes priority over PINN on its 125 species)
        if self.metab_core is not None:
            x_sl_lin     = x.float() * self.metab_span + self.metab_lo
            x_count      = t_signed_expm1(x_sl_lin)
            v_l = self.volume_core(x_count) if self.volume_core is not None else None
            d_count, fluxes = self.metab_core(x_count, dt=self.metab_dt, volume_l=v_l)
            # v14 day 3: stash ATP rate so the training loop can apply the energy-ledger loss
            if self.metab_core.has_atp:
                self.last_atp_rate = self.metab_core.compute_atp_rate(fluxes, volume_l=v_l)
            # v15.2: stash fluxes so rollout-based diagnostics (TUR, F-W action)
            # can collect them without re-running the metab_core themselves.
            self.last_fluxes = fluxes
            x_count_next = (x_count + d_count).clamp(min=0.0)
            x_sl_next    = t_signed_log1p(x_count_next)
            x_norm_metab = (x_sl_next - self.metab_lo) / self.metab_span
            x_norm_metab = x_norm_metab.clamp(CLAMP_LO, CLAMP_HI).to(x_next.dtype)
            mask = self.metab_core.coverage_mask.unsqueeze(0).expand(B, -1)
            x_next = torch.where(mask, x_norm_metab, x_next)

        # 3. CentralDogmaCore: ADDITIVE residual (v14.9, was override).
        #    x_next_combined = x_next + BLEND × (x_cd - x_input)
        #    Lets LGNN keep its learned signal; CD nudges in physics direction.
        if self.cd_core is not None:
            x_sl_lin     = x.float() * self.metab_span + self.metab_lo
            x_count      = t_signed_expm1(x_sl_lin)
            d_count      = self.cd_core(x_count, dt=self.metab_dt)
            x_count_next = (x_count + d_count).clamp(min=0.0)
            x_sl_next    = t_signed_log1p(x_count_next)
            x_norm_cd    = (x_sl_next - self.metab_lo) / self.metab_span
            x_norm_cd    = x_norm_cd.clamp(CLAMP_LO, CLAMP_HI).to(x_next.dtype)
            # Additive blend: cd_delta_norm = x_norm_cd - x (input)
            cd_residual  = x_norm_cd - x.to(x_norm_cd.dtype)
            mask_cd      = self.cd_core.coverage_mask.unsqueeze(0).expand(B, -1)
            x_next       = torch.where(mask_cd,
                                       x_next + CD_BLEND_WEIGHT * cd_residual,
                                       x_next)

        # 4. AssemblyCore: override for complex species (currently disabled by default)
        if self.asm_core is not None:
            x_sl_lin     = x.float() * self.metab_span + self.metab_lo
            x_count      = t_signed_expm1(x_sl_lin)
            d_count      = self.asm_core(x_count, dt=self.metab_dt)
            x_count_next = (x_count + d_count).clamp(min=0.0)
            x_sl_next    = t_signed_log1p(x_count_next)
            x_norm_asm   = (x_sl_next - self.metab_lo) / self.metab_span
            x_norm_asm   = x_norm_asm.clamp(CLAMP_LO, CLAMP_HI).to(x_next.dtype)
            mask_asm     = self.asm_core.coverage_mask.unsqueeze(0).expand(B, -1)
            # v15.4 [6/6]: additive blend (was full override) — same proven-safe
            # pattern as CentralDogma v14.9, bounds the risk of re-enabling a
            # module disabled in v14.1 for override-driven regressions.
            asm_residual = x_norm_asm - x.to(x_norm_asm.dtype)
            x_next       = torch.where(mask_asm,
                                       x_next + ASM_BLEND_WEIGHT * asm_residual,
                                       x_next)
        if self.use_stochastic:
            log_sigma = self.stochastic_head(h)
            return x_next, log_sigma
        return x_next


def _model_pred(out):
    """Strip log_sigma from model output if present (stochastic head case)."""
    return out[0] if isinstance(out, tuple) else out


# ── data ──────────────────────────────────────────────────────────────────────

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def t_signed_log1p(x):
    """Tensor version of signed_log1p — used by the PINN head's log-space bridge."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def t_signed_expm1(x):
    """Inverse of t_signed_log1p."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def load_data(skip_startup=True):
    """Load all parquet trajectories.  v10: pre-allocate the stacked array so we
    don't briefly hold two copies (12 + 12 GB at full resolution).

    v15.8: file-search is forgiving — tries the strict ``counts_and_fluxes*``
    pattern first, then falls back to ANY ``.parquet`` under PARQUET_DIR (or
    /MyDrive recursively if PARQUET_DIR is empty).  Sort key handles names
    like ``foo.001.parquet`` (the original convention) AND plain
    ``something_001.parquet`` (just sorts by filename).
    """
    assert HAS_PANDAS, "pandas required - install or run on Colab"
    if PARQUET_DIR:
        candidates = [
            f"{PARQUET_DIR}/counts_and_fluxes*.parquet",
            f"{PARQUET_DIR}/**/counts_and_fluxes*.parquet",
            f"{PARQUET_DIR}/*.parquet",
            f"{PARQUET_DIR}/**/*.parquet",
        ]
    else:
        candidates = [
            "/content/drive/MyDrive/**/counts_and_fluxes*.parquet",
            "/content/drive/MyDrive/**/*.parquet",
        ]
    files = []
    for pat in candidates:
        files = sorted(set(glob.glob(pat, recursive=True)))
        if files:
            print(f"[data] matched {len(files)} file(s) with pattern: {pat}")
            break

    def _sort_key(p):
        # Strip extension, take trailing integer if present, else fall back to name
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)\s*$", base)
        return (0, int(m.group(1))) if m else (1, base)
    files = sorted(files, key=_sort_key)
    assert files, ("no parquet files - set PARQUET_DIR.  Tried patterns: "
                   + " | ".join(candidates))
    print(f"[data] {len(files)} trajectory files")
    species_names = None
    out = None
    for fi, f in enumerate(files):
        df = pd.read_parquet(f)
        if species_names is None:
            species_names = list(df.index)
        arr = df.to_numpy(dtype=np.float32)[:, ::TIME_STRIDE].T
        if skip_startup:
            arr = arr[SKIP_STARTUP_STEPS:]
        if out is None:
            out = np.empty((len(files), *arr.shape), dtype=np.float32)
            print(f"[data] pre-allocated {out.nbytes / 1e9:.2f} GB for stacked array")
        out[fi] = arr
    if skip_startup:
        print(f"[data] startup skip: dropped first {SKIP_STARTUP_STEPS} decimated step(s)")
    return out, species_names


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-12))


def variance_weighted_r2(pred, true, top_k=VAR_R2_TOP_K):
    """v9: median per-species R² over the top-K highest-variance species.

    Strips the thousands of near-constant species (which inflate mean R²
    artificially) and reports only the species the model has to actually
    predict.  Returns (median_r2, n_used).
    """
    if pred.dim() > 2:
        pred = pred.reshape(-1, pred.shape[-1])
        true = true.reshape(-1, true.shape[-1])
    var = true.var(dim=0)                                                  # (S,)
    top_idx = torch.argsort(var, descending=True)[:top_k]
    r2_list = []
    for s in top_idx.tolist():
        t = true[:, s]; p = pred[:, s]
        ss_tot = ((t - t.mean()) ** 2).sum()
        if float(ss_tot) < 1e-9:
            continue
        ss_res = ((t - p) ** 2).sum()
        r2_list.append(float(1.0 - ss_res / ss_tot))
    if not r2_list:
        return float("nan"), 0
    return float(np.median(r2_list)), len(r2_list)


# ── v15.6: Breuer-as-training-signal helpers ────────────────────────────────

def build_breuer_helpers(species_names, breuer_labels):
    """Build the per-gene species map and the E/NE locus lists used by the
    Breuer-consistency loss.

    Returns:
      gene_species: {locus_key: [species_indices]} — G/R/P/RP species per gene
      e_loci:  list of locus keys labelled Essential  (strict — Quasi excluded)
      ne_loci: list of locus keys labelled Nonessential
    """
    from collections import defaultdict
    gene_species = defaultdict(list)
    for i, name in enumerate(species_names):
        pre, loc = parse_species(name)
        if pre in {"P", "R", "RP", "G"}:
            key = _locus_key(loc)
            if key:
                gene_species[key].append(i)
    e_loci, ne_loci = [], []
    for locus, label in breuer_labels.items():
        if locus not in gene_species:
            continue
        s = str(label).strip().lower()
        if s.startswith("nonessential") or s == "ne" or s == "non-essential":
            ne_loci.append(locus)
        elif s == "essential" or s == "e":          # strict — skip Quasi as ambiguous
            e_loci.append(locus)
    return dict(gene_species), e_loci, ne_loci


def breuer_consistency_loss(model, seed_state, gene_species, e_loci, ne_loci,
                              horizon, n_per_label, margin, rng):
    """v15.6: pairwise hinge loss on KO impact magnitudes vs Breuer 2019 labels.

    Samples n_per_label essential and n_per_label nonessential genes.  For each,
    runs a baseline rollout and a KO rollout of `horizon` steps from the same
    seed state, measures the squared deviation at the final step, and computes a
    pairwise hinge: every essential impact should exceed every nonessential
    impact by `margin`.

    Differentiable end-to-end through model forwards.  KOs use CLAMP_LO as the
    knocked-out floor (same convention as the eval KO sweep — same _ko_rollout
    semantics) and are re-applied at every step so the perturbation persists.

    Returns scalar loss tensor (autograd-tracked), or None if labels missing
    or sampled genes had no associated species.
    """
    if len(e_loci) < n_per_label or len(ne_loci) < n_per_label:
        return None
    e_sample  = list(rng.choice(e_loci,  size=n_per_label, replace=False))
    ne_sample = list(rng.choice(ne_loci, size=n_per_label, replace=False))

    B, S = seed_state.shape
    device = seed_state.device

    def _rollout(start, ko_idx_t):
        cur = start
        if ko_idx_t is not None:
            mask = torch.zeros(B, S, dtype=torch.bool, device=device)
            mask[:, ko_idx_t] = True
            cur = torch.where(mask, torch.full_like(cur, CLAMP_LO), cur)
        for _ in range(horizon):
            out  = model(cur)
            pred = out[0] if isinstance(out, tuple) else out
            cur  = pred.clamp(CLAMP_LO, CLAMP_HI)
            if ko_idx_t is not None:
                cur = torch.where(mask, torch.full_like(cur, CLAMP_LO), cur)
        return cur

    baseline_final = _rollout(seed_state, None)

    def _impact(locus):
        sp = gene_species.get(locus, [])
        if not sp:
            return None
        ko_idx_t = torch.tensor(sp, dtype=torch.long, device=device)
        ko_final = _rollout(seed_state, ko_idx_t)
        return ((ko_final - baseline_final) ** 2).mean()

    imp_e  = [v for v in (_impact(loc) for loc in e_sample)  if v is not None]
    imp_ne = [v for v in (_impact(loc) for loc in ne_sample) if v is not None]
    if not imp_e or not imp_ne:
        return None
    imp_e_t  = torch.stack(imp_e)               # (n_E,)
    imp_ne_t = torch.stack(imp_ne)              # (n_NE,)
    # Pairwise hinge: every NE - E + margin should be <= 0
    pairs = imp_ne_t.unsqueeze(0) - imp_e_t.unsqueeze(1) + margin
    return F.relu(pairs).mean()


# ── v15.7 Move A: training-signal builders ──────────────────────────────────

def build_multitask_groups(species_active):
    """A1: index groups for derived-quantity aux loss.  Returns {name: LongTensor}
    of species indices whose normalised values we sum into a tracked total.
    These are exactly the quantities the paper validates against."""
    groups = {}
    lip = [i for i, n in enumerate(species_active)
           if any(n.startswith(p) for p in LIPID_PREFIXES)]
    if lip:
        groups["lipid_total"] = torch.tensor(lip, dtype=torch.long)
    ribo = [i for i, n in enumerate(species_active)
            if any(n.startswith(p) for p in RIBOSOME_PREFIXES)]
    if ribo:
        groups["ribo_total"] = torch.tensor(ribo, dtype=torch.long)
    aden = [species_active.index(n) for n in (ATP_SPECIES_NAME, "M_adp_c", "M_amp_c")
            if n in species_active]
    if aden:
        groups["adenylate_total"] = torch.tensor(aden, dtype=torch.long)
    prot = [i for i, n in enumerate(species_active) if parse_species(n)[0] == "P"]
    if prot:
        groups["protein_total"] = torch.tensor(prot, dtype=torch.long)
    # ori / ter gene-copy proxies (sum all copy variants)
    for label, locus in (("ori_total", "0001"), ("ter_total", "0421")):
        idx = [i for i, n in enumerate(species_active)
               if n == f"G_{locus}" or n.startswith(f"G_{locus}_")]
        if idx:
            groups[label] = torch.tensor(idx, dtype=torch.long)
    return groups


def build_per_class_weights(train_X, floor=PER_CLASS_W_FLOOR, ceil=PER_CLASS_W_CEIL):
    """A2: per-species loss weight ∝ variance (normalised to mean 1.0, clamped).
    Aligns the training objective with the honest top-K-variance eval metric —
    currently the loss treats all 5933 species equally while the metric only
    scores the high-variance ones."""
    var = train_X.float().var(dim=(0, 1))                      # (S,)
    # normalise so mean weight ≈ 1 (keeps overall loss scale ~unchanged)
    w = var / var.mean().clamp(min=1e-9)
    w = w.clamp(floor, ceil)
    return w


def masked_pretrain(model, train_X):
    """v15.7 Move C1: BERT-style masked-species pretraining.

    Randomly zero MASK_FRAC of species in the input and ask the model to
    reconstruct the FULL current state (not the next step).  This forces the
    graph layers to actually propagate information between species — a species
    can only be reconstructed from its graph neighbours.  Pure self-supervision
    on the same trajectories; runs before the main dynamics training to warm
    the graph weights.

    Reconstruction target is the input itself (x), so we read the model's
    one-step prediction as a denoiser: with masked input, predict the clean
    current state.  Uses the mean head only (ignores σ).
    """
    if not USE_MASKED_PRETRAIN or MASK_PRETRAIN_STEPS <= 0:
        return
    _inner = getattr(model, "_orig_mod", model)
    opt = torch.optim.AdamW(model.parameters(), lr=MASK_PRETRAIN_LR,
                            weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(SEED + 5)
    N, T, S = train_X.shape
    use_nll = getattr(model, "use_stochastic", False)
    zero_norm = getattr(_inner, "zero_norm", None)
    mask_fill = zero_norm if zero_norm is not None else torch.zeros(S, device=train_X.device)

    def _autocast():
        if USE_BF16 and device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    model.train()
    print(f"[pretrain] masked-species pretraining: {MASK_PRETRAIN_STEPS} steps, "
          f"mask {MASK_FRAC:.0%}, lr {MASK_PRETRAIN_LR}")
    t0 = time.time()
    for step in range(MASK_PRETRAIN_STEPS):
        i_t = torch.randint(0, N, (BATCH,), generator=gen)
        t_t = torch.randint(0, T, (BATCH,), generator=gen)
        clean = torch.stack([train_X[i_t[b], t_t[b]] for b in range(BATCH)]).to(device)
        # mask: randomly select MASK_FRAC species per example, set to floor
        mask = torch.rand(BATCH, S, device=device) < MASK_FRAC
        masked_in = torch.where(mask, mask_fill.unsqueeze(0).to(clean.dtype), clean)
        opt.zero_grad()
        with _autocast():
            out = model(masked_in)
            pred = out[0] if use_nll else out
            # reconstruct only the masked positions (where info had to flow via graph)
            err = ((pred - clean) ** 2) * mask.to(pred.dtype)
            loss = err.sum() / mask.to(pred.dtype).sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(f"  [pretrain] step {step+1:4d}  masked-recon MSE {float(loss.detach()):.5f}",
                  flush=True)
    print(f"[pretrain] done in {time.time()-t0:.0f}s — graph warmed, starting dynamics training")


# ── training / eval ───────────────────────────────────────────────────────────

def train_model(model, train_X, ruleset, hyp=None, breuer=None,
                multitask_groups=None, per_class_w=None):
    """v12 LGNN trainer with truncated BPTT + optional BF16 autocast.

    Truncated BPTT (TBPTT): rolls forward K steps but backward only flows
    TBPTT_CHUNK steps at a time (state .detach()'d between chunks).  Memory
    bounded by TBPTT_CHUNK steps' worth of activations instead of K — so K can
    grow much larger than v11's 64 without OOM.

    BF16 autocast: ~2x speedup on Blackwell/Hopper/A100.  Loss + ruleset.project
    cast back to fp32 between rollout iterations to keep state numerically clean.

    v13.8: optional Tier-2 hypothesis aux loss (pass hyp with build_tensors()
    already called).  Weight LAMBDA_HYP is small so training data dominates.
    """
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    gen   = torch.Generator().manual_seed(SEED + 1)
    N, T, S = train_X.shape
    use_nll = getattr(model, "use_stochastic", False)
    use_hyp = hyp is not None and hyp.has_aux_loss()
    # v13.9 module 7: KO augmentation needs zero_norm (the normalised count=0 value).
    # Stash via the unwrapped model so it works under torch.compile.
    _inner = getattr(model, "_orig_mod", model)
    zero_norm = getattr(_inner, "zero_norm", None)
    use_ko_aug = USE_KO_AUGMENTATION and zero_norm is not None
    # v14 day 3: ATP ledger active when MetabolismCore has the ATP species in coverage
    use_atp_ledger = (USE_ATP_LEDGER
                      and _inner.metab_core is not None
                      and _inner.metab_core.has_atp)
    atp_ema = 0.0    # running average of net ATP rate, for the train log
    # v14 day 5: σ-anchor active when target_log_sigma was provided AND stochastic head is on
    target_log_sigma = getattr(_inner, "target_log_sigma", None)
    use_sigma_anchor = USE_SIGMA_ANCHOR and use_nll and target_log_sigma is not None
    sigma_ema = 0.0  # mean predicted log σ, for the train log
    # v15.0: temporal-context transformer — maintain a rolling history buffer per
    # rollout so the transformer can attend over recent dynamics.
    use_temporal_ctx = USE_TEMPORAL_CONTEXT and getattr(_inner, "use_temporal_context", False)
    t_ctx_window = getattr(_inner, "t_ctx_window", T_CTX_WINDOW)
    # v15.1: refinement-pass training — active in the last (1 - REFINE_START_FRAC) of STEPS.
    # Computed per-step inside the rollout loop based on `step`.
    refine_start_step = int(STEPS * REFINE_START_FRAC) if USE_REFINEMENT else STEPS + 1
    refine_ema = 0.0
    # v15.6: Breuer-as-training-signal — pairwise hinge on KO impact magnitudes.
    use_breuer       = USE_BREUER_LOSS and breuer is not None and breuer.get("e_loci") and breuer.get("ne_loci")
    breuer_start_step = int(STEPS * BREUER_LOSS_START_FRAC) if use_breuer else STEPS + 1
    breuer_rng       = np.random.RandomState(SEED + 8) if use_breuer else None
    breuer_ema       = 0.0
    # v15.7 Move A1: multi-task derived-quantity aux loss
    use_multitask = USE_MULTITASK_LOSS and multitask_groups
    if use_multitask:
        mt_groups = {k: v.to(device) for k, v in multitask_groups.items()}
    mt_ema = 0.0
    # v15.7 Move A2: per-class (variance) loss weighting
    use_pcw = USE_PER_CLASS_LOSS_W and per_class_w is not None
    pcw = per_class_w.to(device) if use_pcw else None

    def _autocast():
        if USE_BF16 and device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    t_start = time.time()
    model.train()
    print(f"[train] BF16 autocast: {'ON' if USE_BF16 and device == 'cuda' else 'off'}, "
          f"TBPTT chunk: {TBPTT_CHUNK}, K_MAX: {K_MAX}, STEPS: {STEPS}, "
          f"BATCH: {BATCH}, hidden: {LGNN_HIDDEN}, "
          f"compile: {'ON' if USE_TORCH_COMPILE and device == 'cuda' else 'off'}, "
          f"hyp_aux: {'ON' if use_hyp else 'off'}, "
          f"ko_aug: {'ON p=' + str(KO_AUG_PROB) if use_ko_aug else 'off'}, "
          f"atp_ledger: {'ON floor=' + str(int(ATP_MAINTENANCE_RATE)) if use_atp_ledger else 'off'}, "
          f"sigma_anchor: {'ON λ=' + str(LAMBDA_SIGMA_ANCHOR) if use_sigma_anchor else 'off'}, "
          f"temporal_ctx: {'ON W=' + str(t_ctx_window) if use_temporal_ctx else 'off'}, "
          f"refine: {'ON @step ' + str(refine_start_step) + ' λ=' + str(LAMBDA_REFINE) if USE_REFINEMENT else 'off'}, "
          f"breuer: {'ON @step ' + str(breuer_start_step) + ' λ=' + str(LAMBDA_BREUER) + ' E=' + str(len(breuer['e_loci'])) + ' NE=' + str(len(breuer['ne_loci'])) if use_breuer else 'off'}, "
          f"multitask: {'ON λ=' + str(LAMBDA_MULTITASK) + ' (' + str(len(mt_groups)) + ' groups)' if use_multitask else 'off'}, "
          f"per_class_w: {'ON' if use_pcw else 'off'}")
    hyp_ema = {}    # v13.8: per-kind running average across training (promotion signal)

    for step in range(STEPS):
        K    = 1 + int((K_MAX - 1) * step / STEPS)
        i_t  = torch.randint(0, N, (BATCH,), generator=gen)
        t0_t = torch.randint(0, max(1, T - 1 - K), (BATCH,), generator=gen)
        i, ts = i_t.tolist(), t0_t.tolist()
        state = torch.stack([train_X[i[b], ts[b]] for b in range(BATCH)])    # (B, S)
        i_d, t0_d = i_t.to(device), t0_t.to(device)

        # v13.9 module 7: KO augmentation — knock one random species out per
        # selected batch element, then keep it down through the K-step rollout.
        # Teaches the model that when a species is gone in the input it stays
        # gone in the output (and lets the equation cores propagate the
        # downstream effect for free).
        ko_b_idx, ko_sp_idx = None, None
        if use_ko_aug:
            perturb = torch.rand(BATCH, device=state.device) < KO_AUG_PROB
            if perturb.any():
                ko_b_idx  = perturb.nonzero(as_tuple=True)[0]
                ko_sp_idx = torch.randint(0, S, (ko_b_idx.shape[0],),
                                          device=state.device)
                state = state.clone()
                state[ko_b_idx, ko_sp_idx] = zero_norm[ko_sp_idx]

        opt.zero_grad()
        first_loss = None
        chunk_losses = []
        prev_state = state
        first_chunk = True
        all_loss_means = []                        # for printing
        # v15.0: initialise history buffer = window copies of seed state.
        # Fixed shape (B, W, S) throughout the rollout — friendly to torch.compile.
        state_history = (state.unsqueeze(1).repeat(1, t_ctx_window, 1)
                         if use_temporal_ctx else None)

        for k in range(K):
            with _autocast():
                out = model(state, state_history=state_history) \
                    if use_temporal_ctx else model(state)
                if use_nll:
                    pred, log_sigma = out
                    true   = train_X[i_d, t0_d + 1 + k]
                    if ko_b_idx is not None:
                        true = true.clone()
                        true[ko_b_idx, ko_sp_idx] = zero_norm[ko_sp_idx]
                    sq_err = (pred - true) ** 2
                    inv_var = torch.exp(-2.0 * log_sigma)
                    nll_elem = 0.5 * (inv_var * sq_err + 2.0 * log_sigma)   # (B, S)
                    if use_pcw:   # v15.7 A2: weight per species by variance
                        nll_elem = nll_elem * pcw.to(nll_elem.dtype).unsqueeze(0)
                    loss_k = nll_elem.mean()
                    if use_sigma_anchor:
                        # Pull log σ toward empirical std per species — prevents
                        # the σ → 0 collapse mode that gives wildly negative NLL
                        # while leaving MSE worse than persistence.
                        sigma_anchor = ((log_sigma - target_log_sigma) ** 2).mean()
                        loss_k = loss_k + LAMBDA_SIGMA_ANCHOR * sigma_anchor
                        sigma_ema = 0.99 * sigma_ema + 0.01 * float(log_sigma.mean().detach())
                        # v14.7: explicit anti-mode-collapse — predictions across
                        # the batch should spread as widely as data does for the
                        # same species.  Penalty on log(pred.std_across_batch) vs target_log_σ.
                        if BATCH > 1:
                            pred_batch_log_std = pred.std(dim=0).clamp(min=1e-4).log()
                            traj_var_loss = ((pred_batch_log_std - target_log_sigma) ** 2).mean()
                            loss_k = loss_k + LAMBDA_TRAJ_VAR * traj_var_loss
                else:
                    pred = out
                    true = train_X[i_d, t0_d + 1 + k]
                    if ko_b_idx is not None:
                        true = true.clone()
                        true[ko_b_idx, ko_sp_idx] = zero_norm[ko_sp_idx]
                    if use_pcw:   # v15.7 A2: variance-weighted MSE
                        loss_k = (((pred - true) ** 2)
                                  * pcw.to(pred.dtype).unsqueeze(0)).mean()
                    else:
                        loss_k = F.mse_loss(pred, true)
                if use_hyp:
                    hyp_loss_k, hyp_parts_k = hyp.soft_loss(prev_state, pred)
                    loss_k = loss_k + LAMBDA_HYP * hyp_loss_k
                    for kn, vv in hyp_parts_k.items():
                        prev_t = hyp_ema.get(kn)
                        hyp_ema[kn] = vv if prev_t is None else 0.99 * prev_t + 0.01 * vv
                if use_atp_ledger:
                    atp_rate = getattr(_inner, "last_atp_rate", None)
                    if atp_rate is not None:
                        # Penalise when ATP production falls below maintenance floor.
                        deficit = F.relu(ATP_MAINTENANCE_RATE - atp_rate)   # (B,)
                        atp_loss_k = (deficit / ATP_MAINTENANCE_RATE).pow(2).mean()
                        loss_k = loss_k + LAMBDA_ATP * atp_loss_k
                        atp_ema = 0.99 * atp_ema + 0.01 * float(atp_rate.mean().detach())
                # v15.1: refinement-pass loss.  After the model produces pred (its
                # claim about state(t+1+k)), feed it back through the model as
                # input — the second forward should land at the TRUE state(t+2+k).
                # Gradient flows only through the second forward (pred.detach()),
                # so this teaches "if the input is your own imperfect prediction,
                # still produce a good next-step prediction" — direct attack on
                # the rollout-vs-1step gap.  Cost: 1 extra forward per rollout
                # step in the last 10% of training.
                if step >= refine_start_step and (k + 1) < K:
                    out_refine = (model(pred.detach(),
                                         state_history=state_history)
                                  if use_temporal_ctx
                                  else model(pred.detach()))
                    pred_refine = out_refine[0] if use_nll else out_refine
                    true_refine = train_X[i_d, t0_d + 2 + k]
                    if ko_b_idx is not None:
                        true_refine = true_refine.clone()
                        true_refine[ko_b_idx, ko_sp_idx] = zero_norm[ko_sp_idx]
                    refine_loss = F.mse_loss(pred_refine, true_refine)
                    loss_k = loss_k + LAMBDA_REFINE * refine_loss
                    refine_ema = 0.99 * refine_ema + 0.01 * float(refine_loss.detach())
                # v15.7 A1: multi-task derived-quantity aux loss.  The model's
                # summed group totals (lipid/ribo/adenylate/protein/ori/ter)
                # should match the truth's — these are the paper-validation
                # quantities, so training on them directly drives the biology
                # metrics, not just per-species MSE.
                if use_multitask:
                    mt_loss = pred.new_zeros(())
                    for _gname, _gidx in mt_groups.items():
                        pred_tot = pred[:, _gidx].sum(dim=1)
                        true_tot = true[:, _gidx].sum(dim=1)
                        mt_loss = mt_loss + F.mse_loss(pred_tot, true_tot)
                    mt_loss = mt_loss / max(len(mt_groups), 1)
                    loss_k = loss_k + LAMBDA_MULTITASK * mt_loss
                    mt_ema = 0.99 * mt_ema + 0.01 * float(mt_loss.detach())
            if first_loss is None:
                first_loss = loss_k
            chunk_losses.append(loss_k)

            # next state: clamp + project in fp32 between rollout iterations.
            # v14.7: inject reparameterized noise scaled by predicted σ so the
            # next-step input is a SAMPLE from the predicted distribution, not
            # the mean.  Loss still uses pred (the mean), so NLL is unchanged;
            # but the model now has to produce predictions that are robust to
            # perturbed inputs — kills deterministic mean trajectories.
            if use_nll and use_sigma_anchor and SAMPLE_NOISE_SCALE > 0:
                noise = torch.randn_like(pred)
                sampled = pred + noise * torch.exp(log_sigma.clamp(max=2.0)) * SAMPLE_NOISE_SCALE
                nxt = sampled.float().clamp(CLAMP_LO, CLAMP_HI)
            else:
                nxt = pred.float().clamp(CLAMP_LO, CLAMP_HI)
            if step >= STEPS // 4:
                nxt = ruleset.project(prev_state, nxt)
            # KO augmentation: re-apply the knockdown so it persists across the
            # K-step rollout (matches the eval-time permanent-knockdown semantics).
            if ko_b_idx is not None:
                nxt = nxt.clone()
                nxt[ko_b_idx, ko_sp_idx] = zero_norm[ko_sp_idx]
            prev_state = nxt
            state = nxt
            # v15.0: slide the history window forward by one step.  Shape stays
            # (B, W, S) — drop the oldest, append the new state at the right.
            if use_temporal_ctx:
                state_history = torch.cat(
                    [state_history[:, 1:], nxt.unsqueeze(1)], dim=1)

            # TBPTT: backward at end of chunk
            if (k + 1) % TBPTT_CHUNK == 0 or k == K - 1:
                chunk_loss = torch.stack(chunk_losses).mean()
                # Include the LAMBDA_1STEP·first_loss term only in the chunk that
                # contains first_loss (otherwise its graph is already gone after detach).
                if first_chunk:
                    chunk_loss = chunk_loss + LAMBDA_1STEP * first_loss
                    first_chunk = False
                chunk_loss.backward()
                all_loss_means.append(float(chunk_loss.detach()))
                chunk_losses = []
                state = state.detach()
                prev_state = prev_state.detach()
                if use_temporal_ctx:
                    state_history = state_history.detach()

        # v15.6: Breuer-as-training-signal — pairwise hinge on KO impacts.
        # Independent forward graph; backward accumulates grads onto the
        # already-summed main-rollout grads, then opt.step() applies both.
        if use_breuer and step >= breuer_start_step and (step % BREUER_LOSS_EVERY == 0):
            # Reuse the same batch's first trajectory as the breuer seed (cheap; correct device).
            breuer_seed = train_X[i_d[:1], t0_d[:1]]                     # (1, S)
            with _autocast():
                bl = breuer_consistency_loss(
                    model, breuer_seed,
                    gene_species=breuer["gene_species"],
                    e_loci=breuer["e_loci"], ne_loci=breuer["ne_loci"],
                    horizon=BREUER_HORIZON, n_per_label=BREUER_KO_PER_LABEL,
                    margin=BREUER_MARGIN, rng=breuer_rng,
                )
            if bl is not None:
                (LAMBDA_BREUER * bl).backward()
                breuer_ema = 0.99 * breuer_ema + 0.01 * float(bl.detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 0 or (step + 1) % 250 == 0:
            hyp_str = ""
            if use_hyp and hyp_ema:
                hyp_str = "  hyp[" + " ".join(f"{k}={float(v):.4f}"
                                              for k, v in hyp_ema.items()) + "]"
            atp_str = f"  atp={atp_ema:.2e}/s" if use_atp_ledger else ""
            sigma_str = f"  log σ={sigma_ema:+.2f}" if use_sigma_anchor else ""
            refine_str = (f"  refine={refine_ema:.5f}"
                          if USE_REFINEMENT and step >= refine_start_step else "")
            breuer_str = (f"  breuer={breuer_ema:.5f}"
                          if use_breuer and step >= breuer_start_step else "")
            mt_str = f"  mt={mt_ema:.5f}" if use_multitask else ""
            print(f"  step {step+1:5d}  K={K:4d}  "
                  f"1-step {float(first_loss.detach()):.5f}  "
                  f"rollout {sum(all_loss_means)/len(all_loss_means):.5f}"
                  f"{hyp_str}{atp_str}{sigma_str}{refine_str}{breuer_str}{mt_str}", flush=True)

        # v13.7: rolling mid-training checkpoint so an interrupt mid-run doesn't lose progress
        if (CHECKPOINT_EVERY > 0
                and (step + 1) % CHECKPOINT_EVERY == 0
                and step + 1 < STEPS):
            try:
                # If torch.compile wrapped the model, unwrap for state_dict
                inner = getattr(model, "_orig_mod", model)
                path = f"{SAVE_DIR}/cell_emulator_v13_latest.pt"
                torch.save({
                    "model": inner.state_dict(),
                    "step":  step + 1,
                    "config": dict(S=S, hidden=LGNN_HIDDEN, n_layers=LGNN_N_LAYERS,
                                   architecture="LGNN_v13_intermediate"),
                }, path)
                print(f"  [ckpt] step {step+1} -> cell_emulator_v13_latest.pt")
            except Exception as e:
                print(f"  [ckpt] save failed: {e}")

    print(f"[train] {STEPS} steps in {time.time()-t_start:.0f}s")
    return {k: float(v) for k, v in hyp_ema.items()}


def _eval_autocast():
    """BF16 autocast for inference paths (eval + KO + generation).  Same speedup
    as training, no precision concerns because no gradients."""
    if USE_BF16 and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def one_step(model, Xset, n=400):
    model.eval()
    g  = torch.Generator().manual_seed(SEED + 2)
    nt, Tt, _ = Xset.shape
    i  = torch.randint(0, nt, (n,), generator=g).tolist()
    t  = torch.randint(0, Tt - 1, (n,), generator=g).tolist()
    state = torch.stack([Xset[i[b], t[b]]     for b in range(n)])
    nxt   = torch.stack([Xset[i[b], t[b] + 1] for b in range(n)])
    with _eval_autocast():
        pred = _model_pred(model(state)).float()
    return float(F.mse_loss(pred, nxt)), r2(pred, nxt)


@torch.no_grad()
def full_rollout(model, traj, ruleset):
    """v8/v9: single-step rollout — seed from one frame, generate the rest."""
    model.eval()
    _inner = getattr(model, "_orig_mod", model)
    use_temporal_ctx = getattr(_inner, "use_temporal_context", False)
    t_ctx_window = getattr(_inner, "t_ctx_window", T_CTX_WINDOW)
    state = traj[0].unsqueeze(0)        # (1, S)
    state_history = (state.unsqueeze(1).repeat(1, t_ctx_window, 1)
                     if use_temporal_ctx else None)
    preds = []
    for _ in range(traj.shape[0] - 1):
        with _eval_autocast():
            out = (model(state, state_history=state_history)
                   if use_temporal_ctx else model(state))
            p = _model_pred(out).float().clamp(CLAMP_LO, CLAMP_HI)
        p = ruleset.project(state, p)
        preds.append(p)
        state = p
        if use_temporal_ctx:
            state_history = torch.cat(
                [state_history[:, 1:], state.unsqueeze(1)], dim=1)
    return torch.cat(preds, 0), traj[1:]


# ── v14.4: paper validation metrics (Thornburg 2026) ─────────────────────────

def paper_validation_metrics(preds_norm, truth_norm, species_active, lo, span,
                              time_stride_s):
    """Compute the headline validation metrics the Thornburg 2026 paper reports
    on its 50 simulated cells.  Run on both our model's predicted rollout AND
    the upstream's actual rollout, so we can see how close we are.

    preds_norm, truth_norm: (n_test, T, S) torch tensors in normalised space.
    lo, span: per-species normalisation params (numpy arrays).

    Returns dict of metric -> dict(model=(mean, std), upstream=(mean, std), paper=value).
    """
    def to_count(arr):
        # signed_expm1 inverse of normalisation
        sl  = arr * span + lo
        cnt = np.sign(sl) * np.expm1(np.abs(sl))
        return np.maximum(cnt, 0)

    preds_count = to_count(preds_norm.detach().cpu().numpy())
    truth_count = to_count(truth_norm.detach().cpu().numpy())
    T = preds_count.shape[1]

    name_to_idx = {n: i for i, n in enumerate(species_active)}
    out = {}

    # 1. Doubling time from lipid total (paper: 105 min)
    lipid_idx = [i for i, n in enumerate(species_active)
                 if any(n.startswith(p) for p in LIPID_PREFIXES)]
    if lipid_idx:
        for label, arr in (("model", preds_count), ("upstream", truth_count)):
            lipid_t = arr[:, :, lipid_idx].sum(axis=-1)                # (n_test, T)
            initial = lipid_t[:, :1].clip(min=1.0)
            ratio = lipid_t / initial
            doubled_step = np.argmax(ratio >= 2.0, axis=1)             # (n_test,)
            never = (ratio[:, -1] < 2.0)
            doubled_min = np.where(never, T, doubled_step) * time_stride_s / 60.0
            out.setdefault("doubling_time_min", {})[label] = (
                float(doubled_min.mean()), float(doubled_min.std()))
        out["doubling_time_min"]["paper"] = 105.0

    # 2. ori:ter ratio — proxy from gene 0001 (origin / DnaA region) vs gene 0421 (terminus).
    # Sum ALL gene-copy variants (G_0001_C1, G_0001_C2, etc.) so we capture the
    # full count regardless of which copy the trajectory tracks.  When DNA
    # replicates, both gene copies appear; ori-side replicates first → ratio > 1.
    def find_gene_total(prefix, locus):
        return [i for i, n in enumerate(species_active)
                if n == f"{prefix}_{locus}"
                or n.startswith(f"{prefix}_{locus}_")]

    ori_idxs = find_gene_total("G", "0001")
    ter_idxs = find_gene_total("G", "0421")
    if ori_idxs and ter_idxs:
        print(f"  [validation] ori species ({len(ori_idxs)}): "
              f"{[species_active[i] for i in ori_idxs][:3]}{'...' if len(ori_idxs) > 3 else ''}")
        print(f"  [validation] ter species ({len(ter_idxs)}): "
              f"{[species_active[i] for i in ter_idxs][:3]}{'...' if len(ter_idxs) > 3 else ''}")
        for label, arr in (("model", preds_count), ("upstream", truth_count)):
            ori_final = arr[:, -1, ori_idxs].sum(axis=-1)
            ter_final = arr[:, -1, ter_idxs].sum(axis=-1).clip(min=1.0)
            ratio = ori_final / ter_final
            out.setdefault("ori_ter", {})[label] = (
                float(ratio.mean()), float(ratio.std()))
        out["ori_ter"]["paper"] = 1.28

    # 3. Ribosome fold-change.  STRICT filter: only assembled ribosomes
    # (C_ribosome, exact 'ribosome', 30S/50S full complexes), NOT biogenesis
    # intermediates like RB_pe_*, RB_cp_*, or RPM_* (which are ribosomal-
    # protein mRNAs, not ribosomes).  Paper: 500 initial -> 881 at division = 1.76×.
    ASSEMBLED_RIBO_KEYS = ("ribosomeP",     # confirmed name in our trajectory parquets
                            "C_ribosome", "ribosome",
                            "C_30S_ribosome", "C_50S_ribosome",
                            "30S_ribosome", "50S_ribosome",
                            "C_complete_ribosome", "complete_ribosome")
    def is_assembled_ribosome(name):
        return name in ASSEMBLED_RIBO_KEYS or any(
            name.startswith(k + "_") for k in ASSEMBLED_RIBO_KEYS)
    ribo_idx = [i for i, n in enumerate(species_active) if is_assembled_ribosome(n)]
    if ribo_idx:
        print(f"  [validation] assembled ribosome species ({len(ribo_idx)}): "
              f"{[species_active[i] for i in ribo_idx][:5]}"
              f"{'...' if len(ribo_idx) > 5 else ''}")
        for label, arr in (("model", preds_count), ("upstream", truth_count)):
            initial = arr[:, 0, ribo_idx].sum(axis=-1).clip(min=1.0)
            final   = arr[:, -1, ribo_idx].sum(axis=-1)
            fold    = final / initial
            out.setdefault("ribosome_fold", {})[label] = (
                float(fold.mean()), float(fold.std()))
        out["ribosome_fold"]["paper"] = 1.76
    else:
        print(f"  [validation] no assembled ribosome species found "
              f"(searched: {ASSEMBLED_RIBO_KEYS}); ribosome metric skipped")

    # 4. Protein fold-change (paper: most proteins reach 1.25-1.5× initial)
    protein_idx = [i for i, n in enumerate(species_active)
                   if n.startswith("P_") and not n.startswith("PM_")]
    if protein_idx:
        for label, arr in (("model", preds_count), ("upstream", truth_count)):
            initial = arr[:, 0, protein_idx].clip(min=1.0)
            final   = arr[:, -1, protein_idx]
            # Per-protein fold-change, then median across proteins, then mean across traj
            fold = final / initial                                          # (n_test, n_prot)
            per_traj_median = np.median(fold, axis=-1)                     # (n_test,)
            out.setdefault("protein_fold_median", {})[label] = (
                float(per_traj_median.mean()), float(per_traj_median.std()))
        out["protein_fold_median"]["paper"] = 1.40   # midpoint of 1.25-1.5

    return out


def print_paper_metrics(metrics):
    """Pretty-print the paper-validation metrics."""
    if not metrics:
        return
    print(f"  paper validation (model rollout vs upstream rollout vs paper target):")
    rows = [
        ("doubling time (min)",     "doubling_time_min",   105.0,   "min"),
        ("ori:ter ratio",            "ori_ter",             1.28,    ""),
        ("ribosome fold-change",     "ribosome_fold",       1.76,    "×"),
        ("protein fold-change",      "protein_fold_median", 1.40,    "× (median)"),
    ]
    for label, key, paper, unit in rows:
        m = metrics.get(key)
        if m is None:
            continue
        mm, ms = m.get("model",    (float("nan"), float("nan")))
        um, us = m.get("upstream", (float("nan"), float("nan")))
        print(f"    {label:25s}: model {mm:6.2f}±{ms:5.2f} | "
              f"upstream {um:6.2f}±{us:5.2f} | paper {paper:.2f} {unit}")


@torch.no_grad()
def full_rollout_batched(model, trajs, ruleset, collect_fluxes=False):
    """v10: roll all test trajectories in parallel (one shared time loop).
    7,199 steps × 10 trajs would take ~60 min serially at full resolution;
    batched, it's a single 7,199-step loop = ~6 min.  With v12 BF16 ~3 min.

    trajs: (n, T, S) tensor of seed trajectories.
    Returns (preds, truth) both shaped (n, T-1, S).

    v15.2: when collect_fluxes=True, also returns per-step per-reaction fluxes
    as (n, T-1, R) — read from model.last_fluxes after each forward pass.
    Used by TUR + Friedlin-Wentzell diagnostics.
    """
    model.eval()
    _inner = getattr(model, "_orig_mod", model)
    use_temporal_ctx = getattr(_inner, "use_temporal_context", False)
    t_ctx_window = getattr(_inner, "t_ctx_window", T_CTX_WINDOW)
    state = trajs[:, 0]              # (n, S)
    state_history = (state.unsqueeze(1).repeat(1, t_ctx_window, 1)
                     if use_temporal_ctx else None)
    preds = []
    fluxes_all = []
    metab_core = _inner.metab_core
    for _ in range(trajs.shape[1] - 1):
        with _eval_autocast():
            out = (model(state, state_history=state_history)
                   if use_temporal_ctx else model(state))
            p = _model_pred(out).float().clamp(CLAMP_LO, CLAMP_HI)
        p = ruleset.project(state, p)
        preds.append(p)
        if collect_fluxes and metab_core is not None:
            fl = getattr(_inner, "last_fluxes", None)
            if fl is not None:
                fluxes_all.append(fl.float().detach())
        state = p
        if use_temporal_ctx:
            state_history = torch.cat(
                [state_history[:, 1:], state.unsqueeze(1)], dim=1)
    preds_out = torch.stack(preds, dim=1)
    truth_out = trajs[:, 1:]
    if collect_fluxes and fluxes_all:
        flux_out = torch.stack(fluxes_all, dim=1)   # (n, T-1, R)
        return preds_out, truth_out, flux_out
    return preds_out, truth_out


# ── v15.3 step 0: stochastic ceiling — irreducible R² limit ──────────────────

def data_richness_report(raw_counts, species_names, active_mask, train_idx,
                          time_stride):
    """v15.7 Move D: surface what the active-species filter and time-subsampling
    discard, and what could be recovered as signal.

    Three things we throw away:
      1. Inactive species (filtered by span <= 1e-6) — some are conserved-zero
         (never present: a hard 'stays at 0' constraint), some are conserved-
         constant (always the same nonzero value: another hard constraint).
      2. Sub-stride dynamics — we subsample by `time_stride`; report how much
         step-to-step variance lives below the stride (information loss proxy).
      3. Per-species stochastic floor — cross-trajectory variance that no
         deterministic model can capture (already in the ceiling, surfaced here
         per-class so it's actionable).
    """
    n_traj, T_full, S_full = raw_counts.shape
    inactive = ~active_mask
    n_inactive = int(inactive.sum())
    print()
    print("=" * 72)
    print("  DATA-RICHNESS REPORT  (v15.7 Move D — what the filter discards)")
    print("=" * 72)
    # 1. Inactive species breakdown
    rc = raw_counts[train_idx]                                  # (n_tr, T, S)
    inactive_idx = np.where(inactive)[0]
    n_zero = n_const = 0
    for i in inactive_idx:
        col = rc[:, :, i]
        if np.all(col == 0):
            n_zero += 1
        elif col.std() < 1e-6:
            n_const += 1
    print(f"\n  Inactive species (filtered out): {n_inactive} / {S_full}")
    print(f"    conserved-zero (never present)     : {n_zero}  "
          f"← hard 'stays 0' constraints available")
    print(f"    conserved-constant (fixed nonzero) : {n_const}  "
          f"← hard 'stays C' constraints available")
    print(f"    other low-variance                 : {n_inactive - n_zero - n_const}")
    # 2. Sub-stride information (compare full-res vs strided step variance)
    if time_stride > 1 and T_full > time_stride * 2:
        act = raw_counts[train_idx][:, :, active_mask].astype(np.float64)
        full_step_var = np.diff(act, axis=1).var()
        strided = act[:, ::time_stride]
        strided_step_var = np.diff(strided, axis=1).var()
        ratio = strided_step_var / max(full_step_var, 1e-12)
        print(f"\n  Sub-stride dynamics (stride={time_stride}):")
        print(f"    full-res step variance    = {full_step_var:.3e}")
        print(f"    strided step variance     = {strided_step_var:.3e}")
        print(f"    → strided captures {ratio:.1f}× the per-step variance "
              f"(>1 means coarser steps see bigger jumps, as expected)")
    # 3. Per-class stochastic floor
    act = raw_counts[train_idx][:, :, active_mask].astype(np.float64)
    var_per_t = act.var(axis=0).mean(axis=0)                   # (S_active,) irreducible
    var_total = act.var(axis=(0, 1))
    frac_stoch = np.where(var_total > 1e-9, var_per_t / var_total, np.nan)
    print(f"\n  Stochastic-floor fraction (irreducible noise / total variance):")
    print(f"    median across active species = {np.nanmedian(frac_stoch):.3f}")
    print(f"    (higher = more of that species' variance is unpredictable noise)")
    print(f"  → {n_zero + n_const} hard constraints are recoverable from "
          f"currently-discarded inactive species (future work).")


def stochastic_ceiling_diagnostic(raw_counts_active, train_idx, top_k_var=200):
    """v15.3 step 0: data-only diagnostic — irreducible R² ceiling for any
    deterministic predictor on this stochastic data.

    Theory: trajectories are samples from a stochastic process.  At each
    timepoint t, the cross-trajectory variance Var_traj(x_t) is irreducible —
    no deterministic model can predict which Gillespie seed produced which
    trajectory.  For a deterministic predictor matching E[x_t | x_0]:
        R²_ceil(s) = 1 − E_t[ Var_traj(x_t,s) ] / Var_(traj,t)(x_·,s)

    Tells us how much headroom we have before we hit the wall of stochastic
    uncertainty.  Computed on training trajectories (Poisson-sampled mRNA +
    divergent SSA realisations, same distribution as test set).

    Args:
      raw_counts_active: (n_traj, T, S) raw counts in normalised space domain
      train_idx: indices to use (training set)
      top_k_var: top-K most-variable species to aggregate over

    Returns dict with per-species ceiling + aggregates.
    """
    rc_tr      = raw_counts_active[train_idx]                    # (n_tr, T, S)
    # Per-species variance across trajectories at each timepoint, then mean over t
    var_per_t  = rc_tr.var(axis=0)                               # (T, S)
    sig_stoch  = var_per_t.mean(axis=0)                          # (S,)
    sig_total  = rc_tr.var(axis=(0, 1))                          # (S,)
    valid_mask = sig_total > 1e-9
    ratio      = np.where(valid_mask, sig_stoch / np.maximum(sig_total, 1e-12), np.nan)
    r2_ceil    = np.clip(1.0 - ratio, 0.0, 1.0)

    # Top-K species by total variance — matches honest-R² convention
    var_rank   = np.argsort(sig_total)[::-1][:int(top_k_var)]
    r2_top     = r2_ceil[var_rank]
    r2_top_v   = r2_top[~np.isnan(r2_top)]
    r2_all_v   = r2_ceil[valid_mask & ~np.isnan(r2_ceil)]
    return {
        "r2_ceil_per_species":      r2_ceil,
        "valid_mask":               valid_mask,
        "r2_ceil_median_top_k":     float(np.median(r2_top_v))   if r2_top_v.size else float("nan"),
        "r2_ceil_mean_top_k":       float(np.mean(r2_top_v))     if r2_top_v.size else float("nan"),
        "r2_ceil_q25_top_k":        float(np.percentile(r2_top_v, 25)) if r2_top_v.size else float("nan"),
        "r2_ceil_q75_top_k":        float(np.percentile(r2_top_v, 75)) if r2_top_v.size else float("nan"),
        "r2_ceil_median_all":       float(np.median(r2_all_v))   if r2_all_v.size else float("nan"),
        "top_k":                    int(top_k_var),
        "n_top_valid":              int(r2_top_v.size),
        "stoch_var_total_ratio":    float(np.mean(ratio[var_rank][~np.isnan(ratio[var_rank])]))
                                     if np.any(~np.isnan(ratio[var_rank])) else float("nan"),
    }


def print_ceiling_report(c, current_rollout_r2=None, current_honest_r2=None):
    """Print stochastic ceiling diagnostic."""
    print()
    print("=" * 72)
    print("  STOCHASTIC CEILING — irreducible R² limit for any deterministic model")
    print("  R²_ceil(s) = 1 − E_t[ Var_traj(x_t) ] / Var_(traj,t)(x_·,s)")
    print("=" * 72)
    if c is None:
        print("  Not computed.")
        return
    print(f"\n  R² ceiling on top-{c['top_k']} most-variable species "
          f"({c['n_top_valid']} valid):")
    print(f"    median   = {c['r2_ceil_median_top_k']:.3f}")
    print(f"    mean     = {c['r2_ceil_mean_top_k']:.3f}")
    print(f"    IQR      = [{c['r2_ceil_q25_top_k']:.3f}, "
          f"{c['r2_ceil_q75_top_k']:.3f}]")
    print(f"    fraction of variance that is irreducible stochastic noise = "
          f"{c['stoch_var_total_ratio']:.3f}")
    print(f"\n  Median ceiling across ALL variable species = {c['r2_ceil_median_all']:.3f}")
    # Compare to current observed if provided
    if current_honest_r2 is not None:
        gap = c['r2_ceil_median_top_k'] - current_honest_r2
        print(f"\n  Current honest R² (top-{c['top_k']}, median) = {current_honest_r2:+.3f}")
        print(f"  Gap to ceiling:                          {gap:+.3f}")
        if gap < 0.05:
            verdict = ("AT THE WALL — deterministic R² has almost no room to grow.\n"
                       "    The only meaningful improvement is distributional "
                       "(W₂, paper-stats, stochastic rollout).")
        elif gap < 0.15:
            verdict = ("CLOSE TO THE WALL — modest room for deterministic R² gain.\n"
                       "    Distributional metrics will benefit more than R² will.")
        else:
            verdict = ("REAL HEADROOM — deterministic R² can improve substantially.\n"
                       "    Calibration + horizon fixes are worth pursuing.")
        print(f"  Verdict: {verdict}")
        # v15.4 B1 fix: compare like-with-like — honest top-K R² vs top-K ceiling
        # (the old "fraction captured" line compared all-species mean-R² to the
        #  top-K ceiling — incoherent denominators, gave nonsense like 113%).
        frac = current_honest_r2 / max(c['r2_ceil_median_top_k'], 1e-6)
        print(f"  Fraction of ceiling captured (top-{c['top_k']}, like-for-like) = "
              f"{frac:.1%}")


# ── v15.3 step 5: post-hoc σ-head recalibration + stochastic rollout ─────────

@torch.no_grad()
def recalibrate_sigma_posthoc(model, train_X, n_pairs=2000, batch=128):
    """v15.3 step 5: post-hoc per-species σ recalibration (Kuleshov 2018 style).

    Fit α[s] such that α[s] · exp(predicted_log_σ[s]) matches the empirical
    one-step residual std on training data.  Doesn't retrain anything;
    α is computed from forward passes on random (i, t) pairs.

    Applied at eval time in stochastic rollout so injected noise has the
    correct magnitude.

    Args:
      model: trained DynamicsModel with stochastic head
      train_X: (n_tr, T, S) normalised training trajectories
      n_pairs: number of random (trajectory, timestep) pairs to sample

    Returns dict with α (S,) and calibration summary.
    """
    _inner = getattr(model, "_orig_mod", model)
    if not getattr(_inner, "use_stochastic", False):
        return None
    use_tc   = getattr(_inner, "use_temporal_context", False)
    t_w      = getattr(_inner, "t_ctx_window", T_CTX_WINDOW)
    device   = train_X.device
    n_tr, T, S = train_X.shape
    n_pairs  = int(min(n_pairs, n_tr * (T - 1)))
    rng      = np.random.RandomState(SEED + 42)
    i_samp   = rng.randint(0, n_tr, size=n_pairs)
    t_samp   = rng.randint(0, T - 1, size=n_pairs)

    model.eval()
    pred_chunks, ls_chunks, true_chunks = [], [], []
    for i in range(0, n_pairs, batch):
        i_b = i_samp[i:i+batch]; t_b = t_samp[i:i+batch]
        s_b = train_X[i_b, t_b]                  # (b, S)
        true_b = train_X[i_b, t_b + 1]           # (b, S)
        with _eval_autocast():
            if use_tc:
                hist = s_b.unsqueeze(1).repeat(1, t_w, 1)
                out  = model(s_b, state_history=hist)
            else:
                out  = model(s_b)
            pred_b, ls_b = out
        pred_chunks.append(pred_b.float())
        ls_chunks.append(ls_b.float())
        true_chunks.append(true_b.float())

    preds      = torch.cat(pred_chunks, dim=0)         # (n, S)
    log_sigmas = torch.cat(ls_chunks,   dim=0)         # (n, S)
    true_next  = torch.cat(true_chunks, dim=0)         # (n, S)

    residuals     = (true_next - preds)
    resid_std     = residuals.std(dim=0, unbiased=True).clamp(min=1e-6)         # (S,)
    pred_sigma    = torch.exp(log_sigmas)
    pred_sig_mean = pred_sigma.mean(dim=0).clamp(min=1e-6)                       # (S,)
    alpha         = (resid_std / pred_sig_mean).clamp(min=1e-3, max=1e3)        # (S,)

    return {
        "alpha":              alpha,
        "n_pairs":            int(n_pairs),
        "resid_std_median":   float(resid_std.median().item()),
        "pred_sigma_median":  float(pred_sig_mean.median().item()),
        "alpha_median":       float(alpha.median().item()),
        "alpha_q25":          float(alpha.quantile(0.25).item()),
        "alpha_q75":          float(alpha.quantile(0.75).item()),
    }


def print_recalibration_report(rc):
    """Print σ-recalibration diagnostic."""
    print()
    print("=" * 72)
    print("  σ-HEAD POST-HOC RECALIBRATION  (Kuleshov 2018 / Levi 2022 style)")
    print("  α[s] = empirical_residual_std[s] / mean_predicted_σ[s]")
    print("=" * 72)
    if rc is None:
        print("  Stochastic head not enabled — recalibration skipped.")
        return
    print(f"\n  Fitted on {rc['n_pairs']} random (trajectory, timestep) pairs from train_X")
    print(f"\n  Empirical residual std (median across species) = {rc['resid_std_median']:.5f}")
    print(f"  Predicted σ pre-recal  (median across species)  = {rc['pred_sigma_median']:.5f}")
    print(f"\n  Recalibration multiplier α:")
    print(f"    median = {rc['alpha_median']:.3f}  (1.0 = already calibrated)")
    print(f"    IQR    = [{rc['alpha_q25']:.3f}, {rc['alpha_q75']:.3f}]")
    am = rc["alpha_median"]
    if am > 5.0:
        verdict = f"σ severely under-predicted (over-confident by ~{am:.1f}×)"
    elif am > 2.0:
        verdict = f"σ over-confident by ~{am:.1f}×"
    elif am < 0.5:
        verdict = f"σ under-confident (predicted σ too large by ~{1/am:.1f}×)"
    else:
        verdict = "σ reasonably well-calibrated"
    print(f"  Verdict: {verdict}")


@torch.no_grad()
def full_rollout_batched_stochastic(model, trajs, ruleset, sigma_alpha,
                                      collect_fluxes=False, seed=None):
    """v15.3 step 5: stochastic rollout — sample from N(pred, α·exp(log_σ))
    at each step instead of using pred directly.

    Makes preds_batch.std(dim=0) measure the model's predicted trajectory
    spread (initial-state variation + injected stochasticity), allowing the
    distributional diagnostics (W₂, Helmholtz, paper-stats) to be read
    against a fair comparison.

    Per-trajectory R² will typically drop a little vs deterministic rollout —
    that's expected and correct.  The point is the diagnostics, not R².
    """
    _inner = getattr(model, "_orig_mod", model)
    if not getattr(_inner, "use_stochastic", False):
        return None
    model.eval()
    use_tc = getattr(_inner, "use_temporal_context", False)
    t_w    = getattr(_inner, "t_ctx_window", T_CTX_WINDOW)

    state          = trajs[:, 0]
    state_history  = (state.unsqueeze(1).repeat(1, t_w, 1) if use_tc else None)
    sigma_alpha    = sigma_alpha.to(state.device, state.dtype)

    g  = torch.Generator(device="cpu").manual_seed(
            int(seed) if seed is not None else SEED + 99)
    preds, fluxes_all = [], []
    metab_core = _inner.metab_core

    for _ in range(trajs.shape[1] - 1):
        with _eval_autocast():
            out = (model(state, state_history=state_history)
                   if use_tc else model(state))
            pred, log_sigma = out
            sigma_recal = torch.exp(log_sigma) * sigma_alpha
            noise_cpu   = torch.randn(pred.shape, generator=g).to(pred.device)
            sampled     = pred + noise_cpu.to(pred.dtype) * sigma_recal
            p = sampled.float().clamp(CLAMP_LO, CLAMP_HI)
        p = ruleset.project(state, p)
        preds.append(p)
        if collect_fluxes and metab_core is not None:
            fl = getattr(_inner, "last_fluxes", None)
            if fl is not None:
                fluxes_all.append(fl.float().detach())
        state = p
        if use_tc:
            state_history = torch.cat(
                [state_history[:, 1:], state.unsqueeze(1)], dim=1)

    preds_out = torch.stack(preds, dim=1)
    truth_out = trajs[:, 1:]
    if collect_fluxes and fluxes_all:
        return preds_out, truth_out, torch.stack(fluxes_all, dim=1)
    return preds_out, truth_out


# ── v15.2 TUR (Barato-Seifert 2015) per-reaction diagnostic ──────────────────

@torch.no_grad()
def tur_per_reaction_diagnostic(model, fluxes, gibbs_data, time_stride_s,
                                  volume_l=SYN3A_VOLUME_L, t_kelvin=310.0):
    """v15.2: Thermodynamic Uncertainty Relation per metabolic reaction.

    Barato & Seifert, *PRL* 114:158101 (2015):
        S_tot · Var(J_r) / ⟨J_r⟩²  ≥  2 k_B
    where S_tot is total entropy production over the trajectory (in k_B units)
    and J_r is the integrated flux of reaction r over the trajectory.

    Violations (ratio < 2) flag reactions whose flux statistics are
    thermodynamically inconsistent.  This is a hard physical falsifier —
    it cannot be tuned away by changing the loss; the model is wrong about
    the dissipation-precision trade-off for that reaction.

    Args:
      model: trained DynamicsModel (loaded checkpoint)
      fluxes: (n, T-1, R) per-step per-reaction fluxes from
              full_rollout_batched(collect_fluxes=True), in mM/s
      gibbs_data: dict from parse_gibbs — reaction id → ΔG° (kJ/mol)
      time_stride_s: seconds per timestep
      volume_l: cell volume in litres (used for unit conversion)
      t_kelvin: temperature, default 310 K (~body temperature)

    Returns dict with per-reaction tur_ratio + summary stats.
    """
    if fluxes is None:
        return None
    _inner = getattr(model, "_orig_mod", model)
    if _inner.metab_core is None:
        return None
    R = fluxes.shape[-1]
    rxn_names = _inner.metab_core.wired_reactions
    device = fluxes.device

    # Build ΔG° vector aligned with wired reactions (zero for unknowns)
    gibbs_vec = torch.zeros(R, dtype=torch.float32, device=device)
    covered   = torch.zeros(R, dtype=torch.bool,    device=device)
    for i, name in enumerate(rxn_names):
        if name in gibbs_data:
            gibbs_vec[i] = float(gibbs_data[name])
            covered[i]   = True

    # Integrate flux over trajectory.  fluxes is in mM/s = mmol/(L·s);
    # ∫ flux dt = sum(flux) · Δt has units mM = mmol/L.
    integrated_flux = fluxes.sum(dim=1) * float(time_stride_s)   # (n, R) in mM

    # Total entropy production per trajectory in k_B units.
    # Per reaction r: energy dissipated = -ΔG°_r [kJ/mol] · ∫J_r [mM = mmol/L] · V [L]
    # Unit check: kJ/mol · mmol/L · L = kJ · mmol/mol = J  (since mmol/mol = 1e-3
    # and kJ/J = 1e3 cancel)
    # σ in k_B = energy [J] / (k_B · T) [J]
    K_B    = 1.380649e-23                     # J/K (CODATA 2019)
    k_BT_J = K_B * float(t_kelvin)            # J (≈ 4.28e-21 at 310 K)
    sigma_per_traj = -(gibbs_vec.unsqueeze(0) * integrated_flux).sum(dim=1) \
                     * float(volume_l) / k_BT_J                     # (n,) in k_B
    sigma_total     = float(sigma_per_traj.mean())
    sigma_neg_frac  = float((sigma_per_traj < 0).float().mean())

    # Per-reaction flux statistics
    J_mean = integrated_flux.mean(dim=0)                            # (R,)
    J_var  = integrated_flux.var(dim=0, unbiased=True)              # (R,)

    # TUR ratio: |σ_total| · Var(J) / ⟨J⟩²  — should be ≥ 2 (k_B)
    eps       = 1e-12
    tur_ratio = abs(sigma_total) * J_var / (J_mean ** 2 + eps)
    return {
        "tur_ratio":      tur_ratio.detach().cpu().numpy(),
        "sigma_total":    sigma_total,
        "sigma_per_traj": sigma_per_traj.detach().cpu().numpy(),
        "sigma_neg_frac": sigma_neg_frac,
        "flux_mean":      J_mean.detach().cpu().numpy(),
        "flux_var":       J_var.detach().cpu().numpy(),
        "rxn_names":      list(rxn_names),
        "gibbs":          gibbs_vec.detach().cpu().numpy(),
        "gibbs_covered":  covered.detach().cpu().numpy(),
    }


def print_tur_report(tur, top_k=15):
    """Print TUR per-reaction diagnostic."""
    print()
    print("=" * 72)
    print("  TUR PER-REACTION DIAGNOSTIC  (Barato-Seifert 2015)")
    print("  Constraint: σ · Var(J) / ⟨J⟩²  ≥  2 k_B   per reaction")
    print("=" * 72)
    if tur is None:
        print("  MetabolismCore disabled or no fluxes collected — TUR not computable.")
        return
    covered    = tur["gibbs_covered"]
    rxn_names  = tur["rxn_names"]
    n_covered  = int(covered.sum())
    tur_ratio  = tur["tur_ratio"]
    sigma_tot  = tur["sigma_total"]
    sigma_neg  = tur["sigma_neg_frac"]

    print(f"\n  Total entropy production σ over the rollout:")
    print(f"    mean σ (k_B units)         = {sigma_tot:+.3e}")
    print(f"    fraction with σ < 0       = {sigma_neg:.1%}    "
          f"(>0% means second-law violation in some predicted trajectories)")
    if sigma_tot < 0:
        print("    ⚠ NET NEGATIVE entropy production — model violates 2nd law on average")

    print(f"\n  ΔG° coverage: {n_covered}/{len(rxn_names)} wired reactions have known ΔG°")
    n_viol = int(((tur_ratio < 2.0) & covered).sum())
    print(f"  TUR violations (ratio < 2):   {n_viol}/{n_covered}")

    # Sort by TUR ratio ascending, show worst-offending reactions with known ΔG°
    if n_covered > 0:
        order = np.argsort(tur_ratio)
        shown = 0
        print(f"\n  Worst-{min(top_k, n_covered)} reactions by TUR ratio (covered only):")
        print(f"    {'reaction':<28s} {'TUR ratio':>10s} {'⟨J⟩ (mM·s)':>14s} "
              f"{'Var(J)':>14s} {'ΔG° (kJ/mol)':>13s}")
        for i in order:
            if not covered[i]:
                continue
            mark = " ✗" if tur_ratio[i] < 2.0 else "  "
            print(f"  {mark}{rxn_names[i]:<28s} {tur_ratio[i]:>10.3f} "
                  f"{tur['flux_mean'][i]:>+14.3e} {tur['flux_var'][i]:>14.3e} "
                  f"{tur['gibbs'][i]:>+13.1f}")
            shown += 1
            if shown >= top_k:
                break


# ── v15.2 counterfactual-robustness diagnostic ───────────────────────────────

@torch.no_grad()
def counterfactual_robustness(model, seed_state, species_active, edge_index,
                                n_test_species=100, perturbation_mags=(0.0, 0.1, 0.2, 0.3),
                                n_steps=60, seed=None):
    """v15.2: Counterfactual-robustness metric per species.

    Putnam–Chalmers–Piccinini criterion: a "computing" system preserves
    counterfactual transition structure — perturbing an input produces a
    proportional, sensible output change.  A "memorizing" system does not —
    output is determined by trained input patterns, not by current input
    value.

    For each tested species s, perturb seed_state[s] by Δ ∈ perturbation_mags
    and roll forward n_steps.  Measure response = ||Δ_neighbors||_1 at the
    final step.  Linear-fit slope of response vs Δ:
        - slope > 0           → counterfactually responsive (substrate-driven)
        - slope ≈ 0           → memorized (input value doesn't matter)
        - slope < 0 or noisy  → unstable

    Skips ruleset.project to expose raw model output (rules would mask the
    diagnostic by forcing the species back into its trained range).

    Args:
      model:           trained DynamicsModel
      seed_state:      (1, S) initial normalized state
      species_active:  list of S species names
      edge_index:      (2, E) graph edges for neighbor lookup
      n_test_species:  random subset size
      perturbation_mags: tuple of Δ values to scan
      n_steps:         rollout horizon
      seed:            RNG seed for species subsampling

    Returns dict with per-species slope + aggregate stats.
    """
    _inner = getattr(model, "_orig_mod", model)
    use_tc = getattr(_inner, "use_temporal_context", False)
    t_w    = getattr(_inner, "t_ctx_window", T_CTX_WINDOW)
    S      = seed_state.shape[1]
    mags   = list(perturbation_mags)
    n_mags = len(mags)

    def _rollout(initial):
        st = initial.clone()
        hist = st.unsqueeze(1).repeat(1, t_w, 1) if use_tc else None
        traj = [st.clone()]
        for _ in range(n_steps):
            with _eval_autocast():
                out = model(st, state_history=hist) if use_tc else model(st)
                p = _model_pred(out).float().clamp(CLAMP_LO, CLAMP_HI)
            traj.append(p.clone())
            st = p
            if use_tc:
                hist = torch.cat([hist[:, 1:], st.unsqueeze(1)], dim=1)
        return torch.stack(traj, dim=1)              # (1, n_steps+1, S)

    baseline = _rollout(seed_state)

    edge_arr  = edge_index.cpu().numpy()
    seed_val  = seed if seed is not None else SEED + 7
    rng       = np.random.RandomState(seed_val)
    test_idx  = rng.choice(S, size=min(n_test_species, S), replace=False)

    slopes        = np.zeros(len(test_idx))
    propagation   = np.zeros(len(test_idx))
    n_nbrs_per    = np.zeros(len(test_idx), dtype=int)
    nbrs_cache    = {}

    def _neighbors(sp):
        if sp not in nbrs_cache:
            out_n = edge_arr[1][edge_arr[0] == sp][:5].tolist()
            in_n  = edge_arr[0][edge_arr[1] == sp][:5].tolist()
            nbrs_cache[sp] = [int(n) for n in set(out_n) | set(in_n) if n != sp][:5]
        return nbrs_cache[sp]

    for j, sp in enumerate(test_idx):
        sp_i = int(sp)
        nbrs = _neighbors(sp_i)
        n_nbrs_per[j] = len(nbrs)

        responses = np.zeros(n_mags)
        for k, mag in enumerate(mags):
            pseed     = seed_state.clone()
            orig_val  = float(pseed[0, sp_i].item())
            pseed[0, sp_i] = float(max(CLAMP_LO, min(CLAMP_HI, orig_val + mag)))
            pert      = _rollout(pseed)
            if nbrs:
                resp = float((pert[0, -1, nbrs] - baseline[0, -1, nbrs]).abs().mean().item())
            else:
                resp = float((pert[0, -1, sp_i] - baseline[0, -1, sp_i]).abs().item())
            responses[k] = resp

        mags_arr = np.asarray(mags, dtype=np.float64)
        if mags_arr.std() > 0:
            slopes[j] = float(np.polyfit(mags_arr, responses, 1)[0])
        propagation[j] = responses[-1]

    return {
        "test_species_idx":  test_idx,
        "slopes":            slopes,
        "propagation_R":     propagation,
        "n_neighbors":       n_nbrs_per,
        "perturbation_mags": mags,
    }


def print_counterfactual_report(cf, species_active, top_k=10, slope_thresh=0.01):
    """Print counterfactual-robustness diagnostic."""
    print()
    print("=" * 72)
    print("  COUNTERFACTUAL-ROBUSTNESS DIAGNOSTIC")
    print("  Putnam–Chalmers–Piccinini: computing systems preserve counterfactual structure")
    print("=" * 72)
    if cf is None:
        print("  Not computed.")
        return
    slopes      = cf["slopes"]
    propagation = cf["propagation_R"]
    n_tested    = len(slopes)
    f_responsive = float((slopes > slope_thresh).mean())
    f_memorized  = float((np.abs(slopes) < slope_thresh).mean())

    print(f"\n  Tested {n_tested} species, perturbation magnitudes {cf['perturbation_mags']}")
    print(f"\n  Slope distribution  (∂||Δ_nbrs||/∂Δ):")
    print(f"    median = {float(np.median(slopes)):+.4f}")
    print(f"    mean   = {float(slopes.mean()):+.4f}")
    print(f"    std    = {float(slopes.std()):.4f}")
    print(f"    fraction responsive   (slope > {slope_thresh}): {f_responsive:.1%}")
    print(f"    fraction memorized    (|slope| < {slope_thresh}): {f_memorized:.1%}")
    print(f"\n  Final-step response at largest Δ ({cf['perturbation_mags'][-1]:.2f}):")
    print(f"    median = {float(np.median(propagation)):.4f}")
    print(f"    mean   = {float(propagation.mean()):.4f}")

    order = np.argsort(slopes)[::-1][:top_k]
    print(f"\n  Top-{top_k} most counterfactually-responsive species:")
    for i in order:
        sp_i = int(cf["test_species_idx"][i])
        name = species_active[sp_i]
        print(f"    {name[:28]:>28s}  slope={slopes[i]:+.4f}  "
              f"R_max={propagation[i]:.4f}  n_nbrs={cf['n_neighbors'][i]}")

    if f_responsive > 0.5:
        verdict = "SUBSTRATE-DRIVEN — counterfactual structure preserved"
    elif f_responsive > 0.2:
        verdict = "MIXED — some species substrate-driven, some memorized"
    else:
        verdict = "MEMORIZED — counterfactual structure largely absent"
    print(f"\n  Verdict: {verdict}")


# ── v15.2 Friedlin-Wentzell action (trajectory plausibility) ─────────────────

@torch.no_grad()
def friedlin_wentzell_action(predicted_trajs, upstream_trajs, dt,
                                top_k_var=200, fit_trajs=None):
    """v15.2: Time-marginal Friedlin-Wentzell action.

    Theory (Freidlin-Wentzell 1984, the report §10):
        S[φ] = ∫₀ᵀ (φ̇ − b(φ))ᵀ A(φ)⁻¹ (φ̇ − b(φ)) dt,    A = σσᵀ
    Trajectories minimising S are the most-probable under the inferred SDE.

    Implementation choices:
    - Time-marginal drift b(t) and diagonal diffusion A(t) (diag because
      A is otherwise 5940×5940 estimated from 50 samples, rank-deficient —
      the report's suggested reduction)
    - Restrict to top-K most-variable species (the same set used by the
      honest-R² metric, so we compare like-with-like)
    - Compare action of model-predicted vs upstream trajectories under
      the SAME inferred dynamics — what matters is the ratio, not the
      absolute number

    Args:
      predicted_trajs: (n_pred, T, S) model rollouts in normalised space
      upstream_trajs:  (n_up,   T, S) held-out simulator trajectories
      dt: seconds per timestep
      top_k_var: number of top-variance species to include
      fit_trajs: optional separate set to fit b, A — defaults to upstream_trajs

    Returns dict with per-trajectory actions + ratio.
    """
    if fit_trajs is None:
        fit_trajs = upstream_trajs
    n_pred, T, S = predicted_trajs.shape

    # Fit drift and diffusion from the training-like set.  delta has shape
    # (n_fit, T-1, S); time-marginal mean and variance give b_t and A_t.
    delta_fit = fit_trajs[:, 1:] - fit_trajs[:, :-1]
    b_t = delta_fit.mean(dim=0) / float(dt)                       # (T-1, S)
    a_t = delta_fit.var(dim=0, unbiased=True) / float(dt)         # (T-1, S)
    a_t = a_t.clamp(min=1e-8)

    # Top-K variable species (use the fit set's variance)
    species_var = fit_trajs.reshape(-1, S).var(dim=0)
    top_idx     = torch.argsort(species_var, descending=True)[:int(top_k_var)]
    b_k = b_t[:, top_idx]
    a_k = a_t[:, top_idx]

    def _action(trajs):
        delta = (trajs[:, 1:] - trajs[:, :-1])[:, :, top_idx]     # (n, L-1, K)
        # v15.7.1 fix (my v15.4 [1/6] commit lied — Edit failed silently):
        # fit set (train_X length T_fit) and eval set (rollout length T_fit-1)
        # differ by one step, so b_k/a_k (length T_fit-1) and delta (length L-1)
        # mismatch on the time axis. Align both to the shorter dimension.
        n_t = delta.shape[1]
        m = min(n_t, b_k.shape[0])
        d_m = delta[:, :m]                                        # (n, m, K)
        b_m = b_k[:m].unsqueeze(0)                                # (1, m, K)
        a_m = a_k[:m].unsqueeze(0)                                # (1, m, K)
        resid = d_m - b_m * float(dt)
        # Per-step action contribution: resid² / (A · dt)
        return (resid.pow(2) / (a_m * float(dt))).sum(dim=(1, 2))

    action_pred = _action(predicted_trajs)
    action_up   = _action(upstream_trajs)

    return {
        "action_pred":       action_pred.detach().cpu().numpy(),
        "action_upstream":   action_up.detach().cpu().numpy(),
        "top_k_used":        int(len(top_idx)),
        "mean_pred":         float(action_pred.mean()),
        "mean_upstream":     float(action_up.mean()),
        "ratio":             float(action_pred.mean()
                                   / max(float(action_up.mean()), 1e-9)),
    }


def print_friedlin_wentzell_report(fw):
    """Print F-W action diagnostic."""
    print()
    print("=" * 72)
    print("  FRIEDLIN-WENTZELL ACTION  (trajectory plausibility under inferred SDE)")
    print("  S[φ] = ∫(φ̇ − b)ᵀA⁻¹(φ̇ − b) dt   (time-marginal b, diagonal A)")
    print("=" * 72)
    if fw is None:
        print("  Not computed.")
        return
    print(f"\n  Restricted to top {fw['top_k_used']} most-variable species")
    print(f"\n  Action of upstream trajectories  (empirical baseline):")
    print(f"    mean = {fw['mean_upstream']:.3e}")
    print(f"    range = [{float(fw['action_upstream'].min()):.3e}, "
          f"{float(fw['action_upstream'].max()):.3e}]")
    print(f"\n  Action of model-predicted trajectories:")
    print(f"    mean = {fw['mean_pred']:.3e}")
    print(f"    range = [{float(fw['action_pred'].min()):.3e}, "
          f"{float(fw['action_pred'].max()):.3e}]")
    print(f"\n  Action ratio  predicted / upstream  = {fw['ratio']:.3f}")
    r = fw["ratio"]
    if r < 1.5:
        verdict = "PLAUSIBLE — predicted trajectories are as probable as upstream samples"
    elif r < 5.0:
        verdict = "MODERATELY IMPROBABLE — predicted trajectories under-sample noise"
    elif r < 50.0:
        verdict = f"IMPROBABLE — model trajectories carry {r:.1f}× the action of upstream"
    else:
        verdict = f"VERY IMPROBABLE — model action is {r:.0f}× upstream baseline"
    print(f"  Verdict: {verdict}")


# ── v15.2 Kramers residence times in cell-cycle phase bins ───────────────────

@torch.no_grad()
def kramers_residence_times(predicted_trajs, upstream_trajs, species_active,
                              n_bins=5):
    """v15.2: Residence-time analysis in lipid-total bins (cell-cycle phases).

    Theory: cell cycle has discrete phases that act like wells in a Kramers-
    style landscape — characteristic states the cell lingers in before
    transitioning.  Lipid total is the canonical cell-cycle proxy in the
    paper validation block (used for doubling-time computation), so binning
    by it gives interpretable 'phases.'

    Mean residence time of upstream vs predicted trajectories in each phase
    tells us *why* the doubling time is off — model spending more time at
    low-lipid bins → escape from G1-like state is too slow; less time → too
    fast.  Cell-cycle timing error gets a mechanistic explanation.

    Args:
      predicted_trajs: (n_pred, T, S) normalised
      upstream_trajs:  (n_up, T, S) normalised
      species_active:  list of S species names
      n_bins:          number of lipid-total bins (cell-cycle phases)

    Returns dict with mean residence per bin (model vs upstream).
    """
    lipid_idx = [i for i, n in enumerate(species_active)
                 if any(n.startswith(p) for p in LIPID_PREFIXES)]
    if not lipid_idx:
        return None
    lipid_pred = predicted_trajs[:, :, lipid_idx].sum(dim=-1)        # (n_pred, T)
    lipid_up   = upstream_trajs[:, :, lipid_idx].sum(dim=-1)         # (n_up, T)
    lo_v       = float(lipid_up.min().item())
    hi_v       = float(lipid_up.max().item())
    if hi_v <= lo_v:
        return None
    bin_edges  = np.linspace(lo_v, hi_v, n_bins + 1)
    bin_inner  = bin_edges[1:-1]

    def _residence(lipid):
        # Per-trajectory fraction of timesteps in each bin
        arr    = lipid.detach().cpu().numpy()
        labels = np.searchsorted(bin_inner, arr).clip(0, n_bins - 1)
        n, _T  = arr.shape
        out    = np.zeros((n, n_bins))
        for b in range(n_bins):
            out[:, b] = (labels == b).mean(axis=1)
        return out

    res_pred = _residence(lipid_pred)
    res_up   = _residence(lipid_up)
    return {
        "bin_edges":            bin_edges,
        "mean_residence_pred":  res_pred.mean(axis=0),
        "mean_residence_up":    res_up.mean(axis=0),
        "std_residence_pred":   res_pred.std(axis=0),
        "std_residence_up":     res_up.std(axis=0),
        "lipid_mean_pred":      float(lipid_pred.mean().item()),
        "lipid_mean_up":        float(lipid_up.mean().item()),
    }


def print_kramers_report(kr):
    """Print Kramers residence-time diagnostic."""
    print()
    print("=" * 72)
    print("  KRAMERS RESIDENCE TIMES IN CELL-CYCLE PHASE BINS")
    print("  Theory: cell-cycle phases = wells; residence-time mismatch = wrong dynamics")
    print("=" * 72)
    if kr is None:
        print("  No lipid species found — Kramers diagnostic skipped.")
        return
    edges = kr["bin_edges"]
    n_b   = len(kr["mean_residence_pred"])
    print(f"\n  Lipid-total mean (normalised):  model={kr['lipid_mean_pred']:.3f}  "
          f"upstream={kr['lipid_mean_up']:.3f}")
    print(f"\n  Mean residence fraction per bin (fraction of timesteps):")
    print(f"  {'bin':>4s}  {'lipid range':<22s} {'upstream':>10s} {'model':>10s} {'Δ':>10s}")
    for b in range(n_b):
        diff = kr["mean_residence_pred"][b] - kr["mean_residence_up"][b]
        flag = ""
        if abs(diff) > 0.10:
            flag = "  ⚠ large mismatch"
        elif abs(diff) > 0.05:
            flag = "  *"
        print(f"  {b:>4d}  [{edges[b]:>9.3f}, {edges[b+1]:>9.3f}]  "
              f"{kr['mean_residence_up'][b]:>10.3f} "
              f"{kr['mean_residence_pred'][b]:>10.3f} "
              f"{diff:>+10.3f}{flag}")
    # Verdict: largest mismatch tells us the failure mode
    diffs = kr["mean_residence_pred"] - kr["mean_residence_up"]
    j_max = int(np.argmax(np.abs(diffs)))
    if abs(diffs[j_max]) > 0.05:
        sign = "MORE" if diffs[j_max] > 0 else "LESS"
        print(f"\n  Verdict: model spends {sign} time at bin {j_max} "
              f"(lipid {edges[j_max]:.2f}-{edges[j_max+1]:.2f}) than upstream.")
        print(f"           {sign} time in this phase → escape rate is "
              f"{'too slow' if diffs[j_max] > 0 else 'too fast'}.")


# ── v15.2 sliced Wasserstein-2 distance per species ──────────────────────────

@torch.no_grad()
def sliced_wasserstein_W2(predicted_trajs, upstream_trajs, top_k_var=200,
                            n_subsample=2000, seed=None):
    """v15.2: 1D sliced W₂ distance per species → variance-weighted aggregate.

    For each species s, compute the 1D W₂² distance between the empirical
    distribution of predicted values and upstream values.  1D W₂² between
    two empirical distributions = mean squared difference of sorted samples.

    Aggregate across species via variance-weighted mean — this is the metric
    JKO (1998) shows is the correct distance for Fokker-Planck flows, so it's
    the right replacement for the misleading per-trajectory R².

    Args:
      predicted_trajs: (n_pred, T, S)
      upstream_trajs:  (n_up, T, S)
      top_k_var:       restrict to top-K most-variable species
      n_subsample:     number of samples per side (cap memory)

    Returns dict with per-species W₂ + variance-weighted aggregate.
    """
    _, _, S = predicted_trajs.shape
    var     = upstream_trajs.reshape(-1, S).var(dim=0)
    top_idx = torch.argsort(var, descending=True)[:int(top_k_var)]
    g       = torch.Generator(device="cpu").manual_seed(
                 int(seed) if seed is not None else SEED + 11)

    w2 = torch.zeros(len(top_idx))
    for j, s in enumerate(top_idx):
        pred = predicted_trajs[:, :, s].reshape(-1).cpu()
        upr  = upstream_trajs[:,  :, s].reshape(-1).cpu()
        n    = min(pred.numel(), upr.numel(), n_subsample)
        if n < 4:
            continue
        ip   = torch.randperm(pred.numel(), generator=g)[:n]
        iu   = torch.randperm(upr.numel(),  generator=g)[:n]
        psort, _ = pred[ip].sort()
        usort, _ = upr[iu].sort()
        w2[j] = ((psort - usort) ** 2).mean()
    var_top = var[top_idx].cpu()
    weighted = (w2 * var_top).sum() / var_top.sum().clamp(min=1e-12)
    return {
        "w2_per_species":   w2.numpy(),
        "top_k_used":       int(len(top_idx)),
        "w2_mean":          float(w2.mean()),
        "w2_var_weighted":  float(weighted),
        "w2_median":        float(w2.median()),
    }


def print_wasserstein_report(ws):
    """Print sliced W₂ diagnostic."""
    print()
    print("=" * 72)
    print("  SLICED W₂ DISTANCE PER SPECIES  (JKO 1998 — the right metric for FPE flows)")
    print("=" * 72)
    if ws is None:
        print("  Not computed.")
        return
    print(f"\n  Computed on top {ws['top_k_used']} most-variable species")
    print(f"\n  Per-species 1D W₂² (lower = predicted distribution matches upstream):")
    print(f"    median           = {ws['w2_median']:.5f}")
    print(f"    mean             = {ws['w2_mean']:.5f}")
    print(f"    variance-weighted = {ws['w2_var_weighted']:.5f}")
    print(f"\n  Note: scale depends on normalisation.  Compare across runs; "
          f"smaller is better.")


# ── v15.2 Helmholtz curl detection (NESS signature) ──────────────────────────

@torch.no_grad()
def helmholtz_curl_diagnostic(predicted_trajs, upstream_trajs, species_active):
    """v15.2: Detect curl flux (NESS signature) via signed enclosed area
    of trajectories in a 2D (lipid total, ATP) projection.

    Theory: NESS dynamics have nonzero curl Q ≠ 0 in the Helmholtz decomposition
    b(x) = (Q(x) + Γ) ∇ ln ρ_ss(x) + ∇·(Q(x) + Γ).  In a 2D projection,
    persistent curl flux shows as a net signed area enclosed by trajectories
    (positive = counter-clockwise circulation in the chosen coordinates).
    Pure gradient-only drift → enclosed area ≈ 0 averaged over trajectories.

    Discrete computation: signed area  A = ½ Σ_t (x̄_t · Δy_t − ȳ_t · Δx_t)
    where (x̄, ȳ) are mid-point averages and (Δx, Δy) are step differences.

    Cells are real NESS systems; we EXPECT nonzero curl.  If model trajectories
    have area ≈ 0 while upstream has area ≠ 0, the model has collapsed to an
    equilibrium-like (pure-gradient) approximation — which would explain
    persistent doubling-time / cell-cycle errors.

    Args:
      predicted_trajs, upstream_trajs: (n, T, S) in normalised space
      species_active: list of S names

    Returns dict with area mean/std for model vs upstream.
    """
    lipid_idx = [i for i, n in enumerate(species_active)
                 if any(n.startswith(p) for p in LIPID_PREFIXES)]
    atp_name = ATP_SPECIES_NAME
    if (not lipid_idx) or (atp_name not in species_active):
        return None
    atp_i = species_active.index(atp_name)

    def _proj(trajs):
        x = trajs[:, :, lipid_idx].sum(dim=-1)              # (n, T)
        y = trajs[:, :, atp_i]                              # (n, T)
        return x, y

    def _signed_area(x, y):
        # Discrete approximation of (1/2) ∮ (x dy - y dx) over each trajectory
        dx    = x[:, 1:] - x[:, :-1]
        dy    = y[:, 1:] - y[:, :-1]
        x_avg = 0.5 * (x[:, :-1] + x[:, 1:])
        y_avg = 0.5 * (y[:, :-1] + y[:, 1:])
        return 0.5 * (x_avg * dy - y_avg * dx).sum(dim=1)   # (n,)

    x_p, y_p = _proj(predicted_trajs)
    x_u, y_u = _proj(upstream_trajs)
    area_p   = _signed_area(x_p, y_p)
    area_u   = _signed_area(x_u, y_u)
    return {
        "area_pred_mean":  float(area_p.mean().item()),
        "area_pred_std":   float(area_p.std().item()),
        "area_up_mean":    float(area_u.mean().item()),
        "area_up_std":     float(area_u.std().item()),
        "n_lipid_species": int(len(lipid_idx)),
    }


def print_helmholtz_report(hl):
    """Print Helmholtz curl-detection diagnostic."""
    print()
    print("=" * 72)
    print("  HELMHOLTZ CURL DETECTION  (NESS signature in 2D projection)")
    print("  Signed area in (lipid_total, ATP) plane:")
    print("    nonzero ↔ curl flux Q ≠ 0 (true NESS);  ~0 ↔ pure gradient (equilibrium)")
    print("=" * 72)
    if hl is None:
        print("  Lipid prefix or ATP species not both present — skipped.")
        return
    a_u_mean = hl["area_up_mean"]
    a_p_mean = hl["area_pred_mean"]
    print(f"\n  Signed enclosed area per trajectory (normalised-space units):")
    print(f"    upstream  mean = {a_u_mean:+.3e}  std = {hl['area_up_std']:.3e}")
    print(f"    predicted mean = {a_p_mean:+.3e}  std = {hl['area_pred_std']:.3e}")
    if abs(a_u_mean) < max(hl["area_up_std"] / 3.0, 1e-9):
        print(f"\n  Upstream curl signal weak in this projection "
              f"(|mean| < std/3) — diagnostic uninformative at this scale.")
        return
    ratio = a_p_mean / a_u_mean
    print(f"\n  Curl ratio  predicted / upstream = {ratio:+.3f}")
    if abs(ratio) < 0.3:
        verdict = ("model has near-zero curl flux — likely treating cell as "
                   "equilibrium (pure gradient).  This is consistent with persistent "
                   "doubling-time errors.")
    elif abs(ratio - 1.0) < 0.3:
        verdict = "model captures the curl flux at upstream level."
    elif ratio < 0:
        verdict = (f"model curl has WRONG SIGN (ratio {ratio:+.2f}) — predicted "
                   f"trajectories rotate opposite to upstream in this projection.")
    else:
        verdict = (f"model curl is {ratio:.1f}× upstream — magnitude mismatch but "
                   f"correct sign.")
    print(f"  Verdict: {verdict}")


# ── v15.8 three-reframes tests ────────────────────────────────────────────────
#
# Empirical tests for three philosophical reframes of "what a cell is":
#   R1: cell as computation/algorithm (binary control logic captures dynamics)
#   R2: cell as constraint-satisfaction (invariants determine behaviour, not
#       forward dynamics — like least-action vs Newton)
#   R3: cell as low-dimensional manifold (~10⁴ species collapse to ~10² dims)
#
# Each test produces a numerical verdict the user can compare against the
# trained model.  They are NOT prescriptive — they tell us which reframe is
# most empirically supported by THIS dataset.

def reframe_lowdim_manifold(raw_counts_active, train_idx, n_sample=2000,
                             thresholds=(0.50, 0.80, 0.90, 0.95, 0.99)):
    """v15.8 Reframe 3: intrinsic dimensionality of the trajectory tensor.

    Tests "the cell lives in a low-dimensional manifold" by PCA on the
    log-space normalised concatenated training trajectories.  Reports
    cumulative explained variance at thresholds, plus participation ratio
    (PR = (Σλ_i)² / Σλ_i² — effective number of equally-contributing dims).

    Returns dict with PCs_at_thresholds, participation ratio, total dims.
    """
    rc = raw_counts_active[train_idx].astype(np.float64)
    n_t, T, S = rc.shape
    X = rc.reshape(-1, S)
    # signed-log to match what the model sees
    X = np.sign(X) * np.log1p(np.abs(X))
    X = X - X.mean(axis=0, keepdims=True)
    # subsample timepoints to keep SVD memory-bounded
    n_rows = X.shape[0]
    if n_rows > n_sample:
        idx = np.random.default_rng(0).choice(n_rows, n_sample, replace=False)
        X = X[idx]
    # SVD on the centered data; eigenvalues of the covariance = singular_values²/(N-1)
    try:
        _, sv, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    eig = (sv ** 2) / max(X.shape[0] - 1, 1)
    total = float(eig.sum())
    if total <= 0:
        return None
    var_frac = eig / total
    cv = np.cumsum(var_frac)
    pc_at = {}
    for t in thresholds:
        k = int(np.searchsorted(cv, t)) + 1
        pc_at[t] = min(k, S)
    pr = float(total * total / max(float((eig ** 2).sum()), 1e-30))
    return {
        "n_trajectories":     int(n_t),
        "n_timepoints":       int(T),
        "n_species":          int(S),
        "n_samples_used":     int(X.shape[0]),
        "pc_at_thresholds":   pc_at,
        "participation_ratio": pr,
        "top_eig_frac":       float(var_frac[0]) if eig.size else float("nan"),
        "top10_eig_frac":     float(var_frac[:10].sum()) if eig.size else float("nan"),
    }


@torch.no_grad()
def reframe_constraints_only(test_X, ruleset, top_k_var=VAR_R2_TOP_K):
    """v15.8 Reframe 2: how much rollout R² do constraints alone capture?

    Runs two baselines using the SAME ruleset.project that the trained model
    uses:
      (a) pure persistence:  x_{t+1} = x_t                (no constraints)
      (b) persistence + project: x_{t+1} = ruleset.project(x_t, x_t)
      (c) mean-trend + project:  x_{t+1} = clamp(x_t + Δ̄) then project
          (Δ̄ = mean step delta across all training trajectories — but here
          we compute it from test_X itself as a fair self-reference)

    The gap (c)−(a) quantifies how much predictable trajectory follows from
    constraints + linear trend WITHOUT any learned dynamics.  Compared to
    the trained model's honest R², this tells us whether dynamics learning
    adds real value or constraints carry most of it.
    """
    if test_X.dim() != 3:
        return None
    n, T, S = test_X.shape
    if T < 2:
        return None
    # Mean step delta from test_X itself (self-reference; honest signal)
    dx = test_X[:, 1:] - test_X[:, :-1]                                  # (n, T-1, S)
    mean_delta = dx.mean(dim=(0, 1))                                     # (S,)

    def _rollout(use_project, use_trend):
        state = test_X[:, 0]                                             # (n, S)
        preds = []
        for _ in range(T - 1):
            nxt = (state + mean_delta) if use_trend else state.clone()
            nxt = nxt.clamp(CLAMP_LO, CLAMP_HI)
            if use_project and ruleset is not None:
                nxt = ruleset.project(state, nxt)
            preds.append(nxt)
            state = nxt
        return torch.stack(preds, dim=1)                                 # (n, T-1, S)

    truth = test_X[:, 1:]
    out = {}
    for label, p, t in [
        ("persistence_only",          _rollout(False, False), truth),
        ("persistence_plus_rules",    _rollout(True,  False), truth),
        ("trend_plus_rules",          _rollout(True,  True),  truth),
    ]:
        r2_top, n_top = variance_weighted_r2(p, t, top_k=top_k_var)
        # also mean per-traj R² for cross-reference with rollout R²
        r2_per_traj = []
        for k in range(p.shape[0]):
            num = ((p[k] - t[k]) ** 2).sum()
            den = ((t[k] - t[k].mean(dim=0, keepdim=True)) ** 2).sum().clamp(min=1e-9)
            r2_per_traj.append(float(1.0 - num / den))
        out[label] = {
            "honest_r2_top_k":   float(r2_top),
            "mean_traj_r2":      float(np.mean(r2_per_traj)),
            "n_top_valid":       int(n_top),
        }
    out["top_k"] = int(top_k_var)
    return out


def reframe_binary_algorithm(raw_counts_active, train_idx, species_active,
                              lag=10, n_sample=200, n_bins=2):
    """v15.8 Reframe 1: does binary discretisation preserve predictive power?

    Test: threshold each species at its training-set median.  Compute
    mutual information between binary state at time t and binary state at
    time t+lag (same species, MI is in nats).  If the cell behaves like
    an algorithm operating on ON/OFF state, binary MI should stay close
    to its theoretical maximum (ln 2 ≈ 0.69 nats).  If continuous detail
    matters, binary MI collapses.

    Per-class breakdown via species-name prefix:
      regulatory: G_*, R_* (gene activity, mRNA — natural switch-like)
      protein:    P_*, RP_*
      metabolic:  M_* (metabolites — natural graded)
      other:      ribosomeP, RB_*, RPM_*, lipids, etc.
    """
    rc = raw_counts_active[train_idx]                                # (n, T, S)
    n_t, T, S = rc.shape
    if T < lag + 1 or S < 1:
        return None
    rng = np.random.default_rng(0)
    # Drop near-constant species — MI is uninformative on them
    var = rc.reshape(-1, S).var(axis=0)
    candidates = np.where(var > 1e-9)[0]
    if candidates.size < n_sample:
        chosen = candidates
    else:
        chosen = rng.choice(candidates, n_sample, replace=False)
    medians = np.median(rc.reshape(-1, S)[:, chosen], axis=0)
    binary  = (rc[:, :, chosen] > medians[None, None, :]).astype(np.int8)

    def _bin_mi(x_now, x_fut):
        # MI in nats from a 2x2 contingency table
        flat = (x_now.astype(np.int32) << 1) | x_fut.astype(np.int32)
        counts = np.bincount(flat, minlength=4).astype(np.float64)
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        # marginals
        p_now = np.array([p[0] + p[1], p[2] + p[3]])
        p_fut = np.array([p[0] + p[2], p[1] + p[3]])
        mi = 0.0
        for a in (0, 1):
            for b in (0, 1):
                pij = p[(a << 1) | b]
                if pij > 0 and p_now[a] > 0 and p_fut[b] > 0:
                    mi += pij * np.log(pij / (p_now[a] * p_fut[b]))
        return float(mi)

    mis = np.zeros(len(chosen), dtype=np.float64)
    for s_i, s in enumerate(chosen):
        x_now = binary[:, :T - lag, s_i].ravel()
        x_fut = binary[:, lag:, s_i].ravel()
        mis[s_i] = _bin_mi(x_now, x_fut)

    def _class_of(name):
        if name.startswith(("G_",)):       return "regulatory_gene"
        if name.startswith(("R_",)) and not name.startswith("R_dummy"): return "mrna"
        if name.startswith(("P_", "RP_")): return "protein"
        if name.startswith("M_"):          return "metabolite"
        return "other"

    classes = {}
    for s_i, s in enumerate(chosen):
        c = _class_of(species_active[s])
        classes.setdefault(c, []).append(mis[s_i])
    class_summary = {c: {"mean_mi": float(np.mean(v)),
                          "n":      int(len(v))} for c, v in classes.items()}
    return {
        "lag":         int(lag),
        "n_species_sampled": int(len(chosen)),
        "mean_mi":     float(np.mean(mis)),
        "median_mi":   float(np.median(mis)),
        "max_mi_theoretical": float(np.log(n_bins)),
        "preservation_fraction": float(np.mean(mis) / np.log(n_bins)),
        "per_class":   class_summary,
        "high_mi_count": int(np.sum(mis > 0.5 * np.log(n_bins))),
        "low_mi_count":  int(np.sum(mis < 0.1 * np.log(n_bins))),
    }


def print_reframe_report(r3, r2, r1, current_honest_r2=None, ceiling=None):
    """Print the three-reframes verdict block (v15.8)."""
    print()
    print("=" * 72)
    print("  THREE REFRAMES — empirical verdicts (v15.8)")
    print("  R1: cell as algorithm   R2: cell as constraints   R3: cell as low-dim")
    print("=" * 72)

    # ----- Reframe 3 -----
    print("\n  Reframe 3 (low-dimensional manifold):")
    if r3 is None:
        print("    skipped")
    else:
        pc = r3["pc_at_thresholds"]
        S = r3["n_species"]
        print(f"    PCs needed for X% variance  (out of {S} species, "
              f"{r3['n_samples_used']} timepoints sampled):")
        for thr in sorted(pc):
            k = pc[thr]
            print(f"      {int(thr*100):>2d}% var: {k:>4d} PCs  ({100.0*k/S:>4.1f}% of species)")
        pr = r3["participation_ratio"]
        print(f"    participation ratio (effective dim): {pr:.1f}")
        print(f"    top 1 PC explains {100*r3['top_eig_frac']:.1f}% of variance; "
              f"top 10 explain {100*r3['top10_eig_frac']:.1f}%")
        k95 = pc.get(0.95, S)
        if k95 <= 0.05 * S:
            v3 = (f"STRONGLY SUPPORTED — {k95}/{S} ({100*k95/S:.1f}%) PCs cover 95%. "
                  f"Cell lives in a low-dimensional manifold; molecular detail is mostly noise.")
        elif k95 <= 0.20 * S:
            v3 = (f"PARTIALLY SUPPORTED — {k95}/{S} ({100*k95/S:.1f}%) PCs cover 95%. "
                  f"Modest dimensional collapse; substantial residual detail.")
        else:
            v3 = (f"NOT SUPPORTED — {k95}/{S} ({100*k95/S:.1f}%) PCs needed. "
                  f"Variance is spread across many dimensions.")
        print(f"    verdict: {v3}")

    # ----- Reframe 2 -----
    print("\n  Reframe 2 (cell defined by constraints, not dynamics):")
    if r2 is None:
        print("    skipped")
    else:
        po = r2["persistence_only"]["honest_r2_top_k"]
        pr = r2["persistence_plus_rules"]["honest_r2_top_k"]
        tr = r2["trend_plus_rules"]["honest_r2_top_k"]
        po_t = r2["persistence_only"]["mean_traj_r2"]
        pr_t = r2["persistence_plus_rules"]["mean_traj_r2"]
        tr_t = r2["trend_plus_rules"]["mean_traj_r2"]
        print(f"    honest R² (top-{r2['top_k']} species, median):")
        print(f"      persistence only       = {po:+.3f}   (per-traj {po_t:+.3f})")
        print(f"      persistence + ruleset  = {pr:+.3f}   (per-traj {pr_t:+.3f})")
        print(f"      trend + ruleset        = {tr:+.3f}   (per-traj {tr_t:+.3f})")
        if current_honest_r2 is not None:
            print(f"      trained model          = {current_honest_r2:+.3f}")
        if ceiling is not None:
            print(f"      stochastic ceiling     = {ceiling:+.3f}")
        gain_from_constraints = pr - po
        gain_from_trend       = tr - pr
        if current_honest_r2 is not None:
            gain_from_model = current_honest_r2 - tr
            print(f"    decomposition of honest R²:")
            print(f"      constraints add over persistence:  {gain_from_constraints:+.3f}")
            print(f"      linear trend adds over constraints:{gain_from_trend:+.3f}")
            print(f"      trained dynamics add over trend:   {gain_from_model:+.3f}")
            if abs(gain_from_constraints) > 0.05 and gain_from_model < gain_from_constraints:
                v2 = ("STRONGLY SUPPORTED — constraints carry more R² than learned "
                      "dynamics.  A constraint-discovery loop would likely outperform "
                      "further dynamics tuning.")
            elif gain_from_constraints > 0.02 and gain_from_model > 0.02:
                v2 = ("PARTIALLY SUPPORTED — constraints carry real predictive power, "
                      "but learned dynamics still add similar magnitude.  Both layers matter.")
            elif gain_from_model > 2 * abs(gain_from_constraints):
                v2 = ("NOT SUPPORTED — learned dynamics add much more than constraints. "
                      "The cell is not well-described as constraint-satisfaction here.")
            else:
                v2 = "verdict mixed — gains are comparable and small."
            print(f"    verdict: {v2}")

    # ----- Reframe 1 -----
    print("\n  Reframe 1 (cell as computation/algorithm):")
    if r1 is None:
        print("    skipped")
    else:
        mi   = r1["mean_mi"]
        med  = r1["median_mi"]
        mx   = r1["max_mi_theoretical"]
        pres = r1["preservation_fraction"]
        print(f"    binary MI(state_t, state_t+{r1['lag']}) over "
              f"{r1['n_species_sampled']} species:")
        print(f"      mean   = {mi:.3f} nats  ({100*pres:.1f}% of ln2={mx:.3f})")
        print(f"      median = {med:.3f} nats")
        print(f"      switch-like (MI > 0.5·ln2):     {r1['high_mi_count']}")
        print(f"      noise-like  (MI < 0.1·ln2):     {r1['low_mi_count']}")
        if r1.get("per_class"):
            print(f"    per-class mean binary MI (nats):")
            for c, d in sorted(r1["per_class"].items(), key=lambda kv: -kv[1]["mean_mi"]):
                print(f"      {c:<20s} {d['mean_mi']:.3f}  (n={d['n']})")
        if pres >= 0.70:
            v1 = ("STRONGLY SUPPORTED — binary state preserves >70% of predictive "
                  "structure.  The cell really does look algorithmic.")
        elif pres >= 0.40:
            v1 = ("PARTIALLY SUPPORTED — binary state preserves substantial structure. "
                  "Mixed regime: some species are switch-like, others graded.")
        else:
            v1 = ("NOT SUPPORTED — binary state loses most predictive structure. "
                  "Continuous detail matters.")
        print(f"    verdict: {v1}")

    print("\n  Strategic implication:")
    print("    Which reframe is MOST supported here determines the next architectural")
    print("    move (Koopman/DMD core for R3, constraint-discovery loop for R2,")
    print("    discrete switching layer for R1, or hybrid if multiple are supported).")


# ── v15.2 constraint provenance audit ─────────────────────────────────────────

@torch.no_grad()
def constraint_provenance(model, worst_species_idx, species_active,
                            species_type_ids, edge_index, raw_counts_active,
                            train_idx):
    """v15.2: Trace each worst-predicted species back to its constraint sources.

    The recurring debugging question: when species X is predicted badly, where
    did the bias enter the model?  Walk the chain from the prediction back to
    the data source — MetabolismCore, CentralDogmaCore, PINN, graph edges,
    or pure LGNN residual.  Indicates which knob to actually turn.

    Args:
      model: trained DynamicsModel
      worst_species_idx: list of species indices (from analyze_gaps)
      species_active: list of species names
      species_type_ids: per-species type label (0=protein, 1=tRNA, ...)
      edge_index: (2, E) graph edges
      raw_counts_active: (n_traj, T, S) raw counts
      train_idx: indices into raw_counts_active that were used for training

    Returns: list of per-species trace dicts.
    """
    _inner = getattr(model, "_orig_mod", model)

    def _mask(core_or_head, attr):
        if core_or_head is None:
            return None
        m = getattr(core_or_head, attr, None)
        return m.detach().cpu().numpy() if m is not None else None

    metab_mask = _mask(_inner.metab_core, "coverage_mask")
    cd_mask    = _mask(getattr(_inner, "cd_core", None), "coverage_mask")
    asm_mask   = _mask(getattr(_inner, "asm_core", None), "coverage_mask")
    pinn_mask  = _mask(getattr(_inner, "pinn_head", None), "sbml_mask") \
                  if getattr(_inner, "use_pinn", False) else None

    # Per-species cfc_A/cfc_B norm summed across layers — proxy for how much
    # per-species memorization is loaded for this species
    layers = list(getattr(_inner, "layers", []))
    cfc_norm = np.zeros(len(species_active))
    for layer in layers:
        A = getattr(layer, "cfc_A", None)
        B = getattr(layer, "cfc_B", None)
        if A is not None and B is not None:
            cfc_norm = cfc_norm + (A.norm(dim=1).detach().cpu().numpy()
                                    + B.norm(dim=1).detach().cpu().numpy())
    if layers:
        cfc_norm /= len(layers)

    # Edge degrees per species
    edge_arr   = edge_index.detach().cpu().numpy()
    n_out      = np.zeros(len(species_active), dtype=int)
    n_in       = np.zeros(len(species_active), dtype=int)
    np.add.at(n_out, edge_arr[0], 1)
    np.add.at(n_in,  edge_arr[1], 1)

    init_mean = raw_counts_active[train_idx, 0].mean(axis=0)

    GTYPE_NAMES = {GTYPE_PROTEIN: "protein", GTYPE_TRNA: "tRNA",
                   GTYPE_RRNA: "rRNA", GTYPE_OTHER: "other",
                   GTYPE_GLOBAL: "global"}

    traces = []
    for sp_i in worst_species_idx:
        sp_i = int(sp_i)
        gt   = int(species_type_ids[sp_i])
        traces.append({
            "idx":            sp_i,
            "name":           species_active[sp_i],
            "gtype":          GTYPE_NAMES.get(gt, f"t{gt}"),
            "metab_covered":  bool(metab_mask[sp_i]) if metab_mask is not None else False,
            "cd_covered":     bool(cd_mask[sp_i])    if cd_mask    is not None else False,
            "asm_covered":    bool(asm_mask[sp_i])   if asm_mask   is not None else False,
            "pinn_covered":   bool(pinn_mask[sp_i])  if pinn_mask  is not None else False,
            "n_edges_out":    int(n_out[sp_i]),
            "n_edges_in":     int(n_in[sp_i]),
            "init_mean":      float(init_mean[sp_i]),
            "cfc_norm":       float(cfc_norm[sp_i]),
        })
    return traces


def print_provenance_report(traces):
    """Print constraint-provenance audit."""
    print()
    print("=" * 72)
    print("  CONSTRAINT PROVENANCE AUDIT")
    print("  For each worst-predicted species: where did its dynamics come from?")
    print("=" * 72)
    if not traces:
        print("  Nothing to trace.")
        return
    print(f"\n  Worst {len(traces)} predicted species and their constraint origins:")
    print(f"  {'name':<24s} {'type':<8s} {'metab':>6s} {'cd':>4s} {'pinn':>5s} "
          f"{'asm':>4s} {'n_out':>6s} {'n_in':>5s} {'init':>10s} {'|cfc|':>7s}")
    print("  " + "-" * 78)
    for t in traces:
        print(f"  {t['name'][:24]:<24s} {t['gtype'][:8]:<8s} "
              f"{'Y' if t['metab_covered']  else '-':>6s} "
              f"{'Y' if t['cd_covered']     else '-':>4s} "
              f"{'Y' if t['pinn_covered']   else '-':>5s} "
              f"{'Y' if t['asm_covered']    else '-':>4s} "
              f"{t['n_edges_out']:>6d} {t['n_edges_in']:>5d} "
              f"{t['init_mean']:>10.2e} {t['cfc_norm']:>7.3f}")
    n_metab     = sum(1 for t in traces if t["metab_covered"])
    n_cd        = sum(1 for t in traces if t["cd_covered"])
    n_pinn      = sum(1 for t in traces if t["pinn_covered"])
    n_no_core   = sum(1 for t in traces
                      if not (t["metab_covered"] or t["cd_covered"]
                               or t["pinn_covered"] or t["asm_covered"]))
    print(f"\n  Where the failure lives (multi-coverage possible):")
    print(f"    {n_metab}/{len(traces)} in MetabolismCore   → debug kinetic params (k_cat, K_m, ΔG°)")
    print(f"    {n_cd}/{len(traces)} in CentralDogmaCore  → debug per-gene k_tx/k_tl calibration")
    print(f"    {n_pinn}/{len(traces)} in PINN             → debug SBML stoichiometric matrix")
    print(f"    {n_no_core}/{len(traces)} pure-LGNN          → debug graph edges + cfc_A/cfc_B")


# ── missing-info report ───────────────────────────────────────────────────────

@torch.no_grad()
def analyze_gaps(model, test_X, species_names, species_type_ids,
                 ruleset, hyp, sbml, elem_balances,
                 preds_batch=None, truth_batch=None):
    """Where the model + rules fall short - residuals, drifts, coverage.

    v10: accept pre-computed batched preds/truth from full_rollout_batched to
    avoid re-rolling the trajectory (which is the eval-phase bottleneck at
    full resolution: 7,199 steps × 10 trajs).
    """
    print()
    print("#" * 72)
    print("#  MISSING-INFO REPORT  -  where the model / rules fall short")
    print("#" * 72)

    S = test_X.shape[2]
    se = torch.zeros(S, device=test_X.device)
    if preds_batch is not None and truth_batch is not None:
        se = ((preds_batch - truth_batch) ** 2).mean(dim=(0, 1))   # (S,)
    else:
        for k in range(test_X.shape[0]):
            pred, true = full_rollout(model, test_X[k], ruleset)
            se += ((pred - true) ** 2).mean(0)
        se /= test_X.shape[0]
    worst = torch.argsort(se, descending=True)[:15].tolist()
    tname = {GTYPE_PROTEIN: "protein", GTYPE_TRNA: "tRNA", GTYPE_RRNA: "rRNA",
             GTYPE_OTHER: "other", GTYPE_GLOBAL: "global"}
    print("\n  [1] species the model predicts worst (rollout MSE) - need mechanism we lack:")
    for i in worst:
        print(f"      {species_names[i]:24s}  MSE {float(se[i]):.4f}  "
              f"({tname[int(species_type_ids[i])]})")

    print("\n  [2] element balances (SBML) - drift = unmodeled flux:")
    if elem_balances:
        for d in sorted(elem_balances, key=lambda x: -x["drift_frac"]):
            tag = "CONSERVED" if d["conserved"] else "drifts"
            print(f"      {d['element']:3s}  {tag:9s}  "
                  f"{d['drift_frac']*100:6.1f}%  over the cell cycle")
    else:
        print("      (no SBML provided - skipped)")

    print("\n  [3] SBML <-> trajectory coverage:")
    if sbml is not None:
        traj_set = set(species_names); sbml_set = set(sbml["species"])
        print(f"      SBML species          : {len(sbml_set)}")
        print(f"      matched in trajectory : {len(traj_set & sbml_set)}")
        print(f"      SBML-only (untracked) : {len(sbml_set - traj_set)}")
        rxn_sp = {s for r in sbml["reactions"]
                  for s, _ in r["reactants"] + r["products"]}
        print(f"      SBML reaction species not in trajectory: "
              f"{len(rxn_sp - traj_set)}  (missing linkage)")
    else:
        print("      (no SBML provided - skipped)")

    print()
    print("  [4] " + hyp.summary().replace("\n", "\n  "))
    print("#" * 72)


# ── v9: knockout sweep (Breuer 2019 essentiality MCC) ────────────────────────

@torch.no_grad()
def _ko_rollout(model, state, ko_mask, n_steps):
    """Roll forward N steps with PERMANENT knockdown of species in ko_mask.

    Two deliberate choices vs the normal eval rollout:
      - re-apply the knockout at every step (gene deletion is permanent — if we
        only zero the seed, the model "fills in" the species at step 1 and the
        perturbation evaporates);
      - skip ruleset.project, otherwise the bounds rule clamps the knocked-out
        species back up to its validated training range and the knockout is
        immediately undone.

    state: (B, S);  ko_mask: (B, S) bool;  returns (B, n_steps, S).
    """
    _inner = getattr(model, "_orig_mod", model)
    use_temporal_ctx = getattr(_inner, "use_temporal_context", False)
    t_ctx_window = getattr(_inner, "t_ctx_window", T_CTX_WINDOW)
    state_history = (state.unsqueeze(1).repeat(1, t_ctx_window, 1)
                     if use_temporal_ctx else None)
    preds = []
    for _ in range(n_steps):
        with _eval_autocast():
            out = (model(state, state_history=state_history)
                   if use_temporal_ctx else model(state))
            p = _model_pred(out).float().clamp(CLAMP_LO, CLAMP_HI)
        # Permanent knockdown: force every masked species back to floor
        p = torch.where(ko_mask, torch.full_like(p, CLAMP_LO), p)
        preds.append(p)
        state = p
        if use_temporal_ctx:
            state_history = torch.cat(
                [state_history[:, 1:], state.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


@torch.no_grad()
def knockout_sweep(model, ruleset, test_X, species_names, breuer_labels,
                   n_steps=KO_N_STEPS, batch_size=KO_BATCH_SIZE):
    """v9: in-silico gene knockouts ranked by trajectory deviation, scored
    against Breuer 2019 essentiality.

    For each candidate gene: build a ko_mask flagging its P/R/RP/G species,
    apply permanent knockdown for n_steps, measure MSE deviation from the
    unperturbed baseline rollout.  Top-N predicted-essential = experimentally
    essential set; MCC quantifies overlap.
    """
    if not breuer_labels:
        return None
    model.eval()
    gene_cols = {}
    for i, name in enumerate(species_names):
        pre, loc = parse_species(name)
        if pre in {"P", "R", "RP", "G"}:
            gene_cols.setdefault(_locus_key(loc), []).append(i)
    candidates = sorted(gene_cols.keys())
    if not candidates:
        return None

    S = test_X.shape[2]
    seed = test_X[0, 0]                                              # (S,)
    no_ko_mask = torch.zeros(1, S, dtype=torch.bool, device=seed.device)
    baseline = _ko_rollout(model, seed.unsqueeze(0), no_ko_mask, n_steps).squeeze(0)

    impacts = {}
    for i in range(0, len(candidates), batch_size):
        batch_loci = candidates[i:i + batch_size]
        ko_mask = torch.zeros(len(batch_loci), S, dtype=torch.bool, device=seed.device)
        for b, loc in enumerate(batch_loci):
            ko_mask[b, gene_cols[loc]] = True
        states = seed.unsqueeze(0).expand(len(batch_loci), -1).clone()
        states[ko_mask] = CLAMP_LO                                   # initial KO
        ko_trajs = _ko_rollout(model, states, ko_mask, n_steps)      # (B, n_steps, S)
        for b, loc in enumerate(batch_loci):
            impacts[loc] = float(((baseline - ko_trajs[b]) ** 2).mean())
    ranking = sorted(impacts.items(), key=lambda x: -x[1])

    # MCC: Essential ∪ Quasiessential vs Nonessential
    true_e  = {loc for loc, lab in breuer_labels.items()
               if lab in {"Essential", "Quasiessential"} and loc in impacts}
    true_n  = {loc for loc, lab in breuer_labels.items()
               if lab == "Nonessential" and loc in impacts}
    if not true_e or not true_n:
        return {"ranking": ranking, "mcc": float("nan"),
                "n_genes": len(candidates), "n_essential": len(true_e),
                "n_nonessential": len(true_n)}
    n_top = len(true_e)
    pred_top = {loc for loc, _ in ranking[:n_top]}
    tp = len(pred_top & true_e)
    fp = len(pred_top & true_n)
    fn = len(true_e - pred_top)
    tn = len(true_n - pred_top)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / (denom ** 0.5) if denom > 0 else 0.0

    return {
        "ranking": ranking, "mcc": mcc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_genes": len(candidates),
        "n_essential": len(true_e),
        "n_nonessential": len(true_n),
    }


def print_knockout_report(ko, breuer_labels):
    """Pretty-print the knockout-sweep result."""
    print()
    print("#" * 72)
    print("#  KNOCKOUT SWEEP  -  in-silico essentiality vs Breuer 2019")
    print("#" * 72)
    if ko is None:
        print("  (no Breuer labels available - skipped)")
        print("#" * 72)
        return
    print(f"  {ko['n_genes']} genes tested over {KO_N_STEPS} rollout steps "
          "(permanent knockdown, no rule projection)")
    print(f"  Breuer 2019 labels: {ko['n_essential']} essential, "
          f"{ko['n_nonessential']} non-essential (in our species set)")
    # Impact distribution diagnostic — tells us whether knockouts moved anything
    impact_vals = [v for _, v in ko["ranking"]]
    if impact_vals:
        print(f"  Impact range: min={min(impact_vals):.2e}  "
              f"median={impact_vals[len(impact_vals)//2]:.2e}  "
              f"max={max(impact_vals):.2e}")
        if max(impact_vals) < 1e-3:
            print("  WARNING: all impacts < 1e-3 — knockouts barely perturbing "
                  "the trajectory.  Model is bias-driven, not causally responsive.")
    if not (ko.get("mcc") == ko.get("mcc")):     # NaN check
        print("  MCC: undefined (one class empty)")
    else:
        print(f"  Confusion: TP={ko['tp']}  FP={ko['fp']}  FN={ko['fn']}  TN={ko['tn']}")
        print(f"  MCC = {ko['mcc']:+.3f}  "
              f"({'random' if abs(ko['mcc']) < 0.15 else 'weak' if abs(ko['mcc']) < 0.3 else 'moderate' if abs(ko['mcc']) < 0.5 else 'strong'} agreement)")
    print()
    print("  Top 12 predicted-essential genes (by knockout impact):")
    for loc, impact in ko["ranking"][:12]:
        lab = breuer_labels.get(loc, "unknown")
        flag = "✓" if lab in {"Essential", "Quasiessential"} else "✗" if lab == "Nonessential" else "?"
        print(f"      {flag} JCVISYN3A_{loc}  impact={impact:.3e}  ({lab})")
    print("#" * 72)


# ── v15.9: static essentiality classifier + synthetic lethality ──────────────

# Keyword categories ported from the vectorize-gex branch's static_node_features
# (v15 keyword-prior detector, MCC 0.5372 vs Breuer 2019).  These catch the
# obvious essentials (ribosomes / polymerases / synthetases) from annotation text.
_ESS_KEYWORDS = {
    "is_ribosomal":   ("ribosomal protein", "ribosomal subunit", "30s ", "50s ", "ribosom"),
    "is_synthetase":  ("synthetase", "--trna ligase", "trna ligase", "aminoacyl"),
    "is_trna":        ("trna",),
    "is_polymerase":  ("polymerase", "primase", "helicase", "gyrase", "topoisomerase"),
    "is_dna_machinery": ("dna ", "replication", "chromosom", "ftsz", "cell division", "smc"),
    "is_translation": ("translation", "elongation factor", "initiation factor",
                       "release factor", "ef-tu", "ef-g", "trna-modifying"),
    "is_transcription": ("transcription", "sigma factor", "rna polymerase", "nusa", "nusg"),
    "is_membrane":    ("membrane", "transporter", "permease", "translocase",
                       "abc transport", "secy", "atp synthase"),
    "is_atp_binding": ("atp-binding", "atpase", "atp synthase", "gtpase", "gtp-binding"),
    "is_kinase":      ("kinase",),
    "is_phosphatase": ("phosphatase",),
    "is_synthase":    ("synthase",),
    "is_metabolism":  ("dehydrogenase", "reductase", "hydrolase", "transferase",
                       "isomerase", "lyase", "oxidase", "carboxylase", "phosphorylase",
                       "nuclease", "deaminase", "phosphoribosyltransferase"),
    "is_uncharacterized": ("uncharacterized", "hypothetical", "putative", "unknown"),
}
_ESS_KEYWORD_COLS = list(_ESS_KEYWORDS.keys()) + ["protein_length_aa_log",
                                                  "has_gene_name", "length_bp_log"]
# Currency metabolites (top-degree hubs from the SBML analysis: appear in 27-41%
# of all reactions).  An enzyme that touches one of these sits on a central flux.
_CURRENCY_SPECIES = ("M_h_c", "M_atp_c", "M_h2o_c", "M_adp_c", "M_pi_c",
                     "M_ppi_c", "M_amp_c", "M_nad_c", "M_nadh_c", "M_nadp_c")


def _essentiality_keyword_features(product, gene_name, length_bp):
    """v15.9: 17 annotation-keyword features for one gene (ports v15 detector)."""
    p = (product or "").lower()
    feats = {}
    for col, kws in _ESS_KEYWORDS.items():
        feats[col] = 1.0 if any(k in p for k in kws) else 0.0
    # protein length in amino acids (bp/3); log1p
    aa = (float(length_bp) / 3.0) if length_bp and length_bp > 0 else 0.0
    feats["protein_length_aa_log"] = float(np.log1p(aa))
    feats["has_gene_name"] = 1.0 if (gene_name and str(gene_name).strip()
                                      and str(gene_name).lower() != "nan") else 0.0
    feats["length_bp_log"] = float(np.log1p(float(length_bp))) if length_bp and length_bp > 0 else 0.0
    return feats


def build_essentiality_feature_matrix(
        species_active, gene_products, gene_meta, breuer_labels,
        edge_index=None, ruleset=None, patterns=None, sbml=None,
        kinetics=None, raw_counts_active=None, train_idx=None):
    """v15.9: build keyword + pipeline feature matrices per labelled locus.

    Returns dict with X_kw, X_pipe, X_all, y, loci, names_kw, names_pipe.
    Labels: Essential ∪ Quasiessential = 1, Nonessential = 0 (other dropped).
    """
    # locus -> list of species indices, by prefix category
    locus_species = {}
    for i, name in enumerate(species_active):
        pre, loc = parse_species(name)
        key = _locus_key(loc) if pre else None
        if pre in {"P", "R", "RP", "G", "PM", "RPM"} and key:
            locus_species.setdefault(key, []).append((pre, i))

    # ── precompute pipeline lookups ──
    # graph degree per species
    deg = None
    if edge_index is not None:
        ea = edge_index.detach().cpu().numpy()
        deg = np.zeros(len(species_active), dtype=np.float64)
        np.add.at(deg, ea[0], 1.0); np.add.at(deg, ea[1], 1.0)
    # cpair membership per species (HARD Tier-1 pairs)
    cpair_member = np.zeros(len(species_active), dtype=np.float64)
    if ruleset is not None and getattr(ruleset, "cpair_i", None) is not None:
        for idx in ruleset.cpair_i.tolist() + ruleset.cpair_j.tolist():
            if 0 <= idx < len(cpair_member):
                cpair_member[idx] += 1.0
    # lens-composition count per species (how many lenses flagged it)
    central = getattr(patterns, "central_species", {}) if patterns is not None else {}
    comp_count = np.zeros(len(species_active), dtype=np.float64)
    for idx, lenses in central.items():
        if 0 <= idx < len(comp_count):
            comp_count[idx] = float(len(lenses))
    # granger out-degree per species (how many directed edges it drives)
    granger_out = np.zeros(len(species_active), dtype=np.float64)
    for (a, b, *_ ) in (getattr(patterns, "granger", []) if patterns is not None else []):
        if 0 <= a < len(granger_out):
            granger_out[a] += 1.0
    # n_reactions catalysed + currency coupling per locus, via kinetics enzyme map
    enzyme_rxn = {}
    currency_rxn = set()
    if sbml is not None:
        cur = set(_CURRENCY_SPECIES)
        for r in sbml["reactions"]:
            touches = any(sid in cur for sid, _ in r["reactants"] + r["products"])
            if touches:
                currency_rxn.add(r["id"])
    if kinetics is not None:
        for rid, enz in kinetics.get("enzymes", {}).items():
            # enzyme string may be a locus tag or 'JCVISYN3A_xxxx'
            m = re.search(r"(\d{4})", str(enz))
            if m:
                enzyme_rxn.setdefault(m.group(1), []).append(rid)

    # cross-replicate trajectory stats per species (linear space)
    def _traj_stats(idxs):
        if raw_counts_active is None or train_idx is None or not idxs:
            return 0.0, 0.0, 0.0
        arr = raw_counts_active[train_idx][:, :, idxs].astype(np.float64)  # (R,T,k)
        arr = arr.mean(axis=2)                                            # (R,T)
        xrep_std_end = float(arr[:, -1].std())
        drift = float(arr[:, -1].mean() - arr[:, 0].mean())
        log_init = float(np.log1p(abs(arr[:, 0].mean())))
        return xrep_std_end, drift, log_init

    names_kw = list(_ESS_KEYWORD_COLS)
    names_pipe = ["graph_degree_log", "cpair_member", "lens_composition_count",
                  "granger_outdegree", "n_reactions_catalyzed", "currency_coupling",
                  "traj_xrep_std_end_log", "traj_drift", "traj_log_init_abundance"]

    Xkw, Xpipe, y, loci = [], [], [], []
    for key in sorted(locus_species):
        tag = f"JCVISYN3A_{key}"
        lab = breuer_labels.get(key)
        if lab not in {"Essential", "Quasiessential", "Nonessential"}:
            continue
        # keyword features from gene table
        prod, gname, lbp = gene_meta.get(key, ("", "", 0))
        kw = _essentiality_keyword_features(prod, gname, lbp)
        Xkw.append([kw[c] for c in names_kw])

        sp_idxs = [i for _, i in locus_species[key]]
        prot_idxs = [i for pre, i in locus_species[key] if pre in {"P", "PM"}]
        gdeg = float(np.log1p(deg[sp_idxs].sum())) if deg is not None else 0.0
        cpc  = float(cpair_member[sp_idxs].sum())
        cmp_ = float(comp_count[sp_idxs].sum())
        gro  = float(granger_out[sp_idxs].sum())
        nrxn = float(len(enzyme_rxn.get(key, [])))
        ccpl = float(sum(1 for rid in enzyme_rxn.get(key, []) if rid in currency_rxn))
        xstd, drift, linit = _traj_stats(prot_idxs or sp_idxs)
        Xpipe.append([gdeg, cpc, cmp_, gro, nrxn, ccpl,
                      float(np.log1p(abs(xstd))), drift, linit])

        y.append(1 if lab in {"Essential", "Quasiessential"} else 0)
        loci.append(tag)

    Xkw = np.array(Xkw, dtype=np.float32)
    Xpipe = np.array(Xpipe, dtype=np.float32)
    Xall = np.concatenate([Xkw, Xpipe], axis=1) if len(Xkw) else Xkw
    return {
        "X_kw": Xkw, "X_pipe": Xpipe, "X_all": Xall,
        "y": np.array(y, dtype=np.int8), "loci": loci,
        "names_kw": names_kw, "names_pipe": names_pipe,
        "names_all": names_kw + names_pipe,
    }


def essentiality_classifier_cv(X, y, n_folds=ESS_N_FOLDS, seed=SEED, feature_names=None):
    """v15.9: stratified k-fold gradient-boosted classifier → MCC + confusion +
    out-of-fold predictions + feature importances.  Prefers xgboost, falls back
    to sklearn HistGradientBoostingClassifier (always available on Colab)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import matthews_corrcoef
    if len(X) < n_folds * 2 or len(np.unique(y)) < 2:
        return None

    def _make_clf(pos_weight):
        try:
            import xgboost as xgb
            return ("xgboost", xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                scale_pos_weight=pos_weight, objective="binary:logistic",
                tree_method="hist", eval_metric="logloss", verbosity=0,
                random_state=seed))
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return ("sklearn-hgb", HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=400,
                l2_regularization=1.0, random_state=seed))

    pos_weight = float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    importances = np.zeros(X.shape[1], dtype=np.float64)
    backend = None
    for tr, va in skf.split(X, y):
        backend, clf = _make_clf(pos_weight)
        if backend == "sklearn-hgb":
            # emulate class weighting via sample_weight
            sw = np.where(y[tr] == 1, pos_weight, 1.0)
            clf.fit(X[tr], y[tr], sample_weight=sw)
        else:
            clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        fi = getattr(clf, "feature_importances_", None)
        if fi is not None and len(fi) == X.shape[1]:
            importances += np.asarray(fi, dtype=np.float64) / n_folds
    pred = (oof > 0.5).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0
    top_feats = []
    if feature_names is not None and importances.sum() > 0:
        order = np.argsort(-importances)[:10]
        top_feats = [(feature_names[i], float(importances[i])) for i in order]
    return {"mcc": mcc, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "oof": oof, "pred": pred, "backend": backend, "top_feats": top_feats,
            "n": len(y), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}


def run_essentiality_mcc(species_active, gene_products, gene_meta, breuer_labels,
                          edge_index=None, ruleset=None, patterns=None, sbml=None,
                          kinetics=None, raw_counts_active=None, train_idx=None):
    """v15.9: 3-way essentiality MCC comparison (keyword / pipeline / combined)
    + false-negative recovery inspection.  Prints the report, returns the dict."""
    print()
    print("#" * 72)
    print("#  STATIC ESSENTIALITY CLASSIFIER  -  keyword vs pipeline features")
    print("#  (the LGNN dynamic KO sweep tops out at MCC ~0.06; essentiality is")
    print("#   a STATIC gene property, so we test static features directly)")
    print("#" * 72)
    fm = build_essentiality_feature_matrix(
        species_active, gene_products, gene_meta, breuer_labels,
        edge_index=edge_index, ruleset=ruleset, patterns=patterns, sbml=sbml,
        kinetics=kinetics, raw_counts_active=raw_counts_active, train_idx=train_idx)
    if fm is None or len(fm["y"]) < 20:
        print("  insufficient labelled genes — skipped")
        print("#" * 72)
        return None
    print(f"  labelled genes: {fm['n_pos'] if False else int(fm['y'].sum())} essential "
          f"+ {int((fm['y']==0).sum())} nonessential  (Quasi counted as essential)")
    print(f"  features: {fm['X_kw'].shape[1]} keyword + {fm['X_pipe'].shape[1]} pipeline "
          f"= {fm['X_all'].shape[1]} combined")

    results = {}
    for label, X, names in [
        ("keyword-only",  fm["X_kw"],   fm["names_kw"]),
        ("pipeline-only", fm["X_pipe"], fm["names_pipe"]),
        ("combined",      fm["X_all"],  fm["names_all"]),
    ]:
        r = essentiality_classifier_cv(X, fm["y"], feature_names=names)
        results[label] = r
        if r is None:
            print(f"\n  {label}: skipped (degenerate)")
            continue
        print(f"\n  {label}  [{r['backend']}]:")
        print(f"    MCC = {r['mcc']:+.4f}   (TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']})")
        if r["top_feats"]:
            tops = ", ".join(f"{n}={v:.2f}" for n, v in r["top_feats"][:5])
            print(f"    top features: {tops}")

    # ── false-negative recovery: which genes does keyword MISS that pipeline catches? ──
    rk, rc = results.get("keyword-only"), results.get("combined")
    if rk is not None and rc is not None:
        y = fm["y"]; loci = fm["loci"]
        kw_fn  = (rk["pred"] == 0) & (y == 1)        # essential, keyword missed
        recovered = kw_fn & (rc["pred"] == 1)        # combined now catches
        lost      = (rk["pred"] == 1) & (y == 1) & (rc["pred"] == 0)  # combined newly misses
        print(f"\n  Keyword false-negatives (essential genes keyword missed): {int(kw_fn.sum())}")
        print(f"    recovered by adding pipeline features: {int(recovered.sum())}")
        print(f"    newly lost (regression):               {int(lost.sum())}")
        rec_loci = [loci[i] for i in np.where(recovered)[0]][:12]
        if rec_loci:
            print(f"    recovered genes: {rec_loci}")
        d_mcc = rc["mcc"] - rk["mcc"]
        print(f"\n  Δ MCC (combined − keyword) = {d_mcc:+.4f}  "
              f"→ {'pipeline features ADD signal' if d_mcc > 0.02 else 'no lift — essentiality gap is biological, not feature-extractable' if d_mcc < 0.0 else 'marginal'}")
    print("#" * 72)
    return {"feature_matrix": fm, "results": results}


@torch.no_grad()
def synthetic_lethality_screen(model, test_X, species_active, breuer_labels,
                                edge_index=None, max_pairs=SL_MAX_PAIRS,
                                horizon=SL_HORIZON, batch_size=KO_BATCH_SIZE):
    """v15.9: double-knockout super-additivity screen.

    A gene pair (A, B) is synthetic-lethal if KO(A) and KO(B) are each
    individually tolerable but KO(A,B) is not.  Score = impact(A,B) −
    impact(A) − impact(B); large positive excess = candidate SL interaction.

    Candidate selection (compute-bounded):
      1. rank single-KO impacts; keep the genes BELOW the median (individually
         viable — the only ones that can be synthetic-lethal),
      2. enumerate pairs among them that share a graph edge (interaction is
         only expected between connected genes), cap at max_pairs,
      3. fall back to random viable pairs if too few connected ones.
    """
    model.eval()
    S = test_X.shape[2]
    seed = test_X[0, 0]
    device_ = seed.device
    # locus -> species indices
    gene_cols = {}
    for i, name in enumerate(species_active):
        pre, loc = parse_species(name)
        if pre in {"P", "R", "RP", "G"}:
            gene_cols.setdefault(_locus_key(loc), []).append(i)
    loci = sorted(gene_cols)
    if len(loci) < 4:
        return None

    no_ko = torch.zeros(1, S, dtype=torch.bool, device=device_)
    baseline = _ko_rollout(model, seed.unsqueeze(0), no_ko, horizon).squeeze(0)

    # single-KO impacts (batched)
    single = {}
    for i in range(0, len(loci), batch_size):
        batch = loci[i:i + batch_size]
        km = torch.zeros(len(batch), S, dtype=torch.bool, device=device_)
        for b, loc in enumerate(batch):
            km[b, gene_cols[loc]] = True
        st = seed.unsqueeze(0).expand(len(batch), -1).clone()
        st[km] = CLAMP_LO
        tr = _ko_rollout(model, st, km, horizon)
        for b, loc in enumerate(batch):
            single[loc] = float(((baseline - tr[b]) ** 2).mean())

    # viable genes = single-KO impact below median
    med = float(np.median(list(single.values())))
    viable = [loc for loc in loci if single[loc] <= med]

    # candidate pairs: connected in the graph, among viable genes
    pairs = []
    if edge_index is not None and len(viable) > 1:
        ea = edge_index.detach().cpu().numpy()
        sp_to_locus = {}
        for loc in viable:
            for sp in gene_cols[loc]:
                sp_to_locus[sp] = loc
        seen = set()
        for u, v in zip(ea[0].tolist(), ea[1].tolist()):
            lu, lv = sp_to_locus.get(u), sp_to_locus.get(v)
            if lu and lv and lu != lv:
                key = (lu, lv) if lu < lv else (lv, lu)
                if key not in seen:
                    seen.add(key); pairs.append(key)
            if len(pairs) >= max_pairs:
                break
    # fallback: random viable pairs
    if len(pairs) < min(50, max_pairs) and len(viable) > 1:
        rng = np.random.RandomState(SEED)
        while len(pairs) < min(max_pairs, 200):
            a, b = rng.choice(len(viable), 2, replace=False)
            key = tuple(sorted((viable[a], viable[b])))
            if key not in pairs:
                pairs.append(key)
    pairs = pairs[:max_pairs]
    if not pairs:
        return None

    # double-KO impacts (batched)
    sl = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        km = torch.zeros(len(batch), S, dtype=torch.bool, device=device_)
        for b, (la, lb) in enumerate(batch):
            km[b, gene_cols[la]] = True
            km[b, gene_cols[lb]] = True
        st = seed.unsqueeze(0).expand(len(batch), -1).clone()
        st[km] = CLAMP_LO
        tr = _ko_rollout(model, st, km, horizon)
        for b, (la, lb) in enumerate(batch):
            dbl = float(((baseline - tr[b]) ** 2).mean())
            excess = dbl - single[la] - single[lb]
            sl.append({"a": la, "b": lb, "double": dbl,
                       "single_a": single[la], "single_b": single[lb],
                       "excess": excess})
    sl.sort(key=lambda d: -d["excess"])
    return {"pairs_tested": len(pairs), "n_viable": len(viable),
            "median_single": med, "ranking": sl,
            "breuer_labels": breuer_labels}


def print_synthetic_lethality_report(sl, top=15):
    """v15.9: print the synthetic-lethality screen."""
    print()
    print("#" * 72)
    print("#  SYNTHETIC LETHALITY SCREEN  -  double-KO super-additivity")
    print("#  excess = impact(A,B) − impact(A) − impact(B);  > 0 ⇒ epistatic/SL")
    print("#" * 72)
    if sl is None:
        print("  (no candidate pairs — skipped)")
        print("#" * 72)
        return
    print(f"  viable genes (single-KO impact ≤ median {sl['median_single']:.2e}): {sl['n_viable']}")
    print(f"  candidate pairs tested: {sl['pairs_tested']}")
    exc = [d["excess"] for d in sl["ranking"]]
    if exc:
        pos = sum(1 for e in exc if e > 0)
        print(f"  pairs with positive excess (super-additive): {pos}/{len(exc)} "
              f"({100*pos/len(exc):.1f}%)")
        print(f"  excess range: [{min(exc):.2e}, {max(exc):.2e}]")
    bl = sl.get("breuer_labels", {})
    _abbr = {"Essential": "E", "Quasiessential": "Q", "Nonessential": "N"}
    print(f"\n  Top {top} synthetic-lethal candidates (highest excess):")
    print(f"  (label key: E=Essential Q=Quasiessential N=Nonessential ?=unlabelled)")
    print(f"    {'gene A':<16s} {'gene B':<16s} {'excess':>11s}  {'A':>2s} {'B':>2s}")
    for d in sl["ranking"][:top]:
        la_lab = _abbr.get(bl.get(d["a"]), "?")
        lb_lab = _abbr.get(bl.get(d["b"]), "?")
        print(f"    JCVISYN3A_{d['a']:<6s} JCVISYN3A_{d['b']:<6s} "
              f"{d['excess']:>11.3e}  {la_lab:>2s} {lb_lab:>2s}")
    # Highlight the biologically interesting case: both individually nonessential
    nn_pairs = [d for d in sl["ranking"]
                if bl.get(d["a"]) == "Nonessential" and bl.get(d["b"]) == "Nonessential"
                and d["excess"] > 0]
    if nn_pairs:
        print(f"\n  Both-nonessential SL candidates (the classic SL signature): "
              f"{len(nn_pairs)}")
        for d in nn_pairs[:5]:
            print(f"    JCVISYN3A_{d['a']} + JCVISYN3A_{d['b']}  excess={d['excess']:.3e}")
    print("#" * 72)


# ── main ──────────────────────────────────────────────────────────────────────

def run_reframes_only():
    """v15.8.2: lean path — load data + knowledge phase + three reframe tests.

    Skips model construction, training, eval rollout, and all v15.2 diagnostics.
    Use to verify R1/R2/R3 in ~10–15 min instead of 80–110 min for the full
    pipeline.  Set REFRAMES_ONLY=True (config flag) or REFRAMES_ONLY=1 (env var).
    """
    print(f"[device] {device}")
    print()
    print("=" * 72)
    print("  v15.8.2 — REFRAMES-ONLY fast path (no training, no model build)")
    print("=" * 72)

    raw_counts, species_names = load_data(skip_startup=True)
    n_traj, T, S_full = raw_counts.shape
    print(f"[data] {raw_counts.shape}  (traj, time, species)  {T} steps")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_traj)
    train_idx, test_idx = perm[:N_TRAIN_TRAJ], perm[N_TRAIN_TRAJ:]

    print()
    print("[knowledge] parsing input files for ruleset construction ...")
    sbml         = parse_sbml(SBML_PATH)
    kinetics     = parse_kinetics(KINETICS_PATH)
    initial      = parse_initial_concentrations(INITIAL_CONC_PATH)
    complexes    = parse_complex_formation(COMPLEXES_PATH)
    prot_metab   = parse_protein_metabolites(PROTEIN_METABOLITES_PATH)
    gibbs        = parse_gibbs(GIBBS_PATH)
    largesubunit = parse_largesubunit(LARGESUBUNIT_PATH)
    known = KnownRules(sbml=sbml, kinetics=kinetics,
                       initial=initial, complexes=complexes,
                       protein_metabolites=prot_metab,
                       gibbs=gibbs, largesubunit=largesubunit)

    raw = signed_log(raw_counts)
    dtr = raw[train_idx]
    lo   = np.percentile(dtr, 0.5,  axis=(0, 1))
    hi   = np.percentile(dtr, 99.5, axis=(0, 1))
    span = hi - lo
    active = span > 1e-6
    print(f"[data] active species: {int(active.sum())} / {S_full}")
    raw  = raw[:, :, active]
    lo, span = lo[active], span[active]
    species_active = [species_names[i] for i in range(S_full) if active[i]]
    raw_counts_active = raw_counts[:, :, active]
    raw  = np.clip((raw - lo) / span, CLAMP_LO, CLAMP_HI).astype(np.float32)
    X       = torch.from_numpy(raw)
    test_X  = X[test_idx].to(device)
    print(f"[data] test {tuple(test_X.shape)}")

    print()
    lts = LENS_TIME_STRIDE if TIME_STRIDE == 1 else 1
    print(f"[knowledge] running multi-lens pattern discovery "
          f"(subsampling by {lts}; needed for conservation-pair ruleset) ...")
    train_X = X[train_idx].to(device)
    patterns = DiscoveredPatterns.from_trajectories(
        train_X[:, ::lts].cpu(), test_X[:, ::lts].cpu(),
        raw_counts_active[train_idx][:, ::lts],
        raw_counts_active[test_idx][:, ::lts],
        species_active,
        lo=lo, span=span)
    ruleset, _ = build_enforcement(known, patterns, species_active,
                                    lo=lo, span=span)
    ruleset = ruleset.to(device)
    print(f"[ruleset] {ruleset.summary()}")

    # Stochastic ceiling for the verdict-block reference
    print()
    try:
        ceiling = stochastic_ceiling_diagnostic(
            raw_counts_active, train_idx, top_k_var=VAR_R2_TOP_K)
        print_ceiling_report(ceiling)
        _ceil = ceiling["r2_ceil_median_top_k"] if ceiling is not None else None
    except Exception as e:
        print(f"  stochastic ceiling: skipped ({e})")
        _ceil = None

    # The three reframes
    r3_out = r2_out = r1_out = None
    try:
        r3_out = reframe_lowdim_manifold(raw_counts_active, train_idx)
    except Exception as e:
        print(f"  reframe 3 (low-dim): skipped ({e})")
    try:
        r2_out = reframe_constraints_only(test_X, ruleset)
    except Exception as e:
        print(f"  reframe 2 (constraints): skipped ({e})")
    try:
        r1_out = reframe_binary_algorithm(raw_counts_active, train_idx, species_active)
    except Exception as e:
        print(f"  reframe 1 (algorithm): skipped ({e})")
    print_reframe_report(r3_out, r2_out, r1_out,
                          current_honest_r2=None, ceiling=_ceil)
    print()
    print("=" * 72)
    print("  REFRAMES-ONLY RUN COMPLETE — no model trained, no checkpoint saved")
    print("=" * 72)


def main():
    print(f"[device] {device}")
    print()
    print("=" * 72)
    print("  v13 — v12 + speedup pass (hidden 64→32, BATCH 16→32, optional torch.compile)")
    print("=" * 72)

    raw_counts, species_names = load_data(skip_startup=True)
    n_traj, T, S_full = raw_counts.shape
    print(f"[data] {raw_counts.shape}  (traj, time, species)  {T} steps "
          f"(startup dropped)")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_traj)
    train_idx, test_idx = perm[:N_TRAIN_TRAJ], perm[N_TRAIN_TRAJ:]

    # ── parse every input file ────────────────────────────────────────────
    print()
    print("[knowledge] parsing input files ...")
    sbml         = parse_sbml(SBML_PATH)
    kinetics     = parse_kinetics(KINETICS_PATH)
    initial      = parse_initial_concentrations(INITIAL_CONC_PATH)
    complexes    = parse_complex_formation(COMPLEXES_PATH)
    # v11 additional sources (all graceful-skip if file missing):
    prot_metab   = parse_protein_metabolites(PROTEIN_METABOLITES_PATH)
    gibbs        = parse_gibbs(GIBBS_PATH)
    largesubunit = parse_largesubunit(LARGESUBUNIT_PATH)
    known = KnownRules(sbml=sbml, kinetics=kinetics,
                       initial=initial, complexes=complexes,
                       protein_metabolites=prot_metab,
                       gibbs=gibbs, largesubunit=largesubunit)

    # ── normalise trajectories for the model ─────────────────────────────
    raw = signed_log(raw_counts)
    dtr = raw[train_idx]
    lo   = np.percentile(dtr, 0.5,  axis=(0, 1))
    hi   = np.percentile(dtr, 99.5, axis=(0, 1))
    span = hi - lo
    active = span > 1e-6
    print(f"[data] active species: {int(active.sum())} / {S_full}")
    raw  = raw[:, :, active]
    lo, span = lo[active], span[active]
    species_active = [species_names[i] for i in range(S_full) if active[i]]
    raw_counts_active = raw_counts[:, :, active]
    raw  = np.clip((raw - lo) / span, CLAMP_LO, CLAMP_HI).astype(np.float32)
    S    = raw.shape[2]

    # v15.7 Move D: report what the active-species filter + subsampling discard
    if USE_DATA_RICHNESS_REPORT:
        try:
            data_richness_report(raw_counts, species_names, active, train_idx,
                                 time_stride=TIME_STRIDE)
        except Exception as e:
            print(f"[data-richness] skipped ({e})")

    gene_type_map = load_gene_types(GENE_TABLE_PATH)
    species_type_ids, locus_list = build_gene_index(species_active, gene_type_map)

    X       = torch.from_numpy(raw)
    train_X = X[train_idx].to(device)
    test_X  = X[test_idx].to(device)
    print(f"[data] train {tuple(train_X.shape)}  test {tuple(test_X.shape)}")

    persist_mse = float(F.mse_loss(test_X[:, :-1], test_X[:, 1:]))
    persist_r2  = r2(test_X[:, :-1], test_X[:, 1:])
    print(f"[diag] persistence: MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")

    # ── DiscoveredPatterns (multi-lens) ──────────────────────────────────
    # v10: subsample to LENS_TIME_STRIDE for memory.  At full resolution the
    # pairwise correlation matrix and the SVD conservation lens would
    # materialise tensors of (40*7200, 5933) ≈ 6.5 GB — too tight.  Subsampling
    # by 60 brings it back to current Tier-A scale; statistical patterns are
    # preserved.
    print()
    lts = LENS_TIME_STRIDE if TIME_STRIDE == 1 else 1
    print(f"[knowledge] running multi-lens pattern discovery "
          f"(subsampling by {lts} for memory) ...")
    patterns = DiscoveredPatterns.from_trajectories(
        train_X[:, ::lts].cpu(), test_X[:, ::lts].cpu(),
        raw_counts_active[train_idx][:, ::lts],
        raw_counts_active[test_idx][:, ::lts],
        species_active,
        lo=lo, span=span)

    # ── sort into Tier 1 (enforced) + Tier 2 (reported) ──────────────────
    ruleset, hyp = build_enforcement(known, patterns, species_active,
                                     lo=lo, span=span)
    ruleset = ruleset.to(device)
    # v13.8: compile Tier-2 items into soft-loss tensors for training
    hyp.build_tensors(X[train_idx], device=device)

    # ── cross-validate KnownRules vs trajectory ──────────────────────────
    cross_report = cross_validate_known(known, raw_counts_active, species_active)

    # ── PhD summary ──────────────────────────────────────────────────────
    phd_summary(known, patterns, ruleset, hyp, cross_report, species_active)

    # ── model + training ─────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  TRAINING PHASE  (v10 Tier C: full-res LGNN + PINN + stochastic + richer graph)")
    print("=" * 72)
    edge_index, edge_weight = build_full_graph(
        sbml, kinetics, complexes, species_active,
        protein_metabolites=prot_metab, largesubunit=largesubunit)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    sbml_mask, sbml_indices, stoich_matrix = build_stoich_matrix(sbml, species_active)
    pinn_active = USE_PINN_HEAD and sbml_mask is not None
    if pinn_active:
        sbml_mask     = sbml_mask.to(device)
        sbml_indices  = sbml_indices.to(device)
        stoich_matrix = stoich_matrix.to(device)
    # v13.9: build MetabolismCore tensors (bi-bi rate law for ~160 SBML reactions).
    # When this is non-None, DynamicsModel disables the PINN head and uses the
    # bi-bi formula for covered species.  Disabled gracefully if kinetics sparse.
    metab_tensors = None
    if USE_METABOLISM_CORE:
        metab_tensors = build_metabolism_tensors(sbml, kinetics, species_active,
                                                  gibbs=gibbs)
    # v13.9: build VolumeCore from membrane-lipid total at t=0
    volume_core = None
    if USE_VOLUME_CORE and metab_tensors is not None:
        volume_core = build_volume_core(species_active,
                                         raw_counts_active[train_idx, 0])
    # v14.3: build CentralDogmaCore tensors with per-gene rate calibration
    # from initial_concentrations.xlsx + ribosome counts at t=0
    cd_tensors = None
    if USE_CENTRAL_DOGMA:
        gene_product_map = load_gene_products(GENE_TABLE_PATH)
        cd_tensors = build_central_dogma_tensors(
            species_active,
            initial=initial,
            raw_counts_t0=raw_counts_active[train_idx, 0],
            product_map=gene_product_map,
        )
    # v14 day 5: empirical per-species log σ for the calibration anchor
    target_log_sigma = None
    if USE_SIGMA_ANCHOR:
        train_std = train_X.float().std(dim=(0, 1)).clamp(min=1e-3)
        target_log_sigma = torch.log(train_std).clamp(-6.0, 2.0)
        print(f"[sigma_anchor] empirical log σ: "
              f"median {float(target_log_sigma.median()):+.2f}, "
              f"range [{float(target_log_sigma.min()):+.2f}, "
              f"{float(target_log_sigma.max()):+.2f}]")
    # v14.2: calibrate ATP_MAINTENANCE_RATE from training data instead of
    # using the literature NGAM value.  The literature 4e5/s assumes a
    # specific biomass + growth rate; upstream may operate at a different
    # net rate.  Use median |ΔATP+ΔADP+ΔAMP|/Δt across training as the
    # actual scale the simulator runs at.  Per-second from raw_counts_active
    # (1-second resolution), then scaled to per-step (TIME_STRIDE seconds).
    global ATP_MAINTENANCE_RATE
    aden_names = (ATP_SPECIES_NAME, "M_adp_c", "M_amp_c")
    aden_idx_act = [species_active.index(n) for n in aden_names
                    if n in species_active]
    if len(aden_idx_act) == 3:
        train_aden = raw_counts_active[train_idx][:, :, aden_idx_act].sum(axis=-1)  # (n_train, T)
        pool_deltas = np.abs(np.diff(train_aden, axis=1))                            # (n_train, T-1)
        median_abs_drate = float(np.median(pool_deltas))                             # per stride-step
        # Convert from per-step (TIME_STRIDE seconds) to per-second
        median_per_sec = median_abs_drate / max(1.0, float(TIME_STRIDE))
        # Use the median as the floor, but never lower than 1e4/s (sanity floor)
        calibrated_rate = max(median_per_sec, 1e4)
        print(f"[atp_calibrate] data-derived adenylate-pool |dpool|/dt: "
              f"median = {median_per_sec:.3e}/s (was literature {ATP_MAINTENANCE_RATE:.3e}/s); "
              f"replacing ATP_MAINTENANCE_RATE → {calibrated_rate:.3e}/s")
        ATP_MAINTENANCE_RATE = calibrated_rate
    else:
        print(f"[atp_calibrate] adenylate species not all in trajectory — "
              f"keeping literature default {ATP_MAINTENANCE_RATE:.3e}/s")
    # v13.9: build AssemblyCore tensors (complex_formation.xlsx mass-action +
    # 50S ribosome assembly chain from LargeSubunit.xlsx)
    asm_tensors = None
    if USE_ASSEMBLY_CORE:
        asm_tensors = build_assembly_tensors(complexes, species_active,
                                              lsu_chain=largesubunit)
    model = DynamicsModel(
        S=S, hidden=LGNN_HIDDEN, n_layers=LGNN_N_LAYERS,
        species_type_ids=species_type_ids,
        edge_index=edge_index, edge_weight=edge_weight,
        cfc_tau_min=LGNN_CFC_TAU_MIN, n_type_embed=LGNN_N_TYPE_EMBED,
        use_pinn=pinn_active, sbml_mask=sbml_mask, sbml_indices=sbml_indices,
        stoich_matrix=stoich_matrix, lo_norm=lo, span_norm=span,
        pinn_rate_clip=PINN_RATE_CLIP, use_stochastic=USE_STOCHASTIC_HEAD,
        metab_tensors=metab_tensors, metab_dt=float(TIME_STRIDE),
        volume_core=volume_core, cd_tensors=cd_tensors,
        asm_tensors=asm_tensors,
        target_log_sigma=target_log_sigma,
        use_temporal_context=USE_TEMPORAL_CONTEXT,
        t_ctx_window=T_CTX_WINDOW, t_ctx_hidden=T_CTX_HIDDEN,
        t_ctx_heads=T_CTX_HEADS, t_ctx_layers=T_CTX_LAYERS,
        t_ctx_ff=T_CTX_FF,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    head_tags = []
    if model.metab_core is not None:
        head_tags.append(f"MetabolismCore({model.metab_core.R})")
    if model.cd_core is not None:
        head_tags.append(f"CentralDogma({model.cd_core.n_genes})")
    if model.asm_core is not None:
        head_tags.append(f"Assembly({model.asm_core.R})")
    if model.volume_core is not None:
        head_tags.append(f"VolumeCore({int(model.volume_core.lipid_indices.numel())})")
    if model.use_pinn:        head_tags.append("PINN")
    if USE_STOCHASTIC_HEAD:   head_tags.append("stochastic")
    if model.use_temporal_context:
        head_tags.append(f"TemporalCtx(W={T_CTX_WINDOW},L={T_CTX_LAYERS})")
    print(f"[model] LGNN+{('+'.join(head_tags)) if head_tags else 'plain'}: {n_params:.2f}M parameters, "
          f"{edge_index.shape[1]:,} graph edges")

    # v13.6: optional resume from a previous run's checkpoint
    loaded_from_ckpt = False
    # Prefer the full end-of-run save; fall back to the rolling mid-training save (v13.7)
    ckpt_path = f"{SAVE_DIR}/cell_emulator_v13.pt"
    if not (RESUME_FROM_CHECKPOINT and os.path.exists(ckpt_path)):
        latest = f"{SAVE_DIR}/cell_emulator_v13_latest.pt"
        if RESUME_FROM_CHECKPOINT and os.path.exists(latest):
            ckpt_path = latest
            print(f"[checkpoint] no final checkpoint; will try latest mid-training save {latest}")
    if RESUME_FROM_CHECKPOINT:
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
                cfg = ckpt.get("config", {})
                # Compatibility gate: only resume if the load-bearing dims match.
                # (edge_index and sbml_mask buffers may differ from data changes —
                # those don't matter, we skip them during load.)
                if (cfg.get("S") == S and cfg.get("hidden") == LGNN_HIDDEN
                        and cfg.get("n_layers") == LGNN_N_LAYERS):
                    state = ckpt["model"]
                    # Strip the torch.compile wrapper prefix if the checkpoint
                    # was saved from a compiled model
                    state = {k[10:] if k.startswith("_orig_mod.") else k: v
                             for k, v in state.items()}
                    # Skip buffers — graph and SBML masks come from current setup
                    buf_names = set(dict(model.named_buffers()).keys())
                    state = {k: v for k, v in state.items() if k not in buf_names}
                    result = model.load_state_dict(state, strict=False)
                    if result.unexpected_keys:
                        print(f"[checkpoint] {len(result.unexpected_keys)} unexpected keys ignored")
                    if result.missing_keys:
                        # Only complain about parameter (learnable) misses, not buffers
                        param_names = set(dict(model.named_parameters()).keys())
                        missing_params = [k for k in result.missing_keys if k in param_names]
                        if missing_params:
                            print(f"[checkpoint] WARNING: {len(missing_params)} params missing "
                                  "from ckpt (will use random init for them)")
                    loaded_from_ckpt = True
                    print(f"[checkpoint] resumed from {ckpt_path}")
                else:
                    print(f"[checkpoint] {ckpt_path} architecture mismatch "
                          f"(want S={S}, hidden={LGNN_HIDDEN}, layers={LGNN_N_LAYERS}; "
                          f"got S={cfg.get('S')}, hidden={cfg.get('hidden')}, "
                          f"layers={cfg.get('n_layers')}) — starting fresh")
            except Exception as e:
                print(f"[checkpoint] load failed ({e}) — starting fresh")
        else:
            print(f"[checkpoint] {ckpt_path} not found — starting fresh")

    # v13: optional torch.compile for an extra ~1.5x speedup
    if USE_TORCH_COMPILE and device == "cuda":
        try:
            # dynamic=True handles K varying per training step (curriculum) without
            # forcing recompile each iteration
            model = torch.compile(model, dynamic=True)
            print("[model] torch.compile enabled (dynamic shapes) - first training step "
                  "will be slow (graph capture, ~30-60s)")
        except Exception as e:
            print(f"[model] torch.compile failed ({e}) - running in eager mode")

    # v15.6: pre-load Breuer + build helpers so train_model can use the
    # KO-consistency loss (we already use breuer_labels in eval for KO MCC).
    breuer_labels = load_breuer_essentiality(BREUER_PATH)
    if USE_BREUER_LOSS and breuer_labels:
        _gene_species, _e_loci, _ne_loci = build_breuer_helpers(species_active, breuer_labels)
        breuer_data = {"gene_species": _gene_species,
                       "e_loci": _e_loci, "ne_loci": _ne_loci}
        print(f"[breuer] training signal: {len(_e_loci)} essential + "
              f"{len(_ne_loci)} nonessential genes mapped to species")
    else:
        breuer_data = None

    # v15.7 Move A: build multi-task groups + per-class weights once
    mt_groups = build_multitask_groups(species_active) if USE_MULTITASK_LOSS else None
    pc_weights = build_per_class_weights(train_X) if USE_PER_CLASS_LOSS_W else None
    if mt_groups:
        print(f"[multitask] {len(mt_groups)} derived-quantity groups: "
              f"{', '.join(mt_groups.keys())}")
    if pc_weights is not None:
        print(f"[per_class_w] variance weights: median {float(pc_weights.median()):.2f}, "
              f"range [{float(pc_weights.min()):.2f}, {float(pc_weights.max()):.2f}]")

    hyp_violation_ema = {}
    if loaded_from_ckpt and SKIP_TRAINING_IF_LOADED:
        print("[train] SKIPPED — using loaded checkpoint as-is for evaluation")
    else:
        if loaded_from_ckpt:
            print("[train] continuing training from loaded checkpoint")
        elif USE_MASKED_PRETRAIN:
            # v15.7 Move C1: warm the graph with masked-species pretraining
            # (only on a fresh model — skip when continuing from a checkpoint)
            masked_pretrain(model, train_X)
        hyp_violation_ema = train_model(model, train_X, ruleset,
                                         hyp=hyp, breuer=breuer_data,
                                         multitask_groups=mt_groups,
                                         per_class_w=pc_weights) or {}

    # ── evaluate ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  EVALUATION")
    print("=" * 72)
    s_mse, s_r2 = one_step(model, test_X)
    # v10: batched rollout — all 10 test trajs in one shared time loop, ~10x faster
    # v15.2: also collect per-step fluxes for the TUR + F-W action diagnostics
    _rollout_out = full_rollout_batched(model, test_X, ruleset, collect_fluxes=True)
    if len(_rollout_out) == 3:
        preds_batch, truth_batch, fluxes_batch = _rollout_out
    else:
        preds_batch, truth_batch = _rollout_out
        fluxes_batch = None
    roll_r2 = [r2(preds_batch[k], truth_batch[k]) for k in range(test_X.shape[0])]
    mean_roll = sum(roll_r2) / len(roll_r2)
    # v9: variance-weighted R² (median over top-K high-variance species)
    all_preds = preds_batch.reshape(-1, preds_batch.shape[-1])
    all_true  = truth_batch.reshape(-1, truth_batch.shape[-1])
    var_r2, n_var = variance_weighted_r2(all_preds, all_true, top_k=VAR_R2_TOP_K)
    print()
    print("=" * 72)
    print(f"  persistence 1-step       : MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")
    print(f"  model 1-step (test)      : MSE {s_mse:.5f}  R^2 {s_r2:.3f}"
          f"  {'(beats persistence)' if s_mse < persist_mse else '(still worse)'}")
    print(f"  model full rollout       : R^2 {mean_roll:.3f}  "
          f"(min {min(roll_r2):.3f}  max {max(roll_r2):.3f})  <- mean over all species")
    print(f"  median R² on top-{n_var} variable species : {var_r2:+.3f}  <- HONEST METRIC")
    print(f"  (v6 transformer (60s)    : rollout R^2 ~0.56,  honest unknown)")
    print(f"  (v9 LGNN (60s stride)    : rollout R^2 ~0.64,  honest +0.37)")
    # v14 day 5: trajectory-spread metric.  NOTE (v15.3): this measures
    # cross-trajectory spread of DETERMINISTIC predictions vs truth — it is
    # really a mode-collapse metric, not σ-head calibration.  The σ-head's
    # own calibration is measured separately in step-5 recalibration below.
    if USE_STOCHASTIC_HEAD:   # v15.4: was also gated on USE_SIGMA_ANCHOR (now off)
        sigma_pred  = preds_batch.std(dim=0)          # (T, S) — across trajectories
        sigma_true  = truth_batch.std(dim=0)
        log_ratio = (sigma_pred.clamp(min=1e-6).log10()
                     - sigma_true.clamp(min=1e-6).log10())
        cal_med = float(log_ratio.median())
        cal_iqr = float((log_ratio.quantile(0.75) - log_ratio.quantile(0.25)))
        verdict = ("well-spread"  if abs(cal_med) < 0.3 else
                   "over-diverse" if cal_med > 0 else
                   "mode-collapsed (under-spread)")
        print(f"  trajectory spread (det.) : log10(pred_std/true_std) median {cal_med:+.2f}  "
              f"IQR {cal_iqr:.2f}  ({verdict})")
        print(f"                             ^ this is mode-collapse, NOT σ-head calibration;")
        print(f"                               see step-5 stochastic rollout for the real σ metric")
    print("=" * 72)

    # v15.3 step 0: stochastic ceiling — irreducible R² limit for any deterministic model
    try:
        ceiling = stochastic_ceiling_diagnostic(
            raw_counts_active, train_idx, top_k_var=VAR_R2_TOP_K,
        )
        print_ceiling_report(ceiling, current_rollout_r2=mean_roll,
                             current_honest_r2=var_r2)
    except Exception as e:
        print(f"  stochastic ceiling: skipped ({e})")
    # v14.4: paper validation metrics (Thornburg 2026 §Results)
    try:
        metrics = paper_validation_metrics(
            preds_batch, truth_batch, species_active, lo, span,
            time_stride_s=float(TIME_STRIDE),
        )
        print_paper_metrics(metrics)
    except Exception as e:
        print(f"  paper validation: skipped ({e})")
    print("=" * 72)

    # v15.3 step 5: post-hoc σ recalibration + stochastic rollout for
    # distributional eval.  Per-trajectory R² will typically dip; the
    # σ-spread, W₂, and paper-stats metrics become meaningful.
    preds_sto, truth_sto, fluxes_sto = preds_batch, truth_batch, fluxes_batch
    sigma_alpha = None
    if USE_STOCHASTIC_ROLLOUT_EVAL and USE_STOCHASTIC_HEAD:
        try:
            rc = recalibrate_sigma_posthoc(model, train_X)
            print_recalibration_report(rc)
            if rc is not None:
                sigma_alpha = rc["alpha"]
                sto_out = full_rollout_batched_stochastic(
                    model, test_X, ruleset, sigma_alpha,
                    collect_fluxes=(fluxes_batch is not None),
                )
                if sto_out is not None:
                    if len(sto_out) == 3:
                        preds_sto, truth_sto, fluxes_sto = sto_out
                    else:
                        preds_sto, truth_sto = sto_out
                    # Stochastic R² (typically lower; this is correct)
                    r2_sto = [r2(preds_sto[k], truth_sto[k])
                              for k in range(test_X.shape[0])]
                    mean_sto = sum(r2_sto) / len(r2_sto)
                    # Trajectory-spread metric (the mode-collapse one) on stochastic preds
                    sp_pred_s = preds_sto.std(dim=0)
                    sp_true_s = truth_sto.std(dim=0)
                    cal_sto = float((sp_pred_s.clamp(min=1e-6).log10()
                                     - sp_true_s.clamp(min=1e-6).log10()).median())
                    print()
                    print("=" * 72)
                    print("  STOCHASTIC ROLLOUT EVAL  (v15.3 step 5)")
                    print("=" * 72)
                    print(f"\n  Per-trajectory R² (stochastic) = {mean_sto:.3f}  "
                          f"(deterministic was {mean_roll:.3f})")
                    print(f"    note: lower per-traj R² for stochastic is EXPECTED — "
                          f"injected noise reduces mean-fit but enables population stats")
                    print(f"\n  Trajectory-spread (stochastic) = log10(pred/true) median "
                          f"{cal_sto:+.2f}")
                    if 'cal_med' in dir() and cal_med is not None:
                        delta = cal_sto - cal_med
                        print(f"    deterministic reference  = {cal_med:+.2f}")
                        print(f"    Δ (more positive = closer to truth spread) = {delta:+.2f}")
                    # v15.4 B3: re-run the distributional diagnostics on the
                    # STOCHASTIC rollout — deterministic preds are mode-collapsed,
                    # so W2 / Helmholtz / paper-stats / TUR are only informative here.
                    print("\n  --- distributional diagnostics on STOCHASTIC rollout (v15.4 B3) ---")
                    try:
                        print_paper_metrics(paper_validation_metrics(
                            preds_sto, truth_sto, species_active, lo, span,
                            time_stride_s=float(TIME_STRIDE)))
                    except Exception as _e:
                        print(f"  paper validation (stochastic): skipped ({_e})")
                    try:
                        print_wasserstein_report(sliced_wasserstein_W2(
                            preds_sto, truth_sto, top_k_var=VAR_R2_TOP_K))
                    except Exception as _e:
                        print(f"  W2 (stochastic): skipped ({_e})")
                    try:
                        print_helmholtz_report(helmholtz_curl_diagnostic(
                            preds_sto, truth_sto, species_active))
                    except Exception as _e:
                        print(f"  Helmholtz (stochastic): skipped ({_e})")
                    try:
                        print_kramers_report(kramers_residence_times(
                            preds_sto, truth_sto, species_active, n_bins=5))
                    except Exception as _e:
                        print(f"  Kramers (stochastic): skipped ({_e})")
                    if fluxes_sto is not None:
                        try:
                            print_tur_report(tur_per_reaction_diagnostic(
                                model, fluxes_sto, gibbs_data=(gibbs or {}),
                                time_stride_s=float(TIME_STRIDE), volume_l=SYN3A_VOLUME_L))
                        except Exception as _e:
                            print(f"  TUR (stochastic): skipped ({_e})")
        except Exception as e:
            print(f"  stochastic rollout eval: skipped ({e})")

    # v15.2: TUR per-reaction thermodynamic check (Barato-Seifert 2015)
    try:
        tur = tur_per_reaction_diagnostic(
            model, fluxes_batch,
            gibbs_data=(gibbs or {}),
            time_stride_s=float(TIME_STRIDE),
            volume_l=SYN3A_VOLUME_L,
        )
        print_tur_report(tur)
    except Exception as e:
        print(f"  TUR diagnostic: skipped ({e})")

    # v15.2: counterfactual-robustness — computing vs memorization per species
    try:
        cf = counterfactual_robustness(
            model, test_X[0, 0].unsqueeze(0).to(device),
            species_active, edge_index,
            n_test_species=100, n_steps=60,
        )
        print_counterfactual_report(cf, species_active)
    except Exception as e:
        print(f"  counterfactual-robustness: skipped ({e})")

    # v15.2: Friedlin-Wentzell action under inferred SDE (plausibility score)
    try:
        fw = friedlin_wentzell_action(
            preds_batch, truth_batch, dt=float(TIME_STRIDE),
            top_k_var=VAR_R2_TOP_K, fit_trajs=train_X,
        )
        print_friedlin_wentzell_report(fw)
    except Exception as e:
        print(f"  F-W action: skipped ({e})")

    # v15.2: constraint provenance audit on worst-predicted species
    try:
        _se = ((preds_batch - truth_batch) ** 2).mean(dim=(0, 1))
        worst_sp = torch.argsort(_se, descending=True)[:15].tolist()
        traces = constraint_provenance(
            model, worst_sp, species_active, species_type_ids,
            edge_index, raw_counts_active, train_idx,
        )
        print_provenance_report(traces)
    except Exception as e:
        print(f"  provenance audit: skipped ({e})")

    # v15.2: Kramers residence times in cell-cycle phase bins
    try:
        kr = kramers_residence_times(
            preds_batch, truth_batch, species_active, n_bins=5,
        )
        print_kramers_report(kr)
    except Exception as e:
        print(f"  Kramers residence: skipped ({e})")

    # v15.2: sliced Wasserstein-2 distance per species
    try:
        ws = sliced_wasserstein_W2(
            preds_batch, truth_batch, top_k_var=VAR_R2_TOP_K,
        )
        print_wasserstein_report(ws)
    except Exception as e:
        print(f"  Wasserstein: skipped ({e})")

    # v15.2: Helmholtz curl detection in 2D projection
    try:
        hl = helmholtz_curl_diagnostic(preds_batch, truth_batch, species_active)
        print_helmholtz_report(hl)
    except Exception as e:
        print(f"  Helmholtz curl: skipped ({e})")

    # v15.8: three-reframes empirical tests (cell as algorithm / constraints / low-dim)
    r3_out = r2_out = r1_out = None
    try:
        r3_out = reframe_lowdim_manifold(raw_counts_active, train_idx)
    except Exception as e:
        print(f"  reframe 3 (low-dim): skipped ({e})")
    try:
        r2_out = reframe_constraints_only(test_X, ruleset)
    except Exception as e:
        print(f"  reframe 2 (constraints): skipped ({e})")
    try:
        r1_out = reframe_binary_algorithm(raw_counts_active, train_idx, species_active)
    except Exception as e:
        print(f"  reframe 1 (algorithm): skipped ({e})")
    try:
        _ceil = ceiling["r2_ceil_median_top_k"] if 'ceiling' in dir() and ceiling is not None else None
        print_reframe_report(r3_out, r2_out, r1_out,
                             current_honest_r2=var_r2, ceiling=_ceil)
    except Exception as e:
        print(f"  reframe report: skipped ({e})")

    analyze_gaps(model, test_X, species_active, species_type_ids,
                 ruleset, hyp, sbml,
                 element_balances(sbml, species_active, raw_counts_active),
                 preds_batch=preds_batch, truth_batch=truth_batch)

    # ── v9: knockout sweep vs Breuer 2019 essentiality ───────────────────
    # v15.6: breuer_labels was loaded earlier for the KO-consistency training loss
    ko = knockout_sweep(model, ruleset, test_X, species_active, breuer_labels)
    print_knockout_report(ko, breuer_labels)

    # ── v15.9: static essentiality classifier (keyword + pipeline features) ──
    if USE_ESSENTIALITY_XGB and breuer_labels:
        try:
            gene_products = load_gene_products(GENE_TABLE_PATH)
            gene_meta = {}
            if HAS_PANDAS:
                _gt = pd.read_csv(GENE_TABLE_PATH)
                for _, _r in _gt.iterrows():
                    _tag = str(_r.get("locus_tag", ""))
                    if "_" in _tag:
                        gene_meta[_tag.split("_")[1]] = (
                            str(_r.get("product", "")), str(_r.get("gene_name", "")),
                            _r.get("length_bp", 0))
            run_essentiality_mcc(
                species_active, gene_products, gene_meta, breuer_labels,
                edge_index=edge_index, ruleset=ruleset, patterns=patterns,
                sbml=sbml, kinetics=kinetics,
                raw_counts_active=raw_counts_active, train_idx=train_idx)
        except Exception as e:
            print(f"  essentiality classifier: skipped ({e})")

    # ── v15.9: synthetic lethality screen (double-KO super-additivity) ──
    if USE_SYNTHETIC_LETHALITY:
        try:
            sl = synthetic_lethality_screen(
                model, test_X, species_active, breuer_labels, edge_index=edge_index)
            print_synthetic_lethality_report(sl)
        except Exception as e:
            print(f"  synthetic lethality: skipped ({e})")

    # ── generate + save ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  GENERATION (the 51st trajectory)")
    print("=" * 72)
    gen_norm, _ = full_rollout(model, test_X[0], ruleset)
    full_seq = torch.cat([test_X[0, :1], gen_norm], 0).cpu().numpy()
    sl = full_seq * span + lo
    gen_counts = np.maximum(np.sign(sl) * np.expm1(np.abs(sl)), 0.0)
    print(f"[gen] 51st trajectory {gen_counts.shape}  "
          f"finite={np.isfinite(gen_counts).all()}  "
          f"count range [{gen_counts.min():.0f}, {gen_counts.max():.0f}]")
    np.save(f"{SAVE_DIR}/cell_traj_51_v13.npy", gen_counts)
    torch.save({
        "model": model.state_dict(),
        "lo": lo, "span": span, "active": active,
        "species_active": species_active,
        "species_type_ids": species_type_ids,
        "edge_index": edge_index.cpu(),
        "edge_weight": edge_weight.cpu(),
        "ruleset_mono_up":   ruleset.mono_up.cpu()   if ruleset.mono_up   is not None else None,
        "ruleset_mono_down": ruleset.mono_down.cpu() if ruleset.mono_down is not None else None,
        "ruleset_lo": ruleset.lo_bound.cpu() if ruleset.lo_bound is not None else None,
        "ruleset_hi": ruleset.hi_bound.cpu() if ruleset.hi_bound is not None else None,
        "hypotheses": hyp.items,
        "hyp_violation_ema": hyp_violation_ema,
        "known_rules_summary": known.summary(),
        "knockout_sweep": ko,
        "var_r2_median": var_r2,
        "config": dict(S=S, hidden=LGNN_HIDDEN, n_layers=LGNN_N_LAYERS,
                       cfc_tau_min=LGNN_CFC_TAU_MIN, n_type_embed=LGNN_N_TYPE_EMBED,
                       time_stride=TIME_STRIDE, n_genes=len(locus_list),
                       skip_startup=SKIP_STARTUP_STEPS,
                       use_pinn=model.use_pinn, use_stochastic=USE_STOCHASTIC_HEAD,
                       use_metab_core=model.metab_core is not None,
                       metab_n_reactions=(model.metab_core.R if model.metab_core is not None else 0),
                       use_cd_core=model.cd_core is not None,
                       cd_n_genes=(model.cd_core.n_genes if model.cd_core is not None else 0),
                       use_asm_core=model.asm_core is not None,
                       asm_n_reactions=(model.asm_core.R if model.asm_core is not None else 0),
                       use_volume_core=model.volume_core is not None,
                       architecture="LGNN_v13"),
    }, f"{SAVE_DIR}/cell_emulator_v13.pt")
    print(f"[save] traj  -> {SAVE_DIR}/cell_traj_51_v13.npy")
    print(f"[save] model -> {SAVE_DIR}/cell_emulator_v13.pt")


if __name__ == "__main__":
    if REFRAMES_ONLY:
        run_reframes_only()
    else:
        main()
