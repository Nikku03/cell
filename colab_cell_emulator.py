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
TIME_STRIDE         = 30           # v13.8.2: stride=30 sanity run — 30-sec steps, ~240 timesteps/traj
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
STEPS               = 1500         # v13: was 2000 — fewer steps, BATCH doubled to compensate
K_MAX               = 120          # v13.8.2: 120 steps = 60 min biological at 30s stride (half cell cycle)
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
LGNN_HIDDEN         = 32           # v13: was 64 — half params, ~2x faster forward (set 64 to restore)
LGNN_N_LAYERS       = 3            # NEW v8: number of CfC graph layers
LGNN_CFC_TAU_MIN    = 0.1          # NEW v8: CfC time-constant minimum
LGNN_N_TYPE_EMBED   = 4            # NEW v8: gene-type embed dim
USE_PINN_HEAD       = True         # NEW v9: hardwire mass-balance for SBML species
USE_METABOLISM_CORE = True         # NEW v13.9: replace PINN's neural flux with the bi-bi rate law for ~160 SBML reactions
USE_VOLUME_CORE     = True         # NEW v13.9: dynamic cell volume from membrane-lipid count (vs constant)
USE_CENTRAL_DOGMA   = True         # v14.3: re-enabled with per-gene rate calibration from initial counts
USE_ASSEMBLY_CORE   = False        # v14.1: still disabled — only 2/24 wired on real data, separate issue
USE_KO_AUGMENTATION = True         # NEW v13.9 module 7: random species-zero during training, teaches KO response
KO_AUG_PROB         = 0.3          # probability per batch element of being knockout-perturbed
GIBBS_DG_THRESHOLD_KJ = 10.0       # NEW v14 day 1: ΔG° (kJ/mol) below which a reaction is forced irreversible-forward
USE_ATP_LEDGER      = True         # NEW v14 day 3: soft penalty when net ATP production rate falls below maintenance floor
ATP_SPECIES_NAME    = "M_atp_c"    # which species to track for the energy ledger
ATP_MAINTENANCE_RATE = 4.0e5       # ATP molecules per second per cell — NGAM floor (~5 mmol/g/hr × Syn3A mass)
LAMBDA_ATP          = 0.01         # auxiliary loss weight for ATP-deficit penalty
USE_SIGMA_ANCHOR    = True         # NEW v14 day 5: anchor predicted log σ to empirical std, prevents NLL-shrinkage
LAMBDA_SIGMA_ANCHOR = 0.2          # v14.7: bumped 0.05 → 0.2 to attack mode collapse (model was producing
                                   #        identical trajectories across all test starting states)
LAMBDA_TRAJ_VAR     = 0.05         # v14.7: penalty on pred.std(across batch) vs target σ — explicit
                                   #        anti-mode-collapse signal independent of NLL
SAMPLE_NOISE_SCALE  = 0.5          # v14.7: training-time noise injection in rollout (reparameterized)
                                   #        so model must handle perturbed inputs — kills the deterministic
                                   #        mean trajectory.  0.0 = no noise, 1.0 = full σ sampling.
USE_STOCHASTIC_HEAD = True         # NEW v9: per-species log_sigma + NLL loss
USE_TORCH_COMPILE   = True         # v13.4: was False — enable torch.compile by default (falls back to eager on failure)
RESUME_FROM_CHECKPOINT  = False    # v13.6: load existing cell_emulator_v13.pt if compatible
SKIP_TRAINING_IF_LOADED = False    # v13.6: if RESUME loaded weights, skip training (eval-only run)
CHECKPOINT_EVERY    = 250          # v13.7: write a rolling mid-training checkpoint every N steps (0 to disable)
PINN_RATE_CLIP      = 6.0          # NEW v9: log-space rate clip (prevents expm1 blow-up)
VAR_R2_TOP_K        = 200          # NEW v9: top-K species (by variance) for the honest R²
KO_N_STEPS          = 60           # v13.8.2: 60 steps = 30 min biological at 30s stride
KO_BATCH_SIZE       = 32           # NEW v9: parallel knockouts per batch
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
        return p

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
        ])


# ── enforcement: Tier 1 RuleSet + Tier 2 Hypotheses ──────────────────────────

class RuleSet:
    """Tier 1: validated rules, enforced as hard rollout guardrails."""

    def __init__(self):
        self.mono_up_mask = self.mono_down_mask = None
        self.mono_up = self.mono_down = None
        self.lo_bound = self.hi_bound = None
        self.n_seed = self.n_sbml = self.n_empirical = 0

    def to(self, dev):
        for a in ("mono_up_mask", "mono_down_mask", "mono_up", "mono_down",
                  "lo_bound", "hi_bound"):
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
        return nxt

    def summary(self):
        nu = int(self.mono_up_mask.sum())   if self.mono_up_mask   is not None else 0
        nd = int(self.mono_down_mask.sum()) if self.mono_down_mask is not None else 0
        nb = "yes" if self.lo_bound is not None else "no"
        return (f"Tier 1 RuleSet: {nu} monotone-up + {nd} monotone-down + "
                f"per-species bounds ({nb})  "
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


def build_enforcement(known, patterns, species_names):
    """Sort the validated subset of KnownRules+DiscoveredPatterns into Tier 1.
    Everything else (failed monotone, all conservation/pairwise/etc.) -> Tier 2.

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

    for i in seed_up - val_up:
        print(f"[rules] WARNING: seed monotone '{species_names[i]}' failed validation "
              f"({patterns.up_frac_va[i]*100:.2f}%) - moved to Tier 2")

    print(f"[rules] Tier 1: {len(val_up)} mono-up, {len(val_down)} mono-down, "
          f"{int(ok.sum())}/{S} bounded")
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
CD_TRANSLATION_SCALE   = 0.7           # multiplier on every k_tl_per_gene
                                       # → 0.7 estimated to bring 2.78× → ~2.0× (match upstream)


def build_central_dogma_tensors(species_names, initial=None, raw_counts_t0=None,
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
        # v13.9: MetabolismCore takes priority over PINN for the species
        # it covers — disable PINN entirely when MetabolismCore is wired in.
        self.metab_core = (MetabolismCore(metab_tensors,
                                          atp_idx=metab_tensors.get("atp_idx"))
                           if metab_tensors is not None else None)
        self.metab_dt = float(metab_dt)
        # v13.9: VolumeCore tracks dynamic cell volume; passed to MetabolismCore each step.
        self.volume_core = volume_core
        # v13.9: CentralDogmaCore — first-order tx/tl/decay per gene
        self.cd_core = CentralDogmaCore(cd_tensors) if cd_tensors is not None else None
        # v13.9: AssemblyCore — mass-action complex assembly
        self.asm_core = AssemblyCore(asm_tensors) if asm_tensors is not None else None
        # v9: PINN head (optional) — disabled when MetabolismCore is on
        self.use_pinn = bool(use_pinn and sbml_mask is not None
                             and self.metab_core is None)
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

    def forward(self, x):                     # x: (B, S) normalised
        B, S = x.shape
        te  = self.type_embed(self.stype).unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([x.unsqueeze(-1), te], dim=-1)
        h   = self.in_proj(inp)
        for layer in self.layers:
            h = layer(h, self.edge_index, self.edge_weight)
        h = self.out_norm(h)
        delta  = self.out_head(h).squeeze(-1)
        x_next = x + delta
        # v13.9: MetabolismCore overrides covered species via bi-bi rate law.
        # Round-trip normalised -> count -> bi-bi step -> count -> normalised.
        if self.metab_core is not None:
            x_sl_lin     = x.float() * self.metab_span + self.metab_lo
            x_count      = t_signed_expm1(x_sl_lin)
            v_l = self.volume_core(x_count) if self.volume_core is not None else None
            d_count, fluxes = self.metab_core(x_count, dt=self.metab_dt, volume_l=v_l)
            # v14 day 3: stash ATP rate so the training loop can apply the energy-ledger loss
            if self.metab_core.has_atp:
                self.last_atp_rate = self.metab_core.compute_atp_rate(fluxes, volume_l=v_l)
            x_count_next = (x_count + d_count).clamp(min=0.0)
            x_sl_next    = t_signed_log1p(x_count_next)
            x_norm_metab = (x_sl_next - self.metab_lo) / self.metab_span
            # Keep predictions within the trained normalisation window —
            # otherwise a large bi-bi step can carry state outside the model's
            # input distribution and corrupt the rest of the rollout.
            x_norm_metab = x_norm_metab.clamp(CLAMP_LO, CLAMP_HI).to(x_next.dtype)
            mask = self.metab_core.coverage_mask.unsqueeze(0).expand(B, -1)
            x_next = torch.where(mask, x_norm_metab, x_next)
        # v13.9: CentralDogmaCore overrides mRNA + protein with first-order tx/tl/decay
        if self.cd_core is not None:
            x_sl_lin     = x.float() * self.metab_span + self.metab_lo
            x_count      = t_signed_expm1(x_sl_lin)
            d_count      = self.cd_core(x_count, dt=self.metab_dt)
            x_count_next = (x_count + d_count).clamp(min=0.0)
            x_sl_next    = t_signed_log1p(x_count_next)
            x_norm_cd    = (x_sl_next - self.metab_lo) / self.metab_span
            x_norm_cd    = x_norm_cd.clamp(CLAMP_LO, CLAMP_HI).to(x_next.dtype)
            mask_cd      = self.cd_core.coverage_mask.unsqueeze(0).expand(B, -1)
            x_next       = torch.where(mask_cd, x_norm_cd, x_next)
        # v13.9: AssemblyCore overrides complex + subunit counts via mass-action
        if self.asm_core is not None:
            x_sl_lin     = x.float() * self.metab_span + self.metab_lo
            x_count      = t_signed_expm1(x_sl_lin)
            d_count      = self.asm_core(x_count, dt=self.metab_dt)
            x_count_next = (x_count + d_count).clamp(min=0.0)
            x_sl_next    = t_signed_log1p(x_count_next)
            x_norm_asm   = (x_sl_next - self.metab_lo) / self.metab_span
            x_norm_asm   = x_norm_asm.clamp(CLAMP_LO, CLAMP_HI).to(x_next.dtype)
            mask_asm     = self.asm_core.coverage_mask.unsqueeze(0).expand(B, -1)
            x_next       = torch.where(mask_asm, x_norm_asm, x_next)
        # PINN head overrides SBML species with mass-balanced prediction (disabled when MetabolismCore is on)
        if self.use_pinn:
            x_next_sbml = self.pinn_head(h, x)                              # (B, n_sbml)
            x_pinn_full = torch.zeros_like(x_next)
            x_pinn_full = x_pinn_full.index_copy(1, self.pinn_head.sbml_indices,
                                                  x_next_sbml)
            mask = self.pinn_head.sbml_mask.unsqueeze(0).expand(B, -1)
            x_next = torch.where(mask, x_pinn_full, x_next)
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
    don't briefly hold two copies (12 + 12 GB at full resolution)."""
    assert HAS_PANDAS, "pandas required - install or run on Colab"
    pat = (f"{PARQUET_DIR}/counts_and_fluxes*.parquet" if PARQUET_DIR
           else "/content/drive/MyDrive/**/counts_and_fluxes*.parquet")
    files = sorted(glob.glob(pat, recursive=True),
                   key=lambda p: int(p.rsplit(".", 2)[-2]))
    assert files, "no parquet files - set PARQUET_DIR"
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


# ── training / eval ───────────────────────────────────────────────────────────

def train_model(model, train_X, ruleset, hyp=None):
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
          f"sigma_anchor: {'ON λ=' + str(LAMBDA_SIGMA_ANCHOR) if use_sigma_anchor else 'off'}")
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

        for k in range(K):
            with _autocast():
                out = model(state)
                if use_nll:
                    pred, log_sigma = out
                    true   = train_X[i_d, t0_d + 1 + k]
                    if ko_b_idx is not None:
                        true = true.clone()
                        true[ko_b_idx, ko_sp_idx] = zero_norm[ko_sp_idx]
                    sq_err = (pred - true) ** 2
                    inv_var = torch.exp(-2.0 * log_sigma)
                    loss_k = 0.5 * (inv_var * sq_err + 2.0 * log_sigma).mean()
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
            print(f"  step {step+1:5d}  K={K:4d}  "
                  f"1-step {float(first_loss.detach()):.5f}  "
                  f"rollout {sum(all_loss_means)/len(all_loss_means):.5f}"
                  f"{hyp_str}{atp_str}{sigma_str}", flush=True)

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
    state = traj[0].unsqueeze(0)        # (1, S)
    preds = []
    for _ in range(traj.shape[0] - 1):
        with _eval_autocast():
            p = _model_pred(model(state)).float().clamp(CLAMP_LO, CLAMP_HI)
        p = ruleset.project(state, p)
        preds.append(p)
        state = p
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
    ASSEMBLED_RIBO_KEYS = ("C_ribosome", "ribosome",
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
def full_rollout_batched(model, trajs, ruleset):
    """v10: roll all test trajectories in parallel (one shared time loop).
    7,199 steps × 10 trajs would take ~60 min serially at full resolution;
    batched, it's a single 7,199-step loop = ~6 min.  With v12 BF16 ~3 min.

    trajs: (n, T, S) tensor of seed trajectories.
    Returns (preds, truth) both shaped (n, T-1, S).
    """
    model.eval()
    state = trajs[:, 0]              # (n, S)
    preds = []
    for _ in range(trajs.shape[1] - 1):
        with _eval_autocast():
            p = _model_pred(model(state)).float().clamp(CLAMP_LO, CLAMP_HI)
        p = ruleset.project(state, p)
        preds.append(p)
        state = p
    return torch.stack(preds, dim=1), trajs[:, 1:]   # (n, T-1, S) each


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
    preds = []
    for _ in range(n_steps):
        with _eval_autocast():
            p = _model_pred(model(state)).float().clamp(CLAMP_LO, CLAMP_HI)
        # Permanent knockdown: force every masked species back to floor
        p = torch.where(ko_mask, torch.full_like(p, CLAMP_LO), p)
        preds.append(p)
        state = p
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


# ── main ──────────────────────────────────────────────────────────────────────

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
    ruleset, hyp = build_enforcement(known, patterns, species_active)
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
        cd_tensors = build_central_dogma_tensors(
            species_active,
            initial=initial,
            raw_counts_t0=raw_counts_active[train_idx, 0],
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

    hyp_violation_ema = {}
    if loaded_from_ckpt and SKIP_TRAINING_IF_LOADED:
        print("[train] SKIPPED — using loaded checkpoint as-is for evaluation")
    else:
        if loaded_from_ckpt:
            print("[train] continuing training from loaded checkpoint")
        hyp_violation_ema = train_model(model, train_X, ruleset, hyp=hyp) or {}

    # ── evaluate ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  EVALUATION")
    print("=" * 72)
    s_mse, s_r2 = one_step(model, test_X)
    # v10: batched rollout — all 10 test trajs in one shared time loop, ~10x faster
    preds_batch, truth_batch = full_rollout_batched(model, test_X, ruleset)
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
    # v14 day 5: σ calibration — compare predicted std vs empirical std on test set
    if USE_STOCHASTIC_HEAD and USE_SIGMA_ANCHOR:
        sigma_pred  = preds_batch.std(dim=0)          # (T, S) — across trajectories
        sigma_true  = truth_batch.std(dim=0)
        # Median over species and time of log10(pred/true) — ideal calibration = 0
        log_ratio = (sigma_pred.clamp(min=1e-6).log10()
                     - sigma_true.clamp(min=1e-6).log10())
        cal_med = float(log_ratio.median())
        cal_iqr = float((log_ratio.quantile(0.75) - log_ratio.quantile(0.25)))
        verdict = ("well-calibrated"   if abs(cal_med) < 0.3 else
                   "under-confident"   if cal_med > 0 else
                   "over-confident")
        print(f"  σ calibration            : log10(pred/true) median {cal_med:+.2f}  "
              f"IQR {cal_iqr:.2f}  ({verdict})")
    print("=" * 72)
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

    analyze_gaps(model, test_X, species_active, species_type_ids,
                 ruleset, hyp, sbml,
                 element_balances(sbml, species_active, raw_counts_active),
                 preds_batch=preds_batch, truth_batch=truth_batch)

    # ── v9: knockout sweep vs Breuer 2019 essentiality ───────────────────
    breuer_labels = load_breuer_essentiality(BREUER_PATH)
    ko = knockout_sweep(model, ruleset, test_X, species_active, breuer_labels)
    print_knockout_report(ko, breuer_labels)

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
    main()
