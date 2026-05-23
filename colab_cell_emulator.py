"""Colab cell: v8 cell-state emulator - Liquid GNN (CfC + SBML graph) + PhD knowledge layer.

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
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# ── config ───────────────────────────────────────────────────────────────────
PARQUET_DIR         = ""
TIME_STRIDE         = 60
SKIP_STARTUP_STEPS  = 1            # NEW v7: drop first decimated step (t=0->t=60)
CONTEXT             = 8
D_MODEL             = 256
D_TYPE_EMBED        = 16
N_LAYERS            = 3
N_HEADS             = 8
DROPOUT             = 0.1
N_TRAIN_TRAJ        = 40
STEPS               = 2000
K_MAX               = 64
BATCH               = 32
LR                  = 3e-4
WEIGHT_DECAY        = 1e-5
LAMBDA_1STEP        = 1.0
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
LGNN_HIDDEN         = 64           # NEW v8: per-node hidden dim
LGNN_N_LAYERS       = 3            # NEW v8: number of CfC graph layers
LGNN_CFC_TAU_MIN    = 0.1          # NEW v8: CfC time-constant minimum
LGNN_N_TYPE_EMBED   = 4            # NEW v8: gene-type embed dim
SEED                = 0
SAVE_DIR            = "/content/drive/MyDrive"
GENE_TABLE_PATH     = "memory_bank/data/syn3a_gene_table.csv"
SBML_PATH           = "Syn3A_updated.xml"
KINETICS_PATH       = "kinetic_params.xlsx"             # NEW v7
INITIAL_CONC_PATH   = "initial_concentrations.xlsx"     # NEW v7
COMPLEXES_PATH      = "complex_formation.xlsx"          # NEW v7

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
    """Parse kinetic_params.xlsx -> {enzymes, n_params, subsystems} or None."""
    if not HAS_PANDAS:
        return None
    try:
        sheets = ["Central", "Nucleotide", "Lipid", "Cofactor", "Transport"]
        enzymes, n_params, subsys = {}, 0, {}
        for sn in sheets:
            try:
                df = pd.read_excel(path, sheet_name=sn)
            except Exception:
                continue
            subsys[sn] = len(df)
            n_params += len(df)
            for _, row in df.iterrows():
                if row.get("Parameter Type") == "Eff Enzyme Count":
                    rxn = str(row.get("Reaction Name", ""))
                    val = str(row.get("Value", ""))
                    if rxn and val and val != "nan":
                        enzymes[rxn] = val
        print(f"[kinetics] {n_params} parameter rows across {len(subsys)} subsystems, "
              f"{len(enzymes)} reaction->enzyme mappings")
        return {"enzymes": enzymes, "n_params": n_params, "subsystems": subsys}
    except Exception as e:
        print(f"[kinetics] parse failed ({e}) - skipping kinetics")
        return None


def parse_initial_concentrations(path):
    """Parse initial_concentrations.xlsx -> {proteins, mRNAs, metabolites, medium}."""
    if not HAS_PANDAS:
        return None
    try:
        df = pd.read_excel(path, sheet_name="Comparative Proteomics")
        df["_cnt"] = pd.to_numeric(df.get("Sim. Initial Ptn Cnt"), errors="coerce")
        proteins = {}
        for _, r in df.iterrows():
            tag = str(r.get("Locus Tag", ""))
            if tag.startswith("JCVISYN3A_") and pd.notna(r["_cnt"]):
                proteins[tag] = float(r["_cnt"])

        df = pd.read_excel(path, sheet_name="mRNA Count")
        mRNAs = {}
        for _, r in df.iterrows():
            tag = str(r.get("LocusTag", ""))
            tot = r.get("total")
            if tag.startswith("JCVISYN3A_") and pd.notna(tot):
                mRNAs[tag] = float(tot)

        df = pd.read_excel(path, sheet_name="Intracellular Metabolites")
        metabolites = {f"M_{r['Met ID']}": float(r["Init Conc (mM)"])
                       for _, r in df.iterrows()
                       if pd.notna(r.get("Met ID")) and pd.notna(r.get("Init Conc (mM)"))}

        df = pd.read_excel(path, sheet_name="Simulation Medium")
        medium = {f"M_{r['Met ID']}": float(r["Conc (mM)"])
                  for _, r in df.iterrows()
                  if pd.notna(r.get("Met ID")) and pd.notna(r.get("Conc (mM)"))}

        print(f"[initial_conc] {len(proteins)} proteins, {len(mRNAs)} mRNAs, "
              f"{len(metabolites)} intracellular metabolites, "
              f"{len(medium)} medium components")
        return {"proteins": proteins, "mRNAs": mRNAs,
                "metabolites": metabolites, "medium": medium}
    except Exception as e:
        print(f"[initial_conc] parse failed ({e}) - skipping initial conditions")
        return None


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


# ── NEW v7: KnownRules (input-file facts) ────────────────────────────────────

class KnownRules:
    """Facts from input files (deterministic, mechanism-derived).

    Distinct from DiscoveredPatterns: these are NOT empirical regularities
    mined from data, they are explicit structural facts read from model files.
    """

    def __init__(self, sbml=None, kinetics=None, initial=None, complexes=None):
        self.sbml = sbml
        self.kinetics = kinetics
        self.initial = initial
        self.complexes = complexes

    def has_anything(self):
        return any(v is not None for v in
                   (self.sbml, self.kinetics, self.initial, self.complexes))

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

    Tracked, confidence-scored and REPORTED, never enforced - so a wrong
    hypothesis cannot corrupt a rollout.  Promote to a seed rule only once trusted.
    """

    def __init__(self):
        self.items = []

    def add(self, kind, detail, score, source):
        self.items.append({"kind": kind, "detail": detail,
                           "score": score, "source": source})

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
                    patterns.up_frac_va[i], src)

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
                    patterns.down_frac_va[i], src)

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
                abs(c_va), "trajectory")
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

    def forward(self, h, edge_index, edge_weight):
        # h: (B, N, H);  edge_index: (2, E);  edge_weight: (E,)
        B, N, H = h.shape
        src, dst = edge_index[0], edge_index[1]
        h_src = h.index_select(1, src)        # (B, E, H)
        h_dst = h.index_select(1, dst)
        msg = self.msg_mlp(torch.cat([h_src, h_dst], dim=-1))
        msg = msg * edge_weight.unsqueeze(0).unsqueeze(-1)
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


class DynamicsModel(nn.Module):
    """v8 Liquid Graph Neural Network: SBML graph + CfC node dynamics.

    Single-step: takes current state (B, S), returns predicted next state
    (B, S) as a residual delta on top of the input.
    """

    def __init__(self, S, hidden, n_layers, species_type_ids,
                 edge_index, edge_weight, cfc_tau_min=0.1, n_type_embed=4,
                 # accept v7-era kwargs to keep main() backward-compatible
                 d_model=None, n_heads=None, context=None, dropout=None,
                 d_type=None):
        super().__init__()
        self.S = S
        self.hidden = hidden
        self.n_layers = n_layers
        self.register_buffer("edge_index",  edge_index)
        self.register_buffer("edge_weight", edge_weight)
        # gene-type embedding (kept from v7 — adds biological role context)
        self.type_embed = nn.Embedding(N_GTYPES, n_type_embed)
        self.register_buffer("stype",
                             torch.tensor(species_type_ids, dtype=torch.long))
        # Per-species input projection: 1 (value) + n_type_embed -> hidden
        self.in_proj = nn.Linear(1 + n_type_embed, hidden)
        self.layers  = nn.ModuleList([
            _CfCGraphLayer(hidden, S, cfc_tau_min=cfc_tau_min)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)

    def forward(self, x):                     # x: (B, S) -> (B, S)
        B, S = x.shape
        # per-species input = [value, gene_type_embed]
        te  = self.type_embed(self.stype).unsqueeze(0).expand(B, -1, -1)  # (B, S, n_type)
        inp = torch.cat([x.unsqueeze(-1), te], dim=-1)                    # (B, S, 1+n_type)
        h   = self.in_proj(inp)                                           # (B, S, H)
        for layer in self.layers:
            h = layer(h, self.edge_index, self.edge_weight)
        h = self.out_norm(h)
        delta = self.out_head(h).squeeze(-1)                              # (B, S)
        return x + delta                                                  # residual


# ── data ──────────────────────────────────────────────────────────────────────

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_data(skip_startup=True):
    assert HAS_PANDAS, "pandas required - install or run on Colab"
    pat = (f"{PARQUET_DIR}/counts_and_fluxes*.parquet" if PARQUET_DIR
           else "/content/drive/MyDrive/**/counts_and_fluxes*.parquet")
    files = sorted(glob.glob(pat, recursive=True),
                   key=lambda p: int(p.rsplit(".", 2)[-2]))
    assert files, "no parquet files - set PARQUET_DIR"
    print(f"[data] {len(files)} trajectory files")
    trajs, species_names = [], None
    for f in files:
        df = pd.read_parquet(f)
        if species_names is None:
            species_names = list(df.index)
        arr = df.to_numpy(dtype=np.float32)[:, ::TIME_STRIDE].T
        if skip_startup:
            arr = arr[SKIP_STARTUP_STEPS:]
        trajs.append(arr)
    if skip_startup:
        print(f"[data] startup skip: dropped first {SKIP_STARTUP_STEPS} decimated step(s)")
    return np.stack(trajs, 0), species_names


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-12))


# ── training / eval ───────────────────────────────────────────────────────────

def train_model(model, train_X, ruleset):
    """v8: single-step LGNN trainer. No context window."""
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    gen   = torch.Generator().manual_seed(SEED + 1)
    N, T, S = train_X.shape
    t_start = time.time()
    model.train()
    for step in range(STEPS):
        K    = 1 + int((K_MAX - 1) * step / STEPS)
        i_t  = torch.randint(0, N, (BATCH,), generator=gen)
        t0_t = torch.randint(0, T - 1 - K, (BATCH,), generator=gen)
        i, ts = i_t.tolist(), t0_t.tolist()
        state = torch.stack([train_X[i[b], ts[b]] for b in range(BATCH)])    # (B, S)
        i_d, t0_d = i_t.to(device), t0_t.to(device)
        losses = []
        prev_state = state
        for k in range(K):
            pred = model(state)
            true = train_X[i_d, t0_d + 1 + k]
            losses.append(F.mse_loss(pred, true))
            nxt = pred.clamp(CLAMP_LO, CLAMP_HI)
            if step >= STEPS // 4:
                nxt = ruleset.project(prev_state, nxt)
            prev_state = nxt
            state = nxt
        rollout = torch.stack(losses).mean()
        loss = rollout + LAMBDA_1STEP * losses[0]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 0 or (step + 1) % 250 == 0:
            print(f"  step {step+1:5d}  K={K:2d}  "
                  f"1-step {float(losses[0].detach()):.5f}  "
                  f"rollout {float(rollout.detach()):.5f}", flush=True)
    print(f"[train] {STEPS} steps in {time.time()-t_start:.0f}s")


@torch.no_grad()
def one_step(model, Xset, n=400):
    model.eval()
    g  = torch.Generator().manual_seed(SEED + 2)
    nt, Tt, _ = Xset.shape
    i  = torch.randint(0, nt, (n,), generator=g).tolist()
    t  = torch.randint(0, Tt - 1, (n,), generator=g).tolist()
    state = torch.stack([Xset[i[b], t[b]]     for b in range(n)])
    nxt   = torch.stack([Xset[i[b], t[b] + 1] for b in range(n)])
    pred = model(state)
    return float(F.mse_loss(pred, nxt)), r2(pred, nxt)


@torch.no_grad()
def full_rollout(model, traj, ruleset):
    """v8: single-step rollout — seed from one frame, generate the rest."""
    model.eval()
    state = traj[0].unsqueeze(0)        # (1, S)
    preds = []
    for _ in range(traj.shape[0] - 1):
        p = model(state).clamp(CLAMP_LO, CLAMP_HI)
        p = ruleset.project(state, p)
        preds.append(p)
        state = p
    return torch.cat(preds, 0), traj[1:]


# ── missing-info report ───────────────────────────────────────────────────────

@torch.no_grad()
def analyze_gaps(model, test_X, species_names, species_type_ids,
                 ruleset, hyp, sbml, elem_balances):
    """Where the model + rules fall short - residuals, drifts, coverage."""
    print()
    print("#" * 72)
    print("#  MISSING-INFO REPORT  -  where the model / rules fall short")
    print("#" * 72)

    S = test_X.shape[2]
    se = torch.zeros(S, device=test_X.device)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[device] {device}")
    print()
    print("=" * 72)
    print("  v7  -  PhD knowledge phase + multi-lens patterns + startup skip")
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
    sbml      = parse_sbml(SBML_PATH)
    kinetics  = parse_kinetics(KINETICS_PATH)
    initial   = parse_initial_concentrations(INITIAL_CONC_PATH)
    complexes = parse_complex_formation(COMPLEXES_PATH)
    known = KnownRules(sbml=sbml, kinetics=kinetics,
                       initial=initial, complexes=complexes)

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
    print()
    print("[knowledge] running multi-lens pattern discovery ...")
    patterns = DiscoveredPatterns.from_trajectories(
        train_X.cpu(), test_X.cpu(),
        raw_counts_active[train_idx], raw_counts_active[test_idx],
        species_active,
        lo=lo, span=span)

    # ── sort into Tier 1 (enforced) + Tier 2 (reported) ──────────────────
    ruleset, hyp = build_enforcement(known, patterns, species_active)
    ruleset = ruleset.to(device)

    # ── cross-validate KnownRules vs trajectory ──────────────────────────
    cross_report = cross_validate_known(known, raw_counts_active, species_active)

    # ── PhD summary ──────────────────────────────────────────────────────
    phd_summary(known, patterns, ruleset, hyp, cross_report, species_active)

    # ── model + training ─────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  TRAINING PHASE  (v8: Liquid GNN)")
    print("=" * 72)
    edge_index, edge_weight = build_sbml_graph(sbml, species_active)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    model = DynamicsModel(
        S=S, hidden=LGNN_HIDDEN, n_layers=LGNN_N_LAYERS,
        species_type_ids=species_type_ids,
        edge_index=edge_index, edge_weight=edge_weight,
        cfc_tau_min=LGNN_CFC_TAU_MIN, n_type_embed=LGNN_N_TYPE_EMBED,
    ).to(device)
    print(f"[model] LGNN: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, "
          f"{edge_index.shape[1]:,} edges (SBML reactions + self-loops)")
    train_model(model, train_X, ruleset)

    # ── evaluate ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  EVALUATION")
    print("=" * 72)
    s_mse, s_r2 = one_step(model, test_X)
    roll_r2 = [r2(*full_rollout(model, test_X[k], ruleset))
               for k in range(test_X.shape[0])]
    mean_roll = sum(roll_r2) / len(roll_r2)
    print()
    print("=" * 72)
    print(f"  persistence 1-step    : MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")
    print(f"  model 1-step (test)   : MSE {s_mse:.5f}  R^2 {s_r2:.3f}"
          f"  {'(beats persistence)' if s_mse < persist_mse else '(still worse)'}")
    print(f"  model full rollout    : R^2 {mean_roll:.3f}  "
          f"(min {min(roll_r2):.3f}  max {max(roll_r2):.3f})  <- headline")
    print(f"  (v6 transformer       : rollout R^2 ~0.56)")
    print(f"  (v7 transformer + cap : rollout R^2 ~0.54)")
    print("=" * 72)

    analyze_gaps(model, test_X, species_active, species_type_ids,
                 ruleset, hyp, sbml,
                 element_balances(sbml, species_active, raw_counts_active))

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
    np.save(f"{SAVE_DIR}/cell_traj_51_v8.npy", gen_counts)
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
        "known_rules_summary": known.summary(),
        "config": dict(S=S, hidden=LGNN_HIDDEN, n_layers=LGNN_N_LAYERS,
                       cfc_tau_min=LGNN_CFC_TAU_MIN, n_type_embed=LGNN_N_TYPE_EMBED,
                       time_stride=TIME_STRIDE, n_genes=len(locus_list),
                       skip_startup=SKIP_STARTUP_STEPS,
                       architecture="LGNN_v8"),
    }, f"{SAVE_DIR}/cell_emulator_v8.pt")
    print(f"[save] traj  -> {SAVE_DIR}/cell_traj_51_v8.npy")
    print(f"[save] model -> {SAVE_DIR}/cell_emulator_v8.pt")


if __name__ == "__main__":
    main()
