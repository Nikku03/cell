"""Colab cell: v6 cell-state emulator - two-tier rules + SBML structure + gaps.

v5 -> v6:
  1. SEED RULES  an explicit, editable block of rules we KNOW (monotone
     counters, non-negativity).  Hand-written rules and discovered rules go
     through the SAME held-out validation gate - a seed rule that fails is
     reported, never silently trusted.
  2. RULES FROM INPUT DATA (4DWCM SBML reaction network, Syn3A_updated.xml)
       - pure-product species (only ever produced)  -> monotone-up candidate
       - pure-reactant species (only ever consumed)  -> monotone-down candidate
       - element balances from fbc:chemicalFormula   -> conservation candidates
     SBML candidates are validated against the trajectory, same gate.
  3. TWO TIERS
       Tier 1  RuleSet     - validated, ENFORCED as hard rollout guardrails.
                             "It shouldn't be wrong": enforced == validated.
       Tier 2  Hypotheses  - candidates that did NOT fully validate (a law
                             that almost holds, a drift).  Tracked, scored,
                             REPORTED, never enforced - so they can be wrong
                             without corrupting anything.
  4. MISSING-INFO REPORT  reasons about what the model/rules don't cover:
     worst-predicted species, element balances that drift (unmodeled flux),
     SBML<->trajectory coverage mismatch.

Run on Colab with Drive mounted, GPU runtime (~10 min).
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
PARQUET_DIR    = ""
TIME_STRIDE    = 60
CONTEXT        = 8
D_MODEL        = 256
D_TYPE_EMBED   = 16
N_LAYERS       = 3
N_HEADS        = 8
DROPOUT        = 0.1
N_TRAIN_TRAJ   = 40
STEPS          = 2000
K_MAX          = 64
BATCH          = 32
LR             = 3e-4
WEIGHT_DECAY   = 1e-5
LAMBDA_1STEP   = 1.0
CLAMP_LO, CLAMP_HI = -0.2, 1.2
MONO_EPS        = 1e-4       # tolerance for monotone-rule mining
RULE_COMPLIANCE = 0.999      # held-out compliance threshold for Tier 1
CONSERVE_DRIFT  = 0.02       # a balance drifting < 2% is "conserved"
SEED            = 0
SAVE_DIR        = "/content/drive/MyDrive"
GENE_TABLE_PATH = "memory_bank/data/syn3a_gene_table.csv"
# SBML reaction network. If absent, SBML-derived rules are skipped (non-fatal).
# Re-stage from the Minimal_Cell_ComplexFormation repo or point at your Drive copy.
SBML_PATH       = "Syn3A_updated.xml"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SEED RULES  -  the rules we KNOW.  Edit / extend this block freely.      ║
# ║  Everything here is still validated on held-out data before it is        ║
# ║  enforced (Tier 1).  A seed rule that fails validation is reported.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
SEED_MONOTONE_UP_CHANNELS = {"RPM", "PM", "DM"}   # cumulative made-counters
SEED_NONNEG               = True                  # all molecule counts >= 0
# To add a rule: extend a set above, or add an SBML file via SBML_PATH, or
# append a custom (kind, species, params) entry handled in discover_rules().


# ── species-name parsing ──────────────────────────────────────────────────────

_CD_PREFIXES = sorted([
    "RB_pe", "RB_cp", "RB_p", "RP_f", "P_TC",
    "RPM", "R_d", "C_P",
    "RB", "RP", "DM", "PM", "DT",
    "G", "R", "P", "S", "D",
], key=len, reverse=True)

CHAN_NAMES = [
    "G", "R", "R_d", "RP", "RP_f",
    "RB", "RB_p", "RB_pe", "RB_cp",
    "P", "C_P", "P_TC", "S", "D", "DT",
    "RPM", "PM", "DM",
]
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


def _locus_key(locus):
    """'0412_C1' -> '0412'.  Strips chromosome-copy / variant suffixes so all
    species of one gene (both chromosome copies, bound intermediates) share a
    gene identity and match the gene-table locus numbers."""
    m = re.match(r"\d+", locus)
    return m.group(0) if m else locus


def build_gene_index(species_names, gene_type_map):
    """Per-species gene-type labelling.  Returns species_type_ids (S,) and the
    locus list (used only for the saved config)."""
    locus_to_idx, locus_list = {}, []
    for name in species_names:
        prefix, locus = parse_species(name)
        if prefix is not None:
            key = _locus_key(locus)
            if key not in locus_to_idx:
                locus_to_idx[key] = len(locus_list)
                locus_list.append(key)

    gene_type_ids = np.array(
        [gene_type_map.get(loc, GTYPE_OTHER) for loc in locus_list],
        dtype=np.int32)

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


# ── SBML parsing (input-data rules) ───────────────────────────────────────────

def _formula_atoms(formula):
    """'C10H12N5O13P3' -> {'C':10,'H':12,'N':5,'O':13,'P':3}."""
    atoms = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        atoms[el] = atoms.get(el, 0) + (int(n) if n else 1)
    return atoms


def parse_sbml(path):
    """Parse an SBML L3 (+fbc) reaction network.

    Returns dict {species: {id: {formula, atoms}}, reactions: [...]} or None.
    Namespace-robust: matches tags / attributes by local name.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        print(f"[sbml] could not read {path} ({e}) - SBML rules skipped")
        return None

    def local(tag):
        return tag.rsplit("}", 1)[-1]

    def attr(elem, name):
        for k, v in elem.attrib.items():
            if local(k) == name:
                return v
        return None

    species, reactions = {}, []
    for elem in root.iter():
        ln = local(elem.tag)
        if ln == "species":
            sid = attr(elem, "id")
            formula = attr(elem, "chemicalFormula") or ""
            species[sid] = {"formula": formula, "atoms": _formula_atoms(formula)}
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
    """Structural monotonicity from reaction topology.

    pure product (produced, never consumed)  -> can only increase
    pure reactant (consumed, never produced) -> can only decrease
    A reversible reaction counts a species as BOTH produced and consumed.
    Returns (up_idx set, down_idx set) of indices into species_names.
    """
    if sbml is None:
        return set(), set()
    produced, consumed = set(), set()
    for rxn in sbml["reactions"]:
        for sid, _ in rxn["reactants"]:
            consumed.add(sid)
            if rxn["reversible"]:
                produced.add(sid)
        for sid, _ in rxn["products"]:
            produced.add(sid)
            if rxn["reversible"]:
                consumed.add(sid)
    name_to_idx = {n: i for i, n in enumerate(species_names)}
    up   = {name_to_idx[s] for s in (produced - consumed) if s in name_to_idx}
    down = {name_to_idx[s] for s in (consumed - produced) if s in name_to_idx}
    print(f"[sbml] structural monotone candidates: {len(up)} up, {len(down)} down")
    return up, down


def element_balances(sbml, species_names, raw_counts):
    """Element-conservation candidates from fbc:chemicalFormula.

    For element E, Q_E(t) = sum_s atoms_E(s) * count_s(t) over the metabolite
    species present in the trajectory.  In a closed system Q_E is conserved;
    in a growing cell it drifts - that drift IS the missing-info signal.

    raw_counts : (n_traj, T, S) raw (pre-signed-log) counts.
    Returns list of dicts {element, n_species, drift_frac, conserved}.
    """
    if sbml is None:
        return []
    metab_cols, metab_atoms = [], []
    for i, name in enumerate(species_names):
        if name in sbml["species"] and sbml["species"][name]["atoms"]:
            metab_cols.append(i)
            metab_atoms.append(sbml["species"][name]["atoms"])
    if not metab_cols:
        print("[sbml] no SBML metabolites matched trajectory species")
        return []

    elements = sorted({e for a in metab_atoms for e in a})
    E_mat = np.array([[a.get(e, 0) for e in elements] for a in metab_atoms],
                     dtype=np.float64)                       # (n_metab, n_elem)
    counts = raw_counts[:, :, metab_cols].astype(np.float64) # (n_traj, T, n_metab)
    Q = counts @ E_mat                                       # (n_traj, T, n_elem)

    out = []
    for j, el in enumerate(elements):
        q = Q[:, :, j]
        mean = q.mean(axis=1)
        rng  = q.max(axis=1) - q.min(axis=1)
        drift = float(np.mean(rng / np.clip(np.abs(mean), 1e-9, None)))
        out.append({"element": el, "n_species": len(metab_cols),
                    "drift_frac": drift, "conserved": drift < CONSERVE_DRIFT})
    print(f"[sbml] element balances over {len(metab_cols)} metabolites: "
          + ", ".join(f"{d['element']} {d['drift_frac']*100:.1f}%" for d in out))
    return out


# ── rule system ───────────────────────────────────────────────────────────────

class RuleSet:
    """Tier 1: validated rules, enforced as hard rollout guardrails."""

    def __init__(self):
        self.mono_up_mask   = None   # (S,) bool - must not decrease
        self.mono_down_mask = None   # (S,) bool - must not increase
        self.mono_up        = None   # 1-D int tensor (checkpointing)
        self.mono_down      = None
        self.lo_bound       = None   # (S,) float
        self.hi_bound       = None   # (S,) float
        self.n_seed = self.n_sbml = self.n_empirical = 0

    def to(self, dev):
        for a in ("mono_up_mask", "mono_down_mask", "mono_up", "mono_down",
                  "lo_bound", "hi_bound"):
            t = getattr(self, a, None)
            if t is not None:
                setattr(self, a, t.to(dev))
        return self

    def project(self, prev, nxt):
        """Enforce every rule. prev, nxt: (B, S). Autograd-safe (no in-place)."""
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
        return (f"Tier 1 RuleSet: {nu} monotone-up + {nd} monotone-down "
                f"+ per-species bounds  "
                f"[provenance: {self.n_seed} seed, {self.n_sbml} SBML-backed, "
                f"{self.n_empirical} data-only]")


class Hypotheses:
    """Tier 2: candidate rules that did NOT pass validation.

    Tracked, confidence-scored and REPORTED, but never enforced - so a wrong
    hypothesis cannot corrupt a rollout.  This is the "something that can be
    wrong" tier; promote an item to a seed rule only once you trust it.
    """

    def __init__(self):
        self.items = []   # {kind, detail, score, source}

    def add(self, kind, detail, score, source):
        self.items.append({"kind": kind, "detail": detail,
                            "score": score, "source": source})

    def summary(self):
        if not self.items:
            return "Tier 2 Hypotheses: none"
        lines = [f"Tier 2 Hypotheses: {len(self.items)} (reported, NOT enforced)"]
        for h in sorted(self.items, key=lambda x: -x["score"])[:12]:
            lines.append(f"    [{h['source']:10s}] {h['kind']:14s} "
                         f"{h['detail']}  (score {h['score']:.3f})")
        if len(self.items) > 12:
            lines.append(f"    ... and {len(self.items)-12} more")
        return "\n".join(lines)


def discover_rules(train_X, val_X, species_names, sbml=None):
    """Mine + validate rules, sort into Tier 1 (RuleSet) and Tier 2 (Hypotheses).

    train_X, val_X : (n_traj, T, S) CPU float32 (normalised)

    Every candidate - seed, SBML-derived or empirically mined - is validated
    on HELD-OUT trajectories at the RULE_COMPLIANCE gate.  Pass -> Tier 1.
    Fail -> Tier 2.
    """
    tr, va = train_X.numpy(), val_X.numpy()
    n_tr, T, S = tr.shape

    d_tr = np.diff(tr, axis=1)
    d_va = np.diff(va, axis=1)
    steps_tr = n_tr * (T - 1)
    steps_va = va.shape[0] * (T - 1)

    # per-species: fraction of steps that DECREASE / INCREASE
    dec_tr = (d_tr < -MONO_EPS).sum(axis=(0, 1)) / steps_tr
    inc_tr = (d_tr >  MONO_EPS).sum(axis=(0, 1)) / steps_tr
    dec_va = (d_va < -MONO_EPS).sum(axis=(0, 1)) / steps_va
    inc_va = (d_va >  MONO_EPS).sum(axis=(0, 1)) / steps_va
    ok_up_va   = 1.0 - dec_va     # compliance with "never decreases"
    ok_down_va = 1.0 - inc_va     # compliance with "never increases"

    hyp = Hypotheses()

    # ── seed rules: monotone-up channels ─────────────────────────────────
    seed_up = {i for i, nm in enumerate(species_names)
               if parse_species(nm)[0] in SEED_MONOTONE_UP_CHANNELS}

    # ── SBML structural candidates ───────────────────────────────────────
    sbml_up, sbml_down = sbml_monotone_candidates(sbml, species_names)

    # ── empirical candidates ─────────────────────────────────────────────
    emp_up   = set(np.where(dec_tr < (1.0 - RULE_COMPLIANCE))[0].tolist())
    emp_down = set(np.where(inc_tr < (1.0 - RULE_COMPLIANCE))[0].tolist())

    # ── validate every monotone-up candidate on held-out data ────────────
    cand_up = seed_up | sbml_up | emp_up
    val_up  = set()
    for i in cand_up:
        if ok_up_va[i] >= RULE_COMPLIANCE:
            val_up.add(i)
        else:
            src = ("seed" if i in seed_up else
                   "sbml" if i in sbml_up else "trajectory")
            hyp.add("monotone-up", f"'{species_names[i]}' fails held-out "
                    f"({ok_up_va[i]*100:.2f}% compliant)", ok_up_va[i], src)

    # monotone-down (SBML + empirical; no seed-down by default)
    cand_down = sbml_down | emp_down
    val_down  = set()
    for i in cand_down:
        if ok_down_va[i] >= RULE_COMPLIANCE and i not in val_up:
            val_down.add(i)
        elif i not in val_up:
            src = "sbml" if i in sbml_down else "trajectory"
            hyp.add("monotone-down", f"'{species_names[i]}' fails held-out "
                    f"({ok_down_va[i]*100:.2f}% compliant)", ok_down_va[i], src)

    # ── per-species bounds (validated) ───────────────────────────────────
    tr_flat, va_flat = tr.reshape(-1, S), va.reshape(-1, S)
    lo_cand = tr_flat.min(axis=0) - 0.1
    hi_cand = tr_flat.max(axis=0) + 0.1
    bound_ok = ((va_flat >= lo_cand).all(axis=0) &
                (va_flat <= hi_cand).all(axis=0))

    # ── assemble Tier 1 ──────────────────────────────────────────────────
    rs = RuleSet()
    up_sorted, down_sorted = sorted(val_up), sorted(val_down)
    rs.mono_up   = torch.tensor(up_sorted,   dtype=torch.long)
    rs.mono_down = torch.tensor(down_sorted, dtype=torch.long)
    um = torch.zeros(S, dtype=torch.bool)
    dm = torch.zeros(S, dtype=torch.bool)
    if up_sorted:   um[rs.mono_up]   = True
    if down_sorted: dm[rs.mono_down] = True
    rs.mono_up_mask, rs.mono_down_mask = um, dm
    rs.lo_bound = torch.from_numpy(np.where(bound_ok, lo_cand, CLAMP_LO).astype(np.float32))
    rs.hi_bound = torch.from_numpy(np.where(bound_ok, hi_cand, CLAMP_HI).astype(np.float32))

    rs.n_seed       = len(val_up & seed_up)
    rs.n_sbml       = len((val_up & sbml_up) | (val_down & sbml_down))
    rs.n_empirical  = len((val_up | val_down)
                          - seed_up - sbml_up - sbml_down)

    # warn on seed rules that failed
    for i in seed_up - val_up:
        print(f"[rules] WARNING: seed rule monotone-up '{species_names[i]}' "
              f"failed held-out validation - moved to Tier 2")

    print(f"[rules] monotone-up : {len(cand_up)} candidates -> "
          f"{len(val_up)} validated (Tier 1)")
    print(f"[rules] monotone-dn : {len(cand_down)} candidates -> "
          f"{len(val_down)} validated (Tier 1)")
    print(f"[rules] bounds      : {int(bound_ok.sum())}/{S} validated (Tier 1)")
    return rs, hyp


# ── model ─────────────────────────────────────────────────────────────────────

class DynamicsModel(nn.Module):
    """Transformer over a CONTEXT-step window -> next-state residual.

    Each timestep token = in_proj(species_values) || mean_gene_type_embed.
    """

    def __init__(self, S, d_model, n_layers, n_heads, context, dropout,
                 species_type_ids, d_type=D_TYPE_EMBED):
        super().__init__()
        self.d_type = d_type
        self.type_embed = nn.Embedding(N_GTYPES, d_type)
        self.register_buffer("stype",
                             torch.tensor(species_type_ids, dtype=torch.long))
        self.in_proj = nn.Linear(S, d_model - d_type)
        self.ctx_pos = nn.Parameter(torch.randn(context, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.out = nn.Linear(d_model, S)

    def forward(self, ctx):                         # (B, C, S) -> (B, S)
        B, C, _ = ctx.shape
        h_val = self.in_proj(ctx)
        te = self.type_embed(self.stype).mean(0)
        te = te.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        h = torch.cat([h_val, te], dim=-1) + self.ctx_pos
        h = self.encoder(h)
        return ctx[:, -1] + self.out(h[:, -1])


# ── data ──────────────────────────────────────────────────────────────────────

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_data():
    assert HAS_PANDAS, "pandas is required - install it or run on Colab"
    pat = (f"{PARQUET_DIR}/counts_and_fluxes*.parquet" if PARQUET_DIR
           else "/content/drive/MyDrive/**/counts_and_fluxes*.parquet")
    files = sorted(glob.glob(pat, recursive=True),
                   key=lambda p: int(p.rsplit(".", 2)[-2]))
    assert files, "no parquet files found - set PARQUET_DIR"
    print(f"[data] {len(files)} trajectory files")
    trajs, species_names = [], None
    for f in files:
        df = pd.read_parquet(f)
        if species_names is None:
            species_names = list(df.index)
        trajs.append(df.to_numpy(dtype=np.float32)[:, ::TIME_STRIDE].T)
    return np.stack(trajs, 0), species_names


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-12))


# ── train / eval ──────────────────────────────────────────────────────────────

def train_model(model, train_X, ruleset):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    gen   = torch.Generator().manual_seed(SEED + 1)
    N, T, S = train_X.shape
    t_start = time.time()
    model.train()
    for step in range(STEPS):
        K    = 1 + int((K_MAX - 1) * step / STEPS)
        i_t  = torch.randint(0, N, (BATCH,), generator=gen)
        t0_t = torch.randint(0, T - CONTEXT - K, (BATCH,), generator=gen)
        i, ts = i_t.tolist(), t0_t.tolist()
        ctx   = torch.stack([train_X[i[b], ts[b]:ts[b] + CONTEXT]
                             for b in range(BATCH)])
        i_d, t0_d = i_t.to(device), t0_t.to(device)
        losses = []
        prev_last = ctx[:, -1]
        for k in range(K):
            pred = model(ctx)
            true = train_X[i_d, t0_d + CONTEXT + k]
            losses.append(F.mse_loss(pred, true))
            nxt = pred.clamp(CLAMP_LO, CLAMP_HI)
            if step >= STEPS // 4:
                nxt = ruleset.project(prev_last, nxt)
            prev_last = nxt
            ctx = torch.cat([ctx[:, 1:], nxt.unsqueeze(1)], dim=1)
        rollout = torch.stack(losses).mean()
        loss    = rollout + LAMBDA_1STEP * losses[0]
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
    g = torch.Generator().manual_seed(SEED + 2)
    nt, Tt, _ = Xset.shape
    i  = torch.randint(0, nt, (n,), generator=g).tolist()
    t0 = torch.randint(0, Tt - CONTEXT - 1, (n,), generator=g).tolist()
    ctx = torch.stack([Xset[i[b], t0[b]:t0[b] + CONTEXT] for b in range(n)])
    nxt = torch.stack([Xset[i[b], t0[b] + CONTEXT]       for b in range(n)])
    pred = model(ctx)
    return float(F.mse_loss(pred, nxt)), r2(pred, nxt)


@torch.no_grad()
def full_rollout(model, traj, ruleset):
    model.eval()
    ctx = traj[:CONTEXT].unsqueeze(0)
    preds = []
    for _ in range(traj.shape[0] - CONTEXT):
        p = model(ctx).clamp(CLAMP_LO, CLAMP_HI)
        p = ruleset.project(ctx[:, -1], p)
        preds.append(p)
        ctx = torch.cat([ctx[:, 1:], p.unsqueeze(1)], dim=1)
    return torch.cat(preds, 0), traj[CONTEXT:]


# ── missing-info report ───────────────────────────────────────────────────────

@torch.no_grad()
def analyze_gaps(model, test_X, species_names, species_type_ids,
                 ruleset, hyp, sbml, elem_balances):
    """Reason about what the model / rules do NOT cover.

    Surfaces three kinds of missing info:
      - species the trained model predicts worst (needs mechanism we lack),
      - element balances that drift (unmodeled flux in/out of the pool),
      - SBML <-> trajectory coverage mismatch.
    """
    print()
    print("#" * 64)
    print("#  MISSING-INFO REPORT  -  where the model / rules fall short")
    print("#" * 64)

    # 1. worst-predicted species ------------------------------------------
    S = test_X.shape[2]
    se = torch.zeros(S, device=test_X.device)
    for k in range(test_X.shape[0]):
        pred, true = full_rollout(model, test_X[k], ruleset)
        se += ((pred - true) ** 2).mean(0)
    se /= test_X.shape[0]
    worst = torch.argsort(se, descending=True)[:15].tolist()
    tname = {GTYPE_PROTEIN: "protein", GTYPE_TRNA: "tRNA", GTYPE_RRNA: "rRNA",
             GTYPE_OTHER: "other", GTYPE_GLOBAL: "global"}
    print("\n  [1] species the model predicts worst (rollout MSE) - these")
    print("      most need a rule / mechanism we don't yet have:")
    for i in worst:
        print(f"      {species_names[i]:22s}  MSE {float(se[i]):.4f}  "
              f"({tname[int(species_type_ids[i])]})")

    # 2. drifting element balances ----------------------------------------
    print("\n  [2] element balances (SBML) - drift = unmodeled flux into")
    print("      macromolecules / across the cell boundary:")
    if elem_balances:
        for d in sorted(elem_balances, key=lambda x: -x["drift_frac"]):
            tag = "CONSERVED" if d["conserved"] else "drifts"
            print(f"      {d['element']:3s}  {tag:9s}  "
                  f"{d['drift_frac']*100:6.1f}%  over the cell cycle")
    else:
        print("      (no SBML provided - skipped)")

    # 3. SBML <-> trajectory coverage -------------------------------------
    print("\n  [3] SBML <-> trajectory coverage:")
    if sbml is not None:
        traj_set = set(species_names)
        sbml_set = set(sbml["species"])
        matched  = traj_set & sbml_set
        print(f"      SBML species          : {len(sbml_set)}")
        print(f"      matched in trajectory : {len(matched)}")
        print(f"      SBML-only (untracked) : {len(sbml_set - traj_set)}")
        rxn_sp = {s for r in sbml["reactions"]
                  for s, _ in r["reactants"] + r["products"]}
        print(f"      SBML reaction species not in trajectory: "
              f"{len(rxn_sp - traj_set)}  (missing linkage)")
    else:
        print("      (no SBML provided - skipped)")

    # 4. Tier 2 hypotheses -------------------------------------------------
    print()
    print("  [4] " + hyp.summary().replace("\n", "\n  "))
    print("#" * 64)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[device] {device}")

    raw_counts, species_names = load_data()
    n_traj, T, S_full = raw_counts.shape
    print(f"[data] {raw_counts.shape}  (traj, time, species)  {T} steps")

    rng  = np.random.RandomState(SEED)
    perm = rng.permutation(n_traj)
    train_idx, test_idx = perm[:N_TRAIN_TRAJ], perm[N_TRAIN_TRAJ:]

    raw  = signed_log(raw_counts)
    dtr  = raw[train_idx]
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

    # input-data rules: SBML reaction network
    sbml = parse_sbml(SBML_PATH)
    elem_balances = element_balances(sbml, species_active, raw_counts_active)

    X       = torch.from_numpy(raw)
    train_X = X[train_idx].to(device)
    test_X  = X[test_idx].to(device)
    print(f"[data] train {tuple(train_X.shape)}  test {tuple(test_X.shape)}")

    persist_mse = float(F.mse_loss(test_X[:, :-1], test_X[:, 1:]))
    persist_r2  = r2(test_X[:, :-1], test_X[:, 1:])
    print(f"[diag] persistence: MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")

    # ── two-tier rule discovery ───────────────────────────────────────────
    print("[rules] discovering + validating (seed | SBML | trajectory) ...")
    ruleset, hyp = discover_rules(train_X.cpu(), test_X.cpu(),
                                  species_active, sbml)
    ruleset = ruleset.to(device)
    print(f"[rules] {ruleset.summary()}")

    # ── model ─────────────────────────────────────────────────────────────
    model = DynamicsModel(S, D_MODEL, N_LAYERS, N_HEADS, CONTEXT, DROPOUT,
                          species_type_ids, d_type=D_TYPE_EMBED).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    train_model(model, train_X, ruleset)

    # ── evaluate ──────────────────────────────────────────────────────────
    s_mse, s_r2 = one_step(model, test_X)
    roll_r2 = [r2(*full_rollout(model, test_X[k], ruleset))
               for k in range(test_X.shape[0])]
    mean_roll = sum(roll_r2) / len(roll_r2)

    print()
    print("=" * 64)
    print(f"  persistence 1-step    : MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")
    print(f"  model 1-step (test)   : MSE {s_mse:.5f}  R^2 {s_r2:.3f}"
          f"  {'(beats persistence)' if s_mse < persist_mse else '(still worse)'}")
    print(f"  model full rollout    : R^2 {mean_roll:.3f}  "
          f"(min {min(roll_r2):.3f}  max {max(roll_r2):.3f})  <- headline")
    print(f"  (v4 reference         : rollout R^2 ~0.55)")
    print("=" * 64)
    print()
    print(ruleset.summary())

    # ── missing-info report ───────────────────────────────────────────────
    analyze_gaps(model, test_X, species_active, species_type_ids,
                 ruleset, hyp, sbml, elem_balances)

    # ── generate + save ───────────────────────────────────────────────────
    gen_norm, _ = full_rollout(model, test_X[0], ruleset)
    full_seq    = torch.cat([test_X[0, :CONTEXT], gen_norm], 0).cpu().numpy()
    sl          = full_seq * span + lo
    gen_counts  = np.maximum(np.sign(sl) * np.expm1(np.abs(sl)), 0.0)
    print(f"\n[gen] 51st trajectory {gen_counts.shape}  "
          f"finite={np.isfinite(gen_counts).all()}  "
          f"count range [{gen_counts.min():.0f}, {gen_counts.max():.0f}]")
    np.save(f"{SAVE_DIR}/cell_traj_51_v6.npy", gen_counts)
    torch.save({
        "model": model.state_dict(),
        "lo": lo, "span": span, "active": active,
        "species_active": species_active,
        "species_type_ids": species_type_ids,
        "ruleset_mono_up":   ruleset.mono_up.cpu()   if ruleset.mono_up   is not None else None,
        "ruleset_mono_down": ruleset.mono_down.cpu() if ruleset.mono_down is not None else None,
        "ruleset_lo": ruleset.lo_bound.cpu() if ruleset.lo_bound is not None else None,
        "ruleset_hi": ruleset.hi_bound.cpu() if ruleset.hi_bound is not None else None,
        "hypotheses": hyp.items,
        "config": dict(S=S, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       context=CONTEXT, time_stride=TIME_STRIDE,
                       d_type=D_TYPE_EMBED, n_genes=len(locus_list)),
    }, f"{SAVE_DIR}/cell_emulator_v6.pt")
    print(f"[save] traj  -> {SAVE_DIR}/cell_traj_51_v6.npy")
    print(f"[save] model -> {SAVE_DIR}/cell_emulator_v6.pt")


if __name__ == "__main__":
    main()
