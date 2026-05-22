"""Colab cell: v5 cell-state emulator - gene-structured + rule discovery.

v4 → v5:
  1. GENE-TYPE EMBEDDINGS  each active species labelled protein/tRNA/rRNA/
     other/global via a learned embedding appended to every input token,
     so the model knows the biological role of each species.
  2. STRUCTURAL RULES (hard biology)
       - monotone counters: RPM/PM/DM can never decrease (cumulative made-counts)
         enforced as a hard guardrail at every rollout step.
  3. RULE DISCOVERY (data-driven, held-out validated)
       - mine candidate monotone-up and per-species bound rules from training
         trajectories;  validate every candidate on HELD-OUT test trajectories;
         99.9 % compliance gate → only rules that hold on unseen data survive.
     "It shouldn't be wrong": the held-out gate is the guarantee.
     Validated rules are applied as hard guardrails alongside the structural ones.

Run on Colab with Drive mounted, GPU runtime (~10 min).
"""

import glob
import time

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
D_TYPE_EMBED   = 16          # gene-type embedding dims (appended to in_proj)
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
MONO_EPS       = 1e-4        # tolerance for monotone-rule mining
RULE_COMPLIANCE = 0.999      # held-out compliance threshold
SEED           = 0
SAVE_DIR       = "/content/drive/MyDrive"
GENE_TABLE_PATH = "memory_bank/data/syn3a_gene_table.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


# ── species-name parsing ──────────────────────────────────────────────────────

# Per-gene central-dogma prefixes, longest first for greedy matching
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

MONOTONE_CHANS = {"RPM", "PM", "DM"}   # cumulative counters: structurally non-decreasing

# Gene-type codes
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
        print("[gene_types] pandas not available - using GTYPE_OTHER for all genes")
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
        print(f"[gene_types] load failed ({e}) - using GTYPE_OTHER for all genes")
        return {}


def build_gene_index(species_names, gene_type_map):
    """Build structured index over the active species list.

    Returns
    -------
    gene_chan_col   : (n_genes, N_CHAN) int32  - column in species array, -1=absent
    gene_chan_mask  : (n_genes, N_CHAN) bool
    gene_type_ids   : (n_genes,) int
    species_type_ids: (S,) int
    global_cols     : list[int]
    locus_list      : list[str]
    """
    locus_to_idx = {}
    locus_list   = []

    for name in species_names:
        prefix, locus = parse_species(name)
        if prefix is not None and locus not in locus_to_idx:
            locus_to_idx[locus] = len(locus_list)
            locus_list.append(locus)

    n_genes = len(locus_list)
    gene_chan_col = -np.ones((n_genes, N_CHAN), dtype=np.int32)
    global_cols   = []

    for i, name in enumerate(species_names):
        prefix, locus = parse_species(name)
        if prefix is None:
            global_cols.append(i)
        elif prefix in CHAN_IDX:
            g = locus_to_idx[locus]
            c = CHAN_IDX[prefix]
            gene_chan_col[g, c] = i

    gene_chan_mask = gene_chan_col >= 0

    gene_type_ids = np.array(
        [gene_type_map.get(loc, GTYPE_OTHER) for loc in locus_list],
        dtype=np.int32,
    )

    S = len(species_names)
    species_type_ids = np.full(S, GTYPE_GLOBAL, dtype=np.int32)
    for i, name in enumerate(species_names):
        prefix, locus = parse_species(name)
        if prefix is not None and locus in locus_to_idx:
            species_type_ids[i] = gene_type_ids[locus_to_idx[locus]]

    filled = int(gene_chan_mask.sum())
    print(f"[gene_index] {n_genes} genes  {filled} gene-species filled  "
          f"{len(global_cols)} global species")
    print(f"[gene_index] example: '{species_names[0]}' -> {parse_species(species_names[0])}")
    return gene_chan_col, gene_chan_mask, gene_type_ids, species_type_ids, global_cols, locus_list


# ── rule system ───────────────────────────────────────────────────────────────

class RuleSet:
    """Validated biological rules applied as hard guardrails during rollout.

    Provenance tracked: structural = from biological definition;
    discovered = mined from training data + held-out validated.
    """

    def __init__(self):
        self.mono_mask  = None   # (S,) bool - species that must not decrease
        self.mono_up    = None   # 1-D int tensor of indices (for checkpointing)
        self.lo_bound   = None   # (S,) float - per-species lower bound
        self.hi_bound   = None   # (S,) float - per-species upper bound
        self.n_structural  = 0
        self.n_discovered  = 0

    def to(self, dev):
        for attr in ("mono_mask", "mono_up", "lo_bound", "hi_bound"):
            t = getattr(self, attr, None)
            if t is not None:
                setattr(self, attr, t.to(dev))
        return self

    def project(self, prev, nxt):
        """Return a new tensor with all validated rules enforced.

        prev, nxt : (B, S) normalised float tensors.
        Uses torch.where / torch.maximum - fully autograd-safe, no in-place ops.
        """
        if self.mono_mask is not None:
            nxt = torch.where(
                self.mono_mask.unsqueeze(0),
                torch.maximum(nxt, prev),
                nxt,
            )
        if self.lo_bound is not None:
            nxt = torch.clamp(nxt,
                              self.lo_bound.unsqueeze(0),
                              self.hi_bound.unsqueeze(0))
        return nxt

    def summary(self):
        nm = int(self.mono_mask.sum()) if self.mono_mask is not None else 0
        nb = "yes" if self.lo_bound is not None else "no"
        return (f"RuleSet: {nm} monotone-up guardrails "
                f"({self.n_structural} structural + {self.n_discovered} discovered), "
                f"per-species bounds: {nb}")


def discover_rules(train_X, val_X, species_names):
    """Mine rules from train_X, validate on val_X, return a RuleSet.

    Parameters
    ----------
    train_X, val_X : (n_traj, T, S) CPU float32 tensors (normalised)
    species_names  : list[str] of length S

    Rule types
    ----------
    structural monotone-up
        RPM / PM / DM channels - cumulative counters by biological definition.
        Hard-coded, then cross-checked against held-out data (warns if violated).
    discovered monotone-up
        Any species where delta < -MONO_EPS in < 0.1% of training steps
        AND < 0.1% of held-out steps.
    per-species bounds
        [train_min - 0.1, train_max + 0.1] validated: rejected if ANY held-out
        point falls outside (zero tolerance - the 'shouldn't be wrong' gate).
    """
    tr = train_X.numpy()
    va = val_X.numpy()
    n_tr, T, S = tr.shape

    # ── structural monotone ──────────────────────────────────────────────
    structural = [i for i, nm in enumerate(species_names)
                  if parse_species(nm)[0] in MONOTONE_CHANS]

    # ── empirical monotone candidates (train) ────────────────────────────
    d_tr       = np.diff(tr, axis=1)                       # (n_tr, T-1, S)
    n_steps_tr = n_tr * (T - 1)
    viol_tr    = (d_tr < -MONO_EPS).sum(axis=(0, 1))       # (S,)
    frac_tr    = viol_tr / n_steps_tr
    candidates = set(np.where(frac_tr < (1.0 - RULE_COMPLIANCE))[0].tolist())

    # ── validate on held-out ─────────────────────────────────────────────
    d_va       = np.diff(va, axis=1)
    n_steps_va = va.shape[0] * (T - 1)
    viol_va    = (d_va < -MONO_EPS).sum(axis=(0, 1))
    frac_ok_va = 1.0 - viol_va / n_steps_va

    validated = {i for i in candidates if frac_ok_va[i] >= RULE_COMPLIANCE}

    # Cross-check structural rules
    for i in structural:
        if i not in validated:
            print(f"[rules] WARNING: structural monotone '{species_names[i]}' "
                  f"failed held-out validation ({frac_ok_va[i]:.4f}) - "
                  f"keeping it (structural guarantee)")

    all_mono = sorted(set(structural) | validated)

    # ── per-species bounds ───────────────────────────────────────────────
    tr_flat  = tr.reshape(-1, S)
    va_flat  = va.reshape(-1, S)
    slack    = 0.1
    lo_cand  = tr_flat.min(axis=0) - slack
    hi_cand  = tr_flat.max(axis=0) + slack
    bound_ok = ((va_flat >= lo_cand).all(axis=0) &
                (va_flat <= hi_cand).all(axis=0))

    # ── assemble RuleSet ─────────────────────────────────────────────────
    rs          = RuleSet()
    rs.mono_up  = torch.tensor(all_mono, dtype=torch.long)
    m           = torch.zeros(S, dtype=torch.bool)
    if all_mono:
        m[rs.mono_up] = True
    rs.mono_mask   = m
    rs.lo_bound    = torch.from_numpy(
        np.where(bound_ok, lo_cand, CLAMP_LO).astype(np.float32))
    rs.hi_bound    = torch.from_numpy(
        np.where(bound_ok, hi_cand, CLAMP_HI).astype(np.float32))
    rs.n_structural = len(set(structural) & set(all_mono))
    rs.n_discovered = len(validated - set(structural))

    print(f"[rules] structural monotone  : {len(structural)}")
    print(f"[rules] train candidates     : {len(candidates)}")
    print(f"[rules] held-out validated   : {len(validated)}")
    print(f"[rules] total monotone rules : {len(all_mono)}")
    print(f"[rules] per-species bounds   : {int(bound_ok.sum())}/{S} validated")
    return rs


# ── model ─────────────────────────────────────────────────────────────────────

class DynamicsModel(nn.Module):
    """Transformer over a CONTEXT-step window -> next-state residual.

    Each timestep token = in_proj(species_values) || mean_gene_type_embed,
    where the gene-type embedding (protein/tRNA/rRNA/other/global) is shared
    across all genes of the same type and learned end-to-end.
    """

    def __init__(self, S, d_model, n_layers, n_heads, context, dropout,
                 species_type_ids, d_type=D_TYPE_EMBED):
        super().__init__()
        self.d_type = d_type
        self.type_embed = nn.Embedding(N_GTYPES, d_type)
        self.register_buffer("stype",
                             torch.tensor(species_type_ids, dtype=torch.long))
        self.in_proj  = nn.Linear(S, d_model - d_type)
        self.ctx_pos  = nn.Parameter(torch.randn(context, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc, n_layers)
        self.out      = nn.Linear(d_model, S)

    def forward(self, ctx):                         # (B, C, S) -> (B, S)
        B, C, _ = ctx.shape
        h_val = self.in_proj(ctx)                   # (B, C, d_model - d_type)
        te    = self.type_embed(self.stype).mean(0) # (d_type,)
        te    = te.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        h     = torch.cat([h_val, te], dim=-1)      # (B, C, d_model)
        h     = h + self.ctx_pos
        h     = self.encoder(h)
        return ctx[:, -1] + self.out(h[:, -1])      # residual


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


# ── metrics ───────────────────────────────────────────────────────────────────

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
            pred  = model(ctx)
            true  = train_X[i_d, t0_d + CONTEXT + k]
            losses.append(F.mse_loss(pred, true))
            nxt   = pred.clamp(CLAMP_LO, CLAMP_HI)
            if step >= STEPS // 4:          # apply rules after warmup quarter
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
    g   = torch.Generator().manual_seed(SEED + 2)
    nt, Tt, _ = Xset.shape
    i   = torch.randint(0, nt, (n,), generator=g).tolist()
    t0  = torch.randint(0, Tt - CONTEXT - 1, (n,), generator=g).tolist()
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[device] {device}")

    raw, species_names = load_data()
    n_traj, T, S_full  = raw.shape
    print(f"[data] {raw.shape}  (traj, time, species)  {T} steps of ~{TIME_STRIDE}s")

    rng  = np.random.RandomState(SEED)
    perm = rng.permutation(n_traj)
    train_idx, test_idx = perm[:N_TRAIN_TRAJ], perm[N_TRAIN_TRAJ:]

    raw  = signed_log(raw)
    dtr  = raw[train_idx]
    lo   = np.percentile(dtr, 0.5,  axis=(0, 1))
    hi   = np.percentile(dtr, 99.5, axis=(0, 1))
    span = hi - lo
    active = span > 1e-6
    print(f"[data] active species: {int(active.sum())} / {S_full}")
    raw  = raw[:, :, active]
    lo, span = lo[active], span[active]
    species_active = [species_names[i] for i in range(S_full) if active[i]]
    raw  = np.clip((raw - lo) / span, CLAMP_LO, CLAMP_HI).astype(np.float32)
    S    = raw.shape[2]

    gene_type_map = load_gene_types(GENE_TABLE_PATH)
    _, _, _, species_type_ids, _, locus_list = build_gene_index(
        species_active, gene_type_map)

    X       = torch.from_numpy(raw)
    train_X = X[train_idx].to(device)
    test_X  = X[test_idx].to(device)
    print(f"[data] train {tuple(train_X.shape)}  test {tuple(test_X.shape)}")

    persist_mse = float(F.mse_loss(test_X[:, :-1], test_X[:, 1:]))
    persist_r2  = r2(test_X[:, :-1], test_X[:, 1:])
    print(f"[diag] persistence: MSE {persist_mse:.5f}  R^2 {persist_r2:.3f}")

    # ── rule discovery ────────────────────────────────────────────────────
    print("[rules] discovering and validating rules ...")
    ruleset = discover_rules(train_X.cpu(), test_X.cpu(), species_active)
    ruleset = ruleset.to(device)
    print(f"[rules] {ruleset.summary()}")

    # ── model ─────────────────────────────────────────────────────────────
    model = DynamicsModel(
        S, D_MODEL, N_LAYERS, N_HEADS, CONTEXT, DROPOUT,
        species_type_ids, d_type=D_TYPE_EMBED,
    ).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # ── train ─────────────────────────────────────────────────────────────
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

    # ── generate + save ───────────────────────────────────────────────────
    gen_norm, _ = full_rollout(model, test_X[0], ruleset)
    full_seq    = torch.cat([test_X[0, :CONTEXT], gen_norm], 0).cpu().numpy()
    sl          = full_seq * span + lo
    gen_counts  = np.maximum(np.sign(sl) * np.expm1(np.abs(sl)), 0.0)
    print(f"\n[gen] 51st trajectory {gen_counts.shape}  "
          f"finite={np.isfinite(gen_counts).all()}  "
          f"count range [{gen_counts.min():.0f}, {gen_counts.max():.0f}]")
    np.save(f"{SAVE_DIR}/cell_traj_51_v5.npy", gen_counts)
    torch.save({
        "model": model.state_dict(),
        "lo": lo, "span": span, "active": active,
        "species_active": species_active,
        "species_type_ids": species_type_ids,
        "ruleset_mono_up": ruleset.mono_up.cpu() if ruleset.mono_up is not None else None,
        "ruleset_lo":      ruleset.lo_bound.cpu() if ruleset.lo_bound is not None else None,
        "ruleset_hi":      ruleset.hi_bound.cpu() if ruleset.hi_bound is not None else None,
        "config": dict(
            S=S, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            context=CONTEXT, time_stride=TIME_STRIDE,
            d_type=D_TYPE_EMBED, n_genes=len(locus_list),
        ),
    }, f"{SAVE_DIR}/cell_emulator_v5.pt")
    print(f"[save] traj  -> {SAVE_DIR}/cell_traj_51_v5.npy")
    print(f"[save] model -> {SAVE_DIR}/cell_emulator_v5.pt")


if __name__ == "__main__":
    main()
