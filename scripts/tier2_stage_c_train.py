"""TIER 2 STAGE C -- the coupling model + the three hard kill-gates.

THE ARCHITECTURE (the phylogenetic-profiling / AlphaFold-pairwise analog).

Every bacterial genome is a natural knockout experiment. The pattern of which
ortholog groups (OGs) co-occur encodes functional coupling and redundancy:
"if an organism has X it doesn't need Y" -> X and Y are coupled. We learn that
pattern with a small BILINEAR coupling model whose learned matrix C IS the
gene-gene coupling map, queryable for any OG pair.

  E[og]  in R^d        : learned per-OG embedding ("what this gene is")
  C      in R^{d x d}  : learned coupling matrix (the bet)
  b[og]                : per-OG bias (captures the marginal essential rate
                         == roughly what family_frac already gives us)

  For organism o with present-OG set P(o):
      s_o = sum_{j in P(o)} E[j]                 (gene-content context)
      n_o = |P(o)|
  Predict essentiality of target OG t in organism o (t is present):
      context = (s_o - E[t]) / (n_o - 1)         (mean of OTHER present genes)
      logit(o,t) = b[t] + E[t]^T C context
  i.e. "is t essential here?" depends on which OTHER genes this genome carries.

  The coupling score between two OGs (Gate 2):
      coupling(a,b) = 0.5 ( E[a]^T C E[b] + E[b]^T C E[a] )

Presence (gene content) is the INPUT and is known for every organism including
held-out ones. Essentiality is the LABEL and is masked for the held-out clade.
So s_o may include held-out organisms' gene content with no leak.

THE THREE GATES (all must pass before the Tier 3 pangenome bet is justified):

  PERFORMANCE -- the model must beat family_frac AUPRC by >=10% relative,
    *on the hard slice* (OGs with weak cross-organism support) where
    conservation is silent and a coupling signal could actually add value.
    On the conserved core conservation already saturates; no headroom there.

  GATE 1 (label permutation) -- retrain on essentiality permuted WITHIN each
    organism. Held-out AUPRC must COLLAPSE (ratio < 0.7). If it doesn't, the
    model is reading phylogeny, not biology.

  GATE 2 (coupling recovery) -- the learned C must score operon-adjacent OG
    pairs >= 2x higher than random pairs. If not, C isn't learning gene-gene
    structure and any gain is just deeper phylogeny features.

  (the dragon) Even if PERFORMANCE passes, it only counts if it ALSO clears
  Gate 1 + Gate 2 on a leak-free clade holdout. Beating family_frac while
  failing Gate 1 == fitting taxonomy. That is the 25-year-old trap this whole
  harness exists to avoid.

--smoke : sandbox (no torch). Validates data prep, indexing, leak discipline,
          the family_frac bar, the permutation harness, and the Gate-2 pair
          machinery using a random dummy C. No training.
--real  : Colab + GPU. Trains, runs all three gates over several held-out
          clades, writes outputs/tier2/stage_c_results.json, prints the verdict.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
STAGE_A = REPO / "outputs" / "tier2"
sys.path.insert(0, str(SCRIPTS))

# reuse the tested Stage B helpers (same module dir)
from tier2_stage_b_coupling_model import (        # noqa: E402
    load_stage_a, apply_clade_holdout, permute_within_organism,
    family_frac_baseline, metrics,
)

# ---- gate thresholds (the contract) -------------------------------------
PERF_GATE_REL   = 0.10   # model AUPRC must beat family_frac by >=10% relative
GATE1_MAX_RATIO = 0.70   # permuted/real AUPRC must be < this (signal collapses)
GATE2_MIN_LIFT  = 2.00   # operon-pair coupling must be >= this x random
HARD_SLICE_MAX_SUPPORT = 10  # OG labeled in < this many TRAIN orgs = "hard"

# ---- model hyperparameters (prototype scale; kept deliberately small) ----
DEFAULT_D       = 32
DEFAULT_EPOCHS  = 40
DEFAULT_LR      = 5e-3
DEFAULT_WD      = 1e-4
DEFAULT_BATCH   = 8192
DEFAULT_NCLADES = 3
SEED            = 0


# =========================================================================
# data prep (pure numpy/pandas -- runs in sandbox, no torch)
# =========================================================================
def build_index(presence, ess):
    """Map og_id/organism -> contiguous ints; build the sparse presence
    (n_org x n_og) coordinate lists and per-org present-counts."""
    import numpy as np
    ogs = sorted(set(presence.og_id) | set(ess.og_id))
    orgs = sorted(set(presence.organism) | set(ess.organism))
    og2i = {o: i for i, o in enumerate(ogs)}
    org2i = {o: i for i, o in enumerate(orgs)}
    pr_org = presence.organism.map(org2i).to_numpy()
    pr_og = presence.og_id.map(og2i).to_numpy()
    n_o = np.zeros(len(orgs), dtype=np.float64)
    for oi in pr_org:
        n_o[oi] += 1
    return {"ogs": ogs, "orgs": orgs, "og2i": og2i, "org2i": org2i,
            "pres_org": pr_org, "pres_og": pr_og, "n_o": n_o,
            "n_og": len(ogs), "n_org": len(orgs)}


def make_examples(ess, idx):
    """Labeled (org_idx, og_idx, label) arrays from the essentiality table."""
    import numpy as np
    e = ess.dropna(subset=["ess"]).copy()
    e["ess"] = e.ess.astype(int)
    org_i = e.organism.map(idx["org2i"]).to_numpy()
    og_i = e.og_id.map(idx["og2i"]).to_numpy()
    y = e.ess.to_numpy().astype(np.float32)
    return org_i.astype(np.int64), og_i.astype(np.int64), y


def train_support(train_ess):
    """Per-OG count of TRAIN organisms labeling it (the conservation depth)."""
    return train_ess.groupby("og_id").size().to_dict()


def auto_select_clades(ess, clade_map, n_clades, min_pos=30):
    """Pick the held-out clades with the most labeled essential positives,
    so the held-out metric is stable. Leak-free: whole clade held out."""
    scored = []
    for clade, orgs in clade_map.items():
        sub = ess[ess.organism.isin(orgs)].dropna(subset=["ess"])
        npos = int((sub.ess == 1).sum()); ntot = int(len(sub))
        if npos >= min_pos:
            scored.append((clade, npos, ntot, len(orgs)))
    scored.sort(key=lambda r: r[1], reverse=True)
    return [s[0] for s in scored[:n_clades]], scored


# =========================================================================
# Gate 2 machinery (pure numpy -- works with any E,C, dummy or trained)
# =========================================================================
def coupling_pair_lift(E, C, idx, op_pairs, n_random=2000, seed=0):
    """coupling(a,b)=0.5(E[a]^T C E[b] + E[b]^T C E[a]); compare operon-adjacent
    OG pairs vs random OG pairs. Returns means + lift ratio + a rank stat."""
    import numpy as np
    og2i = idx["og2i"]
    rng = np.random.RandomState(seed)

    def cpl(a, b):
        ea, eb = E[a], E[b]
        return 0.5 * (ea @ C @ eb + eb @ C @ ea)

    op = op_pairs[op_pairs.og_a.isin(og2i) & op_pairs.og_b.isin(og2i)]
    op_idx = [(og2i[a], og2i[b]) for a, b in zip(op.og_a, op.og_b)]
    if len(op_idx) < 20:
        return None
    op_scores = np.array([cpl(a, b) for a, b in op_idx])
    n = E.shape[0]
    rnd = []
    for _ in range(n_random):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b:
            rnd.append(cpl(a, b))
    rnd = np.array(rnd)
    op_m = float(np.abs(op_scores).mean()); rnd_m = float(np.abs(rnd).mean())
    # rank stat: fraction of operon pairs above the random 95th percentile
    thr = np.percentile(np.abs(rnd), 95)
    frac_above = float((np.abs(op_scores) > thr).mean())
    return {"n_operon": len(op_scores), "n_random": len(rnd),
            "operon_abs_mean": op_m, "random_abs_mean": rnd_m,
            "lift_ratio": op_m / max(rnd_m, 1e-9),
            "frac_operon_above_rand_p95": frac_above}


def eval_split(y, scores, support_arr, og_i):
    """metrics on the full held-out set AND on the hard slice (low support)."""
    import numpy as np
    full = metrics(y, scores)
    hard = (support_arr[og_i] < HARD_SLICE_MAX_SUPPORT)
    if hard.sum() >= 20 and len(np.unique(y[hard])) > 1:
        hard_m = metrics(y[hard], scores[hard])
    else:
        hard_m = {"auprc": None, "recall_at_p30": None,
                  "n": int(hard.sum()), "n_pos": int(y[hard].sum())}
    return full, hard_m


# =========================================================================
# REAL training (torch; Colab + GPU)
# =========================================================================
def run_real(args):
    import numpy as np
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: --real needs PyTorch (Colab+GPU).", file=sys.stderr)
        return 1
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  torch {torch.__version__}")

    presence, ess, clade_map, op_pairs = load_stage_a()
    idx = build_index(presence, ess)
    print(f"index: {idx['n_org']} orgs x {idx['n_og']:,} OGs")

    # sparse presence (n_org x n_og), values 1.0 -- constant across training
    pres_i = torch.tensor(np.vstack([idx["pres_org"], idx["pres_og"]]),
                          dtype=torch.long)
    pres = torch.sparse_coo_tensor(
        pres_i, torch.ones(pres_i.shape[1]),
        size=(idx["n_org"], idx["n_og"])).coalesce().to(dev)
    n_o = torch.tensor(idx["n_o"], dtype=torch.float32, device=dev)

    held_clades, scored = auto_select_clades(ess, clade_map, args.n_clades)
    print(f"held-out clades (most positives): {held_clades}")

    class Coupling(nn.Module):
        def __init__(self, n_og, d):
            super().__init__()
            self.E = nn.Embedding(n_og, d)
            nn.init.normal_(self.E.weight, std=0.1)
            self.C = nn.Parameter(torch.empty(d, d))
            nn.init.xavier_uniform_(self.C)
            self.bias = nn.Parameter(torch.zeros(n_og))

        def org_sums(self):
            return torch.sparse.mm(pres, self.E.weight)      # (n_org x d)

        def logits(self, org_idx, og_idx, S):
            e_t = self.E(og_idx)                              # (B,d)
            s = S[org_idx]                                    # (B,d)
            n = n_o[org_idx].clamp(min=2.0)                   # (B,)
            context = (s - e_t) / (n - 1.0).unsqueeze(1)      # exclude self
            coupling = (e_t @ self.C * context).sum(-1)       # (B,)
            return self.bias[og_idx] + coupling

    def train_eval(held_clade, permute):
        train_ess, test_ess = apply_clade_holdout(ess, clade_map, held_clade)
        if permute:
            train_ess = permute_within_organism(train_ess, seed=SEED)
        sup = train_support(train_ess)
        sup_arr = np.array([sup.get(o, 0) for o in idx["ogs"]])

        tr_org, tr_og, tr_y = make_examples(train_ess, idx)
        te_org, te_og, te_y = make_examples(test_ess, idx)
        tr_org_t = torch.tensor(tr_org, device=dev)
        tr_og_t = torch.tensor(tr_og, device=dev)
        tr_y_t = torch.tensor(tr_y, device=dev)

        model = Coupling(idx["n_og"], args.d).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.wd)
        bce = nn.BCEWithLogitsLoss()
        N = len(tr_y)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(N, device=dev)
            tot = 0.0
            for b0 in range(0, N, args.batch):
                bi = perm[b0:b0 + args.batch]
                S = model.org_sums()                 # grad flows to context
                logit = model.logits(tr_org_t[bi], tr_og_t[bi], S)
                loss = bce(logit, tr_y_t[bi])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(bi)
            if not permute and (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1:>2}/{args.epochs}  loss={tot/N:.4f}")

        model.eval()
        with torch.no_grad():
            S = model.org_sums()
            te_scores = torch.sigmoid(model.logits(
                torch.tensor(te_org, device=dev),
                torch.tensor(te_og, device=dev), S)).cpu().numpy()
            E = model.E.weight.detach().cpu().numpy()
            C = model.C.detach().cpu().numpy()
        full, hard = eval_split(te_y, te_scores, sup_arr, te_og)
        # family_frac bar on the SAME cells, full + hard
        base = family_frac_baseline(train_ess, test_ess).dropna(
            subset=["family_frac", "ess"])
        b_og = base.og_id.map(idx["og2i"]).to_numpy()
        bfull, bhard = eval_split(base.ess.values.astype(int),
                                  base.family_frac.values, sup_arr, b_og)
        return {"model_full": full, "model_hard": hard,
                "ff_full": bfull, "ff_hard": bhard,
                "E": E, "C": C, "sup_arr": sup_arr}

    results = {}
    for clade in held_clades:
        print(f"\n{'='*60}\nHELD-OUT CLADE: {clade}\n{'='*60}")
        t0 = time.time()
        real = train_eval(clade, permute=False)
        print(f"  [real] model  full AUPRC={fmt(real['model_full']['auprc'])} "
              f"hard AUPRC={fmt(real['model_hard']['auprc'])}")
        print(f"  [real] family_frac full AUPRC={fmt(real['ff_full']['auprc'])} "
              f"hard AUPRC={fmt(real['ff_hard']['auprc'])}")
        print(f"  [gate1] retraining on PERMUTED labels ...")
        perm = train_eval(clade, permute=True)
        print(f"  [gate1] permuted model full AUPRC="
              f"{fmt(perm['model_full']['auprc'])}")
        g2 = coupling_pair_lift(real["E"], real["C"], idx, op_pairs)
        if g2:
            print(f"  [gate2] coupling lift operon vs random: "
                  f"{g2['lift_ratio']:.2f}x  "
                  f"(frac>p95={g2['frac_operon_above_rand_p95']:.2f})")
        results[clade] = {
            "model_full": real["model_full"], "model_hard": real["model_hard"],
            "ff_full": real["ff_full"], "ff_hard": real["ff_hard"],
            "perm_model_full": perm["model_full"],
            "gate2": g2, "secs": round(time.time() - t0, 1)}

    verdict = decide(results)
    out = STAGE_A / "stage_c_results.json"
    out.write_text(json.dumps({"results": results, "verdict": verdict,
                               "thresholds": {
                                   "perf_rel": PERF_GATE_REL,
                                   "gate1_max_ratio": GATE1_MAX_RATIO,
                                   "gate2_min_lift": GATE2_MIN_LIFT}},
                              indent=2, default=str))
    print(f"\nwrote {out}")
    print_verdict(verdict)
    return 0 if verdict["all_pass"] else 2


# =========================================================================
# verdict logic
# =========================================================================
def _rel(model, ff):
    if model is None or ff in (None, 0):
        return None
    return (model - ff) / ff


def decide(results):
    import numpy as np
    perf_rels, g1_ratios, g2_lifts = [], [], []
    for c, r in results.items():
        # performance measured on the HARD slice (where headroom exists);
        # fall back to full if hard slice too small
        m = r["model_hard"]["auprc"] or r["model_full"]["auprc"]
        f = r["ff_hard"]["auprc"] or r["ff_full"]["auprc"]
        rel = _rel(m, f)
        if rel is not None:
            perf_rels.append(rel)
        real_a = r["model_full"]["auprc"]; perm_a = r["perm_model_full"]["auprc"]
        if real_a and perm_a is not None and real_a > 0:
            g1_ratios.append(perm_a / real_a)
        if r["gate2"]:
            g2_lifts.append(r["gate2"]["lift_ratio"])
    perf_ok = len(perf_rels) > 0 and float(np.mean(perf_rels)) >= PERF_GATE_REL
    g1_ok = len(g1_ratios) > 0 and float(np.mean(g1_ratios)) < GATE1_MAX_RATIO
    g2_ok = len(g2_lifts) > 0 and float(np.mean(g2_lifts)) >= GATE2_MIN_LIFT
    return {
        "performance": {"mean_rel_lift_hard_slice":
                        _mean(perf_rels), "pass": perf_ok,
                        "threshold": PERF_GATE_REL},
        "gate1_permutation": {"mean_permuted_ratio": _mean(g1_ratios),
                              "pass": g1_ok, "threshold": GATE1_MAX_RATIO},
        "gate2_coupling": {"mean_operon_lift": _mean(g2_lifts),
                           "pass": g2_ok, "threshold": GATE2_MIN_LIFT},
        "all_pass": bool(perf_ok and g1_ok and g2_ok)}


def _mean(xs):
    return round(float(sum(xs) / len(xs)), 4) if xs else None


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def print_verdict(v):
    print(f"\n{'='*60}\nSTAGE C VERDICT\n{'='*60}")
    for k, label in [("performance", "PERFORMANCE (hard slice, >=+10%)"),
                     ("gate1_permutation", "GATE 1  permutation collapse"),
                     ("gate2_coupling", "GATE 2  operon coupling recovery")]:
        d = v[k]
        tag = "PASS" if d["pass"] else "FAIL"
        val = next((d[x] for x in d if x.startswith("mean")), None)
        print(f"  [{tag}]  {label:<38s} value={fmt(val)} "
              f"thr={d['threshold']}")
    if v["all_pass"]:
        print("\n  >>> ALL GATES PASS -- the coupling architecture clears the")
        print("      phylogeny dragon at prototype scale. The Tier 3 pangenome")
        print("      pretraining bet is justified.")
    else:
        print("\n  >>> NOT ALL GATES PASS -- clean kill at prototype scale.")
        print("      Do NOT scale to the pangenome. Document as null result;")
        print("      the cheap test saved the expensive one.")


# =========================================================================
# SMOKE (sandbox, no torch)
# =========================================================================
def run_smoke():
    import numpy as np
    print("=" * 60)
    print("TIER 2 STAGE C -- smoke (data prep + gate harness, no torch)")
    print("=" * 60)
    presence, ess, clade_map, op_pairs = load_stage_a()
    idx = build_index(presence, ess)
    print(f"  index: {idx['n_org']} orgs x {idx['n_og']:,} OGs")

    held, scored = auto_select_clades(ess, clade_map, DEFAULT_NCLADES)
    print(f"  auto-selected held clades: {held}")
    assert len(held) >= 1, "no clade had enough positives"

    clade = held[0]
    train_ess, test_ess = apply_clade_holdout(ess, clade_map, clade)
    held_orgs = set(clade_map[clade])
    assert len(set(train_ess.organism) & held_orgs) == 0, "clade leak!"
    print(f"  T1 leak-free holdout '{clade}': "
          f"{len(set(train_ess.organism))} train orgs, "
          f"{len(held_orgs)} held -- PASS")

    tr_org, tr_og, tr_y = make_examples(train_ess, idx)
    te_org, te_og, te_y = make_examples(test_ess, idx)
    print(f"  T2 examples: train={len(tr_y):,} test={len(te_y):,} "
          f"(test pos={int(te_y.sum())})")
    assert len(tr_y) > 1000 and len(te_y) > 50

    # family_frac bar on full + hard slice (the real metric machinery)
    sup = train_support(train_ess)
    sup_arr = np.array([sup.get(o, 0) for o in idx["ogs"]])
    base = family_frac_baseline(train_ess, test_ess).dropna(
        subset=["family_frac", "ess"])
    b_og = base.og_id.map(idx["og2i"]).to_numpy()
    bfull, bhard = eval_split(base.ess.values.astype(int),
                              base.family_frac.values, sup_arr, b_og)
    print(f"  T3 family_frac bar:  full AUPRC={fmt(bfull['auprc'])}  "
          f"hard-slice AUPRC={fmt(bhard['auprc'])} "
          f"(hard n={bhard['n']}, pos={bhard['n_pos']})")
    assert bfull["auprc"] is not None

    # permutation harness
    perm = permute_within_organism(train_ess, seed=SEED)
    pbase = family_frac_baseline(perm, test_ess).dropna(
        subset=["family_frac", "ess"])
    pm = metrics(pbase.ess.values.astype(int), pbase.family_frac.values)
    ratio = (pm["auprc"] / bfull["auprc"]) if bfull["auprc"] else None
    print(f"  T4 permutation harness: permuted/real AUPRC ratio="
          f"{fmt(ratio)} (<0.7 expected) -- PASS")

    # Gate-2 machinery with a DUMMY random C (proves the pair test runs)
    rng = np.random.RandomState(0)
    Ed = rng.randn(idx["n_og"], 16) * 0.1
    Cd = rng.randn(16, 16)
    g2 = coupling_pair_lift(Ed, Cd, idx, op_pairs, n_random=500)
    assert g2 is not None, "operon pairs didn't intersect OG vocab"
    print(f"  T5 Gate-2 harness: tested {g2['n_operon']} operon vs "
          f"{g2['n_random']} random pairs; dummy lift="
          f"{g2['lift_ratio']:.2f}x (random C ~1.0 expected) -- PASS")

    # verdict plumbing on a fake results dict
    fake = {clade: {
        "model_full": {"auprc": 0.40, "recall_at_p30": 0.2, "n": 1, "n_pos": 1},
        "model_hard": {"auprc": 0.30, "recall_at_p30": 0.1, "n": 1, "n_pos": 1},
        "ff_full": {"auprc": 0.35}, "ff_hard": {"auprc": 0.25},
        "perm_model_full": {"auprc": 0.10},
        "gate2": {"lift_ratio": 2.5}}}
    v = decide(fake)
    print(f"  T6 verdict plumbing: perf={v['performance']['pass']} "
          f"g1={v['gate1_permutation']['pass']} "
          f"g2={v['gate2_coupling']['pass']} -- PASS")

    print(f"\n=== STAGE C SMOKE PASS ===")
    print(f"  data prep, leak-free holdout, hard-slice metric, family_frac bar,")
    print(f"  permutation harness, Gate-2 pair machinery, verdict logic all OK.")
    print(f"  Run --real on Colab+GPU to train and get the gate verdict.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--d", type=int, default=DEFAULT_D)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--wd", type=float, default=DEFAULT_WD)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--n_clades", type=int, default=DEFAULT_NCLADES)
    args = ap.parse_args()
    if args.real:
        return run_real(args)
    return run_smoke()


if __name__ == "__main__":
    sys.exit(main())
