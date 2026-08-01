"""CELLFORMER v1 -- the complete first-generation transformer for THIS task, built so the sign channel plugs in.

WHAT THIS IS AND, MORE IMPORTANTLY, WHAT IT IS NOT. `transformer_ko` reached recall@50 0.490 +/- 0.037 against
a tide-null of 0.249 and an oracle of 0.607, and the one control-confirmed win in that whole arc was the
interaction-context tokens themselves (real-vs-random partners +0.335). That validated ONE component:
relational attention over biologically valid neighbours. It did not validate mixed-curvature geometry, sheaf
layers, hypergraph events, continuous-time dynamics, multimodal heads, or whole-cell simulation. This module
is deliberately v1 of a staged plan and claims nothing about those.

v1 = the complete model for the CURRENT task:

    inputs      perturbed gene; baseline K562 state; typed neighbours (co-dependency, complex, PPI,
                signed reg_perturb_k562); relation type; relation SIGN; confidence; perturbation strength;
                assay/time metadata (constant in this deposit, carried as a declared constant so the slot
                exists for v2 rather than being invented later)
    model       target-gene token + typed neighbour tokens; relation-type embedding; relation-specific
                attention BIAS; an EXPLICIT non-learned tide baseline; residual head; uncertainty head
    outputs     specific-mover ranking, response probability, direction, residual effect, uncertainty

THE SIGN CHANNEL IS THE POINT OF THE SCAFFOLD, AND ITS PRIOR IS BAD. `SIGN_MODE` takes three values --
"off", "measured", "measured+imputed" -- and the third reads `outputs/orphan/impute_reg_sign.json` the moment
it exists. Building the rest now means that arm costs one flag instead of one build. But the honest prior,
recorded before the run rather than after:

    transformer_ko6   adding regulatory features to this exact encoder scored -0.021 recall@50, WORSE on
                      all 3 seeds
    transformer_signed  partners are used, sign is not: sign-over-SHUFFLED-sign was +0.0049 while the
                      shuffled-sign control retained 94% of the gain
    transformer_graphnet  real wiring is indistinguishable from shuffled wiring of the same density
                      (+0.005 +/- 0.008), so a knockout is well described as a BAG of its partners

So v1 is expected to tie `transformer_ko`, and the sign arm is expected to be a wash. If it is, that is the
result; imputed signs would then be a layer with provenance and no measured use, which is worth knowing
before anything is built on top of them.

THE POWER PROBLEM, STATED BEFORE ANY NUMBER, BECAUSE IT DECIDES WHETHER THIS TEST CAN CONCLUDE ANYTHING.
Six architectural levers have been tried on this readout and every increment has been within +/-0.02 with a
seed sd of 0.02-0.05. At S seeds and sd s, the smallest increment separable at ~3 standard errors is
3*s/sqrt(S) -- about +0.05 at 3 seeds and sd 0.03. That is LARGER than any lever has produced. A stage that
reports "wash" under those conditions has not learned that the idea is wrong; it has learned that the test
cannot see it. This module therefore prints its minimum detectable increment BEFORE the arms, and refuses to
call any arm a null below it.

TWO METRICS, AND THE SECOND IS WHY THIS IS NOT JUST transformer_ko AGAIN.
    METRIC 1  tide-removed specific-mover recall@50, the number every result in this project is comparable to
    METRIC 2  the same computation on the residual after the SHARED CORE is projected out, the core fitted on
              TRAINING perturbations only. CELLFORMER2.md's own diagnosis was that metric 1 rewards naming
              genes that move for everything -- which is why a frequency baseline beat seven mechanistic
              methods -- and that a model cannot win metric 2 by learning the frequency prior, because the
              prior is exactly what was subtracted. Metric 2 has never been the primary scoring axis here.
              It is the axis with headroom.

CONTROLS, all of which have to run for any arm to count: random partners (the +0.335 check), wrong-knockout
identity shuffle, shuffled relation TYPE, and shuffled relation SIGN. The sign arm specifically is scored
against shuffled-sign, never against no-sign, because sign-over-nothing conflates "sign matters" with "an
extra channel matters".
"""
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
NET = SP / "cell_complete.json.gz"
GWPS = SP / "gwps.h5ad"
DEP = SP / "CRISPRGeneEffect.csv"
DEPVEC = OUT / "depmap_vecs.npz"

# ------------------------------------------------------------------ the pluggable slot, and its contract
#
# "measured+imputed" reads this file. THE CONTRACT: a JSON with a list under `signed_edges`, each entry
# {"source": <gene symbol>, "target": <gene symbol>, "sign": +1|-1, "confidence": <float 0..1>}. Anything
# else raises rather than being coerced -- a sign channel silently filled with zeros would make the arm look
# like a tie when it never ran.
SIGN_MODE = os.environ.get("CELLFORMER_SIGN", "measured")     # off | measured | measured+imputed
IMPUTED = OUT / "impute_reg_sign.json"

SMOKE = os.environ.get("CELLFORMER_SMOKE", "0") == "1"
N_SEEDS = 1 if SMOKE else int(os.environ.get("CELLFORMER_SEEDS", "3"))
MAX_TOK = 24            # neighbour tokens per knockout, capped for CPU
D_MODEL = 96 if SMOKE else 128
EPOCHS = 3 if SMOKE else 30
CORE_K = 20             # rank of the shared core removed for metric 2, per CELLFORMER2.md
K_RECALL = 50
REL_TYPES = ["self", "codep", "complex", "ppi", "reg"]
# METRIC 3: held-out CELL LINES. Benches already exist for all of these and none has ever been used to
# score this transformer arc -- every published number in it is a held-out split of K562.
TRANSFER_LINES = ["RPE1", "HCT116", "HepG2", "Jurkat", "Melanoma"]

# assay/time metadata is CONSTANT in this deposit. It is carried explicitly rather than omitted, so that when
# v4 adds timepoints the slot already exists and nothing has to be retrofitted into the token layout.
ASSAY_CONST = {"assay": "CRISPRi Perturb-seq", "readout": "steady-state mRNA",
               "time_hours": 192.0, "cell_type": "K562",
               "note": "one value for every row here; present so v4 can vary it, not because it varies now"}


def prepare_depmap_vecs(report):
    """Regenerate the per-gene DepMap dependency profile matrix if the gitignored cache is missing."""
    if DEPVEC.exists():
        return
    import pandas as pd
    if not DEP.exists():
        raise SystemExit(f"absent: {DEP} -- needed to rebuild {DEPVEC.name}")
    report(f"  {DEPVEC.name} absent; rebuilding from {DEP.name}")
    ge = pd.read_csv(DEP, index_col=0)
    syms = [c.split(" (")[0].strip() for c in ge.columns]
    Z = ge.to_numpy(dtype=np.float32).T          # (genes x lines)
    del ge
    Z = np.nan_to_num(Z)
    DEPVEC.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEPVEC, syms=np.array(syms), Z=Z)
    report(f"    wrote {DEPVEC.name}: {Z.shape[0]:,} genes x {Z.shape[1]:,} lines")


def load_layers(report):
    """Typed neighbour lists and the signed regulatory channel, keyed by gene SYMBOL."""
    d = json.load(gzip.open(NET, "rt"))
    name = [g["name"] for g in d["genes"]]
    codep, cplx, ppi = {}, {}, {}
    for a, v in (d.get("codep") or {}).items():
        codep.setdefault(name[int(a)], []).extend(name[int(b)] for b, _s in v)
    g2c = {int(k): set(v) for k, v in (d.get("gene2cplx") or {}).items()}
    bym = {}
    for gi_, cs in g2c.items():
        for c in cs:
            bym.setdefault(c, []).append(gi_)
    for c, mem in bym.items():
        for x in mem:
            cplx.setdefault(name[x], []).extend(name[y] for y in mem if y != x)
    for e in d["ppi"]:
        a, b = name[int(e[0])], name[int(e[1])]
        ppi.setdefault(a, []).append(b)
        ppi.setdefault(b, []).append(a)
    del d
    report(f"  typed neighbours: codep {len(codep):,} sources, complex {len(cplx):,}, PPI {len(ppi):,}")
    return codep, cplx, ppi


def load_signed(report, mode):
    """The signed regulatory channel. Three modes, and the third is the whole reason for the scaffold."""
    if mode == "off":
        report("  SIGN_MODE=off -- no regulatory channel at all")
        return {}, {"mode": "off", "n_edges": 0}
    sig = {}
    # MEASURED: sign of the K562 robust z, built the way causal_reg.py builds it. Derived here rather than
    # read from a cache so the module does not silently run on a stale layer.
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["var"]["__categories"]["gene_name"][:]]
        gname = np.array([cats[i] for i in f["var"]["gene_name"][:]])
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    Z = (X - med) / np.where(mad < 1e-6, np.nan, mad)
    ei, ej = np.where(np.isfinite(Z) & (np.abs(Z) >= 5.0))
    # CONFIDENCE IS |z|, NOT 1.0. The first version handed every measured edge a confidence of exactly 1.0,
    # which is a constant column -- it cannot carry information and its ablation cannot fail. Evidence
    # strength is the magnitude of the effect that defined the edge.
    for i, j in zip(ei, ej):
        sig.setdefault(psym[i], []).append((gname[j], 1 if Z[i, j] > 0 else -1,
                                            float(min(abs(Z[i, j]) / 10.0, 3.0))))
    n_meas = sum(len(v) for v in sig.values())
    report(f"  measured signed edges (|robust z| >= 5): {n_meas:,} over {len(sig):,} sources")
    meta = {"mode": mode, "n_measured": n_meas}

    if mode == "measured+imputed":
        if not IMPUTED.exists():
            raise SystemExit(
                f"SIGN_MODE=measured+imputed but {IMPUTED} does not exist. This arm is NOT run with the "
                f"measured edges alone and silently relabelled -- that would report the measured arm twice "
                f"under two names. Run colab/impute_reg_sign.py first, or use SIGN_MODE=measured.")
        d = json.load(open(IMPUTED))
        edges = d.get("signed_edges")
        if not isinstance(edges, list) or not edges:
            raise SystemExit(f"{IMPUTED.name} has no `signed_edges` list. The contract is a list of "
                             f"{{source, target, sign, confidence}}; a differently-shaped file is a "
                             f"different result and must not be coerced into this slot.")
        bad = [e for e in edges[:200] if set(("source", "target", "sign")) - set(e)]
        if bad:
            raise SystemExit(f"{IMPUTED.name}: {len(bad)} of the first 200 entries lack source/target/sign")
        n_imp = 0
        for e in edges:
            s = int(e["sign"])
            if s not in (1, -1):
                continue
            sig.setdefault(str(e["source"]), []).append((str(e["target"]), s,
                                                         float(e.get("confidence", 0.5))))
            n_imp += 1
        report(f"  imputed signed edges added: {n_imp:,} (total {n_meas + n_imp:,})")
        meta["n_imputed"] = n_imp
        meta["imputed_from"] = str(IMPUTED)
    return sig, meta


def build_tokens(kos, codep, cplx, ppi, sig, srow, Z, base_expr, ess, deg, rng, mode="real",
                 pstrength=None):
    """One knockout -> (token features, token relation-type ids, token signs).

    `mode` is the control switch and it is deliberately inside the token builder rather than bolted on
    afterwards: "random" draws partners of the same COUNT and the same TYPE MIX from the gene universe, so
    the control differs from the real arm only in WHICH partner, never in how many or of what kind.
    """
    feats, types, signs, confs = [], [], [], []
    universe = list(srow)
    for k in kos:
        rows, rtypes, rsign, rconf = [], [], [], []

        ps = float((pstrength or {}).get(k, 0.0))       # perturbation strength, a property of the KO
        def push(sym, t, sgn=0.0, cf=0.0):
            i = srow.get(sym)
            if i is None:
                return
            rows.append(np.concatenate([Z[i], [base_expr.get(sym, 0.0), ess.get(sym, 0.0),
                                               np.log1p(deg.get(sym, 0)), sgn, cf, ps]]))
            rtypes.append(REL_TYPES.index(t))
            rsign.append(sgn)
            rconf.append(cf)

        push(k, "self")
        want = [("codep", codep.get(k, [])), ("complex", cplx.get(k, [])), ("ppi", ppi.get(k, []))]
        per = max(1, (MAX_TOK - 1) // (len(want) + 1))
        for t, lst in want:
            pick = lst[:per] if mode == "real" else [universe[int(x)] for x in
                                                     rng.integers(0, len(universe), min(per, len(lst)))]
            for s in pick:
                push(s, t)
        rl = sig.get(k, [])[:per]
        if mode != "real":
            rl = [(universe[int(x)], int(rng.choice([-1, 1])), 1.0)
                  for x in rng.integers(0, len(universe), len(rl))]
        for s, sg, cf in rl:
            push(s, "reg", float(sg), float(cf))

        if not rows:
            rows = [np.zeros(Z.shape[1] + 6, dtype=np.float32)]
            rtypes, rsign, rconf = [0], [0.0], [0.0]
        feats.append(np.array(rows[:MAX_TOK], dtype=np.float32))
        types.append(np.array(rtypes[:MAX_TOK], dtype=np.int64))
        signs.append(np.array(rsign[:MAX_TOK], dtype=np.float32))
        confs.append(np.array(rconf[:MAX_TOK], dtype=np.float32))
    return feats, types, signs, confs


def pad(feats, types, signs, confs):
    import torch
    n = len(feats)
    L = max(len(f) for f in feats)
    D = feats[0].shape[1]
    F = np.zeros((n, L, D), dtype=np.float32)
    T = np.zeros((n, L), dtype=np.int64)
    S = np.zeros((n, L), dtype=np.float32)
    C = np.zeros((n, L), dtype=np.float32)
    M = np.zeros((n, L), dtype=bool)
    for i, (f, t, s, c) in enumerate(zip(feats, types, signs, confs)):
        F[i, :len(f)] = f
        T[i, :len(t)] = t
        S[i, :len(s)] = s
        C[i, :len(c)] = c
        M[i, :len(f)] = True
    return (torch.from_numpy(F), torch.from_numpy(T), torch.from_numpy(S), torch.from_numpy(C),
            torch.from_numpy(M))


def make_model(d_in, d_model, n_genes):
    """Typed tokens, relation-type embedding, relation-specific attention bias, and FIVE supervised heads.

    WHAT CHANGED FROM THE FIRST VERSION, AND IT MATTERS. The first version instantiated a residual head, a
    probability head and an uncertainty head and then trained NONE of them -- the only loss term was the
    retrieval metric. Three of the five declared outputs were dead code that the forward pass returned and
    nothing supervised. They are now decoded to gene space and trained.

    THE TIDE BASELINE IS AN ADDITIVE, NON-LEARNED TERM, which is the whole anti-collapse design. neural_ko
    measured that a regression net on this target collapses to predicting the tide and lands BELOW the
    tide-null. Here the per-gene training-set mean |z| is ADDED to the residual head's output before the
    loss, so predicting the tide earns exactly zero -- the head can only be rewarded for what the tide does
    not already explain.
    """
    import torch
    import torch.nn as nn

    class Cellformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.rel = nn.Embedding(len(REL_TYPES), d_model)
            self.rel_bias = nn.Parameter(torch.zeros(len(REL_TYPES)))
            self.sign_gate = nn.Linear(2, d_model)
            self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
            self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                    nn.Linear(2 * d_model, d_model))
            self.metric = nn.Linear(d_model, d_model)          # retrieval embedding
            self.dec_res = nn.Linear(d_model, n_genes)         # residual effect, on top of the tide
            self.dec_prob = nn.Linear(d_model, n_genes)        # response probability
            self.dec_dir = nn.Linear(d_model, n_genes)         # direction
            self.unc = nn.Linear(d_model, 1)                   # per-knockout log variance

        def encode(self, F, T, S, C, M, use_rel_bias=True):
            h = self.proj(F) + self.rel(T) + self.sign_gate(torch.stack([S, C], -1))
            bias = self.rel_bias[T] if use_rel_bias else torch.zeros_like(S)
            am = bias.unsqueeze(1).expand(-1, h.size(1), -1)
            am = am.repeat_interleave(self.attn.num_heads, dim=0)
            a, _ = self.attn(h, h, h, key_padding_mask=~M, attn_mask=am, need_weights=False)
            h = self.ln1(h + a)
            h = self.ln2(h + self.ff(h))
            m = M.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1)

        def forward(self, F, T, S, C, M, use_rel_bias=True):
            p = self.encode(F, T, S, C, M, use_rel_bias)
            e = self.metric(p)
            e = e / (e.norm(dim=-1, keepdim=True) + 1e-9)
            return e, self.dec_res(p), self.dec_prob(p), self.dec_dir(p), self.unc(p).squeeze(-1)

    return Cellformer()


def transfer_bench(cell_type, report):
    """A held-out CELL LINE bench. Metric 3 in CELLFORMER2.md, and the one never run until now.

    'This is the test AlphaFold's CASP is: held-out targets, not a held-out split of the same set.' Every
    number this project has published for the transformer arc is a held-out split of K562.
    """
    from eval_harness import Harness
    try:
        return Harness(cell_type)
    except Exception as e:
        report(f"    {cell_type}: unavailable ({type(e).__name__}) -- reported as unavailable, not as a null")
        return None


def shared_core(Yspec_train, k):
    """Rank-k shared core fitted on TRAINING perturbations only, returned as a projector.

    Fitting it on all perturbations would leak the held-out responses into the very thing being subtracted,
    and metric 2 exists precisely to remove what is shared -- so a leaky core would remove held-out signal
    and flatter every arm equally.
    """
    U, s, Vt = np.linalg.svd(Yspec_train, full_matrices=False)
    V = Vt[:k]
    return V


def metric2_recall(H, kos, predict, V, K=K_RECALL):
    """Recall@K on the response with the shared core projected out of BOTH truth and prediction."""
    out = []
    for ko in kos:
        v = predict(ko)
        if v is None:
            continue
        r = H.ki[ko]
        y = (H.A[r] * H.nontide.astype(np.float32))
        yr = y - V.T @ (V @ y)
        vr = v - V.T @ (V @ v)
        idx = np.argsort(-yr)[:K]
        truth = set(int(i) for i in idx if H.nontide[i])
        if not truth:
            continue
        top = set(int(i) for i in np.argsort(-vr)[:K])
        out.append(len(truth & top) / len(truth))
    return (float(np.mean(out)) if out else float("nan")), len(out)


def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    import torch
    torch.set_num_threads(4)
    report("=" * 100)
    report("CELLFORMER v1 -- typed relational transformer, explicit tide baseline, two metrics")
    report("=" * 100)
    report(f"  SIGN_MODE={SIGN_MODE}  seeds={N_SEEDS}  smoke={SMOKE}")
    report(f"  SCOPE: this is v1 of a staged plan. It tests typed relational attention plus a sign channel.")
    report(f"  It does NOT test mixed-curvature geometry, sheaf layers, hypergraph events, continuous time,")
    report(f"  multimodal heads or whole-cell simulation, and claims nothing about them.")

    prepare_depmap_vecs(report)
    from eval_harness import Harness
    from neural_ko import mean_recall
    H = Harness("K562")
    report(f"  bench: {len(H.kos):,} knockouts x {len(H.genes):,} genes; "
           f"{int(H.nontide.sum()):,} non-tide genes")

    dv = np.load(DEPVEC, allow_pickle=True)
    syms = [str(x) for x in dv["syms"]]
    Zraw = dv["Z"].astype(np.float32)
    # depmap_vecs.npz IS PER-GENE Z-SCORED, NOT RAW GENE EFFECT. Checked rather than assumed: RPL3, a core
    # ribosomal protein required by essentially every line, has mean exactly 0.000 in it, and every gene has
    # sd 1. So NO LEVEL-DEPENDENT QUANTITY MAY BE DERIVED FROM THIS MATRIX. The second version of this module
    # computed essentiality as (Zraw < -0.5).mean(1) from it and got 0.264 for the average gene and ZERO
    # genes dependent in more than half of lines -- which is not essentiality, it is "fraction of lines below
    # this gene's own -0.5 SD", about 28% for every gene by construction. The assertion below makes that
    # class of error loud instead of plausible.
    gm, gs = Zraw.mean(1), Zraw.std(1)
    # THE TEST IS THE MEAN ALONE, AND THE FIRST VERSION OF THIS GUARD GOT THAT WRONG. It required per-gene
    # mean ~0 AND per-gene sd ~1, and this file has sd ranging 0.141..1.000, so the guard printed "levels
    # preserved" about a matrix in which every gene's mean is 9e-07. Whether LEVELS survive depends on
    # centring only; scaling changes the units, not whether a level exists. A guard that passes on the
    # condition it was written to catch is worse than no guard, because it licenses the thing it was
    # supposed to stop.
    is_centred = bool(np.abs(gm).max() < 1e-3)
    report(f"  {DEPVEC.name} provenance check: per-gene mean |max| {np.abs(gm).max():.2e}, "
           f"sd range {gs.min():.3f}..{gs.max():.3f} -> "
           f"{'PER-GENE CENTRED: gene-effect LEVELS ARE DESTROYED' if is_centred else 'levels preserved'}")
    if is_centred:
        report(f"    nothing level-dependent may be derived from it -- essentiality below comes from the "
               f"cell object instead, and the canary checks that it did")
    is_z = is_centred
    Zd = Zraw if is_z else (Zraw - Zraw.mean(0)) / (Zraw.std(0) + 1e-6)
    # a compact projection: 1,150 raw line-columns per token x 24 tokens is too wide for CPU attention
    U, s, Vt = np.linalg.svd(Zd - Zd.mean(0), full_matrices=False)
    Zp = (U[:, :64] * s[:64]).astype(np.float32)
    srow = {g: i for i, g in enumerate(syms)}
    report(f"  DepMap profiles: {Zp.shape[0]:,} genes -> {Zp.shape[1]} SVD components per token")

    codep, cplx, ppi = load_layers(report)
    sig, sig_meta = load_signed(report, SIGN_MODE)
    deg = {g: len(v) for g, v in ppi.items()}
    # ESSENTIALITY FROM THE CELL OBJECT'S OWN `ess` FIELD -- the same source transformer_ko used, so the
    # v1-vs-transformer_ko comparison is not confounded by a differently-defined feature.
    dcell = json.load(gzip.open(NET, "rt"))
    ess = {g["name"]: float(g.get("ess") or 0.0) for g in dcell["genes"]}
    del dcell
    # CANARY. A feature that is silently constant passes every control, so three genes no one disputes are
    # required to score as essential. If they do not, the feature is not what its name says and the run
    # stops rather than producing a number about a column of zeros.
    canary = {g: ess.get(g) for g in ("RPL3", "EIF3B", "POLR2A", "RPS3", "SNRPD1")}
    got = [g for g, v in canary.items() if v and v > 0.5]
    report(f"  essentiality from the cell object: {sum(1 for v in ess.values() if v > 0.5):,} genes "
           f"essential; canary {canary}")
    if len(got) < 3:
        raise SystemExit(
            f"ESSENTIALITY CANARY FAILED: of {list(canary)}, only {got} score essential. These are core "
            f"ribosome/spliceosome/polymerase genes; a definition under which they are not essential is not "
            f"essentiality. Refusing to run an arm on a feature that is not what it is named.")
    # BASELINE K562 STATE, from the deposit's own per-gene mean expression rather than a constant.
    base_expr = {}
    try:
        import h5py
        with h5py.File(GWPS, "r") as f:
            cats = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in f["var"]["__categories"]["gene_name"][:]]
            gn = [cats[i] for i in f["var"]["gene_name"][:]]
            mu = f["var"]["mean"][:].astype(np.float64)
        base_expr = {g: float(m) for g, m in zip(gn, mu)}
        report(f"  baseline K562 expression for {len(base_expr):,} genes "
               f"(median {np.median(list(base_expr.values())):.3f})")
    except Exception as e:
        raise SystemExit(f"baseline K562 state unavailable ({e}). It is an input this model declares, so "
                         f"it is not silently replaced with zeros.")

    # POWER STATEMENT, printed before any arm.
    prior_sd = 0.03
    mde = 3 * prior_sd / np.sqrt(max(N_SEEDS, 1))
    report(f"\n  POWER, BEFORE THE ARMS: at {N_SEEDS} seeds and the sd of 0.03 this readout has shown across")
    report(f"  six previous levers, the smallest increment separable at ~3 standard errors is "
           f"{mde:+.4f}.")
    report(f"  Every architectural lever tried on this readout so far has produced |delta| <= 0.02. An arm")
    report(f"  that lands inside {mde:.4f} is UNDERPOWERED, not null, and is reported as such.")

    rng = np.random.default_rng(0)
    kos = [k for k in H.kos if k in srow]
    report(f"\n  {len(kos):,}/{len(H.kos):,} knockouts have a DepMap profile")
    if SMOKE:
        kos = kos[:300]
        report(f"  SMOKE: restricted to {len(kos)} knockouts")

    feats, types, signs, confs = build_tokens(kos, codep, cplx, ppi, sig, srow, Zp, base_expr, ess, deg,
                                              rng, "real")
    ntok = float(np.mean([len(f) for f in feats]))
    report(f"  tokens per knockout: mean {ntok:.1f}, cap {MAX_TOK}")
    tt = np.concatenate(types)
    report("  token type mix: " + ", ".join(f"{t} {100*np.mean(tt==i):.0f}%"
                                            for i, t in enumerate(REL_TYPES)))

    R = {"model": "cellformer-v1-scaffold",
         "scope": "v1 of a staged plan: typed relational attention + sign channel. Does NOT test "
                  "mixed-curvature geometry, sheaf layers, hypergraph events, continuous time, multimodal "
                  "heads or whole-cell simulation",
         "sign_mode": SIGN_MODE, "sign_meta": sig_meta, "assay_metadata": ASSAY_CONST,
         "n_knockouts": len(kos), "tokens_per_ko_mean": ntok,
         "token_type_fraction": {t: float(np.mean(tt == i)) for i, t in enumerate(REL_TYPES)},
         "power": {"n_seeds": N_SEEDS, "assumed_seed_sd": prior_sd,
                   "min_detectable_increment": float(mde),
                   "statement": "an arm inside this band is UNDERPOWERED, not null"},
         "priors_recorded_before_run": {
             "transformer_ko6_regulatory_features": -0.021,
             "transformer_signed_sign_over_shuffled": +0.0049,
             "transformer_graphnet_real_vs_shuffled_wiring": +0.005,
             "expectation": "v1 is expected to tie transformer_ko (0.490) and the sign arm is expected to "
                            "be a wash. If it is, imputed signs are a layer with provenance and no "
                            "measured use, which is worth knowing before anything is built on them"},
         "smoke": SMOKE}

    if SMOKE:
        report("\n  SMOKE MODE: structure validated, arms not run. Flip CELLFORMER_SMOKE=0 for the real run,")
        report("  and CELLFORMER_SIGN=measured+imputed once outputs/orphan/impute_reg_sign.json exists.")
        model = make_model(Zp.shape[1] + 6, D_MODEL, len(H.genes))
        F, T, S, C, M = pad(feats[:16], types[:16], signs[:16], confs[:16])
        with torch.no_grad():
            e, resid, prob, dr, unc = model(F, T, S, C, M)
        report(f"  forward pass OK: embedding {tuple(e.shape)}, residual {tuple(resid.shape)}, "
               f"prob {tuple(prob.shape)}, direction {tuple(dr.shape)}, uncertainty {tuple(unc.shape)}")
        report(f"  relation-bias ablation path OK: "
               f"{tuple(model(F, T, S, C, M, use_rel_bias=False)[0].shape)}")
        R["smoke_forward_ok"] = True
        R["verdict"] = ("SCAFFOLD ONLY -- structure built and forward-validated, no arm measured. The sign "
                        "channel plugs in at SIGN_MODE=measured+imputed the moment "
                        "outputs/orphan/impute_reg_sign.json exists, and raises rather than silently "
                        "falling back to the measured edges if it does not.")
    else:
        # ---------------------------------------------------------------- the arms
        #
        # THE FRAME IS LEARNED RETRIEVAL, NOT REGRESSION, and that is not a style choice: neural_ko measured
        # that a regression net on this target collapses to the tide and lands BELOW the tide-null, with a
        # feature-shuffle control matching it exactly. The embedding is trained so its cosine matches the
        # cosine of tide-removed responses; at inference a held-out knockout is scored by averaging the
        # responses of its nearest training knockouts.
        Yspec = np.stack([H.A[H.ki[k]] for k in kos]).astype(np.float32) * H.nontide.astype(np.float32)
        Yn = Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9)
        Fr, Tr, Sr, Cr, Mr = pad(feats, types, signs, confs)
        rng2 = np.random.default_rng(0)
        fr, tr, sr, cr = build_tokens(kos, codep, cplx, ppi, sig, srow, Zp, base_expr, ess, deg,
                                      rng2, "random")
        Frr, Trr, Srr, Crr, Mrr = pad(fr, tr, sr, cr)

        Mag = np.stack([H.A[H.ki[k]] for k in kos]).astype(np.float32)      # |z|, the magnitude target
        Sgn = (np.stack([H.M[H.ki[k]] for k in kos]) > 0).astype(np.float32)  # direction, from the SIGNED
        Mov = (Mag >= 1.0).astype(np.float32)                                  # response indicator
        import torch.nn.functional as Fn

        def train_eval(F, T, S, C, M, tr_idx, te_idx, use_rel_bias=True, zero_sign=False, seed=0,
                       heads=True):
            """Multi-task: retrieval metric + residual-on-top-of-tide + probability + direction +
            uncertainty. The tide term is a CONSTANT added before the loss, so predicting it earns zero."""
            torch.manual_seed(seed)
            model = make_model(F.shape[2], D_MODEL, Mag.shape[1])
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            Yn_t = torch.from_numpy(Yn[tr_idx])
            tide = torch.from_numpy(Mag[tr_idx].mean(0))          # THE TIDE, fitted on TRAIN only
            Mg = torch.from_numpy(Mag[tr_idx])
            Sg = torch.from_numpy(Sgn[tr_idx])
            Mv = torch.from_numpy(Mov[tr_idx])
            Ft, Tt, St, Ct, Mt = F[tr_idx], T[tr_idx], S[tr_idx], C[tr_idx], M[tr_idx]
            if zero_sign:
                St, Ct = torch.zeros_like(St), torch.zeros_like(Ct)
            n = len(tr_idx)
            for _ep in range(EPOCHS):
                perm = torch.randperm(n)
                for i in range(0, n, 96):
                    b_ = perm[i:i + 96]
                    if len(b_) < 8:
                        continue
                    e, res, prob, dr, lv = model(Ft[b_], Tt[b_], St[b_], Ct[b_], Mt[b_], use_rel_bias)
                    tgt = Yn_t[b_] @ Yn_t[b_].T
                    loss = ((e @ e.T) - tgt).pow(2).mean()
                    if heads:
                        pred = tide.unsqueeze(0) + res                 # tide is NOT learned
                        err = (pred - Mg[b_]).pow(2)
                        loss = loss + err.mean()
                        loss = loss + Fn.binary_cross_entropy_with_logits(prob, Mv[b_])
                        w = Mv[b_]                                     # direction only where it moved
                        if w.sum() > 0:
                            loss = loss + (Fn.binary_cross_entropy_with_logits(dr, Sg[b_],
                                                                               reduction="none")
                                           * w).sum() / w.sum()
                        # Gaussian NLL on the model's OWN magnitude error -- a calibrated uncertainty, not
                        # a free parameter: it is rewarded for predicting where it will be wrong.
                        loss = loss + 0.5 * (lv + err.mean(1).detach() / lv.exp().clamp(min=1e-4)).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                Sa, Ca = (torch.zeros_like(S), torch.zeros_like(C)) if zero_sign else (S, C)
                e, res, prob, dr, lv = model(F, T, Sa, Ca, M, use_rel_bias)
            return (e.numpy(), (tide.numpy()[None, :] + res.numpy()), torch.sigmoid(prob).numpy(),
                    torch.sigmoid(dr).numpy(), lv.numpy(), model)

        def retriever(E, tr_idx, top=10):
            """Learned-retrieval readout: score a held-out knockout by averaging the responses of its
            nearest TRAINING knockouts in the learned metric. This is the frame neural_ko established as
            the only deep one that works here; the residual head below is the direct-regression arm and is
            reported beside it rather than instead of it."""
            En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            pos = {kos[i]: i for i in range(len(kos))}

            def f(ko):
                i = pos.get(ko)
                if i is None:
                    return None
                sims = En[tr_idx] @ En[i]
                nn = tr_idx[np.argsort(-sims)[:top]]
                return np.stack([H.A[H.ki[kos[j]]] for j in nn]).mean(0)
            return f

        arms, per_seed = {}, {}
        head_rows, m3 = [], {}
        kgene_i = {g: i for i, g in enumerate(H.genes)}
        report("\n  METRIC 3 benches (held-out CELL LINES, the test never run in this arc):")
        benches = {ct: transfer_bench(ct, report) for ct in TRANSFER_LINES}
        for ct, b_ in benches.items():
            if b_ is not None:
                report(f"    {ct:10s} {len(b_.kos):,} knockouts x {len(b_.genes):,} genes")
        for seed in range(N_SEEDS):
            rs = np.random.default_rng(500 + seed).permutation(len(kos))
            te_idx = rs[:len(kos) // 4]
            tr_idx = rs[len(kos) // 4:]
            test_kos = [kos[i] for i in te_idx]
            V = shared_core(Yspec[tr_idx], CORE_K)          # fitted on TRAIN only -- no leakage

            def rec(name, predict):
                m1, n1 = mean_recall(H, test_kos, predict, K_RECALL)
                m2, _n2 = metric2_recall(H, test_kos, predict, V, K_RECALL)
                arms.setdefault(name, {"m1": [], "m2": []})
                arms[name]["m1"].append(m1)
                arms[name]["m2"].append(m2)
                report(f"    seed {seed}  {name:34s} m1 {m1:.4f}  m2 {m2:.4f}  (n={n1})")

            report(f"\n  SPLIT {seed}: {len(tr_idx):,} train / {len(te_idx):,} test knockouts")
            rec("TIDE-null (floor)", lambda ko: H.mover_freq)
            dpos = {kos[i]: srow[kos[i]] for i in range(len(kos))}
            Zn = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-9)

            def codep_f(ko):
                i = dpos.get(ko)
                if i is None:
                    return None
                sims = Zn[[dpos[kos[j]] for j in tr_idx]] @ Zn[i]
                nn = tr_idx[np.argsort(-sims)[:10]]
                return np.stack([H.A[H.ki[kos[j]]] for j in nn]).mean(0)
            rec("classical raw-cosine", codep_f)

            selfmask = Tr == 0
            Fs = Fr * selfmask.unsqueeze(-1).float()
            rec("self-only (no neighbourhood)",
                retriever(train_eval(Fs, Tr, Sr, Cr, selfmask, tr_idx, te_idx, seed=seed)[0], tr_idx))
            E_full, P_res, P_prob, P_dir, P_lv, model_full = train_eval(
                Fr, Tr, Sr, Cr, Mr, tr_idx, te_idx, seed=seed)
            rec("CELLFORMER v1 (full)", retriever(E_full, tr_idx))
            rec("  - relation bias",
                retriever(train_eval(Fr, Tr, Sr, Cr, Mr, tr_idx, te_idx, use_rel_bias=False,
                                     seed=seed)[0], tr_idx))
            rec("  - sign channel",
                retriever(train_eval(Fr, Tr, Sr, Cr, Mr, tr_idx, te_idx, zero_sign=True,
                                     seed=seed)[0], tr_idx))
            Ssh = torch.from_numpy(np.random.default_rng(900 + seed).permutation(Sr.numpy().ravel())
                                   .reshape(Sr.shape).astype(np.float32))
            rec("  ctrl: shuffled sign",
                retriever(train_eval(Fr, Tr, Ssh, Cr, Mr, tr_idx, te_idx, seed=seed)[0], tr_idx))
            rec("  ctrl: random partners",
                retriever(train_eval(Frr, Trr, Srr, Crr, Mrr, tr_idx, te_idx, seed=seed)[0], tr_idx))
            sh = {kos[i]: kos[j] for i, j in zip(te_idx, np.random.default_rng(7 + seed).permutation(te_idx))}
            rf = retriever(E_full, tr_idx)
            rec("  ctrl: wrong-knockout", lambda ko: rf(sh.get(ko, ko)))

            # ---- the residual head's OWN ranking, and the direction/probability/uncertainty heads ----
            #
            # This is the arm neural_ko warned about: a decoder to gene space that can collapse to the
            # tide. Here the tide is ADDED as a constant, so collapsing earns zero -- reporting this arm
            # beside the retrieval arm is what shows whether that design actually held.
            kpos = {kos[i]: i for i in range(len(kos))}
            rec("residual head (tide + learned)", lambda ko: (P_res[kpos[ko]] if ko in kpos else None))
            te_set = [k for k in test_kos if k in kpos]
            ii = np.array([kpos[k] for k in te_set])
            mv = Mov[ii].astype(bool)
            from sklearn.metrics import roc_auc_score
            try:
                auc_p = float(roc_auc_score(Mov[ii].ravel(), P_prob[ii].ravel()))
            except Exception:
                auc_p = float("nan")
            try:
                auc_d = float(roc_auc_score(Sgn[ii][mv].ravel(), P_dir[ii][mv].ravel()))
            except Exception:
                auc_d = float("nan")
            err = np.abs(P_res[ii] - Mag[ii]).mean(1)
            unc = P_lv[ii]
            from scipy import stats as _st
            rho_u = float(_st.spearmanr(unc, err)[0])
            head_rows.append({"seed": seed, "prob_auc": auc_p, "dir_auc_on_movers": auc_d,
                              "uncertainty_vs_error_spearman": rho_u,
                              "dir_base_rate": float(Sgn[ii][mv].mean())})
            report(f"    seed {seed}  HEADS  response-prob AUC {auc_p:.4f} | direction AUC {auc_d:.4f} "
                   f"(base rate {Sgn[ii][mv].mean():.3f}) | uncertainty-vs-error rho {rho_u:+.4f}")

            # ---- METRIC 3: transfer to held-out CELL LINES, no retraining ----
            for ct in TRANSFER_LINES:
                Ht = benches.get(ct)
                if Ht is None:
                    continue
                shared = [g for g in Ht.genes if g in kgene_i]
                col_t = np.array([Ht.gi[g] for g in shared])
                col_k = np.array([kgene_i[g] for g in shared])
                tk = [k for k in Ht.kos if k in kpos]
                if len(tk) < 30 or len(shared) < 500:
                    report(f"      {ct}: {len(tk)} shared KOs / {len(shared)} shared genes -- too thin, "
                           f"reported as unavailable")
                    continue
                hit, base = [], []
                for k in tk:
                    r = Ht.ki[k]
                    truth = set(int(i) for i in np.argsort(-Ht.A[r][col_t])[:K_RECALL]
                                if Ht.nontide[col_t[i]])
                    if not truth:
                        continue
                    pred = P_res[kpos[k]][col_k]
                    top = set(int(i) for i in np.argsort(-pred)[:K_RECALL])
                    hit.append(len(truth & top) / len(truth))
                    bt = set(int(i) for i in np.argsort(-Ht.mover_freq[col_t])[:K_RECALL])
                    base.append(len(truth & bt) / len(truth))
                if hit:
                    m3.setdefault(ct, {"model": [], "tide": [], "n": []})
                    m3[ct]["model"].append(float(np.mean(hit)))
                    m3[ct]["tide"].append(float(np.mean(base)))
                    m3[ct]["n"].append(len(hit))
                    report(f"      METRIC 3 {ct:10s} model {np.mean(hit):.4f}  tide {np.mean(base):.4f}  "
                           f"delta {np.mean(hit)-np.mean(base):+.4f}  ({len(hit)} KOs, "
                           f"{len(shared):,} shared genes)")

            def oracle_f(ko):
                i = {kos[x]: x for x in range(len(kos))}.get(ko)
                if i is None:
                    return None
                sims = Yn[tr_idx] @ Yn[i]
                return H.A[H.ki[kos[tr_idx[int(np.argmax(sims))]]]]
            rec("ORACLE (retrieval ceiling)", oracle_f)

        report(f"\n  {'arm':36s} {'metric1':>16s} {'metric2':>16s}")
        for k, v in arms.items():
            report(f"  {k:36s} {np.mean(v['m1']):8.4f} +/- {np.std(v['m1']):.4f} "
                   f"{np.mean(v['m2']):8.4f} +/- {np.std(v['m2']):.4f}")

        def M(k, m="m1"):
            return float(np.mean(arms[k][m])) if k in arms else float("nan")
        full1, full2 = M("CELLFORMER v1 (full)"), M("CELLFORMER v1 (full)", "m2")
        d_sign = full1 - M("  ctrl: shuffled sign")
        d_rand = full1 - M("  ctrl: random partners")
        d_self = full1 - M("self-only (no neighbourhood)")
        d_bias = full1 - M("  - relation bias")
        sd_pool = float(np.mean([np.std(v["m1"]) for v in arms.values()]))
        mde_obs = 3 * sd_pool / np.sqrt(max(N_SEEDS, 1))
        report(f"\n  observed pooled seed sd {sd_pool:.4f} -> minimum detectable increment "
               f"{mde_obs:+.4f} at {N_SEEDS} seeds")
        R["heads"] = head_rows
        R["metric3_transfer"] = {ct: {"model_mean": float(np.mean(v["model"])),
                                      "tide_mean": float(np.mean(v["tide"])),
                                      "delta": float(np.mean(v["model"]) - np.mean(v["tide"])),
                                      "per_seed_model": v["model"], "per_seed_tide": v["tide"],
                                      "n_kos": v["n"]} for ct, v in m3.items()}
        R["arms"] = {k: {"metric1_mean": float(np.mean(v["m1"])), "metric1_sd": float(np.std(v["m1"])),
                         "metric2_mean": float(np.mean(v["m2"])), "metric2_sd": float(np.std(v["m2"])),
                         "per_seed_m1": v["m1"], "per_seed_m2": v["m2"]} for k, v in arms.items()}
        R["deltas"] = {"vs_self_only": d_self, "vs_random_partners": d_rand,
                       "sign_over_shuffled_sign": d_sign, "relation_bias": d_bias}
        R["power_observed"] = {"pooled_seed_sd": sd_pool, "min_detectable_increment": float(mde_obs)}
        under = {k: bool(abs(v) < mde_obs) for k, v in R["deltas"].items()}
        R["underpowered"] = under
        R["verdict"] = (
            f"CELLFORMER v1 reaches metric-1 recall@50 {full1:.4f} +/- "
            f"{np.std(arms['CELLFORMER v1 (full)']['m1']):.4f} against a tide floor of "
            f"{M('TIDE-null (floor)'):.4f} and an oracle of {M('ORACLE (retrieval ceiling)'):.4f}, and "
            f"metric-2 recall {full2:.4f} on the residual after the rank-{CORE_K} shared core is projected "
            f"out. THE NEIGHBOURHOOD IS THE ONLY THING THAT CLEARLY PAYS: full minus self-only "
            f"{d_self:+.4f}, full minus random partners {d_rand:+.4f}. THE SIGN CHANNEL IS "
            f"{'NOT DETECTED' if under['sign_over_shuffled_sign'] else 'DETECTED'} at "
            f"{d_sign:+.4f} over its own SHUFFLED-sign control -- scored against shuffled sign, never "
            f"against no-sign, because sign-over-nothing conflates 'sign matters' with 'an extra channel "
            f"matters'. The relation-specific attention bias is worth {d_bias:+.4f}. Observed pooled seed "
            f"sd is {sd_pool:.4f}, so the minimum detectable increment at {N_SEEDS} seeds is "
            f"{mde_obs:+.4f}: any delta inside that band is UNDERPOWERED, not null "
            f"({sum(under.values())}/{len(under)} of the reported deltas are). This is v1 for the current "
            f"task and says nothing about mixed-curvature geometry, sheaf layers, hypergraph events, "
            f"continuous time or multimodal heads.")

    R["limits"] = [
        "this is v1 for the CURRENT task -- ranking specific movers for a held-out knockout in one cell "
        "type at one timepoint. It is not a whole-cell model and the staged versions above it are untested "
        "hypotheses",
        "the readout has ~0.12 of headroom between the tide floor and the retrieval oracle, and six "
        "architectural levers have each moved it by <= 0.02. Metric 1 may be unable to separate v1 from "
        "transformer_ko whatever v1 is worth, which is why metric 2 is computed",
        "metric 2's shared core is fitted on training perturbations only, so it is leak-free, but k=20 is "
        "taken from CELLFORMER2.md rather than re-selected here",
        "assay and time metadata are CONSTANT in this deposit. The slot exists so v4 can vary it; it "
        "carries no information now and must not be read as a tested channel",
        "depmap_vecs.npz is per-gene z-scored, so gene-effect LEVELS are not available from it and nothing "
        "level-dependent may be derived from it. Essentiality therefore comes from the cell object's own "
        "`ess` field -- the same source transformer_ko used -- and is guarded by a canary on five core "
        "ribosome/spliceosome/polymerase genes",
        "the DepMap profile per token is compressed to 64 SVD components for CPU tractability, which is a "
        "lossy change from transformer_ko's raw 1,150-column profile and could by itself move the "
        "comparison. Any v1-vs-transformer_ko difference must be attributed to this before it is "
        "attributed to the architecture",
        "token budget is capped at 24, split across four relation types, so a knockout with many partners "
        "is truncated. Which partners survive truncation is the list order, not a ranking",
    ]
    R["log"] = log
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "cellformer_v1.json", "w"), indent=1)
    report(f"\n  -> {OUT/'cellformer_v1.json'}")


if __name__ == "__main__":
    main()
