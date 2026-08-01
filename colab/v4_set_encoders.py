"""V4 CRITERION 2 -- does a simple set encoder match the transformer? THE DECISIVE UNRUN TEST.

THE QUESTION, AND WHY IT IS THE HIGHEST-PRIORITY UNRUN TEST IN THE PROGRAMME. `CELLFORMER4.md` section 6
states the condition on Block 3 in one line: "The transformer remains only if it beats simpler set
encoders." Section 17 lists it as success criterion 2 -- "a simple set encoder cannot match the
transformer" -- and section 18 records its status as NEVER TESTED. Arms 6-8 of section 6's mandatory
control list (simple weighted mean, Deep Sets, Set Transformer) have never been run against each other.
Everything the project calls "the validated transformer result" is in fact a validated PARTNER-SET result:
it was measured by swapping the partners, never by swapping the encoder.

WHAT THE EXISTING EVIDENCE PREDICTS, STATED BEFORE THE RUN SO IT CANNOT BE RETROFITTED. Two measurements
already in the registry point the same way:

    internal wiring vs shuffled wiring at the same density   +0.005 +/- 0.008   NOT DETECTED
    relation-type-specific attention bias                    +0.0034            NOT DETECTED

Both are statements that the model is insensitive to WHICH partner is WHICH -- the first to topology, the
second to type. An attention layer's entire job is to weight tokens differently from each other. If a
model is already indifferent to the identity and the type of its tokens, an unweighted mean over them
should lose nothing. The prediction is therefore that arms 1-3 match arm 4. If they do, criterion 2 FAILS
and the object this project has validated is a BAG OF TYPED PARTNERS, not a transformer, and Block 3
should be reimplemented as a mean-pool at a fraction of the cost.

WHAT IS HELD FIXED. Task, target, metric, controls and retrieval are taken from `colab/dense_stage1.py`
UNCHANGED, so the numbers sit on the same axis as the +0.0186 real-vs-random result and the 0.0526 dense
oracle. Dense `gwps.h5ad` (11,258 perturbations x 8,248 genes, 100% dense -- never the `nlz_*` benches,
which store ~250 genes per knockout and whose other 96.5% of zeros mean NOT RECORDED). Target
S = |z| - per-gene mean |z| over the TRAINING knockouts, refit inside every partition. Truth = the top-50
non-tide genes by centred |z|. Primary metric recall@50; threshold-free cosine reported alongside so a
metric artifact shows up as the two disagreeing. Every arm is scored through the identical
nearest-10-training-knockouts retrieval, and every arm ranks candidates INSIDE THE NON-TIDE UNIVERSE the
truth set is drawn from. ONLY THE ENCODER VARIES.

THE ARMS. All learned arms are built to a MATCHED PARAMETER BUDGET -- the Set Transformer's count -- by
widening the simpler encoders' hidden layers, and the realised count of every arm is printed and written
to the JSON. A simple encoder that loses because it was given a third of the parameters would prove
nothing.

    0-param  A raw DepMap self-vector           retrieve on the perturbed gene's DepMap vector. No training
    0-param  B raw DepMap bag-of-partners       mean of the DepMap vectors of the real typed partners. No
                                                training, no parameters, no gradient. THE CHEAPEST
                                                SENSIBLE NON-LEARNED ENCODER, reported next to the model.
    0-param  C TIDE-null floor                  rank non-tide genes by how often they move.
    1  unweighted mean-pool                     mean over token features. The simplest set encoder there is
    2  evidence-weighted mean-pool              weighted by a per-relation-type evidence prior FIT ON THE
                                                TRAINING PARTITION ONLY (mean centred-profile cosine
                                                between a knockout and its type-t partners, among train
                                                knockouts). Section 6's "simple weighted mean"
    3  Deep Sets                                per-token MLP -> SUM -> MLP. Section 6's arm 7
    4  Set Transformer                          multihead self-attention + mean-pool. THE INCUMBENT
    5  Set Transformer, attention FROZEN UNIFORM  identical module, identical depth, identical residual
                                                and LayerNorm; the attention matrix is replaced by a
                                                uniform 1/n over valid tokens. THE SHARP ARM: it separates
                                                "attention helps" from "more parameters and more depth
                                                help", which arm 4 vs arm 1 alone cannot do
    6  self-only                                no neighbourhood at all
    7  random partners            CONTROL       same token count, same type mix, uniformly drawn partners
    8  degree-matched rewiring    CONTROL       configuration-model: partner stubs permuted WITHIN each
                                                relation type, so every knockout keeps its exact degree and
                                                every partner gene keeps its exact multiplicity. Uniform
                                                random destroys hub structure as well as wiring; this
                                                destroys only the wiring
    9  neighbour SWAP             CONTROL       COUNTERFACTUAL: arm 4's trained network is handed another
                                                TEST knockout's neighbour tokens while keeping the correct
                                                self token. Accuracy alone cannot distinguish "the
                                                neighbourhood is useless" from "the neighbourhood is
                                                IGNORED"; only this can
    10 wrong-knockout             CONTROL       identity control, arm 4's predictions permuted across test
                                                knockouts. Bounds the metric's own degeneracy

THE UNIT OF REPLICATION IS A DISJOINT KNOCKOUT PARTITION, NOT A MODEL-INIT SEED. The claim under test is
"encoder X generalises to knockouts it has never seen", so the error bar must be computed across UNSEEN
KNOCKOUT SETS. The knockouts are split into FOLDS disjoint blocks; each block is held out once; an arm's
replicates are its FOLDS held-out-block values. Rotating initialisation seeds inside one partition measures
optimiser noise instead, and in this project that exact substitution already flipped a verdict from 4/7 to
0/7 (`dense_context.py`). Model-init seeds ARE run here, on partition 0 only, and their spread is reported
purely so the two floors can be compared -- nothing is graded on it.

    MDE = 3 * sd / sqrt(n),  n = number of DISJOINT KNOCKOUT PARTITIONS (default 3),
    sd = the across-partition sd of an arm's held-out recall@50, pooled over arms (ddof=1).

THE GATE, DECLARED BEFORE ANY NUMBER IS PRODUCED.

    PRIMARY   The Set Transformer (arm 4) is RETAINED only if it beats the BEST of arms 1-3 by more than
              the cross-partition MDE, on recall@50. Otherwise criterion 2 FAILS, and Block 3's validated
              object is a bag of typed partners.
    SHARP     Attention is credited only if arm 4 beats arm 5 by more than the MDE. If arm 4 beats arm 1
              but not arm 5, the gain is depth and parameters, not attention.
    FREE      Any learned arm is worth building only if it beats the best 0-parameter arm (A/B/C) by more
              than the MDE. A trivial baseline that matches the model IS the headline, not a footnote.
    AGREE     The gate is reported as sound only if the threshold-free cosine metric agrees in sign with
              recall@50. Disagreement means metric artifact and the result is not reportable as biology.

EFFECT SIZES ARE RAW HELD-OUT VALUES. The controls (arms 7-10, and the 0-parameter arms) establish
significance and attribution; no arm is ever reported "net of" a control. Where a control reproduces most
of an arm's advantage that is stated as the finding.

NO CIRCULARITY, CHECKED. Token features are DepMap CRISPR gene-effect PCs and the co-dependency edges are
DepMap-derived; the scored target is Replogle Perturb-seq K562 transcriptomics (`gwps.h5ad`). No feature is
scored on the dataset it was derived from. Target-side statistics (the per-gene centring, the truth sets,
the evidence prior) are refit on the training partition only, inside every fold.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
DEPVEC = OUT / "depmap_vecs.npz"

# --- fixed by dense_stage1.py, changed here for no reason whatsoever -------------------------------
TAU = 1.0            # |z| at which a gene counts as moved
TIDE_FRAC = 0.05     # top 5% most-frequently-moving genes are the tide
MIN_SPEC = 5         # a knockout needs this many specific movers to be scorable
K = 50               # retrieval depth of the primary metric
TRUTH_N = 50         # truth = the TRUTH_N largest centred-|z| non-tide genes, matched to K
MAX_TOK = 24
D_MODEL = 128
N_HEAD = 4
N_RETRIEVE = 10      # nearest training knockouts averaged to form a prediction
# --- this module's own knobs ----------------------------------------------------------------------
FOLDS = int(os.environ.get("V4_FOLDS", "3"))          # DISJOINT knockout partitions == unit of replication
EPOCHS = int(os.environ.get("V4_EPOCHS", "25"))
INIT_SEEDS = int(os.environ.get("V4_INIT_SEEDS", "3"))  # model-init seeds on partition 0, REPORTED NOT GRADED
N_KO = int(os.environ.get("V4_N", "0"))               # 0 = every usable knockout
MAX_PAIRS = int(os.environ.get("V4_PRIOR_PAIRS", "20000"))
SMOKE = os.environ.get("V4_SMOKE", "0") == "1"
if SMOKE:
    N_KO, EPOCHS, FOLDS, INIT_SEEDS = 800, 3, 3, 2


def robust_z(M):
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad


def main():
    import gzip
    import math
    import h5py
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)          # another job owns ~3.6 of this box's 4 cores
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 104)
    report("V4 CRITERION 2 -- does a simple set encoder match the transformer? (CELLFORMER4.md sec 6/17/18)")
    report("=" * 104)
    report(f"  GATE, declared before the run: the Set Transformer is retained only if it beats the BEST of")
    report(f"  arms 1-3 by more than the cross-partition MDE = 3*sd/sqrt(n), n = {FOLDS} DISJOINT KNOCKOUT")
    report(f"  PARTITIONS. Model-init seeds are reported but NOT graded on.")

    # ---------------------------------------------------------------- target, verbatim from dense_stage1
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["var"]["__categories"]["gene_name"][:]]
        gname = np.array([cats[i] for i in f["var"]["gene_name"][:]])
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    report(f"\n  dense K562: {X.shape[0]:,} perturbations x {X.shape[1]:,} genes, "
           f"{100*np.isfinite(X).mean():.1f}% finite  (NOT the nlz_* benches: 3.5% dense, zeros = NOT RECORDED)")

    Z = robust_z(X)
    A = np.abs(Z)
    ok_gene = np.isfinite(A).all(0)
    A = A[:, ok_gene]
    gname = gname[ok_gene]
    del X, Z
    mover = A >= TAU
    freq = mover.mean(0)
    tide = freq >= np.quantile(freq, 1 - TIDE_FRAC)
    nontide = ~tide
    spec_n = (mover & nontide[None, :]).sum(1)
    scorable = spec_n >= MIN_SPEC
    del mover
    report(f"  {A.shape[1]:,} genes with a defined robust z; {int(tide.sum()):,} tide / "
           f"{int(nontide.sum()):,} non-tide; scorable knockouts {int(scorable.sum()):,}")
    report(f"  truth = top {TRUTH_N} non-tide genes by TRAIN-CENTRED |z|; random floor recall@{K} = "
           f"{K/A.shape[1]:.4f}")

    dv = np.load(DEPVEC, allow_pickle=True)
    syms = [str(x) for x in dv["syms"]]
    Zd = dv["Z"].astype(np.float32)
    U, s_, _ = np.linalg.svd(Zd - Zd.mean(0), full_matrices=False)
    Zp = (U[:, :64] * s_[:64]).astype(np.float32)
    srow = {g: i for i, g in enumerate(syms)}
    del Zd, U

    d = json.load(gzip.open(NET, "rt"))
    nmz = [g["name"] for g in d["genes"]]
    codep, cplx, ppi = {}, {}, {}
    for a_, v in (d.get("codep") or {}).items():
        codep.setdefault(nmz[int(a_)], []).extend(nmz[int(b)] for b, _s in v)
    g2c = {int(k_): set(v) for k_, v in (d.get("gene2cplx") or {}).items()}
    bym = {}
    for x, cs in g2c.items():
        for c in cs:
            bym.setdefault(c, []).append(x)
    for c, mem in bym.items():
        for x in mem:
            cplx.setdefault(nmz[x], []).extend(nmz[y] for y in mem if y != x)
    for e in d["ppi"]:
        ppi.setdefault(nmz[int(e[0])], []).append(nmz[int(e[1])])
        ppi.setdefault(nmz[int(e[1])], []).append(nmz[int(e[0])])
    del d

    keep = np.where(scorable & np.array([s in srow for s in psym]))[0]
    if N_KO:
        keep = keep[:N_KO]
    A = A[keep]
    psym_k = psym[keep]
    N = len(keep)
    report(f"  usable knockouts (DepMap profile AND >= {MIN_SPEC} specific movers): {N:,}")

    def centred(tr_idx):
        """S = |z| - per-gene mean |z| over the TRAINING knockouts. Refit inside every partition."""
        mu = A[tr_idx].mean(0)
        Sz = np.where(nontide[None, :], A - mu[None, :], 0.0).astype(np.float32)
        T_ = np.empty((len(Sz), TRUTH_N), np.int64)
        for i0 in range(0, len(Sz), 512):
            blk = np.where(nontide[None, :], Sz[i0:i0 + 512], -np.inf)
            T_[i0:i0 + 512] = np.argpartition(-blk, TRUTH_N, axis=1)[:, :TRUTH_N]
        Yn_ = Sz / (np.linalg.norm(Sz, axis=1, keepdims=True) + 1e-9)
        return Sz, T_, Yn_

    # ------------------------------------------------------------------------------ the partner sets
    # Per-knockout, per-type partner lists are built ONCE and every token variant is derived from them,
    # so the three variants are exactly count-matched and type-mix-matched by construction.
    TYPES = ((1, codep), (2, cplx), (3, ppi))
    partners = [{t: [g for g in src.get(k_, [])[:7]] for t, src in TYPES} for k_ in psym_k]
    allg = list(srow)

    def pack(per_ko):
        F = np.zeros((N, MAX_TOK, Zp.shape[1]), np.float32)
        T = np.zeros((N, MAX_TOK), np.int64)
        M = np.zeros((N, MAX_TOK), bool)
        for i, (k_, lst) in enumerate(zip(psym_k, per_ko)):
            rows, types = [k_], [0]
            for t, gs in lst:
                rows.extend(gs)
                types.extend([t] * len(gs))
            for j, (g, t) in enumerate(zip(rows[:MAX_TOK], types[:MAX_TOK])):
                if g in srow:
                    F[i, j] = Zp[srow[g]]
                T[i, j] = t
                M[i, j] = True
        return torch.from_numpy(F), torch.from_numpy(T), torch.from_numpy(M)

    real_ko = [[(t, partners[i][t]) for t, _ in TYPES] for i in range(N)]
    rng_r = np.random.default_rng(1)
    rand_ko = [[(t, [allg[int(x)] for x in rng_r.integers(0, len(allg), len(partners[i][t]))])
                for t, _ in TYPES] for i in range(N)]
    # DEGREE-MATCHED CONFIGURATION MODEL: permute the partner stubs WITHIN a relation type. Every knockout
    # keeps its exact degree and every partner gene keeps its exact multiplicity -- only the wiring moves.
    # Uniform random (arm 7) also flattens the partner-degree distribution, so it destroys two things at
    # once and cannot attribute a drop to wiring.
    rng_c = np.random.default_rng(2)
    cfg_ko = [[] for _ in range(N)]
    for t, _ in TYPES:
        stubs = [g for i in range(N) for g in partners[i][t]]
        perm = rng_c.permutation(len(stubs))
        pos = 0
        for i in range(N):
            n_ = len(partners[i][t])
            cfg_ko[i].append((t, [stubs[perm[pos + j]] for j in range(n_)]))
            pos += n_
    Fr, Tr, Mr = pack(real_ko)
    Frr, Trr, Mrr = pack(rand_ko)
    Frc, Trc, Mrc = pack(cfg_ko)
    report(f"  tokens per knockout: real {Mr.sum(1).float().mean():.2f}, random {Mrr.sum(1).float().mean():.2f}, "
           f"degree-matched {Mrc.sum(1).float().mean():.2f}  (matched by construction)")
    ko_row = {}
    for i, g in enumerate(psym_k):
        ko_row.setdefault(g, i)

    # ------------------------------------------------------------------------------------ the encoders
    class SetEnc(nn.Module):
        """One encoder family. `kind` is the ONLY thing that differs between arms 1-6."""

        def __init__(self, kind, d_in, d, w, prior=None):
            super().__init__()
            self.kind = kind
            self.proj = nn.Linear(d_in, d)
            self.rel = nn.Embedding(4, d)
            self.ln = nn.LayerNorm(d)
            if kind in ("attn", "attn_uniform"):
                self.q = nn.Linear(d, d)
                self.k = nn.Linear(d, d)
                self.v = nn.Linear(d, d)
                self.o = nn.Linear(d, d)
            if kind == "deepsets":
                self.tok = nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Linear(w, d))
            self.head = nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Linear(w, D_MODEL))
            self.register_buffer("prior", torch.ones(4) if prior is None
                                 else torch.from_numpy(np.asarray(prior, np.float32)))

        def forward(self, F, T, M):
            h = self.proj(F) + self.rel(T)
            m = M.unsqueeze(-1).float()
            if self.kind in ("attn", "attn_uniform"):
                B, L, D = h.shape
                dh = D // N_HEAD
                v = self.v(h).view(B, L, N_HEAD, dh).transpose(1, 2)
                if self.kind == "attn":
                    q = self.q(h).view(B, L, N_HEAD, dh).transpose(1, 2)
                    k = self.k(h).view(B, L, N_HEAD, dh).transpose(1, 2)
                    lg = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
                    att = lg.masked_fill(~M[:, None, None, :], -1e9).softmax(-1)
                else:
                    # FROZEN UNIFORM: every query attends equally to every valid token. Same depth, same
                    # residual, same LayerNorm, same value/output projections -- only the attention
                    # PATTERN is removed. self.q and self.k still exist and never receive a gradient;
                    # both the total and the effective parameter counts are reported.
                    wq = M[:, None, None, :].float()
                    att = wq / wq.sum(-1, keepdim=True).clamp(min=1.0)
                a = self.o((att @ v).transpose(1, 2).reshape(B, L, D))
                h = self.ln(h + a)
            elif self.kind == "deepsets":
                h = self.ln(h + self.tok(h))
            else:
                h = self.ln(h)
            if self.kind == "evidence":
                wt = (self.prior[T] * m.squeeze(-1)).unsqueeze(-1)
            else:
                wt = m
            p = (h * wt).sum(1)
            if self.kind != "deepsets":               # Deep Sets is defined with a SUM pool
                p = p / wt.sum(1).clamp(min=1e-6)
            e = self.head(p)
            return e / (e.norm(dim=-1, keepdim=True) + 1e-9)

    def nparam(mod):
        return sum(p.numel() for p in mod.parameters())

    def build(kind, w):
        return SetEnc(kind, Zp.shape[1], D_MODEL, w)

    # MATCHED PARAMETER BUDGET. The Set Transformer sets the budget; every other learned arm's hidden width
    # is bisected to land on it. A simple encoder that lost because it had a third of the parameters would
    # prove nothing about set encoders.
    BUDGET = nparam(build("attn", D_MODEL))

    def match_width(kind):
        lo, hi = 4, 4096
        while lo < hi:
            mid = (lo + hi) // 2
            if nparam(build(kind, mid)) < BUDGET:
                lo = mid + 1
            else:
                hi = mid
        return lo if abs(nparam(build(kind, lo)) - BUDGET) <= abs(nparam(build(kind, lo - 1)) - BUDGET) \
            else lo - 1

    WID = {k_: (D_MODEL if k_ in ("attn", "attn_uniform") else match_width(k_))
           for k_ in ("mean", "evidence", "deepsets", "attn", "attn_uniform")}
    DEAD = {k_: 0 for k_ in WID}
    DEAD["attn_uniform"] = nparam(build("attn", D_MODEL).q) + nparam(build("attn", D_MODEL).k)
    PAR = {k_: nparam(build(k_, WID[k_])) for k_ in WID}
    report(f"\n  MATCHED PARAMETER BUDGET (the Set Transformer's {BUDGET:,}); hidden width bisected per arm:")
    for k_ in WID:
        report(f"    {k_:16s} hidden {WID[k_]:5d}  params {PAR[k_]:8,d}  "
               f"({100*(PAR[k_]-BUDGET)/BUDGET:+.2f}% of budget)"
               + (f"  [{DEAD[k_]:,} of them receive NO gradient: q,k are unused when attention is frozen]"
                  if DEAD[k_] else ""))

    # ------------------------------------------------------------------------------- train / score
    def train(kind, F, T, M, tr, seed, Yn, prior=None):
        torch.manual_seed(seed)
        net = SetEnc(kind, Zp.shape[1], D_MODEL, WID[kind], prior)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Yt = torch.from_numpy(Yn[tr])
        n = len(tr)
        for _e in range(EPOCHS):
            perm = torch.randperm(n)
            for i0 in range(0, n, 128):
                b = perm[i0:i0 + 128]
                if len(b) < 8:
                    continue
                idx = torch.from_numpy(tr[b.numpy()])
                e = net(F[idx], T[idx], M[idx])
                loss = ((e @ e.T) - (Yt[b] @ Yt[b].T)).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            E = net(F, T, M).numpy()
        return net, E

    def embed(net, F, T, M):
        with torch.no_grad():
            return net(F, T, M).numpy()

    def retrieve(Eq, Ek, te, tr, S):
        """Identical retrieval for EVERY arm -- learned or not. Only the embedding differs."""
        q = Eq[te] / (np.linalg.norm(Eq[te], axis=1, keepdims=True) + 1e-9)
        k = Ek[tr] / (np.linalg.norm(Ek[tr], axis=1, keepdims=True) + 1e-9)
        sim = q @ k.T
        nn_ = np.argpartition(-sim, N_RETRIEVE, axis=1)[:, :N_RETRIEVE]
        return np.stack([S[tr[r]].mean(0) for r in nn_])

    def score_arm(scores, te, TRUTH, Yn):
        """recall@K plus a threshold-free cosine, on the same score matrix, dense_stage1 convention.

        BOTH are computed inside the NON-TIDE universe, which is the universe the truth set is drawn
        from. A baseline scored in a different universe than the truth scores ~0 by construction; that
        defect has shipped three times in this project and the assertion below is what stops a fourth.
        """
        rec, cos = [], []
        for j, i in enumerate(te):
            s = np.where(nontide, scores[j], -np.inf)
            top = np.argpartition(-s, K)[:K]
            assert nontide[top].all()
            rec.append(np.intersect1d(top, TRUTH[i]).size / TRUTH_N)
            v = np.where(nontide, scores[j], 0.0).astype(np.float64)
            nv = np.linalg.norm(v)
            cos.append(float(v @ Yn[i] / nv) if nv > 1e-9 else 0.0)
        return float(np.mean(rec)), float(np.mean(cos))

    def evidence_prior(tr, Yn):
        """Per-relation-type evidence weight, FIT ON THE TRAINING PARTITION ONLY.

        prior[t] = mean cosine between a training knockout's centred response profile and that of its
        type-t partners that are themselves TRAINING knockouts. prior[0] = 1 by definition (a knockout's
        cosine with itself). Held-out knockouts contribute nothing to the weights that will be used to
        score them.
        """
        in_tr = np.zeros(N, bool)
        in_tr[tr] = True
        pri = np.ones(4, np.float32)
        for t, _ in TYPES:
            a_, b_ = [], []
            for i in tr:
                for g in partners[i][t]:
                    j = ko_row.get(g)
                    if j is not None and in_tr[j] and j != i:
                        a_.append(i)
                        b_.append(j)
                if len(a_) >= MAX_PAIRS:
                    break
            if a_:
                aa, bb = np.array(a_), np.array(b_)
                v = [float((Yn[aa[i0:i0 + 1024]] * Yn[bb[i0:i0 + 1024]]).sum(1).sum())
                     for i0 in range(0, len(aa), 1024)]
                pri[t] = max(sum(v) / len(aa), 0.0)
            else:
                pri[t] = 0.0
        return pri, {int(t): float(pri[t]) for t, _ in TYPES}

    # ------------------------------------------------------------------------------------ the folds
    # the two zero-parameter encoders. No training, no gradient, no fold dependence.
    Eself = np.stack([Zp[srow[g]] for g in psym_k])
    Ebag = (Fr.numpy() * Mr.numpy()[:, :, None]).sum(1) / Mr.numpy().sum(1)[:, None].clip(min=1)

    order = np.random.default_rng(0).permutation(N)
    blocks = np.array_split(order, FOLDS)              # DISJOINT held-out knockout partitions
    res, extra = {}, {}
    priors_log = []
    for f in range(FOLDS):
        te = np.sort(blocks[f])
        tr = np.sort(np.concatenate([b for g, b in enumerate(blocks) if g != f]))
        S, TRUTH, Yn = centred(tr)
        pri, pri_d = evidence_prior(tr, Yn)
        priors_log.append(pri_d)
        report(f"\n  PARTITION {f}: {len(tr):,} train / {len(te):,} held-out knockouts (disjoint from every "
               f"other partition's held-out block)")
        report(f"    train-only evidence prior  codep {pri[1]:.4f}  complex {pri[2]:.4f}  ppi {pri[3]:.4f} "
               f" (self = 1.0000)")

        def add(name, sc):
            r, c = score_arm(sc, te, TRUTH, Yn)
            dd = res.setdefault(name, {"rec": [], "cos": []})
            dd["rec"].append(r)
            dd["cos"].append(c)
            report(f"    {name:44s} recall@{K} {r:.4f}   cosine {c:+.4f}")

        # ---- 0-parameter arms. Rule: the cheapest sensible non-learned baseline is reported NEXT TO the
        # model, not in a footnote. If one of these matches the transformer, that is the headline.
        add("C TIDE-null floor (0 param)", np.tile(np.where(nontide, freq, -np.inf), (len(te), 1)))
        add("A raw DepMap self-vector (0 param)", retrieve(Eself, Eself, te, tr, S))
        add("B raw DepMap bag-of-partners (0 param)", retrieve(Ebag, Ebag, te, tr, S))

        # ---- learned arms, all at the matched budget
        _, E1 = train("mean", Fr, Tr, Mr, tr, f, Yn)
        add("1 unweighted mean-pool", retrieve(E1, E1, te, tr, S))
        _, E2 = train("evidence", Fr, Tr, Mr, tr, f, Yn, pri)
        add("2 evidence-weighted mean-pool", retrieve(E2, E2, te, tr, S))
        _, E3 = train("deepsets", Fr, Tr, Mr, tr, f, Yn)
        add("3 Deep Sets (MLP-sum-MLP)", retrieve(E3, E3, te, tr, S))
        net4, E4 = train("attn", Fr, Tr, Mr, tr, f, Yn)
        add("4 Set Transformer", retrieve(E4, E4, te, tr, S))
        _, E5 = train("attn_uniform", Fr, Tr, Mr, tr, f, Yn)
        add("5 Set Transformer, attn FROZEN UNIFORM", retrieve(E5, E5, te, tr, S))
        _, E6 = train("mean", Fr, Tr, Mr * (Tr == 0), tr, f, Yn)
        add("6 self-only (no neighbourhood)", retrieve(E6, E6, te, tr, S))
        _, E7 = train("attn", Frr, Trr, Mrr, tr, f, Yn)
        add("7 random partners (CONTROL)", retrieve(E7, E7, te, tr, S))
        _, E8 = train("attn", Frc, Trc, Mrc, tr, f, Yn)
        add("8 degree-matched rewiring (CONTROL)", retrieve(E8, E8, te, tr, S))

        # ---- COUNTERFACTUAL SWAP. Accuracy cannot separate "the neighbourhood is useless" from "the
        # neighbourhood is IGNORED": both look like arm 4 == arm 6. Handing the TRAINED arm-4 network
        # another test knockout's neighbours, while keeping the correct self token, can.
        sw = np.random.default_rng(50 + f).permutation(len(te))
        Fs, Ts, Ms = Fr.clone(), Tr.clone(), Mr.clone()
        tei = torch.from_numpy(te)
        srci = torch.from_numpy(te[sw])
        Fs[tei, 1:] = Fr[srci, 1:]
        Ts[tei, 1:] = Tr[srci, 1:]
        Ms[tei, 1:] = Mr[srci, 1:]
        add("9 neighbour SWAP (COUNTERFACTUAL)", retrieve(embed(net4, Fs, Ts, Ms), E4, te, tr, S))
        add("10 wrong-knockout (identity CONTROL)", retrieve(E4, E4, te, tr, S)[sw])

        # ---- model-init seeds, on partition 0 ONLY, REPORTED AND NOT GRADED. This is the floor that
        # `dense_context.py` proved is the wrong one: it measures optimiser noise inside a fixed
        # partition, and the claim is about unseen knockouts.
        if f == 0:
            for kind, nm in (("attn", "4 Set Transformer"), ("mean", "1 unweighted mean-pool")):
                for sd_ in range(INIT_SEEDS):
                    _, Ee = train(kind, Fr, Tr, Mr, tr, 900 + sd_, Yn)
                    r, _c = score_arm(retrieve(Ee, Ee, te, tr, S), te, TRUTH, Yn)
                    extra.setdefault(nm, []).append(r)
            report(f"    model-init seed spread on this partition ONLY (not graded): "
                   + "; ".join(f"{k_} sd {np.std(v, ddof=1):.4f}" for k_, v in extra.items()))
        S = TRUTH = Yn = None

    # -------------------------------------------------------------------------------- aggregate
    arm_par = {"1 unweighted mean-pool": ("mean", PAR["mean"], PAR["mean"]),
               "2 evidence-weighted mean-pool": ("evidence", PAR["evidence"], PAR["evidence"]),
               "3 Deep Sets (MLP-sum-MLP)": ("deepsets", PAR["deepsets"], PAR["deepsets"]),
               "4 Set Transformer": ("attn", PAR["attn"], PAR["attn"]),
               "5 Set Transformer, attn FROZEN UNIFORM": ("attn_uniform", PAR["attn_uniform"],
                                                          PAR["attn_uniform"] - DEAD["attn_uniform"]),
               "6 self-only (no neighbourhood)": ("mean", PAR["mean"], PAR["mean"]),
               "7 random partners (CONTROL)": ("attn", PAR["attn"], PAR["attn"]),
               "8 degree-matched rewiring (CONTROL)": ("attn", PAR["attn"], PAR["attn"]),
               "9 neighbour SWAP (COUNTERFACTUAL)": ("attn", PAR["attn"], PAR["attn"]),
               "10 wrong-knockout (identity CONTROL)": ("attn", PAR["attn"], PAR["attn"])}

    def m(nm, met="rec"):
        return float(np.mean(res[nm][met]))

    report(f"\n  {'arm':44s} {'params':>9s} {'recall@50':>10s} {'sd':>7s} {'cosine':>9s} {'sd':>7s}")
    for k_, v in res.items():
        p_ = arm_par.get(k_, (None, 0, 0))
        report(f"  {k_:44s} {p_[1]:9,d} {np.mean(v['rec']):10.4f} {np.std(v['rec'], ddof=1):7.4f} "
               f"{np.mean(v['cos']):+9.4f} {np.std(v['cos'], ddof=1):7.4f}")

    # THE MDE. n counts DISJOINT KNOCKOUT PARTITIONS. See the docstring: seeds measure optimiser noise.
    def pooled_sd(met):
        return float(np.mean([np.std(v[met], ddof=1) for v in res.values()]))
    sd_rec, sd_cos = pooled_sd("rec"), pooled_sd("cos")
    MDE = 3 * sd_rec / np.sqrt(FOLDS)
    MDE_COS = 3 * sd_cos / np.sqrt(FOLDS)
    sd_seed = float(np.mean([np.std(v, ddof=1) for v in extra.values()])) if extra else float("nan")
    mde_seed = 3 * sd_seed / np.sqrt(max(INIT_SEEDS, 2))
    report(f"\n  MDE = 3*sd/sqrt(n), n = {FOLDS} DISJOINT KNOCKOUT PARTITIONS (the unit of replication -- the"
           f" claim is\n  generalisation to unseen knockouts). pooled across-partition sd {sd_rec:.4f} -> "
           f"MDE {MDE:+.4f} recall@{K}; cosine MDE {MDE_COS:+.4f}")
    report(f"  the model-init-seed sd inside partition 0 is {sd_seed:.4f} -> a {mde_seed:+.4f} floor, "
           f"{sd_rec/max(sd_seed,1e-9):.1f}x {'smaller' if sd_seed < sd_rec else 'larger'} than the "
           f"across-partition one. IT IS NOT WHAT ANYTHING HERE IS GRADED ON.")

    SIMPLE = ["1 unweighted mean-pool", "2 evidence-weighted mean-pool", "3 Deep Sets (MLP-sum-MLP)"]
    FREEA = ["A raw DepMap self-vector (0 param)", "B raw DepMap bag-of-partners (0 param)",
             "C TIDE-null floor (0 param)"]
    best_simple = max(SIMPLE, key=lambda x: m(x))
    best_free = max(FREEA, key=lambda x: m(x))
    g_simple = m("4 Set Transformer") - m(best_simple)
    g_simple_cos = m("4 Set Transformer", "cos") - m(best_simple, "cos")
    g_attn = m("4 Set Transformer") - m("5 Set Transformer, attn FROZEN UNIFORM")
    g_free = m("4 Set Transformer") - m(best_free)
    g_self = m("4 Set Transformer") - m("6 self-only (no neighbourhood)")
    g_swap = m("4 Set Transformer") - m("9 neighbour SWAP (COUNTERFACTUAL)")
    g_rand = m("4 Set Transformer") - m("7 random partners (CONTROL)")
    g_cfg = m("4 Set Transformer") - m("8 degree-matched rewiring (CONTROL)")
    gates = {
        "PRIMARY  Set Transformer beats the best of arms 1-3 by > MDE": bool(g_simple > MDE),
        "SHARP    attention beats FROZEN-UNIFORM attention by > MDE": bool(g_attn > MDE),
        "FREE     Set Transformer beats the best 0-parameter arm by > MDE": bool(g_free > MDE),
        "AGREE    the threshold-free cosine agrees in sign with recall@50":
            bool((g_simple > MDE) == (g_simple_cos > MDE_COS)),
    }
    report("\n  PREDECLARED GATES")
    for k_, v in gates.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k_}")
    report(f"    best simple encoder = '{best_simple}' at {m(best_simple):.4f}; "
           f"Set Transformer {m('4 Set Transformer'):.4f}; gap {g_simple:+.4f} vs MDE {MDE:.4f}")
    report(f"    best 0-parameter arm = '{best_free}' at {m(best_free):.4f}; gap {g_free:+.4f}")
    report(f"    attention vs frozen-uniform attention: {g_attn:+.4f}; "
           f"neighbourhood counterfactual swap: {g_swap:+.4f}; real vs degree-matched rewiring: {g_cfg:+.4f}")

    # Rule: a control that reproduces most of the effect IS the finding.
    def frac(x, whole):
        return (x / whole) if abs(whole) > 1e-9 else float("nan")
    repro = {"5 frozen-uniform attention reproduces this fraction of arm 4's gain over self-only":
             frac(m("5 Set Transformer, attn FROZEN UNIFORM") - m("6 self-only (no neighbourhood)"), g_self),
             "1 mean-pool reproduces this fraction of arm 4's gain over self-only":
             frac(m("1 unweighted mean-pool") - m("6 self-only (no neighbourhood)"), g_self),
             "8 degree-matched rewiring reproduces this fraction of arm 4's gain over self-only":
             frac(m("8 degree-matched rewiring (CONTROL)") - m("6 self-only (no neighbourhood)"), g_self),
             "B 0-parameter bag-of-partners reproduces this fraction of arm 4's gain over self-only":
             frac(m("B raw DepMap bag-of-partners (0 param)") - m("6 self-only (no neighbourhood)"), g_self)}
    report("\n  CONTROLS THAT REPRODUCE THE EFFECT (reported as the finding, not as a footnote)")
    for k_, v in repro.items():
        report(f"    {100*v:6.1f}%  {k_}")

    passed = gates["PRIMARY  Set Transformer beats the best of arms 1-3 by > MDE"]
    verdict = (
        f"{'CRITERION 2 PASSES' if passed else 'CRITERION 2 FAILS'}: on the dense untruncated K562 target "
        f"({N:,} knockouts, {A.shape[1]:,} genes, 100% density) with the task, target, metric and retrieval "
        f"held identical to dense_stage1 and ONLY the encoder varying at a matched budget of "
        f"{BUDGET:,} parameters, the Set Transformer reaches recall@{K} {m('4 Set Transformer'):.4f} while "
        f"the best simple set encoder ('{best_simple}') reaches {m(best_simple):.4f} -- a gap of "
        f"{g_simple:+.4f} against a cross-partition minimum detectable increment of {MDE:.4f} "
        f"(3*sd/sqrt(n), n = {FOLDS} DISJOINT KNOCKOUT PARTITIONS; the model-init-seed floor of "
        f"{mde_seed:.4f} measures optimiser noise inside one partition and is deliberately NOT what this "
        f"is graded on). CELLFORMER4.md section 6 retains the transformer only if it beats simpler set "
        f"encoders, so on this evidence Block 3 "
        f"{'keeps its attention layer' if passed else 'should be reimplemented as a mean-pool over typed partner tokens: the validated object is a BAG OF TYPED PARTNERS, not a transformer'}. "
        f"THE SHARP ARM separates attention from depth: freezing the attention matrix to a uniform 1/n "
        f"inside the identical module changes recall by {g_attn:+.4f}, so the attention PATTERN "
        f"{'is earning its parameters' if g_attn > MDE else 'is not detected -- whatever arm 4 has over a plain mean-pool is depth and parameters, not content-based weighting'}. "
        f"THE CHEAPEST NON-LEARNED ARM, a zero-parameter cosine retrieval on raw DepMap vectors, scores "
        f"{m(best_free):.4f} ('{best_free}'), i.e. {g_free:+.4f} below the transformer -- "
        f"{'the learned encoder is ahead of it' if g_free > MDE else 'A TRIVIAL BASELINE WITH NO PARAMETERS MATCHES OR BEATS THE MODEL, which is the headline and not a footnote'}. "
        f"Attribution of the neighbourhood itself: real vs uniformly random partners {g_rand:+.4f}, real vs "
        f"a DEGREE-MATCHED configuration-model rewiring {g_cfg:+.4f}, and the counterfactual test in which "
        f"the trained network is handed another held-out knockout's neighbours {g_swap:+.4f} -- a swap gap "
        f"at or below the MDE would mean the neighbourhood channel is IGNORED rather than useless, which "
        f"accuracy alone cannot distinguish. The threshold-free cosine metric "
        f"{'AGREES' if gates['AGREE    the threshold-free cosine agrees in sign with recall@50'] else 'DISAGREES'} "
        f"with recall@{K} on the primary gap ({g_simple_cos:+.4f} against a {MDE_COS:.4f} cosine MDE)"
        f"{'.' if gates['AGREE    the threshold-free cosine agrees in sign with recall@50'] else ', and when the two metrics disagree the result is a metric artifact and is not reportable as biology.'} "
        f"All effect sizes quoted here are RAW held-out values; the control arms establish significance and "
        f"attribution and nothing is reported net of a control.")
    report("\n" + "=" * 104)
    report(f"  VERDICT: {verdict}")

    R = {"model": "v4-set-encoders-v1",
         "question": "CELLFORMER4.md sec 6 / criterion 2 -- does a simple set encoder match the "
                     "transformer? Arms 6-8 of sec 6's mandatory control list, never previously run.",
         "task": {"source": "colab/dense_stage1.py, unchanged",
                  "data": "gwps.h5ad dense K562 pseudobulk logFC, 100% density (NOT the nlz_* benches, "
                          "which are 3.5% dense and whose zeros mean NOT RECORDED)",
                  "target": "S = |z| - per-gene mean |z| over TRAINING knockouts, refit per partition",
                  "truth": f"top {TRUTH_N} non-tide genes by centred |z|",
                  "primary_metric": f"recall@{K} after nearest-{N_RETRIEVE} training-knockout retrieval",
                  "secondary_metric": "threshold-free cosine to the centred profile",
                  "n_knockouts": int(N), "n_genes": int(A.shape[1]),
                  "random_floor_recall": float(K / A.shape[1]),
                  "only_thing_that_varies": "the encoder"},
         "parameter_budget": {"budget": int(BUDGET), "per_kind_hidden_width": WID,
                              "per_kind_params": {k_: int(v) for k_, v in PAR.items()},
                              "note": "hidden widths bisected so every learned arm lands on the Set "
                                      "Transformer's count; arm 5 carries "
                                      f"{DEAD['attn_uniform']:,} q/k parameters that receive no gradient "
                                      "when the attention matrix is frozen, so it is DEPTH-matched and "
                                      "total-parameter-matched but not effective-parameter-matched"},
         "arms": {k_: {"recall_mean": float(np.mean(v["rec"])),
                       "recall_sd": float(np.std(v["rec"], ddof=1)),
                       "recall_per_seed": v["rec"],
                       "cosine_mean": float(np.mean(v["cos"])),
                       "cosine_sd": float(np.std(v["cos"], ddof=1)),
                       "cosine_per_seed": v["cos"],
                       "params": int(arm_par.get(k_, (None, 0, 0))[1]),
                       "params_receiving_gradient": int(arm_par.get(k_, (None, 0, 0))[2])}
                  for k_, v in res.items()},
         "unit_of_replication": f"a DISJOINT KNOCKOUT PARTITION. The per_seed lists above hold one value "
                                f"per held-out partition (n = {FOLDS}), NOT model-init seeds.",
         "mde": {"value": float(MDE), "convention": "3*sd/sqrt(n)",
                 "n": int(FOLDS), "n_counts": "disjoint held-out knockout partitions",
                 "pooled_across_partition_sd": float(sd_rec),
                 "cosine_mde": float(MDE_COS), "pooled_across_partition_sd_cosine": float(sd_cos),
                 "init_seed_sd_within_partition_0_DO_NOT_GRADE_ON_THIS": float(sd_seed),
                 "init_seed_mde_DO_NOT_GRADE_ON_THIS": float(mde_seed),
                 "init_seed_per_arm": {k_: v for k_, v in extra.items()}},
         "gates": {k_: bool(v) for k_, v in gates.items()},
         "gaps": {"transformer_minus_best_simple": float(g_simple),
                  "best_simple_arm": best_simple,
                  "transformer_minus_frozen_uniform_attention": float(g_attn),
                  "transformer_minus_best_zero_parameter_arm": float(g_free),
                  "best_zero_parameter_arm": best_free,
                  "transformer_minus_self_only": float(g_self),
                  "transformer_minus_random_partners": float(g_rand),
                  "transformer_minus_degree_matched_rewiring": float(g_cfg),
                  "transformer_minus_neighbour_swap_counterfactual": float(g_swap)},
         "controls_reproducing_the_effect": {k_: (float(v) if np.isfinite(v) else None)
                                             for k_, v in repro.items()},
         "evidence_prior_per_partition": priors_log,
         "circularity_check": "token features are DepMap CRISPR gene-effect PCs and the co-dependency "
                              "edges are DepMap-derived; the scored target is Replogle Perturb-seq K562 "
                              "transcriptomics. No feature is scored on the dataset it derives from. All "
                              "target-side statistics (centring, truth sets, evidence prior) are refit on "
                              "the training partition inside every fold.",
         "epochs": EPOCHS, "folds": FOLDS, "smoke": bool(SMOKE),
         "limits": [
             f"n = {FOLDS} partitions gives the sd {FOLDS-1} degrees of freedom, so the MDE is itself "
             f"poorly estimated and a gap just under it is not evidence of equality. Raising V4_FOLDS is "
             f"the cheapest way to falsify or confirm a near-threshold call",
             "a null here is a null for THIS encoder at THIS budget on THIS retrieval objective (Gram "
             "matching on centred |z| profiles). A deeper stack, more heads, or a contrastive objective "
             "could still separate the arms; what is falsified is the claim as currently made, that the "
             "single attention block in Block 3 is doing work a mean-pool cannot",
             "at most 24 tokens and at most 7 partners per relation type. Attention has more to do on "
             "large sets than small ones, so this is a conservative regime FOR the mean-pool arms and the "
             "result should not be extrapolated to a wider partner set without rerunning",
             "arm 5 keeps its q and k parameters but they receive no gradient, so it has fewer EFFECTIVE "
             "parameters than arm 4. That biases the comparison in the transformer's favour: if arm 4 "
             "beats arm 5, the cause is ambiguous between attention and those parameters, whereas if arm 4 "
             "does NOT beat arm 5 the conclusion that attention adds nothing is strengthened",
             "the tide definition and the mover frequency that the TIDE-null floor ranks by are computed "
             "over all knockouts including the held-out ones, inherited unchanged from dense_stage1 so the "
             "numbers stay comparable. It is a leak shared identically by every arm and it flatters the "
             "floor arm most",
             "K562 only, one assay, one time point. Nothing here speaks to whether an attention layer is "
             "needed once cell context, dose or time enter the token set",
             "the retrieval head is a nearest-neighbour copier, so every arm is capped by the dense ORACLE "
             "(0.0526 in dense_stage1). Encoders can only be separated up to that ceiling and differences "
             "compress against it",
             "the evidence prior for arm 2 is estimated only from partners that are themselves knockouts "
             "in the screen, which is a biased subset of partners, and it is a single scalar per relation "
             "type -- a richer prior is a different arm and was not run"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "v4_set_encoders.json", "w"), indent=1)
    report(f"\n  -> {OUT/'v4_set_encoders.json'}")


if __name__ == "__main__":
    main()
