"""STAGE 1 + 2 -- does the arc's one confirmed win survive an untruncated target, and what is the real ceiling?

WHAT IS BEING RETESTED AND WHY IT HAS TO BE. Every transformer in this project's arc trained against the
`nlz_*` benches, which store a median of 250 genes per knockout out of 7,223: 349,999 recorded values in a
10,112,200-cell matrix, 3.5% dense, with the other 96.5% being zeros that mean NOT RECORDED and are consumed
as real zeros. The retrieval target -- cosine between tide-removed response profiles -- was therefore a
cosine between vectors that are almost entirely structural zeros, and two knockouts resemble each other
partly because both are mostly zeros.

The one control-confirmed result in the whole arc was that ATTENDING OVER REAL INTERACTION PARTNERS BEATS
RANDOM ONES by +0.335 recall@50. That is an internal, like-for-like comparison, so the truncation does not
obviously invalidate it -- but it has never been measured on a target where the zeros are real. This does
that, and it also reports the ORACLE, which is stage 2: the 0.607 ceiling everything in the arc was measured
against is a ceiling of the truncation and is void.

THE DENSE TARGET. `gwps.h5ad` is the K562 deposit as a complete pseudobulk log fold change, 11,258
perturbations x 8,248 genes, no truncation -- 8x the knockouts of the bench and 100% density. It has been on
disk the entire time. Conventions are taken from `eval_harness` unchanged so the numbers are comparable:
per-gene robust z, |z| >= 1.0 is a mover, the top 5% most-frequently-moving genes are the tide, a knockout
needs >= 5 specific movers to be scorable, recall@50.

WHAT CHANGES AND WHAT DOES NOT. The task, the metric, the controls and the model are identical to
`cellformer_v1`. Only the target changes. So any difference is attributable to the truncation and to the 8x
larger knockout set, and nothing else.

THE METRIC HAD TO BE REDEFINED FOR A DENSE TARGET, AND HERE IS THE MEASUREMENT THAT FORCED IT. Carrying the
bench's rule over verbatim -- truth = every non-tide gene with |z| >= 1 -- gives a knockout a MEDIAN OF 2,328
TRUE SPECIFIC MOVERS on dense data (q25 1,689, q75 3,246) instead of the bench's handful. recall@50 against a
truth set of 2,328 cannot exceed 50/2328 = 0.021 no matter what the model does. A first run of this file
produced 0.0060 for every arm and 0.0075 for the ORACLE, which is that arithmetic ceiling and not a result;
it is NOT reported as a stage 1 failure because it was not a measurement of anything.

The truth set is therefore fixed at the TRUTH_N = 50 most specifically-moved non-tide genes, matched to K,
so recall@50 lands on a [0,1] scale with a 50/8,175 = 0.006 random floor. A second, threshold-free metric --
cosine between the predicted gene score vector and the true profile -- is reported alongside it, because a
metric artifact in one should show up as the two disagreeing.

AND THE TARGET ITSELF HAD TO BE CENTRED, WHICH THE SAME SECOND METRIC IS WHAT CAUGHT. With Y = |z| masked to
non-tide genes, every arm scored cosine ~0.77 including the tide floor, and the ORACLE scored LOWER (0.647)
than the floor. That ordering is only possible when the target is dominated by a component every knockout
shares: the best single guess is then the mean, and a genuine nearest neighbour is merely a noisier version
of it. Because that same Y was also the training Gram target and the oracle's retrieval key, the model had a
near-constant matrix to fit and the oracle was choosing an almost arbitrary neighbour. Masking out the tide
GENES is not the same as removing the tide COMPONENT. Everything now runs on S = |z| minus that gene's mean
|z| over the TRAINING knockouts, refit inside every split.

    Absolute recall is still NOT comparable to the 0.335 / 0.607 numbers this project has quoted. The bench's
    truth sets were variable-sized AND censored. The like-for-like quantity is the GAP BETWEEN ARMS measured
    within one target, which is what stage 1 is judged on and always was.

THE ARMS
    TIDE-null            rank non-tide genes by how often they move. The floor.
    self-only            the perturbed gene's own features, no neighbourhood.
    FULL                 self + real typed neighbours (co-dependency, complex, PPI).
    random partners      THE DECISIVE CONTROL: same token count, same type mix, wrong partners.
    wrong-knockout       identity control: another test knockout's context.
    ORACLE               copy the single best-matching TRAINING knockout's response. Stage 2.

PREDECLARED GATE. Stage 1 passes only if FULL beats random partners by more than three seed spreads, on the
primary metric. That is the comparison the arc's single confirmed result rests on, and it is the only thing
being asked here.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
DEPVEC = OUT / "depmap_vecs.npz"

TAU = 1.0            # |z| at which a gene counts as moved -- eval_harness convention, unchanged
TIDE_FRAC = 0.05     # top 5% most-frequently-moving genes are the tide -- unchanged
MIN_SPEC = 5         # a knockout needs this many specific movers to be scorable -- unchanged
K = 50
TRUTH_N = 50         # truth = the TRUTH_N largest-|z| non-tide genes. Matched to K. See the header: the
                     # bench's threshold rule gives a median truth set of 2,328 genes here and caps
                     # recall@50 at 0.021 by arithmetic.
MAX_TOK = 24
D_MODEL = 128
EPOCHS = int(os.environ.get("S1_EPOCHS", "25"))
N_SEEDS = int(os.environ.get("S1_SEEDS", "3"))
SMOKE = os.environ.get("S1_SMOKE", "0") == "1"


def robust_z(M):
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad


def main():
    import gzip
    import h5py
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("STAGE 1+2 -- the confirmed win, and the real ceiling, on an UNTRUNCATED target")
    report("=" * 100)

    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["var"]["__categories"]["gene_name"][:]]
        gname = np.array([cats[i] for i in f["var"]["gene_name"][:]])
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    report(f"  dense K562: {X.shape[0]:,} perturbations x {X.shape[1]:,} genes, "
           f"{100*np.isfinite(X).mean():.1f}% finite")
    report(f"  the bench this replaces: 1,400 x 7,223 at 3.5% density")

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
    report(f"  after dropping genes with an undefined z: {A.shape[1]:,} genes; "
           f"{int(tide.sum()):,} tide, {int(nontide.sum()):,} non-tide")
    spec_n = (mover & nontide[None, :]).sum(1)
    scorable = spec_n >= MIN_SPEC
    report(f"  scorable knockouts (>= {MIN_SPEC} specific movers): {int(scorable.sum()):,}/{len(psym):,}")
    report(f"  specific movers per knockout at |z| >= {TAU}: median {np.median(spec_n):,.0f}  "
           f"q25 {np.percentile(spec_n, 25):,.0f}  q75 {np.percentile(spec_n, 75):,.0f}")
    report(f"  ^ THIS is why the bench's threshold rule cannot be carried over: recall@{K} against a truth "
           f"set of {np.median(spec_n):,.0f} is capped at {K/np.median(spec_n):.4f}.")
    report(f"  truth is therefore the top {TRUTH_N} non-tide genes by TRAIN-CENTRED |z|; random floor = "
           f"{K/A.shape[1]:.4f}")
    del mover

    dv = np.load(DEPVEC, allow_pickle=True)
    syms = [str(x) for x in dv["syms"]]
    Zd = dv["Z"].astype(np.float32)
    U, s_, _ = np.linalg.svd(Zd - Zd.mean(0), full_matrices=False)
    Zp = (U[:, :64] * s_[:64]).astype(np.float32)
    srow = {g: i for i, g in enumerate(syms)}

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
    if SMOKE:
        keep = keep[:800]
    report(f"  usable: {len(keep):,} knockouts with a DepMap profile AND >= {MIN_SPEC} specific movers "
           f"(the bench gave 1,385)")
    A = A[keep]
    psym_k = psym[keep]
    # MASKING OUT THE TIDE GENES IS NOT THE SAME AS REMOVING THE TIDE, and on a dense target the difference
    # decides the experiment. A first version used Y = |z| * nontide. Every arm scored cosine ~0.77 against
    # it, INCLUDING the tide floor, and the ORACLE scored LOWER (0.647) than the floor -- the signature of a
    # target dominated by a component every knockout shares, where the best single guess is the mean and a
    # genuine nearest neighbour is merely noisier than the mean. That same Y was the training target (Gram
    # matching) and the oracle's retrieval key, so a near-constant Gram matrix left the model nothing to
    # learn and left the oracle choosing an almost arbitrary neighbour. This is the collapse-to-the-tide
    # that `neural_ko` already documented in regression form.
    #
    # Everything therefore moves into per-gene centred space: S = |z| - (that gene's mean |z| over the
    # TRAINING knockouts only). A gene scores when it moves MORE THAN IT USUALLY DOES for a knockout, which
    # is what "specific mover" was always supposed to mean. Centring is refit inside every split, so the
    # test knockouts never contribute to their own baseline.
    def centred(tr_idx):
        mu = A[tr_idx].mean(0)
        Sz = np.where(nontide[None, :], A - mu[None, :], 0.0).astype(np.float32)
        # Chunked: a full argsort of an (n, 8175) matrix returns an int64 array of the same shape, which is
        # 8x the float32 data. argpartition per block keeps the peak bounded. Order inside the truth set is
        # irrelevant -- it is intersected as a set.
        T_ = np.empty((len(Sz), TRUTH_N), np.int64)
        for i0 in range(0, len(Sz), 512):
            blk = np.where(nontide[None, :], Sz[i0:i0 + 512], -np.inf)
            T_[i0:i0 + 512] = np.argpartition(-blk, TRUTH_N, axis=1)[:, :TRUTH_N]
        Yn_ = Sz / (np.linalg.norm(Sz, axis=1, keepdims=True) + 1e-9)
        return Sz, T_, Yn_

    def tokens(mode, rng):
        allg = list(srow)
        F = np.zeros((len(psym_k), MAX_TOK, Zp.shape[1]), np.float32)
        T = np.zeros((len(psym_k), MAX_TOK), np.int64)
        M = np.zeros((len(psym_k), MAX_TOK), bool)
        for i, k_ in enumerate(psym_k):
            rows, types = [k_], [0]
            for t, lst in ((1, codep.get(k_, [])), (2, cplx.get(k_, [])), (3, ppi.get(k_, []))):
                pick = lst[:7] if mode == "real" else [allg[int(x)] for x in
                                                       rng.integers(0, len(allg), min(7, len(lst)))]
                for g in pick:
                    rows.append(g)
                    types.append(t)
            for j, (g, t) in enumerate(zip(rows[:MAX_TOK], types[:MAX_TOK])):
                if g in srow:
                    F[i, j] = Zp[srow[g]]
                T[i, j] = t
                M[i, j] = True
        return torch.from_numpy(F), torch.from_numpy(T), torch.from_numpy(M)

    rng = np.random.default_rng(0)
    Fr, Tr, Mr = tokens("real", rng)
    Frr, Trr, Mrr = tokens("random", np.random.default_rng(1))
    report(f"  tokens per knockout: mean {Mr.sum(1).float().mean():.1f} real, "
           f"{Mrr.sum(1).float().mean():.1f} random")

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(Zp.shape[1], D_MODEL)
            self.rel = nn.Embedding(4, D_MODEL)
            self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
            self.ln = nn.LayerNorm(D_MODEL)
            self.out = nn.Linear(D_MODEL, D_MODEL)

        def forward(self, F, T, M):
            h = self.proj(F) + self.rel(T)
            a, _ = self.attn(h, h, h, key_padding_mask=~M, need_weights=False)
            h = self.ln(h + a)
            m = M.unsqueeze(-1).float()
            p = (h * m).sum(1) / m.sum(1).clamp(min=1)
            e = self.out(p)
            return e / (e.norm(dim=-1, keepdim=True) + 1e-9)

    def train(F, T, M, tr, seed):
        torch.manual_seed(seed)
        net = Enc()
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
            return net(F, T, M).numpy()

    def score_arm(scores, te):
        """Two metrics on the same score matrix, so a metric artifact shows up as the two disagreeing.

        recall@K  -- overlap with the fixed-size truth set. Bounded [0,1], random floor K/G.
        cosine    -- threshold-free agreement with the full non-tide |z| profile. No K, no truth set.
        """
        rec, cos = [], []
        for j, i in enumerate(te):
            # Centred scores are mostly negative (|z| is right-skewed, so the mean sits above the median),
            # so a tide gene left at 0.0 would outrank most real candidates. Rank inside the non-tide
            # universe only -- the same rule the truth set is built under.
            s = np.where(nontide, scores[j], -np.inf)
            top = np.argpartition(-s, K)[:K]
            rec.append(np.intersect1d(top, TRUTH[i]).size / TRUTH_N)
            v = np.where(nontide, scores[j], 0.0).astype(np.float64)
            nv = np.linalg.norm(v)
            cos.append(float(v @ Yn[i] / nv) if nv > 1e-9 else 0.0)
        return float(np.mean(rec)), float(np.mean(cos)), len(rec)

    res = {}
    S = TRUTH = Yn = None
    for seed in range(N_SEEDS):
        p = np.random.default_rng(100 + seed).permutation(len(keep))
        te, tr = p[:len(p) // 4], p[len(p) // 4:]
        S = TRUTH = Yn = None            # free the previous split's arrays before allocating the next
        S, TRUTH, Yn = centred(tr)
        report(f"\n  SPLIT {seed}: {len(tr):,} train / {len(te):,} test; "
               f"train-only centring, target Gram off-diagonal mean "
               f"{float((Yn[tr[:400]] @ Yn[tr[:400]].T).mean()):.4f} "
               f"(it was 0.77 uncentred, which is what left nothing to learn)")

        def add(name, sc):
            r, c, n = score_arm(sc, te)
            d = res.setdefault(name, {"rec": [], "cos": []})
            d["rec"].append(r)
            d["cos"].append(c)
            report(f"    {name:34s} recall@{K} {r:.4f}   cosine {c:.4f}  (n={n})")

        # THE FLOOR, AND IT MUST BE SCORED IN THE SAME UNIVERSE AS THE TRUTH. Truth is restricted to
        # non-tide genes; a baseline that ranks by tide-ness therefore selects exactly the excluded genes
        # and scores ~0 by construction. That defect has appeared twice in this project already, so the
        # floor here ranks by mover frequency WITHIN the non-tide universe.
        floor = np.where(nontide, freq, -np.inf)
        add("TIDE-null (floor)", np.tile(floor, (len(te), 1)))
        E = train(Fr, Tr, Mr, tr, seed)
        Es = train(Fr, Tr, Mr * (Tr == 0), tr, seed)
        Erv = train(Frr, Trr, Mrr, tr, seed)

        def retrieve(E_):
            En = E_ / (np.linalg.norm(E_, axis=1, keepdims=True) + 1e-9)
            sim = En[te] @ En[tr].T
            nn_ = np.argsort(-sim, axis=1)[:, :10]
            return np.stack([S[tr[r]].mean(0) for r in nn_])
        add("self-only (no neighbourhood)", retrieve(Es))
        add("FULL (real neighbours)", retrieve(E))
        add("random partners (CONTROL)", retrieve(Erv))
        sh = np.random.default_rng(7 + seed).permutation(len(te))
        add("wrong-knockout (CONTROL)", retrieve(E)[sh])
        # The oracle retrieves on the TRUE centred profile -- the best a nearest-neighbour copier could do
        # if its embedding were perfect. It is a ceiling on this task, not on the biology.
        sim = Yn[te] @ Yn[tr].T
        add("ORACLE (dense ceiling)", np.stack([S[tr[int(np.argmax(s))]] for s in sim]))

    report(f"\n  {'arm':34s} {'recall@50':>10s} {'sd':>7s}    {'cosine':>8s} {'sd':>7s}")
    for k_, v in res.items():
        report(f"  {k_:34s} {np.mean(v['rec']):10.4f} {np.std(v['rec']):7.4f}    "
               f"{np.mean(v['cos']):8.4f} {np.std(v['cos']):7.4f}")

    out = {}
    for m in ("rec", "cos"):
        sd = float(np.mean([np.std(v[m]) for v in res.values()]))
        mde = 3 * sd / np.sqrt(N_SEEDS)
        gap = float(np.mean(res["FULL (real neighbours)"][m])
                    - np.mean(res["random partners (CONTROL)"][m]))
        out[m] = {"sd": sd, "mde": mde, "gap": gap, "passed": bool(gap > mde)}
        report(f"  {m}: pooled seed sd {sd:.4f} -> MDE {mde:+.4f} at {N_SEEDS} seeds; "
               f"FULL - random = {gap:+.4f} -> {'PASS' if gap > mde else 'FAIL'}")

    passed = out["rec"]["passed"]
    agree = out["rec"]["passed"] == out["cos"]["passed"]
    verdict = (
        f"{'STAGE 1 PASSES' if passed else 'STAGE 1 FAILS'}: on an UNTRUNCATED target the real-neighbour "
        f"arm beats random partners by {out['rec']['gap']:+.4f} recall@{K} against a "
        f"{out['rec']['mde']:.4f} minimum detectable increment, on {len(keep):,} knockouts at 100% "
        f"density. The threshold-free cosine metric {'AGREES' if agree else 'DISAGREES'} "
        f"({out['cos']['gap']:+.4f} against {out['cos']['mde']:.4f})"
        f"{'.' if agree else ' -- when the two metrics disagree the result is a metric artifact and is not reportable as biology.'} "
        f"STAGE 2, the ceiling: the dense ORACLE is "
        f"{np.mean(res['ORACLE (dense ceiling)']['rec']):.4f} and the tide floor is "
        f"{np.mean(res['TIDE-null (floor)']['rec']):.4f}. "
        f"NEITHER absolute number is comparable to the 0.335 gap and 0.607 oracle this project quoted "
        f"from the truncated bench: that bench's truth sets were variable-sized AND 96.5% censored, and "
        f"the truth set here is a fixed top-{TRUTH_N}. The like-for-like quantity across the two targets "
        f"is the gap between arms measured inside one target, and that is what stage 1 is graded on.")
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {"model": "dense-stage1-v1", "n_knockouts": int(len(keep)), "n_genes": int(A.shape[1]),
         "density": 1.0, "replaces": "nlz benches at 3.5% density, 1,385 usable knockouts",
         "metric": {"primary": f"recall@{K} against the top-{TRUTH_N} non-tide genes by train-centred |z|",
                    "secondary": "cosine to the full centred profile, threshold-free",
                    "target": "S = |z| - per-gene mean |z| over TRAINING knockouts, refit per split",
                    "random_floor_recall": float(K / A.shape[1]),
                    "why_redefined": "the bench's |z|>=1 rule gives a median of 2,328 true specific movers "
                                     f"on dense data, capping recall@{K} at 0.021 by arithmetic",
                    "why_centred": "with an uncentred |z| target every arm scored cosine ~0.77 and the "
                                   "ORACLE scored BELOW the tide floor -- a target dominated by a shared "
                                   "component, which also left the training Gram matrix near-constant"},
         "arms": {k_: {"recall_mean": float(np.mean(v["rec"])), "recall_sd": float(np.std(v["rec"])),
                       "cosine_mean": float(np.mean(v["cos"])), "cosine_sd": float(np.std(v["cos"])),
                       "recall_per_seed": v["rec"], "cosine_per_seed": v["cos"]}
                  for k_, v in res.items()},
         "real_vs_random": out, "stage1_passed": bool(passed), "metrics_agree": bool(agree),
         "dense_oracle_recall": float(np.mean(res["ORACLE (dense ceiling)"]["rec"])),
         "truncated_oracle_quoted_previously": 0.607,
         "limits": [
             "absolute recall is not comparable between the truncated and dense targets: different truth "
             "set definitions AND different censoring. Only the between-arm gap is like-for-like",
             "K562 only. Stage 6 is the cross-cell test and this is not it",
             f"TRUTH_N = {TRUTH_N} is a choice. It is matched to K so the metric is on a [0,1] scale, but "
             "no threshold on dense data reproduces the bench's truth-set sizes and the |z| threshold "
             "TAU is used only for the scorability filter, which at this density passes every knockout",
             "the encoder here is a single attention block, smaller than cellformer_v1's, chosen so the "
             "8x larger knockout set fits the compute budget. A capacity difference is therefore "
             "confounded with the target change for the ABSOLUTE numbers, though not for the "
             "real-vs-random gap, which is measured within one architecture"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "dense_stage1.json", "w"), indent=1)
    report(f"\n  -> {OUT/'dense_stage1.json'}")


if __name__ == "__main__":
    main()
