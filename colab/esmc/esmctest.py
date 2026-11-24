"""ESMCTEST (G10) -- what the protein language model ESM C adds to the cell model: addresses read from the whole
sequence, partners that fit, and RNA versions that move a protein. Runs on a GPU (Google Colab) from a fixed input
bundle; written, committed and declared before any real run.

Asked by the user: "we can do it on colab, we have gpu there. so make a notebook to run the test there."
MODEL: ESM C 600M (Biohub; huggingface biohub/ESMC-600M at a fixed revision; MIT + third-party notice). Each protein ->
one fingerprint = mean over its residues of the last hidden layer (proteins longer than 2,046 residues are cut into
equal windows; the mean is over all residues). On DEV proteins the fingerprints are standardised and reduced to 256
principal components (z). Other models (e.g. biohub/ESMC-6B on a large GPU) can be run with --model; such runs are
exploratory and do not count as the declared test.
BUNDLE (colab/esmc/bundle, sha256 in colab/esmc/manifest.json; built by rem/atlas/esmcbundle.py): UniProt 2026_04
reviewed human sequences with isoforms (CC BY 4.0); address tags (UniProt); rooms by imaging (HPA CC BY-SA 3.0,
OpenCell CC BY-SA 4.0) as four classes (nuclear, secretory, mitochondria, cytoplasm); interactions HPA, OpenCell,
Complex Portal (CC0); the G9 partner-finder shortlists (first 500 candidates in G9 order for each EXAM protein whose
known partner is among them, with the G9 compatibility score and destination / made-together flags).
Genes DEV / EXAM by sha256('cellbrain-c1:' + Ensembl) parity. Everything is fit on DEV and scored once on EXAM.

CLAIMS (ids G10-*):
 G10-E1 ADDRESS FROM THE WHOLE SEQUENCE. One logistic regression per room class, fit on DEV proteins (L2; strength
        chosen by 5-fold cross-validation on DEV from 0.01, 0.1, 1) with (a) the tags (signal peptide, transit peptide,
        any transmembrane segment, log(1 + number of segments), GPI, NLS, NES, ER-retention, PTS1, log length) and
        (b) the tags + z. Per EXAM protein with a room, gain = mean over the four classes of log-loss(a) - log-loss(b).
        CLAIM: (b) BETTER (sign test p < 0.05, median gain >= +0.01). Controls: z shuffled between proteins must not
        be BETTER; a planted column (the nuclear label + unit noise) must be BETTER.
 G10-E2 HIDDEN ADDRESSES. EXAM proteins with a room and no tag at all; for each class with >= 30 such proteins in it
        and >= 30 not: AUROC of a z-only model fit on DEV. CLAIM: mean AUROC >= 0.70 and Monte Carlo p < 0.05
        (room labels shuffled among these proteins, 1,000 draws). Controls: a model on shuffled z must not pass; a
        planted score (label + unit noise) must pass.
 G10-P1 PARTNERS THAT FIT. A pair model on z: score(a, b) = <W z_a, W z_b> + u.(z_a + z_b) + c, W 256 x 64, trained on
        DEV-DEV interactions (HPA, OpenCell, Complex Portal) against random DEV-DEV pairs (10 per interaction, redrawn
        each epoch; 8 epochs, Adam 0.003). For each EXAM protein whose G9 shortlist holds a known partner, the G9
        compatibility score gets one more rank term -- the rank of the pair-model score among all 20,082 candidates --
        and the shortlist is re-ordered with the G9 rule (compatibility + 1e6 x same destination + 1e3 x made together).
        Gain = log2(tries to the first known partner in G9 order / tries after re-ordering). CLAIM: BETTER. Controls:
        a pair model trained on shuffled z must not be BETTER; a planted score (known partner + unit noise) must be
        BETTER. Reported: the pair model alone over all candidates (partner within 10 tries vs random) and its AUROC
        on EXAM-EXAM interactions vs random EXAM-EXAM pairs.
 G10-S1 RNA VERSIONS THAT MOVE THE PROTEIN. EXAM genes with isoforms whose tag changes are all named by number: for
        each isoform, shift = sum over classes of |p(isoform) - p(canonical)| from the z-only address model of E2.
        Isoforms whose alternative sequence touches the tag vs the other isoforms, inside five strata of relative
        length change |len(isoform) - len(canonical)| / len(canonical). CLAIM: length-stratified AUROC >= 0.60 and Monte
        Carlo p < 0.05 (labels shuffled inside strata, 1,000 draws). Controls: fingerprints shuffled between the
        sequences must not pass; a planted shift (label + unit noise) must pass. Reported: a catalogue of the
        tag-changing isoforms of all genes with the predicted room class of the canonical and of the isoform.
NOT: fingerprints read the sequence only (no cell type, amount or timing); known interactions are incomplete, so a
"miss" may be a true unknown partner; the re-ordering only acts inside the first 500 G9 candidates; imaging gives
rooms per gene, not per isoform; the 6B model is not the declared test.

Run: python esmctest.py --bundle DIR --out DIR [--model biohub/ESMC-600M] [--trial]
"""

from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261124
CLASSES = ["nuclear", "secretory", "mito", "cyto"]
TAGS = ["signal", "transit", "tm", "gpi", "nls", "nes", "er_ret", "pts1"]
DECLARED_MODEL = "biohub/ESMC-600M"
REVISIONS = {"biohub/ESMC-600M": "5e366569e675ad5ae3512f66bf0b53ffe80ee83e", "biohub/ESMC-6B": "94605596759c3216963f6b7e5ff7adc1994c95f0",
             "biohub/ESMC-300M": "96a2d8d61654ac0c08881be44da6c867686449b9"}
MAXR = 2046            # residues per window (2,048 tokens with the start and end tokens)
NPC, KW = 256, 64      # principal components; pair-model width
CS = (0.01, 0.1, 1.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "esmctest.py"), "rb").read()); h.update(open(os.path.join(HERE, "manifest.json"), "rb").read())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ bundle
def load_bundle(bdir):
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    for f, want in man["files"].items():
        got = hashlib.sha256(open(os.path.join(bdir, f), "rb").read()).hexdigest()
        if got != want:
            raise SystemExit(f"bundle file {f} does not match the manifest ({got[:12]} vs {want[:12]})")
    prot = list(csv.DictReader(gzip.open(os.path.join(bdir, "proteins.tsv.gz"), "rt"), delimiter="\t"))
    gj = json.load(gzip.open(os.path.join(bdir, "genes.json.gz"), "rt"))
    pairs = [(r["a"], r["b"]) for r in csv.DictReader(gzip.open(os.path.join(bdir, "ppi.tsv.gz"), "rt"), delimiter="\t")]
    sl = dict(np.load(os.path.join(bdir, "shortlist.npz"), allow_pickle=False))
    return man, prot, gj["genes"], gj["g"], pairs, sl


# ------------------------------------------------------------------------------------------------ fingerprints
def embed(seqs, model_name, cache, budget=None):
    """mean over residues of the last hidden layer; returns float32 [n, d], number of non-finite rows."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()                                   # the unused masked-LM head is reported as "unexpected"; not needed here
    rev = REVISIONS.get(model_name); cuda = torch.cuda.is_available(); big = "6B" in model_name
    if cuda:
        half = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        wdt = half if big else torch.float32                       # 6B weights in half precision; others full + autocast
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        budget = budget or ((4096 if mem < 30 else 8192 if mem < 60 else 16384) if big else (16384 if mem < 20 else 32768 if mem < 50 else 65536))
    else:
        half = None; wdt = torch.float32; budget = budget or 4096
    dev = "cuda" if cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, revision=rev); tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_name, revision=rev, dtype=wdt).to(dev).eval()
    d = model.config.hidden_size
    chunks = []
    for i, s in enumerate(seqs):
        n = max(1, -(-len(s) // MAXR)); e = np.linspace(0, len(s), n + 1).round().astype(int)
        chunks += [(i, int(a), int(b)) for a, b in zip(e[:-1], e[1:])]
    order = sorted(range(len(chunks)), key=lambda c: chunks[c][2] - chunks[c][1])
    key = hashlib.sha256((model_name + str(rev) + "|".join(seqs)).encode()).hexdigest()[:16]
    sums = np.zeros((len(seqs), d)); cnt = np.zeros(len(seqs)); done = np.zeros(len(chunks), bool)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            sums, cnt, done = z["sums"], z["cnt"], z["done"]; log(f"resuming: {int(done.sum())} of {len(chunks)} pieces already done")
    todo = [c for c in order if not done[c]]; tot_res = sum(chunks[c][2] - chunks[c][1] for c in todo)
    log(f"fingerprints: {len(seqs)} sequences, {len(chunks)} pieces, {tot_res / 1e6:.1f} M residues to read with {model_name} on {dev}"
        f" ({'weights ' + str(wdt).split('.')[-1]}{', autocast ' + str(half).split('.')[-1] if (cuda and not big) else ''}; batch budget {budget} tokens)")
    t0 = time.time(); last = t0; lastsave = t0; res_done = 0; p = 0; nonfinite = 0
    while p < len(todo):
        L = chunks[todo[p]][2] - chunks[todo[p]][1]; q = p
        while q < len(todo) and (q - p + 1) * (chunks[todo[q]][2] - chunks[todo[q]][1] + 2) <= max(budget, L + 2):
            q += 1
        q = max(q, p + 1); batch = todo[p:q]
        strs = [seqs[chunks[c][0]][chunks[c][1]:chunks[c][2]] for c in batch]
        try:
            enc = tok(strs, return_tensors="pt", padding=True); ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=half, enabled=bool(cuda and not big)):
                    h = model(input_ids=ids, attention_mask=am).last_hidden_state
                m = am.clone(); m[:, 0] = 0; m[torch.arange(len(m), device=dev), am.sum(1) - 1] = 0
                v = (h.float() * m[..., None]).sum(1)
                if not torch.isfinite(v).all() and cuda and not big:
                    h = model(input_ids=ids, attention_mask=am).last_hidden_state; v = (h.float() * m[..., None]).sum(1)
                v = v.cpu().numpy(); n = m.sum(1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            del enc; torch.cuda.empty_cache(); budget = max(budget // 2, 512); log(f"out of GPU memory: batch budget now {budget} tokens"); continue
        bad = ~np.isfinite(v).all(1); nonfinite += int(bad.sum())
        for j, c in enumerate(batch):
            i = chunks[c][0]
            if not bad[j]:
                sums[i] += v[j]; cnt[i] += n[j]
            done[c] = True; res_done += chunks[c][2] - chunks[c][1]
        p = q; now = time.time()
        if now - last > 60 or p == len(todo):
            rate = res_done / max(now - t0, 1e-9); log(f"  {res_done / 1e6:.2f} / {tot_res / 1e6:.2f} M residues; about {(tot_res - res_done) / max(rate, 1e-9) / 60:.0f} min left"); last = now
        if cache and (now - lastsave > 300 or p == len(todo)):
            np.savez(cache + ".tmp.npz", key=key, sums=sums, cnt=cnt, done=done); os.replace(cache + ".tmp.npz", cache); lastsave = now
    with np.errstate(invalid="ignore", divide="ignore"):
        E = (sums / cnt[:, None]).astype(np.float32)
    return E, nonfinite, len(chunks)


# ------------------------------------------------------------------------------------------------ statistics
def binom_greater(x):
    from scipy.stats import binomtest
    x = np.asarray(x, float); x = x[np.isfinite(x)]; nz = x[x != 0]
    p = float(binomtest(int((nz > 0).sum()), int(len(nz)), 0.5, alternative="greater").pvalue) if len(nz) else 1.0
    return dict(n=int(len(x)), median=float(np.median(x)) if len(x) else float("nan"), share_positive=float((nz > 0).mean()) if len(nz) else float("nan"), p=p)


def better(gain):
    g = binom_greater(gain); g["passed"] = bool(g["p"] < 0.05 and g["median"] >= 0.01); return g


def auroc(score, lab):
    from scipy.stats import rankdata
    lab = np.asarray(lab, bool); n1 = lab.sum(); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score); return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def derange(n, rng):
    p = rng.permutation(n)
    while n > 1 and np.any(p == np.arange(n)):
        bad = np.nonzero(p == np.arange(n))[0]; sw = rng.integers(0, n, len(bad)); p[bad], p[sw] = p[sw], p[bad].copy()
    return p


def gate(real, neg_passed, planted_passed):
    ok = bool((not neg_passed) and planted_passed)
    return dict(real=bool(real), controls_ok=ok, negative_passed=bool(neg_passed), planted_passed=bool(planted_passed),
                verdict=("PASS" if real else "FAIL") if ok else "VOID (controls)")


# ------------------------------------------------------------------------------------------------ models
class Reducer:
    """standardise on DEV, 256 principal components fit on DEV, components standardised on DEV."""
    def __init__(self, E_dev):
        E_dev = np.nan_to_num(E_dev); self.mu = E_dev.mean(0); self.sd = E_dev.std(0) + 1e-6
        Zs = (E_dev - self.mu) / self.sd; _, _, Vt = np.linalg.svd(Zs, full_matrices=False); self.V = Vt[:min(NPC, Vt.shape[0])]
        Z = Zs @ self.V.T; self.zm = Z.mean(0); self.zs = Z.std(0) + 1e-6

    def __call__(self, E):
        E = np.where(np.isfinite(E), E, self.mu); return (((E - self.mu) / self.sd) @ self.V.T - self.zm) / self.zs


def fit_lr(X, y, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    best = None
    for C in CS:
        ll = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr]); p = np.clip(m.predict_proba(X[va])[:, 1], 1e-6, 1 - 1e-6)
            ll.append(-np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)))
        if best is None or np.mean(ll) < best[0]:
            best = (float(np.mean(ll)), C)
    return LogisticRegression(C=best[1], max_iter=3000).fit(X, y)


def class_models(X_dev, Y_dev):
    return [fit_lr(X_dev, Y_dev[:, c]) for c in range(len(CLASSES))]


def predict(models, X):
    return np.stack([m.predict_proba(X)[:, 1] for m in models], 1).clip(1e-6, 1 - 1e-6)


def logloss(P, Y):
    return -(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def train_pair(z, pos, dev_idx, rng, epochs=8, neg=10):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(SEED)
    Z = torch.tensor(z, dtype=torch.float32, device=dev)
    W = torch.nn.Parameter(torch.randn(z.shape[1], KW, device=dev) * 0.05); u = torch.nn.Parameter(torch.zeros(z.shape[1], device=dev))
    c = torch.nn.Parameter(torch.zeros(1, device=dev)); opt = torch.optim.Adam([W, u, c], lr=3e-3, weight_decay=1e-5)
    for _ in range(epochs):
        ng = dev_idx[rng.integers(0, len(dev_idx), (len(pos) * neg, 2))]; ng = ng[ng[:, 0] != ng[:, 1]]
        A = np.concatenate([pos, ng]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(ng))]).astype(np.float32); o = rng.permutation(len(A))
        A, y = torch.tensor(A[o], device=dev), torch.tensor(y[o], device=dev)
        for s in range(0, len(A), 8192):
            a, b, t = A[s:s + 8192, 0], A[s:s + 8192, 1], y[s:s + 8192]
            sc = ((Z[a] @ W) * (Z[b] @ W)).sum(1) + Z[a] @ u + Z[b] @ u + c
            loss = torch.nn.functional.binary_cross_entropy_with_logits(sc, t); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ZW = (Z @ W).cpu().numpy().astype(np.float64); Zu = (Z @ u).cpu().numpy().astype(np.float64); cc = float(c.item())
    return Pair(ZW, Zu, cc)


class Pair:
    """pair-model scores: all(q) against every protein; pairs(a, b) for index arrays."""
    def __init__(self, ZW, Zu, c):
        self.ZW, self.Zu, self.c = ZW, Zu, c

    def __call__(self, q):
        return self.ZW @ self.ZW[q] + self.Zu + self.Zu[q] + self.c

    def pairs(self, a, b):
        return (self.ZW[a] * self.ZW[b]).sum(1) + self.Zu[a] + self.Zu[b] + self.c


# ------------------------------------------------------------------------------------------------ the tests
def run_e1(D, rng):
    z, Xt, Y, lab, dev, exam = D["z"], D["Xtag"], D["Y"], D["lab"], D["dev"], D["exam"]
    tr = lab & dev; te = lab & exam; perm = derange(len(z), rng)
    planted = (Y[:, 0] + rng.normal(0, 1, len(Y)))[:, None]
    feats = {"tags": Xt, "tags+esmc": np.hstack([Xt, z]), "tags+shuffled": np.hstack([Xt, z[perm]]), "tags+planted": np.hstack([Xt, planted])}
    LL, auc = {}, {}
    for k, X in feats.items():
        M = class_models(X[tr], Y[tr]); P = predict(M, X[te]); LL[k] = logloss(P, Y[te]).mean(1)
        auc[k] = {c: auroc(P[:, j], Y[te][:, j]) for j, c in enumerate(CLASSES)}
        log(f"  E1 {k} fitted")
    b = {k: better(LL["tags"] - LL[k]) for k in ("tags+esmc", "tags+shuffled", "tags+planted")}
    zonly = class_models(z[tr], Y[tr]); Pz = predict(zonly, z[te])
    out = dict(dev_proteins=int(tr.sum()), exam_proteins=int(te.sum()), esmc_vs_tags=b["tags+esmc"], shuffled_vs_tags=b["tags+shuffled"],
               planted_vs_tags=b["tags+planted"], mean_logloss={k: float(v.mean()) for k, v in LL.items()},
               exam_auroc=dict(auc, esmc_only={c: auroc(Pz[:, j], Y[te][:, j]) for j, c in enumerate(CLASSES)}))
    out["gate"] = gate(b["tags+esmc"]["passed"], b["tags+shuffled"]["passed"], b["tags+planted"]["passed"])
    return out, zonly


def run_e2(D, zonly, rng, n_mc):
    z, Y, lab, exam, notag = D["z"], D["Y"], D["lab"], D["exam"], D["notag"]
    te = lab & exam & notag; Yt = Y[te]; P = predict(zonly, z[te])
    tr = lab & D["dev"]; perm = derange(len(z), rng); Ms = class_models(z[perm][tr], Y[tr]); Ps = predict(Ms, z[perm][te])
    Pp = Yt + rng.normal(0, 1, Yt.shape)
    elig = [j for j in range(len(CLASSES)) if Yt[:, j].sum() >= 30 and (1 - Yt[:, j]).sum() >= 30]

    def score(Pm):
        from scipy.stats import rankdata
        R = np.stack([rankdata(Pm[:, j]) for j in elig], 1); n = len(Yt)
        def mean_auc(Yl):
            a = []
            for k, j in enumerate(elig):
                l = Yl[:, j].astype(bool); n1 = l.sum(); n0 = n - n1; a.append((R[l, k].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            return float(np.mean(a))
        real = mean_auc(Yt); nulls = np.array([mean_auc(Yt[rng.permutation(n)]) for _ in range(n_mc)])
        pm = float((1 + (nulls >= real).sum()) / (1 + n_mc))
        return dict(mean_auroc=real, null_95=float(np.percentile(nulls, 95)), p_mc=pm, passed=bool(real >= 0.70 and pm < 0.05),
                    per_class={CLASSES[j]: auroc(Pm[:, j], Yt[:, j]) for j in elig})
    r, s, p_ = score(P), score(Ps), score(Pp)
    out = dict(proteins=int(te.sum()), classes_tested=[CLASSES[j] for j in elig], in_class={CLASSES[j]: int(Yt[:, j].sum()) for j in elig},
               esmc=r, shuffled=s, planted=p_)
    out["gate"] = gate(r["passed"], s["passed"], p_["passed"])
    return out


def run_p1(D, rng):
    from scipy.stats import rankdata
    z, U, sl, truth, dev = D["z"], D["U"], D["sl"], D["truth"], D["dev"]
    dev_idx = np.nonzero(dev)[0]
    pos = np.array([(a, b) for a, b in D["pair_idx"] if dev[a] and dev[b]])
    q, cand, s1, fl, rg9 = sl["q"], sl["cand"].astype(np.int64), sl["s1"].astype(np.float64), sl["flags"], sl["rank_g9"]
    dest = (fl & 1).astype(bool); tog = (fl & 2).astype(bool)
    perm = derange(len(z), rng)
    scorers = {"esmc": train_pair(z, pos, dev_idx, rng), "shuffled": train_pair(z[perm], pos, dev_idx, rng)}
    log("  P1 pair models trained")
    noise = rng.normal(0, 1, len(U))
    def planted(qi):
        v = noise.copy(); v[list(truth.get(qi, ()))] += 1.0; return v
    scorers["planted"] = planted
    ranks = {k: [] for k in scorers}; base = []; mism = 0
    for r in range(len(q)):
        qi = int(q[r]); C = cand[r]; is_p = np.array([c in truth.get(qi, ()) for c in C])
        if not is_p.any():
            continue
        r0 = int(np.argmax(is_p)) + 1; mism += int(r0 != rg9[r]); base.append(r0)
        for k, f in scorers.items():
            sc = f(qi); rk = rankdata(sc)[C]
            key = s1[r] + rk + 1e6 * dest[r] + 1e3 * (dest[r] & tog[r])
            o = np.argsort(-key, kind="stable"); ranks[k].append(int(np.argmax(is_p[o])) + 1)
    base = np.array(base); g = {k: np.log2(base / np.array(v)) for k, v in ranks.items()}
    b = {k: better(v) for k, v in g.items()}
    # reported: the pair model alone over all candidates, and on EXAM-EXAM interactions
    f = scorers["esmc"]; alone, exp10 = [], []
    for qi in sl["q_all"]:
        qi = int(qi); sc = f(qi); sc[qi] = -np.inf; tr_ = np.array(sorted(truth.get(qi, ())))
        if not len(tr_):
            continue
        o = np.argsort(-sc, kind="stable"); posn = np.empty(len(sc), np.int64); posn[o] = np.arange(len(sc)); alone.append(int(posn[tr_].min()) + 1)
        n = len(tr_); N = len(U) - 1; exp10.append(1 - np.prod([(N - n - j) / (N - j) for j in range(10)]))
    ex = np.nonzero(~dev)[0]; ep = np.array([(a, b) for a, b in D["pair_idx"] if not dev[a] and not dev[b]])
    rnd = ex[rng.integers(0, len(ex), (len(ep), 2))]; rnd = rnd[rnd[:, 0] != rnd[:, 1]]
    s_pos = f.pairs(ep[:, 0], ep[:, 1]); s_rnd = f.pairs(rnd[:, 0], rnd[:, 1])
    out = dict(queries=int(len(base)), shortlist_rank_mismatch=int(mism), dev_training_pairs=int(len(pos)),
               median_tries={"g9": float(np.median(base)), **{k: float(np.median(v)) for k, v in ranks.items()}},
               within_10={"g9": float((base <= 10).mean()), **{k: float((np.array(v) <= 10).mean()) for k, v in ranks.items()}},
               first_try={"g9": float((base == 1).mean()), **{k: float((np.array(v) == 1).mean()) for k, v in ranks.items()}},
               esmc_vs_g9=b["esmc"], shuffled_vs_g9=b["shuffled"], planted_vs_g9=b["planted"],
               pair_model_alone=dict(queries=len(alone), within_10=float((np.array(alone) <= 10).mean()), random_within_10=float(np.mean(exp10)),
                                     median_tries=float(np.median(alone)), exam_pair_auroc=auroc(np.r_[s_pos, s_rnd], np.r_[np.ones(len(s_pos)), np.zeros(len(s_rnd))])))
    out["gate"] = gate(b["esmc"]["passed"], b["shuffled"]["passed"], b["planted"]["passed"])
    return out


def run_s1(D, zonly, rng, n_mc):
    from scipy.stats import rankdata
    iso = D["iso"]; sel = [k for k, r in enumerate(iso) if r["exam"]]
    if not sel:
        raise SystemExit("no isoforms")
    zi, zc = D["z_iso"][sel], D["z_can_of_iso"][sel]
    lab = np.array([iso[k]["tag"] for k in sel]); rel = np.array([iso[k]["rel"] for k in sel])
    edges = np.quantile(rel, [0.2, 0.4, 0.6, 0.8]); st = np.digitize(rel, edges)
    shift = np.abs(predict(zonly, zi) - predict(zonly, zc)).sum(1)
    both = np.vstack([zi, zc]); pm = derange(len(both), rng); sh = both[pm]
    shift_neg = np.abs(predict(zonly, sh[:len(zi)]) - predict(zonly, sh[len(zi):])).sum(1)
    shift_pl = lab + rng.normal(0, 1, len(lab))

    def strat(score):
        R = np.zeros(len(score)); w_ = []
        for s in np.unique(st):
            m = st == s; R[m] = rankdata(score[m])
        def auc(l):
            num = den = 0.0
            for s in np.unique(st):
                m = st == s; ll = l[m]; n1 = ll.sum(); n0 = m.sum() - n1
                if n1 and n0:
                    a = (R[m][ll].sum() - n1 * (n1 + 1) / 2) / (n1 * n0); num += a * n1 * n0; den += n1 * n0
            return num / den if den else float("nan")
        real = auc(lab); nulls = []
        for _ in range(n_mc):
            l = lab.copy()
            for s in np.unique(st):
                m = np.nonzero(st == s)[0]; l[m] = l[m][rng.permutation(len(m))]
            nulls.append(auc(l))
        nulls = np.array(nulls); p = float((1 + (nulls >= real).sum()) / (1 + n_mc))
        return dict(auroc=float(real), null_95=float(np.percentile(nulls, 95)), p_mc=p, passed=bool(real >= 0.60 and p < 0.05))
    r, n_, p_ = strat(shift), strat(shift_neg), strat(shift_pl)
    out = dict(isoforms=int(len(lab)), tag_changing=int(lab.sum()), genes=int(len({iso[k]["gene"] for k in sel})),
               median_shift=dict(tag_changing=float(np.median(shift[lab])), other=float(np.median(shift[~lab]))), esmc=r, shuffled=n_, planted=p_)
    out["gate"] = gate(r["passed"], n_["passed"], p_["passed"])
    return out


def catalogue(D, zonly, path):
    iso = D["iso"]; k = [i for i, r in enumerate(iso) if r["tag"]]
    if not k:
        return 0
    Pi = predict(zonly, D["z_iso"][k]); Pc = predict(zonly, D["z_can_of_iso"][k])
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gene", "symbol", "isoform", "tag_changed", "split", "canonical_class", "isoform_class", "shift", "moves"] +
                                      [f"p_canonical_{c}" for c in CLASSES] + [f"p_isoform_{c}" for c in CLASSES])
        for j, i in enumerate(k):
            r = iso[i]; sh = float(np.abs(Pi[j] - Pc[j]).sum()); a, b = CLASSES[int(Pc[j].argmax())], CLASSES[int(Pi[j].argmax())]
            w.writerow([r["gene"], r["sym"], r["pid"], "+".join(r["changed"]), "exam" if r["exam"] else "dev", a, b, round(sh, 3), int(a != b or sh >= 0.5)] +
                       [round(float(x), 3) for x in Pc[j]] + [round(float(x), 3) for x in Pi[j]])
    return len(k)


# ------------------------------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", default=os.path.join(HERE, "bundle")); ap.add_argument("--out", default="esmc_out")
    ap.add_argument("--model", default=DECLARED_MODEL); ap.add_argument("--trial", action="store_true"); ap.add_argument("--budget", type=int, default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(SEED)
    sha = code_sha(); log(f"ESM C test G10, code {sha}, model {a.model}{' (TRIAL: small subset, only controls shown)' if a.trial else ''}")
    man, prot, U, G, pairs, sl = load_bundle(a.bundle); ui = {g: i for i, g in enumerate(U)}
    can = {r["gene"]: r for r in prot if r["kind"] == "canonical"}
    isos = [r for r in prot if r["kind"] == "isoform"]
    # isoform rows for S1 and the catalogue: genes whose tag-changing isoforms are all named by number
    iso_rows = []
    for r in isos:
        g = G[r["gene"]]; ch = g["changing"]
        if any(not k.isdigit() for k in ch):
            continue
        iso_rows.append(dict(pid=r["pid"], gene=r["gene"], sym=g["sym"], exam=g["split"] == "exam", tag=r["iso"] in ch, changed=ch.get(r["iso"], []),
                             rel=abs(len(r["seq"]) - len(can[r["gene"]]["seq"])) / len(can[r["gene"]]["seq"]), seq=r["seq"]))
    n_mc = 1000
    if a.trial:
        n_mc = 50
        labd = [g for g in U if G[g]["classes"] and G[g]["split"] == "dev"]; labe = [g for g in U if G[g]["classes"] and G[g]["split"] == "exam"]
        sub = set(rng.choice(labd, 150, replace=False)) | set(rng.choice(labe, 150, replace=False))
        tg = sorted({r["gene"] for r in iso_rows if r["tag"]}); sub |= set(rng.choice(tg, 40, replace=False))
        sub |= {U[int(i)] for i in sl["q"][:40]}
        iso_rows = [r for r in iso_rows if r["gene"] in sub][:200]
    # fingerprints
    seq_list = [can[g]["seq"] for g in U] + [r["seq"] for r in iso_rows]
    if a.trial:
        idx = sorted([ui[g] for g in sub]) + list(range(len(U), len(seq_list)))
        Es, nonfin, npieces = embed([seq_list[i][:200] for i in idx], a.model, os.path.join(a.out, "trial_cache.npz"), a.budget)
        E = rng.normal(0, 1, (len(seq_list), Es.shape[1])).astype(np.float32); E[idx] = Es
    else:
        E, nonfin, npieces = embed(seq_list, a.model, os.path.join(a.out, f"cache_{a.model.split('/')[-1]}.npz"), a.budget)
        np.savez_compressed(os.path.join(a.out, f"fingerprints_{a.model.split('/')[-1]}.npz"), ids=np.array([can[g]["pid"] for g in U] + [r["pid"] for r in iso_rows]),
                            genes=np.array(U + [r["gene"] for r in iso_rows]), X=E.astype(np.float16))
    t_emb = time.time() - t0; log(f"fingerprints done ({t_emb / 60:.1f} min; {nonfin} non-finite)")
    # data for the tests
    dev = np.array([G[g]["split"] == "dev" for g in U]); exam = ~dev
    Ecan = E[:len(U)]; red = Reducer(Ecan[dev]); z = red(Ecan)
    lab = np.array([bool(G[g]["classes"]) for g in U]); Y = np.array([[c in G[g]["classes"] for c in CLASSES] for g in U], float)
    T = np.array([[G[g]["tags"][t] for t in TAGS] for g in U], float); L = np.array([G[g]["length"] for g in U], float)
    Xt = np.column_stack([T[:, 0], T[:, 1], (T[:, 2] > 0), np.log1p(T[:, 2]), T[:, 3:], np.log(L)]).astype(float)
    Xt = (Xt - Xt[dev].mean(0)) / (Xt[dev].std(0) + 1e-9)
    notag = (T.sum(1) == 0)
    truth = {}
    pair_idx = []
    for x, y in pairs:
        if x in ui and y in ui and x != y:
            i, j = ui[x], ui[y]; pair_idx.append((i, j)); truth.setdefault(i, set()).add(j); truth.setdefault(j, set()).add(i)
    Ei = E[len(U):]
    D = dict(z=z, Xtag=Xt, Y=Y, lab=lab, dev=dev, exam=exam, notag=notag, U=U, sl=sl, truth=truth, pair_idx=pair_idx,
             iso=iso_rows, z_iso=red(Ei) if len(Ei) else np.zeros((0, z.shape[1])), z_can_of_iso=z[[ui[r["gene"]] for r in iso_rows]])
    if a.trial:
        keep = np.isin(sl["q"], [ui[g] for g in sub]); D["sl"] = {k: (v[keep] if k in ("q", "cand", "s1", "flags", "rank_g9") else v) for k, v in sl.items()}
        D["sl"]["q_all"] = D["sl"]["q"]
    res = {}
    log("G10-E1 address from the whole sequence"); res["G10-E1"], zonly = run_e1(D, rng)
    log("G10-E2 hidden addresses"); res["G10-E2"] = run_e2(D, zonly, rng, n_mc)
    log("G10-P1 partners that fit"); res["G10-P1"] = run_p1(D, rng)
    log("G10-S1 RNA versions that move the protein"); res["G10-S1"] = run_s1(D, zonly, rng, n_mc)
    import torch, transformers
    meta = dict(code_sha=sha, model=a.model, revision=REVISIONS.get(a.model), declared_model=DECLARED_MODEL, gating=(a.model == DECLARED_MODEL and not a.trial),
                trial=a.trial, gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), torch=torch.__version__,
                transformers=transformers.__version__, python=platform.python_version(), sequences=len(seq_list), pieces=npieces, non_finite=nonfin,
                fingerprint_minutes=round(t_emb / 60, 1), total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                bundle=man["files"], uniprot_release=man.get("uniprot_release"))
    if a.trial:
        ctl = {k: {x: v["gate"][x] for x in ("controls_ok", "negative_passed", "planted_passed")} for k, v in res.items()}
        json.dump(dict(meta=meta, controls=ctl), open(os.path.join(a.out, "G10_trial_controls.json"), "w"), indent=1)
        for k, v in ctl.items():
            log(k, "controls", "ok" if v["controls_ok"] else "NOT ok", f"(negative passed: {v['negative_passed']}, planted passed: {v['planted_passed']})")
        log("trial ran; outcomes not shown"); return
    n_cat = catalogue(D, zonly, os.path.join(a.out, "G10_isoform_catalogue.csv"))
    res["meta"] = dict(meta, catalogue_rows=n_cat)
    json.dump(res, open(os.path.join(a.out, "G10_results.json"), "w"), indent=1, default=float)
    summary(res)


def summary(res):
    e1, e2, p1, s1 = res["G10-E1"], res["G10-E2"], res["G10-P1"], res["G10-S1"]
    print("\n" + "=" * 100)
    print(f"G10 RESULTS  (model {res['meta']['model']}{'' if res['meta']['gating'] else ' -- exploratory, not the declared test'}; code {res['meta']['code_sha']})")
    print(f" E1 address from the whole sequence: {e1['gate']['verdict']}. Held-out proteins better with ESM C added to the tags: "
          f"{e1['esmc_vs_tags']['share_positive']:.0%} (median gain {e1['esmc_vs_tags']['median']:+.3f}).")
    print(f" E2 hidden addresses (proteins with no tag): {e2['gate']['verdict']}. Mean AUROC {e2['esmc']['mean_auroc']:.2f} (chance 0.50; "
          f"random 95% {e2['esmc']['null_95']:.2f}).")
    print(f" P1 partners that fit: {p1['gate']['verdict']}. Partner within 10 tries {p1['within_10']['g9']:.1%} (G9) -> {p1['within_10']['esmc']:.1%} "
          f"(with ESM C); median tries {p1['median_tries']['g9']:.0f} -> {p1['median_tries']['esmc']:.0f}.")
    print(f" S1 RNA versions that move the protein: {s1['gate']['verdict']}. AUROC {s1['esmc']['auroc']:.2f} (random 95% {s1['esmc']['null_95']:.2f}).")
    print(" Controls: " + ", ".join(f"{k[4:]} {'ok' if res[k]['gate']['controls_ok'] else 'NOT ok'}" for k in ("G10-E1", "G10-E2", "G10-P1", "G10-S1")))
    print("=" * 100)


if __name__ == "__main__":
    main()
