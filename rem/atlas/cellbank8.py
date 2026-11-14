"""Engine v2: translation first, gene-specific neighbours, eight more screens, gene-specific scoring, a locked exam,
and a confidence for every prediction. Built from the model review (modelreview_posthoc.py), which found that a
linear map from a gene's own records in other cell types matched or beat the engine where it was best, that the
snap memory recalled ~12 records of ~8,500, that most of the score sat in 2-3 global programmes, and that the TEST
half had been used for 30+ comparisons.

THE SEVEN CHANGES
 1  TRANSLATION FIRST. An any-to-any translator between cell types predicts the gene's K562 response from its
    records in other screens; a residual network learns only what the translation misses (pred = W.T + f(x)).
 2  GENE-SPECIFIC NEIGHBOURS INSTEAD OF A GLOBAL SNAP. Three fixed inputs: (a) analog transfer: the genes whose
    responses look most like this gene's in the other screens, and their measured K562 responses (top 20 by
    cosine, weights exp(10 cos)); (b) interaction partners' K562 mean; (c) co-essential genes' K562 mean.
    The missing control is run: the same inputs into a plain net (M2) and into the snapping engine (M2e).
 3  GENE-SPECIFIC SCORING. Besides r, 'specific r': r after removing the 3 largest TRAIN response programmes
    (the first 3 components) from prediction and answer. Medians are also shown as a fraction of the ceiling
    (the r of the bank's own compressed record of the answer).
 4  MORE CELL TYPES. From the uniformly reprocessed CRISPRi collection (figshare 34002315; crispyx; licence
    CC BY-NC-SA 4.0): genome-wide HCT116, HEK293T, iPSC (Nourreddine; Feng sf, snf), targeted iPSC (Feng ts),
    hESC (Pan; VCC), and RPE1, HepG2, Jurkat reprocessed the same way: 11 screens. Read with HTTP range
    requests (remotezip.py) so only each effect matrix is fetched. Effect / median control s.d. per gene ~ the
    Replogle z scale; each screen's mean TRAIN-gene response removed; mapped onto screen A's genes by symbol
    (HPA synonyms allowed); 128 components; int8. Because of the licence, these records stay in the local cache
    (rem/atlas/_cache/cell_v8, git-ignored); only code and summary statistics are committed.
 5  ANY-TO-ANY. The translator: one linear encoder per screen into a shared 128-d space (averaged over the
    screens a gene has), one linear decoder per screen. Trained on TRAIN genes only, by hiding one available
    screen (target) and predicting it from a random subset of the others (keep probability 0.7), loss 1 - cosine,
    Adam 1e-3, weight decay 1e-4, 3,000 steps of 256. TRAIN genes' K562 inputs to the residual net come from
    5-fold out-of-fold translators (folds by gene hash). Descriptive: the screen-to-screen accuracy matrix on
    TEST-split genes.
 6  A LOCKED EXAM. TEST knockdowns minus the 150 inspected gene by gene (claudeloop 100, cellbank4 50) are split
    by sha256("exam-2026-11:" + gene) into DEV (development: every arm and metric may be looked at) and EXAM.
    `exam` scores EXAM once, for the gates below only, writes outputs/cellbank8/exam_lock.json, and refuses to
    run again.
 7  CONFIDENCE. Per gene: number of screens with a record, translator agreement across single screens, best
    analog similarity, mean record strength, partner / co-essential coverage, seed disagreement, predicted norm.
    A linear fit of M2's r on these, fitted on DEV, is the confidence; checked once on EXAM.

ARMS (seeds 0-2 averaged where trained; the notebook's section 9 recipe: 150 epochs, early stop on 10% of TRAIN)
  E3   the previous best (cellbank5's E3: snap over A-TRAIN + K562-ess + RPE1/HepG2/Jurkat (Replogle/Nadig records),
       own records in RPE1/HepG2/Jurkat, partners, co-essential), rerun here.
  T    the translator alone (constant TRAIN mean where a gene has no record elsewhere).
  M2   NEW MODEL: pred = W.T + MLP(x); x = notebook non-LLM features + T + screen flags + analog + partners + co-essential.
  M2e  as M2 but f = the snapping engine (memory = A-TRAIN records) on the same x.

GATES ON EXAM, PREDECLARED (paired sign test over genes; BETTER = p < 0.05 and median gain >= +0.01; WORSE mirrored)
  X1  PRIMARY: M2 vs E3, r.
  X2  M2 vs T, r (does anything beyond translation help?).
  X3  M2 vs M2e, r (plain net vs engine on identical inputs; either direction reported).
  X4  M2 vs E3, specific r.
  X5  confidence: Spearman(confidence, M2's r) > 0 with p < 0.05 -> CALIBRATED.
NOT: one translator design, one neighbour rule, one confidence model; symbol matching across screens; the
     reprocessed effects and the Replogle/Nadig records differ in pipeline, which the translator must absorb.

Run: python3 -P rem/atlas/cellbank8.py pull | build | train | dev | exam
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
_l = lambda n: importlib.util.spec_from_file_location(n, os.path.join(HERE, f"{n}.py"))
_s = _l("cellbank4"); cb4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb4)
_s = _l("cellbank5"); cb5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb5)
import remotezip  # noqa: E402

V6 = R("memory_bank", "cell_v6")
BANK8 = os.path.join(HERE, "_cache", "cell_v8")
CACHE = os.path.join(HERE, "_cache", "cellbank8")
OUTD = R("outputs", "cellbank8")
OUT = os.path.join(HERE, "RESULTS_cellbank8.txt")
FIG = "https://ndownloader.figshare.com/files/{}"
SOURCES = [  # name, figshare file id of effect_<ds>.zip, cell type, perturbation set
    ("RPE1", 69471417, "RPE1", "essential"), ("HepG2", 69471420, "HepG2", "essential"), ("Jurkat", 69471419, "Jurkat", "essential"),
    ("HCT116", 69474273, "HCT116", "genome-wide"), ("HEK293T", 69474577, "HEK293T", "genome-wide"),
    ("iPSC-N", 69472512, "iPSC", "genome-wide"), ("iPSC-Fsf", 69471714, "iPSC", "genome-wide sub-screen"),
    ("iPSC-Fsnf", 69471750, "iPSC", "genome-wide sub-screen"), ("iPSC-Fts", 69471407, "iPSC", "targeted"),
    ("hESC-P", 69471805, "H1 hESC", "growth / pluripotency"), ("hESC-V", 69471401, "H1 hESC", "effect strength"),
]
SRC = [s[0] for s in SOURCES]
SEEDS, RULE, N_PC_OUT = (0, 1, 2), "=" * 97, 3
T_STEPS, T_BATCH, T_KEEP, T_LR, T_WD, T_DIM = 3000, 256, 0.7, 1e-3, 1e-4, 128
ANALOG_K, ANALOG_TEMP = 20, 10.0
split2 = lambda g: "DEV" if int(hashlib.sha256(("exam-2026-11:" + g).encode()).hexdigest(), 16) % 2 == 0 else "EXAM"


# ---------------------------------------------------------------------------------------------------------
# 4. pull the effect matrices (range requests; only rows of knocked-down genes with screen-A features)
# ---------------------------------------------------------------------------------------------------------
def _names(ds):
    import numpy as np
    if hasattr(ds, "keys"):                            # categorical
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][()]]
        return [cats[c] for c in ds["codes"][()]]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(ds[()])]


def _read_rows(args):
    import h5py
    import numpy as np
    link, a, b, rows, col_src, col_dst, n_dst = args
    for k in range(4):
        try:
            m, base = remotezip.open_member(link, "_moments.h5ad")
            X = h5py.File(m, "r")["X"]
            blk = X[a:b, :]
            break
        except Exception:
            if k == 3:
                raise
            time.sleep(5 * (k + 1))
    out = np.full((len(rows), n_dst), np.nan, np.float32)
    out[:, col_dst] = blk[np.asarray(rows) - a][:, col_src]
    return rows, out, base.fetched


def aliases(G):
    """name (upper case) -> screen-A gene symbol, via the symbol itself and HPA synonyms."""
    kn = set(G["KN"]); ag = set(G["A"]["genes"])
    to_kn, to_col = {}, {}
    for g in G["KN"]:
        to_kn.setdefault(g.upper(), g)
    for j, g in enumerate(G["A"]["genes"]):
        to_col.setdefault(g.upper(), j)
    for h in G["HPA"].values():
        for x in (h.get("syn") or "").split(","):
            x = x.strip().upper()
            if x and h["sym"] in kn:
                to_kn.setdefault(x, h["sym"])
            if x and h["sym"] in ag and h["sym"] in G["gidx"]:
                to_col.setdefault(x, G["gidx"][h["sym"]])
    return to_kn, to_col


def pull():
    import h5py
    import numpy as np
    from multiprocessing import Pool
    G = cb4.notebook(); os.makedirs(CACHE, exist_ok=True)
    to_kn, to_col = aliases(G)
    n_dst = len(G["A"]["genes"])
    for name, fid, cell, design in SOURCES:
        path = os.path.join(CACHE, f"src_{name}.npz")
        if os.path.exists(path):
            print(f"{name}: cached"); continue
        t0 = time.time(); link = FIG.format(fid)
        m, base = remotezip.open_member(link, "_moments.h5ad")
        f = h5py.File(m, "r"); X = f["X"]
        perts = _names(f["obs"][f["obs"].attrs.get("_index", "_index")])
        genes = _names(f["var"][f["var"].attrs.get("_index", "_index")])
        rows, seen = [], set()
        for i, p in enumerate(perts):
            g = to_kn.get(p.upper())
            if g and g not in seen:
                rows.append(i); seen.add(g)
        pairs = [(j, to_col[g.upper()]) for j, g in enumerate(genes) if g.upper() in to_col]
        dst_seen, col_src, col_dst = set(), [], []
        for j, d in pairs:
            if d not in dst_seen:
                dst_seen.add(d); col_src.append(j); col_dst.append(d)
        c0 = X.chunks[0] if X.chunks else 64
        sd = np.nanmedian(f["layers/control_std_matched"][0:c0, :], 0)[col_src]
        floor = np.nanquantile(sd[sd > 0], 0.10)
        sd = np.where(np.isfinite(sd) & (sd > floor), sd, floor).astype(np.float32)
        step = c0 * max(1, (400 + c0 - 1) // c0)
        tasks = []
        for a in range(0, X.shape[0], step):
            b = min(a + step, X.shape[0]); rr = [r for r in rows if a <= r < b]
            if rr:
                tasks.append((link, a, b, rr, col_src, col_dst, n_dst))
        M = np.full((len(rows), n_dst), np.nan, np.float32); where = {r: k for k, r in enumerate(rows)}; got = 0
        with Pool(4) as pool:
            for k, (rr, out, fetched) in enumerate(pool.imap_unordered(_read_rows, tasks)):
                M[[where[r] for r in rr]] = out; got += fetched
                if k % 10 == 9:
                    print(f"   {name}: {k + 1}/{len(tasks)} blocks, {got / 1e6:.0f} MB, {time.time() - t0:.0f}s", flush=True)
        M[:, col_dst] /= sd
        np.savez_compressed(path, genes=np.array([to_kn[perts[r].upper()] for r in rows]), M=M.astype(np.float16),
                            meta=json.dumps(dict(name=name, cell=cell, design=design, perturbations=len(perts), kept=len(rows),
                                                 genes_measured=len(genes), genes_mapped=len(col_dst), mb_fetched=round(got / 1e6),
                                                 seconds=round(time.time() - t0))))
        print(f"{name}: {len(rows):,} of {len(perts):,} perturbations kept, {len(col_dst):,} genes mapped, {got / 1e6:.0f} MB, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------------------
# bank v8 = v6 + 11 reprocessed screens (local only)
# ---------------------------------------------------------------------------------------------------------
def build():
    import numpy as np
    G = cb4.notebook(); torch = G["torch"]
    old = cb4.Bank(V6); meta = json.load(open(os.path.join(V6, "records_meta.json")))
    Vy = G["Vy"].cpu().numpy(); split = G["split"]
    codes, scale, rep = [old.codes], [old.scale], {}
    meta["strength"] = {}
    for name in SRC:
        z = np.load(os.path.join(CACHE, f"src_{name}.npz")); info = json.loads(str(z["meta"]))
        genes = list(z["genes"]); M = z["M"].astype(np.float32)
        tr = [i for i, g in enumerate(genes) if split(g) == "TRAIN"]
        Z = np.nan_to_num(M - np.nanmean(M[tr], 0)) @ Vy
        nrm = np.linalg.norm(Z, axis=1); strength = nrm / np.median(nrm)
        s = np.maximum(np.abs(Z).max(1), 1e-12)
        codes.append(np.round(Z / s[:, None] * 127).astype(np.int8)); scale.append(s.astype(np.float32))
        for g, st in zip(genes, strength):
            meta["strength"][str(len(meta["gene"]))] = float(st)
            meta["gene"].append(g); meta["ens"].append(""); meta["screen"].append(f"S:{name}"); meta["split"].append(split(g)); meta["ep"].append(float("nan"))
        rep[name] = dict(info, records=len(genes))
    os.makedirs(BANK8, exist_ok=True)
    np.savez_compressed(os.path.join(BANK8, "records.npz"), codes=np.concatenate(codes), scale=np.concatenate(scale), basis=old.basis)
    json.dump(meta, open(os.path.join(BANK8, "records_meta.json"), "w"))
    json.dump(old.genes, open(os.path.join(BANK8, "genes.json"), "w"))
    te = {G["KN"][int(i)] for i in G["rte"]}
    b8 = cb4.Bank(BANK8)
    cov = sum(any(f"S:{n}" in b8.by_gene.get(g, {}) for n in SRC) for g in te)
    old_cov = sum(any(s in old.by_gene.get(g, {}) for s in "CDE") for g in te)
    rep = dict(screens=rep, records=len(b8.gene), test_genes_with_any_record_before=old_cov, test_genes_with_any_record_now=cov,
               bytes=int(b8.codes.nbytes + b8.scale.nbytes + b8.basis.nbytes))
    json.dump(rep, open(os.path.join(BANK8, "build_report.json"), "w"), indent=1)
    os.makedirs(OUTD, exist_ok=True); json.dump(rep, open(os.path.join(OUTD, "build_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "screens"}, indent=1))
    for n, v in rep["screens"].items():
        print(f"  {n:10s} {v['cell']:8s} {v['design']:22s} records {v['records']:6,}  genes mapped {v['genes_mapped']:,}")


# ---------------------------------------------------------------------------------------------------------
# 5. the any-to-any translator
# ---------------------------------------------------------------------------------------------------------
def source_tensor(G, bank):
    """Unit-normalised records per screen (S x NPH x 128), mask, strengths; index S = K562 A (TRAIN only)."""
    torch, np = G["torch"], G["np"]
    S = len(SRC); NPH = G["NPH"]
    meta = json.load(open(os.path.join(BANK8, "records_meta.json")))
    Z = torch.zeros(S + 1, NPH, 128); Mk = torch.zeros(S + 1, NPH, dtype=torch.bool); St = torch.zeros(S, NPH)
    for j, g in enumerate(G["KN"]):
        d = bank.by_gene.get(g, {})
        for s, n in enumerate(SRC):
            i = d.get(f"S:{n}")
            if i is not None:
                v = torch.tensor(bank.z([i])[0]); Z[s, j] = v / v.norm().clamp(min=1e-12); Mk[s, j] = True
                St[s, j] = meta["strength"][str(i)]
    rtr = G["rtr"]
    za = G["Ztr"].cpu(); Z[S, rtr] = za / za.norm(dim=1, keepdim=True).clamp(min=1e-12); Mk[S, rtr] = True
    return Z, Mk, St


def make_translator(G, S):
    nn, torch = G["nn"], G["torch"]

    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Parameter(torch.randn(S + 1, 128, T_DIM) * (1 / math.sqrt(128)))
            self.dec = nn.Parameter(torch.randn(S + 1, T_DIM, 128) * (1 / math.sqrt(T_DIM)))

        def forward(self, Zg, Mg, tgt):                # Zg (n, S+1, 128), Mg (n, S+1) float, tgt (n,)
            H = torch.einsum("nsd,sdh->nsh", Zg, self.enc) * Mg[..., None]
            h = H.sum(1) / Mg.sum(1, keepdim=True).clamp(min=1.0)
            return torch.einsum("nh,nhd->nd", h, self.dec[tgt])
    return T()


def train_translator(G, Z, Mk, genes_ok, seed):
    """Train on the genes in genes_ok (indices into KN) by hiding one available screen at a time."""
    torch, np = G["torch"], G["np"]
    S = Z.shape[0] - 1
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Zg, Mg = Z.permute(1, 0, 2), Mk.T.float()          # (NPH, S+1, 128), (NPH, S+1)
    ok = [j for j in genes_ok if int(Mk[:, j].sum()) >= 2]
    tr = make_translator(G, S); opt = torch.optim.Adam(tr.parameters(), lr=T_LR, weight_decay=T_WD)
    for step in range(T_STEPS):
        b = rng.choice(ok, size=min(T_BATCH, len(ok)), replace=False)
        avail = Mg[b].numpy().astype(bool)
        tgt = np.array([rng.choice(np.nonzero(a)[0]) for a in avail])
        keep = (rng.random(avail.shape) < T_KEEP) & avail
        keep[np.arange(len(b)), tgt] = False
        for i in np.nonzero(keep.sum(1) == 0)[0]:     # at least one input
            others = [s for s in np.nonzero(avail[i])[0] if s != tgt[i]]; keep[i, rng.choice(others)] = True
        x = Zg[torch.as_tensor(b)]; m = torch.tensor(keep, dtype=torch.float32); t = torch.tensor(tgt)
        pred = tr(x, m, t)
        y = x[torch.arange(len(b)), t]
        loss = 1 - torch.nn.functional.cosine_similarity(pred, y, dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return tr


def translate(G, tr, Z, Mk, rows, target, exclude_target=True):
    torch = G["torch"]
    with torch.no_grad():
        Zg, Mg = Z.permute(1, 0, 2)[rows], Mk.T.float()[rows].clone()
        if exclude_target:
            Mg[:, target] = 0.0
        return tr(Zg, Mg, torch.full((len(rows),), target, dtype=torch.long)), Mg.sum(1)


# ---------------------------------------------------------------------------------------------------------
# 2. gene-specific neighbour inputs
# ---------------------------------------------------------------------------------------------------------
def k562_records(G, bank):
    torch = G["torch"]
    rec = {}
    for j, g in enumerate(G["KN"]):
        d = bank.by_gene.get(g, {})
        if "A" in d and bank.split[d["A"]] == "TRAIN":
            rec[j] = d["A"]
        elif "B" in d:
            rec[j] = d["B"]
    K = torch.zeros(G["NPH"], 128); has = torch.zeros(G["NPH"], dtype=torch.bool)
    for j, i in rec.items():
        v = torch.tensor(bank.z([i])[0]); K[j] = v / v.norm().clamp(min=1e-12); has[j] = True
    return K, has


def analog(G, Z, Mk, K, hasK):
    """For each gene: the K562 responses of the genes most alike in the other screens (never its own)."""
    torch = G["torch"]
    S, NPH = Z.shape[0] - 1, G["NPH"]
    best_s = torch.full((NPH, ANALOG_K * S), -2.0); best_i = torch.zeros((NPH, ANALOG_K * S), dtype=torch.long)
    for s in range(S):
        m = Mk[s]
        C = Z[s] @ Z[s].T
        C[:, ~(m & hasK)] = -2.0; C[~m] = -2.0; C.fill_diagonal_(-2.0)
        v, i = C.topk(ANALOG_K, dim=1)
        best_s[:, s * ANALOG_K:(s + 1) * ANALOG_K] = v; best_i[:, s * ANALOG_K:(s + 1) * ANALOG_K] = i
    v, o = best_s.topk(ANALOG_K, dim=1); idx = best_i.gather(1, o)
    w = torch.where(v > -1.5, torch.exp(ANALOG_TEMP * v), torch.zeros_like(v))
    out = torch.zeros(NPH, 130)
    tot = w.sum(1, keepdim=True)
    ok = tot.squeeze(1) > 0
    out[ok, :128] = (w[ok, :, None] * K[idx[ok]]).sum(1) / tot[ok]
    out[ok, 128] = v[ok, 0]; out[ok, 129] = 1.0
    return out


# ---------------------------------------------------------------------------------------------------------
# 1. translation first + residual network (MLP or snapping engine)
# ---------------------------------------------------------------------------------------------------------
def train_residual(G, X, Tz, kind, seed):
    torch, nn, np = G["torch"], G["nn"], G["np"]
    rtr, rte, Y_tr, Ztr, Vy, DEV = G["rtr"], G["rte"], G["Y_tr"], G["Ztr"], G["Vy"], G["DEV"]
    pr = G["pearson_rows"]
    torch.manual_seed(seed)
    n = len(rtr); perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + seed))
    va, fi = perm[: n // 10], perm[n // 10:]
    s0 = float(Ztr.norm(dim=1).median())

    class Res(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Linear(128, 128, bias=False)
            with torch.no_grad():
                self.W.weight.copy_(torch.eye(128) * s0)
            self.f = G["MLP"](X.shape[1], 128, G["mlp_hidden"](X.shape[1], 128)) if kind == "mlp" else G["Engine"](X.shape[1], 128)

        def forward(self, rows, self_idx=None, mem_rows=None):
            base = self.W(Tz[rows])
            if kind == "mlp":
                return base + self.f(X[rows])
            return base + self.f(X[rows], X[mem_rows], Ztr_mem, self_idx=self_idx)
    m = Res().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    Ztr_mem = Ztr[fi]; mem_rows = rtr[fi]
    slot = torch.full((n,), -1, dtype=torch.long); slot[fi] = torch.arange(len(fi))
    best, state = -2.0, None
    for ep in range(G["EPOCHS"]):
        m.train()
        for b in fi[torch.randperm(len(fi))].split(256):
            pz = m(rtr[b], self_idx=slot[b], mem_rows=mem_rows)
            loss = 1 - pr(pz @ Vy.T, Y_tr[b]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 5 == 4 or ep == G["EPOCHS"] - 1:
            m.eval()
            with torch.no_grad():
                r = float(pr(m(rtr[va], mem_rows=mem_rows) @ Vy.T, Y_tr[va]).median())
            if r > best:
                best, state = r, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    Ztr_mem = Ztr; mem_rows = rtr
    with torch.no_grad():
        pz = m(rte, mem_rows=mem_rows)
    return pz.cpu(), best


def train():
    t0 = time.time()
    G = cb4.notebook(); torch, np = G["torch"], G["np"]
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    bank8 = cb4.Bank(BANK8); bank6 = cb4.Bank(V6)
    KN, rtr, rte, split = G["KN"], G["rtr"], G["rte"], G["split"]
    S = len(SRC)
    Z, Mk, St = source_tensor(G, bank8)
    train_genes = [j for j, g in enumerate(KN) if split(g) == "TRAIN"]
    test_genes = [j for j, g in enumerate(KN) if split(g) == "TEST"]

    # 5. translator: final (all TRAIN genes) + 5 out-of-fold for TRAIN rows' K562 input
    fold = lambda g: int(hashlib.sha256(("fold:" + g).encode()).hexdigest(), 16) % 5
    Tz = torch.zeros(G["NPH"], 128); Tn = torch.zeros(G["NPH"])
    tr_rows = [int(i) for i in rtr]
    for k in range(5):
        tr = train_translator(G, Z, Mk, [j for j in train_genes if fold(KN[j]) != k], seed=k)
        rows = [j for j in tr_rows if fold(KN[j]) == k]
        p, nin = translate(G, tr, Z, Mk, rows, S); Tz[rows] = p; Tn[rows] = nin
        print(f"  translator fold {k}: {time.time() - t0:.0f}s", flush=True)
    final = train_translator(G, Z, Mk, train_genes, seed=99)
    te_rows = [int(i) for i in rte]
    p, nin = translate(G, final, Z, Mk, te_rows, S); Tz[te_rows] = p; Tn[te_rows] = nin
    has_T = Tn > 0
    Tu = torch.where(has_T[:, None], Tz / Tz.norm(dim=1, keepdim=True).clamp(min=1e-12), torch.zeros_like(Tz))
    mean_tr = G["Ztr"].cpu().mean(0); mean_tr = mean_tr / mean_tr.norm()
    T_arm = torch.where(has_T[:, None], Tu, mean_tr.expand_as(Tu))
    # single-screen translations (confidence: agreement)
    single = []
    for s in range(S):
        Ms = torch.zeros_like(Mk); Ms[s] = Mk[s]
        ps, _ = translate(G, final, Z, Ms, te_rows, S); single.append(torch.where(Mk[s, te_rows][:, None], ps, torch.full_like(ps, float("nan"))))
    single = torch.stack(single, 1)                    # (n_te, S, 128)
    agree = []
    for t in range(len(te_rows)):
        v = single[t][~torch.isnan(single[t][:, 0])]
        if len(v) >= 2:
            u = v / v.norm(dim=1, keepdim=True); c = u @ u.T; agree.append(float(c[~torch.eye(len(u), dtype=torch.bool)].mean()))
        else:
            agree.append(float("nan"))
    # any-to-any matrix on TEST-split genes (descriptive)
    mat = {}
    for s in range(S):
        rows = [j for j in test_genes if bool(Mk[s, j]) and int(Mk[:S, j].sum()) >= 2]
        if len(rows) >= 20:
            ps, _ = translate(G, final, Z, Mk, rows, s)
            Vt = G["Vy"].cpu().T
            r = G["pearson_rows"](ps @ Vt, Z[s, rows] @ Vt)
            mat[SRC[s]] = dict(n=len(rows), median_r=float(torch.nan_to_num(r).median()))
    print(f"  translator done ({time.time() - t0:.0f}s): {mat}", flush=True)

    # 2. neighbour inputs
    K, hasK = k562_records(G, bank6)
    An = analog(G, Z, Mk, K, hasK)
    Pt, Ct = cb4.inputs(G, bank6, False)
    flags = torch.cat([Mk[:S].T.float(), torch.log1p(Mk[:S].sum(0).float())[:, None], has_T.float()[:, None]], 1)
    X2 = cb4.scaled(G, [Tu, flags, An, Pt, Ct])
    print(f"  inputs ready: {X2.shape[1]} features ({time.time() - t0:.0f}s)", flush=True)

    preds = dict(T=T_arm[te_rows].clone()); info = {}
    Tz_dev = Tu.to(G["DEV"])
    for arm, kind in (("M2", "mlp"), ("M2e", "engine")):
        outs = []
        for s in SEEDS:
            pz, best = train_residual(G, X2, Tz_dev, kind, s)
            outs.append(pz); info.setdefault(arm, []).append(round(best, 4))
            print(f"  {arm} seed {s}: held-out TRAIN r {best:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
        preds[arm] = torch.stack(outs).mean(0); preds[arm + "_seeds"] = torch.stack(outs)
    # E3: the previous best, rerun on its own bank and inputs
    oC, oD, oE = cb5.own_inputs(G, bank6, ("C", "D", "E"))
    X3 = cb4.scaled(G, [Pt, Ct, oC, oD, oE]); cb4.SCREENS = cb5.ALL
    outs = []
    for s in SEEDS:
        pz, best, _ = cb4.run_arm(G, bank6, "E3", s, X3)
        outs.append(pz); info.setdefault("E3", []).append(round(best, 4))
        print(f"  E3 seed {s}: held-out TRAIN r {best:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    preds["E3"] = torch.stack(outs).mean(0)
    conf = dict(n_screens=Mk[:S, te_rows].sum(0).float().tolist(), agreement=agree, analog_sim=An[te_rows, 128].tolist(),
                strength=[float(St[:, j][Mk[:S, j]].mean()) if bool(Mk[:S, j].any()) else 0.0 for j in te_rows],
                partner=(Pt[te_rows, 128] > 0).float().tolist(), coess=Ct[te_rows, 128].tolist())
    torch.save(dict(preds=preds, conf=conf, te_rows=te_rows), os.path.join(CACHE, "train_out.pt"))
    json.dump(dict(heldout_train_r=info, translator_matrix=mat, features=int(X2.shape[1]),
                   test_genes_with_translation=int(has_T[te_rows].sum())), open(os.path.join(OUTD, "train_info.json"), "w"), indent=1)
    print(f"train done ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------
# 3, 6, 7. scoring: DEV (open) and EXAM (once)
# ---------------------------------------------------------------------------------------------------------
def scorer(G):
    torch, np = G["torch"], G["np"]
    A_te, Vy, pr = G["A_te"], G["Vy"].cpu(), G["pearson_rows"]
    K3 = Vy[:, :N_PC_OUT]

    def spec(M):
        M0 = torch.nan_to_num(M); out = M0 - (M0 @ K3) @ K3.T
        return torch.where(torch.isnan(M), torch.full_like(M, float("nan")), out)
    A_spec = spec(A_te)
    bank6 = cb4.Bank(V6)
    ceil_z = torch.tensor(bank6.z([bank6.get(G["KN"][int(i)], "A") for i in G["rte"]]))
    ceil = torch.nan_to_num(pr(ceil_z @ Vy.T, A_te)).numpy()
    ceil_s = torch.nan_to_num(pr(spec(ceil_z @ Vy.T), A_spec)).numpy()

    def score(pz):
        P = pz @ Vy.T
        return torch.nan_to_num(pr(P, A_te)).numpy(), torch.nan_to_num(pr(spec(P), A_spec)).numpy()
    return score, ceil, ceil_s


def sets(G):
    import numpy as np
    seen = {j["gene"] for j in json.load(open(R("outputs", "claudeloop", "menus.json")))["jobs"]}
    seen |= {j["gene"] for j in json.load(open(R("outputs", "cellbank4", "claude_jobs.json")))["jobs"]}
    genes = [G["KN"][int(i)] for i in G["rte"]]
    dev = np.array([g not in seen and split2(g) == "DEV" for g in genes]); exam = np.array([g not in seen and split2(g) == "EXAM" for g in genes])
    return genes, dev, exam, len(seen)


def verdict(a, b):
    import numpy as np
    from scipy.stats import binomtest
    d = a - b; w, l = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0; md = float(np.median(d))
    v = "BETTER" if p < 0.05 and md >= 0.01 else "WORSE" if p < 0.05 and md <= -0.01 else "NO DIFFERENCE"
    return dict(better=w, worse=l, p=float(p), median_diff=md, verdict=v)


def conf_features(G, T, o):
    np = G["np"]; torch = G["torch"]
    c = o["conf"]
    seeds = o["preds"]["M2_seeds"]; u = seeds / seeds.norm(dim=2, keepdim=True)
    spread = 1 - torch.einsum("snd,tnd->stn", u, u).mean((0, 1)).numpy()
    norm = o["preds"]["M2"].norm(dim=1).numpy()
    ag = np.array(c["agreement"], dtype=float)
    F = np.stack([np.log1p(c["n_screens"]), np.nan_to_num(ag, nan=0.0), np.isfinite(ag).astype(float), np.array(c["analog_sim"]),
                  np.array(c["strength"]), np.array(c["partner"]), np.array(c["coess"]), spread, np.log(norm)], 1)
    return F


def dev():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)
    G = cb4.notebook(); np = G["np"]; torch = G["torch"]
    o = torch.load(os.path.join(CACHE, "train_out.pt")); preds = o["preds"]
    score, ceil, ceil_s = scorer(G)
    genes, dv, ex, n_seen = sets(G)
    info = json.load(open(os.path.join(OUTD, "train_info.json"))); rep = json.load(open(os.path.join(OUTD, "build_report.json")))
    P_(RULE); P_("ENGINE V2 -- DEVELOPMENT SET (open; the EXAM half is untouched)"); P_(RULE)
    P_("  screens added (local only, CC BY-NC-SA): " + ", ".join(f"{n} {v['records']:,}" for n, v in rep["screens"].items()))
    P_(f"  TEST genes with a record in another cell type: before {rep['test_genes_with_any_record_before']:,}, now {rep['test_genes_with_any_record_now']:,} of 2,525")
    P_(f"  TEST genes {len(genes):,}: inspected before (excluded) {n_seen}; DEV {dv.sum()}, EXAM {ex.sum()}")
    P_(f"  any-to-any translator, TEST-split genes, median r predicting each screen from the others: " +
       ", ".join(f"{k} {v['median_r']:.2f}" for k, v in info["translator_matrix"].items()))
    S = {a: score(preds[a]) for a in ("E3", "T", "M2", "M2e")}
    P_(f"  DEV (n {dv.sum()}): median r | specific r | fraction of ceiling (r; ceiling {np.median(ceil[dv]):.3f}, specific ceiling {np.median(ceil_s[dv]):.3f})")
    for a, lab in (("E3", "E3 previous best"), ("T", "T translator alone"), ("M2", "M2 new model (plain net)"), ("M2e", "M2e same inputs, snapping engine")):
        r, rs = S[a]
        P_(f"     {lab:34s} {np.median(r[dv]):+.3f} | {np.median(rs[dv]):+.3f} | {np.median(r[dv]) / np.median(ceil[dv]):.0%}")
    covered = np.array([o["conf"]["n_screens"][k] > 0 for k in range(len(genes))])
    for lab, m in (("DEV genes with a record elsewhere", dv & covered), ("DEV genes with none", dv & ~covered)):
        P_(f"     {lab:34s} n {m.sum():4d}: " + ", ".join(f"{a} {np.median(S[a][0][m]):+.3f}" for a in ("E3", "T", "M2", "M2e")))
    for tag, a, b in (("M2 vs E3", "M2", "E3"), ("M2 vs T", "M2", "T"), ("M2 vs M2e", "M2", "M2e")):
        v = verdict(S[a][0][dv], S[b][0][dv])
        P_(f"     DEV {tag:10s} better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g}")
    F = conf_features(G, None, o)
    Xd = np.c_[np.ones(dv.sum()), F[dv]]; beta = np.linalg.lstsq(Xd, S["M2"][0][dv], rcond=None)[0]
    from scipy.stats import spearmanr
    rho = spearmanr(Xd @ beta, S["M2"][0][dv]).correlation
    P_(f"  confidence fitted on DEV (in-sample Spearman {rho:+.2f}); coefficients saved for the exam")
    json.dump(dict(beta=beta.tolist()), open(os.path.join(OUTD, "confidence_model.json"), "w"))
    open(os.path.join(HERE, "RESULTS_cellbank8_dev.txt"), "w").write("\n".join(out) + "\n")


def exam():
    lock = os.path.join(OUTD, "exam_lock.json")
    if os.path.exists(lock):
        print("EXAM already scored (exam_lock.json exists); refusing to score it again."); return
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)
    G = cb4.notebook(); np = G["np"]; torch = G["torch"]
    from scipy.stats import spearmanr
    o = torch.load(os.path.join(CACHE, "train_out.pt")); preds = o["preds"]
    digest = hashlib.sha256(b"".join(preds[a].numpy().tobytes() for a in ("E3", "T", "M2", "M2e"))).hexdigest()
    score, ceil, ceil_s = scorer(G)
    genes, dv, ex, _ = sets(G)
    S = {a: score(preds[a]) for a in ("E3", "T", "M2", "M2e")}
    P_(RULE); P_("ENGINE V2 -- THE LOCKED EXAM (scored once)"); P_(RULE)
    P_(f"  EXAM genes {ex.sum()}; ceiling r {np.median(ceil[ex]):.3f}, specific {np.median(ceil_s[ex]):.3f}")
    for a in ("E3", "T", "M2", "M2e"):
        P_(f"     {a:4s} median r {np.median(S[a][0][ex]):+.3f} ({np.median(S[a][0][ex]) / np.median(ceil[ex]):.0%} of ceiling) | specific r {np.median(S[a][1][ex]):+.3f}")
    V = {"X1": verdict(S["M2"][0][ex], S["E3"][0][ex]), "X2": verdict(S["M2"][0][ex], S["T"][0][ex]),
         "X3": verdict(S["M2"][0][ex], S["M2e"][0][ex]), "X4": verdict(S["M2"][1][ex], S["E3"][1][ex])}
    for tag, lab in (("X1", "PRIMARY M2 vs E3, r"), ("X2", "M2 vs T, r"), ("X3", "M2 vs M2e, r"), ("X4", "M2 vs E3, specific r")):
        v = V[tag]
        P_(f"  {tag} {lab:22s} better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g} -> {v['verdict']}")
    beta = np.array(json.load(open(os.path.join(OUTD, "confidence_model.json")))["beta"])
    F = conf_features(G, None, o); conf = np.c_[np.ones(len(F)), F] @ beta
    sp = spearmanr(conf[ex], S["M2"][0][ex])
    V["X5"] = dict(rho=float(sp.correlation), p=float(sp.pvalue), verdict="CALIBRATED" if sp.correlation > 0 and sp.pvalue < 0.05 else "NOT CALIBRATED")
    P_(f"  X5 confidence on EXAM: Spearman {sp.correlation:+.2f}, p {sp.pvalue:.2g} -> {V['X5']['verdict']}")
    q = np.quantile(conf[ex], [1 / 3, 2 / 3]); rr = S["M2"][0][ex]; ce = conf[ex]
    P_("     by confidence third: " + " | ".join(f"{lab} median r {np.median(rr[m]):+.3f}" for lab, m in
                                               (("low", ce <= q[0]), ("middle", (ce > q[0]) & (ce <= q[1])), ("high", ce > q[1]))))
    covered = np.array([o["conf"]["n_screens"][k] > 0 for k in range(len(genes))])
    for lab, m in (("with a record elsewhere", ex & covered), ("with none", ex & ~covered)):
        P_(f"     EXAM genes {lab:24s} n {m.sum():4d}: " + ", ".join(f"{a} {np.median(S[a][0][m]):+.3f}" for a in ("E3", "T", "M2", "M2e")))
    open(OUT, "w").write("\n".join(out) + "\n")
    json.dump(dict(verdicts=V, predictions_sha256=digest, genes=int(ex.sum()), scored=time.strftime("%Y-%m-%d %H:%M")), open(lock, "w"), indent=1)
    json.dump(dict(verdicts=V), open(os.path.join(OUTD, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    {"pull": pull, "build": build, "train": train, "dev": dev, "exam": exam}[sys.argv[1]]()
