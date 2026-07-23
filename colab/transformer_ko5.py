"""transformer_ko5 -- v5: MORE DATA (already local). Every architecture lever (v2 objective, v3 edge-bias/features, v4 spatial) was marginal
because the bottleneck is DATA, not the model. The one big control-confirmed win was giving the transformer real interaction tokens. So v5
gives it 7x more PERTURBATION data: all 6 Perturb-seq cell lines we have (K562/RPE1/HepG2/Jurkat/HCT116/Melanoma = 9,471 KO-context examples
vs 1,400 K562 alone; the same knockouts measured in 3-4 contexts).

DESIGN: the winning transformer_ko architecture (set-of-tokens over a knockout's partners -> learned-retrieval embedding), trained as a
metric-learner where embedding cosine ~ WITHIN-LINE tide-removed response cosine, pooled across all 6 lines, with a learned CELL-LINE
embedding so the encoder can condition on context. The token context is cell-line-independent (DepMap + network + partners), so the extra
lines act as extra (knockout -> behaviour) supervision for the SHARED encoder. Eval is unchanged: held-out K562, tide-removed specific-mover
recall@50, retrieving from K562-TRAIN and transferring the K562 profile. STRICT held-out: the K562 test KO *genes* are excluded from EVERY
line's training (no cross-context leakage). CONTROL: the identical encoder trained on K562-ONLY -- isolates the pure effect of more data.
Deterministic per split; GPU if present (Colab) else CPU; SMOKE mode (V5_SMOKE=1) for a fast check.

HONEST PRIOR: more same-readout data should help the encoder generalise to held-out K562 KOs (more (KO->response) pairs, and the shared
component seen across contexts) -- this is the most likely of v5/v6/v7 to move the number -- but it is still bounded by the ~0.62 retrieval
oracle (the readout ceiling); it can close the model->oracle gap, not raise the oracle."""
import os, json, pickle
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
LINES = ["K562", "RPE1", "HepG2", "Jurkat", "HCT116", "Melanoma"]
TIDE_FRAC = 0.05
SMOKE = bool(os.environ.get("V5_SMOKE"))


def line_profiles(have):
    """per-line: normalised tide-removed |z| profile for each KO in `have` that the line measured. Returns {line: (ko_idx_list, Yn)}."""
    out = {}; hpos = {k: i for i, k in enumerate(have)}
    for L in LINES:
        d = pickle.load(open(f"{SP}/nlz_{L}.pkl", "rb"))
        genes = sorted({g for v in d.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
        kos = [k for k in have if k in d]
        M = np.zeros((len(kos), len(genes)), np.float32)
        for r, k in enumerate(kos):
            for g, z in d[k].items():
                M[r, gi[g]] = z
        A = np.abs(M)
        mover_freq = (A >= 1.0).mean(0)
        nontide = mover_freq < TIDE_FRAC
        Ys = A * nontide.astype(np.float32)
        Yn = Ys / (np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-9)
        out[L] = ([hpos[k] for k in kos], Yn.astype(np.float32))
    return out


def run(SEED, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, multi):
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); torch.set_num_threads(4)
    rng = np.random.RandomState(SEED)
    # held-out K562 test genes (excluded from EVERY line's training)
    scor = [k for k in H.scorable if k in hidx]
    perm = rng.permutation(len(scor)); n_test = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    k562_have = [i for i, k in enumerate(have) if k in H.ki]
    k562_train_all = [i for i in k562_have if have[i] not in test_set]
    vperm = rng.permutation(len(k562_train_all)); n_val = int(0.12 * len(k562_train_all))
    val = sorted(have[k562_train_all[i]] for i in vperm[:n_val] if have[k562_train_all[i]] in H.scorable); val_set = set(val)
    k562_tr = np.array([i for i in k562_train_all if have[i] not in val_set])          # K562 retrieval pool (train)

    Ft = torch.tensor(Fg).to(DEVICE); TI = torch.tensor(tok_idx); TR = torch.tensor(tok_role); TM = torch.tensor(tok_mask)
    FEATIN = Fg.shape[1] + 4; DM = 128
    # per-line training example indices (exclude test genes); torch tensors of within-`have` indices + line profile tensors
    use_lines = LINES if multi else ["K562"]
    line_examples = {}; line_Yn = {}; line_id = {L: i for i, L in enumerate(LINES)}
    for L in use_lines:
        idx_list, Yn = prof[L]
        keep = [j for j, gi_ in enumerate(idx_list) if have[gi_] not in test_set]      # drop test genes from this line too
        line_examples[L] = np.array([idx_list[j] for j in keep])
        line_Yn[L] = torch.tensor(Yn[keep]).to(DEVICE)

    def gather(idx):
        f = Ft[TI[idx].to(DEVICE)]
        roleoh = torch.nn.functional.one_hot(TR[idx].to(DEVICE), 4).float()
        return torch.cat([f, roleoh], -1), TM[idx].to(DEVICE)

    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM)
            lyr = torch.nn.TransformerEncoderLayer(DM, 4, 256, 0.1, batch_first=True, activation="gelu")
            s.enc = torch.nn.TransformerEncoder(lyr, 2)
            s.line = torch.nn.Embedding(len(LINES), DM)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tf, mask, lid):
            h = s.enc(s.proj(tf), src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            e = s.head(pooled + s.line(lid)); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed(model, idx, lid):
        tf, ms = gather(idx)
        return model(tf, ms, torch.full((len(idx),), lid, device=DEVICE))

    def embed_all_k562(model):
        model.eval()
        with torch.no_grad():
            E = np.zeros((len(have), 64), np.float32)
            for bs in range(0, len(have), 512):
                idx = torch.arange(bs, min(bs + 512, len(have)))
                E[bs:bs + len(idx)] = embed(model, idx, line_id["K562"]).cpu().numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def transfer(E, K=10):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            sim = E[k562_tr] @ E[i]; top = k562_tr[np.argsort(-sim)[:K]]
            return Yk[top].mean(0)
        return f

    EPOCHS = 12 if SMOKE else 120; PAT = 4 if SMOKE else 12
    model = Net().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, wait = -1.0, None, 0
    order = [L for L in use_lines for _ in range(1)]
    for ep in range(EPOCHS):
        model.train()
        rs = np.random.RandomState(SEED * 100 + ep)
        Ls = list(use_lines); rs.shuffle(Ls)
        for L in Ls:
            ex = line_examples[L]; Yn = line_Yn[L]; lid = line_id[L]
            if len(ex) < 8:
                continue
            pi = rs.permutation(len(ex))
            for bs in range(0, len(ex), 256):
                sel = pi[bs:bs + 256]
                if len(sel) < 8:
                    continue
                idx = torch.tensor(ex[sel])
                e = embed(model, idx, lid)
                Se = e @ e.T; St = Yn[sel] @ Yn[sel].T; dm = torch.eye(len(sel), device=DEVICE) * (-1e9)
                loss = -(torch.softmax(St / 0.1 + dm, 1) * torch.log_softmax(Se / 0.1 + dm, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0:
            vr, _ = mean_recall(H, val, transfer(embed_all_k562(model)))
            if vr > best:
                best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= PAT:
                break
    if bstate:
        model.load_state_dict(bstate)
    E = embed_all_k562(model)

    def tide_f(ko): return H.mover_freq
    def oracle_f(ko): return H._references(ko, 50)["ORACLE*"]
    r = {}
    r["recall"], _ = mean_recall(H, test, transfer(E), 50)
    r["tide"], _ = mean_recall(H, test, tide_f, 50)
    r["oracle"], _ = mean_recall(H, test, oracle_f, 50)
    r["n_train_examples"] = int(sum(len(line_examples[L]) for L in use_lines))
    return r


def main():
    H = Harness("K562")
    have0, X, Yk562 = build_xy(H)                              # K562 KOs with features + their |z|
    # union universe of KOs across all lines that have DepMap features
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True); syms = set(dv["syms"])
    allko = set()
    for L in LINES:
        allko |= set(pickle.load(open(f"{SP}/nlz_{L}.pkl", "rb")))
    have = sorted(k for k in allko if k in syms)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask = build_context(H, have)
    # K562 |z| aligned to `have` index (for retrieval transfer)
    Yk = np.zeros((len(have), H.nG), np.float32)
    for i, k in enumerate(have):
        if k in H.ki:
            Yk[i] = H.A[H.ki[k]]
    prof = line_profiles(have)
    print(f"v5: {len(have)} union KO universe; per-line examples: " + ", ".join(f"{L}={len(prof[L][0])}" for L in LINES))

    seeds = [0] if SMOKE else [0, 1, 2]
    multi_res, single_res = [], []
    for s in seeds:
        print(f"\n{'=' * 58}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        rm = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, multi=True)
        rs = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, multi=False)
        print(f"    K562-only (control)   recall {rs['recall']:.3f}   (train examples {rs['n_train_examples']})")
        print(f"    MULTI-LINE (6 lines)  recall {rm['recall']:.3f}   (train examples {rm['n_train_examples']})")
        print(f"    tide {rm['tide']:.3f}  oracle {rm['oracle']:.3f}")
        multi_res.append(rm['recall']); single_res.append(rs['recall'])
    mm, sm = float(np.mean(multi_res)), float(np.mean(single_res))
    d = [multi_res[i] - single_res[i] for i in range(len(seeds))]
    oracle = float(np.mean([r for r in [rm['oracle']]])) if SMOKE else float(np.mean([rm['oracle']]))
    print(f"\n{'=' * 58}\nSTABILITY across {len(seeds)} split(s):")
    print(f"    K562-only (control) : {sm:.3f} +/- {np.std(single_res):.3f}")
    print(f"    MULTI-LINE (6 lines): {mm:.3f} +/- {np.std(multi_res):.3f}")
    print(f"    MULTI - K562-only   : {np.mean(d):+.3f} +/- {np.std(d):.3f}  ({[round(x,3) for x in d]})")
    helps = np.mean(d) > 0.01 and (SMOKE or all(x > 0 for x in d))
    verdict = (
        f"TRANSFORMER v5 (transformer_ko5.py): MORE DATA -- the winning interaction-token transformer trained on all 6 Perturb-seq cell lines "
        f"(~9.5k KO-context examples) vs K562-only, both eval on held-out K562 (strict: test KO genes dropped from every line). "
        f"{'SMOKE.' if SMOKE else f'{len(seeds)} splits.'} K562-only {sm:.3f} -> MULTI-LINE {mm:.3f} = {np.mean(d):+.3f} "
        + ("(positive every split) -- more perturbation data DOES improve the K562 embedding: the shared knockout->response map is learned "
           "better from 7x examples + the same KO seen across contexts. Closes part of the model->oracle gap (data lever, as hypothesised)."
           if helps else
           "-- NO stable gain: 7x more perturbation data does NOT improve held-out K562 beyond the single-line model. The K562-specific "
           "signal is ~85% context-specific (cross_line probe), so other lines add mostly off-target supervision; the bottleneck is the "
           "READOUT/context, not the amount of same-readout data.") +
        " Still bounded by the ~0.62 retrieval oracle (raising that needs a different readout, not more of it). Deterministic per split; GPU-ready.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({"k562_only": round(sm, 4), "multi_line": round(mm, 4), "multi_minus_single": round(float(np.mean(d)), 4),
                   "per_split_delta": [round(x, 4) for x in d], "helps": bool(helps), "oracle": round(oracle, 4),
                   "verdict": verdict, "note": verdict}, open(OUT / "transformer_ko5.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_ko5.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
