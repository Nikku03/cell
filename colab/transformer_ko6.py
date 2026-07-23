"""transformer_ko6 -- v6: EXTERNAL DATA (fetched). v5 showed pooling our 6 local Perturb-seq lines helps a little. v6 tests whether adding
FETCHED external knockouts helps more. Checked HuggingFace: Norman is CRISPR-activation (wrong direction for a KO model -> skipped); the
Replogle K562-essential set (nicolas-lynn/replogle-perturb) is CRISPRi KNOCKDOWN in z-units -- the same readout/direction as ours -- with
630 knockouts we DON'T have (1,971 vs our 1,400). Fetched perturbation_zscores.parquet (157MB), extracted those 630 new KOs (colab/fetch...
-> nlz_Replogle.pkl).

DESIGN: the v5 multi-line metric-learning framework, ablated 6-LOCAL-LINES vs 6-LINES + REPLOGLE-630 (the extra KOs enter as an extra
training source; eval unchanged = held-out K562, strict cross-context exclusion of test genes). Isolates the pure effect of ~630 extra
same-readout K562 knockouts. Deterministic; GPU-ready; SMOKE (V6_SMOKE=1).

HONEST PRIOR: more same-direction K562 knockouts SHOULD help the encoder a little more than v5 (they are the most on-target extra data we can
fetch) -- but still bounded by the ~0.62 oracle."""
import os, json, pickle
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
LOCAL = ["K562", "RPE1", "HepG2", "Jurkat", "HCT116", "Melanoma"]
ALL_SRC = LOCAL + ["Replogle"]
TIDE_FRAC = 0.05
SMOKE = bool(os.environ.get("V6_SMOKE"))


def src_dict(name):
    return pickle.load(open(f"{SP}/nlz_{name}.pkl", "rb"))


def profiles(have):
    hpos = {k: i for i, k in enumerate(have)}; out = {}
    for L in ALL_SRC:
        d = src_dict(L)
        genes = sorted({g for v in d.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
        kos = [k for k in have if k in d]
        M = np.zeros((len(kos), len(genes)), np.float32)
        for r, k in enumerate(kos):
            for g, z in d[k].items():
                M[r, gi[g]] = z
        A = np.abs(M); nontide = (A >= 1.0).mean(0) < TIDE_FRAC
        Ys = A * nontide.astype(np.float32); Yn = Ys / (np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-9)
        out[L] = ([hpos[k] for k in kos], Yn.astype(np.float32))
    return out


def run(SEED, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, train_lines):
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); torch.set_num_threads(4)
    rng = np.random.RandomState(SEED)
    scor = [k for k in H.scorable if k in hidx]
    perm = rng.permutation(len(scor)); n_test = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    k562_have = [i for i, k in enumerate(have) if k in H.ki]
    k562_train_all = [i for i in k562_have if have[i] not in test_set]
    vperm = rng.permutation(len(k562_train_all)); n_val = int(0.12 * len(k562_train_all))
    val = sorted(have[k562_train_all[i]] for i in vperm[:n_val] if have[k562_train_all[i]] in H.scorable); val_set = set(val)
    k562_tr = np.array([i for i in k562_train_all if have[i] not in val_set])

    Ft = torch.tensor(Fg).to(DEVICE); TI = torch.tensor(tok_idx); TR = torch.tensor(tok_role); TM = torch.tensor(tok_mask)
    FEATIN = Fg.shape[1] + 4; DM = 128
    lid = {L: i for i, L in enumerate(ALL_SRC)}
    line_ex, line_Yn = {}, {}
    for L in train_lines:
        idxs, Yn = prof[L]
        keep = [j for j, gi_ in enumerate(idxs) if have[gi_] not in test_set]
        line_ex[L] = np.array([idxs[j] for j in keep]); line_Yn[L] = torch.tensor(Yn[keep]).to(DEVICE)

    def gather(idx):
        f = Ft[TI[idx].to(DEVICE)]; roleoh = torch.nn.functional.one_hot(TR[idx].to(DEVICE), 4).float()
        return torch.cat([f, roleoh], -1), TM[idx].to(DEVICE)

    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM)
            s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(DM, 4, 256, 0.1, batch_first=True, activation="gelu"), 2)
            s.line = torch.nn.Embedding(len(ALL_SRC), DM)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, tf, mask, li):
            h = s.enc(s.proj(tf), src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            e = s.head(pooled + s.line(li)); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_batch(model, idx, li):
        tf, ms = gather(idx)
        return model(tf, ms, torch.full((len(idx),), li, device=DEVICE))

    def embed_all(model):
        model.eval()
        with torch.no_grad():
            E = np.zeros((len(have), 64), np.float32)
            for bs in range(0, len(have), 512):
                idx = torch.arange(bs, min(bs + 512, len(have)))
                E[bs:bs + len(idx)] = embed_batch(model, idx, lid["K562"]).cpu().numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def transfer(E, K=10):
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            sim = E[k562_tr] @ E[i]; top = k562_tr[np.argsort(-sim)[:K]]; return Yk[top].mean(0)
        return f

    EP = 12 if SMOKE else 120; PAT = 4 if SMOKE else 12
    model = Net().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, wait = -1.0, None, 0
    for ep in range(EP):
        model.train(); rs = np.random.RandomState(SEED * 100 + ep); Ls = list(train_lines); rs.shuffle(Ls)
        for L in Ls:
            ex = line_ex[L]; Yn = line_Yn[L]
            if len(ex) < 8:
                continue
            pi = rs.permutation(len(ex))
            for bs in range(0, len(ex), 256):
                sel = pi[bs:bs + 256]
                if len(sel) < 8:
                    continue
                idx = torch.tensor(ex[sel]); e = embed_batch(model, idx, lid[L])
                Se = e @ e.T; St = Yn[sel] @ Yn[sel].T; dm = torch.eye(len(sel), device=DEVICE) * (-1e9)
                loss = -(torch.softmax(St / 0.1 + dm, 1) * torch.log_softmax(Se / 0.1 + dm, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0:
            vr, _ = mean_recall(H, val, transfer(embed_all(model)))
            if vr > best:
                best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= PAT:
                break
    if bstate:
        model.load_state_dict(bstate)
    E = embed_all(model)
    r = {}
    r["recall"], _ = mean_recall(H, test, transfer(E), 50)
    r["oracle"], _ = mean_recall(H, test, lambda ko: H._references(ko, 50)["ORACLE*"], 50)
    r["n_train"] = int(sum(len(line_ex[L]) for L in train_lines))
    return r


def main():
    H = Harness("K562")
    have0, X, Yk562 = build_xy(H)
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True); syms = set(dv["syms"])
    allko = set()
    for L in ALL_SRC:
        allko |= set(src_dict(L))
    have = sorted(k for k in allko if k in syms)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask = build_context(H, have)
    Yk = np.zeros((len(have), H.nG), np.float32)
    for i, k in enumerate(have):
        if k in H.ki:
            Yk[i] = H.A[H.ki[k]]
    prof = profiles(have)
    print(f"v6: {len(have)} KO universe (incl Replogle); sources: " + ", ".join(f"{L}={len(prof[L][0])}" for L in ALL_SRC))

    seeds = [0] if SMOKE else [0, 1, 2]
    v5r, v6r = [], []
    for s in seeds:
        print(f"\n{'=' * 56}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        r5 = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, LOCAL)
        r6 = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, Yk, prof, ALL_SRC)
        print(f"    6 local lines (v5)        recall {r5['recall']:.3f}   (train {r5['n_train']})")
        print(f"    6 lines + Replogle (v6)   recall {r6['recall']:.3f}   (train {r6['n_train']})")
        print(f"    oracle {r6['oracle']:.3f}")
        v5r.append(r5['recall']); v6r.append(r6['recall'])
    m5, m6 = float(np.mean(v5r)), float(np.mean(v6r))
    d = [v6r[i] - v5r[i] for i in range(len(seeds))]
    print(f"\n{'=' * 56}\nSTABILITY across {len(seeds)} split(s):")
    print(f"    6 local lines (v5)      : {m5:.3f} +/- {np.std(v5r):.3f}")
    print(f"    6 lines + Replogle (v6) : {m6:.3f} +/- {np.std(v6r):.3f}")
    print(f"    v6 - v5 (external data) : {np.mean(d):+.3f} +/- {np.std(d):.3f}  ({[round(x,3) for x in d]})")
    helps = np.mean(d) > 0.01 and (SMOKE or all(x > 0 for x in d))
    verdict = (
        f"TRANSFORMER v6 (transformer_ko6.py): EXTERNAL DATA -- +630 fetched Replogle K562 knockdowns (CRISPRi, same direction as ours; Norman "
        f"was CRISPRa so skipped) added to the v5 multi-line pool. {len(seeds)} splits, held-out K562. 6-lines {m5:.3f} -> +Replogle {m6:.3f} "
        f"= {np.mean(d):+.3f} "
        + ("(positive every split) -- the fetched external K562 knockouts add supervision the encoder uses: more same-readout data continues "
           "to help, so v5's data lever was not saturated by our 6 lines."
           if helps else
           "-- NO further gain from the external 630 KOs beyond v5's 6 lines: the multi-line pool already saturated the same-readout data "
           "lever; extra K562 knockdowns are largely redundant with what the encoder already learned. The remaining gap to the oracle is not "
           "closable by more of the SAME readout -- it needs a DIFFERENT readout (transient kinetics).") +
        " Bounded by the ~0.62 oracle. Deterministic; GPU-ready.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({"v5_six_lines": round(m5, 4), "v6_plus_replogle": round(m6, 4), "v6_minus_v5": round(float(np.mean(d)), 4),
                   "per_split_delta": [round(x, 4) for x in d], "helps": bool(helps), "oracle": round(float(np.mean([r6['oracle']])), 4),
                   "verdict": verdict, "note": verdict}, open(OUT / "transformer_ko6.json", "w"), indent=1)
        print("\n  -> outputs/orphan/transformer_ko6.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
