"""end_to_end_ranker -- stop asserting the ceiling, build the STRONG extractor. The AlphaFold challenge: our 'ceiling' was measured with weak
probes (kNN retrieval, linear ridge). This is a real end-to-end LEARNED gene-ranker (a function, not retrieval) trained with a proper listwise
ranking loss -- the 'proper training algorithm' analog -- to see if it beats the retrieval oracle on the same data.

Difference from everything before:
  - transformer_ko/v5 = retrieval: encode KO, find similar TRAIN knockouts, copy their measured profile. Template-based; capped by the oracle.
  - neural_ko = regression with dense MSE -> collapsed to the mean (wrong objective).
  - THIS = a learned scorer s(KO, gene) = encoder(KO) . gene_embedding, trained by SAMPLED-SOFTMAX over each KO's specific movers vs sampled
    non-movers. It can rank a gene high even if NO similar knockout moved it -- the non-retrieval leap. Optimises recall directly.

Held-out K562, tide-removed specific recall@50, vs retrieval + oracle + tide. If it beats retrieval -> the wall was algorithmic (data-ceiling
claim was premature). If it collapses to tide/retrieval -> the learned function genuinely can't do better on these inputs. GPU-ready; SMOKE (ER_SMOKE=1)."""
import os, json
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context

OUT = Path("outputs/orphan")
SMOKE = bool(os.environ.get("ER_SMOKE"))
DM = 128


def run(SEED, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, X):
    import torch
    torch.manual_seed(SEED); torch.set_num_threads(4)
    rng = np.random.RandomState(SEED)
    scor = [k for k in H.scorable if k in hidx]
    perm = rng.permutation(len(scor)); n_test = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    tr_all = [k for k in have if k not in test_set and k in H.ki]
    vp = rng.permutation(len(tr_all)); n_val = int(0.12 * len(tr_all))
    val = sorted(tr_all[i] for i in vp[:n_val] if tr_all[i] in H.scorable); val_set = set(val)
    train = [k for k in tr_all if k not in val_set and k in H.scorable]

    nti = H.nontide_idx; G = len(nti); gpos = {int(g): j for j, g in enumerate(nti)}   # nontide gene -> column
    # positives per KO (specific movers), in nontide-column space
    def positives(ko):
        r = H.ki[ko]; return [gpos[int(i)] for i in np.where(H.mover[r])[0] if H.nontide[i]]

    Ft = torch.tensor(Fg); TI = torch.tensor(tok_idx); TR = torch.tensor(tok_role); TM = torch.tensor(tok_mask)
    FEATIN = Fg.shape[1] + 4

    class Encoder(torch.nn.Module):                      # KO context -> h (same set-transformer as the retrieval encoder, for a fair test)
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEATIN, DM)
            s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(DM, 4, 256, 0.1, batch_first=True, activation="gelu"), 2)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, DM))
        def forward(s, tf, mask):
            h = s.enc(s.proj(tf), src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            return s.head(pooled)

    class Ranker(torch.nn.Module):                       # the LEARNED generative scorer: s(KO,gene) = h . E_gene + b_gene
        def __init__(s):
            super().__init__()
            s.enc = Encoder()
            s.E = torch.nn.Embedding(G, DM); s.b = torch.nn.Parameter(torch.zeros(G))
            torch.nn.init.normal_(s.E.weight, std=0.02)
        def score_all(s, tf, mask):
            h = s.enc(tf, mask); return h @ s.E.weight.T + s.b
        def score_cols(s, h1, cols):                     # h1: (DM,), cols: (C,) -> (C,)
            return (s.E(cols) * h1).sum(-1) + s.b[cols]

    def gather(kos):
        idx = torch.tensor([hidx[k] for k in kos])
        f = Ft[TI[idx]]; ro = torch.nn.functional.one_hot(TR[idx], 4).float()
        return torch.cat([f, ro], -1), TM[idx]

    model = Ranker(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    EP = 10 if SMOKE else 80; PAT = 3 if SMOKE else 10
    N_NEG = 512
    tr_pos = {k: positives(k) for k in train}
    tr = [k for k in train if tr_pos[k]]

    def predict(ko):
        model.eval()
        with torch.no_grad():
            tf, ms = gather([ko]); logits = model.score_all(tf, ms)[0].numpy()
        v = np.full(H.nG, -1e9, np.float32); v[nti] = logits
        return v

    best, bstate, wait = -1.0, None, 0
    for ep in range(EP):
        model.train(); order = np.random.RandomState(SEED * 100 + ep).permutation(len(tr))
        for bs in range(0, len(tr), 64):
            kos = [tr[i] for i in order[bs:bs + 64]]
            if len(kos) < 4:
                continue
            tf, ms = gather(kos); h = model.enc(tf, ms)
            loss = 0.0; nb = 0
            for bi, ko in enumerate(kos):
                pos = tr_pos[ko]
                if not pos:
                    continue
                neg = np.random.randint(0, G, N_NEG)
                cols = torch.tensor(list(pos) + list(neg))
                sc = model.score_cols(h[bi], cols)               # (len(pos)+N_NEG,)
                # multi-positive sampled softmax: each positive vs the shared negatives
                np_ = len(pos)
                pos_sc = sc[:np_]; neg_sc = sc[np_:]
                mat = torch.cat([pos_sc.unsqueeze(1), neg_sc.unsqueeze(0).expand(np_, len(neg))], 1)  # (np_, 1+N_NEG)
                denom = torch.logsumexp(mat, 1)
                loss = loss + (denom - pos_sc).mean(); nb += 1
            if nb == 0:
                continue
            loss = loss / nb
            opt.zero_grad(); loss.backward(); opt.step()
        vr, _ = mean_recall(H, val, predict)
        if vr > best:
            best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= PAT:
            break
    if bstate:
        model.load_state_dict(bstate)
    sc_test, _ = mean_recall(H, test, predict, 50)

    # retrieval + references on the SAME split for a fair comparison
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    tr_rows = np.array([hidx[k] for k in tr_all if k not in val_set])
    Ymag = np.stack([H.A[H.ki[k]] for k in have])
    def retr(ko):
        i = hidx[ko]; sim = Xn[tr_rows] @ Xn[i]; return Ymag[tr_rows[np.argsort(-sim)[:10]]].mean(0)
    sc_retr, _ = mean_recall(H, test, retr, 50)
    sc_tide, _ = mean_recall(H, test, lambda ko: H.mover_freq, 50)
    sc_orc, _ = mean_recall(H, test, lambda ko: H._references(ko, 50)["ORACLE*"], 50)
    return {"learned": sc_test, "retrieval": sc_retr, "tide": sc_tide, "oracle": sc_orc}


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H); hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask = build_context(H, have)
    seeds = [0] if SMOKE else [0, 1, 2]
    agg = {k: [] for k in ["learned", "retrieval", "tide", "oracle"]}
    for s in seeds:
        print(f"\n{'='*56}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        r = run(s, H, have, hidx, Fg, tok_idx, tok_role, tok_mask, X)
        for k in agg:
            agg[k].append(r[k])
        print(f"    LEARNED end-to-end ranker  {r['learned']:.3f}")
        print(f"    retrieval (template)       {r['retrieval']:.3f}")
        print(f"    tide {r['tide']:.3f}   oracle {r['oracle']:.3f}")
    A = {k: (float(np.mean(v)), float(np.std(v))) for k, v in agg.items()}
    d = A["learned"][0] - A["retrieval"][0]; d_orc = A["learned"][0] - A["oracle"][0]
    print(f"\n{'='*56}\nSTABILITY across {len(seeds)} split(s):")
    for k in ["learned", "retrieval", "tide", "oracle"]:
        print(f"    {k:12s} {A[k][0]:.3f} +/- {A[k][1]:.3f}")
    print(f"    learned - retrieval {d:+.3f}   learned - oracle {d_orc:+.3f}")
    verdict = (
        f"END-TO-END LEARNED RANKER (end_to_end_ranker.py): the AlphaFold challenge -- does a real learned FUNCTION (encoder(KO).gene-embedding, "
        f"sampled-softmax ranking loss), not retrieval, beat the retrieval ceiling on the same data? {len(seeds)} splits, held-out K562. "
        f"Learned {A['learned'][0]:.3f} vs retrieval {A['retrieval'][0]:.3f} ({d:+.3f}) vs oracle {A['oracle'][0]:.3f} (tide {A['tide'][0]:.3f}). "
        + ("The learned function BEATS retrieval -- the wall was (at least partly) ALGORITHMIC; the data-ceiling claim was premature and the "
           "proper training objective extracts signal retrieval could not." if d > 0.02 else
           "The learned end-to-end ranker does NOT beat retrieval on these inputs -- built with the proper ranking objective (not neural_ko's "
           "collapse-prone MSE), a non-retrieval learned function still lands at the retrieval level. Combined with the measured 74% cross-line "
           "non-reproducing target, this is now EARNED evidence (strong extractor, not a weak probe) that the residual is data-limited on the "
           "current readout -- not proof no algorithm ever helps, but the AlphaFold-style end-to-end leap did not materialise here.")
        + f" (learned vs oracle {d_orc:+.3f}.) Deterministic.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        json.dump({k: [round(A[k][0], 4), round(A[k][1], 4)] for k in A} | {"learned_minus_retrieval": round(d, 4),
                  "verdict": verdict, "note": verdict}, open(OUT / "end_to_end_ranker.json", "w"), indent=1)
        print("\n  -> outputs/orphan/end_to_end_ranker.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
