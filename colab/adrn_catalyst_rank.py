"""ADRN AS A CATALYST RE-RANKER: trained on the solved half, handed bare candidate lists.

THE SETUP, AND WHAT MAKES IT A FAIR TEST.  The 10,977 annotated reactions are training data: reaction in,
catalyst out. The held-out reactions arrive as a reaction plus TEN CANDIDATE GENES IN NO PARTICULAR ORDER --
the language model's ranking and its confidence are withheld, so the network cannot inherit the ordering it
is being asked to beat. It must produce its own from the chemistry.

WHY ADRN AND NOT A PLAIN NETWORK.  ADRN is this project's dendritic primitive: multi-branch neurons with
NMDA-like nonlinearity, state that unfolds over internal steps rather than one forward pass, fast and slow
weights, episodic memory. It is used here exactly as it is -- an encoder feeds it, it runs its T steps, a
head reads out a score per (reaction, gene) pair.

THE TWO BASELINES THAT HAVE KILLED EVERY LEARNED MODEL ON THIS TASK, and both are here from the start.

    freq            rank candidates by how often each gene catalyses ANY reaction in training. A zero-
                    parameter bag. It beat the set transformer in cell_orphan_ties_nn (0.367 against 0.365)
                    and it is the reason "a learned model scored X" means nothing without it.
    adrn_shuffled_rxn   the trained network scoring each candidate against a DIFFERENT reaction's
                    chemistry. If it scores as well as the real pairing, the network learned which genes
                    are popular catalysts and never read the reaction -- the same failure the ESM work
                    exposed with its enzyme-only arm, and the single most likely way this run produces a
                    flattering number.

    adrn_untrained  random initialisation. Separates the architecture's inductive bias from learning.

REPRESENTATION, kept deliberately small because ADRN's eligibility tensor is (batch, neurons, branches,
inputs) and the input width is the term that hurts. A reaction is the mean of learned embeddings over its
participants; a candidate is a learned gene embedding; the pair is their concatenation plus their
elementwise product, so the network can express interaction rather than only additive preference.

TRAINING is a ranking loss, not classification: the true catalyst against sampled negatives, so the
objective matches the thing being measured. Held-out reactions are excluded from training entirely, and the
assertion is executed rather than assumed.

PREDECLARED, before any number:
    adrn beats the tournament order AND beats freq AND beats adrn_shuffled_rxn
        -> a trained model finally adds to the language model's ordering, and it does so by reading the
           chemistry rather than by memorising which genes are common.
    adrn beats freq but not the tournament order
        -> it learned something real and still less than the language model already knows. Consistent with
           every learned model tried on this task so far.
    adrn does not beat freq, or matches its own shuffled-reaction control
        -> it learned gene popularity. That is the sixth time a scorer has failed here and it would close
           the question of whether a trained re-ranker can beat the LLM ordering on this task.

-> outputs/orphan/adrn_catalyst_rank.json
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
from adrn import ADRN  # noqa: E402
from cell_orphan_fill import NONCAT, ABSTRACT  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
DIM = 24            # embedding width; pair vector is 3*DIM, which is ADRN's input width
NEUR = 48
BRANCH = 4
TSTEP = 4
EPOCHS = 3
BATCH = 96
NNEG = 6
LR = 3e-3
SEED = 257


class Scorer(nn.Module):
    """Embeddings -> pair vector -> ADRN -> one score. The network is ADRN as written; only the encoder
    and the readout are task-specific."""

    def __init__(self, n_met, n_gene):
        super().__init__()
        self.met = nn.Embedding(n_met + 1, DIM, padding_idx=0)
        self.gene = nn.Embedding(n_gene, DIM)
        nn.init.normal_(self.met.weight, std=0.1)
        nn.init.normal_(self.gene.weight, std=0.1)
        self.core = ADRN(3 * DIM, 1, n_neurons=NEUR, n_branch=BRANCH, T=TSTEP, use_memory=False)

    def rxn(self, mets):
        e = self.met(mets)                       # (B, L, DIM), 0 is padding
        m = (mets > 0).float().unsqueeze(-1)
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, r, g):
        ge = self.gene(g)
        x = torch.cat([r, ge, r * ge], -1)
        out = self.core(x)
        return (out[0] if isinstance(out, tuple) else out).squeeze(-1)


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("ADRN AS A CATALYST RE-RANKER -- trained on the solved half, given bare candidate lists")
    report("=" * 100)
    report("  The held-out reactions arrive as chemistry plus ten candidates in NO order: the language")
    report("  model's ranking and confidence are withheld, so the network cannot inherit what it is being")
    report("  asked to beat. Two controls decide whether any number here means anything -- a zero-parameter")
    report("  frequency bag, which has beaten every learned model on this task, and the same trained network")
    report("  scoring candidates against the WRONG reaction, which exposes learning gene popularity.")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    B = T.build()
    steps, cats, P = B["steps"], B["cats"], B["P"]
    meta = json.load(open(OUT / "holdout_meta.json"))
    keep = json.load(open(OUT / "rankv2_meta.json"))
    ans = json.load(open(OUT / "rankv2_answers.json"))
    held = {m["step"] for m in meta.values()}

    # ---- vocabularies -------------------------------------------------------------------------------
    metv, genev = {}, {}
    for i in cats:
        for p in P[i]:
            metv.setdefault(p, len(metv) + 1)          # 0 reserved for padding
        for g in cats[i]:
            genev.setdefault(g.upper(), len(genev))
    for pid, m in keep.items():
        for g in m["cands"]:
            genev.setdefault(g.upper(), len(genev))
    report(f"\n  vocabulary: {len(metv)} participants, {len(genev)} genes")

    def mvec(i, L=24):
        ids = [metv[p] for p in P[i] if p in metv][:L]
        return ids + [0] * (L - len(ids))

    train_i = [i for i in cats if P[i] and i not in held
               and steps[i].get("category") not in NONCAT + ABSTRACT]
    assert not (set(train_i) & held), "held-out reactions leaked into training"
    report(f"  {len(train_i)} training reactions (held-out excluded, asserted)")

    freq = Counter()
    for i in train_i:
        for g in cats[i]:
            freq[g.upper()] += 1

    Xtr = torch.tensor([mvec(i) for i in train_i], dtype=torch.long)
    pos = [[genev[g.upper()] for g in cats[i] if g.upper() in genev] for i in train_i]
    ok = [k for k, p in enumerate(pos) if p]
    Xtr, pos = Xtr[ok], [pos[k] for k in ok]
    report(f"  {len(Xtr)} usable training pairs")

    model = Scorer(len(metv), len(genev))
    untrained = Scorer(len(metv), len(genev))
    untrained.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ng = len(genev)
    report(f"  ADRN core: {NEUR} neurons, {BRANCH} branches, {TSTEP} internal steps, "
           f"{sum(p.numel() for p in model.parameters())} parameters")

    for ep in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            if len(idx) < 8:
                continue
            mets = Xtr[idx]
            r = model.rxn(mets)
            gp = torch.tensor([pos[k][rng.integers(len(pos[k]))] for k in idx], dtype=torch.long)
            gn = torch.tensor(rng.integers(0, ng, size=(len(idx), NNEG)), dtype=torch.long)
            sp = model(r, gp)
            sn = torch.stack([model(r, gn[:, j]) for j in range(NNEG)], 1)
            # ranking loss: the true catalyst must outscore each sampled negative
            loss = torch.nn.functional.softplus(sn - sp.unsqueeze(1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        report(f"    epoch {ep+1}/{EPOCHS}  loss {tot/max(nb,1):.4f}  ({time.time()-t0:.0f}s)")

    # ---- score the held-out candidate lists ----------------------------------------------------------
    rows = []
    for pid, m in keep.items():
        a = ans.get(pid)
        if not a or not a.get("tournament"):
            continue
        o = meta[m["orig"]]
        cands = [g for g in a["tournament"][:10] if g.upper() in genev]
        if len(cands) < 5:
            continue
        rows.append({"step": o["step"], "cands": cands, "truth": {g.upper() for g in o["truth"]},
                     "obscure": o["obscure"], "llm": a["tournament"][:10]})
    report(f"\n  {len(rows)} held-out reactions scored")

    @torch.no_grad()
    def rank(net, shuffle_rxn=False):
        out = []
        for k, r in enumerate(rows):
            src = rows[(k + 7) % len(rows)]["step"] if shuffle_rxn else r["step"]
            mv = torch.tensor([mvec(src)], dtype=torch.long)
            rr = net.rxn(mv)
            g = torch.tensor([genev[c.upper()] for c in r["cands"]], dtype=torch.long)
            sc = net(rr.expand(len(g), -1), g)
            order = [r["cands"][j] for j in torch.argsort(sc, descending=True).tolist()]
            out.append(order)
        return out

    model.eval()
    untrained.eval()
    ARMS = {
        "tournament (LLM)": [r["llm"] for r in rows],
        "adrn": rank(model),
        "adrn_shuffled_rxn": rank(model, shuffle_rxn=True),
        "adrn_untrained": rank(untrained),
        "freq": [sorted(r["cands"], key=lambda g: -freq.get(g.upper(), 0)) for r in rows],
        "random": [list(rng.permutation(r["cands"])) for r in rows],
    }
    ob = np.array([r["obscure"] for r in rows])
    report(f"    {'arm':<22}{'@1':>8}{'@3':>8}{'@5':>8}{'@1 obs':>9}")
    res = {}
    for a, orders in ARMS.items():
        h1 = np.array([bool(r["truth"] & set(o[:1])) for r, o in zip(rows, orders)])
        h3 = np.array([bool(r["truth"] & set(o[:3])) for r, o in zip(rows, orders)])
        h5 = np.array([bool(r["truth"] & set(o[:5])) for r, o in zip(rows, orders)])
        res[a] = {"@1": float(h1.mean()), "@3": float(h3.mean()), "@5": float(h5.mean()),
                  "@1_obscure": float(h1[ob].mean()) if ob.sum() else None}
        report(f"    {a:<22}{h1.mean():>8.3f}{h3.mean():>8.3f}{h5.mean():>8.3f}"
               f"{(h1[ob].mean() if ob.sum() else float('nan')):>9.3f}")

    report("\n  READING")
    A, L, F = res["adrn"]["@1"], res["tournament (LLM)"]["@1"], res["freq"]["@1"]
    S, U = res["adrn_shuffled_rxn"]["@1"], res["adrn_untrained"]["@1"]
    report(f"  adrn {A:.3f} | LLM {L:.3f} | freq {F:.3f} | shuffled-reaction {S:.3f} | untrained {U:.3f}")
    if A <= S + 0.02:
        report("  IT DID NOT READ THE REACTION. Scoring candidates against a DIFFERENT reaction's chemistry")
        report("  does as well, so the network ranked on gene identity alone -- it learned which genes are")
        report("  common catalysts. Whatever the headline, there is no chemistry in this model.")
    elif A <= F + 0.02:
        report("  IT DID NOT BEAT COUNTING. A zero-parameter frequency bag matches it, which is the same")
        report("  outcome the set transformer had here (0.365 against 0.367). Learned scorers keep")
        report("  rediscovering the prior over catalysts rather than the mapping from chemistry.")
    elif A <= L + 0.02:
        report("  IT LEARNED SOMETHING REAL AND STILL LESS THAN THE LANGUAGE MODEL KNOWS. It clears both")
        report("  controls, so it is reading the chemistry, but it does not add to an ordering built from")
        report("  worked examples. That is now the sixth scorer to land in this position.")
    else:
        report("  A TRAINED MODEL FINALLY BEATS THE ORDERING, and it clears the frequency bag and its own")
        report("  shuffled-reaction control, so it is reading chemistry rather than popularity.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "adrn_catalyst_rank", "n": len(rows), "n_train": len(Xtr),
               "arms": res, "log": log}, open(OUT / "adrn_catalyst_rank.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'adrn_catalyst_rank.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
