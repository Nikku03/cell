"""A LEARNED REPRESENTATION INSTEAD OF A BIGGER HEAD: ESM-2 embeddings for catalyst prediction.

WHERE THIS SITS.  Three things have now failed to pick the catalyst out of a shortlist: hand-built docking
statistics (all in [0.450, 0.549]), a learned combination of them (nothing cleared its permutation null), and
co-expression / co-dependency evidence (+0.017, p=0.30). Every one of those failures was about the
REPRESENTATION rather than the model -- eight summary numbers off a rigid-body scan do not contain which
enzyme acts, and no head on top of them conjures it.

So this changes the representation, not the head. ESM-2 embeddings are learned from ~250M sequences and encode
fold, family and active-site context without anyone writing an annotation down. No docking at all.

IT ALSO GETS 37x MORE DATA, which matters after a 60-group experiment.  Docking needed both partners to be
compact enough for a 218 A box; sequence does not. 2,231 reactions have a known catalyst and a protein
substrate with sequences, against 60 that were dockable.

THE CONTROL THAT DECIDES THIS, and it is not the permutation.  A model given (enzyme, substrate) can score
well by learning WHICH PROTEINS ARE ENZYMES and never looking at the substrate at all -- "kinases catalyse
things" is worth a lot of AUC and is not enzyme-substrate matching. So:

    enzyme_only     the identical model with the substrate embedding REPLACED BY ZEROS. If the full model does
                    not beat this, it never used the substrate, and it is a catalyst-prior in disguise.
    shuffled_sub    enzyme paired with a DIFFERENT reaction's substrate. Same question from the other side.

THE SPLIT THAT DECIDES WHETHER IT IS USEFUL.  Filling the 9,186 orphans means naming enzymes for reactions
whose catalyst is unknown, so the question is whether the model transfers to enzymes it has not seen catalyse
anything. Two split schemes are reported and the GAP BETWEEN THEM IS THE RESULT:

    reaction_disjoint   folds split by reaction. The enzyme can appear as a positive in training. Flattering,
                        and the number most papers would quote.
    enzyme_disjoint     folds split by TRUE CATALYST GENE, so no test enzyme is ever a training positive, and
                        decoys are drawn from the SAME FOLD so true and decoy have equal training exposure.
                        This is the number that forecasts orphan work.

FREQUENCY BASELINE, because it beat everything hand-built before.  `cell_orphan_ties` found that "how many
reactions does this gene already catalyse" was the strongest tie-breaker of six. It is computed here from
TRAINING reactions only. Under the enzyme-disjoint split it is uninformative by construction -- which is the
point, and is why that split isolates chemistry from popularity.

PREDECLARED, before any number:
    esm beats freq AND beats enzyme_only, under the ENZYME-DISJOINT split
        -> the model is matching enzyme to substrate, and it transfers to unseen enzymes. This is the first
           thing in this project that would actually move the orphan estimate off 0.463.
    esm beats freq only under reaction_disjoint, and collapses under enzyme_disjoint
        -> it memorised which genes are catalysts. Useful for re-annotating known enzymes, useless for the
           gaps, and the honest number to quote is the enzyme-disjoint one.
    esm ~ enzyme_only
        -> the substrate is being ignored; whatever the AUC, it is a catalyst-prior and not a pairing model.
    esm ~ freq
        -> a 250M-sequence representation adds nothing over counting, which is where six hand-built signals
           and a set transformer already landed on the tie-break task.
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_ties_nn as T

OUT = A.OUT
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
FASTA = SP / "human_seqs.fasta.gz"
EMB = SP / "esm2_35M_embed.npz"
MODEL = "esm2_t12_35M_UR50D"
MAXLEN = 1022
N_DECOY = 9
N_FOLD = 5
SEEDS = (0, 1, 2)
EPOCHS = 25
N_PERM = 20
SEED = 5
JACC_MIN = 0.02      # 5-mer Jaccard. Calibrated on known pairs: ~93% identity reads 0.610, ~84% reads
                     # 0.307, ~50% reads 0.03, unrelated reads 0.000. So 0.02 catches paralogs down to
                     # roughly 50% identity. It does NOT catch remote homology below that, which is why the
                     # homology-split number is still an UPPER bound rather than a clean one.
DECOY_SEED = 99      # FIXED and independent of the model seed: every arm must see the SAME candidate sets,
                     # or an arm's margin is partly a luckier decoy draw rather than a better model


def sequences():
    """gene -> longest reviewed human sequence (canonical-isoform proxy)."""
    seq, h, buf = {}, None, []

    def flush():
        if not h:
            return
        g = next((t[3:] for t in h.split() if t.startswith("GN=")), None)
        if g:
            s = "".join(buf)
            if g not in seq or len(s) > len(seq[g]):
                seq[g] = s
    for l in gzip.open(FASTA, "rt"):
        if l.startswith(">"):
            flush()
            h, buf = l, []
        else:
            buf.append(l.strip())
    flush()
    return seq


def embed():
    import torch
    import esm
    seq = sequences()
    B = T.build()
    cats, steps = B["cats"], B["steps"]
    PROT = {"PROTEIN", "COMPLEX", "SET"}
    need = {c for i in cats for c in cats[i] if c in seq}
    for i in cats:
        s = steps[i]
        for w in ("in", "out"):
            for p in (s.get(w) or []):
                if p.get("type") in PROT:
                    need |= {g for g in (p.get("genes") or []) if g in seq}
    need = sorted(need)
    print(f"  embedding {len(need):,} proteins with {MODEL}")

    model, alphabet = esm.pretrained.__dict__[MODEL]()
    bc = alphabet.get_batch_converter()
    model.eval()
    torch.set_num_threads(4)
    layer = model.num_layers

    done = {}
    if EMB.exists():
        d = np.load(EMB, allow_pickle=True)
        done = {g: v for g, v in zip(d["genes"], d["E"])}
        print(f"  resuming: {len(done):,} already embedded")
    todo = [g for g in need if g not in done]
    order = sorted(todo, key=lambda g: len(seq[g]))     # batch by length: less padding, much faster
    t0 = time.time()
    BATCH = 8
    for k in range(0, len(order), BATCH):
        chunk = order[k:k + BATCH]
        data = [(g, seq[g][:MAXLEN]) for g in chunk]
        _, _, toks = bc(data)
        with torch.no_grad():
            rep = model(toks, repr_layers=[layer])["representations"][layer]
        for j, g in enumerate(chunk):
            n = min(len(seq[g]), MAXLEN)
            done[g] = rep[j, 1:n + 1].mean(0).numpy().astype(np.float32)   # skip BOS, mean over residues
        if k % 400 == 0:
            el = time.time() - t0
            print(f"    {k + len(chunk):,}/{len(order):,}  {el:.0f}s elapsed", flush=True)
            gs = sorted(done)
            np.savez_compressed(EMB, genes=np.array(gs), E=np.stack([done[g] for g in gs]))
    gs = sorted(done)
    np.savez_compressed(EMB, genes=np.array(gs), E=np.stack([done[g] for g in gs]))
    print(f"  wrote {len(gs):,} embeddings ({np.stack([done[g] for g in gs]).shape[1]}-dim) -> {EMB}")
    return 0


def family_of(gene_list, seq):
    """Cluster catalysts so a split can be HOMOLOGY-disjoint, not merely gene-disjoint.

    Gene-disjoint is not family-disjoint: PRKACA in train and PRKACB in test are ~93% identical, and a model
    that merely recognises the paralog has not generalised to an unseen enzyme -- which is exactly what
    filling orphans requires. Two edges are drawn and the connected components are the clusters:
        sequence   5-mer Jaccard >= JACC_MIN
        symbol     same family stem (PRKACA/PRKACB -> PRKAC, SLC7A5/SLC7A8 -> SLC7)
    The symbol rule over-merges on purpose. Over-merging makes the split HARDER, which is the safe direction
    for a number that is meant to be believed.
    """
    import re
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    K, NB = 5, 1 << 20
    rows, cols = [], []
    for i, g in enumerate(gene_list):
        sq = seq[g][:MAXLEN]
        ks = {hash(sq[j:j + K]) % NB for j in range(len(sq) - K + 1)}
        rows += [i] * len(ks)
        cols += list(ks)
    A = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(gene_list), NB))
    n = np.asarray(A.sum(1)).ravel()
    N = len(gene_list)
    ei, ej = [], []
    CH = 512
    for a in range(0, N, CH):
        blk = (A[a:a + CH] @ A.T).toarray()
        for r in range(blk.shape[0]):
            i = a + r
            inter = blk[r]
            jacc = inter / np.maximum(n[i] + n - inter, 1e-9)
            hit = np.where(jacc >= JACC_MIN)[0]
            for j in hit:
                if j > i:
                    ei.append(i); ej.append(int(j))

    def stem(g):
        t = re.sub(r"\d+$", "", g)
        if len(t) > 3:
            t = re.sub(r"[A-Z]$", "", t)
        return t if len(t) >= 3 else g
    by_stem = {}
    for i, g in enumerate(gene_list):
        by_stem.setdefault(stem(g), []).append(i)
    for v in by_stem.values():
        for j in v[1:]:
            ei.append(v[0]); ej.append(j)
    G = csr_matrix((np.ones(len(ei)), (ei, ej)), shape=(N, N))
    ncomp, lab = connected_components(G, directed=False)
    return {g: int(l) for g, l in zip(gene_list, lab)}, ncomp


def load_bench():
    d = np.load(EMB, allow_pickle=True)
    E = {g: v for g, v in zip(d["genes"], d["E"])}
    B = T.build()
    cats, steps = B["cats"], B["steps"]
    PROT = {"PROTEIN", "COMPLEX", "SET"}
    rx = []
    for i in cats:
        s = steps[i]
        cs = [c for c in cats[i] if c in E]
        if not cs:
            continue
        subs = [g for w in ("in", "out") for p in (s.get(w) or []) if p.get("type") in PROT
                for g in (p.get("genes") or []) if g in E and g not in cats[i]]
        if not subs:
            continue
        rx.append({"rx": int(i), "cat": cs[0], "sub": subs[0]})
    vocab = sorted({c for i in cats for c in cats[i] if c in E})
    return E, rx, vocab, cats


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "embed":
        return embed()

    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    torch.set_num_threads(4)

    report("=" * 100)
    report("ESM-2 EMBEDDINGS FOR CATALYST PREDICTION -- a learned representation, not a bigger head")
    report("=" * 100)
    report("  Docking statistics, a learned combination of them, and cell-line evidence have all failed on")
    report("  this task. Every failure was about the REPRESENTATION. This changes that and drops docking.")
    report("  THE DECIDING CONTROL is not the permutation: a model can score well by learning WHICH PROTEINS")
    report("  ARE ENZYMES and never reading the substrate. enzyme_only zeroes the substrate to expose that.")
    report("  THE DECIDING SPLIT is enzyme-disjoint -- no test enzyme is ever a training positive -- because")
    report("  filling orphans means naming enzymes never seen catalysing anything.")

    E, rx, vocab, cats = load_bench()
    dim = len(next(iter(E.values())))
    report(f"\n  {len(rx):,} reactions | catalyst vocabulary {len(vocab):,} | ESM {MODEL}, {dim}-dim")

    Emat = {g: E[g] for g in E}
    mu = np.mean(np.stack(list(Emat.values())), 0)
    sd = np.std(np.stack(list(Emat.values())), 0) + 1e-6

    def vec(g):
        return (Emat[g] - mu) / sd

    rng = np.random.default_rng(SEED)
    cat_of = np.array([r["cat"] for r in rx])
    seq = sequences()
    cvocab = sorted({r["cat"] for r in rx} | {r["sub"] for r in rx})
    fam, ncomp = family_of([g for g in cvocab if g in seq], seq)
    report(f"  homology clusters over {len(cvocab):,} catalysts+substrates: {ncomp:,} components "
           f"(5-mer Jaccard >= {JACC_MIN} OR shared symbol stem)")
    biggest = max(np.bincount(np.array(list(fam.values()))))
    report(f"    largest cluster {biggest} genes | singletons "
           f"{int((np.bincount(np.array(list(fam.values()))) == 1).sum()):,}")
    fold_by = {"reaction_disjoint": np.arange(len(rx)),
               "enzyme_disjoint": np.array([sorted(set(cat_of)).index(c) for c in cat_of]),
               "homology_disjoint": np.array([fam.get(c, -1 - k) for k, c in enumerate(cat_of)])}

    class Pair(nn.Module):
        """score(enzyme, substrate). The interaction terms are what make it a PAIRING model rather than two
        independent scores -- without e*s and |e-s| an MLP on the concatenation can still only add them."""
        def __init__(s, d, h=128):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(4 * d, h), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

        def forward(s, e, sub):
            return s.net(torch.cat([e, sub, e * sub, (e - sub).abs()], -1)).squeeze(-1)

    def build_groups(idx, pool, fold):
        """one group per reaction: true catalyst + N_DECOY DISTINCT decoys from `pool`.

        Keyed on (DECOY_SEED, fold, reaction) rather than on a running RNG, so every arm and every model seed
        is scored on byte-identical candidate sets. With a running RNG the arms would differ by their decoy
        draw as well as by their model, and a 0.03 margin could be either."""
        out = []
        pool = np.array(sorted(pool))
        for i in idx:
            r = rx[i]
            rs = np.random.default_rng((DECOY_SEED, int(fold), int(r["rx"])))
            bad = set(cats[r["rx"]]) | {r["cat"]}
            ok = pool[~np.isin(pool, list(bad))]
            if len(ok) < N_DECOY:
                continue
            d = list(rs.choice(ok, N_DECOY, replace=False))
            out.append({"i": i, "cands": [r["cat"]] + d, "sub": r["sub"], "y": 0})
        return out

    def tensors(groups, mode):
        Ee, Es, Y = [], [], []
        for g in groups:
            e = np.stack([vec(c) for c in g["cands"]])
            s_ = np.repeat(vec(g["sub"])[None, :], len(g["cands"]), 0)
            if mode == "enzyme_only":
                s_ = np.zeros_like(s_)
            Ee.append(e); Es.append(s_)
            Y.append(np.array([1] + [0] * (len(g["cands"]) - 1)))
        return (torch.tensor(np.stack(Ee), dtype=torch.float32),
                torch.tensor(np.stack(Es), dtype=torch.float32),
                torch.tensor(np.stack(Y), dtype=torch.float32))

    def fit_eval(trg, teg, mode, seed, train=True):
        torch.manual_seed(seed)
        m = Pair(dim)
        Etr, Str, Ytr = tensors(trg, mode)
        Ete, Ste, Yte = tensors(teg, mode)
        if train:
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
            n = len(trg)
            rs = np.random.default_rng(seed)
            for ep in range(EPOCHS):
                m.train()
                perm = rs.permutation(n)
                for k in range(0, n, 64):
                    b = perm[k:k + 64]
                    opt.zero_grad()
                    sc = m(Etr[b], Str[b])
                    loss = -(torch.log_softmax(sc, 1) * Ytr[b]).sum(1).mean()
                    loss.backward()
                    opt.step()
        m.eval()
        with torch.no_grad():
            sc = m(Ete, Ste).numpy()
        return sc, Yte.numpy()

    def run(scheme, mode, seed, shuffle_sub=False, train=True):
        gk = GroupKFold(n_splits=N_FOLD)
        gid = fold_by[scheme]
        rs = np.random.default_rng(seed)
        ys, ps, t1 = [], [], []
        for fold, (tr, te) in enumerate(gk.split(np.arange(len(rx)), groups=gid)):
            # decoys drawn from the SAME fold under the enzyme-disjoint split, so the true catalyst and its
            # decoys have identical training exposure -- otherwise decoys are training positives and the
            # true one never is, which is a leak pointing the wrong way
            held = scheme in ("enzyme_disjoint", "homology_disjoint")
            pool_tr = sorted({rx[i]["cat"] for i in tr}) if held else vocab
            pool_te = sorted({rx[i]["cat"] for i in te}) if held else vocab
            trg = build_groups(tr, pool_tr, fold)
            teg = build_groups(te, pool_te, 100 + fold)
            if shuffle_sub:
                subs = [g["sub"] for g in teg]
                rs.shuffle(subs)
                for g, s_ in zip(teg, subs):
                    g["sub"] = s_
            sc, Y = fit_eval(trg, teg, mode, seed, train=train)
            ys.append(Y.ravel()); ps.append(sc.ravel())
            t1 += list((sc.argmax(1) == 0).astype(float))
        return float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))), float(np.mean(t1))

    def freq_arm(scheme, seed):
        gk = GroupKFold(n_splits=N_FOLD)
        gid = fold_by[scheme]
        rs = np.random.default_rng(seed)
        ys, ps, t1 = [], [], []
        for fold, (tr, te) in enumerate(gk.split(np.arange(len(rx)), groups=gid)):
            cnt = {}
            for i in tr:
                cnt[rx[i]["cat"]] = cnt.get(rx[i]["cat"], 0) + 1
            pool_te = (sorted({rx[i]["cat"] for i in te})
                       if scheme in ("enzyme_disjoint", "homology_disjoint") else vocab)
            for g in build_groups(te, pool_te, 100 + fold):
                v = np.array([cnt.get(c, 0) for c in g["cands"]], float)
                ys += [1] + [0] * (len(g["cands"]) - 1)
                ps += list(v)
                t1.append(float(v.argmax() == 0 and v[0] > 0))
        return float(roc_auc_score(ys, ps)), float(np.mean(t1))

    res = {}
    for scheme in ("reaction_disjoint", "enzyme_disjoint", "homology_disjoint"):
        report(f"\n  SPLIT: {scheme}")
        report(f"    {'arm':<26}{'AUC':>8}{'sd':>7}{'top-1':>8}")
        row = {}
        f_a, f_t = freq_arm(scheme, 0)
        row["freq"] = {"auc": f_a, "sd": 0.0, "top1": f_t}
        report(f"    {'freq (train counts)':<26}{f_a:>8.3f}{0.0:>7.3f}{f_t:>8.3f}")
        for nm, kw in (("esm_pair", {}), ("esm_enzyme_only", {"mode": "enzyme_only"}),
                       ("esm_shuffled_substrate", {"shuffle_sub": True}),
                       ("esm_untrained", {"train": False})):
            a, t = zip(*[run(scheme, kw.get("mode", "full"), s,
                             shuffle_sub=kw.get("shuffle_sub", False),
                             train=kw.get("train", True)) for s in SEEDS])
            row[nm] = {"auc": float(np.mean(a)), "sd": float(np.std(a)), "top1": float(np.mean(t))}
            mark = ""
            if nm == "esm_enzyme_only":
                mark = "   <- substrate zeroed: THE control"
            elif nm == "esm_untrained":
                mark = "   <- random init"
            report(f"    {nm:<26}{np.mean(a):>8.3f}{np.std(a):>7.3f}{np.mean(t):>8.3f}{mark}")
        res[scheme] = row

    report("\n  READING")
    rd, ed = res["reaction_disjoint"], res["homology_disjoint"]
    ez = res["enzyme_disjoint"]
    report(f"  reaction-disjoint {rd_ := res['reaction_disjoint']['esm_pair']['auc']:.3f}  ->  gene-disjoint "
           f"{ez['esm_pair']['auc']:.3f}  ->  HOMOLOGY-disjoint {ed['esm_pair']['auc']:.3f}")
    report("  The last column is the honest one: gene-disjoint still lets a ~93%-identical paralog sit in")
    report("  training, and recognising a paralog is not generalising to an unseen enzyme.")
    used_sub = ed["esm_pair"]["auc"] - ed["esm_enzyme_only"]["auc"]
    beat_freq = ed["esm_pair"]["auc"] - ed["freq"]["auc"]
    drop = rd["esm_pair"]["auc"] - ed["esm_pair"]["auc"]
    report(f"  reaction-disjoint ESM {rd['esm_pair']['auc']:.3f} -> enzyme-disjoint "
           f"{ed['esm_pair']['auc']:.3f}  (drop {drop:+.3f})")
    report(f"  enzyme-disjoint: ESM {ed['esm_pair']['auc']:.3f} | enzyme_only "
           f"{ed['esm_enzyme_only']['auc']:.3f} | freq {ed['freq']['auc']:.3f}")
    if used_sub > 0.03 and beat_freq > 0.03:
        report(f"  IT IS A PAIRING MODEL AND IT TRANSFERS. The substrate is worth {used_sub:+.3f} over zeroing")
        report(f"  it, and the model beats counting by {beat_freq:+.3f} on enzymes it never saw catalyse")
        report("  anything. This is the first result here that would move the orphan estimate.")
    elif used_sub <= 0.03:
        report(f"  THE SUBSTRATE IS BEING IGNORED. Zeroing it costs {used_sub:+.3f}, so whatever AUC this")
        report("  reaches is a catalyst-PRIOR -- 'these proteins look like enzymes' -- not enzyme-substrate")
        report("  matching. It cannot tell you WHICH reaction an enzyme runs, which is the entire question.")
    else:
        report(f"  USES THE SUBSTRATE BUT DOES NOT BEAT COUNTING ({beat_freq:+.3f} vs freq). A 250M-sequence")
        report("  representation lands where six hand-built signals and a set transformer already were.")
    if drop > 0.10:
        report(f"  AND NOTE THE SPLIT GAP: {drop:+.3f} lost when test enzymes are unseen. The flattering")
        report("  reaction-disjoint number is mostly memorised enzyme identity; quote the enzyme-disjoint one.")

    json.dump({"test": "nexus_catalyst_esm", "n_reactions": len(rx), "dim": dim, "model": MODEL,
               "results": res, "log": log}, open(OUT / "nexus_catalyst_esm.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_catalyst_esm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
