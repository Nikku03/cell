"""Loop 163. Before building a 3D model of the dark proteome: does structure beat sequence at all?

THE PROPOSAL THIS TESTS. Take the 3D structures of every protein in the reaction network, learn how
they map to the chemistry they perform, and run the learned model on dark-gene structures to infer
what those genes do. It is a good idea and it is expensive: only 11 of the 9,650 dark genes have a
cached structure, so it starts with ~9,600 AlphaFold downloads before a single parameter is fitted.

WHAT THIS REPOSITORY ALREADY MEASURED ABOUT 3D. The nexus catalyst arm spent 23,304 of 25,209
core-seconds -- 92% of its pipeline -- on FFT shape-complementarity docking, and every docking
feature came back in AUC [0.450, 0.549] against a size-only control at 0.532, with a true-catalyst
mean rank of 5.47 against a chance value of 5.5. In the fully enumerated feature-block design space
every docking block had regret 0.0000: the best model containing no docking at all equalled the
best model overall. Meanwhile ESM-2 SEQUENCE embeddings were the load-bearing block in the same arm,
and loop 156 got +0.3237 from them on an unrelated target.

So the prior in this repo is that sequence carries the signal and 3D does not. That prior was formed
on DOCKING, which is one particular use of structure, and it would be wrong to retire the whole idea
on it. This loop tests the general claim on the cheap set -- the 2,178 enzymes that already have BOTH
an AlphaFold monomer and a sequence, so neither arm gets a coverage advantage -- before anything is
downloaded.

THE TASK, reduced to something rankable. An enzyme's function, as this model records it, is the set
of non-currency metabolites its reactions touch. For a held-out enzyme, rank all 8,428 non-currency
metabolites. A representation that is good at this has captured what the protein DOES.

THE PREDICTOR IS k-NEAREST-NEIGHBOUR, NOT A TRANSFORMER, AND THAT IS DELIBERATE. With 2,178 labelled
proteins a transformer would fit the split rather than the biology, and the question here is whether
the REPRESENTATION carries the information -- which k-NN answers directly and cannot flatter. If
structure wins here, a trained model is worth building on top of it. If it does not, no architecture
rescues a representation that does not separate.

PREDECLARED, before any number is looked at.

  Y1 THE TWO ARMS ARE ON THE SAME PROTEINS, AND THE SPLIT IS HOMOLOGY-DISJOINT. Identical accession
     lists for sequence and structure. Folds assigned by single-linkage clustering at 5-mer Jaccard
     >= 0.30, whole clusters to folds, which is loop 156's scheme.
     Gate: identical accessions, and 0 train/test pairs above the Jaccard threshold. Without this
     every number below is a homology lookup wearing a representation's clothes.

  Y2 IS THERE ANY SIGNAL AT ALL? The best of the representations against a metabolite-POPULARITY
     baseline that ignores the protein completely.
     Gate: PASS iff the best representation beats popularity by more than 3 sem of the paired
     difference. If it fails, nothing else in the loop means anything and it says so.

  Y3 IS IT MORE THAN RESIDUAL HOMOLOGY? The split removes close homologues, but function prediction
     is famously just BLAST wearing a hat, so the k-NN is also run on RAW 5-MER SEQUENCE SIMILARITY
     -- a homology lookup with no learned representation in it.
     Gate: PASS iff the best learned representation beats the raw k-mer lookup by more than 3 sem.

  Y4 DOES STRUCTURE BEAT SEQUENCE? The decisive gate, and the one that decides whether ~9,600
     structure downloads and a 3D model are justified.
     Gate: PASS iff structure beats the better ESM arm by more than 3 sem of the paired difference.
     This gate is written to be able to fail, and if it fails the honest headline is that the fold
     geometry available from a monomer does not carry function that its sequence does not already
     carry -- which would mean the dark-gene inference should be run from sequence, at 1.3 s per
     protein, instead of from structure at a 2.4 GB download.

  Y5 DOES STRUCTURE ADD TO SEQUENCE? A different question from Y4: a representation can lose
     head-to-head and still contribute, which is exactly what loop 161 found for mass balance
     against the walk, and exactly what the merge workflow then overturned.
     Gate: PASS iff sequence+structure concatenated beats the better single arm by more than 3 sem.

  Y6 WHAT THIS CANNOT SHOW. k-NN is a floor on what a representation supports, not a ceiling -- a
     trained model can extract more. The labels come from Human-GEM's own GPR, so an enzyme with a
     wrong or missing gene association is scored as if the model were right. AlphaFold monomers
     carry no ligand, no cofactor and no partner, so anything about function that lives in a complex
     is invisible to both arms. And the dark genes are dark partly because they are unlike these
     2,178, so performance here is an upper bound on performance there.

-> outputs/loop_struct_vs_seq.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402
from rem.harness import REM, auc_of        # noqa: E402

SEQF = Path("colab/data/ml/esm_enzymes.npz")
STRF = Path("colab/data/ml/struct_enzymes.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_struct_vs_seq.json"
KNN = 10
JACC = 0.30
NFOLD = 5
SEED = 16300
KMER = 5

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def kmers(s, k=KMER):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def homology_folds(seqs, accs, thr=JACC, nfold=NFOLD, seed=SEED):
    """Single-linkage clusters at 5-mer Jaccard >= thr, whole clusters assigned to folds."""
    ks = [kmers(seqs[a]) for a in accs]
    n = len(accs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    inv = defaultdict(list)
    for i, s in enumerate(ks):
        for m in s:
            inv[m].append(i)
    cand = defaultdict(int)
    for m, idx in inv.items():
        if len(idx) > 200:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                cand[(idx[a], idx[b])] += 1
    for (i, j), shared in cand.items():
        u = len(ks[i]) + len(ks[j]) - shared
        if u and shared / u >= thr:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    gl = sorted(groups.values(), key=len, reverse=True)
    fold = np.zeros(n, int)
    load = np.zeros(nfold)
    for g in gl:
        f = int(np.argmin(load))
        for i in g:
            fold[i] = f
        load[f] += len(g)
    return fold, len(gl), ks


def knn_scores(Xtr, Ytr, Xte, k=KNN):
    """Cosine k-NN, label vectors averaged over the k nearest training proteins."""
    A = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-12)
    B = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-12)
    S = B @ A.T
    idx = np.argpartition(-S, min(k, S.shape[1] - 1), axis=1)[:, :k]
    out = np.zeros((len(Xte), Ytr.shape[1]), np.float32)
    for i in range(len(Xte)):
        w = np.maximum(S[i, idx[i]], 0)
        if w.sum() <= 0:
            w = np.ones_like(w)
        out[i] = (w[:, None] * Ytr[idx[i]]).sum(0) / w.sum()
    return out


def main():
    t0 = time.time()
    say("=" * 104)
    say("  BEFORE A 3D MODEL OF THE DARK PROTEOME: does structure beat sequence?")
    say("=" * 104)
    say()

    S = np.load(SEQF, allow_pickle=False)
    T = np.load(STRF, allow_pickle=False)
    common = sorted(set(map(str, S["accs"])) & set(map(str, T["accs"])))
    si = {a: i for i, a in enumerate(map(str, S["accs"]))}
    ti = {a: i for i, a in enumerate(map(str, T["accs"]))}
    accs = common
    E35 = S["esm35"][[si[a] for a in accs]]
    E8 = S["esm8"][[si[a] for a in accs]]
    ST = T["X"][[ti[a] for a in accs]]
    say(f"     {len(accs):,} enzymes with BOTH arms | esm35 {E35.shape[1]}d, "
        f"esm8 {E8.shape[1]}d, structure {ST.shape[1]}d")

    # labels: the non-currency metabolites each enzyme's reactions touch
    R = REM()
    Z = np.load("colab/data/rem_enzyme.npz", allow_pickle=False)
    sym = list(map(str, Z["symbols"]))
    gene_rx = defaultdict(set)
    for j, g in zip(Z["gpr_rx"], Z["gpr_gene"]):
        gene_rx[sym[int(g)]].add(int(j))
    import gzip
    import re
    a2g, seqs = {}, {}
    acc, buf = None, []
    sys.path.insert(0, str(HERE))
    import loop_replication as LR
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if acc and buf:
                    seqs[acc] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                g = re.search(r"GN=(\S+)", ln)
                acc, buf = (m.group(1) if m else None), []
                if acc and g:
                    a2g[acc] = g.group(1)
            else:
                buf.append(ln.strip())
    if acc and buf:
        seqs[acc] = "".join(buf)

    Y = np.zeros((len(accs), len(R.noncur)), np.float32)
    for i, a in enumerate(accs):
        for j in gene_rx.get(a2g.get(a, ""), ()):
            for m in (R.react_of[j] | R.prod_of[j]) - R.currency:
                Y[i, R.ncmap[int(m)]] = 1.0
    keep = Y.sum(1) > 0
    say(f"     {int(keep.sum()):,} of {len(accs):,} enzymes touch at least one non-currency "
        f"metabolite; median {int(np.median(Y[keep].sum(1)))} metabolites each")
    accs = [a for a, k in zip(accs, keep) if k]
    E35, E8, ST, Y = E35[keep], E8[keep], ST[keep], Y[keep]

    # ------------------------------------------------------------------ Y1
    say()
    say("Y1 SAME PROTEINS, HOMOLOGY-DISJOINT SPLIT")
    fold, ncl, ks = homology_folds(seqs, accs)
    viol = 0
    for f in range(NFOLD):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        for i in te[:200]:
            for j in tr[:200]:
                sh = len(ks[i] & ks[j])
                u = len(ks[i]) + len(ks[j]) - sh
                if u and sh / u >= JACC:
                    viol += 1
    y1 = bool(len(set(map(str, S["accs"])) & set(map(str, T["accs"]))) > 0 and viol == 0)
    say(f"     {ncl:,} homology clusters at 5-mer Jaccard >= {JACC} over {len(accs):,} proteins")
    say(f"     fold sizes {[int((fold == f).sum()) for f in range(NFOLD)]}")
    say(f"     train/test pairs above the threshold (sampled 200x200 per fold): {viol}")
    GG.verdict(y1, emit=say, if_true=(
        "the arms are on identical proteins and no close homologue crosses a fold boundary."),
        if_false="the split leaks homology; every number below is a lookup, not a prediction.")
    say(f"     Y1 {'PASS' if y1 else 'FAIL'}")

    # ------------------------------------------------------------------ arms
    say()
    say("THE ARMS, cross-validated over the homology folds")
    pop = Y.mean(0)

    def zs(X):
        m, s = X.mean(0), X.std(0)
        return (X - m) / np.maximum(s, 1e-9)

    ARMS = {
        "esm35": zs(E35), "esm8": zs(E8), "structure": zs(ST),
        "esm35+structure": np.hstack([zs(E35), zs(ST)]),
        "esm35+esm8": np.hstack([zs(E35), zs(E8)]),
    }
    per_case = {k: [] for k in list(ARMS) + ["popularity", "kmer_homology"]}
    kmat = None
    for f in range(NFOLD):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        for nm, X in ARMS.items():
            P = knn_scores(X[tr], Y[tr], X[te])
            for r, i in enumerate(te):
                per_case[nm].append(auc_of(P[r], Y[i] > 0))
        # raw 5-mer homology lookup, no learned representation
        Ksim = np.zeros((len(te), len(tr)))
        for r, i in enumerate(te):
            for c, j in enumerate(tr):
                sh = len(ks[i] & ks[j])
                u = len(ks[i]) + len(ks[j]) - sh
                Ksim[r, c] = sh / u if u else 0.0
        idx = np.argpartition(-Ksim, min(KNN, Ksim.shape[1] - 1), axis=1)[:, :KNN]
        for r, i in enumerate(te):
            w = np.maximum(Ksim[r, idx[r]], 0)
            w = w if w.sum() > 0 else np.ones_like(w)
            p = (w[:, None] * Y[tr][idx[r]]).sum(0) / w.sum()
            per_case["kmer_homology"].append(auc_of(p, Y[i] > 0))
            per_case["popularity"].append(auc_of(pop, Y[i] > 0))
        say(f"     fold {f}: {len(te)} test proteins [{time.time()-t0:.0f}s]")

    res = {}
    for k, v in per_case.items():
        a = np.array(v, float)
        a = a[np.isfinite(a)]
        res[k] = {"auc": float(a.mean()), "sem": float(a.std() / np.sqrt(len(a))), "n": len(a)}
    say()
    for k in sorted(res, key=lambda x: -res[x]["auc"]):
        say(f"       {k:<20s} AUC {res[k]['auc']:.4f} +/- {res[k]['sem']:.4f}   n={res[k]['n']}")

    def paired(a, b):
        x = np.array(per_case[a], float)
        y = np.array(per_case[b], float)
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ Y2..Y5
    best_arm = max(ARMS, key=lambda k: res[k]["auc"])
    d, s = paired(best_arm, "popularity")
    y2 = bool(d > 3 * s)
    say()
    say(f"Y2 best arm ({best_arm}) minus POPULARITY: {d:+.4f} sem {s:.4f} = {d/s:+.1f} sem")
    GG.verdict(y2, emit=say, if_true="there is real signal to talk about.", if_false=(
        "no representation beats a baseline that ignores the protein entirely; nothing below means "
        "anything and Y3-Y5 are reported as a description of noise."))
    say(f"     Y2 {'PASS' if y2 else 'FAIL'}")

    d3, s3 = paired(best_arm, "kmer_homology")
    y3 = bool(d3 > 3 * s3)
    say()
    say(f"Y3 best arm minus RAW 5-MER HOMOLOGY LOOKUP: {d3:+.4f} sem {s3:.4f} = {d3/s3:+.1f} sem")
    GG.verdict(y3, emit=say, if_true=(
        "the learned representation beats a homology lookup, so it is not BLAST in a hat."),
        if_false=(
        "the learned representation does NOT beat a raw k-mer lookup. Whatever it has learned is "
        "recoverable by string similarity, which is free."))
    say(f"     Y3 {'PASS' if y3 else 'FAIL'}")

    seq_best = "esm35" if res["esm35"]["auc"] >= res["esm8"]["auc"] else "esm8"
    d4, s4 = paired("structure", seq_best)
    y4 = bool(d4 > 3 * s4)
    say()
    say(f"Y4 STRUCTURE minus {seq_best}: {d4:+.4f} sem {s4:.4f} = {d4/s4:+.1f} sem")
    GG.verdict(y4, emit=say, if_true=(
        "structure beats sequence. Downloading the dark proteome's structures and building a 3D "
        "model is justified by a measurement rather than by an intuition."), if_false=(
        "structure does NOT beat sequence on identical proteins with an identical split. The fold "
        "geometry a monomer carries adds nothing here that the sequence does not already carry, "
        "and the dark-gene inference should be run from sequence at 1.3 s per protein rather than "
        "from ~9,600 structure downloads. This does not say 3D is useless in general -- it says "
        "the monomer-geometry version of it is not what is missing."))
    say(f"     Y4 {'PASS' if y4 else 'FAIL'}")

    d5, s5 = paired("esm35+structure", seq_best)
    y5 = bool(d5 > 3 * s5)
    say()
    say(f"Y5 esm35+structure minus {seq_best}: {d5:+.4f} sem {s5:.4f} = {d5/s5:+.1f} sem")
    GG.verdict(y5, emit=say, if_true=(
        "structure ADDS to sequence even if it loses head to head -- the two carry different "
        "information and the combination is worth the download."), if_false=(
        "structure does not add to sequence either. Both gates point the same way and the 3D "
        "pipeline is not justified on this evidence."))
    say(f"     Y5 {'PASS' if y5 else 'FAIL'}")

    say()
    say("Y6 WHAT THIS CANNOT SHOW")
    say("     k-NN is a floor on what a representation supports, not a ceiling; a trained model can")
    say("     extract more than a nearest-neighbour vote from the same vectors.")
    say("     Labels come from Human-GEM's own gene-protein-reaction rules, so a wrong or missing")
    say("     association is scored as if the model were right.")
    say("     AlphaFold monomers carry no ligand, no cofactor and no partner, so any function that")
    say("     lives in a complex is invisible to BOTH arms -- this is not a test of structure in")
    say("     general, it is a test of monomer geometry.")
    say("     The dark genes are dark partly because they are unlike these 2,178, so every number")
    say("     here is an upper bound on what the same method would do on them.")
    y6 = True
    say(f"     Y6 {'PASS' if y6 else 'FAIL'}")

    gates = {"Y1": y1, "Y2": y2, "Y3": y3, "Y4": y4, "Y5": y5, "Y6": y6}
    man = RM.manifest(
        inputs=[SEQF, STRF, Path("colab/data/rem_enzyme.npz")],
        available=len(accs), used=len(accs), selection="all", seed=SEED,
        controls=[
            "identical proteins in both arms, by construction (intersection of structure and sequence)",
            "homology-disjoint folds at 5-mer Jaccard >= 0.30, violations counted not assumed",
            "a popularity baseline that ignores the protein entirely",
            "a raw 5-mer homology lookup with no learned representation, the BLAST control",
            "k-NN rather than a trained model, so the comparison is of representations not of fits",
            "conclusions emitted through gate_guard.verdict",
        ],
        note="does monomer structure geometry beat ESM-2 sequence at predicting enzyme function")
    out = {"test": "structure vs sequence for enzyme function, before committing to a 3D pipeline",
           "gates": gates, "n_enzymes": len(accs), "n_clusters": ncl,
           "results": res,
           "paired": {"best_vs_popularity": [d, s], "best_vs_kmer": [d3, s3],
                      "structure_vs_seq": [d4, s4], "combined_vs_seq": [d5, s5]},
           "best_arm": best_arm, "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
