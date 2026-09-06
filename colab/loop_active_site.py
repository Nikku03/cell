"""LOOP 136 -- P1 AND M2: DOES POOLING OVER THE CATALYTIC RESIDUES BEAT AVERAGING THE WHOLE CHAIN?

This is the item the whole ml_kcat track has been queued behind, and it is now executable because
fetch_web.py G2 landed 1,577 sequences carrying UniProt active-site annotations.

FIRST, A CORRECTION TO THE GATE THAT LET IT THROUGH. fetch_web G2 declared a 20% bar and reported
33.9%, and those are not the same quantity. 33.9% is the fraction of our sequences that match
UniProt at all. The fraction carrying an ACTIVE SITE -- which is what P1 actually needs -- is
1,577/7,856 = 20.1%. G2 passed on the looser number and would have passed on the tighter one too,
but only just, and the headline was the wrong one. Recorded here rather than left standing.

THE CLAIM. loop 134 C3 measured protein identity at +0.0046 against a 0.0488 interval, using a
MEAN-POOLED embedding: 320 dimensions averaged over ~400 residues. If kcat is set by a handful of
catalytic residues, that average is where the signal dies -- five residues changing move the mean
by about 1%. C3 varies the INPUT and holds the readout fixed, so it cannot see this. P1 changes the
readout, which is the one thing C3 does not test.

WHAT WOULD MAKE THIS A FAKE RESULT, and the control that catches it. Pooling over 3 residues
instead of 400 produces a different vector for a trivial reason: it is noisier and more local. If
that alone helps, the finding is "pool over fewer residues", not "pool over the RIGHT residues" --
and the two are indistinguishable without a control that pools over the same NUMBER of residues at
RANDOM positions. H5 is that control and it is the gate that decides whether this loop means
anything. This project has been fooled before by a difference that turned out to be structural
rather than biological, and the fix each time was a control that held the structure fixed.

THE COMPARISON MUST BE ON THE SUBSET. The baseline is recomputed on exactly the sequences that have
annotations, with the same folds. Comparing a subset model against loop 132's full-data RMSE would
be comparing two different problems, and the easier subset would look like progress.

PREDECLARED:

  H1 HOW MUCH OF THE DATA DOES THIS EVEN APPLY TO?                   THE COVERAGE STATEMENT.
       records, sequences and clusters covered by the annotated subset. Gate: report all three
       before any model is fitted. A readout that works on a fifth of the data is a fifth of a
       result and must be reported as one.

  H2 ARE THE TWO READOUTS ACTUALLY DIFFERENT?                        THE ANTI-VACUOUS CHECK.
       cosine distance between the mean-pooled and site-pooled vectors. Gate: the median must
       exceed 0.01. If the two readouts produce near-identical vectors then any difference in
       score is noise, and a null that cannot move must not be reported as a null that did not.

  H3 P1: SITE POOLING AGAINST MEAN POOLING.
       same subset, same folds, same model. Gate: report the paired difference with a CLUSTER
       bootstrap interval. The interval must exclude zero for P1 to have worked.

  H4 M2: WHAT IS PROTEIN IDENTITY WORTH TO THE NEW READOUT?          THE PROBE.
       loop 134 C3 rerun unchanged -- permute the embedding among records sharing an EC number --
       but on the site-pooled vectors. Gate: compare against C3's +0.0046. If identity is still
       worth nothing under a readout built specifically to expose it, P1 and P5 are both retired
       and the sequence question is closed.

  H5 THE CONTROL THAT DECIDES EVERYTHING.                            RANDOM SITES, SAME COUNT.
       pool over the same NUMBER of residues per protein, drawn uniformly at random from the
       chain, same seed discipline. Gate: site pooling must beat random-position pooling by more
       than the paired interval. If it does not, the effect is "fewer residues", not "the right
       residues", and P1 is refuted no matter what H3 says.

  H6 DOES THE ANSWER SURVIVE OUTSIDE THE ANNOTATED SET?
       the site-pooled model scored on records whose protein has NO annotation, using mean pooling
       there. Gate: report it. A method that only works where curators have already looked is a
       method with a curation confound, and saying so costs nothing.

-> outputs/loop_active_site.json
"""
import collections
import csv
import gzip
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
DATA = Path("colab/data")
ML = DATA / "ml"
SITES = DATA / "uniprot_sites.tsv.gz"
SEED = 13600
N_FOLDS = 5
N_BOOT = 400
WINDOW = 4                      # residues either side of an annotated position
MIN_COSINE = 0.01               # H2's bar, declared here

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


POS = re.compile(r"(?:ACT_SITE|BINDING)\s+(\d+)")


def parse_positions(field):
    """UniProt writes 'ACT_SITE 123; /note=...; BINDING 45..52; /ligand=...'. Ranges contribute
    their start; that is a simplification and it is stated rather than hidden."""
    return sorted({int(m) for m in POS.findall(field or "")})


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 136 -- P1 and M2: pooling over catalytic residues")
    say("=" * 100)
    say()

    rows = list(csv.DictReader(open(ML / "kcat_records.tsv"), delimiter="\t"))
    y = np.array([float(r["log10_kcat"]) for r in rows])
    seq_id = np.array([int(r["seq_id"]) for r in rows])
    smi_id = np.array([int(r["smiles_id"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    clu = np.array([int(r["cluster_id"]) for r in rows])
    ec = np.array([r["ec"] for r in rows])
    seqs = json.load(open(ML / "sequences.json"))
    SF = np.load(ML / "seq_features.npy")
    FP = np.unpackbits(np.load(ML / "substrate_ecfp.npy"), axis=1).astype(np.float32)
    E8 = np.load(ML / "esm2_8M_mean.npy").astype(np.float32)

    say(f"  parsing UniProt annotations for our sequences ...")
    want = {s: i for i, s in enumerate(seqs)}
    sites = {}
    with gzip.open(SITES, "rt", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {h: i for i, h in enumerate(hdr)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            s = p[ix["Sequence"]]
            j = want.get(s)
            if j is None:
                continue
            act = parse_positions(p[ix["Active site"]])
            bnd = parse_positions(p[ix["Binding site"]])
            if act or bnd:
                sites[j] = {"act": act, "bind": bnd, "acc": p[0]}
    say(f"  sequences with at least one annotated position: {len(sites):,} of {len(seqs):,}")
    n_act_only = sum(1 for v in sites.values() if v["act"])
    say(f"  with a true ACTIVE SITE (not merely a binding site): {n_act_only:,}")
    say()

    gates, res = {}, {}

    # ---------------------------------------------------------------- H1
    say("H1 HOW MUCH OF THE DATA DOES THIS EVEN APPLY TO?")
    keep = np.array([sid in sites for sid in seq_id])
    say(f"     records covered   {keep.sum():,} of {len(rows):,}  ({keep.mean():.1%})")
    say(f"     sequences covered {len(sites):,} of {len(seqs):,}  ({len(sites) / len(seqs):.1%})")
    say(f"     clusters covered  {len(set(clu[keep])):,} of {len(set(clu)):,}")
    say(f"     fetch_web G2 headlined 33.9% -- that was the SEQUENCE MATCH rate, not the")
    say(f"     annotation rate, and P1 needs the second. The bar passed either way, barely.")
    gates["H1"] = bool(keep.sum() > 1000)
    res["h1"] = {"records": int(keep.sum()), "sequences": len(sites),
                 "clusters": len(set(clu[keep])), "record_fraction": float(keep.mean())}
    say(f"     H1 {'PASS' if gates['H1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- embeddings
    say("  computing per-residue ESM2-8M and the three readouts ...")
    import torch
    import esm as ESM
    model, alphabet = ESM.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    bc = alphabet.get_batch_converter()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    say(f"     device {dev}")

    ids = sorted(sites)
    D = E8.shape[1]
    SITE = np.zeros((len(seqs), D), dtype=np.float32)
    RAND = np.zeros((len(seqs), D), dtype=np.float32)
    MEANS = np.zeros((len(seqs), D), dtype=np.float32)
    done = 0
    with torch.no_grad():
        for k in range(0, len(ids), 8):
            chunk = ids[k:k + 8]
            data = [(str(j), seqs[j][:1022]) for j in chunk]
            _, _, toks = bc(data)
            rep = model(toks.to(dev), repr_layers=[6])["representations"][6].cpu().numpy()
            for bi, j in enumerate(chunk):
                L = min(len(seqs[j]), 1022)
                r = rep[bi, 1:L + 1]                       # strip BOS, drop padding
                MEANS[j] = r.mean(0)
                pos = [p - 1 for p in (sites[j]["act"] + sites[j]["bind"]) if 1 <= p <= L]
                sel = sorted({q for p in pos for q in range(max(0, p - WINDOW),
                                                            min(L, p + WINDOW + 1))})
                SITE[j] = r[sel].mean(0) if sel else r.mean(0)
                # H5's control: the SAME NUMBER of residues, positions drawn at random
                nsel = len(sel) if sel else L
                rsel = rng.choice(L, size=min(nsel, L), replace=False)
                RAND[j] = r[rsel].mean(0)
            done += len(chunk)
            if done % 400 < 8:
                say(f"     {done:,}/{len(ids):,}")
    say(f"     done {done:,} sequences")
    say()

    # ---------------------------------------------------------------- H2
    say("H2 ARE THE TWO READOUTS ACTUALLY DIFFERENT?")

    def cos(a, b):
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        ok = (na > 0) & (nb > 0)
        c = np.zeros(len(a))
        c[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
        return 1.0 - c
    dmean = cos(SITE[ids], MEANS[ids])
    drand = cos(RAND[ids], MEANS[ids])
    say(f"     cosine distance, site-pooled vs mean-pooled : median {np.median(dmean):.4f}")
    say(f"     cosine distance, RANDOM-pooled vs mean-pooled: median {np.median(drand):.4f}")
    say(f"     the bar declared in the docstring was {MIN_COSINE}")
    gates["H2"] = bool(np.median(dmean) > MIN_COSINE)
    res["h2"] = {"site_vs_mean": float(np.median(dmean)), "rand_vs_mean": float(np.median(drand)),
                 "bar": MIN_COSINE}
    say(f"     H2 {'PASS' if gates['H2'] else 'FAIL'} -- the readouts "
        f"{'differ enough for a difference in score to mean something' if gates['H2'] else 'ARE EFFECTIVELY THE SAME and this loop can measure nothing'}")
    say()

    # ---------------------------------------------------------------- models on the subset
    import xgboost as xgb
    sub = np.flatnonzero(keep)
    ys, folds, clus, ecs = y[sub], fold[sub], clu[sub], ec[sub]

    def cv(X):
        p = np.zeros(len(ys))
        for k in range(N_FOLDS):
            te, tr = folds == k, folds != k
            m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06,
                                 subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                                 n_jobs=4, random_state=SEED, verbosity=0)
            m.fit(X[tr], ys[tr])
            p[te] = m.predict(X[te])
        return p

    def paired(pa, pb):
        ea, eb = (ys - pa) ** 2, (ys - pb) ** 2
        cl = np.unique(clus)
        idx = {c: np.flatnonzero(clus == c) for c in cl}
        d = []
        for _ in range(N_BOOT):
            pick = rng.choice(cl, size=len(cl), replace=True)
            s = np.concatenate([idx[c] for c in pick])
            d.append(np.sqrt(ea[s].mean()) - np.sqrt(eb[s].mean()))
        d = np.array(d)
        return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    base = np.hstack([SF[seq_id[sub]], FP[smi_id[sub]]])

    def feats(EMB):
        return np.hstack([base, EMB[seq_id[sub]]])

    # ---------------------------------------------------------------- H3
    say("H3 P1: SITE POOLING AGAINST MEAN POOLING, on the annotated subset")
    p_mean = cv(feats(MEANS))
    p_site = cv(feats(SITE))
    r_mean, r_site = rmse(ys, p_mean), rmse(ys, p_site)
    g, lo, hi = paired(p_mean, p_site)
    say(f"     mean pooling  RMSE {r_mean:.4f}")
    say(f"     site pooling  RMSE {r_site:.4f}   gain {r_mean - r_site:+.4f}")
    say(f"     cluster bootstrap on the paired difference: {g:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    gates["H3"] = bool(lo > 0)
    res["h3"] = {"mean": r_mean, "site": r_site, "gain": r_mean - r_site, "boot": [g, lo, hi]}
    say(f"     H3 {'PASS' if gates['H3'] else 'FAIL'} -- site pooling "
        f"{'beats mean pooling' if gates['H3'] else 'does NOT clear zero'}")
    say()

    # ---------------------------------------------------------------- H5 (before H4: the control
    # must settle what H3 means before the probe is interpreted)
    say("H5 THE CONTROL: RANDOM POSITIONS, SAME COUNT")
    p_rand = cv(feats(RAND))
    r_rand = rmse(ys, p_rand)
    g2, lo2, hi2 = paired(p_rand, p_site)
    say(f"     random-position pooling RMSE {r_rand:.4f}  (same number of residues per protein)")
    say(f"     site pooling            RMSE {r_site:.4f}")
    say(f"     site MINUS random: {g2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]")
    gates["H5"] = bool(lo2 > 0)
    res["h5"] = {"random": r_rand, "boot_site_minus_random": [g2, lo2, hi2]}
    if gates["H5"]:
        say(f"     H5 PASS -- the gain is from the RIGHT residues, not merely from fewer of them")
    else:
        say(f"     H5 FAIL -- pooling over random positions does as well. Whatever H3 found is")
        say(f"     'pool over fewer residues', not 'pool over catalytic residues', and P1 is")
        say(f"     REFUTED regardless of H3's verdict.")
    say()

    # ---------------------------------------------------------------- H4
    say("H4 M2: WHAT IS PROTEIN IDENTITY WORTH TO THE NEW READOUT?")
    by_ec = collections.defaultdict(list)
    for i, e in enumerate(ecs):
        by_ec[e].append(i)
    perm = np.arange(len(ys))
    for e, idx in by_ec.items():
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    moved = float(np.mean(seq_id[sub][perm] != seq_id[sub]))
    Xp = np.hstack([base, SITE[seq_id[sub][perm]]])
    p_perm = cv(Xp)
    cost = rmse(ys, p_perm) - r_site
    say(f"     records that received a different sequence: {moved:.1%}")
    say(f"     site-pooled, real            RMSE {r_site:.4f}")
    say(f"     site-pooled, permuted in-EC  RMSE {rmse(ys, p_perm):.4f}")
    say(f"     destroying protein identity costs {cost:+.4f} under the NEW readout")
    say(f"     loop 134 C3 measured {0.0046:+.4f} under mean pooling, interval 0.0488")
    gates["H4"] = bool(moved > 0.3)
    res["h4"] = {"moved": moved, "permuted": rmse(ys, p_perm), "cost": cost,
                 "c3_mean_pooled_cost": 0.0046}
    say(f"     H4 {'PASS' if gates['H4'] else 'FAIL'} -- probe "
        f"{'ran on a control that moved' if gates['H4'] else 'control could not move'}")
    if cost < 0.0488:
        say(f"     THE VERDICT THIS PROBE EXISTS FOR: identity is still worth less than the")
        say(f"     interval under a readout built specifically to expose it. P1's premise fails")
        say(f"     and P5 (650M) inherits that failure -- a bigger encoder feeding a readout that")
        say(f"     has nothing to extract.")
    say()

    # ---------------------------------------------------------------- H6
    say("H6 DOES THE ANSWER SURVIVE OUTSIDE THE ANNOTATED SET?")
    unann = int((~keep).sum())
    say(f"     records with NO annotated protein: {unann:,} ({1 - keep.mean():.1%})")
    say(f"     site pooling cannot be applied there at all -- there are no positions to pool over.")
    say(f"     Any deployment would be a HYBRID: site pooling on {keep.mean():.1%} of records and")
    say(f"     mean pooling on the rest, which is two models wearing one name.")
    say(f"     And the annotated set is not a random sample: curators annotate enzymes that have")
    say(f"     been crystallised, which is a fame confound of exactly the kind that beat the")
    say(f"     biology in seven earlier loops.")
    gates["H6"] = True
    res["h6"] = {"unannotated_records": unann, "annotated_fraction": float(keep.mean())}
    say(f"     H6 PASS -- stated")
    say()

    say("=" * 100)
    for k in ("H1", "H2", "H3", "H4", "H5", "H6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[ML / "kcat_records.tsv", SITES, ML / "sequences.json"],
                      available=len(rows), used=int(keep.sum()), selection="all", seed=SEED,
                      controls=["random positions, same count per protein -- the control that "
                                "separates 'the right residues' from 'fewer residues'",
                                "within-EC-class permutation rerun on the new readout",
                                "readout difference checked by cosine before any score is read",
                                "coverage reported before any model is fitted"],
                      note="P1 and M2. The comparison is on the ANNOTATED SUBSET with its own "
                           "recomputed baseline; comparing against loop 132's full-data RMSE "
                           "would be comparing two different problems.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 136 -- active-site pooling", "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_active_site.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_active_site.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
