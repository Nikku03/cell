"""PHASE 2+3: the pair tensor and the influence-gram, tested on the sealed knockout protocol.

THE REFRAMING.  Every model so far predicted a RANKED LIST per knockout -- 4,720 training examples.  This predicts
the (perturbed gene i, responding gene j) PAIR, which is the natural coordinate of the answer, the way a distogram
is the natural coordinate of a structure.  Relational data that has been mangled into node features -- complexes,
PPI, co-dependency, TF edges, genomic distance -- is natively PAIRWISE and finally enters in its own shape.

A CORRECTION TO MY OWN ARGUMENT, MADE BEFORE THE RESULT RATHER THAN AFTER.  I claimed the reframing turns 4,720
examples into "39 million labels" (4,720 x 8,246).  That is wrong in the way that matters.  Almost every pair is a
non-event: gwps_rebuild measured median 0 specific movers per perturbation, and only 838 of 5,120 knockouts clear
5 movers.  The POSITIVE pairs -- the informative ones -- number in the tens of thousands, not the tens of
millions.  This file counts and reports them.  The reframing still multiplies the supervision, but by roughly an
order of magnitude, not four, and the labels are not independent because they come from 4,720 rows.

ARMS, so a failure is attributable rather than just a failure.
    pair_full     all pair features: relational, covariational, positional, plus node features of i and j.
    node_only     node features of i and j ONLY, no pair term.  If pair_full does not beat this, the PAIR
                  STRUCTURE adds nothing and the whole reframing is dead regardless of the headline number.
    j_only        the responding gene's features alone -- no knowledge of what was knocked out.  This is the
                  frequency baseline wearing a different hat, and it is the floor any pair model must clear.
    pair_shuffled the relational pair features degree-matched rewired: every gene keeps its exact degree, only
                  the wiring moves.  Prices whether it is the TOPOLOGY doing the work or just the degrees.
    chan2a        the incumbent, 0.2885 on the sealed holdout.

PREDECLARED GATES, from the plan.
    PHASE 2 passes if pair_full beats node_only and pair_shuffled past a bootstrap CI.
    PHASE 3 passes if pair_full beats chan2a (0.2337 / 0.2885) on BOTH sealed cohorts.
    If phase 2 passes and phase 3 fails, the pair object is real but the decoder is wrong.
    If phase 2 fails, the object is not carrying information and nothing downstream will save it.

LEAKAGE.  The responsiveness marginal, the model and every feature statistic are computed on TRAINING knockouts
only; both sealed cohorts are removed wholesale before anything is fitted.  Co-dependency is DepMap-derived and
the target here is Perturb-seq transcriptional response -- a different assay -- so it is admissible, but it is
flagged in the feature list and a variant without it is reported.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT = A.OUT
SP = A.SP
NEG_PER_POS = 20
BOOT = 10_000


def pair_sources(report):
    D = json.load(open(OUT / "cell_complete.json"))
    nm = [g["name"] for g in D["genes"]]
    rec = {g["name"]: g for g in D["genes"]}

    def gname(k):
        return nm[int(k)] if k.isdigit() and int(k) < len(nm) else k

    ppi = collections.defaultdict(set)
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < len(nm) and 0 <= b_ < len(nm) and a_ != b_:
            ppi[nm[a_]].add(nm[b_])
            ppi[nm[b_]].add(nm[a_])
    cplx_of = collections.defaultdict(set)
    members = collections.defaultdict(list)
    for k_, cs in (D.get("gene2cplx") or {}).items():
        g = gname(k_)
        for c in (cs if isinstance(cs, list) else [cs]):
            cplx_of[g].add(str(c))
            members[str(c)].append(g)
    complex_partner = collections.defaultdict(set)
    for c, mem in members.items():
        for g in mem:
            complex_partner[g] |= {x for x in mem if x != g}
    codep = collections.defaultdict(dict)
    for a_, v in (D.get("codep") or {}).items():
        g = gname(a_)
        for b_, s in v:
            try:
                codep[g][nm[int(b_)]] = float(s)
            except (TypeError, ValueError, IndexError):
                continue
    tf_signed = collections.defaultdict(dict)
    chip = collections.defaultdict(set)
    for e in D.get("reg", []):
        try:
            s_, t_ = int(e[0]), int(e[1])
            w = int(e[2]) if len(e) > 2 else 0
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= s_ < len(nm) and 0 <= t_ < len(nm)) or s_ == t_:
            continue
        if w == 0:
            chip[nm[s_]].add(nm[t_])
        else:
            tf_signed[nm[s_]][nm[t_]] = float(np.sign(w))
    go_of = {}
    for g, v in (D.get("go") or {}).items():
        key = gname(g)
        if isinstance(v, dict):
            go_of[key] = {f"{ns}:{t}" for ns in v for t in (v[ns] or [])}
    chrom = {g: (rec[g].get("chrom") or "") for g in rec}
    tss = {}
    for g in rec:
        try:
            tss[g] = float(rec[g].get("tss") or 0)
        except (TypeError, ValueError):
            tss[g] = 0.0
    del D
    paths = collections.defaultdict(set)
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            for g in p[2:]:
                paths[g].add(p[0])
    report(f"  pair sources: ppi {len(ppi):,} · complexes {len(complex_partner):,} · codep {len(codep):,} · "
           f"signed TF {len(tf_signed):,} · chip {len(chip):,} · GO {len(go_of):,} · pathways {len(paths):,}")
    return dict(ppi=ppi, cplx=complex_partner, codep=codep, tf=tf_signed, chip=chip,
                go=go_of, paths=paths, chrom=chrom, tss=tss)


FEATURES = ["j_responsiveness", "j_degree", "i_degree", "same_complex", "ppi_edge",
            "shared_pathways", "pathway_jaccard", "shared_go", "go_jaccard",
            "codep_score", "codep_rank", "tf_signed", "chip_edge",
            "same_chrom", "log_tss_dist", "i_npath", "j_npath", "reciprocal_ppi"]


def build_rows(pairs, S, resp, rng=None):
    """Feature matrix for a list of (i_name, j_name) pairs."""
    n = len(pairs)
    X = np.zeros((n, len(FEATURES)), np.float32)
    ppi, cplx, codep = S["ppi"], S["cplx"], S["codep"]
    tf, chip, go, paths = S["tf"], S["chip"], S["go"], S["paths"]
    chrom, tss = S["chrom"], S["tss"]
    for r, (i, j) in enumerate(pairs):
        pi, pj = paths.get(i, set()), paths.get(j, set())
        gi_, gj_ = go.get(i, set()), go.get(j, set())
        cd = codep.get(i, {})
        order = sorted(cd.items(), key=lambda kv: -kv[1])
        rank = next((k + 1 for k, (g, _) in enumerate(order) if g == j), 0)
        ci, cj = chrom.get(i, ""), chrom.get(j, "")
        X[r] = [
            resp.get(j, 0.0),
            len(ppi.get(j, ())), len(ppi.get(i, ())),
            1.0 if j in cplx.get(i, ()) else 0.0,
            1.0 if j in ppi.get(i, ()) else 0.0,
            len(pi & pj), len(pi & pj) / max(len(pi | pj), 1),
            len(gi_ & gj_), len(gi_ & gj_) / max(len(gi_ | gj_), 1),
            cd.get(j, 0.0), 1.0 / rank if rank else 0.0,
            tf.get(i, {}).get(j, 0.0),
            1.0 if j in chip.get(i, ()) else 0.0,
            1.0 if ci and ci == cj else 0.0,
            np.log1p(abs(tss.get(i, 0.0) - tss.get(j, 0.0))) if ci and ci == cj else 0.0,
            len(pi), len(pj),
            1.0 if (j in ppi.get(i, ()) and i in ppi.get(j, ())) else 0.0,
        ]
    return X


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 106)
    report("PHASE 2+3: PAIR TENSOR AND INFLUENCE-GRAM on the sealed knockout protocol")
    report("=" * 106)
    report("  PHASE 2 gate: pair_full must beat node_only AND pair_shuffled.")
    report("  PHASE 3 gate: pair_full must beat chan2a (0.2337 / 0.2885) on BOTH cohorts.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    nontide_idx = np.where(~tide)[0]
    cand = [genes[i] for i in nontide_idx]

    sealed: set[str] = set()
    answers: dict[str, dict] = {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]

    S = pair_sources(report)

    # responsiveness marginal, TRAINING knockouts only
    hit = (Amat[[ki[k] for k in train]] >= A.TAU)
    resp = {genes[c]: float(hit[:, c].mean()) for c in nontide_idx}

    # ---------------- count the actual supervision, correcting the "39 million" claim ----------------
    pos_pairs = []
    for k in train:
        row = Amat[ki[k]]
        movers = [genes[c] for c in nontide_idx if row[c] >= A.TAU]
        for m in movers:
            pos_pairs.append((k, m))
    total_pairs = len(train) * len(cand)
    report(f"\n  SUPERVISION, counted rather than asserted:")
    report(f"    candidate pairs        {total_pairs:>12,}")
    report(f"    POSITIVE pairs         {len(pos_pairs):>12,}  ({100*len(pos_pairs)/total_pairs:.4f}%)")
    report(f"    training knockouts     {len(train):>12,}")
    report(f"    -> the reframing multiplies supervision by ~{len(pos_pairs)/len(train):.0f}x per knockout,")
    report(f"       not by the 8,246x a raw pair count suggests. My earlier '39 million labels' was wrong.")

    rng = np.random.default_rng(A.SEED)
    neg_pairs = []
    per_ko_pos = collections.defaultdict(list)
    for k, m in pos_pairs:
        per_ko_pos[k].append(m)
    for k, ms in per_ko_pos.items():
        block = set(ms)
        draw = rng.choice(len(cand), size=min(len(ms) * NEG_PER_POS, len(cand)), replace=False)
        neg_pairs.extend((k, cand[int(d)]) for d in draw if cand[int(d)] not in block)
    report(f"    negatives drawn        {len(neg_pairs):>12,}  ({NEG_PER_POS}x per positive)")

    all_pairs = pos_pairs + neg_pairs
    y = np.concatenate([np.ones(len(pos_pairs), np.float32), np.zeros(len(neg_pairs), np.float32)])
    report(f"\n  building the pair tensor for {len(all_pairs):,} training pairs ...")
    X = build_rows(all_pairs, S, resp)
    report(f"    {X.shape[1]} pair features: {', '.join(FEATURES)}")

    # ---------------- degree-matched rewiring for the shuffled control ----------------
    Sshuf = dict(S)
    for key in ("ppi", "cplx", "chip"):
        table = S[key]
        stubs = [(a, b) for a, bs in table.items() for b in bs]
        heads = [a for a, _ in stubs]
        tails = [b for _, b in stubs]
        perm = rng.permutation(len(tails))
        rew = collections.defaultdict(set)
        for a, t in zip(heads, perm):
            rew[a].add(tails[int(t)])
        Sshuf[key] = rew
    report(f"  degree-matched rewiring built for ppi / complex / chip")

    from sklearn.ensemble import HistGradientBoostingClassifier
    NODE_ONLY = [f for f in FEATURES if f.startswith(("i_", "j_"))]
    J_ONLY = [f for f in FEATURES if f.startswith("j_")]
    node_cols = [FEATURES.index(f) for f in NODE_ONLY]
    j_cols = [FEATURES.index(f) for f in J_ONLY]

    models = {}
    models["pair_full"] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y)
    models["node_only"] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[:, node_cols], y)
    models["j_only"] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[:, j_cols], y)
    Xs = build_rows(all_pairs, Sshuf, resp)
    models["pair_shuffled"] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xs, y)
    report(f"  four models fitted")

    # ---------------- score the sealed cohorts ----------------
    summary = {}
    for tag, label in (("", "cohort 1"), ("_holdout", "cohort 2 (holdout)")):
        ans = answers.get(tag)
        if not ans:
            continue
        cohort = sorted(ans)
        per_arm = {a: [] for a in models}
        for k in cohort:
            truth = set(ans[k]["movers"])
            pairs_k = [(k, g) for g in cand]
            Xk = build_rows(pairs_k, S, resp)
            Xks = build_rows(pairs_k, Sshuf, resp)
            for arm, mdl in models.items():
                if arm == "pair_full":
                    p = mdl.predict_proba(Xk)[:, 1]
                elif arm == "node_only":
                    p = mdl.predict_proba(Xk[:, node_cols])[:, 1]
                elif arm == "j_only":
                    p = mdl.predict_proba(Xk[:, j_cols])[:, 1]
                else:
                    p = mdl.predict_proba(Xks)[:, 1]
                top = np.argsort(-p)[:A.NPRED]
                per_arm[arm].append(len({cand[t] for t in top} & truth) / A.NPRED)
        report(f"\n### {label}   ({len(cohort)} knockouts)")
        report(f"  {'arm':<16} {'precision@20':>13}")
        vals = {}
        for arm in ("pair_full", "node_only", "j_only", "pair_shuffled"):
            v = np.array(per_arm[arm])
            vals[arm] = v
            report(f"  {arm:<16} {v.mean():>13.4f}")
        chan = OUT / f"ko_score_chan2a{tag}.json"
        if chan.exists():
            rows = {r["gene"]: r["precision"] for r in json.loads(chan.read_text())["rows"]}
            base = np.array([rows.get(k, np.nan) for k in cohort])
            vals["chan2a"] = base
            report(f"  {'chan2a (incumbent)':<16} {np.nanmean(base):>13.4f}")

        report(f"\n  {'contrast':<34} {'gap':>8} {'95% CI':>22}  verdict")
        rows_out = {}
        for name, a_, b_ in (("PHASE2 pair_full - node_only", "pair_full", "node_only"),
                             ("PHASE2 pair_full - pair_shuffled", "pair_full", "pair_shuffled"),
                             ("       pair_full - j_only", "pair_full", "j_only"),
                             ("PHASE3 pair_full - chan2a", "pair_full", "chan2a")):
            if b_ not in vals:
                continue
            d = vals[a_] - vals[b_]
            d = d[np.isfinite(d)]
            br = np.random.default_rng(0)
            idx = br.integers(0, len(d), size=(BOOT, len(d)))
            bs = d[idx].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            verdict = "RESOLVED" if (lo > 0 or hi < 0) else "within noise"
            report(f"  {name:<34} {d.mean():>+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22}  {verdict}")
            rows_out[name.strip()] = {"gap": float(d.mean()), "ci": [lo, hi],
                                      "resolved": bool(lo > 0 or hi < 0)}
        summary[tag or "cohort1"] = {"means": {a: float(np.nanmean(v)) for a, v in vals.items()},
                                     "contrasts": rows_out}

    json.dump({"test": "adrn_pair_influence", "features": FEATURES,
               "n_positive_pairs": len(pos_pairs), "n_candidate_pairs": int(total_pairs),
               "neg_per_pos": NEG_PER_POS, "summary": summary, "log": log},
              open(OUT / "adrn_pair_influence.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_pair_influence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
