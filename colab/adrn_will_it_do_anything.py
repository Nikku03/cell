"""THE 6x DATASET WE THREW AWAY: will this knockout do ANYTHING at all?

THE OBSERVATION.  Every score in this project is computed on 838 examples:

    11,258 rows in the genome-wide screen
     5,857 dropped -- "no detectable phenotype" (energy test p >= 0.05)
       170 dropped -- too few cells
     5,120 kept in nlz_K562_gwps.npz
       838 of those are SCORABLE (>= 5 specific movers), with a label that is 37% noise

Every number in this project comes from the last line. The other 4,282 kept rows are not missing data -- they are
LABELS FOR A DIFFERENT QUESTION: *does knocking this gene out do anything measurable in K562?*

SCOPE CORRECTION, made before running rather than after. I first described this as a 12x dataset (9,900 rows).
That is wrong: the 5,857 energy-test failures were dropped during the rebuild and are NOT in the npz, which holds
only the 5,120 survivors. So this is **6x**, not 12x, and its negatives are genes that HAVE a detectable global
response but fewer than THRESHOLD specific movers -- a harder and more honest negative class than "nothing
happened". Reaching the full ~9,900 would need a rebuild pass over gwps.h5ad that keeps the failures and their
flags, which is a separate job.

WHY THE LABEL IS ALSO BETTER, not just more numerous.  The sealed task asks which 20 genes move, and
re-measuring the same knockout reproduces only 12.7 of 20 (absolute ceiling 0.633). "Has a phenotype at all" is
a far more stable quantity than "which twenty" -- so this is 6x the examples with a fraction of the label noise.

AND IT IS A QUESTION SOMEBODY WOULD PAY FOR.  Triaging a target list before screening is the actual bottleneck
in a screening lab. "Which of these 500 genes will show anything?" is worth more than a noisy top-20 guess for
a gene you were going to screen regardless.

THE TASK.  Binary: phenotype (>= MIN_MOVERS specific movers) vs none, over genes that PASSED the cell-count
filter, so "not enough cells" is never confused with "no effect". Scored by AUROC and average precision on
held-out genes, with the sealed cohorts kept sealed and excluded entirely.

ARMS
    frequency_prior   predict the base rate. The floor, and it is what AUROC 0.5 means.
    chan              annotation channels through logistic regression -- the same descriptors the ADRN uses.
    chan_perm         the same channels with ROWS PERMUTED across genes. Every gene keeps a real annotation
                      profile, just somebody else's. If this scores like `chan` the descriptors are decorative.
    n_cells           the number of cells the knockout was measured in, ALONE. This is the control that matters
                      most: if "has a phenotype" is really "was measured deeply enough to see one", the whole
                      result is a technical artefact and must be reported as one.
    chan_ncells       both, to see whether annotation adds anything over sequencing depth.

PREDECLARED, before any number:
    chan > chan_perm AND chan > n_cells, resolved   -> annotation genuinely predicts whether a gene matters in
                                                       this cell line, on 6x the data, and this is a usable
                                                       capability the top-20 task was too noisy to reveal.
    chan ~ n_cells                                  -> we are predicting sequencing depth, not biology. Report
                                                       as an artefact.
    chan ~ chan_perm                                -> the descriptors carry nothing even here, which would be
                                                       the strongest negative in the whole project, since this
                                                       is the easiest question we can ask of them.

Swept over the mover threshold that defines "a phenotype" x classifier regularisation, because "what counts as
an effect" is exactly the kind of undeliberated default this project has been burned by.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
THRESHOLDS = (3, 5, 10)
CVALS = (0.03, 0.3)
N_FOLDS = 5


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("WILL THIS KNOCKOUT DO ANYTHING?  The 6x dataset we were throwing away")
    report("=" * 100)
    report("  PREDECLARED: chan > chan_perm AND chan > n_cells => a real, usable capability.")
    report("               chan ~ n_cells => we are predicting sequencing depth. Artefact.")
    report("               chan ~ chan_perm => the descriptors carry nothing even here.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    Aabs = np.abs(z["M"]).astype(np.float32)
    tide = (Aabs >= A.TAU).mean(0) >= 0.05
    ncell = np.array(z["ncell"], np.float32) if "ncell" in z.files else None
    report(f"\n  npz arrays: {sorted(z.files)}")

    sealed = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    pidx = np.array([i for i, k in enumerate(kos) if k not in sealed])
    pool = [kos[i] for i in pidx]
    report(f"  {len(kos):,} filtered rows; sealed {len(sealed)} excluded; modelling pool {len(pidx):,}")

    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}
    base, _ = A.build_channels(universe, set(pool), report)
    extra, _, tb = C.extra_channels(universe, set(pool), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1).astype(np.float32)
    Cp = chan[[urow[k] for k in pool]]

    Am = Aabs[pidx].copy()
    Am[:, tide] = 0.0
    nmov = (Am >= A.TAU).sum(1)
    # THE DEPTH CONTROL, and a mistake I made and caught mid-run rather than reported.
    #
    # The first version fell back to TOTAL ABSOLUTE SIGNAL when no cell count was available. That is circular:
    # the label is "count of genes with |z| >= tau" and the proxy was "sum of |z| over genes", which is very
    # nearly the same quantity. It scored AUC 0.895 and would have read as "depth beats annotation, artefact"
    # when it was really the label predicting itself. A control has to be independent of the label or it is not
    # a control.
    #
    # The real cell counts are in gwps.h5ad's obs as `num_cells_filtered`, keyed by the same row index the
    # rebuild parses (SYMBOL is field 1 of the underscore-separated obs index). So use those.
    import h5py
    depth = None
    hp = SP / "gwps.h5ad"
    if hp.exists():
        with h5py.File(hp, "r") as f:
            obs = f["obs"]
            oi = [x.decode() for x in obs[obs.attrs.get("_index", "_index")][:]]
            symb = [t.split("_")[1] if len(t.split("_")) > 1 else "" for t in oi]
            nc = np.asarray(obs["num_cells_filtered"][:], np.float64)
        best = {}
        for sy, n in zip(symb, nc):
            if sy:
                best[sy] = max(best.get(sy, 0.0), float(n))
        depth = np.array([[best.get(k, np.nan)] for k in pool], np.float64)
        have = int(np.isfinite(depth).sum())
        report(f"  REAL cell counts recovered for {have:,}/{len(pool):,} pool genes from gwps.h5ad "
               f"(median {np.nanmedian(depth):.0f})")
        depth[~np.isfinite(depth)] = np.nanmedian(depth)
    if depth is None:
        report("  gwps.h5ad absent -- NO honest depth control is available, so the n_cells arm is DROPPED")
        report("  rather than replaced with a proxy that correlates with the label by construction.")
    else:
        depth = np.log1p(depth)
        depth = (depth - depth.mean()) / (depth.std() + 1e-9)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import StratifiedKFold

    rngp = np.random.default_rng(A.SEED + 21)
    Cperm = Cp[rngp.permutation(len(Cp))]

    FEATS = {"chan": Cp, "chan_perm": Cperm}
    if depth is not None:
        FEATS["n_cells"] = depth
        FEATS["chan_ncells"] = np.concatenate([Cp, depth], 1)
    sw_p = Sweeper("chan - chan_perm", axes={"threshold": list(THRESHOLDS), "C": list(CVALS)})
    sw_d = Sweeper("chan - n_cells", axes={"threshold": list(THRESHOLDS), "C": list(CVALS)})
    cells = {}
    for thr in THRESHOLDS:
        y = (nmov >= thr).astype(int)
        report(f"\n  threshold >= {thr} movers: {y.sum():,} positive / {len(y):,} ({100*y.mean():.1f}%)")
        if y.sum() < 60 or (len(y) - y.sum()) < 60:
            report("    too few in one class -- skipped")
            continue
        for Cv in CVALS:
            per = {}
            for nm, F in FEATS.items():
                skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
                oof = np.zeros(len(y))
                for tr, te in skf.split(F, y):
                    m = LogisticRegression(C=Cv, max_iter=2000)
                    m.fit(F[tr], y[tr])
                    oof[te] = m.predict_proba(F[te])[:, 1]
                per[nm] = {"auc": float(roc_auc_score(y, oof)),
                           "ap": float(average_precision_score(y, oof)), "oof": oof}
            # bootstrap the per-gene contrast so the sweep sees paired differences, not two summary numbers
            def paired(a, b):
                r = np.random.default_rng(0)
                d = []
                for _ in range(200):
                    i = r.integers(0, len(y), len(y))
                    if len(set(y[i].tolist())) < 2:
                        continue
                    d.append(roc_auc_score(y[i], per[a]["oof"][i]) - roc_auc_score(y[i], per[b]["oof"][i]))
                return np.array(d)
            cfg, tg = {"threshold": thr, "C": Cv}, f"t{thr}|C{Cv}"
            sw_p.add(cfg, tg, paired("chan", "chan_perm"))
            if "n_cells" in FEATS:
                sw_d.add(cfg, tg, paired("chan", "n_cells"))
            cells[tg] = {"base_rate": float(y.mean()),
                         **{nm: {"auc": v["auc"], "ap": v["ap"]} for nm, v in per.items()}}
            report(f"    C={Cv}: " + " | ".join(
                f"{nm} AUC {per[nm]['auc']:.4f} AP {per[nm]['ap']:.4f}" for nm in FEATS))

    if not cells:
        report("\n  nothing measurable")
        return 1
    for nm, s in (("perm", sw_p), ("depth", sw_d)):
        if not s._cells:
            continue
        report(s.report())
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    ma = float(np.mean([c["chan"]["auc"] for c in cells.values()]))
    mp = float(np.mean([c["chan_perm"]["auc"] for c in cells.values()]))
    md = float(np.mean([c["n_cells"]["auc"] for c in cells.values()])) if "n_cells" in FEATS else float("nan")
    mb = float(np.mean([c["chan_ncells"]["auc"] for c in cells.values()])) if "n_cells" in FEATS else float("nan")
    report(f"\n  MEAN AUC  chan {ma:.4f} | chan_perm {mp:.4f} | n_cells {md:.4f} | chan+n_cells {mb:.4f}")
    ppos = sum(1 for c in sw_p._cells.values() if c["lo"] > 0)
    dpos = sum(1 for c in sw_d._cells.values() if c["lo"] > 0) if sw_d._cells else 0
    report("\n  READING")
    if ppos > len(sw_p._cells) / 2 and dpos > len(sw_d._cells) / 2:
        report(f"  Annotation predicts whether a knockout does anything at AUC {ma:.4f}, beating both its own")
        report(f"  permutation ({mp:.4f}) and the depth control ({md:.4f}). On 6x the examples and a far less")
        report("  noisy label than the top-20 task, the descriptors DO carry usable signal -- the sealed task")
        report("  was too noisy to show it. This is a capability, and it is the triage question a screening lab")
        report("  actually has.")
    elif dpos <= len(sw_d._cells) / 2:
        report(f"  Annotation ({ma:.4f}) does not beat the depth control ({md:.4f}). We would be predicting how")
        report("  well a row was measured, not whether the gene matters. Artefact -- do not report as biology.")
    else:
        report(f"  Annotation ({ma:.4f}) does not beat its own permutation ({mp:.4f}). The descriptors carry")
        report("  nothing even on the easiest question we can ask of them, which is the strongest negative in")
        report("  this project.")

    json.dump({"test": "adrn_will_it_do_anything", "thresholds": list(THRESHOLDS), "C_values": list(CVALS),
               "cells": cells, "mean_auc": {"chan": ma, "chan_perm": mp, "n_cells": md, "chan_ncells": mb},
               "sweeps": {"perm": sw_p.verdict(), "depth": sw_d.verdict()}, "log": log},
              open(OUT / "adrn_will_it_do_anything.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_will_it_do_anything.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
