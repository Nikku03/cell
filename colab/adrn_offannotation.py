"""IS IT A PARROT?  Does the model ever predict a gene that shares NO annotation with the knockout -- and is it right?

THE CHALLENGE, stated precisely enough to fail.  The source side of this model is 696 curated annotation channels.
If every correct prediction it makes is a gene that shares an annotation with the knocked-out gene, then the model
has learned nothing that was not already written in the databases it was handed, it cannot propose a relationship
nobody has recorded, and "annotation lookup with a ridge on top" is the honest description.

WHY THAT IS NOT OBVIOUSLY TRUE.  The prediction chain is

    annotation vector --ridge--> programme mix --H--> ranked genes

and **H is estimated from measured data**, never from annotation: 5,120 knockouts x 8,246 genes of real
co-response.  So the model *can* in principle name a gene that no database links to the knockout, because the link
lives in the expression matrix rather than in the vocabulary.  Whether it actually does, and whether those calls
are right, is what this measures.

DEFINITION.  For knockout k and gene g, `off-annotation` means their channel supports are DISJOINT -- not one of
the 696 channels is shared.  No pathway, no complex, no GO term, no Pfam family, no domain, no compartment.

ARMS.  Every arm is scored separately on the on-annotation and off-annotation halves of its own top-20.

    model            the deployed chan2a ridge.
    frequency        rank genes by how often they move across training knockouts.  Carries no query information,
                     so its off-annotation hits are what you get from "these genes move a lot" alone.
    CHANCE           the null that decides it: among genes sharing NO annotation with k, what fraction are movers?
                     That is the hit rate of picking off-annotation genes at random, and it is the number the
                     model's off-annotation precision has to beat.

PREDECLARED.  The model is doing more than lookup only if its OFF-ANNOTATION precision beats CHANCE past a
bootstrap CI on both sealed cohorts.  If off-annotation precision sits at chance, then every correct call is
carried by shared annotation, and the parrot description is correct and should be adopted.

Reported alongside, because it is the quantity that matters for "can it propose a novel hypothesis": the NOVELTY
SHARE -- of all the model's correct calls, what fraction are off-annotation.  A model can beat chance off-annotation
and still make 95% of its living on-annotation; that would be a real but narrow capability, and saying so requires
the share, not just the contrast.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
BOOT = 10_000


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("IS IT A PARROT? off-annotation predictions vs chance")
    report("=" * 100)
    report("  PREDECLARED: off-annotation precision must beat CHANCE on both cohorts, else 'lookup' is correct.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]

    # channels over BOTH knockouts and response genes -- the overlap test needs a vector for the target side too
    universe = sorted(set(kos) | set(genes) | sealed)
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    report(f"  channel matrix {chan.shape} over {len(universe):,} symbols")

    # boolean support, used only for the disjointness test
    sup = (chan[[urow[g] for g in genes]] > 0)
    n_chan_per_gene = sup.sum(1)
    report(f"  response genes with at least one channel: {int((n_chan_per_gene > 0).sum()):,} of {len(genes):,}")

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    w = A.ridge_fit(chan[[urow[k] for k in train]], W, A.PENALTY)

    freq = (Amat[[ki[k] for k in train]] >= A.TAU).mean(0)
    freq[tide] = -np.inf

    out, means = {}, {}
    for tag in ("", "_holdout"):
        ans = answers.get(tag)
        if not ans:
            continue
        cohort = sorted(ans)
        name = tag or "cohort1"
        mix = A.ridge_apply(chan[[urow[k] for k in cohort]], w)
        per = {a: {"on": [], "off": []} for a in ("model", "frequency")}
        chance_off, chance_on, nov = [], [], []
        for j, k in enumerate(cohort):
            movers = set(ans[k]["movers"])
            ksup = chan[urow[k]] > 0
            disjoint = ~(sup & ksup).any(1)          # genes sharing NO channel with k
            s = (mix[j] @ H).copy()
            s[tide] = -np.inf
            gi = genes.index(k) if k in genes else -1
            if gi >= 0:
                s[gi] = -np.inf
            for arm, sc in (("model", s), ("frequency", freq)):
                top = np.argsort(-sc)[:A.NPRED]
                for t in top:
                    hit = 1.0 if genes[t] in movers else 0.0
                    per[arm]["off" if disjoint[t] else "on"].append(hit)
            # CHANCE: among off-annotation genes, what fraction are movers?  (and the on-annotation counterpart)
            mv = np.array([g in movers for g in genes])
            live = np.isfinite(s)
            offp, onp = disjoint & live, (~disjoint) & live
            if offp.sum():
                chance_off.append(mv[offp].mean())
            if onp.sum():
                chance_on.append(mv[onp].mean())
            top = np.argsort(-s)[:A.NPRED]
            hits = [t for t in top if genes[t] in movers]
            if hits:
                nov.append(np.mean([1.0 if disjoint[t] else 0.0 for t in hits]))

        def m(v):
            return float(np.mean(v)) if len(v) else float("nan")

        report(f"\n  === {name} ({len(cohort)} knockouts) ===")
        report(f"  {'arm':<14} {'on-annotation':>16} {'off-annotation':>16}")
        for arm in ("model", "frequency"):
            report(f"  {arm:<14} {m(per[arm]['on']):>16.4f} {m(per[arm]['off']):>16.4f}"
                   f"   (n={len(per[arm]['on'])}/{len(per[arm]['off'])})")
        report(f"  {'CHANCE':<14} {m(chance_on):>16.4f} {m(chance_off):>16.4f}   <- the null, not a competitor")

        # the gate: model off-annotation precision vs chance, bootstrapped over the predictions themselves
        off = np.array(per["model"]["off"], float)
        res = {}
        if len(off) >= 20:
            ch = m(chance_off)
            br = np.random.default_rng(0)
            bs = off[br.integers(0, len(off), size=(BOOT, len(off)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            v = "BEATS CHANCE" if lo > ch else ("at/below chance" if hi <= ch else "unresolved")
            report(f"\n  off-annotation precision {off.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]  vs chance {ch:.4f}"
                   f"   -> {v}")
            res = {"off_prec": float(off.mean()), "ci": [lo, hi], "chance": ch, "verdict": v}
        share = m(nov)
        report(f"  NOVELTY SHARE: {share:.1%} of the model's correct calls share no annotation with the knockout")
        frac_off_slots = len(per["model"]["off"]) / float(len(per["model"]["off"]) + len(per["model"]["on"]))
        report(f"  (it spends {frac_off_slots:.1%} of its 20 slots on off-annotation genes)")
        means[name] = {a: {"on": m(per[a]["on"]), "off": m(per[a]["off"])} for a in per}
        means[name]["chance"] = {"on": m(chance_on), "off": m(chance_off)}
        out[name] = {"gate": res, "novelty_share": share, "off_slot_share": frac_off_slots}

    gate = all(v["gate"].get("verdict") == "BEATS CHANCE" for v in out.values() if v.get("gate"))
    report(f"\n  GATE: {'PASS -- it predicts correctly OFF its own annotations' if gate else 'FAIL -- lookup describes it'}")
    json.dump({"test": "adrn_offannotation", "means": means, "cohorts": out, "gate_pass": bool(gate),
               "log": log}, open(OUT / "adrn_offannotation.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_offannotation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
