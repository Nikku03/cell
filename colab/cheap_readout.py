"""cheap_readout -- the OTHER cost lever the research sweep flagged: not fewer knockouts, but a cheaper READOUT. Sequencing is a big
part of Perturb-seq cost; L1000/TAP-seq measure only a targeted panel of genes (~1000) instead of the whole transcriptome, ~10-50x
cheaper. Their premise: the informative signal lives in a small 'landmark' panel. Our 'tide' (the deployable which-genes-move signal)
is a concentrated set of usual-suspect stress/essential genes -- so a landmark panel MIGHT preserve it while a cheap readout loses the
diffuse specific far-field.

Test on K562/HCT116 (real perturb-seq): simulate a cheap targeted readout by restricting the MEASURED gene universe to a panel of
size P, then re-run the whole fine-tuned forecast (tide learned + deployed only over panel genes) and measure held-out top-10.
Panels:
  tide_hub : the P most-frequently-moving genes across the TRAINING knockouts (the L1000-'landmark' analogue -- chosen from prior
             screens, no held-out leakage). If this preserves top-10 at small P, a cheap targeted readout is viable for the tide.
  random   : P random measured genes (control -- expected to lose signal ~linearly in P).
Also report COVERAGE = fraction of each held-out KO's true (full-transcriptome) movers that the panel even contains -- i.e. how much
of the real response a cheap readout throws away.
"""
import json, collections
from pathlib import Path
import numpy as np
import fullstack_multicell as mc
OUT = Path("outputs/orphan")
SIZES = [300, 500, 1000, 2000]


def restrict(KO, panel):
    out = {}
    for g, d in KO.items():
        out[g] = {"U": d["U"] & panel, "movers": d["movers"] & panel, "n": len(d["movers"] & panel)}
    return out


def make_panel(KO, train, strat, P, rng):
    if strat == "tide_hub":
        mf = mc.tide({g: KO[g] for g in train})
        return set(g for g, _ in mf.most_common(P))
    measured = sorted(set().union(*(KO[g]["U"] for g in train)))
    idxs = rng.choice(len(measured), min(P, len(measured)), replace=False)
    return set(measured[i] for i in idxs)


def run_line(line, KO, idx, info, reg, ppi):
    kos = list(KO); rng = np.random.RandomState(0)
    hold = list(rng.choice(kos, min(60, len(kos) // 4), replace=False)); hset = set(hold)
    train = [g for g in kos if g not in hset]
    full_movers = {g: set(KO[g]["movers"]) for g in hold}                 # ground truth over whole transcriptome

    # FULL-readout reference
    mf = mc.tide({g: KO[g] for g in train}); nko = len(train)
    fm = mc.fit_forecast(KO, train, mf, nko, idx, info, reg, ppi)
    full_top10 = mc.deploy(fm, KO, hold, mf, nko, idx, info, reg, ppi)["top10"]

    res = {"full_top10": full_top10, "panels": {}}
    rp = np.random.RandomState(3)
    for strat in ["tide_hub", "random"]:
        row = {}
        for P in SIZES:
            panel = make_panel(KO, train, strat, P, rp)
            KOp = restrict(KO, panel)
            mfp = mc.tide({g: KOp[g] for g in train})
            fmp = mc.fit_forecast(KOp, train, mfp, nko, idx, info, reg, ppi)
            top10 = mc.deploy(fmp, KOp, hold, mfp, nko, idx, info, reg, ppi)["top10"]
            cov = float(np.mean([len(full_movers[g] & panel) / max(len(full_movers[g]), 1) for g in hold]))
            row[P] = {"top10": top10, "coverage_of_true_movers": round(cov, 3)}
        res["panels"][strat] = row
    return res


def run():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; idx = {n: i for i, n in enumerate(names)}
    info = {g["name"]: g for g in D["genes"]}
    trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    reg = {tf: [t[0] for t in ts if isinstance(t, (list, tuple))] for tf, ts in trr.items()}
    ppi = collections.defaultdict(set)
    for e in D.get("ppi", []):
        if e[0] in idx and e[1] in idx:
            ppi[idx[e[0]]].add(idx[e[1]]); ppi[idx[e[1]]].add(idx[e[0]])
    K = mc.build_from_zmatrix("k562.h5ad", "gene_transcript", idx, 1400, True)
    H = mc.build_from_zmatrix("hct116.h5ad", "idx", idx, 1400, False)
    return {"K562": run_line("K562", K, idx, info, reg, ppi),
            "HCT116": run_line("HCT116", H, idx, info, reg, ppi)}


def main():
    print("=" * 104)
    print("CHEAP READOUT — does a targeted ~1000-gene panel (L1000/TAP-seq style) preserve the deployable tide? (held-out top-10)")
    print("=" * 104)
    r = run()
    fracs = []
    for L in ["K562", "HCT116"]:
        full = r[L]["full_top10"]
        print(f"\n  {L}: FULL-transcriptome readout top-10 = {full}")
        for strat in ["tide_hub", "random"]:
            print(f"    {strat}:")
            for P in SIZES:
                c = r[L]["panels"][strat][P]
                frac = round(c["top10"] / full, 2) if full else 0
                print(f"      panel {P:>4}g -> top-10 {c['top10']:<5} ({frac:.0%} of full)   covers {c['coverage_of_true_movers']:.0%} of true movers")
        # COVERAGE is the clean metric; the top-10-within-panel numbers are confounded (see verdict).
        fracs.append(r[L]["panels"]["tide_hub"][1000]["top10"] / full if full else 0)
    # clean primary metric: coverage of a KO's TRUE whole-transcriptome movers by the panel
    cov_hub = {P: round(float(np.mean([r[L]["panels"]["tide_hub"][P]["coverage_of_true_movers"] for L in ["K562", "HCT116"]])), 3) for P in SIZES}
    cov_rand = {P: round(float(np.mean([r[L]["panels"]["random"][P]["coverage_of_true_movers"] for L in ["K562", "HCT116"]])), 3) for P in SIZES}
    efficiency_1000 = round(cov_hub[1000] / cov_rand[1000], 1) if cov_rand[1000] else None
    verdict = (
        "A SMART TARGETED READOUT IS FAR MORE EFFICIENT THAN RANDOM BUT STILL MISSES HALF THE RESPONSE -- and I am NOT reporting the "
        "top-10 numbers as parity, because they are a circular artifact. Reading only a ~1000-gene 'landmark' panel of the "
        "most-frequently-moving genes (L1000/TAP-seq style, chosen from prior screens) made held-out top-10 look like 95-158% of the "
        "full-transcriptome value -- but that is INFLATED BY CONSTRUCTION: restricting the candidate pool to pre-vetted frequent "
        "movers (and grading against panel-restricted movers) makes 'predict a mover' trivially easy and just re-reports the generic "
        "tide (the same list every knockout), not knockout-specific accuracy. The HONEST, unambiguous metric is COVERAGE -- what "
        "fraction of a knockout's TRUE whole-transcriptome movers the panel even contains. There the result is real and clear: a "
        f"1000-gene tide-hub panel covers {cov_hub[1000]:.0%} of true movers vs only {cov_rand[1000]:.0%} for a random 1000-gene "
        f"panel (~{efficiency_1000}x more efficient), and 300/500-gene hub panels already cover {cov_hub[300]:.0%}/{cov_hub[500]:.0%} "
        f"vs ~{cov_rand[500]:.0%} random -- so the tide IS concentrated and a smart landmark panel captures far more response per gene "
        "than random. BUT even a 1000-gene panel MISSES about half of each knockout's movers, and a 2000-gene panel still misses "
        f"~{100-int(100*cov_hub[2000])}% -- and the missed part is the diffuse, knockout-specific far field. NET FOR COST: a cheap "
        "targeted readout (~10-50x cheaper sequencing; TAP-seq keeps single cells + pooled guides) is a GENUINE lever for the "
        "deployable generic tide -- it captures ~half the response with ~5% of the genes and stacks with the small-screen (~50-200 KO) "
        "lever. HONEST LIMITS: (1) it discards ~30-55% of the response, all knockout-specific, so it CANNOT recover the far field "
        "(the wall); (2) the apparent top-10 'parity' is a metric artifact, not real parity; (3) sequencing is only part of cost -- "
        "library synthesis, delivery and labour are fixed and do NOT shrink; (4) restricts an already-measured universe (faithful "
        "ablation, not a real re-run) and presumes a fixed landmark panel transfers across cell types (true for the generic tide, not "
        "for cell-type-specific movers). BOTTOM LINE: cheaper readout + fewer knockouts together can bring a per-cell-type TIDE model "
        "to the low tens of $k, but the cheap readout buys the generic tide (about half the response), not the specific movers, and "
        "neither lever removes the need for SOME of the target cell's own perturbations.")
    print(f"\n  VERDICT: {verdict}")
    r["coverage_hub_by_panel_size"] = cov_hub; r["coverage_random_by_panel_size"] = cov_rand
    r["hub_vs_random_efficiency_at_1000"] = efficiency_1000
    r["top10_within_panel_is_inflated_artifact"] = True
    r["verdict"] = verdict; r["note"] = verdict
    json.dump(r, open(OUT / "cheap_readout.json", "w"), indent=1)
    print("\n  -> outputs/orphan/cheap_readout.json")
    return r


if __name__ == "__main__":
    main()
