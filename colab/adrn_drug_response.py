"""DRUG RESPONSE COLD START: does a model trained only on GENETIC knockouts predict what a DRUG does?

THE QUESTION.  Every result in this project so far has been trained on CRISPR knockouts and tested on CRISPR
knockouts.  The claim implicitly being made -- that this is a cell model rather than a Perturb-seq interpolator --
requires that the machinery transfer to a perturbation it has never seen the modality of.  A small molecule is
that test.  It hits a protein rather than deleting a transcript, it hits several proteins at once, it acts in
hours rather than days, and nothing in the training data is a drug.

THE CHAIN, and it is deliberately the shortest one available:

    drug --(annotated protein target)--> gene symbol(s) --(mean channel vector)--> ridge --> programme mix
         --(H)--> ranked genes,  scored against the drug's MEASURED movers in sci-Plex K562.

The target annotation is sci-Plex's own `target` field (87 families, e.g. "Aurora Kinase", "Bcr-Abl,c-Kit,Src").
Those are protein families, not HGNC symbols, so TARGET_GENES below maps them.  That map was written from the
target strings alone, before any response was computed, and it is prior knowledge of exactly the same kind as the
annotation channels -- not a fitted parameter.

ARMS.  The first is the model; the next two are what it has to beat; the last two bound the scale.

    target_channels    the real drug -> target gene -> channel chain.
    shuffled_targets   the drug -> target map permuted across drugs (20 draws).  Tests whether the prediction
                       uses WHICH drug it is, or just the average shape of a drug response.
    frequency_prior    rank genes by how many OTHER drugs they move (leave-one-drug-out, so no self-leak).
                       This is the arm that has killed most ideas in this project and it is the real opponent:
                       drugs at high dose produce a large shared stress response, and "the genes that always
                       move" is a very strong guess that requires no model at all.
    random             floor.
    split_half         same drug, half its cells against the other half.  Measurement error on the same
                       quantity at the same depth -- the ceiling, not a competitor.

PREDECLARED GATE.  target_channels must beat BOTH frequency_prior AND shuffled_targets past a 95% bootstrap CI
over drugs.  Beating shuffled but not frequency_prior is a FAILURE, not a partial pass: it would mean the model
distinguishes drugs from each other while still being worse than a lookup table of commonly-moving genes.

SYMMETRIC MASKING.  Perturb-seq tide genes (the ones that move under nearly every knockout) are removed from the
prediction, from the ground truth, and from the frequency prior alike.  Masking predictions but not the answer
key would handicap the model; masking neither would let every arm score on the same generic stress genes.

MEASUREMENT.  K562 (the training cell line), 24 h, the two highest doses (1000 + 10000 nM) pooled -- prior work
in this repo showed response magnitude rises with dose, so this is where there is signal to predict.  Movers are
|z| >= 2 with z standardised per gene across drugs; a drug needs >= MIN_MOVERS to be scored at all, because
precision against an empty answer key is undefined rather than zero.
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
DOSES_KEEP = (1000.0, 10000.0)
Z_TAU, MIN_MOVERS, NSHUF, BOOT = 2.0, 20, 20, 10_000

# Protein-family -> HGNC symbols.  Written from the 87 target strings, before any response was computed.
# Families with no defined protein target ("Others", "DNA alkylator", "Dehydrogenase") are absent on purpose:
# a drug whose only annotation is "Others" carries no information for this chain and is dropped, not guessed.
TARGET_GENES = {
    "ALK": ["ALK"], "c-Met": ["MET"], "AMPK": ["PRKAA1", "PRKAA2", "PRKAB1", "PRKAG1"],
    "Androgen Receptor": ["AR"], "Aromatase": ["CYP19A1"], "Aurora Kinase": ["AURKA", "AURKB", "AURKC"],
    "Bcr-Abl": ["ABL1", "BCR"], "FLT3": ["FLT3"], "JAK": ["JAK1", "JAK2", "JAK3", "TYK2"],
    "c-RET": ["RET"], "FGFR": ["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
    "CDK": ["CDK1", "CDK2", "CDK4", "CDK6", "CDK9"], "VEGFR": ["KDR", "FLT1", "FLT4"],
    "Autophagy": ["ATG5", "ATG7", "ULK1", "MAP1LC3B"], "ROCK": ["ROCK1", "ROCK2"],
    "Sirtuin": ["SIRT1", "SIRT2", "SIRT3"], "Bcl-2": ["BCL2", "BCL2L1", "MCL1"], "c-Kit": ["KIT"],
    "Src": ["SRC", "LYN", "FYN"], "Beta Amyloid": ["APP"], "Gamma-secretase": ["PSEN1", "NCSTN", "APH1A"],
    "CCR": ["CCR2", "CCR5"], "COX": ["PTGS1", "PTGS2"], "CSF-1R": ["CSF1R"], "PDGFR": ["PDGFRA", "PDGFRB"],
    "DNA Methyltransferase": ["DNMT1", "DNMT3A", "DNMT3B"], "DNA/RNA Synthesis": ["POLA1", "RRM1", "RRM2"],
    "Dopamine Receptor": ["DRD1", "DRD2"], "E3 Ligase ": ["MDM2"], "p53": ["TP53"], "EGFR": ["EGFR"],
    "HER2": ["ERBB2"], "HDAC": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
    "Epigenetic Reader Domain": ["BRD2", "BRD3", "BRD4"],
    "Estrogen/progestogen Receptor": ["ESR1", "ESR2", "PGR"], "FAAH": ["FAAH"], "FAK": ["PTK2"],
    "GABA Receptor": ["GABRA1"], "Glucocorticoid Receptor": ["NR3C1"], "HIF": ["HIF1A", "EPAS1"],
    "HSP (e.g. HSP90)": ["HSP90AA1", "HSP90AB1"], "Histamine Receptor": ["HRH1", "HRH2"],
    "Histone Acetyltransferase": ["EP300", "CREBBP", "KAT2B"],
    "Histone Demethylase": ["KDM1A", "KDM4A", "KDM5A", "KDM6A"],
    "Histone Methyltransferase": ["EZH2", "DOT1L", "EHMT2", "SETD2"], "IDH1": ["IDH1"], "IGF-1R": ["IGF1R"],
    "LPA Receptor": ["LPAR1"], "Lipoxygenase": ["ALOX5", "ALOX12"], "MAO": ["MAOA", "MAOB"],
    "MEK": ["MAP2K1", "MAP2K2"], "MT Receptor": ["MTNR1A", "MTNR1B"],
    "Microtubule Associated": ["TUBB", "TUBA1A", "TUBB4B"], "NF-κB": ["NFKB1", "RELA"],
    "Nrf2": ["NFE2L2"], "PARP": ["PARP1", "PARP2"], "Raf": ["BRAF", "RAF1"],
    "PI3K": ["PIK3CA", "PIK3CB", "PIK3CD"], "PKA": ["PRKACA"], "PKC": ["PRKCA", "PRKCB", "PRKCD"],
    "PLK": ["PLK1"], "Pim": ["PIM1", "PIM2"], "STAT": ["STAT3", "STAT5A", "STAT5B"], "Survivin": ["BIRC5"],
    "TGF-beta/Smad": ["TGFBR1", "TGFBR2", "SMAD3"], "TNF-alpha": ["TNF"], "Telomerase": ["TERT"],
    "Tie-2": ["TEK"], "Topoisomerase": ["TOP1", "TOP2A", "TOP2B"], "Trk receptor": ["NTRK1", "NTRK2"],
    "Wnt/beta-catenin": ["CTNNB1", "TNKS", "TNKS2", "PORCN"], "mTOR": ["MTOR"],
}


def _dec(a):
    return [x.decode() if isinstance(x, bytes) else x for x in a]


def targets_of(s):
    """Target string -> gene symbols.  Unmapped families contribute nothing rather than a guess."""
    out = []
    for part in str(s).split(","):
        for g in TARGET_GENES.get(part.strip(), ()):
            if g not in out:
                out.append(g)
    return out


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("DRUG RESPONSE COLD START: knockout-trained model vs measured sci-Plex drug response")
    report("=" * 100)
    report("  PREDECLARED: target_channels must beat BOTH frequency_prior AND shuffled_targets past a 95% CI.")

    # ---------------------------------------------------------------- knockout side (the trained model)
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    report(f"  knockout matrix {Amat.shape}, tide genes {int(tide.sum()):,}")

    # ---------------------------------------------------------------- drug side (the held-out modality)
    hf = h5py.File(SP / "sciplex.h5ad", "r")
    sgenes = _dec(hf["var"]["gene_symbol"][:])
    gset = {g: i for i, g in enumerate(genes)}
    gmap = np.full(len(sgenes), -1, np.int64)
    kept_idx = []                                    # index into `genes` for each kept column
    for j, gn in enumerate(sgenes):
        if gn in gset and gmap[j] == -1 and not tide[gset[gn]]:
            gmap[j] = len(kept_idx)
            kept_idx.append(gset[gn])
    K = len(kept_idx)
    kept_names = [genes[i] for i in kept_idx]
    report(f"  sci-Plex genes {len(sgenes):,}; shared with knockout space and non-tide: {K:,}")

    cl = hf["obs"]["cell_line"]
    clcats = _dec(cl["categories"][:])
    k562 = clcats.index("K562")
    clcodes = cl["codes"][:]
    tm = hf["obs"]["time"][:]
    dv = hf["obs"]["dose_value"][:]
    pc = hf["obs"]["perturbation"]
    pcats = _dec(pc["categories"][:])
    pcodes = pc["codes"][:]
    tg = hf["obs"]["target"]
    tgcats = _dec(tg["categories"][:])
    tgcodes = tg["codes"][:]

    base_sel = (clcodes == k562) & (tm == 24.0)
    is_ctrl = (dv == 0.0) | np.array([("control" in p.lower() or "vehicle" in p.lower()) for p in pcats])[pcodes]
    treat = base_sel & ~is_ctrl & np.isin(dv, DOSES_KEEP)
    ctrl = base_sel & is_ctrl
    report(f"  K562/24h cells: {int(ctrl.sum()):,} vehicle, {int(treat.sum()):,} treated at doses {DOSES_KEEP}")

    # keep only drugs whose target annotation maps to at least one gene
    drug_tg = {}
    for i in np.where(treat)[0]:
        d = pcats[pcodes[i]]
        if d not in drug_tg:
            t = targets_of(tgcats[tgcodes[i]])
            if t:
                drug_tg[d] = t
    drugs = sorted(drug_tg)
    report(f"  drugs with a mapped target: {len(drugs)} of {len(set(pcats[c] for c in pcodes[treat]))}")

    # groups: 0 = vehicle, 2d+1 / 2d+2 = the two random halves of drug d
    didx = {d: i for i, d in enumerate(drugs)}
    ng = 2 * len(drugs) + 1
    grp = np.full(len(pcodes), -1, np.int64)
    grp[ctrl] = 0
    rng = np.random.default_rng(A.SEED)
    ti = np.where(treat)[0]
    half = rng.integers(0, 2, size=len(ti))
    for n, i in enumerate(ti):
        d = pcats[pcodes[i]]
        if d in didx:
            grp[i] = 2 * didx[d] + 1 + int(half[n])

    indptr = hf["X"]["indptr"][:]
    data_ds, ind_ds = hf["X"]["data"], hf["X"]["indices"]
    sums = np.zeros(ng * K)
    tot = np.zeros(ng)
    B = 20000
    for i0 in range(0, len(pcodes), B):
        i1 = min(i0 + B, len(pcodes))
        a, b = int(indptr[i0]), int(indptr[i1])
        if b <= a:
            continue
        d = data_ds[a:b].astype(np.float64)
        cols = ind_ds[a:b]
        rows = np.repeat(np.arange(i0, i1), np.diff(indptr[i0:i1 + 1]))
        gc = grp[rows]
        gn = gmap[cols]
        m = (gc >= 0) & (gn >= 0)
        if not m.any():
            continue
        sums += np.bincount(gc[m] * K + gn[m], weights=d[m], minlength=ng * K)
        tot += np.bincount(gc[m], weights=d[m], minlength=ng)
    hf.close()
    sums = sums.reshape(ng, K)

    def cpm(s, t):
        return np.log1p(s / max(t, 1.0) * 1e6)

    veh = cpm(sums[0], tot[0])
    full, hA, hB = {}, {}, {}
    for d, i in didx.items():
        sa, sb = sums[2 * i + 1], sums[2 * i + 2]
        ta, tb = tot[2 * i + 1], tot[2 * i + 2]
        full[d] = cpm(sa + sb, ta + tb) - veh
        hA[d] = cpm(sa, ta) - veh
        hB[d] = cpm(sb, tb) - veh

    D = np.array([full[d] for d in drugs])
    mu, sd = D.mean(0), D.std(0) + 1e-9
    Z = (D - mu) / sd
    movers = {d: set(np.where(np.abs(Z[i]) >= Z_TAU)[0]) for i, d in enumerate(drugs)}
    scored = [d for d in drugs if len(movers[d]) >= MIN_MOVERS]
    report(f"  drugs with >= {MIN_MOVERS} movers at |z| >= {Z_TAU}: {len(scored)} of {len(drugs)}")
    if len(scored) < 20:
        report("  ABORT: too few responding drugs to bootstrap over.")
        return 1

    # ---------------------------------------------------------------- fit the knockout model
    tset = set(kos)
    universe = sorted(tset | {g for d in scored for g in drug_tg[d]})
    base, _ = A.build_channels(universe, tset, report)
    extra, _, tb_ = C.extra_channels(universe, tset, report)
    chan = np.concatenate([base, extra[:, :tb_]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}

    from sklearn.decomposition import NMF
    X = Amat.copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    w = A.ridge_fit(chan[[urow[k] for k in kos]], W, A.PENALTY)
    Hk = H[:, kept_idx]                                  # programme loadings on the shared non-tide genes
    report(f"  fitted {A.K} programmes on {len(kos):,} knockouts; decoding onto {K:,} shared genes")

    def predict(tgenes):
        v = chan[[urow[g] for g in tgenes]].mean(0, keepdims=True)
        s = A.ridge_apply(v, w)[0] @ Hk
        return np.argsort(-s)[:A.NPRED]

    def prec(top, d):
        return len(set(top.tolist()) & movers[d]) / float(A.NPRED)

    # ---------------------------------------------------------------- arms
    res = collections.defaultdict(list)
    kotrain = set(kos)
    cold = []
    freq = np.zeros(K)
    for d in scored:
        for g in movers[d]:
            freq[g] += 1
    rr = np.random.default_rng(A.SEED + 7)
    perms = [rr.permutation(len(scored)) for _ in range(NSHUF)]

    for i, d in enumerate(scored):
        res["target_channels"].append(prec(predict(drug_tg[d]), d))
        # leave-one-drug-out frequency prior: this drug's own movers do not vote for it
        f = freq.copy()
        for g in movers[d]:
            f[g] -= 1
        res["frequency_prior"].append(prec(np.argsort(-f)[:A.NPRED], d))
        res["random"].append(prec(rr.choice(K, A.NPRED, replace=False), d))
        sh = []
        for p in perms:
            j = int(p[i])
            if scored[j] == d:
                j = (j + 1) % len(scored)
            sh.append(prec(predict(drug_tg[scored[j]]), d))
        res["shuffled_targets"].append(float(np.mean(sh)))
        za = (hA[d] - mu) / sd
        zb = (hB[d] - mu) / sd
        mb = set(np.where(np.abs(zb) >= Z_TAU)[0])
        res["split_half"].append(len(set(np.argsort(-np.abs(za))[:A.NPRED].tolist()) & mb) / float(A.NPRED))
        cold.append(not any(g in kotrain for g in drug_tg[d]))

    cold = np.array(cold)
    report(f"\n  scored {len(scored)} drugs; {int(cold.sum())} are true cold start "
           f"(no target gene was ever knocked out in training)")

    report(f"\n  {'arm':<20} {'precision@10':>13}")
    means = {}
    for k in ("split_half", "target_channels", "frequency_prior", "shuffled_targets", "random"):
        v = np.array(res[k])
        means[k] = float(v.mean())
        note = "  <- CEILING, not a competitor" if k == "split_half" else ""
        report(f"  {k:<20} {v.mean():>13.4f}{note}")

    report(f"\n  {'contrast':<40} {'delta':>9} {'95% CI':>22}  verdict")
    out = {}
    br = np.random.default_rng(0)
    tc = np.array(res["target_channels"])
    for opp in ("frequency_prior", "shuffled_targets", "random"):
        dif = tc - np.array(res[opp])
        bs = dif[br.integers(0, len(dif), size=(BOOT, len(dif)))].mean(1)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        v = "RESOLVED +" if lo > 0 else ("RESOLVED -" if hi < 0 else "unresolved")
        report(f"  {'target_channels - ' + opp:<40} {dif.mean():>+9.4f} [{lo:>+9.4f},{hi:>+9.4f}]  {v}")
        out[opp] = {"delta": float(dif.mean()), "lo": lo, "hi": hi, "verdict": v}

    gate = out["frequency_prior"]["lo"] > 0 and out["shuffled_targets"]["lo"] > 0
    report(f"\n  GATE: {'PASS' if gate else 'FAIL'} -- needs BOTH frequency_prior and shuffled_targets beaten.")
    if not gate and out["shuffled_targets"]["lo"] > 0:
        report("  (drug identity IS used -- shuffling targets hurts -- but the model still loses to a lookup"
               " table of commonly-moving genes, which is the failure this gate was written to catch.)")

    if cold.sum() >= 10:
        report(f"\n  {'stratum':<20} {'n':>4} {'target':>9} {'freq_prior':>11}")
        for name, m in (("cold start", cold), ("target measured", ~cold)):
            if m.sum() == 0:
                continue
            report(f"  {name:<20} {int(m.sum()):>4} {tc[m].mean():>9.4f} "
                   f"{np.array(res['frequency_prior'])[m].mean():>11.4f}")

    json.dump({"test": "adrn_drug_response", "n_drugs_scored": len(scored),
               "n_cold_start": int(cold.sum()), "n_genes": K, "doses": list(DOSES_KEEP),
               "z_tau": Z_TAU, "means": means, "contrasts": out, "gate_pass": bool(gate),
               "drugs": scored, "log": log},
              open(OUT / "adrn_drug_response.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_drug_response.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
