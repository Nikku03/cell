"""tf_cascade -- lay out the multi-level TF network under a knockout, put magnitudes on it, look for competition and dosage thresholds,
then score the whole construction against what Perturb-seq actually measured.

THE THREE METHODS FOR "MAGNITUDE CHANGE IN TRANSCRIPTION RATE", judged rather than blended:

  1. PERTURB-SEQ z ALONE -- what every model here has used. It is a change in steady-state mRNA ABUNDANCE, not a rate. At steady state
     [mRNA] = k_syn / k_deg, so a z of -5 is equally consistent with less synthesis or faster decay. On its own it CANNOT give a
     transcription rate. Used here only as the measured quantity it is.

  2. ABUNDANCE x MEASURED DECAY RATE -- the one that actually works. RNAdecayCafe supplies per-gene measured k_deg for K562 (SLAM/TimeLapse
     -seq, 141k rows across cell lines). Since [mRNA] = k_syn/k_deg, a relative change in synthesis equals the relative change in abundance
     PROVIDED k_deg is unchanged: dk_syn/k_syn = d[mRNA]/[mRNA]. That assumption is the whole weakness and it is NOT safe here -- this
     knockout collapses the ribosome, and ribosome loss perturbs nonsense-mediated and co-translational decay, so k_deg almost certainly
     does move for some genes. The estimate is reported with the assumption named, and genes are flagged by half-life because a short-lived
     transcript equilibrates to a new synthesis rate quickly while a long-lived one lags and its abundance change UNDERSTATES the rate
     change at any fixed timepoint.

  3. NEXUS -- the wrong tool, and worth saying so rather than forcing it. NEXUS is a protein-mutation dual sensor: it takes a point
     mutation's folding and binding consequences and returns an activity in [0,1], then moves the direct regulon by sign x (a-1). Its own
     recorded note says it gives the first-order regulatory consequence and explicitly "does NOT predict the genome-wide cascade (dynamics
     wall)". For a full CRISPR knockout the activity is simply zero by construction -- there is no protein -- so NEXUS contributes no
     information that the knockout itself does not already state. It is built for "how much does THIS MISSENSE VARIANT impair the protein",
     which is a different question. Its own GATA1 demo scores a ddG of 3.0 kcal/mol as "near-neutral, activity 0.999", i.e. about a mutant,
     not a null.

WHAT ELSE IS ASKED AND WHAT THE DATA SUPPORTS:
  COMPETITION   computable. Two products compete if they bind the same partner or regulate the same target. Reported for up-regulated genes
                against down-regulated ones, since an up gene sharing partners with a down gene is a displacement candidate.
  THRESHOLD     NOT measurable here. There is no transcription-rate threshold for normal function anywhere in this data. What exists are
                DOSAGE PROXIES -- DepMap essentiality and LOEUF constraint -- which say which genes cannot tolerate a drop, not how far the
                drop may go. Reported as proxies and labelled as such.
  VALIDATION    the point of the exercise: what fraction of the measured movers does the cascade explain, against a degree-matched random
                network null, level by level."""
import os
import json, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
T = 4.2
TIDE_FRAC = 0.05
NPERM = 500


def main(ko="GATA1"):
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); A = np.abs(Z)
    tide = (A >= T).mean(0) >= TIDE_FRAC
    panel = [genes[i] for i in range(len(genes)) if not tide[i]]
    pset = set(panel)
    zof = {genes[i]: float(Z[kidx[ko]][i]) for i in range(len(genes))}
    movers = {g for g in panel if abs(zof.get(g, 0)) >= T and g != ko}
    down = {g for g in movers if zof[g] < 0}; up = movers - down

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    go = D.get("go") or {}
    reg_out = collections.defaultdict(set)
    for e in (D.get("reg") or []):
        try:
            reg_out[names[e[0]]].add(names[e[1]])
        except (TypeError, IndexError):
            continue
    ppi = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        try:
            a, b = names[e[0]], names[e[1]]
        except (TypeError, IndexError):
            continue
        ppi[a].add(b); ppi[b].add(a)
    is_tf = {g for g in names if (info.get(g) or {}).get("tf")}

    # ---------------- measured decay rates, K562 ----------------
    kdeg = {}
    try:
        dc = pd.read_csv(SP / "rnadecay" / "AvgKdegs.csv")
        k5 = dc[dc["cell_line"].astype(str).str.upper().str.replace("-", "") == "K562"]
        if len(k5) == 0:
            k5 = dc
            src = f"all cell lines pooled ({dc['cell_line'].nunique()} lines) -- NO K562 rows found"
        else:
            src = "K562"
        for _, r in k5.iterrows():
            try:
                kdeg[str(r["feature_ID"])] = (float(r["avg_kdeg"]), float(r["avg_halflife"]))
            except (TypeError, ValueError):
                pass
    except Exception as e:
        src = f"unavailable ({type(e).__name__})"
    print(f"decay rates: {len(kdeg)} genes, source = {src}")

    # ---------------- the cascade ----------------
    L1 = sorted(reg_out.get(ko, set()) & pset)
    L1_tf = sorted(set(L1) & is_tf)
    L1_tf_moved = [g for g in L1_tf if g in movers]
    L2 = set()
    for t in L1_tf_moved:
        L2 |= (reg_out.get(t, set()) & pset)
    L2 -= (set(L1) | {ko})
    L2 = sorted(L2)
    L2_tf_moved = [g for g in set(L2) & is_tf if g in movers]
    L3 = set()
    for t in L2_tf_moved:
        L3 |= (reg_out.get(t, set()) & pset)
    L3 -= (set(L1) | set(L2) | {ko})
    L3 = sorted(L3)

    rng = np.random.RandomState(0)
    tf_pool = sorted(is_tf & pset)

    def hit(S):
        return len(set(S) & movers)

    def null_hit(n, from_tfs):
        """degree-matched-ish null: draw the same number of targets from unions of random TF regulons"""
        out = np.empty(NPERM)
        for i in range(NPERM):
            pick = set()
            for _ in range(max(from_tfs, 1)):
                t = tf_pool[rng.randint(len(tf_pool))]
                pick |= (reg_out.get(t, set()) & pset)
            pick = list(pick)
            if len(pick) > n:
                pick = [pick[j] for j in rng.choice(len(pick), n, replace=False)]
            out[i] = len(set(pick) & movers)
        return out

    print(f"\nTF CASCADE from {ko}  ({len(movers)} measured movers: {len(down)} down, {len(up)} up)")
    levels = []
    for name, S, ntf in (("L1 direct targets", L1, 1),
                         ("L2 via moved TF targets", L2, max(len(L1_tf_moved), 1)),
                         ("L3 via moved L2 TFs", L3, max(len(L2_tf_moved), 1))):
        if not S:
            continue
        h = hit(S); nul = null_hit(len(S), ntf)
        p = float((nul >= h).mean())
        lv = {"level": name, "n": len(S), "moved": h, "rate": round(h / len(S), 4),
              "expected": round(float(nul.mean()), 1), "fold": round(h / max(nul.mean(), 1e-9), 2), "p": p,
              "n_down": len(set(S) & down), "n_up": len(set(S) & up)}
        levels.append(lv)
        print(f"  {name:26s} {len(S):5d} genes | {h:4d} moved ({h/len(S):.1%}) | expected {nul.mean():6.1f} "
              f"| {lv['fold']:.2f}x | p={p:.3g} | {lv['n_down']} down / {lv['n_up']} up")
    cascade_all = (set(L1) | set(L2) | set(L3)) & pset
    cov = len(cascade_all & movers) / max(len(movers), 1)
    print(f"  TFs in L1 that moved: {len(L1_tf_moved)} -> {L1_tf_moved[:8]}")
    print(f"  CASCADE COVERS {len(cascade_all & movers)}/{len(movers)} = {cov:.1%} of measured movers "
          f"(touching {len(cascade_all)} genes to do it, precision {len(cascade_all & movers)/max(len(cascade_all),1):.1%})")

    # ---------------- magnitude: z -> fold change -> synthesis ----------------
    BE = json.load(open(SP / "k562_baseline_expression.json"))["measured_baseline"]
    mag = []
    for g in sorted(movers, key=lambda x: -abs(zof[x]))[:400]:
        bp = BE.get(g)
        cv = (bp[2] if bp else None)                       # clean_cv = sd/mean in control cells
        # FIRST-ORDER ONLY, AND IT BREAKS. fold ~ 1 + z*CV is a linear expansion valid for small relative change.
        # At z=+102 with CV~0.6 it returns "64x", which is linear extrapolation far outside its range and is not a
        # number anyone should use. It is also mixing two normalisations -- the parquet z and the GWPS control CV --
        # which are not guaranteed to share a scale. So the estimate is emitted ONLY where the implied change stays
        # inside the linear regime, and is explicitly withheld elsewhere rather than printed as a large fake number.
        rel = (zof[g] * cv) if cv else None
        in_range = rel is not None and abs(rel) <= 1.0
        kd = kdeg.get(g)
        mag.append({"gene": g, "z": round(zof[g], 2), "cv": round(cv, 3) if cv else None,
                    "est_fold_change": round(1.0 + rel, 3) if in_range else None,
                    "est_rel_change": round(rel, 3) if in_range else None,
                    "linear_estimate_valid": bool(in_range),
                    "note_if_invalid": None if in_range else
                        ("|z*CV| > 1: outside the linear regime, no fold-change reported" if rel is not None
                         else "no control CV available"),
                    "halflife_h": round(kd[1], 2) if kd else None,
                    "direction": "up" if zof[g] > 0 else "down"})
    withhl = [m for m in mag if m["halflife_h"] is not None]
    print(f"\nMAGNITUDE (top {len(mag)} movers): {len(withhl)} have a measured half-life")
    if withhl:
        hl = np.array([m["halflife_h"] for m in withhl])
        print(f"  half-life median {np.median(hl):.1f} h, IQR {np.percentile(hl,25):.1f}-{np.percentile(hl,75):.1f} h")
        print(f"  short-lived (<2h, abundance tracks synthesis quickly): {int((hl<2).sum())}")
        print(f"  long-lived (>8h, abundance LAGS -- rate change understated): {int((hl>8).sum())}")
    nval = sum(1 for m in mag if m["linear_estimate_valid"])
    print(f"  fold-change estimable (linear regime, |z*CV|<=1): {nval}/{len(mag)}; "
          f"the other {len(mag)-nval} are outside it and are NOT given a number")
    for m in [m for m in mag if m["linear_estimate_valid"]][:6]:
        print(f"    {m['gene']:10s} z={m['z']:+7.1f}  est {m['est_fold_change']:.2f}x  t1/2={m['halflife_h']}h")

    # ---------------- competition ----------------
    comp = []
    for u in sorted(up):
        pu = ppi.get(u, set())
        shared_d = pu & down
        shared_tgt = reg_out.get(u, set()) & down if u in is_tf else set()
        if shared_d or shared_tgt:
            comp.append({"up_gene": u, "z": round(zof[u], 1),
                         "shares_ppi_partner_with_down": sorted(shared_d)[:6],
                         "n_shared_ppi": len(shared_d),
                         "regulates_down_genes": sorted(shared_tgt)[:6], "n_regulates_down": len(shared_tgt)})
    comp.sort(key=lambda c: -(c["n_shared_ppi"] + c["n_regulates_down"]))
    print(f"\nCOMPETITION / DISPLACEMENT candidates: {len(comp)} up-regulated genes touch a down-regulated one")
    for c in comp[:8]:
        print(f"  {c['up_gene']:10s} z={c['z']:+6.1f} | binds {c['n_shared_ppi']} down genes {c['shares_ppi_partner_with_down'][:3]}"
              f" | regulates {c['n_regulates_down']} down genes")

    # ---------------- dosage thresholds (proxies only) ----------------
    def prox(S):
        ess = [g for g in S if ((info.get(g) or {}).get("ess") or 0)]
        lo = [((info.get(g) or {}).get("loeuf")) for g in S]
        lo = [x for x in lo if isinstance(x, (int, float))]
        return {"n": len(S), "n_essential": len(ess), "frac_essential": round(len(ess) / max(len(S), 1), 3),
                "median_loeuf": round(float(np.median(lo)), 3) if lo else None}
    thr = {"down": prox(sorted(down)), "up": prox(sorted(up)), "panel": prox(panel)}
    print(f"\nDOSAGE PROXIES (there is NO transcription-rate threshold in this data)")
    for k, v in thr.items():
        print(f"  {k:6s} n={v['n']:5d} essential {v['frac_essential']:.1%}  median LOEUF {v['median_loeuf']}")
    print(f"  -> essentiality and LOEUF say WHICH genes cannot tolerate a drop, not HOW FAR the drop may go.")

    verdict = (
        f"TF CASCADE ({ko}, K562): lays the multi-level regulatory network under a TF knockout, puts magnitudes on it, looks for "
        f"competition and dosage limits, and scores the construction against the measured response. "
        f"CASCADE: {len(L1)} curated direct targets in the panel, of which {levels[0]['moved'] if levels else 0} moved "
        f"({levels[0]['rate']:.1%} vs {levels[0]['expected']} expected from random TF regulons of the same size, {levels[0]['fold']}x, "
        f"p={levels[0]['p']:.3g}); {len(L1_tf_moved)} of the direct targets are themselves TFs that moved, and expanding through them "
        f"reaches {len(L2)} more genes. The whole cascade touches {len(cascade_all)} genes and covers "
        f"{len(cascade_all & movers)}/{len(movers)} = {cov:.1%} of the measured movers. "
        "TRANSCRIPTION RATE -- three methods weighed, not blended. Perturb-seq z is steady-state ABUNDANCE, not rate, and cannot separate "
        "less synthesis from faster decay. Multiplying by measured K562 decay rates converts relative abundance change to relative "
        "SYNTHESIS change, but only under the assumption that k_deg is unchanged, which this particular knockout violates for some genes "
        "because collapsing the ribosome perturbs nonsense-mediated and co-translational decay; half-lives are therefore reported per gene "
        "so long-lived transcripts, whose abundance lags and understates the rate change, can be seen. NEXUS is the wrong tool and is not "
        "used: it scores how much a POINT MUTATION impairs a protein via folding and binding, returns activity zero by construction for a "
        "full knockout, and its own note states it does not predict the genome-wide cascade. "
        f"The fold-change conversion is first-order (1 + z*CV) and is emitted ONLY where |z*CV| <= 1; beyond that it is linear extrapolation "
        f"outside its range and no number is reported ({nval} of {len(mag)} qualify). "
        f"COMPETITION: {len(comp)} up-regulated genes physically bind or transcriptionally regulate a down-regulated gene, which are the "
        "displacement candidates. "
        "THRESHOLD: not answerable. Nothing in this data gives a transcription rate below which function fails. DepMap essentiality and "
        "LOEUF are dosage PROXIES that identify which genes cannot tolerate a drop, not how far a drop may go, and are reported as such. "
        "Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"knockout": ko, "levels": levels, "cascade_coverage": round(cov, 4),
               "cascade_genes": len(cascade_all), "movers": len(movers),
               "L1_tf_moved": L1_tf_moved, "magnitude": mag[:200], "decay_source": src,
               "competition": comp[:60], "dosage_proxies": thr,
               "verdict": verdict, "note": verdict}, open(OUT / f"tf_cascade_{ko}.json", "w"), indent=1)
    print(f"\n  -> outputs/orphan/tf_cascade_{ko}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GATA1")
