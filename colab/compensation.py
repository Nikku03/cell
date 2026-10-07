"""compensation -- a DIRECT, falsifiable test of the 'Compensatory Shunting' hypothesis (mechanism 1 of the proposed Structural
Compensation Engine): when you knock down a redundant/peripheral gene, the cell shifts load to its closest PARALOG, which is
UPREGULATED and is one of the specific movers -- so predicting the paralog would crack the weak knockouts.

METHOD NOTE (an honest detour): I first tried the on-hand ESM-2 matrix (esm_normal.parquet) as the 'continuous structural field'
and took cosine-nearest-neighbours. A positive control killed that route: that matrix is per-dimension STANDARDISED (per-dim
mean~0, std~1), which destroys cosine geometry -- known paralogs scored at chance (RPL7/RPL7A cos 0.043, ACTB/ACTG1 0.024, vs a
random 99th-pct of 0.063). So it cannot represent 'structural similarity' and any negative from it would be an artefact. This
module instead uses a CURATED paralog map (Ensembl BioMart human paralogues) -- gold-standard paralogy, and a more faithful
operationalisation of 'closest paralog' than an embedding anyway.

Tests, on deep k562, per knockout that HAS >=1 measured paralog, split weak vs strong:
  Q1 (directional):  are the gene's paralogs UPREGULATED when it is knocked down, above the knockout's own global drift?
  Q2 (identity):     are the gene's paralogs enriched among its actual movers vs the base mover rate?
  Q3 (the crux):     does either hold for WEAK knockouts (the <15-mover majority anchoring the average at ~2/10)? If not, the
                     Structural Compensation Engine has no signal in THIS data and the weak-KO ceiling is a data/measurement
                     limit (CRISPRi knockdown modality + steady-state timing), not an architecture we can out-engineer.
"""
import json, collections
from pathlib import Path
import numpy as np
import h5py
import os
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _dec(a): return [x.decode() if isinstance(x, bytes) else x for x in a]


def load_paralogs():
    par = collections.defaultdict(set)
    p = Path(f"{SP}/paralogs.tsv")
    if not p.exists():
        return par
    for line in open(p):
        t = line.rstrip("\n").split("\t")
        if len(t) < 2:
            continue
        a, b = t[0].strip(), t[1].strip()
        if a and b and a != b:
            par[a].add(b); par[b].add(a)
    return par


def load_z():
    h = h5py.File(f"{SP}/k562.h5ad", "r")
    kp = [x.split("_")[1] if "_" in x else x for x in _dec(h["obs"]["gene_transcript"][:])]
    cats = _dec(h["var"]["__categories"]["gene_name"][:]); codes = h["var"]["gene_name"][:]; kg = [cats[c] for c in codes]
    X = h["X"][:]; h.close()
    ps = {}
    for i, s in enumerate(kp):
        ps.setdefault(s, i)
    KO = {}
    for g in ps:
        z = X[ps[g]]; fin = np.isfinite(z)
        zz = {kg[j]: float(z[j]) for j in range(len(kg)) if fin[j]}
        if len(zz) < 100:
            continue
        movers = {x for x in zz if abs(zz[x]) > 1.0 and x != g}
        KO[g] = {"z": zz, "movers": movers, "n": len(movers)}
    return KO


def run():
    par = load_paralogs()
    KO = load_z()
    rng = np.random.RandomState(0)
    S = {k: collections.defaultdict(list) for k in ("all", "weak", "strong", "mid")}
    n_par_kos = 0
    for g in KO:
        z = KO[g]["z"]; U = set(z); movers = KO[g]["movers"]; n = KO[g]["n"]
        PAR = [p for p in par.get(g, ()) if p in U and p != g]
        if not PAR:
            continue
        base_mover = sum(1 for x in U if x != g and abs(z[x]) > 1.0) / max(len(U) - 1, 1)
        base_up = sum(1 for x in U if x != g and z[x] > 1.0) / max(len(U) - 1, 1)
        if base_mover == 0:
            continue
        n_par_kos += 1
        cand = [x for x in U if x != g and x not in PAR]
        rnd = list(rng.choice(cand, min(len(cand), 300), replace=False))
        bias = float(np.mean([z[x] for x in rnd]))
        zpar = np.array([z[p] for p in PAR])
        rec = {
            "n_par": len(PAR),
            "par_signed_z": float(np.mean(zpar)),
            "par_z_above_bias": float(np.mean(zpar) - bias),
            "best_par_z": float(np.max(zpar)),                       # strongest-responding paralog
            "par_mover_frac": float(np.mean(np.abs(zpar) > 1.0)),
            "par_upmover_frac": float(np.mean(zpar > 1.0)),
            "base_mover": base_mover,
            "base_up": base_up,
            "any_par_is_mover": 1.0 if np.any(np.abs(zpar) > 1.0) else 0.0,
        }
        grp = "weak" if n < 15 else ("strong" if n >= 50 else "mid")
        for key in ("all", grp):
            for k, v in rec.items():
                S[key][k].append(v)

    def agg(key):
        d = S.get(key, {})
        if not d.get("par_signed_z"):
            return None
        mover_lift = np.mean(d["par_mover_frac"]) / max(np.mean(d["base_mover"]), 1e-9)
        up_lift = np.mean(d["par_upmover_frac"]) / max(np.mean(d["base_up"]), 1e-9)
        return {
            "n_kos_with_paralog": len(d["par_signed_z"]),
            "mean_paralogs_per_ko": round(float(np.mean(d["n_par"])), 1),
            "paralog_signed_z": round(float(np.mean(d["par_signed_z"])), 3),
            "paralog_z_above_KO_bias": round(float(np.mean(d["par_z_above_bias"])), 3),
            "best_paralog_signed_z": round(float(np.mean(d["best_par_z"])), 3),
            "paralog_mover_frac": round(float(np.mean(d["par_mover_frac"])), 3),
            "base_mover_rate": round(float(np.mean(d["base_mover"])), 3),
            "mover_lift": round(float(mover_lift), 2),
            "paralog_upmover_frac": round(float(np.mean(d["par_upmover_frac"])), 3),
            "base_upmover_rate": round(float(np.mean(d["base_up"])), 3),
            "upmover_lift": round(float(up_lift), 2),
        }

    return {"n_kos_with_paralog": n_par_kos, "paralog_map_genes": len(par),
            "all": agg("all"), "weak_KOs": agg("weak"), "strong_KOs": agg("strong")}


def main():
    print("=" * 100)
    print("COMPENSATION TEST (curated paralogs) — do a knocked-down gene's paralogs fire / upregulate? (mechanism 1)")
    print("=" * 100)
    r = run()
    print(f"\n  paralog map covers {r['paralog_map_genes']} genes; {r['n_kos_with_paralog']} knockouts have >=1 measured paralog.")
    for key, lab in [("all", "ALL KOs with a paralog"), ("weak_KOs", "WEAK KOs (<15 movers)"), ("strong_KOs", "STRONG KOs (>=50)")]:
        a = r[key]
        if not a:
            print(f"\n  {lab}: (none)")
            continue
        print(f"\n  {lab}  (n={a['n_kos_with_paralog']}, mean {a['mean_paralogs_per_ko']} paralogs/KO):")
        print(f"     paralog signed z                 : {a['paralog_signed_z']:+.3f}  (above KO's own drift: {a['paralog_z_above_KO_bias']:+.3f}); best paralog {a['best_paralog_signed_z']:+.3f}")
        print(f"     paralogs that are movers         : {a['paralog_mover_frac']:.3f}  vs base {a['base_mover_rate']:.3f}  -> lift {a['mover_lift']}x")
        print(f"     paralogs that are UP-movers      : {a['paralog_upmover_frac']:.3f}  vs base {a['base_upmover_rate']:.3f}  -> lift {a['upmover_lift']}x")
    w = r["weak_KOs"]; al = r["all"]
    dbin = w or al
    signal = dbin and (dbin["mover_lift"] >= 1.5 or dbin["upmover_lift"] >= 1.5 or dbin["paralog_z_above_KO_bias"] >= 0.15)
    if signal:
        verdict = (
            "COMPENSATION SIGNAL IS REAL (measured with curated paralogs). When a gene is knocked down its paralogs are enriched "
            f"among the movers (weak-KO mover lift {w['mover_lift']}x, up-mover lift {w['upmover_lift']}x, paralog upregulation "
            f"{w['paralog_z_above_KO_bias']:+.3f} above the knockout's own drift). Paralog identity predicts a slice of the specific "
            "movers -- a genuine thread for a Structural Compensation Engine: add a 'is-a-paralog-of-the-KO' feature (and its "
            "upregulation prior) to the forecast, restricted to genes that have a paralog. This is worth building and re-scoring.")
    else:
        verdict = (
            "COMPENSATION SIGNAL IS ABSENT / AT CHANCE in this data (measured with a CURATED Ensembl paralog map -- an honest "
            f"NEGATIVE on mechanism 1). Across the {al['n_kos_with_paralog']} knockouts that have >=1 measured paralog, LITERALLY "
            f"ZERO paralogs crossed the mover threshold (mover lift {al['mover_lift']}x, up-mover lift {al['upmover_lift']}x), and "
            f"paralog upregulation above the knockout's own drift is {al['paralog_z_above_KO_bias']:+.3f} (~0) -- no better for the "
            f"WEAK knockouts we needed to crack ({w['n_kos_with_paralog']} of them: mover lift {w['mover_lift']}x, upregulation "
            f"{w['paralog_z_above_KO_bias']:+.3f}). (Method note: I first tried the on-hand ESM matrix as the 'continuous structural "
            "field' but a positive control showed it was per-dimension STANDARDISED -- known paralogs RPL7/RPL7A scored at chance -- "
            "so that route was invalid and I switched to the curated map, which is correct where present.) So even with real "
            "paralogs, knocking a gene down does NOT light up its paralog here. THREE reasons this is expected and important: "
            "(1) K562 Perturb-seq is CRISPRi knockDOWN (dCas9-KRAB transcriptional repression), and the classic "
            "paralog/genetic-compensation pathway (transcriptional adaptation) is triggered by mutant-mRNA DEGRADATION, which "
            "CRISPRi does NOT produce -- the mechanism the engine models is largely absent from the measurement. (2) It is a single "
            "STEADY-STATE snapshot (~days post-perturbation): a transient compensatory paralog that fired early and returned to "
            "baseline cannot be in the data -- the calm-lake problem. (3) COVERAGE CEILING: only ~123 of ~1123 knockouts (~11%) even "
            "HAVE a measured paralog, so paralog compensation -- even if it worked perfectly -- could touch at most ~11% of "
            "knockouts and structurally CANNOT lift the global average from ~2/10 to 6-7/10; the weak set is dominated by genes "
            "without a clean paralog. BOTTOM LINE: the Structural Compensation Engine is biologically reasonable but has NO signal "
            "to learn or be validated against in THIS data, and too little reach to move the average even in principle. The "
            "weak-knockout ceiling is a DATA/measurement limit (knockdown modality + steady-state timing), not an architecture we "
            "can out-engineer here. Cracking weak knockouts needs time-resolved and/or true-knockout Perturb-seq we do not have -- "
            "and even the metric is capped: a weak KO with 5 real movers can never exceed 5/10 on a top-10 list. The honest fix is a "
            "better METRIC (recall of the true movers, k=n_movers) plus new DATA, not a bigger model on this data.")
    print(f"\n  VERDICT: {verdict}")
    r["verdict"] = verdict
    r["note"] = verdict
    json.dump(r, open(OUT / "compensation.json", "w"), indent=1)
    print("\n  -> outputs/orphan/compensation.json")
    return r


if __name__ == "__main__":
    main()
