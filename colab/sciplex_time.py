"""sciplex_time -- the closest available test of the TIME-RESOLVED / transient hypothesis. No sub-24h genetic perturb-seq exists
as a clean h5ad (the transient-compensation wall's ideal data would need GEO MTX assembly), but Sci-Plex 3 has TWO timepoints:
24h (acute drug response) and 72h (adapted). So we can ask, honestly: does the transcriptional response RESHAPE between 24h and
72h, or is it the same program held steady? A large reshape = time-dependent dynamics (transient movers early, adaptation movers
late) exist and matter -- evidence that a steady-state snapshot misses real structure (supporting the transient hypothesis, though
at 24h/72h not 2-12h). A stable program = the response is largely set by 24h.

K562 only; genes restricted to the ~8.4k shared with our K562 KO set; stream the CSR by cell-blocks; pool doses to get enough
cells per (drug, timepoint); delta vs the time-matched vehicle control.
"""
import json, collections
from pathlib import Path
import numpy as np
import h5py
OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _dec(a): return [x.decode() if isinstance(x, bytes) else x for x in a]


def k562_genes():
    h = h5py.File(f"{SP}/k562.h5ad", "r")
    cats = _dec(h["var"]["__categories"]["gene_name"][:]); g = set(cats[c] for c in h["var"]["gene_name"][:]); h.close()
    return g


def run():
    from scipy.stats import spearmanr
    keep = k562_genes()
    h = h5py.File(f"{SP}/sciplex.h5ad", "r")
    genes = _dec(h["var"]["gene_symbol"][:]); n_g = len(genes)
    gmap = np.full(n_g, -1, np.int64); kept = []
    for j, gn in enumerate(genes):
        if gn in keep and gmap[j] == -1:
            gmap[j] = len(kept); kept.append(gn)
    K = len(kept)
    pc = h["obs"]["perturbation"]; pcats = _dec(pc["categories"][:]); pcodes = pc["codes"][:]
    cl = h["obs"]["cell_line"]; clcats = _dec(cl["categories"][:]); clcodes = cl["codes"][:]
    tm = h["obs"]["time"][:]; dv = h["obs"]["dose_value"][:]
    line = clcats.index("A549")  # A549 is the ONLY Sci-Plex line with BOTH 24h and 72h (K562/MCF7 are 24h-only)
    # group = (drug, time); control per time. only the chosen line.
    gid = {}; ng = 0
    group_of = np.full(len(pcodes), -1, np.int64)
    for i in range(len(pcodes)):
        if clcodes[i] != line or tm[i] not in (24.0, 72.0):
            continue
        drug = pcats[pcodes[i]]; t = int(tm[i])
        if "control" in drug.lower() or dv[i] == 0.0:
            key = ("__ctrl__", t)
        else:
            key = (drug, t)
        if key not in gid:
            gid[key] = ng; ng += 1
        group_of[i] = gid[key]
    indptr = h["X"]["indptr"][:]; data_ds = h["X"]["data"]; ind_ds = h["X"]["indices"]
    sums = np.zeros(ng * K, np.float64); gtot = np.zeros(ng, np.float64); ncell = np.zeros(ng, np.int64)
    B = 20000; N = len(pcodes)
    for i0 in range(0, N, B):
        i1 = min(i0 + B, N); a = int(indptr[i0]); b = int(indptr[i1])
        if b <= a:
            continue
        d = data_ds[a:b].astype(np.float64); cols = ind_ds[a:b]
        rows = np.repeat(np.arange(i0, i1), np.diff(indptr[i0:i1 + 1]))
        gc = group_of[rows]; ngid = gmap[cols]
        m = (gc >= 0) & (ngid >= 0)
        if m.any():
            sums += np.bincount(gc[m] * K + ngid[m], weights=d[m], minlength=ng * K)
            gtot += np.bincount(gc[m], weights=d[m], minlength=ng)
    for i in range(N):
        if group_of[i] >= 0:
            ncell[group_of[i]] += 1
    h.close()
    sums = sums.reshape(ng, K)
    L = np.log1p(sums / np.maximum(gtot[:, None], 1.0) * 1e6)
    def prof(drug, t):
        g = gid.get((drug, t)); c = gid.get(("__ctrl__", t))
        if g is None or c is None or ncell[g] < 30:
            return None
        return L[g] - L[c]
    drugs = sorted(set(k[0] for k in gid if k[0] != "__ctrl__"))
    rows = []
    for drug in drugs:
        d24 = prof(drug, 24); d72 = prof(drug, 72)
        if d24 is None or d72 is None:
            continue
        # z-scored movers per timepoint (z across kept genes within this profile)
        def movers(d):
            z = (d - d.mean()) / (d.std() + 1e-9)
            return set(np.where(np.abs(z) > 2.0)[0])
        m24 = movers(d24); m72 = movers(d72)
        if len(m24) < 5 and len(m72) < 5:
            continue
        uni = sorted(m24 | m72)
        sp = spearmanr(d24[uni], d72[uni])[0] if len(uni) > 5 else np.nan
        inter = len(m24 & m72); only24 = len(m24 - m72); only72 = len(m72 - m24)
        jacc = inter / max(len(m24 | m72), 1)
        rows.append({"drug": drug[:36], "n24": len(m24), "n72": len(m72), "spearman_24_72": float(sp) if sp == sp else 0.0,
                     "jaccard": jacc, "transient_only24": only24, "adaptive_only72": only72})
    sp = [r["spearman_24_72"] for r in rows]; jac = [r["jaccard"] for r in rows]
    tr = np.mean([r["transient_only24"] / max(r["n24"], 1) for r in rows])
    ad = np.mean([r["adaptive_only72"] / max(r["n72"], 1) for r in rows])
    return {"n_drugs": len(rows),
            "profile_spearman_24_vs_72_median": round(float(np.median(sp)), 3),
            "mover_jaccard_24_72_median": round(float(np.median(jac)), 3),
            "frac_movers_transient_only24_mean": round(float(tr), 3),
            "frac_movers_adaptive_only72_mean": round(float(ad), 3),
            "example_big_reshape": sorted(rows, key=lambda r: r["spearman_24_72"])[:6],
            "example_stable": sorted(rows, key=lambda r: -r["spearman_24_72"])[:6]}


def main():
    print("=" * 100)
    print("SCI-PLEX TIME — does the response reshape 24h->72h? (A549; closest available time test)")
    print("=" * 100)
    r = run()
    print(f"\n  {r['n_drugs']} A549 drugs with both 24h and 72h.")
    print(f"  profile Spearman(24h, 72h) median : {r['profile_spearman_24_vs_72_median']}")
    print(f"  mover-set Jaccard(24h, 72h) median : {r['mover_jaccard_24_72_median']}")
    print(f"  fraction of movers TRANSIENT (24h-only, gone by 72h) : {r['frac_movers_transient_only24_mean']:.0%}")
    print(f"  fraction of movers ADAPTIVE   (72h-only, new by 72h)  : {r['frac_movers_adaptive_only72_mean']:.0%}")
    sp = r["profile_spearman_24_vs_72_median"]; jac = r["mover_jaccard_24_72_median"]
    tr = r["frac_movers_transient_only24_mean"]; ad = r["frac_movers_adaptive_only72_mean"]
    reshapes = sp < 0.5 or jac < 0.4
    if reshapes:
        verdict = (
            "TIME MATTERS -- the response RESHAPES between 24h and 72h (measured; the closest available time-resolved test). Across "
            f"{r['n_drugs']} A549 drugs, the 24h and 72h response profiles correlate at only Spearman {sp} and the mover sets overlap "
            f"at Jaccard {jac}: {tr:.0%} of 24h movers are TRANSIENT (gone by 72h) and {ad:.0%} of 72h movers are ADAPTIVE (absent at "
            "24h). So a substantial part of the response is time-specific -- a single steady-state snapshot genuinely MISSES real "
            "structure, which is direct support for the transient hypothesis behind the compensation wall (buffering/transients that "
            "a late single timepoint cannot contain). CAVEAT stated plainly: this is 24h vs 72h CHEMICAL perturbation, not the 2-12h "
            "GENETIC window where paralog compensation is expected -- so it demonstrates that time-resolution matters and that "
            "transient/adaptive movers are real and large, but it is a proxy, not the exact experiment. The exact test (sub-24h "
            "genetic perturb-seq) remains unavailable as a clean dataset and is the real highest-value next fetch (GEO).")
    else:
        verdict = (
            f"THE RESPONSE IS LARGELY SET BY 24h -- 24h and 72h profiles are similar (Spearman {sp}, mover Jaccard {jac}), with only "
            f"{tr:.0%} transient and {ad:.0%} adaptive movers. At least for chemical perturbation between 24h and 72h, the program is "
            "mostly stable, so late-timepoint dynamics are modest. This does NOT settle the transient-compensation question (which is "
            "about the 2-12h window, not 24-72h), but it shows the 24->72h reshape is limited. The exact sub-24h genetic test "
            "remains the real unmet need.")
    print(f"\n  VERDICT: {verdict}")
    r["verdict"] = verdict; r["note"] = verdict
    json.dump(r, open(OUT / "sciplex_time.json", "w"), indent=1)
    print("\n  -> outputs/orphan/sciplex_time.json")
    return r


if __name__ == "__main__":
    main()
