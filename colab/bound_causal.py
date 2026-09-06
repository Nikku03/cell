"""BOUND x CAUSAL -- separating direct regulation from downstream consequence, and testing whether it matters.

THE GAP THIS FILLS. Block B built 537,376 causal edges from Replogle K562 Perturb-seq and showed they carry a
sign that survives transfer to RPE1 (+15.0 points over a swapped-source control). But its own recorded limit
was blunt: an edge there is a causal CONSEQUENCE of perturbing the source, not evidence the source acts on the
target. Knock down a TF and a thousand genes move; most of them move because something else moved first.

ENCODE has 776 released K562 TF ChIP-seq experiments over 526 distinct targets, and 396 of those targets also
carry a Replogle perturbation. That is enough to cross the two and ask which causal edges are direct:

                        bound in K562 ChIP        not bound
    causal (Perturb-seq)   DIRECT                 indirect / downstream
    not causal             bound, not regulating  --

The bottom-left cell is not a failure of the data. Occupancy without regulation is a well-measured phenomenon,
and this project already found it once: a GATA1 ChIP-vs-Perturb-seq comparison showed binding and regulation
are largely different sets.

THE TEST, PRESPECIFIED. A 2x2 by itself is a description; it becomes a claim only if directness predicts
something. The prediction, fixed in advance:

    DIRECT edges should transfer to RPE1 BETTER than indirect ones -- both in magnitude and, more
    importantly, in SIGN.

The reasoning is mechanistic rather than statistical: a direct edge is one protein acting at one promoter, and
that relationship survives a change of cell type if the TF and the site are both present. An indirect edge is
the end of a chain whose intermediate steps are themselves cell-type dependent, so it should be the more
fragile of the two. If directness does NOT predict better transfer, then ChIP occupancy is not a useful
admission criterion for this layer and that is worth knowing before it is used as one.

TWO CONFOUNDS THAT WOULD FAKE THE RESULT, both controlled:
  EFFECT SIZE. Direct edges may simply be bigger, and bigger effects transfer better regardless of mechanism.
    So the comparison is made within matched |z| strata, not pooled.
  PROMOTER ACCESSIBILITY. A gene with an open, busy promoter attracts many ChIP peaks and also responds to
    many perturbations. Bound-ness would then be a proxy for "busy promoter". The number of DISTINCT TFs bound
    at a gene is therefore matched as well, so `bound by this TF` is compared against `bound by that many TFs
    in general`.

ONE OUTPUT TYPE ACROSS ALL TFs. narrowPeak signalValue scales differ between conservative-IDR and
pseudoreplicated calls, so mixing them would make `bound` mean different things for different TFs and the
pooled 2x2 would be measuring the pipeline. One type is pinned and the TFs that lack it are dropped and
counted, rather than back-filled.
"""
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
API = "https://www.encodeproject.org"
DIR = SP / "k562chip"
REGL = SP / "cell_reg_perturb.json.gz"
PB = SP / "rpe1_pseudobulk.npz"
GWPS = SP / "gwps.h5ad"
PROMOTER_BP = 2_000          # promoter-proximal window around the TSS
OUTPUT_TYPE = "conservative IDR thresholded peaks"
Z_EDGE = 5.0


def api(path, tries=4):
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=240))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def download(url, path, tries=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=600) as fh, open(path, "wb") as o:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            return path
        except Exception:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def fetch(targets, report):
    """One K562 narrowPeak per TF, all of the same output type."""
    DIR.mkdir(parents=True, exist_ok=True)
    man_p = DIR / "manifest.json"
    man = json.load(open(man_p)) if man_p.exists() else {}
    todo = [t for t in targets if t not in man]
    report(f"  {len(man)} TFs cached, {len(todo)} to fetch")
    for n, t in enumerate(todo):
        try:
            d = api("/search/?type=Experiment&biosample_ontology.term_name=K562"
                    "&assay_title=TF+ChIP-seq&status=released&limit=20&field=accession"
                    f"&target.label={urllib.parse.quote(t)}")
            accs = sorted(x["accession"] for x in d.get("@graph", []))
            pick = None
            for acc in accs:
                fl = api(f"/search/?type=File&dataset=/experiments/{acc}/&status=released"
                         "&limit=300&frame=object")
                cands = [f for f in fl.get("@graph", [])
                         if f.get("assembly") == "GRCh38" and f.get("file_format") == "bed"
                         and f.get("file_format_type") == "narrowPeak"
                         and f.get("output_type") == OUTPUT_TYPE]
                if cands:
                    cands.sort(key=lambda f: (not f.get("preferred_default"), f["accession"]))
                    pick = (acc, cands[0])
                    break
            if pick is None:
                man[t] = None                      # lacks the pinned output type -- dropped, not back-filled
            else:
                acc, f = pick
                p = DIR / f"{t}.bed.gz"
                download(API + f["href"], p)
                man[t] = {"target": t, "experiment": acc, "file": f["accession"],
                          "output_type": f["output_type"], "size": p.stat().st_size,
                          "path": str(p)}
        except Exception as e:
            man[t] = None
            report(f"    {t}: {str(e)[:70]}")
        if (n + 1) % 25 == 0:
            json.dump(man, open(man_p, "w"))
            report(f"    fetched {n+1}/{len(todo)}")
    json.dump(man, open(man_p, "w"))
    got = {k: v for k, v in man.items() if v}
    kinds = {v["output_type"] for v in got.values()}
    if len(kinds) > 1:
        raise SystemExit(f"mixed peak output types across TFs: {sorted(kinds)}")
    report(f"  {len(got)}/{len(targets)} TFs have a GRCh38 '{OUTPUT_TYPE}' file "
           f"({len(targets)-len(got)} dropped for lacking it)")
    return got


def peak_pos(path):
    ch, po = [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            st, en = int(p[1]), int(p[2])
            off = int(p[9]) if len(p) > 9 and p[9] not in ("", "-1") else -1
            ch.append(p[0])
            po.append(st + off if off >= 0 else (st + en) // 2)
    return np.array(ch), np.array(po, dtype=np.int64)


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("BOUND x CAUSAL -- is a causal edge direct, and does directness predict transfer?")
    report("=" * 100)
    if not REGL.exists():
        raise SystemExit(f"absent: {REGL} -- run causal_reg.py first")

    from entity_registry import load as load_reg
    REG = load_reg()
    r = json.load(gzip.open(REGL, "rt"))
    edges = r["edges"]
    srcs = sorted({e["source"] for e in edges})
    report(f"  causal layer: {len(edges):,} edges from {len(srcs):,} perturbed genes")

    idx = json.load(open(SP / "k562_chip_index.json"))
    targets = [t for t in idx["both"]]
    report(f"  TFs with both K562 ChIP and a Replogle perturbation: {len(targets)}")
    man = fetch(targets, report)

    # ---- TSS coordinates from the registry ----
    d = json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))
    ens2pos = {}
    for g in d["genes"]:
        eid = REG.resolve("gene", "symbol", g["name"])
        e = REG.entities.get(eid) if eid else None
        if e and e.get("ensembl") and g.get("chrom") and g.get("tss") not in (None, ""):
            ens2pos[e["ensembl"]] = (g["chrom"], int(float(g["tss"])))
    del d
    report(f"  TSS coordinates for {len(ens2pos):,} genes")

    # ---- bound matrix: TF x gene, promoter-proximal ----
    genes = sorted({e["target_ensembl"] for e in edges if e["target_ensembl"] in ens2pos})
    gi = {g: i for i, g in enumerate(genes)}
    chrom = np.array([ens2pos[g][0] for g in genes])
    tss = np.array([ens2pos[g][1] for g in genes], dtype=np.int64)
    tfs = sorted(man)
    B = np.zeros((len(tfs), len(genes)), dtype=bool)
    for k, t in enumerate(tfs):
        pc, pp = peak_pos(man[t]["path"])
        for c in np.unique(chrom):
            m = pc == c
            if not m.any():
                continue
            pos = np.sort(pp[m])
            gk = np.where(chrom == c)[0]
            lo = np.searchsorted(pos, tss[gk] - PROMOTER_BP)
            hi = np.searchsorted(pos, tss[gk] + PROMOTER_BP)
            B[k, gk] = hi > lo
        if (k + 1) % 50 == 0:
            report(f"    bound matrix {k+1}/{len(tfs)} TFs")
    ntf_bound = B.sum(0)                       # how many distinct TFs sit at this promoter
    report(f"  bound matrix: {len(tfs)} TFs x {len(genes):,} genes, "
           f"{100*B.mean():.2f}% of cells bound; median TFs per promoter {np.median(ntf_bound):.0f}")

    # ---- the 2x2 ----
    ti = {t: k for k, t in enumerate(tfs)}
    ce = [e for e in edges if e["source"] in ti and e["target_ensembl"] in gi]
    report(f"\n  causal edges whose source is one of these TFs and target has a TSS: {len(ce):,}")
    bd = np.array([B[ti[e["source"]], gi[e["target_ensembl"]]] for e in ce])
    report(f"  DIRECT (bound + causal):     {int(bd.sum()):8,d}  ({100*bd.mean():.1f}% of causal edges)")
    report(f"  INDIRECT (causal, unbound):  {int((~bd).sum()):8,d}  ({100*(~bd).mean():.1f}%)")
    tot_bound = int(B[[ti[t] for t in tfs]].sum())
    report(f"  bound but NOT causal:        {tot_bound - int(bd.sum()):8,d}  "
           f"(occupancy without regulation)")
    base = B.mean()
    enr = bd.mean() / base if base > 0 else float("nan")
    p_enr = float(stats.binomtest(int(bd.sum()), len(bd), base).pvalue)
    report(f"  enrichment of binding at causal targets: {enr:.2f}x over the {100*base:.2f}% background "
           f"(p {p_enr:.3g})")

    R = {"n_tfs": len(tfs), "n_genes": len(genes), "promoter_bp": PROMOTER_BP,
         "output_type": OUTPUT_TYPE, "n_causal_edges_testable": len(ce),
         "n_direct": int(bd.sum()), "n_indirect": int((~bd).sum()),
         "bound_background": float(base), "binding_enrichment_at_causal": float(enr),
         "enrichment_p": p_enr}

    # ---- does directness predict transfer to RPE1? ----
    have = [e for e in ce if "rpe1_z" in e]
    report(f"\n  DIRECTNESS -> TRANSFER  ({len(have):,} of these edges carry an RPE1 measurement)")
    if len(have) < 500:
        report("    too few RPE1-tested edges to compare")
        R["transfer"] = None
    else:
        hb = np.array([B[ti[e["source"]], gi[e["target_ensembl"]]] for e in have])
        zk = np.array([abs(e["z"]) for e in have])
        zr = np.array([abs(e["rpe1_z"]) for e in have])
        ag = np.array([bool(e["rpe1_sign_agrees"]) for e in have])
        nb = np.array([ntf_bound[gi[e["target_ensembl"]]] for e in have])
        # MATCHED, because direct edges may simply be bigger and their promoters simply busier. The bins on
        # TFs-per-promoter are DECILES rather than quartiles: with a median of ~96 of 392 TFs at a promoter,
        # quartiles left a residual imbalance at p 0.037, which is precisely the confound this match exists
        # to remove. And the draw is repeated -- one balanced draw is one sample of the matched population,
        # and this project has twice read a single draw as the result.
        zq = np.digitize(zk, np.quantile(zk, np.linspace(.1, .9, 9)))
        bq = np.digitize(nb, np.quantile(nb, np.linspace(.1, .9, 9)))
        pool1, pool0 = {}, {}
        for i in range(len(have)):
            (pool1 if hb[i] else pool0).setdefault((zq[i], bq[i]), []).append(i)
        draws = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            keep = []
            for k, i1 in pool1.items():
                i0 = pool0.get(k, [])
                n = min(len(i1), len(i0))
                if n:
                    keep += list(rng.choice(i1, n, replace=False)) + list(rng.choice(i0, n, replace=False))
            keep = np.array(sorted(keep))
            if len(keep) < 200:
                continue
            s1, s0 = keep[hb[keep]], keep[~hb[keep]]
            draws.append({
                "n": int(len(keep)), "n_direct": int(len(s1)),
                "mag_direct": float(zr[s1].mean()), "mag_indirect": float(zr[s0].mean()),
                "sign_direct": float(ag[s1].mean()), "sign_indirect": float(ag[s0].mean()),
                "mag_p": float(stats.mannwhitneyu(zr[s1], zr[s0], alternative="two-sided")[1]),
                "sign_p": float(stats.fisher_exact([[int(ag[s1].sum()), int((~ag[s1]).sum())],
                                                    [int(ag[s0].sum()), int((~ag[s0]).sum())]])[1]),
                "res_z_p": float(stats.mannwhitneyu(zk[s1], zk[s0], alternative="two-sided")[1]),
                "res_nb_p": float(stats.mannwhitneyu(nb[s1], nb[s0], alternative="two-sided")[1])})
        if not draws:
            R["transfer"] = None
        else:
            def mn(k):
                return float(np.mean([d[k] for d in draws]))
            R["transfer"] = {
                "n_draws": len(draws), "n_matched": int(mn("n")), "n_direct": int(mn("n_direct")),
                "mag_direct": mn("mag_direct"), "mag_indirect": mn("mag_indirect"),
                "sign_direct": mn("sign_direct"), "sign_indirect": mn("sign_indirect"),
                "mag_p_median": float(np.median([d["mag_p"] for d in draws])),
                "sign_p_median": float(np.median([d["sign_p"] for d in draws])),
                "frac_draws_mag_p05": float(np.mean([d["mag_p"] < 0.05 for d in draws])),
                "frac_draws_sign_p05": float(np.mean([d["sign_p"] < 0.05 for d in draws])),
                "residual_z_p": mn("res_z_p"), "residual_nb_p": mn("res_nb_p")}
            t = R["transfer"]
            report(f"    matched on |z_K562| x TFs-per-promoter (deciles), {t['n_draws']} draws of "
                   f"~{t['n_matched']:,} edges ({t['n_direct']:,} direct)")
            report(f"      residual imbalance (mean p): |z| {t['residual_z_p']:.3g}, "
                   f"TFs-per-promoter {t['residual_nb_p']:.3g}")
            report(f"    {'class':12s} {'mean |z| RPE1':>14s} {'sign agrees':>12s}")
            report(f"    {'direct':12s} {t['mag_direct']:14.4f} {t['sign_direct']:12.4f}")
            report(f"    {'indirect':12s} {t['mag_indirect']:14.4f} {t['sign_indirect']:12.4f}")
            report(f"    magnitude: median p {t['mag_p_median']:.3g}, "
                   f"{100*t['frac_draws_mag_p05']:.0f}% of draws p<0.05")
            report(f"    sign:      median p {t['sign_p_median']:.3g}, "
                   f"{100*t['frac_draws_sign_p05']:.0f}% of draws p<0.05  "
                   f"(+{100*(t['sign_direct']-t['sign_indirect']):.1f} points)")

    # ---- verdict ----
    report("\n" + "=" * 100)
    t = R.get("transfer")
    if not t:
        v = (f"DESCRIPTIVE ONLY. {R['n_direct']:,} of {R['n_causal_edges_testable']:,} causal edges "
             f"({100*R['n_direct']/max(R['n_causal_edges_testable'],1):.1f}%) are directly bound, a "
             f"{enr:.2f}x enrichment over background, but too few carry an RPE1 measurement to test whether "
             f"directness predicts transfer.")
    else:
        dsig = t["sign_direct"] - t["sign_indirect"]
        dmag = t["mag_direct"] - t["mag_indirect"]
        # THE TWO HALVES OF THE PREDICTION ARE REPORTED SEPARATELY, because they came out differently and
        # a single verdict covering both would carry the weaker one on the stronger one's evidence.
        # "Reliably significant" means most draws, not one draw.
        magok = t["frac_draws_mag_p05"] >= 0.8 and dmag > 0
        signok = t["frac_draws_sign_p05"] >= 0.8 and dsig > 0.02
        head = (f"Only {100*R['n_direct']/max(R['n_causal_edges_testable'],1):.1f}% of causal edges are "
                f"bound at the promoter -- most transcriptional consequences of knocking down a TF are "
                f"downstream, not direct -- though binding is {enr:.2f}x enriched at causal targets "
                f"(p {p_enr:.3g}) against a {100*R['bound_background']:.1f}% background, which is itself "
                f"high: the median promoter here carries peaks from ~96 of 392 TFs.")
        if magok and signok:
            v = (f"DIRECTNESS IS A REAL ADMISSION CRITERION. {head} Matched on effect size and promoter "
                 f"busyness over {t['n_draws']} draws, DIRECT edges transfer to RPE1 better on BOTH halves "
                 f"of the signed prediction: magnitude {t['mag_direct']:.3f} vs {t['mag_indirect']:.3f} "
                 f"({100*t['frac_draws_mag_p05']:.0f}% of draws p<0.05) and sign "
                 f"{100*t['sign_direct']:.1f}% vs {100*t['sign_indirect']:.1f}% "
                 f"({100*t['frac_draws_sign_p05']:.0f}% of draws p<0.05).")
        elif magok:
            v = (f"DIRECTNESS PREDICTS MAGNITUDE; ON SIGN IT IS TOO WEAK TO BANK. {head} Matched over "
                 f"{t['n_draws']} draws, direct edges reach {t['mag_direct']:.3f} mean |z| in RPE1 against "
                 f"{t['mag_indirect']:.3f} ({100*t['frac_draws_mag_p05']:.0f}% of draws p<0.05, median p "
                 f"{t['mag_p_median']:.3g}) -- that half of the prediction holds. But sign agreement is "
                 f"{100*t['sign_direct']:.1f}% against {100*t['sign_indirect']:.1f}%, only "
                 f"{100*dsig:.1f} points, significant in just {100*t['frac_draws_sign_p05']:.0f}% of draws "
                 f"(median p {t['sign_p_median']:.3g}). So promoter occupancy marks edges whose EFFECT is "
                 f"more robust across cell types, not edges whose DIRECTION is more reliable -- and "
                 f"direction is the property block B showed the layer actually has. Binding earns a "
                 f"confidence tier on magnitude; it does not earn one on sign.")
        elif t["frac_draws_sign_p05"] >= 0.8:
            v = (f"DIRECTNESS PREDICTS SIGN BUT NOT MAGNITUDE. {head} Sign agreement is "
                 f"{100*t['sign_direct']:.1f}% vs {100*t['sign_indirect']:.1f}% "
                 f"({100*t['frac_draws_sign_p05']:.0f}% of draws p<0.05) while magnitude does not separate "
                 f"({t['mag_direct']:.3f} vs {t['mag_indirect']:.3f}).")
        else:
            v = (f"DIRECTNESS DOES NOT PREDICT TRANSFER. {head} Matched over {t['n_draws']} draws, bound "
                 f"causal edges transfer to RPE1 no better than unbound ones (magnitude "
                 f"{t['mag_direct']:.3f} vs {t['mag_indirect']:.3f}, {100*t['frac_draws_mag_p05']:.0f}% of "
                 f"draws p<0.05; sign {100*t['sign_direct']:.1f}% vs {100*t['sign_indirect']:.1f}%, "
                 f"{100*t['frac_draws_sign_p05']:.0f}%). The two assays agree about WHICH genes, but "
                 f"promoter occupancy is not a usable admission criterion here and the prespecified "
                 f"prediction fails.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "bound_causal.json", "w"), indent=1, default=float)
    report(f"\n  -> {OUT/'bound_causal.json'}")


if __name__ == "__main__":
    main()
