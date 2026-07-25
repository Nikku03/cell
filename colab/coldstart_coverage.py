"""coldstart_coverage -- the model is an annotation lookup; give it annotation for the genes it has none for.

THE DIAGNOSIS BEING ACTED ON. coldstart_diagnose showed the headline 0.35 is an average over two very different populations:
    genes in a curated complex   recall 0.428  (n=213)
    genes in NO curated complex  recall 0.143  (n=80)   <- a tide-null of 0.11 is 0.11
The one-hot complex block is the strongest feature in the model, and for a gene in no curated complex that entire block is zeros. The model
has learned where CORUM is, not how the cell works.

IS THAT FIXABLE, OR ARE THOSE GENES JUST IDIOSYNCRATIC? Measured before building anything: the BEST-POSSIBLE-COPY oracle -- the highest
recall obtainable by copying some other measured knockout's profile, chosen by peeking at the answer, so a CEILING and not a predictor --
reaches 0.698 for the no-complex genes against 0.730 for complex members. Their truth sets are the same size (median 30 vs 36) and the same
strength (median |z| 4.97 vs 4.96). A profile that predicts them EXISTS in the data; the model simply has no feature that points at it.
So 0.143 is a coverage failure, not intrinsic unpredictability.

THE FIX, in the same shape as the thing that already works. Curated complex membership is binary and covers 3,257 of 16,492 genes. Two
blocks are added, both of which every gene has a value for:
  GO TERMS         16,412 genes have GO annotation -- 5x the coverage of complexes, and a finer vocabulary. One-hot over terms occurring
                   in at least MIN_GO train sources.
  SOFT COMPLEX AFFINITY   instead of "is g a member of complex C" (zero for uncurated genes), "how similar is g to C's members", averaged
                   over the members, computed from three similarity sources already in cell_complete.json: PPI adjacency, co-expression
                   (16,374 genes) and co-dependency (13,451 genes). A gene in no complex still lands somewhere in complex space. This
                   extends the feature we KNOW works rather than replacing it with something unrelated.

Co-dependency is DepMap, a viability screen, so it is a perturbation measurement of the test gene even though it is not a transcriptome
one. The ablation from program_coldstart is kept: every configuration is also run without it.

THE METRIC THAT MATTERS HERE IS NOT THE HEADLINE. It is recall on the NO-COMPLEX stratum. A configuration that lifts the average by lifting
already-good complex members has done nothing about the actual problem, so both strata are always reported separately."""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TIDE_FRAC = 0.05
MINSRC = 20
MAXCOMPLEX = 200
K_RECALL = 50
KPROG = 50
RIDGE = 10.0
SEEDS = (0, 1, 2)
MIN_GO = 4          # a GO term must occur in this many TRAIN sources to become a column
MIN_CPLX_TR = 3     # a complex needs this many TRAIN members before its affinity column is meaningful

CONFIGS = ("baseline", "+GO", "+softaffinity", "+GO+softaffinity", "+GO+softaffinity-noDepMap")


def main():
    from sklearn.linear_model import Ridge
    from scipy.stats import wilcoxon
    H = Harness("K562")
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); Aall = np.abs(Z)

    tc = [int((np.abs(H.M[H.ki[k]]) >= 1.0).sum()) for k in H.kos
          if k in kidx and (np.abs(H.M[H.ki[k]]) > 0).any() and np.abs(H.M[H.ki[k]])[np.abs(H.M[H.ki[k]]) > 0].min() < 1.0]
    med = float(np.median(tc)); T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        e = abs(float(np.median((Aall >= t).sum(1))) - med)
        if best is None or e < best:
            best, T = e, float(t)
    per = (Aall >= T).sum(1)
    sources = [kos[i] for i in range(len(kos)) if per[i] >= MINSRC]
    si_of = {s: i for i, s in enumerate(sources)}

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    nidx = {n: i for i, n in enumerate(names)}
    info = {g["name"]: g for g in D["genes"]}
    comps = collections.defaultdict(set); cmem = collections.defaultdict(set)
    for cid, mem in D["complexes"].items():
        mm = sorted({names[x] for x in mem if isinstance(x, int) and x < N})
        if 2 <= len(mm) <= MAXCOMPLEX:
            cmem[cid] = set(mm)
            for g in mm:
                comps[g].add(cid)
    doms = {}
    try:
        for k, v in json.load(open(OUT / "domains.json")).get("domains", {}).items():
            try:
                doms[names[int(k)]] = set(str(x) for x in v)
            except (ValueError, IndexError):
                pass
    except Exception:
        pass
    BE = json.load(open(SP / "k562_baseline_expression.json"))
    base_panel = BE["measured_baseline"]; ctrl_expr = BE["source_control_expr"]

    # ---- GO: 16,412 genes, vs 3,257 with a curated complex ----
    GO = D.get("go", {})
    go_of = {}
    for g, d in GO.items():
        ts = set()
        for br in ("P", "C", "F"):
            for t in (d.get(br) or []):
                ts.add(f"{br}:{t}")
        if ts:
            go_of[g] = ts

    # ---- similarity sources for soft affinity, all keyed by gene-name ----
    def load_sim(key):
        out = collections.defaultdict(dict)
        for a, lst in (D.get(key) or {}).items():
            try:
                ga = names[int(a)]
            except (ValueError, IndexError):
                continue
            for pair in lst:
                try:
                    gb = names[int(pair[0])]; sc = float(pair[1])
                except (ValueError, IndexError, TypeError):
                    continue
                out[ga][gb] = max(out[ga].get(gb, 0.0), sc)
                out[gb][ga] = max(out[gb].get(ga, 0.0), sc)
        return out
    sim_coexpr = load_sim("coexpr")
    sim_codep = load_sim("codep")
    ppi_adj = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        try:
            a, b = names[int(e[0])], names[int(e[1])]
        except (ValueError, IndexError, TypeError):
            continue
        ppi_adj[a].add(b); ppi_adj[b].add(a)
    SIMS = [("ppi", None), ("coexpr", sim_coexpr), ("codep", sim_codep)]
    cov = {k: sum(1 for s in sources if (s in (ppi_adj if k == "ppi" else v))) for k, v in SIMS}
    print(f"K562: |z|>={T:.1f} | {len(sources)} sources "
          f"({sum(1 for s in sources if comps.get(s))} in a curated complex, {sum(1 for s in sources if not comps.get(s))} in none)")
    print(f"coverage over sources: GO {sum(1 for s in sources if s in go_of)}, complex {sum(1 for s in sources if comps.get(s))}, "
          + ", ".join(f"{k} {n}" for k, n in cov.items()))

    NUM = ["ess", "loeuf", "tf", "ppi", "npath", "ndis", "cpg", "pubs", "dark", "conf", "enh", "dep_frac"]
    DEPMAP = {"ess", "dep_frac"}

    def build_X(train, cfg):
        use_go = "GO" in cfg
        use_soft = "softaffinity" in cfg
        no_dm = "noDepMap" in cfg

        def vocab(getter, minn):
            c = collections.Counter()
            for s in train:
                c.update(getter(s))
            return sorted(k for k, v in c.items() if v >= minn)
        v_comp = vocab(lambda s: comps.get(s, ()), 2)
        v_dom = vocab(lambda s: doms.get(s, ()), 3)
        v_proc = vocab(lambda s: [(info.get(s, {}) or {}).get("proc") or "?"], 3)
        v_loc = vocab(lambda s: [(info.get(s, {}) or {}).get("comp") or "?"], 3)
        v_go = vocab(lambda s: go_of.get(s, ()), MIN_GO) if use_go else []
        # complexes with enough TRAIN members for a soft-affinity column to mean anything
        trset = set(train)
        v_soft = [c for c in sorted(cmem) if len(cmem[c] & trset) >= MIN_CPLX_TR] if use_soft else []
        sims = [(k, v) for k, v in SIMS if not (no_dm and k == "codep")]
        num = [k for k in NUM if not (no_dm and k in DEPMAP)]

        def affinity(g, memb, kind, tbl):
            if kind == "ppi":
                nb = ppi_adj.get(g)
                return len(nb & memb) / len(memb) if nb else 0.0
            row = tbl.get(g)
            if not row:
                return 0.0
            return float(np.mean([row.get(m, 0.0) for m in memb]))

        def feats(s):
            gi = info.get(s, {}) or {}
            x = []
            for k in num:
                try:
                    x.append(float(gi.get(k) or 0.0))
                except (TypeError, ValueError):
                    x.append(0.0)
            bp = base_panel.get(s)
            x.append(np.log1p(float(ctrl_expr.get(s, 0.0)))); x.append(1.0 if s in ctrl_expr else 0.0)
            x += [np.log1p(bp[0]), bp[1], bp[2], bp[3], 1.0] if bp else [0.0] * 5
            x += [1.0 if (gi.get("proc") or "?") == p else 0.0 for p in v_proc]
            x += [1.0 if (gi.get("comp") or "?") == p else 0.0 for p in v_loc]
            dl = doms.get(s, set()); x += [1.0 if d in dl else 0.0 for d in v_dom]
            cl = comps.get(s, set()); x += [1.0 if c in cl else 0.0 for c in v_comp]
            if use_go:
                gs = go_of.get(s, set()); x += [1.0 if t in gs else 0.0 for t in v_go]
            if use_soft:
                for c in v_soft:
                    memb = cmem[c] - {s}                 # never let a gene score affinity to ITSELF
                    for kind, tbl in sims:
                        x.append(affinity(s, memb, kind, tbl) if memb else 0.0)
            return x
        return np.array([feats(s) for s in sources], float)

    def recall(pred, keep, truth):
        top = keep[np.argsort(-np.abs(pred[keep]))[:K_RECALL]]
        return len({genes[q] for q in top} & truth) / min(len(truth), K_RECALL)

    res = collections.defaultdict(list)          # (cfg, stratum) -> recalls
    paired = collections.defaultdict(dict)       # cfg -> (seed,source) -> recall
    dims = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        ss = sorted(sources); rng.shuffle(ss)
        cut = int(0.7 * len(ss)); train, test = ss[:cut], ss[cut:]
        tr_rows = [kidx[s] for s in train]
        tide = (Aall[tr_rows] >= T).mean(0) >= TIDE_FRAC
        keep = np.where(~tide)[0]
        Mtr = Z[tr_rows][:, keep].astype(np.float64)
        mu = Mtr.mean(0); _, _, Vt = np.linalg.svd(Mtr - mu, full_matrices=False)
        Gk = Vt[:KPROG]; Ptr = (Mtr - mu) @ Gk.T
        truths = {}
        for s in test:
            t = {genes[q] for q in np.where((Aall[kidx[s]] >= T) & ~tide)[0] if genes[q] != s}
            if t:
                truths[s] = t
        tr_rank = np.zeros(len(genes)); tr_rank[keep] = (Aall[tr_rows][:, keep] >= T).mean(0)

        for cfg in CONFIGS:
            X = build_X(train, cfg)
            dims[cfg] = X.shape[1]
            Xtr = X[[si_of[s] for s in train]]; Xte = X[[si_of[s] for s in test]]
            m = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
            Pte = Ridge(alpha=RIDGE).fit((Xtr - m) / sd, Ptr).predict((Xte - m) / sd)
            for j, s in enumerate(test):
                if s not in truths:
                    continue
                pred = np.zeros(len(genes)); pred[keep] = mu + Pte[j] @ Gk
                v = recall(pred, keep, truths[s])
                res[(cfg, "all")].append(v)
                res[(cfg, "in_complex" if comps.get(s) else "no_complex")].append(v)
                paired[cfg][(seed, s)] = v
        for s in test:
            if s not in truths:
                continue
            v = recall(tr_rank, keep, truths[s])
            res[("[ref] tide-null", "all")].append(v)
            res[("[ref] tide-null", "in_complex" if comps.get(s) else "no_complex")].append(v)

    print(f"\n  {'configuration':32s} {'dims':>5s} {'ALL':>8s} {'in complex':>12s} {'NO complex':>12s}")
    tab = []
    for cfg in CONFIGS + ("[ref] tide-null",):
        a = res[(cfg, "all")]; ic = res[(cfg, "in_complex")]; nc = res[(cfg, "no_complex")]
        tab.append({"config": cfg, "dims": dims.get(cfg), "all": round(float(np.mean(a)), 4),
                    "in_complex": round(float(np.mean(ic)), 4), "no_complex": round(float(np.mean(nc)), 4),
                    "n_all": len(a), "n_no_complex": len(nc)})
        print(f"  {cfg:32s} {str(dims.get(cfg, '-')):>5s} {np.mean(a):>8.4f} {np.mean(ic):>12.4f} {np.mean(nc):>12.4f}")
    print(f"  (n = {len(res[('baseline','all')])} evaluations: {len(res[('baseline','in_complex')])} in a complex, "
          f"{len(res[('baseline','no_complex')])} in none)")

    print("\n  paired against baseline, on the stratum that matters (NO curated complex):")
    stats = {}
    base_keys = sorted(paired["baseline"])
    nc_keys = [k for k in base_keys if not comps.get(k[1])]
    for cfg in CONFIGS[1:]:
        b = np.array([paired["baseline"][k] for k in nc_keys])
        c = np.array([paired[cfg][k] for k in nc_keys])
        p = float(wilcoxon(c, b).pvalue) if np.any(c != b) else 1.0
        ba = np.array([paired["baseline"][k] for k in base_keys])
        ca = np.array([paired[cfg][k] for k in base_keys])
        pa = float(wilcoxon(ca, ba).pvalue) if np.any(ca != ba) else 1.0
        stats[cfg] = {"no_complex_delta": round(float(c.mean() - b.mean()), 4), "no_complex_p": p,
                      "all_delta": round(float(ca.mean() - ba.mean()), 4), "all_p": pa}
        print(f"  {cfg:32s} no-complex {c.mean()-b.mean():+.4f} (p={p:.3g}, n={len(nc_keys)})   "
              f"all {ca.mean()-ba.mean():+.4f} (p={pa:.3g})")

    bestcfg = max(CONFIGS[1:], key=lambda c: stats[c]["no_complex_delta"])
    d = stats[bestcfg]
    fixed = bool(d["no_complex_delta"] > 0 and d["no_complex_p"] < 0.05)
    nc_base = float(np.mean(res[("baseline", "no_complex")]))
    nc_best = float(np.mean(res[(bestcfg, "no_complex")]))
    ic_best = float(np.mean(res[(bestcfg, "in_complex")]))

    verdict = (
        "COLD-START COVERAGE (coldstart_coverage.py): acting on the finding that the cold-start model is an annotation lookup -- it scored "
        f"0.428 on genes inside a curated complex and {nc_base:.3f} on genes in none, against a tide-null of "
        f"{np.mean(res[('[ref] tide-null','no_complex')]):.3f}. First established that this is fixable: the best-possible-copy oracle (a "
        "CEILING, since the partner is chosen by peeking at the answer) reaches 0.698 on the no-complex genes versus 0.730 on complex "
        "members, with truth sets of the same size and strength, so a profile that predicts them exists in the data and the model merely "
        "lacks a feature pointing at it. "
        "Two dense blocks were added, both defined for essentially every gene: GO terms (16,412 genes annotated, against 3,257 with a "
        "curated complex) and SOFT COMPLEX AFFINITY, which replaces the binary 'is g in complex C' with 'how similar is g to C's members' "
        "averaged over PPI adjacency, co-expression and co-dependency -- so a gene in no curated complex still lands somewhere in complex "
        "space. A gene never scores affinity to itself. "
        f"RESULT on the stratum that matters, genes in NO curated complex: baseline {nc_base:.4f} -> {nc_best:.4f} with {bestcfg} "
        f"({d['no_complex_delta']:+.4f}, paired p={d['no_complex_p']:.3g}, n={len(nc_keys)}). Overall {d['all_delta']:+.4f} "
        f"(p={d['all_p']:.3g}); complex members {ic_best:.4f}. "
        + ("THE COVERAGE GAP NARROWS: giving uncurated genes a graded position instead of an all-zero annotation block moves precisely the "
           "population that was failing, which supports the diagnosis that the ceiling was curation coverage rather than model capacity. "
           if fixed else
           "THE COVERAGE FIX DOES NOT WORK: dense GO and soft affinity do not significantly help the genes that have no curated complex, so "
           "either these features do not carry the grouping information that curated complexes carry, or the failure on those genes has a "
           "cause other than missing annotation. The oracle says the information exists, so this is a statement about these particular "
           "features, not about the genes. ")
        + "The DepMap-free variant is reported because co-dependency is a viability screen, itself a perturbation of the test gene. Both "
        "strata are always shown separately: a configuration that lifts the average by helping already-good complex members would not have "
        "addressed the problem. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"threshold": T, "table": tab, "paired_vs_baseline": stats, "best_config": bestcfg,
               "coverage_fixed": fixed, "oracle_no_complex": 0.6982, "oracle_in_complex": 0.7300,
               "verdict": verdict, "note": verdict}, open(OUT / "coldstart_coverage.json", "w"), indent=1)
    print("\n  -> outputs/orphan/coldstart_coverage.json")


if __name__ == "__main__":
    main()
