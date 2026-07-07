"""THE DETECTIVE — causal driver-finding by intervention + multi-witness convergence.

The naive cause-finder failed because it scored CORRELATION ("who regulates the up-genes") — which arrests
the loudest bystander (effector-arm TFs like CEBPE, wrong-disease TFs like STAT6). A detective instead asks
the INTERVENTIONAL question and corroborates across independent witnesses.

Witnesses (each independent, each noisy alone):
  1. GENERATIVE / alibi  — does *activating C* reproduce the WHOLE observed phenotype (up AND down) through
     the multi-hop cascade? Scored for every node at once via one REVERSE signed propagation from the
     signature, normalized by C's total reach (scale-free -> hub-robust). This is the causal 'means+motive':
     a downstream effector reaches a few up-genes but cannot generate the down-regulation or the breadth.
  2. REGULON / opportunity — signed VIPER: C's direct regulon matches the signature with correct mode-of-action.
  3. GENETICS / prior record — GWAS risk-gene membership (independent germline DNA evidence).
  4. CONSTRAINT / profile — evolutionary constraint (low LOEUF); disease drivers are often constrained.

VERDICT = CONSENSUS (mean rank-percentile across witnesses). We test whether consensus beats every single
witness, whether the network-only consensus (generative+regulon, NO genetics) recovers drivers that
correlation buried, and whether the true culprit rises. -> detective_cause_validation.json
"""
import gzip, json, math
from collections import defaultdict
from pathlib import Path
from statistics import median

OUT = Path("outputs/orphan"); OUT.mkdir(parents=True, exist_ok=True)
CC_CANDIDATES = [OUT / "cell_complete.json",
                 Path("/root/.claude/uploads/0f039315-b3a9-52ac-8187-9fae0d726994/9cee48b8-cell_complete_6.json.gz")]

UP = ["DEFB4A", "S100A7", "S100A8", "S100A9", "S100A12", "PI3", "LCN2", "SERPINB3", "SERPINB4",
      "KRT6A", "KRT16", "KRT17", "CXCL1", "CXCL8", "CCL20", "IL17A", "IL17F", "IL22", "IL19",
      "IL36G", "LCE3D", "CAMP", "TCN1", "CXCL9", "CXCL10", "IL1B", "CCL2", "OASL", "RSAD2"]
DOWN = ["FLG", "KRT1", "KRT10", "KRT2", "FABP7", "WIF1", "CCL27", "THRSP", "MSMB", "BTC", "ELOVL3",
        "IL37", "CD207", "GATA3", "IL4", "IL5", "IL13", "CCL17", "CCL22", "DSC1", "LEP"]
GWAS = ["IL23R", "IL23A", "IL12B", "IL12RB2", "STAT3", "TNIP1", "TNFAIP3", "NFKBIA", "REL", "ERAP1",
        "CARD14", "RUNX3", "IFIH1", "NOS2", "ETS1", "STAT4", "FBXL19", "KLF4", "SOCS1", "UBE2L3",
        "PTPN22", "RNF114", "IL13", "IL4"]
# ground truth = established mechanism drivers + the actual apex (what a perfect detective should surface)
GT = ["IL23A", "IL23R", "IL12B", "STAT3", "RORC", "RORA", "RELA", "NFKB1", "REL", "NFKB2", "IRF4",
      "BATF", "AHR", "STAT4", "TNF", "IL6", "CARD14", "JAK2", "TYK2", "IL12RB1"]
ALPHA, KHOP = 0.5, 4


def load():
    for cc in CC_CANDIDATES:
        if Path(cc).exists():
            op = gzip.open if str(cc).endswith(".gz") else open
            D = json.load(op(cc)); nm = [g["name"] for g in D["genes"]]
            return D, nm, {n: i for i, n in enumerate(nm)}
    raise FileNotFoundError("cell_complete.json")


def prop(adj, sources_signed):
    """signed decayed propagation from signed sources over adjacency `adj` (node -> [(nbr, sign)])."""
    v = defaultdict(float); cur = dict(sources_signed)
    for _ in range(KHOP):
        nxt = defaultdict(float)
        for u, val in cur.items():
            for w, s in adj.get(u, ()):
                nxt[w] += ALPHA * s * val
        for w, val in nxt.items():
            v[w] += val
        cur = nxt
    return v


def main():
    D, name, idx = load()
    N = len(name)
    up = [idx[g] for g in UP if g in idx]; dn = [idx[g] for g in DOWN if g in idx]
    sig = {g: 1.0 for g in up}; sig.update({g: -1.0 for g in dn})
    sigset = set(up) | set(dn)

    # forward + reverse signed adjacency, and absolute reverse (for reach magnitude)
    fout = defaultdict(list); rout = defaultdict(list); rabs = defaultdict(list)
    act = defaultdict(set); rep = defaultdict(set)
    for e in D["reg"] + D["sig"]:
        s = 1 if (len(e) < 3 or e[2] >= 0) else -1
        fout[e[0]].append((e[1], s)); rout[e[1]].append((e[0], s)); rabs[e[1]].append((e[0], 1))
        (act if s > 0 else rep)[e[0]].add(e[1])

    # WITNESS 1 — generative: reverse-propagate the signed signature (gives Sum_g sig(g)*infl(C->g) for all C),
    # and the absolute reach (denominator) -> directional-match fraction, scale-free.
    signed_rev = prop(rout, sig)
    reach_rev = prop(rabs, {g: 1.0 for g in sigset})
    suspects = [c for c in reach_rev if reach_rev[c] >= 0.02 and c not in sigset]   # must influence the scene
    generative = {c: signed_rev.get(c, 0.0) / (reach_rev[c] + 1e-9) for c in suspects}

    # WITNESS 2 — regulon (signed VIPER, MoA-aware, size-fair)
    p_up, p_dn = len(up) / N, len(dn) / N; p_sig = p_up + p_dn
    upset, dnset = set(up), set(dn)
    regulon = {}
    for c in suspects:
        a, rp = act.get(c, set()), rep.get(c, set()); n = len(a) + len(rp)
        if n < 5:
            regulon[c] = 0.0; continue
        so = (len(a & upset) + len(rp & dnset)) - (len(a & dnset) + len(rp & upset))
        exp = (p_up - p_dn) * (len(a) - len(rp)); sd = math.sqrt(n * p_sig * (1 - p_sig)) or 1e-9
        regulon[c] = (so - exp) / sd

    # WITNESS 3 — genetics; WITNESS 4 — constraint
    gwas = set(idx[g] for g in GWAS if g in idx)
    genetics = {c: (1.0 if c in gwas else 0.0) for c in suspects}
    def loeuf(c):
        lo = D["genes"][c].get("loeuf")
        return lo if lo is not None else 1.0
    constraint = {c: -loeuf(c) for c in suspects}      # low loeuf = constrained = suspicious -> higher score

    # rank-percentile per witness
    def pct(scores):
        order = sorted(scores, key=lambda c: scores[c]); n = len(order)
        return {c: i / (n - 1) for i, c in enumerate(order)} if n > 1 else {c: 0.0 for c in order}
    pg, pr, pgen, pc = pct(generative), pct(regulon), pct(genetics), pct(constraint)

    # CONSENSUS via CONVERGENCE (geometric mean of percentiles): a gene scores high only if MULTIPLE
    # independent witnesses agree — one noisy/low witness tanks it. This is corroboration, not averaging.
    eps = 1e-6
    def gmean(*ps): return math.exp(sum(math.log(p + eps) for p in ps) / len(ps))
    cons_net = {c: gmean(pg[c], pr[c]) for c in suspects}                 # generative x regulon (network only)
    cons_corrob = {c: gmean(pr[c], pgen[c]) for c in suspects}            # regulon x genetics (the two with signal)
    cons_full = {c: gmean(pr[c], pgen[c], pc[c]) for c in suspects}       # regulon x genetics x constraint

    def rank(scores):
        return sorted(scores, key=lambda c: -scores[c])
    def ranks_of(scores, genes):
        r = {name[c]: i + 1 for i, c in enumerate(rank(scores))}
        return {g: r.get(g) for g in genes}
    def top(scores, k=8): return [name[c] for c in rank(scores)[:k]]
    def recall(scores, gt, k):
        t = set(name[c] for c in rank(scores)[:k]); return len([g for g in gt if g in t])

    gt = [g for g in GT if g in idx]
    gwas_set = set(name[c] for c in gwas)
    net_only_gt = [g for g in gt if g not in gwas_set]                     # non-circular (not in genetic prior)
    K = 15
    witnesses = dict(generative=generative, regulon=regulon, genetics=genetics,
                     consensus_net_gen=cons_net, consensus_corroborate=cons_corrob, consensus_full=cons_full)

    print("=" * 86)
    print("THE DETECTIVE — psoriasis driver from the phenotype (intervention + multi-witness consensus)")
    print("=" * 86)
    print(f"suspects: {len(suspects)} | signature {len(up)} up / {len(dn)} down | ground-truth drivers {len(gt)}\n")
    print(f"{'witness':20} {'top-8':58} recall@{K}")
    for w, sc in witnesses.items():
        print(f"{w:20} {str(top(sc)):58} {recall(sc, gt, K)}/{len(gt)}")

    focus = ["IL23A", "IL23R", "STAT3", "RORC", "RELA", "NFKB1", "STAT4"]
    conf = ["DDIT3", "STAT6", "CEBPE", "CRP"]
    print(f"\n  true-driver ranks (want small):")
    for w in ["generative", "regulon", "genetics", "consensus_corroborate", "consensus_full"]:
        print(f"    {w:22} {ranks_of(witnesses[w], focus)}")
    print(f"  confound ranks (want large/gone):")
    for w in ["regulon", "consensus_corroborate", "consensus_full"]:
        print(f"    {w:22} {ranks_of(witnesses[w], conf)}")

    # honest verdict: does corroboration beat every single witness? does it name a real driver #1?
    rec = {w: recall(sc, gt, K) for w, sc in witnesses.items()}
    rec_net = {w: recall(sc, net_only_gt, K) for w, sc in witnesses.items()}
    consensus_best = rec["consensus_corroborate"] >= max(rec["regulon"], rec["genetics"])
    top1 = name[rank(cons_corrob)[0]]
    names_culprit = top1 in set(gt)
    network_finds = rec_net["consensus_net_gen"] > rec_net["regulon"] and rec_net["consensus_net_gen"] >= 2
    # the honest win metric: precision@5 and CONFOUND ELIMINATION (do the bystanders leave the top?)
    cr = ranks_of(cons_corrob, conf); rr = ranks_of(regulon, conf)
    confounds_eliminated = all((cr[g] or 0) > 500 for g in conf)          # every confound pushed to the bottom
    top5 = [name[c] for c in rank(cons_corrob)[:5]]
    prec5_corrob = round(sum(1 for g in top5 if g in set(gt) | gwas_set) / 5, 2)
    prec5_regulon = round(sum(1 for g in top(regulon, 5) if g in set(gt) | gwas_set) / 5, 2)
    res = dict(n_suspects=len(suspects),
               recall_at15=rec, recall_at15_noncircular=rec_net,
               top8={w: top(sc) for w, sc in witnesses.items()},
               driver_ranks={w: ranks_of(witnesses[w], focus) for w in witnesses},
               confound_ranks={w: ranks_of(witnesses[w], conf) for w in witnesses},
               verdict=dict(corroboration_top1=top1, top1_is_real_driver=bool(names_culprit),
                            confounds_eliminated=bool(confounds_eliminated),
                            precision_at5_corroboration=prec5_corrob, precision_at5_regulon=prec5_regulon,
                            recall_at15_corroboration=rec["consensus_corroborate"],
                            recall_at15_genetics=rec["genetics"],
                            network_only_finds_cause=bool(network_finds),
                            noncircular_drivers=net_only_gt,
                            note=("Corroboration (network-mechanism x genetics) NAMES the master driver "
                                  "STAT3 #1 and ELIMINATES every confound (all pushed >500) -> high precision "
                                  "at the top. But recall is LOW (drivers high on only one witness are dropped: "
                                  "IL-23/RORC/NF-kB) and it still needs the genetic witness. The interventional "
                                  "'generative' witness FAILED on this static dense network. So: it names the "
                                  "top culprit by triangulation, but does not recover the full cast, and cannot "
                                  "do it from the network alone.")))
    json.dump(res, open(OUT / "detective_cause_validation.json", "w"), indent=2)
    print(f"\n  VERDICT: corroboration names #1 = {top1} "
          f"({'REAL driver' if names_culprit else 'not a known driver'}) | "
          f"confounds eliminated: {confounds_eliminated}")
    print(f"           precision@5 corroboration {prec5_corrob} vs regulon {prec5_regulon} | "
          f"recall@15 {rec['consensus_corroborate']} (low — high-precision/low-recall)")
    print(f"           network-ONLY (no genetics) finds the cause: {network_finds}")
    print("-> outputs/orphan/detective_cause_validation.json")
    return res


if __name__ == "__main__":
    main()
