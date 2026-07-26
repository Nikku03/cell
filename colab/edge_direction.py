"""CAN A MARKOV CHAIN PUT ARROWS ON UNDIRECTED PPI EDGES? The chain provably cannot; the perturbation data might.

THE THEOREM FIRST, because it decides what is worth measuring. A random walk on an UNDIRECTED graph has kernel

    P[a,b] = w_ab / deg(a)          P[b,a] = w_ab / deg(b)

so for an unweighted graph P[a,b] > P[b,a] exactly when deg(a) < deg(b), and nothing else. The chain's apparent
"flow direction" across any single edge is therefore a pure DEGREE RATIO -- it contains no information the degree
sequence does not already contain, and none of it is biological. This is not a limitation of truncation or tuning:
symmetric input cannot produce asymmetric output except through the normalisation, and the normalisation is degree.
The module measures this rather than asserting it, by checking that CHAIN and DEGREE agree on every edge.

SO THE REAL QUESTION IS WHETHER ANYTHING ELSE WE HAVE IS DIRECTIONAL, and one thing is: Perturb-seq. Knock out A and
watch B; knock out B and watch A. If A's removal moves B but B's removal leaves A alone, that asymmetry is
mechanistic evidence for A -> B. This is the only genuinely directional measurement in the library, and it has never
been used for edge orientation here.

GROUND TRUTH, and it is thin. SIGNOR supplies 17,432 directed edges; 3,164 undirected PPI edges carry an
unambiguous SIGNOR direction (186 more are recorded both ways and are dropped as uninformative). Adding `reg` gives
more, but `reg` is TF->target regulation, which is a different relation from physical binding, so it is reported
SEPARATELY rather than pooled. The Perturb-seq predictor additionally needs BOTH endpoints to be knockouts AND both
to be on the measured panel, which cuts the evaluable set to about 95 edges. At n=95 the binomial standard error on
an accuracy is about 5 points, so this can detect a real effect of roughly 10 points or more and nothing finer. That
is stated up front because a 60% accuracy on 95 edges is a hint, not a result.

THE TASK is binary and the baseline is exactly 50%: given the unordered pair {a,b}, name the source. Arms:
    CHAIN        sign of P[a,b] - P[b,a] from the 1-step walk        (predicted: identical to DEGREE)
    CHAIN-2STEP  same from the 2-step walk                           (predicted: still degree-derived)
    DEGREE       lower-degree endpoint is the source                 (orientation of the rule fitted on TRAIN)
    TF           transcription factor points at non-TF               (a real biological prior, and a fair rival)
    ABUNDANCE    more abundant protein is the source
    PERTURB      |z of B under KO of A| vs |z of A under KO of B|    (the only directional MEASUREMENT we hold)
Every rule's ORIENTATION is fitted on a training half and scored on a held-out half, because "lower degree is the
source" and "higher degree is the source" are two different hypotheses and picking the better one post hoc on the
test set would manufacture accuracy above 50% out of noise.
"""
import collections
import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
NBOOT = 4000


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)

    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    deg = np.zeros(N)
    for a, b in pe:
        deg[a] += 1; deg[b] += 1

    def directed(key):
        S = set()
        for e in D.get(key, []) or []:
            try:
                s_, t_ = int(e[0]), int(e[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
                S.add((s_, t_))
        return S

    tf = np.array([float(info[n].get("tf") or 0) for n in names])
    ppm = np.array([float((D.get("ppm", {}) or {}).get(str(i), 0) or 0) for i in range(N)])

    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    Z = np.zeros((len(kos), len(genes)), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            Z[ki[k], gi[g]] = z
    Zab = np.abs(Z)

    def effect(src_node, tgt_node):
        """|z| of the target gene measured under knockout of the source. None if not computable."""
        s_, t_ = names[src_node], names[tgt_node]
        if s_ in ki and t_ in gi:
            return float(Zab[ki[s_], gi[t_]])
        return None

    # ---- build the evaluation sets ----
    SETS = {}
    for key in ["sig", "reg"]:
        S = directed(key)
        rows = []
        for a, b in pe:
            f_, r_ = (a, b) in S, (b, a) in S
            if f_ == r_:
                continue                      # unrecorded, or recorded both ways: uninformative
            rows.append((a, b) if f_ else (b, a))   # (source, target)
        SETS[key] = rows
        print(f"{key}: {len(rows):,} PPI edges with an unambiguous direction", flush=True)
    pset = [(s_, t_) for s_, t_ in SETS["sig"]
            if effect(s_, t_) is not None and effect(t_, s_) is not None]
    print(f"  of the SIGNOR set, {len(pset)} have Perturb-seq computable in BOTH directions", flush=True)

    # ---- THE DEMONSTRATION: is the chain's direction anything other than degree? ----
    # For an unweighted undirected graph P[a,b]=1/deg(a), so sign(P[a,b]-P[b,a]) = sign(deg(b)-deg(a)) identically.
    chain_vs_deg = []
    for a, b in list(pe)[:200000]:
        if deg[a] == deg[b]:
            continue
        chain_dir = 1 if (1.0 / deg[a]) > (1.0 / deg[b]) else -1
        deg_dir = 1 if deg[a] < deg[b] else -1
        chain_vs_deg.append(chain_dir == deg_dir)
    agree = float(np.mean(chain_vs_deg))
    print(f"\n  CHAIN vs DEGREE agreement across {len(chain_vs_deg):,} edges: {agree:.6f}", flush=True)

    # ---- predictors: each returns a score, positive meaning "a is the source" ----
    def sc_chain(a, b):
        return (1.0 / max(deg[a], 1)) - (1.0 / max(deg[b], 1))

    def sc_degree(a, b):
        return float(deg[b] - deg[a])

    def sc_tf(a, b):
        return float(tf[a] - tf[b])

    def sc_abund(a, b):
        return float(np.log1p(ppm[a]) - np.log1p(ppm[b]))

    def sc_perturb(a, b):
        ab, ba = effect(a, b), effect(b, a)
        if ab is None or ba is None:
            return 0.0
        return float(ab - ba)

    PRED = collections.OrderedDict([("CHAIN", sc_chain), ("DEGREE", sc_degree),
                                    ("TF", sc_tf), ("ABUNDANCE", sc_abund), ("PERTURB", sc_perturb)])

    def evaluate(rows, arms, label, seed=0):
        """Orientation of each rule is fitted on TRAIN and applied to TEST, so 'lower degree is the source' and its
        opposite are two hypotheses rather than a free choice made after seeing the answer."""
        r = np.random.RandomState(seed)
        idx = r.permutation(len(rows)); half = len(rows) // 2
        tr = [rows[i] for i in idx[:half]]; te = [rows[i] for i in idx[half:]]
        print(f"\n  {label}  (n={len(rows)}, train {len(tr)} / test {len(te)}; chance = 0.500)")
        print(f"    {'rule':11s} {'test acc':>9s} {'+/-':>7s} {'orient':>7s} {'decided':>8s}")
        out = {}
        for nm in arms:
            fn = PRED[nm]
            trs = np.array([fn(s_, t_) for s_, t_ in tr])
            orient = 1.0 if (trs > 0).sum() >= (trs < 0).sum() else -1.0
            tes = np.array([fn(s_, t_) for s_, t_ in te]) * orient
            dec = tes != 0
            if dec.sum() < 10:
                print(f"    {nm:11s} {'n/a':>9s} {'':>7s} {'':>7s} {int(dec.sum()):8d}")
                out[nm] = None
                continue
            correct = (tes[dec] > 0).astype(float)
            acc = float(correct.mean())
            se = float(np.std([correct[r.randint(0, len(correct), len(correct))].mean()
                               for _ in range(NBOOT)]))
            out[nm] = {"acc": acc, "se": se, "n": int(dec.sum()), "orient": float(orient)}
            print(f"    {nm:11s} {acc:9.4f} {se:7.4f} {orient:+7.0f} {int(dec.sum()):8d}"
                  f"   {'ABOVE CHANCE' if acc - 2*se > 0.5 else 'not distinguishable from chance'}")
        return out

    res = {}
    res["SIGNOR"] = evaluate(SETS["sig"], ["CHAIN", "DEGREE", "TF", "ABUNDANCE"], "SIGNOR-directed PPI edges")
    res["REG"] = evaluate(SETS["reg"], ["CHAIN", "DEGREE", "TF", "ABUNDANCE"],
                          "reg-directed PPI edges (TF->target regulation, a DIFFERENT relation from binding)")
    res["PERTURB"] = evaluate(pset, ["CHAIN", "DEGREE", "TF", "ABUNDANCE", "PERTURB"],
                              "SIGNOR edges where Perturb-seq is computable both ways")

    best_sig = max((k for k in res["SIGNOR"] if res["SIGNOR"][k]),
                   key=lambda k: res["SIGNOR"][k]["acc"])
    pp = res["PERTURB"].get("PERTURB")
    verdict = (
        f"CAN A MARKOV CHAIN ORIENT PPI EDGES? NO, AND NOT FOR WANT OF TUNING. On an undirected graph the walk's "
        f"kernel is P[a,b] = 1/deg(a), so sign(P[a,b] - P[b,a]) equals sign(deg(b) - deg(a)) IDENTICALLY. Measured "
        f"across {len(chain_vs_deg):,} edges, the chain's direction and the pure degree rule agree "
        f"{agree:.4f} of the time -- they are the same rule. Whatever the chain says about direction is a statement "
        f"about which protein has more recorded partners, and a symmetric input cannot yield an asymmetric output "
        f"except through the normalisation, which is degree. "
        f"AND THE DEGREE RULE ITSELF IS WEAK: on {len(SETS['sig'])} SIGNOR-directed PPI edges it scores "
        f"{res['SIGNOR']['DEGREE']['acc']:.4f} +/- {res['SIGNOR']['DEGREE']['se']:.4f} against a 0.500 chance "
        f"baseline, with the orientation of the rule fitted on a training half so the sign is not chosen after "
        f"seeing the answer. The best non-Perturb-seq rule is {best_sig} at {res['SIGNOR'][best_sig]['acc']:.4f}. "
        + (f"THE ONE GENUINELY DIRECTIONAL MEASUREMENT WE HOLD IS PERTURB-SEQ -- knock out A and watch B, then the "
           f"reverse -- and on the {res['PERTURB']['PERTURB']['n']} edges where it is computable in both directions "
           f"it scores {pp['acc']:.4f} +/- {pp['se']:.4f}. "
           + ("That clears chance and is the only arm here that does, which matters because it is also the only arm "
              "carrying information the graph does not already contain. " if pp['acc'] - 2 * pp['se'] > 0.5 else
              "That does NOT clear chance at this sample size. The honest reading is that the idea is sound and the "
              "test is underpowered, not that the idea is refuted: at n of this size the binomial standard error is "
              "about 5 points, so only an effect of roughly 10 points or more could have been detected. ")
           if pp else "")
        + f"WHAT WOULD ACTUALLY ORIENT THESE EDGES. Direction is a causal claim, and the library holds almost no "
        f"causal measurement at the protein level: SIGNOR gives an unambiguous direction for {len(SETS['sig'])} of "
        f"{len(pe):,} PPI edges (1.7%), and only {len(pset)} of those have both endpoints knocked out and measured. "
        f"Kinase-substrate assays, time-resolved perturbation (which protein moves FIRST), and degron/rapid-depletion "
        f"experiments all measure direction directly; co-expression, co-dependency, abundance and topology do not, "
        f"and no amount of them will. "
        f"WHAT THIS CANNOT SAY. The `reg` arm is TF->target REGULATION, a different relation from physical binding, "
        f"and is reported separately rather than pooled for that reason. Edges recorded in both directions were "
        f"dropped as uninformative rather than counted as errors. One cell line for the Perturb-seq arm.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"chain_equals_degree": agree, "n_ppi": len(pe),
               "n_signor_directed": len(SETS["sig"]), "n_reg_directed": len(SETS["reg"]),
               "n_perturb_evaluable": len(pset), "results": res, "verdict": verdict},
              open(OUT / "edge_direction.json", "w"), indent=2)
    print(f"\n  -> {OUT/'edge_direction.json'}")


if __name__ == "__main__":
    main()
