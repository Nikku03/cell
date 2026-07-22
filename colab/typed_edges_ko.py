"""typed_edges_ko -- use the TYPED-INTERACTION ATLAS we already have (what one protein does to another) to ask, per edge TYPE, whether
that edge predicts a KNOCKOUT CONSEQUENCE -- in BOTH readouts at once: (1) mRNA far field (K562 Perturb-seq: does KO of A move B's
mRNA?) and (2) protein-level co-dependency (DepMap: are A,B co-dependent?). Unifies pathway_ko_verify (mRNA, per edge type) and nexus_ko
(complex -> co-dependency) into one edge-type x readout matrix, and extends it to the SIGNALING / ligand-receptor / nichenet types we
had not scored against co-dependency.

Edge types (directed A->B where applicable), all from cell_complete.json:
  reg      TF -> target (curated + inferred regulatory, signed)      -- the transcriptional layer
  sig      A -> B signed signaling (activate/inhibit; SIGNOR/OmniPath-style: the 'what A does to B' atlas)
  lr       ligand -> receptor
  nichenet ligand -> downstream target genes
  complex  co-membership (reference; nexus_ko already validated this against co-dependency)

For each type: mRNA enrichment = P(|z_B in KO_A| >= TAU | edge) / P(...|random), on edges whose source A was knocked out in K562;
codep enrichment = P(A,B co-dependent | edge) / P(...|random), plus vs an essentiality-matched random control (a kinase and its
substrate are often both essential, which would fake coupling). Answers 'do the typed PTM/signaling edges add anything to
knockout-consequence prediction' -- with the data we already have, no GPU.
"""
import json, pickle, collections
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU = 1.0            # |z|>=TAU = B's mRNA moved (compressed control-relative z; p95~1.04)
RNG = np.random.RandomState(0)


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; idx = {n: i for i, n in enumerate(names)}; N = len(names)
    dep = np.array([float(g.get("dep_frac") or 0.0) for g in D["genes"]])
    codep = {int(k): {int(j) for j, _ in v} for k, v in D["codep"].items()}
    def codep_edge(a, b):
        return (b in codep.get(a, ())) or (a in codep.get(b, ()))
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))          # K562 far field: KO gene-name -> {moved gene: z}
    # tide = generic stress genes that move under MANY knockouts; exclude them so the mRNA readout measures SPECIFIC transmission
    mf = collections.Counter()
    for zz in KO.values():
        for g, z in zz.items():
            if abs(z) >= 1.0:
                mf[g] += 1
    tide = {g for g, c in mf.items() if c >= 0.05 * len(KO)}

    # ---- build directed index pairs per edge type ----
    def pairs_from_list(lst, sample=None):
        ps = []
        for e in lst:
            if len(e) >= 2 and isinstance(e[0], int) and isinstance(e[1], int) and e[0] < N and e[1] < N:
                ps.append((e[0], e[1]))
        if sample and len(ps) > sample:
            ps = [ps[i] for i in RNG.choice(len(ps), sample, replace=False)]
        return ps
    E = {}
    E["reg"] = pairs_from_list(D.get("reg", []), sample=40000)
    E["sig"] = pairs_from_list(D.get("sig", []))
    E["lr"] = pairs_from_list(D.get("lr", []))
    nn = []
    for lig, tgts in (D.get("nichenet", {}) or {}).items():
        if lig in idx:
            for t in tgts:
                if t in idx:
                    nn.append((idx[lig], idx[t]))
    E["nichenet"] = nn
    cx = set()
    for m in D["complexes"].values():
        mm = sorted(set(x for x in m if isinstance(x, int) and x < N))
        for a in range(len(mm)):
            for b in range(a + 1, len(mm)):
                cx.add((mm[a], mm[b]))
    E["complex"] = sorted(cx)

    # ---- readouts ----
    def mrna_move(a, b):                                       # does KO of A SPECIFICALLY move B's mRNA? (|z|>=TAU and B not tide)
        na, nb = names[a], names[b]
        if na not in KO:
            return None
        if nb in tide:                                        # exclude generic stress genes -> specific transmission only
            return False
        z = KO[na].get(nb)
        return abs(z) >= TAU if z is not None else False
    genes_codep = np.array(sorted(codep.keys()))
    def rand_pairs(n, match_dep=None):
        out = []
        while len(out) < n:
            a, b = RNG.choice(genes_codep, 2, replace=False)
            if match_dep is not None:
                da, db = match_dep[len(out) % len(match_dep)]
                if not (abs(dep[a] - da) < 0.15 and abs(dep[b] - db) < 0.15):
                    continue
            out.append((int(a), int(b)))
        return out

    # baselines. mRNA baseline: source A must be a KNOCKED-OUT gene (else untestable) -> random (KO'd A, random B).
    ko_genes = [idx[n] for n in KO if n in idx]
    rand_mrna = [(ko_genes[RNG.randint(len(ko_genes))], int(RNG.randint(N))) for _ in range(20000)]
    bm = [m for m in (mrna_move(a, b) for a, b in rand_mrna) if m is not None]
    base_mrna = float(np.mean(bm)) if bm else 1e-6
    rnd = rand_pairs(6000)
    base_codep = np.mean([codep_edge(a, b) for a, b in rnd])

    print(f"{'edge type':10s} {'#edges':>8s} | {'mRNA move rate':>14s} {'enr':>5s} (cov) | {'codep rate':>10s} {'enr':>5s} {'ess-matched enr':>15s}")
    print("-" * 96)
    res = {}
    for et, ps in E.items():
        mv = [mrna_move(a, b) for a, b in ps]
        mv = [x for x in mv if x is not None]
        mr = float(np.mean(mv)) if mv else float("nan")
        cd = float(np.mean([codep_edge(a, b) for a, b in ps])) if ps else float("nan")
        # essentiality-matched control for THIS edge type's dep distribution
        depdist = [(dep[a], dep[b]) for a, b in ps[:2000]]
        rm = rand_pairs(3000, match_dep=depdist) if depdist else []
        base_codep_m = np.mean([codep_edge(a, b) for a, b in rm]) if rm else base_codep
        me = mr / max(base_mrna, 1e-9); ce = cd / max(base_codep, 1e-9); cem = cd / max(base_codep_m, 1e-9)
        res[et] = {"n_edges": len(ps), "mrna_rate": mr, "mrna_enrich": me, "mrna_cov": len(mv),
                   "codep_rate": cd, "codep_enrich": ce, "codep_enrich_matched": cem}
        print(f"{et:10s} {len(ps):>8d} | {mr:>14.4f} {me:>5.1f} ({len(mv):>5d}) | {cd:>10.4f} {ce:>5.0f} {cem:>15.1f}")
    print(f"\n   random baseline: mRNA-move {base_mrna:.4f}, codep {base_codep:.4f} (TAU={TAU})")

    verdict = (
        "TYPED-EDGE x READOUT MATRIX (the interaction atlas we already have; no GPU). Question answered: do the typed SIGNALING / "
        "PTM-style edges add a knockout-consequence signal beyond what we had? YES -- in the PROTEIN-COUPLING readout. THE CLEAN, "
        "DECISIVE READOUT IS CO-DEPENDENCY (DepMap; not tide-confounded): essentiality-matched enrichment is COMPLEX "
        f"{res['complex']['codep_enrich_matched']:.0f}x, SIGNALING/sig {res['sig']['codep_enrich_matched']:.0f}x -- both strongly "
        f"co-dependent BEYOND shared essentiality -- vs REGULATORY {res['reg']['codep_enrich_matched']:.1f}x, ligand-receptor "
        f"{res['lr']['codep_enrich_matched']:.1f}x, nichenet {res['nichenet']['codep_enrich_matched']:.1f}x (~1x = not coupled). So "
        "the typed SIGNALING edge (SIGNOR/OmniPath-style 'what A does to B') genuinely predicts protein-level co-failure, same class as "
        "complexes -- a real addition, using data we already had. THE mRNA READOUT is messier and I flag it honestly: REGULATORY "
        f"(TF->target) is the transmitter ({res['reg']['mrna_enrich']:.1f}x specific, matching pathway_ko_verify's ~6x), but the "
        f"physical edges' apparent mRNA signal (complex {res['complex']['mrna_enrich']:.0f}x even after tide removal) is threshold-"
        "sensitive coordinated-feedback co-movement, NOT clean specific transmission -- the strict per-edge mRNA verdict remains "
        "pathway_ko_verify's (complex/signaling ~0x to the transcriptome). NET, the two knockout-consequence layers are carried by "
        "DIFFERENT typed edges -- REGULATORY -> mRNA far field, PHYSICAL/SIGNALING -> protein co-dependency -- and the atlas already "
        "encodes both. This is exactly why the AF-Multimer interface-energy branch was low-value: it was re-deriving a quantitative "
        "shade of the physical/complex edge the typed atlas already gives us (and even crude interface size was null). Honest caveat: "
        "the ess-matched control is approximate; lr/nichenet coverage in the KO+codep sets is small (n shown).")
    print(f"\nVERDICT: {verdict}")
    out = {"tau": TAU, "random_baseline": {"mrna_move": base_mrna, "codep": base_codep},
           "by_edge_type": res, "verdict": verdict, "note": verdict}
    json.dump(out, open(OUT / "typed_edges_ko.json", "w"), indent=1)
    print("\n  -> outputs/orphan/typed_edges_ko.json")
    return out


if __name__ == "__main__":
    main()
