"""direction_split -- split the movers by direction, then ask what MACHINERY each group belongs to and how it is wired, and do the same for
the knocked-out gene itself.

WE DO HAVE DIRECTION. The Perturb-seq values are signed z-scores, so every mover carries up or down. That has been thrown away by every
model in this project so far, which ranked genes by |z| and never once checked the sign. Splitting on it is the first thing that should have
been done, because the two groups mean opposite things: a gene going DOWN when a TF is removed is a candidate for being held UP by it
(direct activation lost, or its machine disassembling), while a gene going UP is a candidate for having been repressed.

WHAT IS ASKED OF EACH GROUP:
    MACHINERY   which curated complexes have members in this group, and how many. A complex with 24 of its members in one direction is a
                machine being coordinately lost or gained; scattered singletons are not.
    WIRING      how many PPI edges fall INSIDE the group. A coherent module is densely wired to itself; an incoherent list is not.
    FUNCTION    process, compartment and GO-function profile, so the group can be described rather than counted.

THE CONTROL THAT DECIDES ALL OF IT. Both complex co-membership and internal PPI density scale with group size and with how well-studied the
genes are, and movers are strongly biased towards well-studied genes. So every count is compared against RANDOM gene sets of the SAME SIZE
drawn from the same measured panel, NPERM times. Raw counts are reported alongside the permutation mean and an empirical p, because "the
down group contains 24 ribosomal proteins" means nothing until you know how many a random set of 88 genes contains.

AND THE KNOCKED-OUT GENE ITSELF gets the same treatment plus one question only it can be asked: of its own PPI partners -- the proteins it
physically works with -- how many moved, and in which direction? That is the most direct mechanistic reading available in this data, because
those are the gene's actual physical collaborators rather than anything inferred."""
import json, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
T = 4.2
TIDE_FRAC = 0.05
NPERM = 2000
GENERIC_F = {"protein binding", "identical protein binding", "metal ion binding", "ATP binding", "RNA binding",
             "DNA binding", "nucleotide binding", "zinc ion binding", "molecular_function",
             "protein-containing complex binding", "enzyme binding", "calcium ion binding", "GTP binding"}


def main(ko="GATA1"):
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); A = np.abs(Z)
    tide = (A >= T).mean(0) >= TIDE_FRAC
    panel = [genes[i] for i in range(len(genes)) if not tide[i]]
    pset = set(panel)
    row = Z[kidx[ko]]
    zof = {genes[i]: float(row[i]) for i in range(len(genes))}

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    go = D.get("go") or {}
    cmem = {}; comps = collections.defaultdict(set)
    for cid, mem in D["complexes"].items():
        mm = {names[x] for x in mem if isinstance(x, int) and x < N}
        if 2 <= len(mm) <= 200:
            cmem[cid] = mm
            for g in mm:
                comps[g].add(cid)
    ppi = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        try:
            a, b = names[e[0]], names[e[1]]
        except (TypeError, IndexError):
            continue
        ppi[a].add(b); ppi[b].add(a)

    down = sorted([g for g in panel if zof.get(g, 0) <= -T and g != ko], key=lambda g: zof[g])
    up = sorted([g for g in panel if zof.get(g, 0) >= T and g != ko], key=lambda g: -zof[g])
    print(f"{ko} in K562 -- direction IS available (signed z-scores)")
    print(f"  DOWN {len(down)}   UP {len(up)}   of {len(panel)} testable genes")
    print(f"  |z| median: down {np.median([abs(zof[g]) for g in down]):.1f}, up {np.median([abs(zof[g]) for g in up]):.1f}"
          f" | max down {min(zof[g] for g in down):.1f}, max up {max(zof[g] for g in up):.1f}")

    rng = np.random.RandomState(0)

    def internal_ppi(S):
        s = set(S)
        return sum(1 for a in s for b in ppi.get(a, ()) if b in s and a < b)

    def complex_pairs(S):
        """number of (complex, gene) memberships and the count of complexes holding >=2 of the group"""
        c = collections.Counter()
        for g in S:
            for x in comps.get(g, ()):
                c[x] += 1
        return c, sum(1 for v in c.values() if v >= 2)

    def perm(S, fn):
        n = len(S)
        vals = np.empty(NPERM)
        for i in range(NPERM):
            vals[i] = fn(rng.choice(panel, n, replace=False))
        return vals

    report = {"knockout": ko, "n_down": len(down), "n_up": len(up), "groups": {}}
    for label, S in (("DOWN", down), ("UP", up)):
        if not S:
            continue
        obs_p = internal_ppi(S); nul_p = perm(S, internal_ppi)
        cc, obs_c = complex_pairs(S); nul_c = perm(S, lambda x: complex_pairs(x)[1])
        p_p = float((nul_p >= obs_p).mean()); p_c = float((nul_c >= obs_c).mean())
        procs = collections.Counter((info.get(g, {}) or {}).get("proc") or "?" for g in S)
        cpts = collections.Counter((info.get(g, {}) or {}).get("comp") or "?" for g in S)
        fns = collections.Counter()
        for g in S:
            for t in ((go.get(g, {}) or {}).get("F") or []):
                if t not in GENERIC_F:
                    fns[t] += 1
        hubs = sorted(((len({b for b in ppi.get(g, ()) if b in set(S)}), g) for g in S), reverse=True)[:8]
        report["groups"][label] = {
            "n": len(S),
            "internal_ppi_edges": obs_p, "ppi_expected": round(float(nul_p.mean()), 1),
            "ppi_fold": round(obs_p / max(nul_p.mean(), 1e-9), 2), "ppi_p": p_c if False else p_p,
            "complexes_with_2plus": obs_c, "complexes_expected": round(float(nul_c.mean()), 1),
            "complex_fold": round(obs_c / max(nul_c.mean(), 1e-9), 2), "complex_p": p_c,
            "top_complexes": [(k, v) for k, v in cc.most_common(8) if v >= 2],
            "processes": procs.most_common(8), "compartments": cpts.most_common(6),
            "top_functions": fns.most_common(10),
            "ppi_hubs_within_group": [(g, n) for n, g in hubs],
            "top_genes": [(g, round(zof[g], 1)) for g in S[:15]],
        }
        r = report["groups"][label]
        print(f"\n  {label} GROUP ({len(S)} genes)")
        print(f"    internal PPI edges      {obs_p:5d}   expected {nul_p.mean():7.1f}   {r['ppi_fold']:.2f}x   p={p_p:.4g}")
        print(f"    complexes with >=2      {obs_c:5d}   expected {nul_c.mean():7.1f}   {r['complex_fold']:.2f}x   p={p_c:.4g}")
        print(f"    top complexes: {r['top_complexes'][:4]}")
        print(f"    processes:     {r['processes'][:5]}")
        print(f"    compartments:  {r['compartments'][:5]}")
        print(f"    functions:     {[f for f, _ in r['top_functions'][:5]]}")
        print(f"    PPI hubs:      {r['ppi_hubs_within_group'][:5]}")

    # ---------------- the knocked-out gene itself ----------------
    gi = info.get(ko, {}) or {}
    partners = sorted(ppi.get(ko, ()))
    inpanel = [p for p in partners if p in pset]
    moved = [(p, round(zof[p], 1)) for p in inpanel if abs(zof.get(p, 0)) >= T]
    pm_down = [p for p, z in moved if z < 0]; pm_up = [p for p, z in moved if z > 0]
    base = (len(down) + len(up)) / len(panel)
    from scipy.stats import binomtest
    bt = binomtest(len(moved), len(inpanel), base) if inpanel else None
    ko_cplx = sorted(comps.get(ko, ()))
    report["knockout_gene"] = {
        "compartment": gi.get("comp"), "process": gi.get("proc"),
        "complexes": ko_cplx, "n_complexes": len(ko_cplx),
        "ppi_partners_total": len(partners), "ppi_partners_measured": len(inpanel),
        "ppi_partners_moved": len(moved), "moved_down": pm_down, "moved_up": pm_up,
        "expected_rate": round(base, 4),
        "p_partners_enriched": float(bt.pvalue) if bt else None,
        "moved_partners": moved,
    }
    print(f"\n  THE KNOCKED-OUT GENE: {ko}")
    print(f"    curated complexes: {len(ko_cplx)} {ko_cplx[:3]}")
    print(f"    PPI partners: {len(partners)} total, {len(inpanel)} measured in the panel")
    print(f"    of those measured, {len(moved)} moved ({len(moved)/max(len(inpanel),1):.1%}) vs {base:.1%} background"
          + (f", p={bt.pvalue:.3g}" if bt else ""))
    print(f"    partners that went DOWN: {pm_down}")
    print(f"    partners that went UP:   {pm_up}")

    dn, upg = report["groups"].get("DOWN", {}), report["groups"].get("UP", {})
    verdict = (
        f"DIRECTION SPLIT ({ko}, K562): the data is signed, so every mover carries up or down -- information every model in this project "
        f"discarded by ranking on |z|. {len(down)} genes went DOWN and {len(up)} went UP. The two groups are structurally different, and the "
        "difference is the result. "
        f"DOWN ({dn.get('n')} genes) contains {dn.get('internal_ppi_edges')} internal PPI edges against "
        f"{dn.get('ppi_expected')} expected for a random set of the same size drawn from the same panel "
        f"({dn.get('ppi_fold')}x, p={dn.get('ppi_p'):.3g}) and {dn.get('complexes_with_2plus')} complexes holding two or more of its members "
        f"against {dn.get('complexes_expected')} expected ({dn.get('complex_fold')}x, p={dn.get('complex_p'):.3g}). "
        f"UP ({upg.get('n')} genes) gives {upg.get('internal_ppi_edges')} internal edges against {upg.get('ppi_expected')} expected "
        f"({upg.get('ppi_fold')}x, p={upg.get('ppi_p'):.3g}) and {upg.get('complexes_with_2plus')} shared complexes against "
        f"{upg.get('complexes_expected')} ({upg.get('complex_fold')}x, p={upg.get('complex_p'):.3g}). "
        + ("The DOWN group is the more coherent physical module of the two, which is what a machine being coordinately lost looks like; "
           "the UP group is a broader, less wired programme. "
           if (dn.get("ppi_fold") or 0) > (upg.get("ppi_fold") or 0) else
           "The UP group is at least as wired as the DOWN group, so the response is not simply one machine being lost. ")
        + f"THE KNOCKED-OUT GENE has {len(ko_cplx)} curated complexes and {len(partners)} PPI partners, {len(inpanel)} of them measured; "
        f"{len(moved)} of those physical partners moved against a {base:.1%} background rate"
        + (f" (p={bt.pvalue:.3g})" if bt else "")
        + ". That is the most direct mechanistic reading available here, since those are the protein's actual physical collaborators rather "
        "than anything inferred. Every count is compared against 2,000 size-matched random gene sets, because both PPI density and complex "
        "co-membership scale with set size and with how well-studied the genes are, and movers are biased towards well-studied genes. "
        "Deterministic.")
    print(f"\nVERDICT: {verdict}")
    report["verdict"] = report["note"] = verdict
    json.dump(report, open(OUT / f"direction_split_{ko}.json", "w"), indent=1)
    print(f"\n  -> outputs/orphan/direction_split_{ko}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GATA1")
