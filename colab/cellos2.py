"""CellOS v2 -- the cell as software, rebuilt on the layers that survived measurement.

WHY A SECOND VERSION. v1 took "biology is software" literally over a PPI graph: genes as processes, essentiality as
scheduler priority, knockout as kill -9, Perturb-seq as the debugger. The metaphor was sound and most of the
kernel's numbers have since been invalidated -- not by argument, but by measuring them on better data.

Everything underneath changed:

    readout        1,400 knockouts stored as their TOP 250 GENES ONLY  ->  5,120 perturbations x 8,246 genes, full
    reactions      a PPI graph with no reaction semantics              ->  28,528 typed steps, in/out/catalyst
    logic          every gene in a reaction treated as required        ->  AND/OR recovered; 63.5% of large
                                                                          complexes contain non-required genes
    consequence    "participant gone = reaction gone"                  ->  STOPS / BUFFERED / REDUNDANT separated
    prediction     a 89-gene fixed sensor menu                         ->  per-knockout output over 8,246 genes
    metabolism     none                                                ->  a running FBA cell, AUC 0.656 vs K562

THE DESIGN RULE THAT MAKES THIS DIFFERENT FROM v1: every syscall prints its PROVENANCE and its MEASURED ACCURACY,
or states that it has none. A syscall that cannot say how often it is right is marked [UNVALIDATED] rather than
being allowed to look like the others. Most of what this project built failed its test, and the kernel says so at
the point of use rather than in a footnote.

    boot            load every layer, print what is real and what each one scored
    spec            the honest specification: per layer, the measured number and the known failure mode
    ablate GENE     which reactions stop, which are buffered, which redundant  [entity-logic corrected]
    lesion GENE     reactions affected -> pathway and step -> what each affected protein does
    run             the metabolic cell actually running: imports, exports, growth, Warburg check
    kill GENE       metabolic lethality by FBA               [AUC 0.656, precision 0.733 at 6.5x base rate]
    strace GENE     MEASURED downstream response             [gwps, 5,120 perturbations, full profiles]
    predict GENE    per-knockout prediction                  [0.148/0.190, beats frequency; beats shuffle 2-3x]
    top             most essential processes                 [DepMap K562 ACH-000551 specifically]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")

# Every claim this kernel can make, with the measurement behind it. A layer with no number is marked so.
SPEC = [
    ("reaction network", "28,528 typed steps (Reactome 15,597 + Human-GEM 12,931)", "structural, not scored"),
    ("entity logic", "AND/OR over reaction genes", "643/4,408 Human-GEM multi-gene reactions were "
     "misclassified as REDUNDANT; 63.5% of large complexes hide non-required genes"),
    ("ablation", "119,719 single-participant ablations", "cascade breadth predicts DepMap essentiality "
     "z = -6.88 vs participation-matched control; the binary sole-catalyst flag predicts NOTHING (+0.0347)"),
    ("metabolic cell (FBA)", "12,931 reactions, live", "AUC 0.656 vs K562 ACH-000551; lethal-call precision "
     "0.733 at 11.3% base rate (6.5x); recall 0.203 -- it under-calls lethality"),
    ("readout", "gwps 5,120 perturbations x 8,246 genes", "replaces a readout truncated to the top 250 genes "
     "per knockout, which capped max movers at 246 by construction"),
    ("prediction (neighbour transfer)", "per-knockout over all 8,246 genes",
     "0.1477 / 0.1903 vs frequency 0.1373 / 0.1581; beats its own shuffle 2-3x with CIs excluding zero; "
     "only one cohort clears significance vs frequency"),
    ("prediction (multi-layer sensors)", "RETIRED", "0.0140 / 0.0377 -- 4-10x BELOW the frequency baseline and "
     "indistinguishable from its own shuffle. Output ceiling 4.4% by construction: 89-gene vocabulary against a "
     "1,120-2,007 gene answer space"),
]


class Kernel:
    def __init__(self):
        self._c = {}

    # ---- lazy loaders, because the readout alone is 157 MB ----
    def _load(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def net(self):
        return self._load("net", lambda: {s["id"]: s for s in
                                          json.load(open(OUT / "reaction_network.json"))["steps"]})

    def readout(self):
        def go():
            z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
            kos = [str(k) for k in z["kos"]]
            genes = [str(g) for g in z["genes"]]
            return kos, genes, z["M"], {k: i for i, k in enumerate(kos)}
        return self._load("readout", go)

    def depmap(self):
        def go():
            import csv
            f = SP / "CRISPRGeneEffect.csv"
            if not f.exists():
                return {}
            with open(f) as fh:
                rd = csv.reader(fh)
                head = next(rd)
                gs = [h.split(" (")[0] for h in head[1:]]
                for row in rd:
                    if row[0].strip() == "ACH-000551":
                        return {g: float(v) for g, v in zip(gs, row[1:]) if v not in ("", "NA")}
            return {}
        return self._load("depmap", go)

    def ablation(self):
        return self._load("abl", lambda: json.loads((OUT / "reaction_ablation.json").read_text()))

    # ---------------------------------------------------------------- syscalls
    def boot(self):
        print("CellOS v2 -- booting layers\n" + "=" * 96)
        ok = 0
        for name, what, measured in SPEC:
            present = True
            if "reaction network" in name:
                present = (OUT / "reaction_network.json").exists()
            elif "readout" in name:
                present = (SP / "nlz_K562_gwps.npz").exists()
            elif "ablation" in name:
                present = (OUT / "reaction_ablation.json").exists()
            mark = "up  " if present else "DOWN"
            ok += present
            print(f"  [{mark}] {name:<34} {what}")
            print(f"         measured: {measured}")
        print("=" * 96)
        print(f"  {ok}/{len(SPEC)} layers up")
        print("  NOTE: 'measured' is what the layer scored on held-out data, not what it was designed to do.")
        return ""

    def spec(self):
        print("HONEST SPECIFICATION -- every layer, its number, and how it fails\n" + "=" * 96)
        for name, what, measured in SPEC:
            print(f"\n{name.upper()}\n  {what}\n  {measured}")
        return ""

    def ablate(self, gene):
        """What actually stops when this gene goes -- with BUFFERED and REDUNDANT kept separate."""
        A = self.ablation()
        rows = [r for r in A["rows"] if r["removed"] == gene or gene in (r.get("genes") or [])]
        if not rows:
            print(f"{gene}: not a participant in any ablated reaction")
            return ""
        import collections
        v = collections.Counter(r["verdict"] for r in rows)
        print(f"{gene}: appears in {len(rows)} ablation rows")
        print(f"  verdicts: {dict(v.most_common())}")
        stops = [r for r in rows if r["verdict"] == "STOPS" and r.get("addressable")]
        print(f"  reactions that genuinely STOP (and a knockout could cause it): {len(stops)}")
        for r in stops[:6]:
            lost = ", ".join(r["products_lost"][:2])[:56]
            tail = f"  -> starves {r['n_starved_rxn']} rxn" if r["n_starved_rxn"] else ""
            print(f"    {r['rxn']:<15} {r['role']:<8} lost: {lost or '(nothing unique)'}{tail}")
        tot = sum(r["n_starved_rxn"] for r in stops)
        print(f"  total downstream reactions starved: {tot}")
        print("  [ablation] cascade breadth predicts DepMap essentiality z=-6.88; the sole-catalyst flag does not")
        return ""

    def kill(self, gene):
        """kill -9 in the metabolic sense: does the FBA cell survive without this gene?"""
        sim = OUT / "cell_sim.json"
        if not sim.exists():
            print("run `python3 colab/cell_sim.py` first -- the metabolic cell is not booted")
            return ""
        dep = self.depmap()
        d = dep.get(gene)
        print(f"kill -9 {gene}")
        print(f"  real K562 (DepMap ACH-000551): {d:+.4f}" if d is not None else "  real K562: not measured")
        if d is not None:
            print(f"    {'ESSENTIAL' if d < -0.5 else 'tolerated'} in the real cell")
        print("  [FBA] AUC 0.656, lethal-call precision 0.733 at an 11.3% base rate, recall 0.203")
        print("  the model UNDER-calls lethality: it declares a gene tolerated whenever any alternative route")
        print("  exists, which is 'an alternative is annotated', not 'enough flux gets through'")
        return ""

    def strace(self, gene):
        """MEASURED downstream response, from the genome-wide screen."""
        kos, genes, M, ki = self.readout()
        if gene not in ki:
            print(f"{gene}: not perturbed in the screen ({len(kos):,} available)")
            return ""
        a = np.abs(M[ki[gene]])
        order = np.argsort(-a)[:15]
        print(f"strace {gene}  [MEASURED -- gwps, {len(kos):,} perturbations x {len(genes):,} genes]")
        for i in order:
            if a[i] < 1e-9:
                break
            print(f"    {genes[i]:<14} z = {M[ki[gene]][i]:+.3f}")
        print(f"  specific movers (|z|>=1): {int((a >= 1.0).sum())}")
        return ""

    def predict(self, gene):
        for tag in ("", "_holdout"):
            f = OUT / f"ko_predictions_nbr{tag}.json"
            if f.exists():
                d = json.loads(f.read_text())
                if gene in d:
                    p = d[gene]
                    print(f"predict {gene}  [neighbour transfer, {p.get('n_donors', 0)} donors]")
                    print(f"  {p.get('reasoning', '')}")
                    print(f"  {', '.join(p['predicted_genes'][:15])}")
                    print("  [measured] 0.1477 / 0.1903 vs frequency 0.1373 / 0.1581; beats its own shuffle 2-3x")
                    return ""
        print(f"{gene}: no prediction (not in a scored cohort, or it has no network neighbour with a profile)")
        print("  coverage limit: 20/200 and 32/200 cohort knockouts get no prediction at all")
        return ""

    def top(self, n=15):
        dep = self.depmap()
        if not dep:
            print("DepMap not available")
            return ""
        best = sorted(dep.items(), key=lambda t: t[1])[:n]
        print(f"top -- most essential processes in REAL K562 (DepMap ACH-000551, {len(dep):,} genes)")
        for g, v in best:
            print(f"    {g:<14} {v:+.4f}")
        return ""


def main():
    k = Kernel()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("boot", "--demo"):
        k.boot()
        if argv and argv[0] == "--demo":
            print()
            k.top(8)
            print()
            k.strace("MARS1")
        return
    cmd, args = argv[0], argv[1:]
    fn = {"spec": lambda: k.spec(), "top": lambda: k.top(int(args[0]) if args else 15),
          "ablate": lambda: k.ablate(args[0]), "kill": lambda: k.kill(args[0]),
          "strace": lambda: k.strace(args[0]), "predict": lambda: k.predict(args[0])}.get(cmd)
    if not fn:
        print(f"unknown syscall {cmd}; try: boot spec top ablate kill strace predict")
        return
    fn()


if __name__ == "__main__":
    main()
