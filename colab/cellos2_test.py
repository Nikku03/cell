"""CellOS v2 test: assert the kernel's CLAIMS, not just that its syscalls run without error.

v1's test checked that syscalls returned something and that two hand-picked biological facts came out right. That
catches crashes and nothing else. The failure mode this project actually suffers is different and worse: a syscall
that runs, prints a plausible answer, and is built on an empty join. Four of those happened in one day -- the PPI
index-space bug (0 of 191,447 edges), the shallow catalystActivity (0 of 1,148 catalysts), non-targeting controls
scored as knockouts, and the FBA validation joining Ensembl ids against gene symbols (0 of 2,741 genes).

So every assertion here is of the form "this layer contains real, joined data and its number is what the kernel
claims", and each one is written to FAIL LOUDLY on an empty intersection rather than pass on zero rows.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
fails = []


def check(name, cond, detail=""):
    print(f"   [{'ok  ' if cond else 'FAIL'}] {name:<44} {detail}")
    if not cond:
        fails.append(name)


def main():
    print("A) syscalls run")
    for cmd in (["boot"], ["spec"], ["top", "5"]):
        r = subprocess.run([sys.executable, "colab/cellos2.py"] + cmd, capture_output=True, text=True, timeout=900)
        check(f"cellos2 {' '.join(cmd)}", r.returncode == 0 and len(r.stdout) > 100,
              f"{len(r.stdout)} bytes")

    print("\nB) layers contain REAL JOINED data (the failure mode that actually bites)")
    net = json.load(open(OUT / "reaction_network.json"))
    check("reaction network non-empty", len(net["steps"]) > 20000, f"{len(net['steps']):,} steps")

    A = json.loads((OUT / "reaction_ablation.json").read_text())
    check("ablation rows", A["n_ablations"] > 100000, f"{A['n_ablations']:,} ablations")
    v = A["verdicts"]
    check("ablation separates verdicts", len({k for k, n in v.items() if n > 1000}) >= 4, str(v))

    # THE JOIN TEST: DepMap must actually intersect the model's genes
    import csv
    dep = {}
    with open(SP / "CRISPRGeneEffect.csv") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        gs = [h.split(" (")[0] for h in head[1:]]
        for row in rd:
            if row[0].strip() == "ACH-000551":
                dep = {g: float(x) for g, x in zip(gs, row[1:]) if x not in ("", "NA")}
                break
    check("DepMap K562 row present", len(dep) > 15000, f"{len(dep):,} genes")
    abl_genes = {r["removed"] for r in A["rows"] if r["role"] == "CATALYST"}
    inter = abl_genes & set(dep)
    check("ablation x DepMap JOIN non-empty", len(inter) > 500,
          f"{len(inter):,} genes (a join failure would read 0 here)")

    print("\nC) the kernel's headline claims reproduce")
    # cascade breadth vs essentiality -- the one validated ablation claim
    import collections
    cas = collections.Counter()
    for r in A["rows"]:
        if r["role"] == "CATALYST" and r["n_starved_rxn"]:
            cas[r["removed"]] += r["n_starved_rxn"]
    wide = [g for g in cas if cas[g] >= 5 and g in dep]
    allg = [g for g in abl_genes if g in dep]
    if wide and allg:
        mw = float(np.mean([dep[g] for g in wide]))
        ma = float(np.mean([dep[g] for g in allg]))
        check("cascade breadth -> more essential", mw < ma - 0.2,
              f"starves>=5: {mw:+.4f} vs all {ma:+.4f} (n={len(wide)})")

    # FBA validation, if the sim has been run
    sim = OUT / "cell_sim.json"
    if sim.exists():
        d = json.loads(sim.read_text()).get("validation") or {}
        if d:
            check("FBA AUC beats chance", d.get("auc", 0) > 0.55, f"AUC {d.get('auc'):.4f}")
            check("FBA lethal calls enriched", d.get("precision", 0) > 3 * d.get("base_rate", 1),
                  f"precision {d.get('precision'):.3f} vs base {d.get('base_rate'):.3f}")
            check("FBA join non-empty", d.get("n_common", 0) > 1000, f"{d.get('n_common'):,} genes")
        else:
            check("FBA validation present", False, "cell_sim.json has no validation block")

    # the retired predictor must still be recorded as retired, not quietly revived
    oc = OUT / "output_ceiling.json"
    if oc.exists():
        d = json.loads(oc.read_text())
        check("sensor-menu ceiling still low", d["oracle_recall"] < 0.10,
              f"oracle recall {d['oracle_recall']:.1%} over {d['answer_space']:,} genes")

    print(f"\nRESULT: {len(fails)} failure(s)" + (f": {fails}" if fails else " -- all assertions hold"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
