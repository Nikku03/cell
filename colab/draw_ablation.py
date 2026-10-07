"""DRAW EACH REACTION WITH EACH PARTICIPANT REMOVED, ONE AT A TIME.

Renders the ablation table as readable chemistry. For every reaction it prints the equation once, then one line per
participant showing what happens when that single thing is missing:

    STOPS              nothing else supplies it and there is no second enzyme -- the step dies
    REDUNDANT          another annotated catalyst covers it
    BUFFERED           another reaction produces this substrate
    BUFFERED_CURRENCY  ATP, water, phosphate: supplied by the rest of metabolism, not by this step

and, when a step dies, what stops being made and how many downstream reactions lose their supply.

The struck-through participant is the one removed. Currency species are shown in the equation because a reaction
that spends ATP should say so, but they are marked, because joining reactions on them is the classic way to build a
dense meaningless network.
"""
import collections
import json
import os
import sys
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
MARK = {"STOPS": "STOPS    ", "REDUNDANT": "redundant", "BUFFERED": "buffered ",
        "BUFFERED_CURRENCY": "currency "}


def strike(s):
    """Overstrike a name with combining long stroke, so the removed participant is visible in the equation."""
    return "".join(ch + "̶" for ch in s)


def render(sid, steps, ab, maxw=88):
    s = steps[sid]
    ins = [(p["name"], bool(p.get("currency"))) for p in s["in"]]
    outs = [(p["name"], bool(p.get("currency"))) for p in s["out"]]
    cats = list(s["catalysts"])
    where = ",".join(ab[0].get("where") or []) or "?"

    def side(xs):
        return " + ".join((f"{n}*" if c else n)[:34] for n, c in xs) or "(none)"

    arrow = f"--[{','.join(cats)[:30]}]-->" if cats else "-->"
    print(f"\n{sid}  [{where}]  ({s['source']})")
    print(f"   {side(ins)}  {arrow}  {side(outs)}")
    for r in ab:
        rem = r["removed"]
        eq_in = " + ".join((strike(n) if n == rem else n)[:30] for n, _ in ins) or "(none)"
        eq_cat = ",".join(strike(c) if c == rem else c for c in cats) or "-"
        tail = ""
        if r["verdict"] == "STOPS":
            if r["products_lost"]:
                tail = f"  lost: {', '.join(r['products_lost'][:2])[:44]}"
            if r["n_starved_rxn"]:
                tail += f"  -> starves {r['n_starved_rxn']} rxn"
        elif r["verdict"] == "REDUNDANT":
            tail = f"  ({r['n_alt_catalysts']} other catalyst)"
        elif r["verdict"] == "BUFFERED":
            tail = f"  ({r['n_alt_producers']} other producer)"
        flag = "" if r.get("addressable") else "  ~"
        print(f"     - {rem[:26]:<26} {r['role'][:8]:<8} {MARK[r['verdict']]}{tail}{flag}"[:maxw + 30])
        print(f"         {eq_in}  --[{eq_cat}]-->"[:maxw + 30])


def main():
    A = json.loads((OUT / "reaction_ablation.json").read_text())
    net = json.load(open(OUT / "reaction_network.json"))
    steps = {s["id"]: s for s in net["steps"]}
    by = collections.defaultdict(list)
    for r in A["rows"]:
        by[r["rxn"]].append(r)

    want = sys.argv[1:] if len(sys.argv) > 1 else None
    if want:
        for sid in want:
            if sid in by:
                render(sid, steps, by[sid])
            else:
                print(f"{sid}: not in the ablation table")
        return

    # otherwise show one worked example of each interesting kind. Sections are made DISJOINT as they are filled --
    # the first version printed the same reaction under both "redundant catalyst" and "fully buffered", which is
    # true of the reaction and useless as an illustration.
    shown = set()

    def take(cands, k=2):
        got = []
        for sid in cands:
            if sid not in shown and sid in by:
                shown.add(sid)
                got.append(sid)
                if len(got) == k:
                    break
        return got

    def section(title, cands, k=2):
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        for sid in take(cands, k):
            render(sid, steps, by[sid])

    # a knockout can only remove a gene product, so the cascade ranking is over ADDRESSABLE removals only
    def worst_addr(k):
        return max((x["n_starved_rxn"] for x in by[k] if x.get("addressable")), default=0)

    section("A  EVERY PARTICIPANT IS A SINGLE POINT OF FAILURE -- no redundancy anywhere in the step",
            [k for k, v in by.items() if all(x["verdict"] == "STOPS" for x in v) and 3 <= len(v) <= 6])
    section("B  THE WORST GENE-DRIVEN CASCADE -- one missing PROTEIN starves the most downstream reactions",
            sorted(by, key=lambda k: -worst_addr(k)))
    section("C  A REDUNDANT CATALYST -- removing the enzyme does NOT stop the step",
            [k for k, v in by.items() if any(x["verdict"] == "REDUNDANT" for x in v) and len(v) <= 6])
    section("D  FULLY BUFFERED -- nothing you remove from this step stops it",
            [k for k, v in by.items() if all(x["verdict"] != "STOPS" for x in v) and 3 <= len(v) <= 6])
    print("\n* = currency species (ATP/H2O/O2/phosphate): supplied by the rest of metabolism, not by this step")
    print("~ = the removed thing is a small molecule, so no genetic perturbation can remove it -- the row is true"
          "\n    chemistry but is NOT a knockout prediction")
    print(f"\nrun `python3 colab/draw_ablation.py R-HSA-8848436 ...` to draw any specific reaction")


if __name__ == "__main__":
    main()
