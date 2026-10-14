"""THE DATASET LEDGER -- one table for fifteen items, with verification status attached to each.

WHY THIS EXISTS AND WHY IT CARRIES A STATUS COLUMN. Fifteen dataset-map items were built in one night across
two parallel workflows. Their results currently live in fourteen separate JSON files and a long commit
history, which is not a form anything can read -- not a person deciding what to build next, and not the model
deciding what it is allowed to assert.

But a consolidated table is dangerous in a specific way. Each item was supposed to get an adversarial reviewer
whose brief was to REFUTE its reported effect, and the reviewers earned that place: they marked the HPA
per-compartment breakdown as exploratory rather than prespecified, rewrote BioPlex substantially, and revised
CORUM and translation. The workflows were then stopped early to conserve usage, so only some items got one.

Listing a reviewed result beside an unreviewed one, in the same table, in the same font, asserts that they
carry the same weight. They do not. Every row therefore carries its verification status, derived from the
repository rather than from memory: a module whose file changed after the commit that introduced it was
revised by a reviewer; one that did not, was not.

WHAT THE FAST TEST IS, since every ratio here refers to it. Layer-linked (perturbation, gene) pairs against
pairs matched on perturbation-strength decile x gene-responsiveness decile, in Replogle K562, on |robust z|.
Reference points measured earlier in this project: PPI 1.208x, co-dependency 1.070x, regulatory 1.021x,
signalling 1.028x (null, p 0.99). Below about 1.05 is not interesting. A ratio is a raw held-out value and is
never a difference from a shuffle.

THIS LEDGER DOES NOT ADJUDICATE. It reports what each module concluded and whether anyone tried to break it.
Where a module's own verdict is a null or a blocked source, that is carried through unchanged -- a
well-controlled null and an unreachable host are both results, and burying either would defeat the purpose.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

# item -> (dataset-map number, one-line description of what it was asked to settle)
ITEMS = {
    "scperturb_transfer": (2, "does the perturbational layer's sign survive a change of cell line?"),
    "bioplex": (6, "rebuild PPI retaining cell line and confidence"),
    "corum": (7, "protein complexes with citable provenance"),
    "hpa_localization": (5, "does compartment gating improve the PPI layer?"),
    "ptm_sites": (8, "positioned PTM sites and modifying enzymes"),
    "depmap_codep": (15, "codependency scored on the RIGHT endpoint (viability, not RNA)"),
    "encode_state": (1, "is cell-matched K562 RNA a better covariate than generic abundance?"),
    "measurement_power": (16, "the reliability ceiling, derived from raw counts"),
    "lincs_timedose": (9, "does the dynamics null replicate on time- and dose-resolved data?"),
    "enzyme_capacity": (12, "does kcat x [E] beat a binary required-catalyst flag?"),
    "screen_ccre": (1, "how much of the K562 regulatory landscape can the E-G layer even reach?"),
    "translation": (11, "translation efficiency as the missing mRNA->protein multiplier"),
    "ligand_receptor": (14, "what can a ligand-receptor layer support in monoculture?"),
    "pancancer_proteome": (4, "does cell-matched protein beat generic, across lines?"),
    "environment": (13, "recorded environmental state per dataset"),
}


def review_status(mod):
    """Reviewed iff the module file changed after the commit that introduced it.

    Derived from the repository, not from recollection. An agent that revises a module leaves a second commit
    touching it; an item whose reviewer never ran leaves exactly one.
    """
    p = f"colab/{mod}.py"
    if not (ROOT / p).exists():
        return "not built"
    try:
        n = subprocess.run(["git", "log", "--oneline", "--", p], cwd=ROOT,
                           capture_output=True, text=True, timeout=60).stdout.strip()
        return "reviewed" if len(n.splitlines()) > 1 else "single-pass"
    except Exception:
        return "unknown"


def main():
    print("=" * 100)
    print("DATASET LEDGER -- fifteen items, with verification status")
    print("=" * 100)

    rows = []
    for mod, (num, question) in ITEMS.items():
        jp = OUT / f"{mod}.json"
        d = json.load(open(jp)) if jp.exists() else {}
        rows.append({
            "item": mod, "map_number": num, "question": question,
            "built": bool(d), "verification": review_status(mod),
            # Not every module writes a `verdict` key -- screen_ccre reports a census under its own
            # field names. Falling through to "not built" for those would mislabel a completed item as
            # missing, which is the opposite of what this ledger is for.
            "verdict": (str(d.get("verdict", "")) or
                        (f"(no verdict field; reports {', '.join(list(d)[:4])})" if d else None)),
            "limits": d.get("limits") or d.get("Limits") or [],
        })

    built = [r for r in rows if r["built"]]
    rev = [r for r in built if r["verification"] == "reviewed"]
    sp = [r for r in built if r["verification"] == "single-pass"]
    missing = [r for r in rows if not r["built"]]

    print(f"  {len(built)}/{len(rows)} items built")
    print(f"  {len(rev)} adversarially reviewed, {len(sp)} single-pass, {len(missing)} not built")
    print(f"\n  {'item':22s} {'map':>4s} {'status':12s} verdict (first line)")
    for r in sorted(rows, key=lambda x: (not x["built"], x["verification"], x["item"])):
        head = (r["verdict"] or "NOT BUILT").split(".")[0][:60]
        print(f"  {r['item']:22s} {r['map_number']:>4d} {r['verification']:12s} {head}")

    print(f"\n  REVIEWED ({len(rev)}) -- someone tried to refute these:")
    for r in rev:
        print(f"    {r['item']}")
    print(f"  SINGLE-PASS ({len(sp)}) -- built and self-tested, but nobody tried to break them:")
    for r in sp:
        print(f"    {r['item']}")
    if missing:
        print(f"  NOT BUILT ({len(missing)}):")
        for r in missing:
            print(f"    {r['item']} (map item {r['map_number']}: {r['question']})")

    led = {
        "purpose": "consolidated result of the dataset-map build; consult BEFORE asserting anything about "
                   "these layers, and read the verification column before treating a row as established",
        "generated_by": "colab/dataset_ledger.py",
        "fast_test": "layer-linked (perturbation, gene) pairs vs pairs matched on perturbation-strength x "
                     "gene-responsiveness deciles, Replogle K562, |robust z|. Raw held-out ratios, never a "
                     "difference from a shuffle. Reference: PPI 1.208x, co-dependency 1.070x, "
                     "regulatory 1.021x, signalling 1.028x (null). Below ~1.05 is not interesting.",
        "verification_meaning": {
            "reviewed": "an adversarial agent was tasked with refuting the reported effect and the module "
                        "was revised as a result",
            "single-pass": "built and self-tested against the project's controls, but no independent attempt "
                           "was made to break it. Weaker evidence than a reviewed row, and the workflows "
                           "were stopped early to conserve usage rather than because these were finished",
            "not built": "no module exists"},
        "counts": {"built": len(built), "reviewed": len(rev), "single_pass": len(sp),
                   "not_built": len(missing)},
        "items": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(led, open(OUT / "dataset_ledger.json", "w"), indent=1)
    print(f"\n  -> {OUT/'dataset_ledger.json'}")


if __name__ == "__main__":
    main()
