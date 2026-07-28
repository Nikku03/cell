"""VALIDATE the tissue layer — does it recover textbook cell-cell communication in the RIGHT cells?

The tissue model wires each organ's cell types by ligand->receptor channels. Test whether known cell-cell
signaling axes (immune checkpoint, fibrosis, insulin, angiogenesis, ...) show up as channels, and in
biologically-sensible sender->receiver cell pairs. Ground truth = textbook signalling biology, not trained.
Anti-cheat = random baseline: a random gene pair is almost never a channel (channels come from ~2k real
ligand-receptor pairs), so recovery must be far above chance.

Reads outputs/orphan/tissue_channels.json (compact, from the tissue build) or tissue_model.json.
-> tissue_communication_validation.json
"""
import os
import json, random
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

# (ligand, receptor, sender substring, receiver substring) — '' = any cell.  Fixed a priori from biology.
AXES = [
    ("CTLA4", "CD86", "t cell", "macrophage"),   ("CD28", "CD86", "t cell", "antigen"),
    ("CTLA4", "CD80", "t", "macro"),             ("COL1A1", "ITGB1", "fibro", "endothel"),
    ("COL1A1", "CD93", "stellate", "endothel"),  ("INS", "INSR", "", ""),
    ("GCG", "GCGR", "", ""),                      ("VEGFA", "KDR", "", "endothel"),
    ("GDF2", "ACVRL1", "", "endothel"),          ("PDGFB", "PDGFRB", "", "peric"),
    ("TGFB1", "TGFBR2", "", ""),                 ("CXCL12", "CXCR4", "", ""),
    ("DLL4", "NOTCH1", "endothel", ""),          ("WNT2", "FZD1", "", ""),
    ("EGF", "EGFR", "", ""),                     ("IL7", "IL7R", "", "lymph"),
]


def load():
    p = OUT / "tissue_channels.json"
    if p.exists():
        d = json.load(open(p)); return [tuple(c) for c in d["channels"]]
    tm = OUT / "tissue_model.json"
    if tm.exists():
        T = json.load(open(tm)); ch = []
        for tis, v in T["tissues"].items():
            for e in v.get("comm", []):
                for pr in e[2]:
                    ch.append((pr[0].upper(), pr[1].upper(), e[0].lower(), e[1].lower(), tis))
        return ch
    return None


def validate():
    chans = load()
    if chans is None:
        print("no tissue_channels.json / tissue_model.json"); return None
    lig = list({c[0] for c in chans}); rec_g = list({c[1] for c in chans})

    def present(l, r, sp, rp):
        hits = [c for c in chans if c[0] == l and c[1] == r]
        ctx = [c for c in hits if (sp in c[2] or sp == "") and (rp in c[3] or rp == "")]
        return len(hits), len(ctx), (hits[0] if hits else None)

    rows = []
    for l, r, sp, rp in AXES:
        n, nc, eg = present(l, r, sp, rp)
        rows.append(dict(axis=f"{l}->{r}", present=n > 0, right_cells=nc > 0,
                         example=(f"{eg[4]}:{eg[2]}->{eg[3]}" if eg else None)))
    recall = sum(r["present"] for r in rows) / len(rows)
    ctx_recall = sum(r["right_cells"] for r in rows) / len(rows)
    # random baseline: random (ligand, receptor) drawn from observed channel genes
    rng = random.Random(0); chanset = set((c[0], c[1]) for c in chans); hit = 0; N = 3000
    for _ in range(N):
        if (rng.choice(lig), rng.choice(rec_g)) in chanset:
            hit += 1
    rand = hit / N
    return dict(summary=dict(
        n_axes=len(rows), recall=round(recall, 3), context_recall=round(ctx_recall, 3),
        random_baseline=round(rand, 4),
        recall_over_random=round(recall / rand, 1) if rand else None,
        n_channels=len(chans), n_unique_lr=len(set((c[0], c[1]) for c in chans))), per_axis=rows)


def main():
    res = validate()
    if not res:
        return
    json.dump(res, open(OUT / "tissue_communication_validation.json", "w"), indent=2)
    s = res["summary"]
    print("=" * 74)
    print("TISSUE LAYER — textbook cell-cell communication recovery")
    print("=" * 74)
    print(f"  axes present: recall {s['recall']} | in biologically-right cells {s['context_recall']}")
    print(f"  random (ligand,receptor)-pair baseline {s['random_baseline']} -> {s['recall_over_random']}x lift")
    print(f"  ({s['n_channels']} channels, {s['n_unique_lr']} unique ligand-receptor pairs)")
    for r in res["per_axis"]:
        tag = ("right-cells" if r["right_cells"] else "present") if r["present"] else "ABSENT"
        print(f"    {r['axis']:16} {tag}{'  '+r['example'] if r['example'] else ''}")


if __name__ == "__main__":
    main()
