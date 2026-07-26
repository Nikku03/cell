"""MITOCARTA 3.0: IS IT THE SEPARATE COMPARTMENT OR THE SEPARATE GENOME?

WHERE THIS CAME FROM. compartment_isolation.py measured co-essentiality tightness for 1,867 pathways against
size-matched random gene sets. Mitochondrial pathways took ranks 1, 2, 3, 4 and 5, with a median rank of 30/1,867 --
the top 2%, where chance is 50%. By compartment, mitochondrion averaged +0.1624 against nucleus +0.0367 and
cytosol +0.0232.

TWO EXPLANATIONS FIT THAT, AND THEY MAKE OPPOSITE PREDICTIONS.

  COMPARTMENT   a membrane-bounded organelle isolates a process, so its parts are needed together.
                Predicts peroxisome and lysosome should also be tight. MEASURED: peroxisome +0.0297, lysosome
                +0.0240 -- indistinguishable from cytosol. So isolation alone does NOT do it.

  GENOME        the mitochondrion is the only organelle with its own DNA, and expressing it requires a dedicated,
                non-redundant apparatus -- its own ribosome, its own initiation/elongation factors, its own tRNA
                synthetases -- that is not shared with cytosolic translation. Lose any part and mitochondrial
                protein synthesis fails as a unit.
                Predicts that WITHIN the mitochondrion, genes serving the genome are tighter than genes that merely
                live there.

MitoCarta 3.0 makes that second prediction directly testable, which is why it is worth pulling in. It is an
experimental inventory -- mass spectrometry across 14 tissues, GFP imaging, targeting-signal prediction -- of 1,136
human mitochondrial proteins, annotated with sub-mitochondrial location (Matrix 523, inner membrane 359, outer
membrane 110, IMS 51) and a 154-node MitoPathways hierarchy whose top-level "Mitochondrial central dogma" branch is
exactly "the machinery for the separate genome": mtDNA maintenance, mtDNA transcription, mitochondrial translation.

    CENTRAL DOGMA   genes under Mitochondrial central dogma -- serve the separate genome
    OTHER MITO      MitoCarta genes not under it -- OXPHOS, metabolism, transport: same compartment, no genome role

If tightness is about the compartment, these two should be similar. If it is about the genome, central dogma should
be far tighter. The two groups share an organelle, so compartment is held fixed by construction.

MitoCarta is also independent EXPERIMENTAL ground truth for localisation, so it is used to score the compartment
assignment built from Reactome and Human-GEM annotation -- a check that assignment has not had until now.
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MITO = SP / "Human.MitoCarta3.0.xls"
DEPMAP = SP / "CRISPRGeneEffect.csv"
MIN_LINES, MAX_PAIRS, NRAND = 200, 800, 20


def load_depmap():
    with open(DEPMAP) as f:
        r = csv.reader(f)
        head = next(r)
        genes = [h.split(" (")[0] for h in head[1:]]
        rows = [[np.nan if x == "" else float(x) for x in row[1:]] for row in r]
    M = np.array(rows, np.float32)
    keep = (~np.isnan(M)).sum(0) >= MIN_LINES
    idx = {g: i for i, g in enumerate(genes) if keep[i]}
    Z = np.full(M.shape, np.nan, np.float32)
    for g, i in idx.items():
        v = M[:, i]
        m = ~np.isnan(v)
        s = v[m].std()
        if s > 0:
            Z[m, i] = (v[m] - v[m].mean()) / s
    return Z, idx


def tight(Z, idx, members, rng):
    ids = [idx[g] for g in members if g in idx]
    if len(ids) < 6:
        return float("nan"), len(ids)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    if len(pairs) > MAX_PAIRS:
        pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRS, replace=False)]
    cs = []
    for a, b in pairs:
        x, y = Z[:, a], Z[:, b]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() >= MIN_LINES:
            cs.append(float(np.dot(x[ok], y[ok]) / ok.sum()))
    return (float(np.mean(cs)) if cs else float("nan")), len(ids)


def scored(Z, idx, members, rng, allg):
    """Tightness minus the mean of size-matched random gene sets."""
    t, n = tight(Z, idx, members, rng)
    if t != t:
        return float("nan"), n
    ctrl = []
    for _ in range(NRAND):
        samp = [allg[i] for i in rng.choice(len(allg), n, replace=False)]
        c, _ = tight(Z, idx, samp, rng)
        if c == c:
            ctrl.append(c)
    return (t - float(np.mean(ctrl))) if ctrl else float("nan"), n


def main():
    a = pd.read_excel(MITO, "A Human MitoCarta3.0")
    paths = pd.read_excel(MITO, "C MitoPathways")
    sym = {str(s).strip() for s in a["Symbol"].dropna()}
    subloc = {str(r["Symbol"]).strip(): str(r["MitoCarta3.0_SubMitoLocalization"]).strip()
              for _, r in a.iterrows() if pd.notna(r.get("Symbol"))}
    print(f"MitoCarta 3.0: {len(sym):,} human mitochondrial genes, "
          f"{len(paths):,} MitoPathways nodes")
    print(f"  sub-mitochondrial: {dict(collections.Counter(subloc.values()).most_common(6))}")

    # genes under the "Mitochondrial central dogma" branch = the apparatus for the separate genome
    cd = set()
    for _, r in paths.iterrows():
        h = str(r.get("MitoPathways Hierarchy") or "")
        g = r.get("Genes")
        if h.startswith("Mitochondrial central dogma") and pd.notna(g):
            cd |= {x.strip() for x in str(g).replace(",", " ").split() if x.strip()}
    cd &= sym
    other = sym - cd
    print(f"  central dogma (serves the separate genome): {len(cd):,}; other mitochondrial: {len(other):,}")

    Z, idx = load_depmap()
    allg = sorted(idx)
    rng = np.random.default_rng(0)

    print(f"\n  CO-ESSENTIALITY TIGHTNESS above size-matched random sets")
    res = {}
    for label, members in (("MitoCarta central dogma", cd), ("MitoCarta other mitochondrial", other),
                           ("MitoCarta all", sym)):
        d, n = scored(Z, idx, members, rng, allg)
        res[label] = {"delta": d, "n": n}
        print(f"    {label:<32} n={n:<5,} delta {d:+.4f}")
    for loc in ("Matrix", "MIM", "MOM", "IMS"):
        mem = {g for g, l in subloc.items() if l == loc}
        d, n = scored(Z, idx, mem, rng, allg)
        res[f"subloc:{loc}"] = {"delta": d, "n": n}
        print(f"    sub-mito {loc:<23} n={n:<5,} delta {d:+.4f}")

    cdd, othd = res["MitoCarta central dogma"]["delta"], res["MitoCarta other mitochondrial"]["delta"]
    print(f"\n    central dogma minus other mitochondrial: {cdd - othd:+.4f}")
    print(f"    for reference, measured earlier: peroxisome +0.0297, lysosome +0.0240, cytosol +0.0232 -- "
          f"membrane-bounded isolation alone does not produce tightness")

    # MitoCarta as experimental ground truth for the compartment assignment built from annotation
    comp = json.loads((OUT / "compartments.json").read_text())["steps"]
    net = json.loads((OUT / "reaction_network.json").read_text())
    gene_c = collections.defaultdict(collections.Counter)
    for s in net["steps"]:
        cs = comp.get(s["id"], {}).get("compartments", [])
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        for g in gs:
            for c in cs:
                gene_c[g][c] += 1
    called = {g for g, c in gene_c.items() if c and c.most_common(1)[0][0] == "mitochondrion"}
    known = set(gene_c) & sym
    tp = len(called & sym)
    print(f"\n  COMPARTMENT ASSIGNMENT vs MITOCARTA (experimental: MS across 14 tissues, GFP, targeting signals)")
    print(f"    genes in the reaction network with a location : {len(gene_c):,}")
    print(f"    called mitochondrial by our annotation         : {len(called):,}")
    print(f"    of those, confirmed by MitoCarta               : {tp:,}  precision {tp/max(len(called),1):.3f}")
    print(f"    MitoCarta genes present in our network         : {len(known):,}  "
          f"recall {tp/max(len(known),1):.3f}")

    # dark genes: MitoCarta is a second, experimental opinion on the ones we located to the mitochondrion
    dark = json.loads((OUT / "dark_function_table.json").read_text())["table"]
    dl = json.loads((OUT / "dark_gene_location.json").read_text())
    dark_mito = {r["gene"] for r in dl["rows"] if "mitochondrion" in r["gene_loc"]}
    print(f"\n  DARK GENES: {len(set(dark) & sym):,} of {len(dark):,} are MitoCarta members")
    print(f"    of the dark genes our UniProt mapping put in the mitochondrion ({len(dark_mito):,}), "
          f"MitoCarta confirms {len(dark_mito & sym):,}")

    json.dump({"n_mitocarta": len(sym), "n_central_dogma": len(cd), "n_other_mito": len(other),
               "subloc_counts": dict(collections.Counter(subloc.values())),
               "tightness": res, "central_dogma_minus_other": cdd - othd,
               "assignment_precision": tp / max(len(called), 1),
               "assignment_recall": tp / max(len(known), 1),
               "dark_in_mitocarta": len(set(dark) & sym),
               "dark_mito_confirmed": len(dark_mito & sym)},
              open(OUT / "mitocarta.json", "w"))
    print(f"\n  -> {OUT/'mitocarta.json'}")


if __name__ == "__main__":
    main()
