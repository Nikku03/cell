"""cell_pathway_map — the WHOLE-CELL pathway interconnection map (viz/cell_pathway_map.html). Instead of 5 hand-picked
pathways, this takes the ~25 TOP-LEVEL Reactome systems (Signal Transduction, Metabolism, Gene expression, Cell Cycle,
Immune, ...) -- the honest coarse resolution for "the whole cell" -- and draws how they INTERCONNECT: an edge between two
systems = the proteins they SHARE (Jaccard of membership), corroborated by cross-PPI (proteins of one physically binding
proteins of the other). A circular meta-network: node size = gene count, chord thickness = interconnection strength.

Why top-level, not all 2,792 pathways: Reactome is hierarchical, most pathways are nested sub-pathways sharing ~all
members, so a raw all-pathway graph is a hairball of trivial parent-child overlaps. The ~25 roots partition the annotated
cell into its major systems, and the overlaps BETWEEN roots are the real cross-system interconnection.

HONEST SCOPE: interconnection = SHARED machinery + physical contact -- DESCRIPTIVE crosstalk, NOT causal signal flow and
NOT directional. The biggest hubs (Signal Transduction, Metabolism, Gene expression) connect broadly partly because they
are huge and full of multifunctional proteins, and Reactome's own annotation choices shape the overlaps. The strongest
links ARE textbook (DNA Repair-Replication-Cell Cycle; Signalling-Development; Vesicle transport-protein modification),
which is the sanity check. A map of who-shares-and-touches-whom across the cell, not a wiring diagram of control, and not
a phenotype predictor. Dark / orphan genes are NOT here (they have no pathway -- that's what connect/orphan_network do).
"""
import json, html, math, collections, itertools
from pathlib import Path
OUT = Path("outputs/orphan")
VIZ = Path("viz")

# top-level Reactome systems (roots present in our data), grouped by super-category and ORDERED around the circle so
# related systems sit adjacent. (category -> colour; order = list order below.)
SYSTEMS = [
    # (reactome name, short label, super-category)
    ("Gene expression (Transcription)", "Gene expression", "expression"),
    ("Metabolism of RNA", "RNA metabolism", "expression"),
    ("Chromatin organization", "Chromatin", "expression"),
    ("DNA Replication", "DNA replication", "genome"),
    ("DNA Repair", "DNA repair", "genome"),
    ("Cell Cycle", "Cell cycle", "genome"),
    ("Reproduction", "Reproduction", "genome"),
    ("Programmed Cell Death", "Cell death", "stress"),
    ("Autophagy", "Autophagy", "stress"),
    ("Cellular responses to stimuli", "Responses to stimuli", "stress"),
    ("Signal Transduction", "Signal transduction", "signaling"),
    ("Developmental Biology", "Development", "signaling"),
    ("Cell-Cell communication", "Cell-cell comm.", "signaling"),
    ("Adaptive Immune System", "Adaptive immunity", "immune"),
    ("Innate Immune System", "Innate immunity", "immune"),
    ("Hemostasis", "Hemostasis", "immune"),
    ("Neuronal System", "Neuronal system", "tissue"),
    ("Sensory Perception", "Sensory perception", "tissue"),
    ("Muscle contraction", "Muscle contraction", "tissue"),
    ("Extracellular matrix organization", "ECM organization", "tissue"),
    ("Vesicle-mediated transport", "Vesicle transport", "trafficking"),
    ("Protein localization", "Protein localization", "trafficking"),
    ("Post-translational protein modification", "Protein modification", "trafficking"),
    ("Organelle biogenesis and maintenance", "Organelle biogenesis", "trafficking"),
    ("Transport of small molecules", "Small-mol. transport", "metabolism"),
    ("Metabolism", "Metabolism", "metabolism"),
]
CATCOL = {"expression": "#7c6cd6", "genome": "#3d7fd1", "stress": "#c65b7c", "signaling": "#2ca6a4",
          "immune": "#d1495b", "tissue": "#b07d48", "trafficking": "#e08a2b", "metabolism": "#4c9f70"}
JTHRESH = 0.08                                                    # show interconnections above this shared-gene Jaccard (strongest only)


def build():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    P = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if a < len(names) and b < len(names):
            ppi[a].add(b); ppi[b].add(a)
    sysrows = [(nm, sh, cat) for nm, sh, cat in SYSTEMS if nm in P]
    S = {nm: set(P[nm]) for nm, _, _ in sysrows}
    nodes = [{"name": nm, "short": sh, "cat": cat, "n": len(S[nm])} for nm, sh, cat in sysrows]
    edges = []
    for (a, _, _), (b, _, _) in itertools.combinations(sysrows, 2):
        inter = len(S[a] & S[b])
        uni = len(S[a] | S[b]) or 1
        jac = inter / uni
        if jac >= JTHRESH:
            oa, ob = S[a] - S[b], S[b] - S[a]
            xppi = sum(1 for x in oa for y in (ppi[x] & ob))
            edges.append({"a": a, "b": b, "jac": jac, "shared": inter, "xppi": xppi})
    covered = len(set().union(*S.values()))
    return nodes, edges, covered


def render(nodes, edges, covered):
    W = H = 1000
    cx, cy, R = 500, 520, 330
    N = len(nodes)
    ang = {n["name"]: -90 + i * 360.0 / N for i, n in enumerate(nodes)}       # degrees, start at top
    pos = {}
    for n in nodes:
        a = math.radians(ang[n["name"]])
        pos[n["name"]] = (cx + R * math.cos(a), cy + R * math.sin(a))
    rad = lambda n: 6 + 24 * math.sqrt(n["n"] / max(x["n"] for x in nodes))
    byname = {n["name"]: n for n in nodes}
    esc = lambda s: html.escape(str(s))
    svg = []
    # chords (behind), thickness + opacity by Jaccard
    jmax = max(e["jac"] for e in edges)
    for e in sorted(edges, key=lambda e: e["jac"]):
        (x1, y1), (x2, y2) = pos[e["a"]], pos[e["b"]]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        cxp, cyp = cx + 0.35 * (mx - cx), cy + 0.35 * (my - cy)               # bow toward centre
        w = 0.6 + 5.0 * (e["jac"] / jmax)
        op = 0.14 + 0.5 * (e["jac"] / jmax)
        tip = f"{byname[e['a']]['short']} ↔ {byname[e['b']]['short']}: {e['shared']} shared proteins (Jaccard {e['jac']:.2f}), {e['xppi']:,} cross-PPI"
        svg.append(f'<path d="M{x1:.1f},{y1:.1f} Q{cxp:.1f},{cyp:.1f} {x2:.1f},{y2:.1f}" class="chord" '
                   f'style="stroke-width:{w:.2f};opacity:{op:.2f}"><title>{esc(tip)}</title></path>')
    # nodes + labels
    for n in nodes:
        x, y = pos[n["name"]]
        r = rad(n)
        col = CATCOL[n["cat"]]
        a = ang[n["name"]]
        lx = cx + (R + r + 12) * math.cos(math.radians(a))
        ly = cy + (R + r + 12) * math.sin(math.radians(a))
        anchor = "start" if math.cos(math.radians(a)) > 0.08 else ("end" if math.cos(math.radians(a)) < -0.08 else "middle")
        svg.append(f'<g class="node"><title>{esc(n["name"])} — {n["n"]:,} genes</title>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{col}"/></g>')
        svg.append(f'<text x="{lx:.1f}" y="{ly+3:.1f}" class="nlab" text-anchor="{anchor}" fill="{col}">{esc(n["short"])}</text>')
    svg_s = "\n".join(svg)
    cats = {}
    for n in nodes:
        cats.setdefault(n["cat"], CATCOL[n["cat"]])
    catnames = {"expression": "gene expression", "genome": "genome / cycle", "stress": "stress / death",
                "signaling": "signalling", "immune": "immune / blood", "tissue": "tissue / structure",
                "trafficking": "protein handling", "metabolism": "metabolism"}
    legend = "".join(f'<span class="lg"><i style="background:{c}"></i>{catnames[k]}</span>' for k, c in cats.items())
    top = sorted(edges, key=lambda e: -e["jac"])[:5]
    toptxt = "; ".join(f'{byname[e["a"]]["short"]}&ndash;{byname[e["b"]]["short"]}' for e in top)
    page = _TEMPLATE.format(legend=legend, svg=svg_s, W=W, H=H, nn=len(nodes), ne=len(edges),
                            covered=f"{covered:,}", toptxt=toptxt, jthr=JTHRESH)
    VIZ.mkdir(exist_ok=True)
    (VIZ / "cell_pathway_map.html").write_text(page)
    return len(nodes), len(edges), covered


_TEMPLATE = '''<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6; --chord:#8892a3; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d; --chord:#9aa4b4; }} }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6; --chord:#8892a3; }}
:root[data-theme="dark"] {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d; --chord:#9aa4b4; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 48px}}
h1{{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}} .sub{{color:var(--muted);font-size:13.5px;margin:0 0 14px;max-width:880px;line-height:1.5}}
.legend{{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin:0 0 10px;font-size:12.5px}}
.lg{{display:inline-flex;align-items:center;gap:6px}} .lg i{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px;overflow-x:auto;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
svg{{display:block;width:100%;height:auto;min-width:720px}}
.chord{{fill:none;stroke:var(--chord)}} .node circle{{stroke:var(--panel);stroke-width:2}}
.nlab{{font-size:11.5px;font-weight:600}}
.foot{{color:var(--muted);font-size:12px;line-height:1.55;margin-top:16px;max-width:940px}} .foot b{{color:var(--ink);font-weight:600}} .stats{{font-family:ui-monospace,monospace;color:var(--ink)}}
</style>
<div class="wrap">
  <h1>The whole cell as interconnected pathways</h1>
  <p class="sub">The cell's biology at its coarsest honest resolution: the <b>{nn} top-level Reactome systems</b> (node
  size = number of genes), linked by how much they <b>interconnect</b> &mdash; a chord's thickness is the fraction of
  proteins two systems <b>share</b> (Jaccard &ge; {jthr}), corroborated by cross&#8209;system PPI. These systems span
  <b>{covered}</b> annotated genes of the cell.</p>
  <div class="legend">{legend}</div>
  <div class="card">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Circular interconnection map of the whole cell's top-level pathway systems">{svg}</svg>
  </div>
  <p class="foot">
    <b>What you're seeing.</b> Each node is a major cell system; a chord means the two systems share machinery. The
    strongest links are textbook &mdash; {toptxt} &mdash; which is the sanity check that the map is real. <b>{ne}</b>
    interconnections above the threshold.<br>
    <b>Honest scope.</b> Interconnection = <b>shared proteins + physical contact</b> &mdash; DESCRIPTIVE crosstalk, <b>not</b>
    causal signal flow and <b>not</b> directional. The biggest hubs (Signal transduction, Metabolism, Gene expression)
    connect broadly partly because they are huge and full of multifunctional proteins, and Reactome's own annotation
    choices shape the overlaps. Dark / orphan genes are <b>not</b> here &mdash; they have no pathway (that's what the
    orphan&#8209;network view adds). A map of who&#8209;shares&#8209;and&#8209;touches&#8209;whom across the cell, not a
    wiring diagram of control.
  </p>
</div>'''


def main():
    nodes, edges, covered = build()
    nn, ne, cov = render(nodes, edges, covered)
    print(f"whole-cell pathway map: {nn} top-level systems, {ne} interconnections (Jaccard >= {JTHRESH}), covering {cov:,} genes")
    for e in sorted(edges, key=lambda e: -e["jac"])[:8]:
        A = next(n["short"] for n in nodes if n["name"] == e["a"]); B = next(n["short"] for n in nodes if n["name"] == e["b"])
        print(f"    J={e['jac']:.2f}  {A} <-> {B}  ({e['shared']} shared, {e['xppi']:,} xPPI)")
    print(f"  -> {VIZ/'cell_pathway_map.html'}")


if __name__ == "__main__":
    main()
