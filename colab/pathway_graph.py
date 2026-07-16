"""pathway_graph — render 5 signalling pathways as ONE compartment-layered crosstalk graph (viz/pathway_crosstalk.html).
Built entirely from the cell model's real data: proteins of 5 Reactome pathways, positioned TOP-TO-DOWN by compartment
(localization: extracellular -> membrane -> cytoplasm -> nucleus) and INTERCONNECTED by measured physical interactions
(cell_complete["ppi"]). Proteins that belong to TWO of the 5 pathways are the hand-off nodes (highlighted) -- the literal
points where one pathway's proteins feed the next.

HONEST SCOPE: vertical position = OBSERVED compartment (not a proven flow -- spatial_ladder measured that regulatory tier
does not track depth); edges are MEASURED but UNDIRECTED physical interactions (real crosstalk / shared machinery, NOT a
proven directional signal hand-off); the downward arrow marks the TEXTBOOK cascade direction for these specific pathways,
not something derived from the edges. A descriptive near-field map -- who touches whom and where -- not a phenotype
predictor.
"""
import json, html, collections
from pathlib import Path
OUT = Path("outputs/orphan")
VIZ = Path("viz")

PATHWAYS = ["PI3K Cascade", "Signaling by EGFR", "MAP2K and MAPK activation",
            "MAPK targets/ Nuclear events mediated by MAP kinases", "Signaling by WNT in cancer"]
SHORT = {"PI3K Cascade": "PI3K/AKT", "Signaling by EGFR": "EGFR receptor",
         "MAP2K and MAPK activation": "MAPK activation",
         "MAPK targets/ Nuclear events mediated by MAP kinases": "MAPK nuclear targets",
         "Signaling by WNT in cancer": "WNT"}
PCOL = ["#2ca6a4", "#7c6cd6", "#e08a2b", "#d1495b", "#4c9f70"]
SHARED = "#caa43a"
DEPTH = {"Secreted": 0, "Extracellular": 0, "Extracellular space": 0, "Cell membrane": 1, "Membrane": 1, "Cell surface": 1,
         "Cytoplasm": 2, "Cytosol": 2, "Endoplasmic reticulum": 2, "Golgi apparatus": 2, "Cytoskeleton": 2, "Endosome": 2,
         "Nucleus": 3}
BANDS = ["extracellular", "membrane", "cytoplasm", "nucleus"]
PER = 11                                                          # connective proteins shown per pathway (plus all shared)


def build_graph():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    loc = json.load(open(OUT / "localization.json")).get("labels", {})
    paths = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if a < len(names) and b < len(names):
            ppi[a].add(b); ppi[b].add(a)

    def depth(g):
        ds = [DEPTH[c] for c in (loc.get(g) or []) if c in DEPTH]
        return min(ds) if ds else 2                              # OUTERMOST compartment for the top-down axis

    setmap = {p: set(paths[p]) for p in PATHWAYS}
    union = set().union(*setmap.values())
    udeg = {i: len(ppi[i] & union) for i in union}
    inpath = collections.defaultdict(set)
    for p in PATHWAYS:
        for i in setmap[p]:
            inpath[i].add(p)
    shared = {i for i in union if len(inpath[i]) >= 2}
    sel = set(shared)
    for p in PATHWAYS:                                            # top-by-connectivity per pathway
        got = [i for i in setmap[p] if i in sel]
        for i in sorted(setmap[p], key=lambda i: -udeg.get(i, 0)):
            if len(got) >= PER:
                break
            if i not in sel:
                sel.add(i); got.append(i)
    nodes = []
    for i in sel:
        pw = sorted(inpath[i], key=lambda p: PATHWAYS.index(p))
        nodes.append({"g": names[i], "pw": [SHORT[p] for p in pw], "col": [PATHWAYS.index(p) for p in pw],
                      "depth": depth(names[i]), "shared": len(pw) >= 2})
    seen, edges = set(), []
    for a in sel:
        for b in ppi[a] & sel:
            if a < b and (a, b) not in seen:
                seen.add((a, b))
                edges.append({"a": names[a], "b": names[b], "inter": not (inpath[a] & inpath[b])})
    return [SHORT[p] for p in PATHWAYS], nodes, edges


def _layout(nodes, PX0, PX1, PY0, PY1):
    ncol, nband = len(PATHWAYS), len(BANDS)
    colw = (PX1 - PX0) / ncol
    col_x = [PX0 + (k + 0.5) * colw for k in range(ncol)]
    bandh = (PY1 - PY0) / nband
    band_y = [PY0 + d * bandh + bandh / 2 for d in range(nband)]
    for n in nodes:
        n["x"] = sum(col_x[c] for c in n["col"]) / len(n["col"])
    OFFS, MINDX = [0, -46, 46, -92, 92, -138, 138], 84
    for d in range(nband):
        band = sorted([n for n in nodes if n["depth"] == d], key=lambda n: n["x"])
        rows = [[] for _ in OFFS]
        for n in band:
            for ri, off in enumerate(OFFS):
                if all(abs(n["x"] - m["x"]) >= MINDX for m in rows[ri]):
                    n["y"] = max(PY0 + 18, min(PY1 - 18, band_y[d] + off)); rows[ri].append(n); break
            else:
                n["y"] = band_y[d]; rows[0].append(n)
    return col_x, bandh


def render(cols, nodes, edges):
    W, H, PX0, PX1, PY0, PY1 = 1200, 860, 70, 1130, 150, 760
    col_x, bandh = _layout(nodes, PX0, PX1, PY0, PY1)
    byname = {n["g"]: n for n in nodes}
    esc = lambda s: html.escape(str(s))
    svg = []
    for d in range(len(BANDS)):
        y0 = PY0 + d * bandh
        svg.append(f'<rect x="{PX0-40}" y="{y0:.0f}" width="{PX1-PX0+80}" height="{bandh:.0f}" class="band band{d}"/>')
        svg.append(f'<text x="{PX0-46}" y="{y0+16:.0f}" class="bandlab">{BANDS[d]}</text>')
    for k, c in enumerate(cols):
        svg.append(f'<text x="{col_x[k]:.0f}" y="{PY0-40:.0f}" class="colhdr" fill="{PCOL[k]}">{esc(c)}</text>')
        svg.append(f'<line x1="{col_x[k]:.0f}" y1="{PY0-30:.0f}" x2="{col_x[k]:.0f}" y2="{PY1+8:.0f}" class="colrule"/>')
    for e in edges:
        a, b = byname.get(e["a"]), byname.get(e["b"])
        if a and b:
            svg.append(f'<line x1="{a["x"]:.1f}" y1="{a["y"]:.1f}" x2="{b["x"]:.1f}" y2="{b["y"]:.1f}" '
                       f'class="edge {"inter" if e["inter"] else "intra"}"/>')
    for n in nodes:
        col = SHARED if n["shared"] else PCOL[n["col"][0]]
        r = 13 if n["shared"] else 10
        tip = f"{n['g']} — {' + '.join(n['pw'])} · {BANDS[n['depth']]}"
        svg.append(f'<g class="node{" shared" if n["shared"] else ""}"><title>{esc(tip)}</title>'
                   f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}" fill="{col}"/>'
                   f'<text x="{n["x"]:.1f}" y="{n["y"]+r+11:.1f}" class="glab">{esc(n["g"])}</text></g>')
    svg.append('<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="4" refY="7" orient="auto">'
               '<path d="M1,1 L4,7 L7,1" class="arrhead"/></marker></defs>')
    svg.append(f'<line x1="30" y1="{PY0+10}" x2="30" y2="{PY1-6}" class="dirarrow" marker-end="url(#arr)"/>')
    mid = (PY0 + PY1) / 2
    svg.append(f'<text x="20" y="{mid:.0f}" class="dirlab" transform="rotate(-90 20 {mid:.0f})">'
               'canonical signal direction  →  receptor to nucleus</text>')
    nsh = sum(1 for n in nodes if n["shared"])
    intra = sum(1 for e in edges if not e["inter"]); inter = sum(1 for e in edges if e["inter"])
    legend = "".join(f'<span class="lg"><i style="background:{PCOL[k]}"></i>{esc(c)}</span>' for k, c in enumerate(cols))
    page = _TEMPLATE.format(SHARED=SHARED, legend=legend, svg="\n".join(svg), W=W, H=H,
                            nnodes=len(nodes), nsh=nsh, intra=intra, inter=inter)
    VIZ.mkdir(exist_ok=True)
    (VIZ / "pathway_crosstalk.html").write_text(page)
    return len(nodes), len(edges), nsh, intra, inter


_TEMPLATE = '''<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c2c9d4; --inter:#5b6472; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#8892a3; }} }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c2c9d4; --inter:#5b6472; }}
:root[data-theme="dark"] {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#8892a3; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
.wrap {{ max-width:1240px; margin:0 auto; padding:28px 20px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }}
.sub {{ color:var(--muted); font-size:13.5px; margin:0 0 18px; max-width:900px; line-height:1.5; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px 18px; align-items:center; margin:0 0 14px; font-size:12.5px; }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }}
.lg i {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
.lg.ring i {{ background:{SHARED}; box-shadow:0 0 0 2px var(--panel),0 0 0 3.5px {SHARED}; }}
.lg.e span {{ display:inline-block; width:22px; border-top:2px solid var(--edge); }}
.lg.e.inter span {{ border-top:2.5px solid var(--inter); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:6px; overflow-x:auto; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
svg {{ display:block; width:100%; height:auto; min-width:940px; }}
.band0 {{ fill:var(--band0); }} .band1 {{ fill:var(--band1); }} .band2 {{ fill:var(--band2); }} .band3 {{ fill:var(--band3); }}
.bandlab {{ fill:var(--muted); font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; }}
.colhdr {{ font-size:13px; font-weight:700; text-anchor:middle; }}
.colrule {{ stroke:var(--line); stroke-width:1; stroke-dasharray:2 5; }}
.edge {{ fill:none; }} .edge.intra {{ stroke:var(--edge); stroke-width:1; opacity:.55; }}
.edge.inter {{ stroke:var(--inter); stroke-width:1.8; opacity:.8; }}
.node circle {{ stroke:var(--panel); stroke-width:2; }} .node.shared circle {{ stroke:{SHARED}; stroke-width:2.5; }}
.glab {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:10px; fill:var(--ink); text-anchor:middle; }}
.dirarrow {{ stroke:var(--muted); stroke-width:1.5; fill:none; opacity:.6; }}
.arrhead {{ fill:none; stroke:var(--muted); stroke-width:1.5; }}
.dirlab {{ fill:var(--muted); font-size:10.5px; letter-spacing:.06em; text-anchor:middle; }}
.foot {{ color:var(--muted); font-size:12px; line-height:1.55; margin-top:16px; max-width:960px; }}
.foot b {{ color:var(--ink); font-weight:600; }} .stats {{ font-family:ui-monospace,monospace; color:var(--ink); }}
</style>
<div class="wrap">
  <h1>Five signaling pathways, connected</h1>
  <p class="sub">A crosstalk map from the cell model's real data: proteins of five Reactome pathways, laid out
  <b>top&#8209;to&#8209;down by compartment</b> (where each protein acts) and <b>interconnected by measured physical
  interactions</b> (PPI). Gold ringed nodes belong to <b>two</b> pathways at once &mdash; the hand&#8209;off points.</p>
  <div class="legend">{legend}
    <span class="lg ring"><i></i>shared / hand&#8209;off (in 2 pathways)</span>
    <span class="lg e"><span></span>PPI within a pathway</span>
    <span class="lg e inter"><span></span>PPI across pathways</span>
  </div>
  <div class="card">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Layered crosstalk graph of five signaling pathways by compartment">
      {svg}
    </svg>
  </div>
  <p class="foot">
    <b>How to read it.</b> Columns are the five pathways; rows are compartments from outside (top) to nucleus (bottom).
    Signaling <b>canonically</b> runs downward &mdash; a membrane receptor (EGFR) relays through the RAS proteins
    (<span class="stats">HRAS, KRAS, NRAS, GRB2</span>) into both the MAPK and PI3K arms, converges on ERK
    (<span class="stats">MAPK1/MAPK3</span>), and ends at nuclear targets; WNT crosstalks near the nucleus via PP2A
    (<span class="stats">PPP2*</span>). <b>{nnodes} proteins</b> shown ({nsh} shared), <b>{intra}</b> within&#8209;pathway
    and <b>{inter}</b> across&#8209;pathway PPI edges.<br>
    <b>Honest scope.</b> Vertical position = <b>observed compartment</b>, not a proven flow; edges are <b>measured but
    undirected</b> physical interactions &mdash; real crosstalk and shared machinery, <b>not</b> a proven directional
    hand&#8209;off. The arrow marks the <b>textbook</b> cascade direction for these pathways, not something derived from
    the edges. A descriptive near&#8209;field map, not a phenotype predictor.
  </p>
</div>'''


def main():
    cols, nodes, edges = build_graph()
    nn, ne, nsh, intra, inter = render(cols, nodes, edges)
    print(f"pathway crosstalk graph: {nn} nodes ({nsh} shared), {ne} edges ({intra} intra / {inter} inter)")
    print(f"  pathways: {', '.join(cols)}")
    print(f"  -> {VIZ/'pathway_crosstalk.html'}")


if __name__ == "__main__":
    main()
