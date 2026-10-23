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
import os
import json, html, collections, math
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
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


_RADIAL_TEMPLATE = '''<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --r0:#eae6f2; --r1:#e7edf4; --r2:#e6f0ef; --r3:#f2eef0; --edge:#c2c9d4; --inter:#5b6472; --core:#2b3448; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --r0:#1b1526; --r1:#111a26; --r2:#0f1c1b; --r3:#1c1620; --edge:#2c3644; --inter:#8892a3; --core:#20283a; }} }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --r0:#eae6f2; --r1:#e7edf4; --r2:#e6f0ef; --r3:#f2eef0; --edge:#c2c9d4; --inter:#5b6472; --core:#2b3448; }}
:root[data-theme="dark"] {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --r0:#1b1526; --r1:#111a26; --r2:#0f1c1b; --r3:#1c1620; --edge:#2c3644; --inter:#8892a3; --core:#20283a; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1120px;margin:0 auto;padding:28px 20px 48px}}
h1{{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--muted);font-size:13.5px;margin:0 0 16px;max-width:880px;line-height:1.5}}
.legend{{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:center;margin:0 0 12px;font-size:12.5px}}
.lg{{display:inline-flex;align-items:center;gap:6px}} .lg i{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.lg.ring i{{background:{SHARED};box-shadow:0 0 0 2px var(--panel),0 0 0 3.5px {SHARED}}}
.lg.e span{{display:inline-block;width:22px;border-top:2px solid var(--edge)}} .lg.e.inter span{{border-top:2.5px solid var(--inter)}}
.lg.sp span{{display:inline-block;width:22px;border-top:1.5px dashed var(--muted);opacity:.7}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px;overflow-x:auto;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
svg{{display:block;width:100%;height:auto;min-width:860px}}
.ring0{{fill:var(--r0)}} .ring1{{fill:var(--r1)}} .ring2{{fill:var(--r2)}} .ring3{{fill:var(--r3)}}
.ring{{stroke:var(--line);stroke-width:1}}
.ringlab{{fill:var(--muted);font-size:11px;font-weight:600;text-anchor:middle;text-transform:uppercase;letter-spacing:.08em}}
.core{{fill:var(--core);stroke:{SHARED};stroke-width:1.5}}
.corelab{{fill:#fff;font-size:15px;font-weight:700;text-anchor:middle;letter-spacing:.12em}}
.coresub{{fill:#cfd6e4;font-size:8.5px;text-anchor:middle}}
.spoke{{stroke:var(--muted);stroke-width:.7;opacity:.22;stroke-dasharray:2 3}}
.edge{{fill:none}} .edge.intra{{stroke:var(--edge);stroke-width:1;opacity:.5}} .edge.inter{{stroke:var(--inter);stroke-width:1.8;opacity:.8}}
.node circle{{stroke:var(--panel);stroke-width:2}} .node.shared circle{{stroke:{SHARED};stroke-width:2.5}}
.glab{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:9.5px;fill:var(--ink);text-anchor:middle}}
.sect{{font-size:12.5px;font-weight:700}}
.foot{{color:var(--muted);font-size:12px;line-height:1.55;margin-top:16px;max-width:940px}} .foot b{{color:var(--ink);font-weight:600}}
.stats{{font-family:ui-monospace,monospace;color:var(--ink)}}
</style>
<div class="wrap">
  <h1>Rooted at the genome &mdash; five pathways branching outward</h1>
  <p class="sub">The same five pathways, but <b>rooted at the nucleus</b>: every protein is expressed from a gene at the
  centre &mdash; the one origin we're <i>certain</i> of. Rings branch <b>outward</b> from the genome through the cytoplasm
  to the membrane and outside, following where each protein is deployed. This is the <b>biogenesis</b> direction &mdash;
  the natural complement, and the opposite, of the inward signalling view.</p>
  <div class="legend">{legend}
    <span class="lg ring"><i></i>shared / hand&#8209;off</span>
    <span class="lg e"><span></span>PPI within pathway</span>
    <span class="lg e inter"><span></span>PPI across pathways</span>
    <span class="lg sp"><span></span>traces back to its gene</span>
  </div>
  <div class="card">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Radial graph of five pathways rooted at the genome, branching outward by compartment">
      {svg}
    </svg>
  </div>
  <p class="foot">
    <b>Why root at the nucleus.</b> Reading outward is the certain direction: DNA in the nucleus is transcribed, the
    protein is made and trafficked out to where it works. The dashed spokes make it literal &mdash; every node traces
    back to its gene in the genome core. The RAS relay (<span class="stats">HRAS/KRAS/NRAS/GRB2</span>) and ERK
    (<span class="stats">MAPK1/MAPK3</span>) sit on the sector boundaries as the shared hand&#8209;off proteins.
    <b>{nnodes}</b> proteins ({nsh} shared), <b>{intra}</b> within&#8209;pathway + <b>{inter}</b> across&#8209;pathway PPI edges.<br>
    <b>Honest scope.</b> Ring = <b>observed compartment</b>; outward = the <b>biogenesis</b> sense (made from nuclear
    genes, deployed out) &mdash; a real material&#8209;flow direction, but the PPI edges are still <b>measured and
    undirected</b> (crosstalk, not a per&#8209;protein production tree), and the dashed spokes mark the central&#8209;dogma
    origin, not {nnodes} distinct measured edges. Descriptive near&#8209;field map, not a phenotype predictor.
  </p>
</div>'''


def render_radial(cols, nodes, edges):
    """the NUCLEUS-ROOTED view: genome at the centre (every protein is expressed from a gene there -- the one origin we
    are certain of), rings branching OUTWARD by compartment (nucleus -> cytoplasm -> membrane -> extracellular), pathways
    as angular sectors. Outward = the biogenesis direction (made from nuclear genes, deployed out), the complement/opposite
    of the inward signalling view. Edges are still measured, undirected PPI."""
    W, H, cx, cy = 1060, 1120, 530, 600
    LEVELR = {0: 150, 1: 250, 2: 340, 3: 415}                    # radius by level (center outward); level = 3-depth
    RINGSEP = [200, 297, 380, 450]
    LEVELNAME = ["nucleus", "cytoplasm", "membrane", "extracellular"]
    nw, ARC, START = len(cols), 300.0, -90.0
    centers = [START + (k + 0.5) / nw * ARC for k in range(nw)]
    HW = (ARC / nw) / 2 * 0.72
    lvl = lambda n: 3 - n["depth"]
    polar = lambda r, ad: (cx + r * math.cos(math.radians(ad)), cy + r * math.sin(math.radians(ad)))
    g_single, g_shared = collections.defaultdict(list), collections.defaultdict(list)
    for n in nodes:
        (g_shared if n["shared"] else g_single)[(tuple(sorted(n["col"])) if n["shared"] else n["col"][0], lvl(n))].append(n)
    for (k, lv), ns in g_single.items():
        m = len(ns)
        for i, n in enumerate(sorted(ns, key=lambda x: x["g"])):
            ang = centers[k] - HW + (0.5 if m == 1 else i / (m - 1)) * 2 * HW
            r = LEVELR[lv] + ((i % 2) * 2 - 1) * (9 if m > 3 else 0)
            n["x"], n["y"] = polar(r, ang); n["ang"] = ang
    for (ck, lv), ns in g_shared.items():
        ca = (centers[ck[0]] + centers[ck[1]]) / 2               # boundary angle between the two wedges
        m = len(ns)
        for i, n in enumerate(sorted(ns, key=lambda x: x["g"])):
            n["x"], n["y"] = polar(LEVELR[lv] + (i - (m - 1) / 2) * 20, ca); n["ang"] = ca
    byname = {n["g"]: n for n in nodes}
    esc = lambda s: html.escape(str(s))
    svg = []
    for lv in [3, 2, 1, 0]:
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{RINGSEP[lv]}" class="ring ring{lv}"/>')
    for lv in range(4):
        svg.append(f'<text x="{cx}" y="{cy-RINGSEP[lv]+(16 if lv else 22):.0f}" class="ringlab">{LEVELNAME[lv]}</text>')
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="70" class="core"/>')
    svg.append(f'<text x="{cx}" y="{cy-4}" class="corelab">GENOME</text>')
    svg.append(f'<text x="{cx}" y="{cy+13}" class="coresub">every protein starts as a gene here</text>')
    for n in nodes:                                              # faint biogenesis spokes back to the genome core
        x2, y2 = polar(66, n["ang"])
        svg.append(f'<line x1="{n["x"]:.1f}" y1="{n["y"]:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="spoke"/>')
    for e in edges:
        a, b = byname.get(e["a"]), byname.get(e["b"])
        if a and b:
            svg.append(f'<line x1="{a["x"]:.1f}" y1="{a["y"]:.1f}" x2="{b["x"]:.1f}" y2="{b["y"]:.1f}" '
                       f'class="edge {"inter" if e["inter"] else "intra"}"/>')
    for k, c in enumerate(cols):
        lx, ly = polar(475, centers[k])
        svg.append(f'<text x="{lx:.0f}" y="{ly:.0f}" class="sect" fill="{PCOL[k]}" text-anchor="middle">{esc(c)}</text>')
    for n in nodes:
        col = SHARED if n["shared"] else PCOL[n["col"][0]]
        r = 12 if n["shared"] else 9
        tip = f"{n['g']} — {' + '.join(n['pw'])} · {BANDS[n['depth']]}"
        svg.append(f'<g class="node{" shared" if n["shared"] else ""}"><title>{esc(tip)}</title>'
                   f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}" fill="{col}"/>'
                   f'<text x="{n["x"]:.1f}" y="{n["y"]-r-4:.1f}" class="glab">{esc(n["g"])}</text></g>')
    nsh = sum(1 for n in nodes if n["shared"])
    intra = sum(1 for e in edges if not e["inter"]); inter = len(edges) - intra
    legend = "".join(f'<span class="lg"><i style="background:{PCOL[k]}"></i>{esc(c)}</span>' for k, c in enumerate(cols))
    page = _RADIAL_TEMPLATE.format(SHARED=SHARED, legend=legend, svg="\n".join(svg), W=W, H=H,
                                   nnodes=len(nodes), nsh=nsh, intra=intra, inter=inter)
    VIZ.mkdir(exist_ok=True)
    (VIZ / "pathway_radial.html").write_text(page)
    return len(nodes), len(edges), nsh, intra, inter


def main():
    cols, nodes, edges = build_graph()
    nn, ne, nsh, intra, inter = render(cols, nodes, edges)
    render_radial(cols, [dict(n) for n in nodes], edges)         # fresh node dicts (radial mutates x/y/ang)
    print(f"pathway crosstalk graph: {nn} nodes ({nsh} shared), {ne} edges ({intra} intra / {inter} inter)")
    print(f"  pathways: {', '.join(cols)}")
    print(f"  -> {VIZ/'pathway_crosstalk.html'}  (top-to-down, signalling inward)")
    print(f"  -> {VIZ/'pathway_radial.html'}     (nucleus-rooted, branching outward)")


if __name__ == "__main__":
    main()
