"""pathway_remove — REMOVE a protein from the 5-pathway crosstalk graph and redraw it, showing exactly what changes
(viz/pathway_remove.html). The removed node is struck out (red x), its edges are cut (dashed red), and any co-subunit that
loses it as a complex scaffold/partner is flagged (amber ring). Prints the diff: edges cut, inter-pathway links lost,
hand-off bridges reduced, complexes disassembled.

HONEST SCOPE: this is the DIRECT / STRUCTURAL rewiring only, which the graph predicts for certain (deleting a node deletes
its measured PPI edges; complex-mates that lose a subunit are co-essential ~1000x chance). It does NOT show the downstream
transcriptional cascade — the graph predicts a knockout's real movers at only ~3% recall (4x enriched, misses ~97%), so
the ripple is deliberately not drawn. Usage: python3 colab/pathway_remove.py [GENE]   (default PPP2R1A).
"""
import os
import json, html, sys, collections
from pathlib import Path
import pathway_graph as pg
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
VIZ = Path("viz")


def _affected(removed, graph_genes):
    """co-subunits of the removed protein's complexes that are ALSO in the graph — they lose their assembly partner."""
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    n2i = {x: i for i, x in enumerate(names)}
    g2c, plx = D.get("gene2cplx", {}), D.get("complexes", {})
    i = n2i.get(removed)
    cxs = g2c.get(str(i), []) if i is not None else []
    mates = set()
    for cn in cxs:
        for m in plx.get(cn, []):
            if isinstance(m, int) and m < len(names):
                mates.add(names[m])
    mates.discard(removed)
    return {m for m in mates if m in graph_genes}, len(cxs)


def render(removed):
    cols, nodes, edges = pg.build_graph()
    graph_genes = {n["g"] for n in nodes}
    REMOVED = {removed}
    AFFECTED, ncx = _affected(removed, graph_genes)
    W, H, PX0, PX1, PY0, PY1 = 1200, 860, 70, 1130, 150, 760
    col_x, bandh = pg._layout(nodes, PX0, PX1, PY0, PY1)
    byname = {n["g"]: n for n in nodes}
    rem_edges = [e for e in edges if e["a"] in REMOVED or e["b"] in REMOVED]
    rem_inter = [e for e in rem_edges if e["inter"]]
    # find which two pathways the removed node bridged, and the shared-bridge before/after for that pair
    rn = byname.get(removed)
    bridge = ""
    if rn and len(rn["pw"]) >= 2:
        a, b = rn["pw"][0], rn["pw"][1]
        before = [n["g"] for n in nodes if a in n["pw"] and b in n["pw"]]
        after = [g for g in before if g not in REMOVED]
        bridge = f"{a} &harr; {b}: bridges {len(before)} &rarr; {len(after)} ({', '.join(before)} &rarr; {', '.join(after) or 'none'})"
    esc = lambda s: html.escape(str(s))
    svg = []
    for d in range(4):
        y0 = PY0 + d * bandh
        svg.append(f'<rect x="{PX0-40}" y="{y0:.0f}" width="{PX1-PX0+80}" height="{bandh:.0f}" class="band band{d}"/>')
        svg.append(f'<text x="{PX0-46}" y="{y0+16:.0f}" class="bandlab">{pg.BANDS[d]}</text>')
    for k, c in enumerate(cols):
        svg.append(f'<text x="{col_x[k]:.0f}" y="{PY0-40:.0f}" class="colhdr" fill="{pg.PCOL[k]}">{esc(c)}</text>')
        svg.append(f'<line x1="{col_x[k]:.0f}" y1="{PY0-30:.0f}" x2="{col_x[k]:.0f}" y2="{PY1+8:.0f}" class="colrule"/>')
    for e in edges:
        a, b = byname[e["a"]], byname[e["b"]]
        cls = "cut" if (e["a"] in REMOVED or e["b"] in REMOVED) else ("edge inter" if e["inter"] else "edge intra")
        svg.append(f'<line x1="{a["x"]:.1f}" y1="{a["y"]:.1f}" x2="{b["x"]:.1f}" y2="{b["y"]:.1f}" class="{cls}"/>')
    for n in nodes:
        g = n["g"]
        col = pg.SHARED if n["shared"] else pg.PCOL[n["col"][0]]
        r = 13 if n["shared"] else 10
        if g in REMOVED:
            svg.append(f'<g class="rem"><title>{esc(g)} — REMOVED</title>'
                       f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}"/>'
                       f'<line x1="{n["x"]-r:.1f}" y1="{n["y"]-r:.1f}" x2="{n["x"]+r:.1f}" y2="{n["y"]+r:.1f}"/>'
                       f'<line x1="{n["x"]-r:.1f}" y1="{n["y"]+r:.1f}" x2="{n["x"]+r:.1f}" y2="{n["y"]-r:.1f}"/>'
                       f'<text x="{n["x"]:.1f}" y="{n["y"]+r+11:.1f}" class="glab remlab">{esc(g)}</text></g>')
        elif g in AFFECTED:
            svg.append(f'<g class="aff"><title>{esc(g)} — loses its complex partner {esc(removed)}</title>'
                       f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}" fill="{col}"/>'
                       f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r+4}" class="affring"/>'
                       f'<text x="{n["x"]:.1f}" y="{n["y"]+r+11:.1f}" class="glab afflab">{esc(g)} ⚠</text></g>')
        else:
            svg.append(f'<g class="node{" shared" if n["shared"] else ""}"><title>{esc(g)}</title>'
                       f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}" fill="{col}"/>'
                       f'<text x="{n["x"]:.1f}" y="{n["y"]+r+11:.1f}" class="glab">{esc(g)}</text></g>')
    page = _TEMPLATE.format(removed=esc(removed), svg="\n".join(svg), W=W, H=H, ncut=len(rem_edges),
                            ninter=len(rem_inter), ncx=ncx, aff=", ".join(sorted(AFFECTED)) or "none in this graph",
                            bridge=bridge or "(not a shared hand-off node)")
    VIZ.mkdir(exist_ok=True)
    (VIZ / "pathway_remove.html").write_text(page)
    return removed, len(rem_edges), len(rem_inter), ncx, AFFECTED, bridge


_TEMPLATE = '''<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c2c9d4; --inter:#5b6472; --cut:#d1495b; --aff:#e08a2b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#8892a3; --cut:#e06c82; --aff:#e6a24a; }} }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6; --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c2c9d4; --inter:#5b6472; --cut:#d1495b; --aff:#e08a2b; }}
:root[data-theme="dark"] {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d; --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#8892a3; --cut:#e06c82; --aff:#e6a24a; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1240px;margin:0 auto;padding:28px 20px 48px}} h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--muted);font-size:13.5px;margin:0 0 16px;max-width:900px;line-height:1.5}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px;overflow-x:auto}}
svg{{display:block;width:100%;height:auto;min-width:940px}}
.band0{{fill:var(--band0)}} .band1{{fill:var(--band1)}} .band2{{fill:var(--band2)}} .band3{{fill:var(--band3)}}
.bandlab{{fill:var(--muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em}}
.colhdr{{font-size:13px;font-weight:700;text-anchor:middle}} .colrule{{stroke:var(--line);stroke-width:1;stroke-dasharray:2 5}}
.edge{{fill:none}} .edge.intra{{stroke:var(--edge);stroke-width:1;opacity:.5}} .edge.inter{{stroke:var(--inter);stroke-width:1.8;opacity:.75}}
.cut{{stroke:var(--cut);stroke-width:1.8;opacity:.85;stroke-dasharray:4 3}}
.node circle{{stroke:var(--panel);stroke-width:2}} .node.shared circle{{stroke:#caa43a;stroke-width:2.5}}
.rem circle{{fill:none;stroke:var(--cut);stroke-width:2.5;opacity:.5}} .rem line{{stroke:var(--cut);stroke-width:2.5}}
.remlab{{fill:var(--cut);font-weight:700;text-decoration:line-through}}
.aff circle{{stroke:var(--panel);stroke-width:2}} .affring{{fill:none;stroke:var(--aff);stroke-width:2;stroke-dasharray:3 2}} .afflab{{fill:var(--aff);font-weight:700}}
.glab{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;fill:var(--ink);text-anchor:middle}}
.foot{{color:var(--muted);font-size:12px;line-height:1.55;margin-top:16px;max-width:960px}} .foot b{{color:var(--ink);font-weight:600}} .stats{{font-family:ui-monospace,monospace;color:var(--ink)}}
</style>
<div class="wrap">
  <h1>Remove {removed} &mdash; how the 5&#8209;pathway graph changes</h1>
  <p class="sub">The five&#8209;pathway crosstalk graph with <b style="color:var(--cut)">{removed} removed</b> (red &times;,
  its edges cut in dashed red). Any complex co&#8209;subunit that loses {removed} is flagged with an amber ring &#9888;.</p>
  <div class="card"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Five-pathway graph with a protein removed">{svg}</svg></div>
  <p class="foot">
    <b>What changes (direct, from the graph).</b> Removing <span class="stats">{removed}</span> deletes <b>{ncut} edges</b>
    ({ninter} inter&#8209;pathway). Hand&#8209;off: {bridge}. <b>Structural knock&#8209;on</b> (validated, ~1000&times; complex
    co&#8209;essentiality): {removed} scaffolds {ncx} complex(es), so these co&#8209;subunits lose their partner even though
    their genes are untouched: <span class="stats">{aff}</span>.<br>
    <b>What does NOT change here (the wall).</b> This is the <b>direct / structural</b> rewiring only. The downstream
    transcriptional <b>cascade</b> is <b>not</b> shown &mdash; the graph predicts a knockout's real movers at only ~3%
    recall (4&times; enriched, misses ~97%). We redraw the wiring we can see for certain, and stop at the ripple we can't.
  </p>
</div>'''


def main():
    removed = sys.argv[1] if len(sys.argv) > 1 else "PPP2R1A"
    g, ncut, ninter, ncx, aff, bridge = render(removed)
    print(f"remove {g}: cut {ncut} edges ({ninter} inter-pathway); scaffolds {ncx} complex(es); "
          f"co-subunits orphaned in graph: {sorted(aff) or 'none'}")
    print(f"  hand-off change: {bridge or '(not a shared node)'}")
    print(f"  -> {VIZ/'pathway_remove.html'}")


if __name__ == "__main__":
    main()
