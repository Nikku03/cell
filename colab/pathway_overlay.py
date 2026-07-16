"""pathway_overlay — remove a protein AND overlay its REAL measured knockout effect on the 5-pathway graph
(viz/pathway_overlay.html). The removed node is struck out (red x) with its edges cut; then every graph node that
ACTUALLY moved when this gene was knocked out (measured Perturb-seq) is badged ▲ up (green) / ▼ down (blue), sized by
effect. The caption states the disjointness: how many genes moved cell-wide, how many landed in the graph, how many were
structural neighbours.

The point (honest): the STRUCTURAL cut is local and certain, but the MEASURED movers are a different set — almost entirely
OFF the graph and NOT the removed protein's neighbours. This is the ~3%-recall wall made visual: the pathway wiring does
not carry the real downstream effect. Usage: python3 colab/pathway_overlay.py [GENE]   (default CTNNB1).
"""
import json, html, sys
from pathlib import Path
import numpy as np
import pathway_graph as pg
import dependency as _D
VIZ = Path("viz")


def build(removed):
    cols, nodes, edges = pg.build_graph()
    gset = {n["g"] for n in nodes}
    dep = _D.Dependency()
    syms, M, pidx, thr = dep.syms, dep.M, dep.pidx, dep.thr
    symidx = {s: i for i, s in enumerate(syms)}
    measured = removed in pidx
    moved = {}
    n_cellwide = n_ingraph = 0
    top_offgraph = []
    if measured:
        row = M[pidx[removed]]
        allmv = sorted(((syms[j], float(row[j])) for j in np.where(np.abs(row) > thr)[0]), key=lambda x: -abs(x[1]))
        n_cellwide = len(allmv)
        for g, v in allmv:
            if g in gset and g != removed:
                moved[g] = v
        n_ingraph = len(moved)
        top_offgraph = [(g, round(v, 2)) for g, v in allmv if g not in gset][:8]
    nbrs = set()
    for e in edges:
        if e["a"] == removed:
            nbrs.add(e["b"])
        if e["b"] == removed:
            nbrs.add(e["a"])
    n_nbr_moved = len([g for g in moved if g in nbrs])
    return cols, nodes, edges, removed, moved, measured, n_cellwide, n_ingraph, n_nbr_moved, top_offgraph


def render(removed):
    cols, nodes, edges, removed, moved, measured, ncw, nig, nnb, offg = build(removed)
    W, H, PX0, PX1, PY0, PY1 = 1200, 860, 70, 1130, 150, 760
    col_x, bandh = pg._layout(nodes, PX0, PX1, PY0, PY1)
    byname = {n["g"]: n for n in nodes}
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
        cls = "cut" if (e["a"] == removed or e["b"] == removed) else ("edge inter" if e["inter"] else "edge intra")
        svg.append(f'<line x1="{a["x"]:.1f}" y1="{a["y"]:.1f}" x2="{b["x"]:.1f}" y2="{b["y"]:.1f}" class="{cls}"/>')
    for n in nodes:
        g = n["g"]
        col = pg.SHARED if n["shared"] else pg.PCOL[n["col"][0]]
        r = 13 if n["shared"] else 10
        if g == removed:
            svg.append(f'<g class="rem"><title>{esc(g)} — REMOVED</title>'
                       f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}"/>'
                       f'<line x1="{n["x"]-r:.1f}" y1="{n["y"]-r:.1f}" x2="{n["x"]+r:.1f}" y2="{n["y"]+r:.1f}"/>'
                       f'<line x1="{n["x"]-r:.1f}" y1="{n["y"]+r:.1f}" x2="{n["x"]+r:.1f}" y2="{n["y"]-r:.1f}"/>'
                       f'<text x="{n["x"]:.1f}" y="{n["y"]+r+11:.1f}" class="glab remlab">{esc(g)}</text></g>')
        else:
            svg.append(f'<g class="node{" shared" if n["shared"] else ""}"><title>{esc(g)}</title>'
                       f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}" fill="{col}"/>'
                       f'<text x="{n["x"]:.1f}" y="{n["y"]+r+11:.1f}" class="glab">{esc(g)}</text></g>')
            if g in moved:                                       # measured mover badge
                v = moved[g]
                mr = 7 + min(7, abs(v) * 6)
                cls = "up" if v > 0 else "down"
                arr = "▲" if v > 0 else "▼"
                svg.append(f'<g class="mv"><title>{esc(g)} moved {v:+.2f} when {esc(removed)} knocked out (MEASURED)</title>'
                           f'<circle cx="{n["x"]+r+3:.1f}" cy="{n["y"]-r-2:.1f}" r="{mr:.1f}" class="mvdot {cls}"/>'
                           f'<text x="{n["x"]+r+3:.1f}" y="{n["y"]-r+1:.1f}" class="mvarr">{arr}</text></g>')
    off = ", ".join(f"{g}({v:+.1f})" for g, v in offg) or "—"
    page = _TEMPLATE.format(removed=esc(removed), svg="\n".join(svg), W=W, H=H, ncw=ncw, nig=nig, nnb=nnb,
                            off=esc(off), measured=("" if measured else " (no measured knockout for this gene)"))
    VIZ.mkdir(exist_ok=True)
    (VIZ / "pathway_overlay.html").write_text(page)
    return removed, ncw, nig, nnb, offg


_TEMPLATE = '''<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c2c9d4; --inter:#5b6472; --cut:#d1495b; --up:#3a9d5d; --down:#3d7fd1; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#8892a3; --cut:#e06c82; --up:#4bb06e; --down:#5a97e0; }} }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6; --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c2c9d4; --inter:#5b6472; --cut:#d1495b; --up:#3a9d5d; --down:#3d7fd1; }}
:root[data-theme="dark"] {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d; --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#8892a3; --cut:#e06c82; --up:#4bb06e; --down:#5a97e0; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1240px;margin:0 auto;padding:28px 20px 48px}} h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--muted);font-size:13.5px;margin:0 0 12px;max-width:920px;line-height:1.5}}
.legend{{display:flex;flex-wrap:wrap;gap:12px 16px;align-items:center;margin:0 0 12px;font-size:12.5px}} .lg{{display:inline-flex;align-items:center;gap:6px}}
.lg .d{{width:13px;height:13px;border-radius:50%;display:inline-block}} .lg .up{{background:var(--up)}} .lg .down{{background:var(--down)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px;overflow-x:auto}}
svg{{display:block;width:100%;height:auto;min-width:940px}}
.band0{{fill:var(--band0)}} .band1{{fill:var(--band1)}} .band2{{fill:var(--band2)}} .band3{{fill:var(--band3)}}
.bandlab{{fill:var(--muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em}}
.colhdr{{font-size:13px;font-weight:700;text-anchor:middle}} .colrule{{stroke:var(--line);stroke-width:1;stroke-dasharray:2 5}}
.edge{{fill:none}} .edge.intra{{stroke:var(--edge);stroke-width:1;opacity:.5}} .edge.inter{{stroke:var(--inter);stroke-width:1.8;opacity:.75}}
.cut{{stroke:var(--cut);stroke-width:1.8;opacity:.85;stroke-dasharray:4 3}}
.node circle{{stroke:var(--panel);stroke-width:2}} .node.shared circle{{stroke:#caa43a;stroke-width:2.5}}
.rem circle{{fill:none;stroke:var(--cut);stroke-width:2.5;opacity:.5}} .rem line{{stroke:var(--cut);stroke-width:2.5}} .remlab{{fill:var(--cut);font-weight:700;text-decoration:line-through}}
.mvdot{{stroke:var(--panel);stroke-width:1.5;opacity:.9}} .mvdot.up{{fill:var(--up)}} .mvdot.down{{fill:var(--down)}}
.mvarr{{font-size:8px;fill:#fff;text-anchor:middle;font-weight:700}}
.glab{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;fill:var(--ink);text-anchor:middle}}
.foot{{color:var(--muted);font-size:12px;line-height:1.55;margin-top:16px;max-width:980px}} .foot b{{color:var(--ink);font-weight:600}} .stats{{font-family:ui-monospace,monospace;color:var(--ink)}}
</style>
<div class="wrap">
  <h1>Remove {removed} &mdash; structural cut vs the REAL measured effect</h1>
  <p class="sub">The graph with <b style="color:var(--cut)">{removed} removed</b> (red &times;, edges cut), overlaid with
  what <b>actually moved</b> when {removed} was knocked out in real Perturb&#8209;seq{measured}: <span style="color:var(--up)">&#9650; up</span> /
  <span style="color:var(--down)">&#9660; down</span>, sized by effect. Compare the <b>structural</b> cut (local) with the
  <b>measured</b> movers (elsewhere).</p>
  <div class="legend"><span class="lg"><span class="d up"></span>measured UP on knockout</span><span class="lg"><span class="d down"></span>measured DOWN</span></div>
  <div class="card"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Five-pathway graph with a removal and its measured knockout effect overlaid">{svg}</svg></div>
  <p class="foot">
    <b>Structural vs measured &mdash; they're different sets.</b> Removing <span class="stats">{removed}</span> knocked out
    moved <b>{ncw} genes cell&#8209;wide</b> (measured), but only <b>{nig}</b> land anywhere in this 5&#8209;pathway graph,
    and <b>{nnb}</b> are structural neighbours of {removed}. The real effect is almost entirely <b>off&#8209;graph</b>:
    top movers <span class="stats">{off}</span> &mdash; none of them in the pathway wiring.<br>
    <b>Why this is the honest picture.</b> The pathway graph shows who&#8209;shares&#8209;and&#8209;touches; the measured
    knockout effect goes somewhere else (mostly indirect/regulatory, and signalling acts post&#8209;transcriptionally so
    mRNA is blind to it). This is the ~3%&#8209;recall wall, drawn: the wiring is real but does <b>not</b> carry the
    downstream effect. The structural cut is certain; the cascade must be measured, not read off the graph.
  </p>
</div>'''


def main():
    removed = sys.argv[1] if len(sys.argv) > 1 else "CTNNB1"
    g, ncw, nig, nnb, offg = render(removed)
    print(f"overlay {g}: {ncw} genes moved cell-wide (measured); {nig} in the 5-pathway graph; {nnb} are structural neighbours")
    print(f"  top off-graph movers: {offg}")
    print(f"  -> {VIZ/'pathway_overlay.html'}")


if __name__ == "__main__":
    main()
