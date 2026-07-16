"""orphan_network — WIRE THE UNKNOWNS IN. Take the orphan / dark genes that `connect` places (no pathway annotation) and
draw them INTO the same 5-pathway crosstalk network (viz/orphan_network.html), via their MEASURED PPI links to known
pathway proteins. Known anchors are circles; orphan genes are diamonds, coloured by confidence: GOLD filled = corroborated
(literature and PPI agree on the pathway, ~71% precision), HOLLOW = wired by PPI only (a lead). Same compartment layout as
pathway_graph (columns = pathways, rows = compartments, top-to-down).

HONEST SCOPE: the EDGES are measured (PPI); the PLACEMENT is a hypothesis (gold = the ~71% corroborated tier, hollow =
PPI-only lead). Vertical = observed compartment. A descriptive near-field map that slots each unknown next to the
machinery it physically touches -- a hypothesis to test, not a proven annotation or a phenotype predictor.
"""
import json, html, collections
from pathlib import Path
import connect as C
import pathway_graph as pg

OUT = Path("outputs/orphan")
VIZ = Path("viz")
NCORROB, NPPI, MAXLINK = 12, 3, 3                                # orphans shown: corroborated / ppi-only / links each


def build():
    G = C.build()
    names = G["names"]
    paths = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    loc = json.load(open(OUT / "localization.json")).get("labels", {})

    def depth(g):
        ds = [pg.DEPTH[c] for c in (loc.get(g) or []) if c in pg.DEPTH]
        return min(ds) if ds else 2
    pwmem = {pg.SHORT[p]: {names[i] for i in paths[p] if i < len(names)} for p in pg.PATHWAYS}
    allmem = set().union(*pwmem.values())
    udeg = {g: len(G["ppi"].get(g, set()) & allmem) for g in allmem}
    which = lambda p: [pg.SHORT[q] for q in pg.PATHWAYS if p in pwmem[pg.SHORT[q]]]

    scored = []
    for g in (x for x in G["lit"] if not G["gene_paths"].get(x)):
        links = [p for p in G["ppi"].get(g, ()) if p in allmem]
        if len(links) < 3:
            continue
        links = sorted(links, key=lambda p: (-udeg[p], p))[:MAXLINK]     # name tiebreak -> reproducible
        votes = collections.Counter(s for p in links for s in which(p))
        corr = bool(C.connect(g, G)["corroborated"])
        scored.append({"g": g, "assigned": votes.most_common(1)[0][0], "corrob": corr, "links": links, "depth": depth(g)})
    scored.sort(key=lambda o: (not o["corrob"], o["g"]))                 # corroborated first, then alphabetical
    chosen = [o for o in scored if o["corrob"]][:NCORROB] + [o for o in scored if not o["corrob"]][:NPPI]

    disp = {}
    for o in chosen:
        for p in o["links"]:
            if p not in disp:
                pw = which(p)
                disp[p] = {"g": p, "pw": pw, "col": [pg.PATHWAYS.index(next(q for q in pg.PATHWAYS if pg.SHORT[q] == s)) for s in pw],
                           "depth": depth(p), "shared": len(pw) >= 2, "kind": "known"}
    for o in chosen:
        k = pg.PATHWAYS.index(next(q for q in pg.PATHWAYS if pg.SHORT[q] == o["assigned"]))
        disp[o["g"]] = {"g": o["g"], "pw": [o["assigned"]], "col": [k], "depth": o["depth"], "shared": False,
                        "kind": "orphan", "corrob": o["corrob"], "links": o["links"]}
    nodes = list(disp.values())
    kedges, seen = [], set()
    for a in disp:
        if disp[a]["kind"] != "known":
            continue
        for b in G["ppi"].get(a, ()):
            if b in disp and disp[b]["kind"] == "known" and a < b and (a, b) not in seen:
                seen.add((a, b))
                kedges.append({"a": a, "b": b, "inter": not (set(disp[a]["pw"]) & set(disp[b]["pw"]))})
    oedges = [{"a": o["g"], "b": p, "corrob": o["corrob"]} for o in chosen for p in o["links"] if p in disp]
    return nodes, kedges, oedges, chosen


def render(nodes, kedges, oedges, chosen):
    W, H, PX0, PX1, PY0, PY1 = 1240, 860, 80, 1160, 150, 780
    COL_X = [PX0 + (k + 0.5) * (PX1 - PX0) / 5 for k in range(5)]
    BANDH = (PY1 - PY0) / 4
    BAND_Y = [PY0 + d * BANDH + BANDH / 2 for d in range(4)]
    for n in nodes:
        n["bx"] = sum(COL_X[c] for c in n["col"]) / len(n["col"]) + (50 if n["kind"] == "orphan" else -8)
    OFFS, MINDX = [0, -42, 42, -84, 84, -126, 126], 80
    for d in range(4):
        band = sorted([n for n in nodes if n["depth"] == d], key=lambda n: n["bx"])
        rows = [[] for _ in OFFS]
        for n in band:
            for ri, off in enumerate(OFFS):
                if all(abs(n["bx"] - m["bx"]) >= MINDX for m in rows[ri]):
                    n["x"], n["y"] = n["bx"], max(PY0 + 16, min(PY1 - 16, BAND_Y[d] + off)); rows[ri].append(n); break
            else:
                n["x"], n["y"] = n["bx"], BAND_Y[d]; rows[0].append(n)
    byname = {n["g"]: n for n in nodes}
    esc = lambda s: html.escape(str(s))
    svg = []
    for d in range(4):
        y0 = PY0 + d * BANDH
        svg.append(f'<rect x="{PX0-46}" y="{y0:.0f}" width="{PX1-PX0+92}" height="{BANDH:.0f}" class="band band{d}"/>')
        svg.append(f'<text x="{PX0-52}" y="{y0+16:.0f}" class="bandlab">{pg.BANDS[d]}</text>')
    for k in range(5):
        svg.append(f'<text x="{COL_X[k]:.0f}" y="{PY0-42:.0f}" class="colhdr" fill="{pg.PCOL[k]}">{esc(pg.SHORT[pg.PATHWAYS[k]])}</text>')
        svg.append(f'<line x1="{COL_X[k]:.0f}" y1="{PY0-32:.0f}" x2="{COL_X[k]:.0f}" y2="{PY1+8:.0f}" class="colrule"/>')
    for e in kedges:
        a, b = byname[e["a"]], byname[e["b"]]
        svg.append(f'<line x1="{a["x"]:.1f}" y1="{a["y"]:.1f}" x2="{b["x"]:.1f}" y2="{b["y"]:.1f}" class="edge {"inter" if e["inter"] else "intra"}"/>')
    for e in oedges:
        a, b = byname[e["a"]], byname[e["b"]]
        svg.append(f'<line x1="{a["x"]:.1f}" y1="{a["y"]:.1f}" x2="{b["x"]:.1f}" y2="{b["y"]:.1f}" class="owire {"cor" if e["corrob"] else "un"}"/>')
    for n in nodes:
        if n["kind"] == "known":
            col = pg.SHARED if n["shared"] else pg.PCOL[n["col"][0]]
            r = 10 if n["shared"] else 8
            svg.append(f'<g class="node{" shared" if n["shared"] else ""}"><title>{esc(n["g"])} — {esc("/".join(n["pw"]))}</title>'
                       f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{r}" fill="{col}"/>'
                       f'<text x="{n["x"]:.1f}" y="{n["y"]+r+10:.1f}" class="glab">{esc(n["g"])}</text></g>')
        else:
            col, s = pg.PCOL[n["col"][0]], 8
            tip = f"{n['g']} (orphan) → {n['pw'][0]} · {'corroborated (lit+PPI)' if n['corrob'] else 'PPI-only lead'} · links {', '.join(n['links'])}"
            svg.append(f'<g class="orph {"cor" if n["corrob"] else "un"}"><title>{esc(tip)}</title>'
                       f'<rect x="{n["x"]-s:.1f}" y="{n["y"]-s:.1f}" width="{2*s}" height="{2*s}" transform="rotate(45 {n["x"]:.1f} {n["y"]:.1f})" '
                       f'fill="{col if n["corrob"] else "none"}" stroke="{col}"/>'
                       f'<text x="{n["x"]:.1f}" y="{n["y"]+s+12:.1f}" class="glab orphlab">{esc(n["g"])}</text></g>')
    nk = sum(1 for n in nodes if n["kind"] == "known")
    ncor = sum(1 for o in chosen if o["corrob"])
    legend = "".join(f'<span class="lg"><i style="background:{pg.PCOL[k]}"></i>{esc(pg.SHORT[pg.PATHWAYS[k]])}</span>' for k in range(5))
    ex = [o for o in chosen if o["corrob"]][:3]                  # example sentence built from the ACTUAL shown genes
    examples = "; ".join(f'<span class="stats">{esc(o["g"])}</span> &rarr; {esc(o["assigned"])} '
                         f'(via {esc(", ".join(o["links"][:2]))})' for o in ex)
    page = _TEMPLATE.format(SHARED=pg.SHARED, legend=legend, svg="\n".join(svg), W=W, H=H,
                            nch=len(chosen), ncor=ncor, nk=nk, examples=examples)
    VIZ.mkdir(exist_ok=True)
    (VIZ / "orphan_network.html").write_text(page)
    return nk, len(chosen), ncor, len(kedges), len(oedges)


_TEMPLATE = '''<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c8cfd8; --inter:#7a828f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#7c8698; }} }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#1a2230; --muted:#66707e; --line:#d9dee6;
  --band0:#eef4f3; --band1:#eaf0f6; --band2:#f0eef7; --band3:#f3edf0; --edge:#c8cfd8; --inter:#7a828f; }}
:root[data-theme="dark"] {{ --bg:#0e1219; --panel:#151b25; --ink:#e6eaf1; --muted:#8b95a5; --line:#26303d;
  --band0:#111a1a; --band1:#101722; --band2:#15131f; --band3:#1a1218; --edge:#2c3644; --inter:#7c8698; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1300px;margin:0 auto;padding:28px 20px 48px}}
h1{{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}} .sub{{color:var(--muted);font-size:13.5px;margin:0 0 16px;max-width:960px;line-height:1.5}}
.legend{{display:flex;flex-wrap:wrap;gap:12px 16px;align-items:center;margin:0 0 12px;font-size:12.5px}}
.lg{{display:inline-flex;align-items:center;gap:6px}} .lg i{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.lg.dia i{{width:11px;height:11px;border-radius:2px;transform:rotate(45deg)}} .lg.dia.un i{{background:transparent!important;border:1.6px solid var(--muted)}}
.lg.e span{{display:inline-block;width:20px;border-top:2px solid var(--edge)}} .lg.e.wire span{{border-top:2.5px solid #caa43a}} .lg.e.wire.un span{{border-top:2px dashed var(--muted)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px;overflow-x:auto;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
svg{{display:block;width:100%;height:auto;min-width:1020px}}
.band0{{fill:var(--band0)}} .band1{{fill:var(--band1)}} .band2{{fill:var(--band2)}} .band3{{fill:var(--band3)}}
.bandlab{{fill:var(--muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em}}
.colhdr{{font-size:13px;font-weight:700;text-anchor:middle}} .colrule{{stroke:var(--line);stroke-width:1;stroke-dasharray:2 5}}
.edge{{fill:none}} .edge.intra{{stroke:var(--edge);stroke-width:.9;opacity:.5}} .edge.inter{{stroke:var(--inter);stroke-width:1.5;opacity:.6}}
.owire{{fill:none}} .owire.cor{{stroke:#caa43a;stroke-width:2;opacity:.9}} .owire.un{{stroke:var(--muted);stroke-width:1.3;opacity:.5;stroke-dasharray:3 3}}
.node circle{{stroke:var(--panel);stroke-width:1.8}} .node.shared circle{{stroke:{SHARED};stroke-width:2.4}}
.orph rect{{stroke-width:2.2}} .orph.cor rect{{filter:drop-shadow(0 0 3px #caa43a)}}
.glab{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:9px;fill:var(--ink);text-anchor:middle}} .orphlab{{font-weight:700}}
.foot{{color:var(--muted);font-size:12px;line-height:1.55;margin-top:16px;max-width:1000px}} .foot b{{color:var(--ink);font-weight:600}} .stats{{font-family:ui-monospace,monospace;color:var(--ink)}}
</style>
<div class="wrap">
  <h1>Wiring the unknowns in &mdash; orphan genes connected into the pathway network</h1>
  <p class="sub">Circles are known pathway proteins (the anchors the orphans touch); <b>diamonds are orphan / dark genes</b>
  with <b>no pathway annotation</b>, wired in by their <b>measured PPI links</b>. Gold filled diamonds + gold links are
  <b>corroborated</b> (literature and PPI agree on the pathway, ~71% precision); hollow diamonds + dashed links are wired
  by <b>PPI only</b>. Columns are the five pathways, rows are compartments (top&#8209;to&#8209;down).</p>
  <div class="legend">{legend}
    <span class="lg dia"><i style="background:#caa43a"></i>orphan &mdash; corroborated</span>
    <span class="lg dia un"><i></i>orphan &mdash; PPI&#8209;only</span>
    <span class="lg e wire"><span></span>corroborated link</span>
    <span class="lg e wire un"><span></span>PPI&#8209;only link</span>
  </div>
  <div class="card">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Network of orphan genes wired into five signaling pathways by PPI">{svg}</svg>
  </div>
  <p class="foot">
    <b>What you're seeing.</b> Each diamond is an uncharacterised gene placed by <span class="stats">connect</span>; the
    line from it is a <b>real measured physical interaction</b> to a known pathway protein &mdash; that's how it wires
    into the network. Corroborated placements this run: {examples}. <b>{nch}</b> orphans ({ncor} corroborated) wired to
    <b>{nk}</b> known anchors.<br>
    <b>Honest scope.</b> The <b>edges are measured</b> (PPI); the <b>placement is a hypothesis</b> &mdash; gold is the
    ~71%&#8209;precision corroborated tier, hollow is a PPI&#8209;only lead. Vertical = observed compartment. A descriptive
    near&#8209;field map that slots the unknowns next to the machinery they physically touch &mdash; a hypothesis to test.
  </p>
</div>'''


def main():
    nodes, kedges, oedges, chosen = build()
    nk, nch, ncor, ke, oe = render(nodes, kedges, oedges, chosen)
    print(f"orphan network: {nch} orphans ({ncor} corroborated) wired to {nk} known anchors; {ke} known edges, {oe} wires")
    print("  orphans:", ", ".join(f"{o['g']}{'*' if o['corrob'] else ''}" for o in chosen))
    print(f"  -> {VIZ/'orphan_network.html'}")


if __name__ == "__main__":
    main()
