"""cancer_cell_html — render the perturbed-cell maps (cancer_cell_map.json) as one visual page: each tumour a
card showing its mutation calls (GOF/LOF), the HYPERACTIVE programme (red, driven up) and the LOST programme
(blue, driven down), and the driver/reversal point. This is the literal 'plot the whole cancerous cell'.
-> outputs/orphan/cancer_cell_map.html
"""
import os, sys, json, html
sys.path.insert(0, os.path.dirname(__file__))
OUT = "outputs/orphan"


def _chip(call):
    g = html.escape(call["gene"]); eff = call.get("effect", "")
    if call.get("clamp") == 1:
        bg, fg, tag = "#0b6b3a", "#eaffea", "GOF↑"
    elif call.get("clamp") == -1:
        bg, fg, tag = "#7a1020", "#ffe9ec", "LOF↓"
    else:
        bg, fg, tag = "#333a44", "#cfd6df", "passenger"
    return (f'<span class="chip" style="background:{bg};color:{fg}">{g} '
            f'<b style="opacity:.8">{tag}</b></span>')


def _bars(rows, up=True):
    if not rows:
        return '<div class="empty">— none —</div>'
    m = max(abs(r["delta"]) for r in rows) or 1.0
    col = "#e0522d" if up else "#2d7fe0"
    out = []
    for r in rows[:8]:
        w = int(100 * abs(r["delta"]) / m)
        nm = html.escape(r["pathway"])
        out.append(
            f'<div class="row"><div class="pw" title="{nm}">{nm}</div>'
            f'<div class="track"><div class="fill" style="width:{w}%;background:{col}"></div></div>'
            f'<div class="val">{r["delta"]:+.2f}</div></div>')
    return "".join(out)


def render(report):
    cards = []
    for name, r in report.items():
        chips = " ".join(_chip(c) for c in r["mutation_calls"])
        drv = html.escape(str(r.get("driver")))
        note = html.escape(r.get("target_note", ""))
        cards.append(f"""
        <div class="card">
          <div class="tumor">{html.escape(name)}</div>
          <div class="muts">{chips}</div>
          <div class="meta">{r['n_genes_perturbed']} genes changed state across the cell</div>
          <div class="cols">
            <div class="col"><div class="hd up">HYPERACTIVE programme ↑</div>{_bars(r['hyperactive_pathways'], True)}</div>
            <div class="col"><div class="hd dn">LOST programme ↓</div>{_bars(r['lost_pathways'], False)}</div>
          </div>
          <div class="driver"><b>driver / reversal:</b> {drv} <span class="dnote">— {note}</span></div>
        </div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cancerous Cell Map</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0c0f14; color:#e7edf5; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:22px 26px; border-bottom:1px solid #1e2530; }}
  h1 {{ margin:0; font-size:20px; }}
  .sub {{ color:#8b97a7; margin-top:6px; max-width:820px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:16px; padding:20px; }}
  .card {{ background:#121722; border:1px solid #1e2530; border-radius:12px; padding:16px; }}
  .tumor {{ font-size:16px; font-weight:700; }}
  .muts {{ margin:10px 0; }}
  .chip {{ display:inline-block; padding:3px 9px; margin:2px 3px 2px 0; border-radius:20px; font-size:12px; }}
  .meta {{ color:#7f8b9c; font-size:12px; margin-bottom:8px; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .hd {{ font-size:11px; letter-spacing:.5px; text-transform:uppercase; margin-bottom:6px; font-weight:700; }}
  .hd.up {{ color:#f0865f; }} .hd.dn {{ color:#6aa6f5; }}
  .row {{ display:grid; grid-template-columns:1fr 62px 42px; align-items:center; gap:6px; margin:3px 0; }}
  .pw {{ font-size:11px; color:#c3ccd8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .track {{ height:7px; background:#1c2430; border-radius:4px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:4px; }}
  .val {{ font-size:11px; color:#9aa6b6; text-align:right; font-variant-numeric:tabular-nums; }}
  .empty {{ color:#5a6473; font-size:12px; }}
  .driver {{ margin-top:12px; padding-top:10px; border-top:1px solid #1e2530; font-size:12.5px; }}
  .dnote {{ color:#8b97a7; }}
  code {{ color:#9fe0b0; }}
</style></head><body>
<header>
  <h1>Cancerous Cell Map — the whole perturbed cell, read from its genome</h1>
  <div class="sub">Each mutation is called <b>GOF↑</b> (oncogene, set ON) or <b>LOF↓</b> (suppressor, set OFF)
  by biological role, then propagated through the signed causal + signalling network. Red = programme driven
  hyperactive; blue = programme lost. The target is just the node driving the red programme — and an activating
  mutation far from any active site (JAK2 V617F) is visible here as its <i>downstream</i> pathway state, not its
  location.</div>
</header>
<div class="grid">{''.join(cards)}</div>
</body></html>"""


def main():
    report = json.load(open(f"{OUT}/cancer_cell_map.json"))
    doc = render(report)
    open(f"{OUT}/cancer_cell_map.html", "w").write(doc)
    print(f"wrote {OUT}/cancer_cell_map.html ({len(doc)} bytes, {len(report)} cells)")


if __name__ == "__main__":
    main()
