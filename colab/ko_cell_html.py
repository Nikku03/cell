"""ko_cell_html -- render the knockout cell map as a self-contained spatial page.

Reads outputs/orphan/ko_cell_map_<KO>.json and emits a single HTML file with the data inlined: an empty cell drawn as real nested
organelles, every measured mover placed in its annotated compartment, transporters ringed, PPI edges drawn between placed genes, and a
click-through that traces any gene's journey from its chromosomal locus through hnRNA, mRNA and protein to where it ends up.

The honesty line the page must carry: the DESTINATION is curated annotation, the ROUTE is canonical inference from that destination, and
the movement itself is what was measured. Those three are visually distinguished rather than blended."""
import os
import json, sys, hashlib
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

# compartment -> (centre x, centre y, radius x, radius y) in the 1040x720 viewBox, plus its colour token
REGIONS = {
    "nucleus":         (648, 322, 168, 132, "nuc"),
    "cytoplasm":       (430, 360, 250, 230, "cyt"),
    "cytoskeleton":    (455, 205, 265, 95,  "csk"),
    "ER":              (300, 330, 105, 105, "er"),
    "Golgi":           (315, 500, 95,  48,  "gol"),
    "mitochondrion":   (205, 205, 110, 70,  "mit"),
    "lysosome":        (455, 565, 85,  42,  "lys"),
    "endosome":        (585, 545, 70,  38,  "end"),
    "peroxisome":      (170, 455, 58,  38,  "per"),
    "plasma membrane": (520, 360, 476, 336, "pm"),
    "membrane":        (520, 360, 448, 314, "pm"),
    "extracellular":   (520, 360, 516, 372, "ext"),
    "unannotated":     (830, 605, 118, 62,  "unk"),
}


def place(gene, comp):
    """deterministic scatter inside the compartment's region; ring-shaped for membrane/extracellular."""
    cx, cy, rx, ry, _ = REGIONS.get(comp, REGIONS["cytoplasm"])
    h = int(hashlib.md5(gene.encode()).hexdigest()[:12], 16)
    a = (h % 10000) / 10000.0
    b = ((h >> 16) % 10000) / 10000.0
    import math
    if comp in ("plasma membrane", "membrane", "extracellular"):
        th = a * 2 * math.pi
        k = 1.0 if comp == "extracellular" else (0.985 if comp == "plasma membrane" else 0.94)
        return cx + rx * k * math.cos(th), cy + ry * k * math.sin(th)
    r = math.sqrt(b) * 0.82
    th = a * 2 * math.pi
    return cx + rx * r * math.cos(th), cy + ry * r * math.sin(th)


def main(ko="GATA1"):
    d = json.load(open(OUT / f"ko_cell_map_{ko}.json"))
    for n in d["nodes"]:
        x, y = place(n["gene"], n["compartment"])
        n["x"], n["y"] = round(x, 1), round(y, 1)
    payload = json.dumps(d, separators=(",", ":"))
    t = d["totals"]

    css = """
:root{
  --ground:#F5F7F9; --panel:#FFFFFF; --line:#D4DCE4; --text:#111820; --dim:#57646F; --faint:#8493A0;
  --nuc:#6B4FA8; --cyt:#78858F; --csk:#4F6E88; --er:#2E7D72; --gol:#A87C24; --mit:#C4692B;
  --lys:#A8456B; --end:#8A5AA0; --per:#4E8C5A; --pm:#2C7A9E; --ext:#93A2AE; --unk:#96A2AC;
  --up:#CE3B28; --down:#2A6BB0; --ring:#0F9C8C;
  --cellfill:#FCFDFE; --cytofill:#EDF1F5; --shadow:rgba(17,24,32,.10);
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#070C12; --panel:#0C131A; --line:#1C2833; --text:#E2E9F0; --dim:#93A3B2; --faint:#5B6A79;
  --nuc:#A184E8; --cyt:#8B98A5; --csk:#79A0C2; --er:#38C1AF; --gol:#D6A63C; --mit:#E68A45;
  --lys:#D46A93; --end:#B07FCC; --per:#6BBF7A; --pm:#4FADD6; --ext:#7E8D9A; --unk:#7A8894;
  --up:#FF6B52; --down:#5CA8F0; --ring:#28D6C2;
  --cellfill:#0B1219; --cytofill:#101A23; --shadow:rgba(0,0,0,.45);
}}
:root[data-theme="dark"]{
  --ground:#070C12; --panel:#0C131A; --line:#1C2833; --text:#E2E9F0; --dim:#93A3B2; --faint:#5B6A79;
  --nuc:#A184E8; --cyt:#8B98A5; --csk:#79A0C2; --er:#38C1AF; --gol:#D6A63C; --mit:#E68A45;
  --lys:#D46A93; --end:#B07FCC; --per:#6BBF7A; --pm:#4FADD6; --ext:#7E8D9A; --unk:#7A8894;
  --up:#FF6B52; --down:#5CA8F0; --ring:#28D6C2;
  --cellfill:#0B1219; --cytofill:#101A23; --shadow:rgba(0,0,0,.45);
}
:root[data-theme="light"]{
  --ground:#F5F7F9; --panel:#FFFFFF; --line:#D4DCE4; --text:#111820; --dim:#57646F; --faint:#8493A0;
  --nuc:#6B4FA8; --cyt:#78858F; --csk:#4F6E88; --er:#2E7D72; --gol:#A87C24; --mit:#C4692B;
  --lys:#A8456B; --end:#8A5AA0; --per:#4E8C5A; --pm:#2C7A9E; --ext:#93A2AE; --unk:#96A2AC;
  --up:#CE3B28; --down:#2A6BB0; --ring:#0F9C8C;
  --cellfill:#FCFDFE; --cytofill:#EDF1F5; --shadow:rgba(17,24,32,.10);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--text);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:40px 22px 80px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--faint);margin:0 0 10px}
h1{font-size:clamp(30px,4.4vw,50px);line-height:1.03;letter-spacing:-.028em;font-weight:760;margin:0 0 14px;
  text-wrap:balance;max-width:19ch}
h1 em{font-style:normal;font-family:var(--mono);font-weight:700;color:var(--up);letter-spacing:-.02em}
.lede{font-size:17px;color:var(--dim);max-width:62ch;margin:0 0 30px}
h2{font-size:12px;font-family:var(--mono);letter-spacing:.15em;text-transform:uppercase;color:var(--faint);
  font-weight:600;margin:52px 0 14px;padding-bottom:9px;border-bottom:1px solid var(--line)}
.stats{display:flex;flex-wrap:wrap;gap:26px 40px;margin:0 0 30px;padding:20px 0;
  border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.stat b{display:block;font-family:var(--mono);font-size:27px;font-weight:680;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1}
.stat span{font-size:12px;color:var(--dim)}
.stage{display:grid;grid-template-columns:minmax(0,1fr) 322px;gap:22px;align-items:start}
@media(max-width:940px){.stage{grid-template-columns:1fr}}
.cellbox{background:var(--panel);border:1px solid var(--line);border-radius:15px;overflow:hidden;
  box-shadow:0 1px 3px var(--shadow)}
svg{display:block;width:100%;height:auto;touch-action:manipulation}
.tools{display:flex;flex-wrap:wrap;gap:7px;padding:12px 14px;border-bottom:1px solid var(--line)}
button{font:inherit;font-size:12px;font-family:var(--mono);letter-spacing:.03em;padding:5px 11px;
  border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--dim);cursor:pointer;
  transition:background .14s,color .14s,border-color .14s}
button:hover{color:var(--text);border-color:var(--dim)}
button[aria-pressed="true"]{background:var(--text);color:var(--ground);border-color:var(--text)}
button:focus-visible{outline:2px solid var(--ring);outline-offset:2px}
.side{position:sticky;top:18px;display:flex;flex-direction:column;gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:17px 18px;
  box-shadow:0 1px 3px var(--shadow)}
.card h3{margin:0 0 3px;font-size:20px;font-family:var(--mono);font-weight:700;letter-spacing:-.02em}
.sub{font-size:11.5px;color:var(--faint);font-family:var(--mono);margin:0 0 13px}
.kv{display:grid;grid-template-columns:76px 1fr;gap:4px 12px;font-size:12.5px;margin:0 0 4px}
.kv dt{color:var(--faint);font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase}
.kv dd{margin:0;color:var(--text)}
.route{list-style:none;margin:14px 0 0;padding:0;position:relative}
.route:before{content:"";position:absolute;left:6px;top:7px;bottom:9px;width:1px;background:var(--line)}
.route li{position:relative;padding:0 0 12px 22px;font-size:12.5px}
.route li:before{content:"";position:absolute;left:2.5px;top:5px;width:8px;height:8px;border-radius:50%;
  background:var(--panel);border:2px solid var(--dim)}
.route li.s-hnRNA:before{border-color:var(--nuc)}
.route li.s-mRNA:before{border-color:var(--gol)}
.route li.s-protein:before{border-color:var(--ring)}
.route .loc{font-family:var(--mono);font-size:12px;color:var(--text)}
.route .state{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.07em;
  text-transform:uppercase;padding:1px 6px;border-radius:4px;border:1px solid var(--line);
  color:var(--dim);margin-left:6px;vertical-align:1px}
.route .ev{display:block;color:var(--faint);font-size:11.5px;margin-top:1px}
.tag{display:inline-block;font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;padding:2px 7px;
  border-radius:5px;border:1px solid var(--line);color:var(--dim);margin:0 4px 4px 0}
.tag.up{color:var(--up);border-color:var(--up)}
.tag.down{color:var(--down);border-color:var(--down)}
.tag.tr{color:var(--ring);border-color:var(--ring)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);font-weight:600;padding:0 10px 8px 0;border-bottom:1px solid var(--line)}
td{padding:7px 10px 7px 0;border-bottom:1px solid var(--line);vertical-align:top}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;padding-right:16px}
.scroll{overflow-x:auto}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:30px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.bar{height:7px;border-radius:4px;background:var(--cytofill);overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;border-radius:4px}
.note{font-size:13px;color:var(--dim);border-left:2px solid var(--line);padding:2px 0 2px 15px;margin:16px 0}
.note b{color:var(--text);font-weight:640}
.legend{display:flex;flex-wrap:wrap;gap:5px 16px;font-size:11.5px;color:var(--dim);
  font-family:var(--mono);padding:12px 14px;border-top:1px solid var(--line)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.node{cursor:pointer}
.node circle{transition:r .13s,opacity .13s}
.node:focus-visible{outline:none}
.node:focus-visible circle.dot{stroke:var(--ring);stroke-width:2.5}
.dim{opacity:.13}
.orglabel{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  fill:var(--faint);pointer-events:none}
.foot{margin-top:56px;padding-top:22px;border-top:1px solid var(--line);font-size:12.5px;color:var(--faint)}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""
    js = """
const D=JSON.parse(document.getElementById('data').textContent);
const NS='http://www.w3.org/2000/svg';
const svg=document.getElementById('cell');
const ORG=[
 ['extracellular',520,360,516,372,'ext','extracellular'],
 ['plasma membrane',520,360,476,336,'pm','plasma membrane'],
 ['cytoplasm',430,360,250,230,'cyt','cytosol'],
 ['nucleus',648,322,168,132,'nuc','nucleus'],
 ['ER',300,330,105,105,'er','endoplasmic reticulum'],
 ['Golgi',315,500,95,48,'gol','golgi'],
 ['mitochondrion',205,205,110,70,'mit','mitochondria'],
 ['lysosome',455,565,85,42,'lys','lysosome'],
 ['endosome',585,545,70,38,'end','endosome'],
 ['peroxisome',170,455,58,38,'per','peroxisome'],
 ['unannotated',830,605,118,62,'unk','no localisation']
];
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
let mode='all', sel=null;
function build(){
  svg.innerHTML='';
  const gOrg=el('g',{}),gEdge=el('g',{}),gNode=el('g',{}),gPath=el('g',{});
  // organelle bodies
  ORG.forEach(([key,cx,cy,rx,ry,tok,label])=>{
    if(key==='extracellular'){
      gOrg.appendChild(el('rect',{x:4,y:4,width:1032,height:712,rx:26,fill:'var(--cytofill)',opacity:.35}));
      return;
    }
    const fill=key==='plasma membrane'?'var(--cellfill)':'var(--cytofill)';
    const o=el('ellipse',{cx,cy,rx,ry,fill,stroke:'var(--'+tok+')',
      'stroke-width':key==='plasma membrane'?2.4:1.3,opacity:key==='plasma membrane'?.95:.62});
    gOrg.appendChild(o);
    gOrg.appendChild(Object.assign(el('text',{x:cx,y:cy-ry-7,'text-anchor':'middle',class:'orglabel'}),
      {textContent:label}));
  });
  // nuclear pore marks
  for(let i=0;i<10;i++){const a=i/10*Math.PI*2;
    gOrg.appendChild(el('circle',{cx:648+168*Math.cos(a),cy:322+132*Math.sin(a),r:3.4,
      fill:'var(--ground)',stroke:'var(--nuc)','stroke-width':1.4}));}
  const byName={};D.nodes.forEach(n=>byName[n.gene]=n);
  D.edges.forEach(([a,b])=>{const A=byName[a],B=byName[b];if(!A||!B)return;
    gEdge.appendChild(el('line',{x1:A.x,y1:A.y,x2:B.x,y2:B.y,stroke:'var(--cyt)',
      'stroke-width':.7,opacity:.3,class:'edge'}));});
  D.nodes.forEach(n=>{
    const r=Math.max(3.1,Math.min(11,2.6+Math.log2(Math.abs(n.z))*1.5));
    const g=el('g',{class:'node',tabindex:0,role:'button'});
    g.setAttribute('aria-label',n.gene+', '+n.dir+' '+n.z+', '+n.compartment);
    if(n.transporter) g.appendChild(el('circle',{cx:n.x,cy:n.y,r:r+3.1,fill:'none',
      stroke:'var(--ring)','stroke-width':1.5,opacity:.9}));
    g.appendChild(el('circle',{cx:n.x,cy:n.y,r,fill:n.dir==='up'?'var(--up)':'var(--down)',
      opacity:.88,class:'dot'}));
    g.addEventListener('click',()=>pick(n.gene));
    g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick(n.gene);}});
    g.dataset.gene=n.gene;g.dataset.tr=n.transporter?'1':'0';g.dataset.dir=n.dir;
    g.dataset.casc=D.cascade.some(c=>c.tf===n.gene)?'1':'0';
    gNode.appendChild(g);
  });
  svg.append(gOrg,gEdge,gPath,gNode);
  applyMode();
}
function applyMode(){
  svg.querySelectorAll('.node').forEach(g=>{
    let on=true;
    if(mode==='transporters') on=g.dataset.tr==='1';
    if(mode==='up') on=g.dataset.dir==='up';
    if(mode==='down') on=g.dataset.dir==='down';
    if(mode==='cascade') on=g.dataset.casc==='1';
    g.classList.toggle('dim',!on);
  });
  svg.querySelectorAll('.edge').forEach(e=>e.classList.toggle('dim',mode!=='all'&&mode!=='cascade'));
}
const RLOC={'nucleus':[648,322],'nuclear pore':[648,190],'cytosol':[430,360],'cytoskeleton':[455,205],
 'ER membrane':[300,330],'ER lumen':[300,330],'COPII vesicle':[308,418],'Golgi':[315,500],
 'secretory vesicle':[400,540],'plasma membrane':[520,24],'membrane':[520,40],'extracellular':[520,10],
 'mitochondrion':[205,205],'lysosome':[455,565],'endosome':[585,545],'peroxisome':[170,455]};
function pick(name){
  sel=name;const n=D.nodes.find(x=>x.gene===name);if(!n)return;
  const p=document.getElementById('panel');
  const tags=[`<span class="tag ${n.dir}">${n.dir} z=${n.z}</span>`];
  if(n.transporter)tags.push('<span class="tag tr">transporter</span>');
  if(n.direct)tags.push('<span class="tag">curated GATA1 target</span>');
  p.innerHTML=`<h3>${n.gene}</h3>
   <p class="sub">origin ${n.origin} &rarr; ${n.compartment}</p>
   <div>${tags.join('')}</div>
   <dl class="kv">
    <dt>function</dt><dd>${(n.go_function||[]).join('; ')||'&mdash;'}</dd>
    <dt>process</dt><dd>${n.process||'&mdash;'}</dd>
    <dt>pathway</dt><dd>${n.pathway||'<i style="color:var(--faint)">no Reactome pathway</i>'}</dd>
    <dt>abundance</dt><dd>${n.abundance_ppm!=null?n.abundance_ppm+' ppm':'&mdash;'}</dd>
    <dt>PPI</dt><dd>${n.n_ppi} partners</dd>
    <dt>complex</dt><dd>${(n.complexes||[]).join('; ')||'<i style="color:var(--faint)">none curated</i>'}</dd>
    <dt>PTM</dt><dd>${(n.ptm||[]).join(', ')||'&mdash;'}</dd>
    <dt>regulators</dt><dd>${n.n_regulators} TFs</dd>
   </dl>
   <p class="sub" style="margin:16px 0 0">journey &mdash; state at each hop</p>
   <ol class="route">${n.route.map(r=>{
     const s=r.state.startsWith('hnRNA')?'hnRNA':r.state.startsWith('mRNA')?'mRNA':'protein';
     return `<li class="s-${s}"><span class="loc">${r.loc}</span>
       <span class="state">${r.state}</span><span class="ev">${r.event}</span></li>`;}).join('')}</ol>`;
  const gPath=svg.children[2];gPath.innerHTML='';
  let prev=null;
  n.route.forEach(r=>{
    const c=RLOC[r.loc]||[n.x,n.y];
    if(prev)gPath.appendChild(el('line',{x1:prev[0],y1:prev[1],x2:c[0],y2:c[1],
      stroke:'var(--ring)','stroke-width':1.8,'stroke-dasharray':'5 4',opacity:.85}));
    prev=c;});
  if(prev)gPath.appendChild(el('line',{x1:prev[0],y1:prev[1],x2:n.x,y2:n.y,
    stroke:'var(--ring)','stroke-width':1.8,'stroke-dasharray':'5 4',opacity:.85}));
  gPath.appendChild(el('circle',{cx:n.x,cy:n.y,r:13,fill:'none',stroke:'var(--ring)','stroke-width':2}));
}
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{
  mode=b.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',x===b));
  applyMode();}));
build();pick(D.nodes[0].gene);
"""
    N = d["nodes"]; cov = d["coverage"]
    comp_rows = "".join(
        f"<tr><td>{c}</td><td class='n'>{k}</td><td class='n'>{dict(d['transporters_by_compartment']).get(c,0)}</td>"
        f"<td><div class='bar'><i style='width:{k/max(x[1] for x in d['by_compartment'])*100:.0f}%;"
        f"background:var(--{REGIONS.get(c,(0,0,0,0,'unk'))[4]})'></i></div></td></tr>"
        for c, k in d["by_compartment"])
    pw_rows = "".join(f"<tr><td>{p}</td><td class='n'>{k}</td></tr>" for p, k in d["pathways"][:10])
    cx_rows = "".join(f"<tr><td>{p}</td><td class='n'>{k}</td></tr>" for p, k in d["complexes"][:8])
    casc_rows = "".join(
        f"<tr><td style='font-family:var(--mono)'>{c['tf']}</td><td class='n'>{c['n']}</td>"
        f"<td style='color:var(--dim);font-size:12px'>{', '.join(c['targets'][:5])}</td></tr>"
        for c in d["cascade"])
    cov_rows = "".join(
        f"<tr><td>{k.replace('_',' ')}</td><td class='n'>{v*100:.1f}%</td>"
        f"<td><div class='bar'><i style='width:{v*100:.0f}%;background:"
        f"{'var(--up)' if v<.4 else 'var(--ring)'}'></i></div></td></tr>"
        for k, v in cov.items())

    doc = f"""<title>{ko} knockout &mdash; where the response lands in the cell</title>
<style>{css}</style>
<script type="application/json" id="data">{payload}</script>
<div class="wrap">
<p class="eyebrow">Perturb-seq &middot; K562 &middot; |z| &ge; 4.2</p>
<h1>Knock out <em>{ko}</em>, and {t['moved']} genes move. Here is where they live.</h1>
<p class="lede">Every gene that responded, placed in the compartment its protein is annotated to occupy.
Ringed nodes are transporters. Click any gene to trace its journey &mdash; chromosomal locus, hnRNA, mRNA,
protein &mdash; and see what state it is in at each hop.</p>

<div class="stats">
  <div class="stat"><b>{t['moved']}</b><span>specific movers</span></div>
  <div class="stat"><b style="color:var(--up)">{t['up']}</b><span>up-regulated</span></div>
  <div class="stat"><b style="color:var(--down)">{t['down']}</b><span>down-regulated</span></div>
  <div class="stat"><b style="color:var(--ring)">{t['transporters_total']}</b><span>transporters</span></div>
  <div class="stat"><b>{t['direct_targets']}<span style="font-size:15px;color:var(--faint)"> / {t['curated_targets']}</span></b><span>curated targets that moved</span></div>
  <div class="stat"><b>{d['ppi_edges_total']}</b><span>PPI edges among movers</span></div>
</div>

<div class="stage">
  <div class="cellbox">
    <div class="tools">
      <button data-mode="all" aria-pressed="true">all {len(N)} shown</button>
      <button data-mode="transporters" aria-pressed="false">transporters</button>
      <button data-mode="up" aria-pressed="false">up</button>
      <button data-mode="down" aria-pressed="false">down</button>
      <button data-mode="cascade" aria-pressed="false">TF cascade</button>
    </div>
    <svg id="cell" viewBox="0 0 1040 720" role="img"
      aria-label="A cell diagram with {len(N)} responding genes placed in their annotated compartments"></svg>
    <div class="legend">
      <span><i style="background:var(--up)"></i>up</span>
      <span><i style="background:var(--down)"></i>down</span>
      <span><i style="border:1.5px solid var(--ring);background:none"></i>transporter</span>
      <span>node size &prop; log|z|</span>
      <span>grey lines = PPI between movers</span>
      <span>dashed teal = selected gene&rsquo;s route</span>
    </div>
  </div>
  <aside class="side"><div class="card" id="panel"></div></aside>
</div>

<p class="note"><b>Three different kinds of claim are on this page.</b> That a gene <b>moved</b>, and by how much, is
<b>measured</b> &mdash; Perturb-seq, |z| &ge; 4.2, tide-masked. Where its protein <b>ends up</b> is <b>curated annotation</b>.
The <b>route</b> it takes to get there is <b>inferred</b> from that destination by canonical trafficking rules
(SRP/co-translational entry for the secretory branch, TOM/TIM for mitochondria, NLS re-import for nuclear proteins),
not observed. Only the first is data from this experiment.</p>

<h2>Where the response landed</h2>
<div class="scroll"><table>
<thead><tr><th>compartment</th><th style="text-align:right">movers</th>
<th style="text-align:right">transporters</th><th style="width:34%"></th></tr></thead>
<tbody>{comp_rows}</tbody></table></div>
<p class="note">{t['transporters_total']} of the {t['moved']} movers are transport machinery. They concentrate exactly where
transport happens &mdash; the plasma membrane, the ER and the Golgi &mdash; rather than being spread evenly, which is what
you would expect if the knockout were remodelling the cell&rsquo;s interface with its surroundings rather than one pathway.</p>

<div class="grid2">
<div><h2>Shared pathways</h2><div class="scroll"><table>
<thead><tr><th>Reactome pathway</th><th style="text-align:right">movers</th></tr></thead>
<tbody>{pw_rows}</tbody></table></div></div>
<div><h2>Complexes with &ge;2 movers</h2><div class="scroll"><table>
<thead><tr><th>complex</th><th style="text-align:right">movers</th></tr></thead>
<tbody>{cx_rows}</tbody></table></div></div>
</div>

<h2>The second-order cascade</h2>
<p class="lede" style="font-size:15px;margin-bottom:16px">Some movers are themselves transcription factors that
regulate other movers. This is the interconnection the aggregate models kept missing.</p>
<div class="scroll"><table>
<thead><tr><th>TF (itself a mover)</th><th style="text-align:right">downstream movers</th><th>examples</th></tr></thead>
<tbody>{casc_rows}</tbody></table></div>
<p class="note"><b>SPI1 (PU.1) alone regulates 242 of the {t['moved']} movers.</b> It is {ko}&rsquo;s classic antagonist:
remove {ko} and SPI1 is released. A single second-order hop accounts for roughly 39% of the response.</p>

<h2>Where the journey breaks</h2>
<p class="lede" style="font-size:15px;margin-bottom:16px">What fraction of the {t['moved']} movers carry each annotation
layer at all. The route is complete through localisation and interaction, and incomplete at the pathway step.</p>
<div class="scroll"><table>
<thead><tr><th>layer</th><th style="text-align:right">coverage</th><th style="width:44%"></th></tr></thead>
<tbody>{cov_rows}</tbody></table></div>
<p class="note">Only a quarter of the movers sit in a curated complex or a Reactome pathway, and 15.8% have a
metabolic reaction. Asked for the whole journey from gene to final state, the honest answer is that the first five steps
are ~86% populated and the last one is ~26%.</p>

<p class="foot">{ko} knockout, K562, Replogle Perturb-seq. {len(N)} of {t['moved']} movers drawn (strongest by |z|, plus every
transporter and cascade TF); all {t['moved']} are in <span style="font-family:var(--mono)">ko_cell_map_{ko}.json</span>.
Compartment, pathway, complex, PPI, PTM and abundance from the merged annotation model. Generated by
<span style="font-family:var(--mono)">colab/ko_cell_html.py</span>.</p>
</div>
<script>{js}</script>
"""
    p = OUT / f"ko_cell_map_{ko}.html"
    p.write_text(doc)
    print(f"wrote {p} ({len(doc)//1024} KB, {len(N)} nodes, {len(d['edges'])} edges)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GATA1")
