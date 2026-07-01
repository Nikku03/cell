"""Generate a self-contained interactive spatial cell: pick a gene, watch its product travel
genome -> hnRNA -> mRNA -> nucleus->cytoplasm -> translation -> transport -> final location,
with the machinery at every step. Data embedded from cell_trafficking.json."""
import json
from pathlib import Path
OUT=Path("outputs/orphan")
DATA=json.load(open(OUT/"cell_trafficking.json"))
html = """<!doctype html><html><head><meta charset=utf-8><title>Cell trafficking — gene to final location</title>
<style>
 body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0d1b2a;color:#e0e6ed}
 #wrap{display:flex;height:100vh}
 #left{flex:1;position:relative}
 #right{width:380px;background:#122135;padding:14px 16px;overflow:auto;box-shadow:-2px 0 12px #0006}
 h2{margin:.2em 0;font-size:17px;color:#7fd1ff}
 .genebtn{display:inline-block;margin:3px;padding:5px 9px;border-radius:6px;background:#1b3a5c;color:#cfe8ff;cursor:pointer;font-size:12px;border:1px solid #2a5580}
 .genebtn:hover{background:#26547f}.genebtn.on{background:#2e86de;color:#fff;border-color:#7fd1ff}
 .step{margin:6px 0;padding:7px 9px;border-left:3px solid #2e86de;background:#0f2237;border-radius:0 6px 6px 0;font-size:12px;opacity:.4;transition:.3s}
 .step.active{opacity:1;border-left-color:#ffd166;background:#14304d}
 .step b{color:#7fd1ff}.mach{color:#ffd166;font-size:11px}.sp{color:#9be7a0}
 svg{width:100%;height:100%}
 .cmp{font-size:12px;fill:#8fa8c0;font-weight:600}
 #meta{font-size:12px;color:#9fb3c8;margin-top:8px;line-height:1.5}
 .wp{fill:#ffd166;stroke:#0d1b2a;stroke-width:1.5}
 .wpt{font-size:10px;fill:#0d1b2a;font-weight:700;text-anchor:middle}
</style></head><body><div id=wrap>
<div id=left><svg id=cell viewBox="0 0 920 660" preserveAspectRatio=xMidYMid meet>
 <defs><marker id=arw markerWidth=8 markerHeight=8 refX=6 refY=3 orient=auto><path d="M0,0 L6,3 L0,6 z" fill=#ffd166/></marker></defs>
 <!-- extracellular -->
 <rect x=0 y=0 width=920 height=660 fill=#0d1b2a/>
 <!-- cell / plasma membrane -->
 <ellipse cx=455 cy=330 rx=445 ry=305 fill=#12263d stroke=#5aa0e0 stroke-width=6/>
 <ellipse cx=455 cy=330 rx=445 ry=305 fill=none stroke=#274b6e stroke-width=2 stroke-dasharray="2 5"/>
 <text class=cmp x=770 y=120>plasma membrane</text>
 <text class=cmp x=560 y=70>cytoplasm</text>
 <!-- nucleus -->
 <circle cx=270 cy=330 r=155 fill=#1a2f4d stroke=#8f7fd1 stroke-width=5/>
 <circle cx=270 cy=330 r=155 fill=none stroke=#6a5ba8 stroke-width=2 stroke-dasharray="1 6"/>
 <text class=cmp x=225 y=200 fill=#b9a9ff>nucleus</text>
 <!-- nuclear pore -->
 <rect x=415 y=316 width=16 height=28 rx=3 fill=#8f7fd1/>
 <text class=cmp x=398 y=372 fill=#b9a9ff font-size=10>pore</text>
 <!-- ER -->
 <path d="M455 400 q40 -30 80 -8 q40 22 78 -2 q35 -20 70 4" fill=none stroke=#e08a5a stroke-width=9 opacity=.8/>
 <text class=cmp x=520 y=440 fill=#e0a58a>ER</text>
 <!-- Golgi -->
 <g stroke=#5ad1c0 stroke-width=6 fill=none opacity=.85>
  <path d="M600 300 q35 -12 70 0"/><path d="M596 315 q40 -13 78 0"/><path d="M600 330 q35 -12 70 0"/></g>
 <text class=cmp x=612 y=285 fill=#8ce0d4>Golgi</text>
 <!-- mitochondrion -->
 <ellipse cx=640 cy=500 rx=70 ry=34 fill=#2d1a3a stroke=#d15a9e stroke-width=4/>
 <path d="M585 500 q15 -20 30 0 q15 20 30 0 q15 -20 30 0" fill=none stroke=#d15a9e stroke-width=2/>
 <text class=cmp x=605 y=548 fill=#e08ac0>mitochondrion</text>
 <!-- ribosomes -->
 <g fill=#7fd1ff opacity=.5><circle cx=500 cy=250 r=5 /><circle cx=540 cy=520 r=5 /><circle cx=470 cy=180 r=5 /><circle cx=700 cy=200 r=5 /></g>
 <g id=path></g>
</svg></div>
<div id=right>
 <h2>Gene → product → location</h2>
 <div id=genes></div>
 <div id=meta></div>
 <h2 style="margin-top:12px">Journey</h2>
 <div id=steps></div>
</div></div>
<script>
const DATA=__DATA__;
const LOC={ 'nucleus':[250,330],'nuclear pore':[423,330],'cytoplasm':[540,150],'ER':[520,405],
 'Golgi':[635,315],'plasma membrane':[880,300],'extracellular':[905,250],'mitochondrion':[640,500]};
const NUCSPOTS=[[200,300],[250,360],[320,300]]; // gene, transcription, splicing inside nucleus
function coord(step,i,seen){
  if(step.step=='gene')return NUCSPOTS[0];
  if(step.step=='transcription')return NUCSPOTS[1];
  if(step.step=='splicing')return NUCSPOTS[2];
  let b=LOC[step.loc]||[540,150]; let n=seen[step.loc]||0; seen[step.loc]=n+1;
  return [b[0]+n*26, b[1]+n*22];
}
function draw(gene){
  document.querySelectorAll('.genebtn').forEach(b=>b.classList.toggle('on',b.dataset.g==gene));
  const d=DATA[gene]; const g=document.getElementById('path'); g.innerHTML='';
  document.getElementById('meta').innerHTML=`<b style=color:#7fd1ff>${gene}</b> — ${d.name}<br>chr ${d.chrom} · route: <b>${d.route}</b> · TM:${d.n_tm} · signal:${d.signal}<br>locations: ${d.locations.join(', ')}`;
  let seen={}; let pts=d.steps.map((s,i)=>coord(s,i,seen));
  // polyline
  for(let i=1;i<pts.length;i++){
    const l=document.createElementNS('http://www.w3.org/2000/svg','line');
    l.setAttribute('x1',pts[i-1][0]);l.setAttribute('y1',pts[i-1][1]);l.setAttribute('x2',pts[i][0]);l.setAttribute('y2',pts[i][1]);
    l.setAttribute('stroke','#ffd166');l.setAttribute('stroke-width','2.2');l.setAttribute('marker-end','url(#arw)');l.setAttribute('opacity','.85');
    g.appendChild(l);
  }
  pts.forEach((p,i)=>{
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',p[0]);c.setAttribute('cy',p[1]);c.setAttribute('r',11);c.setAttribute('class','wp');g.appendChild(c);
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',p[0]);t.setAttribute('y',p[1]+3.5);t.setAttribute('class','wpt');t.textContent=(i+1);g.appendChild(t);
  });
  // steps list
  document.getElementById('steps').innerHTML=d.steps.map((s,i)=>
    `<div class=step id=st${i}><b>${i+1}. ${s.step}</b> <span style=color:#8fa8c0>@ ${s.loc}</span><br>
     <span class=sp>${s.species}</span>${s.machinery?` · <span class=mach>${s.machinery}</span>`:''}<br>${s.desc}</div>`).join('');
  // animate token + step highlight
  let tok=document.createElementNS('http://www.w3.org/2000/svg','circle');
  tok.setAttribute('r',7);tok.setAttribute('fill','#9be7a0');tok.setAttribute('stroke','#0d1b2a');tok.setAttribute('stroke-width','1.5');g.appendChild(tok);
  let i=0;
  function hop(){
    if(i>=pts.length){return;}
    tok.setAttribute('cx',pts[i][0]);tok.setAttribute('cy',pts[i][1]);
    document.querySelectorAll('.step').forEach(e=>e.classList.remove('active'));
    let el=document.getElementById('st'+i); if(el){el.classList.add('active');el.scrollIntoView({block:'nearest'});}
    i++; setTimeout(hop,900);
  }
  hop();
}
const gd=document.getElementById('genes');
Object.keys(DATA).forEach(g=>{const b=document.createElement('span');b.className='genebtn';b.dataset.g=g;
  b.textContent=g+' ('+DATA[g].route[0]+')';b.onclick=()=>draw(g);gd.appendChild(b);});
draw('SLC2A1');
</script></body></html>"""
html=html.replace("__DATA__",json.dumps(DATA))
open(OUT/"cell_trafficking.html","w").write(html)
print("wrote cell_trafficking.html ("+str(len(html))+" bytes,",len(DATA),"genes)")
