"""Tissue Explorer (premium) — a two-level living-tissue explorer. BODY view: 9 organs wired by
endocrine hormones; zoom into any organ -> TISSUE view: its cell types wired by cell-cell communication
(colored by signaling pathway, animated, paracrine/juxtacrine/autocrine), druggable channels, downstream
propagation, knockout. Reads tissue_model.json. Self-contained. Separate from the cell model."""
import json
from pathlib import Path
OUT=Path("outputs/orphan")
D=json.load(open(OUT/"tissue_model.json"))
HTML=r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Tissue Explorer</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#03060c;color:#e8f0f8;font-family:-apple-system,'Segoe UI',Inter,sans-serif;overflow:hidden;user-select:none}
#cv{position:fixed;inset:0;display:block;cursor:grab}#cv.drag{cursor:grabbing}
#top{position:fixed;top:0;left:0;right:0;height:50px;display:flex;align-items:center;gap:8px;padding:0 14px;z-index:20;background:linear-gradient(#03060ce6,#03060c88 70%,#03060c00);pointer-events:none}
#top>*{pointer-events:auto}.brand{font-weight:700;color:#cfe6ff;font-size:15px}.brand small{color:#5f7fa6;font-weight:400;font-size:11px}
.spacer{flex:1}.pill{background:#0d1b2ecc;border:1px solid #1e3a5f;border-radius:9px;padding:6px 11px;font-size:12px;color:#cfe6ff;cursor:pointer;backdrop-filter:blur(8px)}
.pill:hover{background:#16304c}.pill.on{background:#1f6feb;border-color:#4a9eff;color:#fff}
select.pill{appearance:none;max-width:150px}input.pill{width:140px;color:#e8f0f8}
#tip{position:fixed;z-index:30;pointer-events:none;background:#081522f2;border:1px solid #274b73;border-radius:10px;padding:8px 11px;font-size:12px;display:none;max-width:260px;box-shadow:0 8px 30px #000a;backdrop-filter:blur(10px)}
#tip b{color:#7fd1ff}#panel{position:fixed;top:0;right:-430px;width:420px;height:100%;z-index:25;overflow-y:auto;padding:58px 18px 30px;background:linear-gradient(#060c16f2,#03060cf8);border-left:1px solid #16304c;backdrop-filter:blur(16px);transition:right .45s cubic-bezier(.2,.8,.2,1);box-shadow:-10px 0 40px #0008}
#panel.open{right:0}#panel h1{font-size:21px;margin:0 0 2px;color:#eaf4ff}.sub{color:#6f8fb5;font-size:12px;margin-bottom:12px}
h3{color:#8fd0ff;font-size:12.5px;margin:15px 0 5px;border-bottom:1px solid #12283f;padding-bottom:3px}
.k{color:#7f9ec0}.v{color:#eaf4ff;font-weight:600}.row{margin:4px 0;font-size:13px}.lg{color:#8aa3c0;font-size:11.5px}.warn{color:#ff9db0}.ok{color:#7ff0b5}
.chip{display:inline-block;font-size:11px;padding:3px 8px;margin:2px;border-radius:7px;background:#12283f;color:#bcd6f2;cursor:pointer}.chip:hover{background:#1c4066}
.lr{padding:5px 9px;border-radius:7px;background:#0b1a2b;margin:4px 0;font-size:12.5px}.lig{color:#7ff0b5}.rec{color:#ffcf8f}.arrow{color:#5c7fa0;margin:0 4px}
.brk{opacity:.4;text-decoration:line-through}.pwtag{font-size:10px;padding:1px 6px;border-radius:5px;margin-left:5px}.drug{color:#7ff0b5;font-size:10px}
#close{position:absolute;top:15px;right:16px;cursor:pointer;color:#6f8fb5;font-size:20px}#close:hover{color:#fff}
#hud{position:fixed;left:14px;bottom:12px;z-index:20;font-size:11px;color:#5f7fa6}#back{position:fixed;left:14px;top:60px;z-index:20}
#legend{position:fixed;right:14px;bottom:12px;z-index:20;font-size:10.5px;color:#8aa3c0;background:#060c16cc;border:1px solid #12283f;border-radius:9px;padding:8px 11px;backdrop-filter:blur(8px);max-width:220px}
</style></head><body>
<canvas id=cv></canvas>
<div id=top><span class=brand>Tissue Explorer <small>&mdash; body &rarr; tissue &rarr; cell-cell signaling</small></span>
 <span class=spacer></span>
 <select id=pw class=pill title="signaling pathway family"></select>
 <input id=ko class=pill placeholder="knock out a gene">
 <span class=pill id=reset>Reset</span></div>
<div id=back><span class=pill id=bodyBtn>&#8592; Body view</span></div>
<div id=tip></div><div id=legend></div><div id=hud></div>
<div id=panel><span id=close>&times;</span><div id=body></div></div>
<script>
const D=__DATA__;const TIS=D.tissues,names=Object.keys(TIS),DOWN=D.downstream||{},DRUGS=D.drugs||{},ENDO=D.endocrine||[];
const PWCOL={COLLAGEN:'#e0a85a',WNT:'#7fd0ff',CCL:'#e05a8a',CXCL:'#e0788a',TGFb:'#c792ea',BMP:'#a0d0ff',FGF:'#5ad0b0',VEGF:'#4dd97a',NOTCH:'#e0d060',IFN:'#ff9db0','MHC-I':'#8fbfe0','MHC-II':'#6fa0d0',NRXN:'#c0a0f0',CNTN:'#a0c0f0',ADGRL:'#9fb0e0',IL1:'#ffb080',IL2:'#ffc090',EGF:'#60d0c0',ICAM:'#d0a0e0',VTN:'#c0b070',SEMA3:'#b0d090',EPHB:'#e0b0d0','':'#5a7a9a'};
function pwc(p){return PWCOL[p]||'#5a7a9a';}
let level='body',tissue=names[0],DPR=Math.min(devicePixelRatio||1,2),W,H;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const view={s:1,x:0,y:0},tgt={s:1,x:0,y:0};let anim=0,hover=null,selCell=null,selEdge=null,koGene='',pwFilter='';
// ---- body layout: organs on a stylized figure ----
const ORG={Brain:[0.5,0.11],Lung:[0.38,0.30],Heart:[0.57,0.31],Liver:[0.40,0.46],Pancreas:[0.58,0.47],
 Kidney:[0.42,0.60],Intestine:[0.52,0.64],Skin:[0.16,0.42],'Immune (blood)':[0.80,0.45]};
let nodes=[]; // tissue-view cell nodes
function resize(){W=cv.width=innerWidth*DPR;H=cv.height=innerHeight*DPR;cv.style.width=innerWidth+'px';cv.style.height=innerHeight+'px';
 view.s=tgt.s=1;view.x=tgt.x=0;view.y=tgt.y=0;if(level=='tissue')layout();draw();}
addEventListener('resize',resize);
function layout(){const t=TIS[tissue],cts=t.cells;nodes=cts.map((c,i)=>{const a=-Math.PI/2+i/cts.length*6.2832;
 return {name:c,x:W/2+Math.cos(a)*Math.min(W,H)*0.31,y:H*0.52+Math.sin(a)*Math.min(W,H)*0.31,r:36*DPR};});}
function shade(hex,d){const n=parseInt(hex.slice(1),16);let R=Math.max(0,Math.min(255,(n>>16)+d)),Gc=Math.max(0,Math.min(255,((n>>8)&255)+d)),B=Math.max(0,Math.min(255,(n&255)+d));return'#'+((R<<16)|(Gc<<8)|B).toString(16).padStart(6,'0');}
const OCOL={Brain:'#c79ae0',Lung:'#8fd0e0',Heart:'#e0788a',Liver:'#e0a85a',Pancreas:'#e0c060',Kidney:'#9fb0e0',Intestine:'#7fd0a0','Immune (blood)':'#e05a8a',Skin:'#e0b090'};
function ocol(t){return OCOL[t]||'#6a9ee0';}
// ---- draw ----
function draw(){ctx.setTransform(1,0,0,1,0,0);const bg=ctx.createRadialGradient(W/2,H*0.42,60,W/2,H/2,Math.max(W,H)*0.72);
 bg.addColorStop(0,'#0a1220');bg.addColorStop(1,'#03060c');ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
 ctx.save();ctx.translate(view.x,view.y);ctx.scale(view.s,view.s);
 anim+=0.02;
 if(level=='body')drawBody();else drawTissue();
 ctx.restore();}
function drawBody(){
 // body silhouette
 ctx.strokeStyle='#12283f';ctx.lineWidth=2*DPR;ctx.fillStyle='#070f1a';
 ctx.beginPath();ctx.ellipse(W*0.5,H*0.12,W*0.05,H*0.055,0,0,7);ctx.fill();ctx.stroke(); // head
 roundBody();
 // endocrine axes
 for(const[h,src,tgtT,rec,eff]of ENDO){const a=ORG[src],b=ORG[tgtT];if(!a||!b)continue;
  const x1=a[0]*W,y1=a[1]*H,x2=b[0]*W,y2=b[1]*H;const mx=(x1+x2)/2+(y2-y1)*0.16,my=(y1+y2)/2-(x2-x1)*0.16;
  const hl=hover&&hover.endo&&hover.endo[0]===h&&hover.endo[1]===src;
  ctx.strokeStyle=hl?'#7ff0b5':'#2f6a55';ctx.globalAlpha=hl?0.95:0.4;ctx.lineWidth=(hl?3:1.6)*DPR;ctx.setLineDash([7*DPR,6*DPR]);ctx.lineDashOffset=-anim*40*DPR;
  ctx.beginPath();ctx.moveTo(x1,y1);ctx.quadraticCurveTo(mx,my,x2,y2);ctx.stroke();ctx.setLineDash([]);
  const t=(anim*0.25)%1,px=(1-t)*(1-t)*x1+2*(1-t)*t*mx+t*t*x2,py=(1-t)*(1-t)*y1+2*(1-t)*t*my+t*t*y2;
  ctx.globalAlpha=0.9;ctx.fillStyle='#7ff0b5';ctx.beginPath();ctx.arc(px,py,2.6*DPR,0,7);ctx.fill();
  if(hl){ctx.fillStyle='#bfeee0';ctx.font=(12*DPR)+'px Inter';ctx.textAlign='center';ctx.fillText(h,mx,my-6*DPR);}}
 ctx.globalAlpha=1;
 // organs
 for(const t of names){const p=ORG[t];if(!p)continue;const x=p[0]*W,y=p[1]*H,r=(hover&&hover.tissue===t?36:32)*DPR;
  const g=ctx.createRadialGradient(x-r*0.3,y-r*0.3,r*0.2,x,y,r);const c=ocol(t);g.addColorStop(0,c);g.addColorStop(1,shade(c,-55));
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill();
  ctx.strokeStyle=hover&&hover.tissue===t?'#fff':shade(c,30);ctx.lineWidth=2*DPR;ctx.stroke();
  ctx.fillStyle='#eaf4ff';ctx.font='600 '+(12*DPR)+'px Inter';ctx.textAlign='center';ctx.fillText(t,x,y+r+15*DPR);
  ctx.fillStyle='#7f9ec0';ctx.font=(9.5*DPR)+'px Inter';ctx.fillText(TIS[t].cells.length+' cell types',x,y+r+28*DPR);}
 ctx.textAlign='left';}
function roundBody(){ctx.strokeStyle='#0f2138';ctx.lineWidth=2*DPR;ctx.beginPath();
 ctx.moveTo(W*0.40,H*0.20);ctx.lineTo(W*0.60,H*0.20);ctx.lineTo(W*0.63,H*0.55);ctx.lineTo(W*0.57,H*0.80);ctx.lineTo(W*0.43,H*0.80);ctx.lineTo(W*0.37,H*0.55);ctx.closePath();ctx.stroke();}
function drawTissue(){const t=TIS[tissue];
 ctx.fillStyle='#1a2f4a';ctx.font=(30*DPR)+'px Inter';ctx.textAlign='center';ctx.fillText(tissue,W/2,H*0.52+8*DPR);
 // edges
 for(const[s,r,pairs]of t.comm){if(s===r)continue;const a=nodeAt(s),b=nodeAt(r);if(!a||!b)continue;
  let vis=pairs;if(pwFilter)vis=pairs.filter(p=>p[4]===pwFilter);if(!vis.length)continue;
  const live=vis.filter(p=>p[0]!==koGene&&p[1]!==koGene).length;
  const dompw=domPathway(vis);const col=live<vis.length?'#e05a6a':pwc(dompw);
  const hl=selCell&&(s===selCell||r===selCell)||selEdge&&selEdge[0]===s&&selEdge[1]===r;
  const dim=(selCell&&s!==selCell&&r!==selCell)||(selEdge&&!(selEdge[0]===s&&selEdge[1]===r))||(pwFilter&&!vis.length);
  const mx=(a.x+b.x)/2+(b.y-a.y)*0.15,my=(a.y+b.y)/2-(b.x-a.x)*0.15;
  ctx.strokeStyle=hl?'#9fd0ff':col;ctx.globalAlpha=dim?0.07:(hl?0.95:0.5);ctx.lineWidth=Math.min(8,1+live*0.35)*DPR;
  const jux=vis[0]&&vis[0][5]==='juxtacrine';if(jux){ctx.setLineDash([3*DPR,4*DPR]);}
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(mx,my,b.x,b.y);ctx.stroke();ctx.setLineDash([]);
  const t2=(anim*0.3)%1,px=(1-t2)*(1-t2)*a.x+2*(1-t2)*t2*mx+t2*t2*b.x,py=(1-t2)*(1-t2)*a.y+2*(1-t2)*t2*my+t2*t2*b.y;
  ctx.globalAlpha=dim?0.07:0.9;ctx.fillStyle=col;ctx.beginPath();ctx.arc(px,py,2.4*DPR,0,7);ctx.fill();
  // arrowhead
  const ang=Math.atan2(b.y-py,b.x-px);ctx.beginPath();ctx.moveTo(b.x-Math.cos(ang)*24*DPR,b.y-Math.sin(ang)*24*DPR);ctx.lineTo(b.x-Math.cos(ang-0.4)*34*DPR,b.y-Math.sin(ang-0.4)*34*DPR);ctx.lineTo(b.x-Math.cos(ang+0.4)*34*DPR,b.y-Math.sin(ang+0.4)*34*DPR);ctx.closePath();ctx.fill();}
 ctx.globalAlpha=1;
 // nodes
 for(const n of nodes){const dim=selCell&&selCell!==n.name&&!conn(selCell,n.name);ctx.globalAlpha=dim?0.3:1;
  const g=ctx.createRadialGradient(n.x-n.r*0.3,n.y-n.r*0.3,n.r*0.2,n.x,n.y,n.r);const c=cellcol(n.name);g.addColorStop(0,c);g.addColorStop(1,shade(c,-45));
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,7);ctx.fill();
  ctx.strokeStyle=n.name===selCell?'#fff':n.name===hover?.cell?'#9fd0ff':shade(c,30);ctx.lineWidth=(n.name===selCell?3:1.5)*DPR;ctx.stroke();
  ctx.fillStyle='#eaf4ff';ctx.font='600 '+(11.5*DPR)+'px Inter';ctx.textAlign='center';wrap(n.name,n.x,n.y+n.r+14*DPR,150*DPR,13*DPR);}
 ctx.globalAlpha=1;ctx.textAlign='left';}
const CCOL={};const pal=['#e0788a','#4dd0a0','#e0b050','#6a9ee0','#c792ea','#00b0c0','#e08a5a','#8fbf6a','#d090c0'];
function cellcol(c){if(!(c in CCOL))CCOL[c]=pal[Object.keys(CCOL).length%pal.length];return CCOL[c];}
function nodeAt(n){return nodes.find(o=>o.name===n);}
function conn(a,b){return TIS[tissue].comm.some(([s,r])=>(s===a&&r===b)||(s===b&&r===a));}
function domPathway(pairs){const c={};pairs.forEach(p=>{if(p[4])c[p[4]]=(c[p[4]]||0)+1;});const e=Object.entries(c).sort((a,b)=>b[1]-a[1]);return e.length?e[0][0]:'';}
function wrap(txt,x,y,mw,lh){const w=txt.split(' ');let line='',yy=y;for(const q of w){if(ctx.measureText(line+q).width>mw&&line){ctx.fillText(line.trim(),x,yy);line='';yy+=lh;}line+=q+' ';}ctx.fillText(line.trim(),x,yy);}
// ---- animation ----
function loop(){view.s+=(tgt.s-view.s)*0.15;view.x+=(tgt.x-view.x)*0.15;view.y+=(tgt.y-view.y)*0.15;draw();requestAnimationFrame(loop);}
// ---- interaction ----
cv.addEventListener('mousemove',e=>{const mx=e.offsetX*DPR,my=e.offsetY*DPR;hover=null;const tip=document.getElementById('tip');tip.style.display='none';
 if(level=='body'){for(const t of names){const p=ORG[t];if(!p)continue;if((mx-p[0]*W)**2+(my-p[1]*H)**2<(34*DPR)**2){hover={tissue:t};tip.style.display='block';tip.style.left=(e.offsetX+14)+'px';tip.style.top=(e.offsetY+10)+'px';const top=Object.keys(TIS[t].pathways).slice(0,4).join(', ');tip.innerHTML=`<b>${t}</b><div class=lg>${TIS[t].cells.length} cell types &middot; ${TIS[t].comm.length} channels</div><div class=lg style=margin-top:3px>top signaling: ${top}</div>`;break;}}
  for(const[h,src,tt,rec,eff]of ENDO){const a=ORG[src],b=ORG[tt];if(!a||!b)continue;const cx=(a[0]*W+b[0]*W)/2+(b[1]*H-a[1]*H)*0.16,cy=(a[1]*H+b[1]*H)/2-(b[0]*W-a[0]*W)*0.16;if((mx-cx)**2+(my-cy)**2<(200*DPR)){hover={endo:[h,src,tt,rec,eff]};tip.style.display='block';tip.style.left=(e.offsetX+14)+'px';tip.style.top=(e.offsetY+10)+'px';tip.innerHTML=`<b>${h}</b> &middot; ${src} &rarr; ${tt}<div class=lg>${eff}</div>`;break;}}
  cv.style.cursor=hover?'pointer':'grab';draw();return;}
 for(const n of nodes){if((mx-n.x)**2+(my-n.y)**2<n.r*n.r){hover={cell:n.name};tip.style.display='block';tip.style.left=(e.offsetX+14)+'px';tip.style.top=(e.offsetY+10)+'px';tip.innerHTML=`<b>${n.name}</b><div class=lg>${(TIS[tissue].cellinfo[n.name].markers||[]).slice(0,5).join(', ')}</div>`;break;}}
 cv.style.cursor=hover?.cell?'pointer':'default';draw();});
cv.addEventListener('click',e=>{const mx=e.offsetX*DPR,my=e.offsetY*DPR;
 if(level=='body'){for(const t of names){const p=ORG[t];if(!p)continue;if((mx-p[0]*W)**2+(my-p[1]*H)**2<(34*DPR)**2){enterTissue(t);return;}}
  for(const[h,src,tt,rec,eff]of ENDO){const a=ORG[src],b=ORG[tt];if(!a||!b)continue;const cx=(a[0]*W+b[0]*W)/2+(b[1]*H-a[1]*H)*0.16,cy=(a[1]*H+b[1]*H)/2-(b[0]*W-a[0]*W)*0.16;if((mx-cx)**2+(my-cy)**2<(220*DPR)){showEndo([h,src,tt,rec,eff]);return;}}return;}
 for(const n of nodes){if((mx-n.x)**2+(my-n.y)**2<n.r*n.r){selCell=n.name;selEdge=null;showCell(n.name);draw();return;}}
 let best=null,bd=1000*DPR*DPR;for(const[s,r,pairs]of TIS[tissue].comm){if(s===r)continue;const a=nodeAt(s),b=nodeAt(r);const cx=(a.x+b.x)/2+(b.y-a.y)*0.15,cy=(a.y+b.y)/2-(b.x-a.x)*0.15;const d=(mx-cx)**2+(my-cy)**2;if(d<bd){bd=d;best=[s,r,pairs];}}
 if(best){selEdge=best;selCell=null;showEdge(best);draw();}});
function enterTissue(t){level='tissue';tissue=t;selCell=selEdge=null;layout();showTissue(t);draw();document.getElementById('back').style.display='block';populatePW();}
function backBody(){level='body';selCell=selEdge=null;document.getElementById('panel').classList.remove('open');document.getElementById('back').style.display='none';draw();}
document.getElementById('bodyBtn').onclick=backBody;document.getElementById('close').onclick=()=>document.getElementById('panel').classList.remove('open');
// ---- panels ----
function open(h){document.getElementById('body').innerHTML=h;document.getElementById('panel').classList.add('open');}
function showEndo(e){open(`<h1>${e[0]}</h1><div class=sub>endocrine hormone</div><div class=row><span class=v>${e[1]}</span> <span class=arrow>&rarr;</span> <span class=v>${e[2]}</span></div><div class=row><span class=k>receptor</span> <span class=rec>${e[3]}</span></div><h3>effect</h3><div class=lg>${e[4]}</div><div class=row style=margin-top:8px><span class=chip onclick="enterTissue('${e[1]}')">open ${e[1]}</span> <span class=chip onclick="enterTissue('${e[2]}')">open ${e[2]}</span></div>`);}
function showTissue(t){const T=TIS[t];const pws=Object.entries(T.pathways).slice(0,10);
 open(`<h1>${t}</h1><div class=sub>${T.cells.length} cell types &middot; ${T.comm.length} communication channels</div>
  <h3>dominant signaling pathways</h3><div>${pws.map(([p,n])=>`<span class=chip style="border-left:3px solid ${pwc(p)}" onclick="setPW('${p}')">${p} <span class=lg>${n}</span></span>`).join('')}</div>
  <h3>strongest cell-cell links</h3>${(T.toplinks||[]).slice(0,6).map(l=>`<div class=row><a class=chip onclick="showEdgeByName('${l[0]}','${l[1]}')">${l[0]} &rarr; ${l[1]}</a> <span class=lg>${l[3]} channels</span></div>`).join('')}
  <h3>cell types</h3><div>${T.cells.map(c=>`<span class=chip onclick="selCell='${c}';showCell('${c}');draw()">${c}</span>`).join('')}</div>`);}
function showCell(name){const info=TIS[tissue].cellinfo[name];const out=TIS[tissue].comm.filter(([s,r])=>s===name&&r!==name),inc=TIS[tissue].comm.filter(([s,r])=>r===name&&s!==name);
 open(`<h1>${name}</h1><div class=sub>${tissue} &middot; cell type</div>
  <h3>identity markers</h3><div>${(info.markers||[]).map(g=>`<span class=chip>${g}</span>`).join('')||'&mdash;'}</div>
  <h3>signals to (${out.length})</h3>${out.map(([s,r])=>`<div class=row><span class=chip onclick="showEdgeByName('${s}','${r}')">&rarr; ${r}</span></div>`).join('')||'<div class=lg>—</div>'}
  <h3>receives from (${inc.length})</h3>${inc.map(([s,r])=>`<div class=row><span class=chip onclick="showEdgeByName('${s}','${r}')">${s} &rarr;</span></div>`).join('')||'<div class=lg>—</div>'}`);}
function showEdgeByName(s,r){const e=TIS[tissue].comm.find(x=>x[0]===s&&x[1]===r);if(e){selEdge=e;selCell=null;showEdge(e);draw();}}
function showEdge(e){const[s,r,pairs]=e;let vis=pwFilter?pairs.filter(p=>p[4]===pwFilter):pairs;
 const recs=[...new Set(vis.map(p=>p[1]))].filter(rc=>DOWN[rc]).slice(0,3);
 open(`<h1>${s} &rarr; ${r}</h1><div class=sub>${tissue} &middot; ${vis.length} ligand&rarr;receptor channels</div>
  <h3>channels</h3>${vis.slice(0,20).map(p=>{const brk=p[0]===koGene||p[1]===koGene;const dl=DRUGS[p[0]]||[],dr=DRUGS[p[1]]||[];
   return `<div class="lr ${brk?'brk':''}"><span class=lig>${p[0]}</span><span class=arrow>&rarr;</span><span class=rec>${p[1]}</span>${p[4]?`<span class=pwtag style="background:${pwc(p[4])}33;color:${pwc(p[4])}">${p[4]}</span>`:''} <span class=lg>${p[5]||''}</span>${(dl.length||dr.length)?`<div class=drug>&#128138; ${[...dl,...dr].slice(0,3).join(', ')}</div>`:''}${brk?' <span class=warn>&#9888;</span>':''}</div>`;}).join('')}
  ${recs.length?`<h3>downstream inside ${r}</h3>${recs.map(rc=>{const d=DOWN[rc];return `<div class=lr><span class=rec>${rc}</span> <span class=arrow>&rarr;</span> ${d.tfs.length?'TFs <span class=v>'+d.tfs.slice(0,3).join(', ')+'</span> &rarr; ':''}<span class=lg>${(d.targets||[]).slice(0,7).join(', ')}</span></div>`;}).join('')}`:''}`);}
window.enterTissue=enterTissue;window.showEdgeByName=showEdgeByName;window.showCell=showCell;window.setPW=setPW;window.selCell=null;
// pathway filter
function populatePW(){const s=document.getElementById('pw');const pws=Object.keys(TIS[tissue].pathways);s.innerHTML='<option value="">all pathways</option>'+pws.map(p=>`<option>${p}</option>`).join('');}
document.getElementById('pw').onchange=function(){pwFilter=this.value;draw();};
function setPW(p){pwFilter=p;document.getElementById('pw').value=p;draw();}
// knockout
const koI=document.getElementById('ko');koI.addEventListener('change',()=>{koGene=koI.value.trim().toUpperCase();applyKO();});
function applyKO(){if(level!=='tissue'){draw();return;}let broke=0;const links=new Set();
 TIS[tissue].comm.forEach(([s,r,pairs])=>pairs.forEach(p=>{if(p[0]===koGene||p[1]===koGene){broke++;links.add(s+' &rarr; '+r);}}));
 if(koGene)open(`<h1 class=warn>Knock out ${koGene}</h1><div class=sub>${tissue}</div><div class=row>${broke?`<span class=warn>&#9888; ${broke} channels lost</span> across ${links.size} links`:'<span class=ok>no channel uses '+koGene+' here</span>'}</div>${broke?'<h3>disrupted links</h3>'+[...links].map(l=>`<div class=row>${l}</div>`).join(''):''}`);
 draw();}
document.getElementById('reset').onclick=()=>{koGene='';koI.value='';pwFilter='';document.getElementById('pw').value='';selCell=selEdge=null;backBody();};
document.getElementById('back').style.display='none';
document.getElementById('legend').innerHTML='<b>signaling pathways</b> (edge color)<br>'+['COLLAGEN','WNT','CCL','TGFb','BMP','FGF','MHC-I'].map(p=>`<span style="color:${pwc(p)}">&#9679;</span> ${p}`).join('  ');
document.getElementById('hud').innerHTML='BODY: click an organ to enter &middot; hover a hormone &middot; TISSUE: click a cell / channel, filter by pathway, knock out a gene';
resize();loop();
</script></body></html>"""
HTML=HTML.replace("__DATA__",json.dumps(D,separators=(',',':')))
(OUT/"tissue_explorer.html").write_text(HTML)
print("wrote tissue_explorer.html (%d KB)"%(len(HTML)//1024))
