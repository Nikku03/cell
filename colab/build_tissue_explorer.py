"""Tissue Explorer — a multi-cell tissue with cell-cell communication (ligand->receptor). Pick a
tissue, see its cell types wired by signaling; click a cell or a channel; knock out a gene and watch
which cell-cell signals break. Reads tissue_model.json. Self-contained. SEPARATE from the cell model."""
import json
from pathlib import Path
OUT=Path("outputs/orphan")
D=json.load(open(OUT/"tissue_model.json"))
HTML=r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Tissue Explorer</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#04070d;color:#e8f0f8;font-family:-apple-system,'Segoe UI',Inter,sans-serif;overflow:hidden;user-select:none}
#top{height:50px;display:flex;align-items:center;gap:8px;padding:0 15px;background:#0a1522;border-bottom:1px solid #16304c;flex-wrap:wrap}
.brand{font-weight:700;color:#cfe6ff}.brand small{color:#5f7fa6;font-weight:400;font-size:11px}
.tab{padding:6px 11px;border-radius:8px;background:#12283f;color:#bcd6f2;cursor:pointer;font-size:12px}.tab.on{background:#1f6feb;color:#fff}
.spacer{flex:1}input{background:#07101b;border:1px solid #24405f;color:#e8f0f8;border-radius:8px;padding:6px 10px;font-size:12px;width:150px}
#wrap{display:flex;height:calc(100vh - 51px)}#cv{flex:1;display:block;cursor:default}
#side{width:400px;background:#060c16;border-left:1px solid #16304c;overflow:auto;padding:16px 18px}
h1{font-size:20px;margin:0 0 3px;color:#eaf4ff}.sub{color:#6f8fb5;font-size:12px;margin-bottom:12px}
h3{color:#8fd0ff;font-size:12.5px;margin:16px 0 5px;border-bottom:1px solid #12283f;padding-bottom:3px}
.k{color:#7f9ec0}.v{color:#eaf4ff;font-weight:600}.row{margin:4px 0;font-size:13px}.lg{color:#8aa3c0;font-size:11.5px}
.chip{display:inline-block;font-size:11px;padding:3px 8px;margin:2px;border-radius:7px;background:#12283f;color:#bcd6f2;cursor:pointer}.chip:hover{background:#1c4066}
.lr{padding:5px 9px;border-radius:7px;background:#0b1a2b;margin:4px 0;font-size:12.5px}
.lig{color:#7ff0b5}.rec{color:#ffcf8f}.brk{opacity:.4;text-decoration:line-through}
.arrow{color:#5c7fa0;margin:0 4px}.act{color:#4dd97a}.inh{color:#e05a6a}
#hud{position:fixed;left:14px;bottom:12px;font-size:11px;color:#5f7fa6}
.warn{color:#ff9db0}.ok{color:#7ff0b5}
</style></head><body>
<div id=top>
 <span class=brand>Tissue Explorer <small>— cell-cell communication &middot; HPA + Omnipath &middot; separate from the cell model</small></span>
 <span class=spacer></span>
 <input id=ko placeholder="knock out a gene"> <span class=tab id=clr>clear KO</span>
</div>
<div id=wrap><canvas id=cv></canvas><div id=side><div id=body></div></div></div>
<div id=hud></div>
<script>
const D=__DATA__;const TIS=D.tissues;const names=Object.keys(TIS);const DOWNSTREAM=D.downstream||{};
let tissue=names[0],selCell=null,selEdge=null,koGene='',hover=null;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');let W,H,DPR=Math.min(devicePixelRatio||1,2);
let nodes=[]; // {name, x, y, r}
function layout(){const t=TIS[tissue],cts=t.cells;nodes=cts.map((c,i)=>{const a=-Math.PI/2+i/cts.length*6.2832;
  return {name:c,x:W/2+Math.cos(a)*Math.min(W,H)*0.30,y:H/2+Math.sin(a)*Math.min(W,H)*0.30,r:38*DPR};});}
function resize(){W=cv.width=cv.clientWidth*DPR;H=cv.height=cv.clientHeight*DPR;cv.style.width=cv.clientWidth+'px';cv.style.height=cv.clientHeight+'px';layout();draw();}
addEventListener('resize',resize);
const CCOL={};const pal=['#e0788a','#4dd0a0','#e0b050','#6a9ee0','#c792ea','#00b0c0','#e08a5a','#8fbf6a'];
function ccol(c){if(!(c in CCOL))CCOL[c]=pal[Object.keys(CCOL).length%pal.length];return CCOL[c];}
function nodeAt(n){return nodes.find(o=>o.name===n);}
function channels(){return TIS[tissue].comm;}
function edgeBroken(pairs){return pairs.filter(p=>p[0]!==koGene&&p[1]!==koGene).length;}
function draw(){ctx.setTransform(1,0,0,1,0,0);const bg=ctx.createRadialGradient(W/2,H/2,50,W/2,H/2,Math.max(W,H)*0.7);
 bg.addColorStop(0,'#0a1422');bg.addColorStop(1,'#04070d');ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
 // tissue label
 ctx.fillStyle='#2a3f5a';ctx.font=(34*DPR)+'px Inter';ctx.textAlign='center';ctx.fillText(tissue,W/2,H/2+10*DPR);
 // edges (communication)
 for(const[s,r,pairs]of channels()){if(s===r)continue;const a=nodeAt(s),b=nodeAt(r);if(!a||!b)continue;
  const live=edgeBroken(pairs),tot=pairs.length;const w=Math.min(9,1+live*0.4)*DPR;
  const hl=selCell&&(s===selCell||r===selCell)||selEdge&&selEdge[0]===s&&selEdge[1]===r;
  const dim=(selCell&&s!==selCell&&r!==selCell)||(selEdge&&!(selEdge[0]===s&&selEdge[1]===r));
  const mx=(a.x+b.x)/2+(b.y-a.y)*0.14,my=(a.y+b.y)/2-(b.x-a.x)*0.14;
  ctx.strokeStyle=hl?'#9fd0ff':(live<tot?'#e05a6a':ccol(s));ctx.globalAlpha=dim?0.08:(hl?0.95:0.5);ctx.lineWidth=w;
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(mx,my,b.x,b.y);ctx.stroke();
  // arrowhead near b
  const t=0.86,px=(1-t)*(1-t)*a.x+2*(1-t)*t*mx+t*t*b.x,py=(1-t)*(1-t)*a.y+2*(1-t)*t*my+t*t*b.y;
  const ang=Math.atan2(b.y-py,b.x-px);ctx.fillStyle=ctx.strokeStyle;ctx.globalAlpha=dim?0.08:0.8;
  ctx.beginPath();ctx.moveTo(b.x-Math.cos(ang)*26*DPR,b.y-Math.sin(ang)*26*DPR);
  ctx.lineTo(b.x-Math.cos(ang-0.4)*36*DPR,b.y-Math.sin(ang-0.4)*36*DPR);ctx.lineTo(b.x-Math.cos(ang+0.4)*36*DPR,b.y-Math.sin(ang+0.4)*36*DPR);ctx.closePath();ctx.fill();}
 ctx.globalAlpha=1;
 // nodes (cell types)
 for(const n of nodes){const dim=selCell&&selCell!==n.name&&!connected(selCell,n.name);
  ctx.globalAlpha=dim?0.3:1;
  const g=ctx.createRadialGradient(n.x-n.r*0.3,n.y-n.r*0.3,n.r*0.2,n.x,n.y,n.r);const c=ccol(n.name);
  g.addColorStop(0,c);g.addColorStop(1,shade(c,-40));ctx.fillStyle=g;ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,7);ctx.fill();
  ctx.strokeStyle=(n.name===selCell)?'#fff':(n.name===hover?'#9fd0ff':shade(c,30));ctx.lineWidth=(n.name===selCell?3:1.5)*DPR;ctx.stroke();
  ctx.fillStyle='#eaf4ff';ctx.font='600 '+(12*DPR)+'px Inter';ctx.textAlign='center';
  wrapText(n.name,n.x,n.y+n.r+16*DPR,150*DPR,13*DPR);}
 ctx.globalAlpha=1;}
function connected(a,b){return channels().some(([s,r])=>(s===a&&r===b)||(s===b&&r===a));}
function shade(hex,d){const n=parseInt(hex.slice(1),16);let R=(n>>16)+d,Gc=((n>>8)&255)+d,B=(n&255)+d;R=Math.max(0,Math.min(255,R));Gc=Math.max(0,Math.min(255,Gc));B=Math.max(0,Math.min(255,B));return'#'+((R<<16)|(Gc<<8)|B).toString(16).padStart(6,'0');}
function wrapText(txt,x,y,maxw,lh){const words=txt.split(' ');let line='',yy=y;ctx.fillStyle='#dfeaf5';
 for(const w of words){if(ctx.measureText(line+w).width>maxw&&line){ctx.fillText(line.trim(),x,yy);line='';yy+=lh;}line+=w+' ';}ctx.fillText(line.trim(),x,yy);}
// interaction
cv.addEventListener('mousemove',e=>{const mx=e.offsetX*DPR,my=e.offsetY*DPR;hover=null;
 for(const n of nodes){if((mx-n.x)**2+(my-n.y)**2<n.r*n.r){hover=n.name;break;}}
 cv.style.cursor=hover?'pointer':'default';draw();});
cv.addEventListener('click',e=>{const mx=e.offsetX*DPR,my=e.offsetY*DPR;
 for(const n of nodes){if((mx-n.x)**2+(my-n.y)**2<n.r*n.r){selCell=n.name;selEdge=null;showCell(n.name);draw();return;}}
 // edge click: nearest edge midpoint
 let best=null,bd=900*DPR*DPR;
 for(const[s,r,pairs]of channels()){if(s===r)continue;const a=nodeAt(s),b=nodeAt(r);const cx=(a.x+b.x)/2+(b.y-a.y)*0.14,cy=(a.y+b.y)/2-(b.x-a.x)*0.14;const d=(mx-cx)**2+(my-cy)**2;if(d<bd){bd=d;best=[s,r,pairs];}}
 if(best){selEdge=best;selCell=null;showEdge(best);draw();}});
// panels
function showCell(name){const t=TIS[tissue],info=t.cellinfo[name];
 const out=channels().filter(([s,r])=>s===name&&r!==name),inc=channels().filter(([s,r])=>r===name&&s!==name);
 let h=`<h1>${name}</h1><div class=sub>${tissue} &middot; cell type</div>
  <h3>identity markers</h3><div>${info.markers.map(g=>`<span class=chip>${g}</span>`).join('')||'<span class=lg>—</span>'}</div>
  <h3>sends signals to (${out.length})</h3>${out.map(([s,r,p])=>`<div class=row><a class=chip onclick="selEdge=['${s}','${r}'];showEdgeByName('${s}','${r}')">&rarr; ${r}</a> <span class=lg>${edgeBroken(p)} channels</span></div>`).join('')||'<div class=lg>—</div>'}
  <h3>receives signals from (${inc.length})</h3>${inc.map(([s,r,p])=>`<div class=row><a class=chip onclick="showEdgeByName('${s}','${r}')">${s} &rarr;</a> <span class=lg>${edgeBroken(p)} channels</span></div>`).join('')||'<div class=lg>—</div>'}
  <h3>secreted ligands (specific)</h3><div class=lg>${info.ligands.slice(0,40).join(', ')}</div>`;
 document.getElementById('body').innerHTML=h;}
function showEdgeByName(s,r){const e=channels().find(x=>x[0]===s&&x[1]===r);if(e){selEdge=e;selCell=null;showEdge(e);draw();}}
function showEdge(e){const[s,r,pairs]=e;const live=pairs.filter(p=>p[0]!==koGene&&p[1]!==koGene);
 let h=`<h1>${s} &rarr; ${r}</h1><div class=sub>${tissue} &middot; cell-cell communication &middot; ${pairs.length} channels${koGene?` &middot; <span class=warn>${pairs.length-live.length} broken by ${koGene} KO</span>`:''}</div>
  <h3>ligand &rarr; receptor channels</h3>
  ${pairs.map(p=>{const brk=p[0]===koGene||p[1]===koGene;return `<div class="lr ${brk?'brk':''}"><span class=lig>${p[0]}</span> <span class=arrow>&rarr;</span> <span class=rec>${p[1]}</span> ${p[2]>0?'<span class=act>(activates)</span>':p[2]<0?'<span class=inh>(inhibits)</span>':''}${brk?' <span class=warn>&#9888; lost</span>':''}</div>`;}).join('')}`;
 const recs=[...new Set(pairs.map(p=>p[1]))].filter(rc=>DOWNSTREAM[rc]).slice(0,4);
 if(recs.length){h+=`<h3>downstream response inside ${r}</h3><div class=lg>what these receptors trigger in the receiving cell (SIGNOR + CollecTRI)</div>`;
  recs.forEach(rc=>{const ds=DOWNSTREAM[rc];h+=`<div class=lr><span class=rec>${rc}</span> <span class=arrow>&rarr;</span> ${ds.tfs.length?'TFs <span class=v>'+ds.tfs.slice(0,4).join(', ')+'</span> <span class=arrow>&rarr;</span> ':''}<span class=lg>${(ds.targets||[]).slice(0,8).join(', ')||ds.sig.slice(0,6).join(', ')}</span></div>`;});}
 document.getElementById('body').innerHTML=h;}
window.showEdgeByName=showEdgeByName;
// knockout
const koI=document.getElementById('ko');
koI.addEventListener('change',()=>{koGene=koI.value.trim().toUpperCase();applyKO();});
document.getElementById('clr').onclick=()=>{koGene='';koI.value='';applyKO();};
function applyKO(){let broke=0,ch=0;const affected=new Set();const lostRec=new Set();
 channels().forEach(([s,r,pairs])=>{pairs.forEach(p=>{ch++;if(p[0]===koGene||p[1]===koGene){broke++;affected.add(s+' -> '+r);if(p[0]===koGene&&DOWNSTREAM[p[1]])lostRec.add(r+'|'+p[1]);if(p[1]===koGene)lostRec.add(r+'|'+koGene);}});});
 // downstream genes that lose input in the receivers
 const dnGenes=new Set();lostRec.forEach(x=>{const rc=x.split('|')[1];(DOWNSTREAM[rc]&&DOWNSTREAM[rc].targets||[]).forEach(g=>dnGenes.add(g));});
 if(koGene){document.getElementById('body').innerHTML=`<h1 class=warn>Knock out ${koGene}</h1><div class=sub>${tissue} &middot; effect on cell-cell communication</div>
   <div class=row>${broke?`<span class=warn>&#9888; ${broke} communication channels lost</span> across <span class=v>${affected.size}</span> cell-cell links`:'<span class=ok>no cell-cell channel in this tissue uses '+koGene+'</span>'}</div>
   ${broke?'<h3>disrupted links</h3>'+[...affected].map(a=>`<div class=row>${a}</div>`).join(''):''}
   ${lostRec.size?`<h3>propagated into receivers</h3><div class=lg>receptors losing input &rarr; downstream genes affected inside the receiving cell</div>${[...lostRec].slice(0,8).map(x=>{const[cell,rc]=x.split('|');return `<div class=lr><span class=v>${cell}</span>: <span class=rec>${rc}</span> input lost</div>`;}).join('')}${dnGenes.size?`<div class=row style=margin-top:6px><span class=k>downstream genes affected:</span> <span class=lg>${[...dnGenes].slice(0,20).join(', ')}</span></div>`:''}`:''}
   <div class=lg style=margin-top:8px>Red/thin edges on the map = weakened. Click a link for the lost channels + receiver response.</div>`;}
 draw();}
// tissue tabs
const bar=document.getElementById('top');const spacer=bar.querySelector('.spacer');
names.forEach(nm=>{const t=document.createElement('span');t.className='tab'+(nm===tissue?' on':'');t.textContent=nm;
 t.onclick=()=>{tissue=nm;selCell=selEdge=null;document.querySelectorAll('#top .tab').forEach(x=>{if(names.includes(x.textContent))x.classList.toggle('on',x.textContent===nm)});layout();draw();welcome();};
 bar.insertBefore(t,spacer);});
function welcome(){document.getElementById('body').innerHTML=`<h1>${tissue}</h1><div class=sub>${TIS[tissue].cells.length} cell types &middot; ${TIS[tissue].comm.length} communication channels</div>
  <div class=lg>A tissue is many cells talking. Each node is a <b>cell type</b> (HPA markers); each arrow is <b>cell-cell communication</b> — a ligand made by the sender binding a receptor on the receiver (Omnipath). <br><br>Click a <b>cell type</b> to see who it signals; click an <b>arrow</b> for the ligand&rarr;receptor channels; <b>knock out a gene</b> to see which signals break.</div>
  <div style=margin-top:10px>${TIS[tissue].cells.map(c=>`<span class=chip onclick="selCell='${c}';showCell('${c}');draw()">${c}</span>`).join('')}</div>`;}
document.getElementById('hud').innerHTML='click a cell type or a communication arrow &middot; knock out a gene (e.g. COL1A1, TGFB1, VEGFA)';
resize();welcome();
</script></body></html>"""
HTML=HTML.replace("__DATA__",json.dumps(D,separators=(',',':')))
(OUT/"tissue_explorer.html").write_text(HTML)
print("wrote tissue_explorer.html (%d KB)"%(len(HTML)//1024))
