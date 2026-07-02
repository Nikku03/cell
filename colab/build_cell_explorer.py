"""Cell Explorer — a premium, continuous-zoom, high-realism interactive human cell.
Reads cell_complete.json (+ gene_uniprot.json for AlphaFold on-demand). Self-contained HTML.
Phase 1: realistic organelles, smooth zoom/pan with level-of-detail, hover-glow, click-to-fly
glass panel (identity, life-history, networks, drugs/disease/biomarkers, provenance, Explain-Why,
AlphaFold structure), selective animated pathways (5-10 at a time, layer toggle), mutation-mode
propagation, cell-type switch, universal search."""
import json
from pathlib import Path
OUT=Path("outputs/orphan")
D=json.load(open(OUT/"cell_complete.json"))
try: ACC=json.load(open(OUT/"gene_uniprot.json"))
except Exception: ACC={}
D["acc"]=ACC
HTML=r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Human Cell Explorer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#04070d;color:#e8f0f8;font-family:-apple-system,'Segoe UI',Inter,Roboto,sans-serif;overflow:hidden;user-select:none}
#cv{position:fixed;inset:0;display:block;cursor:grab}#cv.drag{cursor:grabbing}
/* top bar */
#top{position:fixed;top:0;left:0;right:0;height:52px;display:flex;align-items:center;gap:8px;padding:0 14px;z-index:20;
 background:linear-gradient(#04070de6,#04070d88 70%,#04070d00);pointer-events:none}
#top>*{pointer-events:auto}
.brand{font-weight:700;letter-spacing:.3px;font-size:15px;color:#cfe6ff;margin-right:6px}
.brand small{color:#5f7fa6;font-weight:400;font-size:11px}
.pill{background:#0d1b2ecc;border:1px solid #1e3a5f;border-radius:9px;padding:6px 11px;font-size:12px;color:#cfe6ff;cursor:pointer;backdrop-filter:blur(8px);transition:.15s}
.pill:hover{background:#16304c;border-color:#3a6ea5}.pill.on{background:#1f6feb;border-color:#4a9eff;color:#fff;box-shadow:0 0 14px #1f6feb66}
select.pill{appearance:none}
#search{background:#0d1b2ecc;border:1px solid #1e3a5f;border-radius:9px;padding:6px 11px;color:#e8f0f8;font-size:12px;width:150px;backdrop-filter:blur(8px)}
#search:focus{outline:none;border-color:#4a9eff;box-shadow:0 0 12px #1f6feb44}
.spacer{flex:1}
/* tooltip */
#tip{position:fixed;z-index:30;pointer-events:none;background:#081522f2;border:1px solid #274b73;border-radius:11px;
 padding:9px 12px;font-size:12px;display:none;max-width:270px;box-shadow:0 8px 30px #000a;backdrop-filter:blur(10px)}
#tip b{color:#7fd1ff;font-size:13px}#tip .r{margin-top:3px;color:#9fb6cf}
.badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:6px;margin:2px 3px 0 0}
.b-ess{background:#5a1526;color:#ff9db0}.b-drug{background:#123b2a;color:#7ff0b5}.b-dis{background:#3a2a10;color:#ffcf8f}.b-tf{background:#1a2f5a;color:#a9c8ff}
/* side panel */
#panel{position:fixed;top:0;right:-440px;width:430px;height:100%;z-index:25;overflow-y:auto;padding:60px 20px 30px;
 background:linear-gradient(#060c16f2,#04070df8);border-left:1px solid #16304c;backdrop-filter:blur(16px);
 transition:right .45s cubic-bezier(.2,.8,.2,1);box-shadow:-10px 0 40px #0008}
#panel.open{right:0}
#panel h1{font-size:24px;margin:0 0 2px;color:#eaf4ff;letter-spacing:.3px}
#panel .sub{color:#6f8fb5;font-size:12px;margin-bottom:12px}
.sec{border:1px solid #12283f;border-radius:12px;margin:9px 0;overflow:hidden;background:#0a1522aa}
.sec>.h{padding:9px 13px;font-size:12px;font-weight:600;color:#8fd0ff;cursor:pointer;display:flex;justify-content:space-between;align-items:center}
.sec>.h:hover{background:#0f2137}.sec>.b{padding:2px 13px 11px;font-size:12.5px;line-height:1.55;color:#c4d5e8;display:none}
.sec.open>.b{display:block}.sec>.h .ar{color:#4a6c94;transition:.2s}.sec.open>.h .ar{transform:rotate(90deg)}
.k{color:#7f9ec0}.v{color:#eaf4ff;font-weight:600}.row{margin:4px 0}.warn{color:#ff8a9c}.ok{color:#7ff0b5}.lg{color:#8aa3c0;font-size:11.5px}
.step{margin:3px 0;padding:4px 9px;border-left:2px solid #2e6fd6;background:#0c1c30;border-radius:0 6px 6px 0}
a.g{color:#7fd1ff;cursor:pointer}a.g:hover{text-decoration:underline}
.chip{display:inline-block;font-size:11px;padding:3px 8px;margin:2px;border-radius:7px;background:#12283f;color:#bcd6f2;cursor:pointer}.chip:hover{background:#1c4066}
#struct{width:100%;height:250px;position:relative;background:#02040a;border-radius:10px;margin-top:6px;display:none}
#explain{background:linear-gradient(120deg,#0a1e33,#0d2942);border:1px solid #1f4e7a;border-radius:12px;padding:12px 14px;margin:9px 0;font-size:12.5px;line-height:1.6;color:#d6e6f7}
#explain b{color:#8fd0ff}
#hud{position:fixed;left:14px;bottom:14px;z-index:20;font-size:11px;color:#5f7fa6;background:#060c16aa;border:1px solid #12283f;border-radius:9px;padding:7px 11px;backdrop-filter:blur(8px)}
#legend{position:fixed;right:14px;bottom:14px;z-index:20;font-size:11px;color:#8aa3c0;background:#060c16cc;border:1px solid #12283f;border-radius:10px;padding:9px 12px;backdrop-filter:blur(8px);max-width:210px}
#legend b{color:#cfe6ff;font-size:11px}
.dot{display:inline-block;width:8px;height:8px;border-radius:8px;margin-right:5px;vertical-align:middle}
#close{position:absolute;top:16px;right:16px;cursor:pointer;color:#6f8fb5;font-size:20px;line-height:1}#close:hover{color:#fff}
</style></head><body>
<canvas id=cv></canvas>
<div id=top>
 <span class=brand>Human Cell Explorer <small>&mdash; 16,492 proteins &middot; drag to pan &middot; scroll to zoom</small></span>
 <span class=spacer></span>
 <input id=search placeholder="search gene / protein">
 <select id=ct class=pill title="cell type"></select>
 <span class=pill id=mMut title="mutation mode">Mutate</span>
 <span class=pill id=reset>Reset view</span>
</div>
<div id=tip></div>
<div id=hud></div>
<div id=legend></div>
<div id=panel><span id=close>&times;</span><div id=pbody></div></div>
<script>
const D=__DATA__;const G=D.genes,N=G.length;
const byName={};G.forEach((g,i)=>byName[g.name]=i);
const OUT={},IN={},PP={},SGN={},SIGO={},SIGI={};
for(const e of D.reg){const a=e[0],b=e[1],s=e[2]||0;(OUT[a]=OUT[a]||[]).push(b);(IN[b]=IN[b]||[]).push(a);SGN[a+','+b]=s;}
for(const[a,b]of D.ppi){(PP[a]=PP[a]||[]).push(b);(PP[b]=PP[b]||[]).push(a);}
for(const e of (D.sig||[])){(SIGO[e[0]]=SIGO[e[0]]||[]).push(e[1]);(SIGI[e[1]]=SIGI[e[1]]||[]).push(e[0]);}
const CODEP=D.codep||{},DRUGS=D.drugs||{},BIOM=D.biomarkers||{},G2C=D.gene2cplx||{},ABUND=D.abund||{},ACC=D.acc||{},PTM=D.ptm||{},OTD=D.otdis||{};
// ---------- compartment geometry (real positions, illustrative shapes) ----------
const W0=1600,H0=1000,CX=W0/2,CY=H0/2;
const REG={
 nucleus:{x:560,y:520,rx:250,ry:210,fill:['#2a2350','#141033'],edge:'#8f7fd1',label:'Nucleus'},
 cytoplasm:{x:CX,y:CY,rx:720,ry:450,label:''},
 mitochondrion:{multi:[[1050,360,120,52],[1180,650,105,46],[760,820,110,48],[430,760,96,42]],fill:['#7a2b52','#3a1226'],edge:'#e06a9e',label:'Mitochondria'},
 ER:{x:560,y:520,ring:290,fill:'#c07a4a',label:'Endoplasmic reticulum'},
 Golgi:{x:930,y:430,rx:120,ry:56,fill:'#3fb0a0',label:'Golgi'},
 lysosome:{multi:[[1230,470,40,40],[350,540,32,32]],fill:['#b07820','#4a3208'],edge:'#f0b84a',label:'Lysosome'},
 peroxisome:{multi:[[720,300,26,26],[1090,820,24,24]],fill:['#2f7a3a','#123018'],edge:'#6fd07a',label:'Peroxisome'},
 endosome:{multi:[[1120,540,34,30],[430,420,30,28]],fill:'#5a6aa0',edge:'#9fb0e0',label:'Endosome'},
 'plasma membrane':{ring:true,label:'Plasma membrane'},membrane:{ring:true},
 extracellular:{out:true,label:'Extracellular'},
 cytoskeleton:{x:CX,y:CY,rx:700,ry:440,fibers:1,label:''},
 unknown:{x:1420,y:150,rx:110,ry:90,label:'Unassigned'}
};
const PCOL={'transcription':'#e05a6a','translation':'#43a86a','replication/repair':'#c99a3a','transport/uptake':'#3a86e0','metabolism':'#e08a3a','degradation':'#8a6ad0','trafficking/secretion':'#00b0c0','signaling':'#e0d040','immune':'#d06ab0','structure/cytoskeleton':'#7a9a6a','cell-death':'#e04a4a','other':'#5a7a9a'};
function pcol(g){return PCOL[g.proc]||'#5a7a9a';}
// deterministic protein positions inside compartments
const POS=new Float32Array(N*2);
function hash(i){let h=(i*2654435761)>>>0;return[(h%10007)/10007,((h/10007|0)%10007)/10007];}
for(let i=0;i<N;i++){const c=G[i].comp,[u,v]=hash(i+7),r=REG[c]||REG.unknown;let x,y,a=u*6.2832;
 if(r.ring){x=CX+Math.cos(a)*715;y=CY+Math.sin(a)*452;}
 else if(r.out){const rr=1.04+v*0.12;x=CX+Math.cos(a)*720*rr;y=CY+Math.sin(a)*452*rr;}
 else if(r.multi){const m=r.multi[i%r.multi.length];x=m[0]+Math.cos(a)*m[2]*(0.3+v*0.6);y=m[1]+Math.sin(a)*m[3]*(0.3+v*0.6);}
 else if(r.ring===undefined&&r.fill&&c=='ER'){const rr=r.ring||290,q=rr*(0.72+v*0.5);x=r.x+Math.cos(a)*q;y=r.y+Math.sin(a)*q*0.82;}
 else if(c=='ER'){const q=r.ring*(0.72+v*0.5);x=r.x+Math.cos(a)*q;y=r.y+Math.sin(a)*q*0.82;}
 else{const rr=Math.sqrt(v);x=(r.x||CX)+Math.cos(a)*(r.rx||200)*rr*0.92;y=(r.y||CY)+Math.sin(a)*(r.ry||160)*rr*0.92;}
 POS[i*2]=x;POS[i*2+1]=y;}
// ---------- camera ----------
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let W,H,DPR=Math.min(devicePixelRatio||1,2);
const view={s:1,x:0,y:0},tgt={s:1,x:0,y:0};
function resize(){W=cv.width=innerWidth*DPR;H=cv.height=innerHeight*DPR;cv.style.width=innerWidth+'px';cv.style.height=innerHeight+'px';fit();}
function fit(){const s=Math.min(W/W0,H/H0)*0.9;view.s=tgt.s=s;view.x=tgt.x=(W-W0*s)/2;view.y=tgt.y=(H-H0*s)/2;}
addEventListener('resize',resize);
function toScreen(x,y){return[x*view.s+view.x,y*view.s+view.y];}
function toWorld(px,py){return[(px-view.x)/view.s,(py-view.y)/view.s];}
// ---------- state ----------
let sel=-1,hover=-1,mutMode=false,activeCT='',layer='auto',affected=null,mutSeed=-1;
let edges=[];       // currently displayed pathway edges [{a,b,sign,kind}]
let anim=0;
// ---------- drawing ----------
function ell(x,y,rx,ry,fill,edge,lw){ctx.beginPath();ctx.ellipse(x,y,rx,ry,0,0,7);if(fill){ctx.fillStyle=fill;ctx.fill();}if(edge){ctx.strokeStyle=edge;ctx.lineWidth=lw||2;ctx.stroke();}}
function grad(x,y,r,c0,c1){const g=ctx.createRadialGradient(x-r*0.3,y-r*0.3,r*0.1,x,y,r);g.addColorStop(0,c0);g.addColorStop(1,c1);return g;}
function drawCell(){
 ctx.setTransform(1,0,0,1,0,0);
 const bg=ctx.createRadialGradient(W*0.5,H*0.42,60,W*0.5,H*0.5,Math.max(W,H)*0.75);
 bg.addColorStop(0,'#0a1220');bg.addColorStop(1,'#03060c');ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
 ctx.setTransform(view.s*DPR/DPR,0,0,view.s,view.x,view.y);   // world transform
 const z=view.s;
 // plasma membrane + cytoplasm
 const cyto=grad(CX,CY,760,'#0e1a2b','#081019');ell(CX,CY,720,452,cyto,null);
 ctx.save();ctx.globalAlpha=.9;ell(CX,CY,720,452,null,'#2f5f92',6/z);ell(CX,CY,712,446,null,'#173247',3/z);ctx.restore();
 // cytoskeleton fibers (faint)
 if(z>0.35){ctx.strokeStyle='rgba(120,150,180,'+Math.min(.10,(z-.35)*.3)+')';ctx.lineWidth=1/z;
  for(let k=0;k<60;k++){const[a,b]=hash(k*13+1);ctx.beginPath();ctx.moveTo(CX+(a-.5)*1300,CY+(b-.5)*820);const[a2,b2]=hash(k*13+99);ctx.lineTo(CX+(a2-.5)*1300,CY+(b2-.5)*820);ctx.stroke();}}
 // ER (tubular network around nucleus)
 ctx.strokeStyle='rgba(200,130,80,.5)';ctx.lineWidth=7/z;ctx.lineCap='round';
 for(let k=0;k<22;k++){const a=k/22*6.2832,r1=250,r2=340+40*Math.sin(k*1.7);ctx.beginPath();
  ctx.moveTo(560+Math.cos(a)*r1,520+Math.sin(a)*r1*0.82);ctx.quadraticCurveTo(560+Math.cos(a+.2)*r2*1.1,520+Math.sin(a+.2)*r2*0.7,560+Math.cos(a+.5)*r1,520+Math.sin(a+.5)*r1*0.82);ctx.stroke();}
 // Golgi stacks
 const gg=REG.Golgi;ctx.strokeStyle='#4fbfae';ctx.lineWidth=6/z;
 for(let k=0;k<5;k++){ctx.beginPath();ctx.ellipse(gg.x,gg.y+k*13-26,gg.rx-k*6,gg.ry-k*4,0.35,Math.PI*0.15,Math.PI*0.95);ctx.stroke();}
 // mitochondria with cristae
 for(const m of REG.mitochondrion.multi){ell(m[0],m[1],m[2],m[3],grad(m[0],m[1],m[2],'#7a2b52','#3a1226'),'#e06a9e',3/z);
  ctx.strokeStyle='rgba(240,150,190,.45)';ctx.lineWidth=2/z;for(let k=-3;k<=3;k++){ctx.beginPath();ctx.moveTo(m[0]+k*m[2]*0.24,m[1]-m[3]*0.7);ctx.quadraticCurveTo(m[0]+k*m[2]*0.24+14,m[1],m[0]+k*m[2]*0.24,m[1]+m[3]*0.7);ctx.stroke();}}
 // lysosomes / peroxisomes / endosomes
 for(const key of ['lysosome','peroxisome','endosome']){const r=REG[key];const c0=Array.isArray(r.fill)?r.fill[0]:r.fill,c1=Array.isArray(r.fill)?r.fill[1]:'#0b1018';for(const m of r.multi){ell(m[0],m[1],m[2],m[3],grad(m[0],m[1],m[2],c0,c1),r.edge,2/z);}}
 // nucleus
 const nu=REG.nucleus;ell(nu.x,nu.y,nu.rx,nu.ry,grad(nu.x,nu.y,nu.rx,'#2a2350','#120e2e'),null);
 ctx.save();ctx.globalAlpha=.5;ctx.strokeStyle='#a99fe0';ctx.lineWidth=1/z;   // chromatin texture
 for(let k=0;k<40;k++){const[a,b]=hash(k*7+3);ctx.beginPath();ctx.moveTo(nu.x+(a-.5)*nu.rx*1.5,nu.y+(b-.5)*nu.ry*1.5);const[a2,b2]=hash(k*7+55);ctx.lineTo(nu.x+(a2-.5)*nu.rx*1.4,nu.y+(b2-.5)*nu.ry*1.4);ctx.stroke();}ctx.restore();
 ell(nu.x,nu.y,nu.rx,nu.ry,null,'#8f7fd1',5/z);ell(nu.x,nu.y,nu.rx-7,nu.ry-7,null,'#5a4f8f',2/z); // envelope
 ell(nu.x+40,nu.y-30,70,58,grad(nu.x+40,nu.y-30,70,'#4a3f7f','#241d52'),null); // nucleolus
 // organelle labels (fade with zoom-in)
 if(z<2.2){ctx.globalAlpha=Math.max(.25,1.1-z*0.4);ctx.fillStyle='#9fb6d6';ctx.font=(15/z)+'px Inter,sans-serif';ctx.textAlign='center';
  for(const k in REG){const r=REG[k];if(!r.label)continue;const lx=r.x||(r.multi?r.multi[0][0]:CX),ly=(r.y||(r.multi?r.multi[0][1]:CY))-(r.ry||r.ring||60)-10/z;ctx.fillText(r.label,lx,ly);}
  ctx.globalAlpha=1;ctx.textAlign='left';}
 // ---- proteins (LOD) ----
 const showProt=z>0.34;const rbase=Math.max(1.1,1.6/Math.sqrt(z))* (z>1?1.2:1);
 if(showProt){const alpha=Math.min(1,(z-0.34)*3);
  const dim=(affected||sel>=0)?0.13:0.9;
  for(let i=0;i<N;i++){const X=POS[i*2],Y=POS[i*2+1];let r=rbase+(activeCT&&ABUND[i]!==undefined?ABUND[i]/15*1.6:Math.min(G[i].ppi/60,1.6));
   let on=true,col=pcol(G[i]);
   if(affected){on=affected.has(i);col=i===mutSeed?'#fff':(affected.get?'':'');}
   if(affected){const st=affected.get(i);on=st!==undefined;col=i===mutSeed?'#ffffff':st>0?'#4dd97a':st<0?'#e0555f':'#e6a95a';}
   else if(sel>=0){on=(i===sel)||selNeighbors.has(i);}
   ctx.globalAlpha=(on?alpha*0.95:alpha*dim);
   ctx.beginPath();ctx.arc(X,Y,i===sel?r+2:r,0,7);ctx.fillStyle=on?col:'#33465e';ctx.fill();
   if(G[i].tf&&z>0.8){ctx.globalAlpha=on?.6:.1;ctx.strokeStyle='#4a9eff';ctx.lineWidth=1/z;ctx.stroke();}
  }
  ctx.globalAlpha=1;
  // protein labels when very zoomed
  if(z>3.2){ctx.fillStyle='#bcd6f2';ctx.font=(9/z*3)+'px Inter';ctx.textAlign='center';
   for(let i=0;i<N;i++){if(sel>=0&&i!==sel&&!selNeighbors.has(i))continue;const X=POS[i*2],Y=POS[i*2+1];
    const[sx,sy]=[X,Y];if(X<toWorld(0,0)[0]-50||X>toWorld(W,0)[0]+50)continue;ctx.fillText(G[i].name,X,Y-rbase-3/z*3);}ctx.textAlign='left';}
 }
 // ---- pathway / interaction edges (selective, animated) ----
 if(edges.length){anim+=0.03;
  for(const e of edges){const[x1,y1]=[POS[e.a*2],POS[e.a*2+1]],[x2,y2]=[POS[e.b*2],POS[e.b*2+1]];
   const col=e.kind=='sig'?'#e0d040':e.kind=='ppi'?'#00c0b0':e.kind=='codep'?'#c792ea':e.sign>0?'#4dd97a':e.sign<0?'#e0555f':'#4a9eff';
   ctx.strokeStyle=col;ctx.globalAlpha=.55;ctx.lineWidth=1.6/z;ctx.setLineDash([6/z,6/z]);ctx.lineDashOffset=-anim*40/z;
   ctx.beginPath();ctx.moveTo(x1,y1);
   const mx=(x1+x2)/2+(y2-y1)*0.12,my=(y1+y2)/2-(x2-x1)*0.12;ctx.quadraticCurveTo(mx,my,x2,y2);ctx.stroke();
   // flowing pulse
   const t=(anim*0.3)%1,px=(1-t)*(1-t)*x1+2*(1-t)*t*mx+t*t*x2,py=(1-t)*(1-t)*y1+2*(1-t)*t*my+t*t*y2;
   ctx.setLineDash([]);ctx.globalAlpha=.9;ctx.beginPath();ctx.arc(px,py,2.4/z,0,7);ctx.fillStyle=col;ctx.fill();}
  ctx.setLineDash([]);ctx.globalAlpha=1;}
 // selection halo
 if(sel>=0){const X=POS[sel*2],Y=POS[sel*2+1];ctx.beginPath();ctx.arc(X,Y,10/z,0,7);ctx.strokeStyle='#7fd1ff';ctx.lineWidth=2/z;ctx.globalAlpha=.9;ctx.stroke();
  ctx.beginPath();ctx.arc(X,Y,(10+4*Math.sin(anim*4))/z,0,7);ctx.globalAlpha=.3;ctx.stroke();ctx.globalAlpha=1;}
 // hover glow
 if(hover>=0&&hover!==sel){const X=POS[hover*2],Y=POS[hover*2+1];ctx.beginPath();ctx.arc(X,Y,7/z,0,7);ctx.fillStyle='rgba(127,209,255,.25)';ctx.fill();ctx.strokeStyle='#7fd1ff';ctx.lineWidth=1.5/z;ctx.stroke();}
}
let selNeighbors=new Set();
// ---------- animation loop ----------
function loop(){
 view.s+=(tgt.s-view.s)*0.18;view.x+=(tgt.x-view.x)*0.18;view.y+=(tgt.y-view.y)*0.18;
 drawCell();requestAnimationFrame(loop);
}
// ---------- interaction ----------
let drag=null;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,vx:tgt.x,vy:tgt.y,moved:0};cv.classList.add('drag');});
addEventListener('mouseup',e=>{if(drag&&drag.moved<4){onClick(e);}drag=null;cv.classList.remove('drag');});
addEventListener('mousemove',e=>{
 if(drag){const dx=(e.clientX-drag.x)*DPR,dy=(e.clientY-drag.y)*DPR;drag.moved+=Math.abs(dx)+Math.abs(dy);tgt.x=drag.vx+dx;tgt.y=drag.vy+dy;view.x=tgt.x;view.y=tgt.y;return;}
 const i=pick(e.clientX*DPR,e.clientY*DPR);hover=i;const t=document.getElementById('tip');
 if(i>=0&&view.s>0.34){const g=G[i];t.style.display='block';t.style.left=Math.min(e.clientX+14,innerWidth-280)+'px';t.style.top=(e.clientY+12)+'px';
  t.innerHTML=`<b>${g.name}</b> <span class=lg>${g.comp}</span><div class=r>${g.proc} &middot; ${g.path||'&mdash;'}</div>
   <div style=margin-top:4px>${g.ess==1?'<span class="badge b-ess">essential'+(g.dep_frac!==undefined?' '+Math.round(g.dep_frac*100)+'%':'')+'</span>':''}${DRUGS[i]?'<span class="badge b-drug">druggable</span>':''}${g.ndis>=3?'<span class="badge b-dis">disease</span>':''}${g.tf?'<span class="badge b-tf">TF</span>':''}</div>
   <div class=r style=margin-top:3px>conf ${g.conf||'?'} &middot; ${g.npath} pathways &middot; PPI ${g.ppi} &middot; ${g.pubs||0} pubs</div>`;
 }else t.style.display='none';
});
cv.addEventListener('wheel',e=>{e.preventDefault();const f=e.deltaY<0?1.17:1/1.17;const mx=e.clientX*DPR,my=e.clientY*DPR;
 const ns=Math.max(0.12,Math.min(60,tgt.s*f));const k=ns/tgt.s;tgt.x=mx-(mx-tgt.x)*k;tgt.y=my-(my-tgt.y)*k;tgt.s=ns;},{passive:false});
function pick(px,py){if(view.s<0.30)return -1;const[wx,wy]=toWorld(px,py);let b=-1,bd=(14/view.s)*(14/view.s);
 for(let i=0;i<N;i++){const dx=POS[i*2]-wx,dy=POS[i*2+1]-wy,d=dx*dx+dy*dy;if(d<bd){bd=d;b=i;}}return b;}
function onClick(e){const i=pick(e.clientX*DPR,e.clientY*DPR);if(i<0){return;}
 if(mutMode){mutate(i);}else select(i);}
// ---------- selection ----------
function flyTo(i,zoom){const X=POS[i*2],Y=POS[i*2+1];tgt.s=zoom||Math.max(view.s,2.6);tgt.x=W/2-X*tgt.s;tgt.y=H/2-Y*tgt.s;}
function neighbors(i){const s=new Set([i]);(OUT[i]||[]).slice(0,6).forEach(j=>s.add(j));(IN[i]||[]).slice(0,4).forEach(j=>s.add(j));(PP[i]||[]).slice(0,6).forEach(j=>s.add(j));return s;}
function buildEdges(i){const e=[];const add=(list,kind,dir)=>{(list||[]).slice(0,8-e.length).forEach(j=>{if(j!==i)e.push({a:dir?i:j,b:dir?j:i,kind,sign:SGN[(dir?i:j)+','+(dir?j:i)]||0});});};
 if(layer=='auto'||layer=='reg'){add(OUT[i],'reg',1);}
 if(layer=='auto'||layer=='sig'){add(SIGO[i],'sig',1);}
 if(layer=='ppi'){add(PP[i],'ppi',1);}
 if(layer=='codep'){(CODEP[i]||[]).slice(0,8).forEach(([j])=>e.push({a:i,b:j,kind:'codep'}));}
 if(layer=='auto'&&e.length<6){add(PP[i],'ppi',1);}
 return e.slice(0,10);}
function select(i){sel=i;affected=null;mutSeed=-1;selNeighbors=neighbors(i);edges=buildEdges(i);flyTo(i);openPanel(i);}
function relayout(){edges=sel>=0?buildEdges(sel):[];}
// ---------- mutation propagation ----------
function mutate(i){mutSeed=i;sel=i;const st=new Map();st.set(i,2);let q=[[i,0]];const gate=null;
 while(q.length){const[x,d]=q.shift();if(d>=2)continue;for(const j of [...(OUT[x]||[]),...(PP[x]||[])]){if(!st.has(j)){const s=SGN[x+','+j];st.set(j,s>0?-1:s<0?1:-1);q.push([j,d+1]);}}}
 affected=st;selNeighbors=new Set(st.keys());edges=buildEdges(i);flyTo(i);openPanel(i,true);}
// ---------- side panel ----------
function sec(title,html,open){return `<div class="sec${open?' open':''}"><div class=h onclick="this.parentNode.classList.toggle('open')">${title}<span class=ar>&#9656;</span></div><div class=b>${html}</div></div>`;}
function glnk(j){return `<a class=g onclick="select(${j})">${G[j].name}</a>`;}
function lifeHistory(g){const c=g.comp;const S=[['DNA','gene locus '+(g.chrom||'')+(g.tss?':'+Number(g.tss).toLocaleString():''),'chromatin'],
 ['transcription','pre-mRNA','RNA Pol II'+(g.tf?', self is a TF':'')],['processing','mature mRNA','spliceosome, cap, poly-A'],['export','cytoplasm','nuclear pore']];
 const sec=['plasma membrane','membrane','ER','Golgi','lysosome','endosome','extracellular'].includes(c);
 if(sec){S.push(['translation (rough ER)','protein','ribosome+SRP, SEC61']);S.push(['ER','folded','BiP/calnexin']);if(c!='ER')S.push(['Golgi','glycosylated','COPII vesicle']);S.push([c,'at work','SNAREs / sorting']);}
 else if(c=='mitochondrion'){S.push(['translation (cytosol)','protein','ribosome']);S.push(['import','matrix','TOM/TIM']);}
 else{S.push(['translation (cytosol)','protein','ribosome']);S.push([c,'at work','chaperones'+(c=='nucleus'?', importin':'')]);}
 S.push(['function','see below','']);S.push(['degradation','recycled','proteasome/autophagy']);
 return S.map(s=>`<div class=step><b>${s[0]}</b>${s[1]?' &rarr; '+s[1]:''}${s[2]?` <span class=lg>&middot; ${s[2]}</span>`:''}</div>`).join('');}
function explainWhy(i,mut){const g=G[i];let out=[];
 if(mut){out.push(`<b>${g.name}</b> ${g.ess==1?'is essential ('+(g.dep_frac!==undefined?Math.round(g.dep_frac*100)+'% of cancer lines':'measured')+')':'is '+(g.loeuf<0.35?'strongly constrained':'tolerated')}, so loss `+(g.ess==1?'is predicted <b class=warn>cell-inviable</b>':'may be buffered')+`.`);
  const a=[...(affected||[]).keys()].filter(j=>j!==i);
  const proc={};a.forEach(j=>proc[G[j].proc]=(proc[G[j].proc]||0)+1);const tp=Object.entries(proc).sort((x,y)=>y[1]-x[1]).slice(0,3);
  out.push(`It propagates to <b>${a.length}</b> proteins across ${tp.map(p=>p[0]).join(', ')}.`);
  if(CODEP[i]&&CODEP[i].length)out.push(`It is co-essential with ${CODEP[i].slice(0,3).map(([j])=>glnk(j)).join(', ')} — a functional module that fails together.`);
  const dpart=(CODEP[i]||[]).filter(([j])=>DRUGS[j]&&DRUGS[j].length).slice(0,2);
  if(dpart.length)out.push(`Druggable co-dependency: pairing with ${dpart.map(([j])=>glnk(j)).join(', ')} is a <b>combination</b> candidate.`);
  if(BIOM[g.name]&&BIOM[g.name].mut&&BIOM[g.name].mut.length){const m=BIOM[g.name].mut[0];out.push(`Sensitivity biomarker: ${m[1]<0?'cells mutant in':'resistance if'} <b>${m[0]}</b>.`);}
 }else{
  out.push(`<b>${g.name}</b> works in the <b>${g.comp}</b> as ${g.proc}${g.path?' ('+g.path+')':''}.`);
  const acts=(OUT[i]||[]).filter(j=>SGN[i+','+j]>0).slice(0,3),reps=(OUT[i]||[]).filter(j=>SGN[i+','+j]<0).slice(0,3);
  if(g.tf)out.push(`As a TF it ${acts.length?'activates '+acts.map(glnk).join(', '):''}${reps.length?'; represses '+reps.map(glnk).join(', '):''}.`);
  if(g.ess==1)out.push(`It is <b class=warn>essential</b>${g.dep_frac!==undefined?' (dependent in '+Math.round(g.dep_frac*100)+'% of cancer lines)':''} — confidence ${g.conf||'?'}.`);
  if(g.flag)out.push(`<span class=warn>&#9888; ${g.flag}</span>`);
 }
 return out.join(' ');}
function openPanel(i,mut){const g=G[i];const acc=ACC[g.name];
 let h=`<h1>${g.name}</h1><div class=sub>${g.comp} &middot; ${g.proc}${g.master?' &middot; master: '+g.master:''}</div>
  <div id=explain>${explainWhy(i,mut)}</div>
  ${acc?`<div class=row><span class=chip onclick="showStruct('${acc}','${g.name}')">&#129516; view 3D structure (AlphaFold)</span></div><div id=struct></div>`:''}`;
 h+=sec('Identity & importance',`
   <div class=row><span class=k>genome</span> <span class=v>${g.chrom||'&mdash;'}${g.tss?' : '+Number(g.tss).toLocaleString():''}</span></div>
   <div class=row><span class=k>essential</span> <span class=v>${g.ess==1?'yes':g.ess==0?'no':'?'}</span> ${g.dep_frac!==undefined?'<span class=lg>(DepMap: '+Math.round(g.dep_frac*100)+'% of lines)</span>':g.ess_src=='model1'?'<span class=lg>(our model)</span>':''}</div>
   <div class=row><span class=k>confidence</span> <span class=v style="color:${g.conf=='high'?'#7ff0b5':'#ffb86a'}">${g.conf||'?'}</span> &middot; <span class=k>LOEUF</span> <span class=v>${g.loeuf<0?'&mdash;':g.loeuf}</span></div>
   ${g.flag?`<div class=row><span class=warn>&#9888; ${g.flag}</span></div>`:''}
   ${g.ti=='priority'?'<div class=row><span style="color:#ffd54f">&#9733; target-priority candidate</span></div>':g.ti=='whitespace'?'<div class=row><span style="color:#c792ea">&#9733; white-space target</span></div>':''}
   <div class=row><span class=k>literature</span> <span class=v>${g.pubs||0}</span> publications</div>`,true);
 h+=sec('Complete life history',lifeHistory(g),true);
 const acts=(OUT[i]||[]).filter(j=>SGN[i+','+j]>0),reps=(OUT[i]||[]).filter(j=>SGN[i+','+j]<0);
 h+=sec('Regulation & signaling',`
   ${acts.length?`<div class=row><span style=color:#4dd97a>activates &#9650;</span> ${acts.slice(0,8).map(glnk).join(', ')}</div>`:''}
   ${reps.length?`<div class=row><span style=color:#e0555f>represses &#9660;</span> ${reps.slice(0,8).map(glnk).join(', ')}</div>`:''}
   ${(IN[i]||[]).length?`<div class=row><span class=k>regulated by</span> ${(IN[i]||[]).slice(0,8).map(glnk).join(', ')}</div>`:''}
   ${(SIGO[i]||[]).length?`<div class=row><span class=k>signals to</span> ${(SIGO[i]||[]).slice(0,6).map(glnk).join(', ')}</div>`:''}`);
 h+=sec('Interactions & complexes',`
   ${(PP[i]||[]).length?`<div class=row><span class=k>binds</span> ${(PP[i]||[]).slice(0,10).map(glnk).join(', ')}</div>`:''}
   ${G2C[i]?`<div class=row><span class=k>complexes</span> ${G2C[i].map(n=>`<span class=v>${n}</span>`).join('; ')}</div>`:''}
   ${CODEP[i]?`<div class=row><span class=k>co-essential</span> ${CODEP[i].slice(0,6).map(([j,r])=>glnk(j)+' <span class=lg>'+r+'</span>').join(', ')}</div>`:''}`);
 const bm=BIOM[g.name];
 h+=sec('Drugs, disease & biomarkers',`
   ${DRUGS[i]?`<div class=row><span class=k>drugs</span> ${DRUGS[i].slice(0,6).map(d=>d.d+(d.a?' <span class=lg>(approved)</span>':'')).join(', ')}</div>`:'<div class=lg>no drug in DGIdb</div>'}
   ${OTD[g.name]?`<div class=row><span class=k>disease</span> ${OTD[g.name].top.map(t=>t[0]).join('; ')}</div>`:''}
   ${bm&&bm.mut&&bm.mut.length?`<div class=row><span class=k>biomarker</span> ${bm.mut.filter(x=>x[1]<0).slice(0,3).map(x=>'sensitive if '+x[0]+'-mutant').join('; ')||bm.mut.slice(0,2).map(x=>x[0]+' ('+x[1]+')').join('; ')}</div>`:''}`);
 h+=sec('PTMs & structure',`
   ${PTM[i]?`<div class=row><span class=k>PTMs</span> <span class=v>${PTM[i].n}</span> sites ${PTM[i].c?'<span class=lg>('+PTM[i].c.join(', ')+')</span>':''}</div>`:'<div class=lg>&mdash;</div>'}
   ${g.cpg!==undefined?`<div class=row><span class=k>promoter</span> <span class=v>${g.cpg?'CpG island':'non-CpG'}</span> &middot; ${g.enh} enhancers</div>`:''}`);
 h+=sec('Provenance',`<div class=lg>location: UniProt &middot; networks: CollecTRI/STRING/SIGNOR &middot; essentiality: ${g.dep_frac!==undefined?'DepMap (measured)':g.ess_src=='model1'?'our model':'&mdash;'} &middot; metabolism: Human-GEM &middot; drugs: DGIdb &middot; disease: Open Targets &middot; structure: AlphaFold</div>`);
 h+=`<div class=row style=margin-top:8px><span class=chip onclick="mutMode=true;document.getElementById('mMut').classList.add('on');mutate(${i})">&#9888; mutate this</span> <span class=chip onclick="clearSel()">clear</span></div>`;
 document.getElementById('pbody').innerHTML=h;document.getElementById('panel').classList.add('open');updLegend();}
function clearSel(){sel=-1;affected=null;mutSeed=-1;edges=[];selNeighbors=new Set();document.getElementById('panel').classList.remove('open');}
document.getElementById('close').onclick=clearSel;
// AlphaFold on-demand
function showStruct(acc,name){const box=document.getElementById('struct');box.style.display='block';
 if(typeof $3Dmol==='undefined'){box.innerHTML='<div style="padding:14px;color:#7f9ec0;font-size:12px">3D viewer needs internet (loads AlphaFold model AF-'+acc+'). Open <a class=g href="https://alphafold.ebi.ac.uk/entry/'+acc+'" target=_blank>AlphaFold entry</a>.</div>';return;}
 box.innerHTML='<div style="padding:14px;color:#7f9ec0;font-size:12px">fetching AlphaFold model&hellip;</div>';
 fetch('https://alphafold.ebi.ac.uk/api/prediction/'+acc).then(r=>r.json()).then(d=>{const url=d[0].pdbUrl;
  box.innerHTML='';const v=$3Dmol.createViewer(box,{backgroundColor:'#02040a'});
  fetch(url).then(r=>r.text()).then(pdb=>{v.addModel(pdb,'pdb');v.setStyle({},{cartoon:{colorscheme:'spectrum'}});v.zoomTo();v.render();});
 }).catch(e=>{box.innerHTML='<div style="padding:14px;color:#ff8a9c;font-size:12px">structure unavailable ('+acc+')</div>';});}
window.select=select;window.mutate=mutate;window.clearSel=clearSel;window.showStruct=showStruct;
// ---------- controls ----------
document.getElementById('reset').onclick=()=>{clearSel();mutMode=false;document.getElementById('mMut').classList.remove('on');activeCT='';document.getElementById('ct').value='';fit();};
document.getElementById('mMut').onclick=function(){mutMode=!mutMode;this.classList.toggle('on',mutMode);};
const sB=document.getElementById('search');sB.addEventListener('change',()=>{const n=sB.value.trim();if(byName[n]!==undefined)select(byName[n]);});
sB.addEventListener('keydown',e=>{if(e.key=='Enter'){const n=sB.value.trim();const k=Object.keys(byName).find(x=>x.toUpperCase()==n.toUpperCase());if(k!==undefined)select(byName[k]);}});
// cell type
const ctS=document.getElementById('ct');ctS.innerHTML='<option value="">— cell type —</option>'+Object.keys(D.celltypes).map(c=>`<option>${c}</option>`).join('');
ctS.onchange=()=>{activeCT=ctS.value;const m=(D.celltypes[activeCT]||[]).map(n=>byName[n]).filter(x=>x!==undefined);if(m.length){select(m[0]);}};
function updLegend(){const L=document.getElementById('legend');L.innerHTML='<b>Processes</b><br>'+Object.keys(PCOL).slice(0,8).map(p=>`<span class=dot style=background:${PCOL[p]}></span>${p}`).join('<br>');}
document.getElementById('hud').innerHTML='scroll = zoom &middot; drag = pan &middot; click a protein &middot; Mutate mode';
updLegend();resize();loop();
</script></body></html>"""
HTML=HTML.replace("__DATA__",json.dumps(D,separators=(',',':')))
(OUT/"cell_explorer.html").write_text(HTML)
print("wrote cell_explorer.html (%d KB)"%(len(HTML)//1024))
