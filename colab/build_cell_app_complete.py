"""The complete interactive cell app: spatial cell (proteins in compartments) with a PROCESS
view, a PERTURBATION engine (remove a protein / mutate -> propagate the cascade through the
regulatory+PPI networks -> report the damaged cell), and an HIV INFECTION mode (hijacked
machinery + host-dependency weak points). Reads cell_complete.json, self-contained HTML."""
import json
from pathlib import Path
OUT=Path("outputs/orphan")
D=json.load(open(OUT/"cell_complete.json"))
HTML=r"""<!doctype html><html><head><meta charset=utf-8><title>The cell</title><style>
 html,body{margin:0;height:100%;background:#080e16;color:#dfe8f0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
 #top{height:42px;display:flex;align-items:center;gap:6px;padding:0 10px;background:#0d1a28}
 #top{border-bottom:1px solid #1d3350}#wrap{display:flex;height:calc(100vh - 43px)}
 #cv{flex:1;display:block;cursor:crosshair}
 #side{width:350px;background:#0d1a28;padding:12px 14px;overflow:auto;font-size:13px;box-shadow:-2px 0 12px #0007}
 h2{color:#7fd1ff;font-size:16px;margin:.3em 0}h3{color:#9be7a0;font-size:13px;margin:.6em 0 .2em}
 .k{color:#8fa8c0}.v{color:#e8f0f8;font-weight:600}.row{margin:3px 0}.lg{font-size:11px;color:#9fb3c8}
 .btn{padding:5px 10px;border-radius:6px;background:#16304c;color:#cfe8ff;cursor:pointer;font-size:12px;border:1px solid #27507c;user-select:none}
.btn:hover{background:#1f436a}.btn.on{background:#2e86de;color:#fff}
 .btn.hiv{background:#5a1d2e;border-color:#a03}.btn.hiv.on{background:#d1305a}
 .chip{display:inline-block;margin:2px;padding:2px 7px;border-radius:5px;background:#16304c;color:#cfe8ff;cursor:pointer;font-size:11px}
 .chip:hover{background:#276}.chip.on{background:#2e86de;color:#fff}
 input{background:#080e16;border:1px solid #345;color:#dfe8f0;border-radius:5px;padding:4px 7px;width:130px}
 select{background:#080e16;border:1px solid #345;color:#dfe8f0;border-radius:5px;padding:4px 6px;font-size:12px;max-width:170px}
 .step{margin:2px 0;padding:3px 7px;border-left:2px solid #2e86de;background:#0e2136}
 .mach{color:#9fb3c8;font-size:11px}.arrow{color:#5c7fa0;margin:0 3px}
 a{color:#7fd1ff;cursor:pointer}a:hover{text-decoration:underline}#tip{position:absolute;pointer-events:none;background:#000d;border:1px solid #345;border-radius:5px;padding:5px 8px;font-size:12px;display:none;z-index:9}
 .dot{display:inline-block;width:9px;height:9px;border-radius:9px;vertical-align:middle;margin-right:4px}
 .warn{color:#ff6b6b;font-weight:700}.ok{color:#9be7a0}
</style></head><body>
<div id=top>
 <span class=btn id=mExplore>Explore</span><span class=btn id=mProc>Processes</span>
 <span class=btn id=mMetab>Metabolism</span><span class=btn id=mPerturb>Remove/Mutate</span>
 <span class=btn id=mOE>Overexpress</span><span class=btn id=mDark>Dark genes</span><span class=btn id=mTargets>Targets</span><span class=btn hiv id=mHIV>Infect: HIV</span>
 <select id=ct title="cell type — highlights its master-TF network"></select>
 <select id=pw title="pathway — highlights its member proteins"></select>
 <input id=q placeholder="search protein"><span class=btn id=reset>Reset</span>
 <span class=lg id=hint>&nbsp; click a protein to inspect; switch modes above</span>
</div>
<div id=wrap><canvas id=cv></canvas><div id=side><div id=info></div></div></div><div id=tip></div>
<script>
const D=__DATA__;const G=D.genes,N=G.length;
const idxByName={};G.forEach((g,i)=>idxByName[g.name]=i);
// networks
const OUT={},IN={},PP={},SGN={};for(const e of D.reg){const a=e[0],b=e[1],s=e[2]||0;(OUT[a]=OUT[a]||[]).push(b);(IN[b]=IN[b]||[]).push(a);SGN[a+','+b]=s;}
for(const[a,b]of D.ppi){(PP[a]=PP[a]||[]).push(b);(PP[b]=PP[b]||[]).push(a);}
// reaction-by-enzyme (curated pathway-flow) + genome-scale reactions per gene (Human-GEM) + HIV host index
const RXN={};(D.reactions||[]).forEach(r=>RXN[r.i]=r);
const GRXN=D.generxn||{};   // gene index -> [ "substrate → product", ... ]
const hivHost={};for(const hp in D.hiv)for(const[i,t]of D.hiv[hp]){(hivHost[i]=hivHost[i]||[]).push(hp);}
// external layers: signaling, complexes, drugs, PTMs, ligand-receptor, cell cycle
const SIGO={},SIGI={};for(const e of (D.sig||[])){(SIGO[e[0]]=SIGO[e[0]]||[]).push([e[1],e[2]]);(SIGI[e[1]]=SIGI[e[1]]||[]).push([e[0],e[2]]);}
const CPLX=D.complexes||{},G2C=D.gene2cplx||{},DRUGS=D.drugs||{},PTM=D.ptm||{},CCYC=D.cellcycle||{};
const LRO={},LRI={};for(const[a,b]of (D.lr||[])){(LRO[a]=LRO[a]||[]).push(b);(LRI[b]=LRI[b]||[]).push(a);}
// Bucket 1: DepMap co-essentiality, synthetic-lethal pairs, 3D chromatin loops, biomarkers
const CODEP=D.codep||{},LOOPS3D=D.loops3d||{},BIOM=D.biomarkers||{};
const SLmap={};for(const[a,b,r] of (D.sl||[])){(SLmap[a]=SLmap[a]||{})[b]=r;(SLmap[b]=SLmap[b]||{})[a]=r;}
// Model 2: abundance + per-cell-type expression mask (drives abundance, cell-type wiring, differentiation)
const ABUND=D.abund||{};const CTN=D.ctnames||[];const ctBit={};CTN.forEach((n,t)=>ctBit[n]=t);
const EMASK=new Map();for(const k in (D.emask||{}))EMASK.set(+k,BigInt(D.emask[k]));
let activeCT=null;   // {name,t,set} when a cell type with expression data is selected
function exprIn(i,t){const m=EMASK.get(i);return m!==undefined&&((m>>BigInt(t))&1n)===1n;}
function exprSet(t){const s=new Set();for(const[i,m]of EMASK)if(((m>>BigInt(t))&1n)===1n)s.add(i);return s;}
// compartment regions
const CX=560,CY=400,RX=520,RY=360;
const REG={nucleus:{x:320,y:400,rx:165,ry:145},cytoplasm:{x:630,y:420,rx:330,ry:250},
 'plasma membrane':{ring:1},membrane:{ring:1},extracellular:{out:1},
 mitochondrion:{multi:[[730,230,68,30],[830,530,60,28],[470,590,58,26]]},
 ER:{x:520,y:420,rx:150,ry:118,shell:170},Golgi:{x:690,y:290,rx:66,ry:34},
 cytoskeleton:{x:630,y:420,rx:330,ry:245},lysosome:{multi:[[860,350,26,26]]},
 peroxisome:{multi:[[500,290,20,20]]},endosome:{multi:[[770,350,28,24]]},unknown:{x:1080,y:110,rx:60,ry:60}};
function hash(i){let h=i*2654435761%2147483647;return[(h%997)/997,((h/997|0)%997)/997];}
const POS=new Float32Array(N*2);
for(let i=0;i<N;i++){let c=G[i].comp,[u,v]=hash(i+1),r=REG[c]||REG.unknown,x,y;
 if(r.ring){let a=u*6.2832;x=CX+Math.cos(a)*RX;y=CY+Math.sin(a)*RY;}
 else if(r.out){let a=u*6.2832,rr=1.05+v*0.13;x=CX+Math.cos(a)*RX*rr;y=CY+Math.sin(a)*RY*rr;}
 else if(r.multi){let m=r.multi[i%r.multi.length],a=u*6.2832;x=m[0]+Math.cos(a)*m[2]*v;y=m[1]+Math.sin(a)*m[3]*v;}
 else if(r.shell){let a=u*6.2832,rr=r.rx*(0.85+v*0.35);x=r.x+Math.cos(a)*rr;y=r.y+Math.sin(a)*rr*0.8;}
 else{let a=u*6.2832,rr=Math.sqrt(v);x=r.x+Math.cos(a)*r.rx*rr;y=r.y+Math.sin(a)*r.ry*rr;}
 POS[i*2]=x;POS[i*2+1]=y;}
const PROC=D.procs, pcol={};const palette=['#e53935','#43a047','#1e88e5','#fb8c00','#8e24aa','#00acc1','#fdd835','#6d4c41','#ec407a','#7cb342','#5c6bc0','#26a69a','#8d6e63'];
PROC.forEach((p,i)=>pcol[p]=palette[i%palette.length]);
let mode='Explore',sel=-1,affected=null,hivOn=false,mark=null,metabOn=false;
let W,Hh;const cv=document.getElementById('cv'),ctx=cv.getContext('2d');let view={s:1,ox:0,oy:0};
function resize(){W=cv.width=cv.clientWidth*devicePixelRatio;Hh=cv.height=cv.clientHeight*devicePixelRatio;fit();draw();}
function fit(){let s=Math.min(W/1180,Hh/800)*0.97;view.s=s;view.ox=(W-1180*s)/2;view.oy=(Hh-800*s)/2;}
window.addEventListener('resize',resize);
function baseColor(i){let g=G[i];if(mode=='Processes')return pcol[g.proc];
 if(g.ess==1)return '#e53935';let l=g.loeuf;if(l<0)return '#33506e';if(l<0.35)return '#ff9800';if(l<0.7)return '#c9a24a';return '#3f5f7c';}
function draw(){ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#080e16';ctx.fillRect(0,0,W,Hh);
 ctx.save();ctx.translate(view.ox,view.oy);ctx.scale(view.s,view.s);ctx.font='15px sans-serif';
 el(CX,CY,RX,RY,'#0c1c30','#4a80c0',5,'');ctx.fillStyle='#7fa8d0';ctx.fillText('extracellular',CX+RX-120,CY-RY+16);ctx.fillText('cytoplasm',CX+140,CY-RY+64);
 el(REG.ER.x,REG.ER.y,REG.ER.rx,REG.ER.ry,'',' #e08a5a'.trim(),1.3,'');el(320,400,165,145,'#12233a','#8f7fd1',4,'nucleus');
 for(const m of REG.mitochondrion.multi)el(m[0],m[1],m[2],m[3],'#21122f','#d15a9e',2.2,'');
 el(REG.Golgi.x,REG.Golgi.y,REG.Golgi.rx,REG.Golgi.ry,'#0c2725','#5ad1c0',2.2,'Golgi');
 ctx.fillStyle='#c08a6a';ctx.fillText('ER',REG.ER.x-78,REG.ER.y-118);ctx.fillStyle='#d18ac0';ctx.fillText('mito',700,196);
 // metabolic reaction flow arrows (consecutive enzymes in a pathway)
 if(metabOn){ctx.globalAlpha=.5;ctx.strokeStyle='#fdd835';ctx.lineWidth=1.1;
   for(let k=0;k<D.reactions.length-1;k++){let a=D.reactions[k],b=D.reactions[k+1];
     if(a.pathway!==b.pathway)continue;let x1=POS[a.i*2],y1=POS[a.i*2+1],x2=POS[b.i*2],y2=POS[b.i*2+1];
     ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}}
 for(let i=0;i<N;i++){let X=POS[i*2],Y=POS[i*2+1],r=1.3+Math.min(G[i].ppi/45,2.4);
   if(activeCT&&ABUND[i]!==undefined)r+=(ABUND[i]/15)*2.2;   // size by abundance in the active cell type
   let inMark=mark&&mark.set.has(i);
   let dim=(affected&&!affected.set.has(i)&&i!=sel)||(mark&&mark.dim&&!inMark&&i!=sel);
   let hl=(affected&&affected.set.has(i));
   if(inMark)r+=1;
   ctx.globalAlpha=dim?0.05:0.9;ctx.beginPath();ctx.arc(X,Y,hl?r+1:r,0,7);
   ctx.fillStyle=hl?affected.col(i):(inMark?(mark.colFn?mark.colFn(i):mark.color):baseColor(i));ctx.fill();
   if(hivOn&&hivHost[i]){ctx.globalAlpha=.9;ctx.strokeStyle='#ff3b6b';ctx.lineWidth=1.4;ctx.stroke();}
   else if(G[i].ndis>=5){ctx.globalAlpha=dim?.1:.7;ctx.strokeStyle='#ff9800';ctx.lineWidth=.7;ctx.stroke();}
   if(G[i].tf){ctx.globalAlpha=dim?.12:.8;ctx.strokeStyle='#2e86de';ctx.lineWidth=.9;ctx.stroke();}}
 // HIV virus marker
 if(hivOn){ctx.globalAlpha=1;ctx.fillStyle='#ff3b6b';ctx.beginPath();ctx.arc(CX+RX-30,CY,16,0,7);ctx.fill();ctx.fillStyle='#fff';ctx.font='bold 12px sans-serif';ctx.fillText('HIV',CX+RX-44,CY+4);}
 if(sel>=0){ctx.globalAlpha=1;star(POS[sel*2],POS[sel*2+1],8,'#fff');}
 ctx.restore();}
function el(x,y,rx,ry,f,s,lw,l){ctx.beginPath();ctx.ellipse(x,y,rx,ry,0,0,7);if(f){ctx.fillStyle=f;ctx.fill();}ctx.strokeStyle=s.trim();ctx.lineWidth=lw;ctx.stroke();if(l){ctx.fillStyle=s.trim();ctx.fillText(l,x-ctx.measureText(l).width/2,y-ry-6);}}
function star(x,y,r,c){ctx.beginPath();for(let k=0;k<10;k++){let a=Math.PI/5*k-1.57,rr=k%2?r*.45:r;ctx.lineTo(x+Math.cos(a)*rr,y+Math.sin(a)*rr);}ctx.closePath();ctx.fillStyle=c;ctx.fill();}
// interaction
cv.addEventListener('wheel',e=>{e.preventDefault();let f=e.deltaY<0?1.15:1/1.15,mx=e.offsetX*devicePixelRatio,my=e.offsetY*devicePixelRatio;view.ox=mx-(mx-view.ox)*f;view.oy=my-(my-view.oy)*f;view.s*=f;draw();},{passive:false});
let drag=null;cv.addEventListener('mousedown',e=>drag={x:e.offsetX,y:e.offsetY,ox:view.ox,oy:view.oy});window.addEventListener('mouseup',()=>drag=null);
cv.addEventListener('mousemove',e=>{if(drag){view.ox=drag.ox+(e.offsetX-drag.x)*devicePixelRatio;view.oy=drag.oy+(e.offsetY-drag.y)*devicePixelRatio;draw();return;}
 let i=pick(e.offsetX*devicePixelRatio,e.offsetY*devicePixelRatio),t=document.getElementById('tip');
 if(i>=0){t.style.display='block';t.style.left=(e.offsetX+12)+'px';t.style.top=(e.offsetY+8)+'px';t.innerHTML=`<b>${G[i].name}</b> &middot; ${G[i].comp} &middot; ${G[i].proc}`;}else t.style.display='none';});
function pick(px,py){let wx=(px-view.ox)/view.s,wy=(py-view.oy)/view.s,b=-1,bd=90;for(let i=0;i<N;i++){let dx=POS[i*2]-wx,dy=POS[i*2+1]-wy,d=dx*dx+dy*dy;if(d<bd){bd=d;b=i;}}return b;}
cv.addEventListener('click',e=>{if(drag&&Math.abs(e.offsetX-drag.x)>3)return;let i=pick(e.offsetX*devicePixelRatio,e.offsetY*devicePixelRatio);if(i>=0){sel=i;if(mode=='Remove/Mutate'){if(koSeed!=null&&koSeed!==i){doubleKO(koSeed,i);koSeed=null;}else perturb(i,false);}else if(mode=='Overexpress')overexpress(i);else info(i);draw();}});
// perturbation: BFS 2 hops over reg(out)+ppi, enriched with Geneformer in-silico downstream (Model 3)
const GF=D.gf_perturb||{};
function perturb(i,mut){let set=new Set([i]),dist={};dist[i]=0;let q=[i];
 const gate=activeCT?activeCT.set:null;   // cell-type-specific wiring: only propagate through expressed genes
 while(q.length){let x=q.shift();if(dist[x]>=2)continue;for(const j of [...(OUT[x]||[]),...(PP[x]||[])]){if(gate&&!gate.has(j))continue;if(!set.has(j)){set.add(j);dist[j]=dist[x]+1;q.push(j);}}}
 // Model 3: add Geneformer-predicted downstream genes that the measured graph missed
 const gfHit=new Set();(GF[i]||[]).forEach(j=>{if(gate&&!gate.has(j))return;if(!set.has(j)){set.add(j);dist[j]=2;gfHit.add(j);}else gfHit.add(j);});
 const proc={},path={};set.forEach(j=>{if(j==i)return;proc[G[j].proc]=(proc[G[j].proc]||0)+1;if(G[j].path)path[G[j].path]=(path[G[j].path]||0)+1;});
 affected={set,col:j=>j==i?'#fff':(gfHit.has(j)?'#c792ea':(dist[j]==1?'#ff6b6b':'#ffb36b'))};
 const lethal=G[i].ess==1||(mut&&G[i].loeuf>=0&&G[i].loeuf<0.35);
 const essNote=G[i].ess==1?(G[i].ess_src==='model1'?'essential (predicted by our model'+(G[i].ess_prob?', p='+G[i].ess_prob:'')+')':'essential gene'):'';
 const topP=Object.entries(proc).sort((a,b)=>b[1]-a[1]).slice(0,6);
 const topPath=Object.entries(path).sort((a,b)=>b[1]-a[1]).slice(0,6);
 document.getElementById('info').innerHTML=`<h2>${mut?'Mutate':'Remove'}: ${G[i].name}</h2>
  <div class=row>${lethal?'<span class=warn>&#9888; predicted cell-inviable</span> — '+(G[i].ess==1?essNote:'constrained (LOEUF '+G[i].loeuf+'), likely loss-of-function is lethal'):'<span class=ok>cell likely survives</span> — dispensable / buffered'}</div>
  <div class=row><span class=k>compartment</span> <span class=v>${G[i].comp}</span> &middot; <span class=k>process</span> <span class=v>${G[i].proc}</span></div>
  ${activeCT?`<div class=row class=lg>cell-type context: <b>${activeCT.name}</b> — cascade propagates only through genes active here${!activeCT.set.has(i)?' <span class=warn>(this gene is silent in '+activeCT.name+')</span>':''}</div>`:''}
  <div class=row><span class=k>direct + 2-hop affected</span> <span class=v>${set.size-1}</span> proteins${gfHit.size?` &middot; <span style="color:#c792ea">+${gfHit.size} via Geneformer</span>`:''}</div>
  <h3>processes disrupted</h3>${topP.map(([p,n])=>`<div class=row><span style="color:${pcol[p]}">&#9679;</span> ${p}: <span class=v>${n}</span></div>`).join('')}
  <h3>pathways hit</h3>${topPath.map(([p,n])=>`<div class=row class=lg>${p} (${n})</div>`).join('')||'<div class=lg>-</div>'}
  ${mut?'<h3>mutation impact</h3>'+mutImpact(G[i]):''}
  <div class=row style=margin-top:8px><span class=btn onclick="perturb(${i},true)">as mutation (LoF)</span> <span class=btn onclick="koSeed=${i};document.getElementById('hint').textContent=' now click a 2nd protein to co-knock-out with ${G[i].name}'">+ 2nd knockout</span> <span class=btn onclick="clearP()">clear</span></div>`;}
window.perturb=perturb;window.clearP=()=>{affected=null;koSeed=null;draw();};
// helper: 2-hop cascade set for one gene (cell-type gated), for double-KO
function cascadeSet(i){const set=new Set([i]),dist={};dist[i]=0;let q=[i];const gate=activeCT?activeCT.set:null;
 while(q.length){let x=q.shift();if(dist[x]>=2)continue;for(const j of[...(OUT[x]||[]),...(PP[x]||[])]){if(gate&&!gate.has(j))continue;if(!set.has(j)){set.add(j);dist[j]=dist[x]+1;q.push(j);}}}
 (GF[i]||[]).forEach(j=>{if(!(gate&&!gate.has(j)))set.add(j);});return set;}
let koSeed=null;
// DOUBLE KNOCKOUT / synthetic lethality: combine two KOs, flag emergent lethality
function doubleKO(i,j){const A=cascadeSet(i),B=cascadeSet(j),U=new Set([...A,...B]);
 const both=[...A].filter(x=>B.has(x)&&x!==i&&x!==j);   // hit by BOTH single KOs
 const lethA=G[i].ess==1,lethB=G[j].ess==1;
 // synthetic-lethal signal: neither individually essential, but together they converge on
 // essential genes / a much larger joint cascade than either alone
 const jointEss=[...U].filter(x=>G[x].ess==1&&x!==i&&x!==j).length;
 const sl=!lethA&&!lethB&&(both.length>=3||jointEss>Math.max([...A].filter(x=>G[x].ess==1).length,[...B].filter(x=>G[x].ess==1).length));
 const depR=(SLmap[i]&&SLmap[i][j])||(CODEP[i]&&(CODEP[i].find(([k])=>k===j)||[])[1]);
 affected={set:U,col:x=>(x===i||x===j)?'#fff':(A.has(x)&&B.has(x))?'#c792ea':(A.has(x)?'#ff6b6b':'#ffb36b')};
 const lnk=arr=>arr.slice(0,12).map(x=>`<a onclick=go('${G[x].name}')>${G[x].name}</a>`).join(', ');
 document.getElementById('info').innerHTML=`<h2>Double knockout: ${G[i].name} + ${G[j].name}</h2>
  ${activeCT?`<div class=row class=lg>cell-type context: <b>${activeCT.name}</b></div>`:''}
  <div class=row>${lethA||lethB?'<span class=warn>&#9888; already lethal singly</span> ('+(lethA?G[i].name:'')+(lethA&&lethB?' & ':'')+(lethB?G[j].name:'')+')':(sl?'<span class=warn>&#9888; candidate SYNTHETIC LETHAL</span> — neither is essential alone, but together they converge on essential machinery':'<span class=ok>likely tolerated</span> — buffered even as a double KO')}</div>
  ${depR?`<div class=row><span class=k>DepMap co-dependency</span> <span class=v style="color:#c792ea">r=${depR}</span> <span class=lg>— measured co-essential across cancer lines${SLmap[i]&&SLmap[i][j]?'; flagged synthetic-lethal candidate':''}</span></div>`:''}
  <div class=row><span class=k>combined affected</span> <span class=v>${U.size-2}</span> &middot; <span style="color:#c792ea">hit by both</span> <span class=v>${both.length}</span> &middot; <span class=k>essential in joint cascade</span> <span class=v>${jointEss}</span></div>
  <h3 style="color:#c792ea">shared targets (both KOs converge here)</h3><div class=lg>${lnk(both)||'-'}</div>
  <div class=row style=margin-top:8px><span class=btn onclick="clearP()">clear</span></div>`;
 draw();}
window.doubleKO=doubleKO;
// OVEREXPRESSION / ACTIVATION: push a gene UP, propagate signed regulation (up/down) 2 hops
function overexpress(i){const gate=activeCT?activeCT.set:null;
 const st={};st[i]=1;let q=[[i,0]];        // +1 up, -1 down
 while(q.length){let [x,d]=q.shift();if(d>=2)continue;
   for(const j of (OUT[x]||[])){if(gate&&!gate.has(j))continue;const s=SGN[x+','+j];if(!s)continue;
     const nv=st[x]*s;if(st[j]===undefined){st[j]=nv;q.push([j,d+1]);}}}
 const up=[],down=[];for(const k in st){if(+k===i)continue;(st[k]>0?up:down).push(+k);}
 const set=new Set([i,...up,...down]);
 affected={set,col:j=>j===i?'#fff':(st[j]>0?'#43a047':'#e53935')};
 const proc={};set.forEach(j=>{if(j!==i)proc[G[j].proc]=(proc[G[j].proc]||0)+1;});
 const topP=Object.entries(proc).sort((a,b)=>b[1]-a[1]).slice(0,6);
 const lnk=arr=>arr.slice(0,12).map(j=>`<a onclick=go('${G[j].name}')>${G[j].name}</a>`).join(', ');
 document.getElementById('info').innerHTML=`<h2>Overexpress / activate: ${G[i].name}</h2>
  <div class=lg>Driving ${G[i].name} UP and following signed regulation${G[i].tf?'':' — note: this gene has few/no outgoing regulatory edges, so effect may be local'}.</div>
  ${activeCT?`<div class=row class=lg>cell-type context: <b>${activeCT.name}</b></div>`:''}
  <div class=row><span style="color:#43a047">upregulated</span> <span class=v>${up.length}</span> &middot; <span style="color:#e53935">downregulated</span> <span class=v>${down.length}</span></div>
  <h3>processes shifted</h3>${topP.map(([p,n])=>`<div class=row><span style="color:${pcol[p]}">&#9679;</span> ${p}: <span class=v>${n}</span></div>`).join('')||'<div class=lg>-</div>'}
  <h3 style="color:#43a047">turned up</h3><div class=lg>${lnk(up)||'-'}</div>
  <h3 style="color:#e53935">turned down</h3><div class=lg>${lnk(down)||'-'}</div>
  <div class=row style=margin-top:8px><span class=btn onclick="clearP()">clear</span></div>`;
 draw();}
window.overexpress=overexpress;
function info(i){let g=G[i];
 const lnk=j=>`<a onclick=go('${G[j].name}')>${G[j].name}</a>`;
 const outs=OUT[i]||[];
 let act=outs.filter(j=>SGN[i+','+j]>0),rep=outs.filter(j=>SGN[i+','+j]<0),unk=outs.filter(j=>!SGN[i+','+j]);
 let rgl=(IN[i]||[]).slice(0,10).map(lnk).join(', ');
 let pp=(PP[i]||[]).slice(0,10).map(j=>`<a onclick=go('${G[j].name}')>${G[j].name}</a>`).join(', ');
 document.getElementById('info').innerHTML=`<h2>${g.name}</h2>
  <div class=row><span class=k>genome</span> <span class=v>${g.chrom?(g.chrom+(g.tss?' : '+Number(g.tss).toLocaleString():'')):'-'}</span></div>
  <div class=row><span class=k>compartment</span> <span class=v>${g.comp}</span></div>
  <div class=row><span class=k>process</span> <span class=v style="color:${pcol[g.proc]}">${g.proc}</span></div>
  <div class=row><span class=k>pathway</span> <span class=v>${g.path||'-'}</span></div>
  ${g.dark&&D.darkfn&&D.darkfn[i]?(()=>{const d=D.darkfn[i];return `<div class=row style="background:#2a1e3a;border-left:3px solid #c792ea;padding:5px 8px;margin:4px 0"><span class=k style="color:#c792ea">predicted function</span> <span class=v>${d.pred}</span> <span class=lg>(${d.conf} confidence &middot; <b style="color:#e0a">predicted, not known</b>)</span><div class=lg style=margin-top:3px>guilt-by-association from ${d.n} <b>measured</b> functional neighbors (${d.src}): ${(d.ev||[]).map(n=>`<a onclick=go('${n}')>${n}</a>`).join(', ')}</div></div>`;})():''}
  ${D.model4&&D.model4[i]?(()=>{const d=D.model4[i],lift=(D.model4_meta&&D.model4_meta.lift||0).toFixed(1);return `<div class=row style="background:#1e2a3a;border-left:3px solid #4da3ff;padding:5px 8px;margin:4px 0"><span class=k style="color:#4da3ff">predicted perturbation response</span> <span class=lg>(Model 4 &middot; <b style="color:#e0a">predicted, not measured</b> &middot; held-out ${lift}&times; vs random)</span><div class=lg style=margin-top:3px>if knocked out, its signature should resemble <b>${d.pred}</b>; nearest predicted functional genes: ${(d.neighbors||[]).map(n=>`<a onclick=go('${n}')>${n}</a>`).join(', ')}</div></div>`;})():''}
  <div class=row><span class=k>regulation</span> <span class=v>${g.cpg?'CpG-island promoter':'non-CpG promoter'}</span> &middot; <span class=v>${g.enh}</span> <span class=k>enhancers</span> ${g.enh>=100?'<span class=lg>(highly regulated)</span>':''}</div>
  <div class=row><span class=k>essential</span> <span class=v>${g.ess==1?'yes':g.ess==0?'no':'?'}</span>${g.ess_src==='model1'?' <span class=lg>(our model'+(g.ess_prob?', p='+g.ess_prob+', '+essConf(g)+' confidence':'')+')</span>':(g.dep_frac!==undefined?' <span class=lg>(measured — DepMap: dependent in '+Math.round(g.dep_frac*100)+'% of cancer lines)</span>':(g.ess_src==='measured'&&g.ess>=0?' <span class=lg>(measured)</span>':''))} &middot; <span class=k>LOEUF</span> <span class=v>${g.loeuf<0?'-':g.loeuf}</span> &middot; <span class=k>TF</span> <span class=v>${g.tf?'yes':'no'}</span></div>
  <div class=row><span class=k>PPI</span> <span class=v>${g.ppi}</span> &middot; <span class=k>diseases</span> <span class=v>${g.ndis}</span>${g.master?' &middot; <span class=k>master:</span> <span class=v>'+g.master+'</span>':''}</div>
  ${confBadge(g)}
  ${ABUND[i]!==undefined?`<div class=row><span class=k>abundance</span> <span class=v>${'▮'.repeat(Math.max(1,Math.round(ABUND[i]/2)))}</span> <span class=lg>${ABUND[i]}/15${activeCT?' &middot; '+(exprIn(i,activeCT.t)?'<span class=ok>expressed in '+activeCT.name+'</span>':'<span class=lg>silent in '+activeCT.name+'</span>'):''}</span></div>`:''}
  ${hivHost[i]?`<div class=row><span class=warn>HIV-targeted by:</span> <span class=v>${hivHost[i].join(', ')}</span></div>`:''}
  ${act.length?`<div class=row style=margin-top:6px><span style="color:#43a047">activates &#9650;:</span> ${act.slice(0,10).map(lnk).join(', ')}${act.length>10?' +'+(act.length-10):''}</div>`:''}
  ${rep.length?`<div class=row><span style="color:#e53935">represses &#9660;:</span> ${rep.slice(0,10).map(lnk).join(', ')}${rep.length>10?' +'+(rep.length-10):''}</div>`:''}
  ${unk.length?`<div class=row><span class=k>regulates (unsigned):</span> ${unk.slice(0,10).map(lnk).join(', ')}${unk.length>10?' +'+(unk.length-10):''}</div>`:''}
  ${!outs.length?'<div class=row><span class=k>regulates:</span> -</div>':''}
  <div class=row><span class=k>regulated by:</span> ${rgl||'-'}</div>
  <div class=row><span class=k>binds (PPI):</span> ${pp||'-'}</div>
  <h3>functioning pipeline — input &rarr; gene &rarr; product &rarr; function &rarr; output</h3>${pipeline(i).map(s=>`<div class=step>${s.t}${s.m?` <span class=mach>&middot; ${s.m}</span>`:''}</div>`).join('')}
  ${extLayers(i)}
  ${biomarkerPanel(g.name)}
  ${GRXN[i]?`<h3>reactions catalyzed (Human-GEM)</h3>${GRXN[i].map(r=>`<div class=row class=lg>${r}</div>`).join('')}`:''}
  <h3>mutation impact</h3>${mutImpact(g)}
  <h3>trafficking journey — molecular detail</h3>${journey(g).map(s=>`<div class=step>${s.t}${s.m?` <span class=mach>&middot; ${s.m}</span>`:''}</div>`).join('')}
  <div class=row style=margin-top:8px><span class=btn onclick="setMode('Remove/Mutate');perturb(${i},false)">remove &rarr;</span> <span class=btn onclick="setMode('Overexpress');overexpress(${i})">overexpress &rarr;</span></div>
  ${provenance(g,i)}`;}
// FUNCTIONING PIPELINE: input -> gene -> product -> function -> output, all linked
function pipeline(i){let g=G[i];const lnk=j=>`<a onclick=go('${G[j].name}')>${G[j].name}</a>`;
 const acts=(IN[i]||[]).filter(j=>SGN[j+','+i]>0).slice(0,5).map(lnk);
 const reps=(IN[i]||[]).filter(j=>SGN[j+','+i]<0).slice(0,5).map(lnk);
 // FUNCTION at destination
 let fn='';
 if(RXN[i]) fn=`catalyzes: <span class=v>${RXN[i].sub}</span> <span class=arrow>&rarr;</span> <span class=v>${RXN[i].prod}</span> <span class=lg>(${RXN[i].pathway})</span>`;
 else if(GRXN[i]) fn=`catalyzes (Human-GEM): <span class=v>${GRXN[i][0]}</span>${GRXN[i].length>1?` <span class=lg>+${GRXN[i].length-1} more</span>`:''}`;
 else if(g.tf){let a=(OUT[i]||[]).filter(j=>SGN[i+','+j]>0).length,r=(OUT[i]||[]).filter(j=>SGN[i+','+j]<0).length;
   fn=`transcription factor &rarr; <span style="color:#43a047">activates ${a}</span> / <span style="color:#e53935">represses ${r}</span> genes`;}
 else if((PP[i]||[]).length) fn=`works in a complex — binds ${(PP[i]||[]).slice(0,4).map(lnk).join(', ')}${PP[i].length>4?' +'+(PP[i].length-4):''}`;
 else fn=`acts in <span class=v>${g.comp}</span> (${g.proc})`;
 const outs=(OUT[i]||[]).slice(0,6).map(lnk);
 const steps=[
  {t:`<b>GENOME</b> ${g.chrom?(g.chrom+(g.tss?':'+Number(g.tss).toLocaleString():'')):'(locus)'}`,m:'the gene'},
  {t:`<b>INPUT</b> — turned on by ${acts.length?acts.join(', '):'(basal)'}${reps.length?'; repressed by '+reps.join(', '):''}`,m:'regulatory signals'},
  {t:`<b>EXPRESSED</b> &rarr; mRNA &rarr; protein &rarr; ${g.comp}`,m:'transcription → translation → trafficking'},
  {t:`<b>FUNCTION</b> — ${fn}`,m:'what it does at its destination'},
  {t:`<b>OUTPUT</b> — ${g.tf&&outs.length?'drives '+outs.join(', '):(g.path?'feeds '+g.path:'contributes to '+g.proc)}`,m:'downstream effect'},
 ];
 return steps;}
// derive the full birth-to-destination journey of a protein from its compartment + role
function journey(g){
 const S=[{t:'<b>DNA</b> — gene locus (nucleus)',m:'chromatin, promoter'}];
 S.push({t:'<b>transcription</b> &rarr; pre-mRNA (hnRNA)',m:'RNA Pol II'+(g.tf?', +this is itself a TF':'')+', general TFs'});
 S.push({t:'<b>processing</b> &rarr; mature mRNA',m:'spliceosome, 5&prime;cap, poly-A'});
 S.push({t:'<b>nuclear export</b> &rarr; cytoplasm',m:'nuclear pore, NXF1/TAP'});
 const c=g.comp;
 const secretory=['plasma membrane','membrane','ER','Golgi','lysosome','endosome','extracellular','peroxisome'].includes(c);
 if(secretory && c!=='peroxisome'){
   S.push({t:'<b>translation on rough ER</b> (co-translational)',m:'ribosome + SRP, signal peptide &rarr; SEC61 translocon'});
   S.push({t:'<b>ER</b> — folding, N-glycosylation, QC',m:'BiP/calnexin, disulfide bonds'});
   if(c!=='ER'){S.push({t:'<b>COPII vesicle</b> &rarr; Golgi',m:'SAR1, SEC23/24'});
     S.push({t:'<b>Golgi</b> — glycan maturation, sorting',m:'cisternae, sorting signals'});}
   if(c==='plasma membrane'||c==='membrane') S.push({t:'<b>secretory vesicle &rarr; plasma membrane</b>',m:'SNAREs, exocyst; inserted in bilayer'});
   else if(c==='extracellular') S.push({t:'<b>secretory vesicle &rarr; exocytosis</b> (secreted)',m:'SNAREs; released outside'});
   else if(c==='lysosome') S.push({t:'<b>mannose-6-P route &rarr; lysosome</b>',m:'M6P receptor'});
   else if(c==='endosome') S.push({t:'<b>endosomal system</b>',m:'Rab GTPases'});
 } else if(c==='mitochondrion'){
   S.push({t:'<b>translation on free ribosome</b> (cytosol)',m:'ribosome'});
   S.push({t:'<b>mitochondrial import</b>',m:'TOM/TIM, N-terminal targeting presequence, PAM'});
   S.push({t:'<b>mitochondrion</b> — final fold in matrix/membrane',m:'mtHSP70, MPP cleavage'});
 } else if(c==='peroxisome'){
   S.push({t:'<b>translation on free ribosome</b>, folded in cytosol',m:'ribosome'});
   S.push({t:'<b>peroxisomal import</b> (folded)',m:'PEX5 recognises PTS1 (-SKL)'});
 } else if(c==='nucleus'){
   S.push({t:'<b>translation on free ribosome</b> (cytosol)',m:'ribosome'});
   S.push({t:'<b>nuclear import</b> &rarr; nucleus',m:'importin-&alpha;/&beta; reads NLS'+(g.tf?'; binds DNA as TF':'')});
 } else {
   S.push({t:'<b>translation on free ribosome</b> (cytosol)',m:'ribosome'});
   S.push({t:`<b>${c}</b> — mature protein at work here`,m:'chaperone-assisted folding'});
 }
 return S;}
// confidence of the essentiality prediction (how far our model's prob is from the 0.5 fence)
function essConf(g){if(g.ess_prob===undefined)return'';let d=Math.abs(g.ess_prob-0.5);return d>0.35?'high':d>0.2?'medium':'low';}
// external molecular layers for a gene: complex, signaling, PTMs, drugs, ligand/receptor, cell cycle
function extLayers(i){const lnk=j=>`<a onclick=go('${G[j].name}')>${G[j].name}</a>`;let h='';
 if(CCYC[i])h+=`<div class=row><span class=k>cell cycle</span> <span class=v>${CCYC[i]}</span></div>`;
 if(G2C[i])h+=`<div class=row><span class=k>complex</span> ${G2C[i].map(n=>`<span class=v>${n}</span>`).join('; ')}</div>`;
 const so=(SIGO[i]||[]),si=(SIGI[i]||[]);
 if(so.length)h+=`<div class=row><span class=k>signals to:</span> ${so.slice(0,8).map(([j,s])=>`${lnk(j)}${s>0?'<span style="color:#43a047">↑</span>':s<0?'<span style="color:#e53935">↓</span>':''}`).join(', ')}${so.length>8?' +'+(so.length-8):''}</div>`;
 if(si.length)h+=`<div class=row><span class=k>signaled by:</span> ${si.slice(0,8).map(([j,s])=>`${lnk(j)}`).join(', ')}${si.length>8?' +'+(si.length-8):''}</div>`;
 if(LRO[i])h+=`<div class=row><span class=k>ligand &rarr; receptors:</span> ${LRO[i].slice(0,6).map(lnk).join(', ')}</div>`;
 if(LRI[i])h+=`<div class=row><span class=k>receptor &larr; ligands:</span> ${LRI[i].slice(0,6).map(lnk).join(', ')}</div>`;
 if(PTM[i])h+=`<div class=row><span class=k>PTMs</span> <span class=v>${PTM[i].n}</span> sites${PTM[i].c&&PTM[i].c.length?' <span class=lg>('+PTM[i].c.join(', ')+')</span>':''}</div>`;
 if(DRUGS[i]&&DRUGS[i].length){const ds=DRUGS[i].slice(0,6).map(d=>{const tag=[d.t,d.a?'approved':''].filter(Boolean).join(', ');return `${d.d}${tag?'<span class=lg> ('+tag+')</span>':''}`;}).join(', ');
   h+=`<div class=row><span class=k>drugs</span> ${ds}${DRUGS[i].length>6?' +'+(DRUGS[i].length-6):''}</div>`;}
 if(CODEP[i])h+=`<div class=row><span class=k>co-essential (DepMap):</span> ${CODEP[i].slice(0,6).map(([j,r])=>`${lnk(j)} <span class=lg>${r}</span>`).join(', ')}</div>`;
 if(CODEP[i]){const dc=CODEP[i].filter(([j])=>DRUGS[j]&&DRUGS[j].length).slice(0,4);
   if(dc.length)h+=`<div class=row><span class=k>combo candidates (druggable co-essential):</span> ${dc.map(([j,r])=>`${lnk(j)} <span class=lg>${r}</span>`).join(', ')}</div>`;}
 if(LOOPS3D[i])h+=`<div class=row><span class=k>3D chromatin loops:</span> <span class=v>${LOOPS3D[i].length}</span> <span class=lg>&rarr; ${LOOPS3D[i].slice(0,3).join(', ')}</span></div>`;
 return h?`<h3>molecular layers</h3>${h}`:'';}
// ensemble confidence badge + novelty flag + target-intelligence flags
function confBadge(g){let h='';
 if(g.conf)h+=`<div class=row><span class=k>model confidence</span> <span class=v style="color:${g.conf==='high'?'#43a047':'#fb8c00'}">${g.conf==='high'?'high (measured, constraint &amp; model agree)':'split — models disagree'}</span></div>`;
 if(g.flag)h+=`<div class=row><span class=warn>&#9888; ${g.flag}</span></div>`;
 if(g.pubs!==undefined)h+=`<div class=row><span class=k>literature</span> <span class=v>${g.pubs}</span> publications${g.pubs<5&&g.ess==1?' <span class=warn>(understudied essential)</span>':''}</div>`;
 if(g.ti==='priority')h+=`<div class=row><span style="color:#ffd54f">&#9733; target-priority candidate</span> <span class=lg>(cancer-essential · druggable · wide safety window)</span></div>`;
 if(g.ti==='whitespace')h+=`<div class=row><span style="color:#c792ea">&#9733; white-space target</span> <span class=lg>(essential · undrugged · understudied)</span></div>`;
 return h;}
// biomarkers of sensitivity to targeting this gene (CCLE expression + mutation correlates)
function biomarkerPanel(name){const bm=BIOM[name];if(!bm)return'';
 const lnk=s=>idxByName[s]!==undefined?`<a onclick=go('${s}')>${s}</a>`:s;
 const es=(bm.expr||[]).filter(x=>x[1]<0),er=(bm.expr||[]).filter(x=>x[1]>0);
 const ms=(bm.mut||[]).filter(x=>x[1]<0),mr=(bm.mut||[]).filter(x=>x[1]>0);
 let h='<h3>biomarkers of sensitivity (CCLE) — who responds to targeting this</h3>';
 if(es.length)h+=`<div class=row><span style="color:#43a047">sensitive if high:</span> ${es.map(x=>`${lnk(x[0])} <span class=lg>(r ${x[1]})</span>`).join(', ')}</div>`;
 if(ms.length)h+=`<div class=row><span style="color:#43a047">sensitive if mutant:</span> ${ms.map(x=>`${lnk(x[0])} <span class=lg>(&Delta; ${x[1]})</span>`).join(', ')}</div>`;
 if(mr.length)h+=`<div class=row><span style="color:#e53935">resistant if mutant:</span> ${mr.map(x=>`${lnk(x[0])} <span class=lg>(&Delta; +${x[1]})</span>`).join(', ')}</div>`;
 if(er.length)h+=`<div class=row><span style="color:#e53935">resistant if high:</span> ${er.map(x=>`${lnk(x[0])}`).join(', ')}</div>`;
 return (es.length||ms.length||mr.length||er.length)?h:'';}
// provenance footer: where each shown label comes from
function provenance(g,i){const s=[];
 s.push('location: UniProt');s.push('networks: CollecTRI/DoRothEA + STRING');
 s.push('essentiality: '+(g.ess_src==='measured'?'measured (DepMap/Hart)':g.ess_src==='model1'?'our model (predicted)':'—'));
 if(g.path)s.push('pathway: Reactome');
 if(D.otdis[g.name])s.push('disease: Open Targets');
 if(ABUND[i]!==undefined)s.push('abundance/expression: Tabula Sapiens atlas');
 if((GF||{})[i])s.push('perturbation enrichment: Geneformer');
 if(GRXN[i])s.push('metabolism: Human-GEM');
 if(SIGO[i]||SIGI[i])s.push('signaling: SIGNOR');
 if(G2C[i])s.push('complex: Complex Portal');
 if(DRUGS[i])s.push('drugs: DGIdb');
 if(PTM[i])s.push('PTMs: UniProt');
 if(LRO[i]||LRI[i])s.push('ligand-receptor: CellPhoneDB');
 return `<div class=lg style="margin-top:8px;border-top:1px solid #1d3350;padding-top:5px">sources &middot; ${s.join(' &middot; ')}</div>`;}
// mutation -> structure/disease impact block (curated where available; LOEUF readout for all)
function mutImpact(g){let n=g.name,out='';
 // genome-wide tolerance from constraint
 let tol=g.loeuf<0?'unknown (no constraint estimate)':(g.loeuf<0.35?'<span class=warn>intolerant</span> — most loss-of-function mutations are damaging':(g.loeuf<0.7?'moderately constrained':'<span class=ok>tolerant</span> — loss-of-function is usually benign'));
 out+=`<div class=row><span class=k>mutation tolerance (LOEUF ${g.loeuf<0?'?':g.loeuf}):</span> ${tol}</div>`;
 let s=D.struct[n];
 if(s){let tot=(s.pathogenic||0)+(s.common||0),fp=tot?Math.round(100*(s.pathogenic||0)/tot):0;
  out+=`<h3>structural mutation profile</h3>
   <div class=row><span class=k>UniProt</span> <span class=v>${s.acc}</span> &middot; <span class=k>${s.residues} residues</span></div>
   <div class=row><span class=k>known variants:</span> <span style="color:#e53935">${s.pathogenic} pathogenic</span> vs <span style="color:#43a047">${s.common} common/benign</span></div>
   <div style="height:8px;background:#43a047;border-radius:4px;overflow:hidden;margin:3px 0"><div style="height:100%;width:${fp}%;background:#e53935"></div></div>
   <div class=lg>${fp}% of catalogued variants are pathogenic &middot; ~${(s.pathogenic/s.residues).toFixed(2)} pathogenic hits per residue</div>`;}
 let f=D.fold[n];
 if(f){let same=f.global_rmsd<0.1;
  out+=`<h3>fold: wild-type vs mutant</h3>
   <div class=row><span class=k>${f.mutation}</span></div>
   <div class=row><span class=k>backbone RMSD</span> <span class=v>${f.global_rmsd} &#8491;</span> (local ${f.local_rmsd_around_site} &#8491;) &rarr; ${same?'<span class=ok>fold essentially unchanged</span> — damage is functional, not structural':'<span class=warn>fold distorted</span> — structure is destabilized'}</div>`;}
 let d=D.otdis[n];
 if(d){out+=`<h3>disease associations (Open Targets)</h3>
   <div class=row><span class=k>${d.ndis} associations</span> &middot; evidence: ${d.ev} &middot; ${d.druggable?'<span class=ok>druggable</span>':'<span class=lg>not an established drug target</span>'}</div>
   ${d.top.map(t=>`<div class=row class=lg>${t[0]} <span class=v>(${t[1]})</span></div>`).join('')}`;}
 return out;}
window.mutImpact=mutImpact;
window.go=n=>{let i=idxByName[n];if(i>=0){sel=i;info(i);view.s=3;view.ox=W/2-POS[i*2]*view.s;view.oy=Hh/2-POS[i*2+1]*view.s;draw();}};
// HIV mode
function hivReport(){let proc={};for(const i in hivHost)proc[G[i].proc]=(proc[G[i].proc]||0)+1;
 const topP=Object.entries(proc).sort((a,b)=>b[1]-a[1]).slice(0,7);
 const wk=D.hiv_weakpoints.slice(0,18);
 document.getElementById('info').innerHTML=`<h2 class=warn>HIV infection</h2>
  <div class=row class=lg>19 viral proteins hijacking ${Object.keys(hivHost).length} host proteins (red ring). Entry via CD4 receptor on the membrane.</div>
  <h3>host machinery HIV hijacks</h3>${topP.map(([p,n])=>`<div class=row><span style="color:${pcol[p]}">&#9679;</span> ${p}: <span class=v>${n}</span> proteins</div>`).join('')}
  <h3>HIV's weak points — host-dependency factors</h3>
  <div class=lg>essential/hub host proteins HIV *needs*; block these to block HIV (drug-target candidates):</div>
  ${wk.map(w=>`<div class=row><a onclick=go('${w.gene}')>${w.gene}</a> <span class=lg>(${w.comp}, PPI ${w.ppi}${w.ess==1?', essential':''}) &larr; ${w.by.slice(0,3).join(',')}</span></div>`).join('')}
  <h3>integration preference</h3><div class=lg>HIV integrase inserts the provirus into transcriptionally ACTIVE, open chromatin (high-enhancer, expressed genes) — biasing toward active immune/host genes.</div>`;}
// Metabolism mode: show curated reactions as enzyme->substrate->product chains
function metabView(){metabOn=true;mode='Explore';affected=null;activeCT=null;
 const enzAll=new Set([...D.reactions.map(r=>r.i),...Object.keys(GRXN).map(Number)]);   // full metabolic enzyme complement
 mark={set:enzAll,color:'#fdd835',dim:true};
 const byPw={};D.reactions.forEach(r=>{(byPw[r.pathway]=byPw[r.pathway]||[]).push(r);});
 document.getElementById('info').innerHTML=`<h2>Metabolism</h2><div class=lg><b>${enzAll.size}</b> metabolic enzymes (yellow) — ${Object.keys(GRXN).length} carry genome-scale reactions from Human-GEM; yellow lines trace the core curated pathways. Click any enzyme for the reactions it catalyzes.</div>`+
  Object.entries(byPw).map(([pw,rs])=>`<h3>${pw}</h3>`+rs.map(r=>`<div class=row><a onclick=go('${r.enz}')>${r.enz}</a> <span class=lg>${r.sub} <span class=arrow>&rarr;</span> ${r.prod}</span></div>`).join('')).join('');
 draw();}
// Dark genes: the function frontier (no known pathway/disease)
function darkView(){metabOn=false;mode='Explore';affected=null;activeCT=null;
 const ds=[];for(let i=0;i<N;i++)if(G[i].dark)ds.push(i);
 mark={set:new Set(ds),color:'#c792ea',dim:true};
 const byC={};ds.forEach(i=>{byC[G[i].comp]=(byC[G[i].comp]||0)+1;});
 document.getElementById('info').innerHTML=`<h2>Dark genes</h2><div class=lg>${ds.length} proteins in the map with <b>no annotated pathway and no disease link</b> — the function frontier. They still sit in real compartments and many carry regulatory/PPI edges, so the network places them even where annotation can't. Purple = dark.</div>
  <h3>where they sit</h3>${Object.entries(byC).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([c,n])=>`<div class=row><span class=k>${c}</span> <span class=v>${n}</span></div>`).join('')}
  ${D.darkfn?`<h3>predicted function frontier</h3><div class=lg><b>${Object.keys(D.darkfn).length}</b> of ${ds.length} dark genes now carry a <b>predicted function</b> — guilt-by-association from <b>measured</b> functional neighbors (Perturb-seq perturbation-response similarity + co-essentiality + physical interactions), not from annotation. Every prediction is tagged <b style="color:#e0a">predicted, not known</b>. Click a purple node to see it and its evidence.</div>`:'<div class=lg style=margin-top:6px>Click any purple node: even with no pathway, its regulators/targets/binders hint at function (guilt-by-association).</div>'}`;
 draw();}
// pathway: highlight all member proteins of a Reactome pathway
function setPathway(pw){metabOn=false;affected=null;activeCT=null;
 if(!pw){mark=null;draw();document.getElementById('info').innerHTML=welcome;return;}
 const mem=D.pathways[pw]||[];mark={set:new Set(mem),color:'#00bcd4',dim:true};mode='Explore';
 const byC={};mem.forEach(i=>{byC[G[i].comp]=(byC[G[i].comp]||0)+1;});
 document.getElementById('info').innerHTML=`<h2>${pw}</h2>
  <div class=lg>${mem.length} proteins in this pathway (cyan). This is a functioning module — the proteins that act together to carry out this process, across compartments.</div>
  <h3>where it runs</h3>${Object.entries(byC).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([c,n])=>`<div class=row><span class=k>${c}</span> <span class=v>${n}</span></div>`).join('')}
  <h3>members</h3><div class=lg>${mem.slice(0,60).map(i=>`<a onclick=go('${G[i].name}')>${G[i].name}</a>`).join(', ')}${mem.length>60?' +'+(mem.length-60):''}</div>`;
 draw();}
window.setPathway=setPathway;
// Target Intelligence: ranked drug-target candidates (priority + white-space)
function targetsView(){metabOn=false;mode='Explore';affected=null;activeCT=null;
 const pri=(D.ti_priority||[]).map(t=>idxByName[t.gene]).filter(i=>i!==undefined);
 const wsp=(D.ti_whitespace||[]).map(w=>idxByName[w.gene]).filter(i=>i!==undefined);
 mark={set:new Set([...pri,...wsp]),colFn:i=>pri.includes(i)?'#ffd54f':'#c792ea',color:'#ffd54f',dim:true};
 const drug=(D.ti_priority||[]).filter(t=>t.druggable);
 document.getElementById('info').innerHTML=`<h2>Target Intelligence</h2>
  <div class=lg>Ranked drug-target candidates: DepMap essentiality &times; LOEUF safety window &times; DGIdb druggability &times; literature. <span style="color:#ffd54f">gold=priority</span>, <span style="color:#c792ea">purple=white-space</span>.</div>
  <h3>&#9733; priority targets — cancer-essential, druggable, wide safety window</h3>
  ${drug.slice(0,15).map(t=>`<div class=row><a onclick=go('${t.gene}')>${t.gene}</a> <span class=lg>window ${t.window} &middot; dep ${t.frac_dep} &middot; LOEUF ${t.loeuf}${t.selective?' &middot; <span style="color:#43a047">selective</span>':' &middot; pan-ess'} &middot; ${t.drugs.slice(0,2).join(', ')}</span></div>`).join('')}
  <h3>&#9733; white-space — essential, undrugged, understudied (discovery)</h3>
  ${(D.ti_whitespace||[]).slice(0,12).map(w=>`<div class=row><a onclick=go('${w.gene}')>${w.gene}</a> <span class=lg>dep ${w.frac_dep} &middot; ${w.pubs} pubs</span></div>`).join('')}
  <div class=lg style=margin-top:6px>Click any gene to inspect; combo candidates appear in its molecular layers.</div>`;
 draw();}
window.targetsView=targetsView;
// cell type: show the cell that is ACTUALLY expressed in this lineage (Model 2), with its own wiring
function setCellType(ct){metabOn=false;affected=null;
 if(!ct){activeCT=null;mark=null;draw();document.getElementById('info').innerHTML=welcome;return;}
 const masters=D.celltypes[ct]||[];const t=ctBit[ct];mode='Explore';
 if(t!==undefined&&EMASK.size){                       // real expression data for this cell type
   const eset=exprSet(t);activeCT={name:ct,t,set:eset};
   mark={set:eset,color:'#4dd0a0',dim:true};
   // active edges = measured edges with both ends expressed here
   let re=0;for(const e of D.reg){if(eset.has(e[0])&&eset.has(e[1]))re++;}
   let pe=0;for(const[a,b]of D.ppi){if(eset.has(a)&&eset.has(b))pe++;}
   const others=CTN.filter(n=>n!==ct&&D.celltypes[n]!==undefined);
   document.getElementById('info').innerHTML=`<h2>${ct}</h2>
    <div class=lg>Same genome, this actual cell. <b>${eset.size}</b> genes are expressed here (green, sized by abundance); the network is pruned to what's active in this cell type — a neuron and a hepatocyte get different wiring.</div>
    <div class=row><span class=k>active network</span> <span class=v>${eset.size}</span> genes &middot; <span class=v>${re}</span> regulatory &middot; <span class=v>${pe}</span> physical edges</div>
    <h3>master regulators</h3>${masters.map(n=>idxByName[n]!==undefined?`<a onclick=go('${n}')>${n}</a>`:`<span class=lg>${n}</span>`).join(' &middot; ')}
    <div class=lg style=margin-top:4px>Remove/Mutate now propagates only through genes active in this cell.</div>
    <h3>differentiate to &rarr;</h3><div>${others.map(n=>`<span class=chip onclick="differentiate('${ct}','${n.replace(/'/g,"")}')">${n}</span>`).join('')||'<span class=lg>-</span>'}</div>`;
 } else {                                              // fallback: master-TF highlight only
   activeCT=null;const mi=masters.map(n=>idxByName[n]).filter(i=>i!==undefined);
   const net=new Set(mi);mi.forEach(i=>(OUT[i]||[]).forEach(j=>net.add(j)));
   mark={set:net,color:'#4dd0a0',dim:true};
   document.getElementById('info').innerHTML=`<h2>${ct}</h2>
    <div class=lg>Same genome, different cell. The <b>master transcription factors</b> switch on a distinct network (green). <span class=warn>Run the notebook for real per-cell-type expression</span> (abundance + pruned wiring + differentiation).</div>
    <h3>master regulators</h3>${masters.map(n=>idxByName[n]!==undefined?`<a onclick=go('${n}')>${n}</a>`:`<span class=lg>${n}</span>`).join(' &middot; ')}
    <div class=row style=margin-top:6px><span class=k>active network (masters + targets)</span> <span class=v>${net.size}</span> genes</div>`;
 }
 draw();}
// differentiation: start cell type -> target; which genes switch ON / OFF (Model 2 expression delta)
function differentiate(fromName,toName){const a=ctBit[fromName],b=ctBit[toName];
 if(a===undefined||b===undefined)return;
 const on=[],off=[];for(const[i,m]of EMASK){const ea=((m>>BigInt(a))&1n)===1n,eb=((m>>BigInt(b))&1n)===1n;
   if(eb&&!ea)on.push(i);else if(ea&&!eb)off.push(i);}
 const onS=new Set(on);activeCT={name:toName,t:b,set:exprSet(b)};
 mark={set:new Set([...on,...off]),colFn:i=>onS.has(i)?'#43a047':'#e53935',color:'#888',dim:true};
 const tf=n=>idxByName[n]!==undefined?`<a onclick=go('${n}')>${n}</a>`:`<span class=lg>${n}</span>`;
 document.getElementById('info').innerHTML=`<h2>${fromName} &rarr; ${toName}</h2>
  <div class=lg>Differentiation as a state transition (Model 2). <span style="color:#43a047">green = genes switched ON</span>, <span style="color:#e53935">red = switched OFF</span> to become the target cell.</div>
  <div class=row><span style="color:#43a047">turn ON</span> <span class=v>${on.length}</span> genes &middot; <span style="color:#e53935">turn OFF</span> <span class=v>${off.length}</span></div>
  <h3>drive with these master TFs</h3><div class=lg>${(D.celltypes[toName]||[]).map(tf).join(' &middot; ')||'-'}</div>
  <h3>key genes switching ON</h3><div class=lg>${on.slice(0,24).map(i=>`<a onclick=go('${G[i].name}')>${G[i].name}</a>`).join(', ')||'-'}</div>
  <h3>key genes switching OFF</h3><div class=lg>${off.slice(0,24).map(i=>`<a onclick=go('${G[i].name}')>${G[i].name}</a>`).join(', ')||'-'}</div>`;
 draw();}
window.differentiate=differentiate;
// modes  (note: activeCT cell-type context PERSISTS across modes so Remove/Mutate stays cell-type-specific)
function setMode(m){mode=m;affected=null;mark=null;metabOn=false;
 document.querySelectorAll('#top .btn').forEach(b=>b.classList.remove('on'));
 const B={Explore:'mExplore',Processes:'mProc','Remove/Mutate':'mPerturb',Overexpress:'mOE'};
 B[m]&&document.getElementById(B[m]).classList.add('on');
 const hints={Explore:'click a protein to inspect',Processes:'colored by cellular process',
  'Remove/Mutate':'click a protein to KNOCK IT OUT and see the cascade',
  Overexpress:'click a protein to OVEREXPRESS it — green up, red down via signed regulation'};
 document.getElementById('hint').textContent=' '+hints[m];
 if(m=='Processes')document.getElementById('info').innerHTML='<h2>Processes</h2>'+PROC.map(p=>`<div class=row><span class=dot style="background:${pcol[p]}"></span>${p}</div>`).join('');
 draw();}
window.setMode=setMode;
document.getElementById('mExplore').onclick=()=>{hivOn=false;document.getElementById('mHIV').classList.remove('on');setMode('Explore');};
document.getElementById('mProc').onclick=()=>{hivOn=false;document.getElementById('mHIV').classList.remove('on');setMode('Processes');};
document.getElementById('mPerturb').onclick=()=>{hivOn=false;document.getElementById('mHIV').classList.remove('on');setMode('Remove/Mutate');};
document.getElementById('mOE').onclick=()=>{hivOn=false;document.getElementById('mHIV').classList.remove('on');setMode('Overexpress');};
document.getElementById('mHIV').onclick=()=>{hivOn=!hivOn;document.getElementById('mHIV').classList.toggle('on',hivOn);affected=null;mark=null;metabOn=false;if(hivOn){mode='Explore';hivReport();}else setMode('Explore');draw();};
function clearTopOn(){document.querySelectorAll('#top .btn').forEach(b=>b.classList.remove('on'));hivOn=false;document.getElementById('mHIV').classList.remove('on');}
document.getElementById('mMetab').onclick=()=>{clearTopOn();document.getElementById('ct').value='';document.getElementById('mMetab').classList.add('on');metabView();};
document.getElementById('mDark').onclick=()=>{clearTopOn();document.getElementById('ct').value='';document.getElementById('mDark').classList.add('on');darkView();};
document.getElementById('mTargets').onclick=()=>{clearTopOn();document.getElementById('ct').value='';document.getElementById('mTargets').classList.add('on');targetsView();};
// populate cell-type + pathway selectors
(function(){const s=document.getElementById('ct');s.innerHTML='<option value="">— cell type —</option>'+Object.keys(D.celltypes).map(c=>`<option>${c}</option>`).join('');
 s.onchange=()=>{clearTopOn();metabOn=false;document.getElementById('pw').value='';setCellType(s.value);};
 const p=document.getElementById('pw');p.innerHTML='<option value="">— pathway —</option>'+Object.keys(D.pathways||{}).map(c=>`<option>${c}</option>`).join('');
 p.onchange=()=>{clearTopOn();metabOn=false;document.getElementById('ct').value='';setPathway(p.value);};})();
document.getElementById('reset').onclick=()=>{sel=-1;affected=null;hivOn=false;mark=null;metabOn=false;activeCT=null;document.getElementById('ct').value='';document.getElementById('pw').value='';document.getElementById('mHIV').classList.remove('on');setMode('Explore');fit();document.getElementById('info').innerHTML=welcome;draw();};
document.getElementById('q').addEventListener('change',e=>{let n=e.target.value.trim();if(idxByName[n]!==undefined)go(n);});
const welcome=`<h2>The cell</h2><div class=lg>${N} proteins in their real compartments, wired by ${D.reg.length} regulatory + ${D.ppi.length} physical edges.<br><br>
 <b>Explore</b>: click any protein &rarr; see its full trafficking journey (gene&rarr;mRNA&rarr;ribosome&rarr;final location) + networks.<br>
 <b>Processes</b>: colored by function.<br>
 <b>Metabolism</b>: core reactions substrate&rarr;product.<br>
 <b>Remove/Mutate</b>: knock out a protein &rarr; watch the cascade &amp; whether the cell survives.<br>
 <b>Overexpress</b>: drive a gene up &rarr; green up / red down via signed regulation.<br>
 <b>Dark genes</b>: the function frontier (no known pathway/disease).<br>
 <b>cell type</b> (dropdown): same genome, different master-TF network.<br>
 <b>Infect: HIV</b>: what HIV hijacks + its weak points.</div>
 <div class=row style=margin-top:8px><span class=dot style=background:#e53935></span>essential <span class=dot style=background:#ff9800></span>constrained <span class=dot style=background:#3f5f7c></span>tolerant &middot; ring=disease, blue=TF, red-ring=HIV target</div>`;
document.getElementById('info').innerHTML=welcome;setMode('Explore');resize();
</script></body></html>"""
HTML=HTML.replace("__DATA__",json.dumps(D,separators=(',',':')))
(OUT/"cell_complete.html").write_text(HTML)
print("wrote cell_complete.html (%d KB)"%(len(HTML)//1024))
