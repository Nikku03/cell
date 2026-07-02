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
 <span class=btn id=mDark>Dark genes</span><span class=btn hiv id=mHIV>Infect: HIV</span>
 <select id=ct title="cell type — highlights its master-TF network"></select>
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
// HIV host index
const hivHost={};for(const hp in D.hiv)for(const[i,t]of D.hiv[hp]){(hivHost[i]=hivHost[i]||[]).push(hp);}
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
   let inMark=mark&&mark.set.has(i);
   let dim=(affected&&!affected.set.has(i)&&i!=sel)||(mark&&mark.dim&&!inMark&&i!=sel);
   let hl=(affected&&affected.set.has(i));
   if(inMark)r+=1;
   ctx.globalAlpha=dim?0.05:0.9;ctx.beginPath();ctx.arc(X,Y,hl?r+1:r,0,7);
   ctx.fillStyle=hl?affected.col(i):(inMark?mark.color:baseColor(i));ctx.fill();
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
cv.addEventListener('click',e=>{if(drag&&Math.abs(e.offsetX-drag.x)>3)return;let i=pick(e.offsetX*devicePixelRatio,e.offsetY*devicePixelRatio);if(i>=0){sel=i;if(mode=='Remove/Mutate')perturb(i,false);else info(i);draw();}});
// perturbation: BFS 2 hops over reg(out)+ppi, enriched with Geneformer in-silico downstream (Model 3)
const GF=D.gf_perturb||{};
function perturb(i,mut){let set=new Set([i]),dist={};dist[i]=0;let q=[i];
 while(q.length){let x=q.shift();if(dist[x]>=2)continue;for(const j of [...(OUT[x]||[]),...(PP[x]||[])]){if(!set.has(j)){set.add(j);dist[j]=dist[x]+1;q.push(j);}}}
 // Model 3: add Geneformer-predicted downstream genes that the measured graph missed
 const gfHit=new Set();(GF[i]||[]).forEach(j=>{if(!set.has(j)){set.add(j);dist[j]=2;gfHit.add(j);}else gfHit.add(j);});
 const proc={},path={};set.forEach(j=>{if(j==i)return;proc[G[j].proc]=(proc[G[j].proc]||0)+1;if(G[j].path)path[G[j].path]=(path[G[j].path]||0)+1;});
 affected={set,col:j=>j==i?'#fff':(gfHit.has(j)?'#c792ea':(dist[j]==1?'#ff6b6b':'#ffb36b'))};
 const lethal=G[i].ess==1||(mut&&G[i].loeuf>=0&&G[i].loeuf<0.35);
 const essNote=G[i].ess==1?(G[i].ess_src==='model1'?'essential (predicted by our model'+(G[i].ess_prob?', p='+G[i].ess_prob:'')+')':'essential gene'):'';
 const topP=Object.entries(proc).sort((a,b)=>b[1]-a[1]).slice(0,6);
 const topPath=Object.entries(path).sort((a,b)=>b[1]-a[1]).slice(0,6);
 document.getElementById('info').innerHTML=`<h2>${mut?'Mutate':'Remove'}: ${G[i].name}</h2>
  <div class=row>${lethal?'<span class=warn>&#9888; predicted cell-inviable</span> — '+(G[i].ess==1?essNote:'constrained (LOEUF '+G[i].loeuf+'), likely loss-of-function is lethal'):'<span class=ok>cell likely survives</span> — dispensable / buffered'}</div>
  <div class=row><span class=k>compartment</span> <span class=v>${G[i].comp}</span> &middot; <span class=k>process</span> <span class=v>${G[i].proc}</span></div>
  <div class=row><span class=k>direct + 2-hop affected</span> <span class=v>${set.size-1}</span> proteins${gfHit.size?` &middot; <span style="color:#c792ea">+${gfHit.size} via Geneformer</span>`:''}</div>
  <h3>processes disrupted</h3>${topP.map(([p,n])=>`<div class=row><span style="color:${pcol[p]}">&#9679;</span> ${p}: <span class=v>${n}</span></div>`).join('')}
  <h3>pathways hit</h3>${topPath.map(([p,n])=>`<div class=row class=lg>${p} (${n})</div>`).join('')||'<div class=lg>-</div>'}
  ${mut?'<h3>mutation impact</h3>'+mutImpact(G[i]):''}
  <div class=row style=margin-top:8px><span class=btn onclick="perturb(${i},true)">as mutation (LoF)</span> <span class=btn onclick="clearP()">clear</span></div>`;}
window.perturb=perturb;window.clearP=()=>{affected=null;draw();};
function info(i){let g=G[i];
 const lnk=j=>`<a onclick=go('${G[j].name}')>${G[j].name}</a>`;
 const outs=OUT[i]||[];
 let act=outs.filter(j=>SGN[i+','+j]>0),rep=outs.filter(j=>SGN[i+','+j]<0),unk=outs.filter(j=>!SGN[i+','+j]);
 let rgl=(IN[i]||[]).slice(0,10).map(lnk).join(', ');
 let pp=(PP[i]||[]).slice(0,10).map(j=>`<a onclick=go('${G[j].name}')>${G[j].name}</a>`).join(', ');
 document.getElementById('info').innerHTML=`<h2>${g.name}</h2>
  <div class=row><span class=k>compartment</span> <span class=v>${g.comp}</span></div>
  <div class=row><span class=k>process</span> <span class=v style="color:${pcol[g.proc]}">${g.proc}</span></div>
  <div class=row><span class=k>pathway</span> <span class=v>${g.path||'-'}</span></div>
  <div class=row><span class=k>essential</span> <span class=v>${g.ess==1?'yes':g.ess==0?'no':'?'}</span>${g.ess_src==='model1'?' <span class=lg>(our model'+(g.ess_prob?', p='+g.ess_prob:'')+')</span>':(g.ess_src==='measured'&&g.ess>=0?' <span class=lg>(measured)</span>':'')} &middot; <span class=k>LOEUF</span> <span class=v>${g.loeuf<0?'-':g.loeuf}</span> &middot; <span class=k>TF</span> <span class=v>${g.tf?'yes':'no'}</span></div>
  <div class=row><span class=k>PPI</span> <span class=v>${g.ppi}</span> &middot; <span class=k>diseases</span> <span class=v>${g.ndis}</span>${g.master?' &middot; <span class=k>master:</span> <span class=v>'+g.master+'</span>':''}</div>
  ${hivHost[i]?`<div class=row><span class=warn>HIV-targeted by:</span> <span class=v>${hivHost[i].join(', ')}</span></div>`:''}
  ${act.length?`<div class=row style=margin-top:6px><span style="color:#43a047">activates &#9650;:</span> ${act.slice(0,10).map(lnk).join(', ')}${act.length>10?' +'+(act.length-10):''}</div>`:''}
  ${rep.length?`<div class=row><span style="color:#e53935">represses &#9660;:</span> ${rep.slice(0,10).map(lnk).join(', ')}${rep.length>10?' +'+(rep.length-10):''}</div>`:''}
  ${unk.length?`<div class=row><span class=k>regulates (unsigned):</span> ${unk.slice(0,10).map(lnk).join(', ')}${unk.length>10?' +'+(unk.length-10):''}</div>`:''}
  ${!outs.length?'<div class=row><span class=k>regulates:</span> -</div>':''}
  <div class=row><span class=k>regulated by:</span> ${rgl||'-'}</div>
  <div class=row><span class=k>binds (PPI):</span> ${pp||'-'}</div>
  <h3>mutation impact</h3>${mutImpact(g)}
  <h3>trafficking journey — gene &rarr; final location</h3>${journey(g).map(s=>`<div class=step>${s.t}${s.m?` <span class=mach>&middot; ${s.m}</span>`:''}</div>`).join('')}
  <div class=row style=margin-top:8px><span class=btn onclick="setMode('Remove/Mutate');perturb(${i},false)">remove this protein &rarr;</span></div>`;}
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
function metabView(){metabOn=true;mode='Explore';affected=null;
 mark={set:new Set(D.reactions.map(r=>r.i)),color:'#fdd835',dim:true};
 const byPw={};D.reactions.forEach(r=>{(byPw[r.pathway]=byPw[r.pathway]||[]).push(r);});
 document.getElementById('info').innerHTML='<h2>Metabolism</h2><div class=lg>Core enzymatic reactions wired substrate&rarr;product. Yellow nodes = enzymes; yellow lines trace pathway flow. Click an enzyme to inspect.</div>'+
  Object.entries(byPw).map(([pw,rs])=>`<h3>${pw}</h3>`+rs.map(r=>`<div class=row><a onclick=go('${r.enz}')>${r.enz}</a> <span class=lg>${r.sub} <span class=arrow>&rarr;</span> ${r.prod}</span></div>`).join('')).join('');
 draw();}
// Dark genes: the function frontier (no known pathway/disease)
function darkView(){metabOn=false;mode='Explore';affected=null;
 const ds=[];for(let i=0;i<N;i++)if(G[i].dark)ds.push(i);
 mark={set:new Set(ds),color:'#c792ea',dim:true};
 const byC={};ds.forEach(i=>{byC[G[i].comp]=(byC[G[i].comp]||0)+1;});
 document.getElementById('info').innerHTML=`<h2>Dark genes</h2><div class=lg>${ds.length} proteins in the map with <b>no annotated pathway and no disease link</b> — the function frontier. They still sit in real compartments and many carry regulatory/PPI edges, so the network places them even where annotation can't. Purple = dark.</div>
  <h3>where they sit</h3>${Object.entries(byC).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([c,n])=>`<div class=row><span class=k>${c}</span> <span class=v>${n}</span></div>`).join('')}
  <div class=lg style=margin-top:6px>Click any purple node: even with no pathway, its regulators/targets/binders hint at function (guilt-by-association).</div>`;
 draw();}
// cell type: highlight the master-TF network active in that lineage
function setCellType(ct){metabOn=false;affected=null;
 if(!ct){mark=null;draw();document.getElementById('info').innerHTML=welcome;return;}
 const masters=D.celltypes[ct]||[];const mi=masters.map(n=>idxByName[n]).filter(i=>i!==undefined);
 const net=new Set(mi);mi.forEach(i=>(OUT[i]||[]).forEach(j=>net.add(j)));
 mark={set:net,color:'#4dd0a0',dim:true};mode='Explore';
 document.getElementById('info').innerHTML=`<h2>${ct}</h2>
  <div class=lg>Same genome, different cell. The <b>master transcription factors</b> for this lineage switch on a distinct gene network (green). This is how one zygote genome yields many cell types.</div>
  <h3>master regulators</h3>${masters.map(n=>idxByName[n]!==undefined?`<a onclick=go('${n}')>${n}</a>`:`<span class=lg>${n}</span>`).join(' &middot; ')}
  <div class=row style=margin-top:6px><span class=k>active network (masters + direct targets)</span> <span class=v>${net.size}</span> genes</div>`;
 draw();}
// modes
function setMode(m){mode=m;affected=null;mark=null;metabOn=false;
 document.querySelectorAll('#top .btn').forEach(b=>b.classList.remove('on'));
 ({Explore:'mExplore',Processes:'mProc','Remove/Mutate':'mPerturb'})[m]&&document.getElementById(({Explore:'mExplore',Processes:'mProc','Remove/Mutate':'mPerturb'})[m]).classList.add('on');
 const hints={Explore:'click a protein to inspect',Processes:'colored by cellular process',
  'Remove/Mutate':'click a protein to KNOCK IT OUT and see the cascade'};
 document.getElementById('hint').textContent=' '+hints[m];
 if(m=='Processes')document.getElementById('info').innerHTML='<h2>Processes</h2>'+PROC.map(p=>`<div class=row><span class=dot style="background:${pcol[p]}"></span>${p}</div>`).join('');
 draw();}
window.setMode=setMode;
document.getElementById('mExplore').onclick=()=>{hivOn=false;document.getElementById('mHIV').classList.remove('on');setMode('Explore');};
document.getElementById('mProc').onclick=()=>{hivOn=false;document.getElementById('mHIV').classList.remove('on');setMode('Processes');};
document.getElementById('mPerturb').onclick=()=>{hivOn=false;document.getElementById('mHIV').classList.remove('on');setMode('Remove/Mutate');};
document.getElementById('mHIV').onclick=()=>{hivOn=!hivOn;document.getElementById('mHIV').classList.toggle('on',hivOn);affected=null;mark=null;metabOn=false;if(hivOn){mode='Explore';hivReport();}else setMode('Explore');draw();};
function clearTopOn(){document.querySelectorAll('#top .btn').forEach(b=>b.classList.remove('on'));hivOn=false;document.getElementById('mHIV').classList.remove('on');}
document.getElementById('mMetab').onclick=()=>{clearTopOn();document.getElementById('ct').value='';document.getElementById('mMetab').classList.add('on');metabView();};
document.getElementById('mDark').onclick=()=>{clearTopOn();document.getElementById('ct').value='';document.getElementById('mDark').classList.add('on');darkView();};
// populate cell-type selector
(function(){const s=document.getElementById('ct');s.innerHTML='<option value="">— cell type —</option>'+Object.keys(D.celltypes).map(c=>`<option>${c}</option>`).join('');
 s.onchange=()=>{clearTopOn();metabOn=false;setCellType(s.value);};})();
document.getElementById('reset').onclick=()=>{sel=-1;affected=null;hivOn=false;mark=null;metabOn=false;document.getElementById('ct').value='';document.getElementById('mHIV').classList.remove('on');setMode('Explore');fit();document.getElementById('info').innerHTML=welcome;draw();};
document.getElementById('q').addEventListener('change',e=>{let n=e.target.value.trim();if(idxByName[n]!==undefined)go(n);});
const welcome=`<h2>The cell</h2><div class=lg>${N} proteins in their real compartments, wired by ${D.reg.length} regulatory + ${D.ppi.length} physical edges.<br><br>
 <b>Explore</b>: click any protein &rarr; see its full trafficking journey (gene&rarr;mRNA&rarr;ribosome&rarr;final location) + networks.<br>
 <b>Processes</b>: colored by function.<br>
 <b>Metabolism</b>: core reactions substrate&rarr;product.<br>
 <b>Remove/Mutate</b>: knock out a protein &rarr; watch the cascade &amp; whether the cell survives.<br>
 <b>Dark genes</b>: the function frontier (no known pathway/disease).<br>
 <b>cell type</b> (dropdown): same genome, different master-TF network.<br>
 <b>Infect: HIV</b>: what HIV hijacks + its weak points.</div>
 <div class=row style=margin-top:8px><span class=dot style=background:#e53935></span>essential <span class=dot style=background:#ff9800></span>constrained <span class=dot style=background:#3f5f7c></span>tolerant &middot; ring=disease, blue=TF, red-ring=HIV target</div>`;
document.getElementById('info').innerHTML=welcome;setMode('Explore');resize();
</script></body></html>"""
HTML=HTML.replace("__DATA__",json.dumps(D,separators=(',',':')))
(OUT/"cell_complete.html").write_text(HTML)
print("wrote cell_complete.html (%d KB)"%(len(HTML)//1024))
