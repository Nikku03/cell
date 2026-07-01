"""Spatial cell map: proteins placed in their real subcellular compartments (UniProt),
colored by essentiality/constraint, with TF/disease/master overlays. Click a protein ->
its layers + compartment + pathway (Reactome) + regulatory neighbours. Canvas, zoom/pan,
search, cell-type selector. Data embedded from integrated map + localization + reactome."""
import csv, json, gzip
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
loc=json.load(open(H/"gene_compartment.json"))
COMPS=["nucleus","cytoplasm","plasma membrane","membrane","extracellular","mitochondrion","ER","Golgi","cytoskeleton","lysosome","peroxisome","endosome","unknown"]
comp_i={c:k for k,c in enumerate(COMPS)}
# ENSG->symbol (from STRING aliases) for reactome; top pathway per gene
ensp2sym={}; ensp2ensg={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<3: continue
    if p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
    elif p[2]=="Ensembl_HGNC_ensembl_gene_id": ensp2ensg[p[0]]=p[1]
ensg2sym={ensp2ensg[e]:ensp2sym[e] for e in ensp2sym if e in ensp2ensg}
GENERIC={"Metabolism","Signal Transduction","Immune System","Disease","Gene expression (Transcription)","Metabolism of proteins","Developmental Biology","Homeostasis"}
paths=defaultdict(list)
for l in open(H/"reactome_human.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<4 or not p[0].startswith("ENSG"): continue
    s=ensg2sym.get(p[0])
    if s and p[3] not in GENERIC: paths[s].append(p[3])
def toppath(g):
    ps=paths.get(g,[])
    return min(ps,key=len) if ps else ""
# integrated map
rows=[r for r in csv.DictReader(open(OUT/"integrated_cell_human.csv")) if r["gene"] and loc.get(r["gene"],"unknown")!="unknown"]  # localized proteins only
idx={r["gene"]:i for i,r in enumerate(rows)}
def fv(v,d=-1.0):
    try:return float(v)
    except:return d
names=[r["gene"] for r in rows]
comp=[comp_i.get(loc.get(r["gene"],"unknown"),comp_i["unknown"]) for r in rows]
ess=[1 if r["essential"]=="1" else (0 if r["essential"]=="0" else -1) for r in rows]
loeuf=[round(fv(r["loeuf"]),3) for r in rows]
ppi=[int(fv(r["ppi_degree"],0)) for r in rows]; ndis=[int(fv(r["n_diseases"],0)) for r in rows]
istf=[1 if r["is_tf"]=="1" else 0 for r in rows]; master=[r["lineage_master"] for r in rows]
npath=[int(fv(r["n_pathways"],0)) for r in rows]; tp=[toppath(r["gene"]) for r in rows]
E=[]
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    a=idx.get(r["source_genesymbol"]); b=idx.get(r["target_genesymbol"])
    if a is not None and b is not None and a!=b: E.append([a,b])
CT={"hepatocyte":["HNF4A","HNF1A","FOXA2"],"cardiac muscle cell":["NKX2-5","GATA4","TBX5","MEF2C"],
 "natural killer cell":["EOMES","TBX21"],"macrophage":["SPI1","CEBPB"],"endothelial cell":["ERG","FLI1"],
 "T cell":["GATA3","TCF7"],"B cell":["PAX5","EBF1"],"neuron":["NEUROG2","NEUROD1","ASCL1"],
 "erythrocyte":["GATA1","KLF1"],"pancreatic beta cell":["PDX1","NKX6-1"],"skeletal muscle":["MYOD1","MYF5","MYOG"]}
CTm={k:[g for g in v if g in idx] for k,v in CT.items()}
DATA=dict(names=names,comp=comp,ess=ess,loeuf=loeuf,ppi=ppi,ndis=ndis,istf=istf,master=master,
    npath=npath,toppath=tp,comps=COMPS,edges=E,celltypes=CTm)
from collections import Counter
print("compartment counts:",dict(Counter(COMPS[c] for c in comp).most_common()))
html=r"""<!doctype html><html><head><meta charset=utf-8><title>Spatial human cell</title>
<style>
 html,body{margin:0;height:100%;background:#0a1119;color:#dfe8f0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
 #wrap{display:flex;height:100vh}#cv{flex:1;display:block;cursor:crosshair}
 #side{width:340px;background:#0f1c2b;padding:12px 14px;overflow:auto;box-shadow:-2px 0 12px #0007;font-size:13px}
 h2{color:#7fd1ff;font-size:16px;margin:.3em 0}.k{color:#8fa8c0}.v{color:#e8f0f8;font-weight:600}.row{margin:3px 0}
 .chip{display:inline-block;margin:2px;padding:3px 7px;border-radius:5px;background:#1b3a5c;color:#cfe8ff;cursor:pointer;font-size:11px}
 .chip:hover{background:#276}.chip.on{background:#2e86de;color:#fff}
 #tip{position:absolute;pointer-events:none;background:#000d;border:1px solid #345;border-radius:5px;padding:5px 8px;font-size:12px;display:none;z-index:9}
 input{background:#0a1119;border:1px solid #345;color:#dfe8f0;border-radius:5px;padding:4px 7px;width:150px}
 a{color:#7fd1ff;cursor:pointer}a:hover{text-decoration:underline}.lg{font-size:11px;color:#9fb3c8}
 .dot{display:inline-block;width:9px;height:9px;border-radius:9px;vertical-align:middle;margin-right:4px}
</style></head><body><div id=wrap>
<canvas id=cv></canvas>
<div id=side>
 <h2>Spatial human cell</h2>
 <div class=lg><span class=dot style=background:#e53935></span>essential <span class=dot style=background:#ff9800></span>constrained <span class=dot style=background:#4a6a8a></span>tolerant &middot; ring=disease &middot; blue=TF</div>
 <div class=row style=margin-top:8px><input id=q placeholder="search protein..."></div>
 <h2 style=margin-top:10px>Cell type</h2><div id=cts></div>
 <div id=info style=margin-top:12px><i class=lg>click a protein to see its compartment, pathway, layers, and regulatory neighbours</i></div>
</div></div><div id=tip></div>
<script>
const D=__DATA__;const N=D.names.length;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
// compartment regions (in world coords 1200x820): center + radii for scatter; 'ring' for membranes
const CX=560,CY=410,RX=520,RY=372;
const REG={nucleus:{x:330,y:410,rx:170,ry:150},cytoplasm:{x:620,y:430,rx:330,ry:260},
 'plasma membrane':{ring:1},membrane:{ring:1},extracellular:{out:1},
 mitochondrion:{multi:[[720,240,70,32],[820,540,64,30],[470,600,60,28]]},
 ER:{x:520,y:430,rx:150,ry:120,shell:170},Golgi:{x:690,y:300,rx:70,ry:36},
 cytoskeleton:{x:620,y:430,rx:330,ry:250},lysosome:{multi:[[860,360,26,26],[790,430,22,22]]},
 peroxisome:{multi:[[500,300,20,20]]},endosome:{multi:[[770,360,28,24]]},unknown:{x:1080,y:120,rx:70,ry:70}};
function hash(i){let h=i*2654435761%2147483647;return [(h%1000)/1000,((h/1000|0)%1000)/1000];}
const POS=new Float32Array(N*2);
for(let i=0;i<N;i++){let c=D.comps[D.comp[i]],[u,v]=hash(i+1),reg=REG[c]||REG.unknown,x,y;
 if(reg.ring){let a=u*6.2832,rr=1;x=CX+Math.cos(a)*RX*rr;y=CY+Math.sin(a)*RY*rr;}
 else if(reg.out){let a=u*6.2832,rr=1.06+v*0.12;x=CX+Math.cos(a)*RX*rr;y=CY+Math.sin(a)*RY*rr;}
 else if(reg.multi){let m=reg.multi[(i%reg.multi.length)];let a=u*6.2832;x=m[0]+Math.cos(a)*m[2]*v;y=m[1]+Math.sin(a)*m[3]*v;}
 else if(reg.shell){let a=u*6.2832,rr=reg.rx*(0.85+v*0.35);x=reg.x+Math.cos(a)*rr;y=reg.y+Math.sin(a)*rr*0.8;}
 else{let a=u*6.2832,rr=Math.sqrt(v);x=reg.x+Math.cos(a)*reg.rx*rr;y=reg.y+Math.sin(a)*reg.ry*rr;}
 POS[i*2]=x;POS[i*2+1]=y;}
let W,Hh;function resize(){W=cv.width=cv.clientWidth*devicePixelRatio;Hh=cv.height=cv.clientHeight*devicePixelRatio;fit();draw();}
window.addEventListener('resize',resize);let view={s:1,ox:0,oy:0};
function fit(){let s=Math.min(W/1200,Hh/820)*0.96;view.s=s;view.ox=(W-1200*s)/2;view.oy=(Hh-820*s)/2;}
function color(i){if(D.ess[i]==1)return '#e53935';let l=D.loeuf[i];if(l<0)return '#33506e';if(l<0.35)return '#ff9800';if(l<0.7)return '#c9a24a';return '#41627f';}
let selCT=null,selGene=-1,nbrs=new Set(),mastersOn=new Set();
function draw(){ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#0a1119';ctx.fillRect(0,0,W,Hh);
 ctx.save();ctx.translate(view.ox,view.oy);ctx.scale(view.s,view.s);
 // compartments
 ctx.lineWidth=2;ctx.font='15px sans-serif';
 e(CX,CY,RX,RY,'#0e2036','#4a80c0',5,'');            // plasma membrane
 ctx.fillStyle='#7fa8d0';ctx.fillText('extracellular',CX+RX-120,CY-RY+18);ctx.fillText('cytoplasm',CX+120,CY-RY+70);
 e(REG.ER.x,REG.ER.y,REG.ER.rx,REG.ER.ry,'#15263a00','#e08a5a',1.5,'');
 e(330,410,170,150,'#14243c','#8f7fd1',4,'nucleus');
 for(const m of REG.mitochondrion.multi) e(m[0],m[1],m[2],m[3],'#231533','#d15a9e',2.5,'');
 ctx.fillStyle='#d18ac0';ctx.fillText('mito',REG.mitochondrion.multi[0][0]-15,REG.mitochondrion.multi[0][1]-38);
 e(REG.Golgi.x,REG.Golgi.y,REG.Golgi.rx,REG.Golgi.ry,'#0e2a28','#5ad1c0',2.5,'Golgi');
 ctx.fillStyle='#c08a6a';ctx.fillText('ER',REG.ER.x-80,REG.ER.y-120);
 // proteins
 for(let i=0;i<N;i++){let X=POS[i*2],Y=POS[i*2+1];let r=1.3+Math.min(D.ppi[i]/45,2.6);
   let dim=selCT&&!mastersOn.has(i)&&!(selGene>=0&&nbrs.has(i));ctx.globalAlpha=dim?0.08:0.9;
   ctx.beginPath();ctx.arc(X,Y,r,0,7);ctx.fillStyle=color(i);ctx.fill();
   if(D.ndis[i]>=5){ctx.globalAlpha=dim?.1:.75;ctx.strokeStyle='#ff9800';ctx.lineWidth=.8;ctx.stroke();}
   if(D.istf[i]){ctx.globalAlpha=dim?.15:.85;ctx.strokeStyle='#2e86de';ctx.lineWidth=1;ctx.stroke();}}
 ctx.globalAlpha=1;mastersOn.forEach(i=>star(POS[i*2],POS[i*2+1],7,'#ffd166'));
 if(selGene>=0){let X=POS[selGene*2],Y=POS[selGene*2+1];nbrs.forEach(j=>{ctx.strokeStyle='#7fd166';ctx.globalAlpha=.45;ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(X,Y);ctx.lineTo(POS[j*2],POS[j*2+1]);ctx.stroke();});ctx.globalAlpha=1;star(X,Y,8,'#fff');}
 ctx.restore();}
function e(x,y,rx,ry,fill,stroke,lw,label){ctx.beginPath();ctx.ellipse(x,y,rx,ry,0,0,7);if(fill){ctx.fillStyle=fill;ctx.fill();}ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.stroke();if(label){ctx.fillStyle=stroke;ctx.fillText(label,x-ctx.measureText(label).width/2,y-ry-6);}}
function star(x,y,r,col){ctx.beginPath();for(let k=0;k<10;k++){let a=Math.PI/5*k-1.57,rr=k%2?r*.45:r;ctx.lineTo(x+Math.cos(a)*rr,y+Math.sin(a)*rr);}ctx.closePath();ctx.fillStyle=col;ctx.fill();}
cv.addEventListener('wheel',ev=>{ev.preventDefault();let f=ev.deltaY<0?1.15:1/1.15,mx=ev.offsetX*devicePixelRatio,my=ev.offsetY*devicePixelRatio;view.ox=mx-(mx-view.ox)*f;view.oy=my-(my-view.oy)*f;view.s*=f;draw();},{passive:false});
let drag=null;cv.addEventListener('mousedown',ev=>drag={x:ev.offsetX,y:ev.offsetY,ox:view.ox,oy:view.oy});window.addEventListener('mouseup',()=>drag=null);
cv.addEventListener('mousemove',ev=>{if(drag){view.ox=drag.ox+(ev.offsetX-drag.x)*devicePixelRatio;view.oy=drag.oy+(ev.offsetY-drag.y)*devicePixelRatio;draw();return;}
 let i=pick(ev.offsetX*devicePixelRatio,ev.offsetY*devicePixelRatio),t=document.getElementById('tip');
 if(i>=0){t.style.display='block';t.style.left=(ev.offsetX+12)+'px';t.style.top=(ev.offsetY+8)+'px';t.innerHTML=`<b>${D.names[i]}</b> &middot; ${D.comps[D.comp[i]]}`;}else t.style.display='none';});
function pick(px,py){let wx=(px-view.ox)/view.s,wy=(py-view.oy)/view.s,best=-1,bd=100;for(let i=0;i<N;i++){let dx=POS[i*2]-wx,dy=POS[i*2+1]-wy,d=dx*dx+dy*dy;if(d<bd){bd=d;best=i;}}return best;}
cv.addEventListener('click',ev=>{if(drag&&Math.abs(ev.offsetX-drag.x)>3)return;let i=pick(ev.offsetX*devicePixelRatio,ev.offsetY*devicePixelRatio);if(i>=0)select(i);});
let OUT={},IN={};for(const[a,b]of D.edges){(OUT[a]=OUT[a]||[]).push(b);(IN[b]=IN[b]||[]).push(a);}
function select(i){selGene=i;nbrs=new Set([...(OUT[i]||[]),...(IN[i]||[])]);
 let tg=(OUT[i]||[]).slice(0,10).map(j=>`<a onclick=selByName('${D.names[j]}')>${D.names[j]}</a>`).join(', ');
 let rg=(IN[i]||[]).slice(0,10).map(j=>`<a onclick=selByName('${D.names[j]}')>${D.names[j]}</a>`).join(', ');
 document.getElementById('info').innerHTML=`<h2>${D.names[i]}</h2>
   <div class=row><span class=k>compartment</span> <span class=v>${D.comps[D.comp[i]]}</span></div>
   <div class=row><span class=k>pathway</span> <span class=v>${D.toppath[i]||'-'}</span> <span class=lg>(${D.npath[i]} total)</span></div>
   <div class=row><span class=k>essential</span> <span class=v>${D.ess[i]==1?'yes':D.ess[i]==0?'no':'?'}</span> &middot; <span class=k>LOEUF</span> <span class=v>${D.loeuf[i]<0?'-':D.loeuf[i]}</span> &middot; <span class=k>TF</span> <span class=v>${D.istf[i]?'yes':'no'}</span></div>
   <div class=row><span class=k>PPI</span> <span class=v>${D.ppi[i]}</span> &middot; <span class=k>diseases</span> <span class=v>${D.ndis[i]}</span> ${D.master[i]?'&middot; <span class=k>master:</span> <span class=v>'+D.master[i]+'</span>':''}</div>
   <div class=row style=margin-top:6px><span class=k>regulates (${(OUT[i]||[]).length}):</span> ${tg||'-'}</div>
   <div class=row><span class=k>regulated by (${(IN[i]||[]).length}):</span> ${rg||'-'}</div>`;draw();}
window.selByName=n=>{let i=D.names.indexOf(n);if(i>=0)select(i);};
document.getElementById('q').addEventListener('change',e=>{let i=D.names.indexOf(e.target.value.trim());if(i>=0){select(i);view.s=3;view.ox=W/2-POS[i*2]*view.s;view.oy=Hh/2-POS[i*2+1]*view.s;draw();}});
const cts=document.getElementById('cts');Object.keys(D.celltypes).forEach(ct=>{let b=document.createElement('span');b.className='chip';b.textContent=ct;
 b.onclick=()=>{if(selCT===ct){selCT=null;mastersOn=new Set();}else{selCT=ct;mastersOn=new Set((D.celltypes[ct]||[]).map(g=>D.names.indexOf(g)).filter(x=>x>=0));}document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',c.textContent===selCT));draw();};cts.appendChild(b);});
resize();
</script></body></html>"""
html=html.replace("__DATA__",json.dumps(DATA,separators=(',',':')))
(OUT/"cell_spatial.html").write_text(html)
print(f"wrote cell_spatial.html ({len(html)//1024} KB) | {len(rows)} proteins, {len(E)} edges")
