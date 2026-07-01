"""Interactive integrated human cell map: 28k genes laid out on the genome, colored by
essentiality/constraint, with TF/disease/master overlays, click-to-inspect (all layers +
regulatory neighbors), search, zoom/pan, and a cell-type selector. Canvas for performance.
Data embedded from integrated_cell_human.csv + collectri.tsv."""
import csv, json
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
CH=[f"chr{i}" for i in range(1,23)]+["chrX","chrY"]; ci={c:i for i,c in enumerate(CH)}
sizes={l.split()[0]:int(l.split()[1]) for l in open(H/"hg38.chrom.sizes") if l.split()[0] in ci}
rows=[r for r in csv.DictReader(open(OUT/"integrated_cell_human.csv")) if r["chrom"] in ci and r["tss"]]
idx={r["gene"]:i for i,r in enumerate(rows)}
def fv(v,d=-1.0):
    try:return float(v)
    except:return d
names=[r["gene"] for r in rows]
chrom=[ci[r["chrom"]] for r in rows]
x=[round(int(r["tss"])/sizes[r["chrom"]],4) for r in rows]
ess=[1 if r["essential"]=="1" else (0 if r["essential"]=="0" else -1) for r in rows]
loeuf=[round(fv(r["loeuf"]),3) for r in rows]
ppi=[int(fv(r["ppi_degree"],0)) for r in rows]
ndis=[int(fv(r["n_diseases"],0)) for r in rows]
istf=[1 if r["is_tf"]=="1" else 0 for r in rows]
master=[r["lineage_master"] for r in rows]
regout=[int(fv(r["regulon_out"],0)) for r in rows]; regin=[int(fv(r["regulators_in"],0)) for r in rows]
npath=[int(fv(r["n_pathways"],0)) for r in rows]
# edges among genes present (collectri)
E=[]
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    a=idx.get(r["source_genesymbol"]); b=idx.get(r["target_genesymbol"])
    if a is not None and b is not None and a!=b: E.append([a,b])
# cell-type masters (validated + full lineage set)
CT={"hepatocyte":["HNF4A","HNF1A","FOXA2"],"cardiac muscle cell":["NKX2-5","GATA4","TBX5","MEF2C"],
 "natural killer cell":["EOMES","TBX21"],"macrophage":["SPI1","CEBPB"],"endothelial cell":["ERG","FLI1"],
 "T cell":["GATA3","TCF7"],"B cell":["PAX5","EBF1"],"neuron":["NEUROG2","NEUROD1","ASCL1"],
 "hematopoietic stem cell":["GATA2","RUNX1","TAL1"],"erythrocyte":["GATA1","KLF1"],
 "pancreatic beta cell":["PDX1","NKX6-1"],"skeletal muscle":["MYOD1","MYF5","MYOG"]}
CTm={k:[g for g in v if g in idx] for k,v in CT.items()}
DATA=dict(names=names,chrom=chrom,x=x,ess=ess,loeuf=loeuf,ppi=ppi,ndis=ndis,istf=istf,
    master=master,regout=regout,regin=regin,npath=npath,chroms=CH,edges=E,celltypes=CTm)
html=r"""<!doctype html><html><head><meta charset=utf-8><title>Integrated human cell map</title>
<style>
 html,body{margin:0;height:100%;background:#0b1420;color:#dfe8f0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
 #wrap{display:flex;height:100vh}#cv{flex:1;display:block;cursor:crosshair}
 #side{width:330px;background:#111f30;padding:12px 14px;overflow:auto;box-shadow:-2px 0 12px #0007;font-size:13px}
 h2{color:#7fd1ff;font-size:16px;margin:.3em 0}.k{color:#8fa8c0}.v{color:#e8f0f8;font-weight:600}
 .row{margin:3px 0}.chip{display:inline-block;margin:2px;padding:3px 7px;border-radius:5px;background:#1b3a5c;color:#cfe8ff;cursor:pointer;font-size:11px}
 .chip:hover{background:#276}.chip.on{background:#2e86de;color:#fff}
 #tip{position:absolute;pointer-events:none;background:#000c;border:1px solid #345;border-radius:5px;padding:5px 8px;font-size:12px;display:none;z-index:9}
 #top{position:absolute;top:8px;left:10px;font-size:12px;color:#9fb3c8;background:#0009;padding:6px 10px;border-radius:6px}
 input{background:#0d1b2a;border:1px solid #345;color:#dfe8f0;border-radius:5px;padding:4px 7px;width:150px}
 a{color:#7fd1ff;cursor:pointer;text-decoration:none}a:hover{text-decoration:underline}
 .lg{font-size:11px;color:#9fb3c8}.dot{display:inline-block;width:9px;height:9px;border-radius:9px;vertical-align:middle;margin-right:4px}
</style></head><body><div id=wrap>
<canvas id=cv></canvas>
<div id=side>
 <h2>Integrated human cell</h2>
 <div class=lg><span class=dot style=background:#e53935></span>essential-core &nbsp; <span class=dot style=background:#ff9800></span>constrained &nbsp; <span class=dot style=background:#4a6a8a></span>tolerant</div>
 <div class=lg>ring=disease &middot; blue=TF &middot; star=lineage master</div>
 <div class=row style=margin-top:8px><input id=q placeholder="search gene..."><span id=stat></span></div>
 <h2 style=margin-top:10px>Cell type</h2>
 <div id=cts></div>
 <div id=info style=margin-top:12px><i class=lg>click a gene to inspect its layers + regulatory neighbours</i></div>
</div></div><div id=tip></div>
<script>
const D=__DATA__;const N=D.names.length;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let W,Hh; function resize(){W=cv.width=cv.clientWidth*devicePixelRatio;Hh=cv.height=cv.clientHeight*devicePixelRatio;draw();}
window.addEventListener('resize',resize);
let view={s:1,ox:0,oy:0}; // scale, offset
const NC=D.chroms.length, padL=60*devicePixelRatio, padR=20*devicePixelRatio, padT=20*devicePixelRatio, padB=20*devicePixelRatio;
function rowY(c){return padT+(Hh-padT-padB)*(c+0.5)/NC;}
function gx(i){return padL+(W-padL-padR)*D.x[i];}
function color(i){ if(D.ess[i]==1)return '#e53935'; if(D.ess[i]==0)return '#4a6a8a';
  let l=D.loeuf[i]; if(l<0)return '#33506e'; if(l<0.35)return '#ff9800'; if(l<0.7)return '#c9a24a'; return '#3d6a8a';}
let selCT=null, selGene=-1, nbrs=new Set(), mastersOn=new Set();
function draw(){
 ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,W,Hh);ctx.fillStyle='#0b1420';ctx.fillRect(0,0,W,Hh);
 ctx.save();ctx.translate(view.ox,view.oy);ctx.scale(view.s,view.s);
 // chromosome bars
 ctx.strokeStyle='#22405e';ctx.lineWidth=1*devicePixelRatio;ctx.fillStyle='#7f93a8';ctx.font=(11*devicePixelRatio)+'px sans-serif';
 for(let c=0;c<NC;c++){let y=rowY(c);ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(W-padR,y);ctx.stroke();
   ctx.fillText(D.chroms[c].replace('chr',''),padL-30*devicePixelRatio,y+4*devicePixelRatio);}
 // genes
 const r0=1.4*devicePixelRatio;
 for(let i=0;i<N;i++){
   let X=gx(i),Y=rowY(D.chrom[i]); let r=r0+Math.min(D.ppi[i]/40,3)*devicePixelRatio;
   let dim = selCT && !mastersOn.has(i) && !(selGene>=0&&nbrs.has(i));
   ctx.globalAlpha = dim?0.12:0.9;
   ctx.beginPath();ctx.arc(X,Y,r,0,7);ctx.fillStyle=color(i);ctx.fill();
   if(D.ndis[i]>=5){ctx.globalAlpha=dim?0.15:0.8;ctx.strokeStyle='#ff9800';ctx.lineWidth=1*devicePixelRatio;ctx.stroke();}
   if(D.istf[i]){ctx.globalAlpha=dim?0.2:0.9;ctx.strokeStyle='#2e86de';ctx.lineWidth=1.2*devicePixelRatio;ctx.stroke();}
 }
 // masters of selected cell type = stars
 ctx.globalAlpha=1;
 mastersOn.forEach(i=>{let X=gx(i),Y=rowY(D.chrom[i]);star(X,Y,6*devicePixelRatio,'#ffd166');});
 // selected gene + neighbours
 if(selGene>=0){let X=gx(selGene),Y=rowY(D.chrom[selGene]);
   nbrs.forEach(j=>{ctx.strokeStyle='#7fd166';ctx.globalAlpha=.5;ctx.lineWidth=.6*devicePixelRatio;ctx.beginPath();ctx.moveTo(X,Y);ctx.lineTo(gx(j),rowY(D.chrom[j]));ctx.stroke();});
   ctx.globalAlpha=1;star(X,Y,7*devicePixelRatio,'#fff');}
 ctx.restore();
}
function star(x,y,r,col){ctx.beginPath();for(let k=0;k<10;k++){let a=Math.PI/5*k-Math.PI/2,rr=k%2?r*0.45:r;ctx.lineTo(x+Math.cos(a)*rr,y+Math.sin(a)*rr);}ctx.closePath();ctx.fillStyle=col;ctx.fill();}
// interaction
cv.addEventListener('wheel',e=>{e.preventDefault();let f=e.deltaY<0?1.15:1/1.15;let mx=e.offsetX*devicePixelRatio,my=e.offsetY*devicePixelRatio;
 view.ox=mx-(mx-view.ox)*f;view.oy=my-(my-view.oy)*f;view.s*=f;draw();},{passive:false});
let drag=null;cv.addEventListener('mousedown',e=>drag={x:e.offsetX,y:e.offsetY,ox:view.ox,oy:view.oy});
window.addEventListener('mouseup',()=>drag=null);
cv.addEventListener('mousemove',e=>{ if(drag){view.ox=drag.ox+(e.offsetX-drag.x)*devicePixelRatio;view.oy=drag.oy+(e.offsetY-drag.y)*devicePixelRatio;draw();return;}
 let i=pick(e.offsetX*devicePixelRatio,e.offsetY*devicePixelRatio); const t=document.getElementById('tip');
 if(i>=0){t.style.display='block';t.style.left=(e.offsetX+12)+'px';t.style.top=(e.offsetY+8)+'px';
   t.innerHTML=`<b>${D.names[i]}</b> ${D.chroms[D.chrom[i]]} &middot; ${D.ess[i]==1?'essential':D.ess[i]==0?'non-ess':'?'} &middot; LOEUF ${D.loeuf[i]<0?'-':D.loeuf[i]} &middot; PPI ${D.ppi[i]}`;}
 else t.style.display='none';});
function pick(px,py){ let best=-1,bd=64*devicePixelRatio*devicePixelRatio;
 let wx=(px-view.ox)/view.s,wy=(py-view.oy)/view.s;
 for(let i=0;i<N;i++){let dx=gx(i)-wx,dy=rowY(D.chrom[i])-wy,d=dx*dx+dy*dy;if(d<bd){bd=d;best=i;}}return best;}
cv.addEventListener('click',e=>{if(drag&&(Math.abs(e.offsetX-drag.x)>3))return;let i=pick(e.offsetX*devicePixelRatio,e.offsetY*devicePixelRatio);if(i>=0)select(i);});
// adjacency
let OUT={},IN={};for(const [a,b] of D.edges){(OUT[a]=OUT[a]||[]).push(b);(IN[b]=IN[b]||[]).push(a);}
function select(i){ selGene=i;nbrs=new Set([...(OUT[i]||[]),...(IN[i]||[])]);
 let tg=(OUT[i]||[]).slice(0,12).map(j=>`<a onclick=selectByName('${D.names[j]}')>${D.names[j]}</a>`).join(', ');
 let rg=(IN[i]||[]).slice(0,12).map(j=>`<a onclick=selectByName('${D.names[j]}')>${D.names[j]}</a>`).join(', ');
 document.getElementById('info').innerHTML=
  `<h2>${D.names[i]}</h2>
   <div class=row><span class=k>location</span> <span class=v>${D.chroms[D.chrom[i]]}</span></div>
   <div class=row><span class=k>essential</span> <span class=v>${D.ess[i]==1?'yes (core)':D.ess[i]==0?'no':'unknown'}</span> &middot; <span class=k>LOEUF</span> <span class=v>${D.loeuf[i]<0?'-':D.loeuf[i]}</span></div>
   <div class=row><span class=k>TF</span> <span class=v>${D.istf[i]?'yes':'no'}</span> &middot; <span class=k>PPI</span> <span class=v>${D.ppi[i]}</span> &middot; <span class=k>pathways</span> <span class=v>${D.npath[i]}</span></div>
   <div class=row><span class=k>diseases</span> <span class=v>${D.ndis[i]}</span> ${D.master[i]?'&middot; <span class=k>lineage master:</span> <span class=v>'+D.master[i]+'</span>':''}</div>
   <div class=row style=margin-top:6px><span class=k>regulates (${(OUT[i]||[]).length}):</span> ${tg||'-'}</div>
   <div class=row><span class=k>regulated by (${(IN[i]||[]).length}):</span> ${rg||'-'}</div>`;
 draw();}
window.selectByName=n=>{let i=D.names.indexOf(n);if(i>=0)select(i);};
document.getElementById('q').addEventListener('change',e=>{let i=D.names.indexOf(e.target.value.trim());if(i>=0){select(i);view.s=6;view.ox=W/2-gx(i)*view.s;view.oy=Hh/2-rowY(D.chrom[i])*view.s;draw();}});
// cell type chips
const cts=document.getElementById('cts');
Object.keys(D.celltypes).forEach(ct=>{let b=document.createElement('span');b.className='chip';b.textContent=ct;
 b.onclick=()=>{if(selCT===ct){selCT=null;mastersOn=new Set();}else{selCT=ct;mastersOn=new Set((D.celltypes[ct]||[]).map(g=>D.names.indexOf(g)).filter(x=>x>=0));}
   document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',c.textContent===selCT));
   document.getElementById('info').innerHTML=selCT?`<h2>${selCT}</h2><div class=lg>masters highlighted (stars). Full per-gene activity loads from the atlas (celltype_expression.csv).</div>`:'<i class=lg>click a gene to inspect</i>';draw();};
 cts.appendChild(b);});
document.getElementById('stat').innerHTML=` &nbsp;${N} genes`;
resize();
</script></body></html>"""
html=html.replace("__DATA__",json.dumps(DATA,separators=(',',':')))
(OUT/"cell_human_map.html").write_text(html)
print(f"wrote cell_human_map.html ({len(html)//1024} KB) | {len(rows)} genes, {len(E)} edges, {len(CTm)} cell types")
