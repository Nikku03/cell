"""Metabolic Routes — alternative-pathway explorer over Human-GEM (11,271 reactions, 4,142 metabolites).
Answers: how else can this metabolite be made? which enzymes are interchangeable (isozymes)? if I knock
out an enzyme, is the product still reachable and by what route? Self-contained HTML."""
import json
from pathlib import Path
OUT=Path("outputs/orphan")
G=json.load(open(OUT/"metabolic_graph.json"))
HTML=r"""<!doctype html><html><head><meta charset=utf-8><title>Metabolic Routes</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#05090f;color:#e6eef7;font-family:-apple-system,'Segoe UI',Inter,sans-serif;overflow:hidden}
#top{height:50px;display:flex;align-items:center;gap:9px;padding:0 15px;background:#0a1420;border-bottom:1px solid #16304c}
.brand{font-weight:700;color:#cfe6ff}.brand small{color:#5f7fa6;font-weight:400;font-size:11px}
input{background:#07101b;border:1px solid #24405f;color:#e6eef7;border-radius:8px;padding:7px 11px;font-size:13px}
#q{width:230px}.spacer{flex:1}.tab{padding:7px 13px;border-radius:8px;background:#12283f;color:#bcd6f2;cursor:pointer;font-size:12px}.tab.on{background:#1f6feb;color:#fff}
#wrap{display:flex;height:calc(100vh - 51px)}
#list{width:340px;border-right:1px solid #12283f;overflow:auto;padding:8px}
#main{flex:1;overflow:auto;padding:18px 24px}
.item{padding:8px 10px;border-radius:8px;cursor:pointer;font-size:13px;border:1px solid transparent}
.item:hover{background:#0e1f31}.item.sel{background:#123253;border-color:#2e6fd6}
.item .t{color:#e6eef7}.item .s{color:#6f8fb5;font-size:11px}
h1{font-size:22px;margin:0 0 3px;color:#eaf4ff}h2{font-size:14px;color:#8fd0ff;margin:20px 0 6px;border-bottom:1px solid #12283f;padding-bottom:4px}
.rxn{border:1px solid #14293f;border-radius:10px;padding:10px 13px;margin:7px 0;background:#0a1522}
.rxn .eq{font-size:13.5px;color:#d6e6f7;line-height:1.5}.rxn .meta{margin-top:5px;font-size:11.5px;color:#8aa3c0}
.enz{display:inline-block;background:#123b2a;color:#7ff0b5;border-radius:6px;padding:2px 7px;margin:2px 3px 0 0;font-size:11px;cursor:pointer}
.enz.iso{background:#3a2f10;color:#ffd98f}.enz.ko{background:#5a1526;color:#ff9db0;text-decoration:line-through}
.met{color:#7fd1ff;cursor:pointer}.met:hover{text-decoration:underline}
.route{border-left:3px solid #2e6fd6;padding:4px 0 4px 12px;margin:8px 0}
.step{padding:6px 10px;background:#0c1c30;border-radius:7px;margin:5px 0;font-size:12.5px}
.chip{display:inline-block;background:#12283f;color:#bcd6f2;border-radius:7px;padding:4px 9px;margin:3px 4px 0 0;font-size:12px;cursor:pointer}.chip:hover{background:#1c4066}
.warn{color:#ff9db0}.ok{color:#7ff0b5}.lg{color:#7f9ec0;font-size:11.5px}.big{font-size:15px;color:#eaf4ff;font-weight:600}
#rf{background:#0a1522;border:1px solid #14293f;border-radius:10px;padding:12px;margin-bottom:14px}
#rf input{width:180px}
</style></head><body>
<div id=top>
 <span class=brand>Metabolic Routes <small>— Human-GEM &middot; 11,271 reactions &middot; 4,142 metabolites</small></span>
 <span class=spacer></span>
 <span class=tab id=tMet>Metabolites</span><span class=tab id=tEnz>Enzymes</span>
 <input id=q placeholder="search metabolite or enzyme">
</div>
<div id=wrap><div id=list></div><div id=main></div></div>
<script>
const G=__DATA__;const MET=G.mets,RX=G.rxns,HUB=new Set(G.hubs),G2R=G.gene2rxn;
const mi={};MET.forEach((n,i)=>mi[n]=i);
// producers/consumers (hub-excluded connectivity), respecting reversibility
const cons={},prod={};
RX.forEach((r,ri)=>{
 r.s.forEach(m=>{if(!HUB.has(m))(cons[m]=cons[m]||[]).push(ri);});
 r.p.forEach(m=>{if(!HUB.has(m))(prod[m]=prod[m]||[]).push(ri);});
 if(r.rev){r.p.forEach(m=>{if(!HUB.has(m))(cons[m]=cons[m]||[]).push(ri);});r.s.forEach(m=>{if(!HUB.has(m))(prod[m]=prod[m]||[]).push(ri);});}
});
// reactions that make / use a metabolite (ALL, for display incl hubs)
const makers={},users={};
RX.forEach((r,ri)=>{r.p.forEach(m=>(makers[m]=makers[m]||[]).push(ri));r.s.forEach(m=>(users[m]=users[m]||[]).push(ri));
 if(r.rev){r.s.forEach(m=>(makers[m]=makers[m]||[]).push(ri));r.p.forEach(m=>(users[m]=users[m]||[]).push(ri));}});
let mode='met',blocked=new Set(),koGenes=new Set();
const q=document.getElementById('q'),list=document.getElementById('list'),main=document.getElementById('main');
function setMode(m){mode=m;document.getElementById('tMet').classList.toggle('on',m=='met');document.getElementById('tEnz').classList.toggle('on',m=='enz');q.placeholder='search '+(m=='met'?'metabolite':'enzyme');search();}
document.getElementById('tMet').onclick=()=>setMode('met');document.getElementById('tEnz').onclick=()=>setMode('enz');
q.addEventListener('input',search);
function search(){const s=q.value.trim().toLowerCase();
 if(mode=='met'){const hits=[];for(let i=0;i<MET.length&&hits.length<200;i++){if(!s||MET[i].toLowerCase().includes(s))hits.push(i);}
  hits.sort((a,b)=>MET[a].length-MET[b].length);
  list.innerHTML=hits.slice(0,150).map(i=>`<div class=item onclick=showMet(${i})><div class=t>${MET[i]}</div><div class=s>made by ${(makers[i]||[]).length} &middot; used by ${(users[i]||[]).length}</div></div>`).join('');
 }else{const genes=Object.keys(G2R).filter(g=>!s||g.toLowerCase().includes(s)).sort();
  list.innerHTML=genes.slice(0,200).map(g=>`<div class=item onclick="showEnz('${g}')"><div class=t>${g}</div><div class=s>${G2R[g].length} reactions</div></div>`).join('');}}
function eqHTML(ri){const r=RX[ri];let eq=r.eq;
 // make metabolite names clickable is heavy; keep eq text, add enzyme chips
 const enz=r.g.length?r.g.map(g=>`<span class="enz ${koGenes.has(g)?'ko':''} ${r.g.length>1?'iso':''}" onclick="showEnz('${g}')">${g}</span>`).join(''):'<span class=lg>spontaneous / transport</span>';
 const dead=blocked.has(ri);
 return `<div class=rxn style="${dead?'opacity:.4':''}"><div class=eq>${dead?'&#10007; ':''}${eq}</div><div class=meta>${r.id} ${r.rev?'&middot; reversible':''} ${r.g.length>1?'&middot; <span style=color:#ffd98f>'+r.g.length+' isozymes (interchangeable)</span>':''}<br>${enz}</div></div>`;}
function showMet(i){main.scrollTop=0;const mk=(makers[i]||[]).filter(ri=>!blocked.has(ri)),us=(users[i]||[]);
 const mkAll=makers[i]||[];const lost=mkAll.length-mk.length;
 main.innerHTML=`<h1>${MET[i]}</h1><div class=lg>a metabolite — every way the cell can make or use it</div>
  ${koGenes.size?`<div class=chip onclick="koGenes.clear();blocked.clear();showMet(${i})">&#10007; clear ${koGenes.size} knockout(s)</div>`:''}
  <h2>Produced by ${mk.length} reaction${mk.length!=1?'s':''} ${lost?`<span class=warn>(${lost} route${lost!=1?'s':''} lost to knockout)</span>`:''} — <span class=lg>each is an alternative way to make it</span></h2>
  ${mkAll.length?mkAll.map(eqHTML).join(''):'<div class=lg>no producing reaction</div>'}
  ${mkAll.length&&mk.length==0?'<div class=warn style=margin:8px>&#9888; with the current knockouts, this metabolite can no longer be produced by any single reaction.</div>':''}
  <h2>Consumed by ${us.length} reaction${us.length!=1?'s':''}</h2>
  ${us.length?us.slice(0,40).map(eqHTML).join(''):'<div class=lg>&mdash;</div>'}`;}
function showEnz(g){setMode('enz');q.value=g;const rs=G2R[g]||[];
 main.scrollTop=0;const isKO=koGenes.has(g);
 main.innerHTML=`<h1>${g}</h1><div class=lg>metabolic enzyme — the reactions it catalyzes, and which other enzymes can substitute</div>
  <div class=chip onclick="toggleKO('${g}')">${isKO?'&#10003; restore '+g:'&#9888; knock out '+g+' (block its reactions)'}</div>
  ${koGenes.size?`<span class=chip onclick="koGenes.clear();blocked.clear();showEnz('${g}')">clear all knockouts</span>`:''}
  <h2>Catalyzes ${rs.length} reaction${rs.length!=1?'s':''}</h2>
  ${rs.map(ri=>{const r=RX[ri];const iso=r.g.filter(x=>x!=g);
    return eqHTML(ri)+(iso.length?`<div class=lg style=margin:-4px 0 8px 6px>&#8618; substitute enzymes for this reaction: ${iso.map(x=>`<span class=enz iso onclick="showEnz('${x}')">${x}</span>`).join('')}</div>`:`<div class=lg style="margin:-4px 0 8px 6px;color:#ff9db0">&#9888; no isozyme — ${g} is the sole enzyme for this reaction</div>`);}).join('')}`;}
function toggleKO(g){if(koGenes.has(g))koGenes.delete(g);else koGenes.add(g);
 blocked=new Set();koGenes.forEach(gn=>{ (G2R[gn]||[]).forEach(ri=>{const r=RX[ri];if(r.g.every(x=>koGenes.has(x)))blocked.add(ri);});});
 showEnz(g);}
window.showMet=showMet;window.showEnz=showEnz;window.toggleKO=toggleKO;
// ---- route finder (connectivity-based, K alternative routes) ----
function findRoutes(srcName,dstName,K){const s=mi[srcName],d=mi[dstName];if(s===undefined||d===undefined)return null;
 const routes=[];let ban=new Set(blocked);
 for(let k=0;k<K;k++){const p=bfs(s,d,ban);if(!p)break;routes.push(p);p.forEach(ri=>ban.add(ri));}
 return routes;}
function bfs(s,d,ban){const q=[[s,[]]],seen=new Set([s]);let h=0;
 while(h<q.length){const[m,path]=q[h++];if(path.length>16)continue;
  for(const ri of (cons[m]||[])){if(ban.has(ri))continue;const r=RX[ri];const outs=r.s.includes(m)?r.p:r.s;
   for(const nm of outs){if(HUB.has(nm)||seen.has(nm))continue;const np=path.concat(ri);if(nm==d)return np;seen.add(nm);q.push([nm,np]);}}}
 return null;}
document.getElementById('main');
function routeUI(){main.innerHTML=`<div id=rf><div class=big>Alternative route finder</div>
  <div class=lg style=margin:4px 0 8px>Find routes from one metabolite to another (connectivity-based). Knock out enzymes (in the Enzymes tab) to test if a substitute route survives.</div>
  <input id=src placeholder="from (e.g. glucose)"> &rarr; <input id=dst placeholder="to (e.g. pyruvate)">
  <span class=chip onclick=runRoutes()>find routes</span></div><div id=rout></div>`;}
function runRoutes(){const s=document.getElementById('src').value.trim(),d=document.getElementById('dst').value.trim();
 const rs=findRoutes(s,d,4);const out=document.getElementById('rout');
 if(rs===null){out.innerHTML='<div class=warn>metabolite not found — try exact name (see Metabolites tab)</div>';return;}
 if(!rs.length){out.innerHTML='<div class=warn>&#9888; no route found'+(blocked.size?' (with current knockouts — the pathway is blocked!)':'')+'</div>';return;}
 out.innerHTML=`<h2>${rs.length} alternative route${rs.length!=1?'s':''} ${s} &rarr; ${d}${blocked.size?' <span class=lg>('+blocked.size+' reactions knocked out)</span>':''}</h2>`+
  rs.map((p,k)=>`<div class=route><div class=lg>route ${k+1} — ${p.length} steps</div>${p.map(ri=>{const r=RX[ri];return `<div class=step>${r.eq} <span class=lg>&middot; ${r.g.slice(0,3).join('/')||'spont'}</span></div>`;}).join('')}</div>`).join('');}
window.runRoutes=runRoutes;
// route finder button in top
const rb=document.createElement('span');rb.className='tab';rb.textContent='Route finder';rb.onclick=routeUI;document.getElementById('top').insertBefore(rb,document.getElementById('q'));
setMode('met');
main.innerHTML='<h1>Metabolic Routes</h1><div class=lg>Search a <b>metabolite</b> to see every reaction that makes or uses it (each producer = an alternative way to make it). Search an <b>enzyme</b> to see its reactions + substitute isozymes, and knock it out to test if a route survives. Or use <b>Route finder</b>.</div><div style=margin-top:14px><span class=chip onclick="q.value=\'pyruvate\';search()">try: pyruvate</span> <span class=chip onclick="showEnz(\'PKM\')">try: PKM (pyruvate kinase)</span> <span class=chip onclick=routeUI()>try: route finder</span></div>';
</script></body></html>"""
HTML=HTML.replace("__DATA__",json.dumps(G,separators=(',',':')))
(OUT/"metabolic_routes.html").write_text(HTML)
print("wrote metabolic_routes.html (%d KB)"%(len(HTML)//1024))
