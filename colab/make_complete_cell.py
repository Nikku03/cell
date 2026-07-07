"""Phase 1 — build the COMPLETE cell HTML: every layer we have, merged, in one self-contained file.

Aggregates genome/genes/proteins, function (curated + ghost-patched literature), compartments, PPI/reg/sig
degree, kinetics (measured/predicted + the corrected kcat), ec-flux capacity + essentiality, 200-cell-type
emask, pathways (curated + predicted), disease, drug-target, dark/ghost status, and the consistency scorecard
— into a summary dashboard + a searchable per-gene explorer. Self-contained (embedded JSON + vanilla JS), no
external deps. -> outputs/orphan/complete_cell.html
"""
import json, os, html

OUT = "outputs/orphan"


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def build():
    D = json.load(open(os.path.join(OUT, "cell_complete.json")))
    genes = D["genes"]; name2i = {g["name"]: i for i, g in enumerate(genes)}
    ppm = D.get("ppm", {}); go = D.get("go", {}); emask = D.get("emask", {})
    drugs = D.get("drugs", {}); darkfn = D.get("darkfn", {})
    ghost = (load("ghost_patch.json") or {}).get("patch", {})
    kin = (load("kinetics_refined.json") or {}).get("kinetics_refined", {})
    kinc = (load("kinetics_refined_corrected.json") or {}).get("kinetics_refined", {})
    sc = load("recovery_scorecard.json") or {}

    # PPI / reg / sig degree per gene
    deg = {"ppi": {}, "reg": {}, "sig": {}}
    for rel in ("ppi", "reg", "sig"):
        for e in D.get(rel, []):
            if len(e) >= 2 and isinstance(e[0], int) and isinstance(e[1], int):
                deg[rel][e[0]] = deg[rel].get(e[0], 0) + 1
                deg[rel][e[1]] = deg[rel].get(e[1], 0) + 1
    popcount = lambda x: bin(int(x)).count("1")

    rows = []
    for i, g in enumerate(genes):
        nm = g["name"]
        k = kin.get(nm) or {}
        kc = k.get("kcat_per_s")
        kcc = (kinc.get(nm) or {}).get("kcat_per_s") if (kinc.get(nm) or {}).get("kcat_original") else None
        lit = ghost.get(nm) or {}
        fn = (lit.get("func") or "")[:160] or (g.get("proc") if g.get("proc") not in (None, "other") else "")
        rows.append([
            nm,                                                    # 0 name
            g.get("chrom", ""),                                    # 1 chr
            g.get("comp", ""),                                     # 2 compartment
            fn,                                                    # 3 function
            deg["ppi"].get(i, 0),                                  # 4 PPI degree
            deg["reg"].get(i, 0),                                  # 5 reg degree
            round(k.get("kcat_per_s"), 2) if kc else None,         # 6 kcat
            k.get("tier", ""),                                     # 7 kcat tier
            round(kcc, 1) if kcc else None,                        # 8 corrected kcat
            1 if g.get("ess") else 0,                              # 9 essential
            1 if g.get("dark") else 0,                             # 10 dark
            1 if lit.get("func") else 0,                           # 11 ghost-patched (function filled)
            g.get("ndis", 0),                                      # 12 n disease
            1 if str(i) in drugs else 0,                           # 13 drug target
            popcount(emask.get(str(i), 0)),                        # 14 n cell types expressed
            round(float(ppm.get(str(i), 0) or 0), 1),              # 15 abundance ppm
            g.get("pubs", 0),                                      # 16 publications
            1 if g.get("tf") else 0,                               # 17 is TF
        ])

    # ---- summary stats across every layer ----
    n = len(genes)
    with_fn = sum(1 for r in rows if r[3])
    stats = {
        "genes/proteins": n,
        "protein-coding coverage": f"{round(100*n/19900)}% of genome",
        "with a function": f"{with_fn} ({round(100*with_fn/n)}%)",
        "dark proteome": f"{D.get('dark_count')} ({sum(1 for r in rows if r[11])} literature-patched)",
        "PPI edges": len(D.get("ppi", [])),
        "regulatory edges": len(D.get("reg", [])),
        "signaling edges": len(D.get("sig", [])),
        "protein complexes": len(D.get("complexes", {})),
        "enzymes with kinetics": len(kin),
        "kcat corrections applied": sum(1 for r in rows if r[8]),
        "metabolic reactions (Human-GEM)": 12931,
        "cell types (emask)": len(D.get("ctnames", [])),
        "pathways (curated)": len(D.get("pathways", {})),
        "disease-linked genes": sum(1 for r in rows if r[12]),
        "drug targets": len({int(x) for x in drugs}),
        "GO-annotated": len(go),
        "scorecard": f"{sc.get('n_pass','?')}/{sc.get('n_total','?')} axes pass",
    }

    cols = ["gene", "chr", "compartment", "function", "PPI", "reg", "kcat", "tier", "kcat*", "ess",
            "dark", "fn+", "dis", "drug", "cellT", "ppm", "pubs", "TF"]
    payload = json.dumps({"cols": cols, "rows": rows, "stats": stats}, separators=(",", ":"))
    htmlstr = TEMPLATE.replace("__PAYLOAD__", payload)
    dst = os.path.join(OUT, "complete_cell.html")
    open(dst, "w").write(htmlstr)
    print(f"wrote {dst}  ({round(os.path.getsize(dst)/1e6,1)} MB)")
    print("layers:", " · ".join(f"{k}={v}" for k, v in list(stats.items())[:6]))
    return dst


TEMPLATE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Complete Human Cell</title>
<style>
:root{--bg:#0b0e14;--panel:#141926;--ink:#e6edf3;--mut:#8b97a8;--line:#232b3a;--accent:#4c8dff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fa;--panel:#fff;--ink:#1f2328;--mut:#656d76;--line:#d0d7de;--accent:#0969da}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:20px}
.sub{color:var(--mut);font-size:13px;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;padding:18px 24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .v{font-size:20px;font-weight:650}.card .k{color:var(--mut);font-size:12px;margin-top:2px}
.bar{padding:0 24px 14px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input,select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13px}
input{min-width:240px}.wrap{padding:0 24px 40px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--bg);cursor:pointer;color:var(--mut);font-weight:600;user-select:none}
td.fn{white-space:normal;max-width:360px;color:var(--mut)}
tr:hover td{background:var(--panel)}
.tag{display:inline-block;padding:1px 6px;border-radius:6px;font-size:11px;font-weight:600}
.d{background:rgba(210,153,34,.15);color:var(--warn)}.e{background:rgba(248,81,73,.15);color:var(--bad)}
.g{background:rgba(63,185,80,.15);color:var(--good)}.a{background:rgba(76,141,255,.15);color:var(--accent)}
.cnt{color:var(--mut);font-size:12px}
</style></head><body>
<header><h1>Complete Human Cell <span class=cnt id=ver>· v1 (all measured + predicted layers)</span></h1>
<div class=sub>genome · proteins · PPI/regulatory/signaling · kinetics (measured+predicted+corrected) · flux · 200 cell types · pathways · disease · drugs · dark proteome — every layer, one file.</div></header>
<div class=grid id=cards></div>
<div class=bar>
<input id=q placeholder="search gene / function (e.g. GATA1, kinase, transporter)…" autocomplete=off>
<select id=f><option value="">all genes</option><option value=dark>dark proteome</option><option value=fnpatch>ghost-patched function</option><option value=ess>essential</option><option value=enz>has kinetics</option><option value=corr>kcat corrected</option><option value=drug>drug target</option><option value=tf>transcription factor</option><option value=dis>disease-linked</option></select>
<span class=cnt id=count></span></div>
<div class=wrap><table><thead id=head></thead><tbody id=body></tbody></table></div>
<script>
const DATA=__PAYLOAD__;const {cols,rows,stats}=DATA;
const cards=document.getElementById('cards');
for(const [k,v] of Object.entries(stats)){const d=document.createElement('div');d.className='card';d.innerHTML=`<div class=v>${v}</div><div class=k>${k}</div>`;cards.appendChild(d);}
const head=document.getElementById('head');head.innerHTML='<tr>'+cols.map((c,i)=>`<th data-i=${i}>${c}</th>`).join('')+'</tr>';
const body=document.getElementById('body'),count=document.getElementById('count');
const q=document.getElementById('q'),f=document.getElementById('f');
let sortI=4,sortDir=-1,view=rows;
function tag(r){let t='';if(r[10])t+='<span class="tag d">dark</span> ';if(r[11])t+='<span class="tag g">fn+</span> ';if(r[9])t+='<span class="tag e">ess</span> ';if(r[13])t+='<span class="tag a">drug</span> ';return t;}
function cell(r,i){const v=r[i];if(v===null||v===''){return '<span class=cnt>·</span>';}
 if(i===3)return `<td class=fn>${(''+v).replace(/</g,'&lt;')}</td>`;
 if(i===10||i===11||i===9||i===13||i===17)return v? '✓':'<span class=cnt>·</span>';
 return v;}
function render(){
 view.sort((a,b)=>{let x=a[sortI],y=b[sortI];if(x===null)x=-1;if(y===null)y=-1;if(typeof x==='string')return sortDir*(''+x).localeCompare(''+y);return sortDir*(x-y);});
 const rw=view.slice(0,600).map(r=>'<tr>'+cols.map((c,i)=>{
   if(i===0)return `<td><b>${r[0]}</b> ${tag(r)}</td>`;
   if(i===3)return cell(r,3);
   const v=cell(r,i);return `<td>${v}</td>`;}).join('')+'</tr>').join('');
 body.innerHTML=rw;count.textContent=`${view.length.toLocaleString()} genes${view.length>600?' (top 600 shown — refine search)':''}`;}
function apply(){const s=q.value.trim().toLowerCase(),ff=f.value;
 view=rows.filter(r=>{
   if(s&&!( (''+r[0]).toLowerCase().includes(s) || (''+r[3]).toLowerCase().includes(s) ))return false;
   if(ff==='dark'&&!r[10])return false;if(ff==='fnpatch'&&!r[11])return false;if(ff==='ess'&&!r[9])return false;
   if(ff==='enz'&&r[6]===null)return false;if(ff==='corr'&&!r[8])return false;if(ff==='drug'&&!r[13])return false;
   if(ff==='tf'&&!r[17])return false;if(ff==='dis'&&!r[12])return false;return true;});
 render();}
head.onclick=e=>{const i=+e.target.dataset.i;if(isNaN(i))return;if(sortI===i)sortDir*=-1;else{sortI=i;sortDir=-1;}apply();};
q.oninput=apply;f.onchange=apply;apply();
</script></body></html>"""


if __name__ == "__main__":
    build()
