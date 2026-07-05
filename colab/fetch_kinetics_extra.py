"""FETCH extra MEASURED human kinetics (kcat/Km/Ki) -> augment the measured anchor set.

We already anchor on our own BRENDA/SABIO assembly (kinetics_measured.tsv, ~388 genes). Two commercial-safe
sources add more MEASURED human values (facts, not predictions):
  - CatPred-DB (MIT): processed measured tables (kcat/km/ki), keyed by UniProt+EC+taxonomy. Human = taxid 9606.
    NOTE: CatPred was TRAINED on these, so genes anchored ONLY from CatPred-DB are marked source=CatPred-DB and
    refine_kinetics EXCLUDES them from the held-out CatPred validation (else the fold-error would leak-optimistic).
  - UniProt (CC-BY): the free-text 'Kinetics' annotation, independently curated -> a leak-safer measured set.

-> kinetics_measured_extra.tsv (gene kcat km ph temp source in_vitro)  [merged by compute_kinetics.load_anchors]
   ki_measured.tsv            (gene ki_uM ec source n)                  [new measured inhibition-constant layer]
Skips gracefully if sources are unreachable. Values are facts; we do not redistribute the source files.
"""
import os, csv, io, re, math, urllib.request
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
CATPRED_BASE=os.environ.get("CATPREDDB_BASE",
    "https://raw.githubusercontent.com/maranasgroup/CatPred-DB/main/datasets/processed")
FILES={"kcat":"kcat_max_wt_singleSeqs_wpdbs.csv","km":"km_mean_wt_singleSeqs_wpdbs.csv",
       "ki":"ki_mean_wt_singleSeqs_wpdbs.csv"}

def _fetch(name):
    f=H/name
    if f.exists() and f.stat().st_size>10000: return f
    try:
        print(f"  fetching {name} ..."); urllib.request.urlretrieve(f"{CATPRED_BASE}/{name}", f); return f
    except Exception as e:
        print(f"  fetch failed ({name}): {repr(e)[:90]}"); return None

def acc2gene():
    """UniProt accession -> gene symbol, from uniprot_acc_ptm.tsv (col0 accession, col1 gene names)."""
    m={}; f=H/"uniprot_acc_ptm.tsv"
    if f.exists():
        rd=csv.reader(open(f),delimiter="\t"); next(rd,None)
        for r in rd:
            if len(r)>=2 and r[0] and r[1]: m.setdefault(r[0].strip(), r[1].split()[0])
    return m

def _median(xs): xs=sorted(xs); n=len(xs); return xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])

def load_catpreddb(param, a2g):
    """param CSV -> gene -> list of measured values (human only). kcat in 1/s, km/ki in mM."""
    f=_fetch(FILES[param])
    if not f: return {}
    per=defaultdict(list)
    with open(f, newline="") as fh:
        rd=csv.DictReader(fh)
        for r in rd:
            if (r.get("taxonomy_id") or "").strip() not in ("9606","9606.0"): continue
            g=a2g.get((r.get("uniprot") or "").strip())
            if not g: continue
            try: v=float(r.get("value"))
            except (TypeError,ValueError): continue
            if v<=0 or not math.isfinite(v): continue
            per[g].append((v, r.get("ph") or "", r.get("temperature") or "", (r.get("ec") or "").split(";")[0]))
    return per

def parse_uniprot_kinetics():
    """UniProt human reviewed 'Kinetics' free text -> gene -> (kcat_1/s?, km_mM?). Best-effort regex.
    Independently curated (CC-BY) so leak-safer than CatPred-DB for validating CatPred."""
    f=H/"uniprot_kinetics.tsv"
    if not (f.exists() and f.stat().st_size>1000):
        url=("https://rest.uniprot.org/uniprotkb/stream?query=(organism_id:9606)+AND+(reviewed:true)"
             "&fields=accession,gene_names,ec,kinetics&format=tsv")
        try:
            print("  fetching UniProt kinetics annotation ..."); urllib.request.urlretrieve(url, f)
        except Exception as e:
            print("  UniProt kinetics fetch failed:",repr(e)[:90]); return {},{}
    UNIT={"m":1e3,"mm":1.0,"um":1e-3,"µm":1e-3,"nm":1e-6}      # -> mM
    kcat={}; km={}
    for r in csv.DictReader(open(f), delimiter="\t"):
        g=(r.get("Gene Names") or "").split()
        if not g: continue
        g=g[0]; txt=r.get("Kinetics") or ""
        kms=[]
        for m in re.finditer(r"KM=([\d.]+)\s*(mM|uM|µM|nM|M)\b", txt):
            try: kms.append(float(m.group(1))*UNIT.get(m.group(2).lower(),1.0))
            except ValueError: pass
        if kms: km.setdefault(g, _median(kms))
        kc=[]
        for m in re.finditer(r"[Kk]cat[^.;]{0,40}?([\d.]+)\s*(?:sec|s)\(-1\)", txt):
            try: kc.append(float(m.group(1)))
            except ValueError: pass
        if kc: kcat.setdefault(g, _median(kc))
    return kcat, km

def main():
    OUT.mkdir(parents=True, exist_ok=True); H.mkdir(parents=True, exist_ok=True)
    a2g=acc2gene()
    if not a2g:
        print("  uniprot_acc_ptm.tsv absent -> cannot map CatPred-DB UniProt->gene; extra kinetics skipped");
    cp_kcat=load_catpreddb("kcat", a2g) if a2g else {}
    cp_km  =load_catpreddb("km",   a2g) if a2g else {}
    cp_ki  =load_catpreddb("ki",   a2g) if a2g else {}
    up_kcat, up_km = parse_uniprot_kinetics()

    # merged measured kcat/km anchors (median per gene). Source tag matters: CatPred-DB genes are excluded from
    # the CatPred held-out validation (CatPred trained on them); UniProt genes are leak-safer.
    genes=set(cp_kcat)|set(up_kcat)|set(cp_km)|set(up_km)
    rows=[]
    for g in sorted(genes):
        kc = up_kcat.get(g)                                   # prefer UniProt (independent) then CatPred-DB
        src="UniProt"
        if kc is None and g in cp_kcat: kc=_median([v[0] for v in cp_kcat[g]]); src="CatPred-DB"
        km = up_km.get(g) or (_median([v[0] for v in cp_km[g]]) if g in cp_km else None)
        if kc is None and km is None: continue
        ph=temp=""
        if src=="CatPred-DB" and cp_kcat.get(g):
            _,ph,temp,_=cp_kcat[g][0]
        rows.append((g, kc if kc else "", km if km else "", ph, temp, src, 1))
    with open(OUT/"kinetics_measured_extra.tsv","w") as o:
        o.write("gene\tkcat\tkm\tph\ttemp\tsource\tin_vitro\n")
        for r in rows: o.write("\t".join(str(x) for x in r)+"\n")

    # measured Ki (NEW layer): median per gene, mM -> uM
    with open(OUT/"ki_measured.tsv","w") as o:
        o.write("gene\tki_uM\tec\tsource\tn\n")
        nki=0
        for g in sorted(cp_ki):
            vals=[v[0] for v in cp_ki[g]]; ec=cp_ki[g][0][3]
            o.write(f"{g}\t{round(_median(vals)*1000.0,4)}\t{ec}\tCatPred-DB\t{len(vals)}\n"); nki+=1

    n_up=sum(1 for r in rows if r[5]=="UniProt"); n_cp=sum(1 for r in rows if r[5]=="CatPred-DB")
    print(f"extra kinetics: {len(rows)} measured-kcat/Km genes -> kinetics_measured_extra.tsv "
          f"({n_up} UniProt CC-BY [validation-safe], {n_cp} CatPred-DB MIT [excluded from CatPred validation]) | "
          f"{nki} measured-Ki genes -> ki_measured.tsv (new inhibition-constant layer)")

if __name__=="__main__":
    main()
