"""STRUCTURE lens — AlphaFold + Foldseek function transfer for dark genes.

When a dark gene has no expression/co-essentiality/interaction signal, its 3D FOLD can still match a
CHARACTERIZED protein — the standard way dark genes actually get function assigned. We run Foldseek
(structural search) with dark-gene AlphaFold structures as query against the characterized proteins'
structures, then transfer the function (pathway/process) of the top structural hits.

Needs: AlphaFold structures (AF-<uniprot>-F1-model_v4.pdb in AF_DIR) + the foldseek binary (Colab).
Maps gene<->UniProt via uniprot_acc_ptm.tsv. Skips gracefully if either is absent.
-> structure_function.json  {dark_gene: {pred, ev, n_hits, top}}
"""
import json, os, subprocess, shutil
from collections import defaultdict, Counter
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
AF_DIR=Path(os.environ.get("AF_DIR","data/external_data/human/alphafold"))

def gene_uniprot():
    """symbol <-> UniProt accession from uniprot_acc_ptm.tsv (accession, gene_primary, ...)."""
    a2g={}; g2a={}
    f=H/"uniprot_acc_ptm.tsv"
    if f.exists():
        import csv
        rd=csv.reader(open(f),delimiter="\t"); next(rd,None)
        for r in rd:
            if len(r)>=2 and r[0] and r[1]:
                sym=r[1].split()[0] if r[1] else ""
                if sym: a2g[r[0]]=sym; g2a.setdefault(sym,r[0])
    return a2g,g2a

def parse_foldseek(m8, a2g, dark, func):
    """Foldseek m8: query target ... bits (cols 0,1, last=bits, col10=evalue). Transfer function of
    characterized structural hits to each dark query."""
    votes=defaultdict(Counter); ev=defaultdict(list)
    for l in open(m8):
        p=l.rstrip("\n").split("\t")
        if len(p)<12: continue
        q=p[0].split("-")[1] if p[0].startswith("AF-") else p[0].split(".")[0]   # AF-<acc>-F1... -> acc
        t=p[1].split("-")[1] if p[1].startswith("AF-") else p[1].split(".")[0]
        qg=a2g.get(q); tg=a2g.get(t)
        try: evalue=float(p[10])
        except: evalue=1.0
        if not qg or not tg or qg==tg or qg not in dark: continue
        if evalue>1e-3: continue                       # confident structural match only
        fn=func.get(tg)
        if fn and fn!="other" and tg not in dark:
            votes[qg][fn]+=1; ev[qg].append(tg)
    out={}
    for g,c in votes.items():
        pred,n=c.most_common(1)[0]
        out[g]=dict(pred=pred, ev=ev[g][:5], n_hits=sum(c.values()),
                    conf="high" if n>=3 else "low", src="structure(Foldseek)")
    return out

def main():
    if not shutil.which("foldseek"):
        print("foldseek not installed -> skipping structure lens (Colab: `conda install -c bioconda foldseek`)"); return
    if not AF_DIR.is_dir() or not any(AF_DIR.glob("*.pdb")):
        print(f"no AlphaFold structures in {AF_DIR} -> skipping (download the AF human proteome on Colab)"); return
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]
    dark=set(g["name"] for g in G if g["dark"])
    func={g["name"]:(g.get("path") or g.get("proc") or "other") for g in G}
    a2g,g2a=gene_uniprot()
    # split structures into dark (query) and characterized (target) by filename accession
    qdir=OUT/"af_query"; tdir=OUT/"af_target"; qdir.mkdir(exist_ok=True); tdir.mkdir(exist_ok=True)
    nq=nt=0
    for pdb in AF_DIR.glob("*.pdb"):
        acc=pdb.name.split("-")[1] if pdb.name.startswith("AF-") else pdb.stem
        g=a2g.get(acc)
        if not g: continue
        (shutil.copy(pdb,qdir) if g in dark else shutil.copy(pdb,tdir))
        nq+= (g in dark); nt+= (g not in dark)
    print(f"structure lens: {nq} dark-gene structures vs {nt} characterized structures")
    if nq==0 or nt==0: print("  not enough structures mapped -> skipping"); return
    m8=OUT/"foldseek_hits.m8"; tmp=OUT/"fs_tmp"
    r=subprocess.run(["foldseek","easy-search",str(qdir),str(tdir),str(m8),str(tmp),
                      "--alignment-type","1","-e","0.001","--max-seqs","20"],capture_output=True,text=True)
    if r.returncode!=0 or not m8.exists():
        print("foldseek failed:",r.stderr[-300:]); return
    out=parse_foldseek(m8,a2g,dark,func)
    json.dump(out,open(OUT/"structure_function.json","w"))
    print(f"structure lens: assigned a structure-based function to {len(out)} dark genes")

if __name__=="__main__":
    main()
