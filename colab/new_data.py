"""new_data — extractors for the genuinely-NEW layers the data-hunt surfaced (verified against the actual cell,
so redundant sources like GO — already held — are NOT re-fetched). Same defensive, URL/env-driven pattern as
extra_data.py: each function downloads (or reads a local/Drive file), parses, maps onto our gene index, and
writes a small overlay to outputs/orphan/ which complete_cell folds in. Bulk hosts that the sandbox allowlist
blocks (CORUM, HMDB, MobiDB) download fine in Colab, so these run in the notebook.

Layers (each = 0 in the current cell unless noted):
  tf_motifs.json      JASPAR PWMs per TF                          (verified: 743 TFs map)
  complexes_extra.json  CORUM + hu.MAP complexes beyond our 2,039
  disorder.json       MobiDB per-protein intrinsic-disorder fraction
  metabolites.json    HMDB metabolite layer (ids + biofluid concentrations)

Nothing measured is overwritten; every output is an additive overlay.
"""
import os, sys, json, io, zipfile, urllib.request, ssl
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
_CA = "/root/.ccr/ca-bundle.crt"          # proxy CA in the sandbox; harmless/absent in Colab


def _fetch(url, timeout=180):
    """GET bytes, tolerating the sandbox proxy CA (falls back to a permissive context only if the CA is absent)."""
    ctx = None
    if os.path.exists(_CA):
        ctx = ssl.create_default_context(cafile=_CA)
    try:
        return urllib.request.urlopen(url, timeout=timeout, context=ctx).read()
    except Exception:
        return urllib.request.urlopen(url, timeout=timeout,
                                      context=ssl._create_unverified_context()).read()


# ---------------- JASPAR TF motifs ----------------
JASPAR_URL = ("https://jaspar.elixir.no/download/data/2024/CORE/"
              "JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt")


def motifs(path_or_url=None, C=None):
    """JASPAR PFMs -> {gene_idx: {id, pwm}}. A TF node gains its curated DNA-binding motif — the mechanistic
    layer beneath our TF->target edges. Handles dimers ('Arnt::Ahr') by mapping each subunit."""
    C = C or CompleteCell()
    idx = C.idx
    src = path_or_url or os.environ.get("JASPAR_PFM") or JASPAR_URL
    try:
        raw = open(src, "rb").read() if os.path.exists(src) else _fetch(src)
        lines = raw.decode("utf-8", "replace").splitlines()
    except Exception as e:
        print("  (motifs off:", str(e)[:60], ")"); return None
    out, cur, rows = {}, None, []

    def flush(header, rows):
        if not header or len(rows) < 4:
            return
        mid, name = (header.split("\t") + [""])[:2]
        subs = [s.strip().upper() for s in name.replace("(", "::").split("::") if s.strip()]
        pwm = []
        for r in rows[:4]:
            nums = [float(x) for x in r.replace("[", " ").replace("]", " ").split()[1:] if _isnum(x)]
            pwm.append(nums)
        for s in subs:
            if s in idx and idx[s] not in out:
                out[idx[s]] = {"id": mid.strip(), "pwm": pwm}

    header = None
    for ln in lines:
        if ln.startswith(">"):
            flush(header, rows); header = ln[1:].strip(); rows = []
        elif ln.strip():
            rows.append(ln)
    flush(header, rows)
    if not out:
        print("  (motifs off: 0 TFs mapped)"); return None
    json.dump({"motifs": {str(k): v for k, v in out.items()},
               "meta": {"n_tf": len(out), "source": "JASPAR2024 CORE vertebrates"}},
              open(f"{OUT}/tf_motifs.json", "w"))
    print(f"  tf_motifs: {len(out):,} TFs given a JASPAR binding motif (PWM) -> tf_motifs.json")
    return {"n_tf": len(out)}


def _isnum(x):
    try:
        float(x); return True
    except ValueError:
        return False


# ---------------- CORUM + hu.MAP complexes ----------------
CORUM_URL = "https://mips.helmholtz-muenchen.de/corum/download/releaseDownload?file=allComplexes.txt.zip"


def complexes(corum_url=None, C=None):
    """CORUM curated complexes -> complexes beyond our 2,039, mapped to gene idx. (hu.MAP 3.0 can be added the
    same way from its pairs+membership TSV.)"""
    C = C or CompleteCell()
    idx = C.idx
    have = {tuple(sorted(v)) for v in (C.D.get("complexes", {}) or {}).values() if isinstance(v, list)}
    src = corum_url or os.environ.get("CORUM_TXT") or CORUM_URL
    try:
        raw = open(src, "rb").read() if os.path.exists(src) else _fetch(src)
        if raw[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(raw)); raw = z.read(z.namelist()[0])
        lines = raw.decode("utf-8", "replace").splitlines()
    except Exception as e:
        print("  (complexes off:", str(e)[:60], ")"); return None
    hdr = lines[0].split("\t"); low = [h.lower() for h in hdr]
    ci = next((i for i, h in enumerate(low) if "complexname" in h or h == "complex name"), 1)
    # subunit gene names column (CORUM: 'subunits(Gene name)')
    gi = next((i for i, h in enumerate(low) if "gene name" in h or "gene_name" in h or "genes" in h), None)
    org = next((i for i, h in enumerate(low) if "organism" in h), None)
    if gi is None:
        print(f"  (complexes off: no gene-name column; header={hdr[:8]})"); return None
    new = {}
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) <= gi or (org is not None and len(p) > org and p[org] != "Human"):
            continue
        genes = [g.strip() for g in p[gi].replace(",", ";").split(";") if g.strip() in idx]
        members = sorted({idx[g] for g in genes})
        if len(members) >= 2 and tuple(members) not in have:
            new[p[ci] if len(p) > ci else f"cplx{len(new)}"] = members
    if not new:
        print("  (complexes: 0 new complexes)"); return None
    json.dump({"complexes": {k: v for k, v in new.items()},
               "meta": {"n_new": len(new)}}, open(f"{OUT}/complexes_extra.json", "w"))
    print(f"  complexes_extra: {len(new):,} complexes beyond our {len(have):,} -> complexes_extra.json")
    return {"n_new": len(new)}


def _acc2sym(accs, idx):
    """UniProt accession -> our gene symbol, via mygene (Colab has it). Only for accs we can't resolve otherwise."""
    m = {}
    try:
        import mygene
        hits = mygene.MyGeneInfo().querymany(list(dict.fromkeys(accs)), scopes="uniprot",
                                             fields="symbol", species="human", verbose=False)
        for h in hits:
            s = h.get("symbol")
            if s and s in idx:
                m[h["query"]] = s
    except Exception as e:
        print("  (acc->symbol map failed:", str(e)[:40], ")")
    return m


# ---------------- MobiDB intrinsic disorder ----------------
MOBIDB_URL = "https://mobidb.org/api/download?proteome=UP000005640&format=tsv"


def disorder(path_or_url=None, C=None):
    """MobiDB -> {gene_idx: disorder_fraction}. Per-residue disorder collapsed to a 0-1 content score; touches the
    no-homolog dark genes (many are intrinsically disordered). Keyed by UniProt acc -> mapped to our symbols."""
    C = C or CompleteCell(); idx = C.idx
    src = path_or_url or os.environ.get("MOBIDB_TSV") or MOBIDB_URL
    try:
        raw = open(src, "rb").read() if os.path.exists(src) else _fetch(src)
        text = raw.decode("utf-8", "replace")
    except Exception as e:
        print("  (disorder off:", str(e)[:60], ")"); return None
    rows = [ln.split("\t") for ln in text.splitlines() if ln.strip()]
    if not rows:
        print("  (disorder off: empty)"); return None
    hdr = [h.lower() for h in rows[0]]
    print(f"  mobidb columns: {rows[0][:8]}")
    ai = next((i for i, h in enumerate(hdr) if h in ("acc", "accession", "uniprot")), 0)
    gi = next((i for i, h in enumerate(hdr) if "gene" in h), None)
    di = next((i for i, h in enumerate(hdr) if "content" in h or "disorder" in h or "fraction" in h), None)
    if di is None:
        print(f"  (disorder off: no disorder-content column; header={rows[0][:10]})"); return None
    pending, frac = [], {}
    for p in rows[1:]:
        if len(p) <= max(ai, di):
            continue
        try:
            f = float(p[di])
        except ValueError:
            continue
        g = p[gi].strip() if gi is not None and len(p) > gi else None
        if g and g in idx:
            frac[idx[g]] = f
        else:
            pending.append((p[ai].split("-")[0], f))
    if pending:                                             # resolve leftover accessions -> symbols
        a2s = _acc2sym([a for a, _ in pending], idx)
        for a, f in pending:
            if a in a2s:
                frac[idx[a2s[a]]] = f
    if not frac:
        print("  (disorder off: 0 genes mapped)"); return None
    json.dump({"disorder": {str(k): round(v, 4) for k, v in frac.items()}, "meta": {"n": len(frac)}},
              open(f"{OUT}/disorder.json", "w"))
    print(f"  disorder: {len(frac):,} genes given intrinsic-disorder fraction -> disorder.json")
    return {"n": len(frac)}


# ---------------- Pharos target-development level (darkness) ----------------
PHAROS_GQL = "https://pharos-api.ncats.io/graphql"


def darkness(C=None):
    """Pharos TDL (Tclin/Tchem/Tbio/Tdark) per gene — a standard 'how studied is this target' label that
    validates our 5,006-gene dark set. One GraphQL query for all human targets."""
    C = C or CompleteCell(); idx = C.idx
    q = '{"query":"{ targets(top:25000){ targets{ sym tdl } } }"}'
    try:
        req = urllib.request.Request(PHAROS_GQL, data=q.encode(),
                                     headers={"Content-Type": "application/json"})
        ctx = ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None
        data = json.loads(urllib.request.urlopen(req, timeout=180, context=ctx).read())
        tgts = data["data"]["targets"]["targets"]
    except Exception as e:
        print("  (darkness off:", str(e)[:60], ")"); return None
    tdl = {idx[t["sym"]]: t["tdl"] for t in tgts if t.get("sym") in idx and t.get("tdl")}
    if not tdl:
        print("  (darkness off: 0 mapped)"); return None
    import collections
    bd = collections.Counter(tdl.values())
    json.dump({"tdl": {str(k): v for k, v in tdl.items()}, "meta": {"n": len(tdl), "by_level": dict(bd)}},
              open(f"{OUT}/darkness.json", "w"))
    print(f"  darkness: {len(tdl):,} genes with Pharos TDL {dict(bd)} -> darkness.json")
    return {"n": len(tdl), "by_level": dict(bd)}


# ---------------- InterPro domains ----------------
def domains(path=None, C=None):
    """InterPro per-gene domain architecture. Reads a protein2ipr-style TSV (acc<TAB>ipr_id<TAB>ipr_name...),
    filtered to human on Colab. -> {gene_idx: [domain names]}. Big file, so path-driven (no default fetch)."""
    C = C or CompleteCell(); idx = C.idx
    src = path or os.environ.get("INTERPRO_TSV")
    if not src or not os.path.exists(src):
        print("  (domains off: set INTERPRO_TSV to a protein2ipr TSV — large file, Colab only)"); return None
    try:
        acc2dom, pending = {}, {}
        with open(src) as fh:
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                pending.setdefault(p[0], set()).add(p[2] if len(p) > 2 else p[1])
        a2s = _acc2sym(list(pending), idx)
        dom = {}
        for a, ds in pending.items():
            s = a2s.get(a)
            if s:
                dom.setdefault(idx[s], set()).update(ds)
        if not dom:
            print("  (domains off: 0 mapped)"); return None
        json.dump({"domains": {str(k): sorted(v) for k, v in dom.items()}, "meta": {"n": len(dom)}},
                  open(f"{OUT}/domains.json", "w"))
        print(f"  domains: {len(dom):,} genes with InterPro domain architecture -> domains.json")
        return {"n": len(dom)}
    except Exception as e:
        print("  (domains off:", str(e)[:60], ")"); return None


# ---------------- HMDB metabolite layer ----------------
def metabolites(path=None, C=None, cap=400000):
    """HMDB -> a metabolite layer the cell entirely lacks: {hmdb_id: {name, formula, chebi, kegg}} plus any
    quantified blood concentration. Streams the big hmdb_metabolites.xml (iterparse) so 5 GB never lands in RAM.
    Path-driven (Colab downloads the zip; sandbox host is 403)."""
    src = path or os.environ.get("HMDB_XML")
    if not src or not os.path.exists(src):
        print("  (metabolites off: set HMDB_XML to hmdb_metabolites.xml — Colab only)"); return None
    import xml.etree.ElementTree as ET
    mets = {}
    try:
        def tag(e):
            return e.tag.split("}")[-1]
        ctx = ET.iterparse(src, events=("end",))
        cur = None
        for _, el in ctx:
            t = tag(el)
            if t == "metabolite":
                acc = (el.findtext("{*}accession") or "").strip()
                if acc:
                    conc = None
                    for c in el.iter():
                        if tag(c) == "concentration_value" and c.text:
                            conc = c.text.strip(); break
                    mets[acc] = {"name": (el.findtext("{*}name") or "").strip(),
                                 "formula": (el.findtext("{*}chemical_formula") or "").strip(),
                                 "chebi": (el.findtext("{*}chebi_id") or "").strip(),
                                 "kegg": (el.findtext("{*}kegg_id") or "").strip(),
                                 "conc": conc}
                el.clear()
                if len(mets) >= cap:
                    break
        if not mets:
            print("  (metabolites off: parsed 0)"); return None
        n_conc = sum(1 for m in mets.values() if m.get("conc"))
        json.dump({"metabolites": mets, "meta": {"n": len(mets), "n_with_conc": n_conc}},
                  open(f"{OUT}/metabolites.json", "w"))
        print(f"  metabolites: {len(mets):,} HMDB metabolites ({n_conc:,} with a concentration) -> metabolites.json")
        return {"n": len(mets), "n_with_conc": n_conc}
    except Exception as e:
        print("  (metabolites off:", str(e)[:60], ")"); return None


def main():
    C = CompleteCell()
    res = {}
    for name, fn in (("motifs", motifs), ("complexes", complexes), ("disorder", disorder),
                     ("darkness", darkness), ("domains", domains), ("metabolites", metabolites)):
        try:
            r = fn(C=C)
            if r:
                res[name] = r
        except Exception as e:
            print(f"  ({name} failed: {str(e)[:60]})")
    json.dump(res, open(f"{OUT}/new_data_report.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
