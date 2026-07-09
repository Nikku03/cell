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
import os, sys, json, io, zipfile, tarfile, gzip, shutil, urllib.request, ssl
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


# ---------------- hu.MAP 3.0 + CORUM complexes ----------------
# Our BASE 2,039 complexes ARE CORUM (verified by their curated names), so re-fetching CORUM adds only membership
# variants. hu.MAP 3.0 is the genuinely-orthogonal source: 15,326 complexes derived from >25,000 mass-spec
# experiments, independent of CORUM's literature curation — the real new complex layer.
HUMAP_URL = ("https://humap3.proteincomplexes.org/static/downloads/humap3/"
             "hu.MAP3.0_complexes_wConfidenceScores_total15326_wGenenames_20240922.csv")
CORUM_URLS = [
    "https://mips.helmholtz-muenchen.de/corum/download/allComplexes.txt.zip",
    "http://mips.helmholtz-muenchen.de/corum/download/allComplexes.txt.zip",
    "https://mips.helmholtz-muenchen.de/corum/download/releaseDownload?file=allComplexes.txt.zip",
]


def _humap(idx, have):
    """hu.MAP 3.0 CSV (HuMAP3_ID, ComplexConfidence, Uniprot_ACCs, genenames) -> {name: [idx]}, mass-spec complexes
    not already in `have` (deduped by mapped member-set). Keeps confidence <= HUMAP_MAXCONF (1=Extremely High ..
    4=Moderate; default 4). Columns are header-detected so a column reorder does not silently break the parse."""
    src = os.environ.get("HUMAP_CSV") or HUMAP_URL
    try:
        maxconf = int(os.environ.get("HUMAP_MAXCONF", "4"))
    except ValueError:
        maxconf = 4
    try:
        raw = open(src, "rb").read() if os.path.exists(src) else _fetch(src)
        lines = raw.decode("utf-8", "replace").splitlines()
    except Exception as e:
        print("  (huMAP off:", str(e)[:60], ")"); return {}
    if not lines:
        return {}
    hdr = [h.strip().lower() for h in lines[0].split(",")]
    gi = next((i for i, h in enumerate(hdr) if "genename" in h), 3)      # 'genenames' (space-separated symbols)
    ci = next((i for i, h in enumerate(hdr) if "confidence" in h), 1)
    seen, new = set(have), {}
    for ln in lines[1:]:
        p = ln.split(",")
        if len(p) <= max(gi, ci):
            continue
        try:
            if int(float(p[ci])) > maxconf:
                continue
        except ValueError:
            pass
        members = tuple(sorted(set(idx[g] for g in p[gi].replace('"', " ").split() if g.strip() in idx)))
        if len(members) >= 2 and members not in seen:
            seen.add(members); new[f"huMAP:{p[0].strip()}"] = list(members)
    if new:
        print(f"  huMAP: {len(new):,} mass-spec complexes (conf<={maxconf}) beyond our CORUM base")
    return new


def _corum(idx, have, corum_url=None):
    """CORUM allComplexes -> {name: [idx]} for member-sets not in `have`. Our base already IS CORUM, so this
    surfaces only membership variants / complexes we filtered out. Tries several download URL forms."""
    srcs = [corum_url] if corum_url else ([os.environ["CORUM_TXT"]] if os.environ.get("CORUM_TXT") else CORUM_URLS)
    raw, err = None, ""
    for src in srcs:                                         # first source that yields a real table wins
        try:
            raw = open(src, "rb").read() if os.path.exists(src) else _fetch(src)
            if raw and raw[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(raw)); raw = z.read(z.namelist()[0])
            if raw and b"\t" in raw[:4000]:
                break
            raw = None
        except Exception as e:
            err = str(e)[:60]; raw = None
    if not raw:
        print(f"  (CORUM off: no source reachable — {err})"); return {}
    lines = raw.decode("utf-8", "replace").splitlines()
    hdr = lines[0].split("\t"); low = [h.lower() for h in hdr]
    ci = next((i for i, h in enumerate(low) if "complexname" in h or h == "complex name"), 1)
    gi = next((i for i, h in enumerate(low) if "gene name" in h or "gene_name" in h or "genes" in h), None)
    org = next((i for i, h in enumerate(low) if "organism" in h), None)
    if gi is None:
        print(f"  (CORUM off: no gene-name column; header={hdr[:8]})"); return {}
    seen, new = set(have), {}
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) <= gi or (org is not None and len(p) > org and p[org] != "Human"):
            continue
        genes = [g.strip() for g in p[gi].replace(",", ";").split(";") if g.strip() in idx]
        members = tuple(sorted(set(idx[g] for g in genes)))
        if len(members) >= 2 and members not in seen:
            seen.add(members); new[p[ci] if len(p) > ci else f"CORUM:{len(new)}"] = list(members)
    if new:
        print(f"  CORUM: {len(new):,} complexes beyond base (membership variants — our base already IS CORUM)")
    return new


def complexes(corum_url=None, C=None):
    """Genuinely-new complexes beyond our base (which our 2,039 already ARE CORUM). Primary source is hu.MAP 3.0
    (mass-spec-derived, orthogonal to CORUM curation); CORUM is re-scanned only for membership variants. Members
    mapped to gene idx, deduped by member-set against the base and each other, written to complexes_extra.json."""
    C = C or CompleteCell()
    idx = C.idx
    have = {tuple(sorted(v)) for v in (C.D.get("complexes", {}) or {}).values() if isinstance(v, list)}
    new = {}
    new.update(_humap(idx, have))                            # the real new layer (mass-spec complexes)
    have2 = have | {tuple(sorted(v)) for v in new.values()}
    new.update(_corum(idx, have2, corum_url))               # CORUM variants beyond base + hu.MAP
    if not new:
        print("  (complexes: 0 new beyond base)"); return None
    json.dump({"complexes": new, "meta": {"n_new": len(new), "n_base": len(have)}},
              open(f"{OUT}/complexes_extra.json", "w"))
    print(f"  complexes_extra: {len(new):,} new complexes beyond our {len(have):,} base -> complexes_extra.json")
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
    validates our 5,006-gene dark set. Paginated over all ~20,412 human targets (2,500 per GraphQL call)."""
    C = C or CompleteCell(); idx = C.idx
    ctx = ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None
    tgts = []
    try:                                                    # paginate the INNER targets field (top/skip on the outer
        for skip in range(0, 25000, 2500):                  # query are ignored -> capped at 10); 20,412 total
            q = ('{"query":"{ targets{ targets(top:2500 skip:%d){ sym tdl } } }"}' % skip).encode()
            req = urllib.request.Request(PHAROS_GQL, data=q, headers={"Content-Type": "application/json"})
            batch = json.loads(urllib.request.urlopen(req, timeout=120, context=ctx).read())
            got = batch.get("data", {}).get("targets", {}).get("targets", [])
            if not got:
                break
            tgts.extend(got)
    except Exception as e:
        print("  (darkness: partial —", str(e)[:70], ")")
    if not tgts:
        print("  (darkness off: Pharos returned nothing)"); return None
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
HMDB_URLS = ["https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip",   # ~220k, full (best coverage)
             "https://hmdb.ca/system/downloads/current/serum_metabolites.zip"]  # smaller quantified fallback


def _hmdb_xml():
    """return a path to an HMDB metabolites XML: HMDB_XML if staged, else autofetch+unzip the HMDB download
    (default ON; HMDB_AUTOFETCH=0 disables). The zip holds one big .xml; extracted once, then persist caches the
    small parsed metabolites.json so the multi-GB download never repeats."""
    p = os.environ.get("HMDB_XML")
    if p and os.path.exists(p):
        return p
    if os.environ.get("HMDB_AUTOFETCH", "1") not in ("1", "true", "True"):
        return None
    for url in HMDB_URLS:
        try:
            zpath = _fetch_big(url, os.path.basename(url))
            with zipfile.ZipFile(zpath) as z:
                xmls = [n for n in z.namelist() if n.endswith(".xml")]
                if not xmls:
                    continue
                z.extract(xmls[0], OUT)
                return os.path.join(OUT, xmls[0])
        except Exception as e:
            print("  (hmdb fetch failed:", os.path.basename(url), "-", str(e)[:50], ")")
    return None


def metabolites(path=None, C=None, cap=400000):
    """HMDB -> a metabolite layer the cell entirely lacks: {hmdb_id: {name, formula, chebi, kegg}} plus any
    quantified blood concentration. Streams the big hmdb_metabolites.xml (iterparse) so it never lands in RAM.
    Autofetches the HMDB zip by default (HMDB_XML overrides; HMDB_AUTOFETCH=0 disables); sandbox host is 403."""
    src = path or _hmdb_xml()
    if not src or not os.path.exists(src):
        print("  (metabolites off: no HMDB source — set HMDB_XML or allow HMDB_AUTOFETCH)"); return None
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


def _table(path):
    """read a supp table (tsv/csv/gz/xlsx) into (header_list, row_iterator of lists). pandas on Colab handles xlsx."""
    if path.endswith((".xlsx", ".xls")):
        import pandas as pd
        df = pd.read_csv if False else pd.read_excel(path)
        return list(df.columns.astype(str)), (list(map(str, r)) for r in df.itertuples(index=False))
    op = __import__("gzip").open if path.endswith(".gz") else open
    fh = op(path, "rt")
    head = fh.readline().rstrip("\n")
    sep = "\t" if "\t" in head else ","
    cols = head.split(sep)
    return cols, (ln.rstrip("\n").split(sep) for ln in fh)


def _colidx(cols, *keys):
    low = [c.lower() for c in cols]
    for k in keys:
        for i, c in enumerate(low):
            if k in c:
                return i
    return None


# ---------------- AlphaFold structural confidence ----------------
# AlphaFold human proteome (~5.2 GB tar of per-protein .cif.gz); autofetch fallback if no Drive file is found
AF_HUMAN_TAR = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar"


def _fetch_big(url, dest):
    """stream a GB-scale file straight to disk (never into memory); cached under OUT so it is one-time."""
    path = os.path.join(OUT, dest)
    if os.path.exists(path) and os.path.getsize(path) > 1e8:
        print(f"  (using cached {dest})"); return path
    ctx = ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None
    print(f"  fetching {url.split('/')[-1]} (~5 GB, one-time) ...")
    with urllib.request.urlopen(url, timeout=3600, context=ctx) as r, open(path, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)
    return path


def _cif_plddt(text):
    """mean pLDDT of an AlphaFold model CIF: the global QA metric (fast, exact) else the mean CA B-factor."""
    k = "_ma_qa_metric_global.metric_value"
    i = text.find(k)                                        # fast path: no full-file scan of the atom records
    if i != -1:
        seg = text[i + len(k): i + len(k) + 40].split()
        if seg:
            try:
                return float(seg[0])
            except ValueError:
                pass
    vals = []
    for ln in text.splitlines():
        if ln.startswith("ATOM"):
            p = ln.split()
            if len(p) > 14 and p[3] == "CA":
                try:
                    vals.append(float(p[14]))            # B_iso_or_equiv col = per-residue pLDDT
                except ValueError:
                    pass
    return sum(vals) / len(vals) if vals else None


def _iter_af_cifs(src, cap):
    """yield (basename, cif_text) from an AlphaFold source: a .tar/.tar.gz of .cif(.gz) — streamed, NOT extracted
    (the 5 GB human tar holds ~23k .cif.gz) — or a directory of .cif(.gz). gzip handled transparently."""
    import glob as _g
    if os.path.isfile(src) and src.endswith((".tar", ".tar.gz", ".tgz")):
        mode = "r|gz" if src.endswith((".tar.gz", ".tgz")) else "r|*"
        with tarfile.open(src, mode) as tf:
            n = 0
            for m in tf:
                if not m.isfile() or "model_" not in m.name or not m.name.endswith((".cif", ".cif.gz")):
                    continue
                f = tf.extractfile(m)
                if not f:
                    continue
                raw = f.read()
                if m.name.endswith(".gz"):
                    raw = gzip.decompress(raw)
                yield os.path.basename(m.name), raw.decode("utf-8", "replace")
                n += 1
                if n >= cap:
                    break
    elif os.path.isdir(src):
        files = _g.glob(os.path.join(src, "*model_*.cif*")) or _g.glob(os.path.join(src, "*.cif*"))
        for f in files[:cap]:
            try:
                raw = open(f, "rb").read()
                yield os.path.basename(f), (gzip.decompress(raw) if f.endswith(".gz") else raw).decode("utf-8", "replace")
            except Exception:
                continue


def structure(af_src=None, C=None, cap=40000):
    """Per-gene mean AlphaFold pLDDT ('how confidently foldable'). Reads the AlphaFold human-proteome TAR
    (~5 GB, .cif.gz inside, streamed) or a CIF directory — from AF_DIR/AF_TAR or a Drive file; autofetches the
    EBI human tar when AF_AUTOFETCH is set and nothing local is found. With MobiDB disorder this structurally
    characterises the whole proteome including the dark set."""
    C = C or CompleteCell(); idx = C.idx
    src = af_src or os.environ.get("AF_DIR") or os.environ.get("AF_TAR")
    if (not src or not os.path.exists(src)) and os.environ.get("AF_AUTOFETCH"):
        src = _fetch_big(AF_HUMAN_TAR, "UP000005640_9606_HUMAN_v6.tar")
    if not src or not os.path.exists(src):
        print("  (structure off: set AF_DIR/AF_TAR to an AlphaFold tar or CIF dir, or AF_AUTOFETCH=1 — ~5 GB)")
        return None
    acc_plddt = {}
    for base, text in _iter_af_cifs(src, cap):
        if not base.startswith("AF-"):
            continue
        acc = base.split("-")[1]
        p = _cif_plddt(text)
        if p is not None:
            acc_plddt[acc] = round(p, 1)                 # keyed by UniProt accession
    if not acc_plddt:
        print("  (structure off: parsed 0 CIFs from", src, ")"); return None
    a2s = _acc2sym(list(acc_plddt), idx)                 # UniProt acc -> our gene symbol
    plddt = {idx[a2s[a]]: p for a, p in acc_plddt.items() if a2s.get(a) in idx}
    if not plddt:
        print("  (structure off: 0 mapped to genes)"); return None
    json.dump({"plddt": {str(k): v for k, v in plddt.items()},
               "meta": {"n": len(plddt), "n_structures": len(acc_plddt)}}, open(f"{OUT}/structure.json", "w"))
    print(f"  structure: {len(plddt):,} genes given mean AlphaFold pLDDT (from {len(acc_plddt):,} structures) "
          f"-> structure.json")
    return {"n": len(plddt)}


# ---------------- absolute protein concentration (copies/cell) ----------------
def concentration(path=None, C=None):
    """Absolute copies/cell from a Kuster/OpenCell/proteomic-ruler supp table -> {gene_idx: copies}. The input
    enzyme-constrained flux needs. Path-gated (supp Excel/TSV the user points at)."""
    C = C or CompleteCell(); idx = C.idx
    src = path or os.environ.get("CONC_TSV")
    if not src or not os.path.exists(src):
        print("  (concentration off: set CONC_TSV to a copies/cell supp table)"); return None
    try:
        cols, rows = _table(src)
        gi = _colidx(cols, "gene", "symbol", "genename")
        ai = _colidx(cols, "uniprot", "accession", "protein")
        vi = _colidx(cols, "copies", "copynumber", "copy_number", "abundance", "concentration", "ibaq")
        if vi is None or (gi is None and ai is None):
            print(f"  (concentration off: columns unclear {cols[:8]})"); return None
        conc, pending = {}, []
        for p in rows:
            if len(p) <= vi:
                continue
            try:
                v = float(str(p[vi]).replace(",", ""))
            except ValueError:
                continue
            g = p[gi].strip() if gi is not None and len(p) > gi else None
            if g and g in idx:
                conc[idx[g]] = v
            elif ai is not None and len(p) > ai:
                pending.append((p[ai].split("-")[0].split(";")[0], v))
        if pending:
            a2s = _acc2sym([a for a, _ in pending], idx)
            for a, v in pending:
                if a in a2s:
                    conc[idx[a2s[a]]] = v
        if not conc:
            print("  (concentration off: 0 mapped)"); return None
        json.dump({"copies": {str(k): v for k, v in conc.items()}, "meta": {"n": len(conc)}},
                  open(f"{OUT}/concentration.json", "w"))
        print(f"  concentration: {len(conc):,} genes given absolute copies/cell -> concentration.json")
        return {"n": len(conc)}
    except Exception as e:
        print("  (concentration off:", str(e)[:60], ")"); return None


# ---------------- translation efficiency + turnover ----------------
def translation(te_path=None, hl_path=None, C=None):
    """RiboBase translation efficiency + Mathieson/Schwanhäusser half-lives -> {gene_idx: {te, t_half_h}}."""
    C = C or CompleteCell(); idx = C.idx
    out = {}
    for src, key, names in ((te_path or os.environ.get("TE_TSV"), "te", ("te", "translation_eff", "efficiency")),
                            (hl_path or os.environ.get("HALFLIFE_TSV"), "t_half_h",
                             ("half_life", "halflife", "t_half", "half-life"))):
        if not src or not os.path.exists(src):
            continue
        try:
            cols, rows = _table(src)
            gi = _colidx(cols, "gene", "symbol"); ai = _colidx(cols, "uniprot", "accession")
            vi = _colidx(cols, *names)
            if vi is None:
                print(f"  (translation/{key}: no value column in {cols[:8]})"); continue
            pend = []
            for p in rows:
                if len(p) <= vi:
                    continue
                try:
                    v = float(p[vi])
                except ValueError:
                    continue
                g = p[gi].strip() if gi is not None and len(p) > gi else None
                if g and g in idx:
                    out.setdefault(idx[g], {})[key] = v
                elif ai is not None and len(p) > ai:
                    pend.append((p[ai].split("-")[0], v))
            if pend:
                a2s = _acc2sym([a for a, _ in pend], idx)
                for a, v in pend:
                    if a in a2s:
                        out.setdefault(idx[a2s[a]], {})[key] = v
        except Exception as e:
            print(f"  (translation/{key} off:", str(e)[:50], ")")
    if not out:
        print("  (translation off: set TE_TSV and/or HALFLIFE_TSV)"); return None
    json.dump({"translation": {str(k): v for k, v in out.items()}, "meta": {"n": len(out)}},
              open(f"{OUT}/translation.json", "w"))
    print(f"  translation: {len(out):,} genes given translation-efficiency / half-life -> translation.json")
    return {"n": len(out)}


# ---------------- enhancer -> gene links ----------------
def enhancers(path=None, C=None):
    """ENCODE-rE2G element->gene links -> per-gene enhancer count + summed score. Streamed (13.5M rows).
    Path-gated (RE2G_TSV)."""
    C = C or CompleteCell(); idx = C.idx
    src = path or os.environ.get("RE2G_TSV")
    if not src or not os.path.exists(src):
        print("  (enhancers off: set RE2G_TSV to an ENCODE-rE2G links file — big, Colab only)"); return None
    try:
        cols, rows = _table(src)
        gi = _colidx(cols, "targetgene", "target_gene", "gene", "genesymbol")
        si = _colidx(cols, "score", "e2g", "abc")
        if gi is None:
            print(f"  (enhancers off: no target-gene column {cols[:10]})"); return None
        cnt, sc = {}, {}
        for p in rows:
            if len(p) <= gi:
                continue
            g = p[gi].strip()
            if g in idx:
                cnt[idx[g]] = cnt.get(idx[g], 0) + 1
                if si is not None and len(p) > si:
                    try:
                        sc[idx[g]] = sc.get(idx[g], 0.0) + float(p[si])
                    except ValueError:
                        pass
        if not cnt:
            print("  (enhancers off: 0 mapped)"); return None
        json.dump({"enhancers": {str(k): {"n": cnt[k], "score": round(sc.get(k, 0.0), 2)} for k in cnt},
                   "meta": {"n_genes": len(cnt)}}, open(f"{OUT}/enhancers.json", "w"))
        print(f"  enhancers: {len(cnt):,} genes given enhancer->gene links -> enhancers.json")
        return {"n_genes": len(cnt)}
    except Exception as e:
        print("  (enhancers off:", str(e)[:60], ")"); return None


# ---------------- ChIP-derived TF -> target ----------------
def chip_targets(path=None, C=None):
    """A ChIP-derived TF->target table (ChIP-Atlas / hTFtarget / UniBind-collapsed) -> D['chip_reg']: an
    experimental-binding reg channel kept SEPARATE from our curated reg edges. Path-gated (CHIP_TSV)."""
    C = C or CompleteCell(); idx = C.idx
    src = path or os.environ.get("CHIP_TSV")
    if not src or not os.path.exists(src):
        print("  (chip_targets off: set CHIP_TSV to a TF->target table)"); return None
    try:
        cols, rows = _table(src)
        ti = _colidx(cols, "tf", "source", "regulator", "antigen")
        gi = _colidx(cols, "target", "gene")
        if ti is None or gi is None:
            print(f"  (chip_targets off: columns unclear {cols[:8]})"); return None
        reg = {}
        for p in rows:
            if len(p) <= max(ti, gi):
                continue
            t, g = p[ti].strip(), p[gi].strip()
            if t in idx and g in idx and t != g:
                reg.setdefault(idx[t], set()).add(idx[g])
        if not reg:
            print("  (chip_targets off: 0 mapped)"); return None
        n_edges = sum(len(v) for v in reg.values())
        json.dump({"chip_reg": {str(k): sorted(v) for k, v in reg.items()},
                   "meta": {"n_tf": len(reg), "n_edges": n_edges}}, open(f"{OUT}/chip_reg.json", "w"))
        print(f"  chip_targets: {len(reg):,} TFs, {n_edges:,} ChIP-derived TF->target edges -> chip_reg.json")
        return {"n_tf": len(reg), "n_edges": n_edges}
    except Exception as e:
        print("  (chip_targets off:", str(e)[:60], ")"); return None


# ---------------- extra PPI evidence + negatives ----------------
def ppi_extra(C=None):
    """Orthogonal PPI channels (BioPlex AP-MS, HuRI Y2H) + curated NEGATIVES (Negatome), each path-gated. Kept as
    labeled evidence channels (not merged into the STRING score) plus a hard-negative set for honest training."""
    C = C or CompleteCell(); idx = C.idx

    def pairs(src, a_keys, b_keys):
        if not src or not os.path.exists(src):
            return None
        cols, rows = _table(src)
        ai, bi = _colidx(cols, *a_keys), _colidx(cols, *b_keys)
        if ai is None or bi is None:
            print(f"  (ppi_extra: columns unclear {cols[:8]})"); return None
        out = set()
        for p in rows:
            if len(p) <= max(ai, bi):
                continue
            a, b = p[ai].strip(), p[bi].strip()
            if a in idx and b in idx and a != b:
                out.add((min(idx[a], idx[b]), max(idx[a], idx[b])))
        return out
    chans = {}
    bp = pairs(os.environ.get("BIOPLEX_TSV"), ("symbola", "gene_a", "bait", "symbol1"),
               ("symbolb", "gene_b", "prey", "symbol2"))
    hu = pairs(os.environ.get("HURI_TSV"), ("symbol_a", "gene_a", "ensembl_gene_id_a"),
               ("symbol_b", "gene_b", "ensembl_gene_id_b"))
    neg = pairs(os.environ.get("NEGATOME_TSV"), ("a", "gene_a", "protein_a", "symbola"),
                ("b", "gene_b", "protein_b", "symbolb"))
    if bp:
        chans["bioplex"] = sorted(map(list, bp))
    if hu:
        chans["huri"] = sorted(map(list, hu))
    if neg:
        chans["negatives"] = sorted(map(list, neg))
    if not chans:
        print("  (ppi_extra off: set BIOPLEX_TSV / HURI_TSV / NEGATOME_TSV)"); return None
    meta = {k: len(v) for k, v in chans.items()}
    json.dump({"channels": chans, "meta": meta}, open(f"{OUT}/ppi_extra.json", "w"))
    print(f"  ppi_extra: {meta} -> ppi_extra.json")
    return meta


def main():
    C = CompleteCell()
    res = {}
    for name, fn in (("motifs", motifs), ("complexes", complexes), ("disorder", disorder),
                     ("darkness", darkness), ("domains", domains), ("metabolites", metabolites),
                     ("structure", structure), ("concentration", concentration), ("translation", translation),
                     ("enhancers", enhancers), ("chip_targets", chip_targets), ("ppi_extra", ppi_extra)):
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
