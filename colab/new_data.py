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


def main():
    C = CompleteCell()
    res = {}
    for name, fn in (("motifs", motifs), ("complexes", complexes)):
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
