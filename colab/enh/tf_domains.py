"""What each transcription factor brings to the DNA: which groove it reads, and the charge and bulk
of the domain that reads it.

WHY A SEPARATE MODULE, AND WHAT CORRECTION IT ENCODES. The plan this serves asked to "take the
major groove, eliminating the minor groove" as a per-site filter. That is not how groove readout
works, and the module is built the way it is because of that correction. Which groove a protein
reads is a property of its DNA-BINDING DOMAIN FAMILY and is the same at every site it visits --
zinc fingers and helix-turn-helix insert a recognition helix into the major groove, TBP and the
HMG box splay the minor groove open. Every instance of a motif presents both grooves identically.
So groove is not something to filter sites by; it is something to CONDITION on, because it decides
WHICH shape variable at a site the protein can feel at all. A minor-groove reader is the one for
which minor groove width and minor-groove electrostatic potential are the relevant quantities;
for a major-groove reader they are largely irrelevant and the major groove width is not.

That is the whole reason this file exists next to the pentamer shape table: the shape table says
what the DNA offers, and this says what the protein can accept.

WHERE THE DOMAIN COMES FROM. JASPAR's REST API gives, for each matrix, the structural CLASS and
FAMILY of the factor and its UniProt accession. UniProt gives the sequence and the domain
boundaries. The domain is taken in a fixed order of preference, and which route was used is
recorded per factor rather than averaged away:

    1  a DNA_BIND feature, when UniProt annotates one
    2  the union of ZN_FING features, for the C2H2 and GATA-type factors, spanning first to last
    3  a DOMAIN feature whose description names a known DNA-binding fold (bZIP, bHLH, HMG box,
       homeobox, Fork-head, ETS, T-box, MADS, RHD, IRF, STAT, Runt, p53, SAND, TEA, CUT, and the
       nuclear-receptor DBD)
    4  failing all three, the most positively charged 60-residue window in the protein. This is a
       fallback, not an annotation, and it is labelled as such in the output so a caller can drop
       those factors. It is not arbitrary: a DNA-binding domain is in general the most basic
       stretch of a transcription factor, because it has to sit against a phosphate backbone.

WHAT IS COMPUTED, AND WHY EACH ONE. The point is complementarity to a site, so every quantity has a
counterpart in the shape table:

    net_charge, charge_density    against the minor-groove electrostatic potential. An
                                  electronegative groove attracts a basic domain; the pairing is
                                  the product, not either one alone.
    arg_frac                      separately from total charge, because arginine is the residue
                                  that inserts into a narrow minor groove and reads it by
                                  electrostatics (Rohs et al., Nature 2009); lysine does not do
                                  this nearly as well.
    mean_volume, bulky_frac       against groove WIDTH. This is the steric term: a domain of large
                                  residues cannot enter a groove that is not wide enough for it.
                                  Residue volumes are Zamyatnin's (1972) tabulated values.
    length                        how much of the groove the domain spans, which decides whether a
                                  6 bp core is the whole footprint or a third of it.

NOTHING HERE IS SCORED. This module produces a per-factor property table and stops. Whether any of
it predicts anything is the loop's question, and it is gated there.

Output: colab/data/tf_domains.json
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "tf_domains.json"

JASPAR = "https://jaspar.elixir.no/api/v1/matrix/"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"

# Zamyatnin (1972) residue volumes, cubic Angstrom
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2,
           I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8,
           Y=193.6, V=140.0)
POS, NEG = set("KR"), set("DE")
BULKY = set("FWYLIM")

# family-level groove readout. The assignment is by JASPAR structural class, and it is a property
# of the fold, not of the site.
MINOR = ["beta-scaffold factors with minor groove contacts", "high-mobility group",
         "hmg", "tata-binding"]
BOTH = ["zinc-coordinating dna-binding domains", "c2h2 zinc finger"]

DBD_WORDS = ["bzip", "bhlh", "hmg box", "homeobox", "fork-head", "forkhead", "ets", "t-box",
             "mads-box", "mads", "rel", "irf", "stat", "runt", "p53", "sand", "tea", "cut",
             "nuclear receptor", "myb", "arid", "sam", "grh", "rhd", "paired", "pou-specific",
             "hsf", "cp2", "gcm", "ap-2", "ndt80", "sox", "tcp", "wrky"]


def get(url, tries=4, timeout=60):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cellos", "accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def jaspar_meta(ids, report=print):
    """matrix id -> class, family, uniprot. One request per matrix; JASPAR has no batch endpoint."""
    meta, fail = {}, []
    for n, mid in enumerate(ids, 1):
        try:
            d = get(JASPAR + mid + "/")
        except Exception:
            fail.append(mid)
            continue
        meta[mid] = dict(name=d.get("name"),
                         cls=(d.get("class") or [None])[0],
                         family=(d.get("family") or [None])[0],
                         uniprot=(d.get("uniprot_ids") or [None])[0])
        if n % 100 == 0:
            report(f"    JASPAR {n}/{len(ids)}")
    report(f"    JASPAR: {len(meta)}/{len(ids)} matrices annotated, {len(fail)} failed")
    return meta, fail


def uniprot_batch(accs, report=print, chunk=80):
    """accession -> sequence + the domain features we might use. Batched to keep the request count
    down; a chunk that fails is retried alone so one bad accession cannot lose eighty."""
    out = {}
    accs = sorted(set(a for a in accs if a))
    for i in range(0, len(accs), chunk):
        part = accs[i:i + chunk]
        q = "+OR+".join("accession:" + a for a in part)
        url = (f"{UNIPROT}?query={q}&format=json&size={len(part)}"
               "&fields=accession,sequence,ft_dna_bind,ft_zn_fing,ft_domain")
        try:
            d = get(url)
        except Exception:
            report(f"    UniProt chunk {i}-{i+len(part)} failed, retrying singly")
            d = {"results": []}
            for a in part:
                try:
                    d["results"] += get(f"{UNIPROT}?query=accession:{a}&format=json&size=1"
                                        "&fields=accession,sequence,ft_dna_bind,ft_zn_fing,ft_domain"
                                        )["results"]
                except Exception:
                    pass
        for e in d.get("results", []):
            out[e["primaryAccession"]] = dict(seq=e["sequence"]["value"],
                                              feats=[(f["type"], f["location"]["start"]["value"],
                                                      f["location"]["end"]["value"],
                                                      str(f.get("description", "")))
                                                     for f in e.get("features", [])])
        report(f"    UniProt {min(i + chunk, len(accs))}/{len(accs)}")
    return out


def pick_domain(rec):
    """(start, end, route) 1-based inclusive. See the docstring for the order of preference."""
    seq, feats = rec["seq"], rec["feats"]
    dna = [f for f in feats if f[0] == "DNA binding"]
    if dna:
        f = max(dna, key=lambda x: x[2] - x[1])
        return f[1], f[2], "DNA_BIND"
    zf = [f for f in feats if f[0] == "Zinc finger"]
    if zf:
        return min(f[1] for f in zf), max(f[2] for f in zf), "ZN_FING"
    dom = [f for f in feats if f[0] == "Domain"
           and any(w in f[3].lower() for w in DBD_WORDS)]
    if dom:
        f = max(dom, key=lambda x: x[2] - x[1])
        return f[1], f[2], "DOMAIN"
    # fallback: the most positively charged 60-residue window
    W = min(60, len(seq))
    best, bi = -1e9, 0
    for i in range(len(seq) - W + 1):
        w = seq[i:i + W]
        c = sum(ch in POS for ch in w) - sum(ch in NEG for ch in w)
        if c > best:
            best, bi = c, i
    return bi + 1, bi + W, "BASIC_WINDOW"


def props(seq):
    s = [c for c in seq if c in VOL]
    if not s:
        return None
    n = len(s)
    net = sum(c in POS for c in s) - sum(c in NEG for c in s)
    return dict(length=n,
                net_charge=float(net),
                charge_density=float(net) / n,
                arg_frac=sum(c == "R" for c in s) / n,
                lys_frac=sum(c == "K" for c in s) / n,
                bulky_frac=sum(c in BULKY for c in s) / n,
                mean_volume=sum(VOL[c] for c in s) / n)


def groove(cls):
    c = (cls or "").lower()
    if any(w in c for w in MINOR):
        return "minor"
    if any(w in c for w in BOTH):
        return "both"
    return "major"


def build(report=print):
    raw = json.load(open(OUT_DIR / "tf_motifs.json"))["motifs"]
    ids = sorted({rec["id"] for rec in raw.values()})
    report(f"  {len(raw)} genes carry {len(ids)} distinct JASPAR matrices")
    meta, fail = jaspar_meta(ids, report)
    up = uniprot_batch([m["uniprot"] for m in meta.values()], report)
    report(f"    UniProt: {len(up)} accessions resolved")

    rows, routes, grooves = {}, {}, {}
    for mid, m in meta.items():
        g = groove(m["cls"])
        grooves[g] = grooves.get(g, 0) + 1
        rec = up.get(m["uniprot"])
        r = dict(name=m["name"], cls=m["cls"], family=m["family"],
                 uniprot=m["uniprot"], groove=g, route=None)
        if rec:
            a, b, route = pick_domain(rec)
            p = props(rec["seq"][a - 1:b])
            if p:
                r.update(p)
                r["route"] = route
                r["protein_length"] = len(rec["seq"])
                routes[route] = routes.get(route, 0) + 1
        rows[mid] = r

    have = sum(1 for r in rows.values() if r["route"])
    report(f"  domain resolved for {have}/{len(rows)} matrices")
    for k, v in sorted(routes.items(), key=lambda x: -x[1]):
        report(f"    route {k:14} {v:4d}")
    for k, v in sorted(grooves.items(), key=lambda x: -x[1]):
        report(f"    groove {k:14} {v:4d} matrices")
    named = [r for r in rows.values() if r["route"]]
    if named:
        import statistics as st
        for k in ("length", "net_charge", "charge_density", "arg_frac", "mean_volume"):
            v = [r[k] for r in named]
            report(f"    {k:16} median {st.median(v):8.3f}  "
                   f"range [{min(v):8.3f}, {max(v):8.3f}]")

    DATA.mkdir(parents=True, exist_ok=True)
    json.dump({"matrices": rows, "n": len(rows), "resolved": have,
               "jaspar_failed": fail, "routes": routes, "grooves": grooves},
              open(OUT, "w"), indent=1)
    report(f"  -> {OUT}")


def load():
    if not OUT.exists():
        raise SystemExit(f"{OUT} missing -- run `python colab/enh/tf_domains.py` first")
    return json.load(open(OUT))["matrices"]


if __name__ == "__main__":
    print("=" * 100)
    print("TF DNA-BINDING DOMAINS: which groove the fold reads, and the charge and bulk it reads with")
    print("=" * 100)
    build()
