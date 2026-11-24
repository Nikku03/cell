"""ESMCBUNDLE -- the fixed input bundle for the ESM C test (G10; colab/esmc/esmctest.py runs it on a Colab GPU), the
G10 declarations, and recording the results that come back from Colab.

Asked by the user: "we can do it on colab, we have gpu there. so make a notebook to run the test there."
The bundle holds only data that may be shared (each with its source and licence in manifest.json):
  UniProt 2026_04 reviewed human sequences, canonical and isoforms (rest.uniprot.org; CC BY 4.0);
  address tags and Pfam-free gene facts from ADDR_UNIPROT (UniProt; CC BY 4.0);
  rooms by imaging as four classes (HPA subcellular location, CC BY-SA 3.0; OpenCell, CC BY-SA 4.0);
  interactions HPA (CC BY-SA 3.0), OpenCell (CC BY-SA 4.0), Complex Portal (CC0) -- BioPlex and MitoCarta stay local;
  the G9 partner-finder shortlists: for each EXAM protein with a known partner (those three sources), the first 500
  candidates in G9 order with their G9 compatibility score and destination / made-together flags (derived from
  UniProt Pfam, HPA, OpenCell, Complex Portal, DepMap CC BY 4.0, ProCan CC BY 4.0, Nadig 2025 CC BY 4.0).
Genes DEV / EXAM by sha256('cellbrain-c1:' + Ensembl) parity, as in every G test.

Run: python3 -I rem/atlas/esmcbundle.py pull | check | build | declare | record RESULTS.json
"""

from __future__ import annotations
import collections
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


at = _load("addresstest")
gx, cf, pp = at.gx, at.cf, at.pp
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
RAW = os.path.join(HERE, "_cache", "esmc")
COLAB = R("colab", "esmc")
BUNDLE = os.path.join(COLAB, "bundle")
OUT = R("outputs", "esmc")
K = 500
FASTA = os.path.join(RAW, "uniprot_fasta", "raw", "uniprot_human_sp_isoforms.fasta.gz")
FASTA_URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&includeIsoform=true"
             "&query=%28organism_id%3A9606%29+AND+%28reviewed%3Atrue%29")
CLAIMS = ["G10-E1", "G10-E2", "G10-P1", "G10-S1"]
DECLARED_MODEL = "biohub/ESMC-600M"
cbb.SRC["uniprot_seq"] = ("UniProt Knowledgebase, reviewed human sequences with isoforms (rest.uniprot.org)", "https://www.uniprot.org", "CC BY 4.0")
cbb.SRC["esmc"] = ("Biohub ESM C protein language model (Candido et al. 2026 bioRxiv), run on Google Colab", "https://huggingface.co/biohub/ESMC-600M",
                   "MIT + third-party notice")


# ------------------------------------------------------------------------------------------------ pull
def pull():
    os.makedirs(os.path.dirname(FASTA), exist_ok=True)
    hdr = os.path.join(os.path.dirname(FASTA), "headers.txt")
    if not os.path.exists(FASTA):
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(FASTA_URL)) as r, open(FASTA + ".part", "wb") as f:
            open(hdr, "w").write("".join(f"{k}: {v}\n" for k, v in r.headers.items())); f.write(r.read())
        os.replace(FASTA + ".part", FASTA)
    rel = next((x.split(":", 1)[1].strip() for x in open(hdr) if x.lower().startswith("x-uniprot-release:")), None)
    info = dict(url=FASTA_URL, release=rel, sha256=hashlib.sha256(open(FASTA, "rb").read()).hexdigest(), at=cbb.now())
    json.dump(info, open(os.path.join(RAW, "pull.json"), "w"), indent=1); print(info)


def read_fasta(path):
    seqs, cur, buf = {}, None, []
    for line in gzip.open(path, "rt"):
        if line.startswith(">"):
            if cur:
                seqs[cur] = "".join(buf)
            cur = line[1:].split("|")[1]; buf = []
        else:
            buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs


# ------------------------------------------------------------------------------------------------ G9 order, reproduced
def g9_orders(D, U, queries, truth):
    """The G9-P1 full order (same code as addresstest.run_p1) for the given queries: first K candidates, their compatibility
    score s1, destination and made-together flags, and the rank of the first partner in `truth` over all candidates."""
    from scipy.stats import rankdata
    from scipy import sparse
    P, MC, ppi, rooms, aclass = D
    ui = {g: i for i, g in enumerate(U)}; nU = len(U)
    dev = {g for g in U if at.H_(g) == 0}
    dom = {g: P[g]["pfam"] for g in U}; pair_n = collections.Counter(); dom_n = collections.Counter(); nint = 0
    for p in ppi:
        x, y = tuple(p)
        if x in dev and y in dev and dom.get(x) and dom.get(y):
            nint += 1; w = 1.0 / (len(dom[x]) * len(dom[y]))
            for a in dom[x]:
                for b in dom[y]:
                    pair_n[tuple(sorted((a, b)))] += w
            for d_ in set(dom[x]) | set(dom[y]):
                dom_n[d_] += 1
    tot = sum(dom_n.values()); L = {}
    for (a, b), o in pair_n.items():
        e = nint * (dom_n[a] / tot) * (dom_n[b] / tot) * (2 if a != b else 1)
        L[(a, b)] = float(np.log((o + 0.5) / (e + 0.5)))
    dn = sorted({d_ for g in U for d_ in dom[g]}); di_ = {d_: i for i, d_ in enumerate(dn)}
    rr, cc, vv = [], [], []
    for (a, b), v in L.items():
        if v > 0 and a in di_ and b in di_:
            rr += [di_[a], di_[b]]; cc += [di_[b], di_[a]]; vv += [v, v]
    Ld = sparse.csr_matrix((vv, (rr, cc)), shape=(len(dn), len(dn)))
    gi_r, gi_c = [], []
    for g in U:
        for d_ in dom[g]:
            gi_r.append(ui[g]); gi_c.append(di_[d_])
    Gd = sparse.csr_matrix((np.ones(len(gi_r), np.float32), (gi_r, gi_c)), shape=(nU, len(dn)))
    gptr, gidx = Gd.indptr, Gd.indices; has_dom = np.diff(gptr) > 0
    RN = list(at.CLASS_OF_ROOM); CN = ["nuclear", "secretory", "mito", "cyto"]
    RM = np.array([[r in rooms.get(g, ()) for r in RN] for g in U], np.float32)
    CM = np.array([[c in (aclass(g) | {at.CLASS_OF_ROOM[r] for r in rooms.get(g, ())}) for c in CN] for g in U], np.float32)
    S = gx.npz("EXPR_HPA_SCT"); sr = {g: i for i, g in enumerate(S["rows"].astype(str))}
    SX = S["X"]; MADE = np.zeros((nU, SX.shape[1]), bool)
    for g, i in ui.items():
        if g in sr:
            MADE[i] = SX[sr[g]] >= 1.0
    ev = cf.evidences(); out = dict(cand=[], s1=[], flags=[], rank=[])
    for s in range(0, len(queries), 500):
        Q = queries[s:s + 500]; sim, _ = cf.combined_sim(ev, Q, U); sim = np.nan_to_num(sim, nan=-9)
        for qi, q in enumerate(Q):
            i = ui[q]; ds = np.zeros(nU, np.float32)
            qd = [di_[d_] for d_ in dom[q] if d_ in di_]
            if qd:
                M = np.asarray(Ld[qd].max(axis=0).todense()).ravel()
                vals = M[gidx]; red = np.maximum.reduceat(vals, gptr[:-1][has_dom]) if len(vals) else np.array([])
                ds[has_dom] = np.maximum(red, 0)
            s1 = rankdata(ds) + rankdata(sim[qi]); s1[i] = -1e9
            dest = ((RM @ RM[i]) > 0) | ((CM @ CM[i]) > 0) if (RM[i].any() or CM[i].any()) else np.ones(nU, bool)
            mq = MADE[i]; together = (MADE[:, mq].mean(1) >= 0.5) if mq.any() else np.ones(nU, bool)
            full = s1 + 1e6 * dest + 1e3 * (dest & together)
            rk = (-full).argsort(kind="stable"); pos = np.empty(nU, np.int64); pos[rk] = np.arange(nU)
            tr = np.array([ui[t] for t in truth[q] if t in ui and t != q])
            out["rank"].append(int(pos[tr].min()) + 1)
            top = rk[:K]; out["cand"].append(top.astype(np.int16)); out["s1"].append(s1[top].astype(np.float32))
            out["flags"].append((dest[top].astype(np.uint8) | ((dest & together)[top].astype(np.uint8) << 1)))
        print(f"  G9 order: {min(s + 500, len(queries))}/{len(queries)} queries", flush=True)
    return {k: np.array(v) for k, v in out.items()}


def _gz_write(path, text):
    with open(path, "wb") as raw_:
        with gzip.GzipFile(fileobj=raw_, mode="wb", mtime=0) as f:
            f.write(text.encode())


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ build
def build():
    t0 = time.time(); D = at.load_all(); P, MC, ppi, rooms, aclass = D
    pinfo = json.load(open(os.path.join(RAW, "pull.json"))) if os.path.exists(os.path.join(RAW, "pull.json")) else {}
    seqs = read_fasta(FASTA); H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    U = sorted(P); ui = {g: i for i, g in enumerate(U)}
    iso_of = collections.defaultdict(list)
    for acc in seqs:
        if "-" in acc:
            iso_of[acc.split("-")[0]].append(acc)
    prot_rows, G, miss, lenmis = [], {}, 0, 0
    for g in U:
        v = P[g]; acc = v["uniprot"]; s = seqs.get(acc)
        if s is None:
            miss += 1; continue
        lenmis += int(len(s) != v["length"])
        prot_rows.append((acc, g, "canonical", "", s))
        isos = sorted(iso_of.get(acc, []), key=lambda a: int(a.split("-")[1]) if a.split("-")[1].isdigit() else 999)
        for a in isos:
            prot_rows.append((a, g, "isoform", a.split("-")[1], seqs[a]))
        mot = set(v["motifs"])
        G[g] = dict(sym=sym(g), split="dev" if at.H_(g) == 0 else "exam", uniprot=acc, length=len(s),
                    classes=sorted({at.CLASS_OF_ROOM[r] for r in rooms.get(g, ())}),
                    tags=dict(signal=int(bool(v["signal"])), transit=int(bool(v["transit"])), tm=int(v["tm"]), gpi=int(bool(v["gpi"])),
                              nls=int("NLS" in mot), nes=int("NES" in mot), er_ret=int("ER_RETENTION" in mot), pts1=int("PTS1" in mot)),
                    has_var_seq=bool(v["has_var_seq"]), changing=v["isoforms_changing_tag"], isoforms=isos)
    os.makedirs(BUNDLE, exist_ok=True)
    _gz_write(os.path.join(BUNDLE, "proteins.tsv.gz"), "pid\tgene\tkind\tiso\tseq\n" + "".join("\t".join(r) + "\n" for r in prot_rows))
    _gz_write(os.path.join(BUNDLE, "genes.json.gz"), json.dumps(dict(genes=U, g=G), sort_keys=True, separators=(",", ":")))
    pairs = sorted((min(a, b), max(a, b), "+".join(sorted(src))) for a, b, src in ((*tuple(p), s_) for p, s_ in ppi.items()) if a in ui and b in ui)
    _gz_write(os.path.join(BUNDLE, "ppi.tsv.gz"), "a\tb\tsources\n" + "".join(f"{a}\t{b}\t{s_}\n" for a, b, s_ in pairs))
    # truth: the three shareable sources (G9 also used BioPlex, local only)
    truth3 = collections.defaultdict(set)
    for a, b, _ in pairs:
        truth3[a].add(b); truth3[b].add(a)
    queries = [g for g in U if at.H_(g) == 1 and any(t in ui and t != g for t in truth3.get(g, ()))]
    print(f"build: {len(prot_rows)} sequences, {len(G)} genes ({miss} without a sequence, {lenmis} length differs from ADDR_UNIPROT), "
          f"{len(pairs)} pairs, {len(queries)} EXAM queries", flush=True)
    o = g9_orders(D, U, queries, truth3)
    keep = o["rank"] <= K
    np.savez_compressed(os.path.join(BUNDLE, "shortlist.npz"), genes=np.array(U), q_all=np.array([ui[q] for q in queries], np.int32), rank_all=o["rank"].astype(np.int32),
                        q=np.array([ui[q] for q in queries], np.int32)[keep], cand=o["cand"][keep], s1=o["s1"][keep], flags=o["flags"][keep], rank_g9=o["rank"][keep].astype(np.int32))
    files = ["proteins.tsv.gz", "genes.json.gz", "ppi.tsv.gz", "shortlist.npz"]
    man = dict(uniprot_release=pinfo.get("release"), fasta_sha256=pinfo.get("sha256"), shortlist_k=K, declared_model=DECLARED_MODEL,
               counts=dict(sequences=len(prot_rows), canonical=len(G), isoforms=len(prot_rows) - len(G), genes_without_sequence=miss,
                           length_differs=lenmis, pairs=len(pairs), exam_queries=len(queries), exam_queries_partner_in_top_k=int(keep.sum()),
                           genes_with_rooms=sum(1 for v in G.values() if v["classes"])),
               sources={"UniProt 2026_04 sequences, tags": "CC BY 4.0", "HPA subcellular location, interactions": "CC BY-SA 3.0",
                        "OpenCell localization, interactions": "CC BY-SA 4.0", "Complex Portal": "CC0",
                        "G9 shortlists (derived; also DepMap, ProCan, Nadig 2025 CC BY 4.0)": "CC BY-SA 4.0 as a derived work"},
               files={f: sha256(os.path.join(BUNDLE, f)) for f in files})
    json.dump(man, open(os.path.join(COLAB, "manifest.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(man, indent=1), f"\n({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ ledger
def code_sha():
    h = hashlib.sha256(open(os.path.join(COLAB, "esmctest.py"), "rb").read()); h.update(open(os.path.join(COLAB, "manifest.json"), "rb").read())
    return h.hexdigest()[:16]


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    doc = open(os.path.join(COLAB, "esmctest.py")).read().split('"""')[1]
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    model=DECLARED_MODEL, claim=doc.split(" " + cid)[1].split(" G10-")[0].split("NOT:")[0].strip()[:1800], trial=False),
                               sort_keys=True) + "\n")
            print(cid, "declared")


def record(path):
    res = json.load(open(path)); meta = res["meta"]
    dec = {r["id"]: r for r in _read() if r["op"] == "declare" and r["id"] in CLAIMS}
    if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
        raise SystemExit("G10 already recorded")
    for cid in CLAIMS:
        if cid not in dec or dec[cid]["code_sha"] != meta["code_sha"]:
            raise SystemExit(f"{cid}: results came from code {meta['code_sha']}, declared {dec.get(cid, {}).get('code_sha')}")
    if meta.get("trial"):
        raise SystemExit("this is a trial run")
    gating = meta["model"] == DECLARED_MODEL
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "esmc_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "esmctest:G10", "code_sha": meta["code_sha"], "model": meta["model"]}
    facts = [Fact(f"CLAIM-{cid}", "esmc_test", res[cid], t[:10], t, "derived", src, inputs=("colab", meta.get("gpu", "")))
             for cid in CLAIMS]
    cbb.bank().add_many(facts)
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=meta["code_sha"], model=meta["model"], gating=gating,
                                    ran_on=meta.get("gpu"), ran_at=meta.get("finished"), **res[cid]), sort_keys=True, default=float) + "\n")
            print(cid, res[cid]["gate"]["verdict"], "" if gating else "(exploratory model, not the declared test)")


def check():
    """the reproduced G9 order must give G9-P1's own numbers (four sources incl. BioPlex, same queries)."""
    D = at.load_all(); P, MC, ppi, rooms, aclass = D; U = sorted(P); ui = set(U)
    B = gx.npz("PPI_BIOPLEX"); truth = collections.defaultdict(set)
    for p in ppi:
        x, y = tuple(p); truth[x].add(y); truth[y].add(x)
    for a, b in zip(B["a"].astype(str), B["b"].astype(str)):
        truth[a].add(b); truth[b].add(a)
    dev = {g for g in U if at.H_(g) == 0}
    queries = [g for g in U if g not in dev and truth.get(g) and any(t in ui for t in truth[g])]
    o = g9_orders(D, U, queries, truth); g9 = json.load(open(R("outputs", "address", "address_tests.json")))["G9-P1"]
    mine = dict(queries=len(queries), median=float(np.median(o["rank"])), within_10=float((o["rank"] <= 10).mean()), first=float((o["rank"] == 1).mean()))
    theirs = dict(queries=g9["queries"], median=g9["median_tries_to_first_partner"]["full"], within_10=g9["within_10_tries"]["full"], first=g9["first_try_hit"]["full"])
    print("reproduced", mine, "\nG9       ", theirs); assert mine == theirs, "reproduction differs"


if __name__ == "__main__":
    a = sys.argv[1:]
    {"pull": pull, "build": build, "check": check, "declare": declare, "record": lambda: record(a[1])}[a[0]]()
