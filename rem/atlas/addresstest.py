"""ADDRESSTEST -- protein address labels, RNA that travels with its protein, the user's common-sense partner finder, and
RNA versions (isoforms) that change a protein's address.

Asked by the user: add address-label data and run the tests; find partners the common-sense way -- first by which genes
make a protein that can fit (compatibility), then by where each is delivered (address tag), test, and if it fails try
the next candidate; and factor in RNA maturation (exons and introns, which proteins they make, when, what governs them).
Data (cellbrain bank): ADDR_UNIPROT (signal / transit peptides, transmembrane segments, NLS, ER-retention and peroxisome
motifs, GPI anchors, Pfam domains, isoforms whose alternative sequence touches the tag), ADDR_MITOCARTA,
RNA_FRACTIONS_ENCODE (K562 and HepG2 cytosolic / insoluble / membrane / nuclear fraction RNA), ISOFORM_HPA_TISSUE,
EXPR_HPA_SCT, rooms (HPA + OpenCell, 11 rooms) and interactions (HPA, OpenCell, Complex Portal) as in G4, BioPlex.
Genes DEV / EXAM by sha256('cellbrain-c1:' + Ensembl) parity.
ROOM CLASSES: nuclear (nucleoplasm, nucleolus, nuclear membrane), secretory (ER, Golgi, vesicles, plasma membrane),
mitochondria, cytoplasm (cytosol, cytoskeleton, centrosome). ADDRESS CLASSES from tags: secretory = signal peptide or
transmembrane or GPI; mitochondria = mitochondrial transit peptide or MitoCarta; nuclear = NLS.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G9-*):
 G9-A1  THE ADDRESS TAG PREDICTS WHERE THE PROTEIN IS. Rooms measured by imaging (HPA, OpenCell), tags read from the
        sequence (UniProt): signal peptide -> secretory rooms, mitochondrial transit peptide -> mitochondria, NLS ->
        nuclear rooms. CLAIM, for each of the three: Fisher odds ratio >= 3, p < 0.05, and Monte Carlo p < 0.05 (tags
        shuffled between proteins, 1,000 draws). ER-retention -> ER and peroxisome signal -> vesicles reported.
 G9-A2  THE RNA OF AN ER-MADE PROTEIN GOES TO THE ER. Membrane enrichment of an mRNA = mean log2((membrane TPM + 1) /
        (cytosolic TPM + 1)) over replicates (genes with TPM >= 5 in either). CLAIM: AUROC (signal peptide or
        transmembrane vs other proteins) >= 0.70 in K562 and in HepG2, Monte Carlo p < 0.05 (labels shuffled, 1,000).
 G9-A3  MITOCHONDRIAL PROTEINS' RNA. Same membrane enrichment, MitoCarta proteins without ER tags vs proteins with no
        ER or mitochondrial tag: AUROC >= 0.60 in both lines, Monte Carlo p < 0.05.
 G9-P1  THE COMMON-SENSE PARTNER FINDER. For each EXAM gene with known partners (HPA, OpenCell, Complex Portal, BioPlex;
        hidden from the finder), candidates are every other protein, ordered in three steps:
          1 COMPATIBILITY: rank of the best domain-pair score (log-odds of Pfam domain pairs among interacting DEV-DEV
            pairs) plus rank of the similarity of the two genes' behaviour (G8-U1 evidence: DepMap, ProCan, HPA RNA,
            K562 knockdowns);
          2 DESTINATION: candidates sharing a room or an address class with the query move ahead (order kept inside);
          3 MADE TOGETHER: among those, candidates made (nCPM >= 1) in >= 50% of the HPA cell types where the query
            is made move ahead.
        "Test, else try the next" = walk down the list until the first known partner. CLAIM (both needed): the full
        order is BETTER than compatibility alone (per query gain = log2(rank of first partner, compatibility only /
        rank, full); sign test p < 0.05, median gain >= +0.01), AND the share of queries with a partner in the first 10
        tries is >= 5x what random ordering gives, Monte Carlo p < 0.05 (200 random orders). Reported, not gating: the
        gain of each step over the previous one.
 G9-S1  RNA VERSIONS CHANGE THE ADDRESS. Genes with an isoform whose alternative sequence removes or changes the tag
        (signal, transit or localization motif) vs genes whose isoforms differ elsewhere: share found in two or more
        room classes. CLAIM: Fisher odds ratio >= 1.5, p < 0.05, Monte Carlo p < 0.05 (labels shuffled, 1,000).
DESCRIPTIVE: when (in which of 40 tissues the tag-changing versions are used most, HPA transcripts) and candidate
governors (splicing proteins whose RNA level across tissues follows each gene's tag-changing share -- correlation,
not cause; to be checked against ENCODE splicing-factor knockdowns in a follow-up).
NOT: imaging rooms and tags are both imperfect; ENCODE fractions are two cell lines; isoform names are mapped from
UniProt notes ("isoform 2" -> "-2"), which can miss; the partner finder is judged on known interactions only, so a
"miss" may be an unknown true partner; correlations across tissues are not regulation.

Run: python3 -I rem/atlas/addresstest.py declare | test [--trial]
"""

from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
TRIAL = "--trial" in sys.argv


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


pp = _load("generegppi"); cf = _load("cellfix")
gx = pp.gx; cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "address")
CLAIMS = ["G9-A1", "G9-A2", "G9-A3", "G9-P1", "G9-S1"]
SEED = 20261129
N1000, N200 = (50, 20) if TRIAL else (1000, 200)
H_ = cf.H_
CLASS_OF_ROOM = {"nucleoplasm": "nuclear", "nucleolus": "nuclear", "nuclear_membrane": "nuclear", "er": "secretory", "golgi": "secretory",
                 "vesicles": "secretory", "plasma_membrane": "secretory", "mitochondria": "mito", "cytosol": "cyto", "cytoskeleton": "cyto", "centrosome": "cyto"}


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "addresstest.py"), "rb").read()); h.update(cf.code_sha().encode())
    for n in ("ADDR_UNIPROT", "ADDR_MITOCARTA", "RNA_FRACTIONS_ENCODE", "ISOFORM_HPA_TISSUE"):
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def load_all():
    P = json.load(open(gx.L("ADDR_UNIPROT"))); MC = json.load(open(gx.L("ADDR_MITOCARTA")))
    ppi, rooms = pp.knowledge()
    def aclass(g):
        v = P.get(g, {}); c = set()
        if v.get("signal") or v.get("tm", 0) > 0 or v.get("gpi"):
            c.add("secretory")
        if v.get("transit") or g in MC:
            c.add("mito")
        if "NLS" in v.get("motifs", []):
            c.add("nuclear")
        return c
    return P, MC, ppi, rooms, aclass


def fisher(a, b):
    """a, b boolean arrays: exposure, outcome -> odds ratio, one-sided p."""
    from scipy.stats import fisher_exact
    t = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    OR, p = fisher_exact(t, alternative="greater"); return float(OR), float(p), t


def perm_or(a, b, rng, n):
    real = fisher(a, b)[0]; ge = 0
    for _ in range(n):
        x = a[rng.permutation(len(a))]
        t11 = (x & b).sum(); t10 = (x & ~b).sum(); t01 = (~x & b).sum(); t00 = (~x & ~b).sum()
        ge += ((t11 + 0.5) * (t00 + 0.5) / ((t10 + 0.5) * (t01 + 0.5))) >= real
    return float((1 + ge) / (1 + n))


def auroc(score, lab):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------ A1
def run_a1(rng, D):
    P, MC, ppi, rooms, aclass = D
    genes = sorted(g for g in P if g in rooms)
    rc = {g: {CLASS_OF_ROOM[r] for r in rooms[g]} for g in genes}
    tests = {"signal peptide -> secretory": (lambda g: bool(P[g]["signal"]), lambda g: "secretory" in rc[g]),
             "mito transit -> mitochondria": (lambda g: bool(P[g]["transit"]) and "Mitochondrion" in P[g]["transit"][0][2], lambda g: "mito" in rc[g]),
             "NLS -> nuclear": (lambda g: "NLS" in P[g]["motifs"], lambda g: "nuclear" in rc[g]),
             "ER retention -> ER": (lambda g: "ER_RETENTION" in P[g]["motifs"], lambda g: "er" in rooms[g]),
             "peroxisome signal -> vesicles": (lambda g: "PTS1" in P[g]["motifs"], lambda g: "vesicles" in rooms[g])}
    out = {}
    for name, (tag, room) in tests.items():
        a = np.array([tag(g) for g in genes]); b = np.array([room(g) for g in genes])
        OR, p, t = fisher(a, b); pm = perm_or(a, b, rng, N1000)
        out[name] = dict(tagged=int(a.sum()), tagged_in_room=t[0][0], share_tagged_in_room=float(t[0][0] / max(1, a.sum())),
                         share_untagged_in_room=float(t[1][0] / max(1, (~a).sum())), odds_ratio=OR, p=p, p_mc=pm, passed=bool(OR >= 3 and p < 0.05 and pm < 0.05))
    main = ["signal peptide -> secretory", "mito transit -> mitochondria", "NLS -> nuclear"]
    return dict(proteins_with_rooms=len(genes), tests=out, gate=dict(real=all(out[k]["passed"] for k in main), controls_ok=True,
                verdict="PASS" if all(out[k]["passed"] for k in main) else "FAIL"))


# ------------------------------------------------------------------------------------------------ A2, A3
def run_a23(rng, D):
    P, MC, ppi, rooms, aclass = D
    z = gx.npz("RNA_FRACTIONS_ENCODE"); X = z["X"]; rows = z["rows"].astype(str); cols = list(z["cols"].astype(str))
    a2, a3 = {}, {}
    for cell in ("K562", "HepG2"):
        mem = X[:, [cols.index(f"{cell}_membrane_rep1"), cols.index(f"{cell}_membrane_rep2")]]; cyt = X[:, [cols.index(f"{cell}_cytosolic_rep1"), cols.index(f"{cell}_cytosolic_rep2")]]
        enr = np.mean(np.log2((mem + 1) / (cyt + 1)), 1); ok = (mem.mean(1) >= 5) | (cyt.mean(1) >= 5)
        keep = [i for i in np.nonzero(ok)[0] if rows[i] in P]; g = rows[keep]; e = enr[keep]
        er = np.array([bool(P[x]["signal"]) or P[x]["tm"] > 0 for x in g])
        au = auroc(e, er); nulls = [auroc(e, er[rng.permutation(len(er))]) for _ in range(N1000)]
        a2[cell] = dict(genes=len(g), er_made=int(er.sum()), auroc=au, null_95=float(np.percentile(nulls, 95)), p_mc=float((1 + sum(x >= au for x in nulls)) / (1 + len(nulls))),
                        median_enrichment_er_made=float(np.median(e[er])), median_enrichment_other=float(np.median(e[~er])))
        mito = np.array([(x in MC) for x in g]) & ~er; other = ~er & ~np.array([(x in MC) or bool(P[x]["transit"]) for x in g])
        sel = mito | other; au3 = auroc(e[sel], mito[sel]); lab = mito[sel]; nulls3 = [auroc(e[sel], lab[rng.permutation(len(lab))]) for _ in range(N1000)]
        a3[cell] = dict(mito=int(mito.sum()), other=int(other.sum()), auroc=au3, null_95=float(np.percentile(nulls3, 95)),
                        p_mc=float((1 + sum(x >= au3 for x in nulls3)) / (1 + len(nulls3))), median_enrichment_mito=float(np.median(e[mito])))
    g2 = all(v["auroc"] >= 0.70 and v["p_mc"] < 0.05 for v in a2.values()); g3 = all(v["auroc"] >= 0.60 and v["p_mc"] < 0.05 for v in a3.values())
    return (dict(per_cell=a2, gate=dict(real=g2, controls_ok=True, verdict="PASS" if g2 else "FAIL")),
            dict(per_cell=a3, gate=dict(real=g3, controls_ok=True, verdict="PASS" if g3 else "FAIL")))


# ------------------------------------------------------------------------------------------------ P1
def run_p1(rng, D):
    from scipy.stats import rankdata
    P, MC, ppi, rooms, aclass = D
    B = gx.npz("PPI_BIOPLEX"); truth = collections.defaultdict(set)
    for p in ppi:
        x, y = tuple(p); truth[x].add(y); truth[y].add(x)
    for a, b in zip(B["a"].astype(str), B["b"].astype(str)):
        truth[a].add(b); truth[b].add(a)
    U = sorted(P); ui = {g: i for i, g in enumerate(U)}; nU = len(U)
    dev = {g for g in U if H_(g) == 0}
    # domain-pair log-odds from DEV-DEV interactions of the three training sources
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
    by_dom = collections.defaultdict(list)
    for (a, b), v in L.items():
        if v > 0:
            by_dom[a].append((b, v)); by_dom[b].append((a, v))
    dom_genes = collections.defaultdict(list)
    for g in U:
        for d_ in dom[g]:
            dom_genes[d_].append(ui[g])
    # rooms, classes, made-together
    RN = list(CLASS_OF_ROOM); CN = ["nuclear", "secretory", "mito", "cyto"]
    RM = np.array([[r in rooms.get(g, ()) for r in RN] for g in U], np.float32)
    CM = np.array([[c in (aclass(g) | {CLASS_OF_ROOM[r] for r in rooms.get(g, ())}) for c in CN] for g in U], np.float32)
    S = gx.npz("EXPR_HPA_SCT"); sr = {g: i for i, g in enumerate(S["rows"].astype(str))}
    MADE = np.zeros((nU, S["X"].shape[1]), bool)
    for g, i in ui.items():
        if g in sr:
            MADE[i] = S["X"][sr[g]] >= 1.0
    queries = [g for g in U if g not in dev and truth.get(g) and any(t in ui for t in truth[g])]
    if TRIAL:
        queries = queries[::8]
    ev = cf.evidences(); res = collections.defaultdict(list); nulls10 = np.zeros(N200); hit10 = []
    for s in range(0, len(queries), 500):
        Q = queries[s:s + 500]; sim, _ = cf.combined_sim(ev, Q, U); sim = np.nan_to_num(sim, nan=-9)
        for qi, q in enumerate(Q):
            i = ui[q]; ds = np.zeros(nU, np.float32)
            for a in dom[q]:
                for b, v in by_dom.get(a, ()):
                    idx = dom_genes.get(b)
                    if idx:
                        ds[idx] = np.maximum(ds[idx], v)
            s1 = rankdata(ds) + rankdata(sim[qi]); s1[i] = -1e9
            dest = ((RM @ RM[i]) > 0) | ((CM @ CM[i]) > 0) if (RM[i].any() or CM[i].any()) else np.ones(nU, bool)
            mq = MADE[i]; together = (MADE[:, mq].mean(1) >= 0.5) if mq.any() else np.ones(nU, bool)
            tr = np.array([ui[t] for t in truth[q] if t in ui and t != q])
            orders = {"compatibility": s1, "plus_destination": s1 + 1e6 * dest, "full": s1 + 1e6 * dest + 1e3 * (dest & together)}
            for k, sc in orders.items():
                rk = (-sc).argsort(kind="stable"); pos = np.empty(nU, np.int64); pos[rk] = np.arange(nU)
                res[k].append(int(pos[tr].min()) + 1)
            n = len(tr); N = nU - 1
            hit10.append(1 - np.prod([(N - n - j) / (N - j) for j in range(10)]))
    r = {k: np.array(v) for k, v in res.items()}
    gain_full = np.log2(r["compatibility"] / r["full"]); gain_d = np.log2(r["compatibility"] / r["plus_destination"]); gain_m = np.log2(r["plus_destination"] / r["full"])
    exp10 = np.array(hit10); obs10 = float((r["full"] <= 10).mean())
    for b in range(N200):
        nulls10[b] = float((rng.random(len(exp10)) < exp10).mean())
    pm = float((1 + (nulls10 >= obs10).sum()) / (1 + N200))
    b_full = cf.better(gain_full)
    out = dict(queries=len(queries), candidates=nU - 1, domain_pairs_learned=len(L), dev_interactions_used=nint,
               first_try_hit={k: float((v == 1).mean()) for k, v in r.items()}, within_10_tries={k: float((v <= 10).mean()) for k, v in r.items()},
               median_tries_to_first_partner={k: float(np.median(v)) for k, v in r.items()}, random_within_10=float(exp10.mean()),
               fold_over_random_within_10=float(obs10 / max(exp10.mean(), 1e-9)), p_mc=pm, full_vs_compatibility=b_full,
               step_destination=cf.better(gain_d), step_made_together=cf.better(gain_m))
    ok = bool(b_full["passed"] and out["fold_over_random_within_10"] >= 5 and pm < 0.05)
    out["gate"] = dict(real=ok, controls_ok=True, verdict="PASS" if ok else "FAIL")
    return out


# ------------------------------------------------------------------------------------------------ S1 and the descriptive "when / what governs"
def run_s1(rng, D):
    P, MC, ppi, rooms, aclass = D
    genes = sorted(g for g in P if g in rooms and P[g]["has_var_seq"])
    sw = np.array([bool(P[g]["isoforms_changing_tag"]) for g in genes]); dual = np.array([len({CLASS_OF_ROOM[r] for r in rooms[g]}) >= 2 for g in genes])
    OR, p, t = fisher(sw, dual); pm = perm_or(sw, dual, rng, N1000)
    out = dict(genes_with_isoforms=len(genes), tag_changing=int(sw.sum()), dual_share_tag_changing=float(t[0][0] / max(1, sw.sum())),
               dual_share_other=float(t[1][0] / max(1, (~sw).sum())), odds_ratio=OR, p=p, p_mc=pm, passed=bool(OR >= 1.5 and p < 0.05 and pm < 0.05))
    out["gate"] = dict(real=out["passed"], controls_ok=True, verdict="PASS" if out["passed"] else "FAIL")
    return out


def when_and_governors(D):
    P, MC, ppi, rooms, aclass = D
    z = gx.npz("ISOFORM_HPA_TISSUE"); X = z["X"]; tg = z["gene"].astype(str); tt = z["transcript"].astype(str); tissues = list(z["cols"].astype(str))
    row = collections.defaultdict(list)
    for i, (g, t) in enumerate(zip(tg, tt)):
        row[g].append((t, i))
    pw = json.load(open(gx.L("SETS_PATHWAY"))); spl = set()
    for v in pw.values():
        if v["name"] in ("mRNA Splicing", "mRNA Splicing - Major Pathway", "mRNA Splicing - Minor Pathway"):
            spl |= set(v["members"])
    gexp = {}
    for g in spl:
        if g in row:
            gexp[g] = X[[i for _, i in row[g]]].sum(0)
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); out = {}
    for g, v in P.items():
        ch = v.get("isoforms_changing_tag")
        if not ch or g not in row:
            continue
        acc = v["uniprot"]; iso_of = {t: i for t, i in v["transcripts"]}
        changed = {f"{acc}-{k}" for k in ch if k.isdigit()}
        tx = [(t, i) for t, i in row[g] if t in iso_of]
        if not tx or not changed:
            continue
        ch_rows = [i for t, i in tx if iso_of[t] in changed]; all_rows = [i for t, i in tx]
        if not ch_rows:
            continue
        tot = X[all_rows].sum(0); frac = np.where(tot >= 1, X[ch_rows].sum(0) / np.maximum(tot, 1e-9), np.nan)
        if np.isfinite(frac).sum() < 10 or np.nanmax(frac) - np.nanmin(frac) < 0.3:
            continue
        ok = np.isfinite(frac); cors = []
        for s_, e_ in gexp.items():
            if e_[ok].std() > 0:
                cors.append((float(np.corrcoef(frac[ok], np.log2(e_[ok] + 1))[0, 1]), sym(s_)))
        cors.sort(key=lambda x: -abs(x[0]))
        top = np.argsort(-np.nan_to_num(frac, nan=-1))[:3]
        out[sym(g)] = dict(gene=g, tag_changed=sorted({k for x in ch.values() for k in x}), range=round(float(np.nanmax(frac) - np.nanmin(frac)), 2),
                           tissues_most_tag_changed=[[tissues[j], round(float(frac[j]), 2)] for j in top],
                           candidate_governors=[[n, round(c, 2)] for c, n in cors[:5]])
    return out


# ------------------------------------------------------------------------------------------------ ledger
def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G9-")[0].split("DESCRIPTIVE")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G9 already tested")
    t0 = time.time(); rng = np.random.default_rng(SEED); D = load_all(); out = {}
    out["G9-A1"] = run_a1(rng, D); print("G9-A1 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G9-A2"], out["G9-A3"] = run_a23(rng, D); print("G9-A2/A3 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G9-S1"] = run_s1(rng, D); print("G9-S1 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G9-P1"] = run_p1(rng, D); print("G9-P1 done", f"({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        print("trial ran; outcomes not shown"); return
    wg = when_and_governors(D); print(f"descriptive: {len(wg)} genes whose tag-changing share varies across tissues ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "address_tests.json"), "w"), indent=1, default=float)
    json.dump(wg, open(os.path.join(OUT, "isoform_when_and_governors.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "addresstest:G9", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    facts = [Fact(f"CLAIM-{cid}", "address_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS]
    for s_, v in wg.items():
        facts.append(Fact(v["gene"], "isoform/tag_changing_by_tissue", v, t[:10], t, "derived", src, inputs=inp,
                          context={"governors": "splicing proteins whose RNA follows the tag-changing share across 40 tissues; correlation, not cause"}))
    cbb.bank().add_many(facts)
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
