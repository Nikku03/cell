"""PARTNERCHECK (G13) -- two checks of the partner lists now on the cell-map page (G9 finder + ESM C fit, G10-P1 rule).

Asked by the user: "do the two partner checks" -- (1) is ESM C's gain only the big machines (ribosome-like complexes
whose parts look alike and all bind each other)? (2) an independent check against interactions never used to build
or train the lists.
Inputs: the G9 order (esmcbundle.g9_orders: fit by Pfam domain pairs learned on DEV interactions + similar behaviour,
then same destination, then made together); the ESM C pair model (colab/esmc/esmctest.py train_pair on the banked
ESMC_600M_FINGERPRINTS, DEV-DEV interactions from HPA, OpenCell, Complex Portal; retrained here, reproduces G10-P1 at
50.7% vs 51.0%); re-ordering inside the G9 top 500 exactly as G10-P1. Complexes: Complex Portal (SETS_COMPLEX).
Independent truth: BioPlex 3.0 (HEK293T and HCT116 affinity purification; PPI_BIOPLEX, local only), pairs NOT in
HPA, OpenCell or Complex Portal -- never used by the finder, the pair model or any earlier truth set.
Queries are EXAM proteins only (all models were fit on DEV). Trial: 1/20 of the queries, 50 Monte Carlo draws,
only controls shown.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G13-*):
 G13-B1 ESM C HELPS OUTSIDE THE BIG MACHINES. Truth = a protein's known partners (HPA, OpenCell, Complex Portal) that
        share no Complex Portal complex with it (contacts, not machine parts). For EXAM proteins whose G9 top 500 holds
        such a partner: gain = log2(tries to the first one in G9 order / tries after the ESM C re-ordering). CLAIM:
        BETTER (sign test p < 0.05, median gain >= +0.01). Controls: a pair model trained on shuffled fingerprints
        must not be BETTER; a planted score (that partner + unit noise) must be BETTER. Reported: the same for partners
        in big complexes (>= 20 members) and the share of G10-P1's gain they carry.
 G13-X1 THE LISTS FIND PARTNERS NOBODY GAVE THEM. Truth = BioPlex partners not in HPA, OpenCell or Complex Portal.
        CLAIM (both): (a) the page's first 10 tries (G9 + ESM C) hold such a partner for >= 5x the share expected by
        chance, Monte Carlo p < 0.05 (random orders, 1,000 draws); (b) for EXAM proteins whose G9 top 500 holds one,
        the ESM C order is BETTER than the G9 order. Controls for (b) as in B1.
NOT: Complex Portal misses complexes, so some "contacts" are machine parts; BioPlex is measured in two cell lines
and has its own biases (sticky proteins); a miss may be a true partner nobody has measured.

Run: python3 -I rem/atlas/partnercheck.py declare | test [--trial]
"""

from __future__ import annotations
import collections
import gzip
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


def _load(name, path=None):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, path or os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


eb = _load("esmcbundle")
et = _load("esmctest", R("colab", "esmc", "esmctest.py"))
at, gx = eb.at, eb.gx
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "address")
CLAIMS = ["G13-B1", "G13-X1"]
SEED = 20261127
N_MC = 50 if TRIAL else 1000
BIG = 20


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "partnercheck.py"), "rb").read())
    for p in (os.path.join(HERE, "esmcbundle.py"), R("colab", "esmc", "esmctest.py")):
        h.update(open(p, "rb").read())
    for n in ("ESMC_600M_FINGERPRINTS", "PPI_BIOPLEX", "SETS_COMPLEX", "ADDR_UNIPROT", "EXPR_HPA_SCT"):
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def load():
    D = at.load_all(); P, MC, ppi, rooms, aclass = D
    U = sorted(P); ui = {g: i for i, g in enumerate(U)}; dev = np.array([at.H_(g) == 0 for g in U])
    known = collections.defaultdict(set)
    for p in ppi:
        x, y = tuple(p)
        if x in ui and y in ui:
            known[ui[x]].add(ui[y]); known[ui[y]].add(ui[x])
    comp = collections.defaultdict(set); size = {}
    for k, v in json.load(open(gx.L("SETS_COMPLEX"))).items():
        m = [ui[g] for g in v["members"] if g in ui]; size[k] = len(v["members"])
        for i in m:
            comp[i].add(k)
    B = gx.npz("PPI_BIOPLEX"); bp = collections.defaultdict(set)
    for a, b in zip(B["a"].astype(str), B["b"].astype(str)):
        if a in ui and b in ui and a != b and ui[b] not in known[ui[a]]:
            bp[ui[a]].add(ui[b]); bp[ui[b]].add(ui[a])
    z = np.load(gx.L("ESMC_600M_FINGERPRINTS"), allow_pickle=False); assert list(z["genes"][:len(U)]) == U
    E = z["X"][:len(U)].astype(np.float32); red = et.Reducer(E[dev]); zz = red(E)
    pos = np.array([(a, b) for a in range(len(U)) if dev[a] for b in known[a] if b > a and dev[b]])
    return D, U, ui, dev, known, comp, size, bp, zz, pos


def reorder_rank(cand, s1, fl, score_all, is_p):
    from scipy.stats import rankdata
    dest = (fl & 1).astype(bool); tog = (fl & 2).astype(bool)
    key = s1.astype(np.float64) + rankdata(score_all)[cand] + 1e6 * dest + 1e3 * (dest & tog)
    o = np.argsort(-key, kind="stable"); h = is_p[o]
    return int(np.argmax(h)) + 1 if h.any() else -1, o


def gains(Q, o, truth_of, scorers):
    out = {k: [] for k in scorers}; base = []
    for k_, qi in enumerate(Q):
        C = o["cand"][k_].astype(np.int64); t = truth_of(qi)
        is_p = np.array([c in t for c in C])
        if not is_p.any():
            continue
        base.append(int(np.argmax(is_p)) + 1)
        for name, f in scorers.items():
            out[name].append(reorder_rank(C, o["s1"][k_], o["flags"][k_], f(qi), is_p)[0])
    b = np.array(base); return b, {k: np.log2(b / np.array(v)) for k, v in out.items()}, {k: np.array(v) for k, v in out.items()}


def run(rng, S):
    D, U, ui, dev, known, comp, size, bp, zz, pos = S
    exam = [i for i in range(len(U)) if not dev[i]]
    qB = [i for i in exam if any(not (comp[i] & comp[j]) for j in known[i])]
    qX = [i for i in exam if bp[i]]
    t_big = lambda qi: {j for j in known[qi] if any(size[c] >= BIG for c in comp[qi] & comp[j])}
    qBig = [i for i in exam if t_big(i)]
    if TRIAL:
        qB, qX, qBig = qB[::20], qX[::20], qBig[::20]
    Q = sorted(set(qB) | set(qX) | set(qBig))
    print(f"queries: contacts {len(qB)}, BioPlex-only {len(qX)}; G9 shortlists for {len(Q)}", flush=True)
    o = eb.g9_orders(D, U, [U[i] for i in Q], {}, k=500); oi = {qi: k for k, qi in enumerate(Q)}
    sub = lambda qs: {key: o[key][[oi[q] for q in qs]] for key in ("cand", "s1", "flags")}
    devi = np.nonzero(dev)[0]
    f_e = et.train_pair(zz, pos, devi, np.random.default_rng(et.SEED))
    f_s = et.train_pair(zz[et.derange(len(zz), rng)], pos, devi, np.random.default_rng(et.SEED))
    noise = rng.normal(0, 1, len(U))
    out = {}
    # B1
    t_contact = lambda qi: {j for j in known[qi] if not (comp[qi] & comp[j])}
    def planted_for(tf):
        def f(qi):
            v = noise.copy(); v[list(tf(qi))] += 1.0; return v
        return f
    base, g, rk = gains(qB, sub(qB), t_contact, {"esmc": f_e, "shuffled": f_s, "planted": planted_for(t_contact)})
    bB = {k: et.better(v) for k, v in g.items()}
    big = None
    if qBig:
        bb, gb, rb = gains(qBig, sub(qBig), t_big, {"esmc": f_e})
        big = dict(queries=int(len(bb)), esmc_vs_g9=et.better(gb["esmc"]), within_10=dict(g9=float((bb <= 10).mean()), esmc=float((rb["esmc"] <= 10).mean())))
    out["G13-B1"] = dict(queries=int(len(base)), median_tries=dict(g9=float(np.median(base)), **{k: float(np.median(v)) for k, v in rk.items()}),
                         within_10=dict(g9=float((base <= 10).mean()), **{k: float((v <= 10).mean()) for k, v in rk.items()}),
                         esmc_vs_g9=bB["esmc"], shuffled_vs_g9=bB["shuffled"], planted_vs_g9=bB["planted"], reported_big_complex_partners=big)
    ok = not bB["shuffled"]["passed"] and bB["planted"]["passed"]
    out["G13-B1"]["gate"] = dict(real=bool(bB["esmc"]["passed"]), controls_ok=bool(ok), verdict=("PASS" if bB["esmc"]["passed"] else "FAIL") if ok else "VOID (controls)")
    # X1 (a): page lists' first 10 tries vs chance
    t_bp = lambda qi: bp[qi]
    hit, exp = [], []
    N = len(U) - 1
    for qi in qX:
        k_ = oi[qi]; C = o["cand"][k_].astype(np.int64); _, order = reorder_rank(C, o["s1"][k_], o["flags"][k_], f_e(qi), np.zeros(len(C), bool))
        top = C[order][:10]; n = len(bp[qi]); hit.append(bool(set(top.tolist()) & bp[qi]))
        exp.append(1 - np.prod([(N - n - j) / (N - j) for j in range(10)]))
    hit, exp = np.array(hit), np.array(exp); obs = float(hit.mean())
    nulls = np.array([float((rng.random(len(exp)) < exp).mean()) for _ in range(N_MC)]); pm = float((1 + (nulls >= obs).sum()) / (1 + N_MC))
    fold = obs / max(float(exp.mean()), 1e-12)
    # X1 (b)
    baseX, gX, rkX = gains(qX, sub(qX), t_bp, {"esmc": f_e, "shuffled": f_s, "planted": planted_for(t_bp)})
    bX = {k: et.better(v) for k, v in gX.items()}
    out["G13-X1"] = dict(queries=int(len(qX)), within_10=obs, random_within_10=float(exp.mean()), fold_over_random=fold, p_mc=pm,
                         shortlist_queries=int(len(baseX)), median_tries=dict(g9=float(np.median(baseX)) if len(baseX) else None, **{k: float(np.median(v)) for k, v in rkX.items() if len(v)}),
                         esmc_vs_g9=bX["esmc"], shuffled_vs_g9=bX["shuffled"], planted_vs_g9=bX["planted"])
    real = bool(fold >= 5 and pm < 0.05 and bX["esmc"]["passed"]); ok = not bX["shuffled"]["passed"] and bX["planted"]["passed"]
    out["G13-X1"]["gate"] = dict(real=real, controls_ok=bool(ok), verdict=("PASS" if real else "FAIL") if ok else "VOID (controls)")
    return out


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G13-")[0].split("NOT:")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G13 already tested")
    t0 = time.time(); rng = np.random.default_rng(SEED); S = load(); out = run(rng, S)
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok")
        print("trial ran; outcomes not shown"); return
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "partner_checks.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "partnercheck:G13", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "partner_check", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
