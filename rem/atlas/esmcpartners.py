"""ESMCPARTNERS -- descriptive (no claims): the ESM C fingerprints from the user's Colab run into the memory bank, and the
partner suggestions on the cell-map page re-ordered with ESM C's "do they fit" score, by the rule that passed G10-P1.

Asked by the user: re-ran the Colab notebook keeping the fingerprints, split them into 12 parts in Drive so they could
come through the connector (check code 8d8a962209771305, matched after rejoining here).
Layer ESMC_600M_FINGERPRINTS (local file, 89 MB): 41,603 sequences (20,083 canonical proteins in bundle order + 21,520
isoforms) x 1,152, float16, mean over residues of ESMC-600M's last hidden layer (declared code 6c574b9b3f1f2408).
Partner order (G10-P1 rule, PASS): the pair model (colab/esmc/esmctest.py train_pair; 256 principal components fit
on DEV, trained on DEV-DEV interactions from HPA, OpenCell, Complex Portal) is retrained here with the same code and
seed; for each protein the G9 finder's first 500 candidates are re-ordered with the G9 key plus the rank of the pair
score among all candidates; the first 10 are kept. Check before use: on the 2,648 held-out proteins of G10-P1 the
retrained model must give about the recorded within-10 share (51.0%). Proteins with no domain or behaviour data (no
G9 list) get the pair model's own top 10 (reported in G10: 24% within 10 vs 0.5% random; not a claim).

Run: python3 -I rem/atlas/esmcpartners.py bank | partners
"""

from __future__ import annotations
import collections
import gzip
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))


def _load(name, path=None):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, path or os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


eb = _load("esmcbundle")
et = _load("esmctest", R("colab", "esmc", "esmctest.py"))
at, gx = eb.at, eb.gx
cbb, Fact = gx.cbb, gx.Fact
FP = os.path.join(HERE, "_cache", "esmc", "fingerprints_ESMC-600M.npz")


def bank():
    f = cbb.layer_fact("ESMC_600M_FINGERPRINTS", 1, "esmc", FP, "sequences (UniProt canonical in bundle order, then isoforms)", "1,152 (ESMC-600M last layer, mean over residues)",
                       "float16", tier="derived", extra=dict(code_sha="6c574b9b3f1f2408", check="8d8a962209771305", ran_on="NVIDIA L4 (user's Colab)"))
    print(cbb.bank().add_many([f]), "fact", f.ref)


def partners():
    t0 = time.time()
    z = np.load(gx.L("ESMC_600M_FINGERPRINTS"), allow_pickle=False)
    gj = json.load(gzip.open(R("colab", "esmc", "bundle", "genes.json.gz"), "rt")); U = gj["genes"]; G = gj["g"]; ui = {g: i for i, g in enumerate(U)}
    E = z["X"][:len(U)].astype(np.float32); assert list(z["genes"][:len(U)]) == U
    dev = np.array([G[g]["split"] == "dev" for g in U]); red = et.Reducer(E[dev]); zz = red(E)
    pairs = [l.rstrip("\n").split("\t")[:2] for l in gzip.open(R("colab", "esmc", "bundle", "ppi.tsv.gz"), "rt")][1:]
    truth = collections.defaultdict(set); pair_idx = []
    for a, b in pairs:
        if a in ui and b in ui and a != b:
            truth[ui[a]].add(ui[b]); truth[ui[b]].add(ui[a]); pair_idx.append((ui[a], ui[b]))
    pos = np.array([(a, b) for a, b in pair_idx if dev[a] and dev[b]])
    f = et.train_pair(zz, pos, np.nonzero(dev)[0], np.random.default_rng(et.SEED)); print(f"pair model trained ({time.time() - t0:.0f}s)", flush=True)
    from scipy.stats import rankdata

    def reorder(cand, s1, fl, qi):
        dest = (fl & 1).astype(bool); tog = (fl & 2).astype(bool)
        key = s1.astype(np.float64) + rankdata(f(qi))[cand] + 1e6 * dest + 1e3 * (dest & tog)
        return cand[np.argsort(-key, kind="stable")]
    # check: the G10-P1 held-out proteins
    sl = dict(np.load(R("colab", "esmc", "bundle", "shortlist.npz"), allow_pickle=False)); w10 = []
    for r in range(len(sl["q"])):
        qi = int(sl["q"][r]); C = sl["cand"][r].astype(np.int64); o = reorder(C, sl["s1"][r], sl["flags"][r], qi)
        hit = np.array([c in truth[qi] for c in o]); w10.append(bool(hit[:10].any()))
    chk = float(np.mean(w10)); print(f"check on the 2,648 G10-P1 proteins: within 10 tries {chk:.1%} (recorded 51.0%)", flush=True)
    if abs(chk - 0.510) > 0.03:
        raise SystemExit("the retrained model does not reproduce G10-P1; not using it")
    # every protein
    D = at.load_all(); P = D[0]
    ev = eb.cf.evidences(); with_beh = set().union(*[set(map(str, v[0])) for v in ev.values()])
    has = [g for g in U if P[g]["pfam"] or g in with_beh]
    o = eb.g9_orders(D, U, has, {}, k=500)
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); out = {}
    for k_, g in enumerate(has):
        qi = ui[g]; top = reorder(o["cand"][k_].astype(np.int64), o["s1"][k_], o["flags"][k_], qi)[:10]
        out[sym(g)] = dict(gene=g, how="G9 finder + ESM C fit", suggested=[[sym(U[j]), int(j in truth[qi])] for j in top], known_partners=len(truth[qi]))
    for g in U:
        if sym(g) not in out:
            qi = ui[g]; sc = f(qi); sc[qi] = -np.inf; top = np.argsort(-sc, kind="stable")[:10]
            out[sym(g)] = dict(gene=g, how="ESM C fit only", suggested=[[sym(U[j]), int(j in truth[qi])] for j in top], known_partners=len(truth[qi]))
    with gzip.open(R("outputs", "address", "partner_suggestions_esmc.json.gz"), "wt") as fh:
        json.dump(out, fh, separators=(",", ":"))
    t = cbb.now(); src = {"name": "esmcpartners", "rule": "G10-P1 (PASS)", "check_within_10": round(chk, 4)}
    facts = [Fact(v["gene"], "partners/suggested_esmc", v, t[:10], t, "derived", src, inputs=("ESMC_600M_FINGERPRINTS", "outputs/address/partner_suggestions_esmc.json.gz"),
                  context={"note": "first 10 tries of the partner finder re-ordered with ESM C fit (G10-P1 rule); known partners marked; predictions"})
             for v in out.values()]
    n = cbb.bank().add_many(facts)
    print(f"{len(out)} proteins with suggestions ({sum(v['how'] == 'ESM C fit only' for v in out.values())} from ESM C alone); {n} facts ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    {"bank": bank, "partners": partners}[sys.argv[1]]()
