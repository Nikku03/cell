"""ERLOCTEST (G12) -- does a protein's RNA travel with it to the ER and the mitochondria, and is it the protein's tag
(made first, while the ribosome works) or something in the RNA itself that takes it there?

Asked by the user: "can we get data for which rna will get modified tail which tells the transporter protein to where
that protein needs to be delivered?"; G9-A2/A3 failed because ENCODE's "membrane" fraction is not an ER read-out; then
"do #2". Data: APEX_SEQ_HEK293T (Fazal et al. 2019 Cell; RNA labelled within nanometres of the ER membrane (ERM), the
outer mitochondrial membrane (OMM), the nucleus (NLS) and six more places; enrichment = log2 labelled / unlabelled;
OMM also under puromycin, which pulls the growing protein off the ribosome, and cycloheximide, which freezes it on).
Tags from ADDR_UNIPROT: ER-made = signal peptide or transmembrane segment; mitochondrial = mitochondrial transit peptide
or MitoCarta 3.0 (ADDR_MITOCARTA), and not ER-made; other = neither. A gene is measured at a place when its four
libraries there hold >= 100 reads. No model is fit, so all genes are used; the trial used DEV genes only and showed
only controls.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G12-*):
 G12-A2 ER-MADE PROTEINS' RNA SITS AT THE ER. AUROC of ERM enrichment, ER-made vs other proteins' RNA. CLAIM: AUROC >=
        0.70 and Monte Carlo p < 0.05 (labels shuffled, 1,000 draws). Controls: nuclear enrichment (NLS) with the same
        labels must not pass; a planted score (label + unit noise) must pass. Reported: ER lumen (KDEL).
 G12-A3 MITOCHONDRIAL PROTEINS' RNA SITS AT THE MITOCHONDRIA. AUROC of OMM enrichment, mitochondrial vs other. CLAIM:
        AUROC >= 0.60 and Monte Carlo p < 0.05. Controls as A2 (NLS must not pass; planted must pass).
 G12-T1 THE PROTEIN'S TAG PULLS THE RNA. Drop = OMM enrichment without drug - with puromycin. CLAIM: AUROC of the drop,
        mitochondrial vs other, >= 0.60 and Monte Carlo p < 0.05. Controls: the drop with cycloheximide (ribosome and
        growing protein stay on the RNA) must not pass; a planted drop must pass. Reported: inside mitochondrial
        proteins, transit peptide vs none.
 REPORTED (not a claim): RNAs at the ER whose protein has no ER tag (hidden ER clients), and mitochondrial-protein RNAs
        that stay at the mitochondria under puromycin (their RNA, not the growing protein, holds them there -- candidates
        for an RNA address code).
NOT: one cell line (HEK293T); APEX labels RNA near the place, not bound to it; puromycin also stops new translation,
so "stays under puromycin" means "does not need the growing protein", not proof of an RNA code; MitoCarta is local only
(only summaries are committed).

Run: python3 -I rem/atlas/erloctest.py declare | test [--trial]
"""

from __future__ import annotations
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
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


gx = _load("generegx"); cf = _load("cellfix")
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "erloc")
CLAIMS = ["G12-A2", "G12-A3", "G12-T1"]
SEED = 20261126
N_MC = 50 if TRIAL else 1000
MIN_READS = 100
H_ = cf.H_


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "erloctest.py"), "rb").read())
    for n in ("APEX_SEQ_HEK293T", "ADDR_UNIPROT", "ADDR_MITOCARTA"):
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def auroc(score, lab):
    from scipy.stats import rankdata
    lab = np.asarray(lab, bool); n1 = lab.sum(); n0 = len(lab) - n1
    r = rankdata(score); return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mc(score, lab, rng, bar):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1; real = (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    nulls = np.array([(r[rng.permutation(len(lab))[:n1]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) for _ in range(N_MC)])
    p = float((1 + (nulls >= real).sum()) / (1 + N_MC))
    return dict(auroc=float(real), null_95=float(np.percentile(nulls, 95)), p_mc=p, n_in=int(n1), n_out=int(n0), passed=bool(real >= bar and p < 0.05))


def load():
    z = gx.npz("APEX_SEQ_HEK293T"); X = z["X"].astype(np.float64); rows = z["rows"].astype(str); cols = list(z["cols"].astype(str))
    P = json.load(open(gx.L("ADDR_UNIPROT"))); MC = json.load(open(gx.L("ADDR_MITOCARTA")))
    keep = [i for i, g in enumerate(rows) if g in P and (not TRIAL or H_(g) == 0)]
    X = X[keep]; g = rows[keep]
    col = lambda c: X[:, cols.index(c)]
    er = np.array([bool(P[x]["signal"]) or P[x]["tm"] > 0 for x in g])
    transit = np.array([bool(P[x]["transit"]) for x in g]); mito = (np.array([x in MC for x in g]) | transit) & ~er
    other = ~er & ~mito
    return g, col, er, mito, other, transit, P


def gate(real, neg, planted):
    ok = bool(not neg and planted)
    return dict(real=bool(real), controls_ok=ok, verdict=("PASS" if real else "FAIL") if ok else "VOID (controls)")


def run(rng, D):
    g, col, er, mito, other, transit, P = D; out = {}
    # A2
    m = (col("ERM_count") >= MIN_READS) & np.isfinite(col("ERM_lfc")) & (er | other)
    mn = (col("NLS_count") >= MIN_READS) & np.isfinite(col("NLS_lfc")) & (er | other)
    mk = (col("KDEL_count") >= MIN_READS) & np.isfinite(col("KDEL_lfc")) & (er | other)
    a2 = mc(col("ERM_lfc")[m], er[m], rng, 0.70); neg = mc(col("NLS_lfc")[mn], er[mn], rng, 0.70)
    pl = mc(er[m] + rng.normal(0, 1, m.sum()), er[m], rng, 0.70)
    out["G12-A2"] = dict(erm=a2, nucleus_control=neg, planted=pl, reported_er_lumen=mc(col("KDEL_lfc")[mk], er[mk], rng, 0.70),
                         median_enrichment=dict(er_made=float(np.median(col("ERM_lfc")[m][er[m]])), other=float(np.median(col("ERM_lfc")[m][~er[m]]))))
    out["G12-A2"]["gate"] = gate(a2["passed"], neg["passed"], pl["passed"])
    # A3
    m3 = (col("OMM_count") >= MIN_READS) & np.isfinite(col("OMM_lfc")) & (mito | other)
    mn3 = (col("NLS_count") >= MIN_READS) & np.isfinite(col("NLS_lfc")) & (mito | other)
    a3 = mc(col("OMM_lfc")[m3], mito[m3], rng, 0.60); neg3 = mc(col("NLS_lfc")[mn3], mito[mn3], rng, 0.60)
    pl3 = mc(mito[m3] + rng.normal(0, 1, m3.sum()), mito[m3], rng, 0.60)
    out["G12-A3"] = dict(omm=a3, nucleus_control=neg3, planted=pl3, reported_matrix=mc(col("MITO_lfc")[(col("MITO_count") >= MIN_READS) & np.isfinite(col("MITO_lfc")) & (mito | other)],
                                                                                        mito[(col("MITO_count") >= MIN_READS) & np.isfinite(col("MITO_lfc")) & (mito | other)], rng, 0.60),
                         median_enrichment=dict(mito=float(np.median(col("OMM_lfc")[m3][mito[m3]])), other=float(np.median(col("OMM_lfc")[m3][~mito[m3]]))))
    out["G12-A3"]["gate"] = gate(a3["passed"], neg3["passed"], pl3["passed"])
    # T1
    base = col("OMM_basal_lfc"); drop_p = base - col("OMM_puro_lfc"); drop_c = base - col("OMM_cyc_lfc")
    mt = (col("OMM_count") >= MIN_READS) & (col("OMM_puro_count") >= MIN_READS) & np.isfinite(drop_p) & (mito | other)
    mtc = (col("OMM_count") >= MIN_READS) & (col("OMM_cyc_count") >= MIN_READS) & np.isfinite(drop_c) & (mito | other)
    t1 = mc(drop_p[mt], mito[mt], rng, 0.60); negt = mc(drop_c[mtc], mito[mtc], rng, 0.60); plt = mc(mito[mt] + rng.normal(0, 1, mt.sum()), mito[mt], rng, 0.60)
    mm = mt & mito
    out["G12-T1"] = dict(puromycin=t1, cycloheximide_control=negt, planted=plt,
                         median_drop=dict(mito=float(np.median(drop_p[mt][mito[mt]])), other=float(np.median(drop_p[mt][~mito[mt]]))),
                         reported_transit_vs_none_inside_mito=(mc(drop_p[mm], transit[mm], rng, 0.60) if transit[mm].sum() >= 10 and (~transit[mm]).sum() >= 10 else None))
    out["G12-T1"]["gate"] = gate(t1["passed"], negt["passed"], plt["passed"])
    return out


def catalogues(D):
    g, col, er, mito, other, transit, P = D; H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    m = (col("ERM_count") >= MIN_READS) & np.isfinite(col("ERM_lfc")) & np.isfinite(col("ERM_p"))
    hid = np.nonzero(m & other & (col("ERM_lfc") >= 1) & (col("ERM_p") < 0.05))[0]
    hidden = sorted(({"gene": g[i], "symbol": sym(g[i]), "erm_log2": round(float(col("ERM_lfc")[i]), 2)} for i in hid), key=lambda x: -x["erm_log2"])
    base = col("OMM_basal_lfc"); pur = col("OMM_puro_lfc")
    k = (col("OMM_count") >= MIN_READS) & (col("OMM_puro_count") >= MIN_READS) & np.isfinite(base) & np.isfinite(pur) & mito & (base >= 1) & (pur >= 1)
    stay = sorted(({"gene": g[i], "symbol": sym(g[i]), "omm_log2": round(float(base[i]), 2), "omm_log2_puromycin": round(float(pur[i]), 2),
                    "transit_peptide": bool(transit[i])} for i in np.nonzero(k)[0]), key=lambda x: -x["omm_log2_puromycin"])
    return dict(hidden_er_clients=hidden, mito_rna_without_growing_protein=stay)


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G12-")[0].split("REPORTED")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G12 already tested")
    t0 = time.time(); rng = np.random.default_rng(SEED); D = load(); out = run(rng, D)
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok")
        print("trial ran; outcomes not shown"); return
    cat = catalogues(D); out["reported_lists"] = dict(hidden_er_clients=len(cat["hidden_er_clients"]), mito_rna_without_growing_protein=len(cat["mito_rna_without_growing_protein"]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "erloc_tests.json"), "w"), indent=1, default=float)
    json.dump(cat, open(os.path.join(OUT, "erloc_lists.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "erloctest:G12", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    facts = [Fact(f"CLAIM-{cid}", "erloc_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS]
    for x in cat["hidden_er_clients"]:
        facts.append(Fact(x["gene"], "rna_location/er_membrane_without_er_tag", x, t[:10], t, "derived", src, inputs=inp, context={"cell": "HEK293T", "data": "APEX-seq"}))
    for x in cat["mito_rna_without_growing_protein"]:
        facts.append(Fact(x["gene"], "rna_location/mito_surface_without_growing_protein", x, t[:10], t, "derived", src, inputs=inp,
                          context={"cell": "HEK293T", "data": "APEX-seq + puromycin", "meaning": "stays at the mitochondria when the growing protein is pulled off"}))
    cbb.bank().add_many(facts)
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
