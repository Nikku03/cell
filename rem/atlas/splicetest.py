"""SPLICETEST (G11) -- what governs which RNA version (isoform) a gene makes, checked with real knockdowns.

Asked by the user: RNA maturation -- "we need when and in what condition and what governs them"; then "yes start #1".
G9 described, for 189 genes, the tissues where the address-changing RNA version is used most and the splicing proteins
whose RNA follows it across 40 tissues (a correlation). Here the knockdowns test cause: ENCODE knocked down ~450
RNA-binding proteins one at a time in K562 and HepG2 (SPLICE_KD_ENCODE; two replicates each, matched control-shRNA
experiments). Splicing proteins = Reactome "mRNA Splicing" (Major and Minor pathway) members; other knocked-down RNA-binding
proteins are the comparison.
ADDRESS SHARE of a gene = TPM of its isoforms whose alternative sequence touches the address tag (ADDR_UNIPROT, isoforms
named by number) / TPM of all its UniProt isoforms (Ensembl transcripts summed per isoform). A gene is measured in a
knockdown when its isoforms sum to >= 5 TPM in both the knockdown and its controls, and the knocked-down protein is not
the gene itself. CHANGE = mean share in the knockdown replicates - mean share in its control replicates.
No model is fit, so all genes are used (the DEV / EXAM split is not needed); the trial used DEV genes only and showed
only controls.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G11-*):
 G11-K1 SPLICING PROTEINS MOVE THE ADDRESS VERSION. For each gene and cell line with >= 5 splicing-protein knockdowns
        and >= 5 other knockdowns measured: gain = AUROC(|change| under splicing-protein knockdowns vs under other
        knockdowns) - 0.5. CLAIM: BETTER (sign test p < 0.05, median gain >= +0.01) AND Monte Carlo p < 0.05 (which
        knockdowns count as splicing proteins shuffled among the knockdowns of that line, same number, 1,000 draws; the
        median gain must beat the shuffles). Controls: one shuffle must not be BETTER; a planted effect (+0.05 added to
        |change| under splicing-protein knockdowns) must be BETTER. Reported: the same for genes whose RNA versions do
        not change the tag (share of the versions other than the main one), to see if address versions are special.
 G11-K2 THE TISSUE PATTERN POINTS TO THE GOVERNORS. For the genes whose address share varies across tissues (G9 rule:
        >= 10 tissues with >= 1 nTPM, range >= 0.3): r = correlation across the 40 tissues between the share and each
        splicing protein's RNA (as in G9). If a splicing protein raises the share where it is high, knocking it down
        should lower the share (RNA versions that HPA did not normalise are left out of the sums). For each gene and line
        with >= 10 measured splicing-protein knockdowns: gain =
        Spearman(r, -change). CLAIM: BETTER AND Monte Carlo p < 0.05 (which splicing protein each knockdown hit is shuffled
        among that line's splicing-protein knockdowns, 1,000 draws). Controls: r taken from another gene must not be
        BETTER; a planted change (r + noise of the same spread) must be BETTER.
 REPORTED (not a claim): the knockdowns that move each gene's address share by >= 0.10 consistently (both knockdown
        replicates beyond both control replicates), per line -- the measured governors -- and how often that happens
        for splicing proteins vs other RNA-binding proteins.
NOT: shRNA knockdowns are partial and can act indirectly (a hit is "governs, directly or not, in that cell line");
two cell lines only; isoform levels are estimated from short reads (kallisto), which is noisy for similar isoforms;
the Reactome list misses some splicing regulators.

Run: python3 -I rem/atlas/splicetest.py declare | test [--trial]
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
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


gx = _load("generegx"); cf = _load("cellfix")
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "splice")
CLAIMS = ["G11-K1", "G11-K2"]
SEED = 20261125
N_MC = 50 if TRIAL else 1000
MIN_TPM = 5.0
LINES = ("K562", "HepG2")
H_ = cf.H_


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "splicetest.py"), "rb").read())
    for n in ("SPLICE_KD_ENCODE", "ADDR_UNIPROT", "ISOFORM_HPA_TISSUE", "SETS_PATHWAY"):
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def splicing_genes():
    pw = json.load(open(gx.L("SETS_PATHWAY"))); spl = set()
    for v in pw.values():
        if v["name"] in ("mRNA Splicing", "mRNA Splicing - Major Pathway", "mRNA Splicing - Minor Pathway"):
            spl |= set(v["members"])
    return spl


def load():
    z = gx.npz("SPLICE_KD_ENCODE"); T = z["tpm"].astype(np.float64); iso = z["isoform"].astype(str); gene = z["gene"].astype(str)
    cols = json.loads(str(z["meta"])); P = json.load(open(gx.L("ADDR_UNIPROT"))); H = cbb.hgnc()
    sym2e = lambda s: H["sym"].get(s) or H["ali"].get(s)
    rows = collections.defaultdict(list)
    for i, g in enumerate(gene):
        rows[g].append(i)
    # experiments: knockdowns with >= 2 replicates and controls with files
    byexp = collections.defaultdict(list)
    for j, c in enumerate(cols):
        byexp[c["experiment"]].append(j)
    exps = {ln: [] for ln in LINES}
    for e, js in byexp.items():
        c = cols[js[0]]
        if c["role"] != "knockdown" or len(js) < 2:
            continue
        cj = [j for ce in c["controls"] for j in byexp.get(ce, [])]
        if len(cj) >= 2:
            exps[c["line"]].append(dict(exp=e, target=c["target"], target_ens=sym2e(c["target"]), kd=js, ctl=cj))
    for ln in LINES:
        exps[ln].sort(key=lambda x: x["exp"])
    return T, iso, gene, rows, cols, exps, P


def shares(T, rows, P, iso):
    """per gene: (address-share numerator rows, all rows) for genes with tag-changing isoforms named by number; and for
    genes without, the rows of all isoforms (the 'other versions' share is fixed later from the controls)."""
    tag, other = {}, {}
    for g, rr in rows.items():
        v = P.get(g)
        if v is None or len(rr) < 2:
            continue
        ch = v["isoforms_changing_tag"]
        if ch:
            if any(not k.isdigit() for k in ch):
                continue
            want = {f"{v['uniprot']}-{k}" for k in ch}; num = [i for i in rr if iso[i] in want]
            if num and len(num) < len(rr):
                tag[g] = (num, rr)
        elif v["has_var_seq"]:
            other[g] = rr
    return tag, other


def changes(T, units, exps_ln, exclude_self=True):
    """units: {gene: (num rows, all rows)} -> D [genes x experiments] change in share (NaN if not measured), plus the
    per-replicate shares for the consistency rule."""
    G = sorted(units); D = np.full((len(G), len(exps_ln)), np.nan); cons = np.zeros_like(D, bool)
    for gi, g in enumerate(G):
        num, al = units[g]; tot = T[al].sum(0); sh = np.where(tot > 0, T[num].sum(0) / np.maximum(tot, 1e-12), np.nan)
        for ei, e in enumerate(exps_ln):
            if exclude_self and e["target_ens"] == g:
                continue
            if tot[e["kd"]].mean() < MIN_TPM or tot[e["ctl"]].mean() < MIN_TPM:
                continue
            a, b = sh[e["kd"]], sh[e["ctl"]]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            D[gi, ei] = a.mean() - b.mean(); cons[gi, ei] = (a.min() > b.max()) or (a.max() < b.min())
    return G, D, cons


# ------------------------------------------------------------------------------------------------ K1
def k1_gains(D, lab, plant=0.0):
    from scipy.stats import rankdata
    out = []
    for row in D:
        m = np.isfinite(row); s, o = m & lab, m & ~lab
        if s.sum() < 5 or o.sum() < 5:
            out.append(np.nan); continue
        x = np.abs(row[m]) + plant * lab[m]; r = rankdata(x); l = lab[m]; n1, n0 = l.sum(), (~l).sum()
        out.append((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5)
    return np.array(out)


def run_k1(rng, units_by_line, exps, spl, label="address"):
    gains, nulls, negs, plants, per_line = [], np.zeros(N_MC), [], [], {}
    mats = {}
    for ln in LINES:
        G, D, cons = units_by_line[ln]; lab = np.array([e["target_ens"] in spl for e in exps[ln]])
        mats[ln] = (D, lab); g = k1_gains(D, lab); gains.append(g)
        negs.append(k1_gains(D, lab[rng.permutation(len(lab))]))
        plants.append(k1_gains(D, lab, plant=0.05))
        per_line[ln] = dict(units=int(np.isfinite(g).sum()), splicing_knockdowns=int(lab.sum()), other_knockdowns=int((~lab).sum()),
                            median_gain=float(np.nanmedian(g)) if np.isfinite(g).any() else float("nan"))
    real = np.concatenate(gains); b = cf.better(real[np.isfinite(real)])
    for k in range(N_MC):
        nl = []
        for ln in LINES:
            D, lab = mats[ln]; nl.append(k1_gains(D, lab[rng.permutation(len(lab))]))
        x = np.concatenate(nl); nulls[k] = np.nanmedian(x)
    pm = float((1 + (nulls >= b["median"]).sum()) / (1 + N_MC))
    neg = cf.better(np.concatenate(negs)[np.isfinite(np.concatenate(negs))]); pl = cf.better(np.concatenate(plants)[np.isfinite(np.concatenate(plants))])
    ok = bool(b["passed"] and pm < 0.05)
    out = dict(versions=label, per_line=per_line, splicing_vs_other=b, p_mc=pm, null_median_95=float(np.percentile(nulls, 95)),
               negative=neg, planted=pl)
    out["gate"] = dict(real=ok, controls_ok=bool(not neg["passed"] and pl["passed"]), verdict=("PASS" if ok else "FAIL") if (not neg["passed"] and pl["passed"]) else "VOID (controls)")
    return out


# ------------------------------------------------------------------------------------------------ K2
def tissue_r(P, genes, spl):
    """G9's rule: share of tag-changing versions across 40 tissues (HPA transcripts), and its correlation with each
    splicing protein's RNA."""
    z = gx.npz("ISOFORM_HPA_TISSUE"); tg = z["gene"].astype(str); tt = z["transcript"].astype(str)
    X = np.nan_to_num(z["X"].astype(np.float64))     # versions HPA did not normalise (nTPM NA in every tissue) count as absent,
    row = collections.defaultdict(list)                # instead of blanking the whole gene (the G9 description did the latter)
    for i, (g, t) in enumerate(zip(tg, tt)):
        row[g].append((t, i))
    gexp = {g: X[[i for _, i in row[g]]].sum(0) for g in spl if g in row}
    out = {}
    for g in genes:
        v = P[g]; ch = v.get("isoforms_changing_tag")
        if not ch or g not in row:
            continue
        acc = v["uniprot"]; iso_of = {t: i for t, i in v["transcripts"]}; changed = {f"{acc}-{k}" for k in ch if k.isdigit()}
        tx = [(t, i) for t, i in row[g] if t in iso_of]
        ch_rows = [i for t, i in tx if iso_of[t] in changed]; all_rows = [i for t, i in tx]
        if not tx or not changed or not ch_rows:
            continue
        tot = X[all_rows].sum(0); frac = np.where(tot >= 1, X[ch_rows].sum(0) / np.maximum(tot, 1e-9), np.nan)
        if np.isfinite(frac).sum() < 10 or np.nanmax(frac) - np.nanmin(frac) < 0.3:
            continue
        ok = np.isfinite(frac); r = {}
        for s_, e_ in gexp.items():
            if e_[ok].std() > 0:
                r[s_] = float(np.corrcoef(frac[ok], np.log2(e_[ok] + 1))[0, 1])
        out[g] = r
    return out


def k2_gains(Dsub, Rsub, min_n=10):
    from scipy.stats import spearmanr
    g = []
    for d, r in zip(Dsub, Rsub):
        m = np.isfinite(d) & np.isfinite(r)
        g.append(spearmanr(r[m], -d[m])[0] if m.sum() >= min_n and np.std(r[m]) > 0 and np.std(d[m]) > 0 else np.nan)
    return np.array(g)


def run_k2(rng, units_by_line, exps, spl, P):
    G_all = sorted(set().union(*[set(units_by_line[ln][0]) for ln in LINES]))
    TR = tissue_r(P, G_all, spl); genes = sorted(TR)
    real, negs, plants, nulls, per_line, mats = [], [], [], np.zeros(N_MC), {}, {}
    for ln in LINES:
        G, D, cons = units_by_line[ln]; gi = {g: i for i, g in enumerate(G)}
        sj = [j for j, e in enumerate(exps[ln]) if e["target_ens"] in spl]; tgt = [exps[ln][j]["target_ens"] for j in sj]
        gs = [g for g in genes if g in gi]
        if not gs or not sj:
            per_line[ln] = dict(units=0); continue
        Dsub = D[[gi[g] for g in gs]][:, sj]
        Rm = np.array([[TR[g].get(t, np.nan) for t in tgt] for g in gs])
        mats[ln] = (Dsub, Rm, gs, tgt)
        g_ = k2_gains(Dsub, Rm); real.append(g_)
        perm = rng.permutation(len(gs))
        while len(gs) > 1 and (perm == np.arange(len(gs))).any():
            perm = rng.permutation(len(gs))
        Rneg = Rm[perm]; negs.append(k2_gains(Dsub, Rneg))
        sd = np.nanstd(Rm); Dpl = -(Rm + rng.normal(0, sd if sd > 0 else 1, Rm.shape)); Dpl[~np.isfinite(Dsub)] = np.nan
        plants.append(k2_gains(Dpl, Rm))
        per_line[ln] = dict(units=int(np.isfinite(g_).sum()), splicing_knockdowns=len(sj), median_rho=float(np.nanmedian(g_)) if np.isfinite(g_).any() else float("nan"))
    if not real:
        return dict(gate=dict(real=False, controls_ok=False, verdict="VOID (no data)"))
    x = np.concatenate(real); b = cf.better(x[np.isfinite(x)])
    for k in range(N_MC):
        nl = []
        for ln, (Dsub, Rm, gs, tgt) in mats.items():
            nl.append(k2_gains(Dsub, Rm[:, rng.permutation(Rm.shape[1])]))
        y = np.concatenate(nl); nulls[k] = np.nanmedian(y)
    pm = float((1 + (nulls >= b["median"]).sum()) / (1 + N_MC))
    nx = np.concatenate(negs); px = np.concatenate(plants)
    neg = cf.better(nx[np.isfinite(nx)]); pl = cf.better(px[np.isfinite(px)])
    ok = bool(b["passed"] and pm < 0.05)
    out = dict(genes_tissue_varying=len(genes), per_line=per_line, rho=b, p_mc=pm, null_median_95=float(np.percentile(nulls, 95)), negative=neg, planted=pl)
    out["gate"] = dict(real=ok, controls_ok=bool(not neg["passed"] and pl["passed"]), verdict=("PASS" if ok else "FAIL") if (not neg["passed"] and pl["passed"]) else "VOID (controls)")
    return out


# ------------------------------------------------------------------------------------------------ catalogue
def catalogue(units_by_line, exps, spl):
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); cat = collections.defaultdict(list); rate = {}
    for ln in LINES:
        G, D, cons = units_by_line[ln]; lab = np.array([e["target_ens"] in spl for e in exps[ln]])
        hit = np.isfinite(D) & (np.abs(np.nan_to_num(D)) >= 0.10) & cons
        meas = np.isfinite(D)
        rate[ln] = dict(splicing=float(hit[:, lab].sum() / max(meas[:, lab].sum(), 1)), other=float(hit[:, ~lab].sum() / max(meas[:, ~lab].sum(), 1)),
                        hits=int(hit.sum()), measured_pairs=int(meas.sum()))
        for gi, ei in zip(*np.nonzero(hit)):
            e = exps[ln][ei]
            cat[G[gi]].append(dict(line=ln, knocked_down=e["target"], splicing_protein=bool(lab[ei]), change=round(float(D[gi, ei]), 3), experiment=e["exp"]))
    out = {}
    for g, v in cat.items():
        v.sort(key=lambda x: -abs(x["change"])); out[sym(g)] = dict(gene=g, governors=v[:20], n=len(v))
    return out, rate


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
                                    claim=__doc__.split(" " + cid)[1].split(" G11-")[0].split("REPORTED")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G11 already tested")
    t0 = time.time(); rng = np.random.default_rng(SEED)
    T, iso, gene, rows, cols, exps, P = load(); spl = splicing_genes()
    tag, other = shares(T, rows, P, iso)
    if TRIAL:
        tag = {g: v for g, v in tag.items() if H_(g) == 0}
    # 'other versions' share: all isoforms except the one most used in all controls
    ctl_cols = sorted({j for ln in LINES for e in exps[ln] for j in e["ctl"]})
    oth = {}
    for g, rr in other.items():
        main = rr[int(np.argmax(T[rr][:, ctl_cols].sum(1)))]; oth[g] = ([i for i in rr if i != main], rr)
    if TRIAL:
        oth = {g: v for g, v in oth.items() if H_(g) == 0}
    U = {ln: changes(T, tag, exps[ln]) for ln in LINES}; Uo = {ln: changes(T, oth, exps[ln]) for ln in LINES}
    print(f"measured: {', '.join(f'{ln} {np.isfinite(U[ln][1]).any(1).sum()} genes x {len(exps[ln])} knockdowns' for ln in LINES)} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    out["G11-K1"] = run_k1(rng, U, exps, spl); print("G11-K1 done", f"({time.time() - t0:.0f}s)", flush=True)
    out["G11-K1"]["reported_other_versions"] = {k: v for k, v in run_k1(rng, Uo, exps, spl, label="other versions").items() if k != "gate"}
    out["G11-K2"] = run_k2(rng, U, exps, spl, P); print("G11-K2 done", f"({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        for cid in CLAIMS:
            g = out[cid]["gate"]; print(cid, "controls", "ok" if g["controls_ok"] else "NOT ok")
        print("trial ran; outcomes not shown"); return
    cat, rate = catalogue(U, exps, spl); out["reported_measured_governors"] = dict(genes_with_hits=len(cat), hit_rate=rate)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "splice_tests.json"), "w"), indent=1, default=float)
    json.dump(cat, open(os.path.join(OUT, "governors_measured.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "splicetest:G11", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    facts = [Fact(f"CLAIM-{cid}", "splice_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS]
    for s_, v in cat.items():
        facts.append(Fact(v["gene"], "isoform/governors_measured", v, t[:10], t, "derived", src, inputs=inp,
                          context={"rule": "ENCODE shRNA knockdown moves the address share by >= 0.10, both replicates beyond both controls; direct or indirect"}))
    cbb.bank().add_many(facts)
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
