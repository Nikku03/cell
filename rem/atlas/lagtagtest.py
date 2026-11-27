"""LAGTAGTEST (G15) -- does our regulator map predict the next state of a cell? Mouse embryonic stem cells switch on
Gata4 (Takei et al. 2026, LagTag paper; GEO GSE305653): which genes gain RNA polymerase activity, and did our map
(CollecTRI, human, carried to mouse by one-to-one orthologs) say so in advance?

Asked by the user: "do the small test with their mouse data".
Data: LAGTAG_GATA4 (RNA Pol II Ser2p CUT&Tag, 6 samples before and 6 after 24 h of Gata4 + LIF withdrawal: parental
and recording-clone lines x 3 replicates; plus LagTag recorded-before / endpoint-after in the same cells, 3
replicates); REG_COLLECTRI (human TF -> target, sign; local only); MGI one-to-one orthologs.
Genes: mouse protein-coding genes with a one-to-one human ortholog. Per sample l = log2(signal + 0.1) (signal = mean
CPM over gene body + 3 kb); change = mean l after - mean l before; Welch t-test (6 vs 6), Benjamini-Hochberg q. A gene
SWITCHES ON when change >= 1 and q < 0.05. Baseline = mean l before, in ten equal-count strata.
No model is fit (all genes used); the trial used DEV genes (sha256('cellbrain-c1:' + human Ensembl) parity 0) and
showed only controls.

CLAIMS (declared before running; ids G15-*):
 G15-N1 GATA4'S OWN TARGETS SWITCH ON. Activating CollecTRI targets of GATA4 vs the rest: Mantel-Haenszel odds ratio
        of switching on across baseline strata >= 2, one-sided p < 0.05, AND better than other regulators: GATA4's odds
        ratio is above 95% of those of every other TF with >= 10 activating targets mapped (p_TF < 0.05), AND above
        95% of 1,000 random gene sets of the same size and baseline make-up (p_MC < 0.05).
 G15-N2 THE NEXT STEP. Two-step targets: activating targets of TFs that GATA4 activates, excluding GATA4's direct
        targets and those TFs themselves. Same rule with odds ratio >= 1.5 (two-step sets of every other TF with >= 10
        two-step targets as the TF comparison; 1,000 matched random sets).
 Controls (for both): a random matched gene set of the claim's size must not pass the odds-ratio rule; a planted set
        (half drawn from genes that switch on) must pass.
 REPORTED: number of genes switching on / off; the same tests with switching defined from the LagTag same-cell pair
        (endpoint Ser2p vs recorded m6A, top 5% of change); correlation of the LagTag change with the CUT&Tag change.
NOT: Gata4 is switched on together with LIF withdrawal, so changes include leaving pluripotency; CollecTRI is human
and mostly from other tissues; one-to-one orthologs only; RNA Pol II on a gene is a read-out of activity, not RNA or
protein; one time point (24 h).

Run: python3 -I rem/atlas/lagtagtest.py declare | test [--trial]
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


gx = _load("generegx")
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "lagtag")
CLAIMS = ["G15-N1", "G15-N2"]
SEED = 20261127
N_MC = 50 if TRIAL else 1000
PSEUDO = 0.1
H_ = lambda e: int(hashlib.sha256(("cellbrain-c1:" + e).encode()).hexdigest(), 16) % 2


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "lagtagtest.py"), "rb").read())
    for n in ("LAGTAG_GATA4", "REG_COLLECTRI"):
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = np.empty(len(p)); n = len(p)
    q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]; r[o] = np.minimum(q, 1); return r


def load():
    from scipy.stats import ttest_ind
    z = gx.npz("LAGTAG_GATA4"); X = z["X"].astype(float); grp = z["group"].astype(str); hum = z["human"].astype(str)
    keep = np.array([bool(h) for h in hum])
    if TRIAL:
        keep &= np.array([bool(h) and H_(h) == 0 for h in hum])
    X, hum = X[keep], hum[keep]; L = np.log2(X + PSEUDO)
    a, b = L[:, grp == "after"], L[:, grp == "before"]
    change = a.mean(1) - b.mean(1); t = ttest_ind(a, b, axis=1, equal_var=False); p = np.nan_to_num(t.pvalue, nan=1.0); q = bh(p)
    on = (change >= 1) & (q < 0.05); off = (change <= -1) & (q < 0.05)
    base = b.mean(1); strata = np.searchsorted(np.quantile(base, np.linspace(0.1, 0.9, 9)), base)
    lt = L[:, grp == "lagtag_endpoint_after"].mean(1) - L[:, grp == "lagtag_recorded_before"].mean(1)
    on_lt = lt >= np.quantile(lt, 0.95)
    return dict(hum=hum, change=change, on=on, off=off, strata=strata, lt=lt, on_lt=on_lt)


def network(hum):
    z = gx.npz("REG_COLLECTRI"); tf = z["tf"].astype(str); tg = z["target"].astype(str); sg = z["sign"]; tfs = z["tf_symbol"].astype(str)
    gi = {h: i for i, h in enumerate(hum)}; act = collections.defaultdict(set); sym = {}
    for a, b, s, n in zip(tf, tg, sg, tfs):
        sym[a] = n
        if s > 0:
            act[a].add(b)
    direct = {t: {gi[x] for x in v if x in gi} for t, v in act.items()}
    two = {}
    for t, v in act.items():
        s2 = set()
        for x in v:
            s2 |= act.get(x, set())
        two[t] = {gi[x] for x in s2 - v - {t} - {y for y in v if y in act} if x in gi}
    return direct, two, sym


def cmh(member, outcome, strata):
    """Mantel-Haenszel odds ratio and one-sided p (more outcome among members)."""
    from scipy.stats import norm
    num = den = 0.0; ea = va = obs = 0.0
    for s in np.unique(strata):
        m = strata == s; x, y = member[m], outcome[m]; n = m.sum()
        a = float((x & y).sum()); b = float((x & ~y).sum()); c = float((~x & y).sum()); d = float((~x & ~y).sum())
        if n < 2:
            continue
        num += a * d / n; den += b * c / n
        r1, c1 = a + b, a + c; obs += a; ea += r1 * c1 / n; va += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1)) if n > 1 else 0
    OR = (num + 0.5) / (den + 0.5)
    z = (obs - ea - 0.5) / np.sqrt(va) if va > 0 else 0.0
    return float(OR), float(1 - norm.cdf(z))


def matched_random(idx, strata, rng):
    out = []
    for s, k in collections.Counter(strata[list(idx)]).items():
        pool = np.nonzero(strata == s)[0]; out += list(rng.choice(pool, min(k, len(pool)), replace=False))
    return set(out)


def test_set(name, S, others, D, outcome, bar, rng):
    n = len(D["hum"]); mem = np.zeros(n, bool); mem[list(S)] = True
    OR, p = cmh(mem, outcome, D["strata"])
    tfOR = []
    for t, s2 in others.items():
        if len(s2) >= 10:
            m2 = np.zeros(n, bool); m2[list(s2)] = True; tfOR.append(cmh(m2, outcome, D["strata"])[0])
    tfOR = np.array(tfOR); p_tf = float((1 + (tfOR >= OR).sum()) / (1 + len(tfOR)))
    null = []
    for _ in range(N_MC):
        r = matched_random(S, D["strata"], rng); m2 = np.zeros(n, bool); m2[list(r)] = True; null.append(cmh(m2, outcome, D["strata"])[0])
    null = np.array(null); p_mc = float((1 + (null >= OR).sum()) / (1 + N_MC))
    # controls
    r = matched_random(S, D["strata"], rng); m2 = np.zeros(n, bool); m2[list(r)] = True; ORn, pn = cmh(m2, outcome, D["strata"])
    on_idx = np.nonzero(outcome)[0]; k = len(S)
    pl = set(rng.choice(on_idx, min(k // 2, len(on_idx)), replace=False)) | set(rng.choice(np.nonzero(~outcome)[0], k - k // 2, replace=False))
    m3 = np.zeros(n, bool); m3[list(pl)] = True; ORp, pp = cmh(m3, outcome, D["strata"])
    neg_pass = bool(ORn >= bar and pn < 0.05); pl_pass = bool(ORp >= bar and pp < 0.05)
    real = bool(OR >= bar and p < 0.05 and p_tf < 0.05 and p_mc < 0.05); ok = (not neg_pass) and pl_pass
    return dict(genes=int(mem.sum()), switching_on_in_set=int((mem & outcome).sum()), share_on_in_set=float((mem & outcome).sum() / max(mem.sum(), 1)),
                share_on_elsewhere=float((~mem & outcome).sum() / max((~mem).sum(), 1)), odds_ratio=OR, p=p, p_tf=p_tf, tfs_compared=int(len(tfOR)),
                tf_or_95=float(np.percentile(tfOR, 95)) if len(tfOR) else None, p_mc=p_mc, random_or_95=float(np.percentile(null, 95)),
                negative_control=dict(odds_ratio=ORn, p=pn, passed=neg_pass), planted_control=dict(odds_ratio=ORp, p=pp, passed=pl_pass),
                gate=dict(real=real, controls_ok=ok, verdict=("PASS" if real else "FAIL") if ok else "VOID (controls)"))


def run(rng):
    D = load(); direct, two, sym = network(D["hum"]); g4 = next(t for t, n in sym.items() if n == "GATA4")
    out = {}
    out["G15-N1"] = test_set("direct", direct[g4], {t: s for t, s in direct.items() if t != g4}, D, D["on"], 2.0, rng)
    out["G15-N2"] = test_set("two-step", two[g4], {t: s for t, s in two.items() if t != g4}, D, D["on"], 1.5, rng)
    rep = dict(genes=int(len(D["hum"])), switching_on=int(D["on"].sum()), switching_off=int(D["off"].sum()),
               lagtag_vs_cuttag_change_r=float(np.corrcoef(D["lt"], D["change"])[0, 1]))
    for key, S in (("direct", direct[g4]), ("two_step", two[g4])):
        mem = np.zeros(len(D["hum"]), bool); mem[list(S)] = True; rep[f"lagtag_same_cell_{key}"] = dict(zip(("odds_ratio", "p"), cmh(mem, D["on_lt"], D["strata"])))
    sym_on = sorted(gx.cbb.hgnc()["e2s"].get(D["hum"][i], D["hum"][i]) for i in direct[g4] if D["on"][i])
    rep["gata4_targets_switching_on"] = sym_on
    return out, rep


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G15-")[0].split("Controls (for both)")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G15 already tested")
    t0 = time.time(); out, rep = run(np.random.default_rng(SEED))
    if TRIAL:
        for cid in CLAIMS:
            g = out[cid]; print(cid, "controls", "ok" if g["gate"]["controls_ok"] else "NOT ok", "(negative passed:", g["negative_control"]["passed"], ", planted passed:", g["planted_control"]["passed"], ")")
        print("trial ran; outcomes not shown"); return
    os.makedirs(OUT, exist_ok=True); json.dump(dict(out, reported=rep), open(os.path.join(OUT, "lagtag_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "lagtagtest:G15", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "lagtag_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
