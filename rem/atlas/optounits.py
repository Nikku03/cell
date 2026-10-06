"""Real neurons' input -> output curves under graded optogenetic drive: can they repair our node?

THE WEAK PART THIS TARGETS. netprobe.py and nonbio.py: the channel tree's node,
    out = v + g_dep m (1 - v) + g_hyp m (-1 - v),   ONE gate m = sigmoid(v - theta),
loses by ~0.17 to generic bends (sine, bump) of identical size, 10/10 seeds. Its depolarising and
hyperpolarising conductances share ONE gate, so they always open TOGETHER; the node cannot place its
rise and its fall at different input levels. A Gaussian bump can (free centre and width).
In real neurons the conductances that push up and push down have SEPARATE gates with DIFFERENT
half-activation points (Hodgkin-Huxley: sodium m and potassium n), and in circuits excitation and
inhibition are recruited at different drive strengths. Which comes first is an empirical question.

THE DATA. DANDI 000060 (Finkelstein, Fontolan ... Svoboda 2021): spikes of 4,335 ALM and 829 vS1
units while vS1 was driven by 0.4-s channelrhodopsin pulses at several light powers. On many sessions
the SAMPLE pulse (2.5 s before the go cue) came at 2-4 powers; with pulse-free trials that is an
input -> output curve for every recorded neuron, measured causally.

Written after a STRUCTURE-ONLY census (areas, cell types, unit quality, powers per session, presence
of go-cue labels). No spike counts or responses had been computed when this was committed.

=================================================================================================
GATES AND THE TRANSLATION RULE, PREDECLARED
=================================================================================================
SELECTION. Units of quality 'good' or 'ok'. Trials: no early lick, outcome hit or miss. Level 0:
pure 'l' trials (no pulse at all). Stimulated: exactly one pulse, at 2.5 +/- 0.02 s before the go cue,
not 'NoAudCue'. Drive x = power / session Full power (modal power of standard 'r' trials).
Response = spikes/s in [t0, t0 + 0.5 s), t0 = go - 2.5 s (the same window on level-0 trials).
A unit enters the curve analysis with >= 5 trials at level 0, at x = 1 and at >= 1 interior level.

U0 ALIGNMENT, BLOCKING. Go cue found for >= 99% of kept trials; >= 95% of stimulated trials have a
   PhotostimEvents pulse within 0.02 s of go - 2.5 s whose recorded power equals the trial's.
U1 POSITIVE CONTROL, BLOCKING. >= 20% of vS1 units significantly EXCITED at x = 1 vs x = 0 (Welch,
   p < 0.01). vS1 is where the light lands; if it does not respond, the alignment is wrong.
U2 CENSUS. Units, responsive fractions per area and cell type.
U3 SHAPE. Responsive = Welch p < 0.01 for x = 1 vs x = 0; sign s = direction of that change.
   TUNED (non-monotone) if the interior level with the largest s-signed mean beats x = 1 in the
   s direction (one-sided Welch p < 0.05) AND the same holds in both odd and even trial halves.
   Null: the same test after permuting drive labels among stimulated trials (20 permutations per
   unit); tuned fraction is reported against that false-positive rate.
U4 ORDER OF RECRUITMENT -- THE NUMBER THE TRANSLATION USES. For responsive ALM units, x50 = drive at
   which the normalised response (R(x) - R(0)) / (R(1) - R(0)) first reaches 0.5 (linear
   interpolation; 0 at x = 0, 1 at x = 1). Compare SUPPRESSED (s < 0) with EXCITED (s > 0):
   two-sided Mann-Whitney p < 0.05 AND the same direction in a majority of mice having >= 5 units
   of each kind.
U5 STEEPNESS AND THE ANIMAL. Median normalised response at x = 1/3; behavioural P(lick right) vs x.

TRANSLATION RULE (fixed now, applied mechanically by twogate.py from outputs/opto_units.json):
   The node is given SEPARATE gates for its two conductances,
       out = v + g_dep sigma(v - theta_dep)(1 - v) + g_hyp sigma(v - theta_hyp)(-1 - v),
   initialised theta_dep = 0 and theta_hyp = ORDER * 1.0, where
       ORDER = +1  if U4 says suppression is recruited at HIGHER drive than excitation,
       ORDER = -1  if at LOWER drive,
       ORDER =  0  if U4 does not pass (no ordering signal: the gates are untied but start equal).
   Only the DIRECTION comes from the data; the magnitude 1.0 is a fixed convention, because light
   power and the network's v are not on a common scale.

U6 WHAT THIS IS AND IS NOT. In vivo units are embedded in circuits: a unit's curve is the circuit's
   response, not its membrane's. The rule borrows an ORDERING from the circuit, not a channel model.
"""

from __future__ import annotations
import collections
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CACHE = os.path.join(HERE, "_cache", "opto_000060")
OUT = os.path.join(HERE, "RESULTS_optounits.txt")
ART = R("outputs", "opto_units.json")
RULE = "=" * 97
WIN = 0.5
T_SAMPLE = 2.5
NPERM = 20


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a[()]]


def session(path):
    """Per-unit trial responses at the sample window, plus alignment bookkeeping."""
    import h5py
    f = h5py.File(path, "r")
    t = f["intervals/trials"]
    st, sp = t["start_time"][()], t["stop_time"][()]
    tt, on, pw = dec(t["trial_type_name"]), dec(t["photostim_onset"]), dec(t["photostim_power"])
    out_, el = dec(t["outcome"]), dec(t["early_lick"])
    le = f["acquisition/LabeledEvents"]
    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in le["data"].attrs["labels"]]
    gi = labels.index("go_start_times")
    go_all = np.sort(le["timestamps"][()][le["data"][()] == gi])
    pe = f["acquisition/PhotostimEvents/photostim_start_times"]
    p_t, p_w = pe["timestamps"][()], pe["data"][()]
    rpow = collections.Counter(pw[i].split(",")[0].strip() for i in range(len(tt)) if tt[i] == "r")
    full = float(rpow.most_common(1)[0][0]) if rpow else None
    trials, n_keep, n_go, n_stim, n_match = [], 0, 0, 0, 0
    for i in range(len(tt)):
        if el[i] != "no early" or out_[i] not in ("hit", "miss"):
            continue
        n_keep += 1
        g = go_all[(go_all >= st[i]) & (go_all <= sp[i])]
        if len(g) == 0:
            continue
        n_go += 1
        go = g[0]; t0 = go - T_SAMPLE
        o = [x.strip() for x in on[i].split(",")]; p = [x.strip() for x in pw[i].split(",")]
        if tt[i] == "l" and o == ["N/A"]:
            trials.append(dict(t0=t0, x=0.0, choice=out_[i] == "miss"))
        elif (len(o) == 1 and o[0] not in ("N/A", "") and abs(float(o[0]) - T_SAMPLE) <= 0.02
              and "NoAudCue" not in tt[i] and full):
            n_stim += 1
            j = np.argmin(np.abs(p_t - t0)) if len(p_t) else None
            if j is not None and abs(p_t[j] - t0) <= 0.02 and abs(p_w[j] - float(p[0])) < 1e-6:
                n_match += 1
            right = (out_[i] == "hit") if tt[i].startswith("r") else (out_[i] == "miss")
            trials.append(dict(t0=t0, x=round(float(p[0]) / full, 2), choice=right))
    u = f["units"]
    idx = u["spike_times_index"][()]
    spk = u["spike_times"][()]
    qual, ctyp = dec(u["quality"]), dec(u["cell_type"])
    units = []
    starts = np.concatenate([[0], idx[:-1]])
    t0s = np.array([tr["t0"] for tr in trials]); xs = np.array([tr["x"] for tr in trials])
    for k in range(len(idx)):
        area = json.loads(f[u["electrode_group"][k]].attrs["location"]).get("brain_area")
        if qual[k] not in ("good", "ok") or not len(trials):
            continue
        s = spk[starts[k]:idx[k]]
        cnt = np.searchsorted(s, t0s + WIN) - np.searchsorted(s, t0s)
        units.append(dict(area=area, cell=ctyp[k], rate=cnt / WIN, x=xs))
    f.close()
    return dict(units=units, trials=trials, n_keep=n_keep, n_go=n_go, n_stim=n_stim,
                n_match=n_match, full=full)


def curve(rate, x):
    lv = sorted(set(x.tolist()))
    return {v: rate[x == v] for v in lv}


def classify(rate, x, rng=None):
    from scipy import stats
    c = curve(rate, x)
    if 0.0 not in c or 1.0 not in c or len(c[0.0]) < 5 or len(c[1.0]) < 5:
        return None
    inner = [v for v in c if 0 < v < 1 and len(c[v]) >= 5]
    if not inner:
        return None
    p = stats.ttest_ind(c[1.0], c[0.0], equal_var=False).pvalue
    d = c[1.0].mean() - c[0.0].mean()
    res = dict(responsive=bool(p < 0.01 and d != 0), sign=int(np.sign(d)), p=float(p),
               levels={v: float(c[v].mean()) for v in c})
    if not res["responsive"]:
        return res
    s = res["sign"]

    def tuned_on(cc):
        pk = max(inner, key=lambda v: s * cc[v].mean())
        a, b = s * cc[pk], s * cc[1.0]
        if len(a) < 2 or len(b) < 2 or a.mean() <= b.mean():
            return False
        return stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue < 0.05

    def halves(cc):
        ok = True
        for h in (0, 1):
            hc = {v: cc[v][h::2] for v in cc}
            pk = max(inner, key=lambda v: s * hc[v].mean())
            ok &= s * hc[pk].mean() > s * hc[1.0].mean()
        return ok

    res["tuned"] = bool(tuned_on(c) and halves(c))
    if rng is not None:
        stim = x > 0
        fp = 0
        for _ in range(NPERM):
            xp = x.copy(); xp[stim] = rng.permutation(x[stim])
            cp = curve(rate, xp)
            fp += tuned_on(cp) and halves(cp)
        res["null_rate"] = fp / NPERM
    lv = sorted(c)
    r = [(c[v].mean() - c[0.0].mean()) / d for v in lv]
    x50 = None
    for a in range(1, len(lv)):
        if r[a] >= 0.5:
            x50 = lv[a - 1] + (0.5 - r[a - 1]) / (r[a] - r[a - 1]) * (lv[a] - lv[a - 1]) if r[a] != r[a - 1] else lv[a]
            break
    res["x50"] = float(x50) if x50 is not None else None
    res["norm"] = {v: float(rv) for v, rv in zip(lv, r)}
    return res


def main():
    from scipy import stats
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    man = json.load(open(os.path.join(CACHE, "manifest.json")))
    rng = np.random.default_rng(0)
    P_(RULE); P_("REAL NEURONS' INPUT -> OUTPUT CURVES UNDER GRADED OPTOGENETIC DRIVE"); P_(RULE)
    rows, tot = [], collections.Counter()
    beh = collections.defaultdict(list)
    for m in man:
        S = session(os.path.join(CACHE, m["path"].replace("/", "__")))
        for k in ("n_keep", "n_go", "n_stim", "n_match"):
            tot[k] += S[k]
        mouse = m["path"].split("/")[0]
        if len({tr["x"] for tr in S["trials"]}) >= 3:
            for tr in S["trials"]:
                beh[tr["x"]].append(tr["choice"])
        for u in S["units"]:
            cl = classify(u["rate"], u["x"], rng)
            if cl is not None:
                rows.append(dict(mouse=mouse, session=m["path"], area=u["area"], cell=u["cell"], **cl))
    go_ok = tot["n_go"] / max(tot["n_keep"], 1)
    match_ok = tot["n_match"] / max(tot["n_stim"], 1)
    P_(f"  U0 go cue found {tot['n_go']:,}/{tot['n_keep']:,} kept trials ({go_ok:.4f}); stimulated trials with "
       f"a matching recorded pulse {tot['n_match']:,}/{tot['n_stim']:,} ({match_ok:.4f})")
    if go_ok < 0.99 or match_ok < 0.95:
        P_("  U0: FAIL. Nothing reported."); open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("  U0: PASS")

    def frac(rs, cond):
        return (sum(cond(r) for r in rs), len(rs))

    v1 = [r for r in rows if r["area"] == "vS1"]
    a, n = frac(v1, lambda r: r["responsive"] and r["sign"] > 0)
    P_(f"  U1 vS1 units excited at full drive: {a}/{n} ({a / max(n, 1):.3f})  -> {'PASS' if n and a / n >= 0.20 else 'FAIL'}")
    if not n or a / n < 0.20:
        open(OUT, "w").write("\n".join(out) + "\n"); return

    P_("\n  U2 CENSUS (units entering the curve analysis)")
    for ar in ("vS1", "ALM"):
        for ce in ("Pyr", "FS"):
            rs = [r for r in rows if r["area"] == ar and r["cell"] == ce]
            ex = sum(r["responsive"] and r["sign"] > 0 for r in rs)
            su = sum(r["responsive"] and r["sign"] < 0 for r in rs)
            P_(f"    {ar:<4} {ce:<4} units {len(rs):>5}   excited {ex:>4}   suppressed {su:>4}")

    P_("\n" + RULE); P_("U3  SHAPE: MONOTONE OR TUNED?"); P_(RULE)
    shape = {}
    for ar in ("vS1", "ALM"):
        rs = [r for r in rows if r["area"] == ar and r["responsive"]]
        tn = sum(r["tuned"] for r in rs)
        nul = float(np.mean([r["null_rate"] for r in rs])) if rs else float("nan")
        shape[ar] = dict(responsive=len(rs), tuned=tn, null=nul)
        P_(f"  {ar:<4} responsive {len(rs):>4}   tuned {tn:>4} ({tn / max(len(rs), 1):.3f})   permutation null {nul:.3f}")
        for sg, nm in ((1, "excited"), (-1, "suppressed")):
            q = [r for r in rs if r["sign"] == sg]
            P_(f"         {nm:<10} {len(q):>4}   tuned {sum(r['tuned'] for r in q):>4}")

    P_("\n" + RULE); P_("U4  ORDER OF RECRUITMENT: SUPPRESSION vs EXCITATION (THE TRANSLATED NUMBER)"); P_(RULE)
    alm = [r for r in rows if r["area"] == "ALM" and r["responsive"] and r["x50"] is not None]
    ex = np.array([r["x50"] for r in alm if r["sign"] > 0]); su = np.array([r["x50"] for r in alm if r["sign"] < 0])
    mw = stats.mannwhitneyu(su, ex, alternative="two-sided") if len(ex) and len(su) else None
    P_(f"  ALM x50 excited   n {len(ex):>4}  median {np.median(ex) if len(ex) else float('nan'):.3f}  IQR "
       f"{np.percentile(ex, 25) if len(ex) else float('nan'):.3f}-{np.percentile(ex, 75) if len(ex) else float('nan'):.3f}")
    P_(f"  ALM x50 suppressed n {len(su):>4}  median {np.median(su) if len(su) else float('nan'):.3f}  IQR "
       f"{np.percentile(su, 25) if len(su) else float('nan'):.3f}-{np.percentile(su, 75) if len(su) else float('nan'):.3f}")
    P_(f"  Mann-Whitney two-sided p = {mw.pvalue if mw else float('nan'):.4g}")
    pooled_dir = int(np.sign(np.median(su) - np.median(ex))) if len(ex) and len(su) else 0
    agree = elig = 0
    for mo in sorted({r["mouse"] for r in alm}):
        e_ = [r["x50"] for r in alm if r["mouse"] == mo and r["sign"] > 0]
        s_ = [r["x50"] for r in alm if r["mouse"] == mo and r["sign"] < 0]
        if len(e_) >= 5 and len(s_) >= 5:
            elig += 1
            dd = np.median(s_) - np.median(e_)
            agree += int(np.sign(dd)) == pooled_dir and pooled_dir != 0
            P_(f"    {mo}  excited {len(e_):>3} median {np.median(e_):.3f}   suppressed {len(s_):>3} median {np.median(s_):.3f}   diff {dd:+.3f}")
    passed = bool(mw is not None and mw.pvalue < 0.05 and elig > 0 and agree > elig / 2)
    order = pooled_dir if passed else 0
    P_(f"  mice agreeing with the pooled direction: {agree}/{elig}")
    P_(f"  U4: {'PASS' if passed else 'NOT PASSED'}  ->  ORDER = {order:+d}  "
       + {1: "(suppression recruited at HIGHER drive: theta_hyp starts above theta_dep)",
          -1: "(suppression recruited at LOWER drive: theta_hyp starts below theta_dep)",
          0: "(no ordering signal: gates untied, starting equal)"}[order])
    fs = [r["x50"] for r in alm if r["cell"] == "FS" and r["sign"] > 0]
    py = [r["x50"] for r in alm if r["cell"] == "Pyr" and r["sign"] > 0]
    P_(f"  reported, not used: excited FS (putative inhibitory) x50 median "
       f"{np.median(fs) if fs else float('nan'):.3f} (n {len(fs)}) vs excited Pyr {np.median(py) if py else float('nan'):.3f} (n {len(py)})")

    P_("\n" + RULE); P_("U5  STEEPNESS AND THE WHOLE ANIMAL"); P_(RULE)
    third = [r["norm"][0.33] for r in rows if r["area"] == "ALM" and r["responsive"] and 0.33 in r.get("norm", {})]
    P_(f"  ALM responsive units with a 1/3-drive level: n {len(third)}, median normalised response "
       f"{np.median(third) if third else float('nan'):.3f}  (linear in light 0.33; switch-like ~1)")
    P_("  behaviour, sessions with >= 3 drive levels: P(lick right) by drive")
    for xv in sorted(beh):
        P_(f"    x = {xv:.2f}   {np.mean(beh[xv]):.3f}   (n {len(beh[xv])})")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"dataset": "DANDI:000060", "doi": "10.1038/s41593-021-00840-6", "order": order,
               "u4": dict(passed=passed, p=float(mw.pvalue) if mw else None, n_exc=len(ex), n_sup=len(su),
                          median_exc=float(np.median(ex)) if len(ex) else None,
                          median_sup=float(np.median(su)) if len(su) else None, mice_agree=agree, mice=elig),
               "shape": shape, "behaviour": {str(k): [float(np.mean(v)), len(v)] for k, v in beh.items()},
               "alignment": dict(go=go_ok, match=match_ok)}, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/opto_units.json  (ORDER = {order:+d})")
    P_("\n" + RULE); P_("U6  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  In vivo curves are circuit responses, not membrane properties. The rule borrows one ORDERING")
    P_("  from the circuit. Draft dataset, no licence declared: aggregates only, raw data not redistributed.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
