"""FOLDPAGE -- builds the final page 'How a chromosome folds' (rules, an in-browser simulator, measured vs simulated maps,
the evidence) from the recorded test outputs, so every number on the page comes from a results file.

Reads: outputs/k562_3d, thp1_3d, a549_3d (G16-G18), outputs/loopcode (G19, G20), outputs/foldsim (G21 + HCT116 example
maps, ENCODE-derived). Template: rem/atlas/pages/foldpage.html. Writes index.html + maps.json to the folder given.

Run: python3 -I rem/atlas/foldpage.py OUT_DIR
"""

from __future__ import annotations
import base64
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
_s = importlib.util.spec_from_file_location("foldsim", os.path.join(HERE, "foldsim.py")); fs = importlib.util.module_from_spec(_s); _s.loader.exec_module(fs)
J = lambda *p: json.load(open(R("outputs", *p)))


RANGES = dict(untreated=(-1.5, 1.5), sim=(-0.4, 0.4), auxin=(-1.5, 1.5))          # log2 range stored in a byte
WINDOWS = dict(untreated=(-0.7, 1.1), sim=(-0.22, 0.32), auxin=(-0.7, 1.1))         # log2 range shown (each map its own)


def oe8(M, kind):
    if kind != "sim":
        Mb, valid = fs.balance(M); S = np.where(valid[:, None] & valid[None, :], Mb, np.nan); P = np.pad(S, 1, constant_values=np.nan)
        with np.errstate(all="ignore"):                                              # 3 x 3 mean for display only
            Mb = np.nan_to_num(np.nanmean(np.stack([P[1 + a:1 + a + len(M), 1 + b:1 + b + len(M)] for a in (-1, 0, 1) for b in (-1, 0, 1)]), 0))
        O = fs.obs_over_exp(Mb, valid)
    else:
        O = fs.obs_over_exp(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log2(np.where(O > 0, O, np.nan))
    lo, hi = RANGES[kind]; q = np.clip((L - lo) / (hi - lo), 0, 1) * 254 + 1; q[~np.isfinite(L)] = 0
    return base64.b64encode(q.astype(np.uint8).tobytes()).decode()


def score(P, M):
    Mb, valid = fs.balance(M); Om, Os = fs.obs_over_exp(Mb, valid), fs.obs_over_exp(P, valid); iu = np.triu_indices(len(M), 2); sep = iu[1] - iu[0]
    m = (sep <= 60) & np.isfinite(Om[iu]) & np.isfinite(Os[iu])
    from scipy.stats import spearmanr
    return float(spearmanr(Os[iu][m], Om[iu][m])[0])


def build(out):
    g16, g17, g18 = J("k562_3d", "k562_3d_tests.json"), J("thp1_3d", "thp1_3d_tests.json"), J("a549_3d", "a549_3d_tests.json")
    g19, g20, g21 = J("loopcode", "loopcode_tests.json"), J("loopcode", "loopdyn_tests.json"), J("foldsim", "foldsim_tests.json")
    r1, r2, k1, k2, k3, m1 = (g19[f"G19-{k}"] for k in ("R1", "R2", "K1", "K2", "K3", "M1"))
    d2, e1, m3 = g20["G20-D2"], g20["G20-E1"], g20["G20-M3"]; s1, s2, s3, s4, s5 = (g21[f"G21-S{k}"] for k in range(1, 6)); rep = g21["reported"]
    n = r1["loops"]; t = r1["table"]; E = k1["E_by_condition"]
    charts = [
        dict(title="Stop signs at the two ends of a loop", fmt="pct", max=1, ref=0.25, refLabel="chance",
             items=[dict(label="Facing each other ▶ ◀", v=t["convergent"] / n), dict(label="Same direction", v=(t["both_plus"] + t["both_minus"]) / n),
                    dict(label="Facing away ◀ ▶", v=t["divergent"] / n)],
             note=f"{n:,} loops with one clear stop sign at each end (K562). Chance for facing each other is 25%."),
        dict(title="Take the motor away, then give it back", fmt="num", max=3, ref=1, refLabel="no loop",
             items=[dict(label="Motor present", v=E["untreated"]), dict(label="Motor destroyed (6 h)", v=E["auxin_6h"], alt=True), dict(label="20 min after it returns", v=E["washout_20"]),
                    dict(label="40 min", v=E["washout_40"]), dict(label="60 min", v=E["washout_60"]), dict(label="3 hours", v=E["washout_180"])],
             note=f"Loop strength at {k1['loops']:,} loops in HCT116 cells. 1.0 means no loop at all."),
        dict(title="What has already changed after 6 hours", fmt="pct", max=1,
             items=[dict(label="Gene activity", v=g16["G16-T1"]["FY"]), dict(label="Activity mark at loop ends", v=d2["F_act"]),
                    dict(label="DNA opening", v=d2["other_marks_share_at_6h"]["atac"]["F_mark"]), dict(label="Switch protein (JUN)", v=d2["other_marks_share_at_6h"]["jun"]["F_mark"]),
                    dict(label="Stop-sign protein (CTCF)", v=d2["other_marks_share_at_6h"]["ctcf"]["F_mark"]), dict(label="Loops", v=d2["F_loop"], alt=True),
                    dict(label="Zones", v=m3["F_zone"], alt=True)],
             note="K562 cells after a signal. Share of each one's 72-hour change made by 6 hours. Orange bars are the 3D map; blue bars are activity."),
        dict(title="How well each recipe finds real loops", fmt="num", max=0.5, ref=r2["base_rate"], refLabel="chance",
             items=[dict(label="Distance only", v=r2["ap_pooled"]["M0"]), dict(label="+ stop-sign strength", v=r2["ap_pooled"]["M1"]),
                    dict(label="+ stop-sign direction", v=r2["ap_pooled"]["M2"]), dict(label="The simulator", v=s2["ap_simulator"], alt=True)],
             note="Average precision over all pairs of stop signs on unseen chromosomes (K562). The first three are fitted formulas; the simulator only plays out the rules."),
        dict(title="What sets the zones", fmt="num", max=1,
             items=[dict(label="Gene positions alone", v=m1["median_r_baseline"]), dict(label="Chromatin marks", v=m1["median_r"]),
                    dict(label="Same cells, motor destroyed", v=k3["median_r"], alt=True)],
             note="Match (0 to 1) between predicted and measured zones on unseen chromosomes; the last bar compares zones with and without the motor."),
        dict(title="Genes follow their zone, in three cell types", fmt="pct", max=1, ref=0.5, refLabel="chance",
             items=[dict(label="K562 (slow change)", v=g16["G16-C1"]["share_positive"]), dict(label="THP-1 (immune alarm)", v=g17["G17-C1"]["share_positive"]),
                    dict(label="A549 (steroid)", v=g18["G18-C1"]["share_positive"])],
             note="Share of genes that went up when their stretch of DNA moved toward the open zone, or down when it moved toward the closed zone."),
        dict(title="Does a loop pass activity on to a gene?", fmt="num", max=0.3,
             items=[dict(label="Switch looped to the gene", v=e1["r_looped"]), dict(label="Same distance, not looped", v=e1["r_control"], alt=True)],
             note=f"How closely a gene's change follows the change of a distant switch ({e1['pairs']:,} pairs, K562)."),
    ]
    v = lambda x: "PASS" if x["gate"]["verdict"] == "PASS" else ("FAIL" if x["gate"]["verdict"] == "FAIL" else "void")
    sim = [
        dict(q="Does the simulated map look like the measured one? (K562)", verdict=v(s1),
             n=f"match {s1['median_score']:.2f} with real stop signs, {s1['median_comparison']:.2f} with the signs at random places ({s1['regions']} regions)"),
        dict(q="Does it find the real loops? (K562)", verdict=v(s2),
             n=f"precision {s2['ap_simulator']:.2f} against {s2['ap_distance']:.2f} for distance alone; {s2['ap_simulator_shuffled_directions']:.2f} with directions shuffled"),
        dict(q="Does it predict which loops get stronger or weaker over 72 hours? (K562)", verdict=v(s3),
             n=f"match {s3['median_r']:.2f} ({s3['loops_used']:,} loops); stop signs alone {s3['stop_signs_only']:.2f}"),
        dict(q="Does the zone rule draw the large-scale pattern? (K562)", verdict=v(s4),
             n=(f"match {s4['median_r']:.2f} on unseen chromosomes" if v(s4) != "void" else f"measured match {s4['median_r']:.2f}, but a safety check on this test itself failed, so it is not counted")),
        dict(q="Does it work in another cell type, and only when the motor is there? (HCT116)", verdict=v(s5),
             n=f"match {s5['median_score']:.2f} with the motor, {s5['median_score_vs_no_cohesin_map']:.2f} after the motor is destroyed ({s5['regions']} regions)"),
    ]
    P = rep["params"]
    fitnote = (f"The simulator has four numbers, set once on the odd-numbered chromosomes of K562: a motor reels about {P['processivity_kb']} kb before it falls off, "
               f"there is one motor per {P['separation_kb']} kb, a strong stop sign holds with probability {P['c']:.1f}, and contact falls with distance to the power {P['alpha']}. "
               "It is given the stop-sign map and never sees a measured contact. 'Match' is a rank correlation between simulated and measured maps (0 = none, 1 = identical); measured maps are noisy, so 1 is out of reach.")
    Z = np.load(R("outputs", "foldsim", "hct116_example_maps.npz")); keys = [int(k) for k in Z["keys"]]; use = [i for i, k in enumerate(keys) if str(k) in s5["scores"]]
    sc = np.full(len(keys), np.nan); sc[use] = [s5["scores"][str(keys[i])] for i in use]; o = np.array(sorted(use, key=lambda i: -sc[i]))
    pick = [("Best match", o[0]), ("Typical", o[len(o) // 2]), ("Poor match", o[int(len(o) * 0.85)])]
    regions = []
    for label, i in pick:
        b0 = int(Z["bin0"][i])
        regions.append(dict(label=label, chrom=str(Z["chrom"][i]), start_mb=b0 * 0.025, end_mb=(b0 + 120) * 0.025, score=float(sc[i]), score_auxin=score(Z["sim"][i].astype(float), Z["auxin"][i].astype(float)),
                            untreated=oe8(Z["untreated"][i].astype(float), "untreated"), sim=oe8(Z["sim"][i].astype(float), "sim"), auxin=oe8(Z["auxin"][i].astype(float), "auxin")))
    data = dict(charts=charts, sim=sim, fitnote=fitnote,
                real_intro=("HCT116 cells, a cell type the simulator was not tuned on. It gets only that cell's stop-sign map. Left: the measured map. Middle: the simulator's map. "
                            "Right: the measured map after the motor was destroyed, where the neighbourhoods and loops are gone."),
                real_note=(f"Three of the {s5['regions']} test regions on unseen chromosomes: the best, a typical one and a poor one. The simulator gets the positions of most neighbourhood borders right; "
                           "it misses finer detail and contacts that do not depend on stop signs. Each map uses its own colour range (the simulator's contrasts are fainter), and measured maps are lightly smoothed for display."),
                open=[f"Stop signs explain only part of the loops: {r1['anchors']['at_none'] * 100:.0f}% of loops have no clear stop sign at either end and {r1['anchors']['at_one'] * 100:.0f}% at only one.",
                      f"The simulator's maps match measured ones at about {s1['median_score']:.2f} on a 0-to-1 scale. It captures the main neighbourhoods, not every detail.",
                      "Only the motor-removal experiment proves cause. The other tests show what goes together and in which order.",
                      "This is a simulator on a line of DNA with a distance rule for contacts. It is not yet a full 3D physical model.",
                      "Tested in cancer cell lines (K562, HCT116, THP-1, A549), with one stimulus for the timing."])
    os.makedirs(out, exist_ok=True)
    html = open(os.path.join(HERE, "pages", "foldpage.html"), encoding="utf-8").read().replace("__DATA__", json.dumps(data, ensure_ascii=False))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(html)
    json.dump(dict(n=120, ranges=RANGES, windows=WINDOWS, regions=regions), open(os.path.join(out, "maps.json"), "w"))
    print("wrote", out, os.path.getsize(os.path.join(out, "index.html")), os.path.getsize(os.path.join(out, "maps.json")))


if __name__ == "__main__":
    build(sys.argv[1])
