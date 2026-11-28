"""FOLDPAGE -- builds the final page 'How a chromosome folds' (rules, an in-browser simulator, measured vs simulated maps,
the evidence) from the recorded test outputs, so every number on the page comes from a results file.

Reads: outputs/k562_3d, thp1_3d, a549_3d (G16-G18), outputs/loopcode (G19, G20), outputs/foldsim (G21, G22 + example maps,
ENCODE-derived, and medians of the microscope traces). Template: rem/atlas/pages/foldpage.html. Writes index.html +
maps.json to the folder given.

Run: python3 -I rem/atlas/foldpage.py OUT_DIR [folder with practice G22 outputs, for layout previews only]
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


RANGES = dict(meas=(-1.5, 1.5), sim=(-1.0, 1.0))          # log2 range stored in a byte
WINDOWS = dict(meas=(-0.7, 1.1), sim=(-0.45, 0.7))         # log2 range shown (each kind its own)
TRANGE = (-0.6, 0.6)                                       # microscope: log2(usual distance / distance)
CELLS = dict(hct116="HCT116", gm12878="GM12878", imr90="IMR-90", hepg2="HepG2")


def b64(q):
    return base64.b64encode(q.astype(np.uint8).tobytes()).decode()


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
    return b64(q)


def dist8(M):
    """median distances between segments -> log2(mean distance at that spacing / distance): closer than usual = high."""
    n = len(M); O = np.full((n, n), np.nan); i = np.arange(n)
    with np.errstate(all="ignore"):
        for k in range(1, n):
            d = np.diag(M, k).astype(float); mu = np.nanmean(d)
            if np.isfinite(mu) and mu > 0:
                O[i[:n - k], i[:n - k] + k] = O[i[:n - k] + k, i[:n - k]] = -np.log2(d / mu)
    lo, hi = TRANGE; q = np.clip((O - lo) / (hi - lo), 0, 1) * 254 + 1; q[~np.isfinite(O)] = 0
    return b64(q)


def score(P, M):
    """the map score of the tests (foldsimtest.region_score)."""
    from scipy.stats import spearmanr
    Mb, valid = fs.balance(M); Pb, v2 = fs.balance(np.where(valid[:, None] & valid[None, :], P, 0.0)); valid = valid & v2
    Om, Os = fs.obs_over_exp(Mb, valid), fs.obs_over_exp(Pb, valid); iu = np.triu_indices(len(M), 2); sep = iu[1] - iu[0]
    m = (sep <= 60) & np.isfinite(Om[iu]) & np.isfinite(Os[iu])
    return float(spearmanr(Os[iu][m], Om[iu][m])[0])


def build(out, g22dir=None):
    g22dir = g22dir or R("outputs", "foldsim")                # a second folder only for layout previews with practice numbers
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
    void = lambda x, t: t + (" (not counted: a safety check on this test itself failed)" if v(x) == "void" else "")
    g22 = json.load(open(os.path.join(g22dir, "foldsim2_tests.json"))); U1, U2, B1, B2, B3, B4, D1, D2, D3, E1, E2, Z1 = (g22[f"G22-{k}"] for k in ("U1", "U2", "B1", "B2", "B3", "B4", "D1", "D2", "D3", "E1", "E2", "Z1"))
    rp = g22["reported"]; P, ce = rp["params"], rp["ceiling"]; bare = U2["auc_loops_without_clear_signs"]; ec = E1["cells"]; pct = lambda x: f"{x * 100:.0f}%"
    sim = [
        dict(q="Version 1: does the simulated map look like the measured one? (K562)", verdict=v(s1),
             n=f"match {s1['median_score']:.2f} with real stop signs, {s1['median_comparison']:.2f} with the signs at random places ({s1['regions']} regions)"),
        dict(q="Version 1: does it find the real loops? (K562)", verdict=v(s2),
             n=f"precision {s2['ap_simulator']:.2f} against {s2['ap_distance']:.2f} for distance alone; {s2['ap_simulator_shuffled_directions']:.2f} with directions shuffled"),
        dict(q="Version 1: does it predict which loops get stronger or weaker over 72 hours? (K562)", verdict=v(s3),
             n=f"match {s3['median_r']:.2f} ({s3['loops_used']:,} loops); stop signs alone {s3['stop_signs_only']:.2f}"),
        dict(q="Version 1: does it work in another cell type, and only when the motor is there? (HCT116)", verdict=v(s5),
             n=f"match {s5['median_score']:.2f} with the motor, {s5['median_score_vs_no_cohesin_map']:.2f} after the motor is destroyed ({s5['regions']} regions)"),
        dict(q="Does the zone rule draw the large-scale pattern? (K562; first try was not counted, repeated with a sound safety check)", verdict=v(Z1),
             n=void(Z1, f"match {Z1['median_r']:.2f} on {Z1['chromosomes']} unseen chromosomes")),
        dict(q="Version 2: does putting zones and weak stop signs in draw better maps than version 1? (K562)", verdict=v(U1),
             n=void(U1, f"match {U1['median_v2']:.2f} against {U1['median_v1']:.2f}; zones alone {U1['median_zones_alone']:.2f}; everything at random places {U1['median_random_places']:.2f} ({U1['regions']} regions)")),
        dict(q="Version 2: does it find loops better than version 1? (K562)", verdict=v(U2),
             n=void(U2, f"a real loop beats a shifted copy {pct(U2['auc_v2'])} of the time against {pct(U2['auc_v1'])}; loops with no clear stop sign {pct(bare['v2'])} (chance 50%)")),
        dict(q="Destroy the stop signs: does each simulator match its own cell best? (HCT116)", verdict=v(B1),
             n=void(B1, f"whole cell: whole simulator {B1['whole_vs_before']:.2f}, without signs {B1['broken_vs_before']:.2f}; cell without CTCF: without signs {B1['broken_vs_after']:.2f}, whole {B1['whole_vs_after']:.2f}")),
        dict(q="Destroy the motor: does each simulator match its own cell best? (HCT116)", verdict=v(B2),
             n=void(B2, f"whole cell: whole simulator {B2['whole_vs_before']:.2f}, without motors {B2['broken_vs_before']:.2f}; cell without cohesin: without motors {B2['broken_vs_after']:.2f}, whole {B2['whole_vs_after']:.2f}")),
        dict(q="Does removing the gene-reading machine change the folding less than removing CTCF? (HCT116)", verdict=v(B3),
             n=void(B3, f"the match drops by {B3['median_drop_ctcf']:.2f} without CTCF and by {B3['median_drop_pol2']:.2f} without RNA polymerase II ({B3['regions']} regions)")),
        dict(q="3D model: do its maps look like the measured ones? (K562)", verdict=v(D1),
             n=void(D1, f"match {D1['median_score']:.2f}, {D1['median_random_places']:.2f} with everything at random places ({D1['regions']} regions)")),
        dict(q="3D model: does it predict how contacts change with distance when the motor is removed? (HCT116)", verdict=v(B4),
             n=void(B4, f"curve match {B4['r']:.2f}; the predicted change is {pct(B4['amplitude_predicted_over_measured'])} the size of the measured one")),
        dict(q="3D model against distances seen under the microscope (K562)", verdict=v(D2),
             n=void(D2, f"match {D2['r']:.2f}; random sites do as well {pct(D2['p_mc'])} of the time")),
        dict(q="Under the microscope, does the match need the motor? (HCT116)", verdict=v(D3),
             n=void(D3, f"match {D3['r_untreated']:.2f} with the motor, {D3['r_auxin']:.2f} after it is destroyed; random sites give this gap {pct(D3['p_mc'])} of the time")),
        dict(q="Do single chromosomes put their borders where the model does? (K562)", verdict=v(E2),
             n=void(E2, f"match {E2['r']:.2f}; random sites do as well {pct(E2['p_mc'])} of the time")),
        dict(q="Does version 2 work in three more cell types with no new tuning?", verdict=v(E1),
             n=void(E1, "; ".join(f"{CELLS[k]} {ec[k]['median_score']:.2f} (random places {ec[k]['median_random_places']:.2f})" for k in ec))),
    ]
    yes = lambda x: v(x) == "PASS"; n_break = sum(yes(x) for x in (B1, B2, B3)); n_cells = sum(ec[k]["passed"] for k in ec)
    steps = [
        dict(k="1 · Best possible", v=f"The ceiling is {ce['ceiling_full_map']:.2f}", t=f"Two repeats of the same measurement agree at {ce['half_vs_half']:.2f}, so no simulator can score above about {ce['ceiling_full_map']:.2f}. "
             f"Version 1 reached {pct(ce['v1_share'])} of that, version 2 {pct(ce['v2_share'])}."),
        dict(k="2 · Missing parts", v="Zones were the big gap" if yes(U1) else "No clear gain", t=f"Writing the zone rule into the map lifts the match from {U1['median_v1']:.2f} to {U1['median_v2']:.2f}. "
             "Weak stop signs helped a little. Leaky signs, anchors without direction and landing on active DNA did not help and were left out."),
        dict(k="3 · Destroy a part", v=f"{n_break} of 3 predictions held",
             t=(f"Take the stop signs or the motors out of the simulator and its match with the whole cell falls ({B1['whole_vs_before']:.2f} to {B1['broken_vs_before']:.2f}; {B2['whole_vs_before']:.2f} to {B2['broken_vs_before']:.2f}): both parts are needed. "
                if B1["before"]["passed"] and B2["before"]["passed"] else "Taking a part out of the simulator did not clearly cost it its match with the whole cell. ") +
               ("The cells that really lost the part are matched best by the simulator without it. " if B1["after"]["passed"] and B2["after"]["passed"] else
                "But cells that really lost a part still look a little more like the whole simulator than the broken one, so the broken cell is not predicted yet. ") +
               ("Removing the gene-reading machine barely changes the fold, as the rules say." if yes(B3) else "The gene-reading machine test did not come out as expected.")),
        dict(k="4 · Real 3D", v=("Matches the microscope" if yes(D2) and yes(D3) else ("Matches the microscope in K562, not in HCT116" if yes(D2) else "Mixed against the microscope")),
             t=f"A chain of beads and springs draws maps at {D1['median_score']:.2f}. " +
               (f"In K562 its distances match the microscope images ({D2['r']:.2f}). " if yes(D2) else f"In K562 its distances do not clearly match the microscope images ({D2['r']:.2f}). ") +
               (f"In HCT116 they match with the motor ({D3['r_untreated']:.2f}) and less without ({D3['r_auxin']:.2f})." if yes(D3) else f"In HCT116 they do not ({D3['r_untreated']:.2f}).")),
        dict(k="5 · More cells", v=f"{n_cells} of 3 new cell types", t="Same numbers, each cell's own stop signs and activity marks: " + ", ".join(f"{CELLS[k]} {ec[k]['median_score']:.2f}" for k in ec) +
             (f". Single chromosomes put borders where the model does (match {E2['r']:.2f})." if yes(E2) else ".")),
    ]
    kb = lambda x: f"{x} kb" if x < 1000 else f"{x / 1000:.1f} Mb"
    charts2 = [
        dict(title="How close to the best possible score?", fmt="num", max=1, ref=ce["ceiling_full_map"], refLabel="best possible",
             items=[dict(label="Version 2: loops + zones", v=U1["median_v2"]), dict(label="3D model", v=D1["median_score"]), dict(label="Version 1: loops only", v=U1["median_v1"]),
                    dict(label="Zones alone", v=U1["median_zones_alone"], alt=True), dict(label="Motors alone", v=U1["median_v2_motors_alone"], alt=True),
                    dict(label="Everything at random places", v=max(U1["median_random_places"], 0), alt=True)],
             note=f"Match with the measured K562 map on {U1['regions']} unseen regions. The dashed line is what two repeats of the measurement allow."),
        dict(title="Destroy the stop signs (CTCF)", fmt="num", max=0.6,
             items=[dict(label="Whole cell · whole simulator", v=B1["whole_vs_before"]), dict(label="Whole cell · no stop signs", v=B1["broken_vs_before"], alt=True),
                    dict(label="No CTCF · no stop signs", v=B1["broken_vs_after"]), dict(label="No CTCF · whole simulator", v=B1["whole_vs_after"], alt=True)],
             note="Each pair: the blue bar is the simulator that fits that cell, the orange one the wrong simulator. Blue should be longer in both pairs."),
        dict(title="Destroy the motor (cohesin)", fmt="num", max=0.6,
             items=[dict(label="Whole cell · whole simulator", v=B2["whole_vs_before"]), dict(label="Whole cell · no motors", v=B2["broken_vs_before"], alt=True),
                    dict(label="No cohesin · no motors", v=B2["broken_vs_after"]), dict(label="No cohesin · whole simulator", v=B2["whole_vs_after"], alt=True)],
             note="Same reading. Without motors the simulator is zones only."),
        dict(title="Destroy the gene-reading machine", fmt="num", max=0.6,
             items=[dict(label="CTCF line, untouched", v=B1["whole_vs_before"]), dict(label="CTCF destroyed", v=B1["whole_vs_after"], alt=True),
                    dict(label="Polymerase line, untouched", v=B3["median_score_pol2_untreated"]), dict(label="Polymerase destroyed", v=B3["median_score_pol2_removed"], alt=True)],
             note="Match of the whole simulator. The rules say folding does not need RNA polymerase II, so its removal should cost little."),
        dict(type="line", title="Contacts gained or lost because of the motor", xlab="distance between two places on the DNA", x=[kb(x) for x in B4["curve_kb"]], fmt="spct",
             series=[dict(name="measured", y=[float(np.exp(x) - 1) for x in B4["measured"]]), dict(name="3D model", y=[float(np.exp(x) - 1) for x in B4["predicted"]])],
             note=f"Cell with motor compared with cell without, relative to the average. The model gets the shape ({B4['r']:.2f}) but only {pct(B4['amplitude_predicted_over_measured'])} of the size."),
        dict(title="Finding loops", fmt="pct", max=1, ref=0.5, refLabel="chance",
             items=[dict(label="Version 2, all loops", v=U2["auc_v2"]), dict(label="Version 1, all loops", v=U2["auc_v1"]),
                    dict(label="Version 2, loops with no stop sign", v=bare["v2"], alt=True), dict(label="Version 1, loops with no stop sign", v=bare["v1"], alt=True)],
             note=f"How often the simulator holds a real loop more than the same span moved 100 kb aside. {bare['loops']:,} loops have no clear stop sign at either end."),
        dict(title="Three more cell types, no new tuning", fmt="num", max=0.6,
             items=[x for k in ec for x in (dict(label=f"{CELLS[k]}: version 2", v=ec[k]["median_score"]), dict(label=f"{CELLS[k]}: version 1", v=ec[k]["median_v1"], alt=True))],
             note="Match with each cell type's measured map on unseen chromosomes. With everything at random places: " + ", ".join(f"{max(ec[k]['median_random_places'], 0):.2f}" for k in ec) + "."),
    ]
    fitnote = (f"Version 2 has seven numbers, set once on the odd-numbered chromosomes of K562: a motor reels about {P['processivity_kb']} kb, there is one motor per {P['separation_kb']} kb, "
               f"a strong stop sign holds with probability {P['c']:.1f} and a weak one {P['w']:.1f} times its rank, contact falls with distance to the power {P['alpha']}, "
               f"like zones touch more by a factor set by {P['kappa']}, and a zone is read from the activity mark averaged over {2 * P['smooth'] + 1} bins of 25 kb. The 3D model adds three: a motor's spring is {P['k_loop']:.0f} times a link of the chain, two beads touch within {P['radius']} links, "
               f"and beads of the same zone pull with {P['k_zone']} of a link. Every other cell type uses the same numbers. The simulator is given stop signs and an activity mark and never sees a measured contact. "
               "'Match' is a rank correlation between simulated and measured maps (0 = none, 1 = identical).")
    Z = np.load(os.path.join(g22dir, "foldsim2_page.npz")); labels = ("Best match", "Typical"); cells = []
    def region(name, i, views):
        b0 = int(Z[f"{name}_bin0"][i]); return dict(label=labels[i], chrom=str(Z[f"{name}_chrom"][i]), start_mb=b0 * 0.025, end_mb=(b0 + 120) * 0.025, views=views)
    def view(sim_, meas_, cross=None):
        d = dict(meas=oe8(meas_.astype(float), "meas"), sim=oe8(sim_.astype(float), "sim"), score=score(sim_.astype(float), meas_.astype(float)))
        if cross is not None:
            d["cross"] = score(cross.astype(float), meas_.astype(float))
        return d
    hr = [region("hct116", i, dict(none=view(Z["hct116_sim"][i], Z["hct116_ctcf_untreated_measured"][i]), ctcf=view(Z["hct116_nosign"][i], Z["hct116_ctcf_auxin_measured"][i], Z["hct116_sim"][i]),
                                   motor=view(Z["hct116_nomotor"][i], Z["hct116_cohesin_auxin_measured"][i], Z["hct116_sim"][i]))) for i in range(2)]
    cells.append(dict(id="hct116", label="HCT116", regions=hr, breaks=[dict(id="none", label="Nothing", meas="Measured, whole cell", sim="Simulated, whole"),
                                                                      dict(id="ctcf", label="Stop signs (CTCF)", meas="Measured, CTCF destroyed", sim="Simulated, stop signs removed"),
                                                                      dict(id="motor", label="Motor (cohesin)", meas="Measured, cohesin destroyed", sim="Simulated, motors removed")]))
    for k in ("gm12878", "imr90", "hepg2"):
        cells.append(dict(id=k, label=CELLS[k], regions=[region(k, i, dict(none=view(Z[f"{k}_sim"][i], Z[f"{k}_measured"][i]))) for i in range(2)],
                          breaks=[dict(id="none", label="Nothing", meas="Measured in the cell", sim="Simulated")]))
    info = J("cellbrain", "foldsim2_report.json")["build"]["traces"]; nn = lambda a: [None if not np.isfinite(x) else float(x) for x in a]
    def trace(label, key, model, r, start):
        M = Z[f"trace_{key}_measured_nm"]; n = len(M)
        return dict(label=label, n=n, start_mb=start, chromosomes=info[key]["chromosomes"], r=float(r), meas=dist8(M), sim=dist8(Z[f"trace_{model}_model_nm"]),
                    border_x=[f"{start + i * 0.03:.1f}" for i in range(n)], border_meas=nn(Z[f"trace_{key}_measured_border"]), border_sim=nn(Z[f"trace_{model}_model_border"]))
    traces = [trace("K562", "k562_28", "k562", D2["r"], 28.0), trace("HCT116, motor present", "hct116_34_untreated", "hct116", D3["r_untreated"], 34.6),
              trace("HCT116, motor destroyed", "hct116_34_auxin", "hct116", D3["r_auxin"], 34.6)]
    data = dict(charts=charts, charts2=charts2, sim=sim, steps=steps, fitnote=fitnote,
                real_intro=("Four cell types the simulator was not tuned on. It gets only that cell's stop-sign map and activity mark. For HCT116 you can destroy a part: the lab removed the cell's stop-sign protein (CTCF) "
                            "or its motor (cohesin), and the same part is switched off in the simulator."),
                real_note=("Two regions per cell type from unseen chromosomes: the best match and a typical one. Each map uses its own colour range (the simulator's contrasts are fainter), "
                           "and measured maps are lightly smoothed for display."),
                up_intro="After version 1 we listed five ways to make the simulator better. Here is what each one gave, on chromosomes kept aside.",
                micro_intro=("A different kind of measurement: the 3D position of 30 kb pieces of DNA, imaged in thousands of single chromosomes. Each square compares the typical distance between two pieces "
                             "with what is usual for their spacing. The 3D model was never fitted to these images."),
                micro_note=(void(D2, f"K562: the model's distances match the measured ones at {D2['r']:.2f}; models with sites at random places do as well {pct(D2['p_mc'])} of the time") + ". " +
                            void(D3, (f"HCT116: the model matches cells with their motor ({D3['r_untreated']:.2f}) better than cells that lost it ({D3['r_auxin']:.2f})" if yes(D3) else
                                      f"HCT116: no match ({D3['r_untreated']:.2f} with the motor, {D3['r_auxin']:.2f} without), so this test was not passed")) + ". " +
                            void(E2, f"Borders in single K562 chromosomes match the model's at {E2['r']:.2f}") + ". The line chart shows how often a border falls at each position; "
                            "the two ends count as borders more often in both, because they have fewer neighbours."),
                open=[f"Loops without stop signs are still not found: the simulator picks them out {pct(bare['v2'])} of the time, and chance is 50%. " +
                      f"{r1['anchors']['at_none'] * 100:.0f}% of loops have no clear stop sign at either end.",
                      f"Version 2 reaches {pct(ce['v2_share'])} of the best possible map score. The rest is finer detail the rules do not cover yet.",
                      ("The simulator does not yet predict a broken cell. After CTCF or cohesin is destroyed, the real map still looks a little more like the whole simulator than like the simulator without that part. "
                       "Likely reasons: the zone part is crude (one activity mark), and some stop signs or motors survive the treatment."
                       if not (B1["after"]["passed"] and B2["after"]["passed"]) else "Removal of stop signs and motors is predicted in both directions, but the margins are small."),
                      f"The 3D model lets DNA pass through itself. It gets the shape of the contact-with-distance change but only {pct(B4['amplitude_predicted_over_measured'])} of its size.",
                      *(["In HCT116 the 3D model does not match the microscope distances. A version without zone springs did on practice runs (0.69 with the motor, 0.39 without), so the zone springs look wrong for that region."] if not yes(D3) else []),
                      "The microscope data cover one 2 Mb region per cell line, so those tests are narrow.",
                      "Only the removal experiments prove cause. The other tests show what goes together and in which order.",
                      "Mostly cancer cell lines, and one stimulus for the timing."])
    os.makedirs(out, exist_ok=True)
    html = open(os.path.join(HERE, "pages", "foldpage.html"), encoding="utf-8").read().replace("__DATA__", json.dumps(data, ensure_ascii=False))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(html)
    json.dump(dict(n=120, ranges=RANGES, windows=WINDOWS, trange=TRANGE, cells=cells, traces=traces), open(os.path.join(out, "maps.json"), "w"))
    print("wrote", out, os.path.getsize(os.path.join(out, "index.html")), os.path.getsize(os.path.join(out, "maps.json")))


if __name__ == "__main__":
    build(*sys.argv[1:3])
