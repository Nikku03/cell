"""Drug development in the memory bank: what drugs do to K562 cells, what they are known to hit, and what trials,
safety studies and post-marketing reports found.

WHY. The user asked to add drug development data and the new knowledge about processes found while testing drugs
(trials, safety studies, adverse-event reports). Two questions follow, one from each side:
  (1) drug side: does a drug's measured effect on K562 resemble knocking down its target? This is target
      validation, the step drug development rests on; where it fails, the cell may be telling us about another
      target (an off-target or a different mechanism).
  (2) cell side: does any of this drug data help the engine predict an unmeasured knockdown?

THE DATA (memory_bank/cell_v6 = cell_v5 + the layers below)
  sci-Plex3 (Srivatsan et al. 2020, Science; scPerturb Zenodo 13350497, CC BY 4.0): 188 compounds at 4 doses,
     24 h, single-cell. K562 cells only. Per compound and dose: summed counts -> counts per million -> log1p;
     response = minus the vehicle's; only genes with vehicle CPM >= 20 kept; the mean response of all
     compounds at that dose removed (as every screen's own mean is removed); mapped onto screen A's genes by
     Ensembl id; projected on the same 128 components; int8 codes. Stored as screen F, gene field "drug:<name>".
  Open Targets Platform 26.09 (CC0): drug molecules, mechanisms of action (ChEMBL), clinical reports (trial
     phase, status, why stopped), drug warnings (black box, withdrawals), openFDA significant adverse
     reactions, target safety events (ToxCast, literature), chemical probes, tractability. Stored as
     drugs.json (per drug) and targets.json (per gene), with sources.
  Name matching: a sci-Plex compound matches a ChEMBL molecule if any of its names (main name or the names in
     brackets, salt words removed, letters and digits only, upper case) equals the molecule's name, a synonym
     or a trade name. A target counts for (1) only if the action is loss of function (INHIBITOR, ANTAGONIST,
     BLOCKER, NEGATIVE ALLOSTERIC MODULATOR, NEGATIVE MODULATOR, DEGRADER, DISRUPTING AGENT, ANTISENSE
     INHIBITOR, RNAI INHIBITOR, INVERSE AGONIST).

GATES, PREDECLARED
  T1  PRIMARY (drug side). Compounds at 10 uM with >= 1 loss-of-function target that has a K562 knockdown
      record in the bank (screen A, either split, else screen B). Per compound: r_target = the mean Pearson r
      (decoded gene space) between its response and its targets' knockdowns; null = the median r between its
      response and every K562 knockdown in the bank. Paired sign test over compounds: BETTER if p < 0.05 and
      the median of (r_target - null) >= +0.01. (This task predicts drug responses, not knockdowns, so screen-A
      TEST records may be used here; they are never used for the engine below.)
  T2  descriptive (discovery). For the 30 compounds with the strongest responses, the 5 knockdowns most like
      the response; each written to the bank's journal as an UNVALIDATED hypothesis with its sources and the
      drug's development record (stage, warnings, stopped trials).
  G8  PRIMARY (cell side). E4 vs E3 on all 2,525 TEST knockdowns (cellbank5's E3, rerun here with the same
      seeds so the pair is matched). E4 = E3 + two inputs: (a) drug-development features of the gene (log1p
      number of drugs, of loss-of-function drugs, highest clinical stage, tractability flags, number of
      safety events, a high-quality chemical probe, log1p number of stopped trials of drugs on it, number of
      drug warnings), (b) the mean 10 uM K562 response of the loss-of-function drugs on the gene (with a
      flag). Snap memory as E3 (screens A-E; drug records are not in it). BETTER / WORSE as before
      (p < 0.05 and median gain >= +0.01). Descriptive: genes with a drug response; genes with any drug data.
NOT: one name-matching rule; one dose and time (10 uM, 24 h); sci-Plex is shallow single-cell data; Open Targets
     mechanisms cover approved and clinical drugs better than research compounds.

Run: python3 -P rem/atlas/cellbank6.py build | engine | score
"""

from __future__ import annotations
import collections
import glob
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
_l = lambda n: importlib.util.spec_from_file_location(n, os.path.join(HERE, f"{n}.py"))
_s = _l("cellbank4"); cb4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb4)
_s = _l("cellbank5"); cb5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb5)
BANK = R("memory_bank", "cell_v6")
OUTD = R("outputs", "cellbank6")
CACHE = os.path.join(HERE, "_cache", "cellbank6")
OUT = os.path.join(HERE, "RESULTS_cellbank6.txt")
OT = os.path.join(HERE, "_cache", "opentargets")
SCIPLEX = os.path.join(HERE, "_cache", "sciplex", "sciplex3.h5ad")
LOF = {"INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE ALLOSTERIC MODULATOR", "NEGATIVE MODULATOR", "DEGRADER",
       "DISRUPTING AGENT", "ANTISENSE INHIBITOR", "RNAI INHIBITOR", "INVERSE AGONIST"}
STAGE = ["PRECLINICAL", "IND", "EARLY_PHASE_1", "PHASE_1", "PHASE_1_2", "PHASE_2", "PHASE_2_3", "PHASE_3", "PREAPPROVAL", "APPROVAL", "WITHDRAWAL"]
SALTS = r"\b(HCL|2HCL|MESYLATE|TOSYLATE|DITOSYLATE|CITRATE|PHOSPHATE|SODIUM|TRIHYDRATE|FREE BASE|DIPHOSPHATE|SALT|HYDROCHLORIDE|L-\(\+\)-TARTARIC ACID)\b"
SEEDS, MIN_CPM, RULE = (0, 1, 2), 20.0, "=" * 97


def norm(s):
    s = re.sub(SALTS, " ", s.upper())
    return re.sub(r"[^A-Z0-9]", "", s)


def names_of(compound):
    parts = [compound.split("(")[0]] + [p for grp in re.findall(r"\(([^)]*)\)", compound) for p in grp.split(",")]
    return {n for n in (norm(p) for p in parts) if len(n) >= 3}


def stage_rank(s):
    return STAGE.index(s) if s in STAGE else -1


def read(name):
    import pyarrow.parquet as pq
    return [r for f in sorted(glob.glob(os.path.join(OT, name, "*.parquet"))) for r in pq.read_table(f).to_pylist()]


def knowledge(sym_of_ens):
    """Drug-development knowledge from Open Targets: per drug and per target gene."""
    mols = read("drug_molecule")
    by_name = {}
    for m in mols:
        labels = [m["name"] or ""] + [x["label"] for x in (m["synonyms"] or [])] + [x["label"] for x in (m["tradeNames"] or [])]
        for lab in labels:
            if lab:
                by_name.setdefault(norm(lab), m["id"])
    drugs = {m["id"]: dict(chembl=m["id"], name=m["name"], type=m["drugType"], parent=m["parentId"], max_stage=m.get("maximumClinicalStage"),
                           mechanisms=[], indications=[], warnings=[], adverse_reactions=[], trials=collections.Counter(), stopped=[]) for m in mols}
    targets = collections.defaultdict(lambda: dict(drugs=[], tractability=[], safety_events=[], probes=[], stopped_trials=0, warnings=0))
    for moa in read("drug_mechanism_of_action"):
        for c in moa["chemblIds"] or []:
            for t in moa["targets"] or []:
                if c in drugs:
                    drugs[c]["mechanisms"].append(dict(action=moa["actionType"], target=t, symbol=sym_of_ens.get(t, ""), text=moa["mechanismOfAction"]))
                    targets[t]["drugs"].append(dict(chembl=c, action=moa["actionType"], stage=drugs[c]["max_stage"]))
    for ci in read("clinical_indication"):
        if ci["drugId"] in drugs:
            drugs[ci["drugId"]]["indications"].append([ci["diseaseId"], ci["maxClinicalStage"]])
    for rep in read("clinical_report"):
        if rep["type"] == "INDICATION" and rep["origin"] == "DRUG_LABEL":
            continue
        for d in rep["drugs"] or []:
            c = d["drugId"]
            if c in drugs:
                drugs[c]["trials"][rep["trialOverallStatus"] or "unknown"] += 1
                if rep["trialWhyStopped"] and len(drugs[c]["stopped"]) < 15:
                    drugs[c]["stopped"].append(dict(trial=rep["id"], phase=rep["trialPhase"], why=rep["trialWhyStopped"][:300],
                                                    categories=rep["trialStopReasonCategories"], url=rep["url"]))
    for w in read("drug_warning"):
        for c in w["chemblIds"] or []:
            if c in drugs:
                drugs[c]["warnings"].append(dict(type=w["warningType"], toxicity=w["toxicityClass"], country=w["country"], year=w["year"]))
    adr = collections.defaultdict(list)
    for a in read("openfda_significant_adverse_drug_reactions"):
        adr[a["chembl_id"]].append((a["llr"], a["event"], a["count"]))
    for c, v in adr.items():
        if c in drugs:
            drugs[c]["adverse_reactions"] = [dict(event=e, llr=round(l, 1), reports=n) for l, e, n in sorted(v, reverse=True)[:10]]
    for e in read("target_safety_event"):
        targets[e["targetId"]]["safety_events"].append(dict(event=e["event"], source=e["datasource"],
                                                             effects=[f"{x['direction']}/{x['dosing']}" for x in (e["effects"] or [])]))
    for p in read("chemical_probe"):
        if p["isHighQuality"]:
            targets[p["targetId"]]["probes"].append(p["drugFromSource"])
    for t in read("target_tractability"):
        if t["value"]:
            targets[t["targetId"]]["tractability"].append(f"{t['modality']}:{t['category']}")
    for c, d in drugs.items():
        d["trials"] = dict(d["trials"]); d["indications"] = sorted(d["indications"], key=lambda x: -stage_rank(x[1]))[:10]
        for m in d["mechanisms"]:
            targets[m["target"]]["stopped_trials"] += len(d["stopped"]); targets[m["target"]]["warnings"] += len(d["warnings"])
    drugs = {c: d for c, d in drugs.items() if d["mechanisms"] or d["trials"] or d["warnings"] or d["adverse_reactions"]}
    return drugs, dict(targets), by_name


def sciplex_k562(G):
    """K562 pseudobulk responses per compound and dose, mapped onto screen A's genes."""
    import h5py
    import numpy as np
    import scipy.sparse as sp
    torch = G["torch"]
    f = h5py.File(SCIPLEX, "r"); o = f["obs"]

    def cat(k):
        d = o[k]; c = [x.decode() if isinstance(x, bytes) else x for x in d["categories"][()]]
        return np.array(c, dtype=object)[d["codes"][()]]
    cl, pert, dose = cat("cell_line"), cat("perturbation"), o["dose_value"][()]
    keep = np.nonzero(cl == "K562")[0]
    groups = sorted(set(zip(pert[keep], dose[keep])))
    gi = {g: i for i, g in enumerate(groups)}
    gid = np.full(len(cl), -1); gid[keep] = [gi[(p, d)] for p, d in zip(pert[keep], dose[keep])]
    n_rows, n_genes = (int(x) for x in f["X"].attrs["shape"])
    ip = f["X/indptr"][()]; S = np.zeros((len(groups), n_genes)); ncell = np.bincount(gid[keep], minlength=len(groups))
    for a in range(0, n_rows, 40000):                 # read the matrix in row blocks; keep only K562 cells
        b = min(a + 40000, n_rows); rows = np.nonzero(gid[a:b] >= 0)[0]
        if not len(rows):
            continue
        blk = sp.csr_matrix((f["X/data"][ip[a]:ip[b]], f["X/indices"][ip[a]:ip[b]], ip[a:b + 1] - ip[a]), shape=(b - a, n_genes))[rows]
        Gm = sp.csr_matrix((np.ones(len(rows)), (gid[a + rows], np.arange(len(rows)))), shape=(len(groups), len(rows)))
        S += (Gm @ blk).toarray()
    lc = np.log1p(S / S.sum(1, keepdims=True) * 1e6)
    ens_cat = [x.decode().split(".")[0] for x in f["var/ensembl_id/categories"][()]]
    ens = [ens_cat[c] for c in f["var/ensembl_id/codes"][()]]
    veh = gi[("control", 0.0)]
    expressed = np.expm1(lc[veh]) >= MIN_CPM
    ens_A = [x.decode() for x in h5py.File(G["P"]["gwps.h5ad"], "r")["var/gene_id"][()]]
    col = {}
    for j, e in enumerate(ens_A):
        col.setdefault(e, j)
    pairs = [(j, col[e]) for j, e in enumerate(ens) if expressed[j] and e in col]
    resp = {}
    for (p, d), i in gi.items():
        if p != "control":
            v = np.full(len(ens_A), np.nan, dtype=np.float32)
            v[[c for _, c in pairs]] = (lc[i] - lc[veh])[[j for j, _ in pairs]]
            resp[(p.strip(), float(d))] = (v, int(ncell[i]))
    out = {}
    for d in sorted({d for _, d in resp}):
        ks = [k for k in resp if k[1] == d]
        M = torch.tensor(np.stack([resp[k][0] for k in ks]))
        for k, row in zip(ks, (M - torch.nanmean(M, 0))):
            out[k] = (row, resp[k][1])
    return out, dict(cells=len(keep), vehicle_cells=int(ncell[veh]), genes_kept=len(pairs))


def build():
    t0 = time.time()
    G = cb4.notebook(); torch, np = G["torch"], G["np"]
    sym_of_ens = {e: g for g, e in zip(G["KN"], G["KE"]) if e}
    for e, h in G["HPA"].items():
        sym_of_ens.setdefault(e, h.get("sym", ""))
    drugs, targets, by_name = knowledge(sym_of_ens)
    resp, info = sciplex_k562(G)
    old = cb4.Bank(cb5.BANK); meta = json.load(open(os.path.join(cb5.BANK, "records_meta.json")))
    Vy = G["Vy"].cpu()
    keys = sorted(resp)
    Z = torch.nan_to_num(torch.stack([resp[k][0] for k in keys])) @ Vy
    s = Z.abs().max(1).values.clamp(min=1e-12)
    codes = np.concatenate([old.codes, torch.round(Z / s[:, None] * 127).to(torch.int8).numpy()])
    scale = np.concatenate([old.scale, s.numpy().astype(np.float32)])
    compounds = {}
    for p in sorted({k[0] for k in keys}):
        hits = sorted({by_name[n] for n in names_of(p) if n in by_name})
        mech = [m for c in hits for c2 in {c, (drugs.get(c) or {}).get("parent") or c} for m in (drugs.get(c2) or {}).get("mechanisms", [])]
        compounds[p] = dict(chembl=hits, lof_targets=sorted({m["symbol"] for m in mech if m["action"] in LOF and m["symbol"]}),
                            all_targets=sorted({(m["symbol"], m["action"]) for m in mech}))
    for (p, d) in keys:
        meta["gene"].append(f"drug:{p}"); meta["ens"].append(""); meta["screen"].append("F"); meta["split"].append("DRUG"); meta["ep"].append(float("nan"))
        meta.setdefault("drug", {})[str(len(meta["gene"]) - 1)] = dict(compound=p, dose_nM=d, cells=resp[(p, d)][1], chembl=compounds[p]["chembl"])
    os.makedirs(BANK, exist_ok=True)
    np.savez_compressed(os.path.join(BANK, "records.npz"), codes=codes, scale=scale, basis=old.basis)
    json.dump(meta, open(os.path.join(BANK, "records_meta.json"), "w"))
    json.dump(old.genes, open(os.path.join(BANK, "genes.json"), "w"))
    json.dump(drugs, open(os.path.join(BANK, "drugs.json"), "w"))
    json.dump(targets, open(os.path.join(BANK, "targets.json"), "w"))
    json.dump(compounds, open(os.path.join(BANK, "sciplex_compounds.json"), "w"), indent=1)
    db = os.path.join(BANK, "index.sqlite")
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, gene TEXT, ensembl TEXT, screen TEXT, split TEXT)")
    con.executemany("INSERT INTO records VALUES (?,?,?,?,?)", [(i, meta["gene"][i], meta["ens"][i], meta["screen"][i], meta["split"][i]) for i in range(len(meta["gene"]))])
    con.execute("CREATE TABLE drugs (chembl TEXT PRIMARY KEY, name TEXT, type TEXT, max_stage TEXT, n_mechanisms INT, n_warnings INT, n_stopped INT)")
    con.executemany("INSERT INTO drugs VALUES (?,?,?,?,?,?,?)", [(c, d["name"], d["type"], d["max_stage"], len(d["mechanisms"]), len(d["warnings"]), len(d["stopped"])) for c, d in drugs.items()])
    con.execute("CREATE TABLE mechanisms (chembl TEXT, target_ensembl TEXT, symbol TEXT, action TEXT)")
    con.executemany("INSERT INTO mechanisms VALUES (?,?,?,?)", [(c, m["target"], m["symbol"], m["action"]) for c, d in drugs.items() for m in d["mechanisms"]])
    con.execute("CREATE INDEX by_gene ON records(gene)"); con.execute("CREATE INDEX by_symbol ON mechanisms(symbol)"); con.commit(); con.close()
    open(os.path.join(BANK, "journal.jsonl"), "a").close()
    matched = [p for p in compounds if compounds[p]["chembl"]]
    rep = dict(sciplex=info, compounds=len(compounds), matched_to_chembl=len(matched), with_lof_target=sum(bool(compounds[p]["lof_targets"]) for p in compounds),
               drugs=len(drugs), drugs_with_mechanism=sum(bool(d["mechanisms"]) for d in drugs.values()), target_genes=len(targets),
               stopped_trial_records=sum(len(d["stopped"]) for d in drugs.values()), warnings=sum(len(d["warnings"]) for d in drugs.values()),
               drug_records=len(keys), total_records=len(meta["gene"]), bytes_bank=int(codes.nbytes + scale.nbytes + old.basis.nbytes), seconds=time.time() - t0)
    json.dump(rep, open(os.path.join(BANK, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def drug_inputs(G, bank):
    """Per gene: drug-development features (a) and the mean 10 uM response of loss-of-function drugs on it (b)."""
    torch, np = G["torch"], G["np"]
    targets = json.load(open(os.path.join(BANK, "targets.json")))
    comp = json.load(open(os.path.join(BANK, "sciplex_compounds.json")))
    meta = json.load(open(os.path.join(BANK, "records_meta.json")))
    tract = sorted({x for t in targets.values() for x in t["tractability"]})
    ti = {x: i for i, x in enumerate(tract)}
    Fe = torch.zeros(G["NPH"], 7 + len(tract)); Dr = torch.zeros(G["NPH"], 129)
    rec10 = {v["compound"]: int(k) for k, v in meta["drug"].items() if v["dose_nM"] == 10000.0}
    by_target = collections.defaultdict(list)
    for p, c in comp.items():
        for g in c["lof_targets"]:
            if p in rec10:
                by_target[g].append(rec10[p])
    for j, (g, e) in enumerate(zip(G["KN"], G["KE"])):
        t = targets.get(e or "")
        if t:
            lof = [d for d in t["drugs"] if d["action"] in LOF]
            Fe[j, :7] = torch.tensor([math.log1p(len(t["drugs"])), math.log1p(len(lof)), max([stage_rank(d["stage"]) for d in t["drugs"]] + [-1]) + 1,
                                      math.log1p(len(t["safety_events"])), float(bool(t["probes"])), math.log1p(t["stopped_trials"]), math.log1p(t["warnings"])])
            for x in t["tractability"]:
                Fe[j, 7 + ti[x]] = 1.0
        if by_target.get(g):
            Dr[j, :128] = torch.tensor(bank.z(by_target[g]).mean(0)); Dr[j, 128] = 1.0
    return Fe, Dr


def engine():
    t0 = time.time()
    G = cb4.notebook(); bank = cb4.Bank(BANK); torch = G["torch"]
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    Pt, Ct = cb4.inputs(G, bank, False)
    oC, oD, oE = cb5.own_inputs(G, bank, ("C", "D", "E"))
    Fe, Dr = drug_inputs(G, bank)
    X = dict(E3=cb4.scaled(G, [Pt, Ct, oC, oD, oE]), E4=cb4.scaled(G, [Pt, Ct, oC, oD, oE, Fe, Dr]))
    cb4.SCREENS = cb5.ALL                             # snap memory: screens A-E, as cellbank5's E3
    preds, info = {}, {}
    for arm in ("E3", "E4"):
        outs = []
        for s in SEEDS:
            pz, best, _ = cb4.run_arm(G, bank, arm, s, X[arm])
            outs.append(pz); info.setdefault(arm, []).append(round(best, 4))
            print(f"  {arm} seed {s}: held-out TRAIN r {best:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
        preds[arm] = torch.stack(outs).mean(0)
    torch.save(preds, os.path.join(CACHE, "engine_preds.pt"))
    json.dump(dict(heldout_train_r=info, drug_feature_genes=int((Fe.abs().sum(1) > 0).sum()), drug_response_genes=int(Dr[:, 128].sum())),
              open(os.path.join(OUTD, "engine_info.json"), "w"), indent=1)
    print(f"engine arms done ({time.time() - t0:.0f}s)")


def score():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    G = cb4.notebook(); bank = cb4.Bank(BANK); torch, np = G["torch"], G["np"]
    from scipy.stats import binomtest
    pr, Vt = G["pearson_rows"], torch.tensor(bank.basis.astype(np.float32)).T
    rep = json.load(open(os.path.join(BANK, "build_report.json")))
    meta = json.load(open(os.path.join(BANK, "records_meta.json")))
    comp = json.load(open(os.path.join(BANK, "sciplex_compounds.json")))
    drugs = json.load(open(os.path.join(BANK, "drugs.json")))
    P_(RULE); P_("DRUG DEVELOPMENT IN THE MEMORY BANK: DRUG RESPONSES IN K562, TARGETS, TRIALS AND SAFETY"); P_(RULE)
    P_(f"  sci-Plex3 K562: {rep['sciplex']['cells']:,} cells ({rep['sciplex']['vehicle_cells']:,} vehicle), {rep['compounds']} compounds x 4 doses, "
       f"{rep['sciplex']['genes_kept']:,} expressed genes mapped")
    P_(f"  compounds matched to ChEMBL {rep['matched_to_chembl']}; with a loss-of-function target {rep['with_lof_target']}")
    P_(f"  Open Targets: {rep['drugs']:,} drugs ({rep['drugs_with_mechanism']:,} with a mechanism), {rep['target_genes']:,} target genes, "
       f"{rep['stopped_trial_records']:,} stopped-trial records, {rep['warnings']:,} warnings")
    P_(f"  bank: {rep['total_records']:,} records ({rep['drug_records']} drug records), {rep['bytes_bank'] / 1e6:.1f} MB")

    # T1: drug response vs its targets' knockdowns
    kd = {}
    for g, d in bank.by_gene.items():
        if "A" in d:
            kd[g] = d["A"]
        elif "B" in d:
            kd[g] = d["B"]
    kd_genes = sorted(kd); KD = torch.tensor(bank.z([kd[g] for g in kd_genes])) @ Vt
    KDu = (KD - KD.mean(1, keepdim=True)); KDu = KDu / KDu.norm(dim=1, keepdim=True).clamp(min=1e-12)
    gpos = {g: i for i, g in enumerate(kd_genes)}
    rec = {(v["compound"], v["dose_nM"]): int(k) for k, v in meta["drug"].items()}
    rows, strength = [], {}
    for p, c in comp.items():
        if (p, 10000.0) not in rec:
            continue
        v = torch.tensor(bank.z([rec[(p, 10000.0)]])[0]) @ Vt
        vu = (v - v.mean()); strength[p] = float(np.abs(bank.z([rec[(p, 10000.0)]])[0]).sum()); vu = vu / vu.norm().clamp(min=1e-12)
        r_all = (KDu @ vu).numpy()
        tg = [g for g in c["lof_targets"] if g in gpos]
        if tg:
            rt = float(np.mean([r_all[gpos[g]] for g in tg])); null = float(np.median(r_all))
            pct = float(np.mean([np.mean(r_all < r_all[gpos[g]]) for g in tg]))
            rows.append(dict(compound=p, targets=tg, r_target=rt, null=null, percentile=pct))
        top = np.argsort(-r_all)[:5]
        c["top_knockdowns"] = [[kd_genes[i], round(float(r_all[i]), 3)] for i in top]
    d = np.array([x["r_target"] - x["null"] for x in rows]); w1, l1 = int((d > 0).sum()), int((d < 0).sum())
    p1 = binomtest(w1, w1 + l1, 0.5).pvalue if w1 + l1 else 1.0; md1 = float(np.median(d)) if len(d) else 0.0
    v1 = "BETTER" if p1 < 0.05 and md1 >= 0.01 else "WORSE" if p1 < 0.05 and md1 <= -0.01 else "NO DIFFERENCE"
    P_(f"  T1 PRIMARY drug response vs its targets' knockdowns, {len(rows)} compounds: median r_target {np.median([x['r_target'] for x in rows]):+.3f} "
       f"vs null {np.median([x['null'] for x in rows]):+.3f}; higher for {w1}, lower for {l1}; median gain {md1:+.3f}, p {p1:.2g} -> {v1}")
    P_(f"     targets' median percentile among all {len(kd_genes):,} knockdowns: {np.median([x['percentile'] for x in rows]):.0%}")
    for x in sorted(rows, key=lambda x: -x["r_target"])[:8]:
        P_(f"     {x['compound'][:34]:34s} targets {','.join(x['targets'])[:30]:30s} r {x['r_target']:+.3f} (null {x['null']:+.3f}, percentile {x['percentile']:.0%})")

    # T2: discovery list -> journal
    bank_j = cb4.Bank(BANK); n_h = 0
    P_("  T2 the 30 strongest drug responses and the knockdowns they most resemble (UNVALIDATED):")
    for p in sorted(strength, key=lambda q: -strength[q])[:30]:
        c = comp[p]; dev = [drugs[x] for x in c["chembl"] if x in drugs]
        stage = max([x["max_stage"] or "" for x in dev], key=stage_rank, default="")
        stops = sum(len(x["stopped"]) for x in dev); warns = sum(len(x["warnings"]) for x in dev)
        tops = c.get("top_knockdowns", [])
        hit = any(g in c["lof_targets"] for g, _ in tops)
        P_(f"     {p[:34]:34s} targets {','.join(c['lof_targets'])[:28] or '-':28s} | like knockdown of {', '.join(f'{g} {r:+.2f}' for g, r in tops[:3])}"
           f"{' | TARGET IN TOP 5' if hit else ''} | stage {stage or '-'}, stopped trials {stops}, warnings {warns}")
        for g, r in tops:
            bank_j.add(dict(type="hypothesis", status="UNVALIDATED", claim=f"{p} at 10 uM acts on K562 like knocking down {g}", r=r,
                            source="cellbank6 T2: sci-Plex3 K562 response vs bank knockdown records", annotated_lof_targets=c["lof_targets"],
                            development=dict(max_stage=stage, stopped_trials=stops, warnings=warns))); n_h += 1
    P_(f"     {n_h} hypotheses written to memory_bank/cell_v6/journal.jsonl")
    json.dump(comp, open(os.path.join(BANK, "sciplex_compounds.json"), "w"), indent=1)

    # G8: does the drug data help predict knockdowns?
    A_te, Vy, KN, rte = G["A_te"], G["Vy"].cpu(), G["KN"], G["rte"]
    preds = torch.load(os.path.join(CACHE, "engine_preds.pt"))
    Rr = {a: torch.nan_to_num(pr(preds[a] @ Vy.T, A_te)).numpy() for a in ("E3", "E4")}
    Fe, Dr = drug_inputs(G, bank)
    te_rows = [int(rte[t]) for t in range(len(rte))]
    has_dr = np.array([bool(Dr[j, 128]) for j in te_rows]); has_any = np.array([bool(Fe[j].abs().sum() > 0) or bool(Dr[j, 128]) for j in te_rows])
    V = {}
    for tag, m in (("G8 PRIMARY, all TEST genes", np.ones(len(te_rows), bool)), ("   genes with a drug response (descr.)", has_dr),
                   ("   genes with any drug data (descr.)", has_any)):
        dd = Rr["E4"][m] - Rr["E3"][m]; w, l = int((dd > 0).sum()), int((dd < 0).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0; md = float(np.median(dd)) if len(dd) else 0.0
        v = "BETTER" if p < 0.05 and md >= 0.01 else "WORSE" if p < 0.05 and md <= -0.01 else "NO DIFFERENCE"
        V[tag.strip()] = dict(n=int(m.sum()), better=w, worse=l, p=float(p), median_diff=md, verdict=v)
        P_(f"  {tag:40s} n {m.sum():5d}: E4 {np.median(Rr['E4'][m]):+.3f} vs E3 {np.median(Rr['E3'][m]):+.3f}; better {w} / worse {l}, "
           f"median {md:+.4f}, p {p:.2g} -> {v if tag.startswith('G8') else '(descriptive)'}")
    open(OUT, "w").write("\n".join(out) + "\n")
    json.dump(dict(T1=dict(n=len(rows), better=w1, worse=l1, p=p1, median_gain=md1, verdict=v1, rows=rows), G8=V), open(os.path.join(OUTD, "results.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "engine": engine, "score": score}[sys.argv[1]]()
