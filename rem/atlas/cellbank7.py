"""What clinical trials and drug development revealed about each gene, turned into cell knowledge.

WHY. The user: use the data produced in clinical trials and drug development, and what came to light there,
to aid the cell data. When a drug that blocks gene X is given to people, the trials and the post-marketing
reports record what blocking X does in a whole human: which tissues suffer (blood counts fall, the liver or
heart is hurt), whether the drug worked, why trials stopped. Preclinical safety studies record what blocking X
did in cells and animals. That is evidence about the gene's job, gathered at great cost, which the cell model
has not used. cellbank6 used only coarse counts of it; here it becomes a profile per gene.

THE CLINICAL PROFILE OF A GENE (from Open Targets 26.09, CC0; only drugs whose action on the gene is loss of
function, the same list as cellbank6, since a knockdown is loss of function)
  adverse events in patients (openFDA significant reactions, all of them per drug, including the parent
     molecule's): 14 organ / process categories by fixed keyword rules (blood, liver, heart, kidney, gut, skin,
     nerve, infection, metabolic, lung, immune, bleeding-vascular, tumour, death); per category the share of
     the gene's drugs with such a reaction and the mean log1p summed log-likelihood ratio;
  drug warnings: the 12 most common toxicity classes (black box warnings, withdrawals), log1p counts;
  trials (clinical reports with a trial status): log1p number, share terminated / withdrawn / suspended,
     share of stopped trials per reason (negative result, safety, business, enrolment, design, other),
     highest clinical stage;
  indications: share of the gene's drugs developed for blood cancers, and for any cancer;
  preclinical safety events on the gene (ToxCast, literature): the same 14 categories plus 'growth', log1p.
  Stored as memory_bank/cell_v7/clinical_profiles.json (numbers) and clinical_facts.json (readable text per
  gene, with the drugs named), as a layer on cell_v6 (records unchanged).

GATES, PREDECLARED
  C1  consistency, TRAIN: among screen-A TRAIN knockdowns of genes with >= 1 loss-of-function drug that has
      openFDA data, are the K562 responses (norm of the specific response) larger for genes whose drugs
      cause blood adverse events in patients than for the others? One-sided Mann-Whitney p < 0.05 ->
      CONSISTENT, else NOT SHOWN.
  C2  the same test on screen-A TEST knockdowns (a replication; not a prediction).
  G9  PRIMARY: E5 = E3 + the clinical profile (with a present flag) vs E3 (rerun here with the same seeds), on
      TEST genes that have a clinical profile. BETTER if p < 0.05 and median gain >= +0.01; WORSE mirrored.
  G10 the same on all 2,525 TEST genes.
NOT: keyword categories are coarse; drug effects in patients mix on- and off-target actions and the disease;
     only genes that are drug targets have a profile.

Run: python3 -P rem/atlas/cellbank7.py build | engine | score
"""

from __future__ import annotations
import collections
import glob
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
_l = lambda n: importlib.util.spec_from_file_location(n, os.path.join(HERE, f"{n}.py"))
_s = _l("cellbank4"); cb4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb4)
_s = _l("cellbank5"); cb5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb5)
_s = _l("cellbank6"); cb6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb6)
LAYER = R("memory_bank", "cell_v7")
OUTD = R("outputs", "cellbank7")
CACHE = os.path.join(HERE, "_cache", "cellbank7")
OUT = os.path.join(HERE, "RESULTS_cellbank7.txt")
CATS = {
    "blood": ["anaemia", "anemia", "neutropenia", "thrombocytopenia", "leukopenia", "pancytopenia", "cytopenia", "agranulocytosis",
              "bone marrow", "myelosuppression", "haemoglobin", "hemoglobin", "platelet count", "white blood cell", "neutrophil",
              "lymphopenia", "haemolysis", "hemolysis", "red blood cell", "hematological", "haematological"],
    "liver": ["hepat", "liver", "transaminase", "aminotransferase", "bilirubin", "jaundice", "cholesta", "steatosis"],
    "heart": ["cardiac", "heart", "arrhythmia", "qt prolonged", "tachycardia", "bradycardia", "myocardial", "atrial fibrillation", "ejection fraction", "cardio"],
    "kidney": ["renal", "kidney", "creatinine", "nephr"],
    "gut": ["nausea", "vomiting", "diarrhoea", "diarrhea", "colitis", "abdominal", "stomatitis", "mucositis", "gastro", "constipation"],
    "skin": ["rash", "dermatitis", "alopecia", "pruritus", "skin", "erythema", "palmar-plantar", "dermatolog"],
    "nerve": ["neuropathy", "seizure", "convulsion", "headache", "dizziness", "encephalopathy", "neurotox", "somnolence", "confusion", "coma"],
    "infection": ["infection", "sepsis", "pneumonia", "septic", "herpes", "fungal", "infectious"],
    "metabolic": ["hyperglycaemia", "hyperglycemia", "hypokalaemia", "hypokalemia", "hyponatraemia", "lactic acidosis", "hyperlipid",
                  "diabetes", "weight", "metabolic"],
    "lung": ["pneumonitis", "interstitial lung", "dyspnoea", "dyspnea", "pulmonary", "respiratory"],
    "immune": ["hypersensitivity", "anaphyla", "cytokine release", "autoimmune", "immune"],
    "vascular": ["haemorrhage", "hemorrhage", "bleeding", "thrombosis", "embolism", "hypertension", "oedema", "edema", "blood pressure", "vascular"],
    "tumour": ["neoplasm", "cancer", "carcinoma", "leukaemia", "leukemia", "lymphoma", "malignan", "carcinogen"],
    "death": ["death", "fatal"],
}
SAFETY_CATS = list(CATS) + ["growth"]
STOPS = ["Negative", "Safety_Sideeffects", "Business_Administrative", "Insufficient_Enrollment", "Study_Design"]
BLOOD_CANCER = ["leukemia", "leukaemia", "lymphoma", "myeloma", "myelodysplastic", "myeloproliferative", "polycythemia", "myelofibrosis"]
CANCER = BLOOD_CANCER + ["cancer", "carcinoma", "tumor", "tumour", "neoplasm", "sarcoma", "melanoma", "glioma", "blastoma"]
SEEDS, RULE = (0, 1, 2), "=" * 97


def cats_of(text):
    t = (text or "").lower()
    out = [k for k, words in CATS.items() if any(w in t for w in words)]
    if "growth" in t or "proliferation" in t:
        out.append("growth")
    return out


def build():
    t0 = time.time()
    G = cb4.notebook(); torch, np = G["torch"], G["np"]
    V6 = cb6.BANK
    targets = json.load(open(os.path.join(V6, "targets.json")))
    drugs = json.load(open(os.path.join(V6, "drugs.json")))
    parent = {c: d["parent"] for c, d in drugs.items() if d.get("parent") and d["parent"] != "None"}
    ae = collections.defaultdict(list)
    for a in cb6.read("openfda_significant_adverse_drug_reactions"):
        ae[a["chembl_id"]].append((a["event"], float(a["llr"])))
    trials = collections.defaultdict(list)
    for rep in cb6.read("clinical_report"):
        if rep["trialOverallStatus"]:
            for d in rep["drugs"] or []:
                trials[d["drugId"]].append((rep["trialOverallStatus"], rep["trialStopReasonCategories"] or [], rep["trialPhase"]))
    ind = collections.defaultdict(list)
    for ct in cb6.read("clinical_target"):
        ind[(ct["drugId"], ct["targetId"])] += [x["diseaseFromSource"] for x in (ct["diseases"] or []) if x and x.get("diseaseFromSource")]
    tox = collections.Counter(w["toxicity"] for d in drugs.values() for w in d["warnings"] if w["toxicity"])
    TOX = [t for t, _ in tox.most_common(12)]
    width = 2 * len(CATS) + len(TOX) + 3 + len(STOPS) + 1 + 1 + 2 + len(SAFETY_CATS)
    profiles, facts = {}, {}
    for e, t in targets.items():
        lof = [d for d in t["drugs"] if d["action"] in cb6.LOF]
        chem = sorted({d["chembl"] for d in lof})
        if not chem and not t["safety_events"]:
            continue
        v = np.zeros(width, dtype=np.float32); k = 0
        evs = {c: ae.get(c, []) + ae.get(parent.get(c, ""), []) for c in chem}
        with_ae = [c for c in chem if evs[c]]
        examples = collections.defaultdict(collections.Counter)
        for cat in CATS:
            fr, ll = [], []
            for c in with_ae:
                s = sum(l for ev, l in evs[c] if cat in cats_of(ev))
                fr.append(float(s > 0)); ll.append(math.log1p(s))
                for ev, l in evs[c]:
                    if cat in cats_of(ev):
                        examples[cat][ev] += l
            v[k] = np.mean(fr) if fr else 0.0; v[k + 1] = np.mean(ll) if ll else 0.0; k += 2
        for tc in TOX:
            v[k] = math.log1p(sum(1 for c in chem for w in drugs.get(c, {}).get("warnings", []) if w["toxicity"] == tc)); k += 1
        tr = [x for c in chem for x in trials.get(c, [])]
        stopped = [x for x in tr if x[0] in ("TERMINATED", "WITHDRAWN", "SUSPENDED")]
        v[k] = math.log1p(len(tr)); v[k + 1] = len(stopped) / len(tr) if tr else 0.0; v[k + 2] = float(bool(tr)); k += 3
        for s in STOPS:
            v[k] = sum(s in x[1] for x in stopped) / len(stopped) if stopped else 0.0; k += 1
        v[k] = sum(not any(s in x[1] for s in STOPS) for x in stopped) / len(stopped) if stopped else 0.0; k += 1
        v[k] = (max([cb6.stage_rank(d["stage"]) for d in lof] + [-1]) + 1) / len(cb6.STAGE); k += 1
        dis = {c: " ".join(ind.get((c, e), [])).lower() for c in chem}
        v[k] = np.mean([any(w in dis[c] for w in BLOOD_CANCER) for c in chem]) if chem else 0.0
        v[k + 1] = np.mean([any(w in dis[c] for w in CANCER) for c in chem]) if chem else 0.0; k += 2
        sc = collections.Counter(x for ev in t["safety_events"] for x in cats_of(ev["event"]))
        for cat in SAFETY_CATS:
            v[k] = math.log1p(sc[cat]); k += 1
        assert k == width
        profiles[e] = v.tolist()
        names = [drugs[c]["name"] for c in chem if c in drugs][:6]
        top = sorted(((v[2 * i], cat) for i, cat in enumerate(CATS)), reverse=True)
        harms = [f"{cat} ({', '.join(ev for ev, _ in examples[cat].most_common(3))})" for fr_, cat in top if fr_ >= 0.5][:5]
        stop_why = collections.Counter(s for x in stopped for s in x[1]).most_common(3)
        facts[e] = dict(
            symbol="",
            drugs_that_block_it=names, n_blocking_drugs=len(chem),
            highest_stage=max([d["stage"] or "" for d in lof], key=cb6.stage_rank, default=""),
            in_patients=harms, trials=len(tr), trials_stopped=len(stopped), stop_reasons=[f"{s} x{n}" for s, n in stop_why],
            warnings=sorted({w["toxicity"] for c in chem for w in drugs.get(c, {}).get("warnings", []) if w["toxicity"]}),
            developed_for_blood_cancer=bool(v[k - len(SAFETY_CATS) - 2] > 0),
            preclinical_safety=[f"{ev['event']} ({ev['source']})" for ev in t["safety_events"][:6]])
    sym = {e: g for g, e in zip(G["KN"], G["KE"]) if e}
    for e, f in facts.items():
        f["symbol"] = sym.get(e) or G["HPA"].get(e, {}).get("sym", "")
    os.makedirs(LAYER, exist_ok=True)
    layout = [x for c in CATS for x in (f"ae_share:{c}", f"ae_llr:{c}")] + [f"warning:{t}" for t in TOX] + \
             ["trials_log1p", "share_stopped", "has_trials"] + [f"stop:{s}" for s in STOPS] + ["stop:other", "highest_stage"] + \
             ["blood_cancer_indication", "cancer_indication"] + [f"preclinical:{c}" for c in SAFETY_CATS]
    json.dump(dict(layout=layout, profiles=profiles), open(os.path.join(LAYER, "clinical_profiles.json"), "w"))
    json.dump(facts, open(os.path.join(LAYER, "clinical_facts.json"), "w"), indent=0)
    kn = set(G["KE"]); te = {G["KE"][int(i)] for i in G["rte"]}
    rep = dict(genes_with_profile=len(profiles), knocked_down_genes_with_profile=len(kn & set(profiles)),
               test_genes_with_profile=len(te & set(profiles)), genes_with_patient_adverse_events=sum(1 for e in profiles if profiles[e][0:2 * len(CATS):2] and max(profiles[e][0:2 * len(CATS):2]) > 0),
               width=width, toxicity_classes=TOX, seconds=time.time() - t0)
    json.dump(rep, open(os.path.join(LAYER, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def clinical_inputs(G):
    torch = G["torch"]
    P = json.load(open(os.path.join(LAYER, "clinical_profiles.json")))
    w = len(P["layout"])
    M = torch.zeros(G["NPH"], w + 1)
    for j, e in enumerate(G["KE"]):
        if e and e in P["profiles"]:
            M[j, :w] = torch.tensor(P["profiles"][e]); M[j, w] = 1.0
    return M, P["layout"]


def engine():
    t0 = time.time()
    G = cb4.notebook(); bank = cb4.Bank(cb6.BANK); torch = G["torch"]
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    Pt, Ct = cb4.inputs(G, bank, False)
    oC, oD, oE = cb5.own_inputs(G, bank, ("C", "D", "E"))
    Cl, _ = clinical_inputs(G)
    X = dict(E3=cb4.scaled(G, [Pt, Ct, oC, oD, oE]), E5=cb4.scaled(G, [Pt, Ct, oC, oD, oE, Cl]))
    cb4.SCREENS = cb5.ALL
    preds, info = {}, {}
    for arm in ("E3", "E5"):
        outs = []
        for s in SEEDS:
            pz, best, _ = cb4.run_arm(G, bank, arm, s, X[arm])
            outs.append(pz); info.setdefault(arm, []).append(round(best, 4))
            print(f"  {arm} seed {s}: held-out TRAIN r {best:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
        preds[arm] = torch.stack(outs).mean(0)
    torch.save(preds, os.path.join(CACHE, "engine_preds.pt"))
    json.dump(dict(heldout_train_r=info), open(os.path.join(OUTD, "engine_info.json"), "w"), indent=1)
    print(f"engine arms done ({time.time() - t0:.0f}s)")


def score():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    G = cb4.notebook(); torch, np = G["torch"], G["np"]
    from scipy.stats import binomtest, mannwhitneyu
    rep = json.load(open(os.path.join(LAYER, "build_report.json")))
    Cl, layout = clinical_inputs(G)
    blood = Cl[:, layout.index("ae_share:blood")] > 0
    has_ae = Cl[:, :2 * len(CATS):2].sum(1) > 0
    has = Cl[:, -1] > 0
    P_(RULE); P_("WHAT CLINICAL TRIALS AND DRUG DEVELOPMENT REVEALED ABOUT EACH GENE, USED AS CELL KNOWLEDGE"); P_(RULE)
    P_(f"  genes with a clinical profile {rep['genes_with_profile']:,}; knocked down in screen A {rep['knocked_down_genes_with_profile']:,}; "
       f"TEST {rep['test_genes_with_profile']:,}; with patient adverse events {rep['genes_with_patient_adverse_events']:,}")

    def consistency(rows, Y, tag):
        norm = torch.nan_to_num(Y).norm(dim=1).cpu().numpy()
        sel = [i for i, j in enumerate(rows) if bool(has_ae[j])]
        a = np.array([norm[i] for i in sel if bool(blood[rows[i]])]); b = np.array([norm[i] for i in sel if not bool(blood[rows[i]])])
        p = mannwhitneyu(a, b, alternative="greater").pvalue if len(a) and len(b) else 1.0
        P_(f"  {tag}: K562 response size, genes whose drugs cause blood adverse events {np.median(a):.2f} (n {len(a)}) vs other drug targets "
           f"{np.median(b):.2f} (n {len(b)}); one-sided p {p:.2g} -> {'CONSISTENT' if p < 0.05 else 'NOT SHOWN'}")
        return dict(n_blood=len(a), n_other=len(b), median_blood=float(np.median(a)) if len(a) else None,
                    median_other=float(np.median(b)) if len(b) else None, p=float(p))
    rtr, rte = [int(i) for i in G["rtr"]], [int(i) for i in G["rte"]]
    C = dict(C1=consistency(rtr, G["Y_tr"], "C1 TRAIN"), C2=consistency(rte, G["A_te"], "C2 TEST (replication)"))

    A_te, Vy, pr = G["A_te"], G["Vy"].cpu(), G["pearson_rows"]
    preds = torch.load(os.path.join(CACHE, "engine_preds.pt"))
    Rr = {a: torch.nan_to_num(pr(preds[a] @ Vy.T, A_te)).numpy() for a in ("E3", "E5")}
    m_has = np.array([bool(has[j]) for j in rte]); V = {}
    for tag, m in (("G9 PRIMARY, TEST genes with a clinical profile", m_has), ("G10 all TEST genes", np.ones(len(rte), bool))):
        d = Rr["E5"][m] - Rr["E3"][m]; w, l = int((d > 0).sum()), int((d < 0).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0; md = float(np.median(d))
        v = "BETTER" if p < 0.05 and md >= 0.01 else "WORSE" if p < 0.05 and md <= -0.01 else "NO DIFFERENCE"
        V[tag.split()[0]] = dict(n=int(m.sum()), better=w, worse=l, p=float(p), median_diff=md, verdict=v,
                                 E5=float(np.median(Rr["E5"][m])), E3=float(np.median(Rr["E3"][m])))
        P_(f"  {tag:48s} n {m.sum():5d}: E5 {np.median(Rr['E5'][m]):+.3f} vs E3 {np.median(Rr['E3'][m]):+.3f}; better {w} / worse {l}, "
           f"median {md:+.4f}, p {p:.2g} -> {v}")
    open(OUT, "w").write("\n".join(out) + "\n")
    json.dump(dict(consistency=C, verdicts=V), open(os.path.join(OUTD, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    {"build": build, "engine": engine, "score": score}[sys.argv[1]]()
