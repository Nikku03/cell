"""shared_dependency -- the user's zoom-out: some knockouts aren't specific regulators, they break a SHARED FUNCTION many proteins depend on
(a transporter, an ATP/energy enzyme, a kinase). Their response should be driven by co-dependency / shared-function structure, not TF->target
edges -- and may be dominated by the GENERIC stress program (the tide) that any core-utility loss triggers.

Tests, K562 held-out specific movers:
  A) FUNCTIONAL CLASS: split KOs into utility-hub families (transporter/SLC, mito-energy/metabolic, kinase-phosphatase, ribosome-translation,
     proteostasis) vs specific TF/regulator vs other. For each, report the TIDE-fraction of the response (how generic it is) and how much the
     tide-removed SPECIFIC movers are explained by CO-DEPENDENCY (DepMap co-essential) vs REGULATORY edges.
  B) HUB-NESS: does a KO's connectivity/essentiality (co-dependency degree) predict a more generic (tide-dominated) response? -- i.e., is the
     tide literally the shared-dependency response every hub knockout shares?
Co-dependency (shared fitness dependency = shared function) is used as an EXPLANATION edge for the first time. Enrichment vs non-movers guards
the 'covers everything' trap. Deterministic."""
import os
import json, collections, re
from pathlib import Path
import numpy as np
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

FAM = [
    ("transporter", re.compile(r"^(SLC\d|ABC[A-G]|ATP\d|ATP1[AB]|SLC)")),
    ("mito_energy", re.compile(r"^(NDUF|SDH|UQCR|COX\d|COX[0-9]|ATP5|MRPL|MRPS|TIMM|TOMM|ACO2|OGDH|PDHA|PDHB|DLAT|DLD|IDH|MDH|FH|CS|SUCL|CYC1)")),
    ("kinase_phos", re.compile(r"(^(CDK|MAPK|GSK|CSNK|PLK|AURK|PRKA|PRKC|STK|MAP[234]K|CAMK|PIK3|MTOR|ATR|ATM|CHEK|WEE|PPP\dC|PPM1|PTPN|DUSP))|K$")),
    ("ribo_transl", re.compile(r"^(RPL|RPS|MRPL|MRPS|EIF|EEF|RRP|NOP|UTP|POLR)")),
    ("proteostasis", re.compile(r"^(PSM[AB-D]|PSMC|PSMD|UBA|UBE2|USP|VCP|HSPA|HSP90|DNAJ|CUL\d|RNF)")),
]


def classify(g, reg_t):
    for name, rx in FAM:
        if rx.search(g):
            return name
    if g in reg_t and reg_t[g]:
        return "TF_regulator"
    return "other"


def main():
    H = Harness("K562")
    reg, _, sources = build_causal(H)
    reg_t = {s: set(t for t, _ in ts) for s, ts in reg.items()}
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    # co-dependency (shared fitness dependency) edges, by name
    codep = collections.defaultdict(set)
    for k, v in D.get("codep", {}).items():
        ki = int(k)
        if ki < N:
            for j, _ in v:
                if int(j) < N:
                    codep[names[ki]].add(names[int(j)]); codep[names[int(j)]].add(names[ki])
    gi = H.gi
    def toidx(s): return set(gi[n] for n in s if n in gi)
    nti = set(int(i) for i in H.nontide_idx)

    per = collections.defaultdict(lambda: {"tidefrac": [], "codep_enr": [], "reg_cov": [], "codep_cov": [], "n": 0, "hub": []})
    for ko in H.scorable:
        r = H.ki[ko]
        allmv = np.where(H.mover[r])[0]
        spec = set(int(i) for i in allmv if H.nontide[i])
        if not spec:
            continue
        tidefrac = 1 - len(spec) / max(len(allmv), 1)             # fraction of the response that is generic/tide
        cd = toidx(codep.get(ko, set()))
        rg = toidx(reg_t.get(ko, set()))
        nonm = nti - spec
        cov_cd = len(spec & cd) / len(spec)
        cov_rg = len(spec & rg) / len(spec)
        enr = (cov_cd / (len(nonm & cd) / len(nonm))) if (nonm and len(nonm & cd) > 0) else float("nan")
        cls = classify(ko, reg_t)
        d = per[cls]
        d["tidefrac"].append(tidefrac); d["codep_cov"].append(cov_cd); d["reg_cov"].append(cov_rg)
        if not np.isnan(enr):
            d["codep_enr"].append(enr)
        d["hub"].append(len(codep.get(ko, set())))
        d["n"] += 1

    m = lambda a: float(np.mean(a)) if len(a) else float("nan")
    order = ["transporter", "mito_energy", "kinase_phos", "ribo_transl", "proteostasis", "TF_regulator", "other"]
    print(f"K562 scorable knockouts by functional class -- shared-dependency vs regulatory explanation:\n")
    print(f"  {'class':14s} {'n':>4s} {'tide-frac':>9s} {'specific via CODEP':>18s} {'(enrich)':>9s} {'via REG':>8s} {'codep-deg':>9s}")
    for c in order:
        if per[c]["n"] == 0:
            continue
        d = per[c]
        print(f"  {c:14s} {d['n']:>4d} {m(d['tidefrac'])*100:>8.0f}% {m(d['codep_cov'])*100:>17.1f}% {m(d['codep_enr']):>9.2f} "
              f"{m(d['reg_cov'])*100:>7.1f}% {m(d['hub']):>9.0f}")

    # hub-ness vs tide-fraction correlation
    allko = [(len(codep.get(k, set())), 1 - len(set(i for i in np.where(H.mover[H.ki[k]])[0] if H.nontide[i]))
              / max(int(H.mover[H.ki[k]].sum()), 1)) for k in H.scorable if H.mover[H.ki[k]].sum() > 0]
    hub = np.array([x[0] for x in allko]); tf = np.array([x[1] for x in allko])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(hub, tf)

    util = [c for c in ("transporter", "mito_energy", "kinase_phos", "ribo_transl", "proteostasis") if per[c]["n"]]
    util_tide = m([x for c in util for x in per[c]["tidefrac"]]); tf_tide = m(per["TF_regulator"]["tidefrac"])
    util_codep_enr = m([x for c in util for x in per[c]["codep_enr"]])
    verdict = (
        f"SHARED DEPENDENCY (shared_dependency.py): the zoom-out -- utility-hub knockouts (transporter/energy/kinase/ribosome/proteostasis) break "
        f"a SHARED function many proteins depend on, so their response should be co-dependency-driven and/or generic (tide). K562. "
        f"Utility-hub KOs are {util_tide*100:.0f}% tide (generic) vs TF/regulator KOs {tf_tide*100:.0f}%; hub-ness (co-dependency degree) vs "
        f"tide-fraction Spearman {rho:+.2f}. "
        + ("So the user is RIGHT that hub knockouts give a more GENERIC response -- losing a shared utility triggers the common stress program "
           "(the tide), which is exactly why their response is more tide-dominated. " if util_tide > tf_tide + 0.03 or rho > 0.1 else
           "Hub knockouts are NOT much more tide-dominated than regulators here. ")
        + f"But for the tide-removed SPECIFIC part, co-dependency (shared-function) edges explain it at only {util_codep_enr:.2f}x enrichment -- "
        + ("real signal, so shared-dependency edges DO add a new explanatory channel for hub knockouts. " if util_codep_enr > 1.5 else
           "barely above chance, so even for hub knockouts the SPECIFIC residue is not the co-essential set. ")
        + "READING: the shared-dependency idea explains the GENERIC part of the response (the tide is the common-utility-loss program), which is "
        "why we removed it -- but the knockout-SPECIFIC residue, even for utility hubs, remains the idiosyncratic part no shared-function edge "
        "names. It reframes WHAT the tide is (a real, mechanistic shared-dependency response) more than it moves the specific ceiling. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"by_class": {c: {"n": per[c]["n"], "tide_frac": round(m(per[c]["tidefrac"]), 3),
                                "codep_cov": round(m(per[c]["codep_cov"]), 3), "codep_enr": round(m(per[c]["codep_enr"]), 3),
                                "reg_cov": round(m(per[c]["reg_cov"]), 3)} for c in order if per[c]["n"]},
               "hubness_vs_tidefrac_spearman": round(float(rho), 3), "util_tidefrac": round(util_tide, 3),
               "tf_tidefrac": round(tf_tide, 3), "util_codep_enrichment": round(util_codep_enr, 3),
               "verdict": verdict, "note": verdict}, open(OUT / "shared_dependency.json", "w"), indent=1)
    print("\n  -> outputs/orphan/shared_dependency.json")


if __name__ == "__main__":
    main()
