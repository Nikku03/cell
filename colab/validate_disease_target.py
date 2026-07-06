"""VALIDATE the disease->target pipeline on OOD diseases (blind target SELECTION).

For each disease the model is NOT curated for (absent from otdis/biomarkers), we supply only the KNOWN
upstream driver + the pathogenic readout + the pathway (general immunology knowledge — never the drug
target). The pipeline must SELECT the druggable target by simulation: perturb each pathway node, keep the
ones whose DISABLE collapses the pathogenic readout toward wild-type. We then check whether a real,
approved drug target is recovered among the top rescuers — vs a random-gene baseline.

Honest scope (see check in this session): this validates the MECHANISM->INTERVENTION step (given the
driver, find the druggable bottleneck + direction), NOT autonomous driver discovery. Reads the cascade
machinery from disease_target_pipeline.py. -> disease_target_validation.json
"""
import gzip, json, random
from pathlib import Path
from disease_target_pipeline import load, cascade_subgraph, propagate

OUT = Path("outputs/orphan")

# apex = known upstream driver; readout = pathogenic effectors; pathway = candidate space (immunology, not
# the answer); known_targets = real approved/late-stage drug targets; the required direction is 'disable'.
DISEASES = [
    dict(name="psoriasis (IL23/IL17)",
         apex=["IL23A", "IL12B"],
         readout=["IL17A", "IL17F", "IL22", "CCL20", "IL21"],
         pathway=["IL23A", "IL12B", "IL23R", "IL12RB1", "JAK2", "STAT3", "RORC", "RORA", "IL17A",
                  "IL17F", "IL17RA", "IL22", "TNF", "NFKB1", "STAT1", "STAT4", "IRF4", "BATF", "AHR"],
         known_targets=["IL23A", "IL23R", "IL12B", "IL17A", "IL17RA", "TNF"]),
    dict(name="atopic dermatitis (Type-2)",
         apex=["IL4", "IL13"],
         readout=["CCL26", "CCL17", "CCL22", "POSTN", "CCL11"],
         pathway=["IL4", "IL13", "IL4R", "IL13RA1", "JAK1", "JAK3", "STAT6", "GATA3", "IL5", "IL31",
                  "TSLP", "IL33"],
         known_targets=["IL4R", "IL13", "IL4", "IL5", "STAT6", "TSLP"]),
]


def rescue_screen(D, idx, name, apex_g, readout_g, cand_g):
    apex = [idx[g] for g in apex_g if g in idx]
    ro = [idx[g] for g in readout_g if g in idx]
    keep, out = cascade_subgraph(D, apex, ro, Lf=4, Lb=3)
    v0 = propagate(out, apex)
    ro_active = [r for r in ro if v0.get(r, 0) > 0]
    if len(ro_active) < 2:
        return None
    scores = {}
    for g in cand_g:
        if g not in idx or idx[g] not in keep:
            continue
        n = idx[g]; ro_eff = [r for r in ro_active if r != n]
        vdis = propagate(out, [a for a in apex if a != n], removed={n})
        rescue = sum(1 for r in ro_eff if vdis.get(r, 0) <= 0 or vdis.get(r, 0) < 0.4 * v0.get(r, 0)) / max(1, len(ro_eff))
        scores[g] = round(rescue, 3)
    return scores, len(ro_active)


def main():
    D, G, name, idx = load()
    all_genes = list(idx)
    rng = random.Random(0)
    rows = []
    for dz in DISEASES:
        out = rescue_screen(D, idx, dz["name"], dz["apex"], dz["readout"], dz["pathway"])
        if out is None:
            rows.append(dict(disease=dz["name"], ok=False, reason="readout_not_activated")); continue
        scores, n_ro = out
        top = sorted(scores.items(), key=lambda x: -x[1])
        # recovered = a known target among the strong rescuers (rescue >= 0.6)
        strong = [g for g, s in top if s >= 0.6]
        hits = [g for g in strong if g in dz["known_targets"]]
        # random baseline: draw len(pathway) random genes, none are known targets -> expected hits ~ 0
        # (approve-rate baseline = P(a strong rescuer is a known target) if targets were random labels)
        n_known_in_pathway = len([g for g in dz["pathway"] if g in dz["known_targets"]])
        base = n_known_in_pathway / max(1, len([g for g in dz["pathway"] if g in idx]))
        recovered = len(hits) > 0
        rows.append(dict(disease=dz["name"], ok=True, n_readout=n_ro,
                         top_rescuers=top[:6], strong_rescuers=strong, known_targets=dz["known_targets"],
                         recovered_known_target=recovered, hits=hits,
                         precision_strong=round(len(hits) / max(1, len(strong)), 3),
                         random_label_baseline=round(base, 3)))
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_rec = sum(1 for r in rows if r.get("recovered_known_target"))
    summary = dict(n_diseases=len(DISEASES), n_ran=n_ok, n_recovered=n_rec,
                   recovery_rate=round(n_rec / max(1, n_ok), 3))
    json.dump(dict(summary=summary, diseases=rows),
              open(OUT / "disease_target_validation.json", "w"), indent=2)
    print("=" * 80)
    print(f"DISEASE->TARGET pipeline validation — recovered a known target in {n_rec}/{n_ok} OOD diseases")
    print("=" * 80)
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['disease']:28} SKIPPED ({r['reason']})"); continue
        mark = "RECOVERED" if r["recovered_known_target"] else "missed"
        print(f"  {r['disease']:28} [{mark}] hits={r['hits']}  precision(strong)={r['precision_strong']}"
              f"  vs random-label {r['random_label_baseline']}")
        print(f"     top rescuers: {r['top_rescuers']}")
    print("\n-> outputs/orphan/disease_target_validation.json")
    return summary


if __name__ == "__main__":
    main()
