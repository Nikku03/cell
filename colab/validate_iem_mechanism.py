"""VALIDATE Play B — does the model recover known inborn-error-of-metabolism (IEM) biomarkers?

The cheapest NON-COMPUTATIONAL validation: when a metabolic enzyme is lost, its reaction's SUBSTRATE
accumulates — and for classic inborn errors of metabolism that accumulating metabolite IS the documented
clinical biomarker (PKU→phenylalanine, homocystinuria→homocysteine, ...). Ground truth = clinical
biochemistry, which the model was never trained on. So if the model's metabolic network (Human-GEM reactions
per enzyme) lists the known biomarker as a substrate of the disease enzyme, it has recovered real disease
mechanism from nature, not from its own fit.

Anti-cheat = SPECIFICITY: it's not enough that the biomarker appears somewhere; the disease enzyme must be
among the FEW enzymes whose substrates include it (else it's a common-metabolite artifact). We report recall
(biomarker ∈ the enzyme's substrates), specificity (how many of the 2,549 enzymes share that biomarker), and
a random-pair baseline. -> iem_mechanism_validation.json

Reads outputs/orphan/metab_reactions.json (generxn + names, from the model) — or cell_complete.json on Colab.
"""
import json, re, random
from pathlib import Path
from statistics import mean, median

OUT = Path("outputs/orphan")
ARROW = "→"

# (gene, [biomarker synonyms], disease) — accumulating substrate, from clinical biochemistry, fixed a priori.
IEM = [
    ("CBS",    ["homocysteine"],                                   "homocystinuria"),
    ("ASS1",   ["citrulline"],                                     "citrullinemia I"),
    ("ARG1",   ["arginine"],                                       "argininemia"),
    ("OTC",    ["ornithine", "carbamoyl-phosphate", "carbamoyl phosphate"], "OTC deficiency"),
    ("HPD",    ["hydroxyphenylpyruvate"],                          "tyrosinemia III"),
    ("IVD",    ["isovaleryl-coa", "3-methylbutanoyl-coa"],         "isovaleric acidemia"),
    ("PCCA",   ["propanoyl-coa", "propionyl-coa"],                 "propionic acidemia"),
    ("PCCB",   ["propanoyl-coa", "propionyl-coa"],                 "propionic acidemia"),
    ("ACADM",  ["octanoyl-coa", "decanoyl-coa", "hexanoyl-coa"],   "MCAD deficiency"),
    ("ALDOB",  ["fructose-1-phosphate", "fructose 1-phosphate"],   "hereditary fructose intolerance"),
    ("BCKDHA", ["4-methyl-2-oxopentanoate", "methyl-2-oxopentanoate", "ketoleucine"], "MSUD"),
    ("GAA",    ["glycogen"],                                       "Pompe disease"),
    ("SMPD1",  ["sphingomyelin"],                                  "Niemann-Pick A/B"),
    ("GLA",    ["galactosylceramide", "globotriaosylceramide"],    "Fabry disease"),
    ("ASL",    ["argininosuccinate"],                              "argininosuccinic aciduria"),
    ("MUT",    ["methylmalonyl-coa", "methylmalonate"],            "methylmalonic acidemia"),
]


def _norm(m):
    m = re.sub(r"\[[a-z]\]", "", m)               # drop compartment [c]/[e]/[m]
    m = re.sub(r"^\d+(\.\d+)?\s+", "", m)          # drop stoichiometry
    return m.strip().lower()


def substrates_of(rxns):
    s = set()
    for r in rxns:
        left = r.split(ARROW)[0] if ARROW in r else r
        for m in left.split(" + "):
            s.add(_norm(m))
    return s


def load(path=None):
    p = Path(path) if path else (OUT / "metab_reactions.json")
    if p.exists():
        d = json.load(open(p)); return d["generxn"], d["names"]
    cc = OUT / "cell_complete.json"
    if cc.exists():
        D = json.load(open(cc)); return D["generxn"], [g["name"] for g in D["genes"]]
    return None, None


def validate():
    generxn, names = load()
    if generxn is None:
        print("no metab_reactions.json / cell_complete.json"); return None
    name2i = {n: i for i, n in enumerate(names)}
    subs = {int(k): substrates_of(v) for k, v in generxn.items()}   # enzyme_idx -> substrate set

    def has(bio_syns, sset):
        return any(any(syn in s for s in sset) for syn in bio_syns)

    # specificity: how many enzymes have a given biomarker among substrates (small = specific)
    def spec(bio_syns):
        return sum(1 for i, s in subs.items() if has(bio_syns, s))

    rows = []
    for gene, bios, dis in IEM:
        i = name2i.get(gene)
        present = i is not None and i in subs
        rec = has(bios, subs[i]) if present else None
        rows.append(dict(gene=gene, disease=dis, biomarker=bios[0], present=present,
                         recovered=rec, n_enzymes_sharing=spec(bios) if present else None))
    present = [r for r in rows if r["present"]]
    recov = [r for r in present if r["recovered"]]
    # random baseline: random (enzyme, biomarker) pairs — how often does the biomarker appear?
    rng = random.Random(0); allidx = list(subs); hits = 0; N = 3000
    for _ in range(N):
        e = rng.choice(allidx); _, bios, _ = IEM[rng.randrange(len(IEM))]
        if has(bios, subs[e]): hits += 1
    rand = hits / N
    summary = dict(
        n_iem=len(rows), n_present=len(present), coverage=round(len(present) / len(rows), 3),
        recall=round(len(recov) / len(present), 3) if present else None,
        n_recovered=len(recov),
        median_enzymes_sharing_biomarker=median([r["n_enzymes_sharing"] for r in recov]) if recov else None,
        random_pair_baseline=round(rand, 4),
        specificity_lift=round((len(recov) / len(present)) / rand, 1) if (present and rand) else None,
    )
    return dict(summary=summary, per_iem=rows)


def main():
    res = validate()
    if not res:
        return
    json.dump(res, open(OUT / "iem_mechanism_validation.json", "w"), indent=2)
    s = res["summary"]
    print("=" * 74)
    print("PLAY B — inborn-error-of-metabolism biomarker recovery (clinical ground truth)")
    print("=" * 74)
    print(f"  IEMs tested: {s['n_iem']} | enzyme present in model: {s['n_present']} "
          f"(coverage {s['coverage']})")
    print(f"  recall (biomarker is a substrate of the disease enzyme): {s['recall']}  "
          f"({s['n_recovered']}/{s['n_present']})")
    print(f"  specificity: biomarker shared by a median of {s['median_enzymes_sharing_biomarker']} "
          f"of 2549 enzymes")
    print(f"  random (enzyme,biomarker)-pair baseline: {s['random_pair_baseline']} "
          f"-> {s['specificity_lift']}x lift")
    print("  per-IEM:")
    for r in res["per_iem"]:
        tag = ("recovered" if r["recovered"] else "MISS") if r["present"] else "not-in-model"
        extra = f" (shared by {r['n_enzymes_sharing']} enz)" if r["present"] and r["recovered"] else ""
        print(f"    {r['gene']:8} {r['disease']:28} {r['biomarker']:26} {tag}{extra}")


if __name__ == "__main__":
    main()
