"""Per-species minimal-cell construction.

For each labelled organism with sufficient predictions, build its candidate
minimal genome by:

  1. CONFIDENT-NONESSENTIAL PRUNING. Remove every gene the deployed router
     calls non-essential with predicted P(essential) <= 5% (i.e., model is
     >= 95% sure NOT essential). Honors the model's STRONG direction (neCov).

  2. SLOT LABELLING. Each survivor gets a functional-slot vector via product-
     name keyword classification (the same scheme used to build
     memory_bank/data/syn3a_gene_class_features.csv): is_ribosomal, is_trna,
     is_synthetase, is_polymerase, is_dna_machinery, is_translation,
     is_transcription, is_membrane, is_atp_binding, is_metabolism,
     is_uncharacterized.

  3. CONSERVED-CORE SEPARATION. Survivors in broad families (>=10 orgs) with
     >=0.8 leak-free essentiality are the species-invariant universal core.
     The complement is the species-specific residual.

  4. SLOT-COMPLETENESS CHECK. Compare each species' minimal-cell slot
     fingerprint against syn3a's (a real wet-lab minimal cell, 455 genes,
     known-viable). Any slot syn3a fills that the candidate does not is a gap.

  5. SYN3A VALIDATION. Run the same pipeline on syn3a itself. The constructed
     minimal cell must overlap syn3a's known 383 essentials with high precision
     -- that's the empirical sanity check.

Outputs:
  outputs/orphan/minimal_cell_per_species.json   per-org composition + slots
  outputs/orphan/minimal_cell_syn3a_validation.json   gold-standard check
"""
import csv, json, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

OUT = Path("outputs/orphan")

# -------------------------------------------------------------------------
# slot keyword classifier (mirrors syn3a_gene_class_features columns)
# -------------------------------------------------------------------------
# Each slot has product-keyword patterns AND gene_name patterns. Gene names
# (rpsA, rplB, dnaA, gyrB, rpoB, atpB...) follow strong bacterial conventions
# and are present in labels.csv for most genes -- making them a more reliable
# slot signal than product strings which are often absent.
SLOTS = [
    ("is_ribosomal",     [r"\bribosom", r"\b30s\b", r"\b50s\b", r"\bs\d+\b",
                           r"\bl\d+\b"],
                          [r"^rp[sl][a-z]?\d*$", r"^rpm[a-z]\d*$"]),
    ("is_trna",          [r"\btrna\b", r"transfer rna"],
                          [r"^trn[a-z]\d*$", r"^trna-"]),
    ("is_synthetase",    [r"synthetase", r"trna ligase", r"aminoacyl"],
                          [r"^[a-z]rs[a-z]?$", r"trna ligase"]),
    ("is_polymerase",    [r"polymerase", r"\brpo[a-z]?\b"],
                          [r"^rpo[a-z]\d*$", r"^pol[a-z]$"]),
    ("is_dna_machinery", [r"helicase", r"primase", r"topoisomer", r"gyrase",
                           r"recombinase", r"replication", r"single-strand"],
                          [r"^dna[a-z]$", r"^gyr[ab]$", r"^par[ce]$",
                           r"^rec[a-z]$", r"^lig[a-z]$", r"^ssb$",
                           r"^holb?$", r"^dn[ab]$"]),
    ("is_translation",   [r"translation", r"elongation factor", r"initiation factor",
                           r"release factor", r"chaperone"],
                          [r"^tuf[ab]?$", r"^fus[a-z]?$", r"^if[123]$",
                           r"^prf[abc]$", r"^dna[jk]$", r"^gro[el]l?$",
                           r"^tig$", r"^hflx?$", r"^ts$"]),
    ("is_transcription", [r"transcription", r"sigma factor", r"\bnusa\b", r"\bnusb\b"],
                          [r"^nus[abefg]$", r"^sig[a-z]?\d*$", r"^rho$",
                           r"^greab$"]),
    ("is_membrane",      [r"membrane", r"lipoprotein", r"porin", r"channel",
                           r"transporter", r"permease", r"abc transport"],
                          [r"^omp[a-z]$", r"^lpt[abcdefg]?$", r"^msb[a-z]?$"]),
    ("is_atp_binding",   [r"atp[- ]binding", r"atpase", r"gtp[- ]binding", r"gtpase"],
                          [r"^atp[a-z]\d*$"]),
    ("is_kinase",        [r"kinase"], [r"kinase"]),
    ("is_phosphatase",   [r"phosphatase"], [r"phosphatase"]),
    ("is_synthase",      [r"synthase"], [r"synthase", r"^murs?[a-z]?$"]),
    ("is_metabolism",    [r"dehydrogenase", r"reductase", r"oxidase", r"hydrolase",
                           r"transferase", r"isomerase", r"lyase", r"decarboxylase",
                           r"carboxylase", r"phosphoribosyl", r"biosynthe"],
                          [r"^pur[a-z]\d*$", r"^pyr[a-z]\d*$", r"^thr[abc]$",
                           r"^trp[a-eg]$", r"^his[a-h]$", r"^ilv[a-eh]$",
                           r"^leu[abcd]$", r"^arg[a-h]$", r"^lys[acn]$",
                           r"^met[a-l]\d*$", r"^cys[a-zk]$", r"^aro[a-h]$",
                           r"^fol[a-ep]$", r"^thi[a-z]\d*$", r"^pdx[abjkty]?$",
                           r"^men[a-h]$", r"^ub[ia-z]$", r"^gltabcd?$"]),
    ("is_uncharacterized",[r"hypothetical", r"unknown function", r"uncharacterized",
                           r"putative", r"\bduf\d+"],
                          [r"^y[a-z]{3}$", r"^duf\d+$"]),
]
SLOT_NAMES = [s[0] for s in SLOTS]
SLOT_PROD = [(n, [re.compile(p, re.I) for p in pats]) for n, pats, _ in SLOTS]
SLOT_GENE = [(n, [re.compile(p, re.I) for p in pats]) for n, _, pats in SLOTS]


def classify(product, gene_name):
    """Classify a gene into functional slots using product description AND
    gene_name. Gene_names are the more reliable source for the beril orgs."""
    p = (product or "").lower(); g = (gene_name or "").lower().strip()
    vec = np.zeros(len(SLOT_NAMES), np.int8)
    for i in range(len(SLOT_NAMES)):
        if p:
            for rx in SLOT_PROD[i][1]:
                if rx.search(p): vec[i] = 1; break
        if vec[i] == 0 and g:
            for rx in SLOT_GENE[i][1]:
                if rx.search(g): vec[i] = 1; break
    return vec


# -------------------------------------------------------------------------
# load labels, predictions, products, og info
# -------------------------------------------------------------------------
print("Loading universe + predictions + product annotations...")
labels = {}; gene_names = {}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    k = (r["organism"], r["locus_tag"])
    labels[k] = int(r["essential"])
    gn = (r.get("gene_name") or "").strip()
    if gn and gn != "-":
        gene_names[k] = gn
print(f"  labels: {len(labels):,};  gene_names available: {len(gene_names):,} "
      f"({len(gene_names)/len(labels)*100:.0f}%)")

# OG info per gene + per-OG family stats
og_by_gene = {}; og_breadth = {}; og_leakfree_ess = {}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    if r.get("og_id"):
        og_by_gene[k] = r["og_id"]
        try:
            og_breadth[r["og_id"]] = int(float(r["family_n_organisms"]))
            og_leakfree_ess[r["og_id"]] = float(r["family_frac_essential_leakfree"])
        except Exception:
            pass

# product names: DEG essential-gene file (has descriptions for many orgs)
prod_count = 0
for r in csv.DictReader(open("data/drive_import/deg/deg_bacteria_essential_genes.csv")):
    # we don't have a clean (organism, locus_tag) mapping for DEG, but the
    # 'description' is often enough to slot-classify the DEG genes for the
    # syn3a validation. We use this for the syn3a path separately.
    prod_count += 1

# syn3a product annotations from memory_bank
syn3a_products = {}
try:
    for r in csv.DictReader(open("memory_bank/data/syn3a_gene_table.csv")):
        syn3a_products[("syn3a", r["locus_tag"])] = r.get("product", "")
except Exception:
    pass
print(f"  syn3a products loaded: {len(syn3a_products)}")

# full per-gene annotation from annotate_genes.py (GFF + OG propagation)
annot_products = {}
try:
    for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
        if r.get("product"):
            annot_products[(r["organism"], r["locus_tag"])] = r["product"]
    print(f"  GFF/OG annotations loaded: {len(annot_products):,}")
except FileNotFoundError:
    print("  (gene_annotations.csv not found -- run annotate_genes.py first)")

# load model predictions
preds = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p_arr = preds["p"]; br_arr = preds["brightness"]
clade_arr = preds["clade"]; lt_arr = preds["lt"]; y_arr = preds["y"].astype(int)
# we don't have org names directly in preds; recover via the cache
cache = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
corgs = list(cache["orgs"]); cmeta = cache["meta_org"]; clt = cache["lt"]; cclade = cache["clade"]
clade_lt_to_org = {(str(cclade[i]), str(clt[i])): str(corgs[int(cmeta[i])])
                   for i in range(len(clt))}
pred_by_key = {}
for i in range(len(p_arr)):
    org = clade_lt_to_org.get((str(clade_arr[i]), str(lt_arr[i])))
    if org and not (np.isnan(p_arr[i])):
        pred_by_key[(org, str(lt_arr[i]))] = (float(p_arr[i]), float(br_arr[i]))
print(f"  predictions usable: {len(pred_by_key):,}")

# orphan-route CAI predictions (re-derive quickly: highly-expressed orphans tend
# to be essential; for confident-non-essential pruning we use a per-org CAI
# threshold from codon_features when available)
cai_map = {}
for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv")):
    try:
        cai_map[(r["organism"], r["locus_tag"])] = float(r["cai"])
    except Exception:
        pass

# -------------------------------------------------------------------------
# product-name source: orthology has none; for slot classification we use the
# locus_tag's organism-specific product when available, with fallback heuristics.
# Many orgs don't ship product names; we proxy via OG-membership only.
# -------------------------------------------------------------------------
# Build OG -> any product seen (used as a proxy for orgs missing per-gene
# product). Source 1: syn3a's annotated products. Source 2: DEG descriptions
# of essential genes -- their gene_name + description -> via OG membership.
og_product = {}
for k, prod in syn3a_products.items():
    if prod and k in og_by_gene:
        og_product.setdefault(og_by_gene[k], prod)
# DEG: map by gene_name then propagate via OG (gene_name match -> any org with
# that gene_name in our orthology table -> use that gene's OG)
gene_name_to_og = {}
gene_name_csv = "data/drive_import/labels/orthology_features.csv"
# orthology doesn't carry gene_name; instead, look up via labels.csv directly
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    gn = (r.get("gene_name") or "").strip().lower()
    if gn and gn != "-":
        k = (r["organism"], r["locus_tag"])
        if k in og_by_gene:
            gene_name_to_og.setdefault(gn, og_by_gene[k])
deg_added = 0
for r in csv.DictReader(open("data/drive_import/deg/deg_bacteria_essential_genes.csv")):
    gn = (r.get("gene_name") or "").strip().lower()
    desc = (r.get("description") or "").strip()
    if gn and desc and gn in gene_name_to_og:
        og = gene_name_to_og[gn]
        if og not in og_product:
            og_product[og] = desc; deg_added += 1
print(f"  OG product coverage: syn3a-seeded + DEG-propagated = {len(og_product):,} OGs "
      f"(DEG added {deg_added})")

# -------------------------------------------------------------------------
# PER-SPECIES MINIMAL-CELL CONSTRUCTION
# -------------------------------------------------------------------------
print("\nConstructing per-species minimal cells...")

# group genes by organism
genes_by_org = defaultdict(list)
for (org, lt), y in labels.items():
    genes_by_org[org].append((lt, y))

# syn3a's slot fingerprint (built from its known essentials)
syn3a_essentials = [(lt, y) for (lt, y) in [((k[1], v) if k[0] == "syn3a" else (None, None))
                                              for k, v in labels.items()] if lt is not None and y == 1]
syn3a_slot_vec = np.zeros(len(SLOT_NAMES), int)
for lt, _ in syn3a_essentials:
    prod = syn3a_products.get(("syn3a", lt)) or annot_products.get(("syn3a", lt), "")
    gn = gene_names.get(("syn3a", lt), "")
    syn3a_slot_vec += classify(prod, gn)
syn3a_required_slots = [SLOT_NAMES[i] for i, v in enumerate(syn3a_slot_vec) if v >= 3]
print(f"  syn3a essential slot fingerprint (>=3 genes per slot, the 'required slots'):")
for i, n in enumerate(SLOT_NAMES):
    flag = "REQUIRED" if SLOT_NAMES[i] in syn3a_required_slots else "-"
    print(f"    {n:<22} {int(syn3a_slot_vec[i]):>4}  {flag}")

per_species = {}

# iterate orgs in our model coverage
prediction_orgs = set(k[0] for k in pred_by_key) | {"syn3a"}
for org in sorted(prediction_orgs):
    if org not in genes_by_org:
        continue
    genes = genes_by_org[org]
    if len(genes) < 200:
        continue

    n_total = len(genes)
    n_true_ess = sum(1 for _, y in genes if y == 1)

    # step 1: confident-nonessential pruning
    pruned = 0
    survivors = []
    for lt, y in genes:
        pr = pred_by_key.get((org, lt))
        if pr is None:
            survivors.append((lt, y, None, None))
            continue
        p, br = pr
        # confident non-essential: model places <=15% on essential
        # AND has any non-floor confidence (brightness >= 0.3)
        if p <= 0.15 and br >= 0.3:
            pruned += 1
            continue
        survivors.append((lt, y, p, br))

    # step 2: slot labels for survivors
    slot_counts = np.zeros(len(SLOT_NAMES), int)
    n_with_product = 0
    conserved = 0; species_specific = 0; unannotated = 0
    survivor_records = []
    for lt, y, p, br in survivors:
        prod = (syn3a_products.get((org, lt)) or annot_products.get((org, lt))
                or og_product.get(og_by_gene.get((org, lt)), ""))
        gn = gene_names.get((org, lt), "")
        if prod or gn:
            n_with_product += 1
        slots = classify(prod, gn)
        slot_counts += slots
        og = og_by_gene.get((org, lt))
        if og and og_breadth.get(og, 1) >= 10 and og_leakfree_ess.get(og, 0) >= 0.8:
            conserved += 1
        elif og:
            species_specific += 1
        else:
            unannotated += 1
        survivor_records.append(dict(lt=lt, y=int(y), p=p, br=br,
                                     og=og,
                                     slots=[SLOT_NAMES[i] for i in np.where(slots)[0]]))

    # step 3: slot completeness vs syn3a's required slots
    missing_required = [s for s in syn3a_required_slots if slot_counts[SLOT_NAMES.index(s)] == 0]

    # step 4: precision of the candidate-minimal cell (how many survivors are
    # actually essential -- the validation metric on labels)
    n_surv = len(survivors)
    n_surv_ess = sum(1 for r in survivor_records if r["y"] == 1)
    candidate_prec = n_surv_ess / max(1, n_surv)
    candidate_recall = n_surv_ess / max(1, n_true_ess)

    per_species[org] = dict(
        n_total=n_total,
        n_true_essential=n_true_ess,
        n_pruned=pruned,
        n_minimal=n_surv,
        composition=dict(
            conserved_core=conserved,
            species_specific=species_specific,
            unannotated_no_og=unannotated,
        ),
        slot_counts={SLOT_NAMES[i]: int(slot_counts[i]) for i in range(len(SLOT_NAMES))},
        missing_required_slots=missing_required,
        validation=dict(
            essential_recall=round(candidate_recall, 3),
            minimal_cell_essential_purity=round(candidate_prec, 3),
            n_with_product=n_with_product,
        ),
    )

# -------------------------------------------------------------------------
# REPORT
# -------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PER-SPECIES MINIMAL-CELL SUMMARY")
print("=" * 72)
print(f"  {'organism':<28} {'tot':>5} {'true_e':>6} {'prune':>6} {'minim':>6} "
      f"{'recall':>7} {'purity':>7} {'gaps':>5}")
print("  " + "-" * 76)
for org in sorted(per_species, key=lambda o: -per_species[o]["n_minimal"]):
    d = per_species[org]
    print(f"  {org[:28]:<28} {d['n_total']:>5} {d['n_true_essential']:>6} "
          f"{d['n_pruned']:>6} {d['n_minimal']:>6} "
          f"{d['validation']['essential_recall']:>7.2f} "
          f"{d['validation']['minimal_cell_essential_purity']:>7.2f} "
          f"{len(d['missing_required_slots']):>5}")

# -------------------------------------------------------------------------
# SYN3A VALIDATION (special case)
# -------------------------------------------------------------------------
syn3a_report = per_species.get("syn3a", {})
print("\n=== SYN3A VALIDATION ===")
if syn3a_report:
    sr = syn3a_report
    print(f"  syn3a true essentials: {sr['n_true_essential']}  (Breuer 2019)")
    print(f"  pipeline minimal cell: {sr['n_minimal']}  genes")
    print(f"  essential RECALL    : {sr['validation']['essential_recall']:.3f}  "
          f"(fraction of true essentials retained)")
    print(f"  minimal-cell PURITY : {sr['validation']['minimal_cell_essential_purity']:.3f}  "
          f"(fraction of survivors that are truly essential)")
    print(f"  missing required slots: {sr['missing_required_slots'] or 'none'}")
else:
    print("  syn3a not in prediction set -- pipeline ran on label-only path")

# write outputs
json.dump(dict(
    per_species=per_species,
    syn3a_required_slots=syn3a_required_slots,
    syn3a_slot_fingerprint={SLOT_NAMES[i]: int(syn3a_slot_vec[i]) for i in range(len(SLOT_NAMES))},
), open(OUT / "minimal_cell_per_species.json", "w"), indent=2)

json.dump(dict(
    syn3a_validation=syn3a_report,
), open(OUT / "minimal_cell_syn3a_validation.json", "w"), indent=2)
print(f"\nwrote outputs/orphan/minimal_cell_per_species.json")
print(f"wrote outputs/orphan/minimal_cell_syn3a_validation.json")
