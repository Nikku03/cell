"""Bridge Keio numeric locus_tags <-> MG1655 b-numbers.

Strategy:
  1 ANCHOR  : exact product matches that are unique on both sides (~331 genes)
  2 WALK    : from each anchor, follow Keio synteny prev/next AND walk MG1655
              genome order; if Keio's next is unbridged, assign to anchor_b's
              next b-number. Verify by checking the OPPOSITE direction matches
              (b's prev is anchor when walking right, etc.).
  3 ITERATE : repeat the walk; chains grow from every anchor.
  4 VERIFY  : optionally cross-check with annotation product when available.

The output is data/drive_import/labels/keio_to_bnumber.csv (organism,
locus_tag, b_number, gene_name, match_source). Once written, every downstream
analysis can join Keio to MG1655 b-numbers -> ECOCYC IDs -> FBA model -> etc.

Doesn't need network; pure local. Outputs coverage stats.
"""
import csv, re, json
from collections import defaultdict
from pathlib import Path

OUT_BRIDGE = Path("data/drive_import/labels/keio_to_bnumber.csv")

# ---- MG1655 genome order: parse GFF in file order (already sorted by position) ----
b_order = []                    # list of b-numbers in genome order
b_info = {}                     # b -> dict(gene, product, length, start)
for line in open("data/drive_import/genome_cache/GCF_000005845.2/genomic.gff"):
    if "\tCDS\t" not in line:
        continue
    lt = re.search(r"locus_tag=(b\d+)", line)
    if not lt: continue
    b = lt.group(1)
    if b in b_info: continue          # one row per gene (skip duplicate CDS entries)
    gn = re.search(r"gene=([^;]+)", line)
    pr = re.search(r"product=([^;\n]+)", line)
    fields = line.split("\t")
    try: start = int(fields[3]); end = int(fields[4])
    except: continue
    b_info[b] = dict(gene=gn.group(1) if gn else "",
                     product=pr.group(1) if pr else "",
                     length=end - start + 1, start=start)
    b_order.append(b)
# index by genome position
b_pos = {b: i for i, b in enumerate(b_order)}
print(f"MG1655: {len(b_info)} b-numbered CDS in genome order")

# ---- Keio synteny chain: prev/next locus_tag pairs ----
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    if r["organism"] == "beril_Keio":
        synt[r["locus_tag"]] = (r.get("prev_locus_tag", ""),
                                r.get("next_locus_tag", ""))
# Keio annotation (OG-propagated products)
keio_ann = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r["organism"] == "beril_Keio" and r.get("product"):
        keio_ann[r["locus_tag"]] = r["product"]
keio_lts = set(synt.keys()) | set(keio_ann.keys())
print(f"Keio: {len(keio_lts)} locus_tags, {len(synt)} with synteny, "
      f"{len(keio_ann)} annotated")

# ---- 1. ANCHORS: unique product matches ----
prod_to_keio = defaultdict(list); prod_to_b = defaultdict(list)
for lt, p in keio_ann.items(): prod_to_keio[p].append(lt)
for b, info in b_info.items():
    if info["product"]: prod_to_b[info["product"]].append(b)
anchors = {}                          # keio_lt -> b
for p in prod_to_keio:
    if p in prod_to_b and len(prod_to_keio[p]) == 1 and len(prod_to_b[p]) == 1:
        anchors[prod_to_keio[p][0]] = prod_to_b[p][0]
print(f"\nSTAGE 1  anchors (unique-product matches): {len(anchors)}")

# ---- 2-3. WALK + ITERATE: propagate via synteny + genome order ----
bridge = dict(anchors); b_used = set(anchors.values())
sources = {lt: "anchor_product" for lt in bridge}

def walk_round(bridge, b_used, sources, strict=True):
    added = 0
    for lt, b in list(bridge.items()):
        bp = b_pos.get(b)
        if bp is None: continue
        prev, nxt = synt.get(lt, ("", ""))
        for keio_neighbor, bn_offset in [(nxt, +1), (prev, -1)]:
            if not keio_neighbor or keio_neighbor not in keio_lts: continue
            if keio_neighbor in bridge: continue
            bn_idx = bp + bn_offset
            if bn_idx < 0 or bn_idx >= len(b_order): continue
            bn = b_order[bn_idx]
            if bn in b_used: continue
            kp = keio_ann.get(keio_neighbor); bp_prod = b_info[bn]["product"]
            ok = True
            if strict and kp and bp_prod and kp != bp_prod:
                kp_l = kp.lower(); bp_l = bp_prod.lower()
                common = set(re.findall(r"\w+", kp_l)) & set(re.findall(r"\w+", bp_l))
                common -= {"protein", "domain", "family", "containing", "the", "of",
                           "subunit", "putative", "uncharacterized"}
                ok = len(common) >= 1
            if ok:
                bridge[keio_neighbor] = bn; b_used.add(bn)
                sources[keio_neighbor] = "synteny_walk" if strict else "synteny_chain"
                added += 1
    return added

# round A: strict walks (requires shared token if both annotated)
print("STAGE 2A  strict synteny walk (requires product token agreement)")
for it in range(15):
    n = walk_round(bridge, b_used, sources, strict=True)
    print(f"  iter {it+1}: +{n}  total {len(bridge)}")
    if n == 0: break

# round B: relaxed walks (synteny only) -- propagate through unannotated stretches
print("STAGE 2B  relaxed synteny walk (pure adjacency through unannotated gaps)")
for it in range(15):
    n = walk_round(bridge, b_used, sources, strict=False)
    print(f"  iter {it+1}: +{n}  total {len(bridge)}")
    if n == 0: break

# ---- 4. final fill: any unmatched Keio gene whose product matches an unused b
#      (non-unique products that became unique after anchoring/walking)
final_added = 0
remaining_b = set(b_info) - b_used
prod_to_remb = defaultdict(list)
for b in remaining_b:
    p = b_info[b]["product"]
    if p: prod_to_remb[p].append(b)
remaining_k = [lt for lt in keio_ann if lt not in bridge]
prod_to_remk = defaultdict(list)
for lt in remaining_k:
    p = keio_ann[lt]
    if p: prod_to_remk[p].append(lt)
for p in prod_to_remk:
    if p in prod_to_remb and len(prod_to_remk[p]) == 1 and len(prod_to_remb[p]) == 1:
        lt = prod_to_remk[p][0]; b = prod_to_remb[p][0]
        bridge[lt] = b; sources[lt] = "anchor_product_round2"; final_added += 1
print(f"\nSTAGE 4  round-2 unique-product anchors: +{final_added}")

# ---- write the bridge ----
with open(OUT_BRIDGE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["organism", "locus_tag", "b_number", "gene_name", "product", "source"])
    for lt in sorted(bridge):
        b = bridge[lt]
        info = b_info[b]
        w.writerow(["beril_Keio", lt, b, info["gene"], info["product"], sources[lt]])
print(f"\nwrote {OUT_BRIDGE}  ({len(bridge)} / {len(keio_lts)} Keio genes bridged "
      f"= {100*len(bridge)/len(keio_lts):.0f}%)")
print(f"  by source: " + ", ".join(f"{s}:{sum(1 for v in sources.values() if v==s)}"
                                    for s in sorted(set(sources.values()))))

# ---- VERIFY against FBA features (per-source) ----
fba = [r for r in csv.DictReader(open("data/drive_import/labels/fba_features.csv"))
       if r["organism"] == "beril_Keio"]
fba_b = {r["locus_tag"]: int(r["fba_essential"]) for r in fba}
keio_truth = {r["locus_tag"]: int(r["essential"])
              for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))
              if r["organism"] == "beril_Keio"}
print(f"\n=== VERIFY: bridge -> FBA -> Keio truth, by quality tier ===")
for source_set, label in [
    ({"anchor_product", "anchor_product_round2", "synteny_walk"}, "strict (anchor + product-validated walk)"),
    ({"synteny_chain"},                                            "relaxed (pure adjacency walk)"),
    (set(sources.values()),                                        "ALL bridges"),
]:
    join = [(lt, bridge[lt], fba_b[bridge[lt]], keio_truth.get(lt))
            for lt in bridge if sources[lt] in source_set
            and bridge[lt] in fba_b and lt in keio_truth]
    if not join:
        print(f"  {label:<55} (no overlap)"); continue
    tp = sum(1 for _,_,f,t in join if f==1 and t==1)
    fp = sum(1 for _,_,f,t in join if f==1 and t==0)
    fn = sum(1 for _,_,f,t in join if f==0 and t==1)
    tn = sum(1 for _,_,f,t in join if f==0 and t==0)
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    print(f"  {label:<55} n={len(join):>4}  P={p:.2f} R={r:.2f}  "
          f"(TP {tp} FP {fp} FN {fn} TN {tn})")
