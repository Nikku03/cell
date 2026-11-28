"""export_interface_pairs — build the HONEST discovery pair set for the Colab AF-Multimer notebook.

The fetched Predictomes test showed AF-Multimer's raw contact count = chance (0.50) on genuine discovery, but at
that screen's high-throughput AF settings. The notebook re-tests the ONE untested variant: a careful full-MSA
AF-Multimer run reading ipTM + pDockQ. For that we need a small, balanced, leak-clean pair list:

  positives = real interactions AF could NOT have memorised: our PPI edges with in_pdb=0 (no solved co-structure),
              endpoints share ZERO neighbours (the discovery regime).
  negatives = zero-evidence non-edges (not in graph; no STRING-phys / BioGRID / IntAct / PDB), degree-matched to the
              positives' partners, also 0-shared.

Exports UniProt accession pairs + labels (the notebook fetches sequences by accession). Capped small (default
80/class = 160 folds) because AF-Multimer is ~1-2 min/fold — enough AUC resolution to tell 0.61 from 0.50 or 0.75.
-> outputs/orphan/interface_pairs.json
"""
import os, sys, gzip, json, random
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
SPOC = "/tmp/spoc_pairs.csv.gz"
ACC2SYM = "/tmp/acc2sym.json"


def build(per_class=80, seed=0, spoc_path=SPOC):
    random.seed(seed)
    C = CompleteCell(); idx = C.idx; adj = C.ppi_adj
    acc2sym = json.load(open(ACC2SYM))
    acc2idx = {a: idx[s] for a, s in acc2sym.items() if s in idx}
    idx2acc = {}
    for a, i in acc2idx.items():
        idx2acc.setdefault(i, a)            # one representative accession per gene index
    edgeset = {(min(u, v), max(u, v)) for u, ps in adj.items() for v in ps}

    rows = []
    with gzip.open(spoc_path, "rt") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split(",")
            if ":" not in p[1]:
                continue
            a, b = p[1].split(":")
            i, j = acc2idx.get(a), acc2idx.get(b)
            if i is None or j is None or i == j:
                continue
            try:
                sphys = float(p[6]); big = int(p[7]); inta = int(p[8]); pdb = int(p[9])
            except (ValueError, IndexError):
                continue
            rows.append((i, j, a, b, sphys, big, inta, pdb))

    sh = lambda i, j: len(adj.get(i, set()) & adj.get(j, set()))
    ine = lambda i, j: (min(i, j), max(i, j)) in edgeset
    deg = {g: len(adj[g]) for g in adj}

    pos = [(i, j, a, b) for i, j, a, b, sp, bg, it, pdb in rows
           if ine(i, j) and pdb == 0 and sh(i, j) == 0]
    neg_pool = [(i, j, a, b) for i, j, a, b, sp, bg, it, pdb in rows
                if not ine(i, j) and sp < 50 and bg == 0 and it == 0 and pdb == 0 and sh(i, j) == 0]
    random.shuffle(pos); random.shuffle(neg_pool)
    pos = pos[:per_class]

    # degree-match negatives to the positives' partner degree (control the hub confound)
    neg_by_bin = {}
    for i, j, a, b in neg_pool:
        neg_by_bin.setdefault(min(deg.get(j, 0), 60) // 6, []).append((i, j, a, b))
    negs, used = [], set()
    for i, j, a, b in pos:
        bin_ = min(deg.get(j, 0), 60) // 6
        cands = neg_by_bin.get(bin_) or neg_pool
        for c in cands:
            if c[2] + c[3] not in used:
                negs.append(c); used.add(c[2] + c[3]); break
    negs = negs[:per_class]

    def rec(i, j, a, b, lab):
        return {"gene_a": C.name[i], "gene_b": C.name[j], "acc_a": a, "acc_b": b, "label": lab}
    out = {"note": "honest discovery set: in_pdb=0 real PPIs (0-shared) vs degree-matched zero-evidence non-edges",
           "baseline_domain_auc": 0.61, "baseline_topology_auc": 0.50,
           "predictomes_contacts_auc_genuine": 0.50,
           "pairs": [rec(*p, 1) for p in pos] + [rec(*n, 0) for n in negs]}
    json.dump(out, open(f"{OUT}/interface_pairs.json", "w"), indent=1)
    print(f"exported {len(pos)} positives + {len(negs)} negatives -> {OUT}/interface_pairs.json")
    return out


if __name__ == "__main__":
    build(per_class=int(sys.argv[1]) if len(sys.argv) > 1 else 80)
