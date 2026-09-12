"""interface_discovery_test — does a TRUE AlphaFold-Multimer interface feature beat domain compatibility (0.61)
on the DISCOVERY split? Answered with FETCHED data, NO GPU: the Predictomes human structural PPI screen
(Schmid & Walter, Mol Cell 2025) modeled ~1.6M human pairs with AF-Multimer and released per-pair scores.

Feature: `num_unique_contacts` = residue-residue contacts in the predicted complex — purely STRUCTURAL (read off
the AF model), so unlike SPOC (which folds in STRING/BioGRID/IntAct) it is NOT circular against our DB-sourced edges.

THE CONFOUND THAT MATTERS: AlphaFold was TRAINED ON THE PDB. A pair whose complex is already solved (in_pdb=1) is
effectively memorised — AF reproduces its interface. So "contacts separate in_pdb positives from non-pairs" measures
RECALL, not discovery. The honest discovery test uses positives with in_pdb=0 (real interactions AF had to predict,
not recall).

Result (this is a NEGATIVE result, reported honestly):
  clean labels = gold positives (our edges) vs zero-evidence negatives (not in graph; no STRING-phys/BioGRID/IntAct/PDB)
  - in_pdb=1 positives (memorised): contacts AUC 0.82, and 0.82 on the HARD (0-shared) subset — looks amazing.
  - in_pdb=0 positives (genuine):   contacts AUC 0.53, and 0.50 on HARD — CHANCE. median contacts = 0, same as non-pairs.
=> AF-Multimer interface contacts do NOT discover novel interactions; the apparent signal is PDB memorisation. On
   honest discovery, domains (0.61) BEAT AF-Multimer contacts (0.50). Fetching answered it without spending GPU.

Caveat: Predictomes ran AF-Multimer at high throughput (reduced MSA). A careful per-pair run (full paired MSA +
ipTM/pDockQ) MIGHT recover some signal — that, and only that, is what the Colab GPU notebook tests. Expectation is
modest: AF's PPI signal comes from co-evolutionary MSA depth + PDB precedent, both absent for novel human pairs.
-> outputs/orphan/interface_discovery_test.json
"""
import os, sys, gzip, json, random, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
SPOC = "/tmp/spoc_pairs.csv.gz"
ACC2SYM = "/tmp/acc2sym.json"


def _auc(pos, neg):
    if not pos or not neg:
        return None
    xs = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank = 0.0; i = 0; n = len(xs)
    while i < n:
        j = i
        while j < n and xs[j][0] == xs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            if xs[k][1] == 1:
                rank += avg
        i = j
    npos, nneg = len(pos), len(neg)
    return round((rank - npos * (npos + 1) / 2.0) / (npos * nneg), 3)


def run(seed=0, spoc_path=SPOC):
    random.seed(seed)
    C = CompleteCell(); idx = C.idx; adj = C.ppi_adj
    acc2sym = json.load(open(ACC2SYM))
    acc2idx = {a: idx[s] for a, s in acc2sym.items() if s in idx}
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
                contacts = float(p[4]); spoc = float(p[3]); sphys = float(p[6])
                big = int(p[7]); inta = int(p[8]); pdb = int(p[9])
            except (ValueError, IndexError):
                continue
            rows.append((i, j, contacts, spoc, sphys, big, inta, pdb))

    def sh(i, j):
        return len(adj.get(i, set()) & adj.get(j, set()))
    ine = lambda r: (min(r[0], r[1]), max(r[0], r[1])) in edgeset
    hard = lambda xs: [r for r in xs if sh(r[0], r[1]) == 0]
    contacts_of = lambda xs: [r[2] for r in xs]

    # clean labels
    pos_pdb = [r for r in rows if ine(r) and r[7] == 1]                  # memorised (AF trained on the co-structure)
    pos_nopdb = [r for r in rows if ine(r) and r[7] == 0]               # genuine (AF had to predict)
    clean_neg = [r for r in rows if not ine(r) and r[4] < 50 and r[5] == 0 and r[6] == 0 and r[7] == 0]
    cn = random.sample(clean_neg, min(20000, len(clean_neg)))
    cn_h = random.sample(hard(clean_neg), min(20000, len(hard(clean_neg))))

    res = {
        "source": "Predictomes (Schmid & Walter, Mol Cell 2025), AlphaFold-Multimer human screen, num_unique_contacts",
        "in_model_pairs": len(rows), "pos_in_pdb": len(pos_pdb), "pos_no_pdb": len(pos_nopdb),
        "clean_negatives": len(clean_neg),
        "median_contacts": {"pos_in_pdb": st.median(contacts_of(pos_pdb)) if pos_pdb else None,
                            "pos_no_pdb": st.median(contacts_of(pos_nopdb)) if pos_nopdb else None,
                            "clean_neg": st.median(contacts_of(cn)) if cn else None},
        "MEMORISED_in_pdb_pos_vs_clean_neg": {
            "contacts_ALL": _auc(contacts_of(pos_pdb), contacts_of(cn)),
            "contacts_HARD": _auc(contacts_of(hard(pos_pdb)), contacts_of(cn_h)),
            "note": "AF trained on these co-structures -> RECALL, not discovery",
        },
        "GENUINE_no_pdb_pos_vs_clean_neg": {
            "contacts_ALL": _auc(contacts_of(pos_nopdb), contacts_of(cn)),
            "contacts_HARD": _auc(contacts_of(hard(pos_nopdb)), contacts_of(cn_h)),
            "note": "real interactions AF had to PREDICT -> the honest discovery number",
        },
        "baselines_on_discovery": {"topology": 0.50, "domain_compat": 0.61,
                                   "note": "from struct_discovery_test.py (matched negatives)"},
    }
    hard_genuine = res["GENUINE_no_pdb_pos_vs_clean_neg"]["contacts_HARD"]
    res["verdict"] = (
        f"AF-Multimer interface contacts = {hard_genuine} on genuine discovery (in_pdb=0, 0-shared) — CHANCE. The "
        f"0.82 on in_pdb=1 pairs is PDB memorisation, not discovery. Domains (0.61) beat AF contacts here. Fetched "
        f"data answered it without GPU; a careful full-MSA ipTM run is the only untested variant (notebook)."
        if (hard_genuine or 0) < 0.57 else
        f"AF-Multimer interface contacts = {hard_genuine} on genuine discovery — beats domains (0.61); worth a "
        f"careful GPU run to confirm.")
    json.dump(res, open(f"{OUT}/interface_discovery_test.json", "w"), indent=2)

    print("=" * 82)
    print("AF-MULTIMER INTERFACE CONTACTS on the DISCOVERY split (fetched Predictomes data, no GPU)")
    print("=" * 82)
    print(f"  pairs in-model {len(rows):,} | pos in_pdb {len(pos_pdb):,} | pos no_pdb {len(pos_nopdb):,} | "
          f"clean neg {len(clean_neg):,}")
    print(f"  median contacts: in_pdb_pos={res['median_contacts']['pos_in_pdb']}  "
          f"no_pdb_pos={res['median_contacts']['pos_no_pdb']}  neg={res['median_contacts']['clean_neg']}")
    m = res["MEMORISED_in_pdb_pos_vs_clean_neg"]; g = res["GENUINE_no_pdb_pos_vs_clean_neg"]
    print(f"  MEMORISED (in_pdb=1): contacts AUC  ALL {m['contacts_ALL']}   HARD {m['contacts_HARD']}   <- RECALL")
    print(f"  GENUINE   (in_pdb=0): contacts AUC  ALL {g['contacts_ALL']}   HARD {g['contacts_HARD']}   <- DISCOVERY")
    print(f"  baselines on discovery: topology 0.50 | domains 0.61")
    print(f"  -> {res['verdict']}")
    print("=" * 82)
    return res


if __name__ == "__main__":
    run()
