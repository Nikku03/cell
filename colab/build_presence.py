"""Build the OG presence matrix used by the cooccur partner search, with an
optional expansion to more genomes.

The in-repo orthology table (data/drive_import/labels/orthology_features.csv)
covers the curated ~48-91 labelled organisms. More genomes sharpen the
anti-correlation signal-to-noise of the cooccur channels -- BUT only if those
genomes have OG assignments in the SAME OG space. Raw FASTA alone is not
enough; you must map each extra genome's proteins onto the existing OGs first.

Usage:
  # default: just materialise the in-repo presence as an npz (uniform interface)
  python colab/build_presence.py

  # expanded: provide a CSV of (organism,og_id) rows for extra genomes whose
  # proteins you have already assigned to the existing OG space (e.g. via
  # mmseqs/diamond against data/gtdb/og_reps.faa). These rows are UNIONED into
  # the presence matrix (presence-only; they carry no essentiality labels).
  python colab/build_presence.py --extra_presence_csv /content/drive/.../extra_og.csv

Output: outputs/orphan/presence_matrix.npz  (P, all_orgs, all_ogs)
"""
import csv, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

from af_common import DATA, OUT


def build(extra_csv=None):
    og_present = defaultdict(set)
    for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
        if r["og_id"]:
            og_present[r["og_id"]].add(r["organism"])
    n_base_orgs = len({o for s in og_present.values() for o in s})
    n_extra = 0
    if extra_csv:
        for r in csv.DictReader(open(extra_csv)):
            og = r.get("og_id"); org = r.get("organism")
            if og and org and og in og_present:        # only OGs already in our space
                og_present[og].add(org); n_extra += 1
    all_orgs = sorted({o for s in og_present.values() for o in s})
    all_ogs = sorted(og_present)
    org_idx = {o: i for i, o in enumerate(all_orgs)}
    P = np.zeros((len(all_orgs), len(all_ogs)), np.float32)
    for j, og in enumerate(all_ogs):
        for o in og_present[og]:
            P[org_idx[o], j] = 1.0
    np.savez(OUT / "presence_matrix.npz",
             P=P, all_orgs=np.array(all_orgs, dtype=object),
             all_ogs=np.array(all_ogs, dtype=object))
    print(f"presence matrix: {P.shape[0]} orgs x {P.shape[1]} OGs "
          f"(base orgs={n_base_orgs}, extra rows merged={n_extra})")
    print("wrote presence_matrix.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_presence_csv", default=None)
    a = ap.parse_args()
    build(a.extra_presence_csv)
