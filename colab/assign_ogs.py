"""Assign fetched proteomes to our existing OG space with mmseqs2.

Runs `mmseqs easy-search <each proteome> data/gtdb/og_reps.faa`, keeps the best
hit per query above identity/coverage thresholds, and maps the target header
(>OG#####|org|locus_tag) back to its og_id. Writes one tidy assignment table.

Requires the `mmseqs` binary on PATH (Colab: `!conda install -y -c bioconda mmseqs2`
or download the static build). If mmseqs is missing the script says so and exits.

Usage:
  python colab/assign_ogs.py --proteomes colab/work/proteomes \\
        --reps data/gtdb/og_reps.faa --out colab/work/og_assignments.csv
"""
import csv, argparse, subprocess, shutil, tempfile, os
from pathlib import Path


def have_mmseqs():
    return shutil.which("mmseqs") is not None


def search_one(faa, reps, min_pident, min_qcov, threads):
    """Return list of (query, og_id, pident, qcov) best-hits for one proteome."""
    with tempfile.TemporaryDirectory() as td:
        res = Path(td) / "res.m8"
        tmp = Path(td) / "tmp"
        cmd = ["mmseqs", "easy-search", str(faa), str(reps), str(res), str(tmp),
               "--threads", str(threads), "-s", "5.7",
               "--format-output", "query,target,pident,qcov,evalue,bits"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        best = {}
        for line in open(res):
            q, t, pid, qcov, ev, bits = line.rstrip("\n").split("\t")
            pid = float(pid); qcov = float(qcov); bits = float(bits)
            # mmseqs pident is 0-1 in some builds, 0-100 in others; normalise to %
            if pid <= 1.0:
                pid *= 100.0
            if pid < min_pident or qcov < min_qcov:
                continue
            og = t.split("|")[0]
            if q not in best or bits > best[q][2]:
                best[q] = (og, pid, bits)
        return [(q, og, pid) for q, (og, pid, _) in best.items()]


def main(proteomes, reps, out, min_pident, min_qcov, threads):
    if not have_mmseqs():
        raise SystemExit("mmseqs not found on PATH. Install it first, e.g.\n"
                         "  !conda install -y -c bioconda mmseqs2\n"
                         "or grab the static binary from github.com/soedinglab/MMseqs2")
    proteomes = Path(proteomes)
    faas = sorted(proteomes.glob("*.faa"))
    print(f"  {len(faas)} proteomes; reps={reps}; pident>={min_pident} qcov>={min_qcov}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["organism", "locus_tag", "og_id", "pident"])
        w.writeheader()
        for faa in faas:
            org = faa.stem
            hits = search_one(faa, reps, min_pident, min_qcov, threads)
            for q, og, pid in hits:
                w.writerow(dict(organism=org, locus_tag=q, og_id=og, pident=round(pid, 1)))
            n_total += len(hits)
            print(f"    {org:<34s} {len(hits):>6d} genes assigned")
    print(f"  wrote {out}  ({n_total:,} assignments)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteomes", default="colab/work/proteomes")
    ap.add_argument("--reps", default="data/gtdb/og_reps.faa")
    ap.add_argument("--out", default="colab/work/og_assignments.csv")
    ap.add_argument("--min_pident", type=float, default=30.0)
    ap.add_argument("--min_qcov", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    main(a.proteomes, a.reps, a.out, a.min_pident, a.min_qcov, a.threads)
