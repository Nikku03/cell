"""FETCH GSE148175 -- nascent transcription under a CAUSAL chromatin perturbation.

WHY THIS DATASET, AND WHAT IT IS AND IS NOT FOR. Loop 198 measured that nothing in this project
beats PERSISTENCE at predicting expression change over an unseen interval: informed R2 -0.0520
against persistence -0.0295. One likely reason is that the readout was wrong. Steady-state mRNA
convolves transcription with degradation, so a gene whose transcription rate doubles shows a slow
rise governed by its half-life -- mRNA level is persistent BY CONSTRUCTION, which makes persistence
almost unbeatable and makes the accessibility clock hard to see. The quantity loop 191d's clock
should predict is the transcription RATE, and that needs a nascent assay.

A search over GEO for human nascent time courses (PRO-seq, TT-seq, GRO-seq) returns GSE148175 as
the best match on everything except density:

    NASCENT READOUT     PRO-seq, which measures engaged polymerase rather than accumulated mRNA.
    CAUSAL PERTURBATION BRM014 inhibits the BAF ATPase, so accessibility is FORCED to change rather
                        than observed changing. Every "leading is not causing" caveat in this arc's
                        cannot-show blocks exists because A549 is observational; this is not.
    MATCHED ASSAYS      ATAC and PRO-seq on the same cells under the same drug.
    A VEHICLE CONTROL   DMSO at matched times, which the A549 dexamethasone series does not have.
    TWO REPLICATES      per PRO-seq timepoint.

WHAT IT DOES NOT FIX, stated plainly because it is the constraint that has blocked this arc since
loop 192. The grids are:

    ATAC     5, 10, 30, 60, 360, 1440 min   (6 points)
    PRO-seq     10, 30, 60 min              (3 points)
    matched     10, 30, 60 min              (3 points)

Three shared timepoints is FEWER than the dendritic-cell series' four, and loop 196 measured that
four cannot support a response-time statistic -- four estimators chosen to fail differently all
recovered the A549 lead on eleven points and none on four. So this dataset cannot be used to
replicate the clock as loop 191d measured it, and any loop that tries must not pretend otherwise.

WHAT IT CAN ANSWER is a different question with a different estimator: CROSS-LAGGED PRECEDENCE. With
a forced perturbation and a vehicle control, one can ask whether the accessibility change at 10 min
predicts the nascent transcription change at 30 and 60 min more than the concurrent or reverse
pairing does. That is a directional test rather than a response-time estimate, it needs its own
power calibration in the manner of loop 192's W3 before any result from it is read, and it is
something the A549 series structurally cannot provide because nothing there was perturbed.

Processed count matrices only -- 7 MB against the 20 GB raw tar that GSE172051 offers and this disk
cannot hold.

-> {scratchpad}/gse148175/
"""
import os
import sys
import urllib.request
from pathlib import Path

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
                         "scratchpad"))
DIR = SP / "gse148175"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE148nnn/GSE148175/suppl"
FILES = (
    "GSE148175_Pro-Seq-single-rep-raw-counts-BRM014-timecourse.txt.gz",
    "GSE148175_count_matrix_raw_atac_BRM014_ACBI1.csv.gz",
    "GSE148175_matrix_raw_timecourse.txt.gz",
    "GSE148175_Pro-Seq-single-rep-raw-counts-WT-SMARCA4-dtag-dtag-timecourse.txt.gz",
    "GSE148175_Pro-Seq-single-rep-raw-counts-genebody-BRM014.txt.gz",
    "GSE148175_Pro-Seq-single-rep-raw-counts-genebody-WT-SMARCA4-dtag.txt.gz",
)


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("FETCH -- GSE148175, PRO-seq + ATAC under BAF ATPase inhibition (processed matrices)")
    print("=" * 100)
    total = 0
    for f in FILES:
        p = DIR / f
        if not p.exists():
            for attempt in range(4):
                try:
                    r = urllib.request.Request(f"{BASE}/{f}", headers={"User-Agent": "cellos"})
                    p.write_bytes(urllib.request.urlopen(r, timeout=300).read())
                    break
                except Exception as exc:                          # noqa: BLE001
                    if attempt == 3:
                        print(f"  FAILED {f}: {exc}")
                        continue
        if p.exists():
            total += p.stat().st_size
            print(f"  {p.stat().st_size/1e6:6.2f} MB  {f}")
    print(f"\n  {total/1e6:.1f} MB total -> {DIR}")
    print("  matched ATAC/PRO-seq grid: 10, 30, 60 min -- THREE points, fewer than the")
    print("  dendritic-cell series' four, which loop 196 measured as too few for a response time.")
    print("  This dataset is for a cross-lagged precedence test, not for replicating the clock.")


if __name__ == "__main__":
    main()
