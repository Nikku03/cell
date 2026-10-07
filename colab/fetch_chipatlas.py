"""fetch_chipatlas -- download the K562 transcription-factor ChIP-seq peaks that chip_vs_perturb.py scores.

The peaks themselves are ~190 MB and are NOT committed; this script reproduces them. It selects exactly the factors that are testable -- those
ChIP-seq'd in K562 AND knocked out in the K562 Perturb-seq panel -- and caps the number of experiments per factor so that binding breadth stays
comparable across factors (a factor with 84 deposited experiments would otherwise look bound everywhere purely from sequencing depth).

WHY CHIP-ATLAS RATHER THAN ENCODE: encodeproject.org sits behind an AWS WAF bot challenge that returns an HTML verification page (HTTP 405) for
programmatic downloads from this environment. More importantly, ChIP-Atlas reprocesses every deposited experiment through ONE uniform pipeline,
and chip_vs_perturb's decisive control scores each factor's movers against OTHER factors' binding -- a comparison that would be corrupted if
different factors' peaks came from different peak-callers.

Usage: python fetch_chipatlas.py            (writes outputs/orphan/invivo/chipatlas/<TF>__<SRX>.bed)
Idempotent: existing non-empty files are kept, so an interrupted run can simply be re-run."""
import os
import json, subprocess, sys, collections
from pathlib import Path

SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DEST = Path("outputs/orphan/invivo/chipatlas")
META_URL = "https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab"
PEAK_URL = "https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/{srx}.05.bed"
UA = "Mozilla/5.0 (X11; Linux x86_64)"
GENOME, CELL, ANTIGEN_CLASS = "hg38", "K-562", "TFs and others"     # ChIP-Atlas spells K562 as "K-562"
CAPEXP = 4


def curl(url, out, timeout="600"):
    subprocess.run(["curl", "-sSL", "--max-time", timeout, "-H", f"User-Agent: {UA}", "-o", str(out), url],
                   capture_output=True)
    return out.exists() and out.stat().st_size > 500 and open(out, "rb").read(1) != b"<"


def main():
    import pandas as pd
    DEST.mkdir(parents=True, exist_ok=True)
    meta = SP / "ce.tab"
    if not meta.exists() or meta.stat().st_size < 10 ** 8:
        print(f"downloading ChIP-Atlas experiment metadata (~345 MB) -> {meta}")
        if not curl(META_URL, meta, timeout="1800"):
            print("metadata download failed"); return 1

    kos = {str(k) for k in pd.read_parquet(SP / "repl_k562_zscores.parquet").index}
    rows = collections.defaultdict(list)
    with open(meta, errors="replace") as f:
        for line in f:
            c = line.split("\t")
            if len(c) >= 6 and c[1] == GENOME and c[5].strip() == CELL and c[2].strip() == ANTIGEN_CLASS:
                rows[c[3].strip()].append(c[0])
    hit = {t: sorted(v) for t, v in rows.items() if t in kos}
    jobs = [(t, s) for t in sorted(hit) for s in hit[t][:CAPEXP]]
    print(f"{len(rows)} K562 TF antigens in ChIP-Atlas; {len(hit)} are also K562 knockouts; "
          f"{len(jobs)} experiments to fetch (<= {CAPEXP} per factor)")

    ok = fail = 0
    for i, (tf, srx) in enumerate(jobs):
        out = DEST / f"{tf}__{srx}.bed"
        if out.exists() and out.stat().st_size > 500:
            ok += 1; continue
        if curl(PEAK_URL.format(srx=srx), out):
            ok += 1
        else:
            fail += 1; out.unlink(missing_ok=True)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jobs)} ok={ok} fail={fail}", flush=True)
    print(f"DONE ok={ok} fail={fail} -> {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
