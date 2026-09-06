"""A BUILDER FOR cell_complete.json -- because it did not have one.

WHAT WAS WRONG.  cell_complete.json is 38.5 MB, read by 56 modules in the traced sweep and named by 227
across the repo, and it is EXPLICITLY GITIGNORED. Two independent methods agree that nothing produces it:
a static scan of every write in the repository, and the runtime tracer over 153 executed modules. Zero
writers, 56 readers. When it was patched this morning there was no version-controlled original to diff
against, and the edit could only be verified because an unrelated .gz from an earlier date happened to be
sitting beside it.

So the single most depended-upon artefact in this repository is unversioned, unrebuildable, and was until
today unbacked. That is the largest reproducibility risk here and it is worth fixing before anything else
is built on top of it.

WHAT A BUILDER CAN AND CANNOT RECOVER.  Of the 42 top-level blocks, exactly ONE (cellcycle) has a module
in this repo whose output name matches it. The other 41 came from processes that are not here. No amount
of engineering reconstructs their science, and pretending otherwise would be worse than the current state.

    reg      612,133 edges  10.87 MB     go        16,412 genes   6.88 MB
    genes     16,492         6.61 MB     coexpr    16,374         3.14 MB
    ppi      191,447         2.82 MB     model4     7,700         1.58 MB
    ...and 36 more

WHAT THIS BUILDER DOES INSTEAD, which is the part that is actually fixable.  It converts a monolith with
no builder into an ASSEMBLY with one:

    split      decompose into one gzipped artefact per block, each independently hashed
    assemble   reconstruct cell_complete.json from those parts
    provenance record, per block, whether a producer is known -- 1 of 42 today, and the list of 41 is
               the actual recovery backlog rather than a vague worry

The point is not compression. It is that a block which LATER gains a producer can be regenerated on its
own and dropped back in, without touching the other 41 and without anyone needing to rebuild the whole
file. cell_coords.py already did exactly this by hand for the genes block's coordinate fields, sourcing
them from genes.parquet. The builder makes that the normal path rather than a one-off.

It also makes the file committable. The parts total roughly 6.6 MB gzipped against 38.5 MB raw, which is
small enough to version, so the next session inherits the data instead of discovering it is gone.

THE CONTROL, and it can fail.  assemble(split(x)) must reproduce x EXACTLY -- same top-level keys, same
block contents, same gene count, same sha256 on a canonical re-serialisation. A builder whose round trip
does not close is not a builder, it is a lossy export, and the difference matters when 56 modules read the
result.

PREDECLARED, before any number:
    the round trip reproduces the original hash
        -> the assembly is faithful and cell_complete can be stored and rebuilt as parts.
    it does not
        -> the split loses something. Report which block and do NOT ship the parts as a replacement.
    more than one block turns out to have a producer
        -> better than expected; the recovery backlog is smaller than 41.

-> outputs/orphan/cell_parts/  (the parts)  and  outputs/cell_complete_build.json  (the record)
"""
import collections
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
PARTS = ROOT / "outputs" / "orphan" / "cell_parts"
SEED = 1904

# Blocks whose producer is known. Verified by name-matching every write in the repo, static and traced.
# One entry today. Every addition here shrinks the recovery backlog by one.
KNOWN_PRODUCER = {
    "cellcycle": "colab/cellcycle.py",
    # `genes` is PARTLY sourced: strand/gene_start/gene_end/ens_tss come from
    # outputs/orphan/science_data/genes.parquet via colab/cell_coords.py. The other 20 fields do not.
}
PARTIAL_PRODUCER = {
    "genes": "colab/cell_coords.py rebuilds strand, gene_start, gene_end, ens_tss from genes.parquet "
             "(GRCh38.p14); the remaining 20 fields have no known source",
}


def canon(obj):
    """Canonical serialisation, so a hash compares CONTENT and not key order or whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def split(cell=CELL, dest=PARTS):
    dest.mkdir(parents=True, exist_ok=True)
    D = json.load(open(cell))
    rows = []
    for k, v in D.items():
        blob = canon(v)
        p = dest / f"{k}.json.gz"
        with gzip.open(p, "wb", compresslevel=6) as fh:
            fh.write(blob)
        rows.append({"block": k, "type": type(v).__name__,
                     "items": len(v) if isinstance(v, (list, dict)) else 1,
                     "raw_bytes": len(blob), "gz_bytes": p.stat().st_size, "sha256": sha16(blob),
                     "producer": KNOWN_PRODUCER.get(k),
                     "partial": PARTIAL_PRODUCER.get(k)})
    order = list(D)
    json.dump({"order": order, "parts": rows}, open(dest / "_index.json", "w"), indent=2)
    return rows, order


def assemble(src=PARTS):
    """Rebuild the whole object from parts, preserving the original key order."""
    idx = json.load(open(src / "_index.json"))
    D = {}
    for k in idx["order"]:
        with gzip.open(src / f"{k}.json.gz", "rb") as fh:
            D[k] = json.loads(fh.read())
    return D


def main():
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("A BUILDER FOR cell_complete.json -- because it did not have one")
    report("=" * 100)
    report("  38.5 MB, read by 56 modules in the traced sweep and named by 227, gitignored, and produced")
    report("  by NOTHING: zero writers by static scan and zero by the runtime tracer, against 56 readers.")
    report("  It is the most depended-upon artefact here and the least recoverable.")

    original = json.load(open(CELL))
    oh = sha16(canon(original))
    report(f"\n  original: {len(original)} blocks, {CELL.stat().st_size/1e6:.1f} MB on disk, "
           f"canonical hash {oh}")

    rows, order = split()
    graw = sum(r["raw_bytes"] for r in rows)
    ggz = sum(r["gz_bytes"] for r in rows)
    report(f"\n  SPLIT into {len(rows)} parts: {graw/1e6:.1f} MB raw -> {ggz/1e6:.1f} MB gzipped "
           f"({ggz/graw:.1%})")
    report(f"    {'block':<18}{'items':>10}{'gz MB':>9}  producer")
    for r in sorted(rows, key=lambda x: -x["gz_bytes"])[:10]:
        p = r["producer"] or ("PARTIAL" if r["partial"] else "-- none known --")
        report(f"    {r['block']:<18}{r['items']:>10,}{r['gz_bytes']/1e6:>9.2f}  {p}")

    # ---- THE CONTROL ---------------------------------------------------------------------------------
    rebuilt = assemble()
    rh = sha16(canon(rebuilt))
    same_keys = list(rebuilt) == list(original)
    same_hash = rh == oh
    report(f"\n  ROUND TRIP -- assemble(split(x)) must reproduce x exactly")
    report(f"    key order preserved : {same_keys}   ({len(rebuilt)} blocks)")
    report(f"    canonical hash      : {rh}  {'MATCHES' if same_hash else 'DIFFERS from ' + oh}")
    if same_hash:
        report(f"    gene count          : {len(rebuilt.get('genes', []))} unchanged")
        report("    The assembly is faithful. cell_complete can now be STORED and REBUILT as parts.")
    else:
        bad = [k for k in original if canon(original[k]) != canon(rebuilt.get(k))]
        report(f"    LOSSY -- blocks that differ: {bad[:6]}")
        report("    Not shipping these parts as a replacement. This is a defect in the split.")

    # ---- the recovery backlog ------------------------------------------------------------------------
    known = [r for r in rows if r["producer"]]
    partial = [r for r in rows if r["partial"]]
    unknown = [r for r in rows if not r["producer"] and not r["partial"]]
    report(f"\n  THE RECOVERY BACKLOG -- what a builder still cannot make")
    report(f"    {len(known)} block(s) have a known producer:  {[r['block'] for r in known]}")
    report(f"    {len(partial)} partially sourced:               {[r['block'] for r in partial]}")
    report(f"    {len(unknown)} have NO known producer, {sum(r['gz_bytes'] for r in unknown)/1e6:.1f} MB "
           f"gzipped")
    report("    Those 40 came from processes that are not in this repository. This builder does not")
    report("    invent them, and listing them by name is the point: it converts a vague worry into a")
    report("    finite backlog, where each entry can be closed independently by finding one source.")
    report(f"    largest unsourced: {', '.join(r['block'] for r in sorted(unknown, key=lambda x: -x['gz_bytes'])[:8])}")

    man = RM.manifest(inputs=[str(CELL)], available=len(original), used=len(rows),
                      selection="all", seed=SEED,
                      controls=["round-trip canonical hash", "key-order preservation"],
                      note="one gzipped artefact per top-level block, independently hashed")
    RM.report(man, emit=report)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "cell_complete_build", "manifest": man, "original_hash": oh,
               "rebuilt_hash": rh, "round_trip_ok": bool(same_hash and same_keys),
               "n_blocks": len(rows), "raw_bytes": graw, "gz_bytes": ggz,
               "known_producer": [r["block"] for r in known],
               "partial_producer": [r["block"] for r in partial],
               "no_producer": [r["block"] for r in unknown], "parts": rows, "log": log},
              open(OUT / "cell_complete_build.json", "w"), indent=2)
    report(f"\n  parts -> {PARTS}")
    report(f"  record -> {OUT/'cell_complete_build.json'}")
    return 0 if same_hash else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--assemble":
        D = assemble()
        tmp = CELL.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(D, fh)
        os.replace(tmp, CELL)
        print(f"rebuilt {CELL} from {PARTS}: {len(D)} blocks, "
              f"{len(D.get('genes', []))} genes, {CELL.stat().st_size/1e6:.1f} MB")
        raise SystemExit(0)
    raise SystemExit(main())
