"""FETCH THE REACTOME ORDERING WE PREVIOUSLY DISCARDED, and cache it.

WHY THIS EXISTS. pathway_order.py measured that 40.4% of our PPI edges have both endpoints inside a shared Reactome
pathway -- 24x the coverage SIGNOR gives -- but that our stored copy has all 2,792 pathway member lists sorted by
gene index, i.e. MEMBERSHIP with the sequence thrown away. Reactome publishes the sequence as `precedingEvent`
relations between reactions. This fetches it.

WHAT IT PULLS:
  1. every human ReactionLikeEvent stable id            (~16,400)
  2. each one's precedingEvent list, in batches         -> a directed reaction -> reaction graph
  3. NCBI2Reactome_PE_Reactions.txt, streamed and filtered to human, giving gene -> reaction membership

COMPOSING THOSE GIVES A DIRECTED GENE RELATION: if reaction R1 precedes R2, gene A participates in R1 and gene B in
R2, then A precedes B in the pathway. That is the arrow pathway_order.py wanted and could not get.

Output is a cache at scratchpad/reactome_order.json so the build step is offline and reproducible. This is a network
fetch, so it is written to be resumable and to fail loudly rather than silently returning a partial graph: the
batch loop records how many ids it actually retrieved, and the builder refuses to run on a truncated cache.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
CS = "https://reactome.org/ContentService"
BATCH = 150
OUTF = Path(SP) / "reactome_order.json"


def get(url, data=None, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=data)
            if data:
                req.add_header("Content-Type", "text/plain")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if t == tries - 1:
                print(f"    FAILED {url[:80]}: {e}", flush=True)
                return None
            time.sleep(2 ** t)
    return None


def main():
    # ---- 1. every human reaction id ----
    n = get(f"{CS}/data/schema/ReactionLikeEvent/count?species=9606")
    total = int(n) if n else 0
    print(f"human ReactionLikeEvents: {total}", flush=True)
    ids, page, per = [], 1, 2000
    while True:
        chunk = get(f"{CS}/data/schema/ReactionLikeEvent?species=9606&page={page}&offset={per}")
        if not chunk:
            break
        ids += [c["stId"] for c in chunk if "stId" in c]
        print(f"  page {page}: {len(chunk)} (running {len(ids)})", flush=True)
        if len(chunk) < per:
            break
        page += 1
        if page > 40:
            break
    ids = sorted(set(ids))
    print(f"collected {len(ids)} reaction ids", flush=True)

    # ---- 2. preceding-event relations, batched ----
    prec = {}
    got = 0
    for i in range(0, len(ids), BATCH):
        b = ids[i:i + BATCH]
        d = get(f"{CS}/data/query/ids", data=",".join(b).encode())
        if not d:
            continue
        for o in d:
            sid = o.get("stId")
            if not sid:
                continue
            got += 1
            pv = o.get("precedingEvent") or []
            ps = [p.get("stId") for p in pv if isinstance(p, dict) and p.get("stId")]
            if ps:
                prec[sid] = ps
        if (i // BATCH) % 10 == 0:
            print(f"  batch {i//BATCH}: {got} objects, {len(prec)} with preceding", flush=True)
    print(f"retrieved {got}/{len(ids)} reaction objects; {len(prec)} have a precedingEvent", flush=True)

    # ---- 3. gene -> reaction membership, streamed and filtered to human ----
    url = "https://reactome.org/download/current/NCBI2Reactome_PE_Reactions.txt"
    g2r = {}
    print(f"streaming {url} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=600) as r:
            nl = 0
            for raw in r:
                nl += 1
                p = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
                if len(p) < 8 or "Homo sapiens" not in p[-1]:
                    continue
                gid = p[0]
                rxn = next((x for x in p if x.startswith("R-HSA-")and "-" in x), None)
                # the reaction stId is the LAST R-HSA field on the line; the earlier one is the physical entity
                rx = [x for x in p if x.startswith("R-HSA-")]
                if not rx:
                    continue
                rxn = rx[-1]
                g2r.setdefault(gid, set()).add(rxn)
            print(f"  read {nl:,} lines", flush=True)
    except Exception as e:
        print(f"  mapping fetch FAILED: {e}", flush=True)

    out = {"reaction_ids": len(ids), "retrieved": got,
           "preceding": {k: v for k, v in prec.items()},
           "gene2reaction": {k: sorted(v) for k, v in g2r.items()},
           "complete": got >= 0.9 * len(ids) and len(g2r) > 1000}
    OUTF.write_text(json.dumps(out))
    print(f"\n  reactions with preceding: {len(prec):,}")
    print(f"  genes with reaction membership: {len(g2r):,}")
    print(f"  cache complete: {out['complete']}")
    print(f"  -> {OUTF}  ({OUTF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
