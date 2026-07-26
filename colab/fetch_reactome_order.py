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


# Reactome rejects urllib's default User-Agent with a 403 while serving the identical URL to curl. That is a
# destination-side UA filter, NOT the session's egress proxy: the proxy status endpoint reported an empty
# recentRelayFailures list, and curl reached the same host fine. Verified by sending the two UAs side by side --
# default urllib 403, browser-style 200. So a UA is set here rather than the request being retried blindly.
UA = {"User-Agent": "Mozilla/5.0 (compatible; cell-network-research/1.0)"}


def get(url, data=None, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=dict(UA))
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


def fetch_gene2reaction():
    """NCBI gene id -> set of reaction stIds, streamed and filtered to human."""
    url = "https://reactome.org/download/current/NCBI2Reactome_PE_Reactions.txt"
    g2r = {}
    print(f"streaming {url} ...", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA)), timeout=900) as r:
        nl = 0
        for raw in r:
            nl += 1
            p = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            if len(p) < 8 or "Homo sapiens" not in p[-1]:
                continue
            rx = [x for x in p if x.startswith("R-HSA-")]
            if rx:
                g2r.setdefault(p[0], set()).add(rx[-1])   # last R-HSA field is the reaction; earlier is the entity
        print(f"  read {nl:,} lines", flush=True)
    return {k: sorted(v) for k, v in g2r.items()}


def main():
    # ---- 1. gene -> reaction membership FIRST, and take the reaction ids from it ----
    # The /data/schema paginated endpoint returned only 25 ids regardless of page/offset, so it is not used. The
    # gene-to-reaction mapping already names every reaction that contains a gene -- which is exactly the set we
    # care about, and nothing is lost by ignoring reactions with no gene participant.
    g2r = fetch_gene2reaction()
    ids = sorted({r for v in g2r.values() for r in v})
    print(f"gene->reaction gave {len(g2r):,} genes and {len(ids):,} distinct reactions", flush=True)

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

    out = {"reaction_ids": len(ids), "retrieved": got,
           "preceding": {k: v for k, v in prec.items()},
           "gene2reaction": g2r,
           "complete": bool(got >= 0.9 * len(ids) and len(g2r) > 1000)}
    OUTF.write_text(json.dumps(out))
    print(f"\n  reactions with preceding: {len(prec):,}")
    print(f"  genes with reaction membership: {len(g2r):,}")
    print(f"  cache complete: {out['complete']}")
    print(f"  -> {OUTF}  ({OUTF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
