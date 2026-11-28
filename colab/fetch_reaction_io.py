"""FETCH REACTION INPUTS AND OUTPUTS -- the "one step back" that precedingEvent misses.

WHY. The directed network built from `precedingEvent` covers within-pathway sequence. But a pathway does not begin
spontaneously: its first reaction consumes a protein that had to be translated, folded, modified, complexed and
trafficked first, and those preparatory reactions usually live in OTHER pathways, so no precedingEvent connects
them. Reactome does record the connection, implicitly -- as entity flow. If reaction A OUTPUTS entity E and
reaction B takes E as INPUT, then A prepares B, whatever pathway each belongs to.

    output(A) intersect input(B) != empty   =>   A precedes B

That is strictly more information than precedingEvent, and it is what "take it one step back" means concretely.

THE TRAP THAT MAKES THIS EASY TO GET WRONG. Reaction inputs include currency molecules -- ATP, ADP, H2O, phosphate,
NAD+. ATP is an input to thousands of reactions and an output of thousands more, so joining on it would declare that
essentially every reaction precedes every other. This is the classic failure in building metabolic networks from
stoichiometry, and it does not announce itself: it produces a huge, dense, confident-looking graph. The guard here
is data-driven rather than a hand-written blacklist: count how many distinct reactions touch each entity, and drop
the ones above a threshold. The threshold is swept in the builder so its effect on both coverage and accuracy is
visible rather than assumed.

WHAT IS FETCHED: for all 16,173 reactions, the `input` and `output` physical-entity stIds, plus className so that
Pathway objects (about 1% of the id set) can be separated from genuine Reactions. Batch size is pinned to 20, which
is the measured silent cap on Reactome's bulk endpoint -- posting more returns 20 anyway, with no error.
"""
import json
import time
import urllib.request
from pathlib import Path

SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
CS = "https://reactome.org/ContentService"
UA = {"User-Agent": "Mozilla/5.0 (compatible; cell-network-research/1.0)"}
BATCH = 20                      # the measured silent cap; larger requests return 20 and drop the rest
OUTF = Path(SP) / "reaction_io.json"


def post(ids, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(f"{CS}/data/query/ids", data=",".join(ids).encode(),
                                         headers=dict(UA, **{"Content-Type": "text/plain"}))
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if t == tries - 1:
                return None
            time.sleep(2 ** t)
    return None


def main():
    ids = json.loads((Path(SP) / "rxn_ids.json").read_text())
    print(f"{len(ids):,} reaction ids to resolve, batch {BATCH}", flush=True)
    io, cls, got = {}, {}, 0
    t0 = time.time()
    for i in range(0, len(ids), BATCH):
        d = post(ids[i:i + BATCH])
        if not d:
            continue
        for o in d:
            sid = o.get("stId")
            if not sid:
                continue
            got += 1
            cls[sid] = o.get("className")
            inp = [e.get("stId") for e in (o.get("input") or []) if isinstance(e, dict) and e.get("stId")]
            outp = [e.get("stId") for e in (o.get("output") or []) if isinstance(e, dict) and e.get("stId")]
            if inp or outp:
                io[sid] = {"in": inp, "out": outp}
        if (i // BATCH) % 100 == 0:
            el = time.time() - t0
            print(f"  batch {i//BATCH}/{len(ids)//BATCH}: {got} objects, {len(io)} with io, "
                  f"{el:.0f}s elapsed ({(i//BATCH+1)/max(el,1):.2f} req/s)", flush=True)
    el = time.time() - t0
    nrx = sum(1 for v in cls.values() if v == "Reaction")
    print(f"\nretrieved {got}/{len(ids)} ({nrx} className=Reaction); {len(io)} have input or output")
    print(f"elapsed {el/60:.1f} min at {len(ids)/BATCH/max(el,1):.2f} req/s")
    OUTF.write_text(json.dumps({"io": io, "className": cls, "retrieved": got, "total": len(ids),
                                "complete": bool(got >= 0.9 * len(ids))}))
    print(f"  -> {OUTF} ({OUTF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
