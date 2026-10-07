"""fetch_complexes — pull as many real protein-protein COMPLEX structures as the surface model needs, straight from the
PDB via the RCSB Search API. SKEMPI's 345 complexes are only the LABELLED (measured-ΔΔG) set; the dMaSIF surface sensor
trains on interface GEOMETRY (self-supervised on cross-chain contacts — no labels), so it can consume the ~34,000
heteromeric complexes in the PDB. This is the DIPS-style data source that surface-learning models (dMaSIF, EquiDock) use.

  rcsb_search(...)  -> a list of PDB ids of heteromeric protein complexes matching the filters (resolution, size, human)
  download(...)     -> fetch those .pdb files into a cache dir (skips ones already present; retries)
  fetch(n, cache)   -> search + download n usable complexes; the one call the notebook/driver uses

Filters keep them tractable for the surface+APBS pipeline: heteromeric (a real PPI, not a homo-oligomer), resolution
cutoff, and an atom-count cap (so no ribosomes/viruses that would choke APBS). human_only restricts to Homo sapiens.
"""
import os, sys, json, time, urllib.request
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
DL = "https://files.rcsb.org/download/{}.pdb"


def rcsb_search(n=1000, max_res=3.0, max_atoms=15000, human_only=False, page=1000):
    """return up to n PDB ids of heteromeric protein complexes matching the filters (paginated)."""
    def term(attr, op, val):
        return {"type": "terminal", "service": "text", "parameters": {"attribute": attr, "operator": op, "value": val}}
    nodes = [term("rcsb_entry_info.polymer_composition", "exact_match", "heteromeric protein"),
             term("rcsb_entry_info.resolution_combined", "less_or_equal", max_res),
             term("rcsb_entry_info.deposited_atom_count", "less_or_equal", max_atoms)]
    if human_only:
        nodes.append(term("rcsb_entity_source_organism.taxonomy_lineage.name", "exact_match", "Homo sapiens"))
    ids, start = [], 0
    while len(ids) < n:
        rows = min(page, n - len(ids))
        q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes}, "return_type": "entry",
             "request_options": {"paginate": {"start": start, "rows": rows}, "results_content_type": ["experimental"]}}
        req = urllib.request.Request(SEARCH, data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            break
        batch = [x["identifier"] for x in r.get("result_set", [])]
        if not batch:
            break
        ids += batch; start += len(batch)
        if start >= r.get("total_count", 0):
            break
    return ids[:n]


def _one(p, cache):
    dst = f"{cache}/{p}.pdb"
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return p
    for _ in range(3):
        try:
            urllib.request.urlretrieve(DL.format(p), dst)
            break
        except Exception:
            time.sleep(1)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return p
    if os.path.exists(dst):
        os.remove(dst)
    return None


def download(pdb_ids, cache, limit=None, verbose=True, threads=16):
    """download the given PDB ids into `cache` IN PARALLEL (IO-bound); return the ones that landed. Skips cached files."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(cache, exist_ok=True)
    ids = pdb_ids[:limit] if limit else pdb_ids
    ok, t0, done = [], time.time(), 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for r in ex.map(lambda p: _one(p, cache), ids):
            done += 1
            if r:
                ok.append(r)
            if verbose and done % 200 == 0:
                print(f"    downloaded {len(ok)}/{done} ({time.time()-t0:.0f}s)", flush=True)
    return ok


def fetch(n, cache, max_res=3.0, max_atoms=15000, human_only=False, overshoot=1.3):
    """the one call: search + download n usable protein-protein complexes into `cache`. Over-fetches ids since some
    downloads fail, then trims to n."""
    print(f"  RCSB search: heteromeric protein complexes  (res<={max_res}A, atoms<={max_atoms}"
          f"{', human' if human_only else ''}) ...", flush=True)
    ids = rcsb_search(int(n * overshoot), max_res, max_atoms, human_only)
    print(f"  {len(ids)} candidate ids; downloading up to {n} ...", flush=True)
    got = download(ids, cache, limit=None)
    return got[:n]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cache = sys.argv[2] if len(sys.argv) > 2 else "./pdb_cache"
    human = "--human" in sys.argv
    got = fetch(n, cache, human_only=human)
    print(f"fetched {len(got)} complexes into {cache}: {got[:20]}")
