"""Extract literature-curated function for every 'dark' gene (blank in our model) from UniProt — whose FUNCTION
annotations ARE the distillation of the primary literature by expert curators, with paper-level evidence codes.

For each dark gene: symbol -> UniProt accession (mygene) -> UniProt curated FUNCTION + name + subcellular
location + keywords, with an honest evidence tier read off the ECO codes:
  experimental  : ECO:0000269|PubMed  -> function demonstrated in a paper
  by_similarity : ECO:0000250         -> inferred from a homolog (weaker)
  predicted     : ECO:0000255/0000256 -> sequence-model inference
  none          : no FUNCTION comment -> genuinely uncharacterized (hand to structure/biophysical tier)

Checkpointed. -> outputs/orphan/dark_function_table.json  (+ .tsv)  the annotation layer to overlay the model.
"""
import json, re, time, urllib.request, urllib.parse
from pathlib import Path

OUT = Path("outputs/orphan")
CKPT = OUT / "dark_function_ckpt.json"
ECO_EXP = "ECO:0000269"
ECO_SIM = "ECO:0000250"
ECO_PRED = ("ECO:0000255", "ECO:0000256")


def _clean(text):
    """strip the {ECO:...} evidence blocks from a UniProt comment for a human-readable function string."""
    return re.sub(r"\s*\{[^}]*\}", "", text).replace("FUNCTION: ", "").strip()


def _evidence_tier(raw):
    if ECO_EXP in raw:
        return "experimental"
    if ECO_SIM in raw:
        return "by_similarity"
    if any(e in raw for e in ECO_PRED):
        return "predicted"
    return "curated" if raw.strip() else "none"


def map_symbols(symbols):
    """symbol -> reviewed UniProt accession (Swiss-Prot) via mygene, batched."""
    import mygene
    acc = {}
    hits = mygene.MyGeneInfo().querymany(symbols, scopes="symbol", fields="uniprot",
                                         species="human", verbose=False)
    for h in hits:
        up = (h.get("uniprot") or {}).get("Swiss-Prot")
        if up:
            acc[h["query"]] = up if isinstance(up, str) else up[0]
    return acc


def fetch_batch(accs, timeout=60):
    """UniProt stream: accession -> (name, raw_function, location, keywords) for a batch of accessions."""
    q = "accession:(" + " OR ".join(accs) + ")"
    fields = "accession,protein_name,cc_function,cc_subcellular_location,keyword"
    url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(
        {"query": q, "fields": fields, "format": "tsv"})
    data = urllib.request.urlopen(url, timeout=timeout).read().decode()
    out = {}
    for ln in data.splitlines()[1:]:
        parts = ln.split("\t")
        if len(parts) < 5:
            parts += [""] * (5 - len(parts))
        acc, name, func, loc, kw = parts[:5]
        out[acc] = dict(name=name, raw_function=func, location=_clean(loc), keywords=kw)
    return out


def run(batch_size=120, limit=None):
    D = json.load(open(OUT / "cell_complete.json"))
    genes = [g["name"] for g in D["genes"]]
    darkfn = D.get("darkfn", {})
    dark = [(genes[int(k)], int(k), darkfn[k]) for k in darkfn]
    if limit:
        dark = dark[:limit]
    symbols = [g for g, _, _ in dark]

    ckpt = json.load(open(CKPT)) if CKPT.exists() else {}
    acc_map = ckpt.get("acc_map") or map_symbols(symbols)
    fetched = ckpt.get("fetched", {})

    todo = [a for a in set(acc_map.values()) if a not in fetched]
    print(f"dark genes: {len(dark)} | mapped to UniProt: {len(acc_map)} | to fetch: {len(todo)}")
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            fetched.update(fetch_batch(batch))
        except Exception as e:
            print(f"  batch {i} error: {e}; retrying once"); time.sleep(3)
            try:
                fetched.update(fetch_batch(batch))
            except Exception as e2:
                print(f"  batch {i} failed twice: {e2}")
        if i % (batch_size * 5) == 0:
            json.dump({"acc_map": acc_map, "fetched": fetched}, open(CKPT, "w"))
            print(f"  fetched {len(fetched)}/{len(set(acc_map.values()))} ...", flush=True)
    json.dump({"acc_map": acc_map, "fetched": fetched}, open(CKPT, "w"))

    # assemble the table
    table = {}
    tiers = {"experimental": 0, "by_similarity": 0, "predicted": 0, "curated": 0, "none": 0, "no_uniprot": 0}
    for sym, idx, df in dark:
        acc = acc_map.get(sym)
        if not acc or acc not in fetched:
            table[sym] = dict(uniprot=acc, function=None, evidence="no_uniprot",
                              coessentiality=df.get("pred"))
            tiers["no_uniprot"] += 1
            continue
        u = fetched[acc]
        tier = _evidence_tier(u["raw_function"])
        table[sym] = dict(uniprot=acc, name=u["name"].split(" (")[0], function=_clean(u["raw_function"]) or None,
                          evidence=tier, location=u["location"] or None,
                          keywords=[k for k in u["keywords"].split("; ") if k][:6],
                          coessentiality=df.get("pred"))
        tiers[tier] += 1

    json.dump(dict(n=len(table), evidence_tiers=tiers, table=table),
              open(OUT / "dark_function_table.json", "w"), indent=1)
    # human-readable TSV
    with open(OUT / "dark_function_table.tsv", "w") as f:
        f.write("gene\tuniprot\tname\tevidence\tfunction\tlocation\tco_essentiality\n")
        for sym, r in sorted(table.items()):
            f.write(f"{sym}\t{r.get('uniprot') or ''}\t{r.get('name','') or ''}\t{r['evidence']}\t"
                    f"{(r.get('function') or '')[:200]}\t{r.get('location') or ''}\t{r.get('coessentiality') or ''}\n")

    print("\n=== DARK FUNCTION EXTRACTION ===")
    print(f"  {len(table)} dark genes annotated. Evidence tiers:")
    for t, n in tiers.items():
        print(f"    {t:14} {n:5} ({100*n/len(table):.0f}%)")
    got = sum(1 for r in table.values() if r.get("function"))
    print(f"  => {got}/{len(table)} ({100*got/len(table):.0f}%) got a literature-curated function; "
          f"the rest are the genuinely uncharacterized tail (-> structure/biophysical).")
    print("-> outputs/orphan/dark_function_table.json (+ .tsv)")
    return table, tiers


if __name__ == "__main__":
    import sys
    run(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
