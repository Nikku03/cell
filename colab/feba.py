"""feba — bridge the LBL Fitness Browser (bacterial RB-TnSeq co-fitness) onto HUMAN gene pairs, via KEGG KO.

Two bacterial genes whose fitness tracks across thousands of experiments are functionally coupled. FEBA is
bacterial, so to score a HUMAN pair we bridge through KEGG Orthology (KO numbers are cross-species):

    FEBA (orgId, locusId) --fb_ko_mapping.tsv--> KO     (bacterial side; you already have this)
    cofit(locusId, hitId)  aggregated to  cofit(KO_a, KO_b)   = strongest co-fitness seen for that KO pair
    KO --KEGG 'hsa'--> human gene                            (human side; fetched once, cached)
    => cofit(human_a, human_b)  (sparse; only the conserved core is covered — 0 elsewhere, honest)

Output (same sparse-pair shape as STRING, consumed by signal_combiner._load_feba):
    outputs/orphan/feba_cofit.json   {"edges": [[gene_idx_a, gene_idx_b, cofit], ...], "meta": {...}}

Defensive + diagnostic: introspects the sqlite schema and prints counts at every stage, so the first Colab run
tells us the exact columns (same way STRING / Geneformer / ESM were dialled in). Nothing measured is overwritten;
this only produces a new combiner feature that add->measure->keep then judges.

env:  FEBA_DB (feba.db sqlite)   FEBA_KO_MAP (fb_ko_mapping.tsv)   [FEBA_HUMAN_KO optional pre-made symbol<TAB>KO]
"""
import os, sys, json, csv, sqlite3, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
COFIT_MIN = 0.5        # keep only strong co-fitness (|rho| >= this) — bounds size and keeps the signal meaningful
MAX_GENES_PER_KO = 15  # a KO with a huge human paralog family would explode the pair list; cap it (LOGGED)


def _cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _pick(cols, *names):
    low = [c.lower() for c in cols]
    for nm in names:
        if nm in low:
            return cols[low.index(nm)]
    return None


def load_ko_map(path):
    """fb_ko_mapping.tsv -> {(orgId, locusId): KO}. Header includes orgId, locusId, kegg_ko."""
    if not path or not os.path.exists(path):
        print("  (feba: no fb_ko_mapping.tsv — set FEBA_KO_MAP)"); return None
    with open(path) as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        low = [h.lower() for h in header]
        io = low.index("orgid") if "orgid" in low else 0
        il = low.index("locusid") if "locusid" in low else 1
        ik = next((i for i, h in enumerate(low) if "ko" in h and ("kegg" in h or h == "ko")), len(header) - 1)
        print(f"  feba ko-map columns: org={header[io]} locus={header[il]} ko={header[ik]}")
        m = {}
        for p in rd:
            if len(p) > max(io, il, ik) and p[ik] and p[ik].upper().startswith("K"):
                m[(p[io], p[il])] = p[ik].strip()
    print(f"  feba ko-map: {len(m):,} bacterial loci -> KO")
    return m


def kopairs_from_db(db, ko_map):
    """stream the cofit table, map both loci to KO, aggregate to KO-pair -> strongest |cofit|."""
    con = sqlite3.connect(db)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if "cofit" not in tables:
        print(f"  (feba: no 'cofit' table in db; tables = {tables[:12]})"); con.close(); return None
    cols = _cols(con, "cofit")
    cl = _pick(cols, "locusid", "locus", "geneid")
    ch = _pick(cols, "hitid", "hit", "locusid2")
    cc = _pick(cols, "cofit", "cor", "correlation", "rho")
    co = _pick(cols, "orgid", "org")
    print(f"  feba cofit columns: {cols}  -> org={co} locus={cl} hit={ch} cofit={cc}")
    if not (cl and ch and cc and co):
        print("  (feba: could not identify cofit columns — schema above for refinement)"); con.close(); return None
    q = (f"SELECT {co},{cl},{ch},{cc} FROM cofit "
         f"WHERE abs(CAST({cc} AS REAL)) >= {COFIT_MIN}")
    kop, seen, mapped = {}, 0, 0
    for org, la, lb, val in con.execute(q):
        seen += 1
        ka, kb = ko_map.get((org, la)), ko_map.get((org, lb))
        if not ka or not kb or ka == kb:
            continue
        mapped += 1
        try:
            v = abs(float(val))
        except (TypeError, ValueError):
            continue
        key = (ka, kb) if ka < kb else (kb, ka)
        if v > kop.get(key, 0.0):
            kop[key] = v
    con.close()
    print(f"  feba cofit: scanned {seen:,} strong pairs, {mapped:,} KO-mapped -> {len(kop):,} unique KO pairs")
    return kop


def human_ko(cache=f"{OUT}/feba_human_ko.tsv", pre=None):
    """symbol -> {KO}. From a provided map, a cache, or fetched once from KEGG REST ('hsa')."""
    src = pre if (pre and os.path.exists(pre)) else (cache if os.path.exists(cache) else None)
    if src:
        h = {}
        with open(src) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1]:
                    h.setdefault(p[0], set()).add(p[1])
        print(f"  feba human-KO: {len(h):,} human symbols (from {os.path.basename(src)})")
        return h
    try:
        def kegg(path):
            with urllib.request.urlopen("https://rest.kegg.jp/" + path, timeout=120) as r:
                return r.read().decode()
        ent2sym = {}
        for line in kegg("list/hsa").splitlines():        # 'hsa:7157\tTP53, P53, ...; tumor protein p53'
            p = line.split("\t")
            if len(p) == 2:
                ent2sym[p[0]] = p[1].split(";")[0].split(",")[0].strip()
        h = {}
        for line in kegg("link/ko/hsa").splitlines():     # 'hsa:7157\tko:K04451'
            p = line.split("\t")
            if len(p) == 2:
                sym, ko = ent2sym.get(p[0]), p[1].replace("ko:", "").strip()
                if sym and ko:
                    h.setdefault(sym, set()).add(ko)
        with open(cache, "w") as fh:
            for sym, kos in h.items():
                for ko in kos:
                    fh.write(f"{sym}\t{ko}\n")
        print(f"  feba human-KO: {len(h):,} human symbols (fetched from KEGG, cached)")
        return h
    except Exception as e:
        print("  (feba: KEGG human-KO fetch failed:", str(e)[:60], "— set FEBA_HUMAN_KO to a symbol<TAB>KO file)")
        return None


def main():
    C = CompleteCell()
    ko_map = load_ko_map(os.environ.get("FEBA_KO_MAP"))
    db = os.environ.get("FEBA_DB")
    if not ko_map or not db or not os.path.exists(db):
        print("  (feba off: need FEBA_DB + FEBA_KO_MAP)"); return None
    kop = kopairs_from_db(db, ko_map)
    if not kop:
        return None
    hko = human_ko(pre=os.environ.get("FEBA_HUMAN_KO"))
    if not hko:
        return None
    idx = C.idx
    ko2human = {}                                          # KO -> [our gene idx], capped to avoid paralog blow-up
    for sym, kos in hko.items():
        if sym in idx:
            for ko in kos:
                ko2human.setdefault(ko, []).append(idx[sym])
    big = [k for k, v in ko2human.items() if len(v) > MAX_GENES_PER_KO]
    for k in big:
        del ko2human[k]
    print(f"  feba KO->human: {len(ko2human):,} KOs map to human genes "
          f"({len(big)} huge paralog KOs dropped, cap {MAX_GENES_PER_KO})")
    edges = {}
    for (ka, kb), v in kop.items():
        ha, hb = ko2human.get(ka), ko2human.get(kb)
        if not ha or not hb:
            continue
        for i in ha:
            for j in hb:
                if i != j:
                    key = (i, j) if i < j else (j, i)
                    if v > edges.get(key, 0.0):
                        edges[key] = v
    if not edges:
        print("  (feba: 0 human pairs after bridge — KO overlap too thin)"); return None
    out = {"edges": [[i, j, round(v, 4)] for (i, j), v in edges.items()],
           "meta": dict(ko_pairs=len(kop), human_pairs=len(edges), cofit_min=COFIT_MIN,
                        genes_covered=len({g for e in edges for g in e}))}
    json.dump(out, open(f"{OUT}/feba_cofit.json", "w"))
    m = out["meta"]
    print(f"  feba: {m['human_pairs']:,} human co-fitness pairs covering {m['genes_covered']:,} genes "
          f"(|cofit| >= {COFIT_MIN}) -> feba_cofit.json")
    return m


if __name__ == "__main__":
    main()
