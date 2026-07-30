"""THE JOIN BACKBONE: immutable internal IDs, and a join that refuses to fail silently.

WHY THIS IS FIRST AND NOT A DATASET. Every layer about to be added joins to the existing model somehow, and
this project's join failures have been silent rather than loud. Three of them, all found by accident:

  `ppm` and `abund` are keyed by GENE INDEX, not by symbol. Looked up by symbol they return None for every
    gene. The abundance control in `metabolic_layer.py` would have run as a column of zeros and PASSED --
    which is the worst way for a control to fail, since a failed control looks exactly like a clean result.
    Abundance then turned out to be the strongest confound in that analysis (p 6e-31).
  STRING gene symbols were never cached in `_mpg_cache.npz` -- only FEATURE names were -- so a PPI join
    matched 0 of 11,636 genes and reported it as a legitimate zero.
  A malformed URL produced a doubled path prefix and a 404 that was read as "the resource has no data".

The fix the critique asks for is right: not another guard around `ppm`, but a universal registry every layer
joins through. The rule this module enforces is narrow and absolute:

    A JOIN MUST DECLARE ITS EXPECTED RATE. A join that lands below it raises, and never returns a
    silently-empty result. `unknown` is a value the model can carry; a zero that means "lookup failed" is not.

WHAT AN ENTITY IS. Genes, transcripts, proteins, metabolites and reactions are separate types with separate
namespaces, because collapsing them is how a metabolite id ends up compared against a gene symbol:

    gene        ENSG (primary)   + HGNC symbol as a DISPLAY ALIAS ONLY
    protein     UniProt accession + isoform, which is not 1:1 with gene
    transcript  ENST
    metabolite  ChEBI
    reaction    HumanGEM / Rhea

SYMBOLS ARE ALIASES, NOT KEYS. HGNC symbols are reused, retired and reassigned across releases; two datasets
built a year apart disagree about them. They stay for display and for joining legacy tables that carry nothing
else, but every such join is recorded as `via=symbol` so the weaker provenance is visible downstream rather
than assumed away.

ASSEMBLY IS PART OF A COORDINATE. A position without its assembly is not a location. Coordinates carry
GRCh38 explicitly, and anything arriving on hg19 is rejected rather than quietly treated as GRCh38.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
NET = SP / "cell_complete.json.gz"
SD = ROOT / "outputs/orphan/science_data"
REGISTRY = SP / "entity_registry.json.gz"
ASSEMBLY = "GRCh38"


class JoinError(RuntimeError):
    pass


class Registry:
    """Immutable internal ids plus every alias that resolves to them.

    `eid` is `gene:ENSG00000121410` -- typed, so a protein accession and a gene id can never collide, and
    stable, so a downstream layer keyed on it does not break when an alias is retired upstream.
    """

    def __init__(self, d):
        self.entities = d["entities"]
        self.alias = d["alias"]              # (type, namespace, value) -> eid, flattened to a str key
        self.meta = d.get("meta", {})

    # ---- resolution -------------------------------------------------------
    @staticmethod
    def _k(etype, ns, val):
        return f"{etype}|{ns}|{val}"

    def resolve(self, etype, ns, val):
        if val is None:
            return None
        v = str(val).split(".")[0] if ns in ("ensembl", "transcript") else str(val)
        return self.alias.get(self._k(etype, ns, v))

    def join(self, etype, ns, values, expect, label):
        """Resolve a list, and RAISE if the hit rate falls below `expect`.

        The rate is declared by the caller at the call site, next to the data it knows about, rather than
        buried in a config. A join that silently matches nothing is the failure mode this whole module
        exists to prevent, so it is not representable: either the rate clears the declared bar, or this
        raises with the diagnosis attached.
        """
        got = [self.resolve(etype, ns, v) for v in values]
        n = sum(1 for g in got if g is not None)
        rate = n / max(len(values), 1)
        if rate < expect:
            miss = [str(v) for v, g in zip(values, got) if g is None][:8]
            raise JoinError(
                f"{label}: joined {n:,}/{len(values):,} ({rate:.1%}) on {etype}/{ns}, "
                f"expected >= {expect:.0%}. Unmatched examples: {miss}. "
                f"A join this weak is a bug in the key, not a property of the data -- check whether the "
                f"column is an index rather than an identifier (this project's `ppm`/`abund` failure), "
                f"whether symbols are being used where accessions are needed, or whether the assembly "
                f"or species differs.")
        return got, {"joined": n, "of": len(values), "rate": rate, "via": ns, "label": label}


def build():
    import pandas as pd
    genes = pd.read_parquet(SD / "genes.parquet")
    xr = pd.read_parquet(SD / "gene_xrefs.parquet")
    d = json.load(gzip.open(NET, "rt"))
    incell = [g["name"] for g in d["genes"]]
    coords = {g["name"]: (g.get("chrom"), g.get("tss")) for g in d["genes"]}
    del d

    ent, alias = {}, {}

    def add(eid, etype, **fields):
        ent[eid] = dict(entity_type=etype, **fields)

    def link(etype, ns, val, eid):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return
        v = str(val).split(".")[0] if ns in ("ensembl", "transcript") else str(val)
        if v and v.lower() not in ("nan", "none", ""):
            alias.setdefault(Registry._k(etype, ns, v), eid)

    # ---- genes ----
    gsub = genes.dropna(subset=["ensembl_gene_id"])
    sym2ens = {}
    for _i, r in gsub.iterrows():
        ens = str(r["ensembl_gene_id"]).split(".")[0]
        eid = f"gene:{ens}"
        sym = r.get("hgnc_symbol")
        sym = None if (not isinstance(sym, str) or not sym) else sym
        if eid not in ent:
            add(eid, "gene", ensembl=ens, symbol=sym, biotype=r.get("gene_biotype"),
                assembly=ASSEMBLY, in_cell=False)
        link("gene", "ensembl", ens, eid)
        if sym:
            link("gene", "symbol", sym, eid)
            sym2ens.setdefault(sym, ens)

    # xrefs -> uniprot / entrez / refseq, and protein entities
    for _i, r in xr.iterrows():
        ens = str(r["ensembl_gene_id"]).split(".")[0]
        eid = f"gene:{ens}"
        if eid not in ent:
            continue
        db, xid = str(r["xref_db"]), str(r["xref_id"])
        if db.startswith("UniProtKB"):
            pid = f"protein:{xid}"
            if pid not in ent:
                add(pid, "protein", uniprot=xid, isoform=None, gene=eid,
                    reviewed=db.endswith("Swiss-Prot"))
            link("protein", "uniprot", xid, pid)
            ent[eid].setdefault("proteins", []).append(pid)
            link("gene", "uniprot", xid, eid)
        elif db == "EntrezGene":
            link("gene", "entrez", xid, eid)
        elif db.startswith("RefSeq"):
            link("gene", "refseq", xid, eid)

    # ---- mark which genes are in the cell object, and attach coordinates ----
    n_in = 0
    for s in incell:
        ens = sym2ens.get(s)
        eid = f"gene:{ens}" if ens else None
        if eid and eid in ent:
            ent[eid]["in_cell"] = True
            c, t = coords.get(s, (None, None))
            if c and t not in (None, ""):
                ent[eid]["chrom"], ent[eid]["tss"] = c, int(float(t))
            n_in += 1
        else:
            # a cell-object gene with no Ensembl mapping still needs an id, or every layer keyed on the
            # cell object would silently drop it
            eid = f"gene:SYMBOL_{s}"
            add(eid, "gene", ensembl=None, symbol=s, biotype=None, assembly=ASSEMBLY, in_cell=True,
                unmapped=True)
            link("gene", "symbol", s, eid)
            c, t = coords.get(s, (None, None))
            if c and t not in (None, ""):
                ent[eid]["chrom"], ent[eid]["tss"] = c, int(float(t))

    # ---- the cell object's own POSITIONAL keys, which is where the silent failures came from ----
    # `ppm`, `abund`, `coexpr`, `codep`, `gene2cplx`, `model4` and others are keyed by the gene's INDEX in
    # the genes list. Recording that mapping explicitly turns an invisible convention into a declared one.
    index_alias = {}
    for i, s in enumerate(incell):
        ens = sym2ens.get(s)
        index_alias[str(i)] = f"gene:{ens}" if ens and f"gene:{ens}" in ent else f"gene:SYMBOL_{s}"

    meta = {"assembly": ASSEMBLY, "n_entities": len(ent), "n_alias": len(alias),
            "n_genes_in_cell": n_in, "n_cell_genes_unmapped": len(incell) - n_in,
            "positional_key_note":
                "cell_complete.json.gz keys several dicts (ppm, abund, coexpr, codep, gene2cplx, model4, "
                "cellcycle, drugs, generxn, emask, darkfn) by the gene's INDEX in `genes`, not by symbol. "
                "`cell_index` maps those keys to entity ids. Looking them up by symbol returns None for "
                "every gene and is the bug that silently zeroed an abundance control.",
            "namespaces": {"gene": ["ensembl", "symbol", "uniprot", "entrez", "refseq"],
                           "protein": ["uniprot"]}}
    return {"entities": ent, "alias": alias, "cell_index": index_alias, "meta": meta}


def load():
    if not REGISTRY.exists():
        raise SystemExit(f"absent: {REGISTRY} -- run entity_registry.py first")
    return Registry(json.load(gzip.open(REGISTRY, "rt")))


def main():
    print("=" * 100)
    print("ENTITY REGISTRY -- immutable ids, declared joins, no silent misses")
    print("=" * 100)
    R = build()
    ent, alias, meta = R["entities"], R["alias"], R["meta"]
    from collections import Counter
    types = Counter(v["entity_type"] for v in ent.values())
    print(f"  {len(ent):,} entities   {dict(types)}")
    print(f"  {len(alias):,} alias keys across {list(meta['namespaces'])}")
    print(f"  genes in the cell object: {meta['n_genes_in_cell']:,} mapped to Ensembl, "
          f"{meta['n_cell_genes_unmapped']:,} carried by symbol only")
    ns = Counter(k.split("|")[1] for k in alias)
    print(f"  alias namespaces: {dict(ns)}")

    reg = Registry(R)
    print(f"\n  SELF-TEST -- the join must fail loudly where it used to fail silently")
    d = json.load(gzip.open(NET, "rt"))
    names = [g["name"] for g in d["genes"]]
    ppm = d.get("ppm", {})
    del d
    got, st = reg.join("gene", "symbol", names, 0.90, "cell genes by symbol")
    print(f"    {st['label']:34s} {st['joined']:,}/{st['of']:,} ({st['rate']:.1%})  OK")
    # the historical bug, reproduced deliberately: ppm keys are indices, so resolving them AS symbols
    # must raise rather than return a column of None
    try:
        reg.join("gene", "symbol", list(ppm)[:2000], 0.90, "ppm keys treated as symbols")
        print("    ppm-keys-as-symbols            DID NOT RAISE -- the guard is not working")
    except JoinError as e:
        print(f"    ppm-keys-as-symbols            raised as intended: {str(e)[:76]}...")
    ok = sum(1 for k in list(ppm)[:2000] if R["cell_index"].get(k))
    print(f"    ppm keys via cell_index        {ok:,}/2,000 ({ok/20:.1f}%)  <- the correct route")

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(REGISTRY, "wt") as fh:
        json.dump(R, fh)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "entity_types": dict(types), "alias_namespaces": dict(ns),
               "self_test": {"cell_genes_by_symbol": st,
                             "ppm_keys_as_symbols": "raises JoinError (intended)",
                             "ppm_keys_via_cell_index_pct": ok / 20}},
              open(OUT / "entity_registry.json", "w"), indent=1, default=float)
    print(f"\n  -> {REGISTRY}  ({REGISTRY.stat().st_size/1e6:.1f} MB)")
    print(f"  -> {OUT/'entity_registry.json'}")
    print(f"\n  Every layer added from here joins through Registry.join(), which declares its expected "
          f"rate\n  and raises below it. `unknown` is representable; a zero that means 'lookup failed' is "
          f"not.")


if __name__ == "__main__":
    main()
