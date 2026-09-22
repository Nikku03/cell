"""id_map — fold the claude-science reference tables into ONE canonical per-gene ID + metabolic-membership map, the
join backbone that lets every future dataset (kcat, Perturb-seq, CRISPR — the measurements still to come) plug in
by any identifier without the symbol-mismatch losses we hit before (KEGG symbols → cell genes lost enzymes).

Sources (Ensembl BioMart GRCh38.p14 + Human-GEM, retrieved 2026-07-13, from the user's data run):
  genes.parquet       — 86,411 Ensembl genes: hgnc_symbol, biotype, coords, description
  gene_xrefs.parquet  — 308,270 xrefs: Ensembl ↔ UniProt / RefSeq / EntrezGene
  model_genes.parquet — 2,848 Human-GEM genes: ensembl ↔ symbol ↔ uniprot ↔ entrez + num_reactions
  reactions.parquet   — 12,931 reactions: gpr_rule (ensembl) + subsystem + ec_number  → metabolic membership

Output: id_map.parquet, one row per gene in the cell, with ensembl / uniprot / entrez / biotype and (if metabolic)
its Human-GEM subsystems + reaction count. -> outputs/orphan/id_map.parquet (+ id_map_summary.json)
"""
import os, sys, json, re
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
SD = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan", "science_data")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def build():
    genes = pd.read_parquet(f"{SD}/genes.parquet")
    xr = pd.read_parquet(f"{SD}/gene_xrefs.parquet")
    mg = pd.read_parquet(f"{SD}/model_genes.parquet")
    rx = pd.read_parquet(f"{SD}/reactions.parquet")

    gsub = genes.dropna(subset=["hgnc_symbol"])          # drop rows TOGETHER so symbol↔ensembl stay aligned
    sym2ens = dict(zip(gsub["hgnc_symbol"], gsub["ensembl_gene_id"]))
    ens2sym = {v: k for k, v in sym2ens.items()}
    ens2biotype = dict(zip(genes["ensembl_gene_id"], genes["gene_biotype"]))
    ens2desc = dict(zip(genes["ensembl_gene_id"], genes["description"]))

    # best UniProt per Ensembl (Swiss-Prot reviewed preferred over TrEMBL), + Entrez
    sp = xr[xr["xref_db"] == "UniProtKB/Swiss-Prot"].groupby("ensembl_gene_id")["xref_id"].first()
    tr = xr[xr["xref_db"] == "UniProtKB/TrEMBL"].groupby("ensembl_gene_id")["xref_id"].first()
    ez = xr[xr["xref_db"] == "EntrezGene"].groupby("ensembl_gene_id")["xref_id"].first()
    ens2uni = {**tr.to_dict(), **sp.to_dict()}          # Swiss-Prot overrides TrEMBL
    ens2entrez = ez.to_dict()

    # metabolic: ensembl -> subsystems + reaction count, by parsing GPR rules
    ens2subsys, ens2nrxn = {}, {}
    for _, r in rx.iterrows():
        gpr, sub = r.get("gpr_rule"), r.get("subsystem")
        if not isinstance(gpr, str) or not sub:
            continue
        for e in set(re.findall(r"ENSG\d+", gpr)):
            ens2subsys.setdefault(e, set()).add(sub)
            ens2nrxn[e] = ens2nrxn.get(e, 0) + 1
    metab_ens = set(mg["ensembl_id"].str.extract(r"(ENSG\d+)")[0].dropna()) | set(ens2subsys)

    from complete_cell import CompleteCell
    C = CompleteCell()
    rows = []
    for i in range(len(C.genes)):
        sym = C.genes[i].get("name")
        ens = sym2ens.get(sym)
        subs = sorted(ens2subsys.get(ens, [])) if ens else []
        rows.append({
            "symbol": sym, "ensembl": ens, "uniprot": ens2uni.get(ens), "entrez": ens2entrez.get(ens),
            "biotype": ens2biotype.get(ens), "is_metabolic": bool(ens and ens in metab_ens),
            "n_metabolic_reactions": ens2nrxn.get(ens, 0),
            "metabolic_subsystems": ";".join(s for s in subs if s not in
                                              ("Transport reactions", "Exchange/demand reactions"))[:300],
        })
    idm = pd.DataFrame(rows)
    idm.to_parquet(f"{OUT}/id_map.parquet")
    summary = {
        "n_genes": len(idm), "with_ensembl": int(idm["ensembl"].notna().sum()),
        "with_uniprot": int(idm["uniprot"].notna().sum()), "with_entrez": int(idm["entrez"].notna().sum()),
        "metabolic": int(idm["is_metabolic"].sum()),
        "with_metabolic_subsystem": int((idm["metabolic_subsystems"].str.len() > 0).sum()),
        "n_subsystems": int(rx["subsystem"].nunique()),
        "source": "Ensembl BioMart GRCh38.p14 + Human-GEM (user data run, 2026-07-13)",
    }
    json.dump(summary, open(f"{OUT}/id_map_summary.json", "w"), indent=1)
    return idm, summary


def main():
    idm, s = build()
    print("=" * 78)
    print("ID MAP — canonical join backbone from the science reference tables")
    print("=" * 78)
    print(f"  genes: {s['n_genes']:,}")
    print(f"    ensembl {s['with_ensembl']:,} ({s['with_ensembl']/s['n_genes']:.0%})   "
          f"uniprot {s['with_uniprot']:,} ({s['with_uniprot']/s['n_genes']:.0%})   "
          f"entrez {s['with_entrez']:,} ({s['with_entrez']/s['n_genes']:.0%})")
    print(f"    metabolic {s['metabolic']:,}   with a Human-GEM subsystem {s['with_metabolic_subsystem']:,}  "
          f"({s['n_subsystems']} subsystems)")
    print("  examples:")
    for sym in ("PKM", "FASN", "CS", "TP53", "MAPK1"):
        r = idm[idm["symbol"] == sym]
        if len(r):
            r = r.iloc[0]
            print(f"    {sym:6} {r['ensembl']}  {r['uniprot']}  entrez={r['entrez']}  "
                  f"{'metab:'+r['metabolic_subsystems'][:50] if r['is_metabolic'] else r['biotype']}")
    print("=" * 78)
    return s


if __name__ == "__main__":
    main()
