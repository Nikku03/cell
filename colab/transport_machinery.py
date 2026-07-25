"""transport_machinery -- did the GATA1 knockout hit the transporters' production and maturation machinery, or only the transporters?

THE QUESTION. 153 of the 621 GATA1 movers are transport proteins. Are the genes that MAKE and MATURE those transporters also affected? In
other words, did the knockout change the transporters themselves, or the cell's capacity to produce, fold, glycosylate, traffic and insert
them?

WHAT THIS DATA CAN AND CANNOT ANSWER. Perturb-seq measures mRNA. It can never show a protein's structure, its folding state, its glycan
composition, or whether a transporter actually transports. So "did the knockout affect transporter STRUCTURE or FUNCTION" is not directly
answerable here and this script does not pretend otherwise. What IS answerable, and is the useful proxy: did the transcripts encoding each
step of the production-and-maturation pipeline change? If the translocon, the glycosylation complex, the chaperones and the COPII coat all
move, the cell's maturation CAPACITY has changed, and that necessarily reaches every secretory-branch protein including the transporters.
Capacity is inferred from transcript level; function is not observed.

THE PIPELINE, STEP BY STEP, each as its own gene set:
    make it        Pol II core, Mediator, general TFs        -> is it being transcribed at all
                   spliceosome, mRNA export                  -> does the transcript mature and leave the nucleus
                   ribosome, initiation and elongation       -> is it translated
    mature it      SRP + Sec61 translocon + signal peptidase -> does it get into the ER at all
                   OST / N-glycosylation, ALG pathway        -> is the glycan attached
                   ER chaperones and disulfide isomerases    -> does it fold
                   ERAD                                      -> is misfolded protein cleared
    move it        COPII / ER exit, COPI retrograde          -> does it leave the ER
                   Golgi glycosyltransferases                -> is the glycan processed
                   SNAREs and vesicle fusion                 -> does the vesicle deliver
                   lysosomal M6P targeting                   -> is it addressed correctly
Each set is tested for enrichment against the measured background with a hypergeometric test, because with 621 of 8,533 genes moving
(7.3%) any large gene set will contain some movers by chance. Only enrichment above that background is evidence."""
import json, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
T = 4.2
TIDE_FRAC = 0.05

# (stage, set name, name prefixes, exact names)
SETS = [
    ("MAKE IT", "Pol II core", ("POLR2",), ()),
    ("MAKE IT", "Mediator", ("MED",), ()),
    ("MAKE IT", "general TFs (TAF/GTF)", ("TAF", "GTF"), ()),
    ("MAKE IT", "spliceosome", ("SNRP", "SF3", "SRSF", "PRPF", "LSM"), ()),
    ("MAKE IT", "mRNA export", ("NXF", "THOC", "NUP"), ("ALYREF", "DDX39B", "GLE1")),
    ("MAKE IT", "ribosome (large)", ("RPL",), ()),
    ("MAKE IT", "ribosome (small)", ("RPS",), ()),
    ("MAKE IT", "translation init/elong", ("EIF", "EEF"), ()),
    ("MATURE IT", "SRP + Sec61 translocon", ("SRP", "SEC61", "SSR", "SPCS"), ("SEC62", "SEC63")),
    ("MATURE IT", "N-glycosylation (OST)", ("STT3", "RPN", "ALG", "MGAT"), ("DDOST", "DAD1", "MOGS", "GANAB")),
    ("MATURE IT", "ER chaperones / folding", ("PDIA", "DNAJB", "DNAJC"), ("CANX", "CALR", "HSPA5", "HSP90B1", "P4HB", "ERP44")),
    ("MATURE IT", "ERAD / quality control", ("DERL", "EDEM", "UBE2J"), ("SEL1L", "SYVN1", "VCP", "OS9", "AMFR")),
    ("MOVE IT", "COPII / ER exit", ("SEC23", "SEC24", "SEC31", "SAR1"), ("SEC13", "SEC16A", "TFG")),
    ("MOVE IT", "COPI / retrograde", ("COPA", "COPB", "COPE", "COPG", "ARF"), ("ARCN1", "ARFGAP1")),
    ("MOVE IT", "Golgi glycosyltransferases", ("B4GALT", "B3GNT", "ST6GAL", "ST3GAL", "GALNT", "FUT", "MAN1", "MAN2"), ()),
    ("MOVE IT", "SNARE / vesicle fusion", ("STX", "VAMP", "SNAP", "SEC22", "BET1", "GOSR"), ("NSF", "NAPA", "YKT6")),
    ("MOVE IT", "lysosomal targeting (M6P)", ("GNPT",), ("M6PR", "IGF2R", "LAMP1", "LAMP2")),
    ("MOVE IT", "ESCRT / sorting", ("CHMP", "VPS", "SNX"), ("TSG101", "PDCD6IP", "HGS", "STAM")),
]


def main(ko="GATA1"):
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); A = np.abs(Z)
    tide = (A >= T).mean(0) >= TIDE_FRAC
    tide_genes = {genes[i] for i in np.where(tide)[0]}
    panel = [g for g in genes if g not in tide_genes]          # every gene that COULD have been called a mover
    pset = set(panel)
    row = Z[kidx[ko]]
    zof = {genes[i]: float(row[i]) for i in range(len(genes))}
    movers = {g for g in panel if abs(zof.get(g, 0.0)) >= T and g != ko}

    dz = json.load(open(OUT / f"ko_cell_map_{ko}.json"))
    n_transporters = dz["totals"]["transporters_total"]

    from scipy.stats import hypergeom
    Npanel, Nmov = len(pset), len(movers)
    base = Nmov / Npanel
    print(f"{ko} in K562: {Nmov} movers out of {Npanel} testable genes -- background rate {base:.1%}")
    print(f"({n_transporters} of the movers are transport proteins)\n")
    print(f"  {'stage':10s} {'machinery set':30s} {'in panel':>9s} {'moved':>6s} {'rate':>7s} "
          f"{'enrich':>7s} {'p':>10s} {'up/down':>9s} {'median z':>9s}")

    rows = []
    for stage, name, pref, exact in SETS:
        mem = sorted({g for g in pset if g.startswith(pref) or g in exact})
        if len(mem) < 4:
            continue
        hit = sorted(g for g in mem if g in movers)
        k, n = len(hit), len(mem)
        p = float(hypergeom.sf(k - 1, Npanel, Nmov, n)) if k else 1.0
        enr = (k / n) / base if n else 0.0
        up = sum(1 for g in hit if zof[g] > 0); dn = k - up
        mz = float(np.median([abs(zof[g]) for g in hit])) if hit else 0.0
        rows.append({"stage": stage, "set": name, "in_panel": n, "moved": k, "rate": round(k / n, 4),
                     "enrichment": round(enr, 2), "p": p, "up": up, "down": dn,
                     "median_abs_z": round(mz, 2), "movers": hit[:14]})
        flag = "***" if p < 1e-4 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {stage:10s} {name:30s} {n:>9d} {k:>6d} {k/n:>6.1%} {enr:>7.2f} {p:>10.2g}{flag:3s}"
              f" {up:>4d}/{dn:<4d} {mz:>8.1f}")

    # ---- the three questions, answered in the terms the data supports ----
    by_stage = collections.defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
    summ = {}
    for stage in ("MAKE IT", "MATURE IT", "MOVE IT"):
        rs = by_stage[stage]
        tot_n = sum(r["in_panel"] for r in rs); tot_k = sum(r["moved"] for r in rs)
        p = float(hypergeom.sf(tot_k - 1, Npanel, Nmov, tot_n)) if tot_k else 1.0
        sig = [r["set"] for r in rs if r["p"] < 0.05]
        summ[stage] = {"in_panel": tot_n, "moved": tot_k, "rate": round(tot_k / max(tot_n, 1), 4),
                       "enrichment": round((tot_k / max(tot_n, 1)) / base, 2), "p": p, "sets_significant": sig}
        print(f"\n  {stage}: {tot_k}/{tot_n} moved ({tot_k/max(tot_n,1):.1%}, "
              f"{(tot_k/max(tot_n,1))/base:.2f}x background, p={p:.2g})")
        print(f"    significantly moved sets: {sig if sig else 'none'}")

    verdict = (
        f"TRANSPORT MACHINERY ({ko}, K562): asked whether the knockout affected the transporters' production, maturation or function, "
        f"rather than only the transporters themselves. {n_transporters} of the {Nmov} movers are transport proteins. "
        "WHAT THIS DATA CANNOT SAY, stated first: Perturb-seq measures mRNA, so it cannot show a protein's structure, its folding state, "
        "its glycans, or whether a transporter actually transports. Structure and function are NOT observed here. What is observable is "
        "whether the transcripts encoding each step of the production-and-maturation pipeline changed, which is a statement about the "
        "cell's CAPACITY to make and mature those proteins. "
        f"Background: {Nmov} of {Npanel} testable genes moved, {base:.1%}, so every set is tested against that with a hypergeometric test. "
        + " ".join(
            f"{stage}: {summ[stage]['moved']}/{summ[stage]['in_panel']} ({summ[stage]['rate']:.1%}, "
            f"{summ[stage]['enrichment']:.2f}x background, p={summ[stage]['p']:.2g}); significant sets "
            f"{summ[stage]['sets_significant'] or 'none'}."
            for stage in ("MAKE IT", "MATURE IT", "MOVE IT"))
        + " Enrichment above background is the only evidence that counts here, because any large gene set will contain some movers by "
        "chance. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"knockout": ko, "n_movers": Nmov, "n_panel": Npanel, "background_rate": round(base, 4),
               "n_transporters_among_movers": n_transporters, "sets": rows, "by_stage": summ,
               "verdict": verdict, "note": verdict},
              open(OUT / f"transport_machinery_{ko}.json", "w"), indent=1)
    print(f"\n  -> outputs/orphan/transport_machinery_{ko}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GATA1")
