"""What each nexus-catalyst feature block actually costs, measured on this machine rather than quoted.

Three stages, each timed with the arm's own parameters:
    ESM-2 35M embedding      nexus_catalyst_esm.embed  (batch 8, length-sorted, mean over residues)
    AlphaFold parse + trim   nexus_catalyst_pilot.load_struct (pLDDT >= 70)
    rigid-body docking       nexus_catalyst_pilot.dock_pair   (GRID 128, SPACING 1.7, 300 rotations)
and, for the clash block specifically, the same docking WITHOUT the core-erosion transform -- because
`clash` is the only docking feature that needs that second correlation, so it is the one docking block whose
removal has its own price.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
N_EMB_SAMPLE = 24
N_DOCK_ROT = 300
N_DOCK_ROT_SAMPLE = 30       # time a fraction of the rotations and scale -- the loop is homogeneous
AA = "ACDEFGHIKLMNPQRSTVWY"


def _time_embedding(say, seqs):
    import torch
    import esm
    torch.set_num_threads(4)
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    bc = alphabet.get_batch_converter()
    model.eval()
    layer = model.num_layers
    names = sorted(seqs, key=lambda g: len(seqs[g]))
    step = max(1, len(names) // N_EMB_SAMPLE)
    sample = names[::step][:N_EMB_SAMPLE]
    nres = sum(min(len(seqs[g]), 1022) for g in sample)
    order = sorted(sample, key=lambda g: len(seqs[g]))
    t = time.time()
    for k in range(0, len(order), 8):
        chunk = order[k:k + 8]
        _, _, toks = bc([(g, seqs[g][:1022]) for g in chunk])
        with torch.no_grad():
            model(toks, repr_layers=[layer])["representations"][layer]
    el = time.time() - t
    say(f"     ESM-2 35M forward pass, batch 8, 4 threads: {len(sample)} proteins "
        f"({nres:,} residues) in {el:.1f}s")
    say(f"       -> {el / len(sample):.3f} s/protein, {1000 * el / nres:.3f} ms/residue")
    return {"n": len(sample), "seconds": el, "s_per_protein": el / len(sample),
            "residues": nres, "ms_per_residue": 1000 * el / nres,
            "sampled_lengths": [len(seqs[g]) for g in sample]}


def _time_docking(say, dock):
    import dock_fft as DF
    import nexus_catalyst_pilot as P
    DF.GRID, DF.SPACING = P.GRID, P.SPACING
    g2u, _glen = P._acc_map()
    bench = json.load(open(P.BENCH))
    picks = [e for e in bench["bench"] if str(e["rx"]) in dock][:3]
    if not picks:
        return None
    b0 = picks[0]
    sub = P.load_struct(g2u[b0["substrate"]])
    cand = P.load_struct(g2u[b0["catalyst"]])
    if sub is None or cand is None:
        return None

    t = time.time()
    for _ in range(3):
        P.load_struct(g2u[b0["catalyst"]])
    t_load = (time.time() - t) / 3
    afdir = P.AF
    nfiles = len(list(afdir.glob("*.pdb")))
    nbytes = sum(f.stat().st_size for f in afdir.glob("*.pdb"))
    say(f"     AlphaFold parse + pLDDT>=70 trim, CACHED file: {t_load:.3f} s/structure "
        f"({cand['n']:,} atoms kept)")
    say(f"       the cache itself: {nfiles:,} AlphaFold monomers, {nbytes / 1e9:.2f} GB, one HTTPS "
        f"round trip each -- network-bound, not counted in the CPU figures below")

    rots = [np.asarray(r, float) for r in DF.rotations(N_DOCK_ROT_SAMPLE)]
    centre = sub["co"].mean(0)
    t = time.time()
    solid_R = DF.rasterise(sub["co"], centre, DF.offsets_for(sub["rad"]))
    skin_R = DF._morph(solid_R, DF.SKIN, True) & ~solid_R
    core_R = DF._morph(solid_R, DF.CORE_ERODE, False)
    F_skin = np.fft.rfftn(skin_R.astype(np.float32))
    F_core = np.fft.rfftn(core_R.astype(np.float32))
    t_setup = time.time() - t

    Lc = cand["co"] - cand["co"].mean(0)
    offs_L = DF.offsets_for(cand["rad"])

    def loop(with_clash):
        t = time.time()
        for R_ in rots:
            solid_L = DF.rasterise(Lc @ R_.T + centre, centre, offs_L)
            if solid_L is None:
                continue
            FL = np.conj(np.fft.rfftn(solid_L.astype(np.float32)))
            inter = np.fft.irfftn(F_skin * FL, solid_R.shape)
            if with_clash:
                clash = np.fft.irfftn(F_core * FL, solid_R.shape)
                sc = inter - 1.0 * clash
                j = int(np.argmax(sc))
                float(clash.reshape(-1)[j])
            else:
                sc = inter
                j = int(np.argmax(sc))
            float(sc.reshape(-1)[j])
            float(np.partition(sc.reshape(-1), -50)[-50:].mean())
        return time.time() - t

    loop(True)                       # warm the FFT plans
    fulls, nocls = [], []
    for e in picks:
        c = P.load_struct(g2u[e["catalyst"]])
        if c is None:
            continue
        Lc = c["co"] - c["co"].mean(0)
        offs_L = DF.offsets_for(c["rad"])
        fulls.append(loop(True) / N_DOCK_ROT_SAMPLE * N_DOCK_ROT)
        nocls.append(loop(False) / N_DOCK_ROT_SAMPLE * N_DOCK_ROT)
    t_full = float(np.mean(fulls))
    t_noclash = float(np.mean(nocls))
    say(f"     dock_pair at GRID {P.GRID}, SPACING {P.SPACING}, {N_DOCK_ROT} rotations "
        f"({len(fulls)} candidates timed on {N_DOCK_ROT_SAMPLE} rotations each and scaled):")
    say(f"       receptor setup (rasterise + skin/core morph + 2 rFFTs) {t_setup:.2f} s/substrate")
    say(f"       rotation loop WITH the core-erosion (clash) transform : {t_full:.1f} s/candidate")
    say(f"       rotation loop WITHOUT it                              : {t_noclash:.1f} s/candidate")
    say(f"       -> the clash block alone costs {t_full - t_noclash:.1f} s/candidate "
        f"({100 * (t_full - t_noclash) / t_full:.0f}% of the dock)")
    return {"s_load_struct": t_load, "atoms": int(cand["n"]),
            "af_cached_files": nfiles, "af_cached_bytes": nbytes,
            "s_per_candidate_samples": fulls,
            "s_receptor_setup": t_setup, "s_per_candidate_full": t_full,
            "s_per_candidate_noclash": t_noclash,
            "s_per_candidate_clash_only": t_full - t_noclash,
            "grid": P.GRID, "spacing": P.SPACING, "n_rot": N_DOCK_ROT}


def _time_cheap(say, seqs):
    names = sorted(seqs)[:2000]
    t = time.time()
    for g in names:
        s = seqs[g]
        [s.count(a) / max(len(s), 1) for a in AA]
        np.log(max(len(s), 1))
    el = time.time() - t
    say(f"     aa composition + log length: {1e6 * el / len(names):.1f} us/protein "
        f"({len(names)} proteins in {el:.3f}s)")
    return {"us_per_protein": 1e6 * el / len(names), "n": len(names)}


def measure(say, B, SB):
    seqs = B["seq"]
    rx, cats, dock = B["rx"], B["cats"], B["dock"]

    cand_genes = sorted({r["cat"] for r in rx})
    sub_genes = sorted({r["sub"] for r in rx})
    both = set(cand_genes) & set(sub_genes)
    sub_only = set(sub_genes) - set(cand_genes)
    say(f"     the benchmark's embedding bill: {len(cand_genes):,} distinct catalysts, "
        f"{len(sub_genes):,} distinct substrates, {len(both):,} in both roles")
    say(f"       -> {len(sub_only):,} proteins are embedded ONLY because the substrate side needs them")

    emb = _time_embedding(say, {g: seqs[g] for g in seqs})
    dk = _time_docking(say, dock)
    cheap = _time_cheap(say, seqs)

    ndock_rx = len(dock)
    ncand = sum(len(v["feats"]) for v in dock.values())
    say()
    say("     WHAT DROPPING EACH BLOCK SAVES, at this benchmark's size")
    sp = emb["s_per_protein"]
    rows = []

    def row(name, secs, note):
        rows.append({"block": name, "seconds_saved": secs, "note": note})
        say(f"       {name:<34}{secs / 60:>9.1f} min   {note}")

    row("esm_sub + esm_prod + esm_absdiff",
        len(sub_only) * sp,
        f"{len(sub_only):,} substrate-only proteins never embedded "
        f"({sp:.3f} s each)")
    row("all four ESM blocks",
        (len(set(cand_genes) | set(sub_genes))) * sp,
        f"{len(set(cand_genes) | set(sub_genes)):,} proteins never embedded at all")
    if dk:
        full_dock = ncand * dk["s_per_candidate_full"] + ndock_rx * dk["s_receptor_setup"]
        row("all three docking blocks + size",
            full_dock + (ncand + ndock_rx) * dk["s_load_struct"],
            f"{ncand} candidate docks + {ndock_rx} receptor setups + "
            f"{ncand + ndock_rx} structure parses never run")
        row("dock_clash only",
            ncand * dk["s_per_candidate_clash_only"],
            f"{ncand} candidates lose the core-erosion rFFT/irFFT pair")
        row("size only",
            0.0,
            "n_atoms and diam fall out of the structure parse the docking already needs")
    row("enz_seq + sub_seq",
        len(seqs) * cheap["us_per_protein"] / 1e6,
        f"{len(seqs):,} aa-composition vectors never built -- the cheapest block in the arm")

    say()
    say("     AND THE SCALE THAT MATTERS: the orphan task is 9,186 unannotated reactions against a")
    say(f"     {len(set(c for i in cats for c in cats[i])):,}-gene catalyst vocabulary.")
    voc = len({c for i in cats for c in cats[i]} & set(seqs))
    say(f"       embedding that vocabulary once           {voc * sp / 60:,.1f} min")
    if dk:
        say(f"       docking one 10-candidate shortlist       "
            f"{(10 * dk['s_per_candidate_full'] + dk['s_receptor_setup']) / 60:,.1f} min")
        say(f"       docking 9,186 shortlists of ten          "
            f"{9186 * (10 * dk['s_per_candidate_full'] + dk['s_receptor_setup']) / 3600:,.0f} h")
        say(f"       docking the FULL vocabulary once per orphan reaction "
            f"{9186 * voc * dk['s_per_candidate_full'] / 3600 / 24 / 365:,.0f} core-years")
    return {"embedding": emb, "docking": dk, "cheap": cheap,
            "n_catalysts": len(cand_genes), "n_substrates": len(sub_genes),
            "n_substrate_only": len(sub_only), "n_docked_reactions": ndock_rx,
            "n_docked_candidates": ncand, "vocabulary": voc, "savings": rows}
