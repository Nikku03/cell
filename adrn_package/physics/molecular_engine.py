"""molecular_engine — the sequence→structure→function substrate that turns a gene ID + a mutation into a
GENERALISING variant-effect call, instead of a static role-list lookup. This is where the four foundation tools
earn their place, and each does a distinct job the others can't:

  UniProt   — the SEQUENCE + per-residue functional-site annotation. The input: canonical protein, active/
              binding sites, domains. Everything downstream needs the sequence.
  ESM-2     — a protein LANGUAGE model. Zero-shot variant effect = masked-marginal log-likelihood-ratio
              logP(mut) - logP(wt) at the site. This GENERALISES: it scores a mutation the model has never
              seen, from evolutionary statistics — the extrapolation engine for "is this substitution
              functional?" (magnitude, not direction).
  AlphaFold — the 3D STRUCTURE. Gives burial/contact number and interface context, feeding ΔΔG (stability →
              destabilising = LOF) and pocket/interface hits. Direction evidence ESM can't give.
  Foldseek  — STRUCTURAL-homology search. Transfers function/annotation to a query by its FOLD when sequence
              homology is absent (dark proteome), and finds structural analogues of a mutated site.

Combined call (honest about what each signal can and can't say):
  functional_score  = ESM LLR (+ position conservation)         -> IS the mutation functional? (generalises)
  stability_ddg     = ΔΔG from AlphaFold structure               -> does it DESTABILISE? (LOF evidence)
  direction (GOF/LOF/neutral) = combine: destabilising|truncating|catalytic-site -> LOF; functional-but-
                      stability-neutral-at-recurrent-hotspot -> GOF-candidate; weak everywhere -> neutral.
  Direction stays the hard part — flagged with confidence, never fabricated.

-> caches per-variant in outputs/orphan/variant_effect_cache.json
"""
import os, sys, json, urllib.request, ssl, math
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/orphan"
_CA = "/root/.ccr/ca-bundle.crt"
_CACHE = f"{OUT}/variant_effect_cache.json"
_ESM = {"model": None, "alphabet": None, "bc": None}
_SEQ_CACHE = {}

# catalytic/functional keywords for the LOF-at-active-site signal
CATALYTIC_SITE = ("ACT_SITE", "BINDING", "METAL")


def _ctx():
    return ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None


def uniprot_seq(gene, acc=None):
    """canonical sequence for a gene (reviewed human), cached. Returns (accession, sequence) or (None, None)."""
    key = acc or gene
    if key in _SEQ_CACHE:
        return _SEQ_CACHE[key]
    try:
        if acc:
            url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        else:
            url = (f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene}+AND+organism_id:9606+AND+"
                   f"reviewed:true&format=fasta&size=1")
        data = urllib.request.urlopen(url, timeout=60, context=_ctx()).read().decode()
        if not data.startswith(">"):
            _SEQ_CACHE[key] = (None, None); return None, None
        lines = data.splitlines()
        acc2 = lines[0].split("|")[1] if "|" in lines[0] else (acc or gene)
        seq = "".join(l for l in lines[1:] if not l.startswith(">"))
        _SEQ_CACHE[key] = (acc2, seq)
        return acc2, seq
    except Exception:
        _SEQ_CACHE[key] = (None, None)
        return None, None


def _load_esm():
    if _ESM["model"] is None:
        import esm
        m, a = esm.pretrained.esm2_t12_35M_UR50D()
        m.eval()
        _ESM.update(model=m, alphabet=a, bc=a.get_batch_converter())
    return _ESM


def esm_scores(seq, pos, wt, mut, win=127):
    """zero-shot masked-marginal at `pos` (1-indexed residue). Windows long sequences around the site so it runs
    fast on CPU while keeping local+medium-range context. Returns (llr, entropy) —
      llr     = logP(mut) - logP(wt)  (negative = substitution disfavoured = more likely functional)
      entropy = Shannon entropy of the position's aa distribution (low = conserved = functional if mutated)."""
    import torch
    E = _load_esm(); alphabet = E["alphabet"]; model = E["model"]
    if pos < 1 or pos > len(seq) or seq[pos - 1] != wt:
        return None, None                                        # residue mismatch -> can't score honestly
    lo = max(0, pos - 1 - win); hi = min(len(seq), pos - 1 + win + 1)
    sub = seq[lo:hi]; site = (pos - 1 - lo)                       # 0-index of the site within the window
    toks = E["bc"]([("x", sub)])[2]
    tpos = site + 1                                              # +1 for BOS
    toks_m = toks.clone(); toks_m[0, tpos] = alphabet.mask_idx
    with torch.no_grad():
        logits = model(toks_m)["logits"]
    lp = torch.log_softmax(logits[0, tpos], dim=-1)
    try:
        llr = float(lp[alphabet.get_idx(mut)] - lp[alphabet.get_idx(wt)])
    except Exception:
        return None, None
    # entropy over the 20 aa
    aas = "ACDEFGHIKLMNPQRSTVWY"
    p = torch.tensor([lp[alphabet.get_idx(a)] for a in aas]).exp()
    p = p / p.sum()
    ent = float(-(p * (p + 1e-9).log()).sum())
    return round(llr, 3), round(ent, 3)


def _ddg(gene, acc, pos, wt, mut):
    """ΔΔG from the AlphaFold structure (destabilising = LOF evidence). None if unavailable."""
    try:
        import ddg_predictor as ddg
        P = ddg.DDGPredictor(f"{OUT}/ddg_model.pkl")
        pdb = P.alphafold_pdb(acc)
        d, _ = P.predict_from_structure(pdb, "A", pos, wt, mut)
        return round(d, 2)
    except Exception:
        return None


def variant_effect(gene, pos, wt, mut, acc=None, recurrent=False, at_catalytic=False):
    """the generalising call. Combines ESM (functional?), ΔΔG (destabilising→LOF?), and priors
    (recurrent hotspot→GOF; catalytic-site→LOF). Returns a dict with the call + confidence + evidence."""
    cache = {}
    if os.path.exists(_CACHE):
        try: cache = json.load(open(_CACHE))
        except Exception: cache = {}
    ck = f"{gene}:{wt}{pos}{mut}:{int(recurrent)}{int(at_catalytic)}"
    if ck in cache:
        return cache[ck]

    # truncating variants (frameshift / nonsense) destroy the protein -> LOF, no sequence model needed
    if mut in ("*", "fs", "fs*", "del", "X"):
        out = {"gene": gene, "variant": f"{wt}{pos}{mut}", "esm_llr": None, "esm_entropy": None, "ddg": None,
               "functional_score": 1.0, "call": "LOF", "confidence": "high",
               "evidence": ["truncating (frameshift/nonsense) -> protein destroyed"]}
        cache[ck] = out
        try: json.dump(cache, open(_CACHE, "w"))
        except Exception: pass
        return out

    acc2, seq = uniprot_seq(gene, acc)
    llr = ent = None
    if seq:
        llr, ent = esm_scores(seq, pos, wt, mut)
    ddg = _ddg(gene, acc2 or acc, pos, wt, mut) if (acc2 or acc) else None

    # functional magnitude: strong if ESM disfavours the substitution at a conserved position
    functional = 0.0
    if llr is not None:
        functional = max(0.0, -llr)                              # disfavoured -> functional
        if ent is not None and ent < 1.5:                        # conserved position amplifies
            functional *= 1.4
    # direction
    call, conf, ev = "uncertain", "low", []
    if llr is not None:
        ev.append(f"ESM LLR {llr}" + (f", entropy {ent}" if ent is not None else ""))
    if ddg is not None:
        ev.append(f"ΔΔG {ddg}")
    destabilising = (ddg is not None and ddg > 1.5)
    if destabilising or at_catalytic:
        call = "LOF"; conf = "medium" if functional > 0.3 else "low"
        ev.append("destabilising" if destabilising else "at catalytic site")
    elif recurrent and functional > 0.3 and not destabilising:
        call = "GOF-candidate"; conf = "medium"
        ev.append("recurrent hotspot, functional but stability-neutral")
    elif functional > 0.6:
        call = "functional (direction unresolved)"; conf = "medium"
    elif functional < 0.15 and (ddg is None or abs(ddg) < 0.6):
        call = "likely-neutral"; conf = "medium"; ev.append("ESM-tolerated, stability-neutral")
    out = {"gene": gene, "variant": f"{wt}{pos}{mut}", "esm_llr": llr, "esm_entropy": ent,
           "ddg": ddg, "functional_score": round(functional, 3), "call": call, "confidence": conf,
           "evidence": ev}
    cache[ck] = out
    try: json.dump(cache, open(_CACHE, "w"))
    except Exception: pass
    return out


if __name__ == "__main__":
    # smoke test: activating hotspots (BRAF/KRAS/JAK2) vs a classic destabilising LOF (TP53 R175H)
    tests = [("BRAF", 600, "V", "E", "P15056", True, False),
             ("KRAS", 12, "G", "D", "P01116", True, False),
             ("JAK2", 617, "V", "F", "O60674", True, False),
             ("TP53", 175, "R", "H", "P04637", False, False),
             ("VHL", 167, "R", "W", "P40337", False, False)]
    for g, p, w, m, a, rec, cat in tests:
        r = variant_effect(g, p, w, m, acc=a, recurrent=rec, at_catalytic=cat)
        print(f"  {g:6}{w}{p}{m}: call={r['call']:34} func={r['functional_score']:>5} "
              f"ddg={r['ddg']} llr={r['esm_llr']}  [{r['confidence']}]")
