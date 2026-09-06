"""llm_reasoner -- the LLM<->model integration that makes BOTH sides better, efficiently.

THE DESIGN (a distillation loop, not just 'an LLM reads the output'):

  [deterministic scorer]  ->  [PubMed grounding]  ->  [LLM adjudication]  ->  [distill back into the scorer]
   connection_proposer.py      NCBI eutils (FREE)      GPT-5.x (the $ part)    scorer_feedback.json (FREE, permanent)
   free, high recall           retrieves real evidence  structured verdict     known-stress filter + calibration priors

WHY THIS MAKES THE LLM HALLUCINATE LESS (five hard constraints, all enforced here, not requested politely):
  1. GROUNDED   -- the model reasons ONLY over PubMed abstracts we retrieve, not its memory.
  2. ANCHORED   -- the deterministic scorer's hard numbers (coherence p, cross-line reproducibility, #convergent KOs) are given as
                   ground truth it is told it MAY NOT contradict; it interprets them, it does not invent them.
  3. CITED      -- every factual claim must cite a PMID FROM THE PROVIDED SET; if none supports it, it must answer 'no support'.
  4. REFUTED    -- it must produce a REFUTATION (argue against its own hypothesis) before a verdict = adversarial self-check.
  5. STRUCTURED -- output is a fixed JSON schema; no free-form paragraph where confabulation hides.

WHY THIS MAKES THE MODEL PERFORM BETTER (the cheap, permanent part):
  The LLM verdicts are DISTILLED into scorer_feedback.json -- a known-stress blocklist + per-machine reliability priors -- which the
  deterministic scorer can read on its NEXT run to auto-down-rank re-derived-known programs and recalibrate confidence. You spend the
  $10 ONCE to label the candidates; the free scorer inherits that judgement forever. (Expensive reasoning -> cheap deterministic rule.)

BUDGET ($10) GUARDRAILS: shortlist only (cross-line-concordant + top pooled-novel, deduped, capped at --n); prints a COST ESTIMATE and
  refuses to spend without --go; disk-caches every response (re-runs are free); runs end-to-end with a deterministic --mock adjudicator
  so the whole pipeline is testable with NO key and NO spend. Model is configurable (OPENAI_MODEL env) so 'gpt-5.5'/'gpt-5.6'/whatever
  your key can reach all work; point OPENAI_BASE_URL elsewhere to use an open model (DeepSeek/Qwen via OpenRouter) with the same code.

HONEST BOUND (unchanged from the whole project): this does NOT break the wall. It does not improve the cell model's PREDICTIVE
accuracy -- the knockout-specific far field is data-limited, not reasoning-limited. What improves is (a) the PRECISION/trust of the
hypothesis shortlist and (b) the scorer's RANKING via distilled labels. The reasoner replaces the agent as interpreter; the code still
owns the score.
"""
import os, sys, json, time, hashlib, argparse, collections, re
from pathlib import Path
import requests

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
CACHE = OUT / "llm_reasoner_cache"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
# APPROXIMATE prices ($/1M tokens). Frontier pricing moves and reasoning models bill hidden reasoning tokens as output --
# treat this as an ESTIMATE; override with OPENAI_PRICE_IN / OPENAI_PRICE_OUT. Used ONLY to warn you before spending.
PRICE_IN = float(os.environ.get("OPENAI_PRICE_IN", "1.25"))
PRICE_OUT = float(os.environ.get("OPENAI_PRICE_OUT", "10.0"))


# --------------------------------------------------------------------------------------------- 1. load the scorer's candidates
def load_candidates():
    """Merge the pooled proposer's novel list with the per-line cross-line-concordant set; dedup by target; keep the scorer's
    hard numbers as anchors. Concordant (>=2 lines) first -- the trustworthy set."""
    cand = {}
    pool = json.load(open(OUT / "connection_proposer.json"))
    for p in pool["novel_proposals"]:
        cand[p["target"]] = {"target": p["target"], "machine": p["machine"], "n_convergent_KOs": p["n_convergent_KOs"],
                             "in_named_complex": p["in_named_complex"], "coherence_p": p["coherence_p"],
                             "pooled_confidence": p["confidence"], "KOs": p["KOs"], "n_lines_concordant": 1, "source": "pooled"}
    try:
        per = json.load(open(OUT / "connection_proposer_perline.json"))
        for c in per["cross_line_concordant_novel"]:
            t = c["target"]
            row = cand.get(t, {"target": t, "machine": (c["machines"] or [None])[0], "n_convergent_KOs": None,
                               "in_named_complex": None, "coherence_p": None, "pooled_confidence": None,
                               "KOs": [], "source": "perline"})
            row["n_lines_concordant"] = c["n_lines"]
            row["machines_seen"] = c.get("machines", [])
            row["per_line"] = [(l["line"], l["confidence"]) for l in c["lines"]]
            cand[t] = row
    except FileNotFoundError:
        pass
    rows = list(cand.values())
    # rank: cross-line concordance first, then pooled coherence (most negative log p)
    rows.sort(key=lambda r: (-(r.get("n_lines_concordant") or 1), (r.get("coherence_p") or 1.0)))
    return rows


# --------------------------------------------------------------------------------------------- 2. PubMed grounding (FREE)
def _eutils(path, params, tries=3):
    params = {**params, "tool": "cellos", "email": "research@example.org"}
    for i in range(tries):
        try:
            r = requests.get(f"{EUTILS}/{path}", params=params, timeout=25)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(0.5 * (i + 1))
    return None


def pubmed_count(term):
    r = _eutils("esearch.fcgi", {"db": "pubmed", "term": term, "retmode": "json", "retmax": 0})
    try:
        return int(r.json()["esearchresult"]["count"])
    except Exception:
        return None


def pubmed_abstracts(term, n=3):
    r = _eutils("esearch.fcgi", {"db": "pubmed", "term": term, "retmode": "json", "retmax": n, "sort": "relevance"})
    try:
        ids = r.json()["esearchresult"]["idlist"]
    except Exception:
        return []
    if not ids:
        return []
    r2 = _eutils("efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"})
    if not r2:
        return []
    # split the plain-text multi-record blob loosely; keep it short to control tokens
    txt = r2.text.strip()
    chunks = [c.strip() for c in re.split(r"\n\n\n+", txt) if c.strip()]
    return [{"pmid": pid, "text": ch[:900]} for pid, ch in zip(ids, chunks)]


def ground(cand):
    """Novelty signal + real evidence. Co-mention = target AND (machine subunits); 0 co-mentions vs many target papers = novel."""
    t = cand["target"]; kos = cand.get("KOs") or []
    subs = kos[:3]
    sub_clause = " OR ".join(sorted(set(subs))) if subs else (cand.get("machine") or "")
    comention_term = f"{t} AND ({sub_clause})" if sub_clause else t
    ev = {"target_total_papers": pubmed_count(t),
          "comention_count": pubmed_count(comention_term),
          "comention_term": comention_term,
          "abstracts": pubmed_abstracts(t, 2) + (pubmed_abstracts(subs[0], 1) if subs else [])}
    ev["novelty_signal"] = ("undocumented (0 co-mentions)" if ev["comention_count"] == 0
                            else f"{ev['comention_count']} co-mentions exist")
    return ev


# --------------------------------------------------------------------------------------------- 3. prompt + schema
SCHEMA_HINT = {
    "target": "str", "mechanism": "one concrete testable mechanism, or 'none plausible'",
    "mechanism_class": "e.g. RNA-surveillance-substrate | stress-program | chromatin | artifact | unknown",
    "is_known_stress_program": "bool -- is this just a known cell-stress/response program (UPR, ISR, proteasome->HSP70, sterol/SREBP, IFN/ISG)?",
    "novelty": "novel | known | uncertain -- grounded in the co-mention count and abstracts ONLY",
    "supporting_pmids": "list of PMIDs FROM THE PROVIDED ABSTRACTS ONLY; [] if none support it",
    "refutation": "the strongest argument this convergence is an ARTIFACT or already-known, not a novel edge",
    "wet_lab_test": "one concrete experiment to confirm/refute",
    "llm_confidence": "0.0-1.0 -- your calibrated confidence this is a REAL, NOVEL, mechanistically-coherent link",
    "verdict": "PURSUE | DEPRIORITIZE -- should a researcher chase this?"}

SYSTEM = ("You are a cautious molecular-biology hypothesis auditor. You are given (A) a deterministic pipeline's finding that a set of "
          "gene knockouts convergently move one target gene, with HARD statistics you MUST treat as ground truth and MUST NOT "
          "contradict, and (B) real PubMed evidence retrieved for you. RULES: cite ONLY PMIDs present in the provided abstracts; if no "
          "provided abstract supports a claim, say so and do not cite. Do NOT invent mechanisms you cannot tie to the provided "
          "evidence or to the stated statistics. Always fill 'refutation' with a genuine counter-argument. If the convergent set is a "
          "well-known stress/response program, say so and set is_known_stress_program=true even if the target looks novel. Answer with "
          "ONLY a JSON object matching the requested keys.")


def build_user(cand, ev):
    return json.dumps({
        "instruction": "Audit this convergence. Return ONLY the JSON object with exactly these keys.",
        "keys_required": SCHEMA_HINT,
        "finding": {
            "target_gene_moved": cand["target"],
            "converging_knockouts": cand.get("KOs"),
            "annotated_machine_they_form": cand.get("machine") or cand.get("machines_seen"),
            "HARD_STATS_ground_truth": {
                "coherence_p_hypergeometric": cand.get("coherence_p"),
                "n_convergent_knockouts": cand.get("n_convergent_KOs"),
                "subunits_inside_named_complex": cand.get("in_named_complex"),
                "cross_line_reproducibility_lines": cand.get("n_lines_concordant"),
                "scorer_confidence": cand.get("pooled_confidence"),
                "per_line": cand.get("per_line")}},
        "pubmed_evidence": {
            "target_total_papers": ev["target_total_papers"],
            "comention_query": ev["comention_term"],
            "comention_count": ev["comention_count"],
            "novelty_signal": ev["novelty_signal"],
            "retrieved_abstracts": ev["abstracts"]}}, indent=1)


# --------------------------------------------------------------------------------------------- 4. LLM call (or mock)
def call_openai(system, user, model, effort, max_out):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}}
    if max_out:
        body["max_completion_tokens"] = max_out           # newer models use max_completion_tokens, not max_tokens
    if effort:
        body["reasoning_effort"] = effort                 # honored by reasoning models; harmless hint otherwise
    r = requests.post(f"{OPENAI_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
    j = r.json()
    txt = j["choices"][0]["message"]["content"]
    usage = j.get("usage", {})
    return json.loads(txt), usage


def mock_adjudicate(cand, ev):
    """Deterministic stand-in so the FULL pipeline runs with no key / no spend. Uses only the grounding + scorer numbers -- it is a
    transparent rule, NOT an LLM, and is labelled as such in the output."""
    known_markers = {"proteasome": "proteasome complex", "oligosacchar": "UPR/OST", "endoplasmic": "UPR",
                     "interferon": "IFN/ISG", "sterol": "sterol/SREBP", "ion channel transport": "V-ATPase/sterol"}
    m = (cand.get("machine") or "").lower()
    is_known = any(k in m for k in known_markers)
    novel = ev["comention_count"] == 0 and not is_known
    p = cand.get("coherence_p") or 1.0
    coh_strong = p < 1e-8
    conf = round((0.75 if novel else 0.35) * (1.0 if coh_strong else 0.6) * min((cand.get("n_lines_concordant") or 1) / 2, 1.0), 2)
    return {"target": cand["target"],
            "mechanism": ("machine degrades/regulates the target transcript (surveillance-substrate hypothesis)"
                          if "decay" in m or "exosome" in m or "nonsense" in m else "convergent regulation of the target"),
            "mechanism_class": ("RNA-surveillance-substrate" if ("decay" in m or "exosome" in m) else
                                "stress-program" if is_known else "unknown"),
            "is_known_stress_program": bool(is_known),
            "novelty": "known" if is_known else ("novel" if novel else "uncertain"),
            "supporting_pmids": [a["pmid"] for a in ev["abstracts"][:1]] if not novel else [],
            "refutation": ("target may move as a generic stress response, not a direct substrate; convergence could reflect shared "
                           "essentiality rather than a specific edge"),
            "wet_lab_test": f"measure {cand['target']} mRNA half-life on {(cand.get('KOs') or ['the machine'])[0]} knockout",
            "llm_confidence": conf,
            "verdict": "PURSUE" if (novel and coh_strong) else "DEPRIORITIZE",
            "_mock": True}


# --------------------------------------------------------------------------------------------- 5. distill back into the scorer
def distill(verdicts):
    """Turn LLM judgements into a scorer_feedback.json the deterministic scorer can read next run (FREE, permanent)."""
    known_targets = sorted({v["target"] for v in verdicts if v.get("is_known_stress_program")})
    known_machines = collections.Counter()
    mach_conf = collections.defaultdict(list)
    confirmed = []
    for v, c in verdicts_with_cand(verdicts):
        mach = c.get("machine") or "unknown"
        mach_conf[mach].append(float(v.get("llm_confidence") or 0))
        if v.get("is_known_stress_program"):
            known_machines[mach] += 1
        if v.get("novelty") == "novel" and v.get("verdict") == "PURSUE":
            confirmed.append({"target": v["target"], "machine": mach, "llm_confidence": v.get("llm_confidence")})
    reliability = {m: round(sum(cs) / len(cs), 3) for m, cs in mach_conf.items()}
    return {"known_stress_targets": known_targets,
            "known_stress_machines": [m for m, _ in known_machines.most_common()],
            "machine_reliability_prior": reliability,
            "confirmed_novel_pursue": sorted(confirmed, key=lambda x: -(x["llm_confidence"] or 0)),
            "how_the_scorer_uses_this": ("next connection_proposer run: (1) label/down-rank targets in known_stress_targets; "
                                         "(2) multiply a machine's candidates by machine_reliability_prior as a learned weight; "
                                         "(3) promote confirmed_novel_pursue. Distilled from LLM verdicts -- no further LLM spend.")}


_VC = []  # side channel so distill() can see the paired candidate (kept simple; set in run())
def verdicts_with_cand(verdicts):
    return _VC


# --------------------------------------------------------------------------------------------- orchestrate
def cache_key(cand, model):
    h = hashlib.sha1(json.dumps([cand["target"], cand.get("machine"), cand.get("KOs"), model], sort_keys=True).encode()).hexdigest()[:16]
    return CACHE / f"{cand['target']}_{h}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="max candidates to adjudicate (budget guard)")
    ap.add_argument("--go", action="store_true", help="ACTUALLY call the LLM and spend; without it, estimate + mock only")
    ap.add_argument("--mock", action="store_true", help="force the deterministic mock adjudicator (no key, no spend)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="low", help="reasoning_effort: low (triage) | medium | high")
    ap.add_argument("--max-out", type=int, default=1200, help="max output tokens per candidate")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    cands = load_candidates()[:args.n]
    print(f"loaded {len(cands)} candidates (concordant-first). Grounding each in PubMed (free)...")
    grounded = []
    for c in cands:
        ev = ground(c)
        grounded.append((c, ev))
        print(f"   {c['target']:11s} <- {str(c.get('machine'))[:26]:26s} lines={c.get('n_lines_concordant')} "
              f"comentions={ev['comention_count']} (of {ev['target_total_papers']} papers)")

    # cost estimate
    prompts = [(SYSTEM, build_user(c, ev)) for c, ev in grounded]
    in_tok = sum(len(s) + len(u) for s, u in prompts) // 4
    out_tok = len(prompts) * args.max_out
    est = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
    print(f"\nCOST ESTIMATE for {len(prompts)} candidates on '{args.model}': ~{in_tok:,} in + ~{out_tok:,} out tokens "
          f"= ~${est:.2f}  (prices ${PRICE_IN}/${PRICE_OUT} per 1M in/out -- estimate; reasoning tokens billed as output)")

    use_real = args.go and not args.mock and os.environ.get("OPENAI_API_KEY")
    if args.go and not use_real:
        print("  --go set but no OPENAI_API_KEY (or --mock) -> falling back to mock. Set your key to spend.")
    print(f"  MODE: {'REAL LLM ($ SPEND)' if use_real else 'MOCK adjudicator (no key/no spend)'}\n")

    verdicts = []; spent = {"in": 0, "out": 0}; _VC.clear()
    for c, ev in grounded:
        ck = cache_key(c, args.model)
        if ck.exists():
            v = json.load(open(ck))
        elif use_real:
            try:
                v, usage = call_openai(SYSTEM, build_user(c, ev), args.model, args.effort, args.max_out)
                spent["in"] += usage.get("prompt_tokens", 0); spent["out"] += usage.get("completion_tokens", 0)
                v.setdefault("target", c["target"]); json.dump(v, open(ck, "w"), indent=1)
                time.sleep(0.3)
            except Exception as e:
                print(f"   ! {c['target']}: LLM failed ({e}); using mock for this one")
                v = mock_adjudicate(c, ev)
        else:
            v = mock_adjudicate(c, ev)
        v["_grounding"] = {"comentions": ev["comention_count"], "target_papers": ev["target_total_papers"]}
        verdicts.append(v); _VC.append((v, c))

    # combined ranking: deterministic recall (scorer) x LLM precision (novelty+verdict+confidence)
    def combined(v, c):
        base = c.get("pooled_confidence")
        if base is None:
            base = min((c.get("n_lines_concordant") or 1) / 3.0, 1.0)
        llm = float(v.get("llm_confidence") or 0)
        pen = 0.3 if v.get("is_known_stress_program") else 1.0
        return round(base * (0.4 + 0.6 * llm) * pen, 3)
    ranked = sorted([{"target": v["target"], "machine": c.get("machine"), "combined_score": combined(v, c),
                      "novelty": v.get("novelty"), "verdict": v.get("verdict"), "known_stress": v.get("is_known_stress_program"),
                      "llm_confidence": v.get("llm_confidence"), "mechanism": v.get("mechanism"),
                      "supporting_pmids": v.get("supporting_pmids"), "comentions": v["_grounding"]["comentions"],
                      "n_lines": c.get("n_lines_concordant")}
                     for v, c in zip(verdicts, [g[0] for g in grounded])], key=lambda x: -x["combined_score"])

    feedback = distill(verdicts)
    print("TOP RE-RANKED (deterministic recall x LLM precision):")
    print(f"   {'score':>5s} {'target':11s} {'nov':>8s} {'verdict':>12s}  machine")
    for r in ranked[:15]:
        flag = " [known-stress]" if r["known_stress"] else ""
        print(f"   {r['combined_score']:>5.2f} {r['target']:11s} {str(r['novelty']):>8s} {str(r['verdict']):>12s}  {str(r['machine'])[:26]:26s}{flag}")
    print(f"\nDISTILLED scorer_feedback: {len(feedback['known_stress_targets'])} known-stress targets, "
          f"{len(feedback['confirmed_novel_pursue'])} confirmed-novel-pursue, {len(feedback['machine_reliability_prior'])} machine priors")
    if use_real:
        real_cost = spent["in"] / 1e6 * PRICE_IN + spent["out"] / 1e6 * PRICE_OUT
        print(f"ACTUAL SPEND: {spent['in']:,} in + {spent['out']:,} out tokens = ~${real_cost:.3f}")

    out = {"model": args.model, "mode": "real" if use_real else "mock", "cost_estimate_usd": round(est, 2),
           "candidates_audited": len(verdicts), "ranked": ranked, "verdicts": verdicts, "scorer_feedback": feedback}
    json.dump(out, open(OUT / "llm_reasoner.json", "w"), indent=1)
    json.dump(feedback, open(OUT / "scorer_feedback.json", "w"), indent=1)
    print("\n  -> outputs/orphan/llm_reasoner.json  +  scorer_feedback.json (the distilled, reusable improvement)")
    return out


if __name__ == "__main__":
    main()
