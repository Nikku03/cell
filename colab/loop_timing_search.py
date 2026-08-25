"""Loop 197. Is there a public series dense enough to replicate the accessibility clock -- or not?

WHERE THIS SITS. Loop 191d measured that promoter accessibility reaches half its plateau before the
mRNA does in A549 under dexamethasone. Loop 192 found a clean replication candidate -- dendritic
cells under LPS, one lab for all 59 experiments, a graded clock, donor split-half +0.366 -- and
established it cannot answer the question: the A549 lead reverses when downsampled to that series'
four timepoints. Loop 196 then closed the other cheap route by showing that four estimators chosen
to fail differently -- a level-crossing interpolation, an increment-weighted mean, an area integral
and a parametric exponential fit -- all recover the lead on eleven points and none on four. Methods
with that little in common do not collapse together by accident; the limit is the information in
four timepoints, not the estimator.

So one route remains: a denser series. This loop looks for one, and it is written so that "there
isn't one" is a real, reportable outcome rather than a dead end. If nothing qualifies, the output is
a written caveat attached to the capability record, because the alternative is leaving loop 191d's
number standing unqualified in a project whose census table is supposed to be the honest statement
of what is known.

WHY THE EARLIER SEARCH MAY HAVE MISSED SOMETHING, which is the reason this is a loop and not a
one-line query. Loop 192's ENCODE search required a treatment DURATION on the biosample. That is how
a drug time course records time, and it is not how a differentiation time course records it --
those use post-synchronization or post-differentiation time, or nothing structured at all. A series
following cells over hours after a stimulus that ENCODE does not model as a "treatment" would have
been invisible. The search is therefore broadened here rather than repeated.

THE GATE THAT COMES FIRST IS A CONTROL ON THE SEARCH ITSELF. Y1 requires the broadened query to
find the A549 series -- the one system known to qualify, because this project already used it. A
search that cannot find the positive control it is holding cannot support a negative conclusion,
and "we looked and found nothing" is exactly the kind of claim that needs that check. It is loop
192's W3 discipline applied to a query instead of a statistic: before believing an absence,
demonstrate the instrument can detect a presence.

WHAT COUNTS AS QUALIFYING, declared before the search runs.

    at least 8 timepoints carrying BOTH an accessibility assay and RNA, in one biosample system.
    Eight because four is measured to be too few and the A549 series that works has eleven; this
    is the midpoint and it is a guess, stated as one.

    a span of at most 24 hours. The effect is on the order of an hour, so a series sampled over
    days cannot resolve it however many points it has -- that is why loop 192 rejected the K562
    chromatin-drug panel at 4, 12, 24 and 48 hours, and the same arithmetic applies here.

    one cell system. Pooling cell types along a time axis puts a biological difference where the
    measurement is, which is the defect loop 191c had to discard 25 minutes of A549 data over.

PREDECLARED, BEFORE ANY NUMBER.

  Y1 CAN THE SEARCH FIND WHAT IT IS LOOKING FOR? The broadened ENCODE query, checked against the
     A549 dexamethasone series.
     Gate: PASS iff A549 + dexamethasone appears with at least 8 matched timepoints. A FAIL means
     the query is broken and every count below is uninterpretable, so Y2 and Y4 become VOID rather
     than negative.

  Y2 DOES ENCODE HOLD ANYTHING ELSE? The same query, with A549 excluded, ranked by matched
     timepoint count.
     Gate: PASS iff at least one other system reaches 8 matched timepoints within 24 hours.

  Y3 DOES GEO? NCBI E-utilities over the GEO DataSets index, with the declared criteria.
     Gate: descriptive, and deliberately so. GEO stores timepoints in free-text summaries that are
     not reliably machine-readable, so this arm produces a RANKED SHORTLIST for a human or a later
     loop to check, and does not claim to have decided anything automatically. Reporting a parsed
     count from a GEO summary as though it were structured metadata would be inventing precision.

  Y4 IS THE CLOCK REPLICABLE WITH PUBLIC DATA? The union of Y2 and Y3's structured findings.
     Gate: PASS iff a qualifying series exists. A FAIL is the finding, not a failure of the loop.

  Y5 THE CAVEAT IS WRITTEN EITHER WAY. Whatever Y4 returns, the capability record gets a line
     stating what is measured, where, and whether it has been reproduced anywhere else.
     Gate: PASS iff the record is written.

  Y6 WHAT THIS CANNOT SHOW.

-> outputs/loop_timing_search.json and NOTES_accessibility_clock_status.md
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_timing_search.json"
NOTE = Path("NOTES_accessibility_clock_status.md")
ENCODE = "https://www.encodeproject.org"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

MIN_POINTS = 8            # matched accessibility+RNA timepoints; 4 is measured too few, A549 has 11
MAX_SPAN_MIN = 1440.0     # 24 h -- an hour-scale effect cannot be resolved over days
ACC_ASSAYS = ("DNase-seq", "ATAC-seq")
RNA_ASSAYS = ("polyA plus RNA-seq", "total RNA-seq", "polyA minus RNA-seq")
UNITS = {"minute": 1.0, "hour": 60.0, "day": 1440.0, "week": 10080.0, "month": 43200.0}
GEO_QUERIES = (
    "(ATAC-seq[All Fields] AND RNA-seq[All Fields]) AND time course[All Fields] "
    "AND Homo sapiens[Organism]",
    "(DNase[All Fields] AND RNA-seq[All Fields]) AND time course[All Fields] "
    "AND Homo sapiens[Organism]",
    "chromatin accessibility[All Fields] AND time course[All Fields] AND transcriptome[All Fields] "
    "AND Homo sapiens[Organism]",
)

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def get(url, tries=4, timeout=120):
    last = None
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers={"accept": "application/json",
                                                     "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=timeout))
        except Exception as exc:                                  # noqa: BLE001
            last = exc
            time.sleep(2 ** i)
    raise RuntimeError(f"{url}: {last}")


def encode_series(report=print):
    """Every released human accessibility or RNA experiment carrying ANY structured time field.

    Loop 192 asked only for treatment duration. A differentiation series records time as
    post-synchronization or post-differentiation time instead, so it could not have been seen. All
    three are read here and the field that supplied each timepoint is kept, because a series whose
    time comes from a different field is not necessarily comparable to one whose time is a drug
    exposure -- and that has to be visible rather than merged away."""
    fields = ("&field=accession&field=assay_title&field=lab.title"
              "&field=biosample_ontology.term_name"
              "&field=replicates.library.biosample.treatments.duration"
              "&field=replicates.library.biosample.treatments.duration_units"
              "&field=replicates.library.biosample.treatments.treatment_term_name"
              "&field=replicates.library.biosample.post_synchronization_time"
              "&field=replicates.library.biosample.post_synchronization_time_units"
              "&field=replicates.library.biosample.post_differentiation_time"
              "&field=replicates.library.biosample.post_differentiation_time_units")
    assays = "".join(f"&assay_title={urllib.parse.quote(a)}" for a in ACC_ASSAYS + RNA_ASSAYS)
    q = ("/search/?type=Experiment&status=released&limit=all&format=json"
         "&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens"
         + assays + fields)
    rows = get(ENCODE + q).get("@graph", [])
    report(f"     {len(rows):,} released human accessibility/RNA experiments returned")
    series = defaultdict(lambda: defaultdict(set))
    source = defaultdict(set)
    for r in rows:
        cell = (r.get("biosample_ontology") or {}).get("term_name", "?")
        assay = r.get("assay_title")
        pts = set()
        for rep in r.get("replicates", []):
            bs = (rep.get("library") or {}).get("biosample") or {}
            for t in bs.get("treatments", []) or []:
                if t.get("duration"):
                    pts.add((float(t["duration"]) * UNITS.get(t.get("duration_units"), 0.0),
                             f"treatment:{t.get('treatment_term_name')}"))
            for fld in ("post_synchronization_time", "post_differentiation_time"):
                if bs.get(fld):
                    try:
                        v = float(bs[fld])
                    except (TypeError, ValueError):
                        continue
                    pts.add((v * UNITS.get(bs.get(fld + "_units"), 0.0), fld))
        for v, src in pts:
            key = (cell, src)
            series[key][assay].add(v)
            source[key].add(src)
    return series


def summarise(series, exclude=None):
    """Matched timepoints per (cell, time-source) key, with the span, ranked."""
    out = []
    for (cell, src), assays in series.items():
        if exclude and exclude(cell, src):
            continue
        acc = set().union(*[assays.get(a, set()) for a in ACC_ASSAYS]) if assays else set()
        rna = set().union(*[assays.get(a, set()) for a in RNA_ASSAYS]) if assays else set()
        shared = sorted(acc & rna)
        if not shared:
            continue
        span = max(shared) - min(shared)
        out.append(dict(cell=cell, source=src, n_shared=len(shared), span_min=span,
                        n_acc=len(acc), n_rna=len(rna),
                        shared=[round(x, 1) for x in shared[:16]]))
    out.sort(key=lambda d: (-d["n_shared"], d["span_min"]))
    return out


def geo_shortlist(report=print):
    """A ranked shortlist, explicitly NOT an automated decision.

    GEO records timepoints in free-text summaries. Parsing a count out of prose and reporting it
    beside ENCODE's structured counts would give the two the same apparent standing, and they do
    not have it. What is returned is titles and sample counts for a human or a later loop."""
    seen, rows = set(), []
    for term in GEO_QUERIES:
        try:
            es = get(EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(
                dict(db="gds", term=term, retmax=40, retmode="json")))
            ids = es["esearchresult"]["idlist"]
        except Exception as exc:                                  # noqa: BLE001
            report(f"     query failed ({type(exc).__name__}); skipped")
            continue
        ids = [i for i in ids if i not in seen]
        seen.update(ids)
        if not ids:
            continue
        try:
            su = get(EUTILS + "esummary.fcgi?" + urllib.parse.urlencode(
                dict(db="gds", id=",".join(ids), retmode="json")))
        except Exception as exc:                                  # noqa: BLE001
            report(f"     summary failed ({type(exc).__name__}); skipped")
            continue
        for i in ids:
            d = su.get("result", {}).get(i)
            if not d:
                continue
            rows.append(dict(acc=d.get("accession"), title=(d.get("title") or "")[:110],
                             n_samples=int(d.get("n_samples") or 0),
                             gdstype=(d.get("gdstype") or "")[:60]))
    rows.sort(key=lambda r: -r["n_samples"])
    return rows


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 197  IS THERE A PUBLIC SERIES DENSE ENOUGH TO REPLICATE THE ACCESSIBILITY CLOCK?")
    say("=" * 104)
    say(f"  PREDECLARED: qualifying means >= {MIN_POINTS} timepoints carrying BOTH an accessibility")
    say(f"  assay and RNA, in ONE cell system, spanning at most {MAX_SPAN_MIN/60:.0f} hours -- four")
    say("  is measured too few (loops 192, 196), the A549 series that works has eleven, and an")
    say("  hour-scale effect cannot be resolved over days however many points there are. The")
    say("  ENCODE query is BROADENED beyond loop 192's treatment-duration filter, which could not")
    say("  see a differentiation series. Y1 checks the query can find A549 before any absence is")
    say("  believed. The GEO arm is a shortlist and says so: its timepoints live in prose and")
    say("  parsing a count out of prose would give it standing it has not earned.")
    say()

    # ---- Y1 ------------------------------------------------------------------------------------
    say("Y1 CAN THE SEARCH FIND WHAT IT IS LOOKING FOR?")
    series = encode_series(say)
    allrank = summarise(series)
    a549 = [d for d in allrank if d["cell"] == "A549" and "dexamethasone" in d["source"]]
    if a549:
        d = a549[0]
        say(f"     positive control: {d['cell']} / {d['source']} -- {d['n_shared']} matched "
            f"timepoints over {d['span_min']/60:.1f} h  {d['shared']}")
    else:
        say("     positive control NOT FOUND")
    y1 = bool(a549 and a549[0]["n_shared"] >= MIN_POINTS)
    GG.verdict(y1, emit=say,
               if_true=f"Y1 PASS -- the broadened query recovers the A549 series at "
                       f"{a549[0]['n_shared']} matched timepoints, so an absence below is about "
                       f"the archive and not about the query",
               if_false="Y1 FAIL -- the query cannot find the one series known to qualify, so "
                        "every count below is uninterpretable and Y2 and Y4 are VOID")

    void = set()
    if not y1:
        void |= {"Y2", "Y4"}

    # ---- Y2 ------------------------------------------------------------------------------------
    say()
    say("Y2 DOES ENCODE HOLD ANYTHING ELSE?")
    others = summarise(series, exclude=lambda c, s: c == "A549" and "dexamethasone" in s)
    say(f"     {len(others)} (cell, time-source) series carry at least one matched timepoint")
    say("     top by matched timepoint count:")
    for d in others[:12]:
        flag = "QUALIFIES" if (d["n_shared"] >= MIN_POINTS and d["span_min"] <= MAX_SPAN_MIN) \
            else ("too coarse" if d["n_shared"] < MIN_POINTS else "span too long")
        say(f"       {d['n_shared']:3d} pts  span {d['span_min']/60:7.1f} h  "
            f"{d['cell'][:34]:34s} {d['source'][:34]:34s} {flag}")
    qual = [d for d in others if d["n_shared"] >= MIN_POINTS and d["span_min"] <= MAX_SPAN_MIN]
    y2 = bool(qual)
    if "Y2" in void:
        say("     Y2 VOID -- Y1 failed")
    else:
        GG.verdict(y2, emit=say,
                   if_true=f"Y2 PASS -- {len(qual)} other ENCODE series qualify",
                   if_false=f"Y2 FAIL -- no other ENCODE series reaches {MIN_POINTS} matched "
                            f"timepoints within {MAX_SPAN_MIN/60:.0f} hours. The A549 series is "
                            f"the archive's only densely sampled matched accessibility/RNA course")

    # ---- Y3 ------------------------------------------------------------------------------------
    say()
    say("Y3 DOES GEO? -- a shortlist, not a decision")
    geo = geo_shortlist(say)
    say(f"     {len(geo)} distinct GEO series across {len(GEO_QUERIES)} queries")
    for r in geo[:12]:
        say(f"       {r['acc']:>12s}  {r['n_samples']:4d} samples  {r['title']}")
    say("     these counts are SAMPLES, not matched timepoints. GEO records the time grid in free")
    say("     text, so which of these carry >= 8 matched accessibility and RNA points cannot be")
    say("     read from the metadata and is not claimed here.")
    say("     Y3 (descriptive)")

    # ---- Y4 ------------------------------------------------------------------------------------
    say()
    say("Y4 IS THE CLOCK REPLICABLE WITH PUBLIC DATA?")
    y4 = bool(qual)
    if "Y4" in void:
        say("     Y4 VOID -- Y1 failed")
    else:
        # PRECOMPUTED because GG.verdict evaluates BOTH branch f-strings before the call. Loop
        # 196's X4 crashed on d4[None] for this reason, was fixed, and the identical defect was
        # written again here as qual[0] on an empty list. Any success message referencing a
        # success-only value must be built defensively, not inside the call.
        best = (f"{qual[0]['cell']} / {qual[0]['source']} at {qual[0]['n_shared']} matched points"
                if qual else "n/a")
        GG.verdict(y4, emit=say,
                   if_true=f"Y4 PASS -- {best} is a candidate. It must still pass loop 192's W3 "
                           f"downsampling calibration before any replication result from it is "
                           f"read",
                   if_false="Y4 FAIL -- and this is the finding. No structured public series other "
                            "than A549 is dense enough. The GEO shortlist may contain one, but "
                            "that cannot be established from metadata and needs a per-series check")

    # ---- Y5 ------------------------------------------------------------------------------------
    say()
    say("Y5 THE CAVEAT IS WRITTEN EITHER WAY")
    status = ("a qualifying candidate exists and is untested"
              if y4 else "no qualifying public series was found")
    NOTE.write_text(f"""# Status of the accessibility clock

The one result in this project carrying a fourth dimension is loop 191d's finding that promoter
accessibility reaches half its plateau before the mRNA does. This file records what is and is not
established about it, so the number is never quoted without its qualification.

## What is measured

A549 lung carcinoma, 100 nM dexamethasone, ENCODE GGR series, one lab pinned. Promoter DNase
against polyA RNA on a shared grid, 1,310 responding genes, one-sided Wilcoxon p 6.4e-58, holding
inside all three magnitude terciles. Two negative controls pass: CTCF +0.061 and RAD21 -0.022,
architectural factors that should be and are inert to a steroid response.

## What is NOT established

REPLICATION. None. Loop 192 identified the ENCODE dendritic-cell LPS series as a clean candidate --
one lab for all 59 experiments, a graded clock, donor split-half +0.366 -- and measured that it
cannot answer the question: the A549 lead REVERSES when downsampled to that series' four
timepoints. Loop 196 then tested four estimators chosen to fail differently, a level-crossing
interpolation, an increment-weighted mean, an area integral and a parametric exponential fit. All
four recover the lead on eleven points and none on four. The limit is the information in four
timepoints, not the estimator.

THE SIZE OF THE EFFECT. "The A549 lead" is not one number across these loops. Loop 191d reports
+48 min, loop 192 reports +154 and loop 196 reports +101.6 for the same statistic, because each
uses a different grid, replicate set and accessibility baseline convention. Each loop's internal
comparison is like for like; the cross-loop numbers are not, and none of them should be quoted as
"the" lead.

CAUSATION. Nothing perturbs accessibility anywhere in this arc. Accessibility opening before
transcription is equally consistent with chromatin gating transcription and with a third factor
driving both at different lags.

## Search for a denser series ({status})

Loop 197 broadened loop 192's ENCODE query beyond its treatment-duration filter, which could not
have seen a differentiation series, and checked the broadened query against A549 as a positive
control before believing any absence.

## How this must be quoted

In the census capability table and anywhere else: measured in A549 under dexamethasone,
UNREPLICATED, with the effect size unstable across analysis choices. Not as a general property of
human gene regulation.
""")
    say(f"     wrote {NOTE}")
    y5 = NOTE.exists()
    GG.verdict(y5, emit=say,
               if_true=f"Y5 PASS -- {NOTE} records what is measured, what is not established, and "
                       f"how the number must be quoted",
               if_false="Y5 FAIL -- the record was not written")

    # ---- Y6 ------------------------------------------------------------------------------------
    say()
    say("Y6 WHAT THIS CANNOT SHOW")
    say("     An absence in a structured archive is not an absence in the literature. ENCODE models")
    say("     time in three fields and this loop reads all three, but a series that records its")
    say("     grid only in a description is invisible to it and would be invisible to any query.")
    say("     The 8-point threshold is a guess placed between the 4 that is measured to fail and")
    say("     the 11 that works. Nothing here establishes that 8 is sufficient -- a candidate found")
    say("     at 8 would still have to pass loop 192's W3 calibration before being believed.")
    say("     The GEO arm ranks by sample count, which is not timepoint count. A 200-sample series")
    say("     may be one timepoint across many conditions.")
    say("     This loop searched human data only. A dense mouse series would be a weaker but real")
    say("     test and is not considered here.")
    say("     Y6 PASS")

    gates = {"Y1": y1, "Y2": y2, "Y3": True, "Y4": y4, "Y5": y5, "Y6": True}
    man = RM.manifest(inputs=[], available=len(allrank), used=len(others),
                      selection="filtered", seed=0,
                      controls=["the query checked against A549 as a positive control before any "
                                "absence is believed",
                                "GEO reported as a shortlist rather than a parsed count"],
                      note="search for a series dense enough to replicate the accessibility clock")
    out_d = dict(test="timing replication search", gates=gates, void=sorted(void),
                 min_points=MIN_POINTS, max_span_min=MAX_SPAN_MIN,
                 positive_control=a549[0] if a549 else None,
                 encode_ranked=others[:40], qualifying=qual, geo_shortlist=geo[:40],
                 manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'VOID' if k in void else ('PASS' if v else 'FAIL')}")
    scored = [k for k in gates if k not in void]
    say(f"  {sum(gates[k] for k in scored)}/{len(scored)}   [{time.time()-t0:.0f}s]"
        + (f"   ({len(void)} VOID: {', '.join(sorted(void))})" if void else ""))
    say("=" * 104)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
