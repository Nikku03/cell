"""kcat_invivo_validate — the HONEST, non-circular test of the kcat flag/prediction, against the user's Drive set of
in-vivo (flux-derived) kcat values (davidi_kcat.json: 592 enzymes, kapp = max over 13 NCI-60 conditions of |v|/[E],
the Davidi & Milo 2016 in-vivo-kcat method). This SUPERSEDES the earlier kcat_flag.json / kcat_verify.json results,
which were CIRCULAR: they used enzyme_records `incell_rate_per_s`, which is literally `kcat_invitro * sigma`
(incell_rates.py:161 → enzyme_record.py:70). Correlating kcat*sigma against kcat is not a prediction, and
"kcat >= kcat*sigma" is trivially true — so the old 0.93 Spearman and 99% flag were artifacts.

Here the operating rate (kapp) is computed from FBA flux ÷ abundance with NO kcat input — a genuinely independent
number — and compared ONLY to experimentally-MEASURED reference kcats (tiers 'measured'/'EC-measured'; EC-class /
network-propagated / global-prior references are themselves defaults and would re-introduce circularity).

Three honest questions:
  (1) FLAG — does the real-flux kapp stay at/below the measured kcat (kcat is the max turnover; the cell can't run
      faster)? The fraction obeying it, and the false-flag rate when we treat kapp as a lower bound on a proposed kcat.
  (2) PREDICT — does kapp recover the measured kcat's VALUE (Spearman, fold-error vs the ~8.7x experimental noise)?
  (3) why it differs from E. coli, where the same method works (Davidi 2016 r≈0.6): measured flux + absolute
      proteomics there vs FBA-PREDICTED flux + a single reference proteome here.
-> outputs/orphan/kcat_invivo_validate.json  (and stamps a 'superseded' note into the two circular artifacts)
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr, pearsonr
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")
MEASURED = ("measured", "EC-measured")   # non-circular reference tiers only


def main():
    obj = json.load(open(f"{OUT}/davidi_kcat.json"))
    inner = obj["davidi_kcat"]
    val = obj.get("validation", {})
    # non-circular: real flux kapp vs experimentally-measured reference kcat
    rows = [(g, v["kcat_max_per_s"], v["kcat_measured_per_s"]) for g, v in inner.items()
            if v.get("measured_tier") in MEASURED and v.get("kcat_max_per_s", 0) > 0 and v.get("kcat_measured_per_s", 0) > 0]
    genes, kapp, meas = zip(*rows)
    kapp, meas = np.array(kapp), np.array(meas)
    sat = kapp / meas                                          # real saturation (flux kapp / measured kcat)

    # (1) FLAG — kapp as a lower bound on the true kcat
    frac_consistent = float((sat <= 1).mean())                 # measured kcat >= its own flux kapp (physics holds)
    false_flag = float((meas < kapp).mean())                   # correct kcat wrongly flagged as "below the floor"
    def flagged(mult):                                         # flag a proposed kcat = measured*mult if below kapp
        return float((meas * mult < kapp).mean())
    curve = {str(m): flagged(m) for m in (100, 10, 3, 1, 0.3, 0.1, 0.01, 0.001)}
    d100, d10 = curve["0.01"], curve["0.1"]

    # (2) PREDICT — does kapp recover the kcat value?
    fold = 10 ** np.abs(np.log10(sat))
    rho = float(spearmanr(kapp, meas).statistic)
    rlog = float(pearsonr(np.log10(kapp), np.log10(meas))[0])
    med_fold = float(np.median(fold))
    noise = 8.7                                                # experimental kcat noise floor (fold), from records
    ecoli_r = 0.60                                             # Davidi & Milo 2016 / Heckmann 2018 in E. coli

    print("=" * 84)
    print("HONEST (non-circular) validation of the kcat flag/prediction — against Drive in-vivo kapp")
    print("=" * 84)
    print(f"  source: davidi_kcat.json  ({len(inner)} enzymes, kapp = max |v|/[E] over 13 NCI-60 conditions)")
    print(f"  tested: {len(kapp)} with an EXPERIMENTALLY-MEASURED reference kcat (measured/EC-measured only)")
    print(f"  file's own validation block: median fold-error {val.get('median_fold_error','?')}x, "
          f"within-2x {val.get('within_2x','?')}\n")
    print("  (1) FLAG  — is the flux kapp at/below the measured kcat (a valid lower bound)?")
    print(f"        kapp <= measured kcat (physics holds): {frac_consistent:.0%}   "
          f"→ false-flag on correct kcats: {false_flag:.0%}")
    print(f"        detection: 100x-too-slow flagged {d100:.0%}, 10x-too-slow {d10:.0%}  "
          f"(shallow: kapp uncertain by ~{np.median(fold):.0f}x)")
    print("  (2) PREDICT — does the flux kapp recover the measured kcat VALUE?")
    print(f"        Spearman {rho:+.2f}   Pearson(log) {rlog:+.2f}   median fold-error {med_fold:.0f}x  "
          f"vs noise floor {noise:.1f}x → {'within' if med_fold < noise else 'FAR ABOVE'} noise")
    print(f"        median saturation kapp/kcat {np.median(sat):.3f}  (enzymes run far below their measured max here)")
    print("  (3) WHY weaker than E. coli (Davidi 2016 r≈0.6): there flux is MEASURED (13C-MFA) + absolute")
    print("        proteomics; here flux is FBA-PREDICTED and abundance is one reference proteome → too noisy.")

    flag_works = frac_consistent >= 0.9 and false_flag <= 0.1 and d100 >= 0.8
    predict_works = rho >= 0.5 and med_fold < noise
    verdict = (
        f"CORRECTED / HONEST NULL. On {len(kapp)} enzymes with a MEASURED kcat and an INDEPENDENT flux-derived kapp: "
        f"the flag is WEAK (kapp ≤ kcat only {frac_consistent:.0%}, so {false_flag:.0%} of correct kcats would be "
        f"false-flagged; a 100x-too-slow kcat is caught only {d100:.0%}), and kapp does NOT predict the kcat value "
        f"(Spearman {rho:+.2f}, {med_fold:.0f}x fold-error ≫ {noise:.1f}x noise). This OVERTURNS the earlier "
        f"kcat_flag/kcat_verify claims (99% flag, 0.93 Spearman), which were CIRCULAR — they used incell_rate = "
        f"kcat×sigma. The kapp method is real (works in E. coli, r≈{ecoli_r}), but our human reconstruction uses "
        f"FBA-PREDICTED flux + a reference proteome, which is too noisy to flag or recover kcat per-enzyme. "
        f"What survives: flux kapp is a genuine but noisy LOWER-BOUND signal, useful in aggregate, not as a "
        f"per-enzyme verdict."
        if not (flag_works or predict_works) else
        f"kapp usable: flag_works={flag_works}, predict_works={predict_works} (rho {rho:.2f}, fold {med_fold:.0f}x).")
    print("\n" + "=" * 84 + f"\nVERDICT: {verdict}\n" + "=" * 84)

    out = {
        "source": "davidi_kcat.json (Drive) — 592 enzymes, in-vivo kapp = max|v|/[E] over 13 NCI-60 conditions",
        "n_total_enzymes": len(inner), "n_measured_reference": len(kapp),
        "reference_tiers_used": list(MEASURED),
        "flag": {"frac_kapp_le_measured_kcat": frac_consistent, "false_flag_rate": false_flag,
                 "flag_100x_too_slow": d100, "flag_10x_too_slow": d10, "detection_curve": curve},
        "predict": {"spearman_kapp_vs_measured_kcat": rho, "pearson_log": rlog,
                    "median_fold_error": med_fold, "within_3x": float((fold <= 3).mean()),
                    "within_10x": float((fold <= 10).mean()), "experimental_noise_floor_fold": noise},
        "median_saturation_kapp_over_kcat": float(np.median(sat)),
        "file_own_validation": val,
        "ecoli_reference_r": ecoli_r,
        "supersedes": ["kcat_flag.json", "kcat_verify.json"],
        "circularity_note": ("earlier kcat_flag/kcat_verify used enzyme_records incell_rate_per_s = kcat_invitro*sigma "
                             "(incell_rates.py:161 -> enzyme_record.py:70); correlating/kcat-bounding against it was "
                             "circular. This file uses flux-derived kapp with NO kcat input."),
        "verdict": verdict,
    }
    json.dump(out, open(f"{OUT}/kcat_invivo_validate.json", "w"), indent=1)

    # stamp the two circular artifacts so the record is self-consistent
    for f, oldnum in (("kcat_flag.json", "99% flag / 6% false-flag"), ("kcat_verify.json", "Spearman 0.93 / 1.9x")):
        p = f"{OUT}/{f}"
        if os.path.exists(p):
            j = json.load(open(p))
            j["_SUPERSEDED"] = (f"CIRCULAR ({oldnum}): used incell_rate = kcat*sigma. See kcat_invivo_validate.json "
                                f"for the honest non-circular result (flag ~{frac_consistent:.0%}, Spearman {rho:+.2f}).")
            json.dump(j, open(p, "w"), indent=1)
    return verdict


if __name__ == "__main__":
    main()
