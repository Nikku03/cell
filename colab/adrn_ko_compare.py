"""PAIRED COMPARISON of the ADRN arms against each other, the frequency baseline, and the incumbents.

Every method on the sealed protocol has been judged by the same three statistics, and this file applies them
unchanged so the new arms are directly comparable to the published numbers:

    paired mean difference in per-knockout precision
    95% bootstrap CI over knockouts (10,000 resamples)
    sign test on the per-knockout wins/losses

The unit of replication is the KNOCKOUT, not the prediction, because the claim is about knockouts.

ONE ASYMMETRY, STATED RATHER THAN QUIETLY EXPLOITED: ko_predict.score builds its frequency baseline by counting
movers over ALL 5,120 perturbations, INCLUDING the 400 sealed ones. The baseline therefore sees the held-out
answers and the ADRN arms do not, which makes every "vs frequency" comparison below conservative in the ADRN
arms' favour rather than the reverse.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
BOOT = 10_000


def rows(tag: str) -> dict[str, dict[str, float]]:
    p = OUT / f"ko_score_{tag}.json"
    if not p.exists():
        return {}
    return {r["gene"]: r for r in json.loads(p.read_text())["rows"]}


def paired(a: dict, b: dict, key_a: str = "precision", key_b: str = "precision"):
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    x = np.array([a[g][key_a] for g in shared])
    y = np.array([b[g][key_b] for g in shared])
    diff = x - y
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    # exact two-sided sign test over the decided pairs
    n = wins + losses
    if n:
        from math import comb
        k = min(wins, losses)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    else:
        p = 1.0
    return {"n": len(shared), "mean": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "wins": wins, "losses": losses, "sign_p": float(p),
            "a": float(x.mean()), "b": float(y.mean())}


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    ARMS = ["chan2a", "chan2b", "chan2perm", "mlk562", "mlpool", "mlstrict", "mlother", "mlbank",
            "adrnlin", "adrnconj", "adrnshuf", "adrnrand", "adrnrot", "adrnles", "adrnperm"]
    INCUMBENTS = ["basis", "nbr", "multilayer"]
    COHORTS = [("cohort 1", ""), ("cohort 2 (holdout)", "_holdout")]

    report("=" * 112)
    report("ADRN ON THE SEALED KNOCKOUT PROTOCOL -- paired comparisons")
    report("=" * 112)

    summary: dict[str, dict] = {}
    for label, tag in COHORTS:
        report(f"\n### {label}")
        report(f"  {'arm':<12} {'n':>4} {'precision':>10} {'frequency':>10} {'vs freq':>9} "
               f"{'95% CI':>22} {'sign p':>8}  {'wins/losses':>12}")
        for arm in ARMS + INCUMBENTS:
            r = rows(f"{arm}{tag}")
            if not r:
                continue
            cmp = paired(r, r, "precision", "frequency_hits")
            genes = sorted(r)
            prec = np.array([r[g]["precision"] for g in genes])
            freq = np.array([r[g]["frequency_hits"] / max(r[g]["n_pred"], 1) for g in genes])
            diff = prec - freq
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
            from math import comb
            n = wins + losses
            k = min(wins, losses)
            sp = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
            star = " *" if lo > 0 else ""
            report(f"  {arm:<12} {len(genes):>4} {prec.mean():>10.4f} {freq.mean():>10.4f} "
                   f"{diff.mean():>+9.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22} {sp:>8.4f}  "
                   f"{f'{wins}/{losses}':>12}{star}")
            summary.setdefault(tag or "cohort1", {})[arm] = {
                "precision": float(prec.mean()), "frequency": float(freq.mean()),
                "vs_freq": float(diff.mean()), "ci": [float(lo), float(hi)], "sign_p": float(sp),
                "wins": wins, "losses": losses, "n": len(genes)}

        report(f"\n  the decisive within-ADRN contrasts on {label}")
        report(f"  {'contrast':<28} {'mean diff':>10} {'95% CI':>22} {'sign p':>8}  {'wins/losses':>12}  verdict")
        for name, x, y in (("chan2a - adrnlin (channels)", "chan2a", "adrnlin"),
                           ("chan2b - chan2a (measured)", "chan2b", "chan2a"),
                           ("chan2a - PERMUTED (leakage)", "chan2a", "chan2perm"),
                           ("mlpool - mlk562 (pooling)", "mlpool", "mlk562"),
                           ("mlstrict - mlk562 (scale)", "mlstrict", "mlk562"),
                           ("mlpool - mlstrict (transfer)", "mlpool", "mlstrict"),
                           ("mlbank - mlpool (banking)", "mlbank", "mlpool"),
                           ("mlother - mlk562 (no K562)", "mlother", "mlk562"),
                           ("conj - linear (mechanism)", "adrnconj", "adrnlin"),
                           ("conj - shuffled (noise)", "adrnconj", "adrnshuf"),
                           ("conj - random (structure)", "adrnconj", "adrnrand"),
                           ("conj - rotated (alignment)", "adrnconj", "adrnrot"),
                           ("linear - PERMUTED (leakage)", "adrnlin", "adrnperm"),
                           ("linear - lesion-only (channels)", "adrnlin", "adrnles"),
                           ("lesion-only - basis (estimator)", "adrnles", "basis"),
                           ("linear - basis (incumbent)", "adrnlin", "basis"),
                           ("linear - nbr (incumbent)", "adrnlin", "nbr")):
            a, b = rows(f"{x}{tag}"), rows(f"{y}{tag}")
            res = paired(a, b)
            if not res:
                continue
            verdict = "RESOLVED" if (res["lo"] > 0 or res["hi"] < 0) else "within noise"
            interval = "[{:+.4f}, {:+.4f}]".format(res["lo"], res["hi"])
            record = "{}/{}".format(res["wins"], res["losses"])
            report(f"  {name:<28} {res['mean']:>+10.4f} {interval:>22} "
                   f"{res['sign_p']:>8.4f}  {record:>12}  {verdict}")
            summary.setdefault(tag or "cohort1", {}).setdefault("contrasts", {})[name] = res

    json.dump({"test": "adrn_ko_sealed_paired", "bootstrap": BOOT, "summary": summary, "log": log},
              open(OUT / "adrn_ko_compare.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_ko_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
