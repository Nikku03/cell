"""Paper 1, Figure 2 -- decomposition of the continuous reproducibility floor
and the calibration-corrected ceiling.

Panel A: variance partition of the 17,950-pair cross-experiment Spearman
         (technical / condition / shared / residual).
Panel B: the floor on the Spearman scale -- raw median, 10-90% band, and the
         matched-quality ceiling, showing the headroom recovered by removing
         technical noise.

Reads outputs/kappa_floor_decomposition.json if present; otherwise uses the
measured values (commit 9fd1ea7 run). Writes outputs/paper1_figure2.png.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"

# measured values (fallback if the JSON isn't local)
MEASURED = {
    "floor_median": 0.376, "floor_mean": 0.402,
    "floor_p10": 0.136, "floor_p90": 0.751,
    "matched_highquality_ceiling": 0.769,
    "n_pairs": 17950,
    "variance_partition": {
        "technical_unique_pct": 47.7,
        "shared_pct": 4.3,
        "condition_unique_pct": 2.6,
        "residual_irreducible_pct": 45.4,
    },
}


def load():
    p = OUT / "kappa_floor_decomposition.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            # ensure required keys
            d.setdefault("variance_partition", MEASURED["variance_partition"])
            return d
        except Exception:
            pass
    return MEASURED


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    d = load()
    vp = d["variance_partition"]
    med = d["floor_median"]; p10 = d["floor_p10"]; p90 = d["floor_p90"]
    ceil = d["matched_highquality_ceiling"]
    npairs = d.get("n_pairs", MEASURED["n_pairs"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.3),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---------------- Panel A: variance partition ----------------
    parts = [
        ("Technical\n(library quality)", vp["technical_unique_pct"], "#c44e52"),
        ("Shared", vp["shared_pct"], "#dd8452"),
        ("Condition\nmismatch", vp["condition_unique_pct"], "#55a868"),
        ("Residual / irreducible\n(+ buffering)", vp["residual_irreducible_pct"],
         "#8c8c8c"),
    ]
    left = 0.0
    for label, pct, color in parts:
        axA.barh(0, pct, left=left, color=color, edgecolor="white",
                 height=0.55)
        if pct >= 4:
            axA.text(left + pct / 2, 0, f"{pct:.0f}%", ha="center",
                     va="center", color="white", fontsize=11, fontweight="bold")
        left += pct
    # legend below
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    axA.legend(handles, [p[0] for p in parts], loc="upper center",
               bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5,
               frameon=False, handlelength=1.2)
    axA.set_xlim(0, 100); axA.set_ylim(-0.5, 0.5)
    axA.set_yticks([]); axA.set_xlabel("share of cross-experiment variance (%)")
    axA.set_title(f"A  Decomposition of the floor\n({npairs:,} same-condition "
                  f"pairs, 48 organisms)", fontsize=11, loc="left")
    axA.spines[["top", "right", "left"]].set_visible(False)

    # annotate the headline
    axA.text(vp["technical_unique_pct"] / 2, 0.42,
             "≈ half is technical,\nnot biology", ha="center", fontsize=8.5,
             color="#c44e52", style="italic")

    # ---------------- Panel B: floor vs ceiling on the Spearman scale -------
    axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    y = 0.6
    # 10-90% band of the raw floor
    axB.add_patch(plt.Rectangle((p10, y - 0.06), p90 - p10, 0.12,
                                color="#cfe0ee", zorder=1))
    axB.text((p10 + p90) / 2, y + 0.30, "raw floor 10–90%",
             ha="center", fontsize=8, color="#3b6ea5")
    # raw median floor (below)
    axB.plot([med, med], [y - 0.10, y + 0.10], color="#c44e52", lw=2.5, zorder=3)
    axB.text(med, y - 0.30, f"raw median floor\nρ={med:.2f}", ha="center",
             va="top", fontsize=9, color="#c44e52", fontweight="bold")
    # matched-quality ceiling (above)
    axB.plot([ceil, ceil], [y - 0.10, y + 0.10], color="#2a7a2a", lw=2.5,
             zorder=3)
    axB.text(ceil, y + 0.16, f"matched-quality\nceiling  ρ={ceil:.2f}",
             ha="center", va="bottom", fontsize=9, color="#2a7a2a",
             fontweight="bold")
    # recovered-headroom arrow (well below the floor label)
    ay = y - 0.46
    arr = FancyArrowPatch((med, ay), (ceil, ay), arrowstyle="<->",
                          color="black", lw=1.3, mutation_scale=12, zorder=4)
    axB.add_patch(arr)
    axB.text((med + ceil) / 2, ay - 0.05,
             "headroom recovered by\nremoving technical noise",
             ha="center", va="top", fontsize=8, style="italic")
    axB.set_yticks([]); axB.set_xlabel("cross-experiment Spearman ρ")
    axB.set_title("B  The corrected ceiling\n(what two high-quality matched "
                  "screens agree on)", fontsize=11, loc="left")
    axB.spines[["top", "right", "left"]].set_visible(False)

    fig.tight_layout()
    out = OUT / "paper1_figure2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  technical {vp['technical_unique_pct']}%  "
          f"condition {vp['condition_unique_pct']}%  "
          f"residual {vp['residual_irreducible_pct']}%")
    print(f"  raw median floor ρ={med:.2f} -> matched-quality ceiling ρ={ceil:.2f}")


if __name__ == "__main__":
    main()
