"""Live functioning of the E. coli minimal cell.

Takes the static essential-gene module network (cell_network.json) and animates
it as a *working cell*: a flux wave sweeps the central-dogma + metabolism
pipeline once per cell cycle --

  membrane (import)  ->  energy + cofactors  ->  building blocks
  (nucleotide/amino_acid/lipid)  ->  transcription/replication  ->
  tRNA/synthetase/ribosome  ->  translation  ->  CELL DIVISION  -> (repeat)

Each module's glow = its current activity; each arrow's brightness = the flux
flowing through it right now. A division flash fires when the wave reaches the
divisome. Module sizes are the real essential-gene counts.

Output: outputs/orphan/cell_live.gif         (the animation)
        outputs/orphan/cell_live_filmstrip.png (6 key frames, for a quick look)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = Path("outputs/orphan")
net = json.load(open(OUT / "cell_network.json"))
SIZ = net["module_sizes"]            # real essential-gene counts per module

# ---- curated flow layout: left (import) -> right (division) ----
POS = {
    "membrane":     (0.05, 0.50),
    "energy":       (0.23, 0.82),
    "cofactor":     (0.23, 0.18),
    "lipid":        (0.20, 0.50),
    "nucleotide":   (0.42, 0.84),
    "amino_acid":   (0.42, 0.16),
    "transcription":(0.60, 0.74),
    "replication":  (0.60, 0.92),
    "synthetase":   (0.55, 0.24),
    "trna":         (0.70, 0.22),
    "ribosomal":    (0.78, 0.42),
    "translation":  (0.82, 0.66),
    "cell_division":(0.96, 0.50),
    "other":        (0.50, 0.03),
}
# pipeline stage = wave phase (when this module fires in the cycle)
STAGE = {
    "membrane": 0, "energy": 1, "cofactor": 1, "lipid": 1,
    "nucleotide": 2, "amino_acid": 2,
    "transcription": 3, "replication": 3, "synthetase": 3,
    "trna": 4, "ribosomal": 4,
    "translation": 5, "cell_division": 6, "other": 2,
}
MAXSTAGE = 6
# directed material/information flows (src -> dst, label, dashed?)
FLOWS = [
    ("membrane", "energy", "nutrients", 0), ("membrane", "lipid", "", 0),
    ("membrane", "nucleotide", "", 0), ("membrane", "amino_acid", "", 0),
    ("membrane", "cofactor", "", 0),
    ("cofactor", "energy", "", 0),
    ("energy", "nucleotide", "ATP", 0), ("energy", "amino_acid", "ATP", 0),
    ("energy", "transcription", "ATP", 0), ("energy", "replication", "ATP", 0),
    ("energy", "translation", "ATP", 0),
    ("nucleotide", "transcription", "NTP", 0),
    ("nucleotide", "replication", "dNTP", 0),
    ("amino_acid", "synthetase", "aa", 0),
    ("synthetase", "trna", "charge", 0),
    ("trna", "ribosomal", "tRNA", 0),
    ("ribosomal", "translation", "ribosome", 0),
    ("transcription", "translation", "mRNA", 0),
    ("translation", "cell_division", "proteins", 0),
    ("replication", "cell_division", "genome", 0),
    ("lipid", "membrane", "", 1),                  # feedback (dashed)
    ("cell_division", "membrane", "", 1),          # new daughter cell
]
COLORS = plt.cm.tab20(np.linspace(0, 1, 14))
MODS = list(POS.keys())
CMAP = {m: COLORS[i % 14] for i, m in enumerate(SIZ.keys())}

T = 45                      # frames per cell cycle
SIGMA = T * 0.11
N_FRAMES = 90              # ~2 cell cycles


def activity(mod, t):
    """Gaussian flux pulse centred at the module's pipeline stage, repeating
    every cycle (wraparound)."""
    if mod == "other":
        return 0.30 + 0.12 * np.sin(2 * np.pi * t / T)   # background hum
    center = (STAGE[mod] / MAXSTAGE) * (T * 0.82)
    a = 0.0
    for k in (-1, 0, 1):
        a += np.exp(-((t - center - k * T) ** 2) / (2 * SIGMA ** 2))
    return min(1.0, a) * 0.95 + 0.05


def node_r(mod):
    return 0.018 + 0.010 * np.sqrt(SIZ.get(mod, 1))


def render(ax, t):
    ax.clear()
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_facecolor("#0b0f1a")
    # cell envelope
    env = plt.matplotlib.patches.Ellipse((0.52, 0.5), 1.02, 0.96,
            fill=False, edgecolor="#3a5f8a", lw=2.5, alpha=0.6)
    ax.add_patch(env)

    # flows (arrows) -- brightness follows source activity
    for s, d, lbl, dashed in FLOWS:
        if s not in POS or d not in POS:
            continue
        x1, y1 = POS[s]; x2, y2 = POS[d]
        flux = activity(s, t)
        col = (0.4 + 0.6 * flux, 0.85 * flux + 0.15, 0.3 + 0.3 * (1 - flux))
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=0.6 + 3.0 * flux, alpha=0.25 + 0.7 * flux,
                                    linestyle="--" if dashed else "-",
                                    shrinkA=14, shrinkB=14,
                                    connectionstyle="arc3,rad=0.08"))

    # nodes -- glow halo scaled by activity
    for m in MODS:
        if m not in SIZ:
            continue
        x, y = POS[m]; a = activity(m, t); r = node_r(m)
        base = np.array(CMAP[m][:3])
        # halo
        ax.scatter([x], [y], s=(r * 1100) ** 1.0 * (1 + 6 * a),
                   color=base, alpha=0.18 * a + 0.04, zorder=2,
                   edgecolors="none")
        # core node, brightened by activity
        face = tuple(np.clip(base * (0.45 + 0.55 * a) + 0.45 * a, 0, 1))
        ax.scatter([x], [y], s=(r * 1100),
                   color=face, edgecolors="white",
                   linewidths=0.4 + 1.2 * a, zorder=3)
        ax.text(x, y - r - 0.025, f"{m}\n{SIZ.get(m,0)}", ha="center",
                va="top", fontsize=7.5, color="#cdd6e6",
                fontweight="bold" if a > 0.5 else "normal", zorder=4)

    # division flash when the wave hits the divisome
    da = activity("cell_division", t)
    cyc = int(t / T) + 1
    if da > 0.6:
        xd, yd = POS["cell_division"]
        ax.scatter([xd], [yd], s=4000 * da, color="#ffe680",
                   alpha=0.5 * (da - 0.5), zorder=1, edgecolors="none")
        ax.text(0.52, 0.93, "CELL DIVISION", ha="center", color="#ffe680",
                fontsize=14, fontweight="bold", zorder=6)

    phase_name = ["import", "energize", "make precursors",
                  "transcribe / replicate", "load ribosomes",
                  "translate", "DIVIDE"][min(int((t % T) / T * 7), 6)]
    ax.set_title(f"E. coli minimal cell -- LIVE   |   cycle {cyc}   |   "
                 f"phase: {phase_name}", color="white", fontsize=13,
                 fontweight="bold", pad=10)
    # progress bar
    prog = (t % T) / T
    ax.add_patch(plt.Rectangle((0.04, 0.005), 0.92 * prog, 0.012,
                 color="#41b6c4", zorder=5))
    ax.add_patch(plt.Rectangle((0.04, 0.005), 0.92, 0.012, fill=False,
                 edgecolor="#41b6c4", lw=0.5, zorder=5))


# ---- animation ----
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("#0b0f1a")
anim = FuncAnimation(fig, lambda i: render(ax, i), frames=N_FRAMES, interval=70)
anim.save(OUT / "cell_live.gif", writer=PillowWriter(fps=15))
print("wrote outputs/orphan/cell_live.gif")

# ---- filmstrip: 6 key frames across one cycle ----
fig2, axes = plt.subplots(2, 3, figsize=(18, 11))
fig2.patch.set_facecolor("#0b0f1a")
for k, ax2 in enumerate(axes.ravel()):
    render(ax2, int(k / 6 * T) + 2)
fig2.suptitle("One cell cycle of the E. coli minimal cell (flux wave through "
              "the essential-gene modules)", color="white", fontsize=15,
              fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT / "cell_live_filmstrip.png", dpi=110,
            facecolor="#0b0f1a", bbox_inches="tight")
print("wrote outputs/orphan/cell_live_filmstrip.png")
