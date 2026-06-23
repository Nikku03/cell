"""Watch the cell die: knock out an essential module and the AND gate flips.

Same flux-wave model as cell_live.py, but at a chosen frame we set the
RIBOSOMAL module's activity permanently to zero (an essential-gene knockout).
Translation can no longer fire -> no proteins -> no division -> every
downstream module starves and the whole wave flatlines. The cell goes dark.

This is the visual statement of the AND gate: one essential lost = death,
not shrinkage.

Output: outputs/orphan/cell_die.gif
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = Path("outputs/orphan")
SIZ = json.load(open(OUT / "cell_network.json"))["module_sizes"]

POS = {
    "membrane": (0.05, 0.50), "energy": (0.23, 0.82), "cofactor": (0.23, 0.18),
    "lipid": (0.20, 0.50), "nucleotide": (0.42, 0.84), "amino_acid": (0.42, 0.16),
    "transcription": (0.60, 0.74), "replication": (0.60, 0.92),
    "synthetase": (0.55, 0.24), "trna": (0.70, 0.22), "ribosomal": (0.78, 0.42),
    "translation": (0.82, 0.66), "cell_division": (0.96, 0.50), "other": (0.50, 0.03),
}
STAGE = {"membrane": 0, "energy": 1, "cofactor": 1, "lipid": 1, "nucleotide": 2,
         "amino_acid": 2, "transcription": 3, "replication": 3, "synthetase": 3,
         "trna": 4, "ribosomal": 4, "translation": 5, "cell_division": 6, "other": 2}
MAXSTAGE = 6
FLOWS = [
    ("membrane","energy",0),("membrane","lipid",0),("membrane","nucleotide",0),
    ("membrane","amino_acid",0),("membrane","cofactor",0),("cofactor","energy",0),
    ("energy","nucleotide",0),("energy","amino_acid",0),("energy","transcription",0),
    ("energy","replication",0),("energy","translation",0),("nucleotide","transcription",0),
    ("nucleotide","replication",0),("amino_acid","synthetase",0),("synthetase","trna",0),
    ("trna","ribosomal",0),("ribosomal","translation",0),("transcription","translation",0),
    ("translation","cell_division",0),("replication","cell_division",0),
    ("lipid","membrane",1),("cell_division","membrane",1),
]
# who depends on whom (for starvation cascade after knockout)
DEPENDS = {
    "translation": ["ribosomal", "trna", "synthetase"],
    "cell_division": ["translation", "replication"],
    "trna": ["synthetase"], "ribosomal": ["trna"],
}
COLORS = plt.cm.tab20(np.linspace(0, 1, 14))
CMAP = {m: COLORS[i % 14] for i, m in enumerate(SIZ.keys())}
MODS = list(POS.keys())

T = 45; SIGMA = T * 0.11
KO_MODULE = "ribosomal"
KO_FRAME = T + 18                  # knockout mid-second-cycle
N_FRAMES = 100


def base_activity(mod, t):
    if mod == "other":
        return 0.30 + 0.12 * np.sin(2 * np.pi * t / T)
    center = (STAGE[mod] / MAXSTAGE) * (T * 0.82)
    a = sum(np.exp(-((t - center - k * T) ** 2) / (2 * SIGMA ** 2)) for k in (-1, 0, 1))
    return min(1.0, a) * 0.95 + 0.05


def activity(mod, t, dead_set):
    if mod in dead_set:
        return 0.0
    a = base_activity(mod, t)
    # starvation: if a dependency is dead, this module decays after knockout
    if t >= KO_FRAME:
        for dep in DEPENDS.get(mod, []):
            if dep in dead_set:
                decay = max(0.0, 1 - (t - KO_FRAME) / 10.0)
                a *= decay
    return a


def node_r(m): return 0.018 + 0.010 * np.sqrt(SIZ.get(m, 1))


def dead_at(t):
    """which modules are non-functional at frame t."""
    if t < KO_FRAME:
        return set()
    d = {KO_MODULE}
    # cascade: anything whose dependency chain hits the KO, with a delay
    for step in range(4):
        for m, deps in DEPENDS.items():
            if any(x in d for x in deps) and (t - KO_FRAME) > 3 * (step + 1):
                d.add(m)
    return d


def render(ax, t):
    ax.clear(); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    dead = dead_at(t)
    cell_dead = "cell_division" in dead or (t > KO_FRAME + 22)
    ax.set_facecolor("#1a0b0b" if cell_dead else "#0b0f1a")
    env_col = "#7a2a2a" if cell_dead else "#3a5f8a"
    env = plt.matplotlib.patches.Ellipse((0.52, 0.5), 1.02, 0.96, fill=False,
            edgecolor=env_col, lw=2.5, alpha=0.6)
    ax.add_patch(env)

    for s, d, dashed in FLOWS:
        x1, y1 = POS[s]; x2, y2 = POS[d]
        flux = activity(s, t, dead)
        col = (0.4 + 0.6*flux, 0.85*flux + 0.15, 0.3 + 0.3*(1-flux))
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.6+3*flux,
                                    alpha=0.2+0.7*flux, linestyle="--" if dashed else "-",
                                    shrinkA=14, shrinkB=14, connectionstyle="arc3,rad=0.08"))
    for m in MODS:
        if m not in SIZ: continue
        x, y = POS[m]; a = activity(m, t, dead); r = node_r(m)
        base = np.array(CMAP[m][:3])
        if m in dead:
            ax.scatter([x], [y], s=(r*1100), color="#3a3030", edgecolors="#7a2a2a",
                       linewidths=1.4, zorder=3)
            ax.scatter([x], [y], s=(r*1100)*0.5, marker="x", color="#ff4444", zorder=4)
        else:
            ax.scatter([x], [y], s=(r*1100)*(1+6*a), color=base, alpha=0.18*a+0.04,
                       zorder=2, edgecolors="none")
            face = tuple(np.clip(base*(0.45+0.55*a)+0.45*a, 0, 1))
            ax.scatter([x], [y], s=(r*1100), color=face, edgecolors="white",
                       linewidths=0.4+1.2*a, zorder=3)
        ax.text(x, y - r - 0.025, f"{m}\n{SIZ.get(m,0)}", ha="center", va="top",
                fontsize=7.5, color="#cdd6e6" if m not in dead else "#aa6666",
                fontweight="bold" if a > 0.5 else "normal", zorder=4)

    cyc = int(t / T) + 1
    if t < KO_FRAME:
        phase = ["import","energize","make precursors","transcribe/replicate",
                 "load ribosomes","translate","DIVIDE"][min(int((t%T)/T*7),6)]
        ax.set_title(f"E. coli minimal cell -- LIVE  |  cycle {cyc}  |  phase: {phase}",
                     color="white", fontsize=13, fontweight="bold", pad=10)
    elif not cell_dead:
        ax.set_title(f"KNOCKOUT: ribosomal proteins deleted  |  translation failing...",
                     color="#ffcc44", fontsize=13, fontweight="bold", pad=10)
        ax.scatter([POS[KO_MODULE][0]], [POS[KO_MODULE][1]], s=3000,
                   color="#ff4444", alpha=0.25, zorder=1)
    else:
        ax.set_title("CELL DEAD  --  one essential lost, AND gate flips, "
                     "everything starves", color="#ff5555", fontsize=13,
                     fontweight="bold", pad=10)
        ax.text(0.52, 0.5, "DEAD", ha="center", va="center", color="#ff3333",
                fontsize=46, fontweight="bold", alpha=0.55, zorder=6)


fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("#0b0f1a")
anim = FuncAnimation(fig, lambda i: render(ax, i), frames=N_FRAMES, interval=70)
anim.save(OUT / "cell_die.gif", writer=PillowWriter(fps=15))
print("wrote outputs/orphan/cell_die.gif")
