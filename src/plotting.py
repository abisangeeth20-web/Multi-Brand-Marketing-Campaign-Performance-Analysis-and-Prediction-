"""Shared matplotlib style: recessive grid/axes, thin marks, text in ink colours (never series colours)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": C.INK["surface"], "axes.facecolor": C.INK["surface"],
        "savefig.facecolor": C.INK["surface"], "figure.dpi": 110, "savefig.dpi": 130,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": C.INK["primary"], "axes.labelcolor": C.INK["secondary"],
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": C.INK["primary"], "axes.edgecolor": C.INK["axis"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C.INK["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": C.INK["muted"], "ytick.color": C.INK["muted"],
        "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, name: str):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def inr(x, _=None):
    """Compact Indian-rupee axis formatter (₹ L = lakh, ₹ Cr = crore)."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.1f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.1f} L"
    if abs(x) >= 1e3:
        return f"₹{x / 1e3:.0f}K"
    return f"₹{x:.0f}"
