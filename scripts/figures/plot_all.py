"""Generate all vector PDF figures (Figures 2, 3, 4, 5, 6) into ``paper/figures/``.

Usage:
    python scripts/figures/plot_all.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL_RECON   = "#1d4ed8"   # Royal Blue — ReconRAG
COL_JUDGE   = "#7c3aed"   # Violet — Query-time judge
COL_RECENCY = "#0d9488"   # Teal — Recency prompt
COL_LATEST  = "#d97706"   # Amber — Latest doc
COL_NAIVE   = "#dc2626"   # Crimson/Coral — Naive RAG
COL_STATUS  = "#60a5fa"   # Sky Blue — Status labels only
SLATE_DARK  = "#0f172a"
SLATE_MID   = "#64748b"
SLATE_LIGHT = "#cbd5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#94a3b8",
    "axes.labelcolor": SLATE_DARK,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.color": SLATE_MID,
    "ytick.color": SLATE_MID,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def make_fig2() -> None:
    res = {
        "gpt-oss-120b": {
            "B0": (.33, .51),
            "B1": (.28, .63),
            "B2": (.05, .27),
            "Q":  (.28, .64),
            "ReconRAG": (.16, .66),
        },
        "qwen3.8-27b": {
            "B0": (.32, .51),
            "B1": (.30, .66),
            "B2": (.04, .24),
            "Q":  (.26, .67),
            "ReconRAG": (.04, .68),
        },
    }
    style_map = {
        "B0":       ("Naive RAG",        COL_NAIVE,   "o", 46),
        "B1":       ("Recency prompt",   COL_RECENCY, "s", 48),
        "B2":       ("Latest doc",       COL_LATEST,  "^", 52),
        "Q":        ("Query-time judge", COL_JUDGE,   "D", 46),
        "ReconRAG": ("ReconRAG",         COL_RECON,   "o", 78),
    }
    offsets = {
        "gpt-oss-120b": {
            "B0": (-8, 0, "right"),
            "B1": (-8, -7, "right"),
            "B2": (8, 1, "left"),
            "Q":  (0, 9, "center"),
            "ReconRAG": (-9, -1, "right"),
        },
        "qwen3.8-27b": {
            "B0": (-8, 0, "right"),
            "B1": (-8, -6, "right"),
            "Q":  (-4, 9, "center"),
            "B2": (8, 1, "left"),
            "ReconRAG": (8, 1, "left"),
        },
    }

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.95), sharey=True)

    for ax, (model_name, data) in zip(axs, res.items()):
        ax.axvspan(-0.01, 0.175, ymin=0.60, ymax=1.0, color="#eff6ff", alpha=0.75, zorder=0)
        ax.text(0.005, 0.74, "Optimal region", fontsize=7.0, color=COL_RECON,
                fontstyle="italic", va="top")

        for cond, (stale, acc) in data.items():
            label, col, mk, sz = style_map[cond]
            is_ours = cond == "ReconRAG"
            ax.scatter(
                stale, acc, s=sz, marker=mk, color=col,
                edgecolors=SLATE_DARK if is_ours else "white",
                linewidths=1.0 if is_ours else 0.7,
                zorder=6 if is_ours else 4,
            )
            dx, dy, ha = offsets[model_name][cond]
            ax.annotate(
                label, (stale, acc), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7.6 if is_ours else 7.3,
                color=col if is_ours else SLATE_DARK,
                fontweight="bold" if is_ours else "normal",
                zorder=7,
            )

        ax.set_title(f"Generator: {model_name}", fontweight="bold", loc="left", pad=7)
        ax.set_xlim(-0.01, 0.37)
        ax.set_ylim(0.20, 0.76)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(True, linestyle=":", color=SLATE_LIGHT, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Stale Answers on Current-Rule Questions (↓)")

    axs[0].set_ylabel("Total Accuracy Across All Questions (↑)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_stale_vs_accuracy.pdf", dpi=300)
    plt.close(fig)
    print("✓ fig2_stale_vs_accuracy.pdf generated")


def make_fig3() -> None:
    types = ["Changed", "Unchanged", "Historical"]
    by = {
        "gpt-oss-120b": {
            "Naive RAG":        [.34, .83, .17],
            "Latest doc":       [.83, .11, .01],
            "Recency prompt":   [.51, .82, .47],
            "Query-time judge": [.51, .85, .47],
            "ReconRAG":         [.57, .83, .51],
        },
        "qwen3.8-27b": {
            "Naive RAG":        [.37, .84, .13],
            "Latest doc":       [.87, .03, .01],
            "Recency prompt":   [.54, .84, .50],
            "Query-time judge": [.57, .85, .51],
            "ReconRAG":         [.67, .84, .47],
        },
    }

    colors = {
        "Naive RAG":        "#94a3b8",
        "Latest doc":       "#f59e0b",
        "Recency prompt":   "#14b8a6",
        "Query-time judge": "#8b5cf6",
        "ReconRAG":         COL_RECON,
    }

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.95), sharey=True)

    for ax, (model_name, data) in zip(axs, by.items()):
        n_bars = len(data)
        width = 0.80 / n_bars
        indices = range(len(types))

        for i, (cond_name, vals) in enumerate(data.items()):
            xs = [x + (i - (n_bars - 1) / 2) * width for x in indices]
            is_ours = cond_name == "ReconRAG"
            ax.bar(
                xs, vals, width=width * 0.90, color=colors[cond_name],
                label=cond_name, zorder=4 if is_ours else 3,
                edgecolor=SLATE_DARK if is_ours else "white",
                linewidth=0.8 if is_ours else 0.4,
            )
            for bx, v in zip(xs, vals):
                if v >= 0.08:
                    ax.text(
                        bx, v + 0.018, f"{int(round(v * 100))}",
                        ha="center", va="bottom", fontsize=6.0,
                        color=COL_RECON if is_ours else "#475569",
                        fontweight="bold" if is_ours else "normal",
                    )

        ax.set_xticks(list(indices))
        ax.set_xticklabels(types, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", linestyle=":", color=SLATE_LIGHT, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"Generator: {model_name}", fontweight="bold", loc="left", pad=6)

    axs[0].set_ylabel("Accuracy (Correct Answers, %)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=True, facecolor="white", edgecolor="#e2e8f0",
        fontsize=7.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig3_by_question_type.pdf", dpi=300)
    plt.close(fig)
    print("✓ fig3_by_question_type.pdf generated")


def make_fig4() -> None:
    refs = ["Three-model consensus", "Claude 3.5 Sonnet", "Gemini 3.5 Flash Lite*", "GPT-OSS 120B"]
    pp = [.767, .649, .704, .688]
    gl = [.808, .695, .745, .695]
    l3 = [.842, .755, .773, .716]

    fig, ax = plt.subplots(figsize=(5.5, 2.4))

    for y, (p, g, l) in enumerate(zip(pp, gl, l3)):
        ax.plot([p, l], [y, y], color="#cbd5e1", lw=2.2, zorder=1)
        ax.scatter(p, y, color="#94a3b8", s=34, edgecolors="#475569", linewidths=0.6,
                   zorder=2, label="Per-provision matching" if y == 0 else None)
        ax.scatter(g, y, color=COL_RECENCY, s=38, edgecolors="#0f172a", linewidths=0.6,
                   zorder=3, label="Global ILP matching" if y == 0 else None)
        ax.scatter(l, y, color=COL_RECON, s=46, edgecolors="#0f172a", linewidths=0.8,
                   zorder=4, label="ReconRAG" if y == 0 else None)
        ax.text(p - 0.008, y, f"{p:.3f}", ha="right", va="center", fontsize=6.8, color="#64748b")
        ax.text(l + 0.008, y, f"{l:.3f}", ha="left", va="center", fontsize=7.0, color=COL_RECON, fontweight="bold")

    ax.set_yticks(range(len(refs)))
    ax.set_yticklabels(refs, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 on Full Replacements")
    ax.set_xlim(0.61, 0.88)
    ax.grid(axis="x", linestyle=":", color=SLATE_LIGHT, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=6.7,
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, columnspacing=0.6,
              handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_replacement_robustness.pdf", dpi=300)
    plt.close(fig)
    print("✓ fig4_replacement_robustness.pdf generated")


def make_fig5() -> None:
    conds = ["Unreconciled RAG", "Latest doc", "Status labels only", "ReconRAG"]
    present = [.643, .971, .643, .786]
    unmarked = [.786, .043, .100, .114]
    unch = [.919, .185, .911, .911]

    # Consistent method colors across all panels
    col_methods = ["#94a3b8", "#f59e0b", "#14b8a6", "#1d4ed8"]

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.15), sharey=True)
    datasets = [
        (present,  "Changed Questions\nIn-force text retrieved"),
        (unmarked, "Changed Questions\nUnmarked stale text"),
        (unch,     "Unchanged Controls\nActive provision kept"),
    ]

    for ax, (vals, title) in zip(axs, datasets):
        for i, (v, c) in enumerate(zip(vals, col_methods)):
            ax.barh(i, v, color=c, edgecolor="#0f172a" if i == 3 else c,
                    linewidth=0.8 if i == 3 else 0, height=0.62, zorder=3)
            fontw = "bold" if i == 3 else "normal"
            ax.text(v + 0.02, i, f"{v:.1%}", va="center", fontsize=7.2, color=SLATE_DARK, fontweight=fontw)

        ax.set_xlim(0, 1.18)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=8.0, loc="left", fontweight="bold", pad=6)
        ax.grid(axis="x", linestyle=":", color=SLATE_LIGHT, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

    axs[0].set_yticks(range(len(conds)))
    axs[0].set_yticklabels(conds, fontweight="bold")
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig5_retrieval.pdf", dpi=300)
    plt.close(fig)
    print("✓ fig5_retrieval.pdf generated")


def make_fig6() -> None:
    conds = ["Naive RAG", "Recency prompt", "Latest doc", "Query-time judge", "ReconRAG"]
    acc = {
        "gpt-oss-120b": [(.51, .45, .57), (.63, .56, .69), (.27, .24, .30),
                         (.64, .58, .70), (.66, .61, .71)],
        "qwen3.8-27b":  [(.51, .45, .58), (.66, .59, .72), (.24, .22, .27),
                         (.67, .60, .73), (.68, .62, .75)],
    }
    stale = {
        "gpt-oss-120b": [(.33, .22, .47), (.28, .18, .40), (.05, .01, .12),
                         (.28, .17, .42), (.16, .09, .24)],
        "qwen3.8-27b":  [(.32, .21, .45), (.30, .20, .42), (.04, .00, .09),
                         (.26, .16, .40), (.04, .00, .08)],
    }
    row_colors = [COL_NAIVE, COL_RECENCY, COL_LATEST, COL_JUDGE, COL_RECON]
    markers = {"gpt-oss-120b": ("o", -0.16), "qwen3.8-27b": ("s", 0.16)}

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
    panels = [
        (axs[0], acc,   "Total accuracy (↑ better)"),
        (axs[1], stale, "Stale rate on current rules (↓ better)"),
    ]
    for ax, data, title in panels:
        ax.axhspan(3.5, 4.5, color="#eff6ff", zorder=0)
        for model, vals in data.items():
            mk, off = markers[model]
            for i, (p, lo, hi) in enumerate(vals):
                ours = i == 4
                col = row_colors[i]
                ax.errorbar(
                    p, i + off, xerr=[[p - lo], [hi - p]], fmt=mk,
                    ms=4.8 if ours else 4.0,
                    color=col, mfc=col if model == "gpt-oss-120b" else "white", mec=col,
                    ecolor=col, elinewidth=1.1, capsize=2.2, zorder=3,
                )
                ax.text(
                    hi + 0.012, i + off, f"{p:.2f}", va="center", fontsize=6.5,
                    color=COL_RECON if ours else "#334155",
                    fontweight="bold" if ours else "normal",
                )
        ax.set_title(title, fontsize=8.2, fontweight="bold", loc="left", pad=6)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", linestyle=":", color=SLATE_LIGHT, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
    axs[0].set_xlim(0.18, 0.83)
    axs[1].set_xlim(-0.01, 0.55)
    axs[0].set_yticks(range(len(conds)))
    axs[0].set_yticklabels(conds)
    for t in axs[0].get_yticklabels():
        if t.get_text() == "ReconRAG":
            t.set_fontweight("bold")
            t.set_color(COL_RECON)
    axs[0].invert_yaxis()
    h = [
        Line2D([], [], marker="o", ls="", color=SLATE_DARK, mfc=SLATE_DARK, ms=4.2, label="gpt-oss-120b (filled circle)"),
        Line2D([], [], marker="s", ls="", color=SLATE_DARK, mfc="white", ms=4.2, label="qwen3.8-27b (hollow square)"),
        Line2D([], [], color=SLATE_DARK, lw=1.1, label="95% family-cluster bootstrap CI"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.53, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig6_accuracy_ci.pdf", dpi=300)
    plt.close(fig)
    print("✓ fig6_accuracy_ci.pdf generated")


if __name__ == "__main__":
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    print("All paper figures generated into paper/figures/")
