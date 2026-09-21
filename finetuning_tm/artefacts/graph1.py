"""
Rank-Scaling Line Plots for LoRA target modules (All / mlp / attn)
across 6 metrics: Precision, Recall, F1, IoU, clDice, APLS.

X-axis: LoRA Rank (4, 16, 64)
Y-axis: metric score
Shaded ribbon: +/- std/variance value from the table
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data transcribed from Table 2 (TerraMind base-sized backbone, ROSA dataset)
# ---------------------------------------------------------------------------
ranks = [4, 16, 64]

data = {
    "All": {
        "Precision": {"mean": [0.4904, 0.5112, 0.5205], "err": [0.0024, 0.0085, 0.0025]},
        "Recall":    {"mean": [0.5757, 0.5797, 0.5773], "err": [0.0044, 0.0044, 0.0036]},
        "F1":        {"mean": [0.5296, 0.5433, 0.5474], "err": [0.0018, 0.0029, 0.0023]},
        "IoU":       {"mean": [0.3602, 0.3729, 0.3769], "err": [0.0017, 0.0027, 0.0021]},
        "clDice":    {"mean": [0.4362, 0.4493, 0.4545], "err": [0.0013, 0.0049, 0.0038]},
        "APLS":      {"mean": [0.2340, 0.2505, 0.2509], "err": [0.0034, 0.0043, 0.0036]},
    },
    "mlp": {
        "Precision": {"mean": [0.4766, 0.4989, 0.4995], "err": [0.0025, 0.0006, 0.0081]},
        "Recall":    {"mean": [0.5810, 0.5834, 0.5897], "err": [0.0061, 0.0058, 0.0080]},
        "F1":        {"mean": [0.5236, 0.5379, 0.5408], "err": [0.0032, 0.0028, 0.0015]},
        "IoU":       {"mean": [0.3547, 0.3679, 0.3706], "err": [0.0030, 0.0026, 0.0014]},
        "clDice":    {"mean": [0.4259, 0.4393, 0.4450], "err": [0.0021, 0.0033, 0.0012]},
        "APLS":      {"mean": [0.2294, 0.2403, 0.2455], "err": [0.0081, 0.0025, 0.0006]},
    },
    "attn": {
        "Precision": {"mean": [0.4554, 0.4729, 0.4704], "err": [0.0060, 0.0039, 0.0026]},
        "Recall":    {"mean": [0.5690, 0.5687, 0.5697], "err": [0.0089, 0.0020, 0.0181]},
        "F1":        {"mean": [0.5058, 0.5164, 0.5152], "err": [0.0015, 0.0017, 0.0073]},
        "IoU":       {"mean": [0.3386, 0.3481, 0.3470], "err": [0.0014, 0.0016, 0.0066]},
        "clDice":    {"mean": [0.4113, 0.4202, 0.4205], "err": [0.0023, 0.0032, 0.0019]},
        "APLS":      {"mean": [0.2131, 0.2246, 0.2207], "err": [0.0070, 0.0046, 0.0089]},
    },
}

full_finetune = {
    "Precision": {"mean": 0.5684, "err": 0.0046},
    "Recall":    {"mean": 0.5811, "err": 0.0056},
    "F1":        {"mean": 0.5747, "err": 0.0017},
    "IoU":       {"mean": 0.4032, "err": 0.0016},
    "clDice":    {"mean": 0.4898, "err": 0.0031},
    "APLS":      {"mean": 0.2909, "err": 0.0012},
}

metrics = ["Precision", "Recall", "F1", "IoU", "clDice", "APLS"]
targets = ["All", "mlp", "attn"]
colors = {"All": "#0072B2", "mlp": "#E69F00", "attn": "#CC79A7"}

# ---------------------------------------------------------------------------
# Plot 1
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 6, figsize=(14, 4))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    for target in targets:
        mean = np.array(data[target][metric]["mean"])
        err = np.array(data[target][metric]["err"])
        color = colors[target]

        ax.plot(ranks, mean, marker="o", markersize=1.5, label=target, color=color, linewidth=1)
        ax.fill_between(ranks, mean - err, mean + err, color=color, alpha=0.2)

    # --- full fine-tuning reference line ---
    ft_mean = full_finetune[metric]["mean"]
    ft_err = full_finetune[metric]["err"]

    ax.axhline(
        ft_mean, color="0.6", linestyle="--", linewidth=1.5,
        label="Full fine-tuning"
    )
    ax.axhspan(
        ft_mean - ft_err, ft_mean + ft_err,
        color="grey", alpha=0.12  # lighter than the LoRA ribbons
    )

    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xticks(ranks)
    # ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.2, 0.6)

    for spine in ax.spines.values():
        spine.set_color("0.8")
        spine.set_linewidth(0.8)
    

# Single shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=11, frameon=False)
fig.suptitle(
    "TerraMind and Linear Decoder: LoRA Performance vs. Rank by Target Module",
    fontsize=16, y=1.06
)
fig.supxlabel("LoRA Rank", fontsize=12)
fig.supylabel("Score", fontsize=12)

plt.tight_layout()
plt.savefig("/Users/catherineli/Desktop/instaroad/InstaRoadPrototype/src/finetuning_tm/artefacts/lora_rank_scaling.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Plot 2
# ---------------------------------------------------------------------------
# # Facet on the 4 key metrics: F1, IoU, clDice, APLS
# facet_metrics = ["F1", "IoU", "clDice", "APLS"]
# ranks = [4, 16, 64]
# bar_colors = {4: "#1f77b4", 16: "#ff7f0e", 64: "#2ca02c"}  # color by rank instead of target

# x = np.arange(len(targets))  # group positions: All, mlp, attn
# width = 0.25  # width of each bar within a group

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# axes = axes.flatten()

# for ax, metric in zip(axes, facet_metrics):
#     for i, rank in enumerate(ranks):
#         means = [data[target][metric]["mean"][i] for target in targets]
#         errs = [data[target][metric]["err"][i] for target in targets]
#         offset = (i - 1) * width  # center the group of 3 bars

#         ax.bar(
#             x + offset, means, width,
#             yerr=errs, capsize=4,
#             label=f"Rank {rank}", color=bar_colors[rank],
#             edgecolor="black", linewidth=0.5,
#         )

#     ax.set_title(metric, fontsize=12, fontweight="bold")
#     ax.set_xticks(x)
#     ax.set_xticklabels(targets)
#     ax.set_ylabel("Score")
#     ax.set_ylim(0, 0.6)
#     ax.grid(axis="y", alpha=0.3)

# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=11)

# fig.suptitle(
#     "LoRA Target Module Comparison by Rank\n(TerraMind base, ROSA dataset)",
#     fontsize=14, y=1.07
# )

# plt.tight_layout()
# plt.savefig("/Users/catherineli/Desktop/instaroad/InstaRoadPrototype/src/finetuning_tm/artefacts/lora_rank_scaling2.png", dpi=200, bbox_inches="tight")
# plt.show()