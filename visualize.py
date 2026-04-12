"""
visualize.py  —  Generate all comparison charts for thesis results.

Charts produced (saved to charts/):
  1. chart1_accuracy.png          — Bar: macro-avg accuracy per model
  2. chart2_accuracy_per_ds.png   — Grouped bar: accuracy per model × dataset
  3. chart3_cost_vs_accuracy.png  — Scatter: avg cost/query vs macro-avg accuracy (Pareto)
  4. chart4_reward.png            — Bar: macro-avg economic reward per model
  5. chart5_reward_vs_lambda_error.png   — Lines: reward sensitivity to λ_error
  6. chart6_reward_vs_lambda_latency.png — Lines: reward sensitivity to λ_latency
  7. chart7_best_model_heatmap.png       — Heatmap: best model by (λ_error, λ_latency)
  8. chart8_best_model_heatmap_per_dataset.png — Heatmap: best standalone model per dataset by (λ_error, λ_latency)

Cascade charts (C prefix):
  C1. chartC1_cascade_combined_overview.png        — Escalation rate & accuracy overview
  C2. chartC2_cascade_dual_heatmap.png             — Reward & cost heatmap
  C3. chartC3_cascade_best_config_heatmap.png      — Heatmap: best option (cascade OR standalone) by (λ_error, λ_latency)
  C4. chartC4_cascade_reward_vs_lambda_error.png   — 2×2: penalty vs λ_error per config pair
  C5. chartC5_cascade_reward_vs_lambda_latency.png — 2×2: penalty vs λ_latency per config pair
  C6. chartC6_cascade_per_dataset_decision_map.png — Per-dataset decision maps (5 subplots)

Self-consistency charts (SC prefix):
  SC1. chartSC1_selfcons_overview.png                  — Accuracy + vote agreement rate per dataset
  SC2. chartSC2_selfcons_dual_heatmap.png              — Reward & cost heatmap (SC configs vs standalone)
  SC3. chartSC3_selfcons_best_config_heatmap.png       — Heatmap: best option (SC OR standalone) by (λ_error, λ_latency)
  SC4. chartSC4_selfcons_reward_vs_lambda_error.png    — Penalty vs λ_error: SC vs base model
  SC6. chartSC6_selfcons_accuracy_comparison.png        — Grouped bar: accuracy per dataset, SC vs standalone

Router charts (R prefix):
  R1.  chartR1_router_overview.png                — Accuracy + routing split per dataset
  R2.  chartR2_router_dual_heatmap.png            — Reward & cost heatmap (router configs vs standalone)
  R3.  chartR3_router_best_config_heatmap.png     — Heatmap: best option (router OR standalone) by (λ_error, λ_latency)
  R4.  chartR4_router_reward_vs_lambda_error.png  — Penalty vs λ_error: router vs small/large standalone
  R5.  chartR5_router_reward_vs_lambda_latency.png — Penalty vs λ_latency: router vs small/large standalone
  R6.  chartR6_router_per_dataset_decision_map.png — Per-dataset decision maps (5 subplots)
"""

import json
import pathlib
import re
import textwrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

# ---------------------------------------------------------------------------
# Config — must match tester.py
# ---------------------------------------------------------------------------
LAMBDA_LATENCY_DEFAULT = 0.01
LAMBDA_ERROR_DEFAULT   = 1.0

DATASETS = ["humaneval", "mbpp", "mmlu_pro", "gpqa", "gsm8k"]
DS_LABELS = {"humaneval": "HumanEval", "mbpp": "MBPP",
             "mmlu_pro": "MMLU-Pro", "gpqa": "GPQA", "gsm8k": "GSM8K"}

RESULTS_DIR     = pathlib.Path("results")
OPT_RESULTS_DIR = pathlib.Path("optimization_results")
CHARTS_DIR      = pathlib.Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

# Reasoning vs non-reasoning classification (have reasoning_effort set)
REASONING_MODELS = {"o3-mini", "gpt-5.4", "gpt-5.4-pro"}
REASONING_HATCH  = "////"

# ---------------------------------------------------------------------------
# Heatmap grid ranges — edit these to change the λ axes on ALL heatmaps
# ---------------------------------------------------------------------------
# chart7: base-model best-config heatmap
HEATMAP7_LAM_ERROR = np.linspace(0.1,   25.0,  125)   # x-axis: λ_error
HEATMAP7_LAM_LAT   = np.linspace(0.001, 0.5,  25)   # y-axis: λ_latency

# C3/C6, SC3/SC5, R3/R6 decision-map heatmaps
HEATMAP_LAM_ERROR  = np.linspace(0.01,  25.0,  125)   # x-axis: λ_error
HEATMAP_LAM_LAT    = np.linspace(0.0001, 0.5, 50)   # y-axis: λ_latency


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _find_model_folders() -> dict[str, pathlib.Path]:
    """Return {model_name: latest_folder}."""
    model_latest: dict = {}
    for folder in RESULTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        if len(name) > 16 and name[8] == "_" and name[15] == "_":
            model = name[16:]
            prev = model_latest.get(model)
            if prev is None or folder.name > prev.name:
                model_latest[model] = folder
    return model_latest


def _load_records(folder: pathlib.Path, stem: str) -> list[dict]:
    path = folder / f"{stem}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("type") == "dataset_summary":
                    continue
                records.append(rec)
    return records


def compute_reward(cost: float, latency: float, is_correct: int,
                   lam_e: float, lam_l: float) -> float:
    return -(cost + lam_l * latency + lam_e * (1 - is_correct))


# ---------------------------------------------------------------------------
# Vectorised heatmap helpers
# reward(lam_e, lam_l) = intercept + coeff_e * lam_e + coeff_l * lam_l
# is linear, so we precompute the 3 scalars per (option, dataset) once and
# use NumPy broadcasting instead of a Python double loop over the λ grid.
# ---------------------------------------------------------------------------

def _option_stats_macro(records_per_dataset: dict):
    """
    Returns (intercept, coeff_e, coeff_l) for macro-avg reward over DATASETS,
    or None if no dataset has records.
      intercept = mean_d(-mean_cost_d)
      coeff_e   = mean_d(accuracy_d - 1)
      coeff_l   = mean_d(-mean_elapsed_d)
    """
    ds_i, ds_e, ds_l = [], [], []
    for stem in DATASETS:
        recs = records_per_dataset.get(stem, [])
        if not recs:
            continue
        costs   = np.array([r.get("cost_usd",  0.0) or 0.0 for r in recs], dtype=np.float64)
        elapsed = np.array([r.get("elapsed_s", 0.0) or 0.0 for r in recs], dtype=np.float64)
        correct = np.array([float(r.get("is_correct", 0))   for r in recs], dtype=np.float64)
        ds_i.append(-costs.mean())
        ds_e.append(correct.mean() - 1.0)
        ds_l.append(-elapsed.mean())
    if not ds_i:
        return None
    return (float(np.mean(ds_i)), float(np.mean(ds_e)), float(np.mean(ds_l)))


def _option_stats_macro_sc(records_per_dataset: dict, base_lats: dict):
    """Like _option_stats_macro but replaces elapsed_s with base_lats[stem]."""
    ds_i, ds_e, ds_l = [], [], []
    for stem in DATASETS:
        recs = records_per_dataset.get(stem, [])
        if not recs:
            continue
        base_lat = base_lats.get(stem, recs[0].get("elapsed_s", 0.0) or 0.0)
        costs   = np.array([r.get("cost_usd", 0.0) or 0.0 for r in recs], dtype=np.float64)
        correct = np.array([float(r.get("is_correct", 0))  for r in recs], dtype=np.float64)
        ds_i.append(-costs.mean())
        ds_e.append(correct.mean() - 1.0)
        ds_l.append(-float(base_lat))
    if not ds_i:
        return None
    return (float(np.mean(ds_i)), float(np.mean(ds_e)), float(np.mean(ds_l)))


def _option_stats_single(records: list, elapsed_override=None):
    """
    Precompute linear coefficients for a single-dataset reward.
    elapsed_override: scalar to use instead of per-record elapsed_s.
    Returns None if records is empty.
    """
    if not records:
        return None
    costs   = np.array([r.get("cost_usd",  0.0) or 0.0 for r in records], dtype=np.float64)
    correct = np.array([float(r.get("is_correct", 0))   for r in records], dtype=np.float64)
    if elapsed_override is not None:
        mean_lat = float(elapsed_override)
    else:
        elapsed = np.array([r.get("elapsed_s", 0.0) or 0.0 for r in records], dtype=np.float64)
        mean_lat = float(elapsed.mean())
    return (-float(costs.mean()), float(correct.mean()) - 1.0, -mean_lat)


def _build_best_grid(options: list, lam_errors: np.ndarray, lam_lats: np.ndarray) -> np.ndarray:
    """
    Vectorised best-option grid.
    options: [(key, intercept, coeff_e, coeff_l), ...]
    Returns object array of shape (len(lam_lats), len(lam_errors)).
    """
    n = len(options)
    if n == 0:
        return np.full((len(lam_lats), len(lam_errors)), None, dtype=object)
    # reward_stack shape: (n_options, L, E)
    reward_stack = np.empty((n, len(lam_lats), len(lam_errors)), dtype=np.float64)
    for idx, (_, intercept, coeff_e, coeff_l) in enumerate(options):
        reward_stack[idx] = (
            intercept
            + coeff_e * lam_errors[np.newaxis, :]
            + coeff_l * lam_lats[:, np.newaxis]
        )
    best_idx = np.argmax(reward_stack, axis=0)   # (L, E)
    # Build 1-D object array of keys, then index it with best_idx
    keys_arr = np.empty(n, dtype=object)
    for i, (key, *_) in enumerate(options):
        keys_arr[i] = key
    grid = keys_arr[best_idx]   # fancy indexing → (L, E) object array
    return grid


def build_model_stats(lam_e: float = LAMBDA_ERROR_DEFAULT,
                      lam_l: float = LAMBDA_LATENCY_DEFAULT) -> dict:
    """
    Returns {model: {
        "macro_accuracy":  float,        # macro-avg pass rate across datasets
        "macro_reward":    float,        # macro-avg reward
        "avg_cost":        float,        # macro-avg cost per query
        "avg_latency":     float,        # macro-avg latency per query
        "ds_accuracy":     {ds: float},  # per-dataset accuracy
        "ds_reward":       {ds: float},  # per-dataset reward
    }}
    """
    model_folders = _find_model_folders()
    stats: dict = {}

    for model, folder in model_folders.items():
        ds_acc:     list[float] = []
        ds_rew:     list[float] = []
        ds_cost:    list[float] = []
        ds_lat:     list[float] = []
        ds_accuracy_map: dict  = {}
        ds_reward_map:   dict  = {}

        for stem in DATASETS:
            records = _load_records(folder, stem)
            if not records:
                continue

            acc     = sum(r.get("is_correct", 0) for r in records) / len(records)
            costs   = [r.get("cost_usd", 0.0) or 0.0 for r in records]
            lats    = [r.get("elapsed_s", 0.0) or 0.0 for r in records]
            rewards = [
                compute_reward(c, l, r.get("is_correct", 0), lam_e, lam_l)
                for r, c, l in zip(records, costs, lats)
            ]

            ds_acc.append(acc)
            ds_rew.append(sum(rewards) / len(rewards))
            ds_cost.append(sum(costs) / len(costs))
            ds_lat.append(sum(lats) / len(lats))
            ds_accuracy_map[DS_LABELS[stem]] = acc
            ds_reward_map[DS_LABELS[stem]]   = sum(rewards) / len(rewards)

        if not ds_acc:
            continue

        stats[model] = {
            "macro_accuracy": sum(ds_acc) / len(ds_acc),
            "macro_reward":   sum(ds_rew) / len(ds_rew),
            "avg_cost":       sum(ds_cost) / len(ds_cost),
            "avg_latency":    sum(ds_lat) / len(ds_lat),
            "ds_accuracy":    ds_accuracy_map,
            "ds_reward":      ds_reward_map,
        }

    return stats


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

# ColorBrewer Paired palette — family-grouped by model tier:
# gpt-4.1 (dark blue / light blue), gpt-5.4 (dark green / light green),
# gpt-5.4-pro (deep purple), o3-mini (vivid orange)
MODEL_COLORS = [
    "#1F78B4",  # gpt-4.1:      rich blue
    "#A6CEE3",  # gpt-4.1-mini: light blue
    "#33A02C",  # gpt-5.4:      forest green
    "#B2DF8A",  # gpt-5.4-mini: light green
    "#6A3D9A",  # gpt-5.4-pro:  deep purple
    "#FF7F00",  # o3-mini:      vivid orange
]

def _model_color_map(models: list[str]) -> dict[str, str]:
    return {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}


def _save(fig, name: str):
    path = CHARTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Chart 1 — Macro-avg accuracy per model (bar)
# ---------------------------------------------------------------------------

def chart1_accuracy(stats: dict):
    models = sorted(stats)
    accs   = [stats[m]["macro_accuracy"] * 100 for m in models]
    cmap   = _model_color_map(models)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (m, a) in enumerate(zip(models, accs)):
        hatch = REASONING_HATCH if m in REASONING_MODELS else None
        bar = ax.bar(m, a, color=cmap[m], hatch=hatch, width=0.55, zorder=3,
                     edgecolor="white" if hatch is None else "#333333", linewidth=0.8)
        ax.bar_label(bar, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_ylabel("Avg Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Legend distinguishing reasoning models
    legend_patches = [
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="#333333",
                       hatch=REASONING_HATCH, label="Reasoning model"),
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", label="Standard model"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, framealpha=0.85,
              loc="lower center", bbox_to_anchor=(0.5, 1),
              ncol=2, borderaxespad=0)
    fig.tight_layout()
    _save(fig, "chart1_accuracy.png")


# ---------------------------------------------------------------------------
# Chart 2 — Accuracy per model × dataset (grouped bar)
# ---------------------------------------------------------------------------

def chart2_accuracy_per_dataset(stats: dict):
    models   = sorted(stats)
    datasets = list(DS_LABELS.values())
    n_m, n_d = len(models), len(datasets)
    x        = np.arange(n_d)
    width    = 0.8 / n_m
    cmap     = _model_color_map(models)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(models):
        accs = [stats[model]["ds_accuracy"].get(ds, 0) * 100 for ds in datasets]
        offset = (i - n_m / 2 + 0.5) * width
        hatch = REASONING_HATCH if model in REASONING_MODELS else None
        ax.bar(x + offset, accs, width=width * 0.9,
               color=cmap[model], hatch=hatch, label=model,
               edgecolor="white" if hatch is None else "#333333",
               linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Model colour legend
    model_handles, model_labels = ax.get_legend_handles_labels()
    ax.legend(handles=model_handles, labels=model_labels,
              fontsize=10, framealpha=0.85, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    _save(fig, "chart2_accuracy_per_dataset.png")


# ---------------------------------------------------------------------------
# Chart 3 — Avg cost/query vs macro-avg accuracy (scatter + Pareto frontier)
# ---------------------------------------------------------------------------

def chart3_cost_vs_accuracy(stats: dict):
    models = sorted(stats)
    costs  = [stats[m]["avg_cost"] * 1000 for m in models]   # convert to milli-$
    accs   = [stats[m]["macro_accuracy"] * 100 for m in models]
    cmap   = _model_color_map(models)

    # Pareto frontier (higher accuracy & lower cost dominates)
    pts = sorted(zip(costs, accs, models))
    pareto: list = []
    best_acc = -1.0
    for c, a, m in reversed(pts):  # iterate from highest cost
        if a > best_acc:
            best_acc = a
            pareto.append((c, a))
    pareto.sort()

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color="gray", lw=1.2, label="Pareto frontier", zorder=2)

    for m, c, a in zip(models, costs, accs):
        ax.scatter(c, a, color=cmap[m], s=90, zorder=4)
        ax.annotate(m, (c, a), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Avg Cost per Query (m$)")
    ax.set_ylabel("Avg Accuracy (%)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "chart3_cost_vs_accuracy.png")


# ---------------------------------------------------------------------------
# Chart 4 — Macro-avg economic reward per model (bar)
# ---------------------------------------------------------------------------

def chart4_reward(stats: dict):
    models   = sorted(stats)
    penalties = [-stats[m]["macro_reward"] for m in models]  # negate: lower penalty = better
    cmap     = _model_color_map(models)

    fig, ax = plt.subplots(figsize=(8, 6))
    for m, p in zip(models, penalties):
        hatch = REASONING_HATCH if m in REASONING_MODELS else None
        bar = ax.bar(m, p, color=cmap[m], hatch=hatch, width=0.55, zorder=3,
                     edgecolor="white" if hatch is None else "#333333", linewidth=0.8)
        ax.bar_label(bar, fmt="%.4f", padding=4, fontsize=9)
    ax.set_ylabel("Economic Penalty  (lower = better)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    legend_patches = [
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="#333333",
                       hatch=REASONING_HATCH, label="Reasoning model"),
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", label="Standard model"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, framealpha=0.85,
              loc="lower center", bbox_to_anchor=(0.5, 1.09),
              ncol=2, borderaxespad=0)
    fig.tight_layout()
    _save(fig, "chart4_reward.png")


# ---------------------------------------------------------------------------
# Chart 5 — Reward vs λ_error (one line per model)
# ---------------------------------------------------------------------------

def chart5_reward_vs_lambda_error(model_folders: dict):
    lambda_errors = np.linspace(0.1, 20.0, 160)
    models = sorted(model_folders)
    cmap   = _model_color_map(models)

    max_penalty = 8.0

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        folder = model_folders[model]
        is_reasoning = model in REASONING_MODELS
        ls = "--" if is_reasoning else "-"
        rewards = []
        lam_vals_used = []
        for lam_e in lambda_errors:
            ds_avgs = []
            for stem in DATASETS:
                records = _load_records(folder, stem)
                if not records:
                    continue
                rews = [
                    compute_reward(
                        r.get("cost_usd", 0.0) or 0.0,
                        r.get("elapsed_s", 0.0) or 0.0,
                        r.get("is_correct", 0),
                        lam_e, LAMBDA_LATENCY_DEFAULT
                    ) for r in records
                ]
                ds_avgs.append(sum(rews) / len(rews))
            val = sum(ds_avgs) / len(ds_avgs) if ds_avgs else 0.0
            rewards.append(val)
            lam_vals_used.append(lam_e)
            if -val >= max_penalty:
                break
        ax.plot(lam_vals_used, [-r for r in rewards], label=model,
                color=cmap[model], lw=2.3, linestyle=ls)

    ax.axvline(LAMBDA_ERROR_DEFAULT, color="gray", linestyle=":", lw=1, label=f"default λ_error={LAMBDA_ERROR_DEFAULT}")
    ax.set_xlabel("λ_error  (error penalty weight)")
    ax.set_ylabel("Economic Penalty  (lower = better)")
    # Build legend with model lines + reasoning/standard type indicators
    handles, labels = ax.get_legend_handles_labels()
    type_handles = [
        plt.Line2D([0], [0], color="gray", lw=2, linestyle="--", label="Reasoning model"),
        plt.Line2D([0], [0], color="gray", lw=2, linestyle="-",  label="Standard model"),
    ]
    ax.legend(handles=handles + type_handles, labels=labels + ["Reasoning model", "Standard model"],
              fontsize=8, framealpha=0.8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "chart5_reward_vs_lambda_error.png")


# ---------------------------------------------------------------------------
# Chart 6 — Reward vs λ_latency (one line per model)
# ---------------------------------------------------------------------------

def chart6_reward_vs_lambda_latency(model_folders: dict):
    lambda_lats = np.linspace(0.001, 0.5, 40)
    models = sorted(model_folders)
    cmap   = _model_color_map(models)

    # Custom y-axis: equally spaced visually, non-uniform in value
    ytick_vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
    ytick_pos  = list(range(len(ytick_vals)))          # 0..8, equal spacing
    def to_pos(v):
        return float(np.interp(v, ytick_vals, ytick_pos))

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        folder = model_folders[model]
        is_reasoning = model in REASONING_MODELS
        ls = "--" if is_reasoning else "-"
        rewards = []
        lam_vals_used = []
        for lam_l in lambda_lats:
            ds_avgs = []
            for stem in DATASETS:
                records = _load_records(folder, stem)
                if not records:
                    continue
                rews = [
                    compute_reward(
                        r.get("cost_usd", 0.0) or 0.0,
                        r.get("elapsed_s", 0.0) or 0.0,
                        r.get("is_correct", 0),
                        LAMBDA_ERROR_DEFAULT, lam_l
                    ) for r in records
                ]
                ds_avgs.append(sum(rews) / len(rews))
            val = sum(ds_avgs) / len(ds_avgs) if ds_avgs else 0.0
            rewards.append(val)
            lam_vals_used.append(lam_l)
            if -val >= max(ytick_vals):
                break
        ax.plot(lam_vals_used, [to_pos(-r) for r in rewards], label=model,
                color=cmap[model], lw=2.3, linestyle=ls)

    ax.axvline(LAMBDA_LATENCY_DEFAULT, color="gray", linestyle=":", lw=1, label=f"default λ_latency={LAMBDA_LATENCY_DEFAULT}")
    ax.set_xlabel("λ_latency  (latency penalty weight)")
    ax.set_ylabel("Economic Penalty  (lower = better)")
    # Build legend with model lines + reasoning/standard type indicators
    handles, labels = ax.get_legend_handles_labels()
    type_handles = [
        plt.Line2D([0], [0], color="gray", lw=2, linestyle="--", label="Reasoning model"),
        plt.Line2D([0], [0], color="gray", lw=2, linestyle="-",  label="Standard model"),
    ]
    ax.legend(handles=handles + type_handles, labels=labels + ["Reasoning model", "Standard model"],
              fontsize=8, framealpha=0.8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_vals)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "chart6_reward_vs_lambda_latency.png")


# ---------------------------------------------------------------------------
# Chart 7 — Best model heatmap over (λ_error × λ_latency)
# ---------------------------------------------------------------------------

def chart7_best_model_heatmap(model_folders: dict):
    models     = sorted(model_folders)
    lam_errors = HEATMAP7_LAM_ERROR
    lam_lats   = HEATMAP7_LAM_LAT
    cmap_m     = _model_color_map(models)

    # Pre-load all records once
    all_records: dict = {
        model: {stem: _load_records(folder, stem) for stem in DATASETS}
        for model, folder in model_folders.items()
    }

    options_macro = []
    for model in models:
        s = _option_stats_macro(all_records[model])
        if s is not None:
            options_macro.append((model, *s))
    grid = _build_best_grid(options_macro, lam_errors, lam_lats)

    # Encode to int for imshow
    model_idx = {m: k for k, m in enumerate(models)}
    grid_int  = np.vectorize(model_idx.get)(grid).astype(float)

    colors  = [cmap_m[m] for m in models]
    cmap    = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(
        grid_int,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(models) - 0.5,
        interpolation="nearest",
    )

    # Axes ticks
    x_ticks = np.linspace(0, len(lam_errors) - 1, 6, dtype=int)
    y_ticks = np.linspace(0, len(lam_lats)   - 1, 6, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{lam_errors[t]:.2f}" for t in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{lam_lats[t]:.3f}" for t in y_ticks])
    ax.set_xlabel("λ_error  (error penalty weight)")
    ax.set_ylabel("λ_latency  (latency penalty weight)")
    ax.set_title("Best Model Decision Map\n(cell colour = model with highest macro-avg reward)")

    # Mark default operating point
    def_e = np.argmin(np.abs(lam_errors - LAMBDA_ERROR_DEFAULT))
    def_l = np.argmin(np.abs(lam_lats   - LAMBDA_LATENCY_DEFAULT))
    ax.plot(def_e, def_l, "w*", markersize=12, label="default (λ_e=1.0, λ_l=0.01)")

    # Legend patches (includes default marker entry)
    patches = [mpatches.Patch(color=cmap_m[m], label=m) for m in models]
    ax.legend(handles=patches, fontsize=12, framealpha=0.8, title="Best model",
              title_fontsize=12, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.tight_layout()
    _save(fig, "chart7_best_model_heatmap.png")


# ---------------------------------------------------------------------------
# Chart 8 — Best standalone model heatmap per dataset
# ---------------------------------------------------------------------------

def chart8_best_model_heatmap_per_dataset(model_folders: dict):
    """
    One subplot per dataset.  Each cell shows the standalone model with the
    highest average economic reward at that (λ_error, λ_latency) combination.
    Same colour palette as chart7 for easy cross-reference.
    """
    models     = sorted(model_folders)
    lam_errors = HEATMAP7_LAM_ERROR
    lam_lats   = HEATMAP7_LAM_LAT
    cmap_m     = _model_color_map(models)

    # Pre-load all records once
    all_records: dict = {
        model: {stem: _load_records(folder, stem) for stem in DATASETS}
        for model, folder in model_folders.items()
    }

    ncols = 3
    nrows = (len(DATASETS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows),
                             squeeze=False)

    for ds_idx, stem in enumerate(DATASETS):
        row, col = divmod(ds_idx, ncols)
        ax = axes[row][col]

        # Build options for this dataset only
        options_single = []
        for model in models:
            recs = all_records[model].get(stem, [])
            s = _option_stats_single(recs)
            if s is not None:
                options_single.append((model, *s))

        grid = _build_best_grid(options_single, lam_errors, lam_lats)

        model_idx = {m: k for k, m in enumerate(models)}
        grid_int  = np.array(
            [[model_idx.get(grid[i, j], 0) for j in range(len(lam_errors))]
             for i in range(len(lam_lats))],
            dtype=float,
        )
        listed_cmap = ListedColormap([cmap_m[m] for m in models])

        ax.imshow(grid_int, aspect="auto", origin="lower", cmap=listed_cmap,
                  vmin=-0.5, vmax=len(models) - 0.5, interpolation="nearest")

        x_ticks = np.linspace(0, len(lam_errors) - 1, 5, dtype=int)
        y_ticks = np.linspace(0, len(lam_lats)   - 1, 5, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{lam_errors[t]:.1f}" for t in x_ticks], fontsize=8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{lam_lats[t]:.3f}" for t in y_ticks], fontsize=8)
        ax.set_title(stem.upper(), fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("λ_latency", fontsize=9)
        if row == nrows - 1:
            ax.set_xlabel("λ_error", fontsize=9)

        # Mark default operating point
        def_e = int(np.argmin(np.abs(lam_errors - LAMBDA_ERROR_DEFAULT)))
        def_l = int(np.argmin(np.abs(lam_lats   - LAMBDA_LATENCY_DEFAULT)))
        ax.plot(def_e, def_l, "w*", markersize=10)

    # Hide any unused subplots
    for ds_idx in range(len(DATASETS), nrows * ncols):
        row, col = divmod(ds_idx, ncols)
        axes[row][col].set_visible(False)

    # Shared legend
    patches = [mpatches.Patch(color=cmap_m[m], label=m) for m in models]
    fig.legend(handles=patches, fontsize=10, framealpha=0.85,
               title="Best model", title_fontsize=10,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(models), borderaxespad=0)

    fig.suptitle(
        "Best Standalone Model per Dataset\n"
        "(cell colour = model with highest avg reward at that λ_error × λ_latency)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "chart8_best_model_heatmap_per_dataset.png")


# ===========================================================================
# CASCADE CHARTS (C1 – C5)
# ===========================================================================

_CASCADE_TAG_TO_MODEL: dict[str, str] = {
    "gpt41mini": "gpt-4.1-mini",
    "gpt54mini": "gpt-5.4-mini",
    "gpt41":     "gpt-4.1",
    "gpt54":     "gpt-5.4",
}

THRESHOLDS = [60, 75, 90]


def _find_cascade_folders() -> list[pathlib.Path]:
    if not OPT_RESULTS_DIR.exists():
        return []
    return sorted(d for d in OPT_RESULTS_DIR.iterdir()
                  if d.is_dir() and d.name.startswith("cascade__"))


def _parse_cascade_name(name: str) -> "tuple[str, str, int] | None":
    m = re.match(r"cascade__small-(\w+)__large-(\w+)__T(\d+)$", name)
    if not m:
        return None
    small = _CASCADE_TAG_TO_MODEL.get(m.group(1), m.group(1))
    large = _CASCADE_TAG_TO_MODEL.get(m.group(2), m.group(2))
    return small, large, int(m.group(3))


def _load_cascade_data() -> dict:
    """
    Returns {(small, large, threshold): {
        'n_total':        int,
        'n_escalated':    int,
        'escalation_pct': float,
        'avg_cost':       float,
        'macro_accuracy': float | None,
        'acc_escalated':  float | None,
        'acc_kept':       float | None,
    }}
    """
    data: dict = {}
    for folder in _find_cascade_folders():
        parsed = _parse_cascade_name(folder.name)
        if parsed is None:
            continue
        small, large, threshold = parsed
        n_total = n_esc = n_corr = n_with = 0
        n_esc_corr = n_esc_with = n_kept_corr = n_kept_with = 0
        cost_sum = reward_sum = reward_with = 0.0
        for stem in DATASETS:
            for rec in _load_records(folder, stem):
                n_total  += 1
                cost_sum += rec.get("cost_usd") or 0.0
                is_esc    = bool(rec.get("escalated", False))
                if is_esc:
                    n_esc += 1
                ic = rec.get("is_correct")
                if ic is not None:
                    n_with += 1
                    n_corr += ic
                    if is_esc:
                        n_esc_with  += 1
                        n_esc_corr  += ic
                    else:
                        n_kept_with += 1
                        n_kept_corr += ic
                er = rec.get("economic_reward")
                if er is not None:
                    reward_sum  += float(er)
                    reward_with += 1
        if n_total == 0:
            continue
        data[(small, large, threshold)] = {
            "n_total":        n_total,
            "n_escalated":    n_esc,
            "escalation_pct": 100.0 * n_esc / n_total,
            "avg_cost":       cost_sum / n_total,
            "macro_accuracy": (100.0 * n_corr / n_with)           if n_with      else None,
            "acc_escalated":  (100.0 * n_esc_corr / n_esc_with)   if n_esc_with  else None,
            "acc_kept":       (100.0 * n_kept_corr / n_kept_with)  if n_kept_with else None,
            "avg_reward":     (reward_sum / reward_with)           if reward_with else None,
        }
    return data


def _cascade_pairs(data: dict) -> list[tuple[str, str]]:
    return sorted({(sm, lg) for sm, lg, _ in data})


def _pair_color_map(pairs: list) -> dict:
    return {p: MODEL_COLORS[i % len(MODEL_COLORS)] for i, p in enumerate(pairs)}


# ---------------------------------------------------------------------------
# Chart C1 — Escalation rate + Accuracy across all thresholds
# ---------------------------------------------------------------------------

def chart_c1_cascade_combined_overview(cascade_data: dict):
    """
    One subplot per (small, large) pair, combining:
      - Bars  (right y-axis, grey): escalation rate % at each threshold
      - Lines (left  y-axis):
          * Overall macro accuracy  (green dashed)
          * Kept-by-small accuracy  (blue solid)
          * Escalated-to-large acc  (orange solid)
    Every escalated-accuracy point is annotated with 'n=X' (number of questions
    actually escalated), so 0 % due to "nothing escalated" is visually distinct
    from 0 % due to genuinely wrong answers.
    """
    pairs = _cascade_pairs(cascade_data)
    if not pairs:
        print("  C_combined: no cascade data, skipping.")
        return

    has_acc = any(
        cascade_data.get((sm, lg, t), {}).get("acc_kept") is not None
        or cascade_data.get((sm, lg, t), {}).get("acc_escalated") is not None
        for sm, lg in pairs
        for t in THRESHOLDS
    )
    if not has_acc:
        print("  C_combined: is_correct not populated — run tester.py to evaluate "
              "optimization_results first.")
        return

    ncols = 2
    nrows = (len(pairs) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5.5 * nrows), squeeze=False)

    bar_color = "#AAAAAA"
    lc = {
        "overall":   "#55A868",
        "kept":      "#4C72B0",
        "escalated": "#DD8452",
    }

    for idx, (sm, lg) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax_l = axes[row][col]
        ax_r = ax_l.twinx()

        xi = np.arange(len(THRESHOLDS))

        esc_pcts, n_escs, kept_vals, esc_vals, macro_vals = [], [], [], [], []
        for t in THRESHOLDS:
            entry = cascade_data.get((sm, lg, t), {})
            esc_pcts.append(entry.get("escalation_pct", 0.0))
            n_escs.append(entry.get("n_escalated", 0))
            kept_vals.append(entry.get("acc_kept"))
            esc_vals.append(entry.get("acc_escalated"))
            macro_vals.append(entry.get("macro_accuracy"))

        # ---- right axis: escalation-rate bars ----
        ax_r.bar(xi, esc_pcts, width=0.5, color=bar_color, alpha=0.30,
                 zorder=2, label="Escalation rate %")
        for xv, pct in zip(xi, esc_pcts):
            ax_r.text(xv, pct + 1.5, f"{pct:.1f}%", ha="center",
                      fontsize=8, color="#888888")
        ax_r.set_ylabel("Escalation Rate (%)", color="#888888", fontsize=9)
        ax_r.tick_params(axis="y", labelcolor="#888888")
        ax_r.set_ylim(0, 140)
        ax_r.yaxis.grid(False)
        ax_r.spines["right"].set_visible(True)

        # ---- left axis: accuracy lines ----
        # overall macro accuracy
        mv = [v if v is not None else float("nan") for v in macro_vals]
        ax_l.plot(xi, mv, marker="s", lw=1.8, linestyle="--",
                  color=lc["overall"], label="Overall accuracy", zorder=5)

        # kept accuracy
        kv = [v if v is not None else float("nan") for v in kept_vals]
        ax_l.plot(xi, kv, marker="o", lw=2,
                  color=lc["kept"], label="Kept (conf ≥ T) accuracy", zorder=5)
        for xv, v in zip(xi, kv):
            if not np.isnan(v):
                ax_l.annotate(f"{v:.1f}%", (xv, v),
                              textcoords="offset points", xytext=(0, 8),
                              fontsize=8, ha="center", color=lc["kept"])

        # escalated accuracy — NaN when nothing was escalated
        ev_plot = [
            (v if v is not None else float("nan")) if n > 0 else float("nan")
            for v, n in zip(esc_vals, n_escs)
        ]
        ax_l.plot(xi, ev_plot, marker="^", lw=2,
                  color=lc["escalated"], label="Escalated (conf < T) accuracy", zorder=5)

        # annotate every escalated point
        for xv, t, n, v in zip(xi, THRESHOLDS, n_escs, esc_vals):
            if n == 0:
                ax_l.annotate("0 escalated", (xv, 3),
                               fontsize=7.5, ha="center", va="bottom",
                               color=lc["escalated"], fontstyle="italic")
            else:
                ypos = v if v is not None else 5.0
                ax_l.annotate(f"{v:.1f}%\n(n={n})", (xv, ypos),
                               textcoords="offset points", xytext=(0, 9),
                               fontsize=8, ha="center", color=lc["escalated"])

        ax_l.set_ylabel("Accuracy (%)", fontsize=10)
        ax_l.set_ylim(0, 120)
        ax_l.set_xticks(xi)
        ax_l.set_xticklabels([f"T={t}" for t in THRESHOLDS], fontsize=10)
        ax_l.set_xlabel("Confidence Threshold", fontsize=10)
        ax_l.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_l.set_axisbelow(True)
        ax_l.set_title(f"small = {sm}   →   large = {lg}",
                       fontsize=10, fontweight="bold")

        # combined legend
        handles, labels_leg = ax_l.get_legend_handles_labels()
        bar_patch = mpatches.Patch(color=bar_color, alpha=0.5,
                                   label="Escalation rate % (right axis)")
        ax_l.legend(handles=handles + [bar_patch],
                    fontsize=8, framealpha=0.88, loc="upper right")

    # hide unused subplots
    for idx in range(len(pairs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Cascade — Escalation Rate & Accuracy across Confidence Thresholds\n"
        "Grey bars = % questions sent to large model  |  "
        "Lines = accuracy  |  n=X = actual number of escalated questions",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartC1_cascade_combined_overview.png")


# ---------------------------------------------------------------------------
# Chart C5 — Confidence score distribution (histogram per small model)
# ---------------------------------------------------------------------------

def _build_heatmap(ax, mat, annot, row_labels, col_labels, title, cmap, cbar_label):
    """Draw a single annotated heatmap on ax; returns the imshow object."""
    import numpy as np
    im = ax.imshow(mat, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Confidence Threshold", fontsize=10)
    ax.set_title(title, fontsize=11)
    for r, row in enumerate(annot):
        for c, text in enumerate(row):
            val = mat[r][c]
            norm = (val - np.nanmin(mat)) / max(np.nanmax(mat) - np.nanmin(mat), 1e-9)
            color = "black" if 0.3 < norm < 0.75 else "white"
            ax.text(c, r, text, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    return im


def chart_c2_cascade_dual_heatmap(cascade_data: dict):
    """C2: Side-by-side heatmaps – Avg Economic Reward | Avg Cost per Query."""
    import numpy as np
    pairs      = _cascade_pairs(cascade_data)
    thresholds = sorted({t for (_, _, t) in cascade_data})
    labels     = [f"{s}\n→ {l}" for s, l in pairs]
    col_labels = [f"T={t}" for t in thresholds]

    reward_mat, reward_ann = [], []
    cost_mat,   cost_ann   = [], []
    for pair in pairs:
        rr, ra, cr, ca = [], [], [], []
        for t in thresholds:
            stats = cascade_data.get((*pair, t), {})
            rv = stats.get("avg_reward")
            cv = stats.get("avg_cost")
            # cost in m$
            cv_ms = cv * 1000 if cv is not None else None
            rr.append(rv  if rv  is not None else float("nan"))
            cr.append(cv_ms if cv_ms is not None else float("nan"))
            ra.append(f"{rv:.3f}"   if rv  is not None else "N/A")
            ca.append(f"{cv_ms:.3f} m$" if cv_ms is not None else "N/A")
        reward_mat.append(rr);  reward_ann.append(ra)
        cost_mat.append(cr);    cost_ann.append(ca)

    rmat = np.array(reward_mat, dtype=float)
    cmat = np.array(cost_mat,   dtype=float)

    h = max(4, len(pairs) * 1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, h))

    im1 = _build_heatmap(ax1, rmat, reward_ann, labels, col_labels,
                         "Avg Economic Reward\n(higher = better)",
                         "RdYlGn", "Avg Reward")
    ax1.set_ylabel("Cascade Configuration", fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Avg Reward")

    im2 = _build_heatmap(ax2, cmat, cost_ann, labels, col_labels,
                         "Avg Cost per Query (m$)\n(lower = better)",
                         "RdYlGn_r", "Avg Cost (m$)")
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Avg Cost (m$)")

    fig.suptitle("Cascade: Reward vs Cost by Configuration × Threshold",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "chartC2_cascade_dual_heatmap.png")


def chart_c3_cascade_best_config_heatmap(model_folders: dict):
    """
    C3: Decision-map heatmap comparing ALL options — cascade configs and standalone
    models — across the (λ_error, λ_latency) plane.  Each cell is coloured by
    whichever option achieves the highest macro-avg reward at those penalty weights.

    Standalone models use warm/neutral tones; cascade configs use dark academic tones
    so the two groups are immediately distinguishable in the legend.
    """
    # ------------------------------------------------------------------ cascade
    all_cascade_folders: dict[tuple, pathlib.Path] = {}
    for folder in _find_cascade_folders():
        parsed = _parse_cascade_name(folder.name)
        if parsed is not None:
            all_cascade_folders[parsed] = folder

    if not all_cascade_folders and not model_folders:
        print("  C3: no data found, skipping.")
        return

    # Only cascade configs that actually escalated at least once
    cascade_configs = [
        cfg for cfg in sorted(all_cascade_folders.keys())
        if any(
            r.get("escalated", False)
            for stem in DATASETS
            for r in _load_records(all_cascade_folders[cfg], stem)
        )
    ]

    # ------------------------------------------------------------------ colours
    # Standalone: consistent with MODEL_COLORS (ColorBrewer Paired)
    # Cascade: warm earth tones grouped by config pair
    #   gpt41mini→gpt41:  reds   (T60/75/90)
    #   gpt41mini→gpt54:  burnt-oranges
    #   gpt54mini→gpt41:  browns
    #   gpt54mini→gpt54:  pinks/magentas
    _cascade_palette = [
        "#B71C1C", "#D32F2F", "#EF5350",  # reds
        "#BF360C", "#E64A19", "#FF7043",  # burnt-oranges
        "#4E342E", "#6D4C41", "#A1887F",  # browns
        "#880E4F", "#C2185B", "#F06292",  # pinks
    ]

    standalone_models = sorted(model_folders.keys())
    standalone_colors = _model_color_map(standalone_models)
    cascade_colors = {
        cfg: _cascade_palette[i % len(_cascade_palette)]
        for i, cfg in enumerate(cascade_configs)
    }

    # ------------------------------------------------------------------ records
    standalone_records: dict = {
        m: {stem: _load_records(folder, stem) for stem in DATASETS}
        for m, folder in model_folders.items()
    }
    cascade_records: dict = {
        cfg: {stem: _load_records(all_cascade_folders[cfg], stem) for stem in DATASETS}
        for cfg in cascade_configs
    }

    # ------------------------------------------------------------------ grid
    lam_errors = HEATMAP_LAM_ERROR
    lam_lats   = HEATMAP_LAM_LAT

    options_macro = []
    for m in standalone_models:
        s = _option_stats_macro(standalone_records[m])
        if s is not None:
            options_macro.append((("standalone", m), *s))
    for cfg in cascade_configs:
        s = _option_stats_macro(cascade_records[cfg])
        if s is not None:
            options_macro.append((("cascade", cfg), *s))
    grid = _build_best_grid(options_macro, lam_errors, lam_lats)

    # Build unified key list and colour map for imshow
    all_keys = (
        [("standalone", m) for m in standalone_models] +
        [("cascade", cfg) for cfg in cascade_configs]
    )
    key_to_idx   = {k: idx for idx, k in enumerate(all_keys)}
    key_to_color = {}
    for k in all_keys:
        if k[0] == "standalone":
            key_to_color[k] = standalone_colors[k[1]]
        else:
            key_to_color[k] = cascade_colors[k[1]]

    grid_int = np.array(
        [[key_to_idx.get(grid[i, j], 0) for j in range(len(lam_errors))]
         for i in range(len(lam_lats))],
        dtype=float,
    )

    cmap = ListedColormap([key_to_color[k] for k in all_keys])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(
        grid_int,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(all_keys) - 0.5,
        interpolation="nearest",
    )

    x_ticks = np.linspace(0, len(lam_errors) - 1, 7, dtype=int)
    y_ticks = np.linspace(0, len(lam_lats)   - 1, 7, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{lam_errors[t]:.2f}" for t in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{lam_lats[t]:.4f}" for t in y_ticks])
    ax.set_xlabel("λ_error  (error penalty weight)", fontsize=11)
    ax.set_ylabel("λ_latency  (latency penalty weight)", fontsize=11)
    ax.set_title(
        "Best Option Decision Map: Cascade vs Standalone\n"
        "(cell colour = option with highest macro-avg reward at those penalty weights)",
        fontsize=12, fontweight="bold",
    )

    # Mark default operating point
    def_e = int(np.argmin(np.abs(lam_errors - LAMBDA_ERROR_DEFAULT)))
    def_l = int(np.argmin(np.abs(lam_lats   - LAMBDA_LATENCY_DEFAULT)))
    ax.plot(def_e, def_l, "w*", markersize=14,
            label=f"default (λ_e={LAMBDA_ERROR_DEFAULT}, λ_l={LAMBDA_LATENCY_DEFAULT})")

    # Legend: only options that actually win at least one cell
    winning_keys = sorted(
        {grid[i, j] for i in range(len(lam_lats)) for j in range(len(lam_errors))
         if grid[i, j] is not None},
        key=lambda k: (0 if k[0] == "standalone" else 1, k[1]),
    )

    standalone_patches = [
        mpatches.Patch(color=key_to_color[k], label=k[1])
        for k in winning_keys if k[0] == "standalone"
    ]
    cascade_patches = [
        mpatches.Patch(
            color=key_to_color[k],
            label=f"{k[1][0]} → {k[1][1]}  (T={k[1][2]})"
        )
        for k in winning_keys if k[0] == "cascade"
    ]

    # Two-section legend using a title spacer trick
    spacer = mpatches.Patch(color="none", label="")
    legend_handles = []
    if standalone_patches:
        legend_handles += [mpatches.Patch(color="none", label="— Standalone models —")]
        legend_handles += standalone_patches
    if cascade_patches:
        legend_handles += [mpatches.Patch(color="none", label="— Cascade configs —")]
        legend_handles += cascade_patches

    ax.legend(
        handles=legend_handles,
        fontsize=9, framealpha=0.85,
        title="Best option", title_fontsize=10,
        loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
    )

    fig.tight_layout()
    _save(fig, "chartC3_cascade_best_config_heatmap.png")


# ---------------------------------------------------------------------------
# Chart C6 — Per-dataset decision map: best cascade/standalone option per dataset
# ---------------------------------------------------------------------------

def chart_c6_cascade_per_dataset_decision_map(model_folders: dict):
    """
    C6: One subplot per dataset (5 total, left-to-right then second row).
    Each subplot is a decision map on the (λ_error × λ_latency) plane where
    the cell colour shows which cascade config OR standalone model achieves the
    highest reward on THAT SINGLE DATASET at those penalty weights.
    Same colour convention as C3.
    """
    all_cascade_folders: dict[tuple, pathlib.Path] = {}
    for folder in _find_cascade_folders():
        parsed = _parse_cascade_name(folder.name)
        if parsed is not None:
            all_cascade_folders[parsed] = folder

    if not all_cascade_folders and not model_folders:
        print("  C6: no data found, skipping.")
        return

    cascade_configs = [
        cfg for cfg in sorted(all_cascade_folders.keys())
        if any(
            r.get("escalated", False)
            for stem in DATASETS
            for r in _load_records(all_cascade_folders[cfg], stem)
        )
    ]

    _cascade_palette = [
        "#B71C1C", "#D32F2F", "#EF5350",  # reds
        "#BF360C", "#E64A19", "#FF7043",  # burnt-oranges
        "#4E342E", "#6D4C41", "#A1887F",  # browns
        "#880E4F", "#C2185B", "#F06292",  # pinks
    ]
    standalone_models = sorted(model_folders.keys())
    standalone_colors = _model_color_map(standalone_models)
    cascade_colors    = {cfg: _cascade_palette[i % len(_cascade_palette)]
                         for i, cfg in enumerate(cascade_configs)}

    sa_records = {m: {stem: _load_records(folder, stem) for stem in DATASETS}
                  for m, folder in model_folders.items()}
    cas_records = {cfg: {stem: _load_records(all_cascade_folders[cfg], stem) for stem in DATASETS}
                   for cfg in cascade_configs}

    all_keys = (["standalone", m] for m in standalone_models)
    # Build ordered key list and colour map once
    ordered_keys = (
        [("standalone", m) for m in standalone_models] +
        [("cascade",    cfg) for cfg in cascade_configs]
    )
    key_to_color = {}
    for k in ordered_keys:
        key_to_color[k] = standalone_colors[k[1]] if k[0] == "standalone" else cascade_colors[k[1]]
    key_to_idx = {k: i for i, k in enumerate(ordered_keys)}

    lam_errors = HEATMAP_LAM_ERROR
    lam_lats   = HEATMAP_LAM_LAT

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10), squeeze=False)

    # Track which keys win at least one cell across any panel for shared legend
    global_winning_keys: set = set()

    for ds_idx, stem in enumerate(DATASETS):
        row, col = divmod(ds_idx, ncols)
        ax = axes[row][col]

        options_ds = []
        for m in standalone_models:
            s = _option_stats_single(sa_records[m][stem])
            if s is not None:
                options_ds.append((("standalone", m), *s))
        for cfg in cascade_configs:
            s = _option_stats_single(cas_records[cfg][stem])
            if s is not None:
                options_ds.append((("cascade", cfg), *s))
        grid = _build_best_grid(options_ds, lam_errors, lam_lats)
        global_winning_keys.update(k for k in grid.flat if k is not None)

        grid_int = np.array(
            [[key_to_idx.get(grid[i, j], 0) for j in range(len(lam_errors))]
             for i in range(len(lam_lats))],
            dtype=float,
        )
        cmap = ListedColormap([key_to_color[k] for k in ordered_keys])
        ax.imshow(grid_int, aspect="auto", origin="lower", cmap=cmap,
                  vmin=-0.5, vmax=len(ordered_keys) - 0.5, interpolation="nearest")

        x_ticks = np.linspace(0, len(lam_errors) - 1, 5, dtype=int)
        y_ticks = np.linspace(0, len(lam_lats)   - 1, 5, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{lam_errors[t]:.2f}" for t in x_ticks], fontsize=7)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{lam_lats[t]:.4f}" for t in y_ticks], fontsize=7)
        ax.set_title(DS_LABELS[stem], fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel("λ_latency", fontsize=9)
        if row == nrows - 1 or ds_idx == len(DATASETS) - 1:
            ax.set_xlabel("λ_error", fontsize=9)

        # Default operating point
        def_e = int(np.argmin(np.abs(lam_errors - LAMBDA_ERROR_DEFAULT)))
        def_l = int(np.argmin(np.abs(lam_lats   - LAMBDA_LATENCY_DEFAULT)))
        ax.plot(def_e, def_l, "w*", markersize=10)

    # Hide the unused 6th cell
    axes[1][2].set_visible(False)

    # Shared legend in the empty cell
    legend_ax = axes[1][2]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    winning_ordered = [k for k in ordered_keys if k in global_winning_keys]
    legend_handles = []
    sa_win = [k for k in winning_ordered if k[0] == "standalone"]
    cas_win = [k for k in winning_ordered if k[0] == "cascade"]
    if sa_win:
        legend_handles.append(mpatches.Patch(color="none", label="— Standalone —"))
        for k in sa_win:
            legend_handles.append(mpatches.Patch(color=key_to_color[k], label=k[1]))
    if cas_win:
        legend_handles.append(mpatches.Patch(color="none", label="— Cascade —"))
        for k in cas_win:
            legend_handles.append(mpatches.Patch(
                color=key_to_color[k],
                label=f"{k[1][0]} → {k[1][1]}  T={k[1][2]}"
            ))
    legend_handles.append(mpatches.Patch(color="none", label=""))
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="gray", linestyle="none",
                   markersize=10, label=f"default (λe={LAMBDA_ERROR_DEFAULT}, λl={LAMBDA_LATENCY_DEFAULT})")
    )
    legend_ax.legend(handles=legend_handles, loc="center", fontsize=8,
                     framealpha=0.85, title="Best option", title_fontsize=9)

    fig.suptitle(
        "Cascade: Best Option Decision Map — Per Dataset\n"
        "(cell = option with highest reward at those λ_error × λ_latency weights)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "chartC6_cascade_per_dataset_decision_map.png")


# ---------------------------------------------------------------------------
# Chart C4 — Cascade reward vs λ_error (2×2 grid: one panel per config pair,
#             3 lines per panel = one per threshold)
# ---------------------------------------------------------------------------

def chart_c4_cascade_reward_vs_lambda_error():
    """
    2×2 subplots — one per (small, large) config pair.
    Each subplot shows 3 lines (T=60 / T=75 / T=90): how the macro-avg
    economic penalty changes as λ_error is swept from 0.1 to 10.
    λ_latency is held at its default value throughout.
    """
    # Collect all cascade folders keyed by (small, large, threshold)
    all_folders: dict[tuple, pathlib.Path] = {}
    for folder in _find_cascade_folders():
        parsed = _parse_cascade_name(folder.name)
        if parsed is not None:
            all_folders[parsed] = folder

    if not all_folders:
        print("  C8: no cascade data, skipping.")
        return

    pairs      = sorted({(sm, lg) for sm, lg, _ in all_folders})
    thresholds = sorted({t for (_, _, t) in all_folders})

    # Dense coverage: fine-grained at low values (0.01–1), then linear up to 25
    lambda_errors = np.unique(np.concatenate([
        np.linspace(0.01, 1.0,  80),   # fine-grained in the sensitive low range
        np.linspace(1.0,  5.0, 100),   # moderate sweep through typical values
        np.linspace(5.0,  15.0, 60),   # upper range
        np.linspace(15.0, 50.0, 40),   # extended upper range
    ]))
    threshold_colors = {60: "#2ca02c", 75: "#ff7f0e", 90: "#d62728"}
    threshold_styles = {60: "-", 75: "--", 90: ":"}

    ncols = 2
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for idx, (sm, lg) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for t in thresholds:
            folder = all_folders.get((sm, lg, t))
            if folder is None:
                continue

            # Pre-load records once per (config, threshold)
            all_records = {stem: _load_records(folder, stem) for stem in DATASETS}

            # Skip threshold lines where no escalation occurred
            if not any(
                r.get("escalated", False)
                for stem in DATASETS
                for r in all_records[stem]
            ):
                continue

            penalties = []
            lam_vals_used = []
            for lam_e in lambda_errors:
                ds_avgs = []
                for stem in DATASETS:
                    records = all_records[stem]
                    if not records:
                        continue
                    rews = [
                        compute_reward(
                            r.get("cost_usd", 0.0) or 0.0,
                            r.get("elapsed_s", 0.0) or 0.0,
                            r.get("is_correct", 0),
                            lam_e, LAMBDA_LATENCY_DEFAULT,
                        )
                        for r in records
                    ]
                    ds_avgs.append(sum(rews) / len(rews))
                val = sum(ds_avgs) / len(ds_avgs) if ds_avgs else 0.0
                penalties.append(-val)
                lam_vals_used.append(lam_e)

            ax.plot(lam_vals_used, penalties,
                    color=threshold_colors[t],
                    linestyle=threshold_styles[t],
                    lw=2.2,
                    label=f"T={t}")

        ax.axvline(LAMBDA_ERROR_DEFAULT, color="gray", linestyle=":", lw=1,
                   label=f"default λ_e={LAMBDA_ERROR_DEFAULT}")
        ax.set_title(f"small = {sm}   →   large = {lg}", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.85)

    # Shared axis labels
    for row in range(nrows):
        axes[row][0].set_ylabel("Economic Penalty  (lower = better)", fontsize=10)
    for col in range(ncols):
        axes[nrows - 1][col].set_xlabel("λ_error  (error penalty weight)", fontsize=10)

    # Hide unused subplots
    for idx in range(len(pairs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Cascade: Economic Penalty vs λ_error\n"
        "Each panel = one small→large config pair  |  "
        "Lines = confidence threshold (T=60 / 75 / 90)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartC4_cascade_reward_vs_lambda_error.png")


# ---------------------------------------------------------------------------
# Chart C5 — Cascade reward vs λ_latency (2×2 grid: one panel per config pair,
#             3 lines per panel = one per threshold)
# ---------------------------------------------------------------------------

def chart_c5_cascade_reward_vs_lambda_latency():
    """
    2×2 subplots — one per (small, large) config pair.
    Each subplot shows 3 lines (T=60 / T=75 / T=90): how the macro-avg
    economic penalty changes as λ_latency is swept from 0.001 to 0.5.
    λ_error is held at its default value throughout.
    Uses the same non-linear y-axis as chart6 so visual spacing is uniform.
    """
    all_folders: dict[tuple, pathlib.Path] = {}
    for folder in _find_cascade_folders():
        parsed = _parse_cascade_name(folder.name)
        if parsed is not None:
            all_folders[parsed] = folder

    if not all_folders:
        print("  C9: no cascade data, skipping.")
        return

    pairs      = sorted({(sm, lg) for sm, lg, _ in all_folders})
    thresholds = sorted({t for (_, _, t) in all_folders})

    # Dense coverage: very fine at near-zero, then extended up to 2.0
    lambda_lats = np.unique(np.concatenate([
        np.linspace(0.0001, 0.01,  100),   # very fine near zero
        np.linspace(0.01,   0.1,   100),   # fine in the sensitive low range
        np.linspace(0.1,    0.5,  100),   # standard sweep
        np.linspace(0.5,    1.0,   60),   # extended upper range
        np.linspace(1.0,    10.0,   60),   # far upper range
    ]))
    threshold_colors = {60: "#2ca02c", 75: "#ff7f0e", 90: "#d62728"}
    threshold_styles = {60: "-", 75: "--", 90: ":"}

    # Non-linear y-axis — extended to cover larger penalties at high λ_latency
    ytick_vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 30, 50]
    ytick_pos  = list(range(len(ytick_vals)))
    def to_pos(v):
        return float(np.interp(v, ytick_vals, ytick_pos))

    ncols = 2
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for idx, (sm, lg) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for t in thresholds:
            folder = all_folders.get((sm, lg, t))
            if folder is None:
                continue

            all_records = {stem: _load_records(folder, stem) for stem in DATASETS}

            # Skip threshold lines where no escalation occurred
            if not any(
                r.get("escalated", False)
                for stem in DATASETS
                for r in all_records[stem]
            ):
                continue

            penalties_pos = []
            lam_vals_used = []
            for lam_l in lambda_lats:
                ds_avgs = []
                for stem in DATASETS:
                    records = all_records[stem]
                    if not records:
                        continue
                    rews = [
                        compute_reward(
                            r.get("cost_usd", 0.0) or 0.0,
                            r.get("elapsed_s", 0.0) or 0.0,
                            r.get("is_correct", 0),
                            LAMBDA_ERROR_DEFAULT, lam_l,
                        )
                        for r in records
                    ]
                    ds_avgs.append(sum(rews) / len(rews))
                val = sum(ds_avgs) / len(ds_avgs) if ds_avgs else 0.0
                penalties_pos.append(to_pos(-val))
                lam_vals_used.append(lam_l)
                if -val >= max(ytick_vals):
                    break

            ax.plot(lam_vals_used, penalties_pos,
                    color=threshold_colors[t],
                    linestyle=threshold_styles[t],
                    lw=2.2,
                    label=f"T={t}")

        ax.axvline(LAMBDA_LATENCY_DEFAULT, color="gray", linestyle=":", lw=1,
                   label=f"default λ_l={LAMBDA_LATENCY_DEFAULT}")
        ax.set_title(f"small = {sm}   →   large = {lg}", fontsize=10, fontweight="bold")
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_vals)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.85)

    for row in range(nrows):
        axes[row][0].set_ylabel("Economic Penalty  (lower = better)", fontsize=10)
    for col in range(ncols):
        axes[nrows - 1][col].set_xlabel("λ_latency  (latency penalty weight)", fontsize=10)

    for idx in range(len(pairs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Cascade: Economic Penalty vs λ_latency\n"
        "Each panel = one small→large config pair  |  "
        "Lines = confidence threshold (T=60 / 75 / 90)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartC5_cascade_reward_vs_lambda_latency.png")


# ===========================================================================
# SELF-CONSISTENCY CHARTS (SC prefix)
# ===========================================================================

def _find_selfcons_folders() -> list[pathlib.Path]:
    if not OPT_RESULTS_DIR.exists():
        return []
    return sorted(d for d in OPT_RESULTS_DIR.iterdir()
                  if d.is_dir() and d.name.startswith("selfcons__"))


def _parse_selfcons_name(name: str) -> "tuple[str, int] | None":
    m = re.match(r"selfcons__(\w+)__N(\d+)$", name)
    if not m:
        return None
    model = _CASCADE_TAG_TO_MODEL.get(m.group(1), m.group(1))
    return model, int(m.group(2))


def _load_selfcons_data() -> dict:
    """
    Returns {(model, n): {
        'n_total':        int,
        'macro_accuracy': float | None,   # % correct
        'avg_cost':       float,
        'avg_latency':    float,
        'avg_reward':     float | None,
        'agreement_rate': float,          # fraction of records where all N votes agree
        'ds_accuracy':    {stem: float},
        'ds_agreement':   {stem: float},
        'ds_cost':        {stem: float},
        'ds_latency':     {stem: float},
        'ds_reward':      {stem: float},
    }}
    """
    data: dict = {}
    for folder in _find_selfcons_folders():
        parsed = _parse_selfcons_name(folder.name)
        if parsed is None:
            continue
        model, n = parsed
        ds_acc, ds_agr, ds_cost, ds_lat, ds_rew = {}, {}, {}, {}, {}
        total_n = total_agree = total_all = 0

        for stem in DATASETS:
            records = _load_records(folder, stem)
            if not records:
                continue
            n_stem = len(records)
            accs   = [r.get("is_correct", 0) for r in records]
            costs  = [r.get("cost_usd",   0.0) or 0.0 for r in records]
            lats   = [r.get("elapsed_s",  0.0) or 0.0 for r in records]
            rews   = [r.get("economic_reward") for r in records]
            agrees = [
                int(len(set(r.get("vote_responses") or [])) == 1)
                for r in records
            ]
            ds_acc[stem]  = 100.0 * sum(accs)  / n_stem
            ds_agr[stem]  = 100.0 * sum(agrees) / n_stem
            ds_cost[stem] = sum(costs) / n_stem
            ds_lat[stem]  = sum(lats)  / n_stem
            valid_rews    = [r for r in rews if r is not None]
            ds_rew[stem]  = sum(valid_rews) / len(valid_rews) if valid_rews else None

            total_n     += n_stem
            total_agree += sum(agrees)
            total_all   += n_stem

        if not ds_acc:
            continue

        macro_acc = sum(ds_acc.values()) / len(ds_acc)
        avg_cost  = sum(ds_cost.values()) / len(ds_cost)
        avg_lat   = sum(ds_lat.values())  / len(ds_lat)
        valid_rew = [v for v in ds_rew.values() if v is not None]
        avg_rew   = sum(valid_rew) / len(valid_rew) if valid_rew else None
        agr_rate  = 100.0 * total_agree / total_all if total_all else 0.0

        data[(model, n)] = {
            "n_total":        total_n,
            "macro_accuracy": macro_acc,
            "avg_cost":       avg_cost,
            "avg_latency":    avg_lat,
            "avg_reward":     avg_rew,
            "agreement_rate": agr_rate,
            "ds_accuracy":    ds_acc,
            "ds_agreement":   ds_agr,
            "ds_cost":        ds_cost,
            "ds_latency":     ds_lat,
            "ds_reward":      ds_rew,
        }
    return data


# ---------------------------------------------------------------------------
# Chart SC1 — SC overview: accuracy + agreement rate per dataset
# ---------------------------------------------------------------------------

def chart_sc1_selfcons_overview(sc_data: dict, model_folders: dict):
    """
    SC1: One subplot per SC config (model × N).
    x-axis = datasets.
    Left y-axis  (lines): SC accuracy vs single-call base model accuracy.
    Right y-axis (bars):  vote agreement rate (% questions where all N agree).
    """
    if not sc_data:
        print("  SC1: no self-consistency data, skipping.")
        return

    configs = sorted(sc_data.keys())
    ncols   = 2
    nrows   = (len(configs) + ncols - 1) // ncols

    bar_color  = "#BBBBBB"
    sc_color   = "#1B4F72"    # dark navy for SC line
    base_color = "#B5451B"    # burnt orange for base line

    # Pre-load base model accuracy per dataset (single-call standalone)
    base_ds_acc: dict[str, dict[str, float]] = {}
    for (model, _n) in sc_data.keys():
        folder = _resolve_base_folder(model, model_folders)
        if folder is None:
            continue
        acc_map = {}
        for stem in DATASETS:
            records = _load_records(folder, stem)
            if records:
                acc_map[stem] = 100.0 * sum(r.get("is_correct", 0) for r in records) / len(records)
        if acc_map:
            base_ds_acc[model] = acc_map

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5.5 * nrows), squeeze=False)

    for idx, (model, n) in enumerate(configs):
        row, col = divmod(idx, ncols)
        ax_l = axes[row][col]
        ax_r = ax_l.twinx()

        stats = sc_data[(model, n)]
        stems = [s for s in DATASETS if s in stats["ds_accuracy"]]
        xi    = np.arange(len(stems))
        labels = [DS_LABELS[s] for s in stems]

        sc_accs   = [stats["ds_accuracy"][s]  for s in stems]
        base_accs = [base_ds_acc.get(model, {}).get(s) for s in stems]
        agrs      = [stats["ds_agreement"][s] for s in stems]

        # Right axis: agreement rate bars
        ax_r.bar(xi, agrs, width=0.45, color=bar_color, alpha=0.30,
                 zorder=2, label="Agreement rate %")
        for xv, agr in zip(xi, agrs):
            ax_r.text(xv, agr + 1.5, f"{agr:.0f}%", ha="center",
                      fontsize=8, color="#888888")
        ax_r.set_ylabel("Vote Agreement Rate (%)", color="#888888", fontsize=9)
        ax_r.tick_params(axis="y", labelcolor="#888888")
        ax_r.set_ylim(0, 140)
        ax_r.yaxis.grid(False)
        ax_r.spines["right"].set_visible(True)

        # Left axis: SC accuracy line
        ax_l.plot(xi, sc_accs, marker="o", lw=2, color=sc_color,
                  label=f"SC (N={n}) accuracy", zorder=5)
        for xv, v in zip(xi, sc_accs):
            ax_l.annotate(f"{v:.1f}%", (xv, v),
                          textcoords="offset points", xytext=(0, 8),
                          fontsize=8, ha="center", color=sc_color)

        # Left axis: base model accuracy line
        base_plot = [v if v is not None else float("nan") for v in base_accs]
        ax_l.plot(xi, base_plot, marker="s", lw=1.8, linestyle="--",
                  color=base_color, label=f"{model} (single-call)", zorder=5)
        for xv, v in zip(xi, base_plot):
            if not np.isnan(v):
                ax_l.annotate(f"{v:.1f}%", (xv, v),
                              textcoords="offset points", xytext=(0, -14),
                              fontsize=8, ha="center", color=base_color)

        ax_l.set_ylabel("Accuracy (%)", fontsize=10)
        ax_l.set_ylim(0, 120)
        ax_l.set_xticks(xi)
        ax_l.set_xticklabels(labels, fontsize=9)
        ax_l.set_xlabel("Dataset", fontsize=10)
        ax_l.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_l.set_axisbelow(True)
        ax_l.set_title(f"Self-Consistency: {model}  (N={n})",
                       fontsize=10, fontweight="bold")

        handles, labels_leg = ax_l.get_legend_handles_labels()
        bar_patch = mpatches.Patch(color=bar_color, alpha=0.5,
                                   label="Agreement rate % (right axis)")
        ax_l.legend(handles=handles + [bar_patch],
                    fontsize=8, framealpha=0.88, loc="upper right")

    for idx in range(len(configs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Self-Consistency — Accuracy vs Vote Agreement Rate per Dataset\n"
        "Grey bars = % questions where all N votes agree  |  "
        "Lines: SC majority-vote vs single-call baseline",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartSC1_selfcons_overview.png")


# ---------------------------------------------------------------------------
# Chart SC2 — SC dual heatmap: reward + cost (SC configs + standalone)
# ---------------------------------------------------------------------------

def chart_sc2_selfcons_dual_heatmap(sc_data: dict, model_folders: dict):
    """
    SC2: Side-by-side heatmaps (reward | cost) with rows = SC configs + standalone
    models and columns = datasets, so reward and cost can be compared across every
    option head-to-head at a glance.
    """
    if not sc_data:
        print("  SC2: no self-consistency data, skipping.")
        return

    # Build standalone per-dataset reward + cost
    standalone_ds_rew:  dict[str, dict[str, float]] = {}
    standalone_ds_cost: dict[str, dict[str, float]] = {}
    for model, folder in sorted(model_folders.items()):
        rmap, cmap = {}, {}
        for stem in DATASETS:
            records = _load_records(folder, stem)
            if not records:
                continue
            rews  = [compute_reward(
                r.get("cost_usd", 0.0) or 0.0,
                r.get("elapsed_s", 0.0) or 0.0,
                r.get("is_correct", 0),
                LAMBDA_ERROR_DEFAULT, LAMBDA_LATENCY_DEFAULT,
            ) for r in records]
            costs = [r.get("cost_usd", 0.0) or 0.0 for r in records]
            rmap[stem]  = sum(rews) / len(rews)
            cmap[stem]  = sum(costs) / len(costs) * 1000  # to milli-dollars
        standalone_ds_rew[model]  = rmap
        standalone_ds_cost[model] = cmap

    sc_configs = sorted(sc_data.keys())

    # Build row labels + data matrices
    row_labels  = []
    reward_mat  = []
    reward_ann  = []
    cost_mat    = []
    cost_ann    = []

    # SC rows
    for (model, n) in sc_configs:
        stats = sc_data[(model, n)]
        row_labels.append(f"SC {model}\n(N={n})")
        rr, ra, cr, ca = [], [], [], []
        for stem in DATASETS:
            rv = stats["ds_reward"].get(stem)
            cv = stats["ds_cost"].get(stem)
            cv_ms = cv * 1000 if cv is not None else None
            rr.append(rv    if rv    is not None else float("nan"))
            cr.append(cv_ms if cv_ms is not None else float("nan"))
            ra.append(f"{rv:.3f}"       if rv    is not None else "N/A")
            ca.append(f"{cv_ms:.3f} m$" if cv_ms is not None else "N/A")
        reward_mat.append(rr); reward_ann.append(ra)
        cost_mat.append(cr);   cost_ann.append(ca)

    # Standalone rows
    for model in sorted(standalone_ds_rew.keys()):
        row_labels.append(f"{model}\n(standalone)")
        rr, ra, cr, ca = [], [], [], []
        for stem in DATASETS:
            rv = standalone_ds_rew[model].get(stem)
            cv = standalone_ds_cost[model].get(stem)
            rr.append(rv if rv is not None else float("nan"))
            cr.append(cv if cv is not None else float("nan"))
            ra.append(f"{rv:.3f}"       if rv is not None else "N/A")
            ca.append(f"{cv:.3f} m$"    if cv is not None else "N/A")
        reward_mat.append(rr); reward_ann.append(ra)
        cost_mat.append(cr);   cost_ann.append(ca)

    col_labels = [DS_LABELS[s] for s in DATASETS]
    rmat = np.array(reward_mat, dtype=float)
    cmat = np.array(cost_mat,   dtype=float)

    h = max(4, len(row_labels) * 1.3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, h))

    im1 = _build_heatmap(ax1, rmat, reward_ann, row_labels, col_labels,
                         "Avg Economic Reward per Dataset\n(higher = better)",
                         "RdYlGn", "Avg Reward")
    ax1.set_ylabel("Configuration", fontsize=10)
    ax1.set_xlabel("Dataset", fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Avg Reward")

    im2 = _build_heatmap(ax2, cmat, cost_ann, row_labels, col_labels,
                         "Avg Cost per Query (m$)\n(lower = better)",
                         "RdYlGn_r", "Avg Cost (m$)")
    ax2.set_xlabel("Dataset", fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Avg Cost (m$)")

    fig.suptitle("Self-Consistency vs Standalone: Reward & Cost by Dataset",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "chartSC2_selfcons_dual_heatmap.png")


# ---------------------------------------------------------------------------
# Chart SC3 — SC best-option decision map (λ_error × λ_latency)
# ---------------------------------------------------------------------------

def chart_sc3_selfcons_best_config_heatmap(sc_data: dict, model_folders: dict):
    """
    SC3: Decision map (λ_error × λ_latency) comparing SC configs against their
    base standalone models.  Cell colour = option with highest macro-avg reward.
    SC configs: dark tones. Standalone models: warm tones.  Same visual
    convention as C3 so both charts can be compared side by side.
    """
    if not sc_data:
        print("  SC3: no self-consistency data, skipping.")
        return

    # ---- colours ----
    # Standalone: consistent with MODEL_COLORS (ColorBrewer Paired)
    # SC configs: dark teal family
    _sc_palette = [
        "#00695C",  # dark teal  (gpt-4.1-mini SC)
        "#00ACC1",  # medium cyan (gpt-5.4-mini SC)
    ]

    sc_configs       = sorted(sc_data.keys())
    standalone_models = sorted(model_folders.keys())
    sc_colors  = {cfg: _sc_palette[i % len(_sc_palette)] for i, cfg in enumerate(sc_configs)}
    sa_colors  = _model_color_map(standalone_models)

    # ---- records ----
    sc_records = {
        cfg: {stem: _load_records(_find_selfcons_folder(cfg), stem) for stem in DATASETS}
        for cfg in sc_configs
    }
    sa_records = {
        m: {stem: _load_records(folder, stem) for stem in DATASETS}
        for m, folder in model_folders.items()
    }
    # Normalise SC latency: use the base model's measured latency per dataset
    # so that server-timing differences between benchmark runs do not skew the map.
    sc_base_lats = {cfg: _base_avg_latency(cfg[0], model_folders) for cfg in sc_configs}

    lam_errors = HEATMAP_LAM_ERROR
    lam_lats   = HEATMAP_LAM_LAT

    options_macro = []
    for m in standalone_models:
        s = _option_stats_macro(sa_records[m])
        if s is not None:
            options_macro.append((("standalone", m), *s))
    for cfg in sc_configs:
        s = _option_stats_macro_sc(sc_records[cfg], sc_base_lats[cfg])
        if s is not None:
            options_macro.append((("sc", cfg), *s))
    grid = _build_best_grid(options_macro, lam_errors, lam_lats)

    all_keys = (
        [("standalone", m) for m in standalone_models] +
        [("sc", cfg) for cfg in sc_configs]
    )
    key_to_idx   = {k: idx for idx, k in enumerate(all_keys)}
    key_to_color = {
        **{("standalone", m):   sa_colors[m]   for m in standalone_models},
        **{("sc",         cfg): sc_colors[cfg] for cfg in sc_configs},
    }

    grid_int = np.array(
        [[key_to_idx.get(grid[i, j], 0) for j in range(len(lam_errors))]
         for i in range(len(lam_lats))],
        dtype=float,
    )
    cmap = ListedColormap([key_to_color[k] for k in all_keys])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(
        grid_int, aspect="auto", origin="lower", cmap=cmap,
        vmin=-0.5, vmax=len(all_keys) - 0.5, interpolation="nearest",
    )

    x_ticks = np.linspace(0, len(lam_errors) - 1, 7, dtype=int)
    y_ticks = np.linspace(0, len(lam_lats)   - 1, 7, dtype=int)
    ax.set_xticks(x_ticks); ax.set_xticklabels([f"{lam_errors[t]:.2f}" for t in x_ticks])
    ax.set_yticks(y_ticks); ax.set_yticklabels([f"{lam_lats[t]:.4f}" for t in y_ticks])
    ax.set_xlabel("λ_error  (error penalty weight)", fontsize=11)
    ax.set_ylabel("λ_latency  (latency penalty weight)", fontsize=11)
    ax.set_title(
        "Best Option Decision Map: Self-Consistency vs Standalone\n"
        "(cell colour = option with highest macro-avg reward at those penalty weights)",
        fontsize=12, fontweight="bold",
    )

    def_e = int(np.argmin(np.abs(lam_errors - LAMBDA_ERROR_DEFAULT)))
    def_l = int(np.argmin(np.abs(lam_lats   - LAMBDA_LATENCY_DEFAULT)))
    ax.plot(def_e, def_l, "w*", markersize=14,
            label=f"default (λ_e={LAMBDA_ERROR_DEFAULT}, λ_l={LAMBDA_LATENCY_DEFAULT})")

    winning_keys = sorted(
        {grid[i, j] for i in range(len(lam_lats)) for j in range(len(lam_errors))
         if grid[i, j] is not None},
        key=lambda k: (0 if k[0] == "standalone" else 1, k[1]),
    )
    standalone_patches = [
        mpatches.Patch(color=key_to_color[k], label=k[1])
        for k in winning_keys if k[0] == "standalone"
    ]
    sc_patches = [
        mpatches.Patch(color=key_to_color[k], label=f"SC {k[1][0]}  (N={k[1][1]})")
        for k in winning_keys if k[0] == "sc"
    ]
    legend_handles = []
    if standalone_patches:
        legend_handles += [mpatches.Patch(color="none", label="— Standalone models —")]
        legend_handles += standalone_patches
    if sc_patches:
        legend_handles += [mpatches.Patch(color="none", label="— SC configs —")]
        legend_handles += sc_patches

    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.85,
              title="Best option", title_fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    _save(fig, "chartSC3_selfcons_best_config_heatmap.png")


def _find_selfcons_folder(cfg: tuple) -> pathlib.Path:
    """Return the optimization_results folder for a (model, n) SC config."""
    model, n = cfg
    # Reverse-map model display name → tag
    reverse_tag = {v: k for k, v in _CASCADE_TAG_TO_MODEL.items()}
    tag = reverse_tag.get(model, model)
    name = f"selfcons__{tag}__N{n}"
    return OPT_RESULTS_DIR / name


def _resolve_base_folder(sc_model: str, model_folders: dict) -> "pathlib.Path | None":
    """Find the standalone folder that best matches an SC base model name.

    Tries an exact match first, then falls back to a prefix match.
    """
    if sc_model in model_folders:
        return model_folders[sc_model]
    for key, folder in model_folders.items():
        if key.startswith(sc_model):
            return folder
    return None


def _base_avg_latency(model: str, model_folders: dict) -> dict:
    """Return {stem: avg_elapsed_s} from the standalone base model.

    Used to normalise SC reward calculations so that latency differences
    caused by running benchmarks at different times do not distort results.
    SC calls N models in parallel (wall-clock ≈ single call), so the fairest
    comparison uses the base model's own measured latency for both options.
    """
    folder = _resolve_base_folder(model, model_folders)
    if folder is None:
        return {}
    result = {}
    for stem in DATASETS:
        recs = _load_records(folder, stem)
        if recs:
            result[stem] = sum(r.get("elapsed_s", 0.0) or 0.0 for r in recs) / len(recs)
    return result


# ---------------------------------------------------------------------------
# Chart SC4 — SC reward vs λ_error (one panel per SC config)
# ---------------------------------------------------------------------------

def chart_sc4_selfcons_reward_vs_lambda_error(sc_data: dict, model_folders: dict):
    """
    SC4: One subplot per SC config.
    Solid line  = SC majority-vote reward vs λ_error.
    Dashed line = single-call base model (for comparison).
    λ_latency held at default throughout.
    """
    if not sc_data:
        print("  SC4: no self-consistency data, skipping.")
        return

    sc_configs = sorted(sc_data.keys())
    lambda_errors = np.unique(np.concatenate([
        np.linspace(0.01, 1.0,  80),
        np.linspace(1.0,  5.0, 100),
        np.linspace(5.0,  15.0, 60),
        np.linspace(15.0, 50.0, 40),
    ]))

    sc_color   = "#1B4F72"
    base_color = "#B5451B"

    ncols = 2
    nrows = (len(sc_configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for idx, (model, n) in enumerate(sc_configs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # ---- SC config ----
        sc_folder = _find_selfcons_folder((model, n))
        sc_recs   = {stem: _load_records(sc_folder, stem) for stem in DATASETS}
        # Use base model's latency to normalise SC timing across benchmark runs
        base_lat_map = _base_avg_latency(model, model_folders)

        sc_penalties = []
        for lam_e in lambda_errors:
            ds_avgs = []
            for stem in DATASETS:
                records = sc_recs[stem]
                if not records:
                    continue
                base_lat = base_lat_map.get(stem, records[0].get("elapsed_s", 0.0) or 0.0)
                rews = [compute_reward(
                    r.get("cost_usd", 0.0) or 0.0,
                    base_lat,
                    r.get("is_correct", 0),
                    lam_e, LAMBDA_LATENCY_DEFAULT,
                ) for r in records]
                ds_avgs.append(sum(rews) / len(rews))
            sc_penalties.append(-(sum(ds_avgs) / len(ds_avgs)) if ds_avgs else 0.0)

        ax.plot(lambda_errors, sc_penalties,
                color=sc_color, lw=2.2, linestyle="-",
                label=f"SC (N={n})")

        # ---- base standalone model ----
        base_folder = _resolve_base_folder(model, model_folders)
        if base_folder is not None:
            base_recs = {stem: _load_records(base_folder, stem) for stem in DATASETS}
            base_penalties = []
            for lam_e in lambda_errors:
                ds_avgs = []
                for stem in DATASETS:
                    records = base_recs[stem]
                    if not records:
                        continue
                    rews = [compute_reward(
                        r.get("cost_usd", 0.0) or 0.0,
                        r.get("elapsed_s", 0.0) or 0.0,
                        r.get("is_correct", 0),
                        lam_e, LAMBDA_LATENCY_DEFAULT,
                    ) for r in records]
                    ds_avgs.append(sum(rews) / len(rews))
                base_penalties.append(-(sum(ds_avgs) / len(ds_avgs)) if ds_avgs else 0.0)
            ax.plot(lambda_errors, base_penalties,
                    color=base_color, lw=1.8, linestyle="--",
                    label=f"{model} (single-call)")

        ax.axvline(LAMBDA_ERROR_DEFAULT, color="gray", linestyle=":", lw=1,
                   label=f"default λ_e={LAMBDA_ERROR_DEFAULT}")
        ax.set_title(f"SC: {model}  (N={n})", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.85)

    for row in range(nrows):
        axes[row][0].set_ylabel("Economic Penalty  (lower = better)", fontsize=10)
    for col in range(ncols):
        axes[nrows - 1][col].set_xlabel("λ_error  (error penalty weight)", fontsize=10)
    for idx in range(len(sc_configs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Self-Consistency: Economic Penalty vs λ_error\n"
        "Solid = SC majority-vote  |  Dashed = single-call base model",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartSC4_selfcons_reward_vs_lambda_error.png")


# ---------------------------------------------------------------------------
# Chart SC6 — SC vs standalone accuracy comparison (grouped bar per dataset)
# ---------------------------------------------------------------------------

def chart_sc6_selfcons_accuracy_comparison(sc_data: dict, model_folders: dict):
    """
    SC6: Grouped bar chart — accuracy per dataset for every SC config and its
    matched standalone base model, all side-by-side so gains/losses are
    immediately visible.  One group of bars per dataset (x-axis); one bar per
    option.  SC configs use solid fill; their matched standalone uses a hatched
    fill with the same hue so the pairing is visually clear.
    """
    if not sc_data:
        print("  SC6: no self-consistency data, skipping.")
        return

    sc_configs = sorted(sc_data.keys())

    # Colour pairs: (SC solid colour, standalone hatched colour)
    palette = [
        ("#1B4F72", "#4A9CC7"),   # navy / light-blue
        ("#1D6A5A", "#52B09A"),   # teal / mint
        ("#6E2C00", "#C47A3A"),   # brown / tan
        ("#4A235A", "#9B6AB5"),   # purple / lavender
    ]

    # Build ordered list of (label, {stem: accuracy}, color, hatch)
    entries: list[tuple] = []
    for i, (model, n) in enumerate(sc_configs):
        sc_color, base_color = palette[i % len(palette)]
        stats = sc_data[(model, n)]
        entries.append((
            f"SC {model}\n(N={n})",
            stats["ds_accuracy"],
            sc_color,
            None,
        ))
        base_folder = _resolve_base_folder(model, model_folders)
        if base_folder is not None:
            base_acc = {}
            for stem in DATASETS:
                records = _load_records(base_folder, stem)
                if records:
                    base_acc[stem] = 100.0 * sum(r.get("is_correct", 0) for r in records) / len(records)
            entries.append((
                f"{model}\n(single-call)",
                base_acc,
                base_color,
                "////",
            ))

    ds_labels = [DS_LABELS[s] for s in DATASETS]
    n_entries = len(entries)
    x = np.arange(len(DATASETS))
    width = 0.8 / n_entries

    fig, ax = plt.subplots(figsize=(max(11, n_entries * 1.5), 6))

    for i, (label, acc_map, color, hatch) in enumerate(entries):
        accs = [acc_map.get(s, 0) for s in DATASETS]
        offset = (i - n_entries / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, accs, width=width * 0.9,
            color=color, hatch=hatch, label=label,
            edgecolor="white" if hatch is None else "#333333",
            linewidth=0.6, zorder=3,
        )
        for bar, v in zip(bars, accs):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 1.0,
                    f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                    color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 125)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, labels=labels_leg,
        fontsize=9, framealpha=0.88,
        loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
    )
    fig.suptitle(
        "Self-Consistency vs Single-Call: Accuracy per Dataset\n"
        "Solid bars = SC majority-vote  |  Hatched bars = single-call base model",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "chartSC6_selfcons_accuracy_comparison.png")


# ===========================================================================
# ROUTER CHARTS (R prefix)
# ===========================================================================

def _find_router_folders() -> list[pathlib.Path]:
    if not OPT_RESULTS_DIR.exists():
        return []
    return sorted(d for d in OPT_RESULTS_DIR.iterdir()
                  if d.is_dir() and d.name.startswith("router__"))


def _parse_router_name(name: str) -> "tuple[str, str, str] | None":
    """Parse 'router__rtr-{rtag}__small-{stag}__large-{ltag}'
    → (router_model, small_model, large_model) using display names."""
    m = re.match(r"router__rtr-(\w+)__small-(\w+)__large-(\w+)$", name)
    if not m:
        return None
    to_model = _CASCADE_TAG_TO_MODEL
    return (
        to_model.get(m.group(1), m.group(1)),
        to_model.get(m.group(2), m.group(2)),
        to_model.get(m.group(3), m.group(3)),
    )


def _find_router_folder(cfg: tuple) -> pathlib.Path:
    """Return the optimization_results folder for a (rtr, small, large) router config."""
    rtr, small, large = cfg
    reverse = {v: k for k, v in _CASCADE_TAG_TO_MODEL.items()}
    name = (f"router__rtr-{reverse.get(rtr, rtr)}"
            f"__small-{reverse.get(small, small)}"
            f"__large-{reverse.get(large, large)}")
    return OPT_RESULTS_DIR / name


def _load_router_data() -> dict:
    """
    Returns {(router_model, small_model, large_model): {
        'n_total':           int,
        'macro_accuracy':    float,
        'avg_cost':          float,
        'avg_latency':       float,
        'avg_reward':        float | None,
        'pct_large':         float,   # % queries sent to large model
        'ds_accuracy':       {stem: float},
        'ds_cost':           {stem: float},
        'ds_latency':        {stem: float},
        'ds_reward':         {stem: float | None},
        'ds_pct_large':      {stem: float},
    }}
    """
    data: dict = {}
    for folder in _find_router_folders():
        parsed = _parse_router_name(folder.name)
        if parsed is None:
            continue
        cfg = parsed
        ds_acc, ds_cost, ds_lat, ds_rew, ds_pct = {}, {}, {}, {}, {}
        total_n = total_large = 0

        for stem in DATASETS:
            records = _load_records(folder, stem)
            if not records:
                continue
            n_stem = len(records)
            ds_acc[stem]  = 100.0 * sum(r.get("is_correct", 0) for r in records) / n_stem
            ds_cost[stem] = sum((r.get("cost_usd", 0.0) or 0.0) for r in records) / n_stem
            ds_lat[stem]  = sum((r.get("elapsed_s", 0.0) or 0.0) for r in records) / n_stem
            valid_r = [r.get("economic_reward") for r in records if r.get("economic_reward") is not None]
            ds_rew[stem]  = sum(valid_r) / len(valid_r) if valid_r else None
            n_large = sum(1 for r in records if r.get("routing_decision") == "large")
            ds_pct[stem]  = 100.0 * n_large / n_stem
            total_n     += n_stem
            total_large += n_large

        if not ds_acc:
            continue

        data[cfg] = {
            "n_total":        total_n,
            "macro_accuracy": sum(ds_acc.values()) / len(ds_acc),
            "avg_cost":       sum(ds_cost.values()) / len(ds_cost),
            "avg_latency":    sum(ds_lat.values()) / len(ds_lat),
            "avg_reward":     (sum(v for v in ds_rew.values() if v is not None) /
                               sum(1 for v in ds_rew.values() if v is not None))
                              if any(v is not None for v in ds_rew.values()) else None,
            "pct_large":      100.0 * total_large / total_n if total_n else 0.0,
            "ds_accuracy":    ds_acc,
            "ds_cost":        ds_cost,
            "ds_latency":     ds_lat,
            "ds_reward":      ds_rew,
            "ds_pct_large":   ds_pct,
        }
    return data


# ---------------------------------------------------------------------------
# Chart R1 — Router overview: accuracy + routing split per dataset
# ---------------------------------------------------------------------------

def chart_r1_router_overview(router_data: dict, model_folders: dict):
    """
    R1: One subplot per router config.
    x-axis = datasets.
    Left y-axis  (lines): router accuracy vs small-model standalone vs large-model standalone.
    Right y-axis (bars):  % of queries routed to the large model.
    """
    if not router_data:
        print("  R1: no router data, skipping.")
        return

    configs = sorted(router_data.keys())
    ncols   = 2
    nrows   = (len(configs) + ncols - 1) // ncols

    bar_color    = "#BBBBBB"
    router_color = "#1B4F72"   # navy  — router accuracy
    small_color  = "#2ca02c"   # green — small model
    large_color  = "#d62728"   # red   — large model

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5.5 * nrows), squeeze=False)

    for idx, (rtr, small, large) in enumerate(configs):
        row, col = divmod(idx, ncols)
        ax_l = axes[row][col]
        ax_r = ax_l.twinx()

        stats  = router_data[(rtr, small, large)]
        stems  = [s for s in DATASETS if s in stats["ds_accuracy"]]
        xi     = np.arange(len(stems))
        labels = [DS_LABELS[s] for s in stems]

        rtr_accs   = [stats["ds_accuracy"][s]  for s in stems]
        pct_large  = [stats["ds_pct_large"][s] for s in stems]

        # Pre-fetch standalone accuracies
        small_accs, large_accs = [], []
        for s in stems:
            sf = _resolve_base_folder(small, model_folders)
            lf = _resolve_base_folder(large, model_folders)
            def _acc(folder, stem):
                recs = _load_records(folder, stem) if folder else []
                return 100.0 * sum(r.get("is_correct", 0) for r in recs) / len(recs) if recs else float("nan")
            small_accs.append(_acc(sf, s))
            large_accs.append(_acc(lf, s))

        # Right axis: % routed to large
        ax_r.bar(xi, pct_large, width=0.45, color=bar_color, alpha=0.30, zorder=2)
        for xv, v in zip(xi, pct_large):
            ax_r.text(xv, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                      fontsize=7, color="#888888")
        ax_r.set_ylabel("% Queries → Large Model", color="#888888", fontsize=9)
        ax_r.tick_params(axis="y", labelcolor="#888888")
        ax_r.set_ylim(0, 140)
        ax_r.yaxis.grid(False)
        ax_r.spines["right"].set_visible(True)

        # Left axis: accuracy lines
        ax_l.plot(xi, rtr_accs,   marker="o", lw=2,   color=router_color,
                  label=f"Router (via {rtr})", zorder=5)
        ax_l.plot(xi, small_accs, marker="s", lw=1.5, color=small_color,
                  linestyle="--", label=f"{small} (small, single-call)", zorder=5)
        ax_l.plot(xi, large_accs, marker="^", lw=1.5, color=large_color,
                  linestyle=":",  label=f"{large} (large, single-call)", zorder=5)

        for xv, v in zip(xi, rtr_accs):
            if not np.isnan(v):
                ax_l.text(xv, v + 2, f"{v:.0f}%", ha="center", va="bottom",
                          fontsize=7, color=router_color)

        ax_l.set_ylabel("Accuracy (%)", fontsize=10)
        ax_l.set_ylim(0, 120)
        ax_l.set_xticks(xi)
        ax_l.set_xticklabels(labels, fontsize=9)
        ax_l.set_xlabel("Dataset", fontsize=10)
        ax_l.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_l.set_axisbelow(True)
        ax_l.set_title(f"Router: {rtr}  |  {small} → {large}",
                       fontsize=10, fontweight="bold")
        bar_patch = mpatches.Patch(color=bar_color, alpha=0.5,
                                   label="% routed to large (right axis)")
        handles, _ = ax_l.get_legend_handles_labels()
        ax_l.legend(handles=handles + [bar_patch], fontsize=8,
                    framealpha=0.88, loc="upper right")

    for idx in range(len(configs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "LLM-as-Router — Accuracy vs Routing Distribution per Dataset\n"
        "Grey bars = % queries sent to large model  |  "
        "Lines: router vs small/large standalone",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartR1_router_overview.png")


# ---------------------------------------------------------------------------
# Chart R2 — Router dual heatmap: reward + cost
# ---------------------------------------------------------------------------

def chart_r2_router_dual_heatmap(router_data: dict, model_folders: dict):
    """
    R2: Side-by-side heatmaps (reward | cost).
    Rows = router configs + standalone models.  Columns = datasets.
    """
    if not router_data:
        print("  R2: no router data, skipping.")
        return

    # Standalone reward + cost
    standalone_ds_rew:  dict[str, dict] = {}
    standalone_ds_cost: dict[str, dict] = {}
    for model, folder in sorted(model_folders.items()):
        rmap, cmap = {}, {}
        for stem in DATASETS:
            records = _load_records(folder, stem)
            if not records:
                continue
            rews  = [compute_reward(r.get("cost_usd", 0.0) or 0.0,
                                     r.get("elapsed_s", 0.0) or 0.0,
                                     r.get("is_correct", 0),
                                     LAMBDA_ERROR_DEFAULT, LAMBDA_LATENCY_DEFAULT)
                     for r in records]
            rmap[stem]  = sum(rews) / len(rews)
            cmap[stem]  = sum(r.get("cost_usd", 0.0) or 0.0 for r in records) / len(records) * 1000
        standalone_ds_rew[model]  = rmap
        standalone_ds_cost[model] = cmap

    router_configs = sorted(router_data.keys())
    row_labels, reward_mat, reward_ann, cost_mat, cost_ann = [], [], [], [], []

    for cfg in router_configs:
        rtr, small, large = cfg
        stats = router_data[cfg]
        row_labels.append(f"Router {rtr}\n({small}→{large})")
        rr, ra, cr, ca = [], [], [], []
        for stem in DATASETS:
            rv = stats["ds_reward"].get(stem)
            cv = stats["ds_cost"].get(stem)
            cv_ms = cv * 1000 if cv is not None else None
            rr.append(rv    if rv    is not None else float("nan"))
            cr.append(cv_ms if cv_ms is not None else float("nan"))
            ra.append(f"{rv:.3f}"        if rv    is not None else "N/A")
            ca.append(f"{cv_ms:.3f} m$"  if cv_ms is not None else "N/A")
        reward_mat.append(rr); reward_ann.append(ra)
        cost_mat.append(cr);   cost_ann.append(ca)

    col_labels = [DS_LABELS[s] for s in DATASETS]
    rmat = np.array(reward_mat, dtype=float)
    cmat = np.array(cost_mat,   dtype=float)

    h = max(4, len(row_labels) * 1.3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, h))

    im1 = _build_heatmap(ax1, rmat, reward_ann, row_labels, col_labels,
                         "Avg Economic Reward per Dataset\n(higher = better)",
                         "RdYlGn", "Avg Reward")
    ax1.set_ylabel("Configuration", fontsize=10)
    ax1.set_xlabel("Dataset", fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Avg Reward")

    im2 = _build_heatmap(ax2, cmat, cost_ann, row_labels, col_labels,
                         "Avg Cost per Query (m$)\n(lower = better)",
                         "RdYlGn_r", "Avg Cost (m$)")
    ax2.set_xlabel("Dataset", fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Avg Cost (m$)")

    fig.suptitle("LLM-as-Router: Reward & Cost by Dataset",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "chartR2_router_dual_heatmap.png")


# ---------------------------------------------------------------------------
# Chart R3 — Router best-option decision map (λ_error × λ_latency)
# ---------------------------------------------------------------------------

def chart_r3_router_best_config_heatmap(router_data: dict, model_folders: dict):
    """
    R3: Decision map (λ_error × λ_latency) comparing all router configs against
    all standalone models.  Same visual convention as C3 / SC3.
    """
    if not router_data:
        print("  R3: no router data, skipping.")
        return

    # Standalone: consistent with MODEL_COLORS (ColorBrewer Paired)
    # Router configs: distinct accent family (deep-violet, teal, dark-magenta, amber-gold)
    _router_palette = [
        "#4527A0",  # deep violet
        "#00838F",  # dark teal/cyan
        "#AD1457",  # dark magenta
        "#FFB300",  # amber gold
    ]

    router_configs    = sorted(router_data.keys())
    standalone_models = sorted(model_folders.keys())
    rtr_colors = {cfg: _router_palette[i % len(_router_palette)]
                  for i, cfg in enumerate(router_configs)}
    sa_colors  = _model_color_map(standalone_models)

    rtr_records = {
        cfg: {stem: _load_records(_find_router_folder(cfg), stem) for stem in DATASETS}
        for cfg in router_configs
    }
    sa_records = {
        m: {stem: _load_records(folder, stem) for stem in DATASETS}
        for m, folder in model_folders.items()
    }

    lam_errors = HEATMAP_LAM_ERROR
    lam_lats   = HEATMAP_LAM_LAT

    options_macro = []
    for m in standalone_models:
        s = _option_stats_macro(sa_records[m])
        if s is not None:
            options_macro.append((("standalone", m), *s))
    for cfg in router_configs:
        s = _option_stats_macro(rtr_records[cfg])
        if s is not None:
            options_macro.append((("router", cfg), *s))
    grid = _build_best_grid(options_macro, lam_errors, lam_lats)

    all_keys = ([("standalone", m) for m in standalone_models] +
                [("router", cfg)   for cfg in router_configs])
    key_to_idx   = {k: idx for idx, k in enumerate(all_keys)}
    key_to_color = {
        **{("standalone", m):   sa_colors[m]   for m in standalone_models},
        **{("router",     cfg): rtr_colors[cfg] for cfg in router_configs},
    }

    grid_int = np.array(
        [[key_to_idx.get(grid[i, j], 0) for j in range(len(lam_errors))]
         for i in range(len(lam_lats))],
        dtype=float,
    )
    cmap = ListedColormap([key_to_color[k] for k in all_keys])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(grid_int, aspect="auto", origin="lower", cmap=cmap,
              vmin=-0.5, vmax=len(all_keys) - 0.5, interpolation="nearest")

    x_ticks = np.linspace(0, len(lam_errors) - 1, 7, dtype=int)
    y_ticks = np.linspace(0, len(lam_lats)   - 1, 7, dtype=int)
    ax.set_xticks(x_ticks); ax.set_xticklabels([f"{lam_errors[t]:.2f}" for t in x_ticks])
    ax.set_yticks(y_ticks); ax.set_yticklabels([f"{lam_lats[t]:.4f}"   for t in y_ticks])
    ax.set_xlabel("λ_error  (error penalty weight)", fontsize=11)
    ax.set_ylabel("λ_latency  (latency penalty weight)", fontsize=11)
    ax.set_title(
        "Best Option Decision Map: LLM-as-Router vs Standalone\n"
        "(cell colour = option with highest macro-avg reward at those penalty weights)",
        fontsize=12, fontweight="bold",
    )

    def_e = int(np.argmin(np.abs(lam_errors - LAMBDA_ERROR_DEFAULT)))
    def_l = int(np.argmin(np.abs(lam_lats   - LAMBDA_LATENCY_DEFAULT)))
    ax.plot(def_e, def_l, "w*", markersize=14,
            label=f"default (λ_e={LAMBDA_ERROR_DEFAULT}, λ_l={LAMBDA_LATENCY_DEFAULT})")

    winning_keys = sorted(
        {grid[i, j] for i in range(len(lam_lats)) for j in range(len(lam_errors))
         if grid[i, j] is not None},
        key=lambda k: (0 if k[0] == "standalone" else 1, k[1]),
    )
    legend_handles = []
    sa_patches = [mpatches.Patch(color=key_to_color[k], label=k[1])
                  for k in winning_keys if k[0] == "standalone"]
    rtr_patches = [mpatches.Patch(color=key_to_color[k],
                                   label=f"Router {k[1][0]}  ({k[1][1]}→{k[1][2]})")
                   for k in winning_keys if k[0] == "router"]
    if sa_patches:
        legend_handles += [mpatches.Patch(color="none", label="— Standalone models —")]
        legend_handles += sa_patches
    if rtr_patches:
        legend_handles += [mpatches.Patch(color="none", label="— Router configs —")]
        legend_handles += rtr_patches

    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.85,
              title="Best option", title_fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    _save(fig, "chartR3_router_best_config_heatmap.png")


# ---------------------------------------------------------------------------
# Chart R6 — Per-dataset decision map: best router/standalone option per dataset
# ---------------------------------------------------------------------------

def chart_r6_router_per_dataset_decision_map(router_data: dict, model_folders: dict):
    """
    R6: One subplot per dataset (5 total).
    Each subplot is a decision map on the (λ_error × λ_latency) plane where
    the cell colour shows which router config OR standalone model achieves the
    highest reward on THAT SINGLE DATASET.
    Same colour convention as R3.
    """
    if not router_data:
        print("  R6: no router data, skipping.")
        return

    _router_palette = [
        "#4527A0",  # deep violet
        "#00838F",  # dark teal/cyan
        "#AD1457",  # dark magenta
        "#FFB300",  # amber gold
    ]
    router_configs    = sorted(router_data.keys())
    standalone_models = sorted(model_folders.keys())
    rtr_colors = {cfg: _router_palette[i % len(_router_palette)]
                  for i, cfg in enumerate(router_configs)}
    sa_colors  = _model_color_map(standalone_models)

    rtr_records = {cfg: {stem: _load_records(_find_router_folder(cfg), stem) for stem in DATASETS}
                   for cfg in router_configs}
    sa_records  = {m: {stem: _load_records(folder, stem) for stem in DATASETS}
                   for m, folder in model_folders.items()}

    ordered_keys = (
        [("standalone", m)   for m in standalone_models] +
        [("router",     cfg) for cfg in router_configs]
    )
    key_to_color = {
        **{("standalone", m):   sa_colors[m]   for m in standalone_models},
        **{("router",     cfg): rtr_colors[cfg] for cfg in router_configs},
    }
    key_to_idx = {k: i for i, k in enumerate(ordered_keys)}

    lam_errors = HEATMAP_LAM_ERROR
    lam_lats   = HEATMAP_LAM_LAT

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10), squeeze=False)
    global_winning_keys: set = set()

    for ds_idx, stem in enumerate(DATASETS):
        row, col = divmod(ds_idx, ncols)
        ax = axes[row][col]

        options_ds = []
        for m in standalone_models:
            s = _option_stats_single(sa_records[m][stem])
            if s is not None:
                options_ds.append((("standalone", m), *s))
        for cfg in router_configs:
            s = _option_stats_single(rtr_records[cfg][stem])
            if s is not None:
                options_ds.append((("router", cfg), *s))
        grid = _build_best_grid(options_ds, lam_errors, lam_lats)
        global_winning_keys.update(k for k in grid.flat if k is not None)

        grid_int = np.array(
            [[key_to_idx.get(grid[i, j], 0) for j in range(len(lam_errors))]
             for i in range(len(lam_lats))],
            dtype=float,
        )
        cmap = ListedColormap([key_to_color[k] for k in ordered_keys])
        ax.imshow(grid_int, aspect="auto", origin="lower", cmap=cmap,
                  vmin=-0.5, vmax=len(ordered_keys) - 0.5, interpolation="nearest")

        x_ticks = np.linspace(0, len(lam_errors) - 1, 5, dtype=int)
        y_ticks = np.linspace(0, len(lam_lats)   - 1, 5, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{lam_errors[t]:.2f}" for t in x_ticks], fontsize=7)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{lam_lats[t]:.4f}" for t in y_ticks], fontsize=7)
        ax.set_title(DS_LABELS[stem], fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel("λ_latency", fontsize=9)
        if row == nrows - 1 or ds_idx == len(DATASETS) - 1:
            ax.set_xlabel("λ_error", fontsize=9)
        def_e = int(np.argmin(np.abs(lam_errors - LAMBDA_ERROR_DEFAULT)))
        def_l = int(np.argmin(np.abs(lam_lats   - LAMBDA_LATENCY_DEFAULT)))
        ax.plot(def_e, def_l, "w*", markersize=10)

    # Shared legend in the unused 6th cell
    axes[1][2].set_visible(True)
    axes[1][2].axis("off")
    winning_ordered = [k for k in ordered_keys if k in global_winning_keys]
    legend_handles = []
    sa_win  = [k for k in winning_ordered if k[0] == "standalone"]
    rtr_win = [k for k in winning_ordered if k[0] == "router"]
    if sa_win:
        legend_handles.append(mpatches.Patch(color="none", label="— Standalone —"))
        for k in sa_win:
            legend_handles.append(mpatches.Patch(color=key_to_color[k], label=k[1]))
    if rtr_win:
        legend_handles.append(mpatches.Patch(color="none", label="— Router configs —"))
        for k in rtr_win:
            legend_handles.append(mpatches.Patch(
                color=key_to_color[k],
                label=f"Router {k[1][0]}  ({k[1][1]}→{k[1][2]})"
            ))
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="gray", linestyle="none",
                   markersize=10, label=f"default (λe={LAMBDA_ERROR_DEFAULT}, λl={LAMBDA_LATENCY_DEFAULT})")
    )
    axes[1][2].legend(handles=legend_handles, loc="center", fontsize=8,
                      framealpha=0.85, title="Best option", title_fontsize=9)

    fig.suptitle(
        "Router: Best Option Decision Map — Per Dataset\n"
        "(cell = option with highest reward at those λ_error × λ_latency weights)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "chartR6_router_per_dataset_decision_map.png")


# ---------------------------------------------------------------------------
# Chart R4 — Router reward vs λ_error (one panel per router config)
# ---------------------------------------------------------------------------

def chart_r4_router_reward_vs_lambda_error(router_data: dict, model_folders: dict):
    """
    R4: One subplot per router config.
    Solid line  = router reward vs λ_error.
    Dashed line = small-model standalone.
    Dotted line = large-model standalone.
    λ_latency held at default throughout.
    """
    if not router_data:
        print("  R4: no router data, skipping.")
        return

    router_configs = sorted(router_data.keys())
    lambda_errors = np.unique(np.concatenate([
        np.linspace(0.01, 1.0,  80),
        np.linspace(1.0,  5.0, 100),
        np.linspace(5.0,  15.0, 60),
        np.linspace(15.0, 50.0, 40),
    ]))

    router_color = "#1B4F72"
    small_color  = "#2ca02c"
    large_color  = "#d62728"

    ncols = 2
    nrows = (len(router_configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for idx, (rtr, small, large) in enumerate(router_configs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        rtr_folder = _find_router_folder((rtr, small, large))
        rtr_recs   = {stem: _load_records(rtr_folder, stem) for stem in DATASETS}

        rtr_penalties = []
        for lam_e in lambda_errors:
            ds_avgs = []
            for stem in DATASETS:
                records = rtr_recs[stem]
                if not records:
                    continue
                rews = [compute_reward(r.get("cost_usd", 0.0) or 0.0,
                                       r.get("elapsed_s", 0.0) or 0.0,
                                       r.get("is_correct", 0),
                                       lam_e, LAMBDA_LATENCY_DEFAULT)
                        for r in records]
                ds_avgs.append(sum(rews) / len(rews))
            rtr_penalties.append(-(sum(ds_avgs) / len(ds_avgs)) if ds_avgs else 0.0)

        ax.plot(lambda_errors, rtr_penalties,
                color=router_color, lw=2.2, linestyle="-",
                label=f"Router (via {rtr})")

        for label, model, color, ls in [
            (f"{small} (small)", small, small_color, "--"),
            (f"{large} (large)", large, large_color, ":"),
        ]:
            base_folder = _resolve_base_folder(model, model_folders)
            if base_folder is None:
                continue
            base_recs = {stem: _load_records(base_folder, stem) for stem in DATASETS}
            base_penalties = []
            for lam_e in lambda_errors:
                ds_avgs = []
                for stem in DATASETS:
                    records = base_recs[stem]
                    if not records:
                        continue
                    rews = [compute_reward(r.get("cost_usd", 0.0) or 0.0,
                                           r.get("elapsed_s", 0.0) or 0.0,
                                           r.get("is_correct", 0),
                                           lam_e, LAMBDA_LATENCY_DEFAULT)
                            for r in records]
                    ds_avgs.append(sum(rews) / len(rews))
                base_penalties.append(-(sum(ds_avgs) / len(ds_avgs)) if ds_avgs else 0.0)
            ax.plot(lambda_errors, base_penalties,
                    color=color, lw=1.8, linestyle=ls, label=label)

        ax.axvline(LAMBDA_ERROR_DEFAULT, color="gray", linestyle=":", lw=1,
                   label=f"default λ_e={LAMBDA_ERROR_DEFAULT}")
        ax.set_title(f"Router: {rtr}  |  {small} → {large}",
                     fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.85)

    for row in range(nrows):
        axes[row][0].set_ylabel("Economic Penalty  (lower = better)", fontsize=10)
    for col in range(ncols):
        axes[nrows - 1][col].set_xlabel("λ_error  (error penalty weight)", fontsize=10)
    for idx in range(len(router_configs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "LLM-as-Router: Economic Penalty vs λ_error\n"
        "Solid = router  |  Dashed = small standalone  |  Dotted = large standalone",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartR4_router_reward_vs_lambda_error.png")


# ---------------------------------------------------------------------------
# Chart R5 — Router reward vs λ_latency (one panel per router config)
# ---------------------------------------------------------------------------

def chart_r5_router_reward_vs_lambda_latency(router_data: dict, model_folders: dict):
    """
    R5: One subplot per router config.
    Solid line  = router reward vs λ_latency.
    Dashed line = small-model standalone.
    Dotted line = large-model standalone.
    λ_error held at default throughout.  Non-linear y-axis identical to C5.
    """
    if not router_data:
        print("  R5: no router data, skipping.")
        return

    router_configs = sorted(router_data.keys())
    lambda_lats = np.unique(np.concatenate([
        np.linspace(0.0001, 0.01,  100),
        np.linspace(0.01,   0.1,   100),
        np.linspace(0.1,    0.5,   100),
        np.linspace(0.5,    1.0,    60),
        np.linspace(1.0,    10.0,   60),
    ]))

    router_color = "#1B4F72"
    small_color  = "#2ca02c"
    large_color  = "#d62728"

    ytick_vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 30, 50]
    ytick_pos  = list(range(len(ytick_vals)))
    def to_pos(v):
        return float(np.interp(v, ytick_vals, ytick_pos))

    ncols = 2
    nrows = (len(router_configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for idx, (rtr, small, large) in enumerate(router_configs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        rtr_folder = _find_router_folder((rtr, small, large))
        rtr_recs   = {stem: _load_records(rtr_folder, stem) for stem in DATASETS}

        rtr_raw = []
        for lam_l in lambda_lats:
            ds_avgs = []
            for stem in DATASETS:
                records = rtr_recs[stem]
                if not records:
                    continue
                rews = [compute_reward(r.get("cost_usd", 0.0) or 0.0,
                                       r.get("elapsed_s", 0.0) or 0.0,
                                       r.get("is_correct", 0),
                                       LAMBDA_ERROR_DEFAULT, lam_l)
                        for r in records]
                ds_avgs.append(sum(rews) / len(rews))
            rtr_raw.append(-(sum(ds_avgs) / len(ds_avgs)) if ds_avgs else 0.0)

        ax.plot(lambda_lats, [to_pos(max(0.0, v)) for v in rtr_raw],
                color=router_color, lw=2.2, linestyle="-",
                label=f"Router (via {rtr})")

        for label, model, color, ls in [
            (f"{small} (small)", small, small_color, "--"),
            (f"{large} (large)", large, large_color, ":"),
        ]:
            base_folder = _resolve_base_folder(model, model_folders)
            if base_folder is None:
                continue
            base_recs = {stem: _load_records(base_folder, stem) for stem in DATASETS}
            base_raw = []
            for lam_l in lambda_lats:
                ds_avgs = []
                for stem in DATASETS:
                    records = base_recs[stem]
                    if not records:
                        continue
                    rews = [compute_reward(r.get("cost_usd", 0.0) or 0.0,
                                           r.get("elapsed_s", 0.0) or 0.0,
                                           r.get("is_correct", 0),
                                           LAMBDA_ERROR_DEFAULT, lam_l)
                            for r in records]
                    ds_avgs.append(sum(rews) / len(rews))
                base_raw.append(-(sum(ds_avgs) / len(ds_avgs)) if ds_avgs else 0.0)
            ax.plot(lambda_lats, [to_pos(max(0.0, v)) for v in base_raw],
                    color=color, lw=1.8, linestyle=ls, label=label)

        ax.axvline(LAMBDA_LATENCY_DEFAULT, color="gray", linestyle=":", lw=1,
                   label=f"default λ_l={LAMBDA_LATENCY_DEFAULT}")
        ax.set_title(f"Router: {rtr}  |  {small} → {large}",
                     fontsize=10, fontweight="bold")
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_vals)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.85)

    for row in range(nrows):
        axes[row][0].set_ylabel("Economic Penalty  (lower = better)", fontsize=10)
    for col in range(ncols):
        axes[nrows - 1][col].set_xlabel("λ_latency  (latency penalty weight)", fontsize=10)
    for idx in range(len(router_configs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "LLM-as-Router: Economic Penalty vs λ_latency\n"
        "Solid = router  |  Dashed = small standalone  |  Dotted = large standalone",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "chartR5_router_reward_vs_lambda_latency.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nLoading results...")
    model_folders = _find_model_folders()
    if not model_folders:
        print("No result folders found. Run tester.py first.")
        raise SystemExit(1)

    print(f"  Models found: {', '.join(sorted(model_folders))}\n")

    print("Computing statistics (default \u03bb)...")
    stats = build_model_stats()

    print("\nGenerating base model charts...")
    chart1_accuracy(stats)
    chart2_accuracy_per_dataset(stats)
    chart3_cost_vs_accuracy(stats)
    chart4_reward(stats)
    chart5_reward_vs_lambda_error(model_folders)
    chart6_reward_vs_lambda_latency(model_folders)
    chart7_best_model_heatmap(model_folders)
    chart8_best_model_heatmap_per_dataset(model_folders)

    print("\nGenerating cascade charts...")
    cascade_data = _load_cascade_data()
    if cascade_data:
        print(f"  Cascade configs found: {len(cascade_data)}")
        chart_c1_cascade_combined_overview(cascade_data)
        chart_c2_cascade_dual_heatmap(cascade_data)
        chart_c3_cascade_best_config_heatmap(model_folders)
        chart_c4_cascade_reward_vs_lambda_error()
        chart_c5_cascade_reward_vs_lambda_latency()
        chart_c6_cascade_per_dataset_decision_map(model_folders)
    else:
        print("  No cascade results found. Run optimizer.py --cascade first.")

    print("\nGenerating self-consistency charts...")
    sc_data = _load_selfcons_data()
    if sc_data:
        print(f"  SC configs found: {len(sc_data)}")
        chart_sc1_selfcons_overview(sc_data, model_folders)
        chart_sc2_selfcons_dual_heatmap(sc_data, model_folders)
        chart_sc3_selfcons_best_config_heatmap(sc_data, model_folders)
        chart_sc4_selfcons_reward_vs_lambda_error(sc_data, model_folders)
        chart_sc6_selfcons_accuracy_comparison(sc_data, model_folders)
    else:
        print("  No self-consistency results found. Run optimizer.py --selfcons first.")

    print("\nGenerating router charts...")
    router_data = _load_router_data()
    if router_data:
        print(f"  Router configs found: {len(router_data)}")
        chart_r1_router_overview(router_data, model_folders)
        chart_r2_router_dual_heatmap(router_data, model_folders)
        chart_r3_router_best_config_heatmap(router_data, model_folders)
        chart_r4_router_reward_vs_lambda_error(router_data, model_folders)
        chart_r5_router_reward_vs_lambda_latency(router_data, model_folders)
        chart_r6_router_per_dataset_decision_map(router_data, model_folders)
    else:
        print("  No router results found. Run optimizer.py --router first.")

    print(f"\nAll charts saved to '{CHARTS_DIR}/'")

