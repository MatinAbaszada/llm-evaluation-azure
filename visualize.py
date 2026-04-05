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
"""

import json
import pathlib
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

RESULTS_DIR   = pathlib.Path("results")
CHARTS_DIR    = pathlib.Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

# Reasoning vs non-reasoning classification
REASONING_MODELS = {"o3-mini"}


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

MODEL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"
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

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, accs, color=[cmap[m] for m in models], width=0.55, zorder=3)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_ylabel("Macro-Avg Accuracy (%)")
    ax.set_title("Model Comparison — Overall Accuracy\n(macro-average across all benchmarks)")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
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

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, model in enumerate(models):
        accs = [stats[model]["ds_accuracy"].get(ds, 0) * 100 for ds in datasets]
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, accs, width=width * 0.9,
               color=cmap[model], label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Dataset Accuracy by Model")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
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
    ax.set_ylabel("Macro-Avg Accuracy (%)")
    ax.set_title("Cost vs Accuracy — Pareto Frontier")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "chart3_cost_vs_accuracy.png")


# ---------------------------------------------------------------------------
# Chart 4 — Macro-avg economic reward per model (bar)
# ---------------------------------------------------------------------------

def chart4_reward(stats: dict):
    models  = sorted(stats)
    rewards = [stats[m]["macro_reward"] for m in models]
    cmap    = _model_color_map(models)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, rewards, color=[cmap[m] for m in models], width=0.55, zorder=3)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_ylabel("Macro-Avg Economic Reward")
    ax.set_title(
        f"Model Comparison — Economic Reward\n"
        f"(λ_error={LAMBDA_ERROR_DEFAULT}, λ_latency={LAMBDA_LATENCY_DEFAULT})"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "chart4_reward.png")


# ---------------------------------------------------------------------------
# Chart 5 — Reward vs λ_error (one line per model)
# ---------------------------------------------------------------------------

def chart5_reward_vs_lambda_error(model_folders: dict):
    lambda_errors = np.linspace(0.1, 5.0, 40)
    models = sorted(model_folders)
    cmap   = _model_color_map(models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        folder = model_folders[model]
        rewards = []
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
            rewards.append(sum(ds_avgs) / len(ds_avgs) if ds_avgs else 0.0)
        ax.plot(lambda_errors, rewards, label=model, color=cmap[model], lw=1.8)

    ax.axvline(LAMBDA_ERROR_DEFAULT, color="gray", linestyle=":", lw=1, label=f"default λ_error={LAMBDA_ERROR_DEFAULT}")
    ax.set_xlabel("λ_error  (error penalty weight)")
    ax.set_ylabel("Macro-Avg Economic Reward")
    ax.set_title("Reward Sensitivity to Error Penalty (λ_error)")
    ax.legend(fontsize=8, framealpha=0.8)
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

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        folder = model_folders[model]
        rewards = []
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
            rewards.append(sum(ds_avgs) / len(ds_avgs) if ds_avgs else 0.0)
        ax.plot(lambda_lats, rewards, label=model, color=cmap[model], lw=1.8)

    ax.axvline(LAMBDA_LATENCY_DEFAULT, color="gray", linestyle=":", lw=1, label=f"default λ_latency={LAMBDA_LATENCY_DEFAULT}")
    ax.set_xlabel("λ_latency  (latency penalty weight)")
    ax.set_ylabel("Macro-Avg Economic Reward")
    ax.set_title("Reward Sensitivity to Latency Penalty (λ_latency)")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "chart6_reward_vs_lambda_latency.png")


# ---------------------------------------------------------------------------
# Chart 7 — Best model heatmap over (λ_error × λ_latency)
# ---------------------------------------------------------------------------

def chart7_best_model_heatmap(model_folders: dict):
    models     = sorted(model_folders)
    lam_errors = np.linspace(0.1, 5.0, 25)
    lam_lats   = np.linspace(0.001, 0.5, 25)
    cmap_m     = _model_color_map(models)

    # Pre-load all records once
    all_records: dict = {
        model: {stem: _load_records(folder, stem) for stem in DATASETS}
        for model, folder in model_folders.items()
    }

    grid = np.empty((len(lam_lats), len(lam_errors)), dtype=object)

    for j, lam_e in enumerate(lam_errors):
        for i, lam_l in enumerate(lam_lats):
            best_model  = None
            best_reward = -np.inf
            for model in models:
                ds_avgs = []
                for stem in DATASETS:
                    records = all_records[model][stem]
                    if not records:
                        continue
                    rews = [
                        compute_reward(
                            r.get("cost_usd", 0.0) or 0.0,
                            r.get("elapsed_s", 0.0) or 0.0,
                            r.get("is_correct", 0),
                            lam_e, lam_l
                        ) for r in records
                    ]
                    ds_avgs.append(sum(rews) / len(rews))
                macro = sum(ds_avgs) / len(ds_avgs) if ds_avgs else -np.inf
                if macro > best_reward:
                    best_reward = macro
                    best_model  = model
            grid[i, j] = best_model

    # Encode to int for imshow
    model_idx = {m: k for k, m in enumerate(models)}
    grid_int  = np.vectorize(model_idx.get)(grid).astype(float)

    colors  = [cmap_m[m] for m in models]
    cmap    = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10, 7))
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
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    # Legend patches
    patches = [mpatches.Patch(color=cmap_m[m], label=m) for m in models]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.8, title="Best model")

    fig.tight_layout()
    _save(fig, "chart7_best_model_heatmap.png")


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

    print("Computing statistics (default λ)...")
    stats = build_model_stats()

    print("\nGenerating charts...")
    chart1_accuracy(stats)
    chart2_accuracy_per_dataset(stats)
    chart3_cost_vs_accuracy(stats)
    chart4_reward(stats)
    chart5_reward_vs_lambda_error(model_folders)
    chart6_reward_vs_lambda_latency(model_folders)
    chart7_best_model_heatmap(model_folders)

    print(f"\nAll charts saved to '{CHARTS_DIR}/'")
