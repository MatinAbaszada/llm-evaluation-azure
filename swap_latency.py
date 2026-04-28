"""
swap_latency.py — Swap mean latency between paired models for a "what-if"
visualization requested by the supervisor.

Pairs swapped:
    gpt-4.1   <->  gpt-4.1-mini
    gpt-5.4   <->  gpt-5.4-mini

What it does
------------
1. Standalone results (`results/<latest>/<dataset>.jsonl`):
   - For each (dataset, pair) joins records by `task_id` and SWAPS `elapsed_s`
     between the two paired models record-by-record. Records present in only
     one of the two folders are left untouched (latency cannot be swapped).
   - Recomputes `economic_reward = -(cost_usd + 0.01*elapsed_s
                                     + 1.0*(1 - is_correct))`.

2. Optimization results (`optimization_results/*/<dataset>.jsonl`) — cascade,
   router, and self-consistency:
   - For each affected record, computes a per-(dataset, model) latency
     mean-shift using the ORIGINAL standalone means and applies it to
     `elapsed_s` whenever `chosen_model` (or the configuration's underlying
     model for self-consistency) is one of the four affected models.
   - Cascade: shift uses the chosen model (small if not escalated, large if
     escalated). Router: shift uses the chosen model. Self-consistency:
     shift uses the configuration's base model.
   - elapsed_s is clipped to >= 0.05 to avoid negatives.
   - Recomputes `economic_reward`.

3. Writes a complete revert manifest to `_latency_swap_revert.json` containing
   the original `elapsed_s` and `economic_reward` for every record changed,
   keyed by file path and 0-based line index. Re-running this script with
   `--revert` restores everything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict


REPO_ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
OPT_RESULTS_DIR = REPO_ROOT / "optimization_results"
MANIFEST_PATH = REPO_ROOT / "_latency_swap_revert.json"

DATASETS = ["humaneval", "mbpp", "mmlu_pro", "gpqa", "gsm8k"]

PAIRS = [
    ("gpt-4.1", "gpt-4.1-mini"),
    ("gpt-5.4", "gpt-5.4-mini"),
]
AFFECTED_MODELS = {m for pair in PAIRS for m in pair}

LAMBDA_LATENCY = 0.01
LAMBDA_ERROR = 1.0
MIN_LATENCY = 0.05  # clip floor for shifted latencies


def reward(cost: float, elapsed: float, is_correct: int) -> float:
    return -(cost + LAMBDA_LATENCY * elapsed + LAMBDA_ERROR * (1 - int(is_correct)))


# ---------------------------------------------------------------------------
# Folder discovery (mirror of visualize.py)
# ---------------------------------------------------------------------------
def latest_standalone_folders() -> dict[str, pathlib.Path]:
    out: dict[str, pathlib.Path] = {}
    for folder in RESULTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        if len(name) > 16 and name[8] == "_" and name[15] == "_":
            model = name[16:]
            prev = out.get(model)
            if prev is None or folder.name > prev.name:
                out[model] = folder
    return out


def read_lines(path: pathlib.Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_lines(path: pathlib.Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)


def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(REPO_ROOT)).replace("\\", "/")


# ---------------------------------------------------------------------------
# Standalone swap
# ---------------------------------------------------------------------------
def swap_standalone(folders: dict[str, pathlib.Path], manifest: dict) -> None:
    for model_a, model_b in PAIRS:
        if model_a not in folders or model_b not in folders:
            print(f"  ! missing folder for {model_a} or {model_b}; skipping pair")
            continue
        for ds in DATASETS:
            path_a = folders[model_a] / f"{ds}.jsonl"
            path_b = folders[model_b] / f"{ds}.jsonl"
            if not path_a.exists() or not path_b.exists():
                continue
            _swap_pair_file(path_a, path_b, manifest)


def _swap_pair_file(path_a: pathlib.Path, path_b: pathlib.Path, manifest: dict) -> None:
    lines_a = read_lines(path_a)
    lines_b = read_lines(path_b)

    # Parse with original line index preserved
    def parse(lines: list[str]) -> tuple[list[dict | None], dict[str, int]]:
        recs: list[dict | None] = []
        idx_by_id: dict[str, int] = {}
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                recs.append(None)
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                recs.append(None)
                continue
            recs.append(rec)
            tid = rec.get("task_id")
            if tid is not None and rec.get("type") != "dataset_summary":
                idx_by_id[str(tid)] = i
        return recs, idx_by_id

    recs_a, idx_a = parse(lines_a)
    recs_b, idx_b = parse(lines_b)

    common = sorted(set(idx_a) & set(idx_b))
    if not common:
        return

    rel_a = rel(path_a)
    rel_b = rel(path_b)
    manifest.setdefault(rel_a, {})
    manifest.setdefault(rel_b, {})

    for tid in common:
        ia = idx_a[tid]
        ib = idx_b[tid]
        ra = recs_a[ia]
        rb = recs_b[ib]
        if ra is None or rb is None:
            continue
        ea = ra.get("elapsed_s")
        eb = rb.get("elapsed_s")
        if ea is None or eb is None:
            continue

        # Save originals (only the very first time we touch a record)
        manifest[rel_a].setdefault(str(ia), {
            "elapsed_s": ea,
            "economic_reward": ra.get("economic_reward"),
        })
        manifest[rel_b].setdefault(str(ib), {
            "elapsed_s": eb,
            "economic_reward": rb.get("economic_reward"),
        })

        # Swap latency
        ra["elapsed_s"], rb["elapsed_s"] = eb, ea

        # Recompute reward
        if "cost_usd" in ra and "is_correct" in ra:
            ra["economic_reward"] = reward(ra["cost_usd"], ra["elapsed_s"], ra["is_correct"])
        if "cost_usd" in rb and "is_correct" in rb:
            rb["economic_reward"] = reward(rb["cost_usd"], rb["elapsed_s"], rb["is_correct"])

        lines_a[ia] = json.dumps(ra) + "\n"
        lines_b[ib] = json.dumps(rb) + "\n"

    write_lines(path_a, lines_a)
    write_lines(path_b, lines_b)


# ---------------------------------------------------------------------------
# Optimization shift
# ---------------------------------------------------------------------------
def compute_original_means(folders: dict[str, pathlib.Path], manifest: dict) -> dict:
    """Per-(model, dataset) ORIGINAL mean elapsed_s.

    Reads the standalone files BUT reconstructs original values using the
    manifest (since by this point we may have already swapped). This way we
    can be called either before or after the swap and get the same answer.
    """
    means: dict = {}
    for model, folder in folders.items():
        if model not in AFFECTED_MODELS:
            continue
        means[model] = {}
        for ds in DATASETS:
            path = folder / f"{ds}.jsonl"
            if not path.exists():
                continue
            rel_p = rel(path)
            saved = manifest.get(rel_p, {})
            total = 0.0
            n = 0
            for i, line in enumerate(read_lines(path)):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "dataset_summary":
                    continue
                # Prefer manifest's original value if recorded
                if str(i) in saved and saved[str(i)].get("elapsed_s") is not None:
                    e = saved[str(i)]["elapsed_s"]
                else:
                    e = rec.get("elapsed_s")
                if e is None:
                    continue
                total += float(e)
                n += 1
            if n > 0:
                means[model][ds] = total / n
    return means


def configure_optimization_shifts(orig_means: dict) -> dict:
    """For each affected model and dataset, the per-record shift to apply.

    shift[model][dataset] = mean_partner - mean_self
    """
    partner = {}
    for a, b in PAIRS:
        partner[a] = b
        partner[b] = a
    shifts: dict = {}
    for model, by_ds in orig_means.items():
        p = partner.get(model)
        if p is None or p not in orig_means:
            continue
        shifts[model] = {}
        for ds, mean_self in by_ds.items():
            mean_p = orig_means[p].get(ds)
            if mean_p is None:
                continue
            shifts[model][ds] = mean_p - mean_self
    return shifts


def shift_optimization(shifts: dict, manifest: dict) -> None:
    for cfg_dir in OPT_RESULTS_DIR.iterdir():
        if not cfg_dir.is_dir():
            continue
        for ds in DATASETS:
            path = cfg_dir / f"{ds}.jsonl"
            if not path.exists():
                continue
            _shift_opt_file(path, ds, shifts, manifest)


def _opt_record_target_model(rec: dict, cfg_name: str) -> str | None:
    """Decide which model's shift to apply to this record."""
    cm = rec.get("chosen_model")
    if cm:
        return cm if cm in AFFECTED_MODELS else None
    # self-consistency: configuration name has the model
    # selfcons__<model>__N3
    if cfg_name.startswith("selfcons__"):
        try:
            mtoken = cfg_name.split("__")[1]
        except IndexError:
            return None
        # mtoken e.g. "gpt41mini" or "gpt54mini"
        canonical = {
            "gpt41": "gpt-4.1",
            "gpt41mini": "gpt-4.1-mini",
            "gpt54": "gpt-5.4",
            "gpt54mini": "gpt-5.4-mini",
        }.get(mtoken)
        return canonical if canonical in AFFECTED_MODELS else None
    return None


def _shift_opt_file(path: pathlib.Path, ds: str, shifts: dict, manifest: dict) -> None:
    lines = read_lines(path)
    rel_p = rel(path)
    manifest.setdefault(rel_p, {})
    cfg_name = path.parent.name
    changed = False

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "dataset_summary":
            continue

        target = _opt_record_target_model(rec, cfg_name)
        if target is None:
            continue
        delta = shifts.get(target, {}).get(ds)
        if delta is None:
            continue

        e_orig = rec.get("elapsed_s")
        if e_orig is None:
            continue

        # Save originals once
        manifest[rel_p].setdefault(str(i), {
            "elapsed_s": e_orig,
            "economic_reward": rec.get("economic_reward"),
        })

        new_e = max(MIN_LATENCY, float(e_orig) + float(delta))
        rec["elapsed_s"] = new_e
        if "cost_usd" in rec and "is_correct" in rec:
            rec["economic_reward"] = reward(rec["cost_usd"], new_e, rec["is_correct"])

        lines[i] = json.dumps(rec) + "\n"
        changed = True

    if changed:
        write_lines(path, lines)


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------
def revert(manifest: dict) -> None:
    for rel_p, by_idx in manifest.items():
        path = REPO_ROOT / rel_p
        if not path.exists():
            print(f"  ! missing file {rel_p}")
            continue
        lines = read_lines(path)
        for idx_str, orig in by_idx.items():
            i = int(idx_str)
            if i >= len(lines):
                continue
            s = lines[i].strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "elapsed_s" in orig and orig["elapsed_s"] is not None:
                rec["elapsed_s"] = orig["elapsed_s"]
            if "economic_reward" in orig:
                rec["economic_reward"] = orig["economic_reward"]
            lines[i] = json.dumps(rec) + "\n"
        write_lines(path, lines)


# ---------------------------------------------------------------------------
# Stats helper (post-swap means for the user)
# ---------------------------------------------------------------------------
def print_summary(folders: dict[str, pathlib.Path]) -> None:
    print()
    print("Post-swap mean elapsed_s per (model, dataset):")
    print(f"{'model':<14} {'macro':>8} " + " ".join(f"{d:>10}" for d in DATASETS))
    for model in ["gpt-4.1", "gpt-4.1-mini", "gpt-5.4", "gpt-5.4-mini",
                  "gpt-5.4-pro", "o3-mini"]:
        if model not in folders:
            continue
        per_ds: list[float] = []
        for ds in DATASETS:
            path = folders[model] / f"{ds}.jsonl"
            if not path.exists():
                per_ds.append(float("nan"))
                continue
            tot = 0.0
            n = 0
            for line in read_lines(path):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "dataset_summary":
                    continue
                e = rec.get("elapsed_s")
                if e is None:
                    continue
                tot += float(e)
                n += 1
            per_ds.append(tot / n if n else float("nan"))
        macro = sum(x for x in per_ds if x == x) / sum(1 for x in per_ds if x == x) if per_ds else float("nan")
        print(f"{model:<14} {macro:>8.2f} " + " ".join(f"{x:>10.2f}" for x in per_ds))


def print_reward_summary(folders: dict[str, pathlib.Path]) -> None:
    print()
    print("Post-swap macro-avg reward (default lambdas):")
    for model in ["gpt-4.1", "gpt-4.1-mini", "gpt-5.4", "gpt-5.4-mini",
                  "gpt-5.4-pro", "o3-mini"]:
        if model not in folders:
            continue
        per_ds: list[float] = []
        for ds in DATASETS:
            path = folders[model] / f"{ds}.jsonl"
            if not path.exists():
                continue
            tot = 0.0
            n = 0
            for line in read_lines(path):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "dataset_summary":
                    continue
                r = rec.get("economic_reward")
                if r is None:
                    continue
                tot += float(r)
                n += 1
            if n:
                per_ds.append(tot / n)
        macro = sum(per_ds) / len(per_ds) if per_ds else float("nan")
        print(f"  {model:<14} {macro:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true",
                    help="Restore original values from manifest")
    ap.add_argument("--standalone-only", action="store_true",
                    help="Only swap standalone results; skip optimization shift")
    args = ap.parse_args()

    folders = latest_standalone_folders()
    print("Standalone folders:")
    for m, f in sorted(folders.items()):
        print(f"  {m:<14} -> {f.name}")

    manifest = load_manifest()

    if args.revert:
        if not manifest:
            print("Manifest is empty; nothing to revert.")
            return 0
        print(f"Reverting {sum(len(v) for v in manifest.values())} records...")
        revert(manifest)
        MANIFEST_PATH.unlink()
        print("Reverted. Manifest deleted.")
        return 0

    if manifest:
        print("ERROR: an existing manifest is present. Run with --revert first.")
        return 1

    print("\nComputing original per-(model, dataset) latency means...")
    orig_means = compute_original_means(folders, manifest)

    print("\nSwapping standalone elapsed_s by task_id within each pair...")
    swap_standalone(folders, manifest)

    if not args.standalone_only:
        print("\nApplying mean-shift to optimization records...")
        shifts = configure_optimization_shifts(orig_means)
        for m, by_ds in shifts.items():
            print(f"  {m}: " + ", ".join(f"{ds}={v:+.2f}" for ds, v in by_ds.items()))
        shift_optimization(shifts, manifest)

    save_manifest(manifest)
    n_changed = sum(len(v) for v in manifest.values())
    print(f"\nDone. {n_changed} record entries written to {MANIFEST_PATH.name}")

    print_summary(folders)
    print_reward_summary(folders)
    return 0


if __name__ == "__main__":
    sys.exit(main())
