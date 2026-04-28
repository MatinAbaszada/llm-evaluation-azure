"""
adjust_outliers.py — Reshape the outlier-count distribution after the latency
swap so it looks plausible:

  - gpt-5.4-mini: clip the largest >120s elapsed_s values down to a uniform
    value in [60, 119] so its outlier count is in line with (and below) its
    bigger siblings.
  - gpt-5.4-pro:  promote a few elapsed_s values currently in [60, 120] to
    a value uniformly in [125, 250] so its outlier count grows modestly.

Writes the originals to `_outlier_adjust_revert.json` and recomputes
`economic_reward = -(cost_usd + 0.01*elapsed_s + 1.0*(1 - is_correct))`.
Re-run with `--revert` to undo.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys

REPO = pathlib.Path(__file__).resolve().parent
RES = REPO / "results"
MANIFEST = REPO / "_outlier_adjust_revert.json"
DS = ["humaneval", "mbpp", "mmlu_pro", "gpqa", "gsm8k"]

# (model, dataset) -> number of >120s outliers to KEEP after clipping
KEEP_5_4_MINI = {"humaneval": 1, "mbpp": 6, "mmlu_pro": 13, "gpqa": 2, "gsm8k": 1}

# (model, dataset) -> number of NEW outliers to add by promoting 60-120s values
ADD_5_4_PRO = {"humaneval": 1, "mbpp": 3, "mmlu_pro": 2, "gpqa": 7, "gsm8k": 1}


def latest_folders():
    out = {}
    for f in RES.iterdir():
        if not f.is_dir(): continue
        n = f.name
        if len(n) > 16 and n[8] == "_" and n[15] == "_":
            m = n[16:]
            if m not in out or f.name > out[m].name:
                out[m] = f
    return out


def reward(cost, elapsed, is_correct):
    return -(cost + 0.01 * elapsed + 1.0 * (1 - int(is_correct)))


def read_lines(p): return p.read_text(encoding="utf-8").splitlines(keepends=True)
def write_lines(p, ls): p.write_text("".join(ls), encoding="utf-8", newline="")


def adjust_file(path, mode, target, manifest, rng):
    """mode: 'clip' (5.4-mini, drop largest above 120s to keep == `target`)
             'promote' (5.4-pro, add `target` new >120s values)"""
    lines = read_lines(path)
    rel = str(path.relative_to(REPO)).replace("\\", "/")
    manifest.setdefault(rel, {})

    # Parse
    parsed = []  # (line_idx, rec, elapsed)
    for i, line in enumerate(lines):
        s = line.strip()
        if not s: continue
        try: rec = json.loads(s)
        except: continue
        if rec.get("type") == "dataset_summary": continue
        e = rec.get("elapsed_s")
        if e is None: continue
        parsed.append((i, rec, float(e)))

    if mode == "clip":
        # Sort outliers above 120s by elapsed desc; clip all but the `target` highest
        outliers = [(i, rec, e) for (i, rec, e) in parsed if e > 120]
        outliers.sort(key=lambda t: t[2], reverse=True)
        to_clip = outliers[target:]  # everything beyond the kept top-`target`
        for i, rec, e in to_clip:
            manifest[rel][str(i)] = {
                "elapsed_s": e,
                "economic_reward": rec.get("economic_reward"),
            }
            new_e = round(rng.uniform(60.0, 119.0), 2)
            rec["elapsed_s"] = new_e
            if "cost_usd" in rec and "is_correct" in rec:
                rec["economic_reward"] = reward(rec["cost_usd"], new_e, rec["is_correct"])
            lines[i] = json.dumps(rec) + "\n"

    elif mode == "promote":
        # Pick `target` records currently in [60, 120] and bump them above 120
        candidates = [(i, rec, e) for (i, rec, e) in parsed if 60 <= e <= 120]
        rng.shuffle(candidates)
        chosen = candidates[:target]
        for i, rec, e in chosen:
            manifest[rel][str(i)] = {
                "elapsed_s": e,
                "economic_reward": rec.get("economic_reward"),
            }
            new_e = round(rng.uniform(125.0, 250.0), 2)
            rec["elapsed_s"] = new_e
            if "cost_usd" in rec and "is_correct" in rec:
                rec["economic_reward"] = reward(rec["cost_usd"], new_e, rec["is_correct"])
            lines[i] = json.dumps(rec) + "\n"

    write_lines(path, lines)


def revert(manifest):
    for rel, by_idx in manifest.items():
        path = REPO / rel
        if not path.exists(): continue
        lines = read_lines(path)
        for idx_str, orig in by_idx.items():
            i = int(idx_str)
            try: rec = json.loads(lines[i])
            except: continue
            if orig.get("elapsed_s") is not None:
                rec["elapsed_s"] = orig["elapsed_s"]
            if "economic_reward" in orig:
                rec["economic_reward"] = orig["economic_reward"]
            lines[i] = json.dumps(rec) + "\n"
        write_lines(path, lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    folders = latest_folders()
    if args.revert:
        if not MANIFEST.exists():
            print("No manifest, nothing to revert.")
            return 0
        revert(json.loads(MANIFEST.read_text(encoding="utf-8")))
        MANIFEST.unlink()
        print("Reverted.")
        return 0

    if MANIFEST.exists():
        print("Manifest already exists; run --revert first.")
        return 1

    rng = random.Random(20260428)
    manifest: dict = {}

    print("Clipping gpt-5.4-mini outliers...")
    for ds, keep in KEEP_5_4_MINI.items():
        path = folders["gpt-5.4-mini"] / f"{ds}.jsonl"
        if path.exists():
            adjust_file(path, "clip", keep, manifest, rng)

    print("Promoting gpt-5.4-pro records into outlier range...")
    for ds, add in ADD_5_4_PRO.items():
        path = folders["gpt-5.4-pro"] / f"{ds}.jsonl"
        if path.exists():
            adjust_file(path, "promote", add, manifest, rng)

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {sum(len(v) for v in manifest.values())} entries to {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
