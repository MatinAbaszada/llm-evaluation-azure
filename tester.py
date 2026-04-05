import os
import sys
import json
import re
import threading
from pathlib import Path

# ANSI colors
RED   = "\033[91m"
GREEN = "\033[92m"
WHITE = "\033[97m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Economic reward parameters
# ---------------------------------------------------------------------------
LAMBDA_LATENCY = 0.01   # penalty weight per second of latency
LAMBDA_ERROR   = 1.0    # penalty weight per wrong answer


def sep(char="=", width=80):
    print(f"{WHITE}{char * width}{RESET}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_folder() -> Path:
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("No 'results' directory found. Run a test first.")
    folders = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not folders:
        raise FileNotFoundError("No result folders found inside 'results/'.")
    return folders[-1]


def find_latest_folder_for_model(model: str) -> "Path | None":
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    suffix = f"_{model}"
    folders = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.endswith(suffix)])
    return folders[-1] if folders else None


def find_all_model_folders() -> dict:
    """Return {model_name: latest_folder} for every model that has results."""
    results_dir = Path("results")
    if not results_dir.exists():
        return {}
    model_latest: dict = {}
    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        # folder format: YYYYMMDD_HHMMSS_<deployment>  (timestamp = 15 chars + underscore)
        if len(name) > 16 and name[8] == "_" and name[15] == "_":
            model = name[16:]
            prev = model_latest.get(model)
            if prev is None or folder.name > prev.name:
                model_latest[model] = folder
    return model_latest


def _load_all(folder: Path, stem: str) -> list:
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


# ---------------------------------------------------------------------------
# Per-dataset evaluators  —  each returns (passed: bool, detail: str)
# ---------------------------------------------------------------------------

def _fmt_exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def evaluate_humaneval(record: dict) -> tuple[bool, str, "bool | None", str]:
    """Returns (ai_passed, ai_error, ref_passed, ref_error)."""
    code    = record.get("solution") or record.get("model_response", "")
    he_test = record.get("he_test", "")
    if not he_test:
        return True, "No test code available \u2014 skipped", None, ""
    namespace: dict = {}
    ai_passed, ai_error = True, ""
    try:
        exec(code,    namespace)
        exec(he_test, namespace)
    except Exception as exc:
        ai_passed, ai_error = False, _fmt_exc(exc)

    ref_passed, ref_error = True, ""
    if not ai_passed:
        he_prompt = record.get("question", "")
        correct   = record.get("correct_answer", "")
        ref_ns: dict = {}
        try:
            exec(he_prompt + correct, ref_ns)
            exec(he_test, ref_ns)
        except Exception as exc:
            ref_passed, ref_error = False, _fmt_exc(exc)

    return ai_passed, ai_error, ref_passed if not ai_passed else None, ref_error


def evaluate_mbpp(record: dict) -> tuple[bool, str, "bool | None", str]:
    """Returns (ai_passed, ai_error, ref_passed, ref_error)."""
    code      = record.get("model_response", "")
    test_list = record.get("test_list", [])
    if not test_list:
        return True, "No test_list available \u2014 skipped", None, ""
    namespace: dict = {}
    ai_passed, ai_error = True, ""
    try:
        exec(code, namespace)
        for assertion in test_list:
            exec(assertion, namespace)
    except Exception as exc:
        ai_passed, ai_error = False, _fmt_exc(exc)

    ref_passed, ref_error = True, ""
    if not ai_passed:
        correct = record.get("correct_answer", "")
        ref_ns: dict = {}
        try:
            exec(correct, ref_ns)
            for assertion in test_list:
                exec(assertion, ref_ns)
        except Exception as exc:
            ref_passed, ref_error = False, _fmt_exc(exc)

    return ai_passed, ai_error, ref_passed if not ai_passed else None, ref_error


def evaluate_mmlu_pro(record: dict) -> tuple[bool, str, None, str]:
    """Compare first letter of model response against correct letter."""
    raw     = record.get("model_response", "").strip()
    correct = record.get("correct_answer", "").strip().upper()
    letter  = ""
    for ch in raw:
        if ch.isalpha():
            letter = ch.upper()
            break
    passed = letter == correct
    return passed, f"Got '{letter}', expected '{correct}'", None, ""


def evaluate_gpqa(record: dict) -> tuple[bool, str, None, str]:
    """Case-insensitive exact match of model response against correct answer text."""
    ai  = record.get("model_response", "").strip()
    ans = record.get("correct_answer", "").strip()
    passed = ai.lower() == ans.lower()
    return passed, f"Got '{ai}', expected '{ans}'", None, ""


def evaluate_gsm8k(record: dict) -> tuple[bool, str, None, str]:
    """Extract final number from both model response and correct answer, compare."""
    model_resp  = record.get("model_response", "")
    correct_raw = record.get("correct_answer", "")

    m = re.search(r"####\s*([\d,]+)", correct_raw)
    expected = m.group(1).replace(",", "") if m else re.sub(r"[^\d.]", "", correct_raw.strip())

    nums = re.findall(r"\d[\d,.]*", model_resp.replace(",", ""))
    got  = nums[-1].replace(",", "") if nums else ""

    passed = got == expected
    return passed, f"Got '{got}', expected '{expected}'", None, ""


# ---------------------------------------------------------------------------
# Evaluator registry  —  (file_stem, display_name, evaluator_fn)
# ---------------------------------------------------------------------------

EVALUATORS = [
    ("humaneval", "HumanEval", evaluate_humaneval),
    ("mbpp",      "MBPP",      evaluate_mbpp),
    ("mmlu_pro",  "MMLU-Pro",  evaluate_mmlu_pro),
    ("gpqa",      "GPQA",      evaluate_gpqa),
    ("gsm8k",     "GSM8K",     evaluate_gsm8k),
]


# ---------------------------------------------------------------------------
# Economic reward
# ---------------------------------------------------------------------------

def compute_economic_reward(cost_usd: float, elapsed_s: float, is_correct: int) -> float:
    """economic_reward = -(cost_usd + λ_latency·elapsed_s + λ_error·(1−is_correct))"""
    return -(cost_usd + LAMBDA_LATENCY * elapsed_s + LAMBDA_ERROR * (1 - is_correct))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all_tests(folder: "Path | None" = None, show_details: bool = False) -> None:
    if folder is None:
        folder = find_latest_folder()

    # Derive model name from folder: YYYYMMDD_HHMMSS_<model>
    name_parts = folder.name.split("_", 2)
    model_name = name_parts[2] if len(name_parts) == 3 else folder.name

    def p(line: str = ""):
        print(line)

    def psep(char="=", width=80):
        print(f"{WHITE}{char * width}{RESET}")

    p(f"\n{BOLD}{WHITE}Model: {model_name}  |  Results from: {folder}{RESET}")
    psep()

    eval_results: dict = {}
    total_cost    = 0.0
    elapsed_times: list = []

    for stem, name, fn in EVALUATORS:
        records = _load_all(folder, stem)
        results_for_ds = []
        if not records:
            results_for_ds.append(("skipped", "", None, None, ""))
        else:
            for record in records:
                passed, detail, ref_passed, ref_error = fn(record)
                record["is_correct"] = 1 if passed else 0
                _cost    = record.get("cost_usd", 0.0) or 0.0
                _latency = record.get("elapsed_s", 0.0) or 0.0
                record["economic_reward"] = compute_economic_reward(_cost, _latency, record["is_correct"])
                results_for_ds.append(("pass" if passed else "fail", detail, record, ref_passed, ref_error))
                if record.get("cost_usd") is not None:
                    total_cost += record["cost_usd"]
                if record.get("elapsed_s") is not None:
                    elapsed_times.append(record["elapsed_s"])
            # Write updated records (with is_correct + economic_reward) back to the JSONL file,
            # then append a single summary line with the average economic reward for this dataset.
            ds_rewards_pre = [r["economic_reward"] for r in records]
            avg_reward = sum(ds_rewards_pre) / len(ds_rewards_pre)
            jsonl_path = folder / f"{stem}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as _f:
                for _rec in records:
                    _f.write(json.dumps(_rec) + "\n")
                _f.write(json.dumps({"type": "dataset_summary", "avg_economic_reward": avg_reward}) + "\n")
        eval_results[name] = results_for_ds

    total_passed  = 0
    total_run     = 0
    skipped_count = 0
    total_reward       = 0.0
    total_reward_count = 0

    for _, display_name, _ in EVALUATORS:
        entries = eval_results.get(display_name, [])

        if len(entries) == 1 and entries[0][0] == "skipped":
            skipped_count += 1
            p(f"  {WHITE}[{model_name}] [{display_name:12}]  SKIPPED (no result file){RESET}")
            continue

        ds_passed = sum(1 for s, *_ in entries if s == "pass")
        ds_total  = len(entries)
        total_passed += ds_passed
        total_run    += ds_total

        ds_rewards = [r.get("economic_reward") for _, __, r, ___, ____ in entries if r and r.get("economic_reward") is not None]
        ds_avg_reward = sum(ds_rewards) / len(ds_rewards) if ds_rewards else None
        total_reward       += sum(ds_rewards)
        total_reward_count += len(ds_rewards)

        color = GREEN if ds_passed == ds_total else RED
        p(f"  {color}{BOLD}[{model_name}] [{display_name:12}]  {ds_passed}/{ds_total} passed{RESET}")

        if show_details:
            for idx, (status, detail, record, ref_passed, ref_error) in enumerate(entries):
                if status == "fail":
                    psep("-")
                    label = f"#{idx + 1}" if ds_total > 1 else ""
                    p(f"  {RED}  Sample {label} WRONG ANSWER{RESET}")
                    p(f"  {WHITE}  Question   :{RESET}")
                    p(f"  {WHITE}  {record.get('question', '').strip()[:500]}{RESET}")
                    p(f"  {WHITE}  AI Answer  :{RESET}")
                    p(f"  {RED}  {record['model_response'].strip()[:300]}{RESET}")
                    p(f"  {WHITE}  Correct    :{RESET}")
                    p(f"  {GREEN}  {record['correct_answer'].strip()[:300]}{RESET}")
                    if detail:
                        p(f"  {RED}  Error      : {detail[:300]}{RESET}")
                    if ref_passed is not None:
                        if ref_passed:
                            p(f"  {GREEN}  Ref check  : correct answer PASSES its own tests{RESET}")
                        else:
                            ref_note = f" ({ref_error[:200]})" if ref_error else ""
                            p(f"  {RED}  Ref check  : correct answer FAILS its own tests{ref_note}{RESET}")
                    p()

    psep()
    summary = f"{total_passed}/{total_run} passed"
    if skipped_count:
        summary += f"  ({skipped_count} dataset(s) skipped)"

    if total_run > 0 and total_passed == total_run:
        p(f"{BOLD}{GREEN}  [{model_name}]  All tests passed!  {summary}{RESET}")
    else:
        p(f"{BOLD}{CYAN}  [{model_name}]  Results: {summary}{RESET}")

    cost_str = f"${total_cost:.6f}" if total_cost > 0 else "n/a"
    if elapsed_times:
        avg_s = sum(elapsed_times) / len(elapsed_times)
        time_str = f"{avg_s:.3f}s avg  ({len(elapsed_times)} tasks,  total {sum(elapsed_times):.1f}s)"
    else:
        time_str = "n/a"
    overall_reward_str = f"{total_reward / total_reward_count:.6f}" if total_reward_count > 0 else "n/a"
    p(f"  {WHITE}  Cost:        {cost_str}{RESET}")
    p(f"  {WHITE}  Time:        {time_str}{RESET}")
    p(f"  {WHITE}  Avg reward:  {overall_reward_str}  (λ_latency={LAMBDA_LATENCY}, λ_error={LAMBDA_ERROR}){RESET}")
    psep()


if __name__ == "__main__":
    show_details = "--details" in sys.argv
    model_folders = find_all_model_folders()
    if not model_folders:
        print(f"{WHITE}No result folders found. Run tests first.{RESET}")
    else:
        print(f"\n{BOLD}{WHITE}Found results for {len(model_folders)} model(s): {', '.join(model_folders)}{RESET}")
        if show_details:
            print(f"{WHITE}  Showing wrong answer details (--details){RESET}")
        for folder in model_folders.values():
            run_all_tests(folder, show_details=show_details)
