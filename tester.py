import os
import sys
import json
import re
import multiprocessing
import queue as _queue
from pathlib import Path

EXEC_TIMEOUT_S = 10  # seconds before code execution is considered hung


# ---------------------------------------------------------------------------
# Subprocess-based code executor  (can actually be killed on timeout)
# ---------------------------------------------------------------------------

def _worker_main(recv_q: multiprocessing.Queue, send_q: multiprocessing.Queue) -> None:
    """Runs in a child process. Executes code-piece lists sent via recv_q."""
    while True:
        try:
            job = recv_q.get()
        except Exception:
            break
        if job is None:          # sentinel – clean shutdown
            break
        pieces: list = job
        ns: dict = {}
        try:
            for piece in pieces:
                exec(piece, ns)  # noqa: S102
            send_q.put(("ok",))
        except BaseException as exc:
            send_q.put(("err", type(exc).__name__, str(exc)[:500]))


class _CodeExecutor:
    """Persistent subprocess worker. Killed and restarted automatically on timeout."""

    def __init__(self, timeout: float = EXEC_TIMEOUT_S) -> None:
        self.timeout = timeout
        self._ctx = multiprocessing.get_context("spawn")
        self._proc: "multiprocessing.Process | None" = None
        self._send: "multiprocessing.Queue | None" = None
        self._recv: "multiprocessing.Queue | None" = None
        self._start()

    def _start(self) -> None:
        self._send = self._ctx.Queue()
        self._recv = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_main, args=(self._send, self._recv), daemon=True
        )
        self._proc.start()

    def run(self, code_pieces: list) -> None:
        """Execute pieces in sequence in the worker. Raises on error or timeout."""
        if self._proc is None or not self._proc.is_alive():
            self._start()
        self._send.put(code_pieces)
        try:
            result = self._recv.get(timeout=self.timeout)
        except _queue.Empty:
            # Worker is hung – kill it and restart for next call
            self._proc.kill()
            self._proc.join(2)
            self._proc = None
            raise TimeoutError(f"Code execution timed out after {self.timeout}s (infinite loop?)")
        if result[0] == "err":
            _, exc_type, exc_msg = result
            exc_class = {
                "AssertionError": AssertionError, "NameError": NameError,
                "TypeError": TypeError, "ValueError": ValueError,
                "RecursionError": RecursionError, "AttributeError": AttributeError,
                "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
            }.get(exc_type, RuntimeError)
            raise exc_class(exc_msg)

    def shutdown(self) -> None:
        if self._proc and self._proc.is_alive():
            try:
                self._send.put(None)
                self._proc.join(2)
            except Exception:
                self._proc.kill()


_EXECUTOR: "_CodeExecutor | None" = None


def _get_executor() -> _CodeExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = _CodeExecutor()
    return _EXECUTOR

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
        return True, "No test code available — skipped", None, ""

    # Extract the entry-point function name from the code (first `def` line)
    fn_match = re.search(r"def (\w+)\s*\(", code)
    if not fn_match:
        return False, "Could not find function definition in solution", None, ""
    fn_name = fn_match.group(1)

    executor = _get_executor()
    ai_passed, ai_error = True, ""
    try:
        # exec code + test harness + call check(fn) — all in one subprocess namespace
        executor.run([code, he_test, f"check({fn_name})"])
    except Exception as exc:
        ai_passed, ai_error = False, _fmt_exc(exc)

    ref_passed, ref_error = True, ""
    if not ai_passed:
        he_prompt = record.get("question", "")
        correct   = record.get("correct_answer", "")
        try:
            executor.run([he_prompt + correct, he_test, f"check({fn_name})"])
        except Exception as exc:
            ref_passed, ref_error = False, _fmt_exc(exc)

    return ai_passed, ai_error, ref_passed if not ai_passed else None, ref_error


def evaluate_mbpp(record: dict) -> tuple[bool, str, "bool | None", str]:
    """Returns (ai_passed, ai_error, ref_passed, ref_error)."""
    code      = record.get("model_response", "")
    test_list = record.get("test_list", [])
    if not test_list:
        return True, "No test_list available — skipped", None, ""

    executor = _get_executor()
    ai_passed, ai_error = True, ""
    try:
        executor.run([code] + test_list)
    except Exception as exc:
        ai_passed, ai_error = False, _fmt_exc(exc)

    ref_passed, ref_error = True, ""
    if not ai_passed:
        correct = record.get("correct_answer", "")
        try:
            executor.run([correct] + test_list)
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

    # GSM8K format: chain-of-thought ending with "#### <number>"
    m = re.search(r"####\s*([\d,\.]+)", correct_raw)
    expected = m.group(1).replace(",", "") if m else re.sub(r"[^\d.]", "", correct_raw.strip())

    nums = re.findall(r"[\d,\.]+", model_resp.replace(",", ""))
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
            print(f"  [DEBUG] {name}: no records found, skipping", flush=True)
            results_for_ds.append(("skipped", "", None, None, ""))
        else:
            print(f"  [DEBUG] {name}: evaluating {len(records)} records ...", flush=True)
            for idx, record in enumerate(records):
                if idx % 50 == 0 and idx > 0:
                    print(f"  [DEBUG] {name}: {idx}/{len(records)} done", flush=True)
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
            print(f"  [DEBUG] {name}: done, writing results ...", flush=True)
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
    p(f"  {WHITE}  Avg reward:  {overall_reward_str}  (lam_latency={LAMBDA_LATENCY}, lam_error={LAMBDA_ERROR}){RESET}")
    psep()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for Windows spawn
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

    # Also evaluate all optimization results folders
    opt_dir = Path("optimization_results")
    if opt_dir.exists():
        opt_folders = sorted(d for d in opt_dir.iterdir() if d.is_dir())
        if opt_folders:
            print(f"\n{BOLD}{WHITE}Evaluating {len(opt_folders)} optimization result folder(s)...{RESET}")
            for folder in opt_folders:
                run_all_tests(folder, show_details=False)
