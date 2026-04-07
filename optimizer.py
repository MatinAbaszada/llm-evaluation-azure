"""
optimizer.py — LLM Optimization Methods
========================================
Three optimization strategies, each reading from existing result folders and/or
making minimal new API calls, then writing output to optimization_results/.

Methods
-------
1. LLM Router  — a cheap model classifies each question as "small" or "large",
                 then we pull the answer from the matching pre-existing result folder.
   Configs (4):
     router__rtr-gpt41mini__small-gpt41mini__large-gpt41
     router__rtr-gpt41mini__small-gpt41mini__large-gpt54
     router__rtr-gpt54mini__small-gpt41mini__large-gpt41
     router__rtr-gpt54mini__small-gpt54mini__large-gpt54

2. Cascade — Per-question: if confidence >= threshold → keep small answer,
                           else → use large model's stored answer (cost = small + large).
   Configs (12):
     small in {gpt-4.1-mini, gpt-5.4-mini}  ×  large in {gpt-4.1, gpt-5.4}  ×  T in {60, 75, 90}

3. Self-consistency — run small model N=3 times per question with temperature>0,
                      majority-vote the answer, store combined cost & max latency.
   Configs (2):
     selfcons__gpt41mini__N3
     selfcons__gpt54mini__N3

Output folder: optimization_results/<config_name>/
Each output folder contains one JSONL per dataset, and tester.py / visualize.py work on these folders without changes.

Usage
-----
    python optimizer.py                    # interactive menu
    python optimizer.py --router           # run all router configs
    python optimizer.py --cascade          # run all cascade configs
    python optimizer.py --selfcons         # run all self-consistency configs
    python optimizer.py --all              # run everything
"""

from __future__ import annotations

import os
import sys
import json
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI, OpenAI

# ---------------------------------------------------------------------------
# Shared constants from humaneval_explore.py (kept in sync manually)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1":      (2.00,   8.00),
    "gpt-4.1-mini": (0.40,   1.60),
    "o3-mini":      (1.10,   4.40),
    "gpt-5.4":      (1.25,  10.00),
    "gpt-5.4-mini": (0.25,   2.00),
    "gpt-5.4-pro":  (15.00, 120.00),
}

MODEL_CONFIG: dict[str, dict] = {
    "gpt-4.1":      {"endpoint": "duerr", "api": "chat",      "extra": {}},
    "gpt-4.1-mini": {"endpoint": "duerr", "api": "chat",      "extra": {}},
    "o3-mini":      {"endpoint": "duerr", "api": "chat",      "extra": {"reasoning_effort": "medium"}},
    "gpt-5.4":      {"endpoint": "duerr", "api": "chat",      "extra": {"reasoning_effort": "medium"}},
    "gpt-5.4-mini": {"endpoint": "exp2",  "api": "responses", "extra": {}, "reasoning_effort": "none"},
    "gpt-5.4-pro":  {"endpoint": "exp2",  "api": "responses", "extra": {}, "reasoning_effort": "medium"},
}

RESULTS_DIR       = Path("results")
OPT_RESULTS_DIR   = Path("optimization_results")
DATASET_STEMS     = ["humaneval", "mbpp", "mmlu_pro", "gpqa", "gsm8k"]

# ANSI colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
WHITE  = "\033[97m"

_PRINT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Azure client factory
# ---------------------------------------------------------------------------

def _get_client(endpoint_key: str = "duerr") -> "AzureOpenAI | OpenAI | None":
    if endpoint_key == "exp2":
        base_url = os.environ.get("AZURE_FOUNDRY_EXP2_ENDPOINT", "").strip()
        key      = os.environ.get("AZURE_FOUNDRY_EXP2_KEY", "").strip()
        if not base_url or not key or "<" in base_url or "<" in key:
            print(f"{MAGENTA}  exp2 credentials not configured in .env{RESET}")
            return None
        return OpenAI(base_url=base_url, api_key=key)
    endpoint    = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    key         = os.environ.get("AZURE_OPENAI_KEY", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
    if not endpoint or not key or "<" in endpoint or "<" in key:
        print(f"{MAGENTA}  duerr credentials not configured in .env{RESET}")
        return None
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)


# ---------------------------------------------------------------------------
# Result folder helpers
# ---------------------------------------------------------------------------

def _find_latest_for_model(model: str) -> "Path | None":
    """Find the most recent results folder for a model (conf or plain, whichever is latest)."""
    conf_suffix  = f"_{model}_conf"
    plain_suffix = f"_{model}"
    matches = [
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (
            d.name.endswith(conf_suffix)
            or (d.name.endswith(plain_suffix) and not d.name.endswith("_conf"))
        )
    ]
    return sorted(matches)[-1] if matches else None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("type") != "dataset_summary":
                    records.append(rec)
    return records


def _save_jsonl(out_dir: Path, stem: str, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _cost_for(model: str, prompt_toks: int, compl_toks: int) -> float:
    inp, out = MODEL_PRICING.get(model, (0.0, 0.0))
    return (prompt_toks / 1_000_000) * inp + (compl_toks / 1_000_000) * out


# ---------------------------------------------------------------------------
# Single-call API helpers
# ---------------------------------------------------------------------------

def _call_chat(client, deployment: str, system: str, user: str,
               temperature: float = 0.0, extra: dict | None = None) -> dict:
    """Make a chat completion call. Returns dict with answer, cost, elapsed_s, tokens."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
        **(extra or {}),
    )
    elapsed = round(time.perf_counter() - t0, 3)
    answer  = response.choices[0].message.content or ""
    prompt_tokens   = response.usage.prompt_tokens
    completion_tokens   = response.usage.completion_tokens
    return {
        "answer":             answer,
        "cost_usd":           _cost_for(deployment, prompt_tokens, completion_tokens),
        "elapsed_s":          elapsed,
        "prompt_tokens":      prompt_tokens,
        "completion_tokens":  completion_tokens,
        "total_tokens":       response.usage.total_tokens,
        "finish_reason":      response.choices[0].finish_reason,
    }


def _call_responses(client, deployment: str, system: str, user: str,
                    reasoning_effort: str | None = None) -> dict:
    """Make a Responses-API call (gpt-5.x on exp2). Returns same dict as _call_chat."""
    t0 = time.perf_counter()
    kwargs: dict = {}
    if reasoning_effort and reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(
        model=deployment,
        instructions=system,
        input=user,
        **kwargs,
    )
    elapsed = round(time.perf_counter() - t0, 3)
    answer  = response.output_text or ""
    prompt_tokens   = response.usage.input_tokens
    completion_tokens   = response.usage.output_tokens
    return {
        "answer":             answer,
        "cost_usd":           _cost_for(deployment, prompt_tokens, completion_tokens),
        "elapsed_s":          elapsed,
        "prompt_tokens":      prompt_tokens,
        "completion_tokens":  completion_tokens,
        "total_tokens":       response.usage.total_tokens,
        "finish_reason":      response.status,
    }


def _call_model(llm_model_name: str, system: str, user: str, temperature: float = 0.0) -> dict:
    """Unified call that picks chat vs responses API automatically."""
    llm_model    = MODEL_CONFIG[llm_model_name]
    client = _get_client(llm_model["endpoint"])
    if client is None:
        raise RuntimeError(f"Could not build client for {llm_model_name}")
    if llm_model["api"] == "responses":
        return _call_responses(client, llm_model_name, system, user, reasoning_effort=llm_model.get("reasoning_effort"))
    return _call_chat(client, llm_model_name, system, user, temperature, llm_model.get("extra"))


# ---------------------------------------------------------------------------
# Router prompt
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = (
    "You are a task-difficulty classifier. "
    "Classify whether the following question requires a large, powerful language model "
    "or whether a small, cheap model is sufficient. "
    "Reply with exactly one word: 'small' or 'large'. Nothing else."
)


def _send_router_decision_request(router_llm_model: str, question: str) -> tuple[str, float, float]:
    """Returns ('small'|'large', cost_usd, elapsed_s)."""
    request = _call_model(router_llm_model, _ROUTER_SYSTEM, question, temperature=0.0)
    answer = request["answer"].strip().lower()
    decision = "large" if "large" in answer else "small"
    return decision, request["cost_usd"], request["elapsed_s"]


# ---------------------------------------------------------------------------
# Self-consistency vote helpers
# ---------------------------------------------------------------------------

def _majority_vote_text(answers: list[str]) -> str:
    """Return most common non-empty answer; fall back to first if all unique."""
    clean = [a.strip() for a in answers if a.strip()]
    if not clean:
        return answers[0] if answers else ""
    counter = Counter(clean)
    winner, count = counter.most_common(1)[0]
    if count == 1:
        # All unique — fall back to first answer
        return clean[0]
    return winner


def _majority_vote_code(answers: list[str]) -> str:
    """For code tasks: return first answer that has no syntax error; else first."""
    import ast
    for code in answers:
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            continue
    return answers[0] if answers else ""


# ===========================================================================
# METHOD 1: LLM ROUTER
# ===========================================================================

# Each config: (config_file_name, router_llm_model, small_llm_model, large_llm_model)
ROUTER_CONFIGS: list[tuple[str, str, str, str]] = [
    (
        "router__rtr-gpt41mini__small-gpt41mini__large-gpt41",
        "gpt-4.1-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
    ),
    (
        "router__rtr-gpt41mini__small-gpt54mini__large-gpt54",
        "gpt-4.1-mini",
        "gpt-5.4-mini",
        "gpt-5.4",
    ),
    (
        "router__rtr-gpt54mini__small-gpt41mini__large-gpt41",
        "gpt-5.4-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
    ),
    (
        "router__rtr-gpt54mini__small-gpt54mini__large-gpt54",
        "gpt-5.4-mini",
        "gpt-5.4-mini",
        "gpt-5.4",
    ),
]


def run_router(config_name: str, router_llm_model: str,
               small_llm_model: str, large_llm_model: str) -> None:
    """
    For each question in the benchmark:
    1. Call router_llm_model to classify as 'small' or 'large'.
    2. Look up the pre-stored answer from small_llm_model or large_llm_model result folder.
    3. Combine router cost + answer cost → single record.
    Store output in optimization_results/<config_name>/.
    """
    print(f"\n{BOLD}{CYAN}[Router] {config_name}{RESET}")

    small_folder = _find_latest_for_model(small_llm_model)
    large_folder = _find_latest_for_model(large_llm_model)

    if small_folder is None:
        print(f"  {MAGENTA}ERROR: No result folder for small model '{small_llm_model}'. Run benchmarks first.{RESET}")
        return
    if large_folder is None:
        print(f"  {MAGENTA}ERROR: No result folder for large model '{large_llm_model}'. Run benchmarks first.{RESET}")
        return

    print(f"  Small LLM answers : {small_folder.name}")
    print(f"  Large LLM answers : {large_folder.name}")

    out_dir = OPT_RESULTS_DIR / config_name
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    for stem in DATASET_STEMS:
        small_records = _load_jsonl(small_folder / f"{stem}.jsonl")
        large_records = _load_jsonl(large_folder / f"{stem}.jsonl")

        if not small_records and not large_records:
            #TODO: break the code here and stop everything, and print the error
            continue

        # Build lookup by task_id for large records
        large_llm_answers_by_task_id: dict[str, dict] = {r["task_id"]: r for r in large_records}
        small_llm_answers_by_task_id: dict[str, dict] = {r["task_id"]: r for r in small_records}

        # Use small_records as the reference list of questions to iterate
        reference_records = small_records if small_records else large_records
        output_records: list[dict] = []

        print(f"  {stem}: routing {len(reference_records)} questions ...", flush=True)

        lock = threading.Lock()

        def _route_one(rec: dict,
                       _large_llm_answers_by_task_id=large_llm_answers_by_task_id,
                       _small_llm_answers_by_task_id=small_llm_answers_by_task_id,
                       _lock=lock) -> dict:
            task_id  = rec["task_id"]
            question = rec.get("question", "")

            try:
                decision, router_cost, router_elapsed = _send_router_decision_request(router_llm_model, question)
            except Exception as exc:
                print(f"    {MAGENTA}Router call failed for {task_id}: {exc}{RESET}")
                decision, router_cost, router_elapsed = "small", 0.0, 0.0

            if decision == "large" and task_id in _large_llm_answers_by_task_id:
                chosen = dict(_large_llm_answers_by_task_id[task_id])
                chosen_model = large_llm_model
            else:
                chosen = dict(_small_llm_answers_by_task_id.get(task_id, rec))
                chosen_model = small_llm_model

            # Remove old evaluation fields so tester.py re-evaluates cleanly
            chosen.pop("is_correct", None)
            chosen.pop("economic_reward", None)

            # Combine costs and latencies
            combined_cost    = (chosen.get("cost_usd") or 0.0) + router_cost
            combined_elapsed = (chosen.get("elapsed_s") or 0.0) + router_elapsed

            chosen.update({
                "model":             config_name,
                "timestamp":         ts,
                "cost_usd":          combined_cost,
                "elapsed_s":         combined_elapsed,
                "routing_decision":  decision,
                "router_model":      router_llm_model,
                "router_cost_usd":   router_cost,
                "chosen_model":      chosen_model,
            })

            with _lock:
                print(f"    → {task_id}  [{decision}] → {chosen_model}", flush=True)

            return chosen

        with ThreadPoolExecutor(max_workers=8) as pool:
            output_records = list(pool.map(_route_one, reference_records))

        _save_jsonl(out_dir, stem, output_records)
        print(f"  {GREEN}Saved {len(output_records)} records → {out_dir}/{stem}.jsonl{RESET}")

    print(f"{GREEN}[Router] Done → {out_dir}{RESET}")


def run_all_router_configs() -> None:
    for cfg in ROUTER_CONFIGS:
        run_router(*cfg)


# ===========================================================================
# METHOD 2: CASCADE
# ===========================================================================

# (cascade_config_name, small_model, large_model, threshold)
def _cascade_configs() -> list[tuple[str, str, str, int]]:
    configs = []
    for small in ["gpt-4.1-mini", "gpt-5.4-mini"]:
        for large in ["gpt-4.1", "gpt-5.4"]:
            for threshold in [60, 75, 90]:
                small_tag = small.replace(".", "").replace("-", "")
                large_tag = large.replace(".", "").replace("-", "")
                name = f"cascade__small-{small_tag}__large-{large_tag}__T{threshold}"
                configs.append((name, small, large, threshold))
    return configs


def run_cascade(config_name: str, small_model: str,
                large_model: str, threshold: int) -> None:
    """
    Offline simulation — no new API calls needed.
    Reads small model's _conf folder (has confidence_score) and large model's folder.
    Per question:
      - if confidence_score >= threshold  →  use small answer, cost = small cost
      - if confidence_score < threshold   →  use large answer, cost = small + large cost
                                             elapsed = small + large elapsed
    """
    print(f"\n{BOLD}{CYAN}[Cascade] {config_name}{RESET}")

    small_folder = _find_latest_for_model(small_model)
    large_folder = _find_latest_for_model(large_model)

    if small_folder is None:
        print(f"  {MAGENTA}ERROR: No result folder for small model '{small_model}'. Run benchmarks first.{RESET}")
        return
    if large_folder is None:
        print(f"  {MAGENTA}ERROR: No result folder for large model '{large_model}'.{RESET}")
        return

    print(f"  Small (conf)  : {small_folder.name}")
    print(f"  Large         : {large_folder.name}")

    out_dir = OPT_RESULTS_DIR / config_name
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    for stem in DATASET_STEMS:
        small_records = _load_jsonl(small_folder / f"{stem}.jsonl")
        large_records = _load_jsonl(large_folder / f"{stem}.jsonl")

        if not small_records:
            continue

        large_by_id: dict[str, dict] = {r["task_id"]: r for r in large_records}
        output_records: list[dict] = []
        escalated_count = 0

        for rec in small_records:
            task_id    = rec["task_id"]
            confidence = rec.get("confidence_score")

            small_cost    = rec.get("cost_usd") or 0.0
            small_elapsed = rec.get("elapsed_s") or 0.0

            if confidence is None or confidence >= threshold:
                # Keep small answer
                chosen = dict(rec)
                chosen.pop("is_correct", None)
                chosen.pop("economic_reward", None)
                chosen.update({
                    "model":        config_name,
                    "timestamp":    ts,
                    "escalated":    False,
                    "threshold":    threshold,
                    "chosen_model": small_model,
                })
            else:
                # Escalate to large model
                escalated_count += 1
                large_rec = large_by_id.get(task_id)
                if large_rec is None:
                    # Large model has no stored answer for this task — fall back to small
                    chosen = dict(rec)
                    chosen.pop("is_correct", None)
                    chosen.pop("economic_reward", None)
                    chosen.update({
                        "model":        config_name,
                        "timestamp":    ts,
                        "escalated":    False,
                        "escalation_fallback": True,
                        "threshold":    threshold,
                        "chosen_model": small_model,
                    })
                else:
                    large_cost    = large_rec.get("cost_usd") or 0.0
                    large_elapsed = large_rec.get("elapsed_s") or 0.0
                    chosen = dict(large_rec)
                    chosen.pop("is_correct", None)
                    chosen.pop("economic_reward", None)
                    chosen.update({
                        "model":        config_name,
                        "timestamp":    ts,
                        "cost_usd":     round(small_cost + large_cost, 8),
                        "elapsed_s":    round(small_elapsed + large_elapsed, 3),
                        "escalated":    True,
                        "threshold":    threshold,
                        "confidence_score": confidence,
                        "chosen_model": large_model,
                    })

            output_records.append(chosen)

        _save_jsonl(out_dir, stem, output_records)
        pct = round(100 * escalated_count / len(output_records)) if output_records else 0
        print(f"  {stem}: {len(output_records)} records, "
              f"{escalated_count} escalated ({pct}%) "
              f"→ {out_dir}/{stem}.jsonl")

    print(f"{GREEN}[Cascade] Done → {out_dir}{RESET}")


def run_all_cascade_configs() -> None:
    for cfg in _cascade_configs():
        run_cascade(*cfg)


# ===========================================================================
# METHOD 3: SELF-CONSISTENCY
# ===========================================================================

SELFCONS_CONFIGS: list[tuple[str, str, int]] = [
    ("selfcons__gpt41mini__N3", "gpt-4.1-mini", 3),
    ("selfcons__gpt54mini__N3", "gpt-5.4-mini", 3),
]

_CODE_PROMPT = "You are a coding assistant. Return only raw Python code with no markdown fences, no comments, and no explanation."
_MC_PROMPT   = "You are answering a multiple choice question. Respond with only the single letter of the correct option (e.g. 'A'). Nothing else."
_GPQA_PROMPT = "You are answering a multiple choice question. Respond with only the exact answer text (not the letter). Nothing else."
_MATH_PROMPT = "You are solving a math problem. Respond with only the final numerical answer. No explanation."

_STEM_TO_SYSTEM: dict[str, str] = {
    "humaneval": _CODE_PROMPT,
    "mbpp":      _CODE_PROMPT,
    "mmlu_pro":  _MC_PROMPT,
    "gpqa":      _GPQA_PROMPT,
    "gsm8k":     _MATH_PROMPT,
}

_CODE_STEMS = {"humaneval", "mbpp"}


def _selfcons_vote(answers: list[str], is_code: bool) -> str:
    if is_code:
        return _majority_vote_code(answers)
    return _majority_vote_text(answers)


def run_selfcons(config_name: str, model: str, n_votes: int = 3) -> None:
    """
    Makes n_votes real API calls per question (temperature=0.7), then majority-votes.
    Total cost = sum of all n_votes calls.
    Elapsed = max(individual elapsed) since calls run in parallel.
    """
    print(f"\n{BOLD}{CYAN}[Self-consistency] {config_name}{RESET}")
    print(f"  Model: {model}  |  N={n_votes}")

    source_folder = _find_latest_for_model(model)
    if source_folder is None:
        print(f"  {MAGENTA}ERROR: No result folder for model '{model}'. Run benchmarks first.{RESET}")
        return

    print(f"  Question source : {source_folder.name}")

    out_dir = OPT_RESULTS_DIR / config_name
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    for stem in DATASET_STEMS:
        source_records = _load_jsonl(source_folder / f"{stem}.jsonl")
        if not source_records:
            continue

        is_code   = stem in _CODE_STEMS
        system    = _STEM_TO_SYSTEM[stem]
        output_records: list[dict] = []
        lock = threading.Lock()

        print(f"  {stem}: running {len(source_records)} × {n_votes} calls ...", flush=True)

        def _run_one(rec: dict,
                     _system=system,
                     _is_code=is_code,
                     _stem=stem,
                     _lock=lock) -> dict:
            task_id  = rec["task_id"]
            question = rec.get("question", "")

            def _single_vote(_: int) -> dict:
                return _call_model(model, _system, question, temperature=0.7)

            with ThreadPoolExecutor(max_workers=n_votes) as inner:
                votes = list(inner.map(_single_vote, range(n_votes)))

            answers  = [v["answer"] for v in votes]
            winner   = _selfcons_vote(answers, _is_code)

            # Combined cost = sum; elapsed = max (parallel execution)
            total_cost    = sum(v["cost_usd"]  for v in votes)
            max_elapsed   = max(v["elapsed_s"] for v in votes)
            total_ptoks   = sum(v["prompt_tokens"]     for v in votes)
            total_ctoks   = sum(v["completion_tokens"] for v in votes)
            total_toks    = sum(v["total_tokens"]      for v in votes)

            output = dict(rec)
            output.pop("is_correct",      None)
            output.pop("economic_reward", None)
            output.pop("confidence_score", None)

            # For code tasks the JSONL needs 'solution' = prompt + code
            if _stem == "humaneval":
                he_prompt = rec.get("question", "")
                output["solution"] = he_prompt + winner
            elif _stem == "mbpp":
                output["solution"] = winner

            output.update({
                "model":              config_name,
                "timestamp":          ts,
                "model_response":     winner,
                "cost_usd":           round(total_cost, 8),
                "elapsed_s":          round(max_elapsed, 3),
                "prompt_tokens":      total_ptoks,
                "completion_tokens":  total_ctoks,
                "total_tokens":       total_toks,
                "finish_reason":      votes[-1]["finish_reason"],
                "vote_responses":     answers,
                "vote_winner":        winner,
                "n_votes":            n_votes,
            })

            with _lock:
                print(f"    → {task_id}  votes={answers[:3]!r}  winner={winner[:60]!r}", flush=True)

            return output

        with ThreadPoolExecutor(max_workers=4) as pool:
            output_records = list(pool.map(_run_one, source_records))

        _save_jsonl(out_dir, stem, output_records)
        print(f"  {GREEN}Saved {len(output_records)} records → {out_dir}/{stem}.jsonl{RESET}")

    print(f"{GREEN}[Self-consistency] Done → {out_dir}{RESET}")


def run_all_selfcons_configs() -> None:
    for cfg in SELFCONS_CONFIGS:
        run_selfcons(*cfg)


# ===========================================================================
# CLI entry point
# ===========================================================================

def _print_help() -> None:
    print(f"""
{BOLD}{WHITE}optimizer.py — LLM Optimization Methods{RESET}

Usage:
  python optimizer.py --router     Run all {len(ROUTER_CONFIGS)} router configurations
  python optimizer.py --cascade    Run all {len(_cascade_configs())} cascade configurations (requires _conf folders)
  python optimizer.py --selfcons   Run all {len(SELFCONS_CONFIGS)} self-consistency configurations
  python optimizer.py --all        Run all methods in sequence
  python optimizer.py --list       List all configurations and their folder names
  python optimizer.py --help       Show this help

{BOLD}Router configs ({len(ROUTER_CONFIGS)}):{RESET}""")
    for name, rtr, sm, lg in ROUTER_CONFIGS:
        print(f"  {CYAN}{name}{RESET}")
        print(f"      router={rtr}  small={sm}  large={lg}")

    print(f"\n{BOLD}Cascade configs ({len(_cascade_configs())}):{RESET}")
    for name, sm, lg, t in _cascade_configs():
        print(f"  {CYAN}{name}{RESET}")

    print(f"\n{BOLD}Self-consistency configs ({len(SELFCONS_CONFIGS)}):{RESET}")
    for name, mdl, n in SELFCONS_CONFIGS:
        print(f"  {CYAN}{name}{RESET}  model={mdl}  N={n}")

    print(f"\n{WHITE}Output folder: {OPT_RESULTS_DIR}/<config_name>/{RESET}")
    print(f"{WHITE}After running, use tester.py to evaluate and visualize.py to chart the results.{RESET}\n")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        _print_help()
        sys.exit(0)

    if "--list" in args:
        _print_help()
        sys.exit(0)

    if "--all" in args:
        run_all_router_configs()
        run_all_cascade_configs()
        run_all_selfcons_configs()
    else:
        if "--router" in args:
            run_all_router_configs()
        if "--cascade" in args:
            run_all_cascade_configs()
        if "--selfcons" in args:
            run_all_selfcons_configs()
