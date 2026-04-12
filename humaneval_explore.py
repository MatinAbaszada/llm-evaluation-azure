import os
import re
import json
import time
import random
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Suppress HuggingFace symlink warning on Windows (cosmetic only)
warnings.filterwarnings(
    "ignore",
    message=".*symlinks.*",
    category=UserWarning,
    module="huggingface_hub",
)

from dotenv import load_dotenv
load_dotenv()  # reads HF_TOKEN from .env if present

# Lock that serialises the results output block when models run in parallel
_MODEL_PRINT_LOCK = threading.Lock()

import questionary
from openai import AzureOpenAI, OpenAI
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

# ANSI color codes
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"   # task id / header
YELLOW  = "\033[93m"   # prompt / question
GREEN   = "\033[92m"   # canonical solution / answer
MAGENTA = "\033[95m"   # entry point
BLUE    = "\033[94m"   # test cases
WHITE   = "\033[97m"   # separators / meta info


def sep(char="=", width=80, color=WHITE):
    print(f"{color}{char * width}{RESET}")


def label(text, color):
    print(f"\n{BOLD}{color}{text}{RESET}\n")


def print_humaneval():
    print(f"{WHITE}Loading HumanEval dataset...{RESET}")
    dataset = load_dataset("openai_humaneval", split="test")

    sep()
    print(f"{WHITE}Dataset loaded: {len(dataset)} problems{RESET}")
    print(f"{WHITE}Columns: {dataset.column_names}{RESET}")
    sep()

    for i, sample in enumerate(dataset):
        sep(color=CYAN)
        print(f"{BOLD}{CYAN}Problem #{i + 1}  —  Task ID: {sample['task_id']}{RESET}")
        sep("-", color=CYAN)

        label("[PROMPT / QUESTION]", YELLOW)
        print(f"{YELLOW}{sample['prompt']}{RESET}")

        label("[CANONICAL SOLUTION / ANSWER]", GREEN)
        print(f"{GREEN}{sample['canonical_solution']}{RESET}")

        label("[ENTRY POINT]", MAGENTA)
        print(f"{MAGENTA}{sample['entry_point']}{RESET}")

        label("[TEST CASES]", BLUE)
        print(f"{BLUE}{sample['test']}{RESET}")

        print()


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

def print_mbpp():
    print(f"{WHITE}Loading MBPP dataset...{RESET}")
    dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")

    sep()
    print(f"{WHITE}Dataset loaded: {len(dataset)} problems{RESET}")
    print(f"{WHITE}Columns: {dataset.column_names}{RESET}")
    sep()

    for i, sample in enumerate(dataset):
        sep(color=CYAN)
        print(f"{BOLD}{CYAN}Problem #{i + 1}  —  Task ID: {sample['task_id']}{RESET}")
        sep("-", color=CYAN)

        label("[QUESTION / DESCRIPTION]", YELLOW)
        print(f"{YELLOW}{sample['text']}{RESET}")

        label("[SOLUTION / CODE]", GREEN)
        print(f"{GREEN}{sample['code']}{RESET}")

        label("[TEST LIST]", BLUE)
        for t in sample["test_list"]:
            print(f"{BLUE}  {t}{RESET}")

        if sample.get("test_setup_code"):
            label("[TEST SETUP CODE]", MAGENTA)
            print(f"{MAGENTA}{sample['test_setup_code']}{RESET}")

        if sample.get("challenge_test_list"):
            label("[CHALLENGE TEST LIST]", MAGENTA)
            for t in sample["challenge_test_list"]:
                print(f"{MAGENTA}  {t}{RESET}")

        print()


# ---------------------------------------------------------------------------
# MMLU-Pro
# ---------------------------------------------------------------------------

def print_mmlu_pro():
    print(f"{WHITE}Loading MMLU-Pro dataset...{RESET}")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    sep()
    print(f"{WHITE}Dataset loaded: {len(dataset)} problems{RESET}")
    print(f"{WHITE}Columns: {dataset.column_names}{RESET}")
    sep()

    for i, sample in enumerate(dataset):
        sep(color=CYAN)
        print(
            f"{BOLD}{CYAN}Problem #{i + 1}  —  ID: {sample['question_id']}"
            f"  |  Category: {sample['category']}{RESET}"
        )
        sep("-", color=CYAN)

        label("[QUESTION]", YELLOW)
        print(f"{YELLOW}{sample['question']}{RESET}")

        label("[OPTIONS]", MAGENTA)
        for idx, opt in enumerate(sample["options"]):
            marker = f"  {chr(65 + idx)}. "
            print(f"{MAGENTA}{marker}{opt}{RESET}")

        label("[ANSWER]", GREEN)
        print(f"{GREEN}{sample['answer']}{RESET}")

        if sample.get("cot_content"):
            label("[CHAIN-OF-THOUGHT]", BLUE)
            print(f"{BLUE}{sample['cot_content']}{RESET}")

        print()


# ---------------------------------------------------------------------------
# GPQA
# ---------------------------------------------------------------------------

def print_gpqa():
    print(f"{WHITE}Loading GPQA dataset (gpqa_main)...{RESET}")
    try:
        dataset = load_dataset(
            "Idavidrein/gpqa",
            "gpqa_main",
            split="train",
            token=os.environ.get("HF_TOKEN"),
        )
    except DatasetNotFoundError:
        sep(color=MAGENTA)
        print(f"{BOLD}{MAGENTA}GPQA is a gated dataset — access steps:{RESET}")
        print(f"{YELLOW}  1. Visit  https://huggingface.co/datasets/Idavidrein/gpqa{RESET}")
        print(f"{YELLOW}     and click \"Access repository\" (approved instantly).{RESET}")
        print(f"{YELLOW}  2. Create a read token at https://huggingface.co/settings/tokens{RESET}")
        print(f"{YELLOW}  3. Paste it into the .env file in this project:{RESET}")
        print(f"{GREEN}     HF_TOKEN=hf_your_token_here{RESET}")
        sep(color=MAGENTA)
        return

    sep()
    print(f"{WHITE}Dataset loaded: {len(dataset)} problems{RESET}")
    print(f"{WHITE}Columns: {dataset.column_names}{RESET}")
    sep()

    for i, sample in enumerate(dataset):
        sep(color=CYAN)
        domain    = sample.get("High-level domain") or sample.get("high_level_domain", "")
        subdomain = sample.get("Subdomain")         or sample.get("subdomain", "")
        print(f"{BOLD}{CYAN}Problem #{i + 1}  —  Domain: {domain}  |  Subdomain: {subdomain}{RESET}")
        sep("-", color=CYAN)

        label("[QUESTION]", YELLOW)
        print(f"{YELLOW}{sample['Question']}{RESET}")

        label("[CORRECT ANSWER]", GREEN)
        print(f"{GREEN}{sample['Correct Answer']}{RESET}")

        label("[INCORRECT ANSWERS]", MAGENTA)
        for key in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            if sample.get(key):
                print(f"{MAGENTA}  • {sample[key]}{RESET}")

        print()


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def print_gsm8k():
    print(f"{WHITE}Loading GSM8K dataset...{RESET}")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    sep()
    print(f"{WHITE}Dataset loaded: {len(dataset)} problems{RESET}")
    print(f"{WHITE}Columns: {dataset.column_names}{RESET}")
    sep()

    for i, sample in enumerate(dataset):
        sep(color=CYAN)
        print(f"{BOLD}{CYAN}Problem #{i + 1}{RESET}")
        sep("-", color=CYAN)

        label("[QUESTION]", YELLOW)
        print(f"{YELLOW}{sample['question']}{RESET}")

        label("[ANSWER]", GREEN)
        print(f"{GREEN}{sample['answer']}{RESET}")

        print()


# ---------------------------------------------------------------------------
# Azure OpenAI — test: 1 sample per dataset
# ---------------------------------------------------------------------------

# Pricing in USD per 1 000 000 tokens (input / output)
# Source: Azure OpenAI pricing page (April 2026)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1":      (2.00,   8.00),
    "gpt-4.1-mini": (0.40,   1.60),
    "o3-mini":      (1.10,   4.40),
    "gpt-5.4":      (1.25,  10.00),  
    "gpt-5.4-mini": (0.25,   2.00),
    "gpt-5.4-pro":  (15.00, 120.00),
}

# Per-model routing: endpoint key + extra API parameters
# endpoint "duerr" = aoi-duerr-chat (env: AZURE_OPENAI_*)
# endpoint "exp2"  = ar-foundry-experiments2 (env: AZURE_FOUNDRY_EXP2_*)
MODEL_CONFIG: dict[str, dict] = {
    "gpt-4.1":      {"endpoint": "duerr", "api": "chat",      "extra": {}},
    "gpt-4.1-mini": {"endpoint": "duerr", "api": "chat",      "extra": {}},
    "o3-mini":      {"endpoint": "duerr", "api": "chat",      "extra": {"reasoning_effort": "medium"}},
    "gpt-5.4":      {"endpoint": "duerr", "api": "chat",      "extra": {"reasoning_effort": "medium"}},
    "gpt-5.4-mini": {"endpoint": "exp2",  "api": "responses", "extra": {}, "reasoning_effort": "none"},
    "gpt-5.4-pro":  {"endpoint": "exp2",  "api": "responses", "extra": {}, "reasoning_effort": "medium"},
}

# Pricing: (input $/1M tokens, output $/1M tokens) — Azure pricing page April 2026
# gpt-5.4 family pricing is estimated (not listed on public page)
AZURE_MODELS = [
    questionary.Separator("►  Non-reasoning  ◄"),
    {
        "name": "gpt-4.1-mini     — GPT-4.1 Mini      (duerr-chat)   | $0.40 / $1.60 per 1M",
        "deployment": "gpt-4.1-mini",
    },
    {
        "name": "gpt-4.1          — GPT-4.1           (duerr-chat)   | $2.00 / $8.00 per 1M",
        "deployment": "gpt-4.1",
    },
    {
        "name": "gpt-5.4-mini     — GPT-5.4 Mini      (exp2, no rsn) | ~$0.25 / $2.00 per 1M",
        "deployment": "gpt-5.4-mini",
    },
    questionary.Separator("►  Reasoning  ◄"),
    {
        "name": "o3-mini          — O3-Mini           (duerr-chat)   | $1.10 / $4.40 per 1M",
        "deployment": "o3-mini",
    },
    {
        "name": "gpt-5.4          — GPT-5.4           (duerr-chat, reasoning on) | ~$15.00 / $60.00 per 1M",
        "deployment": "gpt-5.4",
    },
    {
        "name": "gpt-5.4-pro      — GPT-5.4 Pro       (exp2)         | ~$15.00 / $120.00 per 1M",
        "deployment": "gpt-5.4-pro",
    },
]

DATASETS = {
    "HumanEval  — OpenAI code generation (164 problems)": print_humaneval,
    "MBPP       — Google basic Python programming (500 problems)": print_mbpp,
    "MMLU-Pro   — Multi-task language understanding, pro edition": print_mmlu_pro,
    "GPQA       — Graduate-level PhD science QA": print_gpqa,
    "GSM8K      — Grade-school math word problems": print_gsm8k,
}

ALL_MODELS_SENTINEL = "__ALL__"

# Set to True to skip gpt-5.4-pro when "All models" is selected
# (useful during bulk runs to avoid slow/expensive reasoning calls)
EXCLUDE_GPT54_PRO_FROM_ALL = True

_MENU_STYLE = questionary.Style(
    [
        ("selected", "fg:cyan bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan"),
        ("answer", "fg:green bold"),
        ("separator", "fg:yellow"),
    ]
)

_MODEL_STYLE = questionary.Style(
    [
        ("selected", "fg:yellow bold"),
        ("pointer", "fg:yellow bold"),
        ("highlighted", "fg:yellow"),
        ("answer", "fg:green bold"),
        ("separator", "fg:cyan"),
    ]
)


def _get_azure_client(endpoint_key: str = "duerr") -> "AzureOpenAI | OpenAI | None":
    if endpoint_key == "exp2":
        # AI Foundry experiments2 — uses standard OpenAI client with base_url
        base_url = os.environ.get("AZURE_FOUNDRY_EXP2_ENDPOINT", "").strip()
        key      = os.environ.get("AZURE_FOUNDRY_EXP2_KEY", "").strip()
        if not base_url or not key or "<" in base_url or "<" in key:
            sep(color=MAGENTA)
            print(f"{BOLD}{MAGENTA}Azure OpenAI credentials for 'ar-foundry-experiments2' are not configured.{RESET}")
            print(f"{YELLOW}  Open the .env file and fill in the required values.{RESET}")
            sep(color=MAGENTA)
            return None
        return OpenAI(base_url=base_url, api_key=key)
    else:
        endpoint    = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        key         = os.environ.get("AZURE_OPENAI_KEY", "").strip()
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
        if not endpoint or not key or "<" in endpoint or "<" in key:
            sep(color=MAGENTA)
            print(f"{BOLD}{MAGENTA}Azure OpenAI credentials for 'aoi-duerr-chat' are not configured.{RESET}")
            print(f"{YELLOW}  Open the .env file and fill in the required values.{RESET}")
            sep(color=MAGENTA)
            return None
        return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)


def _send_and_log(client, deployment, dataset_name, question_text, correct_answer: str = "", options: list | None = None, system_prompt: str = "You are an AI assistant. Answer the following question concisely.", extra_params: dict | None = None, use_responses_api: bool = False, reasoning_effort: str | None = None, confidence_mode: bool = False):
    """Sends question to Azure OpenAI. Returns (formatted_output: str, cost: float, raw_answer: str, meta: dict)."""
    out  = []
    _sep = lambda char="=", w=80, c=WHITE: out.append(f"{c}{char * w}{RESET}")
    _lbl = lambda text, c: (out.append(f"\n{BOLD}{c}{text}{RESET}"), _sep(c=c))

    _sep(c=CYAN)
    out.append(f"{BOLD}{CYAN}Dataset : {dataset_name}{RESET}")
    _sep("-", c=CYAN)

    _lbl("[QUESTION SENT]", YELLOW)
    out.append(f"{YELLOW}{question_text}{RESET}")

    if options:
        _lbl("[OPTIONS]", MAGENTA)
        for i, opt in enumerate(options):
            out.append(f"{MAGENTA}  {chr(65 + i)}. {opt}{RESET}")

    if confidence_mode:
        is_code_task = "Return only raw Python" in system_prompt
        if is_code_task:
            system_prompt = (
                system_prompt
                + "\n\nIMPORTANT: After your code, on the very last line add a standalone "
                "Python comment with your confidence: # CONFIDENCE:[number] (0-100). "
                "Example final line: # CONFIDENCE:85"
            )
        else:
            system_prompt = (
                system_prompt
                + "\n\nIMPORTANT: On the very last line of your response write exactly: "
                "CONFIDENCE:[number] where [number] is your confidence from 0 to 100. "
                "Example last line: CONFIDENCE:85"
            )

    t_start = time.perf_counter()
    try:
        if use_responses_api:
            # GPT-5 series on exp2 uses the Responses API, not Chat Completions
            # reasoning_effort is passed as reasoning={"effort": ...}, not as a flat kwarg
            responses_kwargs = {**(extra_params or {})}
            if reasoning_effort and reasoning_effort != "none":
                responses_kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(
                model=deployment,
                instructions=system_prompt,
                input=question_text,
                **responses_kwargs
            )
            answer        = response.output_text
            prompt_toks   = response.usage.input_tokens
            compl_toks    = response.usage.output_tokens
            total_toks    = response.usage.total_tokens
            finish_reason = response.status
            model_name    = response.model
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question_text},
            ]
            response      = client.chat.completions.create(model=deployment, messages=messages, **(extra_params or {}))
            answer        = response.choices[0].message.content
            prompt_toks   = response.usage.prompt_tokens
            compl_toks    = response.usage.completion_tokens
            total_toks    = response.usage.total_tokens
            finish_reason = response.choices[0].finish_reason
            model_name    = response.model
    except Exception as exc:
        out.append(f"{MAGENTA}  Request failed: {exc}{RESET}")
        return "\n".join(out), 0.0, "", {"elapsed_s": None, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "finish_reason": None, "confidence_score": None}
    elapsed = time.perf_counter() - t_start

    # Extract confidence score from last line (pattern: # CONFIDENCE:N or CONFIDENCE:N)
    confidence_score = None
    if confidence_mode and answer:
        lines = answer.rstrip("\n").split("\n")
        if lines:
            last = lines[-1].strip()
            m = re.match(r"^#?\s*CONFIDENCE:(\d+)\s*$", last, re.IGNORECASE)
            if m:
                confidence_score = min(100, max(0, int(m.group(1))))
                answer = "\n".join(lines[:-1]).rstrip()

    # Cost calculation
    input_m, output_m = MODEL_PRICING.get(deployment, (0.0, 0.0))
    cost_input  = (prompt_toks / 1_000_000) * input_m
    cost_output = (compl_toks  / 1_000_000) * output_m
    cost_total  = cost_input + cost_output
    has_pricing = input_m > 0

    _lbl("[MODEL RESPONSE]", GREEN)
    out.append(f"{GREEN}{answer}{RESET}")

    if correct_answer:
        _lbl("[CORRECT ANSWER]", CYAN)
        out.append(f"{CYAN}{correct_answer}{RESET}")

    _lbl("[USAGE & TIMING]", WHITE)
    out.append(f"{WHITE}  Time elapsed    : {elapsed:.3f}s{RESET}")
    out.append(f"{WHITE}  Prompt tokens   : {prompt_toks:,}{RESET}")
    out.append(f"{WHITE}  Completion tok  : {compl_toks:,}{RESET}")
    out.append(f"{WHITE}  Total tokens    : {total_toks:,}{RESET}")
    out.append(f"{WHITE}  Model           : {model_name}{RESET}")
    out.append(f"{WHITE}  Finish reason   : {finish_reason}{RESET}")
    if has_pricing:
        out.append(f"{WHITE}  Input cost      : ${cost_input:.6f}  (${input_m:.2f} / 1M tokens){RESET}")
        out.append(f"{WHITE}  Output cost     : ${cost_output:.6f}  (${output_m:.2f} / 1M tokens){RESET}")
        out.append(f"{BOLD}{WHITE}  Total cost      : ${cost_total:.6f}{RESET}")
    else:
        out.append(f"{MAGENTA}  Cost            : pricing not available for this deployment{RESET}")
    out.append("")

    meta = {
        "elapsed_s":        round(elapsed, 3),
        "prompt_tokens":     prompt_toks,
        "completion_tokens": compl_toks,
        "total_tokens":      total_toks,
        "finish_reason":     finish_reason,
        "confidence_score":  confidence_score,
    }
    return "\n".join(out), cost_total if has_pricing else 0.0, answer, meta


def run_azure_samples(deployment: str, count: int = 1, confidence_mode: bool = False):
    cfg               = MODEL_CONFIG.get(deployment, {"endpoint": "duerr", "api": "chat", "extra": {}})
    client            = _get_azure_client(cfg["endpoint"])
    extra_params      = cfg["extra"]
    use_responses_api = cfg.get("api") == "responses"
    reasoning_effort  = cfg.get("reasoning_effort")  # only used for responses API
    if client is None:
        return

    input_m, output_m = MODEL_PRICING.get(deployment, (0.0, 0.0))
    pricing_note = f"  |  ${input_m:.2f}/${output_m:.2f} per 1M tokens (in/out)" if input_m else ""
    sep()
    print(f"{BOLD}{WHITE}Azure OpenAI  —  up to {count} sample(s) per dataset  —  Model: {deployment}{pricing_note}{RESET}")
    sep()

    print(f"{WHITE}Loading dataset samples (up to {count} per dataset)...{RESET}")

    _CODE_PROMPT = "You are a coding assistant. Return only raw Python code with no markdown fences, no comments, and no explanation."
    _MC_PROMPT   = "You are answering a multiple choice question. Respond with only the single letter of the correct option (e.g. 'A'). Nothing else."
    _GPQA_PROMPT = "You are answering a multiple choice question. Respond with only the exact answer text (not the letter). Nothing else."
    _MATH_PROMPT = "You are solving a math problem. Respond with only the final numerical answer. No explanation."

    tasks = []  # list of dicts, one per sample

    he_ds = load_dataset("openai_humaneval", split="test")
    for i, sample in enumerate(he_ds.select(range(min(count, len(he_ds))))):
        tasks.append({
            "ds": "HumanEval", "idx": i, "task_id": sample["task_id"],
            "question": sample["prompt"], "correct": sample["canonical_solution"],
            "options": None, "system": _CODE_PROMPT,
            "he_prompt": sample["prompt"], "he_test": sample["test"], "test_list": None,
        })

    mb_ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    for i, sample in enumerate(mb_ds.select(range(min(count, len(mb_ds))))):
        mbpp_question = (
            sample["text"]
            + "\n\nYour function must pass these exact assertions (use the same function name):\n"
            + "\n".join(sample["test_list"])
        )
        tasks.append({
            "ds": "MBPP", "idx": i, "task_id": f"Mbpp/{sample['task_id']}",
            "question": mbpp_question, "correct": sample["code"],
            "options": None, "system": _CODE_PROMPT,
            "he_prompt": None, "he_test": None, "test_list": sample["test_list"],
        })

    mm_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    for i, sample in enumerate(mm_ds.select(range(min(count, len(mm_ds))))):
        opts     = sample["options"]
        question = f"{sample['question']}\n\nOptions:\n" + "\n".join(f"  {chr(65+j)}. {o}" for j, o in enumerate(opts))
        tasks.append({
            "ds": "MMLU-Pro", "idx": i, "task_id": f"mmlu_pro/{i}",
            "question": question, "correct": sample["answer"],
            "options": opts, "system": _MC_PROMPT,
            "he_prompt": None, "he_test": None, "test_list": None,
        })

    try:
        gp_ds = load_dataset(
            "Idavidrein/gpqa", "gpqa_main", split="train",
            token=os.environ.get("HF_TOKEN"),
        )
        for i, sample in enumerate(gp_ds.select(range(min(count, len(gp_ds))))):
            gpqa_opts = [sample["Correct Answer"]] + [
                sample[k] for k in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
                if sample.get(k)
            ]
            random.shuffle(gpqa_opts)
            gpqa_question = sample["Question"] + "\n\nOptions:\n" + "\n".join(
                f"  {chr(65+j)}. {o}" for j, o in enumerate(gpqa_opts)
            )
            tasks.append({
                "ds": "GPQA", "idx": i, "task_id": f"gpqa/{i}",
                "question": gpqa_question, "correct": sample["Correct Answer"],
                "options": gpqa_opts, "system": _GPQA_PROMPT,
                "he_prompt": None, "he_test": None, "test_list": None,
            })
    except DatasetNotFoundError:
        print(f"{MAGENTA}  GPQA skipped \u2014 set HF_TOKEN and request access first.{RESET}")

    gsm_ds = load_dataset("openai/gsm8k", "main", split="test")
    for i, sample in enumerate(gsm_ds.select(range(min(count, len(gsm_ds))))):
        tasks.append({
            "ds": "GSM8K", "idx": i, "task_id": f"gsm8k/{i}",
            "question": sample["question"], "correct": sample["answer"],
            "options": None, "system": _MATH_PROMPT,
            "he_prompt": None, "he_test": None, "test_list": None,
        })

    # --- Fire requests through a bounded thread pool (max_workers = count) ---
    print(f"\n{WHITE}[{deployment}] Sending {len(tasks)} requests (max {count} concurrent)...{RESET}\n")

    results = {}   # (ds, idx) -> (output_str, cost, raw_answer, meta)
    _lock   = threading.Lock()

    def _run(task):
        key = (task["ds"], task["idx"])
        with _lock:
            print(f"{CYAN}  \u2192 [{deployment}] [{task['ds']} #{task['idx'] + 1}] Sending...{RESET}", flush=True)
        out, cost, raw_answer, meta = _send_and_log(
            client, deployment,
            f"{task['ds']} #{task['idx'] + 1}",
            task["question"], task["correct"], task["options"], task["system"],
            extra_params=extra_params,
            use_responses_api=use_responses_api,
            reasoning_effort=reasoning_effort,
            confidence_mode=confidence_mode,
        )
        results[key] = (out, cost, raw_answer, meta)

    with ThreadPoolExecutor(max_workers=count) as pool:
        pool.map(_run, tasks)

    # --- Collect results into a buffer, then print atomically (safe for parallel model runs) ---
    _out  = []
    _p    = _out.append
    _psep = lambda char="=": _p(f"{WHITE}{char * 80}{RESET}")

    _psep()
    _p(f"{BOLD}{WHITE}  [{deployment}] RESULTS{RESET}")
    _psep()

    total_cost = 0.0
    for task in tasks:
        key = (task["ds"], task["idx"])
        out, cost, _raw, _meta = results[key]
        _p(out)
        total_cost += cost

    _psep()
    _p(f"{BOLD}{CYAN}  [{deployment}] TOTAL TEST COST : ${total_cost:.6f}{RESET}")
    _psep()

    # --- Save results to timestamped + model-named folder ---
    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir    = os.path.join("results", f"{ts}_{deployment}")
    os.makedirs(result_dir, exist_ok=True)

    DS_FILE_MAP = {
        "HumanEval": "humaneval",
        "MBPP":      "mbpp",
        "MMLU-Pro":  "mmlu_pro",
        "GPQA":      "gpqa",
        "GSM8K":     "gsm8k",
    }

    ds_task_groups: dict = {}
    for t in tasks:
        ds_task_groups.setdefault(t["ds"], []).append(t)

    for ds_name, ds_tasks in ds_task_groups.items():
        file_stem  = DS_FILE_MAP[ds_name]
        jsonl_path = os.path.join(result_dir, f"{file_stem}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for t in ds_tasks:
                key = (t["ds"], t["idx"])
                if key not in results:
                    continue
                _, cost, raw_answer, meta = results[key]
                usage_fields = {
                    "elapsed_s":        meta["elapsed_s"],
                    "prompt_tokens":     meta["prompt_tokens"],
                    "completion_tokens": meta["completion_tokens"],
                    "total_tokens":      meta["total_tokens"],
                    "finish_reason":     meta["finish_reason"],
                }
                conf_fields = (
                    {"confidence_score": meta["confidence_score"]}
                    if meta.get("confidence_score") is not None else {}
                )
                if ds_name == "HumanEval":
                    record = {
                        "task_id": t["task_id"],
                        "solution": t["he_prompt"] + raw_answer,
                        "model_response": raw_answer,
                        "correct_answer": t["correct"],
                        "question": t["question"],
                        "he_test": t["he_test"],
                        "model": deployment, "timestamp": ts, "cost_usd": cost,
                        **usage_fields, **conf_fields,
                    }
                elif ds_name == "MBPP":
                    record = {
                        "task_id": t["task_id"],
                        "solution": raw_answer,
                        "model_response": raw_answer,
                        "correct_answer": t["correct"],
                        "question": t["question"],
                        "test_list": t["test_list"],
                        "model": deployment, "timestamp": ts, "cost_usd": cost,
                        **usage_fields, **conf_fields,
                    }
                else:
                    record = {
                        "task_id": t["task_id"],
                        "model_response": raw_answer,
                        "correct_answer": t["correct"],
                        "question": t["question"],
                        "model": deployment, "timestamp": ts, "cost_usd": cost,
                        **usage_fields, **conf_fields,
                    }
                f.write(json.dumps(record) + "\n")

    _p(f"{WHITE}  Results saved \u2192 {result_dir}{os.sep}{RESET}")
    with _MODEL_PRINT_LOCK:
        for _line in _out:
            print(_line)


# ---------------------------------------------------------------------------
# Interactive menu controller
# ---------------------------------------------------------------------------

def _select_model() -> str | None:
    choices = [
        questionary.Choice(
            title="\u2605  All models  \u2014 run every model in sequence",
            value=ALL_MODELS_SENTINEL,
        ),
        questionary.Separator("\u2500" * 50),
    ] + [
        questionary.Choice(title=m["name"], value=m["deployment"]) if isinstance(m, dict) else m
        for m in AZURE_MODELS
    ]
    choices.append(questionary.Choice(title="\u2190  Back", value=None))

    print(f"\n{BOLD}{WHITE}Select a model:{RESET}")
    return questionary.select(
        "Choose deployment (\u2191\u2193 to move, Enter to confirm):",
        choices=choices,
        style=_MODEL_STYLE,
    ).ask()


def _select_question_count() -> int | None:
    choices = [
        questionary.Choice(title="  1 question  per dataset", value=1),
        questionary.Choice(title="  3 questions per dataset", value=3),
        questionary.Choice(title="  5 questions per dataset", value=5),
        questionary.Choice(title=" 10 questions per dataset", value=10),
        questionary.Choice(title=" 25 questions per dataset", value=25),
        questionary.Choice(title="  50 questions per dataset", value=50),
        questionary.Choice(title=" 100 questions per dataset", value=100),
        questionary.Choice(
            title="1000 questions per dataset  (HumanEval/GPQA capped at full set  ⚠ expensive)",
            value=1000,
        ),
        questionary.Choice(
            title=" All questions  (HumanEval=164, MBPP=500, GPQA=198, GSM8K=1319, MMLU-Pro=12k+  ⚠ very expensive)",
            value=999_999,
        ),
        questionary.Choice(title="\u2190  Back",              value=None),
    ]
    print(f"\n{BOLD}{WHITE}Select question count:{RESET}")
    return questionary.select(
        "How many questions per dataset? (\u2191\u2193 to move, Enter to confirm):",
        choices=choices,
        style=_MODEL_STYLE,
    ).ask()


def _select_dataset() -> "callable | None":
    choices = [questionary.Choice(title=label, value=fn) for label, fn in DATASETS.items()]
    choices.append(questionary.Choice(title="\u2190  Back", value=None))
    print(f"\n{BOLD}{WHITE}Select a dataset to browse:{RESET}")
    return questionary.select(
        "Choose dataset (\u2191\u2193 to move, Enter to confirm):",
        choices=choices,
        style=_MENU_STYLE,
    ).ask()


def run_menu():
    main_choices = [
        questionary.Choice(
            title="Browse Dataset Questions  \u2014 explore raw dataset entries",
            value="browse",
        ),
        questionary.Choice(
            title="Run Azure Tests           \u2014 send questions, save results",
            value="test",
        ),
        questionary.Choice(
            title="Evaluate Results          \u2014 check answers from last run",
            value="evaluate",
        ),
        questionary.Separator("\u2500" * 50),
        questionary.Choice(title="Exit", value="exit"),
    ]

    while True:
        print(f"\n{BOLD}{WHITE}===  Dataset Explorer  ==={RESET}")
        choice = questionary.select(
            "What would you like to do? (\u2191\u2193 to move, Enter to confirm):",
            choices=main_choices,
            style=_MENU_STYLE,
        ).ask()

        if choice is None or choice == "exit":
            print(f"{WHITE}Goodbye!{RESET}")
            break

        elif choice == "browse":
            fn = _select_dataset()
            if fn is None:
                continue
            fn()
            input(f"\n{BOLD}{WHITE}Press Enter to return to the menu...{RESET}")

        elif choice == "test":
            deployment = _select_model()
            if deployment is None:
                continue
            count = _select_question_count()
            if count is None:
                continue
            models = (
                [m["deployment"] for m in AZURE_MODELS if isinstance(m, dict)
                 if not (EXCLUDE_GPT54_PRO_FROM_ALL and m["deployment"] == "gpt-5.4-pro")]
                if deployment == ALL_MODELS_SENTINEL
                else [deployment]
            )
            if len(models) == 1:
                run_azure_samples(models[0], count, confidence_mode=True)
            else:
                with ThreadPoolExecutor(max_workers=len(models)) as pool:
                    pool.map(lambda dep, _n=count: run_azure_samples(dep, _n, confidence_mode=True), models)
            input(f"\n{BOLD}{WHITE}Press Enter to return to the menu...{RESET}")

        elif choice == "evaluate":
            deployment = _select_model()
            if deployment is None:
                continue
            from tester import run_all_tests, find_latest_folder_for_model, find_all_model_folders
            if deployment == ALL_MODELS_SENTINEL:
                model_folders = find_all_model_folders()
                if not model_folders:
                    print(f"{MAGENTA}  No results found in results/ folder.{RESET}")
                else:
                    print(f"\n{BOLD}{WHITE}Evaluating {len(model_folders)} model(s): {', '.join(model_folders)}{RESET}\n")
                    for folder in model_folders.values():
                        run_all_tests(folder)
            else:
                folder = find_latest_folder_for_model(deployment)
                if folder:
                    run_all_tests(folder)
                else:
                    print(f"{MAGENTA}  No results found for model: {deployment}{RESET}")
            input(f"\n{BOLD}{WHITE}Press Enter to return to the menu...{RESET}")


if __name__ == "__main__":
    run_menu()
