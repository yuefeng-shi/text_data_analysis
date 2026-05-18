import os
import re
import gc
import argparse
import torch
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Optional

from pipeline_class import PipelineMethod, OPENROUTER_MODEL_MAP
from task_descriptions import TASK_DESCRIPTIONS, ONE_SHOT_EXAMPLES

PRICING_MAP = {
    "openai/gpt-5":                       {"prompt": 15.0,  "completion": 60.0},
    "openai/gpt-5-mini":                  {"prompt":  3.0,  "completion": 12.0},
    "openai/gpt-5.4-mini":                {"prompt":  3.0,  "completion": 12.0},
    "openai/gpt-4.1":                     {"prompt":  2.0,  "completion":  8.0},
    "openai/gpt-4.1-mini":                {"prompt":  0.4,  "completion":  1.6},
    "openai/gpt-4o-mini":                 {"prompt":  0.15, "completion":  0.60},
    "meta-llama/llama-3.3-70b-instruct":  {"prompt":  0.40, "completion":  0.40},
    "qwen/qwen3.5-397b-a17b":            {"prompt":  0.39, "completion":  2.34},
    "default":                            {"prompt":  0.0,  "completion":  0.0},
}

MODEL_CONFIG = {
    "llama-3.3-70b":    {"hf_path": "meta-llama/Llama-3-70b-chat-hf",  "arg_name": "llama_70b_path"},
    "llama-3.1-8b":     {"hf_path": "meta-llama/Llama-3-8b-chat-hf",   "arg_name": "llama_8b_path"},
    "llama-3.2-3b":     {"hf_path": "meta-llama/Llama-3-3b-chat-hf",   "arg_name": "llama_3b_path"},
    "qwen-3b-instruct": {"hf_path": "Qwen/Qwen2.5-3B-Instruct",        "arg_name": "qwen_3b_path"},
    "qwen-7b-instruct": {"hf_path": "Qwen/Qwen2.5-7B-Instruct",        "arg_name": "qwen_7b_path"},
    "qwen-72b-instruct":{"hf_path": "Qwen/Qwen2.5-72B-Instruct",       "arg_name": "qwen_72b_path"},
    "google/gemma-4-26b-it":              {"prompt":  0.06, "completion":  0.33}, 
    "google/gemma-4-31b-it":              {"prompt":  0.13, "completion":  0.38}, 
    "qwen3-4b": {
        "type":     "local",
        "hf_path":  "Qwen/Qwen3-4B-Instruct-2507",
        "arg_name": "qwen3_4b_path",       
    },
}


# ── File discovery ────────────────────────────────────────────────────────────

def parse_filepath_metadata(file_path: str) -> Optional[Dict[str, str]]:
    """
    Extract metadata from the two-level directory structure:

        {data_dir}/{dataset_name}/{size}/{dataset_name}_{index}.txt

    dataset_group : parent directory name (e.g. tw-sentiment_analysis)
    data_size     : middle directory — digits only (e.g. 10, 100, 1000)
                    or alpha-prefix + digits for targeted datasets (e.g. microsoft50)
    target        : alpha prefix extracted from middle dir (empty for non-targeted)
    run_number    : last numeric segment of the filename before any suffix
    """
    try:
        filename   = os.path.basename(file_path)
        dir_path   = os.path.dirname(file_path)
        middle_dir = os.path.basename(dir_path)
        parent_dir = os.path.basename(os.path.dirname(dir_path))

        if not filename.endswith(".txt"):
            return None

        m = re.match(r"^([a-zA-Z][a-zA-Z-]*)(\d+)$", middle_dir)
        if m:
            target, data_size = m.group(1), m.group(2)
        elif middle_dir.isdigit():
            target, data_size = "", middle_dir
        else:
            return None

        num_match = re.search(r"_(\d+)(?:_[^_]+)?$", filename[:-4])
        if not num_match:
            return None

        return {
            "dataset_group": parent_dir,
            "data_size":     data_size,
            "run_number":    num_match.group(1),
            "target":        target,
            "full_path":     file_path,
            "source_file":   filename,
        }
    except Exception as e:
        print(f"  [Error] Failed to parse path '{file_path}': {e}")
        return None


# ── Model initialisation ──────────────────────────────────────────────────────

def initialize_model(model_name: str, args: argparse.Namespace) -> PipelineMethod:
    max_tokens = 256 if args.with_explanation else 20
    is_local   = model_name in MODEL_CONFIG and not args.use_openrouter

    if not is_local:
        return PipelineMethod(
            model_name=model_name,
            loading_method="openrouter" if args.use_openrouter else "openai",
            openai_api_key=args.openai_api_key,
            openrouter_api_key=args.openrouter_api_key,
            max_response_tokens=max_tokens,
        )

    cfg        = MODEL_CONFIG[model_name]
    model_path = getattr(args, cfg["arg_name"], None) or \
                 os.path.join(args.local_model_root, cfg["hf_path"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model not found at: {model_path}")

    return PipelineMethod(
        model_name=model_name,
        loading_method="local",
        local_model_paths={model_name: model_path},
        use_quantization=args.use_quantization,
        max_response_tokens=max_tokens,
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def process_response(response: str, answer_type: str, with_explanation: bool = False) -> str:
    response = re.sub(r"\[Answer\]", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"^(answer:|response:|result:)", "", response, flags=re.IGNORECASE).strip()

    if with_explanation:
        lines = response.strip().split("\n")
        response = lines[0].strip() if lines else "no_answer"

    if answer_type == "yes/no":
        low = response.lower()
        if "yes" in low and "no" not in low:  return "yes"
        if "no"  in low and "yes" not in low: return "no"
        if low.strip() in ("yes", "no"):       return low.strip()
        if any(p in low for p in ("true", "correct")):              return "yes"
        if any(p in low for p in ("false", "incorrect", "not")):    return "no"
    elif answer_type == "number":
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", response)
        if nums: return nums[0]
    elif answer_type == "percentage":
        pcts = re.findall(r"\b\d+(?:\.\d+)?%\b", response)
        if pcts: return pcts[0]
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", response)
        if nums: return f"{nums[0]}%"

    clean = response.lower().replace("\n", " ").replace('"', "").strip()[:30]
    return clean if clean else "no_answer"


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(
    model: PipelineMethod,
    file_data: dict,
    questions: list,
    args: argparse.Namespace,
) -> pd.DataFrame:
    results  = []
    metadata = {k: v for k, v in file_data.items() if k != "content"}

    instructions_map = {
        "yes/no": (
            'OUTPUT REQUIREMENT: Respond with "yes" or "no" on the first line. '
            'On a new line, provide a brief explanation.'
            if args.with_explanation else
            'OUTPUT REQUIREMENT: Respond with exactly "yes" or "no" only.'
        ),
        "number": (
            'OUTPUT REQUIREMENT: Respond with only a single integer (e.g., "42") '
            'on the first line. On a new line, provide explanation.'
            if args.with_explanation else
            'OUTPUT REQUIREMENT: Respond with only a single integer (e.g., "42").'
        ),
        "percentage": (
            'OUTPUT REQUIREMENT: Respond with only a percentage in format "X%" '
            '(e.g., "85%") on the first line. On a new line, provide explanation.'
            if args.with_explanation else
            'OUTPUT REQUIREMENT: Respond with only a percentage in format "X%" (e.g., "85%").'
        ),
    }
    default_instruction = (
        "OUTPUT REQUIREMENT: Provide only the direct answer on the first line. "
        "On a new line, provide explanation."
        if args.with_explanation else
        "OUTPUT REQUIREMENT: Provide only the direct answer without explanations."
    )

    with tqdm(questions, desc=f"Processing {metadata['source_file']}",
              unit="q", leave=False) as pbar:
        for q in pbar:
            q_text      = q["Question"]
            answer_type = q.get("Answer-type", "default").strip().lower()
            dataset_key = q.get("Dataset", "")
            instruction = instructions_map.get(answer_type, default_instruction)

            # Prompt-level guard: prevent stray <think> tags leaking through
    
            if any(k in model.model_name.lower() for k in ("qwen3", "qwen3.5", "deepseek", "gemma-4")):
                instruction += (
                    "\nCRITICAL: Output ONLY the final direct answer. "
                    "Do NOT use <think>, <reasoning>, or any internal monologue tags."
                )

            task_desc    = TASK_DESCRIPTIONS.get(dataset_key.strip()) if (args.use_task_description and dataset_key) else None
            one_shot_ex  = ONE_SHOT_EXAMPLES.get(dataset_key.strip())  if (args.use_one_shot       and dataset_key) else None

            try:
                raw, p_tok, c_tok = model.generate_method(
                    text=file_data["content"],
                    question=q_text,
                    instructions=instruction,
                    answer_type=answer_type,
                    task_description=task_desc,
                    one_shot_example=one_shot_ex,
                    dataset_key=dataset_key,
                )

                if raw == "ERROR::CONTEXT_WINDOW_EXCEEDED":
                    final_answer = "context_too_long"
                else:
                    if args.save_raw_output:
                        raw_path = os.path.join(args.output_dir, f"raw_output_{model.model_name.replace('/', '-')}.txt")
                        os.makedirs(args.output_dir, exist_ok=True)
                        with open(raw_path, "a", encoding="utf-8") as f:
                            f.write(f"FILE: {metadata['source_file']}\nQ: {q_text}\n{raw}\n\n")
                    final_answer = process_response(raw, answer_type, args.with_explanation)

            except Exception as e:
                print(f"\n[Error] {metadata['source_file']} | {q_text[:60]}: {e}")
                final_answer, p_tok, c_tok = f"Error: {e}", 0, 0

            results.append({"question": q_text, "answer": final_answer,
                             "prompt_tokens": p_tok, "completion_tokens": c_tok,
                             **metadata})
            pbar.set_postfix_str(f"→ {final_answer}")

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    try:
        questions_df = pd.read_csv(args.question_csv)
        if "target" not in questions_df.columns:
            questions_df["target"] = ""
        questions_df["target"] = questions_df["target"].fillna("")
    except Exception as e:
        print(f"FATAL: Could not read question CSV: {e}")
        return

    all_files: List[dict] = []
    print(f"Scanning '{args.data_dir}'...")
    for root, _, files in os.walk(args.data_dir):
        for name in files:
            if not name.endswith(".txt"):
                continue
            meta = parse_filepath_metadata(os.path.join(root, name))
            if not meta:
                continue
            if not meta.get("target"):
                meta["target"] = ""
            with open(meta["full_path"], "r", encoding="utf-8") as f:
                meta["content"] = f.read()
            all_files.append(meta)

    if not all_files:
        print("[WARN] No .txt files found.")
        return

    questions_by_dataset: Dict[str, list] = {
        ds: grp.to_dict("records")
        for ds, grp in questions_df.groupby("Dataset")
    }

    os.makedirs(args.output_dir, exist_ok=True)
    all_results: List[pd.DataFrame] = []

    for model_name in args.models:
        print(f"\n{'='*40}\nInitialising {model_name}\n{'='*40}")
        model = None
        try:
            model = initialize_model(model_name, args)
            print(f"  [INFO] Files to process: {len(all_files)}")

            for file_data in tqdm(all_files, desc=f"Model: {model_name}", unit="file"):
                dg = file_data["dataset_group"]
                if dg not in questions_by_dataset:
                    print(f"  [SKIP] {file_data['source_file']} — dataset '{dg}' not in CSV")
                    continue

                ft = file_data["target"]
                questions_for_file = [
                    q for q in questions_by_dataset[dg] if q.get("target", "") == ft
                ]
                if not questions_for_file:
                    csv_targets = {q.get("target", "") for q in questions_by_dataset[dg]}
                    print(f"  [SKIP] {file_data['source_file']} — "
                          f"no questions match target='{ft}' (CSV has: {csv_targets})")
                    continue

                print(f"  [RUN]  {file_data['source_file']} — "
                      f"size={file_data['data_size']}, run={file_data['run_number']}, "
                      f"questions={len(questions_for_file)}")

                df = process_file(model, file_data, questions_for_file, args)
                df["model"] = model_name
                all_results.append(df)

        except Exception as e:
            print(f"\nFATAL ERROR for model '{model_name}': {e}")
        finally:
            if model:
                if hasattr(model, "local_pipe") and hasattr(model.local_pipe, "model"):
                    del model.local_pipe.model, model.local_pipe
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not all_results:
        print("\n[WARN] No results collected — check [SKIP] messages above.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    print(f"\n[INFO] Writing results to: {args.output_dir}  ({len(final_df)} rows)")

    # ── Build result column names: res_{size}_{run}_{model} ──────────────────
    sanitized = final_df["model"].str.replace("/", "-").str.replace("_", "-")
    final_df["result_col_name"] = (
        "res_" + final_df["data_size"].astype(str) + "_"
        + final_df["run_number"].astype(str) + "_" + sanitized
    )

    pivoted = (
        final_df
        .pivot_table(
            index=["dataset_group", "question", "target"],
            columns="result_col_name", values="answer", aggfunc="first",
        )
        .reset_index()
    )

    # Merge onto questions filtered to processed datasets only, then drop
    # redundant right-side key columns produced by the merge.
    processed_ds = set(final_df["dataset_group"].unique())
    q_filtered   = questions_df[questions_df["Dataset"].isin(processed_ds)].copy()

    merged = pd.merge(
        q_filtered, pivoted, how="left",
        left_on=["Dataset", "Question", "target"],
        right_on=["dataset_group", "question", "target"],
    )
    merged = merged.drop(
        columns=[c for c in ["dataset_group", "question", "target"] if c in merged.columns]
    )
    merged.to_csv(os.path.join(args.output_dir, "consolidated_results.csv"), index=False)

    # ── Billing report ────────────────────────────────────────────────────────
    if "prompt_tokens" in final_df.columns:
        usage = (
            final_df.groupby(["model", "dataset_group"])
            .agg({"prompt_tokens": "sum", "completion_tokens": "sum"})
            .reset_index()
        )

        def _cost(row: pd.Series) -> float:
            or_id = OPENROUTER_MODEL_MAP.get(row["model"], row["model"])
            price = PRICING_MAP.get(or_id, PRICING_MAP["default"])
            return (row["prompt_tokens"] / 1_000_000 * price["prompt"] +
                    row["completion_tokens"] / 1_000_000 * price["completion"])

        usage["estimated_cost_usd"] = usage.apply(_cost, axis=1)
        total = pd.DataFrame([{
            "model": "TOTAL", "dataset_group": "ALL",
            "prompt_tokens":     usage["prompt_tokens"].sum(),
            "completion_tokens": usage["completion_tokens"].sum(),
            "estimated_cost_usd": usage["estimated_cost_usd"].sum(),
        }])
        usage = pd.concat([usage, total], ignore_index=True)
        usage.to_csv(os.path.join(args.output_dir, "billing_report.csv"), index=False)
        print(f"Total Estimated Cost: ${usage['estimated_cost_usd'].iloc[-1]:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Setting A: Direct Inference Pipeline")
    p.add_argument("--models",               nargs="+", required=True)
    p.add_argument("--data_dir",             required=True)
    p.add_argument("--question_csv",         required=True)
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--openai_api_key",       default=None)
    p.add_argument("--openrouter_api_key",   default=None)
    p.add_argument("--use_openrouter",       action="store_true")
    p.add_argument("--local_model_root",     default="./models")
    p.add_argument("--use_quantization",     action="store_true")
    p.add_argument("--save_raw_output",      action="store_true")
    p.add_argument("--with_explanation",     action="store_true")
    p.add_argument("--use_task_description", action="store_true")
    p.add_argument("--use_one_shot",         action="store_true")
    p.add_argument("--llama_70b_path",  default=None)
    p.add_argument("--llama_8b_path",   default=None)
    p.add_argument("--llama_3b_path",   default=None)
    p.add_argument("--qwen_3b_path",    default=None)
    p.add_argument("--qwen_7b_path",    default=None)
    p.add_argument("--qwen_72b_path",   default=None)
    p.add_argument("--qwen3_4b_path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
