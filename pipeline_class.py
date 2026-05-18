import os
import re
import time
import logging
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from openai import OpenAI
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_FIRST_DATASETS = {"stw-sentiment", "tw-stance_detection"}

# Module-level map: imported directly by run_pipeline_a.py for billing calculations.
OPENROUTER_MODEL_MAP: Dict[str, str] = {
    "gpt-5":                         "openai/gpt-5",
    "gpt-5-mini":                    "openai/gpt-5-mini",
    "gpt-5.1":                       "openai/gpt-5.1",
    "gpt-5.2":                       "openai/gpt-5.2",
    "gpt-5.4":                       "openai/gpt-5.4",
    "gpt-5.4-mini":                  "openai/gpt-5.4-mini",
    "gpt-4.1":                       "openai/gpt-4.1",
    "gpt-4.1-mini":                  "openai/gpt-4.1-mini",
    "claude-4":                      "anthropic/claude-sonnet-4",
    "gemini-2.5-flash":              "google/gemini-2.5-flash",
    "gemini-2.5-pro":                "google/gemini-2.5-pro",
    "gemini-3-flash":                "google/gemini-3-flash-preview",
    "gemini-3-pro":                  "google/gemini-3-pro-preview",
    "gemini-3.1-flash-lite":         "google/gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro":                "google/gemini-3.1-pro-preview",
    "grok-4":                        "x-ai/grok-4",
    "grok-4-fast":                   "x-ai/grok-4-fast",
    "grok-4.1-fast":                 "x-ai/grok-4.1-fast",
    "deepseek-v3-0324":              "deepseek/deepseek-chat-v3-0324",
    "deepseek-v3.1":                 "deepseek/deepseek-chat-v3.1",
    "deepseek-v3.1-terminus":        "deepseek/deepseek-v3.1-terminus",
    "deepseek-v3.2-exp":             "deepseek/deepseek-v3.2-exp",
    "llama-3.1-8b-instruct":         "meta-llama/llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct":         "meta-llama/llama-3.2-3b-instruct",
    "llama-3.3-70b-instruct":        "meta-llama/llama-3.3-70b-instruct",
    "llama-4-scout":                 "meta-llama/llama-4-scout",
    "qwen-2.5-72b-instruct":         "qwen/qwen-2.5-72b-instruct",
    "qwen-2.5-7b-instruct":          "qwen/qwen-2.5-7b-instruct",
    "qwen-2.5-3b-instruct":          "qwen/qwen-2.5-3b-instruct",
    "qwen3-235b-a22b-2507-instruct": "qwen/qwen3-235b-a22b-2507",
    "qwen3-30b-a3b-instruct-2507":   "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen3-235b-a22b":               "qwen/qwen3-235b-a22b",
    "qwen3-coder-480b-a35b":         "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen3-next-80b-instruct":       "qwen/qwen3-next-80b-a3b-instruct",
    "qwen3-next-80b-thinking":       "qwen/qwen3-next-80b-a3b-thinking",
    "qwen3-0.6b":                    "qwen/qwen3-0.6b-04-28",
    "qwen3-1.7b":                    "qwen/qwen3-1.7b",
    "qwen3-4b":                      "qwen/qwen3-4b",
    "qwen3-8b":                      "qwen/qwen3-8b",
    "qwen3-14b":                     "qwen/qwen3-14b",
    "qwen3-32b":                     "qwen/qwen3-32b",
    "qwen3.5-9b":                    "qwen/qwen3.5-9b",
    "qwen3.5-27b":                   "qwen/qwen3.5-27b",
    "qwen3.5-35b-a3b":               "qwen/qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b":             "qwen/qwen3.5-122b-a10b",
    "qwen3.5-397b-a17b":             "qwen/qwen3.5-397b-a17b",
    "qwen3.5-flash":                 "qwen/qwen3.5-flash",
    "gemma-4-26b":                   "google/gemma-4-26b-a4b-it", 
    "gemma-4-31b":                   "google/gemma-4-31b-it",
}


class PipelineMethod:
    """
    Setting A direct inference pipeline.

    """

    _EFFORT_NONE_MODELS: frozenset = frozenset({
        "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
        "gpt-5.4", "gpt-5.4-mini", "gpt-4.1", "gpt-4.1-mini",
        "grok-4", "grok-4-fast", "grok-4.1-fast",
        "deepseek-v3-0324", "deepseek-v3.1",
        "deepseek-v3.1-terminus", "deepseek-v3.2-exp",
        "qwen3-235b-a22b-2507-instruct", "qwen3-30b-a3b-instruct-2507",
        "qwen3-235b-a22b", "qwen3-coder-480b-a35b",
        "qwen3-next-80b-instruct", "qwen3-next-80b-thinking",
        "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b",
        "qwen3-14b", "qwen3-32b",
        "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b",
        "qwen3.5-122b-a10b", "qwen3.5-397b-a17b", "qwen3.5-flash",
        "llama-4-scout", "gemma-4-26b", "gemma-4-31b",
    })

    _GEMINI25_MODELS: frozenset = frozenset({
        "gemini-2.5-flash", "gemini-2.5-pro",
    })

    _GEMINI3_MODELS: frozenset = frozenset({
        "gemini-3-flash", "gemini-3-pro",
        "gemini-3.1-flash-lite", "gemini-3.1-pro",
    })

    def __init__(
        self,
        model_name: str,
        loading_method: str = "local",
        local_model_paths: Dict[str, str] = None,
        use_quantization: bool = True,
        max_response_tokens: int = 80,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        valid_models = list(OPENROUTER_MODEL_MAP.keys()) + [
            "llama-3.3-70b", "llama-3.1-8b", "llama-3.2-3b",
            "qwen-3b-instruct", "qwen-7b-instruct", "qwen-72b-instruct",
        ]
        if model_name not in valid_models:
            logger.warning(f"Model '{model_name}' is not in the predefined list.")

        self.model_name          = model_name
        self.loading_method      = loading_method
        self.local_model_paths   = local_model_paths or self._default_local_paths()
        self.use_quantization    = use_quantization
        self.max_response_tokens = max_response_tokens
        self.openai_api_key      = openai_api_key
        self.openrouter_api_key  = openrouter_api_key
        self.model_max_length    = 8_192
        self.openrouter_model_map = OPENROUTER_MODEL_MAP

        if self.loading_method == "local":
            self._init_local_model()
        else:
            self._init_online_config()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _default_local_paths(self) -> Dict[str, str]:
        return {
            "llama-3.3-70b":    "meta-llama/Llama-3-70b-chat-hf",
            "llama-3.1-8b":     "meta-llama/Llama-3-8b-chat-hf",
            "llama-3.2-3b":     "meta-llama/Llama-3-3b-chat-hf",
            "qwen-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
            "qwen-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
            "qwen-72b-instruct":"Qwen/Qwen2.5-72B-Instruct",
        }

    def _init_online_config(self):
        context_lengths = {
            "gpt-5": 400_000,       "gpt-5-mini": 131_072,    "gpt-5.1": 400_000,
            "gpt-5.2": 400_000,     "gpt-5.4": 1_050_000,     "gpt-5.4-mini": 1_050_000,
            "gpt-4.1": 1_000_000,   "gpt-4.1-mini": 1_000_000,
            "claude-4": 1_000_000,
            "gemini-2.5-flash": 128_000,  "gemini-2.5-pro": 128_000,
            "gemini-3-flash": 1_048_576,  "gemini-3-pro": 1_048_576,
            "gemini-3.1-flash-lite": 1_048_576, "gemini-3.1-pro": 1_048_576,
            "grok-4": 256_000,      "grok-4-fast": 2_000_000, "grok-4.1-fast": 2_000_000,
            "deepseek-v3-0324": 128_000,  "deepseek-v3.1": 128_000,
            "deepseek-v3.1-terminus": 128_000, "deepseek-v3.2-exp": 128_000,
            "llama-3.1-8b-instruct": 131_072, "llama-3.2-3b-instruct": 131_072,
            "llama-3.3-70b-instruct": 128_000, "llama-4-scout": 328_000,
            "qwen-2.5-72b-instruct": 131_072, "qwen-2.5-7b-instruct": 32_768,
            "qwen-2.5-3b-instruct": 131_072,
            "qwen3-coder-480b-a35b": 1_050_000, "qwen3.5-flash": 1_000_000, "gemma-4-26b": 262_144,
            "gemma-4-31b": 262_144,
        }
        self.model_max_length = context_lengths.get(self.model_name, 262_144)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _init_local_model(self):
        try:
            model_path = self.local_model_paths[self.model_name]
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            hf_len = self.tokenizer.model_max_length
            self.model_max_length = hf_len if hf_len and hf_len < 500_000 else 8_192
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            ) if self.use_quantization else None
            model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=quant_config,
                device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
            )
            self.local_pipe = pipeline(
                "text-generation", model=model, tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        except Exception as e:
            raise RuntimeError(f"Local model init failed for '{self.model_name}': {e}") from e

    # ── API helpers ───────────────────────────────────────────────────────────

    def _get_client_and_model_id(self) -> Tuple[OpenAI, str]:
        if self.loading_method == "openai":
            if not self.model_name.startswith("gpt"):
                raise ValueError("Only GPT models are supported via 'openai' method.")
            openai_name_map = {
                "gpt-5": "gpt-5", "gpt-5-mini": "gpt-5-mini", "gpt-5.1": "gpt-5.1",
                "gpt-5.2": "gpt-5.2", "gpt-5.4": "gpt-5.4", "gpt-5.4-mini": "gpt-5.4-mini",
                "gpt-4.1": "gpt-4-turbo", "gpt-4.1-mini": "gpt-4o-mini",
            }
            return OpenAI(api_key=self.openai_api_key), openai_name_map.get(self.model_name, self.model_name)
        if self.loading_method == "openrouter":
            model_id = OPENROUTER_MODEL_MAP.get(self.model_name)
            if not model_id:
                raise ValueError(f"Model '{self.model_name}' has no OpenRouter mapping.")
            return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.openrouter_api_key), model_id
        raise ValueError(f"Unsupported loading_method: {self.loading_method}")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ── Reasoning suppression ─────────────────────────────────────────────────

    def _reasoning_extra_body(self) -> dict:
        name = self.model_name
        if name in self._EFFORT_NONE_MODELS:
            return {"reasoning": {"effort": "none"}}
        if name in self._GEMINI25_MODELS:
            return {"reasoning": {"max_tokens": 0}}
        if name in self._GEMINI3_MODELS:
            return {"reasoning": {"effort": "minimal"}}
        return {}

    # ── Prompt construction ───────────────────────────────────────────────────

    def _format_data(self, text: str, dataset_key: Optional[str]) -> Tuple[str, str]:
        """
        Return (document_data, data_description) for Setting A.

        The data is kept in its original tab-separated format — no XML wrapping.
        A concise format description is generated so the model understands the
        column structure without being told how to analyse the content.

        Supported line formats
        ----------------------
        Non-target : {entry}
        Targeted   : {target}\t{entry}    (target_first=True datasets)
        """
        target_first = dataset_key in TARGET_FIRST_DATASETS

        lines_out = []
        has_target = False
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            if "\t" in line and target_first:
                # Reorder columns to target\tentry if target_first
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    has_target = True
                    lines_out.append(f"{parts[0].strip()}\t{parts[1].strip()}")
                    continue
            lines_out.append(line.strip())

        document_data = "\n".join(lines_out)

        if has_target or target_first:
            description = (
                "Each line follows the format: {target}\t{entry} "
                "(tab-separated, one record per line). "
                "The first column is the target word; the second is the text entry."
            )
        else:
            description = (
                "Each line is a text entry, one record per line."
            )

        return document_data, description

    def _build_messages(self, static_content: str, dynamic_content: str) -> list:
        """Build messages with provider-appropriate prefix caching."""
        name = self.model_name.lower()
        if "claude" in name:
            return [{"role": "user", "content": [
                {"type": "text", "text": static_content,
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "text", "text": "\n\n" + dynamic_content},
            ]}]
        if "gemini" in name:
            return [{"role": "user", "content": [
                {"type": "text", "text": static_content,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "\n\n" + dynamic_content},
            ]}]
        if "gemma" in name:
            pass
        return [{"role": "user", "content": f"{static_content}\n\n{dynamic_content}"}]

    # ── Generation ────────────────────────────────────────────────────────────

    def _call_api(
        self,
        messages: list,
        max_retries: int = 3,
    ) -> Tuple[str, int, int]:
        for attempt in range(max_retries):
            try:
                client, model_id = self._get_client_and_model_id()
                params: dict = {
                    "model":      model_id,
                    "messages":   messages,
                    "temperature": 0.01,
                    "top_p":      0.9,
                    "max_tokens": self.max_response_tokens,
                }
                extra = self._reasoning_extra_body()
                if extra:
                    params["extra_body"] = extra
                response = client.chat.completions.create(**params)
                if response.choices and response.choices[0].message.content is not None:
                    content = response.choices[0].message.content
                    p = response.usage.prompt_tokens     if response.usage else 0
                    c = response.usage.completion_tokens if response.usage else 0
                    return content, p, c
                logger.warning(f"Empty response on attempt {attempt + 1}.")
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                else:
                    raise
        return "Error: all retries failed", 0, 0

    # ── Public entry point ────────────────────────────────────────────────────

    def generate_method(
        self,
        text: str,
        question: str,
        instructions: str,
        answer_type: str = "default",
        temperature: float = 0.01,
        top_p: float = 0.9,
        with_explanation: bool = False,
        task_description: Optional[str] = None,
        one_shot_example: Optional[str] = None,
        dataset_key: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """
        Build the full prompt and call the model.

        Prompt structure
        ----------------
        static_content  (cached across all questions for the same file):
            role_line
            [task_description]
            [one_shot_example]
            Dataset: {format description}
            --- DATA BEGIN ---
            {raw TSV lines}
            --- DATA END ---

        dynamic_content (changes per question):
            QUESTION: {question}
            OUTPUT REQUIREMENT: {instructions}
            WARNING / explanation note
            ANSWER:

        Returns (response_text, prompt_tokens, completion_tokens).
        """
        document_data, data_description = self._format_data(text, dataset_key)

        # ── Role and task description ─────────────────────────────────────────
        # The role establishes the analyst frame; the task hint primes the model
        # for the type of reasoning required (statistical, not document QA).
        role_line = (
            "You are an expert Text Data Analyst. "
            "Your task is to perform statistical analysis over the text entries "
            "in the dataset below and answer the question based on patterns "
            "observed across the full collection of entries."
        )

        # Per-question type hint: tells the model what kind of analysis to perform
        # before it reads the data, reducing ambiguity on the expected output.
        type_hints = {
            "yes/no":     "The question asks whether a particular property holds "
                          "across the dataset entries.",
            "number":     "The question asks you to count entries that satisfy "
                          "a specific criterion.",
            "percentage": "The question asks you to compute the proportion of "
                          "entries that satisfy a specific criterion.",
        }
        type_hint = type_hints.get(answer_type.lower(), "")

        warning = (
            "Ensure the direct answer is on the first line, followed by your explanation on a new line."
            if with_explanation else
            "WARNING: Any deviation from the specified format will be considered incorrect. "
            "Do not provide explanations, reasoning, or additional context."
        )

        static_parts = [role_line]
        if task_description:
            static_parts.append(task_description.strip())
        if one_shot_example:
            static_parts.append(f"{one_shot_example.strip()}\n\nNow, for the actual task:")
        static_parts += [
            f"Dataset: {data_description}",
            "",
            "--- DATA BEGIN ---",
            document_data,
            "--- DATA END ---",
        ]
        static_content = "\n".join(static_parts)

        dynamic_parts = [f"QUESTION: {question}"]
        if type_hint:
            dynamic_parts.append(type_hint)
        dynamic_parts += [
            "",
            f"OUTPUT REQUIREMENT:\n{instructions}",
            "",
            warning,
            "",
            "ANSWER:",
        ]
        dynamic_content = "\n".join(dynamic_parts)

        total_tokens = self._count_tokens(static_content) + self._count_tokens(dynamic_content)
        if total_tokens >= self.model_max_length - self.max_response_tokens - 5:
            return "ERROR::CONTEXT_WINDOW_EXCEEDED", 0, 0

        try:
            if self.loading_method == "local":
                prompt = static_content + "\n\n" + dynamic_content
                params = {
                    "max_new_tokens": self.max_response_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.local_pipe.tokenizer.pad_token_id,
                }
                output = self.local_pipe(prompt, **params)
                return output[0]["generated_text"][len(prompt):].strip(), 0, 0

            messages = self._build_messages(static_content, dynamic_content)
            return self._call_api(messages)

        except Exception as e:
            return f"Generation Error: {type(e).__name__}: {e}", 0, 0
