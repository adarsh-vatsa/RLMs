"""Shared grading configuration for generation and saved-answer regrading."""

from __future__ import annotations

import hashlib
import json
import os

from aa_lcr.api import call_chat_completion, call_with_retries
from aa_lcr.prompting import (
    GRADER_PROMPT_VERSION,
    GRADER_PROMPT_VERSIONS,
    build_grader_messages,
    chat_token_count,
    non_thinking_extra_body,
    parse_grade,
    prompt_contract_metadata,
)


def add_grader_arguments(parser):
    parser.add_argument(
        "--grader-prompt-version",
        choices=GRADER_PROMPT_VERSIONS,
        default=GRADER_PROMPT_VERSION,
    )
    parser.add_argument(
        "--grader-api-style", choices=("vllm", "openai"), default="vllm"
    )
    parser.add_argument(
        "--grader-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--grader-max-output-tokens", type=int)
    parser.add_argument(
        "--grader-context-window",
        type=int,
        default=32768,
        help="Total local vLLM grader context; must match the service",
    )
    parser.add_argument(
        "--evaluator-api-key-env",
        help="Defaults to executor key for vLLM or OPENAI_API_KEY for OpenAI",
    )


class Grader:
    def __init__(
        self, args, *, tokenizer_factory, completion_caller=call_chat_completion
    ):
        self.args = args
        self.caller = completion_caller
        self.version = args.grader_prompt_version
        self.style = args.grader_api_style
        self.max_tokens = args.grader_max_output_tokens
        if self.max_tokens is None:
            if self.style == "openai":
                self.max_tokens = 16384
            else:
                self.max_tokens = 8 if self.version == GRADER_PROMPT_VERSION else 64
        if self.max_tokens <= 0 or args.grader_context_window <= 0:
            raise ValueError("Grader token budgets must be positive")
        if self.style == "vllm" and args.grader_reasoning_effort not in {None, "none"}:
            raise ValueError("The vLLM grader profile is non-thinking")
        self.reasoning_effort = (
            args.grader_reasoning_effort or "medium"
            if self.style == "openai"
            else "none"
        )
        key_env = args.evaluator_api_key_env
        if key_env is None:
            key_env = "OPENAI_API_KEY" if self.style == "openai" else args.api_key_env
        self.api_key = os.getenv(key_env, "") if key_env else ""
        self.tokenizer = (
            tokenizer_factory(args.evaluator_model) if self.style == "vllm" else None
        )
        if self.style == "vllm" and self.max_tokens >= args.grader_context_window:
            raise ValueError("Grader output budget leaves no space for its prompt")

    def metadata(self):
        template = getattr(self.tokenizer, "chat_template", None)
        return {
            **{
                key: value
                for key, value in prompt_contract_metadata(self.version).items()
                if key.startswith("grader_")
            },
            "evaluator_model": self.args.evaluator_model,
            "evaluator_base_url": self.args.evaluator_base_url,
            "grader_api_style": self.style,
            "grader_reasoning_effort": self.reasoning_effort,
            "grader_max_output_tokens": self.max_tokens,
            "grader_temperature": 0 if self.style == "vllm" else None,
            "grader_context_window": (
                self.args.grader_context_window if self.tokenizer is not None else None
            ),
            "grader_tokenizer_model": (
                self.args.evaluator_model if self.tokenizer is not None else None
            ),
            "grader_chat_template_sha256": (
                hashlib.sha256(
                    json.dumps(template, sort_keys=True).encode()
                ).hexdigest()
                if template is not None
                else None
            ),
        }

    def grade(self, question, answer, candidate):
        messages = build_grader_messages(question, answer, candidate, self.version)
        if self.tokenizer is not None:
            count = chat_token_count(self.tokenizer, messages)
            if count + self.max_tokens > self.args.grader_context_window:
                raise ValueError(
                    f"Grader prompt ({count}) plus output ({self.max_tokens}) exceeds "
                    f"its {self.args.grader_context_window}-token context; answer not truncated"
                )
        if self.style == "openai":
            extra = {"reasoning_effort": self.reasoning_effort}
        else:
            extra = non_thinking_extra_body(
                grader=self.version == GRADER_PROMPT_VERSION
            )
        if self.version != GRADER_PROMPT_VERSION:
            extra["response_format"] = {"type": "json_object"}
        completion, attempts = call_with_retries(
            lambda: self.caller(
                base_url=self.args.evaluator_base_url,
                model=self.args.evaluator_model,
                messages=messages,
                max_tokens=self.max_tokens,
                extra_body=extra,
                api_style=self.style,
                api_key=self.api_key,
                timeout_seconds=self.args.request_timeout_seconds,
            ),
            max_retries=self.args.max_retries,
        )
        grade = parse_grade(completion.text, self.version)
        if completion.finish_reason in {"length", "content_filter"}:
            grade = ""
        return completion, attempts, grade
