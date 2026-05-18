"""Smoke-test the repo's local MLX model wiring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_llm import DEFAULT_MAX_KV_SIZE, DEFAULT_MAX_TOKENS, DEFAULT_MLX_MODEL, quick_generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-shot local MLX generation.")
    parser.add_argument(
        "--prompt",
        default="In one short sentence, say the RLMs local LLM is connected.",
        help="Prompt to send to the local model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MLX_MODEL,
        help="MLX model id or local path. Defaults to RLMS_MLX_MODEL or the repo recommendation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Optional generation cap. Omit to use the MLX backend default.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=DEFAULT_MAX_KV_SIZE,
        help="MLX KV cache limit. Use 2048 by default for the Q3 27B smoke path.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"model: {args.model}")
    print(f"max_kv_size: {args.max_kv_size}")
    print("output:")
    print(quick_generate(args.prompt, model=args.model, max_tokens=args.max_tokens, max_kv_size=args.max_kv_size))


if __name__ == "__main__":
    main()
