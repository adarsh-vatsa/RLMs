"""Compatibility imports for the shared execution HTTP client."""
from execution.client import Completion, CompletionRetryError, call_chat_completion, call_with_retries

__all__ = ["Completion", "CompletionRetryError", "call_chat_completion", "call_with_retries"]
