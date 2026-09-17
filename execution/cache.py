import os
from types import SimpleNamespace

from execution.client import call_chat_completion, call_with_retries


def configure_verifier(controller, args, *, default_model=None, default_url=None, default_key_env=""):
    """Give cache verification its own transport; it never uses the answer grader."""
    model = args.cache_verifier_model or default_model
    url = args.cache_verifier_base_url or default_url
    if not model or not url:
        raise ValueError("Semantic caching requires a cache verifier model and endpoint")
    controller.EVALUATOR_MODEL = model
    controller.verifier_metadata = {"model": model, "base_url": url,
                                   "prompt_version": "semantic_cache_equivalence_v1", "max_output_tokens": 50}
    key_env = args.cache_verifier_api_key_env or default_key_env

    def complete(**kwargs):
        attempts = 0
        try:
            response, attempts = call_with_retries(lambda: call_chat_completion(
                base_url=url, model=model, messages=kwargs["messages"], max_tokens=kwargs["max_tokens"],
                extra_body={"temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                            "response_format": {"type": "json_object"}},
                api_key=os.getenv(key_env, "") if key_env else "",
                timeout_seconds=getattr(args, "request_timeout_seconds", 1800)),
                max_retries=getattr(args, "max_retries", 1))
        except Exception as exc:
            attempts = getattr(exc, "attempts", 0)
            raise
        finally:
            controller.verifier_attempts = getattr(controller, "verifier_attempts", 0) + attempts
        return SimpleNamespace(content=[SimpleNamespace(text=response.text)],
            usage=SimpleNamespace(input_tokens=response.input_tokens, output_tokens=response.output_tokens))

    controller.verifier_complete = complete
