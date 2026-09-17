#!/usr/bin/env bash

# Resolve readiness requirements without importing Python models or contacting services.
jarvis_execution_services() {
  local legacy_cache="$1" grading="$2"
  shift 2
  local profile=legacy cache_read="" matching=semantic preflight=0 grader_style=vllm
  local grader_url="${OPENAI_COMPAT_EVALUATOR_BASE_URL:-}"
  local verifier_url="" verifier_model=""
  while (( $# )); do
    case "$1" in
      --execution-profile) profile="$2"; shift ;;
      --execution-profile=*) profile="${1#*=}" ;;
      --answer-cache-read) cache_read=1 ;;
      --no-answer-cache-read) cache_read=0 ;;
      --cache-matching) matching="$2"; shift ;;
      --cache-matching=*) matching="${1#*=}" ;;
      --execution-only) grading=0 ;;
      --preflight-only|--route-audit-only) preflight=1 ;;
      --grader-api-style) grader_style="$2"; shift ;;
      --grader-api-style=*) grader_style="${1#*=}" ;;
      --executor-base-url) OPENAI_COMPAT_EXECUTOR_BASE_URL="$2"; shift ;;
      --executor-base-url=*) OPENAI_COMPAT_EXECUTOR_BASE_URL="${1#*=}" ;;
      --evaluator-base-url) grader_url="$2"; shift ;;
      --evaluator-base-url=*) grader_url="${1#*=}" ;;
      --cache-verifier-base-url) verifier_url="$2"; shift ;;
      --cache-verifier-base-url=*) verifier_url="${1#*=}" ;;
      --cache-verifier-model) verifier_model="$2"; shift ;;
      --cache-verifier-model=*) verifier_model="${1#*=}" ;;
    esac
    shift
  done
  if [[ -z "$cache_read" ]]; then
    cache_read="$legacy_cache"
    [[ "$profile" == common ]] && cache_read=0
  fi
  export JARVIS_REQUIRED_SERVICES_SET=1 JARVIS_WAIT_FOR_EXECUTOR=1
  export JARVIS_GRADER_BASE_URL="" JARVIS_VERIFIER_BASE_URL=""
  if [[ "$preflight" == 1 ]]; then
    export JARVIS_WAIT_FOR_EXECUTOR=0
    return
  fi
  : "${OPENAI_COMPAT_EXECUTOR_BASE_URL:?Set the executor endpoint or pass --executor-base-url}"
  export OPENAI_COMPAT_EXECUTOR_BASE_URL
  if [[ "$grading" == 1 ]]; then
    : "${grader_url:?Set the grader endpoint or pass --evaluator-base-url}"
    [[ "$grader_style" == vllm ]] && export JARVIS_GRADER_BASE_URL="$grader_url"
  fi
  if [[ "$cache_read" == 1 && "$matching" == semantic ]]; then
    if [[ "$legacy_cache" == 1 && "$grader_style" == vllm ]]; then
      verifier_url="${verifier_url:-$grader_url}"
    else
      : "${verifier_model:?Pass --cache-verifier-model for semantic caching}"
    fi
    : "${verifier_url:?Pass --cache-verifier-base-url for semantic caching}"
    export JARVIS_VERIFIER_BASE_URL="$verifier_url"
  fi
  echo "[EXECUTION] executor=$OPENAI_COMPAT_EXECUTOR_BASE_URL grader=${JARVIS_GRADER_BASE_URL:-none} cache-verifier=${JARVIS_VERIFIER_BASE_URL:-none}"
}
