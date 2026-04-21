#!/usr/bin/env bash
set -euo pipefail

# Downloads NoLiMa benchmark assets into benchmark_data/nolima by default.
# Usage:
#   bash nolima/scripts/fetch_dataset.sh
#   bash nolima/scripts/fetch_dataset.sh --target-root benchmark_data/nolima --no-long

TARGET_ROOT="benchmark_data/nolima"
INCLUDE_LONG=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-root)
      TARGET_ROOT="$2"
      shift 2
      ;;
    --no-long)
      INCLUDE_LONG=0
      shift
      ;;
    --with-long)
      INCLUDE_LONG=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Fetch NoLiMa dataset assets.

Options:
  --target-root <path>  Target folder root (default: benchmark_data/nolima)
  --no-long             Skip rand_shuffle_long downloads
  --with-long           Force rand_shuffle_long downloads (default)
  -h, --help            Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not found in PATH" >&2
  exit 1
fi

NEEDLE_DIR="$TARGET_ROOT/needlesets"
HAYSTACK_DIR="$TARGET_ROOT/haystack/rand_shuffle"
HAYSTACK_LONG_DIR="$TARGET_ROOT/haystack/rand_shuffle_long"

mkdir -p "$NEEDLE_DIR" "$HAYSTACK_DIR"
if [[ "$INCLUDE_LONG" -eq 1 ]]; then
  mkdir -p "$HAYSTACK_LONG_DIR"
fi

echo "[NoLiMa] Downloading needlesets into $NEEDLE_DIR"
for f in needle_set.json needle_set_MC.json needle_set_ONLYDirect.json needle_set_hard.json needle_set_w_CoT.json needle_set_w_Distractor.json; do
  curl -fL --retry 3 --retry-delay 2 \
    "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/${f}" \
    -o "$NEEDLE_DIR/$f"
done

echo "[NoLiMa] Downloading rand_shuffle haystacks into $HAYSTACK_DIR"
for i in 1 2 3 4 5; do
  curl -fL --retry 3 --retry-delay 2 \
    "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_${i}.txt" \
    -o "$HAYSTACK_DIR/rand_book_${i}.txt"
done

if [[ "$INCLUDE_LONG" -eq 1 ]]; then
  echo "[NoLiMa] Downloading rand_shuffle_long haystacks into $HAYSTACK_LONG_DIR"
  for i in 1 2 3 4 5; do
    curl -fL --retry 3 --retry-delay 2 \
      "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_${i}.txt" \
      -o "$HAYSTACK_LONG_DIR/rand_book_${i}.txt"
  done
fi

echo
ls -lh "$NEEDLE_DIR"
ls -lh "$HAYSTACK_DIR"
if [[ "$INCLUDE_LONG" -eq 1 ]]; then
  ls -lh "$HAYSTACK_LONG_DIR"
fi
du -sh "$TARGET_ROOT"

echo "[NoLiMa] Dataset fetch complete."
