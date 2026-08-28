#!/usr/bin/env bash
# Downloads huggingface models
set -euo pipefail

# 1. Ensure CLI is installed
if ! command -v huggingface-cli &> /dev/null && ! command -v hf &> /dev/null; then
  echo "Please install huggingface-cli (pip install huggingface_hub[cli])."
  exit 1
fi


# 2. Authenticate (supports non-interactive login)
if [[ -n "${HF_TOKEN-}" ]]; then
  echo "🔐 Logging in using HF_TOKEN environment variable"
  hf auth login --token "$HF_TOKEN"
else
  echo "🔐 Logging in via interactive prompt"
  hf auth login
fi

echo "✅ Login successful:"
hf auth whoami || true

# 3. Model definitions: "repo_id target_dir"
# Assumes MODEL_DIR is already set in env.
models=(
  "meta-llama/Llama-3.3-70B-Instruct ${MODEL_DIR}/datasets/huggingface/models/Llama-3.3-70B-Instruct"
  "lmsys/sglang-EAGLE3-LLaMA3.3-Instruct-70B ${MODEL_DIR}/datasets/huggingface/models/sglang-EAGLE3-LLaMA3.3-Instruct-70B"
)

# 4. Loop through and download
for entry in "${models[@]}"; do
  read -r repo target <<< "$entry"
  echo
  echo "➡️  Downloading '$repo' (default branch) into '$target'"

  mkdir -p "$target"

  hf download "$repo" \
    --repo-type model \
    --local-dir "$target" \
    --exclude "*consolidated*"

  echo "✅ Done: $repo"
done
