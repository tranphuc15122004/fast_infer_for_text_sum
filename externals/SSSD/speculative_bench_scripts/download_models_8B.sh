#!/usr/bin/env bash
# Downloads huggingface models
set -euo pipefail

# 1. Ensure CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
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

# 3. Model definitions: "repo_id commit target_dir"
models=(
  "meta-llama/Meta-Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659 ${MODEL_DIR}/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  "jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B main ${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B"
  "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B main ${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
  "meta-llama/Llama-3.2-1B-Instruct main ${MODEL_DIR}/datasets/huggingface/models/Llama-3.2-1B-Instruct"
)

# 4. Loop through and download
for entry in "${models[@]}"; do
  read -r repo commit target <<< "$entry"
  echo
  echo "➡️  Downloading '$repo' at revision '$commit' into '$target'"

  mkdir -p "$target"
    hf download "$repo" \
    ${commit:+--revision "$commit"} \
    --repo-type model \
    --local-dir "$target" \
    --exclude "*consolidated*"

  echo "✅ Done: $repo"
done
