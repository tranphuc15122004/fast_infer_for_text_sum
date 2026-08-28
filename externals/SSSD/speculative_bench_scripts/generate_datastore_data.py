import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

DEEPINFRA_COMPLETIONS_URL = "https://api.deepinfra.com/v1/openai/completions"


def load_sharegpt_from_hf(dataset_name: str, split: str = "train"):
    """
    Load a ShareGPT-style dataset from the Hugging Face Hub.
    """
    ds = load_dataset(dataset_name, split=split)
    conversations = []
    for i, row in enumerate(ds):
        conv_id = row.get("id")
        if conv_id is None:
            conv_id = row.get("idx")
        if conv_id is None:
            conv_id = f"conv_{i}"  # fallback

        conversations.append(
            {
                "id": conv_id,
                "conversations": row.get("conversations"),
            }
        )
    return conversations


def load_wildchat_first_questions_sharegpt(
    num_conv=150_000,
    dataset_name: str = "allenai/WildChat-1M",
    split: str = "train",
):
    """
    Load English, non-toxic conversations from WildChat-1M and convert them
    into a ShareGPT-style format, keeping only the first user question.

    Returns a list of:
        {
            "id": <conversation_hash>,
            "conversations": [
                {"from": "human", "value": <first user message>},
                {"from": "gpt",   "value": ""}   # dummy assistant turn
            ]
        }
    """
    tot_conv = 0
    ds = load_dataset(dataset_name, split=split)

    conversations = []
    for row in ds:
        # Filter by conversation-level language and toxicity
        if row.get("language") != "English":
            continue
        if row.get("toxic", False):
            continue

        # Find the first non-empty user message in this conversation
        first_user_msg = None
        for turn in row.get("conversation", []):
            if turn.get("role") != "user":
                continue
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            # (optional) enforce per-turn language as English as well
            if turn.get("language") not in (None, "", "English"):
                continue
            first_user_msg = content
            break

        # Skip conversations without a usable first user question
        if not first_user_msg:
            continue

        conv_id = row.get("conversation_hash")
        if conv_id is None:
            conv_id = f"conv_{len(conversations)}"

        conversations.append(
            {
                "id": conv_id,
                "conversations": [
                    {
                        "from": "human",
                        "value": first_user_msg,
                    },
                    {
                        # dummy assistant turn so generate_offpolicy_for_conversation
                        # will call DeepInfra here
                        "from": "gpt",
                        "value": "",
                    },
                ],
            }
        )

        tot_conv += 1
        if num_conv is not None and tot_conv >= num_conv:
            return conversations

    return conversations


def build_qwen_prompt_from_messages(
    tokenizer,
    messages: List[Dict[str, str]],
) -> str:
    """
    Use Qwen's chat template to convert OpenAI-style messages
    into a single prompt string, with thinking DISABLED.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # disable thinking at the template level
    )
    return prompt


def call_deepinfra_completions(
    api_key: str,
    model: str,
    tokenizer,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_tokens: int = 512,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Tuple[str, str]:
    """
    Call DeepInfra's OpenAI-compatible /completions endpoint
    for a (mixed) thinking/non-thinking Qwen/Qwen3 model.

    Returns:
      (model_output_text, prompt_string_used)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = build_qwen_prompt_from_messages(tokenizer, messages)

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_status = None
    last_body = None

    for attempt in range(retries):
        resp = requests.post(
            DEEPINFRA_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=500,
        )
        last_status = resp.status_code
        last_body = resp.text

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices")

            # Robust handling of weird responses
            if not choices:
                raise RuntimeError(
                    f"DeepInfra returned no choices (choices={choices!r}); "
                    f"full response: {json.dumps(data)[:500]!r}"
                )

            first = choices[0]
            # Try text (completions-style), then message.content (chat-style)
            text = first.get("text")
            if text is None and isinstance(first.get("message"), dict):
                text = first["message"].get("content")

            if text is None:
                raise RuntimeError(
                    f"DeepInfra response has choices but no 'text' or "
                    f"'message.content': {json.dumps(data)[:500]!r}"
                )

            return text, prompt

        # Non-200: retry with backoff
        time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(
        f"DeepInfra request failed after {retries} attempts "
        f"(last status {last_status}, body={last_body[:500]!r})"
    )


def normalize_role(from_field: str) -> str:
    """
    Map dataset 'from' values to logical roles: 'human' vs 'gpt'.
    """
    f = from_field.lower()
    if f in {"human", "user", "utente"}:
        return "human"
    if f in {"gpt", "assistant", "chatgpt"}:
        return "gpt"
    return f  # fall back to original


def generate_offpolicy_for_conversation(
    conv: Dict[str, Any],
    api_key: str,
    model: str,
    tokenizer,
    system_prompt: Optional[str],
    max_assistant_turns_per_conv: Optional[int],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    sleep_between_calls: float,
) -> Dict[str, Any]:
    """
    Process a single ShareGPT conversation in off-policy mode,
    saving the exact prompt string we send to DeepInfra.

    If one DeepInfra call fails, we log the error and stop further
    sampling for this conversation, but still return partial results.
    """
    conv_id = conv.get("id", None)
    turns = conv.get("conversations", [])

    history: List[Dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    assistant_samples = []
    assistant_turn_index = 0

    for step_idx, step in enumerate(turns):
        role_src = normalize_role(step.get("from", ""))
        text = step.get("value", "")

        if role_src == "human":
            history.append({"role": "user", "content": text})

        elif role_src == "gpt":
            assistant_turn_index += 1

            if (
                max_assistant_turns_per_conv is not None
                and assistant_turn_index > max_assistant_turns_per_conv
            ):
                # Just maintain context, no more queries
                history.append({"role": "assistant", "content": text})
                continue

            # Copy of history is what we use for the prompt
            prompt_messages = [msg.copy() for msg in history]

            try:
                model_output, prompt_str = call_deepinfra_completions(
                    api_key=api_key,
                    model=model,
                    tokenizer=tokenizer,
                    messages=prompt_messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                # Log detailed info and stop sampling further turns for this conversation
                print(f"ERROR inside conversation {conv_id!r} at step {step_idx}: {e}")
                break

            assistant_samples.append(
                {
                    "assistant_turn_index": assistant_turn_index,
                    "conversation_step_index": step_idx,
                    "prompt": prompt_str,  # exact prompt sent
                    "model_output": model_output,  # DeepInfra/Qwen output
                }
            )

            # Stay on the dataset trajectory
            history.append({"role": "assistant", "content": text})

            if sleep_between_calls > 0:
                time.sleep(sleep_between_calls)

        else:
            continue

    return {
        "id": conv_id,
        "model": model,
        "assistant_samples": assistant_samples,
    }


def load_processed_ids_from_output(path: str) -> Set[str]:
    """
    For JSONL output: return set of conversation IDs already processed.
    """
    processed: Set[str] = set()
    if not os.path.exists(path):
        return processed

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("id")
                if cid is not None:
                    processed.add(cid)
            except json.JSONDecodeError:
                continue
    return processed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel off-policy sampling on FreedomIntelligence ShareGPT "
            "datasets via DeepInfra Qwen/Qwen3 mixed thinking/non-thinking models, "
            "with THINKING DISABLED via client-side chat templates."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="HF dataset, such as 'FreedomIntelligence/sharegpt-italian'",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-14B",
        help='DeepInfra model name (and HF tokenizer id) (default: "Qwen/Qwen3-14B").',
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt (in any language, e.g. Italian).",
    )
    parser.add_argument(
        "--max-assistant-turns-per-conv",
        type=int,
        default=None,
        help="Max assistant turns to sample per conversation.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Max number of conversations to take from the input.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="top_k sampling parameter (forwarded to DeepInfra).",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls inside each conversation.",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPINFRA_TOKEN",
        help="Env var containing your DeepInfra API key (default: DEEPINFRA_TOKEN).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL (skip already processed conversation IDs).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker threads.",
    )

    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"API key not found in env var {args.api_key_env}. "
            f"Set it with: export {args.api_key_env}=<your_token>"
        )

    print(f"Loading tokenizer for model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading dataset {args.input}...")
    if args.input == "allenai/WildChat-1M":
        dataset = load_wildchat_first_questions_sharegpt(
            num_conv=args.max_conversations
        )
    else:
        dataset = load_sharegpt_from_hf(args.input, split="train")
    print(f"Loaded {len(dataset)} conversations.")

    if args.max_conversations is not None:
        dataset = dataset[: args.max_conversations]
        print(f"Using first {len(dataset)} conversations due to --max-conversations.")

    processed_ids: Set[str] = set()
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}...")
        processed_ids = load_processed_ids_from_output(args.output)
        print(f"Found {len(processed_ids)} already processed conversation IDs.")

    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for idx, conv in enumerate(dataset):
        conv_id = conv.get("id", f"conv_{idx}")
        if args.resume and conv_id in processed_ids:
            continue
        tasks.append((idx, conv))

    print(f"Conversations to process in this run: {len(tasks)}")

    if not tasks:
        print("Nothing to do. Exiting.")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    num_done_this_run = 0

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_conv_id = {}
            for idx, conv in tasks:
                conv_id = conv.get("id", f"conv_{idx}")
                future = executor.submit(
                    generate_offpolicy_for_conversation,
                    conv=conv,
                    api_key=api_key,
                    model=args.model,
                    tokenizer=tokenizer,
                    system_prompt=args.system_prompt,
                    max_assistant_turns_per_conv=args.max_assistant_turns_per_conv,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    sleep_between_calls=args.sleep,
                )
                future_to_conv_id[future] = conv_id

            for future in as_completed(future_to_conv_id):
                conv_id = future_to_conv_id[future]
                try:
                    result = future.result()
                except Exception as e:
                    # This should now be rare, since inner errors are handled
                    print(f"ERROR processing conversation {conv_id!r}: {e}")
                    continue

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())
                num_done_this_run += 1
                print(
                    f"Finished conversation {conv_id!r}. "
                    f"Total this run: {num_done_this_run}"
                )

    finally:
        out_f.close()

    print(
        f"Done. Wrote {num_done_this_run} conversations in this run to {args.output}."
    )


if __name__ == "__main__":
    main()


"""
export DEEPINFRA_TOKEN=yR3OiRFkdEMsDdx9mwWADEAoLRNFchk2

nohup python generate_datastore_data.py \
  --input Aeala/ShareGPT_Vicuna_unfiltered \
  --output deepinfra_outputs/sharegpt_english_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 4 \
  --num-workers 40 \
  --resume >> deepinfra_logs/sharegpt_en.log 2>&1 &

nohup python generate_datastore_data.py \
  --input FreedomIntelligence/sharegpt-italian \
  --output deepinfra_outputs/sharegpt_italian_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 4 \
  --num-workers 20 \
  --resume >> deepinfra_logs/sharegpt_ita.log 2>&1 &

nohup python generate_datastore_data.py \
  --input FreedomIntelligence/sharegpt-japanese \
  --output deepinfra_outputs/sharegpt_japanese_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 4 \
  --num-workers 20 \
  --resume >>  deepinfra_logs/sharegpt_jap.log 2>&1 &

nohup python generate_datastore_data.py \
  --input FreedomIntelligence/sharegpt-indonesian \
  --output deepinfra_outputs/sharegpt_indonesian_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 4 \
  --num-workers 20 \
  --resume >>  deepinfra_logs/sharegpt_indonesian.log 2>&1 &

nohup python generate_datastore_data.py \
  --input WizardLMTeam/WizardLM_evol_instruct_V2_196k \
  --output deepinfra_outputs/evol-instruct_english_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 1 \
  --num-workers 150 > deepinfra_logs/evol-instruct_english_outputs.log 2>&1 &

nohup python generate_datastore_data.py \
  --input FreedomIntelligence/evol-instruct-italian \
  --output deepinfra_outputs/evol-instruct_italian_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 4 \
  --num-workers 20 \
  --resume >>  deepinfra_logs/evol-instruct_ita.log 2>&1 &

nohup python generate_datastore_data.py \
  --input FreedomIntelligence/evol-instruct-japanese \
  --output deepinfra_outputs/evol-instruct_japanese_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 4 \
  --num-workers 20 \
  --resume >>  deepinfra_logs/evol-instruct_jap.log 2>&1 &

nohup python generate_datastore_data.py \
  --input FreedomIntelligence/evol-instruct-indonesian \
  --output deepinfra_outputs/evol-instruct_indonesian_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-assistant-turns-per-conv 4 \
  --num-workers 20 \
  --resume >>  deepinfra_logs/evol-instruct_indonesian.log 2>&1 &

nohup python generate_datastore_data.py \
  --input allenai/WildChat-1M \
  --output deepinfra_outputs/wildchat_english_outputs.jsonl \
  --model Qwen/Qwen3-14B \
  --max-conversations 250000 \
  --max-assistant-turns-per-conv 1 \
  --num-workers 180 \
  --resume > deepinfra_logs/wildchat_en_cont.log 2>&1 &
"""
