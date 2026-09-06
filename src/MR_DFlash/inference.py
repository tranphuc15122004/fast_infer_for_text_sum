"""Reference speculative inference cho MR-DFlash.

Bản này ưu tiên tính đúng và testability: verifier chạy target full-prefix
trên mỗi vòng. Memory API vẫn incremental và không bao giờ nhận token bị
reject; backend KV/paged-attention có thể thay ở benchmark GPU sau.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from .checkpoint import warm_start_draft_model
from .memory import MRMemoryState
from .mr_model import MRDFlashDraftModel
from .training import build_mr_draft_spec_from_target_config


@dataclass
class PrefillOutput:
    input_ids: torch.Tensor
    memory: MRMemoryState
    target_logits: torch.Tensor


@dataclass
class DraftOutput:
    proposed_ids: torch.Tensor
    logits: torch.Tensor


@dataclass
class VerifyOutput:
    accepted_proposal_count: int
    accepted_ids: torch.Tensor
    memory: MRMemoryState
    target_logits: torch.Tensor


@dataclass
class GenerationOutput:
    input_ids: torch.Tensor
    memory: MRMemoryState
    accepted_proposal_tokens: int


class MRDFlashInferenceEngine:
    """Greedy target verification cho một MR-DFlash draft model."""

    def __init__(
        self,
        target_model: torch.nn.Module,
        draft_model: MRDFlashDraftModel,
        *,
        mask_token_id: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.device = device or next(target_model.parameters()).device
        self.mask_token_id = int(mask_token_id)
        self.embed_tokens = target_model.get_input_embeddings()
        self.lm_head = target_model.get_output_embeddings()
        if self.embed_tokens is None or self.lm_head is None:
            raise ValueError("target model phải có input embedding và output head")
        self.target_model.to(self.device).eval()
        self.draft_model.to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

    def _extract_features(self, outputs: Any) -> torch.Tensor:
        layer_ids = self.draft_model.spec.target_layer_ids
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("target output thiếu hidden_states")
        return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)

    @torch.no_grad()
    def _target_forward(self, input_ids: torch.Tensor) -> Any:
        return self.target_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> PrefillOutput:
        """Chạy target trên prefix và tạo HCA/CSA memory."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids phải có dạng [B,S], S>=1")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        outputs = self._target_forward(ids)
        features = self._extract_features(outputs)
        memory = self.draft_model.build_memory(features)
        return PrefillOutput(
            input_ids=ids,
            memory=memory,
            target_logits=outputs.logits[:, -1],
        )

    def _block_mask(self, batch: int, length: int, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full(
            (batch, 1, length, length),
            torch.finfo(dtype).min,
            device=self.device,
            dtype=dtype,
        )
        allow = torch.tril(torch.ones((length, length), device=self.device, dtype=torch.bool))
        return mask.masked_fill(allow.view(1, 1, length, length), 0.0)

    @torch.no_grad()
    def draft_block(
        self,
        input_ids: torch.Tensor,
        memory: MRMemoryState,
    ) -> DraftOutput:
        """Sinh tối đa ``block_size-1`` token sau token cuối prefix."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids phải có dạng [B,S], S>=1")
        if input_ids.shape[0] != 1:
            raise ValueError("reference inference hiện chỉ hỗ trợ batch=1")
        if memory.total_tokens != input_ids.shape[1]:
            raise ValueError("memory.total_tokens không khớp prefix length")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        block_length = self.draft_model.block_size
        noise_ids = torch.full(
            (1, block_length),
            self.mask_token_id,
            device=self.device,
            dtype=torch.long,
        )
        noise_ids[:, 0] = ids[:, -1]
        noise_embedding = self.embed_tokens(noise_ids)
        start = memory.total_tokens - 1
        position_ids = torch.arange(
            start, start + block_length, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        dtype = next(self.draft_model.parameters()).dtype
        hidden = self.draft_model(
            noise_embedding=noise_embedding,
            memory=memory,
            position_ids=position_ids,
            attention_mask=self._block_mask(1, block_length, dtype),
        )
        full_logits = self.lm_head(hidden)
        logits = full_logits[:, 1:]
        return DraftOutput(proposed_ids=logits.argmax(dim=-1), logits=logits)

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,
        proposed_ids: torch.Tensor,
        memory: MRMemoryState,
    ) -> VerifyOutput:
        """Greedy verify; chỉ append accepted proposals hoặc một replacement."""
        if input_ids.shape[0] != 1 or proposed_ids.shape[0] != 1:
            raise ValueError("reference inference hiện chỉ hỗ trợ batch=1")
        if proposed_ids.ndim != 2 or proposed_ids.shape[1] < 1:
            raise ValueError("proposed_ids phải có dạng [1,K], K>=1")
        prefix = input_ids.to(device=self.device, dtype=torch.long)
        proposals = proposed_ids.to(device=self.device, dtype=torch.long)
        if memory.total_tokens != prefix.shape[1]:
            raise ValueError("memory.total_tokens không khớp prefix length")
        prefix_len = prefix.shape[1]
        candidate = torch.cat([prefix, proposals], dim=1)
        candidate_outputs = self._target_forward(candidate)
        candidate_features = self._extract_features(candidate_outputs)
        choices = candidate_outputs.logits[:, prefix_len - 1 : prefix_len - 1 + proposals.shape[1]].argmax(dim=-1)
        equal = choices.eq(proposals)
        accepted_count = 0
        while accepted_count < proposals.shape[1] and bool(equal[0, accepted_count]):
            accepted_count += 1
        if accepted_count == proposals.shape[1]:
            accepted_ids = proposals
            new_outputs = candidate_outputs
            new_features = candidate_features
        else:
            replacement = choices[:, accepted_count : accepted_count + 1]
            accepted_ids = torch.cat([proposals[:, :accepted_count], replacement], dim=1)
            new_ids = torch.cat([prefix, accepted_ids], dim=1)
            new_outputs = self._target_forward(new_ids)
            new_features = self._extract_features(new_outputs)
        appended_features = new_features[:, prefix_len:]
        appended_positions = torch.arange(
            prefix_len,
            prefix_len + accepted_ids.shape[1],
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        new_memory = self.draft_model.append_memory(
            memory,
            appended_features,
            positions=appended_positions,
        )
        return VerifyOutput(
            accepted_proposal_count=accepted_count,
            accepted_ids=accepted_ids,
            memory=new_memory,
            target_logits=new_outputs.logits[:, -1],
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> GenerationOutput:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens phải >= 1")
        prefill = self.prefill(input_ids)
        current = prefill.input_ids
        memory = prefill.memory
        generated = 0
        accepted_proposal_tokens = 0
        while generated < max_new_tokens:
            draft = self.draft_block(current, memory)
            remaining = max_new_tokens - generated
            proposals = draft.proposed_ids[:, :remaining]
            verified = self.verify(current, proposals, memory)
            accepted_proposal_tokens += verified.accepted_proposal_count
            current = torch.cat([current, verified.accepted_ids], dim=1)
            memory = verified.memory
            generated += verified.accepted_ids.shape[1]
            if eos_token_id is not None and int(current[0, -1]) == int(eos_token_id):
                break
        return GenerationOutput(
            input_ids=current,
            memory=memory,
            accepted_proposal_tokens=accepted_proposal_tokens,
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MR-DFlash reference inference")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-checkpoint-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mask-token-id", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=None)
    parser.add_argument("--hca-compression-ratio", type=int, default=128)
    parser.add_argument("--csa-compression-ratio", type=int, default=4)
    parser.add_argument("--memory-local-window", type=int, default=128)
    parser.add_argument("--csa-top-k", type=int, default=64)
    parser.add_argument("--mr-num-stages", type=int, default=2)
    parser.add_argument("--indexer-dim", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.torch_dtype]
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    spec = build_mr_draft_spec_from_target_config(
        target.config,
        draft_num_hidden_layers=1,
        block_size=args.block_size,
        target_layer_ids=args.target_layer_ids,
        mask_token_id=args.mask_token_id,
        num_stages=args.mr_num_stages,
        hca_compression_ratio=args.hca_compression_ratio,
        csa_compression_ratio=args.csa_compression_ratio,
        local_window=args.memory_local_window,
        csa_top_k=args.csa_top_k,
        indexer_dim=args.indexer_dim,
    )
    draft = MRDFlashDraftModel(spec).to(device=device, dtype=dtype)
    warm_start_draft_model(draft, args.draft_checkpoint_path, key_prefix="draft_model.", strategy_name="mr_dflash")
    encoded = tokenizer(args.prompt, return_tensors="pt", return_attention_mask=False)
    result = MRDFlashInferenceEngine(
        target,
        draft,
        mask_token_id=args.mask_token_id,
        device=device,
    ).generate(encoded["input_ids"], max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(result.input_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DraftOutput",
    "GenerationOutput",
    "MRDFlashInferenceEngine",
    "PrefillOutput",
    "VerifyOutput",
]
