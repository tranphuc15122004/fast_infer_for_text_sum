"""Top-N logits không đại diện một phân phối full-vocabulary."""
import json
import pytest
from scripts.run_claim1_pre_method_experiment import compute_prob_and_margin, load_traces


def test_partial_logits_need_full_normalizer_before_claiming_confidence():
    with pytest.raises(ValueError, match="normalizer"):
        compute_prob_and_margin([3., 2.])


def test_first_failure_stops_risk_set_even_when_later_tokens_match(tmp_path):
    rows = [{"status":"ok", "sample_id":"doc::prefix0", "document_id":"doc", "dataset":"gov_report",
             "round_index":0, "draft_position":j, "candidate_logits":[3.,2.],
             "dflash_selected_token_id":10, "target_token_id":11 if j==2 else 10,
             "self_condition_stage":"baseline"} for j in range(1,16)]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    records, metadata = load_traces([path])
    assert [row["label"] for row in records] == [1,0]
    assert [row["position"] for row in records] == [1,2]
    assert metadata["complete_blocks"] == 1
