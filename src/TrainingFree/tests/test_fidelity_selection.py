import pytest

from src.TrainingFree.fidelity_run import filter_records_by_id


def test_record_filter_preserves_input_order_and_returns_only_requested_ids() -> None:
    records = [({"id": "a"}, "first"), ({"id": "b"}, "first"), ({"id": "c"}, "second")]

    selected = filter_records_by_id(records, ("c", "a"))

    assert [row[0]["id"] for row in selected] == ["a", "c"]


def test_record_filter_rejects_missing_sample_ids() -> None:
    with pytest.raises(ValueError, match="not found"):
        filter_records_by_id([({"id": "a"}, "first")], ("missing",))


def test_record_filter_accepts_comma_separated_cli_value() -> None:
    records = [({"id": "a"}, "first"), ({"id": "b"}, "first")]

    selected = filter_records_by_id(records, "a,b")

    assert [row[0]["id"] for row in selected] == ["a", "b"]
