"""Tests: robust JSON extraction / decision parsing and the output contract."""
from ai_structural_holes.task.prompts import build_messages
from ai_structural_holes.task.protocol import _extract_json, parse_decision

IDS = ["a1", "a2", "a3"]


def test_output_contract_is_valid_json_shape():
    from ai_structural_holes.task.prompts import output_contract

    for mode in ("minimal", "full"):
        contract = output_contract(mode)
        assert "{{" not in contract
        assert "}}" not in contract


def test_extract_plain_json():
    obj = _extract_json('{"choice": "A"}')
    assert obj == {"choice": "A"}


def test_extract_json_from_code_fence():
    raw = '这是我的选择：\n```json\n{"choice": "B", "scores": {"A": 10, "B": 90}}\n```'
    obj = _extract_json(raw)
    assert obj["choice"] == "B"
    assert obj["scores"]["B"] == 90


def test_extract_json_with_prose_prefix_and_suffix():
    raw = '好的，我选择：{"choice": "C"} 以上是我的答案。'
    obj = _extract_json(raw)
    assert obj["choice"] == "C"


def test_extract_prefers_last_object():
    # a reasoning object followed by the final answer object
    raw = '{"thought": "先想一下"}\n最终：{"choice": "A"}'
    obj = _extract_json(raw)
    assert obj["choice"] == "A"


def test_extract_ignores_braces_inside_strings():
    raw = '{"reason": "包含 } 和 { 的字符串", "choice": "B"}'
    obj = _extract_json(raw)
    assert obj["choice"] == "B"


def test_extract_double_brace_still_fails():
    # sanity: the old double-brace bug produced invalid JSON
    assert _extract_json('{{"choice": "A"}}') is None


def test_parse_decision_maps_letter_to_id():
    parsed = parse_decision('{"choice": "B"}', IDS)
    assert parsed["parse_ok"] is True
    assert parsed["y"][IDS[1]] == 1
    assert parsed["chosen_ids"] == [IDS[1]]


def test_parse_decision_from_fenced_response():
    parsed = parse_decision('```json\n{"choice": "A"}\n```', IDS)
    assert parsed["parse_ok"] is True
    assert parsed["y"][IDS[0]] == 1


def test_build_messages_embeds_single_brace_contract():
    messages, _ = build_messages("查询", ["文一", "文二"], "neutral", "health")
    user = messages[-1]["content"]
    assert "{{" not in user and "}}" not in user
    assert '"choice"' in user
    assert "ranking" not in user


def test_build_messages_full_mode_includes_ranking():
    messages, _ = build_messages(
        "查询", ["文一", "文二"], "neutral", "health", output_mode="full",
    )
    user = messages[-1]["content"]
    assert "ranking" in user
    assert "scores" in user
