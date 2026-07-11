"""Kimi thinking-mode payload helpers and KimiClient augmentation."""
from ai_structural_holes.llm.client import (
    KimiClient,
    kimi_disable_thinking,
    kimi_thinking_can_disable,
    resolve_kimi_model,
)


def test_resolve_kimi_model():
    assert resolve_kimi_model("kimi/kimi-k2.6") == "kimi-k2.6"
    assert resolve_kimi_model("moonshot/moonshot-v1-8k") == "moonshot-v1-8k"


def test_kimi_thinking_can_disable():
    assert kimi_thinking_can_disable("kimi-k2.6") is True
    assert kimi_thinking_can_disable("kimi-k2.5") is True
    assert kimi_thinking_can_disable("kimi-k2") is True
    assert kimi_thinking_can_disable("kimi-k2.7-code") is False
    assert kimi_thinking_can_disable("moonshot-v1-8k") is False


def test_kimi_client_disables_thinking(monkeypatch):
    monkeypatch.setenv("KIMI_DISABLE_THINKING", "1")
    client = KimiClient(api_key="test-key")
    payload = client._augment_payload(
        {"model": "kimi-k2.6", "messages": [], "temperature": 0.0, "max_tokens": 800}
    )
    assert payload["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "temperature" not in payload


def test_kimi_client_omits_temperature_for_k25():
    client = KimiClient(api_key="test-key")
    payload = client._augment_payload(
        {"model": "kimi-k2.5", "messages": [], "temperature": 0.0, "max_tokens": 800}
    )
    assert "temperature" not in payload


def test_kimi_client_skips_k27_code(monkeypatch):
    monkeypatch.setenv("KIMI_DISABLE_THINKING", "1")
    client = KimiClient(api_key="test-key")
    payload = client._augment_payload({"model": "kimi-k2.7-code", "messages": []})
    assert "extra_body" not in payload


def test_kimi_disable_thinking_env(monkeypatch):
    monkeypatch.setenv("KIMI_DISABLE_THINKING", "0")
    client = KimiClient(api_key="test-key")
    payload = client._augment_payload({"model": "kimi-k2.6", "messages": []})
    assert "extra_body" not in payload
