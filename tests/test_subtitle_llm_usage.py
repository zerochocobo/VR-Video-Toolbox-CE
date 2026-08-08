import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool_subtitle import logic


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_post(captured, usage):
    def post(url, headers=None, json=None, timeout=None):
        captured.append({"url": url, "payload": json, "timeout": timeout})
        return _FakeResponse({
            "choices": [{"message": {"content": "<1>ok</1>"}}],
            "usage": usage,
        })
    return post


def _usage(prompt=100, completion=40, reasoning=0, cache_hit=0):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_cache_hit_tokens": cache_hit,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


def test_deepseek_thinking_effort_is_pinned_low_by_default():
    """Never inherit the server-side default effort ("high"), which is what
    made a single 168-line proofread chunk crawl."""
    captured = []
    client = logic.LLMClient("https://api.deepseek.com/", "k", "deepseek-v4-flash")
    with patch("requests.post", _fake_post(captured, _usage())):
        client.complete("hi")
    payload = captured[0]["payload"]
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "low"
    assert captured[0]["timeout"] == 900


def test_api_test_timeout_is_short():
    captured = []
    client = logic.LLMClient("https://api.deepseek.com/", "k", "deepseek-v4-flash",
                             request_timeout=logic.API_TEST_TIMEOUT)
    with patch("requests.post", _fake_post(captured, _usage())):
        client.complete("hi")
    assert logic.API_TEST_TIMEOUT == 30
    assert captured[0]["timeout"] == 30


def test_thinking_can_be_disabled():
    captured = []
    client = logic.LLMClient(
        "https://api.deepseek.com/", "k", "deepseek-v4-pro", enable_thinking=False,
    )
    with patch("requests.post", _fake_post(captured, _usage())):
        client.complete("hi")
    payload = captured[0]["payload"]
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload


def test_unknown_provider_gets_no_thinking_field():
    captured = []
    client = logic.LLMClient("https://api.example.com/v1", "k", "gpt-fake")
    with patch("requests.post", _fake_post(captured, _usage())):
        client.complete("hi")
    assert "thinking" not in captured[0]["payload"]


def test_usage_is_reported_per_phase():
    client = logic.LLMClient("https://api.deepseek.com/", "k", "deepseek-v4-flash")
    with patch("requests.post", _fake_post([], _usage(prompt=100, completion=40, reasoning=10, cache_hit=64))):
        client.set_phase("AI proofread")
        client.complete("a")
        client.set_phase("AI translate")
        client.complete("b")
        client.complete("c")

    lines = client.usage_lines()
    assert any(line.startswith("AI proofread: 1 calls") for line in lines)
    assert any(line.startswith("AI translate: 2 calls") for line in lines)
    total = lines[-1]
    assert total.startswith("total: 3 calls")
    assert "input 300" in total
    assert "cache hit 192" in total
    assert "output 120" in total
    assert "reasoning 30" in total

    logged = []
    logic.log_llm_usage(client, logged.append)
    assert logged[0] == "[INFO] API usage:"
    assert len(logged) == 4


def test_log_llm_usage_tolerates_stub_client():
    logged = []
    logic.log_llm_usage(object(), logged.append)
    assert logged == []


def test_make_llm_client_reads_config():
    cfg = dict(logic.DEFAULT_TRANS_CONFIG)
    client = logic.make_llm_client(cfg, "key")
    assert client.enable_thinking is True
    assert client.reasoning_effort == "low"
    assert client.request_timeout == 900
    assert client.send_thinking_field is True


def test_shipped_config_relies_on_code_defaults():
    """The shipped JSON stays lean; thinking/timeout live in DEFAULT_TRANS_CONFIG
    so existing user config files pick them up on upgrade."""
    import json

    path = Path(__file__).resolve().parent.parent / "config" / "subtitle_trans_config.json"
    shipped = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("enable_thinking", "reasoning_effort", "request_timeout"):
        assert key not in shipped

    merged = dict(logic.DEFAULT_TRANS_CONFIG)
    merged.update(shipped)
    client = logic.make_llm_client(merged, "key")
    assert client.enable_thinking is True
    assert client.reasoning_effort == "low"
    assert client.request_timeout == 900
