"""DeepSeek P2h request and response shape without a network call."""

from demo.p2h_inference import p2h_inference


def test_p2h_answer_tracks_context_usage_and_history(monkeypatch):
    requests_seen = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "ans: Warsaw"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 90, "completion_tokens": 4, "total_tokens": 94},
                "model": "test-model",
            }

    def fake_post(url, json, headers, timeout):
        requests_seen.append((url, json, headers, timeout))
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    result = p2h_inference(
        triplets=[("Marie Curie", "born in", "Warsaw")],
        question="Where was she born?", topic_entities=["Marie Curie"],
        model="test-model", api_key="dummy", max_tokens=32,
        retries=0, repair=False,
        history=[{"role": "user", "content": "Tell me about Marie Curie"},
                 {"role": "assistant", "content": "Previous answer"}],
    )

    url, body, headers, _ = requests_seen[0]
    assert url.endswith("/chat/completions")
    assert url.startswith("https://api.deepseek.com/")
    assert headers["Authorization"] == "Bearer dummy"
    assert body["model"] == "test-model" and body["max_tokens"] == 32
    assert body["thinking"] == {"type": "disabled"}
    assert "seed" not in body
    assert [item["role"] for item in body["messages"]] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    assert "Marie Curie" in body["messages"][-1]["content"]
    assert result["answers"] == ["Warsaw"]
    assert result["context"]["n_triplets"] == 1
    assert result["usage"]["total_tokens"] == 94
    assert len(result["calls"]) == 1
