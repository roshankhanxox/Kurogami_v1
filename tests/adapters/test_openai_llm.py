"""OpenAILLM: maps chat.completions.parse()/create() -> LLMResponse. No network --
a stand-in client satisfying just the two methods used is injected instead of
mocking the SDK internals.
"""

import pytest
from pydantic import BaseModel

from kurogami.adapters.llm.openai import OpenAILLM, OpenAIRefusalError


class _Goal(BaseModel):
    x: int


class _StubMessage:
    def __init__(self, content=None, parsed=None, refusal=None):
        self.content = content
        self.parsed = parsed
        self.refusal = refusal


class _StubUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _StubCompletion:
    def __init__(self, message, model="gpt-4o-test"):
        self.choices = [type("Choice", (), {"message": message})()]
        self.usage = _StubUsage()
        self.model = model


class _StubCompletionsAPI:
    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict | None = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _StubClient:
    def __init__(self, response):
        completions = _StubCompletionsAPI(response)
        self.chat = type("Chat", (), {"completions": completions})()


def test_schema_constrained_call_returns_parsed_model():
    parsed = _Goal(x=5)
    client = _StubClient(_StubCompletion(_StubMessage(content='{"x": 5}', parsed=parsed)))
    llm = OpenAILLM(client=client, model="gpt-4o-test")

    response = llm.complete(prompt="p", schema=_Goal)

    assert response.parsed == parsed
    assert response.text == '{"x": 5}'
    assert response.tokens_in == 10
    assert response.tokens_out == 5
    assert response.model_id == "gpt-4o-test"
    assert client.chat.completions.last_kwargs["response_format"] is _Goal


def test_refusal_raises_openai_refusal_error():
    client = _StubClient(_StubCompletion(_StubMessage(refusal="cannot comply")))
    llm = OpenAILLM(client=client)

    with pytest.raises(OpenAIRefusalError, match="cannot comply"):
        llm.complete(prompt="p", schema=_Goal)


def test_unschema_call_uses_create_and_returns_content_text():
    client = _StubClient(_StubCompletion(_StubMessage(content="hello")))
    llm = OpenAILLM(client=client)

    response = llm.complete(prompt="p")

    assert response.text == "hello"
    assert response.parsed is None
    assert "response_format" not in client.chat.completions.last_kwargs


def test_system_prompt_is_sent_as_a_leading_message():
    client = _StubClient(_StubCompletion(_StubMessage(content="hi")))
    llm = OpenAILLM(client=client)

    llm.complete(prompt="p", system="sys")

    messages = client.chat.completions.last_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "p"}


def test_temperature_is_forwarded():
    client = _StubClient(_StubCompletion(_StubMessage(content="hi")))
    llm = OpenAILLM(client=client)

    llm.complete(prompt="p", temperature=0.7)

    assert client.chat.completions.last_kwargs["temperature"] == 0.7
