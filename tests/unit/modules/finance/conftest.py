class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, messages):
        return self._result


class FakeLLM:
    """Stands in for create_llm_client() in node tests — returns a fixed
    structured-output result regardless of the prompt/messages passed in."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema, method=None):
        return _FakeStructuredLLM(self._result)
