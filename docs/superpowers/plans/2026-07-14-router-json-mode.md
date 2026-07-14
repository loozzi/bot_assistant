# Router json_mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch `router.py`'s structured-output call from LangChain's default tool-calling method to `json_mode`, so classification works against OpenAI-compatible proxies/models that don't correctly honor `tool_choice`.

**Architecture:** Two-line change inside `app/orchestrator/router.py`: `with_structured_output(schema, method="json_mode")`, plus one added sentence to `_build_system_prompt()`'s returned string instructing the model to respond as JSON (a general requirement of JSON mode).

**Tech Stack:** Python 3.10, LangChain (`ChatOpenAI.with_structured_output(..., method="json_mode")`, confirmed supported in the installed `langchain-openai` version).

## Global Constraints

- No pytest configured; verify with the manual `uv run python -c "..."` commands below.
- **Do not run `git add` or `git commit`.** Leave changes uncommitted for the user to review and commit themselves.
- Do not change `_build_intent_schema`, `_load_module_specs`, the `except Exception` failure-handling boundary, or anything in `app/infra/providers/*` — all out of scope, already reviewed.

---

### Task 1: Switch to `json_mode` and add the JSON-response instruction

**Files:**
- Modify: `app/orchestrator/router.py:57-84` (`_build_system_prompt`'s return statement and the `with_structured_output` call inside `classify_intent`)

**Interfaces:**
- Consumes/produces: no signature changes — `classify_intent(state: AgentState) -> str` and `_build_system_prompt(specs: dict[str, dict]) -> str` keep their exact current signatures.

- [ ] **Step 1: Update `_build_system_prompt`'s return statement**

Current (`app/orchestrator/router.py:49-54`):

```python
    return (
        "You are an intent classifier for a Telegram personal-assistant bot. "
        "Given the user's latest message, choose exactly one module that should "
        'handle it. If no module clearly applies, choose "unknown".\n\n'
        f"Available modules:\n\n{modules_block}"
    )
```

Replace with:

```python
    return (
        "You are an intent classifier for a Telegram personal-assistant bot. "
        "Given the user's latest message, choose exactly one module that should "
        'handle it. If no module clearly applies, choose "unknown".\n\n'
        f"Available modules:\n\n{modules_block}\n\n"
        'Respond only with a JSON object of the form {"intent": "<module_name>"}.'
    )
```

- [ ] **Step 2: Update the `with_structured_output` call in `classify_intent`**

Current (`app/orchestrator/router.py:74`):

```python
        classifier = llm.with_structured_output(_build_intent_schema(specs))
```

Replace with:

```python
        classifier = llm.with_structured_output(_build_intent_schema(specs), method="json_mode")
```

No other line in `classify_intent` changes.

- [ ] **Step 3: Manually verify structural correctness (stubbed LLM, no live endpoint needed)**

Run:

```bash
uv run python -c "
import asyncio
from unittest.mock import patch

from langchain_core.messages import HumanMessage

import app.orchestrator.router as router_mod

captured_method = {}

class FakeClassifier:
    def __init__(self, schema):
        self._schema = schema
    async def ainvoke(self, messages):
        sys_content = messages[0]['content']
        assert sys_content.rstrip().endswith('}.'), 'system prompt missing JSON-response instruction'
        return self._schema(intent='search')

class FakeLLM:
    def with_structured_output(self, schema, method=None):
        captured_method['method'] = method
        return FakeClassifier(schema)

async def main():
    state = {
        'messages': [HumanMessage(content='Search for the latest news on AI regulation')],
        'user_id': 'u1', 'intent': '', 'retrieved_memories': [], 'metadata': {},
    }
    with patch.object(router_mod, 'create_llm_client', return_value=FakeLLM()):
        intent = await router_mod.classify_intent(state)
        print(intent)
    print(captured_method['method'])

asyncio.run(main())
"
```

Expected output (two lines):
```
search
json_mode
```

- [ ] **Step 4: Manually verify against the live endpoint (if reachable)**

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.orchestrator.router import classify_intent

async def main():
    for text in ('Search for the latest news on AI regulation', 'Hôm nay tôi cảm thấy khá mệt mỏi'):
        state = {
            'messages': [HumanMessage(content=text)],
            'user_id': 'u1', 'intent': '', 'retrieved_memories': [], 'metadata': {},
        }
        print(await classify_intent(state))

asyncio.run(main())
"
```

Expected output, if the endpoint is reachable and honors `json_mode`: `search` then `journal`. If the endpoint is still unreachable or doesn't honor `json_mode`, the expected output is `unknown` twice, with no traceback — report exactly which of these two outcomes occurred; both are acceptable results for this task (the second confirms the failure path still degrades gracefully, which is the documented, acceptable behavior per the design spec).

- [ ] **Step 5: Do not commit**

Leave `app/orchestrator/router.py` as an uncommitted working-tree change. Do not run `git add` or `git commit`.

## Self-Review Notes

- **Spec coverage:** the design's two-line change (method="json_mode" + JSON-response sentence) is fully covered by Steps 1–2. The design's two testing scenarios (live classification success, or graceful degradation if still unsupported) are covered by Step 4, phrased so either outcome is a valid task result rather than a failure.
- **Placeholder scan:** none.
- **Type/name consistency:** no signature changes anywhere; `_build_system_prompt` and `classify_intent` keep their existing call sites in the file untouched.
