# Router structured output: switch to json_mode

Date: 2026-07-14
Status: Approved (resolved via direct user confirmation — see below)

## Problem

`classify_intent()` in `app/orchestrator/router.py` calls `llm.with_structured_output(schema)` with no `method` argument. LangChain's default for `ChatOpenAI` is `method="function_calling"`, which requires the backend to correctly honor OpenAI's tool-calling / `tool_choice` contract. Live testing against the user's configured proxy (`.env`'s `LLM_BASE_URL`, model `gemini`) showed the endpoint is now reachable, but the model responds with plain conversational text (`"unknown How can I help you today?"`) instead of the required JSON — the proxy/model does not honor `tool_choice`. `classify_intent`'s broad `except Exception` correctly catches the resulting Pydantic validation error and falls back to `"unknown"`, so the bot doesn't crash, but real classification never succeeds.

## Goal

Change the structured-output method from the default (`function_calling`) to `json_mode`, which asks the model for a JSON-formatted response via `response_format` instead of tool-calling — many OpenAI-compatible proxies support this even when they don't properly implement `tool_choice`.

## Non-goals

- Does not change the schema (`_build_intent_schema`), the prompt (`_build_system_prompt`), the module-discovery logic (`_load_module_specs`), or the failure-handling boundary (`except Exception` → `"unknown"`) — all unchanged, all already reviewed.
- Does not touch `app/infra/providers/*`.
- Does not guarantee the fix works against the user's specific proxy/model — `json_mode` is a different request shape that many more OpenAI-compatible backends support than full tool-calling, but it is not universal. If it still fails, the failure path is unchanged (falls back to `"unknown"`, no crash) and further investigation would be a separate follow-up.

## Design

One-line change in `app/orchestrator/router.py`, inside `classify_intent()`:

```python
# Before
classifier = llm.with_structured_output(_build_intent_schema(specs))

# After
classifier = llm.with_structured_output(_build_intent_schema(specs), method="json_mode")
```

`json_mode` requires the prompt to explicitly instruct the model to respond in JSON (a general requirement of OpenAI-style JSON mode, independent of this codebase) — `_build_system_prompt()`'s existing instruction to "choose exactly one module" is not itself a JSON-formatting instruction. Add one sentence to the end of `_build_system_prompt()`'s returned string so the model knows the expected output shape:

```python
# Before (end of _build_system_prompt's return statement)
return (
    "You are an intent classifier for a Telegram personal-assistant bot. "
    "Given the user's latest message, choose exactly one module that should "
    'handle it. If no module clearly applies, choose "unknown".\n\n'
    f"Available modules:\n\n{modules_block}"
)

# After
return (
    "You are an intent classifier for a Telegram personal-assistant bot. "
    "Given the user's latest message, choose exactly one module that should "
    'handle it. If no module clearly applies, choose "unknown".\n\n'
    f"Available modules:\n\n{modules_block}\n\n"
    'Respond only with a JSON object of the form {"intent": "<module_name>"}.'
)
```

## Testing

Manual verification (no pytest configured, unchanged constraint):
1. Run the same `classify_intent()` live-LLM check from the prior plan (search message → expect `"search"`; journal message → expect `"journal"`) against the user's now-reachable endpoint, and confirm no `intent_classification_failed` warning is logged.
2. If the endpoint still fails to produce valid JSON under `json_mode`, confirm the failure path is unchanged: `classify_intent()` still returns `"unknown"` without raising, and `_dispatch()` still replies with the friendly fallback message — this is the documented, acceptable degradation, not a new bug to fix in this change.
