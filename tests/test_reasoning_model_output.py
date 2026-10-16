"""A reasoning model's output must not vanish, and must not look like a dead stream.

MEASURED against the live jetson engine on 2026-09-13:

    POST /vllm-shim/jetson/v1/chat/completions  model=nemotron35
    message keys: ['annotations','audio','content','function_call',
                   'reasoning','refusal','role']
      content    NoneType  None
      reasoning  str       "Here's a thinking process:\\n\\n1. **Analyze User Input:**..."
    finish_reason: length
    usage: {'prompt_tokens': 21, 'completion_tokens': 64}

`nemotron35` is a reasoning model. It puts its generation in `reasoning` and
leaves `content` null. Nothing in this SDK reads that field, which produced two
different-looking production failures from ONE cause, over 7 days:

  * NON-STREAMING — `text = msg.get("content") or ""` yields an empty string, and
    the caller logs `EMPTY RESPONSE from v3_bear_agent (ADBE): raw=''`.
    **87 occurrences across 32 cycles**, 53 of them nemotron35.
  * STREAMING — the loop forwards only `delta["content"]`, so a reasoning model's
    deltas are dropped and the consumer receives NOTHING while the engine is
    happily generating. The caller's watchdog then fires
    `Provider stream stalled: no data received for 300s`.
    **44 occurrences in 7 days**, retried up to 5x = ~25 minutes burned per agent.

The completion is NOT empty — `completion_tokens` is 64. So this is invisible to
the zero-usage truncation guard, which is the repo's usual detector for a lost
generation.

⚠ Reading `reasoning` is a repair, not a cure. `finish_reason: "length"` means
the model spent its whole budget thinking and never reached an answer; the
recovered text is a truncated thought. The point of surfacing it is that
"the model reasoned for 64 tokens and ran out" is a diagnosable fact, while
`raw=''` is not.
"""

from __future__ import annotations

import json

import pytest

from lazycat.llm import LLMStreamWrapper


class TestStreaming:
    @staticmethod
    def _chunks(lines):
        """Drive the real SSE decode path and collect what a consumer sees."""
        import asyncio

        w = LLMStreamWrapper.__new__(LLMStreamWrapper)
        w.is_openai = True
        w._pending_tool_calls = {}

        class _Resp:
            @staticmethod
            async def aiter_text():
                for ln in lines:
                    yield ln + "\n"

        w.response = _Resp()

        async def _drive():
            out = []
            async for ev in w.aiter_lines():
                if not ev.startswith("data: ") or ev == "data: [DONE]":
                    continue
                d = json.loads(ev[6:])
                if d.get("type") == "chunk":
                    out.append(d["content"])
            return out

        return asyncio.run(_drive())

    @staticmethod
    def _line(delta):
        return "data: " + json.dumps({"choices": [{"delta": delta}]})

    def test_a_reasoning_delta_reaches_the_consumer(self):
        """Otherwise the stream looks dead and the 300s watchdog fires."""
        assert self._chunks([self._line({"reasoning": "thinking..."}),
                             "data: [DONE]"]) == ["thinking..."], (
            "a reasoning delta produced no event — the consumer receives "
            "nothing while the engine is generating, and reports a stall"
        )

    def test_a_content_delta_is_unchanged(self):
        assert self._chunks([self._line({"content": "hello"}),
                             "data: [DONE]"]) == ["hello"]

    def test_content_and_reasoning_together_yield_content_once(self):
        assert self._chunks([self._line({"content": "hi", "reasoning": "think"}),
                             "data: [DONE]"]) == ["hi"]

    def test_an_empty_delta_yields_nothing(self):
        assert self._chunks([self._line({}), "data: [DONE]"]) == []
