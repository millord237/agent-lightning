# Copyright (c) Microsoft. All rights reserved.

from typing import List

from agentlightning.adapter.base import Chain
from agentlightning.adapter.call import IdentifyChatCompletionCalls
from agentlightning.adapter.preprocess import RepairMalformedSpans, ToAdaptingSpans
from agentlightning.types.tracer import Span


def test_openai_calls():
    spans: List[Span] = [
        Span.from_attributes(
            attributes={
                "tool.name": "get_rooms_and_availability",
                "tool.parameters": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "tool.call.id": "call_owd6...l9Gv",
                "tool.call.type": "function",
            },
            name="tool_call.get_rooms_and_availability",
            span_id="18edfd18ea659820",
            parent_id="e858708413368c22",
            sequence_id=0,
        ),
        Span.from_attributes(
            attributes={
                "tool.name": "get_rooms_and_availability",
                "tool.parameters": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "tool.call.id": "call_VKsy...5ULX",
                "tool.call.type": "function",
            },
            name="tool_call.get_rooms_and_availability",
            span_id="6c9ba649e512a7f8",
            parent_id="e858708413368c22",
            sequence_id=1,
        ),
        Span.from_attributes(
            attributes={
                "gen_ai.request.type": "chat",
                "gen_ai.system": "OpenAI",
                "gen_ai.request.model": "gpt-4.1-nano",
                "gen_ai.request.temperature": 0.0,
                "gen_ai.request.streaming": False,
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": "You are a scheduling assistant.",
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": "Find a room 2025-10-13 16:30 for 60m, 12 attendees; needs confphone+whiteboard; accessible.",
                "gen_ai.request.functions.0.name": "get_rooms_and_availability",
                "gen_ai.request.functions.0.description": "Return rooms with capacity/equipment/accessibility/bookings.",
                "gen_ai.request.functions.0.parameters": '{"type":"object","properties":{"date":{"type":"string"},"time":{"type":"string"},"duration_min":{"type":"integer"}},"required":["date","time","duration_min"]}',
                "gen_ai.response.id": "chatcmpl-...1jC4",
                "gen_ai.response.model": "gpt-4.1-nano-2025-04-14",
                "gen_ai.openai.system_fingerprint": "fp_03e44fcc34",
                "gen_ai.usage.total_tokens": 211,
                "gen_ai.usage.prompt_tokens": 128,
                "gen_ai.usage.completion_tokens": 83,
                "gen_ai.completion.0.finish_reason": "tool_calls",
                "gen_ai.completion.0.role": "assistant",
            },
            name="openai.chat.completion",
            span_id="e858708413368c22",
            parent_id="3db86425087d211f",
            sequence_id=2,
        ),
        Span.from_attributes(
            attributes={
                "gen_ai.request.type": "chat",
                "gen_ai.system": "OpenAI",
                "gen_ai.request.model": "gpt-4.1-nano",
                "gen_ai.request.temperature": 0.0,
                "gen_ai.request.streaming": False,
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": "You are a scheduling assistant.",
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": "Find a room 2025-10-13 16:30 for 60m, 12 attendees; needs confphone+whiteboard; accessible.",
                "gen_ai.prompt.2.role": "assistant",
                "gen_ai.prompt.2.tool_calls.0.id": "call_owd6...l9Gv",
                "gen_ai.prompt.2.tool_calls.0.name": "get_rooms_and_availability",
                "gen_ai.prompt.2.tool_calls.0.arguments": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "gen_ai.prompt.2.tool_calls.1.id": "call_VKsy...5ULX",
                "gen_ai.prompt.2.tool_calls.1.name": "get_rooms_and_availability",
                "gen_ai.prompt.2.tool_calls.1.arguments": '{"date":"2025-10-13","time":"16:30","duration_min":60}',
                "gen_ai.prompt.3.role": "tool",
                "gen_ai.prompt.3.content": '{"rooms":[{"id":"Nova","capacity":12,"equipment":["whiteboard","confphone"],"accessible":true,"distance_m":45,"booked":[],"free":true},{"id":"Pulse","capacity":8,"equipment":["whiteboard","confphone"],"accessible":true,"booked":[["2025-10-13","16:30",30]],"free":false}]}',
                "gen_ai.prompt.3.tool_call_id": "call_owd6...l9Gv",
                "gen_ai.prompt.4.role": "tool",
                "gen_ai.prompt.4.content": '{"rooms":[{"id":"Nova","capacity":12,"equipment":["whiteboard","confphone"],"accessible":true,"free":true},{"id":"Pulse","capacity":8,"equipment":["whiteboard","confphone"],"accessible":true,"free":false}]}',
                "gen_ai.prompt.4.tool_call_id": "call_VKsy...5ULX",
                "gen_ai.response.id": "chatcmpl-...Syso",
                "gen_ai.response.model": "gpt-4.1-nano-2025-04-14",
                "gen_ai.openai.system_fingerprint": "fp_03e44fcc34",
                "gen_ai.usage.total_tokens": 1189,
                "gen_ai.usage.prompt_tokens": 1082,
                "gen_ai.usage.completion_tokens": 107,
                "gen_ai.completion.0.finish_reason": "stop",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "Available rooms: Nova (cap 12, whiteboard+confphone, accessible); Lyra (cap 10...).",
            },
            name="openai.chat.completion",
            span_id="9a44818e0901d0a1",
            parent_id="3db86425087d211f",
            sequence_id=3,
        ),
        Span.from_attributes(
            attributes={
                "agentops.span.kind": "session",
                "operation.name": "ro-90201d0a24cb",
            },
            name="ro-90201d0a24cb.session",
            span_id="3db86425087d211f",
            parent_id=None,
            sequence_id=4,
        ),
        Span.from_attributes(
            attributes={
                "agentlightning.reward.0.name": "primary",
                "agentlightning.reward.0.value": 1.0,
            },
            name="agentlightning.annotation",
            span_id="dc5e3c27f4378b6e",
            parent_id=None,
            sequence_id=5,
        ),
    ]

    adapter = Chain(
        RepairMalformedSpans(),
        ToAdaptingSpans(),
        IdentifyChatCompletionCalls(),
    )

    adapted_spans = adapter(spans)

    for span in adapted_spans:
        print(span)


test_openai_calls()
