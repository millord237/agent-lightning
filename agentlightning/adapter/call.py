# Copyright (c) Microsoft. All rights reserved.

"""Handles chat completion / response calls. Extracts them from spans and annotates them with annotations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeGuard, Union, cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic.type_adapter import TypeAdapter

from agentlightning.types.adapter import AnnotatedChatCompletionCall, Annotation, ChatCompletionCall, Tree
from agentlightning.types.tracer import Span
from agentlightning.utils.otel import filter_and_unflatten_attributes

from .base import Adapter


class IdentifyChatCompletionCalls(Adapter[Sequence[Span], Sequence[ChatCompletionCall]]):
    """Curate the chat completion calls from the spans."""

    def _validate_metadata(self, metadata: Any) -> TypeGuard[Dict[str, Any]]:
        if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata.keys()):  # type: ignore
            return False
        return True

    def _validate_completion(self, completion: Any) -> TypeGuard[List[Dict[str, Any]]]:
        if not isinstance(completion, list):
            return False
        for choice in cast(List[Any], completion):
            if not isinstance(choice, dict):
                return False
            if "message" not in choice or not isinstance(choice["message"], dict):
                return False
        return True

    def _parse_agentops_tool_calls(self, span: Span) -> Optional[ChatCompletionMessageFunctionToolCall]:
        if span.name.startswith("tool_call."):
            tool_call_data = filter_and_unflatten_attributes(span.attributes, "tool.")
            if isinstance(tool_call_data, dict) and "call" in tool_call_data:
                # Example tool_call_data:
                # {'tool.name': 'get_rooms', 'tool.parameters': '{"date": ...}',
                #  'tool.call.id': 'call_owd6', 'tool.call.type': 'function'}
                tool_call_data = {
                    **tool_call_data["call"],
                    "function": {k: v for k, v in tool_call_data.items() if k != "call"},
                }
            return ChatCompletionMessageFunctionToolCall.model_validate(tool_call_data)
        return None

    def _parse_openai_chat_completion_create(self, span: Union[Span, Tree[Span]]) -> ChatCompletionCall:
        core_content = span.attributes if isinstance(span, Span) else span.item.attributes
        prompt_messages = filter_and_unflatten_attributes(core_content, "gen_ai.prompt")
        request_metadata = filter_and_unflatten_attributes(core_content, "gen_ai.request")
        completion_choices = filter_and_unflatten_attributes(core_content, "gen_ai.completion")
        usages = filter_and_unflatten_attributes(core_content, "gen_ai.usage")
        response_metadata = filter_and_unflatten_attributes(core_content, "gen_ai.response")

        if not self._validate_metadata(request_metadata):
            raise ValueError(f"Invalid request metadata format in span attributes: {request_metadata}")
        if not self._validate_metadata(response_metadata):
            raise ValueError(f"Invalid response metadata format in span attributes: {response_metadata}")
        if not self._validate_completion(completion_choices):
            raise ValueError(
                "Invalid completion choices format in span attributes. Must be a list of dict, "
                f"each containing a 'message' dict: {completion_choices}"
            )

        request_body = cast(
            CompletionCreateParams,
            TypeAdapter(CompletionCreateParams).validate_python({"messages": prompt_messages, **request_metadata}),
        )

        if isinstance(span, Tree):
            # Get additional tool calls from child spans
            additional_tool_calls: List[ChatCompletionMessageFunctionToolCall] = []
            for child in span.children:
                tool_call = self._parse_agentops_tool_calls(child.item)
                if tool_call is not None:
                    additional_tool_calls.append(tool_call)

            for choice in completion_choices:
                tool_calls = choice["message"].get("tool_calls", [])
                if isinstance(tool_calls, list):
                    cast(List[Any], tool_calls).extend(additional_tool_calls)
                else:
                    raise ValueError(
                        f"Invalid tool_calls format in completion choice message. Must be a list: {completion_choices}"
                    )
                choice["message"]["tool_calls"] = tool_calls
                # Only assign to the first choice.
                break

        return ChatCompletionCall(
            request=request_body,
            response=ChatCompletion.model_validate(
                {
                    **response_metadata,
                    "choices": completion_choices,
                    "usage": usages,
                }
            ),
            malformed_fields={},  # TODO: malformed fields
        )

    def _parse_litellm_request(self, span: Union[Span, Tree[Span]]) -> ChatCompletionCall: ...

    def adapt(self, source: Sequence[Span]) -> Sequence[ChatCompletionCall]: ...


class AnnotateChatCompletionCalls(
    Adapter[Tuple[Sequence[ChatCompletionCall], Sequence[Annotation]], Sequence[AnnotatedChatCompletionCall]]
):
    """Annotate chat completion calls with the given annotations.

    The intersection of "effective radius" of annotations and chat completion calls is used to determine
    which annotations apply to which chat completion calls.

    If an annotation is not linked to any span, try to use `RepairMissingLinks` first to link it to spans.
    """

    def adapt(
        self,
        source: Tuple[Sequence[ChatCompletionCall], Sequence[Annotation]],
    ) -> Sequence[AnnotatedChatCompletionCall]: ...
