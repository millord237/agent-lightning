# Copyright (c) Microsoft. All rights reserved.

"""Handles chat completion / response calls. Extracts them from spans and annotates them with annotations."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypeGuard, Union, cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic import TypeAdapter

from agentlightning.types.adapter import (
    AdaptingSpan,
    AnnotatedChatCompletionCall,
    Annotation,
    BaseAdaptingSequence,
    ChatCompletionCall,
    Tree,
)
from agentlightning.types.tracer import Span
from agentlightning.utils.otel import filter_and_unflatten_attributes, query_linked_spans

from .base import SequenceAdapter


class IdentifyChatCompletionCalls(SequenceAdapter[AdaptingSpan, AdaptingSpan]):
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

    def _parse_openai_chat_completion_create(self, span: AdaptingSpan) -> ChatCompletionCall:
        prompt_messages = filter_and_unflatten_attributes(span.attributes, "gen_ai.prompt")
        request_metadata = filter_and_unflatten_attributes(span.attributes, "gen_ai.request")
        completion_choices = filter_and_unflatten_attributes(span.attributes, "gen_ai.completion")
        usages = filter_and_unflatten_attributes(span.attributes, "gen_ai.usage")
        response_metadata = filter_and_unflatten_attributes(span.attributes, "gen_ai.response")

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

        if isinstance(span.container, Tree):
            # Get additional tool calls from child spans
            additional_tool_calls: List[ChatCompletionMessageFunctionToolCall] = []
            for child in span.children():
                tool_call = self._parse_agentops_tool_calls(child)
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

    def adapt_one(self, source: AdaptingSpan) -> AdaptingSpan:
        if source.name == "openai.chat.completion.create" or source.name == "openai.chat.completion":
            chat_completion_call = self._parse_openai_chat_completion_create(source)
            return source.with_data(chat_completion_call)
        elif source.name == "raw_gen_ai_request":
            # Litellm request span
            chat_completion_call = self._parse_litellm_request(source)
            return source.with_data(chat_completion_call)
        else:
            # Not a chat completion call span. Do nothing
            return source


class AnnotateChatCompletionCalls(SequenceAdapter[AdaptingSpan, AdaptingSpan]):
    """Annotate chat completion calls with the given annotations.

    The intersection of "effective radius" of annotations and chat completion calls is used to determine
    which annotations apply to which chat completion calls.

    If an annotation is not linked to any span, try to use `RepairMissingLinks` first to link it to spans.
    """

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> BaseAdaptingSequence[AdaptingSpan]:
        annotation_spans = [span for span in source if isinstance(span.data, Annotation)]
        span_id_to_updated_annotation: Dict[str, AnnotatedChatCompletionCall] = {}
        for annotation_span in annotation_spans:
            annotation = cast(Annotation, annotation_span.data)
            for linked_span in query_linked_spans(source, annotation.links):
                if isinstance(linked_span.container, Tree):
                    linked_spans = list(linked_span.container.traverse())
                else:
                    linked_spans = [linked_span]

                for linked_span in linked_spans:
                    if isinstance(linked_span.data, ChatCompletionCall):
                        existing_annotations: List[Annotation] = (
                            list(linked_span.data.annotations)
                            if isinstance(linked_span.data, AnnotatedChatCompletionCall)
                            else []
                        )
                        # Annotate the chat completion call
                        annotated_call = AnnotatedChatCompletionCall(
                            request=linked_span.data.request,
                            response=linked_span.data.response,
                            malformed_fields=linked_span.data.malformed_fields,
                            annotations=existing_annotations + [annotation],
                        )
                        # Update the linked span with the annotated call
                        span_id_to_updated_annotation[linked_span.span_id] = annotated_call

        def _update_span(span: AdaptingSpan) -> AdaptingSpan:
            if span.span_id in span_id_to_updated_annotation:
                annotated_call = span_id_to_updated_annotation[span.span_id]
                return span.with_data(annotated_call, override="silent")  # override is expected here
            else:
                return span

        return source.map(_update_span)
