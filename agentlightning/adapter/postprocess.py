# Copyright (c) Microsoft. All rights reserved.

"""Post-process the data to make it more suitable for training."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence, TypeVar, Union, cast

from agentlightning.types.adapter import (
    AdaptingSpan,
    AnnotatedChatCompletionCall,
    BaseAdaptingSequence,
    ChatCompletionCall,
    GeneralAnnotation,
    PromptCompletionAccumulation,
    PromptCompletionTriplet,
    TokenInput,
    TokenOutput,
    TokensAccumulation,
    TokensTriplet,
)

from .base import Adapter

T_triplet_or_accumulation = TypeVar(
    "T_triplet_or_accumulation",
    bound=Union[TokensTriplet, TokensAccumulation, PromptCompletionTriplet, PromptCompletionAccumulation],
)


class ToTokensTriplets(Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[TokensTriplet]]):
    """Convert adapting spans to token input-output triplets.

    Args:
        strict: Whether to raise an exception if the triplet cannot be created.
            If False, the exception will be added to the list of exceptions and the triplet will be skipped.
            The exceptions will also be raised when the resulting sequence is empty.
            If True, the exception will be raised.
            Default is False.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    def _get_prompt_token_ids(self, call: Union[AnnotatedChatCompletionCall, ChatCompletionCall]) -> Sequence[int]:
        prompt = call.request
        if "prompt_token_ids" in prompt:
            prompt_token_ids = prompt["prompt_token_ids"]
        else:
            raise ValueError(f"Prompt token ids not found in call: {call}")
        # Validate the prompt token ids
        if not isinstance(prompt_token_ids, list) or not all(isinstance(x, int) for x in prompt_token_ids):  # type: ignore
            raise ValueError(f"Invalid prompt token ids. Must be a list of ints. Got: {prompt_token_ids}")
        if len(prompt_token_ids) == 0:
            raise ValueError("Prompt token ids is empty.")
        return prompt_token_ids

    def _get_image_urls(self, call: Union[AnnotatedChatCompletionCall, ChatCompletionCall]) -> Sequence[str]:
        image_urls: List[str] = []
        for message in call.request["messages"]:
            if "content" not in message:
                continue
            content = message["content"]
            if content is None:
                continue
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "image_url":
                        image_urls.append(part["image_url"]["url"])
        return image_urls

    def _get_completion_token_ids(self, call: Union[AnnotatedChatCompletionCall, ChatCompletionCall]) -> Sequence[int]:
        completion_choice = call.response.choices[0]
        if hasattr(completion_choice, "token_ids") and completion_choice.token_ids is not None:  # type: ignore
            response_token_ids = cast(Any, completion_choice.token_ids)  # type: ignore
        elif hasattr(completion_choice, "provider_specific_fields") and "token_ids" in completion_choice.provider_specific_fields:  # type: ignore
            response_token_ids = cast(Any, completion_choice.provider_specific_fields["token_ids"])  # type: ignore
        else:
            raise ValueError(f"Completion token ids not found in call: {call}")
        if not isinstance(response_token_ids, list) or not all(isinstance(x, int) for x in response_token_ids):  # type: ignore
            raise ValueError(f"Invalid completion token ids. Must be a list of ints. Got: {response_token_ids}")
        response_token_ids = cast(Sequence[int], response_token_ids)
        if len(response_token_ids) == 0:
            raise ValueError("Completion token ids is empty.")
        return response_token_ids

    def _get_logprobs(self, call: Union[AnnotatedChatCompletionCall, ChatCompletionCall]) -> Optional[Sequence[float]]:
        logprobs = call.response.choices[0].logprobs
        if logprobs is not None:
            content_logprobs = logprobs.content
            if content_logprobs is not None:
                return [logprob.logprob for logprob in content_logprobs]
        return None

    def _get_reward(self, call: Union[AnnotatedChatCompletionCall, ChatCompletionCall]) -> Optional[float]:
        if isinstance(call, AnnotatedChatCompletionCall):
            for annotation in call.annotations:
                # The first general annotation
                if isinstance(annotation, GeneralAnnotation):
                    # The primary reward
                    return annotation.primary_reward
        return None

    def _to_tokens_triplet(
        self, call: Union[AnnotatedChatCompletionCall, ChatCompletionCall]
    ) -> Union[TokensTriplet, BaseException]:
        try:
            return TokensTriplet(
                observation=TokenInput(
                    token_ids=self._get_prompt_token_ids(call), image_urls=self._get_image_urls(call)
                ),
                completion=TokenOutput(
                    token_ids=self._get_completion_token_ids(call), logprobs=self._get_logprobs(call)
                ),
                reward=self._get_reward(call),
                done=False,  # False by now
                raw_call=call,
            )
        except Exception as exc:
            if self.strict:
                raise exc
            return exc

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[TokensTriplet]:
        exceptions: List[BaseException] = []
        triplets: List[TokensTriplet] = []
        for span in source:
            if isinstance(span.data, (AnnotatedChatCompletionCall, ChatCompletionCall)):
                triplet = self._to_tokens_triplet(span.data)
                if isinstance(triplet, BaseException):
                    exceptions.append(triplet)
                else:
                    triplets.append(triplet)
        if len(triplets) == 0:
            error_msg = (
                f"{self.__class__.__name__} failed to create any triplets. "
                "The adapter has raised {len(exceptions)} exceptions when processing the spans:\n"
                + "\n".join([f"  - {exc}" for exc in exceptions])
            )
            raise RuntimeError(error_msg)
        return triplets


class ToTokensAccumulations(Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[TokensAccumulation]]):
    """Assemble multiple token input-output triplets into accumulated token sequences."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[TokensAccumulation]: ...


class ToPromptCompletionTriplets(Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[PromptCompletionTriplet]]):
    """Convert annotated chat completion calls to prompt-completion triplets."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[PromptCompletionTriplet]: ...


class ToPromptCompletionAccumulations(
    Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[PromptCompletionAccumulation]]
):
    """Assemble multiple prompt-completion triplets into accumulated prompt-completion pairs."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[PromptCompletionAccumulation]: ...


class PropagateRewards(Adapter[Sequence[T_triplet_or_accumulation], Sequence[T_triplet_or_accumulation]]):
    """Propagate rewards forward or backward from one triplet or accumulation to the next."""

    def __init__(self, direction: Literal["forward", "backward"]) -> None:
        self.direction = direction

    def adapt(self, source: Sequence[T_triplet_or_accumulation]) -> Sequence[T_triplet_or_accumulation]: ...
