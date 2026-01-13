# Copyright (c) Microsoft. All rights reserved.

"""Post-process the data to make it more suitable for training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Sequence, TypeVar, Union, cast

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
    TokensAccumulationDiagnosis,
    TokensTriplet,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from .base import Adapter

T_triplet_or_accumulation = TypeVar(
    "T_triplet_or_accumulation",
    bound=Union[TokensTriplet, TokensAccumulation, PromptCompletionTriplet, PromptCompletionAccumulation],
)


class TokensTripletMixin:

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

    def to_triplet(
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

    def to_triplets(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[TokensTriplet]:
        exceptions: List[BaseException] = []
        triplets: List[TokensTriplet] = []
        for span in source:
            if isinstance(span.data, (AnnotatedChatCompletionCall, ChatCompletionCall)):
                triplet = self.to_triplet(span.data)
                if isinstance(triplet, BaseException):
                    exceptions.append(triplet)
                else:
                    triplets.append(triplet)
        if len(triplets) == 0:
            error_msg = (
                f"{self.__class__.__name__} failed to create any triplets. "
                f"The adapter has raised {len(exceptions)} exceptions when processing the spans:\n"
                + "\n".join([f"  - {exc}" for exc in exceptions])
            )
            raise RuntimeError(error_msg)
        triplets[-1] = triplets[-1].model_copy(update={"done": True})
        return triplets


class ToTokensTriplets(TokensTripletMixin, Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[TokensTriplet]]):
    """Convert adapting spans to token input-output triplets.

    Args:
        strict: Whether to raise an exception if the triplet cannot be created.
            If False, the exception will be added to the list of exceptions and the triplet will be skipped.
            The exceptions will also be raised when the resulting sequence is empty.
            If True, the exception will be raised.
            Default is False.
    """

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[TokensTriplet]:
        return self.to_triplets(source)


class ToTokensAccumulations(
    TokensTripletMixin, Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[TokensAccumulation]]
):
    """Assemble multiple token input-output triplets into accumulated token sequences.

    Args:
        diagnosis: Whether to include diagnosis information in the resulting TokensAccumulation.
        strict: Whether to raise an exception if the triplet cannot be created.
            If False, the exception will be added to the list of exceptions and the triplet will be skipped.
            The exceptions will also be raised when the resulting sequence is empty.
            If True, the exception will be raised.
            Default is False.
        tokenizer: An optional tokenizer to decode token IDs to text for diagnosis.
    """

    def __init__(self, strict: bool = False, diagnosis: bool = False, tokenizer: Optional[PreTrainedTokenizer] = None):
        super().__init__(strict=strict)
        self.diagnosis = diagnosis
        self.tokenizer = tokenizer

    def _triplet_to_accumulation(
        self, triplet: TokensTriplet, diagnosis_info: Optional[TokensAccumulationDiagnosis]
    ) -> TokensAccumulation:
        if triplet.completion.logprobs is not None:
            logprobs = [0.0] * len(triplet.observation.token_ids) + list(triplet.completion.logprobs)
        else:
            logprobs = None

        return TokensAccumulation(
            token_ids=[*triplet.observation.token_ids, *triplet.completion.token_ids],
            image_urls=triplet.observation.image_urls,
            logprobs=logprobs,
            response_mask=[0] * len(triplet.observation.token_ids) + [1] * len(triplet.completion.token_ids),
            final_reward=triplet.reward,
            raw_calls=[triplet.raw_call],
            diagnosis_info=diagnosis_info,
        )

    def _special_token_sequence(self, ids: Sequence[int]) -> List[int]:
        assert self.tokenizer is not None, "Tokenizer must be provided for special token sequence extraction."
        return [id for id in ids if id in self.tokenizer.all_special_ids]

    def _non_special_token_sequence(self, ids: Sequence[int]) -> List[int]:
        assert self.tokenizer is not None, "Tokenizer must be provided for non-special token sequence extraction."
        return [id for id in ids if id not in self.tokenizer.all_special_ids]

    def _diagnose_mismatch(
        self, prev: TokensAccumulation, next: TokensTriplet
    ) -> Optional[TokensAccumulationDiagnosis]:
        if not self.diagnosis:
            return None

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for diagnosis.")

        image_urls_match = self.is_prefix(prev.image_urls, next.observation.image_urls)

        # Check whether the special tokens match
        next_special_ids = self._special_token_sequence(next.observation.token_ids)
        prev_special_ids = self._special_token_sequence(prev.token_ids)
        special_tokens_match = self.is_prefix(prev_special_ids, next_special_ids)

        # Check whether the non-special tokens match
        next_non_special_ids = self._non_special_token_sequence(next.observation.token_ids)
        prev_non_special_ids = self._non_special_token_sequence(prev.token_ids)
        non_special_tokens_match = self.is_prefix(prev_non_special_ids, next_non_special_ids)

        # Check whether the detokenized text matches
        next_string = self.tokenizer.decode(next.observation.token_ids, skip_special_tokens=True)  # type: ignore
        prev_string = self.tokenizer.decode(prev.token_ids, skip_special_tokens=True)  # type: ignore
        detokenized_text_match = next_string.startswith(prev_string)

        return TokensAccumulationDiagnosis(
            special_tokens_mismatch=not special_tokens_match,
            non_special_tokens_mismatch=not non_special_tokens_match,
            detokenized_text_mismatch=not detokenized_text_match,
            image_urls_mismatch=not image_urls_match,
            accumulation_prev=prev,
            special_tokens_prev=prev_special_ids,
            special_tokens_next=next_special_ids,
            detokenized_text_prev=prev_string,
            detokenized_text_next=next_string,
        )

    @staticmethod
    def is_prefix(shorter: Iterable[Any], longer: Iterable[Any]) -> bool:
        """Check if the shorter sequence is a prefix of the longer sequence."""
        longer_iter = iter(longer)
        for item in shorter:
            try:
                expected_item = next(longer_iter)
                if item != expected_item:
                    return False
            except StopIteration:
                return False
        return True

    def _attempt_to_merge(self, prev: TokensAccumulation, next: TokensTriplet) -> List[TokensAccumulation]:
        # Check if we can merge the next triplet into the previous accumulation
        if not self.is_prefix(prev.image_urls, next.observation.image_urls):
            return [prev, self._triplet_to_accumulation(next, self._diagnose_mismatch(prev, next))]
        # Merge token ids
        if not self.is_prefix(prev.token_ids, next.observation.token_ids):
            return [prev, self._triplet_to_accumulation(next, self._diagnose_mismatch(prev, next))]
        tokens_to_add = [*next.observation.token_ids[len(prev.token_ids) :], *next.completion.token_ids]
        if prev.logprobs is not None and next.completion.logprobs is not None:
            new_logprobs = list(prev.logprobs) + [0.0] * len(tokens_to_add) + list(next.completion.logprobs)
        else:
            new_logprobs = None
        response_mask_to_add = [0] * (len(next.observation.token_ids) - len(prev.token_ids)) + [1] * len(
            next.completion.token_ids
        )
        new_reward = (
            (prev.final_reward or 0.0) + (next.reward or 0.0)
            if next.reward is not None or prev.final_reward is not None
            else None
        )

        return [
            TokensAccumulation(
                token_ids=[*prev.token_ids, *tokens_to_add],
                image_urls=next.observation.image_urls,
                logprobs=new_logprobs,
                response_mask=[*prev.response_mask, *response_mask_to_add],
                final_reward=new_reward,
                raw_calls=[*prev.raw_calls, next.raw_call],
                diagnosis_info=None,
            )
        ]

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[TokensAccumulation]:
        triplets = self.to_triplets(source)

        accumulations: List[TokensAccumulation] = []
        for triplet in triplets:
            if not accumulations:
                accumulations.append(self._triplet_to_accumulation(triplet, None))
            else:
                last_accumulation = accumulations[-1]
                accumulations = accumulations[:-1] + self._attempt_to_merge(last_accumulation, triplet)
        return accumulations


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
