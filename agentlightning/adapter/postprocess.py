# Copyright (c) Microsoft. All rights reserved.

"""Post-process the data to make it more suitable for training."""

from __future__ import annotations

from typing import Sequence

from agentlightning.types.adapter import (
    AccumulatedMessages,
    AccumulatedTokenSequence,
    AdaptingSpan,
    AnnotatedChatCompletionCall,
    BaseAdaptingSequence,
    TokenInputOutputTriplet,
)

from .base import Adapter


class ToTokensTriplets(Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[TokenTriplet]]):
    """Convert adapting spans to token input-output triplets."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[TokenInputOutputTriplet]: ...


class ToTokensAccumulations(Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[TokensAccumulation]]):
    """Assemble multiple token input-output triplets into accumulated token sequences."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[AccumulatedTokenSequence]: ...


class ToPromptCompletionTriplets(Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[PromptCompletionTriplet]]):
    """Convert annotated chat completion calls to prompt-completion triplets."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[PromptCompletionTriplet]: ...


class ToPromptCompletionAccumulations(
    Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[PromptCompletionAccumulation]]
):
    """Assemble multiple prompt-completion triplets into accumulated prompt-completion pairs."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[PromptCompletionAccumulation]: ...
