# Copyright (c) Microsoft. All rights reserved.

"""Post-process the data to make it more suitable for training."""

from __future__ import annotations

from typing import Literal, Sequence, TypeVar, Union

from agentlightning.types.adapter import (
    AdaptingSpan,
    BaseAdaptingSequence,
    PromptCompletionAccumulation,
    PromptCompletionTriplet,
    TokensAccumulation,
    TokensTriplet,
)

from .base import Adapter

T_triplet_or_accumulation = TypeVar(
    "T_triplet_or_accumulation",
    bound=Union[TokensTriplet, TokensAccumulation, PromptCompletionTriplet, PromptCompletionAccumulation],
)


class ToTokensTriplets(Adapter[BaseAdaptingSequence[AdaptingSpan], Sequence[TokensTriplet]]):
    """Convert adapting spans to token input-output triplets."""

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Sequence[TokensTriplet]: ...


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
