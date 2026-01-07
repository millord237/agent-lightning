# Copyright (c) Microsoft. All rights reserved.

"""Data formats used by adapters, usually the target format converted from trace spans."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic import BaseModel, Field

from agentlightning.semconv import LinkPydanticModel, RewardPydanticModel

from .tracer import Attributes, Span

T = TypeVar("T")
V = TypeVar("V")


class BaseAdaptingSequence(Sequence[T], Generic[T]):
    """Interface that makes adapter easier to work with sequences."""

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        return self.get(index)

    def __iter__(self) -> Iterator[T]:
        return iter(self.traverse())

    def __len__(self) -> int:
        return self.size()

    def get(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        """Get the index-th item in the sequence."""
        raise NotImplementedError()

    def map(self, func: Callable[[T], V]) -> BaseAdaptingSequence[V]:
        """Map a function over all items in the sequence."""
        raise NotImplementedError()

    def retain(self, predicate: Callable[[T], bool]) -> BaseAdaptingSequence[T]:
        """Filter items in the sequence by a predicate (true for items to be kept).

        Depending on the implementation, the returned sequence may contain more or less items than a standard filter.
        """
        raise NotImplementedError()

    def prune(self, predicate: Callable[[T], bool]) -> BaseAdaptingSequence[T]:
        """Prune items in the sequence by a predicate (true for items to be pruned).

        Depending on the implementation, the returned sequence may contain more or less items than a standard prune.
        """
        raise NotImplementedError()

    def size(self) -> int:
        """Get the size of the sequence."""
        raise NotImplementedError()

    def traverse(self) -> Iterable[T]:
        """Traverse all items in the sequence."""
        raise NotImplementedError()


# General containers


class Tree(BaseAdaptingSequence[T], Generic[T]):
    """This is a generic tree data structure that can be used to represent the structure of a tree."""

    def __init__(self, item: T, children: MutableSequence[Tree[T]]) -> None:
        self._item = item
        self._children = children

    @property
    def item(self) -> T:
        return self._item

    @property
    def children(self) -> Sequence[Tree[T]]:
        return self._children

    def traverse(self) -> Iterable[T]:
        yield self._item
        for child in self._children:
            yield from child.traverse()

    def size(self) -> int:
        return 1 + sum(child.size() for child in self._children)

    def get(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        """Get the index-th item in the tree (O(n) time complexity).

        I think this is not efficient, but it's seldomly used.
        """
        return list(self.traverse())[index]

    def add(self, child: Tree[T]) -> None:
        self._children.append(child)

    def map(self, func: Callable[[T], V]) -> Tree[V]:
        """Map a function over all items in the tree."""
        return Tree(func(self._item), [child.map(func) for child in self._children])

    def _retain_subtree(self, predicate: Callable[[T], bool]) -> Optional[Tree[T]]:
        if predicate(self._item):
            # If the current node satisfies the predicate, retain the subtree
            return self

        subtrees = [child._retain_subtree(predicate) for child in self._children]
        if all(subtree is None for subtree in subtrees):
            # no subtrees satisfy the predicate, remove the current node
            return None

        return Tree(self._item, [subtree for subtree in subtrees if subtree is not None])

    def retain(self, predicate: Callable[[T], bool]) -> Tree[T]:
        """Prune the tree by retaining subtrees with root nodes that satisfy the predicate.

        The root node is always retained.
        """
        return self._retain_subtree(predicate) or Tree(self._item, [])

    def prune(self, predicate: Callable[[T], bool]) -> Tree[T]:
        """Prune the tree by removing nodes that satisfy the predicate.

        The root node is always retained.
        """
        return Tree(self._item, [child.prune(predicate) for child in self._children if not predicate(child._item)])

    def visualize(self, filename: str, item_to_str: Callable[[T], str]) -> None:
        """Render the tree with Graphviz for debugging purposes.

        Args:
            filename: Base filename for the generated `.png` diagram (without extension).

        !!! note

            The method requires the optional `graphviz` dependency to be available in the runtime
            environment.
        """
        import graphviz

        dot = graphviz.Digraph(comment="Tree")

        def visit(node: Tree[T]):
            dot.node(str(id(node)), item_to_str(node.item))  # type: ignore
            for child in node._children:
                visit(child)
                dot.edge(str(id(node)), str(id(child)))  # type: ignore

        visit(self)
        dot.render(filename, format="png", cleanup=True)  # type: ignore


class AdaptingSequence(BaseAdaptingSequence[T], Generic[T]):
    """A simple list implementation of AdaptingSequence."""

    def __init__(self, items: Sequence[T]) -> None:
        self._items = list(items)

    def get(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        return self._items[index]

    def traverse(self) -> Iterable[T]:
        return iter(self._items)

    def size(self) -> int:
        return len(self._items)

    def map(self, func: Callable[[T], V]) -> AdaptingSequence[V]:
        return AdaptingSequence([func(item) for item in self._items])

    def retain(self, predicate: Callable[[T], bool]) -> AdaptingSequence[T]:
        return AdaptingSequence([item for item in self._items if predicate(item)])

    def prune(self, predicate: Callable[[T], bool]) -> AdaptingSequence[T]:
        return AdaptingSequence([item for item in self._items if not predicate(item)])


class AdaptingSpan(Span):
    """A span that has been adapted to a different format.

    This class extends the base [`Span`][agentlightning.Span] class to represent spans that have
    been converted to a different format by an adapter.
    """

    data: Any
    """The data in the adapted format. Could be annotations, calls, or other structured data."""

    @classmethod
    def from_span(cls, span: Span, data: Any) -> AdaptingSpan:
        """Create an [`AdaptingSpan`][agentlightning.AdaptingSpan] from a base [`Span`][agentlightning.Span].

        Args:
            span: The base span to convert.
            data: The data in the adapted format.

        Returns:
            An instance of [`AdaptingSpan`][agentlightning.AdaptingSpan] with the same properties as
            the input span and the provided adapted data.
        """
        if isinstance(span, AdaptingSpan):
            return span.model_copy(update={"data": data})
        else:
            return AdaptingSpan.model_validate(span.model_dump()).model_copy(update={"data": data})


# Annotation-related types


class Annotation(BaseModel):
    """An annotation is an approach to parse a span into some kind of structured attachments to another object.

    Not necessarily an [AGL_ANNOTATION][agentlightning.semconv.AGL_ANNOTATION] span.

    Note that a span can be parsed in multiple ways, and annotation is just one of them.
    """

    annotation_type: Literal["agent", "general", "message", "object", "exception", "operation"]
    """Type of the annotation."""

    links: Sequence[LinkPydanticModel] = Field(default_factory=list[LinkPydanticModel])
    """Links to other spans or objects."""


class AgentAnnotation(Annotation):
    """Parsed from [OTel Agent Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)."""

    annotation_type = "agent"
    """Type of the annotation."""

    id: Optional[str] = None
    """The unique identifier of the GenAI agent."""

    name: Optional[str] = None
    """Human-readable name of the GenAI agent provided by the application."""

    description: Optional[str] = None
    """Free-form description of the GenAI agent provided by the application."""


class GeneralAnnotation(Annotation):
    """An annotation payload that is parsed from an [annotation][agentlightning.semconv.AGL_ANNOTATION] span."""

    annotation_type = "general"
    """Type of the annotation."""

    rewards: Sequence[RewardPydanticModel] = Field(default_factory=list[RewardPydanticModel])
    """Reward dimensions and values."""

    primary_reward: Optional[float] = None
    """Primary reward value."""

    tags: Sequence[str] = Field(default_factory=list)
    """Tags for the annotation."""

    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    """Raw payload from the annotation."""


class MessageAnnotation(Annotation):
    """A log message that is parsed from a [message][agentlightning.semconv.AGL_MESSAGE] span."""

    annotation_type = "message"
    """Type of the annotation."""

    message: str
    """Message text."""


class ObjectAnnotation(Annotation):
    """An artifact that is parsed from a [object][agentlightning.semconv.AGL_OBJECT] span."""

    annotation_type = "object"
    """Type of the annotation."""

    object: Any
    """The object payload."""


class ExceptionAnnotation(Annotation):
    """An exception that is parsed from an [exception][agentlightning.semconv.AGL_EXCEPTION] span."""

    annotation_type = "exception"
    """Type of the annotation."""

    type: str
    """Type of the exception."""

    message: str
    """Message of the exception."""

    stacktrace: Optional[str] = None
    """Stacktrace of the exception."""


class OperationAnnotation(Annotation):
    """An operation that is parsed from an [operation][agentlightning.semconv.AGL_OPERATION] span."""

    annotation_type = "operation"
    """Type of the annotation."""

    name: str
    """Name of the operation."""

    input: Optional[Any] = None
    """Input of the operation."""

    output: Optional[Any] = None
    """Output of the operation."""


class ChatCompletionCall(BaseModel):
    """Corresponding to exactly one chat completion call.

    OpenAI chat completion request and response are used as standards here.
    Convert to other chat completion formats if needed.
    """

    request: CompletionCreateParams
    """OpenAI chat completion request parameters."""

    response: ChatCompletion
    """OpenAI chat completion response payload."""

    malformed_fields: Dict[str, Attributes]
    """Fields that are not supported by the adapter.

    Mapping from span names to a dict of malformed fields.
    """


class AnnotatedChatCompletionCall(ChatCompletionCall):
    """A chat completion call with annotations."""

    annotations: Sequence[Annotation]
    """Annotations for the chat completion call."""


# Algorithm-specific requirements


class TokenInput(BaseModel):
    """Token-based model input."""

    token_ids: Sequence[int]
    """Token IDs of the model input."""

    image_urls: Any
    """A list of image URLs. Could be pointers to local files or base64-encoded images."""


class TokenOutput(BaseModel):
    """Token-based model output."""

    token_ids: Sequence[int]
    """Token IDs of the model output."""


class TokenInputOutputTriplet(BaseModel):
    """A triplet of token IDs for the input and output, useful for reinforcement learning.

    This is not a stable interface and the fields here highly depend on RL implementations.
    """

    observation: TokenInput
    """Observation for the model input. Corresponding to prompt."""

    action: TokenOutput
    """Action, corresponding to completion result."""

    reward: Optional[float]
    """Reward of the model input."""

    done: bool
    """Whether it's the end of the trajectory."""

    raw_call: AnnotatedChatCompletionCall
    """Raw chat completion call."""


class AccumulatedTokenSequence(TokenInput):
    """A sequence of token IDs that are accumulated from multiple model calls.

    Output is implied in the token IDs.
    """

    response_mask: Sequence[int]
    """Mask for the response tokens. Must a sequence of 0s and 1s, with 1s for the completion tokens and 0s for the prompt tokens."""

    final_reward: Optional[float]
    """Single reward value for the entire sequence."""

    raw_calls: Sequence[AnnotatedChatCompletionCall]
    """Raw chat completion calls. The order of the calls must be the same as the order of the token IDs."""


class AccumulatedMessages(BaseModel):
    """A conversation that is accumulated from multiple model calls."""

    messages: Sequence[ChatCompletionMessageParam]
    """Messages of the conversation."""

    tools: Optional[Sequence[ChatCompletionFunctionToolParam]]
    """Tools provided for the conversation."""

    final_reward: Optional[float]
    """Single reward value for the entire conversation."""

    raw_calls: Sequence[AnnotatedChatCompletionCall]
    """Raw chat completion calls. The order of the calls must be the same as the order of the messages."""
