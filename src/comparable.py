"""
This file provides the type which could be a key of a dictionary and
could be comparable. That is it is endowed with hash function, and
comparison operators.
"""

from typing import Protocol, TypeVar


class ComparableType(Protocol):
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


Comparable = TypeVar('Comparable', bound=ComparableType)
