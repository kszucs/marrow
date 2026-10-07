# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Sets of a join chain's participants, as the bits of an `Int`."""

from std.bit import count_leading_zeros, count_trailing_zeros, pop_count


@fieldwise_init
struct ParticipantSet(
    Equatable,
    Hashable,
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    Movable,
    TrivialRegisterPassable,
    Writable,
):
    """A set of a join chain's participants, by their index in
    `JoinChain.inputs` — below 63. Iterating yields them in ascending order.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Int

    var mask: Int

    @always_inline
    def __init__(out self):
        """The empty set."""
        self.mask = 0

    @always_inline
    @staticmethod
    def of(i: Int) -> Self:
        """The set holding only `i`."""
        return Self(1 << i)

    @always_inline
    @staticmethod
    def below(n: Int) -> Self:
        """`0` through `n - 1`."""
        return Self((1 << n) - 1)

    @always_inline
    def is_empty(self) -> Bool:
        return self.mask == 0

    @always_inline
    def is_single(self) -> Bool:
        return self.mask != 0 and self.mask & (self.mask - 1) == 0

    @always_inline
    def count(self) -> Int:
        return Int(pop_count(self.mask))

    @always_inline
    def lowest(self) -> Int:
        """The least member. Undefined for the empty set."""
        return Int(count_trailing_zeros(self.mask))

    @always_inline
    def highest(self) -> Int:
        """The greatest member. Undefined for the empty set."""
        return 63 - Int(count_leading_zeros(self.mask))

    @always_inline
    def within(self, other: Self) -> Bool:
        """Is every member of this set one of `other`'s?"""
        return self.mask & ~other.mask == 0

    @always_inline
    def meets(self, other: Self) -> Bool:
        """Do the two sets share a member?"""
        return self.mask & other.mask != 0

    @always_inline
    def __contains__(self, i: Int) -> Bool:
        return self.mask & (1 << i) != 0

    @always_inline
    def __or__(self, other: Self) -> Self:
        return Self(self.mask | other.mask)

    @always_inline
    def __and__(self, other: Self) -> Self:
        return Self(self.mask & other.mask)

    @always_inline
    def __sub__(self, other: Self) -> Self:
        return Self(self.mask & ~other.mask)

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        if self.mask == 0:
            raise StopIteration()
        var i = self.lowest()
        self.mask &= self.mask - 1
        return i

    @always_inline
    def __has_next__(self) -> Bool:
        return self.mask != 0

    def subsets(self) -> Subsets:
        """Every non-empty subset, the whole set first, descending by mask."""
        return Subsets(self.mask, self.mask)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("{")
        var first = True
        for i in self:
            if not first:
                writer.write(", ")
            writer.write(i)
            first = False
        writer.write("}")


@fieldwise_init
struct Subsets(
    ImplicitlyCopyable, Iterable, Iterator, Movable, TrivialRegisterPassable
):
    """The non-empty subsets of a set, descending by mask: `(sub - 1) & set`
    steps from one to the next. `next` is the subset to yield, `0` once
    done."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = ParticipantSet

    var mask: Int
    var next: Int

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> ParticipantSet:
        var sub = self.next
        if sub == 0:
            raise StopIteration()
        self.next = (sub - 1) & self.mask
        return ParticipantSet(sub)

    @always_inline
    def __has_next__(self) -> Bool:
        return self.next != 0
