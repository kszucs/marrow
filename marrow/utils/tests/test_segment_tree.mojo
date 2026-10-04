# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`SegmentTree` against a brute-force fold over every range.

The sizes cover the empty tree, a single leaf, odd sizes and powers of two
with one either side — the unpadded `2n` layout is where an off-by-one would
live, and only a non-power-of-two `n` reaches it.
"""

from std.testing import assert_equal

from ..segment_tree import Monoid, SegmentTree


@fieldwise_init
struct _Sum(Monoid):
    var v: Int

    @staticmethod
    def identity() -> Self:
        return Self(0)

    @staticmethod
    def combine(a: Self, b: Self) -> Self:
        return Self(a.v + b.v)


@fieldwise_init
struct _Concat(Monoid):
    """Associative but not commutative: any reordering shows in the answer."""

    var s: String

    @staticmethod
    def identity() -> Self:
        return Self(String())

    @staticmethod
    def combine(a: Self, b: Self) -> Self:
        return Self(a.s + b.s)


comptime _SIZES: List[Int] = [0, 1, 2, 3, 5, 7, 8, 9, 13, 16, 17]


def test_segment_tree_sums_every_range() raises:
    for n in materialize[_SIZES]():
        var leaves = List[_Sum]()
        for i in range(n):
            leaves.append(_Sum((i * 7919) % 101 - 50))
        var tree = SegmentTree(leaves)
        assert_equal(len(tree), n)
        for lo in range(n + 1):
            for hi in range(lo, n + 1):
                var expected = 0
                for i in range(lo, hi):
                    expected += leaves[i].v
                assert_equal(tree.query(lo, hi).v, expected)


def test_segment_tree_preserves_order() raises:
    for n in materialize[_SIZES]():
        var leaves = List[_Concat]()
        for i in range(n):
            leaves.append(_Concat(String(chr(ord("a") + i))))
        var tree = SegmentTree(leaves)
        for lo in range(n + 1):
            for hi in range(lo, n + 1):
                var expected = String()
                for i in range(lo, hi):
                    expected += leaves[i].s
                assert_equal(tree.query(lo, hi).s, expected)


def test_segment_tree_empty_range_is_identity() raises:
    var tree = SegmentTree([_Sum(4), _Sum(5), _Sum(6)])
    assert_equal(tree.query(0, 0).v, 0)
    assert_equal(tree.query(3, 3).v, 0)
