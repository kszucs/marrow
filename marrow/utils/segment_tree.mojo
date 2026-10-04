# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`SegmentTree` — range folds over a fixed sequence in O(log n).

Built once from `n` leaves, then asked for the fold of any `[lo, hi)`. The
layout is the unpadded `2n` array of Al.Cash's "Efficient and easy segment
trees" (codeforces.com/blog/entry/18051), as in the Rust `segment-tree` crate;
`query` keeps a left and a right accumulator, as ac-library-rs's
`Segtree::prod` does, so a non-commutative monoid folds in order. Build once,
query many: there is no point update.
"""


trait Monoid(Copyable, Deinitable):
    """A value with an associative `combine` and an `identity` — the element
    a `SegmentTree` folds is its own monoid, as in the `segment-tree` crate."""

    @staticmethod
    def identity() -> Self:
        """The element `combine` leaves unchanged on either side."""
        ...

    @staticmethod
    def combine(a: Self, b: Self) -> Self:
        """Fold `a` then `b`. Must be associative; need not be commutative."""
        ...


struct SegmentTree[T: Monoid](Copyable, Movable, Sized):
    """Folds of `T` over every range of a fixed sequence of leaves."""

    var _nodes: List[Self.T]
    """Leaves at `[n, 2n)`, node `i` folding `2i` and `2i + 1`, slot 0 unused."""

    def __init__(out self, leaves: List[Self.T]):
        """Build over `leaves` in O(n)."""
        var n = len(leaves)
        self._nodes = List[Self.T](capacity=2 * n)
        for _ in range(n):
            self._nodes.append(Self.T.identity())
        for ref leaf in leaves:
            self._nodes.append(leaf.copy())
        for i in reversed(range(1, n)):
            self._nodes[i] = Self.T.combine(
                self._nodes[2 * i], self._nodes[2 * i + 1]
            )

    def __len__(self) -> Int:
        return len(self._nodes) // 2

    def query(self, lo: Int, hi: Int) -> Self.T:
        """The fold of leaves `[lo, hi)`, in order; `identity()` if empty."""
        var n = len(self)
        debug_assert(0 <= lo and hi <= n, "query: range out of bounds")
        var left = Self.T.identity()
        var right = Self.T.identity()
        var l = lo + n
        var r = hi + n
        while l < r:
            if l & 1:
                left = Self.T.combine(left, self._nodes[l])
                l += 1
            if r & 1:
                r -= 1
                right = Self.T.combine(self._nodes[r], right)
            l >>= 1
            r >>= 1
        return Self.T.combine(left, right)
