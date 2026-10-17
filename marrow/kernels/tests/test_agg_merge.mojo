# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`AggKernel.partials` / `merge`: aggregates over separate parts of an input
merge into the aggregate of the whole.

Every case splits one column into `k` parts, aggregates each part on its own,
and merges the parts' partial states into a fresh aggregate whose slots are
a permutation of the parts' slots. The answer must equal one aggregate over
the whole column with the same permutation applied to its group ids.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import (
    DynArray,
    Float64Array,
    Int32Array,
    Int64Array,
    StringArray,
)
from ...builders import (
    Float64Builder,
    Int32Builder,
    Int64Builder,
    StringBuilder,
)
from ...kernels.groupby import Groups
from ...kernels.aggregate import (
    AggKernel,
    ApproxDistinctCount,
    CountFold,
    Dispersion,
    DistinctCount,
    Fold,
    LexicalExtremum,
    MaxOp,
    MeanFold,
    MinFold,
    MinOp,
    SumFold,
    ValidCount,
)
from ...dtypes import Float64Type, Int64Type

comptime ROWS = 1000
comptime GROUPS = 17


def _ints() raises -> DynArray:
    var b = Int64Builder(ROWS)
    for i in range(ROWS):
        if i % 11 == 3:
            b.append_null()
        else:
            b.append(Int64((i * 7919) % 113) - 50)
    return b.finish()


def _floats() raises -> DynArray:
    var b = Float64Builder(ROWS)
    for i in range(ROWS):
        if i % 13 == 5:
            b.append_null()
        else:
            b.append(Float64((i * 7919) % 1009) * 0.37 + 1e6)
    return b.finish()


def _strings() raises -> DynArray:
    var b = StringBuilder(ROWS)
    for i in range(ROWS):
        if i % 7 == 2:
            b.append_null()
        else:
            b.append(String("k", (i * 31) % 97))
    return b.finish()


def _group_ids(permuted: Bool) raises -> Int32Array:
    """Row `i`'s group, through the slot permutation when `permuted`."""
    var b = Int32Builder(ROWS)
    for i in range(ROWS):
        var g = (i * 5 + i // 3) % GROUPS
        b.append(Int32(_slot(g) if permuted else g))
    return b.finish()


def _slot(g: Int) -> Int:
    """Where a part's slot `g` lands in the combined aggregate."""
    return (g * 7 + 3) % GROUPS


def _slots() raises -> Int32Array:
    var b = Int32Builder(GROUPS)
    for g in range(GROUPS):
        b.append(Int32(_slot(g)))
    return b.finish()


def _whole[A: AggKernel](column: DynArray, grouped: Bool) raises -> A.OutArray:
    var agg = A(column.dtype(), grouped)
    var values = A.InArray(column.to_data())
    if grouped:
        agg.update(Groups(_group_ids(permuted=True), GROUPS), values)
    else:
        agg.update(Groups.single(ROWS), values)
    return agg.finish()


comptime FORWARD = 0
comptime REVERSED = 1
comptime TREE = 2


def _part_states[
    A: AggKernel
](column: DynArray, grouped: Bool, parts: Int) raises -> List[A]:
    """`column` in `parts` parts, each aggregated on its own, with a part that
    saw no rows after the first: what workers hand the combine."""
    var values = A.InArray(column.to_data())
    var ids = _group_ids(permuted=False)
    var out = List[A]()
    var step = (ROWS + parts - 1) // parts
    for p in range(parts):
        var offset = min(p * step, ROWS)
        var length = min(step, ROWS - offset)
        var part = A(column.dtype(), grouped)
        if grouped:
            part.update(
                Groups(ids.slice(offset, length), GROUPS),
                values.slice(offset, length),
            )
        else:
            part.update(Groups.single(length), values.slice(offset, length))
        out.append(part^)
        if p == 0:
            out.append(A(column.dtype(), grouped))
    return out^


def _merged[
    A: AggKernel
](
    column: DynArray,
    grouped: Bool,
    mut states: List[A],
    lo: Int,
    hi: Int,
    permute: Bool,
) raises -> A:
    """`states[lo:hi]` merged into one aggregate; `permute` moves each part's
    slot `g` to `_slot(g)`, as a combine's keys do."""
    var agg = A(column.dtype(), grouped)
    for i in range(lo, hi):
        if not grouped:
            agg.merge(Groups.single(1), states[i].partials())
        elif permute:
            agg.merge(Groups(_slots(), GROUPS), states[i].partials())
        else:
            agg.merge(Groups(_identity(), GROUPS), states[i].partials())
    return agg^


def _identity() raises -> Int32Array:
    var b = Int32Builder(GROUPS)
    for g in range(GROUPS):
        b.append(Int32(g))
    return b.finish()


def _combined[
    A: AggKernel
](
    column: DynArray, grouped: Bool, parts: Int, order: Int = FORWARD
) raises -> A.OutArray:
    """The parts' states merged forwards, backwards, or as a tree: each half
    merged on its own, then the two halves' merged states merged."""
    var states = _part_states[A](column, grouped, parts)
    if order == REVERSED:
        states.reverse()
    if order != TREE or len(states) < 2:
        var agg = _merged[A](column, grouped, states, 0, len(states), True)
        return agg.finish()
    var half = len(states) // 2
    var halves = List[A]()
    halves.append(_merged[A](column, grouped, states, 0, half, True))
    halves.append(_merged[A](column, grouped, states, half, len(states), True))
    var agg = _merged[A](column, grouped, halves, 0, 2, False)
    return agg.finish()


def _check[A: AggKernel](column: DynArray) raises:
    for grouped in [False, True]:
        var expected = _whole[A](column, grouped)
        for parts in [1, 2, 3, 7, 1000]:
            for order in [FORWARD, REVERSED, TREE]:
                var got = _combined[A](column, grouped, parts, order)
                assert_true(
                    got == expected,
                    String(
                        "grouped=", grouped, " parts=", parts, " order=", order
                    ),
                )


def test_agg_merge_sum() raises:
    _check[Fold[SumFold, Int64Type]](_ints())


def test_agg_merge_mean() raises:
    _check[Fold[MeanFold, Int64Type]](_ints())


def test_agg_merge_min() raises:
    _check[Fold[MinFold, Int64Type]](_ints())


def test_agg_merge_count() raises:
    _check[Fold[CountFold, Int64Type]](_ints())


def test_agg_merge_valid_count() raises:
    _check[ValidCount[StringArray]](_strings())


def test_agg_merge_string_min() raises:
    _check[LexicalExtremum[MinOp, StringArray]](_strings())


def test_agg_merge_string_max() raises:
    _check[LexicalExtremum[MaxOp, StringArray]](_strings())


def test_agg_merge_distinct_ints() raises:
    _check[DistinctCount[Int64Array]](_ints())


def test_agg_merge_distinct_strings() raises:
    _check[DistinctCount[StringArray]](_strings())


def test_agg_merge_approx_distinct() raises:
    _check[ApproxDistinctCount[Int64Array]](_ints())


def test_agg_merge_variance() raises:
    """Chan's formula is not bit-identical to Welford's sequential steps, so
    this one compares within a relative tolerance."""
    comptime Var = Dispersion[1, False, Float64Type]
    var column = _floats()
    for grouped in [False, True]:
        var expected = _whole[Var](column, grouped)
        for parts in [1, 2, 3, 7, 1000]:
            for order in [FORWARD, REVERSED, TREE]:
                var got = _combined[Var](column, grouped, parts, order)
                assert_equal(len(got), len(expected))
                for g in range(len(got)):
                    assert_equal(got.is_valid(g), expected.is_valid(g))
                    if expected.is_valid(g):
                        var e = Float64(expected.unsafe_get(g))
                        var d = abs(Float64(got.unsafe_get(g)) - e)
                        assert_true(
                            d <= 1e-12 * max(1.0, abs(e)),
                            String(
                                "grouped=",
                                grouped,
                                " parts=",
                                parts,
                                " order=",
                                order,
                                " g=",
                                g,
                            ),
                        )


def test_agg_merge_empty_parts() raises:
    """A part that saw no rows contributes untouched slots, which change
    nothing."""
    comptime Sum = Fold[SumFold, Int64Type]
    var column = _ints()
    var expected = _whole[Sum](column, True)
    var agg = Sum(column.dtype(), True)
    var empty = Sum(column.dtype(), True)
    empty.reserve(GROUPS)
    agg.merge(Groups(_slots(), GROUPS), empty.partials())
    var part = Sum(column.dtype(), True)
    part.update(Groups(_group_ids(False), GROUPS), Int64Array(column.to_data()))
    agg.merge(Groups(_slots(), GROUPS), part.partials())
    assert_true(agg.finish() == expected)


def test_agg_merge_refuses_a_sketch_of_another_width() raises:
    """A grouped sketch merged into an ungrouped count is an error, not a
    read past the end of the narrower sketch."""
    comptime Approx = ApproxDistinctCount[Int64Array]
    var column = _ints()
    var grouped = Approx(column.dtype(), True)
    grouped.update(
        Groups(_group_ids(permuted=False), GROUPS), Int64Array(column.to_data())
    )
    var whole = Approx(column.dtype(), False)
    with assert_raises(contains="sketch"):
        whole.merge(Groups(_identity(), GROUPS), grouped.partials())
