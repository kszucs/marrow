"""The comptime lane's aggregates.

Two nodes, on one axis — what the aggregate consumes:

- A fusing `Aggregate` folds its operand's **lanes** into registers and never
  materialises it. That is the 14.6x, and it is why `A` must be a
  `PrimitiveValue`: a lane is a `SIMD`, which needs a fixed width — and needs
  nothing else, so a temporal `min`/`max` fuses too.
- A non-fusing one materialises the operand **once** and computes over the
  column, for the aggregates that have no fold algebra to fuse into. Its
  operand stays typed, so `count_distinct(upper(s))` still fuses `upper(s)`.

Plus the cases a single-batch test would miss.
"""

from std.testing import assert_equal, assert_true
from ....schema import Schema

from ...builders import col, count_star, lit, table
from ....arrays import Int32Array
from ....builders import array, arange
from ....dtypes import (
    DynType,
    float64,
    Int32Type,
    Int64Type,
    int32,
    int64,
    microsecond,
    string,
    timestamp,
)
from ....builders import TimestampBuilder
from ....tabular import RecordBatch, record_batch
from ....kernels.core import Groups
from ...logical import DynValue
from ...physical import Morsel
from ..aggregates import Count, Max, Mean, Min, Product, Sum
from ..leaves import Column, Literal
from ..numeric import Mul
from ..strings import Upper


def _strings(var values: List[Optional[String]]) raises -> RecordBatch:
    return record_batch([array(values).to_dyn()], names=["s"])


def _keyed() raises -> RecordBatch:
    """`g` groups rows 0-2 and rows 3-4; `s` is what the aggregates read."""
    var values: List[Optional[String]] = ["a", "b", "a", "c", "c"]
    return record_batch(
        [array([1, 1, 1, 2, 2], int64).to_dyn(), array(values).to_dyn()],
        names=["g", "s"],
    )


def _stamps(var values: List[Int]) raises -> RecordBatch:
    var b = TimestampBuilder(timestamp(microsecond, "UTC"), len(values))
    for v in values:
        b.append(Scalar[int64.native](v))
    return record_batch(
        [array([1, 1, 2, 2], int64).to_dyn(), b.finish().to_dyn()],
        names=["g", "ts"],
    )


def _b(var v: List[Optional[Int]]) raises -> RecordBatch:
    return record_batch([array(v^, int64).copy()], names=["a"])


def _groups(var g: List[Optional[Int]]) raises -> Int32Array:
    return array(g^, int32).copy()


def _m(var batch: RecordBatch, var ids: Int32Array, n: Int) raises -> Morsel:
    """A morsel carries its grouping, which is what lets a fold be an
    `Operator` rather than needing a trait of its own."""
    return Morsel(batch.to_struct_array(), Groups(ids^, n))


def test_fused_sum_folds_across_morsels() raises:
    """One state, several batches — what `to_state` exists for."""
    var s = col("a", int64).sum().alias("total").to_operator(Schema(), False)
    _ = s.push(_m(_b([1, 2]), _groups(List[Optional[Int]]()), 1))
    _ = s.push(_m(_b([3, 4]), _groups(List[Optional[Int]]()), 1))
    _ = s.push(_m(_b([5]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([15], int64))


def test_fused_sum_skips_nulls() raises:
    var s = col("a", int64).sum().alias("t").to_operator(Schema(), False)
    _ = s.push(_m(_b([1, None, 3]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([4], int64))


def test_min_max_expose_null_blindness() raises:
    """`sum` can be silently right over a null whose payload is 0; `min` cannot.
    These are the cases that prove the lane mask is applied."""
    var lo = col("a", int64).min().alias("lo").to_operator(Schema(), False)
    var hi = col("a", int64).max().alias("hi").to_operator(Schema(), False)
    _ = lo.push(_m(_b([5, None, 9]), _groups(List[Optional[Int]]()), 1))
    _ = hi.push(_m(_b([5, None, 9]), _groups(List[Optional[Int]]()), 1))
    assert_true(lo.drain().value().to_array(1) == array([5], int64))
    assert_true(hi.drain().value().to_array(1) == array([9], int64))


def test_fused_sum_over_no_rows_is_null() raises:
    """R10, and the live out-of-bounds this design was blocked on: `AggState`
    only grew in `update`, so an aggregate that never updated read a slot that
    did not exist — a crash under ASSERT=all, a silent bad read otherwise."""
    var s = col("a", int64).sum().alias("t").to_operator(Schema(), False)
    assert_true(s.drain().value().to_array(1).as_int64().is_null(0))


def test_fused_sum_over_empty_batch_is_null() raises:
    var s = col("a", int64).sum().alias("t").to_operator(Schema(), False)
    _ = s.push(_m(_b(List[Optional[Int]]()), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1).as_int64().is_null(0))


def test_fused_sum_over_all_nulls_is_null() raises:
    var s = col("a", int64).sum().alias("t").to_operator(Schema(), False)
    _ = s.push(_m(_b([None, None]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1).as_int64().is_null(0))


def test_a_fused_subtree_never_materialises() raises:
    """`sum(a * 2)` — the input is a fused node, so the fold reads its lane."""
    var s = (
        (col("a", int64) * lit(2, int64))
        .sum()
        .alias("t")
        .to_operator(Schema(), False)
    )
    _ = s.push(_m(_b([1, 2, 3]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([12], int64))


def test_a_ragged_tail_stays_in_bounds() raises:
    """The row count is not a multiple of the SIMD width. A body without a
    scalar tail
    reads past the view and aborts the process."""
    var b = record_batch([arange[Int64Type](0, 1003).to_dyn()], names=["a"])
    var s = col("a", int64).sum().alias("t").to_operator(Schema(), False)
    _ = s.push(_m(b.copy(), _groups(List[Optional[Int]]()), 1))
    assert_true(
        s.drain().value().to_array(1) == array([1003 * 1002 // 2], int64)
    )


def test_sum_widens_to_the_accumulator_type() raises:
    var b = record_batch([array([1, 2, 3], int32).copy()], names=["a"])
    var agg = col("a", int32).sum().alias("t")
    var s = agg.to_operator(Schema(), False)
    _ = s.push(_m(b.copy(), _groups(List[Optional[Int]]()), 1))
    var got = s.drain().value().to_array(1)
    assert_true(got.dtype() == agg.dtype(b.schema))
    assert_true(got == array([6], int64))


def test_grouped_folds_into_slots() raises:
    var s = col("a", int64).sum().alias("t").to_operator(Schema(), True)
    _ = s.push(_m(_b([1, 2, 3, 4]), _groups([0, 1, 0, 1]), 2))
    _ = s.push(_m(_b([10, 20]), _groups([1, 0]), 2))
    assert_true(
        s.drain().value().to_array(2) == array([1 + 3 + 20, 2 + 4 + 10], int64)
    )


def test_grouped_skips_nulls_per_group() raises:
    var s = col("a", int64).min().alias("t").to_operator(Schema(), True)
    _ = s.push(_m(_b([5, None, 1, 9]), _groups([0, 0, 1, 1]), 2))
    assert_true(s.drain().value().to_array(2) == array([5, 1], int64))


def test_mean_uses_the_valid_count_as_divisor() raises:
    """The count is not bookkeeping: it is `finalize`'s divisor, and a null
    must not be in it."""
    var s = col("a", int64).mean().alias("t").to_operator(Schema(), False)
    _ = s.push(_m(_b([1, None, 5]), _groups(List[Optional[Int]]()), 1))
    assert_equal(String(s.drain().value().to_array(1).as_float64()[0]), "3.0")


def test_erasure_answers_as_the_value_it_holds() raises:
    var b = _b([1, 2, 3])
    var agg = col("a", int64).sum().alias("total")
    var boxed = agg.copy()
    assert_equal(boxed.name(), "total")
    assert_equal(boxed.columns()[0], "a")
    assert_true(boxed.dtype(b.schema) == agg.dtype(b.schema))
    var s = boxed.to_operator(Schema(), False)
    _ = s.push(_m(b.copy(), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([6], int64))


def test_a_fold_reports_spent_on_a_second_drain() raises:
    """`drain` is repeatable, so a fold must be able to say "nothing left".

    The driver calls `drain` until it answers `None`. A fold that answered
    `Some` every time would spin `while True: drain()` forever — it is only
    safe today because `GroupByOperator` happens to call it once, and
    "happens to" is not a contract.
    """
    var s = col("a", int64).sum().alias("t").to_operator(Schema(), False)
    _ = s.push(_m(_b([1, 2]), _groups(List[Optional[Int]]()), 1))
    assert_true(Bool(s.drain()))
    assert_true(not Bool(s.drain()))


def test_product_folds() raises:
    """`Product` had no test at all — found by auditing public names against
    test references."""
    var s = col("a", int64).product().alias("p").to_operator(Schema(), False)
    _ = s.push(_m(_b([2, 3, 4]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([24], int64))


def test_count_skips_nulls() raises:
    """`COUNT(x)` is the *valid* count, which is what separates it from
    `COUNT(*)` on any nullable column."""
    var s = col("a", int64).count().to_operator(Schema(), False)
    _ = s.push(_m(_b([1, None, 3, None, 5]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([3], int64))


def test_count_star_counts_every_row_including_nulls() raises:
    """`count_star()` is `CountKernel` over a literal, and a literal is valid
    on every row — so the valid-count of a constant column is the row count.
    The trick is only correct if it survives a *nullable* input column, which
    is the whole point of testing it against one.
    """
    var s = count_star().to_operator(Schema(), False)
    _ = s.push(_m(_b([1, None, 3, None, 5]), _groups(List[Optional[Int]]()), 1))
    assert_true(s.drain().value().to_array(1) == array([5], int64))


def test_count_and_count_star_disagree_on_a_nullable_column() raises:
    """Stated as one case because the two are constantly confused, and a test
    that pins each separately does not show that they must differ."""
    var batch = _b([1, None, 3])
    var ids = _groups(List[Optional[Int]]())

    var counted = col("a", int64).count().to_operator(Schema(), False)
    _ = counted.push(_m(batch.copy(), ids.copy(), 1))

    var starred = count_star().to_operator(Schema(), False)
    _ = starred.push(_m(batch.copy(), ids.copy(), 1))

    assert_true(counted.drain().value().to_array(1) == array([2], int64))
    assert_true(starred.drain().value().to_array(1) == array([3], int64))


def test_count_is_named_and_aliasable() raises:
    """`count_star()` arrives pre-aliased; `.count()` takes the kernel's name
    until something renames it."""
    assert_equal(col("a", int64).count().name(), "count")
    assert_equal(count_star().name(), "count_star")
    assert_equal(col("a", int64).count().alias("n").name(), "n")


# ---------------------------------------------------------------------------
# Aggregate that cannot fuse — count_distinct
#
# Two failure modes are specific to this node and each has cases of its own:
#
# - `Agg.dtype` reaches the plan's schema and `Agg.grouped` produces the
#   column. `grouped` returns `DynArray`, so a disagreement is a `Variant`
#   misaccess at emit rather than a raise — every case asserts the *schema*
#   dtype equals the *produced* dtype.
# - The keyless query is the one that carries no group ids, so it is the one
#   that silently answers `[0]` if an implementation forgets its
#   `Groups.is_single()` branch.
# ---------------------------------------------------------------------------
def test_column_agg_count_distinct_keyless_string() raises:
    """No `GROUP BY`, so the morsel carries the one-slot assignment and the
    fold sees an **empty** id array. Get the branch wrong and this is 0."""
    var plan = table(_strings(["a", "b", "a", "c", "b"])).aggregate(
        [col("s", string).count_distinct()], List[DynValue]()
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[0].as_int64() == array([3], int64))
    assert_true(plan.schema() == out.schema)


def test_column_agg_count_distinct_keyless_numeric() raises:
    """The trait default is on `ComptimeValue`, so a numeric node has it too —
    and a cardinality is int64 whatever was counted."""
    var batch = record_batch(
        [array([10, 20, 10, 30, 20], int64).to_dyn()], names=["n"]
    )
    var plan = table(batch^).aggregate(
        [col("n", int64).count_distinct()], List[DynValue]()
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([3], int64))
    assert_true(plan.schema().fields[0].dtype == DynType(int64))
    assert_true(plan.schema() == out.schema)


def test_column_agg_count_distinct_grouped_string() raises:
    """Group 1 sees a, b, a; group 2 sees c, c."""
    var plan = table(_keyed()).aggregate(
        [col("s", string).count_distinct().alias("distinct_s")],
        [col("g", int64)],
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[0].as_int64() == array([1, 2], int64))
    assert_true(out.columns[1].as_int64() == array([2, 1], int64))
    assert_true(plan.schema() == out.schema)


def test_column_agg_count_distinct_excludes_nulls() raises:
    """SQL `COUNT(DISTINCT x)` counts values, and NULL is not one."""
    var plan = table(_strings(["a", None, "a", "b", None])).aggregate(
        [col("s", string).count_distinct()], List[DynValue]()
    )
    assert_true(plan.execute().columns[0].as_int64() == array([2], int64))


def test_column_agg_approx_count_distinct_keyless() raises:
    """Exact at this cardinality — linear counting takes over far below the
    HyperLogLog register count."""
    var plan = table(_strings(["a", "b", "a", "c", "b"])).aggregate(
        [col("s", string).approx_count_distinct()], List[DynValue]()
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([3], int64))
    assert_true(plan.schema() == out.schema)


# ---------------------------------------------------------------------------
# string min / max
# ---------------------------------------------------------------------------
def test_column_agg_string_min_max_keyless() raises:
    """Lexicographic (bytewise), matching Arrow's `hash_min`/`hash_max`. The
    output keeps the input's dtype, so the schema must say `string`."""
    var plan = table(_strings(["b", "a", "c"])).aggregate(
        [
            col("s", string).min().alias("lo"),
            col("s", string).max().alias("hi"),
        ],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_string() == array(["a"]))
    assert_true(out.columns[1].as_string() == array(["c"]))
    assert_true(plan.schema().fields[0].dtype == DynType(string))
    assert_true(plan.schema() == out.schema)


def test_column_agg_string_min_max_grouped() raises:
    var plan = table(_keyed()).aggregate(
        [
            col("s", string).min().alias("lo"),
            col("s", string).max().alias("hi"),
        ],
        [col("g", int64)],
    )
    var out = plan.execute()
    assert_true(out.columns[1].as_string() == array(["a", "c"]))
    assert_true(out.columns[2].as_string() == array(["b", "c"]))
    assert_true(plan.schema() == out.schema)


def test_column_agg_string_min_skips_nulls() raises:
    var plan = table(_strings([None, "b", "a"])).aggregate(
        [col("s", string).min()], List[DynValue]()
    )
    assert_true(plan.execute().columns[0].as_string() == array(["a"]))


# ---------------------------------------------------------------------------
# temporal min / max
# ---------------------------------------------------------------------------
def test_column_agg_timestamp_min_keeps_unit_and_timezone() raises:
    """The pairing the compiler no longer checks. `MinMax.acc_dtype` answers
    with the *input* dtype, so `aggregate_out_dtype` must too — a schema
    saying `timestamp[s]` over a `timestamp[us]` column would be a `Variant`
    misaccess at emit, not a raise."""
    var plan = table(_stamps([30, 10, 50, 40])).aggregate(
        [col("ts", timestamp(microsecond, "UTC")).min().alias("first_seen")],
        List[DynValue](),
    )
    var out = plan.execute()
    var expected = DynType(timestamp(microsecond, "UTC"))
    assert_true(plan.schema().fields[0].dtype == expected)
    assert_true(out.columns[0].dtype() == expected)
    assert_true(plan.schema() == out.schema)
    assert_equal(Int(out.columns[0].as_timestamp()[0].value()), 10)


def test_column_agg_timestamp_max_grouped_keeps_unit() raises:
    var plan = table(_stamps([30, 10, 50, 40])).aggregate(
        [col("ts", timestamp(microsecond, "UTC")).max().alias("last_seen")],
        [col("g", int64)],
    )
    var out = plan.execute()
    var expected = DynType(timestamp(microsecond, "UTC"))
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].dtype() == expected)
    assert_true(plan.schema() == out.schema)
    assert_equal(Int(out.columns[1].as_timestamp()[0].value()), 30)
    assert_equal(Int(out.columns[1].as_timestamp()[1].value()), 50)


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------
def test_column_agg_alias_renames_without_changing_the_function() raises:
    """`alias` cannot change which kernel runs, structurally.

    The aggregate is `Agg`, a comptime parameter, so the only thing an alias
    can touch is the name the output schema reads — which is why this node
    needs no second name field where `RuntimeAggregate` does.
    """
    var plain = col("s", string).count_distinct()
    var renamed = plain.alias("n")
    assert_equal(plain.name(), "count_distinct")
    assert_equal(renamed.name(), "n")
    assert_true(String(renamed).startswith("count_distinct("))

    var plan = table(_strings(["a", "b", "a"])).aggregate(
        [renamed.copy()], List[DynValue]()
    )
    assert_equal(plan.schema().fields[0].name, "n")
    # Still the same aggregate: two distinct values, not a renamed no-op.
    assert_true(plan.execute().columns[0].as_int64() == array([2], int64))


# ---------------------------------------------------------------------------
# an input that yields no morsel at all
# ---------------------------------------------------------------------------
def test_column_agg_count_distinct_over_no_rows_is_zero() raises:
    """A filter that keeps nothing answers `None` rather than an empty batch,
    so the fold is never pushed and never learns its input dtype. `COUNT
    (DISTINCT x)` of nothing is 0 — the one answer that needs no dtype."""
    var plan = (
        table(_strings(["a", "b"]))
        .filter(col("s", string) > lit(String("zzz"), string))
        .aggregate([col("s", string).count_distinct()], List[DynValue]())
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[0].as_int64() == array([0], int64))


def test_column_agg_string_min_over_no_rows_is_null() raises:
    """`min` of nothing is NULL, and its dtype is only still known to the
    aggregate stage's output schema."""
    var plan = (
        table(_strings(["a", "b"]))
        .filter(col("s", string) > lit(String("zzz"), string))
        .aggregate([col("s", string).min()], List[DynValue]())
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[0].dtype() == DynType(string))
    assert_true(out.columns[0].is_null(0))


# ---------------------------------------------------------------------------
# the operand keeps its fusion
# ---------------------------------------------------------------------------
def test_column_agg_counts_distinct_over_a_fused_operand() raises:
    """The point of `Aggregate[Agg, A]` keeping `A` typed.


    `Upper[StringColumn[StringType]]` is a type, so `upper(s)` compiles to one
    loop and reaches the aggregate as a column; only the distinct count
    materialises. Boxing the operand into a `DynValue` would have thrown that
    away too, which nothing here requires.

    Case-folding collapses `A`/`a` and `B`/`b`, so five values become two.
    """
    var plan = table(_strings(["a", "A", "b", "B", "a"])).aggregate(
        [Upper(col("s", string)).count_distinct().alias("n")],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([2], int64))
    assert_true(plan.schema() == out.schema)


def test_column_agg_min_over_a_fused_operand() raises:
    """Same for an extremum: `min(upper(s))` is `"A"`, not `"a"`."""
    var plan = table(_strings(["b", "a", "C"])).aggregate(
        [Upper(col("s", string)).min().alias("lo")], List[DynValue]()
    )
    assert_true(plan.execute().columns[0].as_string() == array(["A"]))


# ---------------------------------------------------------------------------
# variance / stddev — a fold that is not a `Fold`
#
# Welford's accumulator is a triple (count, mean, M2) where `AggState` holds one
# accumulator column plus one count, so `Dispersion` is not a `FoldKernel` and
# `Aggregate.fuses` answers False for it. These cases pin the semantics against
# PyArrow and pin the *non*-fusion, since that is the property the naming turns
# on.
# ---------------------------------------------------------------------------
def _spread() raises -> RecordBatch:
    """Group 1 holds [1, 3] and group 2 holds [10, 20, 30]."""
    return record_batch(
        [
            array([1, 1, 2, 2, 2], int64).to_dyn(),
            array([1, 3, 10, 20, 30], int64).to_dyn(),
        ],
        names=["g", "v"],
    )


def test_variance_does_not_fuse_but_its_operand_does() raises:
    """The whole reason `Dispersion` is a sibling of `Fold` rather than a
    `FoldKernel`. `fuses` is a comptime fact, so this is a compile-time
    assertion dressed as a test."""
    comptime Var = type_of(col("v", int64).variance())
    comptime Summed = type_of(col("v", int64).sum())
    assert_true(not Var.fuses, "a composite accumulator cannot fuse")
    assert_true(Summed.fuses, "a scalar fold still does")


def test_variance_keyless_is_the_population_form() raises:
    """`pc.variance([1,3,10,20,30])` is 118.16 at the default ddof=0."""
    var plan = table(_spread()).aggregate(
        [col("v", int64).variance()], List[DynValue]()
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(plan.schema().fields[0].dtype == DynType(float64))
    assert_true(abs(out.columns[0].as_float64()[0].value() - 118.16) < 1e-9)
    assert_true(plan.schema() == out.schema)


def test_variance_ddof_is_a_comptime_parameter() raises:
    """`variance[1]()` is the sample form — a different *type*, resolved where
    the query is written, so no runtime option is threaded anywhere."""
    var plan = table(_spread()).aggregate(
        [col("v", int64).variance[1]()], List[DynValue]()
    )
    var got = plan.execute().columns[0].as_float64()[0].value()
    assert_true(abs(got - 147.7) < 1e-9)


def test_stddev_is_the_root_of_the_variance_expression() raises:
    var plan = table(_spread()).aggregate(
        [col("v", int64).stddev().alias("sd")], List[DynValue]()
    )
    var got = plan.execute().columns[0].as_float64()[0].value()
    assert_true(abs(got - 10.870142593360953) < 1e-9)


def test_variance_grouped() raises:
    """Group 1 is [1,3] -> 1.0; group 2 is [10,20,30] -> 200/3."""
    var plan = table(_spread()).aggregate(
        [col("v", int64).variance().alias("var_v")], [col("g", int64)]
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    ref got = out.columns[1].as_float64()
    assert_true(abs(got[0].value() - 1.0) < 1e-9)
    assert_true(abs(got[1].value() - 200.0 / 3.0) < 1e-9)


def test_variance_of_a_fused_subtree() raises:
    """Only the dispersion materialises: `v * 2` is still one loop, and
    scaling by 2 scales the variance by 4."""
    var plan = table(_spread()).aggregate(
        [(col("v", int64) * lit(2, int64)).variance().alias("v4")],
        List[DynValue](),
    )
    var got = plan.execute().columns[0].as_float64()[0].value()
    assert_true(abs(got - 118.16 * 4.0) < 1e-9)


# ---------------------------------------------------------------------------
# A fused subtree under a GROUP BY
#
# A fused fold is two separate operators — `ScatteredAggregateOperator` walks
# group ids and does a random write per row, `RegisterAggregateOperator`
# accumulates in registers and hands off once per morsel. Every other
# fused-subtree case in this tree is **keyless**, so all of them exercise the
# register operator and none exercise the scatter loop reading `lane[W]` out of
# a computed operand. These do.
# ---------------------------------------------------------------------------
def _lines() raises -> RecordBatch:
    """Two groups of order lines, with a null quantity in the second."""
    var qty: List[Optional[Int]] = [3, 7, 10, 2, None]
    return record_batch(
        [
            array([1, 1, 2, 2, 2], int64).to_dyn(),
            array(qty, int64).to_dyn(),
            array([100, 100, 20, 250, 60], int64).to_dyn(),
        ],
        names=["g", "qty", "price"],
    )


def test_a_fused_subtree_folds_per_group() raises:
    """`sum(qty * price)` grouped: the product is never materialised, and the
    null row propagates *through* the multiply rather than contributing its
    price alone.

    Group 1 is 3*100 + 7*100 = 1000. Group 2 is 10*20 + 2*250 = 700, with the
    null line contributing nothing.
    """
    var plan = table(_lines()).aggregate(
        [
            (col("qty", int64) * col("price", int64)).sum().alias("revenue"),
        ],
        [col("g", int64)],
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].as_int64() == array([1000, 700], int64))
    assert_true(plan.schema() == out.schema)


def test_a_fused_subtree_folds_per_group_under_every_kernel() raises:
    """The scatter loop is `K`-generic, so one kernel passing is not evidence
    for the rest: `max` keeps the input's type where `mean` widens to float64,
    and both read the same fused operand."""
    var plan = table(_lines()).aggregate(
        [
            (col("qty", int64) * col("price", int64)).max().alias("peak"),
            (col("qty", int64) * lit(2, int64)).mean().alias("avg2"),
        ],
        [col("g", int64)],
    )
    var out = plan.execute()
    assert_true(out.columns[1].as_int64() == array([700, 500], int64))
    ref avg = out.columns[2].as_float64()
    assert_true(abs(avg[0].value() - 10.0) < 1e-9)  # (6 + 14) / 2
    assert_true(abs(avg[1].value() - 12.0) < 1e-9)  # (20 + 4) / 2, null skipped


def test_a_fused_subtree_and_a_buffered_aggregate_share_one_grouping() raises:
    """A fused fold and a buffered aggregate in the same query: `GroupByOperator`
    assigns ids once and both read them, so the two operators must agree on the
    slot count even though only one of them buffers."""
    var plan = table(_lines()).aggregate(
        [
            (col("qty", int64) * col("price", int64)).sum().alias("revenue"),
            col("price", int64).count_distinct().alias("prices"),
        ],
        [col("g", int64)],
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].as_int64() == array([1000, 700], int64))
    assert_true(out.columns[2].as_int64() == array([1, 3], int64))
