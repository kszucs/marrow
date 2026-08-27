"""The nodes whose validity is data-dependent.

Every other comptime node is null-in-null-out: it intersects its operands'
bitmaps and never reads their values. These four are the exception, in two
different directions.

`IsNull` / `NotNull` read validity and produce a value that is *never* null.
`Coalesce` / `Nullif` / `FillNull` read values and decide validity from them —
`coalesce` takes the second operand exactly where the first was null, and
`nullif` manufactures a null that neither operand has.

Both are why those nodes are breakers conforming to `ColumnBound`: the rule is
the kernel's, applied once in `bind`, and the answer lives in the bound.
`test_coalesce_null_survives_a_fused_parent` is the case that fails if validity
is ever re-derived structurally instead.
"""

from std.testing import assert_equal, assert_true

from ...builders import col, lit, table
from ...params import Bindings
from ....builders import Int64Builder, array
from ....arrays import Int64Array, StringArray
from ....dtypes import int64, string
from ....tabular import RecordBatch, record_batch
from ..boolean import IsNull, NotNull
from ..numeric import Coalesce, FillNull, Nullif


def _a() raises -> Int64Array:
    var b = Int64Builder(4)
    b.append(1)
    b.append_null()
    b.append(3)
    b.append_null()
    return b.finish()


def _b() raises -> Int64Array:
    var b = Int64Builder(4)
    b.append(10)
    b.append(20)
    b.append_null()
    b.append_null()
    return b.finish()


def _s() raises -> StringArray:
    var values: List[Optional[String]] = ["x", None, "z", "w"]
    return array(values)


def _batch() raises -> RecordBatch:
    """`a` and `b` are null in overlapping-but-different places, so every
    coalesce branch is exercised: first, second, and neither."""
    return record_batch(
        [_a().to_dyn(), _b().to_dyn(), _s().to_dyn()],
        names=["a", "b", "s"],
    )


# ---------------------------------------------------------------------------
# IsNull / NotNull — read validity, produce a value that is never null
# ---------------------------------------------------------------------------
def test_is_null_reads_validity_not_values() raises:
    var b = _batch()
    var got = (
        IsNull(col("a", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(not got[0].value())  # 1
    assert_true(got[1].value())  # null
    assert_true(not got[2].value())  # 3
    assert_true(got[3].value())  # null


def test_is_null_result_is_never_null() raises:
    """The whole content of a validity predicate: `is_null(NULL)` is `TRUE`,
    a perfectly good value. If this node inherited structural validity it
    would report null exactly where the answer is most interesting."""
    var b = _batch()
    var got = (
        IsNull(col("a", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_equal(got.null_count(), 0)


def test_not_null_is_the_complement_of_is_null() raises:
    var b = _batch()
    var got = (
        NotNull(col("a", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())
    assert_true(not got[1].value())
    assert_true(got[2].value())
    assert_true(not got[3].value())


def test_is_null_accepts_any_operand_family() raises:
    """The operand is bound on `ComptimeValue`, not on a family: reading a
    bitmap is one operation whatever the dtype underneath it is."""
    var b = _batch()
    var got = (
        IsNull(col("s", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(not got[0].value())  # "x"
    assert_true(got[1].value())  # null
    assert_true(not got[3].value())  # "w"


def test_is_null_feeds_a_filter() raises:
    """Bit-packed like any other predicate, so it drives `Filter` unchanged."""
    var b = _batch()
    var plan = table(b^).filter(IsNull(col("a", int64)))
    assert_equal(plan.execute().num_rows(), 2)


# ---------------------------------------------------------------------------
# Coalesce / FillNull / Nullif — read values, decide validity from them
# ---------------------------------------------------------------------------
def test_coalesce_takes_the_second_where_the_first_is_null() raises:
    var b = _batch()
    var got = (
        Coalesce(col("a", int64), col("b", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_int64()
        .copy()
    )
    assert_equal(got[0].value(), 1)  # a
    assert_equal(got[1].value(), 20)  # a null -> b
    assert_equal(got[2].value(), 3)  # a


def test_coalesce_is_null_only_where_both_operands_are() raises:
    """Not an intersection of the operands' bitmaps — a *union* of their
    non-nullness. Structural validity gets this exactly backwards."""
    var b = _batch()
    var got = (
        Coalesce(col("a", int64), col("b", int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_true(not got.is_null(1))  # a null, b present
    assert_true(not got.is_null(2))  # a present, b null
    assert_true(got.is_null(3))  # both null
    assert_equal(got.null_count(), 1)


def test_fill_null_replaces_nulls_from_the_second_operand() raises:
    var b = _batch()
    var got = (
        FillNull(col("a", int64), lit(0, int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_int64()
        .copy()
    )
    assert_equal(got[0].value(), 1)
    assert_equal(got[1].value(), 0)
    assert_equal(got[2].value(), 3)
    assert_equal(got[3].value(), 0)


def test_fill_null_leaves_no_nulls_behind() raises:
    var b = _batch()
    var got = (
        FillNull(col("a", int64), lit(0, int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_equal(got.null_count(), 0)


def test_nullif_manufactures_a_null_neither_operand_has() raises:
    """The clearest case for data-dependent validity: both operands are valid
    at row 2 and the result is null, because the *values* matched."""
    var b = _batch()
    var got = (
        Nullif(col("a", int64), lit(3, int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_equal(got.as_int64()[0].value(), 1)
    assert_true(got.is_null(2))  # a == 3, so nulled
    assert_equal(got.null_count(), 3)  # rows 1 and 3 were already null


def test_coalesce_null_survives_a_fused_parent() raises:
    """The regression this port exists to not reintroduce.

    `exprold` answered validity from the batch as well as from the state, and
    its batch-side answer re-ran `combine` over both operands to recover a
    bitmap the result already carried (`marrow/exprold/values.mojo:2578-2588`).
    Here `bind` computes once and a fused parent reads `validity(bound)`.
    """
    var b = _batch()
    var got = (
        (Coalesce(col("a", int64), col("b", int64)) + lit(1, int64))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_equal(got.as_int64()[0].value(), 2)
    assert_equal(got.as_int64()[1].value(), 21)
    assert_true(got.is_null(3), "both operands null, so the sum is null")
    assert_equal(got.null_count(), 1)
