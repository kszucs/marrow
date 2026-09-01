"""The runtime lane on its own terms.

`RuntimeValue` is one struct holding children, a payload and a function
pointer, so its correctness is mostly about **structure**: what it reports about
itself before anything evaluates it. Those answers are what the optimizer reads,
and a wrong one is silent — a plan that narrows the wrong columns still runs.
"""

from std.testing import assert_equal, assert_true

from ....arrays import StructArray, DynArray
from ...bindings import Bindings
from ....builders import array
from ....dtypes import Date32Type, DynType, date32, float64, int32, int64
from ....scalars import DynScalar, Int32Scalar, Int64Scalar, StringScalar
from ....tabular import RecordBatch, record_batch
from ...logical import Shape
from ....builders import (
    Int64Builder,
    ListBuilder,
    PrimitiveBuilder,
    StringBuilder,
)
from ....dtypes import string
from ..values import (
    Payload,
    RuntimeValue,
    abs,
    add,
    and_,
    array_length,
    cast,
    ceil,
    coalesce,
    column,
    date_trunc,
    eq,
    fill_null,
    floor,
    floordiv,
    ge,
    gt,
    ilike,
    if_else,
    is_null,
    is_valid,
    isin,
    le,
    length,
    like,
    literal,
    lt,
    mod,
    month,
    mul,
    ne,
    neg,
    not_,
    nullif,
    lpad,
    left,
    or_,
    position,
    quarter,
    replace,
    right,
    split_part,
    sqrt,
    startswith,
    strip,
    sub,
    substr,
    trim_chars,
    truediv,
    upper,
    xor,
    year,
)


def _batch() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_runtime_shape_is_always_columnar() raises:
    """The lane materialises unconditionally, so it answers truthfully rather
    than aspirationally — `Datum.to_array` never has to broadcast its result."""
    assert_true(RuntimeValue.shape == Shape.columnar)
    assert_true(column("a").shape == Shape.columnar)
    assert_true(literal(DynScalar(Int64Scalar(1))).shape == Shape.columnar)


def test_runtime_columns_are_deduped_in_first_seen_order() raises:
    """Projection pushdown reads this list; a repeat would narrow twice and a
    reorder would build a schema in the wrong order."""
    var v = RuntimeValue(
        "add",
        column("b"),
        RuntimeValue("add", column("a"), column("b")),
    )
    var cols = v.columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "b")
    assert_equal(cols[1], "a")


def test_runtime_dtype_agrees_with_evaluation() raises:
    """The lane that has to look itself up in a schema must agree with what it
    then produces."""
    var b = _batch()
    var v = column("b")
    var produced = (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .dtype()
    )
    assert_true(v.dtype(b.schema) == produced)


def test_runtime_column_reads_the_named_column() raises:
    var b = _batch()
    var got = (
        column("b")
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
    )
    assert_true(got == b.column("b"))


def test_runtime_literal_broadcasts_to_the_batch_length() raises:
    """A literal owes a full column here, unlike the comptime lane's, which
    stays a scalar until something asks."""
    var b = _batch()
    var got = (
        literal(DynScalar(Int64Scalar(7)))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
    )
    assert_equal(len(got), 4)
    assert_true(got == array([7, 7, 7, 7], int64))


def test_runtime_a_subtree_copies_in_constant_time() raises:
    """Children sit behind `ArcPointer`, so copying a plan shares rather than
    clones. Equality of what the copy reports is the observable half."""
    var v = RuntimeValue("add", column("a"), column("b"))
    var w = v.copy()
    assert_equal(len(w.columns()), 2)
    assert_equal(w.columns()[0], "a")
    assert_equal(w.name(), v.name())


# ---------------------------------------------------------------------------
# Comparisons and boolean connectives
# ---------------------------------------------------------------------------


def _bits(v: RuntimeValue) raises -> String:
    """Render a predicate's result as `t`/`f`/`?` per row, over `_batch()`."""
    var b = _batch()
    var got = (
        v.evaluate(b.to_struct_array(), Bindings()).to_array(4).as_bool().copy()
    )
    var out = String()
    for i in range(len(got)):
        if got.is_null(i):
            out += "?"
        else:
            out += "t" if got[i].value() else "f"
    return out^


def _lit(i: Int) -> RuntimeValue:
    return literal(DynScalar(Int64Scalar(Int64(i))))


def test_runtime_all_six_comparisons() raises:
    """`a` is [1, 2, null, 4]; compared against 2.

    All six, not just the ordering pair -- `eq` is what statistics pruning and
    bloom filters key on, and the lane had none of them.
    """
    # a = [1, 2, null, 4]; the null sits at index 2.
    assert_equal(_bits(eq(column("a"), _lit(2))), "ft?f")
    assert_equal(_bits(ne(column("a"), _lit(2))), "tf?t")
    assert_equal(_bits(lt(column("a"), _lit(2))), "tf?f")
    assert_equal(_bits(le(column("a"), _lit(2))), "tt?f")
    assert_equal(_bits(gt(column("a"), _lit(2))), "ff?t")
    assert_equal(_bits(ge(column("a"), _lit(2))), "ft?t")


def test_runtime_comparison_between_two_columns() raises:
    """`a` [1,2,null,4] vs `b` [10,20,30,40] -- null propagates from either."""
    assert_equal(_bits(lt(column("a"), column("b"))), "tt?t")


def test_runtime_boolean_connectives_are_three_valued() raises:
    """Kleene, matching the fused lane: `null AND false` is false, not null."""
    var t = gt(column("b"), _lit(0))  # tttt
    var n = gt(column("a"), _lit(2))  # ff?t
    assert_equal(_bits(and_(t.copy(), n.copy())), "ff?t")
    assert_equal(_bits(or_(t.copy(), n.copy())), "tttt")
    assert_equal(_bits(xor(t.copy(), n.copy())), "tt?f")
    assert_equal(_bits(not_(n.copy())), "tt?f")


def test_runtime_comparison_promotes_mixed_widths() raises:
    """An int64 column against an int32 literal compares rather than raising."""
    var lit32 = literal(DynScalar(Int32Scalar(Int32(2))))
    assert_equal(_bits(gt(column("a"), lit32^)), "ff?t")


def test_runtime_string_comparison_uses_the_string_kernel() raises:
    """The same operator, dispatched on the runtime dtype."""
    var sb = StringBuilder(4)
    sb.append("apple")
    sb.append("pear")
    sb.append_null()
    sb.append("quince")
    var b = record_batch([sb.finish().to_dyn()], names=["s"])
    var pred = lt(column("s"), literal(DynScalar(StringScalar("pear"))))
    var got = (
        pred.evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())  # "apple" < "pear"
    assert_true(not got[1].value())  # "pear" == "pear"
    assert_true(got.is_null(2))
    assert_true(not got[3].value())  # "quince" > "pear"


def test_runtime_predicate_reports_its_columns() raises:
    """What the optimizer reads before anything evaluates."""
    var p = and_(gt(column("a"), _lit(1)), lt(column("b"), _lit(99)))
    var cols = String()
    for ref c in p.columns():
        cols += c
        cols += ","
    assert_equal(cols, "a,b,")


# ---------------------------------------------------------------------------
# Arithmetic, math, strings, temporal — the surface the Python frontend binds
# ---------------------------------------------------------------------------
#
# These are the tags the lane grew so a query built at run time can express
# what the comptime lane already could. Each case pins the *answer*, not the
# rendering: a tag that reaches the wrong kernel still renders correctly.


def _ints(v: RuntimeValue) raises -> DynArray:
    """Evaluate over `_batch()` — `a` is [1, 2, null, 4], `b` [10, 20, 30, 40].
    """
    var b = _batch()
    return v.evaluate(b.to_struct_array(), Bindings()).to_array(4)


def test_runtime_arithmetic_over_two_columns() raises:
    assert_true(
        _ints(add(column("b"), column("b"))) == array([20, 40, 60, 80], int64)
    )
    assert_true(
        _ints(sub(column("b"), _lit(10))) == array([0, 10, 20, 30], int64)
    )
    assert_true(
        _ints(mul(column("b"), _lit(2))) == array([20, 40, 60, 80], int64)
    )
    assert_true(
        _ints(floordiv(column("b"), _lit(3))) == array([3, 6, 10, 13], int64)
    )
    assert_true(_ints(mod(column("b"), _lit(3))) == array([1, 2, 0, 1], int64))


def test_runtime_arithmetic_propagates_nulls() raises:
    """Null in, null out — `a`'s null at index 2 survives the add.

    Checked per element rather than against `array([11, 22, None, 44])`:
    `__eq__` is structural, and the kernel computes every lane before masking,
    so the byte under the null holds `0 + 30` while the builder writes 0
    there. Both are valid Arrow — the value under a null is unspecified — so
    the assertion has to read the validity, not the buffer.
    """
    var got = _ints(add(column("a"), column("b"))).as_int64().copy()
    assert_equal(Int(got[0].value()), 11)
    assert_equal(Int(got[1].value()), 22)
    assert_true(got.is_null(2))
    assert_equal(Int(got[3].value()), 44)


def test_runtime_division_is_always_float64() raises:
    """`5 / 2` is 2.5, not 2 — `FloatBinary`'s rule, and the reason marrow
    diverges from `pc.divide` on two integer columns."""
    var got = _ints(truediv(column("b"), _lit(4)))
    assert_true(got.dtype() == DynType(float64))
    assert_true(got == array([2.5, 5.0, 7.5, 10.0], float64))


def test_runtime_arithmetic_widens_rather_than_narrows() raises:
    """An int32 column plus a literal too large for int32 must widen to int64.

    The earlier promotion rule cast the right operand to the left's type and
    fell back on failure, so this raised instead of adding.
    """
    var b = record_batch([array([1, 2], int32).to_dyn()], names=["c"])
    var big = literal(DynScalar(Int64Scalar(Int64(1) << 40)))
    var v = add(column("c"), big^)
    var got = v.evaluate(b.to_struct_array(), Bindings()).to_array(2)
    assert_true(got.dtype() == DynType(int64))
    assert_true(got == array([(1 << 40) + 1, (1 << 40) + 2], int64))


def test_runtime_arithmetic_rejects_non_numeric_operands() raises:
    """A clear message at the operand, rather than a kernel's "no arm matched".
    """
    var sb = StringBuilder(1)
    sb.append("x")
    var b = record_batch([sb.finish().to_dyn()], names=["s"])
    var v = add(column("s"), column("s"))
    # A string `+` is concatenation, so it must *not* raise.
    var joined = v.evaluate(b.to_struct_array(), Bindings()).to_array(1)
    assert_true(joined.dtype().is_string_like())

    var bad = sub(column("s"), column("s"))
    var raised = False
    try:
        _ = bad.evaluate(b.to_struct_array(), Bindings())
    except:
        raised = True
    assert_true(raised)


def _over(b: RecordBatch, var v: RuntimeValue) raises -> DynArray:
    """Evaluate `v` over `b`. A free function rather than a closure per test:
    a nested `def` capturing the batch needs an explicit capture list, and
    four of them would say the same thing four ways."""
    return v.evaluate(b.to_struct_array(), Bindings()).to_array(b.num_rows())


def test_runtime_unary_math() raises:
    var b = record_batch(
        [array([-2.5, 2.5, -1.0, 4.0], float64).to_dyn()], names=["f"]
    )

    assert_true(
        _over(b, neg(column("f"))) == array([2.5, -2.5, 1.0, -4.0], float64)
    )
    assert_true(
        _over(b, abs(column("f"))) == array([2.5, 2.5, 1.0, 4.0], float64)
    )
    assert_true(
        _over(b, floor(column("f"))) == array([-3.0, 2.0, -1.0, 4.0], float64)
    )
    assert_true(
        _over(b, ceil(column("f"))) == array([-2.0, 3.0, -1.0, 4.0], float64)
    )
    assert_true(_over(b, sqrt(column("f")).copy()).dtype() == DynType(float64))


def test_runtime_float_unary_casts_an_integer_operand() raises:
    """`sqrt`/`exp`/`ln` are float64 out whatever went in — `FloatUnary`."""
    var got = _ints(sqrt(column("b")))
    assert_true(got.dtype() == DynType(float64))


def test_runtime_null_predicates_read_validity() raises:
    """`is_null` is never null itself; that is what separates it from a
    comparison against NULL."""
    assert_equal(_bits(is_null(column("a"))), "fftf")
    assert_equal(_bits(is_valid(column("a"))), "ttft")


def test_runtime_fill_null_and_nullif() raises:
    assert_true(
        _ints(fill_null(column("a"), _lit(0))) == array([1, 2, 0, 4], int64)
    )
    var nulled = _ints(nullif(column("a"), _lit(2))).as_int64().copy()
    assert_equal(Int(nulled[0].value()), 1)
    assert_true(nulled.is_null(1))  # equal to 2 -> nulled
    assert_true(nulled.is_null(2))  # already null -> stays null
    assert_equal(Int(nulled[3].value()), 4)


def test_runtime_coalesce_and_if_else() raises:
    assert_true(
        _ints(coalesce([column("a"), column("b")]))
        == array([1, 2, 30, 4], int64)
    )
    var v = if_else(gt(column("b"), _lit(20)), column("b"), _lit(0))
    assert_true(_ints(v^) == array([0, 0, 30, 40], int64))


def test_runtime_isin_hashes_the_set_once() raises:
    """The value set is a payload, not a child — one probe table per batch."""
    var v = isin(column("b"), array([20, 40], int64).to_dyn())
    assert_equal(_bits(v^), "ftft")


def test_runtime_cast_carries_its_safety_on_the_tag() raises:
    """`Payload` holds one value and the target dtype is already it, so `safe`
    rides the tag. Both spellings must reach the same kernel."""
    var safe = cast(column("b"), DynType(int32))
    assert_true(_ints(safe^).dtype() == DynType(int32))
    var unsafe = cast(column("b"), DynType(int32), safe=False)
    assert_true(_ints(unsafe^).dtype() == DynType(int32))


def test_runtime_string_verbs() raises:
    var sb = StringBuilder(3)
    sb.append("  Apple ")
    sb.append("pear")
    sb.append_null()
    var b = record_batch([sb.finish().to_dyn()], names=["s"])

    var up = _over(b, upper(column("s"))).as_string().copy()
    assert_equal(up[0].value(), "  APPLE ")
    var st = _over(b, strip(column("s"))).as_string().copy()
    assert_equal(st[0].value(), "Apple")
    var ln_ = _over(b, length(column("s"))).as_int32().copy()
    assert_equal(Int(ln_[0].value()), 8)
    assert_true(ln_.is_null(2))

    var pred = _over(b, like(column("s"), "%ear")).as_bool().copy()
    assert_true(not pred[0].value())
    assert_true(pred[1].value())
    var ipred = _over(b, ilike(column("s"), "%APPLE%")).as_bool().copy()
    assert_true(ipred[0].value())

    var sw = _over(
        b, startswith(column("s"), literal(DynScalar(StringScalar("pe"))))
    )
    assert_true(sw.as_bool()[1].value())


def test_runtime_temporal_verbs() raises:
    """`date32` days since epoch: 0 is 1970-01-01, 19723 is 2024-01-01."""
    var db = PrimitiveBuilder[Date32Type](date32(), capacity=2)
    db.append(Int32(0))
    db.append(Int32(19723))
    var b = record_batch([db.finish().to_dyn()], names=["d"])

    var y = _over(b, year(column("d"))).as_int32().copy()
    assert_equal(Int(y[0].value()), 1970)
    assert_equal(Int(y[1].value()), 2024)
    var m = _over(b, month(column("d"))).as_int32().copy()
    assert_equal(Int(m[1].value()), 1)
    var q = _over(b, quarter(column("d"))).as_int32().copy()
    assert_equal(Int(q[1].value()), 1)
    # date_trunc keeps the input's type rather than widening it.
    assert_true(
        _over(b, date_trunc(column("d"), "year")).dtype() == DynType(date32())
    )


def test_runtime_date_trunc_rejects_a_bad_unit_at_construction() raises:
    """The unit is parsed when the plan is built, not on the first row that
    evaluates it — which is why `CalendarUnit` is a type and not a `String`."""
    var raised = False
    try:
        _ = date_trunc(column("d"), "fortnight")
    except:
        raised = True
    assert_true(raised)


def test_runtime_array_length_over_a_list_column() raises:
    var lb = ListBuilder(Int64Builder())
    var child_any = lb.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    child.append(3)
    lb.append_valid()  # [1, 2, 3]
    lb.append_valid()  # []
    lb.append_null()  # null
    var b = record_batch([lb.finish().to_dyn()], names=["xs"])
    var got = (
        array_length(column("xs"))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(3)
        .as_int32()
        .copy()
    )
    assert_equal(Int(got[0].value()), 3)
    assert_equal(Int(got[1].value()), 0)
    assert_true(got.is_null(2))


def test_runtime_unknown_tag_names_itself() raises:
    """The one error message `evaluate` still owns after the split into
    `_unary` / `_binary` — both of which answer `None` rather than raising so
    this stays in one place."""
    var v = RuntimeValue("frobnicate", column("a"))
    var b = _batch()
    var msg = String()
    try:
        _ = v.evaluate(b.to_struct_array(), Bindings())
    except e:
        msg = String(e)
    assert_true("frobnicate" in msg)


def test_runtime_conditionals_unify_mixed_branches() raises:
    """`coalesce`, `case_when`, `nullif` and `fill_null` all pick one of their
    inputs per row, so their kernels call `expect_same_dtype`.

    Without unification `float_col.fill_null(0)` raises on a literal that
    inferred `int64` — which is exactly what a Python caller writes.
    """
    var b = record_batch(
        [array([1.5, None, 3.5], float64).to_dyn()], names=["f"]
    )
    var int_lit = literal(DynScalar(Int64Scalar(0)))

    var filled = _over(b, fill_null(column("f"), int_lit.copy()))
    assert_true(filled.dtype() == DynType(float64))
    assert_true(filled == array([1.5, 0.0, 3.5], float64))

    var merged_ = _over(b, coalesce([column("f"), int_lit.copy()]))
    assert_true(merged_.dtype() == DynType(float64))

    var chosen = _over(
        b, if_else(is_null(column("f")), int_lit.copy(), column("f"))
    )
    assert_true(chosen.dtype() == DynType(float64))
    assert_true(chosen == array([1.5, 0.0, 3.5], float64))


def test_runtime_conditionals_leave_a_hopeless_mix_to_the_kernel() raises:
    """`coalesce(string, int)` has no common type, so the guess is not made
    here — the kernel raises and its message names both dtypes."""
    var sb = StringBuilder(1)
    sb.append("x")
    var b = record_batch(
        [sb.finish().to_dyn(), array([1], int64).to_dyn()], names=["s", "n"]
    )
    var raised = False
    try:
        _ = _over(b, coalesce([column("s"), column("n")]))
    except:
        raised = True
    assert_true(raised)


def _text() raises -> RecordBatch:
    """The SQL string surface's fixture: a comma, no comma, and a null."""
    var sb = StringBuilder(3)
    sb.append("alpha,beta")
    sb.append("gamma")
    sb.append_null()
    return record_batch(
        [sb.finish().to_dyn(), array([2, 1, 1], int64).copy()],
        names=["s", "n"],
    )


def test_runtime_sql_string_functions_are_reachable() raises:
    """All nine argument-taking string functions, in the lane that could not
    spell one of them while their arguments were constants on the node.

    This is the whole point of the operand carrier: a `RuntimeValue` has
    nowhere to put a constant a kernel takes directly, and it has always had
    somewhere to put a child.
    """
    var b = _text()

    var sub_ = (
        _over(b, substr(column("s"), _lit(1), _lit(5))).as_string().copy()
    )
    assert_equal(sub_[0].value(), "alpha")
    assert_true(sub_.is_null(2))

    var l = _over(b, left(column("s"), _lit(5))).as_string().copy()
    assert_equal(l[0].value(), "alpha")

    var r = _over(b, right(column("s"), _lit(4))).as_string().copy()
    assert_equal(r[0].value(), "beta")

    var lp = _over(b, lpad(column("s"), _lit(12), _str("*"))).as_string().copy()
    assert_equal(lp[0].value(), "**alpha,beta")

    var rep = (
        _over(b, replace(column("s"), _str(","), _str("-"))).as_string().copy()
    )
    assert_equal(rep[0].value(), "alpha-beta")

    var sp = (
        _over(b, split_part(column("s"), _str(","), _lit(2))).as_string().copy()
    )
    assert_equal(sp[0].value(), "beta")

    # Both ends: the trailing "a" of "beta" is a set member too.
    var tr = _over(b, trim_chars(column("s"), _str("aplh"))).as_string().copy()
    assert_equal(tr[0].value(), ",bet")

    var pos = _over(b, position(column("s"), _str(","))).as_int64().copy()
    assert_equal(Int(pos[0].value()), 6)
    assert_equal(Int(pos[1].value()), 0)
    assert_true(pos.is_null(2))


def test_runtime_sql_string_arguments_can_be_columns() raises:
    """`substr(s, 1, n)` with `n` a column — the expression the constants
    version had no way to build in either lane."""
    var b = _text()
    var got = (
        _over(b, substr(column("s"), _lit(1), column("n"))).as_string().copy()
    )
    assert_equal(got[0].value(), "al")
    assert_equal(got[1].value(), "g")
    assert_true(got.is_null(2))


def test_runtime_sql_string_null_argument_nulls_the_row() raises:
    """A null argument makes the row null, as DuckDB does — and only its own
    row, which is what separates it from a null constant."""
    var nb = Int64Builder(capacity=3)
    nb.append(Int64(2))
    nb.append_null()
    nb.append(Int64(2))
    var b = record_batch(
        [
            array(["abc", "abc", "abc"]).to_dyn(),
            nb.finish().to_dyn(),
        ],
        names=["s", "n"],
    )
    var got = _over(b, left(column("s"), column("n"))).as_string().copy()
    assert_equal(got[0].value(), "ab")
    assert_true(got.is_null(1))
    assert_equal(got[2].value(), "ab")


def test_runtime_sql_string_arguments_promote_their_width() raises:
    """An `int32` count reaches an `int64` slot: this lane normalises operands
    with `cast`, as `_arith` and `_compare` already do, so the carrier is
    instantiated once rather than per integer width."""
    var b = record_batch(
        [
            array(["abcde"]).to_dyn(),
            array([3], int32).to_dyn(),
        ],
        names=["s", "n"],
    )
    var got = _over(b, left(column("s"), column("n"))).as_string().copy()
    assert_equal(got[0].value(), "abc")


def _str(var v: String) -> RuntimeValue:
    """A string literal operand — `lit` for the string half."""
    return literal(DynScalar(StringScalar(v^)))
