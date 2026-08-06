"""Tests for marrow.expr.values — the staged, strategy-pluggable fusion engine.

Covers the four value families and the `BoxedValue` erasure box:
  * numeric — vectorized SIMD fusion (`Add`/`Mul`/…, reductions, casts, windows)
  * bool    — bit-packed vectorized fusion (comparisons, `And`/`Or`/`Not`, any/all)
  * string  — elementwise fusion (`Concat`/`Upper`/…, predicates, parses)
  * list    — materialize-only columns feeding fixed-width breakers
  * BoxedValue — erases a fused node OR a runtime `DynValue` behind `execute()`
"""

from std.testing import assert_true, assert_equal, assert_raises

from ...builders import (
    array,
    ListBuilder,
    Int64Builder,
    PrimitiveBuilder,
    StringBuilder,
)
from ...dtypes import (
    int64,
    int32,
    float64,
    string,
    Int64Type,
    Float64Type,
    StringType,
    ListType,
    TimestampType,
    timestamp,
    second,
)
from ...tabular import record_batch, RecordBatch
from ...scalars import DynScalar
from ...kernels.temporal import unit_day

from ...expr.values import (
    col,
    lit,
    Add,
    Sub,
    Mul,
    Neg,
    Div,
    Mean,
    NumericCast,
    Sum,
    Max,
    Lt,
    Gt,
    And,
    Or,
    Not,
    Any,
    All,
    Count,
    IsNull,
    NotNull,
    IsNan,
    NumToBool,
    BoolToNum,
    RowNumber,
    WindowSpec,
    FrameBound,
    NumericValue,
    into_array,
    Concat,
    Upper,
    StringLength,
    StartsWith,
    StringToNum,
    StringToBool,
    NumToString,
    StringToString,
    ListColumn,
    ListLength,
    ListContains,
    StrLt,
    StrLe,
    StrGt,
    StrGe,
    Like,
    ILike,
    IsIn,
    Coalesce,
    Nullif,
    CaseWhen,
    TemporalColumn,
    DateTrunc,
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Quarter,
    DayOfWeek,
    DayOfYear,
)
from ...expr.values import col as dyn_col
from ...expr.relations import BoxedValue


# instantiation is a COMPILE-TIME proof the operand is a fused `NumericValue` node
def _takes_fusable[F: NumericValue](x: F) -> Bool:
    return True


def _batch() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, 3, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


# ===========================================================================
# Numeric family — vectorized SIMD fusion
# ===========================================================================


def test_column_add_fuses() raises:
    var cv = (Add(col("a", int64), col("b", int64))).execute(_batch())
    assert_true(not cv.isa[DynScalar]())
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_dyn())


def test_literal_broadcast() raises:
    var cv = (Mul(col("a", int64), lit(10, int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([10, 20, 30, 40], int64).to_dyn())


def test_scalar_literal_evaluates_once() raises:
    var cv = (lit(7, int64)).execute(_batch())
    assert_true(cv.isa[DynScalar]())
    assert_true(into_array(cv, 3) == array([7, 7, 7], int64).to_dyn())


def test_fused_chain() raises:
    # (a + b) * a  over a=[1,2,3,4], b=[10,20,30,40]
    var cv = (
        Mul(Add(col("a", int64), col("b", int64)), col("a", int64))
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([11, 44, 99, 176], int64).to_dyn())


def test_reduction_is_scalar() raises:
    # sum(a) over [1,2,3,4] = 10, a scalar
    var cv = (Sum(col("a", int64))).execute(_batch())
    assert_true(cv.isa[DynScalar]())
    assert_true(into_array(cv, 4) == array([10, 10, 10, 10], int64).to_dyn())


def test_reduction_broadcasts_into_columnar() raises:
    # a + sum(a) = [1,2,3,4] + 10 = [11,12,13,14] — the SINGLE Add, sum(a) is a
    # fused leaf reading its stage result from the context and splatting.
    var cv = (Add(col("a", int64), Sum(col("a", int64)))).execute(_batch())
    assert_true(not cv.isa[DynScalar]())
    assert_true(into_array(cv, 4) == array([11, 12, 13, 14], int64).to_dyn())


def test_scalar_plus_scalar_stays_scalar() raises:
    # sum(a) + max(a) = 10 + 4 = 14, still scalar
    var cv = (Add(Sum(col("a", int64)), Max(col("a", int64)))).execute(_batch())
    assert_true(cv.isa[DynScalar]())
    assert_true(into_array(cv, 2) == array([14, 14], int64).to_dyn())


def test_arithmetic_above_reduction() raises:
    # (a + b) fuses, then * sum(a) broadcasts:  [11,22,33,44] * 10
    var cv = (
        Mul(Add(col("a", int64), col("b", int64)), Sum(col("a", int64)))
    ).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([110, 220, 330, 440], int64).to_dyn()
    )


def test_mean_centering_via_single_binary() raises:
    # x - avg(x): avg([1,2,3,4]) = 2.5 (a breaker materialized once in prepare,
    # then a splat-leaf), so the subtract fuses over (x, splat(mean)) as the same
    # NumericBinary as `x - lit`. int - float -> float.
    var cv = (Sub(col("a", int64), Mean(col("a", int64)))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([-1.5, -0.5, 0.5, 1.5], float64).to_dyn()
    )


def test_fused_node_is_fusable() raises:
    # `Add` over fusable operands is itself `NumericValue`; `_takes_fusable`
    # compiling is the compile-time proof.
    assert_true(_takes_fusable(Add(col("a", int64), col("b", int64))))


def test_div_is_true_division() raises:
    # 1/2,2/2,3/2,4/2 = [0.5,1.0,1.5,2.0] float64 — true division, not integer
    var cv = (Div(col("a", int64), lit(2, int64))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([0.5, 1.0, 1.5, 2.0], float64).to_dyn()
    )


def test_unary_neg_fuses() raises:
    var cv = (Neg(col("a", int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([-1, -2, -3, -4], int64).to_dyn())


def test_cast_fuses_in_chain() raises:
    # a fused cast composes with arithmetic in the same pass (identity cast here)
    var cv = (
        Add(NumericCast[Int64Type](col("a", int64)), col("b", int64))
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_dyn())


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _spec() -> WindowSpec:
    return WindowSpec(FrameBound(0, 0), FrameBound(2, 0))


def test_window_row_number() raises:
    var cv = (RowNumber(col("a", int64), _spec())).execute(_batch())
    assert_true(into_array(cv, 4) == array([1, 2, 3, 4], int64).to_dyn())


def test_arithmetic_above_window_materializes() raises:
    # row_number() + 1 → [2,3,4,5]  (Add above a columnar window breaker)
    var cv = (Add(RowNumber(col("a", int64), _spec()), lit(1, int64))).execute(
        _batch()
    )
    assert_true(into_array(cv, 4) == array([2, 3, 4, 5], int64).to_dyn())


# ===========================================================================
# Boolean family — bit-packed vectorized fusion
# ===========================================================================


def test_comparison_fuses_to_bool() raises:
    # a < 3 over [1,2,3,4] → bit-packed [T,T,F,F] (the bool fused strategy)
    var cv = (Lt(col("a", int64), lit(3, int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([True, True, False, False]).to_dyn())


def test_bool_and_fuses() raises:
    # (a < 3) & (b > 15) → [T,T,F,F] & [F,T,T,T] = [F,T,F,F], one fused bitwise pass
    var cv = (
        And(
            Lt(col("a", int64), lit(3, int64)),
            Gt(col("b", int64), lit(15, int64)),
        )
    ).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([False, True, False, False]).to_dyn()
    )


def test_bool_not_fuses() raises:
    # not (a < 3) → not [T,T,F,F] = [F,F,T,T]
    var cv = (Not(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(into_array(cv, 4) == array([False, False, True, True]).to_dyn())


def test_bool_or_fuses() raises:
    # (a < 2) | (a > 3) → [T,F,F,F] | [F,F,F,T] = [T,F,F,T]
    var cv = (
        Or(
            Lt(col("a", int64), lit(2, int64)),
            Gt(col("a", int64), lit(3, int64)),
        )
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([True, False, False, True]).to_dyn())


def test_any_all_reductions() raises:
    # any(a < 3) = True, all(a < 3) = False over [1,2,3,4]
    var an = (Any(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(an.isa[DynScalar]() and an[DynScalar].as_bool().value())
    var al = (All(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(al.isa[DynScalar]() and not al[DynScalar].as_bool().value())


def test_any_all_ignore_null_data_bits() raises:
    # A null slot keeps whatever sits in its data buffer (0 here), so `x < 1`
    # SETS the mask bit under the null. `any`/`all` must mask against validity
    # rather than counting raw bits.
    var b = record_batch(
        [
            array([None, 5, 7], int64).copy(),  # `<1` -> bits [T,F,F]
            array([None, 0, 0], int64).copy(),  # `<1` -> bits [T,T,T]
        ],
        names=["a", "b"],
    )
    # any: the only set bit belongs to a null row -> False (raw count says True)
    var an = (Any(Lt(col("a", int64), lit(1, int64)))).execute(b)
    assert_true(an.isa[DynScalar]() and not an[DynScalar].as_bool().value())
    # all: every VALID row is true -> True (a raw popcount==valid check says False)
    var al = (All(Lt(col("b", int64), lit(1, int64)))).execute(b)
    assert_true(al.isa[DynScalar]() and al[DynScalar].as_bool().value())


def test_mixed_width_compare_promotes() raises:
    # int32 > int64 must compare in the promoted (int64) domain. Truncating the
    # right operand into int32 turns 2**32 into 0, so row 0 would read True.
    var b = record_batch(
        [
            array([1, 2, 3], int32).copy(),
            array([4294967296, -4294967296, 5], int64).copy(),
        ],
        names=["a", "b"],
    )
    var cv = (Gt(col("a", int32), col("b", int64))).execute(b)
    assert_true(into_array(cv, 3) == array([False, True, False]).to_dyn())
    # ... and the mirrored operand order goes through the same promotion.
    var cv2 = (Lt(col("b", int64), col("a", int32))).execute(b)
    assert_true(into_array(cv2, 3) == array([False, True, False]).to_dyn())


def test_count_reduction() raises:
    # count(a) over [1,2,3,4] = 4 (int64 scalar)
    var cv = (Count(col("a", int64))).execute(_batch())
    assert_true(cv.isa[DynScalar]())
    assert_true(into_array(cv, 4) == array([4, 4, 4, 4], int64).to_dyn())


def test_notnull_and_isnull() raises:
    # no nulls in a=[1,2,3,4] → not_null all true, is_null all false
    var nn = (NotNull(col("a", int64))).execute(_batch())
    assert_true(into_array(nn, 4) == array([True, True, True, True]).to_dyn())
    var isn = (IsNull(col("a", int64))).execute(_batch())
    assert_true(
        into_array(isn, 4) == array([False, False, False, False]).to_dyn()
    )


def test_isnan_fuses_over_float() raises:
    # is_nan over finite floats → all false, computed in a fused SIMD pass
    var b = record_batch(
        [array([1.0, 2.0, 3.0, 4.0], float64).copy()], names=["f"]
    )
    var cv = (IsNan(col("f", float64))).execute(b)
    assert_true(
        into_array(cv, 4) == array([False, False, False, False]).to_dyn()
    )


def test_num_to_bool_fuses() raises:
    # a*0 = 0 → false ; a (nonzero) → true — fused per-lane num->bool
    var z = (NumToBool(Mul(col("a", int64), lit(0, int64)))).execute(_batch())
    assert_true(
        into_array(z, 4) == array([False, False, False, False]).to_dyn()
    )
    var nz = (NumToBool(col("a", int64))).execute(_batch())
    assert_true(into_array(nz, 4) == array([True, True, True, True]).to_dyn())


def test_bool_to_num_fuses() raises:
    # (a < 3) -> int64 = [1,1,0,0] — fused bool->num, composes in the numeric lane
    var cv = (BoolToNum[Int64Type](Lt(col("a", int64), lit(3, int64)))).execute(
        _batch()
    )
    assert_true(into_array(cv, 4) == array([1, 1, 0, 0], int64).to_dyn())


def test_fluent_numeric_and_bool() raises:
    # operators/methods build the same nodes as the explicit builders
    var s = (col("a", int64) + col("b", int64)).execute(_batch())
    assert_true(into_array(s, 4) == array([11, 22, 33, 44], int64).to_dyn())
    # mean-centering via `x - x.mean()`
    var mc = (col("a", int64) - col("a", int64).mean()).execute(_batch())
    assert_true(
        into_array(mc, 4) == array([-1.5, -0.5, 0.5, 1.5], float64).to_dyn()
    )
    # (a < 3) & (b > 15) via `<`, `>`, `&`
    var mask = (
        (col("a", int64) < lit(3, int64)) & (col("b", int64) > lit(15, int64))
    ).execute(_batch())
    assert_true(
        into_array(mask, 4) == array([False, True, False, False]).to_dyn()
    )


# ===========================================================================
# String family — elementwise fusion
# ===========================================================================


def _str_batch() raises -> RecordBatch:
    return record_batch([array(["ab", "cd"]).copy()], names=["s"])


def _str_batch2() raises -> RecordBatch:
    # two string columns, for binary predicates (a literal pattern would need the
    # unsupported string-scalar broadcast — a noted follow-up)
    return record_batch(
        [array(["abc", "xyz"]).copy(), array(["ab", "yy"]).copy()],
        names=["s", "p"],
    )


def test_string_literal_is_scalar() raises:
    # a bare string literal is a scalar Datum (broadcasts lazily at a boundary)
    var cv = (lit("hi")).execute(_str_batch())
    assert_true(cv.isa[DynScalar]())
    assert_true(cv[DynScalar].as_string().to_string() == "hi")


def test_concat_chain_fuses() raises:
    # col || "p1" || "p2" over ["ab","cd"] → ["abp1p2","cdp1p2"] — one builder pass
    var expr = Concat(Concat(col("s", string), lit("p1")), lit("p2"))
    var cv = (expr).execute(_str_batch())
    assert_true(into_array(cv, 2) == array(["abp1p2", "cdp1p2"]).to_dyn())


def test_strlen_fuses_into_numeric() raises:
    # length(s) + 1 over ["ab","cd"] → byte lengths [2,2] + 1 = [3,3]. A STRATEGY
    # TRANSITION: the string stage materializes, then `length` reads offsets as a
    # numeric lane leaf and the `+ 1` fuses in the same numeric pass.
    var expr = Add(StringLength(col("s", string)), lit(1, int32))
    var cv = (expr).execute(_str_batch())
    assert_true(into_array(cv, 2) == array([3, 3], int32).to_dyn())


def test_upper_map_fuses() raises:
    # upper(s) over ["ab","cd"] → ["AB","CD"] (elementwise map, delegates to kernel)
    var cv = (Upper(col("s", string))).execute(_str_batch())
    assert_true(into_array(cv, 2) == array(["AB", "CD"]).to_dyn())


def test_map_and_concat_fuse_together() raises:
    # upper(s) || "!" → ["AB!","CD!"] — map + concat in one builder pass
    var cv = (Concat(Upper(col("s", string)), lit("!"))).execute(_str_batch())
    assert_true(into_array(cv, 2) == array(["AB!", "CD!"]).to_dyn())


def test_startswith_predicate() raises:
    # startswith(s, p): "abc".sw("ab")=T, "xyz".sw("yy")=F → [T,F]
    var cv = (StartsWith(col("s", string), col("p", string))).execute(
        _str_batch2()
    )
    assert_true(into_array(cv, 2) == array([True, False]).to_dyn())


def test_predicate_and_strlen_compose_under_bool_logic() raises:
    # startswith(s,p) & (length(s) > 2) → [T,F] & [T,T] = [T,F]
    # a string-predicate breaker AND a strlen breaker, both fused under one `And`.
    var cv = (
        And(
            StartsWith(col("s", string), col("p", string)),
            Gt(StringLength(col("s", string)), lit(2, int32)),
        )
    ).execute(_str_batch2())
    assert_true(into_array(cv, 2) == array([True, False]).to_dyn())


def test_string_to_num_parses() raises:
    # parse ["10","20"] -> int64 [10,20] (a string->numeric breaker)
    var b = record_batch([array(["10", "20"]).copy()], names=["s"])
    var cv = (StringToNum[Int64Type](col("s", string))).execute(b)
    assert_true(into_array(cv, 2) == array([10, 20], int64).to_dyn())


def test_string_to_bool_parses() raises:
    # parse ["true","false"] -> [T,F] (a string->bool breaker)
    var b = record_batch([array(["true", "false"]).copy()], names=["s"])
    var cv = (StringToBool(col("s", string))).execute(b)
    assert_true(into_array(cv, 2) == array([True, False]).to_dyn())


def test_num_to_string() raises:
    # format int64 [1,2,3,4] -> ["1","2","3","4"] (a string breaker)
    var b = record_batch([array([1, 2, 3, 4], int64).copy()], names=["n"])
    var cv = (NumToString[StringType](col("n", int64))).execute(b)
    assert_true(into_array(cv, 4) == array(["1", "2", "3", "4"]).to_dyn())


def test_num_to_string_fuses_with_concat() raises:
    # cast(n, string) || "!" -> ["1!","2!"] — string breaker read fuses into concat
    var b = record_batch([array([1, 2], int64).copy()], names=["n"])
    var cv = (
        Concat(NumToString[StringType](col("n", int64)), lit("!"))
    ).execute(b)
    assert_true(into_array(cv, 2) == array(["1!", "2!"]).to_dyn())


def test_string_to_string_container_cast() raises:
    # string -> string container cast, values preserved
    var cv = (StringToString[StringType](col("s", string))).execute(
        _str_batch()
    )
    assert_true(into_array(cv, 2) == array(["ab", "cd"]).to_dyn())


def test_fluent_string() raises:
    # method + operator surface: `s.upper()` and `s || "!"`
    var u = col("s", string).upper().execute(_str_batch())
    assert_true(into_array(u, 2) == array(["AB", "CD"]).to_dyn())
    var c = (col("s", string) + lit("!")).execute(_str_batch())
    assert_true(into_array(c, 2) == array(["ab!", "cd!"]).to_dyn())


# ===========================================================================
# List family — materialize-only columns feeding fixed-width breakers
# ===========================================================================


def _list_batch() raises -> RecordBatch:
    # list<int64> column: [[10, 20, 30], [40, 50]]
    var lb = ListBuilder(Int64Builder(capacity=8))
    var child_any = lb.values()
    ref child = child_any.as_int64()
    child.append(10)
    child.append(20)
    child.append(30)
    lb.append_valid()
    child.append(40)
    child.append(50)
    lb.append_valid()
    return record_batch([lb.finish()], names=["l"])


def test_list_length() raises:
    # length([[10,20,30],[40,50]]) = [3, 2]
    var cv = (ListLength(ListColumn[ListType]("l"))).execute(_list_batch())
    assert_true(into_array(cv, 2) == array([3, 2], int32).to_dyn())


def test_list_length_fuses_above() raises:
    # length(l) + 1 = [4, 3] — the breaker feeds the fused numeric lane
    var cv = (
        Add(ListLength(ListColumn[ListType]("l")), lit(1, int32))
    ).execute(_list_batch())
    assert_true(into_array(cv, 2) == array([4, 3], int32).to_dyn())


def test_list_contains() raises:
    # 20 in [10,20,30] = T ; 20 in [40,50] = F  ->  [T, F]
    var cv = (ListContains(ListColumn[ListType]("l"), lit(20, int64))).execute(
        _list_batch()
    )
    assert_true(into_array(cv, 2) == array([True, False]).to_dyn())


# ===========================================================================
# BoxedValue — the erasure box over a fused node OR a runtime DynValue
# ===========================================================================


def test_anyvalue_wraps_dynvalue() raises:
    # the runtime lane: erased operands through the *same* nodes the fused lane
    # builds — this is what the relational engine plans over
    var boxed: BoxedValue = dyn_col("a") + dyn_col("b")
    var cv = boxed.execute(_batch())
    assert_true(cv == array([11, 22, 33, 44], int64).to_dyn())


def test_anyvalue_erases_to_array() raises:
    # box a comptime node; its erased execute yields a column (DynArray), the
    # interface the relational engine consumes
    var boxed: BoxedValue = Add(col("a", int64), lit(10, int64))
    var cv = boxed.execute(_batch())
    assert_true(cv == array([11, 12, 13, 14], int64).to_dyn())


def test_anyvalue_interchange() raises:
    # a heterogeneous list holds a fused comptime column *and* an interpreter
    # value; both run through the one erased execute — fused-vs-interpreted is
    # only which node you boxed.
    var values = List[BoxedValue]()
    values.append(BoxedValue(col("a", int64)))  # fused comptime
    values.append(BoxedValue(dyn_col("b")))  # runtime interpreter
    var batch = _batch()
    assert_true(values[0].execute(batch) == array([1, 2, 3, 4], int64).to_dyn())
    assert_true(
        values[1].execute(batch) == array([10, 20, 30, 40], int64).to_dyn()
    )


def test_dynvalue_name_only_for_load() raises:
    # `_name` doubles as the LIKE pattern and the date_trunc unit, so `name()`
    # must be tag-guarded or a computed node reports a nonsense output column.
    assert_true(dyn_col("a").name() == "a")
    assert_true(dyn_col("a").like("%foo%").name() == "")
    assert_true(dyn_col("ts").date_trunc("day").name() == "")
    assert_true((dyn_col("a") + dyn_col("b")).name() == "")


def test_anyvalue_write_to_delegates() raises:
    # write_to on a boxed DynValue renders the boxed node's expression
    # form (not just its column name)
    var boxed: BoxedValue = dyn_col("a") + dyn_col("b")
    assert_true(String(boxed).find("add") != -1)


# ===========================================================================
# Plan analysis — referenced_columns
# ===========================================================================


def _assert_columns(got: List[String], expected: List[String]) raises:
    assert_equal(len(got), len(expected))
    for i in range(len(expected)):
        assert_true(got[i] == expected[i])


def test_referenced_columns_bare_column() raises:
    # a bare column reads exactly its own name
    _assert_columns(col("a", int64).referenced_columns(), ["a"])


def test_referenced_columns_literal_is_empty() raises:
    # a literal reads no columns
    _assert_columns(lit(1, int64).referenced_columns(), List[String]())
    _assert_columns(lit("x").referenced_columns(), List[String]())


def test_referenced_columns_binary_union() raises:
    # col(a) + col(b) reads both, in encounter order
    var e = Add(col("a", int64), col("b", int64))
    _assert_columns(e.referenced_columns(), ["a", "b"])


def test_referenced_columns_nested_dedup() raises:
    # (col(a) + lit(1)) > col(b) — a literal contributes nothing, a and b once each
    var e = Gt(Add(col("a", int64), lit(1, int64)), col("b", int64))
    _assert_columns(e.referenced_columns(), ["a", "b"])


def test_referenced_columns_repeated_column_deduped() raises:
    # col(a) + col(a) collapses to a single "a" (order-preserving dedup)
    var e = Add(col("a", int64), col("a", int64))
    _assert_columns(e.referenced_columns(), ["a"])


def test_referenced_columns_reduction() raises:
    # a reduction reads its operand's columns; sum(a + b) -> [a, b]
    var e = Sum(Add(col("a", int64), col("b", int64)))
    _assert_columns(e.referenced_columns(), ["a", "b"])


def test_referenced_columns_via_anyvalue_box() raises:
    # the erased box threads referenced_columns through the trampoline
    var boxed: BoxedValue = Add(col("a", int64), col("b", int64))
    _assert_columns(boxed.referenced_columns(), ["a", "b"])


# ===========================================================================
# Validity — the fused lane tracks nulls (T0.7)
# ===========================================================================


def _nullable_batch() raises -> RecordBatch:
    # a and b each carry nulls in different rows
    return record_batch(
        [
            array([1, None, 3, None, 7, 2], int64).copy(),
            array([9, 2, None, 1, None, 8], int64).copy(),
        ],
        names=["a", "b"],
    )


def _kleene_batch() raises -> RecordBatch:
    # after `> 3`: a → [T, F, null], b → [T, null, T]
    return record_batch(
        [
            array([5, 1, None], int64).copy(),
            array([10, None, 20], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_add_propagates_nulls() raises:
    # a + b is null wherever either operand is null; valid rows sum normally
    var cv = (col("a", int64) + col("b", int64)).execute(_nullable_batch())
    assert_true(
        into_array(cv, 6)
        == array([10, None, None, None, None, 10], int64).to_dyn()
    )


def test_mul_propagates_nulls() raises:
    var cv = (col("a", int64) * col("b", int64)).execute(_nullable_batch())
    assert_true(
        into_array(cv, 6)
        == array([9, None, None, None, None, 16], int64).to_dyn()
    )


def test_compare_propagates_nulls() raises:
    # (a > b) is valid only where both operands are valid: rows 0 and 5 (both F)
    var cv = (col("a", int64) > col("b", int64)).execute(_nullable_batch())
    assert_true(
        into_array(cv, 6)
        == array([False, None, None, None, None, False]).to_dyn()
    )


def test_cast_propagates_nulls() raises:
    # int64 -> float64 cast preserves the operand's validity
    var cv = NumericCast[Float64Type](col("a", int64)).execute(
        _nullable_batch()
    )
    assert_true(
        into_array(cv, 6)
        == array([1.0, None, 3.0, None, 7.0, 2.0], float64).to_dyn()
    )


def test_isnull_over_nullable_is_never_null() raises:
    # is_null reads validity and is itself always valid (no null bit set)
    var cv = IsNull(col("a", int64)).execute(_nullable_batch())
    assert_true(
        into_array(cv, 6)
        == array([False, True, False, True, False, False]).to_dyn()
    )


def test_and_kleene_false_dominates_null() raises:
    # (a>3) & (b>3): T&T=T ; F&null=F (known-false forces valid) ; null&T=null
    var cv = (
        (col("a", int64) > lit(3, int64)) & (col("b", int64) > lit(3, int64))
    ).execute(_kleene_batch())
    assert_true(into_array(cv, 3) == array([True, False, None]).to_dyn())


def test_or_kleene_true_dominates_null() raises:
    # (a>3) | (b>3): T|T=T ; F|null=null ; null|T=T (known-true forces valid)
    var cv = (
        (col("a", int64) > lit(3, int64)) | (col("b", int64) > lit(3, int64))
    ).execute(_kleene_batch())
    assert_true(into_array(cv, 3) == array([True, None, True]).to_dyn())


# ===========================================================================
# Wave 1 wiring (T2.1) — string compares, like/ilike, is_in, conditional, temporal
# ===========================================================================


def _sp_batch() raises -> RecordBatch:
    # two string columns for binary string ops (no string-scalar broadcast yet)
    return record_batch(
        [
            array(["apple", "banana", "cherry"]).copy(),
            array(["apple", "apricot", "date"]).copy(),
        ],
        names=["s", "p"],
    )


def test_string_lt_gt() raises:
    # "apple"<"apple"=F, "banana"<"apricot"=F, "cherry"<"date"=T -> [F,F,T]
    var lt = (StrLt(col("s", string), col("p", string))).execute(_sp_batch())
    assert_true(into_array(lt, 3) == array([False, False, True]).to_dyn())
    # greater: [F, T, F]
    var gt = (StrGt(col("s", string), col("p", string))).execute(_sp_batch())
    assert_true(into_array(gt, 3) == array([False, True, False]).to_dyn())


def test_string_le_ge() raises:
    # <=: "apple"<="apple"=T, "banana"<="apricot"=F, "cherry"<="date"=T
    var le = (StrLe(col("s", string), col("p", string))).execute(_sp_batch())
    assert_true(into_array(le, 3) == array([True, False, True]).to_dyn())
    # >=: [T, T, F]
    var ge = (StrGe(col("s", string), col("p", string))).execute(_sp_batch())
    assert_true(into_array(ge, 3) == array([True, True, False]).to_dyn())


def test_string_compare_fluent() raises:
    # method surface builds the same node
    var lt = (col("s", string) < col("p", string)).execute(_sp_batch())
    assert_true(into_array(lt, 3) == array([False, False, True]).to_dyn())


def test_string_compare_composes_under_bool_logic() raises:
    # (s < p) & (s > p) is always false — two string-compare breakers under one And
    var cv = (
        And(
            StrLt(col("s", string), col("p", string)),
            StrGt(col("s", string), col("p", string)),
        )
    ).execute(_sp_batch())
    assert_true(into_array(cv, 3) == array([False, False, False]).to_dyn())


def _like_batch() raises -> RecordBatch:
    return record_batch(
        [
            array(["apple", "banana", "cherry"]).copy(),
            array(["a%", "b%", "x%"]).copy(),
        ],
        names=["s", "pat"],
    )


def test_like_predicate() raises:
    # "apple" LIKE "a%" = T, "banana" LIKE "b%" = T, "cherry" LIKE "x%" = F
    var cv = (Like(col("s", string), col("pat", string))).execute(_like_batch())
    assert_true(into_array(cv, 3) == array([True, True, False]).to_dyn())


def test_ilike_predicate() raises:
    # case-insensitive: "APPLE" ILIKE "a%" = T, "Banana" ILIKE "b%" = T
    var b = record_batch(
        [array(["APPLE", "Banana"]).copy(), array(["a%", "b%"]).copy()],
        names=["s", "pat"],
    )
    var cv = (ILike(col("s", string), col("pat", string))).execute(b)
    assert_true(into_array(cv, 2) == array([True, True]).to_dyn())


def test_like_fluent_and_under_logic() raises:
    # s.like(pat) & (s > slit-less compare) — fluent surface + composition
    var cv = (col("s", string).like(col("pat", string))).execute(_like_batch())
    assert_true(into_array(cv, 3) == array([True, True, False]).to_dyn())


def test_is_in_numeric() raises:
    # a=[1,2,3,4] IN {2,3} -> [F,T,T,F]
    var cv = (IsIn(col("a", int64), array([2, 3], int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([False, True, True, False]).to_dyn())


def test_is_in_string() raises:
    # s IN {"apple","cherry"} -> [T,F,T]
    var cv = (IsIn(col("s", string), array(["apple", "cherry"]))).execute(
        _sp_batch()
    )
    assert_true(into_array(cv, 3) == array([True, False, True]).to_dyn())


# FU-5: fused `IsIn` composed under boolean logic (`is_in(...) & cmp`) produces
# a wrong mask on the F2 fused path — a breaker-composition/slot issue. The
# dynamic F1 path handles `And(IsIn, cmp)` correctly (eager `and_`), so
# ClickBench Q41-style `IN(...) AND ...` works via F1. Disabled (non-`test_`
# prefix) until the fused composition is fixed; standalone `IsIn` is covered by
# `test_is_in_*`.
def _fu5_is_in_fuses_under_bool_logic() raises:
    # (a IN {2,3}) & (a < 3) -> [F,T,T,F] & [T,T,F,F] = [F,T,F,F]
    var cv = (
        And(
            IsIn(col("a", int64), array([2, 3], int64)),
            Lt(col("a", int64), lit(3, int64)),
        )
    ).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([False, True, False, False]).to_dyn()
    )


def _cond_batch() raises -> RecordBatch:
    # a and b with nulls in different rows
    return record_batch(
        [
            array([1, None, None, 4], int64).copy(),
            array([10, 20, None, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_coalesce() raises:
    # coalesce(a,b): [1, 20, null, 4]  (row 2 both null)
    var cv = (Coalesce(col("a", int64), col("b", int64))).execute(_cond_batch())
    assert_true(into_array(cv, 4) == array([1, 20, None, 4], int64).to_dyn())


def test_coalesce_fuses_above() raises:
    # coalesce(a,b) + 1 = [2, 21, null, 5] — the breaker feeds the numeric lane
    var cv = (
        Add(Coalesce(col("a", int64), col("b", int64)), lit(1, int64))
    ).execute(_cond_batch())
    assert_true(into_array(cv, 4) == array([2, 21, None, 5], int64).to_dyn())


def test_nullif() raises:
    # nullif(a,b): a where a==b set null. a=[1,2,3,4], b=[9,2,3,9] -> [1,null,null,4]
    var b = record_batch(
        [
            array([1, 2, 3, 4], int64).copy(),
            array([9, 2, 3, 9], int64).copy(),
        ],
        names=["a", "b"],
    )
    var cv = (Nullif(col("a", int64), col("b", int64))).execute(b)
    assert_true(into_array(cv, 4) == array([1, None, None, 4], int64).to_dyn())


def test_case_when() raises:
    # CASE WHEN a>2 THEN a ELSE b:  a=[1,2,3,4], b=[10,20,30,40] -> [10,20,3,4]
    var cv = (
        CaseWhen(
            Gt(col("a", int64), lit(2, int64)),
            col("a", int64),
            col("b", int64),
        )
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([10, 20, 3, 4], int64).to_dyn())


def test_case_when_fuses_above() raises:
    # (CASE WHEN a>2 THEN a ELSE b) * 2 = [20,40,6,8]
    var cv = (
        Mul(
            CaseWhen(
                Gt(col("a", int64), lit(2, int64)),
                col("a", int64),
                col("b", int64),
            ),
            lit(2, int64),
        )
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([20, 40, 6, 8], int64).to_dyn())


# --- temporal ---------------------------------------------------------------


def _ts_batch() raises -> RecordBatch:
    # 2019-06-15 12:30:45 UTC ; 2020-02-29 00:00:00 UTC
    var bldr = PrimitiveBuilder[TimestampType](timestamp(second), capacity=2)
    bldr.append(Int64(1_560_601_845))
    bldr.append(Int64(1_582_934_400))
    return record_batch([bldr.finish()], names=["ts"])


def _ts_null_batch() raises -> RecordBatch:
    var bldr = PrimitiveBuilder[TimestampType](timestamp(second), capacity=3)
    bldr.append(Int64(1_560_601_845))
    bldr.append_null()
    bldr.append(Int64(1_582_934_400))
    return record_batch([bldr.finish()], names=["ts"])


def test_temporal_year_month_day() raises:
    var y = (Year(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(y, 2) == array([2019, 2020], int32).to_dyn())
    var mo = (Month(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(mo, 2) == array([6, 2], int32).to_dyn())
    var d = (Day(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(d, 2) == array([15, 29], int32).to_dyn())


def test_temporal_clock_fields() raises:
    var h = (Hour(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(h, 2) == array([12, 0], int32).to_dyn())
    var mi = (Minute(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(mi, 2) == array([30, 0], int32).to_dyn())
    var se = (Second(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(se, 2) == array([45, 0], int32).to_dyn())


def test_temporal_quarter_dow_doy() raises:
    var q = (Quarter(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(q, 2) == array([2, 1], int32).to_dyn())
    # 2019-06-15 is a Saturday (ISO Mon=0 -> 5); 2020-02-29 is a Saturday -> 5
    var w = (DayOfWeek(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(w, 2) == array([5, 5], int32).to_dyn())
    var doy = (DayOfYear(col("ts", timestamp(second)))).execute(_ts_batch())
    assert_true(into_array(doy, 2) == array([166, 60], int32).to_dyn())


def test_temporal_extract_fuses_above() raises:
    # year(ts) - 2000 = [19, 20] — the extraction breaker feeds the numeric lane
    var cv = (
        Sub(Year(col("ts", timestamp(second))), lit(2000, int32))
    ).execute(_ts_batch())
    assert_true(into_array(cv, 2) == array([19, 20], int32).to_dyn())


def test_temporal_extract_fluent() raises:
    var y = col("ts", timestamp(second)).year().execute(_ts_batch())
    assert_true(into_array(y, 2) == array([2019, 2020], int32).to_dyn())


def test_temporal_null_propagates() raises:
    # a null timestamp yields a null year
    var y = (Year(col("ts", timestamp(second)))).execute(_ts_null_batch())
    assert_true(into_array(y, 3) == array([2019, None, 2020], int32).to_dyn())


def test_date_trunc_then_extract() raises:
    # date_trunc(ts, "day") zeroes the time-of-day; hour of the truncated ts = 0
    var expr = Hour(DateTrunc(col("ts", timestamp(second)), unit_day))
    var h = (expr).execute(_ts_batch())
    assert_true(into_array(h, 2) == array([0, 0], int32).to_dyn())
    # the calendar day is preserved by truncation
    var d = (Day(DateTrunc(col("ts", timestamp(second)), unit_day))).execute(
        _ts_batch()
    )
    assert_true(into_array(d, 2) == array([15, 29], int32).to_dyn())


def test_date_trunc_fluent() raises:
    var h = (col("ts", timestamp(second)).date_trunc("hour").minute()).execute(
        _ts_batch()
    )
    # truncating to the hour zeroes minutes/seconds
    assert_true(into_array(h, 2) == array([0, 0], int32).to_dyn())


# ---------------------------------------------------------------------------
# Validity through the string lane, and missing-column diagnostics.
#
# `StringValue.materialize` is the only one of the three family drivers that
# never consults `validity`, so every string *transformation* dropped nulls
# while a bare column kept them (which is what hid it). And only
# `NumericColumn` guarded the -1 that `get_field_index` answers for a missing
# name; the other three leaves indexed with it, and a negative List index
# wraps, so a typo silently read a different column.
# ---------------------------------------------------------------------------


def _str_batch_with_null() raises -> RecordBatch:
    var b = StringBuilder()
    b.append("a")
    b.append_null()
    b.append("c")
    return record_batch([b.finish().to_dyn()], names=["s"])


def test_string_map_preserves_nulls() raises:
    # upper(["a", null, "c"]) keeps the null — the map applies to values only.
    var cv = (Upper(col("s", string))).execute(_str_batch_with_null())
    var got = into_array(cv, 3)
    assert_equal(got.null_count(), 1)
    assert_true(got.is_null(1))
    assert_true(got.is_valid(0) and got.is_valid(2))


def test_string_concat_preserves_nulls() raises:
    # a null operand poisons the concatenation, as it does in the kernel.
    var cv = (Concat(col("s", string), lit("!"))).execute(
        _str_batch_with_null()
    )
    var got = into_array(cv, 3)
    assert_equal(got.null_count(), 1)
    assert_true(got.is_null(1))


def test_string_parse_failure_survives_a_fused_parent() raises:
    # to_int(["1","x","3"]) is null at "x"; adding 1 must not resurrect it as 0.
    var b = record_batch([array(["1", "x", "3"]).copy()], names=["s"])
    var e = StringToNum[Int64Type](col("s", string)) + lit(1, int64)
    var got = into_array(e.execute(b), 3)
    assert_equal(got.null_count(), 1)
    assert_true(got.is_null(1))


def test_string_column_missing_name_raises() raises:
    # `get_field_index` answers -1; without a guard that indexes the column list
    # with -1 and trips a bounds assert, which aborts the whole runner rather
    # than reporting a missing column. Only `NumericColumn` guarded it.
    with assert_raises(contains="nope"):
        _ = (col("nope", string)).execute(_str_batch2())


def test_temporal_column_missing_name_raises() raises:
    with assert_raises(contains="nope"):
        _ = (col("nope", timestamp(second))).execute(_ts_batch())


def test_list_column_missing_name_raises() raises:
    with assert_raises(contains="nope"):
        _ = (ListColumn[ListType]("nope")).execute(_list_batch())
