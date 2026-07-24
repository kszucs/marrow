"""Tests for marrow.expr.values — the staged, strategy-pluggable fusion engine.

Covers the four value families and the universal `AnyValue` box:
  * numeric — vectorized SIMD fusion (`Add`/`Mul`/…, reductions, casts, windows)
  * bool    — bit-packed vectorized fusion (comparisons, `And`/`Or`/`Not`, any/all)
  * string  — elementwise fusion (`Concat`/`Upper`/…, predicates, parses)
  * list    — materialize-only columns feeding fixed-width breakers
  * AnyValue — erases any comptime node OR a runtime `DynValue` behind `execute()`
"""

from std.testing import assert_true, assert_equal

from marrow.testing import TestSuite
from marrow.builders import array, ListBuilder, Int64Builder
from marrow.dtypes import (
    int64,
    int32,
    float64,
    string,
    Int64Type,
    StringType,
    ListType,
)
from marrow.tabular import record_batch, RecordBatch
from marrow.scalars import AnyScalar
from marrow.expr.values import (
    col,
    lit,
    slit,
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
    AnyValue,
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
)
from marrow.expr.dynamic import col as dyn_col


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
    assert_true(not cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_any())


def test_literal_broadcast() raises:
    var cv = (Mul(col("a", int64), lit(10, int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([10, 20, 30, 40], int64).to_any())


def test_scalar_literal_evaluates_once() raises:
    var cv = (lit(7, int64)).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 3) == array([7, 7, 7], int64).to_any())


def test_fused_chain() raises:
    # (a + b) * a  over a=[1,2,3,4], b=[10,20,30,40]
    var cv = (
        Mul(Add(col("a", int64), col("b", int64)), col("a", int64))
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([11, 44, 99, 176], int64).to_any())


def test_reduction_is_scalar() raises:
    # sum(a) over [1,2,3,4] = 10, a scalar
    var cv = (Sum(col("a", int64))).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([10, 10, 10, 10], int64).to_any())


def test_reduction_broadcasts_into_columnar() raises:
    # a + sum(a) = [1,2,3,4] + 10 = [11,12,13,14] — the SINGLE Add, sum(a) is a
    # fused leaf reading its stage result from the context and splatting.
    var cv = (Add(col("a", int64), Sum(col("a", int64)))).execute(_batch())
    assert_true(not cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([11, 12, 13, 14], int64).to_any())


def test_scalar_plus_scalar_stays_scalar() raises:
    # sum(a) + max(a) = 10 + 4 = 14, still scalar
    var cv = (Add(Sum(col("a", int64)), Max(col("a", int64)))).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 2) == array([14, 14], int64).to_any())


def test_arithmetic_above_reduction() raises:
    # (a + b) fuses, then * sum(a) broadcasts:  [11,22,33,44] * 10
    var cv = (
        Mul(Add(col("a", int64), col("b", int64)), Sum(col("a", int64)))
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([110, 220, 330, 440], int64).to_any())


def test_mean_centering_via_single_binary() raises:
    # x - avg(x): avg([1,2,3,4]) = 2.5 (a breaker materialized once in prepare,
    # then a splat-leaf), so the subtract fuses over (x, splat(mean)) as the same
    # NumericBinary as `x - lit`. int - float -> float.
    var cv = (Sub(col("a", int64), Mean(col("a", int64)))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([-1.5, -0.5, 0.5, 1.5], float64).to_any()
    )


def test_fused_node_is_fusable() raises:
    # `Add` over fusable operands is itself `NumericValue`; `_takes_fusable`
    # compiling is the compile-time proof.
    assert_true(_takes_fusable(Add(col("a", int64), col("b", int64))))


def test_div_is_true_division() raises:
    # 1/2,2/2,3/2,4/2 = [0.5,1.0,1.5,2.0] float64 — true division, not integer
    var cv = (Div(col("a", int64), lit(2, int64))).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([0.5, 1.0, 1.5, 2.0], float64).to_any()
    )


def test_unary_neg_fuses() raises:
    var cv = (Neg(col("a", int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([-1, -2, -3, -4], int64).to_any())


def test_cast_fuses_in_chain() raises:
    # a fused cast composes with arithmetic in the same pass (identity cast here)
    var cv = (
        Add(NumericCast[Int64Type](col("a", int64)), col("b", int64))
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([11, 22, 33, 44], int64).to_any())


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _spec() -> WindowSpec:
    return WindowSpec(FrameBound(0, 0), FrameBound(2, 0))


def test_window_row_number() raises:
    var cv = (RowNumber(col("a", int64), _spec())).execute(_batch())
    assert_true(into_array(cv, 4) == array([1, 2, 3, 4], int64).to_any())


def test_arithmetic_above_window_materializes() raises:
    # row_number() + 1 → [2,3,4,5]  (Add above a columnar window breaker)
    var cv = (Add(RowNumber(col("a", int64), _spec()), lit(1, int64))).execute(
        _batch()
    )
    assert_true(into_array(cv, 4) == array([2, 3, 4, 5], int64).to_any())


# ===========================================================================
# Boolean family — bit-packed vectorized fusion
# ===========================================================================


def test_comparison_fuses_to_bool() raises:
    # a < 3 over [1,2,3,4] → bit-packed [T,T,F,F] (the bool fused strategy)
    var cv = (Lt(col("a", int64), lit(3, int64))).execute(_batch())
    assert_true(into_array(cv, 4) == array([True, True, False, False]).to_any())


def test_bool_and_fuses() raises:
    # (a < 3) & (b > 15) → [T,T,F,F] & [F,T,T,T] = [F,T,F,F], one fused bitwise pass
    var cv = (
        And(
            Lt(col("a", int64), lit(3, int64)),
            Gt(col("b", int64), lit(15, int64)),
        )
    ).execute(_batch())
    assert_true(
        into_array(cv, 4) == array([False, True, False, False]).to_any()
    )


def test_bool_not_fuses() raises:
    # not (a < 3) → not [T,T,F,F] = [F,F,T,T]
    var cv = (Not(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(into_array(cv, 4) == array([False, False, True, True]).to_any())


def test_bool_or_fuses() raises:
    # (a < 2) | (a > 3) → [T,F,F,F] | [F,F,F,T] = [T,F,F,T]
    var cv = (
        Or(
            Lt(col("a", int64), lit(2, int64)),
            Gt(col("a", int64), lit(3, int64)),
        )
    ).execute(_batch())
    assert_true(into_array(cv, 4) == array([True, False, False, True]).to_any())


def test_any_all_reductions() raises:
    # any(a < 3) = True, all(a < 3) = False over [1,2,3,4]
    var an = (Any(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(an.isa[AnyScalar]() and an[AnyScalar].as_bool().value())
    var al = (All(Lt(col("a", int64), lit(3, int64)))).execute(_batch())
    assert_true(al.isa[AnyScalar]() and not al[AnyScalar].as_bool().value())


def test_count_reduction() raises:
    # count(a) over [1,2,3,4] = 4 (int64 scalar)
    var cv = (Count(col("a", int64))).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(into_array(cv, 4) == array([4, 4, 4, 4], int64).to_any())


def test_notnull_and_isnull() raises:
    # no nulls in a=[1,2,3,4] → not_null all true, is_null all false
    var nn = (NotNull(col("a", int64))).execute(_batch())
    assert_true(into_array(nn, 4) == array([True, True, True, True]).to_any())
    var isn = (IsNull(col("a", int64))).execute(_batch())
    assert_true(
        into_array(isn, 4) == array([False, False, False, False]).to_any()
    )


def test_isnan_fuses_over_float() raises:
    # is_nan over finite floats → all false, computed in a fused SIMD pass
    var b = record_batch(
        [array([1.0, 2.0, 3.0, 4.0], float64).copy()], names=["f"]
    )
    var cv = (IsNan(col("f", float64))).execute(b)
    assert_true(
        into_array(cv, 4) == array([False, False, False, False]).to_any()
    )


def test_num_to_bool_fuses() raises:
    # a*0 = 0 → false ; a (nonzero) → true — fused per-lane num->bool
    var z = (NumToBool(Mul(col("a", int64), lit(0, int64)))).execute(_batch())
    assert_true(
        into_array(z, 4) == array([False, False, False, False]).to_any()
    )
    var nz = (NumToBool(col("a", int64))).execute(_batch())
    assert_true(into_array(nz, 4) == array([True, True, True, True]).to_any())


def test_bool_to_num_fuses() raises:
    # (a < 3) -> int64 = [1,1,0,0] — fused bool->num, composes in the numeric lane
    var cv = (BoolToNum[Int64Type](Lt(col("a", int64), lit(3, int64)))).execute(
        _batch()
    )
    assert_true(into_array(cv, 4) == array([1, 1, 0, 0], int64).to_any())


def test_fluent_numeric_and_bool() raises:
    # operators/methods build the same nodes as the explicit builders
    var s = (col("a", int64) + col("b", int64)).execute(_batch())
    assert_true(into_array(s, 4) == array([11, 22, 33, 44], int64).to_any())
    # mean-centering via `x - x.mean()`
    var mc = (col("a", int64) - col("a", int64).mean()).execute(_batch())
    assert_true(
        into_array(mc, 4) == array([-1.5, -0.5, 0.5, 1.5], float64).to_any()
    )
    # (a < 3) & (b > 15) via `<`, `>`, `&`
    var mask = (
        (col("a", int64) < lit(3, int64)) & (col("b", int64) > lit(15, int64))
    ).execute(_batch())
    assert_true(
        into_array(mask, 4) == array([False, True, False, False]).to_any()
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
    var cv = (slit("hi")).execute(_str_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(cv[AnyScalar].as_string().to_string() == "hi")


def test_concat_chain_fuses() raises:
    # col || "p1" || "p2" over ["ab","cd"] → ["abp1p2","cdp1p2"] — one builder pass
    var expr = Concat(Concat(col("s", string), slit("p1")), slit("p2"))
    var cv = (expr).execute(_str_batch())
    assert_true(into_array(cv, 2) == array(["abp1p2", "cdp1p2"]).to_any())


def test_strlen_fuses_into_numeric() raises:
    # length(s) + 1 over ["ab","cd"] → byte lengths [2,2] + 1 = [3,3]. A STRATEGY
    # TRANSITION: the string stage materializes, then `length` reads offsets as a
    # vectorwise numeric leaf and the `+ 1` fuses in the same numeric pass.
    var expr = Add(StringLength(col("s", string)), lit(1, int32))
    var cv = (expr).execute(_str_batch())
    assert_true(into_array(cv, 2) == array([3, 3], int32).to_any())


def test_upper_map_fuses() raises:
    # upper(s) over ["ab","cd"] → ["AB","CD"] (elementwise map, delegates to kernel)
    var cv = (Upper(col("s", string))).execute(_str_batch())
    assert_true(into_array(cv, 2) == array(["AB", "CD"]).to_any())


def test_map_and_concat_fuse_together() raises:
    # upper(s) || "!" → ["AB!","CD!"] — map + concat in one builder pass
    var cv = (Concat(Upper(col("s", string)), slit("!"))).execute(_str_batch())
    assert_true(into_array(cv, 2) == array(["AB!", "CD!"]).to_any())


def test_startswith_predicate() raises:
    # startswith(s, p): "abc".sw("ab")=T, "xyz".sw("yy")=F → [T,F]
    var cv = (StartsWith(col("s", string), col("p", string))).execute(
        _str_batch2()
    )
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def test_predicate_and_strlen_compose_under_bool_logic() raises:
    # startswith(s,p) & (length(s) > 2) → [T,F] & [T,T] = [T,F]
    # a string-predicate breaker AND a strlen breaker, both fused under one `And`.
    var cv = (
        And(
            StartsWith(col("s", string), col("p", string)),
            Gt(StringLength(col("s", string)), lit(2, int32)),
        )
    ).execute(_str_batch2())
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def test_string_to_num_parses() raises:
    # parse ["10","20"] -> int64 [10,20] (a string->numeric breaker)
    var b = record_batch([array(["10", "20"]).copy()], names=["s"])
    var cv = (StringToNum[Int64Type](col("s", string))).execute(b)
    assert_true(into_array(cv, 2) == array([10, 20], int64).to_any())


def test_string_to_bool_parses() raises:
    # parse ["true","false"] -> [T,F] (a string->bool breaker)
    var b = record_batch([array(["true", "false"]).copy()], names=["s"])
    var cv = (StringToBool(col("s", string))).execute(b)
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def test_num_to_string() raises:
    # format int64 [1,2,3,4] -> ["1","2","3","4"] (a string breaker)
    var b = record_batch([array([1, 2, 3, 4], int64).copy()], names=["n"])
    var cv = (NumToString[StringType](col("n", int64))).execute(b)
    assert_true(into_array(cv, 4) == array(["1", "2", "3", "4"]).to_any())


def test_num_to_string_fuses_with_concat() raises:
    # cast(n, string) || "!" -> ["1!","2!"] — string breaker read fuses into concat
    var b = record_batch([array([1, 2], int64).copy()], names=["n"])
    var cv = (Concat(NumToString[StringType](col("n", int64)), slit("!"))).execute(
        b
    )
    assert_true(into_array(cv, 2) == array(["1!", "2!"]).to_any())


def test_string_to_string_container_cast() raises:
    # string -> string container cast, values preserved
    var cv = (StringToString[StringType](col("s", string))).execute(_str_batch())
    assert_true(into_array(cv, 2) == array(["ab", "cd"]).to_any())


def test_fluent_string() raises:
    # method + operator surface: `s.upper()` and `s || "!"`
    var u = col("s", string).upper().execute(_str_batch())
    assert_true(into_array(u, 2) == array(["AB", "CD"]).to_any())
    var c = (col("s", string) + slit("!")).execute(_str_batch())
    assert_true(into_array(c, 2) == array(["ab!", "cd!"]).to_any())


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
    assert_true(into_array(cv, 2) == array([3, 2], int32).to_any())


def test_list_length_fuses_above() raises:
    # length(l) + 1 = [4, 3] — the breaker feeds the fused numeric lane
    var cv = (Add(ListLength(ListColumn[ListType]("l")), lit(1, int32))).execute(
        _list_batch()
    )
    assert_true(into_array(cv, 2) == array([4, 3], int32).to_any())


def test_list_contains() raises:
    # 20 in [10,20,30] = T ; 20 in [40,50] = F  ->  [T, F]
    var cv = (ListContains(ListColumn[ListType]("l"), lit(20, int64))).execute(
        _list_batch()
    )
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


# ===========================================================================
# AnyValue — the universal box over a comptime node OR a runtime DynValue
# ===========================================================================


def test_anyvalue_wraps_dynvalue() raises:
    # the untyped runtime interpreter (DynValue), boxed in AnyValue, runs via the
    # tag dispatch — this is what the relational engine builds plans from
    var boxed: AnyValue = dyn_col(0) + dyn_col(1)
    var cv = boxed.execute(_batch())
    assert_true(cv == array([11, 22, 33, 44], int64).to_any())


def test_anyvalue_erases_to_array() raises:
    # box a comptime node; its erased execute yields a column (AnyArray), the
    # interface the relational engine consumes
    var boxed: AnyValue = Add(col("a", int64), lit(10, int64))
    var cv = boxed.execute(_batch())
    assert_true(cv == array([11, 12, 13, 14], int64).to_any())


def test_anyvalue_interchange() raises:
    # a heterogeneous list holds a fused comptime column *and* an interpreter
    # value; both run through the one erased execute — fused-vs-interpreted is
    # only which node you boxed.
    var values = List[AnyValue]()
    values.append(AnyValue(col("a", int64)))  # fused comptime
    values.append(AnyValue(dyn_col("b")))  # runtime interpreter
    var batch = _batch()
    assert_true(values[0].execute(batch) == array([1, 2, 3, 4], int64).to_any())
    assert_true(
        values[1].execute(batch) == array([10, 20, 30, 40], int64).to_any()
    )


def test_anyvalue_write_to_delegates() raises:
    # write_to on a DynValue-boxed AnyValue renders the boxed node's expression
    # form (not just its column name)
    var boxed: AnyValue = dyn_col("a") + dyn_col("b")
    assert_true(String(boxed).find("add") != -1)


def main() raises:
    TestSuite.run[__functions_in_module()]()
