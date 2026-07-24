"""The elementwise string strategy — `col || "p1" || "p2"` fuses in one builder
pass, with no intermediate `StringArray` for `col || "p1"`. This is a *different
kind of fusion* (elementwise, not vectorized) under the same staged model.
"""

from std.testing import assert_true

from marrow.testing import TestSuite
from marrow.builders import array
from marrow.dtypes import string, int32, int64, Int64Type, StringType
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.lane import (
    col,
    slit,
    lit,
    Concat,
    Upper,
    StringLength,
    StartsWith,
    StringToNum,
    StringToBool,
    NumToString,
    StringToString,
    Add,
    And,
    Gt,
    into_array,
)
from marrow.scalars import AnyScalar


def _batch() raises -> RecordBatch:
    return record_batch([array(["ab", "cd"]).copy()], names=["s"])


def _batch2() raises -> RecordBatch:
    # two string columns, for binary predicates (a literal pattern would need the
    # unsupported string-scalar broadcast — a noted follow-up)
    return record_batch(
        [array(["abc", "xyz"]).copy(), array(["ab", "yy"]).copy()],
        names=["s", "p"],
    )


def test_string_literal_is_scalar() raises:
    # a bare string literal is a scalar Datum (broadcasts lazily at a boundary)
    var cv = (slit("hi")).execute(_batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(cv[AnyScalar].as_string().to_string() == "hi")


def test_concat_chain_fuses() raises:
    # col || "p1" || "p2" over ["ab","cd"] → ["abp1p2","cdp1p2"] — one builder pass
    var expr = Concat(Concat(col("s", string), slit("p1")), slit("p2"))
    var cv = (expr).execute(_batch())
    assert_true(
        into_array(cv, 2) == array(["abp1p2", "cdp1p2"]).to_any()
    )


def test_strlen_fuses_into_numeric() raises:
    # length(s) + 1 over ["ab","cd"] → byte lengths [2,2] + 1 = [3,3]. A STRATEGY
    # TRANSITION: the string stage materializes, then `length` reads offsets as a
    # vectorwise numeric leaf and the `+ 1` fuses in the same numeric pass.
    var expr = Add(StringLength(col("s", string)), lit(1, int32))
    var cv = (expr).execute(_batch())
    assert_true(into_array(cv, 2) == array([3, 3], int32).to_any())


def test_upper_map_fuses() raises:
    # upper(s) over ["ab","cd"] → ["AB","CD"] (elementwise map, delegates to kernel)
    var cv = (Upper(col("s", string))).execute(_batch())
    assert_true(into_array(cv, 2) == array(["AB", "CD"]).to_any())


def test_map_and_concat_fuse_together() raises:
    # upper(s) || "!" → ["AB!","CD!"] — map + concat in one builder pass
    var cv = (Concat(Upper(col("s", string)), slit("!"))).execute(_batch())
    assert_true(into_array(cv, 2) == array(["AB!", "CD!"]).to_any())


def test_startswith_predicate() raises:
    # startswith(s, p): "abc".sw("ab")=T, "xyz".sw("yy")=F → [T,F]
    var cv = (StartsWith(col("s", string), col("p", string))).execute(_batch2())
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def test_predicate_and_strlen_compose_under_bool_logic() raises:
    # startswith(s,p) & (length(s) > 2) → [T,F] & [T,T] = [T,F]
    # a string-predicate breaker AND a strlen breaker, both fused under one `And`.
    var cv = (And(
            StartsWith(col("s", string), col("p", string)),
            Gt(StringLength(col("s", string)), lit(2, int32)),
        )).execute(_batch2())
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
    var cv = (Concat(NumToString[StringType](col("n", int64)), slit("!"))).execute(b)
    assert_true(into_array(cv, 2) == array(["1!", "2!"]).to_any())


def test_string_to_string_container_cast() raises:
    # string -> string container cast, values preserved
    var cv = (StringToString[StringType](col("s", string))).execute(_batch())
    assert_true(into_array(cv, 2) == array(["ab", "cd"]).to_any())


def test_fluent_string() raises:
    # method + operator surface: `s.upper()` and `s || "!"`
    var u = col("s", string).upper().execute(_batch())
    assert_true(into_array(u, 2) == array(["AB", "CD"]).to_any())
    var c = (col("s", string) + slit("!")).execute(_batch())
    assert_true(into_array(c, 2) == array(["ab!", "cd!"]).to_any())


def main() raises:
    TestSuite.run[__functions_in_module()]()
