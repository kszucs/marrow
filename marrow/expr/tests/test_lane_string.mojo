"""The elementwise string strategy — `col || "p1" || "p2"` fuses in one builder
pass, with no intermediate `StringArray` for `col || "p1"`. This is a *different
kind of fusion* (elementwise, not vectorized) under the same staged model.
"""

from std.testing import assert_true

from marrow.testing import TestSuite
from marrow.builders import array
from marrow.dtypes import string, int32, int64, Int64Type
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.lane import (
    scol,
    slit,
    lit,
    run,
    Concat,
    Upper,
    StringLength,
    StartsWith,
    StringToNum,
    StringToBool,
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
    var cv = run(slit("hi"), _batch())
    assert_true(cv.isa[AnyScalar]())
    assert_true(cv[AnyScalar].as_string().to_string() == "hi")


def test_concat_chain_fuses() raises:
    # col || "p1" || "p2" over ["ab","cd"] → ["abp1p2","cdp1p2"] — one builder pass
    var expr = Concat(Concat(scol(0, string), slit("p1")), slit("p2"))
    var cv = run(expr, _batch())
    assert_true(
        into_array(cv, 2) == array(["abp1p2", "cdp1p2"]).to_any()
    )


def test_strlen_fuses_into_numeric() raises:
    # length(s) + 1 over ["ab","cd"] → byte lengths [2,2] + 1 = [3,3]. A STRATEGY
    # TRANSITION: the string stage materializes, then `length` reads offsets as a
    # vectorwise numeric leaf and the `+ 1` fuses in the same numeric pass.
    var expr = Add(StringLength(scol(0, string)), lit(1, int32))
    var cv = run(expr, _batch())
    assert_true(into_array(cv, 2) == array([3, 3], int32).to_any())


def test_upper_map_fuses() raises:
    # upper(s) over ["ab","cd"] → ["AB","CD"] (elementwise map, delegates to kernel)
    var cv = run(Upper(scol(0, string)), _batch())
    assert_true(into_array(cv, 2) == array(["AB", "CD"]).to_any())


def test_map_and_concat_fuse_together() raises:
    # upper(s) || "!" → ["AB!","CD!"] — map + concat in one builder pass
    var cv = run(Concat(Upper(scol(0, string)), slit("!")), _batch())
    assert_true(into_array(cv, 2) == array(["AB!", "CD!"]).to_any())


def test_startswith_predicate() raises:
    # startswith(s, p): "abc".sw("ab")=T, "xyz".sw("yy")=F → [T,F]
    var cv = run(StartsWith(scol(0, string), scol(1, string)), _batch2())
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def test_predicate_and_strlen_compose_under_bool_logic() raises:
    # startswith(s,p) & (length(s) > 2) → [T,F] & [T,T] = [T,F]
    # a string-predicate breaker AND a strlen breaker, both fused under one `And`.
    var cv = run(
        And(
            StartsWith(scol(0, string), scol(1, string)),
            Gt(StringLength(scol(0, string)), lit(2, int32)),
        ),
        _batch2(),
    )
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def test_string_to_num_parses() raises:
    # parse ["10","20"] -> int64 [10,20] (a string->numeric breaker)
    var b = record_batch([array(["10", "20"]).copy()], names=["s"])
    var cv = run(StringToNum[Int64Type](scol(0, string)), b)
    assert_true(into_array(cv, 2) == array([10, 20], int64).to_any())


def test_string_to_bool_parses() raises:
    # parse ["true","false"] -> [T,F] (a string->bool breaker)
    var b = record_batch([array(["true", "false"]).copy()], names=["s"])
    var cv = run(StringToBool(scol(0, string)), b)
    assert_true(into_array(cv, 2) == array([True, False]).to_any())


def main() raises:
    TestSuite.run[__functions_in_module()]()
