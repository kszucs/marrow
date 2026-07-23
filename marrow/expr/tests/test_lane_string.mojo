"""The elementwise string strategy — `col || "p1" || "p2"` fuses in one builder
pass, with no intermediate `StringArray` for `col || "p1"`. This is a *different
kind of fusion* (elementwise, not vectorized) under the same staged model.
"""

from std.testing import assert_true

from marrow.testing import TestSuite
from marrow.builders import array
from marrow.dtypes import string, int32
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.lane import (
    scol,
    slit,
    lit,
    run,
    Concat,
    StringLength,
    Add,
    into_array,
)
from marrow.scalars import AnyScalar


def _batch() raises -> RecordBatch:
    return record_batch([array(["ab", "cd"]).copy()], names=["s"])


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


def main() raises:
    TestSuite.run[__functions_in_module()]()
