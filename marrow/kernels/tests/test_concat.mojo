# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The batch concatenation on a context: a large one on several threads
answers what the serial builder does."""

from std.testing import assert_equal, assert_true

from ...arrays import DynArray, StructArray
from ...builders import (
    Float64Builder,
    Int64Builder,
    PrimitiveBuilder,
    StringBuilder,
)
from ...dtypes import (
    Field,
    TimestampType,
    field,
    int64,
    millisecond,
    string,
    struct_,
    timestamp,
)
from ...execution import ExecContext
from ...kernels.concat import concat
from ...schema import Schema
from ...utils.testing import assert_values_equal


comptime _ROWS = 200_000


def _batch(var children: List[DynArray]) raises -> StructArray:
    """`children` as one batch, named `c0`, `c1`, ..."""
    var fields = List[Field]()
    for i in range(len(children)):
        fields.append(field(String("c", i), children[i].dtype()))
    return StructArray(
        dtype=struct_(fields^),
        length=_ROWS,
        nulls=0,
        offset=0,
        bitmap=None,
        children=children^,
    )


def _assert_parallel_matches_serial(whole: StructArray) raises:
    """`whole` cut into uneven slices, so every piece but the first starts at
    an offset and some end mid-byte of their validity bitmap, concatenated on
    one thread and on four."""
    var cuts: List[Int] = [0, 3, 70_001, 70_009, 150_000, _ROWS]
    var parts = List[StructArray]()
    for i in range(len(cuts) - 1):
        parts.append(whole.slice(cuts[i], cuts[i + 1] - cuts[i]))
    var schema = Schema.from_dtype(whole.dtype)
    var serial = concat(parts, schema, ExecContext.serial())
    var parallel = concat(parts, schema, ExecContext.parallel(4))
    assert_true(parallel.dtype == serial.dtype)
    assert_equal(len(parallel), _ROWS)
    for f in range(len(serial.children)):
        assert_values_equal(parallel.field(f), serial.field(f))


def _ints() raises -> DynArray:
    var b = Int64Builder(_ROWS)
    for i in range(_ROWS):
        if i % 7 == 0:
            b.append_null()
        else:
            b.append(Int64(i * 3))
    return b.finish().to_dyn()


def _strings() raises -> DynArray:
    var b = StringBuilder()
    for i in range(_ROWS):
        if i % 5 == 0:
            b.append_null()
        else:
            b.append(String("s", i % 97))
    return b.finish().to_dyn()


def test_concat_in_parallel_matches_serial_for_integers() raises:
    _assert_parallel_matches_serial(_batch([_ints()]))


def test_concat_in_parallel_matches_serial_for_floats() raises:
    var b = Float64Builder(_ROWS)
    for i in range(_ROWS):
        b.append(Float64(i) / 4)
    _assert_parallel_matches_serial(_batch([b.finish().to_dyn()]))


def test_concat_in_parallel_matches_serial_for_strings() raises:
    _assert_parallel_matches_serial(_batch([_strings()]))


def test_concat_in_parallel_keeps_a_parameterised_dtype() raises:
    """A timestamp's unit and zone survive: the output takes its dtype from
    the inputs, not from a default instance of the type."""
    var ts = timestamp(millisecond, "UTC")
    var b = PrimitiveBuilder[TimestampType](ts, capacity=_ROWS)
    for i in range(_ROWS):
        b.append(Int64(i))
    _assert_parallel_matches_serial(_batch([b.finish().to_dyn()]))


def test_concat_in_parallel_matches_serial_for_a_struct() raises:
    var inner = _batch([_ints(), _strings()])
    _assert_parallel_matches_serial(_batch([inner^.to_dyn(), _ints()]))
