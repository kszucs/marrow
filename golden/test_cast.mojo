"""Golden cases — casts, the AOT lane.

The fused lane names its conversion node directly — `NumericCast`,
`NumToString`, `StringToNum`, `BoolToNum`, `NumToBool` — where the runtime
lane resolves `cast(target)` against the operand's dtype. Both must land on
the same kernel.
"""

from golden.helpers import check as _check, fixture as _fixture
from marrow.dtypes import (
    BoolType,
    Float64Type,
    Int32Type,
    Int64Type,
    StringType,
    bool_,
    float64,
    int64,
    string,
)
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation, in_memory_table
from marrow.expr.values import (
    BoolToNum,
    NumToBool,
    NumToString,
    NumericCast,
    StringToNum,
)


def _nums() raises -> DynRelation:
    return in_memory_table(_fixture("nums"))


def test_golden_cast_int_to_float() raises:
    _check(
        "test_golden_cast_int_to_float",
        _nums().project(["c"], [NumericCast[Float64Type](col("i", int64))]),
    )


def test_golden_cast_int_to_int32() raises:
    _check(
        "test_golden_cast_int_to_int32",
        _nums().project(["c"], [NumericCast[Int32Type](col("i", int64))]),
    )


def test_golden_cast_float_to_int() raises:
    _check(
        "test_golden_cast_float_to_int",
        _nums().project(["c"], [NumericCast[Int64Type](col("f", float64))]),
    )


def test_golden_cast_int_to_string() raises:
    _check(
        "test_golden_cast_int_to_string",
        _nums().project(["c"], [NumToString[StringType](col("i", int64))]),
    )


def test_golden_cast_string_to_int() raises:
    _check(
        "test_golden_cast_string_to_int",
        _nums().project(["c"], [StringToNum[Int64Type](col("s", string))]),
    )


def test_golden_cast_bool_to_int() raises:
    _check(
        "test_golden_cast_bool_to_int",
        _nums().project(["c"], [BoolToNum[Int64Type](col("b", bool_))]),
    )


def test_golden_cast_int_to_bool() raises:
    _check(
        "test_golden_cast_int_to_bool",
        _nums().project(["c"], [NumToBool(col("i", int64))]),
    )
