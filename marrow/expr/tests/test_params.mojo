from std.testing import assert_true, assert_false, assert_raises
from std.memory import ArcPointer
from ...builders import array
from ...tabular import record_batch
from ...dtypes import DynType, int64, second, string, timestamp
from ...scalars import DynScalar, Int64Scalar, StringScalar, TimestampScalar
from ..builders import col, param
from ..values import BoxedValue
from ..params import (
    ParamCell,
    ParamDecl,
    PathSpec,
    drain_params,
    register_param,
)


def test_param_cell_unbound_raises() raises:
    var cell = ParamCell()
    assert_false(cell.is_bound())
    with assert_raises():
        _ = cell.get()


def test_param_cell_binds_and_reads() raises:
    var cell = ParamCell()
    cell.set(Int64Scalar(7).to_dyn())
    assert_true(cell.is_bound())
    assert_true(cell.get().as_int64().value() == 7)


def test_param_registry_drains_empty() raises:
    _ = drain_params()
    assert_true(len(drain_params()) == 0)


def test_param_registry_drains_once() raises:
    _ = drain_params()
    register_param(
        ParamDecl(name="src", dtype=DynType(string), help=String("in"))
    )
    var first = drain_params()
    assert_true(len(first) == 1)
    assert_true(first[0].name == "src")
    assert_true(len(drain_params()) == 0)


def test_path_spec_literal_resolves() raises:
    var spec = PathSpec(String("a.parquet"))
    assert_true(spec.resolve() == "a.parquet")


def test_path_spec_cell_resolves() raises:
    var cell = ArcPointer(ParamCell())
    var spec = PathSpec(cell)
    cell[].set(StringScalar(String("b.parquet")).to_dyn())
    assert_true(spec.resolve() == "b.parquet")


def test_numeric_param_binds_into_a_fused_predicate() raises:
    _ = drain_params()
    var a = array([1, 5, 3, 8, 2], int64)
    var batch = record_batch([a.copy()], names=["a"])

    var min_a = param("min-a", int64)
    var pred = col("a", int64) > min_a

    var decls = drain_params()
    assert_true(len(decls) == 1)
    decls[0].cell[].set(Int64Scalar(3).to_dyn())

    var out = BoxedValue(pred).execute(batch)
    assert_true(out.as_bool() == array([False, True, False, True, False]))


def test_numeric_param_unbound_raises_at_execute() raises:
    _ = drain_params()
    var a = array([1, 2], int64)
    var batch = record_batch([a.copy()], names=["a"])
    var pred = col("a", int64) > param("min-a", int64)
    with assert_raises():
        _ = BoxedValue(pred).execute(batch)


def test_string_param_binds_into_a_fused_predicate() raises:
    _ = drain_params()
    var s = array(["p", "q", "p"])
    var batch = record_batch([s.copy()], names=["s"])
    var want = param("want", string)
    var pred = col("s", string) == want
    var decls = drain_params()
    decls[0].cell[].set(StringScalar(String("p")).to_dyn())
    var out = BoxedValue(pred).execute(batch)
    assert_true(out.as_bool() == array([True, False, True]))


def test_string_param_default_is_used_when_unset() raises:
    _ = drain_params()
    _ = param("want", string, default=String("q"))
    var decls = drain_params()
    assert_true(decls[0].default.value().as_string().value() == "q")
    assert_false(decls[0].is_required())


def test_temporal_param_binds_into_execute() raises:
    _ = drain_params()
    var batch = record_batch([array([1, 2], int64)], names=["a"])
    var cutoff = param("cutoff", timestamp(second))
    var decls = drain_params()
    decls[0].cell[].set(
        TimestampScalar(
            Optional(Int64(1_560_601_845)), timestamp(second)
        ).to_dyn()
    )
    var out = cutoff.execute(batch)
    assert_true(out.isa[DynScalar]())
    assert_true(out[DynScalar].as_timestamp().value() == 1_560_601_845)


def test_temporal_param_default_converts_to_native_scalar() raises:
    _ = drain_params()
    _ = param("cutoff", timestamp(second), default=1_560_601_845)
    var decls = drain_params()
    assert_true(
        decls[0].default.value().as_timestamp().value() == 1_560_601_845
    )
    assert_false(decls[0].is_required())

