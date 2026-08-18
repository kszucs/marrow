from std.testing import assert_true, assert_false, assert_raises
from std.memory import ArcPointer
from ...dtypes import DynType, int64, string
from ...scalars import Int64Scalar, StringScalar
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
