from std.testing import assert_true, assert_false, assert_raises
from std.memory import ArcPointer
from ...builders import array
from ...tabular import record_batch
from ...dtypes import DynType, bool_, field, int64, second, string, timestamp
from ...schema import schema
from ...scalars import DynScalar, Int64Scalar, StringScalar, TimestampScalar
from ...parquet import LeafSet
from ..builders import col, param
from ..relations import ParquetScan
from ..values import BoxedValue
from ..params import (
    ParamCell,
    ParamDecl,
    PathSpec,
    drain_params,
    parse_params,
    register_param,
    render_describe,
    render_usage,
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


def test_parquet_scan_accepts_a_param_path() raises:
    _ = drain_params()
    var src = param("src", string)
    var scan = ParquetScan[LeafSet.all()](
        path=src, schema=schema([field("a", int64)])
    )
    var decls = drain_params()
    decls[0].cell[].set(StringScalar(String("x.parquet")).to_dyn())
    assert_true(scan.path.resolve() == "x.parquet")


def test_parse_params_binds_by_name() raises:
    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="min-a", dtype=DynType(int64)))
    parse_params(["--min-a", "5"], decls)
    assert_true(decls[0].cell[].get().as_int64().value() == 5)


def test_parse_params_applies_defaults() raises:
    var decls = List[ParamDecl]()
    decls.append(
        ParamDecl(
            name="min-a",
            dtype=DynType(int64),
            default=Optional(Int64Scalar(9).to_dyn()),
        )
    )
    parse_params(List[String](), decls)
    assert_true(decls[0].cell[].get().as_int64().value() == 9)


def test_parse_params_missing_required_raises() raises:
    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="src", dtype=DynType(string)))
    with assert_raises():
        parse_params(List[String](), decls)


def test_parse_params_unknown_flag_raises() raises:
    var decls = List[ParamDecl]()
    with assert_raises():
        parse_params(["--nope", "1"], decls)


def test_render_usage_names_every_param() raises:
    var decls = List[ParamDecl]()
    decls.append(
        ParamDecl(name="src", dtype=DynType(string), help=String("input"))
    )
    var usage = render_usage(decls)
    assert_true("--src" in usage)
    assert_true("input" in usage)


def test_parse_params_bool_binds_true_and_false() raises:
    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="verbose", dtype=DynType(bool_)))
    parse_params(["--verbose", "TRUE"], decls)
    assert_true(decls[0].cell[].get().as_bool().value())

    var decls2 = List[ParamDecl]()
    decls2.append(ParamDecl(name="verbose", dtype=DynType(bool_)))
    parse_params(["--verbose", "0"], decls2)
    assert_false(decls2[0].cell[].get().as_bool().value())


def test_parse_params_bool_rejects_unrecognized_value() raises:
    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="verbose", dtype=DynType(bool_)))
    with assert_raises():
        parse_params(["--verbose", "yes"], decls)


def test_render_describe_escapes_control_characters() raises:
    var backslash = String("\\")
    var quote = String('"')
    var raw_help = String("line one") + "\n" + "line " + quote + "two" + quote

    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="src", dtype=DynType(string), help=raw_help))
    var out = render_describe(decls)

    var escaped_newline = "line one" + backslash + "n" + "line "
    var escaped_quote = backslash + quote + "two" + backslash + quote
    assert_true(escaped_newline in out)
    assert_true(escaped_quote in out)
    # The raw, unescaped help text (real newline, bare quotes) must not
    # survive into the JSON: if either character came through unescaped,
    # this exact substring would still be present.
    assert_false(raw_help in out)


def test_render_describe_shape_for_one_parameter() raises:
    var decls = List[ParamDecl]()
    decls.append(
        ParamDecl(name="min-a", dtype=DynType(int64), help=String("cutoff"))
    )
    var out = render_describe(decls)
    var want = String(
        '[\n  {"name": "min-a", "dtype": "'
        + String(DynType(int64))
        + '", "help": "cutoff", "required": true}\n]'
    )
    assert_true(out == want)
