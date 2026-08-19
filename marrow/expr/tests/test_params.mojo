from std.testing import assert_true, assert_false, assert_raises
from std.memory import ArcPointer
from ...builders import array
from ...tabular import record_batch
from ...dtypes import DynType, bool_, field, int64, second, string, timestamp
from ...schema import schema
from ...scalars import DynScalar, Int64Scalar, StringScalar, TimestampScalar
from ...parquet import LeafSet
from ..builders import col, param
from ..relations import (
    CLI_WRITERS_ENABLED,
    ParquetScan,
    cli_output_format,
    split_cli_args,
    write_cli_output,
)
from ..values import BoxedValue
from ..params import (
    ParamCell,
    ParamDecl,
    PathSpec,
    drain_params,
    lookup_param,
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
    _ = register_param(
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


# ---------------------------------------------------------------------------
# One name = one parameter = one cell
# ---------------------------------------------------------------------------


def test_param_repeated_name_reuses_one_cell() raises:
    """A second `param("min-a", ...)` shares the first one's cell and adds no
    second declaration — a CLI flag names one parameter, not one per mention.
    """
    _ = drain_params()
    var lo = param("min-a", int64)
    var lo_again = param("min-a", int64)
    var decls = drain_params()
    assert_true(len(decls) == 1)

    parse_params(["--min-a", "4"], decls)
    var batch = record_batch([array([1, 5, 3], int64)], names=["a"])
    assert_true(lo.state(batch) == 4)
    assert_true(lo_again.state(batch) == 4)


def test_param_repeated_name_across_lanes_binds_once() raises:
    """The cross-lane case, bound through `parse_params` rather than by hand.

    This is the regression the hand-bound-cell tests could not see: the fused
    node and the runtime node must both read the value `--min-a` bound, not
    one of them a stale cell or its default.
    """
    _ = drain_params()
    var batch = record_batch([array([1, 5, 3, 8, 2], int64)], names=["a"])
    var fused = col("a", int64) > param("min-a", int64, default=99)
    var dyn = col("a") > param("min-a", DynType(int64))
    var decls = drain_params()
    assert_true(len(decls) == 1)

    parse_params(["--min-a", "3"], decls)
    var want = array([False, True, False, True, False])
    assert_true(BoxedValue(fused).execute(batch).as_bool() == want)
    assert_true(dyn.execute(batch).as_bool() == want)


def test_param_conflicting_dtype_raises() raises:
    _ = drain_params()
    _ = param("x", int64)
    with assert_raises():
        _ = param("x", string)
    _ = drain_params()


def test_param_repeated_name_keeps_the_first_default_and_help() raises:
    """`default`/`help` are first-wins on a redeclaration — documented in
    `params.mojo`, asserted here so the rule cannot drift silently."""
    _ = drain_params()
    _ = param("min-a", int64, default=7, help=String("first"))
    _ = param("min-a", int64, default=8, help=String("second"))
    var decls = drain_params()
    assert_true(len(decls) == 1)
    assert_true(decls[0].help == "first")
    assert_true(decls[0].default.value().as_int64().value() == 7)


# ---------------------------------------------------------------------------
# Temporal parameters through parse_params
# ---------------------------------------------------------------------------


def test_parse_params_binds_a_required_temporal_param() raises:
    """A temporal parameter with no default used to be unbindable: every
    `--cutoff` value fell through `_parse_scalar` to "unsupported parameter
    dtype", because `DynType.is_integer()` is variant-based and a timestamp
    never reached the integer arm."""
    _ = drain_params()
    var batch = record_batch([array([1, 2], int64)], names=["a"])
    var cutoff = param("cutoff", timestamp(second))
    var decls = drain_params()
    assert_true(decls[0].is_required())

    parse_params(["--cutoff", "1560601845"], decls)
    var out = cutoff.execute(batch)
    assert_true(out.isa[DynScalar]())
    assert_true(out[DynScalar].as_timestamp().value() == 1_560_601_845)
    assert_true(out[DynScalar].type() == DynType(timestamp(second)))


def test_parse_params_overrides_a_temporal_default() raises:
    _ = drain_params()
    var cutoff = param("cutoff", timestamp(second), default=1)
    var decls = drain_params()
    parse_params(["--cutoff", "1560601845"], decls)
    var batch = record_batch([array([1, 2], int64)], names=["a"])
    var out = cutoff.execute(batch)
    assert_true(out[DynScalar].as_timestamp().value() == 1_560_601_845)


def test_parse_params_rejects_a_malformed_temporal_token() raises:
    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="cutoff", dtype=DynType(timestamp(second))))
    with assert_raises():
        parse_params(["--cutoff", "yesterday"], decls)


# ---------------------------------------------------------------------------
# The runtime lane's param() declares defaults and help too
# ---------------------------------------------------------------------------


def test_runtime_param_declares_default_and_help() raises:
    _ = drain_params()
    _ = param(
        "min-a",
        DynType(int64),
        default=Optional(Int64Scalar(11).to_dyn()),
        help=String("lower bound"),
    )
    var decls = drain_params()
    assert_true(len(decls) == 1)
    assert_false(decls[0].is_required())
    assert_true(decls[0].default.value().as_int64().value() == 11)
    assert_true("lower bound" in render_usage(decls))


def test_runtime_param_default_applies_when_the_flag_is_absent() raises:
    _ = drain_params()
    var batch = record_batch([array([1, 5, 3, 8, 2], int64)], names=["a"])
    var dyn = col("a") > param(
        "min-a", DynType(int64), default=Optional(Int64Scalar(3).to_dyn())
    )
    var decls = drain_params()
    parse_params(List[String](), decls)
    assert_true(
        dyn.execute(batch).as_bool() == array([False, True, False, True, False])
    )


# ---------------------------------------------------------------------------
# Registry re-entry
# ---------------------------------------------------------------------------


def test_empty_drain_leaves_the_lookup_table_intact() raises:
    """A second, empty drain must not strand the runtime lane's names — that
    is what made a second `execute_cli()` in one process diverge, with fused
    cells still bound and `lookup_param` raising `unknown parameter`."""
    _ = drain_params()
    _ = param("min-a", DynType(int64))
    var decls = drain_params()
    parse_params(["--min-a", "5"], decls)
    assert_true(lookup_param("min-a").get().as_int64().value() == 5)

    _ = drain_params()
    assert_true(lookup_param("min-a").get().as_int64().value() == 5)


# ---------------------------------------------------------------------------
# execute_cli's argv splitting and output-format dispatch
# ---------------------------------------------------------------------------


def test_split_cli_args_extracts_the_output_path() raises:
    var split = split_cli_args(["--min-a", "5", "-o", "r.parquet"])
    assert_true(len(split.param_args) == 2)
    assert_true(split.param_args[0] == "--min-a")
    assert_true(split.param_args[1] == "5")
    assert_true(split.out_path.value() == "r.parquet")
    assert_false(Bool(split.format))


def test_split_cli_args_extracts_the_format_override() raises:
    var split = split_cli_args(["-o", "r.bin", "--format", "ipc", "--src", "x"])
    assert_true(split.out_path.value() == "r.bin")
    assert_true(split.format.value() == "ipc")
    assert_true(len(split.param_args) == 2)
    assert_true(split.param_args[0] == "--src")


def test_split_cli_args_keeps_everything_else_for_parse_params() raises:
    var split = split_cli_args(["--src", "a.parquet", "--min-a", "5"])
    assert_false(Bool(split.out_path))
    assert_false(Bool(split.format))
    assert_true(len(split.param_args) == 4)


def test_split_cli_args_dangling_output_flag_raises() raises:
    with assert_raises():
        _ = split_cli_args(["--min-a", "5", "-o"])


def test_split_cli_args_dangling_format_flag_raises() raises:
    with assert_raises():
        _ = split_cli_args(["--format"])


def test_cli_output_format_picks_the_writer_by_extension() raises:
    var none = Optional[String](None)
    assert_true(cli_output_format("r.parquet", none) == "parquet")
    assert_true(cli_output_format("r.arrow", none) == "ipc")
    assert_true(cli_output_format("r.txt", none) == "table")


def test_cli_output_format_override_beats_the_extension() raises:
    assert_true(
        cli_output_format("r.parquet", Optional(String("ipc"))) == "ipc"
    )


def test_cli_output_format_rejects_an_unknown_format() raises:
    with assert_raises():
        _ = cli_output_format("r.parquet", Optional(String("csv")))


def test_cli_output_parquet_needs_the_writers_define() raises:
    """Without `-D MARROW_CLI_WRITERS=true` the Parquet/IPC writers are not
    linked and `-o r.parquet` raises rather than silently printing."""
    var batch = record_batch([array([1, 2], int64)], names=["a"])
    comptime if CLI_WRITERS_ENABLED:
        pass
    else:
        with assert_raises():
            write_cli_output(
                batch, Optional(String("r.parquet")), Optional[String](None)
            )
        with assert_raises():
            write_cli_output(
                batch, Optional(String("r.arrow")), Optional[String](None)
            )
