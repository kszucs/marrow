"""The AOT size gate: reading the platform tools, deriving modules, the floor."""

import json
from pathlib import Path

import pytest

from devkit.footprint import Baseline, Gates, MachO, Modules, Report
from devkit.mojo import BuildOptions, Repo


# ---------------------------------------------------------------------------
# Reading the tools
# ---------------------------------------------------------------------------


def test_text_section_is_read_not_the_segment_or_the_file():
    """`__text` is code only; `__TEXT` and the file size are page-padded.

    Measured 2026-07-29: a change adding 1,728 bytes of code moved the stripped
    file size by 16,504 and the segment by exactly 16,384 -- one 16 KB page on
    Apple Silicon.  A gate reading either cannot see a change smaller than a
    page, so the section is the only usable figure.
    """
    output = "\n".join(
        [
            "Segment __TEXT: 5533696",
            "\tSection __text: 5266164",
            "\tSection __stubs: 4104",
            "\ttotal 5522804",
        ]
    )
    assert MachO.parse_text(output) == 5266164


def test_a_binary_without_a_text_section_answers_none():
    assert MachO.parse_text("Segment __TEXT: 100\n") is None
    assert MachO.parse_text("") is None


def test_symbol_names_survive_embedded_spaces():
    """Demangled Mojo names carry nested generic signatures, spaces included.

    Taking the last whitespace-separated token would truncate every one of them.
    """
    output = "\n".join(
        [
            "0000000100003a10 T marrow::kernels::filter::Filter[T: DType]::apply",
            "                 U _malloc",
            "0000000100004000 T marrow::arrays::Primitive[Int64Type, mut=True]::init",
            "",
        ]
    )
    assert MachO.parse_names(output) == [
        "marrow::kernels::filter::Filter[T: DType]::apply",
        "_malloc",
        "marrow::arrays::Primitive[Int64Type, mut=True]::init",
    ]


def test_blank_lines_are_not_symbols():
    assert MachO.parse_names("\n\n   \n") == []


# ---------------------------------------------------------------------------
# Deriving the modules
# ---------------------------------------------------------------------------


def package(tmp_path, *relative):
    root = tmp_path / "marrow"
    for path in relative:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    return root


def test_modules_come_from_the_source_tree(tmp_path):
    """The bucket list is a fact about the checkout, not a maintained constant.

    The hand-written list this replaced named four `marrow::expr::*` modules
    that had been renamed, so they reported 0 for every gate -- a table that
    looks complete and measures nothing.
    """
    root = package(
        tmp_path, "arrays.mojo", "kernels/filter.mojo", "expr/comptime/core.mojo"
    )
    assert Modules(root).names() == [
        "marrow::arrays",
        "marrow::expr::comptime::core",
        "marrow::kernels::filter",
    ]


def test_tests_and_package_markers_are_not_modules(tmp_path):
    """Nothing under a `tests/` directory is ever linked into a gate."""
    root = package(
        tmp_path,
        "arrays.mojo",
        "__init__.mojo",
        "tests/test_arrays.mojo",
        "kernels/tests/bench_filter.mojo",
        "kernels/__init__.mojo",
    )
    assert Modules(root).names() == ["marrow::arrays"]


def test_counting_keeps_only_the_modules_a_binary_names(tmp_path):
    root = package(tmp_path, "arrays.mojo", "buffers.mojo", "parquet/reader.mojo")
    counts = Modules(root).count(
        ["marrow::arrays::alloc", "marrow::arrays::slice", "marrow::buffers::view"]
    )
    assert counts == {"marrow::arrays": 2, "marrow::buffers": 1}
    assert "marrow::parquet::reader" not in counts


def test_a_nested_module_name_does_not_absorb_its_neighbour(tmp_path):
    """`marrow::kernels::cast` is a prefix of `marrow::kernels::cast_decimal`.

    A bare substring test counts every `cast_decimal` symbol as a `cast` symbol
    too, and the table then prints both rows as if they were independent.
    Measured on the real `query_dynvalue`, that inflated `cast` by 7.6%.
    """
    root = package(tmp_path, "kernels/cast.mojo", "kernels/cast_decimal.mojo")
    counts = Modules(root).count(
        [
            "marrow::kernels::cast::Cast::apply",
            "marrow::kernels::cast_decimal::Widen::apply",
            "marrow::kernels::cast_decimal::Narrow::apply",
        ]
    )
    assert counts == {
        "marrow::kernels::cast": 1,
        "marrow::kernels::cast_decimal": 2,
    }


def test_buckets_are_proportional_not_a_partition(tmp_path):
    """A mangled name embeds nested type parameters, so it may name two modules.

    That is deliberate: the table measures where a binary's code came from, and
    a symbol genuinely belonging to both should count for both.
    """
    root = package(tmp_path, "arrays.mojo", "kernels/filter.mojo")
    counts = Modules(root).count(["marrow::kernels::filter::over[marrow::arrays::T]"])
    assert counts == {"marrow::arrays": 1, "marrow::kernels::filter": 1}


def test_a_missing_package_tree_is_an_error_not_an_empty_table(tmp_path):
    """Answering [] would print an empty attribution table.

    That reads as "this binary links nothing", which is a plausible-looking
    wrong answer rather than a visible failure.
    """
    import pytest

    with pytest.raises(RuntimeError, match="no package source tree"):
        Modules(tmp_path / "nowhere").names()


# ---------------------------------------------------------------------------
# Discovering the gates
# ---------------------------------------------------------------------------


def gate_dir(tmp_path, *names):
    directory = tmp_path / "benchmarks" / "binary_size"
    directory.mkdir(parents=True)
    for name in names:
        (directory / f"{name}.mojo").touch()
    return Gates(Repo(tmp_path), None, None)


def test_gates_are_whichever_programs_exist(tmp_path):
    """Adding a gate program must not also require editing a list."""
    gates = gate_dir(tmp_path, "query_join", "query_streaming", "query_sort")
    assert gates.available() == ["query_join", "query_sort", "query_streaming"]


def test_naming_a_gate_keeps_the_ratio_baseline(tmp_path):
    gates = gate_dir(tmp_path, "query_join", "query_streaming")
    assert gates.resolve(["query_join"]) == ["query_join", Gates.BASELINE]
    assert gates.resolve([]) == ["query_join", "query_streaming"]


def test_an_unknown_gate_is_rejected_by_name(tmp_path):
    """A typo must not silently measure the baseline alone."""
    gates = gate_dir(tmp_path, "query_streaming")
    try:
        gates.resolve(["query_nope"])
    except ValueError as error:
        assert "unknown gate" in str(error)
    else:
        raise AssertionError("expected a ValueError")


# ---------------------------------------------------------------------------
# The recorded floor
# ---------------------------------------------------------------------------


def baseline(tmp_path, gates, threshold=0.5):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"threshold_pct": threshold, "gates": gates}))
    return Baseline(path)


def test_growth_past_the_threshold_is_a_regression(tmp_path):
    """0.5%, not the usual 1%: the regression that motivated this gate was
    0.63% of `query_streaming`, and a 1% threshold would have missed it."""
    recorded = baseline(tmp_path, {"a": 1_000_000})
    [(name, floor, text, delta, pct, regressed)] = recorded.check({"a": 1_010_000})
    assert (name, floor, text, delta) == ("a", 1_000_000, 1_010_000, 10_000)
    assert round(pct, 3) == 1.0 and regressed


def test_growth_within_the_threshold_is_not(tmp_path):
    assert not baseline(tmp_path, {"a": 1_000_000}).check({"a": 1_004_000})[0][5]


def test_shrinking_is_never_a_regression(tmp_path):
    _, _, _, delta, pct, regressed = baseline(tmp_path, {"a": 1_000_000}).check(
        {"a": 500_000}
    )[0]
    assert delta == -500_000 and pct == -50.0 and not regressed


def test_an_unmeasurable_gate_is_an_error(tmp_path):
    """`__text` can come back None, and `None - floor` surfaces frames away."""
    import pytest

    recorded = baseline(tmp_path, {"a": 1_000})
    with pytest.raises(RuntimeError, match="no __text measurement"):
        recorded.check({"a": None})
    with pytest.raises(RuntimeError, match="no __text measurement"):
        recorded.check({})


def test_a_zero_baseline_is_an_error(tmp_path):
    recorded = baseline(tmp_path, {"a": 0})
    with pytest.raises(RuntimeError, match="zero baseline"):
        recorded.check({"a": 1})


def test_the_report_names_every_regressed_gate(tmp_path):
    """CI reads the returned list; an empty one is what lets the job pass."""
    recorded = baseline(tmp_path, {"grew": 1_000, "held": 1_000})
    assert Report().gate(
        recorded.check({"grew": 2_000, "held": 1_001}), recorded.threshold_pct
    ) == ["grew"]
    assert Report().gate(recorded.check({"grew": 1_000, "held": 1_000}), 0.5) == []


def test_updating_rewrites_only_the_gates(tmp_path):
    """The `_comment` is a changelog of why each number moved; it must survive."""
    recorded = baseline(tmp_path, {"a": 1_000})
    stored = json.loads(recorded.path.read_text())
    stored["_comment"] = "why these numbers are what they are"
    recorded.path.write_text(json.dumps(stored))

    Baseline(recorded.path).update({"a": 2_000})
    after = json.loads(recorded.path.read_text())
    assert after["gates"] == {"a": 2_000}
    assert after["threshold_pct"] == 0.5
    assert after["_comment"] == "why these numbers are what they are"


def test_growth_exactly_at_the_threshold_is_not_a_regression(tmp_path):
    """The threshold is the largest tolerated growth, not the smallest rejected."""
    recorded = baseline(tmp_path, {"a": 1_000_000}, threshold=0.5)
    assert not recorded.check({"a": 1_005_000})[0][5]  # exactly +0.500%
    assert recorded.check({"a": 1_005_001})[0][5]


# ---------------------------------------------------------------------------
# Guards that exist to prevent a plausible-but-wrong number
# ---------------------------------------------------------------------------


def test_only_the_text_section_matches():
    """Other `Section` lines, and the `__TEXT` segment, are not code size."""
    assert MachO.parse_text("\tSection __stubs: 4104\n") is None
    assert MachO.parse_text("Segment __TEXT: 5533696\n") is None
    assert MachO.parse_text("\tSection __TEXT_EXEC: 99\n") is None
    assert MachO.parse_text("\tSection __text: 12\n") == 12


def test_a_failing_tool_raises_rather_than_measuring_nothing(tmp_path):
    """`nm` or `size` failing silently would report a plausible zero."""
    from devkit.mojo import CommandResult

    class FailingRunner:
        def run(self, argv, label):
            return CommandResult(
                argv=(), returncode=1, stdout="", stderr="no such tool", elapsed=0.0
            )

    binary = MachO(tmp_path / "x", FailingRunner())
    for read in (lambda: binary.names, lambda: binary.text):
        with pytest.raises(RuntimeError, match="failed on"):
            read()


def test_a_failed_build_leaves_no_stale_binary(tmp_path):
    """`mojo build` leaves the previous artefact in place when it fails.

    Measuring without deleting first reports the stale binary's size as if the
    failed build had succeeded -- exactly the plausible-but-wrong number this
    gate exists to prevent.
    """
    from devkit.mojo import CommandResult

    directory = tmp_path / "benchmarks" / "binary_size"
    directory.mkdir(parents=True)
    (directory / "query_x.mojo").touch()
    (directory / "query_x").write_bytes(b"stale binary")
    (directory / "query_x_stripped").write_bytes(b"stale stripped")

    class FailingToolchain:
        def build(self, source, out, options, label):
            return CommandResult(
                argv=(), returncode=1, stdout="", stderr="boom", elapsed=0.0
            )

    gates = Gates(Repo(tmp_path), FailingToolchain(), None)
    assert gates.build_all(["query_x"]) == ["query_x"]
    assert not (directory / "query_x").exists()
    assert not (directory / "query_x_stripped").exists()


def test_marrow_compile_builds_a_query_the_way_the_gate_builds_one():
    """`marrow compile` carries its own copy of the recipe, and must not drift.

    `python/marrow/compile.py` ships *inside the wheel*, where `devkit` does not
    exist, so it cannot import `BuildOptions` the way every other caller does --
    the duplication is forced.  What it buys is the guarantee its docstring
    makes: a user's compiled query is measured on the same terms as the gate
    programs, so the numbers in `benchmarks/binary_size/` describe their binary
    too.  A change to `for_size_gate` alone silently ends that.

    Loaded by path rather than imported: `marrow/__init__.py` pulls in the
    compiled extension, and this suite must run without `libmarrow.so`.
    `compile.py` needs only the standard library, so a path load is enough.
    """
    import importlib.util

    path = Repo.locate().python_dir / Repo.PACKAGE / "compile.py"
    spec = importlib.util.spec_from_file_location("_marrow_compile", path)
    compile_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compile_module)

    gate = BuildOptions.for_size_gate()
    expected = ["mojo", "build", *gate.flags(), "q.mojo", "-o", "q"]
    # The gate compiles from the repository root, so its `-I .` *is* the
    # checkout; `marrow compile` runs from anywhere and names the path outright.
    expected[expected.index(".")] = "/marrow/checkout"

    assert (
        compile_module.build_command(
            Path("q.mojo"), Path("q"), Path("/marrow/checkout")
        )
        == expected
    )
