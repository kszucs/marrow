"""The developer CLI: `python -m devkit --help`.

The only module that imports click, and the only one that assembles the objects
the rest of `devkit` defines.  Optional dependencies -- duckdb for regenerating
the golden corpus, archery for the Arrow conformance suite -- are imported
inside the command that needs them, because the `dev` environment has neither.
"""

import json
import sys
from pathlib import Path

import click

from .mojo import (
    AsanRuntime,
    BuildOptions,
    MojoToolchain,
    ProcessRunner,
    Repo,
    SilentProgress,
)
from .progress import ConsoleProgress
from .wheel import check_wheel, compile_module


class Context:
    """The objects every command needs, built once per invocation."""

    def __init__(self, repo=None, quiet=False):
        self.repo = repo or Repo.locate()
        self.quiet = quiet

    @property
    def toolchain(self):
        return MojoToolchain(self._runner(ConsoleProgress()))

    @property
    def asan_toolchain(self):
        return MojoToolchain(self._runner(ConsoleProgress()), AsanRuntime.locate())

    def timed_toolchain(self, timeout):
        """A toolchain with a deadline, for a command that compiles many
        programs in a row and would otherwise hang a CI job on one of them."""
        return MojoToolchain(self._runner(ConsoleProgress(), timeout))

    def _runner(self, progress, timeout=0):
        return ProcessRunner(
            self.repo.root,
            SilentProgress() if self.quiet else progress,
            timeout=timeout,
        )

    def fail(self, message):
        raise click.ClickException(message)


pass_context = click.make_pass_decorator(Context, ensure=False)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--quiet", is_flag=True, help="Suppress progress output.")
@click.pass_context
def cli(ctx, quiet):
    """marrow's developer tooling."""
    # Honour an injected context so a test can run a command body against a
    # scratch repository and a stub toolchain.  `--help` does not enter a body,
    # so without this the only coverage a command gets is its signature.
    if ctx.obj is None:
        ctx.obj = Context(quiet=quiet)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@cli.group()
def build():
    """Compile the tree's artefacts."""


@build.command("lib")
@click.option("--bench", is_flag=True, help="Optimize for benchmarking (-O3).")
@click.option("--asan", is_flag=True, help="Link the AddressSanitizer runtime.")
@pass_context
def build_lib(ctx, bench, asan):
    """Build python/marrow/libmarrow.so, the Python bindings."""
    toolchain = ctx.asan_toolchain if asan else ctx.toolchain
    result = toolchain.build_shared_lib(
        ctx.repo.bindings_entry,
        ctx.repo.libmarrow,
        BuildOptions.for_shared_lib(bench=bench, asan=asan),
        f"compiling {ctx.repo.libmarrow.relative_to(ctx.repo.root)}",
    )
    if not result.ok:
        ctx.fail(result.output)


@build.command("precompile")
@pass_context
def build_precompile(ctx):
    """Compile every module under marrow/ and report all diagnostics at once.

    Judge this by its output, never by its exit status: a `use of unknown
    declaration` failure ends with `failed to parse the provided Mojo source
    module` and still exits 0.
    """
    _precompile(ctx, ctx.repo.precompiled)


@build.command("package")
@pass_context
def build_package(ctx):
    """Build the distributable package artifact."""
    _precompile(ctx, ctx.repo.artifact)


def _precompile(ctx, out):
    toolchain = ctx.toolchain
    result = toolchain.precompile(ctx.repo.PACKAGE, out)
    if result.output:
        click.echo(result.output, err=True)
    if toolchain.reports_errors(result):
        ctx.fail(f"precompile reported errors; {out} is not usable")
    click.echo(f"precompiled to {out}")


# ---------------------------------------------------------------------------
# wheel
# ---------------------------------------------------------------------------


@cli.group()
def wheel():
    """Check what a built wheel ships against what it declares."""


@wheel.command("check")
@click.argument(
    "wheels",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--require",
    multiple=True,
    metavar="NAME",
    help="Also require an optional library (e.g. `opendal`); every codec is "
    "always required.",
)
@pass_context
def wheel_check(ctx, wheels, require):
    """Fail unless every shared library in each (repaired) wheel carries its
    licence texts, METADATA declares them, every codec (and each --require'd
    library) is present, and nothing forbidden ships."""
    catalog = compile_module(ctx.repo)
    failures = []
    for path in wheels:
        problems = check_wheel(path, catalog, require)
        failures.extend(f"{path.name}: {problem}" for problem in problems)
        if not problems:
            click.echo(f"{path.name}: ok")
    if failures:
        ctx.fail("\n".join(failures))


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------


@cli.group()
def size():
    """Measure the AOT lane's binary size."""


def _gates(ctx):
    """The gate programs, with a silent runner for nm/size/strip.

    Those are sub-second and there are hundreds of them; a progress display per
    call would bury the tables they exist to produce.  The *builds* still go
    through the toolchain's own display.
    """
    from .footprint import Gates

    return Gates(
        ctx.repo, ctx.toolchain, ProcessRunner(ctx.repo.root, SilentProgress())
    )


@size.command("compare")
@click.argument("gates_wanted", metavar="[GATES]...", nargs=-1)
@pass_context
def size_compare(ctx, gates_wanted):
    """Build every gate and report sizes, ratios and per-module symbol counts.

    Naming gates measures only those; the ratio baseline is always included.
    A full sweep is every program in the gate directory, at -O3.
    """
    from .footprint import Report

    gates = _gates(ctx)
    try:
        names = gates.resolve(gates_wanted)
    except ValueError as error:
        ctx.fail(str(error))

    failed = gates.build_all(names)
    measured = [name for name in names if name not in failed]
    Report().comparison(
        [gates.measure(name) for name in measured],
        failed,
        gates.attribute(measured),
    )
    if failed:
        ctx.fail(
            f"{', '.join(failed)} did not build; every other gate above was "
            "still measured."
        )


@size.command("check")
@click.option("--update", is_flag=True, help="Re-record the baseline instead.")
@click.option(
    "--repo",
    "repo_path",
    type=click.Path(exists=True, file_okay=False),
    help="Build the gates from this checkout rather than this one.",
)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Measurements to compare against, instead of baseline.json.",
)
@click.option(
    "--out", "out_path", type=click.Path(), help="Also write the measurements here."
)
@click.option(
    "--measure-only",
    is_flag=True,
    help="Measure and write --out, without comparing.",
)
@pass_context
def size_check(ctx, update, repo_path, baseline_path, out_path, measure_only):
    """Fail if any recorded gate grew past its baseline `__text` size.

    The committed floor is a developer-machine record and is what a local run
    compares against. CI cannot: the same source builds 0.5-1.6% larger on a
    runner, which trips the threshold on its own and reported six REGRESSIONs
    against a baseline re-recorded the same day. It measures both ends on one
    machine instead --

        devkit size check --repo ../base --out base.json --measure-only
        devkit size check --baseline base.json

    -- so what is compared is the change, not the machine.
    """
    from .footprint import Baseline, Report

    gates = _gates(ctx)
    # The gate *list* always comes from this checkout: a gate this commit adds
    # is absent from the commit it descends from, and has nothing to compare to
    # yet rather than being a failure.
    reference = Baseline(gates.directory / "baseline.json")
    names = list(reference.gates)

    if repo_path:
        gates = _gates(Context(repo=Repo(repo_path), quiet=ctx.quiet))
        names = [name for name in names if gates.source(name).exists()]

    failed = gates.build_all(names)
    if failed:
        ctx.fail(f"{', '.join(failed)} did not build")

    measured = {name: gates.text_size(name) for name in names}
    if out_path:
        Path(out_path).write_text(json.dumps(measured, indent=2) + "\n")
        click.echo(f"wrote {len(measured)} measurements to {out_path}")
    if measure_only:
        return
    if update:
        reference.update(measured)
        click.echo(f"wrote new baseline to {reference.path}")
        return

    against = reference
    if baseline_path:
        against = Baseline(baseline_path, threshold_pct=reference.threshold_pct)
    if Report().gate(against.check(measured), against.threshold_pct):
        sys.exit(1)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("script")
@click.option(
    "--sample",
    "use_sample",
    is_flag=True,
    help="Use macOS `sample` instead of xctrace; it resolves Mojo frames better.",
)
@click.option("--open/--no-open", "open_after", default=True, help="Open the result.")
@pass_context
def profile(ctx, script, use_sample, open_after):
    """Profile a Mojo or Python workload with Instruments or `sample`."""
    import subprocess

    from .profiling import ProfileTarget, Profiler, SampleRecorder, TraceRecorder

    try:
        target = ProfileTarget(ctx.repo, script)
    except (FileNotFoundError, ValueError) as error:
        ctx.fail(str(error))

    recorder = SampleRecorder() if use_sample else TraceRecorder()
    try:
        destination, kept = Profiler(ctx.repo, ctx.toolchain, recorder).run(target)
    except RuntimeError as error:
        ctx.fail(str(error))

    click.echo(f"saved to {destination}")
    if kept is not None:
        click.echo(f"binary kept at {kept} (needed for symbolication)")
    if open_after:
        subprocess.run(["open", str(destination)])


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


@cli.group()
def golden():
    """The cross-lane golden query corpus."""


@golden.command("prepare")
@pass_context
def golden_prepare(ctx):
    """Write the fixtures, expectations and generated Mojo wrappers."""
    from .golden import Corpus

    click.echo(f"prepared {len(Corpus(ctx.repo).prepare())} cases")


@golden.command("sql")
@pass_context
def golden_sql(ctx):
    """Report how much of the corpus the SQL front end answers.

    The declined column is the list to work down when widening the grammar.
    """
    from .golden import SqlLane, corpus

    # `marrow` lives under `python/`, which only `pytest.ini` puts on the path.
    # The predecessor of this command was documented as
    # `python golden/test_sql_cases.py` and had never worked for that reason.
    sys.path.insert(0, str(ctx.repo.python_dir))
    # Through the singleton, and prepared: the lane reads the fixture *files*,
    # so measuring against a stale `.arrow` would report the previous `TABLES`.
    lane = corpus()
    lane.prepare()
    wrong = SqlLane(lane).report(echo=click.echo)
    if wrong:
        ctx.fail(f"{len(wrong)} case(s) answered differently from DuckDB")


@golden.command("regenerate")
@pass_context
def golden_regenerate(ctx):
    """Recompute every case's expected result with DuckDB and rewrite it in place.

    Needs the `bench` environment: duckdb is not in `dev`.
    """
    from .golden import Corpus

    Corpus(ctx.repo).regenerate(echo=click.echo)


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------


@cli.group()
def docs():
    """The documentation site's Mojo listings."""


@docs.command("check")
@pass_context
def docs_check(ctx):
    """Compile every Mojo listing under docs/, so the guides cannot rot.

    Judged by its output, never by its exit status: `mojo build` reports a
    parse failure and still exits 0.
    """
    from .docs import Report, SnippetCheck

    check = SnippetCheck(ctx.repo, ctx.timed_toolchain(SnippetCheck.TIMEOUT))
    if not check.run(Report()):
        sys.exit(1)


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------


@cli.group()
def integration():
    """Arrow protocol conformance, via apache/arrow's archery."""


@integration.command("run")
@click.option("--run-ipc", is_flag=True, help="Run IPC producer/consumer tests.")
@click.option("--run-c-data", is_flag=True, help="Run C Data Interface tests.")
@click.option("--with-cpp", is_flag=True, help="Cross-test against C++ Arrow.")
@click.option("--with-rust", is_flag=True, help="Cross-test against arrow-rs.")
@click.option("--with-go", is_flag=True, help="Cross-test against arrow-go.")
@click.option("--no-stop-on-error", is_flag=True, help="Continue after failures.")
@click.option("--match", default=None, metavar="PATTERN", help="Filter case names.")
@click.option(
    "--gold-dirs",
    multiple=True,
    metavar="DIR",
    help="Paths to golden .arrow_file directories.",
)
@pass_context
def integration_run(
    ctx,
    run_ipc,
    run_c_data,
    with_cpp,
    with_rust,
    with_go,
    no_stop_on_error,
    match,
    gold_dirs,
):
    """Run the archery integration suite with marrow as a participant."""
    if not (run_ipc or run_c_data):
        ctx.fail("specify at least one of --run-ipc or --run-c-data")

    from .integration import ArcherySuite

    suite = ArcherySuite()
    ok = suite.run(
        run_ipc=run_ipc,
        run_c_data=run_c_data,
        with_cpp=with_cpp,
        with_rust=with_rust,
        with_go=with_go,
        stop_on_error=not no_stop_on_error,
        match=match,
        gold_dirs=list(gold_dirs) or None,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    cli()
