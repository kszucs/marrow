"""Benchmark suite infrastructure mirroring TestSuite.

Provides `BenchSuite` for collecting, filtering, and executing benchmarks
with automatic discovery of `bench_*` functions, `--only`/`--skip` CLI
filtering, and structured JSON output consumable by pytest.

Example usage::

    from std.benchmark import Bencher, BenchMetric, keep
    from marrow.testing import BenchSuite, Benchmark

    def bench_sum_1k(mut b: Benchmark) raises:
        var arr = arange[Int64Type](0, 1_000)
        b.throughput(BenchMetric.elements, 1_000)

        @always_inline
        @parameter
        def call():
            var r = sum_(arr)
            keep(r)

        b.iter[call]()

    def main() raises:
        var suite = BenchSuite.discover_benches[__functions_in_module()]()
        suite.run()
"""

from std.collections import Set, Dict
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric
from std.reflection import get_function_name, call_location, SourceLocation
from std.sys import argv
from std.sys.intrinsics import _type_is_eq
from std.time import perf_counter_ns


struct CLIFlags:
    """Parsed CLI flags for bench suites."""

    var list_cases: Bool
    var json_output: Bool
    var args: List[StaticString]

    def __init__(out self):
        self.args = List[StaticString](argv())
        self.list_cases = False
        self.json_output = False

        var i = 1
        while i < len(self.args):
            if self.args[i] == "--list":
                self.list_cases = True
            elif self.args[i] == "--json":
                self.json_output = True
            i += 1


def _print_json_array(names: List[String]):
    """Print a JSON array of strings to stdout."""
    print("[")
    for i in range(len(names)):
        var comma = "," if i < len(names) - 1 else ""
        print('  "' + names[i] + '"' + comma)
    print("]")


# ── Public types ───────────────────────────────────────────────────────────


struct Benchmark:
    """Wraps ``Bencher`` with throughput annotations.

    Benchmark functions receive this instead of raw ``Bencher``.  Call
    ``b.throughput(metric, count)`` to declare throughput, then use
    ``b.iter[fn]()`` exactly like ``Bencher``.
    """

    var _bencher: Bencher
    var _throughput: Optional[_ThroughputMeasure]

    def __init__(out self, num_iters: Int):
        self._bencher = Bencher(num_iters)
        self._throughput = None

    def throughput(mut self, metric: BenchMetric, count: Int):
        """Declare throughput for this benchmark.

        Args:
            metric: The metric kind (e.g. ``BenchMetric.elements``, ``.bytes``).
            count: Elements/bytes/flops processed **per iteration**.
        """
        self._throughput = _ThroughputMeasure(metric.name, metric.unit, count)

    # ── Forward all Bencher methods ────────────────────────────────────

    def iter[iter_fn: def() capturing[_] -> None](mut self):
        self._bencher.iter[iter_fn]()

    def iter[iter_fn: def() capturing raises -> None](mut self) raises:
        self._bencher.iter[iter_fn]()

    def iter_custom[iter_fn: def(Int) raises capturing[_] -> Int](mut self):
        self._bencher.iter_custom[iter_fn]()

    def get_num_iters(self) -> Int:
        return self._bencher.num_iters

    def get_elapsed(self) -> Int:
        return self._bencher.elapsed


# ── Internal types ──────────────────────────────────────────────────────────


@fieldwise_init
struct _Bench(Copyable):
    """A single benchmark to run."""

    comptime fn_type = def(mut Benchmark) capturing raises

    var bench_fn: Self.fn_type
    var name: StaticString


@fieldwise_init
struct _ThroughputMeasure(Copyable, Movable):
    """A throughput annotation: metric name/unit + element count."""

    var metric_name: String
    var metric_unit: String
    var count: Int


# ── BenchSuite ──────────────────────────────────────────────────────────────


struct BenchSuite(Movable):
    """A suite of benchmarks to discover, filter, execute and report.

    Mirrors the TestSuite API:
    - `discover_benches[__functions_in_module()]()` auto-discovers `bench_*`
    - `--only` / `--skip` CLI filtering
    - `run()` executes and prints JSON results for pytest consumption

    Each benchmark function has the signature `def bench_*(mut b: Bencher) raises`.
    Inside the function, call `b.iter[fn]()` (or `b.iter_custom`, etc.) to
    time the hot loop — exactly like `std.benchmark.Bench` usage.
    """

    var benches: List[_Bench]
    var location: SourceLocation
    var skip_list: Set[String]
    var allow_list: Optional[Set[String]]
    var cli_args: List[StaticString]
    var config: BenchConfig

    @always_inline
    def __init__(
        out self,
        *,
        location: Optional[SourceLocation] = None,
        var cli_args: Optional[List[StaticString]] = None,
        config: Optional[BenchConfig] = None,
    ) raises:
        self.benches = List[_Bench]()
        self.location = location.or_else(call_location())
        self.skip_list = Set[String]()
        self.allow_list = None
        self.cli_args = cli_args^.or_else(List[StaticString](argv()))
        self.config = config.value().copy() if config else BenchConfig()

    # ── Registration ────────────────────────────────────────────────────

    def _register_benches[bench_funcs: Tuple, /](mut self) raises:
        comptime for idx in range(len(bench_funcs)):
            comptime func = bench_funcs[idx]

            comptime if get_function_name[func]().startswith("bench_"):
                # Only register functions matching the expected signature.
                # Other bench_* functions (e.g. parameterized helpers) are
                # silently skipped — they can be registered manually.
                comptime if _type_is_eq[type_of(func), _Bench.fn_type]():
                    self.bench[rebind[_Bench.fn_type](func)]()

    @always_inline
    @staticmethod
    def discover_benches[
        bench_funcs: Tuple, /
    ](
        *,
        location: Optional[SourceLocation] = None,
        var cli_args: Optional[List[StaticString]] = None,
        config: Optional[BenchConfig] = None,
    ) raises -> Self:
        """Discover benchmarks from `__functions_in_module()`.

        Registers all functions whose names start with `bench_`.

        Parameters:
            bench_funcs: The pack of functions (pass `__functions_in_module()`).

        Args:
            location: Source location (defaults to call site).
            cli_args: CLI arguments (defaults to `sys.argv()`).
            config: Benchmark configuration.

        Returns:
            A new BenchSuite with all discovered benchmarks.
        """
        var suite = Self(
            location=location.or_else(call_location()),
            cli_args=cli_args^,
            config=config,
        )
        suite._register_benches[bench_funcs]()
        return suite^

    @staticmethod
    def run[bench_funcs: Tuple, /]() raises:
        """Discover benchmarks, optionally list them, and run.

        Parameters:
            bench_funcs: Pass ``__functions_in_module__()``.
        """
        var suite = Self.discover_benches[bench_funcs]()
        suite.run()

    def bench[f: _Bench.fn_type](mut self):
        """Register a benchmark function.

        Parameters:
            f: The benchmark function.
        """
        self.benches.append(_Bench(f, get_function_name[f]()))

    def skip[f: _Bench.fn_type](mut self):
        """Mark a benchmark to be skipped.

        Parameters:
            f: The benchmark function to skip.
        """
        comptime name = get_function_name[f]()
        self.skip_list.add(name)

    # ── CLI filtering ───────────────────────────────────────────────────

    def _parse_filter_lists(mut self) raises:
        ref args = self.cli_args
        var num_args = len(args)
        if num_args <= 1:
            return

        if args[1] == "--only":
            self.allow_list = Set[String]()
        elif args[1] == "--skip-all":
            if num_args > 2:
                raise Error("'--skip-all' does not take any arguments")
            self.allow_list = Set[String]()
            return
        elif args[1] != "--skip":
            raise Error(
                "invalid argument: ",
                args[1],
                " (expected '--only' or '--skip')",
            )

        if num_args == 2:
            raise Error("expected benchmark name(s) after '--only' or '--skip'")

        var discovered = Set[String]()
        for b in self.benches:
            discovered.add(b.name)

        for idx in range(2, num_args):
            var arg = args[idx]
            if arg not in discovered:
                raise Error(
                    "explicitly ",
                    "allowed" if self.allow_list else "skipped",
                    " benchmark not found in suite: ",
                    arg,
                )
            if self.allow_list:
                self.allow_list[].add(arg)
            else:
                self.skip_list.add(arg)

    def _should_skip(self, b: _Bench) -> Bool:
        if b.name in self.skip_list:
            return True
        if not self.allow_list:
            return False
        return b.name not in self.allow_list.unsafe_value()

    def _validate_skip_list(self) raises:
        for name in self.skip_list:
            var found = False
            for b in self.benches:
                if b.name == name:
                    found = True
                    break
            if not found:
                raise Error(
                    "trying to skip a benchmark not registered in the suite: ",
                    name,
                )

    # ── Execution ───────────────────────────────────────────────────────

    def run(mut self) raises:
        """Run all benchmarks, print results.

        CLI flags:
        - ``--list``  Print JSON array of bench names and exit.
        - ``--json``  Print JSON results (for pytest consumption).
        - Default: human-readable output + JSON markers.
        """
        var flags = CLIFlags()

        if flags.list_cases:
            var names = List[String]()
            for b in self.benches:
                names.append(String(b.name))
            _print_json_array(names)
            return

        # Strip --list and --json so _parse_filter_lists only sees
        # --only / --skip / name arguments.
        var stripped = List[StaticString]()
        for arg in flags.args:
            if arg != "--list" and arg != "--json":
                stripped.append(arg)
        self.cli_args = stripped^

        self._validate_skip_list()
        self._parse_filter_lists()

        var results = List[_BenchResult]()

        for b in self.benches:
            if self._should_skip(b):
                if not flags.json_output:
                    print("  SKIP", b.name)
                continue
            var result = self._run_one(b)
            if not flags.json_output:
                var line = (
                    "  " + String(b.name) + "  " + _format_ns(result.mean_ns())
                )
                if result.throughput:
                    var tp = result.throughput.value().copy()
                    var rate = (
                        Float64(tp.count) * 1e-9 / (result.mean_ns() * 1e-9)
                    )
                    line += "  (" + String(rate) + " " + tp.metric_unit + ")"
                print(line)
            results.append(result^)

        if flags.json_output:
            Self._print_json(results)

    @staticmethod
    def _print_json(results: List[_BenchResult]):
        """Print benchmark results as a JSON array."""
        print("[")
        for i in range(len(results)):
            ref r = results[i]
            var comma = "," if i < len(results) - 1 else ""
            var runs_str = String("[")
            for j in range(len(r.runs_ns)):
                if j > 0:
                    runs_str += ", "
                runs_str += String(r.runs_ns[j])
            runs_str += "]"
            var tp_str = String("")
            if r.throughput:
                var tp = r.throughput.value().copy()
                tp_str = (
                    ', "throughput_metric": "'
                    + tp.metric_name
                    + '", "throughput_unit": "'
                    + tp.metric_unit
                    + '", "throughput_count": '
                    + String(tp.count)
                )
            print(
                '  {"name": "'
                + r.name
                + '", "unit": "ns", "iters": '
                + String(r.iters)
                + ', "runs": '
                + runs_str
                + tp_str
                + "}"
                + comma
            )
        print("]")

    def _run_one(self, b: _Bench) raises -> _BenchResult:
        """Run a single benchmark, returning per-repetition timings."""
        var num_iters = 1
        var target_ns = Int(self.config.min_runtime_secs * 1e9)
        if target_ns <= 0:
            target_ns = 100_000_000  # 100ms default

        # Warmup — also captures throughput declaration from the first call.
        var warmup_bm = Benchmark(1)
        b.bench_fn(warmup_bm)
        var tp: Optional[_ThroughputMeasure] = None
        if warmup_bm._throughput:
            tp = warmup_bm._throughput.value().copy()

        for _ in range(self.config.num_warmup_iters):
            var bm = Benchmark(1)
            b.bench_fn(bm)

        # Calibrate: find an iteration count that runs for at least target_ns.
        while True:
            var bm = Benchmark(num_iters)
            b.bench_fn(bm)
            if bm.get_elapsed() >= target_ns or num_iters >= 1_000_000_000:
                break
            if bm.get_elapsed() <= 0:
                num_iters *= 10
            else:
                num_iters = Int(
                    Float64(target_ns)
                    * Float64(num_iters)
                    / Float64(bm.get_elapsed())
                    * 1.2
                )
                num_iters = max(num_iters, 1)

        # Measure — collect per-repetition mean-per-iteration times.
        var num_reps = max(self.config.num_repetitions, 5)
        var runs_ns = List[Float64](capacity=num_reps)
        for _ in range(num_reps):
            var bm = Benchmark(num_iters)
            b.bench_fn(bm)
            runs_ns.append(Float64(bm.get_elapsed()) / Float64(num_iters))

        return _BenchResult(String(b.name), num_iters, runs_ns^, tp^)


@fieldwise_init
struct _BenchResult(Copyable, Movable):
    var name: String
    var iters: Int
    var runs_ns: List[Float64]
    var throughput: Optional[_ThroughputMeasure]

    def mean_ns(self) -> Float64:
        if len(self.runs_ns) == 0:
            return 0.0
        var total = Float64(0)
        for i in range(len(self.runs_ns)):
            total += self.runs_ns[i]
        return total / Float64(len(self.runs_ns))


def _format_ns(ns: Float64) -> String:
    """Format nanoseconds as a human-readable string."""
    if ns < 1_000:
        return String(ns) + " ns"
    if ns < 1_000_000:
        return String(ns / 1_000) + " us"
    if ns < 1_000_000_000:
        return String(ns / 1_000_000) + " ms"
    return String(ns / 1_000_000_000) + " s"
