"""Benchmark results: injecting them, comparing them, and keeping them.

The only module in `devkit` that talks to pytest-benchmark.  A Mojo benchmark is
measured by the generated runner and arrives here as JSON, so `BenchmarkInjector`
has to hand pytest-benchmark timings it did not take itself; everything
downstream -- the throughput column, the competition table, the rolling history
-- then treats Mojo and Python benchmarks identically.
"""

import json
import re
from itertools import chain
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

#: The extra_info key the terminal report reads a throughput off.  Spelled with
#: its unit because pytest-benchmark prints the key verbatim as a column header.
THROUGHPUT_KEY = "throughput (GElems/s)"


# ---------------------------------------------------------------------------
# Getting Mojo timings into pytest-benchmark
# ---------------------------------------------------------------------------


class BenchmarkInjector:
    """Feeds an already-measured Mojo benchmark into pytest-benchmark.

    The Mojo runner times its own cases and reports them as JSON, so there is
    nothing here for pytest-benchmark to call.  A fake timer that yields the
    recorded durations lets `pedantic()` record them as ordinary rounds, which
    is what gives Mojo benchmarks real min/max/stddev statistics and puts them
    in the same table as the Python ones.
    """

    #: Divisors, not reciprocals: `100 / 1e9` is exactly 1e-7 where
    #: `100 * 1e-9` is not, and the history compares numbers across commits.
    PER_SECOND = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}

    def __init__(self, session):
        self._session = session

    @classmethod
    def to_seconds(cls, value, unit):
        return value / cls.PER_SECOND.get(unit, 1.0)

    def inject(self, name, node_id, entry):
        """Record *entry*'s timings against a synthetic benchmark node."""
        import types

        from pytest_benchmark.fixture import BenchmarkFixture
        from pytest_benchmark.utils import NameWrapper

        if self._session.disabled:
            return None

        unit = entry.get("unit", "ns")
        runs = entry.get("runs")
        if runs:
            durations = [self.to_seconds(value, unit) for value in runs]
        else:
            durations = [self.to_seconds(entry["value"], unit)]

        # pedantic() calls the timer twice per round -- start, then end -- so a
        # (0, d1, 0, d2, ...) sequence reads back as one round per duration.
        ticks = iter([tick for duration in durations for tick in (0.0, duration)])
        nothing = lambda *_: None  # noqa: E731 - a logger/warner sink

        fixture = BenchmarkFixture(
            node=types.SimpleNamespace(name=name, _nodeid=node_id),
            add_stats=self._session.benchmarks.append,
            logger=nothing,
            warner=nothing,
            disabled=self._session.disabled,
            timer=NameWrapper(lambda: next(ticks)),
            disable_gc=False,
            min_rounds=1,
            min_time=0,
            max_time=0,
            calibration_precision=10,
            warmup=False,
            warmup_iterations=0,
            cprofile=False,
            cprofile_loops=None,
            cprofile_dump=None,
        )
        fixture.pedantic(
            lambda: None, rounds=len(durations), iterations=1, warmup_rounds=0
        )
        self._attach_throughput(fixture, entry)
        return fixture

    @staticmethod
    def _attach_throughput(fixture, entry):
        count = entry.get("throughput_count")
        if not (count and fixture.stats):
            return
        mean = fixture.stats.stats.mean
        if mean <= 0:
            return
        metric = entry.get("throughput_metric", "throughput")
        unit = entry.get("throughput_unit", "GElems/s")
        fixture.extra_info[f"{metric} ({unit})"] = round(count * 1e-9 / mean, 4)


class BenchmarkGrouping:
    """Groups and orders the rows of the terminal benchmark report.

    Ordering by `(n, name, mean)` puts each operation's sizes in ascending order
    inside its group, which is the only arrangement in which a throughput column
    reads as a curve rather than as noise.
    """

    @classmethod
    def group(cls, benchmarks, group_by):
        """Return `[(group, [benchmark, ...])]`, or None to defer to the default.

        Only the default `group_by="group"` is handled; an explicit
        `--benchmark-group-by` is the user overriding this, so it passes through.
        """
        if group_by != "group":
            return None

        groups = {}
        for bench in benchmarks:
            key = bench.get("group") or bench["name"].split("[")[0]
            groups.setdefault(key, []).append(bench)

        for rows in groups.values():
            rows.sort(
                key=lambda b: (
                    b.get("extra_info", {}).get("n", 0),
                    b["name"],
                    b["mean"],
                )
            )
            for bench in rows:
                cls.annotate_throughput(bench)
        return sorted(groups.items(), key=lambda pair: pair[0] or "")

    @staticmethod
    def annotate_throughput(bench):
        """Derive elements/second from `n` when the benchmark did not report one."""
        extra = bench.get("extra_info", {})
        count = extra.get("n")
        mean = bench.get("mean", 0)
        if count and mean > 0 and THROUGHPUT_KEY not in extra:
            extra[THROUGHPUT_KEY] = round(count / mean / 1e9, 4)


# ---------------------------------------------------------------------------
# Comparing libraries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """One operation at one size, across every library that measured it.

    Knows its own verdict, so the winner, the spread and the tie rule are
    decided once instead of being recomputed by the tally, the table and each
    cell in turn.
    """

    operation: str
    count: int
    timings: dict
    #: Below this spread the difference is machine noise, not a result.  The
    #: machine this runs on drifts up to ~8% per case run to run.
    tie_ratio: float = 1.05

    @property
    def fastest(self):
        return min(self.timings.values())

    @property
    def winner(self):
        return min(self.timings, key=self.timings.get)

    @property
    def spread(self):
        return max(self.timings.values()) / self.fastest if self.fastest > 0 else 1.0

    @property
    def tied(self):
        return self.spread < self.tie_ratio

    @property
    def group(self):
        """The operation without its parametrization, for sectioning."""
        return self.operation.split("[")[0]


class CompetitionReport:
    """Side-by-side timings for every library that measured the same operation.

    A benchmark opts in by carrying `lib` and `n` in its `extra_info`; an
    operation appears only when at least two libraries measured it, because a
    single-library row compares nothing.
    """

    #: extra_info keys that are bookkeeping rather than a column.
    HIDDEN = frozenset({"lib", "n", THROUGHPUT_KEY})

    #: The leading `[n=NNN]` / `[n=NNN-` of a parametrized pytest id.  The `n`
    #: fixture always comes first, and it is already its own column.
    N_PREFIX = re.compile(r"\[n=\d+(-|\])")

    SEPARATOR = "[dim]│[/]"

    def __init__(self, benchmarks):
        self._timings = {}
        self._meta = {}
        for bench in benchmarks:
            lib, operation, count = self._parse(bench)
            if lib is None:
                continue
            self._timings.setdefault((operation, count), {})[lib] = bench["mean"]
            extra = {
                key: value
                for key, value in bench.get("extra_info", {}).items()
                if key not in self.HIDDEN
            }
            if extra:
                self._meta.setdefault((operation, count), {}).update(extra)

    @classmethod
    def _parse(cls, bench):
        """Return `(lib, operation, n)`, or a triple of Nones if not comparable."""
        extra = bench.get("extra_info", {})
        lib = extra.get("lib")
        count = extra.get("n")
        if not (lib and count is not None):
            return None, None, None
        name = bench["name"]
        prefix = f"test_{lib}_"
        operation = name[len(prefix) :] if name.startswith(prefix) else name
        # "[n=10000]"       -> ""        (fixture only)
        # "[n=10000-inner]" -> "[inner]" (fixture plus a mark suffix)
        operation = cls.N_PREFIX.sub(
            lambda match: "[" if match.group(1) == "-" else "", operation
        )
        return lib, operation, count

    #: `(upper bound in seconds, scale, unit, decimals)`, smallest first.
    UNITS = ((1e-6, 1e9, "ns", 1), (1e-3, 1e6, "\u00b5s", 2), (1.0, 1e3, "ms", 2))

    @classmethod
    def _format(cls, seconds):
        for limit, scale, unit, decimals in cls.UNITS:
            if seconds < limit:
                return f"{seconds * scale:.{decimals}f} {unit}"
        return f"{seconds:.2f} s"

    @staticmethod
    def _distinct(sources):
        """Insertion-ordered union, so column order is stable across runs."""
        return list(dict.fromkeys(chain.from_iterable(sources)))

    @property
    def libs(self):
        return self._distinct(self._timings.values())

    @property
    def meta_keys(self):
        return self._distinct(self._meta.values())

    def rows(self):
        """Only operations at least two libraries measured."""
        return [
            Comparison(operation, count, timings)
            for (operation, count), timings in sorted(self._timings.items())
            if len(timings) >= 2
        ]

    def tally(self):
        """`(wins per lib, tie count)` across every comparable row."""
        wins = dict.fromkeys(self.libs, 0)
        ties = 0
        for row in self.rows():
            if row.tied:
                ties += 1
            else:
                wins[row.winner] += 1
        return wins, ties

    def render(self, width=220):
        """The table as terminal lines, or a one-line explanation of its absence."""
        if not self.libs:
            return ["No benchmarks with lib metadata found."]
        if not self.rows():
            return ["No operations with multiple libs measured."]
        console = Console(highlight=False, width=width)
        with console.capture() as capture:
            console.print(self._table())
        return capture.get().splitlines()

    def _separated(self, cells):
        """Lib cells with a rule between each pair, and one on either side."""
        divided = [self.SEPARATOR]
        for index, cell in enumerate(cells):
            if index:
                divided.append(self.SEPARATOR)
            divided.append(cell)
        return divided + [self.SEPARATOR]

    def _table(self):
        libs = self.libs
        wins, ties = self.tally()

        table = Table(title="Competition", box=box.SIMPLE_HEAD, show_footer=True)
        table.add_column("Operation", no_wrap=True)
        table.add_column("n", justify="right")
        # One rule column per gap, matching what `_separated` puts in each row.
        for index, lib in enumerate(libs):
            table.add_column("", no_wrap=True)  # the rule before this lib
            footer = f"[bold green]{wins[lib]} wins[/]" if wins[lib] else ""
            table.add_column(
                lib.capitalize(), justify="right", footer=footer, no_wrap=True
            )
        table.add_column("", no_wrap=True)  # the rule after the last lib
        table.add_column("Fastest", justify="right", footer=f"[dim]{ties} ties[/]")
        for key in self.meta_keys:
            table.add_column(key.capitalize(), justify="right", no_wrap=True)

        group = None
        for row in self.rows():
            if group is not None and row.group != group:
                table.add_section()
            group = row.group

            cells = [self._cell(row, lib) for lib in libs]
            verdict = (
                "[dim]~tie[/dim]"
                if row.tied
                else f"[bold green]{row.winner} {row.spread:.1f}x[/bold green]"
            )
            extra = self._meta.get((row.operation, row.count), {})
            table.add_row(
                row.operation.replace("[", "\\["),
                f"{row.count:,}",
                *self._separated(cells),
                verdict,
                *[str(extra.get(key, "")) for key in self.meta_keys],
            )
        return table

    def _cell(self, row, lib):
        seconds = row.timings.get(lib)
        if seconds is None:
            return "—"
        if row.tied:
            return self._format(seconds)
        if seconds == row.fastest:
            return f"[bold green]{self._format(seconds)} 1.0x[/bold green]"
        return f"{self._format(seconds)} {seconds / row.fastest:.1f}x"


# ---------------------------------------------------------------------------
# Keeping results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkEnvelope:
    """One run's results: what was measured, and which commit measured it."""

    commit: str
    timestamp: str
    ref: str
    results: list = field(default_factory=list)

    #: Per-result keys the rolling history carries alongside the headline mean.
    DETAIL_KEYS = (
        "file",
        "min_ns",
        "max_ns",
        "median_ns",
        "stddev_ns",
        "rounds",
        "extra_info",
    )

    @classmethod
    def from_benchmarks(cls, vcs, benchmarks):
        return cls(
            commit=vcs.commit,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ref=vcs.ref,
            results=[cls._result(bench) for bench in benchmarks],
        )

    @staticmethod
    def _result(bench):
        stats = bench.stats
        # "python/tests/bench_compute.py::test_marrow_add[...]" -> "bench_compute.py"
        fullname = getattr(bench, "fullname", "") or ""
        source = (
            fullname.split("::")[0].rsplit("/", 1)[-1] if "::" in fullname else None
        )
        extra = {
            key: value
            for key, value in bench.extra_info.items()
            if key != THROUGHPUT_KEY
        }

        result = {
            "name": bench.name,
            "file": source,
            "mean_ns": (stats.mean if stats else 0.0) * 1e9,
            "throughput_gelems_s": bench.extra_info.get(THROUGHPUT_KEY),
        }
        if stats:
            result["min_ns"] = stats.min * 1e9
            result["max_ns"] = stats.max * 1e9
            result["median_ns"] = stats.median * 1e9
            result["stddev_ns"] = stats.stddev * 1e9
            result["rounds"] = stats.rounds
        if extra:
            result["extra_info"] = extra
        return result

    def to_dict(self):
        return {
            "commit": self.commit,
            "timestamp": self.timestamp,
            "ref": self.ref,
            "results": self.results,
        }


class BenchmarkHistory:
    """Per-commit snapshots plus the rolling series the dashboard reads.

    Two artefacts, because they answer different questions: `<commit>.json` is
    one run in full, and `data.json` is every run's headline numbers keyed by
    benchmark name, which is what a chart needs and what would be unreadable if
    it carried everything.
    """

    MAX_RUNS = 200

    #: Where the rolling series lives when the caller names no file.  The
    #: dashboard reads this path, so the default belongs with the writer.
    DEFAULT_HISTORY = Path("benchmarks") / "data.json"

    def __init__(self, results_dir, history_file=None, root=None):
        self.results_dir = Path(results_dir)
        self.history_file = Path(
            history_file or Path(root or ".") / self.DEFAULT_HISTORY
        )

    def save(self, envelope):
        """Write the snapshot, merge into the series, and report both counts."""
        written = self.write_envelope(envelope)
        runs = self.update_history(envelope)
        return written, len(envelope.results), runs

    def write_envelope(self, envelope):
        """Write the snapshot, folding into whatever this commit already has.

        A commit's results arrive in several pieces: one `pytest` selection is
        one compilation unit, so CI benchmarks the tree as a series of calls
        rather than one, and each call saves only what it measured.  Replacing
        the file left the dashboard holding whichever selection happened to run
        last.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        out = self.results_dir / f"{envelope.commit}.json"
        payload = self._merge_snapshot(out, envelope)
        # `latest.json` is a stable URL for the dashboard; the commit file is
        # the archive.  Same bytes, two names.
        for path in (out, self.results_dir / "latest.json"):
            path.write_text(json.dumps(payload, indent=2) + "\n")
        return out

    @staticmethod
    def _merge_snapshot(out, envelope):
        """`envelope` as a dict, carrying forward any results already on disk
        for the same commit."""
        payload = envelope.to_dict()
        if not out.exists():
            return payload
        previous = json.loads(out.read_text())
        if previous.get("commit") != envelope.commit:
            return payload
        by_name = {result["name"]: result for result in previous.get("results", [])}
        for result in payload["results"]:
            by_name[result["name"]] = result
        payload["results"] = list(by_name.values())
        return payload

    def update_history(self, envelope):
        history = self._load()
        # Same reason as `write_envelope`: a commit's benchmarks arrive in
        # several calls, so a run already recorded for it is extended rather
        # than left alone.
        for run in history["runs"]:
            if run["commit"] == envelope.commit:
                entry = self._run_entry(envelope)
                run["timestamp"] = entry["timestamp"]
                run["ref"] = entry["ref"]
                run.setdefault("results", {}).update(entry["results"])
                break
        else:
            history["runs"].append(self._run_entry(envelope))

        history["runs"].sort(key=lambda run: run.get("timestamp", ""), reverse=True)
        history["runs"] = history["runs"][: self.MAX_RUNS]
        # The dashboard's series picker; insertion order keeps it stable.
        history["operations"] = list(
            dict.fromkeys(
                chain.from_iterable(run.get("results", {}) for run in history["runs"])
            )
        )

        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text(json.dumps(history, indent=2) + "\n")
        return len(history["runs"])

    def _load(self):
        if self.history_file.exists():
            return json.loads(self.history_file.read_text())
        return {"runs": [], "operations": []}

    @staticmethod
    def _run_entry(envelope):
        results = {}
        for result in envelope.results:
            entry = {
                "mean_ns": result["mean_ns"],
                "throughput_gelems_s": result["throughput_gelems_s"],
            }
            for key in BenchmarkEnvelope.DETAIL_KEYS:
                if key in result:
                    entry[key] = result[key]
            results[result["name"]] = entry
        return {
            "commit": envelope.commit,
            "short_commit": envelope.commit[:7],
            "timestamp": envelope.timestamp,
            "ref": envelope.ref,
            "results": results,
        }
