"""Benchmark injection, the competition table and the rolling history."""

import dataclasses
import json
from dataclasses import dataclass

from devkit.benches import (
    THROUGHPUT_KEY,
    BenchmarkEnvelope,
    BenchmarkGrouping,
    BenchmarkHistory,
    BenchmarkInjector,
    CompetitionReport,
)


class FakeStats:
    def __init__(self, mean):
        self.mean = mean
        self.min = mean * 0.9
        self.max = mean * 1.1
        self.median = mean
        self.stddev = mean * 0.02
        self.rounds = 10


class FakeBenchmark:
    """Stands in for pytest-benchmark's `Metadata`, lookup order included.

    `Metadata.__getitem__` resolves against `stats` first and falls back to the
    object, which is why `b["mean"]` and `b["name"]` both work on one record.
    """

    def __init__(
        self, name, mean, *, throughput=None, fullname=None, extra_info=None, group=None
    ):
        self.name = name
        self.fullname = fullname or f"tests/bench_test.py::{name}"
        self.stats = FakeStats(mean)
        self.group = group
        self.extra_info = dict(extra_info or {})
        if throughput is not None:
            self.extra_info[THROUGHPUT_KEY] = throughput

    def __getitem__(self, key):
        try:
            return getattr(self.stats, key)
        except AttributeError:
            return getattr(self, key)

    def get(self, key, default=None):
        try:
            return getattr(self.stats, key)
        except AttributeError:
            return getattr(self, key, default)


@dataclass
class FakeVcs:
    commit: str = "abc123def456"
    ref: str = "main"


def envelope(commit="abc123def456", timestamp="2026-04-16T00:00:00Z"):
    made = BenchmarkEnvelope.from_benchmarks(
        FakeVcs(),
        [
            FakeBenchmark("bench_add_10k", 0.001, throughput=1.234),
            FakeBenchmark("bench_add_100k", 0.01),
        ],
    )
    return dataclasses.replace(made, commit=commit, timestamp=timestamp)


def history(tmp_path):
    return BenchmarkHistory(tmp_path / "results", tmp_path / "benchmarks" / "data.json")


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_envelope_carries_every_statistic():
    made = BenchmarkEnvelope.from_benchmarks(
        FakeVcs(),
        [
            FakeBenchmark(
                "bench_add_10k",
                0.001,
                throughput=1.234,
                fullname="python/tests/bench_compute.py::bench_add_10k",
                extra_info={"lib": "marrow", "n": 10000},
            ),
            FakeBenchmark("bench_add_100k", 0.01),
        ],
    )

    first = made.results[0]
    assert first["name"] == "bench_add_10k"
    assert first["mean_ns"] == 0.001 * 1e9
    assert first["throughput_gelems_s"] == 1.234
    assert first["file"] == "bench_compute.py"
    assert first["min_ns"] == 0.001 * 0.9 * 1e9
    assert first["max_ns"] == 0.001 * 1.1 * 1e9
    assert first["median_ns"] == 0.001 * 1e9
    assert first["stddev_ns"] == 0.001 * 0.02 * 1e9
    assert first["rounds"] == 10
    # The throughput key rides its own field, not extra_info.
    assert first["extra_info"] == {"lib": "marrow", "n": 10000}

    second = made.results[1]
    assert second["throughput_gelems_s"] is None
    assert second["file"] == "bench_test.py"
    assert "extra_info" not in second  # omitted rather than written empty

    assert made.commit == "abc123def456" and made.ref == "main"
    assert made.timestamp.endswith("Z")


def test_write_envelope_writes_the_commit_file_and_latest(tmp_path):
    """`latest.json` is a stable URL for the dashboard; the commit file is the archive."""
    keeper = history(tmp_path)
    made = envelope()
    written = keeper.write_envelope(made)
    assert written.name == "abc123def456.json"
    for path in (written, keeper.results_dir / "latest.json"):
        assert json.loads(path.read_text()) == made.to_dict()


# ---------------------------------------------------------------------------
# The rolling history
# ---------------------------------------------------------------------------


def test_history_first_run(tmp_path):
    keeper = history(tmp_path)
    assert keeper.update_history(envelope()) == 1

    stored = json.loads(keeper.history_file.read_text())
    run = stored["runs"][0]
    assert run["commit"] == "abc123def456"
    assert run["short_commit"] == "abc123d"
    result = run["results"]["bench_add_10k"]
    assert result["mean_ns"] == 0.001 * 1e9
    assert result["throughput_gelems_s"] == 1.234
    assert result["file"] == "bench_test.py"
    assert result["stddev_ns"] == 0.001 * 0.02 * 1e9
    assert result["rounds"] == 10
    assert set(stored["operations"]) == {"bench_add_10k", "bench_add_100k"}


def test_history_is_idempotent_per_commit(tmp_path):
    """Re-running the same commit must not double its row."""
    keeper = history(tmp_path)
    keeper.update_history(envelope())
    assert keeper.update_history(envelope()) == 1


def selection(name, commit="abc123def456"):
    """One CI selection's worth of results for `commit`."""
    made = BenchmarkEnvelope.from_benchmarks(FakeVcs(), [FakeBenchmark(name, 0.001)])
    return dataclasses.replace(made, commit=commit, timestamp="2026-04-16T00:00:00Z")


def test_a_commit_measured_in_several_selections_accumulates(tmp_path):
    """CI benchmarks the tree as a series of `pytest` calls, one per compilation
    unit, each saving only what it measured -- all under the same commit."""
    keeper = history(tmp_path)
    written = keeper.write_envelope(selection("bench_kernels"))
    keeper.update_history(selection("bench_kernels"))
    keeper.write_envelope(selection("bench_parquet"))
    runs = keeper.update_history(selection("bench_parquet"))

    assert runs == 1  # one commit, not two
    snapshot = json.loads(written.read_text())
    assert {r["name"] for r in snapshot["results"]} == {
        "bench_kernels",
        "bench_parquet",
    }
    stored = json.loads(keeper.history_file.read_text())
    assert set(stored["runs"][0]["results"]) == {"bench_kernels", "bench_parquet"}
    assert set(stored["operations"]) == {"bench_kernels", "bench_parquet"}


def test_history_appends_new_commits(tmp_path):
    keeper = history(tmp_path)
    keeper.update_history(envelope(commit="aaa"))
    keeper.update_history(envelope(commit="bbb", timestamp="2026-04-16T00:00:01Z"))
    stored = json.loads(keeper.history_file.read_text())
    assert [run["commit"] for run in stored["runs"]] == ["bbb", "aaa"]  # newest first


def test_history_is_capped(tmp_path):
    keeper = history(tmp_path)
    keeper.MAX_RUNS = 2
    for index, commit in enumerate("abc"):
        keeper.update_history(
            envelope(commit=commit, timestamp=f"2026-04-16T00:00:0{index}Z")
        )
    stored = json.loads(keeper.history_file.read_text())
    assert [run["commit"] for run in stored["runs"]] == ["c", "b"]


def test_save_reports_what_it_wrote(tmp_path):
    keeper = history(tmp_path)
    written, count, runs = keeper.save(envelope())
    assert written.exists() and count == 2 and runs == 1


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def test_durations_convert_from_every_reported_unit():
    assert BenchmarkInjector.to_seconds(1, "ns") == 1e-9
    assert BenchmarkInjector.to_seconds(1, "us") == 1e-6
    assert BenchmarkInjector.to_seconds(1, "ms") == 1e-3
    assert BenchmarkInjector.to_seconds(1, "s") == 1.0
    # An unknown unit is treated as seconds rather than silently scaled.
    assert BenchmarkInjector.to_seconds(2, "??") == 2.0


class FakeBenchmarkSession:
    def __init__(self, disabled=False):
        self.disabled = disabled
        self.benchmarks = []


def test_injection_records_one_round_per_measured_run():
    """Individual runs are what give a Mojo benchmark real min/max/stddev."""
    session = FakeBenchmarkSession()
    fixture = BenchmarkInjector(session).inject(
        "bench_and", "f.mojo::bench_and", {"runs": [100, 200, 300], "unit": "ns"}
    )
    assert fixture.stats.stats.rounds == 3
    assert fixture.stats.stats.min == 100e-9
    assert fixture.stats.stats.max == 300e-9
    assert len(session.benchmarks) == 1


def test_injection_accepts_a_single_mean():
    session = FakeBenchmarkSession()
    fixture = BenchmarkInjector(session).inject(
        "bench_one", "f.mojo::bench_one", {"value": 5, "unit": "us"}
    )
    assert fixture.stats.stats.rounds == 1
    assert fixture.stats.stats.mean == 5e-6


def test_injection_attaches_throughput_when_the_runner_counted():
    session = FakeBenchmarkSession()
    fixture = BenchmarkInjector(session).inject(
        "bench_filter",
        "f.mojo::bench_filter",
        {
            "runs": [1_000_000],
            "unit": "ns",
            "throughput_count": 1_000_000,
            "throughput_metric": "elements",
            "throughput_unit": "GElems/s",
        },
    )
    # 1e6 elements in 1e6 ns is exactly one element per ns: 1 GElem/s.
    assert fixture.extra_info["elements (GElems/s)"] == 1.0


def test_injection_is_a_no_op_when_benchmarking_is_disabled():
    session = FakeBenchmarkSession(disabled=True)
    assert (
        BenchmarkInjector(session).inject("b", "f::b", {"runs": [1], "unit": "ns"})
        is None
    )
    assert session.benchmarks == []


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_grouping_orders_by_size_then_name():
    rows = [
        FakeBenchmark("bench_add", 0.3, group="add", extra_info={"n": 1000}),
        FakeBenchmark("bench_add", 0.1, group="add", extra_info={"n": 10}),
        FakeBenchmark("bench_add", 0.2, group="add", extra_info={"n": 100}),
    ]
    [(name, ordered)] = BenchmarkGrouping.group(rows, "group")
    assert name == "add"
    assert [b.extra_info["n"] for b in ordered] == [10, 100, 1000]


def test_grouping_derives_a_throughput_from_n():
    rows = [FakeBenchmark("bench_add", 1e-3, group="add", extra_info={"n": 1_000_000})]
    BenchmarkGrouping.group(rows, "group")
    # 1e6 elements in 1 ms.
    assert rows[0].extra_info[THROUGHPUT_KEY] == 1.0


def test_grouping_does_not_overwrite_a_measured_throughput():
    rows = [
        FakeBenchmark(
            "bench_add", 1e-3, group="add", extra_info={"n": 1_000_000}, throughput=7.0
        )
    ]
    BenchmarkGrouping.group(rows, "group")
    assert rows[0].extra_info[THROUGHPUT_KEY] == 7.0


def test_grouping_defers_to_an_explicit_group_by():
    """`--benchmark-group-by` is the user overriding this; it must pass through."""
    assert BenchmarkGrouping.group([], "param:n") is None


def test_grouping_falls_back_to_the_name_stem():
    rows = [FakeBenchmark("test_marrow_add[n=10]", 0.1, extra_info={"n": 10})]
    [(name, _)] = BenchmarkGrouping.group(rows, "group")
    assert name == "test_marrow_add"


# ---------------------------------------------------------------------------
# The competition table
# ---------------------------------------------------------------------------


def competitors(*specs):
    return [
        FakeBenchmark(
            f"test_{lib}_{operation}[n={count}]",
            mean,
            extra_info={"lib": lib, "n": count},
        )
        for lib, operation, count, mean in specs
    ]


def test_competition_strips_the_lib_prefix_and_the_n_fixture():
    lib, operation, count = CompetitionReport._parse(
        FakeBenchmark(
            "test_marrow_filter[n=10000]", 1.0, extra_info={"lib": "marrow", "n": 10000}
        )
    )
    assert (lib, operation, count) == ("marrow", "filter", 10000)


def test_competition_keeps_a_mark_suffix():
    _, operation, _ = CompetitionReport._parse(
        FakeBenchmark(
            "test_marrow_filter[n=10000-inner]",
            1.0,
            extra_info={"lib": "marrow", "n": 10000},
        )
    )
    assert operation == "filter[inner]"


def test_competition_ignores_a_benchmark_without_lib_metadata():
    assert CompetitionReport._parse(FakeBenchmark("bench_x", 1.0)) == (None, None, None)


def test_competition_needs_two_libs_to_compare():
    report = CompetitionReport(competitors(("marrow", "filter", 10, 1.0)))
    assert report.rows() == []
    assert report.render() == ["No operations with multiple libs measured."]


def test_competition_with_no_metadata_at_all_says_so():
    assert CompetitionReport([FakeBenchmark("bench_x", 1.0)]).render() == [
        "No benchmarks with lib metadata found."
    ]


def test_competition_counts_a_win():
    report = CompetitionReport(
        competitors(
            ("marrow", "filter", 10, 1.0),
            ("polars", "filter", 10, 4.0),
        )
    )
    wins, ties = report.tally()
    assert wins == {"marrow": 1, "polars": 0}
    assert ties == 0


def test_competition_calls_a_close_result_a_tie():
    """This machine drifts up to ~8% per case, so a 1% spread is not a result."""
    report = CompetitionReport(
        competitors(
            ("marrow", "filter", 10, 1.00),
            ("polars", "filter", 10, 1.01),
        )
    )
    wins, ties = report.tally()
    assert ties == 1
    assert sum(wins.values()) == 0


def test_competition_renders_a_table():
    lines = CompetitionReport(
        competitors(
            ("marrow", "filter", 10_000, 1e-6),
            ("polars", "filter", 10_000, 4e-6),
            ("marrow", "sort", 10_000, 2e-6),
            ("polars", "sort", 10_000, 1e-6),
        )
    ).render()
    body = "\n".join(lines)
    assert "Competition" in body
    assert "Marrow" in body and "Polars" in body
    assert "filter" in body and "sort" in body
    assert "1 wins" in body  # one each
    assert "10,000" in body


def test_competition_formats_each_magnitude():
    assert CompetitionReport._format(5e-9) == "5.0 ns"
    assert CompetitionReport._format(5e-6) == "5.00 µs"
    assert CompetitionReport._format(5e-3) == "5.00 ms"
    assert CompetitionReport._format(5.0) == "5.00 s"
