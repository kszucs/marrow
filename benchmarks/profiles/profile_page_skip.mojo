# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Driver for the cost of a `RowSelection` inside `ParquetFile.read`.

    pixi run profile benchmarks/profiles/profile_page_skip.mojo --sample --no-open

Runs the same read as `bench_read_selected_*`, without the harness, so the
selection shape can be swept without recompiling. One difference on purpose:
the selections are built once outside the loop, where the benchmark builds
them inside its timed body -- see the comment at the build site, and the
backlog item that turns on exactly that distinction.

Two standing facts change how the output reads. A local `ByteSource` is
`Buffer.mmap_file`, so a trimmed fetch skips no I/O and only the decode saving
is visible here -- the byte saving is asserted instead, by the recorder tests in
`marrow/parquet/tests/test_page_io.mojo`. And the default corpus is
uncompressed, so an all-present page decodes to a memcpy: point
`MARROW_PROFILE_PATH` at a snappy file to see skipping win. What this driver
found, and what is still open, is in `backlog.md`.

Without `MARROW_PROFILE_PATH` it first writes the file
`bench_read_selected_prefix_1m` reads -- the same corpus through the same
pyarrow options -- into a scratch directory removed on exit. The recording then
opens with the pyarrow import and that write, a few seconds before the first
read; point `MARROW_PROFILE_PATH` at a kept file to profile the reads alone.

Overrides: `MARROW_PROFILE_ITERS` (default 20), `MARROW_PROFILE_FIRST` (rows
kept at the front of each group; default an eighth, negative for all),
`MARROW_PROFILE_KEEP_EVERY` (default 1), `MARROW_PROFILE_SELECTION` (0 reads
with no selection at all -- the control), `MARROW_PROFILE_PATH`.
"""

from std.benchmark import keep
from std.os.env import getenv
from std.os.path import join
from std.python import Python

from marrow.parquet import ParquetFile, RowSelection
from marrow.utils.testing import ScratchDir


def _parse_int(name: String, default: Int) -> Int:
    var s = getenv(name, "")
    if s.byte_length() == 0:
        return default
    try:
        return Int(s)
    except:
        return default


def _write_default(path: String) raises:
    """`bench_parquet`'s `_prepare_groups` corpus at 4 x 250,000 rows."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var n = 1_000_000
    var table = pa.table(
        Python.dict(
            a=pa.array(np.arange(n, dtype="int64")),
            b=pa.array(np.arange(n, dtype="float64")),
            c=pa.array(np.arange(n, dtype="int32")),
        )
    )
    pq.write_table(
        table,
        path,
        compression="none",
        row_group_size=250_000,
        data_page_size=64 * 1024,
        write_page_index=True,
    )


def _profile(path: String) raises:
    var iters = _parse_int("MARROW_PROFILE_ITERS", 20)
    var keep_every = _parse_int("MARROW_PROFILE_KEEP_EVERY", 1)
    var use_selection = _parse_int("MARROW_PROFILE_SELECTION", 1) != 0

    var pf = ParquetFile(path)
    ref md = pf.metadata()
    var num_groups = len(md.row_groups)
    var rows_per_group = md.row_groups[0].num_rows
    var first = _parse_int("MARROW_PROFILE_FIRST", rows_per_group // 8)
    if first < 0:
        first = rows_per_group

    var pattern = List[Bool](capacity=rows_per_group)
    for i in range(rows_per_group):
        pattern.append(i < first and i % keep_every == 0)

    # Built once, outside the timed loop, because that is what a caller does:
    # `ParquetScanOperator` copies a selection it already has, which is a
    # refcount bump. Building one per iteration would time a megabyte of
    # allocation that no read performs.
    var built = List[RowSelection](capacity=num_groups)
    for _ in range(num_groups):
        built.append(RowSelection(pattern.copy()))

    var total = 0
    for _ in range(iters):
        if use_selection:
            var sels = List[RowSelection](capacity=num_groups)
            for g in range(num_groups):
                sels.append(built[g].copy())
            total += pf.read(row_selections=sels^).num_rows()
        else:
            total += pf.read().num_rows()
    keep(total)
    keep(built)


def main() raises:
    var path = getenv("MARROW_PROFILE_PATH", "")
    if path.byte_length() > 0:
        _profile(path)
    else:
        with ScratchDir() as dir:
            var fixture = join(dir, "marrow_bench_selected_prefix.parquet")
            _write_default(fixture)
            _profile(fixture)
