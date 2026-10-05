# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What every harness shares: reading back what a decoder returned.

A decoder that returns without raising has claimed its output is well formed,
so a harness reads every value of it. Formatting an array walks its offsets
and buffers, which is where a layout that does not match its data shows up --
as a bounds assertion, or as a read past an allocation under AddressSanitizer.
"""

from marrow.tabular import RecordBatch, Table

# Past this many rows a batch is only counted. A few hundred input bytes can
# declare millions of rows of a constant column, and formatting all of them
# turns one input into a timeout rather than a finding.
comptime MAX_ROWS_FORMATTED = 4096


def consume(batch: RecordBatch):
    """Read every value of `batch`."""
    if batch.num_rows() > MAX_ROWS_FORMATTED:
        return
    for i in range(batch.num_columns()):
        _ = String(batch.column(i))


def consume(table: Table):
    """Read every value of `table`."""
    if table.num_rows() > MAX_ROWS_FORMATTED:
        return
    for i in range(table.num_columns()):
        _ = String(table.column(i))
