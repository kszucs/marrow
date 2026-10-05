# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target: the Arrow IPC file reader.

The input is an IPC file: footer first, then every batch it lists. Raising is
the correct answer to a malformed one; an abort or a crash is a bug.
"""

from marrow.io import BufferSource
from marrow.ipc import RecordBatchFileReader

from common import consume


def fuzz_one(data: Span[UInt8, _]) raises:
    var reader = RecordBatchFileReader(BufferSource(data))
    for i in range(reader.num_record_batches()):
        consume(reader.read_batch(i))
