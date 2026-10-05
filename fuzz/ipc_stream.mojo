# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target: the Arrow IPC stream reader.

The input is an IPC stream. Raising is the correct answer to a malformed one;
an abort or a crash is a bug.
"""

from marrow.io import BufferSource
from marrow.ipc import RecordBatchStreamReader

from common import consume


def fuzz_one(data: Span[UInt8, _]) raises:
    var reader = RecordBatchStreamReader(BufferSource(data))
    for batch in reader.read_all():
        consume(batch)
