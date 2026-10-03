# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""How many threads marrow computes and fetches on: PyArrow's ``cpu_count``
and ``io_thread_count``, and their setters.

CPU work runs on one process-wide pool and blocking I/O on a second. Each
starts on first use with ``MARROW_NUM_THREADS`` or ``MARROW_IO_THREADS``
threads when that is set to a positive integer. Otherwise the CPU pool takes
one thread per performance core the process may use, and the I/O pool one per
logical core, at least two. A count includes the calling thread, which always
takes part in the work it starts.
"""

import operator

from . import libmarrow as _ma

__all__ = ["cpu_count", "io_thread_count", "set_cpu_count", "set_io_thread_count"]


def cpu_count():
    """The number of threads parallel operations use: what ``num_threads=0``
    resolves to."""
    return _ma.cpu_count()


def set_cpu_count(count):
    """Use ``count`` threads for parallel operations from now on. Work already
    running finishes on the threads it started on. Raises ``ArrowInvalid`` (a
    ``ValueError``) for a count below 1."""
    _ma.set_cpu_count(operator.index(count))


def io_thread_count():
    """The number of threads blocking I/O, such as an object-store fetch,
    runs on."""
    return _ma.io_thread_count()


def set_io_thread_count(count):
    """Run blocking I/O on ``count`` threads from now on; see
    :func:`set_cpu_count`."""
    _ma.set_io_thread_count(operator.index(count))
