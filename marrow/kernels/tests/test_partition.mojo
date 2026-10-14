# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for `RadixPartitioner` — the radix split behind group-by and joins.

It had no direct coverage: its correctness was only ever implied by whichever
join or group-by benchmark happened to produce right answers. Its two hot loops
(per-thread histogram, parallel scatter) are the kind that break *silently* — a
bad write cursor duplicates or drops rows rather than crashing, and the result
is still a well-formed list of partitions.

So the invariants are asserted directly: the partitions must contain every input
row exactly once, `row_indices` must map each row back to the hash it came from,
and routing must follow the top `num_bits`. Serial and parallel must agree —
which is what makes the threshold at `_MIN_PARALLEL_PARTITION_ROWS` an
optimisation rather than a behaviour switch.
"""

from std.testing import assert_equal, assert_true

from ...arrays import DynArray, StructArray, UInt64Array
from ...builders import Int64Builder, UInt64Builder
from ...dtypes import uint64
from ...execution import ExecContext
from ...kernels.partition import (
    RadixPartitioner,
    Partition,
    partition_bits,
    repartition,
)
from ...tabular import record_batch


def _hashes(n: Int) raises -> UInt64Array:
    """Hashes spread across the top bits, so every partition gets rows.

    Multiplied by a large odd constant rather than shifted, so the low bits
    vary too — a router that used the wrong end would still pass otherwise.
    """
    var b = UInt64Builder(capacity=n)
    for i in range(n):
        b.append(UInt64(i) * 0x9E3779B97F4A7C15)
    return b.finish()


def _assert_covers_every_row(
    parts: List[Partition], src: UInt64Array, n: Int
) raises:
    """Every input row appears in exactly one partition, at its own hash."""
    var seen = List[Int](length=n, fill=0)
    var total = 0
    for p in range(len(parts)):
        ref part = parts[p]
        total += len(part.hashes)
        for i in range(len(part.hashes)):
            var row = Int(part.row_indices[i].value())
            assert_true(row >= 0 and row < n)
            seen[row] += 1
            # The partition's hash at this slot is the hash of that row —
            # a scatter that mixed up the two flat buffers would break here
            # while still producing the right row count.
            assert_equal(
                part.hashes[i].value(),
                src[row].value(),
            )
    assert_equal(total, n)
    for i in range(n):
        assert_equal(seen[i], 1)


def test_partition_serial_covers_every_row_once() raises:
    """The serial path is the reference: no row lost, none duplicated."""
    var n = 5_000
    var src = _hashes(n)
    var parts = RadixPartitioner(
        num_bits=4, ctx=ExecContext.serial()
    ).partition(src.copy())
    _assert_covers_every_row(parts, src, n)


def test_partition_parallel_covers_every_row_once() raises:
    """The parallel path must hold the same invariant. Sized above
    `_MIN_PARALLEL_PARTITION_ROWS` (65_536) so the striped path actually runs —
    below it the partitioner falls back to serial and this would silently be a
    second copy of the serial test."""
    var n = 100_000
    var src = _hashes(n)
    var parts = RadixPartitioner(
        num_bits=4, ctx=ExecContext.parallel(4)
    ).partition(src.copy())
    _assert_covers_every_row(parts, src, n)


def test_partition_routes_by_top_bits() raises:
    """A row lands in the partition named by its top `num_bits`."""
    var n = 5_000
    var src = _hashes(n)
    var bits = 4
    var parts = RadixPartitioner(
        num_bits=bits, ctx=ExecContext.serial()
    ).partition(src.copy())
    for p in range(len(parts)):
        ref part = parts[p]
        for i in range(len(part.hashes)):
            assert_equal(Int(part.hashes[i].value() >> UInt64(64 - bits)), p)


def test_partition_serial_and_parallel_agree() raises:
    """Thread count is an optimisation, not a behaviour switch: the same input
    produces the same partition sizes either way."""
    var n = 100_000
    var src = _hashes(n)
    var serial = RadixPartitioner(
        num_bits=4, ctx=ExecContext.serial()
    ).partition(src.copy())
    var parallel = RadixPartitioner(
        num_bits=4, ctx=ExecContext.parallel(4)
    ).partition(src.copy())
    assert_equal(len(serial), len(parallel))
    for p in range(len(serial)):
        assert_equal(len(serial[p].hashes), len(parallel[p].hashes))


def test_partition_empty_input() raises:
    """No rows still yields the full partition list, every one empty."""
    var src = _hashes(0)
    var parts = RadixPartitioner(
        num_bits=3, ctx=ExecContext.parallel(4)
    ).partition(src.copy())
    assert_equal(len(parts), 8)
    var total = 0
    for p in range(len(parts)):
        total += len(parts[p].hashes)
    assert_equal(total, 0)


def _batch(start: Int, n: Int) raises -> StructArray:
    """Rows `start .. start + n`: key `i % 97` (NULL every 31st row) and
    value `i`."""
    var k = Int64Builder(capacity=n)
    var v = Int64Builder(capacity=n)
    for i in range(start, start + n):
        if i % 31 == 0:
            k.append_null()
        else:
            k.append(Int64(i % 97))
        v.append(Int64(i))
    var cols: List[DynArray] = [k.finish(), v.finish()]
    return record_batch(cols^, names=["k", "v"]).to_struct_array()


def test_partition_bits() raises:
    assert_equal(partition_bits(1), 4)
    assert_equal(partition_bits(4), 4)
    assert_equal(partition_bits(12), 6)
    assert_equal(partition_bits(16), 6)
    assert_equal(partition_bits(17), 7)
    assert_equal(partition_bits(1000), 9)


def test_repartition_places_every_row_once_by_key() raises:
    var batches: List[StructArray] = [
        _batch(0, 1000),
        _batch(1000, 0),
        _batch(1000, 2500),
    ]
    var keys: List[Int] = [0]
    var parts = repartition(batches, keys, 4, ExecContext.parallel(3))
    assert_equal(len(parts), 16)
    # The partition each key (NULL as -1) landed in, and every value seen.
    var home = List[Int](length=98, fill=-1)
    var seen = List[Int](length=3500, fill=0)
    for p in range(16):
        assert_equal(len(parts[p]), 3)
        for b in range(3):
            ref batch = parts[p][b]
            var k = batch.field(0)
            var v = batch.field(1)
            var last = -1
            for i in range(len(batch)):
                var key = Int(k.as_int64()[i].value()) if k.is_valid(i) else 97
                if home[key] == -1:
                    home[key] = p
                assert_equal(home[key], p)
                var value = Int(v.as_int64()[i].value())
                assert_true(value > last, "a batch's rows keep their order")
                last = value
                seen[value] += 1
    for value in range(3500):
        assert_equal(seen[value], 1)


def test_repartition_no_batches() raises:
    var batches = List[StructArray]()
    var keys: List[Int] = [0]
    var parts = repartition(batches, keys, 4, ExecContext.serial())
    assert_equal(len(parts), 16)
    for p in range(16):
        assert_equal(len(parts[p]), 0)
