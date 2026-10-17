# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`HyperLogLog`: estimates per sketch, and sketches independent of each other."""

from std.testing import assert_equal, assert_raises, assert_true

from ..hashing import Fmix64
from ..hll import HyperLogLog


def test_hll_small_cardinality_counts_duplicates_once() raises:
    # Linear counting, while few registers are set, is within one.
    var sketch = HyperLogLog(11, 1)
    for i in range(100):
        sketch.add(0, Fmix64.mix[1](UInt64(i % 50)))
    assert_true(abs(sketch.estimate(0) - 50) <= 1)


def test_hll_sketches_are_independent() raises:
    var sketches = HyperLogLog(11, 2)
    for i in range(30):
        sketches.add(1, Fmix64.mix[1](UInt64(i)))
    assert_equal(sketches.estimate(0), 0)
    assert_equal(sketches.estimate(1), 30)


def test_hll_grow_keeps_registers() raises:
    var sketches = HyperLogLog(11)
    sketches.grow(1)
    for i in range(20):
        sketches.add(0, Fmix64.mix[1](UInt64(i)))
    sketches.grow(3)
    assert_equal(len(sketches), 3)
    assert_equal(sketches.estimate(0), 20)
    assert_equal(sketches.estimate(2), 0)


def test_hll_large_cardinality_within_error() raises:
    var sketch = HyperLogLog(14, 1)
    comptime n = 100_000
    for i in range(n):
        sketch.add(0, Fmix64.mix[1](UInt64(i)))
    var error = abs(Float64(sketch.estimate(0)) - Float64(n)) / Float64(n)
    assert_true(error < 0.02)


def test_hll_refuses_a_sketch_of_another_precision() raises:
    var narrow = HyperLogLog(11, 1)
    narrow.add(0, UInt64(12345))
    var wide = HyperLogLog(14, 1)
    with assert_raises(contains="precision"):
        wide.merge(0, narrow.sketch(0))
