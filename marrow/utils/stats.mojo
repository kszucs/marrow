# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Counters inside the kernels, for a build that asks for them.

`-D MARROW_STATS=true` turns each `count[name](n)` into an atomic add to a
process-wide counter, and `analyze` reports how much each one moved in every
phase: how many rows a hash table looked up and how many new keys it took,
how often it grew, how many rows were partitioned, how many bytes a Parquet
scan fetched and how many row groups its predicates pruned. Without the
define, `count` compiles to nothing and `snapshot` answers an empty list.

The counters are global, so two queries running at once in one process add
to the same ones; a report is exact for a query that runs alone.
"""

from std.atomic import Atomic
from std.ffi import _Global
from std.sys import get_defined_bool


comptime STATS_ENABLED = get_defined_bool["MARROW_STATS", False]()
"""Whether the kernels count; `-D MARROW_STATS=true`."""

comptime STATS: List[StaticString] = [
    "hash.lookups",
    "hash.inserts",
    "hash.new_keys",
    "hash.collisions",
    "hash.resizes",
    "hash.rehashed",
    "partition.rows",
    "join.probe_rows",
    "join.matches",
    "parquet.row_groups",
    "parquet.row_groups_pruned",
    "parquet.bytes",
]
"""Every counter, in the order a report lists them."""

comptime _COUNT = len(STATS)


def _zero() -> Atomic[Int64]:
    return Atomic[Int64](0)


@always_inline
def count[name: StaticString](n: Int = 1):
    """Add `n` to the counter `name`, when the build counts."""
    comptime if STATS_ENABLED:
        comptime assert _known(name), "unknown counter: " + name
        try:
            _ = _Global[name, _zero].get_or_create_ptr()[].fetch_add(Int64(n))
        except:
            pass


def _known(name: StaticString) -> Bool:
    comptime for i in range(_COUNT):
        comptime known = STATS[i]
        if known == name:
            return True
    return False


def snapshot() -> List[Int]:
    """Every counter's value now, in `STATS` order; empty when the build does
    not count."""
    var out = List[Int]()
    comptime if STATS_ENABLED:
        comptime for i in range(_COUNT):
            try:
                out.append(
                    Int(_Global[STATS[i], _zero].get_or_create_ptr()[].load())
                )
            except:
                out.append(0)
    return out^


def since(before: List[Int]) -> List[Int]:
    """How much each counter moved since `before`, a `snapshot`; empty when
    the build does not count."""
    var now = snapshot()
    var moved = List[Int](capacity=len(now))
    for i in range(min(len(now), len(before))):
        moved.append(now[i] - before[i])
    return moved^


def counter(counters: List[Int], name: StaticString) -> Int:
    """Counter `name` of `counters`, as `snapshot` or `since` answer them;
    0 when the build does not count."""
    comptime for i in range(_COUNT):
        comptime known = STATS[i]
        if known == name and i < len(counters):
            return counters[i]
    return 0


def names() -> List[String]:
    """Every counter's name, in `STATS` order."""
    var out = List[String](capacity=_COUNT)
    comptime for i in range(_COUNT):
        comptime name = STATS[i]
        out.append(String(name))
    return out^
