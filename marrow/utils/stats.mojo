# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Counters inside the kernels, for a build that asks for them.

`-D MARROW_STATS=true` turns each `count[name](n)` into an atomic add to a
process-wide counter, and `analyze` reports how much each one moved in every
phase: how many rows a hash table looked up and how many new keys it took,
how often it grew, how many rows were partitioned, how many bytes a Parquet
scan fetched and how many row groups its predicates pruned. Without the
define, `count` compiles to nothing and `Counters.now` counts nothing.

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


struct Counters(Copyable, Movable, Writable):
    """Every counter's value, in `STATS` order, or how much each moved over a
    stretch of a run; empty when the build does not count."""

    var _values: List[Int]

    def __init__(out self):
        """No counts."""
        self._values = List[Int]()

    @staticmethod
    def now() -> Self:
        """Every counter's value now."""
        var out = Self()
        comptime if STATS_ENABLED:
            comptime for i in range(_COUNT):
                try:
                    out._values.append(
                        Int(
                            _Global[STATS[i], _zero]
                            .get_or_create_ptr()[]
                            .load()
                        )
                    )
                except:
                    out._values.append(0)
        return out^

    @staticmethod
    def since(before: Self) -> Self:
        """How much each counter moved since `before`, taken with `now`."""
        var now = Self.now()
        var moved = Self()
        for i in range(min(len(now._values), len(before._values))):
            moved._values.append(now._values[i] - before._values[i])
        return moved^

    def __getitem__(self, name: StaticString) -> Int:
        """Counter `name`; 0 when the build does not count."""
        comptime for i in range(_COUNT):
            comptime known = STATS[i]
            if known == name and i < len(self._values):
                return self._values[i]
        return 0

    def moved(self) -> Bool:
        """Whether any counter is not zero."""
        for v in self._values:
            if v != 0:
                return True
        return False

    def write_to[W: Writer](self, mut writer: W):
        """The counters that are not zero, as `name=value`, space-separated."""
        var first = True
        comptime for i in range(_COUNT):
            comptime name = STATS[i]
            if i < len(self._values) and self._values[i] != 0:
                writer.write("" if first else " ", name, "=", self._values[i])
                first = False

    def to_json(self) -> String:
        """The counters that are not zero, as a JSON object."""
        var out = String("{")
        comptime for i in range(_COUNT):
            comptime name = STATS[i]
            if i < len(self._values) and self._values[i] != 0:
                out += String("" if out == "{" else ",", '"', name, '":')
                out += String(self._values[i])
        return out + "}"
