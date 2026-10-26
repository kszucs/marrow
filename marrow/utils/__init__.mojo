# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Format-agnostic primitives shared across marrow.

Each submodule depends on nothing in marrow but its errors, buffers and views
-- no arrays, no dtypes, no execution context -- so it can be read and tested
on its own. `testing` is the exception: it compares arrays for the test
harness.

| module | what |
|---|---|
| `argparse` | `ArgumentParser` — argv into named values, flags and `--help` |
| `bits` | `BitReader`, `BitWriter`, `BitCursor` — streams of bit fields |
| `byteorder` | `LittleEndian`, `BigEndian`, `Leb128` — values as bytes and as varints |
| `checksum` | `Crc32`, `Crc32c` — the zlib CRC and CRC-32C |
| `datetime` | `CivilDate`, `Epoch`, `floor_div`, `split_days`, `truncate_div`, `weekday`, `iso_week_thursday`, `weekday_name` — proleptic-Gregorian arithmetic; `CalendarUnit` — the boundaries a date is floored to; `WallClock`, `TimestampFormat`, `Iso8601` — time zones and timestamp text |
| `decimal` | `float_to_decimal`, `parse_decimal`, `format_decimal`, `decimal_bound` — decimals as unscaled integers |
| `dylib` | `LibSet`, `LibSpec`, `Dylib` — declaring and opening an optional C library |
| `enum` | `Enum`, `EnumVariant`, `Members` / `ArcMembers` / `OwnedMembers` — the storage behind every erased handle |
| `hashing` | `Hasher` plus `RapidHash64`, `XxHash64`, `XxHash32`, `Fmix64`, `TruncatedHash64`, `KeyHash`; `SwissHashTable` — 64-bit hash to dense id |
| `hll` | `HyperLogLog` — approximate distinct counts over 64-bit hashes |
| `integers` | `Unsigned`, `zigzag` / `unzigzag`, `ordered` / `unordered` — integer bit tricks |
| `segment_tree` | `SegmentTree`, `Monoid` — range folds over a fixed sequence |
| `stats` | `count`, `Counters` — kernel counters under `-D MARROW_STATS=true` |
| `testing` | `TestSuite`, `BenchSuite`, `Benchmark` — the harness pytest drives |
| `text` | `Utf8`, `LikePattern`, `RegexPattern` — column-wide UTF-8 scans and patterns |
| `threads` | `ThreadPool`, `Thread`, `Mutex`, `Condition` — the CPU thread pool |

The names most callers want are re-exported here, so `from ..utils import
Crc32` is the usual import. `bits`, `enum`, `integers`, `stats` and `threads`
are imported from their module. `testing` is deliberately not re-exported:
every module in the tree imports `marrow.utils`, and none of them should pull
`std.benchmark` in behind it.
"""

from .argparse import ArgSpec, ArgumentParser, ParsedArgs, parse_bool
from .byteorder import BigEndian, Leb128, LittleEndian
from .checksum import Crc32, Crc32c
from .decimal import (
    decimal_bound,
    float_to_decimal,
    format_decimal,
    parse_decimal,
)
from .datetime import (
    CalendarUnit,
    CivilDate,
    Epoch,
    TimestampFormat,
    WallClock,
    floor_div,
    iso_week_thursday,
    split_days,
    truncate_div,
    weekday,
    weekday_name,
)
from .dylib import Dylib, LibSpec
from .segment_tree import Monoid, SegmentTree
from .hll import HyperLogLog
from .text import LikePattern, RegexPattern, Utf8
from .hashing import (
    Hasher,
    Fmix64,
    KeyHash,
    RapidHash64,
    RapidSecret,
    SwissHashTable,
    TruncatedHash64,
    XxHash32,
    XxHash64,
)
