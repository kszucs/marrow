# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Format-agnostic primitives shared across marrow.

Each submodule is a self-contained block that depends on nothing in marrow
but its errors, buffers and views -- no arrays, no dtypes, no execution
context -- so it can be read, tested and lifted out on its own:

| module | what |
|---|---|
| `argparse` | `ArgumentParser` — argv into named values, flags and `--help` |
| `decimal` | `float_to_decimal` — the exact unscaled decimal of a double; `parse_decimal`, `format_decimal` — decimal text; `decimal_bound` — the digit limit of a precision |
| `datetime` | `CivilDate`, `Epoch`, `floor_div`, `split_days`, `truncate_div`, `weekday`, `iso_week_thursday`, `weekday_name` — proleptic-Gregorian arithmetic; `CalendarUnit` — the boundaries a date is floored to; `WallClock`, `TimestampFormat` — time zones |
| `byteorder` | `LittleEndian`, `BigEndian`, `Leb128` — values as bytes and as varints |
| `checksum` | `Crc32` / `Crc32c` — the ISO-3309 / zlib / gzip CRC and CRC-32C, over one `Crc` |
| `hex` | `hex_digit` — one hexadecimal digit's value |
| `hashing` | `Hasher` plus `RapidHash64`, `XxHash64`, `AHash64`, `Fmix64`, `TruncatedHash64`, `KeyHash`; `XxHash32`; `SwissHashTable` — 64-bit hash to dense id |
| `dylib` | `Dylib`, `LibSpec` — declaring and opening an optional C library |
| `uri` | `Uri`, `StorageOptions` — a location and the service config it implies |
| `segment_tree` | `SegmentTree`, `Monoid` — range folds over a fixed sequence |
| `hll` | `HyperLogLog` — approximate distinct counts over 64-bit hashes |
| `text` | `Utf8`, `LikePattern`, `RegexPattern` — code points and patterns over UTF-8 |
| `testing` | `TestSuite` / `BenchSuite` / `Benchmark` — the harness pytest drives |

The names are re-exported here, so `from ..utils import Crc32` is the
import everywhere; the submodule split is about where the code *lives*, not
about making callers spell out a path.

**`testing` is the one exception and is deliberately not re-exported.** Every
module in the tree imports `marrow.utils`, and none of them should pull
`std.benchmark` in behind it — test and bench files import
`..utils.testing` explicitly.

This replaced a single 312-line `marrow/utils.mojo` that was four unrelated
things. Device capability
(`GPU_ENABLED`, `has_accelerator_support`) went the other way, to
`marrow.execution`, where `ExecContext` already owns every question about
whether there is a device.
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
    unit_day,
    unit_hour,
    unit_minute,
    unit_month,
    unit_quarter,
    unit_second,
    unit_year,
    weekday,
    weekday_name,
)
from .dylib import Dylib, LibSpec
from .segment_tree import Monoid, SegmentTree
from .hll import HyperLogLog
from .text import LikePattern, RegexPattern, Utf8
from .hex import hex_digit
from .hashing import (
    AHash64,
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
