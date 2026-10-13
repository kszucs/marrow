# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Temporal component-extraction and truncation kernels.

Two shapes:

- **Component extraction** (`YearKernel`, `MonthKernel`, `DayKernel`,
  `HourKernel`, `MinuteKernel`, `SecondKernel`, `DayOfWeekKernel`,
  `QuarterKernel`, `DayOfYearKernel`) — pull a single calendar/clock field out of
  a temporal array, producing an `Int32Array` (validity propagated unchanged).
  Concrete kernels only define `component`; the typed `apply` and the
  type-erased `dispatch` are defaulted on the `TemporalExtractKernel` trait.
- **Truncation** (`DateTruncKernel`) — floor a timestamp / date / time array down
  to a `CalendarUnit` boundary (`second` / `minute` / `hour` / `day`), returning
  the same temporal type. The unit is a type, not a `String`, so an unsupported
  one cannot reach the kernel; callers that start from text parse once at their
  own boundary with `CalendarUnit.parse`.

Every entry point is a kernel: `YearKernel.dispatch(a)`, not `year(a)`. There
A free function per kernel would mean ten forwarders here, each with
no caller outside the tests that existed to cover them — the expression layer
already imported the kernels directly.

Representation (see `marrow.dtypes`): temporal values are integer counts since an
epoch at a `TimeUnit` resolution — `date32` = days since 1970-01-01, `date64` =
milliseconds, `timestamp[unit]` = `unit` ticks since the Unix epoch, `time32` /
`time64` = ticks since midnight. Everything is normalised to *seconds since
epoch* (an `Int` per element), then split into ``days`` since epoch and
``tod`` (seconds-of-day, in ``[0, 86400)``).

The civil-date decomposition (days-since-epoch → year/month/day) is
``CivilDate`` in ``marrow/utils/datetime.mojo`` — Howard Hinnant's algorithm,
the same one Arrow C++ and arrow-rs use, so extraction agrees with them by
construction. It lives there rather than here because ``parquet/reader.mojo``
needs the same epoch arithmetic, and ``utils/`` is the only layer both can
depend on. ``quarter = (month - 1) / 3 + 1`` and ``day_of_week`` returns
ISO weekday with Monday = 0 (matching PyArrow's default).

**Time zones.** A `timestamp` with a time zone holds UTC instants, as Arrow
specifies, and every field is read in *its* zone: extraction and truncation
localise each value through a `WallClock` first, and
truncation maps the floored wall-clock time back to an instant. This is Arrow's
reading, not DuckDB's, which reads a `TIMESTAMPTZ` in the session's zone.
`AssumeTimezoneKernel` attaches a zone to a naive timestamp, and
`ConvertTimezoneKernel` relabels one, leaving the instant unchanged.

**Vectorisation note**: the per-element civil-date decomposition branches, so the
extraction loop is scalar per element (TODO: vectorise the pure-arithmetic clock
fields — hour/minute/second — via ``views.apply``).
"""


from ..arrays import (
    DynArray,
    ArrayData,
    Date32Array,
    Int32Array,
    Int64Array,
    PrimitiveArray,
    StringArray,
)
from ..buffers import Buffer, Bitmap
from ..builders import StringBuilder
from ..dtypes import (
    Date32Type,
    Date64Type,
    DynType,
    DType,
    TemporalType,
    Time32Type,
    Time64Type,
    TimestampType,
    TimeUnit,
)
from ..errors import InternalError, InvalidError, TypeError
from .core import Kernel, one_of

from ..utils import CivilDate, WallClock, floor_div


# ---------------------------------------------------------------------------
# Integer helpers
# ---------------------------------------------------------------------------


@always_inline
def _unit_tps(u: TimeUnit) -> Int:
    """Ticks per second for a sub-second time unit (s=1, ms=1e3, us=1e6, ns=1e9).
    """
    return 10 ** u.fraction_digits()


def ticks_per_second(dt: DynType) raises -> Int:
    """Ticks per second for a temporal dtype whose one tick is <= 1 second
    (date64, timestamp, time32, time64, duration)."""
    if dt.is_date64():
        return 1_000  # milliseconds
    elif dt.is_timestamp():
        return _unit_tps(dt.as_timestamp().unit)
    elif dt.is_time32():
        return _unit_tps(dt.as_time32().unit)
    elif dt.is_time64():
        return _unit_tps(dt.as_time64().unit)
    elif dt.is_duration():
        return _unit_tps(dt.as_duration().unit)
    raise TypeError(t"temporal: {dt} has no sub-second tick resolution")


# ---------------------------------------------------------------------------
# Time zones over arrays
# ---------------------------------------------------------------------------


def _is_zoned(dt: DynType) -> Bool:
    """Whether `dt` is a timestamp whose wall clock is not UTC."""
    return dt.is_timestamp() and not WallClock.is_utc(
        dt.as_timestamp().timezone
    )


def _clock(dt: DynType) raises -> Optional[WallClock]:
    """The clock `dt`'s fields are read on: None for anything but a zoned
    timestamp."""
    if not dt.is_timestamp():
        return None
    ref ts = dt.as_timestamp()
    return WallClock.of(ts.timezone, _unit_tps(ts.unit))


def _shift[
    to_utc: Bool
](clock: WallClock, array: DynArray, var to: DynType) raises -> DynArray:
    """Every tick of a timestamp array moved onto `clock`, or off it with
    `to_utc`, typed `to`."""
    var data = array.to_data()
    var n = data.length
    var out = Buffer.alloc_uninit[DType.int64](n)
    var dst = out.view[DType.int64](0, n)
    var src = data.buffers[0].view[DType.int64](data.offset, n)
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        comptime if to_utc:
            raw = clock.to_utc(raw)
        else:
            raw = clock.to_local(raw)
        dst.unsafe_set(i, Int64(raw))
    return DynArray.from_data(
        ArrayData(
            like=data,
            dtype=to^,
            buffers=[out.to_immutable()],
        )
    )


@no_inline
def localise(array: DynArray) raises -> DynArray:
    """A zoned timestamp's wall-clock times, as a naive timestamp of the same
    unit — what every field extractor reads. Any other array comes back as
    itself, an O(1) copy.

    The extractors' `dispatch` and the expression nodes call this on the erased
    array; the typed `apply[T]` overloads read the values as they are, UTC for
    a zoned timestamp."""
    var dt = array.dtype()
    var clock = _clock(dt)
    if not clock:
        return array.copy()
    return _shift[False](
        clock.value(), array, DynType(TimestampType(dt.as_timestamp().unit))
    )


comptime DATES = one_of[Date32Type, Date64Type, TimestampType]
"""The dtypes with a date part, which the calendar fields read."""

comptime TIMES = one_of[Time32Type, Time64Type, TimestampType]
"""The dtypes with a time of day, which the clock fields read."""


# ---------------------------------------------------------------------------
# Extraction engine + kernel trait
# ---------------------------------------------------------------------------


def _extract[
    T: TemporalType,
    component: def(days: Int, tod: Int) thin -> Scalar[DType.int32],
](array: PrimitiveArray[T]) raises -> Int32Array:
    """Normalise each element to (days-since-epoch, seconds-of-day), run
    ``component``, and write an ``Int32Array`` with the input's validity."""
    var dt = array.type()
    debug_assert(
        not _is_zoned(dt),
        (
            "_extract: a zoned timestamp reaches the typed loop; call"
            " localise first"
        ),
    )

    # Normalisation: total seconds = raw * mul / div.
    var mul = 86400 if dt.is_date32() else 1
    var div = 1 if dt.is_date32() else ticks_per_second(dt)

    var n = len(array)
    var out = Buffer.alloc_uninit[DType.int32](n)
    var dst = out.view[DType.int32](0, n)
    var src = array.values()
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        var total = floor_div(raw * mul, div)
        var days = floor_div(total, 86400)
        var tod = total - days * 86400  # [0, 86400)
        dst.unsafe_set(i, component(days, tod))

    # A null input element yields a null output element.
    return Int32Array(
        like=array,
        buffer=out.to_immutable(),
    )


trait TemporalExtractKernel(Kernel):
    """A temporal-field extraction kernel. Concrete kernels define ``name``,
    ``accepts`` and ``component``; ``apply`` (typed, per temporal type) and
    ``dispatch`` (type-erased) are defaulted."""

    comptime accepts[T: AnyType]: Bool
    """Whether the field exists for dtype `T`: `DATES` or `TIMES`."""

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        """Compute the field from days-since-epoch and seconds-of-day."""
        ...

    @staticmethod
    def apply[T: TemporalType](array: PrimitiveArray[T]) raises -> Int32Array:
        comptime if Self.accepts[T]:
            return _extract[T, Self.component](array)
        else:
            raise Self.error[TypeError](t"unsupported dtype {array.type()}")

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        """The typed `apply` for `array`'s dtype, which must be one `accepts`
        admits; a zoned timestamp is read on its wall clock."""

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(localise(array).as_primitive[T]()).to_dyn()

        return array.dtype().dispatch_temporal(leaf)


# ---------------------------------------------------------------------------
# Concrete extraction kernels
# ---------------------------------------------------------------------------


struct YearKernel(TemporalExtractKernel):
    comptime name = "year"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(CivilDate.from_days(days).year)


struct MonthKernel(TemporalExtractKernel):
    comptime name = "month"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(CivilDate.from_days(days).month)


struct DayKernel(TemporalExtractKernel):
    comptime name = "day"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(CivilDate.from_days(days).day)


struct QuarterKernel(TemporalExtractKernel):
    comptime name = "quarter"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(CivilDate.from_days(days).quarter())


struct DayOfYearKernel(TemporalExtractKernel):
    comptime name = "day_of_year"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(CivilDate.from_days(days).day_of_year())


struct DayOfWeekKernel(TemporalExtractKernel):
    comptime name = "day_of_week"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        # ISO weekday, Monday = 0 (PyArrow default). 1970-01-01 is a Thursday.
        var dow = days - floor_div(days, 7) * 7  # [0, 6]
        var w = dow + 3
        if w >= 7:
            w -= 7
        return Int32(w)


struct HourKernel(TemporalExtractKernel):
    comptime name = "hour"
    comptime accepts = TIMES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(tod // 3600)


struct MinuteKernel(TemporalExtractKernel):
    comptime name = "minute"
    comptime accepts = TIMES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32((tod // 60) % 60)


struct SecondKernel(TemporalExtractKernel):
    comptime name = "second"
    comptime accepts = TIMES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(tod % 60)


# ---------------------------------------------------------------------------
# date_trunc — floor a temporal array to a unit boundary
# ---------------------------------------------------------------------------


struct CalendarUnit(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The boundary `date_trunc` floors to — `second`, `minute`, `hour`, `day`.

    A *calendar* unit, not a storage resolution: distinct from `dtypes.TimeUnit`
    (`s`/`ms`/`us`/`ns`), which says how finely a timestamp is stored. Named
    after Arrow C++'s `CalendarUnit`, which plays the same role in
    `RoundTemporalOptions`.

    This is a type rather than a `String` so an unsupported unit cannot reach
    the kernel at all. String-driven callers (Python, SQL) convert once at their
    boundary with `parse`, which is the only place the spelling is known — the
    expression frontends do it at *construction*, so a bad unit fails when the
    plan is built rather than on the row that first evaluates it.
    """

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    @staticmethod
    def parse(unit: String) raises -> Self:
        """The unit named by `unit`; raises if it names none."""
        if unit == "second":
            return Self(0)
        elif unit == "minute":
            return Self(1)
        elif unit == "hour":
            return Self(2)
        elif unit == "day":
            return Self(3)
        elif unit == "month":
            return Self(4)
        elif unit == "quarter":
            return Self(5)
        elif unit == "year":
            return Self(6)
        raise InvalidError(
            t"date_trunc: unsupported unit '{unit}' (expected "
            t"second/minute/hour/day/month/quarter/year)"
        )

    def is_calendar(self) -> Bool:
        """Whether this unit has no fixed length in seconds.

        A month is 28-31 days and a year 365 or 366, so `seconds()` cannot
        answer for these and flooring cannot be a division on the tick count --
        they go through the civil calendar instead. Everything up to `day` is
        fixed-length.
        """
        return self.value >= 4

    def seconds(self) -> Int:
        """How many seconds one of these spans."""
        if self.value == 0:
            return 1
        elif self.value == 1:
            return 60
        elif self.value == 2:
            return 3600
        else:
            return 86400

    def to_string(self) -> StaticString:
        """The unit's spelling. `StaticString`, not `String`: these are four
        literals, and returning an owned `String` would drag its construction
        into every binary that only ever renders a plan."""
        if self.value == 0:
            return "second"
        elif self.value == 1:
            return "minute"
        elif self.value == 2:
            return "hour"
        elif self.value == 3:
            return "day"
        elif self.value == 4:
            return "month"
        elif self.value == 5:
            return "quarter"
        else:
            return "year"

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.to_string())


comptime unit_second = CalendarUnit(0)
comptime unit_minute = CalendarUnit(1)
comptime unit_hour = CalendarUnit(2)
comptime unit_day = CalendarUnit(3)
comptime unit_month = CalendarUnit(4)
comptime unit_quarter = CalendarUnit(5)
comptime unit_year = CalendarUnit(6)


def _trunc[
    N: DType
](
    data: ArrayData,
    dt: DynType,
    ticks_per_unit: Int,
    n: Int,
    clock: Optional[WallClock],
) raises -> DynArray:
    """Floor each ``N``-typed tick count to a multiple of ``ticks_per_unit``,
    keeping the same tick resolution and dtype. With a `clock` the floor is
    taken on wall-clock time, so an hour boundary in Asia/Kolkata falls on
    the half hour in UTC."""
    var out = Buffer.alloc_uninit[N](n)
    var dst = out.view[N](0, n)
    var src = data.buffers[0].view[N](data.offset, n)
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        if clock:
            raw = clock.value().to_local(raw)
        raw = floor_div(raw, ticks_per_unit) * ticks_per_unit
        if clock:
            raw = clock.value().to_utc(raw)
        dst.unsafe_set(i, Scalar[N](raw))

    return DynArray.from_data(
        ArrayData(
            like=data,
            dtype=dt.copy(),
            buffers=[out.to_immutable()],
        )
    )


def _floor_civil(days: Int, unit: CalendarUnit) -> Int:
    """Floor a day count to the start of its month, quarter or year.

    Goes through the civil calendar because these units have no fixed length:
    decompose to (y, m, d), zero the finer fields, recompose. `CivilDate`
    (`utils/datetime.mojo`) is Hinnant's algorithm, already used by the
    extraction kernels, so this adds no new date arithmetic.
    """
    var c = CivilDate.from_days(days)
    if unit == unit_year:
        return c.start_of_year().to_days()
    if unit == unit_quarter:
        return c.start_of_quarter().to_days()
    return c.start_of_month().to_days()


def _trunc_calendar[
    N: DType
](
    data: ArrayData,
    dt: DynType,
    ticks_per_day: Int,
    unit: CalendarUnit,
    n: Int,
    clock: Optional[WallClock],
) raises -> DynArray:
    """Floor each tick count to a month/quarter/year boundary — in wall-clock
    time when there is a `clock`, as `_trunc` does."""
    var out = Buffer.alloc_uninit[N](n)
    var dst = out.view[N](0, n)
    var src = data.buffers[0].view[N](data.offset, n)
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        if clock:
            raw = clock.value().to_local(raw)
        # Floor-divide, so pre-epoch instants land on the day containing them
        # rather than the day after -- the same reason `_trunc` uses `floor_div`.
        var days = floor_div(raw, ticks_per_day)
        var floored = _floor_civil(days, unit) * ticks_per_day
        if clock:
            floored = clock.value().to_utc(floored)
        dst.unsafe_set(i, Scalar[N](floored))

    return DynArray.from_data(
        ArrayData(
            like=data,
            dtype=dt.copy(),
            buffers=[out.to_immutable()],
        )
    )


struct DateTruncKernel(Kernel):
    """Floor a temporal array to a `CalendarUnit` boundary, keeping its type."""

    comptime name = "date_trunc"

    @staticmethod
    def apply(array: DynArray, unit: CalendarUnit) raises -> DynArray:
        var dt = array.dtype()
        if dt.is_date32() and not unit.is_calendar():
            # date32 is already day-granular, so any *fixed-length* unit up to a
            # day is a no-op. Month, quarter and year are not -- returning the
            # input for those silently ignored the request, which is what this
            # guard used to do for every unit.
            return array.copy()
        if not dt.is_temporal():
            raise Self.error[TypeError](t"unsupported type {dt}")

        var data = array.to_data()
        var n = data.length
        var width = dt.byte_width()
        var clock = _clock(dt)
        var tps = 1 if dt.is_date32() else ticks_per_second(dt)

        if unit.is_calendar():
            # date32 counts days directly; everything else counts sub-day ticks.
            var ticks_per_day = 1 if dt.is_date32() else tps * 86400
            if width == 4:
                return _trunc_calendar[DType.int32](
                    data, dt, ticks_per_day, unit, n, clock
                )
            elif width == 8:
                return _trunc_calendar[DType.int64](
                    data, dt, ticks_per_day, unit, n, clock
                )
            else:
                raise Self.error[InternalError](
                    t"{dt} is {width} bytes wide; expected 4 or 8",
                )

        var ticks_per_unit = ticks_per_second(dt) * unit.seconds()
        if width == 4:  # time32
            return _trunc[DType.int32](data, dt, ticks_per_unit, n, clock)
        elif width == 8:  # date64 / timestamp / time64
            return _trunc[DType.int64](data, dt, ticks_per_unit, n, clock)
        else:
            # Every temporal type Arrow defines is int32- or int64-backed, so
            # this is unreachable today. It is spelled out rather than folded
            # into the int64 branch because a wider temporal type would
            # otherwise be read through the wrong lane width, silently.
            raise Self.error[InternalError](
                t"{dt} is {width} bytes wide; expected 4 or 8",
            )


# ---------------------------------------------------------------------------
# int64 extraction — the SQL-shaped temporal fields
# ---------------------------------------------------------------------------
#
# `week`, `iso_year` and `epoch` answer `int64` where the nine kernels above
# answer `int32`. Same reason `CharLengthKernel` does: those nine are Arrow's
# `year`/`month`/`day` and return what PyArrow returns, while these are the SQL
# spellings and SQL's `week`/`isoyear`/`epoch` are `BIGINT`. `epoch` needs the
# width for its own sake — seconds since 1970 leave `int32` in 2038.
#
# `_extract64` is `_extract` with a wider destination. The two are not merged
# behind an `OutType: PrimitiveType` parameter because the component's return
# type would have to be written `Scalar[OutType.native]`, and a `comptime
# native: DType` requirement does not resolve reliably off an externally-bound
# generic parameter (CLAUDE.md, "Associated types, traits, reflection").


def _extract64[
    T: TemporalType,
    component: def(days: Int, tod: Int) thin -> Scalar[DType.int64],
](array: PrimitiveArray[T]) raises -> Int64Array:
    """`_extract`, writing `int64`. See that function for the normalisation."""
    var dt = array.type()
    debug_assert(
        not _is_zoned(dt),
        (
            "_extract64: a zoned timestamp reaches the typed loop; call"
            " localise first"
        ),
    )

    var mul = 86400 if dt.is_date32() else 1
    var div = 1 if dt.is_date32() else ticks_per_second(dt)

    var n = len(array)
    var out = Buffer.alloc_uninit[DType.int64](n)
    var dst = out.view[DType.int64](0, n)
    var src = array.values()
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        var total = floor_div(raw * mul, div)
        var days = floor_div(total, 86400)
        var tod = total - days * 86400  # [0, 86400)
        dst.unsafe_set(i, component(days, tod))

    return Int64Array(
        like=array,
        buffer=out.to_immutable(),
    )


trait TemporalExtract64Kernel(Kernel):
    """A temporal-field extraction kernel answering `int64`. The `int32`
    family's shape exactly — concrete kernels define `accepts` and
    `component`; `apply` and `dispatch` are defaulted."""

    comptime accepts[T: AnyType]: Bool
    """Whether the field exists for dtype `T`: `DATES` or `TIMES`."""

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int64]:
        ...

    @staticmethod
    def apply[T: TemporalType](array: PrimitiveArray[T]) raises -> Int64Array:
        comptime if Self.accepts[T]:
            return _extract64[T, Self.component](array)
        else:
            raise Self.error[TypeError](t"unsupported dtype {array.type()}")

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        """The typed `apply` for `array`'s dtype, which must be one `accepts`
        admits; a zoned timestamp is read on its wall clock."""

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(localise(array).as_primitive[T]()).to_dyn()

        return array.dtype().dispatch_temporal(leaf)


# -- ISO-8601 week numbering ------------------------------------------------
#
# The rule, and the one place the calendar stops agreeing with itself: ISO week
# 1 is the week containing the first Thursday of January, so a date in early
# January can belong to the previous ISO year and a date in late December to
# the next one. `2021-01-01` is ISO week **53 of 2020**.
#
# The implementation takes the Thursday of the date's own ISO week and reads
# its calendar year: by construction that Thursday lies in the ISO year, which
# removes the "is it week 1 or week 53" case analysis entirely. Verified
# against Arrow C++'s own disagreement table (`scalar_temporal_test.cc`):
# 1899-01-01 -> 1898-W52, 2019-12-30 -> 2020-W1, 2010-01-01 -> 2009-W53.


@always_inline
def _iso_weekday(days: Int) -> Int:
    """ISO weekday of a days-since-epoch value: Monday = 1 ... Sunday = 7."""
    # 1970-01-01 was a Thursday, so `days % 7 == 0` is a Thursday.
    var dow = days - floor_div(days, 7) * 7  # [0, 6], 0 == Thursday
    return dow + 4 if dow <= 3 else dow - 3


@always_inline
def _iso_thursday(days: Int) -> Int:
    """Days-since-epoch of the Thursday in the same ISO week as `days`."""
    return days - (_iso_weekday(days) - 4)


struct IsoYearKernel(TemporalExtract64Kernel):
    """SQL `isoyear` — the calendar year of this ISO week's Thursday, which is
    not always the calendar year of the date itself."""

    comptime name = "iso_year"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int64]:
        return Int64(CivilDate.from_days(_iso_thursday(days)).year)


struct WeekKernel(TemporalExtract64Kernel):
    """SQL `week` — the ISO-8601 week number, 1 to 53."""

    comptime name = "week"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int64]:
        var thu = _iso_thursday(days)
        # January 1st of the ISO year is always inside ISO week 1, and the
        # Thursday of week 1 is within three days of it, so counting whole
        # weeks from January 1st to this week's Thursday gives the week index.
        var jan1 = CivilDate.days_from(CivilDate.from_days(thu).year, 1, 1)
        return Int64(floor_div(thu - jan1, 7) + 1)


struct EpochKernel(TemporalExtract64Kernel):
    """SQL `epoch` — whole seconds since 1970-01-01, **floored**.

    Floor, not truncation toward zero, so a pre-1970 instant answers the second
    it falls *in* rather than the one after it. Arrow is internally inconsistent
    here — its temporal extraction kernels floor while its timestamp cast
    divides toward zero — and flooring is the half that agrees with DuckDB.
    """

    comptime name = "epoch"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int64]:
        # `days` and `tod` are already the floored split, so re-joining them
        # cannot round the wrong way.
        return Int64(days * 86400 + tod)


# ---------------------------------------------------------------------------
# last_day — the last day of the month, as date32
# ---------------------------------------------------------------------------


@always_inline
def _last_day_component(days: Int, tod: Int) -> Scalar[DType.int32]:
    """Days-since-epoch of the last day of this date's month.

    The first of the *next* month minus one day, which gets February right from
    the calendar algorithm rather than from a table of month lengths: 2020-02
    answers the 29th and 2100-02 the 28th without either being written down.
    """
    var c = CivilDate.from_days(days)
    var y = c.year + 1 if c.month == 12 else c.year
    var m = 1 if c.month == 12 else c.month + 1
    return Int32(CivilDate.days_from(y, m, 1) - 1)


struct LastDayKernel(Kernel):
    """SQL `last_day(d)` -> `date32`.

    A plain `Kernel` rather than a `TemporalExtractKernel` conformer, because
    the answer is a *date* and the trait's `apply` returns `Int32Array`. A
    conformer cannot change a trait default's return type — the two become
    competing overloads and every call reports `ambiguous call` — so this
    reuses the shared `_extract` engine by calling it directly and re-wraps the
    result, which is O(1): both arrays are int32-backed, so only the dtype
    differs.
    """

    comptime name = "last_day"

    comptime accepts = DATES

    @staticmethod
    def apply[T: TemporalType](array: PrimitiveArray[T]) raises -> Date32Array:
        comptime if Self.accepts[T]:
            var d = _extract[T, _last_day_component](array)
            return Date32Array(
                Date32Type(),
                length=len(d),
                nulls=d.nulls,
                offset=d.offset,
                bitmap=d.bitmap,
                buffer=d.buffer,
            )
        else:
            raise Self.error[TypeError](t"unsupported dtype {array.type()}")

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        """The typed `apply` for `array`'s dtype, which must be one `accepts`
        admits; a zoned timestamp is read on its wall clock."""

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(localise(array).as_primitive[T]()).to_dyn()

        return array.dtype().dispatch_temporal(leaf)


# ---------------------------------------------------------------------------
# day_name / month_name — temporal -> string
# ---------------------------------------------------------------------------
#
# The English tables are written out here rather than taken from a locale.
# Arrow has no `day_name`/`month_name` kernel at all; its only route to a name
# is `strftime`'s `%A`/`%B`, whose answer depends on a `std::locale` and so is
# not a specification. DuckDB's `dayname`/`monthname` are the full, capitalized
# English names, and that is what these return.


def _day_name_of(days: Int) -> String:
    """Full English weekday name of a days-since-epoch value."""
    var dow = days - floor_div(days, 7) * 7  # [0, 6], 0 == Thursday
    var mon0 = dow + 3  # 0 == Monday, matching `DayOfWeekKernel`
    if mon0 >= 7:
        mon0 -= 7
    if mon0 == 0:
        return "Monday"
    elif mon0 == 1:
        return "Tuesday"
    elif mon0 == 2:
        return "Wednesday"
    elif mon0 == 3:
        return "Thursday"
    elif mon0 == 4:
        return "Friday"
    elif mon0 == 5:
        return "Saturday"
    else:
        return "Sunday"


def _month_name_of(days: Int) -> String:
    """Full English month name of a days-since-epoch value."""
    var m = CivilDate.from_days(days).month
    if m == 1:
        return "January"
    elif m == 2:
        return "February"
    elif m == 3:
        return "March"
    elif m == 4:
        return "April"
    elif m == 5:
        return "May"
    elif m == 6:
        return "June"
    elif m == 7:
        return "July"
    elif m == 8:
        return "August"
    elif m == 9:
        return "September"
    elif m == 10:
        return "October"
    elif m == 11:
        return "November"
    else:
        return "December"


def _extract_name[
    T: TemporalType,
    component: def(days: Int) thin -> String,
](array: PrimitiveArray[T]) raises -> StringArray:
    """`_extract`'s shape with a variable-width destination, so a
    `StringBuilder` replaces the preallocated buffer and a null input appends a
    null rather than writing a placeholder."""
    var dt = array.type()
    debug_assert(
        not _is_zoned(dt),
        (
            "_extract_name: a zoned timestamp reaches the typed loop; call"
            " localise first"
        ),
    )

    var mul = 86400 if dt.is_date32() else 1
    var div = 1 if dt.is_date32() else ticks_per_second(dt)

    var n = len(array)
    var builder = StringBuilder(capacity=n)
    var src = array.values()
    for i in range(n):
        if array.is_valid(i):
            var total = floor_div(Int(src.unsafe_get(i)) * mul, div)
            builder.append(component(floor_div(total, 86400)))
        else:
            builder.append_null()
    return builder.finish()


trait TemporalNameKernel(Kernel):
    """A temporal-field kernel answering a *name* rather than a number."""

    comptime accepts[T: AnyType]: Bool
    """Whether the field exists for dtype `T`: `DATES` or `TIMES`."""

    @staticmethod
    def component(days: Int) -> String:
        ...

    @staticmethod
    def apply[T: TemporalType](array: PrimitiveArray[T]) raises -> StringArray:
        comptime if Self.accepts[T]:
            return _extract_name[T, Self.component](array)
        else:
            raise Self.error[TypeError](t"unsupported dtype {array.type()}")

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        """The typed `apply` for `array`'s dtype, which must be one `accepts`
        admits; a zoned timestamp is read on its wall clock."""

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(localise(array).as_primitive[T]()).to_dyn()

        return array.dtype().dispatch_temporal(leaf)


struct DayNameKernel(TemporalNameKernel):
    """SQL `dayname` — `Monday` … `Sunday`."""

    comptime name = "day_name"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int) -> String:
        return _day_name_of(days)


struct MonthNameKernel(TemporalNameKernel):
    """SQL `monthname` — `January` … `December`."""

    comptime name = "month_name"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int) -> String:
        return _month_name_of(days)


# ---------------------------------------------------------------------------
# Time zones — attaching, converting and rendering
# ---------------------------------------------------------------------------


struct AssumeTimezoneKernel(Kernel):
    """Read a naive timestamp as wall-clock time in `zone` and answer the
    instants it names, typed `timestamp[unit, tz=zone]` — pyarrow's
    `assume_timezone`, DuckDB's `timezone(zone, timestamp)`.

    A wall-clock time the zone skips or repeats resolves as `WallClock`
    describes, where pyarrow raises by default. A timestamp that already has
    a zone is refused rather than reinterpreted: that is
    `ConvertTimezoneKernel`."""

    comptime name = "assume_timezone"

    @staticmethod
    def apply(array: DynArray, zone: String) raises -> DynArray:
        var dt = array.dtype()
        if not dt.is_timestamp():
            raise Self.error[TypeError](t"expected a timestamp array, got {dt}")
        ref ts = dt.as_timestamp()
        if ts.timezone:
            raise Self.error[TypeError](
                t"{dt} already has a time zone; convert_timezone relabels it"
            )
        if WallClock.is_utc(zone):
            # A naive timestamp is already read as UTC: nothing moves.
            return ConvertTimezoneKernel.apply(array, zone)
        var clock = WallClock(WallClock.load_zone(zone), _unit_tps(ts.unit))
        return _shift[True](clock, array, DynType(TimestampType(ts.unit, zone)))


struct ConvertTimezoneKernel(Kernel):
    """Give a timestamp the zone `zone`, keeping every instant — only the
    wall-clock reading changes, which is Arrow's cast between two zoned
    timestamps. A naive timestamp is read as UTC, as Arrow's cast reads it.
    The buffers are shared; the zone is loaded only to validate it."""

    comptime name = "convert_timezone"

    @staticmethod
    def apply(array: DynArray, zone: String) raises -> DynArray:
        var dt = array.dtype()
        if not dt.is_timestamp():
            raise Self.error[TypeError](t"expected a timestamp array, got {dt}")
        _ = WallClock.load_zone(zone)
        var data = array.to_data()
        data.dtype = DynType(TimestampType(dt.as_timestamp().unit, zone))
        return DynArray.from_data(data^)
