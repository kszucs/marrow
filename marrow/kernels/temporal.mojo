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

from ..utils import (
    CalendarUnit,
    CivilDate,
    WallClock,
    floor_div,
    iso_week_thursday,
    weekday,
    weekday_name,
)


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
    T: TemporalType, to_utc: Bool
](
    clock: WallClock, array: PrimitiveArray[T], to: DynType
) raises -> PrimitiveArray[T]:
    """Every tick of a timestamp array moved onto `clock`, or off it with
    `to_utc`, typed `to`."""
    var n = len(array)
    var out = Buffer.alloc_uninit[T.native](n)
    var dst = out.view[T.native](0, n)
    var src = array.values()
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        comptime if to_utc:
            raw = clock.to_utc(raw)
        else:
            raw = clock.to_local(raw)
        dst.unsafe_set(i, Scalar[T.native](raw))
    return PrimitiveArray[T](
        like=array, dtype=to.get[T]().copy(), buffer=out^.to_immutable()
    )


struct LocalTimestampKernel(Kernel):
    """A zoned timestamp's wall-clock times, as a naive timestamp of the same
    unit — what every field extractor reads.
    Any other temporal array comes back as itself, an O(1) copy.

    The extractors' `dispatch` and the expression nodes call this before the
    typed field loop, which reads the values as they are, UTC for a zoned
    timestamp."""

    comptime name = "local_timestamp"

    @staticmethod
    def apply[
        T: TemporalType
    ](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
        comptime if one_of[TimestampType][T]:
            var dt = array.type()
            var clock = _clock(dt)
            if clock:
                return _shift[T, False](
                    clock.value(),
                    array,
                    DynType(TimestampType(dt.as_timestamp().unit)),
                )
        return array.copy()

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        """The typed `apply` for `array`'s temporal dtype."""

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_primitive[T]()).to_dyn()

        return array.dtype().try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
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
            " LocalTimestampKernel first"
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
            return Self.apply(
                LocalTimestampKernel.apply(array.as_primitive[T]())
            ).to_dyn()

        return array.dtype().try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
        )


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
        return Int32(weekday(days))


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


struct DateTruncKernel(Kernel):
    """Floor a temporal array to a `CalendarUnit` boundary, keeping its type."""

    comptime name = "date_trunc"

    @staticmethod
    def apply[
        T: TemporalType
    ](array: PrimitiveArray[T], unit: CalendarUnit) raises -> PrimitiveArray[T]:
        var dt = array.type()
        if dt.is_date32() and not unit.is_calendar():
            # date32 is already day-granular, so any *fixed-length* unit up to a
            # day is a no-op. Month, quarter and year are not -- returning the
            # input for those silently ignored the request, which is what this
            # guard used to do for every unit.
            return array.copy()

        # date32 counts days directly; everything else counts sub-day ticks.
        var ticks_per_day = (
            1 if dt.is_date32() else ticks_per_second(dt) * 86400
        )
        var clock = _clock(dt)
        var n = len(array)
        var out = Buffer.alloc_uninit[T.native](n)
        var dst = out.view[T.native](0, n)
        var src = array.values()
        for i in range(n):
            # With a clock the floor is taken on wall-clock time, so an hour
            # boundary in Asia/Kolkata falls on the half hour in UTC.
            var raw = Int(src.unsafe_get(i))
            if clock:
                raw = clock.value().to_local(raw)
            raw = unit.floor(raw, ticks_per_day)
            if clock:
                raw = clock.value().to_utc(raw)
            dst.unsafe_set(i, Scalar[T.native](raw))
        return PrimitiveArray[T](
            like=array, dtype=array.dtype.copy(), buffer=out^.to_immutable()
        )

    @staticmethod
    def dispatch(array: DynArray, unit: CalendarUnit) raises -> DynArray:
        """The typed `apply` for `array`'s temporal dtype."""
        var dt = array.dtype()
        if not dt.is_temporal():
            raise Self.error[TypeError](t"unsupported type {dt}")

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_primitive[T](), unit).to_dyn()

        return dt.try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
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
            " LocalTimestampKernel first"
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
            return Self.apply(
                LocalTimestampKernel.apply(array.as_primitive[T]())
            ).to_dyn()

        return array.dtype().try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
        )


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


struct IsoYearKernel(TemporalExtract64Kernel):
    """SQL `isoyear` — the calendar year of this ISO week's Thursday, which is
    not always the calendar year of the date itself."""

    comptime name = "iso_year"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int64]:
        return Int64(CivilDate.from_days(iso_week_thursday(days)).year)


struct WeekKernel(TemporalExtract64Kernel):
    """SQL `week` — the ISO-8601 week number, 1 to 53."""

    comptime name = "week"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int64]:
        var thu = iso_week_thursday(days)
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
    """Days-since-epoch of the last day of this date's month."""
    return Int32(CivilDate.from_days(days).end_of_month().to_days())


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
            return Self.apply(
                LocalTimestampKernel.apply(array.as_primitive[T]())
            ).to_dyn()

        return array.dtype().try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
        )


# ---------------------------------------------------------------------------
# day_name / month_name — temporal -> string
# ---------------------------------------------------------------------------
#
# The names are `weekday_name` and `CivilDate.month_name`: full English names,
# never a locale's.


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
            " LocalTimestampKernel first"
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
            return Self.apply(
                LocalTimestampKernel.apply(array.as_primitive[T]())
            ).to_dyn()

        return array.dtype().try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
        )


struct DayNameKernel(TemporalNameKernel):
    """SQL `dayname` — `Monday` … `Sunday`."""

    comptime name = "day_name"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int) -> String:
        return String(weekday_name(weekday(days)))


struct MonthNameKernel(TemporalNameKernel):
    """SQL `monthname` — `January` … `December`."""

    comptime name = "month_name"
    comptime accepts = DATES

    @staticmethod
    def component(days: Int) -> String:
        return String(CivilDate.from_days(days).month_name())


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
    def apply[
        T: TemporalType
    ](array: PrimitiveArray[T], zone: String) raises -> PrimitiveArray[T]:
        var dt = array.type()
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
        return _shift[T, True](
            clock, array, DynType(TimestampType(ts.unit, zone))
        )

    @staticmethod
    def dispatch(array: DynArray, zone: String) raises -> DynArray:
        """The typed `apply` for `array`'s temporal dtype."""
        var dt = array.dtype()
        if not dt.is_temporal():
            raise Self.error[TypeError](t"expected a timestamp array, got {dt}")

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_primitive[T](), zone).to_dyn()

        return dt.try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
        )


struct ConvertTimezoneKernel(Kernel):
    """Give a timestamp the zone `zone`, keeping every instant — only the
    wall-clock reading changes, which is Arrow's cast between two zoned
    timestamps. A naive timestamp is read as UTC, as Arrow's cast reads it.
    The buffers are shared; the zone is loaded only to validate it."""

    comptime name = "convert_timezone"

    @staticmethod
    def apply[
        T: TemporalType
    ](array: PrimitiveArray[T], zone: String) raises -> PrimitiveArray[T]:
        var dt = array.type()
        if not dt.is_timestamp():
            raise Self.error[TypeError](t"expected a timestamp array, got {dt}")
        _ = WallClock.load_zone(zone)
        var data = array.to_data()
        data.dtype = DynType(TimestampType(dt.as_timestamp().unit, zone))
        return PrimitiveArray[T](data)

    @staticmethod
    def dispatch(array: DynArray, zone: String) raises -> DynArray:
        """The typed `apply` for `array`'s temporal dtype."""
        var dt = array.dtype()
        if not dt.is_temporal():
            raise Self.error[TypeError](t"expected a timestamp array, got {dt}")

        def leaf[T: TemporalType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_primitive[T](), zone).to_dyn()

        return dt.try_visit[DynArray, TemporalType](
            leaf, "dtype is not temporal"
        )
