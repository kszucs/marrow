"""Temporal component-extraction and truncation kernels.

Two shapes:

- **Component extraction** (`YearKernel`, `MonthKernel`, `DayKernel`,
  `HourKernel`, `MinuteKernel`, `SecondKernel`, `DayOfWeekKernel`,
  `QuarterKernel`, `DayOfYearKernel`) — pull a single calendar/clock field out of
  a temporal array, producing an `Int32Array` (validity propagated unchanged).
  Concrete kernels only define `component`; the typed `apply` and the
  type-erased `dispatch` are defaulted on the `TemporalExtractKernel` trait.
- **Truncation** (`date_trunc`) — floor a timestamp / date / time array down to a
  unit boundary (`second` / `minute` / `hour` / `day`), returning the same
  temporal type.

Representation (see `marrow.dtypes`): temporal values are integer counts since an
epoch at a `TimeUnit` resolution — `date32` = days since 1970-01-01, `date64` =
milliseconds, `timestamp[unit]` = `unit` ticks since the Unix epoch, `time32` /
`time64` = ticks since midnight. Everything is normalised to *seconds since
epoch* (an `Int` per element), then split into ``days`` since epoch and
``tod`` (seconds-of-day, in ``[0, 86400)``).

The civil-date decomposition (days-since-epoch → year/month/day) uses Howard
Hinnant's ``civil_from_days`` algorithm (the same one Arrow C++ and arrow-rs
use); ``day_of_year`` inverts it with ``days_from_civil``. Both use truncating
integer math kept non-negative by construction, so they are correct regardless
of ``//`` rounding. ``quarter = (month - 1) / 3 + 1`` and ``day_of_week`` returns
ISO weekday with Monday = 0 (matching PyArrow's default).

**Timezone caveat**: timezones are treated as UTC. A `timestamp` with a non-UTC
``tz`` is decomposed in UTC (its wall-clock zone offset is ignored). Localised
extraction is future work.

**Vectorisation note**: the per-element civil-date decomposition branches, so the
extraction loop is scalar per element (TODO: vectorise the pure-arithmetic clock
fields — hour/minute/second — via ``views.apply``).
"""

from ..arrays import (
    AnyArray,
    ArrayData,
    Int32Array,
    PrimitiveArray,
)
from ..buffers import Buffer, Bitmap
from ..dtypes import AnyDataType, DType, TemporalType, TimeUnit
from .core import Kernel


# ---------------------------------------------------------------------------
# Integer helpers
# ---------------------------------------------------------------------------


@always_inline
def _fdiv(a: Int, b: Int) -> Int:
    """Floor division for a positive divisor ``b``, independent of ``//``
    rounding semantics."""
    var q = a // b
    if a - q * b < 0:
        q -= 1
    return q


@always_inline
def _unit_tps(u: TimeUnit) -> Int:
    """Ticks per second for a sub-second time unit (s=1, ms=1e3, us=1e6, ns=1e9).
    """
    if u.value == 0:
        return 1
    elif u.value == 1:
        return 1_000
    elif u.value == 2:
        return 1_000_000
    else:
        return 1_000_000_000


def _ticks_per_second(dt: AnyDataType) raises -> Int:
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
    raise Error(t"temporal: {dt} has no sub-second tick resolution")


# ---------------------------------------------------------------------------
# Civil-date algorithm (Howard Hinnant)
# ---------------------------------------------------------------------------


@always_inline
def _civil_from_days(z: Int) -> Tuple[Int, Int, Int]:
    """Days since 1970-01-01 -> (year, month, day). Hinnant's civil_from_days.
    """
    var zz = z + 719468
    var era = _fdiv(zz, 146097)
    var doe = zz - era * 146097  # [0, 146096]
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)  # [0, 365]
    var mp = (5 * doy + 2) // 153  # [0, 11]
    var d = doy - (153 * mp + 2) // 5 + 1  # [1, 31]
    var m = mp + 3 if mp < 10 else mp - 9  # [1, 12]
    var year = y + 1 if m <= 2 else y
    return (year, m, d)


@always_inline
def _days_from_civil(y: Int, m: Int, d: Int) -> Int:
    """(year, month, day) -> days since 1970-01-01. Inverse of civil_from_days.
    """
    var yy = y - 1 if m <= 2 else y
    var era = _fdiv(yy, 400)
    var yoe = yy - era * 400  # [0, 399]
    var mp = m - 3 if m > 2 else m + 9
    var doy = (153 * mp + 2) // 5 + d - 1  # [0, 365]
    var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


# ---------------------------------------------------------------------------
# Extraction engine + kernel trait
# ---------------------------------------------------------------------------


def _extract[
    T: TemporalType,
    component: def(days: Int, tod: Int) thin -> Scalar[DType.int32],
](array: PrimitiveArray[T], calendar: Bool, name: String) raises -> Int32Array:
    """Normalise each element to (days-since-epoch, seconds-of-day), run
    ``component``, and write an ``Int32Array`` with the input's validity."""
    var dt = array.type()
    var is_date = dt.is_date32() or dt.is_date64()
    var is_ts = dt.is_timestamp()
    var is_time = dt.is_time32() or dt.is_time64()
    if calendar and not (is_date or is_ts):
        raise Error(t"{name}: requires a date or timestamp array, got {dt}")
    if not calendar and not (is_ts or is_time):
        raise Error(t"{name}: requires a timestamp or time array, got {dt}")

    # Normalisation: total seconds = raw * mul / div.
    var mul = 86400 if dt.is_date32() else 1
    var div = 1 if dt.is_date32() else _ticks_per_second(dt)

    var n = len(array)
    var out = Buffer.alloc_uninit[DType.int32](n)
    var dst = out.view[DType.int32](0, n)
    var src = array.values()
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        var total = _fdiv(raw * mul, div)
        var days = _fdiv(total, 86400)
        var tod = total - days * 86400  # [0, 86400)
        dst.unsafe_set(i, component(days, tod))

    # A null input element yields a null output element.
    var vbm: Optional[Bitmap[mut=False]] = None
    if array.bitmap:
        var v = array.bitmap.value().view(array.offset, n)
        vbm = v.union(v).to_immutable()
    return Int32Array(
        length=n,
        nulls=array.nulls,
        offset=0,
        bitmap=vbm^,
        buffer=out.to_immutable(),
    )


trait TemporalExtractKernel(Kernel):
    """A temporal-field extraction kernel. Concrete kernels define ``component``
    (and ``calendar``); ``apply`` (typed, per temporal type) and ``dispatch``
    (type-erased) are defaulted."""

    comptime calendar: Bool
    """Whether the field is a calendar field (date/timestamp) or a clock field
    (timestamp/time)."""

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        """Compute the field from days-since-epoch and seconds-of-day."""
        ...

    @staticmethod
    def apply[T: TemporalType](array: PrimitiveArray[T]) raises -> Int32Array:
        return _extract[T, Self.component](array, Self.calendar, Self.name)

    @staticmethod
    def dispatch(array: AnyArray) raises -> AnyArray:
        var dt = array.dtype()
        if dt.is_date32():
            return Self.apply(array.as_date32()).to_any()
        elif dt.is_date64():
            return Self.apply(array.as_date64()).to_any()
        elif dt.is_timestamp():
            return Self.apply(array.as_timestamp()).to_any()
        elif dt.is_time32():
            return Self.apply(array.as_time32()).to_any()
        elif dt.is_time64():
            return Self.apply(array.as_time64()).to_any()
        else:
            raise Error(t"{Self.name}: expected a temporal array, got {dt}")


# ---------------------------------------------------------------------------
# Concrete extraction kernels
# ---------------------------------------------------------------------------


struct YearKernel(TemporalExtractKernel):
    comptime name = "year"
    comptime calendar = True

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        var c = _civil_from_days(days)
        return Int32(c[0])


struct MonthKernel(TemporalExtractKernel):
    comptime name = "month"
    comptime calendar = True

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        var c = _civil_from_days(days)
        return Int32(c[1])


struct DayKernel(TemporalExtractKernel):
    comptime name = "day"
    comptime calendar = True

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        var c = _civil_from_days(days)
        return Int32(c[2])


struct QuarterKernel(TemporalExtractKernel):
    comptime name = "quarter"
    comptime calendar = True

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        var c = _civil_from_days(days)
        return Int32((c[1] - 1) // 3 + 1)


struct DayOfYearKernel(TemporalExtractKernel):
    comptime name = "day_of_year"
    comptime calendar = True

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        var c = _civil_from_days(days)
        return Int32(days - _days_from_civil(c[0], 1, 1) + 1)


struct DayOfWeekKernel(TemporalExtractKernel):
    comptime name = "day_of_week"
    comptime calendar = True

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        # ISO weekday, Monday = 0 (PyArrow default). 1970-01-01 is a Thursday.
        var dow = days - _fdiv(days, 7) * 7  # [0, 6]
        var w = dow + 3
        if w >= 7:
            w -= 7
        return Int32(w)


struct HourKernel(TemporalExtractKernel):
    comptime name = "hour"
    comptime calendar = False

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(tod // 3600)


struct MinuteKernel(TemporalExtractKernel):
    comptime name = "minute"
    comptime calendar = False

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32((tod // 60) % 60)


struct SecondKernel(TemporalExtractKernel):
    comptime name = "second"
    comptime calendar = False

    @staticmethod
    def component(days: Int, tod: Int) -> Scalar[DType.int32]:
        return Int32(tod % 60)


# ---------------------------------------------------------------------------
# Free-function entry points (PyArrow-style names)
# ---------------------------------------------------------------------------


def year(array: AnyArray) raises -> AnyArray:
    return YearKernel.dispatch(array)


def month(array: AnyArray) raises -> AnyArray:
    return MonthKernel.dispatch(array)


def day(array: AnyArray) raises -> AnyArray:
    return DayKernel.dispatch(array)


def hour(array: AnyArray) raises -> AnyArray:
    return HourKernel.dispatch(array)


def minute(array: AnyArray) raises -> AnyArray:
    return MinuteKernel.dispatch(array)


def second(array: AnyArray) raises -> AnyArray:
    return SecondKernel.dispatch(array)


def day_of_week(array: AnyArray) raises -> AnyArray:
    return DayOfWeekKernel.dispatch(array)


def quarter(array: AnyArray) raises -> AnyArray:
    return QuarterKernel.dispatch(array)


def day_of_year(array: AnyArray) raises -> AnyArray:
    return DayOfYearKernel.dispatch(array)


# ---------------------------------------------------------------------------
# date_trunc — floor a temporal array to a unit boundary
# ---------------------------------------------------------------------------


def _unit_seconds(unit: String) raises -> Int:
    """Seconds per truncation unit."""
    if unit == "second":
        return 1
    elif unit == "minute":
        return 60
    elif unit == "hour":
        return 3600
    elif unit == "day":
        return 86400
    raise Error(
        t"date_trunc: unsupported unit '{unit}' (expected"
        t" second/minute/hour/day)"
    )


def _trunc[
    N: DType
](
    data: ArrayData, dt: AnyDataType, ticks_per_unit: Int, n: Int
) raises -> AnyArray:
    """Floor each ``N``-typed tick count to a multiple of ``ticks_per_unit``,
    keeping the same tick resolution and dtype."""
    var out = Buffer.alloc_uninit[N](n)
    var dst = out.view[N](0, n)
    var src = data.buffers[0].view[N](data.offset, n)
    for i in range(n):
        var raw = Int(src.unsafe_get(i))
        dst.unsafe_set(
            i, Scalar[N](_fdiv(raw, ticks_per_unit) * ticks_per_unit)
        )

    var vbm: Optional[Bitmap[mut=False]] = None
    if data.bitmap:
        var v = data.bitmap.value().view(data.offset, n)
        vbm = v.union(v).to_immutable()
    return AnyArray.from_data(
        ArrayData(
            dtype=dt.copy(),
            length=n,
            nulls=data.nulls,
            offset=0,
            bitmap=vbm^,
            buffers=[out.to_immutable()],
            children=[],
        )
    )


def date_trunc(array: AnyArray, unit: String) raises -> AnyArray:
    """Truncate a temporal array down to ``unit`` (``second`` / ``minute`` /
    ``hour`` / ``day``), returning the same temporal type. Timezones are treated
    as UTC. Matches ``pc.floor_temporal`` for whole-second units."""
    var dt = array.dtype()
    var secs = _unit_seconds(unit)  # validate the unit up front

    if dt.is_date32():
        # date32 has day granularity: any unit <= day leaves the value unchanged.
        return array.copy()
    if not (
        dt.is_date64() or dt.is_timestamp() or dt.is_time32() or dt.is_time64()
    ):
        raise Error(t"date_trunc: unsupported type {dt}")

    var ticks_per_unit = _ticks_per_second(dt) * secs
    var data = array.to_data()
    var n = data.length
    if dt.byte_width() == 4:  # time32
        return _trunc[DType.int32](data, dt, ticks_per_unit, n)
    else:  # date64 / timestamp / time64
        return _trunc[DType.int64](data, dt, ticks_per_unit, n)
