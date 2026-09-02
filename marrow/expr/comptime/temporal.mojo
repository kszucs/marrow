"""Temporal operators: extraction and truncation over a temporal operand.

Both nodes here are **breakers** — `bind` runs a whole-column kernel and `lane`
reads the answer back — and both are breakers for the same reason
`StringLength` is: the kernel is not a per-element transform of a SIMD lane.
`_extract` walks days-since-epoch through `CivilDate`, which is a division
chain and a leap-year table per row, and `DateTruncKernel` floors against a
calendar for `month`/`quarter`/`year`. Neither has a `core[T, W]` a fused loop
could inline.

That costs one materialisation and buys the rest of the fusion: the *operand*
still fuses, so `date_trunc(col("ts"), "month")` reads the column directly and
`year(col("d", date32()))` in a projection alongside four other expressions
still runs each of them in its own single pass.

**Why this is a fifth module rather than more of `numeric.mojo`.** The two
nodes here land in *different families* — `TemporalExtract` produces `int32`
and is a `NumericValue`, `DateTrunc` keeps its operand's dtype and is a
`TemporalValue` — so they share an operand bound and nothing else. Splitting
by family instead would put `TemporalExtract` next to `NumericBinary`, whose
operands are numeric, and `DateTrunc` next to nothing at all.
"""

from ...arrays import (
    Date32Array,
    Int32Array,
    Int64Array,
    PrimitiveArray,
    StringArray,
    StructArray,
)
from ...buffers import Bitmap
from ...dtypes import (
    Date32Type,
    DynType,
    Int32Type,
    Int64Type,
    StringType,
    TemporalType,
)
from ...kernels.temporal import (
    CalendarUnit,
    DateTruncKernel,
    DayKernel,
    DayNameKernel,
    EpochKernel,
    IsoYearKernel,
    LastDayKernel,
    MonthNameKernel,
    TemporalExtract64Kernel,
    TemporalNameKernel,
    WeekKernel,
    DayOfWeekKernel,
    DayOfYearKernel,
    HourKernel,
    MinuteKernel,
    MonthKernel,
    QuarterKernel,
    SecondKernel,
    TemporalExtractKernel,
    YearKernel,
)
from ...schema import Schema
from ..logical import Shape
from ..bindings import Bindings
from .core import (
    ColumnBound,
    NumericValue,
    StringValue,
    TemporalValue,
    Unnamed,
)


# ---------------------------------------------------------------------------
# TemporalExtract — temporal -> int32
# ---------------------------------------------------------------------------
struct TemporalExtract[K: TemporalExtractKernel, A: TemporalValue](
    ColumnBound, NumericValue, Unnamed
):
    """One calendar or clock field of a temporal value, as `int32`.

    `int32` for every field, `year` included, because that is what Arrow C++'s
    `year`/`month`/`day` return and therefore what PyArrow answers; narrowing
    `month` to `int8` would make `year - month` a mixed-width expression for no
    saving a column of 4-byte integers does not already have.

    **Validity comes from the bound, not from the operand**, which is what
    `ColumnBound` says. `_extract` copies the operand's bitmap into its result,
    so a null timestamp already has a null year by the time this node sees it,
    and re-deriving the rule here would state it in two places.

    The operand is a `TemporalValue` and so cannot be a duration: `_extract`
    raises for a calendar field over one, naming the reason. That is a runtime
    error rather than a compile-time one because `DurationType` *is* a
    `TemporalType` — the family is right and only the calendar half is
    meaningless for it.
    """

    comptime Type = Int32Type
    comptime shape = Shape.columnar
    """Columnar whatever the operand was: `bind` materialises a length-N
    column, as `StringLength` and `NullPredicate` do."""

    comptime Bound = Int32Array

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        # The typed `apply`, not `dispatch`: `A.Type` is a comptime parameter
        # here, so nothing has to resolve a runtime dtype first. Same trade as
        # `StringLength.bind`.
        #
        # `evaluate` rather than `bind` on the operand, even though a leaf's
        # bound *is* the column: `Self.A.Bound` is opaque at this point —
        # declared `Copyable & Deinitable` — so it cannot be handed to a kernel
        # that wants a `PrimitiveArray`. Only the family's `Type` projects.
        var t = self.a.evaluate(batch, bindings).to_array(len(batch))
        return Self.K.apply(t.as_type[PrimitiveArray[Self.A.Type]]())

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime Year = TemporalExtract[YearKernel, _]
comptime Month = TemporalExtract[MonthKernel, _]
comptime Day = TemporalExtract[DayKernel, _]
comptime Hour = TemporalExtract[HourKernel, _]
comptime Minute = TemporalExtract[MinuteKernel, _]
comptime Second = TemporalExtract[SecondKernel, _]
comptime Quarter = TemporalExtract[QuarterKernel, _]
comptime DayOfWeek = TemporalExtract[DayOfWeekKernel, _]
comptime DayOfYear = TemporalExtract[DayOfYearKernel, _]


# ---------------------------------------------------------------------------
# DateTrunc — temporal -> the same temporal type
# ---------------------------------------------------------------------------
struct DateTrunc[A: TemporalValue](ColumnBound, TemporalValue, Unnamed):
    """Floor a temporal value to a `CalendarUnit` boundary, keeping its type.

    **The unit is a field, not a parameter, and that is deliberate.** Making it
    comptime would monomorphise this node seven ways and buy nothing: the unit
    is not read per row — `DateTruncKernel` resolves it once per batch into a
    tick count or a calendar walk — so a comptime unit would trade binary size
    for no inner-loop win. It is a `CalendarUnit` rather than a `String` so an
    unsupported spelling cannot reach the kernel; `TemporalValue.date_trunc`
    parses at *construction*, which is what makes `date_trunc(ts, "fortnight")`
    fail when the plan is built rather than on the first row that evaluates it.

    Output dtype is the operand's, unit and timezone included, which is why
    `dtype` forwards rather than answering `DynType(Self.Type())` the way an
    always-`int32` node can: a `TemporalType` is not `Defaultable`, exactly as
    `TemporalColumn.dtype` records.
    """

    comptime Type = Self.A.Type
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.Type]

    var a: Self.A
    var _unit: CalendarUnit

    def __init__(out self, var a: Self.A, unit: CalendarUnit):
        self.a = a^
        self._unit = unit

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (
            DateTruncKernel.apply(
                self.a.evaluate(batch, bindings).to_array(len(batch)),
                self._unit,
            )
            .as_primitive[Self.Type]()
            .copy()
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("date_trunc(", self.a, ", ", self._unit, ")")


# ---------------------------------------------------------------------------
# TemporalExtract64 — temporal -> int64
# ---------------------------------------------------------------------------
struct TemporalExtract64[K: TemporalExtract64Kernel, A: TemporalValue](
    ColumnBound, NumericValue, Unnamed
):
    """`week`, `iso_year` and `epoch` — the temporal fields SQL types as
    `BIGINT`.

    A separate node from `TemporalExtract` only because the width differs.
    Those nine are Arrow's `year`/`month`/`day` and answer `int32`, which is
    what PyArrow answers; these are the SQL spellings, and `epoch` needs the
    width for its own sake — seconds since 1970 leave `int32` in 2038.
    """

    comptime Type = Int64Type
    comptime shape = Shape.columnar
    comptime Bound = Int64Array

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var t = self.a.evaluate(batch, bindings).to_array(len(batch))
        return Self.K.apply(t.as_type[PrimitiveArray[Self.A.Type]]())

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime Week = TemporalExtract64[WeekKernel, _]
comptime IsoYear = TemporalExtract64[IsoYearKernel, _]
comptime Epoch = TemporalExtract64[EpochKernel, _]


# ---------------------------------------------------------------------------
# LastDay — temporal -> date32
# ---------------------------------------------------------------------------
struct LastDay[A: TemporalValue](ColumnBound, TemporalValue, Unnamed):
    """SQL `last_day(d)` — the last day of the operand's month, as `date32`.

    The one node here whose output type is *fixed* rather than forwarded:
    `DateTrunc` keeps its operand's dtype because flooring a timestamp yields a
    timestamp, but the last day of a month is a date whatever the operand was.
    `Date32Type` is default-constructible, so `dtype` can answer from the type
    rather than from the value — which is exactly what `DateTrunc` cannot do.
    """

    comptime Type = Date32Type
    comptime shape = Shape.columnar
    comptime Bound = Date32Array

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Date32Type())

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var t = self.a.evaluate(batch, bindings).to_array(len(batch))
        return LastDayKernel.apply(t.as_type[PrimitiveArray[Self.A.Type]]())

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("last_day(", self.a, ")")


# ---------------------------------------------------------------------------
# TemporalName — temporal -> string
# ---------------------------------------------------------------------------
struct TemporalName[K: TemporalNameKernel, A: TemporalValue](
    StringValue, Unnamed
):
    """`dayname` and `monthname` — extraction that answers a *name*.

    The only temporal node that crosses into the string family, so it is the
    only one whose `lane` borrows rather than loading a SIMD vector. `bind`
    runs the kernel once per batch and `lane` borrows out of the result, the
    same shape `StringUnary` uses.
    """

    comptime Type = StringType
    comptime shape = Shape.columnar
    comptime Bound = StringArray

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var t = self.a.evaluate(batch, bindings).to_array(len(batch))
        return Self.K.apply(t.as_type[PrimitiveArray[Self.A.Type]]())

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # `_extract_name` appends a null for a null input, so the bound column
        # already carries the rule.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime DayName = TemporalName[DayNameKernel, _]
comptime MonthName = TemporalName[MonthNameKernel, _]
