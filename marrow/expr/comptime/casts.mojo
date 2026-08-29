"""Fused casts: nodes that change a subtree's output type.

Two shapes live here, and the split is the same one `ColumnBound` names.

**Three fuse.** `NumericCast`, `NumToBool` and `BoolToNum` are one instruction
on a SIMD lane, so they carry their operand's `Bound` unchanged and their
`lane` wraps the operand's. `col("a", int32).cast(int64) + other` stays a
single pass with no intermediate column.

**Two break.** `StringToNum` and `NumToString` cross the fixed-width boundary,
which no lane can do: parsing `"12"` reads a variable number of bytes, and
formatting `12` writes one. Both run their kernel once over the whole batch in
`bind` and read the answer back in `lane` — the `CaseWhen` shape, and the
reason they conform to `ColumnBound`.

That conformance is worth being explicit about, because it removes a real
defect rather than merely tidying one. The previous expression layer's
`StringToNum` had to answer validity from the *batch* as well as from the
state, and its batch-side answer re-ran the whole parse to recover a bitmap it
had already computed. Here validity has one source — the bound — so the parse
runs once. A parse failure is still a null the input does
not have (`"x"` -> null), and it is still `ColumnBound`'s
`bound.to_data().owned_validity()` that reports it.

The two breakers also call the kernels' **typed** `apply` rather than
`dispatch`. The previous layer erased to `DynArray` and narrowed back with
`as_type` because its operand's type was reachable only at runtime; here `A.Type` is a
comptime parameter, so the dispatch and the round trip both disappear.
"""

from ...arrays import BinaryLikeArray, PrimitiveArray, StructArray
from ...buffers import Bitmap
from ...dtypes import BoolType, DynType, NumericType, StringLikeType
from ...kernels.cast import (
    BoolToNum as BoolToNumKernel,
    NumToBool as NumToBoolKernel,
    NumToString as NumToStringKernel,
    NumericCast as NumericCastKernel,
    StringToNum as StringToNumKernel,
)
from ...schema import Schema
from ..logical import Shape
from ..bindings import Bindings
from .core import (
    BoolValue,
    ColumnBound,
    NumericValue,
    StringValue,
    Unnamed,
)


# ---------------------------------------------------------------------------
# The three that fuse
# ---------------------------------------------------------------------------
struct NumericCast[To: NumericType, A: NumericValue](NumericValue, Unnamed):
    """Numeric -> numeric, reinterpreting the operand's lane at the target
    dtype."""

    comptime Type = Self.To
    comptime ArgType = Self.A.Type
    """The operand's type, named as a member — see `NumericCompare.ArgType`.
    A chained projection is not canonicalised, so `Self.A.Type.native` at a
    call site is reported as unconvertible to *itself*; one projection off a
    local member is."""

    comptime shape = Self.A.shape
    comptime Bound = Self.A.Bound

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()


    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self.a.bind(batch, bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # Structural: a cast changes what a value *is*, never whether there is
        # one. The two breakers below are exactly the exception.
        return self.a.validity(bound)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        # The `.cast` is a no-op that exists to spell the operand's type the
        # same way the parameter does — the projection workaround
        # `TemporalCompare.lane` documents.
        return NumericCastKernel.core[Self.ArgType.native, Self.Type.native, W](
            self.a.lane[W](bound, idx).cast[Self.ArgType.native]()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", Self.Type(), ")")


struct NumToBool[A: NumericValue](BoolValue, Unnamed):
    """Numeric -> bool, as `x != 0`."""

    comptime ArgType = Self.A.Type
    comptime NativeType = Self.ArgType.native
    """The *operand's* width sizes the lane — see `BoolValue.NativeType`. A
    comparison against zero iterates the operand's registers even though it
    emits one bit per row."""

    comptime shape = Self.A.shape
    comptime Bound = Self.A.Bound

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(BoolType())

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self.a.bind(batch, bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(bound)

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        # No-op `.cast`, for the reason `NumericCast.lane` records.
        return NumToBoolKernel.core[Self.NativeType, W](
            self.a.lane[W](bound, idx).cast[Self.NativeType]()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", bool)")


struct BoolToNum[To: NumericType, A: BoolValue](NumericValue, Unnamed):
    """Bool -> numeric, as `True -> 1`, `False -> 0`."""

    comptime Type = Self.To
    comptime shape = Self.A.shape
    comptime Bound = Self.A.Bound

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()


    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self.a.bind(batch, bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return self.a.validity(bound)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return BoolToNumKernel.core[Self.Type.native, W](
            self.a.lane[W](bound, idx)
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", Self.Type(), ")")


# ---------------------------------------------------------------------------
# The two that break
# ---------------------------------------------------------------------------
struct StringToNum[To: NumericType, A: StringValue](
    ColumnBound, NumericValue, Unnamed
):
    """Parse string -> numeric, nulling what will not parse.

    `safe=False` on the kernel is what makes an unparseable value a null rather
    than an error, and it is the reason this node cannot inherit structural
    validity: `to_int("x")` is null where its operand is not.
    """

    comptime Type = Self.To
    comptime From = Self.A.Type
    comptime shape = Shape.columnar
    """Columnar even over a scalar operand: the kernel runs over a column and
    this node reports what it does."""

    comptime Bound = PrimitiveArray[Self.To]
    """The parsed column. `bind` is where the work happens, and — via
    `ColumnBound` — where the nulls come from."""

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()


    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var arr = self.a.evaluate(batch, bindings).to_array(len(batch))
        return StringToNumKernel.apply[Self.From, Self.To, False](
            arr.as_type[BinaryLikeArray[Self.From]]()
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", Self.Type(), ")")


struct NumToString[To: StringLikeType, A: NumericValue](
    ColumnBound, StringValue, Unnamed
):
    """Format numeric -> string."""

    comptime Type = Self.To
    comptime From = Self.A.Type
    comptime shape = Shape.columnar
    comptime Bound = BinaryLikeArray[Self.To]

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()


    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var arr = self.a.evaluate(batch, bindings).to_array(len(batch))
        return NumToStringKernel.apply[Self.From, Self.To](
            arr.as_type[PrimitiveArray[Self.From]]()
        )

    @always_inline
    def lane(self, bound: Self.Bound, idx: Int) -> String:
        # `unsafe_get`, not `bound[idx]`: `lane` cannot raise, and the driver
        # has already consulted validity before asking for this row.
        return String(bound.unsafe_get(UInt(idx)))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.a, ", ", Self.Type(), ")")
