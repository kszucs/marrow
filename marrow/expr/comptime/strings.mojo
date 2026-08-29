"""Fused string operators.

The family that cannot vectorise, kept honest about it. A `StringValue`'s
`lane` answers one `String`, so every node here loops where a numeric node
issues one SIMD instruction. Fusion still pays: it removes *dispatch*, not
just width, so `name = 'x' AND amount > 10` is still one pass with no
per-node materialisation.

**Not everything here fuses, and the split is a measured one rather than a
taste.** `StringCompare` and `StringUnary` fuse: their per-row work is one
comparison or one transform, so a `Bound` of operand bounds and a looping
`lane` cost nothing an intermediate column would save.

`StringPredicate` and `StringLength` break — they run a kernel once over the
whole batch in `bind` and read the answer back in `lane`, the `CaseWhen` shape.
For `StringLength` that is because `LengthKernel` is a column fold. For
`StringPredicate` it is `LIKE`: `LikeKernel` and `ILikeKernel` override
`apply_scalar` to compile their pattern **once** (`kernels/string.mojo:340`),
and a fused `lane` would recompile it per row. `StartsWith` and `EndsWith` ride
along on the same node because sharing it costs them only a bitmap they would
otherwise have bit-packed themselves.
"""

from ...arrays import (
    BinaryLikeArray,
    BoolArray,
    Int32Array,
    StructArray,
)
from ...buffers import Bitmap
from ...dtypes import DynType, Int32Type, StringLikeType
from ...kernels.string import (
    CapitalizeKernel,
    ContainsKernel,
    EndsWithKernel,
    ILikeKernel,
    LStripKernel,
    LengthKernel,
    LikeKernel,
    LowerKernel,
    RStripKernel,
    ReverseKernel,
    StartsWithKernel,
    StringEqKernel,
    StringGeKernel,
    StringGtKernel,
    StringLeKernel,
    StringLtKernel,
    StringMapKernel,
    StringNeKernel,
    StringPredicateKernel,
    StripKernel,
    UpperKernel,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape, merged
from ..bindings import Bindings
from ..physical import Datum

from .rules import widest_shape
from .core import BoolValue, ColumnBound, NumericValue, StringValue, Unnamed


struct StringCompare[K: StringPredicateKernel, L: StringValue, R: StringValue](
    BoolValue, Unnamed
):
    """A comparison over two string operands, producing packed bits.

    `NativeType` sizes the SIMD lane the bool driver iterates with, and there is
    no honest answer for strings — the operands have no width. `uint64` is
    chosen to keep `W` small (2 on a 128-bit register): the lane body is a loop
    of `W` string comparisons, so a wide `W` would only mean a longer loop per
    call with no vectorisation to show for it. The *output* is still bit-packed
    exactly as a numeric comparison's is, which is what lets a string predicate
    feed `And`/`Or` and `Filter` unchanged.
    """

    comptime NativeType = DType.uint64
    comptime shape = widest_shape[Self.L, Self.R]
    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (self.l.bind(batch, bindings), self.r.bind(batch, bindings))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """Null-in, null-out: valid exactly where both operands are.

        As in `NumericCompare`, the data bit is computed regardless of validity
        — the loop compares whatever bytes are there — so this bitmap is the
        only record that the result is meaningless in that row.
        """
        return Bitmap.intersect(
            self.l.validity(bound[0]), self.r.validity(bound[1])
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        var out = SIMD[DType.bool, W](fill=False)
        for j in range(W):
            out[j] = Self.K.predicate(
                self.l.lane(bound[0], idx + j),
                self.r.lane(bound[1], idx + j),
            )
        return out

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime StrEq = StringCompare[StringEqKernel, _, _]
comptime StrNe = StringCompare[StringNeKernel, _, _]
comptime StrLt = StringCompare[StringLtKernel, _, _]
comptime StrGt = StringCompare[StringGtKernel, _, _]
comptime StrLe = StringCompare[StringLeKernel, _, _]
comptime StrGe = StringCompare[StringGeKernel, _, _]
"""All six comparisons, not the four the port shipped with.

`StringValue` had `__lt__`/`__gt__` and no `__le__`/`__ge__` precisely because
these two aliases did not exist, so `region >= 'north'` — an ordinary SQL
predicate, and the one `golden/cases/filter_string_ordering.mojo` asks for —
was unwritable. Nothing about the ordering is new: `String.__le__` is the same
bytewise comparison `String.__lt__` already was.
"""


# ---------------------------------------------------------------------------
# StringUnary — string -> string, fused
# ---------------------------------------------------------------------------
struct StringUnary[K: StringMapKernel, A: StringValue](StringValue, Unnamed):
    """Elementwise `string -> string`: `upper`, `lower`, `strip`, and friends.

    Composes in one builder pass, so `upper(col)` feeding a comparison never
    materialises `upper(col)`. The transform itself lives in the kernel; this
    node only says where it goes.
    """

    comptime Type = Self.A.Type
    comptime shape = Self.A.shape
    comptime Bound = Self.A.Bound

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self.a.bind(batch, bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # A map transforms values, never validity: `upper(null)` is null.
        return self.a.validity(bound)

    @always_inline
    def lane(self, bound: Self.Bound, idx: Int) -> String:
        # Bound to a local first: a `StringSlice` over a temporary `String`
        # dangles the moment the temporary is destroyed.
        var s = self.a.lane(bound, idx)
        return Self.K.transform(StringSlice(s))

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime Upper = StringUnary[UpperKernel, _]
comptime Lower = StringUnary[LowerKernel, _]
comptime Strip = StringUnary[StripKernel, _]
comptime LStrip = StringUnary[LStripKernel, _]
comptime RStrip = StringUnary[RStripKernel, _]
comptime Reverse = StringUnary[ReverseKernel, _]
comptime Capitalize = StringUnary[CapitalizeKernel, _]


# ---------------------------------------------------------------------------
# StringPredicate — string x string -> bool, a breaker
# ---------------------------------------------------------------------------
struct StringPredicate[
    K: StringPredicateKernel, L: StringValue, R: StringValue
](BoolValue, ColumnBound, Unnamed):
    """`startswith` / `endswith` / `contains` / `LIKE` / `ILIKE`.

    A breaker, for `LIKE`'s sake: `bind` runs the kernel over the whole batch
    and `lane` reads bits back out of the result.

    **The scalar-pattern branch is the point.** When the right operand is
    `Shape.scalar` this calls `apply_scalar`, which `LikeKernel` and
    `ILikeKernel` override to compile the pattern once
    (`kernels/string.mojo:340`) rather than per row. Going through `apply`
    instead would first broadcast the constant into `n` copies of the same
    string and then recompile it against every one of them.

    `apply_scalar`'s validity comes from the left operand alone, which is
    correct only because no `Shape.scalar` string node can be null:
    `StringLiteral` holds a plain `String` with no validity flag, and
    `StringUnary` forwards its operand's shape *and* its operand's validity, so
    reaching `Shape.scalar` at all forces every leaf under it to be a literal.
    A nullable string literal would break that and would need revisiting here.
    """

    comptime NativeType = DType.int32
    """Sizes the bit-pack driver's lane. The operands have no width — this
    node's `lane` reads bits out of a `BoolArray` the kernel already produced,
    so the choice is about the driver's stride and nothing else."""

    comptime shape = Shape.columnar
    """Columnar even over two scalar operands. The kernel runs over a column
    regardless, so this reports what it does rather than what its operands are
    — as `BoolBinary` and `CaseWhen` do."""

    comptime Bound = BoolArray
    """The computed result, and — via `ColumnBound` — the sole source of
    validity. The previous expression layer re-intersected the operands'
    bitmaps here because its `validity` took the batch rather than the state; the kernel had already
    done that AND, so the answer is simply read off the bound."""

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.l.columns(), self.r.columns())

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var n = len(batch)
        var left = self.l.evaluate(batch, bindings).to_array(n)
        comptime if Self.R.shape == Shape.scalar:
            var rb = self.r.bind(batch, bindings)
            var pattern = self.r.lane(rb, 0)
            return Self.K.apply_scalar(
                left.as_type[BinaryLikeArray[Self.L.Type]](),
                StringSlice(pattern),
            )
        else:
            var right = self.r.evaluate(batch, bindings).to_array(n)
            return Self.K.apply(
                left.as_type[BinaryLikeArray[Self.L.Type]](),
                right.as_type[BinaryLikeArray[Self.R.Type]](),
            )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.l, ", ", self.r, ")")


comptime StartsWith = StringPredicate[StartsWithKernel, _, _]
comptime EndsWith = StringPredicate[EndsWithKernel, _, _]
comptime StrContains = StringPredicate[ContainsKernel, _, _]
comptime Like = StringPredicate[LikeKernel, _, _]
comptime ILike = StringPredicate[ILikeKernel, _, _]


# ---------------------------------------------------------------------------
# StringLength — string -> int32, a breaker
# ---------------------------------------------------------------------------
struct StringLength[A: StringValue](ColumnBound, NumericValue, Unnamed):
    """Byte length of a string value.

    A breaker because `LengthKernel` is a column fold, not a per-row transform:
    `bind` materialises the string stage and folds it to an `Int32Array`, and
    `lane` loads chunks of that. A null string has a null length, and the kernel
    records it — which is why `ColumnBound` is the whole of validity here.
    """

    comptime Type = Int32Type
    comptime shape = Shape.columnar
    comptime Bound = Int32Array

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var s = self.a.evaluate(batch, bindings).to_array(len(batch))
        # The typed `apply`, not `dispatch`: `A.Type` is a comptime parameter
        # here, so the runtime dtype resolution the previous layer needed is
        # gone.
        return LengthKernel.apply(s.as_type[BinaryLikeArray[Self.A.Type]]())

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("length(", self.a, ")")
