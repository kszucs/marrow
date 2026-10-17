# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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
`apply_scalar` to compile their pattern **once** (`kernels/string.mojo:343`),
and a fused `lane` would recompile it per row. `StartsWith` and `EndsWith` ride
along on the same node because sharing it costs them only a bitmap they would
otherwise have bit-packed themselves.
"""

from ...arrays import (
    BinaryLikeArray,
    BoolArray,
    Int32Array,
    Int64Array,
    PrimitiveArray,
    StructArray,
)
from ...buffers import Bitmap
from ...dtypes import (
    DynType,
    Int32Type,
    Int64Type,
    StringLikeType,
    StringType,
)
from ...kernels.regex import (
    RegexpExtractKernel,
    RegexpMatchesKernel,
    RegexpReplaceKernel,
)
from ...kernels.string import (
    AsciiKernel,
    CapitalizeKernel,
    CharLengthKernel,
    ConcatKernel,
    ContainsKernel,
    EndsWithKernel,
    Pad,
    LeftKernel,
    PositionKernel,
    RepeatKernel,
    ReplaceKernel,
    RightKernel,
    SplitPartKernel,
    StringIntKernel,
    StringTextIntKernel,
    StringTextTextKernel,
    StringToIntKernel,
    SubstrKernel,
    TrimCharsKernel,
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
from ..logical import Shape
from ..logical import Bindings
from ...engine.core import Datum

from .rules import widest_shape
from .subqueries import bare_reads
from .core import BoolValue, ColumnBound, NumericValue, StringValue, Unnamed
from .leaves import NumericLiteral, StringLiteral


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

    def equates(self) -> Bool:
        """Equality between two bare reads — a decorrelation join key."""
        comptime if Self.K.name == StringEqKernel.name:
            return bare_reads(self.l, self.r)
        else:
            return False

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

    The transform lives in the kernel; this node only says where it goes.

    **The one string node that produces new bytes, so it is the one whose
    `bind` materialises.** `StringValue.lane` borrows from `Bound`, and a
    transform has nowhere to borrow from -- it has to put the result
    somewhere. Doing it per row is what the old owning `lane` did: one
    allocation per element. `bind` runs the kernel's own null-preserving
    `apply` once per batch instead, which is the same shape `NumToString`
    already had, and `lane` then borrows like every other conformer.
    """

    comptime Type = Self.A.Type
    comptime shape = Self.A.shape
    comptime Bound = BinaryLikeArray[Self.Type]

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var arr = self.a.evaluate(batch, bindings).to_array(len(batch))
        return Self.K.apply[Self.Type](arr.get[Self.Bound]())

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # A map transforms values, never validity: `upper(null)` is null, and
        # `K.apply` is null-preserving, so the bound column already says so.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # `unsafe_get` borrows from `bound.values`, a *field*; the trait
        # promises a borrow from `bound`. A field borrow is valid for at least
        # as long as the struct that holds it, so widening is sound -- Mojo
        # just will not do it implicitly.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

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
    (`kernels/string.mojo:343`) rather than per row. Going through `apply`
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

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var n = len(batch)
        var left = self.l.evaluate(batch, bindings).to_array(n)
        comptime if Self.R.shape == Shape.scalar:
            var rb = self.r.bind(batch, bindings)
            var pattern = self.r.lane(rb, 0)
            return Self.K.apply_scalar(
                left.get[BinaryLikeArray[Self.L.Type]](),
                StringSlice(pattern),
            )
        else:
            var right = self.r.evaluate(batch, bindings).to_array(n)
            return Self.K.apply(
                left.get[BinaryLikeArray[Self.L.Type]](),
                right.get[BinaryLikeArray[Self.R.Type]](),
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
comptime RegexpMatches = StringPredicate[RegexpMatchesKernel, _, _]


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

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var s = self.a.evaluate(batch, bindings).to_array(len(batch))
        # The typed `apply`, not `dispatch`: `A.Type` is a comptime parameter
        # here, so the runtime dtype resolution the previous layer needed is
        # gone.
        return LengthKernel.apply(s.get[BinaryLikeArray[Self.A.Type]]())

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("length(", self.a, ")")


# ---------------------------------------------------------------------------
# The SQL string functions — breakers over each kernel's own signature
# ---------------------------------------------------------------------------
#
# `substr`, `left`, `right`, `repeat`, `lpad`, `rpad`, `replace`,
# `regexp_replace`, `split_part`, `regexp_extract`, `trim(chars)` answer a
# string; `char_length`, `ascii` and `position` an `int64`. Their arguments are
# **operands**, so `substr(s, 2, 3)` and `substr(s, col("from"), col("len"))`
# are one node with different operand types, and a literal operand broadcasts
# once per batch. Each node binds its operands and runs its kernel's `apply`
# once per batch, which is what makes it a breaker: the transform produces new
# bytes, and `lane` borrows out of the result.


def _text[
    V: StringValue
](value: V, batch: StructArray, bindings: Bindings) raises -> BinaryLikeArray[
    V.Type
]:
    """`value` over `batch` as a column."""
    return (
        value.evaluate(batch, bindings)
        .to_array(len(batch))
        .get[BinaryLikeArray[V.Type]]()
        .copy()
    )


def _number[
    V: NumericValue
](value: V, batch: StructArray, bindings: Bindings) raises -> PrimitiveArray[
    V.Type
]:
    """`value` over `batch` as a column."""
    return (
        value.evaluate(batch, bindings)
        .to_array(len(batch))
        .get[PrimitiveArray[V.Type]]()
        .copy()
    )


comptime TextLiteral = StringLiteral[StringType]
"""The leaf a constant string argument becomes."""
comptime IntLiteral = NumericLiteral[Int64Type]
"""The leaf a constant integer argument becomes."""


struct StringIntFunction[K: StringIntKernel, A: StringValue, C: NumericValue](
    StringValue, Unnamed
):
    """`left`, `right` and `repeat`: a string and a count."""

    comptime Type = Self.A.Type
    comptime shape = Shape.columnar
    """Columnar even over scalar operands: the kernel runs over a column
    regardless."""
    comptime Bound = BinaryLikeArray[Self.Type]

    var a: Self.A
    var count: Self.C

    def __init__(out self, var a: Self.A, var count: Self.C):
        self.a = a^
        self.count = count^

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return Self.K.apply(
            _text(self.a, batch, bindings), _number(self.count, batch, bindings)
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # The kernel already intersected every operand's validity.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # Widening a field borrow to the struct's origin, as `StringUnary` does.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ", ", self.count, ")")


comptime Left = StringIntFunction[LeftKernel, _, _]
comptime Right = StringIntFunction[RightKernel, _, _]
comptime Repeat = StringIntFunction[RepeatKernel, _, _]


struct Substr[A: StringValue, S: NumericValue, C: NumericValue](
    StringValue, Unnamed
):
    """`substr(s, start, count)`."""

    comptime Type = Self.A.Type
    comptime shape = Shape.columnar
    """Columnar even over scalar operands: the kernel runs over a column
    regardless."""
    comptime Bound = BinaryLikeArray[Self.Type]

    var a: Self.A
    var start: Self.S
    var count: Self.C

    def __init__(out self, var a: Self.A, var start: Self.S, var count: Self.C):
        self.a = a^
        self.start = start^
        self.count = count^

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return SubstrKernel.apply(
            _text(self.a, batch, bindings),
            _number(self.start, batch, bindings),
            _number(self.count, batch, bindings),
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # The kernel already intersected every operand's validity.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # Widening a field borrow to the struct's origin, as `StringUnary` does.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            SubstrKernel.name,
            "(",
            self.a,
            ", ",
            self.start,
            ", ",
            self.count,
            ")",
        )


struct StringPad[
    on_left: Bool, A: StringValue, F: StringValue, W: NumericValue
](StringValue, Unnamed):
    """`lpad`/`rpad(s, width, fill)`."""

    comptime Type = Self.A.Type
    comptime shape = Shape.columnar
    """Columnar even over scalar operands: the kernel runs over a column
    regardless."""
    comptime Bound = BinaryLikeArray[Self.Type]

    var a: Self.A
    var fill: Self.F
    var width: Self.W

    def __init__(out self, var a: Self.A, var fill: Self.F, var width: Self.W):
        self.a = a^
        self.fill = fill^
        self.width = width^

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return Pad[Self.on_left].apply(
            _text(self.a, batch, bindings),
            _number(self.width, batch, bindings),
            _text(self.fill, batch, bindings),
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # The kernel already intersected every operand's validity.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # Widening a field borrow to the struct's origin, as `StringUnary` does.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[Wr: Writer](self, mut writer: Wr):
        writer.write(
            Pad[Self.on_left].name,
            "(",
            self.a,
            ", ",
            self.width,
            ", ",
            self.fill,
            ")",
        )


comptime LPad = StringPad[True, _, _, _]
comptime RPad = StringPad[False, _, _, _]


struct StringTextTextFunction[
    K: StringTextTextKernel, A: StringValue, X: StringValue, Y: StringValue
](StringValue, Unnamed):
    """`replace` and `regexp_replace`: a string, a pattern and a
    replacement."""

    comptime Type = Self.A.Type
    comptime shape = Shape.columnar
    """Columnar even over scalar operands: the kernel runs over a column
    regardless."""
    comptime Bound = BinaryLikeArray[Self.Type]

    var a: Self.A
    var x: Self.X
    var y: Self.Y

    def __init__(out self, var a: Self.A, var x: Self.X, var y: Self.Y):
        self.a = a^
        self.x = x^
        self.y = y^

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return Self.K.apply(
            _text(self.a, batch, bindings),
            _text(self.x, batch, bindings),
            _text(self.y, batch, bindings),
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # The kernel already intersected every operand's validity.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # Widening a field borrow to the struct's origin, as `StringUnary` does.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ", ", self.x, ", ", self.y, ")")


comptime Replace = StringTextTextFunction[ReplaceKernel, _, _, _]
comptime RegexpReplace = StringTextTextFunction[RegexpReplaceKernel, _, _, _]


struct StringTextIntFunction[
    K: StringTextIntKernel, A: StringValue, X: StringValue, C: NumericValue
](StringValue, Unnamed):
    """`split_part` and `regexp_extract`: a string, a separator or pattern,
    and an index."""

    comptime Type = Self.A.Type
    comptime shape = Shape.columnar
    """Columnar even over scalar operands: the kernel runs over a column
    regardless."""
    comptime Bound = BinaryLikeArray[Self.Type]

    var a: Self.A
    var x: Self.X
    var c: Self.C

    def __init__(out self, var a: Self.A, var x: Self.X, var c: Self.C):
        self.a = a^
        self.x = x^
        self.c = c^

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return Self.K.apply(
            _text(self.a, batch, bindings),
            _text(self.x, batch, bindings),
            _number(self.c, batch, bindings),
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # The kernel already intersected every operand's validity.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # Widening a field borrow to the struct's origin, as `StringUnary` does.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ", ", self.x, ", ", self.c, ")")


comptime SplitPart = StringTextIntFunction[SplitPartKernel, _, _, _]
comptime RegexpExtract = StringTextIntFunction[RegexpExtractKernel, _, _, _]


struct TrimChars[A: StringValue, X: StringValue](StringValue, Unnamed):
    """`trim(s, characters)`."""

    comptime Type = Self.A.Type
    comptime shape = Shape.columnar
    """Columnar even over scalar operands: the kernel runs over a column
    regardless."""
    comptime Bound = BinaryLikeArray[Self.Type]

    var a: Self.A
    var chars: Self.X

    def __init__(out self, var a: Self.A, var chars: Self.X):
        self.a = a^
        self.chars = chars^

    def dtype(self, schema: Schema) raises -> DynType:
        return self.a.dtype(schema)

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return TrimCharsKernel.apply(
            _text(self.a, batch, bindings), _text(self.chars, batch, bindings)
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # The kernel already intersected every operand's validity.
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # Widening a field borrow to the struct's origin, as `StringUnary` does.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(TrimCharsKernel.name, "(", self.a, ", ", self.chars, ")")


struct StringMeasure[K: StringToIntKernel, A: StringValue](
    ColumnBound, NumericValue, Unnamed
):
    """`char_length` and `ascii`: a string measured as `int64`.

    `int64`, where `StringLength` answers `int32`: `StringLength` is Arrow's
    `utf8_length`, while these are the SQL spellings, and SQL's `length` and
    `ascii` are `BIGINT`."""

    comptime Type = Int64Type
    comptime shape = Shape.columnar
    comptime Bound = Int64Array

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return Self.K.apply(_text(self.a, batch, bindings))

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.a, ")")


comptime CharLength = StringMeasure[CharLengthKernel, _]
comptime Ascii = StringMeasure[AsciiKernel, _]


struct Position[A: StringValue, X: StringValue](
    ColumnBound, NumericValue, Unnamed
):
    """`position(needle IN s)`, as `int64`."""

    comptime Type = Int64Type
    comptime shape = Shape.columnar
    comptime Bound = Int64Array

    var a: Self.A
    var needle: Self.X

    def __init__(out self, var a: Self.A, var needle: Self.X):
        self.a = a^
        self.needle = needle^

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return PositionKernel.apply(
            _text(self.a, batch, bindings), _text(self.needle, batch, bindings)
        )

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(PositionKernel.name, "(", self.a, ", ", self.needle, ")")


# ---------------------------------------------------------------------------
# StringConcat — string x string -> string, a breaker
# ---------------------------------------------------------------------------
struct StringConcat[L: StringValue, R: StringValue](StringValue, Unnamed):
    """SQL `||` — concatenation that **propagates null**.

    The half of the concat family that answers null when either operand is,
    which is what separates `||` from `concat()`. `ConcatKernel.apply` already
    had exactly this rule and, until now, no node that reached it.

    A breaker: it produces new bytes, so `bind` materialises and `lane` borrows.
    """

    comptime Type = Self.L.Type
    comptime shape = Shape.columnar
    comptime Bound = BinaryLikeArray[Self.Type]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    # -- Value --------------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        return self.l.dtype(schema)

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var n = len(batch)
        var left = self.l.evaluate(batch, bindings).to_array(n)
        var right = self.r.evaluate(batch, bindings).to_array(n)
        return ConcatKernel.apply(
            left.get[BinaryLikeArray[Self.L.Type]](),
            right.get[BinaryLikeArray[Self.R.Type]](),
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return bound.bitmap

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("concat(", self.l, ", ", self.r, ")")
