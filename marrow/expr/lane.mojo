"""Expression execution — staged, strategy-pluggable fusion (see
`docs/lane-shape-window-design.md`).

Model
-----
- `execute(batch, ctx) -> Datum` is the **one universal verb** every node has.
  `Datum = Scalar | Array` (Arrow's Datum / DataFusion's ColumnarValue) is the
  strategy-agnostic wire format between stages.
- Fusion is a **pluggable strategy**, not a single primitive. A strategy = a
  composable per-element `core` + a driver that runs a whole same-strategy subtree
  in one pass:
    * `NumericValue` — vectorized: `vectorwise[W] -> SIMD[native, W]`, driver fills a `Buffer`.
    * `BoolValue`    — vectorized: `vectorwise[W] -> SIMD[bool, W]`, driver bit-packs a `Bitmap`.
    * `StringValue`  — elementwise: `elementwise(idx) -> String`, driver appends to a builder
      (variable-width UTF-8 has no W-wide lane, so `col || "a" || "b"` fuses one
      row at a time — no intermediate `StringArray`).
- **Pipeline breakers** — cross-row ops (`Reduction`, `WindowFunction`) that can't
  fuse in any strategy — cut the tree into a forest of fused stages. A breaker
  materializes its stage once in `materialize` into a shared `Context`, then behaves as
  an ordinary fused leaf: a scalar reduction reads the context and **splats** (like
  a literal), a columnar window reads it and **loads** (like a column). So the stage
  above still fuses through the single `NumericBinary` — there is no separate
  "materialized" binary.
- Expressions are **immutable**: all per-execute state lives in the `Context`.
  Breaker results are stored positionally; `materialize` (which fills them) and `core`
  (which reads them) visit breakers in the same DFS order, so a plain integer
  `slot` matches reads to writes — no per-lane keying in the hot loop. A breaker
  materializes its operand through a *fresh* sub-context (`run`), so nested breakers
  never perturb the outer slot order.
"""

from std.sys import bit_width_of
from std.builtin.rebind import downcast
from std.utils import Variant
from std.memory import ArcPointer

from ..tabular import RecordBatch
from ..arrays import (
    Array,
    AnyArray,
    PrimitiveArray,
    Int64Array,
    Int32Array,
    BoolArray,
    BinaryLikeArray,
    StringArray,
)
from ..scalars import AnyScalar, PrimitiveScalar, StringScalar
from ..buffers import Buffer, Bitmap
from ..builders import Int64Builder, BinaryLikeBuilder
from ..views import apply
from ..dtypes import (
    DataType,
    NumericType,
    DType,
    Int32Type,
    Int64Type,
    BoolType,
    StringLikeType,
    StringType,
)
from ..kernels.compare import (
    BinaryCompareKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
    EqKernel,
    NeKernel,
)
from ..kernels.arithmetic import (
    BinaryKernel,
    UnaryKernel,
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    ModKernel,
    NegKernel,
    AbsKernel,
)
from ..kernels.aggregate import (
    AggKernel,
    SumKernel,
    MeanKernel,
    MinKernel,
    MaxKernel,
    ProductKernel,
    CountKernel,
)
from ..kernels.string import (
    LengthKernel,
    ConcatKernel,
    StringMapKernel,
    UpperKernel,
    LowerKernel,
    StripKernel,
    LStripKernel,
    RStripKernel,
    ReverseKernel,
    CapitalizeKernel,
    StringPredicateKernel,
    StartsWithKernel,
    EndsWithKernel,
    ContainsKernel,
    StringEqKernel,
    StringNeKernel,
)
from ..kernels.boolean import (
    BoolBinaryKernel,
    BoolUnaryKernel,
    AndKernel,
    OrKernel,
    XorKernel,
    NotKernel,
    UnaryPredicateKernel,
    ValuePredicateKernel,
    IsNullKernel,
    NotNullKernel,
    IsNanKernel,
    IsInfKernel,
)


# ---------------------------------------------------------------------------
# Datum — Scalar | Array, the uniform `execute` result.
# ---------------------------------------------------------------------------
comptime Datum = Variant[AnyScalar, AnyArray]


def into_array(d: Datum, n: Int) raises -> AnyArray:
    """Force `d` to an array of length `n` — broadcasting a scalar (lazy until here)."""
    if d.isa[AnyScalar]():
        return d[AnyScalar].repeat(n)
    return d[AnyArray].copy()


# ---------------------------------------------------------------------------
# Context — per-execute shared scratch. Pipeline-breaker stage results live here,
# positionally: `materialize` appends them in DFS order and `core` reads them back in
# the same order via a `slot`. Keeping results here (not on the nodes) is what
# makes expressions immutable.
# ---------------------------------------------------------------------------
struct Context(Copyable, Movable):
    var _slots: List[Datum]

    def __init__(out self):
        self._slots = List[Datum]()

    def append(mut self, var d: Datum):
        self._slots.append(d^)

    def get(self, i: Int) -> Datum:
        # a `Datum` copy is a ref-count bump (no heap); cheap enough per lane.
        return self._slots[i].copy()

    def get[A: Array](self, i: Int) -> A:
        """Typed slot read — `ctx.get[BoolArray](i)`. Pulls the typed array straight
        out of the slot's `Datum` (a ref-count bump), skipping the `as_xxx().copy()`
        dance at every breaker read."""
        return self._slots[i][AnyArray]._v[A].copy()

    def size(self) -> Int:
        return len(self._slots)


# Known follow-ups (flagged during design; not yet addressed):
#  - PERF: `Context.get` copies a `Datum` per lane for a fused breaker. A pass could
#    hoist scalar splats out of the loop or intern slots to plain scalars/pointers.
#  - CSE: positional slots forgo dedup — identical breakers (`sum(a)` used twice)
#    recompute. A keyed dedup in `materialize` (map subtree-key -> slot) can restore it.
#  - SCHEDULER: independent breakers run sequentially in `materialize`; they are
#    independent stages and can be scheduled to run concurrently.
#  - `AnyScalar.repeat` has no string support, so a string *scalar* cannot broadcast
#    to a column yet (core-array machinery, orthogonal to fusion).


# ---------------------------------------------------------------------------
# Promotion — output dtype is the wider operand (Add(int32,int64) -> int64)
# ---------------------------------------------------------------------------
def _rank[T: DataType]() -> Int:
    comptime if conforms_to(T, NumericType):
        comptime N = downcast[T, NumericType]()
        return bit_width_of[N.native]() + (
            1000 if N.native.is_floating_point() else 0
        )
    else:
        return 0


comptime promote[L: NumericType, R: NumericType] = L if (
    _rank[L]() >= _rank[R]()
) else R


# ---------------------------------------------------------------------------
# Value — every node. `execute` is abstract; `materialize` defaults to a no-op (only
# composites recurse and breakers materialize).
# ---------------------------------------------------------------------------
trait Value(Copyable, ImplicitlyDeletable, Movable):
    comptime OutType: DataType
    comptime Shape: Int  # 0 scalar, 1 columnar

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        ...

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        """Pre-pass before a fused loop: run pipeline-breaker stages into `ctx` in
        DFS order. No-op for leaves (this default); composites recurse; breakers
        override to materialize their stage and append it."""
        pass


def run[V: Value](value: V, batch: RecordBatch) raises -> Datum:
    """Execute a node against a batch with a fresh context — the top-level entry
    (and the fresh sub-context each breaker uses to materialize its operand)."""
    var ctx = Context()
    return value.execute(batch, ctx)


def materialized[V: Value](
    value: V, batch: RecordBatch, mut ctx: Context
) raises -> Datum:
    """A pipeline breaker's `execute`: run its `materialize` (which appends exactly
    one slot — its stage result) and read that slot straight back. Compute lives in
    `materialize` alone; `execute` and the fused path (`materialize` + `vectorwise`)
    share it. Not on the fusion hot path — a fused parent calls a breaker's
    `materialize`/`vectorwise`, never its `execute`."""
    var i = ctx.size()
    value.materialize(batch, ctx)
    return ctx.get(i)


# ---------------------------------------------------------------------------
# NumericValue — the vectorized numeric strategy.
# ---------------------------------------------------------------------------
trait NumericValue(Value):
    comptime OutType: NumericType
    comptime NativeType: DType

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        ...

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        self.materialize(batch, ctx)
        comptime native = Self.NativeType
        comptime if Self.Shape == 0:  # scalar → evaluate the lane once, then splat
            var slot = 0
            var v = self.vectorwise[1](batch, ctx, slot, 0)[0].cast[
                Self.OutType.native
            ]()
            return Datum(PrimitiveScalar[Self.OutType](v).to_any())
        else:  # columnar → one fused vectorized pass
            var length = batch.num_rows()
            var buf = Buffer.alloc_uninit[native](length)

            @parameter
            @always_inline
            def producer[W: Int](i: Int) -> SIMD[native, W]:
                var slot = 0
                return self.vectorwise[W](batch, ctx, slot, i)

            apply[native, producer](buf.view[native](0, length))
            var arr = PrimitiveArray[Self.OutType](
                dtype=Self.OutType(),
                length=length,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buf.to_immutable(),
            )
            return Datum(arr^.to_any())


@fieldwise_init
struct NumericColumn[T: NumericType](NumericValue):
    """A numeric column, read from the batch by position."""

    comptime OutType = Self.T
    comptime Shape = 1
    comptime NativeType = Self.T.native
    var col: Int

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[self.col].as_primitive[Self.T]().values().load[W](idx)
        )


@fieldwise_init
struct NumericLiteral[T: NumericType](NumericValue):
    """A numeric constant, broadcast into every lane."""

    comptime OutType = Self.T
    comptime Shape = 0
    comptime NativeType = Self.T.native
    var _value: Scalar[Self.NativeType]

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        return SIMD[Self.NativeType, W](self._value)


@fieldwise_init
struct NumericBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Fused arithmetic over two operands, widening to the wider dtype. There is no
    "materialized" counterpart: a breaker operand is itself a fused leaf (it reads
    its stage result from `ctx`), so it composes here like any column/literal."""

    comptime OutType = promote[Self.L.OutType, Self.R.OutType]
    comptime Shape = max(Self.L.Shape, Self.R.Shape)
    comptime NativeType = Self.OutType.native
    var l: Self.L
    var r: Self.R

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx).cast[Self.NativeType]()
        var b = self.r.vectorwise[W](batch, ctx, slot, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](a, b)

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.materialize(batch, ctx)
        self.r.materialize(batch, ctx)


@fieldwise_init
struct NumericUnary[K: UnaryKernel, A: NumericValue](NumericValue):
    """Fused unary op preserving the operand dtype — `neg`, `abs`, …."""

    comptime OutType = Self.A.OutType
    comptime Shape = Self.A.Shape
    comptime NativeType = Self.A.NativeType
    var a: Self.A

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        return Self.K.core[Self.NativeType, W](
            self.a.vectorwise[W](batch, ctx, slot, idx)
        )

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.materialize(batch, ctx)


@fieldwise_init
struct NumericCast[To: NumericType, A: NumericValue](NumericValue):
    """Fused numeric → numeric cast — reinterprets the operand's SIMD lane at the
    target dtype, so `col.cast(int64) + other` stays a single fused pass."""

    comptime OutType = Self.To
    comptime Shape = Self.A.Shape
    comptime NativeType = Self.To.native
    var a: Self.A

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        return self.a.vectorwise[W](batch, ctx, slot, idx).cast[Self.NativeType]()

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.materialize(batch, ctx)


comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
comptime Div = NumericBinary[DivKernel, _, _]
comptime Mod = NumericBinary[ModKernel, _, _]
comptime Neg = NumericUnary[NegKernel, _]
comptime Abs = NumericUnary[AbsKernel, _]


# ---------------------------------------------------------------------------
# BoolValue — the vectorized bool strategy. Same SIMD `core`, but the driver
# bit-packs a `Bitmap` (the one physical difference from the numeric lane).
# ---------------------------------------------------------------------------
trait BoolValue(Value):
    comptime NativeType: DType  # operand width (sizes the SIMD lane), not the output

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[DType.bool, W]:
        ...

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        self.materialize(batch, ctx)
        var length = batch.num_rows()
        var bm = Bitmap.alloc_uninit(length)

        @parameter
        @always_inline
        def producer[W: Int](i: Int) -> SIMD[DType.bool, W]:
            var slot = 0
            return self.vectorwise[W](batch, ctx, slot, i)

        apply[Self.NativeType, producer](bm.view())  # bit-packing overload
        return Datum(
            BoolArray(
                length=length,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=bm.to_immutable(),
            ).to_any()
        )


@fieldwise_init
struct NumericCompare[K: BinaryCompareKernel, L: NumericValue, R: NumericValue](
    BoolValue
):
    """Fused numeric comparison → a bit-packed `BoolArray`."""

    comptime OutType = BoolType
    comptime Shape = max(Self.L.Shape, Self.R.Shape)
    comptime NativeType = Self.L.NativeType
    var l: Self.L
    var r: Self.R

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[DType.bool, W]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx)
        var b = self.r.vectorwise[W](batch, ctx, slot, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](a, b)

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.materialize(batch, ctx)
        self.r.materialize(batch, ctx)


comptime Lt = NumericCompare[LtKernel, _, _]
comptime Le = NumericCompare[LeKernel, _, _]
comptime Gt = NumericCompare[GtKernel, _, _]
comptime Ge = NumericCompare[GeKernel, _, _]
comptime Eq = NumericCompare[EqKernel, _, _]
comptime Ne = NumericCompare[NeKernel, _, _]


# ---------------------------------------------------------------------------
# Boolean logic — fused vectorwise over bit-packed masks (bitwise SIMD). Unlike
# `values.mojo` (which materializes bool children), these stay in the bool lane:
# `(a < 3) & (b > 15)` is one fused pass. Compute lives in the boolean kernels.
# ---------------------------------------------------------------------------
@fieldwise_init
struct BoolBinary[K: BoolBinaryKernel, L: BoolValue, R: BoolValue](BoolValue):
    """Fused `and`/`or`/`xor` over two bool masks."""

    comptime OutType = BoolType
    comptime Shape = max(Self.L.Shape, Self.R.Shape)
    comptime NativeType = Self.L.NativeType
    var l: Self.L
    var r: Self.R

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[DType.bool, W]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx)
        var b = self.r.vectorwise[W](batch, ctx, slot, idx)
        return Self.K.core[W](a, b)

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.materialize(batch, ctx)
        self.r.materialize(batch, ctx)


@fieldwise_init
struct BoolUnary[K: BoolUnaryKernel, A: BoolValue](BoolValue):
    """Fused `not` over a bool mask."""

    comptime OutType = BoolType
    comptime Shape = Self.A.Shape
    comptime NativeType = Self.A.NativeType
    var a: Self.A

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[DType.bool, W]:
        return Self.K.core[W](self.a.vectorwise[W](batch, ctx, slot, idx))

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.materialize(batch, ctx)


comptime And = BoolBinary[AndKernel, _, _]
comptime Or = BoolBinary[OrKernel, _, _]
comptime Xor = BoolBinary[XorKernel, _, _]
comptime Not = BoolUnary[NotKernel, _]


# ---------------------------------------------------------------------------
# Unary predicates. `is_nan`/`is_inf` fuse — a per-lane SIMD predicate over a float
# operand (values.mojo materializes them). `is_null`/`not_null` read only validity
# (no value lane), so they're bool breakers over any family. Compute in the kernels.
# ---------------------------------------------------------------------------
@fieldwise_init
struct NumericPredicate[K: ValuePredicateKernel, A: NumericValue](BoolValue):
    """Fused `is_nan`/`is_inf` — a per-lane SIMD predicate over a numeric operand."""

    comptime OutType = BoolType
    comptime Shape = Self.A.Shape
    comptime NativeType = Self.A.NativeType
    var a: Self.A

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[DType.bool, W]:
        return Self.K.core[Self.NativeType, W](
            self.a.vectorwise[W](batch, ctx, slot, idx)
        )

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.materialize(batch, ctx)


@fieldwise_init
struct NullPredicate[K: UnaryPredicateKernel, A: Value](BoolValue):
    """`is_null`/`not_null` — reads the operand's validity (any family), so a bool
    breaker: materialize the operand, run the kernel into a `BoolArray`, read it."""

    comptime OutType = BoolType
    comptime Shape = Self.A.Shape
    comptime NativeType = DType.int32  # lane width for the bit-pack driver
    var a: Self.A

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        var arr = into_array(run(self.a, batch), batch.num_rows())
        ctx.append(Datum(Self.K.apply(arr).to_any()))

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[DType.bool, W]:
        var s = slot
        slot += 1
        return ctx.get[BoolArray](s).values().load[DType.bool, W](idx)

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return materialized(self, batch, ctx)


comptime IsNan = NumericPredicate[IsNanKernel, _]
comptime IsInf = NumericPredicate[IsInfKernel, _]
comptime IsNull = NullPredicate[IsNullKernel, _]
comptime NotNull = NullPredicate[NotNullKernel, _]


# ---------------------------------------------------------------------------
# StringValue — the elementwise string strategy. No W-wide lane: `core(idx)`
# yields one row's `String`, and the driver appends them into a builder, so a
# concat chain fuses without materializing intermediate string arrays.
# ---------------------------------------------------------------------------
trait StringValue(Value):
    comptime OutType: StringLikeType

    @always_inline
    def elementwise(self, batch: RecordBatch, ctx: Context, idx: Int) -> String:
        ...

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        self.materialize(batch, ctx)
        comptime if Self.Shape == 0:
            return Datum(StringScalar(self.elementwise(batch, ctx, 0)).to_any())
        else:
            var n = batch.num_rows()
            var builder = BinaryLikeBuilder[Self.OutType](capacity=n)
            for i in range(n):
                builder.append(self.elementwise(batch, ctx, i))
            return Datum(builder.finish().to_any())


@fieldwise_init
struct StringColumn[T: StringLikeType](StringValue):
    """A string column, read from the batch by position."""

    comptime OutType = Self.T
    comptime Shape = 1
    var col: Int

    @always_inline
    def elementwise(self, batch: RecordBatch, ctx: Context, idx: Int) -> String:
        return String(batch.columns[self.col].as_string().unsafe_get(UInt(idx)))


@fieldwise_init
struct StringLiteral[T: StringLikeType](StringValue):
    """A string constant, broadcast into every row."""

    comptime OutType = Self.T
    comptime Shape = 0
    var _value: String

    @always_inline
    def elementwise(self, batch: RecordBatch, ctx: Context, idx: Int) -> String:
        return self._value.copy()


@fieldwise_init
struct Concat[L: StringValue, R: StringValue](StringValue):
    """Fused elementwise concatenation — `col || "a" || "b"` builds each row once,
    no intermediate `StringArray` for `col || "a"`."""

    comptime OutType = Self.L.OutType
    comptime Shape = max(Self.L.Shape, Self.R.Shape)
    var l: Self.L
    var r: Self.R

    @always_inline
    def elementwise(self, batch: RecordBatch, ctx: Context, idx: Int) -> String:
        return ConcatKernel.combine(
            self.l.elementwise(batch, ctx, idx),
            self.r.elementwise(batch, ctx, idx),
        )

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.l.materialize(batch, ctx)
        self.r.materialize(batch, ctx)


@fieldwise_init
struct StringUnary[K: StringMapKernel, A: StringValue](StringValue):
    """Fused elementwise `string -> string` (`upper`/`lower`/`strip`/…). Composes in
    one builder pass with concat: `upper(col) || "!"` never materializes `upper(col)`.
    The transform lives in the kernel."""

    comptime OutType = Self.A.OutType
    comptime Shape = Self.A.Shape
    var a: Self.A

    @always_inline
    def elementwise(self, batch: RecordBatch, ctx: Context, idx: Int) -> String:
        var s = self.a.elementwise(batch, ctx, idx)
        return Self.K.transform(StringSlice(s))

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        self.a.materialize(batch, ctx)


comptime Upper = StringUnary[UpperKernel, _]
comptime Lower = StringUnary[LowerKernel, _]
comptime Strip = StringUnary[StripKernel, _]
comptime LStrip = StringUnary[LStripKernel, _]
comptime RStrip = StringUnary[RStripKernel, _]
comptime Reverse = StringUnary[ReverseKernel, _]
comptime Capitalize = StringUnary[CapitalizeKernel, _]


# ---------------------------------------------------------------------------
# String predicates — `string × string -> bool`. Variable-width comparison has no
# vectorwise lane, so this is a bool *breaker*: `materialize` materializes both string
# stages and runs the kernel into a `BoolArray`; `vectorwise` reads that mask, so a
# predicate still fuses under boolean logic. The comparison lives in the kernel.
# ---------------------------------------------------------------------------
@fieldwise_init
struct StringPredicate[
    K: StringPredicateKernel, L: StringValue, R: StringValue
](BoolValue):
    comptime OutType = BoolType
    comptime Shape = max(Self.L.Shape, Self.R.Shape)
    comptime NativeType = DType.int32  # lane width for the bit-pack driver
    var l: Self.L
    var r: Self.R

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        var n = batch.num_rows()
        var la = into_array(run(self.l, batch), n).as_string().copy()
        var ra = into_array(run(self.r, batch), n).as_string().copy()
        ctx.append(Datum(Self.K.apply(la, ra).to_any()))

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[DType.bool, W]:
        var s = slot
        slot += 1
        return ctx.get[BoolArray](s).values().load[DType.bool, W](idx)

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return materialized(self, batch, ctx)


comptime StartsWith = StringPredicate[StartsWithKernel, _, _]
comptime EndsWith = StringPredicate[EndsWithKernel, _, _]
comptime StrContains = StringPredicate[ContainsKernel, _, _]
comptime StrEq = StringPredicate[StringEqKernel, _, _]
comptime StrNe = StringPredicate[StringNeKernel, _, _]


# ---------------------------------------------------------------------------
# Strategy transition (string -> numeric) — modelled as a plain breaker. The two
# strategies don't compose, so the string (elementwise) stage materializes; the
# `LengthKernel` folds it to the int32 length column (vectorized offset subtraction,
# handling string / large_string). `vectorwise` then just loads that column, so the
# arithmetic above fuses — exactly like a window: `length(s) + 1` is one numeric pass
# over the materialized lengths. Same shape as every other breaker.
# ---------------------------------------------------------------------------
@fieldwise_init
struct StringLength[A: StringValue](NumericValue):
    """Byte length of a string value → int32. `materialize` materializes the string
    stage and folds it to the length column via `LengthKernel`; `vectorwise` loads
    that column per lane."""

    comptime OutType = Int32Type
    comptime Shape = 1
    comptime NativeType = DType.int32
    var a: Self.A

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        var s = into_array(run(self.a, batch), batch.num_rows())
        ctx.append(Datum(LengthKernel.dispatch(s)))

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        var s = slot
        slot += 1
        return ctx.get[Int32Array](s).values().load[W](idx)

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return materialized(self, batch, ctx)


# ---------------------------------------------------------------------------
# Pipeline breakers — cross-row `Value`s that cut the tree into stages. They
# materialize their operand through a *fresh* sub-context (`run`, so nested
# breakers don't perturb the outer slot order), then act as fused leaves: a scalar
# reduction splats, a columnar window loads. `materialize` appends the result; `core`
# reads it back positionally via `slot`.
# ---------------------------------------------------------------------------
@fieldwise_init
struct Reduction[K: AggKernel, A: NumericValue](NumericValue):
    """Whole-array reduction → a scalar. Output dtype is the kernel's accumulator
    algebra `K.AccType[A.OutType]` (sum widens, mean → float64, min/max keep it)."""

    comptime OutType = Self.K.AccType[Self.A.OutType]
    comptime Shape = 0
    comptime NativeType = Self.OutType.native
    var a: Self.A

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        var arg = into_array(run(self.a, batch), batch.num_rows())
        ctx.append(Datum(Self.K.reduce(arg)))

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        var d = ctx.get(slot)
        slot += 1
        return SIMD[Self.NativeType, W](
            d[AnyScalar].as_primitive[Self.OutType]().value()
        )

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return materialized(self, batch, ctx)


comptime Sum = Reduction[SumKernel, _]
comptime Mean = Reduction[MeanKernel, _]
comptime Min = Reduction[MinKernel, _]
comptime Max = Reduction[MaxKernel, _]
comptime Product = Reduction[ProductKernel, _]
comptime Count = Reduction[CountKernel, _]


@fieldwise_init
struct FrameBound(Copyable, ImplicitlyCopyable, Movable):
    var kind: UInt8
    var offset: Int64


@fieldwise_init
struct WindowSpec(Copyable, Movable):
    """Partition/order/frame — the toy carries frame bounds only."""

    var start: FrameBound
    var end: FrameBound


trait WindowKernel:
    comptime name: String
    comptime OutType: NumericType

    @staticmethod
    def evaluate_all(values: AnyArray) raises -> AnyArray:
        ...


struct RowNumberKernel(WindowKernel):
    comptime name = "row_number"
    comptime OutType = Int64Type

    @staticmethod
    def evaluate_all(values: AnyArray) raises -> AnyArray:
        # frame-independent (DataFusion `evaluate_all`): row_number = 1..n.
        var n = len(values)
        var b = Int64Builder(n)
        for i in range(n):
            b.append(Int64(i + 1))
        return b.finish().to_any()


@fieldwise_init
struct WindowFunction[Func: WindowKernel, A: Value](NumericValue):
    """`func.over(spec)` → a columnar breaker. `materialize` materializes the whole
    output column into `ctx`; `core` then loads it per lane like a column."""

    comptime OutType = Self.Func.OutType
    comptime Shape = 1
    comptime NativeType = Self.Func.OutType.native
    var a: Self.A
    var spec: WindowSpec

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises:
        var v = into_array(run(self.a, batch), batch.num_rows())
        ctx.append(Datum(Self.Func.evaluate_all(v)))

    @always_inline
    def vectorwise[
        W: Int
    ](
        self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int
    ) -> SIMD[Self.NativeType, W]:
        var s = slot
        slot += 1
        return ctx.get[PrimitiveArray[Self.OutType]](s).values().load[W](idx)

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return materialized(self, batch, ctx)


comptime RowNumber = WindowFunction[RowNumberKernel, _]


# ---------------------------------------------------------------------------
# AnyValue — erase any node to a boxed handle. Because `execute` already returns
# a concrete `Datum`, erasure is a plain fn-pointer trampoline.
# ---------------------------------------------------------------------------
struct AnyValue(Copyable, Movable):
    var _boxed: ArcPointer[NoneType]
    var _execute: def (
        ArcPointer[NoneType], RecordBatch, mut Context
    ) thin raises -> Datum

    @staticmethod
    def _exec_tramp[
        V: Value
    ](
        ptr: ArcPointer[NoneType], batch: RecordBatch, mut ctx: Context
    ) raises -> Datum:
        return rebind[ArcPointer[V]](ptr)[].execute(batch, ctx)

    @implicit
    def __init__[V: Value](out self, value: V):
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._execute = Self._exec_tramp[V]

    def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        return self._execute(self._boxed, batch, ctx)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def col[T: NumericType](i: Int, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column by position — `col(0, int64)`."""
    return NumericColumn[T](i)


def lit[T: NumericType](value: Int, dtype: T) -> NumericLiteral[T]:
    """A numeric constant — `lit(10, int64)`."""
    return NumericLiteral[T](Scalar[T.native](value))


def scol[T: StringLikeType](i: Int, dtype: T) -> StringColumn[T]:
    """Reference a string column by position — `scol(0, string)`."""
    return StringColumn[T](i)


def slit(value: String) -> StringLiteral[StringType]:
    """A string constant — `slit("suffix")`."""
    return StringLiteral[StringType](value)
