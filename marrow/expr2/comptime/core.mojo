"""The comptime lane: expressions whose structure is their type.

`And[Gt[Column[Int64Type], Literal[Int64Type]], …]` is a type, not data. A whole
subtree therefore compiles to one inlined SIMD loop with no dispatch anywhere
inside it — worth a measured **3.4x** on binary size against the same plan built
from runtime expressions (1.46 MB vs 4.91 MB), because the fused form
instantiates only the kernels it names.

Two consequences follow, and both are load-bearing elsewhere:

- **Nothing outside can inspect the structure.** A rewriter cannot open a type,
  so every question it asks is answered by the node itself — which is what
  `Analyzable` exists for, and why its methods are total.
- **LLVM already optimises the interior.** Constant folding, GVN/CSE and
  instcombine all apply to an inlined loop, so an expression-level rewriter has
  nothing to find here. Interior rewrites belong to the runtime lane alone.

This module holds only what the lane *shares* — the base trait and the family
traits that refine it. The nodes themselves live beside it: `leaves.mojo`,
`operators.mojo`, `reductions.mojo`.

`ComptimeValue` is the base every node here shares. It does *not* declare
`lane` — that returns `SIMD[Type.native, W]`, which only means something for
a fixed-width type, so each family refines the base with its own. `NumericValue`
is the first such family; string, bool, temporal and list follow the same shape.
"""

from ...buffers import Bitmap
from ...dtypes import (
    BoolType,
    PrimitiveType,
    TemporalType,
    DataType,
    DynType,
    NumericType,
    StringLikeType,
)
from ...schema import Schema
from ...arrays import BinaryLikeArray, BoolArray, PrimitiveArray
from ...buffers import Bitmap, Buffer
from ...builders import BinaryLikeBuilder
from ...scalars import PrimitiveScalar
from ...tabular import RecordBatch
from ...views import apply
from ..logical import Shape, Value
from ..physical import Datum
from ..physical import Evaluable, DynOperator, EvalOperator


# ---------------------------------------------------------------------------
# ComptimeValue — what every node in this lane shares
# ---------------------------------------------------------------------------
trait ComptimeValue(Evaluable, Value):
    """A `Value` whose type states its output type and its per-batch state.

    This is where `evaluate` lives — **not** on `Value`. A logical node is
    stateless and has no business exposing a way to run itself; `evaluate` here
    is the *lane's fused driver*, the thing this lane's processor calls once it
    has been handed a batch. It is invisible to `DynValue` and to every
    consumer outside the lane, which reach a value only through
    `to_operator`.

    The two comptime members are what the runtime lane cannot supply, and are
    therefore the whole reason this is a separate trait rather than a naming
    convention: a `RuntimeValue` learns its type from a schema at run time and
    materialises a `DynArray` per node, so it can answer neither.
    """

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        """One fused pass over the batch. The lane's driver, called by
        `EvalOperator`; each family below supplies the default body."""
        ...

    def to_operator(self, grouped: Bool) raises -> DynOperator[Datum]:
        """Every comptime node becomes the same operator — one that forwards
        each batch to the fused driver. `grouped` is ignored: an elementwise
        value has no placement. Aggregates override this with a `FoldOperator`.
        """
        return EvalOperator[Self](self.copy())

    comptime Type: DataType
    """This node's output type, known without a schema.

    `Analyzable.dtype(schema)` ignores its argument in this lane and answers
    from here. The runtime lane is the reason that method takes a schema at all.
    """

    # `Bound`, `bind` and `validity` are deliberately **not** here. All
    # three are *fusion* machinery, and this trait does not mean "fuses" --
    # it means "knows its output type without a schema". A Kleene `AND`
    # knows it produces `bool` and does not fuse at all; making it invent a
    # `Bound` it never reads, to satisfy a base it only needs `Type` from,
    # is how a trait starts describing its first implementer instead of its
    # concept.
    #
    # They are declared on the fusing families, which also do not agree on
    # what validity needs.
    #
    # Structural validity — numeric, comparison, string — is null-in-null-out
    # and answers from the `Bound` alone. Data-dependent validity does not:
    # Kleene `AND` decides its nulls from the operand *values* (`NULL AND
    # FALSE` is `FALSE`), so it needs the batch, and measurement says it should
    # not compute the rule per lane at all — the bitmap algebra in
    # `kernels.boolean._kleene` runs 64 bits per instruction against a SIMD
    # lane's one bit per byte, and beat a fused per-lane version by 4-10x
    # (`bench_boolean.mojo`, 2026-08-22).
    #
    # One signature on the base would therefore have to serve both, which is
    # how `expr/` ended up with two methods (`validity` and `state_validity`)
    # and evaluated `coalesce`/`nullif`/`case_when` twice per fused pass.


# ---------------------------------------------------------------------------
# NumericValue — the first family, refining the base with a lane
# ---------------------------------------------------------------------------
trait PrimitiveValue(ComptimeValue):
    """A comptime node producing a fixed-width column.

    **The machinery, not a domain claim.** Everything below — `Bound`, `bind`,
    `lane[W]`, `validity`, the fused `evaluate` — needs only that the output is
    fixed-width: a `native` dtype, a buffer, a SIMD lane. It says nothing about
    which *operations* the type supports.

    That distinction used to be missing: this trait was `NumericValue`, and
    `Type: NumericType` was a single bound standing for both "I can be read by
    a lane loop" and "I support arithmetic". Two claims in one bound meant a
    temporal column could not be read at all, because dates are not numeric —
    even though reading one is the same instruction. `expr/` answered with a
    second `TemporalColumn` and a duplicated set of comparison arms.

    The domains are now markers on top: `NumericValue` and `TemporalValue` add
    no members and exist so a node can require the *operations* it needs. A
    comparison binds on this trait and serves both; arithmetic binds on
    `NumericValue` and rejects dates at compile time. Same split as
    `AggKernel`'s `OrderedAgg` / `ArithmeticAgg`, for the same reason.
    """

    comptime Type: PrimitiveType

    comptime Bound: Copyable & Deinitable
    """Everything the lane loop needs, resolved once per batch.

    A column leaf's is its typed column; a literal's is nothing; a binary node's
    is `Tuple[L.Bound, R.Bound]`. Declared per concrete struct rather than
    defaulted: a trait default cannot reduce at a `-> Self.Bound` return site
    unless the bound is `ImplicitlyCopyable`, and marrow's array types
    deliberately are not.

    `expr/` called this `State`, which could mean anything. It is specifically
    *this subtree's column references, bound to this batch* — the stage between
    an expression and a per-element read.
    """

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        """Resolve this subtree against `batch`, once, before the lane loop.

        Every schema lookup and every `Variant` unwrap happens here so that
        `lane` does none. That removal is the optimisation.
        """
        ...

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        """One fused pass over the batch — `bind` once, then `lane` per chunk.

        A trait **default**, not a free driver, because it is the same for
        every numeric node and it is precisely what this trait means by
        evaluating. `expr/` kept it as a free `_drive_numeric`; CLAUDE.md
        records that re-defaulting a base trait's abstract method in a
        sub-trait recurses, but that limit is about returning
        `Self.ArrayType`, and `Datum` is concrete.

        Leaves override it: a column returns itself rather than copying through
        a fresh buffer, and a literal stays a scalar.
        """
        comptime native = Self.Type.native
        var bound = self.bind(batch)

        comptime if Self.shape == Shape.scalar:
            # Nothing to iterate — evaluate the lane once and stay lazy, which
            # is what `Shape.scalar` promises its caller.
            return Datum(
                PrimitiveScalar[Self.Type](
                    Optional(self.lane[1](bound, 0)[0]),
                    self.dtype(batch.schema).as_type[Self.Type](),
                ).to_dyn()
            )
        else:
            var length = batch.num_rows()
            var buf = Buffer.alloc_uninit[native](length)

            @always_inline
            def producer[W: Int](i: Int) {imm} -> SIMD[native, W]:
                return self.lane[W](bound, i)

            apply[native](buf.view[native](0, length), producer)

            # Validity once, from the bound — never per lane.
            var v = self.validity(bound)
            var arr = PrimitiveArray[Self.Type](
                dtype=self.dtype(batch.schema).as_type[Self.Type](),
                length=length,
                nulls=v.value().unset_count() if v else 0,
                offset=0,
                bitmap=v^,
                buffer=buf.to_immutable(),
            )
            return arr^.to_dyn()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, or `None` when it cannot be null.

        Structural: a numeric node is null exactly where an operand is, so this
        intersects the operands' bitmaps and never reads their values. That is
        what lets it take the `Bound` and not the batch — a leaf's `Bound` *is*
        its column, bitmap included. `expr/` needed a second method
        (`state_validity`) precisely because its first one took the batch and
        re-ran the whole selection kernel for `coalesce`, `nullif` and
        `case_when`.

        `lane` produces the **data** bits only, so this stays a separate
        question from evaluation. It is a second recursive walk, not a free
        one, but it is affordable: a bitmap is one bit per row against a data
        element's 32 or 64, so a depth-`d` tree costs `d` passes over
        `rows/64` words next to one pass over `rows` elements.
        """
        ...

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        """One SIMD chunk.

        Reads `bound` and `idx` and nothing else — not `self`. That is what
        lets the whole subtree inline into a single loop, and it is why `bind`
        exists as a separate stage rather than the lane reaching through
        `self`.
        """
        ...


# ---------------------------------------------------------------------------
# StringValue — the family whose lane has no width
# ---------------------------------------------------------------------------
trait StringValue(ComptimeValue):
    """A comptime node producing a variable-width string column.

    The family that breaks the SIMD shape, and it is worth being explicit about
    why rather than treating it as an exception. `NumericValue.lane[W]` and
    `BoolValue.lane[W]` both answer `W` elements at once because their storage
    is fixed-width: element `i` is at a computable offset. UTF-8 is not — a
    string's position depends on every string before it — so **`lane` here takes
    no `W` and answers one `String`**. There is no vector to widen to.

    Everything else is unchanged, and that is the point: `bind` still resolves
    the subtree once per batch, `validity` is still structural and still reads
    the `Bound` rather than the batch, and a string subtree still fuses into one
    loop. Fusion is about eliminating dispatch, not about SIMD width, so it
    survives a family that cannot vectorise.
    """

    comptime Type: StringLikeType

    comptime Bound: Copyable & Deinitable
    """This subtree's column references, bound to this batch — as
    `NumericValue.Bound`, and declared per concrete struct for the same
    reason."""

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        """Resolve this subtree against `batch`, once, before the lane loop."""
        ...

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        """One fused pass — `bind` once, then `lane` per row.

        A builder rather than `apply`: `apply` writes fixed-width elements into
        a preallocated buffer, and neither the width nor the total byte count of
        a string result is known before the loop runs. The offsets buffer is
        built as it goes, which is what a `BinaryLikeBuilder` already does.

        Leaves override this: a column hands back its own array rather than
        copying every byte through a fresh builder.
        """
        var bound = self.bind(batch)
        var length = batch.num_rows()
        var builder = BinaryLikeBuilder[Self.Type](length)
        var v = self.validity(bound)
        if v:
            ref bm = v.value()
            var bits = bm.view()
            for i in range(length):
                if bits[i]:
                    builder.append(self.lane(bound, i))
                else:
                    builder.append_null()
        else:
            for i in range(length):
                builder.append(self.lane(bound, i))
        return builder.finish().to_dyn()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, or `None` when it cannot be null.

        Structural, exactly as in `NumericValue`: intersect the operands'
        bitmaps, never read their values.
        """
        ...

    @always_inline
    def lane(self, bound: Self.Bound, idx: Int) -> String:
        """One row. The elementwise counterpart of the SIMD families'
        `lane[W]`, and the only shape a variable-width encoding admits."""
        ...


trait NumericValue(PrimitiveValue):
    """This value supports arithmetic.

    No members — it exists only so a node can say it needs `+` rather than
    merely a readable lane. Mojo has no conditional conformance, so a single
    leaf cannot be numeric for `int64` and temporal for `date32`; the leaves
    therefore differ while everything above them is shared.
    """

    comptime Type: NumericType
    """Narrowed from `PrimitiveValue`. A sub-trait *can* narrow an associated
    type — a conformer cannot, which is why the domains are traits and not a
    bound on the leaf."""


trait TemporalValue(PrimitiveValue):
    """Marker: ordered and comparable, but not arithmetic.

    `date + date` is meaningless, so temporal values are deliberately not
    accepted by `NumericBinary`. Comparison, `min`/`max` and grouping all work,
    because those bind on `PrimitiveValue`.
    """

    comptime Type: TemporalType


# ---------------------------------------------------------------------------
# BoolValue — the family whose lane is bit-packed
# ---------------------------------------------------------------------------
trait BoolValue(ComptimeValue):
    """A comptime node producing a bit-packed boolean column.

    Separate from `NumericValue` because the output is *packed*: a lane yields
    `SIMD[DType.bool, W]` and the driver writes bits, not elements. That is a
    different destination, not a different dtype, which is why `Type` is fixed
    here rather than declared per node — a bool node has no choice about what
    it produces.
    """

    comptime Type = BoolType

    comptime Bound: Copyable & Deinitable
    """Everything the lane loop needs, resolved once per batch.

    A column leaf's is its typed column; a literal's is nothing; a binary node's
    is `Tuple[L.Bound, R.Bound]`. Declared per concrete struct rather than
    defaulted: a trait default cannot reduce at a `-> Self.Bound` return site
    unless the bound is `ImplicitlyCopyable`, and marrow's array types
    deliberately are not.

    `expr/` called this `State`, which could mean anything. It is specifically
    *this subtree's column references, bound to this batch* — the stage between
    an expression and a per-element read.
    """

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        """Resolve this subtree against `batch`, once, before the lane loop.

        Every schema lookup and every `Variant` unwrap happens here so that
        `lane` does none. That removal is the optimisation.
        """
        ...

    comptime NativeType: DType
    """The **operand** width, which sizes the SIMD lane — not the output.

    A comparison over `int64` iterates 64-bit lanes even though it emits one
    bit per row, so `W` follows the operands. Sizing it from the output would
    give a `W` wide enough to overflow the register the operands are loaded
    into. A node with two operands of different widths takes the wider.
    """

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        """One fused bool pass: bit-pack a `Bitmap` from `lane`.

        The numeric default's sibling, and separate for the one reason above —
        the destination is a bitmap, so `apply` takes its bit-packing overload
        and the lane width comes from `NativeType`.
        """
        var length = batch.num_rows()
        var bound = self.bind(batch)
        var bits = Bitmap.alloc_uninit(length)

        @always_inline
        def producer[W: Int](i: Int) {imm} -> SIMD[DType.bool, W]:
            return self.lane[W](bound, i)

        apply[Self.NativeType](bits.view(), producer)

        var v = self.validity(bound)
        return Datum(
            BoolArray(
                length=length,
                nulls=v.value().unset_count() if v else 0,
                offset=0,
                bitmap=v^,
                buffer=bits.to_immutable(),
            ).to_dyn()
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, or `None` when it cannot be null.

        Structural, exactly as `NumericValue.validity` is: a comparison is null
        where an operand is, and never because of what the operands *say*.

        **Kleene `AND`/`OR` do not belong to this family**, and that is the
        reason this signature is allowed to stay simple. Three-valued logic
        decides nulls from operand values, so it can answer neither from the
        `Bound` nor per lane — measured at 4-10x slower per-lane than the
        bitmap algebra in `kernels.boolean._kleene`, which runs 64 bits per
        instruction (`bench_boolean.mojo`, 2026-08-22). They get their own
        family, whose `evaluate` calls the kernel rather than driving a lane.
        """
        ...

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        ...
