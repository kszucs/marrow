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
from ...dtypes import BoolType, DataType, DynType, NumericType
from ...schema import Schema
from ...arrays import BoolArray, PrimitiveArray
from ...buffers import Bitmap, Buffer
from ...scalars import PrimitiveScalar
from ...tabular import RecordBatch
from ...views import apply
from ..core import Analyzable, Datum, Evaluable, Shape


# ---------------------------------------------------------------------------
# ComptimeValue — what every node in this lane shares
# ---------------------------------------------------------------------------
trait ComptimeValue(Analyzable, Copyable, Deinitable, Evaluable, Writable):
    """A `Value` whose type states its output type and its per-batch state.

    The two comptime members are what the runtime lane cannot supply, and are
    therefore the whole reason this is a separate trait rather than a naming
    convention: a `RuntimeValue` learns its type from a schema at run time and
    materialises a `DynArray` per node, so it can answer neither.
    """

    comptime Type: DataType
    """This node's output type, known without a schema.

    `Analyzable.dtype(schema)` ignores its argument in this lane and answers
    from here. The runtime lane is the reason that method takes a schema at all.
    """

    comptime Bound: Copyable & Deinitable
    """Everything the lane loop needs, resolved once per batch.

    A column leaf's is its typed column; a literal's is nothing; a binary node's
    is `Pair[L.Bound, R.Bound]`. Declared per concrete struct rather than
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

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        """This node's result validity, or `None` when it cannot be null.

        `lane` produces the **data** bits only, so validity is a separate and
        genuinely data-dependent question: a Kleene `AND` derives three-valued
        nulls from its operands, which the plain bitwise `a & b` in the lane
        does not give. Fusing without this contract reproduces the defect class
        where a comparison's data bit is read while its validity says the bit
        is meaningless.

        **This is not fused, and that is a deliberate asymmetry.** `evaluate`
        collapses a whole subtree into one loop; validity is a second recursive
        walk that intersects a bitmap per level. It is affordable because a
        bitmap is one bit per row against a data element's 32 or 64 — a
        depth-`d` tree costs `d` passes over `rows/64` words, next to one pass
        over `rows` elements — but it is a second walk, not a free one.

        Folding it into `bind` would remove the walk, since `bind` already
        descends the tree. That works for **structural** validity (null in,
        null out) and not for data-dependent validity: a Kleene `AND` decides
        its nulls from the values, so it cannot be known before the lane runs.
        Any fused form has to keep both paths, which is why this stays a
        separate method until something measures the walk as costing.

        Takes the `Bound`, not the batch. `expr/` had **two** methods —
        `validity(batch)` and `state_validity(batch, state)` — the second added
        because the first re-ran the whole selection kernel for `coalesce`,
        `nullif` and `case_when`, so every fused pass over them did the work
        twice. One method reading the already-resolved `Bound` cannot: a leaf's
        `Bound` *is* its column, bitmap included.
        """
        ...


# ---------------------------------------------------------------------------
# NumericValue — the first family, refining the base with a lane
# ---------------------------------------------------------------------------
trait NumericValue(ComptimeValue):
    """A comptime node producing a fixed-width numeric column."""

    comptime Type: NumericType

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
                PrimitiveScalar[Self.Type](self.lane[1](bound, 0)[0]).to_dyn()
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
                dtype=Self.Type(),
                length=length,
                nulls=v.value().unset_count() if v else 0,
                offset=0,
                bitmap=v^,
                buffer=buf.to_immutable(),
            )
            return Datum(arr^.to_dyn())

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

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        ...
