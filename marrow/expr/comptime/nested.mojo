"""The comptime lane's list-consuming nodes.

`ListLength` and `ArrayContains` share an operand bound and nothing else:
both take a `ListValue`, and they land in *different* families — a length is
`NumericValue`, a membership test is `BoolValue`. That is the same reason
`temporal.mojo` is its own module rather than more of `numeric.mojo`, and it
is why these are not in `numeric.mojo` and `boolean.mojo` respectively.

They are not leaves, so they do not belong in `leaves.mojo` either — every
node there resolves a name or holds a constant and takes no operand at all.
`ListLength` lived there historically, from before this package had a module
per operand family; `ArrayContains` was added beside it and the pair moved
here together.

Neither node fuses. `ListValue` declares no lane, because a list element is a
whole sub-array rather than a register-width value, so both are breakers:
`bind` runs the kernel over the entire column and `lane` reads the result
back.
"""

from ...arrays import BoolArray, Int32Array, StructArray
from ...dtypes import DynType, Int32Type
from ...kernels.nested import ArrayContainsKernel, ArrayLengthKernel
from ...schema import Schema
from ..bindings import Bindings
from ..logical import Shape, merged
from .core import (
    BoolValue,
    ColumnBound,
    ListValue,
    NumericValue,
    Unnamed,
)


struct ListLength[A: ListValue](ColumnBound, NumericValue, Unnamed):
    """`array_length(list)` — a list consumed into a fixed-width column.

    The shape every list operation takes: it binds a `ListValue` and produces a
    lane of its own family. That is why `ListValue` needs no `lane` — nothing
    reads a list element as a value, only as something to measure or search.

    `bind` runs `ArrayLengthKernel` over the whole column and `lane` reads the
    result, as `CaseWhen` does: the work is offset arithmetic the kernel
    already vectorises, and a per-element lane would only re-derive it.
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

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Int32Type())

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return ArrayLengthKernel.apply(self.a.bind(batch, bindings))

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("array_length(", self.a, ")")


struct ArrayContains[L: ListValue, E: NumericValue](
    BoolValue, ColumnBound, Unnamed
):
    """`array_contains(list, elem)` — does `list[i]` hold the value `elem[i]`?

    `ListLength`'s sibling and the second instance of the shape that entry
    describes: a `ListValue` bound, a lane of another family out. The
    difference is only which family — a length is numeric, a membership test
    is boolean — so this is a `BoolValue` breaker in the mould of
    `NullPredicate`: `bind` runs the kernel over the whole column and `lane`
    reads the bits back.

    Two parameters rather than one because both operands are typed at compile
    time: `L.Type` picks the offset width and `E.Type` the element type, so
    `ArrayContainsKernel.apply` binds directly. Going through its `dispatch`
    would open a listlike x numeric ladder — every offset width against every
    numeric type — for a pair this lane already knows.

    **The search value is a column, not a constant**, matching the kernel:
    row `i` looks for `elem[i]` in `list[i]`. A constant search value is
    `lit(3, int64)`, which stays `Shape.scalar` until `evaluate` broadcasts it.

    Nulls follow the kernel and are not restated here: the result is null
    exactly where the list row is, and a null search value gives `FALSE`.
    """

    comptime NativeType = DType.int32
    """Sizes the bit-pack driver's lane. Neither operand's width is relevant —
    `lane` reads bits out of a `BoolArray` the kernel already produced."""

    comptime shape = Shape.columnar
    comptime Bound = BoolArray

    var list: Self.L
    var elem: Self.E

    def __init__(out self, var list: Self.L, var elem: Self.E):
        self.list = list^
        self.elem = elem^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return merged(self.list.columns(), self.elem.columns())

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        """Bind the list, materialise the search column, run the kernel.

        `as_primitive` needs no dtype guard here, unlike `_as_bool` one file
        over. That guard exists because a Kleene operand is bound on
        `ComptimeValue`, so an `int64` operand type-checks and is only wrong at
        run time. `E` is a `NumericValue`, and a `PrimitiveValue` evaluates to
        an array of its own `Type` by construction — a batch whose column
        disagrees fails earlier, in the leaf's own `bind`.
        """
        return ArrayContainsKernel.apply(
            self.list.bind(batch, bindings),
            self.elem.evaluate(batch, bindings)
            .to_array(len(batch))
            .as_primitive[Self.E.Type](),
        )

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("array_contains(", self.list, ", ", self.elem, ")")
