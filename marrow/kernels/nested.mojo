# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Nested (list/struct) compute kernels.

`ArrayLengthKernel` — element count per list → `Int32Array`, the list analogue of
`string.LengthKernel`: the count is `offsets[i+1] - offsets[i]`, a SIMD subtract
over the (fixed-width) offsets buffer, so `apply` runs a single vectorized pass.

`ArrayContainsKernel` — element-wise membership `elem[i] ∈ list[i]` → `BoolArray`.
The list is variable-width so it can't lane-fuse; each row scans its sublist for a
value equal to that row's search element (a constant broadcasts through the
expression layer). Null list rows propagate to null results.

`ListReduceKernel[F]` — a fold kernel over each list's elements, one value per
list (`list_sum`, `list_min`, …): the list's offsets are segments of its child,
and `Segmented[F].reduce` folds them.
"""

from std.sys import size_of
from std.sys.info import simd_byte_width
from std.algorithm.backend.vectorize import vectorize
from std.utils.index import IndexList

from ..arrays import (
    DynArray,
    ListLikeArray,
    Int32Array,
    BoolArray,
    PrimitiveArray,
)
from ..buffers import Buffer, Bitmap
from ..dtypes import (
    DType,
    IntegerType,
    ListLikeType,
    NumericType,
    PrimitiveType,
)
from ..views import apply, valid_or_zero
from .aggregate import FoldKernel, IntegerAgg
from .core import Kernel
from .segments import ByEnds, Segmented
from ..execution import ExecContext
from ..errors import InvalidError, TypeError


struct ArrayLengthKernel(Kernel):
    """Per-element element count of a list array → `Int32Array` (matches
    pyarrow's `list_value_length`).

    Vectorized: loads `W` contiguous offsets at `i` and at `i+1` and subtracts,
    so each SIMD step computes `W` lengths at once. Null input elements yield a
    null output element (the input's validity bitmap is propagated unchanged),
    matching `string.LengthKernel`.
    """

    comptime name = "array_length"

    @staticmethod
    def apply[T: ListLikeType](array: ListLikeArray[T]) raises -> Int32Array:
        comptime off = T.offset
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int32](n)
        var offs = array.offsets.view[off](array.offset)
        comptime width = simd_byte_width() // size_of[Scalar[off]]()

        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]) {mut out, imm}:
            var i = idx[0]
            out.view[DType.int32](i).store[W](
                0, (offs.load[W](i + 1) - offs.load[W](i)).cast[DType.int32]()
            )

        @always_inline
        def lane[
            W: Int
        ](i: Int) {imm fill,}:
            fill[W, rank=1](IndexList[1](i))

        vectorize[width](n, lane)

        # Propagate the input's validity: a null list yields a null length.
        return Int32Array(
            like=array,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        if not array.dtype().is_list_like():
            raise Self.error[TypeError](
                t"expected a list array, got {array.dtype()}"
            )

        def leaf[T: ListLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_list_like[T]()).to_dyn()

        return array.dtype().try_visit[DynArray, ListLikeType](
            leaf, "dtype is not listlike"
        )


struct ArrayContainsKernel(Kernel):
    """Element-wise list membership: `result[i]` is True iff the search value
    `elem[i]` appears among the (valid) elements of the sublist `list[i]`. Result
    is null exactly where the list row is null; a null / absent search value gives
    False. Numeric element types only."""

    comptime name = "array_contains"

    @staticmethod
    def apply[
        T: ListLikeType, V: NumericType
    ](
        list: ListLikeArray[T],
        elem: PrimitiveArray[V],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BoolArray:
        comptime off = T.offset
        var n = len(list)
        if len(elem) != n:
            raise InvalidError(
                t"array_contains: list and element arrays must have equal "
                t"length, got {n} and {len(elem)}"
            )
        if not list.values().dtype().isa[V]():
            # `dispatch` picks `V` from the *element* array and `T` from the
            # list's offset width; neither reads the list's child type, so a
            # `list<int32>` column with an `int64` search value arrives here
            # with `V = Int64Type`. `as_primitive[V]` on a mismatched child is
            # a bad variant access -- it aborts the process rather than
            # raising, so the check has to be here and not at the call site.
            raise Self.error[TypeError](
                (
                    t"list element type {list.values().dtype()} does not match"
                    t" the search value type {V().to_dyn()}"
                ),
            )
        ref child = list.values().as_primitive[V]()
        var offs = list.offsets.view[off](list.offset)
        var data = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if list.is_valid(i):
                if elem.is_valid(i):
                    var lo = Int(offs.load[1](i))
                    var hi = Int(offs.load[1](i + 1))
                    var target = elem.unsafe_get(i)
                    for j in range(lo, hi):
                        if child.is_valid(j) and child.unsafe_get(j) == target:
                            data.set(i)
                            break
        return BoolArray(like=list, buffer=data.to_immutable())

    @staticmethod
    def dispatch(
        list: DynArray,
        elem: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        if not list.dtype().is_list_like():
            raise Self.error[TypeError](
                t"expected a list array, got {list.dtype()}"
            )

        def leaf[V: NumericType](d: V) raises {imm} -> DynArray:
            # Two nested family walks: the element type picks `V`, the list
            # offset width picks `T`. Neither is a hand-written arm.
            def outer[T: ListLikeType](o: T) raises {imm} -> DynArray:
                return Self.apply(
                    list.as_list_like[T](), elem.as_primitive[V](), ctx
                ).to_dyn()

            return list.dtype().try_visit[DynArray, ListLikeType](
                outer, "dtype is not listlike"
            )

        return elem.dtype().try_visit[DynArray, NumericType](
            leaf, "dtype is not numeric"
        )


struct ListReduceKernel[F: FoldKernel](Kernel):
    """A fold over each list's elements, one value per list: `list_sum`,
    `list_min`, `list_max`, `list_mean` and `list_count`, with
    `F = SumFold`, `MinFold`, `MaxFold`, `MeanFold` and `CountFold`.

    The answer is `F`'s accumulator type (a sum is int64, uint64 or float64, a
    mean float64). Null elements are skipped; a list with no valid element
    answers NULL (`count` answers 0), and so does a null list. Numeric
    elements only, and integer elements for the bitwise folds: `list_bit_and`,
    `list_bit_or` and `list_bit_xor`, with `F = BitAndFold`, `BitOrFold` and
    `BitXorFold`, which keep the element type.
    """

    comptime name = "list_reduce"

    @staticmethod
    def apply[
        T: ListLikeType, //, V: PrimitiveType
    ](
        list: ListLikeArray[T], ctx: ExecContext = ExecContext.serial()
    ) raises -> PrimitiveArray[Self.F.AccType[V]]:
        if not list.values().dtype().isa[V]():
            # `as_primitive[V]` on a mismatched child is a bad variant access,
            # which aborts rather than raises.
            raise Self.error[TypeError](
                t"list element type {list.values().dtype()} is not the type"
                t" this kernel was resolved for"
            )
        ref child = list.values().as_primitive[V]()
        var n = len(list)
        var result: PrimitiveArray[Self.F.AccType[V]]
        if n == 0:
            # An empty list array may carry no offsets at all.
            result = PrimitiveArray[Self.F.AccType[V]].empty(
                Self.F.acc_dtype[V](child.dtype)
            )
        else:
            var ends = list.offsets.view[T.offset](list.offset, n + 1)
            var reduced = Segmented[Self.F].reduce(child, ByEnds(ends))
            if list.null_count() == 0:
                result = reduced^
            else:
                result = Self._null_lists(reduced, list)
        return result^

    @staticmethod
    def _null_lists[
        A: PrimitiveType, T: ListLikeType
    ](
        reduced: PrimitiveArray[A], list: ListLikeArray[T]
    ) raises -> PrimitiveArray[A]:
        """`reduced`, null where `list` is: a null list's offsets may still
        span elements. A null row's payload is zero."""
        comptime N = A.native
        var n = len(reduced)
        var bm = Bitmap.intersect_views(reduced.validity(), list.validity())
        var out = Buffer.alloc_uninit[N](n)
        apply[N, N, valid_or_zero[N, _]](
            reduced.values(), bm.value().view(), out.view[N](0, n)
        )
        return PrimitiveArray[A](
            reduced.dtype.copy(),
            length=n,
            nulls=bm.value().unset_count(),
            offset=0,
            bitmap=bm,
            buffer=out^.to_immutable(),
        )

    @staticmethod
    def dispatch(
        list: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> DynArray:
        if not list.dtype().is_list_like():
            raise Self.error[TypeError](
                t"expected a list array, got {list.dtype()}"
            )

        def outer[T: ListLikeType](o: T) raises {imm} -> DynArray:
            # The offset width picks `T`, the child's dtype `V`.
            def leaf[V: NumericType](d: V) raises {imm} -> DynArray:
                comptime if conforms_to(Self.F, IntegerAgg) and not conforms_to(
                    V, IntegerType
                ):
                    raise Self.error[TypeError](
                        t"list element type {d} is not an integer type"
                    )
                else:
                    return Self.apply[V](list.as_list_like[T](), ctx).to_dyn()

            return (
                list.as_list_like[T]()
                .values()
                .dtype()
                .try_visit[DynArray, NumericType](leaf, "dtype is not numeric")
            )

        return list.dtype().try_visit[DynArray, ListLikeType](
            outer, "dtype is not listlike"
        )
