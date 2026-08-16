"""Nested (list/struct) compute kernels.

`ArrayLengthKernel` — element count per list → `Int32Array`, the list analogue of
`string.LengthKernel`: the count is `offsets[i+1] - offsets[i]`, a SIMD subtract
over the (fixed-width) offsets buffer, so `apply` runs a single vectorized pass.

`ArrayContainsKernel` — element-wise membership `elem[i] ∈ list[i]` → `BoolArray`.
The list is variable-width so it can't lane-fuse; each row scans its sublist for a
value equal to that row's search element (a constant broadcasts through the
expression layer). Null list rows propagate to null results.
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
from ..dtypes import ListLikeType, NumericType, DType
from .core import Kernel
from ..execution import ExecContext


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

        @__parameter
        @always_inline
        def fill[W: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            out.view[DType.int32](i).store[W](
                0, (offs.load[W](i + 1) - offs.load[W](i)).cast[DType.int32]()
            )

        @always_inline
        def lane[W: Int](i: Int):
            fill[W, rank=1](IndexList[1](i))

        vectorize[width](n, lane)

        # Propagate the input's validity: a null list yields a null length.
        var vbm: Optional[Bitmap[mut=False]] = None
        var validity = array.validity()
        if validity:
            vbm = validity.value().union(validity.value()).to_immutable()
        return Int32Array(
            length=n,
            nulls=array.null_count(),
            offset=0,
            bitmap=vbm^,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        var dt = array.dtype()
        if not dt.is_list_like():
            raise Self.error(t"expected a list array, got {dt}")

        @__parameter
        def leaf[T: ListLikeType](d: T) raises -> DynArray:
            return Self.apply(array.as_list_like[T]()).to_dyn()

        return dt.dispatch_listlike[leaf]()


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
            raise Error(
                t"array_contains: list and element arrays must have equal"
                t" length, got {n} and {len(elem)}"
            )
        ref child = list.values().as_primitive[V]()
        var offs = list.offsets.view[off](list.offset)
        var data = Bitmap.alloc_zeroed(n)
        var valid = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if list.is_valid(i):
                valid.set(i)
                if elem.is_valid(i):
                    var lo = Int(offs.load[1](i))
                    var hi = Int(offs.load[1](i + 1))
                    var target = elem.unsafe_get(i)
                    for j in range(lo, hi):
                        if child.is_valid(j) and child.unsafe_get(j) == target:
                            data.set(i)
                            break
        var nulls = list.null_count()
        var bm: Optional[Bitmap[]] = None
        if nulls > 0:
            bm = valid.to_immutable()
        return BoolArray(
            length=n,
            nulls=nulls,
            offset=0,
            bitmap=bm^,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def dispatch(
        list: DynArray,
        elem: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var list_dt = list.dtype()
        if not list_dt.is_list_like():
            raise Self.error(t"expected a list array, got {list_dt}")

        @__parameter
        def leaf[V: NumericType](d: V) raises -> DynArray:
            # Two nested family walks: the element type picks `V`, the list
            # offset width picks `T`. Neither is a hand-written arm.
            @__parameter
            def outer[T: ListLikeType](o: T) raises -> DynArray:
                return Self.apply(
                    list.as_list_like[T](), elem.as_primitive[V](), ctx
                ).to_dyn()

            return list_dt.dispatch_listlike[outer]()

        return elem.dtype().dispatch_numeric[leaf]()
