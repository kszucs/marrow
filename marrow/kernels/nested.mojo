"""Nested (list/struct) compute kernels.

`ArrayLengthKernel` — element count per list → `Int32Array`, the list analogue of
`string.LengthKernel`: the count is `offsets[i+1] - offsets[i]`, a SIMD subtract
over the (fixed-width) offsets buffer, so `apply` runs a single vectorized pass.

`ArrayContainsKernel` is still a name-only marker (compute is TODO) — it exists so
the typed expression layer (`marrow.expr.values`) can name the operation.
"""

from std.sys import size_of
from std.sys.info import simd_byte_width
from std.algorithm.backend.vectorize import vectorize
from std.utils.index import IndexList

from ..arrays import AnyArray, ListLikeArray, Int32Array
from ..buffers import Buffer
from ..dtypes import ListLikeType, DType
from .helpers import Kernel


struct ArrayLengthKernel(Kernel):
    """Per-element element count of a list array → `Int32Array` (matches
    pyarrow's `list_value_length`).

    Vectorized: loads `W` contiguous offsets at `i` and at `i+1` and subtracts,
    so each SIMD step computes `W` lengths at once. Null positions yield length 0
    with an all-valid result bitmap (matches `string.LengthKernel`); full null
    propagation is a follow-up.
    """

    comptime name = "array_length"

    @staticmethod
    def apply[T: ListLikeType](array: ListLikeArray[T]) raises -> Int32Array:
        comptime off = T.offset
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int32](n)
        var offs = array.offsets.view[off](array.offset)
        comptime width = simd_byte_width() // size_of[Scalar[off]]()

        @parameter
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
        return Int32Array(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def dispatch(array: AnyArray) raises -> AnyArray:
        if array.dtype().is_list():
            return Self.apply(array.as_list()).to_any()
        elif array.dtype().is_large_list():
            return Self.apply(array.as_large_list()).to_any()
        else:
            raise Error(
                t"array_length: expected a list array, got {array.dtype()}"
            )


struct ArrayContainsKernel(Kernel):
    comptime name = "array_contains"
