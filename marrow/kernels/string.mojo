"""String compute kernels.

Three shapes, each following the tier scheme used by the numeric kernels
(`arithmetic.mojo`), adapted to variable-width UTF-8 data:

- **Length** (`LengthKernel`) — byte length per element → `Int32Array`. The one
  string op that vectorizes cleanly: byte length is `offsets[i+1] - offsets[i]`,
  a SIMD subtract over the (fixed-width) offsets buffer, so `apply` runs a single
  vectorized pass.
- **Unary string → string** (`Upper`, `Lower`, `Reverse`, `Strip`, `LStrip`,
  `RStrip`, `Capitalize`) — variable output width, so they build a fresh
  `StringArray` element-wise. Concrete kernels only define `transform`.
- **Binary string predicate → bool** (`StartsWith`, `EndsWith`, `Contains`) —
  compare each element against a pattern element, producing a bit-packed
  `BoolArray`. Concrete kernels only define `predicate`.

Variable-width ops cannot lane-fuse the way numeric kernels do (there is no
fixed W-wide lane), so the expression layer (`marrow.expr.values`) materializes
them. Only `LengthKernel` exposes a fusable, offset-based fast path there.
"""

from std.sys import size_of
from std.sys.info import simd_byte_width
from std.algorithm.backend.vectorize import vectorize
from std.utils.index import IndexList

from ..arrays import (
    AnyArray,
    BinaryLikeArray,
    BoolArray,
    Int32Array,
)
from ..buffers import Buffer, Bitmap
from ..builders import BinaryLikeBuilder
from ..dtypes import StringLikeType, DType
from .helpers import Kernel, bitmap_and


# ---------------------------------------------------------------------------
# Length — byte length per element (vectorized offset subtraction)
# ---------------------------------------------------------------------------


struct LengthKernel(Kernel):
    """Per-element byte length of a string array → `Int32Array` (matches
    pyarrow's `utf8_length`).

    Vectorized: loads `W` contiguous offsets at `i` and at `i+1` and subtracts,
    so each SIMD step computes `W` lengths at once. Null positions currently
    yield length 0 with an all-valid result bitmap (matches the prior
    `string_lengths` behaviour); full null propagation is a follow-up.
    """

    comptime name = "length"

    @staticmethod
    def apply[
        T: StringLikeType
    ](array: BinaryLikeArray[T]) raises -> Int32Array:
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
        if array.dtype().is_string():
            return Self.apply(array.as_string()).to_any()
        elif array.dtype().is_large_string():
            return Self.apply(array.as_large_string()).to_any()
        else:
            raise Error(t"length: expected a string array, got {array.dtype()}")


# ---------------------------------------------------------------------------
# Unary string → string kernels
# ---------------------------------------------------------------------------


trait StringUnaryKernel(Kernel):
    """Element-wise string → string op. Concrete kernels define `transform`;
    `apply` (element-wise build, null-preserving) and `dispatch` are defaulted.
    """

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        ...

    @staticmethod
    def apply[
        T: StringLikeType
    ](array: BinaryLikeArray[T]) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i):
                builder.append(Self.transform(array.unsafe_get(UInt(i))))
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(array: AnyArray) raises -> AnyArray:
        if array.dtype().is_string():
            return Self.apply(array.as_string()).to_any()
        elif array.dtype().is_large_string():
            return Self.apply(array.as_large_string()).to_any()
        else:
            raise Error(
                t"{Self.name}: expected a string array, got {array.dtype()}"
            )


struct UpperKernel(StringUnaryKernel):
    comptime name = "upper"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return s.upper()


struct LowerKernel(StringUnaryKernel):
    comptime name = "lower"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return s.lower()


struct StripKernel(StringUnaryKernel):
    comptime name = "strip"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return String(s.strip())


struct LStripKernel(StringUnaryKernel):
    comptime name = "lstrip"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return String(s.lstrip())


struct RStripKernel(StringUnaryKernel):
    comptime name = "rstrip"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return String(s.rstrip())


struct ReverseKernel(StringUnaryKernel):
    comptime name = "reverse"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        # Reverse by grapheme cluster so multi-byte characters stay intact.
        var out = String()
        for g in s.__reversed__():
            out += String(g)
        return out


struct CapitalizeKernel(StringUnaryKernel):
    comptime name = "capitalize"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        # First grapheme upper-cased, the rest lower-cased (pyarrow semantics).
        var out = String()
        var first = True
        for g in s:
            if first:
                out += String(g).upper()
                first = False
            else:
                out += String(g).lower()
        return out


# ---------------------------------------------------------------------------
# Binary string predicate → BoolArray
# ---------------------------------------------------------------------------


trait StringBinaryPredicateKernel(Kernel):
    """Element-wise `string × string → bool` predicate. Concrete kernels define
    `predicate`; `apply` (bit-packed, null-propagating) and `dispatch` default.
    """

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        ...

    @staticmethod
    def apply[
        T: StringLikeType
    ](left: BinaryLikeArray[T], right: BinaryLikeArray[T]) raises -> BoolArray:
        var n = len(left)
        if len(right) != n:
            raise Error(
                t"{Self.name}: arrays must have the same length, got {n} and"
                t" {len(right)}"
            )
        var bm = bitmap_and(left.bitmap, right.bitmap)
        var data = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if left.is_valid(i) and right.is_valid(i):
                if Self.predicate(
                    left.unsafe_get(UInt(i)), right.unsafe_get(UInt(i))
                ):
                    data.set(i)
        return BoolArray(
            length=n,
            nulls=n - bm.value().view().count_set_bits() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def dispatch(left: AnyArray, right: AnyArray) raises -> AnyArray:
        if left.dtype().is_string() and right.dtype().is_string():
            return Self.apply(left.as_string(), right.as_string()).to_any()
        elif left.dtype().is_large_string() and right.dtype().is_large_string():
            return Self.apply(
                left.as_large_string(), right.as_large_string()
            ).to_any()
        else:
            raise Error(
                t"{Self.name}: expected string arrays, got {left.dtype()} and"
                t" {right.dtype()}"
            )


struct StartsWithKernel(StringBinaryPredicateKernel):
    comptime name = "startswith"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s.startswith(pat)


struct EndsWithKernel(StringBinaryPredicateKernel):
    comptime name = "endswith"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s.endswith(pat)


struct ContainsKernel(StringBinaryPredicateKernel):
    comptime name = "contains"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return pat in s
