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
    so each SIMD step computes `W` lengths at once. Null input elements yield a
    null output element (the input's validity bitmap is propagated unchanged).
    """

    comptime name = "length"

    @staticmethod
    @always_inline
    def core[
        T: DType, W: Int
    ](hi: SIMD[T, W], lo: SIMD[T, W]) -> SIMD[DType.int32, W]:
        """The fusable per-lane primitive: byte length from two loaded offset
        vectors (`offsets[i+1] - offsets[i]`). Both `apply` and the expression
        layer's `StringLength` build on it, so the compute lives here only."""
        return (hi - lo).cast[DType.int32]()

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
                0, Self.core(offs.load[W](i + 1), offs.load[W](i))
            )

        @always_inline
        def lane[W: Int](i: Int):
            fill[W, rank=1](IndexList[1](i))

        vectorize[width](n, lane)

        # Propagate the input's validity: a null string yields a null length.
        var vbm: Optional[Bitmap[mut=False]] = None
        if array.bitmap:
            var v = array.bitmap.value().view(array.offset, n)
            vbm = v.union(v).to_immutable()
        return Int32Array(
            length=n,
            nulls=array.null_count(),
            offset=0,
            bitmap=vbm^,
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


trait StringMapKernel(Kernel):
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


struct UpperKernel(StringMapKernel):
    comptime name = "upper"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return s.upper()


struct LowerKernel(StringMapKernel):
    comptime name = "lower"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return s.lower()


struct StripKernel(StringMapKernel):
    comptime name = "strip"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return String(s.strip())


struct LStripKernel(StringMapKernel):
    comptime name = "lstrip"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return String(s.lstrip())


struct RStripKernel(StringMapKernel):
    comptime name = "rstrip"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        return String(s.rstrip())


struct ReverseKernel(StringMapKernel):
    comptime name = "reverse"

    @staticmethod
    def transform[o: Origin](s: StringSlice[o]) -> String:
        # Reverse by grapheme cluster so multi-byte characters stay intact.
        var out = String()
        for g in s.__reversed__():
            out += String(g)
        return out


struct CapitalizeKernel(StringMapKernel):
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
# Binary string → string (element-wise concatenation)
# ---------------------------------------------------------------------------


struct ConcatKernel(Kernel):
    """Element-wise binary string concatenation (`a || b`). `combine` is the fusable
    per-element primitive (the expression layer's `Concat` builds on it); `apply`
    materializes the whole array, null-propagating."""

    comptime name = "binary_join_element_wise"

    @staticmethod
    @always_inline
    def combine(a: String, b: String) -> String:
        return a + b

    @staticmethod
    def apply[
        T: StringLikeType
    ](
        left: BinaryLikeArray[T], right: BinaryLikeArray[T]
    ) raises -> BinaryLikeArray[T]:
        var n = len(left)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if left.is_valid(i) and right.is_valid(i):
                builder.append(
                    Self.combine(
                        String(left.unsafe_get(UInt(i))),
                        String(right.unsafe_get(UInt(i))),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()


# ---------------------------------------------------------------------------
# Binary string predicate → BoolArray
# ---------------------------------------------------------------------------


trait StringPredicateKernel(Kernel):
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
        L: StringLikeType, R: StringLikeType
    ](left: BinaryLikeArray[L], right: BinaryLikeArray[R]) raises -> BoolArray:
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


struct StartsWithKernel(StringPredicateKernel):
    comptime name = "startswith"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s.startswith(pat)


struct EndsWithKernel(StringPredicateKernel):
    comptime name = "endswith"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s.endswith(pat)


struct ContainsKernel(StringPredicateKernel):
    comptime name = "contains"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return pat in s


struct StringEqKernel(StringPredicateKernel):
    comptime name = "equal"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s == pat


struct StringNeKernel(StringPredicateKernel):
    comptime name = "not_equal"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s != pat


# ---------------------------------------------------------------------------
# SQL LIKE / ILIKE pattern matching
# ---------------------------------------------------------------------------


# Token sentinels for a compiled LIKE pattern. Literal code points are stored
# as their (non-negative) value; the two wildcards use negative sentinels.
comptime _LIKE_ANY = -1  # '%' — any run of characters (incl. empty)
comptime _LIKE_ONE = -2  # '_' — exactly one character


def _codepoints[o: Origin](s: StringSlice[o]) -> List[Int]:
    var out = List[Int]()
    for cp in s.codepoints():
        out.append(Int(cp))
    return out^


def _compile_like[o: Origin](pat: StringSlice[o]) -> List[Int]:
    """Compile a SQL ``LIKE`` pattern into a token list.

    ``%`` → ``_LIKE_ANY``, ``_`` → ``_LIKE_ONE``, and ``\\`` escapes the next
    character to a literal (``\\%`` → literal ``%``, ``\\\\`` → literal ``\\``);
    a trailing ``\\`` is dropped. This matches pyarrow's ``match_like``.
    """
    comptime BSL = 0x5C  # ord('\\')
    comptime PCT = 0x25  # ord('%')
    comptime UND = 0x5F  # ord('_')
    var cps = _codepoints(pat)
    var m = len(cps)
    var out = List[Int]()
    var k = 0
    while k < m:
        var c = cps[k]
        if c == BSL:
            if k + 1 < m:
                out.append(cps[k + 1])  # escaped literal
                k += 2
            else:
                k += 1  # trailing backslash: drop
        elif c == PCT:
            out.append(_LIKE_ANY)
            k += 1
        elif c == UND:
            out.append(_LIKE_ONE)
            k += 1
        else:
            out.append(c)
            k += 1
    return out^


def _wildcard_match(text: List[Int], toks: List[Int]) -> Bool:
    """Greedy SQL ``LIKE`` matcher over a text codepoint list and a compiled
    token list (see ``_compile_like``).

    O(len(text) * len(toks)) worst case, O(1) extra space via the classic
    backtracking-on-star algorithm.
    """
    var n = len(text)
    var m = len(toks)
    var i = 0  # cursor in text
    var j = 0  # cursor in tokens
    var star_j = -1  # token index of the last '%' seen, or -1
    var star_i = 0  # text index matched against that '%'
    while i < n:
        if j < m and (toks[j] == _LIKE_ONE or toks[j] == text[i]):
            i += 1
            j += 1
        elif j < m and toks[j] == _LIKE_ANY:
            star_j = j
            star_i = i
            j += 1
        elif star_j != -1:
            # backtrack: let the last '%' absorb one more character
            j = star_j + 1
            star_i += 1
            i = star_i
        else:
            return False
    # trailing '%' tokens can match the empty remainder
    while j < m and toks[j] == _LIKE_ANY:
        j += 1
    return j == m


struct LikeKernel(StringPredicateKernel):
    """SQL ``LIKE`` (``pc.match_like``): ``%`` = any run, ``_`` = any single
    character, ``\\`` escapes, everything else literal, case-sensitive."""

    comptime name = "match_like"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return _wildcard_match(_codepoints(s), _compile_like(pat))


struct ILikeKernel(StringPredicateKernel):
    """Case-insensitive SQL ``LIKE`` (``pc.match_like`` with
    ``ignore_case=True``): both operands are lower-cased before matching."""

    comptime name = "match_like_ci"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        var sl = s.lower()
        var pl = pat.lower()
        return _wildcard_match(
            _codepoints(StringSlice(sl)), _compile_like(StringSlice(pl))
        )
