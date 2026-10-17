# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""String compute kernels.

Three shapes, each following the tier scheme used by the numeric kernels
(`numeric.mojo`), adapted to variable-width UTF-8 data:

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
- **SQL pattern match → bool** (`MatchLikeKernel`) — the predicate shape
  above plus an array × scalar-pattern overload over `utils.LikePattern`,
  which compiles the pattern once instead of per row.

Variable-width ops cannot lane-fuse the way numeric kernels do (there is no
fixed W-wide lane), so the expression layer
(`marrow/expr/comptime/strings.mojo`) materializes them. Only `LengthKernel` exposes a fusable, offset-based fast path there.
"""

from std.sys import size_of
from std.sys.info import simd_byte_width
from std.utils.index import IndexList

from ..arrays import (
    DynArray,
    BinaryLikeArray,
    BinaryViewLikeArray,
    BytesArray,
    BoolArray,
    Int32Array,
    Int64Array,
    PrimitiveArray,
    StringArray,
)
from ..buffers import Buffer, Bitmap
from ..builders import BinaryLikeBuilder, BinaryViewLikeBuilder, BytesBuilder
from ..dtypes import (
    DType,
    Int64Type,
    PrimitiveType,
    StringLikeType,
    StringType,
    StringViewLikeType,
    string,
)
from .cast import CastKernel
from .core import Kernel
from ..views import apply
from ..execution import ExecContext
from ..errors import TypeError
from ..utils import LikePattern, Utf8


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
        vectors (`offsets[i+1] - offsets[i]`), used by `apply`. The expression
        layer's `StringLength` does *not* build on this — it is a breaker and
        calls `LengthKernel.dispatch`, materialising an `Int32Array`. Making it
        fuse through `core` is Q7.1."""
        return (hi - lo).cast[DType.int32]()

    @staticmethod
    def apply[
        T: StringLikeType
    ](
        array: BinaryLikeArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        comptime off = T.offset
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int32](n)
        var offs = array.offsets.view[off](array.offset)

        # `views.apply`, not stdlib `vectorize` directly. Kernels and the
        # expression layer go through the `apply` family: it owns the SIMD width,
        # the serial/parallel choice and the tail, so a kernel that reaches past
        # it opts out of all three and diverges from every other kernel here.
        @always_inline
        def producer[W: Int](i: Int) {imm} -> SIMD[DType.int32, W]:
            return Self.core(offs.load[W](i + 1), offs.load[W](i))

        apply[DType.int32](out.view[DType.int32](0, n), producer, ctx)

        # Propagate the input's validity: a null string yields a null length.
        return Int32Array(
            like=array,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def apply[
        T: StringViewLikeType
    ](
        array: BinaryViewLikeArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int32Array:
        """Byte length of each view, read from the view's length word alone
        -- no data buffer is touched."""
        _ = ctx
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int32](n)
        for i in range(n):
            out.unsafe_set[DType.int32](i, Int32(array.view_length(i)))
        return Int32Array(
            like=array,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        # Guard before dispatching so the diagnostic names *this kernel* and the
        # family it wanted. `try_visit[StringLikeType]` would otherwise fall through to
        # `visit`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view()).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.try_visit[DynArray, StringLikeType](
            leaf, "dtype is not stringlike"
        )


# ---------------------------------------------------------------------------
# Unary string → string kernels
# ---------------------------------------------------------------------------


trait StringMapKernel(Kernel):
    """Element-wise string → string op. Concrete kernels define `transform`;
    `apply` (element-wise build, null-preserving) and `dispatch` are defaulted.
    """

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        ...

    @staticmethod
    def apply[
        T: StringLikeType
    ](array: BinaryLikeArray[T]) raises -> BinaryLikeArray[T]:
        var builder = BinaryLikeBuilder[T](capacity=len(array))
        Self._map_into(array, builder)
        return builder.finish()

    @staticmethod
    def apply[
        T: StringViewLikeType
    ](array: BinaryViewLikeArray[T]) raises -> BinaryViewLikeArray[T]:
        """The same map over a view array; the output keeps the view layout."""
        var builder = BinaryViewLikeBuilder[T](capacity=len(array))
        Self._map_into(array, builder)
        return builder.finish()

    @staticmethod
    def _map_into[
        Arr: BytesArray, Bld: BytesBuilder
    ](array: Arr, mut builder: Bld) raises:
        """The null-preserving map loop, shared by both layouts' `apply`."""
        for i in range(len(array)):
            if array.is_valid(i):
                builder.append(Self.transform(array.unsafe_get(UInt(i))))
            else:
                builder.append_null()

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        # Guard before dispatching so the diagnostic names *this kernel* and the
        # family it wanted. `try_visit[StringLikeType]` would otherwise fall through to
        # `visit`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view()).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.try_visit[DynArray, StringLikeType](
            leaf, "dtype is not stringlike"
        )


struct UpperKernel(StringMapKernel):
    """Arrow's `utf8_upper`: each code point to exactly one, so `'ß'` gives
    `'ẞ'` where the full case mapping would give `"SS"`."""

    comptime name = "upper"

    @always_inline
    @staticmethod
    def core(cp: Codepoint) -> Codepoint:
        """One code point to one, as Arrow's `utf8_upper` maps it."""
        if cp.is_ascii():
            if cp.is_ascii_lower():
                return Codepoint(unsafe_unchecked_codepoint=cp.to_u32() - 32)
            return cp
        var full = String(cp).upper()
        if full.count_codepoints() == 1:
            return Codepoint.ord(full)
        return Codepoint(unsafe_unchecked_codepoint=Self.expanding(cp.to_u32()))

    @always_inline
    @staticmethod
    def expanding(cp: UInt32) -> UInt32:
        """The one-code-point uppercase of a code point whose full mapping
        expands.

        `'ß'` maps to `"SS"` and `'ᾳ'` to `"ΑΙ"` in full; Arrow's `utf8_upper`
        takes a single code point instead: `'ẞ'`, and the titlecase
        letter with the iota subscript kept. Every other expanding code point
        (`'ŉ'`, the `'ﬁ'` ligatures, ...) has none and stays as it is.
        """
        if cp == 0xDF:
            return 0x1E9E
        if (
            (cp >= 0x1F80 and cp <= 0x1F87)
            or (cp >= 0x1F90 and cp <= 0x1F97)
            or (cp >= 0x1FA0 and cp <= 0x1FA7)
        ):
            return cp + 8
        if cp == 0x1FB3 or cp == 0x1FC3 or cp == 0x1FF3:
            return cp + 9
        return cp

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        var out = String(capacity_bytes=s.byte_length())
        for cp in s.codepoints():
            out.append(Self.core(cp))
        return out^


struct LowerKernel(StringMapKernel):
    """Arrow's `utf8_lower`: each code point to exactly one, so `'İ'` gives
    `'i'`."""

    comptime name = "lower"

    @always_inline
    @staticmethod
    def core(cp: Codepoint) -> Codepoint:
        """One code point to one, as Arrow's `utf8_lower` maps it."""
        if cp.is_ascii():
            if cp.is_ascii_upper():
                return Codepoint(unsafe_unchecked_codepoint=cp.to_u32() + 32)
            return cp
        var full = String(cp).lower()
        if full.count_codepoints() == 1:
            return Codepoint.ord(full)
        return cp

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        # The stdlib's lowercase mapping is already one code point to one.
        return s.lower()


struct StripKernel(StringMapKernel):
    comptime name = "strip"

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        return String(s.strip())


struct LStripKernel(StringMapKernel):
    comptime name = "lstrip"

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        return String(s.lstrip())


struct RStripKernel(StringMapKernel):
    comptime name = "rstrip"

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        return String(s.rstrip())


struct ReverseKernel(StringMapKernel):
    comptime name = "reverse"

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        # Reverse by grapheme cluster so multi-byte characters stay intact.
        var out = String()
        for g in s.__reversed__():
            out += String(g)
        return out


struct CapitalizeKernel(StringMapKernel):
    comptime name = "capitalize"

    @staticmethod
    def transform[o: Origin[mut=False]](s: StringSlice[o]) -> String:
        # Arrow's `utf8_capitalize`: the first code point upper-cased, the rest
        # lower-cased, each to exactly one.
        var out = String(capacity_bytes=s.byte_length())
        var first = True
        for cp in s.codepoints():
            if first:
                out.append(UpperKernel.core(cp))
                first = False
            else:
                out.append(LowerKernel.core(cp))
        return out^


# ---------------------------------------------------------------------------
# Binary string → string (element-wise concatenation)
# ---------------------------------------------------------------------------


struct ConcatKernel(Kernel):
    """Element-wise binary string concatenation (`a || b`). `combine` is the
    fusable per-element primitive; `apply` materializes the whole array,
    null-propagating.

    **No expression node reaches this yet.** The docstring used to say the
    expression layer's `Concat` builds on it; there is no such node in either
    lane. Kept because a string-concat node is a real gap in the expression
    surface, not because anything currently calls it."""

    comptime name = "binary_join_element_wise"

    @staticmethod
    @always_inline
    def combine(a: String, b: String) -> String:
        return a + b

    @staticmethod
    def dispatch(left: DynArray, right: DynArray) raises -> DynArray:
        """Resolve two runtime string columns and concatenate them.

        The erased counterpart of `apply`. `DynValue.__add__` needs it: `+` over
        erased operands cannot know at build time whether it means addition or
        concatenation, so the choice is made on the runtime dtype."""
        var dt = left.dtype()
        if dt.is_string_view():
            Self.expect_same_dtype(dt, right.dtype())
            Self.expect_same_length(len(left), len(right))
            return Self.apply(
                left.as_string_view(), right.as_string_view()
            ).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")
        Self.expect_same_dtype(dt, right.dtype())
        Self.expect_same_length(len(left), len(right))

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                left.as_binary_like[T](), right.as_binary_like[T]()
            ).to_dyn()

        return dt.try_visit[DynArray, StringLikeType](
            leaf, "dtype is not stringlike"
        )

    @staticmethod
    def apply[
        L: StringLikeType, R: StringLikeType
    ](
        left: BinaryLikeArray[L], right: BinaryLikeArray[R]
    ) raises -> BinaryLikeArray[L]:
        """The output takes the **left** operand's type.

        Two type parameters rather than one, as `StringPredicateKernel.apply`
        already had: the expression node above binds `L.Type` and `R.Type`
        separately, so a single `T` would reject `string || large_string` at the
        call site rather than at a place that could say why.
        """
        var builder = BinaryLikeBuilder[L](capacity=len(left))
        Self._concat_into(left, right, builder)
        return builder.finish()

    @staticmethod
    def apply[
        L: StringViewLikeType, R: StringViewLikeType
    ](
        left: BinaryViewLikeArray[L], right: BinaryViewLikeArray[R]
    ) raises -> BinaryViewLikeArray[L]:
        """The same concatenation over view arrays, output in the view layout.
        """
        var builder = BinaryViewLikeBuilder[L](capacity=len(left))
        Self._concat_into(left, right, builder)
        return builder.finish()

    @staticmethod
    def _concat_into[
        L: BytesArray, R: BytesArray, B: BytesBuilder
    ](left: L, right: R, mut builder: B) raises:
        """The null-propagating concat loop, shared by both layouts' `apply`."""
        Self.expect_same_length(len(left), len(right))
        for i in range(len(left)):
            if left.is_valid(i) and right.is_valid(i):
                builder.append(
                    Self.combine(
                        String(left.unsafe_get(UInt(i))),
                        String(right.unsafe_get(UInt(i))),
                    )
                )
            else:
                builder.append_null()


# ---------------------------------------------------------------------------
# Binary string predicate → BoolArray
# ---------------------------------------------------------------------------


trait StringPredicateKernel(Kernel):
    """Element-wise `bytes × bytes → bool` predicate. Concrete kernels define
    `predicate`; `apply` (bit-packed, null-propagating) and `dispatch` default.

    Over text and binary alike, in any layout, as Arrow's comparisons,
    `starts_with`, `ends_with` and `match_substring` are. `LIKE` is the text
    operation of the family, and stays one through its own pattern
    `dispatch`, which `MatchLikeKernel` guards.
    """

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        ...

    @staticmethod
    def apply[
        L: BytesArray, R: BytesArray
    ](left: L, right: R) raises -> BoolArray:
        """Over either string layout, on either side."""
        Self.expect_same_length(len(left), len(right))
        var n = len(left)
        var bm = Bitmap.intersect_views(left.validity(), right.validity())
        var data = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if left.is_valid(i) and right.is_valid(i):
                if Self.predicate(
                    left.unsafe_get(UInt(i)), right.unsafe_get(UInt(i))
                ):
                    data.set(i)
        return BoolArray(
            length=n,
            nulls=bm.value().unset_count() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def apply_scalar[
        A: BytesArray
    ](array: A, pattern: StringSlice) raises -> BoolArray:
        """`array × one constant pattern`, without materialising the constant.

        The peer of `apply` for the case where the right operand is a scalar.
        `apply` needs a `BinaryLikeArray` on both sides, so the expression layer
        had to splat a literal into an n-row array first — n copies of the same
        string, allocated per morsel.

        The default body is the same loop as `apply` with the right operand
        hoisted. `LikeKernel`/`ILikeKernel` override it to compile their pattern
        once instead of per row.
        """
        var n = len(array)
        var data = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if array.is_valid(i) and Self.predicate(
                array.unsafe_get(UInt(i)), pattern
            ):
                data.set(i)
        # Validity is the left operand's: a constant right operand is never
        # null, so `Bitmap.intersect(l, None)` reduces to `l`.
        return BoolArray(
            like=array,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def dispatch(left: DynArray, right: DynArray) raises -> DynArray:
        Self.expect_same_dtype(left.dtype(), right.dtype())

        def leaf[A: BytesArray](l: A) raises {imm} -> DynArray:
            return Self.apply(l, right.get[A]()).to_dyn()

        return left.try_visit[DynArray, BytesArray](
            leaf, "expected a binary or string array"
        )


struct StartsWithKernel(StringPredicateKernel):
    comptime name = "startswith"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s.startswith(pat)


struct EndsWithKernel(StringPredicateKernel):
    comptime name = "endswith"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s.endswith(pat)


struct ContainsKernel(StringPredicateKernel):
    comptime name = "contains"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return pat in s


struct StringEqKernel(StringPredicateKernel):
    comptime name = "equal"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s == pat


struct StringNeKernel(StringPredicateKernel):
    comptime name = "not_equal"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s != pat


# Ordering comparisons — lexicographic over UTF-8 bytes, which equals codepoint
# order. These are the string half of `<` `<=` `>` `>=` `==` `!=`; the numeric
# half lives in `compare.mojo` as `NumericCompareKernel` conformers. The two are
# separate families because a variable-width predicate is elementwise and cannot
# vectorize, so whoever interprets the operator pairs them — the runtime lane
# in `marrow/expr/runtime/values.mojo`.


struct StringLtKernel(StringPredicateKernel):
    comptime name = "less"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s < pat


struct StringLeKernel(StringPredicateKernel):
    comptime name = "less_equal"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s <= pat


struct StringGtKernel(StringPredicateKernel):
    comptime name = "greater"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s > pat


struct StringGeKernel(StringPredicateKernel):
    comptime name = "greater_equal"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return not (s < pat)  # StringSlice has no __ge__(StringSlice) overload


# ---------------------------------------------------------------------------
# SQL LIKE / ILIKE pattern matching
# ---------------------------------------------------------------------------


struct MatchLikeKernel[ignore_case: Bool](StringPredicateKernel):
    """SQL ``LIKE`` (``pc.match_like``): ``%`` = any run, ``_`` = any single
    character, ``\\`` escapes, everything else literal. With `ignore_case` it
    is ``ILIKE`` (``ignore_case=True``): both operands are lower-cased before
    matching.

    A pattern is compiled once per scalar, and once per run of equal pattern
    texts in the array x array form: the runtime expression lane hands a
    literal pattern over as n identical rows, and those collapse to one
    compile."""

    comptime name = "match_like_ci" if Self.ignore_case else "match_like"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return LikePattern[Self.ignore_case](pat).matches(s)

    @staticmethod
    def apply[
        L: BytesArray, R: BytesArray
    ](left: L, right: R) raises -> BoolArray:
        # Overrides the trait default, which would compile the pattern per row.
        Self.expect_same_length(len(left), len(right))
        var n = len(left)
        var bm = Bitmap.intersect_views(left.validity(), right.validity())
        var data = Bitmap.alloc_zeroed(n)
        var compiled = LikePattern[Self.ignore_case]("")
        var current = String()
        var primed = False
        for i in range(n):
            if left.is_valid(i) and right.is_valid(i):
                var pat = right.unsafe_get(UInt(i))
                if not primed or StringSlice(current) != pat:
                    compiled = LikePattern[Self.ignore_case](pat)
                    current = String(pat)
                    primed = True
                if compiled.matches(left.unsafe_get(UInt(i))):
                    data.set(i)
        return BoolArray(
            length=n,
            nulls=bm.value().unset_count() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def apply[
        A: BytesArray
    ](array: A, pattern: StringSlice) raises -> BoolArray:
        var compiled = LikePattern[Self.ignore_case](pattern)
        var n = len(array)
        var data = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if array.is_valid(i) and compiled.matches(
                array.unsafe_get(UInt(i))
            ):
                data.set(i)
        # Null input yields a null output element.
        return BoolArray(
            like=array,
            buffer=data.to_immutable(),
        )

    @staticmethod
    def apply_scalar[
        A: BytesArray
    ](array: A, pattern: StringSlice) raises -> BoolArray:
        return Self.apply(array, pattern)

    @staticmethod
    def dispatch(array: DynArray, pattern: StringSlice) raises -> DynArray:
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view(), pattern).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T](), pattern).to_dyn()

        return dt.try_visit[DynArray, StringLikeType](
            leaf, "dtype is not stringlike"
        )


comptime LikeKernel = MatchLikeKernel[False]
comptime ILikeKernel = MatchLikeKernel[True]


# ---------------------------------------------------------------------------
# Parameterised string kernels — the SQL function surface
# ---------------------------------------------------------------------------
#
# `substr`, `left`, `right`, `lpad`, `rpad`, `replace`, `split_part`,
# `trim(chars)`, `repeat`, `position`, `char_length`, `ascii`.
#
# Each kernel states its own signature: a per-row `transform` (or `measure`)
# over borrowed `StringSlice`s and `Int`s, a typed `apply` over the argument
# columns, and an erased `dispatch` over the subject. The arguments are
# columns, so `substr(s, start_col, len_col)` is as writable as
# `substr(s, 2, 3)`, and **null in any argument means null out**, as in
# DuckDB. A shape with several kernels shares its loop as a trait default:
# `StringIntKernel` for `left`/`right`/`repeat`, `StringToIntKernel` for
# `char_length`/`ascii`.
#
# **Every one of these counts characters, not bytes**: `LengthKernel` answers
# the byte question and SQL asks the other one.


def dispatch_text[
    K: Kernel,
    Func: def[T: StringLikeType](BinaryLikeArray[T]) raises -> DynArray,
](array: DynArray, func: Func) raises -> DynArray:
    """Run `func` over a text subject of any layout. A string view goes
    through the offsets layout and back: each of these functions builds a new
    string per row anyway, so the conversion is one more copy of bytes it was
    about to write."""
    var dt = array.dtype()
    if dt.is_string_view():
        var flat = CastKernel.dispatch(array, string.to_dyn())
        return CastKernel.dispatch(func(flat.as_string()), dt)
    if not dt.is_string_like():
        raise K.error[TypeError](t"expected a string array, got {dt}")

    def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
        return func(array.as_binary_like[T]())

    return dt.try_visit[DynArray, StringLikeType](
        leaf, "dtype is not stringlike"
    )


# -- string -> string ------------------------------------------------------


trait StringTextTextKernel(Kernel):
    """`string x string x string -> string` — `replace`, `regexp_replace`."""

    @staticmethod
    def apply[
        T: StringLikeType, X: StringLikeType, Y: StringLikeType
    ](
        array: BinaryLikeArray[T], x: BinaryLikeArray[X], y: BinaryLikeArray[Y]
    ) raises -> BinaryLikeArray[T]:
        ...


trait StringTextIntKernel(Kernel):
    """`string x string x int -> string` — `split_part`, `regexp_extract`."""

    @staticmethod
    def apply[
        T: StringLikeType, X: StringLikeType, C: PrimitiveType
    ](
        array: BinaryLikeArray[T], x: BinaryLikeArray[X], c: PrimitiveArray[C]
    ) raises -> BinaryLikeArray[T]:
        ...


trait StringIntKernel(Kernel):
    """`string x int -> string` — `left`, `right`, `repeat`."""

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], count: Int) -> String:
        ...

    @staticmethod
    def apply[
        T: StringLikeType, C: PrimitiveType
    ](
        array: BinaryLikeArray[T], count: PrimitiveArray[C]
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and count.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        Int(count.values().unsafe_get(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(array: DynArray, count: Int64Array) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, count).to_dyn()

        return dispatch_text[Self](array, run)


struct SubstrKernel(Kernel):
    """SQL `substr(s, start, count)` — **1-based**, counting characters.

    `start` below 1 is not clamped to 1: the window still ends at
    `start + count`, so the characters before the string are *consumed* by the
    count. `substr('abc', 0, 2)` is therefore `'a'` and not `'ab'` — Postgres
    and DuckDB agree, and clamping the start alone is the natural way to get it
    wrong.
    """

    comptime name = "substr"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], start: Int, count: Int) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var first = start - 1  # 0-based, may be negative
        var last = first + count  # exclusive
        if first < 0:
            first = 0
        if last < 0:
            last = 0
        if first > chars:
            first = chars
        if last > chars:
            last = chars
        return Utf8.slice(s, bounds[first], bounds[last])

    @staticmethod
    def apply[
        T: StringLikeType, S: PrimitiveType, C: PrimitiveType
    ](
        array: BinaryLikeArray[T],
        start: PrimitiveArray[S],
        count: PrimitiveArray[C],
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and start.is_valid(i) and count.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        Int(start.values().unsafe_get(i)),
                        Int(count.values().unsafe_get(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(
        array: DynArray, start: Int64Array, count: Int64Array
    ) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, start, count).to_dyn()

        return dispatch_text[Self](array, run)


struct LeftKernel(StringIntKernel):
    """SQL `left(s, count)` — the first `count` characters.

    A negative `count` means "all but the last `|count|`", which is DuckDB's
    and Postgres's reading and not the "empty string" a clamp would give.
    """

    comptime name = "left"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], count: Int) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var take = count if count >= 0 else chars + count
        if take < 0:
            take = 0
        if take > chars:
            take = chars
        return Utf8.slice(s, bounds[0], bounds[take])


struct RightKernel(StringIntKernel):
    """SQL `right(s, count)` — the last `count` characters.

    A negative `count` means "all but the first `|count|`", the mirror of
    `LeftKernel`.
    """

    comptime name = "right"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], count: Int) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var skip = chars - count if count >= 0 else -count
        if skip < 0:
            skip = 0
        if skip > chars:
            skip = chars
        return Utf8.slice(s, bounds[skip], bounds[chars])


struct RepeatKernel(StringIntKernel):
    """SQL `repeat(s, n)` — `s` concatenated `n` times.

    Zero and negative counts both answer the empty string — DuckDB's reading;
    several engines raise on a negative instead.
    """

    comptime name = "repeat"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], count: Int) -> String:
        var out = String()
        for _ in range(count):
            out += s
        return out^


struct Pad[left: Bool](Kernel):
    """SQL `lpad`/`rpad(s, width, fill)` — pad to `width` characters.

    **The side is a comptime parameter, not a flag.** `_pad(s, args, left)`
    tested a `Bool` argument on every row to choose between two concatenation
    orders, though it cannot vary within a call. This is the shape
    `LikePattern[ignore_case]` already uses to let LIKE and ILIKE share one
    implementation, and the branch is gone at monomorphisation.

    **Padding is also truncation**: a string longer than the width is cut
    rather than left alone, which is the half of `lpad` an implementation that
    only ever appends will miss. The fill cycles when it is more than one
    character, and an **empty fill cannot pad**, so the input comes back
    (truncated, if it was too long) instead of looping forever.
    """

    comptime name = "lpad" if Self.left else "rpad"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], width: Int, fill: StringSlice) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var w = width
        if w < 0:
            w = 0
        if chars >= w:
            # Truncation keeps the *first* `width` characters whichever end the
            # padding would have gone on — `rpad` does not truncate from the
            # left.
            return Utf8.slice(s, bounds[0], bounds[w])
        var fb = Utf8.bounds(fill)
        var fchars = len(fb) - 1
        if fchars == 0:
            return String(s)  # nothing to pad with
        var pad = String()
        var k = 0
        while k < w - chars:
            pad += Utf8.slice(fill, fb[k % fchars], fb[k % fchars + 1])
            k += 1
        comptime if Self.left:
            return pad + String(s)
        else:
            return String(s) + pad

    @staticmethod
    def apply[
        T: StringLikeType, W: PrimitiveType, F: StringLikeType
    ](
        array: BinaryLikeArray[T],
        width: PrimitiveArray[W],
        fill: BinaryLikeArray[F],
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and width.is_valid(i) and fill.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        Int(width.values().unsafe_get(i)),
                        fill.unsafe_get(UInt(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(
        array: DynArray, width: Int64Array, fill: StringArray
    ) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, width, fill).to_dyn()

        return dispatch_text[Self](array, run)


comptime LPadKernel = Pad[True]
comptime RPadKernel = Pad[False]


struct ReplaceKernel(StringTextTextKernel):
    """SQL `replace(s, needle, repl)` — **every** occurrence, literally.

    An empty `needle` returns the input unchanged rather than interleaving the
    replacement between every character, which is both what DuckDB does and the
    only reading that terminates.
    """

    comptime name = "replace"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], needle: StringSlice, repl: StringSlice) -> String:
        var m = len(needle.as_bytes())
        if m == 0:
            return String(s)
        var out = String()
        var n = len(s.as_bytes())
        var i = 0
        while i <= n:
            var j = Utf8.find(s, needle, i)
            if j < 0:
                break
            out += Utf8.slice(s, i, j)
            out += repl
            i = j + m
        out += Utf8.slice(s, i, n)
        return out^

    @staticmethod
    def apply[
        T: StringLikeType, N: StringLikeType, R: StringLikeType
    ](
        array: BinaryLikeArray[T],
        needle: BinaryLikeArray[N],
        repl: BinaryLikeArray[R],
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and needle.is_valid(i) and repl.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        needle.unsafe_get(UInt(i)),
                        repl.unsafe_get(UInt(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(
        array: DynArray, needle: StringArray, repl: StringArray
    ) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, needle, repl).to_dyn()

        return dispatch_text[Self](array, run)


struct SplitPartKernel(StringTextIntKernel):
    """SQL `split_part(s, sep, index)` — the `index`-th field, **1-based**.

    An index past the last field answers the **empty string**, not null: that
    is DuckDB's choice and the one this kernel is written against. An index
    below 1 answers the empty string too, where DuckDB raises — a kernel in a
    SIMD-shaped engine has nowhere to put an error, and the empty string is the
    same answer it gives for every other out-of-range index.
    """

    comptime name = "split_part"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], sep: StringSlice, index: Int) -> String:
        if index < 1:
            return String()
        var m = len(sep.as_bytes())
        var n = len(s.as_bytes())
        if m == 0:
            # No separator, so the whole string is the one and only field.
            return String(s) if index == 1 else String()
        var start = 0
        var part = 1
        while part < index:
            var j = Utf8.find(s, sep, start)
            if j < 0:
                return String()  # fewer fields than asked for
            start = j + m
            part += 1
        var j = Utf8.find(s, sep, start)
        return Utf8.slice(s, start, j if j >= 0 else n)

    @staticmethod
    def apply[
        T: StringLikeType, P: StringLikeType, I: PrimitiveType
    ](
        array: BinaryLikeArray[T],
        sep: BinaryLikeArray[P],
        index: PrimitiveArray[I],
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and sep.is_valid(i) and index.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        sep.unsafe_get(UInt(i)),
                        Int(index.values().unsafe_get(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(
        array: DynArray, sep: StringArray, index: Int64Array
    ) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, sep, index).to_dyn()

        return dispatch_text[Self](array, run)


struct TrimCharsKernel(Kernel):
    """SQL `trim(s, chars)` — strip any leading or trailing character that is a
    **member of the set** `chars`, not the literal substring `chars`.

    The set is compared code point by code point, so a multi-byte member never
    matches part of a multi-byte character. An empty set strips nothing.
    """

    comptime name = "trim_chars"

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], text: StringSlice) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var lo = 0
        while lo < chars and Utf8.charset_has(
            text, s, bounds[lo], bounds[lo + 1]
        ):
            lo += 1
        var hi = chars
        while hi > lo and Utf8.charset_has(text, s, bounds[hi - 1], bounds[hi]):
            hi -= 1
        return Utf8.slice(s, bounds[lo], bounds[hi])

    @staticmethod
    def apply[
        T: StringLikeType, C: StringLikeType
    ](
        array: BinaryLikeArray[T], chars: BinaryLikeArray[C]
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and chars.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)), chars.unsafe_get(UInt(i))
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(array: DynArray, chars: StringArray) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, chars).to_dyn()

        return dispatch_text[Self](array, run)


# -- string -> int64 -------------------------------------------------------
#
# `int64` rather than the `int32` `LengthKernel` answers with, because these
# are the SQL spellings and SQL's `length` / `position` / `ascii` are `BIGINT`.
# A null input measures null — never 0, which is the answer for the *empty*
# string and a different fact. These read the view layout directly.


def _measured(
    var values: Buffer[mut=True],
    var valid: Bitmap[mut=True],
    n: Int,
    nulls: Int,
) -> Int64Array:
    """The column every measure finishes with.

    No bitmap at all when nothing is null: an all-ones bitmap is the same
    information, but `PrimitiveArray.__eq__` compares validity views and would
    read it as different from a column that never had one.
    """
    var bitmap = Optional[Bitmap[mut=False]](None)
    if nulls > 0:
        bitmap = valid^.to_immutable()
    return Int64Array(
        length=n,
        nulls=nulls,
        offset=0,
        bitmap=bitmap^,
        buffer=values^.to_immutable(),
    )


trait StringToIntKernel(Kernel):
    """`string -> int64` — `char_length`, `ascii`."""

    @staticmethod
    def measure[o: Origin[mut=False]](s: StringSlice[o]) -> Int64:
        ...

    @staticmethod
    def apply[A: BytesArray](array: A) raises -> Int64Array:
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int64](n)
        var dst = out.view[DType.int64](0, n)
        var valid = Bitmap.alloc_zeroed(n)
        var nulls = 0
        for i in range(n):
            if array.is_valid(i):
                dst.unsafe_set(i, Self.measure(array.unsafe_get(UInt(i))))
                valid.unsafe_set(i)
            else:
                dst.unsafe_set(i, Int64(0))
                nulls += 1
        return _measured(out^, valid^, n, nulls)

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view()).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.try_visit[DynArray, StringLikeType](
            leaf, "dtype is not stringlike"
        )


struct CharLengthKernel(StringToIntKernel):
    """SQL `length(s)` — **characters**, where `LengthKernel` counts bytes.

    Two different functions, not one with a bug: `héllo wörld` is 11 here and
    13 there, and DuckDB spells the pair `length` and `octet_length`.
    """

    comptime name = "char_length"

    @staticmethod
    def measure[o: Origin[mut=False]](s: StringSlice[o]) -> Int64:
        return Int64(Utf8.count(s))


struct AsciiKernel(StringToIntKernel):
    """SQL `ascii(s)` — the first character's code point, **0** for the empty
    string.

    The smallest function here that has to *decode* UTF-8 rather than move
    bytes: `héllo wörld` answers 104 for `h`, and a string starting with `é`
    must answer 233 rather than the lead byte 0xC3.
    """

    comptime name = "ascii"

    @staticmethod
    def measure[o: Origin[mut=False]](s: StringSlice[o]) -> Int64:
        var bytes = s.as_bytes()
        var n = len(bytes)
        if n == 0:
            return Int64(0)
        var w = Utf8.width(bytes[0])
        if w == 1 or n < w:
            return Int64(Int(bytes[0]))
        elif w == 2:
            return Int64(((Int(bytes[0]) & 0x1F) << 6) | (Int(bytes[1]) & 0x3F))
        elif w == 3:
            return Int64(
                ((Int(bytes[0]) & 0x0F) << 12)
                | ((Int(bytes[1]) & 0x3F) << 6)
                | (Int(bytes[2]) & 0x3F)
            )
        else:
            return Int64(
                ((Int(bytes[0]) & 0x07) << 18)
                | ((Int(bytes[1]) & 0x3F) << 12)
                | ((Int(bytes[2]) & 0x3F) << 6)
                | (Int(bytes[3]) & 0x3F)
            )


struct PositionKernel(Kernel):
    """SQL `position(needle IN s)` — the 1-based **character** index of the
    first occurrence, and **0** when there is none.

    Zero rather than null is what makes the result an index into a 1-based
    world; a 0-based engine cannot reuse the convention. An empty needle is
    found at position 1, which follows from `Utf8.find` matching it at the
    cursor. A *null* needle is a different matter and answers null.
    """

    comptime name = "position"

    @staticmethod
    def measure[
        o: Origin[mut=False]
    ](s: StringSlice[o], needle: StringSlice) -> Int64:
        var at = Utf8.find(s, needle, 0)
        if at < 0:
            return Int64(0)
        # Byte offset -> 1-based character index.
        var bounds = Utf8.bounds(s)
        for k in range(len(bounds) - 1):
            if bounds[k] == at:
                return Int64(k + 1)
        return Int64(len(bounds))

    @staticmethod
    def apply[
        A: BytesArray, N: StringLikeType
    ](array: A, needle: BinaryLikeArray[N]) raises -> Int64Array:
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int64](n)
        var dst = out.view[DType.int64](0, n)
        var valid = Bitmap.alloc_zeroed(n)
        var nulls = 0
        for i in range(n):
            if array.is_valid(i) and needle.is_valid(i):
                dst.unsafe_set(
                    i,
                    Self.measure(
                        array.unsafe_get(UInt(i)), needle.unsafe_get(UInt(i))
                    ),
                )
                valid.unsafe_set(i)
            else:
                dst.unsafe_set(i, Int64(0))
                nulls += 1
        return _measured(out^, valid^, n, nulls)

    @staticmethod
    def dispatch(array: DynArray, needle: StringArray) raises -> DynArray:
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view(), needle).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T](), needle).to_dyn()

        return dt.try_visit[DynArray, StringLikeType](
            leaf, "dtype is not stringlike"
        )
