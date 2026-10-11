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
from .cast import cast
from .core import Kernel
from ..views import apply
from ..execution import ExecContext
from ..errors import InternalError, TypeError
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
            length=n,
            nulls=array.null_count(),
            offset=0,
            bitmap=_passthrough_validity(array),
            buffer=out.to_immutable(),
        )

    @staticmethod
    def dispatch(array: DynArray) raises -> DynArray:
        # Guard before dispatching so the diagnostic names *this kernel* and the
        # family it wanted. `dispatch_stringlike` would otherwise fall through to
        # `_dispatch`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view()).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.dispatch_stringlike(leaf)


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
        # family it wanted. `dispatch_stringlike` would otherwise fall through to
        # `_dispatch`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view()).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.dispatch_stringlike(leaf)


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

        return dt.dispatch_stringlike(leaf)

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
            length=n,
            nulls=array.null_count(),
            offset=0,
            bitmap=_passthrough_validity(array),
            buffer=data.to_immutable(),
        )

    @staticmethod
    def dispatch(left: DynArray, right: DynArray) raises -> DynArray:
        Self.expect_same_dtype(left.dtype(), right.dtype())

        def leaf[A: BytesArray](l: A) raises {imm} -> DynArray:
            return Self.apply(l, right.as_type[A]()).to_dyn()

        return left.dispatch_bytes(leaf)


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


def _passthrough_validity[
    A: BytesArray
](array: A) raises -> Optional[Bitmap[mut=False]]:
    """The operand's validity, offset-applied and owned — what a unary result
    or a predicate against a constant returns, since a constant operand is
    never null."""
    var v = array.validity()
    if v:
        return v.value().to_owned()
    return None


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
            length=n,
            nulls=array.null_count(),
            offset=0,
            bitmap=_passthrough_validity(array),
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

        return dt.dispatch_stringlike(leaf)


comptime LikeKernel = MatchLikeKernel[False]
comptime ILikeKernel = MatchLikeKernel[True]


# ---------------------------------------------------------------------------
# Parameterised string kernels — the SQL function surface
# ---------------------------------------------------------------------------
#
# `substr`, `left`, `right`, `lpad`, `rpad`, `replace`, `split_part`,
# `trim(chars)`, `repeat`, `position`, `char_length`, `ascii`.
#
# **Each of these states its own signature.** `substr(s, start: Int, count:
# Int)`, `replace(s, needle: StringSlice, repl: StringSlice)`,
# `lpad(s, width: Int, fill: StringSlice)` — twelve kernels across eight
# signatures, which cannot honestly share one `transform(s, args)`.
#
# **A kernel owns its own loop.** `apply` reads the operand slots its
# signature names and walks the array itself, so the signature is stated once,
# where the kernel is. Only a shape with *several* kernels factors the loop
# out: `StringIntKernel` for `left`/`right`/`repeat` and `StringToIntKernel`
# for `char_length`/`ascii`, each a trait plus one free generic function. A
# shape with a single kernel gets no trait at all — it would be a bound with
# one conformer, used by a function with one call site.
#
# Two invariants are load-bearing. **The arguments are operands, not
# constants**, so `substr(s, start_col, len_col)` is as writable as
# `substr(s, 2, 3)`, and the whole family reaches the runtime lane. And **a
# string argument is borrowed, not copied** — it goes in as a `StringSlice`
# cut straight out of its column, because a per-row `String` copy of what is
# usually one repeated literal is the cost this arrangement exists to avoid.
#
# `StringOperands` is the *node's* carrier of those operand columns, filled
# once per batch; it never reaches `transform`. Each `apply` reads the slots
# its own signature names and passes them positionally.
#
# **Every one of these counts characters, not bytes**, which is the whole
# reason they are here: `LengthKernel` already answers the byte question and
# SQL asks the other one. `héllo wörld` is 11 characters and 13 bytes, so the
# two disagree on every row that leaves ASCII. The implementations are
# deliberately scalar loops over `Utf8.bounds` — this is coverage, not
# throughput, and a correct loop that matches DuckDB beats a vectorised one
# that nearly does.


struct StringOperands[
    T: StringLikeType = StringType,
    A: StringLikeType = StringType,
    S: PrimitiveType = Int64Type,
    C: PrimitiveType = Int64Type,
](Copyable, Movable, Writable):
    """The **columns** each argument of a SQL string function evaluated to.

    Four slots because that is how many distinct arguments the family has
    between them, not because any one kernel reads four: `substr` reads
    `start` and `count`, `replace` reads `text` and `alt`, `lpad` one of each,
    `char_length` none. An absent slot is `None`.

    This is the *node's* carrier — the thing a breaker's `bind` fills once per
    batch. It is not a per-row argument bundle: each kernel's `apply` reaches
    into exactly the slots its own signature names and hands them positionally
    to a `transform` that never sees this struct. A per-row argument bundle
    would force a `String(...)` heap copy for a `text` operand that is usually
    one literal repeated down the batch, and would let six different
    signatures pretend to be one.

    **Four type parameters, all defaulted, so no operand is ever cast.** The
    comptime lane knows each operand's type at compile time and fills the
    parameters from it; the runtime lane, which does not, normalises to the
    defaults with `cast` and instantiates this once. Neither pays for the
    other: an AOT binary gets exactly the widths it named, and the interpreter
    pays a cast it already pays on every arithmetic operand.
    """

    var text: Optional[BinaryLikeArray[Self.T]]
    """The needle, pattern, fill, separator or character set."""
    var alt: Optional[BinaryLikeArray[Self.A]]
    """The replacement — `replace` alone reads it."""
    var start: Optional[PrimitiveArray[Self.S]]
    """`substr`'s 1-based start position."""
    var count: Optional[PrimitiveArray[Self.C]]
    """A length, width, repeat count or field index."""

    def __init__(out self):
        """All four slots absent — what `char_length` and `ascii` take."""
        self.text = None
        self.alt = None
        self.start = None
        self.count = None

    def is_valid(self, i: Int) -> Bool:
        """Whether every present slot has a value at row `i`.

        **Null in any argument position means null out** — `substr('abc', NULL,
        2)`, `lpad('x', 5, NULL)` and `position(NULL IN 'abc')` are all NULL in
        DuckDB, checked against it directly. The constants this replaced could
        never be null, so the case did not exist before and no kernel body has
        to learn about it: `apply` skips the row exactly as it skips a null
        input string.

        Every *present* slot, not every slot the kernel reads: a caller that
        fills one the kernel ignores gets it counted here too, which is
        harmless because the node fills exactly the slots `uses_*` names.
        """
        var ok = True
        if self.text:
            ok = ok and self.text.value().is_valid(i)
        if self.alt:
            ok = ok and self.alt.value().is_valid(i)
        if self.start:
            ok = ok and self.start.value().is_valid(i)
        if self.count:
            ok = ok and self.count.value().is_valid(i)
        return ok

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "StringOperands(",
            "text" if self.text else "_",
            ", ",
            "alt" if self.alt else "_",
            ", ",
            "start" if self.start else "_",
            ", ",
            "count" if self.count else "_",
            ")",
        )


# -- string -> string ------------------------------------------------------
#
# Nine kernels over six signatures — `(str,int)`, `(str,str)`,
# `(str,int,int)`, `(str,int,str)`, `(str,str,str)`, `(str,str,int)` — really
# are six different functions, and one `transform(s, args)` covering all of
# them was six signatures pretending to be one.
#
# Only `(str,int)` has more than one kernel, so it is the only one with a
# trait: `StringIntKernel` plus the free `_map_int`, shared by `left`, `right`
# and `repeat`. The other five state their signature in `transform` and walk
# the array in their own `apply`.
#
# **Where a shape trait does exist it declares `transform` and nothing else.**
# It does not extend `StringArgKernel`, and it supplies no defaults. That is
# not a style choice — **a trait method whose default lives in a sub-trait
# crashes the compiler when it is called through the base.** Both callers hit
# it: a base-trait `dispatch` default whose closure calls `Self.apply`, and
# `StringFunction[K: StringArgKernel]` calling `Self.K.apply`. Exit 139, no
# source location, no note. `mojo precompile marrow` reports 0 errors / 0
# warnings throughout, because the crash needs the instantiation; `pytest
# golden` reports it as `exit code 0`, because `conftest`'s `libmarrow.so`
# build raises `_pytest.outcomes.Exit` before a single case runs. Bisected
# twice.
#
# So the arrangement is the one that was already proved safe: a shared loop
# lives in a free generic function, never in a sub-trait default. Every default
# a kernel inherits comes from `StringArgKernel` itself, never from a layer
# above it.
#
# A string argument arrives as a `StringSlice` borrowed straight out of its
# column: a bare `StringSlice` parameter takes its own inferred origin, so
# nothing is cast and nothing is copied. Naming the argument type on the trait
# instead — `comptime Arg: AnyType` — compiles, and then rejects every real
# call site; see the note on `StringIntKernel`.


trait StringIntKernel:
    """`string x int -> string` — `left`, `right`, `repeat`.

    Three conformers, which is the whole reason this shape has a trait at all:
    it is the only `string -> string` signature with more than one kernel, so
    it is the only one where a shared `_map_int` beats three copies of the
    loop. Every other shape states its signature in `transform` and walks the
    array in its own `apply`.

    **On why the argument is a written-out parameter and not an associated
    type.** One trait carrying `comptime Arg: AnyType` would have collapsed
    this shape and the `(str,str)` one into a single trait. It declares fine
    and even
    resolves through a generic — and then rejects every real call, because the
    argument always borrows from a column the caller owns:

        error: invalid call to 'drive': value passed to 'a' cannot be converted
        from 'StringSpan[origin_of(owned)]' to 'TakesSlice.Arg'

    A concrete origin does not widen to `ImmutAnyOrigin`, and closing that gap
    needs `unsafe_origin_cast`, which this codebase does not use. A written-out
    parameter takes its own inferred origin and costs nothing.
    """

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], count: Int) -> String:
        ...


trait StringArgKernel(Kernel):
    """What `StringFunction` needs of a kernel: columns in, a column out.

    Deliberately says nothing about arity — each kernel's own `transform` and
    `apply` state that. What it does carry is which operand slots the kernel
    reads, because the *node* needs that before it evaluates anything:
    `comptime if Self.K.uses_text` keeps an unused operand from being bound,
    materialised or printed.

    Every conformer implements `apply` itself — most by walking the array
    directly, the three `(str,int)` kernels by calling the shared `_map_int`.
    See the note above for why a shared loop cannot be a sub-trait default.
    """

    comptime uses_text: Bool
    """Whether `apply` reads `ops.text`."""
    comptime uses_alt: Bool
    """Whether `apply` reads `ops.alt`."""
    comptime uses_start: Bool
    """Whether `apply` reads `ops.start`."""
    comptime uses_count: Bool
    """Whether `apply` reads `ops.count`."""

    @staticmethod
    def apply[
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        """Null-propagating map over `array` and the slots this kernel reads."""
        ...

    @staticmethod
    def dispatch[
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: DynArray, ops: StringOperands[OT, OA, OS, OC]) raises -> DynArray:
        var dt = array.dtype()
        if dt.is_string_view():
            # Through the offsets layout and back: each of these functions
            # builds a new string per row anyway, so the conversion is one
            # more copy of bytes it was about to write.
            var out = Self.dispatch(cast(array, string.to_dyn()), ops)
            return cast(out, dt)
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T](), ops).to_dyn()

        return dt.dispatch_stringlike(leaf)


# -- the one shared loop ----------------------------------------------------
#
# `left`, `right` and `repeat` are the only kernels of one shape, so they are
# the only ones that factor their loop out. Every other kernel writes this
# same walk inline in its `apply`: read exactly the slots the signature names,
# raise rather than abort if the node failed to fill one, and skip a row whose
# input string or whose arguments are null. That null rule is the whole of
# `StringOperands.is_valid`, and no `transform` body has to know about it.


def _map_int[
    K: StringIntKernel,
    T: StringLikeType,
    OT: StringLikeType,
    OA: StringLikeType,
    OS: PrimitiveType,
    OC: PrimitiveType,
](
    array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
) raises -> BinaryLikeArray[T]:
    if not ops.count:
        raise InternalError("missing operand: count")
    ref counts = ops.count.value()
    var n = len(array)
    var builder = BinaryLikeBuilder[T](capacity=n)
    for i in range(n):
        if array.is_valid(i) and ops.is_valid(i):
            builder.append(
                K.transform(
                    array.unsafe_get(UInt(i)),
                    Int(counts.values().unsafe_get(i)),
                )
            )
        else:
            builder.append_null()
    return builder.finish()


struct SubstrKernel(StringArgKernel):
    """SQL `substr(s, start, count)` — **1-based**, counting characters.

    `start` below 1 is not clamped to 1: the window still ends at
    `start + count`, so the characters before the string are *consumed* by the
    count. `substr('abc', 0, 2)` is therefore `'a'` and not `'ab'` — Postgres
    and DuckDB agree, and clamping the start alone is the natural way to get it
    wrong.
    """

    comptime name = "substr"
    comptime uses_text = False
    comptime uses_alt = False
    comptime uses_start = True
    comptime uses_count = True

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
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        if not ops.start:
            raise InternalError("missing operand: start")
        if not ops.count:
            raise InternalError("missing operand: count")
        ref starts = ops.start.value()
        ref counts = ops.count.value()
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        Int(starts.values().unsafe_get(i)),
                        Int(counts.values().unsafe_get(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()


struct LeftKernel(StringArgKernel, StringIntKernel):
    """SQL `left(s, count)` — the first `count` characters.

    A negative `count` means "all but the last `|count|`", which is DuckDB's
    and Postgres's reading and not the "empty string" a clamp would give.
    """

    comptime name = "left"
    comptime uses_text = False
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = True

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

    @staticmethod
    def apply[
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        return _map_int[Self](array, ops)


struct RightKernel(StringArgKernel, StringIntKernel):
    """SQL `right(s, count)` — the last `count` characters.

    A negative `count` means "all but the first `|count|`", the mirror of
    `LeftKernel`.
    """

    comptime name = "right"
    comptime uses_text = False
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = True

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

    @staticmethod
    def apply[
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        return _map_int[Self](array, ops)


struct RepeatKernel(StringArgKernel, StringIntKernel):
    """SQL `repeat(s, n)` — `s` concatenated `n` times.

    Zero and negative counts both answer the empty string — DuckDB's reading;
    several engines raise on a negative instead.

    It shares `StringIntKernel` with `left` and `right` because it *is* that
    signature. It used to stand apart with its own `apply(array, counts)`,
    which was the only way to say "this argument is a column" while the rest of
    the family carried constants. Now that every argument is a column, the
    exception has nothing to be an exception to.
    """

    comptime name = "repeat"
    comptime uses_text = False
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = True

    @staticmethod
    def transform[
        o: Origin[mut=False]
    ](s: StringSlice[o], count: Int) -> String:
        var out = String()
        for _ in range(count):
            out += s
        return out^

    @staticmethod
    def apply[
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        return _map_int[Self](array, ops)


struct Pad[left: Bool](StringArgKernel):
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
    comptime uses_text = True
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = True

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
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        if not ops.count:
            raise InternalError("missing operand: count")
        if not ops.text:
            raise InternalError("missing operand: text")
        ref widths = ops.count.value()
        ref fills = ops.text.value()
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        Int(widths.values().unsafe_get(i)),
                        fills.unsafe_get(UInt(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()


comptime LPadKernel = Pad[True]
comptime RPadKernel = Pad[False]


struct ReplaceKernel(StringArgKernel):
    """SQL `replace(s, needle, repl)` — **every** occurrence, literally.

    An empty `needle` returns the input unchanged rather than interleaving the
    replacement between every character, which is both what DuckDB does and the
    only reading that terminates.
    """

    comptime name = "replace"
    comptime uses_text = True
    comptime uses_alt = True
    comptime uses_start = False
    comptime uses_count = False

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
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        if not ops.text:
            raise InternalError("missing operand: text")
        if not ops.alt:
            raise InternalError("missing operand: alt")
        ref needles = ops.text.value()
        ref repls = ops.alt.value()
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        needles.unsafe_get(UInt(i)),
                        repls.unsafe_get(UInt(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()


struct SplitPartKernel(StringArgKernel):
    """SQL `split_part(s, sep, index)` — the `index`-th field, **1-based**.

    An index past the last field answers the **empty string**, not null: that
    is DuckDB's choice and the one this kernel is written against. An index
    below 1 answers the empty string too, where DuckDB raises — a kernel in a
    SIMD-shaped engine has nowhere to put an error, and the empty string is the
    same answer it gives for every other out-of-range index.
    """

    comptime name = "split_part"
    comptime uses_text = True
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = True

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
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        if not ops.text:
            raise InternalError("missing operand: text")
        if not ops.count:
            raise InternalError("missing operand: count")
        ref seps = ops.text.value()
        ref indices = ops.count.value()
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)),
                        seps.unsafe_get(UInt(i)),
                        Int(indices.values().unsafe_get(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()


struct TrimCharsKernel(StringArgKernel):
    """SQL `trim(s, chars)` — strip any leading or trailing character that is a
    **member of the set** `chars`, not the literal substring `chars`.

    The set is compared code point by code point, so a multi-byte member never
    matches part of a multi-byte character. An empty set strips nothing.
    """

    comptime name = "trim_chars"
    comptime uses_text = True
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = False

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
        T: StringLikeType,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](
        array: BinaryLikeArray[T], ops: StringOperands[OT, OA, OS, OC]
    ) raises -> BinaryLikeArray[T]:
        if not ops.text:
            raise InternalError("missing operand: text")
        ref text = ops.text.value()
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                builder.append(
                    Self.transform(
                        array.unsafe_get(UInt(i)), text.unsafe_get(UInt(i))
                    )
                )
            else:
                builder.append_null()
        return builder.finish()


# -- string -> int64 -------------------------------------------------------
#
# The same arrangement at the other return type. `char_length` and `ascii`
# share the `(str) -> int64` shape, so it gets `StringToIntKernel` and the free
# `_measure_plain`; `position` is the only `(str,str) -> int64` kernel, so it
# states `measure` and walks the array in its own `apply`. Both routes end at
# `_measured`, which owns the null bookkeeping.
#
# `int64` rather than the `int32` `LengthKernel` answers with, because these
# are the SQL spellings and SQL's `length` / `position` / `ascii` are `BIGINT`.
# A null input measures null — never 0, which is the answer for the *empty*
# string and a different fact. A null **argument** measures null too:
# `position(NULL IN 'abc')` is NULL in DuckDB, not 0.


trait StringToIntKernel:
    """`string -> int64` — `char_length`, `ascii`.

    Two conformers, so the loop is worth sharing; see `StringIntKernel` for the
    same reasoning on the `string -> string` side."""

    @staticmethod
    def measure[o: Origin[mut=False]](s: StringSlice[o]) -> Int64:
        ...


trait StringMeasureKernel(Kernel):
    """What `StringMeasure` needs of a kernel: columns in, an `int64` column
    out. `StringArgKernel`, one return type over."""

    comptime uses_text: Bool
    """Whether `apply` reads `ops.text`."""

    @staticmethod
    def apply[
        A: BytesArray,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: A, ops: StringOperands[OT, OA, OS, OC]) raises -> Int64Array:
        ...

    @staticmethod
    def dispatch[
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: DynArray, ops: StringOperands[OT, OA, OS, OC]) raises -> DynArray:
        var dt = array.dtype()
        if dt.is_string_view():
            return Self.apply(array.as_string_view(), ops).to_dyn()
        if not dt.is_string_like():
            raise Self.error[TypeError](t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T](), ops).to_dyn()

        return dt.dispatch_stringlike(leaf)


def _measured(
    var values: Buffer[mut=True],
    var valid: Bitmap[mut=True],
    n: Int,
    nulls: Int,
) -> Int64Array:
    """The column both measure shapes finish with.

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


def _measure_plain[
    K: StringToIntKernel,
    A: BytesArray,
    OT: StringLikeType,
    OA: StringLikeType,
    OS: PrimitiveType,
    OC: PrimitiveType,
](array: A, ops: StringOperands[OT, OA, OS, OC]) raises -> Int64Array:
    var n = len(array)
    var out = Buffer.alloc_uninit[DType.int64](n)
    var dst = out.view[DType.int64](0, n)
    var valid = Bitmap.alloc_zeroed(n)
    var nulls = 0
    for i in range(n):
        if array.is_valid(i) and ops.is_valid(i):
            dst.unsafe_set(i, K.measure(array.unsafe_get(UInt(i))))
            valid.unsafe_set(i)
        else:
            dst.unsafe_set(i, Int64(0))
            nulls += 1
    return _measured(out^, valid^, n, nulls)


struct CharLengthKernel(StringMeasureKernel, StringToIntKernel):
    """SQL `length(s)` — **characters**, where `LengthKernel` counts bytes.

    Two different functions, not one with a bug: `héllo wörld` is 11 here and
    13 there, and DuckDB spells the pair `length` and `octet_length`.
    """

    comptime name = "char_length"
    comptime uses_text = False

    @staticmethod
    def measure[o: Origin[mut=False]](s: StringSlice[o]) -> Int64:
        return Int64(Utf8.count(s))

    @staticmethod
    def apply[
        A: BytesArray,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: A, ops: StringOperands[OT, OA, OS, OC]) raises -> Int64Array:
        return _measure_plain[Self](array, ops)


struct AsciiKernel(StringMeasureKernel, StringToIntKernel):
    """SQL `ascii(s)` — the first character's code point, **0** for the empty
    string.

    The smallest function here that has to *decode* UTF-8 rather than move
    bytes: `héllo wörld` answers 104 for `h`, and a string starting with `é`
    must answer 233 rather than the lead byte 0xC3.
    """

    comptime name = "ascii"
    comptime uses_text = False

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

    @staticmethod
    def apply[
        A: BytesArray,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: A, ops: StringOperands[OT, OA, OS, OC]) raises -> Int64Array:
        return _measure_plain[Self](array, ops)


struct PositionKernel(StringMeasureKernel):
    """SQL `position(needle IN s)` — the 1-based **character** index of the
    first occurrence, and **0** when there is none.

    Zero rather than null is what makes the result an index into a 1-based
    world; a 0-based engine cannot reuse the convention. An empty needle is
    found at position 1, which follows from `Utf8.find` matching it at the
    cursor. A *null* needle is a different matter and answers null — see
    `StringOperands.is_valid`.
    """

    comptime name = "position"
    comptime uses_text = True

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
        A: BytesArray,
        //,
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: A, ops: StringOperands[OT, OA, OS, OC]) raises -> Int64Array:
        if not ops.text:
            raise InternalError("missing operand: text")
        ref needles = ops.text.value()
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int64](n)
        var dst = out.view[DType.int64](0, n)
        var valid = Bitmap.alloc_zeroed(n)
        var nulls = 0
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                dst.unsafe_set(
                    i,
                    Self.measure(
                        array.unsafe_get(UInt(i)), needles.unsafe_get(UInt(i))
                    ),
                )
                valid.unsafe_set(i)
            else:
                dst.unsafe_set(i, Int64(0))
                nulls += 1
        return _measured(out^, valid^, n, nulls)
