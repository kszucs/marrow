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
- **SQL pattern match → bool** (`Like`, `ILike`) — the predicate shape above
  plus an array × scalar-pattern overload built on `LikePattern`, which
  compiles the pattern once instead of per row.

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
    BoolArray,
    Int32Array,
    Int64Array,
    PrimitiveArray,
)
from ..buffers import Buffer, Bitmap
from ..builders import BinaryLikeBuilder
from ..dtypes import (
    DType,
    Int64Type,
    PrimitiveType,
    StringLikeType,
    StringType,
)
from .core import Kernel
from ..views import apply
from ..execution import ExecContext


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
    def dispatch(array: DynArray) raises -> DynArray:
        # Guard before dispatching so the diagnostic names *this kernel* and the
        # family it wanted. `dispatch_stringlike` would otherwise fall through to
        # `_dispatch`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if not dt.is_string_like():
            raise Self.error(t"expected a string array, got {dt}")

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
    def dispatch(array: DynArray) raises -> DynArray:
        # Guard before dispatching so the diagnostic names *this kernel* and the
        # family it wanted. `dispatch_stringlike` would otherwise fall through to
        # `_dispatch`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if not dt.is_string_like():
            raise Self.error(t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.dispatch_stringlike(leaf)


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
        if not dt.is_string_like():
            raise Self.error(t"expected a string array, got {dt}")
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
        Self.expect_same_length(len(left), len(right))
        var n = len(left)
        var builder = BinaryLikeBuilder[L](capacity=n)
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
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
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
            bitmap=_passthrough_validity(array, n),
            buffer=data.to_immutable(),
        )

    @staticmethod
    def dispatch(left: DynArray, right: DynArray) raises -> DynArray:
        Self.expect_same_dtype(left.dtype(), right.dtype())
        var dt = left.dtype()
        if not dt.is_string_like():
            raise Self.error(t"expected string arrays, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                left.as_binary_like[T](), right.as_binary_like[T]()
            ).to_dyn()

        return dt.dispatch_stringlike(leaf)


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
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s < pat


struct StringLeKernel(StringPredicateKernel):
    comptime name = "less_equal"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s <= pat


struct StringGtKernel(StringPredicateKernel):
    comptime name = "greater"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return s > pat


struct StringGeKernel(StringPredicateKernel):
    comptime name = "greater_equal"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return not (s < pat)  # StringSlice has no __ge__(StringSlice) overload


# ---------------------------------------------------------------------------
# SQL LIKE / ILIKE pattern matching
# ---------------------------------------------------------------------------


# Token sentinels for a compiled LIKE pattern. Non-negative tokens are literal
# *bytes* of the UTF-8 encoded pattern; the two wildcards use negative
# sentinels. Byte-wise literals let the matcher run straight over a row's
# `as_bytes()` span with no per-row allocation: a multi-byte character expands
# to consecutive byte tokens, so code-point boundaries stay aligned whenever a
# literal run matches.
comptime _LIKE_ANY = -1  # '%' — any run of characters (incl. empty)
comptime _LIKE_ONE = -2  # '_' — exactly one character

# Pattern shapes recognised at compile time. The four literal shapes cover the
# patterns that dominate real workloads (cf. arrow-rs `Predicate::like`) and
# reduce matching to an optimized substring primitive.
comptime _LIKE_GENERAL = 0  # anything else — backtracking token match
comptime _LIKE_EXACT = 1  # 'foo'  → ==
comptime _LIKE_PREFIX = 2  # 'foo%' → startswith
comptime _LIKE_SUFFIX = 3  # '%foo' → endswith
comptime _LIKE_CONTAINS = 4  # '%foo%' → in

comptime _PCT = UInt8(0x25)  # '%'
comptime _UND = UInt8(0x5F)  # '_'
comptime _BSL = UInt8(0x5C)  # '\'


struct Utf8:
    """The character-vs-byte machinery this file runs on.

    Six primitives that were six free functions with no shared home, and
    between them roughly thirty call sites. Grouping them names the layer: the
    SQL surface (`substr`, `left`, `lpad`, `trim`, `position`, `length`) counts
    **code points**, `LengthKernel` counts bytes, and every one of the
    character answers is an index clamp into `bounds` rather than a decode.

    `bounds` is the one primitive the rest are built on: the byte offset of
    every code point start plus the end offset, so a string of `c` characters
    yields `c + 1` entries.

    Nothing here changed shape when it moved -- same signatures, same bodies.
    """

    @staticmethod
    @always_inline
    def width(lead: UInt8) -> Int:
        """Byte width of the UTF-8 sequence starting at this lead byte."""
        if lead < 0x80:
            return 1
        elif lead < 0xE0:
            return 2
        elif lead < 0xF0:
            return 3
        else:
            return 4

    @staticmethod
    def bounds(s: StringSlice) -> List[Int]:
        """Byte offset of each code point in `s`, terminated by `len(s)`."""
        var bytes = s.as_bytes()
        var n = len(bytes)
        var out = List[Int]()
        var i = 0
        while i < n:
            out.append(i)
            i += Self.width(bytes[i])
        out.append(n)
        return out^

    @staticmethod
    def count(s: StringSlice) -> Int:
        """Number of UTF-8 code points in `s` — SQL's `length`."""
        return len(Self.bounds(s)) - 1

    @staticmethod
    def slice(s: StringSlice, start: Int, end: Int) -> String:
        """Bytes `[start, end)` of `s`, copied into a fresh `String`.

        Both ends are assumed to sit on code point boundaries; every caller
        derives them from `bounds`, so a multi-byte character is never split.
        """
        var bytes = s.as_bytes()
        if end <= start:
            return String()
        var out = List[UInt8](capacity=end - start)
        for i in range(start, end):
            out.append(bytes[i])
        return String(from_utf8_lossy=Span(out))

    @staticmethod
    def find(s: StringSlice, needle: StringSlice, var from_byte: Int) -> Int:
        """Byte index of the first `needle` at or after `from_byte`, or -1.

        The naive O(n*m) scan. `LikePattern` has the optimised shapes for the
        pattern case; the literal searches here run once per row over short
        strings and a smarter algorithm would be unverified complexity for no
        measured win.

        An **empty needle matches at `from_byte`**, which is what makes
        `position('' IN 'abc')` answer 1 rather than 0 — the SQL convention,
        and the trap a "found at index -1" implementation falls into.
        """
        var hay = s.as_bytes()
        var pat = needle.as_bytes()
        var n = len(hay)
        var m = len(pat)
        if from_byte < 0:
            from_byte = 0
        if m == 0:
            return from_byte if from_byte <= n else -1
        var i = from_byte
        while i + m <= n:
            var j = 0
            while j < m and hay[i + j] == pat[j]:
                j += 1
            if j == m:
                return i
            i += 1
        return -1

    @staticmethod
    def charset_has(chars: StringSlice, s: StringSlice, a: Int, b: Int) -> Bool:
        """Is the code point occupying bytes `[a, b)` of `s` a member of
        `chars`?

        A *set* of characters, not a substring — `trim('  Ab  ', ' Ab')` strips
        each of the three independently and answers the empty string. Comparing
        whole code points rather than bytes is what keeps a multi-byte member
        from matching half of a multi-byte character.
        """
        var cb = chars.as_bytes()
        var sb = s.as_bytes()
        var w = b - a
        var i = 0
        while i < len(cb):
            var cw = Self.width(cb[i])
            if cw == w and i + cw <= len(cb):
                var same = True
                for k in range(w):
                    if cb[i + k] != sb[a + k]:
                        same = False
                        break
                if same:
                    return True
            i += cw
        return False


# How one ILIKE row must be case-folded, decided by a single byte scan. Unicode
# `lower()` costs microseconds per row, so it is worth avoiding for the common
# ASCII row: on pure-ASCII input an ASCII fold is *identical* to the Unicode one
# (no ASCII character folds outside ASCII), and an already-lower row needs none.
comptime _FOLD_NONE = 0  # pure ASCII, no upper case — match as-is
comptime _FOLD_ASCII = 1  # pure ASCII with upper case — cheap byte fold
comptime _FOLD_UNICODE = 2  # non-ASCII — fall back to `lower()`


def _fold_kind(s: StringSlice) -> Int:
    """Classify how `s` has to be case-folded for an ILIKE comparison."""
    var bytes = s.as_bytes()
    var kind = _FOLD_NONE
    for i in range(len(bytes)):
        var b = bytes[i]
        if b >= 0x80:
            return _FOLD_UNICODE
        elif b >= 0x41 and b <= 0x5A:
            kind = _FOLD_ASCII
    return kind


def _ascii_lower(s: StringSlice) -> String:
    """An ASCII-lowercased copy of a pure-ASCII `s` (see `_fold_kind`)."""
    var bytes = s.as_bytes()
    var out = List[UInt8](capacity=len(bytes))
    for i in range(len(bytes)):
        var b = bytes[i]
        out.append(b + 32 if b >= 0x41 and b <= 0x5A else b)
    return String(from_utf8_lossy=Span(out))


def _match_tokens(tokens: List[Int], text: StringSlice) -> Bool:
    """Greedy SQL ``LIKE`` matcher over a row's UTF-8 bytes and a compiled
    token list.

    O(len(text) * len(tokens)) worst case, O(1) extra space via the classic
    backtracking-on-star algorithm. Literal tokens advance one byte at a time
    while ``_`` and ``%`` advance whole code points, so the text cursor is
    always on a code-point boundary when a wildcard is consumed.
    """
    var bytes = text.as_bytes()
    var n = len(bytes)
    var m = len(tokens)
    var i = 0  # byte cursor in text
    var j = 0  # token cursor
    var star_j = -1  # token index of the last '%' seen, or -1
    var star_i = 0  # byte index matched against that '%'
    while i < n:
        if j < m and tokens[j] == Int(bytes[i]):
            i += 1
            j += 1
        elif j < m and tokens[j] == _LIKE_ONE:
            i += Utf8.width(bytes[i])
            j += 1
        elif j < m and tokens[j] == _LIKE_ANY:
            star_j = j
            star_i = i
            j += 1
        elif star_j != -1:
            # backtrack: let the last '%' absorb one more character
            j = star_j + 1
            star_i += Utf8.width(bytes[star_i])
            i = star_i
        else:
            return False
    # trailing '%' tokens can match the empty remainder
    while j < m and tokens[j] == _LIKE_ANY:
        j += 1
    return j == m


struct LikePattern[ignore_case: Bool = False](Copyable, Movable):
    """A SQL ``LIKE`` pattern compiled once and matched against many rows.

    ``%`` matches any run of characters, ``_`` exactly one, ``\\`` escapes the
    next character to a literal (``\\%`` → literal ``%``), and a trailing
    ``\\`` is dropped — pyarrow's ``match_like`` semantics.

    Compilation classifies the pattern into one of the four literal shapes
    (``foo``, ``foo%``, ``%foo``, ``%foo%``), which match via the optimized
    string primitives, or falls back to a wildcard token stream. With
    ``ignore_case`` the pattern is lower-cased once here and each row is
    case-folded before matching (see `_fold_kind`), which is the only thing
    that makes ``ILIKE`` differ from ``LIKE``.
    """

    var kind: Int
    """One of the `_LIKE_*` shape constants."""

    var literal: String
    """Literal characters of the pattern, case-folded when `ignore_case`.

    Meaningful for every shape but `_LIKE_GENERAL`.
    """

    var tokens: List[Int]
    """Byte/wildcard token stream driving the `_LIKE_GENERAL` matcher."""

    def __init__(out self, pattern: StringSlice):
        self.kind = _LIKE_GENERAL
        self.literal = String()
        self.tokens = List[Int]()

        comptime if Self.ignore_case:
            var folded = pattern.lower()
            self._compile(StringSlice(folded))
        else:
            self._compile(pattern)

    def _compile(mut self, pattern: StringSlice):
        """Tokenize `pattern` and classify it into a `_LIKE_*` shape."""
        var bytes = pattern.as_bytes()
        var n = len(bytes)
        var literal = List[UInt8](capacity=n)
        var leading_any = False  # pattern starts with '%'
        var trailing_any = False  # pattern ends with '%'
        var inner_wild = False  # any other '%' or any '_'
        var i = 0
        while i < n:
            var b = bytes[i]
            if b == _BSL:
                # '\' escapes the next character; a trailing '\' is dropped.
                if i + 1 < n:
                    var w = Utf8.width(bytes[i + 1])
                    for k in range(i + 1, i + 1 + w):
                        self.tokens.append(Int(bytes[k]))
                        literal.append(bytes[k])
                    i += 1 + w
                else:
                    i += 1
            elif b == _PCT:
                if len(self.tokens) == 0:
                    leading_any = True
                elif i + 1 == n:
                    trailing_any = True
                else:
                    inner_wild = True
                self.tokens.append(_LIKE_ANY)
                i += 1
            elif b == _UND:
                inner_wild = True
                self.tokens.append(_LIKE_ONE)
                i += 1
            else:
                var w = Utf8.width(b)
                for k in range(i, i + w):
                    self.tokens.append(Int(bytes[k]))
                    literal.append(bytes[k])
                i += w

        if inner_wild:
            self.kind = _LIKE_GENERAL
        elif leading_any and trailing_any:
            self.kind = _LIKE_CONTAINS
        elif leading_any:
            self.kind = _LIKE_SUFFIX
        elif trailing_any:
            self.kind = _LIKE_PREFIX
        else:
            self.kind = _LIKE_EXACT
        self.literal = String(from_utf8_lossy=Span(literal))

    def matches(self, s: StringSlice) -> Bool:
        """Return True if `s` matches this pattern."""

        comptime if Self.ignore_case:
            var fold = _fold_kind(s)
            if fold == _FOLD_NONE:
                return self._matches_folded(s)
            elif fold == _FOLD_ASCII:
                var folded = _ascii_lower(s)
                return self._matches_folded(StringSlice(folded))
            else:
                var folded = s.lower()
                return self._matches_folded(StringSlice(folded))
        else:
            return self._matches_folded(s)

    def _matches_folded(self, s: StringSlice) -> Bool:
        """Match `s`, which the caller has already case-folded if needed."""
        var lit = StringSlice(self.literal)
        if self.kind == _LIKE_EXACT:
            return s == lit
        elif self.kind == _LIKE_PREFIX:
            return s.startswith(lit)
        elif self.kind == _LIKE_SUFFIX:
            return s.endswith(lit)
        elif self.kind == _LIKE_CONTAINS:
            return lit in s
        else:
            return _match_tokens(self.tokens, s)


def _passthrough_validity[
    T: StringLikeType
](array: BinaryLikeArray[T], n: Int) raises -> Optional[Bitmap[mut=False]]:
    """The left operand's validity, offset-applied — what a predicate against a
    constant returns, since a constant operand is never null."""
    if array.bitmap:
        return array.bitmap.value().view(array.offset, n).to_owned()
    return None


def _match_pattern[
    T: StringLikeType, ignore_case: Bool
](
    array: BinaryLikeArray[T], pattern: LikePattern[ignore_case]
) raises -> BoolArray:
    """Evaluate an already-compiled pattern over every element of `array`.

    The pattern is compiled by the caller, so this is O(rows × pattern) work
    at worst and O(rows) for the literal shapes — versus recompiling the
    pattern per row through the array × array `predicate` path.
    """
    var n = len(array)
    var data = Bitmap.alloc_zeroed(n)
    for i in range(n):
        if array.is_valid(i) and pattern.matches(array.unsafe_get(UInt(i))):
            data.set(i)
    # Null input yields a null output element.
    return BoolArray(
        length=n,
        nulls=array.null_count(),
        offset=0,
        bitmap=_passthrough_validity(array, n),
        buffer=data.to_immutable(),
    )


def _match_arrays[
    L: StringLikeType, R: StringLikeType, ignore_case: Bool
](left: BinaryLikeArray[L], right: BinaryLikeArray[R]) raises -> BoolArray:
    """Array x array ``LIKE``, compiling the right operand once per *run* of
    equal patterns instead of once per row.

    The inherited `StringPredicateKernel.apply` calls `predicate` per row, and
    `LikeKernel.predicate` has to compile its pattern before it can match --
    so an array x array LIKE rebuilt the whole `LikePattern` (a token list, a
    literal buffer and a `String`) for every element. That is the shape the
    runtime expression lane produces: it evaluates a literal by
    `DynScalar.repeat(num_rows)`, so a constant pattern arrives as n identical
    rows and every one of those n compiles was redundant.

    Remembering the last pattern text collapses the constant case to a single
    compile without special-casing it: a genuinely varying right operand still
    recompiles, just only when the text actually changes, and the memo costs
    one comparison against a string that is already in cache.
    """
    var n = len(left)
    var bm = Bitmap.intersect_views(left.validity(), right.validity())
    var data = Bitmap.alloc_zeroed(n)
    var compiled = LikePattern[ignore_case]("")
    var current = String()
    var primed = False
    for i in range(n):
        if left.is_valid(i) and right.is_valid(i):
            var pat = right.unsafe_get(UInt(i))
            if not primed or StringSlice(current) != pat:
                compiled = LikePattern[ignore_case](pat)
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


def _dispatch_pattern[
    ignore_case: Bool
](name: StringSlice, array: DynArray, pattern: StringSlice) raises -> DynArray:
    """Type-erased entry point for the scalar-pattern overloads."""
    var compiled = LikePattern[ignore_case](pattern)
    var dt = array.dtype()
    if not dt.is_string_like():
        raise Error(t"{name}: expected a string array, got {dt}")

    def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
        return _match_pattern(array.as_binary_like[T](), compiled).to_dyn()

    return dt.dispatch_stringlike(leaf)


struct LikeKernel(StringPredicateKernel):
    """SQL ``LIKE`` (``pc.match_like``): ``%`` = any run, ``_`` = any single
    character, ``\\`` escapes, everything else literal, case-sensitive.

    Two shapes: the inherited array × array `apply`/`dispatch` (general case,
    one pattern per row) and the array × scalar-pattern `apply`/`dispatch`
    below, which compiles the pattern once — `apply_scalar` is what the
    expression layer calls for a constant pattern."""

    comptime name = "match_like"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return LikePattern[False](pat).matches(s)

    @staticmethod
    def apply[
        L: StringLikeType, R: StringLikeType
    ](left: BinaryLikeArray[L], right: BinaryLikeArray[R]) raises -> BoolArray:
        # Overrides the trait default, which would compile the pattern per row.
        Self.expect_same_length(len(left), len(right))
        return _match_arrays[L, R, False](left, right)

    @staticmethod
    def apply[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        return _match_pattern(array, LikePattern[False](pattern))

    @staticmethod
    def apply_scalar[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        # Compile once, not once per row — which is the whole point of
        # `LikePattern`, and had no non-test caller before this.
        return Self.apply(array, pattern)

    @staticmethod
    def dispatch(array: DynArray, pattern: StringSlice) raises -> DynArray:
        return _dispatch_pattern[False](Self.name, array, pattern)


struct ILikeKernel(StringPredicateKernel):
    """Case-insensitive SQL ``LIKE`` (``pc.match_like`` with
    ``ignore_case=True``): both operands are lower-cased before matching."""

    comptime name = "match_like_ci"

    @staticmethod
    def predicate[
        o1: Origin, o2: Origin
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        return LikePattern[True](pat).matches(s)

    @staticmethod
    def apply[
        L: StringLikeType, R: StringLikeType
    ](left: BinaryLikeArray[L], right: BinaryLikeArray[R]) raises -> BoolArray:
        # Overrides the trait default, which would compile the pattern per row.
        Self.expect_same_length(len(left), len(right))
        return _match_arrays[L, R, True](left, right)

    @staticmethod
    def apply[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        return _match_pattern(array, LikePattern[True](pattern))

    @staticmethod
    def apply_scalar[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        return Self.apply(array, pattern)

    @staticmethod
    def dispatch(array: DynArray, pattern: StringSlice) raises -> DynArray:
        return _dispatch_pattern[True](Self.name, array, pattern)


# ---------------------------------------------------------------------------
# Parameterised string kernels — the SQL function surface
# ---------------------------------------------------------------------------
#
# `substr`, `left`, `right`, `lpad`, `rpad`, `replace`, `split_part`,
# `trim(chars)`, `position`, `char_length`, `ascii`. Two shapes — `string ->
# string` and `string -> int64` — each reading one `StringArgs` **per row**.
#
# `StringArgs` used to be the function's *configuration*: four constants the
# caller filled in once. That made `substr(s, 2, 3)` writable and
# `substr(s, start_col, len_col)` unrepresentable, which is the wrong shape —
# these are binary and ternary SQL functions over expressions that merely
# happen to be literals at most call sites. It is also what kept every one of
# them out of the runtime lane, since a constant has nowhere to come from
# there.
#
# `StringOperands` is the fix: the same four slots holding the *columns* each
# argument evaluated to, `args(i)` cutting the row out of them and `is_valid(i)`
# answering whether that row has one. A literal argument broadcasts into a
# column exactly as `Datum.to_array` already broadcasts one, and the kernels
# below are untouched — `transform` still reads a `StringArgs` and knows
# nothing about where it came from.
#
# **Every one of these counts characters, not bytes**, which is the whole
# reason they are here: `LengthKernel` already answers the byte question and
# SQL asks the other one. `héllo wörld` is 11 characters and 13 bytes, so the
# two disagree on every row that leaves ASCII. The implementations are
# deliberately scalar loops over `Utf8.bounds` — this is coverage, not
# throughput, and a correct loop that matches DuckDB beats a vectorised one
# that nearly does.


struct StringArgs(Copyable, ImplicitlyCopyable, Movable, Writable):
    """One row's arguments to a parameterised string kernel.

    One struct for the whole family rather than one per kernel: the alternative
    is eleven near-identical two-field structs and eleven node types to carry
    them, and no kernel here needs a field the others cannot spare. Each kernel
    declares which fields it reads — see `StringArgKernel.uses_text` and its
    three siblings — and ignores the rest.
    """

    var text: String
    """The needle, pattern, fill, separator or character set."""
    var alt: String
    """The replacement — `replace` alone reads it."""
    var start: Int
    """`substr`'s 1-based start position."""
    var count: Int
    """A length, width, repeat count or field index."""

    def __init__(
        out self,
        var text: String = String(),
        var alt: String = String(),
        start: Int = 0,
        count: Int = 0,
    ):
        self.text = text^
        self.alt = alt^
        self.start = start
        self.count = count

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "StringArgs('",
            self.text,
            "', '",
            self.alt,
            "', ",
            self.start,
            ", ",
            self.count,
            ")",
        )


struct StringOperands[
    T: StringLikeType = StringType,
    A: StringLikeType = StringType,
    S: PrimitiveType = Int64Type,
    C: PrimitiveType = Int64Type,
](Copyable, Movable, Writable):
    """`StringArgs`' four slots, each holding the **column** its argument
    evaluated to rather than a constant.

    An absent slot is `None` and reads as `StringArgs`' own default — that is
    what lets one carrier serve `substr` (two numeric slots), `replace` (two
    string slots), `lpad` (one of each) and `char_length` (none at all).

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

    def args(self, i: Int) -> StringArgs:
        """Row `i` of every present slot, as the kernels' per-row arguments.

        A slot the caller left empty keeps `StringArgs`' default, so a kernel
        that never reads it cannot tell the difference.
        """
        var out = StringArgs()
        if self.text:
            out.text = String(self.text.value().unsafe_get(UInt(i)))
        if self.alt:
            out.alt = String(self.alt.value().unsafe_get(UInt(i)))
        if self.start:
            out.start = Int(self.start.value().values().unsafe_get(i))
        if self.count:
            out.count = Int(self.count.value().values().unsafe_get(i))
        return out^

    def is_valid(self, i: Int) -> Bool:
        """Whether every present slot has a value at row `i`.

        **Null in any argument position means null out** — `substr('abc', NULL,
        2)`, `lpad('x', 5, NULL)` and `position(NULL IN 'abc')` are all NULL in
        DuckDB, checked against it directly. The constants this replaced could
        never be null, so the case did not exist before and no kernel body has
        to learn about it: `apply` skips the row exactly as it skips a null
        input string.
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


# -- string -> string, configured ------------------------------------------


trait StringArgKernel(Kernel):
    """Element-wise `string -> string` over a `StringOperands`.

    The argument-taking counterpart of `StringMapKernel`: concrete kernels
    define `transform` plus the four `uses_*` constants, and the
    null-propagating `apply` and the type-erased `dispatch` are defaulted here
    exactly as they are there.

    **The `uses_*` constants are read by the expression node, not by `apply`.**
    A node carries an operand per slot and only the ones its kernel names are
    evaluated, so `left(s, n)` never materialises the string slot it does not
    have. Stating it on the kernel is what keeps that knowledge in the one
    place that has it — `transform` is the code that reads the fields.
    """

    comptime uses_text: Bool
    """Whether `transform` reads `args.text`."""
    comptime uses_alt: Bool
    """Whether `transform` reads `args.alt`."""
    comptime uses_start: Bool
    """Whether `transform` reads `args.start`."""
    comptime uses_count: Bool
    """Whether `transform` reads `args.count`."""

    @staticmethod
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        ...

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
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                builder.append(
                    Self.transform(array.unsafe_get(UInt(i)), ops.args(i))
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch[
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: DynArray, ops: StringOperands[OT, OA, OS, OC]) raises -> DynArray:
        var dt = array.dtype()
        if not dt.is_string_like():
            raise Self.error(t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T](), ops).to_dyn()

        return dt.dispatch_stringlike(leaf)


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
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var first = args.start - 1  # 0-based, may be negative
        var last = first + args.count  # exclusive
        if first < 0:
            first = 0
        if last < 0:
            last = 0
        if first > chars:
            first = chars
        if last > chars:
            last = chars
        return Utf8.slice(s, bounds[first], bounds[last])


struct LeftKernel(StringArgKernel):
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
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var take = args.count if args.count >= 0 else chars + args.count
        if take < 0:
            take = 0
        if take > chars:
            take = chars
        return Utf8.slice(s, bounds[0], bounds[take])


struct RightKernel(StringArgKernel):
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
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var skip = chars - args.count if args.count >= 0 else -args.count
        if skip < 0:
            skip = 0
        if skip > chars:
            skip = chars
        return Utf8.slice(s, bounds[skip], bounds[chars])


struct Pad[left: Bool](StringArgKernel):
    """SQL `lpad`/`rpad(s, count, text)` — pad to `count` characters.

    **The side is a comptime parameter, not a flag.** `_pad(s, args, left)`
    tested a `Bool` argument on every row to choose between two concatenation
    orders, though it cannot vary within a call. This is the shape
    `_match_pattern[T, ignore_case]` already uses to let LIKE and ILIKE share
    one implementation, and the branch is gone at monomorphisation.

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
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var width = args.count
        if width < 0:
            width = 0
        if chars >= width:
            # Truncation keeps the *first* `width` characters whichever end the
            # padding would have gone on — `rpad` does not truncate from the
            # left.
            return Utf8.slice(s, bounds[0], bounds[width])
        var fill = StringSlice(args.text)
        var fb = Utf8.bounds(fill)
        var fchars = len(fb) - 1
        if fchars == 0:
            return String(s)  # nothing to pad with
        var pad = String()
        var k = 0
        while k < width - chars:
            pad += Utf8.slice(fill, fb[k % fchars], fb[k % fchars + 1])
            k += 1
        comptime if Self.left:
            return pad + String(s)
        else:
            return String(s) + pad


comptime LPadKernel = Pad[True]
comptime RPadKernel = Pad[False]


struct ReplaceKernel(StringArgKernel):
    """SQL `replace(s, text, alt)` — **every** occurrence, literally.

    An empty `text` returns the input unchanged rather than interleaving the
    replacement between every character, which is both what DuckDB does and the
    only reading that terminates.
    """

    comptime name = "replace"
    comptime uses_text = True
    comptime uses_alt = True
    comptime uses_start = False
    comptime uses_count = False

    @staticmethod
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        var needle = StringSlice(args.text)
        if len(needle.as_bytes()) == 0:
            return String(s)
        var out = String()
        var n = len(s.as_bytes())
        var i = 0
        while i <= n:
            var j = Utf8.find(s, needle, i)
            if j < 0:
                break
            out += Utf8.slice(s, i, j)
            out += args.alt
            i = j + len(needle.as_bytes())
        out += Utf8.slice(s, i, n)
        return out^


struct SplitPartKernel(StringArgKernel):
    """SQL `split_part(s, text, count)` — the `count`-th field, **1-based**.

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
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        if args.count < 1:
            return String()
        var sep = StringSlice(args.text)
        var m = len(sep.as_bytes())
        var n = len(s.as_bytes())
        if m == 0:
            # No separator, so the whole string is the one and only field.
            return String(s) if args.count == 1 else String()
        var start = 0
        var part = 1
        while part < args.count:
            var j = Utf8.find(s, sep, start)
            if j < 0:
                return String()  # fewer fields than asked for
            start = j + m
            part += 1
        var j = Utf8.find(s, sep, start)
        return Utf8.slice(s, start, j if j >= 0 else n)


struct TrimCharsKernel(StringArgKernel):
    """SQL `trim(s, text)` — strip any leading or trailing character that is a
    **member of the set** `text`, not the literal substring `text`.

    The set is compared code point by code point, so a multi-byte member never
    matches part of a multi-byte character. An empty set strips nothing.
    """

    comptime name = "trim_chars"
    comptime uses_text = True
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = False

    @staticmethod
    def transform[o: Origin](s: StringSlice[o], args: StringArgs) -> String:
        var bounds = Utf8.bounds(s)
        var chars = len(bounds) - 1
        var set = StringSlice(args.text)
        var lo = 0
        while lo < chars and Utf8.charset_has(
            set, s, bounds[lo], bounds[lo + 1]
        ):
            lo += 1
        var hi = chars
        while hi > lo and Utf8.charset_has(set, s, bounds[hi - 1], bounds[hi]):
            hi -= 1
        return Utf8.slice(s, bounds[lo], bounds[hi])


# -- string -> int64, configured -------------------------------------------


trait StringMeasureKernel(Kernel):
    """Element-wise `string -> int64` over a `StringOperands`.

    `int64` rather than the `int32` `LengthKernel` answers with, because these
    are the SQL spellings and SQL's `length` / `position` / `ascii` are
    `BIGINT`. A null input measures null — never 0, which is the answer for the
    *empty* string and a different fact. A null **argument** measures null too,
    for the same reason and by the same rule: `position(NULL IN 'abc')` is NULL
    in DuckDB, not 0.
    """

    comptime uses_text: Bool
    """Whether `measure` reads `args.text`."""

    @staticmethod
    def measure[o: Origin](s: StringSlice[o], args: StringArgs) -> Int64:
        ...

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
    ) raises -> Int64Array:
        var n = len(array)
        var out = Buffer.alloc_uninit[DType.int64](n)
        var dst = out.view[DType.int64](0, n)
        # A null *argument* now makes the row null too, so the input's own
        # bitmap is no longer the answer and `_passthrough_validity` no longer
        # applies -- the two have to meet, which the same loop does for free.
        var valid = Bitmap.alloc_zeroed(n)
        var nulls = 0
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                dst.unsafe_set(
                    i, Self.measure(array.unsafe_get(UInt(i)), ops.args(i))
                )
                valid.unsafe_set(i)
            else:
                dst.unsafe_set(i, Int64(0))
                nulls += 1
        # No bitmap at all when nothing is null: an all-ones bitmap is the same
        # information, but `PrimitiveArray.__eq__` compares validity views and
        # would read it as different from a column that never had one.
        var bitmap = Optional[Bitmap[mut=False]](None)
        if nulls > 0:
            bitmap = valid^.to_immutable()
        return Int64Array(
            length=n,
            nulls=nulls,
            offset=0,
            bitmap=bitmap^,
            buffer=out.to_immutable(),
        )

    @staticmethod
    def dispatch[
        OT: StringLikeType,
        OA: StringLikeType,
        OS: PrimitiveType,
        OC: PrimitiveType,
    ](array: DynArray, ops: StringOperands[OT, OA, OS, OC]) raises -> DynArray:
        var dt = array.dtype()
        if not dt.is_string_like():
            raise Self.error(t"expected a string array, got {dt}")

        def leaf[T: StringLikeType](d: T) raises {imm} -> DynArray:
            return Self.apply(array.as_binary_like[T](), ops).to_dyn()

        return dt.dispatch_stringlike(leaf)


struct CharLengthKernel(StringMeasureKernel):
    """SQL `length(s)` — **characters**, where `LengthKernel` counts bytes.

    Two different functions, not one with a bug: `héllo wörld` is 11 here and
    13 there, and DuckDB spells the pair `length` and `octet_length`.
    """

    comptime name = "char_length"
    comptime uses_text = False

    @staticmethod
    def measure[o: Origin](s: StringSlice[o], args: StringArgs) -> Int64:
        return Int64(Utf8.count(s))


struct PositionKernel(StringMeasureKernel):
    """SQL `position(text IN s)` — the 1-based **character** index of the first
    occurrence, and **0** when there is none.

    Zero rather than null is what makes the result an index into a 1-based
    world; a 0-based engine cannot reuse the convention. An empty needle is
    found at position 1, which follows from `Utf8.find` matching it at the
    cursor.
    """

    comptime name = "position"
    comptime uses_text = True

    @staticmethod
    def measure[o: Origin](s: StringSlice[o], args: StringArgs) -> Int64:
        var at = Utf8.find(s, StringSlice(args.text), 0)
        if at < 0:
            return Int64(0)
        # Byte offset -> 1-based character index.
        var bounds = Utf8.bounds(s)
        for k in range(len(bounds) - 1):
            if bounds[k] == at:
                return Int64(k + 1)
        return Int64(len(bounds))


struct AsciiKernel(StringMeasureKernel):
    """SQL `ascii(s)` — the first character's code point, **0** for the empty
    string.

    The smallest function here that has to *decode* UTF-8 rather than move
    bytes: `héllo wörld` answers 104 for `h`, and a string starting with `é`
    must answer 233 rather than the lead byte 0xC3.
    """

    comptime name = "ascii"
    comptime uses_text = False

    @staticmethod
    def measure[o: Origin](s: StringSlice[o], args: StringArgs) -> Int64:
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


# -- string x int -> string -------------------------------------------------


struct RepeatKernel(Kernel):
    """SQL `repeat(s, n)` with the count taken from a **column**.

    Zero and negative counts both answer the empty string — DuckDB's reading;
    several engines raise on a negative instead. Null on either side gives
    null, which is why this cannot be a `StringArgKernel`: the count is per row
    and carries its own validity.
    """

    comptime name = "repeat"

    @staticmethod
    def apply[
        T: StringLikeType, N: PrimitiveType
    ](
        array: BinaryLikeArray[T], counts: PrimitiveArray[N]
    ) raises -> BinaryLikeArray[T]:
        # `PrimitiveType`, not `NumericType`: `NumericValue` is a *marker* over
        # `PrimitiveValue`, so its `Type` is only ever bounded by
        # `PrimitiveType` and a narrower bound here could not be satisfied by
        # the node that calls this.
        Self.expect_same_length(len(array), len(counts))
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        for i in range(n):
            if array.is_valid(i) and counts.is_valid(i):
                var times = Int(counts.values().unsafe_get(i))
                var out = String()
                for _ in range(times):
                    out += array.unsafe_get(UInt(i))
                builder.append(out)
            else:
                builder.append_null()
        return builder.finish()
