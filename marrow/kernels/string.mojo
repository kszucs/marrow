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
- **SQL pattern match → bool** (`Like`, `ILike`) — the predicate shape above
  plus an array × scalar-pattern overload built on `LikePattern`, which
  compiles the pattern once instead of per row.

Variable-width ops cannot lane-fuse the way numeric kernels do (there is no
fixed W-wide lane), so the expression layer (`marrow.expr.values`) materializes
them. Only `LengthKernel` exposes a fusable, offset-based fast path there.
"""

from std.sys import size_of
from std.sys.info import simd_byte_width
from std.algorithm.backend.vectorize import vectorize
from std.utils.index import IndexList

from ..arrays import (
    DynArray,
    BinaryLikeArray,
    BoolArray,
    Int32Array,
)
from ..buffers import Buffer, Bitmap
from ..builders import BinaryLikeBuilder
from ..dtypes import StringLikeType, DType
from .core import Kernel


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
    def dispatch(array: DynArray) raises -> DynArray:
        # Guard before dispatching so the diagnostic names *this kernel* and the
        # family it wanted. `dispatch_stringlike` would otherwise fall through to
        # `variant_dispatch`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if not dt.is_string_like():
            raise Self.error(t"expected a string array, got {dt}")

        @parameter
        def leaf[T: StringLikeType](d: T) raises -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.dispatch_stringlike[leaf]()


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
        # `variant_dispatch`'s generic "no arm matched", which says neither.
        var dt = array.dtype()
        if not dt.is_string_like():
            raise Self.error(t"expected a string array, got {dt}")

        @parameter
        def leaf[T: StringLikeType](d: T) raises -> DynArray:
            return Self.apply(array.as_binary_like[T]()).to_dyn()

        return dt.dispatch_stringlike[leaf]()


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

        @parameter
        def leaf[T: StringLikeType](d: T) raises -> DynArray:
            return Self.apply(
                left.as_binary_like[T](), right.as_binary_like[T]()
            ).to_dyn()

        return dt.dispatch_stringlike[leaf]()

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
        Self.expect_same_length(len(left), len(right))
        var n = len(left)
        var bm = Bitmap.intersect(left.bitmap.copy(), right.bitmap.copy())
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

        @parameter
        def leaf[T: StringLikeType](d: T) raises -> DynArray:
            return Self.apply(
                left.as_binary_like[T](), right.as_binary_like[T]()
            ).to_dyn()

        return dt.dispatch_stringlike[leaf]()


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
# vectorize, so whoever interprets the operator pairs them (see `_compare` in
# `marrow/expr/dynamic.mojo`).


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


@always_inline
def _utf8_width(lead: UInt8) -> Int:
    """Byte width of the UTF-8 sequence starting at this lead byte."""
    if lead < 0x80:
        return 1
    elif lead < 0xE0:
        return 2
    elif lead < 0xF0:
        return 3
    else:
        return 4


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
            i += _utf8_width(bytes[i])
            j += 1
        elif j < m and tokens[j] == _LIKE_ANY:
            star_j = j
            star_i = i
            j += 1
        elif star_j != -1:
            # backtrack: let the last '%' absorb one more character
            j = star_j + 1
            star_i += _utf8_width(bytes[star_i])
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
                    var w = _utf8_width(bytes[i + 1])
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
                var w = _utf8_width(b)
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
        var v = array.bitmap.value().view(array.offset, n)
        return v.union(v).to_immutable()
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


def _dispatch_pattern[
    ignore_case: Bool
](name: StringSlice, array: DynArray, pattern: StringSlice) raises -> DynArray:
    """Type-erased entry point for the scalar-pattern overloads."""
    var compiled = LikePattern[ignore_case](pattern)
    var dt = array.dtype()
    if not dt.is_string_like():
        raise Error(t"{name}: expected a string array, got {dt}")

    @parameter
    def leaf[T: StringLikeType](d: T) raises -> DynArray:
        return _match_pattern(array.as_binary_like[T](), compiled).to_dyn()

    return dt.dispatch_stringlike[leaf]()


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
