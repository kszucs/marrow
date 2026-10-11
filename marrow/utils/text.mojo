# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Text over UTF-8 bytes: code points (`Utf8`), SQL `LIKE` patterns
(`LikePattern`) and regular expressions (`RegexPattern`, DuckDB's semantics).

`RegexPattern` is the one place that talks to the engine, `mojo-regex` (the
`regex` package), so a change of engine is a change to that struct. Two paths
through it, and they do not always agree: `CompiledRegex.match_next` finds the
overall match, and capture groups come from the NFA's
`match_next_with_groups`, which never enters an optional group, so
`(?:foo)?(bar)` on `"foobar"` captures from offset 3 rather than 0. Only a
group above 0 and a `\\N` replacement reach the second path.
"""

from regex.matcher import CompiledRegex, Match


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
    """Code points over UTF-8 bytes: what a SQL string function counts in,
    where a byte length counts bytes.

    `bounds` is the primitive the rest are built on: the byte offset of every
    code point start plus the end offset, so a string of `c` characters yields
    `c + 1` entries, and a character answer is an index into it.
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
        var bytes = s.as_bytes()
        var n = 0
        var i = 0
        while i < len(bytes):
            i += Self.width(bytes[i])
            n += 1
        return n

    @staticmethod
    def slice(s: StringSlice, start: Int, end: Int) -> String:
        """Bytes `[start, end)` of `s`, copied into a fresh `String`.

        Both ends are assumed to sit on code point boundaries; every caller
        derives them from `bounds`, so a multi-byte character is never split.
        """
        if end <= start:
            return String()
        return String(from_utf8_lossy=s.as_bytes()[start:end])

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


struct LikePattern[ignore_case: Bool](Copyable, Movable):
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
                    # A truncated sequence keeps the bytes it has.
                    var w = min(Utf8.width(bytes[i + 1]), n - i - 1)
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
                var w = min(Utf8.width(b), n - i)
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
            var fold = Self._fold_kind(s)
            if fold == _FOLD_NONE:
                return self._matches_folded(s)
            elif fold == _FOLD_ASCII:
                var folded = Self._ascii_lower(s)
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
            return self._match_tokens(s)

    @staticmethod
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

    @staticmethod
    def _ascii_lower(s: StringSlice) -> String:
        """An ASCII-lowercased copy of a pure-ASCII `s` (see `_fold_kind`)."""
        var bytes = s.as_bytes()
        var out = List[UInt8](capacity=len(bytes))
        for i in range(len(bytes)):
            var b = bytes[i]
            out.append(b + 32 if b >= 0x41 and b <= 0x5A else b)
        return String(from_utf8_lossy=Span(out))

    def _match_tokens(self, text: StringSlice) -> Bool:
        """Greedy matcher over a row's UTF-8 bytes and this pattern's tokens.

        O(len(text) * len(tokens)) worst case, O(1) extra space via the classic
        backtracking-on-star algorithm. Literal tokens advance one byte at a
        time while ``_`` and ``%`` advance whole code points, so the text
        cursor is always on a code-point boundary when a wildcard is consumed.
        """
        var bytes = text.as_bytes()
        var n = len(bytes)
        var m = len(self.tokens)
        var i = 0  # byte cursor in text
        var j = 0  # token cursor
        var star_j = -1  # token index of the last '%' seen, or -1
        var star_i = 0  # byte index matched against that '%'
        while i < n:
            if j < m and self.tokens[j] == Int(bytes[i]):
                i += 1
                j += 1
            elif j < m and self.tokens[j] == _LIKE_ONE:
                i += Utf8.width(bytes[i])
                j += 1
            elif j < m and self.tokens[j] == _LIKE_ANY:
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
        while j < m and self.tokens[j] == _LIKE_ANY:
            j += 1
        return j == m


struct RegexPattern(Movable):
    """A pattern compiled on demand, and the three questions the kernels ask
    of it. `use` recompiles only when the text changes."""

    var compiled: Optional[CompiledRegex]
    var text: String

    def __init__(out self):
        self.compiled = None
        self.text = String()

    def use(mut self, pattern: StringSlice) raises:
        """Make `pattern` the compiled one, unless it already is."""
        if not self.compiled or StringSlice(self.text) != pattern:
            var text = String(pattern)
            self.compiled = CompiledRegex(text)
            self.text = text^

    @no_inline
    def _first_match[
        o: Origin[mut=False]
    ](self, s: StringSlice[o]) -> Optional[Match[o]]:
        """The first match in `s`, from the overall matcher.

        Out of line on purpose: the library inlines its whole engine dispatch
        into every caller, and search, extract and replace each carried a
        copy, 40-47 KB apiece in a runtime-lane binary."""
        return self.compiled.value().match_next(s)

    def search[o: Origin[mut=False]](self, s: StringSlice[o]) -> Bool:
        """Whether the pattern matches anywhere in `s` — a partial match."""
        return Bool(self._first_match(s))

    def _groups[
        o: Origin[mut=False]
    ](self, s: StringSlice[o]) -> Tuple[Optional[Match[o]], List[Match[o]]]:
        """The first match and its capture groups, from the NFA path — the
        only place the library reports groups."""
        return self.compiled.value().matcher.nfa_matcher.engine.match_next_with_groups(
            s, 0
        )

    def extract[
        o: Origin[mut=False]
    ](self, s: StringSlice[o], group: Int) -> StringSlice[o]:
        """Capture group `group` of the first match; group 0 is the whole
        match. No match, or a group that did not participate, answers `''`.
        """
        if group == 0:
            var m = self._first_match(s)
            if m:
                return m.value().get_match_text()
            return StringSlice(unsafe_from_utf8=s.as_bytes()[:0])
        var found = self._groups(s)
        ref groups = found[1]
        for i in range(len(groups)):
            if groups[i].group_id == group:
                return groups[i].get_match_text()
        return StringSlice(unsafe_from_utf8=s.as_bytes()[:0])

    def replace[
        o: Origin[mut=False], r: Origin[mut=False]
    ](self, s: StringSlice[o], repl: StringSlice[r]) -> String:
        """`s` with its first match replaced by the template `repl`."""
        # Not the library's `sub`: its template has no `\\0` or `\\\\`.
        var template = Replacement(repl)
        var bytes = s.as_bytes()
        var spans = Array[Tuple[Int, Int], 10](fill=(-1, -1))
        if template.uses_groups:
            var found = self._groups(s)
            if not found[0]:
                return String(s)
            spans[0] = (found[0].value().start_idx, found[0].value().end_idx)
            ref groups = found[1]
            for i in range(len(groups)):
                var g = groups[i].group_id
                if 1 <= g and g <= 9:
                    spans[g] = (groups[i].start_idx, groups[i].end_idx)
        else:
            var m = self._first_match(s)
            if not m:
                return String(s)
            spans[0] = (m.value().start_idx, m.value().end_idx)
        var out = String(StringSlice(unsafe_from_utf8=bytes[: spans[0][0]]))
        template.expand(out, s, spans)
        out += StringSlice(unsafe_from_utf8=bytes[spans[0][1] :])
        return out^


struct Replacement[origin: Origin[mut=False]]:
    """A `regexp_replace` template: literal text in which `\\0`..`\\9` insert a
    capture group and `\\\\` a backslash, as in RE2 and DuckDB."""

    var text: StringSlice[Self.origin]
    var uses_groups: Bool
    """Whether the template names a group, which decides the matcher."""

    def __init__(out self, text: StringSlice[Self.origin]):
        self.text = text
        self.uses_groups = False
        var b = text.as_bytes()
        var i = 0
        while i + 1 < len(b):
            if b[i] == UInt8(ord("\\")):
                if Self._is_digit(b[i + 1]):
                    self.uses_groups = True
                    return
                i += 2
            else:
                i += 1

    @staticmethod
    def _is_digit(c: UInt8) -> Bool:
        return c >= UInt8(ord("0")) and c <= UInt8(ord("9"))

    def expand[
        o: Origin[mut=False]
    ](
        self,
        mut out: String,
        s: StringSlice[o],
        spans: Array[Tuple[Int, Int], 10],
    ):
        """Append this template to `out`, taking group `N` from `spans[N]` over
        `s`; a group that did not participate is `(-1, -1)` and inserts
        nothing."""
        var b = self.text.as_bytes()
        var text = s.as_bytes()
        var lit = 0
        var i = 0
        while i + 1 < len(b):
            if b[i] != UInt8(ord("\\")):
                i += 1
                continue
            var c = b[i + 1]
            if Self._is_digit(c):
                out += StringSlice(unsafe_from_utf8=b[lit:i])
                var g = Int(c - UInt8(ord("0")))
                if spans[g][0] >= 0:
                    out += StringSlice(
                        unsafe_from_utf8=text[spans[g][0] : spans[g][1]]
                    )
                lit = i + 2
            elif c == UInt8(ord("\\")):
                out += StringSlice(unsafe_from_utf8=b[lit : i + 1])
                lit = i + 2
            i += 2
        out += StringSlice(unsafe_from_utf8=b[lit:])
