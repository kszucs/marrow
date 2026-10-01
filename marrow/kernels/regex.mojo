# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Regular-expression kernels: `regexp_matches`, `regexp_extract` and
`regexp_replace`, with DuckDB's semantics.

The engine is `mojo-regex` (the `regex` package), taken from a fork that builds
on marrow's pinned nightly — see `pixi.toml`. `RegexPattern` is the one place
that talks to it, so a change of engine is a change to that struct.

Two paths through the engine, and they do not always agree:

- `CompiledRegex.match_next` finds the overall match. It is what
  `regexp_matches` and group 0 use, and it is right on every case the golden
  corpus has.
- Capture groups come from the NFA's `match_next_with_groups`, the only
  place the library reports them. That path never enters an optional group, so
  `(?:foo)?(bar)` on `"foobar"` captures from offset 3 rather than 0 — the
  known upstream bug, pinned by `test_regexp_replace_optional_group_known_bug`.
  Only `regexp_extract` with a group above 0 and `regexp_replace` with a `\\N`
  reference reach it.

The library's own `sub` is not used: its template knows `\\1`..`\\9` but not
DuckDB's `\\0` or `\\\\`, and it takes `StringSlice[ImmutAnyOrigin]`, which an
array's value slice reaches only through an origin cast. Replacement is a
first-match splice over the match spans instead.

A pattern is compiled once per *run* of equal pattern texts, the same memo
`LikePattern.match_arrays` uses: the runtime lane hands a literal pattern over
as n identical rows, and those collapse to one compile.
"""

from regex.matcher import CompiledRegex, Match

from ..arrays import BinaryLikeArray, BoolArray
from ..builders import BinaryLikeBuilder, BoolBuilder
from ..dtypes import PrimitiveType, StringLikeType
from ..errors import InternalError
from .string import StringArgKernel, StringOperands, StringPredicateKernel


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
            self.text = String(pattern)
            self.compiled = CompiledRegex(self.text)

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


struct RegexpMatchesKernel(StringPredicateKernel):
    """DuckDB `regexp_matches(s, pattern)`: whether `pattern` matches
    **anywhere** in `s`. Anchor with `^`/`$` for a full match."""

    comptime name = "regexp_matches"

    @staticmethod
    def predicate[
        o1: Origin[mut=False], o2: Origin[mut=False]
    ](s: StringSlice[o1], pat: StringSlice[o2]) -> Bool:
        # Required by the trait and reached by neither `apply`: both compile
        # through `RegexPattern`, which raises on a malformed pattern. This
        # signature cannot raise, so a malformed pattern answers False here.
        try:
            var compiled = RegexPattern()
            compiled.use(pat)
            return compiled.search(s)
        except:
            return False

    @staticmethod
    def apply[
        L: StringLikeType, R: StringLikeType
    ](left: BinaryLikeArray[L], right: BinaryLikeArray[R]) raises -> BoolArray:
        Self.expect_same_length(len(left), len(right))
        var n = len(left)
        var out = BoolBuilder(capacity=n)
        var compiled = RegexPattern()
        for i in range(n):
            if left.is_valid(i) and right.is_valid(i):
                compiled.use(right.unsafe_get(UInt(i)))
                out.append(compiled.search(left.unsafe_get(UInt(i))))
            else:
                out.append_null()
        return out.finish()

    @staticmethod
    def apply_scalar[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        var compiled = RegexPattern()
        compiled.use(pattern)
        var n = len(array)
        var out = BoolBuilder(capacity=n)
        for i in range(n):
            if array.is_valid(i):
                out.append(compiled.search(array.unsafe_get(UInt(i))))
            else:
                out.append_null()
        return out.finish()


struct RegexpExtractKernel(StringArgKernel):
    """DuckDB `regexp_extract(s, pattern, group)`: capture group `group` of
    the first match. A row that does not match answers `''`, not null."""

    comptime name = "regexp_extract"
    comptime uses_text = True
    comptime uses_alt = False
    comptime uses_start = False
    comptime uses_count = True

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
        ref patterns = ops.text.value()
        ref groups = ops.count.value()
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        var compiled = RegexPattern()
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                compiled.use(patterns.unsafe_get(UInt(i)))
                builder.append(
                    compiled.extract(
                        array.unsafe_get(UInt(i)),
                        Int(groups.values().unsafe_get(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()


struct RegexpReplaceKernel(StringArgKernel):
    """DuckDB `regexp_replace(s, pattern, repl)`: the **first** match only,
    where the literal `replace` replaces every occurrence. `\\1`..`\\9` in
    `repl` insert a capture group."""

    comptime name = "regexp_replace"
    comptime uses_text = True
    comptime uses_alt = True
    comptime uses_start = False
    comptime uses_count = False

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
        ref patterns = ops.text.value()
        ref repls = ops.alt.value()
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        var compiled = RegexPattern()
        for i in range(n):
            if array.is_valid(i) and ops.is_valid(i):
                compiled.use(patterns.unsafe_get(UInt(i)))
                builder.append(
                    compiled.replace(
                        array.unsafe_get(UInt(i)), repls.unsafe_get(UInt(i))
                    )
                )
            else:
                builder.append_null()
        return builder.finish()
