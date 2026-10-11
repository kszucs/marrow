# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Regular-expression kernels: `regexp_matches`, `regexp_extract` and
`regexp_replace`, with DuckDB's semantics, over `utils.RegexPattern`.

A pattern is compiled once per *run* of equal pattern texts, as `LIKE` does:
the runtime lane hands a literal pattern over as n identical rows, and those
collapse to one compile.
"""

from ..arrays import BinaryLikeArray, BoolArray, BytesArray
from ..builders import BinaryLikeBuilder, BoolBuilder
from ..dtypes import PrimitiveType, StringLikeType
from ..errors import InternalError
from ..utils import RegexPattern
from .string import StringArgKernel, StringOperands, StringPredicateKernel


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
        L: BytesArray, R: BytesArray
    ](left: L, right: R) raises -> BoolArray:
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
        A: BytesArray
    ](array: A, pattern: StringSlice) raises -> BoolArray:
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
