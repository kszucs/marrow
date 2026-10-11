# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Regular-expression kernels: `regexp_matches`, `regexp_extract` and
`regexp_replace`, with DuckDB's semantics, over `utils.RegexPattern`.

A pattern is compiled once per *run* of equal pattern texts, as `LIKE` does:
the runtime lane hands a literal pattern over as n identical rows, and those
collapse to one compile.
"""

from ..arrays import (
    BinaryLikeArray,
    BoolArray,
    BytesArray,
    DynArray,
    Int64Array,
    PrimitiveArray,
    StringArray,
)
from ..builders import BinaryLikeBuilder, BoolBuilder
from ..dtypes import PrimitiveType, StringLikeType
from ..utils import RegexPattern
from .string import (
    StringPredicateKernel,
    StringTextIntKernel,
    StringTextTextKernel,
    dispatch_text,
)


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


struct RegexpExtractKernel(StringTextIntKernel):
    """DuckDB `regexp_extract(s, pattern, group)`: capture group `group` of
    the first match. A row that does not match answers `''`, not null."""

    comptime name = "regexp_extract"

    @staticmethod
    def apply[
        T: StringLikeType, P: StringLikeType, G: PrimitiveType
    ](
        array: BinaryLikeArray[T],
        pattern: BinaryLikeArray[P],
        group: PrimitiveArray[G],
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        var compiled = RegexPattern()
        for i in range(n):
            if array.is_valid(i) and pattern.is_valid(i) and group.is_valid(i):
                compiled.use(pattern.unsafe_get(UInt(i)))
                builder.append(
                    compiled.extract(
                        array.unsafe_get(UInt(i)),
                        Int(group.values().unsafe_get(i)),
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(
        array: DynArray, pattern: StringArray, group: Int64Array
    ) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, pattern, group).to_dyn()

        return dispatch_text[Self](array, run)


struct RegexpReplaceKernel(StringTextTextKernel):
    """DuckDB `regexp_replace(s, pattern, repl)`: the **first** match only,
    where the literal `replace` replaces every occurrence. `\\1`..`\\9` in
    `repl` insert a capture group."""

    comptime name = "regexp_replace"

    @staticmethod
    def apply[
        T: StringLikeType, P: StringLikeType, R: StringLikeType
    ](
        array: BinaryLikeArray[T],
        pattern: BinaryLikeArray[P],
        repl: BinaryLikeArray[R],
    ) raises -> BinaryLikeArray[T]:
        var n = len(array)
        var builder = BinaryLikeBuilder[T](capacity=n)
        var compiled = RegexPattern()
        for i in range(n):
            if array.is_valid(i) and pattern.is_valid(i) and repl.is_valid(i):
                compiled.use(pattern.unsafe_get(UInt(i)))
                builder.append(
                    compiled.replace(
                        array.unsafe_get(UInt(i)), repl.unsafe_get(UInt(i))
                    )
                )
            else:
                builder.append_null()
        return builder.finish()

    @staticmethod
    def dispatch(
        array: DynArray, pattern: StringArray, repl: StringArray
    ) raises -> DynArray:
        def run[
            T: StringLikeType
        ](s: BinaryLikeArray[T]) raises {imm} -> DynArray:
            return Self.apply(s, pattern, repl).to_dyn()

        return dispatch_text[Self](array, run)
