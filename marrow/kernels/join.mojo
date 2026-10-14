# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Join kernels for Arrow StructArrays.

Public API
----------
``hash_join``   — equijoin two StructArrays on positional key columns.
``HashJoin``    — hash join over ``JoinHashTable``; reusable across morsels.

Internal types
--------------
``JoinIndex``   — (build_indices, probe_indices) result of a probe phase.

Supported join kinds (pass the JOIN_* constants defined below):
  JOIN_INNER       — only matched rows
  JOIN_LEFT        — all left + matched right (NULLs for non-matches)
  JOIN_RIGHT       — all right + matched left (NULLs for non-matches)
  JOIN_FULL        — all rows from both sides
  JOIN_SEMI        — left rows with at least one match (left columns only)
  JOIN_ANTI        — left rows with no match (left columns only)
  JOIN_RIGHT_SEMI  — right rows with at least one match (right columns only)
  JOIN_RIGHT_ANTI  — right rows with no match (right columns only)

Supported strictness:
  JOIN_ALL    — default: return all matching pairs (Cartesian for multi-match)
  JOIN_ANY    — return at most one matching right row per left row

Which side is indexed is separate from which side's columns come first:
``hash_join(a, b, kind, build_side=BUILD_RIGHT)`` indexes ``b``, streams ``a``
and returns the same fields in the same order as ``BUILD_LEFT``.

Future join algorithms (see `backlog.md`); operators name the concrete
algorithm, so a new one is a new struct, not a conformance:
  SortMergeJoin   — sort both sides, two-pointer merge (no hash table)

``JoinHashTable`` is a ``DictionaryEncoder`` over the build keys plus each
code's build rows, stored contiguously.
"""


from std.math import ceildiv

from ..utils.stats import count
from ..arrays import (
    DynArray,
    StructArray,
    Int32Array,
)
from ..buffers import Bitmap, Buffer
from ..builders import Int32Builder, nulls
from ..dtypes import (
    DynType,
    Field,
    int32,
    struct_,
    null,
)
from ..execution import ExecContext
from ..errors import InvalidError, NotImplementedError
from .core import Kernel
from .filter import FilterKernel, TakeKernel
from .boolean import NotNullKernel
from .dictionary import DictionaryEncoder
from ..utils import Hasher, KeyHash

# ---------------------------------------------------------------------------
# Join kind constants — what rows appear in output
#
# Owned here, not in the plan layer, since these describe the join kernel's own
# algorithm/behavior; the relational-plan layer is a consumer of this
# vocabulary, not its owner. `marrow/expr/logical.mojo` imports these.
# ---------------------------------------------------------------------------


struct JoinKind(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which rows a join emits — and therefore which *columns*.

    "Left" and "right" mean a kind's first and second side. A caller reads
    them against its own inputs; inside the kernel they are read against
    `(build, probe)`, which `JoinBuildSide.physical` converts to.

    A value type rather than a bare `UInt8` for two reasons, both of which had
    already cost something:

    1. **The column question had four answers.** "Does this kind emit the right
       side's columns?" was re-derived inline at `_output_dtype`, `_assemble`,
       `relations.Join.schema` and `tabular.join`, and they did not agree — the
       first two differed on MARK, so a MARK join built a `StructArray`
       declaring the right side's fields while carrying only the left's. That is
       a corrupt array, and nothing checked. It is one method now.
    2. **`kind` and `strictness` were both `UInt8`.** Passing them swapped
       compiled silently, and the numbering makes it invisible rather than merely
       plausible: `JOIN_INNER` and `JOIN_ALL` are both 0, `JOIN_LEFT` and
       `JOIN_ANY` are both 1. Strictness stays `UInt8` for now, but the two
       are no longer interchangeable at a call site.

    Both references put these predicates on the type — polars has
    `JoinType::is_semi_anti()` / `is_equi()`, ClickHouse a set of `constexpr
    isLeft(kind)` free functions. Neither answers the question inline at a use
    site. marrow follows polars' *flat* model, where SEMI and ANTI are kinds;
    ClickHouse instead files them under strictness.
    """

    var code: UInt8
    """The wire value. Stable — `expr.relations` and the Python bindings both
    round-trip it."""

    @implicit
    def __init__(out self, code: UInt8):
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        return self.code != other.code

    def emits_left_columns(self) -> Bool:
        """Whether the output carries the left side's columns: false only for
        RIGHT_SEMI and RIGHT_ANTI. With `emits_right_columns` it fixes the
        output width, which `_output_dtype` and `_assemble` must agree on.
        """
        return self != JOIN_RIGHT_SEMI and self != JOIN_RIGHT_ANTI

    def emits_right_columns(self) -> Bool:
        """Whether the output carries the right side's columns: false only
        for SEMI and ANTI."""
        return self != JOIN_SEMI and self != JOIN_ANTI

    def emits_unmatched_left(self) -> Bool:
        """Whether unmatched left-side rows appear, padded with nulls."""
        return self == JOIN_LEFT or self == JOIN_FULL

    def emits_unmatched_right(self) -> Bool:
        """Whether unmatched right-side rows appear, padded with nulls."""
        return self == JOIN_RIGHT or self == JOIN_FULL

    def needs_left_matches(self) -> Bool:
        """Whether which left-side rows matched is only known once every
        right-side row has been seen: true for LEFT, FULL, SEMI and ANTI,
        whose left rows are emitted (or not) after the last right row."""
        return (
            self == JOIN_LEFT
            or self == JOIN_FULL
            or self == JOIN_SEMI
            or self == JOIN_ANTI
        )

    def negates(self) -> Bool:
        """Whether this existence filter keeps the rows that did not match:
        true for ANTI and RIGHT_ANTI. With `emits_left_columns` and
        `emits_right_columns`, which say which side a filter projects, it
        describes an existence filter completely.
        """
        return self == JOIN_ANTI or self == JOIN_RIGHT_ANTI

    def commutes(self) -> Bool:
        """Whether the two sides may exchange roles: `mirror` without the
        raise, for a rule that declines rather than fails. It answers what
        `is_supported` does, because every supported kind has a mirror; it is
        a separate method because the two ask different questions.
        """
        return self.is_supported()

    def mirror(self) raises -> Self:
        """This kind with its two sides exchanged: joining `(b, a)` under
        `k.mirror()` returns the rows that joining `(a, b)` under `k` does.
        An involution over the supported kinds.

        Raises:
            Error: for CROSS, MARK and SINGLE, which have no mirror.
        """
        if self == JOIN_INNER:
            return JOIN_INNER
        elif self == JOIN_FULL:
            return JOIN_FULL
        elif self == JOIN_LEFT:
            return JOIN_RIGHT
        elif self == JOIN_RIGHT:
            return JOIN_LEFT
        elif self == JOIN_SEMI:
            return JOIN_RIGHT_SEMI
        elif self == JOIN_RIGHT_SEMI:
            return JOIN_SEMI
        elif self == JOIN_ANTI:
            return JOIN_RIGHT_ANTI
        elif self == JOIN_RIGHT_ANTI:
            return JOIN_ANTI
        else:
            raise InvalidError(t"join: join kind '{self}' cannot be mirrored")

    def is_supported(self) -> Bool:
        """Whether a kernel actually implements this kind.

        CROSS, MARK and SINGLE have constants and no implementation. They used
        to fall through to the outer-join arm and silently produce wrong output;
        `hash_join` now rejects them. Every supported kind has a `mirror`, or
        the build side could change the answer.
        """
        return (
            self == JOIN_INNER
            or self == JOIN_LEFT
            or self == JOIN_RIGHT
            or self == JOIN_FULL
            or self == JOIN_SEMI
            or self == JOIN_ANTI
            or self == JOIN_RIGHT_SEMI
            or self == JOIN_RIGHT_ANTI
        )

    def write_to[W: Writer](self, mut writer: W):
        """PyArrow's spelling, so error messages and `tabular.join`'s `how=`
        argument use one vocabulary."""
        if self == JOIN_INNER:
            writer.write("inner")
        elif self == JOIN_LEFT:
            writer.write("left outer")
        elif self == JOIN_RIGHT:
            writer.write("right outer")
        elif self == JOIN_FULL:
            writer.write("full outer")
        elif self == JOIN_SEMI:
            writer.write("left semi")
        elif self == JOIN_ANTI:
            writer.write("left anti")
        elif self == JOIN_RIGHT_SEMI:
            writer.write("right semi")
        elif self == JOIN_RIGHT_ANTI:
            writer.write("right anti")
        elif self == JOIN_CROSS:
            writer.write("cross")
        elif self == JOIN_MARK:
            writer.write("mark")
        elif self == JOIN_SINGLE:
            writer.write("single")
        else:
            writer.write("join kind ", self.code)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    @staticmethod
    def parse(how: String) raises -> Self:
        """PyArrow's `how=` spelling, with the short forms also accepted.

        The inverse of `write_to`, and it lives here so the name-to-kind mapping
        has one owner. `tabular.join` had its own copy — a fourth place that
        knew which kinds exist, and the one that would silently disagree when a
        kind was added."""
        if how == "inner":
            return JOIN_INNER
        elif how == "left outer" or how == "left":
            return JOIN_LEFT
        elif how == "right outer" or how == "right":
            return JOIN_RIGHT
        elif how == "full outer" or how == "full":
            return JOIN_FULL
        elif how == "left semi" or how == "semi":
            return JOIN_SEMI
        elif how == "left anti" or how == "anti":
            return JOIN_ANTI
        elif how == "right semi":
            return JOIN_RIGHT_SEMI
        elif how == "right anti":
            return JOIN_RIGHT_ANTI
        else:
            raise InvalidError(t"join: unknown join type '{how}'")


comptime JOIN_INNER = JoinKind(0)
"""INNER JOIN: only rows with matching keys on both sides."""

comptime JOIN_LEFT = JoinKind(1)
"""LEFT JOIN: all left rows + matched right rows; NULLs for non-matches."""

comptime JOIN_RIGHT = JoinKind(2)
"""RIGHT JOIN: all right rows + matched left rows; NULLs for non-matches."""

comptime JOIN_FULL = JoinKind(3)
"""FULL OUTER JOIN: all rows from both sides; NULLs for non-matches."""

comptime JOIN_SEMI = JoinKind(4)
"""LEFT SEMI JOIN: left rows that have at least one match in right (left columns only)."""

comptime JOIN_ANTI = JoinKind(5)
"""LEFT ANTI JOIN: left rows with no match in right (left columns only)."""

comptime JOIN_CROSS = JoinKind(6)
"""CROSS JOIN: Cartesian product; no key columns required. **Not implemented** —
`is_supported()` is False and `hash_join` rejects it."""

comptime JOIN_RIGHT_SEMI = JoinKind(7)
"""RIGHT SEMI JOIN: right rows that have at least one match in left (right
columns only). `JOIN_SEMI.mirror()`, and the reason a semi-join may index
either side."""

comptime JOIN_RIGHT_ANTI = JoinKind(8)
"""RIGHT ANTI JOIN: right rows with no match in left (right columns only).
`JOIN_ANTI.mirror()`."""

# Internal join kinds — generated by the planner for subquery decorrelation.
# Not intended for direct use. **Neither is implemented**; both are rejected.
comptime JOIN_MARK = JoinKind(10)
"""MARK JOIN: adds a boolean marker column for EXISTS/IN subquery rewriting."""

comptime JOIN_SINGLE = JoinKind(11)
"""SINGLE JOIN: at-most-1 right row per left row; for scalar subqueries."""

# ---------------------------------------------------------------------------
# Join strictness constants — how many matches are used
# ---------------------------------------------------------------------------

comptime JOIN_ALL: UInt8 = 0
"""ALL strictness (default): return all matching rows (Cartesian product for multi-match)."""

comptime JOIN_ANY: UInt8 = 1
"""ANY strictness: return at most one matching right row per left row (no row duplication).

Which matching row is unspecified: any one of them is a correct answer, so a
plan may produce the build side in any order."""


# ---------------------------------------------------------------------------
# Join build side — which input is materialised and indexed
# ---------------------------------------------------------------------------


struct JoinBuildSide(
    Copyable, Equatable, ImplicitlyCopyable, Movable, Writable
):
    """Which of a join's two logical sides is hashed into the table.

    A cost decision only: either way a join returns the same rows in the
    caller's column order and field names. That takes two things: `physical`
    restates the kind for the probe, and `_assemble` and
    `_output_dtype` use this to put the logical left side first.

    Unlike `JoinKind` it has no `@implicit` constructor. It sits beside a bare
    `UInt8 strictness`, and `BUILD_LEFT` and `JOIN_ALL` are both 0, so an
    implicit conversion would let the two be swapped silently.
    """

    var code: UInt8
    """The wire value. Stable — `expr.logical.Join` stores it."""

    def __init__(out self, code: UInt8):
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        return self.code != other.code

    def physical(self, kind: JoinKind) raises -> JoinKind:
        """`kind` restated against `(build, probe)`, the terms a probe reads
        it in: the kind itself when the left side is built, its mirror
        otherwise.

        Raises:
            Error: through `JoinKind.mirror`, for a kind with no mirror.
        """
        if self == BUILD_LEFT:
            return kind
        else:
            return kind.mirror()

    def write_to[W: Writer](self, mut writer: W):
        if self == BUILD_LEFT:
            writer.write("build=left")
        elif self == BUILD_RIGHT:
            writer.write("build=right")
        else:
            writer.write("build side ", self.code)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


comptime BUILD_LEFT = JoinBuildSide(0)
"""Index the left input and stream the right; the default."""

comptime BUILD_RIGHT = JoinBuildSide(1)
"""Index the right input, stream the left. Same answer, different cost."""


comptime _PARALLEL_GROUP_ROWS = 65_536
"""Build rows from which grouping them by code runs on several threads."""

comptime _GROUP_RANGES = 256
"""How many ranges of codes the parallel grouping deals out to threads."""


@no_inline
def _group_by_code_parallel(
    codes: Int32Array,
    num_codes: Int,
    mut offsets: Buffer[mut=True],
    mut rows: Buffer[mut=True],
    ctx: ExecContext,
) raises:
    """Rows `[0, len(codes))` grouped by code into `rows`, ascending within
    a code, with `offsets[c]` where code `c`'s rows start and
    `offsets[num_codes]` their total — what the serial counting sort answers.

    The codes are cut into contiguous ranges. Each thread counts its stripe
    of rows per range, and deals them out by range into the slots a prefix
    sum gave it, so each range's rows stay in row order; then each range is
    counted and placed on its own thread, into its own slice."""
    var n = len(codes)
    var ranges = min(_GROUP_RANGES, num_codes)
    var threads = max(1, min(ctx.resolved_num_threads(), n // 4096))
    var stripe = ceildiv(n, threads)
    var counts = Buffer.alloc_zeroed[DType.int64](threads * ranges)
    var cv = counts.view[DType.int64](0, threads * ranges)
    var starts = Buffer.alloc_uninit[DType.int64](ranges + 1)
    var sv = starts.view[DType.int64](0, ranges + 1)
    var dealt = Buffer.alloc_uninit[int32.native](n)
    var dv = dealt.view[int32.native](0, n)
    var ov = offsets.view[int32.native](0, num_codes + 1)
    var rv = rows.view[int32.native](0, n)
    var phase = Buffer.alloc_zeroed[DType.int64](1)
    var pv = phase.view[DType.int64](0, 1)

    # One body for the three steps, so the pool is instantiated once.
    def step(i: Int) {imm}:
        var which = pv.unsafe_get(0)
        if which < 2:
            var base = i * ranges
            for row in range(i * stripe, min(n, (i + 1) * stripe)):
                var r = Int(codes.unsafe_get(row)) * ranges // num_codes
                var at = Int(cv.unsafe_get(base + r))
                if which == 1:
                    dv.unsafe_set(at, Int32(row))
                cv.unsafe_set(base + r, Int64(at + 1))
        else:
            var lo = ceildiv(i * num_codes, ranges)
            var hi = ceildiv((i + 1) * num_codes, ranges)
            var first = Int(sv.unsafe_get(i))
            var last = Int(sv.unsafe_get(i + 1))
            var at = List[Int](length=hi - lo, fill=0)
            for p in range(first, last):
                at[Int(codes.unsafe_get(Int(dv.unsafe_get(p)))) - lo] += 1
            var pos = first
            for c in range(hi - lo):
                ov.unsafe_set(lo + c, Int32(pos))
                var count = at[c]
                at[c] = pos
                pos += count
            for p in range(first, last):
                var row = dv.unsafe_get(p)
                var c = Int(codes.unsafe_get(Int(row))) - lo
                rv.unsafe_set(at[c], row)
                at[c] += 1

    ctx.run(threads, step)
    # Range-major, thread-minor: thread t's rows of range r follow every
    # earlier thread's, which is what keeps each range in row order.
    var running = 0
    for r in range(ranges):
        sv.unsafe_set(r, Int64(running))
        for t in range(threads):
            var count = Int(cv.unsafe_get(t * ranges + r))
            cv.unsafe_set(t * ranges + r, Int64(running))
            running += count
    sv.unsafe_set(ranges, Int64(running))
    pv.unsafe_set(0, 1)
    ctx.run(threads, step)
    pv.unsafe_set(0, 2)
    ctx.run(ranges, step)
    ov.unsafe_set(num_codes, Int32(n))
    _ = counts^
    _ = starts^
    _ = dealt^
    _ = phase^


@fieldwise_init
struct JoinIndex(Copyable, Movable, Sized):
    """Which build row pairs with which probe row, one entry per output row.

    A named pair rather than `Tuple[Int32Array, Int32Array]`, which is what this
    was. The tuple spelling made every consumer write `pairs[0]` and
    `pairs[1]`, and nothing distinguished them -- reading the build side as the
    probe side is a silent wrong answer, and `_assemble` gathers the two sides
    from different arrays, so getting them the wrong way round produces a
    plausible-looking result with the columns crossed.

    A null entry on either side means "no match": an outer join emits an
    unmatched build row as `(row, null)` and an unmatched probe row as
    `(null, row)`.
    """

    var build: Int32Array
    """Row indices into the build side, whichever input that was."""
    var probe: Int32Array
    """Row indices into the probe side."""

    def __len__(self) -> Int:
        return len(self.build)


# ---------------------------------------------------------------------------
# JoinHashTable and HashJoin
# ---------------------------------------------------------------------------


struct BuildMatches(Movable):
    """Which build keys some probe row has matched: one bit per code of a
    `JoinHashTable`.

    A probe split across workers gives each worker its own and `merge`s them
    once every probe row has been seen, so no bit is shared while it is
    written."""

    var _bits: Bitmap[mut=True]

    def __init__(out self, num_codes: Int):
        """No key matched yet, of `num_codes`."""
        self._bits = Bitmap.alloc_zeroed(max(num_codes, 1))

    def mark(mut self, codes: Int32Array):
        """Record the code of every probe row; a `-1` matched nothing."""
        for i in range(len(codes)):
            var code = Int(codes.unsafe_get(i))
            if code >= 0:
                self._bits.unsafe_set(code)

    def merge(mut self, other: Self) raises:
        """Every key either has matched."""
        self._bits = self._bits.view() | other._bits.view()

    def __getitem__(self, code: Int) -> Bool:
        """Whether key `code` has matched."""
        return self._bits.unsafe_test(code)


struct JoinHashTable[Hash: Hasher = KeyHash](Movable):
    """A hash join's build side: every distinct build key, and the rows
    holding it.

    A ``DictionaryEncoder`` gives each distinct key a code — exactly, so rows
    sharing a code share their key — and the rows of each code are stored
    contiguously, so a probe key's build rows are one range::

        _offsets[code] .. _offsets[code + 1]   ->  range in _rows
        _rows[j]                               ->  build-side row

    **A build whose keys are all distinct has no offsets.** Each code then
    holds one row, ``_rows[code]``. A join on a key takes this path.

    A probe looks its keys up in the same encoder, so a candidate *is* a match
    — there is nothing left to compare — with one exception, which ``=``
    makes: the encoder holds a NULL key equal to another, and ``=`` matches no
    NULL at all, so a probe key holding one matches nothing. A build key
    holding one is encoded like any other and is then never looked up.
    """

    var _keys: DictionaryEncoder[Self.Hash]
    var _codes: Int32Array
    """The code of every build row."""
    var _rows: Int32Array
    """Build rows, grouped by code."""
    var _offsets: Buffer[mut=True]
    """One past each code's range in ``_rows``; empty when every code holds
    one row."""

    def __init__(
        out self, keys: List[DynArray], var ctx: ExecContext = ExecContext()
    ) raises:
        """Index a build side by its key columns ``keys``."""
        var types = List[DynType](capacity=len(keys))
        for ref key in keys:
            types.append(key.dtype())
        var grouping = ctx.copy()
        self._keys = DictionaryEncoder[Self.Hash](types^, ctx^)
        var num_rows = len(keys[0])
        var placed = self._keys.encode(keys, expected_new=num_rows)
        ref codes = placed.ids
        self._codes = codes.copy()
        var num_codes = len(self._keys)
        if num_codes == num_rows:
            # Every row introduced its own code, so the rows that introduced
            # the codes, in code order, are the rows by code.
            self._rows = placed.firsts.copy()
            self._offsets = Buffer.alloc_uninit(0)
        else:
            # Several rows share a code: a counting sort by code.
            self._offsets = Buffer.alloc_uninit[int32.native](num_codes + 1)
            var rows = Buffer.alloc_uninit[int32.native](max(num_rows, 1))
            if grouping.worth_parallel(num_rows, _PARALLEL_GROUP_ROWS):
                _group_by_code_parallel(
                    codes, num_codes, self._offsets, rows, grouping
                )
            else:
                var at = List[Int](length=num_codes, fill=0)
                for i in range(num_rows):
                    at[Int(codes.unsafe_get(i))] += 1
                self._offsets.unsafe_set[int32.native](0, 0)
                for c in range(num_codes):
                    self._offsets.unsafe_set[int32.native](
                        c + 1,
                        self._offsets.unsafe_get[int32.native](c)
                        + Int32(at[c]),
                    )
                    at[c] = Int(self._offsets.unsafe_get[int32.native](c))
                for i in range(num_rows):
                    var c = Int(codes.unsafe_get(i))
                    rows.unsafe_set[int32.native](at[c], Int32(i))
                    at[c] += 1
            self._rows = Int32Array(
                length=num_rows,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=rows^.to_immutable(),
            )

    def is_partitioned(self) -> Bool:
        """Whether the build keys sit on radix-partitioned tables."""
        return self._keys.is_partitioned()

    def num_codes(self) -> Int:
        """How many distinct build keys there are — what a `BuildMatches`
        for this table is sized to."""
        return len(self._keys)

    def candidates(
        self,
        keys: List[DynArray],
        single_match: Bool = False,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> JoinIndex:
        """Every ``(build_row, probe_row)`` whose keys are equal under ``=`` —
        at most one per probe row when ``single_match`` (``JOIN_ANY``,
        semi-join): ``expand`` over ``codes``."""
        return self.expand(self.codes(keys, ctx), single_match, ctx)

    def codes(
        self, keys: List[DynArray], ctx: ExecContext
    ) raises -> Int32Array:
        """The build key every probe row's key equals under ``=``, by code, or
        ``-1`` where none does — a NULL key included. Looked up on ``ctx``."""
        var codes = self._keys.lookup(keys, ctx)
        for ref key in keys:
            if key.null_count() != 0:
                codes = Self._unmatched_where_null(codes, key)
        return codes^

    def expand(
        self, codes: Int32Array, single_match: Bool, ctx: ExecContext
    ) raises -> JoinIndex:
        """Every ``(build_row, probe_row)`` the probe rows' ``codes`` name — at
        most one per probe row when ``single_match``.

        Two passes over the probe rows, striped over ``ctx``: one counts each
        stripe's pairs, a prefix sum turns the counts into where each stripe
        writes, and the other writes them — a large probe expands in parallel
        into one pair of arrays, with nothing to merge.
        """
        var num_rows = len(codes)
        var stripes = ctx.stripe_workers(num_rows)
        var at = List[Int](length=stripes + 1, fill=0)

        @always_inline
        def count(wid: Int, start: Int, end: Int) {mut at, imm}:
            var pairs = 0
            for row in range(start, end):
                var span = self._span(codes, row, single_match)
                pairs += span[1] - span[0]
            at[wid + 1] = pairs

        ctx.stripe(num_rows, count)
        for w in range(stripes):
            at[w + 1] += at[w]
        var total = at[stripes]
        var left = Buffer.alloc_uninit[int32.native](max(total, 1))
        var right = Buffer.alloc_uninit[int32.native](max(total, 1))
        var build_rows = left.view[int32.native](0, total)
        var probe_rows = right.view[int32.native](0, total)

        @always_inline
        def write(
            wid: Int, start: Int, end: Int
        ) {mut build_rows, mut probe_rows, imm}:
            var p = at[wid]
            for row in range(start, end):
                var span = self._span(codes, row, single_match)
                for j in range(span[0], span[1]):
                    build_rows.store[1](p, self._rows.unsafe_get(j))
                    probe_rows.store[1](p, Int32(row))
                    p += 1

        ctx.stripe(num_rows, write)
        return JoinIndex(
            build=Int32Array(
                length=total,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=left^.to_immutable(),
            ),
            probe=Int32Array(
                length=total,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=right^.to_immutable(),
            ),
        )

    @staticmethod
    def _unmatched_where_null(
        codes: Int32Array, key: DynArray
    ) raises -> Int32Array:
        """``codes`` with ``-1`` wherever ``key`` is NULL — ``=`` matches no
        NULL, not even another, where the encoder's identity does."""
        var n = len(codes)
        var valid = NotNullKernel.dispatch(key)
        var bits = valid.values()
        var src = codes.values()
        var buf = Buffer.alloc_uninit[int32.native](max(n, 1))
        var out = buf.view[int32.native](0, n)
        for i in range(n):
            out.store[1](i, src.load[1](i) if bits.test(i) else Int32(-1))
        return Int32Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=buf^.to_immutable()
        )

    @always_inline
    def _span(
        self, codes: Int32Array, row: Int, single_match: Bool
    ) -> Tuple[Int, Int]:
        """Where probe ``row``'s build rows sit in ``_rows`` — an empty range
        when its key has no code (``-1``)."""
        var code = Int(codes.unsafe_get(row))
        if code < 0:
            return (0, 0)
        var start = self._first(code)
        if single_match or len(self._offsets) == 0:
            return (start, start + 1)
        return (start, Int(self._offsets.unsafe_get[int32.native](code + 1)))

    def rows(
        self,
        matches: BuildMatches,
        strictness: UInt8,
        matched: Bool,
        start: Int,
        end: Int,
    ) raises -> Int32Array:
        """The build rows in ``[start, end)``, ascending, that have matched
        (``matched``) or have not.

        A row has matched when its key has — under ``JOIN_ANY`` only the
        first row of each key, the one a probe row was paired with."""
        var out = Int32Builder(capacity=max(end - start, 0))
        for row in range(start, end):
            var code = Int(self._codes.unsafe_get(row))
            var hit = matches[code]
            if hit and strictness == JOIN_ANY:
                hit = Int(self._rows.unsafe_get(self._first(code))) == row
            if hit == matched:
                out.append(Int32(row))
        return out.finish()

    @always_inline
    def _first(self, code: Int) -> Int:
        """Where ``code``'s first build row sits in ``_rows``."""
        if len(self._offsets) == 0:
            return code
        return Int(self._offsets.unsafe_get[int32.native](code))


struct HashJoin[Hash: Hasher = KeyHash]:
    """Hash join over ``JoinHashTable``.

    Build phase: encode the build side's key columns and group its rows by
    code. Probe phase: look the probe side's keys up — an exact lookup, so
    every candidate is a match — then add the unmatched rows the join kind
    asks for and gather the output columns.

    Everything here is in build/probe terms. Which of the caller's inputs was
    built is the `JoinBuildSide` passed to `probe`, which restates the
    caller's kind and puts the columns back in the caller's order.

    **The predicate is SQL's ``=``**: the encoder's key identity — NaN equal
    to NaN, as ``WHERE a.k = b.k`` agrees — less NULL, which matches nothing,
    not even another NULL (``JoinHashTable``).

    **One path.** Parallelism belongs to the pieces: the encoder places a large
    and distinct build side on radix-partitioned tables and stripes a batch
    big enough to share out, and expanding candidates stripes over
    ``ExecContext.for_batch``.
    """

    var _ctx: ExecContext
    """How this join executes — held whole rather than destructured to a worker
    count. It used to be a bare `_num_threads: Int`, which five internal sites
    then rebuilt into `ExecContext.parallel(n)`; every one of those
    silently dropped the caller's GPU device, since that factory sets
    `device=None`."""
    var _build: StructArray
    """The build side, whose rows `_table` indexes."""
    var _build_columns: List[DynArray]
    """`_build`'s columns, each narrowed to its window, gathered from by every
    probe."""
    var _table: JoinHashTable[Self.Hash]

    def __init__(
        out self,
        build: StructArray,
        key_indices: List[Int],
        var ctx: ExecContext = ExecContext(),
    ) raises:
        """The build phase: index ``build`` by the columns at
        ``key_indices``.

        Args:
            build: The side to index.
            key_indices: Its key columns.
            ctx: How to execute. ``ExecContext.serial()`` keeps every step on
                the calling thread; ``.parallel(n)`` lets the build place its
                keys on radix-partitioned tables and each step stripe across
                ``n`` workers when its input is large enough; ``.parallel()`` /
                ``.auto()`` picks ``num_physical_cores()``.
        """
        self._table = JoinHashTable[Self.Hash](
            Self._key_columns(build, key_indices), ctx.copy()
        )
        self._build = build.copy()
        self._build_columns = build.flatten()
        self._ctx = ctx^

    def probe(
        self,
        data: StructArray,
        key_indices: List[Int],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
        build_side: JoinBuildSide = BUILD_LEFT,
    ) raises -> StructArray:
        """Probe `data`, the whole probe side, and assemble the result.

        `kind` is the caller's, read against its own `(left, right)`, and
        `build_side` says which of those was built. The rows come in one
        order: the matched pairs in probe order, then the unmatched build
        rows, then the unmatched probe rows.
        """
        var keys = Self._key_columns(data, key_indices)
        var ctx = self._ctx.for_batch(len(data))
        var physical = build_side.physical(kind)
        var codes = self._table.codes(keys, ctx)
        # Only the kinds that emit build rows by whether they matched read
        # the marks.
        var marked = physical.needs_left_matches()
        var matches = BuildMatches(self._table.num_codes() if marked else 0)
        if marked:
            matches.mark(codes)
        var n = self._build.length
        var index: JoinIndex
        if physical == JOIN_SEMI or physical == JOIN_ANTI:
            # Whether a build row has a match is not which match a probe row
            # keeps: SEMI and ANTI over the build side read every match,
            # JOIN_ANY or not.
            var rows = self._table.rows(
                matches, JOIN_ALL, physical == JOIN_SEMI, 0, n
            )
            index = JoinIndex(rows.copy(), nulls(len(rows), int32))
        elif physical == JOIN_RIGHT_SEMI or physical == JOIN_RIGHT_ANTI:
            var rows = Self._probe_rows(codes, physical == JOIN_RIGHT_SEMI)
            index = JoinIndex(nulls(len(rows), int32), rows^)
        else:
            var parts: List[JoinIndex] = [
                self._table.expand(codes, strictness == JOIN_ANY, ctx)
            ]
            if physical.emits_unmatched_left():
                var rows = self._table.rows(matches, strictness, False, 0, n)
                parts.append(JoinIndex(rows.copy(), nulls(len(rows), int32)))
            if physical.emits_unmatched_right():
                var rows = Self._probe_rows(codes, False)
                parts.append(JoinIndex(nulls(len(rows), int32), rows^))
            index = Self._concat(parts)
        return self._assemble(data, index, kind, build_side, self._ctx)

    def probe(
        self,
        data: StructArray,
        key_indices: List[Int],
        kind: JoinKind,
        strictness: UInt8,
        build_side: JoinBuildSide,
        mut matches: BuildMatches,
        ctx: ExecContext,
    ) raises -> StructArray:
        """Probe one part of the probe side, on `ctx`, recording in `matches`
        which build keys it matched.

        What a part answers alone: the matched pairs in probe order, then —
        for RIGHT and FULL — its own unmatched probe rows, and for the
        existence filters on the probe side, its own rows. The build rows
        LEFT, FULL, SEMI and ANTI emit are only known once every part has
        been probed, and come from `unmatched` over the merged `matches`.
        """
        var keys = Self._key_columns(data, key_indices)
        var batch_ctx = ctx.for_batch(len(data))
        var physical = build_side.physical(kind)
        var codes = self._table.codes(keys, batch_ctx)
        if physical.needs_left_matches():
            matches.mark(codes)
        var index: JoinIndex
        if physical == JOIN_SEMI or physical == JOIN_ANTI:
            index = JoinIndex(nulls(0, int32), nulls(0, int32))
        elif physical == JOIN_RIGHT_SEMI or physical == JOIN_RIGHT_ANTI:
            var rows = Self._probe_rows(codes, physical == JOIN_RIGHT_SEMI)
            index = JoinIndex(nulls(len(rows), int32), rows^)
        else:
            index = self._table.expand(codes, strictness == JOIN_ANY, batch_ctx)
            if physical.emits_unmatched_right():
                var rows = Self._probe_rows(codes, False)
                var parts: List[JoinIndex] = [
                    index^,
                    JoinIndex(nulls(len(rows), int32), rows^),
                ]
                index = Self._concat(parts)
        count["join.probe_rows"](len(data))
        count["join.matches"](len(index))
        return self._assemble(data, index, kind, build_side, ctx)

    def mark(
        self,
        data: StructArray,
        key_indices: List[Int],
        mut matches: BuildMatches,
        ctx: ExecContext,
    ) raises:
        """Record in `matches` the build keys `data` matches, and nothing
        else: all a part of the probe side contributes to SEMI and ANTI,
        whose rows come from `unmatched`."""
        var keys = Self._key_columns(data, key_indices)
        matches.mark(self._table.codes(keys, ctx.for_batch(len(data))))

    def unmatched(
        self,
        matches: BuildMatches,
        like: StructArray,
        kind: JoinKind,
        strictness: UInt8,
        build_side: JoinBuildSide,
        start: Int,
        end: Int,
        ctx: ExecContext,
    ) raises -> StructArray:
        """The rows a split probe owes after its last part: the build rows in
        `[start, end)` that LEFT, FULL and ANTI emit unmatched and SEMI emits
        matched, in ascending order, by the merged `matches`. Any other kind
        owes none.

        `like` is any part of the probe side — empty will do — for its
        columns' types; they are all null here."""
        var physical = build_side.physical(kind)
        var rows: Int32Array
        # Whether a build row has a match is not which match a probe row
        # keeps: SEMI and ANTI read every match, JOIN_ANY or not.
        if physical == JOIN_SEMI:
            rows = self._table.rows(matches, JOIN_ALL, True, start, end)
        elif physical == JOIN_ANTI:
            rows = self._table.rows(matches, JOIN_ALL, False, start, end)
        elif physical.emits_unmatched_left():
            rows = self._table.rows(matches, strictness, False, start, end)
        else:
            rows = nulls(0, int32)
        # The probe side is gathered at null indices, so `like` may be empty.
        var index = JoinIndex(rows.copy(), nulls(len(rows), int32))
        return self._assemble(like, index, kind, build_side, ctx)

    @staticmethod
    def _key_columns(
        data: StructArray, indices: List[Int]
    ) raises -> List[DynArray]:
        """The key columns, each windowed to ``data``'s slice. Only their
        dtypes and values matter — a ``dept`` key joins a ``did`` key, and a
        required column a nullable one."""
        var keys = List[DynArray](capacity=len(indices))
        for i in indices:
            keys.append(data.field(i))
        return keys^

    @staticmethod
    def _probe_rows(codes: Int32Array, matched: Bool) raises -> Int32Array:
        """The probe rows, ascending, whose key matched (``matched``) or did
        not."""
        var out = Int32Builder(capacity=len(codes))
        for row in range(len(codes)):
            if (Int(codes.unsafe_get(row)) >= 0) == matched:
                out.append(Int32(row))
        return out.finish()

    @staticmethod
    def _concat(parts: List[JoinIndex]) raises -> JoinIndex:
        """Each part's pairs, in order."""
        var build = List[Int32Array](capacity=len(parts))
        var probe = List[Int32Array](capacity=len(parts))
        for i in range(len(parts)):
            build.append(parts[i].build.copy())
            probe.append(parts[i].probe.copy())
        return JoinIndex(Int32Builder.concat(build), Int32Builder.concat(probe))

    def num_codes(self) -> Int:
        """How many distinct build keys there are — what a `BuildMatches`
        for this join is sized to."""
        return self._table.num_codes()

    def num_build_rows(self) -> Int:
        return self._build.length

    def built_parallel(self) -> Bool:
        """Whether the build placed the build side on radix-partitioned tables.

        Exposed so a test can prove it exercised the partitioned lookups rather
        than passing vacuously on the single table — the two are supposed to
        be indistinguishable in their results, which is exactly what makes an
        accidental fallback invisible.
        """
        return self._table.is_partitioned()

    def _output_dtype(
        self,
        probe: StructArray,
        kind: JoinKind,
        build_side: JoinBuildSide = BUILD_LEFT,
    ) -> DynType:
        """The output struct dtype for a join result: the logical left side's
        fields, then the right side's, whichever side was built.
        """
        if build_side == BUILD_LEFT:
            return Self._joined_dtype(
                self._build.dtype.as_struct().fields,
                probe.dtype.as_struct().fields,
                kind,
            )
        else:
            return Self._joined_dtype(
                probe.dtype.as_struct().fields,
                self._build.dtype.as_struct().fields,
                kind,
            )

    @staticmethod
    def _joined_dtype(
        left: List[Field], right: List[Field], kind: JoinKind
    ) -> DynType:
        """Left fields then right fields, a right field suffixed `_right` where
        its name collides. The suffix follows the logical right side, not the
        probe side, and a renamed field keeps its nullability and metadata.
        """
        var fields = List[Field]()
        if kind.emits_left_columns():
            for ref f in left:
                fields.append(f.copy())

        if kind.emits_right_columns():
            # Only the fields emitted above can be collided with, which is why
            # this reads `fields` rather than `left`: a kind that emits one
            # side alone has no collisions to resolve.
            var emitted = len(fields)
            for ref f in right:
                var collides = False
                for i in range(emitted):
                    if fields[i].name == f.name:
                        collides = True
                        break
                if collides:
                    fields.append(
                        Field(
                            f.name + "_right",
                            f.dtype.copy(),
                            f.nullable,
                            f.metadata.copy(),
                        )
                    )
                else:
                    fields.append(f.copy())

        return struct_(fields^)

    def _gather(
        self,
        columns: List[DynArray],
        indices: Int32Array,
        mut into: List[DynArray],
        ctx: ExecContext,
    ) raises:
        """One ``take`` per column, appended in order, on ``ctx``.

        Each per-column ``take`` may fan its SIMD gather loop across
        ``ctx``'s workers, and decides per column whether it is big enough to
        stripe (its own grain threshold inside ``apply``).
        """
        for c in range(len(columns)):
            into.append(TakeKernel.dispatch(columns[c].copy(), indices, ctx))

    def _assemble(
        self,
        probe: StructArray,
        pairs: JoinIndex,
        kind: JoinKind,
        build_side: JoinBuildSide,
        ctx: ExecContext,
    ) raises -> StructArray:
        """Gather the output columns in the caller's left-then-right order.

        Each side is gathered with its own indices, `pairs.build` or
        `pairs.probe`, so `build_side` only decides which goes first. The row
        count is `len(pairs)`, which holds even when a side has no columns.
        """
        ref build = self._build_columns
        var probe_cols = probe.flatten()
        var out_cols = List[DynArray]()

        if kind.emits_left_columns():
            if build_side == BUILD_LEFT:
                self._gather(build, pairs.build, out_cols, ctx)
            else:
                self._gather(probe_cols, pairs.probe, out_cols, ctx)

        if kind.emits_right_columns():
            if build_side == BUILD_LEFT:
                self._gather(probe_cols, pairs.probe, out_cols, ctx)
            else:
                self._gather(build, pairs.build, out_cols, ctx)

        return StructArray(
            dtype=self._output_dtype(probe, kind, build_side),
            length=len(pairs),
            nulls=0,
            offset=0,
            bitmap=None,
            children=out_cols^,
        )


# ---------------------------------------------------------------------------
# HashJoinKernel — the public entry point
# ---------------------------------------------------------------------------


struct HashJoinKernel(Kernel):
    """Equijoin of two `StructArray`s on positional key columns, through
    `HashJoin`."""

    comptime name = "hash_join"

    @staticmethod
    def apply(
        left: StructArray,
        right: StructArray,
        left_on: List[Int],
        right_on: List[Int],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
        build_side: JoinBuildSide = BUILD_LEFT,
        ctx: ExecContext = ExecContext.auto(),
    ) raises -> StructArray:
        """Equijoin two StructArrays on positional key column indices.

        ``build_side`` decides which input is materialised and indexed; it decides
        nothing else. The answer — field names, field order and rows — is the same
        either way, which is the property that makes "index the smaller side" a
        cost decision an optimizer may take on its own.

        Args:
            left: The left side as a StructArray (one child per column).
            right: The right side as a StructArray (one child per column).
            left_on: Positional column indices in ``left`` to join on.
            right_on: Positional column indices in ``right`` to join on.
            kind: Join direction (JOIN_INNER, JOIN_LEFT, JOIN_RIGHT, JOIN_FULL,
                  JOIN_SEMI, JOIN_ANTI, JOIN_RIGHT_SEMI, JOIN_RIGHT_ANTI).
            strictness: JOIN_ALL (default) or JOIN_ANY.
            build_side: BUILD_LEFT (default) indexes ``left`` and streams
                ``right``; BUILD_RIGHT does the opposite. Row *order* within the
                result follows the probe side and so does change; the multiset of
                rows does not.
            ctx: How to execute. ``.auto()`` (default) picks
                ``num_physical_cores()`` workers; ``.serial()`` keeps every step on
                the calling thread; ``.parallel(n)`` lets each step use ``n``
                workers when its input is large enough. Any GPU device on the
                context survives into the join's internal dispatches.

        Returns:
            Output StructArray:
            * INNER/LEFT/RIGHT/FULL: left columns + right columns.
            * SEMI/ANTI: left columns only.
            * RIGHT_SEMI/RIGHT_ANTI: right columns only.
        """
        if len(left_on) != len(right_on):
            raise Self.error[InvalidError]("len(left_on) != len(right_on)")
        if not kind.is_supported():
            # CROSS, MARK and SINGLE have constants but no implementation. They used
            # to fall through to the outer-join arm, and MARK additionally built a
            # result whose declared schema had more fields than it had columns.
            raise Self.error[NotImplementedError](
                t"join kind '{kind}' is not implemented"
            )

        if build_side == BUILD_LEFT:
            var join = HashJoin(left, left_on, ctx.copy())
            return join.probe(right, right_on, kind, strictness, build_side)
        else:
            var join = HashJoin(right, right_on, ctx.copy())
            return join.probe(left, left_on, kind, strictness, build_side)
