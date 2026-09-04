"""`col` and `lit` — the one surface that spans both lanes.

Every other module in this package belongs to exactly one lane. This one cannot, and
that is deliberate rather than a compromise: `col` is a single name whose
overloads select a lane by **what the caller knows**.

    col("amount", int64)   # a dtype is known  -> comptime lane, fuses
    col("amount")          # it is not         -> runtime lane, resolves later

The choice is therefore made by the caller's information, not by a flag they
have to reason about, and neither lane has to name the other to offer it.

The overload set cannot be split across the two lane packages. Mojo resolves an
overload from the candidates visible at one name, so putting `col(name, dtype)`
in `comptime/` and `col(name)` in `runtime/` gives two unrelated functions that
shadow rather than overload — and which one a call site got would depend on its
imports. That is the same failure the wildcard-import ban exists to prevent.
"""

from .`comptime`.aggregates import Aggregate
from .`comptime`.boolean import IsIn
from .`comptime`.numeric import CaseWhen, Maximum, Minimum
from .`comptime`.core import BoolValue, ComptimeValue, ListValue, NumericValue
from .`comptime`.nested import ArrayContains, ListLength
from .`comptime`.leaves import (
    BoolColumn,
    Column,
    ListColumn,
    Param,
    TemporalColumn,
    Literal,
    StringColumn,
    StringLiteral,
)
from .logical import (
    DynRelation,
    InMemoryTable,
    ParquetScan,
    WindowExpr,
)

from .runtime.values import RuntimeValue, column, literal
from .runtime.values import array_contains as _rt_array_contains
from .runtime.values import array_length as _rt_array_length
from .runtime.values import isin as _rt_isin
from .runtime.values import maximum as _rt_maximum
from .runtime.values import minimum as _rt_minimum
from ..arrays import DynArray
from ..dtypes import (
    BoolType,
    Int64Type,
    ListLikeType,
    FloatingType,
    NumericType,
    StringLikeType,
    TemporalType,
    int64,
)
from ..kernels.aggregate import CountFold, Fold
from ..kernels.window import (
    CumeDist,
    DenseRank,
    NTile,
    PercentRank,
    Rank,
    RowNumber,
)
from ..scalars import DynScalar
from ..schema import Schema
from ..tabular import RecordBatch


# ---------------------------------------------------------------------------
# col — a column reference
# ---------------------------------------------------------------------------
def col[T: NumericType](var name: String, dtype: T) -> Column[T]:
    """A typed column read, fused into whatever it is combined with.

    `dtype` is a *value* parameter whose type carries the information — the
    caller writes `col("a", int64)`, and `T` is deduced. It is never read at
    run time; `Column[T].Type` answers from `T`.
    """
    return Column[T](name^)


def col[T: StringLikeType](var name: String, dtype: T) -> StringColumn[T]:
    """A typed string column, fused like its numeric sibling.

    A separate overload rather than a wider bound on the numeric one: strings
    are variable-width, so `StringColumn` is a different family with a
    width-less `lane`. Mojo picks between them from `dtype`'s type, so the
    caller writes `col("name", string)` either way.
    """
    return StringColumn[T](name^)


def col[T: TemporalType](var name: String, dtype: T) -> TemporalColumn[T]:
    """A date/time/timestamp/duration column, fused like its numeric sibling.

    A separate overload for the same reason `TemporalColumn` is a separate
    struct: Mojo has no conditional conformance, so one leaf cannot be numeric
    for `int64` and temporal for `date32`. The caller still writes
    `col("d", date32)` and never sees the difference.
    """
    return TemporalColumn[T](name^)


def col(var name: String, dtype: BoolType) -> BoolColumn:
    """A boolean column, fused like its numeric and string siblings.

    Its own overload because booleans are **bit-packed**: `BoolColumn`'s lane
    loads through a `BitmapView` rather than a typed buffer, and `Column[T]` is
    bound on `NumericType` and cannot take `BoolType` — the same reason
    `PrimitiveArray[bool_]` exists nowhere in the tree.

    The leaf already existed; without this overload a fused expression could
    only reach a bool column by spelling `BoolColumn("flag")` directly, so any
    three-valued-logic test had to synthesise its operands from comparisons.
    the previous expression package shipped without it for exactly that reason
    and had to add it later.
    """
    return BoolColumn(name^)


def col[T: ListLikeType](var name: String, dtype: T) -> ListColumn[T]:
    """A list column. `list`, `large_list` and `map` are the same leaf."""
    return ListColumn[T](name^)


def col(var name: String) -> RuntimeValue:
    """An untyped column read, resolved against the schema at evaluation.

    The lane a caller falls into when the dtype is not known where the
    expression is written — a plan parsed from Python, or built from a schema
    only available at run time.
    """
    return column(name^)


# ---------------------------------------------------------------------------
# lit — a constant
# ---------------------------------------------------------------------------
def lit[T: NumericType](value: Int, dtype: T) -> Literal[T]:
    """A typed constant. Stays `Shape.scalar`, so it never materialises unless
    something asks it to.

    Takes `Int` rather than `Scalar[T.native]` because `T` is not yet resolved
    when the first argument is checked — the compiler reports *"cannot be
    converted from 'Int64' to 'Scalar[T.native]', it depends on an unresolved
    parameter 'T'"*. The dtype argument is what resolves `T`, and it is read
    second. the previous expression package reached the same two overloads by
    the same route.
    """
    return Literal[T](Scalar[T.native](value))


def lit[T: FloatingType](value: Float64, dtype: T) -> Literal[T]:
    """The floating counterpart, for the literals `Int` cannot spell."""
    return Literal[T](Scalar[T.native](value))


def lit(var value: DynScalar) -> RuntimeValue:
    """An erased constant, broadcast on evaluation."""
    return literal(value^)


def lit[T: StringLikeType](var value: String, dtype: T) -> StringLiteral[T]:
    """A typed constant string, `Shape.scalar` like its numeric sibling."""
    return StringLiteral[T](value^)


# ---------------------------------------------------------------------------
# conditionals
# ---------------------------------------------------------------------------
def if_else[
    C: BoolValue, T: NumericValue, E: NumericValue
](var cond: C, var then: T, var otherwise: E) -> CaseWhen[C, T, E]:
    """`CASE WHEN cond THEN then ELSE otherwise END`, fused.

    A null condition counts as **false** rather than producing a null — Arrow's
    `ExecArrayCaseWhen` rule and PyArrow's `pc.case_when`. A selected value
    that is itself null does stay null.
    """
    return CaseWhen[C, T, E](cond^, then^, otherwise^)


# ---------------------------------------------------------------------------
# nested — the verbs that read a list column
# ---------------------------------------------------------------------------
# A list is only ever *consumed*: `ListValue` declares no lane because a list
# element is a whole sub-array, so every verb here binds a list and hands back
# a fixed-width result. Both spellings live in this file rather than one per
# lane, for the reason `col`'s docstring gives — a split overload set shadows
# rather than overloads, and which one a call site got would depend on its
# imports.


def array_length[A: ListValue](var a: A) -> ListLength[A]:
    """The number of elements in each list. A list consumed into a numeric
    column — which is the only way a list is read, since a list element is a
    whole sub-array rather than a value a lane can hold."""
    return ListLength[A](a^)


def array_length(var a: RuntimeValue) -> RuntimeValue:
    """The runtime lane's `array_length`, so one name serves both.

    The overload existed in `runtime/values.mojo` and only there, which made
    `from marrow.expr import array_length` a comptime-only verb: a caller who
    wrote `col("xs")` had to reach past this module to measure it.
    """
    return _rt_array_length(a^)


def array_contains[
    L: ListValue, E: NumericValue
](var list: L, var elem: E) -> ArrayContains[L, E]:
    """True where `list[i]` contains the value `elem[i]` — SQL's
    `array_contains`, PyArrow has no direct equivalent.

    The search value is a *column*, one value per row, so a constant is
    `lit(3, int64)` and stays `Shape.scalar` until it is broadcast. Numeric
    element types only, which is `ArrayContainsKernel`'s limit rather than the
    node's.
    """
    return ArrayContains[L, E](list^, elem^)


def array_contains(
    var list: RuntimeValue, var elem: RuntimeValue
) -> RuntimeValue:
    """The runtime lane's `array_contains`."""
    return _rt_array_contains(list^, elem^)


# ---------------------------------------------------------------------------
# is_in — set membership
# ---------------------------------------------------------------------------
def is_in[A: ComptimeValue](var a: A, var value_set: DynArray) -> IsIn[A]:
    """`a IN (...)` — is each of `a`'s values one of `value_set`'s?

    The set is a `DynArray` fixed when the plan is built, not an operand: it is
    the same set on every row, so hashing it belongs to the node. `a` and the
    set must share a dtype, which `IsInKernel` checks and reports.

    The operand is bound on `ComptimeValue` and not on a family because
    membership is decided on the 64-bit hash alone — numeric, string, bool and
    temporal all funnel through one kernel. That also makes this a breaker:
    the subtree *under* `a` still fuses, and only the membership test runs over
    a materialised column.
    """
    return IsIn[A](a^, value_set^)


def is_in(var a: RuntimeValue, var value_set: DynArray) -> RuntimeValue:
    """The runtime lane's `is_in`, spelled `isin` inside that lane."""
    return _rt_isin(a^, value_set^)


# ---------------------------------------------------------------------------
# the row-wise extrema
# ---------------------------------------------------------------------------
# Free verbs rather than methods, and `minimum`/`maximum` rather than
# `min`/`max`, because `.min()` and `.max()` are already the *aggregates* on
# both lanes' value surfaces. NumPy draws the same line for the same reason:
# `np.minimum(a, b)` is element-wise, `np.min(a)` folds.
#
# **Every SQL-flavoured name for this operation skips nulls, and this one does
# not.** `LEAST`/`GREATEST` skip, Ibis spells them the same way, and
# `pc.min_element_wise` skips too — `ElementWiseAggregateOptions` defaults
# `skip_nulls=True`. `MinKernel` intersects validity like every other
# `BinaryNumericKernel`, so this is the propagating form and only that, which
# is what NumPy's name means as well: `minimum` propagates NaN where `fmin`
# skips it. `least`/`greatest` stay free for the skipping verbs — see
# `Minimum`.


def minimum[
    L: NumericValue, Rhs: NumericValue
](var l: L, var r: Rhs) -> Minimum[L, Rhs]:
    """The smaller of the two, per row, fused.

    The wider operand type wins and a null on either side makes the row null —
    `NumericBinary`'s rules. That null rule is what stops this being called
    `least`: SQL's `LEAST(NULL, 3)` is 3.
    """
    return Minimum[L, Rhs](l^, r^)


def minimum(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """The runtime lane's `minimum`."""
    return _rt_minimum(l^, r^)


def maximum[
    L: NumericValue, Rhs: NumericValue
](var l: L, var r: Rhs) -> Maximum[L, Rhs]:
    """The larger of the two, per row, fused."""
    return Maximum[L, Rhs](l^, r^)


def maximum(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """The runtime lane's `maximum`."""
    return _rt_maximum(l^, r^)


# ---------------------------------------------------------------------------
# param — a literal whose value arrives later
# ---------------------------------------------------------------------------
def param[
    T: NumericType
](
    var name: String,
    dtype: T,
    var help: String = String(),
    var default: Optional[Scalar[T.native]] = None,
) -> Param[T]:
    """Declare a late-bound scalar.

    **Declare once and reuse it.** Copies share the cell, so binding once is
    visible everywhere:

        var min_a = param("min-a", int64)
        t.filter(col("a", int64) > min_a).project(["m"], [min_a])

    Calling `param("min-a", int64)` twice makes two *independent* parameters
    that happen to share a name — which is why there is no registry, no
    name-keyed dedup and no dtype-conflict check. the previous expression
    package needs all three
    because it declares parameters inline at each use site.
    """
    return Param[T](name^, help^, default^)


def table(var batch: RecordBatch) raises -> DynRelation:
    """A batch already in memory, as a plan.

    The entry point the verbs need: every other verb is a method on
    `DynRelation`, so a plan has to *start* as one. Without this a caller has
    to name `InMemoryTable` and wrap it by hand — which is how every test in
    the layer ended up spelling its source.
    """
    return InMemoryTable(batch^)


def scan(var path: String, var schema: Schema) raises -> DynRelation:
    """A Parquet file, as a plan.

    The `table` of the file world: a source has to *be* a `DynRelation` before
    any verb applies, so without this a caller names `ParquetScan` and wraps it
    by hand.
    """
    return DynRelation(ParquetScan(path^, schema^))


def count_star() -> Aggregate[Fold[CountFold, Int64Type], Literal[Int64Type]]:
    """`COUNT(*)` — how many rows each group has.

    Not the same aggregate as `col("x", int64).count()`, which counts the
    *non-null* values of `x`; the two differ on any nullable column, and
    `COUNT(*)` is what ~30 of ClickBench's 43 queries ask for.

    It needs no new kernel and no new node. `CountFold` counts valid values
    and a literal is valid on every row, so the valid-count of a constant
    column *is* the row count. This is that expression, under the name SQL
    gives it, so callers stop rediscovering the trick.
    """
    return lit(1, int64).count().alias("count_star")


# ---------------------------------------------------------------------------
# Window functions with no argument
# ---------------------------------------------------------------------------
#
# Free functions rather than methods, because they read no column: `RANK()`
# answers from the ordering alone, so there is no receiver to hang them on.
# The four that *do* read a column — `lag`, `lead`, `first_value`,
# `last_value` — are `Value` trait defaults instead, and an aggregate reaches
# a frame through `Value.over`.


def row_number() raises -> WindowExpr:
    """`ROW_NUMBER()` — a distinct position per row within the partition.

    Insensitive to ties, so a non-total `ORDER BY` leaves it deterministic
    only up to the sort's stability. `rank` and `dense_rank` are the two that
    answer equally for tied rows.
    """
    return WindowExpr.of[RowNumber](None)


def rank() raises -> WindowExpr:
    """`RANK()` — tied rows share the first position of their tie, and the
    next distinct row skips the gap: `1, 2, 2, 2, 5`."""
    return WindowExpr.of[Rank](None)


def dense_rank() raises -> WindowExpr:
    """`DENSE_RANK()` — tied rows share a position and nothing is skipped:
    `1, 2, 2, 2, 3`. The only difference from `rank` is the gap."""
    return WindowExpr.of[DenseRank](None)


def percent_rank() raises -> WindowExpr:
    """`PERCENT_RANK()` — `(rank - 1) / (rows - 1)`, from 0 to 1 inclusive.

    Built on `rank`, so tied rows share a value and gaps carry through. A
    one-row partition answers 0 rather than dividing by zero.
    """
    return WindowExpr.of[PercentRank](None)


def cume_dist() raises -> WindowExpr:
    """`CUME_DIST()` — the fraction of the partition at or before this row's
    *peer group*, from just above 0 to 1.

    Counts through the whole peer group, not to this row, so tied rows share a
    value and the final group is always exactly 1.
    """
    return WindowExpr.of[CumeDist](None)


def ntile(buckets: Int) raises -> WindowExpr:
    """`NTILE(n)` — the partition split into `buckets` as evenly as it divides.

    A plain `Int` and not an expression: the bucket count is a constant of the
    window, the same as `lag`'s distance, and there is nothing per-row for a
    column to hold. The remainder goes to the earliest buckets, so 10 rows in
    3 buckets is 4, 3, 3.
    """
    return WindowExpr.of[NTile](None, buckets)
