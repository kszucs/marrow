"""The public expression builders — `col`, `lit`, `if_else`, `coalesce`,
`case_when`.

**The whole overload set lives here, both lanes together, and that is the
point.** `col("a", int64)` builds a fused AOT leaf and `col("a")` builds a
runtime-lane `DynValue`; they are deliberately the same verb, one argument
apart. An overload set cannot span modules — splitting it makes
`marrow/expr/__init__.mojo` re-export `col` and `lit` from two places and Mojo
answers *"importing 'col' from multiple modules is deprecated"*, which is what
reverted the earlier attempt (backlog L2). Keeping the set whole in one module
above both lanes is what lets `values.mojo` stop importing `dynamic.mojo` for
the untyped half without renaming anything.

So this module sits above both lanes and below `relations.mojo`: it imports
`values` and `dynamic`, and nothing imports it except the plan layer and the
package `__init__`.
"""

from ..dtypes import (
    FloatingType,
    ListLikeType,
    NumericType,
    StringLikeType,
    StringType,
    TemporalType,
)
from ..scalars import PrimitiveScalar
from .dynamic import DynValue
from .values import (
    ListColumn,
    NumericColumn,
    NumericLiteral,
    StringColumn,
    StringLiteral,
    TemporalColumn,
)


# ---------------------------------------------------------------------------
# The AOT lane — a dtype argument names the fused leaf's comptime type.
# ---------------------------------------------------------------------------
def col[T: NumericType](var name: String, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column by name — `col("a", int64)`."""
    return NumericColumn[T](name^)


def col[T: StringLikeType](var name: String, dtype: T) -> StringColumn[T]:
    """Reference a string column by name — `col("s", string)`."""
    return StringColumn[T](name^)


def col[T: ListLikeType](var name: String, dtype: T) -> ListColumn[T]:
    """Reference a list column by name — `col("l", list_(int64))`."""
    return ListColumn[T](name^)


def col[T: TemporalType](var name: String, dtype: T) -> TemporalColumn[T]:
    """Reference a temporal column by name — `col("ts", timestamp(second))`."""
    return TemporalColumn[T](name^)


def lit[T: NumericType](value: Int, dtype: T) -> NumericLiteral[T]:
    """An integral constant — `lit(10, int64)`."""
    return NumericLiteral[T](Scalar[T.native](value))


def lit[T: FloatingType](value: Float64, dtype: T) -> NumericLiteral[T]:
    """A fractional constant — `lit(3.5, float64)`.

    Without this overload the only spelling took an `Int`, so `lit(3.5,
    float64)` was unrepresentable: it truncated to 3."""
    return NumericLiteral[T](Scalar[T.native](value))


def lit(value: String) -> StringLiteral[StringType]:
    """A string constant — `lit("suffix")`. Same verb as the numeric ones; the
    argument type picks the literal."""
    return StringLiteral[StringType](value)


# ---------------------------------------------------------------------------
# The runtime lane — no dtype argument, so the node is a `DynValue`.
# ---------------------------------------------------------------------------
def col(var name: String) -> DynValue:
    """Reference a column whose dtype is not known here — `col("a")`.

    Same verb as `col(name, dtype)`, one argument shorter, and that argument is
    the whole difference between the lanes: with a dtype the fused
    `NumericColumn[T]` leaf is built, without one the column's type is found on
    the batch and this is a runtime-lane node."""
    return DynValue.column(name^)


def lit[T: NumericType](value: Scalar[T.native]) -> DynValue:
    """A scalar constant for the runtime lane — `lit[Int64Type](3)`.

    A literal always knows its type where it is written; what is erased here is
    the *expression*, so the value goes in as a `DynScalar` payload."""
    return DynValue.literal(PrimitiveScalar[T](value))


def if_else(cond: DynValue, then_: DynValue, else_: DynValue) -> DynValue:
    """Element-wise conditional — the single-branch `CaseWhen`."""
    return DynValue.if_else(cond, then_, else_)


def coalesce(values: List[DynValue]) raises -> DynValue:
    """First non-null across N expressions.

    Folds the binary `Coalesce` node rather than introducing an n-ary one:
    `coalesce(a, b, c)` is `Coalesce(Coalesce(a, b), c)`, which is the same
    result because the operation is associative and null-propagating."""
    if len(values) == 0:
        raise Error("coalesce: needs at least one value")
    var acc = values[0].copy()
    for k in range(1, len(values)):
        acc = acc.coalesce(values[k])
    return acc^


def case_when(
    conditions: List[DynValue],
    values: List[DynValue],
    var else_: Optional[DynValue] = None,
) raises -> DynValue:
    """Multi-branch `CASE WHEN`, built by nesting the single-branch `CaseWhen`.

    `conditions[k]` selects `values[k]` for the first branch that is
    valid-and-true. Nesting right-to-left gives first-match-wins, which is what
    the interpreter's interleaved-args form computed."""
    if len(conditions) != len(values):
        raise Error("case_when: len(conditions) != len(values)")
    if len(conditions) == 0:
        raise Error("case_when: needs at least one branch")
    # No `else_` means "null where nothing matched". `CaseWhen` always has a
    # third operand, so the null is built from an existing node rather than a new
    # one: `Nullif(v, v)` is `v` with every element equal to itself removed — an
    # all-null column of the right dtype.
    var acc = else_.value().copy() if else_ else values[0].nullif(values[0])
    for k in range(len(conditions) - 1, -1, -1):
        acc = DynValue.if_else(conditions[k], values[k], acc)
    return acc^
