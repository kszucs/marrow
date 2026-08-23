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

from .`comptime`.numeric import CaseWhen
from .`comptime`.core import BoolValue, ListValue, NumericValue
from .`comptime`.leaves import (
    BoolColumn,
    Column,
    ListColumn,
    ListLength,
    TemporalColumn,
    Literal,
    StringColumn,
    StringLiteral,
)
from .params import Param
from .runtime.values import RuntimeValue, column, literal
from ..dtypes import (
    BoolType,
    ListLikeType,
    FloatingType,
    NumericType,
    StringLikeType,
    TemporalType,
)
from ..scalars import DynScalar


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
    `expr/` shipped without it for exactly that reason and had to add it later.
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
    second. `expr/` reached the same two overloads by the same route.
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


def array_length[A: ListValue](var a: A) -> ListLength[A]:
    """The number of elements in each list. A list consumed into a numeric
    column — which is the only way a list is read, since a list element is a
    whole sub-array rather than a value a lane can hold."""
    return ListLength[A](a^)


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
    name-keyed dedup and no dtype-conflict check. `expr/` needs all three
    because it declares parameters inline at each use site.
    """
    return Param[T](name^, help^, default^)
