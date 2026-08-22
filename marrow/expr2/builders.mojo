"""`col` and `lit` — the one surface that spans both lanes.

Every other module in `expr2` belongs to exactly one lane. This one cannot, and
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

from .`comptime`.leaves import Column, Literal
from .runtime.values import RuntimeValue, column, literal
from ..dtypes import FloatingType, NumericType
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
