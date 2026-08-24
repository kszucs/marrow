"""Type-level rules: what a composite node's type and shape follow from.

These are neither traits nor nodes — they are the arithmetic *on* types that a
node performs at compile time to answer what it produces. Keeping them apart
means a node states which rule it uses rather than restating the rule, and two
nodes cannot drift into disagreeing about promotion.

Only the comptime lane needs them: a `RuntimeValue` is always columnar and
learns its type from a schema, so it has nothing to compute.

ibis keeps the same file for the same reason (`ibis/expr/rules.py`), with
`highest_precedence_shape` and `highest_precedence_dtype` answering what
`widest_shape` and `promote` answer here. Its `castable` and `comparable` are
the shape of what belongs here next: a comparison node asking whether two
operand types *may* be compared is a rule, not a node's business.
"""

from std.sys import bit_width_of

from ...dtypes import NumericType
from ..logical import Shape, Value


comptime widest_shape[A: Value, B: Value] = (
    Shape.columnar if (
        A.shape == Shape.columnar or B.shape == Shape.columnar
    ) else Shape.scalar
)
"""The wider of two operands' shapes.

ibis calls this `highest_precedence_shape` (`ibis/expr/rules.py:18`). It is
**not** their `shape_like`, which means "the same shape as the argument named
X" — a different idea, and the reason this is not called that here.

A composite is columnar if *either* operand is: `col("a") + lit(1)` produces a
column, `lit(1) + lit(2)` stays a scalar. Written once because every binary
node in every family answers the same way, and restating the condition per node
is how two of them would eventually disagree.
"""


def _outranks[L: NumericType, R: NumericType]() -> Bool:
    """Does `L` win promotion over `R`?

    Two rules, and they are stated rather than encoded:

    1. A float outranks any integer, whatever the widths — `int64 + float32`
       is `float32`, because the value domain matters and the register size
       does not.
    2. Otherwise the wider one wins.

    `expr/` folded both into a single rank by adding `1000` to a float's bit
    width, which works only while no integer is 1000 bits wide and requires
    the reader to reconstruct rule 1 from the constant.
    """
    comptime l_float = L.native.is_floating_point()
    comptime r_float = R.native.is_floating_point()
    comptime if l_float != r_float:
        return l_float
    else:
        return bit_width_of[L.native]() >= bit_width_of[R.native]()


comptime promote[L: NumericType, R: NumericType] = L if (
    _outranks[L, R]()
) else R


comptime wider[L: DType, R: DType] = L if (
    bit_width_of[L]() >= bit_width_of[R]()
) else R
"""The wider of two machine types — a different question from `promote`.

`promote` decides the **value domain**, where a float outranks any integer.
This decides the **register size** a lane iterates at, where only bit width
matters. A bit-packing driver sizes `W` from a DType, and a narrower one yields
a *larger* `W`, so sizing from the narrower operand would overflow the register
the wider one is loaded into.
"""
