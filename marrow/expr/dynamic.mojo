"""Runtime-lane pieces that are not expression nodes.

This module used to hold `TagValue`, a 41-tag interpreter: one struct with seven
fields and a switch per method (`eval`, `prune`, `_op_name`, `write_to`). It is
gone. A runtime-built expression is now made of the *same* nodes the fused lane
uses — `col("a") + col("b")` is an `Add[DynValue, DynValue]` — so there is no
second representation to interpret, and the factories live beside the nodes in
`marrow.expr.values`.

What is left is what genuinely belongs to the runtime lane:

- `DynAgg` — an aggregate named by *string*, resolved against the input's dtype
  when the plan is built. The fused lane names its `Aggregation` type outright;
  an erased operand has no type to name one with.
- `_numeric_rank` / `_promote_operands` — what `a + b` means when the operands
  are different numeric types. The fused counterpart is the comptime
  `promote[L, R]` in `values.mojo`, and `test_numeric_rank_agrees_across_lanes`
  pins the two together.
"""

from ..arrays import DynArray
from ..dtypes import DynType
from .values import DynValue
from ..kernels.cast import cast as cast_array


# ---------------------------------------------------------------------------
# Operand promotion — what `a + b` means across numeric types
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


def _numeric_rank(t: DynType) -> Int:
    """Runtime twin of `values._rank`: bit width, with every float outranking
    every integer. The two must stay in step — that they agree is exactly what
    makes the fused and interpreted lanes accept the same operand pairings and
    produce the same output dtype.

    The width comes from the `is_*` predicates rather than `byte_width()`,
    which resolves through `variant_dispatch[PrimitiveType]` and would
    instantiate its closure for *every* primitive dtype — temporal, interval
    and decimal included — to answer a question this only ever asks about the
    eleven numeric ones."""
    var width: Int
    if t.is_int8() or t.is_uint8():
        width = 8
    elif t.is_int16() or t.is_uint16() or t.is_float16():
        width = 16
    elif t.is_int32() or t.is_uint32() or t.is_float32():
        width = 32
    else:
        width = 64
    return width + (1000 if t.is_floating_point() else 0)


def _promote_operands(mut left: DynArray, mut right: DynArray) raises:
    """Widen the narrower of two numeric operands so the kernel sees one dtype.

    Only numeric-to-numeric pairs promote; anything else is passed through
    untouched and a genuine mismatch still raises from the kernel. It costs
    nothing in the interpreter's reachable set: `cast_array` is already linked
    in by the `CAST` tag (measured — stubbing these two calls out left
    `query_dynvalue` byte-identical)."""
    var lt = left.dtype()
    var rt = right.dtype()
    if lt == rt or not lt.is_numeric() or not rt.is_numeric():
        return
    if _numeric_rank(lt) >= _numeric_rank(rt):
        right = cast_array(right, lt)
    else:
        left = cast_array(left, rt)


# ---------------------------------------------------------------------------
# DynAgg — an aggregate named at run time
# ---------------------------------------------------------------------------


struct DynAgg(Copyable, Movable, Writable):
    """An aggregate applied to a runtime expression — ``col("x").sum()``.

    The dynamic counterpart of the fused ``AggExpr`` (``marrow.expr.values``):
    it names the aggregate rather than naming its ``Aggregation`` type, so the
    function is resolved once — against the input's dtype — when the plan is
    built. ``alias`` sets the output column name; without one the function's own
    name is used."""

    var func: String
    var input: DynValue
    var out_name: String

    def __init__(
        out self,
        var func: String,
        var input: DynValue,
        var out_name: String = String(),
    ):
        self.func = func^
        self.input = input^
        self.out_name = out_name^

    def alias(self, var name: String) -> DynAgg:
        """Name this aggregate's output column."""
        return DynAgg(self.func, self.input.copy(), name^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.func, "(")
        self.input.write_to(writer)
        writer.write(")")
        if self.out_name:
            writer.write(" as ", self.out_name)
