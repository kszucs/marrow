"""Late-bound parameters.

A parameter is a literal whose value arrives later, so these cases check the
two things that distinguish it from `Literal`: that binding reaches every use
site — including one nested inside a fused subtree — and that the value
belongs to an execution rather than to the plan.
"""

from std.testing import assert_true

from ...builders import array
from ...dtypes import Int64Type, int64
from ...scalars import Int64Scalar
from ...tabular import record_batch
from ..builders import param
from ..params import Bindings
from ..logical import DynRelation, DynValue, InMemoryTable
from ..`comptime`.leaves import Column
from ..`comptime`.boolean import Not
from ..`comptime`.numeric import Gt


def _table() raises -> DynRelation:
    return DynRelation(
        InMemoryTable(
            record_batch([array([1, 5, 9], int64).copy()], names=["a"])
        )
    )


def test_a_value_is_supplied_per_execution() raises:
    """The rule this design exists to obey: a logical node is stateless.

    The plan holds no value. Two executions of the *same* plan with different
    bindings give different answers and cannot interfere — an earlier version
    kept the value in a shared cell inside the node, so `set()` reached into a
    built plan and changed what it computed.
    """
    var min_a = param("min-a", int64)
    var plan = _table().filter(
        DynValue(Gt(Column[Int64Type]("a"), min_a.copy()))
    )

    var low = plan.execute(
        bindings=Bindings().set("min-a", Int64Scalar(4).to_dyn())
    )
    assert_true(low.columns[0].as_int64() == array([5, 9], int64))

    var high = plan.execute(
        bindings=Bindings().set("min-a", Int64Scalar(8).to_dyn())
    )
    assert_true(high.columns[0].as_int64() == array([9], int64))


def test_binding_reaches_a_nested_parameter() raises:
    """`bind` walks the whole tree, so depth does not matter.

    The parameter here is the right operand of a `Gt` that is itself the
    operand of a `Not` — two composites deep. `to_operator` copies a node
    without descending into it, so binding at that seam would leave this one
    unread; the per-batch walk is what reaches it.
    """
    var t = param("t", int64)
    var plan = _table().filter(
        DynValue(Not(Gt(Column[Int64Type]("a"), t.copy())))
    )

    var got = plan.execute(
        bindings=Bindings().set("t", Int64Scalar(4).to_dyn())
    )
    assert_true(got.columns[0].as_int64() == array([1], int64))


def test_an_unbound_parameter_names_itself() raises:
    """`expr/`'s cell raises "parameter is not bound" without naming it,
    because a cell cannot know the name it is read through. Here the node is
    the parameter, so it can."""
    var missing = param("threshold", int64)
    var plan = _table().filter(
        DynValue(Gt(Column[Int64Type]("a"), missing.copy()))
    )

    var raised = False
    try:
        _ = plan.execute()
    except e:
        raised = True
        assert_true("threshold" in String(e))
    assert_true(raised)


def test_a_default_is_used_until_something_binds() raises:
    var t = param("t", int64, default=Optional(Int64(4)))
    var plan = _table().filter(DynValue(Gt(Column[Int64Type]("a"), t.copy())))
    assert_true(plan.execute().columns[0].as_int64() == array([5, 9], int64))

    var bound = plan.execute(
        bindings=Bindings().set("t", Int64Scalar(8).to_dyn())
    )
    assert_true(bound.columns[0].as_int64() == array([9], int64))
