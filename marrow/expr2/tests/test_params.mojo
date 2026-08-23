"""Late-bound parameters.

A parameter is a literal whose value arrives later, so these cases check the
two things that distinguishes it from `Literal`: that binding reaches every use
site, and that a plan can say what it needs before anything binds it.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import Int64Type, int64
from ...scalars import Int64Scalar
from ...tabular import RecordBatch, record_batch
from ..builders import col, param
from ..params import Bindings
from ..logical import DynRelation, DynValue, InMemoryTable
from ..`comptime`.leaves import Column
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


def test_a_plan_reports_the_parameters_it_reads() raises:
    """`plan.params()` walks the tree — a sibling to `columns()`.

    That is what makes parameters a property of the *plan*: no registry to
    drain, so a plan built and never executed leaks nothing into the next one.
    """
    var min_a = param("min-a", int64, help=String("lower bound"))
    var plan = _table().filter(
        DynValue(Gt(Column[Int64Type]("a"), min_a.copy()))
    )

    var ps = plan.params()
    assert_equal(len(ps), 1)
    assert_equal(ps[0].name, "min-a")
    assert_equal(ps[0].help, "lower bound")
    assert_true(ps[0].is_required())


def test_two_plans_do_not_see_each_others_parameters() raises:
    """The limitation `expr/` records and this design does not have.

    Its registry is process-global, so a plan built but never drained leaves
    its declarations for the next plan's `--help`. Walking the tree cannot do
    that.
    """
    var a = param("only-a", int64)
    var _abandoned = _table().filter(
        DynValue(Gt(Column[Int64Type]("a"), a.copy()))
    )

    var b = param("only-b", int64)
    var second = _table().filter(DynValue(Gt(Column[Int64Type]("a"), b.copy())))

    var ps = second.params()
    assert_equal(len(ps), 1)
    assert_equal(ps[0].name, "only-b")


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
    assert_true(not plan.params()[0].is_required())
    assert_true(plan.execute().columns[0].as_int64() == array([5, 9], int64))

    var bound = plan.execute(
        bindings=Bindings().set("t", Int64Scalar(8).to_dyn())
    )
    assert_true(bound.columns[0].as_int64() == array([9], int64))
