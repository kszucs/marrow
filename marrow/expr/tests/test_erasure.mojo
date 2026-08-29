"""The erased boxes destroy what they hold.

`DynValue`, `DynRelation`, `DynOperator` and `DynAgg` each erase a
typed value by `rebind`ing an `ArcPointer[T]` to `ArcPointer[NoneType]`. That
keeps the allocation and the refcount and **forgets the destructor**: the final
release runs `NoneType`'s, so the boxed object's `__deinit__` never runs and
everything it owns leaks. Nothing else in the suite can see this — every answer
is correct, and only memory is lost — which is how it survived from the first
box to the fourth.

The fix is a `_drop` trampoline per box, and these are the tests that hold it in
place. They count destructions rather than measuring memory, so they fail
deterministically rather than statistically.

The leak is worst where it is least visible: `DynAgg` boxes a fold
state whose size is O(distinct values), so an un-dropped `count_distinct` loses
a whole `SwissHashTable` per query, while the plan-node boxes lose a few
strings.
"""

from std.memory import ArcPointer
from std.testing import assert_equal
from ...dtypes import DynType, int64
from ...execution import ExecContext
from ...kernels.core import Groups
from ...schema import Schema
from ..logical import DynRelation, DynValue, Relation, Shape, Value
from ..params import Bindings
from ..pushdown import Pushdown
from ..physical import Datum, DynOperator, Morsel, Operator, Pipeline


comptime Deaths = ArcPointer[List[Int]]
"""A destruction tally the probe shares with the test. A `List` rather than an
`Int` because the count has to survive the probe it is counting."""


def _tally() -> Deaths:
    return Deaths(List[Int]())


# ---------------------------------------------------------------------------
# probes — one per erased trait, each recording its own destruction
# ---------------------------------------------------------------------------
struct _OpProbe(Operator):
    var _deaths: Deaths

    def __init__(out self, var deaths: Deaths):
        self._deaths = deaths^

    def __deinit__(deinit self):
        self._deaths[].append(1)

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        return None

    def drain(mut self) raises -> Optional[Datum]:
        return None

    def done(self) -> Bool:
        return True


struct _ValueProbe(Copyable, Movable, Value, Writable):
    comptime shape = Shape.columnar

    var _deaths: Deaths

    def __init__(out self, var deaths: Deaths):
        self._deaths = deaths^

    def __deinit__(deinit self):
        self._deaths[].append(1)

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        return String("probe")

    def dtype(self, schema: Schema) raises -> DynType:
        return int64

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        raise Error("probe is not runnable")

    def write_to(self, mut writer: Some[Writer]):
        writer.write("probe")


struct _RelationProbe(Copyable, Movable, Relation, Writable):
    var _deaths: Deaths

    def __init__(out self, var deaths: Deaths):
        self._deaths = deaths^

    def __deinit__(deinit self):
        self._deaths[].append(1)

    def schema(self) -> Schema:
        return Schema()

    def to_operator(
        self,
        ctx: ExecContext,
        bindings: Bindings = Bindings(),
        var pushed: Pushdown = Pushdown(),
    ) raises -> Pipeline:
        raise Error("probe is not runnable")

    def write_to(self, mut writer: Some[Writer]):
        writer.write("probe")


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------
# Every destruction below is an explicit transfer, so the tally is read at a
# known point rather than wherever ASAP destruction happens to land.
#
# The two counts differ because the boxes take their argument differently:
# `DynOperator.__init__` takes `var value: O` and **moves** it, so boxing costs
# no extra object, while `DynValue`/`DynRelation` take `value: V` by borrow and
# box a **copy** — so the caller's original is a second object with its own
# destruction. Both are correct; only the second needs saying.


def test_dyn_operator_destroys_its_operator() raises:
    """The one that matters most in practice: `GroupByOperator` holds a
    `List[DynOperator]`, so every aggregate's state is behind this box."""
    var deaths = _tally()
    var boxed = DynOperator(_OpProbe(deaths.copy()))
    assert_equal(boxed.done(), True)
    assert_equal(len(deaths[]), 0, "the operator was moved into the box")
    _ = boxed^
    assert_equal(len(deaths[]), 1, "erasure must not drop the destructor")


def test_dyn_value_destroys_its_node() raises:
    var deaths = _tally()
    var probe = _ValueProbe(deaths.copy())
    var boxed = DynValue(probe)
    _ = probe^
    assert_equal(len(deaths[]), 1, "the caller's original, not the box's copy")
    assert_equal(boxed.name(), "probe")
    _ = boxed^
    assert_equal(len(deaths[]), 2, "erasure must not drop the destructor")


def test_dyn_relation_destroys_its_node() raises:
    var deaths = _tally()
    var probe = _RelationProbe(deaths.copy())
    var boxed = DynRelation(probe)
    _ = probe^
    assert_equal(len(deaths[]), 1)
    assert_equal(len(boxed.schema()), 0)
    _ = boxed^
    assert_equal(len(deaths[]), 2, "erasure must not drop the destructor")


def test_erased_copies_share_one_destruction() raises:
    """Sharing still works. A box is released once per copy and the *last*
    release is the one that runs the boxed destructor — so N copies destroy
    the boxed node exactly once, not N times (a double free) and not zero
    times (the leak)."""
    var deaths = _tally()
    var probe = _ValueProbe(deaths.copy())
    var first = DynValue(probe)
    _ = probe^
    var second = first.copy()
    var third = second.copy()
    assert_equal(third.name(), "probe")
    _ = first^
    _ = second^
    assert_equal(len(deaths[]), 1, "released twice, but a copy is still alive")
    _ = third^
    assert_equal(len(deaths[]), 2, "the last release destroys, exactly once")
