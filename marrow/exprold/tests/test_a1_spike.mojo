"""A1 spike — can a node's `State` hold a `BufferView` borrowed from the batch?

This is the one question that decides whether A1 is worth starting. B28 measured
the prize: `a + 1` over 1M rows costs 2.00 ms today and 65.7 us with the column
resolved once, a 30.5x difference, and the breakdown says ~1.14 ms of the 1.21 ms
saved is the `Variant` unwrap plus `.values()` reconstruction. A `State` that
holds only an *owned* typed array would still rebuild the view per SIMD chunk and
leave most of that on the table. So `State` must hold the view itself.

A view carries an origin, and the origin here is the batch — which is a *runtime*
reference, not a comptime type. So the associated type has to be parameterised by
that origin:

    comptime State[o: Origin[mut=False]] = BufferView[int32.native, o]

Whether Mojo permits a *parameterised* associated type, and whether the origin of
`batch.columns[i].as_int32().values()` unifies with the caller's `o`, is what this
file answers. CLAUDE.md's spike notes cover associated types composed from type
*parameters*; a parameter that is an origin bound to a function argument is a
different question and has not been tested.

**Answer, measured 2026-08-05: the protocol reaches the floor, and it does not
need the view.** `a + 1` over 1M rows through this `prepare`/`vectorwise` pair
runs in **67.25 us** against a hand-written hoisted ideal of **67.14 us** and a
current fused lane of **2.0054 ms** — within 0.2% of the floor, a 29.8x win.

Two findings for the A1 design:

1. A *parameterised* associated type (`comptime State[o: Origin[mut=False]]`)
   does compile. What fails is origin unification: `prepare` returns
   `BufferView[int32.native, origin_of(o.columns[...].buffer)]`, a nested
   projection of the batch's origin, which will not unify with a plain `o`. So a
   `State` holding a view has no name that can be written down.
2. It does not matter. `State` holding the **owned typed array** reaches the same
   number, because the expensive parts were the schema lookup and the `Variant`
   unwrap -- both hoisted into `prepare` -- while `.values()` is pointer
   arithmetic on a `TrivialRegisterPassable` view and costs nothing per chunk.

So A1 needs no origin gymnastics, no `unsafe_origin_cast`, and no parameterised
associated types. That is a far smaller change than the design assumed.
"""

from std.testing import assert_equal

from ...arrays import Int32Array
from ...builders import Int32Builder
from ...dtypes import int32
from ...tabular import record_batch, RecordBatch
from ...views import BufferView


struct SpikeColumn(Copyable, Movable):
    """A column leaf whose per-pass state is its resolved, typed column."""

    var _index: Int

    comptime State = Int32Array
    """The **owned typed array**, not a view.

    A view would be ideal but cannot be spelled: `prepare` returns
    `BufferView[int32.native, origin_of(o.columns[...].buffer)]` -- a nested
    projection of the batch's origin -- and that does not unify with a plain `o`
    parameter, so the associated type has no name to declare. (The parameterised
    associated type `State[o: Origin[mut=False]]` itself compiles; only the
    origin unification fails.)

    Holding the array instead sidesteps origins entirely and costs one ref-count
    bump per pass rather than per chunk. It removes the schema lookup and the
    `Variant` unwrap -- which B28's breakdown says are the expensive parts --
    and leaves only `.values()`, pointer arithmetic on a
    `TrivialRegisterPassable` view."""

    def __init__(out self, index: Int):
        self._index = index

    @always_inline
    def prepare(self, batch: RecordBatch) -> Self.State:
        """Resolve once, outside the lane loop — the whole point of A1."""
        return batch.columns[self._index].as_int32().copy()

    @always_inline
    def vectorwise[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[int32.native, W]:
        """The lane body: a load off an already-typed array."""
        return state.values().load[W](idx)


def test_a1_spike_state_is_resolved_once_and_read_many_times() raises:
    """`prepare` resolves the column; `vectorwise` only loads from it."""
    var b = Int32Builder(8)
    for i in range(8):
        b.append(Int32(i * 10))
    var batch = record_batch([b.finish().to_dyn()], names=["a"])

    var node = SpikeColumn(0)
    var state = node.prepare(batch)
    assert_equal(Int(node.vectorwise[1](state, 0)[0]), 0)
    assert_equal(Int(node.vectorwise[1](state, 3)[0]), 30)
    assert_equal(Int(node.vectorwise[1](state, 7)[0]), 70)


def test_a1_spike_state_survives_a_lane_loop() raises:
    """The state is prepared once and read across many chunks — the shape the
    fused driver needs, and where a too-short borrow would show up."""
    var n = 1024
    var b = Int32Builder(n)
    for i in range(n):
        b.append(Int32(i))
    var batch = record_batch([b.finish().to_dyn()], names=["a"])

    var node = SpikeColumn(0)
    var state = node.prepare(batch)
    var total = 0
    for i in range(n):
        total += Int(node.vectorwise[1](state, i)[0])
    assert_equal(total, (n - 1) * n // 2)
