"""The comptime lane's aggregate executor.

`Fold` is **physical**: it owns an accumulator and mutates it as morsels
arrive. It lived beside the logical `Sum` / `Min` nodes in `aggregates.mojo`,
which mixed a description with the thing that runs it — the one split this
package is otherwise strict about.

It is not a duplicate of `kernels.aggregate.AggState`, and the difference is
the point of the whole lane. `AggState.update` takes an
`PrimitiveArray[V]` — an **already-materialised column**. `Fold` holds the
fused subtree instead and reads `lane[W]` straight into the accumulator, so
`sum(a * 2 + b)` never builds `a * 2 + b`. `AggState` is storage and scatter;
`Fold` is what avoids handing it a column in the first place.

It lives in the comptime lane rather than in `physical.mojo` because it is
parameterised on `A: NumericValue` — a fused type. `physical.mojo` stays
lane-agnostic, holding only operators that speak `DynValue`.
"""

from std.sys.info import simd_width_of

from ...arrays import DynArray, Int32Array
from ...kernels.groupby import Grouping
from ...kernels.aggregate import AggKernel, AggState
from ...tabular import RecordBatch
from ..physical import Morsel, Operator
from .core import NumericValue


struct Fold[K: AggKernel, A: NumericValue, G: Grouping](Operator):
    """The fold in progress: this aggregate's input, plus the kernel's state.

    Three axes, all comptime: the **algebra** (`K`), the whole **input**
    subtree (`A`), and the **placement** (`G`). `Fold[SumKernel,
    Mul[Column[Int64Type], Literal[Int64Type]], ScalarGrouping]` is one type,
    and therefore one loop with nothing interpreted inside it.

    `G` is a *phantom* parameter — the fold reads `G.scatters` and never holds
    a `G`. The grouping instance itself belongs to the operator above, which
    assigns every row once and shares the result with every aggregate in the
    query; a fold that owned its own grouping would re-hash the keys once per
    aggregate.

    Placement being a **type** rather than a flag is what makes the two loops
    two instantiations of one struct, neither compiling the other's body. It is
    not a runtime branch: which one a query needs is known when the plan is
    built, and running the scattering loop at a single group costs **14.6x**.
    A sorted or partitioned placement arrives as another conformer, not as
    another `Bool`.
    """

    comptime Acc = Self.K.AccType[Self.A.Type]
    comptime acc = Self.Acc.native
    comptime W = simd_width_of[Scalar[Self.acc]]()

    comptime Out = DynArray
    """A **column**, not a batch — one value per slot.

    This is what `Operator.Out` was introduced for. A fold and a filter are the
    same shape of thing (`push` until there is nothing left, then `finish`) and
    differ only in what they produce, so parameterising the output is all that
    was ever needed to make them one trait. Before this, an aggregate had its
    own trait *and* its own erased box for exactly that difference.
    """

    var _input: Self.A
    var _state: AggState[Self.K, Self.A.Type]
    var _num_groups: Int

    def __init__(out self, var input: Self.A):
        self._input = input^
        # One implicit slot when this fold does not scatter, including over an
        # input that yields nothing: `sum` of no rows is one NULL, not no rows.
        self._num_groups = 1 if not Self.G.scatters else 0
        self._state = AggState[Self.K, Self.A.Type]()

    def push(mut self, morsel: Morsel) raises -> Optional[DynArray]:
        """Fold one morsel in and answer nothing — the result arrives at
        `finish`. The grouping travels *with* the batch, which is what lets this
        be an `Operator` at all."""
        ref batch = morsel.batch
        ref groups = morsel.groups.ids
        comptime if Self.G.scatters:
            self._num_groups = morsel.groups.num_groups
        var num_groups = self._num_groups
        var n = batch.num_rows()
        if n == 0:
            return None
        comptime W = Self.W
        var bound = self._input.bind(batch)
        var v = self._input.validity(bound)

        # The SIMD body stops at the last whole chunk. A `range(0, n, W)` loop
        # reads past the view on the final chunk and **aborts the process**:
        # buffer *size* is rounded up to 64 bytes, and that is not slack.
        var simd_end = (n // W) * W

        comptime if Self.G.scatters:
            var gids = groups.values()
            var i = 0
            if v:
                var vb = v.value()
                var bits = vb.view()
                while i < simd_end:
                    self._state.accumulate[W](
                        gids.load[W](i),
                        self._input.lane[W](bound, i).cast[Self.acc](),
                        bits.load[W](i),
                        num_groups,
                    )
                    i += W
                while i < n:
                    self._state.accumulate[1](
                        gids.load[1](i),
                        self._input.lane[1](bound, i).cast[Self.acc](),
                        bits.load[1](i),
                        num_groups,
                    )
                    i += 1
            else:
                while i < simd_end:
                    self._state.accumulate[W](
                        gids.load[W](i),
                        self._input.lane[W](bound, i).cast[Self.acc](),
                        SIMD[DType.bool, W](fill=True),
                        num_groups,
                    )
                    i += W
                while i < n:
                    self._state.accumulate[1](
                        gids.load[1](i),
                        self._input.lane[1](bound, i).cast[Self.acc](),
                        SIMD[DType.bool, 1](True),
                        num_groups,
                    )
                    i += 1
        else:
            # One group by construction, so `groups` is unused: building a zero
            # vector to say so is exactly the cost this path exists to avoid.
            var ident = Self.K.identity[Self.acc]()
            var vec = SIMD[Self.acc, W](ident)
            var acc = ident
            var count = 0
            var i = 0
            if v:
                var vb = v.value()
                var bits = vb.view()
                # A second accumulator, not a horizontal reduce per chunk: the
                # count is `K.finalize`'s divisor for `mean`, and the only thing
                # distinguishing "sum of nothing" from "sum of zeros" — both 0.
                # int64, not the accumulator type: `mean` accumulates in
                # float64, and a count is a count.
                var cnt = SIMD[DType.int64, W](0)
                while i < simd_end:
                    # `lane[W]` is null-blind: it returns data bits regardless
                    # of validity, so a null must become the identity here.
                    # Without the mask an unmasked `sum` is *silently correct*
                    # whenever the null slot's payload happens to be zero —
                    # only min/max expose it.
                    var mask = bits.load[W](i)
                    vec = Self.K.combine[Self.acc, W](
                        vec,
                        mask.select(
                            self._input.lane[W](bound, i).cast[Self.acc](),
                            SIMD[Self.acc, W](ident),
                        ),
                    )
                    cnt += mask.select(
                        SIMD[DType.int64, W](1), SIMD[DType.int64, W](0)
                    )
                    i += W
                while i < n:
                    if bits.load[1](i)[0]:
                        acc = Self.K.combine[Self.acc, 1](
                            acc, self._input.lane[1](bound, i).cast[Self.acc]()
                        )
                        count += 1
                    i += 1
                for j in range(W):
                    count += Int(cnt[j])
            else:
                while i < simd_end:
                    vec = Self.K.combine[Self.acc, W](
                        vec, self._input.lane[W](bound, i).cast[Self.acc]()
                    )
                    i += W
                while i < n:
                    acc = Self.K.combine[Self.acc, 1](
                        acc, self._input.lane[1](bound, i).cast[Self.acc]()
                    )
                    i += 1
                count += n
            for j in range(W):
                acc = Self.K.combine[Self.acc, 1](acc, vec[j])

            # The registers were per-batch scratch; this is the hand-off, once
            # per morsel rather than once per row.
            if count > 0:
                self._state.combine_at(0, acc, count)
            else:
                self._state.combine_at(0, ident, 0)

        return None

    def finish(mut self) raises -> Optional[DynArray]:
        return self._state.finish(self._num_groups).to_dyn()
