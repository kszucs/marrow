"""The root of the kernel trait hierarchy.

Every kernel family in `marrow.kernels` — element-wise arithmetic, comparison,
boolean, string, temporal, aggregate — descends from `Kernel`, so a kernel is
nameable without knowing which family it belongs to. Family traits add the
call shape (`core`/`apply`/`dispatch`, `combine`/`finalize`, ...).

`Kernel` also owns the argument checks every family repeats, so the diagnostic
a caller sees is the same sentence whichever kernel raised it, and a new family
inherits it by conforming rather than by copying the message.
"""

from ..arrays import Int32Array
from ..dtypes import DynType


trait Kernel:
    """Base trait for all compute kernels."""

    comptime name: String
    """This kernel's identity — for display and diagnostics, never dispatch."""

    @staticmethod
    def error[M: Writable](message: M) -> Error:
        """This kernel's failure, attributed to it: `"<name>: <message>"`."""
        return Error(Self.name, ": ", message)

    @staticmethod
    def expect_same_length(left: Int, right: Int) raises:
        """Raise unless both operands hold the same number of elements."""
        if left != right:
            raise Self.error(
                t"arrays must have the same length, got {left} and {right}"
            )

    @staticmethod
    def expect_same_dtype(left: DynType, right: DynType) raises:
        """Raise unless both operands carry the same dtype."""
        if left != right:
            raise Self.error(t"dtype mismatch: {left} vs {right}")


# TODO: have vectorwise and elementwise kernels conform to a common trait


@fieldwise_init
struct Groups(Copyable, Movable):
    """A batch's rows assigned to dense group ids, with how many groups exist.

    The two always travel together: `ids[i]` is row `i`'s group, and
    `num_groups` sizes every per-group accumulator the ids then scatter into.
    They were passed as two parameters through ~22 signatures across `groupby`,
    `aggregate`, `distinct` and `expr.aggregates`, which let a caller size an
    accumulator from one grouping and index it with another's ids — an
    out-of-bounds scatter rather than a type error, and silent when the
    mismatched count happens to be larger.

    Sibling of `JoinIndex`, which named `Tuple[Int32Array, Int32Array]` for the
    same reason.

    Named `Groups` rather than `Grouping` because `Grouping` is the *strategy*
    that produces this — `ScalarGrouping`, `HashGrouping` in `groupby.mojo`.
    The trait is the grouping; this is the groups it assigned.
    """

    var ids: Int32Array
    """Dense group id per row of the batch."""

    var num_groups: Int
    """How many distinct groups exist — the size of a per-group accumulator."""

    def __len__(self) -> Int:
        """Number of rows assigned, not number of groups."""
        return len(self.ids)
