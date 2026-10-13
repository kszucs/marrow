# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The `Kernel` base trait: what every compute kernel shares.

Every kernel family in `marrow.kernels` — element-wise arithmetic, comparison,
boolean, string, temporal, aggregate — descends from `Kernel`, so a kernel is
nameable without knowing which family it belongs to. Family traits add the
call shape (`core`/`apply`/`dispatch`, `combine`/`finalize`, ...).

`Kernel` also owns the argument checks every family repeats, so the diagnostic
a caller sees is the same sentence whichever kernel raised it, and a new family
inherits it by conforming rather than by copying the message.
"""

from std.builtin.variadics import TypeList

from ..dtypes import DynType
from ..errors import ArrowError, InvalidError, TypeError


comptime conforms[F: type_of(AnyType), T: AnyType]: Bool = conforms_to(T, F)
"""Whether dtype `T` belongs to family `F`."""

comptime family[F: type_of(AnyType)] = conforms[F, ...]
"""A kernel's `accepts`: every dtype of family `F` —
`comptime accepts = family[FloatingType]`."""

comptime _member[L: TypeList[Trait=AnyType, ...], T: AnyType]: Bool = (
    L.contains[T]()
)

comptime one_of[*Ts: AnyType] = _member[TypeList.of[Trait=AnyType, *Ts](), ...]
"""A kernel's `accepts`: exactly the dtypes `Ts` —
`comptime accepts = one_of[Date32Type, Date64Type, TimestampType]`."""


trait Kernel:
    """Base trait for all compute kernels."""

    comptime name: String
    """This kernel's identity — for display and diagnostics, never dispatch."""

    @staticmethod
    def error[E: ArrowError, M: Writable](message: M) -> E:
        """This kernel's failure, attributed to it: `Self.error[TypeError](...)`
        reads `"TypeError: <name>: <message>"`."""
        return E(t"{Self.name}: {message}")

    @staticmethod
    def expect_same_length(left: Int, right: Int) raises:
        """Raise unless both operands hold the same number of elements."""
        if left != right:
            raise Self.error[InvalidError](
                t"arrays must have the same length, got {left} and {right}"
            )

    @staticmethod
    def expect_same_dtype(left: DynType, right: DynType) raises:
        """Raise unless both operands carry the same dtype."""
        if left != right:
            raise Self.error[TypeError](t"dtype mismatch: {left} vs {right}")


# TODO: have vectorwise and elementwise kernels conform to a common trait
