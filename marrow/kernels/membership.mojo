"""Set-membership kernel — ``is_in``.

``is_in(values, value_set) -> BoolArray`` marks each element of ``values`` with
whether it appears in ``value_set`` (SQL ``x IN (...)``, PyArrow
``pyarrow.compute.is_in``).

The set is built **once** from ``value_set`` and every value is probed against
it, reusing the exact hashing / hash-table building blocks that group-by, join,
and ``count_distinct`` use:

- ``rapidhash`` (``hashing.mojo``) hashes both sides column-wise.
- ``SwissHashTable`` (``hashtable.mojo``) holds the set: ``build_hashes`` inserts
  the ``value_set`` hashes and builds the CSR row index; ``probe_hashes`` looks
  every ``values`` hash up (``single_match=True``) and reports which value rows
  found a matching bucket.

Membership is exact to the same 64-bit-hash basis the group-by / join dedup on
(collision probability ~n·m/2^64) — the same trade ``count_distinct`` documents.

Null handling matches PyArrow ``is_in``'s default (``null_matching_behavior=
"match"``): the output is always valid (never null), and a null in ``values``
is ``true`` iff ``value_set`` itself contains a null and ``false`` otherwise.
This falls out for free from ``rapidhash`` mapping every null to a single
``NULL_HASH_SENTINEL`` bucket — a null probes as ``true`` exactly when
``value_set`` inserted that sentinel bucket.
"""

from ..arrays import DynArray, Array, BoolArray
from ..builders import BoolBuilder
from .core import Kernel
from .execution import ExecutionContext
from .hashing import rapidhash
from .hashtable import SwissHashTable


struct IsInKernel(Kernel):
    """Membership predicate — is each element of ``values`` in ``value_set``?

    Unlike the element-wise kernel families this one has **no typed leaves**:
    membership is decided entirely on the 64-bit hash, so every data type
    ``rapidhash`` supports funnels through the same code. There is one
    implementation, and a newly supported type is one ``rapidhash`` learns —
    not one this kernel gains an overload for.
    """

    comptime name = "is_in"

    @staticmethod
    def apply(
        values: DynArray, value_set: DynArray, ctx: ExecutionContext
    ) raises -> BoolArray:
        """Hash ``value_set`` into a ``SwissHashTable`` once, then probe each
        value.

        Returns an all-valid ``BoolArray`` of ``len(values)`` — ``true`` where
        the value's hash is present in the set. Null semantics follow from the
        shared ``NULL_HASH_SENTINEL`` bucket (see module docstring)."""
        var n = len(values)

        var table = SwissHashTable[rapidhash]()
        table.build_hashes(rapidhash(value_set, ctx))
        var indices = table.probe_hashes(
            rapidhash(values, ctx),
            num_build_rows=len(value_set),
            single_match=True,
        )
        ref probe_rows = indices[1]

        # ``single_match`` → each matching value row appears exactly once.
        var mask = List[Bool](length=n, fill=False)
        for i in range(len(probe_rows)):
            mask[Int(probe_rows.unsafe_get(i))] = True

        var out = BoolBuilder(n)
        for i in range(n):
            out.append(mask[i])
        return out.finish()

    @staticmethod
    def dispatch(
        values: DynArray,
        value_set: DynArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> BoolArray:
        """Validate that both operands carry the same type, then probe."""
        Self.expect_same_dtype(values.dtype(), value_set.dtype())
        return Self.apply(values, value_set, ctx)


# ---------------------------------------------------------------------------
# Public API — the `pc.*` entry point, in its erased and typed forms. There
# used to be three typed overloads (`PrimitiveArray[T]`, `BoolArray`,
# `StringArray`) with byte-identical bodies; one bound on `Array` covers every
# array type there is, including the ones they omitted. It exists for the call
# site, not for the kernel: typed arrays are deliberately not
# `ImplicitlyCopyable`, so without it every caller holding a typed array would
# have to spell `.copy()` to reach the erased form.
# ---------------------------------------------------------------------------


def is_in(
    values: DynArray,
    value_set: DynArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Membership of each value in ``value_set``.

    ``values`` and ``value_set`` must share the same data type. Supports every
    type ``rapidhash`` handles — numeric, bool, string, and the nested types —
    covering the ClickBench ``IN (...)`` case (int) and strings.
    """
    return IsInKernel.dispatch(values, value_set, ctx)


def is_in[
    A: Array
](
    values: A,
    value_set: A,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Membership of each value in ``value_set``, for two arrays of one type.

    Still validated rather than trusted: a shared Mojo type is not a shared
    dtype for the types that carry theirs at runtime — two `ListArray`s can
    disagree about their element type — so this goes through `dispatch` like
    any other caller.
    """
    return IsInKernel.dispatch(values.copy(), value_set.copy(), ctx)
