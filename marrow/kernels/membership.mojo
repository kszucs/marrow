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

from ..arrays import (
    AnyArray,
    BoolArray,
    PrimitiveArray,
    StringArray,
)
from ..builders import BoolBuilder
from ..dtypes import PrimitiveType
from .execution import ExecutionContext
from .hashing import rapidhash
from .hashtable import SwissHashTable


# ---------------------------------------------------------------------------
# Core hash-set build + probe (type-erased — membership is inherently uniform
# over the 64-bit hash, so all concrete types funnel through here).
# ---------------------------------------------------------------------------


def _is_in(
    values: AnyArray, value_set: AnyArray, ctx: ExecutionContext
) raises -> BoolArray:
    """Hash ``value_set`` into a ``SwissHashTable`` once, then probe each value.

    Returns an all-valid ``BoolArray`` of ``len(values)`` — ``true`` where the
    value's hash is present in the set. Null semantics follow from the shared
    ``NULL_HASH_SENTINEL`` bucket (see module docstring).
    """
    var n = len(values)

    var table = SwissHashTable[rapidhash]()
    table.build_hashes(rapidhash(value_set, ctx))
    var indices = table.probe_hashes(
        rapidhash(values, ctx),
        num_build_rows=len(value_set),
        single_match=True,
    )
    ref probe_rows = indices[1]

    # ``single_match`` → each matching value row appears exactly once; mark it.
    var mask = List[Bool](length=n, fill=False)
    for i in range(len(probe_rows)):
        mask[Int(probe_rows.unsafe_get(i))] = True

    var out = BoolBuilder(n)
    for i in range(n):
        out.append(mask[i])
    return out.finish()


# ---------------------------------------------------------------------------
# Typed overloads — thin typed entry points that guarantee matching element
# types at compile time and delegate to the shared hash-set probe.
# ---------------------------------------------------------------------------


def is_in[
    T: PrimitiveType
](
    values: PrimitiveArray[T],
    value_set: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Membership of each ``values[i]`` in ``value_set`` for a numeric type."""
    return _is_in(values.copy(), value_set.copy(), ctx)


def is_in(
    values: BoolArray,
    value_set: BoolArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Membership of each boolean value in ``value_set``."""
    return _is_in(values.copy(), value_set.copy(), ctx)


def is_in(
    values: StringArray,
    value_set: StringArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Membership of each string in ``value_set``."""
    return _is_in(values.copy(), value_set.copy(), ctx)


# ---------------------------------------------------------------------------
# Type-erased dispatch — the runtime-typed entry point.
# ---------------------------------------------------------------------------


def is_in(
    values: AnyArray,
    value_set: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> BoolArray:
    """Membership of each value in ``value_set`` (runtime-typed).

    ``values`` and ``value_set`` must share the same data type. Supports every
    type ``rapidhash`` handles — numeric, bool, string, and the nested types —
    covering the ClickBench ``IN (...)`` case (int) and strings.
    """
    if values.dtype() != value_set.dtype():
        raise Error(
            "is_in: values and value_set must have the same type, got ",
            values.dtype(),
            " and ",
            value_set.dtype(),
        )
    return _is_in(values.copy(), value_set.copy(), ctx)
