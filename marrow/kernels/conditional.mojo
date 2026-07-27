"""Conditional / null-handling compute kernels.

Four SQL-style value-selection kernels that pick, per row, one element from a
set of candidate arrays:

- ``case_when`` — multi-branch ``CASE WHEN``: the value of the first case whose
  condition is valid-and-true, otherwise the ``else`` value (or null).
- ``coalesce``  — the first non-null value across N arrays.
- ``nullif``    — ``a`` with the elements where ``a == b`` set to null.
- ``fill_null`` — ``a`` with its nulls replaced by a scalar or an array.

All four are **type-agnostic**: they work for numeric, bool and string/binary
values (and any other type ``take`` supports) because they are built entirely
from existing kernel primitives rather than per-type element loops:

1. every kernel computes a per-row ``Int32`` *branch selector* (which candidate
   to pick, or null), and
2. delegates to ``_multiplex`` — which ``concat``s the candidates into one array
   and gathers the chosen elements with a single ``take``.

This selection-vector / gather formulation matches Arrow C++ semantics
(``ExecArrayCaseWhen`` treats a null condition as *false*; ``coalesce`` /
``if_else`` propagate value nulls) while reusing ``concat`` + ``take`` + the
comparison kernels instead of reimplementing null handling per dtype.

Cross-checked against PyArrow ``pc.case_when``, ``pc.coalesce``,
``pc.if_else`` and ``pc.fill_null``.
"""

from ..arrays import AnyArray, BoolArray, Int32Array, PrimitiveArray
from ..scalars import AnyScalar
from ..builders import Int32Builder
from ..dtypes import PrimitiveType
from .execution import ExecutionContext
from .concat import concat
from .filter import take
from .compare import EqKernel


# ---------------------------------------------------------------------------
# Shared selection engine
# ---------------------------------------------------------------------------


def _require_uniform(candidates: List[AnyArray]) raises -> Int:
    """Validate that every candidate shares one length and one dtype; return the
    common length. Raises on an empty list or any mismatch."""
    if len(candidates) == 0:
        raise Error("conditional: at least one candidate array is required")
    var length = candidates[0].length()
    var dt = candidates[0].dtype()
    for k in range(1, len(candidates)):
        if candidates[k].length() != length:
            raise Error(
                t"conditional: candidate length mismatch:"
                t" {candidates[k].length()} != {length}"
            )
        if candidates[k].dtype() != dt:
            raise Error(
                t"conditional: candidate dtype mismatch:"
                t" {candidates[k].dtype()} != {dt}"
            )
    return length


def _multiplex(
    candidates: List[AnyArray],
    sel: Int32Array,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Gather one element per row from `candidates` according to `sel`.

    `candidates` must be N same-length (L) same-dtype arrays and `sel` a
    length-L branch-selector array whose valid values lie in ``[0, N)`` (a null
    selector yields a null output element). The candidates are concatenated into
    a single length-``N*L`` array and the per-row absolute index
    ``sel[i] * L + i`` is gathered with one `take` — so the result dtype follows
    `take` and no per-type logic is needed here.
    """
    var length = len(sel)
    var big = concat(candidates, ctx)
    var idx = Int32Builder(capacity=length)
    for i in range(length):
        if sel.is_valid(i):
            idx.append(Int32(Int(sel.unsafe_get(i)) * length + i))
        else:
            idx.append_null()
    return take(big, idx.finish(), ctx)


# ---------------------------------------------------------------------------
# case_when
# ---------------------------------------------------------------------------


def case_when(
    conditions: List[BoolArray],
    values: List[AnyArray],
    else_: Optional[AnyArray] = None,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Multi-branch ``CASE WHEN`` selection (PyArrow ``pc.case_when``).

    For each row the output is the value of the first ``values[k]`` whose
    ``conditions[k]`` is **valid and true**; a null condition counts as false
    (Arrow ``ExecArrayCaseWhen`` semantics). If no condition matches, the
    ``else_`` value is used, or null when no ``else_`` is given. A selected
    value that is itself null produces a null output element.

    Args:
        conditions: One boolean array per branch (all the same length).
        values: One value array per branch — same count as `conditions`, all
            sharing one dtype (and the `else_` dtype).
        else_: Optional fallback value array; null-fallback when omitted.
        ctx: Execution context, forwarded to `concat` / `take`.
    """
    var m = len(conditions)
    if m == 0:
        raise Error("case_when: at least one condition is required")
    if len(values) != m:
        raise Error(
            t"case_when: got {m} conditions but {len(values)} value arrays"
        )

    var candidates = List[AnyArray]()
    for k in range(m):
        candidates.append(values[k].copy())
    var have_else = Bool(else_)
    if have_else:
        candidates.append(else_.value().copy())

    var length = _require_uniform(candidates)
    for k in range(m):
        if len(conditions[k]) != length:
            raise Error(
                t"case_when: condition length {len(conditions[k])} != value"
                t" length {length}"
            )

    var sel = Int32Builder(capacity=length)
    for i in range(length):
        var chosen = -1
        for k in range(m):
            var c = conditions[k][i]
            if c.is_valid() and c.value():
                chosen = k
                break
        if chosen >= 0:
            sel.append(Int32(chosen))
        elif have_else:
            sel.append(Int32(m))
        else:
            sel.append_null()

    return _multiplex(candidates, sel.finish(), ctx)


def case_when[
    T: PrimitiveType
](
    conditions: List[BoolArray],
    values: List[PrimitiveArray[T]],
    else_: Optional[PrimitiveArray[T]] = None,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Typed ``case_when`` over primitive branches — delegates to the erased
    engine and re-wraps the result."""
    var vv = List[AnyArray]()
    for k in range(len(values)):
        vv.append(values[k].copy())
    var e = Optional[AnyArray](None)
    if else_:
        var ea: AnyArray = else_.value().copy()
        e = ea^
    return case_when(conditions, vv, e^, ctx).as_primitive[T]().copy()


# ---------------------------------------------------------------------------
# coalesce
# ---------------------------------------------------------------------------


def coalesce(
    arrays: List[AnyArray],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """First non-null value across `arrays`, elementwise (PyArrow
    ``pc.coalesce``). If every input is null in a row, the output is null."""
    var candidates = List[AnyArray]()
    for k in range(len(arrays)):
        candidates.append(arrays[k].copy())
    var length = _require_uniform(candidates)

    var sel = Int32Builder(capacity=length)
    for i in range(length):
        var chosen = -1
        for k in range(len(candidates)):
            if candidates[k].is_valid(i):
                chosen = k
                break
        if chosen >= 0:
            sel.append(Int32(chosen))
        else:
            sel.append_null()

    return _multiplex(candidates, sel.finish(), ctx)


def coalesce[
    T: PrimitiveType
](
    arrays: List[PrimitiveArray[T]],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Typed ``coalesce`` over primitive arrays."""
    var aa = List[AnyArray]()
    for k in range(len(arrays)):
        aa.append(arrays[k].copy())
    return coalesce(aa, ctx).as_primitive[T]().copy()


# ---------------------------------------------------------------------------
# nullif
# ---------------------------------------------------------------------------


def nullif(
    a: AnyArray,
    b: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """``nullif(a, b)`` — ``a`` with the elements equal to ``b`` set to null
    (SQL ``NULLIF``). A row is nulled only where both are valid and equal; where
    either is null the comparison is not true, so ``a`` is kept (and remains
    null if ``a`` was null there)."""
    if a.dtype() != b.dtype():
        raise Error(t"nullif: dtype mismatch: {a.dtype()} != {b.dtype()}")
    var length = a.length()
    if b.length() != length:
        raise Error(t"nullif: length mismatch: {a.length()} != {b.length()}")

    var eq = EqKernel.dispatch(a.copy(), b.copy(), ctx).as_bool().copy()
    var candidates = List[AnyArray]()
    candidates.append(a.copy())

    var sel = Int32Builder(capacity=length)
    for i in range(length):
        var e = eq[i]
        if e.is_valid() and e.value():
            sel.append_null()  # a == b  ->  null
        else:
            sel.append(Int32(0))  # keep a[i]

    return _multiplex(candidates, sel.finish(), ctx)


def nullif[
    T: PrimitiveType
](
    a: PrimitiveArray[T],
    b: PrimitiveArray[T],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Typed ``nullif`` over primitive arrays."""
    var aa: AnyArray = a.copy()
    var bb: AnyArray = b.copy()
    return nullif(aa, bb, ctx).as_primitive[T]().copy()


# ---------------------------------------------------------------------------
# fill_null
# ---------------------------------------------------------------------------


def fill_null(
    a: AnyArray,
    fill: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Replace the nulls of `a` with the corresponding elements of `fill`
    (PyArrow ``pc.fill_null`` with an array replacement). Where `a` is valid the
    output keeps `a`; where `a` is null it takes `fill` (which itself may be
    null, leaving a null)."""
    if a.dtype() != fill.dtype():
        raise Error(t"fill_null: dtype mismatch: {a.dtype()} != {fill.dtype()}")
    var length = a.length()
    if fill.length() != length:
        raise Error(
            t"fill_null: length mismatch: {a.length()} != {fill.length()}"
        )

    var candidates = List[AnyArray]()
    candidates.append(a.copy())
    candidates.append(fill.copy())

    var sel = Int32Builder(capacity=length)
    for i in range(length):
        if a.is_valid(i):
            sel.append(Int32(0))
        else:
            sel.append(Int32(1))

    return _multiplex(candidates, sel.finish(), ctx)


def fill_null(
    a: AnyArray,
    fill: AnyScalar,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> AnyArray:
    """Replace the nulls of `a` with a scalar (PyArrow ``pc.fill_null`` with a
    scalar replacement). The scalar is broadcast to `a`'s length via
    ``AnyScalar.repeat`` and forwarded to the array overload."""
    return fill_null(a, fill.repeat(a.length()), ctx)


def fill_null[
    T: PrimitiveType
](
    a: PrimitiveArray[T],
    fill: Scalar[T.native],
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> PrimitiveArray[T]:
    """Typed ``fill_null`` replacing nulls of `a` with a primitive scalar."""
    from ..scalars import PrimitiveScalar

    var aa: AnyArray = a.copy()
    var s = PrimitiveScalar[T](Optional[Scalar[T.native]](fill), a.dtype.copy())
    return fill_null(aa, s^.to_any(), ctx).as_primitive[T]().copy()
