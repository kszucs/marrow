# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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
from existing kernel primitives rather than per-type element loops. They are
also all the *same shape*, and `Selection` below is that shape made explicit:
decide per row which candidate supplies the value (or that the row is null),
then gather. A kernel body is only the decision.

This selection-vector / gather formulation matches Arrow C++ semantics
(``ExecArrayCaseWhen`` treats a null condition as *false*; ``coalesce`` /
``if_else`` propagate value nulls) while reusing ``concat`` + ``take`` + the
comparison kernels instead of reimplementing null handling per dtype.

Cross-checked against PyArrow ``pc.case_when``, ``pc.coalesce``,
``pc.if_else`` and ``pc.fill_null``.
"""

from ..arrays import DynArray, BoolArray, PrimitiveArray
from ..scalars import DynScalar, PrimitiveScalar
from ..builders import DynBuilder, Int32Builder
from ..dtypes import PrimitiveType
from .core import Kernel
from ..execution import ExecContext
from ..errors import InvalidError, TypeError
from .filter import TakeKernel
from .numeric import EqualKernel


# ---------------------------------------------------------------------------
# Selection — the shared engine
# ---------------------------------------------------------------------------


struct Selection:
    """N same-shaped candidate arrays plus the per-row branch selector over
    them.

    Construction validates that the candidates share one length and one dtype;
    `choose`/`choose_null` record a decision per row; `gather` resolves the
    whole thing with one `concat` and one `take`. The candidates are
    concatenated into a single length-``N*L`` array and row ``i``'s absolute
    index is ``sel[i] * L + i``, so the result dtype follows `take` and no
    per-type logic is needed anywhere in this module.

    **Deliberately not parameterised on the kernel** (`Selection[K: Kernel]`
    would read better and let errors attribute themselves statically). It holds
    `concat` and `take`, whose fanout is large, and a parameterised type is
    instantiated per kernel — four copies of that fanout. Q0.4 measured the
    same mistake at +115,600 bytes; the kernel's name is carried as a runtime
    field instead, and mirrors `Kernel.error`'s format because it cannot call
    it.
    """

    var _name: StaticString
    var _candidates: List[DynArray]
    var _length: Int
    var _sel: Int32Builder

    def __init__(
        out self, name: StaticString, var candidates: List[DynArray]
    ) raises:
        """Validate that every candidate shares one length and one dtype."""
        if len(candidates) == 0:
            raise InvalidError(
                t"{name}: at least one candidate array is required"
            )
        var length = candidates[0].length()
        var dtype = candidates[0].dtype()
        for k in range(1, len(candidates)):
            if candidates[k].length() != length:
                raise InvalidError(
                    t"{name}: candidate length mismatch: "
                    t"{candidates[k].length()} != {length}"
                )
            if candidates[k].dtype() != dtype:
                raise TypeError(
                    t"{name}: candidate dtype mismatch: "
                    t"{candidates[k].dtype()} != {dtype}"
                )
        self._name = name
        self._candidates = candidates^
        self._length = length
        self._sel = Int32Builder(capacity=length)

    def name(self) -> StaticString:
        """The owning kernel's name, for diagnostics raised from here."""
        return self._name

    def length(self) -> Int:
        """The common candidate length — the number of decisions to record."""
        return self._length

    def choose(mut self, k: Int) raises:
        """Row takes candidate `k`."""
        self._sel.append(Int32(k))

    def choose_null(mut self) raises:
        """Row is null — no candidate supplies it."""
        self._sel.append_null()

    def gather(mut self, ctx: ExecContext) raises -> DynArray:
        """Resolve every recorded decision: one `concat`, one `take`."""
        var length = self._length
        var big = DynBuilder.concat(self._candidates)
        var sel = self._sel.finish()
        var idx = Int32Builder(capacity=length)
        for i in range(length):
            if sel.is_valid(i):
                idx.append(Int32(Int(sel.unsafe_get(i)) * length + i))
            else:
                idx.append_null()
        return TakeKernel.dispatch(big, idx.finish(), ctx)


def _as_any[
    T: PrimitiveType
](values: List[PrimitiveArray[T]]) -> List[DynArray]:
    """Erase a typed candidate list — what every typed overload below opens
    with."""
    var out = List[DynArray](capacity=len(values))
    for k in range(len(values)):
        out.append(values[k].copy())
    return out^


# ---------------------------------------------------------------------------
# case_when
# ---------------------------------------------------------------------------


struct CaseWhenKernel(Kernel):
    """Multi-branch ``CASE WHEN`` — the first branch whose condition is
    valid-and-true."""

    comptime name = "case_when"

    @staticmethod
    def dispatch(
        conditions: List[BoolArray],
        values: List[DynArray],
        var else_: Optional[DynArray] = None,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Multi-branch ``CASE WHEN`` selection (PyArrow ``pc.case_when``).

        For each row the output is the value of the first ``values[k]`` whose
        ``conditions[k]`` is **valid and true**; a null condition counts as
        false (Arrow ``ExecArrayCaseWhen`` semantics). If no condition matches,
        the ``else_`` value is used, or null when no ``else_`` is given. A
        selected value that is itself null produces a null output element.
        """
        var m = len(conditions)
        if m == 0:
            raise Self.error[InvalidError]("at least one condition is required")
        if len(values) != m:
            raise Self.error[InvalidError](
                t"got {m} conditions but {len(values)} value arrays",
            )

        var candidates = List[DynArray](capacity=m + 1)
        for k in range(m):
            candidates.append(values[k].copy())
        var have_else = Bool(else_)
        if have_else:
            candidates.append(else_.value().copy())

        var sel = Selection(Self.name, candidates^)
        for k in range(m):
            Self.expect_same_length(len(conditions[k]), sel.length())

        for i in range(sel.length()):
            var chosen = -1
            for k in range(m):
                var c = conditions[k][i]
                if c.is_valid() and c.value():
                    chosen = k
                    break
            if chosen >= 0:
                sel.choose(chosen)
            elif have_else:
                sel.choose(m)  # the else branch is the last candidate
            else:
                sel.choose_null()
        return sel.gather(ctx)

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        conditions: List[BoolArray],
        values: List[PrimitiveArray[T]],
        var else_: Optional[PrimitiveArray[T]] = None,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """``dispatch`` over primitive branches of one type."""
        var e = Optional[DynArray](None)
        if else_:
            var ea: DynArray = else_.value().copy()
            e = ea^
        return (
            Self.dispatch(conditions, _as_any(values), e^, ctx)
            .as_primitive[T]()
            .copy()
        )


# ---------------------------------------------------------------------------
# coalesce
# ---------------------------------------------------------------------------


trait BinaryConditionalKernel(Kernel):
    """A conditional kernel in its two-operand form.

    `coalesce` is N-ary and `nullif` is binary, so they share no N-ary
    signature — which is why the expression layer used to declare its own
    `ConditionalBinaryKernel` trait plus `CoalesceOp`/`NullifOp` structs wrapping
    the free functions. That put a second, parallel notion of "kernel" in
    `values.mojo`. The binary form is a property of the kernel, so it is declared
    here and implemented by the kernels themselves."""

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        ...

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        left: PrimitiveArray[T],
        right: PrimitiveArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """`dispatch` over primitive operands of one type."""
        return (
            Self.dispatch(left.copy(), right.copy(), ctx)
            .as_primitive[T]()
            .copy()
        )


struct CoalesceKernel(BinaryConditionalKernel):
    """First non-null value across N arrays, elementwise."""

    comptime name = "coalesce"

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        var candidates = List[DynArray](capacity=2)
        candidates.append(left.copy())
        candidates.append(right.copy())
        return Self.dispatch(candidates, ctx)

    @staticmethod
    def dispatch(
        arrays: List[DynArray], ctx: ExecContext = ExecContext.serial()
    ) raises -> DynArray:
        """First non-null value across `arrays`, elementwise (PyArrow
        ``pc.coalesce``); null where every input is."""
        var candidates = List[DynArray](capacity=len(arrays))
        for k in range(len(arrays)):
            candidates.append(arrays[k].copy())
        var n = len(candidates)
        var sel = Selection(Self.name, candidates^)

        for i in range(sel.length()):
            var chosen = -1
            for k in range(n):
                if arrays[k].is_valid(i):
                    chosen = k
                    break
            if chosen >= 0:
                sel.choose(chosen)
            else:
                sel.choose_null()
        return sel.gather(ctx)

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        arrays: List[PrimitiveArray[T]],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """``dispatch`` over primitive arrays of one type."""
        return Self.dispatch(_as_any(arrays), ctx).as_primitive[T]().copy()


# ---------------------------------------------------------------------------
# nullif
# ---------------------------------------------------------------------------


struct NullifKernel(BinaryConditionalKernel):
    """``a`` with the elements equal to ``b`` set to null (SQL ``NULLIF``)."""

    comptime name = "nullif"

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """``left`` with the elements equal to ``right`` set to null (SQL
        ``NULLIF``). A row is nulled only where both are valid and equal;
        where either is null the comparison is not true, so ``left`` is kept.
        """
        var a = left.copy()
        var b = right.copy()
        Self.expect_same_dtype(a.dtype(), b.dtype())
        Self.expect_same_length(a.length(), b.length())

        # `nullif` is defined for any dtype with an equality, so it needs the
        # family-picking primitive rather than either comparison kernel directly.
        var eq = EqualKernel.dispatch(a, b, ctx)
        var candidates = List[DynArray](capacity=1)
        candidates.append(a.copy())
        var sel = Selection(Self.name, candidates^)

        for i in range(sel.length()):
            var e = eq[i]
            if e.is_valid() and e.value():
                sel.choose_null()  # a == b  ->  null
            else:
                sel.choose(0)  # keep a[i]
        return sel.gather(ctx)


# ---------------------------------------------------------------------------
# fill_null
# ---------------------------------------------------------------------------


struct FillNullKernel(BinaryConditionalKernel):
    """``a`` with its nulls replaced from a second array."""

    comptime name = "fill_null"

    @staticmethod
    def dispatch(
        left: DynArray,
        right: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """``left`` with its nulls replaced by the corresponding elements of
        ``right`` (PyArrow ``pc.fill_null`` with an array replacement)."""
        var a = left.copy()
        var fill = right.copy()
        Self.expect_same_dtype(a.dtype(), fill.dtype())
        Self.expect_same_length(a.length(), fill.length())

        var candidates = List[DynArray](capacity=2)
        candidates.append(a.copy())
        candidates.append(fill.copy())
        var sel = Selection(Self.name, candidates^)

        for i in range(sel.length()):
            sel.choose(0 if a.is_valid(i) else 1)
        return sel.gather(ctx)

    @staticmethod
    def dispatch(
        left: DynArray,
        fill: DynScalar,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """``left`` with its nulls replaced by a scalar, broadcast to its
        length."""
        return Self.dispatch(left, fill.to_array(left.length()), ctx)

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        a: PrimitiveArray[T],
        fill: Scalar[T.native],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """``a`` with its nulls replaced by a primitive scalar."""
        var s = PrimitiveScalar[T](
            Optional[Scalar[T.native]](fill), a.dtype.copy()
        )
        return (
            Self.dispatch(a.copy(), s^.to_dyn().to_array(len(a)), ctx)
            .as_primitive[T]()
            .copy()
        )
