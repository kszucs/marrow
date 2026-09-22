"""Free-standing compute functions exposed to Python.

All GPU-capable functions accept an ``ExecContext`` as their last positional
argument.

**Three shapes cover every kernel here**, so the wrappers below are written out
once each rather than reached for from a general-purpose helper module. This is
the same arrangement `expressions.mojo` uses for `Expr`, and for the same
reason: a binding is easier to read when the conversion it performs is visible
in the file that performs it.
"""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import DynArray
from marrow.dtypes import DynType, bool_, int32
import marrow.kernels as mk

# ``mk.filter`` collides with the like-named submodule, so the package alias
# resolves to the submodule rather than the function. Import it directly.
from marrow.kernels.filter import filter as _filter_kernel
from marrow.kernels.boolean import IsNullKernel, NotNullKernel
from marrow.execution import ExecContext


# ---------------------------------------------------------------------------
# The three kernel shapes
# ---------------------------------------------------------------------------
#
# Each pins the full signature, which is also what lets Mojo resolve the
# `DynArray` runtime overload from a kernel reference like `mk.AddKernel.dispatch`
# when the kernel carries parametric and concrete-type overloads besides.


def _unary[
    f: def(DynArray, ExecContext) raises thin -> DynArray,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    """``(array, ctx) -> array``."""

    def wrapper(array: PythonObject, ctx: PythonObject) raises -> PythonObject:
        return f(DynArray(py=array), ExecContext(py=ctx)).to_python_object()

    return wrapper


def _binary[
    f: def(DynArray, DynArray, ExecContext) raises thin -> DynArray,
]() -> def(
    PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    """``(array, array, ctx) -> array``."""

    def wrapper(
        left: PythonObject, right: PythonObject, ctx: PythonObject
    ) raises -> PythonObject:
        return f(
            DynArray(py=left), DynArray(py=right), ExecContext(py=ctx)
        ).to_python_object()

    return wrapper


def _reduce[
    f: def(DynArray, ExecContext) raises thin -> Bool,
]() -> def(PythonObject, PythonObject) raises thin -> PythonObject:
    """``(array, ctx) -> bool`` — the boolean reductions."""

    def wrapper(array: PythonObject, ctx: PythonObject) raises -> PythonObject:
        return PythonObject(f(DynArray(py=array), ExecContext(py=ctx)))

    return wrapper


# ---------------------------------------------------------------------------
# ExecContext
# ---------------------------------------------------------------------------


def _ctx_serial() raises -> PythonObject:
    return ExecContext.serial().to_python_object()


def _ctx_parallel() raises -> PythonObject:
    return ExecContext.parallel().to_python_object()


# ---------------------------------------------------------------------------
# Composite and argument-carrying kernels — one signature each
# ---------------------------------------------------------------------------


def _sort_indices(
    array: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    return (
        mk.sort_indices(
            DynArray(py=array),
            Bool(py=ascending),
            Bool(py=nulls_first),
            ctx=ExecContext(py=ctx),
        )
        .to_dyn()
        .to_python_object()
    )


def _sort(
    array: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    """`sort_indices` then `take` — marrow has no fused sort kernel."""
    var values = DynArray(py=array)
    var context = ExecContext(py=ctx)
    var indices = mk.sort_indices(
        values, Bool(py=ascending), Bool(py=nulls_first), ctx=context
    )
    # `sort_indices` answers a *typed* `Int32Array`, so it needs `to_dyn`
    # above; `take` over an erased input already answers a `DynArray`.
    return mk.take(values, indices, context).to_python_object()


def _take(
    array: PythonObject, indices: PythonObject, ctx: PythonObject
) raises -> PythonObject:
    var context = ExecContext(py=ctx)
    var idx = DynArray(py=indices)
    # `as_int32()` asserts the variant rather than converting, so an `int64`
    # index array -- what `array([2, 0])` infers -- used to abort the process
    # rather than raise. Cast instead: PyArrow's `take` accepts any integer
    # index type, and an abort is not a diagnosis.
    if not idx.dtype().is_int32():
        idx = mk.cast(idx, DynType(int32), True, context)
    return mk.take(
        DynArray(py=array), idx.as_int32().copy(), context
    ).to_python_object()


def _cast(
    array: PythonObject,
    target: PythonObject,
    safe: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    return mk.cast(
        DynArray(py=array),
        DynType(py=target),
        Bool(py=safe),
        ExecContext(py=ctx),
    ).to_python_object()


def _divide(
    left: DynArray, right: DynArray, ctx: ExecContext
) raises -> DynArray:
    """`divide`, with pyarrow's error on an integer zero divisor.

    `DivKernel.core` answers the *dividend* for a zero integer divisor, which
    is a harmless value rather than an answer; it documents why, and why the
    expression lanes never reach that arm. What is local to this binding is
    the policy: `pyarrow.compute.divide`, which these names mirror, is the
    *checked* kernel and raises, and `pc.divide` matches it on everything
    else — integer truncation, null propagation, dtype.

    Asked after the kernel, so `divide`'s own length and dtype diagnostics
    come out under their own name; and only of rows that would produce a
    value, since pyarrow answers a null dividend over a zero divisor with a
    null rather than an error. The cast is `NumToBoolKernel`, `x != 0`
    bit-packed with validity preserved, so a zero divisor is one bit.

    **Three kernels, but one pass over the column**: the cast reads the
    divisor and `or` and `all` then run over bitmaps, 1/64th of the bytes. One
    read of the divisor is the floor for any exact check, and the cast is it,
    so a cheaper scan in front of this does not exist.

    The rule the three kernels spell out is that `ok` is *false* exactly where
    a valid zero divisor meets a valid dividend. Every other shape is a null
    rather than a false: the cast leaves a null divisor null, Kleene `or`
    keeps it null against a false `is_null`, and `AllKernel` reads only valid
    elements — which is how pyarrow's "a null row is an answer, not an error"
    falls out of the composition instead of out of a branch.
    """
    var out = mk.DivKernel.dispatch(left, right, ctx)
    if right.dtype().is_integer():
        var nonzero = mk.cast(right, DynType(bool_), True, ctx)
        var ok = mk.OrKernel.dispatch(
            nonzero, IsNullKernel.dispatch(left, ctx), ctx
        )
        if not mk.AllKernel.dispatch(ok, ctx):
            raise mk.DivKernel.error("divide by zero")
    return out^


def _concat(arrays: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """Concatenate arrays of one dtype into a single array."""
    var n = Int(py=arrays.__len__())
    var out = List[DynArray](capacity=n)
    for i in range(n):
        out.append(DynArray(py=arrays[i]))
    return mk.concat(out, ExecContext(py=ctx)).to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    _ = (
        mb.add_type[ExecContext]("ExecContext")
        .def_staticmethod[_ctx_serial]("serial")
        .def_staticmethod[_ctx_parallel]("parallel")
    )
    mb.def_function[_binary[mk.AddKernel.dispatch]()]("add")
    mb.def_function[_binary[mk.SubKernel.dispatch]()]("subtract")
    mb.def_function[_binary[mk.MulKernel.dispatch]()]("multiply")
    mb.def_function[_binary[_divide]()]("divide")
    mb.def_function[_reduce[mk.AnyKernel.dispatch]()]("any")
    mb.def_function[_reduce[mk.AllKernel.dispatch]()]("all")
    mb.def_function[_unary[IsNullKernel.dispatch]()]("is_null")
    mb.def_function[_unary[NotNullKernel.dispatch]()]("is_valid")
    mb.def_function[_unary[mk.drop_null]()]("drop_null")
    mb.def_function[_binary[_filter_kernel]()]("filter")
    mb.def_function[_binary[mk.EqKernel.dispatch]()]("equal")
    mb.def_function[_binary[mk.NeKernel.dispatch]()]("not_equal")
    mb.def_function[_binary[mk.LtKernel.dispatch]()]("less")
    mb.def_function[_binary[mk.LeKernel.dispatch]()]("less_equal")
    mb.def_function[_binary[mk.GtKernel.dispatch]()]("greater")
    mb.def_function[_binary[mk.GeKernel.dispatch]()]("greater_equal")
    mb.def_function[_sort_indices]("sort_indices")
    mb.def_function[_sort]("sort")
    mb.def_function[_take]("take")
    mb.def_function[_cast]("cast")
    mb.def_function[_concat]("concat")
