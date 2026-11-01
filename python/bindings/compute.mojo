# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Free-standing compute functions exposed to Python.

All GPU-capable functions accept an ``ExecContext`` as their last positional
argument.

**A few shapes cover every kernel here**, so the wrappers below are written out
once each rather than reached for from a general-purpose helper module. This is
the same arrangement `expressions.mojo` uses for `Expr`, and for the same
reason: a binding is easier to read when the conversion it performs is visible
in the file that performs it.
"""

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from marrow.arrays import DynArray, Int32Array
from marrow.builders import DynBuilder
from marrow.scalars import BoolScalar
from marrow.errors import InvalidError
from marrow.dtypes import DynType, IntegerType, bool_, int32
import marrow.kernels as mk

from marrow.kernels.boolean import IsNullKernel, NotNullKernel
from marrow.kernels.segments import BySize
from marrow.execution import ExecContext
from marrow.utils.threads import ThreadPool


# ---------------------------------------------------------------------------
# The kernel shapes
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
        return f(DynArray(py=array), ExecContext(py=ctx))

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
        return f(DynArray(py=left), DynArray(py=right), ExecContext(py=ctx))

    return wrapper


def _flag[
    f: def(DynArray, Bool, ExecContext) raises thin -> DynArray,
]() -> def(
    PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    """``(array, flag, ctx) -> array``: ``cumulative_*``'s ``skip_nulls``,
    ``dictionary_encode``'s ``encode_nulls``."""

    def wrapper(
        array: PythonObject, flag: PythonObject, ctx: PythonObject
    ) raises -> PythonObject:
        return f(DynArray(py=array), Bool(py=flag), ExecContext(py=ctx))

    return wrapper


def _offset[
    f: def(DynArray, Int, ExecContext) raises thin -> DynArray,
]() -> def(
    PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    """``(array, offset, ctx) -> array``."""

    def wrapper(
        array: PythonObject, offset: PythonObject, ctx: PythonObject
    ) raises -> PythonObject:
        return f(DynArray(py=array), Int(py=offset), ExecContext(py=ctx))

    return wrapper


def _rolling[
    f: def(DynArray, Int, Int, Int, ExecContext) raises thin -> DynArray,
]() -> def(
    PythonObject, PythonObject, PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    """``(array, lo, hi, min_samples, ctx) -> array``: the frame
    ``ROWS BETWEEN lo AND hi``."""

    def wrapper(
        array: PythonObject,
        lo: PythonObject,
        hi: PythonObject,
        min_samples: PythonObject,
        ctx: PythonObject,
    ) raises -> PythonObject:
        return f(
            DynArray(py=array),
            Int(py=lo),
            Int(py=hi),
            Int(py=min_samples),
            ExecContext(py=ctx),
        )

    return wrapper


def _reduce[
    f: def(DynArray, Bool, Int, ExecContext) raises thin -> BoolScalar,
]() -> def(
    PythonObject, PythonObject, PythonObject, PythonObject
) raises thin -> PythonObject:
    """``(array, skip_nulls, min_count, ctx) -> bool | None`` — the boolean
    reductions."""

    def wrapper(
        array: PythonObject,
        skip_nulls: PythonObject,
        min_count: PythonObject,
        ctx: PythonObject,
    ) raises -> PythonObject:
        var out = f(
            DynArray(py=array),
            Bool(py=skip_nulls),
            Int(py=min_count),
            ExecContext(py=ctx),
        )
        if out.is_valid():
            return PythonObject(out.value())
        return PythonObject(None)

    return wrapper


# ---------------------------------------------------------------------------
# ExecContext
# ---------------------------------------------------------------------------


def _ctx_serial() raises -> PythonObject:
    return ExecContext.serial()


def _ctx_parallel() raises -> PythonObject:
    return ExecContext.parallel()


def _cpu_count() raises -> PythonObject:
    return PythonObject(ThreadPool.shared()[].concurrency())


def _set_cpu_count(count: PythonObject) raises -> PythonObject:
    ThreadPool.resize_shared(Int(py=count))
    return PythonObject(None)


def _io_thread_count() raises -> PythonObject:
    return PythonObject(ThreadPool.shared_io()[].concurrency())


def _set_io_thread_count(count: PythonObject) raises -> PythonObject:
    ThreadPool.resize_shared_io(Int(py=count))
    return PythonObject(None)


# ---------------------------------------------------------------------------
# Composite and argument-carrying kernels — one signature each
# ---------------------------------------------------------------------------


def _sort_indices(
    array: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    return mk.SortIndices.dispatch(
        DynArray(py=array),
        Bool(py=ascending),
        Bool(py=nulls_first),
        ctx=ExecContext(py=ctx),
    ).to_dyn()


def _sort(
    array: PythonObject,
    ascending: PythonObject,
    nulls_first: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    """`sort_indices` then `take` — marrow has no fused sort kernel."""
    var values = DynArray(py=array)
    var context = ExecContext(py=ctx)
    var indices = mk.SortIndices.dispatch(
        values, Bool(py=ascending), Bool(py=nulls_first), ctx=context
    )
    # `sort_indices` answers a *typed* `Int32Array`, so it needs `to_dyn`
    # above; `take` over an erased input already answers a `DynArray`.
    return mk.TakeKernel.dispatch(values, indices, context)


def _take(
    array: PythonObject, indices: PythonObject, ctx: PythonObject
) raises -> PythonObject:
    var context = ExecContext(py=ctx)
    var idx = DynArray(py=indices)
    # `as_int32()` asserts the variant rather than converting, so an `int64`
    # index array -- what `array([2, 0])` infers -- used to abort the process
    # rather than raise. Cast instead: PyArrow's `take` accepts any integer
    # index type, and an abort is not a diagnosis.
    if not idx.isa[Int32Array]():
        idx = mk.CastKernel.dispatch(idx, DynType(int32), True, context)
    var values = DynArray(py=array)
    mk.TakeKernel.check_bounds(len(values), idx.as_int32(), context)
    return mk.TakeKernel.dispatch(values, idx.as_int32().copy(), context)


def _cast(
    array: PythonObject,
    target: PythonObject,
    safe: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    return mk.CastKernel.dispatch(
        DynArray(py=array),
        DynType(py=target),
        Bool(py=safe),
        ExecContext(py=ctx),
    )


def _run_end_encode(
    array: PythonObject, run_end_type: PythonObject, ctx: PythonObject
) raises -> PythonObject:
    return (
        mk.RunEndEncodeKernel.dispatch(
            DynArray(py=array), DynType(py=run_end_type), ExecContext(py=ctx)
        )
        .to_dyn()
        .to_python_object()
    )


def _run_index(array: PythonObject, ctx: PythonObject) raises -> PythonObject:
    return (
        mk.RunIndexKernel.dispatch(DynArray(py=array), ExecContext(py=ctx))
        .to_dyn()
        .to_python_object()
    )


def _filter(
    array: DynArray, mask: DynArray, ctx: ExecContext
) raises -> DynArray:
    return mk.FilterKernel.dispatch(array, mask, ctx)


def _shift(array: DynArray, offset: Int, ctx: ExecContext) raises -> DynArray:
    """`shift` over the whole column, one segment."""
    return mk.ShiftKernel.dispatch(array, offset, BySize.whole(len(array)), ctx)


def _replace_with_mask(
    values: PythonObject,
    mask: PythonObject,
    replacements: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    return mk.ReplaceWithMaskKernel.dispatch(
        DynArray(py=values),
        DynArray(py=mask),
        DynArray(py=replacements),
        ExecContext(py=ctx),
    )


def _mode(
    array: PythonObject,
    n: PythonObject,
    skip_nulls: PythonObject,
    min_count: PythonObject,
    ctx: PythonObject,
) raises -> PythonObject:
    return mk.ModeKernel.dispatch(
        DynArray(py=array),
        Int(py=n),
        Bool(py=skip_nulls),
        Int(py=min_count),
        ExecContext(py=ctx),
    )


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
        var nonzero = mk.CastKernel.dispatch(right, DynType(bool_), True, ctx)
        var ok = mk.OrKernel.dispatch(
            nonzero, IsNullKernel.dispatch(left, ctx), ctx
        )
        if not mk.AllKernel.dispatch(ok, ctx):
            raise mk.DivKernel.error[InvalidError]("divide by zero")
    return out^


def _concat(arrays: PythonObject) raises -> PythonObject:
    """Concatenate arrays of one dtype into a single array."""
    var n = Int(py=arrays.__len__())
    var out = List[DynArray](capacity=n)
    for i in range(n):
        out.append(DynArray(py=arrays[i]))
    return DynBuilder.concat(out)


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    _ = (
        mb.add_type[ExecContext]("ExecContext")
        .def_staticmethod[_ctx_serial]("serial")
        .def_staticmethod[_ctx_parallel]("parallel")
    )
    mb.def_function[_cpu_count]("cpu_count")
    mb.def_function[_set_cpu_count]("set_cpu_count")
    mb.def_function[_io_thread_count]("io_thread_count")
    mb.def_function[_set_io_thread_count]("set_io_thread_count")
    mb.def_function[_binary[mk.AddKernel.dispatch]()]("add")
    mb.def_function[_binary[mk.SubKernel.dispatch]()]("subtract")
    mb.def_function[_binary[mk.MulKernel.dispatch]()]("multiply")
    mb.def_function[_binary[mk.BitAndKernel.dispatch]()]("bit_wise_and")
    mb.def_function[_binary[mk.BitOrKernel.dispatch]()]("bit_wise_or")
    mb.def_function[_binary[mk.BitXorKernel.dispatch]()]("bit_wise_xor")
    mb.def_function[_unary[mk.BitNotKernel.dispatch]()]("bit_wise_not")
    mb.def_function[_binary[mk.ShiftLeftKernel.dispatch]()]("shift_left")
    mb.def_function[_binary[mk.ShiftRightKernel.dispatch]()]("shift_right")
    mb.def_function[_binary[_divide]()]("divide")
    mb.def_function[_reduce[mk.AnyKernel.dispatch]()]("any")
    mb.def_function[_reduce[mk.AllKernel.dispatch]()]("all")
    mb.def_function[_unary[IsNullKernel.dispatch]()]("is_null")
    mb.def_function[_unary[NotNullKernel.dispatch]()]("is_valid")
    mb.def_function[_unary[mk.FilterKernel.drop_null]()]("drop_null")
    mb.def_function[_binary[_filter]()]("filter")
    mb.def_function[_unary[mk.IndicesNonzeroKernel.dispatch]()](
        "indices_nonzero"
    )
    mb.def_function[_replace_with_mask]("replace_with_mask")
    mb.def_function[_flag[mk.DictionaryEncodeKernel.dispatch]()](
        "dictionary_encode"
    )
    mb.def_function[_unary[mk.ValueCountsKernel.dispatch]()]("value_counts")
    mb.def_function[_mode]("mode")
    mb.def_function[_run_end_encode]("run_end_encode")
    mb.def_function[_unary[mk.RunEndDecodeKernel.dispatch]()]("run_end_decode")
    mb.def_function[_run_index]("run_index")
    mb.def_function[_binary[mk.EqKernel[nan_safe=False].dispatch]()]("equal")
    mb.def_function[_binary[mk.NeKernel[nan_safe=False].dispatch]()](
        "not_equal"
    )
    mb.def_function[_binary[mk.LtKernel[nan_safe=False].dispatch]()]("less")
    mb.def_function[_binary[mk.LeKernel[nan_safe=False].dispatch]()](
        "less_equal"
    )
    mb.def_function[_binary[mk.GtKernel[nan_safe=False].dispatch]()]("greater")
    mb.def_function[_binary[mk.GeKernel[nan_safe=False].dispatch]()](
        "greater_equal"
    )
    mb.def_function[_sort_indices]("sort_indices")
    mb.def_function[_sort]("sort")
    mb.def_function[_take]("take")
    mb.def_function[_cast]("cast")
    mb.def_function[_concat]("concat")
    mb.def_function[_unary[mk.ListReduceKernel[mk.SumFold].dispatch]()](
        "list_sum"
    )
    mb.def_function[_unary[mk.ListReduceKernel[mk.MinFold].dispatch]()](
        "list_min"
    )
    mb.def_function[_unary[mk.ListReduceKernel[mk.MaxFold].dispatch]()](
        "list_max"
    )
    mb.def_function[_unary[mk.ListReduceKernel[mk.MeanFold].dispatch]()](
        "list_mean"
    )
    mb.def_function[_unary[mk.ListReduceKernel[mk.CountFold].dispatch]()](
        "list_count"
    )
    mb.def_function[_unary[mk.ListReduceKernel[mk.BitAndFold].dispatch]()](
        "list_bit_and"
    )
    mb.def_function[_unary[mk.ListReduceKernel[mk.BitOrFold].dispatch]()](
        "list_bit_or"
    )
    mb.def_function[_unary[mk.ListReduceKernel[mk.BitXorFold].dispatch]()](
        "list_bit_xor"
    )
    mb.def_function[_flag[mk.CumulativeKernel[mk.AddKernel].dispatch]()](
        "cumulative_sum"
    )
    mb.def_function[_flag[mk.CumulativeKernel[mk.MulKernel].dispatch]()](
        "cumulative_prod"
    )
    mb.def_function[_flag[mk.CumulativeKernel[mk.MinKernel].dispatch]()](
        "cumulative_min"
    )
    mb.def_function[_flag[mk.CumulativeKernel[mk.MaxKernel].dispatch]()](
        "cumulative_max"
    )
    mb.def_function[_offset[_shift]()]("shift")
    mb.def_function[_offset[mk.PairwiseDiffKernel.dispatch]()]("pairwise_diff")
    mb.def_function[_rolling[mk.RollingKernel[mk.AddKernel].dispatch]()](
        "rolling_sum"
    )
    mb.def_function[_rolling[mk.RollingKernel[mk.MinKernel].dispatch]()](
        "rolling_min"
    )
    mb.def_function[_rolling[mk.RollingKernel[mk.MaxKernel].dispatch]()](
        "rolling_max"
    )
    mb.def_function[_rolling[mk.RollingMeanKernel.dispatch]()]("rolling_mean")
