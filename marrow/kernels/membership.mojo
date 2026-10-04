# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Set-membership kernel — ``is_in``.

``is_in(values, value_set) -> BoolArray`` marks each element of ``values`` with
whether it appears in ``value_set`` (SQL ``x IN (...)``, PyArrow
``pyarrow.compute.is_in``).

The set is a ``DictionaryEncoder`` over ``value_set``, and every value is
looked up in it: membership is exact, as a group-by's keys are — two values
whose hashes collide are still told apart.

Null handling matches PyArrow ``is_in``'s default (``null_matching_behavior=
"match"``): the output is always valid, and a null in ``values`` is ``true``
iff ``value_set`` itself contains a null — NULL is a key like any other to the
encoder.
"""

from ..arrays import DynArray, Array, BoolArray
from ..dtypes import DynType, int32
from ..views import apply
from ..buffers import Bitmap
from .core import Kernel
from ..execution import ExecContext
from .dictionary import DictionaryEncoder, has_code


struct IsInKernel(Kernel):
    """Membership predicate — is each element of ``values`` in ``value_set``?

    No typed leaves of its own: the encoder resolves the type, so every type it
    encodes is supported here with no overload per type.
    """

    comptime name = "is_in"

    @staticmethod
    def apply(
        values: DynArray, value_set: DynArray, ctx: ExecContext
    ) raises -> BoolArray:
        """Encode ``value_set`` once, then look every value up in it.

        Returns an all-valid ``BoolArray`` of ``len(values)`` — ``true`` where
        the value has a code."""
        var types = List[DynType]()
        types.append(value_set.dtype())
        var encoder = DictionaryEncoder(types^, ctx.copy())
        var set_column = List[DynArray]()
        set_column.append(value_set.copy())
        _ = encoder.encode(set_column)
        var value_column = List[DynArray]()
        value_column.append(values.copy())
        var codes = encoder.lookup(value_column)
        var n = len(values)
        var found = Bitmap.alloc_uninit(n)
        apply[int32.native, has_code](codes.values(), found.view())
        return BoolArray(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=found^.to_immutable(),
        )

    @staticmethod
    def dispatch(
        values: DynArray,
        value_set: DynArray,
        ctx: ExecContext = ExecContext.serial(),
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
    ctx: ExecContext = ExecContext.serial(),
) raises -> BoolArray:
    """Membership of each value in ``value_set``.

    ``values`` and ``value_set`` must share the same data type: numeric, bool,
    string, binary, temporal, decimal, dictionary and the nested types.
    """
    return IsInKernel.dispatch(values, value_set, ctx)


def is_in[
    A: Array
](
    values: A,
    value_set: A,
    ctx: ExecContext = ExecContext.serial(),
) raises -> BoolArray:
    """Membership of each value in ``value_set``, for two arrays of one type.

    Still validated rather than trusted: a shared Mojo type is not a shared
    dtype for the types that carry theirs at runtime — two `ListArray`s can
    disagree about their element type — so this goes through `dispatch` like
    any other caller.
    """
    return IsInKernel.dispatch(values.copy(), value_set.copy(), ctx)
