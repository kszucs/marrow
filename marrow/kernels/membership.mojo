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
    def dispatch(
        values: DynArray,
        value_set: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> BoolArray:
        """Encode ``value_set`` once, then look every value up in it; both
        operands must share one dtype.

        Returns an all-valid ``BoolArray`` of ``len(values)`` — ``true`` where
        the value has a code."""
        Self.expect_same_dtype(values.dtype(), value_set.dtype())
        var types = List[DynType]()
        types.append(value_set.dtype())
        var encoder = DictionaryEncoder(types^, ctx.copy())
        var set_column = List[DynArray]()
        set_column.append(value_set.copy())
        _ = encoder.encode(set_column)
        var value_column = List[DynArray]()
        value_column.append(values.copy())
        var codes = encoder.lookup(value_column, ctx)
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
    def apply[
        A: Array
    ](
        values: A, value_set: A, ctx: ExecContext = ExecContext.serial()
    ) raises -> BoolArray:
        """`dispatch` over typed operands of one array type."""
        return Self.dispatch(values.copy(), value_set.copy(), ctx)
