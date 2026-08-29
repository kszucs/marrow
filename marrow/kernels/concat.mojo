"""Array concatenation kernel.

Combines a list of arrays into a single array by concatenating their contents.
Typed first, erased on top: a caller that knows its element type keeps it. Matches the semantics of PyArrow's `pyarrow.concat_arrays()`
and Arrow C++'s `arrow::Concatenate()`.

Delegates to the appropriate builder's `extend()` method, which handles
offset-awareness, bitmap concatenation, and recursive child concatenation
for all supported array types.
"""

from ..arrays import DynArray, PrimitiveArray
from ..builders import DynBuilder, PrimitiveBuilder
from ..dtypes import PrimitiveType
from ..execution import ExecContext


def concat[
    T: PrimitiveType
](
    arrays: List[PrimitiveArray[T]],
    ctx: ExecContext = ExecContext.serial(),
) raises -> PrimitiveArray[T]:
    """Concatenate fixed-width arrays of a known type.

    The typed half of this kernel, and the one the module was missing:
    everything here went through `DynBuilder`, so a caller that already knew
    its element type had to erase, concatenate, and narrow back. `Groups.ids`
    is the case that made it visible — always `Int32Array`, round-tripped
    through `DynArray` on every morsel by the aggregate operators.

    Args:
        arrays: Non-empty list of arrays with the same dtype.
        ctx: Execution context, accepted for signature parity with the erased
            overload; concatenation is a sequential copy.

    Raises:
        If arrays is empty.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")
    var total_length = 0
    for ref arr in arrays:
        total_length += len(arr)
    var builder = PrimitiveBuilder[T](arrays[0].dtype.copy(), total_length)
    for ref arr in arrays:
        builder.extend(arr)
    return builder.finish()


def concat(
    arrays: List[DynArray],
    ctx: ExecContext = ExecContext.serial(),
) raises -> DynArray:
    """Concatenate a list of arrays into a single array.

    All arrays must have the same dtype. Validity bitmaps and buffer contents
    are correctly concatenated, including support for arrays with non-zero
    offsets (slices).

    Args:
        arrays: Non-empty list of arrays with the same dtype.
        ctx: Execution context, forwarded to the per-type kernels.

    Raises:
        If arrays is empty or the dtype is unsupported.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")
    var total_length = 0
    for arr in arrays:
        total_length += arr.length()
    var builder = DynBuilder(arrays[0].dtype(), total_length)
    for arr in arrays:
        builder.extend(arr)
    return builder.finish()
