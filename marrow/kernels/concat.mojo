"""Array concatenation kernel.

Combines a list of type-erased arrays into a single array by concatenating
their contents. Matches the semantics of PyArrow's `pyarrow.concat_arrays()`
and Arrow C++'s `arrow::Concatenate()`.

Delegates to the appropriate builder's `extend()` method, which handles
offset-awareness, bitmap concatenation, and recursive child concatenation
for all supported array types.
"""

from ..arrays import Array
from ..builders import make_builder


def concat(arrays: List[Array]) raises -> Array:
    """Concatenate a list of arrays into a single array.

    All arrays must have the same dtype. Validity bitmaps and buffer contents
    are correctly concatenated, including support for arrays with non-zero
    offsets (slices).

    Args:
        arrays: Non-empty list of arrays with the same dtype.

    Raises:
        If arrays is empty or the dtype is unsupported.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")
    var total_length = 0
    for arr in arrays:
        total_length += arr.length
    var builder = make_builder(arrays[0].dtype, total_length)
    for arr in arrays:
        builder.extend(arr)
    return builder.finish()
