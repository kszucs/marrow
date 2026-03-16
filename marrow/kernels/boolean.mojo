"""Boolean and bitwise kernels."""

from ..arrays import PrimitiveArray
from ..bitmap import Bitmap
from ..dtypes import bool_ as bool_dt


fn count_true(array: PrimitiveArray[bool_dt]) raises -> Int:
    """Count True values in a bit-packed boolean array.

    Args:
        array: A bit-packed boolean array.

    Returns:
        Number of True (and non-null) elements.
    """
    var n = len(array)
    var data_bm = Bitmap(array.buffer, array.offset, n)
    if array.nulls > 0:
        var combined = data_bm & array.bitmap.value()
        return combined.count_set_bits()
    return data_bm.count_set_bits()
