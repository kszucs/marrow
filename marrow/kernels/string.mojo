"""String compute kernels."""

from ..arrays import StringArray, PrimitiveArray
from ..buffers import BufferBuilder
from ..dtypes import uint32


# TODO: implement using SIMD
fn string_lengths(array: StringArray) -> PrimitiveArray[uint32]:
    """Compute per-element byte lengths of a StringArray.

    Handles arrays with non-zero offsets (sliced arrays).

    Args:
        array: The input string array.

    Returns:
        A PrimitiveArray[uint32] of byte lengths with all-valid bitmap.
    """
    var n = len(array)
    var off = array.offset
    var buf = BufferBuilder.alloc[DType.uint32](n)
    for i in range(n):
        var start = array.offsets.unsafe_get[DType.uint32](off + i)
        var end = array.offsets.unsafe_get[DType.uint32](off + i + 1)
        buf.unsafe_set[DType.uint32](i, end - start)
    return PrimitiveArray[uint32](
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.finish(),
    )
