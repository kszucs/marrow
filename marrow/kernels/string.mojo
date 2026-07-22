"""String compute kernels."""

from ..arrays import AnyArray, StringArray, PrimitiveArray, UInt32Array
from ..buffers import Buffer
from ..dtypes import uint32
from .helpers import Kernel


# ---------------------------------------------------------------------------
# String kernel markers — NOT IMPLEMENTED yet (compute `core`/`apply` are TODO).
# They exist so the typed expression layer (`marrow.expr.ibis`) can name string
# operations; execution wires up later (see `string_lengths` below for `length`).
# ---------------------------------------------------------------------------


struct StartsWithKernel(Kernel):
    comptime name = "startswith"


struct EndsWithKernel(Kernel):
    comptime name = "endswith"


struct ContainsKernel(Kernel):
    comptime name = "contains"


struct LengthKernel(Kernel):
    comptime name = "length"


struct UpperKernel(Kernel):
    comptime name = "upper"


struct LowerKernel(Kernel):
    comptime name = "lower"


struct ReverseKernel(Kernel):
    comptime name = "reverse"


struct StripKernel(Kernel):
    comptime name = "strip"


struct LStripKernel(Kernel):
    comptime name = "lstrip"


struct RStripKernel(Kernel):
    comptime name = "rstrip"


struct CapitalizeKernel(Kernel):
    comptime name = "capitalize"


# TODO: implement using SIMD
def string_lengths(array: StringArray) -> UInt32Array:
    """Compute per-element byte lengths of a StringArray.

    Handles arrays with non-zero offsets (sliced arrays).

    Args:
        array: The input string array.

    Returns:
        A UInt32Array of byte lengths with all-valid bitmap.
    """
    var n = len(array)
    var off = array.offset
    var buf = Buffer.alloc_zeroed[DType.uint32](n)
    for i in range(n):
        var start = array.offsets.unsafe_get[DType.uint32](off + i)
        var end = array.offsets.unsafe_get[DType.uint32](off + i + 1)
        buf.unsafe_set[DType.uint32](i, end - start)
    return UInt32Array(
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.to_immutable(),
    )


def string_lengths(array: AnyArray) -> UInt32Array:
    """Runtime-typed string_lengths: dispatches to the typed StringArray overload.
    """
    return string_lengths(array.as_string())
